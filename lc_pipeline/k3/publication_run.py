"""Operational entry points for the long-running K3 publication stages."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence

import numpy as np
import torch

from ..v2.preprocessing import KnownPeriod
from .calibration import conformal_radius
from .config import K3ScoreModelConfig, K3TrainingConfig
from .damit import load_damit_object
from .evaluation import (
    ensemble_score_grids,
    modes_from_score_grid,
    oracle_at_k_error_deg,
    summarize_errors,
)
from .inference import refine_axes, refine_ensemble_axes, score_axial_grid
from .model import CandidateConditionedScorer
from .protocol import K3_PROTOCOL_SHA256
from .synthetic import load_synthetic_shard, verify_synthetic_shard
from .tokenizer import (
    K3_EPOCH_FEATURE_NAMES,
    K3_PHASE_FEATURE_NAMES,
    tokenize_epochs,
)
from .training import (
    K3FitResult,
    K3TrainingExample,
    collate_examples,
    deterministic_derangement_indices,
    examples_from_synthetic_records,
    fit_k3,
    interleave_synthetic_and_real,
    shuffle_training_labels,
)


class K3PublicationRunError(ValueError):
    """Raised when a definitive run directory or resume request is invalid."""


K3_INPUT_SWAP_SEED = 20260901
K3_LABEL_SHUFFLE_SEED = 20260901
K3_OOF_SEEDS = (17, 42, 137, 777, 2027)
K3_LABEL_SHUFFLE_MODEL_SEED = 17
# CUDA kernels may use different floating-point accumulation orders when the
# same object is scored alone rather than in the four-object publication
# batches.  Gate the candidate-dependent (mean-centered) landscape at 0.1% of
# max(cached standard deviation, one score unit).  Coarse-mode differences are
# retained explicitly because near-tied local maxima can legitimately exchange
# order even when the complete landscapes agree far more closely than this.
K3_DEPLOYED_GRID_MAX_NORMALIZED_RMS = 1e-3
K3DiagnosticControl = Literal[
    "zero-numeric",
    "brightness",
    "geometry",
    "period-scalar",
    "sampling-summary",
]
K3_DIAGNOSTIC_CONTROLS: tuple[K3DiagnosticControl, ...] = (
    "zero-numeric",
    "brightness",
    "geometry",
    "period-scalar",
    "sampling-summary",
)


def ablate_k3_inputs(
    inputs: Mapping[str, torch.Tensor], control: K3DiagnosticControl
) -> dict[str, torch.Tensor]:
    """Apply one inference-only diagnostic without changing padding masks.

    ``period-scalar`` removes only the explicit normalized-period feature;
    phase folding remains present. ``sampling-summary`` removes numeric count,
    span, duration, and coverage summaries, while phase occupancy and padding
    masks remain present. These deliberately narrow names prevent either arm
    from being misreported as complete removal of time or period information.
    """
    if control not in K3_DIAGNOSTIC_CONTROLS:
        raise K3PublicationRunError(f"unsupported K3 diagnostic control {control!r}")
    required = {
        "phase_features",
        "phase_mask",
        "geometry_features",
        "epoch_features",
        "epoch_mask",
    }
    if set(inputs) != required or not all(
        isinstance(inputs[key], torch.Tensor) for key in required
    ):
        raise K3PublicationRunError("K3 diagnostic inputs do not match the model schema")
    output = {key: value.clone() for key, value in inputs.items()}
    if control == "zero-numeric":
        for key in ("phase_features", "geometry_features", "epoch_features"):
            output[key].zero_()
        return output

    phase_indices: tuple[int, ...] = ()
    epoch_indices: tuple[int, ...] = ()
    if control == "brightness":
        phase_indices = tuple(
            K3_PHASE_FEATURE_NAMES.index(name)
            for name in ("centered_log_flux_mean", "centered_log_flux_scatter")
        )
        epoch_indices = tuple(
            index
            for index, name in enumerate(K3_EPOCH_FEATURE_NAMES)
            if name.startswith("fourier_") or name == "log_flux_peak_to_peak"
        )
    elif control == "geometry":
        output["geometry_features"].zero_()
    elif control == "period-scalar":
        epoch_indices = (K3_EPOCH_FEATURE_NAMES.index("normalized_log_period_hours"),)
    elif control == "sampling-summary":
        phase_indices = tuple(
            K3_PHASE_FEATURE_NAMES.index(name)
            for name in ("log1p_observation_count", "within_bin_phase_span_fraction")
        )
        epoch_indices = tuple(
            K3_EPOCH_FEATURE_NAMES.index(name)
            for name in (
                "occupied_phase_fraction",
                "log1p_observation_count",
                "log1p_duration_days",
            )
        )
    if phase_indices:
        output["phase_features"][..., list(phase_indices)] = 0.0
    if epoch_indices:
        output["epoch_features"][..., list(epoch_indices)] = 0.0
    return output


def _write_atomic_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_real_examples(
    catalog_path: str | Path,
    dump_root: str | Path,
    object_ids: Sequence[str],
) -> tuple[K3TrainingExample, ...]:
    """Build fixed-period real examples with every catalogue axis positive."""
    try:
        rows = {
            row["object_id"]: row
            for row in (
                json.loads(line)
                for line in Path(catalog_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise K3PublicationRunError(f"cannot read real-data catalogue: {exc}") from exc
    requested = tuple(object_ids)
    if len(requested) != len(set(requested)) or any(value not in rows for value in requested):
        raise K3PublicationRunError("real object IDs must be unique and present in the frozen catalogue")
    examples: list[K3TrainingExample] = []
    for object_id in requested:
        row = rows[object_id]
        solutions = row.get("solutions", [])
        if not row.get("eligible") or not solutions:
            raise K3PublicationRunError(f"{object_id} is not an eligible labelled object")
        periods = np.asarray([solution["period_hours"] for solution in solutions], dtype=np.float64)
        if np.any(~np.isfinite(periods)) or np.any(periods <= 0):
            raise K3PublicationRunError(f"{object_id} has invalid catalogue periods")
        source = load_damit_object(dump_root, object_id)
        # The catalogue order defines the primary DAMIT solution. Alternative
        # axes remain valid oracle targets, but one deployable fixed period is
        # required at inference; this affects only asteroid_3572 in the cohort.
        period = float(periods[0])
        tokenized = tokenize_epochs(
            source.epochs,
            known_period=KnownPeriod(period, "frozen-DAMIT-primary-solution-fixed-period"),
        )
        examples.append(
            K3TrainingExample(
                object_id=object_id,
                tokenized=tokenized,
                target_axes=np.asarray([solution["vector"] for solution in solutions], dtype=np.float32),
            )
        )
    return tuple(examples)


def load_synthetic_examples(directory: str | Path) -> tuple[K3TrainingExample, ...]:
    """Verify, load, and tokenize a complete synthetic split in shard order."""
    root = Path(directory)
    shards = sorted(root.glob("*.npz"))
    if not shards:
        raise K3PublicationRunError(f"no synthetic shards found in {root}")
    examples: list[K3TrainingExample] = []
    identifiers: set[str] = set()
    for shard in shards:
        summary = verify_synthetic_shard(shard)
        records = load_synthetic_shard(shard, verify_manifest=False)
        if len(records) != summary["objects"]:
            raise K3PublicationRunError(f"verified object count changed while loading {shard}")
        values = examples_from_synthetic_records(records)
        overlap = identifiers.intersection(value.object_id for value in values)
        if overlap:
            raise K3PublicationRunError(f"duplicate synthetic object IDs: {sorted(overlap)[:3]}")
        identifiers.update(value.object_id for value in values)
        examples.extend(values)
    return tuple(examples)


def load_synthetic_subset(
    directory: str | Path, *, count: int, seed: int
) -> tuple[K3TrainingExample, ...]:
    """Load the hash-ranked synthetic subset used for real-data regularization."""
    shards = sorted(Path(directory).glob("*.npz"))
    identifiers: list[str] = []
    for shard in shards:
        summary = verify_synthetic_shard(shard)
        identifiers.extend(summary["object_ids"])
    if count <= 0 or count > len(identifiers) or len(identifiers) != len(set(identifiers)):
        raise K3PublicationRunError("synthetic subset count/identifiers are invalid")
    ranked = sorted(
        identifiers,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode("ascii")).digest(),
    )
    selected = set(ranked[:count])
    examples: list[K3TrainingExample] = []
    for shard in shards:
        records = load_synthetic_shard(
            shard, verify_manifest=False, selected_object_ids=selected
        )
        examples.extend(examples_from_synthetic_records(records))
    if {value.object_id for value in examples} != selected:
        raise K3PublicationRunError("synthetic subset reconstruction is incomplete")
    return tuple(examples)


def run_synthetic_pretraining(
    train_directory: str | Path,
    validation_directory: str | Path,
    output_directory: str | Path,
    *,
    seed: int,
    device: str = "cuda",
    resume: bool = False,
    model_config: K3ScoreModelConfig | None = None,
    training_config: K3TrainingConfig | None = None,
    run_provenance: Mapping[str, str] | None = None,
) -> K3FitResult:
    """Run one locked synthetic seed with durable epoch logs/checkpoints."""
    selected_model = model_config or K3ScoreModelConfig()
    selected_training = training_config or K3TrainingConfig(seed=seed)
    if selected_training.seed != seed:
        raise K3PublicationRunError("training config seed differs from requested seed")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise K3PublicationRunError("CUDA was requested but is unavailable")
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / f"synthetic-seed-{seed}.pt"
    history = destination / f"synthetic-seed-{seed}.jsonl"
    if not resume and (checkpoint.exists() or history.exists()):
        raise K3PublicationRunError("refusing to overwrite an existing seed run; use resume")
    if resume and not checkpoint.exists():
        raise K3PublicationRunError("resume requested but checkpoint does not exist")
    train_examples = load_synthetic_examples(train_directory)
    validation_examples = load_synthetic_examples(validation_directory)
    model = CandidateConditionedScorer(selected_model)

    def record(row: dict[str, float]) -> None:
        payload = {"schema": "delphi.k3-epoch.v1", "seed": seed, **row}
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
        print(json.dumps(payload, sort_keys=True), flush=True)

    return fit_k3(
        model,
        train_examples,
        validation_examples,
        config=selected_training,
        stage="synthetic",
        device=device,
        checkpoint_path=checkpoint,
        resume_checkpoint=checkpoint if resume else None,
        progress_callback=record,
        run_provenance=run_provenance,
    )


def evaluate_synthetic_checkpoint(
    checkpoint_path: str | Path,
    test_directory: str | Path,
    output_path: str | Path,
    *,
    device: str = "cuda",
    batch_size: int = 4,
) -> dict[str, float | str]:
    """Retain complete score maps and refined oracle-at-3 errors."""
    examples = load_synthetic_examples(test_directory)
    return _evaluate_examples(
        checkpoint_path, examples, output_path, device=device, batch_size=batch_size
    )


def evaluate_synthetic_input_swap_checkpoint(
    checkpoint_path: str | Path,
    test_directory: str | Path,
    raw_artifact_path: str | Path,
    output_path: str | Path,
    *,
    seed: int,
    device: str = "cuda",
    batch_size: int = 4,
) -> dict[str, float | str]:
    """Evaluate the synthetic test set with complete inputs deranged across labels."""
    checkpoint_source = Path(checkpoint_path)
    raw_source = Path(raw_artifact_path)
    destination = Path(output_path)
    expected_checkpoint = f"synthetic-seed-{seed}.pt"
    expected_raw = f"synthetic-test-seed-{seed}.npz"
    expected_output = f"synthetic-test-seed-{seed}-input-swap.npz"
    if (
        checkpoint_source.name != expected_checkpoint
        or raw_source.name != expected_raw
        or destination.name != expected_output
    ):
        raise K3PublicationRunError(
            "synthetic input-swap paths must use "
            f"{expected_checkpoint}, {expected_raw}, and {expected_output}"
        )
    try:
        checkpoint = torch.load(
            checkpoint_source, map_location="cpu", weights_only=False
        )
        with np.load(raw_source, allow_pickle=False) as raw:
            raw_ids = np.asarray(raw["object_ids"]).astype(str)
            raw_errors = np.asarray(raw["oracle_errors_deg"], dtype=np.float64)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise K3PublicationRunError(
            f"cannot load synthetic input-swap inputs: {exc}"
        ) from exc
    if checkpoint.get("stage") != "synthetic" or int(
        checkpoint.get("config", {}).get("seed", -1)
    ) != seed:
        raise K3PublicationRunError(
            "synthetic input-swap checkpoint stage/training seed is invalid"
        )
    examples = load_synthetic_examples(test_directory)
    object_ids = np.asarray([example.object_id for example in examples])
    if (
        len(examples) < 2
        or len(set(object_ids.tolist())) != len(examples)
        or not np.array_equal(raw_ids, object_ids)
        or raw_errors.shape != (len(examples),)
        or not np.all(np.isfinite(raw_errors))
    ):
        raise K3PublicationRunError(
            "raw synthetic artifact is not exactly aligned to the finite test cohort"
        )
    donors = deterministic_derangement_indices(
        len(examples), seed=K3_INPUT_SWAP_SEED
    )
    swapped = tuple(
        K3TrainingExample(
            object_id=target.object_id,
            tokenized=examples[int(donor)].tokenized,
            target_axes=target.target_axes,
        )
        for target, donor in zip(examples, donors, strict=True)
    )
    raw_sha256 = hashlib.sha256(raw_source.read_bytes()).hexdigest()
    summary = _evaluate_examples(
        checkpoint_source,
        swapped,
        destination,
        device=device,
        batch_size=batch_size,
        artifact_arrays={
            "input_donor_ids": object_ids[donors],
            "derangement_seeds": np.full(
                len(examples), K3_INPUT_SWAP_SEED, dtype=np.int64
            ),
            "raw_oracle_errors_deg": raw_errors,
            "raw_artifact_sha256": np.full(len(examples), raw_sha256),
        },
    )
    swapped_mean = float(summary["mean_error_deg"])
    summary.update(
        {
            "seed": float(seed),
            "derangement_seed": float(K3_INPUT_SWAP_SEED),
            "control": "input-swap",
            "raw_artifact_sha256": raw_sha256,
            "input_swap_gap_deg": swapped_mean - float(np.mean(raw_errors)),
        }
    )
    return summary


def run_synthetic_label_shuffle_control(
    train_directory: str | Path,
    validation_directory: str | Path,
    output_directory: str | Path,
    *,
    seed: int,
    device: str = "cuda",
    resume: bool = False,
    model_config: K3ScoreModelConfig | None = None,
    training_config: K3TrainingConfig | None = None,
    run_provenance: Mapping[str, str] | None = None,
) -> K3FitResult:
    """Train from scratch with only the synthetic training labels deranged."""
    selected_model = model_config or K3ScoreModelConfig()
    selected_training = training_config or K3TrainingConfig(seed=seed)
    if selected_training.seed != seed:
        raise K3PublicationRunError("training config seed differs from requested seed")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise K3PublicationRunError("CUDA was requested but is unavailable")
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / f"synthetic-label-shuffle-seed-{seed}.pt"
    history = destination / f"synthetic-label-shuffle-seed-{seed}.jsonl"
    mapping_path = destination / f"synthetic-label-shuffle-seed-{seed}.labels.json"
    if not resume and any(path.exists() for path in (checkpoint, history, mapping_path)):
        raise K3PublicationRunError("refusing to overwrite an existing label-shuffle run; use resume")
    if resume and (not checkpoint.exists() or not mapping_path.exists()):
        raise K3PublicationRunError("label-shuffle resume requires checkpoint and mapping artifacts")

    train_examples = load_synthetic_examples(train_directory)
    validation_examples = load_synthetic_examples(validation_directory)
    donors = deterministic_derangement_indices(
        len(train_examples), seed=K3_LABEL_SHUFFLE_SEED
    )
    shuffled = shuffle_training_labels(train_examples, seed=K3_LABEL_SHUFFLE_SEED)
    mapping = {
        "schema": "delphi.k3-label-shuffle-map.v1",
        "control_seed": K3_LABEL_SHUFFLE_SEED,
        "model_seed": seed,
        "training_object_count": len(train_examples),
        "assignments": [
            {
                "input_object_id": example.object_id,
                "label_donor_object_id": train_examples[int(donor)].object_id,
            }
            for example, donor in zip(train_examples, donors, strict=True)
        ],
    }
    if resume:
        try:
            existing = json.loads(mapping_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise K3PublicationRunError(f"cannot validate label-shuffle mapping: {exc}") from exc
        if existing != mapping:
            raise K3PublicationRunError("label-shuffle mapping differs from the resume request")
        mapping_sha256 = hashlib.sha256(mapping_path.read_bytes()).hexdigest()
    else:
        mapping_sha256 = _write_atomic_json(mapping_path, mapping)
    model = CandidateConditionedScorer(selected_model)
    bound_provenance = None if run_provenance is None else {
        **run_provenance,
        "label_mapping_sha256": mapping_sha256,
    }

    def record(row: dict[str, float]) -> None:
        payload = {
            "schema": "delphi.k3-label-shuffle-epoch.v1",
            "seed": seed,
            "label_mapping_sha256": mapping_sha256,
            **row,
        }
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
        print(json.dumps(payload, sort_keys=True), flush=True)

    return fit_k3(
        model,
        shuffled,
        validation_examples,
        config=selected_training,
        stage="synthetic-label-shuffle",
        device=device,
        checkpoint_path=checkpoint,
        resume_checkpoint=checkpoint if resume else None,
        progress_callback=record,
        run_provenance=bound_provenance,
    )


def evaluate_synthetic_label_shuffle_checkpoint(
    checkpoint_path: str | Path,
    test_directory: str | Path,
    output_path: str | Path,
    *,
    seed: int,
    device: str = "cuda",
    batch_size: int = 4,
) -> dict[str, float | str]:
    """Evaluate a label-shuffle checkpoint on the untouched synthetic test set."""
    source = Path(checkpoint_path)
    expected_checkpoint = f"synthetic-label-shuffle-seed-{seed}.pt"
    expected_output = f"synthetic-label-shuffle-test-seed-{seed}.npz"
    if source.name != expected_checkpoint or Path(output_path).name != expected_output:
        raise K3PublicationRunError(
            f"label-shuffle paths must use {expected_checkpoint} and {expected_output}"
        )
    try:
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise K3PublicationRunError(f"cannot load label-shuffle checkpoint: {exc}") from exc
    if checkpoint.get("stage") != "synthetic-label-shuffle" or int(
        checkpoint.get("config", {}).get("seed", -1)
    ) != seed:
        raise K3PublicationRunError("label-shuffle checkpoint stage/training seed is invalid")
    examples = load_synthetic_examples(test_directory)
    summary = _evaluate_examples(
        source, examples, output_path, device=device, batch_size=batch_size
    )
    summary.update({"seed": float(seed), "control": "label-shuffle-training"})
    return summary


def _evaluate_examples(
    checkpoint_path: str | Path,
    examples: Sequence[K3TrainingExample],
    output_path: str | Path,
    *,
    device: str,
    batch_size: int,
    artifact_arrays: Mapping[str, np.ndarray] | None = None,
    input_transform: Callable[
        [Mapping[str, torch.Tensor]], dict[str, torch.Tensor]
    ]
    | None = None,
) -> dict[str, float | str]:
    """Evaluate a frozen ordered cohort and atomically retain its score maps."""
    checkpoint_source, destination = Path(checkpoint_path), Path(output_path)
    if destination.exists():
        raise K3PublicationRunError(f"refusing to overwrite evaluation artifact {destination}")
    if batch_size <= 0 or (device.startswith("cuda") and not torch.cuda.is_available()):
        raise K3PublicationRunError("evaluation batch/device configuration is invalid")
    extras = {} if artifact_arrays is None else {
        str(key): np.asarray(value) for key, value in artifact_arrays.items()
    }
    reserved = {"object_ids", "score_grids", "oracle_errors_deg"}
    if reserved.intersection(extras):
        raise K3PublicationRunError("supplementary evaluation arrays use a reserved name")
    if any(
        value.ndim == 0 or value.shape[0] != len(examples) or value.dtype.hasobject
        for value in extras.values()
    ):
        raise K3PublicationRunError(
            "supplementary evaluation arrays must be non-object arrays aligned by example"
        )
    checkpoint = torch.load(checkpoint_source, map_location="cpu", weights_only=False)
    model = CandidateConditionedScorer(K3ScoreModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    grids = np.empty((len(examples), 6144), dtype=np.float32)
    errors = np.empty(len(examples), dtype=np.float64)
    identifiers: list[str] = []
    started = time.time()
    for start in range(0, len(examples), batch_size):
        values = examples[start : start + batch_size]
        raw = collate_examples(values)
        inputs: dict[str, torch.Tensor] = {
            key: value
            for key, value in raw.items()
            if key in {"phase_features", "phase_mask", "geometry_features", "epoch_features", "epoch_mask"}
        }
        if input_transform is not None:
            inputs = input_transform(inputs)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        scores = np.asarray(score_axial_grid(model, inputs, chunk_size=1024))
        if scores.ndim == 1:
            scores = scores[None, :]
        grids[start : start + len(values)] = scores
        for index, example in enumerate(values):
            modes = modes_from_score_grid(scores[index])
            initial = np.asarray([mode.axis_xyz for mode in modes])
            one = {key: value[index : index + 1] for key, value in inputs.items()}
            axes, _ = refine_axes(model, one, initial)
            errors[start + index] = oracle_at_k_error_deg(axes, example.target_axes, k=3)
            identifiers.append(example.object_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                object_ids=np.asarray(identifiers),
                score_grids=grids,
                oracle_errors_deg=errors,
                **extras,
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    summary: dict[str, float | str] = summarize_errors(errors)
    summary.update(
        {
            "checkpoint_sha256": hashlib.sha256(checkpoint_source.read_bytes()).hexdigest(),
            "artifact_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "elapsed_seconds": float(time.time() - started),
        }
    )
    return summary


def evaluate_real_oof_checkpoint(
    checkpoint_path: str | Path,
    catalog_path: str | Path,
    dump_root: str | Path,
    split_path: str | Path,
    output_path: str | Path,
    *,
    fold: int,
    seed: int,
    device: str = "cuda",
    batch_size: int = 4,
) -> dict[str, float | str]:
    """Evaluate exactly one frozen real OOF test fold and retain full score maps."""
    expected_name = f"real-fold-{fold}-seed-{seed}.pt"
    checkpoint_source = Path(checkpoint_path)
    if checkpoint_source.name != expected_name:
        raise K3PublicationRunError(
            f"checkpoint name must be {expected_name} to prevent fold/seed mix-ups"
        )
    try:
        split = json.loads(Path(split_path).read_text(encoding="utf-8"))
        fold_row = next(row for row in split["folds"] if int(row["fold"]) == fold)
    except (OSError, json.JSONDecodeError, KeyError, StopIteration) as exc:
        raise K3PublicationRunError(f"cannot load requested OOF fold: {exc}") from exc
    test_ids = tuple(fold_row["test_ids"])
    if not test_ids or len(test_ids) != len(set(test_ids)):
        raise K3PublicationRunError("OOF test IDs must be nonempty and unique")
    checkpoint = torch.load(checkpoint_source, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "real-oof":
        raise K3PublicationRunError("real OOF evaluation requires a real-oof checkpoint")
    configuration = checkpoint.get("config", {})
    if int(configuration.get("seed", -1)) != seed:
        raise K3PublicationRunError("checkpoint training seed differs from requested seed")
    examples = load_real_examples(catalog_path, dump_root, test_ids)
    summary = _evaluate_examples(
        checkpoint_source, examples, output_path, device=device, batch_size=batch_size
    )
    summary.update({"fold": float(fold), "seed": float(seed)})
    return summary


def evaluate_real_calibration_checkpoint(
    checkpoint_path: str | Path,
    catalog_path: str | Path,
    dump_root: str | Path,
    split_path: str | Path,
    output_path: str | Path,
    *,
    fold: int,
    seed: int,
    device: str = "cuda",
    batch_size: int = 4,
) -> dict[str, float | str]:
    """Evaluate one fold's untouched inner-calibration role."""
    checkpoint_source = Path(checkpoint_path)
    expected_checkpoint = f"real-fold-{fold}-seed-{seed}.pt"
    expected_output = f"real-fold-{fold}-seed-{seed}-calibration.npz"
    if checkpoint_source.name != expected_checkpoint or Path(output_path).name != expected_output:
        raise K3PublicationRunError(
            f"calibration paths must use {expected_checkpoint} and {expected_output}"
        )
    try:
        split = json.loads(Path(split_path).read_text(encoding="utf-8"))
        fold_row = next(row for row in split["folds"] if int(row["fold"]) == fold)
        checkpoint = torch.load(checkpoint_source, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError, KeyError, StopIteration) as exc:
        raise K3PublicationRunError(f"cannot load requested calibration fold: {exc}") from exc
    calibration_ids = tuple(fold_row["calibration_ids"])
    if len(calibration_ids) != 14 or len(calibration_ids) != len(set(calibration_ids)):
        raise K3PublicationRunError(
            "definitive inner-calibration role must contain exactly 14 unique objects"
        )
    if checkpoint.get("stage") != "real-oof" or int(
        checkpoint.get("config", {}).get("seed", -1)
    ) != seed:
        raise K3PublicationRunError("calibration checkpoint stage/training seed is invalid")
    examples = load_real_examples(catalog_path, dump_root, calibration_ids)
    summary = _evaluate_examples(
        checkpoint_source, examples, output_path, device=device, batch_size=batch_size
    )
    summary.update({"fold": float(fold), "seed": float(seed), "role": "calibration"})
    return summary


def evaluate_real_oof_diagnostic_checkpoint(
    checkpoint_path: str | Path,
    catalog_path: str | Path,
    dump_root: str | Path,
    split_path: str | Path,
    output_path: str | Path,
    *,
    fold: int,
    seed: int,
    control: K3DiagnosticControl,
    device: str = "cuda",
    batch_size: int = 4,
) -> dict[str, float | str]:
    """Evaluate one real OOF fold under a precisely defined input ablation."""
    if control not in K3_DIAGNOSTIC_CONTROLS:
        raise K3PublicationRunError(f"unsupported K3 diagnostic control {control!r}")
    checkpoint_source = Path(checkpoint_path)
    expected_checkpoint = f"real-fold-{fold}-seed-{seed}.pt"
    expected_output = f"real-fold-{fold}-seed-{seed}-{control}.npz"
    if checkpoint_source.name != expected_checkpoint or Path(output_path).name != expected_output:
        raise K3PublicationRunError(
            f"diagnostic paths must use {expected_checkpoint} and {expected_output}"
        )
    try:
        split = json.loads(Path(split_path).read_text(encoding="utf-8"))
        fold_row = next(row for row in split["folds"] if int(row["fold"]) == fold)
        checkpoint = torch.load(checkpoint_source, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError, KeyError, StopIteration) as exc:
        raise K3PublicationRunError(f"cannot load requested diagnostic fold: {exc}") from exc
    test_ids = tuple(fold_row["test_ids"])
    if not test_ids or len(test_ids) != len(set(test_ids)):
        raise K3PublicationRunError("diagnostic OOF test IDs must be nonempty and unique")
    if checkpoint.get("stage") != "real-oof" or int(
        checkpoint.get("config", {}).get("seed", -1)
    ) != seed:
        raise K3PublicationRunError("diagnostic checkpoint stage/training seed is invalid")
    examples = load_real_examples(catalog_path, dump_root, test_ids)
    summary = _evaluate_examples(
        checkpoint_source,
        examples,
        output_path,
        device=device,
        batch_size=batch_size,
        artifact_arrays={
            "diagnostic_control": np.full(len(examples), control),
        },
        input_transform=lambda inputs: ablate_k3_inputs(inputs, control),
    )
    summary.update({"fold": float(fold), "seed": float(seed), "control": control})
    return summary


def evaluate_real_oof_input_swap_checkpoint(
    checkpoint_path: str | Path,
    catalog_path: str | Path,
    dump_root: str | Path,
    split_path: str | Path,
    output_path: str | Path,
    *,
    fold: int,
    seed: int,
    device: str = "cuda",
    batch_size: int = 4,
) -> dict[str, float | str]:
    """Evaluate one OOF checkpoint after a fixed fold-local input derangement."""
    checkpoint_source = Path(checkpoint_path)
    expected_checkpoint = f"real-fold-{fold}-seed-{seed}.pt"
    expected_output = f"real-fold-{fold}-seed-{seed}-input-swap.npz"
    if checkpoint_source.name != expected_checkpoint or Path(output_path).name != expected_output:
        raise K3PublicationRunError(
            f"input-swap paths must use {expected_checkpoint} and {expected_output}"
        )
    try:
        split = json.loads(Path(split_path).read_text(encoding="utf-8"))
        fold_row = next(row for row in split["folds"] if int(row["fold"]) == fold)
    except (OSError, json.JSONDecodeError, KeyError, StopIteration) as exc:
        raise K3PublicationRunError(f"cannot load requested OOF fold: {exc}") from exc
    test_ids = tuple(fold_row["test_ids"])
    if len(test_ids) < 2 or len(test_ids) != len(set(test_ids)):
        raise K3PublicationRunError("input-swap OOF test IDs must be unique and contain two objects")
    try:
        checkpoint = torch.load(checkpoint_source, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise K3PublicationRunError(f"cannot load input-swap checkpoint: {exc}") from exc
    if checkpoint.get("stage") != "real-oof" or int(
        checkpoint.get("config", {}).get("seed", -1)
    ) != seed:
        raise K3PublicationRunError("input-swap checkpoint stage/training seed is invalid")

    examples = load_real_examples(catalog_path, dump_root, test_ids)
    control_seed = K3_INPUT_SWAP_SEED + fold
    donors = deterministic_derangement_indices(len(examples), seed=control_seed)
    swapped = tuple(
        K3TrainingExample(
            object_id=target.object_id,
            tokenized=examples[int(donor)].tokenized,
            target_axes=target.target_axes,
        )
        for target, donor in zip(examples, donors, strict=True)
    )
    donor_ids = np.asarray([test_ids[int(index)] for index in donors])
    summary = _evaluate_examples(
        checkpoint_source,
        swapped,
        output_path,
        device=device,
        batch_size=batch_size,
        artifact_arrays={
            "input_donor_ids": donor_ids,
            "derangement_seeds": np.full(len(swapped), control_seed, dtype=np.int64),
        },
    )
    summary.update(
        {
            "fold": float(fold),
            "seed": float(seed),
            "derangement_seed": float(control_seed),
            "control": "input-swap",
        }
    )
    return summary


def aggregate_real_oof_artifacts(
    evaluation_directory: str | Path,
    split_path: str | Path,
    output_path: str | Path,
    *,
    seed: int,
) -> dict[str, float | str]:
    """Combine five disjoint fold artifacts after exact frozen-cohort validation."""
    destination = Path(output_path)
    if destination.exists():
        raise K3PublicationRunError(f"refusing to overwrite aggregate artifact {destination}")
    try:
        split = json.loads(Path(split_path).read_text(encoding="utf-8"))
        fold_rows = sorted(split["folds"], key=lambda row: int(row["fold"]))
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise K3PublicationRunError(f"cannot load OOF splits: {exc}") from exc
    if [int(row["fold"]) for row in fold_rows] != list(range(5)):
        raise K3PublicationRunError("OOF aggregation requires exactly folds 0 through 4")
    identifiers: list[str] = []
    errors: list[np.ndarray] = []
    grids: list[np.ndarray] = []
    folds: list[np.ndarray] = []
    input_hashes: list[str] = []
    root = Path(evaluation_directory)
    for row in fold_rows:
        fold = int(row["fold"])
        source = root / f"real-fold-{fold}-seed-{seed}.npz"
        try:
            with np.load(source, allow_pickle=False) as artifact:
                fold_ids = artifact["object_ids"].astype(str)
                fold_errors = np.asarray(artifact["oracle_errors_deg"], dtype=np.float64)
                fold_grids = np.asarray(artifact["score_grids"], dtype=np.float32)
        except (OSError, ValueError, KeyError) as exc:
            raise K3PublicationRunError(f"cannot load fold {fold} artifact: {exc}") from exc
        expected = tuple(row["test_ids"])
        if tuple(fold_ids) != expected:
            raise K3PublicationRunError(f"fold {fold} artifact IDs/order differ from frozen test split")
        if fold_errors.shape != (len(expected),) or fold_grids.shape != (len(expected), 6144):
            raise K3PublicationRunError(f"fold {fold} artifact arrays have invalid shapes")
        identifiers.extend(fold_ids.tolist())
        errors.append(fold_errors)
        grids.append(fold_grids)
        folds.append(np.full(len(expected), fold, dtype=np.int8))
        input_hashes.append(hashlib.sha256(source.read_bytes()).hexdigest())
    if len(identifiers) != len(set(identifiers)):
        raise K3PublicationRunError("OOF fold artifacts contain duplicate object IDs")
    all_errors = np.concatenate(errors)
    all_grids = np.concatenate(grids)
    all_folds = np.concatenate(folds)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                object_ids=np.asarray(identifiers),
                folds=all_folds,
                score_grids=all_grids,
                oracle_errors_deg=all_errors,
                input_artifact_sha256=np.asarray(input_hashes),
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    summary: dict[str, float | str] = summarize_errors(all_errors)
    summary.update(
        {
            "seed": float(seed),
            "artifact_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "splits_sha256": hashlib.sha256(Path(split_path).read_bytes()).hexdigest(),
        }
    )
    return summary


def build_v1_comparator_artifact(
    v1_oof_root: str | Path,
    split_path: str | Path,
    output_path: str | Path,
) -> dict[str, float | str]:
    """Bind the completed V1 study to the exact K3 OOF object order."""
    destination = Path(output_path)
    if destination.name != "v1-comparators.npz":
        raise K3PublicationRunError("V1 comparator artifact must be named v1-comparators.npz")
    if destination.exists():
        raise K3PublicationRunError(f"refusing to overwrite comparator artifact {destination}")
    try:
        split_source = Path(split_path)
        split = json.loads(split_source.read_text(encoding="utf-8"))
        fold_rows = sorted(split["folds"], key=lambda row: int(row["fold"]))
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise K3PublicationRunError(f"cannot load OOF splits: {exc}") from exc
    if [int(row["fold"]) for row in fold_rows] != list(range(5)) or any(
        len(row.get("test_ids", ())) != 34 for row in fold_rows
    ):
        raise K3PublicationRunError("V1 alignment requires five 34-object test folds")
    object_ids = tuple(
        object_id for row in fold_rows for object_id in row["test_ids"]
    )
    if len(object_ids) != 170 or len(object_ids) != len(set(object_ids)):
        raise K3PublicationRunError("V1 alignment requires 170 unique OOF objects")
    folds = np.repeat(np.arange(5, dtype=np.int8), 34)
    conditions = ("raw", "atlas", "exhaustive_deranged", "zero")
    seed_errors = {
        condition: np.empty((len(K3_OOF_SEEDS), 170), dtype=np.float64)
        for condition in conditions
    }
    row_hashes: list[str] = []
    summary_hashes: list[str] = []
    implementation_commits: list[str] = []
    root = Path(v1_oof_root)
    for seed_index, seed in enumerate(K3_OOF_SEEDS):
        offset = 0
        for row in fold_rows:
            fold = int(row["fold"])
            run_id = f"publication-oof-v1-faithful-f{fold}-s{seed}"
            run_root = root / "runs" / run_id
            rows_path = run_root / "evaluation-rows.jsonl"
            summary_path = run_root / "axial-experiment-summary.json"
            try:
                evaluation_rows = [
                    json.loads(line)
                    for line in rows_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise K3PublicationRunError(f"cannot load V1 run {run_id}: {exc}") from exc
            rows_hash = hashlib.sha256(rows_path.read_bytes()).hexdigest()
            if (
                run_summary.get("schema") != "delphi.axial-experiment-summary.v1"
                or run_summary.get("run_id") != run_id
                or run_summary.get("model_kind") != "v1_faithful"
                or int(run_summary.get("fold", -1)) != fold
                or int(run_summary.get("seed", -1)) != seed
                or run_summary.get("artifacts", {}).get("evaluation-rows.jsonl")
                != rows_hash
            ):
                raise K3PublicationRunError(f"V1 summary provenance is invalid for {run_id}")
            expected_ids = tuple(row["test_ids"])
            for condition in conditions:
                selected = [
                    value
                    for value in evaluation_rows
                    if value.get("condition") == condition
                ]
                if tuple(value.get("object_id") for value in selected) != expected_ids:
                    raise K3PublicationRunError(
                        f"V1 {run_id}/{condition} differs from frozen test order"
                    )
                errors = np.asarray(
                    [value.get("axis_oracle_at3_error_deg") for value in selected],
                    dtype=np.float64,
                )
                metadata_invalid = any(
                    value.get("model_kind") != "v1_faithful"
                    or int(value.get("seed", -1)) != seed
                    for value in selected
                )
                axes = [np.asarray(value.get("axes"), dtype=np.float64) for value in selected]
                axes_invalid = False
                for row_value, axes_value, error in zip(
                    selected, axes, errors, strict=True
                ):
                    valid_axes = (
                        axes_value.shape == (3, 3)
                        and np.all(np.isfinite(axes_value))
                        and np.allclose(
                            np.linalg.norm(axes_value, axis=1),
                            1.0,
                            atol=1e-5,
                            rtol=0,
                        )
                    )
                    explicit_failure = (
                        row_value.get("has_valid_axis") is False
                        and row_value.get("axis_valid") == [False, False, False]
                        and axes_value.shape == (3, 3)
                        and np.array_equal(axes_value, np.zeros((3, 3)))
                        and error == 90.0
                    )
                    axes_invalid = axes_invalid or not (valid_axes or explicit_failure)
                if metadata_invalid or axes_invalid:
                    raise K3PublicationRunError(
                        f"V1 {run_id}/{condition} contains invalid prediction metadata"
                    )
                if (
                    errors.shape != (34,)
                    or not np.all(np.isfinite(errors))
                    or np.any(errors < 0)
                    or np.any(errors > 90)
                ):
                    raise K3PublicationRunError(
                        f"V1 {run_id}/{condition} contains invalid axial errors"
                    )
                seed_errors[condition][seed_index, offset : offset + 34] = errors
            if len(evaluation_rows) != 34 * len(conditions):
                raise K3PublicationRunError(f"V1 run {run_id} has unexpected extra rows")
            commit = str(run_summary.get("git_commit", ""))
            if len(commit) != 40 or any(
                character not in "0123456789abcdef" for character in commit
            ):
                raise K3PublicationRunError(f"V1 run {run_id} has invalid Git provenance")
            implementation_commits.append(commit)
            row_hashes.append(rows_hash)
            summary_hashes.append(hashlib.sha256(summary_path.read_bytes()).hexdigest())
            offset += 34
        if offset != 170:
            raise K3PublicationRunError(f"V1 seed {seed} did not cover 170 objects")
    if len(set(implementation_commits)) != 1:
        raise K3PublicationRunError("V1 comparator runs use multiple implementation commits")

    mean_errors = {
        condition: np.mean(values, axis=0)
        for condition, values in seed_errors.items()
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                schema=np.asarray("delphi.k3-v1-comparators.v1"),
                seeds=np.asarray(K3_OOF_SEEDS, dtype=np.int64),
                object_ids=np.asarray(object_ids),
                folds=folds,
                v1_seed_errors_deg=seed_errors["raw"],
                atlas_seed_errors_deg=seed_errors["atlas"],
                deranged_seed_errors_deg=seed_errors["exhaustive_deranged"],
                zero_seed_errors_deg=seed_errors["zero"],
                v1_errors_deg=mean_errors["raw"],
                atlas_errors_deg=mean_errors["atlas"],
                deranged_errors_deg=mean_errors["exhaustive_deranged"],
                zero_errors_deg=mean_errors["zero"],
                source_evaluation_sha256=np.asarray(row_hashes),
                source_summary_sha256=np.asarray(summary_hashes),
                source_implementation_commit=np.asarray(implementation_commits[0]),
                splits_sha256=np.asarray(
                    hashlib.sha256(split_source.read_bytes()).hexdigest()
                ),
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    summary: dict[str, float | str] = {
        "n_objects": 170.0,
        "seed_count": float(len(K3_OOF_SEEDS)),
        "source_implementation_commit": implementation_commits[0],
        "artifact_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
    for label, condition in (
        ("v1", "raw"),
        ("atlas", "atlas"),
        ("deranged", "exhaustive_deranged"),
        ("zero", "zero"),
    ):
        metrics = summarize_errors(mean_errors[condition])
        summary.update(
            {f"{label}_{key}": value for key, value in metrics.items() if key != "n_objects"}
        )
    return summary


def audit_publication_checkpoint_set(
    model_directory: str | Path,
    output_path: str | Path,
    *,
    require_label_shuffle: bool = True,
) -> dict[str, object]:
    """Verify the complete provenance-bound checkpoint graph."""
    destination = Path(output_path)
    if destination.name != "checkpoint-audit.json":
        raise K3PublicationRunError("checkpoint audit must be named checkpoint-audit.json")
    if destination.exists():
        raise K3PublicationRunError(f"refusing to overwrite checkpoint audit {destination}")
    root = Path(model_directory)
    checkpoint_hashes: dict[str, str] = {}
    implementation_commits: set[str] = set()
    model_configurations: set[str] = set()

    def load_bound(path: Path, *, stage: str, seed: int) -> dict[str, object]:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, ValueError) as exc:
            raise K3PublicationRunError(
                f"cannot load publication checkpoint {path.name}: {exc}"
            ) from exc
        provenance = checkpoint.get("run_provenance")
        if (
            checkpoint.get("schema") != "delphi.k3-checkpoint.v2"
            or checkpoint.get("stage") != stage
            or int(checkpoint.get("config", {}).get("seed", -1)) != seed
            or not isinstance(provenance, dict)
            or provenance.get("protocol_sha256") != K3_PROTOCOL_SHA256
        ):
            raise K3PublicationRunError(
                f"checkpoint {path.name} has invalid schema/stage/seed/protocol provenance"
            )
        commit = str(provenance.get("implementation_commit", ""))
        if len(commit) != 40 or any(
            character not in "0123456789abcdef" for character in commit
        ):
            raise K3PublicationRunError(
                f"checkpoint {path.name} has invalid implementation provenance"
            )
        try:
            model_configuration = json.dumps(
                checkpoint["model_config"], sort_keys=True, separators=(",", ":")
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise K3PublicationRunError(
                f"checkpoint {path.name} has invalid model configuration: {exc}"
            ) from exc
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checkpoint_hashes[path.name] = digest
        implementation_commits.add(commit)
        model_configurations.add(model_configuration)
        return checkpoint

    synthetic_hashes: dict[int, str] = {}
    for seed in K3_OOF_SEEDS:
        path = root / f"synthetic-seed-{seed}.pt"
        load_bound(path, stage="synthetic", seed=seed)
        synthetic_hashes[seed] = checkpoint_hashes[path.name]
    for seed in K3_OOF_SEEDS:
        for fold in range(5):
            path = root / f"real-fold-{fold}-seed-{seed}.pt"
            checkpoint = load_bound(path, stage="real-oof", seed=seed)
            if (
                checkpoint["run_provenance"].get("pretrained_checkpoint_sha256")
                != synthetic_hashes[seed]
            ):
                raise K3PublicationRunError(
                    f"checkpoint {path.name} is not bound to synthetic seed {seed}"
                )
    label_mapping_hash: str | None = None
    if require_label_shuffle:
        seed = K3_LABEL_SHUFFLE_MODEL_SEED
        path = root / f"synthetic-label-shuffle-seed-{seed}.pt"
        checkpoint = load_bound(path, stage="synthetic-label-shuffle", seed=seed)
        mapping_path = root / f"synthetic-label-shuffle-seed-{seed}.labels.json"
        try:
            label_mapping_hash = hashlib.sha256(mapping_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise K3PublicationRunError(
                f"cannot load label-shuffle mapping: {exc}"
            ) from exc
        if checkpoint["run_provenance"].get("label_mapping_sha256") != label_mapping_hash:
            raise K3PublicationRunError(
                "label-shuffle checkpoint does not bind the exact label mapping"
            )
    if len(implementation_commits) != 1:
        raise K3PublicationRunError("publication checkpoints use multiple Git commits")
    if len(model_configurations) != 1:
        raise K3PublicationRunError("publication checkpoints use multiple model configurations")

    payload: dict[str, object] = {
        "schema": "delphi.k3-checkpoint-audit.v1",
        "passed": True,
        "protocol_sha256": K3_PROTOCOL_SHA256,
        "implementation_commit": next(iter(implementation_commits)),
        "seeds": list(K3_OOF_SEEDS),
        "synthetic_checkpoint_count": len(K3_OOF_SEEDS),
        "real_oof_checkpoint_count": 5 * len(K3_OOF_SEEDS),
        "label_shuffle_required": require_label_shuffle,
        "label_mapping_sha256": label_mapping_hash,
        "checkpoint_sha256": checkpoint_hashes,
    }
    audit_sha256 = _write_atomic_json(destination, payload)
    return {**payload, "artifact_sha256": audit_sha256}


def calibrate_real_oof_artifacts(
    calibration_directory: str | Path,
    split_path: str | Path,
    raw_aggregate_path: str | Path,
    output_path: str | Path,
    *,
    seed: int,
) -> dict[str, float | str]:
    """Cross-fit exact 90%/95% cone radii without outer-test contact."""
    destination = Path(output_path)
    expected_output = f"real-oof-seed-{seed}-calibrated.npz"
    if destination.name != expected_output:
        raise K3PublicationRunError(f"calibrated artifact must be named {expected_output}")
    if destination.exists():
        raise K3PublicationRunError(f"refusing to overwrite calibrated artifact {destination}")
    try:
        split = json.loads(Path(split_path).read_text(encoding="utf-8"))
        fold_rows = sorted(split["folds"], key=lambda row: int(row["fold"]))
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise K3PublicationRunError(f"cannot load OOF splits: {exc}") from exc
    if [int(row["fold"]) for row in fold_rows] != list(range(5)):
        raise K3PublicationRunError("cross-fitted calibration requires folds 0 through 4")
    if any(
        len(row.get("calibration_ids", ())) != 14 or len(row.get("test_ids", ())) != 34
        for row in fold_rows
    ):
        raise K3PublicationRunError(
            "definitive calibration requires 14 calibration and 34 test objects per fold"
        )

    raw_source = Path(raw_aggregate_path)
    try:
        with np.load(raw_source, allow_pickle=False) as raw:
            raw_ids = tuple(raw["object_ids"].astype(str))
            raw_folds = np.asarray(raw["folds"], dtype=np.int8)
            raw_errors = np.asarray(raw["oracle_errors_deg"], dtype=np.float64)
    except (OSError, ValueError, KeyError) as exc:
        raise K3PublicationRunError(f"cannot load raw OOF artifact for calibration: {exc}") from exc
    expected_ids = tuple(
        object_id for row in fold_rows for object_id in row["test_ids"]
    )
    expected_folds = np.concatenate(
        [
            np.full(34, int(row["fold"]), dtype=np.int8)
            for row in fold_rows
        ]
    )
    if (
        raw_ids != expected_ids
        or not np.array_equal(raw_folds, expected_folds)
        or raw_errors.shape != (170,)
        or not np.all(np.isfinite(raw_errors))
        or np.any(raw_errors < 0)
        or np.any(raw_errors > 90)
    ):
        raise K3PublicationRunError(
            "raw OOF artifact does not match the exact 170-object frozen split"
        )

    calibration_ids = np.empty((5, 14), dtype="U64")
    calibration_errors = np.empty((5, 14), dtype=np.float64)
    cone90 = np.empty(5, dtype=np.float64)
    cone95 = np.empty(5, dtype=np.float64)
    input_hashes: list[str] = []
    root = Path(calibration_directory)
    for row in fold_rows:
        fold = int(row["fold"])
        source = root / f"real-fold-{fold}-seed-{seed}-calibration.npz"
        try:
            with np.load(source, allow_pickle=False) as artifact:
                identifiers = artifact["object_ids"].astype(str)
                errors = np.asarray(artifact["oracle_errors_deg"], dtype=np.float64)
        except (OSError, ValueError, KeyError) as exc:
            raise K3PublicationRunError(
                f"cannot load inner-calibration fold {fold}: {exc}"
            ) from exc
        expected_calibration = tuple(row["calibration_ids"])
        if tuple(identifiers) != expected_calibration or errors.shape != (14,):
            raise K3PublicationRunError(
                f"fold {fold} calibration IDs/errors differ from the frozen role"
            )
        if not np.all(np.isfinite(errors)) or np.any(errors < 0) or np.any(errors > 90):
            raise K3PublicationRunError(f"fold {fold} calibration errors are invalid")
        calibration_ids[fold] = identifiers
        calibration_errors[fold] = errors
        cone90[fold] = conformal_radius(errors, 0.90)
        cone95[fold] = conformal_radius(errors, 0.95)
        input_hashes.append(hashlib.sha256(source.read_bytes()).hexdigest())
    if not np.all(cone95 == 90.0):
        raise K3PublicationRunError(
            "14-object fold calibration must yield the bounded 90-degree 95% radius"
        )

    object_cone90 = cone90[raw_folds]
    object_cone95 = cone95[raw_folds]
    covered90 = raw_errors <= object_cone90
    covered95 = raw_errors <= object_cone95
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                schema=np.asarray("delphi.k3-real-oof-calibration.v1"),
                seed=np.asarray(seed, dtype=np.int64),
                object_ids=np.asarray(raw_ids),
                folds=raw_folds,
                oracle_errors_deg=raw_errors,
                object_cone90_deg=object_cone90,
                object_cone95_deg=object_cone95,
                covered90=covered90,
                covered95=covered95,
                fold_cone90_deg=cone90,
                fold_cone95_deg=cone95,
                calibration_object_ids=calibration_ids,
                calibration_errors_deg=calibration_errors,
                calibration_artifact_sha256=np.asarray(input_hashes),
                raw_artifact_sha256=np.asarray(
                    hashlib.sha256(raw_source.read_bytes()).hexdigest()
                ),
                splits_sha256=np.asarray(
                    hashlib.sha256(Path(split_path).read_bytes()).hexdigest()
                ),
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "seed": float(seed),
        "n_objects": float(raw_errors.size),
        "cone90_min_deg": float(np.min(cone90)),
        "cone90_max_deg": float(np.max(cone90)),
        "cone95_min_deg": float(np.min(cone95)),
        "cone95_max_deg": float(np.max(cone95)),
        "empirical_coverage90": float(np.mean(covered90)),
        "empirical_coverage95": float(np.mean(covered95)),
        "artifact_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }


def aggregate_real_oof_input_swap_artifacts(
    evaluation_directory: str | Path,
    split_path: str | Path,
    raw_aggregate_path: str | Path,
    output_path: str | Path,
    *,
    seed: int,
) -> dict[str, float | str]:
    """Aggregate aligned input-swap folds and compute paired degradation."""
    destination = Path(output_path)
    if destination.exists():
        raise K3PublicationRunError(f"refusing to overwrite control artifact {destination}")
    try:
        split = json.loads(Path(split_path).read_text(encoding="utf-8"))
        fold_rows = sorted(split["folds"], key=lambda row: int(row["fold"]))
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise K3PublicationRunError(f"cannot load OOF splits: {exc}") from exc
    if [int(row["fold"]) for row in fold_rows] != list(range(5)):
        raise K3PublicationRunError("input-swap aggregation requires folds 0 through 4")

    identifiers: list[str] = []
    donor_identifiers: list[str] = []
    errors: list[np.ndarray] = []
    grids: list[np.ndarray] = []
    folds: list[np.ndarray] = []
    input_hashes: list[str] = []
    root = Path(evaluation_directory)
    for row in fold_rows:
        fold = int(row["fold"])
        source = root / f"real-fold-{fold}-seed-{seed}-input-swap.npz"
        try:
            with np.load(source, allow_pickle=False) as artifact:
                fold_ids = artifact["object_ids"].astype(str)
                donor_ids = artifact["input_donor_ids"].astype(str)
                fold_errors = np.asarray(artifact["oracle_errors_deg"], dtype=np.float64)
                fold_grids = np.asarray(artifact["score_grids"], dtype=np.float32)
                derangement_seeds = np.asarray(artifact["derangement_seeds"], dtype=np.int64)
        except (OSError, ValueError, KeyError) as exc:
            raise K3PublicationRunError(f"cannot load input-swap fold {fold}: {exc}") from exc
        expected_ids = tuple(row["test_ids"])
        donors = deterministic_derangement_indices(
            len(expected_ids), seed=K3_INPUT_SWAP_SEED + fold
        )
        expected_donors = tuple(expected_ids[int(index)] for index in donors)
        if tuple(fold_ids) != expected_ids or tuple(donor_ids) != expected_donors:
            raise K3PublicationRunError(
                f"input-swap fold {fold} target/donor mapping differs from the fixed derangement"
            )
        count = len(expected_ids)
        if (
            fold_errors.shape != (count,)
            or fold_grids.shape != (count, 6144)
            or not np.all(np.isfinite(fold_errors))
            or not np.all(np.isfinite(fold_grids))
            or not np.array_equal(
                derangement_seeds, np.full(count, K3_INPUT_SWAP_SEED + fold)
            )
        ):
            raise K3PublicationRunError(f"input-swap fold {fold} arrays are invalid")
        identifiers.extend(fold_ids.tolist())
        donor_identifiers.extend(donor_ids.tolist())
        errors.append(fold_errors)
        grids.append(fold_grids)
        folds.append(np.full(count, fold, dtype=np.int8))
        input_hashes.append(hashlib.sha256(source.read_bytes()).hexdigest())

    all_ids = tuple(identifiers)
    all_folds = np.concatenate(folds)
    all_errors = np.concatenate(errors)
    all_grids = np.concatenate(grids)
    raw_source = Path(raw_aggregate_path)
    try:
        with np.load(raw_source, allow_pickle=False) as raw:
            raw_ids = tuple(raw["object_ids"].astype(str))
            raw_folds = np.asarray(raw["folds"], dtype=np.int8)
            raw_errors = np.asarray(raw["oracle_errors_deg"], dtype=np.float64)
    except (OSError, ValueError, KeyError) as exc:
        raise K3PublicationRunError(f"cannot load aligned raw OOF artifact: {exc}") from exc
    if (
        raw_ids != all_ids
        or not np.array_equal(raw_folds, all_folds)
        or raw_errors.shape != all_errors.shape
        or not np.all(np.isfinite(raw_errors))
    ):
        raise K3PublicationRunError("raw and input-swap OOF artifacts are not exactly aligned")
    degradation = all_errors - raw_errors

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                schema=np.asarray("delphi.k3-real-oof-input-swap.v1"),
                seed=np.asarray(seed, dtype=np.int64),
                object_ids=np.asarray(all_ids),
                input_donor_ids=np.asarray(donor_identifiers),
                folds=all_folds,
                score_grids=all_grids,
                oracle_errors_deg=all_errors,
                raw_oracle_errors_deg=raw_errors,
                degradation_deg=degradation,
                input_artifact_sha256=np.asarray(input_hashes),
                raw_artifact_sha256=np.asarray(hashlib.sha256(raw_source.read_bytes()).hexdigest()),
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    summary: dict[str, float | str] = summarize_errors(all_errors)
    summary.update(
        {
            "seed": float(seed),
            "raw_mean_error_deg": float(np.mean(raw_errors)),
            "input_swap_gap_deg": float(np.mean(degradation)),
            "artifact_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "splits_sha256": hashlib.sha256(Path(split_path).read_bytes()).hexdigest(),
        }
    )
    return summary


def aggregate_real_oof_diagnostic_artifacts(
    evaluation_directory: str | Path,
    split_path: str | Path,
    raw_aggregate_path: str | Path,
    output_path: str | Path,
    *,
    seed: int,
    control: K3DiagnosticControl,
) -> dict[str, float | str]:
    """Aggregate an ablation arm and retain its exactly paired raw degradation."""
    if control not in K3_DIAGNOSTIC_CONTROLS:
        raise K3PublicationRunError(f"unsupported K3 diagnostic control {control!r}")
    destination = Path(output_path)
    expected_output = f"real-oof-seed-{seed}-{control}.npz"
    if destination.name != expected_output:
        raise K3PublicationRunError(f"diagnostic aggregate must be named {expected_output}")
    if destination.exists():
        raise K3PublicationRunError(f"refusing to overwrite diagnostic artifact {destination}")
    try:
        split = json.loads(Path(split_path).read_text(encoding="utf-8"))
        fold_rows = sorted(split["folds"], key=lambda row: int(row["fold"]))
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise K3PublicationRunError(f"cannot load OOF splits: {exc}") from exc
    if [int(row["fold"]) for row in fold_rows] != list(range(5)):
        raise K3PublicationRunError("diagnostic aggregation requires folds 0 through 4")

    identifiers: list[str] = []
    errors: list[np.ndarray] = []
    grids: list[np.ndarray] = []
    folds: list[np.ndarray] = []
    input_hashes: list[str] = []
    root = Path(evaluation_directory)
    for row in fold_rows:
        fold = int(row["fold"])
        source = root / f"real-fold-{fold}-seed-{seed}-{control}.npz"
        try:
            with np.load(source, allow_pickle=False) as artifact:
                fold_ids = artifact["object_ids"].astype(str)
                fold_errors = np.asarray(artifact["oracle_errors_deg"], dtype=np.float64)
                fold_grids = np.asarray(artifact["score_grids"], dtype=np.float32)
                controls = artifact["diagnostic_control"].astype(str)
        except (OSError, ValueError, KeyError) as exc:
            raise K3PublicationRunError(
                f"cannot load {control} diagnostic fold {fold}: {exc}"
            ) from exc
        expected_ids = tuple(row["test_ids"])
        count = len(expected_ids)
        if tuple(fold_ids) != expected_ids:
            raise K3PublicationRunError(
                f"diagnostic fold {fold} IDs/order differ from the frozen test split"
            )
        if (
            fold_errors.shape != (count,)
            or fold_grids.shape != (count, 6144)
            or controls.shape != (count,)
            or not np.all(controls == control)
            or not np.all(np.isfinite(fold_errors))
            or not np.all(np.isfinite(fold_grids))
        ):
            raise K3PublicationRunError(f"diagnostic fold {fold} arrays are invalid")
        identifiers.extend(fold_ids.tolist())
        errors.append(fold_errors)
        grids.append(fold_grids)
        folds.append(np.full(count, fold, dtype=np.int8))
        input_hashes.append(hashlib.sha256(source.read_bytes()).hexdigest())

    all_ids = tuple(identifiers)
    if len(all_ids) != len(set(all_ids)):
        raise K3PublicationRunError("diagnostic OOF folds contain duplicate object IDs")
    all_folds = np.concatenate(folds)
    all_errors = np.concatenate(errors)
    all_grids = np.concatenate(grids)
    raw_source = Path(raw_aggregate_path)
    try:
        with np.load(raw_source, allow_pickle=False) as raw:
            raw_ids = tuple(raw["object_ids"].astype(str))
            raw_folds = np.asarray(raw["folds"], dtype=np.int8)
            raw_errors = np.asarray(raw["oracle_errors_deg"], dtype=np.float64)
    except (OSError, ValueError, KeyError) as exc:
        raise K3PublicationRunError(f"cannot load aligned raw OOF artifact: {exc}") from exc
    if (
        raw_ids != all_ids
        or not np.array_equal(raw_folds, all_folds)
        or raw_errors.shape != all_errors.shape
        or not np.all(np.isfinite(raw_errors))
    ):
        raise K3PublicationRunError("raw and diagnostic OOF artifacts are not exactly aligned")
    degradation = all_errors - raw_errors

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                schema=np.asarray("delphi.k3-real-oof-diagnostic.v1"),
                seed=np.asarray(seed, dtype=np.int64),
                control=np.asarray(control),
                object_ids=np.asarray(all_ids),
                folds=all_folds,
                score_grids=all_grids,
                oracle_errors_deg=all_errors,
                raw_oracle_errors_deg=raw_errors,
                degradation_deg=degradation,
                input_artifact_sha256=np.asarray(input_hashes),
                raw_artifact_sha256=np.asarray(
                    hashlib.sha256(raw_source.read_bytes()).hexdigest()
                ),
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    summary: dict[str, float | str] = summarize_errors(all_errors)
    summary.update(
        {
            "seed": float(seed),
            "control": control,
            "raw_mean_error_deg": float(np.mean(raw_errors)),
            "diagnostic_gap_deg": float(np.mean(degradation)),
            "artifact_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "splits_sha256": hashlib.sha256(Path(split_path).read_bytes()).hexdigest(),
        }
    )
    return summary


def evaluate_real_oof_ensemble(
    evaluation_directory: str | Path,
    model_directory: str | Path,
    catalog_path: str | Path,
    dump_root: str | Path,
    split_path: str | Path,
    output_path: str | Path,
    *,
    seeds: Sequence[int],
    device: str = "cuda",
) -> dict[str, float | str]:
    """Evaluate the locked seed ensemble by averaging complete score maps.

    One K=3 set is extracted from the arithmetic-mean map for each object and
    then jointly refined against the arithmetic mean of the seed scorers.  At
    no point are independently predicted axes averaged.
    """
    selected_seeds = tuple(int(seed) for seed in seeds)
    if len(selected_seeds) < 2 or len(selected_seeds) != len(set(selected_seeds)):
        raise K3PublicationRunError("ensemble seeds must contain at least two unique values")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise K3PublicationRunError("CUDA was requested but is unavailable")
    destination = Path(output_path)
    if destination.exists():
        raise K3PublicationRunError(f"refusing to overwrite ensemble artifact {destination}")
    try:
        split = json.loads(Path(split_path).read_text(encoding="utf-8"))
        fold_rows = sorted(split["folds"], key=lambda row: int(row["fold"]))
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise K3PublicationRunError(f"cannot load OOF splits: {exc}") from exc
    if [int(row["fold"]) for row in fold_rows] != list(range(5)):
        raise K3PublicationRunError("OOF ensemble requires exactly folds 0 through 4")
    expected_ids = tuple(
        object_id for row in fold_rows for object_id in row["test_ids"]
    )
    expected_folds = np.concatenate(
        [np.full(len(row["test_ids"]), int(row["fold"]), dtype=np.int8) for row in fold_rows]
    )
    if not expected_ids or len(expected_ids) != len(set(expected_ids)):
        raise K3PublicationRunError("frozen OOF splits must contain unique nonempty object IDs")

    evaluation_root = Path(evaluation_directory)
    seed_grids: list[np.ndarray] = []
    aggregate_hashes: list[str] = []
    for seed in selected_seeds:
        source = evaluation_root / f"real-oof-seed-{seed}.npz"
        try:
            with np.load(source, allow_pickle=False) as artifact:
                identifiers = tuple(artifact["object_ids"].astype(str))
                folds = np.asarray(artifact["folds"], dtype=np.int8)
                grids = np.asarray(artifact["score_grids"], dtype=np.float32)
        except (OSError, ValueError, KeyError) as exc:
            raise K3PublicationRunError(f"cannot load seed {seed} aggregate artifact: {exc}") from exc
        if identifiers != expected_ids or not np.array_equal(folds, expected_folds):
            raise K3PublicationRunError(
                f"seed {seed} aggregate IDs/folds differ from the frozen OOF splits"
            )
        if grids.shape != (len(expected_ids), 6144) or not np.all(np.isfinite(grids)):
            raise K3PublicationRunError(f"seed {seed} aggregate score grids are invalid")
        seed_grids.append(grids)
        aggregate_hashes.append(hashlib.sha256(source.read_bytes()).hexdigest())
    mean_grids = ensemble_score_grids(seed_grids)

    errors = np.empty(len(expected_ids), dtype=np.float64)
    grid_errors = np.empty(len(expected_ids), dtype=np.float64)
    initial_axes = np.empty((len(expected_ids), 3, 3), dtype=np.float64)
    refined_axes = np.empty_like(initial_axes)
    refined_scores = np.empty((len(expected_ids), 3), dtype=np.float64)
    inference_wall_seconds = np.empty(len(expected_ids), dtype=np.float64)
    cached_grid_max_abs_differences = np.empty(len(expected_ids), dtype=np.float64)
    cached_grid_mean_abs_differences = np.empty(len(expected_ids), dtype=np.float64)
    cached_grid_centered_rms_differences = np.empty(
        len(expected_ids), dtype=np.float64
    )
    cached_grid_normalized_rms_differences = np.empty(
        len(expected_ids), dtype=np.float64
    )
    cached_mode_indices = np.empty((len(expected_ids), 3), dtype=np.int64)
    deployed_mode_indices = np.empty_like(cached_mode_indices)
    checkpoint_names: list[str] = []
    checkpoint_hashes: list[str] = []
    common_model_config: K3ScoreModelConfig | None = None
    model_root = Path(model_directory)
    offset = 0
    started = time.time()
    for row in fold_rows:
        fold = int(row["fold"])
        test_ids = tuple(row["test_ids"])
        examples = load_real_examples(catalog_path, dump_root, test_ids)
        models: list[CandidateConditionedScorer] = []
        for seed in selected_seeds:
            checkpoint_path = model_root / f"real-fold-{fold}-seed-{seed}.pt"
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if checkpoint.get("stage") != "real-oof":
                    raise K3PublicationRunError(
                        f"checkpoint {checkpoint_path.name} is not a real-oof checkpoint"
                    )
                if int(checkpoint.get("config", {}).get("seed", -1)) != seed:
                    raise K3PublicationRunError(
                        f"checkpoint {checkpoint_path.name} has the wrong training seed"
                    )
                model_config = K3ScoreModelConfig(**checkpoint["model_config"])
                model = CandidateConditionedScorer(model_config)
                model.load_state_dict(checkpoint["model_state_dict"])
            except K3PublicationRunError:
                raise
            except (OSError, KeyError, TypeError, RuntimeError, ValueError) as exc:
                raise K3PublicationRunError(
                    f"cannot load ensemble checkpoint {checkpoint_path.name}: {exc}"
                ) from exc
            if common_model_config is None:
                common_model_config = model_config
            elif model_config != common_model_config:
                raise K3PublicationRunError("ensemble checkpoints have different model configurations")
            model.to(device).eval()
            models.append(model)
            checkpoint_names.append(checkpoint_path.name)
            checkpoint_hashes.append(hashlib.sha256(checkpoint_path.read_bytes()).hexdigest())

        for local_index, example in enumerate(examples):
            index = offset + local_index
            inference_started = time.perf_counter()
            raw = collate_examples((example,))
            inputs = {
                key: value.to(device)
                for key, value in raw.items()
                if key in {
                    "phase_features", "phase_mask", "geometry_features",
                    "epoch_features", "epoch_mask",
                }
            }
            deployed_grids = np.stack(
                [
                    np.asarray(
                        score_axial_grid(model, inputs, chunk_size=1024),
                        dtype=np.float64,
                    )
                    for model in models
                ]
            )
            deployed_mean_grid = ensemble_score_grids(
                deployed_grids[:, None, :]
            )[0]
            cached_mean_grid = mean_grids[index]
            absolute_difference = np.abs(deployed_mean_grid - cached_mean_grid)
            cached_grid_max_abs_differences[index] = float(np.max(absolute_difference))
            cached_grid_mean_abs_differences[index] = float(np.mean(absolute_difference))
            centered_difference = (
                deployed_mean_grid
                - float(np.mean(deployed_mean_grid))
                - cached_mean_grid
                + float(np.mean(cached_mean_grid))
            )
            centered_rms = float(np.sqrt(np.mean(np.square(centered_difference))))
            normalized_rms = centered_rms / max(
                float(np.std(cached_mean_grid)), 1.0
            )
            cached_grid_centered_rms_differences[index] = centered_rms
            cached_grid_normalized_rms_differences[index] = normalized_rms
            if normalized_rms > K3_DEPLOYED_GRID_MAX_NORMALIZED_RMS:
                raise K3PublicationRunError(
                    "deployed ensemble score landscape differs materially from cached "
                    f"seed maps for {example.object_id} (normalized centered RMS "
                    f"{normalized_rms:.6g})"
                )
            cached_modes = modes_from_score_grid(cached_mean_grid)
            modes = modes_from_score_grid(deployed_mean_grid)
            cached_indices = tuple(int(mode.grid_index) for mode in cached_modes)
            deployed_indices = tuple(int(mode.grid_index) for mode in modes)
            cached_mode_indices[index] = cached_indices
            deployed_mode_indices[index] = deployed_indices
            starts = np.asarray([mode.axis_xyz for mode in modes], dtype=np.float64)
            axes, scores = refine_ensemble_axes(models, inputs, starts)
            inference_wall_seconds[index] = time.perf_counter() - inference_started
            mean_grids[index] = deployed_mean_grid
            initial_axes[index] = starts
            refined_axes[index] = axes
            refined_scores[index] = scores
            grid_errors[index] = oracle_at_k_error_deg(starts, example.target_axes, k=3)
            errors[index] = oracle_at_k_error_deg(axes, example.target_axes, k=3)
        offset += len(examples)
    if offset != len(expected_ids):
        raise K3PublicationRunError("real-data loader changed the frozen OOF cohort size")

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                schema=np.asarray("delphi.k3-real-oof-ensemble.v1"),
                seeds=np.asarray(selected_seeds, dtype=np.int64),
                object_ids=np.asarray(expected_ids),
                folds=expected_folds,
                score_grids=mean_grids.astype(np.float32),
                initial_axes=initial_axes,
                refined_axes=refined_axes,
                refined_scores=refined_scores,
                inference_wall_seconds=inference_wall_seconds,
                cached_grid_max_abs_differences=cached_grid_max_abs_differences,
                cached_grid_mean_abs_differences=cached_grid_mean_abs_differences,
                cached_grid_centered_rms_differences=(
                    cached_grid_centered_rms_differences
                ),
                cached_grid_normalized_rms_differences=(
                    cached_grid_normalized_rms_differences
                ),
                cached_mode_indices=cached_mode_indices,
                deployed_mode_indices=deployed_mode_indices,
                grid_oracle_errors_deg=grid_errors,
                oracle_errors_deg=errors,
                input_aggregate_sha256=np.asarray(aggregate_hashes),
                checkpoint_names=np.asarray(checkpoint_names),
                checkpoint_sha256=np.asarray(checkpoint_hashes),
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    summary: dict[str, float | str] = summarize_errors(errors)
    summary.update(
        {
            "grid_mean_error_deg": float(np.mean(grid_errors)),
            "seed_count": float(len(selected_seeds)),
            "artifact_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "splits_sha256": hashlib.sha256(Path(split_path).read_bytes()).hexdigest(),
            "elapsed_seconds": float(time.time() - started),
            "summed_inference_wall_seconds": float(
                np.sum(inference_wall_seconds)
            ),
            "cached_grid_max_abs_difference": float(
                np.max(cached_grid_max_abs_differences)
            ),
            "cached_grid_mean_abs_difference": float(
                np.mean(cached_grid_mean_abs_differences)
            ),
            "cached_grid_max_centered_rms_difference": float(
                np.max(cached_grid_centered_rms_differences)
            ),
            "cached_grid_max_normalized_rms_difference": float(
                np.max(cached_grid_normalized_rms_differences)
            ),
            "cached_mode_agreement_fraction": float(
                np.mean(np.all(cached_mode_indices == deployed_mode_indices, axis=1))
            ),
        }
    )
    return summary


def evaluate_real_calibration_ensemble(
    evaluation_directory: str | Path,
    model_directory: str | Path,
    catalog_path: str | Path,
    dump_root: str | Path,
    split_path: str | Path,
    output_path: str | Path,
    *,
    seeds: Sequence[int],
    device: str = "cuda",
) -> dict[str, float | str]:
    """Evaluate the deployed ensemble on each untouched calibration role."""
    selected_seeds = tuple(int(seed) for seed in seeds)
    if len(selected_seeds) < 2 or len(selected_seeds) != len(set(selected_seeds)):
        raise K3PublicationRunError("ensemble seeds must contain at least two unique values")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise K3PublicationRunError("CUDA was requested but is unavailable")
    destination = Path(output_path)
    if destination.name != "real-calibration-ensemble.npz":
        raise K3PublicationRunError(
            "ensemble calibration artifact must be named real-calibration-ensemble.npz"
        )
    if destination.exists():
        raise K3PublicationRunError(
            f"refusing to overwrite ensemble calibration artifact {destination}"
        )
    try:
        split = json.loads(Path(split_path).read_text(encoding="utf-8"))
        fold_rows = sorted(split["folds"], key=lambda row: int(row["fold"]))
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise K3PublicationRunError(f"cannot load OOF splits: {exc}") from exc
    if [int(row["fold"]) for row in fold_rows] != list(range(5)) or any(
        len(row.get("calibration_ids", ())) != 14 for row in fold_rows
    ):
        raise K3PublicationRunError(
            "ensemble calibration requires five folds of 14 calibration objects"
        )

    errors = np.empty(70, dtype=np.float64)
    grid_errors = np.empty(70, dtype=np.float64)
    mean_grids = np.empty((70, 6144), dtype=np.float32)
    initial_axes = np.empty((70, 3, 3), dtype=np.float64)
    refined_axes = np.empty_like(initial_axes)
    refined_scores = np.empty((70, 3), dtype=np.float64)
    identifiers: list[str] = []
    folds = np.empty(70, dtype=np.int8)
    calibration_hashes: list[str] = []
    checkpoint_names: list[str] = []
    checkpoint_hashes: list[str] = []
    common_model_config: K3ScoreModelConfig | None = None
    evaluation_root = Path(evaluation_directory)
    model_root = Path(model_directory)
    offset = 0
    started = time.time()
    for row in fold_rows:
        fold = int(row["fold"])
        calibration_ids = tuple(row["calibration_ids"])
        examples = load_real_examples(catalog_path, dump_root, calibration_ids)
        if tuple(example.object_id for example in examples) != calibration_ids:
            raise K3PublicationRunError(
                f"fold {fold} calibration loader changed the frozen object order"
            )
        seed_grids: list[np.ndarray] = []
        for seed in selected_seeds:
            source = evaluation_root / f"real-fold-{fold}-seed-{seed}-calibration.npz"
            try:
                with np.load(source, allow_pickle=False) as artifact:
                    source_ids = tuple(artifact["object_ids"].astype(str))
                    grids = np.asarray(artifact["score_grids"], dtype=np.float32)
            except (OSError, ValueError, KeyError) as exc:
                raise K3PublicationRunError(
                    f"cannot load seed {seed} calibration fold {fold}: {exc}"
                ) from exc
            if source_ids != calibration_ids or grids.shape != (14, 6144):
                raise K3PublicationRunError(
                    f"seed {seed} calibration fold {fold} is not exactly aligned"
                )
            if not np.all(np.isfinite(grids)):
                raise K3PublicationRunError(
                    f"seed {seed} calibration fold {fold} has non-finite scores"
                )
            seed_grids.append(grids)
            calibration_hashes.append(hashlib.sha256(source.read_bytes()).hexdigest())
        fold_mean_grids = ensemble_score_grids(seed_grids).astype(np.float32)

        models: list[CandidateConditionedScorer] = []
        for seed in selected_seeds:
            checkpoint_path = model_root / f"real-fold-{fold}-seed-{seed}.pt"
            try:
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                if checkpoint.get("stage") != "real-oof" or int(
                    checkpoint.get("config", {}).get("seed", -1)
                ) != seed:
                    raise K3PublicationRunError(
                        f"checkpoint {checkpoint_path.name} has wrong stage or seed"
                    )
                model_config = K3ScoreModelConfig(**checkpoint["model_config"])
                model = CandidateConditionedScorer(model_config)
                model.load_state_dict(checkpoint["model_state_dict"])
            except K3PublicationRunError:
                raise
            except (OSError, KeyError, TypeError, RuntimeError, ValueError) as exc:
                raise K3PublicationRunError(
                    f"cannot load ensemble checkpoint {checkpoint_path.name}: {exc}"
                ) from exc
            if common_model_config is None:
                common_model_config = model_config
            elif model_config != common_model_config:
                raise K3PublicationRunError(
                    "ensemble checkpoints have different model configurations"
                )
            model.to(device).eval()
            models.append(model)
            checkpoint_names.append(checkpoint_path.name)
            checkpoint_hashes.append(hashlib.sha256(checkpoint_path.read_bytes()).hexdigest())

        for local_index, example in enumerate(examples):
            index = offset + local_index
            modes = modes_from_score_grid(fold_mean_grids[local_index])
            starts = np.asarray([mode.axis_xyz for mode in modes], dtype=np.float64)
            raw = collate_examples((example,))
            inputs = {
                key: value.to(device)
                for key, value in raw.items()
                if key
                in {
                    "phase_features",
                    "phase_mask",
                    "geometry_features",
                    "epoch_features",
                    "epoch_mask",
                }
            }
            axes, scores = refine_ensemble_axes(models, inputs, starts)
            mean_grids[index] = fold_mean_grids[local_index]
            initial_axes[index] = starts
            refined_axes[index] = axes
            refined_scores[index] = scores
            grid_errors[index] = oracle_at_k_error_deg(
                starts, example.target_axes, k=3
            )
            errors[index] = oracle_at_k_error_deg(axes, example.target_axes, k=3)
            identifiers.append(example.object_id)
            folds[index] = fold
        offset += len(examples)
    if offset != 70:
        raise K3PublicationRunError("ensemble calibration did not evaluate 70 fold roles")

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                schema=np.asarray("delphi.k3-real-calibration-ensemble.v1"),
                seeds=np.asarray(selected_seeds, dtype=np.int64),
                object_ids=np.asarray(identifiers),
                folds=folds,
                score_grids=mean_grids,
                initial_axes=initial_axes,
                refined_axes=refined_axes,
                refined_scores=refined_scores,
                grid_oracle_errors_deg=grid_errors,
                oracle_errors_deg=errors,
                input_calibration_sha256=np.asarray(calibration_hashes),
                checkpoint_names=np.asarray(checkpoint_names),
                checkpoint_sha256=np.asarray(checkpoint_hashes),
                splits_sha256=np.asarray(
                    hashlib.sha256(Path(split_path).read_bytes()).hexdigest()
                ),
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    summary: dict[str, float | str] = summarize_errors(errors)
    summary.update(
        {
            "grid_mean_error_deg": float(np.mean(grid_errors)),
            "seed_count": float(len(selected_seeds)),
            "artifact_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "elapsed_seconds": float(time.time() - started),
        }
    )
    return summary


def calibrate_real_oof_ensemble_artifacts(
    calibration_ensemble_path: str | Path,
    split_path: str | Path,
    oof_ensemble_path: str | Path,
    output_path: str | Path,
) -> dict[str, float | str]:
    """Apply fold-local calibration radii to the matching ensemble OOF rows."""
    destination = Path(output_path)
    if destination.name != "real-oof-ensemble-calibrated.npz":
        raise K3PublicationRunError(
            "calibrated ensemble must be named real-oof-ensemble-calibrated.npz"
        )
    if destination.exists():
        raise K3PublicationRunError(
            f"refusing to overwrite calibrated ensemble artifact {destination}"
        )
    try:
        split_source = Path(split_path)
        split = json.loads(split_source.read_text(encoding="utf-8"))
        fold_rows = sorted(split["folds"], key=lambda row: int(row["fold"]))
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise K3PublicationRunError(f"cannot load OOF splits: {exc}") from exc
    if [int(row["fold"]) for row in fold_rows] != list(range(5)) or any(
        len(row.get("calibration_ids", ())) != 14
        or len(row.get("test_ids", ())) != 34
        for row in fold_rows
    ):
        raise K3PublicationRunError(
            "ensemble calibration requires the exact five 14/34 fold roles"
        )
    expected_calibration_ids = tuple(
        object_id for row in fold_rows for object_id in row["calibration_ids"]
    )
    expected_calibration_folds = np.repeat(np.arange(5, dtype=np.int8), 14)
    expected_test_ids = tuple(
        object_id for row in fold_rows for object_id in row["test_ids"]
    )
    expected_test_folds = np.repeat(np.arange(5, dtype=np.int8), 34)
    expected_split_hash = hashlib.sha256(split_source.read_bytes()).hexdigest()

    calibration_source = Path(calibration_ensemble_path)
    try:
        with np.load(calibration_source, allow_pickle=False) as calibration:
            calibration_schema = calibration["schema"].item()
            calibration_seeds = np.asarray(calibration["seeds"], dtype=np.int64)
            calibration_ids = tuple(calibration["object_ids"].astype(str))
            calibration_folds = np.asarray(calibration["folds"], dtype=np.int8)
            calibration_errors = np.asarray(
                calibration["oracle_errors_deg"], dtype=np.float64
            )
            calibration_split_hash = calibration["splits_sha256"].item()
    except (OSError, ValueError, KeyError) as exc:
        raise K3PublicationRunError(
            f"cannot load ensemble calibration artifact: {exc}"
        ) from exc
    if (
        calibration_schema != "delphi.k3-real-calibration-ensemble.v1"
        or calibration_ids != expected_calibration_ids
        or not np.array_equal(calibration_folds, expected_calibration_folds)
        or calibration_errors.shape != (70,)
        or not np.all(np.isfinite(calibration_errors))
        or np.any(calibration_errors < 0)
        or np.any(calibration_errors > 90)
        or calibration_split_hash != expected_split_hash
    ):
        raise K3PublicationRunError(
            "ensemble calibration artifact differs from the frozen fold roles"
        )

    oof_source = Path(oof_ensemble_path)
    try:
        with np.load(oof_source, allow_pickle=False) as oof:
            oof_schema = oof["schema"].item()
            oof_seeds = np.asarray(oof["seeds"], dtype=np.int64)
            test_ids = tuple(oof["object_ids"].astype(str))
            test_folds = np.asarray(oof["folds"], dtype=np.int8)
            test_errors = np.asarray(oof["oracle_errors_deg"], dtype=np.float64)
    except (OSError, ValueError, KeyError) as exc:
        raise K3PublicationRunError(f"cannot load OOF ensemble artifact: {exc}") from exc
    if (
        oof_schema != "delphi.k3-real-oof-ensemble.v1"
        or not np.array_equal(oof_seeds, calibration_seeds)
        or test_ids != expected_test_ids
        or not np.array_equal(test_folds, expected_test_folds)
        or test_errors.shape != (170,)
        or not np.all(np.isfinite(test_errors))
        or np.any(test_errors < 0)
        or np.any(test_errors > 90)
    ):
        raise K3PublicationRunError(
            "OOF and calibration ensembles are not exactly seed/fold aligned"
        )

    fold_errors = calibration_errors.reshape(5, 14)
    cone90 = np.asarray(
        [conformal_radius(errors, 0.90) for errors in fold_errors], dtype=np.float64
    )
    cone95 = np.asarray(
        [conformal_radius(errors, 0.95) for errors in fold_errors], dtype=np.float64
    )
    if not np.all(cone95 == 90.0):
        raise K3PublicationRunError(
            "14-object ensemble calibration must yield 90-degree 95% radii"
        )
    object_cone90 = cone90[test_folds]
    object_cone95 = cone95[test_folds]
    covered90 = test_errors <= object_cone90
    covered95 = test_errors <= object_cone95

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                schema=np.asarray("delphi.k3-real-oof-ensemble-calibration.v1"),
                seeds=calibration_seeds,
                object_ids=np.asarray(test_ids),
                folds=test_folds,
                oracle_errors_deg=test_errors,
                object_cone90_deg=object_cone90,
                object_cone95_deg=object_cone95,
                covered90=covered90,
                covered95=covered95,
                fold_cone90_deg=cone90,
                fold_cone95_deg=cone95,
                calibration_object_ids=np.asarray(calibration_ids).reshape(5, 14),
                calibration_errors_deg=fold_errors,
                calibration_ensemble_sha256=np.asarray(
                    hashlib.sha256(calibration_source.read_bytes()).hexdigest()
                ),
                oof_ensemble_sha256=np.asarray(
                    hashlib.sha256(oof_source.read_bytes()).hexdigest()
                ),
                splits_sha256=np.asarray(expected_split_hash),
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "seed_count": float(calibration_seeds.size),
        "n_objects": float(test_errors.size),
        "cone90_min_deg": float(np.min(cone90)),
        "cone90_max_deg": float(np.max(cone90)),
        "cone95_min_deg": float(np.min(cone95)),
        "cone95_max_deg": float(np.max(cone95)),
        "empirical_coverage90": float(np.mean(covered90)),
        "empirical_coverage95": float(np.mean(covered95)),
        "artifact_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }


def run_real_oof_finetuning(
    pretrained_checkpoint: str | Path,
    synthetic_train_directory: str | Path,
    catalog_path: str | Path,
    dump_root: str | Path,
    split_path: str | Path,
    output_directory: str | Path,
    *,
    fold: int,
    seed: int,
    device: str = "cuda",
    resume: bool = False,
    run_provenance: Mapping[str, str] | None = None,
) -> K3FitResult:
    """Fine-tune one real OOF fold from the matching synthetic seed."""
    try:
        split = json.loads(Path(split_path).read_text(encoding="utf-8"))
        fold_row = next(row for row in split["folds"] if int(row["fold"]) == fold)
    except (OSError, json.JSONDecodeError, KeyError, StopIteration) as exc:
        raise K3PublicationRunError(f"cannot load requested OOF fold: {exc}") from exc
    real_train = load_real_examples(catalog_path, dump_root, fold_row["train_ids"])
    real_validation = load_real_examples(catalog_path, dump_root, fold_row["validation_ids"])
    synthetic = load_synthetic_subset(
        synthetic_train_directory, count=3 * len(real_train), seed=seed + 1000 * fold
    )
    stream = interleave_synthetic_and_real(synthetic, real_train, seed=seed + 1000 * fold)
    source = Path(pretrained_checkpoint)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if run_provenance is not None:
        source_provenance = checkpoint.get("run_provenance")
        if checkpoint.get("schema") != "delphi.k3-checkpoint.v2" or not isinstance(
            source_provenance, dict
        ):
            raise K3PublicationRunError(
                "publication fine-tuning requires a provenance-bound v2 pretraining checkpoint"
            )
        for key in ("protocol_sha256", "implementation_commit"):
            if source_provenance.get(key) != run_provenance.get(key):
                raise K3PublicationRunError(
                    "pretraining and real fine-tuning provenance must use one protocol/commit"
                )
    model = CandidateConditionedScorer(K3ScoreModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    config = K3TrainingConfig(seed=seed)
    bound_provenance = None if run_provenance is None else {
        **run_provenance,
        "pretrained_checkpoint_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"real-fold-{fold}-seed-{seed}.pt"
    history = destination / f"real-fold-{fold}-seed-{seed}.jsonl"
    if not resume and (target.exists() or history.exists()):
        raise K3PublicationRunError("refusing to overwrite an existing OOF run; use resume")
    if resume and not target.exists():
        raise K3PublicationRunError("OOF resume requested but checkpoint does not exist")

    def record(row: dict[str, float]) -> None:
        payload = {"schema": "delphi.k3-real-oof-epoch.v1", "fold": fold, "seed": seed, **row}
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
        print(json.dumps(payload, sort_keys=True), flush=True)

    return fit_k3(
        model,
        stream,
        real_validation,
        config=config,
        stage="real-oof",
        device=device,
        checkpoint_path=target,
        resume_checkpoint=target if resume else None,
        progress_callback=record,
        run_provenance=bound_provenance,
    )
