"""Label-blind, parity-checked neural timing for the convergence study.

Only prediction-side fields are opened from the frozen OOF ensemble.  This
module has no catalog input and never loads a DAMIT spin file or reference
axis.  A timing artifact is written only after both measured executions for
all 170 held-out objects reproduce the frozen score landscape, mode indices,
refined axes, and refined scores within the declared numerical tolerances.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

import numpy as np
import torch

from ..v2.data import canonical_json, sha256_file
from ..v2.preprocessing import KnownPeriod, ObservationEpoch
from .bundle import SEEDS, K3EnsemblePredictor, _model_inputs
from .convergence_study import (
    TIMING_AXIS_COMPONENT_DIFFERENCE_MAXIMUM,
    TIMING_COLD_DEFINITION,
    TIMING_GRID_NORMALIZED_RMS_MAXIMUM,
    TIMING_PROVENANCE_SCHEMA,
    TIMING_REFINED_SCORE_DIFFERENCE_MAXIMUM,
    TIMING_SCHEMA,
    TIMING_WARM_DEFINITION,
    _load_blind_inputs,
    _read_spec,
    _split_rows,
    _study_section,
)
from .damit import _read_lc
from .downstream import DownstreamBenchmarkError, _atomic_json
from .evaluation import ensemble_score_grids, modes_from_score_grid
from .inference import refine_ensemble_axes, score_axial_grid

OOF_ENSEMBLE_SCHEMA = "delphi.k3-real-oof-ensemble.v1"
FIXED_PERIOD_PROVENANCE = "frozen-DAMIT-primary-solution-fixed-before-prediction"
_DIGEST_CHARACTERS = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class _FrozenOutputs:
    object_ids: tuple[str, ...]
    folds: np.ndarray
    score_grids: np.ndarray
    mode_indices: np.ndarray
    refined_axes: np.ndarray
    refined_scores: np.ndarray
    checkpoint_sha256: Mapping[str, str]


@dataclass(frozen=True)
class _InferenceOutputs:
    score_grid: np.ndarray
    mode_indices: tuple[int, int, int]
    refined_axes: np.ndarray
    refined_scores: np.ndarray


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and not (set(value) - _DIGEST_CHARACTERS)


def _load_frozen_outputs(
    path: str | Path,
    *,
    expected_ids: Sequence[str],
    expected_folds: Sequence[int],
) -> _FrozenOutputs:
    """Read prediction outputs and checkpoint identities, never label fields."""
    try:
        with np.load(path, allow_pickle=False) as artifact:
            # Deliberately enumerate only label-blind prediction/provenance keys.
            schema = str(artifact["schema"].item())
            seeds = tuple(int(value) for value in artifact["seeds"])
            object_ids = tuple(artifact["object_ids"].astype(str))
            folds = np.asarray(artifact["folds"], dtype=np.int64)
            score_grids = np.asarray(artifact["score_grids"], dtype=np.float64)
            mode_indices = np.asarray(artifact["deployed_mode_indices"], dtype=np.int64)
            refined_axes = np.asarray(artifact["refined_axes"], dtype=np.float64)
            refined_scores = np.asarray(artifact["refined_scores"], dtype=np.float64)
            checkpoint_names = tuple(artifact["checkpoint_names"].astype(str))
            checkpoint_hashes = tuple(artifact["checkpoint_sha256"].astype(str))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise DownstreamBenchmarkError(
            f"cannot read label-blind fields from the frozen OOF ensemble: {exc}"
        ) from exc

    count = len(expected_ids)
    expected_names = tuple(
        f"real-fold-{fold}-seed-{seed}.pt" for fold in range(5) for seed in SEEDS
    )
    if (
        schema != OOF_ENSEMBLE_SCHEMA
        or seeds != SEEDS
        or object_ids != tuple(expected_ids)
        or folds.shape != (count,)
        or not np.array_equal(folds, np.asarray(expected_folds, dtype=np.int64))
        or score_grids.shape != (count, 6144)
        or mode_indices.shape != (count, 3)
        or refined_axes.shape != (count, 3, 3)
        or refined_scores.shape != (count, 3)
        or not np.all(np.isfinite(score_grids))
        or not np.all(np.isfinite(refined_axes))
        or not np.all(np.isfinite(refined_scores))
        or np.any(mode_indices < 0)
        or np.any(mode_indices >= 6144)
        or checkpoint_names != expected_names
        or len(checkpoint_hashes) != len(expected_names)
        or any(not _is_sha256(value) for value in checkpoint_hashes)
    ):
        raise DownstreamBenchmarkError(
            "frozen ensemble prediction outputs or checkpoint identities are invalid"
        )
    norms = np.linalg.norm(refined_axes, axis=2)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-4):
        raise DownstreamBenchmarkError("frozen ensemble refined axes are not unit vectors")
    return _FrozenOutputs(
        object_ids=object_ids,
        folds=folds,
        score_grids=score_grids,
        mode_indices=mode_indices,
        refined_axes=refined_axes,
        refined_scores=refined_scores,
        checkpoint_sha256=dict(zip(checkpoint_names, checkpoint_hashes, strict=True)),
    )


def _bind_bundles(
    bundle_root: str | Path,
    *,
    split_rows: Sequence[Mapping[str, object]],
    split_sha256: str,
    checkpoint_sha256: Mapping[str, str],
) -> tuple[dict[str, object], ...]:
    """Hash-bind five safe bundles to the 25 archived source checkpoints."""
    root = Path(bundle_root).resolve()
    if not root.is_dir():
        raise DownstreamBenchmarkError("neural timing bundle root does not exist")
    bindings: list[dict[str, object]] = []
    for fold, split_row in enumerate(split_rows):
        directory = (root / f"k3-oof-fold-{fold}").resolve()
        if directory.parent != root or not directory.is_dir():
            raise DownstreamBenchmarkError(f"missing held-out fold bundle {fold}")
        manifest_path = directory / "bundle.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DownstreamBenchmarkError(
                f"cannot read held-out fold bundle {fold}: {exc}"
            ) from exc
        if not isinstance(manifest, Mapping):
            raise DownstreamBenchmarkError(f"held-out fold bundle {fold} manifest is not an object")
        roles = manifest.get("object_roles")
        role_names = ("train_ids", "validation_ids", "calibration_ids", "test_ids")
        members = manifest.get("members")
        model_config = manifest.get("model_config")
        if (
            manifest.get("schema") != "delphi.k3-ensemble-bundle.v1"
            or manifest.get("fold") != fold
            or manifest.get("splits_sha256") != split_sha256
            or not isinstance(manifest.get("bundle_id"), str)
            or not manifest["bundle_id"]
            or not isinstance(roles, Mapping)
            or any(roles.get(role) != split_row.get(role) for role in role_names)
            or not isinstance(model_config, Mapping)
            or not isinstance(members, list)
            or len(members) != len(SEEDS)
        ):
            raise DownstreamBenchmarkError(
                f"held-out fold bundle {fold} differs from the frozen split"
            )
        member_bindings: list[dict[str, object]] = []
        for seed, member in zip(SEEDS, members, strict=True):
            checkpoint_name = f"real-fold-{fold}-seed-{seed}.pt"
            if not isinstance(member, Mapping):
                raise DownstreamBenchmarkError(f"held-out fold bundle {fold} has malformed members")
            filename = member.get("file")
            safetensors_hash = member.get("sha256")
            source_hash = member.get("source_checkpoint_sha256")
            if (
                member.get("seed") != seed
                or not isinstance(filename, str)
                or not filename
                or PurePosixPath(filename).name != filename
                or not _is_sha256(safetensors_hash)
                or not _is_sha256(source_hash)
                or source_hash != checkpoint_sha256.get(checkpoint_name)
            ):
                raise DownstreamBenchmarkError(
                    f"fold {fold} seed {seed} is not bound to the frozen checkpoint"
                )
            weight_path = (directory / filename).resolve()
            if (
                weight_path.parent != directory
                or not weight_path.is_file()
                or sha256_file(weight_path) != safetensors_hash
            ):
                raise DownstreamBenchmarkError(
                    f"fold {fold} seed {seed} safe weight checksum differs"
                )
            member_bindings.append(
                {
                    "seed": seed,
                    "file": filename,
                    "safetensors_sha256": safetensors_hash,
                    "source_checkpoint_name": checkpoint_name,
                    "source_checkpoint_sha256": source_hash,
                }
            )
        bindings.append(
            {
                "fold": fold,
                "bundle_id": manifest["bundle_id"],
                "manifest_sha256": sha256_file(manifest_path),
                "model_config_sha256": hashlib.sha256(
                    canonical_json(model_config).encode("utf-8")
                ).hexdigest(),
                "members": member_bindings,
            }
        )
    return tuple(bindings)


def _resolve_device(requested: str) -> torch.device:
    if not isinstance(requested, str) or not requested.strip() or requested != requested.strip():
        raise DownstreamBenchmarkError("neural timing requires an explicit device")
    try:
        parsed = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise DownstreamBenchmarkError(f"invalid neural timing device: {exc}") from exc
    if parsed.type not in {"cpu", "cuda"}:
        raise DownstreamBenchmarkError("neural timing supports only explicit CPU or CUDA devices")
    if parsed.type == "cpu":
        if parsed.index is not None:
            raise DownstreamBenchmarkError("CPU timing device may not have an index")
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise DownstreamBenchmarkError("CUDA timing was requested but CUDA is unavailable")
    index = torch.cuda.current_device() if parsed.index is None else parsed.index
    if index < 0 or index >= torch.cuda.device_count():
        raise DownstreamBenchmarkError("CUDA timing device index is unavailable")
    return torch.device(f"cuda:{index}")


def _device_provenance(requested: str, resolved: torch.device) -> dict[str, object]:
    if resolved.type == "cuda":
        assert resolved.index is not None
        name = torch.cuda.get_device_name(resolved.index)
        cuda_version: str | None = torch.version.cuda
        cudnn_version: int | None = torch.backends.cudnn.version()
        index: int | None = resolved.index
    else:
        name = platform.processor() or platform.machine() or "cpu"
        cuda_version = None
        cudnn_version = None
        index = None
    return {
        "requested": requested,
        "resolved": str(resolved),
        "type": resolved.type,
        "index": index,
        "name": name,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": cuda_version,
        "cudnn_version": cudnn_version,
        "numpy_version": str(np.__version__),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_threads": int(torch.get_num_threads()),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_label_blind_epochs(
    dump_root: str | Path, row: Mapping[str, object]
) -> tuple[ObservationEpoch, ...]:
    object_id = str(row["object_id"])
    lightcurve = row["lightcurve"]
    assert isinstance(lightcurve, Mapping)
    relative = lightcurve["source_path"]
    expected_relative = PurePosixPath("files", object_id, "lc.txt").as_posix()
    if relative != expected_relative:
        raise DownstreamBenchmarkError(
            f"lightcurve path for {object_id} is not the canonical label-blind input"
        )
    root = Path(dump_root).resolve()
    source = (root / Path(relative)).resolve()
    expected_source = (root / "files" / object_id / "lc.txt").resolve()
    if source != expected_source or not source.is_file():
        raise DownstreamBenchmarkError(f"missing frozen lightcurve for {object_id}")
    if sha256_file(source) != lightcurve["source_sha256"]:
        raise DownstreamBenchmarkError(f"frozen lightcurve checksum differs for {object_id}")
    # `_read_lc` opens lc.txt only; unlike load_damit_object it never opens spin.txt.
    return _read_lc(source)


def _predict_full(
    predictor: K3EnsemblePredictor,
    epochs: Sequence[ObservationEpoch],
    *,
    known_period: KnownPeriod,
) -> _InferenceOutputs:
    if len(predictor.models) != len(SEEDS):
        raise DownstreamBenchmarkError("timed predictor does not contain five models")
    inputs = _model_inputs(epochs, known_period, predictor.device)
    model_maps = np.stack(
        [
            np.asarray(
                score_axial_grid(model, inputs, chunk_size=1024),
                dtype=np.float64,
            )
            for model in predictor.models
        ]
    )
    score_grid = ensemble_score_grids(model_maps[:, None, :])[0]
    modes = modes_from_score_grid(score_grid)
    mode_indices = tuple(int(mode.grid_index) for mode in modes)
    if len(mode_indices) != 3:
        raise DownstreamBenchmarkError("timed ensemble did not emit three modes")
    starts = np.asarray([mode.axis_xyz for mode in modes], dtype=np.float64)
    axes, scores = refine_ensemble_axes(predictor.models, inputs, starts)
    return _InferenceOutputs(
        score_grid=np.asarray(score_grid, dtype=np.float64),
        mode_indices=(mode_indices[0], mode_indices[1], mode_indices[2]),
        refined_axes=np.asarray(axes, dtype=np.float64),
        refined_scores=np.asarray(scores, dtype=np.float64),
    )


def _parity_result(
    actual: _InferenceOutputs,
    frozen: _FrozenOutputs,
    index: int,
) -> dict[str, object]:
    expected_grid = frozen.score_grids[index]
    if (
        actual.score_grid.shape != (6144,)
        or actual.refined_axes.shape != (3, 3)
        or actual.refined_scores.shape != (3,)
        or not np.all(np.isfinite(actual.score_grid))
        or not np.all(np.isfinite(actual.refined_axes))
        or not np.all(np.isfinite(actual.refined_scores))
    ):
        raise DownstreamBenchmarkError("timed inference produced malformed numerical output")
    centered_difference = (
        actual.score_grid
        - float(np.mean(actual.score_grid))
        - expected_grid
        + float(np.mean(expected_grid))
    )
    centered_rms = float(np.sqrt(np.mean(np.square(centered_difference))))
    normalized_rms = centered_rms / max(float(np.std(expected_grid)), 1.0)
    axis_difference = float(np.max(np.abs(actual.refined_axes - frozen.refined_axes[index])))
    score_difference = float(np.max(np.abs(actual.refined_scores - frozen.refined_scores[index])))
    mode_match = actual.mode_indices == tuple(int(value) for value in frozen.mode_indices[index])
    passed = (
        math.isfinite(normalized_rms)
        and normalized_rms <= TIMING_GRID_NORMALIZED_RMS_MAXIMUM
        and mode_match
        and axis_difference <= TIMING_AXIS_COMPONENT_DIFFERENCE_MAXIMUM
        and score_difference <= TIMING_REFINED_SCORE_DIFFERENCE_MAXIMUM
    )
    return {
        "grid_centered_normalized_rms": normalized_rms,
        "mode_indices_match": mode_match,
        "max_axis_component_difference": axis_difference,
        "max_refined_score_absolute_difference": score_difference,
        "passed": passed,
    }


def _require_parity(result: Mapping[str, object], object_id: str, condition: str) -> None:
    if result.get("passed") is not True:
        raise DownstreamBenchmarkError(
            f"{condition} neural output for {object_id} differs from the frozen ensemble"
        )


def _measurement_contract() -> dict[str, object]:
    return {
        "clock": "time.perf_counter",
        "warm_definition": TIMING_WARM_DEFINITION,
        "cold_definition": TIMING_COLD_DEFINITION,
        "warmup_runs_per_object": 1,
        "timed_runs_per_object": 1,
        "score_grid_centered_normalized_rms_maximum": (TIMING_GRID_NORMALIZED_RMS_MAXIMUM),
        "axis_component_difference_maximum": (TIMING_AXIS_COMPONENT_DIFFERENCE_MAXIMUM),
        "refined_score_absolute_difference_maximum": (TIMING_REFINED_SCORE_DIFFERENCE_MAXIMUM),
    }


def capture_neural_timing(
    spec_path: str | Path,
    spec_checksum_path: str | Path,
    split_path: str | Path,
    blind_inputs_path: str | Path,
    ensemble_path: str | Path,
    bundle_root: str | Path,
    dump_root: str | Path,
    output_path: str | Path,
    *,
    device: str,
) -> dict[str, object]:
    """Measure and freeze real held-out-fold inference without opening labels."""
    destination = Path(output_path)
    if destination.exists():
        raise DownstreamBenchmarkError(
            f"refusing to overwrite neural timing artifact {destination}"
        )
    spec, spec_hash = _read_spec(spec_path, spec_checksum_path)
    study = _study_section(spec)
    frozen_inputs = study["frozen_inputs"]
    split_hash = sha256_file(split_path)
    ensemble_hash = sha256_file(ensemble_path)
    blind_hash = sha256_file(blind_inputs_path)
    if (
        split_hash != frozen_inputs["publication_split_sha256"]
        or ensemble_hash != frozen_inputs["oof_ensemble_sha256"]
        or blind_hash != frozen_inputs["blind_execution_inputs_sha256"]
        or frozen_inputs.get("neural_timing_schema") != TIMING_SCHEMA
    ):
        raise DownstreamBenchmarkError("neural timing input differs from the frozen study")
    _, split_rows = _split_rows(split_path)
    object_ids = tuple(str(object_id) for row in split_rows for object_id in row["test_ids"])
    expected_folds = tuple(int(row["fold"]) for row in split_rows for _ in row["test_ids"])
    _, blind_lookup = _load_blind_inputs(
        blind_inputs_path,
        expected_hash=blind_hash,
        split_hash=split_hash,
        ensemble_hash=ensemble_hash,
        catalog_hash=str(frozen_inputs["reference_catalog_sha256"]),
    )
    if tuple(blind_lookup) != object_ids or any(
        int(blind_lookup[object_id]["fold"]) != expected_folds[index]
        for index, object_id in enumerate(object_ids)
    ):
        raise DownstreamBenchmarkError("label-blind inputs are not aligned to the held-out folds")
    frozen = _load_frozen_outputs(
        ensemble_path, expected_ids=object_ids, expected_folds=expected_folds
    )
    blind_axes = np.asarray(
        [blind_lookup[object_id]["guided_axes"] for object_id in object_ids],
        dtype=np.float64,
    )
    if not np.array_equal(blind_axes, frozen.refined_axes):
        raise DownstreamBenchmarkError(
            "label-blind guided axes differ from the frozen OOF predictions"
        )
    bundle_bindings = _bind_bundles(
        bundle_root,
        split_rows=split_rows,
        split_sha256=split_hash,
        checkpoint_sha256=frozen.checkpoint_sha256,
    )
    resolved_device = _resolve_device(device)

    warm_seconds = np.empty(170, dtype=np.float64)
    cold_seconds = np.empty(170, dtype=np.float64)
    cold_parity: dict[str, dict[str, object]] = {}
    warm_parity: dict[str, dict[str, object]] = {}
    offsets = np.cumsum([0] + [len(row["test_ids"]) for row in split_rows])
    for fold, split_row in enumerate(split_rows):
        fold_ids = tuple(str(value) for value in split_row["test_ids"])
        prepared: list[tuple[int, str, tuple[ObservationEpoch, ...], KnownPeriod]] = []
        for local_index, object_id in enumerate(fold_ids):
            index = int(offsets[fold]) + local_index
            row = blind_lookup[object_id]
            epochs = _load_label_blind_epochs(dump_root, row)
            known_period = KnownPeriod(float(row["period_hours"]), FIXED_PERIOD_PROVENANCE)
            prepared.append((index, object_id, epochs, known_period))

        bundle_directory = Path(bundle_root) / f"k3-oof-fold-{fold}"
        # Model-cold sensitivity: a fresh five-model bundle for every object.
        # Garbage collection and CUDA cache release happen outside the timer.
        for index, object_id, epochs, known_period in prepared:
            gc.collect()
            if resolved_device.type == "cuda":
                torch.cuda.empty_cache()
            _synchronize(resolved_device)
            started = time.perf_counter()
            cold_predictor = K3EnsemblePredictor(bundle_directory, device=str(resolved_device))
            cold_outputs = _predict_full(cold_predictor, epochs, known_period=known_period)
            _synchronize(resolved_device)
            elapsed = time.perf_counter() - started
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise DownstreamBenchmarkError(
                    f"cold neural timing is not positive for {object_id}"
                )
            cold_seconds[index] = elapsed
            parity = _parity_result(cold_outputs, frozen, index)
            _require_parity(parity, object_id, "cold")
            cold_parity[object_id] = parity
            del cold_predictor, cold_outputs

        # Primary warm timing: one persistent held-out-fold ensemble.  Each
        # object receives its own untimed warmup before its single timed run.
        warm_predictor = K3EnsemblePredictor(bundle_directory, device=str(resolved_device))
        for index, object_id, epochs, known_period in prepared:
            _predict_full(warm_predictor, epochs, known_period=known_period)
            _synchronize(resolved_device)
            started = time.perf_counter()
            warm_outputs = _predict_full(warm_predictor, epochs, known_period=known_period)
            _synchronize(resolved_device)
            elapsed = time.perf_counter() - started
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise DownstreamBenchmarkError(
                    f"warm neural timing is not positive for {object_id}"
                )
            warm_seconds[index] = elapsed
            parity = _parity_result(warm_outputs, frozen, index)
            _require_parity(parity, object_id, "warm")
            warm_parity[object_id] = parity
        del warm_predictor
        print(
            f"captured label-blind cold and warm neural timing for fold {fold}",
            file=sys.stderr,
            flush=True,
        )

    if (
        len(cold_parity) != 170
        or len(warm_parity) != 170
        or not np.all(np.isfinite(warm_seconds))
        or not np.all(np.isfinite(cold_seconds))
        or np.any(warm_seconds <= 0)
        or np.any(cold_seconds <= 0)
    ):
        raise DownstreamBenchmarkError(
            "neural timing did not produce two verified measurements for every object"
        )
    bundles_payload = list(bundle_bindings)
    provenance = {
        "schema": TIMING_PROVENANCE_SCHEMA,
        "source_spec_sha256": spec_hash,
        "source_split_sha256": split_hash,
        "source_blind_inputs_sha256": blind_hash,
        "source_ensemble_sha256": ensemble_hash,
        "object_count": 170,
        "seeds": list(SEEDS),
        "measurement": _measurement_contract(),
        "device": _device_provenance(device, resolved_device),
        "bundle_set_sha256": hashlib.sha256(
            canonical_json(bundles_payload).encode("utf-8")
        ).hexdigest(),
        "bundles": bundles_payload,
        "parity_rows": [
            {
                "object_id": object_id,
                "fold": expected_folds[index],
                "cold": cold_parity[object_id],
                "warm": warm_parity[object_id],
            }
            for index, object_id in enumerate(object_ids)
        ],
    }
    payload = {
        "schema": TIMING_SCHEMA,
        "source_ensemble_sha256": ensemble_hash,
        "object_ids": list(object_ids),
        "warm_wall_seconds": warm_seconds.tolist(),
        "cold_wall_seconds": cold_seconds.tolist(),
        "provenance": provenance,
    }
    artifact_hash = _atomic_json(destination, payload)
    return {
        "schema": TIMING_SCHEMA,
        "artifact_path": str(destination),
        "artifact_sha256": artifact_hash,
        "object_count": 170,
        "device": str(resolved_device),
        "summed_warm_wall_seconds": float(np.sum(warm_seconds)),
        "summed_cold_wall_seconds": float(np.sum(cold_seconds)),
        "bundle_set_sha256": provenance["bundle_set_sha256"],
    }
