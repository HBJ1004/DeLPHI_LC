"""Deterministic synthetic pretraining and real 3:1 fine-tuning utilities."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

# CuBLAS reads this at process/CUDA initialization time.  Set it before
# importing/initializing Torch so deterministic GPU training does not fail at
# the first matrix operation.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch.utils.data import DataLoader, Dataset

from ..v2.preprocessing import KnownPeriod
from .config import K3TrainingConfig
from .losses import K3LossError, density_ratio_training_loss
from .model import CandidateConditionedScorer
from .protocol import K3_PROTOCOL_SHA256
from .synthetic import SyntheticRecord
from .tokenizer import K3TokenizedObject, pad_tokenized_objects, tokenize_epochs


class K3TrainingError(ValueError):
    """Raised when training data or checkpoint provenance is invalid."""


@dataclass(frozen=True)
class K3TrainingExample:
    object_id: str
    tokenized: K3TokenizedObject
    target_axes: np.ndarray

    def __post_init__(self) -> None:
        targets = np.asarray(self.target_axes, dtype=np.float32)
        if targets.ndim != 2 or targets.shape[1] != 3 or targets.shape[0] == 0 or not np.all(np.isfinite(targets)):
            raise K3TrainingError("each example needs at least one finite target axis")
        norms = np.linalg.norm(targets, axis=1)
        if np.any(norms <= 1e-12):
            raise K3TrainingError("target axes must be nonzero")
        targets = targets / norms[:, None]
        targets.setflags(write=False)
        object.__setattr__(self, "target_axes", targets)


def examples_from_synthetic_records(records: Sequence[SyntheticRecord]) -> tuple[K3TrainingExample, ...]:
    """Tokenize records with their supplied synthetic periods and axes."""
    examples = []
    for record in records:
        tokenized = tokenize_epochs(
            record.epochs, known_period=KnownPeriod(record.period_hours, "synthetic-known-period")
        )
        examples.append(
            K3TrainingExample(
                object_id=record.object_id,
                tokenized=tokenized,
                target_axes=np.asarray(record.target_axis, dtype=np.float32)[None, :],
            )
        )
    return tuple(examples)


def swap_input_examples(
    values: Sequence[K3TrainingExample], *, seed: int
) -> tuple[K3TrainingExample, ...]:
    """Pair each target with another object's complete deployable input."""
    examples = tuple(values)
    order = deterministic_derangement_indices(len(examples), seed=seed)
    return tuple(
        K3TrainingExample(
            object_id=target.object_id,
            tokenized=examples[int(donor_index)].tokenized,
            target_axes=target.target_axes,
        )
        for target, donor_index in zip(examples, order)
    )


def deterministic_derangement_indices(count: int, *, seed: int) -> np.ndarray:
    """Return a reproducible donor index for every object, with no fixed point."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise K3TrainingError("a deterministic derangement requires at least two objects")
    order = np.random.default_rng(seed).permutation(count)
    donors = np.empty(count, dtype=np.int64)
    for position, target_index in enumerate(order):
        donors[target_index] = order[(position + 1) % count]
    if np.any(donors == np.arange(count)) or len(np.unique(donors)) != count:
        raise AssertionError("internal derangement construction failed")
    donors.setflags(write=False)
    return donors


def shuffle_training_labels(
    values: Sequence[K3TrainingExample], *, seed: int
) -> tuple[K3TrainingExample, ...]:
    """Permute complete explicit axis sets across training objects only."""
    examples = tuple(values)
    order = deterministic_derangement_indices(len(examples), seed=seed)
    return tuple(
        K3TrainingExample(
            object_id=target.object_id,
            tokenized=target.tokenized,
            target_axes=examples[int(donor_index)].target_axes,
        )
        for target, donor_index in zip(examples, order)
    )


def interleave_synthetic_and_real(
    synthetic: Sequence[K3TrainingExample],
    real: Sequence[K3TrainingExample],
    *,
    seed: int,
    ratio: tuple[int, int] = (3, 1),
) -> tuple[K3TrainingExample, ...]:
    """Build the fixed 3:1 synthetic:real fine-tuning stream."""
    synthetic_values, real_values = tuple(synthetic), tuple(real)
    if not synthetic_values or not real_values or ratio != (3, 1):
        raise K3TrainingError("fine-tuning requires nonempty data and the locked 3:1 ratio")
    if len({example.object_id for example in (*synthetic_values, *real_values)}) != len(synthetic_values) + len(real_values):
        raise K3TrainingError("synthetic and real fine-tune object IDs must be disjoint")
    rng = np.random.default_rng(seed)
    required_synthetic = 3 * len(real_values)
    if len(synthetic_values) < required_synthetic:
        raise K3TrainingError("the locked 3:1 stream needs at least three synthetic examples per real object")
    synthetic_order = rng.permutation(len(synthetic_values))[:required_synthetic]
    real_order = rng.permutation(len(real_values))
    output: list[K3TrainingExample] = []
    synthetic_cursor = real_cursor = 0
    while real_cursor < len(real_order):
        for _ in range(3):
            output.append(synthetic_values[int(synthetic_order[synthetic_cursor])])
            synthetic_cursor += 1
        output.append(real_values[int(real_order[real_cursor])])
        real_cursor += 1
    return tuple(output)


class _ExampleDataset(Dataset[K3TrainingExample]):
    def __init__(self, values: Sequence[K3TrainingExample]) -> None:
        if not values:
            raise K3TrainingError("training/evaluation data must not be empty")
        self.values = tuple(values)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> K3TrainingExample:
        return self.values[index]


def collate_examples(values: Sequence[K3TrainingExample]) -> dict[str, object]:
    examples = tuple(values)
    arrays = pad_tokenized_objects(tuple(example.tokenized for example in examples))
    max_targets = max(example.target_axes.shape[0] for example in examples)
    targets = np.zeros((len(examples), max_targets, 3), dtype=np.float32)
    mask = np.zeros((len(examples), max_targets), dtype=np.bool_)
    for index, example in enumerate(examples):
        count = example.target_axes.shape[0]
        targets[index, :count] = example.target_axes
        mask[index, :count] = True
    return {
        "phase_features": torch.from_numpy(arrays["phase_features"]),
        "phase_mask": torch.from_numpy(arrays["phase_mask"]),
        "geometry_features": torch.from_numpy(arrays["geometry_features"]),
        "epoch_features": torch.from_numpy(arrays["epoch_features"]),
        "epoch_mask": torch.from_numpy(arrays["epoch_mask"]),
        "target_axes": torch.from_numpy(targets),
        "target_mask": torch.from_numpy(mask),
        "object_ids": tuple(example.object_id for example in examples),
    }


def configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _move_batch(batch: dict[str, object], device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    inputs = {
        key: value.to(device)
        for key, value in batch.items()
        if key in {"phase_features", "phase_mask", "geometry_features", "epoch_features", "epoch_mask"}
    }
    return inputs, batch["target_axes"].to(device), batch["target_mask"].to(device)


@dataclass(frozen=True)
class K3FitResult:
    stage: str
    best_epoch: int
    best_validation_loss: float
    epochs_completed: int
    history: tuple[dict[str, float], ...]
    checkpoint_sha256: str | None


def _configuration_sha256(config: K3TrainingConfig) -> str:
    return hashlib.sha256(json.dumps(config.as_mapping(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validated_run_provenance(
    value: Mapping[str, str] | None,
) -> dict[str, str] | None:
    if value is None:
        return None
    provenance = {str(key): str(item) for key, item in value.items()}
    if not {"protocol_sha256", "implementation_commit"}.issubset(provenance):
        raise K3TrainingError(
            "run provenance requires protocol_sha256 and implementation_commit"
        )
    if provenance["protocol_sha256"] != K3_PROTOCOL_SHA256:
        raise K3TrainingError("run provenance differs from the locked K3 protocol")
    commit = provenance["implementation_commit"]
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise K3TrainingError("run provenance requires a full lowercase Git commit")
    for key, item in provenance.items():
        if key.endswith("_sha256") and (
            len(item) != 64
            or any(character not in "0123456789abcdef" for character in item)
        ):
            raise K3TrainingError(f"run provenance field {key} is not a SHA-256 digest")
    return dict(sorted(provenance.items()))


def _save_checkpoint(
    path: Path,
    model: CandidateConditionedScorer,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    config: K3TrainingConfig,
    stage: str,
    epoch: int,
    best_loss: float,
    run_provenance: Mapping[str, str] | None,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(
            {
                "schema": "delphi.k3-checkpoint.v2",
                "stage": stage,
                "epoch": epoch,
                "best_validation_loss": best_loss,
                "configuration_sha256": _configuration_sha256(config),
                "config": config.as_mapping(),
                "model_config": model.config.as_mapping(),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "gradient_scaler_state_dict": scaler.state_dict(),
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_random_state": torch.random.get_rng_state(),
                "run_provenance": _validated_run_provenance(run_provenance),
            },
            temporary,
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def fit_k3(
    model: CandidateConditionedScorer,
    train_examples: Sequence[K3TrainingExample],
    validation_examples: Sequence[K3TrainingExample],
    *,
    config: K3TrainingConfig,
    stage: str = "synthetic",
    device: str | torch.device = "cpu",
    checkpoint_path: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    progress_callback: Callable[[dict[str, float]], None] | None = None,
    run_provenance: Mapping[str, str] | None = None,
) -> K3FitResult:
    """Fit one stage with deterministic loaders and optional atomic checkpoints."""
    if stage not in {"synthetic", "synthetic-label-shuffle", "real-oof"}:
        raise K3TrainingError("stage must be synthetic, synthetic-label-shuffle, or real-oof")
    normalized_provenance = _validated_run_provenance(run_provenance)
    configure_determinism(config.seed)
    device_value = torch.device(device)
    model.to(device_value)
    amp_enabled = bool(config.mixed_precision and device_value.type == "cuda")
    scaler = torch.amp.GradScaler(device_value.type, enabled=amp_enabled)
    if stage.startswith("synthetic"):
        maximum_epochs, patience_limit = config.synthetic_max_epochs, config.synthetic_patience
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        maximum_epochs, patience_limit = config.real_max_epochs, config.real_patience
        optimizer = torch.optim.AdamW(
            [
                {"params": tuple(model.encoder_parameters()), "lr": config.learning_rate * config.encoder_finetune_lr_multiplier},
                {"params": tuple(model.scorer_parameters()), "lr": config.learning_rate},
            ],
            weight_decay=config.weight_decay,
        )
    train_loader = DataLoader(_ExampleDataset(train_examples), batch_size=config.batch_size, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(config.seed), collate_fn=collate_examples)
    validation_loader = DataLoader(_ExampleDataset(validation_examples), batch_size=config.batch_size, shuffle=False, num_workers=0, collate_fn=collate_examples)
    best_loss, best_epoch, stale, start_epoch = float("inf"), 0, 0, 0
    if resume_checkpoint is not None:
        checkpoint = load_k3_checkpoint(
            resume_checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            stage=stage,
            run_provenance=normalized_provenance,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint["best_validation_loss"])
        best_epoch = int(checkpoint["epoch"])
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.random.set_rng_state(checkpoint["torch_random_state"])
    history: list[dict[str, float]] = []
    checkpoint_digest: str | None = None
    for epoch in range(start_epoch, maximum_epochs):
        model.train()
        train_losses: list[float] = []
        for batch_index, raw_batch in enumerate(train_loader):
            inputs, targets, target_mask = _move_batch(raw_batch, device_value)
            optimizer.zero_grad(set_to_none=True)
            generator = torch.Generator(device=device_value).manual_seed(config.seed + epoch * 1009 + batch_index)
            try:
                with torch.autocast(device_type=device_value.type, dtype=torch.float16, enabled=amp_enabled):
                    loss, _ = density_ratio_training_loss(model, inputs, targets, target_mask, config=config, generator=generator)
            except K3LossError as exc:
                raise K3TrainingError(str(exc)) from exc
            if not bool(torch.isfinite(loss)):
                raise K3TrainingError("non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        validation_losses: list[float] = []
        with torch.no_grad():
            for batch_index, raw_batch in enumerate(validation_loader):
                inputs, targets, target_mask = _move_batch(raw_batch, device_value)
                generator = torch.Generator(device=device_value).manual_seed(config.seed + 1_000_000 + epoch * 1009 + batch_index)
                with torch.autocast(device_type=device_value.type, dtype=torch.float16, enabled=amp_enabled):
                    loss, _ = density_ratio_training_loss(model, inputs, targets, target_mask, config=config, generator=generator)
                validation_losses.append(float(loss.detach().cpu()))
        train_loss, validation_loss = float(np.mean(train_losses)), float(np.mean(validation_losses))
        history.append({"epoch": float(epoch), "train_loss": train_loss, "validation_loss": validation_loss})
        if progress_callback is not None:
            progress_callback(dict(history[-1]))
        if validation_loss < best_loss:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            if checkpoint_path is not None:
                checkpoint_digest = _save_checkpoint(
                    Path(checkpoint_path),
                    model,
                    optimizer,
                    scaler,
                    config=config,
                    stage=stage,
                    epoch=epoch,
                    best_loss=best_loss,
                    run_provenance=normalized_provenance,
                )
        else:
            stale += 1
            if stale >= patience_limit:
                break
    return K3FitResult(stage, best_epoch, best_loss, len(history), tuple(history), checkpoint_digest)


def load_k3_checkpoint(
    path: str | Path,
    *,
    model: CandidateConditionedScorer,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None = None,
    config: K3TrainingConfig,
    stage: str,
    run_provenance: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Restore a checkpoint only when model, stage, and training config match."""
    try:
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    except (OSError, RuntimeError) as exc:
        raise K3TrainingError(f"cannot load K3 checkpoint: {exc}") from exc
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") not in {
        "delphi.k3-checkpoint.v1",
        "delphi.k3-checkpoint.v2",
    }:
        raise K3TrainingError("checkpoint schema mismatch")
    expected_provenance = _validated_run_provenance(run_provenance)
    if (
        expected_provenance is not None
        and checkpoint.get("run_provenance") != expected_provenance
    ):
        raise K3TrainingError("checkpoint run provenance differs from requested resume")
    if checkpoint.get("stage") != stage or checkpoint.get("configuration_sha256") != _configuration_sha256(config):
        raise K3TrainingError("checkpoint stage/configuration differs from requested resume")
    if checkpoint.get("model_config") != model.config.as_mapping():
        raise K3TrainingError("checkpoint model configuration differs from requested model")
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scaler is not None:
            scaler.load_state_dict(checkpoint.get("gradient_scaler_state_dict", {}))
    except (KeyError, RuntimeError, TypeError) as exc:
        raise K3TrainingError(f"checkpoint state cannot be restored: {exc}") from exc
    for key in ("epoch", "best_validation_loss", "python_random_state", "numpy_random_state", "torch_random_state"):
        if key not in checkpoint:
            raise K3TrainingError(f"checkpoint is missing {key}")
    return checkpoint
