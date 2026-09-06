"""Deterministic training and evaluation for the frozen axial comparison."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..physics.axial import (
    axial_oracle_training_loss,
    nearest_source_axis_errors_deg,
    oracle_source_axis_match,
)
from ..physics.directional import oracle_source_pole_match
from .baselines import (
    AmplitudeAxisObject,
    FeatureStandardizer,
    LegacyAxisObject,
    v1_training_loss,
)
from .comparison_contract import AXIAL_COMPARISON_SPEC_SHA256
from .data import PreparedObject, canonical_json, collate_prepared_objects
from .model import AxisModelOutput, residual_angle_regularization
from .preprocessing import GEOMETRY_FEATURE_SLICE, PHASE_FEATURE_SLICE
from .training import FROZEN_SEEDS, configure_determinism

AXIAL_TRAINING_STATE_SCHEMA = "delphi.axial-training-state.v1"
AxisModelKind = Literal["v1_faithful", "v1_corrected", "v2_plain", "v2_residual", "amplitude_mlp"]
AxisControl = Literal["primary", "geometry_disabled", "label_shuffle"]
AxisObject = PreparedObject | LegacyAxisObject | AmplitudeAxisObject


class AxialTrainingContractError(ValueError):
    """Raised when an axis run departs from its declared comparison arm."""


@dataclass(frozen=True)
class AxialTrainingConfig:
    model_kind: AxisModelKind
    seed: int
    batch_size: int
    gradient_accumulation: int
    max_epochs: int = 150
    patience: int = 25
    warmup_epochs: int = 5
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    mixed_precision: bool = True
    num_workers: int = 0
    control: AxisControl = "primary"
    comparison_spec_sha256: str = AXIAL_COMPARISON_SPEC_SHA256

    def __post_init__(self) -> None:
        if self.model_kind not in {
            "v1_faithful",
            "v1_corrected",
            "v2_plain",
            "v2_residual",
            "amplitude_mlp",
        }:
            raise AxialTrainingContractError("unsupported comparison model kind")
        if self.seed not in FROZEN_SEEDS:
            raise AxialTrainingContractError(f"seed must be one of {FROZEN_SEEDS}")
        integers = (
            self.batch_size,
            self.gradient_accumulation,
            self.max_epochs,
            self.patience,
            self.warmup_epochs,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in integers
        ):
            raise AxialTrainingContractError("batch and epoch values must be positive integers")
        if self.warmup_epochs >= self.max_epochs:
            raise AxialTrainingContractError("warmup_epochs must be below max_epochs")
        if self.batch_size * self.gradient_accumulation != 16:
            raise AxialTrainingContractError("the comparison fixes effective batch size at 16")
        positive = (self.learning_rate, self.gradient_clip_norm)
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise AxialTrainingContractError("training scales must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise AxialTrainingContractError("weight_decay must be finite and nonnegative")
        if self.num_workers != 0:
            raise AxialTrainingContractError("comparison loading fixes num_workers=0")
        if self.control not in {"primary", "geometry_disabled", "label_shuffle"}:
            raise AxialTrainingContractError("unsupported training control")
        if self.control == "geometry_disabled" and not self.model_kind.startswith("v2_"):
            raise AxialTrainingContractError("geometry-disabled training applies only to V2")
        if self.comparison_spec_sha256 != AXIAL_COMPARISON_SPEC_SHA256:
            raise AxialTrainingContractError("comparison spec hash mismatch")

    @classmethod
    def frozen(
        cls,
        model_kind: AxisModelKind,
        seed: int,
        *,
        control: AxisControl = "primary",
    ) -> "AxialTrainingConfig":
        if model_kind.startswith("v2_"):
            return cls(model_kind, seed, batch_size=2, gradient_accumulation=8, control=control)
        return cls(model_kind, seed, batch_size=16, gradient_accumulation=1, control=control)


@dataclass(frozen=True)
class AxialEpochMetrics:
    epoch: int
    learning_rate: float
    train_loss: float
    validation_mean_axis_oracle_at3_deg: float


@dataclass(frozen=True)
class AxialFitResult:
    best_epoch: int
    best_validation_mean_axis_oracle_at3_deg: float
    epochs_completed: int
    history: tuple[AxialEpochMetrics, ...]
    stopped_early: bool


@dataclass(frozen=True)
class AxisPrediction:
    object_id: str
    axes: tuple[tuple[float, float, float], ...]
    target_vectors: tuple[tuple[float, float, float], ...]
    period_hours_targets: tuple[float, ...]
    axis_oracle_at3_error_deg: float
    axis_top1_error_deg: float
    directed_oracle_at3_error_deg: float
    # A zero/non-finite candidate is retained as an explicit failed output,
    # never normalized into an arbitrary pole or silently dropped.
    axis_valid: tuple[bool, ...] = ()


class _AxisSequence(Dataset[AxisObject]):
    def __init__(self, values: Sequence[AxisObject]) -> None:
        if not values:
            raise AxialTrainingContractError("training/evaluation partition must not be empty")
        self.values = tuple(values)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> AxisObject:
        return self.values[index]


def deterministic_derangement_indices(size: int, seed: int) -> np.ndarray:
    """Return a deterministic true derangement for every size greater than one."""
    if isinstance(size, bool) or not isinstance(size, int) or size < 2:
        raise AxialTrainingContractError("a derangement requires at least two objects")
    order = np.random.default_rng(seed).permutation(size)
    donors = np.empty(size, dtype=np.int64)
    for position, target_index in enumerate(order):
        donors[target_index] = order[(position + 1) % size]
    if np.any(donors == np.arange(size)):
        raise AssertionError("internal derangement construction failed")
    return donors


def shuffled_axis_labels(values: Sequence[AxisObject], seed: int) -> tuple[AxisObject, ...]:
    """Move complete solution sets among training objects without fixed points."""
    donors = deterministic_derangement_indices(len(values), seed)
    result: list[AxisObject] = []
    for index, donor_index in enumerate(donors):
        target = values[index]
        donor = values[int(donor_index)]
        result.append(
            replace(
                target,
                target_vectors=np.asarray(donor.target_vectors).copy(),
                period_hours_targets=np.asarray(donor.period_hours_targets).copy(),
                **(
                    {"target_pixels": np.asarray(donor.target_pixels).copy()}
                    if isinstance(target, PreparedObject)
                    else {}
                ),
            )
        )
    return tuple(result)


def zero_axis_inputs(values: Sequence[AxisObject]) -> tuple[AxisObject, ...]:
    """Zero deployable values while retaining only padding structure and labels."""
    result: list[AxisObject] = []
    for value in values:
        if isinstance(value, PreparedObject):
            result.append(
                replace(
                    value,
                    tokens=np.zeros_like(value.tokens),
                    epoch_descriptors=np.zeros_like(value.epoch_descriptors),
                    period_values=np.zeros_like(value.period_values),
                    period_mask=np.zeros_like(value.period_mask),
                )
            )
        elif isinstance(value, LegacyAxisObject):
            result.append(replace(value, tokens=np.zeros_like(value.tokens)))
        else:
            result.append(replace(value, features=np.zeros_like(value.features)))
    return tuple(result)


def ablate_v2_inputs(
    values: Sequence[PreparedObject], channel: Literal["brightness", "time", "geometry", "period"]
) -> tuple[PreparedObject, ...]:
    """Apply one inference-time V2 channel ablation without changing masks."""
    if channel not in {"brightness", "time", "geometry", "period"}:
        raise AxialTrainingContractError("unsupported V2 ablation channel")
    result: list[PreparedObject] = []
    for value in values:
        tokens = value.tokens.copy()
        descriptors = value.epoch_descriptors.copy()
        period_values = value.period_values.copy()
        period_mask = value.period_mask.copy()
        if channel == "brightness":
            tokens[..., 2] = 0.0
            descriptors[..., 0] = 0.0
        elif channel == "time":
            tokens[..., 0:2] = 0.0
            descriptors[..., 1] = 0.0
        elif channel == "geometry":
            tokens[..., GEOMETRY_FEATURE_SLICE] = 0.0
        else:
            tokens[..., PHASE_FEATURE_SLICE] = 0.0
            period_values[:] = 0.0
            period_mask[:] = False
        result.append(
            replace(
                value,
                tokens=tokens,
                epoch_descriptors=descriptors,
                period_values=period_values,
                period_mask=period_mask,
            )
        )
    return tuple(result)


def _collate_targets(values: Sequence[AxisObject]) -> dict[str, object]:
    maximum = max(value.target_vectors.shape[0] for value in values)
    targets = np.zeros((len(values), maximum, 3), dtype=np.float64)
    target_mask = np.zeros((len(values), maximum), dtype=bool)
    period_targets = np.zeros((len(values), maximum), dtype=np.float64)
    for index, value in enumerate(values):
        count = value.target_vectors.shape[0]
        targets[index, :count] = value.target_vectors
        target_mask[index, :count] = True
        period_targets[index, :count] = value.period_hours_targets
    return {
        "object_ids": tuple(value.object_id for value in values),
        "target_vectors": torch.from_numpy(targets),
        "target_mask": torch.from_numpy(target_mask),
        "period_hours_targets": torch.from_numpy(period_targets),
    }


def _collate_legacy(values: Sequence[LegacyAxisObject]) -> dict[str, object]:
    for value in values:
        value.validate()
    maximum = max(value.tokens.shape[1] for value in values)
    tokens = np.zeros((len(values), 8, maximum, 13), dtype=np.float32)
    mask = np.zeros((len(values), 8, maximum), dtype=np.float32)
    for index, value in enumerate(values):
        length = value.tokens.shape[1]
        tokens[index, :, :length] = value.tokens
        mask[index, :, :length] = value.mask
    return {
        **_collate_targets(values),
        "tokens": torch.from_numpy(tokens),
        "mask": torch.from_numpy(mask),
    }


def _collate_amplitude(
    values: Sequence[AmplitudeAxisObject], standardizer: FeatureStandardizer
) -> dict[str, object]:
    return {
        **_collate_targets(values),
        "features": torch.from_numpy(standardizer.transform(values)),
    }


def _make_loader(
    values: Sequence[AxisObject],
    config: AxialTrainingConfig,
    *,
    epoch: int,
    shuffle: bool,
    standardizer: FeatureStandardizer | None,
) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed + epoch * 1_000_003)

    def collate(batch: Sequence[AxisObject]) -> dict[str, object]:
        if config.model_kind.startswith("v2_"):
            if not all(isinstance(item, PreparedObject) for item in batch):
                raise AxialTrainingContractError("V2 model requires PreparedObject inputs")
            return collate_prepared_objects(batch)  # type: ignore[arg-type]
        if config.model_kind.startswith("v1_"):
            if not all(isinstance(item, LegacyAxisObject) for item in batch):
                raise AxialTrainingContractError("V1 model requires LegacyAxisObject inputs")
            return _collate_legacy(batch)  # type: ignore[arg-type]
        if standardizer is None:
            raise AxialTrainingContractError("amplitude MLP requires a train-only standardizer")
        if not all(isinstance(item, AmplitudeAxisObject) for item in batch):
            raise AxialTrainingContractError("amplitude MLP requires AmplitudeAxisObject inputs")
        return _collate_amplitude(batch, standardizer)  # type: ignore[arg-type]

    return DataLoader(
        _AxisSequence(values),
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        collate_fn=collate,
        drop_last=False,
    )


def _move_batch(
    batch: Mapping[str, object], device: torch.device, config: AxialTrainingConfig
) -> dict[str, object]:
    moved = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    if config.control == "geometry_disabled":
        tokens = moved["tokens"].clone()
        tokens[..., GEOMETRY_FEATURE_SLICE] = 0.0
        moved["tokens"] = tokens
    return moved


def _forward_axes(
    model: nn.Module, batch: Mapping[str, object], model_kind: AxisModelKind
) -> AxisModelOutput:
    if model_kind.startswith("v2_"):
        output = model(
            batch["tokens"].float(),
            batch["observation_mask"].bool(),
            batch["epoch_descriptors"].float(),
            batch["epoch_mask"].bool(),
            batch["period_values"].float(),
            batch["period_mask"].bool(),
        )
        if not isinstance(output, AxisModelOutput):
            raise AxialTrainingContractError("V2 axis model returned an incompatible output")
        return output
    if model_kind.startswith("v1_"):
        poles, _ = model(batch["tokens"].float(), batch["mask"].float())
        return AxisModelOutput(axes=poles)
    return AxisModelOutput(axes=model(batch["features"].float()))


def _training_loss(
    model: nn.Module, batch: Mapping[str, object], config: AxialTrainingConfig
) -> tuple[torch.Tensor, dict[str, float]]:
    output = _forward_axes(model, batch, config.model_kind)
    targets = batch["target_vectors"].to(dtype=output.axes.dtype)
    target_mask = batch["target_mask"].bool()
    if config.model_kind.startswith("v1_"):
        mode = "faithful" if config.model_kind == "v1_faithful" else "corrected"
        total, components = v1_training_loss(output.axes, targets, target_mask, mode=mode)
    else:
        total, components = axial_oracle_training_loss(output.axes, targets, target_mask)
        if config.model_kind == "v2_residual":
            regularization = residual_angle_regularization(output)
            weight = float(model.config.residual_regularization_weight)
            total = total + weight * regularization
            components["residual_angle_regularization"] = regularization
    return total, {name: float(value.detach()) for name, value in components.items()}


def _learning_rate(config: AxialTrainingConfig, epoch: int) -> float:
    if epoch < config.warmup_epochs:
        return config.learning_rate * (epoch + 1) / config.warmup_epochs
    progress = (epoch - config.warmup_epochs) / max(config.max_epochs - config.warmup_epochs - 1, 1)
    return config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def _prediction_from_arrays(
    object_id: str,
    axes: np.ndarray,
    target_vectors: np.ndarray,
    period_targets: np.ndarray,
) -> AxisPrediction:
    candidates = np.asarray(axes[:3], dtype=np.float64)
    valid = np.all(np.isfinite(candidates), axis=1) & (
        np.linalg.norm(np.where(np.isfinite(candidates), candidates, 0.0), axis=1) > 1e-12
    )
    stored = np.where(np.isfinite(candidates), candidates, 0.0)
    usable = stored[valid]
    if usable.size:
        axial = oracle_source_axis_match(usable, target_vectors)
        directed = oracle_source_pole_match(usable, target_vectors)
        top1 = nearest_source_axis_errors_deg(usable[:1], target_vectors)[0]
        axial_error, directed_error = float(axial.error_deg), float(directed.error_deg)
    else:
        # The maximum possible axial separation is 90 degrees; the directed
        # diagnostic maximum is 180.  This pessimistic score cannot improve a
        # model through numerical collapse and keeps every object in the table.
        axial_error, top1, directed_error = 90.0, 90.0, 180.0
    return AxisPrediction(
        object_id=object_id,
        axes=tuple(tuple(float(component) for component in row) for row in stored),
        target_vectors=tuple(
            tuple(float(component) for component in row) for row in target_vectors
        ),
        period_hours_targets=tuple(float(value) for value in period_targets),
        axis_oracle_at3_error_deg=axial_error,
        axis_top1_error_deg=float(top1),
        directed_oracle_at3_error_deg=directed_error,
        axis_valid=tuple(bool(value) for value in valid),
    )


def predict_axes(
    model: nn.Module,
    values: Sequence[AxisObject],
    config: AxialTrainingConfig,
    *,
    standardizer: FeatureStandardizer | None = None,
    device: str | torch.device = "cpu",
) -> tuple[AxisPrediction, ...]:
    selected_device = torch.device(device)
    model.to(selected_device).eval()
    predictions: list[AxisPrediction] = []
    with torch.no_grad():
        loader = _make_loader(values, config, epoch=0, shuffle=False, standardizer=standardizer)
        for raw_batch in loader:
            batch = _move_batch(raw_batch, selected_device, config)
            output = _forward_axes(model, batch, config.model_kind)
            axes = output.axes.detach().to(dtype=torch.float64).cpu().numpy()
            targets = batch["target_vectors"].to(dtype=torch.float64).cpu().numpy()
            masks = batch["target_mask"].cpu().numpy().astype(bool)
            periods = batch["period_hours_targets"].to(dtype=torch.float64).cpu().numpy()
            for index, object_id in enumerate(batch["object_ids"]):
                predictions.append(
                    _prediction_from_arrays(
                        object_id,
                        axes[index],
                        targets[index, masks[index]],
                        periods[index, masks[index]],
                    )
                )
    if len({item.object_id for item in predictions}) != len(predictions):
        raise AxialTrainingContractError("prediction partition contains duplicate object IDs")
    return tuple(predictions)


def exhaustive_deranged_axis_errors(
    predictions: Sequence[AxisPrediction],
) -> dict[str, float]:
    """Average each target's error under every *other* object's input output."""
    if len(predictions) < 2:
        raise AxialTrainingContractError("exhaustive derangement requires at least two objects")
    identifiers = [item.object_id for item in predictions]
    if len(identifiers) != len(set(identifiers)):
        raise AxialTrainingContractError("exhaustive derangement requires unique object IDs")
    result: dict[str, float] = {}
    for target in predictions:
        target_vectors = np.asarray(target.target_vectors, dtype=np.float64)
        errors = []
        for donor in predictions:
            if donor.object_id == target.object_id:
                continue
            donor_axes = np.asarray(donor.axes, dtype=np.float64)
            mask = (
                np.asarray(donor.axis_valid, dtype=bool)
                if donor.axis_valid
                else np.all(np.isfinite(donor_axes), axis=1)
                & (np.linalg.norm(donor_axes, axis=1) > 1e-12)
            )
            errors.append(
                float(oracle_source_axis_match(donor_axes[mask], target_vectors).error_deg)
                if bool(np.any(mask))
                else 90.0
            )
        result[target.object_id] = float(np.mean(errors))
    return result


def _validation_mean_error(
    model: nn.Module,
    values: Sequence[AxisObject],
    config: AxialTrainingConfig,
    standardizer: FeatureStandardizer | None,
    device: torch.device,
) -> float:
    predictions = predict_axes(model, values, config, standardizer=standardizer, device=device)
    return float(np.mean([item.axis_oracle_at3_error_deg for item in predictions]))


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _config_sha256(config: AxialTrainingConfig) -> str:
    return hashlib.sha256(canonical_json(asdict(config)).encode("utf-8")).hexdigest()


def _checkpoint_tensors(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, torch.Tensor]:
    tensors = {
        f"model::{name}": value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    for parameter, state in optimizer.state.items():
        name = parameter_names[id(parameter)]
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                tensors[f"optimizer::{name}::{key}"] = value.detach().cpu().contiguous()
    tensors["rng::torch_cpu"] = torch.get_rng_state().cpu()
    if torch.cuda.is_available():
        for index, state in enumerate(torch.cuda.get_rng_state_all()):
            tensors[f"rng::torch_cuda::{index}"] = state.cpu()
    return tensors


def _save_training_state(
    directory: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: AxialTrainingConfig,
    *,
    epoch: int,
    best_epoch: int,
    best_validation: float,
    stale_epochs: int,
    history: Sequence[AxialEpochMetrics],
) -> None:
    tensor_path = directory / "last-state.safetensors"
    temporary_tensor = directory / ".last-state.safetensors.tmp"
    save_file(_checkpoint_tensors(model, optimizer), str(temporary_tensor))
    os.replace(temporary_tensor, tensor_path)
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    optimizer_scalars: dict[str, dict[str, float | int | bool]] = {}
    for parameter, state in optimizer.state.items():
        scalar_state = {
            key: value for key, value in state.items() if isinstance(value, (float, int, bool))
        }
        if scalar_state:
            optimizer_scalars[parameter_names[id(parameter)]] = scalar_state
    _atomic_json(
        directory / "last-state.json",
        {
            "schema": AXIAL_TRAINING_STATE_SCHEMA,
            "configuration_sha256": _config_sha256(config),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_validation": best_validation,
            "stale_epochs": stale_epochs,
            "history": [asdict(item) for item in history],
            "optimizer_scalars": optimizer_scalars,
            "scaler": scaler.state_dict(),
        },
    )


def _load_training_state(
    directory: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: AxialTrainingConfig,
    device: torch.device,
) -> tuple[int, int, float, int, list[AxialEpochMetrics]]:
    try:
        metadata = json.loads((directory / "last-state.json").read_text(encoding="utf-8"))
        tensors = load_file(str(directory / "last-state.safetensors"), device=str(device))
    except (OSError, json.JSONDecodeError, RuntimeError) as exc:
        raise AxialTrainingContractError(f"cannot load axial training state: {exc}") from exc
    if metadata.get("schema") != AXIAL_TRAINING_STATE_SCHEMA:
        raise AxialTrainingContractError("axial training state schema mismatch")
    if metadata.get("configuration_sha256") != _config_sha256(config):
        raise AxialTrainingContractError("axial training state configuration mismatch")
    model_state = {
        key.removeprefix("model::"): value
        for key, value in tensors.items()
        if key.startswith("model::")
    }
    model.load_state_dict(model_state, strict=True)
    named_parameters = dict(model.named_parameters())
    for name, parameter in named_parameters.items():
        prefix = f"optimizer::{name}::"
        tensor_state = {
            key.removeprefix(prefix): value.to(device)
            for key, value in tensors.items()
            if key.startswith(prefix)
        }
        scalar_state = metadata.get("optimizer_scalars", {}).get(name, {})
        if tensor_state or scalar_state:
            optimizer.state[parameter] = {**tensor_state, **scalar_state}
    torch.set_rng_state(tensors["rng::torch_cpu"].cpu())
    if device.type == "cuda":
        cuda_states = [
            tensors[key].cpu() for key in sorted(tensors) if key.startswith("rng::torch_cuda::")
        ]
        if cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)
    scaler.load_state_dict(metadata.get("scaler", {}))
    return (
        int(metadata["epoch"]) + 1,
        int(metadata["best_epoch"]),
        float(metadata["best_validation"]),
        int(metadata["stale_epochs"]),
        [AxialEpochMetrics(**item) for item in metadata["history"]],
    )


def fit_axis_model(
    model: nn.Module,
    train_values: Sequence[AxisObject],
    validation_values: Sequence[AxisObject],
    config: AxialTrainingConfig,
    output_directory: str | Path,
    *,
    standardizer: FeatureStandardizer | None = None,
    device: str | torch.device = "cpu",
    resume: bool = False,
    stop_after_epoch: int | None = None,
) -> AxialFitResult:
    """Fit one declared arm, selecting checkpoints only by validation mean error."""
    configure_determinism(config.seed)
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise AxialTrainingContractError("CUDA was requested but is unavailable")
    if stop_after_epoch is not None and not 0 <= stop_after_epoch < config.max_epochs:
        raise AxialTrainingContractError("stop_after_epoch is outside the training schedule")
    if config.model_kind == "amplitude_mlp" and standardizer is None:
        raise AxialTrainingContractError("amplitude MLP requires a train-only standardizer")
    if config.model_kind.startswith("v2_"):
        geometry_required = bool(model.config.geometry_required)
        if geometry_required == (config.control == "geometry_disabled"):
            raise AxialTrainingContractError("V2 model geometry contract mismatches its control")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    training_values = (
        shuffled_axis_labels(train_values, config.seed)
        if config.control == "label_shuffle"
        else tuple(train_values)
    )
    model.to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    amp_enabled = config.mixed_precision and selected_device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch, best_epoch, stale_epochs = 0, -1, 0
    best_validation = math.inf
    history: list[AxialEpochMetrics] = []
    if resume:
        start_epoch, best_epoch, best_validation, stale_epochs, history = _load_training_state(
            output, model, optimizer, scaler, config, selected_device
        )
    for epoch in range(start_epoch, config.max_epochs):
        learning_rate = _learning_rate(config, epoch)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_total, train_count = 0.0, 0
        loader = _make_loader(
            training_values,
            config,
            epoch=epoch,
            shuffle=True,
            standardizer=standardizer,
        )
        for batch_index, raw_batch in enumerate(loader):
            batch = _move_batch(raw_batch, selected_device, config)
            group_start = (
                batch_index // config.gradient_accumulation
            ) * config.gradient_accumulation
            group_size = min(config.gradient_accumulation, len(loader) - group_start)
            with torch.autocast(
                device_type=selected_device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                loss, _ = _training_loss(model, batch, config)
                scaled_loss = loss / group_size
            scaler.scale(scaled_loss).backward()
            final_batch = batch_index + 1 == len(loader)
            if (batch_index + 1) % config.gradient_accumulation == 0 or final_batch:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            size = len(batch["object_ids"])
            train_total += float(loss.detach()) * size
            train_count += size
        validation = _validation_mean_error(
            model, validation_values, config, standardizer, selected_device
        )
        metrics = AxialEpochMetrics(epoch, learning_rate, train_total / train_count, validation)
        history.append(metrics)
        if validation < best_validation:
            best_validation = validation
            best_epoch = epoch
            stale_epochs = 0
            save_file(
                {
                    name: tensor.detach().cpu().contiguous()
                    for name, tensor in model.state_dict().items()
                },
                str(output / "best-model.safetensors"),
            )
        else:
            stale_epochs += 1
        _save_training_state(
            output,
            model,
            optimizer,
            scaler,
            config,
            epoch=epoch,
            best_epoch=best_epoch,
            best_validation=best_validation,
            stale_epochs=stale_epochs,
            history=history,
        )
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            break
        if stale_epochs >= config.patience:
            break
    model.load_state_dict(
        load_file(str(output / "best-model.safetensors"), device=str(selected_device)),
        strict=True,
    )
    result = AxialFitResult(
        best_epoch=best_epoch,
        best_validation_mean_axis_oracle_at3_deg=best_validation,
        epochs_completed=len(history),
        history=tuple(history),
        stopped_early=len(history) < config.max_epochs,
    )
    _atomic_json(
        output / "training-history.json",
        {**asdict(result), "history": [asdict(item) for item in history]},
    )
    return result


__all__ = [
    "AXIAL_TRAINING_STATE_SCHEMA",
    "AxisPrediction",
    "AxialEpochMetrics",
    "AxialFitResult",
    "AxialTrainingConfig",
    "AxialTrainingContractError",
    "ablate_v2_inputs",
    "deterministic_derangement_indices",
    "exhaustive_deranged_axis_errors",
    "fit_axis_model",
    "predict_axes",
    "shuffled_axis_labels",
    "zero_axis_inputs",
]
