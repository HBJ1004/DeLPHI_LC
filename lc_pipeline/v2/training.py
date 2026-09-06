"""Deterministic object-balanced training and raw OOF prediction for V2."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

# CUDA 10.2+ requires this process-level setting for deterministic cuBLAS
# matrix products.  It must be present before the first CUDA operation, not
# merely before ``torch.use_deterministic_algorithms(True)`` is called.
# Preserve an explicitly supplied compatible workspace choice for operators.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ..physics.directional import directed_angular_error_deg, torch_directed_angular_error_deg
from ..torch_compat import make_grad_scaler
from .data import PreparedObject, collate_prepared_objects
from .density import pixel_centers_xyz, vector_to_pixel
from .model import (
    GeometryHierarchicalDensityModel,
    V2ModelConfig,
    hierarchical_multi_solution_density_nll,
)
from .preprocessing import GEOMETRY_FEATURE_SLICE

FROZEN_SEEDS = (17, 42, 137, 777, 2027)
TRAINING_STATE_SCHEMA = "delphi.training-state.v2"


class TrainingContractError(ValueError):
    """Raised when a run could violate deterministic scientific evaluation."""


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    batch_size: int = 4
    gradient_accumulation: int = 4
    max_epochs: int = 200
    patience: int = 25
    warmup_epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    risk_loss_weight: float = 0.1
    risk_huber_delta_deg: float = 10.0
    risk_scale_deg: float = 30.0
    num_workers: int = 0
    mixed_precision: bool = True
    control: str = "primary"
    longitude_rotation_augmentation: bool = False

    def __post_init__(self) -> None:
        if self.seed not in FROZEN_SEEDS:
            raise TrainingContractError(f"seed must be one of {FROZEN_SEEDS}")
        integer_fields = (
            self.batch_size,
            self.gradient_accumulation,
            self.max_epochs,
            self.patience,
            self.warmup_epochs,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in integer_fields
        ):
            raise TrainingContractError(
                "batch, accumulation, epoch, patience, and warmup values must be positive integers"
            )
        if self.warmup_epochs >= self.max_epochs:
            raise TrainingContractError("warmup_epochs must be smaller than max_epochs")
        positive = (
            self.learning_rate,
            self.gradient_clip_norm,
            self.risk_loss_weight,
            self.risk_huber_delta_deg,
            self.risk_scale_deg,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise TrainingContractError("training scales must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise TrainingContractError("weight_decay must be finite and nonnegative")
        if self.num_workers != 0:
            raise TrainingContractError(
                "V2 release training fixes num_workers=0 for deterministic loading"
            )
        if self.control not in {"primary", "geometry_disabled", "label_shuffle"}:
            raise TrainingContractError("unsupported training control")
        if not isinstance(self.longitude_rotation_augmentation, bool):
            raise TrainingContractError("longitude_rotation_augmentation must be boolean")


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    learning_rate: float
    train_loss: float
    validation_loss: float


@dataclass(frozen=True)
class FitResult:
    best_epoch: int
    best_validation_loss: float
    epochs_completed: int
    history: tuple[EpochMetrics, ...]
    stopped_early: bool


@dataclass(frozen=True)
class RawPrediction:
    object_id: str
    logits: np.ndarray
    raw_risk: float
    top1_pixel: int
    top1_vector: tuple[float, float, float]
    directed_top1_error_deg: float
    target_pixels: tuple[int, ...]
    target_vectors: tuple[tuple[float, float, float], ...]
    period_hours_targets: tuple[float, ...]


class _ObjectSequence(Dataset[PreparedObject]):
    def __init__(self, values: Sequence[PreparedObject]) -> None:
        if not values:
            raise TrainingContractError("training/evaluation partition must not be empty")
        self.values = tuple(values)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> PreparedObject:
        return self.values[index]


def configure_determinism(seed: int) -> None:
    if seed not in FROZEN_SEEDS:
        raise TrainingContractError(f"seed must be one of {FROZEN_SEEDS}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _loader(
    values: Sequence[PreparedObject] | Dataset[PreparedObject],
    config: TrainingConfig,
    *,
    epoch: int,
    shuffle: bool,
) -> DataLoader:
    dataset = values if isinstance(values, Dataset) else _ObjectSequence(values)
    generator = torch.Generator().manual_seed(config.seed + epoch * 1_000_003)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
        collate_fn=collate_prepared_objects,
        drop_last=False,
    )


def _move_batch(
    batch: Mapping[str, object], device: torch.device, control: str
) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    if control == "geometry_disabled":
        tokens = moved["tokens"].clone()
        tokens[..., GEOMETRY_FEATURE_SLICE] = 0.0
        moved["tokens"] = tokens
    return moved


def deterministic_label_shuffle(
    values: Sequence[PreparedObject], seed: int
) -> tuple[PreparedObject, ...]:
    """Permute complete source-solution sets across training objects only."""
    if len(values) < 2:
        raise TrainingContractError("label shuffle requires at least two training objects")
    order = np.random.default_rng(seed).permutation(len(values))
    if np.any(order == np.arange(len(values))):
        order = np.roll(np.arange(len(values)), 1)
    shuffled: list[PreparedObject] = []
    for index, donor_index in enumerate(order):
        value = values[index]
        donor = values[int(donor_index)]
        shuffled.append(
            PreparedObject(
                object_id=value.object_id,
                tokens=value.tokens,
                observation_mask=value.observation_mask,
                epoch_descriptors=value.epoch_descriptors,
                epoch_mask=value.epoch_mask,
                period_values=value.period_values,
                period_mask=value.period_mask,
                target_pixels=donor.target_pixels,
                target_vectors=donor.target_vectors,
                period_hours_targets=value.period_hours_targets,
            )
        )
    return tuple(shuffled)


def _derangement(size: int, seed: int) -> np.ndarray:
    if size < 2:
        raise TrainingContractError("a derangement requires at least two objects")
    order = np.random.default_rng(seed).permutation(size)
    if np.any(order == np.arange(size)):
        order = np.roll(np.arange(size), 1)
    return order


def deterministic_input_derangement(
    values: Sequence[PreparedObject], seed: int
) -> tuple[PreparedObject, ...]:
    """Pair each target with another object's complete deployable input."""
    order = _derangement(len(values), seed)
    result: list[PreparedObject] = []
    for index, donor_index in enumerate(order):
        target = values[index]
        donor = values[int(donor_index)]
        result.append(
            PreparedObject(
                object_id=target.object_id,
                tokens=donor.tokens,
                observation_mask=donor.observation_mask,
                epoch_descriptors=donor.epoch_descriptors,
                epoch_mask=donor.epoch_mask,
                period_values=donor.period_values,
                period_mask=donor.period_mask,
                target_pixels=target.target_pixels,
                target_vectors=target.target_vectors,
                period_hours_targets=target.period_hours_targets,
            )
        )
    return tuple(result)


def longitude_rotation_augmentation(
    values: Sequence[PreparedObject], *, seed: int, epoch: int, nside: int
) -> tuple[PreparedObject, ...]:
    """Apply a deterministic ecliptic-z coordinate rotation to training only."""
    angles = (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)
    augmented: list[PreparedObject] = []
    for index, value in enumerate(values):
        choice = (seed + epoch + index * 17) % len(angles)
        angle = angles[choice]
        cosine, sine = math.cos(angle), math.sin(angle)
        rotation = np.array(
            ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64
        )
        tokens = value.tokens.copy()
        for start in (5, 9):
            vectors = tokens[..., start : start + 3]
            tokens[..., start : start + 3] = vectors @ rotation.T
        vectors = value.target_vectors @ rotation.T
        pixels = np.asarray([vector_to_pixel(vector, nside) for vector in vectors], dtype=np.int64)
        augmented.append(
            PreparedObject(
                object_id=value.object_id,
                tokens=tokens,
                observation_mask=value.observation_mask,
                epoch_descriptors=value.epoch_descriptors,
                epoch_mask=value.epoch_mask,
                period_values=value.period_values,
                period_mask=value.period_mask,
                target_pixels=pixels,
                target_vectors=vectors,
                period_hours_targets=value.period_hours_targets,
            )
        )
    return tuple(augmented)


def zero_signal_inputs(values: Sequence[PreparedObject]) -> tuple[PreparedObject, ...]:
    """Zero measured features while preserving only auditable padding structure."""
    return tuple(
        PreparedObject(
            object_id=value.object_id,
            tokens=np.zeros_like(value.tokens),
            observation_mask=value.observation_mask,
            epoch_descriptors=np.zeros_like(value.epoch_descriptors),
            epoch_mask=value.epoch_mask,
            period_values=np.zeros_like(value.period_values),
            period_mask=np.zeros_like(value.period_mask),
            target_pixels=value.target_pixels,
            target_vectors=value.target_vectors,
            period_hours_targets=value.period_hours_targets,
        )
        for value in values
    )


def _forward(model: GeometryHierarchicalDensityModel, batch: Mapping[str, object]):
    return model(
        batch["tokens"].float(),
        batch["observation_mask"].bool(),
        batch["epoch_descriptors"].float(),
        batch["epoch_mask"].bool(),
        batch["period_values"].float(),
        batch["period_mask"].bool(),
    )


def training_loss(
    model: GeometryHierarchicalDensityModel,
    batch: Mapping[str, object],
    config: TrainingConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    output = _forward(model, batch)
    density_loss = hierarchical_multi_solution_density_nll(
        output, batch["target_pixels"].long(), batch["target_mask"].bool()
    )
    centers = torch.as_tensor(
        pixel_centers_xyz(model.config.nside).copy(),
        device=output.logits.device,
        dtype=torch.float64,
    )
    top_vectors = centers[torch.argmax(output.logits.detach(), dim=1)]
    target_vectors = batch["target_vectors"].to(dtype=torch.float64)
    target_mask = batch["target_mask"].bool()
    # Collation pads solution sets with zero vectors.  Validate and score only
    # genuine catalogue solutions; passing padding to the directional helper
    # would correctly reject it as an undefined direction before masking.
    errors = torch.full(
        target_mask.shape, float("inf"), device=target_vectors.device, dtype=torch.float64
    )
    expanded_top_vectors = top_vectors[:, None, :].expand_as(target_vectors)
    errors[target_mask] = torch_directed_angular_error_deg(
        expanded_top_vectors[target_mask], target_vectors[target_mask]
    )
    risk_target = torch.min(errors, dim=1).values.detach()
    risk_prediction = F.softplus(output.raw_risk.float()) * config.risk_scale_deg
    risk_loss = (
        F.huber_loss(
            risk_prediction,
            risk_target.to(dtype=risk_prediction.dtype),
            delta=config.risk_huber_delta_deg,
        )
        / config.risk_scale_deg
    )
    total = density_loss + config.risk_loss_weight * risk_loss
    return total, {
        "density_loss": float(density_loss.detach()),
        "risk_loss": float(risk_loss.detach()),
    }


def _learning_rate(config: TrainingConfig, epoch: int) -> float:
    if epoch < config.warmup_epochs:
        return config.learning_rate * (epoch + 1) / config.warmup_epochs
    progress = (epoch - config.warmup_epochs) / max(config.max_epochs - config.warmup_epochs - 1, 1)
    return config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


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


def save_training_state(
    directory: str | Path,
    model: GeometryHierarchicalDensityModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    epoch: int,
    best_epoch: int,
    best_validation_loss: float,
    stale_epochs: int,
    history: Sequence[EpochMetrics],
) -> None:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    tensor_path = destination / "last-state.safetensors"
    temporary_tensor = destination / ".last-state.safetensors.tmp"
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
        destination / "last-state.json",
        {
            "schema": TRAINING_STATE_SCHEMA,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "stale_epochs": stale_epochs,
            "history": [asdict(item) for item in history],
            "optimizer_scalars": optimizer_scalars,
            "scaler": scaler.state_dict(),
        },
    )


def load_training_state(
    directory: str | Path,
    model: GeometryHierarchicalDensityModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, int, float, int, list[EpochMetrics]]:
    source = Path(directory)
    try:
        metadata = json.loads((source / "last-state.json").read_text())
        tensors = load_file(str(source / "last-state.safetensors"), device=str(device))
    except (OSError, json.JSONDecodeError, RuntimeError) as exc:
        raise TrainingContractError(f"cannot load training state: {exc}") from exc
    if metadata.get("schema") != TRAINING_STATE_SCHEMA:
        raise TrainingContractError("training state schema mismatch")
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
    history = [EpochMetrics(**value) for value in metadata["history"]]
    return (
        int(metadata["epoch"]) + 1,
        int(metadata["best_epoch"]),
        float(metadata["best_validation_loss"]),
        int(metadata["stale_epochs"]),
        history,
    )


def _mean_partition_loss(
    model: GeometryHierarchicalDensityModel,
    values: Sequence[PreparedObject] | Dataset[PreparedObject],
    config: TrainingConfig,
    device: torch.device,
    epoch: int,
) -> float:
    model.eval()
    weighted_loss = 0.0
    n_objects = 0
    with torch.no_grad():
        for raw_batch in _loader(values, config, epoch=epoch, shuffle=False):
            batch = _move_batch(raw_batch, device, config.control)
            loss, _ = training_loss(model, batch, config)
            size = len(batch["object_ids"])
            weighted_loss += float(loss) * size
            n_objects += size
    return weighted_loss / n_objects


def fit_model(
    model: GeometryHierarchicalDensityModel,
    train_values: Sequence[PreparedObject],
    validation_values: Sequence[PreparedObject],
    config: TrainingConfig,
    output_directory: str | Path,
    *,
    device: str | torch.device = "cpu",
    resume: bool = False,
    stop_after_epoch: int | None = None,
) -> FitResult:
    configure_determinism(config.seed)
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise TrainingContractError("CUDA was requested but is unavailable")
    if model.config.geometry_required == (config.control == "geometry_disabled"):
        raise TrainingContractError("model geometry contract does not match the training control")
    if stop_after_epoch is not None and (
        stop_after_epoch < 0 or stop_after_epoch >= config.max_epochs
    ):
        raise TrainingContractError("stop_after_epoch must identify an epoch before max_epochs")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    training_values = (
        deterministic_label_shuffle(train_values, config.seed)
        if config.control == "label_shuffle"
        else tuple(train_values)
    )
    model.to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    amp_enabled = config.mixed_precision and selected_device.type == "cuda"
    scaler = make_grad_scaler("cuda", enabled=amp_enabled)
    start_epoch = 0
    best_epoch = -1
    best_validation = math.inf
    stale_epochs = 0
    history: list[EpochMetrics] = []
    if resume:
        start_epoch, best_epoch, best_validation, stale_epochs, history = load_training_state(
            output, model, optimizer, scaler, selected_device
        )
    for epoch in range(start_epoch, config.max_epochs):
        learning_rate = _learning_rate(config, epoch)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_total = 0.0
        train_count = 0
        epoch_values = (
            longitude_rotation_augmentation(
                training_values, seed=config.seed, epoch=epoch, nside=model.config.nside
            )
            if config.longitude_rotation_augmentation
            else training_values
        )
        loader = _loader(epoch_values, config, epoch=epoch, shuffle=True)
        for batch_index, raw_batch in enumerate(loader):
            batch = _move_batch(raw_batch, selected_device, config.control)
            with torch.autocast(
                device_type=selected_device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                loss, _ = training_loss(model, batch, config)
                scaled_loss = loss / config.gradient_accumulation
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
        validation = _mean_partition_loss(model, validation_values, config, selected_device, epoch)
        metrics = EpochMetrics(epoch, learning_rate, train_total / train_count, validation)
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
        save_training_state(
            output,
            model,
            optimizer,
            scaler,
            epoch=epoch,
            best_epoch=best_epoch,
            best_validation_loss=best_validation,
            stale_epochs=stale_epochs,
            history=history,
        )
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            break
        if stale_epochs >= config.patience:
            break
    model.load_state_dict(
        load_file(str(output / "best-model.safetensors"), device=str(selected_device)), strict=True
    )
    result = FitResult(
        best_epoch=best_epoch,
        best_validation_loss=best_validation,
        epochs_completed=len(history),
        history=tuple(history),
        stopped_early=len(history) < config.max_epochs,
    )
    _atomic_json(
        output / "training-history.json",
        {**asdict(result), "history": [asdict(item) for item in history]},
    )
    return result


def predict_objects(
    model: GeometryHierarchicalDensityModel,
    values: Sequence[PreparedObject],
    config: TrainingConfig,
    *,
    device: str | torch.device = "cpu",
) -> tuple[RawPrediction, ...]:
    selected_device = torch.device(device)
    model.to(selected_device).eval()
    centers = pixel_centers_xyz(model.config.nside)
    predictions: list[RawPrediction] = []
    with torch.no_grad():
        for raw_batch in _loader(values, config, epoch=0, shuffle=False):
            batch = _move_batch(raw_batch, selected_device, config.control)
            output = _forward(model, batch)
            logits = output.logits.detach().cpu().numpy().astype(np.float64)
            raw_risks = output.raw_risk.detach().cpu().numpy().astype(np.float64)
            target_pixels = batch["target_pixels"].detach().cpu().numpy()
            target_mask = batch["target_mask"].detach().cpu().numpy().astype(bool)
            target_vectors = batch["target_vectors"].detach().cpu().numpy().astype(np.float64)
            period_targets = batch["period_hours_targets"].detach().cpu().numpy()
            for index, object_id in enumerate(batch["object_ids"]):
                top_pixel = int(np.argmax(logits[index]))
                vector = tuple(float(value) for value in centers[top_pixel])
                valid_vectors = target_vectors[index, target_mask[index]]
                errors = directed_angular_error_deg(np.asarray(vector)[None, :], valid_vectors)
                predictions.append(
                    RawPrediction(
                        object_id=object_id,
                        logits=logits[index],
                        raw_risk=float(raw_risks[index]),
                        top1_pixel=top_pixel,
                        top1_vector=vector,
                        directed_top1_error_deg=float(np.min(errors)),
                        target_pixels=tuple(
                            int(value) for value in target_pixels[index, target_mask[index]]
                        ),
                        target_vectors=tuple(
                            tuple(float(component) for component in row) for row in valid_vectors
                        ),
                        period_hours_targets=tuple(
                            float(value) for value in period_targets[index, target_mask[index]]
                        ),
                    )
                )
    if len({value.object_id for value in predictions}) != len(predictions):
        raise TrainingContractError("prediction partition contains duplicate object IDs")
    return tuple(predictions)


def model_from_config(config: V2ModelConfig, seed: int) -> GeometryHierarchicalDensityModel:
    configure_determinism(seed)
    return GeometryHierarchicalDensityModel(config)
