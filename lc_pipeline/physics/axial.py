"""Canonical antipode-aware axis operations for the V2 comparison study.

The comparison study evaluates an *axis* rather than an oriented pole.  The
equivalence ``u ~ -u`` is therefore an explicit metric convention; it does not
assert that two mirror solutions in an asteroid catalogue are exact antipodes.

Reported angles use float64 ``atan2(||p x s||, |p . s|)`` and span [0, 90]
degrees.  Training uses ``1 - (p . s)^2`` to avoid inverse-trigonometric
boundary gradients.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Sequence

if TYPE_CHECKING:
    import numpy as np
    import torch

_VECTOR_SIZE = 3


@dataclass(frozen=True)
class AxialOracleMatch:
    """Deterministic best prediction/source match under axis equivalence."""

    error_deg: float
    prediction_index: int
    source_axis_index: int


def _unit_vector_scalar(vector: Sequence[float], *, name: str) -> tuple[float, float, float]:
    try:
        values = tuple(float(component) for component in vector)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an iterable of three real values") from exc
    if len(values) != _VECTOR_SIZE:
        raise ValueError(f"{name} must have shape (3,); got length {len(values)}")
    if not all(math.isfinite(component) for component in values):
        raise ValueError(f"{name} must contain only finite values")
    scale = max(abs(component) for component in values)
    if scale == 0.0:
        raise ValueError(f"{name} is a zero vector, which has no axis")
    scaled = tuple(component / scale for component in values)
    norm = math.hypot(*scaled)
    return tuple(component / norm for component in scaled)


def axial_angular_error_deg_scalar(
    first: Sequence[float], second: Sequence[float]
) -> float:
    """Return one antipode-invariant axis separation in binary64 arithmetic."""
    first_unit = _unit_vector_scalar(first, name="first")
    second_unit = _unit_vector_scalar(second, name="second")
    cross = (
        first_unit[1] * second_unit[2] - first_unit[2] * second_unit[1],
        first_unit[2] * second_unit[0] - first_unit[0] * second_unit[2],
        first_unit[0] * second_unit[1] - first_unit[1] * second_unit[0],
    )
    cross_norm = math.hypot(*cross)
    dot = math.fsum(left * right for left, right in zip(first_unit, second_unit))
    return math.degrees(math.atan2(cross_norm, abs(dot)))


def _unit_vectors_numpy(vectors: object, *, name: str) -> np.ndarray:
    import numpy as np

    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != _VECTOR_SIZE:
        raise ValueError(f"{name} must have shape (..., 3); got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    scales = np.max(np.abs(array), axis=-1)
    if np.any(scales == 0.0):
        raise ValueError(f"{name} contains a zero vector, which has no axis")
    scaled = array / np.expand_dims(scales, axis=-1)
    # ``numpy.linalg.vector_norm`` was introduced in NumPy 2.0.  The project
    # supports NumPy 1.26, where the equivalent API is ``numpy.linalg.norm``.
    norms = np.linalg.norm(scaled, axis=-1)
    if not np.all(np.isfinite(norms)):
        raise ValueError(f"{name} has a non-finite vector norm")
    return scaled / np.expand_dims(norms, axis=-1)


def _vector_collection_numpy(vectors: object, *, name: str) -> np.ndarray:
    import numpy as np

    array = np.asarray(vectors, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one vector")
    if array.shape == (_VECTOR_SIZE,):
        array = array[np.newaxis, :]
    if array.ndim != 2 or array.shape[1] != _VECTOR_SIZE:
        raise ValueError(f"{name} must have shape (N, 3) or (3,); got {array.shape}")
    return _unit_vectors_numpy(array, name=name)


def axial_angular_error_deg(first: object, second: object) -> float | np.ndarray:
    """Return antipode-invariant angles for broadcastable vector arrays."""
    import numpy as np

    first_unit = _unit_vectors_numpy(first, name="first")
    second_unit = _unit_vectors_numpy(second, name="second")
    try:
        first_unit, second_unit = np.broadcast_arrays(first_unit, second_unit)
    except ValueError as exc:
        raise ValueError("first and second must have broadcastable leading dimensions") from exc
    cross_norm = np.linalg.norm(np.cross(first_unit, second_unit, axis=-1), axis=-1)
    dot = np.sum(first_unit * second_unit, axis=-1)
    result = np.rad2deg(np.arctan2(cross_norm, np.abs(dot)))
    return float(result) if result.ndim == 0 else result


def pairwise_axial_angular_errors_deg(
    predictions: object, source_axes: object
) -> np.ndarray:
    """Return every prediction/source error as a float64 ``(K, S)`` matrix."""
    import numpy as np

    prediction_vectors = _vector_collection_numpy(predictions, name="predictions")
    source_vectors = _vector_collection_numpy(source_axes, name="source_axes")
    return np.asarray(
        axial_angular_error_deg(
            prediction_vectors[:, np.newaxis, :], source_vectors[np.newaxis, :, :]
        ),
        dtype=np.float64,
    )


def nearest_source_axis_errors_deg(predictions: object, source_axes: object) -> np.ndarray:
    """Return the best explicit source-axis error for each prediction."""
    return pairwise_axial_angular_errors_deg(predictions, source_axes).min(axis=1)


def oracle_source_axis_match(predictions: object, source_axes: object) -> AxialOracleMatch:
    """Return the row-major deterministic oracle match over K by S pairs."""
    import numpy as np

    errors = pairwise_axial_angular_errors_deg(predictions, source_axes)
    flat_index = int(np.argmin(errors))
    prediction_index, source_axis_index = np.unravel_index(flat_index, errors.shape)
    return AxialOracleMatch(
        error_deg=float(errors[prediction_index, source_axis_index]),
        prediction_index=int(prediction_index),
        source_axis_index=int(source_axis_index),
    )


def _unit_vectors_torch(vectors: torch.Tensor, *, name: str) -> torch.Tensor:
    import torch

    if not isinstance(vectors, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if vectors.ndim == 0 or vectors.shape[-1] != _VECTOR_SIZE:
        raise ValueError(f"{name} must have shape (..., 3); got {tuple(vectors.shape)}")
    if not torch.is_floating_point(vectors):
        raise TypeError(f"{name} must have a floating-point dtype")
    if not bool(torch.isfinite(vectors).all()):
        raise ValueError(f"{name} must contain only finite values")
    scales = torch.amax(torch.abs(vectors), dim=-1, keepdim=True)
    if bool(torch.any(scales == 0)):
        raise ValueError(f"{name} contains a zero vector, which has no axis")
    scaled = vectors / scales
    norms = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
    if not bool(torch.isfinite(norms).all()):
        raise ValueError(f"{name} has a non-finite vector norm")
    return scaled / norms


def torch_axial_angular_error_deg(
    first: torch.Tensor, second: torch.Tensor
) -> torch.Tensor:
    """Differentiable float64 counterpart of :func:`axial_angular_error_deg`."""
    import torch

    if not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor):
        raise TypeError("first and second must be torch.Tensor instances")
    if first.device != second.device:
        raise ValueError("first and second must be on the same device")
    first_unit = _unit_vectors_torch(first.to(dtype=torch.float64), name="first")
    second_unit = _unit_vectors_torch(second.to(dtype=torch.float64), name="second")
    try:
        first_unit, second_unit = torch.broadcast_tensors(first_unit, second_unit)
    except RuntimeError as exc:
        raise ValueError("first and second must have broadcastable leading dimensions") from exc
    cross_norm = torch.linalg.vector_norm(
        torch.linalg.cross(first_unit, second_unit, dim=-1), dim=-1
    )
    dot = torch.sum(first_unit * second_unit, dim=-1)
    return torch.rad2deg(torch.atan2(cross_norm, torch.abs(dot)))


def axial_sine_squared_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> torch.Tensor:
    """Return ``sin(angle)^2 = 1 - (p . s)^2`` for normalized axes."""
    import torch

    if not isinstance(predictions, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError("predictions and targets must be torch.Tensor instances")
    if predictions.device != targets.device:
        raise ValueError("predictions and targets must be on the same device")
    prediction_unit = _unit_vectors_torch(predictions, name="predictions")
    target_unit = _unit_vectors_torch(targets, name="targets")
    try:
        prediction_unit, target_unit = torch.broadcast_tensors(prediction_unit, target_unit)
    except RuntimeError as exc:
        raise ValueError(
            "predictions and targets must have broadcastable leading dimensions"
        ) from exc
    dot = torch.sum(prediction_unit * target_unit, dim=-1)
    losses = (1.0 - dot.square()).clamp_min(0.0)
    if reduction == "none":
        return losses
    if reduction == "mean":
        return losses.mean()
    if reduction == "sum":
        return losses.sum()
    raise ValueError(f"unsupported reduction: {reduction!r}")


def masked_axial_softmin_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    tau: float = math.sin(math.radians(5.0)) ** 2,
) -> torch.Tensor:
    """Normalized soft minimum over every valid prediction/source pair.

    ``predictions`` has shape ``[B, K, 3]`` and targets/mask have shapes
    ``[B, S, 3]``/``[B, S]``.  Normalizing by the number of valid pairs makes
    objects with different source-solution counts comparable.
    """
    import torch

    if predictions.ndim != 3 or predictions.shape[-1] != 3:
        raise ValueError("predictions must have shape [batch, candidate, 3]")
    if targets.ndim != 3 or targets.shape[-1] != 3:
        raise ValueError("targets must have shape [batch, solution, 3]")
    if target_mask.shape != targets.shape[:2] or predictions.shape[0] != targets.shape[0]:
        raise ValueError("target mask and batch dimensions must align")
    if target_mask.dtype is not torch.bool:
        raise TypeError("target_mask must have boolean dtype")
    if not bool(target_mask.any(dim=1).all()):
        raise ValueError("each object requires at least one explicit source solution")
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive")

    prediction_unit = _unit_vectors_torch(predictions, name="predictions")
    # Padded zero targets are excluded before normalization.
    safe_targets = torch.where(target_mask[..., None], targets, torch.ones_like(targets))
    target_unit = _unit_vectors_torch(safe_targets, name="targets")
    dots = torch.einsum("bkc,bsc->bks", prediction_unit, target_unit)
    costs = (1.0 - dots.square()).clamp_min(0.0)
    pair_mask = target_mask[:, None, :].expand_as(costs)
    logits = (-costs / tau).masked_fill(~pair_mask, float("-inf"))
    valid_counts = pair_mask.sum(dim=(1, 2)).to(dtype=costs.dtype)
    per_object = -tau * (torch.logsumexp(logits.flatten(1), dim=1) - torch.log(valid_counts))
    return per_object.mean()


def axial_separation_penalty(
    predictions: torch.Tensor, *, minimum_separation_deg: float = 25.0
) -> torch.Tensor:
    """Penalize candidate axes closer than the fixed minimum separation."""
    import torch

    if predictions.ndim != 3 or predictions.shape[-1] != 3:
        raise ValueError("predictions must have shape [batch, candidate, 3]")
    if predictions.shape[1] < 2:
        return predictions.sum() * 0.0
    if not math.isfinite(minimum_separation_deg) or not 0 < minimum_separation_deg < 90:
        raise ValueError("minimum_separation_deg must lie in (0, 90)")
    unit = _unit_vectors_torch(predictions, name="predictions")
    row, column = torch.triu_indices(
        predictions.shape[1], predictions.shape[1], offset=1, device=predictions.device
    )
    dots = torch.sum(unit[:, row] * unit[:, column], dim=-1)
    sine_squared = (1.0 - dots.square()).clamp_min(0.0)
    threshold = math.sin(math.radians(minimum_separation_deg)) ** 2
    return torch.relu(threshold - sine_squared).mean()


def axial_oracle_training_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    separation_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Comparison-study loss: axial soft oracle plus anti-collapse penalty."""
    if not math.isfinite(separation_weight) or separation_weight < 0:
        raise ValueError("separation_weight must be finite and nonnegative")
    oracle = masked_axial_softmin_loss(predictions, targets, target_mask)
    separation = axial_separation_penalty(predictions)
    total = oracle + separation_weight * separation
    return total, {"axial_oracle_loss": oracle, "separation_penalty": separation}


__all__ = [
    "AxialOracleMatch",
    "axial_angular_error_deg",
    "axial_angular_error_deg_scalar",
    "axial_oracle_training_loss",
    "axial_separation_penalty",
    "axial_sine_squared_loss",
    "masked_axial_softmin_loss",
    "nearest_source_axis_errors_deg",
    "oracle_source_axis_match",
    "pairwise_axial_angular_errors_deg",
    "torch_axial_angular_error_deg",
]
