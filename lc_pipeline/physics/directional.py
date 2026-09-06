"""Canonical directed-sphere operations for DeLPHI V2 pole vectors.

A spin pole is an oriented vector on :math:`S^2`.  Consequently, ``p`` and
``-p`` are different predictions unless both vectors are independently present
in the source catalogue.  This module is the only implementation of pole-angle
metrics used by V2 evaluation and reporting code.

Evaluation is deliberately performed in float64 using the well-conditioned
``atan2(||p x s||, p . s)`` identity.  It represents the complete [0, 180]
degree interval without clipping and retains arbitrarily small non-zero angles.
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
class OraclePoleMatch:
    """The deterministic best match between predictions and supplied poles."""

    error_deg: float
    prediction_index: int
    source_pole_index: int


def _unit_vector_scalar(vector: Sequence[float], *, name: str) -> tuple[float, float, float]:
    """Validate and robustly normalize one vector using binary64 arithmetic."""
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
        raise ValueError(f"{name} is a zero vector, which has no direction")
    scaled = tuple(component / scale for component in values)
    norm = math.hypot(*scaled)
    return tuple(component / norm for component in scaled)


def directed_angular_error_deg_scalar(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    """Return one directed pole separation without importing NumPy or PyTorch."""
    first_unit = _unit_vector_scalar(first, name="first")
    second_unit = _unit_vector_scalar(second, name="second")
    cross = (
        first_unit[1] * second_unit[2] - first_unit[2] * second_unit[1],
        first_unit[2] * second_unit[0] - first_unit[0] * second_unit[2],
        first_unit[0] * second_unit[1] - first_unit[1] * second_unit[0],
    )
    cross_norm = math.hypot(*cross)
    dot = math.fsum(left * right for left, right in zip(first_unit, second_unit))
    return math.degrees(math.atan2(cross_norm, dot))


def _unit_vectors_numpy(vectors: object, *, name: str) -> np.ndarray:
    """Validate vectors and return a float64, unit-normalized array."""
    import numpy as np

    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != _VECTOR_SIZE:
        raise ValueError(f"{name} must have shape (..., 3); got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")

    scales = np.max(np.abs(array), axis=-1)
    if np.any(scales == 0.0):
        raise ValueError(f"{name} contains a zero vector, which has no direction")
    scaled = array / np.expand_dims(scales, axis=-1)
    norms = np.linalg.norm(scaled, axis=-1)
    if not np.all(np.isfinite(norms)):
        raise ValueError(f"{name} has a non-finite vector norm")

    return scaled / np.expand_dims(norms, axis=-1)


def _vector_collection_numpy(vectors: object, *, name: str) -> np.ndarray:
    """Validate one vector or a non-empty two-dimensional vector collection."""
    import numpy as np

    array = np.asarray(vectors, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one vector")
    if array.shape == (_VECTOR_SIZE,):
        array = array[np.newaxis, :]
    if array.ndim != 2 or array.shape[1] != _VECTOR_SIZE:
        raise ValueError(f"{name} must have shape (N, 3) or (3,); got {array.shape}")
    return _unit_vectors_numpy(array, name=name)


def directed_angular_error_deg(first: object, second: object) -> float | np.ndarray:
    """Return the directed angular separation of broadcastable pole vectors.

    Inputs may be non-unit vectors but must be finite, non-zero, broadcastable,
    and have final dimension three.  Scalar vector pairs return ``float``;
    batched inputs return a float64 ``numpy.ndarray``.
    """
    import numpy as np

    first_unit = _unit_vectors_numpy(first, name="first")
    second_unit = _unit_vectors_numpy(second, name="second")
    try:
        first_unit, second_unit = np.broadcast_arrays(first_unit, second_unit)
    except ValueError as exc:
        raise ValueError("first and second must have broadcastable leading dimensions") from exc

    cross_norm = np.linalg.norm(np.cross(first_unit, second_unit, axis=-1), axis=-1)
    dot = np.sum(first_unit * second_unit, axis=-1)
    angle_deg = np.rad2deg(np.arctan2(cross_norm, dot))

    if angle_deg.ndim == 0:
        return float(angle_deg)
    return angle_deg


def pairwise_directed_angular_errors_deg(
    predictions: object,
    source_poles: object,
) -> np.ndarray:
    """Return all directed prediction/source-pole errors as a ``(K, S)`` matrix.

    ``source_poles`` is exhaustive: this function never creates reflected,
    negated, or otherwise inferred catalogue solutions.
    """
    import numpy as np

    prediction_vectors = _vector_collection_numpy(predictions, name="predictions")
    source_vectors = _vector_collection_numpy(source_poles, name="source_poles")
    errors = directed_angular_error_deg(
        prediction_vectors[:, np.newaxis, :],
        source_vectors[np.newaxis, :, :],
    )
    return np.asarray(errors, dtype=np.float64)


def nearest_source_pole_errors_deg(
    predictions: object,
    source_poles: object,
) -> np.ndarray:
    """Return each prediction's minimum error to an explicitly supplied pole."""
    return pairwise_directed_angular_errors_deg(predictions, source_poles).min(axis=1)


def oracle_source_pole_match(
    predictions: object,
    source_poles: object,
) -> OraclePoleMatch:
    """Return the best prediction/source pair with deterministic row-major ties.

    This is a matched-K oracle diagnostic, not a deployable top-1 metric.  Only
    entries in ``source_poles`` are eligible ground truths.
    """
    import numpy as np

    errors = pairwise_directed_angular_errors_deg(predictions, source_poles)
    flat_index = int(np.argmin(errors))
    prediction_index, source_pole_index = np.unravel_index(flat_index, errors.shape)
    return OraclePoleMatch(
        error_deg=float(errors[prediction_index, source_pole_index]),
        prediction_index=int(prediction_index),
        source_pole_index=int(source_pole_index),
    )


def _unit_vectors_torch(vectors: torch.Tensor, *, name: str) -> torch.Tensor:
    """Validate a floating tensor and return unit vectors without detaching it."""
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
        raise ValueError(f"{name} contains a zero vector, which has no direction")
    scaled = vectors / scales
    norms = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
    if not bool(torch.isfinite(norms).all()):
        raise ValueError(f"{name} has a non-finite vector norm")
    return scaled / norms


def torch_directed_angular_error_deg(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    """Torch counterpart of :func:`directed_angular_error_deg`.

    The calculation is promoted to float64 so values exported as scientific
    metrics obey the same numerical contract on CPU and accelerator devices.
    """
    import torch

    if not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor):
        raise TypeError("first and second must be torch.Tensor instances")
    if not torch.is_floating_point(first):
        raise TypeError("first must have a floating-point dtype")
    if not torch.is_floating_point(second):
        raise TypeError("second must have a floating-point dtype")
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
    return torch.rad2deg(torch.atan2(cross_norm, dot))


def directed_chord_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> torch.Tensor:
    """Stable directional training loss equal to half squared chord distance.

    For normalized vectors the unreduced loss is ``1 - p . s`` and spans
    [0, 2].  Computing it from the squared vector difference avoids inverse
    trigonometric boundary gradients while preserving pole orientation.
    """
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

    losses = 0.5 * torch.sum((prediction_unit - target_unit).square(), dim=-1)
    if reduction == "none":
        return losses
    if reduction == "mean":
        return losses.mean()
    if reduction == "sum":
        return losses.sum()
    raise ValueError(f"unsupported reduction: {reduction!r}")


__all__ = [
    "OraclePoleMatch",
    "directed_angular_error_deg",
    "directed_angular_error_deg_scalar",
    "directed_chord_loss",
    "nearest_source_pole_errors_deg",
    "oracle_source_pole_match",
    "pairwise_directed_angular_errors_deg",
    "torch_directed_angular_error_deg",
]
