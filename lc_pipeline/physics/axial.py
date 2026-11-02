"""Shared antipode-aware axis operations.

The comparison study evaluates an *axis* rather than an oriented pole.  The
equivalence ``u ~ -u`` is therefore an explicit metric convention; it does not
assert that two mirror solutions in an asteroid catalogue are exact antipodes.

Reported angles use float64 ``atan2(||p x s||, |p . s|)`` and span [0, 90]
degrees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

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


__all__ = [
    "AxialOracleMatch",
    "axial_angular_error_deg",
    "axial_angular_error_deg_scalar",
    "nearest_source_axis_errors_deg",
    "oracle_source_axis_match",
    "pairwise_axial_angular_errors_deg",
    "torch_axial_angular_error_deg",
]
