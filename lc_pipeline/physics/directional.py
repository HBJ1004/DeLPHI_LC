"""Shared directed-sphere operations for pole vectors.

A spin pole is an oriented vector on :math:`S^2`.  Consequently, ``p`` and
``-p`` are different predictions unless both vectors are independently present
in the source catalogue. Evaluation uses the axial metrics in
``lc_pipeline.physics.axial``.

Directed angles are computed in float64 using the well-conditioned
``atan2(||p x s||, p . s)`` identity.  It represents the complete [0, 180]
degree interval without clipping and retains arbitrarily small non-zero angles.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import numpy as np

_VECTOR_SIZE = 3


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


__all__ = [
    "directed_angular_error_deg",
    "directed_angular_error_deg_scalar",
]
