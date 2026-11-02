"""Finite-sample calibration for K3 axial cones."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


class K3CalibrationError(ValueError):
    """Raised for invalid or insufficient K3 calibration data."""


def conformal_radius(errors_deg: Sequence[float], coverage: float) -> float:
    """Finite-sample split-conformal radius on the bounded axial domain.

    The usual rank is ``ceil((n + 1) * coverage)``. When that rank is ``n + 1``
    there is no observed order statistic capable of providing the requested
    guarantee. Because axial error is known to be at most 90 degrees, the
    rigorous fallback is 90 rather than the observed maximum.
    """
    values = np.asarray(errors_deg, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise K3CalibrationError("conformal errors must be a finite vector of length at least two")
    if np.any(values < 0) or np.any(values > 90) or not 0 < coverage < 1:
        raise K3CalibrationError("axial errors must lie in [0,90] and coverage in (0,1)")
    rank = math.ceil((values.size + 1) * coverage)
    if rank > values.size:
        return 90.0
    return float(np.sort(values, kind="stable")[rank - 1])
