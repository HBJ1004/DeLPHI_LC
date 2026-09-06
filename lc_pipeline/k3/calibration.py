"""Calibration contract for K3 compatibility surfaces and axial cones."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


class K3CalibrationError(ValueError):
    """Raised for invalid or insufficient K3 calibration data."""


@dataclass(frozen=True)
class K3Calibration:
    calibration_id: str
    temperature: float
    cone90_deg: float
    cone95_deg: float
    risk_raw_knots: tuple[float, ...]
    risk_deg_knots: tuple[float, ...]
    ambiguous_risk_deg: float = 25.0
    abstain_risk_deg: float = 45.0
    uninformative_entropy: float = 0.995

    def __post_init__(self) -> None:
        if not self.calibration_id:
            raise K3CalibrationError("calibration_id is required")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise K3CalibrationError("temperature must be finite and positive")
        if not 0 <= self.cone90_deg <= self.cone95_deg <= 90:
            raise K3CalibrationError("cone radii must satisfy 0 <= cone90 <= cone95 <= 90")
        if len(self.risk_raw_knots) != len(self.risk_deg_knots) or len(self.risk_raw_knots) < 2:
            raise K3CalibrationError("risk calibration requires at least two aligned knots")
        if any(not math.isfinite(value) for value in self.risk_raw_knots + self.risk_deg_knots):
            raise K3CalibrationError("risk calibration knots must be finite")
        if any(left >= right for left, right in zip(self.risk_raw_knots, self.risk_raw_knots[1:])):
            raise K3CalibrationError("raw risk knots must be strictly increasing")
        if any(left > right for left, right in zip(self.risk_deg_knots, self.risk_deg_knots[1:])):
            raise K3CalibrationError("calibrated risk must be monotonic")
        if not 0 <= self.ambiguous_risk_deg <= self.abstain_risk_deg <= 90:
            raise K3CalibrationError("risk thresholds must be ordered within [0,90]")
        if not 0 < self.uninformative_entropy <= 1:
            raise K3CalibrationError("uninformative entropy threshold must lie in (0,1]")

    def calibrated_risk_deg(self, raw_risk: float) -> float:
        if not math.isfinite(raw_risk):
            raise K3CalibrationError("raw risk must be finite")
        return float(np.interp(raw_risk, self.risk_raw_knots, self.risk_deg_knots))


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


def normalized_entropy(mass: Sequence[float]) -> float:
    values = np.asarray(mass, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise K3CalibrationError("compatibility mass must be a finite vector")
    if np.any(values < 0) or not np.isclose(values.sum(), 1.0, atol=1e-12, rtol=0):
        raise K3CalibrationError("compatibility mass must be nonnegative and sum to one")
    positive = values[values > 0]
    return float(-np.sum(positive * np.log(positive)) / math.log(values.size))
