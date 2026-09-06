"""Strict public result schema for candidate-conditioned K=3 inference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

K3_PREDICTION_SCHEMA = "delphi.k3-axial-prediction.v1"


class K3PredictionStatus(str, Enum):
    OK = "ok"
    AMBIGUOUS = "ambiguous"
    ABSTAIN = "abstain"
    UNINFORMATIVE = "uninformative"
    FAILED = "failed"


@dataclass(frozen=True)
class K3AxisCandidate:
    """One normalized, unsigned search axis and its calibrated diagnostics."""

    axis_xyz: tuple[float, float, float]
    score_logit: float
    compatibility_mass: float
    cone90_deg: float
    cone95_deg: float
    grid_index: int

    def __post_init__(self) -> None:
        vector = np.asarray(self.axis_xyz, dtype=np.float64)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError("axis_xyz must be one finite three-vector")
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or abs(norm - 1.0) > 1e-6:
            raise ValueError("axis_xyz must be normalized within 1e-6")
        scalar_values = (self.score_logit, self.compatibility_mass, self.cone90_deg, self.cone95_deg)
        if not all(math.isfinite(value) for value in scalar_values):
            raise ValueError("candidate scores and cone radii must be finite")
        if not 0 <= self.compatibility_mass <= 1:
            raise ValueError("compatibility_mass must lie in [0, 1]")
        if not 0 <= self.cone90_deg <= self.cone95_deg <= 90:
            raise ValueError("cone radii must satisfy 0 <= cone90 <= cone95 <= 90")
        if isinstance(self.grid_index, bool) or not isinstance(self.grid_index, int) or self.grid_index < 0:
            raise ValueError("grid_index must be a nonnegative integer")

    def as_mapping(self) -> dict[str, Any]:
        return {
            "axis_xyz": list(self.axis_xyz),
            "score_logit": self.score_logit,
            "compatibility_mass": self.compatibility_mass,
            "cone90_deg": self.cone90_deg,
            "cone95_deg": self.cone95_deg,
            "grid_index": self.grid_index,
        }


@dataclass(frozen=True)
class K3AxialPrediction:
    """Deployable result: exactly three axes, or an explicit failure."""

    status: K3PredictionStatus
    axes: tuple[K3AxisCandidate, ...]
    risk_deg: float | None
    model_id: str
    period_provenance: str | None
    tokenizer_schema_sha256: str
    failure_reason: str | None = None
    schema: str = K3_PREDICTION_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.status, K3PredictionStatus):
            raise ValueError("status must be K3PredictionStatus")
        if not self.model_id:
            raise ValueError("model_id is required")
        if len(self.tokenizer_schema_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.tokenizer_schema_sha256
        ):
            raise ValueError("tokenizer_schema_sha256 must be a lowercase SHA-256 digest")
        if self.schema != K3_PREDICTION_SCHEMA:
            raise ValueError("prediction schema mismatch")
        if self.status is K3PredictionStatus.FAILED:
            if self.axes or self.risk_deg is not None or not self.failure_reason:
                raise ValueError("failed predictions require no axes/risk and a failure reason")
            return
        if len(self.axes) != 3 or not all(isinstance(axis, K3AxisCandidate) for axis in self.axes):
            raise ValueError("non-failed predictions require exactly three K3AxisCandidate values")
        if self.risk_deg is None or not math.isfinite(self.risk_deg) or self.risk_deg < 0:
            raise ValueError("non-failed predictions require finite nonnegative risk_deg")
        if self.failure_reason is not None:
            raise ValueError("non-failed predictions may not carry failure_reason")

    @classmethod
    def failed(
        cls,
        *,
        model_id: str,
        tokenizer_schema_sha256: str,
        reason: str,
        period_provenance: str | None = None,
    ) -> "K3AxialPrediction":
        return cls(
            status=K3PredictionStatus.FAILED,
            axes=(),
            risk_deg=None,
            model_id=model_id,
            period_provenance=period_provenance,
            tokenizer_schema_sha256=tokenizer_schema_sha256,
            failure_reason=reason,
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": self.status.value,
            "axes": [axis.as_mapping() for axis in self.axes],
            "risk_deg": self.risk_deg,
            "model_id": self.model_id,
            "period_provenance": self.period_provenance,
            "tokenizer_schema_sha256": self.tokenizer_schema_sha256,
            "failure_reason": self.failure_reason,
        }
