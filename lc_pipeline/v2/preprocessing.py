"""Shared observation types and historical tensor-schema metadata.

The prediction command builds tensors with ``lc_pipeline.k3.tokenizer``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Sequence

import numpy as np

TOKENIZER_SCHEMA_NAME = "delphi.observation-tokenizer"
TOKENIZER_SCHEMA_VERSION = "2.3.0"
COORDINATE_FRAME = "asteroid_centric_ecliptic_j2000"
FLOAT_DTYPE = "float32"
GEOMETRY_NORM_EPS = 1e-12

TOKEN_FEATURE_NAMES = (
    "time_from_observation_origin_days",
    "time_from_epoch_start_days",
    "log_object_relative_brightness",
    "measured_error_or_zero",
    "has_measured_error",
    "sun_direction_x",
    "sun_direction_y",
    "sun_direction_z",
    "sun_distance_au",
    "observer_direction_x",
    "observer_direction_y",
    "observer_direction_z",
    "observer_distance_au",
    "inferred_phase_sin",
    "inferred_phase_cos",
    "inferred_phase_harmonic2_sin",
    "inferred_phase_harmonic2_cos",
    "inferred_phase_available",
)
GEOMETRY_FEATURE_NAMES = (
    "sun_direction_x",
    "sun_direction_y",
    "sun_direction_z",
    "sun_distance_au",
    "observer_direction_x",
    "observer_direction_y",
    "observer_direction_z",
    "observer_distance_au",
)
EPOCH_DESCRIPTOR_NAMES = (
    "log_flux_peak_to_peak",
    "log1p_epoch_duration_days",
    "log1p_epoch_observation_count",
)
PERIOD_VALUE_NAMES = ("period_hours", "period_uncertainty_hours")

GEOMETRY_FEATURE_SLICE = slice(5, 13)
PHASE_FEATURE_SLICE = slice(13, 18)


class PreprocessingError(ValueError):
    """Raised when an input violates the fail-closed V2 tensor contract."""


class GeometryMode(str, Enum):
    """Whether geometry channels are exposed to the downstream model."""

    ENABLED = "enabled"
    DISABLED = "disabled"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


_TOKENIZER_SCHEMA_PAYLOAD = MappingProxyType(
    {
        "coordinate_frame": COORDINATE_FRAME,
        "brightness_scaling": "log_positive_flux_divided_by_object_global_median",
        "phase_features": "first_and_second_harmonic_from_ok_inferred_period_only",
        "temporal_scaling": "log1p_day_deltas_and_epoch_observation_count",
        "dtype": FLOAT_DTYPE,
        "epoch_descriptors": EPOCH_DESCRIPTOR_NAMES,
        "epoch_order": "supplied",
        "epoch_split": "stable_time_sort_then_contiguous_chunks",
        "geometry_ablation": "explicit_zero_model_channels_retain_descriptors",
        "geometry_descriptors": GEOMETRY_FEATURE_NAMES,
        "missing_error": "zero_value_with_false_error_mask",
        "padding": "all_zero_with_false_observation_and_epoch_masks_no_repetition",
        "period_values": PERIOD_VALUE_NAMES,
        "schema": TOKENIZER_SCHEMA_NAME,
        "token_features": TOKEN_FEATURE_NAMES,
        "version": TOKENIZER_SCHEMA_VERSION,
    }
)
TOKENIZER_SCHEMA_HASH = hashlib.sha256(
    _canonical_json(dict(_TOKENIZER_SCHEMA_PAYLOAD)).encode("ascii")
).hexdigest()


@dataclass(frozen=True)
class TokenizerSchema:
    """Immutable, hash-bound description of tensor semantics."""

    name: str = TOKENIZER_SCHEMA_NAME
    version: str = TOKENIZER_SCHEMA_VERSION
    sha256: str = TOKENIZER_SCHEMA_HASH
    coordinate_frame: str = COORDINATE_FRAME
    token_features: tuple[str, ...] = TOKEN_FEATURE_NAMES
    geometry_descriptors: tuple[str, ...] = GEOMETRY_FEATURE_NAMES
    epoch_descriptors: tuple[str, ...] = EPOCH_DESCRIPTOR_NAMES
    period_values: tuple[str, ...] = PERIOD_VALUE_NAMES


TOKENIZER_SCHEMA = TokenizerSchema()


def _finite_number(value: Any, field_name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise PreprocessingError(f"{field_name} must be a real number, not a boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PreprocessingError(f"{field_name} must be a finite real number") from exc
    if not math.isfinite(number):
        raise PreprocessingError(f"{field_name} must be finite")
    if positive and number <= 0.0:
        raise PreprocessingError(f"{field_name} must be positive")
    return number


def _geometry_vector(value: Sequence[float], field_name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)):
        raise PreprocessingError(f"{field_name} must contain exactly three numbers")
    try:
        components = tuple(value)
    except TypeError as exc:
        raise PreprocessingError(f"{field_name} must contain exactly three numbers") from exc
    if len(components) != 3:
        raise PreprocessingError(f"{field_name} must contain exactly three numbers")
    vector = tuple(_finite_number(component, field_name) for component in components)
    norm = math.sqrt(sum(component * component for component in vector))
    if not math.isfinite(norm) or norm <= GEOMETRY_NORM_EPS:
        raise PreprocessingError(f"{field_name} must be a finite nonzero vector")
    return vector  # type: ignore[return-value]


@dataclass(frozen=True)
class Observation:
    """One validated relative-photometry observation.

    Geometry vectors are asteroid-to-Sun and asteroid-to-observer Cartesian
    vectors in ecliptic J2000 coordinates, expressed in astronomical units.
    """

    time_jd: float
    relative_brightness: float
    sun_asteroid_ecliptic_j2000_au: tuple[float, float, float]
    observer_asteroid_ecliptic_j2000_au: tuple[float, float, float]
    measured_error: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_jd", _finite_number(self.time_jd, "time_jd", positive=True))
        object.__setattr__(
            self,
            "relative_brightness",
            _finite_number(self.relative_brightness, "relative_brightness", positive=True),
        )
        object.__setattr__(
            self,
            "sun_asteroid_ecliptic_j2000_au",
            _geometry_vector(
                self.sun_asteroid_ecliptic_j2000_au,
                "sun_asteroid_ecliptic_j2000_au",
            ),
        )
        object.__setattr__(
            self,
            "observer_asteroid_ecliptic_j2000_au",
            _geometry_vector(
                self.observer_asteroid_ecliptic_j2000_au,
                "observer_asteroid_ecliptic_j2000_au",
            ),
        )
        if self.measured_error is not None:
            object.__setattr__(
                self,
                "measured_error",
                _finite_number(self.measured_error, "measured_error", positive=True),
            )


@dataclass(frozen=True)
class ObservationEpoch:
    """An explicitly identified source epoch with one or more observations."""

    epoch_id: str
    observations: tuple[Observation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.epoch_id, str) or not self.epoch_id.strip():
            raise PreprocessingError("epoch_id must be a nonempty string")
        if self.epoch_id != self.epoch_id.strip():
            raise PreprocessingError("epoch_id may not have leading or trailing whitespace")
        try:
            observations = tuple(self.observations)
        except TypeError as exc:
            raise PreprocessingError("observations must be a nonempty sequence") from exc
        if not observations:
            raise PreprocessingError("each epoch must contain at least one observation")
        if not all(isinstance(item, Observation) for item in observations):
            raise PreprocessingError("every epoch member must be an Observation")
        object.__setattr__(self, "observations", observations)


@dataclass(frozen=True)
class KnownPeriod:
    """A known rotation period with an explicit source and optional uncertainty."""

    hours: float
    provenance: str
    uncertainty_hours: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "hours", _finite_number(self.hours, "period hours", positive=True))
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise PreprocessingError("known periods require nonempty provenance")
        if self.provenance != self.provenance.strip():
            raise PreprocessingError("period provenance may not have surrounding whitespace")
        if self.uncertainty_hours is not None:
            object.__setattr__(
                self,
                "uncertainty_hours",
                _finite_number(
                    self.uncertainty_hours,
                    "period uncertainty_hours",
                    positive=True,
                ),
            )
