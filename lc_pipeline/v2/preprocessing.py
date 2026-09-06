"""Canonical observation preprocessing contract for DeLPHI V2.

This module is the only supported path from validated lightcurve observations
to fixed-shape model tensors.  It is intentionally independent of PyTorch so
the exact same builder can be used by training, validation, and inference.

The contract is deliberately lossless with respect to the scientifically
meaningful inputs:

* source epoch membership and epoch identifiers are preserved;
* observations are stable-sorted by time *within* an epoch only;
* oversized epochs are split into contiguous, deterministic chunks;
* padding is zero-filled and masked, never populated by repeated observations;
* relative brightness is retained without per-epoch amplitude normalization;
* asteroid-centric ecliptic-J2000 geometry is retained as directions and
  distances; and
* a supplied period carries explicit provenance and nullable uncertainty.

Geometry ablation must be requested explicitly through :class:`GeometryMode`.
Even in an ablation, geometry is validated and retained in
``geometry_descriptors`` for auditability while the model feature channels are
zeroed.
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


@dataclass(frozen=True)
class PreprocessingConfig:
    """Fixed-shape tensor dimensions and explicit geometry policy."""

    max_epoch_slots: int
    max_observations_per_chunk: int
    geometry_mode: GeometryMode

    def __post_init__(self) -> None:
        if isinstance(self.max_epoch_slots, bool) or not isinstance(self.max_epoch_slots, int):
            raise PreprocessingError("max_epoch_slots must be a positive integer")
        if self.max_epoch_slots <= 0:
            raise PreprocessingError("max_epoch_slots must be a positive integer")
        if isinstance(self.max_observations_per_chunk, bool) or not isinstance(
            self.max_observations_per_chunk, int
        ):
            raise PreprocessingError("max_observations_per_chunk must be a positive integer")
        if self.max_observations_per_chunk <= 0:
            raise PreprocessingError("max_observations_per_chunk must be a positive integer")
        if not isinstance(self.geometry_mode, GeometryMode):
            raise PreprocessingError(
                "geometry_mode must explicitly be GeometryMode.ENABLED or GeometryMode.DISABLED"
            )


@dataclass(frozen=True)
class PreprocessedObservations:
    """Fixed-shape, audit-ready result of canonical preprocessing."""

    tokens: np.ndarray
    geometry_descriptors: np.ndarray
    epoch_descriptors: np.ndarray
    observation_mask: np.ndarray
    error_mask: np.ndarray
    epoch_mask: np.ndarray
    period_values: np.ndarray
    period_mask: np.ndarray
    epoch_ids: tuple[str | None, ...]
    chunk_ids: tuple[str | None, ...]
    period_provenance: str | None
    geometry_mode: GeometryMode
    observation_time_origin_jd: float
    schema: TokenizerSchema = TOKENIZER_SCHEMA


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _direction_and_distance(
    vector: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    distance = math.sqrt(sum(component * component for component in vector))
    # Validation in Observation guarantees this cannot be zero or non-finite.
    return (
        vector[0] / distance,
        vector[1] / distance,
        vector[2] / distance,
        distance,
    )


def _chunk_id(epoch_id: str, chunk_index: int) -> str:
    return f"{epoch_id}::chunk-{chunk_index:04d}"


def build_observation_tensors(
    epochs: Sequence[ObservationEpoch],
    *,
    config: PreprocessingConfig,
    known_period: KnownPeriod | None = None,
) -> PreprocessedObservations:
    """Build the canonical fixed-shape tensors used by training and inference.

    The function never truncates or samples observations.  If deterministic
    chunking needs more slots than ``config.max_epoch_slots``, preprocessing
    fails and the caller must choose a larger declared shape.
    """

    if not isinstance(config, PreprocessingConfig):
        raise PreprocessingError("config must be a PreprocessingConfig")
    try:
        source_epochs = tuple(epochs)
    except TypeError as exc:
        raise PreprocessingError("epochs must be a nonempty sequence") from exc
    if not source_epochs:
        raise PreprocessingError("at least one observation epoch is required")
    if not all(isinstance(epoch, ObservationEpoch) for epoch in source_epochs):
        raise PreprocessingError("every input epoch must be an ObservationEpoch")
    epoch_ids = [epoch.epoch_id for epoch in source_epochs]
    if len(epoch_ids) != len(set(epoch_ids)):
        raise PreprocessingError("epoch_id values must be unique within an object")
    if known_period is not None and not isinstance(known_period, KnownPeriod):
        raise PreprocessingError("known_period must be KnownPeriod or None")

    sorted_epochs: list[tuple[ObservationEpoch, tuple[Observation, ...]]] = []
    for epoch in source_epochs:
        # Python's sort is stable: equal-time measurements retain source order.
        sorted_observations = tuple(sorted(epoch.observations, key=lambda item: item.time_jd))
        sorted_epochs.append((epoch, sorted_observations))

    # DAMIT's uncalibrated epochs can use radically different positive flux
    # units (from order unity to >10^7). Convert every object to one
    # dimensionless relative-flux scale. This is not per-epoch normalization:
    # relative levels and amplitudes among the object's epochs remain intact.
    brightness_scale = float(
        np.median(
            [
                observation.relative_brightness
                for _, sorted_observations in sorted_epochs
                for observation in sorted_observations
            ]
        )
    )
    if not math.isfinite(brightness_scale) or brightness_scale <= 0.0:
        raise PreprocessingError("object-global brightness scale must be finite and positive")

    chunk_plan: list[tuple[ObservationEpoch, tuple[Observation, ...], int, float, float, int]] = []
    chunk_size = config.max_observations_per_chunk
    for epoch, sorted_observations in sorted_epochs:
        brightness = [item.relative_brightness for item in sorted_observations]
        log_brightness = [math.log(value / brightness_scale) for value in brightness]
        raw_amplitude = max(log_brightness) - min(log_brightness)
        epoch_duration = sorted_observations[-1].time_jd - sorted_observations[0].time_jd
        epoch_count = len(sorted_observations)
        for chunk_index, start in enumerate(range(0, epoch_count, chunk_size)):
            chunk_plan.append(
                (
                    epoch,
                    sorted_observations[start : start + chunk_size],
                    chunk_index,
                    raw_amplitude,
                    epoch_duration,
                    epoch_count,
                )
            )

    if len(chunk_plan) > config.max_epoch_slots:
        raise PreprocessingError(
            "deterministic epoch chunking requires "
            f"{len(chunk_plan)} slots, exceeding max_epoch_slots={config.max_epoch_slots}; "
            "V2 preprocessing never drops observations"
        )

    n_slots = config.max_epoch_slots
    n_points = config.max_observations_per_chunk
    tokens = np.zeros((n_slots, n_points, len(TOKEN_FEATURE_NAMES)), dtype=np.float32)
    geometry = np.zeros((n_slots, n_points, len(GEOMETRY_FEATURE_NAMES)), dtype=np.float32)
    epoch_descriptors = np.zeros((n_slots, len(EPOCH_DESCRIPTOR_NAMES)), dtype=np.float32)
    observation_mask = np.zeros((n_slots, n_points), dtype=np.bool_)
    error_mask = np.zeros((n_slots, n_points), dtype=np.bool_)
    epoch_mask = np.zeros(n_slots, dtype=np.bool_)
    period_values = np.zeros(len(PERIOD_VALUE_NAMES), dtype=np.float32)
    period_mask = np.zeros(len(PERIOD_VALUE_NAMES), dtype=np.bool_)
    output_epoch_ids: list[str | None] = [None] * n_slots
    output_chunk_ids: list[str | None] = [None] * n_slots

    time_origin = min(
        observation.time_jd
        for _, sorted_observations in sorted_epochs
        for observation in sorted_observations
    )
    # Only an evidence-derived, non-ambiguous period may phase-fold the model
    # input.  Explicit DAMIT solution periods are labels and must never be
    # transformed into deployable sequence features.
    inferred_phase_period_days: float | None = None
    if known_period is not None and known_period.provenance == "delphi-v2-period:ok":
        inferred_phase_period_days = known_period.hours / 24.0

    for slot, (
        epoch,
        observations,
        chunk_index,
        raw_amplitude,
        epoch_duration,
        epoch_count,
    ) in enumerate(chunk_plan):
        epoch_start = min(item.time_jd for item in epoch.observations)
        epoch_mask[slot] = True
        output_epoch_ids[slot] = epoch.epoch_id
        output_chunk_ids[slot] = _chunk_id(epoch.epoch_id, chunk_index)
        epoch_descriptors[slot] = (
            raw_amplitude,
            math.log1p(epoch_duration),
            math.log1p(float(epoch_count)),
        )

        for row, observation in enumerate(observations):
            sun = _direction_and_distance(observation.sun_asteroid_ecliptic_j2000_au)
            observer = _direction_and_distance(observation.observer_asteroid_ecliptic_j2000_au)
            geometry_row = (*sun, *observer)
            geometry[slot, row] = geometry_row
            measured_error = (
                observation.measured_error if observation.measured_error is not None else 0.0
            )
            has_error = observation.measured_error is not None
            tokens[slot, row, :5] = (
                math.log1p(observation.time_jd - time_origin),
                math.log1p(observation.time_jd - epoch_start),
                math.log(observation.relative_brightness / brightness_scale),
                measured_error,
                float(has_error),
            )
            if config.geometry_mode is GeometryMode.ENABLED:
                tokens[slot, row, GEOMETRY_FEATURE_SLICE] = geometry_row
            if inferred_phase_period_days is not None:
                phase = (
                    2.0 * math.pi * (observation.time_jd - time_origin) / inferred_phase_period_days
                )
                tokens[slot, row, 13:18] = (
                    math.sin(phase),
                    math.cos(phase),
                    math.sin(2.0 * phase),
                    math.cos(2.0 * phase),
                    1.0,
                )
            observation_mask[slot, row] = True
            error_mask[slot, row] = has_error

    if known_period is not None:
        period_values[0] = known_period.hours
        period_mask[0] = True
        if known_period.uncertainty_hours is not None:
            period_values[1] = known_period.uncertainty_hours
            period_mask[1] = True

    return PreprocessedObservations(
        tokens=_readonly(tokens),
        geometry_descriptors=_readonly(geometry),
        epoch_descriptors=_readonly(epoch_descriptors),
        observation_mask=_readonly(observation_mask),
        error_mask=_readonly(error_mask),
        epoch_mask=_readonly(epoch_mask),
        period_values=_readonly(period_values),
        period_mask=_readonly(period_mask),
        epoch_ids=tuple(output_epoch_ids),
        chunk_ids=tuple(output_chunk_ids),
        period_provenance=known_period.provenance if known_period is not None else None,
        geometry_mode=config.geometry_mode,
        observation_time_origin_jd=time_origin,
    )
