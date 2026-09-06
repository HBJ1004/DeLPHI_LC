"""Period-folded, source-epoch-preserving tokenizer for the K3 scorer."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Sequence

import numpy as np

from ..v2.preprocessing import KnownPeriod, ObservationEpoch
from .config import K3TokenizerConfig

K3_TOKENIZER_NAME = "delphi.k3-phase-tokenizer"
K3_TOKENIZER_VERSION = "1.0.0"
K3_PHASE_FEATURE_NAMES = (
    "centered_log_flux_mean",
    "centered_log_flux_scatter",
    "log1p_observation_count",
    "within_bin_phase_span_fraction",
    "mean_relative_photometric_error",
    "measured_error_fraction",
)
K3_GEOMETRY_FEATURE_NAMES = (
    "sun_direction_x",
    "sun_direction_y",
    "sun_direction_z",
    "sun_distance_au",
    "observer_direction_x",
    "observer_direction_y",
    "observer_direction_z",
    "observer_distance_au",
)
K3_EPOCH_FEATURE_NAMES = tuple(
    name
    for harmonic in range(1, 9)
    for name in (f"fourier_h{harmonic}_cos", f"fourier_h{harmonic}_sin")
) + (
    "log_flux_peak_to_peak",
    "occupied_phase_fraction",
    "log1p_observation_count",
    "log1p_duration_days",
    "normalized_log_period_hours",
    "measured_error_fraction",
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


_SCHEMA_PAYLOAD = MappingProxyType(
    {
        "schema": K3_TOKENIZER_NAME,
        "version": K3_TOKENIZER_VERSION,
        "coordinate_frame": "asteroid_centric_ecliptic_j2000",
        "phase_bins": 64,
        "fourier_harmonics": 8,
        "phase_reference": "minimum_jd_across_original_object_epochs",
        "epoch_policy": "preserve_supplied_epoch_ids_no_time_regrouping",
        "brightness": "natural_log_then_per_epoch_center_without_amplitude_rescaling",
        "observation_policy": "every_observation_contributes_to_exactly_one_phase_bin",
        "minimum_source_epoch_observations": 2,
        "phase_features": K3_PHASE_FEATURE_NAMES,
        "geometry_features": K3_GEOMETRY_FEATURE_NAMES,
        "epoch_features": K3_EPOCH_FEATURE_NAMES,
        "dtype": "float32",
    }
)
K3_TOKENIZER_SCHEMA_SHA256 = hashlib.sha256(
    _canonical_json(dict(_SCHEMA_PAYLOAD)).encode("ascii")
).hexdigest()


class K3TokenizationError(ValueError):
    """Raised when fixed-period phase tokenization cannot be performed."""


@dataclass(frozen=True)
class K3TokenizedObject:
    phase_features: np.ndarray
    phase_mask: np.ndarray
    geometry_features: np.ndarray
    epoch_features: np.ndarray
    epoch_mask: np.ndarray
    epoch_ids: tuple[str, ...]
    observation_counts: tuple[int, ...]
    period_hours: float
    period_provenance: str
    observation_time_origin_jd: float
    tokenizer_schema_sha256: str = K3_TOKENIZER_SCHEMA_SHA256

    def __post_init__(self) -> None:
        phase = np.asarray(self.phase_features)
        mask = np.asarray(self.phase_mask)
        geometry = np.asarray(self.geometry_features)
        epoch = np.asarray(self.epoch_features)
        epoch_mask = np.asarray(self.epoch_mask)
        expected = (len(self.epoch_ids), 64)
        if phase.shape != (*expected, len(K3_PHASE_FEATURE_NAMES)):
            raise K3TokenizationError("phase feature shape does not match the tokenizer schema")
        if mask.shape != expected or geometry.shape != (*expected, len(K3_GEOMETRY_FEATURE_NAMES)):
            raise K3TokenizationError("phase mask/geometry shape does not match the tokenizer schema")
        if epoch.shape != (len(self.epoch_ids), len(K3_EPOCH_FEATURE_NAMES)):
            raise K3TokenizationError("epoch feature shape does not match the tokenizer schema")
        if epoch_mask.shape != (len(self.epoch_ids),) or not np.all(epoch_mask):
            raise K3TokenizationError("tokenized objects contain only real, unpadded epochs")
        if mask.dtype != np.bool_ or epoch_mask.dtype != np.bool_:
            raise K3TokenizationError("tokenizer masks must have boolean dtype")
        if len(self.observation_counts) != len(self.epoch_ids) or any(count < 2 for count in self.observation_counts):
            raise K3TokenizationError("each retained epoch requires at least two observations")
        if int(np.sum(np.expm1(phase[..., 2]).round())) != sum(self.observation_counts):
            raise K3TokenizationError("phase-bin counts do not conserve observations")
        if not all(np.all(np.isfinite(value)) for value in (phase, geometry, epoch)):
            raise K3TokenizationError("tokenized arrays must be finite")
        if not math.isfinite(self.period_hours) or self.period_hours <= 0 or not self.period_provenance:
            raise K3TokenizationError("tokenized period and provenance are required")
        for value in (phase, mask, geometry, epoch, epoch_mask):
            value.setflags(write=False)


def _direction_and_distance(vector: Sequence[float]) -> tuple[np.ndarray, float]:
    values = np.asarray(vector, dtype=np.float64)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise K3TokenizationError("geometry vectors must be finite three-vectors")
    distance = float(np.linalg.norm(values))
    if distance <= 1e-12:
        raise K3TokenizationError("geometry vectors must be nonzero")
    return values / distance, distance


def _normalized_log_period(period_hours: float) -> float:
    # Maps the declared 2--200 h primary domain approximately to [-1, 1],
    # while remaining finite and auditable for sensitivity cases outside it.
    midpoint = 0.5 * (math.log(2.0) + math.log(200.0))
    half_range = 0.5 * (math.log(200.0) - math.log(2.0))
    return float(np.clip((math.log(period_hours) - midpoint) / half_range, -3.0, 3.0))


def tokenize_epochs(
    epochs: Sequence[ObservationEpoch],
    *,
    known_period: KnownPeriod,
    config: K3TokenizerConfig = K3TokenizerConfig(),
) -> K3TokenizedObject:
    """Fold original epochs around one object-global reference and bin all data."""
    if not isinstance(config, K3TokenizerConfig):
        raise K3TokenizationError("config must be K3TokenizerConfig")
    if not isinstance(known_period, KnownPeriod):
        raise K3TokenizationError("K3 primary inference requires a KnownPeriod with provenance")
    source_epochs = tuple(epochs)
    if not source_epochs or not all(isinstance(epoch, ObservationEpoch) for epoch in source_epochs):
        raise K3TokenizationError("epochs must be a nonempty sequence of ObservationEpoch values")
    epoch_ids = tuple(epoch.epoch_id for epoch in source_epochs)
    if len(epoch_ids) != len(set(epoch_ids)):
        raise K3TokenizationError("epoch IDs must be unique and are never reconstructed from time gaps")
    if any(len(epoch.observations) < config.minimum_epoch_observations for epoch in source_epochs):
        raise K3TokenizationError(
            f"each original epoch requires at least {config.minimum_epoch_observations} observations"
        )
    origin = min(observation.time_jd for epoch in source_epochs for observation in epoch.observations)
    period_days = known_period.hours / 24.0
    epoch_count, bin_count = len(source_epochs), config.phase_bins
    phase_features = np.zeros(
        (epoch_count, bin_count, len(K3_PHASE_FEATURE_NAMES)), dtype=np.float32
    )
    phase_mask = np.zeros((epoch_count, bin_count), dtype=np.bool_)
    geometry_features = np.zeros(
        (epoch_count, bin_count, len(K3_GEOMETRY_FEATURE_NAMES)), dtype=np.float32
    )
    epoch_features = np.zeros(
        (epoch_count, len(K3_EPOCH_FEATURE_NAMES)), dtype=np.float32
    )
    observation_counts: list[int] = []
    for epoch_index, epoch in enumerate(source_epochs):
        observations = tuple(sorted(epoch.observations, key=lambda value: value.time_jd))
        observation_counts.append(len(observations))
        times = np.asarray([value.time_jd for value in observations], dtype=np.float64)
        brightness = np.asarray([value.relative_brightness for value in observations], dtype=np.float64)
        if not np.all(np.isfinite(brightness)) or np.any(brightness <= 0):
            raise K3TokenizationError("relative brightness must be finite and positive")
        log_flux = np.log(brightness)
        centered = log_flux - float(np.mean(log_flux))
        unwrapped_phase = (times - origin) / period_days
        phase = np.remainder(unwrapped_phase, 1.0)
        bin_indices = np.minimum((phase * bin_count).astype(np.int64), bin_count - 1)
        relative_errors = np.zeros(len(observations), dtype=np.float64)
        has_error = np.zeros(len(observations), dtype=np.float64)
        sun_directions = np.empty((len(observations), 3), dtype=np.float64)
        observer_directions = np.empty_like(sun_directions)
        sun_distances = np.empty(len(observations), dtype=np.float64)
        observer_distances = np.empty(len(observations), dtype=np.float64)
        for index, observation in enumerate(observations):
            sun_directions[index], sun_distances[index] = _direction_and_distance(
                observation.sun_asteroid_ecliptic_j2000_au
            )
            observer_directions[index], observer_distances[index] = _direction_and_distance(
                observation.observer_asteroid_ecliptic_j2000_au
            )
            if observation.measured_error is not None:
                relative_errors[index] = observation.measured_error / observation.relative_brightness
                has_error[index] = 1.0
        for bin_index in np.unique(bin_indices):
            selected = bin_indices == bin_index
            count = int(np.sum(selected))
            selected_phase = phase[selected]
            phase_span = float(np.ptp(selected_phase) * bin_count) if count > 1 else 0.0
            phase_features[epoch_index, bin_index] = (
                float(np.mean(centered[selected])),
                float(np.std(centered[selected], ddof=0)),
                math.log1p(count),
                min(phase_span, 1.0),
                float(np.mean(relative_errors[selected])),
                float(np.mean(has_error[selected])),
            )
            sun_mean = np.mean(sun_directions[selected], axis=0)
            observer_mean = np.mean(observer_directions[selected], axis=0)
            sun_norm, observer_norm = np.linalg.norm(sun_mean), np.linalg.norm(observer_mean)
            if sun_norm <= 1e-12 or observer_norm <= 1e-12:
                raise K3TokenizationError("geometry directions cancel within a phase bin")
            geometry_features[epoch_index, bin_index] = (
                *(sun_mean / sun_norm),
                float(np.mean(sun_distances[selected])),
                *(observer_mean / observer_norm),
                float(np.mean(observer_distances[selected])),
            )
            phase_mask[epoch_index, bin_index] = True
        fourier: list[float] = []
        for harmonic in range(1, config.fourier_harmonics + 1):
            angle = 2.0 * math.pi * harmonic * phase
            fourier.extend(
                (
                    float(2.0 * np.mean(centered * np.cos(angle))),
                    float(2.0 * np.mean(centered * np.sin(angle))),
                )
            )
        epoch_features[epoch_index] = (
            *fourier,
            float(np.ptp(log_flux)),
            float(np.mean(phase_mask[epoch_index])),
            math.log1p(len(observations)),
            math.log1p(float(times[-1] - times[0])),
            _normalized_log_period(known_period.hours),
            float(np.mean(has_error)),
        )
    return K3TokenizedObject(
        phase_features=phase_features,
        phase_mask=phase_mask,
        geometry_features=geometry_features,
        epoch_features=epoch_features,
        epoch_mask=np.ones(epoch_count, dtype=np.bool_),
        epoch_ids=epoch_ids,
        observation_counts=tuple(observation_counts),
        period_hours=known_period.hours,
        period_provenance=known_period.provenance,
        observation_time_origin_jd=origin,
    )


def pad_tokenized_objects(values: Sequence[K3TokenizedObject]) -> dict[str, np.ndarray | tuple[str, ...]]:
    """Pad only the epoch set dimension for deterministic object minibatches."""
    objects = tuple(values)
    if not objects or not all(isinstance(value, K3TokenizedObject) for value in objects):
        raise K3TokenizationError("collation requires at least one K3TokenizedObject")
    max_epochs = max(len(value.epoch_ids) for value in objects)
    batch_size = len(objects)
    phase = np.zeros((batch_size, max_epochs, 64, len(K3_PHASE_FEATURE_NAMES)), dtype=np.float32)
    phase_mask = np.zeros((batch_size, max_epochs, 64), dtype=np.bool_)
    geometry = np.zeros((batch_size, max_epochs, 64, len(K3_GEOMETRY_FEATURE_NAMES)), dtype=np.float32)
    epoch = np.zeros((batch_size, max_epochs, len(K3_EPOCH_FEATURE_NAMES)), dtype=np.float32)
    epoch_mask = np.zeros((batch_size, max_epochs), dtype=np.bool_)
    periods = np.empty(batch_size, dtype=np.float32)
    for index, value in enumerate(objects):
        count = len(value.epoch_ids)
        phase[index, :count] = value.phase_features
        phase_mask[index, :count] = value.phase_mask
        geometry[index, :count] = value.geometry_features
        epoch[index, :count] = value.epoch_features
        epoch_mask[index, :count] = True
        periods[index] = value.period_hours
    return {
        "phase_features": phase,
        "phase_mask": phase_mask,
        "geometry_features": geometry,
        "epoch_features": epoch,
        "epoch_mask": epoch_mask,
        "period_hours": periods,
        "period_provenance": tuple(value.period_provenance for value in objects),
    }
