"""Evidence-based V2 rotation-period inference with explicit uncertainty states."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
from astropy.timeseries import LombScargle

from .preprocessing import KnownPeriod, ObservationEpoch


class PeriodStatus(str, Enum):
    OK = "ok"
    AMBIGUOUS = "ambiguous"
    UNINFORMATIVE = "uninformative"
    PROVIDED = "provided"


@dataclass(frozen=True)
class PeriodSearchConfig:
    min_period_hours: float = 2.0
    max_period_hours: float = 200.0
    n_frequency: int = 4096
    min_points_per_epoch: int = 10
    n_harmonics: int = 1
    evidence_temperature: float = 1.0
    ambiguity_relative_mass: float = 0.15
    uninformative_peak_mass: float = 0.001

    def __post_init__(self) -> None:
        if not (0 < self.min_period_hours < self.max_period_hours):
            raise ValueError("period bounds must be positive and ordered")
        if self.n_frequency < 128 or self.min_points_per_epoch < 3 or self.n_harmonics < 1:
            raise ValueError("period search configuration is too small")
        if self.evidence_temperature <= 0:
            raise ValueError("evidence_temperature must be positive")
        if not 0 < self.ambiguity_relative_mass <= 1 or not 0 < self.uninformative_peak_mass < 1:
            raise ValueError("period probability thresholds must lie in (0, 1]")


@dataclass(frozen=True)
class PeriodCalibration:
    """Inner-fold-only calibration parameters; never fit against outer tests."""

    calibration_id: str
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if not self.calibration_id or not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("period calibration requires an ID and positive finite temperature")


@dataclass(frozen=True)
class PeriodMode:
    period_hours: float
    posterior_mass: float


@dataclass(frozen=True)
class PeriodResultV2:
    status: PeriodStatus
    period_hours: float | None
    posterior_period_hours: np.ndarray | None
    posterior_mass: np.ndarray | None
    modes: tuple[PeriodMode, ...]
    ci68_hours: tuple[float, float] | None
    epoch_ids_used: tuple[str, ...]
    assumed_error_epoch_ids: tuple[str, ...]
    calibration_id: str | None
    period_provenance: str | None


def _epoch_arrays(epoch: ObservationEpoch) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, bool]:
    observations = sorted(epoch.observations, key=lambda row: row.time_jd)
    times = np.asarray([row.time_jd for row in observations], dtype=np.float64)
    brightness = np.asarray([row.relative_brightness for row in observations], dtype=np.float64)
    supplied = [row.measured_error for row in observations]
    all_supplied = all(value is not None for value in supplied)
    errors = np.asarray(supplied, dtype=np.float64) if all_supplied else None
    return times, brightness, errors, all_supplied


def _epoch_fingerprint(epoch: ObservationEpoch) -> str:
    rows = [
        [
            row.time_jd,
            row.relative_brightness,
            row.measured_error,
            *row.sun_asteroid_ecliptic_j2000_au,
            *row.observer_asteroid_ecliptic_j2000_au,
        ]
        for row in sorted(epoch.observations, key=lambda row: row.time_jd)
    ]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _frequency_grid(config: PeriodSearchConfig) -> tuple[np.ndarray, np.ndarray]:
    # A log-spaced period grid gives comparable relative resolution at slow and
    # fast rotation. Frequencies are increasing as LombScargle expects.
    periods = np.geomspace(config.min_period_hours, config.max_period_hours, config.n_frequency)
    frequencies = 24.0 / periods
    order = np.argsort(frequencies)
    return frequencies[order], periods[order]


def _epoch_log_evidence(
    epoch: ObservationEpoch, frequencies: np.ndarray, config: PeriodSearchConfig
) -> tuple[np.ndarray, bool]:
    times, brightness, errors, errors_supplied = _epoch_arrays(epoch)
    if times.size < config.min_points_per_epoch:
        raise ValueError(f"epoch {epoch.epoch_id!r} has too few observations")
    # `standard` power is a likelihood-ratio statistic under the configured
    # Fourier model. It is measured evidence, not a hand-assigned alias score.
    lomb = LombScargle(
        times,
        brightness,
        dy=errors,
        fit_mean=True,
        center_data=True,
        nterms=config.n_harmonics,
    )
    power = lomb.power(frequencies, normalization="standard")
    if not np.all(np.isfinite(power)):
        raise ValueError(f"epoch {epoch.epoch_id!r} yielded non-finite evidence")
    # Map the finite likelihood-ratio statistic to a relative log-evidence.
    # The temperature is fitted only on the frozen inner calibration partition.
    scale = max(float(times.size - (2 * config.n_harmonics + 1)), 1.0)
    return scale * power, errors_supplied


def _posterior_from_log_evidence(log_evidence: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(log_evidence, dtype=np.float64) / temperature
    scaled -= np.max(scaled)
    mass = np.exp(scaled)
    mass /= mass.sum()
    return mass


def _local_modes(periods: np.ndarray, mass: np.ndarray, maximum: int = 8) -> tuple[PeriodMode, ...]:
    local = np.flatnonzero((mass[1:-1] >= mass[:-2]) & (mass[1:-1] >= mass[2:])) + 1
    if not local.size:
        local = np.array([int(np.argmax(mass))])
    ordered = local[np.argsort(-mass[local], kind="stable")][:maximum]
    return tuple(PeriodMode(float(periods[index]), float(mass[index])) for index in ordered)


def _central_interval(periods: np.ndarray, mass: np.ndarray, probability: float = 0.68) -> tuple[float, float]:
    cdf = np.cumsum(mass)
    lower = float(np.interp((1.0 - probability) / 2.0, cdf, periods))
    upper = float(np.interp(1.0 - (1.0 - probability) / 2.0, cdf, periods))
    return lower, upper


def infer_period(
    epochs: Sequence[ObservationEpoch],
    *,
    config: PeriodSearchConfig = PeriodSearchConfig(),
    calibration: PeriodCalibration | None = None,
    known_period: KnownPeriod | None = None,
) -> PeriodResultV2:
    """Infer one period density without injected aliases or duplicate evidence."""
    if known_period is not None:
        interval = (
            (known_period.hours - known_period.uncertainty_hours, known_period.hours + known_period.uncertainty_hours)
            if known_period.uncertainty_hours is not None
            else None
        )
        return PeriodResultV2(
            status=PeriodStatus.PROVIDED,
            period_hours=known_period.hours,
            posterior_period_hours=None,
            posterior_mass=None,
            modes=(PeriodMode(known_period.hours, 1.0),),
            ci68_hours=interval,
            epoch_ids_used=tuple(epoch.epoch_id for epoch in epochs),
            assumed_error_epoch_ids=(),
            calibration_id=None,
            period_provenance=known_period.provenance,
        )
    if not epochs:
        raise ValueError("at least one epoch is required")
    frequencies, periods = _frequency_grid(config)
    evidence = np.zeros_like(frequencies)
    epoch_ids: list[str] = []
    assumed_error_ids: list[str] = []
    fingerprints: set[str] = set()
    for epoch in epochs:
        fingerprint = _epoch_fingerprint(epoch)
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        try:
            epoch_evidence, supplied_errors = _epoch_log_evidence(epoch, frequencies, config)
        except ValueError:
            continue
        evidence += epoch_evidence
        epoch_ids.append(epoch.epoch_id)
        if not supplied_errors:
            assumed_error_ids.append(epoch.epoch_id)
    if not epoch_ids:
        return PeriodResultV2(
            status=PeriodStatus.UNINFORMATIVE,
            period_hours=None,
            posterior_period_hours=None,
            posterior_mass=None,
            modes=(),
            ci68_hours=None,
            epoch_ids_used=(),
            assumed_error_epoch_ids=(),
            calibration_id=calibration.calibration_id if calibration else None,
            period_provenance=None,
        )
    temperature = calibration.temperature if calibration else config.evidence_temperature
    # Present the public density in increasing period order. Lomb--Scargle
    # requires increasing frequency, which is decreasing period.
    period_order = np.argsort(periods)
    periods = periods[period_order]
    evidence = evidence[period_order]
    mass = _posterior_from_log_evidence(evidence, temperature)
    modes = _local_modes(periods, mass)
    top = modes[0]
    ambiguous = any(
        mode.posterior_mass >= top.posterior_mass * config.ambiguity_relative_mass
        and 1.8 <= max(mode.period_hours, top.period_hours) / min(mode.period_hours, top.period_hours) <= 2.2
        for mode in modes[1:]
    )
    status = (
        PeriodStatus.UNINFORMATIVE
        if top.posterior_mass < config.uninformative_peak_mass
        else PeriodStatus.AMBIGUOUS if ambiguous else PeriodStatus.OK
    )
    return PeriodResultV2(
        status=status,
        period_hours=top.period_hours,
        posterior_period_hours=periods.copy(),
        posterior_mass=mass.copy(),
        modes=modes,
        ci68_hours=_central_interval(periods, mass),
        epoch_ids_used=tuple(epoch_ids),
        assumed_error_epoch_ids=tuple(assumed_error_ids),
        calibration_id=calibration.calibration_id if calibration else None,
        period_provenance=None,
    )
