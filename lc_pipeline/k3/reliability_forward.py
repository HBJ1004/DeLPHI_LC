"""Forward-only renderer for convexinv withheld-observation reliability work.

This module deliberately contains no fitting code.  It reads the *exported*
solution (rather than a rounded console summary), evaluates its fixed shape at
new geometries, and only normalizes a curve after all of its predictions have
been made.  Consequently a held-out flux can be used for scoring but cannot
affect a prediction.

The formulas are a small, direct translation of ``bright.c``, ``matrix.c``,
``blmatrix.c`` and the relative-lightcurve output path in convexinv 0.2.1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


class ReliabilityForwardError(ValueError):
    """A solution, geometry, or score is unsuitable for forward prediction."""


@dataclass(frozen=True)
class ConvexinvSolution:
    """Full-precision values emitted by the patched convexinv exporter."""

    longitude_deg: float
    latitude_deg: float
    period_hours: float
    epoch_jd: float
    phase_deg: float
    phase_a: float
    phase_d: float
    phase_k: float
    lambert: float
    areas: np.ndarray
    normals: np.ndarray


@dataclass(frozen=True)
class ForwardObservation:
    time_jd: float
    flux: float
    sun: tuple[float, float, float]
    observer: tuple[float, float, float]


@dataclass(frozen=True)
class ForwardLightcurve:
    """One source lightcurve. ``relative`` mirrors convexinv's input flag."""

    observations: tuple[ForwardObservation, ...]
    relative: bool


def _finite_scalar(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise ReliabilityForwardError(f"{name} must be finite")
    return value


def _numbers(path: Path) -> list[float]:
    try:
        tokens = path.read_text(encoding="ascii").split()
        values = [float(token) for token in tokens]
    except (OSError, UnicodeError, ValueError) as exc:
        raise ReliabilityForwardError(f"cannot parse numeric export {path}: {exc}") from exc
    if not values or not all(math.isfinite(value) for value in values):
        raise ReliabilityForwardError(f"{path} has missing or non-finite numbers")
    return values


def read_solution_parameters(path: str | Path) -> dict[str, float]:
    """Read the four-line convexinv ``-p`` export without canonicalizing angles.

    In particular, latitude, longitude, and phase are intentionally retained
    as emitted: canonicalizing a pole would change an Euler rotation.
    """
    values = _numbers(Path(path))
    if len(values) != 9:
        raise ReliabilityForwardError("solution parameters must contain exactly nine numbers")
    names = ("longitude_deg", "latitude_deg", "period_hours", "epoch_jd", "phase_deg",
             "phase_a", "phase_d", "phase_k", "lambert")
    result = dict(zip(names, values, strict=True))
    if result["period_hours"] <= 0:
        raise ReliabilityForwardError("period_hours must be positive")
    if result["phase_d"] == 0:
        raise ReliabilityForwardError("phase_d must not be zero")
    if result["lambert"] < 0:
        raise ReliabilityForwardError("lambert must not be negative")
    return result


def read_solution_areas(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read convexinv ``-o`` output and remove its final dark closure facet."""
    try:
        values = [float(token) for token in Path(path).read_text(encoding="ascii").split()]
    except (OSError, UnicodeError, ValueError) as exc:
        raise ReliabilityForwardError(f"cannot parse numeric export {path}: {exc}") from exc
    if not values or not math.isfinite(values[0]):
        raise ReliabilityForwardError("invalid convexinv area export")
    count = int(values[0])
    if count <= 1 or values[0] != count or len(values) != 1 + 4 * count:
        raise ReliabilityForwardError("invalid convexinv area export")
    table = np.asarray(values[1:], dtype=np.float64).reshape(count, 4)
    # The last record is explicitly added by convexinv after the real facets
    # to close a non-convex residual.  It has no physical brightness term.
    physical = table[:-1]
    areas, normals = physical[:, 0].copy(), physical[:, 1:].copy()
    # Do not validate the discarded record.  Convexinv computes its normal as
    # ``-chck/dark``; for a perfectly closed shape ``dark == 0`` yields a
    # non-finite *nonphysical* closure normal that must not poison rendering.
    if np.any(~np.isfinite(areas)) or np.any(~np.isfinite(normals)) or np.any(areas < 0):
        raise ReliabilityForwardError("physical facet export is non-finite or has a negative area")
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths == 0) or not np.allclose(lengths, 1.0, rtol=0, atol=5e-10):
        raise ReliabilityForwardError("facet normals must be unit vectors")
    return areas, normals


def load_solution(parameters_path: str | Path, areas_path: str | Path) -> ConvexinvSolution:
    values = read_solution_parameters(parameters_path)
    areas, normals = read_solution_areas(areas_path)
    return ConvexinvSolution(**values, areas=areas, normals=normals)


def read_lightcurves(path: str | Path) -> tuple[ForwardLightcurve, ...]:
    """Parse the native convexinv input stream, preserving all geometry."""
    values = _numbers(Path(path))
    cursor = 0
    try:
        ncurves = int(values[cursor])
        cursor += 1
        if ncurves <= 0 or values[0] != ncurves:
            raise ReliabilityForwardError("lightcurve count must be a positive integer")
        curves: list[ForwardLightcurve] = []
        for _ in range(ncurves):
            points, absolute = int(values[cursor]), int(values[cursor + 1])
            cursor += 2
            if points <= 0 or absolute not in (0, 1):
                raise ReliabilityForwardError("invalid lightcurve header")
            observations: list[ForwardObservation] = []
            for _ in range(points):
                time_jd, flux = values[cursor], values[cursor + 1]
                sun = tuple(values[cursor + 2:cursor + 5])
                observer = tuple(values[cursor + 5:cursor + 8])
                cursor += 8
                if flux <= 0:
                    raise ReliabilityForwardError("lightcurve flux must be positive")
                observations.append(ForwardObservation(time_jd, flux, sun, observer))
            curves.append(ForwardLightcurve(tuple(observations), relative=not bool(absolute)))
    except IndexError as exc:
        raise ReliabilityForwardError("truncated convexinv lightcurve input") from exc
    if cursor != len(values):
        raise ReliabilityForwardError("trailing data in convexinv lightcurve input")
    return tuple(curves)


def _unit(vector: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ReliabilityForwardError(f"{name} must be three finite values")
    norm = float(np.linalg.norm(result))
    if not math.isfinite(norm) or norm == 0:
        raise ReliabilityForwardError(f"{name} must have non-zero length")
    return result / norm


def _rotation(solution: ConvexinvSolution, time_jd: float) -> np.ndarray:
    _finite_scalar(time_jd, "time_jd")
    # Corresponds to fmat @ Blmat.  beta is convexinv's pole colatitude.
    beta = math.radians(90.0 - solution.latitude_deg)
    longitude = math.radians(solution.longitude_deg)
    phase = math.fmod((24.0 * 2.0 * math.pi / solution.period_hours) *
                      (time_jd - solution.epoch_jd) + math.radians(solution.phase_deg),
                      2.0 * math.pi)
    cb, sb, cl, sl = math.cos(beta), math.sin(beta), math.cos(longitude), math.sin(longitude)
    bl = np.array(((cb * cl, cb * sl, -sb), (-sl, cl, 0.0),
                   (sb * cl, sb * sl, cb)), dtype=np.float64)
    cf, sf = math.cos(phase), math.sin(phase)
    fmat = np.array(((cf, sf, 0.0), (-sf, cf, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64)
    return fmat @ bl


def phase_function(solution: ConvexinvSolution, sun: Sequence[float], observer: Sequence[float]) -> float:
    """The convexinv exp-linear phase function for a geometry."""
    cosine = float(np.clip(np.dot(_unit(sun, "sun"), _unit(observer, "observer")), -1.0, 1.0))
    alpha = math.acos(cosine)
    value = 1.0 + solution.phase_a * math.exp(-alpha / solution.phase_d) + solution.phase_k * alpha
    if not math.isfinite(value):
        raise ReliabilityForwardError("phase function is non-finite")
    return value


def facet_brightness(solution: ConvexinvSolution, observation: ForwardObservation, *, include_phase: bool = False) -> float:
    """Evaluate fixed Lommel--Seeliger + Lambert facet brightness at one time."""
    rotation = _rotation(solution, observation.time_jd)
    observer = rotation @ _unit(observation.observer, "observer")
    sun = rotation @ _unit(observation.sun, "sun")
    mu = solution.normals @ observer
    mu0 = solution.normals @ sun
    visible = (mu > 1e-8) & (mu0 > 1e-8)
    if not np.any(visible):
        raw = 0.0
    else:
        m, m0 = mu[visible], mu0[visible]
        scatter = m * m0 * (solution.lambert + 1.0 / (m + m0))
        raw = float(np.dot(solution.areas[visible], scatter))
    if not math.isfinite(raw) or raw < 0:
        raise ReliabilityForwardError("non-finite forward brightness")
    return raw * phase_function(solution, observation.sun, observation.observer) if include_phase else raw


def predict_curve(solution: ConvexinvSolution, observations: Iterable[ForwardObservation], *, include_phase: bool = False) -> np.ndarray:
    values = np.asarray([facet_brightness(solution, item, include_phase=include_phase) for item in observations], dtype=np.float64)
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise ReliabilityForwardError("prediction has no finite points")
    return values


def normalize_mean(values: Sequence[float] | np.ndarray, *, name: str = "curve") -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or len(result) == 0 or not np.all(np.isfinite(result)):
        raise ReliabilityForwardError(f"{name} must be a non-empty finite vector")
    mean = float(np.mean(result))
    if not math.isfinite(mean) or mean == 0:
        raise ReliabilityForwardError(f"{name} has zero or non-finite mean")
    return result / mean


def native_relative_shape(solution: ConvexinvSolution, observations: Iterable[ForwardObservation]) -> np.ndarray:
    """Shape of native relative output after phase division and curve scaling.

    ``mrqcof`` stores ``raw * phase`` but output divides by phase before a
    curve-wide scale.  The scale is immaterial after mean normalization, so
    this returns normalized raw scattering brightness exactly for comparison.
    """
    return normalize_mean(predict_curve(solution, observations, include_phase=False), name="forward prediction")


def native_output_shape(solution: ConvexinvSolution, observations: Iterable[ForwardObservation], *, relative: bool) -> np.ndarray:
    """Mean-normalized native output shape for either convexinv input flag.

    Native output removes the fitted phase function only for relative input
    curves.  Calibrated (``absolute == 1`` in the input stream) curves retain
    ``Yout``, which includes phase; normalize only after taking that branch.
    """
    return normalize_mean(
        predict_curve(solution, observations, include_phase=not relative),
        name="forward prediction",
    )


def holdout_shape_rmse(solution: ConvexinvSolution, curves: Iterable[ForwardLightcurve]) -> float:
    """Forward-only primary metric: per-curve MSE, mean curves, then sqrt."""
    errors: list[float] = []
    for curve in curves:
        predicted = native_output_shape(solution, curve.observations, relative=curve.relative)
        observed = normalize_mean([item.flux for item in curve.observations], name="withheld flux")
        if len(predicted) != len(observed):  # defensive; iterable may be exotic
            raise ReliabilityForwardError("prediction/observation length mismatch")
        errors.append(float(np.mean(np.square(predicted - observed))))
    if not errors or not all(math.isfinite(value) for value in errors):
        raise ReliabilityForwardError("holdout contains no finite lightcurves")
    return math.sqrt(float(np.mean(errors)))
