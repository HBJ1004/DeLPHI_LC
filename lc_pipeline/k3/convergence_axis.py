"""Interpret unconstrained angular coordinates printed by ``convexinv``.

The convergence follow-up uses this adapter only for trusted solver output.
It does not change the input-coordinate contract for public APIs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

AXIS_INTERPRETATION_VERSION = "delphi.k3-convergence-axis.v1"


class ConvergenceAxisError(ValueError):
    """Raised when a solver-reported angular direction cannot be interpreted."""


@dataclass(frozen=True)
class DecodedConvergenceAxis:
    """Raw solver angles and their standard-coordinate directed direction."""

    raw_lambda_deg: float
    raw_beta_deg: float
    standard_lambda_deg: float
    standard_beta_deg: float
    directed_unit_vector: tuple[float, float, float]
    interpretation_version: str = AXIS_INTERPRETATION_VERSION


def _finite_angle(value: object, *, name: str) -> float:
    """Accept finite real scalars while rejecting bools and string lookalikes."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConvergenceAxisError(f"{name} must be a finite real scalar")
    converted = float(value)
    if not math.isfinite(converted):
        raise ConvergenceAxisError(f"{name} must be finite")
    return converted


def _degrees_modulo(value: float, period: float) -> float:
    """Reduce finite degrees without using modulo arithmetic on caller state."""
    reduced = math.fmod(value, period)
    if reduced < 0.0:
        reduced += period
    # ``fmod`` normally returns a value below the period, but guard a rare
    # rounded endpoint so the public standard range remains half-open.
    return 0.0 if reduced == period else reduced


def decode_convergence_axis(
    raw_lambda_deg: object, raw_beta_deg: object
) -> DecodedConvergenceAxis:
    """Decode the directed vector represented by unconstrained solver angles.

    ``convexinv`` prints longitude and ``90 - colatitude`` while optimizing
    unconstrained trigonometric parameters.  The printed latitude may therefore
    fall outside the conventional interval.  This function first evaluates the
    represented direction, then derives standard longitude and latitude from
    that vector.  It never clips latitude and it never mutates caller data.
    """
    raw_lambda = _finite_angle(raw_lambda_deg, name="raw_lambda_deg")
    raw_beta = _finite_angle(raw_beta_deg, name="raw_beta_deg")

    # Trigonometric functions are periodic.  Reducing both arguments before
    # conversion makes several-full-turn output stable while preserving the
    # represented direction.
    longitude = math.radians(_degrees_modulo(raw_lambda, 360.0))
    latitude = math.radians(_degrees_modulo(raw_beta, 360.0))
    vector = (
        math.cos(latitude) * math.cos(longitude),
        math.cos(latitude) * math.sin(longitude),
        math.sin(latitude),
    )
    norm = math.hypot(*vector)
    if not math.isfinite(norm) or norm <= 0.0:  # Defensive for future changes.
        raise ConvergenceAxisError("solver angles do not yield a finite direction")
    unit = tuple(component / norm for component in vector)
    standard_lambda = _degrees_modulo(math.degrees(math.atan2(unit[1], unit[0])), 360.0)
    standard_beta = math.degrees(math.atan2(unit[2], math.hypot(unit[0], unit[1])))
    return DecodedConvergenceAxis(
        raw_lambda_deg=raw_lambda,
        raw_beta_deg=raw_beta,
        standard_lambda_deg=standard_lambda,
        standard_beta_deg=standard_beta,
        directed_unit_vector=unit,
    )


__all__ = [
    "AXIS_INTERPRETATION_VERSION",
    "ConvergenceAxisError",
    "DecodedConvergenceAxis",
    "decode_convergence_axis",
]
