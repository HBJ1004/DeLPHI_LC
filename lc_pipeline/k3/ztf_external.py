"""Offline, fail-closed conversion of enriched Fink photometry for K3.

No ephemeris is fabricated here: callers must provide cached, hash-bound JPL
Horizons asteroid-centric ecliptic-J2000 vectors for every retained exposure.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..v2.preprocessing import Observation, ObservationEpoch


class ZTFExternalError(ValueError):
    """Raised when external survey data are not physically auditable."""


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _vector(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ZTFExternalError(f"{name} must be a three-vector")
    try:
        vector = tuple(float(component) for component in value)
    except (TypeError, ValueError) as exc:
        raise ZTFExternalError(f"{name} must be numeric") from exc
    if not all(math.isfinite(component) for component in vector) or np.linalg.norm(vector) <= 0:
        raise ZTFExternalError(f"{name} must be finite and nonzero")
    return vector


def _angle(left: Sequence[float], right: Sequence[float]) -> float:
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    return math.degrees(math.acos(float(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def _equatorial(vector: Sequence[float]) -> np.ndarray:
    # IAU J2000 mean obliquity, sufficient only for an input-frame audit.
    epsilon = math.radians(23.4392911)
    x, y, z = vector
    return np.asarray((x, math.cos(epsilon) * y - math.sin(epsilon) * z, math.sin(epsilon) * y + math.cos(epsilon) * z))


@dataclass(frozen=True)
class HorizonsCache:
    query_metadata: Mapping[str, object]
    rows: tuple[Mapping[str, object], ...]
    query_metadata_sha256: str
    rows_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "HorizonsCache":
        required = {"source", "query_metadata", "rows", "query_metadata_sha256", "rows_sha256"}
        if set(value) != required or value.get("source") != "JPL Horizons":
            raise ZTFExternalError("Horizons cache requires JPL Horizons source and complete metadata")
        metadata, rows = value["query_metadata"], value["rows"]
        if not isinstance(metadata, Mapping) or not isinstance(rows, list) or not rows:
            raise ZTFExternalError("Horizons cache metadata and rows are required")
        if value["query_metadata_sha256"] != _hash(metadata) or value["rows_sha256"] != _hash(rows):
            raise ZTFExternalError("Horizons cache metadata or vector hash mismatch")
        return cls(metadata, tuple(rows), str(value["query_metadata_sha256"]), str(value["rows_sha256"]))


def fink_rows_to_epoch(
    object_id: str, rows: Sequence[Mapping[str, object]], horizons_cache: Mapping[str, object], *,
    norm_relative_tolerance: float = .02, phase_tolerance_deg: float = 2.0,
    radec_tolerance_deg: float = 2.0,
) -> ObservationEpoch:
    """Return one r-band epoch after validating cached vector geometry offline."""
    if not isinstance(object_id, str) or not object_id.strip():
        raise ZTFExternalError("object_id is required")
    if not 0 < norm_relative_tolerance < 1 or phase_tolerance_deg <= 0 or radec_tolerance_deg <= 0:
        raise ZTFExternalError("geometry tolerances must be positive")
    cache = HorizonsCache.from_mapping(horizons_cache)
    indexed: dict[float, Mapping[str, object]] = {}
    for row in cache.rows:
        try:
            jd = float(row["jd"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ZTFExternalError("Horizons vector row requires JD") from exc
        if not math.isfinite(jd) or jd in indexed:
            raise ZTFExternalError("Horizons vector JDs must be finite and unique")
        indexed[jd] = row
    retained: list[Mapping[str, object]] = []
    for row in rows:
        try:
            fid, mag, sigma = int(row["fid"]), float(row["magpsf"]), float(row["sigmapsf"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ZTFExternalError("Fink row requires fid, magpsf, and sigmapsf") from exc
        if fid == 2 and math.isfinite(mag) and math.isfinite(sigma) and 0 < sigma <= .2:
            retained.append(row)
    if not retained:
        raise ZTFExternalError("no finite r-band Fink detections satisfy 0 < sigmapsf <= 0.2")
    magnitudes = np.asarray([float(row["magpsf"]) for row in retained])
    reference_magnitude = float(np.median(magnitudes))
    observations: list[Observation] = []
    for row in retained:
        try:
            jd = float(row["jd"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ZTFExternalError("retained Fink row requires finite jd") from exc
        matching = [key for key in indexed if abs(key - jd) <= 1e-9]
        if len(matching) != 1:
            raise ZTFExternalError("every retained Fink JD requires exactly one cached Horizons vector")
        ephemeris = indexed[matching[0]]
        sun = _vector(ephemeris.get("asteroid_to_sun_ecliptic_j2000_au"), "asteroid_to_sun")
        observer = _vector(ephemeris.get("asteroid_to_earth_ecliptic_j2000_au"), "asteroid_to_earth")
        try:
            dhelio, dobs, phase = float(row["Dhelio"]), float(row["Dobs"]), float(row["phase"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ZTFExternalError("retained Fink row requires Dhelio, Dobs, and phase") from exc
        if not all(math.isfinite(value) and value > 0 for value in (dhelio, dobs)) or not math.isfinite(phase):
            raise ZTFExternalError("Fink geometry scalars must be finite")
        if abs(np.linalg.norm(sun) - dhelio) / dhelio > norm_relative_tolerance or abs(np.linalg.norm(observer) - dobs) / dobs > norm_relative_tolerance:
            raise ZTFExternalError("cached Horizons vector norm disagrees with Fink distance")
        if abs(_angle(sun, observer) - phase) > phase_tolerance_deg:
            raise ZTFExternalError("cached Horizons vector angle disagrees with Fink phase")
        if "ra" in row or "dec" in row:
            if "ra" not in row or "dec" not in row:
                raise ZTFExternalError("RA and DEC must be supplied together")
            ra, dec = float(row["ra"]), float(row["dec"])
            direction = _equatorial(-np.asarray(observer)); direction /= np.linalg.norm(direction)
            expected = np.asarray((math.cos(math.radians(dec)) * math.cos(math.radians(ra)), math.cos(math.radians(dec)) * math.sin(math.radians(ra)), math.sin(math.radians(dec))))
            if _angle(direction, expected) > radec_tolerance_deg:
                raise ZTFExternalError("cached Horizons observer vector disagrees with Fink RA/DEC")
        flux = 10 ** (-.4 * (float(row["magpsf"]) - reference_magnitude))
        flux_error = flux * math.log(10) / 2.5 * float(row["sigmapsf"])
        observations.append(Observation(time_jd=jd, relative_brightness=flux, measured_error=flux_error, sun_asteroid_ecliptic_j2000_au=sun, observer_asteroid_ecliptic_j2000_au=observer))
    return ObservationEpoch(f"ztf-r-{object_id}", tuple(observations))


def prepared_ztf_object(object_id: str, rows: Sequence[Mapping[str, object]], horizons_cache: Mapping[str, object], *, known_period_hours: float, period_provenance: str) -> dict[str, object]:
    """Create a label-free frozen external object document; never queries a service."""
    epoch = fink_rows_to_epoch(object_id, rows, horizons_cache)
    return {"schema": "delphi.k3-ztf-prepared.v1", "object_id": object_id, "known_period_hours": known_period_hours, "period_provenance": period_provenance, "epochs": [{"epoch_id": epoch.epoch_id, "observations": [observation.__dict__ for observation in epoch.observations]}], "horizons_query_metadata_sha256": HorizonsCache.from_mapping(horizons_cache).query_metadata_sha256, "horizons_rows_sha256": HorizonsCache.from_mapping(horizons_cache).rows_sha256}
