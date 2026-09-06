"""HEALPix probability-density primitives for the DeLPHI V2 pole API."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
from astropy import units as u
from astropy_healpix import HEALPix

from ..physics.directional import directed_angular_error_deg

HEALPIX_ORDER = "nested"


class DensityError(ValueError):
    """Raised for an invalid spherical probability mass function."""


def npix_for_nside(nside: int) -> int:
    if isinstance(nside, bool) or not isinstance(nside, int) or nside <= 0:
        raise DensityError("nside must be a positive integer")
    return 12 * nside * nside


@lru_cache(maxsize=8)
def pixel_centers_xyz(nside: int) -> np.ndarray:
    """Return immutable ecliptic Cartesian centers of nested HEALPix pixels."""
    healpix = HEALPix(nside=nside, order=HEALPIX_ORDER)
    lon, lat = healpix.healpix_to_lonlat(np.arange(npix_for_nside(nside)))
    lon_rad = lon.to_value(u.rad)
    lat_rad = lat.to_value(u.rad)
    centers = np.stack(
        [
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad),
        ],
        axis=1,
    ).astype(np.float64, copy=False)
    centers.setflags(write=False)
    return centers


def vector_to_pixel(vector: Sequence[float], nside: int) -> int:
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise DensityError("vector must be one finite three-vector")
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise DensityError("vector must be nonzero")
    x, y, z = vector / norm
    lon = math.atan2(y, x) % (2 * math.pi)
    lat = math.asin(float(np.clip(z, -1.0, 1.0)))
    healpix = HEALPix(nside=nside, order=HEALPIX_ORDER)
    return int(healpix.lonlat_to_healpix(lon * u.rad, lat * u.rad))


def logits_to_probability_mass(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Convert one finite logit vector to a normalized float64 probability mass."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise DensityError("logits must be a nonempty finite one-dimensional array")
    if not math.isfinite(temperature) or temperature <= 0:
        raise DensityError("temperature must be finite and positive")
    scaled = values / temperature
    scaled -= np.max(scaled)
    mass = np.exp(scaled)
    mass /= mass.sum()
    mass.setflags(write=False)
    return mass


@dataclass(frozen=True)
class HighestDensityRegion:
    mass_target: float
    pixel_indices: tuple[int, ...]
    achieved_mass: float


@dataclass(frozen=True)
class SphericalDensity:
    """Normalized probability mass on a fixed ecliptic-J2000 HEALPix grid."""

    nside: int
    probability_mass: np.ndarray
    calibration_id: str
    coordinate_frame: str = "ecliptic_j2000"
    directionality: str = "directed_spin_vector"
    order: str = HEALPIX_ORDER

    def __post_init__(self) -> None:
        expected = npix_for_nside(self.nside)
        values = np.asarray(self.probability_mass, dtype=np.float64)
        if values.shape != (expected,):
            raise DensityError(f"probability_mass must have shape ({expected},)")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise DensityError("probability_mass must be finite and nonnegative")
        if not np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-12):
            raise DensityError("probability_mass must sum to one within 1e-12")
        if not self.calibration_id:
            raise DensityError("calibration_id is required")
        copy = values.copy()
        copy.setflags(write=False)
        object.__setattr__(self, "probability_mass", copy)

    @property
    def pixel_area_sr(self) -> float:
        return 4.0 * math.pi / self.probability_mass.size

    @property
    def density_per_sr(self) -> np.ndarray:
        values = self.probability_mass / self.pixel_area_sr
        values.setflags(write=False)
        return values

    @property
    def top1_pixel(self) -> int:
        return int(np.argmax(self.probability_mass))

    @property
    def top1_vector(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in pixel_centers_xyz(self.nside)[self.top1_pixel])

    @property
    def normalized_entropy(self) -> float:
        positive = self.probability_mass[self.probability_mass > 0]
        entropy = -float(np.sum(positive * np.log(positive)))
        return entropy / math.log(self.probability_mass.size)

    def highest_density_region(self, mass_target: float) -> HighestDensityRegion:
        if not 0 < mass_target <= 1:
            raise DensityError("mass_target must lie in (0, 1]")
        order = np.argsort(-self.probability_mass, kind="stable")
        cumulative = np.cumsum(self.probability_mass[order])
        count = int(np.searchsorted(cumulative, mass_target, side="left")) + 1
        selected = order[:count]
        return HighestDensityRegion(
            mass_target=float(mass_target),
            pixel_indices=tuple(int(index) for index in selected),
            achieved_mass=float(cumulative[count - 1]),
        )

    def expected_directed_error_deg(self, reference_vector: Sequence[float] | None = None) -> float:
        """Expected directed error to top-1 or a supplied candidate direction."""
        reference = self.top1_vector if reference_vector is None else reference_vector
        errors = directed_angular_error_deg(pixel_centers_xyz(self.nside), reference)
        return float(np.dot(self.probability_mass, errors))
