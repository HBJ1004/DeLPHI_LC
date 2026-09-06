"""Fixed projective HEALPix grid and deterministic K=3 mode extraction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from astropy_healpix import HEALPix

from ..physics.axial import axial_angular_error_deg
from ..v2.density import HEALPIX_ORDER, pixel_centers_xyz, vector_to_pixel
from .synthetic import canonicalize_axes


class K3GridError(ValueError):
    """Raised for invalid score grids or mode extraction failures."""


@dataclass(frozen=True)
class AxialGrid:
    nside: int
    vectors: np.ndarray
    healpix_indices: np.ndarray
    full_to_axial: np.ndarray
    neighbour_indices: np.ndarray

    def __post_init__(self) -> None:
        expected = 6 * self.nside * self.nside
        if self.vectors.shape != (expected, 3) or self.healpix_indices.shape != (expected,):
            raise K3GridError("axial grid has inconsistent dimensions")
        if self.full_to_axial.shape != (12 * self.nside * self.nside,):
            raise K3GridError("full-to-axial map has inconsistent dimensions")
        if self.neighbour_indices.shape != (expected, 8):
            raise K3GridError("axial neighbour map must contain eight slots per axis")
        if not np.all(np.isfinite(self.vectors)) or not np.allclose(
            np.linalg.norm(self.vectors, axis=1), 1.0, atol=1e-12
        ):
            raise K3GridError("axial grid vectors must be finite and normalized")
        for value in (
            self.vectors,
            self.healpix_indices,
            self.full_to_axial,
            self.neighbour_indices,
        ):
            value.setflags(write=False)


@lru_cache(maxsize=4)
def axial_healpix_grid(nside: int = 32) -> AxialGrid:
    """Return one deterministic representative of every antipodal pixel pair."""
    if isinstance(nside, bool) or not isinstance(nside, int) or nside <= 0:
        raise K3GridError("nside must be a positive integer")
    full = pixel_centers_xyz(nside)
    tolerance = 1e-14
    selected = (full[:, 2] > tolerance) | (
        (np.abs(full[:, 2]) <= tolerance)
        & (
            (full[:, 1] > tolerance)
            | ((np.abs(full[:, 1]) <= tolerance) & (full[:, 0] >= 0))
        )
    )
    healpix_indices = np.flatnonzero(selected).astype(np.int64)
    vectors = np.array(full[healpix_indices], copy=True)
    if len(vectors) != 6 * nside * nside:
        raise K3GridError("HEALPix hemisphere did not yield the expected axial grid size")
    source_to_axial = {int(source): index for index, source in enumerate(healpix_indices)}
    canonical_full = canonicalize_axes(full)
    canonical_pixels = np.asarray(
        [vector_to_pixel(vector, nside) for vector in canonical_full], dtype=np.int64
    )
    try:
        full_to_axial = np.asarray(
            [source_to_axial[int(source)] for source in canonical_pixels], dtype=np.int64
        )
    except KeyError as exc:
        raise K3GridError("could not pair full-sphere HEALPix pixels antipodally") from exc
    healpix = HEALPix(nside=nside, order=HEALPIX_ORDER)
    # astropy-healpix emits an expected invalid warning for the few pixels with
    # seven rather than eight neighbours and encodes the missing slot as -1.
    with np.errstate(invalid="ignore"):
        full_neighbours = np.asarray(
            healpix.neighbours(healpix_indices), dtype=np.int64
        ).T
    neighbour_indices = np.full_like(full_neighbours, -1)
    valid = full_neighbours >= 0
    neighbour_indices[valid] = full_to_axial[full_neighbours[valid]]
    return AxialGrid(nside, vectors, healpix_indices, full_to_axial, neighbour_indices)


@dataclass(frozen=True)
class AxialMode:
    grid_index: int
    healpix_index: int
    axis_xyz: tuple[float, float, float]
    score_logit: float


def _score_order(scores: np.ndarray) -> np.ndarray:
    indices = np.arange(scores.size, dtype=np.int64)
    return np.lexsort((indices, -scores))


def local_maximum_indices(scores: np.ndarray, grid: AxialGrid) -> np.ndarray:
    """Return one-ring maxima with stable lowest-index resolution of ties."""
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(grid.vectors),) or not np.all(np.isfinite(values)):
        raise K3GridError("scores must be one finite value per axial grid point")
    maxima: list[int] = []
    for index, neighbours in enumerate(grid.neighbour_indices):
        neighbours = np.unique(neighbours[neighbours >= 0])
        neighbours = neighbours[neighbours != index]
        if neighbours.size == 0:
            maxima.append(index)
            continue
        neighbour_scores = values[neighbours]
        if np.all(values[index] >= neighbour_scores) and not np.any(
            (values[index] == neighbour_scores) & (neighbours < index)
        ):
            maxima.append(index)
    result = np.asarray(maxima, dtype=np.int64)
    return result[np.lexsort((result, -values[result]))]


def extract_axial_modes(
    scores: np.ndarray,
    *,
    grid: AxialGrid | None = None,
    candidate_count: int = 3,
    minimum_separation_deg: float = 15.0,
) -> tuple[AxialMode, ...]:
    """Select local maxima, suppress nearby axes, and deterministically fill K."""
    selected_grid = axial_healpix_grid() if grid is None else grid
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(selected_grid.vectors),) or not np.all(np.isfinite(values)):
        raise K3GridError("scores must be one finite value per axial grid point")
    if candidate_count != 3:
        raise K3GridError("the locked inference contract requires candidate_count=3")
    if not math.isfinite(minimum_separation_deg) or not 0 < minimum_separation_deg < 90:
        raise K3GridError("minimum separation must lie in (0,90) degrees")
    chosen: list[int] = []

    def consider(index: int) -> None:
        if index in chosen:
            return
        if all(
            float(axial_angular_error_deg(selected_grid.vectors[index], selected_grid.vectors[other]))
            >= minimum_separation_deg
            for other in chosen
        ):
            chosen.append(index)

    for index in local_maximum_indices(values, selected_grid):
        consider(int(index))
        if len(chosen) == candidate_count:
            break
    if len(chosen) < candidate_count:
        for index in _score_order(values):
            consider(int(index))
            if len(chosen) == candidate_count:
                break
    if len(chosen) != candidate_count:
        raise K3GridError("could not extract three separated finite axial modes")
    return tuple(
        AxialMode(
            grid_index=index,
            healpix_index=int(selected_grid.healpix_indices[index]),
            axis_xyz=tuple(float(value) for value in selected_grid.vectors[index]),
            score_logit=float(values[index]),
        )
        for index in chosen
    )


def compatibility_mass(scores: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    """Normalize grid compatibility scores without calling them a posterior."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise K3GridError("compatibility scores must be a finite vector")
    if not math.isfinite(temperature) or temperature <= 0:
        raise K3GridError("temperature must be finite and positive")
    shifted = values / temperature
    shifted -= np.max(shifted)
    mass = np.exp(shifted)
    mass /= np.sum(mass)
    mass.setflags(write=False)
    return mass
