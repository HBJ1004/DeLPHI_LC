"""Batched differentiable convex lightcurve renderer for K3 simulation.

The implementation follows the facet law used by DAMIT-style convex inversion:

``A * (mu * mu0 / (mu + mu0) + c_L * mu * mu0)``

for facets that are simultaneously illuminated and visible.  Geometry vectors
are asteroid-to-Sun and asteroid-to-observer vectors in ecliptic J2000.  The
independent NumPy path is intentionally kept free of Torch so numerical
agreement can be tested rather than asserted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

RENDERER_SCHEMA = "delphi.k3-convex-renderer.v1"
DEFAULT_LAMBERT_COEFFICIENT = 0.1
_EPS = 1e-9


class RendererError(ValueError):
    """Raised when a shape or rendering request violates the contract."""


@dataclass(frozen=True)
class ConvexFacets:
    normals: np.ndarray
    areas: np.ndarray
    source_id: str

    def __post_init__(self) -> None:
        normals = np.asarray(self.normals, dtype=np.float64)
        areas = np.asarray(self.areas, dtype=np.float64)
        if normals.ndim != 2 or normals.shape[1] != 3 or areas.shape != (normals.shape[0],):
            raise RendererError("facet normals/areas must have shapes [F,3] and [F]")
        if normals.shape[0] < 4 or not np.all(np.isfinite(normals)):
            raise RendererError("a convex shape requires at least four finite facet normals")
        norms = np.linalg.norm(normals, axis=1)
        if np.any(norms <= 0) or not np.all(np.isfinite(areas)) or np.any(areas <= 0):
            raise RendererError("facet normals must be nonzero and areas positive")
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise RendererError("shape source_id must be nonempty")
        normalized = normals / norms[:, None]
        normalized.setflags(write=False)
        copied_areas = areas.copy()
        copied_areas.setflags(write=False)
        object.__setattr__(self, "normals", normalized)
        object.__setattr__(self, "areas", copied_areas)


def parse_damit_shape(path: str | Path, *, source_id: str | None = None) -> ConvexFacets:
    """Parse a convex DAMIT vertex/facet file and orient normals outwards."""
    source = Path(path)
    try:
        rows = [line.split() for line in source.read_text(encoding="ascii").splitlines() if line.strip()]
        n_vertices, n_facets = int(rows[0][0]), int(rows[0][1])
        vertices = np.asarray(rows[1 : 1 + n_vertices], dtype=np.float64)[:, :3]
        facets = np.asarray(rows[1 + n_vertices : 1 + n_vertices + n_facets], dtype=np.int64)[:, :3] - 1
    except (OSError, UnicodeDecodeError, IndexError, ValueError) as exc:
        raise RendererError(f"cannot parse DAMIT shape {source}: {exc}") from exc
    if vertices.shape != (n_vertices, 3) or facets.shape != (n_facets, 3):
        raise RendererError(f"truncated DAMIT shape: {source}")
    if np.any(facets < 0) or np.any(facets >= n_vertices):
        raise RendererError(f"facet index outside vertex table: {source}")
    v0, v1, v2 = vertices[facets[:, 0]], vertices[facets[:, 1]], vertices[facets[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    if np.any(areas <= 0):
        raise RendererError(f"shape contains a degenerate facet: {source}")
    normals = cross / np.linalg.norm(cross, axis=1, keepdims=True)
    centroid = vertices.mean(axis=0)
    face_centers = (v0 + v1 + v2) / 3.0
    normals[np.sum(normals * (face_centers - centroid), axis=1) < 0] *= -1
    return ConvexFacets(normals, areas, source_id or source.as_posix())


def fibonacci_directions(count: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Return deterministic equal-area unit-sphere quadrature directions."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 16:
        raise RendererError("facet quadrature requires at least 16 directions")
    index = torch.arange(count, dtype=dtype)
    z = 1.0 - 2.0 * (index + 0.5) / count
    longitude = index * (math.pi * (3.0 - math.sqrt(5.0)))
    radius = torch.sqrt(torch.clamp(1.0 - z.square(), min=0.0))
    return torch.stack((radius * torch.cos(longitude), radius * torch.sin(longitude), z), dim=-1)


def ellipsoid_facet_quadrature(axis_lengths: torch.Tensor, facet_count: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate convex ellipsoids by surface-normal/area quadrature.

    ``axis_lengths`` has shape ``[B,3]``. The area Jacobian for the linear map
    from a unit sphere to an ellipsoid is ``abc * ||A^-T u||``.
    """
    if not isinstance(axis_lengths, torch.Tensor) or axis_lengths.ndim != 2 or axis_lengths.shape[1] != 3:
        raise RendererError("axis_lengths must be a Torch tensor with shape [B,3]")
    if not torch.is_floating_point(axis_lengths) or not bool(torch.isfinite(axis_lengths).all()) or bool(torch.any(axis_lengths <= 0)):
        raise RendererError("axis_lengths must contain finite positive values")
    directions = fibonacci_directions(facet_count, dtype=axis_lengths.dtype).to(axis_lengths.device)
    inverse_mapped = directions.unsqueeze(0) / axis_lengths.unsqueeze(1)
    inverse_norm = torch.linalg.vector_norm(inverse_mapped, dim=-1)
    normals = inverse_mapped / inverse_norm.unsqueeze(-1)
    sphere_area = 4.0 * math.pi / facet_count
    areas = torch.prod(axis_lengths, dim=-1, keepdim=True) * inverse_norm * sphere_area
    return normals, areas


def sample_convex_ellipsoids(
    batch_size: int,
    *,
    generator: torch.Generator,
    facet_count: int = 512,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample principal-axis convex ellipsoids with declared axis ratios."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise RendererError("batch_size must be a positive integer")
    # a=1; b and c avoid nearly spherical examples that carry negligible pole information.
    b = 0.65 + 0.30 * torch.rand(batch_size, generator=generator, dtype=dtype, device=device)
    c_fraction = 0.65 + 0.30 * torch.rand(batch_size, generator=generator, dtype=dtype, device=device)
    axes = torch.stack((torch.ones_like(b), b, b * c_fraction), dim=-1)
    normals, areas = ellipsoid_facet_quadrature(axes, facet_count)
    return normals, areas, axes


def _validated_render_inputs(
    facet_normals: torch.Tensor,
    facet_areas: torch.Tensor,
    poles: torch.Tensor,
    phases_rad: torch.Tensor,
    sun_vectors: torch.Tensor,
    observer_vectors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if poles.ndim != 2 or poles.shape[1] != 3:
        raise RendererError("poles must have shape [B,3]")
    batch_size = poles.shape[0]
    if phases_rad.ndim != 2 or phases_rad.shape[0] != batch_size:
        raise RendererError("phases_rad must have shape [B,N]")
    expected_geometry = (*phases_rad.shape, 3)
    if sun_vectors.shape != expected_geometry or observer_vectors.shape != expected_geometry:
        raise RendererError("Sun and observer geometry must have shape [B,N,3]")
    if facet_normals.ndim == 2:
        facet_normals = facet_normals.unsqueeze(0).expand(batch_size, -1, -1)
    if facet_areas.ndim == 1:
        facet_areas = facet_areas.unsqueeze(0).expand(batch_size, -1)
    if facet_normals.ndim != 3 or facet_normals.shape[0] != batch_size or facet_normals.shape[2] != 3:
        raise RendererError("facet_normals must have shape [F,3] or [B,F,3]")
    if facet_areas.shape != facet_normals.shape[:2]:
        raise RendererError("facet_areas must align with facet_normals")
    tensors = (facet_normals, facet_areas, poles, phases_rad, sun_vectors, observer_vectors)
    if any(not torch.is_floating_point(value) for value in tensors):
        raise RendererError("renderer inputs must have floating-point dtypes")
    if any(value.device != poles.device for value in tensors):
        raise RendererError("renderer inputs must share one device")
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise RendererError("renderer inputs must be finite")
    if bool(torch.any(facet_areas <= 0)):
        raise RendererError("facet areas must be positive")
    normal_norm = torch.linalg.vector_norm(facet_normals, dim=-1, keepdim=True)
    pole_norm = torch.linalg.vector_norm(poles, dim=-1, keepdim=True)
    sun_distance = torch.linalg.vector_norm(sun_vectors, dim=-1, keepdim=True)
    observer_distance = torch.linalg.vector_norm(observer_vectors, dim=-1, keepdim=True)
    if any(bool(torch.any(value <= 1e-12)) for value in (normal_norm, pole_norm, sun_distance, observer_distance)):
        raise RendererError("normals, poles, and geometry vectors must be nonzero")
    return (
        facet_normals / normal_norm,
        facet_areas,
        poles / pole_norm,
        sun_vectors / sun_distance,
        observer_vectors / observer_distance,
    )


def _directions_ecliptic_to_body(
    directions: torch.Tensor, poles: torch.Tensor, phases_rad: torch.Tensor
) -> torch.Tensor:
    """Apply DAMIT's ``Rz(lambda) Ry(pi/2-beta) Rz(phi)`` convention."""
    longitude = torch.atan2(poles[:, 1], poles[:, 0])
    latitude = torch.asin(torch.clamp(poles[:, 2], -1.0, 1.0))
    cos_lon, sin_lon = torch.cos(longitude)[:, None], torch.sin(longitude)[:, None]
    x0 = directions[..., 0] * cos_lon + directions[..., 1] * sin_lon
    y0 = -directions[..., 0] * sin_lon + directions[..., 1] * cos_lon
    z0 = directions[..., 2]
    theta = math.pi / 2.0 - latitude
    cos_theta, sin_theta = torch.cos(theta)[:, None], torch.sin(theta)[:, None]
    x1 = x0 * cos_theta - z0 * sin_theta
    y1 = y0
    z1 = x0 * sin_theta + z0 * cos_theta
    cos_phase, sin_phase = torch.cos(phases_rad), torch.sin(phases_rad)
    x_body = cos_phase * x1 + sin_phase * y1
    y_body = -sin_phase * x1 + cos_phase * y1
    return torch.stack((x_body, y_body, z1), dim=-1)


def render_brightness(
    facet_normals: torch.Tensor,
    facet_areas: torch.Tensor,
    poles: torch.Tensor,
    phases_rad: torch.Tensor,
    sun_vectors: torch.Tensor,
    observer_vectors: torch.Tensor,
    *,
    lambert_coefficient: float | torch.Tensor = DEFAULT_LAMBERT_COEFFICIENT,
    normalize: bool = False,
) -> torch.Tensor:
    """Render batched disk-integrated brightness with gradients through poles."""
    normals, areas, unit_poles, sun, observer = _validated_render_inputs(
        facet_normals, facet_areas, poles, phases_rad, sun_vectors, observer_vectors
    )
    if isinstance(lambert_coefficient, torch.Tensor):
        coefficient = lambert_coefficient.to(device=poles.device, dtype=poles.dtype)
        if coefficient.ndim == 1:
            coefficient = coefficient[:, None, None]
        if coefficient.shape not in ((), (poles.shape[0], 1, 1)):
            raise RendererError("Lambert coefficient tensor must be scalar or shape [B]")
        if not bool(torch.isfinite(coefficient).all()) or bool(torch.any(coefficient < 0)):
            raise RendererError("Lambert coefficient must be finite and nonnegative")
    else:
        if not math.isfinite(lambert_coefficient) or lambert_coefficient < 0:
            raise RendererError("Lambert coefficient must be finite and nonnegative")
        coefficient = lambert_coefficient
    sun_body = _directions_ecliptic_to_body(sun, unit_poles, phases_rad)
    observer_body = _directions_ecliptic_to_body(observer, unit_poles, phases_rad)
    mu0 = torch.clamp(torch.einsum("bnc,bfc->bnf", sun_body, normals), min=0.0)
    mu = torch.clamp(torch.einsum("bnc,bfc->bnf", observer_body, normals), min=0.0)
    product = mu * mu0
    scatter = product / (mu + mu0 + _EPS) + coefficient * product
    brightness = torch.sum(scatter * areas[:, None, :], dim=-1)
    if normalize:
        brightness = brightness / torch.clamp(brightness.mean(dim=1, keepdim=True), min=_EPS)
    return brightness


def normalize_by_epoch(
    brightness: torch.Tensor,
    epoch_index: torch.Tensor,
    observation_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean-normalize every source epoch without discarding observations."""
    if brightness.ndim != 2 or epoch_index.shape != brightness.shape:
        raise RendererError("brightness and epoch_index must have aligned [B,N] shapes")
    if epoch_index.dtype not in (torch.int32, torch.int64):
        raise RendererError("epoch_index must use an integer dtype")
    mask = torch.ones_like(brightness, dtype=torch.bool) if observation_mask is None else observation_mask
    if mask.shape != brightness.shape or mask.dtype is not torch.bool:
        raise RendererError("observation_mask must be boolean and align with brightness")
    result = torch.zeros_like(brightness)
    for batch in range(brightness.shape[0]):
        valid_epochs = torch.unique(epoch_index[batch, mask[batch]], sorted=True)
        for epoch in valid_epochs:
            selected = mask[batch] & (epoch_index[batch] == epoch)
            mean = brightness[batch, selected].mean()
            result[batch, selected] = brightness[batch, selected] / torch.clamp(mean, min=_EPS)
    return result


def render_brightness_numpy(
    facet_normals: np.ndarray,
    facet_areas: np.ndarray,
    pole: Sequence[float],
    phases_rad: np.ndarray,
    sun_vectors: np.ndarray,
    observer_vectors: np.ndarray,
    *,
    lambert_coefficient: float = DEFAULT_LAMBERT_COEFFICIENT,
    normalize: bool = False,
) -> np.ndarray:
    """Independent NumPy reference for one object."""
    normals = np.asarray(facet_normals, dtype=np.float64)
    areas = np.asarray(facet_areas, dtype=np.float64)
    pole_array = np.asarray(pole, dtype=np.float64)
    phases = np.asarray(phases_rad, dtype=np.float64)
    sun = np.asarray(sun_vectors, dtype=np.float64)
    observer = np.asarray(observer_vectors, dtype=np.float64)
    if normals.ndim != 2 or normals.shape[1] != 3 or areas.shape != (normals.shape[0],):
        raise RendererError("NumPy facet arrays have invalid shapes")
    if phases.ndim != 1 or sun.shape != (phases.size, 3) or observer.shape != sun.shape or pole_array.shape != (3,):
        raise RendererError("NumPy pole/phase/geometry arrays have invalid shapes")
    values = (normals, areas, pole_array, phases, sun, observer)
    if any(not np.all(np.isfinite(value)) for value in values) or not math.isfinite(lambert_coefficient):
        raise RendererError("NumPy renderer inputs must be finite")
    if np.any(areas <= 0) or lambert_coefficient < 0:
        raise RendererError("areas must be positive and Lambert coefficient nonnegative")
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    pole_array = pole_array / np.linalg.norm(pole_array)
    sun = sun / np.linalg.norm(sun, axis=1, keepdims=True)
    observer = observer / np.linalg.norm(observer, axis=1, keepdims=True)
    longitude = math.atan2(pole_array[1], pole_array[0])
    latitude = math.asin(float(np.clip(pole_array[2], -1.0, 1.0)))

    def to_body(directions: np.ndarray) -> np.ndarray:
        cosine, sine = math.cos(longitude), math.sin(longitude)
        x0 = directions[:, 0] * cosine + directions[:, 1] * sine
        y0 = -directions[:, 0] * sine + directions[:, 1] * cosine
        z0 = directions[:, 2]
        theta = math.pi / 2.0 - latitude
        x1 = x0 * math.cos(theta) - z0 * math.sin(theta)
        z1 = x0 * math.sin(theta) + z0 * math.cos(theta)
        x_body = np.cos(phases) * x1 + np.sin(phases) * y0
        y_body = -np.sin(phases) * x1 + np.cos(phases) * y0
        return np.stack((x_body, y_body, z1), axis=-1)

    sun_body, observer_body = to_body(sun), to_body(observer)
    mu0 = np.maximum(sun_body @ normals.T, 0.0)
    mu = np.maximum(observer_body @ normals.T, 0.0)
    product = mu * mu0
    brightness = (product / (mu + mu0 + _EPS) + lambert_coefficient * product) @ areas
    if normalize:
        brightness = brightness / max(float(brightness.mean()), _EPS)
    return brightness


@dataclass(frozen=True)
class RendererAgreement:
    maximum_absolute_error: float
    maximum_relative_error: float
    passed: bool


def validate_random_agreement(*, seed: int = 20260901, cases: int = 8, tolerance: float = 1e-5) -> RendererAgreement:
    """Exercise random shapes/geometries against the independent reference."""
    if cases <= 0 or tolerance <= 0:
        raise RendererError("cases and tolerance must be positive")
    generator = torch.Generator().manual_seed(seed)
    normals, areas, _ = sample_convex_ellipsoids(
        cases, generator=generator, facet_count=128, dtype=torch.float64
    )
    poles = torch.randn(cases, 3, generator=generator, dtype=torch.float64)
    phases = torch.rand(cases, 31, generator=generator, dtype=torch.float64) * (2 * math.pi)
    sun = torch.randn(cases, 31, 3, generator=generator, dtype=torch.float64)
    observer = torch.randn(cases, 31, 3, generator=generator, dtype=torch.float64)
    rendered = render_brightness(normals, areas, poles, phases, sun, observer).detach().numpy()
    reference = np.stack(
        [
            render_brightness_numpy(
                normals[index].numpy(),
                areas[index].numpy(),
                poles[index].numpy(),
                phases[index].numpy(),
                sun[index].numpy(),
                observer[index].numpy(),
            )
            for index in range(cases)
        ]
    )
    absolute = np.abs(rendered - reference)
    relative = absolute / np.maximum(np.abs(reference), 1e-12)
    maximum_absolute = float(np.max(absolute))
    return RendererAgreement(
        maximum_absolute_error=maximum_absolute,
        maximum_relative_error=float(np.max(relative)),
        passed=maximum_absolute <= tolerance,
    )
