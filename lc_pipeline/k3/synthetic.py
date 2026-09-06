"""Leakage-safe synthetic object generation and content-addressed shards."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch

from ..v2.preprocessing import Observation, ObservationEpoch
from .renderer import (
    DEFAULT_LAMBERT_COEFFICIENT,
    ConvexFacets,
    normalize_by_epoch,
    render_brightness,
)

SYNTHETIC_RECIPE_SCHEMA = "delphi.k3-synthetic-recipe.v1"
SYNTHETIC_SHARD_SCHEMA = "delphi.k3-synthetic-shard.v1"


class SyntheticDataError(ValueError):
    """Raised when simulation data violate the prospective contract."""


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class EmpiricalNoiseRecipe:
    """Train-only empirical noise approximation recorded in every shard."""

    provenance: str
    relative_sigma_median: float = 0.015
    relative_sigma_log_scatter: float = 0.35
    outlier_fraction: float = 0.01
    outlier_sigma_multiplier: float = 5.0

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise SyntheticDataError("noise recipe requires explicit train-only provenance")
        values = (
            self.relative_sigma_median,
            self.relative_sigma_log_scatter,
            self.outlier_fraction,
            self.outlier_sigma_multiplier,
        )
        if not all(math.isfinite(value) for value in values):
            raise SyntheticDataError("noise recipe values must be finite")
        if self.relative_sigma_median <= 0 or self.relative_sigma_log_scatter < 0:
            raise SyntheticDataError("noise scale must be positive and scatter nonnegative")
        if not 0 <= self.outlier_fraction < 1 or self.outlier_sigma_multiplier < 1:
            raise SyntheticDataError("outlier recipe is invalid")


@dataclass(frozen=True)
class ResidualNoiseBank:
    """Immutable real-fit residual snippets with explicit file provenance."""

    residuals: tuple[np.ndarray, ...]
    mad_sigmas: np.ndarray
    provenance: str
    sha256: str

    def __post_init__(self) -> None:
        snippets = tuple(np.asarray(value, dtype=np.float64) for value in self.residuals)
        sigmas = np.asarray(self.mad_sigmas, dtype=np.float64)
        if not snippets or sigmas.shape != (len(snippets),):
            raise SyntheticDataError("residual bank snippets and MAD scales must align")
        if any(value.ndim != 1 or value.size < 3 or not np.all(np.isfinite(value)) for value in snippets):
            raise SyntheticDataError("residual bank snippets must be finite one-dimensional arrays")
        if np.any(~np.isfinite(sigmas)) or np.any(sigmas <= 0) or not self.provenance or len(self.sha256) != 64:
            raise SyntheticDataError("residual bank provenance/scales/hash are invalid")
        for value in snippets:
            value.setflags(write=False)
        sigmas.setflags(write=False)
        object.__setattr__(self, "residuals", snippets)
        object.__setattr__(self, "mad_sigmas", sigmas)

    @classmethod
    def load(cls, path: str | Path, *, provenance: str) -> "ResidualNoiseBank":
        source = Path(path)
        try:
            archive = np.load(source, allow_pickle=False)
            count = int(np.asarray(archive["n"]).item())
            residuals = tuple(np.asarray(archive[f"r_{index}"], dtype=np.float64) for index in range(count))
            sigmas = np.asarray(archive["mad"], dtype=np.float64)
        except (OSError, ValueError, KeyError) as exc:
            raise SyntheticDataError(f"cannot load residual noise bank {source}: {exc}") from exc
        return cls(residuals, sigmas, provenance, sha256_file(source))

    def sample_like(self, generator: np.random.Generator, count: int) -> tuple[np.ndarray, float]:
        sigma = float(self.mad_sigmas[int(generator.integers(0, len(self.mad_sigmas)))])
        pieces: list[np.ndarray] = []
        total = 0
        while total < count:
            value = self.residuals[int(generator.integers(0, len(self.residuals)))]
            pieces.append(value)
            total += value.size
        pool = np.concatenate(pieces)
        start = int(generator.integers(0, max(1, pool.size - count + 1)))
        sampled = np.array(pool[start : start + count], copy=True)
        sampled -= sampled.mean()
        scale = float(sampled.std())
        if scale > 1e-12:
            sampled *= sigma / scale
        return sampled, sigma


@dataclass(frozen=True)
class SyntheticRecipe:
    split: Literal["train", "validation", "test", "smoke"]
    object_count: int
    master_seed: int
    noise: EmpiricalNoiseRecipe
    lambert_coefficient: float = DEFAULT_LAMBERT_COEFFICIENT
    period_range_hours: tuple[float, float] = (2.0, 200.0)
    schema: str = SYNTHETIC_RECIPE_SCHEMA

    def __post_init__(self) -> None:
        locked_counts = {"train": 20000, "validation": 2000, "test": 2000}
        if self.split in locked_counts and self.object_count != locked_counts[self.split]:
            raise SyntheticDataError(
                f"publication {self.split} split requires {locked_counts[self.split]} objects"
            )
        if isinstance(self.object_count, bool) or not isinstance(self.object_count, int) or self.object_count <= 0:
            raise SyntheticDataError("object_count must be a positive integer")
        if isinstance(self.master_seed, bool) or not isinstance(self.master_seed, int) or self.master_seed < 0:
            raise SyntheticDataError("master_seed must be a nonnegative integer")
        if not math.isfinite(self.lambert_coefficient) or self.lambert_coefficient < 0:
            raise SyntheticDataError("Lambert coefficient must be finite and nonnegative")
        low, high = self.period_range_hours
        if not all(math.isfinite(value) for value in (low, high)) or not 0 < low < high:
            raise SyntheticDataError("period range must be finite, positive, and increasing")
        if self.schema != SYNTHETIC_RECIPE_SCHEMA:
            raise SyntheticDataError("synthetic recipe schema mismatch")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json(asdict(self)).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SyntheticRecord:
    object_id: str
    epochs: tuple[ObservationEpoch, ...]
    target_axis: tuple[float, float, float]
    period_hours: float
    shape_donor_id: str
    geometry_donor_id: str
    seed: int

    def __post_init__(self) -> None:
        if not self.object_id or not self.shape_donor_id or not self.geometry_donor_id:
            raise SyntheticDataError("synthetic object and donor IDs are required")
        if not self.epochs or not all(isinstance(epoch, ObservationEpoch) for epoch in self.epochs):
            raise SyntheticDataError("synthetic records require source-preserving epochs")
        axis = np.asarray(self.target_axis, dtype=np.float64)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)) or abs(np.linalg.norm(axis) - 1) > 1e-6:
            raise SyntheticDataError("synthetic target axis must be finite and normalized")
        if not math.isfinite(self.period_hours) or self.period_hours <= 0:
            raise SyntheticDataError("synthetic period must be finite and positive")


def canonicalize_axes(axes: np.ndarray) -> np.ndarray:
    """Map directed unit vectors to one deterministic projective hemisphere."""
    values = np.asarray(axes, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        raise SyntheticDataError("axes must have shape [N,3] and be finite")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0):
        raise SyntheticDataError("axes must be nonzero")
    result = values / norms[:, None]
    tolerance = 1e-14
    flip = (result[:, 2] < -tolerance) | (
        (np.abs(result[:, 2]) <= tolerance)
        & ((result[:, 1] < -tolerance) | ((np.abs(result[:, 1]) <= tolerance) & (result[:, 0] < 0)))
    )
    result[flip] *= -1
    return result


def sample_uniform_axial_axes(count: int, *, seed: int) -> np.ndarray:
    """Sample uniformly on the sphere and quotient by the antipodal relation."""
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise SyntheticDataError("axis count must be a positive integer")
    generator = np.random.default_rng(seed)
    z = generator.uniform(-1.0, 1.0, count)
    longitude = generator.uniform(0.0, 2.0 * math.pi, count)
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    directed = np.stack((radius * np.cos(longitude), radius * np.sin(longitude), z), axis=1)
    return canonicalize_axes(directed)


def _sample_period(recipe: SyntheticRecipe, generator: np.random.Generator) -> float:
    low, high = recipe.period_range_hours
    return float(np.exp(generator.uniform(math.log(low), math.log(high))))


def generate_synthetic_record(
    *,
    object_id: str,
    geometry_donor_id: str,
    geometry_epochs: Sequence[ObservationEpoch],
    shape: ConvexFacets,
    recipe: SyntheticRecipe,
    seed: int,
    target_axis: Sequence[float] | None = None,
    period_hours: float | None = None,
    residual_noise_bank: ResidualNoiseBank | None = None,
) -> SyntheticRecord:
    """Render one noisy object using every observation in each donor epoch."""
    epochs = tuple(geometry_epochs)
    if not epochs:
        raise SyntheticDataError("geometry donor must contain at least one epoch")
    if any(len(epoch.observations) < 3 for epoch in epochs):
        raise SyntheticDataError("each geometry donor epoch requires at least three observations")
    generator = np.random.default_rng(seed)
    axis = (
        sample_uniform_axial_axes(1, seed=seed)[0]
        if target_axis is None
        else canonicalize_axes(np.asarray(target_axis, dtype=np.float64))[0]
    )
    period = _sample_period(recipe, generator) if period_hours is None else float(period_hours)
    if not math.isfinite(period) or period <= 0:
        raise SyntheticDataError("period_hours must be finite and positive")
    flattened = [observation for epoch in epochs for observation in epoch.observations]
    times = np.asarray([observation.time_jd for observation in flattened], dtype=np.float64)
    sun = np.asarray(
        [observation.sun_asteroid_ecliptic_j2000_au for observation in flattened], dtype=np.float64
    )
    observer = np.asarray(
        [observation.observer_asteroid_ecliptic_j2000_au for observation in flattened], dtype=np.float64
    )
    epoch_index = np.concatenate(
        [np.full(len(epoch.observations), index, dtype=np.int64) for index, epoch in enumerate(epochs)]
    )
    initial_phase = generator.uniform(0.0, 2.0 * math.pi)
    phases = initial_phase + 2.0 * math.pi * (times - times.min()) / (period / 24.0)
    dtype = torch.float64
    brightness = render_brightness(
        torch.as_tensor(np.array(shape.normals, copy=True), dtype=dtype),
        torch.as_tensor(np.array(shape.areas, copy=True), dtype=dtype),
        torch.as_tensor(axis[None, :], dtype=dtype),
        torch.as_tensor(phases[None, :], dtype=dtype),
        torch.as_tensor(sun[None, :, :], dtype=dtype),
        torch.as_tensor(observer[None, :, :], dtype=dtype),
        lambert_coefficient=recipe.lambert_coefficient,
    )
    brightness = normalize_by_epoch(
        brightness, torch.as_tensor(epoch_index[None, :], dtype=torch.int64)
    )[0].detach().numpy()
    if not np.all(np.isfinite(brightness)) or np.any(brightness <= 0):
        raise SyntheticDataError("renderer produced invalid or unilluminated synthetic observations")
    if recipe.split != "smoke" and residual_noise_bank is None:
        raise SyntheticDataError("publication synthetic splits require a hashed residual noise bank")
    if residual_noise_bank is not None:
        noisy = None
        sampled_sigma = 0.0
        for _attempt in range(32):
            residual, sampled_sigma = residual_noise_bank.sample_like(generator, brightness.size)
            proposal = brightness + residual
            if np.all(np.isfinite(proposal)) and np.all(proposal > 0):
                noisy = proposal
                break
        if noisy is None:
            raise SyntheticDataError("residual bank could not produce positive synthetic brightness in 32 attempts")
        sigma = np.full(brightness.size, sampled_sigma, dtype=np.float64)
    else:
        sigma = recipe.noise.relative_sigma_median * np.exp(
            recipe.noise.relative_sigma_log_scatter * generator.normal(size=brightness.size)
        )
        outliers = generator.random(brightness.size) < recipe.noise.outlier_fraction
        sigma[outliers] *= recipe.noise.outlier_sigma_multiplier
        noisy = brightness * np.exp(generator.normal(scale=sigma))
    if not np.all(np.isfinite(noisy)) or np.any(noisy <= 0):
        raise SyntheticDataError("noise recipe produced invalid synthetic brightness")
    output_epochs: list[ObservationEpoch] = []
    cursor = 0
    for epoch in epochs:
        observations: list[Observation] = []
        for donor in epoch.observations:
            observations.append(
                Observation(
                    time_jd=donor.time_jd,
                    relative_brightness=float(noisy[cursor]),
                    sun_asteroid_ecliptic_j2000_au=donor.sun_asteroid_ecliptic_j2000_au,
                    observer_asteroid_ecliptic_j2000_au=donor.observer_asteroid_ecliptic_j2000_au,
                    measured_error=float(sigma[cursor] * noisy[cursor]),
                )
            )
            cursor += 1
        output_epochs.append(ObservationEpoch(epoch.epoch_id, tuple(observations)))
    return SyntheticRecord(
        object_id=object_id,
        epochs=tuple(output_epochs),
        target_axis=tuple(float(value) for value in axis),
        period_hours=period,
        shape_donor_id=shape.source_id,
        geometry_donor_id=geometry_donor_id,
        seed=seed,
    )


def _atomic_json(path: Path, value: object) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_synthetic_shard(
    path: str | Path,
    records: Sequence[SyntheticRecord],
    *,
    recipe: SyntheticRecipe,
    shard_index: int,
) -> dict[str, Any]:
    """Write a non-pickle NPZ shard and immutable SHA-256 manifest."""
    destination = Path(path)
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    if destination.exists() or manifest_path.exists():
        raise SyntheticDataError("refusing to overwrite a synthetic shard or manifest")
    values = tuple(records)
    if not values:
        raise SyntheticDataError("cannot write an empty synthetic shard")
    if len({value.object_id for value in values}) != len(values):
        raise SyntheticDataError("synthetic shard object IDs must be unique")
    destination.parent.mkdir(parents=True, exist_ok=True)
    times: list[float] = []
    brightness: list[float] = []
    measured_error: list[float] = []
    sun: list[tuple[float, float, float]] = []
    observer: list[tuple[float, float, float]] = []
    epoch_ids: list[str] = []
    epoch_offsets = [0]
    object_epoch_offsets = [0]
    for record in values:
        for epoch in record.epochs:
            epoch_ids.append(epoch.epoch_id)
            for observation in epoch.observations:
                times.append(observation.time_jd)
                brightness.append(observation.relative_brightness)
                measured_error.append(observation.measured_error or 0.0)
                sun.append(observation.sun_asteroid_ecliptic_j2000_au)
                observer.append(observation.observer_asteroid_ecliptic_j2000_au)
            epoch_offsets.append(len(times))
        object_epoch_offsets.append(len(epoch_ids))
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                object_ids=np.asarray([value.object_id for value in values]),
                target_axes=np.asarray([value.target_axis for value in values], dtype=np.float64),
                periods_hours=np.asarray([value.period_hours for value in values], dtype=np.float64),
                shape_donor_ids=np.asarray([value.shape_donor_id for value in values]),
                geometry_donor_ids=np.asarray([value.geometry_donor_id for value in values]),
                seeds=np.asarray([value.seed for value in values], dtype=np.int64),
                object_epoch_offsets=np.asarray(object_epoch_offsets, dtype=np.int64),
                epoch_ids=np.asarray(epoch_ids),
                epoch_offsets=np.asarray(epoch_offsets, dtype=np.int64),
                times_jd=np.asarray(times, dtype=np.float64),
                relative_brightness=np.asarray(brightness, dtype=np.float64),
                measured_error=np.asarray(measured_error, dtype=np.float64),
                sun_vectors=np.asarray(sun, dtype=np.float64),
                observer_vectors=np.asarray(observer, dtype=np.float64),
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    manifest = {
        "schema": SYNTHETIC_SHARD_SCHEMA,
        "split": recipe.split,
        "shard_index": int(shard_index),
        "object_count": len(values),
        "recipe_sha256": recipe.sha256,
        "data_file": destination.name,
        "data_sha256": sha256_file(destination),
        "shape_donor_ids": sorted({value.shape_donor_id for value in values}),
        "geometry_donor_ids": sorted({value.geometry_donor_id for value in values}),
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def load_synthetic_shard(
    path: str | Path,
    *,
    verify_manifest: bool = True,
    selected_object_ids: set[str] | None = None,
) -> tuple[SyntheticRecord, ...]:
    """Load a safe NPZ shard and reconstruct exact source epoch boundaries."""
    source = Path(path)
    if verify_manifest:
        try:
            manifest = json.loads(source.with_suffix(source.suffix + ".manifest.json").read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SyntheticDataError(f"cannot read synthetic shard manifest: {exc}") from exc
        if manifest.get("schema") != SYNTHETIC_SHARD_SCHEMA or manifest.get("data_sha256") != sha256_file(source):
            raise SyntheticDataError("synthetic shard manifest/hash mismatch")
    try:
        archive = np.load(source, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise SyntheticDataError(f"cannot load synthetic shard: {exc}") from exc
    required = {
        "object_ids", "target_axes", "periods_hours", "shape_donor_ids", "geometry_donor_ids",
        "seeds", "object_epoch_offsets", "epoch_ids", "epoch_offsets", "times_jd",
        "relative_brightness", "measured_error", "sun_vectors", "observer_vectors",
    }
    if required - set(archive.files):
        raise SyntheticDataError("synthetic shard is missing required arrays")
    # Accessing a compressed NPZ member repeatedly decompresses that complete
    # member on every lookup. Materialize each required array once before the
    # object loop; otherwise a publication shard scales effectively as O(N^2).
    arrays = {name: np.asarray(archive[name]) for name in required}
    archive.close()
    records: list[SyntheticRecord] = []
    object_ids = arrays["object_ids"]
    object_epoch_offsets = arrays["object_epoch_offsets"]
    epoch_offsets = arrays["epoch_offsets"]
    if object_epoch_offsets.shape != (len(object_ids) + 1,):
        raise SyntheticDataError("synthetic object epoch offsets are malformed")
    for object_index, object_id in enumerate(object_ids):
        if selected_object_ids is not None and str(object_id) not in selected_object_ids:
            continue
        epochs: list[ObservationEpoch] = []
        first_epoch, final_epoch = object_epoch_offsets[object_index : object_index + 2]
        for epoch_index in range(int(first_epoch), int(final_epoch)):
            start, stop = (int(value) for value in epoch_offsets[epoch_index : epoch_index + 2])
            observations = tuple(
                Observation(
                    time_jd=float(arrays["times_jd"][point]),
                    relative_brightness=float(arrays["relative_brightness"][point]),
                    measured_error=float(arrays["measured_error"][point]),
                    sun_asteroid_ecliptic_j2000_au=tuple(arrays["sun_vectors"][point]),
                    observer_asteroid_ecliptic_j2000_au=tuple(arrays["observer_vectors"][point]),
                )
                for point in range(start, stop)
            )
            epochs.append(ObservationEpoch(str(arrays["epoch_ids"][epoch_index]), observations))
        records.append(
            SyntheticRecord(
                object_id=str(object_id),
                epochs=tuple(epochs),
                target_axis=tuple(float(value) for value in arrays["target_axes"][object_index]),
                period_hours=float(arrays["periods_hours"][object_index]),
                shape_donor_id=str(arrays["shape_donor_ids"][object_index]),
                geometry_donor_id=str(arrays["geometry_donor_ids"][object_index]),
                seed=int(arrays["seeds"][object_index]),
            )
        )
    return tuple(records)


def verify_synthetic_shard(path: str | Path) -> dict[str, object]:
    """Verify a shard/hash/offset contract without constructing Python records."""
    source = Path(path)
    manifest_path = source.with_suffix(source.suffix + ".manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SyntheticDataError(f"cannot read synthetic shard manifest: {exc}") from exc
    if manifest.get("schema") != SYNTHETIC_SHARD_SCHEMA or manifest.get("data_sha256") != sha256_file(source):
        raise SyntheticDataError("synthetic shard manifest/hash mismatch")
    try:
        with np.load(source, allow_pickle=False) as archive:
            required = {
                "object_ids", "target_axes", "periods_hours", "shape_donor_ids", "geometry_donor_ids",
                "seeds", "object_epoch_offsets", "epoch_ids", "epoch_offsets", "times_jd",
                "relative_brightness", "measured_error", "sun_vectors", "observer_vectors",
            }
            if required - set(archive.files):
                raise SyntheticDataError("synthetic shard is missing required arrays")
            arrays = {name: np.asarray(archive[name]) for name in required}
    except (OSError, ValueError) as exc:
        raise SyntheticDataError(f"cannot verify synthetic shard {source}: {exc}") from exc
    count = len(arrays["object_ids"])
    object_offsets, epoch_offsets = arrays["object_epoch_offsets"], arrays["epoch_offsets"]
    epochs, points = len(arrays["epoch_ids"]), len(arrays["times_jd"])
    if count != manifest.get("object_count") or len(set(map(str, arrays["object_ids"]))) != count:
        raise SyntheticDataError("synthetic shard object IDs/count are invalid")
    if object_offsets.shape != (count + 1,) or object_offsets[0] != 0 or object_offsets[-1] != epochs or np.any(np.diff(object_offsets) <= 0):
        raise SyntheticDataError("synthetic object epoch offsets are malformed")
    if epoch_offsets.shape != (epochs + 1,) or epoch_offsets[0] != 0 or epoch_offsets[-1] != points or np.any(np.diff(epoch_offsets) < 3):
        raise SyntheticDataError("synthetic epoch observation offsets are malformed")
    aligned_object = ("target_axes", "periods_hours", "shape_donor_ids", "geometry_donor_ids", "seeds")
    aligned_point = ("relative_brightness", "measured_error", "sun_vectors", "observer_vectors")
    if any(len(arrays[name]) != count for name in aligned_object) or any(len(arrays[name]) != points for name in aligned_point):
        raise SyntheticDataError("synthetic shard arrays are not aligned")
    numeric = ("target_axes", "periods_hours", "times_jd", *aligned_point)
    if any(not np.all(np.isfinite(arrays[name])) for name in numeric) or np.any(arrays["relative_brightness"] <= 0):
        raise SyntheticDataError("synthetic shard contains invalid numeric values")
    return {
        "path": source.as_posix(),
        "sha256": manifest["data_sha256"],
        "objects": count,
        "epochs": epochs,
        "observations": points,
        "object_ids": tuple(map(str, arrays["object_ids"])),
        "shape_donor_ids": tuple(sorted(set(map(str, arrays["shape_donor_ids"])))),
        "geometry_donor_ids": tuple(sorted(set(map(str, arrays["geometry_donor_ids"])))),
    }
