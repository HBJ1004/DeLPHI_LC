"""Leakage-safe, training-fold-only three-axis atlas baseline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ..physics.axial import pairwise_axial_angular_errors_deg
from .data import PreparedObject, canonical_json

ATLAS_SCHEMA = "delphi.axis-atlas.v1"


class AtlasContractError(ValueError):
    """Raised when atlas fitting or loading could violate fold isolation."""


@dataclass(frozen=True)
class AtlasConfig:
    candidate_count: int = 3
    restart_count: int = 16
    refinement_steps: int = 2_000
    learning_rate: float = 0.05
    greedy_grid_size: int = 256
    initialization_seed: int = 20260902

    def __post_init__(self) -> None:
        integer_values = (
            self.candidate_count,
            self.restart_count,
            self.refinement_steps,
            self.greedy_grid_size,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in integer_values
        ):
            raise AtlasContractError("atlas counts must be positive integers")
        if self.candidate_count != 3:
            raise AtlasContractError("the comparison contract fixes candidate_count=3")
        if self.greedy_grid_size < self.candidate_count:
            raise AtlasContractError("greedy grid must contain at least three axes")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise AtlasContractError("atlas learning rate must be finite and positive")
        if isinstance(self.initialization_seed, bool) or not isinstance(
            self.initialization_seed, int
        ):
            raise AtlasContractError("initialization_seed must be an integer")


@dataclass(frozen=True)
class AxisAtlas:
    axes: tuple[tuple[float, float, float], ...]
    mean_training_oracle_error_deg: float
    object_count: int
    train_object_ids_sha256: str
    source_labels_sha256: str
    configuration: AtlasConfig
    best_restart: int
    atlas_sha256: str
    schema: str = ATLAS_SCHEMA

    def as_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "axes": [list(axis) for axis in self.axes],
            "mean_training_oracle_error_deg": self.mean_training_oracle_error_deg,
            "object_count": self.object_count,
            "train_object_ids_sha256": self.train_object_ids_sha256,
            "source_labels_sha256": self.source_labels_sha256,
            "configuration": asdict(self.configuration),
            "best_restart": self.best_restart,
        }
        if include_hash:
            value["atlas_sha256"] = self.atlas_sha256
        return value


def _sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_axis_sign(axes: np.ndarray) -> np.ndarray:
    """Choose a stable representative for each equivalence class ``u ~ -u``."""
    result = np.asarray(axes, dtype=np.float64).copy()
    for index, axis in enumerate(result):
        for component in axis:
            if abs(component) > 1.0e-15:
                if component < 0:
                    result[index] *= -1.0
                break
    return result


def _normalized_source_objects(
    values: Sequence[PreparedObject],
    *,
    partition_role: str,
    forbidden_object_ids: Sequence[str],
) -> tuple[tuple[str, np.ndarray], ...]:
    if partition_role != "train":
        raise AtlasContractError("axis atlas labels may only come from a partition named 'train'")
    if not values:
        raise AtlasContractError("axis atlas requires at least one training object")
    sorted_values = sorted(values, key=lambda value: value.object_id)
    object_ids = [value.object_id for value in sorted_values]
    if len(object_ids) != len(set(object_ids)):
        raise AtlasContractError("axis atlas requires unique training object IDs")
    overlap = sorted(set(object_ids) & set(forbidden_object_ids))
    if overlap:
        raise AtlasContractError(
            f"atlas training IDs overlap forbidden evaluation IDs: {overlap[:10]}"
        )
    result: list[tuple[str, np.ndarray]] = []
    for value in sorted_values:
        value.validate()
        vectors = np.asarray(value.target_vectors, dtype=np.float64)
        norms = np.linalg.vector_norm(vectors, axis=1)
        if not vectors.size or not np.all(np.isfinite(vectors)) or np.any(norms == 0):
            raise AtlasContractError(f"invalid source axes for {value.object_id}")
        result.append((value.object_id, vectors / norms[:, None]))
    return tuple(result)


def _fibonacci_axis_grid(size: int, phase: float) -> np.ndarray:
    """Return a deterministic approximately uniform projective-sphere grid."""
    index = np.arange(size, dtype=np.float64)
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    z = (index + 0.5) / size  # one canonical hemisphere represents each axis
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    azimuth = 2.0 * math.pi * np.mod(index / golden + phase, 1.0)
    return np.column_stack((radius * np.cos(azimuth), radius * np.sin(azimuth), z))


def _mean_oracle_error(axes: np.ndarray, source_objects: Sequence[tuple[str, np.ndarray]]) -> float:
    errors = [
        float(np.min(pairwise_axial_angular_errors_deg(axes, source_axes)))
        for _, source_axes in source_objects
    ]
    return float(np.mean(errors))


def _greedy_initialization(
    source_objects: Sequence[tuple[str, np.ndarray]], config: AtlasConfig, restart: int
) -> np.ndarray:
    phase = ((config.initialization_seed + restart * 104_729) % 1_000_003) / 1_000_003
    grid = _fibonacci_axis_grid(config.greedy_grid_size, phase)
    selected: list[np.ndarray] = []
    available = np.ones(config.greedy_grid_size, dtype=bool)
    for _ in range(config.candidate_count):
        best_index = -1
        best_objective = math.inf
        for index in np.flatnonzero(available):
            trial = np.vstack([*selected, grid[index]])
            objective = _mean_oracle_error(trial, source_objects)
            if objective < best_objective:
                best_objective = objective
                best_index = int(index)
        selected.append(grid[best_index])
        available[best_index] = False
    return np.asarray(selected, dtype=np.float64)


def _padded_targets(
    source_objects: Sequence[tuple[str, np.ndarray]],
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = max(vectors.shape[0] for _, vectors in source_objects)
    targets = torch.zeros((len(source_objects), maximum, 3), dtype=torch.float64)
    mask = torch.zeros((len(source_objects), maximum), dtype=torch.bool)
    for index, (_, vectors) in enumerate(source_objects):
        targets[index, : vectors.shape[0]] = torch.from_numpy(vectors)
        mask[index, : vectors.shape[0]] = True
    return targets, mask


def _refine_axes(
    initial_axes: np.ndarray,
    targets: torch.Tensor,
    mask: torch.Tensor,
    config: AtlasConfig,
) -> np.ndarray:
    parameters = torch.nn.Parameter(torch.as_tensor(initial_axes, dtype=torch.float64).clone())
    optimizer = torch.optim.Adam([parameters], lr=config.learning_rate)
    for _ in range(config.refinement_steps):
        optimizer.zero_grad(set_to_none=True)
        axes = torch.nn.functional.normalize(parameters, dim=-1)
        dots = torch.einsum("kc,bsc->bks", axes, targets)
        costs = (1.0 - dots.square()).clamp_min(0.0)
        costs = costs.masked_fill(~mask[:, None, :], float("inf"))
        loss = costs.flatten(1).amin(dim=1).mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        result = torch.nn.functional.normalize(parameters, dim=-1).cpu().numpy()
    return _canonical_axis_sign(result)


def fit_axis_atlas(
    values: Sequence[PreparedObject],
    *,
    partition_role: str,
    forbidden_object_ids: Sequence[str] = (),
    config: AtlasConfig = AtlasConfig(),
) -> AxisAtlas:
    """Fit K=3 axes using training labels and no evaluation labels.

    Every object contributes one mean-oracle term no matter how many explicit
    source solutions it has. Inputs are sorted by object ID before fitting, so
    file or loader order cannot affect the result.
    """
    source_objects = _normalized_source_objects(
        values,
        partition_role=partition_role,
        forbidden_object_ids=forbidden_object_ids,
    )
    targets, mask = _padded_targets(source_objects)
    best_axes: np.ndarray | None = None
    best_objective = math.inf
    best_restart = -1
    for restart in range(config.restart_count):
        initial = _greedy_initialization(source_objects, config, restart)
        refined = _refine_axes(initial, targets, mask, config)
        objective = _mean_oracle_error(refined, source_objects)
        if objective < best_objective:
            best_axes = refined
            best_objective = objective
            best_restart = restart
    assert best_axes is not None
    identifiers = [object_id for object_id, _ in source_objects]
    label_payload = [
        {"object_id": object_id, "source_axes": vectors.tolist()}
        for object_id, vectors in source_objects
    ]
    base = {
        "schema": ATLAS_SCHEMA,
        "axes": best_axes.tolist(),
        "mean_training_oracle_error_deg": best_objective,
        "object_count": len(source_objects),
        "train_object_ids_sha256": _sha256_json(identifiers),
        "source_labels_sha256": _sha256_json(label_payload),
        "configuration": asdict(config),
        "best_restart": best_restart,
    }
    return AxisAtlas(
        axes=tuple(tuple(float(component) for component in axis) for axis in best_axes),
        mean_training_oracle_error_deg=best_objective,
        object_count=len(source_objects),
        train_object_ids_sha256=base["train_object_ids_sha256"],
        source_labels_sha256=base["source_labels_sha256"],
        configuration=config,
        best_restart=best_restart,
        atlas_sha256=_sha256_json(base),
    )


def write_axis_atlas(path: str | Path, atlas: AxisAtlas) -> None:
    """Atomically write the compact, content-addressed atlas artifact."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(canonical_json(atlas.as_payload()) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_axis_atlas(path: str | Path) -> AxisAtlas:
    """Read an atlas and reject any schema, shape, norm, or hash mismatch."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AtlasContractError(f"cannot read axis atlas: {exc}") from exc
    if payload.get("schema") != ATLAS_SCHEMA:
        raise AtlasContractError("axis atlas schema mismatch")
    declared_hash = payload.pop("atlas_sha256", None)
    if not isinstance(declared_hash, str) or declared_hash != _sha256_json(payload):
        raise AtlasContractError("axis atlas hash mismatch")
    try:
        configuration = AtlasConfig(**payload["configuration"])
        axes_array = np.asarray(payload["axes"], dtype=np.float64)
        if axes_array.shape != (3, 3) or not np.all(np.isfinite(axes_array)):
            raise AtlasContractError("axis atlas requires three finite 3-vectors")
        norms = np.linalg.vector_norm(axes_array, axis=1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=1.0e-12):
            raise AtlasContractError("axis atlas vectors must be unit normalized")
        return AxisAtlas(
            axes=tuple(tuple(float(component) for component in row) for row in axes_array),
            mean_training_oracle_error_deg=float(payload["mean_training_oracle_error_deg"]),
            object_count=int(payload["object_count"]),
            train_object_ids_sha256=str(payload["train_object_ids_sha256"]),
            source_labels_sha256=str(payload["source_labels_sha256"]),
            configuration=configuration,
            best_restart=int(payload["best_restart"]),
            atlas_sha256=declared_hash,
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, AtlasContractError):
            raise
        raise AtlasContractError(f"malformed axis atlas: {exc}") from exc


__all__ = [
    "ATLAS_SCHEMA",
    "AtlasConfig",
    "AtlasContractError",
    "AxisAtlas",
    "fit_axis_atlas",
    "read_axis_atlas",
    "write_axis_atlas",
]
