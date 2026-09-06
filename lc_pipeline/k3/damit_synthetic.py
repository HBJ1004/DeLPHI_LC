"""Deterministic synthetic-shard generation from a local DAMIT snapshot."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from .damit import DAMITObject, load_damit_object
from .synthetic import (
    SYNTHETIC_SHARD_SCHEMA,
    ResidualNoiseBank,
    SyntheticDataError,
    SyntheticRecipe,
    generate_synthetic_record,
    sha256_file,
    write_synthetic_shard,
)


def discover_damit_donors(
    dump_root: str | Path,
    *,
    excluded_object_ids: Sequence[str],
    minimum_epochs: int = 5,
    minimum_observations: int = 150,
) -> tuple[str, ...]:
    """Discover valid non-study donors using frozen, explicit thresholds."""
    root = Path(dump_root)
    excluded = set(excluded_object_ids)
    candidates: list[str] = []
    try:
        entries = sorted(
            (entry for entry in os.scandir(root / "files") if entry.is_dir() and entry.name.startswith("asteroid_")),
            key=lambda entry: entry.name,
        )
    except OSError as exc:
        raise SyntheticDataError(f"cannot enumerate DAMIT donor snapshot: {exc}") from exc
    for entry in entries:
        if entry.name in excluded:
            continue
        try:
            metadata = json.loads((Path(entry.path) / "lc.json").read_text(encoding="utf-8"))
            epoch_count = len(metadata)
            observation_count = sum(int(epoch["points_count"]) for epoch in metadata)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
        if epoch_count < minimum_epochs or observation_count < minimum_observations:
            continue
        try:
            value = load_damit_object(root, entry.name)
        except (ValueError, OSError):
            continue
        if all(len(epoch.observations) >= 3 for epoch in value.epochs):
            candidates.append(entry.name)
    if not candidates:
        raise SyntheticDataError("no non-study DAMIT donors satisfy the locked criteria")
    return tuple(candidates)


def partition_damit_donors(
    donor_object_ids: Sequence[str], *, master_seed: int
) -> Mapping[str, tuple[str, ...]]:
    """Assign every donor to exactly one 80/10/10 synthetic split."""
    values = tuple(sorted(set(donor_object_ids)))
    if len(values) < 10:
        raise SyntheticDataError("at least ten donors are required for disjoint split assignment")
    ranked = sorted(
        values,
        key=lambda value: hashlib.sha256(f"{master_seed}:{value}".encode("ascii")).digest(),
    )
    train_end, validation_end = int(0.8 * len(ranked)), int(0.9 * len(ranked))
    result = {
        "train": tuple(ranked[:train_end]),
        "validation": tuple(ranked[train_end:validation_end]),
        "test": tuple(ranked[validation_end:]),
    }
    if any(not group for group in result.values()):
        raise SyntheticDataError("donor split assignment produced an empty partition")
    return result


def generate_damit_synthetic_shards(
    dump_root: str | Path,
    donor_object_ids: Sequence[str],
    *,
    recipe: SyntheticRecipe,
    output_dir: str | Path,
    shard_size: int = 500,
    resume: bool = False,
    residual_noise_bank: ResidualNoiseBank | None = None,
) -> tuple[dict[str, object], ...]:
    """Render a locked synthetic split using only train-listed DAMIT donors.

    Donors are loaded once and selected by a deterministic round-robin stream;
    no object outside ``donor_object_ids`` can enter the generated split.
    """
    donors = tuple(sorted(set(donor_object_ids)))
    if not donors or shard_size <= 0 or recipe.object_count <= 0:
        raise SyntheticDataError("synthetic generation requires nonempty donors, count, and shard size")
    loaded: tuple[DAMITObject, ...] = tuple(load_damit_object(dump_root, object_id) for object_id in donors)
    usable = tuple(value for value in loaded if all(len(epoch.observations) >= 3 for epoch in value.epochs))
    if not usable:
        raise SyntheticDataError("no DAMIT donor has three observations in every source epoch")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifests: list[dict[str, object]] = []
    # Generate and commit one shard at a time. The publication split contains
    # millions of observations and must not retain every rendered record in
    # memory before the first artifact is written.
    for shard_index, start in enumerate(range(0, recipe.object_count, shard_size)):
        stop = min(start + shard_size, recipe.object_count)
        path = destination / f"{recipe.split}-{shard_index:04d}.npz"
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        if resume and path.exists() and manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise SyntheticDataError(f"cannot resume invalid shard manifest {manifest_path}: {exc}") from exc
            expected_count = stop - start
            if (
                manifest.get("schema") != SYNTHETIC_SHARD_SCHEMA
                or manifest.get("split") != recipe.split
                or manifest.get("shard_index") != shard_index
                or manifest.get("object_count") != expected_count
                or manifest.get("recipe_sha256") != recipe.sha256
                or manifest.get("data_sha256") != sha256_file(path)
            ):
                raise SyntheticDataError(f"cannot resume mismatched shard {path}")
            manifests.append(manifest)
            continue
        if path.exists() or manifest_path.exists():
            raise SyntheticDataError(f"refusing to overwrite existing shard {path}")
        records = []
        for index in range(start, stop):
            geometry = usable[index % len(usable)]
            shape = usable[(index * 9973 + 17) % len(usable)]
            records.append(
                generate_synthetic_record(
                    object_id=f"synthetic-{recipe.split}-{index:06d}",
                    geometry_donor_id=geometry.object_id,
                    geometry_epochs=geometry.epochs,
                    shape=shape.shape,
                    recipe=recipe,
                    seed=recipe.master_seed + index,
                    residual_noise_bank=residual_noise_bank,
                )
            )
        manifests.append(write_synthetic_shard(path, records, recipe=recipe, shard_index=shard_index))
    return tuple(manifests)
