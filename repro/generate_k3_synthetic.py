"""Regenerate the frozen K3 simulations with independent v2 random streams.

Run ``python -B -m repro.generate_k3_synthetic --check --frozen-dir <v1-synthetic>
--dump-root <DAMIT-snapshot> --noise-bank <original-noise-bank.npz>`` before generation.
Only the recipe schema changes; global object indices, seeds, donor pairing,
and the original residual bank are retained. Initial phase is not stored by
the shard writer, so summaries never infer it from a seed or from photometry.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import time

import numpy as np
from scipy.stats import ks_2samp, kstest, spearmanr
import torch

from lc_pipeline.k3.damit import load_damit_object
from lc_pipeline.k3.damit_synthetic import discover_damit_donors, partition_damit_donors
from lc_pipeline.k3.synthetic import (
    EmpiricalNoiseRecipe,
    LEGACY_SYNTHETIC_SHARD_SCHEMA,
    ResidualNoiseBank,
    SYNTHETIC_RECIPE_SCHEMA,
    SYNTHETIC_SHARD_SCHEMA,
    SyntheticDataError,
    SyntheticRecipe,
    _atomic_json,
    canonical_json,
    generate_synthetic_record,
    sha256_file,
    write_synthetic_shard,
)


REPOSITORY = Path(__file__).resolve().parents[1]
OLD_SCHEMA = "delphi.k3-synthetic-recipe.v1"
OLD_HASHES = {
    "train": "8d673d04fa25772aed316681d1f7683b37fbbf4f3f8644175bfad6d32342df28",
    "validation": "0be2bdbbf95f07a6a4379666bf4bbe73dcb48e52d00b63c0af5fdb3a7ac82393",
    "test": "911683ccd6f6305caf95d79eb8bf46ca8ec747d42a73f1dc29f0ab02a11d2a75",
}
SHARD_SIZE = 250
# Filled from the historical recipe, with all three v1 hashes as the proof.
NOISE_PROVENANCE = "damit-20250610T000301Z:nonstudy-pool-219"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SyntheticDataError(message)


def recipe_versions(recipe: SyntheticRecipe, expected_old_hash: str) -> dict:
    current = asdict(recipe)
    legacy = {**current, "schema": OLD_SCHEMA}
    old_hash = hashlib.sha256(canonical_json(legacy).encode("utf-8")).hexdigest()
    require(old_hash == expected_old_hash, f"{recipe.split}: v1 recipe hash mismatch: {old_hash}")
    require(current["schema"] == SYNTHETIC_RECIPE_SCHEMA != OLD_SCHEMA, "recipe version was not bumped")
    changed = [key for key in current if current[key] != legacy[key]]
    require(changed == ["schema"], "v2 recipe differs beyond its schema")
    return {"v1": legacy, "v2": current, "v1_sha256": old_hash,
            "v2_sha256": recipe.sha256, "changed_fields": changed}


def expected_arrays(recipe: SyntheticRecipe, donors: tuple[str, ...], shapes: dict[str, str],
                    shard_index: int, shard_size: int = SHARD_SIZE) -> dict[str, np.ndarray]:
    start = shard_index * shard_size
    stop = min(start + shard_size, recipe.object_count)
    return {
        "object_ids": np.asarray([f"synthetic-{recipe.split}-{i:06d}" for i in range(start, stop)]),
        "seeds": np.arange(recipe.master_seed + start, recipe.master_seed + stop, dtype=np.int64),
        "geometry_donor_ids": np.asarray([donors[i % len(donors)] for i in range(start, stop)]),
        "shape_donor_ids": np.asarray([shapes[donors[(i * 9973 + 17) % len(donors)]]
                                       for i in range(start, stop)]),
    }


def check_shard(path: Path, recipe: SyntheticRecipe, donors: tuple[str, ...], shapes: dict[str, str],
                index: int, *, legacy: bool, shard_size: int = SHARD_SIZE) -> dict:
    manifest = json.loads(path.with_suffix(".npz.manifest.json").read_text(encoding="utf-8"))
    expected = expected_arrays(recipe, donors, shapes, index, shard_size)
    schema = LEGACY_SYNTHETIC_SHARD_SCHEMA if legacy else SYNTHETIC_SHARD_SCHEMA
    digest = OLD_HASHES[recipe.split] if legacy else recipe.sha256
    fields = {"schema": schema, "split": recipe.split, "shard_index": index,
              "object_count": len(expected["object_ids"]), "recipe_sha256": digest,
              "data_file": path.name,
              "geometry_donor_ids": sorted(set(expected["geometry_donor_ids"].tolist())),
              "shape_donor_ids": sorted(set(expected["shape_donor_ids"].tolist()))}
    if not legacy:
        fields["recipe_schema"] = recipe.schema
    for key, value in fields.items():
        require(manifest.get(key) == value, f"{path}: manifest {key} mismatch")
    require(manifest.get("data_sha256") == sha256_file(path), f"{path}: data hash mismatch")
    with np.load(path, allow_pickle=False) as archive:
        for key, value in expected.items():
            require(np.array_equal(archive[key], value), f"{path}: stored {key} mismatch")
    return manifest


def check_frozen_split(root: Path, recipe: SyntheticRecipe, donors: tuple[str, ...],
                       shapes: dict[str, str], *, shard_size: int = SHARD_SIZE) -> dict:
    versions = recipe_versions(recipe, OLD_HASHES[recipe.split])
    count = math.ceil(recipe.object_count / shard_size)
    paths = [root / f"{recipe.split}-{i:04d}.npz" for i in range(count)]
    require(sorted(root.glob("*.npz")) == paths, f"{recipe.split}: frozen shard list mismatch")
    require(sorted(root.glob("*.npz.manifest.json")) == [p.with_suffix(".npz.manifest.json") for p in paths],
            f"{recipe.split}: frozen manifest list mismatch")
    manifests = [check_shard(p, recipe, donors, shapes, i, legacy=True, shard_size=shard_size)
                 for i, p in enumerate(paths)]
    geometry = sorted({d for m in manifests for d in m["geometry_donor_ids"]})
    shape_ids = sorted({d for m in manifests for d in m["shape_donor_ids"]})
    require(geometry == list(donors), f"{recipe.split}: frozen geometry donor pool mismatch")
    require(shape_ids == sorted(shapes.values()), f"{recipe.split}: frozen shape donor pool mismatch")
    return {"objects": recipe.object_count, "shards": count, "donors": len(donors),
            "geometry_donor_ids": geometry, "shape_donor_ids": shape_ids,
            "all_record_ids_seeds_and_shape_pairs_match": True, "all_data_hashes_match": True,
            "recipes": versions}


def check_inputs(args: argparse.Namespace) -> tuple[dict, dict, dict, dict]:
    spec = json.loads((REPOSITORY / "repro/k3_redesign_spec.yaml").read_text(encoding="utf-8"))
    settings = spec["synthetic_data"]
    noise_spec = settings["noise_implementation"]
    require(sha256_file(args.noise_bank) == noise_spec["bank_sha256"], "residual noise bank hash mismatch")
    require(bool(args.noise_provenance), "historical noise provenance is required to prove recipe equivalence")
    bank = ResidualNoiseBank.load(args.noise_bank, provenance=args.noise_provenance)
    require(len(bank.residuals) == noise_spec["snippet_count"], "residual bank snippet count mismatch")
    recipes = {s: SyntheticRecipe(s, settings[f"{s}_objects"], settings["master_seeds"][s],
                                EmpiricalNoiseRecipe(bank.provenance),
                                lambert_coefficient=spec["renderer"]["lambert_coefficient"])
               for s in OLD_HASHES}
    # Fail on recipe drift before spending time discovering donors.
    for split, recipe in recipes.items():
        recipe_versions(recipe, OLD_HASHES[split])
    rows = [json.loads(line) for line in args.catalog.read_text(encoding="utf-8").splitlines() if line.strip()]
    selection = settings["donor_selection"]
    donors = discover_damit_donors(
        args.dump_root, excluded_object_ids=[row["object_id"] for row in rows],
        minimum_epochs=selection["minimum_source_epochs"],
        minimum_observations=selection["minimum_observations"],
    )
    require(len(donors) == selection["validated_donor_count"], "discovered donor count mismatch")
    partitions = {s: tuple(sorted(ids)) for s, ids in partition_damit_donors(
        donors, master_seed=selection["partition_master_seed"]).items()}
    shapes = {}
    checks = {}
    for split, ids in partitions.items():
        require(len(ids) == selection["split_counts"][split], f"{split}: donor count mismatch")
        shapes[split] = {d: load_damit_object(args.dump_root, d).shape.source_id for d in ids}
        checks[split] = check_frozen_split(args.frozen_dir / split, recipes[split], ids, shapes[split])
    result = {"schema": "delphi.k3-synthetic-regeneration.v1", "status": "passed",
              "frozen_dir": str(args.frozen_dir.resolve()), "dump_root": str(args.dump_root.resolve()),
              "catalog_sha256": sha256_file(args.catalog), "noise_bank_sha256": bank.sha256,
              "noise_provenance": bank.provenance, "shard_size": SHARD_SIZE, "splits": checks}
    return result, recipes, partitions, shapes


def circular_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    first = np.sin(first - np.angle(np.mean(np.exp(1j * first))))
    second = np.sin(second - np.angle(np.mean(np.exp(1j * second))))
    scale = np.linalg.norm(first) * np.linalg.norm(second)
    return float(np.dot(first, second) / scale) if scale > 0 else None


def stored_statistics(paths: list[Path], recipe: SyntheticRecipe) -> tuple[dict, np.ndarray, np.ndarray]:
    axes, periods, phases = [], [], []
    epochs = observations = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            axes.append(archive["target_axes"])
            periods.append(archive["periods_hours"])
            epochs += len(archive["epoch_offsets"]) - 1
            observations += int(archive["epoch_offsets"][-1])
            if "initial_phases" in archive.files:
                phases.append(archive["initial_phases"])
    axis = np.concatenate(axes)
    z = axis[:, 2]
    log_period = np.log(np.concatenate(periods))
    low, high = np.log(recipe.period_range_hours)
    q = 2 * (log_period - low) / (high - low) - 1
    phase_stored = len(phases) == len(paths)
    correlation = circular_correlation(np.arctan2(axis[:, 1], axis[:, 0]), np.concatenate(phases)) if phase_stored else None
    result = {"objects": len(z), "shards": len(paths), "epochs": epochs, "observations": observations,
              "spearman_abs_z_abs_normalized_log_period": float(spearmanr(np.abs(z), np.abs(q)).statistic),
              "normalized_log_period_definition": "2*(log(P)-log(2))/log(100)-1",
              "longitude_initial_phase_circular_correlation": correlation,
              "initial_phase_status": "stored radians" if phase_stored else "not stored; correlation unavailable",
              "circular_correlation_method": "Jammalamadaka-Sarma on canonical pole longitude",
              "ks_z_uniform_pvalue": float(kstest(z, "uniform", args=(0, 1)).pvalue),
              "ks_z_reference": "uniform[0,1]: stored poles are antipodally canonicalized",
              "ks_log_period_uniform_pvalue": float(kstest(log_period, "uniform", args=(low, high-low)).pvalue)}
    return result, z, log_period


# Each worker retains only the current split's donors. Tasks use global indices
# exactly as generate_damit_synthetic_shards, never a shard-local master seed.
_WORKER_DONORS = {}
_WORKER_BANK = None


def initialize_worker(noise_path: Path, provenance: str) -> None:
    global _WORKER_BANK
    torch.set_num_threads(1)
    _WORKER_BANK = ResidualNoiseBank.load(noise_path, provenance=provenance)


def generate_shard(dump: Path, ids: tuple[str, ...], recipe: SyntheticRecipe,
                   directory: Path, index: int) -> dict:
    if tuple(_WORKER_DONORS) != ids:
        _WORKER_DONORS.clear()
        _WORKER_DONORS.update((d, load_damit_object(dump, d)) for d in ids)
    started = time.perf_counter()
    records = []
    for i in range(index * SHARD_SIZE, min((index + 1) * SHARD_SIZE, recipe.object_count)):
        geometry = _WORKER_DONORS[ids[i % len(ids)]]
        shape = _WORKER_DONORS[ids[(i * 9973 + 17) % len(ids)]]
        records.append(generate_synthetic_record(
            object_id=f"synthetic-{recipe.split}-{i:06d}", geometry_donor_id=geometry.object_id,
            geometry_epochs=geometry.epochs, shape=shape.shape, recipe=recipe,
            seed=recipe.master_seed + i, residual_noise_bank=_WORKER_BANK))
    path = directory / f"{recipe.split}-{index:04d}.npz"
    write_synthetic_shard(path, records, recipe=recipe, shard_index=index)
    return {"split": recipe.split, "shard_index": index,
            "render_and_write_seconds": time.perf_counter() - started}


def generate(args: argparse.Namespace, check: dict, recipes: dict, partitions: dict, shapes: dict) -> None:
    root = args.output_dir
    if root.exists():
        require(root.is_dir(), "--output-dir must be a directory")
        if any(root.iterdir()):
            require(args.resume, "--output-dir must be new/empty unless --resume is supplied")
            require((root / "check.json").is_file(), "resume requires the original check.json")
            require(json.loads((root / "check.json").read_text()) == json.loads(canonical_json(check)),
                    "resume input/recipe check differs from the original run")
    root.mkdir(parents=True, exist_ok=True)
    require(not {p.name for p in root.iterdir()} - {"check.json", *OLD_HASHES}, "unexpected output directory contents")
    _atomic_json(root / "check.json", check)
    tasks = []
    for split in args.splits:
        directory = root / split
        directory.mkdir(exist_ok=True)
        recipe = recipes[split]
        all_paths = [directory / f"{split}-{i:04d}.npz" for i in range(math.ceil(recipe.object_count / SHARD_SIZE))]
        allowed = {"recipe.json", "summary.json"} | {p.name for p in all_paths} | {p.name + ".manifest.json" for p in all_paths}
        require(not {p.name for p in directory.iterdir()} - allowed, f"{split}: unexpected output files")
        recipe_path = directory / "recipe.json"
        if recipe_path.exists():
            require(json.loads(recipe_path.read_text()) == json.loads(canonical_json(asdict(recipe))),
                    f"{split}: existing recipe mismatch")
        _atomic_json(recipe_path, asdict(recipe))
        for index, path in enumerate(all_paths):
            if path.exists() or path.with_suffix(".npz.manifest.json").exists():
                require(args.resume, f"refusing to overwrite {path}")
                check_shard(path, recipe, partitions[split], shapes[split], index, legacy=False)
            elif args.max_shards is None or index < args.max_shards:
                tasks.append((args.dump_root, partitions[split], recipe, directory, index))
    started = time.perf_counter()
    timings = []
    if args.workers == 1:
        initialize_worker(args.noise_bank, args.noise_provenance)
        for task in tasks:
            result = generate_shard(*task)
            timings.append(result)
            print(canonical_json(result), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=initialize_worker,
                                 initargs=(args.noise_bank, args.noise_provenance)) as executor:
            futures = [executor.submit(generate_shard, *task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                timings.append(result)
                print(canonical_json(result), flush=True)
    generation_seconds = time.perf_counter() - started
    for split in args.splits:
        directory = root / split
        recipe = recipes[split]
        paths = sorted(directory.glob("*.npz"))
        if not paths:
            continue
        current, z, log_period = stored_statistics(paths, recipe)
        frozen, old_z, old_log_period = stored_statistics(sorted((args.frozen_dir / split).glob("*.npz")), recipe)
        complete = current["objects"] == recipe.object_count
        if args.max_shards is None:
            require(complete, f"{split}: incomplete generated split")
            require(current["epochs"] == frozen["epochs"] and current["observations"] == frozen["observations"],
                    f"{split}: generated epoch/observation counts differ from frozen")
        summary = {"split": split, "complete": complete, "expected_objects": recipe.object_count,
                   "recipe_sha256": recipe.sha256, "frozen_recipe_sha256": OLD_HASHES[split],
                   "generated": current, "frozen_v1": frozen,
                   "ks_generated_vs_frozen_z_pvalue": float(ks_2samp(z, old_z).pvalue),
                   "ks_generated_vs_frozen_log_period_pvalue": float(ks_2samp(log_period, old_log_period).pvalue),
                   "workers": args.workers, "generation_wall_seconds": generation_seconds,
                   "new_shard_timings": sorted((t for t in timings if t["split"] == split), key=lambda t: t["shard_index"])}
        _atomic_json(directory / "summary.json", summary)
        print(canonical_json(summary), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--splits", nargs="+", choices=tuple(OLD_HASHES), default=list(OLD_HASHES))
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check", action="store_true", help="verify all frozen splits without writing or rendering")
    parser.add_argument("--max-shards", type=int, help="generate only this many leading shards per split for a quick check")
    parser.add_argument("--dump-root", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=REPOSITORY / "repro/data/damit-20250610T000301Z/catalog.jsonl")
    parser.add_argument("--noise-bank", type=Path, required=True)
    parser.add_argument("--noise-provenance", default=NOISE_PROVENANCE,
                        help="exact historical recipe string; v1 hash verification is mandatory")
    args = parser.parse_args(argv)
    if not args.check and args.output_dir is None:
        parser.error("--output-dir is required for generation")
    if args.max_shards is not None and args.max_shards < 1:
        parser.error("--max-shards must be positive")
    args.splits = list(dict.fromkeys(args.splits))
    torch.set_num_threads(1)
    try:
        check, recipes, partitions, shapes = check_inputs(args)
        print(canonical_json(check), flush=True)
        if not args.check:
            generate(args, check, recipes, partitions, shapes)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"K3 synthetic generation refused: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
