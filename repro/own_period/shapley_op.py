"""Group Shapley values (repro/group_shapley.py) at a chosen period.

    python -m repro.own_period.shapley_op run --periods damit --out-dir D --folds 0 --seeds 17
    python -m repro.own_period.shapley_op run --periods own-periods-170.json --out-dir O
    python -m repro.own_period.shapley_op summary --runs O --output S.json [--compare-published]

``run`` is ``group_shapley.main`` with the examples folded at the chosen period
(same grid, coalitions, removal values, fold-wise shared means and batch size).
``summary`` is ``group_shapley_summary.main`` with explicit input and output
paths (same flat-map correction, bootstrap seed 20260926), plus the numbers
quoted in the text: per-seed share ranges (as the Shapley figure), and the
fractions of asteroids for which a group lowers or raises the grid error.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch

from lc_pipeline.k3.config import K3ScoreModelConfig
from lc_pipeline.k3.grid import axial_healpix_grid
from lc_pipeline.k3.model import CandidateConditionedScorer
from lc_pipeline.k3.training import collate_examples
from repro import group_shapley as gs
from repro import group_shapley_summary as su

from .nettests_common import FOLLOWUP, MODELS, SEEDS, fold_rows, load_examples, max_abs, period_override, write_json_once

PUBLISHED = FOLLOWUP / "data/interpretation-20260926/shapley"


def run(args) -> int:
    periods, provenance, label = period_override(args.periods)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = args.out_dir / "period-source.json"
    if stamp.exists():
        if json.loads(stamp.read_text())["period_source"] != label:
            raise SystemExit("output directory belongs to another period source")
    else:
        with stamp.open("x") as stream:
            json.dump({"period_source": label, "provenance": provenance, "spec": args.periods}, stream)
    grid = axial_healpix_grid(args.nside)
    fast_modes = gs.FastModes(grid)
    grid_tensor = torch.as_tensor(np.array(grid.vectors, copy=True), dtype=torch.float32, device=args.device)
    rows = {int(r["fold"]): r for r in fold_rows()}
    cache = {}
    for seed in args.seeds:
        for fold in args.folds:
            output = args.out_dir / f"fold-{fold}-seed-{seed}.npz"
            if output.exists():
                continue
            if fold not in cache:
                examples = load_examples(tuple(rows[fold]["test_ids"]), periods, provenance)
                batches = [
                    {k: v for k, v in collate_examples(examples[i : i + 4]).items() if k in gs.INPUT_KEYS}
                    for i in range(0, len(examples), 4)
                ]
                cache[fold] = (examples, gs.shared_channel_means(batches, args.device))
            examples, shared_means = cache[fold]
            checkpoint = torch.load(MODELS / f"real-fold-{fold}-seed-{seed}.pt", map_location="cpu", weights_only=False)
            model = CandidateConditionedScorer(K3ScoreModelConfig(**checkpoint["model_config"]))
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(args.device).eval()
            started = time.time()
            percentile, grid_error = gs.coalition_values(model, examples, args.device, shared_means, grid_tensor, fast_modes)
            with output.open("xb") as handle:
                np.savez_compressed(
                    handle,
                    object_ids=np.array([e.object_id for e in examples]),
                    groups=np.array(gs.GROUPS),
                    nside=np.array(args.nside),
                    percentile=percentile,
                    grid_error=grid_error,
                    shapley_percentile=gs.shapley(percentile),
                    shapley_grid_error=gs.shapley(grid_error),
                    shared_means=np.array([shared_means[c] for c in (3, 4, 5, 6)]),
                    period_hours=np.array([e.tokenized.period_hours for e in examples]),
                )
            print(json.dumps({"fold": fold, "seed": seed, "seconds": round(time.time() - started, 1),
                              "full_percentile": float(percentile[:, 255].mean()),
                              "full_grid_error": float(grid_error[:, 255].mean())}), flush=True)
    return 0


def corrected(files, catalog):
    """Per-file corrected combination values, exactly as group_shapley_summary."""
    masks = su.flat_map_masks()
    for name in files:
        data = np.load(name)
        grid = axial_healpix_grid(int(data["nside"]))
        flat_axes = gs.FastModes(grid).vectors[gs.FastModes(grid)(np.zeros(len(grid.vectors)))]
        percentile = data["percentile"].copy()
        grid_error = data["grid_error"].copy()
        for index, object_id in enumerate(data["object_ids"]):
            flat_error = gs.axial_error(flat_axes, catalog[str(object_id)])
            for mask in masks:
                percentile[index, mask] = 0.5
                grid_error[index, mask] = flat_error
        seed = int(Path(name).stem.split("seed-")[1])
        yield seed, data["object_ids"].astype(str), percentile, grid_error, int(data["nside"])


def summary(args) -> int:
    files = sorted(glob.glob(str(args.runs / "fold-*-seed-*.npz")))
    if not files:
        raise SystemExit("no Shapley runs found")
    catalog = {}
    for line in gs.CATALOG.read_text().splitlines():
        row = json.loads(line)
        catalog[row["object_id"]] = np.array([s["vector"] for s in row["solutions"]], dtype=float)
    per_object, per_seed, nsides = {}, {}, set()
    for seed, ids, percentile, grid_error, nside in corrected(files, catalog):
        nsides.add(nside)
        sp = gs.shapley(percentile)
        per_seed.setdefault(seed, []).append(sp)
        for index, object_id in enumerate(ids):
            per_object.setdefault(object_id, []).append(
                (sp[index], gs.shapley(grid_error[index]), percentile[index, 255], percentile[index, 0],
                 grid_error[index, 255], grid_error[index, 0]))
    names = sorted(per_object)
    shap_p = np.array([np.mean([r[0] for r in per_object[n]], axis=0) for n in names])
    shap_e = np.array([np.mean([r[1] for r in per_object[n]], axis=0) for n in names])
    full_p = np.array([np.mean([r[2] for r in per_object[n]]) for n in names])
    empty_p = np.array([np.mean([r[3] for r in per_object[n]]) for n in names])
    full_e = np.array([np.mean([r[4] for r in per_object[n]]) for n in names])
    empty_e = np.array([np.mean([r[5] for r in per_object[n]]) for n in names])
    efficiency = float(np.max(np.abs(shap_p.sum(1) - (full_p - empty_p))))
    generator = np.random.default_rng(20260926)
    draws = generator.integers(0, len(names), size=(10000, len(names)))
    low_p, high_p = np.percentile(shap_p[draws].mean(axis=1), (2.5, 97.5), axis=0)
    low_e, high_e = np.percentile(shap_e[draws].mean(axis=1), (2.5, 97.5), axis=0)
    total_p = float((full_p - empty_p).mean())
    seed_shares = {
        seed: 100 * np.concatenate(v).mean(axis=0) / np.concatenate(v).sum(axis=1).mean()
        for seed, v in sorted(per_seed.items())
    }
    groups = []
    for g, label in enumerate(gs.GROUPS):
        shares = np.array([s[g] for s in seed_shares.values()])
        groups.append({
            "group": label,
            "percentile_mean": float(shap_p[:, g].mean()),
            "percentile_ci95": [float(low_p[g]), float(high_p[g])],
            "percentile_share": float(shap_p[:, g].mean() / total_p),
            "grid_error_mean_deg": float(shap_e[:, g].mean()),
            "grid_error_ci95_deg": [float(low_e[g]), float(high_e[g])],
            "share_by_seed_percent": {str(s): float(v[g]) for s, v in seed_shares.items()},
            "share_seed_range_percent": [float(shares.min()), float(shares.max())],
            "fraction_lowering_error_by_more_than_1_deg": float(np.mean(shap_e[:, g] < -1.0)),
            "fraction_raising_error_by_more_than_1_deg": float(np.mean(shap_e[:, g] > 1.0)),
            "fraction_lowering_error": float(np.mean(shap_e[:, g] < 0.0)),
            "fraction_raising_error": float(np.mean(shap_e[:, g] > 0.0)),
        })
    value = {
        "runs": str(args.runs),
        "period_source": json.loads((args.runs / "period-source.json").read_text())
        if (args.runs / "period-source.json").exists()
        else "published run directory",
        "networks": len(files),
        "asteroids": len(names),
        "networks_per_asteroid": sorted({len(v) for v in per_object.values()}),
        "nside": sorted(nsides),
        "full_percentile": float(full_p.mean()),
        "empty_percentile": float(empty_p.mean()),
        "full_grid_error_deg": float(full_e.mean()),
        "empty_grid_error_deg": float(empty_e.mean()),
        "efficiency_max_abs_error": efficiency,
        "groups": groups,
    }
    if args.compare_published:
        rows = []
        for name in files:
            new, old = np.load(name), np.load(PUBLISHED / Path(name).name)
            rows.append({
                "file": Path(name).name,
                "same_object_order": bool(np.array_equal(new["object_ids"], old["object_ids"])),
                "max_abs_percentile_diff": max_abs(new["percentile"], old["percentile"]),
                "max_abs_grid_error_diff_deg": max_abs(new["grid_error"], old["grid_error"]),
                "max_abs_shared_means_diff": max_abs(new["shared_means"], old["shared_means"]),
            })
        value["comparison_with_published"] = rows
    write_json_once(args.output, value)
    print(json.dumps({g["group"]: [round(100 * g["percentile_share"], 1), g["share_seed_range_percent"]] for g in groups}, indent=1))
    if args.compare_published:
        print(json.dumps(value["comparison_with_published"], indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    a = commands.add_parser("run")
    a.add_argument("--periods", required=True)
    a.add_argument("--out-dir", type=Path, required=True)
    a.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    a.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    a.add_argument("--device", default="cuda")
    a.add_argument("--nside", type=int, default=16)
    b = commands.add_parser("summary")
    b.add_argument("--runs", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    b.add_argument("--compare-published", action="store_true")
    args = parser.parse_args()
    return run(args) if args.command == "run" else summary(args)


if __name__ == "__main__":
    raise SystemExit(main())
