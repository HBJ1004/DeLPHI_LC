"""Aggregate the group Shapley runs of repro/group_shapley.py.

Correction applied here to every network: when the Sun, observer and product
terms are all removed, the score no longer depends on the trial axis, so every
axis has the same score mathematically.  In the stored runs the reference axis
was scored in a separate batch from the grid, and differences of order 1e-7
between the two batch shapes broke this tie arbitrarily.  For these 64 of the
256 combinations we therefore set the percentile to its exact value of 0.5 and
the grid error to that of the peaks of a flat map, which the locked peak rule
defines uniquely (lowest grid indices).  All other combinations are unchanged.

The Shapley values are recomputed from the corrected combination values and
averaged first over the networks of each asteroid (one per seed) and then over
asteroids, with 10,000 bootstrap resamples of asteroids for the intervals.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import group_shapley as gs  # noqa: E402
from lc_pipeline.k3.grid import axial_healpix_grid  # noqa: E402

FOLLOWUP = HERE.parents[1]
RUNS = FOLLOWUP / "data/interpretation-20260926/shapley"
OUTPUT = FOLLOWUP / "data/interpretation-20260926/shapley-summary.json"
AXIS_GROUPS = (4, 5, 6)   # Sun term, observer term, product term


def flat_map_masks():
    return [mask for mask in range(256) if not any(mask >> g & 1 for g in AXIS_GROUPS)]


def main():
    files = sorted(glob.glob(str(RUNS / "fold-*-seed-*.npz")))
    if not files:
        raise SystemExit("no Shapley runs found")
    catalog = {}
    for line in gs.CATALOG.read_text().splitlines():
        row = json.loads(line)
        catalog[row["object_id"]] = np.array([s["vector"] for s in row["solutions"]], dtype=float)
    per_object = {}
    nsides = set()
    for name in files:
        data = np.load(name)
        nside = int(data["nside"])
        nsides.add(nside)
        grid = axial_healpix_grid(nside)
        flat_axes = gs.FastModes(grid).vectors[gs.FastModes(grid)(np.zeros(len(grid.vectors)))]
        percentile = data["percentile"].copy()
        grid_error = data["grid_error"].copy()
        for index, object_id in enumerate(data["object_ids"]):
            flat_error = gs.axial_error(flat_axes, catalog[str(object_id)])
            for mask in flat_map_masks():
                percentile[index, mask] = 0.5
                grid_error[index, mask] = flat_error
        for index, object_id in enumerate(data["object_ids"]):
            per_object.setdefault(str(object_id), []).append(
                (gs.shapley(percentile[index]), gs.shapley(grid_error[index]),
                 percentile[index, 255], percentile[index, 0], grid_error[index, 255], grid_error[index, 0]))
    names = sorted(per_object)
    shap_p = np.array([np.mean([r[0] for r in per_object[n]], axis=0) for n in names])
    shap_e = np.array([np.mean([r[1] for r in per_object[n]], axis=0) for n in names])
    full_p = np.array([np.mean([r[2] for r in per_object[n]]) for n in names])
    empty_p = np.array([np.mean([r[3] for r in per_object[n]]) for n in names])
    full_e = np.array([np.mean([r[4] for r in per_object[n]]) for n in names])
    empty_e = np.array([np.mean([r[5] for r in per_object[n]]) for n in names])
    # Shapley efficiency: values add up to full minus empty for every asteroid.
    efficiency = float(np.max(np.abs(shap_p.sum(1) - (full_p - empty_p))))
    generator = np.random.default_rng(20260926)
    draws = generator.integers(0, len(names), size=(10000, len(names)))

    def interval(values):
        means = values[draws].mean(axis=1)
        return [float(v) for v in np.percentile(means, (2.5, 97.5), axis=0).T.ravel()]

    groups = []
    low_p, high_p = np.percentile(shap_p[draws].mean(axis=1), (2.5, 97.5), axis=0)
    low_e, high_e = np.percentile(shap_e[draws].mean(axis=1), (2.5, 97.5), axis=0)
    total_p = float((full_p - empty_p).mean())
    for g, label in enumerate(gs.GROUPS):
        groups.append({
            "group": label,
            "percentile_mean": float(shap_p[:, g].mean()),
            "percentile_ci95": [float(low_p[g]), float(high_p[g])],
            "percentile_share": float(shap_p[:, g].mean() / total_p),
            "grid_error_mean_deg": float(shap_e[:, g].mean()),
            "grid_error_ci95_deg": [float(low_e[g]), float(high_e[g])],
        })
    summary = {
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
    OUTPUT.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
