"""Compare DeLPHI score maps with the classical amplitude-aspect relation.

For every trial axis p of the 6,144-axis grid, the aspect angle phi_i of each
lightcurve i is the angle between p and the mean asteroid-to-observer
direction of that lightcurve.  A triaxial ellipsoid with semi-axes a >= b >= c
rotating about c has the amplitude (Zappala et al. 1990)

    A(phi) = 1.25 log10[(a^2 sin^2 phi + c^2 cos^2 phi) / (b^2 sin^2 phi + c^2 cos^2 phi)].

We fit a/b and b/c on a fixed grid for each trial axis and use the negative
residual sum of squares as a physical score map.  Three candidates are taken
from it with the same mode extraction as DeLPHI, and both maps are compared.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from lc_pipeline.k3.evaluation import modes_from_score_grid  # noqa: E402
from lc_pipeline.k3.grid import axial_healpix_grid  # noqa: E402
import compute_reference_baselines as base  # noqa: E402

FOLLOWUP = HERE.parents[1]
ENSEMBLE = FOLLOWUP / "inputs/frozen-artifacts/k3-definitive-7874092/evaluations/real-oof-ensemble.npz"
DUMP = FOLLOWUP.parent / "damit-20250610T000301Z/files"
MIN_POINTS = 15
AB = np.linspace(1.0, 3.0, 41)
BC = np.linspace(1.0, 2.0, 21)


def lightcurve_amplitudes(object_id):
    tokens = (DUMP / object_id / "lc.txt").read_text().split()
    count, cursor = int(tokens[0]), 1
    amplitudes, directions = [], []
    for _ in range(count):
        n = int(tokens[cursor]); cursor += 2
        rows = np.asarray(tokens[cursor:cursor + 8 * n], dtype=float).reshape(n, 8)
        cursor += 8 * n
        if n < MIN_POINTS:
            continue
        magnitude = -2.5 * np.log10(np.clip(rows[:, 1], 1e-6, None))
        low, high = np.percentile(magnitude, (2, 98))
        observer = rows[:, 5:8] / np.linalg.norm(rows[:, 5:8], axis=1, keepdims=True)
        mean = observer.mean(axis=0)
        amplitudes.append(high - low)
        directions.append(mean / np.linalg.norm(mean))
    return np.array(amplitudes), np.array(directions)


def amplitude_aspect_scores(amplitudes, directions, axes):
    sin2 = 1.0 - (axes @ directions.T) ** 2              # [axes, lightcurves]
    cos2 = 1.0 - sin2
    ab, bc = np.meshgrid(AB, BC, indexing="ij")
    a2 = (ab * bc) ** 2
    b2 = bc ** 2                                         # c = 1
    best = np.full(len(axes), np.inf)
    for a2v, b2v in zip(a2.ravel(), b2.ravel()):
        model = 1.25 * np.log10((a2v * sin2 + cos2) / (b2v * sin2 + cos2))
        rss = np.sum((model - amplitudes[None, :]) ** 2, axis=1)
        best = np.minimum(best, rss)
    return -best


def main():
    ensemble = np.load(ENSEMBLE)
    grid = axial_healpix_grid(32)
    poles = base.reference_poles()
    standard = base.standard_axes()
    rows, maps, kept = [], [], []
    for index, name in enumerate(ensemble["object_ids"]):
        name = str(name)
        amplitudes, directions = lightcurve_amplitudes(name)
        if len(amplitudes) < 3:
            continue
        physical = amplitude_aspect_scores(amplitudes, directions, grid.vectors)
        maps.append(physical.astype(np.float32)); kept.append(index)
        delphi = ensemble["score_grids"][index].astype(float)
        modes = modes_from_score_grid(physical.astype(np.float32))
        axes = np.array([mode.axis_xyz for mode in modes])
        rows.append({
            "object_id": name,
            "lightcurves_used": int(len(amplitudes)),
            "map_correlation": float(np.corrcoef(physical, delphi)[0, 1]),
            "amplitude_aspect_oracle_deg": base.axial_error_deg(axes, poles[name]),
            "delphi_oracle_deg": float(ensemble["oracle_errors_deg"][index]),
            "delphi_grid_oracle_deg": float(ensemble["grid_oracle_errors_deg"][index]),
            "standard_oracle_deg": base.axial_error_deg(standard, poles[name]),
        })
    output = FOLLOWUP / "data/interpretation-20260923/amplitude-aspect.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=1))
    maps = np.array(maps)
    delphi_maps = ensemble["score_grids"][kept].astype(float)
    np.savez_compressed(output.with_suffix(".npz"), object_ids=np.array([rows[i]["object_id"] for i in range(len(rows))]),
                        amplitude_aspect_maps=maps)
    # Null: DeLPHI map of one asteroid against the amplitude-aspect map of another.
    generator = np.random.default_rng(20260923)
    null = []
    for _ in range(20):
        order = generator.permutation(len(kept))
        while np.any(order == np.arange(len(kept))):
            order = generator.permutation(len(kept))
        null.append(np.median([np.corrcoef(maps[order[i]], delphi_maps[i])[0, 1] for i in range(len(kept))]))
    print("cross-asteroid null median correlation", np.round(np.mean(null), 3), "range", np.round(min(null), 3), np.round(max(null), 3))
    table = {key: np.array([row[key] for row in rows]) for key in rows[0] if key != "object_id"}
    print("asteroids", len(rows))
    print("map correlation median", np.median(table["map_correlation"]).round(3),
          "IQR", np.percentile(table["map_correlation"], (25, 75)).round(3))
    for key in ("amplitude_aspect_oracle_deg", "delphi_grid_oracle_deg", "delphi_oracle_deg", "standard_oracle_deg"):
        print(key, "mean", table[key].mean().round(2), "median", np.median(table[key]).round(2),
              "within20", np.mean(table[key] < 20).round(3))


if __name__ == "__main__":
    main()
