"""Candidate spread of the label-shuffle control (Section 6.1.1).

For the seed-17 simulation network trained on correct axes and the one trained
on shuffled axes, compute on the 2,000 simulated test asteroids
  - the mean oracle error,
  - the largest angle between any two of the three candidate axes (axial),
  - the spread of the score map (standard deviation over trial axes),
and the mean oracle error of three independent uniform random axes against the
single reference axis of each simulated asteroid, analytically
(integral of cos^3 over [0, pi/2] = 2/3 rad) and by simulation.
Writes data/interpretation-20260930/label-shuffle-spread.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lc_pipeline.k3.evaluation import modes_from_score_grid  # noqa: E402

FOLLOWUP = HERE.parents[1]
EVALUATIONS = FOLLOWUP / "inputs/frozen-artifacts/k3-definitive-7874092/evaluations"


def spread(path):
    data = np.load(path)
    largest = []
    for grid in data["score_grids"]:
        axes = np.array([mode.axis_xyz for mode in modes_from_score_grid(grid.astype(np.float32))])
        cosines = np.clip(np.abs(axes @ axes.T)[np.triu_indices(3, 1)], 0, 1)
        largest.append(np.degrees(np.arccos(cosines)).max())
    largest = np.array(largest)
    return {
        "mean_oracle_error_deg": float(data["oracle_errors_deg"].mean()),
        "largest_candidate_separation_deg_median": float(np.median(largest)),
        "largest_candidate_separation_deg_iqr": [float(v) for v in np.percentile(largest, (25, 75))],
        "score_map_std_median": float(np.median(data["score_grids"].std(axis=1))),
        "n_objects": int(len(largest)),
    }


def random_three_axes(draws=200000, seed=20260930):
    generator = np.random.default_rng(seed)
    axes = generator.normal(size=(draws, 3, 3))
    axes /= np.linalg.norm(axes, axis=2, keepdims=True)
    # The reference axis can be fixed at +z by symmetry.
    angles = np.degrees(np.arccos(np.clip(np.abs(axes[:, :, 2]), 0, 1))).min(axis=1)
    return float(angles.mean())


def main():
    out = {
        "correct_axes": spread(EVALUATIONS / "synthetic-test-seed-17.npz"),
        "shuffled_axes": spread(EVALUATIONS / "synthetic-label-shuffle-test-seed-17.npz"),
        "random_three_axes_one_reference_deg_analytic": float(np.degrees(2.0 / 3.0)),
        "random_three_axes_one_reference_deg_simulated": random_three_axes(),
    }
    target = FOLLOWUP / "data/interpretation-20260930/label-shuffle-spread.json"
    target.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
