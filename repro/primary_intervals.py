"""Bootstrap intervals of the primary DeLPHI result and the random-axes CDF over 10,000 draws.

Reads the per-object oracle errors of repro/compute_reference_baselines.py and the DAMIT
reference axes.  Writes data/interpretation-20260926/primary-intervals.json and
random-axes-cdf.npz (mean cumulative distribution of the oracle error of three random axes over
10,000 draws per asteroid, with the 2.5 and 97.5 percentiles of the per-draw curves).
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FOLLOWUP = HERE.parents[1]
BASELINES = HERE / "baselines_out/reference-baselines-per-object.npz"
CATALOG = FOLLOWUP / "inputs/training-run/repro/data/damit-20250610T000301Z/catalog.jsonl"
OUT = FOLLOWUP / "data/interpretation-20260926"


def main():
    data = np.load(BASELINES)
    errors = data["delphi_errors_deg"]
    draws = np.random.default_rng(20260930).integers(0, len(errors), (10000, len(errors)))
    within = errors <= 20
    summary = dict(
        n=int(len(errors)),
        mean_deg=float(errors.mean()), mean_ci95_deg=[float(v) for v in np.percentile(errors[draws].mean(1), (2.5, 97.5))],
        within20=float(within.mean()), within20_ci95=[float(v) for v in np.percentile(within[draws].mean(1), (2.5, 97.5))],
    )
    catalog = {r["object_id"]: np.array([s["vector"] for s in r["solutions"]], dtype=float)
               for r in map(json.loads, CATALOG.read_text().splitlines())}
    generator = np.random.default_rng(20260931)
    grid = np.arange(0.0, 90.01, 0.25)
    per_draw = np.zeros((10000, len(grid)))
    random_errors = []
    for object_id in data["object_ids"]:
        reference = catalog[str(object_id)]
        reference /= np.linalg.norm(reference, axis=1, keepdims=True)
        axes = generator.normal(size=(10000, 3, 3))
        axes /= np.linalg.norm(axes, axis=2, keepdims=True)
        cosine = np.abs(np.einsum("dkx,rx->dkr", axes, reference)).max(axis=(1, 2))
        error = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        random_errors.append(error)
        per_draw += error[:, None] <= grid[None, :]
    per_draw /= len(data["object_ids"])
    summary["random_mean_over_draws_deg"] = float(np.mean(random_errors))
    np.savez(OUT / "random-axes-cdf.npz", grid_deg=grid, mean_cdf=per_draw.mean(0),
             low_cdf=np.percentile(per_draw, 2.5, axis=0), high_cdf=np.percentile(per_draw, 97.5, axis=0))
    (OUT / "primary-intervals.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
