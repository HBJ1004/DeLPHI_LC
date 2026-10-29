"""Compare the DAMIT-period rerun of predict_oof_full.py with the published files.

    python -m repro.own_period.maps.validate_damit_run RUN_DIR OUTPUT.json

Exits with status 1 if the rerun does not reproduce the published results.
"""
import json
import sys
from pathlib import Path

import numpy as np

PUBLISHED = Path(__file__).resolve().parents[4] / (
    "inputs/frozen-artifacts/k3-definitive-7874092/evaluations")
SEEDS = (17, 42, 137, 777, 2027)


def axis_angle_deg(a, b):
    """Axial angle, computed from the cross product (precise near zero)."""
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.abs(np.sum(a * b, -1))))


def main():
    run, output = Path(sys.argv[1]) / "evaluations", Path(sys.argv[2])
    p = np.load(PUBLISHED / "real-oof-ensemble.npz")
    r = np.load(run / "real-oof-ensemble.npz")
    out = {"keys_identical": sorted(p.files) == sorted(r.files),
           "dtypes_identical": all(p[k].dtype == r[k].dtype for k in p.files),
           "shapes_identical": all(p[k].shape == r[k].shape for k in p.files),
           "ids_identical": bool((p["object_ids"] == r["object_ids"]).all()),
           "folds_identical": bool((p["folds"] == r["folds"]).all()),
           "checkpoint_sha256_identical": bool((p["checkpoint_sha256"] == r["checkpoint_sha256"]).all())}
    out["score_grid_max_abs_diff"] = float(np.abs(p["score_grids"].astype(float) - r["score_grids"]).max())
    out["initial_axes_max_angle_deg"] = float(axis_angle_deg(p["initial_axes"], r["initial_axes"]).max())
    out["refined_axes_max_angle_deg"] = float(axis_angle_deg(p["refined_axes"], r["refined_axes"]).max())
    out["deployed_mode_indices_mismatch"] = int((p["deployed_mode_indices"] != r["deployed_mode_indices"]).any(1).sum())
    out["oracle_error_max_abs_diff_deg"] = float(np.abs(p["oracle_errors_deg"] - r["oracle_errors_deg"]).max())
    out["grid_oracle_error_max_abs_diff_deg"] = float(np.abs(p["grid_oracle_errors_deg"] - r["grid_oracle_errors_deg"]).max())
    out["mean_published_deg"] = float(p["oracle_errors_deg"].mean())
    out["mean_rerun_deg"] = float(r["oracle_errors_deg"].mean())
    out["grid_mean_published_deg"] = float(p["grid_oracle_errors_deg"].mean())
    out["grid_mean_rerun_deg"] = float(r["grid_oracle_errors_deg"].mean())
    out["inference_median_s"] = {"published": float(np.median(p["inference_wall_seconds"])),
                                 "rerun": float(np.median(r["inference_wall_seconds"]))}
    seeds = {}
    for s in SEEDS:
        grid_diff, err_diff, pub, new = 0.0, 0.0, [], []
        for f in range(5):
            a = np.load(PUBLISHED / f"real-fold-{f}-seed-{s}.npz")
            b = np.load(run / f"real-fold-{f}-seed-{s}.npz")
            assert a.files == b.files and (a["object_ids"] == b["object_ids"]).all()
            grid_diff = max(grid_diff, float(np.abs(a["score_grids"].astype(float) - b["score_grids"]).max()))
            err_diff = max(err_diff, float(np.abs(a["oracle_errors_deg"] - b["oracle_errors_deg"]).max()))
            pub.append(a["oracle_errors_deg"])
            new.append(b["oracle_errors_deg"])
        seeds[str(s)] = {"published_mean_deg": float(np.concatenate(pub).mean()),
                         "rerun_mean_deg": float(np.concatenate(new).mean()),
                         "score_grid_max_abs_diff": grid_diff, "oracle_error_max_abs_diff_deg": err_diff}
    out["per_seed"] = seeds
    out["passed"] = bool(
        out["keys_identical"] and out["ids_identical"] and out["folds_identical"]
        and out["checkpoint_sha256_identical"] and out["refined_axes_max_angle_deg"] < 1e-3
        and out["oracle_error_max_abs_diff_deg"] < 1e-4
        and all(abs(v["published_mean_deg"] - v["rerun_mean_deg"]) < 1e-3 for v in seeds.values()))
    with output.open("x") as handle:
        handle.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
