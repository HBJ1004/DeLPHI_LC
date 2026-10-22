"""Summary-level validation of the DAMIT-period Shapley rerun (fold 0, seed 17).

    python -m repro.own_period.shapley_check --output CHECK.json

Compares (1) the rerun with the published run of the same network and (2) the
rerun with a repeat of itself.  Cell by cell the published run is not
reproduced exactly (near-tied peaks of the score map change the grid error of
single combinations), so the gate is on the quantities that are reported:
per-group shares of the percentile measure (tolerance 0.05 percentage points),
per-group mean effects on the grid error (0.05 deg), full-input percentile
(1e-3) and grid error (0.05 deg).  The repeat must be bit-identical.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from .nettests_common import OUT, max_abs, write_json_once
from .shapley_op import PUBLISHED, compute_summary


def compact(summary):
    return {
        "full_percentile": summary["full_percentile"],
        "full_grid_error_deg": summary["full_grid_error_deg"],
        "share_percent": {g["group"]: 100 * g["percentile_share"] for g in summary["groups"]},
        "grid_error_mean_deg": {g["group"]: g["grid_error_mean_deg"] for g in summary["groups"]},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    name = "fold-0-seed-17.npz"
    rerun, repeat = OUT / "shapley-damit", OUT / "shapley-damit-repeat"
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / name).symlink_to(PUBLISHED / name)
        pub = compact(compute_summary(Path(tmp)))
    new = compact(compute_summary(rerun))
    share = max(abs(new["share_percent"][k] - pub["share_percent"][k]) for k in pub["share_percent"])
    effect = max(abs(new["grid_error_mean_deg"][k] - pub["grid_error_mean_deg"][k]) for k in pub["grid_error_mean_deg"])
    a, b = np.load(rerun / name), np.load(repeat / name)
    repeat_diff = max(max_abs(a["percentile"], b["percentile"]), max_abs(a["grid_error"], b["grid_error"]))
    value = {
        "published": pub,
        "rerun": new,
        "max_abs_share_diff_percentage_points": share,
        "max_abs_effect_diff_deg": effect,
        "full_percentile_diff": abs(new["full_percentile"] - pub["full_percentile"]),
        "full_grid_error_diff_deg": abs(new["full_grid_error_deg"] - pub["full_grid_error_deg"]),
        "repeat_max_abs_diff": repeat_diff,
    }
    value["passes"] = bool(
        share <= 0.05 and effect <= 0.05 and value["full_percentile_diff"] <= 1e-3
        and value["full_grid_error_diff_deg"] <= 0.05 and repeat_diff == 0.0
    )
    write_json_once(args.output, value)
    print(json.dumps({k: v for k, v in value.items() if k not in ("published", "rerun")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
