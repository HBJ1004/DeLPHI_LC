"""Check that the DAMIT-period rerun reproduces the published census axes.

    python -m repro.own_period.lowq_validate --rerun V.jsonl --output V-check.json

Compares, asteroid by asteroid in the published order, the status, the three
selected grid indices and the three refined axes of the rerun with the
published ``predictions-fold0-five-seed/predictions.jsonl``.  No reference
pole is read here (the errors are compared later, in ``lowq_score``).
Exit code 1 if any axis differs by more than 1e-6 deg.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .lowq_search import CENSUS, sha256

PUBLISHED = CENSUS / "predictions-fold0-five-seed/predictions.jsonl"
PUBLISHED_SHA256 = "b3adff514191f4f72f325587c64342859224a75333aa9b7ade18bf2e6ae8f861"
TOLERANCE_DEG = 1e-6


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256(PUBLISHED) != PUBLISHED_SHA256:
        raise ValueError("published predictions differ from the census manifest")
    published = {}
    with PUBLISHED.open() as handle:
        for line in handle:
            item = json.loads(line)
            published[item["object_id"]] = item
    rerun = [json.loads(line) for line in args.rerun.open() if line.strip()]
    order = list(published)[: len(rerun)]
    if [r["object_id"] for r in rerun] != order:
        raise ValueError("rerun is not the first N asteroids in the published order")
    worst, status_mismatch, index_mismatch, bit_identical = 0.0, [], [], 0
    for new in rerun:
        old = published[new["object_id"]]
        if new["status"] != old["status"]:
            status_mismatch.append(new["object_id"])
            continue
        if new["status"] != "ok":
            continue
        a = np.asarray([x["axis_xyz"] for x in old["axes"]])
        b = np.asarray([x["axis_xyz"] for x in new["axes"]])
        angle = np.degrees(np.arccos(np.clip(np.abs(np.sum(a * b, axis=1)), 0.0, 1.0)))
        worst = max(worst, float(angle.max()))
        bit_identical += int(np.array_equal(a, b))
        if [x["grid_index"] for x in old["axes"]] != [x["grid_index"] for x in new["axes"]]:
            index_mismatch.append(new["object_id"])
    ok_count = sum(r["status"] == "ok" for r in rerun)
    report = {
        "schema": "delphi.own-period-lowq-validation.v1",
        "subset": f"first {len(rerun)} asteroids of the published census order",
        "rerun": str(args.rerun),
        "rerun_sha256": sha256(args.rerun),
        "published_predictions_sha256": PUBLISHED_SHA256,
        "rows": len(rerun),
        "ok_rows": ok_count,
        "status_mismatches": status_mismatch,
        "grid_index_mismatches": index_mismatch,
        "bit_identical_axis_sets": bit_identical,
        "max_axis_angle_deg": worst,
        "tolerance_deg": TOLERANCE_DEG,
        "passed": not status_mismatch and not index_mismatch and worst <= TOLERANCE_DEG,
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
