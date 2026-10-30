"""Rebuild the fixed 140-object period comparison; no network input is needed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from repro import network_root as network
from repro.own_period import DataPath


def lightcurve_span(path):
    # Same complete eight-column input grammar as period_cost.lightcurve_span;
    # keep this compute-only producer independent of live scan/solver modules.
    words = path.read_text(encoding="ascii").split()
    cursor, times = 1, []
    for _ in range(int(words[0])):
        count, calibrated = map(int, words[cursor : cursor + 2])
        if count < 1 or calibrated not in (0, 1):
            raise ValueError("invalid lightcurve header")
        cursor += 2
        for _ in range(count):
            row = list(map(float, words[cursor : cursor + 8]))
            if len(row) != 8 or not np.isfinite(row).all():
                raise ValueError("invalid observation")
            times.append(row[0])
            cursor += 8
    if cursor != len(words) or not times:
        raise ValueError("malformed lightcurve")
    return max(times) - min(times)


def build(own_lock, damit_lock):
    own = json.loads(Path(own_lock).read_text())
    damit = json.loads(Path(damit_lock).read_text())
    by_id = {r["object_id"]: r for r in damit["objects"]}
    if len(by_id) != len(damit["objects"]) or set(by_id) != {
        r["object_id"] for r in own["objects"]
    }:
        raise ValueError("period comparison identities differ")
    rows = []
    for row in own["objects"]:
        ref = by_id[row["object_id"]]
        if row["lightcurve"]["sha256"] != ref["lightcurve"]["sha256"]:
            raise ValueError("period comparison photometry differs")
        path = Path(row["lightcurve"]["path"])
        if network.sha256(path) != row["lightcurve"]["sha256"]:
            raise ValueError("photometry hash mismatch")
        period, published = row["period_hours"], ref["period_hours"]
        span = lightcurve_span(path)
        rows.append(
            dict(
                object_id=row["object_id"],
                own_period_hours=period,
                damit_period_hours=published,
                relative_difference=abs(period - published) / published,
                span_days=span,
                drift_rotations=span * 24 * abs(1 / period - 1 / published),
            )
        )
    relative = np.array([r["relative_difference"] for r in rows])
    drift = np.array([r["drift_rotations"] for r in rows])
    return dict(
        scope="140 evaluation asteroids of the own-period benchmark; drift = span x |1/P_own - 1/P_DAMIT|",
        inputs={str(p): network.sha256(p) for p in (own_lock, damit_lock)},
        n=len(rows),
        median_relative_difference=float(np.median(relative)),
        median_relative_difference_percent=float(np.median(relative) * 100),
        median_drift_rotations=float(np.median(drift)),
        n_within_1_percent=int((relative <= 0.01).sum()),
        n_drift_above_1_rotation=int((drift > 1).sum()),
        objects=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--own-lock", type=Path, default=DataPath("benchmark/evaluation-lock.json"))
    parser.add_argument(
        "--damit-lock",
        type=Path,
        default=DataPath("../direct-workflow-benchmark/evaluation-lock.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-published", type=Path)
    args = parser.parse_args()
    network.guard_output(args.output)
    result = build(args.own_lock, args.damit_lock)
    content = json.dumps(result, indent=1)
    if args.check_published and content != args.check_published.read_text():
        raise ValueError("period-drift does not reproduce the published file exactly")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)


if __name__ == "__main__":
    main()
