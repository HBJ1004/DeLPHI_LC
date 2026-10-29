"""Own periods of the 170 asteroids rounded to six decimals (packages V6 and V7).

    python repro/own_period/v67_periods.py

The inversion solver prints the period with six decimals, and the grid
benchmark rejects a fit whose printed period differs from the fixed input by
more than 1e-7 relative.  With our period held fixed the fits are very
sensitive to the period, so every compared run of V6 and V7 uses exactly the
periods written here: ``round(p, 6)`` of ``own-periods-170.json``.  For the 140
evaluation asteroids these are checked to equal the periods of the rounded
benchmark lock (``benchmark/evaluation-lock.json``), which is only read.
No reference period or pole is read.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from repro.own_period import DataPath

ROOT = DataPath("../../..")
OP = DataPath()
SOURCE = OP / "own-periods-170.json"
LOCK = OP / "benchmark/evaluation-lock.json"
OUT = OP / "v67/own-periods-170-rounded6.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    source = json.loads(SOURCE.read_text())
    periods = source["periods_hours"]
    rounded = {oid: round(float(p), 6) for oid, p in periods.items()}
    changes = {oid: abs(rounded[oid] - float(periods[oid])) for oid in periods}
    lock = json.loads(LOCK.read_text())
    mismatched = [
        row["object_id"]
        for row in lock["objects"]
        if rounded[row["object_id"]] != row["period_hours"]
    ]
    if mismatched:
        raise RuntimeError(f"rounded periods differ from the benchmark lock: {mismatched}")
    payload = {
        "schema": "delphi.own-periods.v1",
        "provenance": source["provenance"] + "-rounded6",
        "periods_hours": rounded,
        "rounding": {
            "rule": "round(p, 6) of own-periods-170.json",
            "reason": "solver prints the period with six decimals; grid benchmark fixed-period check is 1e-7 relative",
            "objects": len(rounded),
            "objects_changed": sum(v > 0 for v in changes.values()),
            "largest_change_hours": max(changes.values()),
            "largest_relative_change": max(changes[o] / float(periods[o]) for o in periods),
            "matches_benchmark_lock_objects": len(lock["objects"]),
        },
        "source": {"path": str(SOURCE), "sha256": digest(SOURCE)},
        "benchmark_lock": {"path": str(LOCK), "sha256": digest(LOCK)},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=1, sort_keys=True)
        stream.write("\n")
    print(json.dumps(payload["rounding"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
