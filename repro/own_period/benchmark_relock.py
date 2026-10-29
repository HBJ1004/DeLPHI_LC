"""Round the periods of the own-period benchmark lock to six decimals.

    PYTHONPATH=<source-bench-8a67c52> <pinned python> repro/own_period/benchmark_relock.py

The solver prints the fitted period with six decimals and the published
benchmark rejects a fit whose printed period differs from the fixed input by
more than 1e-7 relative.  DAMIT periods pass this check, but our own periods
carry full float precision, and for six short-period asteroids every start was
rejected as ``fixed_period_changed``.  This script reads the first lock (moved
to ``failed-relative-output-20261021/lock-v1-unrounded/``), rounds every
period to six decimals (a change below 5e-7 h) and writes the lock and the
scoring lock again.  Nothing else changes.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmark_lock import OUT, digest, write_once

OLD = OUT / "failed-relative-output-20261021/lock-v1-unrounded"


def main() -> int:
    lock = json.loads((OLD / "evaluation-lock.json").read_text())
    scoring = json.loads((OLD / "scoring-lock.json").read_text())
    if scoring["evaluation_lock"]["sha256"] != digest(OLD / "evaluation-lock.json"):
        raise RuntimeError("old scoring lock does not bind the old lock")
    objects = [{**row, "period_hours": round(row["period_hours"], 6)} for row in lock["objects"]]
    changed = sum(row["period_hours"] != old["period_hours"] for row, old in zip(objects, lock["objects"]))
    largest = max(abs(row["period_hours"] - old["period_hours"]) for row, old in zip(objects, lock["objects"]))
    evaluation = {
        **lock,
        "objects": objects,
        "own_period": {
            **lock["own_period"],
            "rounding": {
                "decimals": 6,
                "reason": "solver prints the period with six decimals; the fixed-period check is 1e-7 relative",
                "previous_lock": {"path": str(OLD / "evaluation-lock.json"), "sha256": digest(OLD / "evaluation-lock.json")},
                "objects_changed": changed,
                "largest_change_hours": largest,
            },
        },
    }
    write_once(OUT / "evaluation-lock.json", evaluation)
    write_once(
        OUT / "scoring-lock.json",
        {**scoring, "evaluation_lock": {"path": str(OUT / "evaluation-lock.json"), "sha256": digest(OUT / "evaluation-lock.json")}},
    )
    print(OUT / "evaluation-lock.json", len(objects), "objects", changed, "rounded", f"largest {largest:.2e} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
