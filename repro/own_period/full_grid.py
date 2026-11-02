"""Full classical grid of the inversion benchmark at our own periods.

    PYTHONPATH=<source-bench-8a67c52> <pinned python> repro/own_period/full_grid.py

For the 140 evaluation asteroids of ``benchmark/evaluation-lock.json`` this
fits the six standard poles and the 146 poles of the 20-degree grid with the
solver, settings and six workers of the published benchmark, at the period of
the lock.  The fits are written in the layout of the published archive
(``cases/<id>-r0-<arm>-cold/start-NNN/record.json``), so the budget ladder is
rebuilt in the same way.  It does not read reference poles.  Finished cases are
kept when the run is restarted.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from lc_pipeline import grid_benchmark as grid
from lc_pipeline import workflow_benchmark as prior
from lc_pipeline.v2.convexinv import _parse_lightcurve_brightness
from repro.own_period import DataPath

ROOT = DataPath("../../..")
BENCH = DataPath('benchmark')
OUT = BENCH / "full-grid"
ARMS = ("standard6", "classical20")


def main() -> int:
    from repro import network_root as network
    if network.switched():
        raise ValueError("reuse the fixed classical full-grid fits in a network rerun")
    lock = json.loads((BENCH / "evaluation-lock.json").read_text())
    if prior._runtime_sources(Path(lock["code_root"])) != lock["runtime_sources"]:
        raise RuntimeError("runtime source changed")
    (OUT / "cases").mkdir(parents=True, exist_ok=True)
    for number, row in enumerate(lock["objects"], start=1):
        observed = _parse_lightcurve_brightness(Path(row["lightcurve"]["path"]))
        for arm in ARMS:
            case = OUT / "cases" / f"{row['object_id']}-r0-{arm}-cold"
            done = case / "case.json"
            if done.exists():
                continue
            if case.exists():
                raise RuntimeError(f"incomplete case left by an earlier run: {case}")
            started = time.perf_counter()
            records, wall = prior._parallel_starts(
                row, grid.starts_for_arm(arm), observed, lock, case
            )
            with done.open("x", encoding="utf-8") as stream:
                json.dump(
                    {
                        "object_id": row["object_id"],
                        "arm": arm,
                        "period_hours": row["period_hours"],
                        "starts": len(records),
                        "valid": sum(r["selected_fit"] is not None for r in records),
                        "solver_wall_seconds": wall,
                        "case_seconds": time.perf_counter() - started,
                    },
                    stream,
                )
            print(number, row["object_id"], arm, len(records), f"{wall:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
