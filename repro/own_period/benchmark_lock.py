"""Locks of the inversion benchmark at our own periods.

    PYTHONPATH=<source-bench-8a67c52> <pinned python> repro/own_period/benchmark_lock.py

The published lock of ``data/direct-workflow-benchmark`` is copied with one
change, the period of every asteroid, which becomes our own period in both
searches.  The benchmark scripts are copied byte for byte next to the new lock
(they read and write next to themselves).  The code root is a worktree at the
commit whose runtime sources equal those recorded in the published lock, and
this is checked.  The environment is recorded again on this computer.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

from repro.own_period import DataPath

ROOT = DataPath("../../..")
FOLLOWUP = DataPath("../..")
PUBLISHED = FOLLOWUP / "data/direct-workflow-benchmark"
CODE_ROOT = FOLLOWUP / "source-bench-8a67c52"
PERIODS = DataPath('own-periods-170.json')
OUT = DataPath('benchmark')
SCRIPTS = (
    "run_direct_benchmark.py",
    "run_loaded_benchmark.py",
    "time_exploratory_budgets.py",
    "score_exploratory_timing.py",
    "score_classical_ladder.py",
    "budget_equivalence.py",
    "candidate_accuracy_cost.py",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


def main() -> int:
    from lc_pipeline import workflow_benchmark as prior
    from lc_pipeline.grid_benchmark import _environment

    if Path(prior.__file__).resolve().parents[1] != CODE_ROOT.resolve():
        raise RuntimeError("run with PYTHONPATH set to the benchmark code root")
    lock = json.loads((PUBLISHED / "evaluation-lock.json").read_text())
    scoring = json.loads((PUBLISHED / "scoring-lock.json").read_text())
    if prior._runtime_sources(CODE_ROOT) != lock["runtime_sources"]:
        raise RuntimeError("benchmark code root differs from the published runtime sources")
    OUT.mkdir(parents=True, exist_ok=False)
    for name in SCRIPTS:
        shutil.copyfile(PUBLISHED / name, OUT / name)
        if digest(OUT / name) != digest(PUBLISHED / name):
            raise RuntimeError(f"copy of {name} differs")
    if digest(OUT / "run_direct_benchmark.py") != lock["script"]["sha256"]:
        raise RuntimeError("published benchmark script differs from its lock")
    periods = json.loads(PERIODS.read_text())
    objects = []
    for row in lock["objects"]:
        own = periods["periods_hours"][row["object_id"]]
        if own is None:
            raise RuntimeError(f"{row['object_id']}: no own period")
        objects.append({**row, "period_hours": float(own)})
    evaluation = {
        **lock,
        "objects": objects,
        "script": {
            "path": str(OUT / "run_direct_benchmark.py"),
            "sha256": digest(OUT / "run_direct_benchmark.py"),
        },
        "code_root": str(CODE_ROOT),
        "environment": _environment(lock["device"]),
        "solver_settings": {
            **lock["solver_settings"],
            "period": "own_fourier_gpu_top1_fixed_same_in_both_arms",
        },
        "own_period": {
            "published_lock": {
                "path": str(PUBLISHED / "evaluation-lock.json"),
                "sha256": digest(PUBLISHED / "evaluation-lock.json"),
            },
            "periods": {
                "path": str(PERIODS),
                "sha256": digest(PERIODS),
                "provenance": periods["provenance"],
            },
            "change": "period_hours of every object and the period label replaced; nothing else changed",
            "python": sys.executable,
        },
    }
    write_once(OUT / "evaluation-lock.json", evaluation)
    write_once(
        OUT / "scoring-lock.json",
        {
            "schema": scoring["schema"],
            "phase": "reference_binding_for_later_scoring",
            "evaluation_lock": {
                "path": str(OUT / "evaluation-lock.json"),
                "sha256": digest(OUT / "evaluation-lock.json"),
            },
            "reference_catalog": scoring["reference_catalog"],
        },
    )
    print(OUT / "evaluation-lock.json", len(objects), "objects")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
