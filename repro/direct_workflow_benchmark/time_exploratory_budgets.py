"""Time exploratory 18- and 96-start searches beside loaded DeLPHI.

The budgets were chosen after inspecting previously opened evaluation results.
This run is a timing sensitivity, not an independent confirmatory benchmark.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from lc_pipeline import grid_benchmark as grid

from run_direct_benchmark import classical_starts, digest, validate_lock, write_once
from run_loaded_benchmark import case


BUDGETS = (18, 96)
SEED = 20260924


def execute(lock_path: Path, output: Path) -> None:
    lock = validate_lock(lock_path)
    output.mkdir(parents=True, exist_ok=False)
    starts = classical_starts()
    write_once(output / "execution-start.json", {
        "phase": "exploratory_loaded_ladder_started",
        "lock_sha256": digest(lock_path),
        "script_sha256": digest(Path(__file__)),
        "arms": ["delphi", "classical18", "classical96"],
        "repeats": 1,
        "selection_note": "budgets chosen after evaluation scores opened; not a new holdout",
    })
    bindings = []
    setups = []
    counter = 0
    for fold in range(5):
        loading = time.perf_counter()
        predictor = grid._predictor(lock, fold)
        if predictor.device.type == "cuda":
            import torch

            torch.cuda.synchronize(predictor.device)
        setups.append({"fold": fold, "setup_seconds": time.perf_counter() - loading})
        jobs = [
            {**row, "repeat_index": 0, "arm": arm}
            for row in lock["objects"] if row["fold"] == fold
            for arm in ("delphi", "classical18", "classical96")
        ]
        random.Random(SEED + fold).shuffle(jobs)
        for job in jobs:
            arm = job["arm"]
            case_lock = lock if arm == "delphi" else {
                **lock, "classical_starts": starts[: int(arm.removeprefix("classical"))].tolist()
            }
            name = f"{job['object_id']}-r0-{arm}"
            result = case(case_lock, job, predictor, output / "cases" / name)
            timed = output / "cases" / name / "timed-result.json"
            bindings.append({"path": str(timed.relative_to(output)), "sha256": digest(timed)})
            counter += 1
            print(f"{counter}/420 fold={fold} {name} {result['wall_seconds']:.2f}s", flush=True)
        del predictor
    write_once(output / "execution.json", {
        "phase": "exploratory_loaded_ladder_complete",
        "lock_sha256": digest(lock_path),
        "script_sha256": digest(Path(__file__)),
        "rows": sorted(bindings, key=lambda row: row["path"]),
        "model_setup_seconds_by_fold": setups,
        "references_opened": False,
    })
    print(json.dumps({"completed": counter, "model_setup_seconds_by_fold": setups}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    execute(args.lock, args.output)


if __name__ == "__main__":
    main()
