"""Time the locked direct comparison with DeLPHI loaded once per fold.

This is a separate batch-use sensitivity, not a replacement for the frozen
fresh-process comparison. It never opens the DAMIT reference catalog.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from lc_pipeline import grid_benchmark as grid
from lc_pipeline import workflow_benchmark as prior
from lc_pipeline.k3.damit import _read_lc
from lc_pipeline.v2.convexinv import _parse_lightcurve_brightness
from lc_pipeline.v2.preprocessing import KnownPeriod

from run_direct_benchmark import REPEATS, SCHEMA, SEED, digest, read, validate_lock, write_once


def case(lock: dict, job: dict, predictor: object, directory: Path) -> dict:
    """Run one arm without loading models or starting a new Python process."""
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    observed = _parse_lightcurve_brightness(Path(job["lightcurve"]["path"]))
    inference_seconds = 0.0
    if job["arm"] == "delphi":
        inference_start = time.perf_counter()
        prediction = predictor.predict(
            _read_lc(Path(job["lightcurve"]["path"])),
            known_period=KnownPeriod(
                hours=job["period_hours"], provenance="fixed_published_period"
            ),
            object_id=job["object_id"],
        )
        if predictor.device.type == "cuda":
            import torch

            torch.cuda.synchronize(predictor.device)
        inference_seconds = time.perf_counter() - inference_start
        write_once(directory / "prediction.json", prediction)
        starts = prior.delphi_starts([axis["axis_xyz"] for axis in prediction["axes"]])
    else:
        starts = np.asarray(lock["classical_starts"], dtype=np.float64)
    records, solver_seconds = prior._parallel_starts(
        job, starts, observed, lock, directory / "starts"
    )
    fits = prior._successful_fits(records, "direct_loaded")
    selected = min(fits, key=prior._fit_key) if fits else None
    result = {
        "schema": SCHEMA,
        "phase": "reference_blind_loaded_case",
        "object_id": job["object_id"],
        "fold": job["fold"],
        "repeat_index": job["repeat_index"],
        "arm": job["arm"],
        "completed": selected is not None,
        "selected_fit": selected,
        "starts_requested": len(starts),
        "valid_starts": len(fits),
        "inference_seconds": inference_seconds,
        "solver_wall_seconds": solver_seconds,
        "wall_seconds": time.perf_counter() - started,
    }
    write_once(directory / "timed-result.json", result)
    return result


def execute(lock_path: Path, output: Path) -> None:
    lock = validate_lock(lock_path)
    output.mkdir(parents=True, exist_ok=False)
    write_once(
        output / "execution-start.json",
        {
            "schema": SCHEMA,
            "phase": "reference_blind_loaded_execution_started",
            "lock_sha256": digest(lock_path),
            "script_sha256": digest(Path(__file__)),
            "case_count": 140 * 2 * REPEATS,
            "timing_scope": "persistent_process_per_fold_with_one_model_load",
            "selection_rule": "same_frozen_six_delphi_and_twelve_classical_starts",
        },
    )
    bindings = []
    setups = []
    counter = 0
    for fold in range(5):
        load_start = time.perf_counter()
        predictor = grid._predictor(lock, fold)
        if predictor.device.type == "cuda":
            import torch

            torch.cuda.synchronize(predictor.device)
        setup_seconds = time.perf_counter() - load_start
        setups.append({"fold": fold, "models": 5, "setup_seconds": setup_seconds})
        jobs = [
            {**row, "repeat_index": repeat, "arm": arm}
            for row in lock["objects"]
            if row["fold"] == fold
            for repeat in range(REPEATS)
            for arm in ("classical", "delphi")
        ]
        random.Random(SEED + fold).shuffle(jobs)
        for job in jobs:
            name = f"{job['object_id']}-r{job['repeat_index']}-{job['arm']}"
            result = case(lock, job, predictor, output / "cases" / name)
            timed = output / "cases" / name / "timed-result.json"
            bindings.append({"path": str(timed.relative_to(output)), "sha256": digest(timed)})
            counter += 1
            print(
                f"{counter}/840 fold={fold} {name} {result['wall_seconds']:.2f}s",
                flush=True,
            )
        del predictor
    completion = {
        "schema": SCHEMA,
        "phase": "reference_blind_execution_complete",
        "lock_sha256": digest(lock_path),
        "rows": sorted(bindings, key=lambda row: row["path"]),
        "references_opened": False,
        "batch_timing": {
            "scope": "one_persistent_process_and_one_five-model_load_per_fold",
            "model_setup_excluded_from_case_wall_seconds": True,
            "model_setup_seconds_by_fold": setups,
            "script_sha256": digest(Path(__file__)),
        },
    }
    write_once(output / "execution.json", completion)
    print(json.dumps(completion["batch_timing"], indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    execute(args.lock, args.output)


if __name__ == "__main__":
    main()
