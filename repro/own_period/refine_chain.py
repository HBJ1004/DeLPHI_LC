"""Protocol-fixed, reference-free refinement of the 20 development periods.

Run ``python -m repro.own_period.refine_chain refine``. Later chain stages are
not implemented here. ``--plan-only`` writes the cost/grid plan without fitting.
Completed chunks are immutable; a restart recovers their records even if the
previous process stopped before appending them to refine.jsonl. Saved failures
are retained, and a directory without a record requires manual inspection.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
from statistics import median

from repro.own_period import period_cost as cost

from repro.own_period import DataPath

DATA = DataPath()
OUTPUT = DATA / "refine-chain"
EVALUATION_LOCK = DATA / "benchmark/evaluation-lock.json"
COST_SUMMARY = cost.OUTPUT / "period-cost-summary.json"
PROTOCOL_OBJECTS = tuple(f"asteroid_{number}" for number in (
    103, 111, 118, 120, 125, 153, 156, 166, 101, 104,
    115, 117, 105, 106, 121, 122, 108, 114, 143, 150,
))
CHUNK_TRIALS = 150
TIMEOUT_SECONDS = 12 * 3600


def protocol_objects(lock: dict) -> list[dict]:
    rows = lock["objects"]
    if (len({row["object_id"] for row in rows}) != len(rows)
            or {row["fold"] for row in rows} != set(range(5))):
        raise ValueError("evaluation lock must contain unique objects in five folds")
    selected = [row for fold in range(5)
                for row in [r for r in rows if r["fold"] == fold][:4]]
    if tuple(row["object_id"] for row in selected) != PROTOCOL_OBJECTS:
        raise ValueError("first four objects per fold disagree with the protocol list")
    return selected


def build_refine_plan(args) -> dict:
    """Read only the own-period lock, own periods, lightcurves, and CPU costs."""
    if args.chunk_trials < 1:
        raise ValueError("chunk trial target must be positive")
    selected = protocol_objects(json.loads(args.evaluation_lock.read_text()))
    periods = json.loads(args.periods_file.read_text())["periods_hours"]
    estimates = json.loads(args.cost_summary.read_text())["estimates"]
    means = {}
    for oid, estimate in estimates.items():
        value = estimate["E_core"]["cpu_seconds"].get("mean_per_trial")
        # A timed-out sample's mean_per_trial_at_least is not a measured mean.
        if value is not None:
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid E-core CPU mean for {oid}")
            means[oid] = value
    if not means:
        raise ValueError("at least one measured E-core CPU mean is required")
    fallback = median(means.values())
    executable_sha256 = cost.sha256_file(args.solver_source / "period_scan")
    objects, tasks = [], []
    for row in selected:
        oid = row["object_id"]
        period = float(periods[oid])
        if not math.isfinite(period) or period <= 0:
            raise ValueError(f"invalid own period for {oid}")
        lightcurve = Path(row["lightcurve"]["path"]).resolve()
        lightcurve_sha256 = cost.sha256_file(lightcurve)
        if lightcurve_sha256 != row["lightcurve"]["sha256"]:
            raise ValueError(f"lightcurve disagrees with evaluation lock: {oid}")
        span = cost.lightcurve_span(lightcurve)
        f_own = 1 / period
        grid = cost.TrialGrid(span["jd_min"], span["jd_max"], f_own * 1.01, f_own * 0.99)
        cpu_mean = means.get(oid, fallback)
        chunks = math.ceil(grid.count / args.chunk_trials)
        info = {"object_id": oid, "fold": row["fold"], "own_period_hours": period,
                "lightcurve_file": str(lightcurve), "lightcurve_sha256": lightcurve_sha256,
                **span, "start_frequency_per_hour": grid.start_frequency_per_hour,
                "end_frequency_per_hour": grid.end_frequency_per_hour,
                "frequency_step_per_hour": grid.step, "trial_count": grid.count,
                "chunks": chunks, "expected_cpu_seconds_per_trial": cpu_mean,
                "cost_source": "object_mean" if oid in means else "median_of_measured_means",
                "expected_cpu_seconds": cpu_mean * grid.count,
                "expected_cpu_hours": cpu_mean * grid.count / 3600}
        objects.append(info)
        for block, (start, count) in enumerate(cost.chunk_ranges(grid.count, chunks)):
            task = cost._task(info, start, count,
                              args.output_dir / "refine-runs" / oid / f"chunk-{block:03d}",
                              args.solver_source, block, TIMEOUT_SECONDS, "refine")
            tasks.append({**task, "executable_sha256": executable_sha256,
                          "expected_cpu_seconds": cpu_mean * count})
    # Submit all objects together; stable ordering resolves equal-cost ties.
    tasks.sort(key=lambda task: task["expected_cpu_seconds"], reverse=True)
    settings = asdict(cost.PeriodScanParameters(2, 100))
    for key in ("period_start_hours", "period_end_hours"):
        del settings[key]
    return {"schema": "delphi.refine-chain.plan.v1",
            "evaluation_lock": str(args.evaluation_lock.resolve()),
            "evaluation_lock_sha256": cost.sha256_file(args.evaluation_lock),
            "periods_file": str(args.periods_file.resolve()),
            "periods_file_sha256": cost.sha256_file(args.periods_file),
            "cost_summary": str(args.cost_summary.resolve()),
            "cost_summary_sha256": cost.sha256_file(args.cost_summary),
            "cpus": list(args.cpus), "workers": args.workers, "nice": 19,
            "chunk_trial_target": args.chunk_trials, "timeout_seconds": TIMEOUT_SECONDS,
            "solver_settings": settings, "median_cpu_seconds_per_trial": fallback,
            "objects": objects, "tasks": tasks,
            "total_trials": sum(row["trial_count"] for row in objects),
            "expected_cpu_hours": sum(row["expected_cpu_hours"] for row in objects)}


def score_periods(rows) -> dict:
    """Reference-free counterpart of period_cost.score_periods, in grid order."""
    if not rows:
        raise ValueError("nonempty fitted table required")
    if any(not math.isfinite(r.period_hours) or r.period_hours <= 0
           or not math.isfinite(r.chi2) or r.chi2 < 0 for r in rows):
        raise ValueError("invalid fitted period or chi2")
    best = min(range(len(rows)), key=lambda index: rows[index].chi2)
    period, chi2 = rows[best].period_hours, rows[best].chi2
    minima, index = [], 0
    while index < len(rows):
        end = index
        while end + 1 < len(rows) and rows[end + 1].chi2 == rows[index].chi2:
            end += 1
        left = rows[index - 1].chi2 if index else math.inf
        right = rows[end + 1].chi2 if end + 1 < len(rows) else math.inf
        if rows[index].chi2 < left and rows[index].chi2 < right:
            minima.append(index)
        index = end + 1
    alternatives = [i for i in minima if i != best
                    and abs(rows[i].period_hours / period - 1) > 0.01 + 1e-12
                    and rows[i].chi2 <= chi2 * 1.1 + 1e-12]
    return {"period_found_hours_raw": period, "period_found_hours": round(period, 6),
            "chi2": chi2, "best_trial_index": best, "ambiguity_count": len(alternatives),
            "alternative_minima": [{"trial_index": i, **asdict(rows[i])} for i in alternatives],
            "minima_definition": "Adjacent start-grid rows; plateaus counted once; endpoints included."}


def _validate_record(task: dict, record: dict):
    if any(record.get(key) != value for key, value in task.items()):
        raise ValueError(f"existing record mismatches planned task: {task['raw_directory']}")
    if record.get("status") not in ("complete", "not_finished"):
        raise ValueError("invalid saved chunk status")
    complete = record["status"] == "complete"
    for key in ("cpu_seconds", "wall_seconds"):
        value = record.get(key)
        if ((value is None and complete)
                or (value is not None and (not math.isfinite(value) or value < 0))):
            raise ValueError("invalid saved chunk timing")
    if not complete:
        return
    if (record.get("return_code") != 0 or record.get("timed_out") is not False
            or record.get("rows_returned") != task["expected_rows"]):
        raise ValueError("saved complete record has inconsistent completion fields")
    directory = Path(task["raw_directory"])
    raw = directory / "period-scan.txt"
    if cost.sha256_file(raw) != record.get("output_periods_sha256"):
        raise ValueError("saved chunk output changed")
    parameters = directory / "input-period-scan"
    provenance = record["provenance"]
    if (cost.sha256_file(parameters) != provenance["parameter_sha256"]
            or provenance["input_sha256"] != task["lightcurve_sha256"]
            or provenance["executable_sha256"] != task["executable_sha256"]):
        raise ValueError("saved chunk provenance changed")
    rows = cost.parse_period_scan_rows(raw)
    grid = cost.TrialGrid(task["jd_min"], task["jd_max"],
                          task["start_frequency_per_hour"], task["end_frequency_per_hour"])
    if (len(rows) != task["expected_rows"] or record.get("fitted_rows") != cost.fitted_rows(
            rows, grid, task["start_index"], task["period_bounds_hours"][0])):
        raise ValueError("saved fitted rows disagree with the chunk output")


def _resume_records(tasks) -> tuple[list[dict], list[dict]]:
    saved, pending = [], []
    for task in tasks:
        directory = Path(task["raw_directory"])
        if not directory.exists():
            pending.append(task)
            continue
        path = directory / "record.json"
        if not path.is_file():
            raise FileExistsError(f"incomplete chunk directory without record; inspect manually: {directory}")
        record = json.loads(path.read_text())
        _validate_record(task, record)
        saved.append(record)
    return saved, pending


def _existing_log(path: Path, saved) -> set[str]:
    by_directory = {record["raw_directory"]: cost._compact(record) for record in saved}
    seen = set()
    if path.exists():
        for row in cost._read_jsonl(path):
            directory = row.get("raw_directory")
            if directory in seen or row != by_directory.get(directory):
                raise ValueError("refine.jsonl contains a duplicate or mismatching chunk record")
            seen.add(directory)
    return seen


def summarize(plan: dict, records) -> dict:
    objects, periods = [], {}
    for info in plan["objects"]:
        chunks = sorted((r for r in records if r["object_id"] == info["object_id"]),
                        key=lambda r: r["start_index"])
        complete = (len(chunks) == info["chunks"]
                    and all(r["status"] == "complete" for r in chunks)
                    and sum(r["rows_returned"] for r in chunks) == info["trial_count"])
        timestamps = all(r.get("started_utc") and r.get("finished_utc") for r in chunks)
        wall = ((max(datetime.fromisoformat(r["finished_utc"]) for r in chunks)
                 - min(datetime.fromisoformat(r["started_utc"]) for r in chunks)).total_seconds()
                if chunks and timestamps else None)
        result = {"object_id": info["object_id"], "fold": info["fold"],
                  "status": "complete" if complete else "not_finished",
                  "expected_rows": info["trial_count"],
                  "rows_returned": sum(r["rows_returned"] for r in chunks),
                  "cpu_seconds": sum(r["cpu_seconds"] or 0 for r in chunks),
                  "wall_seconds": wall,
                  "sum_solver_wall_seconds": sum(r["wall_seconds"] or 0 for r in chunks),
                  "period_found_hours_raw": None, "period_found_hours": None,
                  "chi2": None, "best_trial_index": None, "ambiguity_count": None,
                  "alternative_minima": []}
        if complete:
            rows = [row for chunk in chunks for row in cost.parse_period_scan_rows(
                Path(chunk["raw_directory"]) / "period-scan.txt")]
            result.update(score_periods(rows))
            periods[info["object_id"]] = result["period_found_hours"]
        objects.append(result)
    return {"schema": "delphi.refine-chain.periods.v1", "objects": objects,
            "periods_hours": periods,
            "wall_seconds_definition": "First chunk start to last chunk finish per object, including interruptions.",
            "failure_rule": "Incomplete scans have no selected period and count as failures in both arms."}


def _check_existing_json(path: Path, expected: dict) -> None:
    value = json.loads(path.read_text())
    value.pop("created_utc", None)
    if value != expected:
        raise ValueError(f"existing output mismatches current refinement: {path}")


def command_refine(args) -> int:
    from repro import network_root as network
    if network.switched():
        raise ValueError("reuse the fixed narrow period scans in a network rerun")
    cost._check_solver(args.solver_source)
    cost._check_cpus(args.cpus, args.workers)
    plan = build_refine_plan(args)
    plan_path = args.output_dir / "refine-plan.json"
    log_path = args.output_dir / "refine.jsonl"
    final_path = args.output_dir / "refined-periods.json"
    if plan_path.exists():
        _check_existing_json(plan_path, plan)
    elif any(path.exists() for path in (log_path, final_path, args.output_dir / "refine-runs")):
        raise FileExistsError("refinement artifacts exist without refine-plan.json")
    saved, pending = _resume_records(plan["tasks"])
    seen = _existing_log(log_path, saved)
    if final_path.exists():
        if pending:
            raise ValueError("refined-periods.json exists with missing chunks")
        _check_existing_json(final_path, summarize(plan, saved))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not plan_path.exists():
        cost._write_json(plan_path, {**plan, "created_utc": cost._utc()})
    print(f"Planned {plan['total_trials']} trials in {len(plan['tasks'])} chunks; "
          f"{plan['expected_cpu_hours']:.3f} E-core CPU hours", flush=True)
    if args.plan_only:
        return 0
    # Append only missing records. A crash before logging is recovered from the
    # worker's exclusive record.json; no raw chunk or prior log row is rewritten.
    with log_path.open("a" if log_path.exists() else "x", encoding="utf-8") as stream:
        for record in saved:
            if record["raw_directory"] not in seen:
                cost._dump(stream, cost._compact(record))
        new_records = cost.run_tasks(pending, args.cpus, args.workers) if pending else []
        for record in new_records:
            cost._dump(stream, cost._compact(record))
    records = saved + new_records
    # Recheck saved artifacts and the newly written records before selecting a
    # period. Worker failures still appear in the final per-object failure list.
    by_directory = {record["raw_directory"]: record for record in records}
    for task in plan["tasks"]:
        _validate_record(task, by_directory[task["raw_directory"]])
    summary = summarize(plan, records)
    if not final_path.exists():
        cost._write_json(final_path, {**summary, "created_utc": cost._utc()})
    failures = sum(row["status"] != "complete" for row in summary["objects"])
    print(f"Refinement: {len(summary['objects']) - failures}/20 complete; {failures} not finished")
    return int(failures > 0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    refine = subparsers.add_parser("refine")
    refine.add_argument("--output-dir", type=Path, default=OUTPUT)
    refine.add_argument("--evaluation-lock", type=Path, default=EVALUATION_LOCK)
    refine.add_argument("--periods-file", type=Path, default=cost.COHORT)
    refine.add_argument("--cost-summary", type=Path, default=COST_SUMMARY)
    refine.add_argument("--solver-source", type=Path, default=cost.SOURCE)
    refine.add_argument("--cpus", type=cost.parse_cpus, default=cost.parse_cpus("12-23"))
    refine.add_argument("--workers", type=int, default=12)
    refine.add_argument("--chunk-trials", type=int, default=CHUNK_TRIALS)
    refine.add_argument("--plan-only", action="store_true")
    refine.set_defaults(function=command_refine)
    args = parser.parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
