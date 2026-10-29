"""Protocol-fixed timing of DAMIT's classical 2--100 h period search.

Run with ``python -m repro.own_period.period_cost {count,sample,full,score}``.
Only ``score`` accesses reference solutions. Trial indices are zero based;
returned periods are fitted values, not the starting frequency grid.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import resource
import tempfile
import time

import numpy as np

from lc_pipeline.v2.convexinv import (
    PeriodScanParameters,
    PeriodScanRow,
    parse_period_scan_rows,
    run_period_scan,
    write_period_scan_parameters,
)
from lc_pipeline.v2.serialization import sha256_file

REPO = Path(__file__).resolve().parents[2]
from repro.own_period import DataPath

BASE = DataPath("../..")
ROOT = BASE.parent
SOURCE = BASE / "data/solver-internal-capacity-corrected/20260911/expanded-solver/convexinv"
CATALOG = BASE / "inputs/training-run/repro/data/damit-20250610T000301Z/catalog.jsonl"
COHORT = DataPath('own-periods-170.json')
LIGHTCURVES = ROOT / "damit-20250610T000301Z"
OUTPUT = DataPath('period-cost')
SAMPLE_OBJECTS = tuple(f"asteroid_{n}" for n in (
    101, 103, 104, 105, 106, 107, 108, 109, 111, 114,
    115, 116, 117, 118, 120, 121, 123, 125, 134, 153,
)) + ("asteroid_2516", "asteroid_1733")
FULL_OBJECTS = ("asteroid_2516", "asteroid_1733")
CALIBRATION_OBJECTS = tuple(f"asteroid_{n}" for n in (101, 111, 120, 153))
RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261026
FOURIER_SECONDS = 7.13
HARDWARE_NOTE = "Classical: one CPU core; Fourier: warm GPU search, 7.13 s per asteroid (existing warm170-first.jsonl record)."
_CPU_QUEUE = None


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dump(stream, value) -> None:
    stream.write(json.dumps(value, allow_nan=False, sort_keys=True) + "\n")
    stream.flush()


def _write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        _dump(stream, value)


@dataclass(frozen=True)
class TrialGrid:
    jd_min: float
    jd_max: float
    start_frequency_per_hour: float = 1 / 2
    end_frequency_per_hour: float = 1 / 100

    def __post_init__(self):
        if not (math.isfinite(self.jd_min) and math.isfinite(self.jd_max)
                and self.jd_max > self.jd_min):
            raise ValueError("observation span must be finite and positive")
        if not (math.isfinite(self.start_frequency_per_hour)
                and math.isfinite(self.end_frequency_per_hour)
                and self.start_frequency_per_hour >= self.end_frequency_per_hour > 0):
            raise ValueError("grid frequencies must be finite, positive, and descending")

    @property
    def step(self) -> float:
        # Preserve the C expression's evaluation order.
        return 0.5 / (self.jd_max - self.jd_min) / 24 * 0.8

    @property
    def count(self) -> int:
        return int((self.start_frequency_per_hour - self.end_frequency_per_hour) / self.step) + 1

    def frequency(self, index: int) -> float:
        if not 0 <= index < self.count:
            raise ValueError("trial index outside the global grid")
        return self.start_frequency_per_hour - index * self.step

    def bounds(self, start: int, count: int) -> tuple[float, float]:
        if count < 1 or start < 0 or start + count > self.count:
            raise ValueError("block outside the global grid")
        first = self.frequency(start)
        # Half a step of slack avoids integer truncation losing an endpoint.
        # The final bound may exceed the window by half a step; the last
        # starting trial is still exactly the last member of the global grid.
        end_frequency = first - (count - 0.5) * self.step
        if end_frequency <= 0:
            end_frequency = first / 2  # Degenerate one-trial, very short spans.
        return 1 / first, 1 / end_frequency


def integer_trial_count(start_hours: float, end_hours: float, step: float) -> int:
    """The C loop bound, without ceil or an approximate endpoint comparison."""
    return int((1 / start_hours - 1 / end_hours) / step) + 1


def lightcurve_span(path: Path) -> dict:
    """Read all JDs in the same eight-value observations as period_scan."""
    words = path.read_text(encoding="ascii").split()
    try:
        curves, cursor, observations = int(words[0]), 1, 0
        jd_min, jd_max = math.inf, -math.inf
        if curves < 1:
            raise ValueError("no lightcurves")
        for _ in range(curves):
            count, calibrated = int(words[cursor]), int(words[cursor + 1])
            cursor += 2
            if count < 1 or calibrated not in (0, 1):
                raise ValueError("invalid lightcurve header")
            for _ in range(count):
                values = list(map(float, words[cursor:cursor + 8]))
                if len(values) != 8 or not all(map(math.isfinite, values)):
                    raise ValueError("invalid observation")
                jd_min, jd_max = min(jd_min, values[0]), max(jd_max, values[0])
                cursor += 8
            observations += count
        if cursor != len(words):
            raise ValueError("extra lightcurve data")
        TrialGrid(jd_min, jd_max)
    except (IndexError, ValueError) as exc:
        raise ValueError(f"malformed lightcurve {path}: {exc}") from exc
    return {"jd_min": jd_min, "jd_max": jd_max,
            "baseline_days": jd_max - jd_min,
            "lightcurves": curves, "observations": observations}


def sample_starts(object_id: str, trial_count: int) -> tuple[int, ...]:
    if trial_count < 10:
        raise ValueError("sample requires at least ten trials")
    seed = int.from_bytes(hashlib.sha256(
        ("period-cost-20261026:" + object_id).encode("ascii")
    ).digest(), "big")
    rng = random.Random(seed)
    # Ten independent uniform draws, with replacement, over valid block starts.
    return tuple(rng.randrange(trial_count - 9) for _ in range(10))


def chunk_ranges(trial_count: int, chunks: int) -> tuple[tuple[int, int], ...]:
    if not 1 <= chunks <= trial_count:
        raise ValueError("chunks must be between one and the trial count")
    q, r = divmod(trial_count, chunks)
    start, result = 0, []
    for index in range(chunks):
        count = q + (index < r)
        result.append((start, count))
        start += count
    return tuple(result)


def _write_parameters(path: Path, grid: TrialGrid, start: int, count: int):
    bounds = grid.bounds(start, count)
    params = PeriodScanParameters(*bounds)
    # Reuse the adapter's defaults and writer, staging outside the project.
    # Its usual 12 significant digits needlessly perturb chunk starting grids;
    # serialize only the bounds at round-trip precision in our exclusive output.
    with tempfile.TemporaryDirectory(prefix="period-cost-params-") as temporary:
        staged = write_period_scan_parameters(Path(temporary) / "parameters", params)
        _, rest = staged.read_text(encoding="ascii").split("\n", 1)
    text = f"{bounds[0]:.17g} {bounds[1]:.17g} {params.step_coefficient:.17g}\n" + rest
    with path.open("x", encoding="ascii") as stream:
        stream.write(text)
    fs, fe, coefficient = map(float, text.splitlines()[0].split())
    if coefficient != 0.8 or integer_trial_count(fs, fe, grid.step) != count:
        raise ValueError("serialized bounds do not produce the requested trial count")
    return fs, fe


def command_count(args) -> int:
    ids = sorted(json.loads(args.cohort.read_text())["periods_hours"])
    if len(ids) != 170:
        raise ValueError("expected the fixed 170-object cohort")
    # Project catalog metadata immediately; reference solutions are never used.
    paths = {}
    with args.catalog.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["object_id"] in ids:
                paths[row["object_id"]] = row["lightcurve"]["source_path"]
    rows = []
    for oid in ids:
        path = (args.lightcurve_root / paths[oid]).resolve()
        span = lightcurve_span(path)
        grid = TrialGrid(span["jd_min"], span["jd_max"])
        rows.append({"object_id": oid, "lightcurve_file": str(path),
                     "lightcurve_sha256": sha256_file(path), **span,
                     "frequency_step_per_hour": grid.step, "trial_count": grid.count})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "trial-counts.json", {
        "schema": "delphi.period-cost.counts.v1", "created_utc": _utc(),
        "cohort": str(args.cohort.resolve()), "catalog": str(args.catalog.resolve()),
        "parameters": asdict(PeriodScanParameters(2, 100)), "objects": rows,
    })
    print(f"Counted {len(rows)} objects; {sum(r['trial_count'] for r in rows)} trials")
    return 0


def _counts(output: Path) -> dict[str, dict]:
    value = json.loads((output / "trial-counts.json").read_text())
    rows = value["objects"]
    result = {row["object_id"]: row for row in rows}
    if len(result) != 170 or len(rows) != 170:
        raise ValueError("counts must contain 170 unique objects")
    for row in rows:
        grid = TrialGrid(row["jd_min"], row["jd_max"])
        if grid.count != row["trial_count"] or grid.step != row["frequency_step_per_hour"]:
            raise ValueError("recorded grid disagrees with the protocol")
    return result


def parse_cpus(value: str) -> tuple[int, ...]:
    result = []
    try:
        for part in value.split(","):
            limits = part.split("-")
            if len(limits) == 1:
                result.append(int(part))
            elif len(limits) == 2 and int(limits[1]) >= int(limits[0]):
                result.extend(range(int(limits[0]), int(limits[1]) + 1))
            else:
                raise ValueError
        if not result or min(result) < 0 or len(set(result)) != len(result):
            raise ValueError
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use unique CPU numbers/ranges, e.g. 12-23") from exc
    return tuple(result)


def _check_cpus(cpus, workers):
    if not 1 <= workers <= len(cpus):
        raise ValueError("workers must be between one and the number of CPUs")
    if not set(cpus) <= os.sched_getaffinity(0):
        raise ValueError("requested CPUs are outside the current affinity")


def _task(row: dict, start: int, count: int, directory: Path,
          source: Path, block: int, timeout: float, kind: str) -> dict:
    frequencies = {key: row[key] for key in
                   ("start_frequency_per_hour", "end_frequency_per_hour") if key in row}
    grid = TrialGrid(row["jd_min"], row["jd_max"], **frequencies)
    return {"object_id": row["object_id"], "kind": kind, "block": block,
            **frequencies,
            "start_index": start, "stop_index_exclusive": start + count,
            "expected_rows": count, "jd_min": grid.jd_min, "jd_max": grid.jd_max,
            "frequency_step_per_hour": grid.step,
            "period_bounds_hours": list(grid.bounds(start, count)),
            "lightcurve_file": row["lightcurve_file"],
            "lightcurve_sha256": row["lightcurve_sha256"],
            "source": str(source.resolve()), "raw_directory": str(directory.resolve()),
            "timeout_seconds": timeout}


def _init_worker(queue):
    global _CPU_QUEUE
    _CPU_QUEUE = queue


def fitted_rows(rows, grid: TrialGrid, start: int, period_start_hours=None) -> list[dict]:
    result = []
    first = grid.frequency(start) if period_start_hours is None else 1 / period_start_hours
    for local_index, row in enumerate(rows):
        if row.period_hours <= 0 or row.chi2 < 0:
            raise ValueError("invalid fitted period or chi2")
        frequency = first - local_index * grid.step
        result.append({**asdict(row), "trial_index": start + local_index,
                       "start_frequency_per_hour": frequency,
                       "start_period_hours": 1 / frequency,
                       "frequency_offset_steps": (1 / row.period_hours - frequency) / grid.step})
    return result


def _run_task(task: dict) -> dict:
    cpu = _CPU_QUEUE.get()
    record = {**task, "started_utc": _utc(), "worker_pid": os.getpid(),
              "return_code": None, "timed_out": False, "rows_returned": 0,
              "wall_seconds": None, "cpu_seconds": None, "fitted_rows": [],
              "status": "not_finished", "error": None}
    directory = Path(task["raw_directory"])
    directory_created = False
    started = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        os.sched_setaffinity(0, {cpu})
        os.nice(19)
        record.update(cpu_list=sorted(os.sched_getaffinity(0)),
                      nice=os.getpriority(os.PRIO_PROCESS, 0),
                      load_average=list(os.getloadavg()))
        directory.mkdir(parents=True, exist_ok=False)
        directory_created = True
        if sha256_file(task["lightcurve_file"]) != task["lightcurve_sha256"]:
            raise ValueError("lightcurve changed since count")
        grid = TrialGrid(task["jd_min"], task["jd_max"], **{
            key: task[key] for key in ("start_frequency_per_hour", "end_frequency_per_hour")
            if key in task})
        parameter_file = directory / "input-period-scan"
        record["period_bounds_hours"] = list(_write_parameters(
            parameter_file, grid, task["start_index"], task["expected_rows"]))
        result = run_period_scan(
            executable=task.get("executable", Path(task["source"]) / "period_scan"),
            source_root=task["source"],
            lightcurve_file=task["lightcurve_file"], parameter_file=parameter_file,
            output_directory=directory, timeout_seconds=task["timeout_seconds"],
        )
        record.update(return_code=result.return_code, timed_out=result.timed_out,
                      wall_seconds=result.wall_time_seconds,
                      cpu_seconds=result.process_cpu_time_seconds,
                      provenance=asdict(result.provenance),
                      stdout_sha256=result.stdout_sha256, stderr_sha256=result.stderr_sha256,
                      output_periods_sha256=result.output_periods_sha256)
        rows = result.rows
        raw = directory / "period-scan.txt"
        # Preserve partial results on a timeout/nonzero exit, without calling it complete.
        if not rows and raw.exists() and raw.stat().st_size:
            rows = parse_period_scan_rows(raw)
        record["rows_returned"] = len(rows)
        record["fitted_rows"] = fitted_rows(rows, grid, task["start_index"], record["period_bounds_hours"][0])
        if (result.return_code == 0 and not result.timed_out
                and len(rows) == task["expected_rows"]
                and result.process_cpu_time_seconds is not None
                and math.isfinite(result.process_cpu_time_seconds)):
            record["status"] = "complete"
        else:
            record["error"] = "timeout, nonzero exit, unavailable CPU timing, or incorrect row count"
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        if record["wall_seconds"] is None:
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            record.update(wall_seconds=time.monotonic() - started,
                          cpu_seconds=after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime)
    finally:
        record["finished_utc"] = _utc()
        record["load_average_end"] = list(os.getloadavg())
        _CPU_QUEUE.put(cpu)
    # Never overwrite an old record, including when the run directory already existed.
    if directory_created:
        _write_json(directory / "record.json", record)
    return record


def _compact(record):
    return {key: value for key, value in record.items() if key != "fitted_rows"}


def run_tasks(tasks, cpus, workers, stream=None) -> list[dict]:
    """Each call gets a fresh process and an exclusive lease on one CPU."""
    _check_cpus(cpus, workers)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    for cpu in cpus:
        queue.put(cpu)
    records = []
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                 max_tasks_per_child=1, initializer=_init_worker,
                                 initargs=(queue,)) as executor:
            futures = {executor.submit(_run_task, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    record = future.result()
                except Exception as exc:
                    record = {**task, "status": "not_finished", "return_code": None,
                              "timed_out": False, "rows_returned": 0,
                              "wall_seconds": None, "cpu_seconds": None,
                              "error": f"worker failed: {exc}"}
                records.append(record)
                if stream is not None:
                    _dump(stream, record if task["kind"] != "full" else _compact(record))
    finally:
        queue.close()
        queue.join_thread()
    return records


def _sample_tasks(counts, output, source, calibration=False):
    tasks = []
    objects = CALIBRATION_OBJECTS if calibration else SAMPLE_OBJECTS
    root = "calibration-runs" if calibration else "sample-runs"
    for oid in objects:
        row = counts[oid]
        for block, start in enumerate(sample_starts(oid, row["trial_count"])):
            tasks.append(_task(row, start, 10, output / root / oid / f"block-{block:02d}",
                               source, block, 600, "calibration" if calibration else "sample"))
        if not calibration:
            tasks.append(_task(row, 0, 1, output / root / oid / "startup", source, -1, 600, "startup"))
    return tasks


def _check_solver(source):
    executable = source / "period_scan"
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError(f"period_scan must exist and have execute permission before timing: {executable}")


def command_sample(args) -> int:
    _check_solver(args.solver_source)
    _check_cpus(args.cpus, args.workers)
    if args.calibrate_cpu is not None:
        _check_cpus((args.calibrate_cpu,), 1)
    counts = _counts(args.output_dir)
    tasks = _sample_tasks(counts, args.output_dir, args.solver_source)
    calibration = _sample_tasks(counts, args.output_dir, args.solver_source, True)
    destinations = [args.output_dir / "sample.jsonl"]
    if args.calibrate_cpu is not None:
        destinations.append(args.output_dir / "calibration.jsonl")
    if any(path.exists() for path in destinations):
        raise FileExistsError("sample/calibration output exists; protocol forbids rerunning")
    with destinations[0].open("x", encoding="utf-8") as stream:
        records = run_tasks(tasks, args.cpus, args.workers, stream)
    if args.calibrate_cpu is not None:
        with destinations[1].open("x", encoding="utf-8") as stream:
            records += run_tasks(calibration, (args.calibrate_cpu,), 1, stream)
    failures = sum(row["status"] != "complete" for row in records)
    print(f"Recorded {len(records)} calls; {failures} not finished")
    return int(failures > 0)


def command_full(args) -> int:
    _check_solver(args.solver_source)
    _check_cpus(args.cpus, min(args.chunks, len(args.cpus)))
    if not args.objects or len(set(args.objects)) != len(args.objects):
        raise ValueError("select unique full-scan objects")
    counts = _counts(args.output_dir)
    failures = 0
    with (args.output_dir / "full.jsonl").open("x", encoding="utf-8") as stream:
        for oid in args.objects:
            row = counts[oid]
            tasks = [_task(row, start, count,
                           args.output_dir / "full-runs" / oid / f"chunk-{block:02d}",
                           args.solver_source, block, 8 * 3600, "full")
                     for block, (start, count) in enumerate(chunk_ranges(row["trial_count"], args.chunks))]
            start_utc, started = _utc(), time.monotonic()
            records = run_tasks(tasks, args.cpus, min(args.chunks, len(args.cpus)))
            records.sort(key=lambda record: record["start_index"])
            complete = all(r["status"] == "complete" for r in records)
            returned = sum(r["rows_returned"] for r in records)
            complete = complete and returned == row["trial_count"]
            summary = {"object_id": oid, "started_utc": start_utc, "finished_utc": _utc(),
                       "wall_seconds": time.monotonic() - started,
                       "cpu_seconds": sum(r["cpu_seconds"] or 0 for r in records),
                       "sum_solver_wall_seconds": sum(r["wall_seconds"] or 0 for r in records),
                       "expected_rows": row["trial_count"], "rows_returned": returned,
                       "status": "complete" if complete else "not_finished",
                       "chunks": [_compact(r) for r in records]}
            _dump(stream, summary)
            failures += not complete
            print(f"{oid}: {returned}/{row['trial_count']} rows, {summary['status']}", flush=True)
    return int(failures > 0)


def estimate_cost(blocks, trial_count: int) -> dict:
    if len(blocks) != 10 or any(r["status"] != "complete" or r["rows_returned"] != 10 for r in blocks):
        raise ValueError("an estimate requires all ten complete ten-trial blocks")
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 10, (RESAMPLES, 10))
    result = {}
    for key in ("wall_seconds", "cpu_seconds"):
        values = np.asarray([r[key] / 10 for r in blocks], dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("invalid timing measurement")
        low, high = np.percentile(values[indices].mean(axis=1) * trial_count, [2.5, 97.5])
        result[key] = {"mean_per_trial": float(values.mean()),
                       "estimated_full": float(values.mean() * trial_count),
                       "ci95": [float(low), float(high)]}
    return result


def lower_bound_cost(blocks, trial_count: int) -> dict:
    """Timed-out blocks did fewer than ten trials, so seconds / 10 is a lower bound per trial."""
    if len(blocks) != 10 or any(not (r["status"] == "complete" and r["rows_returned"] == 10)
                                and not r["timed_out"] for r in blocks):
        raise ValueError("a lower bound requires ten complete or timed-out blocks")
    result = {}
    for key in ("wall_seconds", "cpu_seconds"):
        values = np.asarray([r[key] / 10 for r in blocks], dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("invalid timing measurement")
        result[key] = {"mean_per_trial_at_least": float(values.mean()),
                       "estimated_full_at_least": float(values.mean() * trial_count)}
    result["timed_out_blocks"] = sum(bool(r["timed_out"]) for r in blocks)
    return result


def score_periods(rows, reference_period: float) -> dict:
    """Local minima use start-grid order; a flat minimum counts once."""
    if not rows or not math.isfinite(reference_period) or reference_period <= 0:
        raise ValueError("nonempty fitted table and positive reference required")
    if any(not math.isfinite(r.period_hours) or r.period_hours <= 0
           or not math.isfinite(r.chi2) or r.chi2 < 0 for r in rows):
        raise ValueError("invalid fitted period or chi2")
    best = min(range(len(rows)), key=lambda index: rows[index].chi2)
    period, chi2 = rows[best].period_hours, rows[best].chi2
    classification = "other"
    for label, factor in (("P", 1), ("P/2", 0.5), ("2P", 2)):
        if abs(period / (reference_period * factor) - 1) <= 0.01 + 1e-12:
            classification = label
            break
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
    return {"period_found_hours": period, "chi2": chi2, "best_trial_index": best,
            "reference_period_hours": reference_period, "period_class": classification,
            "within_one_percent_of_P": classification == "P",
            "ambiguity_count": len(alternatives),
            "alternative_minima": [{"trial_index": i, **asdict(rows[i])} for i in alternatives],
            "minima_definition": "Adjacent start-grid rows; plateaus counted once; endpoints included."}


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _validated_samples(records, counts, calibration=False, allow_timeouts=False):
    expected = _sample_tasks(counts, Path("unused"), SOURCE, calibration)
    by_key = {(r["object_id"], r["block"]): r for r in records}
    keys = {(r["object_id"], r["block"]) for r in expected}
    if len(by_key) != len(records) or set(by_key) != keys:
        raise ValueError("missing, duplicate, or unexpected sampled calls; no estimates written")
    for task in expected:
        record = by_key[(task["object_id"], task["block"])]
        finished = (record["status"] == "complete" and record["return_code"] == 0
                    and not record["timed_out"] and record["rows_returned"] == task["expected_rows"])
        # Post-run reporting rule (PROTOCOL amendment 2): a timed-out call is kept as a lower bound.
        timed_out = allow_timeouts and record["timed_out"] is True
        if (not (finished or timed_out)
                or any(record[key] != task[key] for key in
                       ("start_index", "stop_index_exclusive", "expected_rows"))):
            raise ValueError(f"not finished or wrong sampled grid: {task['object_id']} block {task['block']}")
    return by_key


def _validated_full(records, counts):
    by_id = {r["object_id"]: r for r in records}
    if len(by_id) != len(records) or set(by_id) != set(FULL_OBJECTS):
        raise ValueError("both complete scans must be written before reference scoring")
    tables = {}
    for oid, record in by_id.items():
        if record["status"] != "complete":
            raise ValueError(f"{oid}: scan not finished; references not read")
        rows, cursor = [], 0
        for chunk in sorted(record["chunks"], key=lambda r: r["start_index"]):
            if (chunk["status"] != "complete" or chunk["return_code"] != 0 or chunk["timed_out"]
                    or chunk["start_index"] != cursor):
                raise ValueError("full scan contains a failure, gap, or overlap")
            raw = Path(chunk["raw_directory"]) / "period-scan.txt"
            if sha256_file(raw) != chunk["output_periods_sha256"]:
                raise ValueError("full-scan output changed")
            table = parse_period_scan_rows(raw)
            if len(table) != chunk["expected_rows"] or len(table) != chunk["rows_returned"]:
                raise ValueError("full-scan chunk row count mismatch")
            cursor += len(table)
            if chunk["stop_index_exclusive"] != cursor:
                raise ValueError("full-scan chunk boundary mismatch")
            rows.extend(table)
        if cursor != counts[oid]["trial_count"] or cursor != record["rows_returned"]:
            raise ValueError("full scan does not cover the complete grid")
        tables[oid] = rows
    return by_id, tables


def command_score(args) -> int:
    counts = _counts(args.output_dir)
    samples = _validated_samples(_read_jsonl(args.output_dir / "sample.jsonl"), counts,
                                 allow_timeouts=True)
    calibration_path = args.output_dir / "calibration.jsonl"
    calibration = (_validated_samples(_read_jsonl(calibration_path), counts, True)
                   if calibration_path.exists() else None)
    full, tables = _validated_full(_read_jsonl(args.output_dir / "full.jsonl"), counts)
    factor, per_object = None, {}
    if calibration is not None:
        e, p = [], []
        for oid in CALIBRATION_OBJECTS:
            for block in range(10):
                e.append(samples[(oid, block)]["cpu_seconds"])
                p.append(calibration[(oid, block)]["cpu_seconds"])
            per_object[oid] = sum(e[-10:]) / sum(p[-10:])
        e, p = np.asarray(e, dtype=float), np.asarray(p, dtype=float)
        if not np.all(np.isfinite(e)) or not np.all(np.isfinite(p)) or np.any(p <= 0):
            raise ValueError("invalid calibration timing")
        factor = float(e.sum() / p.sum())
    estimates = {}
    for oid in SAMPLE_OBJECTS:
        blocks = [samples[(oid, block)] for block in range(10)]
        if all(b["status"] == "complete" for b in blocks):
            estimate, kind = estimate_cost(blocks, counts[oid]["trial_count"]), "estimate"
        else:
            estimate, kind = lower_bound_cost(blocks, counts[oid]["trial_count"]), "lower_bound"
        estimates[oid] = {"trial_count": counts[oid]["trial_count"], "kind": kind, "E_core": estimate,
                          "startup_one_trial": _compact(samples[(oid, -1)]),
                          "startup_note": "One-trial call, including fitting; not a pure startup measurement; no subtraction.",
                          "P_core": (None if not factor else
                                     {key: {"estimated_full": value["estimated_full"] / factor,
                                            "ci95": [v / factor for v in value["ci95"]]}
                                      for key, value in estimate.items()} if kind == "estimate" else
                                     {key: {"estimated_full_at_least": value["estimated_full_at_least"] / factor}
                                      for key in ("wall_seconds", "cpu_seconds")
                                      for value in (estimate[key],)})}
    # This is the only access to DAMIT reference periods, after every scan is validated.
    references = {}
    with args.catalog.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["object_id"] in FULL_OBJECTS:
                references[row["object_id"]] = float(row["solutions"][0]["period_hours"])
    checks = {}
    for oid in FULL_OBJECTS:
        measured = full[oid]
        estimate = estimates[oid]["E_core"]
        checks[oid] = {"measured_cpu_seconds": measured["cpu_seconds"],
                       "parallel_wall_seconds": measured["wall_seconds"],
                       "rows_returned": measured["rows_returned"],
                       "estimated_cpu_seconds": estimate["cpu_seconds"]["estimated_full"],
                       "estimated_cpu_ci95": estimate["cpu_seconds"]["ci95"],
                       "cpu_actual_over_estimate": measured["cpu_seconds"] / estimate["cpu_seconds"]["estimated_full"],
                       "sum_solver_wall_seconds": measured["sum_solver_wall_seconds"],
                       "wall_actual_over_estimate": measured["sum_solver_wall_seconds"] / estimate["wall_seconds"]["estimated_full"],
                       "score": score_periods(tables[oid], references[oid])}
    summary = {"schema": "delphi.period-cost.summary.v1", "created_utc": _utc(),
               "exploratory_development_sample": True, "hardware_note": HARDWARE_NOTE,
               "fourier_seconds": FOURIER_SECONDS, "bootstrap_resamples": RESAMPLES,
               "bootstrap_seed": BOOTSTRAP_SEED,
               "P_over_E_speed_factor": factor, "calibration_object_factors": per_object,
               "factor_definition": "sum of matched E-core CPU seconds / sum of P-core CPU seconds",
               "P_interval_note": "E-core block bootstrap intervals divided by the fixed calibration factor; factor uncertainty excluded.",
               "sampling_note": "SHA256 integer seed, Python Random.randrange, ten independent draws with replacement.",
               "estimates": estimates, "full_scan_checks": checks}
    lines = ["# Classical period-search cost", "", "Exploratory development sample.", "", HARDWARE_NOTE, "",
             "95% intervals: 10,000 bootstrap resamples of ten ten-trial blocks (seed 20261026).",
             "Startup calls include one fitted trial and are reported separately, without subtraction.",
             "P-core estimates divide E-core timings by the matched CPU-time calibration factor.",
             f"P/E speed factor: {factor if factor is not None else 'unavailable (no calibration)' }.", "",
             "| Object | Trials | E-core CPU s (95% interval) | P-core CPU s |",
             "| --- | ---: | ---: | ---: |"]
    for oid, value in estimates.items():
        e = value["E_core"]["cpu_seconds"]
        if value["kind"] == "lower_bound":
            p = f"> {value['P_core']['cpu_seconds']['estimated_full_at_least']:.0f}" if value["P_core"] else "unavailable"
            lines.append(f"| {oid} | {value['trial_count']} | > {e['estimated_full_at_least']:.0f} "
                         f"({value['E_core']['timed_out_blocks']} blocks timed out) | {p} |")
            continue
        p = f"{value['P_core']['cpu_seconds']['estimated_full']:.2f}" if value["P_core"] else "unavailable"
        lines.append(f"| {oid} | {value['trial_count']} | {e['estimated_full']:.2f} ({e['ci95'][0]:.2f}–{e['ci95'][1]:.2f}) | {p} |")
    lines += ["", "Complete scans (parallel wall time includes process preparation):"]
    for oid, value in checks.items():
        score = value["score"]
        lines.append(f"- {oid}: {value['rows_returned']} rows, {value['measured_cpu_seconds']:.2f} CPU s, "
                     f"{value['parallel_wall_seconds']:.2f} wall s; CPU actual/estimate "
                     f"{value['cpu_actual_over_estimate']:.3f}; fitted period {score['period_found_hours']:.6f} h "
                     f"({score['period_class']}), {score['ambiguity_count']} other minima within 10% chi2 and >1% away.")
    lines += ["", "Fitted-period offsets in step units are descriptive; row counts validate the starting grid.",
              "Local minima follow start-grid order, count a plateau once, and include endpoints.",
              "P-core intervals exclude uncertainty in the calibration factor."]
    destinations = [args.output_dir / "period-cost-summary.json", args.output_dir / "RESULTS.md"]
    if any(path.exists() for path in destinations):
        raise FileExistsError("score output already exists")
    _write_json(destinations[0], summary)
    with destinations[1].open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    print(f"Scored {len(checks)} complete scans; wrote {destinations[0]}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-dir", type=Path, default=OUTPUT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    count = subparsers.add_parser("count", parents=[common])
    count.add_argument("--catalog", type=Path, default=CATALOG)
    count.add_argument("--cohort", type=Path, default=COHORT)
    count.add_argument("--lightcurve-root", type=Path, default=LIGHTCURVES)
    count.set_defaults(function=command_count)
    for name in ("sample", "full"):
        sub = subparsers.add_parser(name, parents=[common])
        sub.add_argument("--cpus", type=parse_cpus, default=parse_cpus("12-23"))
        sub.add_argument("--solver-source", type=Path, default=SOURCE)
        if name == "sample":
            sub.add_argument("--workers", type=int, default=12)
            sub.add_argument("--calibrate-cpu", type=int)
            sub.set_defaults(function=command_sample)
        else:
            sub.add_argument("--objects", nargs="+", choices=FULL_OBJECTS, default=list(FULL_OBJECTS))
            sub.add_argument("--chunks", type=int, default=12)
            sub.set_defaults(function=command_full)
    score = subparsers.add_parser("score", parents=[common])
    score.add_argument("--catalog", type=Path, default=CATALOG)
    score.set_defaults(function=command_score)
    args = parser.parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
