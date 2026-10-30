"""V7: the broad-grid test (about 20 starts near the DeLPHI axes vs the 146-start grid) at our period.

    PY=<pinned python of the published run>; export PYTHONPATH=<worktree>
    $PY repro/own_period/v7_grid.py lock     --periods {published|FILE} --output LOCK [--workers N --device cpu]
    $PY repro/own_period/v7_grid.py execute  --lock LOCK --output DIR [--ids ...] [--repeats 0 1 2] [--max-seconds S]
    $PY repro/own_period/v7_grid.py validate --execution DIR          (compare with the published cases)
    $PY repro/own_period/v7_grid.py score    --lock LOCK --execution DIR --output JSON

The published study (``data/pole-grid-benchmark/20260912-environment-relocked-v3``,
``lc_pipeline/grid_benchmark.py``) ran five arms in a cold and a warm mode with
three repeats.  The published V7 numbers (``cold_step20_primary`` and
``angular_comparison_step20``) use only the cold ``guided20`` and ``classical20``
cases, so only those are run here, in the published order (the published
deterministic case order with the other cases left out).  Every case runs as
published: a fresh worker process per case
(``python -m repro.run_k3_grid_benchmark _worker --job``) with the published
solver (expanded convexinv, sha 4df011a5), tolerance 0.0003, iteration cap 1000,
300 s per start, six solver threads; the guided worker loads the five networks
of the asteroid's test fold and predicts at the supplied period.

The lock is the published lock with only ``period_hours`` replaced (our period
rounded to six decimals) and ``code_root`` set to this worktree, whose runtime
sources used by the workers are those of the published lock (``bundle.py``
differs only in an error message).  ``execute`` never opens a reference pole.
Output paths are absolute.  ``score`` runs only on a complete execution.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from lc_pipeline import grid_benchmark as grid
from lc_pipeline.k3 import grid_benchmark_scoring as gs

from repro.own_period import DataPath
from repro import network_root as network

ROOT = DataPath("../../..")
FOLLOWUP = DataPath("../..")
PUBLISHED = FOLLOWUP / "data/pole-grid-benchmark/20260912-environment-relocked-v3"
PUBLISHED_LOCK = PUBLISHED / "lock.json"
PUBLISHED_FULL = PUBLISHED / "full"
CODE_ROOT = Path(__file__).resolve().parents[2]
PERIOD_SEARCH = FOLLOWUP / "experiments/period-refinement-20260921/warm170-first.jsonl"
ARMS = ("guided20", "classical20")
MODE = "cold"


def lock(args: argparse.Namespace) -> int:
    if network.switched():
        return rerun_lock(args)
    published = grid.read_json(PUBLISHED_LOCK)
    objects = published["objects"]
    meta: dict = {"published_lock": {"path": str(PUBLISHED_LOCK), "sha256": grid.digest(PUBLISHED_LOCK)}}
    if args.periods != "published":
        source = json.loads(Path(args.periods).read_text())
        periods = source["periods_hours"]
        objects = [{**row, "period_hours": float(periods[row["object_id"]])} for row in objects]
        bad = [r["object_id"] for r in objects if round(r["period_hours"], 6) != r["period_hours"]]
        if bad:
            raise RuntimeError(f"periods must be rounded to six decimals: {bad}")
        meta["own_period"] = {
            "change": "period_hours of every object replaced; nothing else changed",
            "periods": {"path": str(Path(args.periods).resolve()), "sha256": grid.digest(Path(args.periods)),
                        "provenance": source["provenance"]},
        }
    settings = dict(published["settings"])
    overrides = {}
    if args.workers:
        overrides["workers"] = args.workers
        settings["workers"] = args.workers
    device = args.device or published["device"]
    if device != published["device"]:
        overrides["device"] = device
    if overrides:
        meta["validation_overrides"] = {**overrides, "note": "untimed validation only; fits do not depend on the number of solver threads"}
    payload = {
        **published,
        "objects": objects,
        "settings": settings,
        "device": device,
        "code_root": str(CODE_ROOT),
        "phase": "v67_own_period_frozen_before_execution",
        "v67": meta,
    }
    used = ("lc_pipeline/grid_benchmark.py", "lc_pipeline/v2/convexinv.py", "lc_pipeline/k3/inference.py",
            "lc_pipeline/k3/model.py", "lc_pipeline/k3/tokenizer.py", "lc_pipeline/k3/damit.py",
            "lc_pipeline/v2/preprocessing.py", "repro/run_k3_grid_benchmark.py")
    current = grid._runtime_sources(CODE_ROOT)
    differing = sorted(k for k in published["runtime_sources"] if current.get(k) != published["runtime_sources"][k])
    if any(k in differing for k in used) or set(differing) - {"lc_pipeline/k3/bundle.py", "lc_pipeline/k3/grid_benchmark_scoring.py", "lc_pipeline/version.py"}:
        raise RuntimeError(f"runtime sources differ from the published lock: {differing}")
    payload["v67"]["runtime_sources_differing_from_published"] = differing
    payload["runtime_sources"] = current
    grid.write_once(args.output, payload)
    print(args.output, len(objects), "objects", "overrides", overrides)
    return 0


def jobs_for(lock_doc: dict, ids: list[str] | None, repeats: list[int] | None, arms=ARMS) -> list[dict]:
    jobs = grid._jobs(lock_doc, "full")  # published deterministic order
    keep = set(ids) if ids else None
    return [
        j for j in jobs
        if j["timing_mode"] == MODE and j["arm"] in arms
        and (keep is None or j["object_id"] in keep)
        and (repeats is None or j["repeat_index"] in repeats)
    ]


def case_name(job: dict) -> str:
    return f"{job['object_id']}-r{job['repeat_index']}-{job['arm']}-{job['timing_mode']}"


def execute(args: argparse.Namespace) -> int:
    network.guard_output(args.output)
    lock_path = args.lock.resolve()
    output = args.output.resolve()  # absolute: the solver runs with cwd in each start directory
    doc = grid.read_json(lock_path)
    if network.switched():
        validate_rerun_lock(lock_path)
    for row in doc["objects"]:
        grid._verify_binding(row["lightcurve"])
    if grid.digest(Path(doc["solver"]["executable"]["path"])) != doc["solver"]["executable"]["sha256"]:
        raise RuntimeError("solver binary changed")
    if grid._runtime_sources(Path(doc["code_root"])) != doc["runtime_sources"]:
        raise RuntimeError("runtime source changed since lock")
    jobs = jobs_for(doc, args.ids, args.repeats, tuple(args.arms))
    header = {"schema": "delphi.v67.grid-execution.v1", "lock_sha256": grid.digest(lock_path),
              "arms": list(args.arms), "timing_mode": MODE, "ids": args.ids, "repeats": args.repeats,
              "case_count": len(jobs), "python": sys.executable}
    output.mkdir(parents=True, exist_ok=True)
    header_path = output / "execution-header.json"
    if header_path.exists():
        if grid.read_json(header_path) != header:
            raise RuntimeError("existing execution header differs; refusing to mix runs")
    else:
        grid.write_once(header_path, header)
    environment = grid._worker_environment()
    chunk_started = time.perf_counter()
    ran = []
    for position, job in enumerate(jobs):
        name = case_name(job)
        directory = output / "cases" / name
        if (directory / "timed-result.json").exists():
            if network.switched():
                audit_case(directory)
            continue
        if args.max_seconds and time.perf_counter() - chunk_started > args.max_seconds:
            print(f"chunk limit reached at case {position + 1}/{len(jobs)}", flush=True)
            _write_chunk(output, chunk_started, ran, complete=False)
            return 3
        if directory.exists() or (output / "jobs" / f"{name}.json").exists():
            raise RuntimeError(f"incomplete case left by an earlier run, needs manual audit: {name}")
        if network.switched() and job["arm"] == "classical20" and doc["rerun"]["classical_timing"] == "reuse":
            source = Path(doc["rerun"]["classical_source"]) / "cases" / name
            directory.mkdir(parents=True)
            for path in sorted(source.glob("start-*")):
                (directory / path.name).symlink_to(path, target_is_directory=True)
            row = grid.read_json(source / "timed-result.json")
            from repro.direct_workflow_benchmark.rerun import bound
            grid.write_once(directory / "timed-result.json", {**row, **network.json_metadata(),
                             "classical_source": bound(source / "timed-result.json")})
            audit_case(directory)
            continue
        job = {**job, "directory": str(directory)}
        job_path = output / "jobs" / f"{name}.json"
        grid.write_once(job_path, job)
        started = time.perf_counter()
        child = subprocess.Popen(
            [sys.executable, "-m", "repro.run_k3_grid_benchmark", "_worker", "--lock", str(lock_path), "--job", str(job_path)],
            cwd=doc["code_root"], env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True,
        )
        try:
            stdout, stderr = child.communicate(timeout=15000.0)
        except BaseException:
            grid._terminate_process_group(child)
            raise
        elapsed = time.perf_counter() - started
        with (output / f"{name}.log").open("x") as stream:
            stream.write(stderr)
        if child.returncode != 0:
            raise RuntimeError(f"cold worker failed for {name}; see log")
        record = json.loads(stdout)
        if network.switched():
            record.update(network.json_metadata())
            if job["arm"] == "classical20":
                from repro.direct_workflow_benchmark.rerun import audit_start, bound, equal_start
                source = Path(doc["rerun"]["classical_source"]) / "cases" / name
                old = sorted(source.glob("start-*/record.json"))
                new = sorted(directory.glob("start-*/record.json"))
                if len(old) != len(new):
                    raise ValueError("retimed grid start count changed")
                for a, b in zip(new, old, strict=True):
                    equal_start(audit_start(a), audit_start(b))
                record["classical_source"] = bound(source / "timed-result.json")
        record["wall_seconds"] = elapsed
        record["case_path"] = str(directory.relative_to(output))
        grid.write_once(directory / "timed-result.json", record)
        ran.append({"case": name, "wall_seconds": elapsed})
        print(f"grid v67: {position + 1}/{len(jobs)} {name} {elapsed:.2f}s valid={record['valid_starts']}/{record['starts_requested']}", flush=True)
        if len(ran) <= 3:
            reasons = []
            for cell in sorted(directory.glob("start-*/record.json")):
                reasons.append(grid.read_json(cell)["failure_reason"])
            if "fixed_period_changed" in reasons:
                raise RuntimeError(f"fixed_period_changed in {name}: period rounding problem; stopping")
            if len(ran) == 3 and all(
                grid.read_json(output / "cases" / r["case"] / "timed-result.json")["valid_starts"] == 0 for r in ran
            ):
                raise RuntimeError("no valid fit in the first three cases; stopping")
    _write_chunk(output, chunk_started, ran, complete=True)
    rows = []
    for job in jobs:
        path = output / "cases" / case_name(job) / "timed-result.json"
        rows.append({"path": str(path.relative_to(output)), "sha256": grid.digest(path)})
    if not (output / "execution.json").exists():
        grid.write_once(output / "execution.json", {**header, "phase": "blind_execution_and_fit_selection_complete",
                                                    "execution_complete": True, "reference_records_opened": False,
                                                    "rows": rows})
    return 0


def _write_chunk(output: Path, started: float, ran: list, *, complete: bool) -> None:
    if not ran:
        return
    stamp = time.strftime("%Y%m%dT%H%M%S")
    grid.write_once(output / "chunks" / f"chunk-{stamp}.json", {
        "cases": ran, "chunk_elapsed_seconds": time.perf_counter() - started,
        "serial_case_seconds": sum(r["wall_seconds"] for r in ran), "execution_finished": complete})


def _angle(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), float(np.dot(a, b)))))


def validate(args: argparse.Namespace) -> int:
    """Compare every case of a published-period run with the published case."""
    output = args.execution.resolve()
    cases = sorted(p.parent for p in output.glob("cases/*/timed-result.json"))
    report, worst, worst_pred, mismatches = [], 0.0, 0.0, []
    for case in cases:
        mine = grid.read_json(case / "timed-result.json")
        pub_dir = PUBLISHED_FULL / "cases" / case.name
        pub = grid.read_json(pub_dir / "timed-result.json")
        item = {"case": case.name, "valid_mine": mine["valid_starts"], "valid_published": pub["valid_starts"],
                "starts": mine["starts_requested"]}
        if mine["arm"].startswith("guided"):
            a = grid.read_json(case / "prediction.json")["axes"]
            b = grid.read_json(pub_dir / "prediction.json")["axes"]
            d = max(_angle(x["axis_xyz"], y["axis_xyz"]) for x, y in zip(a, b))
            item["prediction_max_angle_deg"] = d
            worst_pred = max(worst_pred, d)
        fa, fb = mine["selected_fit"], pub["selected_fit"]
        if (fa is None) != (fb is None) or mine["starts_requested"] != pub["starts_requested"] or mine["valid_starts"] != pub["valid_starts"]:
            mismatches.append(case.name)
        elif fa is not None:
            d = _angle(fa["axis"], fb["axis"])
            item.update(selected_angle_deg=d, rms_mine=fa["final_relative_rms"], rms_published=fb["final_relative_rms"],
                        start_mine=fa["start_index"], start_published=fb["start_index"])
            worst = max(worst, d)
            if d > 1e-6 or fa["final_relative_rms"] != fb["final_relative_rms"] or fa["start_index"] != fb["start_index"]:
                mismatches.append(case.name)
        # every start, not only the selected one
        cell_mismatch = 0
        for cell in sorted(case.glob("start-*/record.json")):
            x = grid.read_json(cell)
            y = grid.read_json(pub_dir / cell.parent.name / "record.json")
            sx, sy = x["selected_fit"], y["selected_fit"]
            if (sx is None) != (sy is None) or (sx is not None and (_angle(sx["axis"], sy["axis"]) > 1e-6 or sx["final_relative_rms"] != sy["final_relative_rms"])):
                cell_mismatch += 1
        item["start_records_differing"] = cell_mismatch
        if cell_mismatch:
            mismatches.append(case.name + ":starts")
        report.append(item)
    result = {"cases": len(cases), "max_selected_angle_deg_atan2": worst, "max_prediction_angle_deg": worst_pred,
              "mismatches": sorted(set(mismatches)), "passed": not mismatches, "rows": report}
    grid.write_once(output / "validation-vs-published.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
    return 0 if result["passed"] else 1


def crosscheck(args: argparse.Namespace) -> int:
    """Compare the classical20 fits with the own-period full grid of the 140 (read only)."""
    output = args.execution.resolve()
    grid_root = DataPath('benchmark/full-grid/cases')
    worst, differing, compared, cases = 0.0, [], 0, 0
    for other in sorted(grid_root.glob("*-r0-classical20-cold")):
        oid = other.name.split("-r0-")[0]
        for r in range(3):
            mine = output / "cases" / f"{oid}-r{r}-classical20-cold"
            cases += 1
            for cell in sorted(other.glob("start-*/record.json")):
                x = grid.read_json(cell)["selected_fit"]
                y = grid.read_json(mine / cell.parent.name / "record.json")["selected_fit"]
                compared += 1
                if (x is None) != (y is None):
                    differing.append(f"{oid}-r{r}/{cell.parent.name}:validity")
                elif x is not None:
                    d = _angle(x["axis"], y["axis"])
                    worst = max(worst, d)
                    if d > 1e-6 or x["final_relative_rms"] != y["final_relative_rms"]:
                        differing.append(f"{oid}-r{r}/{cell.parent.name}")
    result = {"compared_cases": cases, "compared_starts": compared, "max_angle_deg_atan2": worst,
              "differing_starts": differing, "identical": not differing}
    grid.write_once(output / "crosscheck-own-period-full-grid.json", result)
    print(json.dumps({k: (v if k != "differing_starts" else len(v)) for k, v in result.items()}))
    return 0


def _period_search_seconds() -> dict[str, float]:
    out = {}
    for line in PERIOD_SEARCH.read_text().splitlines():
        row = json.loads(line)
        if row["method"] == "fourier_gpu" and row["mode"] == "warm" and row["repeat"] == 0:
            out[row["object_id"]] = float(row["seconds"])
    return out


def score(args: argparse.Namespace) -> int:
    network.guard_output(args.output)
    doc = grid.read_json(args.lock)
    if network.switched():
        validate_rerun_lock(args.lock)
    execution = args.execution.resolve()
    published_layout = execution == PUBLISHED_FULL.resolve()
    if not published_layout:
        state = grid.read_json(execution / "execution.json")
        if state.get("execution_complete") is not True or state["lock_sha256"] != grid.digest(args.lock):
            raise RuntimeError("execution incomplete or bound to another lock; scoring refused")
    jobs = jobs_for(doc, None, None)
    if network.switched():
        expected = [{"path": str(Path("cases") / case_name(j) / "timed-result.json"),
                     "sha256": grid.digest(execution / "cases" / case_name(j) / "timed-result.json")}
                    for j in jobs]
        if state["rows"] != expected:
            raise ValueError("grid completion rows changed or are incomplete")
    records = {}
    for job in jobs:
        case = execution / "cases" / case_name(job)
        if network.switched():
            audit_case(case)
        row = grid.read_json(case / "timed-result.json")
        # integrity: the written selection is the minimum-RMS valid start
        fits = [grid.read_json(c)["selected_fit"] for c in sorted(case.glob("start-*/record.json"))]
        fits = [f for f in fits if f is not None]
        best = min(fits, key=lambda f: (f["final_relative_rms"], f["start_index"])) if fits else None
        if best != row["selected_fit"] or len(fits) != row["valid_starts"]:
            raise RuntimeError(f"selection does not match start records: {case.name}")
        records[(job["object_id"], job["arm"], job["repeat_index"])] = row
    # Reference poles are opened only now.
    catalog_path = Path(doc["inputs"]["reference_catalog"]["path"])
    if grid.digest(catalog_path) != doc["inputs"]["reference_catalog"]["sha256"]:
        raise RuntimeError("reference catalogue changed")
    ids = sorted({j["object_id"] for j in jobs})
    refs = {}
    for line in catalog_path.read_text().splitlines():
        entry = json.loads(line)
        if entry["object_id"] in ids:
            refs[entry["object_id"]] = np.asarray([s["vector"] for s in entry["solutions"]], dtype=np.float64)
    fold = {r["object_id"]: int(r["fold"]) for r in doc["objects"]}
    folds = np.asarray([fold[o] for o in ids], dtype=np.int8)
    search = _period_search_seconds()

    def arm_arrays(arm: str, add: dict[str, float] | None = None):
        n = len(ids)
        times, completed, recovered = np.empty((n, 3)), np.zeros((n, 3), bool), np.zeros((n, 3), bool)
        rms, errors = np.full((n, 3), np.nan), np.full((n, 3), np.nan)
        for i, oid in enumerate(ids):
            for r in range(3):
                rec = records[(oid, arm, r)]
                times[i, r] = float(rec["wall_seconds"]) + (add[oid] if add else 0.0)
                completed[i, r] = bool(rec["completed"])
                fit = rec["selected_fit"]
                if fit is not None:
                    e = gs._axial_error_degrees(gs._vector(fit["axis"], name="axis", require_unit=True), refs[oid])
                    errors[i, r] = e
                    recovered[i, r] = e <= gs.RECOVERY_MAX_DEGREES
                    rms[i, r] = float(fit["final_relative_rms"])
        return {"times": times, "completed": completed, "recovered": recovered, "rms": rms, "errors": errors,
                "object_completion": completed.sum(1) >= 2, "object_recovery": recovered.sum(1) >= 2}

    def analysis(add=None, keep=None):
        b, c = arm_arrays("classical20", add), arm_arrays("guided20", add)
        sel = np.ones(len(ids), bool) if keep is None else keep
        sub = lambda d: {k: v[sel] for k, v in d.items()}
        b, c = sub(b), sub(c)
        sel_ids = tuple(o for o, k in zip(ids, sel) if k)
        f = folds[sel]
        runtime = gs._runtime_metric(b["times"], c["times"], f, resamples=10000, seed=20260911)
        recovery = gs._paired_binary_noninferiority(b["object_recovery"].astype(np.int8), c["object_recovery"].astype(np.int8))
        completion = gs._paired_binary_noninferiority(b["object_completion"].astype(np.int8), c["object_completion"].astype(np.int8))
        rmsm = gs._rms_metric(b["rms"], c["rms"], b["completed"] & c["completed"], sel_ids, f, resamples=10000, seed=20260911)
        return {"objects": int(sel.sum()),
                "runtime_point_ratio": runtime["point_ratio"], "runtime_interval_95": runtime["percentile_interval_95"],
                "classical_mean_seconds": runtime["classical_seconds"]["mean"], "guided_mean_seconds": runtime["guided_seconds"]["mean"],
                "rms_point_ratio": rmsm.get("point_ratio"), "rms_interval_95": rmsm.get("percentile_interval_95"),
                "rms_object_support": rmsm["object_support"],
                "baseline_success_count": recovery["baseline_success_count"], "candidate_success_count": recovery["candidate_success_count"],
                "recovery_difference_lower": recovery["simultaneous_exact_lower"],
                "completion": {k: completion[k] for k in ("baseline_success_count", "candidate_success_count")},
                "valid_starts": {"classical20": int(sum(records[(o, 'classical20', r)]["valid_starts"] for o in sel_ids for r in range(3))),
                                 "guided20": int(sum(records[(o, 'guided20', r)]["valid_starts"] for o in sel_ids for r in range(3)))},
                }, b, c, sel_ids, f

    def angular(b, c, sel_ids, f, failure_value=None):
        rows, diffs = [], []
        for i, oid in enumerate(sel_ids):
            vals = []
            for arr in (b["errors"][i], c["errors"][i]):
                fin = arr[np.isfinite(arr)]
                vals.append(float(fin.mean()) if fin.size else failure_value)
            if None in vals:
                continue
            rows.append((oid, int(f[i]), vals[0], vals[1]))
            diffs.append(vals[1] - vals[0])
        diffs = np.asarray(diffs)
        rng = np.random.default_rng(20260915)
        by_fold = [np.asarray([k for k, r in enumerate(rows) if r[1] == fo]) for fo in sorted({r[1] for r in rows})]
        boot = np.empty(10000)
        for d in range(boot.size):
            idx = np.concatenate([rng.choice(v, size=len(v), replace=True) for v in by_fold])
            boot[d] = diffs[idx].mean()
        lower = sum(r[3] < r[2] for r in rows); equal = sum(r[3] == r[2] for r in rows)
        return {"n_objects": len(rows), "guided_minus_classical_mean_error_deg": float(diffs.mean()),
                "guided_minus_classical_median_error_deg": float(np.median(diffs)),
                "interval_95_deg": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                "classical_mean_error_deg": float(np.mean([r[2] for r in rows])), "guided_mean_error_deg": float(np.mean([r[3] for r in rows])),
                "guided_lower_equal_higher_error_counts": [lower, equal, len(rows) - lower - equal],
                "failure_value_deg": failure_value}

    primary, b, c, sel_ids, f = analysis()
    no_fit = [o for i, o in enumerate(ids) if not (np.isfinite(b["errors"][i]).any() and np.isfinite(c["errors"][i]).any())]
    keep = np.asarray([o not in no_fit for o in ids])
    excluded, be, ce, sel_e, fe = analysis(keep=keep)
    with_search_ids = np.asarray([o in search for o in ids])
    timed_search, *_ = analysis(add={o: search.get(o, 0.0) for o in ids}, keep=with_search_ids)
    result = {
        **network.json_metadata(),
        "schema": "delphi.v67.grid-score.v1",
        "lock_sha256": grid.digest(args.lock), "execution": str(execution),
        "convention": "published: classical20 vs guided20, cold, three repeats; runtime = sum classical / sum guided wall seconds; RMS = geometric mean of object-level selected-fit RMS ratios over jointly completed repeats; agreement = per-object mean selected axial error over repeats with a fit",
        "cold_step20_primary": primary,
        "angular_comparison_step20": angular(b, c, sel_ids, f, failure_value=90.0),
        "angular_comparison_step20_failures_excluded": angular(be, ce, sel_e, fe, failure_value=None),
        "objects_without_any_fit_in_an_arm": no_fit,
        "cold_step20_failures_excluded": excluded,
        "cold_step20_with_period_search_time_added": {**timed_search, "period_search": {
            "file": str(PERIOD_SEARCH), "field": "seconds", "method": "fourier_gpu",
            "objects_with_time": int(with_search_ids.sum()),
            "mean_seconds": float(np.mean([search[o] for o in ids if o in search]))}},
    }
    if network.switched():
        result["classical_timing"] = doc["rerun"]["classical_timing"]
        result["classical_retiming_cpu_seconds"] = (sum(
            r["solver_cpu_seconds"] for key, r in records.items() if key[1] == "classical20")
            if doc["rerun"]["classical_timing"] == "retime" else 0)
    grid.write_once(args.output, result)
    print(json.dumps({k: result[k] for k in ("cold_step20_primary", "angular_comparison_step20")}, indent=1, default=str))
    return 0


def audit_case(directory):
    from repro.direct_workflow_benchmark.rerun import audit_start, verify
    path = directory / "timed-result.json"
    network.check_json(path)
    row = grid.read_json(path)
    if "classical_source" in row:
        verify(row["classical_source"])
    for relative, digest in row["cell_files"].items():
        record_path = directory / relative
        if grid.digest(record_path) != digest:
            raise ValueError("grid start receipt changed")
        audit_start(record_path)
    if ("classical_source" not in row and row.get("prediction_sha256") is not None
            and grid.digest(directory / "prediction.json") != row["prediction_sha256"]):
        raise ValueError("grid prediction changed")


def validate_rerun_lock(path):
    from repro.direct_workflow_benchmark.rerun import verify
    network.check_json(path)
    value = grid.read_json(path)
    old = grid.read_json(verify(value["rerun"]["source_lock"]))
    if value["objects"] != old["objects"] or value["settings"] != {**old["settings"], "workers": 4}:
        raise ValueError("fixed V7 objects or settings changed")
    if value["solver"] != old["solver"] or value["inputs"] != old["inputs"]:
        raise ValueError("fixed V7 solver or inputs changed")
    if value["bundles"] != [network.bundle_record(f, old["bundles"]) for f in range(5)]:
        raise ValueError("chosen V7 bundles changed")
    return value


def rerun_lock(args):
    from repro.direct_workflow_benchmark.rerun import bound, save
    source = PUBLISHED_LOCK if args.periods == "published" else DataPath("v67/v7-own-period-lock.json")
    original = grid.read_json(source)
    if args.periods != "published":
        periods = grid.read_json(args.periods)["periods_hours"]
        if any(float(periods[r["object_id"]]) != r["period_hours"] for r in original["objects"]):
            raise ValueError("saved V7 periods changed")
    settings = {**original["settings"], "workers": 4}
    payload = {**original, **network.json_metadata(), "settings": settings,
        "code_root": str(CODE_ROOT), "runtime_sources": grid._runtime_sources(CODE_ROOT),
        "bundles": [network.bundle_record(fold, original["bundles"]) for fold in range(5)],
        "device": args.device or original["device"],
        "environment": grid._environment(args.device or original["device"]),
        "rerun": {"source_lock": bound(source), "classical_timing": args.classical_timing,
                  "classical_source": str(PUBLISHED_FULL if args.periods == "published" else DataPath("v67/v7-own-period")),
                  "order": "original grid order_seed=20260911, all other arms filtered out"}}
    save(args.output, payload)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("lock")
    p.add_argument("--periods", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", choices=("cpu", "cuda"))
    p.add_argument("--classical-timing", choices=("reuse", "retime"), default="reuse")
    e = sub.add_parser("execute")
    e.add_argument("--lock", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    e.add_argument("--ids", nargs="*")
    e.add_argument("--repeats", nargs="*", type=int)
    e.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    e.add_argument("--max-seconds", type=float, default=0.0)
    v = sub.add_parser("validate")
    v.add_argument("--execution", type=Path, required=True)
    x = sub.add_parser("crosscheck")
    x.add_argument("--execution", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--lock", type=Path, required=True)
    s.add_argument("--execution", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return {"lock": lock, "execute": execute, "validate": validate, "score": score, "crosscheck": crosscheck}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
