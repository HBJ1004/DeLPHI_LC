#!/usr/bin/env python3
"""POST HOC spacecraft inversion, PROTOCOL.md Addenda 3/4 (free period).

Order: validate, export, run, score. Use the pinned Python with -I; this script
adds only its own worktree because -I ignores PYTHONPATH. No GPU is used.
All products are exclusive writes under --output-dir (default classical-freep).
--addendum 4 selects Ceres/Vesta and defaults to classical-freep-ceres-vesta.
Run resumes hash-verified starts and archives incomplete starts, as classical.py
does. Other stages refuse existing destinations. Workers may change on resume.

The sole optimization change is parameters.txt line 3: P 1 instead of P 0.
This enables local Levenberg-Marquardt refinement of angular frequency, with
no period scan, alias search or YORP term. Starting periods are unchanged.
Fitted periods use solution-parameters.txt (8 decimals in hours); stdout has 6.
No reference poles or DeLPHI scores are opened until score.

validate runs this execution path with the period flag OFF and compares one
archived classical.py start, including every product byte and all record fields
except nondeterministic solver_wall_seconds. --max-starts is for smoke tests;
score requires --allow-partial for those and labels their reports accordingly.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import fcntl
import json
import math
import resource
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
WORKTREE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKTREE))

# Import before numpy: classical pins BLAS/OpenMP threads to one per worker.
from repro.own_period.spacecraft import classical as c  # noqa: E402
from lc_pipeline.v2.convexinv import ConvexinvParameters  # noqa: E402

np, grid = c.np, c.grid
RUNS = ("itokawa-combined", "itokawa-lowry2014", "eros-damit",
        "lutetia-damit", "ida-damit", "gaspra-damit")
OUTPUT = c.SPACECRAFT / "classical-freep"
ADDENDUM4_RUNS = ("ceres-damit", "vesta-damit")
DAMIT_RESCORE = {"eros-damit": 1714, "ceres-damit": 3414, "vesta-damit": 3415}
IMPLEMENTATIONS = c.IMPLEMENTATIONS + (
    "lc_pipeline/workflow_benchmark.py", "repro/own_period/spacecraft/classical.py")
PERIOD_POLICY = "solver period-fit flag 1; local angular-frequency refinement from exact DeLPHI starting period; no scan or YORP"


def source_inputs(spacecraft: Path, addendum: int = 3) -> list[dict]:
    """Same source checks/export conventions as classical; include DAMIT rescores.

    classical.source_inputs hardcodes the Addendum 2 cohort. Keep this small
    cohort loader separate, reusing its lossless writer, round trip and guards.
    Only DAMIT model IDs and periods are accessed, never its pole columns.
    """
    from lc_pipeline.v2.data import parse_damit_lightcurve

    lock_path = spacecraft / "lock/cohort-lock.json"
    search_path, prepared_path = spacecraft / "search/periods.json", spacecraft / "prepared/manifest.json"
    lock, search, prepared = c.read(lock_path), c.read(search_path), c.read(prepared_path)
    if (search["cohort_lock_sha256"] != grid.digest(lock_path)
            or search["prepared_manifest_sha256"] != grid.digest(prepared_path)
            or prepared["cohort_lock_sha256"] != grid.digest(lock_path)):
        raise ValueError("search/prepared lock differs")
    table_path = c.checked(lock["files"]["model_table"])
    periods = {}
    with table_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            periods.setdefault(f"asteroid_{row['asteroid_id']}", []).append(
                (int(row["id"]), c.period(row["period"])))
    result = []
    for run_id in ADDENDUM4_RUNS if addendum == 4 else RUNS:
        dependencies = [c.binding(lock_path), c.binding(search_path), c.binding(prepared_path)]
        if run_id in DAMIT_RESCORE:
            object_id = f"asteroid_{DAMIT_RESCORE[run_id]}"
            meta = {"run_id": run_id, "name": run_id.removesuffix("-damit"), "object_id": object_id, "kind": "damit"}
            # The same snapshot as run.py, derived from its bound model table.
            path = table_path.parent.parent / "files" / object_id / "lc.txt"
            own_path = spacecraft.parent / "own-periods-170.json"
            own = c.read(own_path)
            if own["schema"] != "delphi.own-periods.v1":
                raise ValueError("unexpected 170-object own-period schema")
            hours = c.period(own["periods_hours"][meta["object_id"]])
            dependencies.append(c.binding(own_path))
        else:
            meta = c.unique(lock["inputs"], "run_id", run_id)
            hours = c.period(search["periods_hours"][run_id])
        if meta["kind"] == "external":
            entry = c.unique(prepared["objects"], "run_id", run_id)
            path = c.checked(entry, spacecraft / "prepared")
            value = c.read(path)
            if value["schema"] != "delphi.k3-external-prepared.v1" or value["object_id"] != meta["object_id"]:
                raise ValueError("prepared identity/schema differs")
            source_path = c.checked(entry["source_manifest"], spacecraft / "prepared")
            receipt_path = spacecraft / "sbdb/periods.json"
            source, receipt = c.read(source_path), c.read(receipt_path)
            if (source["objects"] != [value]
                    or source["period_receipts_sha256"] != grid.digest(receipt_path)
                    or receipt["cohort_lock_sha256"] != grid.digest(lock_path)):
                raise ValueError("prepared object or period receipt changed")
            cat = c.unique(receipt["objects"], "object_id", meta["object_id"])
            catalogue = c.period(cat["known_period_hours"])
            if cat["status"] != "ready" or catalogue != c.period(value["known_period_hours"]):
                raise ValueError("prepared catalogue period differs from SBDB receipt")
            catalogue_source = {"rule": "SBDB rot_per receipt (as prepared for DeLPHI)",
                                "receipt": c.binding(receipt_path)}
            epochs = tuple(c.ObservationEpoch(e["epoch_id"], tuple(c.Observation(**o) for o in e["observations"]))
                           for e in value["epochs"])
            flags = [0] * len(epochs)
            dependencies.extend([c.binding(source_path), c.binding(receipt_path)])
        else:
            if run_id not in DAMIT_RESCORE:
                path = c.checked(meta["source"])
            epochs = parse_damit_lightcurve(path)  # Exactly run.py's census loader.
            flags, _ = c.lc_payload(path)
            model_id, catalogue = min(periods[meta["object_id"]])
            catalogue_source = {"rule": "DAMIT lowest numeric model id, as run.py",
                                "model_id": model_id, "table": c.binding(table_path)}
            dependencies.append(c.binding(table_path))
        if run_id not in DAMIT_RESCORE and search["details"][run_id]["input_sha256"] != grid.digest(path):
            raise ValueError(f"observations differ from DeLPHI period search: {run_id}")
        result.append({"run_id": run_id, "name": meta["name"], "object_id": meta["object_id"],
                       "kind": meta["kind"], "input": c.binding(path), "dependencies": dependencies,
                       "periods_hours": {"own": hours, "catalogue": catalogue},
                       "catalogue_source": catalogue_source, "epochs": epochs, "flags": flags})
    return result


def cmd_export(args) -> dict:
    root = args.output_dir / "export"
    c.available(root)
    lock_path = args.benchmark_dir / "evaluation-lock.json"
    lock = c.benchmark(lock_path)
    rows = source_inputs(args.spacecraft_dir, args.addendum)
    root.mkdir(parents=True)
    proof = c.round_trip(c.unique(lock["objects"], "object_id", "asteroid_1733"), root / "round-trip/lc.txt")
    c.write(root / "round-trip.json", proof)
    entries = []
    for row in rows:
        path = root / "inputs" / f"{row['run_id']}.txt"
        stats = c.write_lightcurves(path, row["epochs"], row["flags"])
        if row["kind"] == "damit" and c.lc_payload(path) != c.lc_payload(c.checked(row["input"])):
            raise ValueError("DAMIT export differs from native input")
        structure = c.parse_lightcurve_structure(path)
        violations = c.capacity_violations(structure, lock["solver"]["capacities"])
        if violations:
            raise ValueError(f"solver capacity exceeded for {row['run_id']}: {violations}")
        entries.append({k: v for k, v in row.items() if k not in ("epochs", "flags")} |
                       {"lightcurve": c.relative_binding(path, args.output_dir), **stats,
                        "structure": structure.as_dict()})
    c.write(root / "manifest.json", {"schema": "delphi.spacecraft-classical-freep-export.v1", "posthoc": True,
            "protocol": c.binding(args.spacecraft_dir / "PROTOCOL.md"), "script": c.binding(Path(__file__)),
            "implementations": {name: c.binding(WORKTREE / name) for name in IMPLEMENTATIONS},
            "benchmark_lock": c.binding(lock_path), "round_trip": c.relative_binding(root / "round-trip.json", args.output_dir),
            "convention": c.CONVENTION, "inputs": entries})
    return {"stage": "export", "inputs": len(entries), "round_trip": proof["status"],
            "all_inputs_exact_float64_round_trip": all(r["exact_float64_round_trip"] for r in entries),
            "observations": {r["run_id"]: r["observations"] for r in entries}}


def load_export(out: Path, addendum: int = 3) -> dict:
    manifest = c.read(out / "export/manifest.json")
    if (manifest["schema"] != "delphi.spacecraft-classical-freep-export.v1"
            or (not c.switched() and c.checked(manifest["script"]) != Path(__file__).resolve())
            or [r["run_id"] for r in manifest["inputs"]] != list(ADDENDUM4_RUNS if addendum == 4 else RUNS)):
        raise ValueError(f"export must use this script and exactly the Addendum {addendum} inputs")
    c.network_run.checked_append_only(manifest["protocol"])
    for entry in manifest["implementations"].values():
        c.checked(entry)
    proof = c.read(c.checked(manifest["round_trip"], out))
    if proof["status"] != "PASS":
        raise ValueError("conversion round trip required")
    c.checked(proof["benchmark_input"])
    c.checked(proof["exported_input"])
    for row in manifest["inputs"]:
        c.checked(row["input"])
        c.checked(row["lightcurve"], out)
        for dep in row["dependencies"]:
            c.checked(dep)
    return manifest


def parameters(job: dict, vector, free_period: bool) -> ConvexinvParameters:
    lon = math.degrees(math.atan2(float(vector[1]), float(vector[0]))) % 360
    lat = math.degrees(math.asin(float(np.clip(vector[2], -1, 1))))
    return ConvexinvParameters(lambda_deg=lon, beta_deg=lat, period_hours=job["period_hours"],
                              free_period=free_period, iteration_stop_condition=grid.SETTINGS["convergence_tolerance"])


def fitted_period(target: Path) -> float | None:
    """Read the solver's higher precision period, including failed fit products."""
    try:
        value = float((target / "solution-parameters.txt").read_text().splitlines()[0].split()[2])
    except (OSError, ValueError, IndexError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def native_start(job: dict, index: int, vector, lock: dict, directory: Path, free_period: bool) -> dict:
    """grid._native_start with a configurable period flag and fitted-period check.

    The fixed adapter hardcodes False and rejects period changes, so this small
    native boundary must be separate. Preserve its operations and RMS summation
    order for bit-exact fixed-period validation. Keep its fixed record schema.
    """
    target = directory / f"start-{index:03d}"
    target.mkdir()
    settings, solver = lock["solver_settings"], lock["solver"]
    observed = c._parse_lightcurve_brightness(c.checked(job["lightcurve"]))
    parameter_path = target / "parameters.txt"
    with parameter_path.open("x", encoding="ascii") as stream:
        stream.write(parameters(job, vector, free_period).render())
    products = {"modelled": target / "modelled-lightcurve.txt",
                "parameters": target / "solution-parameters.txt", "areas": target / "solution-areas.txt"}
    command = [solver["executable"]["path"], "-v", "-o", str(products["areas"]),
               "-p", str(products["parameters"]), str(parameter_path), str(products["modelled"])]
    started = time.perf_counter()
    with Path(job["lightcurve"]["path"]).open("rb") as stream:
        try:
            result = subprocess.run(command, stdin=stream, capture_output=True, cwd=target,
                                    timeout=settings["timeout_seconds_per_start"], check=False)
            code, stdout, stderr, timed_out = result.returncode, result.stdout, result.stderr, False
        except subprocess.TimeoutExpired as exc:
            code, stdout, stderr, timed_out = None, exc.stdout or b"", exc.stderr or b"", True
    elapsed = time.perf_counter() - started
    for name, content in (("stdout.txt", stdout), ("stderr.txt", stderr)):
        with (target / name).open("xb") as stream:
            stream.write(content)
    text = stdout.decode("utf-8", errors="replace")
    iterations, final = list(c._ITERATION.finditer(text)), c._FINAL.search(text)
    count = int(iterations[-1].group(1)) if iterations else None
    selected, error = None, None
    fitted = fitted_period(target)
    try:
        if timed_out or code != 0 or count is None or not 0 < count < settings["iteration_cap"]:
            raise ValueError("timeout_exit_or_iteration_cap")
        diagnostics = [float(iterations[-1].group(i)) for i in (2, 3)]
        if any(not math.isfinite(x) or x < 0 for x in diagnostics):
            raise ValueError("invalid_final_fit_diagnostics")
        if final is None or not all(p.is_file() and p.stat().st_size for p in products.values()):
            raise ValueError("missing_solution_product")
        final_lon, final_lat, hours = (float(final.group(i)) for i in (1, 2, 3))
        if not all(math.isfinite(x) for x in (final_lon, final_lat, hours)):
            raise ValueError("nonfinite_solution")
        if free_period:
            if fitted is None or hours <= 0 or not math.isclose(fitted, hours, rel_tol=0, abs_tol=5.1e-7):
                raise ValueError("invalid_fitted_period")
        elif not math.isclose(hours, job["period_hours"], rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError("fixed_period_changed")
        predicted = np.asarray(products["modelled"].read_text().split(), dtype=np.float64)
        if len(predicted) != sum(len(curve) for curve in observed) or not np.all(np.isfinite(predicted)):
            raise ValueError("invalid_modelled_lightcurve")
        offset, residuals = 0, []
        for curve in observed:
            part = predicted[offset:offset + len(curve)]
            offset += len(curve)
            if float(curve.mean()) == 0 or float(part.mean()) == 0:
                raise ValueError("zero_mean_lightcurve")
            residuals.extend(np.square(curve / curve.mean() - part / part.mean()).tolist())
        rms = math.sqrt(sum(residuals) / len(residuals))
        if not math.isfinite(rms) or rms <= 0:
            raise ValueError("invalid_rms")
        selected = {"axis": grid._xyz(final_lon, final_lat), "final_relative_rms": rms,
                    "start_index": index, "raw_longitude": final_lon, "raw_latitude": final_lat}
        if free_period:
            selected["fitted_period_hours"] = fitted
    except (ValueError, OSError) as exc:
        error = str(exc)
    files = {p.name: grid.digest(p) for p in target.iterdir() if p.is_file()}
    record = {"start_index": index, "initial_axis": vector.tolist(), "return_code": code,
              "timed_out": timed_out, "iterations": count, "solver_wall_seconds": elapsed,
              "selected_fit": selected, "failure_reason": error, "files": files}
    if free_period:
        record.update(period_fit_flag=1, starting_period_hours=job["period_hours"], fitted_period_hours=fitted)
    c.write(target / "record.json", record)
    return record


def diagnostics(record: dict, target: Path) -> dict:
    return {**c.start_diagnostics(record, target), "fitted_period_hours": fitted_period(target),
            "fitted_period_source": "solution-parameters.txt first line, third column (hours, 8 decimals)"}


def pole_degrees(axis) -> dict:
    """Canonical coordinates for comparison with literature mirror poles."""
    x, y, z = axis
    return {"lambda_deg": math.degrees(math.atan2(y, x)) % 360,
            "beta_deg": math.degrees(math.atan2(z, math.hypot(x, y)))}


def verify_start(job: dict, index: int, vector, target: Path, free_period: bool = True) -> dict:
    record = c.verify_record(target, index, vector)
    if (target / "parameters.txt").read_text() != parameters(job, vector, free_period).render():
        raise ValueError(f"solver input/period-fit flag differs: {target}")
    if free_period and (record["period_fit_flag"] != 1
                        or record["starting_period_hours"] != job["period_hours"]
                        or record["fitted_period_hours"] != fitted_period(target)
                        or (record["selected_fit"] is not None
                            and record["selected_fit"]["fitted_period_hours"] != record["fitted_period_hours"])):
        raise ValueError(f"fitted period receipt differs: {target}")
    return record


def execute_start(job: dict, index: int, vector, lock: dict, case: Path, free_period: bool = True) -> dict:
    case.mkdir(parents=True, exist_ok=True)
    target = case / f"start-{index:03d}"
    if (target / "record.json").exists():
        record = verify_start(job, index, vector, target, free_period)
    else:
        if target.exists():
            archive = case / "interrupted"
            archive.mkdir(exist_ok=True)
            attempt = 0
            while (archive / f"{target.name}-{attempt:03d}").exists():
                attempt += 1
            target.rename(archive / f"{target.name}-{attempt:03d}")
        record = native_start(job, index, vector, lock, case, free_period)
    c.write_or_verify(target / "diagnostics.json", diagnostics(record, target))
    return record


def cmd_validate(args) -> dict:
    root = args.output_dir / "validation"
    c.available(root)
    lock_path = args.benchmark_dir / "evaluation-lock.json"
    lock = c.benchmark(lock_path)
    classical_dir = args.classical_dir or args.spacecraft_dir / "classical"
    # Addendum 3 changed PROTOCOL.md after the fixed-period export. Validate
    # numerical sources/observations, not that deliberately obsolete text hash.
    manifest = c.read(classical_dir / "export/manifest.json")
    if (manifest["schema"] != "delphi.spacecraft-classical-export.v1"
            or c.checked(manifest["script"]) != Path(c.__file__).resolve()):
        raise ValueError("validation requires a classical.py export")
    for entry in manifest["convention_sources"]:
        c.checked(entry)
    plan_path = classical_dir / "run.json"
    plan = c.read(plan_path)
    if (plan["export"] != c.binding(classical_dir / "export/manifest.json")
            or plan["benchmark_lock"] != c.binding(lock_path)):
        raise ValueError("classical validation source uses different export/benchmark")
    for entry in plan["implementations"].values():
        c.checked(entry)
    row = c.unique(manifest["inputs"], "run_id", args.run_id)
    c.checked(row["input"])
    for entry in row["dependencies"]:
        c.checked(entry)
    vectors = grid.starts_for_arm(args.arm)
    if not 0 <= args.start_index < len(vectors):
        raise ValueError("validation start index outside chosen arm")
    source = c.case_path(classical_dir, args.run_id, args.period, args.arm) / f"start-{args.start_index:03d}"
    job = {"object_id": row["object_id"], "period_hours": row["periods_hours"][args.period],
           "lightcurve": c.binding(c.checked(row["lightcurve"], classical_dir))}
    expected = verify_start(job, args.start_index, vectors[args.start_index], source, False)
    root.mkdir(parents=True)
    proof = c.round_trip({**row, "lightcurve": job["lightcurve"]}, root / "lc.txt")
    c.write(root / "round-trip.json", proof)
    actual = execute_start({**job, "lightcurve": c.binding(root / "lc.txt")}, args.start_index,
                           vectors[args.start_index], lock, root / "cases", False)
    stable = lambda r: {k: v for k, v in r.items() if k != "solver_wall_seconds"}
    equal = stable(actual) == stable(expected)
    passed = actual["selected_fit"] is not None and equal
    report = {"schema": "delphi.spacecraft-classical-freep-validation.v1", "status": "PASS" if passed else "FAIL",
              "script": c.binding(Path(__file__)), "benchmark_lock": c.binding(lock_path),
              "classical_run_plan": c.binding(plan_path), "classical_record": c.binding(source / "record.json"),
              "run_id": args.run_id, "mode": args.period, "arm": args.arm, "start_index": args.start_index,
              "period_fit_flag": 0, "round_trip": proof, "bit_exact_except_wall_time": equal,
              "matching_product_hashes": actual["files"] == expected["files"],
              "comparison": "all record fields except solver_wall_seconds exact; every product byte hash identical",
              "differing_fields": [k for k in stable(expected).keys() | stable(actual).keys()
                                   if stable(expected).get(k) != stable(actual).get(k)],
              "expected_fit": expected["selected_fit"], "actual_fit": actual["selected_fit"],
              "observed_wall_seconds": actual["solver_wall_seconds"]}
    c.write(root / "result.json", report)
    if not passed:
        raise ValueError(f"classical start not reproduced; see {root / 'result.json'}")
    return {k: report[k] for k in ("status", "run_id", "mode", "arm", "start_index", "period_fit_flag",
                                  "bit_exact_except_wall_time", "matching_product_hashes", "observed_wall_seconds")}


def run_plan(args, manifest: dict, lock: dict) -> dict:
    validation_path = args.output_dir / "validation/result.json"
    validation = c.read(validation_path)
    if (validation["status"] != "PASS" or validation["period_fit_flag"] != 0
            or validation["script"] != c.binding(Path(__file__))
            or validation["benchmark_lock"] != manifest["benchmark_lock"]
            or manifest["benchmark_lock"] != c.binding(args.benchmark_dir / "evaluation-lock.json")):
        raise ValueError("validate and export must use this script and the same benchmark lock")
    c.checked(validation["classical_record"])
    cases, rows = [], []
    for row in manifest["inputs"]:
        if row["run_id"] not in args.runs:
            continue
        rows.append(row["run_id"])
        for mode in c.MODES:
            if mode not in args.periods:
                continue
            remaining = args.max_starts if args.max_starts is not None else 152
            for arm in c.ARMS:
                count = min(len(grid.starts_for_arm(arm)), remaining)
                remaining -= count
                if count:
                    cases.append({"run_id": row["run_id"], "mode": mode, "arm": arm,
                                  "period_hours": row["periods_hours"][mode], "starts": count})
    return {"schema": "delphi.spacecraft-classical-freep-run.v1", "posthoc": True,
            "export": c.binding(args.output_dir / "export/manifest.json"), "validation": c.binding(validation_path),
            "script": c.binding(Path(__file__)), "implementations": manifest["implementations"],
            "benchmark_lock": manifest["benchmark_lock"], "solver": lock["solver"],
            "solver_settings": {**lock["solver_settings"], "period": PERIOD_POLICY},
            "period_fit_flag": 1, "runs": rows, "periods": [m for m in c.MODES if m in args.periods], "cases": cases,
            "full_grid": args.max_starts is None or args.max_starts == 152,
            "reference_poles_opened": False, "period_policy": PERIOD_POLICY,
            "tie_break": "RMS, standard6 before classical20, start index"}


def cmd_run(args) -> dict:
    out = args.output_dir
    manifest = load_export(out, args.addendum)
    lock = c.benchmark(args.benchmark_dir / "evaluation-lock.json")
    plan = run_plan(args, manifest, lock)
    plan_path = out / "run.json"
    c.write_or_verify(plan_path, plan)
    with plan_path.open("rb") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another run is active in this output directory") from exc
        receipts = []
        for spec in plan["cases"]:
            row = c.unique(manifest["inputs"], "run_id", spec["run_id"])
            case = c.case_path(out, spec["run_id"], spec["mode"], spec["arm"])
            job = {"object_id": row["object_id"], "period_hours": spec["period_hours"],
                   "lightcurve": c.binding(c.checked(row["lightcurve"], out))}
            case.mkdir(parents=True, exist_ok=True)
            c.write_or_verify(case / "job.json", {**job, **spec, "period_fit_flag": 1, "run_plan": c.binding(plan_path)})
            records = []
            if spec["period_hours"] is not None:
                vectors = grid.starts_for_arm(spec["arm"])[:spec["starts"]]
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                    futures = [pool.submit(execute_start, job, i, v, lock, case) for i, v in enumerate(vectors)]
                    for future in concurrent.futures.as_completed(futures):
                        r = future.result()
                        records.append(r)
                        print(f"{spec['run_id']} {spec['mode']} {spec['arm']} start-{r['start_index']:03d} "
                              f"converged={r['selected_fit'] is not None} fitted_period_hours={r['fitted_period_hours']}", flush=True)
                records.sort(key=lambda r: r["start_index"])
            fits = [r["selected_fit"] for r in records if r["selected_fit"] is not None]
            result = {**spec, "object_id": row["object_id"], "valid": len(fits),
                      "status": "no_period" if spec["period_hours"] is None else "attempted",
                      "selected_fit": min(fits, key=lambda f: (f["final_relative_rms"], f["start_index"])) if fits else None,
                      "solver_wall_seconds": sum(r["solver_wall_seconds"] for r in records),
                      "records": [{"record": c.relative_binding(case / f"start-{r['start_index']:03d}/record.json", out),
                                   "diagnostics": c.relative_binding(case / f"start-{r['start_index']:03d}/diagnostics.json", out)}
                                  for r in records]}
            c.write_or_verify(case / "case.json", result)
            receipts.append(c.relative_binding(case / "case.json", out))
        c.write_or_verify(out / "execution.json", {"schema": "delphi.spacecraft-classical-freep-execution.v1", "posthoc": True,
                          "run_plan": c.binding(plan_path), "cases": receipts, "full_grid": plan["full_grid"],
                          "reference_poles_opened": False, "status": "all_planned_starts_attempted"})
    return {"stage": "run", "cases": len(receipts), "full_grid": plan["full_grid"],
            "workers": args.workers, "period_fit_flag": 1, "reference_poles_opened": False}


def delphi_comparisons(spacecraft: Path, manifest: dict, reference_path: Path, modes: list[str]) -> dict:
    """Score-only: verify ordinary predictions and DAMIT rescore OOF rows."""
    c.check_json(spacecraft / "scores.json")
    document = c.read(spacecraft / "scores.json")
    lock_path = c.network_run.input_root(spacecraft) / "lock/cohort-lock.json"
    lock = c.network_run.locked(spacecraft)[0] if c.switched() else c.read(lock_path)
    prediction_path = spacecraft / "predictions/manifest.json"
    if (document["reference_poles_sha256"] != grid.digest(reference_path)
            or document["cohort_lock_sha256"] != grid.digest(lock_path)
            or document["predictions_manifest_sha256"] != grid.digest(prediction_path)):
        raise ValueError("DeLPHI score/reference/prediction binding differs")
    c.check_json(prediction_path)
    prediction_manifest = c.read(prediction_path)
    predictions = {}
    for mode in modes:
        path = c.checked(prediction_manifest["outputs"][mode], spacecraft / "predictions")
        c.check_jsonl(path)
        predictions[mode] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = {}
    for row in manifest["inputs"]:
        comparison_id = row["run_id"].removesuffix("-damit") + "-oof" if row["run_id"] in DAMIT_RESCORE else row["run_id"]
        comparison = c.unique(document["objects"], "run_id", comparison_id)
        if comparison["object_id"] != row["object_id"]:
            raise ValueError("DeLPHI object identity differs")
        for mode in modes:
            d = comparison[mode]
            if d["period_hours"] != row["periods_hours"][mode]:
                raise ValueError(f"DeLPHI starting period differs: {row['run_id']} {mode}")
            if row["run_id"] in DAMIT_RESCORE:
                # run.py rescores these stored candidates, never the 25-network ensemble.
                with np.load(c.checked(lock["files"][f"oof_{mode}"]), allow_pickle=False) as data:
                    ids = [str(x) for x in data["object_ids"]]
                    if ids.count(row["object_id"]) != 1:
                        raise ValueError(f"{row['name']} missing/repeated in OOF file")
                    i = ids.index(row["object_id"])
                    if float(data["period_hours"][i]) != d["period_hours"]:
                        raise ValueError(f"{row['name']} OOF period differs")
                    axes, scores = np.asarray(data["refined_axes"][i]), np.asarray(data["refined_scores"][i])
                    ref = comparison["spacecraft_reference"]["unit_vector"]
                    errors = [c.angle(a, ref, True) for a in axes]
                    if (not math.isclose(min(errors), d["oracle_deg"], abs_tol=1e-8)
                            or not math.isclose(errors[int(np.argmax(scores))], d["best_scored_deg"], abs_tol=1e-8)):
                        raise ValueError(f"{row['name']} OOF errors differ from {comparison_id} scores")
            else:
                prediction = c.unique(predictions[mode], "run_id", row["run_id"])
                if (prediction["input_sha256"] != row["input"]["sha256"]
                        or prediction["period_hours"] != row["periods_hours"][mode]):
                    raise ValueError("DeLPHI used different observations/period")
            result[row["run_id"], mode] = comparison
    return result


def cmd_score(args) -> dict:
    destination = args.output_dir
    out = getattr(args, "fit_dir", None) or destination
    c.available(destination / "scores.json")
    c.available(destination / "RESULTS.md")
    destination.mkdir(parents=True, exist_ok=True)
    execution = c.read(out / "execution.json")
    plan = c.read(c.checked(execution["run_plan"]))
    if (execution["status"] != "all_planned_starts_attempted"
            or execution["reference_poles_opened"] is not False or plan["reference_poles_opened"] is not False
            or plan["period_fit_flag"] != 1):
        raise ValueError("complete reference-blind free-period execution required")
    if not plan["full_grid"] and not args.allow_partial:
        raise ValueError("partial grids require --allow-partial; only for a labelled smoke test")
    manifest = load_export(out, args.addendum)
    if plan["export"] != c.binding(out / "export/manifest.json") or (not c.switched() and plan["script"] != c.binding(Path(__file__))):
        raise ValueError("execution inputs/script changed")
    if not c.switched():  # a rerun scores with this code; the Addendum 3 fit script was edited later
        c.checked(plan["script"])
    for entry in plan["implementations"].values():
        c.checked(entry)
    c.checked(plan["validation"])
    cases = [c.read(c.checked(e, out)) for e in execution["cases"]]
    if len(cases) != len(plan["cases"]):
        raise ValueError("missing or duplicate cases")
    grouped = {}
    for spec, case in zip(plan["cases"], cases, strict=True):
        if any(case[k] != v for k, v in spec.items()):
            raise ValueError("case differs from run plan")
        if len(case["records"]) != (0 if spec["period_hours"] is None else spec["starts"]):
            raise ValueError("not every planned start has a record")
        row = c.unique(manifest["inputs"], "run_id", spec["run_id"])
        job = {"period_hours": spec["period_hours"], "lightcurve": c.binding(c.checked(row["lightcurve"], out))}
        fits = grouped.setdefault((spec["run_id"], spec["mode"]), [])
        vectors = grid.starts_for_arm(spec["arm"])
        for i, entry in enumerate(case["records"]):
            record_path = c.checked(entry["record"], out)
            diagnostics_path = c.checked(entry["diagnostics"], out)
            record = verify_start(job, i, vectors[i], record_path.parent)
            if c.read(diagnostics_path) != diagnostics(record, record_path.parent):
                raise ValueError("start diagnostics changed")
            if record["selected_fit"] is not None:
                fits.append({**record["selected_fit"], "arm": spec["arm"],
                             **pole_degrees(record["selected_fit"]["axis"]),
                             "chi2": c.read(diagnostics_path)["chi2"], "record": entry["record"]})
    # These are the first reference-pole / score reads in this workflow.
    reference_path = c.network_run.input_root(args.spacecraft_dir) / "reference-poles.json"
    reference = c.read(reference_path)
    if reference["schema"] != "delphi.spacecraft-reference-poles.v1" or reference["ecliptic_frame"] != "ECLIPJ2000":
        raise ValueError("unexpected reference schema/frame")
    refs = {r["object"].lower(): r for r in reference["objects"]}
    comparisons = delphi_comparisons(args.spacecraft_dir, manifest, reference_path, plan["periods"])
    objects = []
    for (run_id, mode), fits in grouped.items():
        source = c.unique(manifest["inputs"], "run_id", run_id)
        comparison = comparisons[run_id, mode]
        ref = refs[source["name"]]
        if comparison["spacecraft_reference"] != ref:
            raise ValueError("DeLPHI reference row differs")
        counts = [r for r in cases if r["run_id"] == run_id and r["mode"] == mode]
        best = min(fits, key=c.fit_key) if fits else None
        row = {"run_id": run_id, "mode": mode, "object_id": source["object_id"],
               "period_hours": source["periods_hours"][mode],
               "fitted_period_hours": best["fitted_period_hours"] if best else None,
               "status": "no_period" if source["periods_hours"][mode] is None else "ok" if fits else "no_valid_fit",
               "attempted_starts": sum(len(r["records"]) for r in counts), "valid_starts": len(fits),
               "requested_starts": sum(r["starts"] for r in counts),
               "selected_axial_error_deg": c.angle(best["axis"], ref["unit_vector"], True) if best else 90.0,
               "spin_vector": c.outcomes(fits, ref["unit_vector"], False),
               "axial": c.outcomes(fits, ref["unit_vector"], True),
               "delphi_run_id": comparison["run_id"],
               "delphi_axial": {k: comparison[mode][k] for k in ("status", "period_hours", "oracle_deg", "best_scored_deg")},
               "reference": ref}
        objects.append(row)
        for metric in ("spin_vector", "axial"):
            representatives = row[metric]["b_ambiguity"]["representatives"]
            print(f"{run_id} {mode} {metric} near-best (lambda_deg, beta_deg, RMS/best): " +
                  json.dumps([[r["lambda_deg"], r["beta_deg"], r["rms_relative_to_best"]]
                              for r in representatives], allow_nan=False), flush=True)
    label = f"POST HOC: Addendum {args.addendum}" + ("" if plan["full_grid"] else " SMOKE TEST, PARTIAL GRID")
    score_paths = [args.spacecraft_dir / "scores.json"]
    oof = (c.network_run.locked(args.spacecraft_dir)[0] if c.switched() else c.read(args.spacecraft_dir / "lock/cohort-lock.json"))["files"]
    result = {"schema": "delphi.spacecraft-classical-freep-scores.v1", "scope": label, "posthoc": True,
              "execution": c.binding(out / "execution.json"), "reference_poles": c.binding(reference_path),
              "delphi_scores": [c.binding(p) for p in score_paths],
              "ceres_vesta_oof" if args.addendum == 4 else "eros_oof": {mode: oof[f"oof_{mode}"] for mode in plan["periods"]},
              "full_grid": plan["full_grid"], "period_policy": PERIOD_POLICY,
              "comparable_metric": "axial angle in degrees (absolute spin-vector dot product)",
              "ambiguity_rule": "RMS <= 1.05 best; ascending RMS/arm/index greedy representatives >20 deg apart; oracle over representatives",
              "c_rule": "lowest RMS among valid final poles within 20 deg of reference; initial pole irrelevant",
              "failure_policy": "no period or valid fit: selected axial error 90 deg, signed angle undefined",
              "aggregation": "none; each input and starting period listed", "objects": objects}
    lines = [f"# Spacecraft classical inversion with fitted period — {label}", "",
             "Six standard poles plus the 146-pole 20-degree grid; each supplied period is a starting value."
             if plan["full_grid"] else "SMOKE TEST ONLY: incomplete grid; these are software checks.",
             "Period-fit flag 1: local angular-frequency refinement; no period scan, alias search or YORP term.",
             "Selected fitted periods come from solution-parameters.txt (8 decimal places in hours).",
             "Valid: exit 0, finite products, positive fitted period, 0 < iterations < 1000, no timeout (300 s/start).",
             "Tolerance 0.0003; all other settings/selection match Addendum 2. Signed poles are also scored axially for DeLPHI.",
             "Ambiguity: RMS within 5% of best; greedy RMS-ordered representatives more than 20 degrees apart.",
             "Outcome (c): lowest-RMS final pole within 20 degrees of reference; n/a means none.",
             "Failures count as 90 axial degrees; signed error is undefined. " +
             ("Ceres and Vesta use ceres-oof and vesta-oof DeLPHI scores." if args.addendum == 4
              else "Eros uses eros-oof DeLPHI scores."), ""]
    def fmt(value):
        return "n/a" if value is None else f"{value:.8f}"
    for metric, title in (("axial", "Axial angles (comparable)"), ("spin_vector", "Signed spin-vector angles")):
        lines += [f"## {title}", "",
                  "| Input | Starting period | Initial P (h) | Selected fitted P (h) | Valid/attempted | (a) error | (b) minima | (b) oracle error | (c) error | (c) RMS/best | DeLPHI axial oracle | DeLPHI axial best |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in objects:
            m, d = row[metric], row["delphi_axial"]
            a, b, near = m["a_selected"], m["b_ambiguity"], m["c_near_reference"]
            cells = [row["run_id"], row["mode"], fmt(row["period_hours"]), fmt(row["fitted_period_hours"]),
                     f"{row['valid_starts']}/{row['attempted_starts']}",
                     fmt(a["error_deg"] if a else 90.0 if metric == "axial" else None), str(b["distinct_minima"]),
                     fmt(b["oracle"]["error_deg"] if b["oracle"] else None),
                     fmt(near["error_deg"] if near else None), fmt(near["rms_relative_to_best"] if near else None),
                     fmt(d["oracle_deg"]), fmt(d["best_scored_deg"])]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        lines += ["Near-best minima representatives (longitude, latitude in degrees; RMS/best):", ""]
        for row in objects:
            reps = row[metric]["b_ambiguity"]["representatives"]
            values = "; ".join(f"({r['lambda_deg']:.6f}, {r['beta_deg']:.6f}, {r['rms_relative_to_best']:.8f})" for r in reps)
            lines.append(f"- {row['run_id']} {row['mode']}: {values or 'none'}")
        lines.append("")
    c.write(destination / "scores.json", result)
    with (destination / "RESULTS.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return {"stage": "score", "input_period_pairs": len(objects), "scope": label}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("export", cmd_export), ("validate", cmd_validate), ("run", cmd_run), ("score", cmd_score)):
        p = commands.add_parser(name)
        p.add_argument("--addendum", type=int, choices=(3, 4), default=3)
        p.add_argument("--output-dir", type=Path)
        p.add_argument("--spacecraft-dir", type=Path, default=c.SPACECRAFT)
        p.add_argument("--benchmark-dir", type=Path, default=c.BENCHMARK)
        p.set_defaults(handler=handler)
        if name == "run":
            p.add_argument("--workers", type=int, default=grid.SETTINGS["workers"])
            p.add_argument("--runs", nargs="+", choices=RUNS + ADDENDUM4_RUNS)
            p.add_argument("--periods", nargs="+", choices=c.MODES, default=list(c.MODES))
            p.add_argument("--max-starts", type=int, help="SMOKE TEST: first N of 152 starts per input/period")
        if name == "score":
            p.add_argument("--fit-dir", type=Path, help="read-only published standard-start fits")
            p.add_argument("--allow-partial", action="store_true")
        if name == "validate":
            p.add_argument("--classical-dir", type=Path, help="default: --spacecraft-dir/classical (read only)")
            p.add_argument("--run-id", choices=c.RUNS, default="gaspra-damit")
            p.add_argument("--period", choices=c.MODES, default="own")
            p.add_argument("--arm", choices=c.ARMS, default="standard6")
            p.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args(argv)
    if c.switched() and (args.command != "score" or args.fit_dir is None):
        parser.error("switched classical runs score --fit-dir only; reuse the published standard-start fits")
    if args.output_dir is None:
        args.output_dir = c.SPACECRAFT / "classical-freep-ceres-vesta" if args.addendum == 4 else OUTPUT
    for key in ("output_dir", "spacecraft_dir", "benchmark_dir", "classical_dir"):
        if getattr(args, key, None) is not None:
            setattr(args, key, getattr(args, key).resolve())
    if args.command == "run":
        selected_runs = ADDENDUM4_RUNS if args.addendum == 4 else RUNS
        if args.runs is None:
            args.runs = list(selected_runs)
        if any(run_id not in selected_runs for run_id in args.runs):
            parser.error(f"--runs must select Addendum {args.addendum} inputs")
        if args.workers < 1 or (args.max_starts is not None and not 1 <= args.max_starts <= 152):
            parser.error("workers must be positive; max-starts must be in 1..152")
        if len(set(args.runs)) != len(args.runs) or len(set(args.periods)) != len(args.periods):
            parser.error("duplicate run/period selections")
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    try:
        result = args.handler(args)
    except (ValueError, OSError, KeyError, IndexError) as error:
        parser.exit(1, f"classical_freep {args.command}: {error}\n")
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    result.update(wall_seconds=time.perf_counter() - started,
                  solver_cpu_seconds=after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
