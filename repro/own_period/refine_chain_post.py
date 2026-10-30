"""Post-refinement stages of the fixed 20-object development protocol.

Run, in order, ``python -m repro.own_period.refine_chain_post`` with ``lock``,
``predict``, ``fits``, then ``score``. Wrap predict in runners/lock_run.sh.
Only score opens references. Prediction uses predict_oof's numerical path,
with photometry-only loading and a fixed, unused placeholder target. Fits use the
frozen benchmark's native worker unchanged; a separate receipt measures its
child solver CPU time. Every output is exclusively created and validated on
resume. An interrupted start without both receipts needs manual inspection.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import fcntl
import hashlib
import importlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import resource
import shlex
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
from repro.own_period import DataPath
from repro import network_root as network

DATA = DataPath()
OUTPUT = network.NetworkPath("results", "refine-chain", legacy_path=DATA / "refine-chain")
BENCHMARK = network.NetworkPath("results", "delphi-inversions", "own", legacy_path=DATA / "benchmark")
PUBLISHED = network.NetworkPath("results", "delphi-inversions", "damit", legacy_path=DataPath("../direct-workflow-benchmark"))
PROVENANCE = "own-fourier-gpu-top1+narrow-refine"
ARMS = ("delphi6", "classical12", "classical146")
SAVED_ARMS = ("delphi", "standard6", "classical20")
RESAMPLES = 10_000
SEED = 20261026
E_TO_P = 1.406


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def binding(path: Path) -> dict:
    return {"path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def verify(value: dict) -> Path:
    path = Path(value["path"])
    if binding(path) != value:
        raise ValueError(f"bound file changed: {path}")
    return path


def save_outputs(outputs: dict[Path, object]) -> None:
    """Preflight every existing sibling before creating any missing output."""
    for path, value in outputs.items():
        network.guard_output(path)
        if path.exists():
            actual = path.read_text() if isinstance(value, str) else read(path)
            if actual != value:
                raise ValueError(f"existing output mismatches current inputs: {path}")
    for path, value in outputs.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as stream:
                stream.write(value if isinstance(value, str) else
                             json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def _validate_runtime(lock: dict) -> None:
    """Use the frozen runtime, without its obsolete hardware/timing check."""
    root = Path(lock["code_root"]).resolve()
    sys.path.insert(0, str(root))
    for relative, expected in lock["runtime_sources"].items():
        if binding(root / relative)["sha256"] != expected:
            raise ValueError(f"frozen runtime changed: {relative}")
    for name, module in tuple(sys.modules.items()):
        if name.startswith("lc_pipeline") and getattr(module, "__file__", None):
            path = Path(module.__file__)
            relative = "lc_pipeline/" + path.as_posix().split("/lc_pipeline/", 1)[-1]
            expected = lock["runtime_sources"].get(relative)
            if expected is not None and binding(path)["sha256"] != expected:
                raise ValueError(f"already imported runtime differs from lock: {name}")
    verify(lock["solver"]["executable"])
    verify(lock["script"])
    for row in lock["objects"]:
        verify(row["lightcurve"])


def refined_inputs(args) -> tuple[dict, dict, list[dict]]:
    network.check_json(args.benchmark_dir / "evaluation-lock.json")
    parent = read(args.benchmark_dir / "evaluation-lock.json")
    # Import only after choosing the frozen runtime in commands that use it.
    from repro.own_period.refine_chain import protocol_objects

    selected = protocol_objects(parent)
    refined = read(DataPath("refine-chain/refined-periods.json") if network.switched()
                   else args.output_dir / "refined-periods.json")
    rows = refined["objects"]
    if (len(rows) != len(selected)
            or {r["object_id"] for r in rows} != {r["object_id"] for r in selected}):
        raise ValueError("refined periods must contain exactly the 20 protocol objects")
    by_id = {r["object_id"]: r for r in rows}
    expected_periods = {}
    for original in selected:
        row = by_id[original["object_id"]]
        if row["fold"] != original["fold"]:
            raise ValueError("refinement fold changed")
        if row["status"] == "complete":
            period = row["period_found_hours"]
            if (not isinstance(period, (int, float)) or not math.isfinite(period)
                    or period <= 0 or period != round(period, 6)
                    or round(row["period_found_hours_raw"], 6) != period):
                raise ValueError("invalid six-decimal refined period")
            expected_periods[row["object_id"]] = period
        elif row["status"] != "not_finished":
            raise ValueError("unknown refinement status")
    if refined["periods_hours"] != expected_periods:
        raise ValueError("refinement rows disagree with periods_hours")
    return parent, refined, selected


def lock_outputs(args) -> dict[Path, dict]:
    parent, refined, selected = refined_inputs(args)
    periods = refined["periods_hours"]
    objects = [{**row, "period_hours": periods[row["object_id"]]}
               for row in selected if row["object_id"] in periods]
    failed = [row["object_id"] for row in selected if row["object_id"] not in periods]
    provenance = {"provenance": PROVENANCE,
                  "source_lock": binding(args.benchmark_dir / "evaluation-lock.json"),
                  "refined_periods": binding(DataPath("refine-chain/refined-periods.json") if network.switched()
                                            else args.output_dir / "refined-periods.json"),
                  "protocol": binding(args.protocol), "failed_object_ids": failed}
    lock = {**deepcopy(parent), "objects": objects, "refinement": provenance,
            "period_provenance": PROVENANCE}
    if network.switched():
        from repro.direct_workflow_benchmark.rerun import timed_workers
        worker_plan = DataPath("refine-chain/fits/fit-plan.json")
        original_workers = read(worker_plan)["workers"]
        lock["rerun"].update(
            solver_workers=timed_workers(original_workers, parent["rerun"]["classical_timing"]),
            old_solver_workers=original_workers, source_worker_plan=binding(worker_plan))
    period_file = {"periods_hours": {r["object_id"]: periods.get(r["object_id"])
                                     for r in selected}, **provenance}
    return {args.output_dir / "evaluation-lock-refined.json": lock,
            args.output_dir / "periods-refined.json": period_file}


def checked_lock(args) -> dict:
    outputs = lock_outputs(args)
    for path, expected in outputs.items():
        if read(path) != expected:
            raise ValueError(f"refined lock or periods file changed: {path}")
    return outputs[args.output_dir / "evaluation-lock-refined.json"]


def command_lock(args) -> int:
    _validate_runtime(read(args.benchmark_dir / "evaluation-lock.json"))
    save_outputs(lock_outputs(args))
    return 0


def require_gpu_lock() -> None:
    """Check the inherited fd and an independent conflicting flock, not a flag."""
    lock = os.environ.get("LOCK")
    if not lock:
        return
    path = Path(lock)
    try:
        inherited, actual = os.fstat(9), path.stat()
        if (inherited.st_dev, inherited.st_ino) != (actual.st_dev, actual.st_ino):
            raise ValueError("fd 9 does not refer to the shared GPU lock")
        with path.open("rb") as probe:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            fcntl.flock(probe, fcntl.LOCK_UN)
    except (OSError, ValueError):
        pass
    command = [str(DATA / "runners/lock_run.sh"), "--", sys.executable,
               "-m", "repro.own_period.refine_chain_post", *sys.argv[1:]]
    raise RuntimeError("predict requires runners/lock_run.sh: " + shlex.join(command))


def prediction_request(args, lock) -> dict:
    return {**network.json_metadata(), "lock": binding(args.output_dir / "evaluation-lock-refined.json"),
            "periods": binding(args.output_dir / "periods-refined.json"),
            "script": binding(Path(__file__)),
            "predict_oof": binding(REPO / "repro/own_period/predict_oof.py"),
            "model_manifest": binding(network.NetworkPath("binding") if network.switched()
                                      else DATA.parent / "external-models-final.json"),
            "device": args.device, "objects": lock["objects"]}


def validate_prediction(row: dict, request: dict, metadata: dict) -> None:
    if (row.get("request") != request or row.get("object_id") != metadata["object_id"]
            or row.get("fold") != metadata["fold"]
            or row.get("period_hours") != metadata["period_hours"]):
        raise ValueError("prediction belongs to different inputs")
    axes = np.asarray(row["refined_axes"], dtype=float)
    initial = np.asarray(row["initial_axes"], dtype=float)
    scores = np.asarray(row["refined_scores"], dtype=float)
    if (axes.shape != (3, 3) or initial.shape != (3, 3) or scores.shape != (3,)
            or not all(np.isfinite(v).all() for v in (axes, initial, scores))
            or not np.allclose(np.linalg.norm(axes, axis=1), 1, atol=1e-5)
            or not math.isfinite(row["inference_seconds"]) or row["inference_seconds"] < 0):
        raise ValueError("invalid prediction axes, scores or timing")


def predict_one(path, row, models, device, request) -> dict:
    from repro.own_period import predict_oof as p
    from lc_pipeline.k3.damit import _read_lc

    started = time.perf_counter()
    example = p.K3TrainingExample(
        object_id=row["object_id"],
        tokenized=p.tokenize_epochs(_read_lc(Path(row["lightcurve"]["path"])),
                                    known_period=p.KnownPeriod(row["period_hours"], PROVENANCE)),
        # TrainingExample requires one target; this constant is discarded by
        # INPUT_KEYS and cannot affect tokenization, score maps or adjustment.
        target_axes=np.asarray([[1, 0, 0]], dtype=np.float32),
    )
    inputs = {k: v.to(device) for k, v in p.collate_examples((example,)).items()
              if k in p.INPUT_KEYS}
    grids = np.stack([np.asarray(p.score_axial_grid(m, inputs, chunk_size=1024), dtype=np.float64)
                      for m in models])
    modes = p.modes_from_score_grid(p.ensemble_score_grids(grids[:, None, :])[0])
    starts = np.asarray([mode.axis_xyz for mode in modes], dtype=np.float64)
    axes, scores = p.refine_ensemble_axes(models, inputs, starts)
    if device.startswith("cuda"):
        import torch
        torch.cuda.synchronize(device)
    result = {"request": request, "object_id": row["object_id"], "fold": row["fold"],
              "period_hours": row["period_hours"], "initial_axes": starts.tolist(),
              "refined_axes": np.asarray(axes).tolist(),
              "refined_scores": np.asarray(scores).tolist(),
              "inference_seconds": time.perf_counter() - started}
    validate_prediction(result, request, row)
    save_outputs({path: result})
    return result


def command_predict(args) -> int:
    require_gpu_lock()
    _validate_runtime(read(args.benchmark_dir / "evaluation-lock.json"))
    lock = checked_lock(args)
    request = prediction_request(args, lock)
    directory = args.output_dir / "predictions-refined"
    plan = directory / "prediction-plan.json"
    if not plan.exists() and directory.exists():
        raise FileExistsError("prediction directory exists without its plan")
    pending, paths = [], []
    for row in lock["objects"]:
        path = directory / row["object_id"] / "prediction.json"
        paths.append(path)
        if path.exists():
            validate_prediction(read(path), request, row)
        elif path.parent.exists():
            raise FileExistsError(f"incomplete prediction directory: {path.parent}")
        else:
            pending.append((path, row))
    if pending and any((args.output_dir / name).exists() for name in ("scores.json", "RESULTS.md")):
        raise RuntimeError("cannot predict after scoring")
    if pending and (directory / "predictions.json").exists():
        raise RuntimeError("completed predictions have missing records; inspect manually")
    save_outputs({plan: request})
    if pending:
        from repro.own_period import predict_oof as p
        for fold in sorted({row["fold"] for _, row in pending}):
            models, _, _ = p._load_models(p.MODELS, p.MODEL_MANIFEST, (fold,), args.device)
            for path, row in pending:
                if row["fold"] == fold:
                    path.parent.mkdir(parents=True, exist_ok=False)
                    predict_one(path, row, models, args.device, request)
                    print(f"predicted {row['object_id']}", flush=True)
    save_outputs({directory / "predictions.json": {
        "request": request, "records": [binding(path) for path in paths],
        "failed_object_ids": lock["refinement"]["failed_object_ids"]}})
    return 0


def load_predictions(args, lock) -> dict[str, dict]:
    directory = args.output_dir / "predictions-refined"
    manifest = read(directory / "predictions.json")
    request = read(directory / "prediction-plan.json")
    if network.switched() and any(request.get(k) != v for k, v in network.json_metadata().items()):
        raise ValueError("refined predictions belong to different networks")
    if (manifest["request"] != request
            or request["lock"] != binding(args.output_dir / "evaluation-lock-refined.json")
            or request["periods"] != binding(args.output_dir / "periods-refined.json")
            or request["objects"] != lock["objects"]
            or manifest["failed_object_ids"] != lock["refinement"]["failed_object_ids"]):
        raise ValueError("prediction manifest differs from refined lock")
    for key in ("script", "predict_oof", "model_manifest"):
        verify(request[key])
    expected = [directory / row["object_id"] / "prediction.json" for row in lock["objects"]]
    if manifest["records"] != [binding(path) for path in expected]:
        raise ValueError("prediction records changed")
    predictions = {}
    for path, row in zip(expected, lock["objects"]):
        value = read(path)
        validate_prediction(value, request, row)
        predictions[row["object_id"]] = value
    return predictions


def ladder_indexes(benchmark_dir: Path) -> list[int]:
    """Use the original farthest-first ladder, including its duplicate rule."""
    from lc_pipeline import grid_benchmark as grid

    spec = importlib.util.spec_from_file_location("_refine_direct", benchmark_dir / "run_direct_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    starts = module.classical_starts()
    broad = grid.pole_grid(20)
    indexes = []
    for vector in starts[6:]:
        matches = np.where(np.linalg.norm(broad - vector, axis=1) < 1e-10)[0]
        if len(matches) != 1:
            raise ValueError("classical ladder start absent from grid")
        indexes.append(int(matches[0]))
    if len(starts) != 146 or len(indexes) != 140:
        raise ValueError("published ladder changed")
    return indexes


def fit_tasks(args, lock, predictions) -> list[dict]:
    from lc_pipeline import grid_benchmark as grid, workflow_benchmark as prior

    common = {"lock": binding(args.output_dir / "evaluation-lock-refined.json"),
              "predictions": binding(args.output_dir / "predictions-refined/predictions.json"),
              "script": binding(Path(__file__))}
    tasks = []
    for row in lock["objects"]:
        starts = {"delphi": prior.delphi_starts(predictions[row["object_id"]]["refined_axes"]),
                  "standard6": grid.starts_for_arm("standard6"),
                  "classical20": grid.starts_for_arm("classical20")}
        for arm, vectors in starts.items():
            for index, vector in enumerate(vectors):
                tasks.append({**common, "job": row, "arm": arm, "start_index": index,
                              "initial_axis": vector.tolist(),
                              "directory": str((args.output_dir / "fits" / arm / row["object_id"]).resolve())})
                if network.switched() and arm != "delphi":
                    source = Path(DataPath("refine-chain/fits")) / arm / row["object_id"]
                    tasks[-1]["classical_source"] = binding(source / f"start-{index:03d}" / "timing.json")
                    if getattr(args, "classical_timing", "reuse") == "reuse":
                        tasks[-1]["directory"] = str(source.resolve())
    return tasks


def _start_path(task) -> Path:
    return Path(task["directory"]) / f"start-{task['start_index']:03d}"


def validate_start(task: dict) -> dict:
    directory = _start_path(task)
    if not (directory / "record.json").is_file() or not (directory / "timing.json").is_file():
        raise FileExistsError(f"incomplete start; inspect manually: {directory}")
    timing = read(directory / "timing.json")
    if network.switched() and "classical_source" in task:
        from repro.direct_workflow_benchmark.rerun import audit_start, equal_start
        source_path = verify(task["classical_source"])
        old = read(source_path)
        verify(old["record"])
        verify(old["task"]["lock"])
        if (old["task"]["job"] != task["job"] or old["task"]["initial_axis"] != task["initial_axis"]
                or old["task"]["start_index"] != task["start_index"]):
            raise ValueError("reused classical refined inputs differ")
        saved = audit_start(source_path.parent / "record.json")
        if directory.resolve() == source_path.parent.resolve():
            return {**saved, "solver_cpu_seconds": old["solver_cpu_seconds"]}
        equal_start(audit_start(directory / "record.json"), saved)
    if timing.get("task") != task or timing["record"] != binding(directory / "record.json"):
        raise ValueError("saved start belongs to different inputs or its record changed")
    record = read(directory / "record.json")
    if (record["start_index"] != task["start_index"]
            or record["initial_axis"] != task["initial_axis"]
            or not math.isfinite(timing["solver_cpu_seconds"]) or timing["solver_cpu_seconds"] < 0
            or timing["nice"] != 19 or len(timing["cpu_list"]) != 1):
        raise ValueError("invalid saved start or timing")
    for name, digest in record["files"].items():
        if Path(name).name != name or binding(directory / name)["sha256"] != digest:
            raise ValueError("saved solver product changed")
    return {**record, "solver_cpu_seconds": timing["solver_cpu_seconds"]}


def _init_worker(queue, parent_path=None) -> None:
    if parent_path is not None:
        sys.path[:] = parent_path
    os.sched_setaffinity(0, {queue.get()})
    os.nice(19 - os.getpriority(os.PRIO_PROCESS, 0))


def _fit_start(task, lock) -> dict:
    from lc_pipeline import grid_benchmark as grid
    from lc_pipeline.v2.convexinv import _parse_lightcurve_brightness

    observed = _parse_lightcurve_brightness(Path(task["job"]["lightcurve"]["path"]))
    directory = Path(task["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    grid._native_start(task["job"], task["start_index"], np.asarray(task["initial_axis"]),
                       observed, lock["solver"], lock["solver_settings"], directory)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    path = _start_path(task)
    save_outputs({path / "timing.json": {
        "task": task, "record": binding(path / "record.json"),
        "solver_cpu_seconds": after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
        "cpu_list": sorted(os.sched_getaffinity(0)), "nice": os.getpriority(os.PRIO_PROCESS, 0)}})
    return validate_start(task)


def _fit_completion(tasks, plan_path, failed) -> dict:
    for task in tasks:
        validate_start(task)
    return {"plan": binding(plan_path), "failed_object_ids": failed,
            "starts": [{"record": binding(_start_path(t) / "record.json"),
                        "timing": binding(_start_path(t) / "timing.json")} for t in tasks]}


def command_fits(args) -> int:
    _validate_runtime(read(args.benchmark_dir / "evaluation-lock.json"))
    lock = checked_lock(args)
    predictions = load_predictions(args, lock)
    if (not args.cpus or len(set(args.cpus)) != len(args.cpus)
            or not set(args.cpus) <= os.sched_getaffinity(0)):
        raise ValueError("CPUs must be unique and available in the current affinity")
    workers = len(args.cpus) if args.workers is None else args.workers
    if network.switched():
        if args.classical_timing != lock["rerun"]["classical_timing"]:
            raise ValueError("classical timing differs from the refined lock")
        if args.classical_timing == "reuse":
            workers = lock["rerun"]["solver_workers"]
            if args.workers is not None and args.workers != workers:
                raise ValueError("solver workers differ from the original refine plan")
        else:
            workers = min(workers, 4)
    if not 1 <= workers <= len(args.cpus):
        raise ValueError("workers must be between one and the number of CPUs")
    tasks = fit_tasks(args, lock, predictions)
    directory = args.output_dir / "fits"
    plan_path = directory / "fit-plan.json"
    plan = {"tasks": tasks, "cpus": list(args.cpus), "workers": workers, "nice": 19,
            "ladder_indexes": ladder_indexes(args.benchmark_dir)}
    if network.switched():
        plan.update(classical_timing=args.classical_timing, **network.json_metadata())
    if not plan_path.exists() and directory.exists():
        raise FileExistsError("fits directory exists without its plan")
    pending = []
    for task in tasks:
        if _start_path(task).exists():
            validate_start(task)
            if read(_start_path(task) / "timing.json")["cpu_list"][0] not in args.cpus:
                raise ValueError("saved start ran outside the planned CPUs")
        else:
            pending.append(task)
    if pending and any((args.output_dir / n).exists() for n in ("scores.json", "RESULTS.md", "fits/complete.json")):
        raise RuntimeError("cannot fit after completion or scoring")
    save_outputs({plan_path: plan})
    if pending:
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        for cpu in args.cpus:
            queue.put(cpu)
        parent_path = sys.path[:]
        with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                 initializer=_init_worker, initargs=(queue, parent_path)) as pool:
            try:
                # Spawn must import this module before the frozen repro package.
                sys.path.insert(0, str(REPO))
                futures = [pool.submit(_fit_start, task, lock) for task in pending]
            finally:
                sys.path[:] = parent_path
            for number, future in enumerate(as_completed(futures), 1):
                future.result()
                if number % 50 == 0 or number == len(futures):
                    print(f"fits {number}/{len(futures)} pending starts", flush=True)
        queue.close()
        queue.join_thread()
    save_outputs({directory / "complete.json": _fit_completion(
        tasks, plan_path, lock["refinement"]["failed_object_ids"])})
    return 0


def records_by_arm(args, lock, predictions) -> dict:
    """Audit every expected fit before score is allowed to read references."""
    tasks = fit_tasks(args, lock, predictions)
    plan_path = args.output_dir / "fits/fit-plan.json"
    plan = read(plan_path)
    if plan["tasks"] != tasks or plan["ladder_indexes"] != ladder_indexes(args.benchmark_dir):
        raise ValueError("fit plan changed")
    if network.switched():
        if (plan["classical_timing"] != lock["rerun"]["classical_timing"]
                or (plan["classical_timing"] == "reuse" and plan["workers"] != lock["rerun"]["solver_workers"])
                or (plan["classical_timing"] == "retime" and not 1 <= plan["workers"] <= 4)):
            raise ValueError("fit plan solver workers changed")
    expected = _fit_completion(tasks, plan_path, lock["refinement"]["failed_object_ids"])
    if read(args.output_dir / "fits/complete.json") != expected:
        raise ValueError("fits are incomplete or changed; references remain closed")
    records = {row["object_id"]: {arm: [] for arm in SAVED_ARMS} for row in lock["objects"]}
    for task in tasks:
        records[task["job"]["object_id"]][task["arm"]].append(validate_start(task))
    return records


def arm_records(records: dict, indexes: list[int]) -> dict:
    if any(len(records[arm]) != count for arm, count in zip(SAVED_ARMS, (6, 6, 146))):
        raise ValueError("every saved start is required before scoring")
    ladder = records["standard6"] + [records["classical20"][i] for i in indexes]
    return {"delphi6": records["delphi"], "classical12": ladder[:12], "classical146": ladder[:146]}


def selected_fit(records) -> dict | None:
    from lc_pipeline import workflow_benchmark as prior

    fits = prior._successful_fits(records, "initial")
    return min(fits, key=prior._fit_key) if fits else None


def fit_outcome(fit, references) -> dict:
    from lc_pipeline import workflow_benchmark as prior

    return {"failed": fit is None, "selected_fit": fit,
            "error_deg": 90.0 if fit is None else prior._reference_error(fit["axis"], references)}


def describe(outcomes: list[dict]) -> dict:
    def stats(values):
        errors = [v["error_deg"] for v in values]
        return {"n": len(errors), "mean_deg": float(np.mean(errors)) if errors else None,
                "median_deg": float(np.median(errors)) if errors else None,
                "within20": sum(e <= 20 for e in errors)}
    return {"failures_at_90": stats(outcomes),
            "failures_excluded": stats([v for v in outcomes if not v["failed"]]),
            "failed": sum(v["failed"] for v in outcomes)}


def paired_difference(left: list[dict], right: list[dict], exclude=False) -> dict:
    pairs = [(a, b) for a, b in zip(left, right)
             if not exclude or not (a["failed"] or b["failed"])]
    if len(left) != len(right):
        raise ValueError("paired outcomes have different lengths")
    if not pairs:
        return {"n": 0, "mean_error_difference_deg": None, "median_error_difference_deg": None,
                "within20_fraction_difference": None, "bootstrap_95": None}
    a, b = (np.asarray([p[i]["error_deg"] for p in pairs]) for i in (0, 1))
    delta = a - b
    within = (a <= 20).astype(float) - (b <= 20).astype(float)
    draws = np.random.default_rng(SEED).integers(len(pairs), size=(RESAMPLES, len(pairs)))
    return {"n": len(pairs), "mean_error_difference_deg": float(delta.mean()),
            "median_error_difference_deg": float(np.median(a) - np.median(b)),
            "within20_fraction_difference": float(within.mean()),
            "bootstrap_95": {
                "mean_error_difference_deg": np.percentile(delta[draws].mean(axis=1), [2.5, 97.5]).tolist(),
                "median_error_difference_deg": np.percentile(
                    np.median(a[draws], axis=1) - np.median(b[draws], axis=1), [2.5, 97.5]).tolist(),
                "within20_fraction_difference": np.percentile(within[draws].mean(axis=1), [2.5, 97.5]).tolist()}}


def baseline_fits(directory, object_ids) -> tuple[dict, list[dict]]:
    """Same published/own benchmark results used by budget_equivalence.py."""
    direct_path = directory / "score-loaded.json"
    ladder_path = directory / "classical-ladder-exploratory.json"
    direct = {r["object_id"]: r for r in read(direct_path)["object_rows"]}
    ladder = {r["object_id"]: r for r in read(ladder_path)["object_rows"]}
    result = {}
    for oid in object_ids:
        result[oid] = {"delphi6": direct[oid]["arms"]["delphi"],
                       "classical12": ladder[oid]["classical"]["12"],
                       "classical146": ladder[oid]["classical"]["146"]}
    return result, [binding(direct_path), binding(ladder_path)]


def period_outcome(period, reference):
    error = abs(period / reference - 1) if period is not None else None
    return {"period_hours": period, "relative_error": error,
            "within1_percent": error is not None and error <= 0.01 + 1e-12}


def make_scores(selected, refined, own, references, records, predictions, baselines, indexes, estimates):
    refinement = {r["object_id"]: r for r in refined["objects"]}
    rows = []
    for meta in selected:
        oid = meta["object_id"]
        reference = references[oid]
        r = refinement[oid]
        arms = arm_records(records[oid], indexes) if r["status"] == "complete" else None
        scenarios = {"refined": {
            arm: fit_outcome(selected_fit(arms[arm]) if arms else None, reference["axes"])
            for arm in ARMS}}
        for scenario, fits in baselines.items():
            scenarios[scenario] = {arm: fit_outcome(fits[oid][arm], reference["axes"]) for arm in ARMS}
        costs = {"refinement_E_core_cpu_seconds": r["cpu_seconds"],
                 "refinement_P_core_cpu_seconds": r["cpu_seconds"] / E_TO_P,
                 "inference_seconds": predictions.get(oid, {}).get("inference_seconds"),
                 "solver_cpu_seconds": {arm: sum(v["solver_cpu_seconds"] for v in arms[arm]) if arms else 0.0
                                        for arm in ARMS},
                 "all_152_classical_solver_cpu_seconds": sum(v["solver_cpu_seconds"]
                     for arm in ("standard6", "classical20") for v in records.get(oid, {}).get(arm, [])),
                 "full_range_scan_estimate": estimates.get(oid)}
        rows.append({"object_id": oid, "fold": meta["fold"], "refinement_status": r["status"],
                     "periods": {"DAMIT_hours": reference["period_hours"],
                                 "own": period_outcome(own[oid], reference["period_hours"]),
                                 "refined": period_outcome(refined["periods_hours"].get(oid), reference["period_hours"])},
                     "scenarios": scenarios, "costs": costs})
    scenarios = ("refined", *baselines)
    summary = {s: {a: describe([r["scenarios"][s][a] for r in rows]) for a in ARMS} for s in scenarios}
    pairs = {}
    comparisons = [(s, a, s, b) for s in scenarios
                   for a, b in (("classical12", "delphi6"), ("classical146", "delphi6"),
                                ("classical146", "classical12"))]
    comparisons += [("refined", a, s, a) for s in baselines for a in ARMS]
    for ls, la, rs, ra in comparisons:
        left = [r["scenarios"][ls][la] for r in rows]
        right = [r["scenarios"][rs][ra] for r in rows]
        pairs[f"{ls}/{la} minus {rs}/{ra}"] = {
            "failures_at_90": paired_difference(left, right),
            "failures_excluded": paired_difference(left, right, exclude=True)}
    periods = {}
    for name in ("own", "refined"):
        valid = [r["periods"][name]["relative_error"] for r in rows
                 if r["periods"][name]["relative_error"] is not None]
        periods[name] = {"n": len(rows), "valid": len(valid),
                         "within1_percent": sum(r["periods"][name]["within1_percent"] for r in rows),
                         "mean_relative_error_excluding_failures": float(np.mean(valid)) if valid else None,
                         "median_relative_error_excluding_failures": float(np.median(valid)) if valid else None}
    # Period accuracy uses the same paired asteroid bootstrap as pole outcomes.
    difference = np.asarray([int(r["periods"]["refined"]["within1_percent"])
                             - int(r["periods"]["own"]["within1_percent"]) for r in rows])
    draws = np.random.default_rng(SEED).integers(len(rows), size=(RESAMPLES, len(rows)))
    periods["refined_minus_own_within1_fraction"] = {
        "difference": float(difference.mean()),
        "bootstrap_95": np.percentile(difference[draws].mean(axis=1), [2.5, 97.5]).tolist()}
    valid_period_pairs = [r for r in rows if r["periods"]["refined"]["relative_error"] is not None]
    period_differences = np.asarray([r["periods"]["refined"]["relative_error"]
                                    - r["periods"]["own"]["relative_error"] for r in valid_period_pairs])
    if valid_period_pairs:
        period_draws = np.random.default_rng(SEED).integers(
            len(valid_period_pairs), size=(RESAMPLES, len(valid_period_pairs)))
        periods["refined_minus_own_relative_error"] = {
            "n": len(valid_period_pairs), "mean_difference": float(period_differences.mean()),
            "bootstrap_95": np.percentile(period_differences[period_draws].mean(axis=1), [2.5, 97.5]).tolist()}
    else:
        periods["refined_minus_own_relative_error"] = {"n": 0, "mean_difference": None, "bootstrap_95": None}
    cost_totals = {key: sum(r["costs"][key] or 0 for r in rows) for key in (
        "refinement_E_core_cpu_seconds", "refinement_P_core_cpu_seconds", "inference_seconds",
        "all_152_classical_solver_cpu_seconds")}
    cost_totals["solver_cpu_seconds"] = {
        arm: sum(r["costs"]["solver_cpu_seconds"][arm] for r in rows) for arm in ARMS}
    return {"scope": "exploratory development sample; 20 objects; no wider claim",
            "bootstrap": {"unit": "paired asteroid", "resamples": RESAMPLES, "seed": SEED},
            "selection": "minimum valid RMS then start index; published classical ladder",
            "failure_rule": "unfinished refinement or no valid fit: 90 degrees and also excluded",
            "cost_note": "E/P factor 1.406; inference excludes model loading; shared classical starts are not additive",
            "objects": rows, "summary": summary, "paired_differences": pairs,
            "period_summary": periods, "cost_totals": cost_totals}


def results_markdown(scores) -> str:
    def number(v):
        return "NA" if v is None else f"{v:.6g}"
    lines = ["# Refinement chain", "", scores["scope"], "", scores["failure_rule"], "",
             "Selected-fit pole errors (degrees); selection uses RMS only.", "",
             "| Period source | Arm | Mean (90) | Median (90) | Within 20 | Failed | Mean excluded | Median excluded | N excluded |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for scenario, arms in scores["summary"].items():
        for arm, value in arms.items():
            all_, valid = value["failures_at_90"], value["failures_excluded"]
            lines.append(f"| {scenario} | {arm} | {number(all_['mean_deg'])} | {number(all_['median_deg'])} | "
                         f"{all_['within20']} | {value['failed']} | {number(valid['mean_deg'])} | "
                         f"{number(valid['median_deg'])} | {valid['n']} |")
    lines += ["", "Period accuracy and per-object errors (D6 / C12 / C146):", "",
              "| Object | Own relative error | Refined relative error | Own within 1% | Refined within 1% | Refined errors | Own errors | DAMIT errors |",
              "|---|---:|---:|---|---|---|---|---|"]
    for row in scores["objects"]:
        p = row["periods"]
        errors = [" / ".join(number(row["scenarios"][s][a]["error_deg"]) for a in ARMS)
                  for s in ("refined", "own", "published")]
        lines.append(f"| {row['object_id']} | {number(p['own']['relative_error'])} | "
                     f"{number(p['refined']['relative_error'])} | {p['own']['within1_percent']} | "
                     f"{p['refined']['within1_percent']} | " + " | ".join(errors) + " |")
    lines += ["", "Paired differences, 10,000 asteroid resamples, seed 20261026; 95% percentile intervals.", "",
              "| Left minus right | Failures | N | Mean error difference [95%] | Within-20 fraction difference [95%] |",
              "|---|---|---:|---|---|"]
    for name, modes in scores["paired_differences"].items():
        for mode, value in modes.items():
            ci = value["bootstrap_95"]
            interval = lambda key: "NA" if ci is None else ", ".join(number(x) for x in ci[key])
            lines.append(f"| {name} | {mode} | {value['n']} | {number(value['mean_error_difference_deg'])} "
                         f"[{interval('mean_error_difference_deg')}] | {number(value['within20_fraction_difference'])} "
                         f"[{interval('within20_fraction_difference')}] |")
    lines += ["", scores["cost_note"], "",
              "| Object | Refinement E CPU s | P CPU s | Inference s | Solver D6 s | C12 s | C146 s | All 152 classical s | Full-range estimate |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in scores["objects"]:
        c = row["costs"]
        values = [c["refinement_E_core_cpu_seconds"], c["refinement_P_core_cpu_seconds"], c["inference_seconds"],
                  *(c["solver_cpu_seconds"][a] for a in ARMS), c["all_152_classical_solver_cpu_seconds"]]
        estimate = json.dumps(c["full_range_scan_estimate"], sort_keys=True) if c["full_range_scan_estimate"] else "NA"
        lines.append(f"| {row['object_id']} | " + " | ".join(number(v) for v in values) + f" | {estimate} |")
    lines += ["", "Period summary: `" + json.dumps(scores["period_summary"], sort_keys=True) + "`", "",
              "Cost totals (seconds): `" + json.dumps(scores["cost_totals"], sort_keys=True) + "`", "",
              "Input bindings and selected fits are retained in scores.json.", ""]
    return "\n".join(lines)


def command_score(args) -> int:
    _validate_runtime(read(args.benchmark_dir / "evaluation-lock.json"))
    lock = checked_lock(args)
    predictions = load_predictions(args, lock)
    if network.switched():
        # The fit plan records the choice, including for subsequent score calls.
        args.classical_timing = read(args.output_dir / "fits/fit-plan.json").get("classical_timing", "reuse")
    records = records_by_arm(args, lock, predictions)
    # No reference catalog, period, pole or baseline result is opened above.
    parent, refined, selected = refined_inputs(args)
    scoring_path = args.benchmark_dir / "scoring-lock.json"
    scoring = read(scoring_path)
    if scoring["evaluation_lock"] != binding(args.benchmark_dir / "evaluation-lock.json"):
        raise ValueError("reference binding belongs to a different parent lock")
    catalog = verify(scoring["reference_catalog"])
    references = {}
    object_ids = {r["object_id"] for r in selected}
    with catalog.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["object_id"] in object_ids:
                if row["object_id"] in references or not row.get("solutions"):
                    raise ValueError("duplicate reference or missing solutions")
                references[row["object_id"]] = {
                    "period_hours": float(row["solutions"][0]["period_hours"]),
                    "axes": [s["vector"] for s in row["solutions"]]}
    if set(references) != object_ids or any(
            not math.isfinite(r["period_hours"]) or r["period_hours"] <= 0
            or np.asarray(r["axes"]).ndim != 2 or np.asarray(r["axes"]).shape[1] != 3
            or not np.isfinite(r["axes"]).all()
            or np.any(np.linalg.norm(r["axes"], axis=1) == 0) for r in references.values()):
        raise ValueError("missing or invalid scoring references")
    baselines, sources = {}, [binding(scoring_path), binding(catalog)]
    for name, directory in (("own", args.benchmark_dir), ("published", args.published_dir)):
        fits, bindings = baseline_fits(directory, [r["object_id"] for r in selected])
        baselines[name] = fits
        sources.extend(bindings)
    own_path = DATA / "own-periods-170.json"
    own = read(own_path)["periods_hours"]
    sources.append(binding(own_path))
    estimates = {}
    if args.cost_summary.exists():
        estimates = read(args.cost_summary)["estimates"]
        sources.append(binding(args.cost_summary))
    scores = make_scores(selected, refined, own, references, records, predictions, baselines,
                         ladder_indexes(args.benchmark_dir), estimates)
    scores.update(network.json_metadata())
    if network.switched():
        scores["classical_timing"] = args.classical_timing
        scores["solver_workers"] = read(args.output_dir / "fits/fit-plan.json")["workers"]
        scores["classical_retiming_cpu_seconds"] = (sum(
            r["solver_cpu_seconds"] for arms in records.values()
            for arm in ("standard6", "classical20") for r in arms[arm])
            if args.classical_timing == "retime" else 0)
    scores["inputs"] = {"lock": binding(args.output_dir / "evaluation-lock-refined.json"),
                       "fits": binding(args.output_dir / "fits/complete.json"), "sources": sources}
    save_outputs({args.output_dir / "scores.json": scores,
                  args.output_dir / "RESULTS.md": results_markdown(scores)})
    return 0


def parse_cpus(spec: str) -> tuple[int, ...]:
    values = []
    try:
        for item in spec.split(","):
            bounds = [int(v) for v in item.split("-")]
            if len(bounds) == 1:
                values.append(bounds[0])
            elif len(bounds) == 2 and 0 <= bounds[0] <= bounds[1]:
                values.extend(range(bounds[0], bounds[1] + 1))
            else:
                raise ValueError
        if not values or min(values) < 0 or len(values) != len(set(values)):
            raise ValueError
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected unique CPUs, e.g. 12-23 or 12,14") from exc
    return tuple(values)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {"lock": command_lock, "predict": command_predict,
                "fits": command_fits, "score": command_score}
    for name, function in commands.items():
        command = sub.add_parser(name)
        command.set_defaults(function=function)
        command.add_argument("--output-dir", type=Path, default=OUTPUT)
        command.add_argument("--benchmark-dir", type=Path, default=BENCHMARK)
        command.add_argument("--protocol", type=Path, default=DATA / "refine-chain/PROTOCOL.md" if network.switched() else OUTPUT / "PROTOCOL.md")
        if name == "predict":
            command.add_argument("--device", default="cuda")
        if name == "fits":
            command.add_argument("--cpus", type=parse_cpus, default=tuple(range(12, 24)))
            command.add_argument("--workers", type=int)
            command.add_argument("--classical-timing", choices=("reuse", "retime"), default="reuse")
        if name == "score":
            command.add_argument("--published-dir", type=Path, default=PUBLISHED)
            command.add_argument("--cost-summary", type=Path, default=DATA / "period-cost/period-cost-summary.json")
    args = parser.parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
