"""Network-only replay of the locked direct benchmark and its saved analyses.

Historical numerical producers remain read-only. Their SHA-256s below identify
the exact own-period copies mirrored by this adapter. Budgets are never selected
again. Classical receipts explicitly identify reused evidence; retiming must
reproduce every saved start, including solver products and failures.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import resource
import subprocess
import sys
import time
import types
from copy import deepcopy
from pathlib import Path

from lc_pipeline import grid_benchmark as grid
from lc_pipeline import workflow_benchmark as prior
from repro import network_root as network
from repro.own_period import DataPath

REPO = Path(__file__).resolve().parents[2]
SOURCES = DataPath("benchmark")
PREVIOUS_SCRIPT_SHA256 = "c56f35aa8e602e395d909944835230b89ee9a93a2f4dbfdf8e6e191b004a0d2d"
RT4_SCRIPT_SHA256 = "4355cdfa3ef6a4f0830aa39acce3a2c574beb48664577b0b5b37693a1af451fc"
RT5_PRODUCER_SHA256 = "707f6c032eeed39bd188bd1ce60be2e828dcdb9fc87e37fdbb4ef2c681cc8691"
RT5_SCRIPT_SHA256 = "41930e3244ab8e4089415f8b956ed9af5450e8028801cf71b63986958a3860ea"
HASHES = {
    "run_direct_benchmark.py": "a3358ddeec1f17bed219282fab31ec576bb213a6ab5e5ce8845c057d07f51fe4",
    "run_loaded_benchmark.py": "0b3519b9685c6364ac584ad6cf5104592b6f008f091da72033cd0dec9eb82f29",
    "time_exploratory_budgets.py": "e209af0019f3ac3bba4fa14895173a460985d31587598053545b944029b63f8c",
    "explore_classical_ladder.py": "ab18e4d6c16a319749ac57bc1c10c69e3797b4d6e1acd24cf6708d642151809f",
    "score_classical_ladder.py": "396655ab11763494751405548aa6e76dca96f04031f55f116765a96d187ec245",
    "score_exploratory_timing.py": "7a027886a7043d1667ad4085d3953682e7e98184cb9c55e28081d597e09b1bbb",
    "budget_equivalence.py": "76cb39c9c889809e1b42bb27201ee91977a19074a7ea08076a962531ed05ede8",
    "candidate_accuracy_cost.py": "b39e2d121f963e45862a95f33da636127d67fdbef33ca2c7a904b713cf703681",
}


def read(path):
    return json.loads(Path(path).read_text())


def bound(path):
    return {"path": str(Path(path).resolve()), "sha256": network.sha256(path)}


def verify(record):
    # Archived locks may name a path through the lowercase compatibility links, so compare it resolved.
    if bound(record["path"]) != {**record, "path": str(Path(record["path"]).resolve())}:
        raise ValueError(f"bound input changed: {record['path']}")
    return Path(record["path"])


def save(path, value):
    network.guard_output(path)
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"existing output differs: {path}")
    else:
        grid.write_once(path, value)


def producer(name, **overrides):
    """Execute the exact saved functions; override only explicit path globals."""
    path = Path(__file__).with_name(name) if name == "score_exploratory_timing_rt5.py" else SOURCES / name
    expected = RT5_PRODUCER_SHA256 if name == "score_exploratory_timing_rt5.py" else HASHES[name]
    if network.sha256(path) != expected:
        raise ValueError(f"historical producer changed: {path}")
    module = types.ModuleType(name[:-3])
    module.__file__ = str(path)
    module.__dict__.update(overrides)
    tree = ast.parse(path.read_text(), filename=str(path))
    tree.body = [n for n in tree.body if not (
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in overrides for t in n.targets))]
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def modules():
    direct = producer("run_direct_benchmark.py")
    direct.validate_lock = validate
    sys.modules["run_direct_benchmark"] = direct
    loaded = producer("run_loaded_benchmark.py")
    sys.modules["run_loaded_benchmark"] = loaded
    return direct, loaded


def timed_workers(original, timing):
    """Keep the source lock's solver concurrency for reused and fresh timings."""
    if type(original) is not int or original < 1:
        raise ValueError("original solver workers must be a positive integer")
    if timing not in ("reuse", "retime"):
        raise ValueError("unknown classical timing mode")
    return original


def prepare(source, output, device="cuda", timing="reuse", workers=None):
    network.guard_output(output)
    source, output = Path(source), Path(output)
    old = read(source / "evaluation-lock.json")
    selected_workers = timed_workers(old["settings"]["workers"], timing)
    if workers is not None and workers != selected_workers:
        raise ValueError("solver workers differ from the timing protocol")
    output.mkdir(parents=True, exist_ok=True)
    for name, digest in HASHES.items():
        path = SOURCES / name
        if network.sha256(path) != digest:
            raise ValueError("historical producer changed")
        target = output / name
        if target.exists():
            if network.sha256(target) != digest:
                raise ValueError("mirrored producer changed")
        else:
            with target.open("xb") as stream:
                stream.write(path.read_bytes())
    lock = deepcopy(old)
    lock.update(code_root=str(REPO), runtime_sources=prior._runtime_sources(REPO),
                device=device, environment=grid._environment(device), **network.json_metadata())
    lock["bundles"] = [network.bundle_record(fold, old["bundles"]) for fold in range(5)]
    lock["rerun"] = {"source_lock": bound(source / "evaluation-lock.json"),
        "source_scoring": bound(source / "scoring-lock.json"), "source_directory": str(source.resolve()),
        "solver_workers": selected_workers, "classical_timing": timing,
        "script": bound(Path(__file__)), "historical_producers": HASHES,
        "fixed_budgets": [12, 18, 96], "old_solver_workers": old["settings"]["workers"],
        "order": "original seed 20260923; loaded seed+fold; exploratory seed 20260924+fold"}
    if lock["selected_classical_starts"] != 12:
        raise ValueError("published 12-start budget changed")
    save(output / "evaluation-lock.json", lock)
    scoring = read(source / "scoring-lock.json")
    save(output / "scoring-lock.json", {**scoring, **network.json_metadata(),
        "solver_workers": selected_workers,
        "evaluation_lock": bound(output / "evaluation-lock.json")})
    return lock


def validate(path, *, environment=True):
    network.check_json(path)
    lock = read(path)
    if "rerun" not in lock:
        raise ValueError("network rerun requires a distinct lock")
    old = read(verify(lock["rerun"]["source_lock"]))
    verify(lock["rerun"]["source_scoring"])
    script = lock["rerun"]["script"]
    # Locks written before e15c5ad bind this file as it was then; that commit changed only verify().
    # RT4/RT5 evidence remains readable after scoring changes; new executions bind the current file.
    archived = (not environment and any(script == {
        "path": str(Path(__file__).resolve()), "sha256": digest}
        for digest in (RT4_SCRIPT_SHA256, RT5_SCRIPT_SHA256)))
    if not archived and script != {"path": str(Path(__file__).resolve()), "sha256": PREVIOUS_SCRIPT_SHA256}:
        verify(script)
    if (lock["rerun"]["solver_workers"] != timed_workers(old["settings"]["workers"], lock["rerun"]["classical_timing"])
            or lock["rerun"]["old_solver_workers"] != old["settings"]["workers"]):
        raise ValueError("fixed benchmark solver workers changed")
    for key in ("objects", "classical_starts", "selected_classical_starts", "settings", "solver", "solver_settings"):
        if lock[key] != old[key]:
            raise ValueError(f"fixed benchmark input changed: {key}")
    if lock["runtime_sources"] != prior._runtime_sources(Path(lock["code_root"])):
        raise ValueError("runtime sources changed")
    if environment and lock["environment"] != grid._environment(lock["device"]):
        raise ValueError("rerun environment changed")
    for record in [lock["solver"]["executable"], *(row["lightcurve"] for row in lock["objects"])]:
        verify(record)
    for fold in range(5):
        if network.bundle_record(fold, old["bundles"]) != lock["bundles"][fold]:
            raise ValueError("chosen bundle changed")
    return lock


def audit_start(path):
    row = read(path)
    for name, digest in row["files"].items():
        if Path(name).name != name or network.sha256(Path(path).parent / name) != digest:
            raise ValueError(f"saved solver product changed: {path}")
    return row


def equal_start(new, old):
    keys = ("start_index", "initial_axis", "return_code", "timed_out", "iterations",
            "selected_fit", "failure_reason", "files")
    if any(new.get(k) != old.get(k) for k in keys):
        raise ValueError("retimed classical start differs from the saved fit")


def equal_downstream(new, old):
    keys = ("return_code", "timed_out", "iterations", "chi2", "deviation", "final_lambda_deg",
            "final_beta_deg", "final_period_hours", "relative_rms_from_output",
            "output_lightcurve_sha256", "output_parameter_sha256", "output_area_sha256", "stdout_sha256")
    if new["identity"] != old["identity"] or any(new["result"].get(k) != old["result"].get(k) for k in keys):
        raise ValueError("retimed classical downstream fit differs from the saved fit")


def jobs(lock, mode):
    if mode == "cold":
        rows = [{**r, "repeat_index": n, "arm": arm} for r in lock["objects"]
                for n in range(3) for arm in ("classical", "delphi")]
        random.Random(20260923).shuffle(rows)
        return rows
    result = []
    for fold in range(5):
        rows = [{**r, "repeat_index": n, "arm": arm} for r in lock["objects"] if r["fold"] == fold
                for n in (range(1) if mode == "exploratory" else range(3))
                for arm in (("delphi", "classical18", "classical96") if mode == "exploratory"
                            else ("classical", "delphi"))]
        random.Random((20260924 if mode == "exploratory" else 20260923) + fold).shuffle(rows)
        result.extend(rows)
    return result


def execute(lock_path, output, mode):
    lock = validate(lock_path)
    network.guard_output(output)
    direct, loaded = modules()
    source = Path(lock["rerun"]["source_directory"])
    old_run = source / {"cold": "evaluation", "loaded": "evaluation-loaded",
                        "exploratory": "exploratory-loaded-ladder"}[mode]
    old_execution = read(old_run / "execution.json")
    old_bindings = {r["path"]: r["sha256"] for r in old_execution["rows"]}
    save(output / "rerun-request.json", {"lock": bound(lock_path), "mode": mode,
        "source_execution": bound(old_run / "execution.json"), **network.json_metadata()})
    if (output / "execution.json").exists():
        verify_execution(lock_path, output / "execution.json")
        return
    predictor, active_fold, setups, bindings = None, None, [], []
    for job in jobs(lock, mode):
        name = f"{job['object_id']}-r{job['repeat_index']}-{job['arm']}"
        case = output / "cases" / name
        timed = case / "timed-result.json"
        if timed.exists():
            network.check_json(timed)
            row = read(timed)
            for record in row.get("start_records", []):
                audit_start(verify(record))
            bindings.append({"path": str(timed.relative_to(output)), "sha256": network.sha256(timed)})
            continue
        if case.exists():
            raise FileExistsError(f"incomplete benchmark case; inspect before retry: {case}")
        old_case = old_run / "cases" / name
        old_timed = old_case / "timed-result.json"
        if job["arm"] != "delphi":
            if old_bindings.get(str(old_timed.relative_to(old_run))) != network.sha256(old_timed):
                raise ValueError("saved classical case differs from its execution manifest")
            old_records = [audit_start(p) for p in sorted((old_case / "starts").glob("start-*/record.json"))]
            if lock["rerun"]["classical_timing"] == "reuse":
                row = {**read(old_timed), "classical_source": bound(old_timed)}
                case.mkdir(parents=True)
                (case / "starts").symlink_to(old_case / "starts", target_is_directory=True)
            else:
                row = None
        else:
            row = None
        if row is None:
            current = lock
            if job["arm"].startswith("classical") and job["arm"] != "classical":
                current = {**lock, "classical_starts": direct.classical_starts()[:int(job["arm"][9:])].tolist()}
            before = resource.getrusage(resource.RUSAGE_CHILDREN)
            if mode == "cold":
                job_path = output / "jobs" / f"{name}.json"
                save(job_path, {**job, "directory": str(case.resolve())})
                env = os.environ.copy()
                env.update({key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
                started = time.perf_counter()
                result = subprocess.run([sys.executable, "-B", "-m", "repro.direct_workflow_benchmark.rerun", "_worker", "--lock", str(lock_path),
                                         "--job", str(job_path)], env=env, capture_output=True, text=True, timeout=9000)
                elapsed = time.perf_counter() - started
                if result.returncode or len(result.stdout.splitlines()) != 1:
                    raise RuntimeError(f"benchmark worker failed: {name}: {result.stderr}")
                row = {**json.loads(result.stdout), "wall_seconds": elapsed}
            else:
                if job["arm"] == "delphi" and active_fold != job["fold"]:
                    start = time.perf_counter()
                    predictor = grid._predictor(lock, job["fold"])
                    if predictor.device.type == "cuda":
                        import torch
                        torch.cuda.synchronize(predictor.device)
                    active_fold = job["fold"]
                    setups.append({"fold": active_fold, "models": 5, "setup_seconds": time.perf_counter() - start})
                row = loaded.case(current, job, predictor, case)
                timed.unlink()  # producer wrote this just now; add the binding receipt below.
                if job["arm"] == "delphi":
                    from lc_pipeline.k3.inference import gpu_numerics
                    row["gpu_numerics"] = gpu_numerics(predictor.device)
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            row["retiming_cpu_seconds"] = after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
            if job["arm"] != "delphi":
                new_records = [audit_start(p) for p in sorted((case / "starts").glob("start-*/record.json"))]
                if len(new_records) != len(old_records):
                    raise ValueError("classical retiming changed start count")
                for new, old in zip(new_records, old_records, strict=True):
                    equal_start(new, old)
                row["classical_source"] = bound(old_timed)
        row.update(network.json_metadata())
        row["start_records"] = [bound(p) for p in sorted((case / "starts").glob("start-*/record.json"))]
        save(timed, row)
        bindings.append({"path": str(timed.relative_to(output)), "sha256": network.sha256(timed)})
        print(name, flush=True)
    completion = {"schema": direct.SCHEMA, "phase": "reference_blind_execution_complete",
        "lock_sha256": network.sha256(lock_path), "rows": sorted(bindings, key=lambda r: r["path"]),
        "references_opened": False, "mode": mode, **network.json_metadata(),
        "batch_timing": {"model_setup_seconds_by_fold": setups},
        "model_setup_seconds_by_fold": setups, "script_sha256": HASHES["time_exploratory_budgets.py"],
        "environment": lock["environment"], "solver_workers": lock["rerun"]["solver_workers"],
        "classical_timing": lock["rerun"]["classical_timing"]}
    if mode == "exploratory":
        completion["phase"] = "exploratory_loaded_ladder_complete"
    save(output / "execution.json", completion)


def verify_execution(lock_path, execution_path):
    network.check_json(execution_path)
    value = read(execution_path)
    if value["lock_sha256"] != network.sha256(lock_path):
        raise ValueError("execution belongs to different networks or inputs")
    lock = validate(lock_path, environment=False)
    expected = {(j["object_id"], j["repeat_index"], j["arm"]) for j in jobs(lock, value["mode"])}
    actual = set()
    for record in value["rows"]:
        path = execution_path.parent / record["path"]
        if not path.resolve().is_relative_to(execution_path.parent.resolve()) or network.sha256(path) != record["sha256"]:
            raise ValueError("execution case changed or escaped its output folder")
        network.check_json(path)
        row = read(path)
        key = (row["object_id"], row["repeat_index"], row["arm"])
        if key in actual:
            raise ValueError("duplicate execution case")
        actual.add(key)
        for item in row["start_records"]:
            audit_start(verify(item))
        if "classical_source" in row:
            verify(row["classical_source"])
    if actual != expected:
        raise ValueError("execution cases are incomplete")
    return value


def reused_exploratory(source, lock):
    """Audit a reuse-mode ladder without rebinding it to the retimed lock."""
    source = Path(source)
    source_lock = source / "evaluation-lock.json"
    reused = validate(source_lock, environment=False)
    if lock["rerun"]["classical_timing"] != "retime" or reused["rerun"]["classical_timing"] != "reuse":
        raise ValueError("exploratory reuse requires a retime target and a reuse source")
    for key in ("source_lock", "source_scoring"):
        if bound(verify(reused["rerun"][key])) != bound(verify(lock["rerun"][key])):
            raise ValueError("exploratory reuse source belongs to different benchmark inputs")
    for key in ("objects", "classical_starts", "selected_classical_starts", "settings",
                "solver", "solver_settings", "bundles"):
        if reused[key] != lock[key]:
            raise ValueError(f"exploratory reuse source differs: {key}")
    execution_path = source / "exploratory-loaded-ladder/execution.json"
    execution = verify_execution(source_lock, execution_path)
    if execution["mode"] != "exploratory":
        raise ValueError("exploratory reuse source has the wrong execution mode")
    score_path = source / "exploratory-loaded-ladder/score.json"
    network.check_json(score_path)
    score = read(score_path)
    saved_fits = source / "classical-ladder-exploratory.json"
    network.check_json(saved_fits)
    if (score["execution_sha256"] != network.sha256(execution_path)
            or score["saved_fit_sha256"] != network.sha256(saved_fits)):
        raise ValueError("exploratory reuse score bindings changed")
    old = read(verify(reused["rerun"]["source_lock"]))
    metadata = {"status": "reused_not_retimed", "source_lock": bound(source_lock),
        "source_execution": bound(execution_path), "source_score": bound(score_path),
        "source_saved_fits": bound(saved_fits), "classical_environment": old["environment"],
        "delphi_environment": reused["environment"],
        "note": "18/96-start classical ladder times are from the old machine (September); "
                "DeLPHI ladder times are from the existing reuse-mode run. RT4 retimes neither."}
    return score, metadata


def analyses(output, exploratory_source=None):
    """Recompute every original statistic with the unchanged saved producers."""
    lock_path = output / "evaluation-lock.json"
    lock = validate(lock_path, environment=False)
    reused_score, exploratory_metadata = (reused_exploratory(exploratory_source, lock)
        if exploratory_source is not None else (None, None))
    metadata = {**network.json_metadata(), "solver_workers": lock["rerun"]["solver_workers"],
        "environment": lock["environment"], "classical_timing": lock["rerun"]["classical_timing"]}
    if exploratory_metadata is not None:
        metadata["exploratory_timing"] = exploratory_metadata
    verify(read(output / "scoring-lock.json")["reference_catalog"])
    direct, _ = modules()
    for mode, folder, filename in (("cold", "evaluation", "score.json"),
                                   ("loaded", "evaluation-loaded", "score-loaded.json")):
        verify_execution(lock_path, output / folder / "execution.json")
        if not (output / filename).exists():
            direct.score(lock_path, output / "scoring-lock.json", output / folder / "execution.json", output / filename)
            value = read(output / filename)
            value.update(metadata)
            (output / filename).write_text(json.dumps(value, sort_keys=True) + "\n")
        network.check_json(output / filename)
    source = Path(lock["rerun"]["source_directory"])
    if (source / "full-grid").exists():
        archive = source / "full-grid"
    else:
        archive = Path(read(verify(read(source / "evaluation-lock.json")["development_report"]))["source_archive"])
    reused = []
    for row in lock["objects"]:
        for arm in ("standard6", "classical20"):
            case = prior._archived_case(archive, row["object_id"], 0, arm)
            paths = sorted(case.glob("start-*/record.json"))
            if len(paths) != (6 if arm == "standard6" else 146):
                raise ValueError("saved classical grid is incomplete")
            for path in paths:
                audit_start(path)
                reused.append(bound(path))
    save(output / "reused-classical-grid.json", {**network.json_metadata(), "records": reused})
    outputs = {"explore_classical_ladder.py": "classical-ladder-exploratory.json",
        "score_classical_ladder.py": "classical-ladder-intervals.json",
        "score_exploratory_timing.py": "exploratory-loaded-ladder/score.json",
        "budget_equivalence.py": "budget-equivalence.json", "candidate_accuracy_cost.py": "candidate-accuracy-cost.json"}
    if reused_score is None:
        verify_execution(lock_path, output / "exploratory-loaded-ladder/execution.json")
    else:
        save(output / "exploratory-loaded-ladder/score.json", {**reused_score,
            **network.json_metadata(), "environment": exploratory_metadata["delphi_environment"],
            "exploratory_timing": exploratory_metadata})
    for name, filename in outputs.items():
        path = output / filename
        if path.exists():
            network.check_json(path)
            continue
        overrides = dict(ROOT=output, HERE=output, DATA=output, ARCHIVE=archive,
            OUTPUT=path, SOURCE=output / "classical-ladder-exploratory.json",
            RUN=output / "exploratory-loaded-ladder",
            CATALOG=DataPath("../../inputs/training-run/repro/data/damit-20250610T000301Z/catalog.jsonl"))
        module = producer(name, **overrides)
        if hasattr(module, "main"):
            module.main()
        value = read(path)
        value.update(metadata)
        path.write_text(json.dumps(value, sort_keys=True) + "\n")
    folders = ("evaluation", "evaluation-loaded") + (() if reused_score is not None else ("exploratory-loaded-ladder",))
    value = {**metadata, "timing_option": lock["rerun"]["classical_timing"],
        "classical_retiming_cpu_seconds": sum(read(output / folder / row["path"]).get("retiming_cpu_seconds", 0)
            for folder in folders
            for row in read(output / folder / "execution.json")["rows"]
            if read(output / folder / row["path"])["arm"] != "delphi")}
    save(output / "timing-cost.json", value)


def ladder_analyses(output, exploratory_source):
    """Rescore fresh ladder times against its own frozen quality evidence."""
    output, exploratory_source = Path(output), Path(exploratory_source)
    lock_path = output / "evaluation-lock.json"
    lock = validate(lock_path, environment=False)
    _, evidence = reused_exploratory(exploratory_source, lock)
    execution_path = output / "exploratory-loaded-ladder/execution.json"
    execution = verify_execution(lock_path, execution_path)
    if execution["mode"] != "exploratory":
        raise ValueError("fresh ladder has the wrong execution mode")
    metadata = {**network.json_metadata(), "environment": lock["environment"],
        "solver_workers": lock["rerun"]["solver_workers"], "classical_timing": "retime",
        "exploratory_timing": {"status": "retimed_together", "source_execution": bound(execution_path),
            "classical_environment": lock["environment"], "delphi_environment": lock["environment"]},
        "saved_quality": {key: evidence[key] for key in
            ("source_lock", "source_execution", "source_score", "source_saved_fits")}}
    # The ladder has its own saved DeLPHI solutions; RT4's direct fits are a different run.
    # Preserve all quality results, including the original loaded-score binding.
    for filename in ("classical-ladder-exploratory.json", "classical-ladder-intervals.json", "budget-equivalence.json"):
        source = exploratory_source / filename
        network.check_json(source)
        save(output / filename, {**read(source), **metadata, "quality_source": bound(source)})
    modules()
    path = output / "exploratory-loaded-ladder/score.json"
    if path.exists():
        raise FileExistsError(f"ladder scoring refuses to overwrite: {path}")
    module = producer("score_exploratory_timing_rt5.py", ROOT=output,
                      RUN=output / "exploratory-loaded-ladder", OUTPUT=path)
    module.main()
    score = read(path)
    path.write_text(json.dumps({**score, **metadata}, sort_keys=True) + "\n")
    save(output / "timing-cost.json", {**metadata, "timing_option": "retime",
        "classical_retiming_cpu_seconds": sum(read(execution_path.parent / row["path"])["retiming_cpu_seconds"]
            for row in execution["rows"] if read(execution_path.parent / row["path"])["arm"] != "delphi")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute", "score", "score-ladder", "_worker"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--job", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=("cold", "loaded", "exploratory"), default="cold")
    parser.add_argument("--classical-timing", choices=("reuse", "retime"), default="reuse")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--exploratory-source", type=Path,
                        help="reuse-mode benchmark supplying saved ladder quality (score-ladder) or timings (score)")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source, args.output, args.device, args.classical_timing, args.workers)
    elif args.command == "execute":
        execute(args.lock.resolve(), args.output.resolve(), args.mode)
    elif args.command == "score":
        analyses(args.output.resolve(), args.exploratory_source)
    elif args.command == "score-ladder":
        if args.exploratory_source is None:
            parser.error("score-ladder requires --exploratory-source")
        ladder_analyses(args.output.resolve(), args.exploratory_source.resolve())
    else:
        direct, _ = modules()
        original_write_once = direct.write_once
        def write_inference_receipt(path, value):
            if value.get("arm") == "delphi":
                from lc_pipeline.k3.inference import gpu_numerics
                value["gpu_numerics"] = gpu_numerics(read(args.lock)["device"])
            original_write_once(path, value)
        direct.write_once = write_inference_receipt
        direct.worker(args.lock, args.job)


if __name__ == "__main__":
    main()
