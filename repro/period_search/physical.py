"""Budgeted free-period physical fitting; references enter only final scoring.

Each photometric frequency cell is scanned at the DAMIT period_scan spacing.
The cell width is a search resolution, NOT a confidence interval. Poles proposed
at the cell centre are reused for nearby starts; period and pole then vary freely.
Linux process groups enforce the arm deadline, including descendants.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "DeLPHI-followup"
SOURCE = BASE / "data/solver-internal-capacity-corrected/20260911/expanded-solver/convexinv"
CATALOG = BASE / "inputs/training-run/repro/data/damit-20250610T000301Z"
STANDARD = ((0., 0.), (180., 0.), (90., 60.), (240., 60.), (90., -60.), (240., -60.))


def selected_candidates(row, policy):
    count = {"one": 1, "three": 3, "adaptive": row.get("retained_count", 3)}[policy]
    if not isinstance(count, int) or not 1 <= count <= 3:
        raise ValueError("retained_count must be an integer from one to three")
    candidates = row.get("candidates", [])
    ranked = row.get("ranked_periods", [])
    result = []
    for period in ranked[:count]:
        candidate = next((c for c in candidates if math.isclose(float(c["period"]), float(period), rel_tol=1e-10)), None)
        if candidate is None:
            raise ValueError("ranked period has no candidate frequency cell")
        p, width = float(period), float(candidate["frequency_step"])
        if not math.isfinite(p) or not 2 <= p <= 200 or not math.isfinite(width) or width <= 0:
            raise ValueError("invalid period or frequency cell")
        result.append({"period": p, "frequency_step": width})
    if not result:
        raise ValueError("no candidate period")
    return result


def trial_cells(candidates, baseline_days):
    if not math.isfinite(baseline_days) or baseline_days <= 0:
        raise ValueError("a positive observation baseline is required")
    spacing = 0.4 / baseline_days
    cells = []
    for c in candidates:
        centre = 24 / c["period"]
        lo = max(24 / 200, centre - c["frequency_step"] / 2)
        hi = min(24 / 2, centre + c["frequency_step"] / 2)
        # Centre and both sides, with endpoint distances never exceeding spacing.
        left, right = math.ceil((centre - lo) / spacing), math.ceil((hi - centre) / spacing)
        cells.append({"period": c["period"], "lo": lo, "hi": hi, "centre": centre,
                      "left": left, "right": right, "spacing": spacing,
                      "trial_count": 1 + left + right})
    return cells


def iter_trials(cells):
    """Round-robin candidates and centre-out offsets avoid spending all on rank 1."""
    yield from ((i, 0, c["period"]) for i, c in enumerate(cells))
    for offset in range(1, max(max(c["left"], c["right"]) for c in cells) + 1):
        for i, c in enumerate(cells):
            if offset <= c["left"]:
                yield i, -offset, 24 / max(c["lo"], c["centre"] - offset * c["spacing"])
            if offset <= c["right"]:
                yield i, offset, 24 / min(c["hi"], c["centre"] + offset * c["spacing"])


def preflight(path, source=SOURCE):
    import re
    constants = (source / "constants.h").read_text()
    limits = {k: int(re.search(r"#define\s+" + k + r"\s+(\d+)", constants)[1])
              for k in ("MAX_LC", "MAX_N_OBS", "POINTS_MAX")}
    words = path.read_text().split()
    ncurves, cursor, sizes, times = int(words[0]), 1, [], []
    for _ in range(ncurves):
        n = int(words[cursor]); cursor += 2
        sizes.append(n)
        for _ in range(n):
            values = list(map(float, words[cursor:cursor + 8])); cursor += 8
            if len(values) != 8 or not all(map(math.isfinite, values)) or values[1] <= 0:
                raise ValueError("invalid observation")
            times.append(values[0])
    if cursor != len(words) or min(sizes, default=0) < 2:
        raise ValueError("malformed lightcurve file")
    if ncurves > limits["MAX_LC"] or sum(sizes) > limits["MAX_N_OBS"] or max(sizes) > limits["POINTS_MAX"]:
        raise ValueError("solver capacity exceeded")
    return {"lightcurves": ncurves, "observations": sum(sizes), "largest_lightcurve": max(sizes),
            "baseline_days": max(times) - min(times), "solver_limits": limits}


def choose_fit(rows):
    valid = [r for r in rows if r.get("return_code") == 0 and not r.get("timed_out")
             and all(r.get(k) is not None and math.isfinite(r[k]) for k in
                     ("relative_rms_from_output", "final_period_hours", "final_lambda_deg", "final_beta_deg"))
             and r["final_period_hours"] > 0]
    return min(valid, key=lambda r: (r["relative_rms_from_output"], r["trial_index"]), default=None)


def score_selected(selected, reference_period, references):
    if selected is None:
        return {"period_relative_error": None, "pole_error_deg": None, "joint_within_1pct_20deg": False}
    lon, lat = map(math.radians, (selected["final_lambda_deg"], selected["final_beta_deg"]))
    vector = (math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat))
    err = min(math.degrees(math.acos(min(1., abs(sum(a*b for a, b in zip(vector, ref)))))) for ref in references)
    perr = abs(selected["final_period_hours"] / reference_period - 1)
    return {"period_relative_error": perr, "pole_error_deg": err,
            "joint_within_1pct_20deg": perr <= .01 and err <= 20}


def load_models(fold):
    import torch
    from lc_pipeline.k3.ztf_prediction import _load_models
    torch.set_num_threads(1)
    models, _, bindings = _load_models(BASE / "inputs/frozen-artifacts/k3-definitive-7874092/models",
                                      BASE / "data/external-models-final.json", (fold,), "cuda")
    torch.cuda.synchronize()
    return models, bindings


def neural_axes(models, epochs, period):
    import numpy as np
    import torch
    from lc_pipeline.k3.bundle import _model_inputs
    from lc_pipeline.k3.evaluation import ensemble_score_grids, modes_from_score_grid
    from lc_pipeline.k3.inference import refine_ensemble_axes, score_axial_grid
    from lc_pipeline.v2.preprocessing import KnownPeriod
    from lc_pipeline.v2.convexinv import axis_to_six_pole_starts
    inputs = _model_inputs(epochs, KnownPeriod(period, "photometric-period-search"), torch.device("cuda"))
    with torch.inference_mode():
        grids = np.stack([score_axial_grid(m, inputs, chunk_size=1024) for m in models])
        modes = modes_from_score_grid(ensemble_score_grids(grids[:, None, :])[0])
    with torch.enable_grad():
        axes, _ = refine_ensemble_axes(models, inputs, np.asarray([m.axis_xyz for m in modes]))
    torch.cuda.synchronize()
    return axis_to_six_pole_starts(axes)


def worker(config):
    sys.path.insert(0, str(BASE / "source"))
    directory = Path(config["directory"])
    worker_start = time.perf_counter()

    def event(name):
        with (directory / "events.jsonl").open("a", buffering=1) as log:
            log.write(json.dumps({"event": name, "worker_elapsed_seconds": time.perf_counter()-worker_start}) + "\n")

    event("worker_started")
    models = None
    bindings = None
    if config["mode"] == "warm":
        from lc_pipeline.v2.convexinv import run_convexinv  # warm imports as well
        if config["arm"] == "guided":
            models, bindings = load_models(config["fold"])
    print("READY", flush=True)
    if sys.stdin.readline().strip() != "GO":
        return
    started = time.perf_counter()
    event("timed_work_started")
    from lc_pipeline.v2.convexinv import ConvexinvParameters, run_convexinv, write_convexinv_parameters
    event("solver_wrapper_imported")
    path, directory = Path(config["lightcurve"]), Path(config["directory"])
    measured = {}
    load_start = time.perf_counter()
    if config["arm"] == "guided" and models is None:
        event("model_loading_started")
        models, bindings = load_models(config["fold"])
        event("model_loading_finished")
    measured["load_seconds"] = time.perf_counter() - load_start
    measured["checkpoint_bindings"] = bindings
    start = time.perf_counter()
    starts = [STANDARD for _ in config["cells"]]
    if config["arm"] == "guided":
        from lc_pipeline.v2.data import parse_damit_lightcurve
        epochs = parse_damit_lightcurve(path)
        event("neural_inference_started")
        starts = [neural_axes(models, epochs, c["period"]) for c in config["cells"]]
        event("neural_inference_finished")
    measured["neural_seconds"] = time.perf_counter() - start
    (directory / "stages.json").write_text(json.dumps(measured))
    deadline = started + config["budget"]
    tasks = ((ci, offset, period, si, pole) for ci, offset, period in iter_trials(config["cells"])
             for si, pole in enumerate(starts[ci]))

    def fit(index, task):
        ci, offset, period, si, pole = task
        out = directory / f"fit-{index:07d}"
        out.mkdir()
        params = write_convexinv_parameters(out / "parameters.txt", ConvexinvParameters(
            lambda_deg=pole[0], beta_deg=pole[1], period_hours=period, free_period=True,
            iteration_stop_condition=.0003))
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return {"trial_index": index, "status": "budget_exhausted"}
        result = run_convexinv(executable=SOURCE / "convexinv", source_root=SOURCE,
                              lightcurve_file=path, parameter_file=params, output_directory=out,
                              timeout_seconds=remaining, stdout_log_path=out / "stdout.log",
                              stderr_log_path=out / "stderr.log")
        row = dataclasses.asdict(result)
        row.update(trial_index=index, candidate_index=ci, frequency_offset_index=offset,
                   initial_period=period, start_index=si)
        return row

    with (directory / "fits.jsonl").open("x", buffering=1) as stream:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            pending, index, exhausted = {}, 0, False
            while time.perf_counter() < deadline and (pending or not exhausted):
                while len(pending) < 6 and not exhausted and time.perf_counter() < deadline:
                    task = next(tasks, None)
                    if task is None:
                        exhausted = True
                        break
                    pending[pool.submit(fit, index, task)] = index
                    index += 1
                if not pending:
                    break
                done, _ = concurrent.futures.wait(pending, timeout=max(0., deadline-time.perf_counter()),
                                                 return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    try:
                        row = future.result()
                    except Exception as exc:
                        row = {"trial_index": pending[future], "status": "error", "error": repr(exc)}
                    stream.write(json.dumps(row) + "\n")
                    del pending[future]


def run_arm(config):
    directory = Path(config["directory"])
    directory.mkdir(parents=True, exist_ok=False)
    request = directory / "request.json"
    request.write_text(json.dumps(config))
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    started = time.perf_counter()
    stderr = (directory / "worker.stderr").open("w")
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(request)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                               text=True, env=env, start_new_session=True)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    setup_limit = config["budget"] if config["mode"] == "cold" else 120
    ready = False
    while time.perf_counter() - started < setup_limit and process.poll() is None:
        if selector.select(timeout=min(.1, max(0., setup_limit-(time.perf_counter()-started)))):
            if process.stdout.readline().strip() == "READY":
                ready = True
                break
    setup = time.perf_counter() - started
    timed_start = started if config["mode"] == "cold" else time.perf_counter()
    timeout = False
    if ready:
        process.stdin.write("GO\n"); process.stdin.flush()
        try:
            process.wait(timeout=max(.001, config["budget"] - (time.perf_counter()-timed_start)))
        except subprocess.TimeoutExpired:
            timeout = True
    else:
        timeout = True
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    elapsed = time.perf_counter() - timed_start
    selector.close(); stderr.close()
    process.stdin.close(); process.stdout.close()
    fits = []
    if (directory / "fits.jsonl").exists():
        for line in (directory / "fits.jsonl").read_text().splitlines():
            try:
                fits.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a deadline can interrupt the last line
    expected = sum(c["trial_count"] for c in config["cells"]) * 6
    complete = len(fits) == expected and all(choose_fit([r]) is not None for r in fits)
    selected = choose_fit(fits)
    try:
        stages = json.loads((directory / "stages.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        stages = {}
    return {"arm": config["arm"], "seconds": elapsed, "setup_seconds": setup,
            "mode": config["mode"], "timed_out": timeout, "worker_returncode": process.returncode,
            "completed_search": complete, "scheduled_starts": expected, "recorded_starts": len(fits),
            "valid_starts": sum(choose_fit([r]) is not None for r in fits), "selected": selected,
            "status": "complete" if complete else "incomplete", "raw_directory": str(directory),
            "stages": stages, "warm_definition": "worker and model preparation excluded" if config["mode"] == "warm" else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mode", choices=("warm", "cold"), default="cold")
    parser.add_argument("--policies", nargs="+", choices=("one", "three", "adaptive"), default=["one", "three", "adaptive"])
    parser.add_argument("--methods", nargs="+", default=["selected_classical", "random_forest"])
    parser.add_argument("--budget", type=float, default=30.)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(json.loads(args.worker.read_text()))
        return
    if not args.predictions or not args.output or args.budget <= 0:
        parser.error("predictions, output and a positive budget are required")
    catalog = {r["object_id"]: r for r in map(json.loads, (CATALOG / "catalog.jsonl").read_text().splitlines())}
    folds = {oid: f["fold"] for f in json.loads((CATALOG / "publication-splits-v2.3.json").read_text())["folds"] for oid in f["test_ids"]}
    rows = list(map(json.loads, args.predictions.read_text().splitlines()))
    rows = [r for r in rows if r.get("method") in args.methods]
    order = sorted({r["object_id"] for r in rows}, key=lambda x: hashlib.sha256(("period-pilot-20260921:" + x).encode()).hexdigest())[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", buffering=1) as stream:
        for row in rows:
            oid = row["object_id"]
            if oid not in order:
                continue
            cat = catalog[oid]
            lightcurve = ROOT / "damit-20250610T000301Z" / cat["lightcurve"]["source_path"]
            for policy in args.policies:
                result = {"object_id": oid, "method": row["method"], "policy": policy}
                try:
                    info = preflight(lightcurve)
                    cells = trial_cells(selected_candidates(row, policy), info["baseline_days"])
                    result.update(preflight=info, cells=cells, scheduled_starts=sum(c["trial_count"] for c in cells)*6)
                except (KeyError, ValueError) as exc:
                    result.update(status="unresolved", reason=str(exc))
                    stream.write(json.dumps(result) + "\n")
                    continue
                if args.preflight_only:
                    stream.write(json.dumps(result) + "\n")
                    continue
                for repeat in range(args.repeats):
                    arms = ("standard", "guided") if (order.index(oid)+repeat) % 2 == 0 else ("guided", "standard")
                    for arm in arms:
                        token = hashlib.sha256(f"{oid}/{row['method']}/{policy}/{repeat}/{arm}".encode()).hexdigest()[:16]
                        config = {"mode": args.mode, "arm": arm, "fold": folds[oid], "budget": args.budget,
                                  "lightcurve": str(lightcurve), "directory": str(args.output.with_suffix(".runs") / token), "cells": cells}
                        measured = run_arm(config)
                        # References are read only after the arm has selected its fit.
                        refperiod = float(cat["solutions"][0]["period_hours"])
                        measured.update(score_selected(measured["selected"], refperiod, [s["vector"] for s in cat["solutions"]]))
                        stream.write(json.dumps({**result, "repeat": repeat, **measured}) + "\n")
                        print(oid, row["method"], policy, arm, measured["status"], flush=True)


if __name__ == "__main__":
    main()
