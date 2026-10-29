"""Reduced-data, noise and withheld-lightcurve tests with our own period search.

    python -m repro.own_period.reliability search --scope main --max-seconds 3300
    python -m repro.own_period.reliability search --scope withheld --max-seconds 3300
    python -m repro.own_period.reliability predict --mode damit --folds 0
    python -m repro.own_period.reliability predict --mode own
    python -m repro.own_period.reliability validate
    python -m repro.own_period.reliability withheld-candidates --mode own
    python -m repro.own_period.reliability withheld-fits --mode own
    python -m repro.own_period.reliability score

The reduced and noisy data sets are made exactly as in the published run
(``data/reliability-20260915/run-1``, same sampling functions, conditions and
seeds).  On every such data set the period is searched again with
``estimate(curves, "fourier_gpu", resolution=1, timeout=180, sparse_fallback=True)``,
each retained lightcurve being one curve with magnitude -2.5 log10(flux).  The
networks then receive the same data with the period found.  For the complete
data ("full") the period is that of ``own-periods-170.json``.  A data set
without a period is a failure.

The search and the predictions never read a reference period or pole.  Only
``--mode damit`` (the validation path) uses the DAMIT period of the published
run, and only ``score`` reads reference poles and periods.

All records are appended to JSON-lines files, one line per cell, never
rewritten; a restart skips the cells already present.  Each command stops
cleanly after ``--max-seconds`` so that it can be run in chunks.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
FOLLOWUP = ROOT / "DeLPHI-followup"
RUN = FOLLOWUP / "data/reliability-20260915/run-1"
OUT = FOLLOWUP / "data/own-period-20261021/reliability"
OWN_FULL = FOLLOWUP / "data/own-period-20261021/own-periods-170.json"
SEEDS = (20260915, 20260916, 20260917)
SEARCH = dict(method="fourier_gpu", resolution=1, timeout=180, sparse_fallback=True)
PROVENANCE = "supplied-historical-period-reliability-study"  # label only, as published
OWN_PROVENANCE = "own-fourier-gpu-top1-reliability"
WITHHELD_SAMPLINGS = ("full", "thin10")
WITHHELD_ARMS = ("classical20", "k3")


# ----------------------------------------------------------------------------- helpers

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def published_lock() -> dict:
    return json.loads((RUN / "study.json").read_text())["payload"]


def published_schedule() -> dict:
    return json.loads((RUN / "schedule.json").read_text())["payload"]


def sampling_conditions(lock: dict) -> list[dict]:
    """The 43 reduced-data and noise conditions (period-scaling ones excluded)."""
    return [c for c in lock["conditions"] if c["family"] != "period"]


def repeats_of(config: dict, schedule: dict) -> range:
    return range(schedule["repeats"] if config["repeatable"] else 1)


def main_cells(lock: dict, schedule: dict) -> list[tuple[str, str, int]]:
    grid = set(lock["grid_ids"])
    cells = []
    for row in lock["objects"]:
        for config in sampling_conditions(lock):
            if config["family"] == "two_d" and row["object_id"] not in grid:
                continue
            for repeat in repeats_of(config, schedule):
                cells.append((row["object_id"], config["condition"], repeat))
    return cells


def stable_order(cells):
    return sorted(cells, key=lambda c: sha256_bytes(f"own-period-reliability:{c}".encode()))


_SOURCES: dict[str, object] = {}


def source_curve(row: dict):
    """Native curve without any period (the period is attached later)."""
    from lc_pipeline.k3.reliability_sampling import read_lc
    oid = row["object_id"]
    if oid not in _SOURCES:
        path = Path(row["lightcurve"]["path"])
        if sha256_file(path) != row["lightcurve"]["sha256"]:
            raise ValueError(f"{oid}: lightcurve changed")
        _SOURCES[oid] = read_lc(path, object_id=oid, period_hours=None)
    return _SOURCES[oid]


def selected(row: dict, config: dict, repeat: int):
    """Same selection as ``reliability_study.selected_curve``, without a period."""
    from lc_pipeline.k3.reliability_sampling import transform_curve
    selection = {**config["sampling"], "seed": SEEDS[repeat], "repeat": 0}
    return transform_curve(source_curve(row), selection)


def retained(row: dict, sampling: str):
    """Same retained data as ``reliability_study.holdout_input``, without a period."""
    from lc_pipeline.k3.reliability_sampling import select_holdout, transform_curve
    curve = select_holdout(source_curve(row)).train
    if sampling == "thin10":
        curve = transform_curve(curve, {"kind": "point", "fraction": 0.1, "seed": SEEDS[0]})
    return curve


def search_arrays(curve) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each retained lightcurve, in file order, as (time, magnitude) sorted by time.

    This is the conversion of ``repro.period_search.search.read_curves``.
    """
    out = []
    for session in sorted(curve.valid_sessions, key=lambda s: s.source_index):
        a = np.asarray([[o.time_jd, o.flux] for o in session.observations], float)
        a = a[np.isfinite(a[:, :2]).all(axis=1) & (a[:, 1] > 0)]
        a = a[np.argsort(a[:, 0])]
        if len(a) >= 2:
            out.append((a[:, 0].copy(), -2.5 * np.log10(a[:, 1])))
    return out


def search_hash(arrays) -> str:
    h = hashlib.sha256()
    for t, m in arrays:
        h.update(np.asarray(len(t), np.int64).tobytes())
        h.update(np.ascontiguousarray(t, np.float64).tobytes())
        h.update(np.ascontiguousarray(m, np.float64).tobytes())
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_jsonl(stream, record: dict) -> None:
    stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


# ----------------------------------------------------------------------------- search

def _worker_search(arrays):
    from repro.period_search.search import Curve, estimate
    curves = [Curve(t, m) for t, m in arrays]
    started = time.perf_counter()
    try:
        result = estimate(curves, **SEARCH)
    except TimeoutError as error:
        result = {"status": "timeout", "reason": str(error), "ranked_periods": []}
    seconds = time.perf_counter() - started
    ranked = [float(p) for p in (result.get("ranked_periods") or [])]
    period = ranked[0] if result.get("status") == "ok" and ranked else None
    return {"status": result.get("status"), "reason": result.get("reason"),
            "input_regime": result.get("input_regime"), "ranked_periods": ranked[:5],
            "period_hours": period, "seconds": seconds}


def _worker_init():
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))


def command_search(args) -> int:
    lock, schedule = published_lock(), published_schedule()
    by_id = {r["object_id"]: r for r in lock["objects"]}
    configs = {c["condition"]: c for c in lock["conditions"]}
    OUT.mkdir(parents=True, exist_ok=True)
    if args.scope == "main":
        target = OUT / "search" / "periods-main.jsonl"
        cells = main_cells(lock, schedule)
        full = sorted(c for c in cells if c[1] == "full")
        cells = full + stable_order([c for c in cells if c[1] != "full"])
    else:
        target = OUT / "search" / "periods-withheld.jsonl"
        cells = [(oid, f"withheld-{s}", 0) for oid in schedule["holdout_ids"] for s in WITHHELD_SAMPLINGS]
    if args.limit:
        cells = cells[: args.limit]
    target.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["object_id"], r["condition"], r["repeat"]): r for r in read_jsonl(target)}
    by_hash = {r["search_input_sha256"]: r for r in done.values() if r.get("reused_from") is None}
    pending = [c for c in cells if c not in done]
    print(json.dumps({"scope": args.scope, "cells": len(cells), "done": len(done),
                      "pending": len(pending)}), flush=True)
    deadline = time.monotonic() + args.max_seconds
    context = multiprocessing.get_context("spawn")
    in_flight: dict[str, tuple] = {}   # hash -> (future, [cells])
    waiting = list(pending)
    started_run = time.monotonic()
    searched = 0

    def describe(cell):
        oid, condition, repeat = cell
        row = by_id[oid]
        if condition.startswith("withheld-"):
            curve = retained(row, condition.removeprefix("withheld-"))
        else:
            curve = selected(row, configs[condition], repeat)
        arrays = search_arrays(curve)
        return arrays, {"object_id": oid, "fold": row["fold"], "condition": condition,
                        "repeat": repeat, "seed": SEEDS[repeat], "n_points": curve.total_observations,
                        "n_curves": len(arrays), "search_input_sha256": search_hash(arrays)}

    with target.open("a") as stream, concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, mp_context=context, initializer=_worker_init) as pool:
        def finish(future_hash):
            future, members = in_flight.pop(future_hash)
            result = future.result()
            by_hash[future_hash] = None
            first = True
            for cell, meta in members:
                record = {**meta, **result, "search": SEARCH,
                          "reused_from": None if first else "identical-search-input"}
                if not first:
                    record["seconds"] = 0.0
                append_jsonl(stream, record)
                done[cell] = record
                first = False
            by_hash[future_hash] = done[members[0][0]]

        while waiting or in_flight:
            while waiting and len(in_flight) < 2 * args.workers and time.monotonic() < deadline:
                cell = waiting.pop(0)
                arrays, meta = describe(cell)
                key = meta["search_input_sha256"]
                if by_hash.get(key) is not None:
                    prior = by_hash[key]
                    record = {**meta, **{k: prior[k] for k in ("status", "reason", "input_regime",
                                                                "ranked_periods", "period_hours")},
                              "seconds": 0.0, "search": SEARCH, "reused_from": "identical-search-input"}
                    append_jsonl(stream, record)
                    done[cell] = record
                elif key in in_flight:
                    in_flight[key][1].append((cell, meta))
                else:
                    in_flight[key] = (pool.submit(_worker_search, arrays), [(cell, meta)])
                    searched += 1
            if time.monotonic() >= deadline and waiting:
                waiting = []
            if not in_flight:
                continue
            finished, _ = concurrent.futures.wait([f for f, _ in in_flight.values()],
                                                  return_when=concurrent.futures.FIRST_COMPLETED)
            for key in [k for k, (f, _) in in_flight.items() if f in finished]:
                finish(key)
    remaining = [c for c in cells if c not in done]
    summary = {"scope": args.scope, "searched_this_chunk": searched,
               "wall_seconds": time.monotonic() - started_run, "remaining": len(remaining)}
    print(json.dumps(summary), flush=True)
    if not remaining and not args.limit:
        marker = target.with_suffix(".complete")
        if not marker.exists():
            marker.write_text(json.dumps({"cells": len(cells), "finished_epoch": time.time()}) + "\n")
    return 0


# ----------------------------------------------------------------------------- periods

def own_periods(scope: str) -> dict[tuple[str, str, int], float | None]:
    path = OUT / "search" / f"periods-{scope}.jsonl"
    marker = path.with_suffix(".complete")
    if not marker.exists():
        raise ValueError(f"{path} is not complete")
    periods = {(r["object_id"], r["condition"], r["repeat"]): r["period_hours"] for r in read_jsonl(path)}
    if scope == "main":
        full = json.loads(OWN_FULL.read_text())["periods_hours"]
        for key in [k for k in periods if k[1] == "full"]:
            periods[key] = full[key[0]]
    return periods


# ----------------------------------------------------------------------------- predictions

class _NoBudget:
    state = {"phase": "own-period"}

    def check(self, reserve=0):
        return None

    def remaining(self):
        return math.inf


def command_predict(args) -> int:
    import torch

    from lc_pipeline.k3.bundle import K3EnsemblePredictor
    from lc_pipeline.k3.reliability_study import input_hash, input_metadata, predict_curve
    lock, schedule = published_lock(), published_schedule()
    configs = {c["condition"]: c for c in lock["conditions"]}
    by_id = {r["object_id"]: r for r in lock["objects"]}
    target = OUT / "predictions" / f"{args.mode}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    periods = own_periods("main") if args.mode == "own" else None
    cells = [c for c in main_cells(lock, schedule) if by_id[c[0]]["fold"] in args.folds]
    done = {(r["object_id"], r["condition"], r["repeat"]) for r in read_jsonl(target)}
    deadline = time.monotonic() + args.max_seconds
    torch.set_num_threads(6)
    written = 0
    with target.open("a") as stream:
        for fold in args.folds:
            todo = [c for c in cells if by_id[c[0]]["fold"] == fold and c not in done]
            if not todo or time.monotonic() >= deadline:
                continue
            bundle = lock["bundles"][fold]
            for item in bundle["files"]:
                if sha256_file(Path(item["path"])) != item["sha256"]:
                    raise ValueError(f"bundle file changed: {item['path']}")
            predictor = K3EnsemblePredictor(bundle["directory"], device=args.device)
            for cell in todo:
                if time.monotonic() >= deadline:
                    break
                oid, condition, repeat = cell
                row = by_id[oid]
                period = row["period_hours"] if args.mode == "damit" else periods[cell]
                started = time.monotonic()
                record = {"object_id": oid, "fold": fold, "condition": condition, "repeat": repeat,
                          "seed": SEEDS[repeat], "mode": args.mode, "period_hours": period,
                          "source_sha256": row["lightcurve"]["sha256"], "model_count": 5}
                if period is None:
                    record.update(status="failed", axes=None, failure_code="period_search_failed")
                else:
                    try:
                        curve = replace(selected(row, configs[condition], repeat), period_hours=float(period))
                        meta = input_metadata(curve, curve.period_hours)
                        record.update(input_sha256=input_hash(curve),
                                      metadata={k: meta[k] for k in ("n_points", "n_sessions", "merged_blocks",
                                                                     "time_span_days")
                                                if k in meta})
                        record.update(status="ok", axes=predict_curve(predictor, curve, oid))
                    except (ValueError, RuntimeError) as exc:
                        record.update(status="failed", axes=None, failure_code=f"{type(exc).__name__}: {exc}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                record["runtime_s"] = time.monotonic() - started
                append_jsonl(stream, record)
                done.add(cell)
                written += 1
            del predictor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    remaining = sum(c not in done for c in cells)
    print(json.dumps({"mode": args.mode, "written": written, "remaining": remaining}), flush=True)
    if not remaining:
        marker = target.with_suffix(f".folds-{''.join(map(str, args.folds))}.complete")
        if not marker.exists():
            marker.write_text(json.dumps({"cells": len(cells), "finished_epoch": time.time()}) + "\n")
    return 0


def axial_angle_deg(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape == b.shape and np.array_equal(a, b):
        return 0.0  # bit-identical axes
    # atan2(|a x b|, |a . b|) is exact near zero, unlike arccos of a rounded dot product.
    cross = np.linalg.norm(np.cross(a, b), axis=1)
    dot = np.abs(np.sum(a * b, axis=1))
    return float(np.degrees(np.arctan2(cross, dot)).max())


def command_validate(args) -> int:
    """DAMIT-period predictions against the published sealed predictions."""
    records = read_jsonl(OUT / "predictions" / "damit.jsonl")
    published_rows = {(r["object_id"], r["condition"], r["repeat"]): r
                      for r in json.loads((RUN / "scored-rows.json").read_text())["payload"]
                      if r.get("status") != "ineligible"}
    exact, worst, compared, mismatched = 0, 0.0, 0, []
    for r in records:
        cell = (r["object_id"], r["condition"], r["repeat"])
        p = json.loads((RUN / "predictions" / r["object_id"] / r["condition"] /
                        f"repeat-{r['repeat']}.json").read_text())["payload"]
        compared += 1
        if p["status"] != r["status"]:
            mismatched.append(cell)
            continue
        if r["status"] != "ok":
            continue
        if p["input_sha256"] != r["input_sha256"]:
            mismatched.append(cell)
        exact += p["axes"] == r["axes"]
        worst = max(worst, axial_angle_deg(p["axes"], r["axes"]))
    # Per-condition fold means (published scored rows restricted to the same cells).
    from lc_pipeline.k3.reliability_study import oracle_error
    from lc_pipeline.v2.data import load_catalog
    lock = published_lock()
    catalog = load_catalog(Path(lock["catalog"]["path"]))
    by_condition: dict[str, list[tuple[float, float]]] = {}
    for r in records:
        cell = (r["object_id"], r["condition"], r["repeat"])
        new = oracle_error(r["axes"], catalog[r["object_id"]].solution_vectors) if r["status"] == "ok" else 90.0
        by_condition.setdefault(r["condition"], []).append((published_rows[cell]["error_deg"], new))
    conditions = {k: {"n": len(v), "published_mean_deg": float(np.mean([a for a, _ in v])),
                      "rerun_mean_deg": float(np.mean([b for _, b in v])),
                      "max_abs_difference_deg": float(max(abs(a - b) for a, b in v))}
                  for k, v in sorted(by_condition.items())}
    result = {"compared_cells": compared, "bit_identical_axes": exact, "max_axis_angle_deg": worst,
              "status_or_input_mismatches": mismatched, "conditions": conditions,
              "folds": sorted({r["fold"] for r in records})}
    target = OUT / "validation-damit-period.json"
    with target.open("x") as stream:
        stream.write(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "conditions"}))
    return 0


# ----------------------------------------------------------------------------- withheld test

def withheld_root(mode: str) -> Path:
    return OUT / "withheld" / mode


def withheld_row(row: dict, period: float) -> dict:
    return {**row, "period_hours": float(period)}


def withheld_periods(mode: str, lock: dict, ids) -> dict[tuple[str, str], float | None]:
    by_id = {r["object_id"]: r for r in lock["objects"]}
    if mode == "damit":
        return {(oid, s): by_id[oid]["period_hours"] for oid in ids for s in WITHHELD_SAMPLINGS}
    periods = own_periods("withheld")
    return {(oid, s): periods[(oid, f"withheld-{s}", 0)] for oid in ids for s in WITHHELD_SAMPLINGS}


def command_withheld_candidates(args) -> int:
    """K3 candidates on the retained data (GPU); stored where fit_cell reads them."""
    import torch

    from lc_pipeline.k3.bundle import K3EnsemblePredictor
    from lc_pipeline.k3.reliability_study import checkpoint_read, holdout_input, input_hash, predict_curve, seal
    lock, schedule = published_lock(), published_schedule()
    by_id = {r["object_id"]: r for r in lock["objects"]}
    ids = args.ids or schedule["holdout_ids"]
    periods = withheld_periods(args.mode, lock, ids)
    root = withheld_root(args.mode)
    torch.set_num_threads(6)
    for fold in range(5):
        todo = [oid for oid in ids if by_id[oid]["fold"] == fold]
        if not todo:
            continue
        predictor = K3EnsemblePredictor(lock["bundles"][fold]["directory"], device=args.device)
        for oid in todo:
            for sampling in WITHHELD_SAMPLINGS:
                period = periods[(oid, sampling)]
                if period is None:
                    continue
                row = withheld_row(by_id[oid], period)
                directory = root / "fits" / oid / sampling / "k3"
                directory.mkdir(parents=True, exist_ok=True)
                if checkpoint_read(directory / "prediction.json") is not None:
                    continue
                curve, _ = holdout_input(root, row, sampling)
                axes = predict_curve(predictor, curve, oid)
                seal(directory / "prediction.json", {"axes": axes, "input_sha256": input_hash(curve),
                                                     "fold": row["fold"]})
                print(oid, sampling, f"{period:.6f}", flush=True)
        del predictor
        torch.cuda.empty_cache()
    return 0


def command_withheld_fits(args) -> int:
    """Both convexinv arms at the period of the retained data (CPU, 6 workers)."""
    from lc_pipeline.k3.reliability_study import fit_cell
    lock, schedule = published_lock(), published_schedule()
    by_id = {r["object_id"]: r for r in lock["objects"]}
    ids = args.ids or schedule["holdout_ids"]
    periods = withheld_periods(args.mode, lock, ids)
    root = withheld_root(args.mode)
    solver = Path(lock["solver"]["executable"]["path"])
    if sha256_file(solver) != lock["solver"]["executable"]["sha256"]:
        raise ValueError("solver binary changed")
    minimal = {"solver": lock["solver"], "solver_settings": lock["solver_settings"], "atlases": lock["atlases"]}
    deadline = time.monotonic() + args.max_seconds
    arms = args.arms or list(WITHHELD_ARMS)
    samplings = args.samplings or list(WITHHELD_SAMPLINGS)
    for oid in ids:
        for sampling in samplings:
            period = periods[(oid, sampling)]
            if period is None:
                continue
            for arm in arms:
                if time.monotonic() >= deadline:
                    print(json.dumps({"stopped": "max-seconds"}), flush=True)
                    return 0
                row = withheld_row(by_id[oid], period)
                started = time.monotonic()
                value = fit_cell(root, row, sampling, arm, minimal, None, _NoBudget())
                print(oid, sampling, arm, value["status"], f"{time.monotonic() - started:.1f}s", flush=True)
    marker = root / "fits.complete"
    if not args.ids and not args.arms and not args.samplings and not marker.exists():
        marker.write_text(json.dumps({"finished_epoch": time.time()}) + "\n")
    return 0


# ----------------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("search")
    s.add_argument("--scope", choices=("main", "withheld"), required=True)
    s.add_argument("--max-seconds", type=float, default=3300)
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--limit", type=int, default=0, help="only the first N cells (timing pilot)")
    p = sub.add_parser("predict")
    p.add_argument("--mode", choices=("damit", "own"), required=True)
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--max-seconds", type=float, default=3300)
    p.add_argument("--device", default="cuda")
    sub.add_parser("validate")
    for name in ("withheld-candidates", "withheld-fits"):
        w = sub.add_parser(name)
        w.add_argument("--mode", choices=("damit", "own"), required=True)
        w.add_argument("--ids", nargs="*")
        w.add_argument("--device", default="cuda")
        w.add_argument("--max-seconds", type=float, default=1e9)
        w.add_argument("--arms", nargs="*", choices=WITHHELD_ARMS)
        w.add_argument("--samplings", nargs="*", choices=WITHHELD_SAMPLINGS)
    args = parser.parse_args()
    if args.command == "search":
        return command_search(args)
    if args.command == "predict":
        return command_predict(args)
    if args.command == "validate":
        return command_validate(args)
    if args.command == "withheld-candidates":
        return command_withheld_candidates(args)
    if args.command == "withheld-fits":
        return command_withheld_fits(args)
    raise AssertionError


if __name__ == "__main__":
    sys.exit(main())
