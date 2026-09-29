"""End-to-end period candidate construction and quick physical screening.

The candidate and screening stages do not read published periods or poles.
References enter only the separate evaluation stage.  Existing search results
are reused for the ordinary 170-object sample; an unresolved object is rerun
with the sparse-survey window fallback.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from physical import BASE, CATALOG, ROOT, SOURCE, STANDARD, load_models, preflight
from search import estimate, read_curves

METHODS = ("fourier_gpu", "pdm")
# Internal DAMIT catalogue identifiers for Asterope, Jena, Veveri, and Peitho.
# These are not MPC asteroid numbers.
HARD_PILOT_OBJECTS = ("asteroid_660", "asteroid_709", "asteroid_1024", "asteroid_2516")
PERIOD_MIN_HOURS = 2.0
PERIOD_MAX_HOURS = 100.0
MERGE_RELATIVE_TOLERANCE = 0.01
AMBIGUITY_CHI2_RATIO = 1.10


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_period(value) -> float | None:
    try:
        period = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(period) or not PERIOD_MIN_HOURS <= period <= PERIOD_MAX_HOURS:
        return None
    return period


def alias_candidates(method_rows: dict[str, dict]) -> list[dict]:
    """Expand the top Fourier and PDM periods to P/2, P, and 2P."""
    raw = []
    for method in METHODS:
        row = method_rows.get(method, {})
        ranked = row.get("ranked_periods", [])
        base = valid_period(ranked[0]) if ranked else None
        if base is None:
            continue
        source_candidate = next(
            (item for item in row.get("candidates", [])
             if math.isclose(float(item.get("period", math.nan)), base, rel_tol=1e-10)),
            {},
        )
        base_step = source_candidate.get("frequency_step")
        for factor, label in ((0.5, "P/2"), (1.0, "P"), (2.0, "2P")):
            period = valid_period(base * factor)
            if period is None:
                continue
            step = None
            if base_step is not None and math.isfinite(float(base_step)) and float(base_step) > 0:
                # Frequency transforms as f/factor when period transforms as P*factor.
                step = float(base_step) / factor
            raw.append({
                "period_hours": period,
                "frequency_step_per_day": step,
                "sources": [{"method": method, "base_period_hours": base, "alias": label, "factor": factor}],
            })
    merged = []
    for candidate in sorted(raw, key=lambda item: (item["period_hours"], item["sources"][0]["method"])):
        match = next(
            (item for item in merged
             if abs(candidate["period_hours"] / item["period_hours"] - 1) <= MERGE_RELATIVE_TOLERANCE),
            None,
        )
        if match is None:
            candidate["candidate_index"] = len(merged)
            merged.append(candidate)
        else:
            match["sources"].extend(candidate["sources"])
            steps = [value for value in (match["frequency_step_per_day"], candidate["frequency_step_per_day"])
                     if value is not None]
            match["frequency_step_per_day"] = min(steps) if steps else None
    return merged


def load_search_rows(path: Path) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = {}
    for row in map(json.loads, path.read_text().splitlines()):
        if row.get("method") not in METHODS or row.get("repeat") != 0 or row.get("mode") != "warm":
            continue
        oid = row["object_id"]
        if row["method"] in grouped.setdefault(oid, {}):
            raise ValueError(f"duplicate warm repeat-zero row for {oid}/{row['method']}")
        grouped[oid][row["method"]] = row
    return grouped


def build_candidates(search_results: Path, output: Path, timeout: float) -> None:
    if output.exists():
        raise FileExistsError(f"output exists: {output}")
    catalog = {row["object_id"]: row for row in map(json.loads, (CATALOG / "catalog.jsonl").read_text().splitlines())}
    grouped = load_search_rows(search_results)
    expected = sorted(
        {oid for fold in json.loads((CATALOG / "publication-splits-v2.3.json").read_text())["folds"]
         for oid in fold["test_ids"]}
    )
    if len(expected) != 170:
        raise ValueError("expected the fixed 170-object test union")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", buffering=1) as stream:
        for oid in expected:
            rows = dict(grouped.get(oid, {}))
            fallback = {}
            for method in METHODS:
                row = rows.get(method)
                if row is not None and row.get("status") == "ok" and row.get("ranked_periods"):
                    continue
                lightcurve = ROOT / "damit-20250610T000301Z" / catalog[oid]["lightcurve"]["source_path"]
                started = time.perf_counter()
                replacement = estimate(read_curves(lightcurve), method, timeout=timeout, sparse_fallback=True)
                replacement["seconds"] = time.perf_counter() - started
                replacement.update(object_id=oid, method=method, repeat=0, mode="warm")
                rows[method] = replacement
                fallback[method] = replacement.get("input_regime", "unresolved")
            candidates = alias_candidates(rows)
            seconds = sum(float(rows[m].get("seconds", 0.0)) for m in METHODS if m in rows)
            result = {
                "schema": "delphi.end-to-end-period-candidates.v1",
                "object_id": oid,
                "status": "ok" if candidates else "unresolved",
                "candidates": candidates,
                "candidate_count": len(candidates),
                "search_seconds": seconds,
                "method_status": {m: rows.get(m, {}).get("status", "missing") for m in METHODS},
                "fallback": fallback,
                "selection_uses_reference": False,
            }
            if not candidates:
                result["reason"] = "Fourier and PDM produced no valid period candidate"
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            print(oid, result["status"], len(candidates), fallback, flush=True)
    metadata = output.with_suffix(".metadata.json")
    metadata.write_text(json.dumps({
        "schema": "delphi.end-to-end-period-candidates-metadata.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "search_results": str(search_results),
        "search_results_sha256": digest(search_results),
        "script_sha256": digest(Path(__file__)),
        "search_source_sha256": digest(Path(__file__).with_name("search.py")),
        "methods": list(METHODS),
        "aliases": ["P/2", "P", "2P"],
        "period_bounds_hours": [PERIOD_MIN_HOURS, PERIOD_MAX_HOURS],
        "merge_relative_tolerance": MERGE_RELATIVE_TOLERANCE,
        "reference_data_used_for_selection": False,
    }, indent=2) + "\n")


def pilot_ids(all_ids: list[str]) -> list[str]:
    missing = sorted(set(HARD_PILOT_OBJECTS) - set(all_ids))
    if missing:
        raise ValueError(f"hard pilot objects missing: {missing}")
    remaining = [oid for oid in all_ids if oid not in HARD_PILOT_OBJECTS]
    remaining.sort(key=lambda oid: hashlib.sha256(("delphi-e2e-pilot-20260921:" + oid).encode()).hexdigest())
    return list(HARD_PILOT_OBJECTS) + remaining[:16]


def valid_fit(row: dict) -> bool:
    return (
        row.get("return_code") == 0
        and not row.get("timed_out", False)
        and all(row.get(key) is not None and math.isfinite(float(row[key]))
                for key in ("relative_rms_from_output", "chi2", "final_period_hours",
                            "final_lambda_deg", "final_beta_deg"))
    )


def choose_candidate(fits: list[dict]) -> dict:
    by_candidate: dict[int, list[dict]] = {}
    for fit in fits:
        if valid_fit(fit):
            by_candidate.setdefault(int(fit["candidate_index"]), []).append(fit)
    best_per_candidate = [
        min(rows, key=lambda row: (float(row["chi2"]), int(row["start_index"])))
        for rows in by_candidate.values()
    ]
    best_per_candidate.sort(key=lambda row: (
        float(row["chi2"]), int(row["candidate_index"]), int(row["start_index"])
    ))
    if not best_per_candidate:
        return {"status": "no_valid_fit", "ambiguous": True, "selected": None, "runner_up": None}
    selected = best_per_candidate[0]
    runner = best_per_candidate[1] if len(best_per_candidate) > 1 else None
    ratio = None if runner is None else float(runner["chi2"]) / max(float(selected["chi2"]), 1e-300)
    return {
        "status": "ambiguous" if ratio is not None and ratio <= AMBIGUITY_CHI2_RATIO else "selected",
        "ambiguous": bool(ratio is not None and ratio <= AMBIGUITY_CHI2_RATIO),
        "selected": selected,
        "runner_up": runner,
        "runner_up_chi2_ratio": ratio,
        "valid_candidate_count": len(best_per_candidate),
    }


def _run_fit(lightcurve: Path, candidate: dict, start_index: int, pole: tuple[float, float],
             directory: Path, timeout: float) -> dict:
    sys.path.insert(0, str(BASE / "source"))
    from lc_pipeline.v2.convexinv import (
        ConvexinvParameters,
        run_convexinv,
        write_convexinv_parameters,
    )
    run_dir = directory / f"candidate-{candidate['candidate_index']:02d}" / f"start-{start_index:02d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    parameters = write_convexinv_parameters(run_dir / "parameters.txt", ConvexinvParameters(
        lambda_deg=pole[0], beta_deg=pole[1], period_hours=candidate["period_hours"],
        free_period=False, iteration_stop_condition=50.0,
    ))
    result = run_convexinv(
        executable=SOURCE / "convexinv", source_root=SOURCE, lightcurve_file=lightcurve,
        parameter_file=parameters, output_directory=run_dir, timeout_seconds=timeout,
        stdout_log_path=run_dir / "stdout.log", stderr_log_path=run_dir / "stderr.log",
    )
    row = dataclasses.asdict(result)
    row.update(candidate_index=candidate["candidate_index"], initial_period_hours=candidate["period_hours"],
               start_index=start_index, initial_lambda_deg=pole[0], initial_beta_deg=pole[1])
    return row


def screen_object(row: dict, catalog_row: dict, directory: Path, timeout: float) -> dict:
    started = time.perf_counter()
    lightcurve = ROOT / "damit-20250610T000301Z" / catalog_row["lightcurve"]["source_path"]
    info = preflight(lightcurve)
    directory.mkdir(parents=True, exist_ok=False)
    tasks = [(candidate, index, pole) for candidate in row["candidates"] for index, pole in enumerate(STANDARD)]
    fits = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(_run_fit, lightcurve, candidate, index, pole, directory, timeout)
                   for candidate, index, pole in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                fits.append(future.result())
            # One failed solver start must remain visible without discarding
            # the other independently scheduled starts for this object.
            except Exception as exc:  # noqa: BLE001
                fits.append({"status": "error", "error": repr(exc)})
    decision = choose_candidate(fits)
    elapsed = time.perf_counter() - started
    selected = decision.get("selected")
    summary = {
        "schema": "delphi.end-to-end-quick-physical.v1",
        "object_id": row["object_id"],
        "status": decision["status"],
        "ambiguous": decision["ambiguous"],
        "candidate_count": len(row["candidates"]),
        "scheduled_starts": len(tasks),
        "valid_starts": sum(valid_fit(fit) for fit in fits),
        "valid_candidate_count": decision.get("valid_candidate_count", 0),
        "selected_candidate_index": None if selected is None else selected["candidate_index"],
        "selected_period_hours": None if selected is None else selected["initial_period_hours"],
        "selected_fit": selected,
        "runner_up_fit": decision.get("runner_up"),
        "runner_up_chi2_ratio": decision.get("runner_up_chi2_ratio"),
        "search_seconds": row.get("search_seconds"),
        "physical_screen_seconds": elapsed,
        "screen_plus_search_seconds": None if row.get("search_seconds") is None else elapsed + row["search_seconds"],
        "preflight": info,
        "selection_uses_reference": False,
        "raw_directory": str(directory),
    }
    (directory / "fits.jsonl").write_text("".join(json.dumps(fit, allow_nan=False) + "\n" for fit in fits))
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def run_screen(candidates_path: Path, output: Path, sample: str, timeout: float, repeats: int,
               object_ids: list[str] | None = None) -> None:
    if output.exists() or output.with_suffix(".runs").exists():
        raise FileExistsError("screen output or run directory already exists")
    rows = [json.loads(line) for line in candidates_path.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("status") == "ok"]
    all_ids = sorted(row["object_id"] for row in rows)
    if object_ids:
        unknown = sorted(set(object_ids) - set(all_ids))
        if unknown:
            raise ValueError(f"requested objects missing from candidates: {unknown}")
        selected_ids = list(dict.fromkeys(object_ids))
    else:
        selected_ids = pilot_ids(all_ids) if sample == "pilot" else all_ids
    selected = [row for row in rows if row["object_id"] in set(selected_ids)]
    selected.sort(key=lambda row: selected_ids.index(row["object_id"]))
    catalog = {row["object_id"]: row for row in map(json.loads, (CATALOG / "catalog.jsonl").read_text().splitlines())}
    output.parent.mkdir(parents=True, exist_ok=True)
    run_root = output.with_suffix(".runs")
    with output.open("x", buffering=1) as stream:
        for row in selected:
            for repeat in range(repeats):
                token = hashlib.sha256(f"{row['object_id']}/{repeat}".encode()).hexdigest()[:16]
                result = screen_object(row, catalog[row["object_id"]], run_root / token, timeout)
                result["repeat"] = repeat
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                print(row["object_id"], repeat, result["status"],
                      f"{result['physical_screen_seconds']:.2f}s", flush=True)
    output.with_suffix(".metadata.json").write_text(json.dumps({
        "schema": "delphi.end-to-end-quick-physical-metadata.v1",
        "candidate_input_sha256": digest(candidates_path),
        "script_sha256": digest(Path(__file__)),
        "sample": sample,
        "objects": selected_ids,
        "repeats": repeats,
        "fit_timeout_seconds": timeout,
        "starts_per_candidate": len(STANDARD),
        "free_period": False,
        "iteration_stop_condition": 50.0,
        "ambiguity_chi2_ratio": AMBIGUITY_CHI2_RATIO,
        "reference_data_used_for_selection": False,
    }, indent=2) + "\n")


def export_pole_input(screen_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"output exists: {output}")
    rows = [json.loads(line) for line in screen_path.read_text().splitlines() if line.strip()]
    if any(row.get("repeat") != 0 for row in rows):
        rows = [row for row in rows if row.get("repeat") == 0]
    with output.open("x") as stream:
        for row in rows:
            period = row.get("selected_period_hours")
            status = "ok" if period is not None else "unresolved"
            exported = {
                "object_id": row["object_id"], "method": "end_to_end", "status": status,
                "ranked_periods": [] if period is None else [period],
                "candidates": [] if period is None else [{"period": period, "frequency_step": 1e-6}],
                "seconds": row.get("screen_plus_search_seconds"),
                "retained_count": 1,
            }
            stream.write(json.dumps(exported, allow_nan=False) + "\n")


def resummarize_screen(screen_path: Path, output: Path) -> None:
    """Reapply the recorded selector to completed raw fits without rerunning them."""
    if output.exists():
        raise FileExistsError(f"output exists: {output}")
    rows = [json.loads(line) for line in screen_path.read_text().splitlines() if line.strip()]
    with output.open("x", buffering=1) as stream:
        for row in rows:
            raw = Path(row["raw_directory"]) / "fits.jsonl"
            fits = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
            decision = choose_candidate(fits)
            selected = decision.get("selected")
            row.update(
                status=decision["status"], ambiguous=decision["ambiguous"],
                valid_candidate_count=decision.get("valid_candidate_count", 0),
                selected_candidate_index=None if selected is None else selected["candidate_index"],
                selected_period_hours=None if selected is None else selected["initial_period_hours"],
                selected_fit=selected, runner_up_fit=decision.get("runner_up"),
                runner_up_chi2_ratio=decision.get("runner_up_chi2_ratio"),
            )
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    output.with_suffix(".metadata.json").write_text(json.dumps({
        "schema": "delphi.end-to-end-resummarized-screen-metadata.v1",
        "source_sha256": digest(screen_path),
        "script_sha256": digest(Path(__file__)),
        "selection_metric": "minimum solver chi-squared",
        "ambiguity_chi2_ratio": AMBIGUITY_CHI2_RATIO,
        "fits_rerun": False,
        "reference_data_used_for_selection": False,
    }, indent=2) + "\n")


def score_periods(candidates_path: Path, output: Path, sample: str,
                  object_ids: list[str] | None = None) -> None:
    """Record frozen DeLPHI score-map diagnostics for every period candidate."""
    if output.exists():
        raise FileExistsError(f"output exists: {output}")
    rows = [json.loads(line) for line in candidates_path.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("status") == "ok"]
    all_ids = sorted(row["object_id"] for row in rows)
    if object_ids:
        missing = sorted(set(object_ids) - set(all_ids))
        if missing:
            raise ValueError(f"requested objects missing from candidates: {missing}")
        selected_ids = list(dict.fromkeys(object_ids))
    else:
        selected_ids = pilot_ids(all_ids) if sample == "pilot" else all_ids
    selected = {row["object_id"]: row for row in rows if row["object_id"] in set(selected_ids)}
    catalog = {row["object_id"]: row for row in map(json.loads, (CATALOG / "catalog.jsonl").read_text().splitlines())}
    folds = {oid: fold["fold"] for fold in json.loads((CATALOG / "publication-splits-v2.3.json").read_text())["folds"]
             for oid in fold["test_ids"]}
    sys.path.insert(0, str(BASE / "source"))
    import torch
    from lc_pipeline.k3.bundle import _model_inputs
    from lc_pipeline.k3.evaluation import ensemble_score_grids, modes_from_score_grid
    from lc_pipeline.k3.inference import score_axial_grid
    from lc_pipeline.v2.data import parse_damit_lightcurve
    from lc_pipeline.v2.preprocessing import KnownPeriod
    if not torch.cuda.is_available():
        raise RuntimeError("period score comparison requires CUDA")
    torch.set_num_threads(1)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", buffering=1) as stream:
        for fold in range(5):
            fold_ids = [oid for oid in selected_ids if folds[oid] == fold]
            if not fold_ids:
                continue
            models, bindings = load_models(fold)
            for oid in fold_ids:
                row = selected[oid]
                lightcurve = ROOT / "damit-20250610T000301Z" / catalog[oid]["lightcurve"]["source_path"]
                epochs = parse_damit_lightcurve(lightcurve)
                scores = []
                for candidate in row["candidates"]:
                    started = time.perf_counter()
                    inputs = _model_inputs(
                        epochs, KnownPeriod(candidate["period_hours"], "end-to-end-period-candidate"),
                        torch.device("cuda"),
                    )
                    with torch.inference_mode():
                        grids = np.stack([score_axial_grid(model, inputs, chunk_size=1024) for model in models])
                    mean_grid = ensemble_score_grids(grids[:, None, :])[0]
                    modes = modes_from_score_grid(mean_grid)
                    torch.cuda.synchronize()
                    probabilities = 1 / (1 + np.exp(-np.clip(mean_grid, -50, 50)))
                    scores.append({
                        "candidate_index": candidate["candidate_index"],
                        "period_hours": candidate["period_hours"],
                        "peak_logit": float(max(mode.score_logit for mode in modes)),
                        "mean_mode_logit": float(np.mean([mode.score_logit for mode in modes])),
                        "grid_mean_logit": float(np.mean(mean_grid)),
                        "grid_std_logit": float(np.std(mean_grid)),
                        "peak_probability": float(np.max(probabilities)),
                        "seconds": time.perf_counter() - started,
                    })
                result = {
                    "schema": "delphi.end-to-end-period-score-diagnostics.v1",
                    "object_id": oid, "fold": fold, "scores": scores,
                    "checkpoint_bindings": bindings, "selection_uses_reference": False,
                }
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                print(oid, len(scores), f"{sum(item['seconds'] for item in scores):.2f}s", flush=True)
            del models
            torch.cuda.empty_cache()
    output.with_suffix(".metadata.json").write_text(json.dumps({
        "schema": "delphi.end-to-end-period-score-diagnostics-metadata.v1",
        "candidate_input_sha256": digest(candidates_path),
        "script_sha256": digest(Path(__file__)),
        "sample": sample,
        "objects": selected_ids,
        "reference_data_used_for_selection": False,
        "note": "Score-map diagnostics are exploratory and are not calibrated period probabilities.",
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    candidates = sub.add_parser("candidates")
    candidates.add_argument("--search-results", type=Path, required=True)
    candidates.add_argument("--output", type=Path, required=True)
    candidates.add_argument("--timeout", type=float, default=180.0)
    screen = sub.add_parser("screen")
    screen.add_argument("--candidates", type=Path, required=True)
    screen.add_argument("--output", type=Path, required=True)
    screen.add_argument("--sample", choices=("pilot", "all"), default="pilot")
    screen.add_argument("--fit-timeout", type=float, default=30.0)
    screen.add_argument("--repeats", type=int, default=1)
    screen.add_argument("--object-id", action="append", dest="object_ids")
    pole = sub.add_parser("pole-input")
    pole.add_argument("--screen", type=Path, required=True)
    pole.add_argument("--output", type=Path, required=True)
    scoring = sub.add_parser("score-periods")
    scoring.add_argument("--candidates", type=Path, required=True)
    scoring.add_argument("--output", type=Path, required=True)
    scoring.add_argument("--sample", choices=("pilot", "all"), default="pilot")
    scoring.add_argument("--object-id", action="append", dest="object_ids")
    resummary = sub.add_parser("resummarize")
    resummary.add_argument("--screen", type=Path, required=True)
    resummary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if getattr(args, "timeout", 1) <= 0 or getattr(args, "fit_timeout", 1) <= 0:
        parser.error("timeouts must be positive")
    if getattr(args, "repeats", 1) <= 0:
        parser.error("repeats must be positive")
    if args.command == "candidates":
        build_candidates(args.search_results, args.output, args.timeout)
    elif args.command == "screen":
        run_screen(args.candidates, args.output, args.sample, args.fit_timeout, args.repeats, args.object_ids)
    elif args.command == "pole-input":
        export_pole_input(args.screen, args.output)
    elif args.command == "score-periods":
        score_periods(args.candidates, args.output, args.sample, args.object_ids)
    else:
        resummarize_screen(args.screen, args.output)


if __name__ == "__main__":
    main()
