"""Reference-blind benchmark of DeLPHI-assisted and classical pole workflows.

Both workflows receive the same lightcurves and fixed published period.  They
first run six convex-inversion starts and use one shared, frozen diagnostic to
decide whether to run the complete 20-degree pole grid.  DAMIT reference axes
are absent from the execution lock and are opened only by :func:`score`.
"""

from __future__ import annotations

import concurrent.futures
import heapq
import json
import math
import os
import random
import resource
import subprocess
import time
from pathlib import Path

import numpy as np

from . import grid_benchmark as grid

SCHEMA = "delphi.adaptive-pole-workflow.v1"
RULE_SCHEMA = "delphi.adaptive-pole-workflow-rule.v1"
EXECUTION_SCHEMA = "delphi.adaptive-pole-workflow-execution.v1"
SCORE_SCHEMA = "delphi.adaptive-pole-workflow-score.v1"
ARMS = ("classical", "delphi")
SUPPORT_THRESHOLDS = (1 / 3, 1 / 2, 2 / 3)
GAP_THRESHOLDS = (1.005, 1.01, 1.02, 1.05, 1.10)
SETTINGS = {
    "period": "same_fixed_published_period_in_both_workflows",
    "initial_starts": 6,
    "fallback_grid_spacing_degrees": 20,
    "fallback_grid_starts": 146,
    "workers": 6,
    "repeats": 3,
    "cluster_radius_degrees": 20.0,
    "selection": "minimum_final_relative_rms_over_every_completed_fit",
    "fallback_on_any_initial_failure": True,
    "order_seed": 20260922,
    "bootstrap_seed": 20260922,
    "bootstrap_resamples": 10000,
    "mean_error_margin_degrees": 3.0,
    "within20_margin": 0.05,
    "completion_margin": 0.05,
    "rms_ratio_limit": 1.01,
    "minimum_mean_speed_ratio": 1.25,
    "minimum_speed_lower_bound": 1.0,
    "case_timeout_seconds": 9000,
    "full_run_budget_seconds": 604800,
}


class WorkflowBenchmarkError(ValueError):
    """Raised when a benchmark artifact is incomplete or has changed."""


def _read(path: str | Path) -> dict:
    value = grid.read_json(path)
    return value


def _binding(path: str | Path) -> dict:
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": grid.digest(path)}


def _verify(binding: dict) -> Path:
    path = Path(binding["path"])
    if grid.digest(path) != binding["sha256"]:
        raise WorkflowBenchmarkError(f"bound file changed: {path}")
    return path


def _write_once(path: str | Path, value: dict) -> None:
    grid.write_once(path, value)


def _runtime_sources(code_root: Path) -> dict[str, str]:
    paths = sorted((code_root / "lc_pipeline").rglob("*.py"))
    paths.append(code_root / "repro/run_delphi_workflow_benchmark.py")
    return {str(path.relative_to(code_root)): grid.digest(path) for path in paths}


def _environment_equivalent(recorded: dict, current: dict) -> bool:
    """Compare environments while tolerating WSL's case-only mount spelling."""
    old = dict(recorded)
    new = dict(current)
    old_interpreter = old.pop("interpreter", None)
    new_interpreter = new.pop("interpreter", None)
    if old != new or not isinstance(old_interpreter, dict) or not isinstance(new_interpreter, dict):
        return False
    if old_interpreter.get("sha256") != new_interpreter.get("sha256"):
        return False
    try:
        return Path(old_interpreter["path"]).samefile(Path(new_interpreter["path"]))
    except (KeyError, OSError):
        return False


def _validate_parent(parent_path: Path, *, require_environment: bool = True) -> dict:
    """Validate the prior evidence lock without requiring its old source tree."""
    parent = _read(parent_path)
    if parent.get("schema") != grid.SCHEMA or len(parent.get("objects", [])) != 170:
        raise WorkflowBenchmarkError("invalid parent grid-benchmark lock")
    if require_environment and not _environment_equivalent(
        parent.get("environment", {}), grid._environment(parent["device"])
    ):
        raise WorkflowBenchmarkError("timing environment differs from the parent benchmark")
    for binding in parent["inputs"].values():
        _verify(binding)
    for row in parent["objects"]:
        _verify(row["lightcurve"])
    for bundle in parent["bundles"]:
        for binding in bundle["files"]:
            _verify(binding)
    if grid._validated_capacity(
        Path(parent["solver"]["capacity_report"]["path"]), parent["objects"]
    ) != parent["solver"]:
        raise WorkflowBenchmarkError("solver or capacity evidence changed")
    return parent


def _dot(a: object, b: object) -> float:
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    return float(np.dot(av / np.linalg.norm(av), bv / np.linalg.norm(bv)))


def axial_angle_degrees(a: object, b: object) -> float:
    return math.degrees(math.acos(float(np.clip(abs(_dot(a, b)), -1.0, 1.0))))


def delphi_starts(axes: object) -> np.ndarray:
    candidates = grid.unit_vectors(axes)
    if candidates.shape != (3, 3):
        raise WorkflowBenchmarkError("DeLPHI must return exactly three axes")
    starts = np.concatenate((candidates, -candidates))
    unique: list[np.ndarray] = []
    for vector in starts:
        if not any(np.linalg.norm(vector - prior) <= 1e-10 for prior in unique):
            unique.append(vector)
    if len(unique) != 6:
        raise WorkflowBenchmarkError("DeLPHI axes do not produce six directed starts")
    unique.sort(key=lambda vector: tuple(np.round(vector, 12)))
    return np.stack(unique)


def initial_starts(arm: str, axes: object | None = None) -> np.ndarray:
    if arm == "classical":
        return grid.starts_for_arm("standard6")
    if arm == "delphi" and axes is not None:
        return delphi_starts(axes)
    raise WorkflowBenchmarkError("invalid workflow arm or missing DeLPHI axes")


def _fit_key(fit: dict) -> tuple[float, int, int]:
    return (
        float(fit["final_relative_rms"]),
        0 if fit.get("stage") == "initial" else 1,
        int(fit["start_index"]),
    )


def cluster_diagnostic(fits: list[dict], *, expected_initial: int = 6) -> dict:
    """Summarize convergence using only fitted poles and residual RMS."""
    ordered = sorted(fits, key=_fit_key)
    if not ordered:
        return {
            "valid_initial_fits": 0,
            "support_count": 0,
            "support_fraction": 0.0,
            "outside_rms_ratio": None,
            "cluster_count": 0,
        }
    clusters: list[dict] = []
    for fit in ordered:
        for cluster in clusters:
            if axial_angle_degrees(fit["axis"], cluster["representative"]["axis"]) <= SETTINGS[
                "cluster_radius_degrees"
            ]:
                cluster["members"].append(fit)
                break
        else:
            clusters.append({"representative": fit, "members": [fit]})
    best = clusters[0]
    outside = [member for cluster in clusters[1:] for member in cluster["members"]]
    gap = (
        min(float(item["final_relative_rms"]) for item in outside)
        / float(best["representative"]["final_relative_rms"])
        if outside
        else None
    )
    return {
        "valid_initial_fits": len(ordered),
        "expected_initial_fits": expected_initial,
        "support_count": len(best["members"]),
        "support_fraction": len(best["members"]) / len(ordered),
        "outside_rms_ratio": gap,
        "cluster_count": len(clusters),
    }


def requires_fallback(diagnostic: dict, rule: dict) -> bool:
    if rule.get("schema") != RULE_SCHEMA:
        raise WorkflowBenchmarkError("invalid adaptive rule")
    if rule.get("always_fallback") is True:
        return True
    if diagnostic["valid_initial_fits"] != SETTINGS["initial_starts"]:
        return True
    if diagnostic["support_fraction"] + 1e-12 < float(rule["minimum_support_fraction"]):
        return True
    gap = diagnostic["outside_rms_ratio"]
    return gap is not None and gap + 1e-12 < float(rule["minimum_outside_rms_ratio"])


def _load_records(case: Path) -> list[dict]:
    return [_read(path) for path in sorted(case.glob("start-*/record.json"))]


def _successful_fits(records: list[dict], stage: str) -> list[dict]:
    fits = []
    for record in records:
        if record.get("selected_fit") is not None:
            fits.append({**record["selected_fit"], "stage": stage})
    return fits


def _parallel_starts(
    job: dict,
    starts: np.ndarray,
    observed: tuple,
    lock: dict,
    directory: Path,
) -> tuple[list[dict], float]:
    directory.mkdir()
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=SETTINGS["workers"]) as pool:
        futures = [
            pool.submit(
                grid._native_start,
                job,
                index,
                vector,
                observed,
                lock["solver"],
                lock["solver_settings"],
                directory,
            )
            for index, vector in enumerate(starts)
        ]
        records = [future.result() for future in futures]
    return records, time.perf_counter() - started


def worker_case(lock: dict, job: dict, directory: Path) -> dict:
    """Run one complete user workflow without opening reference records."""
    from .v2.convexinv import _parse_lightcurve_brightness
    from .v2.preprocessing import KnownPeriod

    started = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=False)
    _verify(job["lightcurve"])
    observed = _parse_lightcurve_brightness(Path(job["lightcurve"]["path"]))
    axes = None
    prediction = None
    inference_seconds = 0.0
    if job["arm"] == "delphi":
        from .k3.damit import _read_lc

        neural_started = time.perf_counter()
        predictor = grid._predictor(lock, job["fold"])
        epochs = _read_lc(Path(job["lightcurve"]["path"]))
        prediction = predictor.predict(
            epochs,
            known_period=KnownPeriod(
                hours=job["period_hours"], provenance="frozen-supplied-known-period"
            ),
            object_id=job["object_id"],
        )
        if predictor.device.type == "cuda":
            import torch

            torch.cuda.synchronize(predictor.device)
        inference_seconds = time.perf_counter() - neural_started
        axes = [item["axis_xyz"] for item in prediction["axes"]]
        _write_once(directory / "prediction.json", prediction)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    first_records, initial_seconds = _parallel_starts(
        job, initial_starts(job["arm"], axes), observed, lock, directory / "initial"
    )
    initial_fits = _successful_fits(first_records, "initial")
    diagnostic = cluster_diagnostic(initial_fits)
    fallback = requires_fallback(diagnostic, lock["rule"])
    fallback_records: list[dict] = []
    fallback_seconds = 0.0
    if fallback:
        fallback_records, fallback_seconds = _parallel_starts(
            job, grid.pole_grid(20), observed, lock, directory / "fallback"
        )
    fallback_fits = _successful_fits(fallback_records, "fallback")
    fits = initial_fits + fallback_fits
    selected = min(fits, key=_fit_key) if fits else None
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = {key: job[key] for key in ("object_id", "fold", "repeat_index", "arm")}
    result.update(
        {
            "completed": selected is not None,
            "selected_fit": selected,
            "fallback_used": fallback,
            "diagnostic": diagnostic,
            "initial_starts_requested": 6,
            "initial_valid_starts": len(initial_fits),
            "fallback_starts_requested": 146 if fallback else 0,
            "fallback_valid_starts": len(fallback_fits),
            "inference_and_load_seconds": inference_seconds,
            "initial_stage_wall_seconds": initial_seconds,
            "fallback_stage_wall_seconds": fallback_seconds,
            "solver_cpu_seconds": (
                after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
            ),
            "prediction_sha256": (
                grid.digest(directory / "prediction.json") if prediction is not None else None
            ),
        }
    )
    result["worker_wall_seconds"] = time.perf_counter() - started
    _write_once(directory / "selected-result.json", result)
    return result


def worker_main(lock_path: Path, job_path: Path) -> None:
    # The parent validates every bound input, model, executable, and source once
    # before launching any timed case.  Re-hashing multi-gigabyte bundles inside
    # each fresh worker would contaminate the end-to-end timing and add hours of
    # unrelated I/O.  Workers therefore perform only structural checks here.
    lock = _read(lock_path)
    if lock.get("schema") != SCHEMA or lock.get("settings") != SETTINGS:
        raise WorkflowBenchmarkError("worker received an invalid execution lock")
    if "reference_catalog" in lock.get("inputs", {}):
        raise WorkflowBenchmarkError("worker lock contains the scoring reference")
    job = _read(job_path)
    result = worker_case(lock, job, Path(job["directory"]))
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)


def _development_ids(parent: dict) -> list[str]:
    document = _read(_verify(parent["inputs"]["development"]))
    ids = document.get("object_ids")
    if not isinstance(ids, list) or len(ids) != 30 or len(set(ids)) != 30:
        raise WorkflowBenchmarkError("invalid 30-object development manifest")
    return ids


def _reference_axes(parent: dict) -> dict[str, list[list[float]]]:
    path = _verify(parent["inputs"]["reference_catalog"])
    references: dict[str, list[list[float]]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("eligible"):
                references[row["object_id"]] = [item["vector"] for item in row["solutions"]]
    return references


def _reference_error(axis: object, references: list[list[float]]) -> float:
    return min(axial_angle_degrees(axis, reference) for reference in references)


def _makespan(seconds: list[float], workers: int = 6) -> float:
    heap = [0.0] * workers
    for value in seconds:
        earliest = heapq.heappop(heap)
        heapq.heappush(heap, earliest + float(value))
    return max(heap)


def _archived_timed_row(archive: Path, object_id: str, repeat: int, arm: str) -> dict:
    return _read(archive / "cases" / f"{object_id}-r{repeat}-{arm}-cold" / "timed-result.json")


def _archived_case(archive: Path, object_id: str, repeat: int, arm: str) -> Path:
    case = archive / "cases" / f"{object_id}-r{repeat}-{arm}-cold"
    if not case.is_dir():
        raise WorkflowBenchmarkError(f"missing archived development case: {case.name}")
    return case


def _exact_delphi_records(case: Path) -> list[dict]:
    prediction = _read(case / "prediction.json")
    axes = [item["axis_xyz"] for item in prediction["axes"]]
    starts = delphi_starts(axes)
    selected = []
    for record in _load_records(case):
        vector = np.asarray(record["initial_axis"], dtype=np.float64)
        if any(np.linalg.norm(vector - start) <= 1e-10 for start in starts):
            selected.append(record)
    if len(selected) != 6:
        raise WorkflowBenchmarkError("archived guided arm does not contain the exact six DeLPHI starts")
    return selected


def _candidate_rules() -> list[dict]:
    rules = [
        {
            "schema": RULE_SCHEMA,
            "always_fallback": False,
            "minimum_support_fraction": support,
            "minimum_outside_rms_ratio": gap,
        }
        for support in SUPPORT_THRESHOLDS
        for gap in GAP_THRESHOLDS
    ]
    rules.append(
        {
            "schema": RULE_SCHEMA,
            "always_fallback": True,
            "minimum_support_fraction": 1.0,
            "minimum_outside_rms_ratio": 1.0e9,
        }
    )
    return rules


def _geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise WorkflowBenchmarkError("geometric mean requires finite positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def tune(*, parent_lock: Path, archive: Path, output: Path) -> dict:
    """Choose one shared fallback rule using only the 30 development objects."""
    parent = _validate_parent(parent_lock, require_environment=False)
    archive = archive.resolve(strict=True)
    development = _development_ids(parent)
    references = _reference_axes(parent)
    objects: list[dict] = []
    for object_id in development:
        full_case = _archived_case(archive, object_id, 0, "classical20")
        full_fit = _read(full_case / "selected-result.json")["selected_fit"]
        arms = {}
        for arm, archived_arm in (("classical", "standard6"), ("delphi", "guided20")):
            case = _archived_case(archive, object_id, 0, archived_arm)
            records = _load_records(case) if arm == "classical" else _exact_delphi_records(case)
            fits = _successful_fits(records, "initial")
            best = min(fits, key=_fit_key)
            repeat_times = []
            for repeat in range(3):
                timed = _archived_timed_row(archive, object_id, repeat, archived_arm)
                full_timed = _archived_timed_row(archive, object_id, repeat, "classical20")
                if arm == "classical":
                    initial_time = float(timed["wall_seconds"])
                else:
                    guided_case = _archived_case(archive, object_id, repeat, "guided20")
                    exact = _exact_delphi_records(guided_case)
                    all_records = _load_records(guided_case)
                    exact_make = _makespan([r["solver_wall_seconds"] for r in exact])
                    all_make = _makespan([r["solver_wall_seconds"] for r in all_records])
                    overhead = max(
                        0.0,
                        float(timed["wall_seconds"])
                        - float(timed["inference_and_optional_load_seconds"])
                        - all_make,
                    )
                    initial_time = (
                        overhead
                        + float(timed["inference_and_optional_load_seconds"])
                        + exact_make
                    )
                full_records = _load_records(
                    _archived_case(archive, object_id, repeat, "classical20")
                )
                repeat_times.append(
                    {
                        "initial_seconds": initial_time,
                        "fallback_solver_seconds": _makespan(
                            [r["solver_wall_seconds"] for r in full_records]
                        ),
                        "archived_full_wall_seconds": float(full_timed["wall_seconds"]),
                    }
                )
            arms[arm] = {
                "fit": best,
                "diagnostic": cluster_diagnostic(fits),
                "reference_error": _reference_error(best["axis"], references[object_id]),
                "repeat_time_estimates": repeat_times,
            }
        objects.append(
            {
                "object_id": object_id,
                "full_fit": {**full_fit, "stage": "fallback"},
                "full_reference_error": _reference_error(full_fit["axis"], references[object_id]),
                "arms": arms,
            }
        )
    full_mean = sum(row["full_reference_error"] for row in objects) / len(objects)
    full_within = sum(row["full_reference_error"] <= 20 for row in objects) / len(objects)
    candidates = []
    for index, rule in enumerate(_candidate_rules()):
        arm_results = {}
        for arm in ARMS:
            errors, ratios, estimated = [], [], []
            fallbacks = 0
            for row in objects:
                initial = row["arms"][arm]
                fallback = requires_fallback(initial["diagnostic"], rule)
                fallbacks += int(fallback)
                chosen = min((initial["fit"], row["full_fit"]), key=_fit_key) if fallback else initial["fit"]
                errors.append(_reference_error(chosen["axis"], references[row["object_id"]]))
                ratios.append(
                    float(chosen["final_relative_rms"])
                    / float(row["full_fit"]["final_relative_rms"])
                )
                for timing in initial["repeat_time_estimates"]:
                    estimated.append(
                        timing["initial_seconds"]
                        + (timing["fallback_solver_seconds"] if fallback else 0.0)
                    )
            arm_results[arm] = {
                "mean_reference_error_degrees": sum(errors) / len(errors),
                "within20_fraction": sum(value <= 20 for value in errors) / len(errors),
                "geometric_mean_rms_ratio_to_full": _geometric_mean(ratios),
                "fallback_objects": fallbacks,
                "estimated_mean_seconds": sum(estimated) / len(estimated),
            }
        qualifies = all(
            result["mean_reference_error_degrees"] - full_mean
            <= SETTINGS["mean_error_margin_degrees"] + 1e-12
            and result["within20_fraction"]
            >= full_within - SETTINGS["within20_margin"] - 1e-12
            and result["geometric_mean_rms_ratio_to_full"]
            <= SETTINGS["rms_ratio_limit"] + 1e-12
            for result in arm_results.values()
        )
        candidates.append(
            {
                "candidate_index": index,
                "rule": rule,
                "arms": arm_results,
                "qualifies": qualifies,
                "combined_estimated_mean_seconds": sum(
                    value["estimated_mean_seconds"] for value in arm_results.values()
                ),
            }
        )
    qualifying = [candidate for candidate in candidates if candidate["qualifies"]]
    if not qualifying:
        raise WorkflowBenchmarkError("no fallback rule qualifies on development data")
    chosen = min(
        qualifying,
        key=lambda candidate: (
            candidate["combined_estimated_mean_seconds"],
            -float(candidate["rule"]["minimum_support_fraction"]),
            -float(candidate["rule"]["minimum_outside_rms_ratio"]),
            candidate["candidate_index"],
        ),
    )
    selected_rule = {
        **chosen["rule"],
        "development_selection": {
            "source_parent_lock_sha256": grid.digest(parent_lock),
            "source_archive": str(archive),
            "development_objects": development,
            "candidate_index": chosen["candidate_index"],
            "selection": "lowest_combined_estimated_time_among_rules_meeting_all_development_gates",
            "timing_note": "exact-six DeLPHI time reconstructed from archived inference, overhead, and six-worker makespan",
        },
    }
    report = {
        "schema": RULE_SCHEMA,
        "phase": "development_only_rule_selection",
        "settings": SETTINGS,
        "development_full_grid": {
            "objects": len(objects),
            "mean_reference_error_degrees": full_mean,
            "within20_fraction": full_within,
        },
        "selected_rule": selected_rule,
        "selected_candidate": chosen,
        "candidates": candidates,
    }
    _write_once(output, report)
    return report


def freeze(
    *, parent_lock: Path, tuning_report: Path, output: Path, scoring_output: Path
) -> tuple[dict, dict]:
    parent = _validate_parent(parent_lock)
    tuning = _read(tuning_report)
    if tuning.get("schema") != RULE_SCHEMA or tuning.get("phase") != "development_only_rule_selection":
        raise WorkflowBenchmarkError("invalid tuning report")
    rule = tuning["selected_rule"]
    if rule["development_selection"]["source_parent_lock_sha256"] != grid.digest(parent_lock):
        raise WorkflowBenchmarkError("tuning report belongs to another parent lock")
    development = set(_development_ids(parent))
    evaluation = [row for row in parent["objects"] if row["object_id"] not in development]
    if len(evaluation) != 140 or any(
        sum(row["fold"] == fold for row in evaluation) != 28 for fold in range(5)
    ):
        raise WorkflowBenchmarkError("evaluation cohort must contain 28 objects per fold")
    code_root = Path(__file__).resolve().parents[1]
    lock = {
        "schema": SCHEMA,
        "phase": "frozen_before_140_object_evaluation",
        "scope": "retrospective_fixed_period_workflow_comparison",
        "settings": SETTINGS,
        "rule": rule,
        "rule_report": _binding(tuning_report),
        "parent_lock_sha256": grid.digest(parent_lock),
        "objects": evaluation,
        "development_object_ids": sorted(development),
        "bundles": parent["bundles"],
        "device": parent["device"],
        "solver": parent["solver"],
        "solver_settings": parent["settings"],
        "environment": grid._environment(parent["device"]),
        "code_root": str(code_root),
        "runtime_sources": _runtime_sources(code_root),
        "inputs": {
            "blind_inputs": parent["inputs"]["blind_inputs"],
            "splits": parent["inputs"]["splits"],
            "development": parent["inputs"]["development"],
        },
    }
    _write_once(output, lock)
    scoring = {
        "schema": f"{SCHEMA}.scoring-lock",
        "phase": "sealed_reference_binding_not_available_to_execution_workers",
        "execution_lock": _binding(output),
        "reference_catalog": parent["inputs"]["reference_catalog"],
        "bootstrap_seed": SETTINGS["bootstrap_seed"],
        "bootstrap_resamples": SETTINGS["bootstrap_resamples"],
    }
    _write_once(scoring_output, scoring)
    return lock, scoring


def validate_lock(path: Path, *, require_environment: bool = True) -> dict:
    lock = _read(path)
    if lock.get("schema") != SCHEMA or lock.get("settings") != SETTINGS:
        raise WorkflowBenchmarkError("invalid or edited workflow lock")
    if "reference_catalog" in lock.get("inputs", {}) or "reference" in json.dumps(lock).lower():
        raise WorkflowBenchmarkError("execution lock contains a reference-bearing field")
    if len(lock.get("objects", [])) != 140 or len(lock.get("development_object_ids", [])) != 30:
        raise WorkflowBenchmarkError("workflow lock has the wrong cohort size")
    if require_environment and lock["environment"] != grid._environment(lock["device"]):
        raise WorkflowBenchmarkError("timing environment changed after the lock")
    code_root = Path(lock["code_root"])
    if _runtime_sources(code_root) != lock["runtime_sources"]:
        raise WorkflowBenchmarkError("runtime source changed after the lock")
    _verify(lock["rule_report"])
    for binding in lock["inputs"].values():
        _verify(binding)
    for row in lock["objects"]:
        _verify(row["lightcurve"])
    for bundle in lock["bundles"]:
        for binding in bundle["files"]:
            _verify(binding)
    from .k3.solver_capacity import (
        LightcurveStructure,
        capacity_violations,
        read_static_capacities,
        tree_sha256,
    )

    solver = lock["solver"]
    _verify(solver["executable"])
    _verify(solver["capacity_report"])
    source_root = Path(solver["source_root"])
    if (
        tree_sha256(source_root) != solver["source_tree_sha256"]
        or read_static_capacities(source_root) != solver["capacities"]
    ):
        raise WorkflowBenchmarkError("solver source or static capacities changed after the lock")
    for row in lock["objects"]:
        if capacity_violations(LightcurveStructure(**row["structure"]), solver["capacities"]):
            raise WorkflowBenchmarkError(f"unsafe solver capacity for {row['object_id']}")
    return lock


def _jobs(lock: dict) -> list[dict]:
    jobs = [
        {**row, "repeat_index": repeat, "arm": arm}
        for row in lock["objects"]
        for repeat in range(SETTINGS["repeats"])
        for arm in ARMS
    ]
    random.Random(SETTINGS["order_seed"]).shuffle(jobs)
    return jobs


def execute(*, lock_path: Path, output: Path) -> dict:
    lock = validate_lock(lock_path)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    start_path = output / "execution-start.json"
    start = {
        "schema": EXECUTION_SCHEMA,
        "phase": "reference_blind_execution_started",
        "lock": _binding(lock_path),
        "started_unix": time.time(),
        "cases": len(_jobs(lock)),
    }
    if start_path.exists():
        prior = _read(start_path)
        if prior.get("lock") != start["lock"] or prior.get("cases") != start["cases"]:
            raise WorkflowBenchmarkError("existing execution directory belongs to another lock")
    else:
        _write_once(start_path, start)
    interpreter = _verify(lock["environment"]["interpreter"])
    runner = Path(lock["code_root"]) / "repro/run_delphi_workflow_benchmark.py"
    deadline = time.perf_counter() + SETTINGS["full_run_budget_seconds"]
    bindings = []
    jobs = _jobs(lock)
    for number, job in enumerate(jobs, start=1):
        name = f"{job['object_id']}-r{job['repeat_index']}-{job['arm']}"
        case = output / "cases" / name
        timed_path = case / "timed-result.json"
        if timed_path.exists():
            bindings.append({"path": str(timed_path.relative_to(output)), "sha256": grid.digest(timed_path)})
            print(f"[{number}/{len(jobs)}] verified existing {name}", flush=True)
            continue
        job_path = output / "jobs" / f"{name}.json"
        payload = {**job, "directory": str(case)}
        job_path.parent.mkdir(parents=True, exist_ok=True)
        if not job_path.exists():
            _write_once(job_path, payload)
        elif _read(job_path) != payload:
            raise WorkflowBenchmarkError(f"existing job changed: {name}")
        environment = os.environ.copy()
        environment.update(
            {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "PYTHONPATH": str(lock["code_root"]),
            }
        )
        command = [
            str(interpreter),
            str(runner),
            "_worker",
            "--lock",
            str(lock_path.resolve()),
            "--job",
            str(job_path.resolve()),
        ]
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise WorkflowBenchmarkError("full-run elapsed budget reached")
        started = time.perf_counter()
        result = subprocess.run(
            command,
            cwd=lock["code_root"],
            env=environment,
            capture_output=True,
            text=True,
            timeout=min(SETTINGS["case_timeout_seconds"], remaining),
            check=False,
            start_new_session=True,
        )
        wall = time.perf_counter() - started
        log = output / "logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        if not log.exists():
            log.write_text(result.stdout + "\n--- STDERR ---\n" + result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise WorkflowBenchmarkError(f"case failed without retry: {name}; see {log}")
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if len(lines) != 1:
            raise WorkflowBenchmarkError(f"worker returned unexpected output: {name}")
        row = json.loads(lines[0])
        row.update({"wall_seconds": wall, "case_path": str(case.relative_to(output))})
        _write_once(timed_path, row)
        bindings.append({"path": str(timed_path.relative_to(output)), "sha256": grid.digest(timed_path)})
        print(
            f"[{number}/{len(jobs)}] {name} {wall:.2f}s "
            f"fallback={row['fallback_used']} error-blind",
            flush=True,
        )
    execution = {
        "schema": EXECUTION_SCHEMA,
        "phase": "reference_blind_execution_complete",
        "lock": _binding(lock_path),
        "execution_start": _binding(start_path),
        "rows": sorted(bindings, key=lambda item: item["path"]),
        "completed_unix": time.time(),
        "reference_records_opened": False,
    }
    target = output / "execution.json"
    if target.exists():
        if _read(target) != execution:
            raise WorkflowBenchmarkError("existing execution summary differs")
    else:
        _write_once(target, execution)
    return execution


def verify_execution(*, lock_path: Path, execution_path: Path) -> dict:
    """Verify a completed reference-blind run without opening DAMIT axes."""
    lock = validate_lock(lock_path, require_environment=False)
    execution = _read(execution_path)
    if (
        execution.get("schema") != EXECUTION_SCHEMA
        or execution.get("phase") != "reference_blind_execution_complete"
        or execution.get("reference_records_opened") is not False
        or execution.get("lock") != _binding(lock_path)
    ):
        raise WorkflowBenchmarkError("invalid or incomplete execution summary")
    root = execution_path.parent
    rows = []
    for binding in execution.get("rows", []):
        path = (root / binding["path"]).resolve(strict=True)
        if not path.is_relative_to(root.resolve()) or grid.digest(path) != binding["sha256"]:
            raise WorkflowBenchmarkError("execution row escaped or changed")
        rows.append(_read(path))
    expected = {
        (row["object_id"], arm, repeat)
        for row in lock["objects"]
        for arm in ARMS
        for repeat in range(SETTINGS["repeats"])
    }
    observed = {
        (row.get("object_id"), row.get("arm"), row.get("repeat_index")) for row in rows
    }
    if len(rows) != len(expected) or len(observed) != len(rows) or observed != expected:
        raise WorkflowBenchmarkError("execution rows do not match the frozen factorial")
    if not all(row.get("completed") is True and row.get("selected_fit") for row in rows):
        raise WorkflowBenchmarkError("execution contains an incomplete workflow")
    return {
        "schema": EXECUTION_SCHEMA,
        "status": "verified_reference_blind_execution",
        "cases": len(rows),
        "objects": len(lock["objects"]),
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _axis_consistent(axes: list[object]) -> bool:
    return all(axial_angle_degrees(axes[0], axis) <= 1e-6 for axis in axes[1:])


def score(
    *, lock_path: Path, execution_path: Path, scoring_lock: Path, output: Path
) -> dict:
    lock = validate_lock(lock_path, require_environment=False)
    execution = _read(execution_path)
    scoring = _read(scoring_lock)
    if execution.get("schema") != EXECUTION_SCHEMA or execution.get("phase") != "reference_blind_execution_complete":
        raise WorkflowBenchmarkError("execution is incomplete")
    if execution.get("reference_records_opened") is not False:
        raise WorkflowBenchmarkError("execution was not reference blind")
    if (
        scoring.get("schema") != f"{SCHEMA}.scoring-lock"
        or scoring.get("execution_lock") != _binding(lock_path)
    ):
        raise WorkflowBenchmarkError("scoring lock belongs to another execution lock")
    root = execution_path.parent
    rows = []
    for binding in execution["rows"]:
        path = root / binding["path"]
        if grid.digest(path) != binding["sha256"]:
            raise WorkflowBenchmarkError("timed result changed after execution")
        rows.append(_read(path))
    expected = len(lock["objects"]) * len(ARMS) * SETTINGS["repeats"]
    if len(rows) != expected:
        raise WorkflowBenchmarkError("execution has the wrong number of cases")
    reference_path = _verify(scoring["reference_catalog"])
    references = {}
    with reference_path.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if item.get("eligible"):
                references[item["object_id"]] = [solution["vector"] for solution in item["solutions"]]
    by_key = {(row["object_id"], row["arm"], row["repeat_index"]): row for row in rows}
    object_rows = []
    for metadata in sorted(lock["objects"], key=lambda item: item["object_id"]):
        object_id = metadata["object_id"]
        summary = {"object_id": object_id, "fold": metadata["fold"], "arms": {}}
        for arm in ARMS:
            repeats = [by_key[(object_id, arm, repeat)] for repeat in range(SETTINGS["repeats"])]
            if not all(row["completed"] and row["selected_fit"] for row in repeats):
                raise WorkflowBenchmarkError(f"incomplete workflow for {object_id} {arm}")
            axes = [row["selected_fit"]["axis"] for row in repeats]
            if not _axis_consistent(axes):
                raise WorkflowBenchmarkError(f"selected pole varies across repeats for {object_id} {arm}")
            if len({bool(row["fallback_used"]) for row in repeats}) != 1:
                raise WorkflowBenchmarkError(f"fallback decision varies across repeats for {object_id} {arm}")
            rms = [float(row["selected_fit"]["final_relative_rms"]) for row in repeats]
            if max(rms) - min(rms) > 1e-12:
                raise WorkflowBenchmarkError(f"selected RMS varies across repeats for {object_id} {arm}")
            summary["arms"][arm] = {
                "axis": axes[0],
                "reference_error_degrees": _reference_error(axes[0], references[object_id]),
                "final_relative_rms": rms[0],
                "mean_wall_seconds": sum(float(row["wall_seconds"]) for row in repeats) / len(repeats),
                "fallback_used": bool(repeats[0]["fallback_used"]),
            }
        object_rows.append(summary)

    def endpoints(sample: list[dict]) -> dict:
        classical = [row["arms"]["classical"] for row in sample]
        delphi = [row["arms"]["delphi"] for row in sample]
        return {
            "speed_ratio_classical_over_delphi": (
                sum(row["mean_wall_seconds"] for row in classical)
                / sum(row["mean_wall_seconds"] for row in delphi)
            ),
            "mean_error_difference_degrees": (
                sum(row["reference_error_degrees"] for row in delphi)
                - sum(row["reference_error_degrees"] for row in classical)
            )
            / len(sample),
            "within20_difference": (
                sum(row["reference_error_degrees"] <= 20 for row in delphi)
                - sum(row["reference_error_degrees"] <= 20 for row in classical)
            )
            / len(sample),
            "rms_ratio_delphi_over_classical": _geometric_mean(
                [
                    d["final_relative_rms"] / c["final_relative_rms"]
                    for c, d in zip(classical, delphi)
                ]
            ),
            "classical_mean_error_degrees": sum(
                row["reference_error_degrees"] for row in classical
            )
            / len(sample),
            "delphi_mean_error_degrees": sum(row["reference_error_degrees"] for row in delphi)
            / len(sample),
            "classical_within20": sum(row["reference_error_degrees"] <= 20 for row in classical),
            "delphi_within20": sum(row["reference_error_degrees"] <= 20 for row in delphi),
            "classical_mean_seconds": sum(row["mean_wall_seconds"] for row in classical) / len(sample),
            "delphi_mean_seconds": sum(row["mean_wall_seconds"] for row in delphi) / len(sample),
            "classical_fallbacks": sum(row["fallback_used"] for row in classical),
            "delphi_fallbacks": sum(row["fallback_used"] for row in delphi),
        }

    point = endpoints(object_rows)
    rng = random.Random(SETTINGS["bootstrap_seed"])
    folds = {fold: [row for row in object_rows if row["fold"] == fold] for fold in range(5)}
    draws = []
    for _ in range(SETTINGS["bootstrap_resamples"]):
        sample = []
        for fold in range(5):
            group = folds[fold]
            sample.extend(rng.choice(group) for _ in range(len(group)))
        draws.append(endpoints(sample))
    intervals = {
        key: [_percentile([draw[key] for draw in draws], 2.5), _percentile([draw[key] for draw in draws], 97.5)]
        for key in (
            "speed_ratio_classical_over_delphi",
            "mean_error_difference_degrees",
            "within20_difference",
            "rms_ratio_delphi_over_classical",
        )
    }
    gates = {
        "mean_speed_ratio_at_least_1_25": point["speed_ratio_classical_over_delphi"]
        >= SETTINGS["minimum_mean_speed_ratio"],
        "speed_lower_bound_above_1": intervals["speed_ratio_classical_over_delphi"][0]
        > SETTINGS["minimum_speed_lower_bound"],
        "mean_error_upper_bound_at_most_3_degrees": intervals[
            "mean_error_difference_degrees"
        ][1]
        <= SETTINGS["mean_error_margin_degrees"],
        "within20_lower_bound_at_least_minus_5pp": intervals["within20_difference"][0]
        >= -SETTINGS["within20_margin"],
        "rms_upper_bound_at_most_1_01": intervals["rms_ratio_delphi_over_classical"][1]
        <= SETTINGS["rms_ratio_limit"],
        "all_cases_completed": True,
    }
    result = {
        "schema": SCORE_SCHEMA,
        "phase": "reference_scoring_after_sealed_execution",
        "lock": _binding(lock_path),
        "execution": _binding(execution_path),
        "scoring_lock": _binding(scoring_lock),
        "cohort": {"objects": len(object_rows), "objects_per_fold": 28, "repeats": 3},
        "point_estimates": point,
        "bootstrap_95_intervals": intervals,
        "gates": gates,
        "acceleration_at_comparable_accuracy_established": all(gates.values()),
        "object_rows": object_rows,
    }
    _write_once(output, result)
    return result
