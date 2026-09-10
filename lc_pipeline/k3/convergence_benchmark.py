"""Separate convergence-stopped K3/DAMIT acceleration benchmark.

This deliberately does not share the fixed-work publication runner's schemas:
the DAMIT program interprets a stopping value below one as a dev-improvement
tolerance, rather than as an iteration limit.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..physics.axial import axial_angular_error_deg
from ..v2.convexinv import ConvexinvParameters, run_convexinv, write_convexinv_parameters
from ..v2.data import sha256_file
from .downstream import (
    DAMIT_STANDARD_STARTS_DEG,
    DownstreamBenchmarkError,
    DownstreamStart,
    _atomic_json,
    _load_benchmark_inputs,
    evaluate_downstream_gate,
    signed_starts_from_axes,
)
from .protocol import K3_PROTOCOL_SHA256


CONVERGENCE_ITERATION_CAP = 1000


def _validate_tolerance(value: float) -> float:
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise DownstreamBenchmarkError(
            "convergence_tolerance must be finite and strictly between zero and one"
        )
    return float(value)


def _timing_fields(ensemble_path: str | Path, count: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load optional cold/warm timings without changing the frozen ensemble contract."""
    try:
        with np.load(ensemble_path, allow_pickle=False) as artifact:
            values = []
            for key in ("inference_cold_wall_seconds", "inference_warm_wall_seconds"):
                if key not in artifact.files:
                    values.append(None)
                    continue
                timing = np.asarray(artifact[key], dtype=np.float64)
                if timing.shape != (count,) or not np.all(np.isfinite(timing)) or np.any(timing <= 0):
                    raise DownstreamBenchmarkError(f"{key} must be a positive aligned timing vector")
                values.append(timing)
            return values[0], values[1]
    except (OSError, ValueError) as exc:
        raise DownstreamBenchmarkError(f"cannot load optional neural timings: {exc}") from exc


def _completion(result: Mapping[str, object]) -> str:
    if result.get("timed_out") is True:
        return "timeout"
    if result.get("return_code") != 0:
        return "exit-nonzero"
    iterations = result.get("iterations")
    if not isinstance(iterations, int) or iterations <= 0:
        return "missing-iteration-log"
    # DAMIT emits no explicit termination reason.  At its documented 1000
    # iteration guard we conservatively do not claim tolerance convergence.
    if iterations >= CONVERGENCE_ITERATION_CAP:
        return "iteration-cap"
    return "converged"


def _selectable(record: Mapping[str, object]) -> bool:
    result = record.get("result")
    if not isinstance(result, Mapping) or record.get("completion") != "converged":
        return False
    return all(
        isinstance(result.get(key), (int, float)) and math.isfinite(float(result[key]))
        for key in ("relative_rms_from_output", "final_lambda_deg", "final_beta_deg")
    ) and float(result["relative_rms_from_output"]) > 0


def _run_convergence_record(
    *, executable: Path, source_root: Path, lightcurve: Path, output_root: Path,
    object_id: str, arm: str, start_index: int, start: DownstreamStart,
    convergence_tolerance: float, timeout_seconds: float, compiler_command: Sequence[str],
) -> dict[str, object]:
    run_directory = output_root / "runs" / object_id / arm / f"start-{start_index}"
    parameter_path, record_path = run_directory / "parameters.txt", run_directory / "result.json"
    stdout_path, stderr_path = run_directory / "stdout.log", run_directory / "stderr.log"
    parameters = ConvexinvParameters(
        lambda_deg=start.lambda_deg, beta_deg=start.beta_deg, period_hours=start.period_hours,
        free_lambda=True, free_beta=True, free_period=False,
        iteration_stop_condition=convergence_tolerance,
    )
    identity = {
        "object_id": object_id, "arm": arm, "start_index": start_index,
        "lambda_deg": start.lambda_deg, "beta_deg": start.beta_deg,
        "period_hours": start.period_hours, "convergence_tolerance": convergence_tolerance,
    }
    if record_path.exists():
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DownstreamBenchmarkError(f"cannot resume convergence record {record_path}: {exc}") from exc
        result = record.get("result", {})
        provenance = result.get("provenance", {}) if isinstance(result, Mapping) else {}
        logs = record.get("logs", {})
        if (
            record.get("identity") != identity or not parameter_path.is_file()
            or parameter_path.read_text(encoding="ascii") != parameters.render()
            or not stdout_path.is_file() or not stderr_path.is_file()
            or not isinstance(logs, Mapping)
            or logs.get("stdout_sha256") != sha256_file(stdout_path)
            or logs.get("stderr_sha256") != sha256_file(stderr_path)
            or provenance.get("executable_sha256") != sha256_file(executable)
            or provenance.get("input_sha256") != sha256_file(lightcurve)
            or provenance.get("parameter_sha256") != sha256_file(parameter_path)
        ):
            raise DownstreamBenchmarkError(f"existing convergence record does not match this run: {record_path}")
        return record
    if run_directory.exists() and any(run_directory.iterdir()):
        raise DownstreamBenchmarkError(f"incomplete convergence directory requires manual audit: {run_directory}")
    write_convexinv_parameters(parameter_path, parameters)
    result = run_convexinv(
        executable=executable, source_root=source_root, lightcurve_file=lightcurve,
        parameter_file=parameter_path, output_directory=run_directory,
        timeout_seconds=timeout_seconds, compiler_command=compiler_command,
        stdout_log_path=stdout_path, stderr_log_path=stderr_path,
    )
    result_mapping = asdict(result)
    payload: dict[str, object] = {
        "schema": "delphi.k3-convergence-convexinv-run.v1", "identity": identity,
        "completion": _completion(result_mapping), "result": result_mapping,
        "logs": {
            "stdout": "stdout.log", "stderr": "stderr.log",
            "stdout_sha256": sha256_file(stdout_path), "stderr_sha256": sha256_file(stderr_path),
        },
    }
    _atomic_json(record_path, payload)
    return payload


def run_convergence_benchmark(
    *, executable: str | Path, source_root: str | Path, source_archive: str | Path,
    ensemble_path: str | Path, catalog_path: str | Path, dump_root: str | Path,
    split_path: str | Path, output_directory: str | Path, convergence_tolerance: float,
    timeout_seconds: float = 3600.0, compiler_command: Sequence[str] = ("cc", "--version"),
) -> dict[str, object]:
    """Run two matched six-start arms, stopping each DAMIT fit on convergence."""
    tolerance = _validate_tolerance(convergence_tolerance)
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise DownstreamBenchmarkError("timeout must be finite and positive")
    binary, source, archive, destination = map(Path, (executable, source_root, source_archive, output_directory))
    binary, source, archive, destination = binary.resolve(), source.resolve(), archive.resolve(), destination.resolve()
    rows_path, summary_path = destination / "convergence-rows.json", destination / "convergence-summary.json"
    if rows_path.exists() or summary_path.exists():
        raise DownstreamBenchmarkError("refusing to overwrite completed convergence benchmark outputs")
    if not binary.is_file() or not source.is_dir() or not archive.is_file():
        raise DownstreamBenchmarkError("convexinv executable, source root, and downloaded archive must exist")
    object_ids, axes, inference, catalog = _load_benchmark_inputs(ensemble_path, catalog_path, dump_root, split_path)
    cold, warm = _timing_fields(ensemble_path, len(object_ids))
    baseline_times: list[float] = []; candidate_times: list[float] = []
    baseline_rms: list[float] = []; candidate_rms: list[float] = []
    baseline_success: list[bool] = []; candidate_success: list[bool] = []; rows: list[dict[str, object]] = []
    ranks = {object_id: rank for rank, object_id in enumerate(sorted(object_ids, key=lambda value: hashlib.sha256(f"{K3_PROTOCOL_SHA256}:{value}".encode()).hexdigest()))}
    for index, object_id in enumerate(object_ids):
        row = catalog[object_id]
        try:
            solutions = row["solutions"]
            period = float(solutions[0]["period_hours"])
            targets = np.asarray([solution["vector"] for solution in solutions], dtype=np.float64)
            lightcurve = Path(dump_root) / str(row["lightcurve"]["source_path"])
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise DownstreamBenchmarkError(f"catalog solution metadata is invalid for {object_id}") from exc
        if not math.isfinite(period) or period <= 0 or targets.ndim != 2 or targets.shape[1] != 3 or not np.all(np.isfinite(targets)):
            raise DownstreamBenchmarkError(f"catalog targets are invalid for {object_id}")
        starts = {
            "baseline": tuple(DownstreamStart(a, b, period) for a, b in DAMIT_STANDARD_STARTS_DEG),
            "candidate": signed_starts_from_axes(axes[index], period_hours=period),
        }
        first = "baseline" if ranks[object_id] % 2 == 0 else "candidate"
        arm_order = (first, "candidate" if first == "baseline" else "baseline")
        records: dict[str, list[dict[str, object]]] = {"baseline": [], "candidate": []}
        for arm in arm_order:
            for start_index, start in enumerate(starts[arm]):
                records[arm].append(_run_convergence_record(
                    executable=binary, source_root=source, lightcurve=lightcurve, output_root=destination,
                    object_id=object_id, arm=arm, start_index=start_index, start=start,
                    convergence_tolerance=tolerance, timeout_seconds=timeout_seconds, compiler_command=compiler_command,
                ))
        object_row: dict[str, object] = {
            "object_id": object_id, "period_hours": period, "arm_order": list(arm_order),
            "neural_inference_wall_seconds": float(inference[index]),
            "neural_inference_cold_wall_seconds": None if cold is None else float(cold[index]),
            "neural_inference_warm_wall_seconds": None if warm is None else float(warm[index]),
        }
        for arm in ("baseline", "candidate"):
            arm_records = records[arm]
            eligible = [record for record in arm_records if _selectable(record)]
            best = min(eligible, key=lambda record: (float(record["result"]["relative_rms_from_output"]), int(record["identity"]["start_index"])), default=None)
            wall = sum(float(record["result"]["wall_time_seconds"]) for record in arm_records)
            if arm == "candidate":
                wall += float(warm[index] if warm is not None else inference[index])
            if best is None:
                rms, pole_error, success, best_start = float("nan"), float("nan"), False, None
            else:
                result = best["result"]
                longitude, latitude = math.radians(float(result["final_lambda_deg"])), math.radians(float(result["final_beta_deg"]))
                pole = np.asarray([math.cos(latitude) * math.cos(longitude), math.cos(latitude) * math.sin(longitude), math.sin(latitude)])
                rms, pole_error = float(result["relative_rms_from_output"]), float(np.min(axial_angular_error_deg(pole[None, :], targets)))
                success, best_start = pole_error <= 20.0, int(best["identity"]["start_index"])
            completions = {state: sum(record["completion"] == state for record in arm_records) for state in ("converged", "iteration-cap", "timeout", "exit-nonzero", "missing-iteration-log")}
            object_row[arm] = {
                "wall_seconds": float(wall), "process_cpu_seconds": float(sum(float(record["result"].get("process_cpu_time_seconds") or 0.0) for record in arm_records)),
                "iterations": int(sum(int(record["result"].get("iterations") or 0) for record in arm_records)),
                "starts_requested": 6, "starts_completed": int(sum(record["completion"] not in {"timeout", "exit-nonzero"} for record in arm_records)),
                "completion_counts": completions, "residual_selected_converged_starts": len(eligible),
                "best_start_index": best_start, "final_rms": rms, "pole_error_deg": pole_error, "success": success,
            }
            (baseline_times if arm == "baseline" else candidate_times).append(float(wall))
            (baseline_rms if arm == "baseline" else candidate_rms).append(rms)
            (baseline_success if arm == "baseline" else candidate_success).append(success)
        rows.append(object_row)
    verdict = evaluate_downstream_gate(baseline_times, candidate_times, baseline_rms, candidate_rms, baseline_success=baseline_success, candidate_success=candidate_success)
    rows_sha = _atomic_json(rows_path, {"schema": "delphi.k3-convergence-rows.v1", "convergence_tolerance": tolerance, "iteration_cap": CONVERGENCE_ITERATION_CAP, "object_ids": list(object_ids), "rows": rows})
    summary: dict[str, object] = {
        "schema": "delphi.k3-convergence-summary.v1", "object_count": len(object_ids), "starts_per_arm": 6,
        "convergence_tolerance": tolerance, "iteration_cap": CONVERGENCE_ITERATION_CAP,
        "cap_semantics": "1000 iterations is treated as nonconverged because DAMIT emits no termination reason",
        "timeout_seconds": timeout_seconds, "neural_time_included": True,
        "warm_neural_time_used_when_available": True, "rows_sha256": rows_sha,
        "ensemble_sha256": sha256_file(ensemble_path), "catalog_sha256": sha256_file(catalog_path), "splits_sha256": sha256_file(split_path),
        "executable_sha256": sha256_file(binary), "source_archive_sha256": sha256_file(archive), "verdict": verdict.as_mapping(),
    }
    summary["artifact_sha256"] = _atomic_json(summary_path, summary)
    return summary


__all__ = ["CONVERGENCE_ITERATION_CAP", "run_convergence_benchmark"]
