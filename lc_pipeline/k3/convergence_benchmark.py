"""Separate convergence-stopped K3/DAMIT acceleration benchmark.

This deliberately does not share the fixed-work publication runner's schemas:
the DAMIT program interprets a stopping value below one as a dev-improvement
tolerance, rather than as an iteration limit.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..physics.axial import axial_angular_error_deg
from ..v2.convexinv import (
    ConvexinvError,
    ConvexinvParameters,
    run_convexinv,
    write_convexinv_parameters,
)
from ..v2.data import canonical_json, sha256_file
from .convergence_axis import ConvergenceAxisError, decode_convergence_axis
from .downstream import (
    DAMIT_STANDARD_STARTS_DEG,
    DownstreamBenchmarkError,
    DownstreamStart,
    _atomic_json,
    _load_benchmark_inputs,
    signed_starts_from_axes,
)

CONVERGENCE_ITERATION_CAP = 1000
DEFAULT_REPEAT_COUNT = 3
DEFAULT_REPEAT_ORDER_SEED = 20260910
DEFAULT_BOOTSTRAP_SEED = 20260911
OBJECT_SUBSET_SCHEMA = "delphi.k3-convergence-object-subset.v1"


def _validate_tolerance(value: float) -> float:
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise DownstreamBenchmarkError(
            "convergence_tolerance must be finite and strictly between zero and one"
        )
    return float(value)


def _validate_repeat_count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DownstreamBenchmarkError("repeat_count must be a positive integer")
    return value


def _select_object_subset(
    object_ids_path: str | Path | None,
    full_object_ids: tuple[str, ...],
    split_path: str | Path,
) -> tuple[tuple[int, ...], dict[str, object]]:
    """Validate an optional cohort manifest against already-validated full inputs."""
    if object_ids_path is None:
        return tuple(range(len(full_object_ids))), {
            "subset_manifest_sha256": None,
            "subset_role": "full_ensemble",
        }
    path = Path(object_ids_path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DownstreamBenchmarkError(f"cannot read object subset manifest: {exc}") from exc
    required_fields = {"schema", "role", "source_full_split_sha256", "object_ids"}
    provenance_fields = {"study_spec_sha256", "selection", "salt"}
    if (
        not isinstance(document, dict)
        or not required_fields.issubset(document)
        or not set(document).issubset(required_fields | provenance_fields)
    ):
        raise DownstreamBenchmarkError("object subset manifest has an invalid schema or fields")
    if document["schema"] != OBJECT_SUBSET_SCHEMA:
        raise DownstreamBenchmarkError("object subset manifest schema is invalid")
    if document["role"] not in {"development", "locked_evaluation"}:
        raise DownstreamBenchmarkError(
            "object subset manifest role must be development or locked_evaluation"
        )
    if document["source_full_split_sha256"] != sha256_file(split_path):
        raise DownstreamBenchmarkError("object subset manifest does not match the full split")
    if "study_spec_sha256" in document and (
        not isinstance(document["study_spec_sha256"], str)
        or len(document["study_spec_sha256"]) != 64
    ):
        raise DownstreamBenchmarkError("object subset manifest has an invalid study spec hash")
    for field in ("selection", "salt"):
        if field in document and (not isinstance(document[field], str) or not document[field]):
            raise DownstreamBenchmarkError(f"object subset manifest has an invalid {field} value")
    selected = document["object_ids"]
    if (
        not isinstance(selected, list)
        or not selected
        or any(not isinstance(value, str) or not value for value in selected)
    ):
        raise DownstreamBenchmarkError("object subset manifest must contain nonempty object IDs")
    if len(selected) != len(set(selected)):
        raise DownstreamBenchmarkError("object subset manifest contains duplicate object IDs")
    lookup = {object_id: index for index, object_id in enumerate(full_object_ids)}
    unknown = [object_id for object_id in selected if object_id not in lookup]
    if unknown:
        raise DownstreamBenchmarkError("object subset manifest contains unknown object IDs")
    return tuple(lookup[object_id] for object_id in selected), {
        "subset_manifest_sha256": sha256_file(path),
        "subset_role": document["role"],
    }


def _repeat_arm_order(object_id: str, repeat_index: int, seed: int) -> tuple[str, str]:
    """Deterministically assign each paired repeat AB or BA."""
    digest = hashlib.sha256(f"{seed}:{object_id}:{repeat_index}".encode()).digest()
    return ("baseline", "candidate") if digest[0] % 2 == 0 else ("candidate", "baseline")


def evaluate_convergence_gate(
    baseline_wall: np.ndarray,
    candidate_wall: np.ndarray,
    baseline_recovery: np.ndarray,
    candidate_recovery: np.ndarray,
    baseline_completion: np.ndarray,
    candidate_completion: np.ndarray,
    baseline_rms: np.ndarray,
    candidate_rms: np.ndarray,
    *,
    resamples: int = 10000,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Object-clustered one-sided gates for repeated convergence runs.

    Rows are objects and columns are paired repeats.  Resampling objects keeps
    all starts/repeats for an object together, so failures remain in runtime,
    recovery, and completion estimands.
    """
    arrays = tuple(
        np.asarray(value)
        for value in (
            baseline_wall,
            candidate_wall,
            baseline_recovery,
            candidate_recovery,
            baseline_completion,
            candidate_completion,
            baseline_rms,
            candidate_rms,
        )
    )
    shape = arrays[0].shape
    if (
        len(shape) != 2
        or shape[0] < 2
        or shape[1] < 1
        or any(value.shape != shape for value in arrays)
    ):
        raise DownstreamBenchmarkError("convergence gate requires aligned object-by-repeat arrays")
    if resamples < 1:
        raise DownstreamBenchmarkError("bootstrap resamples must be positive")
    (
        base_wall,
        candidate_wall,
        base_rec,
        candidate_rec,
        base_comp,
        candidate_comp,
        base_rms,
        candidate_rms,
    ) = arrays
    if (
        np.any(~np.isfinite(base_wall))
        or np.any(~np.isfinite(candidate_wall))
        or np.any(base_wall <= 0)
        or np.any(candidate_wall <= 0)
    ):
        raise DownstreamBenchmarkError("convergence timing must be finite and positive")
    for values in (base_rec, candidate_rec, base_comp, candidate_comp):
        if not np.all(np.isin(values, (0, 1))):
            raise DownstreamBenchmarkError(
                "binary convergence estimands must contain only zero or one"
            )
    joint = (base_rec == 1) & (candidate_rec == 1)
    if not np.any(joint):
        raise DownstreamBenchmarkError("RMS gate requires at least one joint-success repeat")
    ratios = candidate_rms[joint] / base_rms[joint]
    if np.any(~np.isfinite(ratios)) or np.any(ratios <= 0):
        raise DownstreamBenchmarkError("joint-success RMS values must be finite and positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, shape[0], size=(resamples, shape[0]))
    runtime_samples = np.sum(base_wall[indices], axis=(1, 2)) / np.sum(
        candidate_wall[indices], axis=(1, 2)
    )
    recovery_samples = np.mean(candidate_rec[indices] - base_rec[indices], axis=(1, 2))
    completion_samples = np.mean(candidate_comp[indices] - base_comp[indices], axis=(1, 2))
    rms_samples: list[float] = []
    for sampled in indices:
        sampled_joint = joint[sampled]
        sampled_ratio = candidate_rms[sampled][sampled_joint] / base_rms[sampled][sampled_joint]
        if sampled_ratio.size:
            rms_samples.append(float(np.exp(np.mean(np.log(sampled_ratio)))))
    if not rms_samples:
        raise DownstreamBenchmarkError("bootstrap did not retain a joint-success RMS repeat")
    runtime, recovery, completion = (
        float(np.sum(base_wall) / np.sum(candidate_wall)),
        float(np.mean(candidate_rec - base_rec)),
        float(np.mean(candidate_comp - base_comp)),
    )
    rms_ratio = float(np.exp(np.mean(np.log(ratios))))
    bounds = {
        "runtime_ratio_ci95_lower": float(np.quantile(runtime_samples, 0.05)),
        "recovery_difference_ci95_lower": float(np.quantile(recovery_samples, 0.05)),
        "completion_difference_ci95_lower": float(np.quantile(completion_samples, 0.05)),
        "geometric_rms_ratio_ci95_upper": float(np.quantile(rms_samples, 0.95)),
    }
    failures = []
    if not bounds["runtime_ratio_ci95_lower"] > 1.0:
        failures.append("runtime ratio lower confidence bound is not strictly above one")
    if bounds["recovery_difference_ci95_lower"] < -0.02:
        failures.append("guided recovery lower confidence bound is below -0.02")
    if bounds["completion_difference_ci95_lower"] < -0.02:
        failures.append("guided completion lower confidence bound is below -0.02")
    if not bounds["geometric_rms_ratio_ci95_upper"] < 1.01:
        failures.append("joint-success RMS upper confidence bound is not below 1.01")
    return {
        "passed": not failures,
        "runtime_ratio": runtime,
        "recovery_difference": recovery,
        "completion_difference": completion,
        "geometric_rms_ratio": rms_ratio,
        "joint_success_repeats": int(joint.sum()),
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        **bounds,
        "failures": failures,
    }


def _timing_fields(
    ensemble_path: str | Path, count: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load optional cold/warm timings without changing the frozen ensemble contract."""
    try:
        with np.load(ensemble_path, allow_pickle=False) as artifact:
            values = []
            for key in ("inference_cold_wall_seconds", "inference_warm_wall_seconds"):
                if key not in artifact.files:
                    values.append(None)
                    continue
                timing = np.asarray(artifact[key], dtype=np.float64)
                if (
                    timing.shape != (count,)
                    or not np.all(np.isfinite(timing))
                    or np.any(timing <= 0)
                ):
                    raise DownstreamBenchmarkError(
                        f"{key} must be a positive aligned timing vector"
                    )
                values.append(timing)
            return values[0], values[1]
    except (OSError, ValueError) as exc:
        raise DownstreamBenchmarkError(f"cannot load optional neural timings: {exc}") from exc


def _completion(result: Mapping[str, object], *, expected_period_hours: float) -> str:
    """Classify one official-v0.2.1 run from its explicit trace and products.

    For the hash-verified solver, a successful run below its 1000-iteration
    guard can only leave the loop through the requested dev-improvement
    tolerance.  We nevertheless require the final trace and all numerical
    products so a NaN/parse failure cannot be called convergence.
    """
    if result.get("timed_out") is True:
        return "timeout"
    if result.get("adapter_error") is not None:
        return "adapter-error"
    if result.get("return_code") != 0:
        return "exit-nonzero"
    iterations = result.get("iterations")
    if not isinstance(iterations, int) or iterations <= 0:
        return "missing-iteration-log"
    # DAMIT emits no explicit termination reason.  At its documented 1000
    # iteration guard we conservatively do not claim tolerance convergence.
    if iterations >= CONVERGENCE_ITERATION_CAP:
        return "iteration-cap"
    numeric = (
        "chi2",
        "deviation",
        "final_lambda_deg",
        "final_beta_deg",
        "final_period_hours",
        "relative_rms_from_output",
    )
    if result.get("output_validation_error") is not None or not all(
        isinstance(result.get(key), (int, float))
        and not isinstance(result.get(key), bool)
        and math.isfinite(float(result[key]))
        for key in numeric
    ):
        return "numerical-output-failure"
    try:
        decode_convergence_axis(result["final_lambda_deg"], result["final_beta_deg"])
    except ConvergenceAxisError:
        return "numerical-output-failure"
    if (
        float(result["chi2"]) < 0
        or float(result["deviation"]) < 0
        or float(result["relative_rms_from_output"]) <= 0
        or not math.isclose(
            float(result["final_period_hours"]),
            expected_period_hours,
            rel_tol=1e-7,
            abs_tol=1e-9,
        )
        or any(
            not isinstance(result.get(key), str) or len(str(result[key])) != 64
            for key in (
                "output_lightcurve_sha256",
                "output_parameter_sha256",
                "output_area_sha256",
            )
        )
    ):
        return "numerical-output-failure"
    return "converged"


def _selectable(record: Mapping[str, object]) -> bool:
    result = record.get("result")
    if not isinstance(result, Mapping) or record.get("completion") != "converged":
        return False
    try:
        axis = decode_convergence_axis(
            result.get("final_lambda_deg"), result.get("final_beta_deg")
        )
    except ConvergenceAxisError:
        return False
    return (
        isinstance(result.get("relative_rms_from_output"), (int, float))
        and not isinstance(result.get("relative_rms_from_output"), bool)
        and math.isfinite(float(result["relative_rms_from_output"]))
        and float(result["relative_rms_from_output"]) > 0
        # Touch the decoded axis so this function remains explicitly bound to
        # the same interpretation used for selection and scoring.
        and math.isfinite(axis.standard_beta_deg)
    )


def _run_convergence_record(
    *,
    executable: Path,
    source_root: Path,
    lightcurve: Path,
    output_root: Path,
    object_id: str,
    arm: str,
    repeat_index: int,
    start_index: int,
    start: DownstreamStart,
    convergence_tolerance: float,
    timeout_seconds: float,
    compiler_command: Sequence[str],
    execution_contract: Mapping[str, object],
    arm_order: Sequence[str],
) -> dict[str, object]:
    run_directory = (
        output_root / "runs" / object_id / f"repeat-{repeat_index}" / arm / f"start-{start_index}"
    )
    parameter_path, record_path = run_directory / "parameters.txt", run_directory / "result.json"
    stdout_path, stderr_path = run_directory / "stdout.log", run_directory / "stderr.log"
    parameters = ConvexinvParameters(
        lambda_deg=start.lambda_deg,
        beta_deg=start.beta_deg,
        period_hours=start.period_hours,
        free_lambda=True,
        free_beta=True,
        free_period=False,
        iteration_stop_condition=convergence_tolerance,
    )
    contract_sha256 = hashlib.sha256(canonical_json(dict(execution_contract)).encode()).hexdigest()
    identity = {
        "object_id": object_id,
        "arm": arm,
        "repeat_index": repeat_index,
        "start_index": start_index,
        "lambda_deg": start.lambda_deg,
        "beta_deg": start.beta_deg,
        "period_hours": start.period_hours,
        "convergence_tolerance": convergence_tolerance,
        "timeout_seconds": timeout_seconds,
        "compiler_command": list(compiler_command),
        "repeat_arm_order": list(arm_order),
        "execution_contract_sha256": contract_sha256,
    }
    if record_path.exists():
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DownstreamBenchmarkError(
                f"cannot resume convergence record {record_path}: {exc}"
            ) from exc
        result = record.get("result", {})
        provenance = result.get("provenance", {}) if isinstance(result, Mapping) else {}
        logs = record.get("logs", {})
        if (
            record.get("identity") != identity
            or not parameter_path.is_file()
            or parameter_path.read_text(encoding="ascii") != parameters.render()
            or not stdout_path.is_file()
            or not stderr_path.is_file()
            or not isinstance(logs, Mapping)
            or logs.get("stdout_sha256") != sha256_file(stdout_path)
            or logs.get("stderr_sha256") != sha256_file(stderr_path)
            or provenance.get("executable_sha256") != sha256_file(executable)
            or provenance.get("input_sha256") != sha256_file(lightcurve)
            or provenance.get("parameter_sha256") != sha256_file(parameter_path)
            or record.get("execution_contract") != dict(execution_contract)
        ):
            raise DownstreamBenchmarkError(
                f"existing convergence record does not match this run: {record_path}"
            )
        return record
    if run_directory.exists() and any(run_directory.iterdir()):
        raise DownstreamBenchmarkError(
            f"incomplete convergence directory requires manual audit: {run_directory}"
        )
    write_convexinv_parameters(parameter_path, parameters)
    started = time.monotonic()
    try:
        result = run_convexinv(
            executable=executable,
            source_root=source_root,
            lightcurve_file=lightcurve,
            parameter_file=parameter_path,
            output_directory=run_directory,
            timeout_seconds=timeout_seconds,
            compiler_command=compiler_command,
            stdout_log_path=stdout_path,
            stderr_log_path=stderr_path,
        )
        result_mapping = asdict(result)
    except (ConvexinvError, OSError, ValueError) as exc:
        # Preserve an isolated adapter/numerical failure as one prespecified
        # start outcome.  Missing paths are materialized so the record remains
        # hash-auditable and its paired arm/subsequent starts still execute.
        for path in (stdout_path, stderr_path):
            if not path.exists():
                path.write_bytes(b"")
        result_mapping = {
            "return_code": None,
            "timed_out": False,
            "wall_time_seconds": time.monotonic() - started,
            "process_cpu_time_seconds": None,
            "iterations": None,
            "chi2": None,
            "deviation": None,
            "final_lambda_deg": None,
            "final_beta_deg": None,
            "final_period_hours": None,
            "relative_rms_from_output": None,
            "output_validation_error": None,
            "adapter_error": f"{type(exc).__name__}: {exc}",
            "provenance": dict(execution_contract),
        }
    payload: dict[str, object] = {
        "schema": "delphi.k3-convergence-convexinv-run.v1",
        "identity": identity,
        "completion": _completion(result_mapping, expected_period_hours=start.period_hours),
        "result": result_mapping,
        "execution_contract": dict(execution_contract),
        "logs": {
            "stdout": "stdout.log",
            "stderr": "stderr.log",
            "stdout_sha256": sha256_file(stdout_path),
            "stderr_sha256": sha256_file(stderr_path),
        },
    }
    _atomic_json(record_path, payload)
    return payload


def run_convergence_benchmark(
    *,
    executable: str | Path,
    source_root: str | Path,
    source_archive: str | Path,
    ensemble_path: str | Path,
    catalog_path: str | Path,
    dump_root: str | Path,
    split_path: str | Path,
    output_directory: str | Path,
    convergence_tolerance: float,
    timeout_seconds: float = 3600.0,
    repeat_count: int = DEFAULT_REPEAT_COUNT,
    repeat_order_seed: int = DEFAULT_REPEAT_ORDER_SEED,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    object_ids_path: str | Path | None = None,
    compiler_command: Sequence[str] = ("cc", "--version"),
) -> dict[str, object]:
    """Refuse the superseded, label-aware convergence workflow.

    The original implementation opened reference poles before fit execution,
    accepted loosely specified subsets/settings, and could reuse partial-repeat
    timing.  It is retained below only to preserve review history; publication
    execution must go through :mod:`lc_pipeline.k3.convergence_study`.
    """
    tolerance = _validate_tolerance(convergence_tolerance)
    repeats = _validate_repeat_count(repeat_count)
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise DownstreamBenchmarkError("timeout must be finite and positive")
    binary, source, archive, destination = map(
        Path, (executable, source_root, source_archive, output_directory)
    )
    binary, source, archive, destination = (
        binary.resolve(),
        source.resolve(),
        archive.resolve(),
        destination.resolve(),
    )
    rows_path, summary_path = (
        destination / "convergence-rows.json",
        destination / "convergence-summary.json",
    )
    if rows_path.exists() or summary_path.exists():
        raise DownstreamBenchmarkError(
            "refusing to overwrite completed convergence benchmark outputs"
        )
    raise DownstreamBenchmarkError(
        "legacy convergence runner is disabled; use the frozen phase-separated study orchestrator"
    )
    if not binary.is_file() or not source.is_dir() or not archive.is_file():
        raise DownstreamBenchmarkError(
            "convexinv executable, source root, and downloaded archive must exist"
        )
    full_object_ids, full_axes, full_inference, catalog = _load_benchmark_inputs(
        ensemble_path, catalog_path, dump_root, split_path
    )
    # Full ensemble/split/catalog alignment is deliberately checked before any
    # optional development or locked-evaluation cohort is selected.
    selected_indices, subset = _select_object_subset(object_ids_path, full_object_ids, split_path)
    object_ids = tuple(full_object_ids[index] for index in selected_indices)
    axes, inference = full_axes[list(selected_indices)], full_inference[list(selected_indices)]
    cold, warm = _timing_fields(ensemble_path, len(full_object_ids))
    if cold is not None:
        cold = cold[list(selected_indices)]
    if warm is not None:
        warm = warm[list(selected_indices)]
    baseline_times: list[list[float]] = []
    candidate_times: list[list[float]] = []
    baseline_rms: list[list[float]] = []
    candidate_rms: list[list[float]] = []
    baseline_success: list[list[int]] = []
    candidate_success: list[list[int]] = []
    baseline_completion: list[list[int]] = []
    candidate_completion: list[list[int]] = []
    rows: list[dict[str, object]] = []
    for index, object_id in enumerate(object_ids):
        row = catalog[object_id]
        try:
            solutions = row["solutions"]
            period = float(solutions[0]["period_hours"])
            targets = np.asarray([solution["vector"] for solution in solutions], dtype=np.float64)
            lightcurve = Path(dump_root) / str(row["lightcurve"]["source_path"])
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise DownstreamBenchmarkError(
                f"catalog solution metadata is invalid for {object_id}"
            ) from exc
        if (
            not math.isfinite(period)
            or period <= 0
            or targets.ndim != 2
            or targets.shape[1] != 3
            or not np.all(np.isfinite(targets))
        ):
            raise DownstreamBenchmarkError(f"catalog targets are invalid for {object_id}")
        starts = {
            "baseline": tuple(DownstreamStart(a, b, period) for a, b in DAMIT_STANDARD_STARTS_DEG),
            "candidate": signed_starts_from_axes(axes[index], period_hours=period),
        }
        object_row: dict[str, object] = {
            "object_id": object_id,
            "period_hours": period,
            "neural_inference_wall_seconds": float(inference[index]),
            "neural_inference_cold_wall_seconds": None if cold is None else float(cold[index]),
            "neural_inference_warm_wall_seconds": None if warm is None else float(warm[index]),
            "repetitions": [],
        }
        totals: dict[str, dict[str, object]] = {
            "baseline": {
                "wall_seconds": 0.0,
                "process_cpu_seconds": 0.0,
                "iterations": 0,
                "starts_requested": 0,
                "starts_completed": 0,
                "converged_repetitions": 0,
                "successful_repetitions": 0,
            }
        }
        totals["candidate"] = dict(totals["baseline"])
        per_arm = {
            "baseline": {"wall": [], "rms": [], "success": [], "completion": []},
            "candidate": {"wall": [], "rms": [], "success": [], "completion": []},
        }
        for repeat_index in range(repeats):
            arm_order = _repeat_arm_order(object_id, repeat_index, repeat_order_seed)
            records: dict[str, list[dict[str, object]]] = {"baseline": [], "candidate": []}
            # Pair each start directly: AB/BA is assigned per repeat, while no
            # failure can skip its matched partner or subsequent starts.
            for start_index in range(6):
                for arm in arm_order:
                    records[arm].append(
                        _run_convergence_record(
                            executable=binary,
                            source_root=source,
                            lightcurve=lightcurve,
                            output_root=destination,
                            object_id=object_id,
                            arm=arm,
                            repeat_index=repeat_index,
                            start_index=start_index,
                            start=starts[arm][start_index],
                            convergence_tolerance=tolerance,
                            timeout_seconds=timeout_seconds,
                            compiler_command=compiler_command,
                        )
                    )
            repeat_row: dict[str, object] = {
                "repeat_index": repeat_index,
                "arm_order": list(arm_order),
            }
            for arm in ("baseline", "candidate"):
                arm_records = records[arm]
                eligible = [record for record in arm_records if _selectable(record)]
                best = min(
                    eligible,
                    key=lambda record: (
                        float(record["result"]["relative_rms_from_output"]),
                        int(record["identity"]["start_index"]),
                    ),
                    default=None,
                )
                inversion_wall = sum(
                    float(record["result"]["wall_time_seconds"]) for record in arm_records
                )
                wall = inversion_wall + (
                    float(warm[index] if warm is not None else inference[index])
                    if arm == "candidate"
                    else 0.0
                )
                completed = all(record["completion"] == "converged" for record in arm_records)
                if best is None:
                    rms, pole_error, success, best_start = float("nan"), float("nan"), False, None
                else:
                    result = best["result"]
                    pole = np.asarray(
                        decode_convergence_axis(
                            result["final_lambda_deg"], result["final_beta_deg"]
                        ).directed_unit_vector,
                        dtype=np.float64,
                    )
                    rms, pole_error = (
                        float(result["relative_rms_from_output"]),
                        float(np.min(axial_angular_error_deg(pole[None, :], targets))),
                    )
                    success, best_start = pole_error <= 20.0, int(best["identity"]["start_index"])
                counts = {
                    state: sum(record["completion"] == state for record in arm_records)
                    for state in (
                        "converged",
                        "iteration-cap",
                        "timeout",
                        "exit-nonzero",
                        "missing-iteration-log",
                    )
                }
                arm_row = {
                    "wall_seconds": float(wall),
                    "inversion_wall_seconds": float(inversion_wall),
                    "process_cpu_seconds": float(
                        sum(
                            float(record["result"].get("process_cpu_time_seconds") or 0.0)
                            for record in arm_records
                        )
                    ),
                    "iterations": int(
                        sum(int(record["result"].get("iterations") or 0) for record in arm_records)
                    ),
                    "starts_requested": 6,
                    "starts_completed": int(
                        sum(
                            record["completion"] not in {"timeout", "exit-nonzero"}
                            for record in arm_records
                        )
                    ),
                    "completion_counts": counts,
                    "completed": completed,
                    "residual_selected_converged_starts": len(eligible),
                    "best_start_index": best_start,
                    "final_rms": rms,
                    "pole_error_deg": pole_error,
                    "success": success,
                }
                repeat_row[arm] = arm_row
                total = totals[arm]
                for key in (
                    "wall_seconds",
                    "process_cpu_seconds",
                    "iterations",
                    "starts_requested",
                    "starts_completed",
                ):
                    total[key] = total[key] + arm_row[key]
                total["converged_repetitions"] = total["converged_repetitions"] + int(completed)
                total["successful_repetitions"] = total["successful_repetitions"] + int(success)
                per_arm[arm]["wall"].append(float(wall))
                per_arm[arm]["rms"].append(rms)
                per_arm[arm]["success"].append(int(success))
                per_arm[arm]["completion"].append(int(completed))
            object_row["repetitions"].append(repeat_row)
        for arm in ("baseline", "candidate"):
            object_row[arm] = totals[arm]
        baseline_times.append(per_arm["baseline"]["wall"])
        candidate_times.append(per_arm["candidate"]["wall"])
        baseline_rms.append(per_arm["baseline"]["rms"])
        candidate_rms.append(per_arm["candidate"]["rms"])
        baseline_success.append(per_arm["baseline"]["success"])
        candidate_success.append(per_arm["candidate"]["success"])
        baseline_completion.append(per_arm["baseline"]["completion"])
        candidate_completion.append(per_arm["candidate"]["completion"])
        rows.append(object_row)
    verdict = evaluate_convergence_gate(
        np.asarray(baseline_times),
        np.asarray(candidate_times),
        np.asarray(baseline_success),
        np.asarray(candidate_success),
        np.asarray(baseline_completion),
        np.asarray(candidate_completion),
        np.asarray(baseline_rms),
        np.asarray(candidate_rms),
        seed=bootstrap_seed,
    )
    rows_sha = _atomic_json(
        rows_path,
        {
            "schema": "delphi.k3-convergence-rows.v2",
            "convergence_tolerance": tolerance,
            "iteration_cap": CONVERGENCE_ITERATION_CAP,
            "repeat_count": repeats,
            "repeat_order_seed": repeat_order_seed,
            **subset,
            "object_ids": list(object_ids),
            "rows": rows,
        },
    )
    summary: dict[str, object] = {
        "schema": "delphi.k3-convergence-summary.v2",
        "object_count": len(object_ids),
        "starts_per_arm": 6,
        "repeat_count": repeats,
        **subset,
        "convergence_tolerance": tolerance,
        "iteration_cap": CONVERGENCE_ITERATION_CAP,
        "cap_semantics": "1000 iterations is treated as nonconverged because DAMIT emits no termination reason",
        "timeout_seconds": timeout_seconds,
        "neural_time_included": True,
        "repeat_order_seed": repeat_order_seed,
        "bootstrap_seed": bootstrap_seed,
        "warm_neural_time_used_when_available": True,
        "rows_sha256": rows_sha,
        "ensemble_sha256": sha256_file(ensemble_path),
        "catalog_sha256": sha256_file(catalog_path),
        "splits_sha256": sha256_file(split_path),
        "executable_sha256": sha256_file(binary),
        "source_archive_sha256": sha256_file(archive),
        "verdict": verdict,
    }
    summary["artifact_sha256"] = _atomic_json(summary_path, summary)
    return summary


__all__ = ["CONVERGENCE_ITERATION_CAP", "run_convergence_benchmark"]
