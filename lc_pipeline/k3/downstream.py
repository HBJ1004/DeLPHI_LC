"""Auditable bridge from K3 axes to the fixed-period convexinv benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..physics.axial import axial_angular_error_deg
from ..v2.convexinv import (
    ConvexinvParameters,
    run_convexinv,
    write_convexinv_parameters,
)
from ..v2.data import sha256_file
from .protocol import K3_PROTOCOL_SHA256


class DownstreamBenchmarkError(ValueError):
    """Raised when the downstream benchmark is not comparable."""


@dataclass(frozen=True)
class DownstreamStart:
    lambda_deg: float
    beta_deg: float
    period_hours: float


DAMIT_STANDARD_STARTS_DEG: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (180.0, 0.0),
    (90.0, 60.0),
    (240.0, 60.0),
    (90.0, -60.0),
    (240.0, -60.0),
)


def axis_to_ecliptic(axis: Sequence[float]) -> tuple[float, float]:
    vector = np.asarray(axis, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise DownstreamBenchmarkError("axis must be one finite three-vector")
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        raise DownstreamBenchmarkError("axis must be nonzero")
    vector = vector / norm
    return float(np.degrees(np.arctan2(vector[1], vector[0])) % 360.0), float(
        np.degrees(np.arcsin(np.clip(vector[2], -1.0, 1.0)))
    )


def signed_starts_from_axes(
    axes: Sequence[Sequence[float]], *, period_hours: float
) -> tuple[DownstreamStart, ...]:
    """Expand K=3 unsigned axes to six deterministic signed convexinv starts."""
    values = np.asarray(axes, dtype=np.float64)
    if values.shape != (3, 3) or not np.all(np.isfinite(values)) or not math.isfinite(period_hours) or period_hours <= 0:
        raise DownstreamBenchmarkError("K3 downstream starts require three axes and a positive period")
    starts: list[DownstreamStart] = []
    for axis in values:
        for signed in (axis, -axis):
            longitude, latitude = axis_to_ecliptic(signed)
            starts.append(DownstreamStart(longitude, latitude, period_hours))
    return tuple(starts)


def alias_period_intervals(
    period_hours: float, *, aliases: Sequence[float] = (0.5, 1.0, 2.0), bounds: tuple[float, float] = (2.0, 200.0)
) -> tuple[tuple[float, float], ...]:
    """Return clipped P/2, P, and 2P intervals for a fixed-period input."""
    if not math.isfinite(period_hours) or period_hours <= 0 or not 0 < bounds[0] < bounds[1]:
        raise DownstreamBenchmarkError("period and bounds must be finite and positive")
    intervals: list[tuple[float, float]] = []
    for alias in aliases:
        if not math.isfinite(alias) or alias <= 0:
            raise DownstreamBenchmarkError("period aliases must be positive")
        center = period_hours * alias
        half_width = 0.02 * center
        low, high = max(bounds[0], center - half_width), min(bounds[1], center + half_width)
        if low < high:
            intervals.append((low, high))
    return tuple(sorted(intervals))


@dataclass(frozen=True)
class ArmAggregate:
    arm: str
    successful_objects: int
    failed_objects: int
    summed_wall_seconds: float
    summed_process_cpu_seconds: float
    summed_iterations: int
    median_final_rms: float
    geometric_mean_final_rms: float
    recovery_rate: float


@dataclass(frozen=True)
class DownstreamGateVerdict:
    passed: bool
    paired_successful_objects: int
    speed_ratio: float
    speed_ratio_ci95_low: float
    recovery_rate_inferiority: float
    geometric_mean_rms_ratio: float
    geometric_mean_rms_ratio_ci95_high: float
    failures: tuple[str, ...]

    def as_mapping(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "paired_successful_objects": self.paired_successful_objects,
            "speed_ratio": self.speed_ratio,
            "speed_ratio_ci95_low": self.speed_ratio_ci95_low,
            "recovery_rate_inferiority": self.recovery_rate_inferiority,
            "geometric_mean_rms_ratio": self.geometric_mean_rms_ratio,
            "geometric_mean_rms_ratio_ci95_high": self.geometric_mean_rms_ratio_ci95_high,
            "failures": list(self.failures),
        }


def aggregate_arm(
    arm: str,
    wall_seconds: Sequence[float],
    cpu_seconds: Sequence[float],
    iterations: Sequence[int],
    final_rms: Sequence[float],
    reference_rms: Sequence[float],
) -> ArmAggregate:
    wall, cpu, iteration_values, rms, reference = map(
        lambda values: np.asarray(values),
        (wall_seconds, cpu_seconds, iterations, final_rms, reference_rms),
    )
    if not (wall.ndim == cpu.ndim == iteration_values.ndim == rms.ndim == reference.ndim == 1) or not (wall.size == cpu.size == iteration_values.size == rms.size == reference.size):
        raise DownstreamBenchmarkError("downstream vectors must be aligned one-object arrays")
    # Failed fits are represented by NaN RMS and remain in the denominator for
    # recovery accounting. Timing and reference values must be valid for all
    # objects; only successful RMS values are required to be finite.
    if wall.size == 0 or np.any(~np.isfinite(wall)) or np.any(~np.isfinite(cpu)) or np.any(~np.isfinite(iteration_values)) or np.any(~np.isfinite(reference)) or np.any(reference <= 0) or np.any(iteration_values < 0):
        raise DownstreamBenchmarkError("downstream timing/RMS vectors are invalid")
    successes = np.isfinite(rms)
    if np.any(np.isinf(rms)) or np.any(rms[successes] < 0):
        raise DownstreamBenchmarkError("successful downstream RMS values must be finite and non-negative")
    successful_rms = rms[successes]
    if not successful_rms.size:
        return ArmAggregate(arm, 0, int(rms.size), float(np.sum(wall)), float(np.sum(cpu)), int(np.sum(iteration_values)), float("inf"), float("inf"), 0.0)
    return ArmAggregate(
        arm,
        int(successes.sum()),
        int((~successes).sum()),
        float(np.sum(wall)),
        float(np.sum(cpu)),
        int(np.sum(iteration_values)),
        float(np.median(successful_rms)),
        float(np.exp(np.mean(np.log(np.maximum(successful_rms, 1e-12))))),
        float(np.mean(successful_rms <= 1.05 * reference[successes])),
    )


def _bootstrap_speed_ratio(
    baseline_time: np.ndarray, candidate_time: np.ndarray, *, resamples: int, seed: int
) -> tuple[float, float]:
    if baseline_time.shape != candidate_time.shape or baseline_time.size < 2:
        raise DownstreamBenchmarkError("speed bootstrap requires aligned timing vectors")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, baseline_time.size, size=(resamples, baseline_time.size))
    ratios = np.sum(baseline_time[indices], axis=1) / np.maximum(np.sum(candidate_time[indices], axis=1), 1e-12)
    return float(np.mean(baseline_time) / max(float(np.mean(candidate_time)), 1e-12)), float(np.quantile(ratios, 0.025))


def evaluate_downstream_gate(
    baseline_wall_seconds: Sequence[float],
    candidate_wall_seconds: Sequence[float],
    baseline_rms: Sequence[float],
    candidate_rms: Sequence[float],
    *,
    baseline_success: Sequence[bool] | None = None,
    candidate_success: Sequence[bool] | None = None,
    bootstrap_resamples: int = 10000,
    seed: int = 20260901,
) -> DownstreamGateVerdict:
    """Apply fixed-period acceleration/recovery/RMS gates including neural time."""
    baseline_time, candidate_time = np.asarray(baseline_wall_seconds, dtype=np.float64), np.asarray(candidate_wall_seconds, dtype=np.float64)
    baseline_values, candidate_values = np.asarray(baseline_rms, dtype=np.float64), np.asarray(candidate_rms, dtype=np.float64)
    if baseline_time.shape != candidate_time.shape or baseline_values.shape != candidate_values.shape or baseline_time.ndim != 1 or baseline_time.size < 2:
        raise DownstreamBenchmarkError("downstream arm vectors must be aligned and contain at least two objects")
    if not np.all(np.isfinite(baseline_time)) or not np.all(np.isfinite(candidate_time)) or np.any(baseline_time <= 0) or np.any(candidate_time <= 0):
        raise DownstreamBenchmarkError("downstream timing must be finite and positive")
    if baseline_success is None:
        baseline_success_array = np.isfinite(baseline_values) & (baseline_values > 0)
    else:
        baseline_success_array = np.asarray(baseline_success, dtype=bool)
    if candidate_success is None:
        candidate_success_array = np.isfinite(candidate_values) & (candidate_values > 0)
    else:
        candidate_success_array = np.asarray(candidate_success, dtype=bool)
    if baseline_success_array.shape != baseline_time.shape or candidate_success_array.shape != candidate_time.shape:
        raise DownstreamBenchmarkError("success masks must align with timing vectors")
    if (
        np.any(~np.isfinite(baseline_values[baseline_success_array]))
        or np.any(baseline_values[baseline_success_array] <= 0)
        or np.any(~np.isfinite(candidate_values[candidate_success_array]))
        or np.any(candidate_values[candidate_success_array] <= 0)
    ):
        raise DownstreamBenchmarkError(
            "rows marked successful must have finite positive RMS values"
        )
    paired_success = baseline_success_array & candidate_success_array
    if int(np.sum(paired_success)) < 2:
        raise DownstreamBenchmarkError(
            "RMS noninferiority requires at least two paired successful objects"
        )
    ratio, ratio_low = _bootstrap_speed_ratio(
        baseline_time,
        candidate_time,
        resamples=bootstrap_resamples,
        seed=seed,
    )
    rms_ratio_values = (
        candidate_values[paired_success] / baseline_values[paired_success]
    )
    geometric_ratio = float(np.exp(np.mean(np.log(rms_ratio_values))))
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        rms_ratio_values.size,
        size=(bootstrap_resamples, rms_ratio_values.size),
    )
    ratio_high = float(
        np.quantile(
            np.exp(np.mean(np.log(rms_ratio_values[indices]), axis=1)), 0.975
        )
    )
    baseline_recovery, candidate_recovery = float(np.mean(baseline_success_array)), float(np.mean(candidate_success_array))
    inferiority = baseline_recovery - candidate_recovery
    failures: list[str] = []
    if ratio < 1.25 or ratio_low < 1.0:
        failures.append("candidate wall-time reduction is below 25 percent or its CI includes no speedup")
    if inferiority > 0.02:
        failures.append("candidate recovery rate is more than two percentage points below baseline")
    if ratio_high > 1.01:
        failures.append("candidate geometric-mean RMS ratio CI exceeds 1.01")
    return DownstreamGateVerdict(
        not failures,
        int(np.sum(paired_success)),
        ratio,
        ratio_low,
        inferiority,
        geometric_ratio,
        ratio_high,
        tuple(failures),
    )


def _atomic_json(path: Path, payload: Mapping[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(path)


def _load_benchmark_inputs(
    ensemble_path: str | Path,
    catalog_path: str | Path,
    dump_root: str | Path,
    split_path: str | Path,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, dict[str, dict[str, object]]]:
    try:
        split = json.loads(Path(split_path).read_text(encoding="utf-8"))
        fold_rows = sorted(split["folds"], key=lambda row: int(row["fold"]))
        expected_ids = tuple(
            object_id for row in fold_rows for object_id in row["test_ids"]
        )
        catalog = {
            row["object_id"]: row
            for row in (
                json.loads(line)
                for line in Path(catalog_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
        with np.load(ensemble_path, allow_pickle=False) as artifact:
            schema = str(artifact["schema"].item())
            object_ids = tuple(artifact["object_ids"].astype(str))
            axes = np.asarray(artifact["refined_axes"], dtype=np.float64)
            inference = np.asarray(
                artifact["inference_wall_seconds"], dtype=np.float64
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise DownstreamBenchmarkError(
            f"cannot load downstream benchmark inputs: {exc}"
        ) from exc
    if (
        [int(row["fold"]) for row in fold_rows] != list(range(5))
        or len(expected_ids) != len(set(expected_ids))
        or schema != "delphi.k3-real-oof-ensemble.v1"
        or object_ids != expected_ids
        or axes.shape != (len(expected_ids), 3, 3)
        or inference.shape != (len(expected_ids),)
        or not np.all(np.isfinite(axes))
        or not np.all(np.isfinite(inference))
        or np.any(inference <= 0)
        or any(object_id not in catalog for object_id in expected_ids)
    ):
        raise DownstreamBenchmarkError(
            "ensemble, split, and catalog inputs are not exactly aligned"
        )
    root = Path(dump_root)
    for object_id in expected_ids:
        row = catalog[object_id]
        try:
            lightcurve = root / str(row["lightcurve"]["source_path"])
            expected_hash = str(row["lightcurve"]["source_sha256"])
        except (KeyError, TypeError) as exc:
            raise DownstreamBenchmarkError(
                f"catalog lightcurve metadata is invalid for {object_id}"
            ) from exc
        if not lightcurve.is_file() or sha256_file(lightcurve) != expected_hash:
            raise DownstreamBenchmarkError(
                f"lightcurve does not match the frozen catalog for {object_id}"
            )
    return expected_ids, axes, inference, catalog


def _run_record(
    *,
    executable: Path,
    source_root: Path,
    lightcurve: Path,
    output_root: Path,
    object_id: str,
    arm: str,
    start_index: int,
    start: DownstreamStart,
    timeout_seconds: float,
    compiler_command: Sequence[str],
) -> dict[str, object]:
    run_directory = output_root / "runs" / object_id / arm / f"start-{start_index}"
    parameter_path = run_directory / "parameters.txt"
    record_path = run_directory / "result.json"
    parameters = ConvexinvParameters(
        lambda_deg=start.lambda_deg,
        beta_deg=start.beta_deg,
        period_hours=start.period_hours,
        free_lambda=True,
        free_beta=True,
        free_period=False,
    )
    expected_parameter_text = parameters.render()
    if record_path.exists():
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DownstreamBenchmarkError(
                f"cannot resume benchmark record {record_path}: {exc}"
            ) from exc
        identity = record.get("identity", {})
        result = record.get("result", {})
        provenance = result.get("provenance", {}) if isinstance(result, dict) else {}
        if (
            identity
            != {
                "object_id": object_id,
                "arm": arm,
                "start_index": start_index,
                "lambda_deg": start.lambda_deg,
                "beta_deg": start.beta_deg,
                "period_hours": start.period_hours,
            }
            or not parameter_path.is_file()
            or parameter_path.read_text(encoding="ascii") != expected_parameter_text
            or provenance.get("executable_sha256") != sha256_file(executable)
            or provenance.get("input_sha256") != sha256_file(lightcurve)
            or provenance.get("parameter_sha256") != sha256_file(parameter_path)
        ):
            raise DownstreamBenchmarkError(
                f"existing benchmark record does not match this run: {record_path}"
            )
        return record
    if run_directory.exists() and any(run_directory.iterdir()):
        raise DownstreamBenchmarkError(
            f"incomplete benchmark directory requires manual audit: {run_directory}"
        )
    write_convexinv_parameters(parameter_path, parameters)
    result = run_convexinv(
        executable=executable,
        source_root=source_root,
        lightcurve_file=lightcurve,
        parameter_file=parameter_path,
        output_directory=run_directory,
        timeout_seconds=timeout_seconds,
        compiler_command=compiler_command,
    )
    payload: dict[str, object] = {
        "schema": "delphi.k3-convexinv-run.v1",
        "identity": {
            "object_id": object_id,
            "arm": arm,
            "start_index": start_index,
            "lambda_deg": start.lambda_deg,
            "beta_deg": start.beta_deg,
            "period_hours": start.period_hours,
        },
        "result": asdict(result),
    }
    _atomic_json(record_path, payload)
    return payload


def _valid_result(record: Mapping[str, object]) -> bool:
    result = record.get("result")
    if not isinstance(result, Mapping):
        return False
    values = (
        result.get("relative_rms_from_output"),
        result.get("final_lambda_deg"),
        result.get("final_beta_deg"),
    )
    return bool(
        result.get("return_code") == 0
        and result.get("timed_out") is False
        and all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in values
        )
        and float(values[0]) > 0
    )


def run_fixed_period_benchmark(
    *,
    executable: str | Path,
    source_root: str | Path,
    source_archive: str | Path,
    ensemble_path: str | Path,
    catalog_path: str | Path,
    dump_root: str | Path,
    split_path: str | Path,
    output_directory: str | Path,
    timeout_seconds: float = 3600.0,
    compiler_command: Sequence[str] = ("cc", "--version"),
) -> dict[str, object]:
    """Run matched six-start DAMIT-standard and K3 fixed-period arms.

    Completed start records are immutable and may be resumed. Arm order is
    deterministically balanced by object hash; every child process runs
    serially. The deployable K3 inference time retained in the ensemble
    artifact is added once to each object's guided-arm wall time.
    """
    binary = Path(executable).resolve()
    source = Path(source_root).resolve()
    archive = Path(source_archive).resolve()
    destination = Path(output_directory).resolve()
    summary_path = destination / "fixed-period-summary.json"
    rows_path = destination / "fixed-period-rows.json"
    if summary_path.exists() or rows_path.exists():
        raise DownstreamBenchmarkError(
            "refusing to overwrite completed fixed-period benchmark outputs"
        )
    if not binary.is_file() or not source.is_dir() or not archive.is_file():
        raise DownstreamBenchmarkError(
            "convexinv executable, source root, and downloaded archive must exist"
        )
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise DownstreamBenchmarkError("timeout must be finite and positive")
    object_ids, axes, inference_times, catalog = _load_benchmark_inputs(
        ensemble_path, catalog_path, dump_root, split_path
    )
    rows: list[dict[str, object]] = []
    baseline_times: list[float] = []
    candidate_times: list[float] = []
    baseline_rms: list[float] = []
    candidate_rms: list[float] = []
    baseline_success: list[bool] = []
    candidate_success: list[bool] = []
    source_tree_hashes: set[str] = set()
    executable_hashes: set[str] = set()
    compiler_versions: set[str] = set()
    root = Path(dump_root)
    order_rank = {
        object_id: rank
        for rank, object_id in enumerate(
            sorted(
                object_ids,
                key=lambda value: hashlib.sha256(
                    f"{K3_PROTOCOL_SHA256}:{value}".encode("utf-8")
                ).hexdigest(),
            )
        )
    }
    for object_index, object_id in enumerate(object_ids):
        row = catalog[object_id]
        solutions = row.get("solutions")
        if not isinstance(solutions, list) or not solutions:
            raise DownstreamBenchmarkError(
                f"catalog solutions are invalid for {object_id}"
            )
        try:
            period_hours = float(solutions[0]["period_hours"])
            targets = np.asarray(
                [solution["vector"] for solution in solutions], dtype=np.float64
            )
            lightcurve = root / str(row["lightcurve"]["source_path"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DownstreamBenchmarkError(
                f"catalog solution metadata is invalid for {object_id}"
            ) from exc
        if (
            not math.isfinite(period_hours)
            or period_hours <= 0
            or targets.ndim != 2
            or targets.shape[1] != 3
            or not np.all(np.isfinite(targets))
        ):
            raise DownstreamBenchmarkError(
                f"catalog targets are invalid for {object_id}"
            )
        starts = {
            "baseline": tuple(
                DownstreamStart(longitude, latitude, period_hours)
                for longitude, latitude in DAMIT_STANDARD_STARTS_DEG
            ),
            "candidate": signed_starts_from_axes(
                axes[object_index], period_hours=period_hours
            ),
        }
        first_arm = "baseline" if order_rank[object_id] % 2 == 0 else "candidate"
        arm_order = (first_arm, "candidate" if first_arm == "baseline" else "baseline")
        arm_records: dict[str, list[dict[str, object]]] = {
            "baseline": [],
            "candidate": [],
        }
        for arm in arm_order:
            for start_index, start in enumerate(starts[arm]):
                arm_records[arm].append(
                    _run_record(
                        executable=binary,
                        source_root=source,
                        lightcurve=lightcurve,
                        output_root=destination,
                        object_id=object_id,
                        arm=arm,
                        start_index=start_index,
                        start=start,
                        timeout_seconds=timeout_seconds,
                        compiler_command=compiler_command,
                    )
                )
        for record in arm_records["baseline"] + arm_records["candidate"]:
            try:
                provenance = record["result"]["provenance"]
                source_tree_hashes.add(str(provenance["source_tree_sha256"]))
                executable_hashes.add(str(provenance["executable_sha256"]))
                compiler_versions.add(str(provenance["compiler_version"]))
            except (KeyError, TypeError) as exc:
                raise DownstreamBenchmarkError(
                    f"run provenance is missing for {object_id}"
                ) from exc
        object_row: dict[str, object] = {
            "object_id": object_id,
            "period_hours": period_hours,
            "arm_order": list(arm_order),
            "neural_inference_wall_seconds": float(inference_times[object_index]),
        }
        for arm in ("baseline", "candidate"):
            records = arm_records[arm]
            valid = [record for record in records if _valid_result(record)]
            best = min(
                valid,
                key=lambda record: (
                    float(record["result"]["relative_rms_from_output"]),
                    int(record["identity"]["start_index"]),
                ),
                default=None,
            )
            wall = float(
                sum(float(record["result"]["wall_time_seconds"]) for record in records)
            )
            if arm == "candidate":
                wall += float(inference_times[object_index])
            if best is None:
                rms = float("nan")
                pole_error = float("nan")
                success = False
                best_start = None
            else:
                result = best["result"]
                rms = float(result["relative_rms_from_output"])
                longitude = math.radians(float(result["final_lambda_deg"]))
                latitude = math.radians(float(result["final_beta_deg"]))
                pole = np.asarray(
                    [
                        math.cos(latitude) * math.cos(longitude),
                        math.cos(latitude) * math.sin(longitude),
                        math.sin(latitude),
                    ]
                )
                pole_error = float(
                    np.min(axial_angular_error_deg(pole[None, :], targets))
                )
                success = pole_error <= 20.0
                best_start = int(best["identity"]["start_index"])
            object_row[arm] = {
                "wall_seconds": wall,
                "process_cpu_seconds": float(
                    sum(
                        float(record["result"]["process_cpu_time_seconds"] or 0.0)
                        for record in records
                    )
                ),
                "iterations": int(
                    sum(int(record["result"]["iterations"] or 0) for record in records)
                ),
                "valid_starts": len(valid),
                "best_start_index": best_start,
                "final_rms": rms,
                "pole_error_deg": pole_error,
                "success": success,
            }
            if arm == "baseline":
                baseline_times.append(wall)
                baseline_rms.append(rms)
                baseline_success.append(success)
            else:
                candidate_times.append(wall)
                candidate_rms.append(rms)
                candidate_success.append(success)
        rows.append(object_row)
    if (
        len(source_tree_hashes) != 1
        or len(executable_hashes) != 1
        or len(compiler_versions) != 1
        or executable_hashes != {sha256_file(binary)}
    ):
        raise DownstreamBenchmarkError(
            "fixed-period records contain mixed source/executable/compiler provenance"
        )
    verdict = evaluate_downstream_gate(
        baseline_times,
        candidate_times,
        baseline_rms,
        candidate_rms,
        baseline_success=baseline_success,
        candidate_success=candidate_success,
    )
    rows_payload: dict[str, object] = {
        "schema": "delphi.k3-fixed-period-rows.v1",
        "object_ids": list(object_ids),
        "rows": rows,
    }
    rows_sha256 = _atomic_json(rows_path, rows_payload)
    summary: dict[str, object] = {
        "schema": "delphi.k3-fixed-period-summary.v1",
        "object_count": len(object_ids),
        "starts_per_arm": 6,
        "period_policy": "first frozen catalog solution",
        "baseline_starts_deg": [list(value) for value in DAMIT_STANDARD_STARTS_DEG],
        "candidate_starts": "three refined axial predictions expanded to both signs",
        "arm_order": "protocol-hash/object-id ranked balanced alternation",
        "selection": "minimum final relative RMS within each arm; stable start-index tie break",
        "recovery": "selected run completes and is within 20 axial degrees of any catalog solution",
        "neural_time_included": True,
        "timeout_seconds": timeout_seconds,
        "ensemble_sha256": sha256_file(ensemble_path),
        "catalog_sha256": sha256_file(catalog_path),
        "splits_sha256": sha256_file(split_path),
        "executable_sha256": sha256_file(binary),
        "source_archive_sha256": sha256_file(archive),
        "source_tree_sha256": next(iter(source_tree_hashes)),
        "compiler_version": next(iter(compiler_versions)),
        "rows_sha256": rows_sha256,
        "verdict": verdict.as_mapping(),
    }
    summary["artifact_sha256"] = _atomic_json(summary_path, summary)
    return summary
