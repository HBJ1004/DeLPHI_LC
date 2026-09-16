"""Create-once, label-blind reinterpretation of archived convergence runs.

This module is intentionally separate from :mod:`convergence_study`.  It reads
the hash-bound historical execution graph, keeps every raw result unchanged,
and writes a new selection artifact using the solver-angle decoder introduced
after development execution.  It never opens historical score files or a
reference-axis catalog.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..physics.axial import axial_angular_error_deg
from ..v2.data import canonical_json, sha256_file
from .convergence_axis import (
    AXIS_INTERPRETATION_VERSION,
    ConvergenceAxisError,
    decode_convergence_axis,
)
from .convergence_benchmark import CONVERGENCE_ITERATION_CAP, _completion
from .convergence_study import (
    _paired_binary_noninferiority,
    _paired_rms_ratio,
    _stratified_ratio_bootstrap,
)
from .downstream import DownstreamBenchmarkError

REINTERPRETATION_SCHEMA = "delphi.k3-convergence-angle-reinterpretation.v1"
REINTERPRETATION_RECEIPT = "reinterpretation-receipt.json"
REINTERPRETATION_SCORE_SCHEMA = "delphi.k3-convergence-angle-reinterpretation-score.v1"
EXPECTED_STUDY_SPEC_SHA256 = "ff97151e38584f5b2f5448e9b1f3ab7b6e9a6274eb51cf2a5007280937e971a3"
EXPECTED_REVISED_LOCK_SHA256 = "6388670f1b3c49862176ebcab5669c86d662e8854536b6328069213315fd9e8a"
EXPECTED_PARENT_SELECTION_SHA256 = (
    "68bddcea938fdfef9a61347de7eac6611531e86a3122151ee3984d1e7e0b7d84"
)
TOLERANCES = (0.01, 0.003, 0.001, 0.0003, 0.0001)
EXPECTED_LEGACY_NUMERIC_FAILURES = {0.01: 3, 0.003: 9, 0.001: 24, 0.0003: 30, 0.0001: 39}


class ConvergenceReinterpretationError(DownstreamBenchmarkError):
    """A historical convergence graph cannot be safely reinterpreted."""


def _read_json(path: Path, description: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConvergenceReinterpretationError(f"cannot read {description}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConvergenceReinterpretationError(f"{description} must be a JSON object")
    return value


def _is_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _safe_child(root: Path, relative: object, *, description: str) -> Path:
    if not isinstance(relative, str):
        raise ConvergenceReinterpretationError(f"{description} path is not text")
    candidate = Path(relative)
    resolved = (root / candidate).resolve()
    if candidate.is_absolute() or not resolved.is_relative_to(root):
        raise ConvergenceReinterpretationError(f"{description} path escapes its artifact root")
    return resolved


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _legacy_completion(result: Mapping[str, object], *, expected_period_hours: float) -> str:
    """The archived classifier, retained solely to audit its stored status."""
    if result.get("timed_out") is True:
        return "timeout"
    if result.get("adapter_error") is not None:
        return "adapter-error"
    if result.get("return_code") != 0:
        return "exit-nonzero"
    iterations = result.get("iterations")
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations <= 0:
        return "missing-iteration-log"
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
        _finite_number(result.get(key)) for key in numeric
    ):
        return "numerical-output-failure"
    if (
        float(result["chi2"]) < 0.0
        or float(result["deviation"]) < 0.0
        or float(result["relative_rms_from_output"]) <= 0.0
        or not -90.0 <= float(result["final_beta_deg"]) <= 90.0
        or not math.isclose(
            float(result["final_period_hours"]), expected_period_hours, rel_tol=1e-7, abs_tol=1e-9
        )
        or any(
            not _is_hash(result.get(key))
            for key in ("output_lightcurve_sha256", "output_parameter_sha256", "output_area_sha256")
        )
    ):
        return "numerical-output-failure"
    return "converged"


def _legacy_numeric_reasons(
    result: Mapping[str, object], *, expected_period_hours: float
) -> list[str]:
    """Return every old numerical-output condition, rather than first failure."""
    reasons: list[str] = []
    numeric = (
        "chi2",
        "deviation",
        "final_lambda_deg",
        "final_beta_deg",
        "final_period_hours",
        "relative_rms_from_output",
    )
    if result.get("output_validation_error") is not None:
        reasons.append("output_validation_error")
    if not all(_finite_number(result.get(key)) for key in numeric):
        reasons.append("nonfinite_or_invalid_numeric_field")
        return reasons
    if float(result["chi2"]) < 0.0:
        reasons.append("negative_chi2")
    if float(result["deviation"]) < 0.0:
        reasons.append("negative_deviation")
    if float(result["relative_rms_from_output"]) <= 0.0:
        reasons.append("nonpositive_relative_rms")
    if not -90.0 <= float(result["final_beta_deg"]) <= 90.0:
        reasons.append("latitude_outside_standard_range")
    if not math.isclose(
        float(result["final_period_hours"]), expected_period_hours, rel_tol=1e-7, abs_tol=1e-9
    ):
        reasons.append("fixed_period_mismatch")
    if any(
        not _is_hash(result.get(key))
        for key in ("output_lightcurve_sha256", "output_parameter_sha256", "output_area_sha256")
    ):
        reasons.append("invalid_output_hash")
    return reasons


def _add_snapshot(
    snapshot: dict[Path, str], path: Path, *, expected: object, description: str
) -> None:
    if not path.is_file() or not _is_hash(expected):
        raise ConvergenceReinterpretationError(
            f"{description} is missing or has an invalid expected hash"
        )
    actual = sha256_file(path)
    if actual != expected:
        raise ConvergenceReinterpretationError(f"{description} hash does not match its marker")
    previous = snapshot.setdefault(path, actual)
    if previous != actual:
        raise ConvergenceReinterpretationError(
            f"{description} was referenced with inconsistent hashes"
        )


def _check_lock(
    lock_path: Path, spec_path: Path, manifest_path: Path
) -> tuple[dict[str, object], tuple[str, ...]]:
    if sha256_file(spec_path) != EXPECTED_STUDY_SPEC_SHA256:
        raise ConvergenceReinterpretationError(
            "study specification does not match the required correction parent"
        )
    if sha256_file(lock_path) != EXPECTED_REVISED_LOCK_SHA256:
        raise ConvergenceReinterpretationError(
            "revised study lock does not match the required correction parent"
        )
    lock = _read_json(lock_path, "revised study lock")
    if (
        lock.get("schema") != "delphi.k3-convergence-study-lock.v1"
        or lock.get("study_spec_sha256") != EXPECTED_STUDY_SPEC_SHA256
    ):
        raise ConvergenceReinterpretationError(
            "revised study lock has the wrong schema or study specification"
        )
    if lock.get("development_manifest_sha256") != sha256_file(manifest_path):
        raise ConvergenceReinterpretationError(
            "development manifest does not match revised study lock"
        )
    manifest = _read_json(manifest_path, "development manifest")
    object_ids = manifest.get("object_ids")
    if (
        manifest.get("schema") != "delphi.k3-convergence-object-subset.v1"
        or manifest.get("role") != "development"
        or manifest.get("study_spec_sha256") != EXPECTED_STUDY_SPEC_SHA256
        or not isinstance(object_ids, list)
        or len(object_ids) != 30
        or any(not isinstance(value, str) or not value for value in object_ids)
        or len(set(object_ids)) != 30
    ):
        raise ConvergenceReinterpretationError(
            "development manifest is not the required 30-object cohort"
        )
    return lock, tuple(object_ids)


def _contract_matches_lock(
    contract: Mapping[str, object], lock: Mapping[str, object], *, tolerance: float
) -> bool:
    required = {
        "schema": "delphi.k3-convergence-execution-contract.v1",
        "study_spec_sha256": EXPECTED_STUDY_SPEC_SHA256,
        "study_lock_sha256": EXPECTED_REVISED_LOCK_SHA256,
        "role": "development",
        "convergence_tolerance": tolerance,
        "iteration_cap": 1000,
        "timeout_seconds_per_start": 300,
        "timing_repeats": 3,
        "starts_per_arm": 6,
        "fit_selection": "minimum_final_relative_rms_without_reference_axis_access",
        "period_policy": "fixed_first_frozen_catalog_solution",
        "candidate_semantics": "guided_k3_initialization",
    }
    if any(contract.get(key) != value for key, value in required.items()):
        return False
    for name in (
        "blind_inputs_sha256",
        "cohort_manifest_sha256",
        "neural_timing_sha256",
        "source_archive_sha256",
        "source_payload_sha256",
        "source_tree_sha256",
        "executable_sha256",
        "compiler_command",
        "compiler_executable",
        "compiler_executable_sha256",
        "compiler_version",
    ):
        source = "development_manifest_sha256" if name == "cohort_manifest_sha256" else name
        if contract.get(name) != lock.get(source):
            return False
    return True


def _record_contract(
    contract: Mapping[str, object],
    *,
    object_id: str,
    fold: int,
    lightcurve_sha256: str,
    period_hours: float,
) -> dict[str, object]:
    """Return the archived per-object execution contract binding."""
    return {
        **dict(contract),
        "object_id": object_id,
        "fold": fold,
        "lightcurve_sha256": lightcurve_sha256,
        "period_hours": period_hours,
    }


def _contract_sha256(contract: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json(dict(contract)).encode()).hexdigest()


def _validate_record(
    record: Mapping[str, object],
    *,
    root: Path,
    expected_path: Path,
    expected_hash: object,
    object_id: str,
    fold: int,
    repeat_index: int,
    arm: str,
    start_index: int,
    tolerance: float,
    period_hours: float,
    lightcurve_sha256: str,
    contract: Mapping[str, object],
    snapshot: dict[Path, str],
) -> dict[str, object]:
    _add_snapshot(
        snapshot, expected_path, expected=expected_hash, description="solver execution cell"
    )
    record_value = _read_json(expected_path, "solver execution cell")
    if dict(record) != record_value:
        raise ConvergenceReinterpretationError(
            "repeat marker cell does not match raw solver execution cell"
        )
    identity, result, record_contract, logs = (
        record_value.get("identity"),
        record_value.get("result"),
        record_value.get("execution_contract"),
        record_value.get("logs"),
    )
    if not all(isinstance(value, Mapping) for value in (identity, result, record_contract, logs)):
        raise ConvergenceReinterpretationError("solver execution cell lacks required mappings")
    expected_record_contract = _record_contract(
        contract,
        object_id=object_id,
        fold=fold,
        lightcurve_sha256=lightcurve_sha256,
        period_hours=period_hours,
    )
    if (
        record_value.get("schema") != "delphi.k3-convergence-convexinv-run.v1"
        or identity.get("object_id") != object_id
        or identity.get("arm") != arm
        or identity.get("repeat_index") != repeat_index
        or identity.get("start_index") != start_index
        or identity.get("convergence_tolerance") != tolerance
        or identity.get("timeout_seconds") != 300.0
        or identity.get("period_hours") != period_hours
        or identity.get("repeat_arm_order")
        not in (["baseline", "candidate"], ["candidate", "baseline"])
        # Archived start receipts bind object-specific input fields in addition
        # to the global contract and hash that expanded object-level binding.
        or record_contract != expected_record_contract
        or identity.get("execution_contract_sha256") != _contract_sha256(expected_record_contract)
    ):
        raise ConvergenceReinterpretationError(
            "solver execution identity or contract is inconsistent"
        )
    for name, result_name in (("stdout", "stdout_sha256"), ("stderr", "stderr_sha256")):
        if logs.get(name) != f"{name}.log":
            raise ConvergenceReinterpretationError("solver log path is not the expected local name")
        _add_snapshot(
            snapshot,
            _safe_child(expected_path.parent, logs[name], description="solver log"),
            expected=logs.get(result_name),
            description="solver log",
        )
        if result.get(result_name) != logs.get(result_name):
            raise ConvergenceReinterpretationError("solver result and log receipt disagree")
    for filename, field in (
        ("parameters.txt", "parameter_sha256"),
        ("modelled-lightcurve.txt", "output_lightcurve_sha256"),
        ("solution-parameters.txt", "output_parameter_sha256"),
        ("solution-areas.txt", "output_area_sha256"),
    ):
        expected = (
            result.get("provenance", {}).get(field)
            if field == "parameter_sha256" and isinstance(result.get("provenance"), Mapping)
            else result.get(field)
        )
        _add_snapshot(
            snapshot,
            expected_path.parent / filename,
            expected=expected,
            description="solver product",
        )
    provenance = result.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or any(
            provenance.get(key) != contract.get(key)
            for key in ("executable_sha256", "source_tree_sha256")
        )
        or provenance.get("input_sha256") != lightcurve_sha256
    ):
        raise ConvergenceReinterpretationError(
            "solver product provenance does not bind binary, source, and input"
        )
    raw = _legacy_completion(result, expected_period_hours=period_hours)
    if record_value.get("completion") != raw:
        raise ConvergenceReinterpretationError(
            "stored raw completion is not reproducible from raw solver output"
        )
    interpreted = _completion(result, expected_period_hours=period_hours)
    reasons = (
        _legacy_numeric_reasons(result, expected_period_hours=period_hours)
        if raw == "numerical-output-failure"
        else []
    )
    if raw == "numerical-output-failure" and (
        reasons != ["latitude_outside_standard_range"] or interpreted != "converged"
    ):
        raise ConvergenceReinterpretationError(
            "a legacy numerical failure has a reason other than the latitude-range defect"
        )
    if raw != "numerical-output-failure" and interpreted != raw:
        raise ConvergenceReinterpretationError(
            "reinterpretation changed a non-angle completion state"
        )
    axis: dict[str, object] | None = None
    if interpreted == "converged":
        try:
            decoded = decode_convergence_axis(
                result.get("final_lambda_deg"), result.get("final_beta_deg")
            )
        except ConvergenceAxisError as exc:  # _completion should already have rejected this.
            raise ConvergenceReinterpretationError(
                "converged record lacks a decodable direction"
            ) from exc
        round_trip = decode_convergence_axis(decoded.standard_lambda_deg, decoded.standard_beta_deg)
        maximum = max(
            abs(left - right)
            for left, right in zip(
                decoded.directed_unit_vector, round_trip.directed_unit_vector, strict=True
            )
        )
        if maximum > 1e-12:
            raise ConvergenceReinterpretationError(
                "decoded direction is not preserved by standard coordinates"
            )
        axis = {
            "raw_lambda_deg": decoded.raw_lambda_deg,
            "raw_beta_deg": decoded.raw_beta_deg,
            "standard_lambda_deg": decoded.standard_lambda_deg,
            "standard_beta_deg": decoded.standard_beta_deg,
            "directed_unit_vector": list(decoded.directed_unit_vector),
            "round_trip_component_difference_maximum": maximum,
            "interpretation_version": decoded.interpretation_version,
        }
    return {
        "arm": arm,
        "start_index": start_index,
        "raw_record_sha256": sha256_file(expected_path),
        "raw_completion": raw,
        "interpreted_completion": interpreted,
        "legacy_numerical_failure_reasons": reasons,
        "axis": axis,
        "final_relative_rms": result.get("relative_rms_from_output"),
        "raw_final_lambda_deg": result.get("final_lambda_deg"),
        "raw_final_beta_deg": result.get("final_beta_deg"),
    }


def _select(starts: Sequence[Mapping[str, object]]) -> dict[str, object] | None:
    eligible = [row for row in starts if row.get("interpreted_completion") == "converged"]
    if not eligible:
        return None
    best = min(
        eligible, key=lambda row: (float(row["final_relative_rms"]), int(row["start_index"]))
    )
    axis = best["axis"]
    assert isinstance(axis, Mapping)
    return {
        "criterion": "minimum_final_relative_rms",
        "start_index": int(best["start_index"]),
        "final_relative_rms": float(best["final_relative_rms"]),
        "final_lambda_deg": float(axis["standard_lambda_deg"]),
        "final_beta_deg": float(axis["standard_beta_deg"]),
        "axis_interpretation_version": AXIS_INTERPRETATION_VERSION,
    }


def _arm_summary(starts: Sequence[Mapping[str, object]]) -> dict[str, object]:
    statuses = [str(row["interpreted_completion"]) for row in starts]
    return {
        "starts_requested": 6,
        "raw_completion_counts": dict(
            sorted(Counter(str(row["raw_completion"]) for row in starts).items())
        ),
        "interpreted_completion_counts": dict(sorted(Counter(statuses).items())),
        "completed": all(status == "converged" for status in statuses),
        "selectable": _select(starts) is not None,
        "selected_fit": _select(starts),
    }


def _validate_raw_arm_summary(summary: object, starts: Sequence[Mapping[str, object]]) -> None:
    """Confirm the archived summary is derived from the same raw start records."""
    if (
        not isinstance(summary, Mapping)
        or summary.get("starts_requested") != 6
        or summary.get("all_cells_recorded") is not True
    ):
        raise ConvergenceReinterpretationError("raw arm summary is incomplete")
    raw = [str(row["raw_completion"]) for row in starts]
    old_eligible = [row for row in starts if row["raw_completion"] == "converged"]
    old_selected = None
    if old_eligible:
        best = min(
            old_eligible,
            key=lambda row: (float(row["final_relative_rms"]), int(row["start_index"])),
        )
        old_selected = {
            "criterion": "minimum_final_relative_rms",
            "start_index": int(best["start_index"]),
            "final_relative_rms": float(best["final_relative_rms"]),
            "final_lambda_deg": float(best["raw_final_lambda_deg"]),
            "final_beta_deg": float(best["raw_final_beta_deg"]),
        }
    if (
        summary.get("completion_counts") != dict(sorted(Counter(raw).items()))
        or summary.get("completed") is not all(value == "converged" for value in raw)
        or summary.get("selectable") is not (old_selected is not None)
        or summary.get("selected_fit") != old_selected
    ):
        raise ConvergenceReinterpretationError(
            "raw arm summary does not match raw solver start records"
        )
    for name in ("inversion_wall_seconds", "wall_seconds_warm", "wall_seconds_cold"):
        if not _finite_number(summary.get(name)) or float(summary[name]) <= 0.0:
            raise ConvergenceReinterpretationError("raw arm summary has invalid saved timing")


def _validate_execution(
    path: Path, *, lock: Mapping[str, object], object_ids: Sequence[str], snapshot: dict[Path, str]
) -> dict[str, object]:
    _add_snapshot(
        snapshot, path, expected=sha256_file(path), description="blind execution artifact"
    )
    document = _read_json(path, "blind execution artifact")
    tolerance = document.get("convergence_tolerance")
    if (
        not isinstance(tolerance, (int, float))
        or isinstance(tolerance, bool)
        or float(tolerance) not in TOLERANCES
    ):
        raise ConvergenceReinterpretationError(
            "blind execution has an unexpected convergence tolerance"
        )
    tolerance = float(tolerance)
    contract = document.get("execution_contract")
    if (
        document.get("schema") != "delphi.k3-convergence-blind-execution.v1"
        or document.get("phase") != "label_blind_execution_and_minimum_rms_fit_selection"
        or document.get("reference_catalog_opened") is not False
        or document.get("study_lock_sha256") != EXPECTED_REVISED_LOCK_SHA256
        or document.get("role") != "development"
        or document.get("cohort_manifest_sha256") != lock.get("development_manifest_sha256")
        or document.get("object_ids") != list(object_ids)
        or document.get("object_count") != 30
        or not isinstance(contract, Mapping)
        or contract.get("arm_names") != ["baseline", "candidate"]
        or not _contract_matches_lock(contract, lock, tolerance=tolerance)
        or document.get("execution_contract_sha256") != _contract_sha256(contract)
    ):
        raise ConvergenceReinterpretationError(
            "blind execution does not match the required archived contract"
        )
    root = path.parent.resolve()
    rows = document.get("rows")
    if (
        not isinstance(rows, list)
        or len(rows) != 30
        or [row.get("object_id") if isinstance(row, Mapping) else None for row in rows]
        != list(object_ids)
    ):
        raise ConvergenceReinterpretationError(
            "blind execution does not contain the exact development cohort"
        )
    audits: list[dict[str, object]] = []
    legacy_numeric = 0
    for row in rows:
        assert isinstance(row, Mapping)
        object_id, fold, period = row.get("object_id"), row.get("fold"), row.get("period_hours")
        repeats, markers = row.get("repeats"), row.get("repeat_markers")
        if (
            not isinstance(object_id, str)
            or not isinstance(fold, int)
            or fold not in range(5)
            or not _finite_number(period)
            or float(period) <= 0.0
            or not isinstance(repeats, list)
            or not isinstance(markers, list)
            or len(repeats) != 3
            or len(markers) != 3
        ):
            raise ConvergenceReinterpretationError("blind execution row is incomplete")
        if not _is_hash(row.get("lightcurve_sha256")):
            raise ConvergenceReinterpretationError(
                "blind execution row has an invalid lightcurve hash"
            )
        repeat_audits: list[dict[str, object]] = []
        for repeat_index, (repeat, marker_ref) in enumerate(zip(repeats, markers, strict=True)):
            if not isinstance(repeat, Mapping) or not isinstance(marker_ref, Mapping):
                raise ConvergenceReinterpretationError("repeat marker is malformed")
            marker_relative = f"runs/{object_id}/repeat-{repeat_index}/repeat-complete.json"
            marker_path = _safe_child(root, marker_ref.get("path"), description="repeat marker")
            if (
                marker_ref.get("repeat_index") != repeat_index
                or marker_ref.get("path") != marker_relative
            ):
                raise ConvergenceReinterpretationError(
                    "repeat marker does not use its expected path"
                )
            _add_snapshot(
                snapshot,
                marker_path,
                expected=marker_ref.get("sha256"),
                description="repeat marker",
            )
            if (
                _read_json(marker_path, "repeat marker") != dict(repeat)
                or repeat.get("schema") != "delphi.k3-convergence-repeat-complete.v1"
            ):
                raise ConvergenceReinterpretationError("repeat marker content is not hash-bound")
            plan, cells = repeat.get("plan"), repeat.get("cells")
            record_contract = _record_contract(
                contract,
                object_id=object_id,
                fold=fold,
                lightcurve_sha256=str(row["lightcurve_sha256"]),
                period_hours=float(period),
            )
            if (
                not isinstance(plan, Mapping)
                or not isinstance(cells, list)
                or len(cells) != 12
                or plan.get("object_id") != object_id
                or plan.get("fold") != fold
                or plan.get("repeat_index") != repeat_index
                or plan.get("convergence_tolerance") != tolerance
                or plan.get("timeout_seconds_per_start") != 300.0
                or plan.get("execution_contract_sha256") != _contract_sha256(record_contract)
            ):
                raise ConvergenceReinterpretationError(
                    "repeat plan does not bind the execution contract"
                )
            arm_order = plan.get("arm_order")
            if arm_order not in (["baseline", "candidate"], ["candidate", "baseline"]):
                raise ConvergenceReinterpretationError(
                    "repeat plan has an invalid paired arm order"
                )
            by_arm: dict[str, list[dict[str, object]]] = defaultdict(list)
            starts: list[dict[str, object]] = []
            seen_cells: set[tuple[str, int]] = set()
            for sequence, cell in enumerate(cells):
                if not isinstance(cell, Mapping):
                    raise ConvergenceReinterpretationError("repeat cell is malformed")
                arm, start = cell.get("arm"), cell.get("start_index")
                if (
                    arm not in ("baseline", "candidate")
                    or not isinstance(start, int)
                    or start not in range(6)
                    or (arm, start) in seen_cells
                ):
                    raise ConvergenceReinterpretationError(
                        "repeat has duplicate or invalid arm/start cell"
                    )
                expected_arm, expected_start = arm_order[sequence % 2], sequence // 2
                expected_relative = (
                    f"runs/{object_id}/repeat-{repeat_index}/{arm}/start-{start}/result.json"
                )
                if (
                    (arm, start) != (expected_arm, expected_start)
                    or cell.get("sequence_index") != sequence
                    or cell.get("path") != expected_relative
                ):
                    raise ConvergenceReinterpretationError(
                        "repeat cells do not match the paired execution order"
                    )
                seen_cells.add((arm, start))
                record_path = _safe_child(
                    root, cell.get("path"), description="solver execution cell"
                )
                record = _read_json(record_path, "solver execution cell")
                audit = _validate_record(
                    record,
                    root=root,
                    expected_path=record_path,
                    expected_hash=cell.get("sha256"),
                    object_id=object_id,
                    fold=fold,
                    repeat_index=repeat_index,
                    arm=arm,
                    start_index=start,
                    tolerance=tolerance,
                    period_hours=float(period),
                    lightcurve_sha256=str(row["lightcurve_sha256"]),
                    contract=contract,
                    snapshot=snapshot,
                )
                legacy_numeric += int(audit["raw_completion"] == "numerical-output-failure")
                starts.append(audit)
                by_arm[arm].append(audit)
            if len(seen_cells) != 12:
                raise ConvergenceReinterpretationError(
                    "repeat does not contain twelve unique cells"
                )
            arms = {
                arm: _arm_summary(sorted(by_arm[arm], key=lambda row: int(row["start_index"])))
                for arm in ("baseline", "candidate")
            }
            for arm in ("baseline", "candidate"):
                raw_summary = repeat.get(arm)
                _validate_raw_arm_summary(
                    raw_summary, sorted(by_arm[arm], key=lambda row: int(row["start_index"]))
                )
                assert isinstance(raw_summary, Mapping)
                arms[arm]["saved_timing"] = {
                    name: float(raw_summary[name])
                    for name in (
                        "inversion_wall_seconds",
                        "wall_seconds_warm",
                        "wall_seconds_cold",
                    )
                }
            repeat_audits.append({"repeat_index": repeat_index, "arms": arms, "starts": starts})
        audits.append(
            {
                "object_id": object_id,
                "fold": fold,
                "period_hours": float(period),
                "repeats": repeat_audits,
            }
        )
    if legacy_numeric != EXPECTED_LEGACY_NUMERIC_FAILURES[tolerance]:
        raise ConvergenceReinterpretationError(
            "legacy numerical-output failure count does not match the archived diagnostic"
        )
    return {
        "tolerance": tolerance,
        "execution_path": str(path),
        "execution_sha256": sha256_file(path),
        "rows": audits,
        "legacy_numerical_output_failures": legacy_numeric,
    }


def _parent_execution_hashes(selection_path: Path) -> dict[float, str]:
    """Read only execution digests from the immutable parent selection receipt."""
    selection = _read_json(selection_path, "parent development selection")
    rows = selection.get("score_artifacts")
    if (
        selection.get("schema") != "delphi.k3-convergence-development-selection.v1"
        or selection.get("status") != "no_eligible_tolerance_stop"
        or selection.get("study_spec_sha256") != EXPECTED_STUDY_SPEC_SHA256
        or selection.get("study_lock_sha256") != EXPECTED_REVISED_LOCK_SHA256
        or not isinstance(rows, list)
        or len(rows) != len(TOLERANCES)
    ):
        raise ConvergenceReinterpretationError(
            "parent selection cannot bind the archived execution graph"
        )
    bindings: dict[float, str] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(
            row.get("convergence_tolerance"), (int, float)
        ):
            raise ConvergenceReinterpretationError(
                "parent selection has an invalid execution binding"
            )
        tolerance = float(row["convergence_tolerance"])
        execution_hash = row.get("execution_sha256")
        if tolerance not in TOLERANCES or tolerance in bindings or not _is_hash(execution_hash):
            raise ConvergenceReinterpretationError(
                "parent selection has an invalid execution binding"
            )
        bindings[tolerance] = str(execution_hash)
    return bindings


def reinterpret_development_execution_graph(
    *,
    parent_root: str | Path,
    output_directory: str | Path,
    study_spec_path: str | Path,
    revised_lock_path: str | Path,
    development_manifest_path: str | Path,
) -> dict[str, object]:
    """Reinterpret all five archived development executions without reference access."""
    parent, output = Path(parent_root).resolve(), Path(output_directory).resolve()
    if output.exists() or output.is_relative_to(parent):
        raise ConvergenceReinterpretationError(
            "reinterpretation output must be new and outside the parent artifact root"
        )
    if not parent.is_dir():
        raise ConvergenceReinterpretationError("parent development root is not a directory")
    lock, object_ids = _check_lock(
        Path(revised_lock_path), Path(study_spec_path), Path(development_manifest_path)
    )
    snapshot: dict[Path, str] = {}
    selection = parent / "selection.json"
    _add_snapshot(
        snapshot,
        selection,
        expected=EXPECTED_PARENT_SELECTION_SHA256,
        description="parent development selection",
    )
    parent_execution_hashes = _parent_execution_hashes(selection)
    labels = {
        tolerance: f"tolerance-{format(tolerance, '.4g').replace('.', 'p')}"
        for tolerance in TOLERANCES
    }
    expected_paths = [
        parent / labels[tolerance] / "blind-execution.json" for tolerance in TOLERANCES
    ]
    if any(not path.is_file() for path in expected_paths):
        raise ConvergenceReinterpretationError(
            "parent development graph lacks one required blind execution artifact"
        )
    executions = [
        _validate_execution(path, lock=lock, object_ids=object_ids, snapshot=snapshot)
        for path in expected_paths
    ]
    if any(
        parent_execution_hashes[float(execution["tolerance"])] != execution["execution_sha256"]
        for execution in executions
    ):
        raise ConvergenceReinterpretationError(
            "parent selection execution hashes do not match blind execution artifacts"
        )
    expected_repeats = len(TOLERANCES) * 30 * 3
    expected_starts = expected_repeats * 12
    if (
        sum(len(row["repeats"]) for execution in executions for row in execution["rows"])
        != expected_repeats
        or sum(
            len(repeat["starts"])
            for execution in executions
            for row in execution["rows"]
            for repeat in row["repeats"]
        )
        != expected_starts
    ):
        raise ConvergenceReinterpretationError(
            "reinterpretation did not audit exactly 450 repeats and 5,400 starts"
        )
    snapshot_before = {str(path): digest for path, digest in sorted(snapshot.items())}
    payload: dict[str, object] = {
        "schema": REINTERPRETATION_SCHEMA,
        "phase": "label_blind_historical_output_reinterpretation",
        "reference_catalog_opened": False,
        "decoder_version": AXIS_INTERPRETATION_VERSION,
        "study_spec_sha256": EXPECTED_STUDY_SPEC_SHA256,
        "revised_study_lock_sha256": EXPECTED_REVISED_LOCK_SHA256,
        "reference_catalog_sha256": lock["reference_catalog_sha256"],
        "parent_root": str(parent),
        "parent_selection_sha256": EXPECTED_PARENT_SELECTION_SHA256,
        "development_manifest_sha256": sha256_file(development_manifest_path),
        "scientific_thresholds_changed": False,
        "raw_artifacts_modified": False,
        "parent_file_sha256_before": snapshot_before,
        "executions": executions,
    }
    staging = output.parent / f".{output.name}.reinterpretation-{uuid.uuid4().hex}"
    try:
        staging.mkdir(parents=True, exist_ok=False)
        receipt = staging / REINTERPRETATION_RECEIPT
        temporary = receipt.with_suffix(".tmp")
        temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
        os.replace(temporary, receipt)
        after = {str(path): sha256_file(path) for path in snapshot}
        if after != snapshot_before:
            raise ConvergenceReinterpretationError(
                "a parent artifact changed during reinterpretation"
            )
        payload["parent_file_sha256_after"] = after
        temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
        os.replace(temporary, receipt)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        **payload,
        "receipt_path": str(output / REINTERPRETATION_RECEIPT),
        "receipt_sha256": sha256_file(output / REINTERPRETATION_RECEIPT),
    }


def _load_reference_targets(
    path: Path, *, expected_hash: object, object_ids: Sequence[str]
) -> dict[str, np.ndarray]:
    if not _is_hash(expected_hash) or sha256_file(path) != expected_hash:
        raise ConvergenceReinterpretationError(
            "reference catalog does not match the sealed study lock"
        )
    requested, targets = set(object_ids), {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            object_id = row.get("object_id")
            if object_id not in requested:
                continue
            values = np.asarray([item["vector"] for item in row["solutions"]], dtype=np.float64)
            norms = np.linalg.norm(values, axis=1)
            if (
                values.ndim != 2
                or values.shape[1] != 3
                or not np.all(np.isfinite(values))
                or np.any(norms <= 1e-12)
            ):
                raise ValueError("invalid reference vector")
            targets[str(object_id)] = values / norms[:, None]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ConvergenceReinterpretationError(
            f"cannot parse sealed reference catalog: {exc}"
        ) from exc
    if set(targets) != requested:
        raise ConvergenceReinterpretationError(
            "reference catalog lacks an interpreted cohort object"
        )
    return targets


def _selected_error(selection: Mapping[str, object] | None, targets: np.ndarray) -> float | None:
    if selection is None:
        return None
    try:
        axis = decode_convergence_axis(selection["final_lambda_deg"], selection["final_beta_deg"])
    except (KeyError, ConvergenceAxisError) as exc:
        raise ConvergenceReinterpretationError("sealed selection lacks a valid direction") from exc
    return float(
        np.min(axial_angular_error_deg(np.asarray(axis.directed_unit_vector)[None, :], targets))
    )


def _score_execution(
    execution: Mapping[str, object], targets: Mapping[str, np.ndarray]
) -> dict[str, object]:
    rows = execution.get("rows")
    if not isinstance(rows, list) or len(rows) != 30:
        raise ConvergenceReinterpretationError(
            "sealed execution lacks the 30-object development cohort"
        )
    fold_rows, base_wall, guided_wall, base_completed, guided_completed = [], [], [], [], []
    base_recovered, guided_recovered, base_rms, guided_rms, joint = [], [], [], [], []
    scored_rows: list[dict[str, object]] = []
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("object_id"), str)
            or not isinstance(row.get("fold"), int)
        ):
            raise ConvergenceReinterpretationError("sealed execution row is malformed")
        object_id, repeats = row["object_id"], row.get("repeats")
        if not isinstance(repeats, list) or len(repeats) != 3:
            raise ConvergenceReinterpretationError("sealed execution lacks three repeats")
        per_repeat = []
        for repeat in repeats:
            if not isinstance(repeat, Mapping) or not isinstance(repeat.get("arms"), Mapping):
                raise ConvergenceReinterpretationError("sealed repeat is malformed")
            arms = repeat["arms"]
            baseline, candidate = arms.get("baseline"), arms.get("candidate")
            if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
                raise ConvergenceReinterpretationError("sealed repeat lacks both arms")
            base_error = _selected_error(baseline.get("selected_fit"), targets[object_id])
            guided_error = _selected_error(candidate.get("selected_fit"), targets[object_id])
            base_wall.append(float(baseline["saved_timing"]["wall_seconds_warm"]))
            guided_wall.append(float(candidate["saved_timing"]["wall_seconds_warm"]))
            base_completed.append(int(baseline.get("completed") is True))
            guided_completed.append(int(candidate.get("completed") is True))
            base_recovered.append(int(base_error is not None and base_error <= 20.0))
            guided_recovered.append(int(guided_error is not None and guided_error <= 20.0))
            base_rms.append(
                float(baseline["selected_fit"]["final_relative_rms"])
                if baseline.get("selected_fit")
                else np.nan
            )
            guided_rms.append(
                float(candidate["selected_fit"]["final_relative_rms"])
                if candidate.get("selected_fit")
                else np.nan
            )
            joint.append(
                bool(
                    baseline.get("completed")
                    and candidate.get("completed")
                    and baseline.get("selectable")
                    and candidate.get("selectable")
                )
            )
            per_repeat.append(
                {
                    "repeat_index": repeat.get("repeat_index"),
                    "baseline_error_deg": base_error,
                    "candidate_error_deg": guided_error,
                }
            )
        fold_rows.extend([int(row["fold"])] * 3)
        scored_rows.append(
            {"object_id": object_id, "fold": int(row["fold"]), "repeats": per_repeat}
        )
    shape = (30, 3)
    base_wall_array, guided_wall_array = (
        np.asarray(base_wall).reshape(shape),
        np.asarray(guided_wall).reshape(shape),
    )
    base_completed_array, guided_completed_array = (
        np.asarray(base_completed).reshape(shape),
        np.asarray(guided_completed).reshape(shape),
    )
    base_recovery_array, guided_recovery_array = (
        np.asarray(base_recovered).reshape(shape),
        np.asarray(guided_recovered).reshape(shape),
    )
    base_rms_array, guided_rms_array, joint_array = (
        np.asarray(base_rms).reshape(shape),
        np.asarray(guided_rms).reshape(shape),
        np.asarray(joint).reshape(shape),
    )
    folds = np.asarray(fold_rows).reshape(shape)[:, 0]
    runtime = _stratified_ratio_bootstrap(
        base_wall_array, guided_wall_array, folds, resamples=10000, seed=20260911
    )
    base_completion_object, guided_completion_object = (
        np.sum(base_completed_array, axis=1) >= 2,
        np.sum(guided_completed_array, axis=1) >= 2,
    )
    base_recovery_object, guided_recovery_object = (
        np.sum(base_recovery_array, axis=1) >= 2,
        np.sum(guided_recovery_array, axis=1) >= 2,
    )
    support = np.sum(np.sum(joint_array, axis=1) >= 2)
    rms_ratio: float | None = None
    if support == 30:
        rms = _paired_rms_ratio(
            base_rms_array,
            guided_rms_array,
            joint_array,
            required_support=30,
            folds=folds,
            resamples=10000,
            seed=20260911,
        )
        rms_ratio = float(rms["point_ratio"])
    else:
        rms = {
            "object_support": int(support),
            "required_object_support": 30,
            "point_ratio": None,
            "estimable": False,
        }
    recovery = _paired_binary_noninferiority(base_recovery_object, guided_recovery_object)
    completion = _paired_binary_noninferiority(base_completion_object, guided_completion_object)
    recovery_difference = float(recovery["point_difference"])
    completion_difference = float(completion["point_difference"])
    eligible = (
        recovery_difference >= -0.05
        and completion_difference >= -0.05
        and rms_ratio is not None
        and rms_ratio <= 1.01
    )
    return {
        "convergence_tolerance": execution["tolerance"],
        "runtime_ratio_warm": runtime["point_ratio"],
        "runtime_empirical_acceptance_lower_5pct": runtime["empirical_acceptance_lower_5pct"],
        "recovery": recovery,
        "completion": completion,
        "geometric_rms_ratio": rms_ratio,
        "rms_object_support": int(support),
        "metrics": {
            "runtime_warm": runtime,
            "recovery": recovery,
            "completion": completion,
            "rms": rms,
        },
        "eligible_under_original_development_rule": eligible,
        "rows": scored_rows,
    }


def score_reinterpreted_development(
    *,
    receipt_path: str | Path,
    expected_receipt_sha256: str,
    reference_catalog_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Score a sealed label-blind reinterpretation; this is the first reference read."""
    receipt, output = Path(receipt_path).resolve(), Path(output_path).resolve()
    if output.exists():
        raise ConvergenceReinterpretationError("reinterpretation score output already exists")
    if (
        not _is_hash(expected_receipt_sha256)
        or not receipt.is_file()
        or sha256_file(receipt) != expected_receipt_sha256
    ):
        raise ConvergenceReinterpretationError(
            "sealed reinterpretation receipt hash does not match"
        )
    document = _read_json(receipt, "sealed reinterpretation receipt")
    if (
        document.get("schema") != REINTERPRETATION_SCHEMA
        or document.get("reference_catalog_opened") is not False
    ):
        raise ConvergenceReinterpretationError(
            "reinterpretation receipt is not a label-blind sealed artifact"
        )
    before, after = (
        document.get("parent_file_sha256_before"),
        document.get("parent_file_sha256_after"),
    )
    if (
        not isinstance(before, Mapping)
        or before != after
        or any(
            not isinstance(path, str) or not _is_hash(digest) or sha256_file(path) != digest
            for path, digest in before.items()
        )
    ):
        raise ConvergenceReinterpretationError("parent execution graph changed after sealing")
    executions = document.get("executions")
    if not isinstance(executions, list) or len(executions) != len(TOLERANCES):
        raise ConvergenceReinterpretationError(
            "sealed receipt lacks all five development executions"
        )
    object_ids = [row["object_id"] for row in executions[0]["rows"]]
    if not all(isinstance(value, str) for value in object_ids):
        raise ConvergenceReinterpretationError("sealed receipt has invalid object identities")
    # Reference axes are intentionally opened only after every receipt/seal check above.
    targets = _load_reference_targets(
        Path(reference_catalog_path),
        expected_hash=document.get("reference_catalog_sha256"),
        object_ids=object_ids,
    )
    scores = [_score_execution(execution, targets) for execution in executions]
    eligible = [score for score in scores if score["eligible_under_original_development_rule"]]
    selected = min(
        eligible,
        key=lambda score: (
            -float(score["runtime_empirical_acceptance_lower_5pct"]),
            float(score["convergence_tolerance"]),
        ),
        default=None,
    )
    payload = {
        "schema": REINTERPRETATION_SCORE_SCHEMA,
        "phase": "reference_axis_scoring_after_sealed_label_blind_reinterpretation",
        "reference_catalog_opened": True,
        "receipt_sha256": expected_receipt_sha256,
        "reference_catalog_sha256": document["reference_catalog_sha256"],
        "original_development_thresholds": {
            "guided_minus_baseline_recovery_point_difference_minimum": -0.05,
            "guided_minus_baseline_completion_point_difference_minimum": -0.05,
            "guided_to_baseline_geometric_mean_rms_ratio_maximum": 1.01,
        },
        "scores": scores,
        "selection": {
            "status": "selected" if selected is not None else "no_eligible_tolerance_stop",
            "selected_tolerance": None if selected is None else selected["convergence_tolerance"],
            "selection_rule": "largest_runtime_ratio_lower_bound_then_smaller_tolerance",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {**payload, "artifact_path": str(output), "artifact_sha256": sha256_file(output)}


__all__ = [
    "ConvergenceReinterpretationError",
    "EXPECTED_PARENT_SELECTION_SHA256",
    "EXPECTED_REVISED_LOCK_SHA256",
    "EXPECTED_STUDY_SPEC_SHA256",
    "REINTERPRETATION_RECEIPT",
    "REINTERPRETATION_SCHEMA",
    "REINTERPRETATION_SCORE_SCHEMA",
    "reinterpret_development_execution_graph",
    "score_reinterpreted_development",
]
