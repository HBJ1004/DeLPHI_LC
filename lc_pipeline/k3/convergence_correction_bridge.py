"""Bind the angle-corrected development decision to a future locked run.

This module is deliberately an authorization bridge, not a solver runner.  It
can create a single immutable authorization after the sealed reinterpretation
has been scored, but it cannot start or resume the 140-object locked cohort.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from ..v2.data import canonical_json, sha256_file
from .convergence_reinterpretation import (
    EXPECTED_REVISED_LOCK_SHA256,
    EXPECTED_STUDY_SPEC_SHA256,
    REINTERPRETATION_SCHEMA,
    REINTERPRETATION_SCORE_SCHEMA,
)
from .downstream import DownstreamBenchmarkError

CORRECTION_LOCKED_AUTHORIZATION_SCHEMA = "delphi.k3-correction-locked-authorization.v1"
EXPECTED_SELECTED_TOLERANCE = 0.0003


class CorrectionLockedAuthorizationError(DownstreamBenchmarkError):
    """The correction-bound locked execution cannot be authorized safely."""


def _read_json(path: Path, description: str) -> dict[str, object]:
    import json

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorrectionLockedAuthorizationError(f"cannot read {description}: {exc}") from exc
    if not isinstance(value, dict):
        raise CorrectionLockedAuthorizationError(f"{description} must be a JSON object")
    return value


def _valid_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _require_hash(path: Path, expected: object, description: str) -> str:
    if not path.is_file() or not _valid_hash(expected):
        raise CorrectionLockedAuthorizationError(
            f"{description} is missing or its expected hash is invalid"
        )
    digest = sha256_file(path)
    if digest != expected:
        raise CorrectionLockedAuthorizationError(f"{description} hash does not match")
    return digest


def _validate_receipt_and_score(
    receipt_path: Path,
    receipt_sha256: str,
    score_path: Path,
    score_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    _require_hash(receipt_path, receipt_sha256, "sealed reinterpretation receipt")
    receipt = _read_json(receipt_path, "sealed reinterpretation receipt")
    if (
        receipt.get("schema") != REINTERPRETATION_SCHEMA
        or receipt.get("phase") != "label_blind_historical_output_reinterpretation"
        or receipt.get("reference_catalog_opened") is not False
        or receipt.get("study_spec_sha256") != EXPECTED_STUDY_SPEC_SHA256
        or receipt.get("revised_study_lock_sha256") != EXPECTED_REVISED_LOCK_SHA256
    ):
        raise CorrectionLockedAuthorizationError(
            "receipt is not the required label-blind correction artifact"
        )

    _require_hash(score_path, score_sha256, "sealed reinterpretation score")
    score = _read_json(score_path, "sealed reinterpretation score")
    selection = score.get("selection")
    if (
        score.get("schema") != REINTERPRETATION_SCORE_SCHEMA
        or score.get("phase") != "reference_axis_scoring_after_sealed_label_blind_reinterpretation"
        or score.get("receipt_sha256") != receipt_sha256
        or not isinstance(selection, Mapping)
        or selection.get("status") != "selected"
        or selection.get("selected_tolerance") != EXPECTED_SELECTED_TOLERANCE
        or selection.get("selection_rule")
        != "largest_runtime_ratio_lower_bound_then_smaller_tolerance"
    ):
        raise CorrectionLockedAuthorizationError(
            "sealed score does not select the required corrected development tolerance"
        )
    return receipt, score


def _validate_locked_manifest_and_lock(
    lock_path: Path, locked_manifest_path: Path, score: Mapping[str, object]
) -> tuple[dict[str, object], tuple[str, ...]]:
    _require_hash(lock_path, EXPECTED_REVISED_LOCK_SHA256, "revised study lock")
    lock = _read_json(lock_path, "revised study lock")
    if (
        lock.get("schema") != "delphi.k3-convergence-study-lock.v1"
        or lock.get("study_spec_sha256") != EXPECTED_STUDY_SPEC_SHA256
        or lock.get("reference_catalog_sha256") != score.get("reference_catalog_sha256")
        or lock.get("locked_manifest_sha256") != sha256_file(locked_manifest_path)
    ):
        raise CorrectionLockedAuthorizationError("revised lock does not bind this locked cohort")
    settings = lock.get("settings")
    if not isinstance(settings, Mapping) or (
        settings.get("convergence_tolerance_grid") != [0.01, 0.003, 0.001, 0.0003, 0.0001]
        or settings.get("iteration_cap") != 1000
        or settings.get("timeout_seconds_per_start") != 300
        or settings.get("timing_repeats") != 3
        or settings.get("include_warm_neural_inference_time") is not True
        or settings.get("report_cold_neural_inference_sensitivity") is not True
        or settings.get("period") != "first_frozen_catalog_solution_fixed_in_both_arms"
    ):
        raise CorrectionLockedAuthorizationError(
            "revised lock does not retain the frozen timing conditions"
        )
    manifest = _read_json(locked_manifest_path, "locked evaluation manifest")
    object_ids = manifest.get("object_ids")
    if (
        manifest.get("schema") != "delphi.k3-convergence-object-subset.v1"
        or manifest.get("role") != "locked_evaluation"
        or manifest.get("study_spec_sha256") != EXPECTED_STUDY_SPEC_SHA256
        or not isinstance(object_ids, list)
        or len(object_ids) != 140
        or len(set(object_ids)) != 140
        or any(not isinstance(object_id, str) or not object_id for object_id in object_ids)
    ):
        raise CorrectionLockedAuthorizationError(
            "locked manifest is not the unchanged 140-object cohort"
        )
    for key in (
        "blind_inputs_sha256",
        "neural_timing_sha256",
        "source_archive_sha256",
        "source_payload_sha256",
        "source_tree_sha256",
        "executable_sha256",
        "compiler_executable_sha256",
    ):
        if not _valid_hash(lock.get(key)):
            raise CorrectionLockedAuthorizationError(
                "revised lock is missing a required frozen resource hash"
            )
    return lock, tuple(object_ids)


def _authorization_payload(
    *,
    receipt_path: Path,
    receipt_sha256: str,
    score_path: Path,
    score_sha256: str,
    lock_path: Path,
    locked_manifest_path: Path,
    locked_execution_directory: Path,
) -> dict[str, object]:
    receipt, score = _validate_receipt_and_score(
        receipt_path, receipt_sha256, score_path, score_sha256
    )
    lock, object_ids = _validate_locked_manifest_and_lock(lock_path, locked_manifest_path, score)
    if locked_execution_directory.exists():
        raise CorrectionLockedAuthorizationError("locked execution directory already exists")
    return {
        "schema": CORRECTION_LOCKED_AUTHORIZATION_SCHEMA,
        "phase": "correction_bound_locked_execution_authorization",
        "solver_execution_started": False,
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha256,
        "score_path": str(score_path),
        "score_sha256": score_sha256,
        "revised_study_lock_path": str(lock_path),
        "revised_study_lock_sha256": EXPECTED_REVISED_LOCK_SHA256,
        "study_spec_sha256": EXPECTED_STUDY_SPEC_SHA256,
        "locked_manifest_path": str(locked_manifest_path),
        "locked_manifest_sha256": sha256_file(locked_manifest_path),
        "locked_object_count": len(object_ids),
        "locked_object_ids": list(object_ids),
        "locked_execution_directory": str(locked_execution_directory),
        "runner_contract": {
            "role": "locked_evaluation",
            "convergence_tolerance": EXPECTED_SELECTED_TOLERANCE,
            "iteration_cap": 1000,
            "timeout_seconds_per_start": 300,
            "timing_repeats": 3,
            "starts_per_arm": 6,
            "arm_names": ["baseline", "candidate"],
            "candidate_semantics": "guided_k3_initialization",
            "period_policy": "fixed_first_frozen_catalog_solution",
            "fit_selection": "minimum_final_relative_rms_without_reference_axis_access",
            "include_warm_neural_inference_time": True,
            "report_cold_neural_inference_sensitivity": True,
            "resource_hashes": {
                key: lock[key]
                for key in (
                    "blind_inputs_sha256",
                    "neural_timing_sha256",
                    "source_archive_sha256",
                    "source_payload_sha256",
                    "source_tree_sha256",
                    "executable_sha256",
                    "compiler_executable_sha256",
                )
            },
        },
    }


def create_correction_locked_authorization(
    *,
    receipt_path: str | Path,
    receipt_sha256: str,
    score_path: str | Path,
    score_sha256: str,
    revised_lock_path: str | Path,
    locked_manifest_path: str | Path,
    locked_execution_directory: str | Path,
    authorization_path: str | Path,
) -> dict[str, object]:
    """Create one immutable authorization for a later, separately approved run."""
    receipt, score, lock, manifest = (
        Path(receipt_path).resolve(),
        Path(score_path).resolve(),
        Path(revised_lock_path).resolve(),
        Path(locked_manifest_path).resolve(),
    )
    destination, authorization = (
        Path(locked_execution_directory).resolve(),
        Path(authorization_path).resolve(),
    )
    if authorization.exists() or authorization.is_relative_to(destination):
        raise CorrectionLockedAuthorizationError(
            "authorization path must be new and outside locked execution directory"
        )
    payload = _authorization_payload(
        receipt_path=receipt,
        receipt_sha256=receipt_sha256,
        score_path=score,
        score_sha256=score_sha256,
        lock_path=lock,
        locked_manifest_path=manifest,
        locked_execution_directory=destination,
    )
    authorization.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(authorization, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as exc:
        raise CorrectionLockedAuthorizationError(
            "correction-bound locked authorization already exists"
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(payload) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise
    return {
        **payload,
        "authorization_path": str(authorization),
        "authorization_sha256": sha256_file(authorization),
    }


def validate_correction_locked_authorization(
    *,
    authorization_path: str | Path,
    receipt_path: str | Path,
    score_path: str | Path,
    revised_lock_path: str | Path,
    locked_manifest_path: str | Path,
) -> dict[str, object]:
    """Revalidate an authorization before a later runner is permitted to use it."""
    authorization = Path(authorization_path).resolve()
    observed = _read_json(authorization, "correction locked authorization")
    if observed.get("schema") != CORRECTION_LOCKED_AUTHORIZATION_SCHEMA:
        raise CorrectionLockedAuthorizationError("authorization has the wrong schema")
    expected = _authorization_payload(
        receipt_path=Path(receipt_path).resolve(),
        receipt_sha256=str(observed.get("receipt_sha256")),
        score_path=Path(score_path).resolve(),
        score_sha256=str(observed.get("score_sha256")),
        lock_path=Path(revised_lock_path).resolve(),
        locked_manifest_path=Path(locked_manifest_path).resolve(),
        locked_execution_directory=Path(
            str(observed.get("locked_execution_directory", ""))
        ).resolve(),
    )
    if observed != expected:
        raise CorrectionLockedAuthorizationError(
            "authorization does not match the sealed correction inputs"
        )
    return {
        **observed,
        "authorization_path": str(authorization),
        "authorization_sha256": sha256_file(authorization),
    }


def validate_correction_locked_authorization_artifact(
    *, authorization_path: str | Path
) -> dict[str, object]:
    """Validate an authorization using only the paths bound into that receipt.

    This is the adapter boundary for the frozen runner.  It still recomputes
    every receipt, score, lock, and manifest hash; it does not treat paths in
    the authorization as trusted merely because the authorization exists.
    """
    authorization = Path(authorization_path).resolve()
    observed = _read_json(authorization, "correction locked authorization")
    required_paths = (
        "receipt_path",
        "score_path",
        "revised_study_lock_path",
        "locked_manifest_path",
    )
    if observed.get("schema") != CORRECTION_LOCKED_AUTHORIZATION_SCHEMA or any(
        not isinstance(observed.get(key), str) for key in required_paths
    ):
        raise CorrectionLockedAuthorizationError(
            "authorization lacks its bound correction input paths"
        )
    return validate_correction_locked_authorization(
        authorization_path=authorization,
        receipt_path=str(observed["receipt_path"]),
        score_path=str(observed["score_path"]),
        revised_lock_path=str(observed["revised_study_lock_path"]),
        locked_manifest_path=str(observed["locked_manifest_path"]),
    )


def claim_correction_locked_execution(
    *,
    authorization_path: str | Path,
    authorization: Mapping[str, object],
    output_directory: str | Path,
) -> None:
    """Record the irreversible one-shot claim immediately before solver work."""
    destination = Path(output_directory).resolve()
    expected = Path(str(authorization.get("locked_execution_directory", ""))).resolve()
    if destination != expected:
        raise CorrectionLockedAuthorizationError(
            "locked output differs from correction authorization"
        )
    claim = Path(authorization_path).resolve().with_name("correction-locked-execution-claim.json")
    payload = {
        "schema": "delphi.k3-correction-locked-execution-claim.v1",
        "authorization_sha256": sha256_file(Path(authorization_path)),
        "selected_tolerance": EXPECTED_SELECTED_TOLERANCE,
        "locked_execution_directory": str(destination),
    }
    try:
        descriptor = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as exc:
        raise CorrectionLockedAuthorizationError(
            "corrected locked evaluation was already claimed; rerunning is forbidden"
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(payload) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise


def materialize_frozen_selection_inputs(
    *,
    score_path: str | Path,
    receipt_path: str | Path,
    revised_lock_path: str | Path,
    spec_path: str | Path,
    spec_checksum_path: str | Path,
    output_directory: str | Path,
) -> dict[str, object]:
    """Translate sealed correction scores into the frozen selector schema."""
    from .convergence_study import create_development_selection

    score_file, receipt_file, lock_file, output = (
        Path(score_path).resolve(),
        Path(receipt_path).resolve(),
        Path(revised_lock_path).resolve(),
        Path(output_directory).resolve(),
    )
    if output.exists():
        raise CorrectionLockedAuthorizationError("compatibility selection output already exists")
    receipt = _read_json(receipt_file, "sealed reinterpretation receipt")
    score = _read_json(score_file, "sealed reinterpretation score")
    if score.get("receipt_sha256") != sha256_file(receipt_file):
        raise CorrectionLockedAuthorizationError("correction score is not bound to the receipt")
    executions = {
        float(item["tolerance"]): item
        for item in receipt.get("executions", [])
        if isinstance(item, Mapping)
    }
    score_rows = score.get("scores")
    if not isinstance(score_rows, list) or set(executions) != {0.01, 0.003, 0.001, 0.0003, 0.0001}:
        raise CorrectionLockedAuthorizationError(
            "correction score does not contain the full tolerance grid"
        )
    output.mkdir(parents=True, exist_ok=False)
    paths: list[Path] = []
    try:
        for row in score_rows:
            tolerance = float(row["convergence_tolerance"])
            metrics = {
                "runtime_warm": {
                    "empirical_acceptance_lower_5pct": row[
                        "runtime_empirical_acceptance_lower_5pct"
                    ],
                    "point_ratio": row["runtime_ratio_warm"],
                },
                "recovery": row["recovery"],
                "completion": row["completion"],
                "rms": {
                    "point_ratio": row["geometric_rms_ratio"],
                    "object_support": row["rms_object_support"],
                },
            }
            failures: list[str] = []
            if float(metrics["recovery"]["point_difference"]) < -0.05:
                failures.append("recovery_point_difference")
            if float(metrics["completion"]["point_difference"]) < -0.05:
                failures.append("completion_point_difference")
            if metrics["rms"]["object_support"] != 30 or metrics["rms"]["point_ratio"] is None:
                failures.append("rms_required_object_support")
            elif float(metrics["rms"]["point_ratio"]) > 1.01:
                failures.append("geometric_mean_selected_fit_rms_ratio")
            payload = {
                "schema": "delphi.k3-convergence-score.v1",
                "phase": "reference_axis_scoring_after_frozen_fit_selection",
                "role": "development",
                "study_spec_sha256": EXPECTED_STUDY_SPEC_SHA256,
                "study_lock_sha256": sha256_file(lock_file),
                "cohort_manifest_sha256": receipt["development_manifest_sha256"],
                "reference_catalog_sha256": score["reference_catalog_sha256"],
                "convergence_tolerance": tolerance,
                "execution_sha256": executions[tolerance]["execution_sha256"],
                "object_count": 30,
                "object_ids": [row["object_id"] for row in executions[tolerance]["rows"]],
                "metrics": metrics,
                "decision": {"eligible_for_selection": not failures, "failed_criteria": failures},
            }
            path = output / f"score-{format(tolerance, '.4g').replace('.', 'p')}.json"
            path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
            paths.append(path)
        selection_path = output / "development-selection.json"
        selection = create_development_selection(
            score_paths=paths,
            lock_path=lock_file,
            spec_path=spec_path,
            spec_checksum_path=spec_checksum_path,
            output_path=selection_path,
        )
        return {"score_paths": [str(path) for path in paths], "selection": selection}
    except Exception:
        for path in paths:
            path.unlink(missing_ok=True)
        (output / "development-selection.json").unlink(missing_ok=True)
        output.rmdir()
        raise


def correction_locked_runner_parameters(*, authorization_path: str | Path) -> dict[str, object]:
    """Return the only output and tolerance a correction-aware runner may use.

    This performs validation only.  It is intentionally separate from the
    call that runs the solver so a CLI ``--help`` or a test cannot begin the
    locked benchmark by accident.
    """
    authorization = validate_correction_locked_authorization_artifact(
        authorization_path=authorization_path
    )
    contract = authorization["runner_contract"]
    assert isinstance(contract, Mapping)
    return {
        "locked_execution_directory": authorization["locked_execution_directory"],
        "convergence_tolerance": contract["convergence_tolerance"],
        "authorization_sha256": authorization["authorization_sha256"],
    }


__all__ = [
    "CORRECTION_LOCKED_AUTHORIZATION_SCHEMA",
    "CorrectionLockedAuthorizationError",
    "EXPECTED_SELECTED_TOLERANCE",
    "create_correction_locked_authorization",
    "claim_correction_locked_execution",
    "correction_locked_runner_parameters",
    "validate_correction_locked_authorization",
    "validate_correction_locked_authorization_artifact",
    "materialize_frozen_selection_inputs",
]
