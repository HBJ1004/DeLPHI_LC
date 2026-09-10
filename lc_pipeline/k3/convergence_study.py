"""Frozen, phase-separated orchestration for the K3 convergence follow-up.

The execution phase is deliberately label blind.  It consumes the stripped
execution-input artifact, selects fits by final relative RMS, and atomically
freezes those choices.  A separate scoring phase may then open the reference
catalog and is cryptographically bound to the saved execution artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

import numpy as np
import yaml
from scipy.stats import beta

from ..physics.axial import axial_angular_error_deg
from ..v2.convexinv import _tree_sha256
from ..v2.data import canonical_json, sha256_file
from .convergence_benchmark import (
    _completion,
    _repeat_arm_order,
    _run_convergence_record,
    _selectable,
)
from .downstream import (
    DAMIT_STANDARD_STARTS_DEG,
    DownstreamBenchmarkError,
    DownstreamStart,
    _atomic_json,
    signed_starts_from_axes,
)

SPEC_SCHEMA = "delphi.k3-followup-study-spec.v1"
SUBSET_SCHEMA = "delphi.k3-convergence-object-subset.v1"
BLIND_INPUT_SCHEMA = "delphi.k3-convergence-blind-inputs.v1"
TIMING_SCHEMA = "delphi.k3-convergence-neural-timing.v1"
LOCK_SCHEMA = "delphi.k3-convergence-study-lock.v1"
EXECUTION_SCHEMA = "delphi.k3-convergence-blind-execution.v1"
SCORE_SCHEMA = "delphi.k3-convergence-score.v1"
SELECTION_SCHEMA = "delphi.k3-convergence-development-selection.v1"


def _read_json(path: str | Path, description: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DownstreamBenchmarkError(f"cannot read {description}: {exc}") from exc
    if not isinstance(value, dict):
        raise DownstreamBenchmarkError(f"{description} must be a JSON object")
    return value


def _read_spec(spec_path: str | Path, checksum_path: str | Path) -> tuple[dict[str, object], str]:
    path = Path(spec_path)
    digest = sha256_file(path)
    try:
        expected = Path(checksum_path).read_text(encoding="ascii").strip().split()[0]
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, IndexError, yaml.YAMLError) as exc:
        raise DownstreamBenchmarkError(f"cannot read frozen follow-up specification: {exc}") from exc
    if expected != digest or not isinstance(spec, dict) or spec.get("schema") != SPEC_SCHEMA:
        raise DownstreamBenchmarkError("follow-up specification does not match its frozen checksum")
    return spec, digest


def _study_section(spec: Mapping[str, object]) -> Mapping[str, object]:
    try:
        study = spec["convergence_acceleration"]
        shared = study["shared_conditions"]
        split = study["split"]
        analysis = study["analysis"]
        development = study["development_selection"]
        locked_gate = study["locked_evaluation_gate"]
    except (KeyError, TypeError) as exc:
        raise DownstreamBenchmarkError("follow-up specification is incomplete") from exc
    expected = {
        "iteration_cap": 1000,
        "timeout_seconds_per_start": 300,
        "timing_repeats": 3,
        "repeat_order_seed": 20260910,
        "bootstrap_resamples": 10000,
        "bootstrap_seed": 20260911,
    }
    if not isinstance(study, Mapping) or not isinstance(shared, Mapping):
        raise DownstreamBenchmarkError("convergence study settings are invalid")
    if any(shared.get(key) != value for key, value in expected.items()):
        raise DownstreamBenchmarkError("convergence study execution settings differ from the frozen protocol")
    if [float(value) for value in shared.get("convergence_tolerance_grid", [])] != [
        0.01,
        0.003,
        0.001,
        0.0003,
        0.0001,
    ]:
        raise DownstreamBenchmarkError("convergence tolerance grid differs from the frozen protocol")
    if (
        not isinstance(split, Mapping)
        or split.get("development_count") != 30
        or split.get("locked_evaluation_count") != 140
        or not isinstance(analysis, Mapping)
        or analysis.get("binary_repeat_aggregation")
        != "object_success_requires_at_least_two_of_three_repeats"
        or analysis.get("rms_required_object_support")
        != {"development": 30, "locked_evaluation": 140}
        or development.get("eligible_tolerance_requires")
        != {
            "guided_minus_baseline_recovery_point_difference_minimum": -0.05,
            "guided_minus_baseline_completion_point_difference_minimum": -0.05,
            "guided_to_baseline_geometric_mean_rms_ratio_maximum": 1.01,
        }
        or development.get("choose") != "largest_runtime_ratio_lower_bound"
        or development.get("tie_break") != "smaller_convergence_tolerance"
        or locked_gate
        != {
            "runtime_ratio_stratified_bootstrap_acceptance_lower_strictly_greater_than": 1.0,
            "recovery_simultaneous_exact_lower_minimum": -0.05,
            "completion_simultaneous_exact_lower_minimum": -0.05,
            "rms_ratio_stratified_bootstrap_acceptance_upper_strictly_less_than": 1.01,
            "decision": "all_four_conditions_must_pass",
        }
    ):
        raise DownstreamBenchmarkError("convergence analysis settings differ from the frozen protocol")
    return study


def _split_rows(split_path: str | Path) -> tuple[dict[str, object], list[Mapping[str, object]]]:
    document = _read_json(split_path, "full split")
    try:
        rows = sorted(document["folds"], key=lambda row: int(row["fold"]))
        object_ids = [str(value) for row in rows for value in row["test_ids"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise DownstreamBenchmarkError("full split is malformed") from exc
    if (
        [int(row["fold"]) for row in rows] != list(range(5))
        or len(object_ids) != 170
        or len(set(object_ids)) != 170
    ):
        raise DownstreamBenchmarkError("full split must contain 170 unique objects in five folds")
    return document, rows


def _expected_cohorts(
    split_path: str | Path, study: Mapping[str, object]
) -> dict[str, tuple[str, ...]]:
    _, rows = _split_rows(split_path)
    split_spec = study["split"]
    salt = str(split_spec["salt"])
    development: list[str] = []
    locked: list[str] = []
    for row in rows:
        ranked = sorted(
            (str(value) for value in row["test_ids"]),
            key=lambda value: hashlib.sha256(f"{salt}:{value}".encode("ascii")).hexdigest(),
        )
        development.extend(ranked[:6])
        locked.extend(ranked[6:])
    return {"development": tuple(development), "locked_evaluation": tuple(locked)}


def _validate_subset_manifest(
    path: str | Path,
    *,
    role: str,
    expected_ids: Sequence[str],
    spec_sha256: str,
    split_sha256: str,
    study: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    document = _read_json(path, f"{role} subset manifest")
    required = {
        "schema",
        "role",
        "study_spec_sha256",
        "source_full_split_sha256",
        "selection",
        "salt",
        "object_ids",
    }
    split_spec = study["split"]
    if (
        set(document) != required
        or document.get("schema") != SUBSET_SCHEMA
        or document.get("role") != role
        or document.get("study_spec_sha256") != spec_sha256
        or document.get("source_full_split_sha256") != split_sha256
        or document.get("selection") != split_spec["method"]
        or document.get("salt") != split_spec["salt"]
        or tuple(document.get("object_ids", ())) != tuple(expected_ids)
    ):
        raise DownstreamBenchmarkError(f"{role} manifest is not the exact frozen cohort")
    return document, sha256_file(path)


def _load_blind_inputs(
    path: str | Path,
    *,
    expected_hash: str,
    split_hash: str,
    ensemble_hash: str,
    catalog_hash: str,
) -> tuple[dict[str, object], dict[str, Mapping[str, object]]]:
    if sha256_file(path) != expected_hash:
        raise DownstreamBenchmarkError("blind execution inputs differ from the frozen artifact")
    document = _read_json(path, "blind execution inputs")
    if (
        document.get("schema") != BLIND_INPUT_SCHEMA
        or document.get("source_full_split_sha256") != split_hash
        or document.get("source_ensemble_sha256") != ensemble_hash
        or document.get("source_catalog_sha256") != catalog_hash
    ):
        raise DownstreamBenchmarkError("blind execution input provenance is invalid")
    rows = document.get("objects")
    if not isinstance(rows, list) or len(rows) != 170:
        raise DownstreamBenchmarkError("blind execution inputs must contain 170 objects")
    lookup: dict[str, Mapping[str, object]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise DownstreamBenchmarkError("blind execution object row is invalid")
        object_id = row.get("object_id")
        axes = np.asarray(row.get("guided_axes"), dtype=np.float64)
        period = row.get("period_hours")
        lightcurve = row.get("lightcurve")
        if (
            not isinstance(object_id, str)
            or object_id in lookup
            or not isinstance(row.get("fold"), int)
            or int(row["fold"]) not in range(5)
            or not isinstance(period, (int, float))
            or not math.isfinite(float(period))
            or float(period) <= 0
            or axes.shape != (3, 3)
            or not np.all(np.isfinite(axes))
            or np.any(np.linalg.norm(axes, axis=1) <= 1e-12)
            or not isinstance(lightcurve, Mapping)
            or not isinstance(lightcurve.get("source_path"), str)
            or not isinstance(lightcurve.get("source_sha256"), str)
        ):
            raise DownstreamBenchmarkError(f"blind execution row is invalid for {object_id!r}")
        lookup[object_id] = row
    return document, lookup


def _load_timing(
    path: str | Path,
    *,
    object_ids: Sequence[str],
    ensemble_sha256: str,
) -> tuple[dict[str, tuple[float, float]], str]:
    document = _read_json(path, "neural timing artifact")
    if (
        set(document)
        != {
            "schema",
            "source_ensemble_sha256",
            "object_ids",
            "warm_wall_seconds",
            "cold_wall_seconds",
        }
        or document.get("schema") != TIMING_SCHEMA
        or document.get("source_ensemble_sha256") != ensemble_sha256
        or tuple(document.get("object_ids", ())) != tuple(object_ids)
    ):
        raise DownstreamBenchmarkError("neural timing artifact is not aligned to the frozen ensemble")
    warm = np.asarray(document.get("warm_wall_seconds"), dtype=np.float64)
    cold = np.asarray(document.get("cold_wall_seconds"), dtype=np.float64)
    if (
        warm.shape != (len(object_ids),)
        or cold.shape != warm.shape
        or not np.all(np.isfinite(warm))
        or not np.all(np.isfinite(cold))
        or np.any(warm <= 0)
        or np.any(cold <= 0)
    ):
        raise DownstreamBenchmarkError("cold and warm neural timings must be positive aligned vectors")
    return {
        object_id: (float(warm[index]), float(cold[index]))
        for index, object_id in enumerate(object_ids)
    }, sha256_file(path)


def _compiler_version(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            list(command), capture_output=True, text=True, timeout=30, check=False
        )
        line = (completed.stdout or completed.stderr).splitlines()[0]
    except (OSError, IndexError, subprocess.TimeoutExpired) as exc:
        raise DownstreamBenchmarkError(f"cannot identify the frozen compiler: {exc}") from exc
    if completed.returncode != 0 or not line:
        raise DownstreamBenchmarkError("compiler-version command did not succeed")
    return line


def _compiler_executable(command: Sequence[str]) -> tuple[str, str]:
    if not command:
        raise DownstreamBenchmarkError("compiler-version command is empty")
    resolved = shutil.which(command[0])
    if resolved is None or not Path(resolved).is_file():
        raise DownstreamBenchmarkError("cannot resolve the frozen compiler executable")
    path = str(Path(resolved).resolve())
    return path, sha256_file(path)


def _archive_source_payload(archive_path: str | Path) -> dict[str, bytes]:
    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            payload: dict[str, bytes] = {}
            for member in archive.getmembers():
                path = PurePosixPath(member.name)
                parts = path.parts
                if not member.isfile() or "convexinv" not in parts:
                    continue
                offset = parts.index("convexinv") + 1
                relative = PurePosixPath(*parts[offset:]).as_posix()
                if not relative:
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    raise DownstreamBenchmarkError("solver archive contains an unreadable member")
                payload[relative] = handle.read()
    except (OSError, tarfile.TarError) as exc:
        raise DownstreamBenchmarkError(f"cannot inspect official solver archive: {exc}") from exc
    if not payload or "convexinv.c" not in payload or "constants.h" not in payload:
        raise DownstreamBenchmarkError("solver archive lacks the official convexinv source payload")
    return payload


def _verify_source_matches_archive(
    source_root: str | Path, archive_path: str | Path
) -> str:
    root = Path(source_root)
    payload = _archive_source_payload(archive_path)
    entries: list[dict[str, str]] = []
    for relative, expected in sorted(payload.items()):
        path = root / relative
        if not path.is_file() or path.read_bytes() != expected:
            raise DownstreamBenchmarkError(
                f"solver source tree differs from the official archive at {relative}"
            )
        entries.append(
            {"path": relative, "sha256": hashlib.sha256(expected).hexdigest()}
        )
    return hashlib.sha256(canonical_json(entries).encode("utf-8")).hexdigest()


def create_study_lock(
    *,
    spec_path: str | Path,
    spec_checksum_path: str | Path,
    split_path: str | Path,
    development_manifest_path: str | Path,
    locked_manifest_path: str | Path,
    blind_inputs_path: str | Path,
    ensemble_path: str | Path,
    neural_timing_path: str | Path,
    source_archive: str | Path,
    source_root: str | Path,
    executable: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Freeze every external identity before development execution begins."""
    output = Path(output_path)
    if output.exists():
        raise DownstreamBenchmarkError("refusing to overwrite the pre-execution study lock")
    payload, _, _, _, _ = _current_lock_payload(
        spec_path=spec_path,
        spec_checksum_path=spec_checksum_path,
        split_path=split_path,
        development_manifest_path=development_manifest_path,
        locked_manifest_path=locked_manifest_path,
        blind_inputs_path=blind_inputs_path,
        ensemble_path=ensemble_path,
        neural_timing_path=neural_timing_path,
        source_archive=source_archive,
        source_root=source_root,
        executable=executable,
    )
    _atomic_json(output, payload)
    return {**payload, "study_lock_sha256": sha256_file(output)}


def validate_study_lock(
    *,
    lock_path: str | Path,
    spec_path: str | Path,
    spec_checksum_path: str | Path,
    split_path: str | Path,
    development_manifest_path: str | Path,
    locked_manifest_path: str | Path,
    blind_inputs_path: str | Path,
    ensemble_path: str | Path,
    neural_timing_path: str | Path,
    source_archive: str | Path,
    source_root: str | Path,
    executable: str | Path,
) -> tuple[
    dict[str, object],
    Mapping[str, object],
    dict[str, Mapping[str, object]],
    dict[str, tuple[float, float]],
    dict[str, tuple[str, ...]],
]:
    """Recompute all frozen identities and reject any post-lock drift."""
    lock = _read_json(lock_path, "study lock")
    expected, study, blind_lookup, timing, cohorts = _current_lock_payload(
        spec_path=spec_path,
        spec_checksum_path=spec_checksum_path,
        split_path=split_path,
        development_manifest_path=development_manifest_path,
        locked_manifest_path=locked_manifest_path,
        blind_inputs_path=blind_inputs_path,
        ensemble_path=ensemble_path,
        neural_timing_path=neural_timing_path,
        source_archive=source_archive,
        source_root=source_root,
        executable=executable,
    )
    if lock != expected:
        raise DownstreamBenchmarkError("current resources do not match the pre-execution study lock")
    return lock, study, blind_lookup, timing, cohorts


def _current_lock_payload(
    *,
    spec_path: str | Path,
    spec_checksum_path: str | Path,
    split_path: str | Path,
    development_manifest_path: str | Path,
    locked_manifest_path: str | Path,
    blind_inputs_path: str | Path,
    ensemble_path: str | Path,
    neural_timing_path: str | Path,
    source_archive: str | Path,
    source_root: str | Path,
    executable: str | Path,
) -> tuple[
    dict[str, object],
    Mapping[str, object],
    dict[str, Mapping[str, object]],
    dict[str, tuple[float, float]],
    dict[str, tuple[str, ...]],
]:
    """Build the exact lock value without opening reference-axis values."""
    spec, spec_hash = _read_spec(spec_path, spec_checksum_path)
    study = _study_section(spec)
    split_hash = sha256_file(split_path)
    ensemble_hash = sha256_file(ensemble_path)
    archive_hash = sha256_file(source_archive)
    frozen = study["frozen_inputs"]
    if (
        split_hash != frozen["publication_split_sha256"]
        or ensemble_hash != frozen["oof_ensemble_sha256"]
        or archive_hash != frozen["official_solver_archive_sha256"]
    ):
        raise DownstreamBenchmarkError("a frozen study input has the wrong SHA-256")
    cohorts = _expected_cohorts(split_path, study)
    _, development_hash = _validate_subset_manifest(
        development_manifest_path,
        role="development",
        expected_ids=cohorts["development"],
        spec_sha256=spec_hash,
        split_sha256=split_hash,
        study=study,
    )
    _, locked_hash = _validate_subset_manifest(
        locked_manifest_path,
        role="locked_evaluation",
        expected_ids=cohorts["locked_evaluation"],
        spec_sha256=spec_hash,
        split_sha256=split_hash,
        study=study,
    )
    _, blind_lookup = _load_blind_inputs(
        blind_inputs_path,
        expected_hash=str(frozen["blind_execution_inputs_sha256"]),
        split_hash=split_hash,
        ensemble_hash=ensemble_hash,
        catalog_hash=str(frozen["reference_catalog_sha256"]),
    )
    _, split_rows = _split_rows(split_path)
    full_ids = tuple(str(value) for row in split_rows for value in row["test_ids"])
    timing, timing_hash = _load_timing(
        neural_timing_path, object_ids=full_ids, ensemble_sha256=ensemble_hash
    )
    if tuple(blind_lookup) != full_ids or set(timing) != set(blind_lookup):
        raise DownstreamBenchmarkError("locked execution inputs are not exactly aligned")
    binary = Path(executable)
    source = Path(source_root)
    if not binary.is_file() or not source.is_dir():
        raise DownstreamBenchmarkError("frozen solver executable and source root must exist")
    compiler_command = tuple(study["shared_conditions"]["compiler_version_command"])
    compiler_path, compiler_hash = _compiler_executable(compiler_command)
    payload: dict[str, object] = {
        "schema": LOCK_SCHEMA,
        "study_spec_sha256": spec_hash,
        "split_sha256": split_hash,
        "development_manifest_sha256": development_hash,
        "locked_manifest_sha256": locked_hash,
        "blind_inputs_sha256": sha256_file(blind_inputs_path),
        "reference_catalog_sha256": str(frozen["reference_catalog_sha256"]),
        "ensemble_sha256": ensemble_hash,
        "neural_timing_sha256": timing_hash,
        "source_archive_sha256": archive_hash,
        "source_payload_sha256": _verify_source_matches_archive(source, source_archive),
        "source_tree_sha256": _tree_sha256(source),
        "executable_sha256": sha256_file(binary),
        "compiler_command": list(compiler_command),
        "compiler_executable": compiler_path,
        "compiler_executable_sha256": compiler_hash,
        "compiler_version": _compiler_version(compiler_command),
        "settings": dict(study["shared_conditions"]),
        "cohort_sizes": {"development": 30, "locked_evaluation": 140},
    }
    return payload, study, blind_lookup, timing, cohorts


def _paired_binary_noninferiority(
    baseline: Sequence[int | bool],
    candidate: Sequence[int | bool],
    *,
    alpha_component: float = 0.025,
) -> dict[str, object]:
    """Return the prespecified conservative paired-binary lower bound.

    The difference is ``p01 - p10`` over all objects.  A one-sided lower
    Clopper--Pearson bound for favorable discordance and a one-sided upper
    bound for adverse discordance are combined by Bonferroni, giving joint
    coverage of at least 95% when ``alpha_component`` is 0.025.
    """
    base = np.asarray(baseline)
    guided = np.asarray(candidate)
    if (
        base.ndim != 1
        or base.shape != guided.shape
        or base.size < 1
        or not np.all(np.isin(base, (0, 1)))
        or not np.all(np.isin(guided, (0, 1)))
        or not 0.0 < alpha_component < 0.5
    ):
        raise DownstreamBenchmarkError("paired binary endpoints must be aligned zero/one vectors")
    n = int(base.size)
    favorable = int(np.sum((base == 0) & (guided == 1)))
    adverse = int(np.sum((base == 1) & (guided == 0)))
    favorable_lower = (
        0.0
        if favorable == 0
        else float(beta.ppf(alpha_component, favorable, n - favorable + 1))
    )
    adverse_upper = (
        1.0
        if adverse == n
        else float(beta.ppf(1.0 - alpha_component, adverse + 1, n - adverse))
    )
    return {
        "object_count": n,
        "concordant_failure_count": int(np.sum((base == 0) & (guided == 0))),
        "favorable_discordance_count": favorable,
        "adverse_discordance_count": adverse,
        "concordant_success_count": int(np.sum((base == 1) & (guided == 1))),
        "point_difference": float((favorable - adverse) / n),
        "favorable_rate_cp_lower": favorable_lower,
        "adverse_rate_cp_upper": adverse_upper,
        "simultaneous_exact_lower": favorable_lower - adverse_upper,
        "component_one_sided_alpha": alpha_component,
        "coverage_at_least": 1.0 - 2.0 * alpha_component,
        # Short aliases are kept in the machine-readable artifact so reviewers
        # can map the four-cell table directly onto p01 and p10 notation.
        "favorable_discordances": favorable,
        "adverse_discordances": adverse,
        "tail_probability": alpha_component,
    }


def _stratified_indices(
    folds: np.ndarray, *, resamples: int, seed: int
) -> np.ndarray:
    if (
        folds.ndim != 1
        or folds.size < 2
        or isinstance(resamples, bool)
        or not isinstance(resamples, int)
        or resamples < 1
    ):
        raise DownstreamBenchmarkError("stratified bootstrap inputs are invalid")
    unique = np.unique(folds)
    if unique.size < 1:
        raise DownstreamBenchmarkError("stratified bootstrap requires at least one fold")
    rng = np.random.default_rng(seed)
    sampled_blocks = []
    for fold in sorted(unique.tolist()):
        members = np.flatnonzero(folds == fold)
        if members.size == 0:  # pragma: no cover - protected by np.unique
            raise DownstreamBenchmarkError("stratified bootstrap contains an empty fold")
        sampled_blocks.append(rng.choice(members, size=(resamples, members.size), replace=True))
    return np.concatenate(sampled_blocks, axis=1)


def _stratified_ratio_bootstrap(
    baseline: Sequence[Sequence[float]] | Sequence[float],
    candidate: Sequence[Sequence[float]] | Sequence[float],
    folds: Sequence[int],
    *,
    resamples: int = 10000,
    seed: int = 20260911,
) -> dict[str, object]:
    """Empirical 5th/95th bounds for a total-time ratio, stratified by fold."""
    base = np.asarray(baseline, dtype=np.float64)
    guided = np.asarray(candidate, dtype=np.float64)
    strata = np.asarray(folds)
    if (
        base.ndim not in (1, 2)
        or base.shape != guided.shape
        or base.shape[0] != strata.size
        or np.any(~np.isfinite(base))
        or np.any(~np.isfinite(guided))
        or np.any(base <= 0)
        or np.any(guided <= 0)
    ):
        raise DownstreamBenchmarkError("runtime arrays must be positive and aligned by object")
    sampled = _stratified_indices(strata, resamples=resamples, seed=seed)
    if base.ndim == 1:
        base_totals = np.sum(base[sampled], axis=1)
        guided_totals = np.sum(guided[sampled], axis=1)
    else:
        base_totals = np.sum(base[sampled], axis=(1, 2))
        guided_totals = np.sum(guided[sampled], axis=(1, 2))
    ratios = base_totals / guided_totals
    lower = float(np.quantile(ratios, 0.05))
    upper = float(np.quantile(ratios, 0.95))
    return {
        "point_ratio": float(np.sum(base) / np.sum(guided)),
        "empirical_acceptance_lower_5pct": lower,
        "empirical_acceptance_upper_95pct": upper,
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "acceptance_lower": lower,
        "acceptance_upper": upper,
        "resamples": resamples,
        "seed": seed,
        "stratum_sizes": {
            str(value): int(np.sum(strata == value)) for value in sorted(np.unique(strata).tolist())
        },
        "resampling": "object_clustered_stratified_by_original_outer_fold",
        "interval_label": "prespecified_empirical_acceptance_bounds_not_confidence_intervals",
    }


def _paired_rms_ratio(
    baseline: Sequence[Sequence[float]],
    candidate: Sequence[Sequence[float]],
    joint_selectable: Sequence[Sequence[bool]],
    *,
    required_support: int,
    folds: Sequence[int] | None = None,
    resamples: int = 10000,
    seed: int = 20260911,
) -> dict[str, object]:
    """Geometric mean selected-fit RMS ratio on label-blind joint support."""
    base = np.asarray(baseline, dtype=np.float64)
    guided = np.asarray(candidate, dtype=np.float64)
    joint = np.asarray(joint_selectable, dtype=bool)
    if (
        base.ndim != 2
        or base.shape != guided.shape
        or base.shape != joint.shape
        or base.shape[1] != 3
        or isinstance(required_support, bool)
        or not isinstance(required_support, int)
        or required_support < 1
    ):
        raise DownstreamBenchmarkError("RMS inputs must be aligned object-by-three-repeat arrays")
    if (
        np.any(~np.isfinite(base[joint]))
        or np.any(~np.isfinite(guided[joint]))
        or np.any(base[joint] <= 0)
        or np.any(guided[joint] <= 0)
    ):
        raise DownstreamBenchmarkError("jointly selectable RMS values must be finite and positive")
    supported = np.sum(joint, axis=1) >= 2
    support = int(np.sum(supported))
    if support != required_support:
        raise DownstreamBenchmarkError(
            f"RMS support must be exactly {required_support} objects; observed {support}"
        )
    object_log_ratios = np.asarray(
        [
            np.mean(np.log(guided[index, joint[index]] / base[index, joint[index]]))
            for index in np.flatnonzero(supported)
        ],
        dtype=np.float64,
    )
    strata = np.zeros(support, dtype=int) if folds is None else np.asarray(folds)[supported]
    if strata.shape != (support,):
        raise DownstreamBenchmarkError("RMS fold labels must align with objects")
    sampled = _stratified_indices(strata, resamples=resamples, seed=seed)
    samples = np.exp(np.mean(object_log_ratios[sampled], axis=1))
    return {
        "estimand": "geometric_mean_selected_fit_rms_ratio",
        "support_conditioning": "jointly_completed_and_selectable_without_reference_outcome",
        "minimum_joint_repeats_per_object": 2,
        "object_support": support,
        "required_object_support": required_support,
        "point_ratio": float(np.exp(np.mean(object_log_ratios))),
        "empirical_acceptance_lower_5pct": float(np.quantile(samples, 0.05)),
        "empirical_acceptance_upper_95pct": float(np.quantile(samples, 0.95)),
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "interval_label": "prespecified_empirical_acceptance_bounds_not_confidence_intervals",
    }


def execute_blind_cohort(
    *,
    role: str,
    lock_path: str | Path,
    spec_path: str | Path,
    spec_checksum_path: str | Path,
    split_path: str | Path,
    development_manifest_path: str | Path,
    locked_manifest_path: str | Path,
    blind_inputs_path: str | Path,
    ensemble_path: str | Path,
    neural_timing_path: str | Path,
    source_archive: str | Path,
    source_root: str | Path,
    executable: str | Path,
    dump_root: str | Path,
    output_directory: str | Path,
    convergence_tolerance: float,
    development_selection_path: str | Path | None = None,
    development_score_paths: Sequence[str | Path] = (),
) -> dict[str, object]:
    """Execute one exact cohort without loading any reference-axis values."""
    if role not in {"development", "locked_evaluation"}:
        raise DownstreamBenchmarkError("study role must be development or locked_evaluation")
    if role == "locked_evaluation" and development_selection_path is None:
        raise DownstreamBenchmarkError(
            "locked evaluation requires a valid frozen development selection"
        )
    lock, study, blind_lookup, neural_timing, cohorts = validate_study_lock(
        lock_path=lock_path,
        spec_path=spec_path,
        spec_checksum_path=spec_checksum_path,
        split_path=split_path,
        development_manifest_path=development_manifest_path,
        locked_manifest_path=locked_manifest_path,
        blind_inputs_path=blind_inputs_path,
        ensemble_path=ensemble_path,
        neural_timing_path=neural_timing_path,
        source_archive=source_archive,
        source_root=source_root,
        executable=executable,
    )
    tolerance = float(convergence_tolerance)
    grid = tuple(float(value) for value in study["shared_conditions"]["convergence_tolerance_grid"])
    if not math.isfinite(tolerance) or tolerance not in grid:
        raise DownstreamBenchmarkError("convergence tolerance is not in the frozen development grid")
    selection: Mapping[str, object] | None = None
    if role == "locked_evaluation":
        selection = validate_development_selection(
            selection_path=development_selection_path,
            score_paths=development_score_paths,
            lock_path=lock_path,
            spec_path=spec_path,
            spec_checksum_path=spec_checksum_path,
        )
        selected = selection.get("selected_tolerance")
        if selection.get("status") != "selected" or selected is None:
            raise DownstreamBenchmarkError(
                "development selection found no eligible tolerance; locked evaluation is forbidden"
            )
        if tolerance != float(selected):
            raise DownstreamBenchmarkError(
                "locked evaluation tolerance differs from the frozen development selection"
            )

    destination = Path(output_directory).resolve()
    artifact_path = destination / "blind-execution.json"
    object_ids = cohorts[role]
    expected_count = 30 if role == "development" else 140
    if len(object_ids) != expected_count:
        raise DownstreamBenchmarkError("execution cohort has the wrong frozen size")
    manifest_hash = str(
        lock[
            "development_manifest_sha256"
            if role == "development"
            else "locked_manifest_sha256"
        ]
    )
    conditions = study["shared_conditions"]
    lock_hash = sha256_file(lock_path)
    contract: dict[str, object] = {
        "schema": "delphi.k3-convergence-execution-contract.v1",
        "study_lock_sha256": lock_hash,
        "study_spec_sha256": lock["study_spec_sha256"],
        "cohort_manifest_sha256": manifest_hash,
        "blind_inputs_sha256": lock["blind_inputs_sha256"],
        "neural_timing_sha256": lock["neural_timing_sha256"],
        "source_archive_sha256": lock["source_archive_sha256"],
        "source_tree_sha256": lock["source_tree_sha256"],
        "source_payload_sha256": lock["source_payload_sha256"],
        "executable_sha256": lock["executable_sha256"],
        "compiler_command": lock["compiler_command"],
        "compiler_executable": lock["compiler_executable"],
        "compiler_executable_sha256": lock["compiler_executable_sha256"],
        "compiler_version": lock["compiler_version"],
        "role": role,
        "convergence_tolerance": tolerance,
        "iteration_cap": conditions["iteration_cap"],
        "timeout_seconds_per_start": conditions["timeout_seconds_per_start"],
        "timing_repeats": conditions["timing_repeats"],
        "repeat_order_seed": conditions["repeat_order_seed"],
        "starts_per_arm": 6,
        "arm_names": ["baseline", "candidate"],
        "candidate_semantics": "guided_k3_initialization",
        "period_policy": "fixed_first_frozen_catalog_solution",
        "fit_selection": "minimum_final_relative_rms_without_reference_axis_access",
    }
    contract_hash = hashlib.sha256(canonical_json(contract).encode("utf-8")).hexdigest()
    if artifact_path.exists():
        completed = _validated_execution(
            artifact_path,
            lock=lock,
            lock_path=lock_path,
            study=study,
            role=role,
            object_ids=object_ids,
            manifest_sha256=manifest_hash,
        )
        if completed.get("execution_contract") != contract:
            raise DownstreamBenchmarkError(
                "completed blind execution does not match every current execution condition"
            )
        return {
            **completed,
            "artifact_path": str(artifact_path),
            "artifact_sha256": sha256_file(artifact_path),
            "resumed_completed_artifact": True,
        }
    rows: list[dict[str, object]] = []
    root = Path(dump_root).resolve()
    binary = Path(executable).resolve()
    source = Path(source_root).resolve()
    compiler_command = tuple(str(value) for value in lock["compiler_command"])
    timeout = float(conditions["timeout_seconds_per_start"])
    repeat_count = int(conditions["timing_repeats"])
    order_seed = int(conditions["repeat_order_seed"])
    resolved_lightcurves = {
        object_id: _resolve_lightcurve(root, blind_lookup[object_id], object_id)
        for object_id in object_ids
    }
    if role == "locked_evaluation":
        assert selection is not None and development_selection_path is not None
        _claim_locked_execution(
            selection=selection,
            selection_path=development_selection_path,
            output_directory=destination,
        )
    for object_id in object_ids:
        blind = blind_lookup[object_id]
        lightcurve = resolved_lightcurves[object_id]
        period = float(blind["period_hours"])
        starts = {
            "baseline": tuple(
                DownstreamStart(longitude, latitude, period)
                for longitude, latitude in DAMIT_STANDARD_STARTS_DEG
            ),
            "candidate": signed_starts_from_axes(blind["guided_axes"], period_hours=period),
        }
        warm_time, cold_time = neural_timing[object_id]
        object_contract = {
            **contract,
            "object_id": object_id,
            "fold": int(blind["fold"]),
            "period_hours": period,
            "lightcurve_sha256": sha256_file(lightcurve),
        }
        repeats = [
            _execute_complete_repeat(
                executable=binary,
                source_root=source,
                lightcurve=lightcurve,
                output_root=destination,
                object_id=object_id,
                fold=int(blind["fold"]),
                repeat_index=repeat_index,
                starts=starts,
                convergence_tolerance=tolerance,
                timeout_seconds=timeout,
                compiler_command=compiler_command,
                execution_contract=object_contract,
                arm_order=_repeat_arm_order(object_id, repeat_index, order_seed),
                warm_neural_seconds=warm_time,
                cold_neural_seconds=cold_time,
            )
            for repeat_index in range(repeat_count)
        ]
        rows.append(
            {
                "object_id": object_id,
                "fold": int(blind["fold"]),
                "period_hours": period,
                "lightcurve_sha256": sha256_file(lightcurve),
                "repeat_markers": [
                    {
                        "repeat_index": index,
                        "path": str(
                            (
                                Path("runs")
                                / object_id
                                / f"repeat-{index}"
                                / "repeat-complete.json"
                            ).as_posix()
                        ),
                        "sha256": sha256_file(
                            destination
                            / "runs"
                            / object_id
                            / f"repeat-{index}"
                            / "repeat-complete.json"
                        ),
                    }
                    for index in range(repeat_count)
                ],
                "repeats": repeats,
            }
        )
    payload: dict[str, object] = {
        "schema": EXECUTION_SCHEMA,
        "phase": "label_blind_execution_and_minimum_rms_fit_selection",
        "reference_catalog_opened": False,
        "study_lock_sha256": lock_hash,
        "execution_contract": contract,
        "execution_contract_sha256": contract_hash,
        "role": role,
        "cohort_manifest_sha256": manifest_hash,
        "convergence_tolerance": tolerance,
        "object_ids": list(object_ids),
        "object_count": len(object_ids),
        "rows": rows,
    }
    artifact_hash = _atomic_json(artifact_path, payload)
    return {
        **payload,
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_hash,
    }


def _resolve_lightcurve(
    dump_root: Path, blind: Mapping[str, object], object_id: str
) -> Path:
    lightcurve = blind["lightcurve"]
    assert isinstance(lightcurve, Mapping)
    relative = Path(str(lightcurve["source_path"]))
    path = (dump_root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(dump_root):
        raise DownstreamBenchmarkError(f"lightcurve path escapes dump root for {object_id}")
    if not path.is_file() or sha256_file(path) != lightcurve["source_sha256"]:
        raise DownstreamBenchmarkError(f"lightcurve does not match blind input for {object_id}")
    return path


def _selection_from_records(records: Sequence[Mapping[str, object]]) -> dict[str, object] | None:
    eligible = [record for record in records if _selectable(record)]
    if not eligible:
        return None
    best = min(
        eligible,
        key=lambda record: (
            float(record["result"]["relative_rms_from_output"]),
            int(record["identity"]["start_index"]),
        ),
    )
    result = best["result"]
    identity = best["identity"]
    return {
        "criterion": "minimum_final_relative_rms",
        "start_index": int(identity["start_index"]),
        "final_relative_rms": float(result["relative_rms_from_output"]),
        "final_lambda_deg": float(result["final_lambda_deg"]),
        "final_beta_deg": float(result["final_beta_deg"]),
    }


def _arm_repeat_summary(
    records: Sequence[Mapping[str, object]],
    *,
    arm: str,
    warm_neural_seconds: float,
    cold_neural_seconds: float,
) -> dict[str, object]:
    if len(records) != 6:
        raise DownstreamBenchmarkError("every arm repeat must record exactly six starts")
    wall = np.asarray(
        [record.get("result", {}).get("wall_time_seconds") for record in records],
        dtype=np.float64,
    )
    if wall.shape != (6,) or np.any(~np.isfinite(wall)) or np.any(wall < 0):
        raise DownstreamBenchmarkError("every solver start must retain finite nonnegative wall time")
    inversion_wall = float(np.sum(wall))
    neural_warm = warm_neural_seconds if arm == "candidate" else 0.0
    neural_cold = cold_neural_seconds if arm == "candidate" else 0.0
    states = [str(record["completion"]) for record in records]
    selection = _selection_from_records(records)
    return {
        "starts_requested": 6,
        "all_cells_recorded": True,
        "completion_counts": {state: states.count(state) for state in sorted(set(states))},
        "completed": all(state == "converged" for state in states),
        "selectable": selection is not None,
        "selected_fit": selection,
        "inversion_wall_seconds": inversion_wall,
        "warm_neural_wall_seconds": neural_warm,
        "cold_neural_wall_seconds": neural_cold,
        "wall_seconds_warm": inversion_wall + neural_warm,
        "wall_seconds_cold": inversion_wall + neural_cold,
    }


def _execute_complete_repeat(
    *,
    executable: Path,
    source_root: Path,
    lightcurve: Path,
    output_root: Path,
    object_id: str,
    fold: int,
    repeat_index: int,
    starts: Mapping[str, Sequence[DownstreamStart]],
    convergence_tolerance: float,
    timeout_seconds: float,
    compiler_command: Sequence[str],
    execution_contract: Mapping[str, object],
    arm_order: Sequence[str],
    warm_neural_seconds: float,
    cold_neural_seconds: float,
) -> dict[str, object]:
    repeat_directory = output_root / "runs" / object_id / f"repeat-{repeat_index}"
    marker_path = repeat_directory / "repeat-complete.json"
    plan = {
        "object_id": object_id,
        "fold": fold,
        "repeat_index": repeat_index,
        "arm_order": list(arm_order),
        "convergence_tolerance": convergence_tolerance,
        "timeout_seconds_per_start": timeout_seconds,
        "execution_contract_sha256": hashlib.sha256(
            canonical_json(dict(execution_contract)).encode("utf-8")
        ).hexdigest(),
        "warm_neural_seconds": warm_neural_seconds,
        "cold_neural_seconds": cold_neural_seconds,
    }
    if marker_path.exists():
        marker = _read_json(marker_path, "completed repeat marker")
        if marker.get("schema") != "delphi.k3-convergence-repeat-complete.v1" or marker.get(
            "plan"
        ) != plan:
            raise DownstreamBenchmarkError("completed repeat does not match the execution plan")
        cells = marker.get("cells")
        if not isinstance(cells, list) or len(cells) != 12:
            raise DownstreamBenchmarkError("completed repeat must contain all twelve execution cells")
        for cell in cells:
            if not isinstance(cell, Mapping) or not isinstance(cell.get("path"), str):
                raise DownstreamBenchmarkError("completed repeat contains an invalid cell")
            path = output_root / str(cell["path"])
            if not path.is_file() or cell.get("sha256") != sha256_file(path):
                raise DownstreamBenchmarkError("completed repeat cell hash does not match")
        return marker
    if repeat_directory.exists() and any(repeat_directory.iterdir()):
        raise DownstreamBenchmarkError(
            "partial repeat timing is non-reusable; preserve it for audit and restart in a new study root"
        )

    records: dict[str, list[dict[str, object]]] = {"baseline": [], "candidate": []}
    cells: list[dict[str, object]] = []
    for start_index in range(6):
        for arm in arm_order:
            record = _run_convergence_record(
                executable=executable,
                source_root=source_root,
                lightcurve=lightcurve,
                output_root=output_root,
                object_id=object_id,
                arm=arm,
                repeat_index=repeat_index,
                start_index=start_index,
                start=starts[arm][start_index],
                convergence_tolerance=convergence_tolerance,
                timeout_seconds=timeout_seconds,
                compiler_command=compiler_command,
                execution_contract=execution_contract,
                arm_order=arm_order,
            )
            records[arm].append(record)
            record_path = (
                Path("runs")
                / object_id
                / f"repeat-{repeat_index}"
                / arm
                / f"start-{start_index}"
                / "result.json"
            )
            cells.append(
                {
                    "sequence_index": len(cells),
                    "arm": arm,
                    "start_index": start_index,
                    "completion": record["completion"],
                    "path": record_path.as_posix(),
                    "sha256": sha256_file(output_root / record_path),
                }
            )
    payload: dict[str, object] = {
        "schema": "delphi.k3-convergence-repeat-complete.v1",
        "plan": plan,
        "cells": cells,
        "baseline": _arm_repeat_summary(
            records["baseline"],
            arm="baseline",
            warm_neural_seconds=warm_neural_seconds,
            cold_neural_seconds=cold_neural_seconds,
        ),
        "candidate": _arm_repeat_summary(
            records["candidate"],
            arm="candidate",
            warm_neural_seconds=warm_neural_seconds,
            cold_neural_seconds=cold_neural_seconds,
        ),
    }
    _atomic_json(marker_path, payload)
    return payload


def execute_development_grid(
    *,
    lock_path: str | Path,
    spec_path: str | Path,
    spec_checksum_path: str | Path,
    split_path: str | Path,
    development_manifest_path: str | Path,
    locked_manifest_path: str | Path,
    blind_inputs_path: str | Path,
    ensemble_path: str | Path,
    neural_timing_path: str | Path,
    source_archive: str | Path,
    source_root: str | Path,
    executable: str | Path,
    dump_root: str | Path,
    output_root: str | Path,
) -> dict[str, object]:
    """Execute every frozen development tolerance before any selection."""
    spec, _ = _read_spec(spec_path, spec_checksum_path)
    study = _study_section(spec)
    executions = []
    for tolerance in study["shared_conditions"]["convergence_tolerance_grid"]:
        label = format(float(tolerance), ".4g").replace(".", "p")
        result = execute_blind_cohort(
            role="development",
            lock_path=lock_path,
            spec_path=spec_path,
            spec_checksum_path=spec_checksum_path,
            split_path=split_path,
            development_manifest_path=development_manifest_path,
            locked_manifest_path=locked_manifest_path,
            blind_inputs_path=blind_inputs_path,
            ensemble_path=ensemble_path,
            neural_timing_path=neural_timing_path,
            source_archive=source_archive,
            source_root=source_root,
            executable=executable,
            dump_root=dump_root,
            output_directory=Path(output_root) / f"tolerance-{label}",
            convergence_tolerance=float(tolerance),
        )
        executions.append(
            {
                "tolerance": float(tolerance),
                "execution_path": result["artifact_path"],
                "execution_sha256": result["artifact_sha256"],
            }
        )
    return {"role": "development", "executions": executions}


def _validated_execution(
    execution_path: str | Path,
    *,
    lock: Mapping[str, object],
    lock_path: str | Path,
    study: Mapping[str, object],
    role: str,
    object_ids: Sequence[str],
    manifest_sha256: str,
) -> dict[str, object]:
    document = _read_json(execution_path, "blind execution artifact")
    required = {
        "schema",
        "phase",
        "reference_catalog_opened",
        "study_lock_sha256",
        "execution_contract",
        "execution_contract_sha256",
        "role",
        "cohort_manifest_sha256",
        "convergence_tolerance",
        "object_ids",
        "object_count",
        "rows",
    }
    contract = document.get("execution_contract")
    tolerance = document.get("convergence_tolerance")
    if (
        set(document) != required
        or document.get("schema") != EXECUTION_SCHEMA
        or document.get("phase") != "label_blind_execution_and_minimum_rms_fit_selection"
        or document.get("reference_catalog_opened") is not False
        or document.get("study_lock_sha256") != sha256_file(lock_path)
        or document.get("role") != role
        or document.get("cohort_manifest_sha256") != manifest_sha256
        or tuple(document.get("object_ids", ())) != tuple(object_ids)
        or document.get("object_count") != len(object_ids)
        or not isinstance(tolerance, (int, float))
        or float(tolerance)
        not in tuple(
            float(value) for value in study["shared_conditions"]["convergence_tolerance_grid"]
        )
        or not isinstance(contract, Mapping)
        or hashlib.sha256(canonical_json(dict(contract)).encode("utf-8")).hexdigest()
        != document.get("execution_contract_sha256")
    ):
        raise DownstreamBenchmarkError("blind execution artifact violates the frozen contract")
    contract_checks = {
        "study_spec_sha256": lock["study_spec_sha256"],
        "cohort_manifest_sha256": manifest_sha256,
        "blind_inputs_sha256": lock["blind_inputs_sha256"],
        "neural_timing_sha256": lock["neural_timing_sha256"],
        "source_archive_sha256": lock["source_archive_sha256"],
        "source_tree_sha256": lock["source_tree_sha256"],
        "source_payload_sha256": lock["source_payload_sha256"],
        "executable_sha256": lock["executable_sha256"],
        "compiler_command": lock["compiler_command"],
        "compiler_executable": lock["compiler_executable"],
        "compiler_executable_sha256": lock["compiler_executable_sha256"],
        "compiler_version": lock["compiler_version"],
        "role": role,
        "convergence_tolerance": float(tolerance),
        "iteration_cap": 1000,
        "timeout_seconds_per_start": 300,
        "timing_repeats": 3,
        "starts_per_arm": 6,
    }
    if any(contract.get(key) != value for key, value in contract_checks.items()):
        raise DownstreamBenchmarkError("blind execution contract is not bound to the study lock")
    rows = document.get("rows")
    if (
        not isinstance(rows, list)
        or not all(isinstance(row, Mapping) for row in rows)
        or [row.get("object_id") for row in rows] != list(object_ids)
    ):
        raise DownstreamBenchmarkError("blind execution rows do not match the frozen cohort")
    root = Path(execution_path).resolve().parent
    for row in rows:
        _validate_execution_row(
            row,
            root=root,
            tolerance=float(tolerance),
            repeat_count=3,
        )
    return document


def _validate_execution_row(
    row: Mapping[str, object], *, root: Path, tolerance: float, repeat_count: int
) -> None:
    object_id = row.get("object_id")
    repeats = row.get("repeats")
    markers = row.get("repeat_markers")
    if (
        not isinstance(object_id, str)
        or not isinstance(row.get("fold"), int)
        or int(row["fold"]) not in range(5)
        or not isinstance(row.get("period_hours"), (int, float))
        or not math.isfinite(float(row["period_hours"]))
        or float(row["period_hours"]) <= 0
        or not isinstance(repeats, list)
        or len(repeats) != repeat_count
        or not isinstance(markers, list)
        or len(markers) != repeat_count
    ):
        raise DownstreamBenchmarkError("blind execution object row is incomplete")
    for repeat_index, (repeat, marker_ref) in enumerate(zip(repeats, markers, strict=True)):
        if not isinstance(repeat, Mapping) or not isinstance(marker_ref, Mapping):
            raise DownstreamBenchmarkError("blind execution repeat is invalid")
        relative_text = marker_ref.get("path")
        if not isinstance(relative_text, str):
            raise DownstreamBenchmarkError("blind execution repeat marker path is invalid")
        relative = Path(relative_text)
        marker_path = (root / relative).resolve()
        if relative.is_absolute() or not marker_path.is_relative_to(root):
            raise DownstreamBenchmarkError("blind execution repeat marker escapes its artifact root")
        if (
            marker_ref.get("repeat_index") != repeat_index
            or not marker_path.is_file()
            or marker_ref.get("sha256") != sha256_file(marker_path)
            or _read_json(marker_path, "repeat marker") != dict(repeat)
            or repeat.get("schema") != "delphi.k3-convergence-repeat-complete.v1"
        ):
            raise DownstreamBenchmarkError("blind execution repeat marker is not hash-bound")
        plan = repeat.get("plan")
        cells = repeat.get("cells")
        if (
            not isinstance(plan, Mapping)
            or plan.get("object_id") != object_id
            or plan.get("fold") != row["fold"]
            or plan.get("repeat_index") != repeat_index
            or plan.get("convergence_tolerance") != tolerance
            or plan.get("timeout_seconds_per_start") != 300.0
            or not all(
                isinstance(plan.get(key), (int, float))
                and math.isfinite(float(plan[key]))
                and float(plan[key]) > 0
                for key in ("warm_neural_seconds", "cold_neural_seconds")
            )
            or not isinstance(cells, list)
            or len(cells) != 12
        ):
            raise DownstreamBenchmarkError("blind execution repeat plan is invalid")
        expected_order = [
            (arm, start_index)
            for start_index in range(6)
            for arm in plan.get("arm_order", ())
        ]
        observed_order = [(cell.get("arm"), cell.get("start_index")) for cell in cells]
        if observed_order != expected_order or set(plan.get("arm_order", ())) != {
            "baseline",
            "candidate",
        }:
            raise DownstreamBenchmarkError("blind execution does not contain the paired AB/BA cells")
        arm_records: dict[str, list[Mapping[str, object]]] = {
            "baseline": [],
            "candidate": [],
        }
        for sequence_index, cell in enumerate(cells):
            path_text = cell.get("path")
            if not isinstance(path_text, str):
                raise DownstreamBenchmarkError("blind execution cell path is invalid")
            cell_path = (root / path_text).resolve()
            arm, start_index = observed_order[sequence_index]
            expected_path = (
                Path("runs")
                / object_id
                / f"repeat-{repeat_index}"
                / str(arm)
                / f"start-{start_index}"
                / "result.json"
            ).as_posix()
            if (
                cell.get("sequence_index") != sequence_index
                or path_text != expected_path
                or Path(path_text).is_absolute()
                or not cell_path.is_relative_to(root)
                or not cell_path.is_file()
                or cell.get("sha256") != sha256_file(cell_path)
            ):
                raise DownstreamBenchmarkError("blind execution cell is missing or hash-mismatched")
            record = _read_json(cell_path, "solver execution cell")
            identity = record.get("identity")
            result = record.get("result")
            record_contract = record.get("execution_contract")
            if (
                record.get("schema") != "delphi.k3-convergence-convexinv-run.v1"
                or not isinstance(identity, Mapping)
                or identity.get("object_id") != object_id
                or identity.get("arm") != arm
                or identity.get("repeat_index") != repeat_index
                or identity.get("start_index") != start_index
                or identity.get("convergence_tolerance") != tolerance
                or identity.get("timeout_seconds") != 300.0
                or identity.get("repeat_arm_order") != list(plan["arm_order"])
                or not isinstance(record_contract, Mapping)
                or hashlib.sha256(
                    canonical_json(dict(record_contract)).encode("utf-8")
                ).hexdigest()
                != plan.get("execution_contract_sha256")
                or not isinstance(result, Mapping)
                or record.get("completion")
                != _completion(result, expected_period_hours=float(row["period_hours"]))
                or cell.get("completion") != record.get("completion")
            ):
                raise DownstreamBenchmarkError("solver execution cell content is inconsistent")
            arm_records[str(arm)].append(record)
        for arm in ("baseline", "candidate"):
            summary = repeat.get(arm)
            if (
                not isinstance(summary, Mapping)
                or summary.get("starts_requested") != 6
                or summary.get("all_cells_recorded") is not True
                or not isinstance(summary.get("completed"), bool)
                or not isinstance(summary.get("selectable"), bool)
            ):
                raise DownstreamBenchmarkError("blind execution arm summary is invalid")
            for key in (
                "inversion_wall_seconds",
                "wall_seconds_warm",
                "wall_seconds_cold",
            ):
                value = summary.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
                    raise DownstreamBenchmarkError("blind execution timing is invalid")
            selection = summary.get("selected_fit")
            if (selection is None) != (summary.get("selectable") is False):
                raise DownstreamBenchmarkError("blind execution selectable flag is inconsistent")
            if selection is not None:
                if not isinstance(selection, Mapping) or not all(
                    isinstance(selection.get(key), (int, float))
                    and math.isfinite(float(selection[key]))
                    for key in (
                        "start_index",
                        "final_relative_rms",
                        "final_lambda_deg",
                        "final_beta_deg",
                    )
                ):
                    raise DownstreamBenchmarkError("blind execution selected fit is invalid")
                if (
                    selection.get("criterion") != "minimum_final_relative_rms"
                    or int(selection["start_index"]) not in range(6)
                    or float(selection["final_relative_rms"]) <= 0
                ):
                    raise DownstreamBenchmarkError("blind execution selected fit is invalid")
            expected_summary = _arm_repeat_summary(
                arm_records[arm],
                arm=arm,
                warm_neural_seconds=float(plan["warm_neural_seconds"]),
                cold_neural_seconds=float(plan["cold_neural_seconds"]),
            )
            if dict(summary) != expected_summary:
                raise DownstreamBenchmarkError(
                    "blind execution arm summary does not match its six solver cells"
                )


def _load_reference_targets(
    catalog_path: str | Path, *, expected_hash: str, object_ids: Sequence[str]
) -> dict[str, np.ndarray]:
    if sha256_file(catalog_path) != expected_hash:
        raise DownstreamBenchmarkError("reference catalog does not match the frozen study hash")
    targets: dict[str, np.ndarray] = {}
    try:
        lines = Path(catalog_path).read_text(encoding="utf-8").splitlines()
        for line in lines:
            if not line.strip():
                continue
            row = json.loads(line)
            object_id = str(row["object_id"])
            if object_id in targets:
                raise DownstreamBenchmarkError("reference catalog contains duplicate objects")
            values = np.asarray(
                [solution["vector"] for solution in row["solutions"]], dtype=np.float64
            )
            if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != 3:
                raise DownstreamBenchmarkError("reference catalog has malformed pole vectors")
            norms = np.linalg.norm(values, axis=1)
            if np.any(~np.isfinite(values)) or np.any(norms <= 1e-12):
                raise DownstreamBenchmarkError("reference catalog has invalid pole vectors")
            targets[object_id] = values / norms[:, None]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, DownstreamBenchmarkError):
            raise
        raise DownstreamBenchmarkError(f"cannot parse frozen reference catalog: {exc}") from exc
    if any(object_id not in targets for object_id in object_ids):
        raise DownstreamBenchmarkError("reference catalog lacks a frozen cohort object")
    return targets


def _selected_pole_error(
    selection: Mapping[str, object] | None, targets: np.ndarray
) -> float | None:
    if selection is None:
        return None
    longitude = math.radians(float(selection["final_lambda_deg"]))
    latitude = math.radians(float(selection["final_beta_deg"]))
    pole = np.asarray(
        [
            math.cos(latitude) * math.cos(longitude),
            math.cos(latitude) * math.sin(longitude),
            math.sin(latitude),
        ],
        dtype=np.float64,
    )
    return float(np.min(axial_angular_error_deg(pole[None, :], targets)))


def score_blind_execution(
    *,
    execution_path: str | Path,
    reference_catalog_path: str | Path,
    lock_path: str | Path,
    spec_path: str | Path,
    spec_checksum_path: str | Path,
    split_path: str | Path,
    cohort_manifest_path: str | Path,
    output_path: str | Path,
    expected_role: str,
    development_selection_path: str | Path | None = None,
    development_score_paths: Sequence[str | Path] = (),
) -> dict[str, object]:
    """Open references only after fit selection is frozen and hash-bound."""
    output = Path(output_path)
    if output.exists():
        raise DownstreamBenchmarkError("refusing to overwrite a convergence score artifact")
    if expected_role not in {"development", "locked_evaluation"}:
        raise DownstreamBenchmarkError("score role must be development or locked_evaluation")
    if expected_role == "locked_evaluation" and development_selection_path is None:
        raise DownstreamBenchmarkError("locked scoring requires the frozen development selection")
    spec, spec_hash = _read_spec(spec_path, spec_checksum_path)
    study = _study_section(spec)
    lock = _read_json(lock_path, "study lock")
    if lock.get("schema") != LOCK_SCHEMA or lock.get("study_spec_sha256") != spec_hash:
        raise DownstreamBenchmarkError("study lock is not bound to the frozen specification")
    if sha256_file(split_path) != lock.get("split_sha256"):
        raise DownstreamBenchmarkError("scoring split differs from the study lock")
    cohorts = _expected_cohorts(split_path, study)
    object_ids = cohorts[expected_role]
    _, manifest_hash = _validate_subset_manifest(
        cohort_manifest_path,
        role=expected_role,
        expected_ids=object_ids,
        spec_sha256=spec_hash,
        split_sha256=str(lock["split_sha256"]),
        study=study,
    )
    expected_manifest_hash = lock[
        "development_manifest_sha256"
        if expected_role == "development"
        else "locked_manifest_sha256"
    ]
    if manifest_hash != expected_manifest_hash:
        raise DownstreamBenchmarkError("scoring cohort manifest differs from the study lock")
    if expected_role == "locked_evaluation":
        selection = validate_development_selection(
            selection_path=development_selection_path,
            score_paths=development_score_paths,
            lock_path=lock_path,
            spec_path=spec_path,
            spec_checksum_path=spec_checksum_path,
        )
        if selection.get("status") != "selected":
            raise DownstreamBenchmarkError("locked scoring is forbidden without an eligible selection")
    execution = _validated_execution(
        execution_path,
        lock=lock,
        lock_path=lock_path,
        study=study,
        role=expected_role,
        object_ids=object_ids,
        manifest_sha256=manifest_hash,
    )
    if expected_role == "locked_evaluation" and float(
        execution["convergence_tolerance"]
    ) != float(selection["selected_tolerance"]):
        raise DownstreamBenchmarkError("locked execution does not use the selected tolerance")

    # This is the first point in the workflow at which reference values are parsed.
    targets = _load_reference_targets(
        reference_catalog_path,
        expected_hash=str(lock["reference_catalog_sha256"]),
        object_ids=object_ids,
    )
    count = len(object_ids)
    shape = (count, 3)
    base_warm = np.empty(shape)
    guided_warm = np.empty(shape)
    base_cold = np.empty(shape)
    guided_cold = np.empty(shape)
    base_completion = np.zeros(shape, dtype=np.int8)
    guided_completion = np.zeros(shape, dtype=np.int8)
    base_recovery = np.zeros(shape, dtype=np.int8)
    guided_recovery = np.zeros(shape, dtype=np.int8)
    base_rms = np.full(shape, np.nan)
    guided_rms = np.full(shape, np.nan)
    joint_selectable = np.zeros(shape, dtype=bool)
    folds = np.empty(count, dtype=np.int8)
    audit_rows: list[dict[str, object]] = []
    for object_index, row in enumerate(execution["rows"]):
        object_id = str(row["object_id"])
        folds[object_index] = int(row["fold"])
        repeat_rows = []
        for repeat_index, repeat in enumerate(row["repeats"]):
            scored: dict[str, object] = {"repeat_index": repeat_index}
            for arm, prefix in (("baseline", "base"), ("candidate", "guided")):
                summary = repeat[arm]
                selection_value = summary["selected_fit"]
                error = _selected_pole_error(selection_value, targets[object_id])
                recovered = error is not None and error <= 20.0
                if prefix == "base":
                    base_warm[object_index, repeat_index] = summary["wall_seconds_warm"]
                    base_cold[object_index, repeat_index] = summary["wall_seconds_cold"]
                    base_completion[object_index, repeat_index] = int(summary["completed"])
                    base_recovery[object_index, repeat_index] = int(recovered)
                    if selection_value is not None:
                        base_rms[object_index, repeat_index] = selection_value[
                            "final_relative_rms"
                        ]
                else:
                    guided_warm[object_index, repeat_index] = summary["wall_seconds_warm"]
                    guided_cold[object_index, repeat_index] = summary["wall_seconds_cold"]
                    guided_completion[object_index, repeat_index] = int(summary["completed"])
                    guided_recovery[object_index, repeat_index] = int(recovered)
                    if selection_value is not None:
                        guided_rms[object_index, repeat_index] = selection_value[
                            "final_relative_rms"
                        ]
                scored[arm] = {
                    "completed": bool(summary["completed"]),
                    "selectable": bool(summary["selectable"]),
                    "selected_pole_error_deg": error,
                    "recovered_within_20_deg": recovered,
                }
            joint_selectable[object_index, repeat_index] = bool(
                repeat["baseline"]["completed"]
                and repeat["candidate"]["completed"]
                and repeat["baseline"]["selectable"]
                and repeat["candidate"]["selectable"]
            )
            scored["jointly_completed_and_selectable"] = bool(
                joint_selectable[object_index, repeat_index]
            )
            repeat_rows.append(scored)
        audit_rows.append(
            {
                "object_id": object_id,
                "fold": int(folds[object_index]),
                "repeats": repeat_rows,
            }
        )
    base_object_completion = (np.sum(base_completion, axis=1) >= 2).astype(np.int8)
    guided_object_completion = (np.sum(guided_completion, axis=1) >= 2).astype(np.int8)
    base_object_recovery = (np.sum(base_recovery, axis=1) >= 2).astype(np.int8)
    guided_object_recovery = (np.sum(guided_recovery, axis=1) >= 2).astype(np.int8)
    resamples = int(study["shared_conditions"]["bootstrap_resamples"])
    seed = int(study["shared_conditions"]["bootstrap_seed"])
    rms_support = int(np.sum(np.sum(joint_selectable, axis=1) >= 2))
    if rms_support == count:
        rms_metric = _paired_rms_ratio(
            base_rms,
            guided_rms,
            joint_selectable,
            required_support=count,
            folds=folds,
            resamples=resamples,
            seed=seed,
        )
        rms_metric["estimable"] = True
    else:
        rms_metric = {
            "estimand": "geometric_mean_selected_fit_rms_ratio",
            "support_conditioning": (
                "jointly_completed_and_selectable_without_reference_outcome"
            ),
            "minimum_joint_repeats_per_object": 2,
            "object_support": rms_support,
            "required_object_support": count,
            "point_ratio": None,
            "empirical_acceptance_lower_5pct": None,
            "empirical_acceptance_upper_95pct": None,
            "bootstrap_resamples": resamples,
            "bootstrap_seed": seed,
            "estimable": False,
            "gate_failure": "insufficient_predeclared_object_support",
        }
    metrics = {
        "runtime_warm": _stratified_ratio_bootstrap(
            base_warm, guided_warm, folds, resamples=resamples, seed=seed
        ),
        "runtime_cold_sensitivity": _stratified_ratio_bootstrap(
            base_cold, guided_cold, folds, resamples=resamples, seed=seed
        ),
        "recovery": _paired_binary_noninferiority(
            base_object_recovery, guided_object_recovery
        ),
        "completion": _paired_binary_noninferiority(
            base_object_completion, guided_object_completion
        ),
        "rms": rms_metric,
    }
    decision = _score_decision(expected_role, metrics, study)
    payload: dict[str, object] = {
        "schema": SCORE_SCHEMA,
        "phase": "reference_axis_scoring_after_frozen_fit_selection",
        "role": expected_role,
        "study_lock_sha256": sha256_file(lock_path),
        "study_spec_sha256": spec_hash,
        "cohort_manifest_sha256": manifest_hash,
        "execution_sha256": sha256_file(execution_path),
        "reference_catalog_sha256": sha256_file(reference_catalog_path),
        "convergence_tolerance": float(execution["convergence_tolerance"]),
        "object_count": count,
        "object_ids": list(object_ids),
        "repeat_aggregation": "object_success_requires_at_least_two_of_three_repeats",
        "all_execution_cells_retained": True,
        "metrics": metrics,
        "decision": decision,
        "audit_rows": audit_rows,
    }
    digest = _atomic_json(output, payload)
    return {**payload, "artifact_path": str(output), "artifact_sha256": digest}


def _score_decision(
    role: str, metrics: Mapping[str, Mapping[str, object]], study: Mapping[str, object]
) -> dict[str, object]:
    if role == "development":
        rules = study["development_selection"]["eligible_tolerance_requires"]
        failures = []
        if metrics["recovery"]["point_difference"] < rules[
            "guided_minus_baseline_recovery_point_difference_minimum"
        ]:
            failures.append("recovery_point_difference")
        if metrics["completion"]["point_difference"] < rules[
            "guided_minus_baseline_completion_point_difference_minimum"
        ]:
            failures.append("completion_point_difference")
        rms_point = metrics["rms"].get("point_ratio")
        if not isinstance(rms_point, (int, float)) or not math.isfinite(float(rms_point)):
            failures.append("rms_required_object_support")
        elif rms_point > rules["guided_to_baseline_geometric_mean_rms_ratio_maximum"]:
            failures.append("geometric_mean_selected_fit_rms_ratio")
        return {"eligible_for_selection": not failures, "failed_criteria": failures}
    rules = study["locked_evaluation_gate"]
    failures = []
    if not metrics["runtime_warm"]["empirical_acceptance_lower_5pct"] > rules[
        "runtime_ratio_stratified_bootstrap_acceptance_lower_strictly_greater_than"
    ]:
        failures.append("runtime_warm_empirical_acceptance_lower")
    if metrics["recovery"]["simultaneous_exact_lower"] < rules[
        "recovery_simultaneous_exact_lower_minimum"
    ]:
        failures.append("recovery_simultaneous_exact_lower")
    if metrics["completion"]["simultaneous_exact_lower"] < rules[
        "completion_simultaneous_exact_lower_minimum"
    ]:
        failures.append("completion_simultaneous_exact_lower")
    rms_upper = metrics["rms"].get("empirical_acceptance_upper_95pct")
    if (
        not isinstance(rms_upper, (int, float))
        or not math.isfinite(float(rms_upper))
        or not rms_upper
        < rules["rms_ratio_stratified_bootstrap_acceptance_upper_strictly_less_than"]
    ):
        failures.append("rms_empirical_acceptance_upper")
    return {
        "passed_all_four_locked_conditions": not failures,
        "failed_conditions": failures,
        "cold_runtime_is_sensitivity_only": True,
    }


def _development_selection_payload(
    *,
    score_paths: Sequence[str | Path],
    lock_path: str | Path,
    spec_path: str | Path,
    spec_checksum_path: str | Path,
    selection_path: str | Path,
) -> dict[str, object]:
    spec, spec_hash = _read_spec(spec_path, spec_checksum_path)
    study = _study_section(spec)
    lock = _read_json(lock_path, "study lock")
    if lock.get("schema") != LOCK_SCHEMA or lock.get("study_spec_sha256") != spec_hash:
        raise DownstreamBenchmarkError("development scores are not bound to the frozen study lock")
    grid = tuple(float(value) for value in study["shared_conditions"]["convergence_tolerance_grid"])
    if len(score_paths) != len(grid):
        raise DownstreamBenchmarkError("development selection requires exactly five score artifacts")
    by_tolerance: dict[float, tuple[dict[str, object], str]] = {}
    for path in score_paths:
        score = _read_json(path, "development score")
        tolerance = score.get("convergence_tolerance")
        metrics = score.get("metrics")
        if (
            score.get("schema") != SCORE_SCHEMA
            or score.get("role") != "development"
            or score.get("study_lock_sha256") != sha256_file(lock_path)
            or score.get("study_spec_sha256") != spec_hash
            or score.get("reference_catalog_sha256") != lock.get("reference_catalog_sha256")
            or score.get("object_count") != 30
            or not isinstance(tolerance, (int, float))
            or float(tolerance) not in grid
            or float(tolerance) in by_tolerance
            or not isinstance(metrics, Mapping)
        ):
            raise DownstreamBenchmarkError("development score violates the frozen study contract")
        for key in ("runtime_warm", "recovery", "completion", "rms"):
            if not isinstance(metrics.get(key), Mapping):
                raise DownstreamBenchmarkError("development score metrics are incomplete")
        support = metrics["rms"].get("object_support")
        if not isinstance(support, int) or not 0 <= support <= 30:
            raise DownstreamBenchmarkError("development RMS support is invalid")
        recomputed = _score_decision("development", metrics, study)
        if score.get("decision") != recomputed:
            raise DownstreamBenchmarkError("development eligibility is inconsistent with its metrics")
        runtime_lower = metrics["runtime_warm"].get("empirical_acceptance_lower_5pct")
        if not isinstance(runtime_lower, (int, float)) or not math.isfinite(
            float(runtime_lower)
        ):
            raise DownstreamBenchmarkError("development runtime bound is invalid")
        by_tolerance[float(tolerance)] = (score, sha256_file(path))
    if set(by_tolerance) != set(grid):
        raise DownstreamBenchmarkError("development scores do not cover the exact tolerance grid")
    score_entries = []
    eligible: list[tuple[float, float, str]] = []
    for tolerance in grid:
        score, digest = by_tolerance[tolerance]
        is_eligible = bool(score["decision"]["eligible_for_selection"])
        runtime_lower = float(
            score["metrics"]["runtime_warm"]["empirical_acceptance_lower_5pct"]
        )
        score_entries.append(
            {
                "convergence_tolerance": tolerance,
                "score_sha256": digest,
                "execution_sha256": score["execution_sha256"],
                "eligible": is_eligible,
                "runtime_empirical_acceptance_lower_5pct": runtime_lower,
            }
        )
        if is_eligible:
            eligible.append((runtime_lower, -tolerance, digest))
    if eligible:
        _, negative_tolerance, selected_score_hash = max(eligible)
        selected_tolerance: float | None = -negative_tolerance
        status = "selected"
    else:
        selected_score_hash = None
        selected_tolerance = None
        status = "no_eligible_tolerance_stop"
    selection = Path(selection_path).resolve()
    return {
        "schema": SELECTION_SCHEMA,
        "status": status,
        "study_lock_sha256": sha256_file(lock_path),
        "study_spec_sha256": spec_hash,
        "selection_rule": {
            "eligibility": dict(
                study["development_selection"]["eligible_tolerance_requires"]
            ),
            "choose": "largest_runtime_ratio_lower_bound",
            "tie_break": "smaller_convergence_tolerance",
        },
        "score_artifacts": score_entries,
        "selected_tolerance": selected_tolerance,
        "selected_score_sha256": selected_score_hash,
        "locked_execution_directory": str(selection.parent / "locked-evaluation"),
        "one_shot_claim_path": str(selection.parent / "locked-execution-claim.json"),
    }


def create_development_selection(
    *,
    score_paths: Sequence[str | Path],
    lock_path: str | Path,
    spec_path: str | Path,
    spec_checksum_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Apply the frozen lexicographic rule to exactly five development scores."""
    output = Path(output_path)
    if output.exists():
        raise DownstreamBenchmarkError("refusing to overwrite the development selection")
    payload = _development_selection_payload(
        score_paths=score_paths,
        lock_path=lock_path,
        spec_path=spec_path,
        spec_checksum_path=spec_checksum_path,
        selection_path=output,
    )
    digest = _atomic_json(output, payload)
    return {**payload, "artifact_path": str(output), "artifact_sha256": digest}


def validate_development_selection(
    *,
    selection_path: str | Path,
    score_paths: Sequence[str | Path],
    lock_path: str | Path,
    spec_path: str | Path,
    spec_checksum_path: str | Path,
) -> dict[str, object]:
    """Recompute selection from all frozen development score artifacts."""
    observed = _read_json(selection_path, "development selection")
    expected = _development_selection_payload(
        score_paths=score_paths,
        lock_path=lock_path,
        spec_path=spec_path,
        spec_checksum_path=spec_checksum_path,
        selection_path=selection_path,
    )
    if observed != expected:
        raise DownstreamBenchmarkError("development selection is not the exact frozen rule result")
    return observed


def _claim_locked_execution(
    *, selection: Mapping[str, object], selection_path: str | Path, output_directory: Path
) -> None:
    expected_output = Path(str(selection["locked_execution_directory"]))
    claim_path = Path(str(selection["one_shot_claim_path"]))
    if output_directory != expected_output:
        raise DownstreamBenchmarkError(
            "locked evaluation must use the canonical directory bound into development selection"
        )
    payload = {
        "schema": "delphi.k3-convergence-locked-execution-claim.v1",
        "development_selection_sha256": sha256_file(selection_path),
        "selected_tolerance": selection["selected_tolerance"],
        "locked_execution_directory": str(output_directory),
    }
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as exc:
        raise DownstreamBenchmarkError(
            "locked evaluation was already claimed; rerunning into any directory is forbidden"
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # The exclusive claim remains as evidence even when serialization or
        # later execution fails.  A failed locked attempt is not reusable.
        raise


__all__ = [
    "create_study_lock",
    "create_development_selection",
    "execute_blind_cohort",
    "execute_development_grid",
    "score_blind_execution",
    "validate_development_selection",
    "validate_study_lock",
]
