"""Frozen, phase-separated orchestration for the K3 convergence follow-up.

The execution phase is deliberately label blind.  It consumes the stripped
execution-input artifact, selects fits by final relative RMS, and atomically
freezes those choices.  A separate scoring phase may then open the reference
catalog and is cryptographically bound to the saved execution artifact.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
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
    CONVERGENCE_ITERATION_CAP,
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
        document.get("schema") != TIMING_SCHEMA
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
    development_selection_path: str | Path | None = None,
) -> dict[str, object]:
    """Fail-closed entry point while the audited execution ledger is finalized.

    This intentionally prevents the legacy label-aware runner from being used
    for either frozen cohort.  The complete study orchestrator replaces this
    guard in the next implementation slice; retaining an explicit refusal is
    safer than exposing a partially bound/resumable execution path.
    """
    if role == "locked_evaluation" and development_selection_path is None:
        raise DownstreamBenchmarkError(
            "locked evaluation requires a valid frozen development selection"
        )
    raise DownstreamBenchmarkError(
        "convergence study execution is disabled until the hash-bound blind ledger is complete"
    )


__all__ = [
    "create_study_lock",
    "execute_blind_cohort",
    "validate_study_lock",
]
