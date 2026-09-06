#!/usr/bin/env python3
"""Validate DeLPHI V2 release manifests without third-party dependencies.

The JSON schemas are the structural contract.  This module implements the
small JSON-Schema subset used by those files, then adds scientific and
cross-manifest integrity checks that JSON Schema cannot express.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
DEFAULT_RELEASE_SPEC = ROOT / "release_spec.yaml"
SCHEMA_FILES = {
    "data": ROOT / "schemas" / "data-manifest.schema.json",
    "run": ROOT / "schemas" / "run-manifest.schema.json",
    "artifact": ROOT / "schemas" / "artifact-manifest.schema.json",
    "verdict": ROOT / "schemas" / "verdict-manifest.schema.json",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
OPERATORS = {"eq", "lt", "lte", "gt", "gte"}
LOCKED_GATE_CONTRACTS: dict[str, tuple[str, str, str, Any]] = {
    "MODEL_DERANGED_INPUT_POINT_GAP": ("model", "correct_vs_deranged_input_top1_gap_deg", "gte", 5.0),
    "MODEL_DERANGED_INPUT_CI95_LOWER": ("model", "correct_vs_deranged_input_top1_gap_ci95_lower_deg", "gte", 2.0),
    "MODEL_DERANGED_INPUT_ONE_SIDED_HOLM_P": ("model", "correct_vs_deranged_input_one_sided_holm_adjusted_p_value", "lte", 0.01),
    "MODEL_MATCHED_K_PRIOR_TRAIN_FOLD_ONLY": ("model", "matched_k_strongest_prior_selected_on_training_fold_only", "eq", True),
    "MODEL_MATCHED_K_PRIOR_POINT_GAP": ("model", "matched_k_improvement_over_strongest_train_fold_prior_deg", "gte", 5.0),
    "MODEL_MATCHED_K_PRIOR_CI95_LOWER": ("model", "matched_k_improvement_over_strongest_train_fold_prior_ci95_lower_deg", "gte", 2.0),
    "MODEL_MATCHED_K_PRIOR_ONE_SIDED_HOLM_P": ("model", "matched_k_improvement_over_strongest_train_fold_prior_one_sided_holm_adjusted_p_value", "lte", 0.01),
    "MODEL_LABEL_SHUFFLE_CI95_LOWER": ("model", "label_shuffle_degradation_ci95_lower_deg", "gte", 2.0),
    "MODEL_LABEL_SHUFFLE_ONE_SIDED_HOLM_P": ("model", "label_shuffle_degradation_one_sided_holm_adjusted_p_value", "lte", 0.01),
    "MODEL_SEEDS_POSITIVE_COUNT": ("model", "correct_vs_deranged_positive_seed_count_of_five", "gte", 4),
    "MODEL_SEEDS_MINIMUM_GAP": ("model", "minimum_correct_vs_deranged_seed_gap_deg", "gte", -1.0),
    "MODEL_ALL_SIGNAL_ABLATION_CI95_LOWER": ("model", "all_signal_ablation_degradation_ci95_lower_deg", "gte", 2.0),
    "MODEL_CLAIMED_CHANNEL_COUNT": ("model", "claimed_useful_channel_count", "gte", 1),
    "MODEL_EACH_CLAIMED_CHANNEL_CI95_LOWER": ("model", "minimum_claimed_channel_ablation_degradation_ci95_lower_deg", "gt", 0.5),
    "MODEL_EACH_CLAIMED_CHANNEL_ONE_SIDED_HOLM_P": ("model", "maximum_claimed_channel_ablation_one_sided_holm_adjusted_p_value", "lte", 0.01),
    "EVAL_ONE_OBJECT_ONE_VOTE": ("statistics", "one_object_one_vote_enforced", "eq", True),
    "EVAL_OOF_ROW_COMPLETENESS": ("statistics", "every_expected_oof_object_seed_row_present_exactly_once_with_no_extras", "eq", True),
    "EVAL_POOLED_OBJECT_AGGREGATION": ("statistics", "fold_and_seed_predictions_pooled_to_one_record_per_object_before_summary", "eq", True),
    "EVAL_NO_TEST_SELECTED_CHECKPOINT": ("integrity", "checkpoint_selected_without_test_labels_or_test_metrics", "eq", True),
    "REPORT_TOP1_HEADLINE": ("integrity", "directed_top1_is_headline_pole_metric", "eq", True),
    "REPORT_MATCHED_K_BASELINE_ONLY": ("integrity", "matched_k_is_secondary_oracle_baseline_not_headline", "eq", True),
    "PERIOD_COVERAGE_68_LOWER": ("scientific", "period_empirical_coverage_68", "gte", 0.63),
    "PERIOD_COVERAGE_68_UPPER": ("scientific", "period_empirical_coverage_68", "lte", 0.73),
    "PERIOD_COVERAGE_68_NOMINAL_CONTAINED": ("scientific", "period_nominal_68_inside_object_bootstrap_ci", "eq", True),
    "PERIOD_COVERAGE_95_LOWER": ("scientific", "period_empirical_coverage_95", "gte", 0.92),
    "PERIOD_COVERAGE_95_UPPER": ("scientific", "period_empirical_coverage_95", "lte", 0.98),
    "PERIOD_COVERAGE_95_NOMINAL_CONTAINED": ("scientific", "period_nominal_95_inside_object_bootstrap_ci", "eq", True),
    "PERIOD_CALIBRATION_ECE": ("scientific", "period_expected_calibration_error", "lte", 0.05),
    "PERIOD_CONSENSUS_RETENTION_RULE": (
        "scientific",
        "consensus_absent_or_retained_only_if_acc5_gain_gte5pp_or_median_relative_error_reduction_gte10pct_and_one_sided_holm_p_lte0_01",
        "eq",
        True,
    ),
    "CHECKPOINT_SCHEMA_MISMATCH_REJECTED": ("reproducibility", "checkpoint_schema_mismatch_rejection_test_passed", "eq", True),
    "NUMERIC_CPU_GPU_MAX_ANGULAR_DISAGREEMENT": ("reproducibility", "cpu_gpu_max_per_object_directed_angular_disagreement_deg", "lte", 0.05),
    "NUMERIC_CPU_GPU_AGGREGATE_DISAGREEMENT": ("reproducibility", "cpu_gpu_absolute_aggregate_directed_metric_disagreement_deg", "lte", 0.01),
    "TEST_TRACKING_GUARD": ("testing", "repository_test_tracking_guard_passed", "eq", True),
}
FORBIDDEN_GATE_IDS = {"NUMERIC_CPU_GPU_PARITY"}


@dataclass(frozen=True)
class Issue:
    """One validation failure with a JSON-path-like location."""

    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


class ContractError(ValueError):
    """Raised when a manifest or release specification violates its contract."""

    def __init__(self, issues: Sequence[Issue]):
        self.issues = tuple(issues)
        super().__init__("\n".join(str(issue) for issue in self.issues))


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {token}")


def load_json(path: Path | str) -> Any:
    """Load strict JSON, rejecting duplicate keys and NaN/Infinity."""

    manifest_path = Path(path)
    try:
        with manifest_path.open("r", encoding="utf-8") as stream:
            return json.load(
                stream,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite,
            )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContractError((Issue(str(manifest_path), f"cannot load strict JSON: {exc}"),)) from exc


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_pointer(root: Mapping[str, Any], pointer: str) -> Any:
    if not pointer.startswith("#/"):
        raise ValueError(f"only local JSON pointers are supported, got {pointer!r}")
    value: Any = root
    for raw_part in pointer[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        value = value[part]
    return value


def _is_type(instance: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(instance, dict)
    if expected == "array":
        return isinstance(instance, list)
    if expected == "string":
        return isinstance(instance, str)
    if expected == "boolean":
        return isinstance(instance, bool)
    if expected == "null":
        return instance is None
    if expected == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if expected == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool) and math.isfinite(instance)
    return False


def _json_equal(left: Any, right: Any) -> bool:
    """JSON equality without Python's surprising ``True == 1`` coercion."""

    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(left) and math.isfinite(right) and left == right
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_json_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_json_equal(left[key], right[key]) for key in left)
    return type(left) is type(right) and left == right


def _valid_datetime(value: str) -> bool:
    if not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _valid_uri(value: str) -> bool:
    parsed = urlparse(value)
    return bool(parsed.scheme and (parsed.netloc or parsed.scheme == "doi"))


def _schema_issues(
    instance: Any,
    schema: Mapping[str, Any],
    root_schema: Mapping[str, Any],
    path: str = "$",
) -> list[Issue]:
    issues: list[Issue] = []
    if "$ref" in schema:
        try:
            target = _json_pointer(root_schema, schema["$ref"])
        except (KeyError, TypeError, ValueError) as exc:
            return [Issue(path, f"invalid schema reference {schema['$ref']!r}: {exc}")]
        return _schema_issues(instance, target, root_schema, path)

    expected_types = schema.get("type")
    if expected_types is not None:
        if isinstance(expected_types, str):
            expected_types = [expected_types]
        if not any(_is_type(instance, expected) for expected in expected_types):
            return [Issue(path, f"expected type {expected_types}, got {type(instance).__name__}")]

    if "const" in schema and not _json_equal(instance, schema["const"]):
        issues.append(Issue(path, f"must equal {schema['const']!r}"))
    if "enum" in schema and not any(_json_equal(instance, choice) for choice in schema["enum"]):
        issues.append(Issue(path, f"must be one of {schema['enum']!r}"))

    if isinstance(instance, str):
        if len(instance) < schema.get("minLength", 0):
            issues.append(Issue(path, f"must contain at least {schema['minLength']} characters"))
        pattern = schema.get("pattern")
        if pattern is not None and re.search(pattern, instance) is None:
            issues.append(Issue(path, f"does not match required pattern {pattern!r}"))
        value_format = schema.get("format")
        if value_format == "date-time" and not _valid_datetime(instance):
            issues.append(Issue(path, "must be an RFC 3339 UTC date-time ending in Z"))
        elif value_format == "uri" and not _valid_uri(instance):
            issues.append(Issue(path, "must be an absolute URI"))

    if _is_type(instance, "number") or _is_type(instance, "integer"):
        if "minimum" in schema and instance < schema["minimum"]:
            issues.append(Issue(path, f"must be >= {schema['minimum']}"))

    if isinstance(instance, list):
        if len(instance) < schema.get("minItems", 0):
            issues.append(Issue(path, f"must contain at least {schema['minItems']} items"))
        if schema.get("uniqueItems"):
            serialized = [json.dumps(item, sort_keys=True, separators=(",", ":")) for item in instance]
            if len(serialized) != len(set(serialized)):
                issues.append(Issue(path, "items must be unique"))
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(instance):
                issues.extend(_schema_issues(item, item_schema, root_schema, f"{path}[{index}]"))

    if isinstance(instance, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in instance:
                issues.append(Issue(path, f"missing required field {key!r}"))
        properties = schema.get("properties", {})
        for key, value in instance.items():
            child_path = f"{path}.{key}"
            if key in properties:
                issues.extend(_schema_issues(value, properties[key], root_schema, child_path))
            elif schema.get("additionalProperties") is False:
                issues.append(Issue(child_path, "additional fields are forbidden"))
    return issues


def _walk(value: Any, path: str = "$") -> Iterable[tuple[str, str | None, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            yield child_path, key, child
            yield from _walk(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]"
            yield child_path, None, child
            yield from _walk(child, child_path)


def _looks_like_local_absolute_path(value: str) -> bool:
    return (
        value.startswith(("/", "~/", "file://", "\\\\"))
        or WINDOWS_ABSOLUTE_RE.match(value) is not None
    )


def _is_remote_https_uri(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def _semantic_common(document: Mapping[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    for path, key, value in _walk(document):
        if isinstance(value, str) and _looks_like_local_absolute_path(value):
            issues.append(Issue(path, "absolute local paths are forbidden; use a logical relative path or HTTPS URI"))
        if key is not None and (key == "sha256" or key.endswith("_sha256")):
            if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
                issues.append(Issue(path, "must be a lowercase 64-character SHA-256 digest"))
            elif value == "0" * 64:
                issues.append(Issue(path, "all-zero placeholder digest is forbidden"))
        if key in {"commit", "git_commit"}:
            if not isinstance(value, str) or GIT_COMMIT_RE.fullmatch(value) is None:
                issues.append(Issue(path, "must be a full lowercase 40-character Git commit"))
        if key in {"dirty", "git_dirty"} and value is not False:
            issues.append(Issue(path, "scientific runs must originate from a clean Git commit"))
    return issues


def _artifact_entries(document: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    entries: list[tuple[str, Mapping[str, Any]]] = []
    for path, _, value in _walk(document):
        if isinstance(value, dict) and {"artifact_id", "logical_path", "sha256", "size_bytes"} <= value.keys():
            entries.append((path, value))
    return entries


def _semantic_artifacts(document: Mapping[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    seen_ids: dict[str, tuple[str, Mapping[str, Any]]] = {}
    seen_paths: dict[str, str] = {}
    for path, entry in _artifact_entries(document):
        artifact_id = entry["artifact_id"]
        logical_path = entry["logical_path"]
        if artifact_id in seen_ids:
            issues.append(Issue(f"{path}.artifact_id", f"duplicate artifact_id {artifact_id!r} in one manifest"))
        else:
            seen_ids[artifact_id] = (path, entry)
        if logical_path in seen_paths:
            issues.append(Issue(f"{path}.logical_path", f"duplicate logical_path; first used at {seen_paths[logical_path]}"))
        else:
            seen_paths[logical_path] = path
        if not isinstance(logical_path, str):
            continue
        pure_path = PurePosixPath(logical_path)
        if (
            "\\" in logical_path
            or pure_path.is_absolute()
            or logical_path in {"", "."}
            or any(part in {"", ".", ".."} for part in pure_path.parts)
            or pure_path.as_posix() != logical_path
        ):
            issues.append(Issue(f"{path}.logical_path", "must be a normalized relative POSIX path without '.' or '..'"))
    return issues


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value[:-1] + "+00:00")


def _evaluate(observed: Any, operator: str, threshold: Any) -> bool:
    if operator == "eq":
        return _json_equal(observed, threshold)
    if not (
        isinstance(observed, (int, float))
        and not isinstance(observed, bool)
        and isinstance(threshold, (int, float))
        and not isinstance(threshold, bool)
        and math.isfinite(observed)
        and math.isfinite(threshold)
    ):
        return False
    return {
        "lt": observed < threshold,
        "lte": observed <= threshold,
        "gt": observed > threshold,
        "gte": observed >= threshold,
    }[operator]


def load_release_spec(path: Path | str = DEFAULT_RELEASE_SPEC) -> Mapping[str, Any]:
    spec = load_json(path)
    issues: list[Issue] = []
    if not isinstance(spec, dict):
        raise ContractError((Issue("$", "release specification must be an object"),))

    locked_values = {
        ("spec_version",): "2.1.0",
        ("release_mode",): "positive_only",
        ("canonical_data", "snapshot_id"): "damit-20250610T000301Z",
        ("canonical_data", "minimum_quality_flag"): 3,
        ("canonical_data", "coordinate_frame"): "ecliptic_j2000",
        ("canonical_data", "directed_poles"): True,
        ("canonical_data", "antipodal_equivalence"): False,
        ("randomness", "training_seeds"): [17, 42, 137, 777, 2027],
        ("randomness", "bootstrap_seed"): 20260901,
        ("randomness", "bootstrap_resamples"): 10000,
        ("evaluation", "primary_metrics", "pole"): "directed_top1_angular_error_deg",
        ("evaluation", "primary_metrics", "period"): "period_accuracy_at_5_percent",
        ("evaluation", "hypothesis_tests", "familywise_alpha"): 0.01,
        ("evaluation", "hypothesis_tests", "multiplicity_correction"): "holm",
        ("evaluation", "hypothesis_tests", "directional_superiority_alternative"): "one_sided_preregistered",
        ("evaluation", "hypothesis_tests", "model_comparison"): "paired_object_bootstrap_and_permutation",
        ("evaluation", "hypothesis_tests", "prior_swap_test"): "paired_one_sided_permutation",
        ("evaluation", "hypothesis_tests", "p_values"): "holm_adjusted_within_preregistered_superiority_family",
        ("evaluation", "reporting", "headline_pole_metric"): "directed_top1_angular_error_deg",
        ("evaluation", "reporting", "matched_k_is_oracle_only"): True,
        ("evaluation", "reporting", "matched_k_reporting_role"): "secondary_baseline_not_headline",
        ("evaluation", "aggregation", "one_object_one_vote"): True,
        ("evaluation", "aggregation", "out_of_fold_rows"): "every_expected_object_seed_pair_exactly_once_no_extras",
        ("evaluation", "aggregation", "across_fold_and_seed_summary"): "pool_predictions_to_one_record_per_object_before_summary",
        ("evaluation", "aggregation", "repeated_measurements_do_not_increase_object_weight"): True,
        ("evaluation", "comparators", "deranged_input"): "paired_objectwise_derangement_within_evaluation_fold",
        ("evaluation", "comparators", "matched_k_prior"): "strongest_candidate_selected_on_training_fold_only",
        ("evaluation", "comparators", "label_shuffle"): "training_labels_permuted_before_fit",
        ("evaluation", "model_selection", "checkpoint_selection_data"): "training_and_validation_only",
        ("evaluation", "model_selection", "test_selected_checkpoint_forbidden"): True,
        ("evaluation", "model_selection", "checkpoint_schema_mismatch"): "reject",
        ("evaluation", "period_calibration", "coverage_68_acceptable_range"): [0.63, 0.73],
        ("evaluation", "period_calibration", "coverage_95_acceptable_range"): [0.92, 0.98],
        ("evaluation", "period_calibration", "nominal_coverage_must_be_inside_bootstrap_ci"): True,
        ("evaluation", "period_calibration", "maximum_expected_calibration_error"): 0.05,
        ("evaluation", "period_consensus", "default"): "disabled",
        ("evaluation", "period_consensus", "retain_only_if"): "acc5_absolute_gain_or_median_relative_error_reduction_and_significance",
        ("evaluation", "period_consensus", "minimum_acc5_absolute_gain_percentage_points"): 5.0,
        ("evaluation", "period_consensus", "minimum_median_relative_error_fractional_reduction"): 0.10,
        ("evaluation", "period_consensus", "effect_size_logic"): "either",
        ("evaluation", "period_consensus", "maximum_one_sided_holm_adjusted_p_value"): 0.01,
        ("evaluation", "period_consensus", "otherwise"): "remove_consensus_from_v2_pipeline_and_claims",
        ("prospective_test", "required"): True,
        ("prospective_test", "split_locked_before_label_reveal"): True,
        ("prospective_test", "labels_held_by_independent_custodian"): True,
        ("prospective_test", "no_test_set_tuning"): True,
        ("prospective_test", "single_final_evaluation"): True,
        ("prediction_contract", "output"): "calibrated_healpix_density",
        ("prediction_contract", "abstention_required"): True,
        ("prediction_contract", "risk_coverage_required"): True,
        ("artifact_policy", "archive"): "zenodo",
        ("artifact_policy", "immutable_record_required"): True,
        ("artifact_policy", "clean_git_commit_required"): True,
    }
    for keys, expected in locked_values.items():
        value: Any = spec
        try:
            for key in keys:
                value = value[key]
        except (KeyError, TypeError):
            issues.append(Issue("$." + ".".join(keys), "missing locked release setting"))
            continue
        if not _json_equal(value, expected):
            issues.append(Issue("$." + ".".join(keys), f"locked value must equal {expected!r}"))

    gates = spec.get("required_gates")
    if not isinstance(gates, list) or not gates:
        issues.append(Issue("$.required_gates", "must be a non-empty array"))
    else:
        gate_ids: set[str] = set()
        gate_by_id: dict[str, Mapping[str, Any]] = {}
        for index, gate in enumerate(gates):
            path_prefix = f"$.required_gates[{index}]"
            if not isinstance(gate, dict):
                issues.append(Issue(path_prefix, "must be an object"))
                continue
            required = {"gate_id", "category", "metric", "operator", "threshold"}
            if set(gate) != required:
                issues.append(Issue(path_prefix, f"fields must be exactly {sorted(required)!r}"))
                continue
            gate_id = gate["gate_id"]
            if not isinstance(gate_id, str) or re.fullmatch(r"[A-Z][A-Z0-9_]+", gate_id) is None:
                issues.append(Issue(f"{path_prefix}.gate_id", "must be an uppercase stable identifier"))
            elif gate_id in gate_ids:
                issues.append(Issue(f"{path_prefix}.gate_id", f"duplicate gate_id {gate_id!r}"))
            if isinstance(gate_id, str):
                gate_ids.add(gate_id)
                gate_by_id[gate_id] = gate
            if not isinstance(gate["operator"], str) or gate["operator"] not in OPERATORS:
                issues.append(Issue(f"{path_prefix}.operator", f"must be one of {sorted(OPERATORS)!r}"))
            if not isinstance(gate["category"], str) or not gate["category"]:
                issues.append(Issue(f"{path_prefix}.category", "must be a non-empty string"))
            if not isinstance(gate["metric"], str) or not gate["metric"]:
                issues.append(Issue(f"{path_prefix}.metric", "must be a non-empty string"))
            threshold = gate["threshold"]
            if not isinstance(threshold, (str, bool, int, float, list)) or (
                isinstance(threshold, float) and not math.isfinite(threshold)
            ):
                issues.append(Issue(f"{path_prefix}.threshold", "must be a finite JSON scalar or array"))
            if gate["operator"] != "eq" and not (
                isinstance(threshold, (int, float))
                and not isinstance(threshold, bool)
                and math.isfinite(threshold)
            ):
                issues.append(Issue(f"{path_prefix}.threshold", "ordered comparisons require a finite numeric threshold"))
        for gate_id, expected in LOCKED_GATE_CONTRACTS.items():
            gate = gate_by_id.get(gate_id)
            if gate is None:
                issues.append(Issue("$.required_gates", f"missing locked gate {gate_id!r}"))
                continue
            expected_fields = dict(zip(("category", "metric", "operator", "threshold"), expected))
            for field, expected_value in expected_fields.items():
                if not _json_equal(gate.get(field), expected_value):
                    issues.append(
                        Issue(
                            f"$.required_gates.{gate_id}.{field}",
                            f"locked gate value must equal {expected_value!r}",
                        )
                    )
        forbidden = sorted(FORBIDDEN_GATE_IDS & gate_ids)
        if forbidden:
            issues.append(Issue("$.required_gates", f"obsolete or scientifically inappropriate gates are forbidden: {forbidden!r}"))
    issues.extend(_semantic_common(spec))
    if issues:
        raise ContractError(issues)
    return spec


def _manifest_id(document: Mapping[str, Any]) -> str | None:
    return {
        "data": document.get("manifest_id"),
        "run": document.get("run_id"),
        "artifact": document.get("record_id"),
        "verdict": document.get("release_id"),
    }.get(document.get("manifest_type"))


def validate_manifest(
    document: Any,
    *,
    release_spec: Mapping[str, Any] | None = None,
    release_spec_path: Path | str = DEFAULT_RELEASE_SPEC,
) -> None:
    """Validate one parsed manifest; raise :class:`ContractError` on failure."""

    if not isinstance(document, dict):
        raise ContractError((Issue("$", "manifest must be a JSON object"),))
    manifest_type = document.get("manifest_type")
    if manifest_type not in SCHEMA_FILES:
        raise ContractError((Issue("$.manifest_type", f"must be one of {sorted(SCHEMA_FILES)!r}"),))

    schema = load_json(SCHEMA_FILES[manifest_type])
    issues = _schema_issues(document, schema, schema)
    issues.extend(_semantic_common(document))
    issues.extend(_semantic_artifacts(document))

    spec = release_spec if release_spec is not None else load_release_spec(release_spec_path)
    expected_spec_hash = sha256_file(release_spec_path)
    if manifest_type in {"run", "verdict"} and document.get("release_spec_sha256") != expected_spec_hash:
        issues.append(Issue("$.release_spec_sha256", "does not match the exact release specification bytes"))

    if manifest_type == "data":
        split_names = [split.get("name") for split in document.get("splits", []) if isinstance(split, dict)]
        if len(split_names) != len(set(split_names)):
            issues.append(Issue("$.splits", "split names must be unique"))
        for uri_path in ("canonical_uri",):
            source = document.get("source", {})
            uri = source.get(uri_path) if isinstance(source, dict) else None
            if isinstance(uri, str) and not _is_remote_https_uri(uri):
                issues.append(Issue(f"$.source.{uri_path}", "must be a remote HTTPS URI"))
        repository = document.get("builder", {}).get("git_repository") if isinstance(document.get("builder"), dict) else None
        if isinstance(repository, str) and not _is_remote_https_uri(repository):
            issues.append(Issue("$.builder.git_repository", "must be a remote HTTPS URI"))

    elif manifest_type == "run":
        repository = document.get("git", {}).get("repository") if isinstance(document.get("git"), dict) else None
        if isinstance(repository, str) and not _is_remote_https_uri(repository):
            issues.append(Issue("$.git.repository", "must be a remote HTTPS URI"))
        input_ids = {item.get("artifact_id") for item in document.get("inputs", []) if isinstance(item, dict)}
        output_ids = {item.get("artifact_id") for item in document.get("outputs", []) if isinstance(item, dict)}
        overlap = input_ids & output_ids
        if overlap:
            issues.append(Issue("$.outputs", f"inputs and outputs reuse artifact IDs: {sorted(overlap)!r}"))

    elif manifest_type == "artifact":
        archive = document.get("archive", {})
        record_url = archive.get("record_url") if isinstance(archive, dict) else None
        doi = archive.get("doi") if isinstance(archive, dict) else None
        if isinstance(record_url, str) and not _is_remote_https_uri(record_url):
            issues.append(Issue("$.archive.record_url", "must be a remote HTTPS URI"))
        if isinstance(doi, str) and isinstance(record_url, str) and doi.startswith("10.5281/zenodo."):
            zenodo_id = doi.rsplit(".", 1)[-1]
            if urlparse(record_url).netloc != "zenodo.org" or not urlparse(record_url).path.rstrip("/").endswith("/" + zenodo_id):
                issues.append(Issue("$.archive", "Zenodo DOI and immutable record URL identify different records"))

    elif manifest_type == "verdict":
        gate_contracts = {gate["gate_id"]: gate for gate in spec["required_gates"]}
        results = document.get("gate_results", [])
        result_ids = [result.get("gate_id") for result in results if isinstance(result, dict)]
        if len(result_ids) != len(set(result_ids)):
            issues.append(Issue("$.gate_results", "gate_id values must be unique"))
        missing = sorted(set(gate_contracts) - set(result_ids))
        extra = sorted(set(result_ids) - set(gate_contracts))
        if missing:
            issues.append(Issue("$.gate_results", f"missing required gates: {missing!r}"))
        if extra:
            issues.append(Issue("$.gate_results", f"unknown gates: {extra!r}"))
        calculated_passes: list[bool] = []
        observed_by_metric: dict[str, tuple[Any, str]] = {}
        for index, result in enumerate(results):
            if not isinstance(result, dict) or result.get("gate_id") not in gate_contracts:
                continue
            contract = gate_contracts[result["gate_id"]]
            prefix = f"$.gate_results[{index}]"
            for field in ("metric", "operator", "threshold"):
                if not _json_equal(result.get(field), contract[field]):
                    issues.append(Issue(f"{prefix}.{field}", "does not match the immutable release gate"))
            metric = contract["metric"]
            if metric in observed_by_metric and not _json_equal(result.get("observed"), observed_by_metric[metric][0]):
                issues.append(
                    Issue(
                        f"{prefix}.observed",
                        f"conflicts with the same metric reported at {observed_by_metric[metric][1]}",
                    )
                )
            else:
                observed_by_metric[metric] = (result.get("observed"), f"{prefix}.observed")
            operator = contract["operator"]
            calculated = _evaluate(result.get("observed"), operator, contract["threshold"])
            calculated_passes.append(calculated)
            if result.get("passed") is not calculated:
                issues.append(Issue(f"{prefix}.passed", f"must be {calculated!r} for the observed value"))
        all_passed = len(calculated_passes) == len(gate_contracts) and all(calculated_passes)
        expected_verdict = "PASS" if all_passed else "FAIL"
        if document.get("overall_verdict") != expected_verdict:
            issues.append(Issue("$.overall_verdict", f"must be {expected_verdict!r} from required gate results"))
        if document.get("approved_for_release") is not all_passed:
            issues.append(Issue("$.approved_for_release", f"must be {all_passed!r}; V2 is positive-release-only"))
        prospective = document.get("prospective_test", {})
        if isinstance(prospective, dict):
            lock = prospective.get("locked_at_utc")
            reveal = prospective.get("labels_revealed_at_utc")
            evaluated = document.get("evaluated_at_utc")
            if all(isinstance(item, str) and _valid_datetime(item) for item in (lock, reveal, evaluated)):
                if not (_parse_utc(lock) < _parse_utc(reveal) <= _parse_utc(evaluated)):
                    issues.append(Issue("$.prospective_test", "timestamps must satisfy locked < label reveal <= evaluation"))

    if issues:
        raise ContractError(issues)


def validate_bundle(
    manifest_paths: Sequence[Path | str],
    *,
    release_spec_path: Path | str = DEFAULT_RELEASE_SPEC,
    artifact_root: Path | str | None = None,
) -> None:
    """Validate manifests together, including references and optional file bytes."""

    if not manifest_paths:
        raise ContractError((Issue("$", "at least one manifest is required"),))
    spec = load_release_spec(release_spec_path)
    loaded: list[tuple[Path, Mapping[str, Any]]] = []
    issues: list[Issue] = []
    for raw_path in manifest_paths:
        path = Path(raw_path)
        document = load_json(path)
        try:
            validate_manifest(document, release_spec=spec, release_spec_path=release_spec_path)
        except ContractError as exc:
            issues.extend(Issue(f"{path}:{issue.path}", issue.message) for issue in exc.issues)
        if isinstance(document, dict):
            loaded.append((path, document))

    by_type_and_id: dict[tuple[str, str], tuple[Path, Mapping[str, Any]]] = {}
    for path, document in loaded:
        manifest_type = document.get("manifest_type")
        identity = _manifest_id(document)
        if not isinstance(manifest_type, str) or not isinstance(identity, str):
            continue
        key = (manifest_type, identity)
        if key in by_type_and_id:
            issues.append(Issue(str(path), f"duplicate {manifest_type} manifest identity {identity!r}"))
        else:
            by_type_and_id[key] = (path, document)

    artifact_registry: dict[str, tuple[str, int, str]] = {}
    artifact_catalog: dict[str, Mapping[str, Any]] = {}
    for path, document in loaded:
        for entry_path, entry in _artifact_entries(document):
            artifact_id = entry.get("artifact_id")
            signature = (entry.get("sha256"), entry.get("size_bytes"), entry.get("logical_path"))
            if not isinstance(artifact_id, str):
                continue
            if artifact_id in artifact_registry and artifact_registry[artifact_id] != signature:
                issues.append(Issue(f"{path}:{entry_path}", f"artifact {artifact_id!r} conflicts with another manifest"))
            else:
                artifact_registry[artifact_id] = signature
            if document.get("manifest_type") == "artifact":
                artifact_catalog[artifact_id] = entry

    for path, document in loaded:
        if document.get("manifest_type") == "run":
            for index, reference in enumerate(document.get("data_manifests", [])):
                if not isinstance(reference, dict):
                    continue
                target = by_type_and_id.get(("data", reference.get("manifest_id")))
                ref_path = f"{path}:$.data_manifests[{index}]"
                if target is None:
                    issues.append(Issue(ref_path, "referenced data manifest is not present in the validation bundle"))
                elif reference.get("sha256") != sha256_file(target[0]):
                    issues.append(Issue(ref_path + ".sha256", "does not match referenced data manifest bytes"))

        elif document.get("manifest_type") == "artifact":
            for index, artifact in enumerate(document.get("artifacts", [])):
                if not isinstance(artifact, dict):
                    continue
                run_id = artifact.get("generated_by_run_id")
                target = by_type_and_id.get(("run", run_id))
                ref_path = f"{path}:$.artifacts[{index}].generated_by_run_id"
                if target is None:
                    issues.append(Issue(ref_path, "generating run manifest is not present in the validation bundle"))
                else:
                    matching_outputs = [
                        item for item in target[1].get("outputs", [])
                        if isinstance(item, dict) and item.get("artifact_id") == artifact.get("artifact_id")
                    ]
                    if not matching_outputs:
                        issues.append(Issue(ref_path, "generating run does not declare this artifact as an output"))

        elif document.get("manifest_type") == "verdict":
            for field, target_type in (("run_manifest", "run"), ("artifact_manifest", "artifact")):
                reference = document.get(field)
                if not isinstance(reference, dict):
                    continue
                target = by_type_and_id.get((target_type, reference.get("manifest_id")))
                ref_path = f"{path}:$.{field}"
                if target is None:
                    issues.append(Issue(ref_path, f"referenced {target_type} manifest is not present in the validation bundle"))
                elif reference.get("sha256") != sha256_file(target[0]):
                    issues.append(Issue(ref_path + ".sha256", f"does not match referenced {target_type} manifest bytes"))
            for index, gate in enumerate(document.get("gate_results", [])):
                if not isinstance(gate, dict):
                    continue
                for evidence_index, evidence in enumerate(gate.get("evidence", [])):
                    if not isinstance(evidence, dict):
                        continue
                    artifact_id = evidence.get("artifact_id")
                    catalog_entry = artifact_catalog.get(artifact_id)
                    ref_path = f"{path}:$.gate_results[{index}].evidence[{evidence_index}]"
                    if catalog_entry is None:
                        issues.append(Issue(ref_path, "evidence artifact is absent from the immutable artifact manifest"))
                    elif evidence.get("sha256") != catalog_entry.get("sha256"):
                        issues.append(Issue(ref_path + ".sha256", "does not match immutable artifact manifest"))

    if artifact_root is not None:
        root = Path(artifact_root).resolve()
        if not root.is_dir():
            issues.append(Issue(str(root), "artifact root must be an existing directory"))
        else:
            verified: set[tuple[str, str, int]] = set()
            for path, document in loaded:
                for entry_path, entry in _artifact_entries(document):
                    signature = (entry.get("logical_path"), entry.get("sha256"), entry.get("size_bytes"))
                    if not all(isinstance(item, expected) for item, expected in zip(signature, (str, str, int))):
                        continue
                    if signature in verified:
                        continue
                    verified.add(signature)
                    candidate = root / signature[0]
                    label = f"{path}:{entry_path}.logical_path"
                    try:
                        resolved = candidate.resolve(strict=True)
                        resolved.relative_to(root)
                    except (FileNotFoundError, OSError, ValueError):
                        issues.append(Issue(label, "artifact is missing or resolves outside the artifact root"))
                        continue
                    if candidate.is_symlink() or not resolved.is_file():
                        issues.append(Issue(label, "artifact must be a regular non-symlink file"))
                        continue
                    actual_size = resolved.stat().st_size
                    if actual_size != signature[2]:
                        issues.append(Issue(label, f"size mismatch: declared {signature[2]}, found {actual_size}"))
                    actual_hash = sha256_file(resolved)
                    if actual_hash != signature[1]:
                        issues.append(Issue(label, f"SHA-256 mismatch: declared {signature[1]}, found {actual_hash}"))

    if issues:
        raise ContractError(issues)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="*", type=Path, help="JSON manifest files to validate together")
    parser.add_argument(
        "--release-spec",
        type=Path,
        default=DEFAULT_RELEASE_SPEC,
        help="JSON-compatible YAML release policy (default: repro/release_spec.yaml)",
    )
    parser.add_argument("--artifact-root", type=Path, help="also verify every declared artifact's bytes below this root")
    parser.add_argument("--spec-only", action="store_true", help="validate only the locked release specification")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        load_release_spec(args.release_spec)
        if not args.spec_only:
            if not args.manifests:
                raise ContractError((Issue("$", "provide one or more manifests, or use --spec-only"),))
            validate_bundle(
                args.manifests,
                release_spec_path=args.release_spec,
                artifact_root=args.artifact_root,
            )
    except ContractError as exc:
        print("INVALID", file=sys.stderr)
        for issue in exc.issues:
            print(f"- {issue}", file=sys.stderr)
        return 2
    print("VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
