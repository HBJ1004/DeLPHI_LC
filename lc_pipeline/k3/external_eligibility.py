"""Metadata-only eligibility for external generalization cohorts.

This module never reads predictions, pole coordinates, oracle errors, or model
scores.  It turns a predeclared metadata table into a deterministic inventory
for a later lock.  Unknown provenance remains unknown; it is never promoted to
an independent or confirmatory status.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence

ELIGIBILITY_SCHEMA = "delphi.k3-external-eligibility.v1"
UNKNOWN = "unknown"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PHYSICAL_ID = re.compile(r"^(?:mpc:[1-9][0-9]*|provisional:[^:\s][^\s]*)$")

REQUIRED_FIELDS = (
    "physical_identity", "source_database_id", "alias_receipt", "input_source",
    "raw_source_version", "input_receipt_hash", "valid_points", "native_sessions",
    "documented_apparitions", "apparition_evidence", "observer", "time_scale",
    "photometric_convention", "filter", "period_source", "period_precision",
    "period_quality", "reference_source", "reference_quality_tier",
    "reference_input_overlap", "direct_training_exposure", "synthetic_donor_exposure",
    "selection_exposure", "history_unknown",
)
ALLOWED_INPUT_SOURCES = {"dense_alcdef", "atlas_sscat", "tess_tssys_dr1"}
EXPOSURE_VALUES = {"clear", "exposed", UNKNOWN}
REFERENCE_TIERS = {"catalog_agreement", "independent_supported", "provenance_unknown"}
FORBIDDEN_OUTCOME_KEYS = {
    "pole", "axis", "reference_axis", "oracle_error", "oracle_at_3",
    "score", "model_score", "prediction", "prediction_error", "top1",
    "top_1", "top3", "within_20_deg", "within_30_deg",
}


class ExternalEligibilityError(ValueError):
    """Raised when metadata cannot support a fail-closed eligibility decision."""


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _unknown(value: object) -> bool:
    return value == UNKNOWN


def _positive_int(value: object, field: str) -> int | None:
    if _unknown(value):
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExternalEligibilityError(f"{field} must be a nonnegative integer or 'unknown'")
    return value


def _hash(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value.lower()):
        raise ExternalEligibilityError(f"{field} must be a lowercase SHA-256")
    return value.lower()


def _validate_no_outcomes(row: Mapping[str, Any]) -> None:
    present = {str(key).lower() for key in row}
    bad = sorted(present & FORBIDDEN_OUTCOME_KEYS)
    if bad:
        raise ExternalEligibilityError(
            "eligibility metadata must not contain outcome fields: " + ", ".join(bad)
        )


def _validate_row(row: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise ExternalEligibilityError("each eligibility row must be an object")
    _validate_no_outcomes(row)
    missing = [field for field in REQUIRED_FIELDS if field not in row]
    if missing:
        raise ExternalEligibilityError("missing required metadata: " + ", ".join(missing))
    result = {field: row[field] for field in REQUIRED_FIELDS}
    physical = result["physical_identity"]
    if not isinstance(physical, str) or not _PHYSICAL_ID.fullmatch(physical):
        raise ExternalEligibilityError("physical_identity must use an explicit canonical namespace")
    if not isinstance(result["source_database_id"], str) or not result["source_database_id"].strip():
        raise ExternalEligibilityError("source_database_id must be a nonempty string")
    if not isinstance(result["alias_receipt"], str) or not result["alias_receipt"].strip():
        raise ExternalEligibilityError("alias_receipt must be a nonempty receipt identifier")
    if result["input_source"] not in ALLOWED_INPUT_SOURCES:
        raise ExternalEligibilityError("unsupported input_source")
    if not isinstance(result["raw_source_version"], str) or not result["raw_source_version"].strip():
        raise ExternalEligibilityError("raw_source_version must be a nonempty string")
    result["input_receipt_hash"] = _hash(result["input_receipt_hash"], "input_receipt_hash")
    result["alias_receipt"] = _hash(result["alias_receipt"], "alias_receipt")
    for field in ("valid_points", "native_sessions", "documented_apparitions"):
        _positive_int(result[field], field)
    if not isinstance(result["apparition_evidence"], (Mapping, list, str)):
        raise ExternalEligibilityError("apparition_evidence must be structured metadata or 'unknown'")
    if result["period_quality"] != UNKNOWN and not isinstance(result["period_quality"], str):
        raise ExternalEligibilityError("period_quality must be a string or 'unknown'")
    if result["reference_quality_tier"] not in REFERENCE_TIERS:
        raise ExternalEligibilityError("invalid reference_quality_tier")
    for field in ("direct_training_exposure", "synthetic_donor_exposure", "selection_exposure"):
        if result[field] not in EXPOSURE_VALUES:
            raise ExternalEligibilityError(f"invalid {field}")
    if not isinstance(result["history_unknown"], bool):
        raise ExternalEligibilityError("history_unknown must be boolean")
    return result


def build_eligibility_inventory(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_valid_points: int = 150,
    minimum_sessions: int = 5,
    minimum_apparitions: int = 2,
) -> dict[str, Any]:
    """Return a stable metadata-only inventory and computed eligibility roles."""
    if not rows:
        raise ExternalEligibilityError("at least one metadata row is required")
    if min(minimum_valid_points, minimum_sessions, minimum_apparitions) < 1:
        raise ExternalEligibilityError("eligibility thresholds must be positive")
    normalized = [_validate_row(row) for row in rows]
    seen: set[tuple[str, str]] = set()
    output: list[dict[str, Any]] = []
    for row in sorted(normalized, key=lambda item: (item["physical_identity"], item["input_source"])):
        key = (row["physical_identity"], row["input_source"])
        if key in seen:
            raise ExternalEligibilityError("duplicate physical identity/input source")
        seen.add(key)
        reasons: list[str] = []
        counts = {
            "valid_points": _positive_int(row["valid_points"], "valid_points"),
            "native_sessions": _positive_int(row["native_sessions"], "native_sessions"),
            "documented_apparitions": _positive_int(row["documented_apparitions"], "documented_apparitions"),
        }
        if any(value is None for value in counts.values()):
            reasons.append("missing_required_count")
        else:
            if counts["valid_points"] < minimum_valid_points:
                reasons.append("too_few_valid_points")
            if counts["native_sessions"] < minimum_sessions:
                reasons.append("too_few_native_sessions")
            if counts["documented_apparitions"] < minimum_apparitions:
                reasons.append("too_few_documented_apparitions")
        if row["history_unknown"]:
            reasons.append("history_unknown")
        for field in ("direct_training_exposure", "synthetic_donor_exposure", "selection_exposure"):
            if row[field] != "clear":
                reasons.append(f"{field}:{row[field]}")
        if row["period_source"] == UNKNOWN or row["period_precision"] == UNKNOWN or row["period_quality"] == UNKNOWN:
            reasons.append("period_provenance_unknown")
        if row["reference_quality_tier"] == "provenance_unknown":
            reasons.append("reference_provenance_unknown")
        if row["reference_input_overlap"] not in {False, "no", "unknown"}:
            reasons.append("reference_input_overlap")
        if reasons:
            role = "unknown" if any(
                token in {"missing_required_count", "history_unknown", "period_provenance_unknown", "reference_provenance_unknown", "reference_input_overlap"}
                or token.endswith(":unknown") for token in reasons
            ) else "excluded"
        elif row["reference_quality_tier"] == "independent_supported":
            role = "confirmatory_candidate"
        else:
            role = "development_candidate"
        output.append({**row, "eligible_role": role, "exclusion_reasons": reasons})
    policy = {
        "minimum_valid_points": minimum_valid_points,
        "minimum_sessions": minimum_sessions,
        "minimum_documented_apparitions": minimum_apparitions,
        "reference_pole_values_used": False,
        "model_scores_used": False,
        "selection_errors_used": False,
    }
    policy_hash = _sha256_bytes(canonical_json(policy).encode())
    counts: dict[str, int] = {}
    for row in output:
        counts[row["eligible_role"]] = counts.get(row["eligible_role"], 0) + 1
    return {
        "schema": ELIGIBILITY_SCHEMA,
        "purpose": "metadata-only external cohort eligibility; not validation results",
        "policy": policy,
        "policy_sha256": policy_hash,
        "counts": counts,
        "rows": output,
    }
