"""Conservative contracts and object-level statistics for K3 generalization.

This module deliberately contains no data discovery and no model execution.  It
operates on explicitly supplied manifests, keeps asteroid identity as the unit
of analysis, and treats a missing or malformed prediction as a 90 degree loss.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..physics.axial import axial_angular_error_deg

GENERALIZATION_ANALYSIS_SCHEMA = "delphi.k3-generalization-analysis.v1"
EXPOSURE_AUDIT_SCHEMA = "delphi.k3-identity-exposure-audit.v1"
REFERENCE_OVERLAP_SCHEMA = "delphi.k3-reference-photometry-overlap.v1"
PROTOCOL_LOCK_SCHEMA = "delphi.k3-generalization-protocol-lock.v1"
PREDICTION_RECEIPT_SCHEMA = "delphi.k3-generalization-prediction-receipt.v1"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260910
FAILURE_ERROR_DEG = 90.0
MIN_FAMILY_CLUSTERS = 10
_SHA256 = __import__("re").compile(r"^[0-9a-f]{64}$")
_CANONICAL_ID = __import__("re").compile(
    r"^(?:damit:[1-9][0-9]*|mpc:[1-9][0-9]*|provisional:[^:\s][^\s]*)$"
)


def _canonical_syntax(identity: str) -> bool:
    return bool(_CANONICAL_ID.fullmatch(identity))


class GeneralizationValidationError(ValueError):
    """Raised when a generalization artifact violates its explicit contract."""


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_path(path: str | Path) -> tuple[str, str, int]:
    """Hash one file or a directory tree, binding paths and file bytes."""
    source = Path(path)
    if source.is_file():
        return sha256_file(source), "file", source.stat().st_size
    if not source.is_dir():
        raise GeneralizationValidationError(f"resource does not exist: {source}")
    digest = hashlib.sha256()
    total = 0
    files = sorted(item for item in source.rglob("*") if item.is_file())
    if not files:
        raise GeneralizationValidationError(f"resource directory is empty: {source}")
    for item in files:
        relative = item.relative_to(source).as_posix().encode("utf-8")
        payload_hash = bytes.fromhex(sha256_file(item))
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(payload_hash)
        total += item.stat().st_size
    return digest.hexdigest(), "directory", total


_ID_KEYS = {
    "object_id",
    "object_ids",
    "train_ids",
    "validation_ids",
    "calibration_ids",
    "test_ids",
    "source_object_id",
    "source_object_ids",
    "donor_object_id",
    "donor_object_ids",
    "shape_donor_id",
    "shape_donor_ids",
    "geometry_donor_id",
    "geometry_donor_ids",
    "noise_donor_id",
    "noise_donor_ids",
    "history_object_id",
    "history_object_ids",
    "parent_object_id",
    "parent_object_ids",
    "damit_identity",
    "physical_identity",
}
_DAMIT_NUMERIC_KEYS = {"damit_id", "source_asteroid_id"}
_SHAPE_DONOR_KEYS = {"shape_donor_id", "shape_donor_ids"}
_MPC_NUMERIC_KEYS = {
    "mpc_number",
    "resolved_mpc_number",
    "asteroid_number",
    "expected_mpc_number",
    "observed_mpc_number",
}


def _identity_values(value: object) -> set[str]:
    found: set[str] = set()

    def visit(item: object, key: str | None = None) -> None:
        if key in _DAMIT_NUMERIC_KEYS:
            if isinstance(item, int) and not isinstance(item, bool) and item > 0:
                found.add(f"damit:{item}")
            return
        if key in _MPC_NUMERIC_KEYS:
            if isinstance(item, int) and not isinstance(item, bool) and item > 0:
                found.add(f"mpc:{item}")
            return
        if key in _ID_KEYS:
            values = item if isinstance(item, (list, tuple, set)) else [item]
            for candidate in values:
                if isinstance(candidate, str) and candidate.strip():
                    identity = candidate.strip()
                    # Published synthetic shard manifests qualify a shape with
                    # ``/model_<id>``.  Exposure is attached to the asteroid,
                    # not to that model-row suffix.
                    if key in _SHAPE_DONOR_KEYS:
                        identity = identity.split("/", 1)[0]
                    found.add(identity)
            return
        if isinstance(item, Mapping):
            for child_key, child in item.items():
                visit(child, str(child_key))
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child, key)

    visit(value)
    return found


def _parse_aliases(value: object | None) -> tuple[dict[str, str], set[str], str | None]:
    if value is None:
        return {}, set(), None
    if not isinstance(value, Mapping):
        raise GeneralizationValidationError(
            "alias crosswalk must be an object with an official_table_sha256"
        )
    official_hash = value.get("official_table_sha256", value.get("identity_table_sha256"))
    if not isinstance(official_hash, str) or not _SHA256.fullmatch(official_hash.lower()):
        raise GeneralizationValidationError(
            "alias crosswalk requires a valid official identity-table SHA-256"
        )
    raw = value.get("aliases")
    identity_rows = value.get("objects")
    is_identity_table = (
        value.get("schema") == "delphi.k3-survey-identity-map.v1"
        or value.get("mapping_authority") == "official_DAMIT_asteroids_id_to_number"
        or (raw is None and isinstance(identity_rows, list))
    )
    if raw is None and is_identity_table:
        if not isinstance(identity_rows, list):
            raise GeneralizationValidationError("survey identity map has no objects")
        generated: list[dict[str, object]] = []
        for identity_row in identity_rows:
            if not isinstance(identity_row, Mapping):
                raise GeneralizationValidationError(
                    "survey identity map contains an invalid row"
                )
            legacy = identity_row.get("object_id")
            damit = identity_row.get("damit_identity")
            if damit is None:
                numeric_damit = identity_row.get("damit_id")
                if (
                    isinstance(numeric_damit, int)
                    and not isinstance(numeric_damit, bool)
                    and numeric_damit > 0
                ):
                    damit = f"damit:{numeric_damit}"
            physical = identity_row.get("physical_identity")
            if physical is None:
                numeric_mpc = identity_row.get("mpc_number")
                if (
                    isinstance(numeric_mpc, int)
                    and not isinstance(numeric_mpc, bool)
                    and numeric_mpc > 0
                ):
                    physical = f"mpc:{numeric_mpc}"
            if (
                not isinstance(legacy, str)
                or not isinstance(damit, str)
                or not _canonical_syntax(damit)
            ):
                raise GeneralizationValidationError(
                    "survey identity map row lacks identity fields"
                )
            generated.append({"alias": legacy, "canonical_id": damit})
            if physical is not None:
                if not isinstance(physical, str):
                    raise GeneralizationValidationError(
                        "physical_identity must be a string or null"
                    )
                generated.append({"alias": damit, "canonical_id": physical})
        raw = generated
    known: dict[str, str] = {}
    unknown: set[str] = set()
    if isinstance(raw, Mapping):
        rows = [
            {"alias": str(alias), "canonical_id": canonical}
            for alias, canonical in raw.items()
        ]
    elif isinstance(raw, list):
        rows = raw
    else:
        raise GeneralizationValidationError("aliases must be a mapping or list")
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("alias"), str):
            raise GeneralizationValidationError("invalid alias record")
        alias = str(row["alias"]).strip()
        if not alias or alias in known or alias in unknown:
            raise GeneralizationValidationError(f"duplicate or empty alias: {alias!r}")
        canonical = row.get("canonical_id")
        status = row.get("status")
        if status == "unknown" or canonical is None:
            unknown.add(alias)
        elif isinstance(canonical, str) and canonical.strip():
            known[alias] = canonical.strip()
        else:
            raise GeneralizationValidationError(f"invalid canonical identity for {alias}")
    return known, unknown, official_hash.lower()


def _resolve_alias(identity: str, aliases: Mapping[str, str], unknown: set[str]) -> str | None:
    current = identity
    seen: set[str] = set()
    while current in aliases:
        if current in seen:
            return None
        seen.add(current)
        current = aliases[current]
    if current in unknown:
        return None
    # Namespace is semantic.  In particular, a legacy ``asteroid_101`` token
    # is a DAMIT-internal identifier and must never be guessed to mean MPC 101.
    return current if _CANONICAL_ID.fullmatch(current) else None


def _source_history_unknown(value: object) -> bool:
    if value is None:
        return True
    if not isinstance(value, Mapping):
        return False
    return (
        value.get("complete") is False
        or value.get("registry_complete") is False
        or value.get("status") in {"unknown", "incomplete", "unaudited"}
        or value.get("coverage") in {"unknown", "incomplete"}
    )


def audit_identity_exposure(
    candidate_ids: Sequence[str],
    *,
    exposure_sources: Mapping[str, object],
    aliases: object | None = None,
) -> dict[str, object]:
    """Conservatively audit original splits and all declared donor histories.

    An explicit unknown alias is never silently promoted to ``unexposed``.
    Candidate order and source-document order do not affect the result.
    """
    candidates = [str(value).strip() for value in candidate_ids]
    if not candidates or any(not value for value in candidates):
        raise GeneralizationValidationError("candidate_ids must be nonempty strings")
    if len(candidates) != len(set(candidates)):
        raise GeneralizationValidationError("candidate_ids contain duplicates")
    alias_map, unknown_aliases, official_table_hash = _parse_aliases(aliases)
    exposed_by_canonical: dict[str, set[str]] = defaultdict(set)
    unresolved_source_ids: dict[str, set[str]] = defaultdict(set)
    unknown_history_roles: list[str] = []
    if not exposure_sources:
        unknown_history_roles.append("__no_exposure_sources_declared__")
    for role in sorted(exposure_sources):
        if not isinstance(role, str) or not role:
            raise GeneralizationValidationError("exposure source roles must be nonempty")
        source_identities = _identity_values(exposure_sources[role])
        if _source_history_unknown(exposure_sources[role]) or not source_identities:
            unknown_history_roles.append(role)
        for identity in source_identities:
            canonical = _resolve_alias(identity, alias_map, unknown_aliases)
            if canonical is None:
                unresolved_source_ids[identity].add(role)
            else:
                exposed_by_canonical[canonical].add(role)
    rows: list[dict[str, object]] = []
    for identity in sorted(candidates):
        canonical = _resolve_alias(identity, alias_map, unknown_aliases)
        if canonical is None:
            status = "unknown"
            reasons = ["identity_alias_resolution_unknown"]
        elif canonical in exposed_by_canonical:
            status = "exposed"
            reasons = [f"present_in:{role}" for role in sorted(exposed_by_canonical[canonical])]
        elif unresolved_source_ids or unknown_history_roles:
            status = "unknown"
            reasons = ["unresolved_exposure_history_prevents_clearance"]
        else:
            status = "unexposed"
            reasons = []
        rows.append(
            {
                "object_id": identity,
                "canonical_object_id": canonical,
                "status": status,
                "reasons": reasons,
            }
        )
    counts = Counter(str(row["status"]) for row in rows)
    return {
        "schema": EXPOSURE_AUDIT_SCHEMA,
        "all_candidates_unexposed": bool(rows) and counts.get("unexposed", 0) == len(rows),
        "counts": {key: counts.get(key, 0) for key in ("unexposed", "exposed", "unknown")},
        "identity_namespace_policy": "explicit_damit_mpc_or_provisional_only",
        "alias_official_table_sha256": official_table_hash,
        "objects": rows,
        "unresolved_source_aliases": [
            {"object_id": identity, "roles": sorted(roles)}
            for identity, roles in sorted(unresolved_source_ids.items())
        ],
        "unknown_history_roles": unknown_history_roles,
    }


def validate_external_exposure_clearance(cohort: object, audit: Mapping[str, object]) -> None:
    """Require an exact, conclusive unexposed audit for a new external cohort."""
    cohort_rows = _rows(cohort, role="external cohort")
    cohort_ids = {str(row["object_id"]) for row in cohort_rows}
    audit_rows = audit.get("objects")
    if (
        audit.get("schema") != EXPOSURE_AUDIT_SCHEMA
        or audit.get("all_candidates_unexposed") is not True
        or not isinstance(audit_rows, list)
    ):
        raise GeneralizationValidationError(
            "external scoring requires a conclusive all-unexposed exposure audit"
        )
    by_id: dict[str, Mapping[str, object]] = {}
    for row in audit_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("object_id"), str):
            raise GeneralizationValidationError("exposure audit contains an invalid row")
        object_id = str(row["object_id"])
        if object_id in by_id:
            raise GeneralizationValidationError("exposure audit contains duplicate object_id values")
        by_id[object_id] = row
    if set(by_id) != cohort_ids:
        raise GeneralizationValidationError("exposure audit identities do not exactly match cohort")
    for object_id, row in by_id.items():
        if (
            row.get("status") != "unexposed"
            or row.get("canonical_object_id") != object_id
            or row.get("reasons") != []
        ):
            raise GeneralizationValidationError(
                f"external identity is not conclusively unexposed: {object_id}"
            )


def _hash_set(record: Mapping[str, object], stem: str) -> set[str]:
    raw = record.get(f"{stem}_hashes", record.get(f"{stem}_sha256", []))
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise GeneralizationValidationError(f"{stem} hashes must be a string or list")
    result = {item.lower() for item in raw}
    if any(not _SHA256.fullmatch(item) for item in result):
        raise GeneralizationValidationError(f"{stem} contains an invalid SHA-256")
    return result


def _identifier_set(record: Mapping[str, object], stem: str) -> set[str]:
    raw = record.get(f"{stem}_ids", record.get(f"{stem}_id", []))
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list) or any(
        not isinstance(item, str) or not item.strip() for item in raw
    ):
        raise GeneralizationValidationError(f"{stem} identifiers must be a string or list")
    return {item.strip() for item in raw}


def classify_reference_photometry_overlap(record: Mapping[str, object]) -> dict[str, object]:
    """Classify label support without inferring independence from missing metadata."""
    object_id = record.get("object_id")
    if not isinstance(object_id, str) or not _canonical_syntax(object_id):
        raise GeneralizationValidationError(
            "reference overlap object_id must have explicit damit:, mpc:, or provisional: namespace"
        )
    prediction = _hash_set(record, "prediction_photometry")
    reference = _hash_set(record, "reference_photometry")
    intersection = sorted(prediction & reference)
    prediction_ids = _identifier_set(record, "prediction_photometry")
    reference_ids = _identifier_set(record, "reference_photometry")
    identifier_intersection = sorted(prediction_ids & reference_ids)
    lineage_complete = record.get("lineage_complete")
    if lineage_complete not in (True, False, None):
        raise GeneralizationValidationError("lineage_complete must be true, false, or null")
    declared = record.get("overlap_declared")
    if declared not in (True, False, None):
        raise GeneralizationValidationError("overlap_declared must be true, false, or null")
    support = record.get("independent_support", [])
    if support is True:
        support_items = ["declared_independent_support"]
    elif support in (False, None):
        support_items = []
    elif isinstance(support, list) and all(isinstance(item, str) and item for item in support):
        support_items = sorted(set(support))
    else:
        raise GeneralizationValidationError("independent_support must be boolean or string list")
    overlap = bool(intersection) or bool(identifier_intersection) or declared is True
    disjoint_known = (
        declared is False
        or (
            lineage_complete is True
            and bool(prediction_ids)
            and bool(reference_ids)
            and not identifier_intersection
        )
    )
    if overlap:
        classification = "catalog_agreement"
    elif disjoint_known and support_items:
        classification = "independent_supported"
    else:
        classification = "unknown"
    return {
        "object_id": object_id,
        "classification": classification,
        "catalog_overlap": overlap,
        "overlap_evidence_sha256": intersection,
        "overlap_evidence_identifiers": identifier_intersection,
        "lineage_complete": lineage_complete is True,
        "independent_support": support_items,
    }


def classify_reference_records(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    rows = [classify_reference_photometry_overlap(record) for record in records]
    ids = [str(row["object_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise GeneralizationValidationError("duplicate reference overlap object_id")
    rows.sort(key=lambda row: str(row["object_id"]))
    counts = Counter(str(row["classification"]) for row in rows)
    return {
        "schema": REFERENCE_OVERLAP_SCHEMA,
        "counts": {key: counts.get(key, 0) for key in (
            "catalog_agreement", "independent_supported", "unknown"
        )},
        "objects": rows,
    }


def _resource_rows(paths: Iterable[str | Path]) -> list[dict[str, object]]:
    normalized = sorted({str(Path(value).resolve()) for value in paths})
    rows = []
    for value in normalized:
        digest, kind, size = sha256_path(value)
        rows.append({"path": value, "kind": kind, "sha256": digest, "size_bytes": size})
    return rows


def create_protocol_lock(
    *,
    spec_path: str | Path,
    code_paths: Sequence[str | Path],
    model_paths: Sequence[str | Path],
    data_paths: Sequence[str | Path],
    reference_paths: Sequence[str | Path],
) -> dict[str, object]:
    """Bind all scoring-relevant bytes; this is not a claim of human blinding."""
    groups = {
        "code": code_paths,
        "model": model_paths,
        "data": data_paths,
        "reference": reference_paths,
    }
    if any(not values for values in groups.values()):
        raise GeneralizationValidationError("every lock resource group must be nonempty")
    spec = Path(spec_path).resolve()
    spec_digest, spec_kind, spec_size = sha256_path(spec)
    if spec_kind != "file":
        raise GeneralizationValidationError("protocol spec must be a file")
    return {
        "schema": PROTOCOL_LOCK_SCHEMA,
        "purpose": "cryptographic_pre_scoring_binding",
        "human_blinding_claimed": False,
        "spec": {
            "path": str(spec), "kind": spec_kind, "sha256": spec_digest,
            "size_bytes": spec_size,
        },
        "resources": {role: _resource_rows(paths) for role, paths in sorted(groups.items())},
    }


def validate_protocol_lock(lock: Mapping[str, object]) -> None:
    if (
        lock.get("schema") != PROTOCOL_LOCK_SCHEMA
        or lock.get("human_blinding_claimed") is not False
        or lock.get("purpose") != "cryptographic_pre_scoring_binding"
    ):
        raise GeneralizationValidationError("invalid protocol lock header")
    spec = lock.get("spec")
    resources = lock.get("resources")
    if not isinstance(spec, Mapping) or not isinstance(resources, Mapping):
        raise GeneralizationValidationError("protocol lock is incomplete")
    groups: list[Mapping[str, object]] = [spec]
    if set(resources) != {"code", "model", "data", "reference"}:
        raise GeneralizationValidationError("protocol lock resource groups are invalid")
    for role in sorted(resources):
        rows = resources[role]
        if not isinstance(rows, list) or not rows:
            raise GeneralizationValidationError(f"protocol lock group {role} is empty")
        groups.extend(rows)
    for row in groups:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise GeneralizationValidationError("invalid locked resource row")
        digest, kind, size = sha256_path(str(row["path"]))
        if row.get("sha256") != digest or row.get("kind") != kind or row.get("size_bytes") != size:
            raise GeneralizationValidationError(f"locked resource changed: {row['path']}")


def create_prediction_receipt(
    *, lock_path: str | Path, prediction_paths: Mapping[str, str | Path]
) -> dict[str, object]:
    lock_source = Path(lock_path).resolve()
    lock = _load_json(lock_source)
    validate_protocol_lock(lock)
    if not prediction_paths:
        raise GeneralizationValidationError("at least one prediction artifact is required")
    rows: dict[str, dict[str, object]] = {}
    for role in sorted(prediction_paths):
        if not role or role in rows:
            raise GeneralizationValidationError("prediction roles must be unique and nonempty")
        path = Path(prediction_paths[role]).resolve()
        digest, kind, size = sha256_path(path)
        if kind != "file":
            raise GeneralizationValidationError("prediction receipts bind files, not directories")
        rows[role] = {"path": str(path), "sha256": digest, "size_bytes": size}
    return {
        "schema": PREDICTION_RECEIPT_SCHEMA,
        "purpose": "sealed_predictions_before_scoring",
        "human_blinding_claimed": False,
        "protocol_lock": {"path": str(lock_source), "sha256": sha256_file(lock_source)},
        "predictions": rows,
    }


def validate_prediction_receipt(
    receipt: Mapping[str, object], *, lock_path: str | Path,
    expected_predictions: Mapping[str, str | Path],
) -> None:
    if (
        receipt.get("schema") != PREDICTION_RECEIPT_SCHEMA
        or receipt.get("purpose") != "sealed_predictions_before_scoring"
        or receipt.get("human_blinding_claimed") is not False
    ):
        raise GeneralizationValidationError("invalid prediction receipt header")
    lock = receipt.get("protocol_lock")
    predictions = receipt.get("predictions")
    resolved_lock = str(Path(lock_path).resolve())
    if (
        not isinstance(lock, Mapping)
        or lock.get("path") != resolved_lock
        or lock.get("sha256") != sha256_file(resolved_lock)
        or not isinstance(predictions, Mapping)
        or set(predictions) != set(expected_predictions)
    ):
        raise GeneralizationValidationError("prediction receipt does not bind this analysis")
    for role, source in expected_predictions.items():
        row = predictions[role]
        resolved = str(Path(source).resolve())
        if (
            not isinstance(row, Mapping)
            or row.get("path") != resolved
            or row.get("sha256") != sha256_file(resolved)
            or row.get("size_bytes") != Path(resolved).stat().st_size
        ):
            raise GeneralizationValidationError(f"sealed prediction changed: {role}")


def _load_json(path: str | Path) -> object:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GeneralizationValidationError(f"cannot load JSON {path}: {exc}") from exc


def _unit_axes(value: object, *, exact_three: bool, name: str) -> np.ndarray:
    try:
        axes = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise GeneralizationValidationError(f"{name} axes are not numeric") from exc
    expected = (3, 3) if exact_three else None
    if (exact_three and axes.shape != expected) or (
        not exact_three and (axes.ndim != 2 or axes.shape[1:] != (3,) or axes.shape[0] < 1)
    ):
        requirement = "exactly [3,3]" if exact_three else "[N,3] with N>=1"
        raise GeneralizationValidationError(f"{name} axes must have shape {requirement}")
    if not np.all(np.isfinite(axes)):
        raise GeneralizationValidationError(f"{name} axes are non-finite")
    norms = np.linalg.norm(axes, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-12):
        raise GeneralizationValidationError(f"{name} contains an invalid axis")
    return axes / norms[:, None]


def _rows(document: object, *, role: str) -> list[Mapping[str, object]]:
    raw = document.get("objects") if isinstance(document, Mapping) else document
    if not isinstance(raw, list) or not raw:
        raise GeneralizationValidationError(f"{role} must contain a nonempty objects list")
    if any(not isinstance(row, Mapping) for row in raw):
        raise GeneralizationValidationError(f"{role} contains a non-object row")
    ids = [row.get("object_id") for row in raw]
    if any(not isinstance(value, str) or not value for value in ids):
        raise GeneralizationValidationError(f"{role} object_id values are invalid")
    if len(ids) != len(set(ids)):
        raise GeneralizationValidationError(f"{role} contains duplicate object_id values")
    if any(not _canonical_syntax(str(value)) for value in ids):
        raise GeneralizationValidationError(
            f"{role} object_id values require explicit damit:, mpc:, or provisional: namespace"
        )
    return sorted(raw, key=lambda row: str(row["object_id"]))


def _reference_axes(row: Mapping[str, object]) -> np.ndarray:
    raw = row.get("references", row.get("reference_axes"))
    if raw is None and isinstance(row.get("solutions"), list):
        raw = [solution.get("vector") for solution in row["solutions"]]
    if isinstance(raw, list) and raw and isinstance(raw[0], Mapping):
        raw = [item.get("axis", item.get("vector")) for item in raw]
    return _unit_axes(raw, exact_three=False, name=f"{row.get('object_id')} reference")


def _prediction_lookup(document: object, *, role: str) -> dict[str, Mapping[str, object]]:
    return {str(row["object_id"]): row for row in _rows(document, role=role)}


def _score_prediction(
    row: Mapping[str, object] | None, references: np.ndarray, *, role: str
) -> tuple[float, str | None]:
    if row is None:
        return FAILURE_ERROR_DEG, f"missing_{role}_prediction"
    if row.get("status", "ok") != "ok":
        reason = row.get("failure_reason", row.get("reason", f"{role}_status_not_ok"))
        return FAILURE_ERROR_DEG, str(reason)
    try:
        axes = _unit_axes(row.get("axes", row.get("predictions")), exact_three=True, name=role)
        errors = axial_angular_error_deg(axes[:, None, :], references[None, :, :])
        return float(np.min(errors)), None
    except (GeneralizationValidationError, ValueError, TypeError) as exc:
        return FAILURE_ERROR_DEG, f"invalid_{role}_prediction:{exc}"


def wilson_interval(successes: int, total: int, *, confidence: float = 0.95) -> dict[str, object]:
    if isinstance(successes, bool) or isinstance(total, bool) or total <= 0 or not 0 <= successes <= total:
        raise GeneralizationValidationError("Wilson counts are invalid")
    if not 0.0 < confidence < 1.0:
        raise GeneralizationValidationError("Wilson confidence must lie in (0,1)")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total**2)) / denominator
    return {
        "successes": successes, "n_objects": total, "proportion": proportion,
        "ci95_low": max(0.0, center - half), "ci95_high": min(1.0, center + half),
    }


def _summary(errors: np.ndarray) -> dict[str, object]:
    if errors.ndim != 1 or errors.size == 0 or not np.all(np.isfinite(errors)):
        raise GeneralizationValidationError("error vector is invalid")
    return {
        "n_objects": int(errors.size),
        "mean_oracle_at_3_error_deg": float(np.mean(errors)),
        "median_oracle_at_3_error_deg": float(np.median(errors)),
        "within_20_deg": wilson_interval(int(np.sum(errors <= 20.0)), int(errors.size)),
        "within_30_deg": wilson_interval(int(np.sum(errors <= 30.0)), int(errors.size)),
    }


def _bootstrap_mean(values: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    # Keep the temporary index matrix near 16 MiB even if a future public
    # census is much larger.
    chunk = max(1, min(512, 2_000_000 // values.size, BOOTSTRAP_RESAMPLES))
    for start in range(0, BOOTSTRAP_RESAMPLES, chunk):
        stop = min(BOOTSTRAP_RESAMPLES, start + chunk)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        samples[start:stop] = np.mean(values[indices], axis=1)
    return samples


def _identity_bootstrap_interval(values: np.ndarray) -> dict[str, float | int | str]:
    samples = _bootstrap_mean(values)
    return {
        "method": "asteroid_identity_nonparametric_bootstrap",
        "confidence": 0.95,
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "low_deg": float(np.quantile(samples, 0.025)),
        "high_deg": float(np.quantile(samples, 0.975)),
    }


def _paired_bootstrap(improvements: np.ndarray) -> dict[str, float | int]:
    samples = _bootstrap_mean(improvements)
    return {
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "mean_improvement_deg": float(np.mean(improvements)),
        "ci95_low_deg": float(np.quantile(samples, 0.025)),
        "ci95_high_deg": float(np.quantile(samples, 0.975)),
    }


def _uniform_random_three(
    references: Sequence[np.ndarray], *, resamples: int, seed: int
) -> dict[str, object]:
    if resamples <= 0:
        raise GeneralizationValidationError("uniform resamples must be positive")
    rng = np.random.default_rng(seed)
    error_sums = np.zeros(resamples, dtype=np.float64)
    within20_sum = 0
    within30_sum = 0
    # Only O(resamples) state is retained; each identity still receives its own
    # independently drawn fixed-three candidate set in every Monte Carlo run.
    chunk = max(1, min(2048, resamples))
    for target in references:
        for start in range(0, resamples, chunk):
            stop = min(resamples, start + chunk)
            draws = rng.normal(size=(stop - start, 3, 3))
            draws /= np.linalg.norm(draws, axis=2, keepdims=True)
            dot = np.abs(np.einsum("rkc,sc->rks", draws, target))
            best = np.max(np.clip(dot, 0.0, 1.0), axis=(1, 2))
            errors = np.degrees(np.arccos(best))
            error_sums[start:stop] += errors
            within20_sum += int(np.sum(errors <= 20.0))
            within30_sum += int(np.sum(errors <= 30.0))
    means = error_sums / len(references)
    return {
        "method": "three_independent_uniform_sphere_axes_per_identity",
        "multiple_declared_references_used": True,
        "resamples": resamples,
        "seed": seed,
        "mean_oracle_at_3_error_deg": float(np.mean(means)),
        "randomization_interval95_low_deg": float(np.quantile(means, 0.025)),
        "randomization_interval95_high_deg": float(np.quantile(means, 0.975)),
        "randomization_interval_interpretation": (
            "central_95_percent_of_trial_mean_errors_not_monte_carlo_uncertainty"
        ),
        "expected_fraction_within_20_deg": within20_sum / (resamples * len(references)),
        "expected_fraction_within_30_deg": within30_sum / (resamples * len(references)),
    }


def _family_sensitivity(
    errors: np.ndarray, families: Sequence[str | None]
) -> dict[str, object]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, family in enumerate(families):
        if isinstance(family, str) and family and family.lower() != "unknown":
            groups[family].append(index)
    if len(groups) < MIN_FAMILY_CLUSTERS:
        return {
            "reported": False, "minimum_clusters": MIN_FAMILY_CLUSTERS,
            "observed_clusters": len(groups), "reason": "too_few_known_family_clusters",
        }
    names = sorted(groups)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for repeat in range(BOOTSTRAP_RESAMPLES):
        selected = rng.integers(0, len(names), size=len(names))
        indices = [item for group_index in selected for item in groups[names[group_index]]]
        samples[repeat] = float(np.mean(errors[indices]))
    known_indices = [item for name in names for item in groups[name]]
    return {
        "reported": True, "n_clusters": len(names), "n_objects": len(known_indices),
        "estimand": "identity_weighted_known_family_objects",
        "mean_oracle_at_3_error_deg": float(np.mean(errors[known_indices])),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
        "ci95_low_deg": float(np.quantile(samples, 0.025)),
        "ci95_high_deg": float(np.quantile(samples, 0.975)),
    }


def _source_stratum(row: Mapping[str, object]) -> str:
    source = row.get("source_id", row.get("survey_id", row.get("source")))
    if source is None:
        return "unknown"
    if not isinstance(source, str) or not source.strip():
        raise GeneralizationValidationError(
            f"{row.get('object_id')} source stratum must be a nonempty string"
        )
    return source.strip()


def analyze_generalization(
    *,
    cohort: object,
    candidate_predictions: object,
    atlas_predictions: object,
    uniform_resamples: int = BOOTSTRAP_RESAMPLES,
    uniform_seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Score fixed-three candidate and train-only atlas predictions by identity."""
    cohort_rows = _rows(cohort, role="cohort")
    candidate = _prediction_lookup(candidate_predictions, role="candidate predictions")
    if not isinstance(atlas_predictions, Mapping) or atlas_predictions.get("train_only") is not True:
        raise GeneralizationValidationError("atlas predictions must declare train_only=true")
    eligible = [row for row in cohort_rows if row.get("eligible", True) is True]
    if not eligible:
        raise GeneralizationValidationError("cohort has no eligible identities")
    eligible_ids = {str(row["object_id"]) for row in eligible}
    atlas_kind: str
    if "axes" in atlas_predictions:
        fixed_atlas = _unit_axes(
            atlas_predictions.get("axes"), exact_three=True, name="fixed train-only atlas"
        )
        source_ids = atlas_predictions.get("source_identity_ids", [])
        if not isinstance(source_ids, list) or any(
            not isinstance(value, str) or not _canonical_syntax(value) for value in source_ids
        ):
            raise GeneralizationValidationError(
                "atlas source_identity_ids must be explicitly namespaced strings"
            )
        overlap = eligible_ids.intersection(source_ids)
        if overlap:
            raise GeneralizationValidationError(
                f"train-only atlas includes evaluation identities: {sorted(overlap)}"
            )
        atlas = {
            object_id: {"object_id": object_id, "status": "ok", "axes": fixed_atlas}
            for object_id in eligible_ids
        }
        atlas_kind = "one_fixed_train_only_three_axis_atlas"
    else:
        atlas = _prediction_lookup(atlas_predictions, role="atlas predictions")
        atlas_kind = "predeclared_identity_aligned_train_only_atlas"
    extra_candidate = sorted(set(candidate) - eligible_ids)
    extra_atlas = sorted(set(atlas) - eligible_ids)
    if extra_candidate or extra_atlas:
        raise GeneralizationValidationError(
            f"prediction identities outside eligible cohort: candidate={extra_candidate}, atlas={extra_atlas}"
        )
    missing_atlas = sorted(eligible_ids - set(atlas))
    if missing_atlas:
        raise GeneralizationValidationError(
            f"train-only atlas is missing eligible identities: {missing_atlas}"
        )
    object_rows: list[dict[str, object]] = []
    reference_arrays: list[np.ndarray] = []
    families: list[str | None] = []
    for row in eligible:
        object_id = str(row["object_id"])
        references = _reference_axes(row)
        error, failure = _score_prediction(candidate.get(object_id), references, role="candidate")
        atlas_error, atlas_failure = _score_prediction(atlas.get(object_id), references, role="atlas")
        if atlas_failure is not None:
            raise GeneralizationValidationError(
                f"train-only atlas baseline is invalid for {object_id}: {atlas_failure}"
            )
        family = row.get("family_id")
        families.append(str(family) if isinstance(family, str) and family else None)
        source_stratum = _source_stratum(row)
        reference_arrays.append(references)
        object_rows.append(
            {
                "object_id": object_id,
                "source_stratum": source_stratum,
                "reference_count": int(references.shape[0]),
                "family_id": families[-1],
                "oracle_at_3_error_deg": error,
                "atlas_oracle_at_3_error_deg": atlas_error,
                "prediction_failed": failure is not None,
                "prediction_failure_reason": failure,
                "atlas_failed": atlas_failure is not None,
                "atlas_failure_reason": atlas_failure,
            }
        )
    errors = np.asarray([row["oracle_at_3_error_deg"] for row in object_rows], dtype=np.float64)
    atlas_errors = np.asarray(
        [row["atlas_oracle_at_3_error_deg"] for row in object_rows], dtype=np.float64
    )
    strata: dict[str, object] = {}
    for count in sorted({int(row["reference_count"]) for row in object_rows}):
        selected = np.asarray(
            [float(row["oracle_at_3_error_deg"]) for row in object_rows if row["reference_count"] == count]
        )
        strata[str(count)] = _summary(selected)
    source_strata: dict[str, object] = {}
    for source in sorted({str(row["source_stratum"]) for row in object_rows}):
        selected = np.asarray(
            [
                float(row["oracle_at_3_error_deg"])
                for row in object_rows
                if row["source_stratum"] == source
            ]
        )
        source_strata[source] = _summary(selected)
    failure_counts = Counter(
        str(row["prediction_failure_reason"])
        for row in object_rows if row["prediction_failure_reason"] is not None
    )
    secondary = _summary(errors)
    secondary.pop("mean_oracle_at_3_error_deg")
    result = {
        "schema": GENERALIZATION_ANALYSIS_SCHEMA,
        "unit_of_analysis": "asteroid_identity",
        "candidate_count": 3,
        "axis_equivalence": "antipodal",
        "eligible_denominator": len(object_rows),
        "failure_error_deg": FAILURE_ERROR_DEG,
        "primary": {
            "name": "mean_antipodal_oracle_at_3_error_deg_all_declared_references",
            "value": float(np.mean(errors)),
            "identity_bootstrap_ci95": _identity_bootstrap_interval(errors),
        },
        "secondary": secondary,
        "failures": {
            "n_objects": int(sum(failure_counts.values())),
            "reasons": dict(sorted(failure_counts.items())),
        },
        "paired_train_only_atlas": _paired_bootstrap(atlas_errors - errors),
        "uniform_random_three": _uniform_random_three(
            reference_arrays, resamples=uniform_resamples, seed=uniform_seed
        ),
        "reference_count_strata": strata,
        "source_strata": source_strata,
        "family_cluster_sensitivity": _family_sensitivity(errors, families),
        "objects": object_rows,
    }
    result["paired_train_only_atlas"]["atlas_kind"] = atlas_kind
    return result


def locked_resource_contains(lock: Mapping[str, object], path: str | Path) -> bool:
    """Return whether an exact file/tree path is present in data/reference lock groups."""
    resolved_path = Path(path).resolve()
    resources = lock.get("resources")
    if not isinstance(resources, Mapping):
        return False
    for role in ("data", "reference"):
        for row in resources.get(role, []):
            if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
                continue
            locked_path = Path(str(row["path"])).resolve()
            if resolved_path == locked_path:
                return True
            if row.get("kind") == "directory" and resolved_path.is_relative_to(locked_path):
                return True
    return False


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "FAILURE_ERROR_DEG",
    "GeneralizationValidationError",
    "analyze_generalization",
    "audit_identity_exposure",
    "canonical_json",
    "classify_reference_photometry_overlap",
    "classify_reference_records",
    "create_prediction_receipt",
    "create_protocol_lock",
    "locked_resource_contains",
    "sha256_file",
    "sha256_path",
    "validate_prediction_receipt",
    "validate_external_exposure_clearance",
    "validate_protocol_lock",
    "wilson_interval",
]
