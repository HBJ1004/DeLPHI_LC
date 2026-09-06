"""Hash-bound prospective K3 protocol and leakage checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

K3_PROTOCOL_SHA256 = "84dee816d08a60b7780a6d1d400add16136fa96c0677e84929700cee8bf22a50"


class K3ProtocolError(ValueError):
    """Raised when the prospective scientific contract is invalid."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_path(document: dict[str, Any], *parts: str) -> Any:
    value: Any = document
    for part in parts:
        if not isinstance(value, dict) or part not in value:
            raise K3ProtocolError(f"protocol is missing {'.'.join(parts)}")
        value = value[part]
    return value


def validate_protocol(document: dict[str, Any]) -> None:
    if document.get("schema") != "delphi.k3-redesign-spec.v1":
        raise K3ProtocolError("K3 protocol schema mismatch")
    if document.get("status") != "locked_before_development_and_publication_execution":
        raise K3ProtocolError("K3 protocol is not prospectively locked")
    if _require_path(document, "data_boundaries", "retrospective_oof", "eligible_objects") != 170:
        raise K3ProtocolError("retrospective cohort must remain the frozen 170 objects")
    inference = _require_path(document, "inference")
    if inference.get("candidate_count") != 3:
        raise K3ProtocolError("K3 protocol must emit three candidates")
    if _require_path(document, "inference", "grid", "unique_axial_candidates") != 6144:
        raise K3ProtocolError("K3 protocol must use 6,144 unique axial grid points")
    if _require_path(document, "training", "oof_seeds") != [17, 42, 137, 777, 2027]:
        raise K3ProtocolError("OOF seed set differs from the locked protocol")
    gates = _require_path(document, "stage_gates", "retrospective_oof")
    if gates.get("mean_error_deg_maximum") != 22.0 or gates.get("minimum_point_improvement_deg") != 5.0:
        raise K3ProtocolError("primary OOF performance gates changed")
    if _require_path(document, "statistics", "unit") != "asteroid_object":
        raise K3ProtocolError("statistical unit must be the asteroid object")


def load_protocol(path: str | Path | None = None, *, verify_hash: bool = True) -> dict[str, Any]:
    source = Path(path) if path is not None else repository_root() / "repro" / "k3_redesign_spec.yaml"
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise K3ProtocolError(f"cannot read K3 protocol: {exc}") from exc
    if not isinstance(document, dict):
        raise K3ProtocolError("K3 protocol root must be an object")
    validate_protocol(document)
    if verify_hash and sha256_file(source) != K3_PROTOCOL_SHA256:
        raise K3ProtocolError("K3 protocol SHA-256 differs from the implementation lock")
    return document


def validate_disjoint_ids(**partitions: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """Validate named object/donor partitions and return canonical tuples."""
    normalized: dict[str, tuple[str, ...]] = {}
    owner: dict[str, str] = {}
    for name, identifiers in partitions.items():
        values = tuple(str(identifier) for identifier in identifiers)
        if not values or any(not value or value != value.strip() for value in values):
            raise K3ProtocolError(f"partition {name!r} must contain nonempty canonical IDs")
        if len(values) != len(set(values)):
            raise K3ProtocolError(f"partition {name!r} contains duplicate IDs")
        for identifier in values:
            if identifier in owner:
                raise K3ProtocolError(
                    f"identifier {identifier!r} overlaps partitions {owner[identifier]!r} and {name!r}"
                )
            owner[identifier] = name
        normalized[name] = tuple(sorted(values))
    return normalized


def validate_synthetic_donors(
    *,
    shape_train: Iterable[str],
    shape_validation: Iterable[str],
    shape_test: Iterable[str],
    geometry_train: Iterable[str],
    geometry_validation: Iterable[str],
    geometry_test: Iterable[str],
    retrospective_test_ids: Iterable[str],
    temporal_ids: Iterable[str],
) -> None:
    """Fail closed on shape/geometry split overlap and protected geometry donors."""
    validate_disjoint_ids(
        shape_train=shape_train,
        shape_validation=shape_validation,
        shape_test=shape_test,
    )
    geometry = validate_disjoint_ids(
        geometry_train=geometry_train,
        geometry_validation=geometry_validation,
        geometry_test=geometry_test,
    )
    protected = set(retrospective_test_ids) | set(temporal_ids)
    leaked = protected.intersection(identifier for values in geometry.values() for identifier in values)
    if leaked:
        raise K3ProtocolError(
            "protected evaluation objects used as synthetic geometry donors: " + ", ".join(sorted(leaked))
        )
