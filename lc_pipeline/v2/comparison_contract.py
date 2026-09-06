"""Fail-closed loader for the frozen axial V1/V2 comparison policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

AXIAL_COMPARISON_SCHEMA = "delphi.axial-comparison-spec.v1"
AXIAL_COMPARISON_SPEC_SHA256 = "ca7aeb442b1576dc92a729ae80926b8d2f49f9a74292ceece6db4ceddeabe49a"


class ComparisonContractError(ValueError):
    """Raised when policy bytes or referenced partitions differ from the lock."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_axial_comparison_spec(
    path: str | Path | None = None, *, verify_splits: bool = True
) -> dict[str, Any]:
    """Load only the byte-identical frozen comparison contract.

    A caller may pass a path for testability, but alternate policy bytes are
    rejected. Referenced split files are resolved relative to the repository
    root and checked before any development or confirmatory run starts.
    """
    source = (
        _repository_root() / "repro" / "axial_comparison_spec.yaml" if path is None else Path(path)
    )
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise ComparisonContractError(f"cannot read axial comparison spec: {exc}") from exc
    actual_hash = sha256_bytes(payload)
    if actual_hash != AXIAL_COMPARISON_SPEC_SHA256:
        raise ComparisonContractError(f"axial comparison spec hash mismatch: {actual_hash}")
    try:
        document = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ComparisonContractError(f"invalid axial comparison spec JSON: {exc}") from exc
    if document.get("schema") != AXIAL_COMPARISON_SCHEMA:
        raise ComparisonContractError("axial comparison spec schema mismatch")
    if verify_splits:
        root = _repository_root()
        for section_name in ("development", "confirmation"):
            section = document[section_name]
            split_path = root / section["split_file"]
            try:
                split_payload = split_path.read_bytes()
            except OSError as exc:
                raise ComparisonContractError(f"cannot read {section_name} split: {exc}") from exc
            if sha256_bytes(split_payload) != section["split_file_sha256"]:
                raise ComparisonContractError(f"{section_name} split hash mismatch")
    return document


__all__ = [
    "AXIAL_COMPARISON_SCHEMA",
    "AXIAL_COMPARISON_SPEC_SHA256",
    "ComparisonContractError",
    "load_axial_comparison_spec",
    "sha256_bytes",
]
