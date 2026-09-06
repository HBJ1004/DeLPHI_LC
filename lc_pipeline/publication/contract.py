"""Fail-closed loader and data validator for the definitive V1 publication run."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PUBLICATION_SPEC_SCHEMA = "delphi.v1-publication-spec.v1"
# Updated only when the preregistration file is intentionally replaced before
# publication execution. A byte change after results are opened is a new study.
PUBLICATION_SPEC_SHA256 = "7aaedb65bf5c61e17d954d43aec77b3be0ed10a90c4bda97705750af38ee285c"
EXPECTED_ELIGIBLE_OBJECTS = 170
EXPECTED_QUARANTINED_OBJECTS = 4
EXPECTED_FOLDS = 5
EXPECTED_ROLE_COUNTS = {
    "train_ids": 108,
    "validation_ids": 14,
    "calibration_ids": 14,
    "test_ids": 34,
}


class PublicationContractError(ValueError):
    """Raised before work starts when a frozen publication binding differs."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise PublicationContractError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PublicationContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_bytes(payload: bytes, description: str) -> Any:
    try:
        return json.loads(payload, object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicationContractError(f"invalid {description} JSON: {exc}") from exc


def _read(path: Path, description: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise PublicationContractError(f"cannot read {description}: {exc}") from exc


def _require_digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PublicationContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _verify_bound_file(root: Path, binding: Mapping[str, object], name: str) -> Path:
    logical_path = binding.get("path")
    if not isinstance(logical_path, str) or not logical_path or Path(logical_path).is_absolute():
        raise PublicationContractError(f"{name} path must be repository-relative")
    expected = _require_digest(binding.get("sha256"), f"{name}.sha256")
    path = root / logical_path
    if sha256_file(path) != expected:
        raise PublicationContractError(f"{name} hash mismatch")
    return path


def load_publication_spec(
    path: str | Path | None = None, *, verify_bindings: bool = True
) -> dict[str, Any]:
    """Load only the byte-identical publication policy and optionally its inputs."""
    source = repository_root() / "repro" / "v1_publication_spec.yaml" if path is None else Path(path)
    payload = _read(source, "V1 publication specification")
    actual = sha256_bytes(payload)
    if actual != PUBLICATION_SPEC_SHA256:
        raise PublicationContractError(f"V1 publication specification hash mismatch: {actual}")
    document = _load_json_bytes(payload, "V1 publication specification")
    if not isinstance(document, dict) or document.get("schema") != PUBLICATION_SPEC_SCHEMA:
        raise PublicationContractError("V1 publication specification schema mismatch")
    if document.get("status") != "locked_before_publication_outer_fold_execution":
        raise PublicationContractError("V1 publication specification is not locked")
    if verify_bindings:
        root = repository_root()
        bindings = document.get("bindings")
        if not isinstance(bindings, dict):
            raise PublicationContractError("publication bindings must be an object")
        for name in ("axial_comparison_spec", "catalog", "catalog_manifest", "publication_splits"):
            binding = bindings.get(name)
            if not isinstance(binding, dict):
                raise PublicationContractError(f"missing {name} binding")
            _verify_bound_file(root, binding, name)
    return document


@dataclass(frozen=True)
class PublicationInputs:
    """Validated identities shared by every publication phase."""

    catalog_path: Path
    splits_path: Path
    cache_manifest_path: Path
    eligible_object_ids: tuple[str, ...]
    quarantined_object_ids: tuple[str, ...]
    fold_test_ids: tuple[tuple[str, ...], ...]
    cache_manifest_sha256: str
    cache_configuration_sha256: str

    @property
    def object_count(self) -> int:
        return len(self.eligible_object_ids)


def _load_catalog(path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PublicationContractError(f"cannot read catalog: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        value = _load_json_bytes(line.encode("utf-8"), f"catalog line {line_number}")
        if not isinstance(value, dict):
            raise PublicationContractError(f"catalog line {line_number} is not an object")
        records.append(value)
    identifiers = [record.get("object_id") for record in records]
    if any(not isinstance(value, str) or not value for value in identifiers):
        raise PublicationContractError("catalog contains an invalid object_id")
    if len(set(identifiers)) != len(identifiers):
        raise PublicationContractError("catalog contains duplicate object IDs")
    eligible = tuple(sorted(record["object_id"] for record in records if record.get("eligible") is True))
    quarantined = tuple(
        sorted(record["object_id"] for record in records if record.get("eligible") is False)
    )
    if len(eligible) != EXPECTED_ELIGIBLE_OBJECTS or len(quarantined) != EXPECTED_QUARANTINED_OBJECTS:
        raise PublicationContractError(
            "catalog must contain exactly 170 eligible and four quarantined objects"
        )
    if len(records) != EXPECTED_ELIGIBLE_OBJECTS + EXPECTED_QUARANTINED_OBJECTS:
        raise PublicationContractError("catalog eligibility flag is missing or non-boolean")
    return eligible, quarantined


def _validate_splits(path: Path, eligible: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    value = _load_json_bytes(_read(path, "publication splits"), "publication splits")
    if not isinstance(value, dict) or value.get("n_outer_folds") != EXPECTED_FOLDS:
        raise PublicationContractError("publication split must contain five outer folds")
    audit = value.get("audit")
    if not isinstance(audit, dict) or audit.get("passed") is not True:
        raise PublicationContractError("publication split audit is not passing")
    folds = value.get("folds")
    if not isinstance(folds, list) or len(folds) != EXPECTED_FOLDS:
        raise PublicationContractError("publication split fold list is incomplete")
    expected = set(eligible)
    test_membership: list[str] = []
    fold_tests: list[tuple[str, ...]] = []
    for expected_fold, fold in enumerate(folds):
        if not isinstance(fold, dict) or fold.get("fold") != expected_fold:
            raise PublicationContractError("publication folds must be ordered 0 through 4")
        role_sets: dict[str, set[str]] = {}
        for role, count in EXPECTED_ROLE_COUNTS.items():
            identifiers = fold.get(role)
            if (
                not isinstance(identifiers, list)
                or len(identifiers) != count
                or any(not isinstance(item, str) or not item for item in identifiers)
                or len(set(identifiers)) != count
            ):
                raise PublicationContractError(f"fold {expected_fold} has invalid {role}")
            role_sets[role] = set(identifiers)
        union = set().union(*role_sets.values())
        if union != expected or sum(map(len, role_sets.values())) != len(expected):
            raise PublicationContractError(f"fold {expected_fold} roles overlap or omit objects")
        tests = tuple(sorted(role_sets["test_ids"]))
        test_membership.extend(tests)
        fold_tests.append(tests)
    counts = Counter(test_membership)
    if set(counts) != expected or any(count != 1 for count in counts.values()):
        raise PublicationContractError("each eligible object must be outer test exactly once")
    return tuple(fold_tests)


def _validate_cache(
    path: Path, spec: Mapping[str, Any], eligible: Sequence[str], *, verify_files: bool
) -> tuple[str, str]:
    binding = spec["bindings"]["tensor_cache"]
    expected_manifest = _require_digest(
        binding.get("manifest_sha256"), "tensor_cache.manifest_sha256"
    )
    actual_manifest = sha256_file(path)
    if actual_manifest != expected_manifest:
        raise PublicationContractError("tensor cache manifest hash mismatch")
    value = _load_json_bytes(_read(path, "tensor cache manifest"), "tensor cache manifest")
    if not isinstance(value, dict) or value.get("schema") != "delphi.tensor-cache-manifest.v2":
        raise PublicationContractError("tensor cache schema mismatch")
    configuration = _require_digest(
        value.get("configuration_sha256"), "cache.configuration_sha256"
    )
    if configuration != binding.get("configuration_sha256"):
        raise PublicationContractError("tensor cache configuration hash mismatch")
    if value.get("catalog_sha256") != spec["bindings"]["catalog"]["sha256"]:
        raise PublicationContractError("tensor cache catalog binding mismatch")
    entries = value.get("entries")
    if not isinstance(entries, list) or len(entries) != EXPECTED_ELIGIBLE_OBJECTS:
        raise PublicationContractError("tensor cache must contain 170 entries")
    entry_ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("object_id"), str):
            raise PublicationContractError("tensor cache contains an invalid entry")
        object_id = entry["object_id"]
        if object_id in entry_ids:
            raise PublicationContractError("tensor cache contains duplicate object IDs")
        entry_ids.add(object_id)
        expected_digest = _require_digest(entry.get("sha256"), f"cache entry {object_id}")
        logical_path = entry.get("logical_path")
        if not isinstance(logical_path, str) or Path(logical_path).name != logical_path:
            raise PublicationContractError("cache logical paths must be local filenames")
        if verify_files:
            artifact = path.parent / logical_path
            if sha256_file(artifact) != expected_digest:
                raise PublicationContractError(f"cache artifact hash mismatch: {object_id}")
    if entry_ids != set(eligible):
        raise PublicationContractError("catalog and tensor cache object sets differ")
    return actual_manifest, configuration


def validate_publication_inputs(
    *,
    catalog_path: str | Path | None = None,
    splits_path: str | Path | None = None,
    cache_manifest_path: str | Path | None = None,
    verify_cache_files: bool = False,
) -> PublicationInputs:
    """Validate the exact 170-object catalog/split/cache publication universe."""
    spec = load_publication_spec()
    root = repository_root()
    catalog = Path(catalog_path) if catalog_path is not None else root / spec["bindings"]["catalog"]["path"]
    splits = Path(splits_path) if splits_path is not None else root / spec["bindings"]["publication_splits"]["path"]
    cache = (
        Path(cache_manifest_path)
        if cache_manifest_path is not None
        else root / spec["bindings"]["tensor_cache"]["default_path"]
    )
    if sha256_file(catalog) != spec["bindings"]["catalog"]["sha256"]:
        raise PublicationContractError("catalog hash mismatch")
    if sha256_file(splits) != spec["bindings"]["publication_splits"]["sha256"]:
        raise PublicationContractError("publication split hash mismatch")
    eligible, quarantined = _load_catalog(catalog)
    fold_tests = _validate_splits(splits, eligible)
    cache_hash, configuration_hash = _validate_cache(
        cache, spec, eligible, verify_files=verify_cache_files
    )
    return PublicationInputs(
        catalog_path=catalog.resolve(),
        splits_path=splits.resolve(),
        cache_manifest_path=cache.resolve(),
        eligible_object_ids=eligible,
        quarantined_object_ids=quarantined,
        fold_test_ids=fold_tests,
        cache_manifest_sha256=cache_hash,
        cache_configuration_sha256=configuration_hash,
    )


__all__ = [
    "EXPECTED_ELIGIBLE_OBJECTS",
    "PUBLICATION_SPEC_SCHEMA",
    "PUBLICATION_SPEC_SHA256",
    "PublicationContractError",
    "PublicationInputs",
    "load_publication_spec",
    "repository_root",
    "sha256_bytes",
    "sha256_file",
    "validate_publication_inputs",
]
