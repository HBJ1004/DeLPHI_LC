"""Deterministic, label-separated provenance for K3 external follow-up inputs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

import torch

from .publication_run import K3_OOF_SEEDS
from .ztf_external import (
    FINK_INGEST_MANIFEST_SCHEMA,
    HORIZONS_IDENTITY_AUDIT_SCHEMA,
    HORIZONS_MANIFEST_SCHEMA,
    HorizonsCache,
    ZTFExternalError,
    prepared_ztf_object,
)

ZTF_PERIOD_MANIFEST_SCHEMA = "delphi.k3-ztf-period-manifest.v1"
EXTERNAL_MODEL_MANIFEST_SCHEMA = "delphi.k3-external-model-manifest.v1"
ZTF_PREPARED_MANIFEST_SCHEMA = "delphi.k3-ztf-prepared-directory-manifest.v1"
_REAL_MODEL = re.compile(r"^models/real-fold-([0-4])-seed-(17|42|137|777|2027)\.pt$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash(value: object) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _load_json(path: Path, role: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ZTFExternalError(f"cannot load {role}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ZTFExternalError(f"{role} must be a JSON object")
    return value


def _write_new_json(path: Path, value: object) -> None:
    if path.exists():
        raise ZTFExternalError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_ztf_period_manifest(
    *,
    normalized_manifest_path: str | Path,
    catalog_path: str | Path,
    study_spec_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Extract only externally supplied periods from the frozen labeled catalog."""
    normalized_path = Path(normalized_manifest_path)
    source_catalog = Path(catalog_path)
    spec_path = Path(study_spec_path)
    destination = Path(output_path)
    normalized = _load_json(normalized_path, "normalized Fink manifest")
    if normalized.get("schema") != FINK_INGEST_MANIFEST_SCHEMA:
        raise ZTFExternalError("normalized Fink manifest schema mismatch")
    ready = {
        str(row["object_id"]): row
        for row in normalized.get("objects", [])
        if isinstance(row, Mapping) and row.get("status") == "ready"
    }
    if len(ready) != 169:
        raise ZTFExternalError("period extraction requires exactly 169 ready Fink identities")
    catalog: dict[str, tuple[Mapping[str, object], str]] = {}
    try:
        lines = source_catalog.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if not line.strip():
                continue
            record = json.loads(line)
            object_id = record["object_id"]
            if object_id in catalog:
                raise ZTFExternalError(f"duplicate frozen catalog identity: {object_id}")
            catalog[str(object_id)] = (record, hashlib.sha256(line.encode()).hexdigest())
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ZTFExternalError(f"cannot load frozen catalog: {exc}") from exc
    objects: list[dict[str, object]] = []
    for object_id in sorted(ready, key=lambda value: int(value.removeprefix("asteroid_"))):
        if object_id not in catalog:
            raise ZTFExternalError(f"ready Fink identity absent from frozen catalog: {object_id}")
        record, line_hash = catalog[object_id]
        solutions = record.get("solutions")
        if record.get("eligible") is not True or not isinstance(solutions, list) or not solutions:
            raise ZTFExternalError(f"frozen catalog object has no eligible period: {object_id}")
        chosen = solutions[0]
        if not isinstance(chosen, Mapping):
            raise ZTFExternalError(f"invalid first frozen catalog solution: {object_id}")
        try:
            period = float(chosen["period_hours"])
            model_id = int(chosen["model_id"])
            spin_hash = str(chosen["source_sha256"])
            fold = int(ready[object_id]["fold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ZTFExternalError(f"invalid period provenance for {object_id}: {exc}") from exc
        if (
            not math.isfinite(period)
            or period <= 0
            or model_id <= 0
            or not re.fullmatch(r"[0-9a-f]{64}", spin_hash)
            or fold not in range(5)
        ):
            raise ZTFExternalError(f"invalid period provenance for {object_id}")
        objects.append(
            {
                "object_id": object_id,
                "held_out_fold": fold,
                "known_period_hours": period,
                "period_provenance": {
                    "externally_supplied": True,
                    "source": "frozen DAMIT catalog",
                    "selection_policy": "first_frozen_catalog_solution_in_catalog_order",
                    "source_model_id": model_id,
                    "source_spin_sha256": spin_hash,
                    "source_catalog_record_sha256": line_hash,
                },
            }
        )
    result: dict[str, object] = {
        "schema": ZTF_PERIOD_MANIFEST_SCHEMA,
        "study_spec_sha256": _sha256_file(spec_path),
        "normalized_manifest_sha256": _sha256_file(normalized_path),
        "source_catalog_sha256": _sha256_file(source_catalog),
        "period_role": "externally_supplied_fixed_period",
        "pole_coordinates_exported": False,
        "prediction_preparation_requires_source_catalog": False,
        "object_count": len(objects),
        "objects": objects,
    }
    serialized = json.dumps(result, sort_keys=True).lower()
    forbidden = ('"solutions"', '"lambda_deg"', '"beta_deg"', '"vector"', '"axis')
    if any(token in serialized for token in forbidden):
        raise ZTFExternalError("pole-bearing field leaked into ZTF period manifest")
    _write_new_json(destination, result)
    return result


def build_external_model_manifest(
    *,
    artifact_root: str | Path,
    archive_index_path: str | Path,
    publication_package_manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Verify and bind the exact 25 publication checkpoints before prediction.

    Checkpoints are deserialized only after their bytes match both the archive
    index and checkpoint audit from the trusted publication package.
    """
    root = Path(artifact_root)
    index_path = Path(archive_index_path)
    package_path = Path(publication_package_manifest_path)
    destination = Path(output_path)
    index = _load_json(index_path, "publication archive index")
    package = _load_json(package_path, "publication package manifest")
    audit_path = root / "audits" / "checkpoint-audit.json"
    audit = _load_json(audit_path, "publication checkpoint audit")
    if index.get("schema") != "delphi.k3-publication-archive-index.v1":
        raise ZTFExternalError("publication archive index schema mismatch")
    if package.get("schema") != "delphi.k3-publication-package.v1":
        raise ZTFExternalError("publication package manifest schema mismatch")
    if audit.get("schema") != "delphi.k3-checkpoint-audit.v1" or audit.get("passed") is not True:
        raise ZTFExternalError("publication checkpoint audit did not pass")
    index_hash = _sha256_file(index_path)
    if package.get("archive_index_sha256") != index_hash:
        raise ZTFExternalError("publication package does not bind this archive index")
    analysis_commit = str(index.get("analysis_commit"))
    protocol_hash = str(index.get("protocol_sha256"))
    if (
        package.get("analysis_commit") != analysis_commit
        or package.get("protocol_sha256") != protocol_hash
        or audit.get("implementation_commit") != analysis_commit
        or audit.get("protocol_sha256") != protocol_hash
    ):
        raise ZTFExternalError("publication checkpoint provenance is inconsistent")
    if audit.get("seeds") != list(K3_OOF_SEEDS) or audit.get("real_oof_checkpoint_count") != 25:
        raise ZTFExternalError("publication checkpoint audit has the wrong model set")
    indexed_rows = index.get("files")
    if not isinstance(indexed_rows, list):
        raise ZTFExternalError("publication archive index has no files")
    by_path: dict[str, Mapping[str, object]] = {}
    for row in indexed_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("logical_path"), str):
            raise ZTFExternalError("publication archive index contains an invalid file row")
        logical_path = str(row["logical_path"])
        if logical_path in by_path:
            raise ZTFExternalError(f"duplicate archive-index path: {logical_path}")
        by_path[logical_path] = row
    audit_index_row = by_path.get("audits/checkpoint-audit.json")
    if (
        audit_index_row is None
        or audit_index_row.get("sha256") != _sha256_file(audit_path)
        or audit_index_row.get("size_bytes") != audit_path.stat().st_size
    ):
        raise ZTFExternalError("archive index does not bind the checkpoint audit")
    audit_hashes = audit.get("checkpoint_sha256")
    if not isinstance(audit_hashes, Mapping):
        raise ZTFExternalError("checkpoint audit has no hashes")
    expected_paths = {
        f"models/real-fold-{fold}-seed-{seed}.pt"
        for fold in range(5)
        for seed in K3_OOF_SEEDS
    }
    indexed_model_paths = {path for path in by_path if _REAL_MODEL.fullmatch(path)}
    if indexed_model_paths != expected_paths:
        raise ZTFExternalError("archive index does not contain the exact 25-model OOF set")
    models: list[dict[str, object]] = []
    model_config_hash: str | None = None
    for fold in range(5):
        for seed in K3_OOF_SEEDS:
            logical_path = f"models/real-fold-{fold}-seed-{seed}.pt"
            filename = Path(logical_path).name
            index_row = by_path[logical_path]
            checkpoint_path = root / logical_path
            checkpoint_hash = _sha256_file(checkpoint_path)
            if (
                index_row.get("sha256") != checkpoint_hash
                or audit_hashes.get(filename) != checkpoint_hash
                or index_row.get("size_bytes") != checkpoint_path.stat().st_size
            ):
                raise ZTFExternalError(f"checkpoint bytes are not publication-bound: {filename}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                config_seed = int(checkpoint["config"]["seed"])
                run_provenance = checkpoint["run_provenance"]
                current_model_hash = _hash(checkpoint["model_config"])
            except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
                raise ZTFExternalError(f"cannot validate checkpoint {filename}: {exc}") from exc
            synthetic_path = f"models/synthetic-seed-{seed}.pt"
            synthetic_row = by_path.get(synthetic_path)
            if not isinstance(run_provenance, Mapping) or synthetic_row is None:
                raise ZTFExternalError(f"checkpoint provenance is incomplete: {filename}")
            if (
                checkpoint.get("schema") != "delphi.k3-checkpoint.v2"
                or checkpoint.get("stage") != "real-oof"
                or config_seed != seed
                or run_provenance.get("implementation_commit") != analysis_commit
                or run_provenance.get("protocol_sha256") != protocol_hash
                or run_provenance.get("pretrained_checkpoint_sha256")
                != synthetic_row.get("sha256")
            ):
                raise ZTFExternalError(f"checkpoint metadata is inconsistent: {filename}")
            if model_config_hash is None:
                model_config_hash = current_model_hash
            elif current_model_hash != model_config_hash:
                raise ZTFExternalError("publication checkpoints have incompatible model configs")
            configuration_hash = str(checkpoint.get("configuration_sha256"))
            if not re.fullmatch(r"[0-9a-f]{64}", configuration_hash):
                raise ZTFExternalError(f"checkpoint configuration hash is invalid: {filename}")
            models.append(
                {
                    "filename": filename,
                    "logical_path": logical_path,
                    "fold": fold,
                    "seed": seed,
                    "sha256": checkpoint_hash,
                    "size_bytes": checkpoint_path.stat().st_size,
                    "checkpoint_schema": checkpoint["schema"],
                    "stage": checkpoint["stage"],
                    "configuration_sha256": configuration_hash,
                    "run_provenance": dict(run_provenance),
                }
            )
    artifact_archive = [
        row
        for row in package.get("files", [])
        if isinstance(row, Mapping) and row.get("logical_label") == "artifact-root"
    ]
    if len(artifact_archive) != 1:
        raise ZTFExternalError("publication package has no unique artifact-root archive")
    result: dict[str, object] = {
        "schema": EXTERNAL_MODEL_MANIFEST_SCHEMA,
        "publication_package_manifest_sha256": _sha256_file(package_path),
        "artifact_archive_sha256": artifact_archive[0]["sha256"],
        "archive_index_sha256": index_hash,
        "checkpoint_audit_sha256": _sha256_file(audit_path),
        "analysis_commit": analysis_commit,
        "protocol_sha256": protocol_hash,
        "model_config_sha256": model_config_hash,
        "folds": list(range(5)),
        "seeds": list(K3_OOF_SEEDS),
        "model_count": len(models),
        "models": models,
    }
    _write_new_json(destination, result)
    return result


def _safe_child(root: Path, relative: object, role: str) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ZTFExternalError(f"unsafe {role} path")
    value = (root / relative).resolve()
    try:
        value.relative_to(root.resolve())
    except ValueError as exc:
        raise ZTFExternalError(f"{role} path escapes its root") from exc
    return value


def prepare_ztf_followup_directory(
    *,
    normalized_manifest_path: str | Path,
    horizons_manifest_path: str | Path,
    horizons_identity_audit_path: str | Path,
    period_manifest_path: str | Path,
    study_spec_path: str | Path,
    output_directory: str | Path,
) -> dict[str, object]:
    """Bulk-prepare ZTF objects only after all 169 inputs pass identity audit."""
    normalized_path = Path(normalized_manifest_path)
    horizons_path = Path(horizons_manifest_path)
    audit_path = Path(horizons_identity_audit_path)
    periods_path = Path(period_manifest_path)
    spec_path = Path(study_spec_path)
    destination = Path(output_directory)
    if destination.exists():
        raise ZTFExternalError("ZTF prepared output directory must not already exist")
    normalized = _load_json(normalized_path, "normalized Fink manifest")
    horizons = _load_json(horizons_path, "Horizons directory manifest")
    audit = _load_json(audit_path, "Horizons identity audit")
    periods = _load_json(periods_path, "ZTF period manifest")
    if normalized.get("schema") != FINK_INGEST_MANIFEST_SCHEMA:
        raise ZTFExternalError("normalized Fink manifest schema mismatch")
    if horizons.get("schema") != HORIZONS_MANIFEST_SCHEMA:
        raise ZTFExternalError("Horizons directory manifest schema mismatch")
    if audit.get("schema") != HORIZONS_IDENTITY_AUDIT_SCHEMA:
        raise ZTFExternalError("Horizons identity audit schema mismatch")
    if periods.get("schema") != ZTF_PERIOD_MANIFEST_SCHEMA:
        raise ZTFExternalError("ZTF period manifest schema mismatch")
    normalized_hash = _sha256_file(normalized_path)
    horizons_hash = _sha256_file(horizons_path)
    if (
        horizons.get("normalized_manifest_sha256") != normalized_hash
        or audit.get("normalized_manifest_sha256") != normalized_hash
        or audit.get("horizons_manifest_sha256") != horizons_hash
        or periods.get("normalized_manifest_sha256") != normalized_hash
        or periods.get("study_spec_sha256") != _sha256_file(spec_path)
    ):
        raise ZTFExternalError("ZTF preparation manifests are not hash-bound to one another")
    normalized_rows = {
        str(row["object_id"]): row
        for row in normalized.get("objects", [])
        if isinstance(row, Mapping) and row.get("status") == "ready"
    }
    horizon_rows = {
        str(row["object_id"]): row
        for row in horizons.get("objects", [])
        if isinstance(row, Mapping)
    }
    audit_rows = {
        str(row["object_id"]): row
        for row in audit.get("objects", [])
        if isinstance(row, Mapping)
    }
    period_rows = {
        str(row["object_id"]): row
        for row in periods.get("objects", [])
        if isinstance(row, Mapping)
    }
    identities = set(normalized_rows)
    if (
        len(identities) != 169
        or set(horizon_rows) != identities
        or set(audit_rows) != identities
        or set(period_rows) != identities
        or horizons.get("object_count") != 169
        or audit.get("object_count") != 169
        or periods.get("object_count") != 169
    ):
        raise ZTFExternalError("ZTF preparation requires four complete 169-object manifests")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    prepared_rows: list[dict[str, object]] = []
    try:
        for object_id in sorted(identities, key=lambda value: int(value.removeprefix("asteroid_"))):
            normalized_row = normalized_rows[object_id]
            horizon_row = horizon_rows[object_id]
            audit_row = audit_rows[object_id]
            period_row = period_rows[object_id]
            normalized_object_path = _safe_child(
                normalized_path.parent,
                normalized_row["normalized_path"],
                "normalized Fink object",
            )
            if _sha256_file(normalized_object_path) != normalized_row["normalized_sha256"]:
                raise ZTFExternalError(f"normalized object hash mismatch: {object_id}")
            normalized_object = _load_json(normalized_object_path, "normalized Fink object")
            cache_path = _safe_child(
                horizons_path.parent, horizon_row["cache_path"], "Horizons cache"
            )
            cache_hash = _sha256_file(cache_path)
            if (
                cache_hash != horizon_row["cache_sha256"]
                or cache_hash != audit_row["cache_sha256"]
                or normalized_row["normalized_sha256"] != audit_row["normalized_sha256"]
            ):
                raise ZTFExternalError(f"ZTF source binding mismatch: {object_id}")
            cache = _load_json(cache_path, "Horizons cache")
            parsed_cache = HorizonsCache.from_mapping(cache)
            if len(parsed_cache.rows) != int(normalized_row["retained_row_count"]):
                raise ZTFExternalError(f"ZTF/Horizons row-count mismatch: {object_id}")
            if int(period_row["held_out_fold"]) != int(normalized_row["fold"]):
                raise ZTFExternalError(f"period/split fold mismatch: {object_id}")
            provenance = json.dumps(
                period_row["period_provenance"], sort_keys=True, separators=(",", ":")
            )
            prepared = prepared_ztf_object(
                object_id,
                normalized_object["rows"],
                cache,
                known_period_hours=float(period_row["known_period_hours"]),
                period_provenance=provenance,
            )
            prepared["held_out_fold"] = int(period_row["held_out_fold"])
            prepared["period_manifest_sha256"] = _sha256_file(periods_path)
            prepared["horizons_identity_audit_sha256"] = _sha256_file(audit_path)
            prepared["resolved_target_identity"] = audit_row["resolved_target_identity"]
            serialized = json.dumps(prepared, sort_keys=True).lower()
            if any(
                token in serialized
                for token in ('"solutions"', '"lambda_deg"', '"beta_deg"', '"pole_axis"')
            ):
                raise ZTFExternalError(f"pole label leaked into prepared object: {object_id}")
            output_path = staging / "objects" / f"{object_id}.json"
            _write_new_json(output_path, prepared)
            prepared_rows.append(
                {
                    "object_id": object_id,
                    "held_out_fold": int(period_row["held_out_fold"]),
                    "path": output_path.relative_to(staging).as_posix(),
                    "sha256": _sha256_file(output_path),
                    "observation_count": int(normalized_row["retained_row_count"]),
                }
            )
        result: dict[str, object] = {
            "schema": ZTF_PREPARED_MANIFEST_SCHEMA,
            "study_spec_sha256": _sha256_file(spec_path),
            "normalized_manifest_sha256": normalized_hash,
            "horizons_manifest_sha256": horizons_hash,
            "horizons_identity_audit_sha256": _sha256_file(audit_path),
            "period_manifest_sha256": _sha256_file(periods_path),
            "selection_uses_pole_labels": False,
            "earth_center_observer_approximation": True,
            "object_count": len(prepared_rows),
            "objects": prepared_rows,
        }
        _write_new_json(staging / "manifest.json", result)
        os.replace(staging, destination)
        return result
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "EXTERNAL_MODEL_MANIFEST_SCHEMA",
    "ZTF_PERIOD_MANIFEST_SCHEMA",
    "ZTF_PREPARED_MANIFEST_SCHEMA",
    "build_external_model_manifest",
    "build_ztf_period_manifest",
    "prepare_ztf_followup_directory",
]
