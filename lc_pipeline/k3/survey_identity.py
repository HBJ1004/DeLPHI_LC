"""Explicit joins between DAMIT database IDs and MPC numbered designations."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

from .generalization_diagnostics import sha256, write_new_json

IDENTITY_MAP_SCHEMA = "delphi.k3-survey-identity-map.v1"
BINDING_SCHEMA = "delphi.k3-survey-identity-binding.v1"


def build_identity_map(table_path: Path, splits_path: Path) -> dict:
    """Read identity metadata only; the table has no model pole columns."""
    with table_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"id", "number", "name", "designation"}.issubset(reader.fieldnames or ()):
            raise ValueError("official asteroid table is missing identity columns")
        by_id, numbered = {}, {}
        for row in reader:
            internal = int(row["id"])
            if internal in by_id:
                raise ValueError("duplicate DAMIT database ID")
            by_id[internal] = row
            number = row["number"].strip()
            if number:
                number = int(number)
                if number <= 0 or number in numbered:
                    raise ValueError("ambiguous MPC numbered designation")
                numbered[number] = internal
    splits = json.loads(splits_path.read_text())
    objects, seen = [], set()
    for fold in sorted(splits["folds"], key=lambda row: row["fold"]):
        for object_id in fold["test_ids"]:
            if not re.fullmatch(r"asteroid_[1-9][0-9]*", object_id) or object_id in seen:
                raise ValueError("invalid or duplicate frozen DAMIT object ID")
            seen.add(object_id)
            internal = int(object_id.removeprefix("asteroid_"))
            if internal not in by_id:
                raise ValueError(f"DAMIT ID absent from official identity table: {object_id}")
            row = by_id[internal]
            objects.append({"object_id": object_id, "damit_id": internal,
                            "mpc_number": int(row["number"]) if row["number"].strip() else None,
                            "name": row["name"], "designation": row["designation"],
                            "held_out_fold": int(fold["fold"]),
                            "damit_identity": f"damit:{internal}",
                            "physical_identity": f'mpc:{int(row["number"])}' if row["number"].strip() else None})
    return {"schema": IDENTITY_MAP_SCHEMA, "identity_table_sha256": sha256(table_path),
            "splits_sha256": sha256(splits_path), "objects": sorted(objects, key=lambda row: row["damit_id"]),
            "object_count": len(objects), "mapping_authority": "official_DAMIT_asteroids_id_to_number",
            "unmapped_number_count": sum(row["mpc_number"] is None for row in objects),
            "source_table": str(table_path), "source_splits": str(splits_path),
            "source_object_id_namespace": "damit_database_id_not_mpc_number"}


def validate_survey_binding(value: dict) -> None:
    """Reject old ZTF inputs even if their internally consistent wrong target resolved."""
    binding = value.get("identity_binding")
    if not isinstance(binding, dict) or binding.get("schema") != BINDING_SCHEMA:
        raise ValueError("ZTF input lacks official DAMIT-to-MPC identity binding; legacy suffix-based inputs are invalid")
    object_id = value["object_id"]
    if not re.fullmatch(r"asteroid_[1-9][0-9]*", object_id):
        raise ValueError("expected internal DAMIT object ID")
    if (binding.get("object_id") != object_id
            or binding.get("damit_id") != int(object_id.removeprefix("asteroid_"))):
        raise ValueError("binding does not match internal DAMIT identity")
    number = binding.get("mpc_number")
    if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
        raise ValueError("binding requires explicit positive MPC number")
    if binding.get("resolved_mpc_number") != number:
        raise ValueError("resolved survey/Horizons target differs from mapped MPC number")
    target = value.get("resolved_target_identity", {}).get("asteroid_number")
    if target is not None and target != number:
        raise ValueError("prepared target identity disagrees with mapped MPC number")
    for key in ("identity_map_sha256", "identity_table_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(binding.get(key, ""))):
            raise ValueError("identity binding is not hash-bound")
    fold = binding.get("held_out_fold")
    if isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(5):
        raise ValueError("identity binding requires an explicit valid held-out fold")
    if value.get("held_out_fold") != fold:
        raise ValueError("identity fold binding differs from prepared fold")


def audit_prepared_identities(identity_map: dict, prepared_manifest_path: Path) -> dict:
    expected = {row["object_id"]: row for row in identity_map["objects"]}
    manifest = json.loads(prepared_manifest_path.read_text())
    rows = []
    for record in manifest["objects"]:
        path = prepared_manifest_path.parent / record["path"]
        if sha256(path) != record["sha256"]:
            raise ValueError("original prepared-input hash mismatch")
        value = json.loads(path.read_text())
        object_id = record["object_id"]
        physical = value.get("resolved_target_identity", {}).get("asteroid_number")
        correct = expected[object_id]["mpc_number"]
        rows.append({"object_id": object_id, "damit_id": expected[object_id]["damit_id"],
                     "expected_mpc_number": correct, "expected_name": expected[object_id]["name"],
                     "observed_mpc_number": physical,
                     "observed_name": value.get("resolved_target_identity", {}).get("primary_name"),
                     "identity_matches": correct is not None and physical == correct,
                     "prepared_sha256": record["sha256"]})
    mismatches = sum(not row["identity_matches"] for row in rows)
    return {"schema": "delphi.k3-survey-identity-audit.v1", "identity_table_sha256": identity_map["identity_table_sha256"],
            "original_prepared_manifest_sha256": sha256(prepared_manifest_path), "objects": rows,
            "evaluated_count": len(rows), "mismatch_count": mismatches,
            "verdict": "invalid_cross_survey_comparison" if mismatches else "identities_match",
            "interpretation": "Mismatch scores are not evidence for or against generalization. Original artifacts retained.",
            "reference_pole_values_opened": False}


def write_identity_audit(table: Path, splits: Path, prepared: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("identity audit output directory must be new")
    mapping = build_identity_map(table, splits)
    audit = audit_prepared_identities(mapping, prepared)
    write_new_json(output / "identity-map.json", mapping)
    audit["identity_map_sha256"] = sha256(output / "identity-map.json")
    write_new_json(output / "legacy-ztf-identity-audit.json", audit)
    return audit


def export_mapped_periods(identity_map_path: Path, catalog_path: Path, output: Path) -> dict:
    """Export externally supplied periods for all mapped original objects, not only cached ZTF IDs."""
    mapping = json.loads(identity_map_path.read_text())
    if mapping.get("schema") != IDENTITY_MAP_SCHEMA:
        raise ValueError("invalid identity map")
    catalog = {}
    for line in catalog_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["object_id"] in catalog:
            raise ValueError("duplicate catalog internal ID")
        catalog[row["object_id"]] = row
    objects = []
    for row in mapping["objects"]:
        source = catalog[row["object_id"]]
        if source.get("eligible") is not True or not source.get("solutions"):
            raise ValueError("mapped object has no eligible period")
        chosen = source["solutions"][0]
        period = float(chosen["period_hours"])
        if not math.isfinite(period) or period <= 0:
            raise ValueError("invalid externally supplied period")
        objects.append({"object_id": row["object_id"], "held_out_fold": row["held_out_fold"],
                        "mpc_number": row["mpc_number"], "known_period_hours": period,
                        "period_provenance": {"externally_supplied": True,
                            "selection_policy": "first_frozen_catalog_solution_in_catalog_order",
                            "source_model_id": chosen["model_id"], "source_spin_sha256": chosen["source_sha256"],
                            "source_catalog_sha256": sha256(catalog_path)}})
    document = {"schema": "delphi.k3-mapped-ztf-period-manifest.v1", "objects": objects,
                "object_count": len(objects), "identity_map_sha256": sha256(identity_map_path),
                "source_catalog_sha256": sha256(catalog_path), "pole_coordinates_exported": False}
    write_new_json(output, document)
    return document
