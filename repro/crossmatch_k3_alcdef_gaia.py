#!/usr/bin/env python3
"""Build a label-aware but prediction-free ALCDEF--Gaia candidate inventory.

This is an eligibility screen, not an external evaluation.  It only joins
public ALCDEF metadata, a public Gaia DR3 model-derived reference table, and
the previously frozen exposure audit.  It does not retrieve poles from DAMIT,
prepare model inputs, or generate predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lc_pipeline.k3.external_sources import ExternalSourceError, parse_gaia_dr3_spins


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, schema: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExternalSourceError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ExternalSourceError(f"unexpected schema in {path}: expected {schema}")
    return value


def crossmatch(
    alcdef_census: Path,
    gaia_table: Path,
    exposure_audit: Path,
    historical_metadata_audit: Path | None = None,
) -> dict[str, Any]:
    census = _load(alcdef_census, "delphi.k3-external-source-census.v1")
    exposure = _load(exposure_audit, "delphi.k3-identity-exposure-audit.v1")
    gaia = {row.object_number: row for row in parse_gaia_dr3_spins(gaia_table)}
    exposure_by_id = {str(row["object_id"]): row for row in exposure.get("objects", []) if isinstance(row, dict) and isinstance(row.get("object_id"), str)}
    historical_ids: set[str] = set()
    historical_source: dict[str, Any] | None = None
    if historical_metadata_audit is not None:
        historical = _load(historical_metadata_audit, "delphi.k3-historical-metadata-exposure-audit.v1")
        phase30 = historical.get("phase30")
        if not isinstance(phase30, dict) or not isinstance(phase30.get("candidate_inventory_overlap"), list):
            raise ExternalSourceError("historical metadata audit lacks phase30 candidate overlap")
        historical_ids = {item for item in phase30["candidate_inventory_overlap"] if isinstance(item, str)}
        historical_source = {"path": str(historical_metadata_audit), "sha256": _sha256(historical_metadata_audit), "candidate_ids_marked_exposed": len(historical_ids)}

    candidates: list[dict[str, Any]] = []
    for record in census.get("objects", []):
        if not isinstance(record, dict) or not record.get("primary_screen_approximation"):
            continue
        number = record.get("object_number")
        if not isinstance(number, int) or number not in gaia:
            continue
        spin = gaia[number]
        identity = f"mpc:{number}"
        exposure_record = exposure_by_id.get(identity)
        status = exposure_record.get("status", "not_in_prior_screen") if exposure_record else "not_in_prior_screen"
        reasons = list(exposure_record.get("reasons", []) if exposure_record else [])
        if identity in historical_ids:
            status = "exposed"
            reasons.append("present_in:historical_phase30_metadata")
        candidates.append({
            "object_id": identity,
            "alcdef": {
                "valid_point_count": record.get("valid_point_count"),
                "session_count": record.get("session_count"),
                "distinct_session_years_approximation": record.get("distinct_session_years_approximation"),
                "filters": record.get("filters", []),
            },
            "gaia_reference": {
                "object_name": spin.object_name,
                "reference_type": "Gaia_DR3_model_derived_spin_solution",
                "coordinate_frame": "ecliptic_J2000",
                "axis_count": 1 if spin.lambda2_deg is None else 2,
                "period_hours": spin.period_hours,
                "observation_count": spin.observation_count,
                "method": spin.method,
            },
            "exposure": {
                "status": status,
                "reasons": reasons,
            },
        })
    candidates.sort(key=lambda row: int(str(row["object_id"]).split(":", 1)[1]))
    status_counts: dict[str, int] = {}
    for row in candidates:
        status = str(row["exposure"]["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "schema": "delphi.k3-alcdef-gaia-eligibility.v1",
        "purpose": "candidate inventory only; not an external cohort lock or validation result",
        "sources": {
            "alcdef_census": {"path": str(alcdef_census), "sha256": _sha256(alcdef_census)},
            "gaia_dr3_spins": {"path": str(gaia_table), "sha256": _sha256(gaia_table), "catalog": "J/A+A/675/A24", "reference_is_model_derived": True},
            "exposure_audit": {"path": str(exposure_audit), "sha256": _sha256(exposure_audit)},
            "historical_metadata_audit": historical_source,
        },
        "eligibility": {
            "alcdef_screen": "at least five metadata sessions, 150 valid points, and two distinct session years (approximation)",
            "reference": "published Gaia DR3 model-derived ecliptic J2000 pole solution",
            "not_yet_checked": ["raw ALCDEF point-level usability", "Horizons geometry", "actual apparition separation", "full historical training/data exposure", "reference-photometry lineage"],
        },
        "counts": {"gaia_reference_rows": len(gaia), "joined_candidates": len(candidates), "exposure_status": status_counts},
        "candidates": candidates,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alcdef-census", type=Path, required=True)
    parser.add_argument("--gaia-table", type=Path, required=True)
    parser.add_argument("--exposure-audit", type=Path, required=True)
    parser.add_argument("--historical-metadata-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        payload = crossmatch(args.alcdef_census, args.gaia_table, args.exposure_audit, args.historical_metadata_audit)
    except ExternalSourceError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
