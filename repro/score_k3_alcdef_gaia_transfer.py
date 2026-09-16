#!/usr/bin/env python3
"""Open Gaia axes only after sealed ALCDEF predictions, then score transfer.

The output is explicitly a retrospective, model-specific transfer analysis. It
does not certify that no earlier historical project experiment encountered an
identity, nor that Gaia's model-derived axes are physical ground truth.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from lc_pipeline.k3.external_sources import parse_gaia_dr3_spins
from lc_pipeline.k3.generalization_validation import (
    analyze_generalization,
    locked_resource_contains,
    sha256_file,
    validate_prediction_receipt,
    validate_protocol_lock,
)


def _read(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not an object")
    return value


def _axis(longitude_deg: float, latitude_deg: float) -> list[float]:
    longitude, latitude = math.radians(longitude_deg), math.radians(latitude_deg)
    return [math.cos(latitude) * math.cos(longitude), math.cos(latitude) * math.sin(longitude), math.sin(latitude)]


def score(lock_path: Path, receipt_path: Path, candidate_path: Path, atlas_path: Path, cohort_lock_path: Path, exposure_path: Path, gaia_path: Path, cohort_output: Path, analysis_output: Path) -> dict:
    if cohort_output.exists() or analysis_output.exists():
        raise ValueError("cohort and analysis outputs must be new")
    lock, receipt = _read(lock_path), _read(receipt_path)
    validate_protocol_lock(lock)
    validate_prediction_receipt(receipt, lock_path=lock_path, expected_predictions={"candidate": candidate_path, "atlas": atlas_path})
    if not locked_resource_contains(lock, cohort_lock_path) or not locked_resource_contains(lock, exposure_path) or not locked_resource_contains(lock, gaia_path):
        raise ValueError("cohort, model-specific exposure, and Gaia table must be bound in pre-scoring lock")
    selection, exposure = _read(cohort_lock_path), _read(exposure_path)
    if selection.get("schema") != "delphi.k3-alcdef-gaia-cohort-lock.v1" or exposure.get("schema") != "delphi.k3-model-specific-exposure-audit.v1":
        raise ValueError("cohort or model-specific exposure schema mismatch")
    selected = [row.get("object_id") for row in selection.get("objects", []) if isinstance(row, dict)]
    exposure_by_id = {row.get("object_id"): row for row in exposure.get("objects", []) if isinstance(row, dict)}
    if len(selected) != 30 or len(set(selected)) != 30 or any(not isinstance(value, str) for value in selected):
        raise ValueError("locked cohort identities are invalid")
    if any(exposure_by_id.get(identity, {}).get("status") != "model_training_unexposed" for identity in selected):
        raise ValueError("an external identity is not model-training-unexposed")
    # This is intentionally the first Gaia-axis parse in the scoring command,
    # after verification that both arms have been sealed.
    gaia = {f"mpc:{row.object_number}": row for row in parse_gaia_dr3_spins(gaia_path)}
    rows = []
    for identity in selected:
        reference = gaia.get(identity)
        if reference is None:
            raise ValueError(f"locked identity missing from Gaia table: {identity}")
        axes = [_axis(reference.lambda1_deg, reference.beta1_deg)]
        if reference.lambda2_deg is not None and reference.beta2_deg is not None:
            axes.append(_axis(reference.lambda2_deg, reference.beta2_deg))
        rows.append({"object_id": identity, "eligible": True, "source": "ALCDEF_PDS_v1", "references": axes, "reference_type": "Gaia_DR3_model_derived_spin_solution", "reference_coordinate_frame": "ecliptic_J2000", "reference_method": reference.method, "reference_observation_count": reference.observation_count, "exposure_scope": "model_training_unexposed_not_project_wide_history_unexposed"})
    cohort = {"schema": "delphi.k3-alcdef-gaia-scoring-cohort.v1", "scope": {"retrospective": True, "model_training_unexposed": True, "project_wide_history_unexposed": False, "reference_is_model_derived": True}, "cohort_lock_sha256": sha256_file(cohort_lock_path), "model_specific_exposure_sha256": sha256_file(exposure_path), "gaia_table_sha256": sha256_file(gaia_path), "objects": rows}
    cohort_output.parent.mkdir(parents=True, exist_ok=True)
    cohort_output.write_text(json.dumps(cohort, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    analysis = analyze_generalization(cohort=cohort, candidate_predictions=_read(candidate_path), atlas_predictions=_read(atlas_path))
    analysis["scope"] = cohort["scope"]
    analysis["provenance"] = {"pre_scoring_lock_sha256": sha256_file(lock_path), "prediction_receipt_sha256": sha256_file(receipt_path), "scoring_cohort_sha256": sha256_file(cohort_output), "candidate_predictions_sha256": sha256_file(candidate_path), "atlas_predictions_sha256": sha256_file(atlas_path)}
    analysis_output.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("lock", "receipt", "candidate", "atlas", "cohort-lock", "exposure", "gaia", "cohort-output", "analysis-output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = score(args.lock, args.receipt, args.candidate, args.atlas, args.cohort_lock, args.exposure, args.gaia, args.cohort_output, args.analysis_output)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({"n": result["eligible_denominator"], "mean": result["primary"]["value"]}, sort_keys=True))


if __name__ == "__main__":
    main()
