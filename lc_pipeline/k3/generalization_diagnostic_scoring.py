"""Score the complete, already-exposed development factorial after prediction.

This is deliberately separate from the untouched external-test lock and analysis.
It cannot certify a new-identity result or choose a variant for a final test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .generalization_diagnostics import VARIANTS, sha256, write_new_json
from .generalization_validation import (
    _identity_bootstrap_interval,
    _paired_bootstrap,
    _reference_axes,
    _score_prediction,
    _summary,
)

REFERENCE_CATALOG_SHA256 = "ac1abdab62453fca6e3d7473af0ad3b7de1bcb04cee9b947fb7079d340613679"
FROZEN_SEEDS = (17, 42, 137, 777, 2027)
ROLE = "post_hoc_development_only"


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _rows(rows: list[dict], expected: set[tuple[str, str]], role: str) -> dict:
    keys = [(row["object_id"], row["variant"]) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError(f"{role} must contain every object in all eight variants exactly once")
    return dict(zip(keys, rows, strict=True))


def _bound_path(root: Path, row: dict) -> Path:
    path = root / row["path"]
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes its receipt directory")
    if sha256(path) != row["sha256"]:
        raise ValueError(f"artifact hash mismatch: {path.name}")
    return path


def score_diagnostics(
    *,
    diagnostic_manifest: Path,
    prediction_receipt: Path,
    identity_map: Path,
    splits: Path,
    model_manifest: Path,
    reference_catalog: Path,
    output: Path,
    expected_object_count: int = 170,
    expected_reference_sha256: str = REFERENCE_CATALOG_SHA256,
) -> dict:
    """Validate sealed complete predictions before reading the original labels."""
    if output.exists():
        raise ValueError("refusing to overwrite a development diagnostic report")
    diagnostic = _json(diagnostic_manifest)
    receipt = _json(prediction_receipt)
    mapping = _json(identity_map)
    split_document = _json(splits)
    models = _json(model_manifest)
    if (
        diagnostic.get("schema") != "delphi.k3-generalization-diagnostics.v1"
        or receipt.get("schema") != "delphi.k3-generalization-development-prediction-receipt.v1"
        or diagnostic.get("role") != ROLE
        or receipt.get("role") != ROLE
        or receipt.get("references_opened_by_prediction") is not False
    ):
        raise ValueError("scoring requires label-free, exposed-development prediction receipts")
    if (
        receipt.get("diagnostic_manifest_sha256") != sha256(diagnostic_manifest)
        or receipt.get("splits_sha256") != sha256(splits)
        or receipt.get("model_manifest_sha256") != sha256(model_manifest)
        or diagnostic.get("identity_map_sha256") != sha256(identity_map)
        or mapping.get("splits_sha256") != sha256(splits)
        or mapping.get("schema") != "delphi.k3-survey-identity-map.v1"
    ):
        raise ValueError("development resource hash or identity-map mismatch")
    fold_lookup: dict[str, int] = {}
    for fold in split_document["folds"]:
        for object_id in fold["test_ids"]:
            if object_id in fold_lookup:
                raise ValueError("object appears in more than one held-out fold")
            fold_lookup[object_id] = fold["fold"]
    mapped = {row["object_id"]: row for row in mapping["objects"]}
    if (
        len(mapped) != len(mapping["objects"])
        or len(mapped) != expected_object_count
        or set(mapped) != set(fold_lookup)
        or diagnostic.get("object_count") != expected_object_count
        or set(diagnostic.get("variants", [])) != set(VARIANTS)
    ):
        raise ValueError("diagnostic cohort does not match the complete original split")
    for object_id, row in mapped.items():
        if (
            object_id != f'asteroid_{row["damit_id"]}'
            or row["held_out_fold"] != fold_lookup[object_id]
            or row["physical_identity"] != f'mpc:{row["mpc_number"]}'
        ):
            raise ValueError("mapped physical identity or held-out fold mismatch")
    expected = {(object_id, variant) for object_id in mapped for variant in VARIANTS}
    diagnostic_rows = _rows(diagnostic["records"], expected, "diagnostic manifest")
    prediction_rows = _rows(receipt["predictions"], expected, "prediction receipt")
    if any(row.get("status") not in ("ready", "input_unavailable") for row in diagnostic_rows.values()):
        raise ValueError("unknown diagnostic input status")
    if any(row.get("status") not in ("ok", "failed") for row in prediction_rows.values()):
        raise ValueError("unknown prediction status")
    payloads = {}
    available_ids = {
        object_id for object_id in mapped
        if diagnostic_rows[(object_id, "original")]["status"] == "ready"
    }
    for key in sorted(expected):
        object_id, variant = key
        record = diagnostic_rows[key]
        ready = record["status"] == "ready"
        if ready != (object_id in available_ids):
            raise ValueError("input-availability denominator differs across variants")
        if ready:
            _bound_path(diagnostic_manifest.parent, record)
        payload = _json(_bound_path(prediction_receipt.parent, prediction_rows[key]))
        fold = fold_lookup[object_id]
        expected_models = [
            {key: entry[key] for key in ("filename", "fold", "seed", "sha256")}
            for entry in models["models"] if entry["fold"] == fold
        ]
        expected_models.sort(key=lambda row: row["seed"])
        bindings = sorted(payload.get("model_bindings", []), key=lambda row: row["seed"])
        if (
            payload.get("schema") != "delphi.k3-generalization-development-prediction.v1"
            or len(expected_models) != 5
            or tuple(row["seed"] for row in expected_models) != FROZEN_SEEDS
            or bindings != expected_models
            or payload.get("folds") != [fold]
            or payload.get("model_count") != 5
            or payload.get("model_manifest_sha256") != sha256(model_manifest)
            or payload.get("object_id") != object_id
            or payload.get("variant") != variant
            or payload.get("prepared_sha256") != record.get("sha256")
            or payload.get("candidate_semantics") != "unordered_three_axis_set"
            or payload.get("status") != prediction_rows[key]["status"]
            or (not ready and payload.get("status") != "failed")
        ):
            raise ValueError("prediction identity, model-fold, status, or input binding mismatch")
        payloads[key] = payload

    # All prediction records have now been checked; references enter only here.
    if sha256(reference_catalog) != expected_reference_sha256:
        raise ValueError("reference catalog does not match the frozen original catalog hash")
    catalog = {}
    for line in reference_catalog.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["object_id"] in catalog:
            raise ValueError("duplicate original reference-catalog identity")
        catalog[row["object_id"]] = row
    references = {}
    for object_id in mapped:
        row = catalog.get(object_id)
        if row is None or row.get("eligible") is not True:
            raise ValueError("an intended object has no eligible original reference")
        references[object_id] = _reference_axes(row)
    ids = sorted(mapped)
    available = np.asarray([object_id in available_ids for object_id in ids])
    errors, details = {}, {}
    for variant in VARIANTS:
        object_rows = []
        for object_id in ids:
            error, failure = _score_prediction(payloads[(object_id, variant)], references[object_id], role="candidate")
            object_rows.append({
                "object_id": object_id, "physical_identity": mapped[object_id]["physical_identity"],
                "held_out_fold": fold_lookup[object_id], "input_available": object_id in available_ids,
                "reference_count": len(references[object_id]), "oracle_at_3_error_deg": error,
                "failure_reason": failure,
            })
        errors[variant] = np.asarray([row["oracle_at_3_error_deg"] for row in object_rows])
        details[variant] = object_rows
    summaries = {}
    for variant in VARIANTS:
        values = errors[variant]
        summaries[variant] = {
            "all_intended_objects": {**_summary(values), "mean_ci95": _identity_bootstrap_interval(values)},
            "fixed_available_input_cohort": _summary(values[available]) if available.any() else None,
            "prediction_failure_count": sum(row["failure_reason"] is not None for row in details[variant]),
            "paired_change_vs_original": _paired_bootstrap(errors["original"] - values),
            "objects": details[variant],
        }
    factorial = {}
    for factor_index, factor in enumerate(("distance", "light_time", "geometry_epochs")):
        differences = []
        for variant, flags in VARIANTS.items():
            if not flags[factor_index]:
                on = tuple(True if index == factor_index else flag for index, flag in enumerate(flags))
                matched = next(name for name, other in VARIANTS.items() if other == on)
                differences.append(errors[variant] - errors[matched])
        factorial[factor] = _paired_bootstrap(np.mean(differences, axis=0))
    report = {
        "schema": "delphi.k3-generalization-development-diagnostic-score.v1", "role": ROLE,
        "new_identity_external_validation": False, "model_selection_authorized_by_report": False,
        "interpretation": "descriptive development comparison; not a simultaneous confidence statement or final-test result",
        "reference_status": "original model-derived catalog axes; independent-photometry lineage not established",
        "failure_policy": "90 degrees; preserve every intended object in every variant",
        "n_intended_objects": len(ids), "n_available_inputs": int(available.sum()),
        "candidate_semantics": "unordered_three_axis_set", "variants": summaries,
        "factorial_average_effects": factorial,
        "source_hashes": {
            "diagnostic_manifest": sha256(diagnostic_manifest), "prediction_receipt": sha256(prediction_receipt),
            "identity_map": sha256(identity_map), "splits": sha256(splits),
            "model_manifest": sha256(model_manifest), "reference_catalog": sha256(reference_catalog),
            "scoring_implementation": sha256(Path(__file__)),
        },
    }
    write_new_json(output, report)
    return report
