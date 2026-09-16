import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from lc_pipeline.k3.generalization_diagnostics import (
    LIGHT_DAYS_PER_AU,
    VARIANTS,
    diagnose_object,
    invariant_variant_matrix,
    prepare_diagnostics,
    sha256,
    transform_object,
    write_new_json,
)


def example():
    return {"object_id": "asteroid_1", "known_period_hours": 24.0, "epochs": [
        {"epoch_id": "native", "observations": [
            {"time_jd": 2450000.0 + t, "relative_brightness": 1.0 / r**4,
             "measured_error": 0.1 / r**4,
             "sun_asteroid_ecliptic_j2000_au": [r * math.cos(a), r * math.sin(a), 0],
             "observer_asteroid_ecliptic_j2000_au": [r, 0, 0]}
            for t, r, a in [(0, 1, 0), (1, 2, 0), (100, 3, 1), (101, 4, 1), (200, 5, 2)]
        ]}]}


def test_full_factorial_and_no_input_mutation():
    value = example()
    saved = copy.deepcopy(value)
    assert set(VARIANTS.values()) == set(invariant_variant_matrix())
    for name in VARIANTS:
        transform_object(value, name)
    assert value == saved


def test_distance_reduction_and_uncertainty_preserve_relative_error():
    transformed = transform_object(example(), "distance")
    rows = transformed["epochs"][0]["observations"]
    np.testing.assert_allclose([r["relative_brightness"] for r in rows], 1)
    np.testing.assert_allclose([r["measured_error"] for r in rows], 0.1)


def test_light_time_direction_and_double_correction_guard():
    value = example()
    transformed = transform_object(value, "light_time")
    assert transformed["epochs"][0]["observations"][0]["time_jd"] == pytest.approx(2450000.0 - LIGHT_DAYS_PER_AU)
    with pytest.raises(ValueError, match="already transformed"):
        transform_object(transformed, "distance")


def test_geometry_groups_report_singletons_without_silent_object_exclusion():
    result = transform_object(example(), "geometry_epochs")
    assert [len(epoch["observations"]) for epoch in result["epochs"]] == [2, 2]
    assert result["generalization_transform"]["retained_observations"] == 4
    assert len(result["generalization_transform"]["dropped_observations"]) == 1
    assert result["generalization_transform"]["input_observations"] == 5


def test_diagnostic_detects_mixed_geometry_and_distance_trend():
    result = diagnose_object(example())
    assert result["mixed_bin_directions_above_30_deg"] >= 1
    assert result["log_flux_log_distance_product_correlation"] == pytest.approx(-1)
    assert result["epoch_count"] == 1


def test_no_overwrite(tmp_path):
    path = tmp_path / "result.json"
    write_new_json(path, {"value": 1})
    with pytest.raises(FileExistsError):
        write_new_json(path, {"value": 2})


def _bound_example():
    value = example()
    value.update(schema="delphi.k3-mapped-ztf-prepared.v1", held_out_fold=0, period_provenance="fixture")
    value["identity_binding"] = {
        "schema": "delphi.k3-survey-identity-binding.v1", "object_id": "asteroid_1",
        "damit_id": 1, "mpc_number": 999, "resolved_mpc_number": 999,
        "held_out_fold": 0, "identity_map_sha256": "a" * 64, "identity_table_sha256": "b" * 64,
    }
    return value


def test_prepare_retains_unavailable_objects_in_every_variant(tmp_path):
    source = tmp_path / "prepared"
    object_path = source / "objects" / "asteroid_1.json"
    write_new_json(object_path, _bound_example())
    manifest_path = source / "manifest.json"
    write_new_json(manifest_path, {"expected_object_count": 2, "identity_map_sha256": "a" * 64, "objects": [
        {"object_id": "asteroid_1", "status": "ready", "path": "objects/asteroid_1.json", "sha256": sha256(object_path)},
        {"object_id": "asteroid_2", "status": "missing_local_raw"},
    ]})
    spec = Path(__file__).resolve().parents[1] / "repro" / "k3_generalization_study_spec.yaml"
    result = prepare_diagnostics(manifest_path, spec, tmp_path / "diagnostics")
    assert result["object_count"] == 2
    assert result["available_input_count"] == 1
    assert len(result["records"]) == 16
    missing = [row for row in result["records"] if row["object_id"] == "asteroid_2"]
    assert len(missing) == 8
    assert all(row["status"] == "input_unavailable" and row["path"] is None for row in missing)


def test_prepare_rejects_unbound_legacy_input_and_incomplete_denominator(tmp_path):
    source = tmp_path / "prepared"
    object_path = source / "objects" / "asteroid_1.json"
    write_new_json(object_path, example())
    manifest = {"objects": [{"object_id": "asteroid_1", "sha256": sha256(object_path)}]}
    manifest_path = source / "manifest.json"
    write_new_json(manifest_path, manifest)
    spec = Path(__file__).resolve().parents[1] / "repro" / "k3_generalization_study_spec.yaml"
    with pytest.raises(ValueError, match="identity binding"):
        prepare_diagnostics(manifest_path, spec, tmp_path / "unbound")
    manifest["expected_object_count"] = 2
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="omits intended objects"):
        prepare_diagnostics(manifest_path, spec, tmp_path / "missing")


def test_mapped_prepared_object_loads_without_dropping_identity_guard(tmp_path):
    from lc_pipeline.k3.ztf_prediction import _load_prepared

    path = tmp_path / "mapped.json"
    write_new_json(path, _bound_example())
    object_id, period, epochs = _load_prepared(path)
    assert object_id == "asteroid_1" and period.hours == 24.0
    assert len(epochs) == 1
