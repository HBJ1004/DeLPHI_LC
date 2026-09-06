"""Publication disclosures are derived from aligned per-object evidence."""

from __future__ import annotations

import json

import numpy as np
import pytest

from repro.derive_k3_disclosures import derive_disclosures, render_tex


def _artifacts(tmp_path):
    object_ids = np.asarray(["a", "b", "c"])
    ensemble = tmp_path / "ensemble.npz"
    np.savez(
        ensemble,
        schema=np.asarray("delphi.k3-real-oof-ensemble.v1"),
        object_ids=object_ids,
        oracle_errors_deg=np.asarray([10.0, 20.0, 30.0]),
        grid_oracle_errors_deg=np.asarray([9.0, 18.0, 32.0]),
    )
    controls = tmp_path / "controls.npz"
    np.savez(
        controls,
        schema=np.asarray("delphi.k3-v1-comparators.v1"),
        object_ids=object_ids,
        atlas_errors_deg=np.asarray([20.0, 30.0, 40.0]),
        deranged_errors_deg=np.asarray([25.0, 35.0, 45.0]),
    )
    rows = []
    for index, object_id in enumerate(object_ids):
        rows.append(
            {
                "object_id": str(object_id),
                "neural_inference_wall_seconds": 0.1,
                "baseline": {
                    "wall_seconds": 1.0,
                    "iterations": 300,
                    "valid_starts": 6,
                    "success": index != 2,
                    "final_rms": 1.0,
                    "pole_error_deg": 5.0 + index,
                },
                "candidate": {
                    "wall_seconds": 1.1,
                    "iterations": 300,
                    "valid_starts": 6,
                    "success": True,
                    "final_rms": 0.9,
                    "pole_error_deg": 4.0 + index,
                },
            }
        )
    downstream = tmp_path / "rows.json"
    downstream.write_text(
        json.dumps(
            {
                "schema": "delphi.k3-fixed-period-rows.v1",
                "object_ids": object_ids.tolist(),
                "rows": rows,
            }
        )
    )
    return ensemble, controls, downstream


def test_disclosures_report_fixed_work_and_control_contrasts(tmp_path) -> None:
    values = derive_disclosures(*_artifacts(tmp_path))
    assert values["nonlearned_control_contrasts"]["atlas"]["mean_improvement_deg"] == 10.0
    assert values["downstream"]["iterations_per_arm"] == 900
    assert values["downstream"]["completed_starts_per_arm"] == 18
    assert values["downstream"]["baseline_recovered"] == 2
    assert values["downstream"]["candidate_recovered"] == 3
    assert values["downstream"]["completed_in_both"]["n_objects"] == 3
    assert values["downstream"]["inversion_only_wall_time_ratio"] == pytest.approx(1.0)
    assert values["refinement"]["objects_worsened"] == 2
    assert "KThreeAtlasImprovement" in render_tex(values)


def test_disclosures_reject_misaligned_object_order(tmp_path) -> None:
    ensemble, controls, downstream = _artifacts(tmp_path)
    data = json.loads(downstream.read_text())
    data["object_ids"].reverse()
    downstream.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="object order"):
        derive_disclosures(ensemble, controls, downstream)
