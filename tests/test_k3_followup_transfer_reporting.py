from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from lc_pipeline.k3.followup_transfer_reporting import export_followup_transfer


def test_export_is_read_only_and_has_portable_manifest(tmp_path):
    root = tmp_path / "data"
    ztf = (
        root
        / "generalization-20260910/mapped-ztf-development-diagnostic-score-current-endpoint.json"
    )
    alcdef = (
        root
        / "generalization-20260910/locked-cohort/alcdef-gaia-transfer-analysis-30-20260911.json"
    )
    grid = root / "pole-grid-benchmark/20260912-environment-relocked-v3/full-score.json"
    grid_lock = root / "pole-grid-benchmark/20260912-environment-relocked-v3/lock.json"
    objects = root / "generalization-20260910/mapped-ztf/prepared-current-endpoint/objects"
    temporal = root / "temporal-scores-final"
    comparator = tmp_path / "comparator.npz"
    for path in (ztf, alcdef, grid, grid_lock):
        path.parent.mkdir(parents=True, exist_ok=True)
    objects.mkdir(parents=True)
    temporal.mkdir(parents=True)
    np.savez(comparator, object_ids=np.asarray(["asteroid_1", "asteroid_2"]), atlas_errors_deg=np.asarray([10.0, 20.0]))
    for number, count in ((1, 2), (2, 3)):
        (objects / f"asteroid_{number}.json").write_text(
            json.dumps(
                {
                    "epochs": [{"observations": [{"time_jd": float(index)} for index in range(count)]}]
                }
            )
        )
    (temporal / "asteroid_9.json").write_text(
        json.dumps({"object_id": "asteroid_9", "oracle_at_3_error_deg": 25.0})
    )
    grid_lock.write_text(
        json.dumps(
            {
                "environment": {
                    "platform": "test",
                    "cpu_models": ["test"],
                    "cpu_count": 1,
                    "gpu": [
                        "NVIDIA GeForce RTX 4070, GPU-01234567-abcd, 591.86, 12282 MiB"
                    ],
                    "python": "test",
                    "packages": {},
                    "neural_torch_threads": 1,
                },
                "settings": {"workers": 1},
            }
        )
    )
    ztf.write_text(
        json.dumps(
            {
                "variants": {
                    "original": {
                        "prediction_failure_count": 1,
                        "fixed_available_input_cohort": {
                            "n_objects": 2,
                            "mean_oracle_at_3_error_deg": 20,
                        },
                        "all_intended_objects": {"n_objects": 3, "mean_oracle_at_3_error_deg": 30},
                        "objects": [
                            {"object_id": "asteroid_1", "input_available": True},
                            {"object_id": "asteroid_2", "input_available": True},
                            {"object_id": "asteroid_3", "input_available": False},
                        ],
                    }
                }
            }
        )
    )
    alcdef.write_text(
        json.dumps(
            {
                "objects": [{"atlas_oracle_at_3_error_deg": 10}],
                "secondary": {"n_objects": 1},
                "primary": {"value": 20},
                "failures": {"n_objects": 0},
                "scope": {"retrospective": True},
                "uniform_random_three": {"mean_oracle_at_3_error_deg": 35.0},
            }
        )
    )
    analysis = {
        "scope": "primary",
        "baseline_arm": "a",
        "candidate_arm": "b",
        "recovery": {
            "object_count": 1,
            "baseline_success_count": 1,
            "candidate_success_count": 1,
            "simultaneous_exact_lower": -0.2,
        },
        "runtime": {
            "point_ratio": 1,
            "percentile_interval_95": {"lower": 0.9, "upper": 1.1},
            "classical_seconds": {"mean": 5},
            "guided_seconds": {"mean": 5},
        },
        "rms": {"point_ratio": 1, "percentile_interval_95": {"lower": 0.9, "upper": 1.1}},
        "decision": {"passed_all_four_conditions": False},
    }
    grid.write_text(
        json.dumps(
            {
                "results": {
                    "standard6_descriptive": {"cold": {}, "warm": {}},
                    "analyses": {
                        "cold_step20_primary": analysis,
                        "warm_step20_separately_scoped": {**analysis, "scope": "separately_scoped"},
                    },
                    "arm_summaries": {
                        name: {"cold": {}, "warm": {}} for name in ("standard6", "classical20", "guided20")
                    },
                    "selected_axial_errors_by_object": {
                        "classical20": {
                            "cold": [
                                {"object_id": "asteroid_1", "fold": 0, "mean_selected_error_degrees": 10.0},
                                {"object_id": "asteroid_2", "fold": 1, "mean_selected_error_degrees": 30.0},
                            ]
                        },
                        "guided20": {
                            "cold": [
                                {"object_id": "asteroid_1", "fold": 0, "mean_selected_error_degrees": 12.0},
                                {"object_id": "asteroid_2", "fold": 1, "mean_selected_error_degrees": 18.0},
                            ]
                        },
                    },
                }
            }
        )
    )
    output = tmp_path / "export"
    manifest = export_followup_transfer(root, output, comparator)
    assert manifest["schema"] == "delphi.k3-followup-transfer-publication.v1"
    assert (output / "survey-comparison.pdf").is_file()
    assert (output / "broad-grid-timing-rms.pdf").is_file()
    assert "tmp_path" not in (output / "manifest.json").read_text()
    result = json.loads((output / "followup-summary.json").read_text())
    assert result["broad_grid"]["runtime_point_ratio"] == 1
    assert result["broad_grid"]["host"]["gpu"] == [
        "NVIDIA GeForce RTX 4070, 591.86, 12282 MiB"
    ]
    assert result["broad_grid"]["analyses"]["cold_step20_primary"]["classical_mean_seconds"] == 5
    assert "warm_step20_separately_scoped" not in result["broad_grid"]["analyses"]
    assert result["ztf_sampling"]["points_min_median_max"] == [2, 2.5, 3]
    assert result["temporal_cases"]["object_errors_deg"] == {"asteroid_9": 25.0}
    assert result["broad_grid"]["angular_comparison_step20"]["within_20_status_changes"] == 1
    assert "{20.0000}" in (output / "followup-macros.tex").read_text()
    assert export_followup_transfer(root, output, comparator) == manifest
    (output / "followup-summary.json").write_text("do not overwrite")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        export_followup_transfer(root, output, comparator)
