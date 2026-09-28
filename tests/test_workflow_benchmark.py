"""Contracts for the adaptive DeLPHI/classical workflow comparison."""

from __future__ import annotations

import numpy as np
import pytest

from lc_pipeline import workflow_benchmark as workflow


def _fit(axis, rms, index=0):
    return {
        "axis": axis,
        "final_relative_rms": rms,
        "start_index": index,
        "stage": "initial",
    }


def test_delphi_starts_expand_three_axes_to_both_directed_signs():
    axes = np.eye(3)
    starts = workflow.delphi_starts(axes)
    assert starts.shape == (6, 3)
    for axis in axes:
        assert any(np.allclose(start, axis) for start in starts)
        assert any(np.allclose(start, -axis) for start in starts)


def test_cluster_diagnostic_is_axial_and_uses_best_cluster():
    fits = [
        _fit([1, 0, 0], 1.0, 0),
        _fit([-1, 0, 0], 1.001, 1),
        _fit([0.99, 0.01, 0], 1.002, 2),
        _fit([0, 1, 0], 1.03, 3),
        _fit([0, -1, 0], 1.04, 4),
        _fit([0, 0, 1], 1.05, 5),
    ]
    diagnostic = workflow.cluster_diagnostic(fits)
    assert diagnostic["support_count"] == 3
    assert diagnostic["support_fraction"] == pytest.approx(0.5)
    assert diagnostic["outside_rms_ratio"] == pytest.approx(1.03)
    assert diagnostic["cluster_count"] == 3


def test_shared_rule_falls_back_on_failure_low_support_or_small_gap():
    rule = {
        "schema": workflow.RULE_SCHEMA,
        "always_fallback": False,
        "minimum_support_fraction": 0.5,
        "minimum_outside_rms_ratio": 1.02,
    }
    accepted = {
        "valid_initial_fits": 6,
        "support_fraction": 0.5,
        "outside_rms_ratio": 1.03,
    }
    assert not workflow.requires_fallback(accepted, rule)
    assert workflow.requires_fallback({**accepted, "valid_initial_fits": 5}, rule)
    assert workflow.requires_fallback({**accepted, "support_fraction": 1 / 3}, rule)
    assert workflow.requires_fallback({**accepted, "outside_rms_ratio": 1.01}, rule)
    assert not workflow.requires_fallback({**accepted, "outside_rms_ratio": None}, rule)


def test_always_fallback_candidate_is_available():
    rules = workflow._candidate_rules()
    assert len(rules) == 16
    assert rules[-1]["always_fallback"] is True
    diagnostic = {
        "valid_initial_fits": 6,
        "support_fraction": 1.0,
        "outside_rms_ratio": None,
    }
    assert workflow.requires_fallback(diagnostic, rules[-1])


def test_jobs_are_complete_reproducible_and_balanced():
    lock = {
        "objects": [
            {"object_id": f"asteroid_{index}", "fold": index % 5}
            for index in range(140)
        ]
    }
    jobs = workflow._jobs(lock)
    assert len(jobs) == 840
    assert jobs == workflow._jobs(lock)
    keys = {(row["object_id"], row["repeat_index"], row["arm"]) for row in jobs}
    assert len(keys) == 840


def test_axial_angle_treats_antipodes_as_equal():
    assert workflow.axial_angle_degrees([1, 0, 0], [-1, 0, 0]) == pytest.approx(0)
    assert workflow.axial_angle_degrees([1, 0, 0], [0, 1, 0]) == pytest.approx(90)


def test_environment_comparison_accepts_case_only_path_alias(tmp_path):
    interpreter = tmp_path / "python"
    interpreter.write_text("same bytes")
    recorded = {"device": "cuda", "interpreter": {"path": str(interpreter), "sha256": "x"}}
    current = {"device": "cuda", "interpreter": {"path": str(interpreter), "sha256": "x"}}
    assert workflow._environment_equivalent(recorded, current)
    current["interpreter"]["sha256"] = "y"
    assert not workflow._environment_equivalent(recorded, current)
