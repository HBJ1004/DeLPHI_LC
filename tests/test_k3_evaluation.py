"""Statistical gate tests with deliberately synthetic score/error arrays."""

from __future__ import annotations

import numpy as np
import pytest

from lc_pipeline.k3.evaluation import (
    K3EvaluationError,
    ensemble_score_grids,
    evaluate_oof_promotion_gate,
    holm_adjust,
    paired_bootstrap_ci,
    paired_permutation_pvalue,
)


def test_paired_statistics_and_holm_are_deterministic() -> None:
    improvements = np.linspace(4.0, 8.0, 20)
    first = paired_bootstrap_ci(improvements, resamples=500, seed=4)
    second = paired_bootstrap_ci(improvements, resamples=500, seed=4)
    assert first == second and first[0] > 4.0
    assert paired_permutation_pvalue(improvements, sign_flips=2000, seed=4) < 0.01
    assert holm_adjust({"b": 0.02, "a": 0.001})["a"] <= holm_adjust({"b": 0.02, "a": 0.001})["b"]


def test_ensemble_averages_complete_maps_before_mode_extraction() -> None:
    first = np.zeros((2, 6144))
    second = np.zeros_like(first)
    first[0, 4], second[0, 4] = 10.0, 0.0
    first[0, 5], second[0, 5] = 0.0, 10.0
    averaged = ensemble_score_grids((first, second))
    assert averaged.shape == (2, 6144) and averaged[0, 4] == averaged[0, 5]
    with pytest.raises(K3EvaluationError, match="6,144"):
        ensemble_score_grids((np.zeros((1, 10)), np.zeros((1, 10))))


def test_oof_gate_fails_when_improvement_is_only_marginal() -> None:
    k3 = np.full(40, 27.0)
    comparator = np.full(40, 28.0)
    verdict = evaluate_oof_promotion_gate(
        k3,
        v1_errors=comparator,
        atlas_errors=comparator,
        deranged_errors=comparator,
        bootstrap_resamples=200,
        permutation_sign_flips=500,
    )
    assert not verdict.passed
    assert any("point improvement" in failure for failure in verdict.failures)


def test_oof_gate_passes_only_with_strong_predeclared_improvements() -> None:
    k3 = np.linspace(10.0, 14.0, 40)
    v1 = k3 + 8.0
    atlas = k3 + 9.0
    deranged = k3 + 20.0
    verdict = evaluate_oof_promotion_gate(
        k3,
        v1_errors=v1,
        atlas_errors=atlas,
        deranged_errors=deranged,
        seed_k3_errors={17: k3, 42: k3 + 0.2},
        seed_input_swap_errors={17: k3 + 10.0, 42: k3 + 9.0},
        bootstrap_resamples=300,
        permutation_sign_flips=2000,
    )
    assert verdict.passed, verdict.failures


def test_oof_gate_uses_each_seeds_own_input_swap_control() -> None:
    k3 = np.linspace(10.0, 14.0, 40)
    verdict = evaluate_oof_promotion_gate(
        k3,
        v1_errors=k3 + 8.0,
        atlas_errors=k3 + 9.0,
        deranged_errors=k3 + 20.0,
        seed_k3_errors={17: k3},
        seed_input_swap_errors={17: k3 - 1.0},
        bootstrap_resamples=200,
        permutation_sign_flips=1000,
    )
    assert not verdict.passed
    assert verdict.metrics["seed_17_input_swap_gap_deg"] == -1.0
    assert any("seed 17" in failure for failure in verdict.failures)
