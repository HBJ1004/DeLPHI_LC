"""Hand-calculated checks for the compute-only median reporting script."""

import numpy as np
import pytest

from repro.own_period.median_stats import (
    Evidence,
    OWN,
    audit_manuscript,
    bootstrap_indices,
    check_printed_mean,
    compute_statistics,
    parse_inputs,
    randomized_paired_statistics,
)


def test_median_and_percentile_bootstrap_hand_calculation():
    # Bootstrap medians are 1, 2, 2, 100. Linear 2.5/97.5 percentiles:
    # 1 + .075 * (2 - 1), 2 + .925 * (100 - 2).
    draws = np.array([[0, 0, 1], [1, 1, 1], [0, 1, 2], [2, 2, 2]])
    result = compute_statistics([1, 2, 100], indices=draws)
    assert result["n"] == 3
    assert result["mean_recomputed"] == pytest.approx(103 / 3)
    assert result["median"] == 2
    assert result["interval"] == pytest.approx([1.075, 92.65])


def test_difference_of_medians_is_not_median_of_differences():
    # Pairwise differences: 1, -9, -1. Bootstrap differences of medians:
    # 1, -9, -9, -1. Thus the interval is [-9, -1 + .925 * 2].
    draws = np.array([[0, 0, 1], [1, 1, 1], [0, 1, 2], [2, 2, 2]])
    result = compute_statistics([1, 10, 20], [0, 19, 21], indices=draws)
    assert result["mean_recomputed"] == pytest.approx(-3)
    assert result["median"] == -9
    assert result["median_of_paired_differences"] == -1
    assert result["interval"] == pytest.approx([-9, .85])


@pytest.mark.parametrize("order", ["iid", "python_fold", "numpy_fold_major", "numpy_replicate_major"])
def test_bootstrap_keeps_pairing_and_fixed_fold_counts(order):
    folds = np.array([0, 0, 1, 1, 1])
    draws = bootstrap_indices(5, 17, 12, folds=folds, order=order)
    np.testing.assert_array_equal(draws, bootstrap_indices(5, 17, 12, folds=folds, order=order))
    if order != "iid":
        assert np.all(np.sum(folds[draws] == 0, axis=1) == 2)
        assert np.all(np.sum(folds[draws] == 1, axis=1) == 3)
    a = np.array([1, 3, 8, 17, 90])
    result = compute_statistics(a, a + 7, indices=draws)
    assert result["median"] == -7
    assert result["interval"] == [-7, -7]


def test_iid_random_stream_matches_the_original_seed():
    # First three draws of default_rng(0).integers(0, 3, (3, 3)).
    np.testing.assert_array_equal(bootstrap_indices(3, 0, 3),
                                  [[2, 1, 1], [0, 0, 0], [0, 0, 0]])


@pytest.mark.parametrize(("value", "printed", "matches"), [
    (15.9463285, "15.95", True),
    (15.9463285, "15.94", False),
    (48.101, "48.10", True),
    (-.8632769, "-0.86", True),
    (15.9463285, "16.0", False),
])
def test_mean_check_flags_a_printed_rounding_mismatch(value, printed, matches):
    result = check_printed_mean(value, printed)
    assert result["matches_printed_rounding"] is matches
    assert (result["flag"] == "MEAN_RECOMPUTATION_MISMATCH") is not matches


def test_input_parser_keeps_precision_and_tex_negative_sign(tmp_path):
    path = tmp_path / "means.txt"
    path.write_text("ExampleLow = \\ensuremath{-}0.86    % Source: example.json :: low\n"
                    "ExampleMean = 48.10\\ensuremath{^\\circ}\n")
    low, mean = parse_inputs(path)
    assert low["mean_value_in_paper"] == -.86
    assert low["paper_numeric_text"] == "-0.86"
    assert low["input_source_comment"] == "Source: example.json :: low"
    assert mean["paper_numeric_text"] == "48.10"
    assert mean["paper_precision"] == 2


def test_failure_selection_must_be_explicit():
    with pytest.raises(ValueError, match="select failures explicitly"):
        compute_statistics([1, np.nan, 30])


def test_manuscript_audit_uses_printed_semantics_and_adjacent_sources(tmp_path):
    paper = tmp_path / "sample.tex"
    paper.write_text(r"""% Source: errors.json :: standard_minus_delphi/mean_difference_deg
\newcommand{\FollowLowqFoundDifference}{-2.05}
% Source: errors.json :: latitude_bins/5/mean_error_deg
\newcommand{\InsightLowqFoundLatHigh}{20.50}
\newcommand{\ScPopulationMeanAbsBeta}{43}
\newcommand{\PeriodAliasMeanCount}{3.59}
\newcommand{\FollowAlcdefStandardLow}{\ensuremath{-}19.40}
\newcommand{\CountOneMean}{29.08}
\newcommand{\NoiseSmall}{17.07}
\begin{document}
\FollowLowqFoundDifference{} \InsightLowqFoundLatHigh{}
\ScPopulationMeanAbsBeta{} \PeriodAliasMeanCount{} \FollowAlcdefStandardLow{}
% This is not printed: \CountOneMean
""")
    items, audit = audit_manuscript(paper, [])
    assert {i["mean_macro"] for i in items} == {
        "FollowLowqFoundDifference", "InsightLowqFoundLatHigh", "FollowAlcdefStandardLow"}
    by_name = {r["macro"]: r for r in audit}
    assert not by_name["ScPopulationMeanAbsBeta"]["included"]
    assert not by_name["PeriodAliasMeanCount"]["included"]
    assert by_name["ScPopulationMeanAbsBeta"]["source_comment"] == ""
    assert "CountOneMean" not in by_name
    assert items[-1]["mean_value_in_paper"] == -19.40


def test_additional_bound_names_distinguish_confidence_limits_from_latitude_bins():
    class HandEvidence(Evidence):
        def additional_statistic(self, name, item):
            return {"mean_recomputed": 10., "median": 6., "interval": [2., 8.],
                    "mean_interval_recomputed": [3., 12.], "statistic": "difference_of_medians"}
    evidence = HandEvidence()
    low = evidence.resolve_additional({"mean_macro": "FollowAlcdefStandardLow"})
    assert low["median"] == 2
    assert low["mean_recomputed"] == 3
    assert low["parent_median"] == 6
    latitude = evidence.resolve_additional({"mean_macro": "InsightLowqFoundLatHigh"})
    assert latitude["median"] == 6
    assert latitude["mean_recomputed"] == 10


def test_reassignment_bounds_are_draw_quantiles_not_bootstrap_limits():
    evidence = Evidence()
    evidence.cache["CheckReassignMean"] = {
        "median": 6., "mean_recomputed": 11., "interval": None,
        "randomization_interval95": [5., 7.],
        "randomization_mean_interval_recomputed": [10., 12.],
    }
    result = evidence.resolve_additional({"mean_macro": "CheckReassignHigh"})
    assert result["median"] == 7
    assert result["mean_recomputed"] == 12
    assert result["interval"] is None
    assert result["parent_median"] == 6


def test_alcdef_positive_difference_reverses_both_arms_and_interval():
    evidence = Evidence()
    evidence.cache[OWN / "own-period-scores.json"] = {"alcdef": {
        "objects": [{"object_id": i, "own_error_deg": v} for i, v in zip("abc", [1, 10, 20])],
        "standard_minus_own_failed_at_90": {"ci95_low_deg": -1., "ci95_high_deg": 9.},
    }}
    evidence.cache[OWN / "nettests/no-uncertainty/score-own.json"] = {"variants": {"alcdef": {
        "objects": [{"object_id": i, "standard_error_deg": v} for i, v in zip("abc", [0, 19, 21])],
    }}}
    negative = evidence.pair_from_score("FollowAlcdefStandardDifference")
    positive = evidence.pair_from_score("FollowAlcdefStandardDifferencePositive")
    assert negative["mean_recomputed"] == 3
    assert negative["median"] == 9
    assert negative["median_of_paired_differences"] == 1
    assert positive["mean_recomputed"] == -3
    assert positive["median"] == -9
    assert positive["median_of_paired_differences"] == -1
    assert positive["interval"] == [-negative["interval"][1], -negative["interval"][0]]
    assert positive["mean_interval_in_source"] == [-9, 1]


@pytest.mark.parametrize("n", [3, 4])
def test_randomized_paired_bootstrap_equals_expanded_identity_resampling(n):
    values = np.array([[1., 2., 80., 100.], [12., 4., 8., 9.], [11., 5., 7., 13.]])[:, :n]
    own = np.array([2., 9., 10., 15.])[:n]
    indices = bootstrap_indices(n, 7, 19)
    result = randomized_paired_statistics(values, own, indices)
    expanded = [np.mean(np.median(values[:, draw], axis=1)) - np.median(own[draw]) for draw in indices]
    assert result["interval"] == pytest.approx(np.percentile(expanded, [2.5, 97.5]))
    assert result["median"] == pytest.approx(np.median(values, axis=1).mean() - np.median(own))
    assert result["median_of_paired_differences"] == pytest.approx(np.median(values - own, axis=1).mean())
