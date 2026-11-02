"""Reporting path, provenance, aggregation and plotted-array contracts."""

import json
from pathlib import Path

import numpy as np
import pytest

from lc_pipeline.k3.protocol import K3_PROTOCOL_SHA256
from repro import network_root as network
from repro.own_period import macro_table, mean_macros, period_drift
from repro.own_period import reporting_inputs as inputs
from repro.own_period.figure_data import capture_figures


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def selected(tmp_path, monkeypatch):
    for key in network.VARIABLES:
        monkeypatch.delenv(key, raising=False)
    root = tmp_path / "artifact-root"
    records = [
        dict(filename=f"real-fold-{f}-seed-{s}.pt", sha256="0" * 64, size_bytes=1)
        for f in range(5)
        for s in network.SEEDS
    ]
    dump(
        root / "model-binding.json",
        dict(schema=network.LOCAL_SCHEMA, protocol_sha256=K3_PROTOCOL_SHA256, models=records),
    )
    monkeypatch.setenv("DELPHI_NETWORK_ROOT", str(root))
    monkeypatch.setenv("DELPHI_DATA", str(tmp_path / "fixed/data/own-period-20261021"))
    return root


@pytest.mark.parametrize(
    "old,new",
    [
        (
            "maps/own-period/evaluations/real-oof-ensemble.npz",
            "headline/maps/own-period/evaluations/real-oof-ensemble.npz",
        ),
        ("benchmark/score.json", "delphi-inversions/own/score.json"),
        ("../direct-workflow-benchmark/score.json", "delphi-inversions/damit/score.json"),
        (
            "lowq/scores-own/direction-excluded.json",
            "lower-quality-census/scores-own/direction-excluded.json",
        ),
        ("nettests/shapley-own-summary.json", "shapley/own-summary.json"),
        ("nettests/train-vs-test-own-report.json", "internal-validation/own-report.json"),
        (
            "nettests/controls-own/real-fold-0-seed-17-input-swap.npz",
            "controls-input-swap/own/real-fold-0-seed-17-input-swap.npz",
        ),
        (
            "nettests/controls-own/real-fold-0-seed-17-amplitude.npz",
            "input-removals/own/real-fold-0-seed-17-amplitude.npz",
        ),
        ("v67/v7-own-period-score.json", "delphi-inversions/v7-own-score.json"),
        ("paired-gains.json", "paired-gains.json"),
    ],
)
def test_reporting_paths_use_existing_runner_contracts(selected, old, new):
    assert Path(inputs.DataPath(old)) == selected.parent / "results" / new


def test_paths_are_lazy_and_fixed_periods_stay_fixed(selected, monkeypatch):
    path = inputs.DataPath("median-stats.json")
    assert Path(path) == selected.parent / "results/median-stats.json"
    monkeypatch.delenv("DELPHI_NETWORK_ROOT")
    assert Path(path) == Path(inputs.FixedDataPath("median-stats.json"))
    monkeypatch.setenv("DELPHI_NETWORK_ROOT", str(selected))
    assert Path(inputs.DataPath("own-periods-170.json")) == Path(
        inputs.FixedDataPath("own-periods-170.json")
    )


@pytest.mark.parametrize("container", ["dict", "rows", "sealed"])
def test_reporting_json_refuses_mixed_networks(selected, container):
    path = selected.parent / "results/test.json"
    row = dict(value=12, network_binding_sha256="wrong")
    if container == "sealed":
        from lc_pipeline.k3.reliability_study import seal

        seal(path, row)
    else:
        dump(path, row if container == "dict" else [row])
    with pytest.raises(ValueError, match="network binding"):
        inputs.read_json(path)


@pytest.mark.parametrize("container", ["dict", "rows", "sealed"])
def test_reporting_json_accepts_selected_binding(selected, container):
    path = selected.parent / "results/test.json"
    row = dict(value=12, **network.json_metadata())
    if container == "sealed":
        from lc_pipeline.k3.reliability_study import seal

        seal(path, row)
    else:
        dump(path, row if container == "dict" else [row])
    assert inputs.read_json(path)


def test_npz_override_outside_results_still_needs_selected_binding(selected, tmp_path):
    path = tmp_path / "override.npz"
    np.savez(path, oracle_errors_deg=[1, 2], network_binding_sha256=np.asarray("wrong"))
    with pytest.raises(ValueError, match="network binding"):
        inputs.load_npz(path)
    np.savez(path, oracle_errors_deg=[1, 2], **network.network_metadata())
    with inputs.load_npz(path) as values:
        np.testing.assert_array_equal(values["oracle_errors_deg"], [1, 2])


def test_period_drift_uses_complete_observed_span_and_rounded_periods(tmp_path):
    curve = tmp_path / "lc.txt"
    curve.write_text("1\n2 0\n100 1 1 0 0 1 0 0\n110 1 1 0 0 1 0 0\n")
    lightcurve = dict(path=str(curve), sha256=network.sha256(curve))
    own, damit = tmp_path / "own.json", tmp_path / "damit.json"
    dump(own, dict(objects=[dict(object_id="a", period_hours=4.0, lightcurve=lightcurve)]))
    dump(damit, dict(objects=[dict(object_id="a", period_hours=5.0, lightcurve=lightcurve)]))
    result = period_drift.build(own, damit)
    assert result["median_relative_difference_percent"] == 20
    assert result["median_drift_rotations"] == pytest.approx(12)
    assert result["objects"][0]["span_days"] == 10
    curve.write_text(curve.read_text() + "garbage")
    with pytest.raises(ValueError, match="hash mismatch"):
        period_drift.build(own, damit)


def test_mean_scope_retains_unprinted_exclusions_and_m1b(tmp_path):
    source, manuscript = tmp_path / "saved.json", tmp_path / "sample.tex"
    row = dict(
        mean_macro="ExampleMean",
        paper_numeric_text="12.30",
        input_source_comment="Source: example.json :: mean",
    )
    extra = dict(
        mean_macro="ExtraMean", paper_numeric_text="3.0", input_source_comment="", new_in_m1b=True
    )
    skipped = dict(mean_macro="RemovedMean", paper_numeric_text="2.0", input_source_comment="")
    dump(source, dict(entries=[row, extra], skipped=[skipped]))
    manuscript.write_text(
        "\\newcommand{\\ExampleMean}{12.30} story\n".replace(" story", "")
        + "\\newcommand{\\ExtraMean}{3.0}\n"
    )
    output = mean_macros.build(source, manuscript)
    assert "ExampleMean = 12.30    % Source: example.json :: mean" in output
    assert "ExtraMean = 3.0" in output
    assert "RemovedMean = 2.0" in output


def test_macro_lookup_preserves_slashes_inside_pair_names():
    assert (
        macro_table.lookup(
            {"pairs": {"refined/a minus own/a": {"ci": [1, 4]}}}, "pairs/refined/a minus own/a/ci/1"
        )
        == 4
    )


def test_macro_display_preserves_sign_precision_and_significant_rounding():
    assert macro_table.display_value(2.016, "+2.00", "Example") == "+2.02"
    assert macro_table.display_value(-0.0001, "0.00", "Example") == "0.00"
    assert (
        macro_table.display_value(640951.68, "640{,}000", "WorkflowFullSearchSpeedFactor")
        == "640000"
    )


def test_macro_percentage_formula_does_not_rescale_native_percent():
    class Hand(macro_table.Values):
        def raw(self, macro):
            return macro["fixture"]

    values = Hand()
    assert (
        values.resolve(
            dict(name="CountOneWithin", fixture=0.25, value=r"25\%", source_field="k/0/within_20")
        )
        == 25
    )
    assert (
        values.resolve(
            dict(
                name="FollowLowqFoundWithinTwenty",
                fixture=25,
                value=r"25\%",
                source_field="delphi/within_twenty_percent",
            )
        )
        == 25
    )
    assert (
        values.resolve(
            dict(name="WorkflowColdTimeFactor", fixture=0.4, value="2.5", source_field="ratio")
        )
        == 2.5
    )


def test_saved_plot_arrays_keep_missing_points_explicit(tmp_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [1, np.nan, 3])
    target = tmp_path / "example.pdf"
    with capture_figures():
        fig.savefig(target)
    plt.close(fig)
    arrays = json.loads(target.with_suffix(".arrays.json").read_text())["arrays"]
    assert arrays["axes"][0]["lines"][0]["y"] == [1, None, 3]


def test_plot_capture_refuses_manuscript_output(tmp_path):
    import matplotlib.pyplot as plt

    fig, _ = plt.subplots()
    target = tmp_path / "paper-rev-20261021/example.pdf"
    with capture_figures(), pytest.raises(ValueError, match="manuscript"):
        fig.savefig(target)
    plt.close(fig)
    assert not target.exists()


def test_source_comment_dispatch_uses_original_path_before_translation(selected):
    from repro.own_period.median_stats import source_path

    source = source_path(
        "Source: DeLPHI-followup/data/own-period-20261021/benchmark/score.json :: point_estimates/delphi_mean_error_degrees"
    )
    assert isinstance(source, Path)
    assert "benchmark" in str(source)
    assert "delphi-inversions" not in str(source)
