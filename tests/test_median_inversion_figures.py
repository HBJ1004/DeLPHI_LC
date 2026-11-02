"""Portable checks for inversion timing inputs, labels and censored costs."""

import json
from pathlib import Path

import pytest

from repro import network_root as network
from repro.own_period import median_figures as figures
from repro.own_period.median_figures import full_scan_costs, inversion_time_label


class TimingContext:
    check = figures.Context.check

    def __init__(self):
        self.result = {"points": [], "checks": []}
        self.read_paths = []

    def figure(self, *args, **kwargs):
        return self.result

    def read(self, path):
        self.read_paths.append(Path(path))
        return json.loads(Path(path).read_text())

    def sources_for(self, result, paths):
        result["source_files"] = list(map(str, paths))

    def save(self, fig, name, **kwargs):
        import matplotlib.pyplot as plt

        self.titles = [ax.get_title() for ax in fig.axes]
        self.annotations = [text.get_text() for ax in fig.axes for text in ax.texts]
        plt.close(fig)


@pytest.fixture
def timing_inputs(tmp_path, monkeypatch):
    for key in network.VARIABLES:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DELPHI_DATA", str(tmp_path / "fixed"))
    root = tmp_path / "artifact-root"
    score = {
        "classical_timing": "retime", "solver_workers": 6,
        "object_rows": [
            {"object_id": i, "arms": {
                "classical": {"mean_wall_seconds": seconds},
                "delphi": {"mean_wall_seconds": 1},
            }} for i, seconds in enumerate([0.5, 2])
        ],
        "point_estimates": {"speed_ratio_classical_over_delphi": 1.25},
        "bootstrap_95_intervals": {"speed_ratio_classical_over_delphi": [0.5, 2]},
    }
    for folder in [tmp_path / "fixed/benchmark",
                   root.parent / "results/delphi-inversions/timing-retime/own"]:
        folder.mkdir(parents=True)
        for name in ["score.json", "score-loaded.json"]:
            (folder / name).write_text(json.dumps(score))
    return root, score


@pytest.mark.parametrize("switched", [False, True])
@pytest.mark.parametrize("name", ["score.json", "score-loaded.json"])
def test_inversion_timing_selects_inputs(timing_inputs, monkeypatch, switched, name):
    root, score = timing_inputs
    if switched:
        monkeypatch.setenv("DELPHI_NETWORK_ROOT", str(root))
        expected = root.parent / "results/delphi-inversions/timing-retime/own" / name
    else:
        expected = root.parent / "fixed/benchmark" / name
        # Legacy scores need neither of the new retiming fields.
        score.pop("classical_timing")
        score.pop("solver_workers")
        expected.write_text(json.dumps(score))
    ctx = TimingContext()
    path, actual = figures.inversion_timing_score(ctx, ctx.result, name)
    assert Path(path) == expected
    assert ctx.read_paths == [expected]
    assert actual == score


@pytest.mark.parametrize("name", ["score.json", "score-loaded.json"])
@pytest.mark.parametrize(("field", "value", "message"), [
    ("classical_timing", "reuse", "classical timing is retimed"),
    ("solver_workers", 4, "solver workers"),
])
def test_inversion_timing_rejects_wrong_protocol(timing_inputs, monkeypatch, name, field, value, message):
    root, score = timing_inputs
    monkeypatch.setenv("DELPHI_NETWORK_ROOT", str(root))
    score[field] = value
    path = root.parent / "results/delphi-inversions/timing-retime/own" / name
    path.write_text(json.dumps(score))
    ctx = TimingContext()
    with pytest.raises(ValueError, match=message):
        figures.inversion_timing_score(ctx, ctx.result, name)


def test_inversion_timing_renders_printed_labels(timing_inputs, monkeypatch):
    root, _ = timing_inputs
    monkeypatch.setenv("DELPHI_NETWORK_ROOT", str(root))
    ctx = TimingContext()
    figures.inversion_timing(ctx)
    assert ctx.read_paths == [root.parent / "results/delphi-inversions/timing-retime/own" / name
                              for name in ["score.json", "score-loaded.json"]]
    assert ctx.titles == ["(a) Fresh process\nfor each asteroid", "(b) Networks kept loaded"]
    annotations = [text for text in ctx.annotations if text.startswith("DeLPHI faster for ")]
    assert len(annotations) == 2
    assert all("95% interval 0.50–2.00" in text for text in annotations)


@pytest.mark.integration
def test_shapley_printed_axis_label(tmp_path, monkeypatch):
    try:
        root = Path(figures.DATA).parent / "retrain-20261030/artifact-root"
    except ValueError:
        pytest.skip("published figure inputs are unavailable")
    if not (root / "model-binding.json").is_file():
        pytest.skip("retrained figure inputs are unavailable")
    monkeypatch.setenv("DELPHI_NETWORK_ROOT", str(root))
    monkeypatch.setenv("DELPHI_MODEL_BINDING", str(root / "model-binding.json"))
    monkeypatch.setenv("DELPHI_BUNDLE_ROOT", str(root / "bundles"))
    ctx = figures.Context(tmp_path)
    labels = []

    def save(fig, name, **kwargs):
        import matplotlib.pyplot as plt

        labels.extend(ax.get_xlabel() for ax in fig.axes)
        plt.close(fig)

    monkeypatch.setattr(ctx, "save", save)
    figures.shapley(ctx)
    assert labels[1] == "Change of oracle error (degrees, black = median)"


@pytest.mark.parametrize(("seconds", "label"), [
    (0, "0 s"), (2.618726634585445, "2.62 s"), (59, "59 s"), (60, "1 min"),
    (3600, "1 h"), (86400, "1 d"), (365.25 * 86400, "1 yr"),
])
def test_inversion_time_units(seconds, label):
    assert inversion_time_label(seconds) == label


@pytest.mark.parametrize("seconds", [-1, float("nan"), float("inf")])
def test_inversion_time_rejects_invalid_durations(seconds):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        inversion_time_label(seconds)


def test_inversion_time_annotation_precision():
    assert inversion_time_label(5.793711620831358 * 3600, significant=2) == "5.8 h"
    assert inversion_time_label(72.32961673733482 * 86400, significant=2) == "72 d"


def test_full_scan_costs_keeps_bounds_and_excludes_validation_objects():
    def estimate(**cpu):
        return {"P_core": {"cpu_seconds": cpu}}

    summary = {
        "estimates": {
            "first": estimate(estimated_full=13 * 86400),
            "bounded_middle": estimate(estimated_full_at_least=70 * 86400),
            "middle": estimate(estimated_full=74 * 86400, estimated_full_at_least=1),
            "bounded_maximum": estimate(estimated_full_at_least=346 * 86400),
            "validation": estimate(estimated_full=1),
        },
        "full_scan_checks": {"validation": {}},
    }
    result = full_scan_costs(summary)
    assert result["n"] == 4
    assert result["object_ids"] == ["first", "bounded_middle", "middle", "bounded_maximum"]
    assert result["seconds"] == 72 * 86400
    assert result["range_seconds"] == [13 * 86400, 346 * 86400]
    assert result["range_upper_is_lower_bound"] is True
    assert [row["lower_bound"] for row in result["per_object"]] == [False, True, False, True]
