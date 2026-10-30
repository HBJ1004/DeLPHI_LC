"""Period estimation and JSON prediction command tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lc_pipeline.k3.cli import main as search_main
from lc_pipeline.k3.predict import main, read_observations
from lc_pipeline.k3.tokenizer import tokenize_epochs
from repro.period_search import search

ASTRAEA = Path(__file__).resolve().parents[1] / "examples/astraea/observations.json"


@pytest.fixture(scope="module")
def astraea_search(tmp_path_factory):
    directory = tmp_path_factory.mktemp("astraea-period")
    observations = directory / "observations.json"
    document = json.loads(ASTRAEA.read_text())
    del document["known_period"]
    observations.write_text(json.dumps(document), encoding="utf-8")
    output = directory / "period.json"
    assert search_main([
        "estimate-period", str(observations), "--device", "cpu", "--output", str(output)
    ]) == 0
    return observations, output, json.loads(output.read_text())


def test_estimate_period_astraea_cpu(astraea_search) -> None:
    _, _, result = astraea_search
    assert result["schema"] == "delphi.k3-period.v1"
    assert result["status"] == "ok"
    assert result["object_id"] == "asteroid_103"
    assert result["method"] == "fourier_cpu" and result["device"] == "cpu"
    assert result["settings"] == {"resolution": 1, "timeout": 180, "sparse_fallback": True}
    assert result["period_hours"] == result["ranked_periods"][0]
    assert result["period_hours"] == pytest.approx(16.79973752943218, abs=1e-7)
    assert result["provenance"] == "DeLPHI Fourier search, rank 1"
    assert "roundoff" in result["backend_note"]


def test_estimated_period_loads_into_prediction(astraea_search, tmp_path, monkeypatch) -> None:
    observations, period_path, estimate = astraea_search
    calls = []

    def predict(epochs, *, known_period, object_id):
        tokens = tokenize_epochs(epochs, known_period=known_period)
        assert tokens.period_hours == estimate["period_hours"]
        assert tokens.period_provenance == estimate["provenance"]
        calls.append(object_id)
        return {"object_id": object_id, "period_provenance": known_period.provenance}

    monkeypatch.setattr("lc_pipeline.k3.predict.load_predictor",
                        lambda *args, **kwargs: SimpleNamespace(predict=predict))
    output = tmp_path / "prediction.json"
    assert main([
        "--bundle", str(tmp_path / "bundle"), "--input", str(observations),
        "--period-json", str(period_path), "--output", str(output),
    ]) == 0
    assert calls == ["asteroid_103"]
    assert json.loads(output.read_text())["period_provenance"] == estimate["provenance"]
    # An estimate also overrides the supplied DAMIT period in the worked example.
    _, period, _ = read_observations(ASTRAEA, period_json=period_path)
    assert period.hours == estimate["period_hours"] != 16.80059


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_astraea_cpu_gpu_period_equivalence(astraea_search, tmp_path) -> None:
    observations, _, cpu = astraea_search
    output = tmp_path / "gpu-period.json"
    assert search_main([
        "estimate-period", str(observations), "--device", "cuda", "--output", str(output)
    ]) == 0
    gpu = json.loads(output.read_text())
    assert gpu["device"] == "cuda" and gpu["method"] == "fourier_gpu"
    assert gpu["period_hours"] == pytest.approx(cpu["period_hours"], rel=1e-8, abs=1e-8)


@pytest.mark.parametrize("device,available,method", [
    ("cpu", False, "fourier_cpu"),
    ("auto", False, "fourier_cpu"),
    ("auto", True, "fourier_gpu"),
])
def test_search_matches_damit_conversion_and_settings(
    tmp_path, monkeypatch, capsys, device, available, method
) -> None:
    document = json.loads(ASTRAEA.read_text())
    document["known_period"] = {"must_not_be_read": True}
    rows = []
    for epoch in document["epochs"]:
        epoch["observations"].reverse()  # Both readers must sort within a curve.
        rows.append(f"{len(epoch['observations'])} 0")
        for observation in epoch["observations"]:
            rows.append(" ".join(str(value) for value in (
                observation["time_jd"], observation["relative_brightness"],
                *observation["sun_asteroid_ecliptic_j2000_au"],
                *observation["observer_asteroid_ecliptic_j2000_au"],
            )))
    lc = tmp_path / "lc.txt"
    lc.write_text(f"{len(document['epochs'])}\n" + "\n".join(rows), encoding="utf-8")
    expected = search.read_curves(lc)
    observations = tmp_path / "observations.json"
    observations.write_text(json.dumps(document), encoding="utf-8")
    calls = []

    def estimate(curves, selected_method, **settings):
        assert len(curves) == len(expected) == 25
        for actual, reference in zip(curves, expected):
            np.testing.assert_array_equal(actual.time, reference.time)
            np.testing.assert_array_equal(actual.magnitude, reference.magnitude)
        calls.append((selected_method, settings))
        return {"status": "ok", "ranked_periods": [16.8]}

    monkeypatch.setattr(search, "estimate", estimate)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    assert search_main(["estimate-period", str(observations), "--device", device]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["method"] == method
    assert calls == [(method, {"resolution": 1, "timeout": 180, "sparse_fallback": True})]


@pytest.mark.parametrize("failure", ["constant", "too_few", "timeout"])
def test_period_search_failure_exits_without_output(tmp_path, monkeypatch, capsys, failure) -> None:
    document = json.loads(ASTRAEA.read_text())
    if failure == "constant":
        for epoch in document["epochs"]:
            for observation in epoch["observations"]:
                observation["relative_brightness"] = 1.0
    elif failure == "too_few":
        document["epochs"] = document["epochs"][:1]
        document["epochs"][0]["observations"] = document["epochs"][0]["observations"][:2]
    else:
        def timeout(*args, **kwargs):
            raise TimeoutError("photometric search budget exhausted")
        monkeypatch.setattr(search, "estimate", timeout)
    observations = tmp_path / "observations.json"
    observations.write_text(json.dumps(document), encoding="utf-8")
    output = tmp_path / "period.json"
    with pytest.raises(SystemExit) as error:
        search_main(["estimate-period", str(observations), "--device", "cpu",
                     "--output", str(output)])
    assert error.value.code != 0 and not output.exists()
    message = capsys.readouterr().err
    assert {
        "constant": "constant brightness",
        "too_few": "no short observing interval with 15 points",
        "timeout": "photometric search budget exhausted",
    }[failure] in message
    assert "Traceback" not in message


def test_search_refuses_overwrite_and_unavailable_cuda(tmp_path, monkeypatch, capsys) -> None:
    output = tmp_path / "period.json"
    output.write_text("keep", encoding="utf-8")
    with pytest.raises(SystemExit):
        search_main(["estimate-period", str(ASTRAEA), "--device", "cpu",
                     "--output", str(output)])
    assert output.read_text() == "keep"
    assert "output already exists" in capsys.readouterr().err
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit):
        search_main(["estimate-period", str(ASTRAEA), "--device", "cuda"])
    assert "CUDA is unavailable" in capsys.readouterr().err


@pytest.mark.parametrize("change,message", [
    ({"object_id": "another-asteroid"}, "object_id does not match"),
    ({"schema": "unknown"}, "period JSON schema"),
    ({"status": "unresolved"}, "successful search"),
    ({"period_hours": -1.0}, "period hours"),
])
def test_prediction_rejects_invalid_period_json(tmp_path, change, message) -> None:
    estimate = {
        "schema": "delphi.k3-period.v1", "status": "ok", "object_id": "asteroid_103",
        "period_hours": 16.8, "provenance": "DeLPHI Fourier search, rank 1", **change,
    }
    path = tmp_path / "period.json"
    path.write_text(json.dumps(estimate), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_observations(ASTRAEA, period_json=path)


def test_read_observations_names_missing_required_field(tmp_path) -> None:
    path = tmp_path / "observations.json"
    path.write_text(
        json.dumps(
            {
                "schema": "delphi.k3-observations.v1",
                "object_id": "example",
                "epochs": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing required observation field: known_period"):
        read_observations(path)


def test_cli_reports_validation_error_without_traceback(tmp_path, capsys) -> None:
    path = tmp_path / "observations.json"
    path.write_text(
        json.dumps({"schema": "delphi.k3-observations.v1", "object_id": "example"}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--bundle",
                str(tmp_path / "bundle"),
                "--input",
                str(path),
                "--output",
                str(tmp_path / "prediction.json"),
            ]
        )
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "missing required observation field: known_period" in captured.err
    assert "Traceback" not in captured.err
