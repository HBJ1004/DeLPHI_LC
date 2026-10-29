"""CPU checks of census scoring and the reference-period boundary."""

import json

import numpy as np
import pandas as pd
import pytest

from repro.own_period import census7 as census
from repro.own_period.score import axial_error, period_class


def test_axial_oracle_and_period_aliases():
    assert axial_error([[0, 1, 0], [-2, 0, 0], [0, 0, 1]], np.array([[1, 0, 0]])) == 0
    assert axial_error([[1, 0, 0]], np.array([[0, 1, 0]])) == 90
    assert [period_class(p, 10) for p in (10.09, 5.04, 20.1, 13, None)] == [
        "P",
        "P/2",
        "2P",
        "other",
        "failed",
    ]
    result = census.metric_summary([{"error": v} for v in (0, 20, 21, None)], "error")
    assert result == {
        "n": 4,
        "n_scored": 3,
        "n_failed": 1,
        "median_deg": 20,
        "number_within_20_deg": 2,
    }
    assert census.identity_matches("asteroid_2772/model_1", {"asteroid_2772"})
    assert not census.identity_matches("mpc:2772", {"asteroid_2772"})


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    # Seven synthetic objects with a reference pole on x; no real reference data.
    rows = [
        {"object_id": oid, "mpc_number": mpc, "name": name, "cohort": cohort}
        for oid, mpc, name, cohort in census.OBJECTS
    ]
    census.write(tmp_path / "prepared-manifest.json", {"objects": rows})
    census.write(
        tmp_path / "exposure.json",
        {"objects": [{"object_id": r["object_id"], "exposed": False} for r in rows]},
    )
    manifest = {"objects": rows, "exposure": census.binding(tmp_path / "exposure.json")}
    monkeypatch.setattr(census, "context", lambda out: manifest)
    monkeypatch.setattr(census, "validated", lambda out, manifest: None)
    prediction = {
        r["object_id"]: {
            "status": "ok",
            "period_hours": 10,
            "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        }
        for r in rows
    }
    prediction[rows[-1]["object_id"]] = {"status": "no_period", "period_hours": None}
    for networks in ("fold0", "all25"):
        path = census.prediction_path(tmp_path, "own", networks)
        path.write_text(
            "\n".join(json.dumps({"object_id": oid, **p}) for oid, p in prediction.items()) + "\n"
        )
        census.write(
            path.with_suffix(".manifest.json"),
            {
                "periods": "own",
                "networks": networks,
                "predictions": census.binding(path),
                "period_input": census.binding(tmp_path / "prepared-manifest.json"),
                "prepared_manifest_sha256": census.sha256(tmp_path / "prepared-manifest.json"),
            },
        )
    frame = pd.DataFrame(
        [
            {
                "id": i,
                "asteroid_id": int(row["object_id"].split("_")[1]),
                "quality_flag": 3,
                "lambda": 0,
                "beta": 0,
            }
            for i, row in enumerate(rows)
        ]
    )

    def poles_only(manifest, *, periods):
        assert periods is False, "own scoring attempted to read DAMIT periods"
        return frame

    monkeypatch.setattr(census, "reference_table", poles_only)
    monkeypatch.setattr(
        census,
        "reference_periods",
        lambda manifest: pytest.fail("reference periods read too early"),
    )
    return tmp_path, prediction


def test_own_score_sealed_periods_and_final_adds_secondary(workflow, monkeypatch):
    out, predictions = workflow
    census.score(out, "own")
    own_bytes = (out / "scores-own.json").read_bytes()
    own = census.read(out / "scores-own.json")
    assert own["summaries"]["all7"]["delphi_own_error_deg"] == {
        "n": 7,
        "n_scored": 6,
        "n_failed": 1,
        "median_deg": 0,
        "number_within_20_deg": 6,
    }
    assert own["summaries"]["new3"]["delphi_all25_own_error_deg"]["n_scored"] == 3
    assert all("damit_period_hours" not in r and "period_class" not in r for r in own["objects"])
    assert not (out / "scores.json").exists()
    assert "FAILED" in (out / "RESULTS-own.md").read_text()

    path = census.prediction_path(out, "damit", "fold0")
    path.write_text(
        "\n".join(json.dumps({"object_id": oid, **p}) for oid, p in predictions.items()) + "\n"
    )
    census.write(
        path.with_suffix(".manifest.json"),
        {
            "periods": "damit",
            "networks": "fold0",
            "predictions": census.binding(path),
            "period_input": census.binding(out / "scores-own.json"),
            "prepared_manifest_sha256": census.sha256(out / "prepared-manifest.json"),
        },
    )
    monkeypatch.setattr(
        census, "reference_periods", lambda manifest: {r[0]: 10 for r in census.OBJECTS}
    )
    census.score(out, "final")
    final = census.read(out / "scores.json")
    assert (out / "scores-own.json").read_bytes() == own_bytes
    assert final["objects"][0]["period_class"] == "P"
    assert final["objects"][-1]["period_class"] == "failed"
    assert all(
        all(r[k] == value for k, value in old.items())
        for r, old in zip(final["objects"], own["objects"], strict=True)
    )


def test_damit_prediction_requires_written_own_score(workflow, monkeypatch):
    out, _ = workflow
    monkeypatch.setattr(
        census, "load_models", lambda *args: pytest.fail("GPU touched before own scoring")
    )
    with pytest.raises(FileNotFoundError, match="scores-own.json"):
        census.predict(out, "damit", "fold0", "cuda")
    with pytest.raises(ValueError, match="fold0 only"):
        census.predict(out, "damit", "all25", "cuda")


def test_changed_own_predictions_cannot_unlock_damit(workflow):
    out, _ = workflow
    census.score(out, "own")
    with census.prediction_path(out, "own", "fold0").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        census.own_score_ready(out)


def test_own_reference_reader_excludes_period_column(monkeypatch, tmp_path):
    path = tmp_path / "reference.csv"
    path.write_text("synthetic sealed reference")
    manifest = {
        "sealed_inputs": {str(census.REFERENCE.relative_to(census.FOLLOWUP)): census.binding(path)}
    }
    calls = []

    def fake_read(path, *, usecols):
        calls.append(usecols)
        assert "period" not in usecols
        return pd.DataFrame(
            {
                "id": [3, 2, 1],
                "asteroid_id": [1, 1, 1],
                "quality_flag": [3, 2, 4],
                "lambda": [0, 0, 0],
                "beta": [0, 0, 0],
            }
        )

    monkeypatch.setattr(pd, "read_csv", fake_read)
    result = census.reference_table(manifest, periods=False)
    assert result.id.tolist() == [1, 3]
    assert len(calls) == 1


def test_validation_gate_rejects_failure_and_different_preparation(tmp_path):
    census.write(tmp_path / "prepared-manifest.json", {"synthetic": True})
    census.write(tmp_path / "validation.json", {"passed": False})
    with pytest.raises(ValueError, match="validation has not passed"):
        census.validated(tmp_path, {})


def test_curve_and_epoch_round_trip(tmp_path):
    path = tmp_path / "lc.txt"
    path.write_text("1\n2 0\n2450001 2 1 0 0 0 1 0\n2450000 1 1 0 0 0 1 0\n")
    prepared = tmp_path / "prepared.json"
    epochs = census.prepare_lightcurve(path, prepared)
    assert census.epochs_of(prepared) == epochs
    (curve,) = census.curves_of(prepared)
    assert np.allclose(curve.time, [2450000, 2450001])
    assert np.allclose(curve.magnitude, [0, -2.5 * np.log10(2)])


@pytest.mark.parametrize(("difference", "expected"), [(0, True), (1e-12, True), (1e-3, False)])
def test_validation_uses_prepared_path_and_requires_matching_predictions(
    tmp_path, monkeypatch, difference, expected
):
    snapshot = tmp_path / "snapshot"
    source = snapshot / "files/asteroid_1/lc.txt"
    source.parent.mkdir(parents=True)
    source.write_text("1\n2 0\n2450000 1 1 0 0 0 1 0\n2450001 2 1 0 0 0 1 0\n")
    prepared = tmp_path / "prepared-validation/asteroid_1.json"
    epochs = census.prepare_lightcurve(source, prepared)
    census.write(tmp_path / "prepared-manifest.json", {"synthetic": True})
    index = tmp_path / "index.json"
    census.write(
        index,
        {
            "objects": [
                {
                    "object_id": "damit:1",
                    "lightcurve_path": "files/asteroid_1/lc.txt",
                    "known_period_hours": 10,
                    "period_provenance": "synthetic validation",
                }
            ]
        },
    )
    published = tmp_path / "published.jsonl"
    published.write_text(
        json.dumps(
            {
                "object_id": "damit:1",
                "status": "ok",
                "epoch_count": 1,
                "observation_count": 2,
                "lightcurve_sha256": census.sha256(source),
                "axes": [{"axis_xyz": [1, 0, 0], "score": 0.5}],
            }
        )
        + "\n"
    )
    manifest = {
        "validation_index": census.binding(index),
        "validation_predictions": census.binding(published),
        "validation_inputs": [
            {
                "object_id": "asteroid_1",
                "raw": census.binding(source),
                "prepared": census.binding(prepared),
            }
        ],
    }
    monkeypatch.setattr(census, "OLD_SNAPSHOT", snapshot)
    monkeypatch.setattr(census, "context", lambda out: manifest)
    monkeypatch.setattr(census, "load_models", lambda *args: ([], []))

    def fake_prediction(models, received, period, device):
        assert received == epochs
        assert period.hours == 10
        return {
            "status": "ok",
            "axes": [[1, difference, 0]],
            "scores": [0.5],
            "epoch_count": 1,
            "observation_count": 2,
        }

    monkeypatch.setattr(census, "prediction", fake_prediction)
    # 1e-3 in one component is about 0.057 deg, above the 0.01 deg tolerance.
    if not expected:
        with pytest.raises(ValueError, match="validation differs"):
            census.validate(tmp_path, "cuda")
    else:
        census.validate(tmp_path, "cuda")
    assert census.read(tmp_path / "validation.json")["passed"] is expected
