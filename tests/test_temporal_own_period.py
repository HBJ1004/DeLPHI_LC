"""CPU-only contracts for the own-period temporal runner."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from repro.own_period import temporal as t


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def prediction(oid):
    return {"schema": t.PREDICTION_SCHEMA, "object_id": oid, "policy": t.POLICY,
            "model_count": 25, "folds": list(range(5)),
            "checkpoint_bindings": [{"filename": f"model-{i}", "sha256": str(i)} for i in range(25)],
            "model_manifest_sha256": "fixture-model-manifest",
            "axes": [{"axis_xyz": axis, "score": 0.0, "grid_index": i}
                     for i, axis in enumerate(np.eye(3).tolist())]}


@pytest.fixture
def inputs(tmp_path):
    args = SimpleNamespace(
        output_dir=tmp_path / "own", prepared_dir=tmp_path / "original",
        published_dir=tmp_path / "published", published_scores_dir=tmp_path / "published-scores",
        references=tmp_path / "references.json", models=tmp_path / "models",
        model_manifest=write(tmp_path / "models.json", {}),
        splits=write(tmp_path / "splits.json", {}), device="cuda",
        damit_periods=False, check_dir=None)
    rows, refs = [], {}
    for i, oid in enumerate(t.OBJECT_IDS):
        value = {"schema": "delphi.k3-external-prepared.v1", "object_id": oid,
                 "known_period_hours": 10.0 + i, "period_provenance": "published",
                 "source": {"unchanged": True},
                 "epochs": [{"epoch_id": "e0", "observations": [
                     {"time_jd": 2450000.0 + j, "relative_brightness": brightness,
                      "measured_error": 0.05, "sun_asteroid_ecliptic_j2000_au": [1., 0., 0.],
                      "observer_asteroid_ecliptic_j2000_au": [0., 1., 0.]}
                     for j, brightness in enumerate((1.0, 2.0, 0.5))]}]}
        path = write(args.prepared_dir / "objects" / f"{oid}.json", value)
        rows.append({"object_id": oid, "path": f"objects/{oid}.json", "sha256": t.sha256(path)})
        pred_path = write(args.published_dir / f"{oid}.json", prediction(oid))
        refs[oid] = [[1.0, 1.0, 1.0]]
        write(args.published_scores_dir / f"{oid}.json", {
            "schema": "delphi.k3-ztf-external-score.v1", "object_id": oid,
            "prediction_sha256": t.sha256(pred_path),
            "oracle_at_3_error_deg": t.oracle_error(prediction(oid), refs[oid])})
    write(args.references, refs)
    write(args.prepared_dir / "manifest.json", {
        "schema": "delphi.k3-temporal-prepared-manifest.v1", "objects": rows})
    return args


def own_file(args, periods=None):
    return write(args.output_dir / "own-periods-temporal.json", {
        "schema": "delphi.own-periods.v1", "cohort": "temporal", "provenance": t.PROVENANCE,
        "prepared_manifest_sha256": t.sha256(args.prepared_dir / "manifest.json"),
        "search_source_sha256": t.sha256(t.REPO / "repro/period_search/search.py"),
        "periods_hours": periods or dict(zip(t.OBJECT_IDS, (8.0, None, 12.5))), "details": {}})


@pytest.fixture
def fake_predictor(monkeypatch):
    from lc_pipeline.k3 import ztf_prediction

    calls = []

    def predict(prepared_path, models, splits, output_path, **kwargs):
        value = t.read(prepared_path)
        calls.append((value, models, splits, kwargs))
        assert kwargs["policy"] == "post_cutoff_damit_record"
        # The frozen route really loads these epochs and KnownPeriod.
        oid, period, epochs = ztf_prediction._load_prepared(prepared_path)
        assert period.hours == value["known_period_hours"] and len(epochs) == 1
        row = prediction(oid)
        row["prepared_sha256"] = t.sha256(prepared_path)
        write(output_path, row)
        return row

    monkeypatch.setattr(ztf_prediction, "predict_prepared_ztf_object", predict)
    monkeypatch.setattr(t, "require_gpu_lock", lambda: None)
    return calls


def checked_run(inputs):
    args = deepcopy(inputs)
    args.output_dir = inputs.output_dir.parent / "check"
    args.damit_periods = True
    t.command_predict(args)
    return args.output_dir


def test_search_uses_exact_curves_and_settings_without_reference_periods(inputs, monkeypatch):
    from repro.period_search import search

    # Poison the published period; search must not interpret it or open references.
    manifest_path = inputs.prepared_dir / "manifest.json"
    manifest = t.read(manifest_path)
    for row in manifest["objects"]:
        path = inputs.prepared_dir / row["path"]
        value = t.read(path)
        value["known_period_hours"] = {"must_not_be_used": True}
        write(path, value)
        row["sha256"] = t.sha256(path)
    write(manifest_path, manifest)
    inputs.references.unlink()
    calls = []

    def estimate(curves, method, **kwargs):
        calls.append((method, kwargs))
        assert len(curves) == 1
        np.testing.assert_array_equal(curves[0].time, [2450000., 2450001., 2450002.])
        np.testing.assert_allclose(curves[0].magnitude, -2.5 * np.log10([1., 2., .5]))
        if len(calls) == 2:
            return {"status": "unresolved", "ranked_periods": []}
        if len(calls) == 3:
            raise TimeoutError("fixture timeout")
        return {"status": "ok", "ranked_periods": [8.25, 16.5, 4.125], "input_regime": "short_intervals"}

    monkeypatch.setattr(search, "estimate", estimate)
    monkeypatch.setattr(t, "require_gpu_lock", lambda: None)
    assert t.command_search(inputs) == 0
    value = t.load_own_periods(inputs.output_dir / "own-periods-temporal.json", inputs.prepared_dir)
    assert value["schema"] == "delphi.own-periods.v1" and value["cohort"] == "temporal"
    assert value["periods_hours"] == dict(zip(t.OBJECT_IDS, (8.25, None, None)))
    assert value["details"][t.OBJECT_IDS[2]]["status"] == "timeout"
    assert calls == [("fourier_gpu", {"resolution": 1, "timeout": 180, "sparse_fallback": True})] * 3
    with pytest.raises(FileExistsError):
        t.command_search(inputs)
    assert len(calls) == 3


def test_period_manifest_and_preparation_keep_every_observation(inputs):
    periods = dict(zip(t.OBJECT_IDS, (8., None, 12.5)))
    manifest = t.period_manifest(periods, "a" * 64, t.PROVENANCE)
    assert manifest["schema"] == "delphi.k3-mapped-ztf-period-manifest.v1"
    assert manifest["failed_object_ids"] == ["asteroid_279"]
    assert manifest["pole_coordinates_exported"] is False
    assert [(r["object_id"], r["mpc_number"], r["held_out_fold"], r["known_period_hours"])
            for r in manifest["objects"]] == [("asteroid_49", 49, None, 8.), ("asteroid_366", 366, None, 12.5)]
    t.save_new({inputs.output_dir / "period-manifest.json": manifest})
    _, paths = t.source_objects(inputs.prepared_dir)
    result = t.prepare_inputs(inputs.output_dir / "prepared", paths, manifest)
    assert len(result["objects"]) == 2
    for row in result["objects"]:
        original = t.read(paths[row["object_id"]])
        prepared = t.read(inputs.output_dir / "prepared" / row["path"])
        original["known_period_hours"] = periods[row["object_id"]]
        original["period_provenance"] = t.PROVENANCE
        assert prepared == original
    with pytest.raises(FileExistsError):
        t.prepare_inputs(inputs.output_dir / "prepared", paths, manifest)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True, "10"])
def test_manifest_rejects_invalid_periods(bad):
    periods = dict.fromkeys(t.OBJECT_IDS, 8.)
    periods[t.OBJECT_IDS[0]] = bad
    with pytest.raises(ValueError, match="valid periods"):
        t.period_manifest(periods, "x", t.PROVENANCE)


def test_published_predictions_reproduce_saved_scores():
    # External artifacts are optional in a standalone checkout, mandatory here.
    if not t.PUBLISHED.is_dir():
        pytest.skip("published follow-up artifacts are unavailable")
    references = t.read(t.REFERENCES)
    predictions = t.published_predictions(t.PUBLISHED)
    errors = []
    for oid, displayed in zip(t.OBJECT_IDS, (24.92, 17.88, 36.28)):
        saved = t.read(t.PUBLISHED_SCORES / f"{oid}.json")
        error = t.oracle_error(predictions[oid], references[oid])
        assert t.sha256(t.PUBLISHED / f"{oid}.json") == saved["prediction_sha256"]
        assert error == pytest.approx(saved["oracle_at_3_error_deg"], abs=1e-10)
        assert round(error, 2) == displayed
        errors.append(error)
    assert round(t.summary(errors)["mean_deg"], 2) == 26.36


def test_oracle_uses_all_three_axes_all_reference_axes_and_axial_sign():
    value = prediction("asteroid_49")
    value["axes"] = [{"axis_xyz": [0., 1., 0.]}, {"axis_xyz": [0., 0., 1.]}, {"axis_xyz": [-2., 0., 0.]}]
    assert t.oracle_error(value, [[1., 1., 1.], [4., 0., 0.]]) == pytest.approx(0.)
    with pytest.raises(ValueError, match="invalid"):
        t.oracle_error(value, [[0., 0., 0.]])


def test_failure_at_90_and_excluded_and_empty_summary():
    periods = dict(zip(t.OBJECT_IDS, (None, 10., 10.)))
    preds = {oid: prediction(oid) for oid in t.OBJECT_IDS}
    preds[t.OBJECT_IDS[1]]["status"] = "published_failure"
    refs = dict.fromkeys(t.OBJECT_IDS, [[1., 0., 0.]])
    result = t.make_scores(periods, preds, refs, dict.fromkeys(t.OBJECT_IDS, 20.))
    assert result["failed"] == 2
    assert result["failures_at_90"]["n"] == 3
    assert result["failures_at_90"]["mean_deg"] == pytest.approx(60.)
    assert result["failures_excluded"]["n"] == 1
    assert result["failures_excluded"]["mean_deg"] == pytest.approx(0.)
    assert result["objects"][0]["oracle_at_3_error_deg"] is None
    assert result["objects"][0]["period_ratio_to_damit"] is None
    assert result["objects"][2]["period_ratio_to_damit"] == .5
    empty = t.make_scores(dict.fromkeys(t.OBJECT_IDS), preds, refs, dict.fromkeys(t.OBJECT_IDS, 20.))
    assert empty["failures_at_90"]["mean_deg"] == 90.
    assert empty["failures_excluded"]["mean_deg"] is None
    assert empty["failures_excluded"]["n"] == 0
    assert t.summary([20. + .5e-8, 30. + .5e-8])["within_30_percent"] == 100.
    assert t.summary([20. + .5e-8, 30. + .5e-8])["within_20_percent"] == 50.


def test_predict_check_then_own_then_score(inputs, fake_predictor):
    inputs.check_dir = checked_run(inputs)
    assert t.read(inputs.check_dir / "damit-check.json")["passed"] is True
    own_file(inputs)
    assert t.command_predict(inputs) == 0
    assert len(fake_predictor) == 5  # Three check cases, two with own periods.
    assert [c[0]["known_period_hours"] for c in fake_predictor] == [10., 11., 12., 8., 12.5]
    assert all(c[1] == inputs.models and c[2] == inputs.splits for c in fake_predictor)
    # Receipt is retained with own outputs, so the scratch check can be deleted.
    import shutil
    shutil.rmtree(inputs.check_dir)
    assert t.command_score(inputs) == 0
    scores = t.read(inputs.output_dir / "scores.json")
    assert scores["failed"] == 1 and scores["failures_excluded"]["n"] == 2
    assert "asteroid_279" in (inputs.output_dir / "RESULTS.md").read_text()
    before = {p: p.read_bytes() for p in inputs.output_dir.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        t.command_predict(inputs)
    with pytest.raises(FileExistsError):
        t.command_score(inputs)
    assert before == {p: p.read_bytes() for p in before}
    assert len(fake_predictor) == 5


def test_own_predict_never_opens_reference_axes_or_uses_published_period(inputs, fake_predictor, monkeypatch):
    inputs.check_dir = checked_run(inputs)
    own_file(inputs)
    original_read = t.read

    def read(path):
        assert path != inputs.references
        value = original_read(path)
        if path.parent == inputs.prepared_dir / "objects":
            value["known_period_hours"] = {"forbidden": True}
        return value

    monkeypatch.setattr(t, "read", read)
    assert t.command_predict(inputs) == 0


def test_predict_requires_check_before_creating_files(inputs, fake_predictor):
    own_file(inputs)
    with pytest.raises(ValueError, match="--check-dir"):
        t.command_predict(inputs)
    assert not (inputs.output_dir / "period-manifest.json").exists()
    assert not fake_predictor


def test_failed_damit_check_cannot_authorize_prediction(inputs, fake_predictor):
    path = inputs.published_scores_dir / f"{t.OBJECT_IDS[0]}.json"
    value = t.read(path)
    value["oracle_at_3_error_deg"] += 1.
    write(path, value)
    with pytest.raises(ValueError, match="reproduction failed"):
        checked_run(inputs)
    assert not (inputs.output_dir.parent / "check/damit-check.json").exists()


@pytest.mark.parametrize("change", ["cohort", "prepared_manifest_sha256", "search_source_sha256", "periods_hours"])
def test_own_manifest_rejects_wrong_inputs(inputs, change):
    path = own_file(inputs)
    value = t.read(path)
    value[change] = {} if change == "periods_hours" else "wrong"
    write(path, value)
    with pytest.raises(ValueError, match="search inputs"):
        t.load_own_periods(path, inputs.prepared_dir)


def test_score_rejects_changed_prediction(inputs, fake_predictor):
    inputs.check_dir = checked_run(inputs)
    own_file(inputs)
    t.command_predict(inputs)
    path = inputs.output_dir / "predictions" / f"{t.OBJECT_IDS[0]}.json"
    value = t.read(path)
    value["axes"][0]["axis_xyz"] = [1., 1., 1.]
    write(path, value)
    with pytest.raises(ValueError, match="prediction changed"):
        t.command_score(inputs)
    assert not (inputs.output_dir / "scores.json").exists()


def test_exclusive_create_preflights_siblings_and_rejects_dangling_symlink(tmp_path):
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    second.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        t.save_new({first: {}, second: {}})
    assert not first.exists() and second.is_symlink()
    first.write_text("original")
    with pytest.raises(FileExistsError):
        t.save_new({first: {}})
    assert first.read_text() == "original"


def test_predict_requires_real_inherited_gpu_lock(tmp_path):
    path = tmp_path / "gpu.lock"
    path.touch()
    env = {**os.environ, "LOCK": str(path), "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
    code = "from repro.own_period.temporal import require_gpu_lock; require_gpu_lock()"
    denied = subprocess.run([sys.executable, "-m", "repro.own_period.temporal", "predict",
                             "--output-dir", str(tmp_path / "unused")],
                            env=env, capture_output=True, text=True)
    assert denied.returncode != 0 and "requires runners/lock_run.sh" in denied.stderr
    assert not (tmp_path / "unused").exists()
    for shell, allowed in [('exec 9>"$LOCK"; exec "$@"', False),
                           ('exec 9>"$LOCK"; flock 9; exec "$@"', True)]:
        result = subprocess.run(["bash", "-c", shell, "lock-test", sys.executable, "-c", code],
                                env=env, capture_output=True, text=True)
        assert (result.returncode == 0) == allowed, result.stderr
