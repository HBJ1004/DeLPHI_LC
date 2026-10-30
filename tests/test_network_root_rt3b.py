"""CPU checks of family policies, mixed-network rejection and fresh scores."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from repro import network_root as network
from repro.own_period import census7, lowq_predict, predict_external, score
from repro.own_period.spacecraft import run as spacecraft


@pytest.fixture
def selected(tmp_path, monkeypatch):
    for key in network.VARIABLES:
        monkeypatch.delenv(key, raising=False)
    root = tmp_path / "artifact-root"
    root.mkdir()
    from lc_pipeline.k3.protocol import K3_PROTOCOL_SHA256
    dump(root / "model-binding.json", {
        "schema": network.LOCAL_SCHEMA, "protocol_sha256": K3_PROTOCOL_SHA256,
        "models": [{"filename": f"real-fold-{f}-seed-{s}.pt", "sha256": "a" * 64,
                    "size_bytes": 1} for f in range(5) for s in network.SEEDS],
    })
    monkeypatch.setenv("DELPHI_NETWORK_ROOT", str(root))
    monkeypatch.setenv("DELPHI_DATA", str(tmp_path / "fixed"))
    return root


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_results_legacy_join_remains_lazy(monkeypatch, tmp_path):
    from repro.own_period import DataPath
    for key in network.VARIABLES:
        monkeypatch.delenv(key, raising=False)
    path = network.NetworkPath("results", "spacecraft", legacy_path=DataPath("spacecraft"))
    monkeypatch.setenv("DELPHI_DATA", str(tmp_path))
    assert Path(path / "eros-loo/results.json") == tmp_path / "spacecraft/eros-loo/results.json"


@pytest.mark.parametrize("module", [predict_external, lowq_predict, census7, spacecraft])
def test_family_models_resolve_after_import(selected, module):
    assert Path(module.MODELS) == selected / "models"
    assert Path(module.MODEL_MANIFEST) == selected / "model-binding.json"


@pytest.mark.parametrize("bad", ["unbound", "wrong", "duplicate"])
def test_jsonl_rejects_mixed_or_duplicate_rows(selected, bad):
    path = selected.parent / "predictions.jsonl"
    row = {"object_id": "one", "status": "no_period", **network.json_metadata()}
    if bad == "unbound":
        row.pop("network_binding_sha256")
    elif bad == "wrong":
        row["network_binding_sha256"] = "wrong"
    jsonl(path, [row, row] if bad == "duplicate" else [row])
    with pytest.raises(ValueError, match="binding mismatch or duplicate"):
        network.check_jsonl(path)


@pytest.mark.parametrize(("cohort", "folds"), [("ztf", (3,)), ("alcdef", tuple(range(5)))])
def test_external_preserves_fold_policy(selected, monkeypatch, cohort, folds):
    prepared = selected.parent / "prepared.json"
    dump(prepared, {"input": "sealed"})
    monkeypatch.setattr(predict_external, "ready_objects", lambda cohort: [("one", prepared, network.sha256(prepared))])
    monkeypatch.setattr(predict_external, "_fold_policy", lambda *args: (3,))
    monkeypatch.setattr(predict_external, "_load_prepared", lambda path: ("one", SimpleNamespace(hours=10, provenance="fixed"), []))
    calls = []
    def loader(models, binding, actual, device):
        calls.append(actual)
        return [object()], [], []
    monkeypatch.setattr(predict_external, "_load_models", loader)
    monkeypatch.setattr(predict_external, "predict", lambda *args: (np.eye(3), np.ones(3)))
    output = selected.parent / "result.jsonl"
    monkeypatch.setattr("sys.argv", ["predict", "--cohort", cohort, "--periods", "published", "--output", str(output), "--device", "cpu"])
    assert predict_external.main() == 0
    assert calls == [folds]
    network.check_jsonl(output)
    assert json.loads(output.read_text())["folds"] == list(folds)


def test_external_scoring_uses_fresh_reference_predictions(selected, monkeypatch):
    from repro.own_period import search_external
    out = selected.parent / "results/ztf"
    prepared = selected.parent / "prepared.json"
    dump(prepared, {"known_period_hours": 10})
    own = out / "own-period-ztf.jsonl"
    jsonl(own, [{"object_id": "one", "status": "ok", "axes": [[0, 1, 0]], **network.json_metadata()}])
    jsonl(out / "validation-ztf-published.jsonl", [{"object_id": "one", "status": "ok", "axes": [[1, 0, 0]], **network.json_metadata()}])
    sensitivity = selected.parent / "sensitivity.json"
    dump(sensitivity, {"ztf": {"objects": [{"object_id": "one", "original_error_deg": 51, "train_only_atlas_error_deg": 20}]}})
    periods = selected.parent / "periods.json"
    dump(periods, {"periods_hours": {"one": 10}})
    monkeypatch.setattr(score, "SENSITIVITY", sensitivity)
    monkeypatch.setattr(score, "RESAMPLES", 10)
    monkeypatch.setattr(search_external, "ready_objects", lambda cohort: [("one", prepared, "fixed")])
    for name in ("damit_table_poles", "reference_poles", "gaia_reference_poles"):
        monkeypatch.setattr(score.crb, name, lambda: {})
    monkeypatch.setattr(score.crb, "cohort_reference", lambda *args: np.array([[1, 0, 0]]))
    monkeypatch.setattr(score.crb, "random_reference_errors", lambda *args: (np.array([30.]), None))
    result = score.external("ztf", own, periods)
    assert result["objects"][0]["published_error_deg"] == 0
    assert result["objects"][0]["own_error_deg"] == 90


@pytest.mark.parametrize(("policy", "expected"), [("fold0", (0,)), ("all25", tuple(range(5)))])
def test_census_policy_on_cpu(selected, monkeypatch, policy, expected):
    calls = []
    def loader(models, binding, folds, device):
        calls.append((folds, device))
        return [None] * (5 if folds == (0,) else 25), [], []
    monkeypatch.setattr(network, "load_models", loader)
    census7.load_models(policy, "cpu")
    assert calls == [(expected, "cpu")]


def test_spacecraft_always_all25(selected, monkeypatch):
    calls = []
    def loader(models, binding, folds, device):
        calls.append(folds)
        return [None] * 25, [], []
    monkeypatch.setattr(network, "load_models", loader)
    assert len(spacecraft.ensemble("cpu", 4)[0]) == 25
    assert calls == [tuple(range(5))]


def test_lowq_run_fold0_and_resume_rejects_old_rows(selected, monkeypatch):
    row = {"object_id": "one", "quality_flag": 1, "known_period_hours": 10, "period_provenance": "fixed"}
    monkeypatch.setattr(lowq_predict, "selected_rows", lambda limit: [row])
    calls = []
    def loader(models, binding, folds, device):
        calls.append(folds)
        return [None] * 5, [], []
    monkeypatch.setattr(lowq_predict, "_load_models", loader)
    monkeypatch.setattr(lowq_predict, "predict_one", lambda *args: {"object_id": "one", "status": "ok"})
    args = SimpleNamespace(limit=None, periods="published", device="cpu", output=selected.parent / "lowq.jsonl", max_seconds=float("inf"))
    assert lowq_predict.run(args) == 0
    assert calls == [(0,)]
    network.check_jsonl(args.output)
    jsonl(args.output, [{"object_id": "one", "status": "ok"}])
    with pytest.raises(ValueError, match="binding mismatch"):
        lowq_predict.run(args)


def test_lowq_finalize_checks_completeness_and_chosen_binding(selected, monkeypatch):
    monkeypatch.setattr(lowq_predict, "selected_rows", lambda limit: [{"object_id": "one"}])
    path = selected.parent / "lowq.jsonl"
    jsonl(path, [{"object_id": "one", "status": "ok", **network.json_metadata()}])
    args = SimpleNamespace(output=path, finalize=path.with_suffix(".manifest.json"), limit=None, periods="published", device="cpu")
    lowq_predict.finalize(args)
    network.check_json(args.finalize)
    assert json.loads(args.finalize.read_text())["model_manifest_sha256"] == network.sha256(selected / "model-binding.json")


@pytest.mark.parametrize("module", [predict_external, lowq_predict])
def test_prediction_refuses_published_output_before_inference(selected, monkeypatch, module):
    path = selected.parent / "own-period-20261021/result.jsonl"
    if module is lowq_predict:
        with pytest.raises(ValueError, match="published output"):
            module.run(SimpleNamespace(output=path))
    else:
        monkeypatch.setattr("sys.argv", ["predict", "--cohort", "ztf", "--periods", "published", "--output", str(path)])
        with pytest.raises(ValueError, match="published output"):
            module.main()
    assert not path.exists()


def test_spacecraft_oof_refuses_stale_chosen_input(selected):
    path = selected.parent / "results/headline/own-period-oof.npz"
    path.parent.mkdir(parents=True)
    np.savez(path, object_ids=["asteroid_1714"], network_binding_sha256="old")
    with pytest.raises(ValueError, match="binding hash mismatch"):
        spacecraft.oof_path("own")


def test_census_reader_rejects_unbound_prediction_manifest(selected):
    out = selected.parent / "results/census7"
    dump(out / "predictions.manifest.json", {"periods": "own"})
    with pytest.raises(ValueError, match="binding hash mismatch"):
        census7.load_predictions(out, "own", "fold0")


def test_external_cli_scores_one_family_at_canonical_path(selected, monkeypatch):
    path = selected.parent / "results/ztf/own-period-scores.json"
    calls = []
    def external(cohort, predictions, periods):
        calls.append((cohort, Path(predictions), Path(periods)))
        return {"objects": []}
    monkeypatch.setattr(score, "external", external)
    monkeypatch.setattr(score, "main_sample", lambda path: pytest.fail("headline scoring is outside this step"))
    # Restore the caller-specific legacy helper paths after this test.
    for attr in ("ROOT", "FOLLOWUP", "CATALOG"):
        monkeypatch.setattr(score.crb, attr, getattr(score.crb, attr))
    monkeypatch.setattr("sys.argv", ["score", "--cohort", "ztf", "--output", str(path)])
    assert score.main() == 0
    network.check_json(path)
    assert calls == [("ztf", selected.parent / "results/ztf/own-period-ztf.jsonl",
                      selected.parent / "fixed/own-periods-ztf.json")]
    assert set(json.loads(path.read_text())) == {"schema", "protocol", "seed", "resamples", "network_binding_sha256", "ztf"}


def test_eros_baseline_comes_from_chosen_network_results(selected):
    from repro.own_period.spacecraft import eros_loo
    path = selected.parent / "results/headline/own-period-oof.npz"
    path.parent.mkdir(parents=True)
    np.savez(path, object_ids=[eros_loo.OBJECT_ID], refined_axes=[np.eye(3)], errors_deg=[12.5],
             folds=[1], **network.network_metadata())
    axes, error = eros_loo.expected_baseline()
    assert np.array_equal(axes, np.eye(3))
    assert error == 12.5


@pytest.mark.parametrize("posthoc", [False, True])
def test_spacecraft_scoring_reads_fixed_inputs_and_bound_predictions(selected, monkeypatch, posthoc):
    from repro.own_period.spacecraft import posthoc as posthoc_module
    directory = selected.parent / "results/spacecraft"
    if posthoc:
        directory /= "posthoc"
    fixed = spacecraft.input_root(directory)
    for relative in ("lock/cohort-lock.json", "prepared/manifest.json", "validation/result.json"):
        dump(fixed / relative, {"sealed": relative})
    source = fixed / "object.json"
    dump(source, {"known_period_hours": 10})
    dump(fixed / "search/periods.json", {
        "periods_hours": {"itokawa-combined": 10},
        "details": {"itokawa-combined": {"input_sha256": spacecraft.sha(source)}},
        "cohort_lock_sha256": spacecraft.sha(fixed / "lock/cohort-lock.json"),
        "prepared_manifest_sha256": spacecraft.sha(fixed / "prepared/manifest.json"),
    })
    reference = fixed / "reference-poles.json"
    dump(reference, {"schema": "delphi.spacecraft-reference-poles.v1", "objects": [
        {"object": "Itokawa", "unit_vector": [1, 0, 0], "damit_id": None}]})
    row = {"run_id": "itokawa-combined", "object_id": "one", "name": "itokawa",
           "source_name": "combined", "primary": True, "kind": "external"}
    oof = selected.parent / "empty-oof.npz"
    np.savez(oof, object_ids=np.array([], dtype=str))
    files = {"reference_poles": spacecraft.binding(reference),
             **{f"oof_{mode}": spacecraft.binding(oof) for mode in spacecraft.OOF_FILES}}
    monkeypatch.setattr(spacecraft, "locked", lambda out: ({"files": files}, [dict(row)]))
    monkeypatch.setattr(spacecraft, "prepared", lambda out: {row["run_id"]: source})
    monkeypatch.setattr(spacecraft, "validation", lambda out: {})
    monkeypatch.setattr(spacecraft, "model_rows", lambda lock: {})
    monkeypatch.setattr(spacecraft, "RESCORE", {})
    outputs = {}
    for mode in ("own", "catalogue"):
        path = directory / f"predictions/{mode}.jsonl"
        jsonl(path, [{**row, "period_hours": 10, "input_sha256": spacecraft.sha(source),
                      "status": "ok", "axes": np.eye(3).tolist(), "scores": [1, 0, 0], **network.json_metadata()}])
        outputs[mode] = spacecraft.relative_binding(path, path.parent)
    manifest = {key: spacecraft.sha(fixed / relative) for key, relative in (
        ("cohort_lock_sha256", "lock/cohort-lock.json"), ("search_sha256", "search/periods.json"),
        ("prepared_manifest_sha256", "prepared/manifest.json"), ("validation_sha256", "validation/result.json"))}
    dump(directory / "predictions/manifest.json", {**manifest, "outputs": outputs, **network.json_metadata()})
    module = posthoc_module if posthoc else spacecraft
    module.cmd_score(SimpleNamespace(output_dir=directory))
    network.check_json(directory / "scores.json")
    result = json.loads((directory / "scores.json").read_text())
    assert result["objects"][0]["own"]["oracle_deg"] == 0
    assert (directory / "RESULTS.md").is_file()


def test_lowq_validation_compares_fresh_chosen_reference(selected, monkeypatch):
    from repro.own_period import lowq_validate
    path = Path(lowq_validate.PUBLISHED)
    row = {"object_id": "one", "status": "ok", "axes": [
        {"axis_xyz": [1, 0, 0], "grid_index": 7}], **network.json_metadata()}
    jsonl(path, [row])
    rerun = selected.parent / "subset.jsonl"
    jsonl(rerun, [row])
    output = selected.parent / "validation.json"
    monkeypatch.setattr("sys.argv", ["validate", "--rerun", str(rerun), "--output", str(output)])
    assert lowq_validate.main() == 0
    result = json.loads(output.read_text())
    assert result["passed"] and result["bit_identical_axis_sets"] == 1
    assert result["comparison_role"] == "fresh chosen-network reference-period inference"
