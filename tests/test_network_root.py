"""Network selection must preserve defaults and reject mixed rerun evidence."""

import json
from pathlib import Path

import numpy as np
import pytest

from lc_pipeline.k3.protocol import K3_PROTOCOL_SHA256
from repro import network_root as nr


@pytest.fixture(autouse=True)
def clean_switch(monkeypatch):
    for key in nr.VARIABLES:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def selected(tmp_path, monkeypatch):
    root = tmp_path / "retrain" / "artifact-root"
    (root / "models").mkdir(parents=True)
    records = []
    for fold in range(5):
        for seed in nr.SEEDS:
            name = f"real-fold-{fold}-seed-{seed}.pt"
            path = root / "models" / name
            path.write_bytes(f"{fold}/{seed}".encode())
            records.append({"filename": name, "fold": fold, "seed": seed,
                            "size_bytes": path.stat().st_size, "sha256": nr.sha256(path)})
    binding = root / "model-binding.json"
    binding.write_text(json.dumps({"schema": nr.LOCAL_SCHEMA,
        "protocol_sha256": K3_PROTOCOL_SHA256, "models": records}))
    monkeypatch.setenv("DELPHI_NETWORK_ROOT", str(root))
    return root, binding


def test_unset_preserves_all_historical_paths(temporary_delphi_data):
    followup = temporary_delphi_data.parent.parent
    root = followup / "inputs/frozen-artifacts/k3-definitive-7874092"
    assert Path(nr.NetworkPath("artifacts")) == root
    assert Path(nr.NetworkPath("models")) == root / "models"
    assert Path(nr.NetworkPath("evaluations", "real-oof-ensemble.npz")) == root / "evaluations/real-oof-ensemble.npz"
    assert Path(nr.NetworkPath("binding")) == followup / "data/external-models-final.json"
    assert Path(nr.NetworkPath("bundles")) == followup.parent / "DeLPHI-publication/DeLPHI-k3-release-assets-20260906"


def test_paths_follow_environment_after_import(selected, monkeypatch):
    root, _ = selected
    path = nr.NetworkPath("models") / "a.pt"
    assert Path(path) == root / "models/a.pt"
    other = root.parent / "other-artifacts"
    monkeypatch.setenv("DELPHI_NETWORK_ROOT", str(other))
    assert Path(path) == other / "models/a.pt"


def test_selected_binding_bundles_evaluations_and_results(selected):
    root, binding = selected
    assert Path(nr.NetworkPath("binding")) == binding
    assert Path(nr.NetworkPath("bundles")) == root / "bundles"
    assert Path(nr.NetworkPath("evaluations")) == root / "evaluations"
    assert Path(nr.NetworkPath("results", "ztf", "own-period-scores.json")) == root.parent / "results/ztf/own-period-scores.json"


def test_explicit_binding_and_bundle_overrides(selected, monkeypatch):
    root, _ = selected
    monkeypatch.setenv("DELPHI_MODEL_BINDING", str(root / "chosen.json"))
    monkeypatch.setenv("DELPHI_BUNDLE_ROOT", str(root / "safe"))
    assert Path(nr.NetworkPath("binding")) == root / "chosen.json"
    assert Path(nr.NetworkPath("bundles") / "k3-oof-fold-2") == root / "safe/k3-oof-fold-2"


@pytest.mark.parametrize("folder", [
    "data/own-period-20261021/new.npz", "inputs/frozen-artifacts/new.npz",
    "DeLPHI-k3-release-assets-20260906/new.json",
])
def test_selected_refuses_published_output(selected, folder):
    with pytest.raises(ValueError, match="published output"):
        nr.guard_output(selected[0].parent / folder)


def test_output_guard_resolves_symlinks(selected, tmp_path):
    old = tmp_path / "own-period-20261021"
    old.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(old, target_is_directory=True)
    with pytest.raises(ValueError, match="published output"):
        nr.guard_output(alias / "new.npz")


def test_unset_keeps_published_output_behavior(tmp_path):
    nr.guard_output(tmp_path / "own-period-20261021/file.npz")
    assert nr.network_metadata() == {}


def test_selected_accepts_bound_checkpoint_then_rejects_changed_bytes(selected):
    path = selected[0] / "models/real-fold-0-seed-17.pt"
    nr.check_checkpoint(path)
    path.write_bytes(b"different network")
    with pytest.raises(ValueError, match="hash mismatch"):
        nr.check_checkpoint(path)


def test_selected_rejects_wrong_binding_hash(selected):
    root, manifest = selected
    value = json.loads(manifest.read_text())
    value["models"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="hash mismatch"):
        nr.check_checkpoint(root / "models/real-fold-0-seed-17.pt")


def test_npz_requires_chosen_binding(selected, tmp_path):
    path = tmp_path / "prediction.npz"
    np.savez(path, **nr.network_metadata())
    nr.check_npz(path)
    np.savez(path, network_binding_sha256="0" * 64)
    with pytest.raises(ValueError, match="hash mismatch"):
        nr.check_npz(path)


def test_npz_rejects_wrong_checkpoint_even_with_binding_marker(selected, tmp_path):
    path = tmp_path / "prediction.npz"
    np.savez(path, checkpoint_names=["real-fold-0-seed-17.pt"],
             checkpoint_sha256=["0" * 64], **nr.network_metadata())
    with pytest.raises(ValueError, match="hash mismatch"):
        nr.check_npz(path)


def test_npz_refuses_unbound_external_evidence(selected, tmp_path):
    path = tmp_path / "old.npz"
    np.savez(path, errors_deg=[1.0])
    with pytest.raises(ValueError, match="unbound network artifact"):
        nr.check_npz(path)


def test_json_requires_chosen_binding_and_refuses_old_results(selected, tmp_path):
    path = tmp_path / "fresh.json"
    path.write_text(json.dumps({"network_binding_sha256": nr.sha256(selected[1])}))
    nr.check_json(path)
    path.write_text(json.dumps({"network_binding_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="hash mismatch"):
        nr.check_json(path)
    with pytest.raises(ValueError, match="published output"):
        nr.check_json(tmp_path / "own-period-20261021/own-period-scores.json")


def test_cross_family_paths_preserve_defaults_and_switch(selected, monkeypatch, tmp_path):
    old = tmp_path / "own-period-20261021/own-period-scores.json"
    path = nr.NetworkPath("results", "ztf", "own-period-scores.json", legacy_path=old)
    assert Path(path) == selected[0].parent / "results/ztf/own-period-scores.json"
    monkeypatch.delenv("DELPHI_NETWORK_ROOT")
    assert Path(path) == old


def test_full_prediction_checks_hash_before_unpickling(selected, monkeypatch):
    from repro.own_period import predict_oof_full

    path = selected[0] / "models/real-fold-0-seed-17.pt"
    path.write_bytes(b"wrong checkpoint")
    monkeypatch.setattr(predict_oof_full.torch, "load", lambda *a, **k: pytest.fail("unpickled unbound checkpoint"))
    with pytest.raises(ValueError, match="hash mismatch"):
        predict_oof_full.load_model(0, 17, "cpu")


@pytest.mark.parametrize("loader", ["ensemble", "single"])
def test_checkpoint_seed_bug_requires_opt_in(selected, monkeypatch, loader):
    import torch

    from lc_pipeline.k3 import model
    from lc_pipeline.k3.config import K3ScoreModelConfig
    from repro.own_period import predict_oof_full

    root, manifest = selected
    provenance = {"implementation_commit": "7874092b81d9456e5bf52fba9adb699b8218046f"}
    metadata = json.loads(manifest.read_text())
    for record in metadata["models"]:
        record["run_provenance"] = provenance
    manifest.write_text(json.dumps(metadata))

    class Model:
        def __init__(self, config):
            pass

        def load_state_dict(self, state):
            pass

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(torch, "load", lambda path, **kwargs: {
        "stage": "real-oof", "run_provenance": provenance,
        "config": {"seed": int(Path(path).stem.rsplit("-", 1)[1])},
        "model_config": K3ScoreModelConfig().as_mapping(), "model_state_dict": {},
    })
    monkeypatch.setattr(model, "CandidateConditionedScorer", Model)
    monkeypatch.setattr(predict_oof_full, "CandidateConditionedScorer", Model)

    def load(**kwargs):
        if loader == "ensemble":
            return nr.load_models(root / "models", manifest, (0,), "cpu", **kwargs)[0]
        return [predict_oof_full.load_model(0, 17, "cpu", **kwargs)[0]]

    with pytest.raises(ValueError, match=r"networks trained on simulations that reused random seeds \(v1\.0\.0\)"):
        load()
    assert len(load(allow_legacy=True)) == (5 if loader == "ensemble" else 1)


def test_full_scoring_can_resume_only_identical_bound_outputs(selected, tmp_path):
    from repro.own_period.predict_oof_full import write_npz_once

    path = tmp_path / "scored.npz"
    write_npz_once(path, object_ids=np.array(["asteroid_1"]), errors_deg=np.array([2.0]))
    write_npz_once(path, object_ids=np.array(["asteroid_1"]), errors_deg=np.array([2.0]))
    with pytest.raises(ValueError, match="existing output differs"):
        write_npz_once(path, object_ids=np.array(["asteroid_1"]), errors_deg=np.array([3.0]))


def test_local_adapter_rejects_training(tmp_path):
    from repro.prepare_k3_followup_inputs import local_rerun_step

    with pytest.raises(ValueError, match="only headline prerequisite evaluations"):
        local_rerun_step(["train-smoke", "--output", str(tmp_path / "never.pt")])
    assert not (tmp_path / "never.pt").exists()


def test_local_adapter_records_exact_evaluation_call_and_skipped_gate(tmp_path, monkeypatch):
    from lc_pipeline.k3 import publication_run
    from repro.prepare_k3_followup_inputs import local_rerun_step

    output = tmp_path / "real-fold-0-seed-17.npz"
    checkpoint = tmp_path / "real-fold-0-seed-17.pt"
    checkpoint.write_bytes(b"trusted-local-test-checkpoint")
    calls = []

    def evaluate(*args, **kwargs):
        calls.append((args, kwargs))
        np.savez(args[4], object_ids=["asteroid_1"])
        return {"checkpoint_sha256": nr.sha256(checkpoint)}

    monkeypatch.setattr(publication_run, "evaluate_real_oof_checkpoint", evaluate)
    receipt = local_rerun_step(["evaluate-publication-real-oof", "--checkpoint", str(checkpoint),
        "--catalog", "fixed-catalog", "--dump-root", "fixed-dump", "--splits", "fixed-splits",
        "--output", str(output), "--fold", "0", "--seed", "17", "--device", "cpu"])
    assert calls == [((checkpoint, Path("fixed-catalog"), Path("fixed-dump"), Path("fixed-splits"), output),
                      {"fold": 0, "seed": 17, "device": "cpu", "batch_size": 4})]
    assert receipt["publication_promotion_claimed"] is False
    assert receipt["skipped_publication_gates"] == ["clean committed export/evaluation checkout"]
    assert receipt["artifact_sha256"] == nr.sha256(output)
    assert Path(str(output) + ".receipt.json").is_file()


@pytest.fixture
def local_graph(tmp_path):
    import torch

    root = tmp_path / "artifact-root"
    (root / "models").mkdir(parents=True)
    (root / "evaluations").mkdir()
    provenance = {"protocol_sha256": K3_PROTOCOL_SHA256, "implementation_commit": "a" * 40}

    def save(name, stage, seed, extra=None):
        path = root / "models" / name
        torch.save({"schema": "delphi.k3-checkpoint.v2", "stage": stage,
                    "config": {"seed": seed}, "model_config": {"fixture": True},
                    "run_provenance": {**provenance, **(extra or {})}}, path)
        return nr.sha256(path)

    names, hashes = [], []
    for seed in nr.SEEDS:
        synthetic = save(f"synthetic-seed-{seed}.pt", "synthetic", seed)
        synthetic_evaluation = root / "evaluations" / f"synthetic-test-seed-{seed}.npz"
        np.savez(synthetic_evaluation, oracle_errors_deg=[1.0])
        Path(str(synthetic_evaluation) + ".receipt.json").write_text(json.dumps({
            "artifact_sha256": nr.sha256(synthetic_evaluation), "checkpoint_sha256": synthetic}))
        for fold in range(5):
            name = f"real-fold-{fold}-seed-{seed}.pt"
            digest = save(name, "real-oof", seed, {"pretrained_checkpoint_sha256": synthetic})
            names.append(name)
            hashes.append(digest)
            evaluation = root / "evaluations" / name.replace(".pt", ".npz")
            np.savez(evaluation, object_ids=[f"asteroid_{fold}"], score_grids=np.zeros((1, 1)))
            Path(str(evaluation) + ".receipt.json").write_text(json.dumps({
                "checkpoint_sha256": digest, "artifact_sha256": nr.sha256(evaluation)}))
        aggregate = root / "evaluations" / f"real-oof-seed-{seed}.npz"
        np.savez(aggregate, input_artifact_sha256=[nr.sha256(
            root / "evaluations" / f"real-fold-{fold}-seed-{seed}.npz") for fold in range(5)])
        Path(str(aggregate) + ".receipt.json").write_text(json.dumps({"artifact_sha256": nr.sha256(aggregate)}))
    mapping = root / "models/synthetic-label-shuffle-seed-17.labels.json"
    mapping.write_text("{}")
    save("synthetic-label-shuffle-seed-17.pt", "synthetic-label-shuffle", 17,
         {"label_mapping_sha256": nr.sha256(mapping)})
    ensemble = root / "evaluations/real-oof-ensemble.npz"
    np.savez(ensemble, checkpoint_names=names, checkpoint_sha256=hashes,
             input_aggregate_sha256=[nr.sha256(root / "evaluations" / f"real-oof-seed-{seed}.npz") for seed in nr.SEEDS])
    Path(str(ensemble) + ".receipt.json").write_text(json.dumps({"artifact_sha256": nr.sha256(ensemble)}))
    return root


def test_local_binding_audits_graph_without_inventing_publication_evidence(local_graph, tmp_path):
    from repro.prepare_k3_followup_inputs import bind_local_rerun

    value = bind_local_rerun(local_graph, tmp_path / "metadata/model-binding.json")
    assert value["schema"] == nr.LOCAL_SCHEMA
    assert len(value["models"]) == 25
    assert value["publication_promotion_claimed"] is False
    assert "artifact_archive_sha256" not in value
    assert "archive_index_sha256" not in value
    assert len(value["artifact_sha256"]) == 36
    assert value["checkpoint_audit_sha256"] == nr.sha256(tmp_path / "metadata/checkpoint-audit.json")


def test_local_binding_refuses_evaluation_of_another_network(local_graph, tmp_path):
    from repro.prepare_k3_followup_inputs import bind_local_rerun

    receipt = local_graph / "evaluations/real-fold-0-seed-17.npz.receipt.json"
    value = json.loads(receipt.read_text())
    value["checkpoint_sha256"] = "0" * 64
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="evaluation/checkpoint receipt mismatch"):
        bind_local_rerun(local_graph, tmp_path / "metadata/model-binding.json")
    assert not (tmp_path / "metadata/model-binding.json").exists()


def test_local_binding_accepts_native_training_cli_receipts(local_graph, tmp_path):
    from repro.prepare_k3_followup_inputs import bind_local_rerun

    log = local_graph.parent / "train-queue.log"
    with log.open("w") as stream:
        for receipt in sorted((local_graph / "evaluations").glob("*.receipt.json")):
            stream.write(json.dumps({"summary": json.loads(receipt.read_text())}) + "\n")
            receipt.unlink()
    value = bind_local_rerun(local_graph, tmp_path / "metadata/model-binding.json")
    assert len(value["artifact_sha256"]) == 36
    assert all(receipt["receipt_source"] == str(log) for receipt in value["evaluation_receipts"].values())
