"""Publication runner safeguards that prevent OOF fold/seed leakage."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lc_pipeline.k3.evaluation import ensemble_score_grids
from lc_pipeline.k3.publication_run import (
    K3PublicationRunError,
    ablate_k3_inputs,
    aggregate_real_oof_artifacts,
    aggregate_real_oof_diagnostic_artifacts,
    aggregate_real_oof_input_swap_artifacts,
    audit_publication_checkpoint_set,
    build_v1_comparator_artifact,
    calibrate_real_oof_artifacts,
    calibrate_real_oof_ensemble_artifacts,
    evaluate_real_calibration_checkpoint,
    evaluate_real_calibration_ensemble,
    evaluate_real_oof_checkpoint,
    evaluate_real_oof_diagnostic_checkpoint,
    evaluate_real_oof_ensemble,
    evaluate_real_oof_input_swap_checkpoint,
    evaluate_synthetic_input_swap_checkpoint,
    evaluate_synthetic_label_shuffle_checkpoint,
    run_synthetic_label_shuffle_control,
)
from lc_pipeline.k3.tokenizer import K3_EPOCH_FEATURE_NAMES, K3_PHASE_FEATURE_NAMES


def _diagnostic_inputs() -> dict[str, torch.Tensor]:
    return {
        "phase_features": torch.ones((2, 3, 64, len(K3_PHASE_FEATURE_NAMES))),
        "phase_mask": torch.tensor(
            [[[True] * 64, [False] * 64, [True] * 64]] * 2,
            dtype=torch.bool,
        ),
        "geometry_features": torch.full((2, 3, 64, 8), 2.0),
        "epoch_features": torch.full(
            (2, 3, len(K3_EPOCH_FEATURE_NAMES)), 3.0
        ),
        "epoch_mask": torch.tensor(
            [[True, True, False], [True, False, False]], dtype=torch.bool
        ),
    }


@pytest.mark.parametrize(
    ("control", "zero_phase", "zero_epoch", "zero_geometry"),
    (
        ("brightness", {0, 1}, set(range(17)), False),
        ("geometry", set(), set(), True),
        ("period-scalar", set(), {20}, False),
        ("sampling-summary", {2, 3}, {17, 18, 19}, False),
    ),
)
def test_k3_diagnostic_ablation_is_exact_and_mask_preserving(
    control, zero_phase, zero_epoch, zero_geometry
) -> None:
    values = _diagnostic_inputs()
    result = ablate_k3_inputs(values, control)
    assert torch.equal(result["phase_mask"], values["phase_mask"])
    assert torch.equal(result["epoch_mask"], values["epoch_mask"])
    assert result["phase_mask"].data_ptr() != values["phase_mask"].data_ptr()
    assert result["epoch_mask"].data_ptr() != values["epoch_mask"].data_ptr()
    for index in range(len(K3_PHASE_FEATURE_NAMES)):
        expected = 0.0 if index in zero_phase else 1.0
        assert torch.all(result["phase_features"][..., index] == expected)
    for index in range(len(K3_EPOCH_FEATURE_NAMES)):
        expected = 0.0 if index in zero_epoch else 3.0
        assert torch.all(result["epoch_features"][..., index] == expected)
    assert torch.all(
        result["geometry_features"] == (0.0 if zero_geometry else 2.0)
    )
    assert torch.all(values["phase_features"] == 1.0)
    assert torch.all(values["geometry_features"] == 2.0)
    assert torch.all(values["epoch_features"] == 3.0)


def test_k3_zero_numeric_control_retains_only_masks() -> None:
    values = _diagnostic_inputs()
    result = ablate_k3_inputs(values, "zero-numeric")
    for key in ("phase_features", "geometry_features", "epoch_features"):
        assert torch.count_nonzero(result[key]) == 0
    assert torch.equal(result["phase_mask"], values["phase_mask"])
    assert torch.equal(result["epoch_mask"], values["epoch_mask"])
    with pytest.raises(K3PublicationRunError, match="unsupported"):
        ablate_k3_inputs(values, "time")


def test_real_oof_evaluation_is_bound_to_fold_seed_and_test_ids(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "real-fold-2-seed-17.pt"
    torch.save({"stage": "real-oof", "config": {"seed": 17}}, checkpoint)
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps({"folds": [{"fold": 2, "test_ids": ["a", "b"]}]}),
        encoding="utf-8",
    )
    observed = {}

    def fake_load(catalog, dump_root, object_ids):
        observed["ids"] = tuple(object_ids)
        return ("example-a", "example-b")

    def fake_evaluate(source, examples, output, *, device, batch_size):
        observed.update(source=source, examples=examples, output=output, device=device, batch_size=batch_size)
        return {"mean_error_deg": 12.0}

    monkeypatch.setattr("lc_pipeline.k3.publication_run.load_real_examples", fake_load)
    monkeypatch.setattr("lc_pipeline.k3.publication_run._evaluate_examples", fake_evaluate)
    result = evaluate_real_oof_checkpoint(
        checkpoint, "catalog", "dump", splits, tmp_path / "result.npz",
        fold=2, seed=17, device="cpu", batch_size=3,
    )
    assert observed["ids"] == ("a", "b")
    assert observed["examples"] == ("example-a", "example-b")
    assert result == {"mean_error_deg": 12.0, "fold": 2.0, "seed": 17.0}


def test_label_shuffle_training_retains_exact_derangement_manifest(tmp_path, monkeypatch) -> None:
    training = tuple(SimpleNamespace(object_id=f"train-{index}") for index in range(5))
    validation = (SimpleNamespace(object_id="validation"),)
    calls = {"loads": 0}

    def fake_load(directory):
        calls["loads"] += 1
        return training if calls["loads"] == 1 else validation

    monkeypatch.setattr("lc_pipeline.k3.publication_run.load_synthetic_examples", fake_load)
    monkeypatch.setattr(
        "lc_pipeline.k3.publication_run.shuffle_training_labels",
        lambda values, seed: tuple((value.object_id, seed) for value in values),
    )

    class FakeModel:
        def __init__(self, config):
            self.config = config

    monkeypatch.setattr("lc_pipeline.k3.publication_run.CandidateConditionedScorer", FakeModel)

    def fake_fit(model, train, valid, **kwargs):
        calls.update(train=train, valid=valid, kwargs=kwargs)
        return "fit-result"

    monkeypatch.setattr("lc_pipeline.k3.publication_run.fit_k3", fake_fit)
    result = run_synthetic_label_shuffle_control(
        "train", "validation", tmp_path, seed=17, device="cpu"
    )
    assert result == "fit-result"
    assert calls["kwargs"]["stage"] == "synthetic-label-shuffle"
    mapping = json.loads(
        (tmp_path / "synthetic-label-shuffle-seed-17.labels.json").read_text(
            encoding="utf-8"
        )
    )
    assert mapping["training_object_count"] == 5
    assert mapping["control_seed"] == 20260901
    assignments = mapping["assignments"]
    assert {row["label_donor_object_id"] for row in assignments} == {
        value.object_id for value in training
    }
    assert all(
        row["input_object_id"] != row["label_donor_object_id"] for row in assignments
    )


def test_label_shuffle_evaluation_rejects_wrong_stage_and_uses_untouched_test(tmp_path, monkeypatch) -> None:
    source = tmp_path / "synthetic-label-shuffle-seed-17.pt"
    output = tmp_path / "synthetic-label-shuffle-test-seed-17.npz"
    torch.save({"stage": "synthetic-label-shuffle", "config": {"seed": 17}}, source)
    examples = ("test-a", "test-b")
    monkeypatch.setattr(
        "lc_pipeline.k3.publication_run.load_synthetic_examples", lambda directory: examples
    )
    monkeypatch.setattr(
        "lc_pipeline.k3.publication_run._evaluate_examples",
        lambda checkpoint, values, destination, **kwargs: {
            "mean_error_deg": 50.0,
            "values": values,
        },
    )
    result = evaluate_synthetic_label_shuffle_checkpoint(
        source, "test", output, seed=17, device="cpu"
    )
    assert result["values"] == examples
    assert result["control"] == "label-shuffle-training"

    torch.save({"stage": "synthetic", "config": {"seed": 17}}, source)
    with pytest.raises(K3PublicationRunError, match="stage/training seed"):
        evaluate_synthetic_label_shuffle_checkpoint(
            source, "test", output, seed=17, device="cpu"
        )


def test_real_oof_evaluation_rejects_checkpoint_fold_or_seed_mixup(tmp_path) -> None:
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"folds": [{"fold": 1, "test_ids": ["a"]}]}), encoding="utf-8")
    wrong_name = tmp_path / "real-fold-0-seed-17.pt"
    torch.save({"stage": "real-oof", "config": {"seed": 17}}, wrong_name)
    with pytest.raises(K3PublicationRunError, match="checkpoint name"):
        evaluate_real_oof_checkpoint(
            wrong_name, "catalog", "dump", splits, tmp_path / "result.npz",
            fold=1, seed=17, device="cpu",
        )

    wrong_seed = tmp_path / "real-fold-1-seed-17.pt"
    torch.save({"stage": "real-oof", "config": {"seed": 42}}, wrong_seed)
    with pytest.raises(K3PublicationRunError, match="training seed"):
        evaluate_real_oof_checkpoint(
            wrong_seed, "catalog", "dump", splits, tmp_path / "result.npz",
            fold=1, seed=17, device="cpu",
        )


def test_real_calibration_evaluation_uses_only_frozen_calibration_role(
    tmp_path, monkeypatch
) -> None:
    checkpoint = tmp_path / "real-fold-2-seed-17.pt"
    torch.save({"stage": "real-oof", "config": {"seed": 17}}, checkpoint)
    calibration_ids = [f"calibration-{index}" for index in range(14)]
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps(
            {
                "folds": [
                    {
                        "fold": 2,
                        "calibration_ids": calibration_ids,
                        "test_ids": ["forbidden-test"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    observed = {}

    def fake_load(catalog, dump, ids):
        observed["ids"] = tuple(ids)
        return tuple(ids)

    monkeypatch.setattr(
        "lc_pipeline.k3.publication_run.load_real_examples", fake_load
    )
    monkeypatch.setattr(
        "lc_pipeline.k3.publication_run._evaluate_examples",
        lambda source, examples, output, **kwargs: {"mean_error_deg": 8.0},
    )
    result = evaluate_real_calibration_checkpoint(
        checkpoint,
        "catalog",
        "dump",
        splits,
        tmp_path / "real-fold-2-seed-17-calibration.npz",
        fold=2,
        seed=17,
        device="cpu",
    )
    assert observed["ids"] == tuple(calibration_ids)
    assert "forbidden-test" not in observed["ids"]
    assert result["role"] == "calibration"


def test_real_oof_input_swap_retains_auditable_fold_local_donors(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "real-fold-3-seed-17.pt"
    torch.save({"stage": "real-oof", "config": {"seed": 17}}, checkpoint)
    test_ids = ["a", "b", "c", "d"]
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps({"folds": [{"fold": 3, "test_ids": test_ids}]}), encoding="utf-8"
    )
    examples = tuple(SimpleNamespace(object_id=value, tokenized=value, target_axes=np.eye(3)[:1]) for value in test_ids)
    observed = {}
    monkeypatch.setattr(
        "lc_pipeline.k3.publication_run.load_real_examples",
        lambda catalog, dump, ids: examples,
    )

    def fake_evaluate(source, swapped, output, *, device, batch_size, artifact_arrays):
        observed.update(swapped=swapped, arrays=artifact_arrays)
        return {"mean_error_deg": 40.0}

    monkeypatch.setattr("lc_pipeline.k3.publication_run._evaluate_examples", fake_evaluate)
    result = evaluate_real_oof_input_swap_checkpoint(
        checkpoint,
        "catalog",
        "dump",
        splits,
        tmp_path / "real-fold-3-seed-17-input-swap.npz",
        fold=3,
        seed=17,
        device="cpu",
    )
    donors = observed["arrays"]["input_donor_ids"].tolist()
    assert set(donors) == set(test_ids)
    assert all(target != donor for target, donor in zip(test_ids, donors, strict=True))
    assert [value.tokenized for value in observed["swapped"]] == donors
    assert result["control"] == "input-swap"
    assert result["derangement_seed"] == 20260904.0


def test_synthetic_input_swap_is_aligned_and_reports_paired_gap(
    tmp_path, monkeypatch
) -> None:
    checkpoint = tmp_path / "synthetic-seed-17.pt"
    torch.save({"stage": "synthetic", "config": {"seed": 17}}, checkpoint)
    examples = tuple(
        SimpleNamespace(object_id=value, tokenized=value, target_axes=np.eye(3)[:1])
        for value in ("a", "b", "c", "d")
    )
    raw = tmp_path / "synthetic-test-seed-17.npz"
    np.savez_compressed(
        raw,
        object_ids=np.asarray([value.object_id for value in examples]),
        oracle_errors_deg=np.asarray([10.0, 20.0, 30.0, 40.0]),
    )
    monkeypatch.setattr(
        "lc_pipeline.k3.publication_run.load_synthetic_examples",
        lambda directory: examples,
    )
    observed = {}

    def fake_evaluate(source, swapped, output, **kwargs):
        observed.update(swapped=swapped, arrays=kwargs["artifact_arrays"])
        return {"mean_error_deg": 39.0}

    monkeypatch.setattr(
        "lc_pipeline.k3.publication_run._evaluate_examples", fake_evaluate
    )
    result = evaluate_synthetic_input_swap_checkpoint(
        checkpoint,
        "test",
        raw,
        tmp_path / "synthetic-test-seed-17-input-swap.npz",
        seed=17,
        device="cpu",
    )
    donor_ids = observed["arrays"]["input_donor_ids"].tolist()
    assert set(donor_ids) == {"a", "b", "c", "d"}
    assert all(
        target.object_id != donor
        for target, donor in zip(examples, donor_ids, strict=True)
    )
    assert [value.tokenized for value in observed["swapped"]] == donor_ids
    assert result["input_swap_gap_deg"] == 14.0
    assert result["control"] == "input-swap"


def test_real_oof_diagnostic_applies_named_transform_and_records_control(
    tmp_path, monkeypatch
) -> None:
    checkpoint = tmp_path / "real-fold-1-seed-17.pt"
    torch.save({"stage": "real-oof", "config": {"seed": 17}}, checkpoint)
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps({"folds": [{"fold": 1, "test_ids": ["a", "b"]}]}),
        encoding="utf-8",
    )
    examples = (SimpleNamespace(object_id="a"), SimpleNamespace(object_id="b"))
    monkeypatch.setattr(
        "lc_pipeline.k3.publication_run.load_real_examples",
        lambda catalog, dump, ids: examples,
    )
    observed = {}

    def fake_evaluate(source, values, output, **kwargs):
        observed.update(kwargs)
        transformed = kwargs["input_transform"](_diagnostic_inputs())
        assert torch.count_nonzero(transformed["geometry_features"]) == 0
        return {"mean_error_deg": 31.0}

    monkeypatch.setattr(
        "lc_pipeline.k3.publication_run._evaluate_examples", fake_evaluate
    )
    output = tmp_path / "real-fold-1-seed-17-geometry.npz"
    result = evaluate_real_oof_diagnostic_checkpoint(
        checkpoint,
        "catalog",
        "dump",
        splits,
        output,
        fold=1,
        seed=17,
        control="geometry",
        device="cpu",
    )
    assert observed["artifact_arrays"]["diagnostic_control"].tolist() == [
        "geometry",
        "geometry",
    ]
    assert result["control"] == "geometry"
    with pytest.raises(K3PublicationRunError, match="paths must use"):
        evaluate_real_oof_diagnostic_checkpoint(
            checkpoint,
            "catalog",
            "dump",
            splits,
            tmp_path / "ambiguous-name.npz",
            fold=1,
            seed=17,
            control="geometry",
            device="cpu",
        )

def test_real_oof_aggregation_requires_exact_disjoint_fold_membership(tmp_path) -> None:
    folds = []
    for fold in range(5):
        identifiers = [f"object-{fold}-a", f"object-{fold}-b"]
        folds.append({"fold": fold, "test_ids": identifiers})
        npz = tmp_path / f"real-fold-{fold}-seed-17.npz"
        np.savez_compressed(
            npz,
            object_ids=np.asarray(identifiers),
            oracle_errors_deg=np.asarray([fold + 1.0, fold + 2.0]),
            score_grids=np.zeros((2, 6144), dtype="float32"),
        )
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"folds": folds}), encoding="utf-8")
    output = tmp_path / "aggregate.npz"
    summary = aggregate_real_oof_artifacts(tmp_path, splits, output, seed=17)
    assert summary["n_objects"] == 10.0
    assert summary["mean_error_deg"] == 3.5
    with np.load(output) as artifact:
        assert len(set(artifact["object_ids"].tolist())) == 10
        assert artifact["score_grids"].shape == (10, 6144)

    folds[2]["test_ids"].reverse()
    splits.write_text(json.dumps({"folds": folds}), encoding="utf-8")
    with pytest.raises(K3PublicationRunError, match="IDs/order"):
        aggregate_real_oof_artifacts(tmp_path, splits, tmp_path / "bad.npz", seed=17)


def test_v1_comparator_builder_binds_all_seeds_folds_and_conditions(tmp_path) -> None:
    seeds = (17, 42, 137, 777, 2027)
    folds = [
        {"fold": fold, "test_ids": [f"object-{fold}-{index}" for index in range(34)]}
        for fold in range(5)
    ]
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"folds": folds}), encoding="utf-8")
    oof_root = tmp_path / "v1"
    conditions = {
        "raw": 20.0,
        "atlas": 30.0,
        "exhaustive_deranged": 40.0,
        "zero": 50.0,
    }
    for seed_index, seed in enumerate(seeds):
        for row in folds:
            fold = row["fold"]
            run_id = f"publication-oof-v1-faithful-f{fold}-s{seed}"
            run_root = oof_root / "runs" / run_id
            run_root.mkdir(parents=True)
            rows = []
            for condition, base_error in conditions.items():
                rows.extend(
                    {
                        "object_id": object_id,
                        "condition": condition,
                        "model_kind": "v1_faithful",
                        "seed": seed,
                        "has_valid_axis": True,
                        "axes": np.eye(3).tolist(),
                        "axis_oracle_at3_error_deg": base_error + seed_index,
                    }
                    for object_id in row["test_ids"]
                )
            rows_path = run_root / "evaluation-rows.jsonl"
            rows_path.write_text(
                "".join(json.dumps(value) + "\n" for value in rows),
                encoding="utf-8",
            )
            summary = {
                "schema": "delphi.axial-experiment-summary.v1",
                "run_id": run_id,
                "model_kind": "v1_faithful",
                "fold": fold,
                "seed": seed,
                "git_commit": "a" * 40,
                "artifacts": {
                    "evaluation-rows.jsonl": hashlib.sha256(
                        rows_path.read_bytes()
                    ).hexdigest()
                },
            }
            (run_root / "axial-experiment-summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
    output = tmp_path / "v1-comparators.npz"
    summary = build_v1_comparator_artifact(oof_root, splits, output)
    assert summary["n_objects"] == 170.0
    assert summary["seed_count"] == 5.0
    assert summary["v1_mean_error_deg"] == 22.0
    assert summary["atlas_mean_error_deg"] == 32.0
    assert summary["deranged_mean_error_deg"] == 42.0
    with np.load(output, allow_pickle=False) as artifact:
        assert artifact["schema"].item() == "delphi.k3-v1-comparators.v1"
        assert artifact["v1_seed_errors_deg"].shape == (5, 170)
        assert np.all(artifact["v1_errors_deg"] == 22.0)
        assert artifact["source_evaluation_sha256"].shape == (25,)


def test_checkpoint_audit_requires_complete_single_commit_parent_graph(tmp_path) -> None:
    from lc_pipeline.k3.protocol import K3_PROTOCOL_SHA256

    seeds = (17, 42, 137, 777, 2027)

    def build(root, *, corrupt_parent=False):
        root.mkdir()
        provenance = {
            "protocol_sha256": K3_PROTOCOL_SHA256,
            "implementation_commit": "a" * 40,
        }
        synthetic_hashes = {}
        for seed in seeds:
            path = root / f"synthetic-seed-{seed}.pt"
            torch.save(
                {
                    "schema": "delphi.k3-checkpoint.v2",
                    "stage": "synthetic",
                    "config": {"seed": seed},
                    "model_config": {"hidden_dim": 128},
                    "run_provenance": provenance,
                },
                path,
            )
            synthetic_hashes[seed] = hashlib.sha256(path.read_bytes()).hexdigest()
        for seed in seeds:
            for fold in range(5):
                parent = synthetic_hashes[seed]
                if corrupt_parent and seed == 777 and fold == 3:
                    parent = "b" * 64
                torch.save(
                    {
                        "schema": "delphi.k3-checkpoint.v2",
                        "stage": "real-oof",
                        "config": {"seed": seed},
                        "model_config": {"hidden_dim": 128},
                        "run_provenance": {
                            **provenance,
                            "pretrained_checkpoint_sha256": parent,
                        },
                    },
                    root / f"real-fold-{fold}-seed-{seed}.pt",
                )
        mapping = root / "synthetic-label-shuffle-seed-17.labels.json"
        mapping.write_text('{"mapping":"fixed"}\n', encoding="utf-8")
        mapping_hash = hashlib.sha256(mapping.read_bytes()).hexdigest()
        torch.save(
            {
                "schema": "delphi.k3-checkpoint.v2",
                "stage": "synthetic-label-shuffle",
                "config": {"seed": 17},
                "model_config": {"hidden_dim": 128},
                "run_provenance": {
                    **provenance,
                    "label_mapping_sha256": mapping_hash,
                },
            },
            root / "synthetic-label-shuffle-seed-17.pt",
        )

    valid = tmp_path / "valid"
    build(valid)
    result = audit_publication_checkpoint_set(
        valid, tmp_path / "checkpoint-audit.json"
    )
    assert result["passed"] is True
    assert result["synthetic_checkpoint_count"] == 5
    assert result["real_oof_checkpoint_count"] == 25
    assert len(result["checkpoint_sha256"]) == 31

    invalid = tmp_path / "invalid"
    build(invalid, corrupt_parent=True)
    with pytest.raises(K3PublicationRunError, match="not bound to synthetic"):
        audit_publication_checkpoint_set(
            invalid, tmp_path / "bad" / "checkpoint-audit.json"
        )


def test_cross_fitted_calibration_uses_fold_roles_and_bounded_95_radius(
    tmp_path,
) -> None:
    folds = []
    all_test_ids = []
    raw_errors = []
    for fold in range(5):
        calibration_ids = [f"cal-{fold}-{index}" for index in range(14)]
        test_ids = [f"test-{fold}-{index}" for index in range(34)]
        folds.append(
            {
                "fold": fold,
                "calibration_ids": calibration_ids,
                "test_ids": test_ids,
            }
        )
        all_test_ids.extend(test_ids)
        raw_errors.extend([10.0] * 17 + [20.0] * 17)
        np.savez_compressed(
            tmp_path / f"real-fold-{fold}-seed-17-calibration.npz",
            object_ids=np.asarray(calibration_ids),
            oracle_errors_deg=np.arange(1.0, 15.0),
        )
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"folds": folds}), encoding="utf-8")
    raw = tmp_path / "real-oof-seed-17.npz"
    np.savez_compressed(
        raw,
        object_ids=np.asarray(all_test_ids),
        folds=np.repeat(np.arange(5, dtype=np.int8), 34),
        oracle_errors_deg=np.asarray(raw_errors),
    )
    output = tmp_path / "real-oof-seed-17-calibrated.npz"
    summary = calibrate_real_oof_artifacts(
        tmp_path, splits, raw, output, seed=17
    )
    assert summary["cone90_min_deg"] == 14.0
    assert summary["cone90_max_deg"] == 14.0
    assert summary["cone95_min_deg"] == 90.0
    assert summary["cone95_max_deg"] == 90.0
    assert summary["empirical_coverage90"] == 0.5
    assert summary["empirical_coverage95"] == 1.0
    with np.load(output, allow_pickle=False) as artifact:
        assert artifact["schema"].item() == "delphi.k3-real-oof-calibration.v1"
        assert artifact["calibration_object_ids"].shape == (5, 14)
        assert artifact["object_cone90_deg"].shape == (170,)
        assert np.all(artifact["fold_cone95_deg"] == 90.0)
        assert np.all(artifact["covered95"])


def test_input_swap_aggregation_computes_exact_paired_gap(tmp_path) -> None:
    from lc_pipeline.k3.publication_run import K3_INPUT_SWAP_SEED
    from lc_pipeline.k3.training import deterministic_derangement_indices

    folds = []
    all_ids = []
    raw_errors = []
    for fold in range(5):
        identifiers = [f"object-{fold}-a", f"object-{fold}-b"]
        folds.append({"fold": fold, "test_ids": identifiers})
        all_ids.extend(identifiers)
        raw_errors.extend((10.0, 20.0))
        donors = deterministic_derangement_indices(2, seed=K3_INPUT_SWAP_SEED + fold)
        np.savez_compressed(
            tmp_path / f"real-fold-{fold}-seed-17-input-swap.npz",
            object_ids=np.asarray(identifiers),
            input_donor_ids=np.asarray([identifiers[int(index)] for index in donors]),
            derangement_seeds=np.full(2, K3_INPUT_SWAP_SEED + fold),
            oracle_errors_deg=np.asarray((22.0, 28.0)),
            score_grids=np.zeros((2, 6144), dtype=np.float32),
        )
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"folds": folds}), encoding="utf-8")
    raw = tmp_path / "real-oof-seed-17.npz"
    np.savez_compressed(
        raw,
        object_ids=np.asarray(all_ids),
        folds=np.repeat(np.arange(5, dtype=np.int8), 2),
        oracle_errors_deg=np.asarray(raw_errors),
    )
    output = tmp_path / "real-oof-seed-17-input-swap.npz"
    summary = aggregate_real_oof_input_swap_artifacts(
        tmp_path, splits, raw, output, seed=17
    )
    assert summary["raw_mean_error_deg"] == 15.0
    assert summary["mean_error_deg"] == 25.0
    assert summary["input_swap_gap_deg"] == 10.0
    with np.load(output, allow_pickle=False) as artifact:
        np.testing.assert_allclose(artifact["degradation_deg"], np.tile((12.0, 8.0), 5))
        assert len(set(artifact["input_donor_ids"].tolist())) == 10


def test_diagnostic_aggregation_is_exactly_raw_paired(tmp_path) -> None:
    folds = []
    all_ids = []
    raw_errors = []
    for fold in range(5):
        identifiers = [f"object-{fold}-a", f"object-{fold}-b"]
        folds.append({"fold": fold, "test_ids": identifiers})
        all_ids.extend(identifiers)
        raw_errors.extend((10.0, 20.0))
        np.savez_compressed(
            tmp_path / f"real-fold-{fold}-seed-17-zero-numeric.npz",
            object_ids=np.asarray(identifiers),
            diagnostic_control=np.asarray(["zero-numeric", "zero-numeric"]),
            oracle_errors_deg=np.asarray((30.0, 40.0)),
            score_grids=np.zeros((2, 6144), dtype=np.float32),
        )
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"folds": folds}), encoding="utf-8")
    raw = tmp_path / "real-oof-seed-17.npz"
    np.savez_compressed(
        raw,
        object_ids=np.asarray(all_ids),
        folds=np.repeat(np.arange(5, dtype=np.int8), 2),
        oracle_errors_deg=np.asarray(raw_errors),
    )
    output = tmp_path / "real-oof-seed-17-zero-numeric.npz"
    summary = aggregate_real_oof_diagnostic_artifacts(
        tmp_path,
        splits,
        raw,
        output,
        seed=17,
        control="zero-numeric",
    )
    assert summary["mean_error_deg"] == 35.0
    assert summary["raw_mean_error_deg"] == 15.0
    assert summary["diagnostic_gap_deg"] == 20.0
    with np.load(output, allow_pickle=False) as artifact:
        assert artifact["schema"].item() == "delphi.k3-real-oof-diagnostic.v1"
        assert artifact["control"].item() == "zero-numeric"
        np.testing.assert_allclose(artifact["degradation_deg"], 20.0)


@pytest.mark.parametrize(
    ("deployment_deformation", "expect_rejection"),
    ((5e-4, False), (0.1, True)),
)
def test_real_oof_ensemble_averages_maps_before_one_shared_refinement(
    tmp_path, monkeypatch, deployment_deformation, expect_rejection
) -> None:
    seeds = (17, 42)
    folds = [
        {"fold": fold, "test_ids": [f"object-{fold}"]}
        for fold in range(5)
    ]
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"folds": folds}), encoding="utf-8")
    object_ids = np.asarray([row["test_ids"][0] for row in folds])
    fold_ids = np.arange(5, dtype=np.int8)
    base_grid = np.linspace(-1.0, 1.0, 6144, dtype=np.float32)
    for seed, value in zip(seeds, (2.0, 6.0), strict=True):
        np.savez_compressed(
            tmp_path / f"real-oof-seed-{seed}.npz",
            object_ids=object_ids,
            folds=fold_ids,
            score_grids=np.tile(base_grid + value, (5, 1)),
        )
        for fold in range(5):
            torch.save(
                {
                    "stage": "real-oof",
                    "config": {"seed": seed},
                    "model_config": {},
                    "model_state_dict": {},
                },
                tmp_path / f"real-fold-{fold}-seed-{seed}.pt",
            )

    class FakeModel:
        def __init__(self, config):
            self.config = config

        def load_state_dict(self, state):
            assert state == {}

        def to(self, device):
            assert device == "cpu"
            return self

        def eval(self):
            return self

    starts = np.eye(3, dtype=np.float64)
    observed_maps = []
    monkeypatch.setattr("lc_pipeline.k3.publication_run.CandidateConditionedScorer", FakeModel)
    monkeypatch.setattr(
        "lc_pipeline.k3.publication_run.load_real_examples",
        lambda catalog, dump, ids: tuple(
            SimpleNamespace(object_id=value, target_axes=starts[:1]) for value in ids
        ),
    )
    monkeypatch.setattr(
        "lc_pipeline.k3.publication_run.collate_examples",
        lambda examples: {"phase_features": torch.zeros((1, 1))},
    )
    model_grid_values = iter([2.0, 6.0] * 5)
    deformation = np.linspace(
        -deployment_deformation,
        deployment_deformation,
        6144,
        dtype=np.float64,
    )
    monkeypatch.setattr(
        "lc_pipeline.k3.publication_run.score_axial_grid",
        lambda model, inputs, chunk_size: (
            base_grid.astype(np.float64) + next(model_grid_values) + deformation
        ),
    )

    def fake_modes(score_grid):
        observed_maps.append(np.array(score_grid, copy=True))
        return tuple(
            SimpleNamespace(axis_xyz=axis, grid_index=index)
            for index, axis in enumerate(starts)
        )

    def fake_refine(models, inputs, initial):
        assert len(models) == 2
        assert tuple(inputs) == ("phase_features",)
        return np.array(initial, copy=True), np.asarray([3.0, 2.0, 1.0])

    monkeypatch.setattr("lc_pipeline.k3.publication_run.modes_from_score_grid", fake_modes)
    monkeypatch.setattr("lc_pipeline.k3.publication_run.refine_ensemble_axes", fake_refine)
    output = tmp_path / "ensemble.npz"
    if expect_rejection:
        with pytest.raises(K3PublicationRunError, match="landscape differs materially"):
            evaluate_real_oof_ensemble(
                tmp_path, tmp_path, "catalog", "dump", splits, output,
                seeds=seeds, device="cpu",
            )
        assert not output.exists()
        return
    summary = evaluate_real_oof_ensemble(
        tmp_path, tmp_path, "catalog", "dump", splits, output,
        seeds=seeds, device="cpu",
    )

    assert summary["mean_error_deg"] == 0.0
    assert summary["seed_count"] == 2.0
    assert len(observed_maps) == 10
    expected_cached = ensemble_score_grids(
        np.stack((base_grid + 2.0, base_grid + 6.0))[:, None, :]
    )[0]
    expected_deployed = ensemble_score_grids(
        np.stack(
            (
                base_grid.astype(np.float64) + 2.0 + deformation,
                base_grid.astype(np.float64) + 6.0 + deformation,
            )
        )[:, None, :]
    )[0]
    assert all(
        np.array_equal(values, expected)
        for values, expected in zip(
            observed_maps,
            (expected_cached, expected_deployed) * 5,
            strict=True,
        )
    )
    expected_difference = expected_deployed - expected_cached
    expected_mean_abs = float(np.mean(np.abs(expected_difference)))
    centered_difference = expected_difference - float(np.mean(expected_difference))
    expected_centered_rms = float(
        np.sqrt(np.mean(np.square(centered_difference)))
    )
    expected_max_abs = float(np.max(np.abs(expected_difference)))
    assert summary["cached_grid_max_abs_difference"] == pytest.approx(
        expected_max_abs, abs=1e-12
    )
    assert summary["cached_grid_mean_abs_difference"] == pytest.approx(
        expected_mean_abs, abs=1e-7
    )
    assert summary["cached_grid_max_centered_rms_difference"] == pytest.approx(
        expected_centered_rms, abs=1e-7
    )
    assert summary["cached_grid_max_normalized_rms_difference"] < 1e-3
    assert summary["cached_mode_agreement_fraction"] == 1.0
    with np.load(output, allow_pickle=False) as artifact:
        assert artifact["schema"].item() == "delphi.k3-real-oof-ensemble.v1"
        assert np.array_equal(artifact["seeds"], seeds)
        assert np.array_equal(
            artifact["score_grids"],
            np.tile(
                expected_deployed.astype(np.float32),
                (5, 1),
            ),
        )
        assert artifact["checkpoint_sha256"].shape == (10,)
        assert artifact["inference_wall_seconds"].shape == (5,)
        assert np.all(artifact["inference_wall_seconds"] > 0)
        np.testing.assert_allclose(
            artifact["cached_grid_max_abs_differences"],
            expected_max_abs,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            artifact["cached_grid_mean_abs_differences"],
            expected_mean_abs,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            artifact["cached_grid_centered_rms_differences"],
            expected_centered_rms,
            atol=1e-7,
        )
        assert np.all(artifact["cached_grid_normalized_rms_differences"] < 1e-3)
        assert np.array_equal(
            artifact["cached_mode_indices"], artifact["deployed_mode_indices"]
        )


def test_real_oof_ensemble_rejects_seed_alignment_mismatch(tmp_path) -> None:
    folds = [{"fold": fold, "test_ids": [f"object-{fold}"]} for fold in range(5)]
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"folds": folds}), encoding="utf-8")
    expected = np.asarray([row["test_ids"][0] for row in folds])
    for seed in (17, 42):
        identifiers = expected.copy()
        if seed == 42:
            identifiers[[0, 1]] = identifiers[[1, 0]]
        np.savez_compressed(
            tmp_path / f"real-oof-seed-{seed}.npz",
            object_ids=identifiers,
            folds=np.arange(5, dtype=np.int8),
            score_grids=np.zeros((5, 6144), dtype=np.float32),
        )
    with pytest.raises(K3PublicationRunError, match="IDs/folds"):
        evaluate_real_oof_ensemble(
            tmp_path, tmp_path, "catalog", "dump", splits, tmp_path / "ensemble.npz",
            seeds=(17, 42), device="cpu",
        )


def test_calibration_ensemble_averages_maps_before_shared_refinement(
    tmp_path, monkeypatch
) -> None:
    seeds = (17, 42)
    folds = [
        {
            "fold": fold,
            "calibration_ids": [f"cal-{fold}-{index}" for index in range(14)],
            "test_ids": [f"test-{fold}-{index}" for index in range(34)],
        }
        for fold in range(5)
    ]
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"folds": folds}), encoding="utf-8")
    for row in folds:
        fold = row["fold"]
        for seed, value in zip(seeds, (2.0, 6.0), strict=True):
            np.savez_compressed(
                tmp_path / f"real-fold-{fold}-seed-{seed}-calibration.npz",
                object_ids=np.asarray(row["calibration_ids"]),
                score_grids=np.full((14, 6144), value, dtype=np.float32),
            )
            torch.save(
                {
                    "stage": "real-oof",
                    "config": {"seed": seed},
                    "model_config": {},
                    "model_state_dict": {},
                },
                tmp_path / f"real-fold-{fold}-seed-{seed}.pt",
            )

    class FakeModel:
        def __init__(self, config):
            self.config = config

        def load_state_dict(self, state):
            assert state == {}

        def to(self, device):
            assert device == "cpu"
            return self

        def eval(self):
            return self

    starts = np.eye(3, dtype=np.float64)
    observed_maps = []
    monkeypatch.setattr(
        "lc_pipeline.k3.publication_run.CandidateConditionedScorer", FakeModel
    )
    monkeypatch.setattr(
        "lc_pipeline.k3.publication_run.load_real_examples",
        lambda catalog, dump, ids: tuple(
            SimpleNamespace(object_id=value, target_axes=starts[:1]) for value in ids
        ),
    )
    monkeypatch.setattr(
        "lc_pipeline.k3.publication_run.collate_examples",
        lambda examples: {"phase_features": torch.zeros((1, 1))},
    )

    def fake_modes(score_grid):
        observed_maps.append(np.array(score_grid, copy=True))
        return tuple(SimpleNamespace(axis_xyz=axis) for axis in starts)

    monkeypatch.setattr(
        "lc_pipeline.k3.publication_run.modes_from_score_grid", fake_modes
    )
    monkeypatch.setattr(
        "lc_pipeline.k3.publication_run.refine_ensemble_axes",
        lambda models, inputs, initial: (
            np.array(initial, copy=True),
            np.asarray([3.0, 2.0, 1.0]),
        ),
    )
    output = tmp_path / "real-calibration-ensemble.npz"
    summary = evaluate_real_calibration_ensemble(
        tmp_path,
        tmp_path,
        "catalog",
        "dump",
        splits,
        output,
        seeds=seeds,
        device="cpu",
    )
    assert summary["mean_error_deg"] == 0.0
    assert len(observed_maps) == 70
    assert all(np.array_equal(values, np.full(6144, 4.0)) for values in observed_maps)
    with np.load(output, allow_pickle=False) as artifact:
        assert artifact["schema"].item() == "delphi.k3-real-calibration-ensemble.v1"
        assert artifact["object_ids"].shape == (70,)
        assert artifact["checkpoint_sha256"].shape == (10,)


def test_ensemble_calibration_requires_exact_seed_and_fold_alignment(tmp_path) -> None:
    seeds = np.asarray((17, 42), dtype=np.int64)
    folds = [
        {
            "fold": fold,
            "calibration_ids": [f"cal-{fold}-{index}" for index in range(14)],
            "test_ids": [f"test-{fold}-{index}" for index in range(34)],
        }
        for fold in range(5)
    ]
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"folds": folds}), encoding="utf-8")
    split_hash = hashlib.sha256(splits.read_bytes()).hexdigest()
    calibration = tmp_path / "real-calibration-ensemble.npz"
    np.savez_compressed(
        calibration,
        schema=np.asarray("delphi.k3-real-calibration-ensemble.v1"),
        seeds=seeds,
        object_ids=np.asarray(
            [value for row in folds for value in row["calibration_ids"]]
        ),
        folds=np.repeat(np.arange(5, dtype=np.int8), 14),
        oracle_errors_deg=np.tile(np.arange(1.0, 15.0), 5),
        splits_sha256=np.asarray(split_hash),
    )
    oof = tmp_path / "real-oof-ensemble.npz"
    np.savez_compressed(
        oof,
        schema=np.asarray("delphi.k3-real-oof-ensemble.v1"),
        seeds=seeds,
        object_ids=np.asarray([value for row in folds for value in row["test_ids"]]),
        folds=np.repeat(np.arange(5, dtype=np.int8), 34),
        oracle_errors_deg=np.tile([10.0] * 17 + [20.0] * 17, 5),
    )
    output = tmp_path / "real-oof-ensemble-calibrated.npz"
    summary = calibrate_real_oof_ensemble_artifacts(
        calibration, splits, oof, output
    )
    assert summary["cone90_min_deg"] == 14.0
    assert summary["cone95_max_deg"] == 90.0
    assert summary["empirical_coverage90"] == 0.5
    assert summary["empirical_coverage95"] == 1.0

    broken = tmp_path / "broken-oof.npz"
    np.savez_compressed(
        broken,
        schema=np.asarray("delphi.k3-real-oof-ensemble.v1"),
        seeds=np.asarray((17, 137)),
        object_ids=np.asarray([value for row in folds for value in row["test_ids"]]),
        folds=np.repeat(np.arange(5, dtype=np.int8), 34),
        oracle_errors_deg=np.zeros(170),
    )
    with pytest.raises(K3PublicationRunError, match="seed/fold aligned"):
        calibrate_real_oof_ensemble_artifacts(
            calibration,
            splits,
            broken,
            tmp_path / "bad" / "real-oof-ensemble-calibrated.npz",
        )
