"""Contracts for label-blind, provenance-bound convergence timing."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import lc_pipeline.k3.convergence_timing as timing
from lc_pipeline.k3.convergence_study import (
    TIMING_AXIS_COMPONENT_DIFFERENCE_MAXIMUM,
    TIMING_COLD_DEFINITION,
    TIMING_GRID_NORMALIZED_RMS_MAXIMUM,
    TIMING_PROVENANCE_SCHEMA,
    TIMING_REFINED_SCORE_DIFFERENCE_MAXIMUM,
    TIMING_SCHEMA,
    TIMING_WARM_DEFINITION,
    _load_timing,
)
from lc_pipeline.k3.downstream import DownstreamBenchmarkError
from lc_pipeline.v2.data import canonical_json, sha256_file


SPEC_HASH = "1" * 64
SPLIT_HASH = "2" * 64
BLIND_HASH = "3" * 64
ENSEMBLE_HASH = "4" * 64


def _object_layout() -> tuple[tuple[str, ...], tuple[int, ...], list[dict[str, object]]]:
    split_rows: list[dict[str, object]] = []
    object_ids: list[str] = []
    folds: list[int] = []
    for fold in range(5):
        fold_ids = [f"synthetic-{fold}-{index:02d}" for index in range(34)]
        split_rows.append({"fold": fold, "test_ids": fold_ids})
        object_ids.extend(fold_ids)
        folds.extend([fold] * len(fold_ids))
    return tuple(object_ids), tuple(folds), split_rows


def _bundle_bindings() -> tuple[dict[str, object], ...]:
    bindings: list[dict[str, object]] = []
    for fold in range(5):
        members = [
            {
                "seed": seed,
                "file": f"model-{seed}.safetensors",
                "safetensors_sha256": f"{fold + 5:x}" * 64,
                "source_checkpoint_name": f"real-fold-{fold}-seed-{seed}.pt",
                "source_checkpoint_sha256": f"{fold + 10:x}" * 64,
            }
            for seed in (17, 42, 137, 777, 2027)
        ]
        bindings.append(
            {
                "fold": fold,
                "bundle_id": f"synthetic-fold-{fold}",
                "manifest_sha256": f"{fold + 1:x}" * 64,
                "model_config_sha256": f"{fold + 6:x}" * 64,
                "members": members,
            }
        )
    return tuple(bindings)


def _parity(*, passed: bool = True) -> dict[str, object]:
    return {
        "grid_centered_normalized_rms": 0.0,
        "mode_indices_match": True,
        "max_axis_component_difference": 0.0,
        "max_refined_score_absolute_difference": 0.0,
        "passed": passed,
    }


def _device_provenance() -> dict[str, object]:
    return {
        "requested": "cpu",
        "resolved": "cpu",
        "type": "cpu",
        "index": None,
        "name": "synthetic-cpu",
        "torch_version": "test",
        "torch_cuda_version": None,
        "cudnn_version": None,
        "numpy_version": np.__version__,
        "python_version": "test",
        "platform": "synthetic",
        "cpu_threads": 1,
    }


def _measurement_contract() -> dict[str, object]:
    return {
        "clock": "time.perf_counter",
        "warm_definition": TIMING_WARM_DEFINITION,
        "cold_definition": TIMING_COLD_DEFINITION,
        "warmup_runs_per_object": 1,
        "timed_runs_per_object": 1,
        "score_grid_centered_normalized_rms_maximum": (
            TIMING_GRID_NORMALIZED_RMS_MAXIMUM
        ),
        "axis_component_difference_maximum": (
            TIMING_AXIS_COMPONENT_DIFFERENCE_MAXIMUM
        ),
        "refined_score_absolute_difference_maximum": (
            TIMING_REFINED_SCORE_DIFFERENCE_MAXIMUM
        ),
    }


def _valid_timing_payload() -> dict[str, object]:
    object_ids, folds, _ = _object_layout()
    bundles = list(_bundle_bindings())
    return {
        "schema": TIMING_SCHEMA,
        "source_ensemble_sha256": ENSEMBLE_HASH,
        "object_ids": list(object_ids),
        "warm_wall_seconds": [0.5] * 170,
        "cold_wall_seconds": [2.0] * 170,
        "provenance": {
            "schema": TIMING_PROVENANCE_SCHEMA,
            "source_spec_sha256": SPEC_HASH,
            "source_split_sha256": SPLIT_HASH,
            "source_blind_inputs_sha256": BLIND_HASH,
            "source_ensemble_sha256": ENSEMBLE_HASH,
            "object_count": 170,
            "seeds": [17, 42, 137, 777, 2027],
            "measurement": _measurement_contract(),
            "device": _device_provenance(),
            "bundle_set_sha256": hashlib.sha256(
                canonical_json(bundles).encode("utf-8")
            ).hexdigest(),
            "bundles": bundles,
            "parity_rows": [
                {
                    "object_id": object_id,
                    "fold": folds[index],
                    "cold": _parity(),
                    "warm": _parity(),
                }
                for index, object_id in enumerate(object_ids)
            ],
        },
    }


def test_capture_api_has_no_catalog_or_reference_parameter():
    signature = inspect.signature(timing.capture_neural_timing)

    assert tuple(signature.parameters) == (
        "spec_path",
        "spec_checksum_path",
        "split_path",
        "blind_inputs_path",
        "ensemble_path",
        "bundle_root",
        "dump_root",
        "output_path",
        "device",
    )
    assert signature.parameters["device"].kind is inspect.Parameter.KEYWORD_ONLY
    assert all(
        "catalog" not in name.lower() and "reference" not in name.lower()
        for name in signature.parameters
    )


class _PairClock:
    """Return 2.0 s for each cold pair and 0.5 s for each warm pair."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> float:
        pair_index, endpoint = divmod(self.calls, 2)
        within_fold = pair_index % 68
        duration = 2.0 if within_fold < 34 else 0.5
        value = pair_index * 4.0 + (duration if endpoint else 0.0)
        self.calls += 1
        return value


def _install_capture_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    fail_first_parity: bool = False,
) -> tuple[dict[str, Path], dict[str, int], _PairClock]:
    object_ids, folds, split_rows = _object_layout()
    axes = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], 170, axis=0)
    blind_lookup = {
        object_id: {
            "object_id": object_id,
            "fold": folds[index],
            "period_hours": 8.0,
            "guided_axes": axes[index].tolist(),
        }
        for index, object_id in enumerate(object_ids)
    }
    paths = {
        name: tmp_path / name
        for name in (
            "spec",
            "checksum",
            "split",
            "blind",
            "ensemble",
            "bundles",
            "dump",
            "timing.json",
        )
    }
    hashes = {
        paths["split"]: SPLIT_HASH,
        paths["blind"]: BLIND_HASH,
        paths["ensemble"]: ENSEMBLE_HASH,
    }
    frozen = timing._FrozenOutputs(
        object_ids=object_ids,
        folds=np.asarray(folds),
        score_grids=np.empty((170, 0)),
        mode_indices=np.empty((170, 0), dtype=np.int64),
        refined_axes=axes,
        refined_scores=np.empty((170, 0)),
        checkpoint_sha256={},
    )
    study = {
        "frozen_inputs": {
            "publication_split_sha256": SPLIT_HASH,
            "oof_ensemble_sha256": ENSEMBLE_HASH,
            "blind_execution_inputs_sha256": BLIND_HASH,
            "reference_catalog_sha256": "5" * 64,
            "neural_timing_schema": TIMING_SCHEMA,
        }
    }
    counters = {"predictor": 0, "predict": 0, "parity": 0}

    class FakePredictor:
        def __init__(self, _directory: Path, *, device: str) -> None:
            assert device == "cpu"
            counters["predictor"] += 1

    def fake_predict(*_args: object, **_kwargs: object) -> object:
        counters["predict"] += 1
        return object()

    def fake_parity(*_args: object, **_kwargs: object) -> dict[str, object]:
        counters["parity"] += 1
        return _parity(
            passed=not (fail_first_parity and counters["parity"] == 1)
        )

    clock = _PairClock()
    monkeypatch.setattr(timing, "_read_spec", lambda *_args: ({}, SPEC_HASH))
    monkeypatch.setattr(timing, "_study_section", lambda _spec: study)
    monkeypatch.setattr(timing, "sha256_file", lambda path: hashes[Path(path)])
    monkeypatch.setattr(timing, "_split_rows", lambda _path: ({}, split_rows))
    monkeypatch.setattr(
        timing, "_load_blind_inputs", lambda *_args, **_kwargs: ({}, blind_lookup)
    )
    monkeypatch.setattr(
        timing, "_load_frozen_outputs", lambda *_args, **_kwargs: frozen
    )
    monkeypatch.setattr(
        timing, "_bind_bundles", lambda *_args, **_kwargs: _bundle_bindings()
    )
    monkeypatch.setattr(timing, "_resolve_device", lambda _value: timing.torch.device("cpu"))
    monkeypatch.setattr(timing, "_device_provenance", lambda *_args: _device_provenance())
    monkeypatch.setattr(timing, "_load_label_blind_epochs", lambda *_args: ())
    monkeypatch.setattr(timing, "K3EnsemblePredictor", FakePredictor)
    monkeypatch.setattr(timing, "_predict_full", fake_predict)
    monkeypatch.setattr(timing, "_parity_result", fake_parity)
    monkeypatch.setattr(timing, "_synchronize", lambda _device: None)
    monkeypatch.setattr(timing.gc, "collect", lambda: 0)
    monkeypatch.setattr(timing.time, "perf_counter", clock)
    return paths, counters, clock


def test_capture_writes_exact_170_distinct_cold_and_warm_measurements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths, counters, clock = _install_capture_fakes(monkeypatch, tmp_path)

    result = timing.capture_neural_timing(
        paths["spec"],
        paths["checksum"],
        paths["split"],
        paths["blind"],
        paths["ensemble"],
        paths["bundles"],
        paths["dump"],
        paths["timing.json"],
        device="cpu",
    )

    artifact = json.loads(paths["timing.json"].read_text(encoding="utf-8"))
    object_ids, folds, _ = _object_layout()
    assert result["object_count"] == 170
    assert artifact["object_ids"] == list(object_ids)
    assert len(artifact["cold_wall_seconds"]) == 170
    assert len(artifact["warm_wall_seconds"]) == 170
    assert artifact["cold_wall_seconds"] == [2.0] * 170
    assert artifact["warm_wall_seconds"] == [0.5] * 170
    assert all(
        cold > warm
        for cold, warm in zip(
            artifact["cold_wall_seconds"], artifact["warm_wall_seconds"], strict=True
        )
    )
    assert len(artifact["provenance"]["parity_rows"]) == 170
    assert counters == {"predictor": 175, "predict": 510, "parity": 340}
    assert clock.calls == 680

    loaded, digest = _load_timing(
        paths["timing.json"],
        object_ids=object_ids,
        folds=folds,
        ensemble_sha256=ENSEMBLE_HASH,
        spec_sha256=SPEC_HASH,
        split_sha256=SPLIT_HASH,
        blind_inputs_sha256=BLIND_HASH,
    )
    assert len(loaded) == 170
    assert loaded[object_ids[0]] == (0.5, 2.0)
    assert digest == sha256_file(paths["timing.json"])


def test_capture_parity_failure_is_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths, _, _ = _install_capture_fakes(
        monkeypatch, tmp_path, fail_first_parity=True
    )

    with pytest.raises(DownstreamBenchmarkError, match="differs from the frozen ensemble"):
        timing.capture_neural_timing(
            paths["spec"],
            paths["checksum"],
            paths["split"],
            paths["blind"],
            paths["ensemble"],
            paths["bundles"],
            paths["dump"],
            paths["timing.json"],
            device="cpu",
        )
    assert not paths["timing.json"].exists()


def test_bundle_binding_rejects_source_checkpoint_mismatch(tmp_path: Path):
    bundle_root = tmp_path / "bundles"
    directory = bundle_root / "k3-oof-fold-0"
    directory.mkdir(parents=True)
    split_row: dict[str, object] = {
        "fold": 0,
        "train_ids": ["train"],
        "validation_ids": ["validation"],
        "calibration_ids": ["calibration"],
        "test_ids": ["test"],
    }
    expected_source_hash = "a" * 64
    checkpoint_hashes: dict[str, str] = {}
    members: list[dict[str, object]] = []
    for seed in (17, 42, 137, 777, 2027):
        weight = directory / f"model-{seed}.safetensors"
        weight.write_bytes(f"safe-{seed}".encode("ascii"))
        checkpoint_name = f"real-fold-0-seed-{seed}.pt"
        checkpoint_hashes[checkpoint_name] = expected_source_hash
        members.append(
            {
                "seed": seed,
                "file": weight.name,
                "sha256": sha256_file(weight),
                "source_checkpoint_sha256": (
                    "b" * 64 if seed == 2027 else expected_source_hash
                ),
            }
        )
    manifest = {
        "schema": "delphi.k3-ensemble-bundle.v1",
        "fold": 0,
        "splits_sha256": SPLIT_HASH,
        "bundle_id": "synthetic-fold-0",
        "object_roles": {
            name: split_row[name]
            for name in ("train_ids", "validation_ids", "calibration_ids", "test_ids")
        },
        "members": members,
        "model_config": {"synthetic": True},
    }
    (directory / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DownstreamBenchmarkError, match="not bound to the frozen checkpoint"):
        timing._bind_bundles(
            bundle_root,
            split_rows=[split_row],
            split_sha256=SPLIT_HASH,
            checkpoint_sha256=checkpoint_hashes,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("absent", "not aligned"),
        ("extra", "not bound to the frozen study"),
        ("unverified", "unverified model output"),
    ],
)
def test_load_timing_rejects_absent_extra_or_unverified_provenance(
    tmp_path: Path, mutation: str, message: str
):
    payload = _valid_timing_payload()
    provenance = payload["provenance"]
    assert isinstance(provenance, dict)
    if mutation == "absent":
        payload.pop("provenance")
    elif mutation == "extra":
        provenance["unfrozen_extra_field"] = True
    else:
        parity_rows = provenance["parity_rows"]
        assert isinstance(parity_rows, list)
        parity_rows[0]["cold"]["passed"] = False
    path = tmp_path / "timing.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    object_ids, folds, _ = _object_layout()

    with pytest.raises(DownstreamBenchmarkError, match=message):
        _load_timing(
            path,
            object_ids=object_ids,
            folds=folds,
            ensemble_sha256=ENSEMBLE_HASH,
            spec_sha256=SPEC_HASH,
            split_sha256=SPLIT_HASH,
            blind_inputs_sha256=BLIND_HASH,
        )
