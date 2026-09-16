"""Contracts for correction-bound locked-execution authorization."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import lc_pipeline.k3.convergence_correction_bridge as bridge
import lc_pipeline.k3.convergence_study as study
from lc_pipeline.v2.data import canonical_json, sha256_file


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path | str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    spec = tmp_path / "spec.yaml"
    spec.write_text("schema: fixture\n", encoding="utf-8")
    manifest = tmp_path / "locked.json"
    object_ids = [f"locked-{index:03d}" for index in range(140)]
    _write(
        manifest,
        {
            "schema": "delphi.k3-convergence-object-subset.v1",
            "role": "locked_evaluation",
            "study_spec_sha256": sha256_file(spec),
            "object_ids": object_ids,
        },
    )
    lock = tmp_path / "lock.json"
    hashes = {
        "blind_inputs_sha256": "a" * 64,
        "neural_timing_sha256": "b" * 64,
        "source_archive_sha256": "c" * 64,
        "source_payload_sha256": "d" * 64,
        "source_tree_sha256": "e" * 64,
        "executable_sha256": "f" * 64,
        "compiler_executable_sha256": "1" * 64,
    }
    _write(
        lock,
        {
            "schema": "delphi.k3-convergence-study-lock.v1",
            "study_spec_sha256": sha256_file(spec),
            "locked_manifest_sha256": sha256_file(manifest),
            "reference_catalog_sha256": "2" * 64,
            **hashes,
            "settings": {
                "convergence_tolerance_grid": [0.01, 0.003, 0.001, 0.0003, 0.0001],
                "iteration_cap": 1000,
                "timeout_seconds_per_start": 300,
                "timing_repeats": 3,
                "include_warm_neural_inference_time": True,
                "report_cold_neural_inference_sensitivity": True,
                "period": "first_frozen_catalog_solution_fixed_in_both_arms",
            },
        },
    )
    monkeypatch.setattr(bridge, "EXPECTED_STUDY_SPEC_SHA256", sha256_file(spec))
    monkeypatch.setattr(bridge, "EXPECTED_REVISED_LOCK_SHA256", sha256_file(lock))
    receipt = tmp_path / "receipt.json"
    _write(
        receipt,
        {
            "schema": "delphi.k3-convergence-angle-reinterpretation.v1",
            "phase": "label_blind_historical_output_reinterpretation",
            "reference_catalog_opened": False,
            "study_spec_sha256": sha256_file(spec),
            "revised_study_lock_sha256": sha256_file(lock),
            "reference_catalog_sha256": "2" * 64,
        },
    )
    score = tmp_path / "score.json"
    _write(
        score,
        {
            "schema": "delphi.k3-convergence-angle-reinterpretation-score.v1",
            "phase": "reference_axis_scoring_after_sealed_label_blind_reinterpretation",
            "receipt_sha256": sha256_file(receipt),
            "reference_catalog_sha256": "2" * 64,
            "selection": {
                "status": "selected",
                "selected_tolerance": 0.0003,
                "selection_rule": "largest_runtime_ratio_lower_bound_then_smaller_tolerance",
            },
        },
    )
    return {
        "receipt": receipt,
        "score": score,
        "lock": lock,
        "manifest": manifest,
        "authorization": tmp_path / "authorization.json",
        "locked": tmp_path / "locked-execution",
    }


def _authorize(paths: dict[str, Path | str]) -> dict[str, object]:
    return bridge.create_correction_locked_authorization(
        receipt_path=paths["receipt"],
        receipt_sha256=sha256_file(Path(paths["receipt"])),
        score_path=paths["score"],
        score_sha256=sha256_file(Path(paths["score"])),
        revised_lock_path=paths["lock"],
        locked_manifest_path=paths["manifest"],
        locked_execution_directory=paths["locked"],
        authorization_path=paths["authorization"],
    )


def test_authorization_binds_selected_score_and_all_locked_conditions(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch)
    result = _authorize(paths)
    assert result["score_sha256"] == sha256_file(Path(paths["score"]))
    assert result["runner_contract"]["convergence_tolerance"] == 0.0003
    assert result["runner_contract"]["starts_per_arm"] == 6
    assert result["runner_contract"]["timing_repeats"] == 3
    assert result["locked_object_count"] == 140
    assert not Path(paths["locked"]).exists()
    runner = bridge.correction_locked_runner_parameters(authorization_path=paths["authorization"])
    assert runner["locked_execution_directory"] == str(Path(paths["locked"]).resolve())
    assert runner["convergence_tolerance"] == 0.0003
    verified = bridge.validate_correction_locked_authorization(
        authorization_path=paths["authorization"],
        receipt_path=paths["receipt"],
        score_path=paths["score"],
        revised_lock_path=paths["lock"],
        locked_manifest_path=paths["manifest"],
    )
    assert verified["authorization_sha256"] == result["authorization_sha256"]
    with pytest.raises(bridge.CorrectionLockedAuthorizationError, match="authorization"):
        _authorize(paths)


def test_authorization_refuses_modified_score_or_receipt(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch)
    score = Path(paths["score"])
    original_score_hash = sha256_file(score)
    score.write_text(score.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(bridge.CorrectionLockedAuthorizationError, match="score.*hash"):
        bridge.create_correction_locked_authorization(
            receipt_path=paths["receipt"],
            receipt_sha256=sha256_file(Path(paths["receipt"])),
            score_path=score,
            score_sha256=original_score_hash,
            revised_lock_path=paths["lock"],
            locked_manifest_path=paths["manifest"],
            locked_execution_directory=paths["locked"],
            authorization_path=paths["authorization"],
        )
    paths = _fixture(tmp_path / "receipt-case", monkeypatch)
    receipt = Path(paths["receipt"])
    original_receipt_hash = sha256_file(receipt)
    receipt.write_text(receipt.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(bridge.CorrectionLockedAuthorizationError, match="receipt.*hash"):
        bridge.create_correction_locked_authorization(
            receipt_path=receipt,
            receipt_sha256=original_receipt_hash,
            score_path=paths["score"],
            score_sha256=sha256_file(Path(paths["score"])),
            revised_lock_path=paths["lock"],
            locked_manifest_path=paths["manifest"],
            locked_execution_directory=paths["locked"],
            authorization_path=paths["authorization"],
        )


def test_authorization_refuses_no_eligible_or_wrong_selected_tolerance(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch)
    score = json.loads(Path(paths["score"]).read_text(encoding="utf-8"))
    score["selection"]["status"] = "no_eligible_tolerance_stop"
    score["selection"]["selected_tolerance"] = None
    _write(Path(paths["score"]), score)
    with pytest.raises(bridge.CorrectionLockedAuthorizationError, match="does not select"):
        _authorize(paths)


def test_bridge_cli_help_runs_from_outside_checkout_without_pythonpath(tmp_path):
    script = Path(__file__).parents[1] / "repro" / "authorize_k3_correction_locked_execution.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr
    assert "correction-bound authorization" in result.stdout


def test_correction_runner_command_is_exposed_without_running_solver():
    script = Path(__file__).parents[1] / "repro" / "run_k3_convergence_study.py"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "repro.run_k3_convergence_study",
            "execute-correction-locked",
            "--help",
        ],
        cwd=script.parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--correction-authorization" in result.stdout


def test_frozen_runner_accepts_only_the_validated_correction_adapter_before_work(
    tmp_path, monkeypatch
):
    lock_path = tmp_path / "lock.json"
    lock_path.write_text("{}\n", encoding="utf-8")
    locked_manifest = tmp_path / "locked.json"
    locked_manifest.write_text("{}\n", encoding="utf-8")
    destination = tmp_path / "locked-output"
    (destination / "blind-execution.json").parent.mkdir(parents=True)
    (destination / "blind-execution.json").write_text("{}\n", encoding="utf-8")
    frozen_hashes = {
        "blind_inputs_sha256": "a" * 64,
        "neural_timing_sha256": "b" * 64,
        "source_archive_sha256": "c" * 64,
        "source_payload_sha256": "d" * 64,
        "source_tree_sha256": "e" * 64,
        "executable_sha256": "f" * 64,
        "compiler_executable_sha256": "1" * 64,
    }
    lock = {
        **frozen_hashes,
        "locked_manifest_sha256": "2" * 64,
        "study_spec_sha256": "3" * 64,
        "compiler_command": ["cc", "--version"],
        "compiler_executable": "cc",
        "compiler_version": "fixture",
    }
    study_description = {
        "shared_conditions": {
            "convergence_tolerance_grid": [0.01, 0.003, 0.001, 0.0003, 0.0001],
            "iteration_cap": 1000,
            "timeout_seconds_per_start": 300,
            "timing_repeats": 3,
            "repeat_order_seed": 20260910,
        }
    }
    cohorts = {
        "development": tuple(),
        "locked_evaluation": tuple(f"o{index}" for index in range(140)),
    }
    monkeypatch.setattr(
        study, "validate_study_lock", lambda **_kwargs: (lock, study_description, {}, {}, cohorts)
    )
    adapter_calls: list[Path] = []
    authorization = {
        "revised_study_lock_path": str(lock_path),
        "locked_manifest_path": str(locked_manifest),
        "locked_manifest_sha256": lock["locked_manifest_sha256"],
        "locked_execution_directory": str(destination),
        "runner_contract": {"convergence_tolerance": 0.0003},
    }

    def validated_adapter(*, authorization_path):
        adapter_calls.append(Path(authorization_path))
        return authorization

    monkeypatch.setattr(
        bridge, "validate_correction_locked_authorization_artifact", validated_adapter
    )
    monkeypatch.setattr(
        study, "_validated_execution", lambda *_args, **_kwargs: {"execution_contract": {}}
    )
    with pytest.raises(study.DownstreamBenchmarkError, match="completed blind execution"):
        study.execute_blind_cohort(
            role="locked_evaluation",
            lock_path=lock_path,
            spec_path=tmp_path / "spec",
            spec_checksum_path=tmp_path / "checksum",
            split_path=tmp_path / "split",
            development_manifest_path=tmp_path / "development",
            locked_manifest_path=locked_manifest,
            blind_inputs_path=tmp_path / "blind",
            ensemble_path=tmp_path / "ensemble",
            neural_timing_path=tmp_path / "timing",
            source_archive=tmp_path / "source.tar.gz",
            source_root=tmp_path / "source",
            executable=tmp_path / "convexinv",
            dump_root=tmp_path / "dump",
            output_directory=destination,
            convergence_tolerance=0.0003,
            correction_authorization_path=tmp_path / "authorization.json",
        )
    assert adapter_calls == [tmp_path / "authorization.json"]
