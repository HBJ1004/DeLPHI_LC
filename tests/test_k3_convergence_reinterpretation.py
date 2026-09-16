"""Small, network-free contracts for the historical convergence reinterpretation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import lc_pipeline.k3.convergence_reinterpretation as replay
from lc_pipeline.v2.data import canonical_json, sha256_file


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _digest_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="ascii")
    return sha256_file(path)


def _summary(
    cells: list[dict[str, object]], records: list[dict[str, object]], arm: str
) -> dict[str, object]:
    selected = [record for record in records if record["identity"]["arm"] == arm]
    best = min(
        selected,
        key=lambda record: (
            record["result"]["relative_rms_from_output"],
            record["identity"]["start_index"],
        ),
    )
    result = best["result"]
    return {
        "all_cells_recorded": True,
        "cold_neural_wall_seconds": 0.0 if arm == "baseline" else 0.2,
        "completed": True,
        "completion_counts": {"converged": 6},
        "inversion_wall_seconds": 6.0,
        "selectable": True,
        "selected_fit": {
            "criterion": "minimum_final_relative_rms",
            "start_index": best["identity"]["start_index"],
            "final_relative_rms": result["relative_rms_from_output"],
            "final_lambda_deg": result["final_lambda_deg"],
            "final_beta_deg": result["final_beta_deg"],
        },
        "starts_requested": 6,
        "wall_seconds_cold": 6.0 if arm == "baseline" else 6.2,
        "wall_seconds_warm": 6.0 if arm == "baseline" else 6.1,
        "warm_neural_wall_seconds": 0.0 if arm == "baseline" else 0.1,
    }


def _fixture_graph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """One tolerance but the full 30×3×12 execution shape for a quick unit test."""
    tolerance = 0.01
    monkeypatch.setattr(replay, "TOLERANCES", (tolerance,))
    monkeypatch.setattr(replay, "EXPECTED_LEGACY_NUMERIC_FAILURES", {tolerance: 0})
    object_ids = [f"object-{index:02d}" for index in range(30)]
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(
        "".join(
            json.dumps({"object_id": value, "solutions": [{"vector": [1.0, 0.0, 0.0]}]}) + "\n"
            for value in object_ids
        ),
        encoding="utf-8",
    )
    spec = tmp_path / "study.yaml"
    spec.write_text("schema: fixture\n", encoding="utf-8")
    manifest = tmp_path / "development.json"
    _write_json(
        manifest,
        {
            "schema": "delphi.k3-convergence-object-subset.v1",
            "role": "development",
            "study_spec_sha256": sha256_file(spec),
            "object_ids": object_ids,
        },
    )
    lock = tmp_path / "lock.json"
    lock_value = {
        "schema": "delphi.k3-convergence-study-lock.v1",
        "study_spec_sha256": sha256_file(spec),
        "development_manifest_sha256": sha256_file(manifest),
        "reference_catalog_sha256": sha256_file(catalog),
        "blind_inputs_sha256": "a" * 64,
        "neural_timing_sha256": "b" * 64,
        "source_archive_sha256": "c" * 64,
        "source_payload_sha256": "d" * 64,
        "source_tree_sha256": "e" * 64,
        "executable_sha256": "f" * 64,
        "compiler_command": ["cc", "--version"],
        "compiler_executable": "fixture-cc",
        "compiler_executable_sha256": "1" * 64,
        "compiler_version": "fixture",
    }
    _write_json(lock, lock_value)
    monkeypatch.setattr(replay, "EXPECTED_STUDY_SPEC_SHA256", sha256_file(spec))
    monkeypatch.setattr(replay, "EXPECTED_REVISED_LOCK_SHA256", sha256_file(lock))
    contract = {
        "schema": "delphi.k3-convergence-execution-contract.v1",
        "study_spec_sha256": sha256_file(spec),
        "study_lock_sha256": sha256_file(lock),
        "role": "development",
        "convergence_tolerance": tolerance,
        "iteration_cap": 1000,
        "timeout_seconds_per_start": 300,
        "timing_repeats": 3,
        "starts_per_arm": 6,
        "fit_selection": "minimum_final_relative_rms_without_reference_axis_access",
        "period_policy": "fixed_first_frozen_catalog_solution",
        "candidate_semantics": "guided_k3_initialization",
        "arm_names": ["baseline", "candidate"],
        "blind_inputs_sha256": lock_value["blind_inputs_sha256"],
        "cohort_manifest_sha256": lock_value["development_manifest_sha256"],
        "neural_timing_sha256": lock_value["neural_timing_sha256"],
        "source_archive_sha256": lock_value["source_archive_sha256"],
        "source_payload_sha256": lock_value["source_payload_sha256"],
        "source_tree_sha256": lock_value["source_tree_sha256"],
        "executable_sha256": lock_value["executable_sha256"],
        "compiler_command": lock_value["compiler_command"],
        "compiler_executable": lock_value["compiler_executable"],
        "compiler_executable_sha256": lock_value["compiler_executable_sha256"],
        "compiler_version": lock_value["compiler_version"],
    }
    root = tmp_path / "parent"
    execution_path = root / "tolerance-0p01" / "blind-execution.json"
    rows = []
    contract_hash = hashlib.sha256(canonical_json(contract).encode()).hexdigest()
    for object_index, object_id in enumerate(object_ids):
        repeats, markers = [], []
        record_contract = {
            **contract,
            "object_id": object_id,
            "fold": object_index % 5,
            "lightcurve_sha256": "9" * 64,
            "period_hours": 8.0,
        }
        record_contract_hash = hashlib.sha256(canonical_json(record_contract).encode()).hexdigest()
        for repeat_index in range(3):
            cells, records = [], []
            for start_index in range(6):
                for sequence_arm, arm in enumerate(("baseline", "candidate")):
                    run = (
                        execution_path.parent
                        / "runs"
                        / object_id
                        / f"repeat-{repeat_index}"
                        / arm
                        / f"start-{start_index}"
                    )
                    stdout_hash = _digest_text(run / "stdout.log", "")
                    stderr_hash = _digest_text(run / "stderr.log", "")
                    parameter_hash = _digest_text(run / "parameters.txt", "parameters\n")
                    lc_hash = _digest_text(run / "modelled-lightcurve.txt", "model\n")
                    par_hash = _digest_text(run / "solution-parameters.txt", "solution\n")
                    area_hash = _digest_text(run / "solution-areas.txt", "area\n")
                    result = {
                        "timed_out": False,
                        "adapter_error": None,
                        "return_code": 0,
                        "iterations": 5,
                        "chi2": 1.0,
                        "deviation": 0.1,
                        "final_lambda_deg": 10.0 + start_index,
                        "final_beta_deg": 20.0,
                        "final_period_hours": 8.0,
                        "relative_rms_from_output": 0.01 + start_index / 1000.0,
                        "output_validation_error": None,
                        "output_lightcurve_sha256": lc_hash,
                        "output_parameter_sha256": par_hash,
                        "output_area_sha256": area_hash,
                        "stdout_sha256": stdout_hash,
                        "stderr_sha256": stderr_hash,
                        "provenance": {
                            "executable_sha256": contract["executable_sha256"],
                            "source_tree_sha256": contract["source_tree_sha256"],
                            "input_sha256": "9" * 64,
                            "parameter_sha256": parameter_hash,
                        },
                    }
                    record = {
                        "schema": "delphi.k3-convergence-convexinv-run.v1",
                        "identity": {
                            "object_id": object_id,
                            "arm": arm,
                            "repeat_index": repeat_index,
                            "start_index": start_index,
                            "convergence_tolerance": tolerance,
                            "timeout_seconds": 300.0,
                            "period_hours": 8.0,
                            "repeat_arm_order": ["baseline", "candidate"],
                            "execution_contract_sha256": record_contract_hash,
                        },
                        "completion": "converged",
                        "result": result,
                        "execution_contract": record_contract,
                        "logs": {
                            "stdout": "stdout.log",
                            "stderr": "stderr.log",
                            "stdout_sha256": stdout_hash,
                            "stderr_sha256": stderr_hash,
                        },
                    }
                    record_path = run / "result.json"
                    _write_json(record_path, record)
                    records.append(record)
                    relative = record_path.relative_to(execution_path.parent).as_posix()
                    cells.append(
                        {
                            "arm": arm,
                            "start_index": start_index,
                            "sequence_index": 2 * start_index + sequence_arm,
                            "path": relative,
                            "sha256": sha256_file(record_path),
                        }
                    )
            repeat = {
                "schema": "delphi.k3-convergence-repeat-complete.v1",
                "plan": {
                    "object_id": object_id,
                    "fold": object_index % 5,
                    "repeat_index": repeat_index,
                    "convergence_tolerance": tolerance,
                    "timeout_seconds_per_start": 300.0,
                    "execution_contract_sha256": record_contract_hash,
                    "arm_order": ["baseline", "candidate"],
                },
                "cells": cells,
                "baseline": _summary(cells, records, "baseline"),
                "candidate": _summary(cells, records, "candidate"),
            }
            marker = (
                execution_path.parent
                / "runs"
                / object_id
                / f"repeat-{repeat_index}"
                / "repeat-complete.json"
            )
            _write_json(marker, repeat)
            repeats.append(repeat)
            markers.append(
                {
                    "repeat_index": repeat_index,
                    "path": marker.relative_to(execution_path.parent).as_posix(),
                    "sha256": sha256_file(marker),
                }
            )
        rows.append(
            {
                "object_id": object_id,
                "fold": object_index % 5,
                "period_hours": 8.0,
                "lightcurve_sha256": "9" * 64,
                "repeats": repeats,
                "repeat_markers": markers,
            }
        )
    execution = {
        "schema": "delphi.k3-convergence-blind-execution.v1",
        "phase": "label_blind_execution_and_minimum_rms_fit_selection",
        "reference_catalog_opened": False,
        "study_lock_sha256": sha256_file(lock),
        "execution_contract": contract,
        "execution_contract_sha256": contract_hash,
        "role": "development",
        "cohort_manifest_sha256": sha256_file(manifest),
        "convergence_tolerance": tolerance,
        "object_ids": object_ids,
        "object_count": 30,
        "rows": rows,
    }
    _write_json(execution_path, execution)
    selection = {
        "schema": "delphi.k3-convergence-development-selection.v1",
        "status": "no_eligible_tolerance_stop",
        "study_spec_sha256": sha256_file(spec),
        "study_lock_sha256": sha256_file(lock),
        "score_artifacts": [
            {"convergence_tolerance": tolerance, "execution_sha256": sha256_file(execution_path)}
        ],
    }
    _write_json(root / "selection.json", selection)
    old_score = root / "tolerance-0p01" / "score.json"
    _write_json(old_score, {"not_permitted_during_reinterpretation": True})
    monkeypatch.setattr(
        replay, "EXPECTED_PARENT_SELECTION_SHA256", sha256_file(root / "selection.json")
    )
    return {
        "root": root,
        "out": tmp_path / "out",
        "spec": spec,
        "lock": lock,
        "manifest": manifest,
        "catalog": catalog,
        "execution": execution_path,
        "old_score": old_score,
    }


def test_reinterpretation_is_create_once_label_blind_and_leaves_parent_unchanged(
    tmp_path, monkeypatch
):
    paths = _fixture_graph(tmp_path, monkeypatch)
    before = sha256_file(paths["execution"])
    result = replay.reinterpret_development_execution_graph(
        parent_root=paths["root"],
        output_directory=paths["out"],
        study_spec_path=paths["spec"],
        revised_lock_path=paths["lock"],
        development_manifest_path=paths["manifest"],
    )
    assert sha256_file(paths["execution"]) == before
    assert result["reference_catalog_opened"] is False
    assert result["raw_artifacts_modified"] is False
    assert len(result["executions"][0]["rows"]) == 30
    assert len(result["executions"][0]["rows"][0]["repeats"][0]["starts"]) == 12
    with pytest.raises(replay.ConvergenceReinterpretationError, match="output"):
        replay.reinterpret_development_execution_graph(
            parent_root=paths["root"],
            output_directory=paths["out"],
            study_spec_path=paths["spec"],
            revised_lock_path=paths["lock"],
            development_manifest_path=paths["manifest"],
        )


def test_reinterpretation_rejects_tamper_and_path_traversal_before_writing(tmp_path, monkeypatch):
    paths = _fixture_graph(tmp_path, monkeypatch)
    record = (
        paths["execution"].parent
        / "runs"
        / "object-00"
        / "repeat-0"
        / "baseline"
        / "start-0"
        / "result.json"
    )
    record.write_text("{}", encoding="utf-8")
    with pytest.raises(replay.ConvergenceReinterpretationError, match="hash"):
        replay.reinterpret_development_execution_graph(
            parent_root=paths["root"],
            output_directory=paths["out"],
            study_spec_path=paths["spec"],
            revised_lock_path=paths["lock"],
            development_manifest_path=paths["manifest"],
        )
    assert not paths["out"].exists()


def test_scoring_requires_an_unmodified_sealed_receipt_before_reference_access(
    tmp_path, monkeypatch
):
    paths = _fixture_graph(tmp_path, monkeypatch)
    result = replay.reinterpret_development_execution_graph(
        parent_root=paths["root"],
        output_directory=paths["out"],
        study_spec_path=paths["spec"],
        revised_lock_path=paths["lock"],
        development_manifest_path=paths["manifest"],
    )
    receipt = Path(result["receipt_path"])
    receipt.write_text(receipt.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(replay.ConvergenceReinterpretationError, match="receipt hash"):
        replay.score_reinterpreted_development(
            receipt_path=receipt,
            expected_receipt_sha256=result["receipt_sha256"],
            reference_catalog_path=paths["catalog"],
            output_path=tmp_path / "score.json",
        )


def test_sealed_scoring_uses_the_original_development_rule(tmp_path, monkeypatch):
    paths = _fixture_graph(tmp_path, monkeypatch)
    receipt = replay.reinterpret_development_execution_graph(
        parent_root=paths["root"],
        output_directory=paths["out"],
        study_spec_path=paths["spec"],
        revised_lock_path=paths["lock"],
        development_manifest_path=paths["manifest"],
    )
    result = replay.score_reinterpreted_development(
        receipt_path=receipt["receipt_path"],
        expected_receipt_sha256=receipt["receipt_sha256"],
        reference_catalog_path=paths["catalog"],
        output_path=tmp_path / "score.json",
    )
    assert result["schema"] == replay.REINTERPRETATION_SCORE_SCHEMA
    assert result["reference_catalog_opened"] is True
    assert (
        result["selection"]["selection_rule"]
        == "largest_runtime_ratio_lower_bound_then_smaller_tolerance"
    )


def test_reinterpretation_does_not_open_labels_or_historical_scores(tmp_path, monkeypatch):
    paths = _fixture_graph(tmp_path, monkeypatch)
    original_read_text = Path.read_text
    prohibited = {paths["catalog"].resolve(), paths["old_score"].resolve()}

    def guarded_read_text(path: Path, *args, **kwargs):
        if path.resolve() in prohibited:
            raise AssertionError("selection attempted to open labels or a historical score")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    result = replay.reinterpret_development_execution_graph(
        parent_root=paths["root"],
        output_directory=paths["out"],
        study_spec_path=paths["spec"],
        revised_lock_path=paths["lock"],
        development_manifest_path=paths["manifest"],
    )
    assert result["reference_catalog_opened"] is False


def test_reinterpretation_rejects_marker_path_traversal_before_writing(tmp_path, monkeypatch):
    paths = _fixture_graph(tmp_path, monkeypatch)
    execution = json.loads(paths["execution"].read_text(encoding="utf-8"))
    execution["rows"][0]["repeat_markers"][0]["path"] = "../outside/repeat-complete.json"
    _write_json(paths["execution"], execution)
    selection_path = paths["root"] / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["score_artifacts"][0]["execution_sha256"] = sha256_file(paths["execution"])
    _write_json(selection_path, selection)
    monkeypatch.setattr(replay, "EXPECTED_PARENT_SELECTION_SHA256", sha256_file(selection_path))
    with pytest.raises(replay.ConvergenceReinterpretationError, match="escapes"):
        replay.reinterpret_development_execution_graph(
            parent_root=paths["root"],
            output_directory=paths["out"],
            study_spec_path=paths["spec"],
            revised_lock_path=paths["lock"],
            development_manifest_path=paths["manifest"],
        )
    assert not paths["out"].exists()


def test_script_help_bootstraps_checkout_without_pythonpath(tmp_path):
    script = Path(__file__).parents[1] / "repro" / "reinterpret_k3_convergence_development.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr
    assert "create a label-blind interpretation receipt" in result.stdout
