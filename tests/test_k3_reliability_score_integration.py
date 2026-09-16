import csv
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lc_pipeline.k3 import reliability_forward, reliability_sampling, reliability_scoring
from lc_pipeline.k3 import reliability_study as study
from lc_pipeline.v2 import data as v2_data

AXES = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _prediction(object_id: str, fold: int, condition: str, *, status: str = "ok") -> dict:
    return {
        "object_id": object_id,
        "fold": fold,
        "condition": condition,
        "condition_family": condition.split("-", 1)[0],
        "repeat": 0,
        "seed": 20260915,
        "status": status,
        "axes": AXES if status == "ok" else None,
        "model_id": "k3",
        "model_count": 5,
        "source_sha256": "a" * 64,
        "input_sha256": "b" * 64,
        "runtime_s": 0.01,
        "metadata": {
            "n_points": 200,
            "n_sessions": 10,
            "merged_blocks": 10,
            "period_hours": 10.0,
        },
    }


def test_score_materializes_and_resumes_complete_small_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    catalog_file = tmp_path / "catalog-placeholder.json"
    catalog_file.write_text("{}\n", encoding="utf-8")
    conditions = [
        {
            "condition": "full",
            "family": "full",
            "sampling": {"kind": "full"},
            "period_multiplier": 1.0,
            "repeatable": False,
        },
        {
            "condition": "point-fraction-0.5",
            "family": "point",
            "sampling": {"kind": "point", "fraction": 0.5},
            "period_multiplier": 1.0,
            "repeatable": True,
        },
        {
            "condition": "two_d-block_count-1-per_block_cap-5",
            "family": "two_d",
            "sampling": {"kind": "two_d", "block_count": 1, "per_block_cap": 5},
            "period_multiplier": 1.0,
            "repeatable": True,
        },
        {
            "condition": "period-1.01",
            "family": "period",
            "sampling": {"kind": "full"},
            "period_multiplier": 1.01,
            "repeatable": False,
        },
    ]
    objects = [
        {
            "object_id": "asteroid-a",
            "fold": 0,
            "period_hours": 10.0,
            "n_points": 200,
            "merged_blocks": 12,
        },
        {
            "object_id": "asteroid-b",
            "fold": 1,
            "period_hours": 12.0,
            "n_points": 180,
            "merged_blocks": 8,
        },
    ]
    for entry in objects:
        source = tmp_path / f"{entry['object_id']}-lc.txt"
        source.write_text("synthetic integration placeholder\n", encoding="utf-8")
        entry["lightcurve"] = study.binding(source)
    atlases = [
        {
            "fold": fold,
            "axes": AXES,
            "source_ids": [f"train-{fold}-{index}" for index in range(108)],
        }
        for fold in range(5)
    ]
    lock = {
        "schema": study.SCHEMA,
        "objects": objects,
        "conditions": conditions,
        "grid_ids": ["asteroid-a"],
        "atlases": atlases,
        "catalog": study.binding(catalog_file),
    }
    schedule = {"repeats": 1, "holdout_ids": ["asteroid-a"]}

    study.seal(tmp_path / "inference-complete.json", {"durations": {}})
    study.seal(tmp_path / "fits-complete.json", {"durations": {}})
    for entry in objects:
        object_id, fold = entry["object_id"], entry["fold"]
        for config in conditions:
            if config["family"] == "two_d" and object_id not in lock["grid_ids"]:
                continue
            status = (
                "failed"
                if object_id == "asteroid-b" and config["condition"] == "point-fraction-0.5"
                else "ok"
            )
            study.seal(
                tmp_path / "predictions" / object_id / config["condition"] / "repeat-0.json",
                _prediction(object_id, fold, config["condition"], status=status),
            )

    fit_input = tmp_path / "fit-input-placeholder.txt"
    fit_input.write_text("sealed training-only fit input\n", encoding="utf-8")
    for sampling in ("full", "thin10"):
        for arm in ("classical20", "k3", "atlas"):
            directory = tmp_path / "fits" / "asteroid-a" / sampling / arm
            (directory / "start-000").mkdir(parents=True)
            study.seal(
                directory / "result.json",
                {
                    "object_id": "asteroid-a",
                    "fold": 0,
                    "sampling": sampling,
                    "arm": arm,
                    "status": "ok",
                    "selected_fit": {"start_index": 0, "final_relative_rms": 0.1},
                    "fit_input": study.binding(fit_input),
                },
            )

    references = {
        "asteroid-a": SimpleNamespace(solution_vectors=np.asarray([[1.0, 0.0, 0.0]], dtype=float)),
        "asteroid-b": SimpleNamespace(solution_vectors=np.asarray([[0.0, 1.0, 0.0]], dtype=float)),
    }
    monkeypatch.setattr(study, "verify_study", lambda _root: lock)
    monkeypatch.setattr(v2_data, "load_catalog", lambda _path: references)
    withheld_observations = tuple(
        SimpleNamespace(
            time_jd=2_451_000.0 + index,
            flux=1.0 + index / 100,
            sun=(1.0, 0.0, 0.0),
            observer=(0.0, 1.0, 0.0),
        )
        for index in range(20)
    )
    withheld = SimpleNamespace(
        sessions=(SimpleNamespace(observations=withheld_observations, calibrated=0),)
    )
    monkeypatch.setattr(reliability_sampling, "read_lc", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        reliability_sampling,
        "select_holdout",
        lambda _curve: SimpleNamespace(withheld=withheld),
    )
    monkeypatch.setattr(
        reliability_forward,
        "load_solution",
        lambda *_args: SimpleNamespace(longitude_deg=0.0, latitude_deg=0.0),
    )
    monkeypatch.setattr(reliability_forward, "holdout_shape_rmse", lambda _solution, _curves: 0.25)
    monkeypatch.setattr(study, "validate_start", lambda _path: {"validated": True})

    original_score = reliability_scoring.score_reliability
    original_bootstrap = reliability_scoring.stratified_asteroid_bootstrap

    def fast_score(*args, **kwargs):
        kwargs["bootstrap_resamples"] = 20
        return original_score(*args, **kwargs)

    def fast_bootstrap(*args, **kwargs):
        kwargs["resamples"] = 20
        return original_bootstrap(*args, **kwargs)

    monkeypatch.setattr(reliability_scoring, "score_reliability", fast_score)
    monkeypatch.setattr(reliability_scoring, "stratified_asteroid_bootstrap", fast_bootstrap)

    report = study.score(tmp_path, lock, schedule)

    assert report["schema"] == study.SCHEMA
    point = report["sampling"]["models"]["k3"]["point-fraction-0.5"]
    assert point["n_eligible_objects"] == 2
    assert point["n_attempted_repeats"] == 2
    assert point["n_failed_repeats"] == 1
    assert point["failure_rate"] == pytest.approx(0.5)
    assert point["mean_error_deg"] == pytest.approx(45.0)
    grid = report["sampling"]["models"]["k3"]["two_d-block_count-1-per_block_cap-5"]
    assert grid["n_eligible_objects"] == 1
    assert grid["n_ineligible_objects"] == 1
    assert (
        report["sampling"]["matched_comparisons"]["k3"]["two_d-block_count-1-per_block_cap-5"][
            "baseline_condition"
        ]
        == "grid-full"
    )
    assert report["withheld"]["full-k3"]["n_scheduled"] == 1
    assert report["withheld"]["full-k3"]["n_fit_completed"] == 1
    assert report["withheld"]["full-k3"]["n_forward_valid"] == 1
    assert report["withheld"]["full-k3"]["n_completed_pairs"] == 1
    assert report["withheld"]["full-k3"]["withheld_rms"]["estimate"] == 0.25
    assert report["withheld"]["full-k3"]["candidate_minus_classical_rms"]["estimate"] == 0.0
    assert report["withheld"]["thin10-minus-full-k3"]["difference"]["estimate"] == 0.0
    assert len(study.unseal(tmp_path / "withheld-rows.json")) == 6
    assert report["random_control"]["n_objects"] == 2
    assert report["random_control_expected_1000_draws"]["n_objects"] == 2
    assert report["atlas_control"]["n_objects"] == 2

    scored_rows = study.unseal(tmp_path / "scored-rows.json")
    assert len(scored_rows) == 10
    failed = next(
        row
        for row in scored_rows
        if row["object_id"] == "asteroid-b" and row["condition"] == "point-fraction-0.5"
    )
    assert failed["status"] == "failed" and failed["error_deg"] == 90.0
    ineligible = next(
        row
        for row in scored_rows
        if row["object_id"] == "asteroid-b"
        and row["condition"] == "two_d-block_count-1-per_block_cap-5"
    )
    assert ineligible["status"] == "ineligible" and ineligible["error_deg"] is None

    table = list(csv.DictReader(io.StringIO((tmp_path / "sampling-table.csv").read_text())))
    assert len(table) == 5
    assert {row["condition"] for row in table} == {
        "full",
        "point-fraction-0.5",
        "two_d-block_count-1-per_block_cap-5",
        "period-1.01",
        "grid-full",
    }
    assert study.unseal(tmp_path / "report.json") == report
    assert report["report_files"]
    for artifact in report["report_files"]:
        path = tmp_path / artifact["path"]
        assert path.is_file()
        assert path.resolve().is_relative_to(tmp_path.resolve())
        assert study.digest(path) == artifact["sha256"]

    # A completed resume validates all exported paths and returns byte-identical data.
    assert study.score(tmp_path, lock, schedule) == report
    first_export = tmp_path / report["report_files"][0]["path"]
    first_export.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="completed report export changed"):
        study.score(tmp_path, lock, schedule)
