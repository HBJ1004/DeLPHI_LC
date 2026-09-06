"""Downstream fixed-period benchmark contract tests."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from lc_pipeline.k3.downstream import (
    DownstreamBenchmarkError,
    aggregate_arm,
    alias_period_intervals,
    evaluate_downstream_gate,
    run_fixed_period_benchmark,
    signed_starts_from_axes,
)


def test_aggregate_arm_retains_failed_nan_rms_for_recovery_rate() -> None:
    aggregate = aggregate_arm(
        "candidate",
        [1.0, 2.0, 3.0],
        [0.5, 1.0, 1.5],
        [4, 5, 6],
        [0.8, np.nan, 1.2],
        [0.8, 0.9, 1.2],
    )
    assert aggregate.successful_objects == 2
    assert aggregate.failed_objects == 1
    # Recovery is evaluated among successful fits; the failed object is
    # retained separately in ``failed_objects`` and downstream success masks.
    assert aggregate.recovery_rate == 1.0


def test_k3_axes_expand_to_six_signed_convexinv_starts() -> None:
    axes = np.eye(3)
    starts = signed_starts_from_axes(axes, period_hours=8.0)
    assert len(starts) == 6
    assert starts[0].period_hours == 8.0
    assert starts[0].lambda_deg != starts[1].lambda_deg


def test_alias_intervals_are_clipped_to_declared_domain() -> None:
    intervals = alias_period_intervals(3.0)
    assert intervals[0][0] >= 2.0 and intervals[-1][1] <= 200.0
    assert len(intervals) == 2  # P/2 lies below the locked 2-hour lower bound.


def test_downstream_gate_rejects_no_speedup_and_accepts_strong_smoke_case() -> None:
    baseline_time = np.full(30, 10.0)
    candidate_time = np.full(30, 7.0)
    baseline_rms = np.full(30, 1.0)
    candidate_rms = np.full(30, 1.001)
    verdict = evaluate_downstream_gate(
        baseline_time, candidate_time, baseline_rms, candidate_rms,
        bootstrap_resamples=300,
    )
    assert verdict.passed and verdict.speed_ratio > 1.25
    failed = evaluate_downstream_gate(
        baseline_time, baseline_time, baseline_rms, candidate_rms,
        bootstrap_resamples=300,
    )
    assert not failed.passed


def test_downstream_gate_retains_failures_and_uses_paired_successes_for_rms() -> None:
    verdict = evaluate_downstream_gate(
        [10.0] * 4,
        [5.0] * 4,
        [1.0, 1.0, np.nan, 1.0],
        [1.0, np.nan, 1.0, 1.0],
        baseline_success=[True, True, False, True],
        candidate_success=[True, False, True, True],
        bootstrap_resamples=100,
    )
    assert verdict.passed
    assert verdict.paired_successful_objects == 2
    assert verdict.recovery_rate_inferiority == 0.0
    assert verdict.geometric_mean_rms_ratio == 1.0

    with pytest.raises(DownstreamBenchmarkError, match="paired successful"):
        evaluate_downstream_gate(
            [10.0] * 3,
            [5.0] * 3,
            [1.0, 1.0, 1.0],
            [1.0, np.nan, np.nan],
            candidate_success=[True, False, False],
            bootstrap_resamples=100,
        )


def test_downstream_gate_rejects_nonfinite_rms_marked_successful() -> None:
    with pytest.raises(DownstreamBenchmarkError, match="marked successful"):
        evaluate_downstream_gate(
            [10.0] * 3,
            [5.0] * 3,
            [1.0, 1.0, 1.0],
            [1.0, np.nan, 1.0],
            candidate_success=[True, True, True],
            bootstrap_resamples=100,
        )


def test_fixed_period_runner_uses_matched_six_start_arms_and_neural_time(
    tmp_path, monkeypatch
) -> None:
    object_ids = [f"object-{fold}" for fold in range(5)]
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps(
            {
                "folds": [
                    {"fold": fold, "test_ids": [object_ids[fold]]}
                    for fold in range(5)
                ]
            }
        ),
        encoding="utf-8",
    )
    dump = tmp_path / "dump"
    rows = []
    for object_id in object_ids:
        lightcurve = dump / "files" / object_id / "lc.txt"
        lightcurve.parent.mkdir(parents=True)
        lightcurve.write_text("test\n", encoding="ascii")
        rows.append(
            {
                "object_id": object_id,
                "lightcurve": {
                    "source_path": f"files/{object_id}/lc.txt",
                    "source_sha256": hashlib.sha256(lightcurve.read_bytes()).hexdigest(),
                },
                "solutions": [
                    {
                        "period_hours": 8.0,
                        "vector": [1.0, 0.0, 0.0],
                    }
                ],
            }
        )
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    ensemble = tmp_path / "real-oof-ensemble.npz"
    np.savez_compressed(
        ensemble,
        schema=np.asarray("delphi.k3-real-oof-ensemble.v1"),
        object_ids=np.asarray(object_ids),
        refined_axes=np.repeat(np.eye(3)[None, :, :], 5, axis=0),
        inference_wall_seconds=np.full(5, 0.1),
    )
    executable = tmp_path / "convexinv"
    executable.write_text("binary", encoding="ascii")
    archive = tmp_path / "version_0.2.1.tar.gz"
    archive.write_text("archive", encoding="ascii")
    source = tmp_path / "source"
    source.mkdir()
    calls = []

    def fake_run_record(**kwargs):
        calls.append(kwargs)
        arm = kwargs["arm"]
        return {
            "identity": {
                "start_index": kwargs["start_index"],
            },
            "result": {
                "return_code": 0,
                "timed_out": False,
                "wall_time_seconds": 2.0 if arm == "baseline" else 1.0,
                "process_cpu_time_seconds": 0.5,
                "iterations": 10,
                "relative_rms_from_output": 1.0,
                "final_lambda_deg": 0.0,
                "final_beta_deg": 0.0,
                "provenance": {
                    "source_tree_sha256": "b" * 64,
                    "executable_sha256": hashlib.sha256(
                        executable.read_bytes()
                    ).hexdigest(),
                    "compiler_version": "cc test",
                },
            },
        }

    monkeypatch.setattr(
        "lc_pipeline.k3.downstream._run_record", fake_run_record
    )
    result = run_fixed_period_benchmark(
        executable=executable,
        source_root=source,
        source_archive=archive,
        ensemble_path=ensemble,
        catalog_path=catalog,
        dump_root=dump,
        split_path=splits,
        output_directory=tmp_path / "benchmark",
    )
    assert len(calls) == 5 * 2 * 6
    assert all(call["start"].period_hours == 8.0 for call in calls)
    assert result["verdict"]["passed"] is True
    payload = json.loads(
        (tmp_path / "benchmark" / "fixed-period-rows.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["rows"][0]["candidate"]["wall_seconds"] == 6.1
    assert payload["rows"][0]["baseline"]["wall_seconds"] == 12.0
