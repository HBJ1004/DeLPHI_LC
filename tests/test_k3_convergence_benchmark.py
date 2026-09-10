"""Contract tests for the separate convergence-stopped benchmark."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from lc_pipeline.k3.convergence_benchmark import (
    CONVERGENCE_ITERATION_CAP,
    OBJECT_SUBSET_SCHEMA,
    _select_object_subset,
    evaluate_convergence_gate,
    run_convergence_benchmark,
)
from lc_pipeline.k3.downstream import DownstreamBenchmarkError
from lc_pipeline.v2.convexinv import ConvexinvParameters, run_convexinv, write_convexinv_parameters
from lc_pipeline.v2.data import sha256_file


def _inputs(tmp_path):
    object_ids = [f"object-{number}" for number in range(5)]
    split = tmp_path / "splits.json"
    split.write_text(
        json.dumps({"folds": [{"fold": i, "test_ids": [object_ids[i]]} for i in range(5)]}),
        encoding="utf-8",
    )
    dump = tmp_path / "dump"
    records = []
    for object_id in object_ids:
        lightcurve = dump / "files" / object_id / "lc.txt"
        lightcurve.parent.mkdir(parents=True)
        lightcurve.write_text("test\n", encoding="ascii")
        records.append(
            {
                "object_id": object_id,
                "lightcurve": {
                    "source_path": f"files/{object_id}/lc.txt",
                    "source_sha256": hashlib.sha256(lightcurve.read_bytes()).hexdigest(),
                },
                "solutions": [{"period_hours": 8.0, "vector": [1.0, 0.0, 0.0]}],
            }
        )
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    ensemble = tmp_path / "ensemble.npz"
    np.savez_compressed(
        ensemble,
        schema=np.asarray("delphi.k3-real-oof-ensemble.v1"),
        object_ids=np.asarray(object_ids),
        refined_axes=np.repeat(np.eye(3)[None, :, :], 5, axis=0),
        inference_wall_seconds=np.full(5, 0.1),
        inference_cold_wall_seconds=np.full(5, 0.4),
        inference_warm_wall_seconds=np.full(5, 0.05),
    )
    executable = tmp_path / "convexinv"
    executable.write_text("binary", encoding="ascii")
    source = tmp_path / "source"
    source.mkdir()
    archive = tmp_path / "source.tar.gz"
    archive.write_text("archive", encoding="ascii")
    return dict(
        executable=executable,
        source_root=source,
        source_archive=archive,
        ensemble_path=ensemble,
        catalog_path=catalog,
        dump_root=dump,
        split_path=split,
        output_directory=tmp_path / "out",
    )


def test_legacy_convergence_runner_fails_closed_before_solver_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "lc_pipeline.k3.convergence_benchmark._run_convergence_record",
        lambda **_kwargs: pytest.fail("disabled legacy runner reached solver execution"),
    )
    with pytest.raises(DownstreamBenchmarkError, match="legacy convergence runner is disabled"):
        run_convergence_benchmark(**_inputs(tmp_path), convergence_tolerance=1e-4)


@pytest.mark.parametrize("tolerance", [0.0, -1e-4, 1.0, 2.0, float("nan")])
def test_convergence_runner_rejects_nonconvergence_stop_values(tmp_path, tolerance):
    with pytest.raises(DownstreamBenchmarkError, match="strictly between zero and one"):
        run_convergence_benchmark(**_inputs(tmp_path), convergence_tolerance=tolerance)


def test_convergence_runner_refuses_existing_summary_or_rows(tmp_path):
    values = _inputs(tmp_path)
    values["output_directory"].mkdir()
    (values["output_directory"] / "convergence-summary.json").write_text("{}")
    with pytest.raises(DownstreamBenchmarkError, match="refusing to overwrite"):
        run_convergence_benchmark(**values, convergence_tolerance=1e-4)


def test_legacy_convergence_runner_rejects_subset_execution(tmp_path):
    values = _inputs(tmp_path)
    manifest = tmp_path / "subset.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": OBJECT_SUBSET_SCHEMA,
                "role": "development",
                "source_full_split_sha256": sha256_file(values["split_path"]),
                "object_ids": ["object-3", "object-1"],
                "study_spec_sha256": "a" * 64,
                "selection": "frozen test selection",
                "salt": "frozen-test-salt",
            }
        )
    )

    with pytest.raises(DownstreamBenchmarkError, match="legacy convergence runner is disabled"):
        run_convergence_benchmark(
            **values, convergence_tolerance=1e-4, object_ids_path=manifest
        )


@pytest.mark.parametrize(
    "object_ids,role,split_hash,error",
    [
        ([], "development", "valid", "nonempty"),
        (["object-0", "object-0"], "development", "valid", "duplicate"),
        (["unknown"], "development", "valid", "unknown"),
        (["object-0"], "other", "valid", "role"),
        (["object-0"], "development", "bad", "full split"),
    ],
)
def test_object_subset_manifest_rejects_invalid_cohorts(
    tmp_path, object_ids, role, split_hash, error
):
    values = _inputs(tmp_path)
    manifest = tmp_path / "subset.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": OBJECT_SUBSET_SCHEMA,
                "role": role,
                "source_full_split_sha256": sha256_file(values["split_path"])
                if split_hash == "valid"
                else "0" * 64,
                "object_ids": object_ids,
            }
        )
    )
    with pytest.raises(DownstreamBenchmarkError, match=error):
        _select_object_subset(
            manifest, tuple(f"object-{index}" for index in range(5)), values["split_path"]
        )


def test_convexinv_adapter_retains_partial_timeout_logs(tmp_path, monkeypatch):
    executable = tmp_path / "convexinv"
    executable.write_bytes(b"binary")
    source = tmp_path / "source"
    source.mkdir()
    (source / "main.c").write_text("x")
    lightcurve = tmp_path / "lightcurve.txt"
    lightcurve.write_text("not parsed on timeout")
    parameters = tmp_path / "parameters.txt"
    write_convexinv_parameters(
        parameters, ConvexinvParameters(0, 0, 8, iteration_stop_condition=1e-4)
    )
    monkeypatch.setattr(
        "lc_pipeline.v2.convexinv._run_external",
        lambda *args, **kwargs: (None, True, 3.0, 1.5, b"partial stdout", b"partial stderr"),
    )
    result = run_convexinv(
        executable=executable,
        source_root=source,
        lightcurve_file=lightcurve,
        parameter_file=parameters,
        output_directory=tmp_path / "run",
        timeout_seconds=5,
        stdout_log_path=tmp_path / "run" / "stdout.log",
        stderr_log_path=tmp_path / "run" / "stderr.log",
    )
    assert result.timed_out is True
    assert (tmp_path / "run" / "stdout.log").read_bytes() == b"partial stdout"
    assert (tmp_path / "run" / "stderr.log").read_bytes() == b"partial stderr"


def test_convergence_gate_is_deterministic_and_enforces_one_sided_boundaries():
    shape = (4, 3)
    baseline_wall = np.full(shape, 2.0)
    candidate_wall = np.full(shape, 1.5)
    recovery = np.ones(shape, dtype=int)
    completion = np.ones(shape, dtype=int)
    baseline_rms = np.ones(shape)
    candidate_rms = np.full(shape, 1.005)
    first = evaluate_convergence_gate(
        baseline_wall,
        candidate_wall,
        recovery,
        recovery,
        completion,
        completion,
        baseline_rms,
        candidate_rms,
        seed=19,
    )
    second = evaluate_convergence_gate(
        baseline_wall,
        candidate_wall,
        recovery,
        recovery,
        completion,
        completion,
        baseline_rms,
        candidate_rms,
        seed=19,
    )
    assert first == second
    assert first["passed"] is True
    # Exactly one is not a runtime improvement, and a completion loss below
    # the noninferiority boundary is rejected.
    at_one = evaluate_convergence_gate(
        baseline_wall,
        baseline_wall,
        recovery,
        recovery,
        completion,
        completion,
        baseline_rms,
        candidate_rms,
        seed=19,
    )
    assert "runtime ratio lower confidence bound is not strictly above one" in at_one["failures"]
    no_completion = evaluate_convergence_gate(
        baseline_wall,
        candidate_wall,
        recovery,
        recovery,
        completion,
        np.zeros(shape, dtype=int),
        baseline_rms,
        candidate_rms,
        seed=19,
    )
    assert "guided completion lower confidence bound is below -0.02" in no_completion["failures"]
