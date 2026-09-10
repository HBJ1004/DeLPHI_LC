"""Contract tests for the separate convergence-stopped benchmark."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from lc_pipeline.k3.convergence_benchmark import (
    CONVERGENCE_ITERATION_CAP,
    run_convergence_benchmark,
)
from lc_pipeline.k3.downstream import DownstreamBenchmarkError
from lc_pipeline.v2.convexinv import ConvexinvParameters, run_convexinv, write_convexinv_parameters


def _inputs(tmp_path):
    object_ids = [f"object-{number}" for number in range(5)]
    split = tmp_path / "splits.json"
    split.write_text(json.dumps({"folds": [{"fold": i, "test_ids": [object_ids[i]]} for i in range(5)]}), encoding="utf-8")
    dump = tmp_path / "dump"; records = []
    for object_id in object_ids:
        lightcurve = dump / "files" / object_id / "lc.txt"; lightcurve.parent.mkdir(parents=True)
        lightcurve.write_text("test\n", encoding="ascii")
        records.append({"object_id": object_id, "lightcurve": {"source_path": f"files/{object_id}/lc.txt", "source_sha256": hashlib.sha256(lightcurve.read_bytes()).hexdigest()}, "solutions": [{"period_hours": 8.0, "vector": [1.0, 0.0, 0.0]}]})
    catalog = tmp_path / "catalog.jsonl"; catalog.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    ensemble = tmp_path / "ensemble.npz"
    np.savez_compressed(ensemble, schema=np.asarray("delphi.k3-real-oof-ensemble.v1"), object_ids=np.asarray(object_ids), refined_axes=np.repeat(np.eye(3)[None, :, :], 5, axis=0), inference_wall_seconds=np.full(5, .1), inference_cold_wall_seconds=np.full(5, .4), inference_warm_wall_seconds=np.full(5, .05))
    executable = tmp_path / "convexinv"; executable.write_text("binary", encoding="ascii")
    source = tmp_path / "source"; source.mkdir()
    archive = tmp_path / "source.tar.gz"; archive.write_text("archive", encoding="ascii")
    return dict(executable=executable, source_root=source, source_archive=archive, ensemble_path=ensemble, catalog_path=catalog, dump_root=dump, split_path=split, output_directory=tmp_path / "out")


def test_convergence_runner_propagates_tolerance_selects_by_residual_and_keeps_failures(tmp_path, monkeypatch):
    calls = []
    def fake_record(**kwargs):
        calls.append(kwargs)
        start = kwargs["start_index"]
        # The timeout and cap must still be included in aggregate work.
        completion = "timeout" if start == 4 else ("iteration-cap" if start == 5 else "converged")
        arm = kwargs["arm"]
        return {"identity": {"start_index": start}, "completion": completion, "result": {
            "return_code": None if completion == "timeout" else 0, "timed_out": completion == "timeout",
            "wall_time_seconds": 3.0 if completion == "timeout" else 1.0,
            "process_cpu_time_seconds": .5, "iterations": CONVERGENCE_ITERATION_CAP if completion == "iteration-cap" else 10,
            # Start one is chosen by residual even though its pole fails recovery.
            "relative_rms_from_output": .5 if start == 1 else 1.0,
            "final_lambda_deg": 90.0 if (start == 1 and arm == "candidate" and kwargs["object_id"] == "object-0") else 0.0, "final_beta_deg": 0.0,
        }}
    monkeypatch.setattr("lc_pipeline.k3.convergence_benchmark._run_convergence_record", fake_record)
    result = run_convergence_benchmark(**_inputs(tmp_path), convergence_tolerance=1e-4)
    assert len(calls) == 60
    assert all(call["convergence_tolerance"] == 1e-4 for call in calls)
    # Each object gets a deterministic six-then-six ordering, rather than
    # interleaving starts or letting failures change the arm ordering.
    assert all(
        len({call["arm"] for call in calls[offset : offset + 6]}) == 1
        and len({call["arm"] for call in calls[offset + 6 : offset + 12]}) == 1
        and calls[offset]["arm"] != calls[offset + 6]["arm"]
        for offset in range(0, len(calls), 12)
    )
    payload = json.loads((tmp_path / "out" / "convergence-rows.json").read_text())
    first = payload["rows"][0]
    assert first["candidate"]["wall_seconds"] == 8.05  # six attempts plus warm neural overhead
    assert first["candidate"]["completion_counts"]["timeout"] == 1
    assert first["candidate"]["completion_counts"]["iteration-cap"] == 1
    assert first["candidate"]["best_start_index"] == 1
    assert first["candidate"]["success"] is False  # recovery is evaluated after residual selection
    assert first["neural_inference_cold_wall_seconds"] == .4
    assert result["convergence_tolerance"] == 1e-4


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


def test_convexinv_adapter_retains_partial_timeout_logs(tmp_path, monkeypatch):
    executable = tmp_path / "convexinv"; executable.write_bytes(b"binary")
    source = tmp_path / "source"; source.mkdir(); (source / "main.c").write_text("x")
    lightcurve = tmp_path / "lightcurve.txt"; lightcurve.write_text("not parsed on timeout")
    parameters = tmp_path / "parameters.txt"
    write_convexinv_parameters(parameters, ConvexinvParameters(0, 0, 8, iteration_stop_condition=1e-4))
    monkeypatch.setattr("lc_pipeline.v2.convexinv._run_external", lambda *args, **kwargs: (None, True, 3.0, 1.5, b"partial stdout", b"partial stderr"))
    result = run_convexinv(
        executable=executable, source_root=source, lightcurve_file=lightcurve,
        parameter_file=parameters, output_directory=tmp_path / "run", timeout_seconds=5,
        stdout_log_path=tmp_path / "run" / "stdout.log", stderr_log_path=tmp_path / "run" / "stderr.log",
    )
    assert result.timed_out is True
    assert (tmp_path / "run" / "stdout.log").read_bytes() == b"partial stdout"
    assert (tmp_path / "run" / "stderr.log").read_bytes() == b"partial stderr"
