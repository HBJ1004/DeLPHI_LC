"""Integration contracts for direct pole-search timings, without model weights."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from lc_pipeline import grid_benchmark as grid


def _lightcurve(path: Path) -> Path:
    path.write_text("1\n3 0\n2450000 1 1 0 0 0 1 0\n"
                    "2450000.1 2 1 0 0 0 1 0\n2450000.2 1 1 0 0 0 1 0\n")
    return path


def _fake_native(command, *, stdin, **kwargs):
    del kwargs
    assert stdin.read()
    Path(command[command.index("-o") + 1]).write_text("1 1 1\n")
    Path(command[command.index("-p") + 1]).write_text("10 100 6\n")
    Path(command[-1]).write_text("1 1.8 1\n")
    return subprocess.CompletedProcess(command, 0, stdout=(
        b"10 chi2 0.1 dev 0.01\nlambda, beta and period (hrs): 10 100 6\n"), stderr=b"")


def test_classical_worker_selects_without_neural_imports_or_references(tmp_path, monkeypatch):
    source = _lightcurve(tmp_path / "lc.txt")
    monkeypatch.setattr(grid.subprocess, "run", _fake_native)
    monkeypatch.setattr(grid, "_predictor", lambda *a: pytest.fail("classical arm loaded neural model"))
    job = {"object_id": "asteroid_1", "fold": 0, "period_hours": 6.0,
           "lightcurve": grid._binding(source), "arm": "standard6", "repeat_index": 0,
           "timing_mode": "cold"}
    lock = {"solver": {"executable": {"path": "/fake/convexinv"}}, "settings": grid.SETTINGS}
    result = grid.worker_case(lock, job, tmp_path / "case")
    assert result["completed"] and result["valid_starts"] == 6
    assert result["selected_fit"]["start_index"] == 0
    np.testing.assert_allclose(result["selected_fit"]["axis"], grid._xyz(190, 80), atol=1e-12)
    assert result["inference_and_optional_load_seconds"] == 0
    assert result["prediction_sha256"] is None
    assert result["worker_wall_seconds"] > 0
    assert (tmp_path / "case/selected-result.json").is_file()
    assert len(result["cell_files"]) == 6
    with pytest.raises(FileExistsError):
        grid.worker_case(lock, job, tmp_path / "case")


def test_native_timeout_remains_failure(tmp_path, monkeypatch):
    source = _lightcurve(tmp_path / "lc.txt")
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 0.01, output=b"partial", stderr=b"error")
    monkeypatch.setattr(grid.subprocess, "run", timeout)
    result = grid._native_start({"period_hours": 6, "lightcurve": grid._binding(source)}, 0,
                                np.array([1., 0., 0.]), (np.array([1., 2., 1.]),),
                                {"executable": {"path": "/fake"}}, grid.SETTINGS, tmp_path)
    assert result["timed_out"] and result["selected_fit"] is None
    assert result["failure_reason"] == "timeout_exit_or_iteration_cap"
    assert (tmp_path / "start-000/stdout.txt").read_bytes() == b"partial"
    assert (tmp_path / "start-000/record.json").is_file()


def test_native_iteration_cap_is_not_success(tmp_path, monkeypatch):
    source = _lightcurve(tmp_path / "lc.txt")
    def capped(command, **kwargs):
        value = _fake_native(command, **kwargs)
        value.stdout = value.stdout.replace(b"10 chi2", b"1000 chi2")
        return value
    monkeypatch.setattr(grid.subprocess, "run", capped)
    result = grid._native_start({"period_hours": 6, "lightcurve": grid._binding(source)}, 0,
                                np.array([1., 0., 0.]), (np.array([1., 2., 1.]),),
                                {"executable": {"path": "/fake"}}, grid.SETTINGS, tmp_path)
    assert result["selected_fit"] is None and result["iterations"] == 1000


@pytest.mark.parametrize("replacement", [b"0 chi2 0.1 dev 0.01", b"10 chi2 -1 dev 0.01",
                                         b"10 chi2 1e999 dev 0.01", b"10 chi2 0.1 dev -1"])
def test_invalid_native_diagnostics_fail_closed(tmp_path, monkeypatch, replacement):
    source = _lightcurve(tmp_path / "lc.txt")
    def invalid(command, **kwargs):
        value = _fake_native(command, **kwargs)
        value.stdout = value.stdout.replace(b"10 chi2 0.1 dev 0.01", replacement)
        return value
    monkeypatch.setattr(grid.subprocess, "run", invalid)
    result = grid._native_start({"period_hours": 6, "lightcurve": grid._binding(source)}, 0,
                                np.array([1., 0., 0.]), (np.array([1., 2., 1.]),),
                                {"executable": {"path": "/fake"}}, grid.SETTINGS, tmp_path)
    assert result["selected_fit"] is None


def test_unit_normalization_handles_finite_extreme_scales():
    np.testing.assert_allclose(grid.unit_vectors([[1e300, 0, 0], [0, 1e-300, 0]]),
                               [[1, 0, 0], [0, 1, 0]])


def test_full_run_requires_completed_pilot_before_creating_output(tmp_path, monkeypatch):
    monkeypatch.setattr(grid, "validate_lock", lambda p: {})
    with pytest.raises(grid.GridBenchmarkError, match="completed pilot"):
        grid.execute(lock_path=tmp_path / "lock", output=tmp_path / "full", role="full")
    assert not (tmp_path / "full").exists()


def test_pilot_selection_and_factorial_are_complete():
    objects = [{"object_id": f"asteroid_{i}", "fold": i % 5} for i in range(170)]
    lock = {"objects": objects, "pilot_ids": [f"asteroid_{i}" for i in range(5)]}
    assert len(grid._jobs(lock, "pilot")) == 150
    assert len(grid._jobs(lock, "full")) == 5100
    assert grid._jobs(lock, "pilot") == grid._jobs(lock, "pilot")
    with pytest.raises(grid.GridBenchmarkError):
        grid._jobs(lock, "unknown")


def test_write_once_rejects_overwrite_and_nan(tmp_path):
    path = tmp_path / "value.json"
    grid.write_once(path, {"value": 1})
    with pytest.raises(FileExistsError):
        grid.write_once(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 1}
    with pytest.raises(ValueError):
        grid.write_once(tmp_path / "nan.json", {"value": float("nan")})


def test_artifact_paths_cannot_escape(tmp_path):
    inner = tmp_path / "inner"
    inner.mkdir()
    (tmp_path / "outside").write_text("a")
    with pytest.raises(grid.GridBenchmarkError):
        grid._contained(inner, "../outside")


def test_classical_module_does_not_import_torch():
    import sys
    result = subprocess.run([sys.executable, "-c", "import sys; import lc_pipeline.grid_benchmark; "
                             "import lc_pipeline.v2.convexinv; assert 'torch' not in sys.modules"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
