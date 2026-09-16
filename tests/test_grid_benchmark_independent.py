"""Independent adversarial checks for the known-period grid benchmark.

These tests deliberately exercise the search-start construction and the
execution-artifact boundary without starting a native solver or a neural
worker.  They are kept separate from the benchmark module's own tests so
that simple implementation changes cannot make all checks move together.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import lc_pipeline.grid_benchmark as benchmark


def _contains_axis(vectors: np.ndarray, target: np.ndarray, atol: float = 1e-12) -> bool:
    return bool(np.any(np.linalg.norm(vectors - target[None, :], axis=1) <= atol))


@pytest.mark.parametrize(
    ("spacing", "expected_count"),
    [(15, 266), (20, 146)],
)
def test_pole_grid_has_frozen_cardinality_and_signed_partners(spacing: int, expected_count: int):
    grid = benchmark.pole_grid(spacing)

    assert grid.shape == (expected_count, 3)
    assert np.all(np.isfinite(grid))
    assert np.allclose(np.linalg.norm(grid, axis=1), 1.0, rtol=0.0, atol=1e-14)
    for vector in grid:
        assert _contains_axis(grid, -vector, atol=2e-14)


def test_guided_starts_are_normalized_antipodal_and_permutation_invariant():
    candidates = np.asarray(
        [[2.0, 0.0, 0.0], [0.0, -3.0, 0.0], [0.0, 0.0, 4.0]], dtype=float
    )
    reference = benchmark.guided_starts(candidates, 20, radius=15.0)
    permuted_and_signed = benchmark.guided_starts(
        candidates[[2, 0, 1]] * np.asarray([[-1.0], [1.0], [-1.0]]),
        20,
        radius=15.0,
    )

    assert np.array_equal(reference, permuted_and_signed)
    assert np.allclose(np.linalg.norm(reference, axis=1), 1.0, rtol=0.0, atol=1e-14)
    for vector in reference:
        assert _contains_axis(reference, -vector, atol=2e-14)


def test_guided_starts_remove_duplicates_between_centers_and_grid():
    # Each of these is already a pole-grid direction at spacing 15 degrees.
    candidates = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [np.cos(np.deg2rad(15.0)), np.sin(np.deg2rad(15.0)), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    starts = benchmark.guided_starts(candidates, 15, radius=0.0)

    # Radius zero leaves only the three candidate axes and their signed copies.
    assert starts.shape == (6, 3)
    assert np.unique(np.round(starts, 12), axis=0).shape[0] == 6
    for candidate in candidates / np.linalg.norm(candidates, axis=1)[:, None]:
        assert _contains_axis(starts, candidate)
        assert _contains_axis(starts, -candidate)


def test_guided_starts_exact_cap_boundary_is_included_but_just_inside_is_not():
    candidate = np.eye(3)
    boundary = np.asarray(
        [np.cos(np.deg2rad(15.0)), np.sin(np.deg2rad(15.0)), 0.0], dtype=float
    )

    at_cap = benchmark.guided_starts(candidate, 15, radius=15.0)
    below_cap = benchmark.guided_starts(candidate, 15, radius=14.99)

    assert _contains_axis(at_cap, boundary, atol=2e-12)
    assert _contains_axis(at_cap, -boundary, atol=2e-12)
    assert not _contains_axis(below_cap, boundary, atol=2e-12)
    assert not _contains_axis(below_cap, -boundary, atol=2e-12)


def test_zero_and_ninety_degree_caps_have_expected_extremes():
    candidates = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    zero = benchmark.guided_starts(candidates, 20, radius=0.0)
    ninety = benchmark.guided_starts(candidates, 20, radius=90.0)

    assert zero.shape == (6, 3)
    # Every grid direction is in a 90-degree axial cap.  At spacing 20 the
    # grid has only the two pole directions in common with these candidates;
    # the equatorial x/y directions are not grid points.
    assert ninety.shape[0] == 146 + 6 - 2
    assert np.allclose(np.linalg.norm(ninety, axis=1), 1.0, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize(
    "bad_axes",
    [
        [[0.0, 0.0, 0.0]],
        [[np.nan, 0.0, 0.0]],
        [[np.inf, 1.0, 0.0]],
        [[1.0, 2.0]],
    ],
)
def test_axis_and_start_validation_rejects_zero_nonfinite_and_wrong_shapes(bad_axes):
    with pytest.raises(benchmark.GridBenchmarkError):
        benchmark.unit_vectors(bad_axes)
    with pytest.raises(benchmark.GridBenchmarkError):
        benchmark.guided_starts(bad_axes, 20)


@pytest.mark.parametrize("radius", [-1.0, 90.000001, np.nan, np.inf])
def test_guided_start_validation_rejects_invalid_radius(radius: float):
    with pytest.raises(benchmark.GridBenchmarkError):
        benchmark.guided_starts(np.eye(3), 20, radius=radius)


def test_grid_and_arm_validation_rejects_unknown_settings():
    with pytest.raises(benchmark.GridBenchmarkError):
        benchmark.pole_grid(10)
    with pytest.raises(benchmark.GridBenchmarkError):
        benchmark.starts_for_arm("unknown")
    with pytest.raises(benchmark.GridBenchmarkError):
        benchmark.starts_for_arm("guided20", np.eye(2))


def _minimal_execution_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    """Create one valid case for validate_execution with its factorial checks patched."""
    lock_path = tmp_path / "lock.json"
    lock_path.write_text("{}\n", encoding="utf-8")
    case = tmp_path / "cases" / "one"
    start = case / "start-000"
    start.mkdir(parents=True)
    product = start / "product.txt"
    product.write_text("native-product\n", encoding="ascii")
    selected_fit = {
        "axis": [1.0, 0.0, 0.0],
        "final_relative_rms": 0.5,
        "start_index": 0,
    }
    cell = {
        "start_index": 0,
        "initial_axis": [1.0, 0.0, 0.0],
        "files": {"product.txt": hashlib.sha256(product.read_bytes()).hexdigest()},
        "selected_fit": selected_fit,
    }
    cell_path = start / "record.json"
    cell_path.write_text(json.dumps(cell, sort_keys=True), encoding="utf-8")
    row = {
        "object_id": "asteroid_1",
        "fold": 0,
        "repeat_index": 0,
        "arm": "classical20",
        "timing_mode": "cold",
        "wall_seconds": 1.0,
        "completed": True,
        "selected_fit": selected_fit,
        "starts_requested": 1,
        "valid_starts": 1,
        "prediction_sha256": None,
        "cell_files": {"start-000/record.json": hashlib.sha256(cell_path.read_bytes()).hexdigest()},
    }
    selected_path = case / "selected-result.json"
    selected_path.write_text(json.dumps(row, sort_keys=True), encoding="utf-8")
    timed_path = case / "timed-result.json"
    timed_path.write_text(json.dumps(row, sort_keys=True), encoding="utf-8")
    document = {
        "schema": benchmark.SCHEMA,
        "lock_sha256": benchmark.digest(lock_path),
        "phase": "blind_execution_and_fit_selection_complete",
        "reference_records_opened": False,
        "execution_complete": True,
        "execution_elapsed_seconds": 1.1,
        "worker_setup": [{"fold": 0, "timing_mode": "cold", "ready": True,
                          "persistent_worker": False, "fresh_process_per_case": True,
                          "setup_seconds": 0.0, "setup_in_cell_timing": True}],
        "warm_worker_setup": [],
        "budget": {},
        "role": "pilot",
        "rows": [
            {
                "path": "cases/one/timed-result.json",
                "sha256": benchmark.digest(timed_path),
            }
        ],
    }
    execution_path = tmp_path / "execution.json"
    execution_path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    job = {"object_id": "asteroid_1", "fold": 0, "repeat_index": 0,
           "arm": "classical20", "timing_mode": "cold"}
    return lock_path, execution_path, job


def _patch_minimal_execution(monkeypatch: pytest.MonkeyPatch, job: dict) -> None:
    monkeypatch.setattr(benchmark, "validate_lock", lambda _path: {"fixture": True})
    monkeypatch.setattr(benchmark, "_jobs", lambda _lock, _role: [job])
    monkeypatch.setattr(benchmark, "_pilot_budget", lambda *args: {})
    monkeypatch.setattr(
        benchmark,
        "starts_for_arm",
        lambda _arm, _axes=None: np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
    )


def test_validate_execution_accepts_valid_case_and_checks_all_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    lock_path, execution_path, job = _minimal_execution_fixture(tmp_path)
    _patch_minimal_execution(monkeypatch, job)

    result = benchmark.validate_execution(execution_path, lock_path)

    assert len(result["validated_rows"]) == 1
    assert result["validated_rows"][0]["selected_fit"]["start_index"] == 0


@pytest.mark.parametrize(("field", "value", "message"), [
    ("execution_complete", False, "identity/phase"),
    ("execution_elapsed_seconds", 0.5, "shorter than serial"),
    ("execution_elapsed_seconds", float("inf"), "invalid execution elapsed"),
    ("worker_setup", [], "worker setup"),
    ("warm_worker_setup", [{"ready": True}], "warm setup"),
    ("budget", {"within_full_run_budget": True}, "recomputed timing"),
])
def test_execution_metadata_cannot_bypass_completion_or_budget(
    tmp_path, monkeypatch, field, value, message
):
    lock_path, execution_path, job = _minimal_execution_fixture(tmp_path)
    _patch_minimal_execution(monkeypatch, job)
    document = json.loads(execution_path.read_text())
    document[field] = value
    execution_path.write_text(json.dumps(document))
    with pytest.raises(benchmark.GridBenchmarkError, match=message):
        benchmark.validate_execution(execution_path, lock_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("phase", "reference_scoring_after_sealed_selection", "identity/phase mismatch"),
        ("lock_sha256", "0" * 64, "identity/phase mismatch"),
        ("reference_records_opened", True, "identity/phase mismatch"),
    ],
)
def test_validate_execution_rejects_wrong_phase_or_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object, message: str
):
    lock_path, execution_path, job = _minimal_execution_fixture(tmp_path)
    _patch_minimal_execution(monkeypatch, job)
    document = json.loads(execution_path.read_text(encoding="utf-8"))
    document[field] = value
    execution_path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")

    with pytest.raises(benchmark.GridBenchmarkError, match=message):
        benchmark.validate_execution(execution_path, lock_path)


def test_validate_execution_rejects_tampered_case_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    lock_path, execution_path, job = _minimal_execution_fixture(tmp_path)
    _patch_minimal_execution(monkeypatch, job)
    timed = tmp_path / "cases" / "one" / "timed-result.json"
    timed.write_text(timed.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")

    with pytest.raises(benchmark.GridBenchmarkError, match="timed record checksum changed"):
        benchmark.validate_execution(execution_path, lock_path)


def test_validate_execution_rejects_case_path_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    lock_path, execution_path, job = _minimal_execution_fixture(tmp_path)
    _patch_minimal_execution(monkeypatch, job)
    outside = tmp_path.parent / f"outside-{tmp_path.name}.json"
    outside.write_text("{}\n", encoding="utf-8")
    document = json.loads(execution_path.read_text(encoding="utf-8"))
    document["rows"][0] = {
        "path": f"../{outside.name}",
        "sha256": benchmark.digest(outside),
    }
    execution_path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")

    try:
        with pytest.raises(benchmark.GridBenchmarkError, match="escapes execution root"):
            benchmark.validate_execution(execution_path, lock_path)
    finally:
        outside.unlink()
