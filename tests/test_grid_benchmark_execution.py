"""Bounded scheduling and lifecycle contracts for the pole-grid benchmark."""

from __future__ import annotations

import io
import json
import signal
from pathlib import Path

import pytest

from lc_pipeline import grid_benchmark as grid


def _lock() -> dict:
    objects = [
        {
            "object_id": f"asteroid_{index}",
            "fold": index % 5,
            "structure": {"total_observations": index + 10},
        }
        for index in range(170)
    ]
    return {
        "objects": objects,
        "pilot_ids": [f"asteroid_{index}" for index in range(5)],
        "code_root": str(Path.cwd()),
    }


def _pilot_rows() -> list[dict]:
    return [
        {
            "object_id": f"asteroid_{index}",
            "repeat_index": repeat,
            "arm": arm,
            "timing_mode": mode,
            "wall_seconds": 1.0,
        }
        for index in range(5)
        for repeat in range(3)
        for arm in grid.ARMS
        for mode in grid.MODES
    ]


def _job(mode: str, arm: str) -> dict:
    return {
        "object_id": "asteroid_0",
        "fold": 0,
        "repeat_index": 0,
        "arm": arm,
        "timing_mode": mode,
    }


def _record(job: dict, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    return {
        "object_id": job["object_id"],
        "fold": job["fold"],
        "repeat_index": job["repeat_index"],
        "arm": job["arm"],
        "timing_mode": job["timing_mode"],
        "completed": False,
        "selected_fit": None,
        "prediction_sha256": None,
        "starts_requested": 1,
        "valid_starts": 0,
    }


class _Input:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def write(self, value: str) -> None:
        self.lines.append(value)

    def flush(self) -> None:
        pass


class _Process:
    events: list[tuple[str, bool]] = []
    persistent_alive = False

    def __init__(self, command: list[str], **kwargs: object) -> None:
        self.command = command
        self.persistent = "--fold" in command
        self.returncode = None
        self.stdin = _Input()
        self.stdout = io.StringIO()
        self.pid = 999999
        assert kwargs["start_new_session"] is True
        if self.persistent:
            assert not self.persistent_alive
            type(self).persistent_alive = True
        else:
            # This is the central cold-isolation contract.
            assert not self.persistent_alive
        self.events.append(("spawn", self.persistent))

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        del timeout
        job_path = Path(self.command[self.command.index("--job") + 1])
        job = json.loads(job_path.read_text())
        result = _record(job, Path(job["directory"]))
        self.returncode = 0
        return json.dumps(result), ""

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.returncode = 0
        if self.persistent:
            type(self).persistent_alive = False
        return 0


def test_jobs_are_deterministic_randomized_fold_mode_blocks() -> None:
    lock = _lock()
    first = grid._jobs(lock, "pilot")
    second = grid._jobs(lock, "pilot")

    assert first == second
    assert len(first) == 150
    block_runs = []
    for job in first:
        block = (job["fold"], job["timing_mode"])
        if not block_runs or block_runs[-1] != block:
            block_runs.append(block)
    assert len(block_runs) == 10
    assert set(block_runs) == {
        (fold, mode) for fold in range(5) for mode in grid.MODES
    }
    assert block_runs != [
        (fold, mode) for fold in range(5) for mode in grid.MODES
    ]
    for fold, mode in block_runs:
        block = [
            job
            for job in first
            if (job["fold"], job["timing_mode"]) == (fold, mode)
        ]
        assert [job["arm"] for job in block] != list(grid.ARMS) * 3


def test_pilot_budget_uses_validated_cells_and_observed_elapsed() -> None:
    lock = _lock()
    rows = _pilot_rows()

    result = grid._pilot_budget(lock, rows, 200.0)

    expected_observation_scale = sum(
        row["structure"]["total_observations"] for row in lock["objects"]
    ) / sum(
        row["structure"]["total_observations"]
        for row in lock["objects"]
        if row["object_id"] in lock["pilot_ids"]
    )
    assert result["observed_pilot_cell_wall_seconds"] == 150.0
    assert result["observed_pilot_elapsed_seconds"] == 200.0
    assert result["projected_full_elapsed_seconds"] == pytest.approx(
        2 * 200 * expected_observation_scale
    )
    assert result["method"] == (
        "2x_pilot_elapsed_x_max_object_count_or_observation_count_scale"
    )
    with pytest.raises(grid.GridBenchmarkError, match="complete validated 150-cell"):
        grid._pilot_budget(lock, rows[:-1], 200.0)


def test_execute_never_has_warm_resident_during_cold_and_recomputes_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = _lock()
    warm_zero = _job("warm", "classical20")
    cold_zero = _job("cold", "guided20")
    warm_one = {**_job("warm", "guided20"), "object_id": "asteroid_1", "fold": 1}
    cold_one = {**_job("cold", "classical20"), "object_id": "asteroid_1", "fold": 1}
    jobs = [warm_zero, cold_zero, warm_one, cold_one]
    validated_rows = [{"sentinel": "validated-not-stored-budget"}]
    calls = []
    _Process.events = []
    _Process.persistent_alive = False
    (tmp_path / "lock.json").write_text("{}")
    (tmp_path / "pilot.json").write_text("{}")
    monkeypatch.setattr(grid, "validate_lock", lambda _path: lock)
    monkeypatch.setattr(
        grid,
        "validate_execution",
        lambda _path, _lock_path: {
            "role": "pilot",
            "validated_rows": validated_rows,
            "execution_elapsed_seconds": 12.0,
            "budget": {"within_full_run_budget": False},
        },
    )

    def budget(_lock: dict, rows: list[dict], elapsed: float) -> dict:
        calls.append((rows, elapsed))
        return {"within_full_run_budget": True, "recomputed": True}

    monkeypatch.setattr(grid, "_pilot_budget", budget)
    monkeypatch.setattr(grid, "_jobs", lambda _lock, _role: [dict(job) for job in jobs])
    monkeypatch.setattr(grid.subprocess, "Popen", _Process)

    def readline(process: _Process, _deadline: float | None, *, description: str) -> str:
        if description == "persistent setup":
            return json.dumps({"ready": True, "setup_seconds": 0.25}) + "\n"
        job = json.loads(process.stdin.lines[-1])
        return json.dumps(_record(job, Path(job["directory"]))) + "\n"

    monkeypatch.setattr(grid, "_readline_before", readline)
    result = grid.execute(
        lock_path=tmp_path / "lock.json",
        output=tmp_path / "run",
        role="full",
        pilot_path=tmp_path / "pilot.json",
    )

    assert calls == [(validated_rows, 12.0)]
    assert _Process.events == [
        ("spawn", True),
        ("spawn", False),
        ("spawn", True),
        ("spawn", False),
    ]
    assert result["execution_complete"] is True
    assert result["pilot_budget_authorization"]["recomputed"] is True
    assert result["pilot_execution"]["path"] == str((tmp_path / "pilot.json").resolve())
    assert json.loads((tmp_path / "run/execution-start.json").read_text())[
        "pilot_execution"
    ] == result["pilot_execution"]
    assert [(row["timing_mode"], row["fold"]) for row in result["worker_setup"]] == [
        ("warm", 0),
        ("cold", 0),
        ("warm", 1),
        ("cold", 1),
    ]
    assert result["worker_setup"][0]["persistent_worker"] is True
    assert result["worker_setup"][1]["fresh_process_per_case"] is True


def test_interruption_retains_diagnostic_but_never_complete_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = _lock()
    (tmp_path / "lock.json").write_text("{}")
    (tmp_path / "pilot.json").write_text("{}")
    monkeypatch.setattr(grid, "validate_lock", lambda _path: lock)
    monkeypatch.setattr(grid, "_jobs", lambda _lock, _role: [_job("cold", "classical20")])
    monkeypatch.setattr(
        grid,
        "_remaining_seconds",
        lambda _deadline, _maximum: (_ for _ in ()).throw(
            grid.GridBenchmarkError("configured elapsed budget reached; no complete result")
        ),
    )
    monkeypatch.setattr(
        grid,
        "validate_execution",
        lambda _path, _lock_path: {
            "role": "pilot",
            "validated_rows": [],
            "execution_elapsed_seconds": 1.0,
        },
    )
    monkeypatch.setattr(
        grid,
        "_pilot_budget",
        lambda *_args: {"within_full_run_budget": True},
    )

    output = tmp_path / "run"
    with pytest.raises(grid.GridBenchmarkError, match="configured elapsed budget"):
        grid.execute(
            lock_path=tmp_path / "lock.json",
            output=output,
            role="full",
            pilot_path=tmp_path / "pilot.json",
        )

    diagnostic = json.loads((output / "execution-interrupted.json").read_text())
    assert diagnostic["execution_complete"] is False
    assert diagnostic["phase"] == "execution_interrupted_no_scoreable_execution"
    assert diagnostic["completed_case_count"] == 0
    assert not (output / "execution.json").exists()


def test_termination_targets_known_group_after_leader_has_exited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class ExitedLeader:
        pid = 31415

        @staticmethod
        def poll() -> int:
            return 1

    def killpg(group: int, requested_signal: int) -> None:
        calls.append((group, requested_signal))
        if requested_signal == 0:
            raise ProcessLookupError

    monkeypatch.setattr(grid.os, "killpg", killpg)
    grid._terminate_process_group(ExitedLeader())

    assert calls[0] == (31415, signal.SIGTERM)
    assert all(group == 31415 for group, _ in calls)


def test_lock_audit_can_skip_environment_but_execution_cannot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "lock.json"
    path.write_text("{}\n", encoding="utf-8")
    solver = {"capacity_report": {"path": str(tmp_path / "capacity.json")}}
    lock = {
        "schema": grid.SCHEMA,
        "settings": grid.SETTINGS,
        "environment": {"machine": "frozen"},
        "device": "cpu",
        "code_root": str(tmp_path),
        "runtime_sources": {},
        "inputs": {},
        "objects": [],
        "bundles": [],
        "solver": solver,
    }
    monkeypatch.setattr(grid, "read_json", lambda _path: lock)
    monkeypatch.setattr(grid, "_runtime_sources", lambda _root: {})
    monkeypatch.setattr(grid, "_validated_capacity", lambda _path, _rows: solver)
    monkeypatch.setattr(grid, "_environment", lambda _device: {"machine": "current"})

    with pytest.raises(grid.GridBenchmarkError, match="timing environment changed"):
        grid.validate_lock(path)
    assert grid.validate_lock(path, require_environment=False) is lock


def test_continuation_scores_only_after_full_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = tmp_path / "lock.json"
    pilot_path = tmp_path / "pilot" / "execution.json"
    pilot_path.parent.mkdir()
    lock_path.write_text("{}\n", encoding="utf-8")
    pilot_path.write_text("{}\n", encoding="utf-8")
    events: list[str] = []
    pilot = {
        "role": "pilot",
        "validated_rows": _pilot_rows(),
        "execution_elapsed_seconds": 200.0,
    }
    monkeypatch.setattr(grid, "validate_lock", lambda _path: _lock())
    monkeypatch.setattr(grid, "validate_execution", lambda _path, _lock_path: pilot)
    monkeypatch.setattr(
        grid,
        "_pilot_budget",
        lambda *_args: {"within_full_run_budget": True},
    )

    def execute(**kwargs: object) -> dict:
        events.append("execute-full")
        output = Path(kwargs["output"])
        output.mkdir()
        (output / "execution.json").write_text("{}\n", encoding="utf-8")
        return {"rows": [1, 2, 3]}

    def score(**kwargs: object) -> dict:
        events.append(f"score-{Path(kwargs['execution_path']).parent.name}")
        Path(kwargs["output"]).write_text("{}\n", encoding="utf-8")
        return {}

    monkeypatch.setattr(grid, "execute", execute)
    monkeypatch.setattr(grid, "score", score)

    result = grid.continue_after_pilot(
        lock_path=lock_path, pilot_path=pilot_path, root=tmp_path
    )

    assert events == ["execute-full", "score-full", "score-pilot"]
    assert result["phase"] == "full_execution_and_post_selection_scoring_complete"
    assert result["full_case_count"] == 3
