"""Refinement checks; all generated inputs and outputs live in tmp_path."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from lc_pipeline.v2.convexinv import PeriodScanRow
from repro.own_period import period_cost as cost
from repro.own_period import refine_chain as refine


@pytest.fixture
def inputs(tmp_path):
    lightcurve = tmp_path / "lightcurve.txt"
    lightcurve.write_text("1\n2 0\n2450000 1 1 0 0 1 0 0\n2451000 1 1 0 0 1 0 0\n")
    rows = [{"object_id": oid, "fold": index // 4,
             "lightcurve": {"path": str(lightcurve), "sha256": cost.sha256_file(lightcurve)},
             # This field must never supply the refinement's period.
             "period_hours": -999}
            for index, oid in enumerate(refine.PROTOCOL_OBJECTS)]
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"objects": rows}))
    periods = tmp_path / "periods.json"
    periods.write_text(json.dumps({"periods_hours": {
        oid: 5 + index for index, oid in enumerate(refine.PROTOCOL_OBJECTS)}}))
    costs = tmp_path / "costs.json"
    costs.write_text(json.dumps({"estimates": {
        refine.PROTOCOL_OBJECTS[0]: {"E_core": {"cpu_seconds": {"mean_per_trial": 2}}},
        refine.PROTOCOL_OBJECTS[1]: {"E_core": {"cpu_seconds": {"mean_per_trial": 6}}},
        refine.PROTOCOL_OBJECTS[2]: {"E_core": {"cpu_seconds": {"mean_per_trial_at_least": 99}}},
    }}))
    source = tmp_path / "solver"
    source.mkdir()
    binary = source / "period_scan"
    binary.write_text("mock binary, never executed\n")
    binary.chmod(0o700)
    return SimpleNamespace(evaluation_lock=lock, periods_file=periods,
        cost_summary=costs, solver_source=source, output_dir=tmp_path / "output",
        cpus=(min(os.sched_getaffinity(0)),), workers=1, chunk_trials=150, plan_only=False)


def fake_record(task, *, failed=False):
    directory = Path(task["raw_directory"])
    directory.mkdir(parents=True, exist_ok=False)
    grid = cost.TrialGrid(task["jd_min"], task["jd_max"],
                          task["start_frequency_per_hour"], task["end_frequency_per_hour"])
    parameters = directory / "input-period-scan"
    cost._write_parameters(parameters, grid, task["start_index"], task["expected_rows"])
    raw = directory / "period-scan.txt"
    raw.write_text("".join(f"{10.123456789 if i == 0 else 12} 0.1 {1 if i == 0 else 2} 50 0.1\n"
                           for i in range(task["start_index"], task["stop_index_exclusive"])))
    rows = cost.parse_period_scan_rows(raw)
    record = {**task, "started_utc": "2026-10-26T01:00:00+00:00",
              "finished_utc": "2026-10-26T01:00:10+00:00",
              "status": "not_finished" if failed else "complete",
              "return_code": None if failed else 0, "timed_out": failed,
              "rows_returned": len(rows), "cpu_seconds": 8, "wall_seconds": 10,
              "fitted_rows": cost.fitted_rows(rows, grid, task["start_index"],
                                               task["period_bounds_hours"][0]),
              "output_periods_sha256": cost.sha256_file(raw),
              "provenance": {"parameter_sha256": cost.sha256_file(parameters),
                             "input_sha256": task["lightcurve_sha256"],
                             "executable_sha256": task["executable_sha256"]}}
    cost._write_json(directory / "record.json", record)
    return record


def fake_runner(tasks, cpus, workers):
    return [fake_record(task) for task in tasks]


@pytest.mark.parametrize("period,span", [(2.1, 1155), (10, 12619.984927), (99, 40000), (100, 1)])
def test_window_grid_bounds_counts_and_chunk_coverage(tmp_path, period, span):
    f_own = 1 / period
    grid = cost.TrialGrid(2450000, 2450000 + span, f_own * 1.01, f_own * 0.99)
    expected_step = 0.5 / (grid.jd_max - grid.jd_min) / 24 * 0.8
    assert grid.step == expected_step
    assert grid.count == int((f_own * 1.01 - f_own * 0.99) / expected_step) + 1
    assert grid.frequency(0) == f_own * 1.01
    assert grid.frequency(grid.count - 1) >= f_own * 0.99 - 1e-15
    assert grid.frequency(grid.count - 1) - grid.step < f_own * 0.99
    covered = []
    for block, (start, count) in enumerate(cost.chunk_ranges(grid.count, math.ceil(grid.count / 150))):
        covered.extend(range(start, start + count))
        assert count <= 150
        bounds = cost._write_parameters(tmp_path / f"parameters-{block}", grid, start, count)
        assert cost.integer_trial_count(*bounds, grid.step) == count
        fs = 1 / bounds[0]
        for local in range(count):
            assert fs - local * grid.step == pytest.approx(grid.frequency(start + local), abs=2e-16)
        assert (tmp_path / f"parameters-{block}").read_text().splitlines()[1:] == (
            cost.PeriodScanParameters(2, 100).render().splitlines()[1:])
    assert covered == list(range(grid.count))


@pytest.mark.parametrize("high,low", [(0, 0), (1, -1), (0.1, 0.2), (math.nan, 0.1), (1, math.inf)])
def test_invalid_custom_frequencies(high, low):
    with pytest.raises(ValueError, match="frequencies"):
        cost.TrialGrid(0, 100, high, low)


def test_protocol_selection_preserves_order_within_each_fold(inputs):
    lock = json.loads(inputs.evaluation_lock.read_text())
    lock["objects"].extend({"object_id": f"extra_{fold}", "fold": fold} for fold in range(5))
    assert tuple(r["object_id"] for r in refine.protocol_objects(lock)) == refine.PROTOCOL_OBJECTS
    lock["objects"][0], lock["objects"][1] = lock["objects"][1], lock["objects"][0]
    with pytest.raises(ValueError, match="protocol list"):
        refine.protocol_objects(lock)


def test_plan_cost_fallback_and_all_objects_queued_longest_first(inputs):
    inputs.chunk_trials = 50
    plan = refine.build_refine_plan(inputs)
    assert [r["object_id"] for r in plan["objects"]] == list(refine.PROTOCOL_OBJECTS)
    assert plan["median_cpu_seconds_per_trial"] == 4  # excludes the lower bound 99
    for index, row in enumerate(plan["objects"]):
        assert row["own_period_hours"] == 5 + index
        mean = (2, 6)[index] if index < 2 else 4
        assert row["expected_cpu_seconds_per_trial"] == mean
        assert row["expected_cpu_hours"] == mean * row["trial_count"] / 3600
        assert row["cost_source"] == ("object_mean" if index < 2 else "median_of_measured_means")
        chunks = sorted((task for task in plan["tasks"] if task["object_id"] == row["object_id"]),
                        key=lambda task: task["block"])
        covered = [i for chunk in chunks
                   for i in range(chunk["start_index"], chunk["stop_index_exclusive"])]
        assert covered == list(range(row["trial_count"]))
        assert all(chunk["expected_rows"] <= 50 for chunk in chunks)
        assert all(chunk["timeout_seconds"] == 12 * 3600 for chunk in chunks)
        assert Path(chunks[0]["raw_directory"]).name == "chunk-000"
    costs = [task["expected_cpu_seconds"] for task in plan["tasks"]]
    assert costs == sorted(costs, reverse=True)
    assert plan["total_trials"] == sum(r["trial_count"] for r in plan["objects"])
    assert plan["expected_cpu_hours"] == sum(r["expected_cpu_hours"] for r in plan["objects"])


def test_plan_only_writes_no_runs_or_fit_results(inputs, monkeypatch):
    inputs.plan_only = True
    monkeypatch.setattr(cost, "run_tasks", lambda *args: pytest.fail("plan-only launched workers"))
    assert refine.command_refine(inputs) == 0
    assert [p.name for p in inputs.output_dir.iterdir()] == ["refine-plan.json"]
    plan = inputs.output_dir / "refine-plan.json"
    original = plan.read_bytes()
    assert refine.command_refine(inputs) == 0
    assert plan.read_bytes() == original
    inputs.chunk_trials += 1
    with pytest.raises(ValueError, match="mismatches"):
        refine.command_refine(inputs)
    assert plan.read_bytes() == original


def test_no_measured_costs_or_changed_lightcurve_stops_planning(inputs):
    value = json.loads(inputs.cost_summary.read_text())
    inputs.cost_summary.write_text(json.dumps({"estimates": {}}))
    with pytest.raises(ValueError, match="measured E-core"):
        refine.build_refine_plan(inputs)
    inputs.cost_summary.write_text(json.dumps(value))
    Path(json.loads(inputs.evaluation_lock.read_text())["objects"][0]["lightcurve"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="lightcurve disagrees"):
        refine.build_refine_plan(inputs)
    assert not inputs.output_dir.exists()


def test_resume_recovers_unlogged_complete_chunks_and_never_overwrites(inputs, monkeypatch):
    calls = []

    def interrupted(tasks, cpus, workers):
        assert (inputs.output_dir / "refine-plan.json").is_file()
        calls.append(tasks[0])
        fake_record(tasks[0])
        raise KeyboardInterrupt

    monkeypatch.setattr(cost, "run_tasks", interrupted)
    with pytest.raises(KeyboardInterrupt):
        refine.command_refine(inputs)
    completed = Path(calls[0]["raw_directory"])
    original = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in completed.iterdir()}

    def resumed(tasks, cpus, workers):
        assert cpus == inputs.cpus and workers == inputs.workers
        assert calls[0] not in tasks
        assert {task["object_id"] for task in tasks} | {calls[0]["object_id"]} == set(refine.PROTOCOL_OBJECTS)
        return fake_runner(tasks, cpus, workers)

    monkeypatch.setattr(cost, "run_tasks", resumed)
    assert refine.command_refine(inputs) == 0
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == before for p, before in original.items())
    journal = cost._read_jsonl(inputs.output_dir / "refine.jsonl")
    plan = json.loads((inputs.output_dir / "refine-plan.json").read_text())
    assert len(journal) == len(plan["tasks"])
    assert all("fitted_rows" not in record for record in journal)
    final = json.loads((inputs.output_dir / "refined-periods.json").read_text())
    assert len(final["periods_hours"]) == 20
    for row in final["objects"]:
        assert row["status"] == "complete"
        assert row["period_found_hours_raw"] == 10.123456789
        assert row["period_found_hours"] == 10.123457
        assert row["cpu_seconds"] > 0 and row["wall_seconds"] == 10
    originals = {p: (p.read_bytes(), p.stat().st_mtime_ns)
                 for p in inputs.output_dir.rglob("*") if p.is_file()}
    monkeypatch.setattr(cost, "run_tasks", lambda *args: pytest.fail("reran a saved chunk"))
    assert refine.command_refine(inputs) == 0
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == before for p, before in originals.items())


@pytest.mark.parametrize("change", ["task", "output", "parameters", "fitted_rows", "journal", "binary"])
def test_mismatching_saved_artifacts_stop_resume(inputs, monkeypatch, change):
    monkeypatch.setattr(cost, "run_tasks", fake_runner)
    assert refine.command_refine(inputs) == 0
    plan = json.loads((inputs.output_dir / "refine-plan.json").read_text())
    directory = Path(plan["tasks"][0]["raw_directory"])
    record_path = directory / "record.json"
    if change in ("task", "fitted_rows"):
        record = json.loads(record_path.read_text())
        if change == "task":
            record["start_index"] += 1
        else:
            record["fitted_rows"][0]["chi2"] += 1
        record_path.write_text(json.dumps(record))
    elif change == "journal":
        with (inputs.output_dir / "refine.jsonl").open("a") as stream:
            cost._dump(stream, cost._read_jsonl(inputs.output_dir / "refine.jsonl")[0])
    elif change == "binary":
        (inputs.solver_source / "period_scan").write_text("changed binary")
    else:
        path = directory / ("period-scan.txt" if change == "output" else "input-period-scan")
        path.write_text(path.read_text() + "\n")
    monkeypatch.setattr(cost, "run_tasks", lambda *args: pytest.fail("ran after mismatch"))
    with pytest.raises(ValueError):
        refine.command_refine(inputs)


def test_unrecorded_directory_is_not_overwritten(inputs, monkeypatch):
    inputs.plan_only = True
    assert refine.command_refine(inputs) == 0
    plan = json.loads((inputs.output_dir / "refine-plan.json").read_text())
    directory = Path(plan["tasks"][0]["raw_directory"])
    directory.mkdir(parents=True)
    sentinel = directory / "unfinished.txt"
    sentinel.write_text("keep")
    inputs.plan_only = False
    monkeypatch.setattr(cost, "run_tasks", lambda *args: pytest.fail("ran after incomplete directory"))
    with pytest.raises(FileExistsError, match="without record"):
        refine.command_refine(inputs)
    assert sentinel.read_text() == "keep"


def test_failed_chunk_is_reported_without_selecting_partial_period_or_rerunning(inputs, monkeypatch):
    def runner(tasks, cpus, workers):
        return [fake_record(task, failed=task["object_id"] == refine.PROTOCOL_OBJECTS[0]) for task in tasks]

    monkeypatch.setattr(cost, "run_tasks", runner)
    assert refine.command_refine(inputs) == 1
    final = json.loads((inputs.output_dir / "refined-periods.json").read_text())
    failed = final["objects"][0]
    assert failed["status"] == "not_finished"
    assert failed["rows_returned"] > 0 and failed["cpu_seconds"] > 0
    assert failed["period_found_hours"] is None
    assert failed["ambiguity_count"] is None
    assert failed["object_id"] not in final["periods_hours"]
    monkeypatch.setattr(cost, "run_tasks", lambda *args: pytest.fail("reran failed scan"))
    assert refine.command_refine(inputs) == 1


def row(period, chi2):
    return PeriodScanRow(period, 0.1, chi2, 50, 1)


def test_reference_free_minima_match_cost_logic_with_plateaus_and_endpoints():
    rows = [row(4, 10.5), row(5, 20), row(10, 10), row(10.05, 12), row(10.06, 10.2),
            row(11, 20), row(20, 11), row(21, 11), row(22, 20), row(30, 11.01)]
    score = refine.score_periods(rows)
    previous = cost.score_periods(rows, 123)
    for key in ("chi2", "best_trial_index", "ambiguity_count", "alternative_minima", "minima_definition"):
        assert score[key] == previous[key]
    assert score["period_found_hours_raw"] == 10
    assert score["ambiguity_count"] == 2
    assert [r["trial_index"] for r in score["alternative_minima"]] == [0, 6]
    assert not any("reference" in key or "class" in key for key in score)


@pytest.mark.parametrize("rows", [[], [row(-1, 1)], [row(math.nan, 1)], [row(1, -1)], [row(1, math.inf)]])
def test_reference_free_minima_reject_invalid_rows(rows):
    with pytest.raises(ValueError):
        refine.score_periods(rows)


def test_real_two_trial_refinement_smoke(tmp_path, require_delphi_files):
    binary = cost.SOURCE / "period_scan"
    require_delphi_files(binary)
    assert os.access(binary, os.X_OK)
    lightcurve = tmp_path / "tiny-lightcurve.txt"
    lines = ["2"]
    for epoch in range(2):
        lines.append("80 0")
        for index in range(80):
            jd = 2450000 + epoch * 1155 + index / 800
            angle = epoch * 0.7 + index / 700
            brightness = 1 + 0.12 * math.cos(2 * math.pi * index / 23)
            lines.append(f"{jd:.12f} {brightness:.12f} "
                         f"{math.cos(angle):.12f} {math.sin(angle):.12f} 0.2 "
                         f"{math.cos(angle + 0.2):.12f} {math.sin(angle + 0.2):.12f} 0.1")
    lightcurve.write_text("\n".join(lines) + "\n")
    span = cost.lightcurve_span(lightcurve)
    own_frequency = 1 / 10
    info = {"object_id": "smoke", **span, "lightcurve_file": str(lightcurve),
            "lightcurve_sha256": cost.sha256_file(lightcurve),
            "start_frequency_per_hour": own_frequency * 1.01,
            "end_frequency_per_hour": own_frequency * 0.99}
    grid = cost.TrialGrid(span["jd_min"], span["jd_max"],
                          info["start_frequency_per_hour"], info["end_frequency_per_hour"])
    task = cost._task(info, 0, 2, tmp_path / "refine-runs/smoke/chunk-000", cost.SOURCE,
                      0, refine.TIMEOUT_SECONDS, "refine")
    task["executable_sha256"] = cost.sha256_file(binary)
    cpu = min(os.sched_getaffinity(0))
    before_affinity = os.sched_getaffinity(0)
    before_nice = os.getpriority(os.PRIO_PROCESS, 0)
    records = cost.run_tasks([task], (cpu,), 1)
    assert len(records) == 1
    record = records[0]
    assert record["status"] == "complete", record
    assert record["rows_returned"] == 2
    assert record["cpu_seconds"] > 0 and record["wall_seconds"] > 0
    assert record["cpu_list"] == [cpu] and record["nice"] == 19
    refine._validate_record(task, record)
    assert cost.integer_trial_count(*record["period_bounds_hours"], grid.step) == 2
    for i, fitted in enumerate(record["fitted_rows"]):
        assert fitted["start_frequency_per_hour"] == pytest.approx(grid.frequency(i), abs=2e-16)
    saved, pending = refine._resume_records([task])
    assert saved == json.loads(json.dumps(records)) and pending == []
    assert os.sched_getaffinity(0) == before_affinity
    assert os.getpriority(os.PRIO_PROCESS, 0) == before_nice


def test_only_refine_subcommand_is_available():
    for subcommand in ("lock", "predict", "fits", "score"):
        with pytest.raises(SystemExit):
            refine.main([subcommand])
