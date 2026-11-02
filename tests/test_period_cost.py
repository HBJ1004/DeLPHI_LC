"""Period-cost protocol checks; every generated artifact stays under tmp_path."""

from __future__ import annotations

import json
import math
import os
import queue
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from lc_pipeline.v2.convexinv import ConvexinvProvenance, PeriodScanParameters, PeriodScanRow
from lc_pipeline.v2.serialization import sha256_file
from repro.own_period import period_cost as cost


def write_lightcurve(path: Path):
    # Two apparitions, adequate observations for the unmodified degree-6 fit.
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
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


@pytest.mark.parametrize("span", [1155.0, 1506.0, 12619.984927, 40000.0])
def test_serialized_block_bounds_reproduce_start_grid_and_integer_rule(tmp_path, span):
    grid = cost.TrialGrid(2450000, 2450000 + span)
    span = grid.jd_max - grid.jd_min
    assert grid.step == 0.5 / span / 24 * 0.8
    assert grid.count == int((0.5 - 0.01) / grid.step) + 1
    starts = (0, grid.count // 2, grid.count - 10,
              *cost.sample_starts("asteroid_101", grid.count))
    for block, start in enumerate(starts):
        path = tmp_path / f"parameters-{block}"
        cost._write_parameters(path, grid, start, 10)
        period_start, period_end, coefficient = map(float, path.read_text().splitlines()[0].split())
        fs, fe = 1 / period_start, 1 / period_end
        # Directly reproduce period_scan.c's reciprocal bounds and C integer cast.
        step = 0.5 / span / 24 * coefficient
        assert int((fs - fe) / step) + 1 == 10
        assert cost.integer_trial_count(period_start, period_end, step) == 10
        for local_index in range(10):
            actual_frequency = fs - local_index * step
            global_frequency = 0.5 - (start + local_index) * step
            assert actual_frequency == pytest.approx(global_frequency, rel=2e-14, abs=2e-16)
        assert path.read_text().splitlines()[1:] == PeriodScanParameters(2, 100).render().splitlines()[1:]


@pytest.mark.parametrize("trials,chunks", [(10, 1), (10, 10), (35, 12), (33958, 12), (44277, 12)])
def test_chunks_cover_the_grid_exactly_once(trials, chunks):
    ranges = cost.chunk_ranges(trials, chunks)
    covered = [index for start, count in ranges for index in range(start, start + count)]
    assert covered == list(range(trials))
    assert len(ranges) == chunks
    assert max(count for _, count in ranges) - min(count for _, count in ranges) <= 1


@pytest.mark.parametrize("span", [1155.0, 1506.0, 12619.984927, 1.0, 100.0 / 3])
def test_chunks_serialized_integer_rule_including_global_endpoint(tmp_path, span):
    grid = cost.TrialGrid(0, span)
    for block, (start, count) in enumerate(cost.chunk_ranges(grid.count, min(12, grid.count))):
        bounds = cost._write_parameters(tmp_path / f"chunk-{block}", grid, start, count)
        assert int((1 / bounds[0] - 1 / bounds[1]) / grid.step) + 1 == count
        assert 1 / bounds[0] == pytest.approx(0.5 - start * grid.step, abs=1e-16)
    assert grid.frequency(grid.count - 1) >= 0.01 - 1e-16


def test_one_trial_bounds_and_exclusive_parameters(tmp_path):
    grid = cost.TrialGrid(0, 1155)
    path = tmp_path / "parameters"
    bounds = cost._write_parameters(path, grid, 17, 1)
    assert cost.integer_trial_count(*bounds, grid.step) == 1
    contents = path.read_bytes()
    with pytest.raises(FileExistsError):
        cost._write_parameters(path, grid, 17, 1)
    assert path.read_bytes() == contents


def test_sampling_seed_is_stable_and_object_specific():
    starts = cost.sample_starts("asteroid_101", 100000)
    assert starts == cost.sample_starts("asteroid_101", 100000)
    assert starts != cost.sample_starts("asteroid_111", 100000)
    assert len(starts) == 10
    assert all(0 <= start <= 99990 for start in starts)
    # A fixed fixture catches accidental seed encoding or RNG changes.
    assert starts == (54699, 1760, 30343, 17197, 32158, 54293, 53305, 60375, 32481, 81817)
    assert cost.sample_starts("asteroid_101", 10) == (0,) * 10


@pytest.mark.parametrize("function,args", [
    (cost.TrialGrid, (1, 1)), (cost.TrialGrid, (math.nan, 1)),
    (cost.sample_starts, ("asteroid_101", 9)),
    (cost.chunk_ranges, (10, 0)), (cost.chunk_ranges, (10, 11)),
])
def test_invalid_grid_requests(function, args):
    with pytest.raises(ValueError):
        function(*args)


def test_lightcurve_times_are_first_column_and_all_rows(tmp_path):
    path = tmp_path / "lc.txt"
    # Unsorted times in separate curves; brightness/geometry are not timestamps.
    path.write_text("2\n2 0\n20 9999 1 0 0 1 1 0\n10 1 1 0 0 1 1 0\n"
                    "2 1\n50 1 1 0 0 1 1 0\n30 1 1 0 0 1 1 0\n")
    info = cost.lightcurve_span(path)
    assert info == {"jd_min": 10, "jd_max": 50, "baseline_days": 40,
                    "lightcurves": 2, "observations": 4}
    path.write_text(path.read_text() + "unexpected\n")
    with pytest.raises(ValueError, match="extra"):
        cost.lightcurve_span(path)


def row(period, chi2):
    return PeriodScanRow(period, 0.1, chi2, 50, 1.0)


def test_fitted_period_offsets_are_descriptive():
    grid = cost.TrialGrid(0, 1155)
    start = 200
    fitted_frequency = grid.frequency(start) + 2.75 * grid.step
    result = cost.fitted_rows([row(1 / fitted_frequency, 1)], grid, start)[0]
    assert result["frequency_offset_steps"] == pytest.approx(2.75)
    assert result["period_hours"] != result["start_period_hours"]
    assert result["trial_index"] == start


@pytest.mark.parametrize("returned,timed_out,return_code,expected_status", [
    (10, False, 0, "complete"),
    (9, False, 0, "not_finished"),
    (11, False, 0, "not_finished"),
    (3, True, None, "not_finished"),
    (10, False, 2, "not_finished"),
])
def test_worker_records_exact_row_count_and_partial_failures(
        tmp_path, monkeypatch, returned, timed_out, return_code, expected_status):
    # Exercise reporting without launching any extra solver calls or changing
    # this pytest process's affinity/priority.
    cpus = queue.Queue()
    cpus.put(12)
    monkeypatch.setattr(cost, "_CPU_QUEUE", cpus)
    monkeypatch.setattr(cost.os, "sched_setaffinity", lambda *args: None)
    monkeypatch.setattr(cost.os, "sched_getaffinity", lambda *args: {12})
    monkeypatch.setattr(cost.os, "nice", lambda *args: 19)
    monkeypatch.setattr(cost.os, "getpriority", lambda *args: 19)
    path = tmp_path / "lc.txt"
    write_lightcurve(path)
    span = cost.lightcurve_span(path)
    info = {"object_id": "synthetic", **span, "lightcurve_file": str(path),
            "lightcurve_sha256": sha256_file(path)}
    task = cost._task(info, 10, 10, tmp_path / "run", tmp_path, 0, 600, "sample")

    def fake_run(**kwargs):
        raw = kwargs["output_directory"] / "period-scan.txt"
        raw.write_text("10 0.1 1 50 0.1\n" * returned)
        provenance = ConvexinvProvenance("binary", "source", "input", "compiler", ("solver",))
        return SimpleNamespace(return_code=return_code, timed_out=timed_out,
            wall_time_seconds=4, process_cpu_time_seconds=3,
            rows=() if timed_out or return_code else tuple(row(10, 1) for _ in range(returned)),
            provenance=provenance, stdout_sha256="stdout", stderr_sha256="stderr",
            output_periods_sha256=sha256_file(raw))

    monkeypatch.setattr(cost, "run_period_scan", fake_run)
    result = cost._run_task(task)
    assert result["status"] == expected_status
    assert result["rows_returned"] == returned
    assert result["timed_out"] == timed_out
    assert result["return_code"] == return_code
    assert result["wall_seconds"] == 4
    assert result["cpu_seconds"] == 3
    assert len(result["fitted_rows"]) == returned
    assert cpus.get_nowait() == 12
    assert json.loads((tmp_path / "run/record.json").read_text())["status"] == expected_status
    # All fitted periods were deliberately far from the start grid. Only the
    # exact count, exit status, and timeout decide whether this call completed.
    if returned == 10 and return_code == 0 and not timed_out:
        assert abs(result["fitted_rows"][0]["frequency_offset_steps"]) > 100


def test_scoring_global_minimum_and_distant_nearby_chi2_minima():
    rows = [row(4, 20), row(5, 10.5), row(6, 20), row(10, 10),
            row(10.05, 12), row(10.06, 10.2), row(11, 20),
            row(20, 11), row(21, 20), row(30, 11.01), row(31, 20)]
    score = cost.score_periods(rows, 10)
    assert score["period_found_hours"] == 10
    assert score["best_trial_index"] == 3
    assert score["period_class"] == "P"
    assert score["ambiguity_count"] == 2
    assert [r["trial_index"] for r in score["alternative_minima"]] == [1, 7]


@pytest.mark.parametrize("period,label", [(5, "P/2"), (20, "2P"), (10.1, "P"), (7, "other")])
def test_period_aliases(period, label):
    assert cost.score_periods([row(period, 1)], 10)["period_class"] == label


def test_plateau_minimum_counted_once_and_endpoints_included():
    rows = [row(10, 10), row(11, 20), row(20, 10.5), row(21, 10.5), row(22, 20)]
    score = cost.score_periods(rows, 10)
    assert score["ambiguity_count"] == 1
    assert score["alternative_minima"][0]["trial_index"] == 2


def test_bootstrap_cost_and_incomplete_calls():
    blocks = [{"wall_seconds": 20, "cpu_seconds": 10, "rows_returned": 10,
               "status": "complete"} for _ in range(10)]
    estimate = cost.estimate_cost(blocks, 1000)
    assert estimate["cpu_seconds"]["estimated_full"] == 1000
    assert estimate["cpu_seconds"]["ci95"] == [1000, 1000]
    assert estimate["wall_seconds"]["ci95"] == [2000, 2000]
    blocks[0]["cpu_seconds"] = 40
    assert cost.estimate_cost(blocks, 1000) == cost.estimate_cost(blocks, 1000)
    blocks[0]["status"] = "not_finished"
    with pytest.raises(ValueError, match="all ten"):
        cost.estimate_cost(blocks, 1000)


def test_lower_bound_keeps_timed_out_blocks(tmp_path, temporary_delphi_data):
    blocks = [{"wall_seconds": 20, "cpu_seconds": 10, "rows_returned": 10,
               "status": "complete", "timed_out": False} for _ in range(10)]
    blocks[0].update(status="not_finished", timed_out=True, rows_returned=0,
                     wall_seconds=600, cpu_seconds=590)
    bound = cost.lower_bound_cost(blocks, 1000)
    assert bound["timed_out_blocks"] == 1
    assert bound["cpu_seconds"]["estimated_full_at_least"] == pytest.approx((590 + 90) / 100 * 1000)
    blocks[1].update(status="not_finished", timed_out=False)
    with pytest.raises(ValueError, match="lower bound"):
        cost.lower_bound_cost(blocks, 1000)
    counts = fixture_counts()
    records = [fake_complete(task) for task in cost._sample_tasks(counts, tmp_path, cost.SOURCE)]
    records[0].update(status="not_finished", timed_out=True, rows_returned=0)
    assert len(cost._validated_samples(records, counts, allow_timeouts=True)) == 242
    records[1].update(status="not_finished", return_code=1)
    with pytest.raises(ValueError, match="not finished"):
        cost._validated_samples(records, counts, allow_timeouts=True)


def test_cpu_ranges():
    assert cost.parse_cpus("7,12-14") == (7, 12, 13, 14)
    for text in ("12,12", "14-12", "-1", "", "12-13-14"):
        with pytest.raises(Exception):
            cost.parse_cpus(text)


def fixture_counts():
    grid = cost.TrialGrid(0, 1155)
    ids = list(cost.SAMPLE_OBJECTS)
    ids.extend(f"synthetic_{index}" for index in range(170 - len(ids)))
    return {oid: {"object_id": oid, "jd_min": 0, "jd_max": 1155,
                  "frequency_step_per_hour": grid.step, "trial_count": grid.count,
                  "lightcurve_file": "unused", "lightcurve_sha256": "unused"}
            for oid in ids}


def fake_complete(task, cpu_seconds=10):
    return {**task, "status": "complete", "return_code": 0, "timed_out": False,
            "rows_returned": task["expected_rows"], "cpu_seconds": cpu_seconds,
            "wall_seconds": cpu_seconds * 2, "fitted_rows": []}


def test_calibration_repeats_identical_blocks_without_startup(tmp_path, temporary_delphi_data):
    counts = fixture_counts()
    sample = cost._sample_tasks(counts, tmp_path, cost.SOURCE)
    calibration = cost._sample_tasks(counts, tmp_path, cost.SOURCE, True)
    assert len(sample) == 22 * 11
    assert len(calibration) == 4 * 10
    by_key = {(r["object_id"], r["block"]): r for r in sample}
    for task in calibration:
        other = by_key[(task["object_id"], task["block"])]
        assert task["start_index"] == other["start_index"]
        assert task["period_bounds_hours"] == other["period_bounds_hours"]
        assert task["timeout_seconds"] == 600


def test_sample_validation_does_not_drop_failed_or_missing_blocks(tmp_path, temporary_delphi_data):
    counts = fixture_counts()
    records = [fake_complete(task) for task in cost._sample_tasks(counts, tmp_path, cost.SOURCE)]
    assert len(cost._validated_samples(records, counts)) == 242
    with pytest.raises(ValueError, match="missing"):
        cost._validated_samples(records[:-1], counts)
    records[0]["timed_out"] = True
    with pytest.raises(ValueError, match="not finished"):
        cost._validated_samples(records, counts)


def test_count_cli_reads_observation_span_without_reference_solutions(tmp_path):
    lightcurve = tmp_path / "lc.txt"
    write_lightcurve(lightcurve)
    ids = [f"synthetic_{index}" for index in range(170)]
    cohort = tmp_path / "cohort.json"
    cohort.write_text(json.dumps({"periods_hours": dict.fromkeys(ids, None)}))
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text("".join(json.dumps({"object_id": oid,
        "lightcurve": {"source_path": "lc.txt"}}) + "\n" for oid in ids))
    output = tmp_path / "output"
    command = ["count", "--output-dir", str(output), "--catalog", str(catalog),
               "--cohort", str(cohort), "--lightcurve-root", str(tmp_path)]
    assert cost.main(command) == 0
    counts = cost._counts(output)
    assert len(counts) == 170
    assert counts[ids[0]]["observations"] == 160
    assert counts[ids[0]]["baseline_days"] == pytest.approx(1155.09875)
    with pytest.raises(FileExistsError):
        cost.main(command)


def test_score_waits_for_full_scans_before_reading_references(tmp_path, temporary_delphi_data):
    counts = fixture_counts()
    cost._write_json(tmp_path / "trial-counts.json", {"objects": list(counts.values())})
    with (tmp_path / "sample.jsonl").open("x") as stream:
        for task in cost._sample_tasks(counts, tmp_path, cost.SOURCE):
            cost._dump(stream, fake_complete(task))
    (tmp_path / "full.jsonl").write_text("")
    with pytest.raises(ValueError, match="both complete"):
        cost.main(["score", "--output-dir", str(tmp_path),
                   "--catalog", str(tmp_path / "must-not-read.jsonl")])
    assert not (tmp_path / "period-cost-summary.json").exists()


def test_score_end_to_end_with_synthetic_tables(tmp_path, temporary_delphi_data):
    counts = fixture_counts()
    grid = cost.TrialGrid(0, 1)
    for oid in cost.FULL_OBJECTS:
        counts[oid].update(jd_max=1, frequency_step_per_hour=grid.step, trial_count=grid.count)
    cost._write_json(tmp_path / "trial-counts.json", {"objects": list(counts.values())})
    with (tmp_path / "sample.jsonl").open("x") as stream:
        for task in cost._sample_tasks(counts, tmp_path, cost.SOURCE):
            cost._dump(stream, fake_complete(task))
    with (tmp_path / "calibration.jsonl").open("x") as stream:
        for task in cost._sample_tasks(counts, tmp_path, cost.SOURCE, True):
            cost._dump(stream, fake_complete(task, cpu_seconds=5))
    with (tmp_path / "full.jsonl").open("x") as stream:
        for oid in cost.FULL_OBJECTS:
            chunks = []
            for block, (start, count) in enumerate(cost.chunk_ranges(grid.count, 3)):
                directory = tmp_path / oid / f"chunk-{block}"
                directory.mkdir(parents=True)
                raw = directory / "period-scan.txt"
                raw.write_text("".join(f"{10 if i == 3 else 20 + i} 0.1 {1 if i == 3 else 2} 50 0.1\n"
                                       for i in range(start, start + count)))
                task = cost._task(counts[oid], start, count, directory, cost.SOURCE, block, 28800, "full")
                chunks.append({**fake_complete(task), "output_periods_sha256": sha256_file(raw)})
            cost._dump(stream, {"object_id": oid, "status": "complete", "chunks": chunks,
                                "rows_returned": grid.count, "cpu_seconds": grid.count,
                                "wall_seconds": 30, "sum_solver_wall_seconds": 2 * grid.count})
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text("".join(json.dumps({"object_id": oid, "solutions": [{"period_hours": 10}]})
                               + "\n" for oid in cost.FULL_OBJECTS))
    assert cost.main(["score", "--output-dir", str(tmp_path), "--catalog", str(catalog)]) == 0
    summary = json.loads((tmp_path / "period-cost-summary.json").read_text())
    assert summary["P_over_E_speed_factor"] == 2
    assert len(summary["estimates"]) == 22
    for value in summary["full_scan_checks"].values():
        assert value["score"]["period_class"] == "P"
        assert value["rows_returned"] == grid.count
        assert value["cpu_actual_over_estimate"] == 1
    assert "one CPU core" in (tmp_path / "RESULTS.md").read_text()
    with pytest.raises(FileExistsError):
        cost.main(["score", "--output-dir", str(tmp_path), "--catalog", str(catalog)])


def test_real_two_trial_smoke(tmp_path, require_delphi_files):
    binary = cost.SOURCE / "period_scan"
    require_delphi_files(binary)
    lightcurve = tmp_path / "tiny-lightcurve.txt"
    write_lightcurve(lightcurve)
    span = cost.lightcurve_span(lightcurve)
    grid = cost.TrialGrid(span["jd_min"], span["jd_max"])
    info = {"object_id": "smoke", **span, "lightcurve_file": str(lightcurve),
            "lightcurve_sha256": sha256_file(lightcurve)}
    task = cost._task(info, 0, 2, tmp_path / "smoke-run", cost.SOURCE, 0, 600, "smoke")
    # The read-only data binary may have lost its execute bit on the mounted
    # filesystem. Execute an identical snapshot, with permissions changed only
    # inside tmp_path; retain the original source root for provenance.
    executable = tmp_path / "period_scan"
    shutil.copyfile(binary, executable)
    executable.chmod(0o700)
    assert sha256_file(executable) == sha256_file(binary)
    task["executable"] = str(executable)
    cpu = min(os.sched_getaffinity(0))
    records = cost.run_tasks([task], (cpu,), 1)
    assert len(records) == 1
    record = records[0]
    assert record["status"] == "complete", record
    assert record["return_code"] == 0
    assert record["timed_out"] is False
    assert record["rows_returned"] == 2
    assert record["cpu_seconds"] > 0
    assert record["wall_seconds"] > 0
    assert record["cpu_list"] == [cpu]
    assert record["nice"] == 19
    assert cost.integer_trial_count(*record["period_bounds_hours"], grid.step) == 2
    assert len(record["fitted_rows"]) == 2
    assert all(math.isfinite(r["frequency_offset_steps"]) for r in record["fitted_rows"])
    assert (tmp_path / "smoke-run/period-scan.txt").is_file()
    assert json.loads((tmp_path / "smoke-run/record.json").read_text())["rows_returned"] == 2
    # Pinning/nicing happen only inside the worker process.
    assert os.sched_getaffinity(0) >= {cpu}
