"""Synthetic pipeline checks; the sole real inversion is tiny and CPU-only."""

from concurrent.futures import Future
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

from repro.own_period import refine_chain_post as post
from repro.own_period.refine_chain import PROTOCOL_OBJECTS


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    data = tmp_path / "data/own-period"
    benchmark = data / "benchmark"
    output = data / "chain"
    output.mkdir(parents=True)
    benchmark.mkdir()
    monkeypatch.setattr(post, "DATA", data)
    monkeypatch.setattr(post, "_validate_runtime", lambda lock: None)
    protocol = output / "PROTOCOL.md"
    protocol.write_text("synthetic binding")
    (data.parent / "external-models-final.json").write_text("{}")
    shutil.copyfile(post.BENCHMARK / "run_direct_benchmark.py", benchmark / "run_direct_benchmark.py")
    rows, refined = [], []
    for i, oid in enumerate(PROTOCOL_OBJECTS):
        lc = tmp_path / f"{oid}.txt"
        lc.write_text("photometry only")
        rows.append({"object_id": oid, "fold": i // 4, "lightcurve": post.binding(lc),
                     "period_hours": 9.0, "structure": {"total_observations": 3}})
        refined.append({"object_id": oid, "fold": i // 4, "status": "complete",
                        "period_found_hours_raw": 10.123456789, "period_found_hours": 10.123457,
                        "cpu_seconds": 14.06})
    lock = {"schema": "test", "objects": rows, "solver": {"test": "unchanged"},
            "solver_settings": {"convergence_tolerance": 0.0003, "iteration_cap": 1000,
                                "timeout_seconds_per_start": 300},
            "environment": {"interpreter": "unchanged"}, "runtime_sources": {},
            "code_root": "unchanged", "classical_starts": [[1, 0, 0]]}
    write(benchmark / "evaluation-lock.json", lock)
    write(output / "refined-periods.json", {"objects": refined, "periods_hours": {
        r["object_id"]: r["period_found_hours"] for r in refined}})
    write(data / "own-periods-170.json", {"periods_hours": {r["object_id"]: 9 for r in rows}})
    return SimpleNamespace(output_dir=output, benchmark_dir=benchmark, protocol=protocol,
                           cpus=(23,), workers=1, device="cuda", published_dir=data / "published",
                           cost_summary=data / "cost-summary.json")


def one_success(args):
    path = args.output_dir / "refined-periods.json"
    value = post.read(path)
    for row in value["objects"][1:]:
        row.update(status="not_finished", period_found_hours=None, period_found_hours_raw=None)
        del value["periods_hours"][row["object_id"]]
    write(path, value)


def prediction(args, row, request):
    return {"request": request, "object_id": row["object_id"], "fold": row["fold"],
            "period_hours": row["period_hours"], "initial_axes": np.eye(3).tolist(),
            "refined_axes": np.eye(3).tolist(), "refined_scores": [0.8, 0.5, 0.3],
            "inference_seconds": 0.25}


def fake_predictions(args):
    assert post.command_lock(args) == 0
    lock = post.checked_lock(args)
    request = post.prediction_request(args, lock)
    directory = args.output_dir / "predictions-refined"
    paths = []
    for row in lock["objects"]:
        path = directory / row["object_id"] / "prediction.json"
        write(path, prediction(args, row, request))
        paths.append(path)
    post.save_outputs({directory / "prediction-plan.json": request,
                       directory / "predictions.json": {
                           "request": request, "records": [post.binding(p) for p in paths],
                           "failed_object_ids": lock["refinement"]["failed_object_ids"]}})


def fake_start(task, lock):
    directory = post._start_path(task)
    directory.mkdir(parents=True, exist_ok=False)
    product = directory / "parameters.txt"
    product.write_text("synthetic fixed-period fit")
    record = {"start_index": task["start_index"], "initial_axis": task["initial_axis"],
              "selected_fit": {"axis": task["initial_axis"], "start_index": task["start_index"],
                               "final_relative_rms": 1 + task["start_index"] / 1000},
              "files": {product.name: post.binding(product)["sha256"]},
              "solver_wall_seconds": 2.0, "failure_reason": None}
    write(directory / "record.json", record)
    write(directory / "timing.json", {"task": task, "record": post.binding(directory / "record.json"),
                                     "solver_cpu_seconds": 1.5, "nice": 19, "cpu_list": [23]})
    return post.validate_start(task)


class FakePool:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def submit(self, function, *args):
        future = Future()
        future.set_result(function(*args))
        return future


@pytest.fixture
def fitted(inputs, monkeypatch):
    one_success(inputs)
    fake_predictions(inputs)
    monkeypatch.setattr(post, "ProcessPoolExecutor", FakePool)
    monkeypatch.setattr(post, "_fit_start", fake_start)
    assert post.command_fits(inputs) == 0
    return inputs


def test_lock_preserves_every_benchmark_field_except_period_and_provenance(inputs):
    before = post.read(inputs.benchmark_dir / "evaluation-lock.json")
    assert post.command_lock(inputs) == 0
    actual = post.read(inputs.output_dir / "evaluation-lock-refined.json")
    assert actual["period_provenance"] == post.PROVENANCE
    for key, value in before.items():
        if key != "objects":
            assert actual[key] == value
    assert len(actual["objects"]) == 20
    for old, new in zip(before["objects"], actual["objects"]):
        assert new == {**old, "period_hours": 10.123457}
    assert actual["refinement"]["failed_object_ids"] == []


def test_lock_failure_omitted_and_listed_but_periods_file_keeps_twenty(inputs):
    one_success(inputs)
    post.command_lock(inputs)
    lock = post.checked_lock(inputs)
    assert len(lock["objects"]) == 1
    assert lock["refinement"]["failed_object_ids"] == list(PROTOCOL_OBJECTS[1:])
    periods = post.read(inputs.output_dir / "periods-refined.json")["periods_hours"]
    assert len(periods) == 20
    assert all(periods[oid] is None for oid in PROTOCOL_OBJECTS[1:])


def test_lock_resume_no_overwrite_and_changed_input_stops(inputs):
    post.command_lock(inputs)
    paths = [inputs.output_dir / n for n in ("evaluation-lock-refined.json", "periods-refined.json")]
    originals = [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths]
    post.command_lock(inputs)
    assert [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths] == originals
    value = post.read(inputs.output_dir / "refined-periods.json")
    value["objects"][0]["cpu_seconds"] += 1
    write(inputs.output_dir / "refined-periods.json", value)
    with pytest.raises(ValueError, match="mismatches"):
        post.command_lock(inputs)
    assert [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths] == originals


@pytest.mark.parametrize("change", ["extra", "duplicate", "fold", "rounding", "mapping", "status"])
def test_invalid_refinement_rejected_before_lock(inputs, change):
    path = inputs.output_dir / "refined-periods.json"
    value = post.read(path)
    if change == "extra":
        value["objects"].append(value["objects"][0])
    elif change == "duplicate":
        value["objects"][1] = value["objects"][0]
    elif change == "fold":
        value["objects"][0]["fold"] = 3
    elif change == "rounding":
        value["objects"][0]["period_found_hours"] = 10.123456789
    elif change == "mapping":
        value["periods_hours"][PROTOCOL_OBJECTS[0]] = 12
    else:
        value["objects"][0]["status"] = "unknown"
    write(path, value)
    with pytest.raises(ValueError):
        post.command_lock(inputs)
    assert not (inputs.output_dir / "evaluation-lock-refined.json").exists()


def test_save_preflights_all_outputs(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    write(b, {"old": 1})
    with pytest.raises(ValueError):
        post.save_outputs({a: {}, b: {"new": 2}})
    assert not a.exists()
    assert post.read(b) == {"old": 1}


def test_predict_requires_real_inherited_flock(tmp_path):
    path = tmp_path / "gpu.lock"
    path.touch()
    env = {**os.environ, "LOCK": str(path), "PYTHONDONTWRITEBYTECODE": "1"}
    code = "from repro.own_period.refine_chain_post import require_gpu_lock; require_gpu_lock()"
    denied = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert denied.returncode != 0
    assert "requires runners/lock_run.sh" in denied.stderr
    # Same fd inheritance and flock semantics as the real wrapper, without GPU checks.
    allowed = subprocess.run(["bash", "-c", 'exec 9>"$LOCK"; flock 9; exec "$@"',
                              "lock-test", sys.executable, "-c", code],
                             env=env, capture_output=True, text=True)
    assert allowed.returncode == 0, allowed.stderr


def test_predict_resume_does_not_reload_models_or_read_targets(inputs, monkeypatch):
    from repro.own_period import predict_oof as p

    post.command_lock(inputs)
    monkeypatch.setattr(post, "require_gpu_lock", lambda: None)
    calls = []
    monkeypatch.setattr(p, "_load_models", lambda *args: (calls.append(args[2]) or [], [], []))
    monkeypatch.setattr(p, "load_damit_object", lambda *args: pytest.fail("read DAMIT spin/shape"))

    def predict(path, row, models, device, request):
        value = prediction(inputs, row, request)
        post.save_outputs({path: value})
        return value

    monkeypatch.setattr(post, "predict_one", predict)
    assert post.command_predict(inputs) == 0
    assert calls == [(0,), (1,), (2,), (3,), (4,)]
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns)
              for p in (inputs.output_dir / "predictions-refined").rglob("*.json")}
    monkeypatch.setattr(p, "_load_models", lambda *a: pytest.fail("loaded models during resume"))
    assert post.command_predict(inputs) == 0
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == v for p, v in before.items())
    first = next(p for p in before if p.name == "prediction.json")
    value = post.read(first)
    value["period_hours"] += 1
    write(first, value)
    with pytest.raises(ValueError, match="different inputs"):
        post.command_predict(inputs)


def test_photometry_only_prediction_follows_oof_numerical_path(inputs, monkeypatch):
    from repro.own_period import predict_oof as p
    from lc_pipeline.k3 import damit

    row = post.read(inputs.benchmark_dir / "evaluation-lock.json")["objects"][0]
    seen = []
    monkeypatch.setattr(damit, "_read_lc", lambda path: seen.append(path) or ("epochs",))
    monkeypatch.setattr(p, "load_damit_object", lambda *a: pytest.fail("reference loader called"))
    monkeypatch.setattr(p, "tokenize_epochs", lambda epochs, known_period: (epochs, known_period.hours))

    class Tensor:
        def to(self, device):
            assert device == "cpu"
            return "input"

    def collate(examples):
        assert examples[0].target_axes.tolist() == [[1, 0, 0]]
        assert examples[0].tokenized == (("epochs",), row["period_hours"])
        return {"phase_features": Tensor(), "target_axes": object()}

    monkeypatch.setattr(p, "collate_examples", collate)

    def score(model, inputs, chunk_size):
        assert inputs == {"phase_features": "input"} and chunk_size == 1024
        return [model, 2]

    monkeypatch.setattr(p, "score_axial_grid", score)
    monkeypatch.setattr(p, "ensemble_score_grids", lambda a: seen.append(a.shape) or np.ones((1, 2)))
    monkeypatch.setattr(p, "modes_from_score_grid", lambda a: [SimpleNamespace(axis_xyz=v) for v in np.eye(3)])
    monkeypatch.setattr(p, "refine_ensemble_axes", lambda models, inputs, starts: (starts, np.ones(3)))
    path = inputs.output_dir / "test-prediction.json"
    result = post.predict_one(path, row, [1, 2, 3, 4, 5], "cpu", {})
    assert seen == [Path(row["lightcurve"]["path"]), (5, 1, 2)]
    assert result["refined_axes"] == np.eye(3).tolist()
    assert "errors_deg" not in result


def test_completed_predictions_cannot_regenerate_deleted_object(inputs, monkeypatch):
    fake_predictions(inputs)
    shutil.rmtree(inputs.output_dir / "predictions-refined" / PROTOCOL_OBJECTS[0])
    monkeypatch.setattr(post, "require_gpu_lock", lambda: None)
    monkeypatch.setattr(post, "predict_one", lambda *a: pytest.fail("regenerated completed prediction"))
    with pytest.raises(RuntimeError, match="missing records"):
        post.command_predict(inputs)


def test_partial_prediction_directory_is_preserved(inputs, monkeypatch):
    post.command_lock(inputs)
    request = post.prediction_request(inputs, post.checked_lock(inputs))
    directory = inputs.output_dir / "predictions-refined"
    write(directory / "prediction-plan.json", request)
    incomplete = directory / PROTOCOL_OBJECTS[0]
    incomplete.mkdir()
    (incomplete / "keep.txt").write_text("unfinished")
    monkeypatch.setattr(post, "require_gpu_lock", lambda: None)
    with pytest.raises(FileExistsError, match="incomplete prediction"):
        post.command_predict(inputs)
    assert (incomplete / "keep.txt").read_text() == "unfinished"


def test_published_ladder_mapping(inputs):
    from lc_pipeline import grid_benchmark as grid

    indexes = post.ladder_indexes(inputs.benchmark_dir)
    assert len(indexes) == len(set(indexes)) == 140
    broad = [{"selected_fit": None, "id": f"g{i}"} for i in range(146)]
    standard = [{"selected_fit": None, "id": f"s{i}"} for i in range(6)]
    arms = post.arm_records({"delphi": standard, "standard6": standard, "classical20": broad}, indexes)
    assert len(grid.pole_grid(20)) == 146
    assert [r["id"] for r in arms["classical12"]] == [f"s{i}" for i in range(6)] + [f"g{i}" for i in indexes[:6]]
    assert len(arms["classical146"]) == 146
    assert arms["classical146"][-1]["id"] == f"g{indexes[-1]}"


def test_all_starts_saved_at_refined_period_and_resume_immutable(fitted, monkeypatch):
    args = fitted
    plan = post.read(args.output_dir / "fits/fit-plan.json")
    assert len(plan["tasks"]) == 158  # six DeLPHI, six standard, 146 grid
    assert {t["job"]["period_hours"] for t in plan["tasks"]} == {10.123457}
    assert plan["nice"] == 19 and plan["cpus"] == [23]
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns)
              for p in (args.output_dir / "fits").rglob("*") if p.is_file()}
    monkeypatch.setattr(post, "_fit_start", lambda *a: pytest.fail("reran saved start"))
    assert post.command_fits(args) == 0
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == v for p, v in before.items())


def test_spawn_workers_import_repo_repro_and_frozen_runtime(inputs, tmp_path, monkeypatch):
    one_success(inputs)
    fake_predictions(inputs)
    task = {"directory": str(inputs.output_dir / "fits/probe"), "start_index": 0}
    monkeypatch.setattr(post, "fit_tasks", lambda *args: [task])
    monkeypatch.setattr(post, "ladder_indexes", lambda *args: [])
    monkeypatch.setattr(post, "_fit_completion", lambda *args: {})

    frozen = tmp_path / "frozen"
    for name in ("repro", "lc_pipeline"):
        (frozen / name).mkdir(parents=True)
        (frozen / name / "__init__.py").write_text("MARKER = 'frozen'\n")
    # Match _validate_runtime: the frozen repro has no own_period package.
    monkeypatch.syspath_prepend(str(frozen))
    parent_path = sys.path[:]
    probe = ("(__import__('repro.own_period.refine_chain_post', fromlist=['__file__']).__file__, "
             "__import__('lc_pipeline').__file__, __import__('lc_pipeline').MARKER, "
             "__import__('sys').path)")
    futures = []

    class ImportProbePool(post.ProcessPoolExecutor):
        def submit(self, function, *args):
            # Use command_fits' real spawn setup and initializer, without a solver.
            future = super().submit(eval, probe)
            futures.append(future)
            return future

    monkeypatch.setattr(post, "ProcessPoolExecutor", ImportProbePool)
    assert post.command_fits(inputs) == 0
    assert sys.path == parent_path
    assert len(futures) == 1
    module, runtime, marker, child_path = futures[0].result()
    assert Path(module).resolve() == Path(post.__file__).resolve()
    assert Path(runtime).resolve() == frozen / "lc_pipeline/__init__.py"
    assert marker == "frozen"
    assert child_path == parent_path


@pytest.mark.parametrize("workers", [0, -1, 2])
def test_invalid_worker_count_stops_before_solver(inputs, monkeypatch, workers):
    one_success(inputs)
    fake_predictions(inputs)
    inputs.workers = workers
    monkeypatch.setattr(post, "ProcessPoolExecutor", lambda *a, **k: pytest.fail("launched invalid worker count"))
    with pytest.raises(ValueError, match="workers must be"):
        post.command_fits(inputs)
    assert not (inputs.output_dir / "fits").exists()


def test_resume_after_interruption_skips_finished_start(inputs, monkeypatch):
    one_success(inputs)
    fake_predictions(inputs)
    monkeypatch.setattr(post, "ProcessPoolExecutor", FakePool)
    saved = []

    def interrupted(task, lock):
        if saved:
            raise KeyboardInterrupt
        saved.append(task)
        return fake_start(task, lock)

    monkeypatch.setattr(post, "_fit_start", interrupted)
    with pytest.raises(KeyboardInterrupt):
        post.command_fits(inputs)
    before = (post._start_path(saved[0]) / "record.json").read_bytes()
    calls = []

    def resumed(task, lock):
        assert task != saved[0]
        calls.append(task)
        return fake_start(task, lock)

    monkeypatch.setattr(post, "_fit_start", resumed)
    assert post.command_fits(inputs) == 0
    assert len(calls) == 157
    assert (post._start_path(saved[0]) / "record.json").read_bytes() == before


@pytest.mark.parametrize("change", ["product", "record", "timing", "missing_timing", "plan", "periods"])
def test_changed_or_incomplete_saved_fit_stops_resume(fitted, monkeypatch, change):
    plan_path = fitted.output_dir / "fits/fit-plan.json"
    task = post.read(plan_path)["tasks"][0]
    directory = post._start_path(task)
    if change == "product":
        (directory / "parameters.txt").write_text("changed")
    elif change == "record":
        value = post.read(directory / "record.json")
        value["start_index"] += 1
        write(directory / "record.json", value)
    elif change == "timing":
        value = post.read(directory / "timing.json")
        value["task"]["initial_axis"] = [0, 0, 1]
        write(directory / "timing.json", value)
    elif change == "missing_timing":
        (directory / "timing.json").unlink()
    elif change == "plan":
        value = post.read(plan_path)
        value["workers"] += 1
        write(plan_path, value)
    else:
        value = post.read(fitted.output_dir / "refined-periods.json")
        value["objects"][0]["cpu_seconds"] += 1
        write(fitted.output_dir / "refined-periods.json", value)
    monkeypatch.setattr(post, "_fit_start", lambda *a: pytest.fail("launched after mismatch"))
    with pytest.raises((ValueError, FileExistsError)):
        post.command_fits(fitted)


def test_select_by_rms_then_index_never_reference_error():
    records = [{"selected_fit": {"axis": [1, 0, 0], "final_relative_rms": 0.2, "start_index": 0}},
               {"selected_fit": {"axis": [0, 1, 0], "final_relative_rms": 0.1, "start_index": 2}},
               {"selected_fit": {"axis": [0, 0, 1], "final_relative_rms": 0.1, "start_index": 1}}]
    fit = post.selected_fit(records)
    assert fit["start_index"] == 1
    assert post.fit_outcome(fit, [[1, 0, 0]])["error_deg"] == 90
    assert post.selected_fit([{"selected_fit": None}]) is None


def test_failure_conventions_and_axial_reference_error():
    failed = post.fit_outcome(None, [[1, 0, 0]])
    valid = post.fit_outcome({"axis": [-1, 0, 0]}, [[1, 0, 0]])
    assert failed == {"failed": True, "selected_fit": None, "error_deg": 90}
    assert valid["error_deg"] == 0
    summary = post.describe([failed, valid])
    assert summary["failures_at_90"] == {"n": 2, "mean_deg": 45, "median_deg": 45, "within20": 1}
    assert summary["failures_excluded"] == {"n": 1, "mean_deg": 0, "median_deg": 0, "within20": 1}
    assert post.describe([failed])["failures_excluded"]["mean_deg"] is None


def test_bootstrap_reproducible_paired_and_failure_exclusion():
    left = [{"error_deg": v, "failed": v == 90} for v in (0, 30, 90)]
    right = [{"error_deg": v, "failed": False} for v in (10, 20, 50)]
    result = post.paired_difference(left, right)
    assert result == post.paired_difference(left, right)
    assert result["mean_error_difference_deg"] == pytest.approx(40 / 3)
    assert result["within20_fraction_difference"] == pytest.approx(-1 / 3)
    assert post.paired_difference(left, right, exclude=True)["n"] == 2
    rng = np.random.default_rng(20261026)
    draws = rng.integers(3, size=(10_000, 3))
    expected = np.percentile(np.array([-10, 10, 40])[draws].mean(axis=1), [2.5, 97.5]).tolist()
    assert result["bootstrap_95"]["mean_error_difference_deg"] == expected
    assert post.paired_difference([left[-1]], [right[-1]], exclude=True)["bootstrap_95"] is None
    assert post.period_outcome(10.1, 10)["within1_percent"]
    assert not post.period_outcome(None, 10)["within1_percent"]


def scoring_inputs(args):
    rows = post.read(args.benchmark_dir / "evaluation-lock.json")["objects"]
    catalog = args.output_dir.parent / "references.jsonl"
    catalog.write_text("\n".join(json.dumps({"object_id": r["object_id"],
        "solutions": [{"period_hours": 10, "vector": [1, 0, 0]}]}) for r in rows))
    write(args.benchmark_dir / "scoring-lock.json", {
        "evaluation_lock": post.binding(args.benchmark_dir / "evaluation-lock.json"),
        "reference_catalog": post.binding(catalog)})
    for directory in (args.benchmark_dir, args.published_dir):
        fit = {"axis": [1, 0, 0], "final_relative_rms": 0.1}
        write(directory / "score-loaded.json", {"object_rows": [
            {"object_id": r["object_id"], "arms": {"delphi": fit}} for r in rows]})
        write(directory / "classical-ladder-exploratory.json", {"object_rows": [
            {"object_id": r["object_id"], "classical": {"12": fit, "146": fit}} for r in rows]})
    write(args.cost_summary, {"estimates": {rows[0]["object_id"]: {"E_core": {"cpu_seconds": {"mean": 123}}}}})
    return catalog


def test_score_full_synthetic_pipeline_and_resume(fitted):
    args = fitted
    scoring_inputs(args)
    assert post.command_score(args) == 0
    scores = post.read(args.output_dir / "scores.json")
    assert len(scores["objects"]) == 20
    assert scores["summary"]["refined"]["delphi6"]["failed"] == 19
    assert scores["summary"]["own"]["delphi6"]["failures_at_90"]["within20"] == 20
    first = scores["objects"][0]
    assert first["costs"]["refinement_P_core_cpu_seconds"] == pytest.approx(10)
    assert first["costs"]["inference_seconds"] == 0.25
    assert first["costs"]["solver_cpu_seconds"] == {"delphi6": 9, "classical12": 18, "classical146": 219}
    assert first["costs"]["all_152_classical_solver_cpu_seconds"] == 228
    assert first["costs"]["full_range_scan_estimate"]["E_core"]["cpu_seconds"]["mean"] == 123
    assert scores["bootstrap"] == {"unit": "paired asteroid", "resamples": 10000, "seed": 20261026}
    assert len(scores["paired_differences"]) == 15
    assert "exploratory development sample" in (args.output_dir / "RESULTS.md").read_text()
    paths = [args.output_dir / n for n in ("scores.json", "RESULTS.md")]
    before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths]
    assert post.command_score(args) == 0
    assert [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths] == before


def test_score_cannot_open_references_before_all_fit_receipts(fitted, monkeypatch):
    catalog = scoring_inputs(fitted)
    task = post.read(fitted.output_dir / "fits/fit-plan.json")["tasks"][-1]
    (post._start_path(task) / "timing.json").unlink()
    original = Path.open

    def guarded(path, *args, **kwargs):
        if path == catalog or path.name in ("score-loaded.json", "classical-ladder-exploratory.json"):
            pytest.fail("read references before every fit was audited")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    with pytest.raises(FileExistsError, match="incomplete start"):
        post.command_score(fitted)
    assert not (fitted.output_dir / "scores.json").exists()


def test_all_refinements_failed_still_reports_twenty_failures(inputs, monkeypatch):
    path = inputs.output_dir / "refined-periods.json"
    value = post.read(path)
    for row in value["objects"]:
        row.update(status="not_finished", period_found_hours=None, period_found_hours_raw=None)
    value["periods_hours"] = {}
    write(path, value)
    post.command_lock(inputs)
    monkeypatch.setattr(post, "require_gpu_lock", lambda: None)
    monkeypatch.setattr(post, "predict_one", lambda *a: pytest.fail("inference on failed refinement"))
    monkeypatch.setattr(post, "ProcessPoolExecutor", lambda *a, **k: pytest.fail("solver on failed refinement"))
    assert post.command_predict(inputs) == 0
    assert post.command_fits(inputs) == 0
    scoring_inputs(inputs)
    assert post.command_score(inputs) == 0
    scores = post.read(inputs.output_dir / "scores.json")
    for arm in post.ARMS:
        assert scores["summary"]["refined"][arm]["failed"] == 20
        assert scores["summary"]["refined"][arm]["failures_at_90"]["mean_deg"] == 90
        assert scores["summary"]["refined"][arm]["failures_excluded"]["n"] == 0
    assert scores["period_summary"]["refined_minus_own_relative_error"]["bootstrap_95"] is None


@pytest.mark.parametrize("spec, expected", [("12-23", tuple(range(12, 24))), ("23", (23,)), ("12,14-15", (12, 14, 15))])
def test_cpu_parser(spec, expected):
    assert post.parse_cpus(spec) == expected


@pytest.mark.parametrize("spec", ["", "23-12", "12,12", "a", "-1", "1-2-3"])
def test_invalid_cpus(spec):
    with pytest.raises(Exception, match="expected unique CPUs"):
        post.parse_cpus(spec)


def test_real_one_start_smoke_cpu23_nice19():
    from repro.own_period import period_cost as cost

    binary = cost.SOURCE / "convexinv"
    if not binary.is_file():
        pytest.skip("expanded convexinv executable absent")
    if 23 not in os.sched_getaffinity(0):
        pytest.skip("CPU 23 unavailable")
    with tempfile.TemporaryDirectory(prefix="refine-chain-post-smoke-", dir="/tmp") as temporary:
        root = Path(temporary)
        executable = root / "convexinv"
        shutil.copyfile(binary, executable)
        executable.chmod(0o700)
        assert post.binding(executable)["sha256"] == post.binding(binary)["sha256"]
        lc = root / "lightcurve.txt"
        lines = ["2"]
        for epoch in range(2):
            lines.append("80 0")
            for index in range(80):
                angle = epoch * 0.7 + index / 700
                lines.append(f"{2450000 + epoch * 1155 + index / 800:.12f} "
                             f"{1 + 0.12 * math.cos(2 * math.pi * index / 23):.12f} "
                             f"{math.cos(angle):.12f} {math.sin(angle):.12f} 0.2 "
                             f"{math.cos(angle + 0.2):.12f} {math.sin(angle + 0.2):.12f} 0.1")
        lc.write_text("\n".join(lines) + "\n")
        task = {"job": {"object_id": "smoke", "period_hours": 10, "lightcurve": post.binding(lc)},
                "arm": "standard6", "start_index": 0, "initial_axis": [1, 0, 0],
                "directory": str(root / "fits")}
        lock = {"solver": {"executable": {"path": str(executable)}},
                "solver_settings": {"convergence_tolerance": 3, "iteration_cap": 1000,
                                    "timeout_seconds_per_start": 60}}
        write(root / "task.json", task)
        write(root / "lock.json", lock)
        code = ("import json,sys,multiprocessing; from pathlib import Path; "
                "from concurrent.futures import ProcessPoolExecutor; "
                "from repro.own_period.refine_chain_post import _fit_start,_init_worker; "
                "p=Path(sys.argv[1]); "
                "ctx=multiprocessing.get_context('spawn'); q=ctx.Queue(); q.put(23); "
                "pool=ProcessPoolExecutor(max_workers=1,mp_context=ctx,initializer=_init_worker,initargs=(q,)); "
                "r=pool.submit(_fit_start,json.loads((p/'task.json').read_text()), "
                "json.loads((p/'lock.json').read_text())).result(); pool.shutdown(); q.close(); q.join_thread(); "
                "assert r['iterations'] == 3, r; assert r['solver_cpu_seconds'] > 0, r; "
                "assert r['selected_fit'] is not None, r")
        result = subprocess.run(["taskset", "-c", "23", "nice", "-n", str(max(0, 19 - os.nice(0))),
                                 sys.executable, "-c", code, str(root)],
                                env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
                                capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        receipt = post.read(root / "fits/start-000/timing.json")
        assert receipt["cpu_list"] == [23] and receipt["nice"] == 19
    assert not root.exists()
