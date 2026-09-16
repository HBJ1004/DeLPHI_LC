import json
from pathlib import Path

import numpy as np
import pytest

from lc_pipeline import grid_benchmark
from lc_pipeline.k3 import reliability_study as study
from lc_pipeline.k3.reliability_sampling import (
    CurveObservation,
    LightCurve,
    NativeSession,
    write_lc,
)
from lc_pipeline.k3.tokenizer import tokenize_epochs


def _observation(time_jd: float, flux: float = 1.0) -> CurveObservation:
    return CurveObservation(time_jd, flux, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def _curve(*, withheld_scale: float = 1.0) -> LightCurve:
    sessions = []
    for session_index in range(6):
        scale = withheld_scale if session_index == 5 else 1.0
        observations = tuple(
            _observation(
                2_450_000.0 + session_index * 100 + point_index / 50,
                scale * (1.0 + point_index / 100),
            )
            for point_index in range(40)
        )
        sessions.append(
            NativeSession(
                f"session-{session_index}",
                observations,
                calibrated=session_index % 2,
                source_index=session_index,
            )
        )
    return LightCurve(tuple(sessions), "asteroid-test", 10.0)


def _bound_curve(path: Path, curve: LightCurve) -> dict:
    write_lc(curve, path)
    return {
        "object_id": curve.object_id,
        "fold": 0,
        "period_hours": curve.period_hours,
        "lightcurve": study.binding(path),
    }


def test_budget_deadline_and_elapsed_time_survive_resume(tmp_path: Path, monkeypatch):
    clock = {"now": 1_000.0}
    monkeypatch.setattr(study.time, "time", lambda: clock["now"])

    initial = study.Budget(tmp_path, "inference", 100)
    deadline = initial.state["deadline_epoch"]
    clock["now"] = 1_040.0
    resumed = study.Budget(tmp_path, "inference", 100)

    assert resumed.state["start_epoch"] == 1_000.0
    assert resumed.state["deadline_epoch"] == deadline
    assert resumed.remaining() == pytest.approx(60.0)
    with pytest.raises(ValueError, match="cannot change"):
        study.Budget(tmp_path, "inference", 101)
    clock["now"] = deadline
    with pytest.raises(TimeoutError, match="budget exhausted"):
        resumed.check()


def test_phase_budget_is_also_bounded_by_persisted_global_deadline(tmp_path: Path, monkeypatch):
    clock = {"now": 2_000.0}
    monkeypatch.setattr(study.time, "time", lambda: clock["now"])
    total = study.Budget(tmp_path, "total", 100)
    clock["now"] = 2_001.0
    phase = study.Budget(tmp_path, "inference", 1_000)

    assert phase.state["deadline_epoch"] == 3_001.0
    assert total.state["deadline_epoch"] == 2_100.0
    clock["now"] = 2_050.0
    assert phase.remaining() == pytest.approx(50.0)
    clock["now"] = 2_100.0
    with pytest.raises(TimeoutError, match="inference computation budget exhausted"):
        phase.check()


def test_sealed_lock_detects_payload_or_checksum_mutation(tmp_path: Path):
    target = tmp_path / "study.json"
    payload = {"schema": study.SCHEMA, "objects": ["a", "b"]}
    study.seal(target, payload)
    assert study.unseal(target) == payload

    envelope = json.loads(target.read_text(encoding="utf-8"))
    envelope["payload"]["objects"].append("c")
    target.write_text(study.canonical(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum changed"):
        study.unseal(target)


def test_native_start_quarantines_interrupted_directory_before_restart(tmp_path: Path, monkeypatch):
    directory = tmp_path / "fit-cell"
    partial = directory / "start-003"
    partial.mkdir(parents=True)
    (partial / "partial-output.txt").write_text("interrupted", encoding="utf-8")
    calls = []

    def fake_native_start(job, index, axis, observed, solver, settings, output):
        calls.append((job, index, axis, observed, solver, settings, output))
        assert not partial.exists()
        return {"start_index": index, "selected_fit": None}

    class RecordingBudget:
        def __init__(self):
            self.reserves = []

        def check(self, reserve=0):
            self.reserves.append(reserve)

    monkeypatch.setattr(grid_benchmark, "_native_start", fake_native_start)
    budget = RecordingBudget()
    lock = {
        "solver": {"executable": "fixture"},
        "solver_settings": {"timeout_seconds_per_start": 17},
    }
    result = study.native_start_resumable(
        {"period_hours": 10.0},
        3,
        (1.0, 0.0, 0.0),
        (np.asarray([1.0, 1.1]),),
        lock,
        directory,
        budget,
    )

    assert result["start_index"] == 3
    assert budget.reserves == [22]
    quarantined = list((directory / "interrupted").glob("start-003-*"))
    assert len(quarantined) == 1
    assert (quarantined[0] / "partial-output.txt").read_text() == "interrupted"
    assert len(calls) == 1


class _NoScientificScoreRow(dict):
    def __getitem__(self, key):
        if key in {"error_deg", "axes", "reference", "withheld_normalized_curve_rms"}:
            raise AssertionError(f"schedule inspected prohibited scientific field {key}")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in {"error_deg", "axes", "reference", "withheld_normalized_curve_rms"}:
            raise AssertionError(f"schedule inspected prohibited scientific field {key}")
        return super().get(key, default)


def test_choose_schedule_uses_runtime_projection_and_never_scientific_scores():
    pilot_inference = {f"pilot-{index}": 10.0 for index in range(10)}
    pilot_fits = {f"pilot-{index}": 30.0 for index in range(10)}
    rows = [
        _NoScientificScoreRow(
            object_id=f"asteroid-{index}",
            fold=index % 5,
            merged_blocks=10 if index < 80 else 9,
        )
        for index in range(170)
    ]
    eligible = [row["object_id"] for row in rows[:157]]
    projection = {"3": 35_999.0, "1": 12_000.0}

    chosen = study.choose_schedule(
        pilot_inference,
        rows,
        pilot_fits,
        eligible,
        inference_projection_by_repeat=projection,
    )

    assert chosen["repeats"] == 3
    assert chosen["inference_estimate_seconds"] == 35_999.0
    assert chosen["holdout_count"] == 157
    assert chosen["selection_uses"] == "technical runtime only; no reference or withheld scores"

    one_repeat = study.choose_schedule(
        pilot_inference,
        rows,
        {key: 200.0 for key in pilot_fits},
        eligible,
        inference_projection_by_repeat={"3": 36_001.0, "1": 35_999.0},
    )
    assert one_repeat["repeats"] == 1
    # Twice the 200-second worst six-cell pilot time admits 75, but not 100.
    assert one_repeat["holdout_count"] == 75

    with pytest.raises(ValueError, match="infeasible"):
        study.choose_schedule(
            pilot_inference,
            rows,
            pilot_fits,
            eligible,
            inference_projection_by_repeat={"3": 36_001.0, "1": 36_001.0},
        )


def test_inference_projection_counts_every_planned_cell_and_uses_global_fallback(
    tmp_path: Path,
):
    for object_id, condition, runtime in (
        ("pilot-a", "full", 1.0),
        ("pilot-a", "thin", 2.0),
        ("pilot-b", "full", 0.5),
        ("pilot-b", "thin", 1.5),
    ):
        study.seal(
            tmp_path / "pilot" / "predictions" / object_id / condition / "repeat-0.json",
            {"condition": condition, "status": "ok", "runtime_s": runtime},
        )
    lock = {
        "objects": [{"object_id": f"asteroid-{index}"} for index in range(170)],
        "grid_ids": [f"asteroid-{index}" for index in range(80)],
        "conditions": [
            {"condition": "full", "family": "full", "repeatable": False},
            {"condition": "thin", "family": "point", "repeatable": True},
            # Deliberately absent from the pilot: it must use the global maximum,
            # not disappear from the full-cohort projection.
            {"condition": "grid", "family": "two_d", "repeatable": True},
        ],
    }

    result = study.project_inference(tmp_path, lock)

    assert result["condition_max_seconds"] == {"full": 1.0, "thin": 2.0}
    assert result["unmeasured_condition_fallback_seconds"] == 2.0
    # 2 * [170*1 + 170*3*2 + 80*3*2] + 300 seconds overhead.
    assert result["estimates_seconds"]["3"] == pytest.approx(3_640.0)
    # 2 * [170*1 + 170*1*2 + 80*1*2] + 300 seconds overhead.
    assert result["estimates_seconds"]["1"] == pytest.approx(1_640.0)


def test_inference_projection_rejects_any_failed_pilot_cell(tmp_path: Path):
    study.seal(
        tmp_path / "pilot" / "predictions" / "pilot-a" / "full" / "repeat-0.json",
        {"condition": "full", "status": "failed", "runtime_s": 1.0},
    )
    lock = {
        "objects": [{"object_id": "a"}],
        "grid_ids": [],
        "conditions": [{"condition": "full", "family": "full", "repeatable": False}],
    }
    with pytest.raises(ValueError, match="pilot contains a failure"):
        study.project_inference(tmp_path, lock)


def test_balanced_order_is_stable_and_fold_interleaved():
    rows = [
        {"object_id": f"f{fold}-object-{index}", "fold": fold}
        for fold in range(5)
        for index in range(4)
    ]
    one = study.balanced_order(rows)
    two = study.balanced_order(list(reversed(rows)))

    assert one == two
    assert len(one) == len(set(one)) == 20
    assert [next(row["fold"] for row in rows if row["object_id"] == oid) for oid in one] == [
        0,
        1,
        2,
        3,
        4,
    ] * 4


def test_period_condition_retokenizes_every_period_dependent_feature(tmp_path: Path):
    source = tmp_path / "lc.txt"
    row = _bound_curve(source, _curve())
    full = {
        "condition": "full",
        "family": "full",
        "sampling": {"kind": "full"},
        "period_multiplier": 1.0,
    }
    shifted = {**full, "condition": "period-1.01", "period_multiplier": 1.01}
    curves = [study.selected_curve(row, config, 0) for config in (full, shifted)]

    class TokenizingPredictor:
        def __init__(self):
            self.tokens = []

        def predict(self, epochs, *, known_period, object_id):
            self.tokens.append(tokenize_epochs(epochs, known_period=known_period))
            return {
                "axes": [
                    {"axis_xyz": [1.0, 0.0, 0.0]},
                    {"axis_xyz": [0.0, 1.0, 0.0]},
                    {"axis_xyz": [0.0, 0.0, 1.0]},
                ]
            }

    predictor = TokenizingPredictor()
    for curve in curves:
        study.predict_curve(predictor, curve, row["object_id"])

    original, perturbed = predictor.tokens
    assert original.period_hours == 10.0
    assert perturbed.period_hours == pytest.approx(10.1)
    assert not np.array_equal(original.phase_mask, perturbed.phase_mask)
    assert not np.array_equal(original.phase_features, perturbed.phase_features)
    assert not np.array_equal(original.epoch_features, perturbed.epoch_features)
    assert study.input_hash(curves[0]) != study.input_hash(curves[1])


@pytest.mark.parametrize("sampling", ["full", "thin10"])
def test_withheld_flux_mutation_cannot_change_fit_input(tmp_path: Path, sampling: str):
    row_one = _bound_curve(tmp_path / "one.txt", _curve(withheld_scale=1.0))
    row_two = _bound_curve(tmp_path / "two.txt", _curve(withheld_scale=7.0))

    retained_one, path_one = study.holdout_input(tmp_path / "run-one", row_one, sampling)
    retained_two, path_two = study.holdout_input(tmp_path / "run-two", row_two, sampling)

    assert study.input_hash(retained_one) == study.input_hash(retained_two)
    assert path_one.read_bytes() == path_two.read_bytes()
    assert [item.flux for item in retained_one.observations] == [
        item.flux for item in retained_two.observations
    ]
