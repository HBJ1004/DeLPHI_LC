"""Verify and summarize the post hoc loaded timing comparison."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "exploratory-loaded-ladder"
OUTPUT = RUN / "score.json"
ARMS = ("delphi", "classical18", "classical96")
BOOTSTRAPS = 10_000
SEED = 20260924


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def angle_degrees(first: list[float], second: list[float]) -> float:
    cosine = abs(sum(a * b for a, b in zip(first, second)))
    return math.degrees(math.acos(min(1.0, max(0.0, cosine))))


def main() -> None:
    execution = read(RUN / "execution.json")
    if execution["phase"] != "exploratory_loaded_ladder_complete" or execution["references_opened"] is not False:
        raise ValueError("invalid or incomplete execution")
    if execution["script_sha256"] != digest(ROOT / "time_exploratory_budgets.py"):
        raise ValueError("timing script changed")
    if execution["lock_sha256"] != digest(ROOT / "evaluation-lock.json"):
        raise ValueError("evaluation lock changed")
    saved = read(ROOT / "classical-ladder-exploratory.json")
    saved_by_object = {row["object_id"]: row for row in saved["object_rows"]}
    timed = {}
    for binding in execution["rows"]:
        path = RUN / binding["path"]
        if digest(path) != binding["sha256"]:
            raise ValueError(f"timed case changed: {path}")
        row = read(path)
        key = (row["object_id"], row["arm"])
        if key in timed or row["repeat_index"] != 0 or row["arm"] not in ARMS:
            raise ValueError(f"duplicate or invalid case: {key}")
        if not row["completed"] or row["selected_fit"] is None:
            raise ValueError(f"incomplete fit: {key}")
        timed[key] = row
    if len(timed) != 140 * len(ARMS):
        raise ValueError("not all 420 timed cases are present")

    max_axis = 0.0
    max_rms = 0.0
    rows = []
    for object_id, saved_row in saved_by_object.items():
        arms = {}
        for arm in ARMS:
            row = timed[(object_id, arm)]
            reference = saved_row["delphi"] if arm == "delphi" else saved_row["classical"][arm.removeprefix("classical")]
            fit = row["selected_fit"]
            max_axis = max(max_axis, angle_degrees(fit["axis"], reference["axis"]))
            max_rms = max(max_rms, abs(fit["final_relative_rms"] - reference["final_relative_rms"]))
            arms[arm] = {
                "wall_seconds": row["wall_seconds"],
                "solver_wall_seconds": row["solver_wall_seconds"],
                "reference_error_degrees": reference["reference_error_degrees"],
                "final_relative_rms": fit["final_relative_rms"],
            }
        rows.append({"object_id": object_id, "fold": saved_row["fold"], "arms": arms})
    if max_axis > 0.0001 or max_rms > 1e-8:
        raise ValueError("timed fits do not reproduce archived solutions")

    folds = [np.array([i for i, row in enumerate(rows) if row["fold"] == fold]) for fold in range(5)]
    if any(len(group) != 28 for group in folds):
        raise ValueError("fold membership changed")
    rng = np.random.default_rng(SEED)
    draws = np.concatenate([rng.choice(group, size=(BOOTSTRAPS, len(group))) for group in folds], axis=1)
    times = {arm: np.array([row["arms"][arm]["wall_seconds"] for row in rows]) for arm in ARMS}
    summaries = []
    for arm in ARMS:
        values = times[arm]
        summaries.append({
            "arm": arm,
            "mean_wall_seconds": float(values.mean()),
            "median_wall_seconds": float(np.median(values)),
            "mean_solver_wall_seconds": float(np.mean([row["arms"][arm]["solver_wall_seconds"] for row in rows])),
        })
    ratios = []
    for arm in ("classical18", "classical96"):
        point = float(times[arm].mean() / times["delphi"].mean())
        sampled = times[arm][draws].mean(axis=1) / times["delphi"][draws].mean(axis=1)
        ratios.append({
            "classical_arm": arm,
            "classical_over_delphi_ratio_of_means": point,
            "bootstrap_95_interval": np.percentile(sampled, [2.5, 97.5]).tolist(),
            "objects_faster_with_delphi": int(np.sum(times["delphi"] < times[arm])),
        })
    setup = sum(item["setup_seconds"] for item in execution["model_setup_seconds_by_fold"])
    result = {
        "scope": "post_hoc_one_repeat_loaded_timing_on_previously_opened_evaluation_objects",
        "execution_sha256": digest(RUN / "execution.json"),
        "saved_fit_sha256": digest(ROOT / "classical-ladder-exploratory.json"),
        "bootstrap": {"unit": "asteroid_within_fold", "draws": BOOTSTRAPS, "seed": SEED},
        "validation": {"cases": len(timed), "max_axis_difference_degrees": max_axis, "max_rms_difference": max_rms},
        "model_setup_seconds_total": setup,
        "model_setup_seconds_per_asteroid": setup / len(rows),
        "arm_summaries": summaries,
        "time_ratios": ratios,
        "object_rows": rows,
    }
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "object_rows"}, indent=2))


if __name__ == "__main__":
    main()
