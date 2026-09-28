"""Paired uncertainty for the post hoc classical-start ladder.

Scanning budgets on the same objects makes these intervals descriptive, not
independent confirmation of a selected budget.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "classical-ladder-exploratory.json"
OUTPUT = ROOT / "classical-ladder-intervals.json"
BOOTSTRAPS = 10_000
SEED = 20260924


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = source["object_rows"]
    folds = [np.array([i for i, row in enumerate(rows) if row["fold"] == fold]) for fold in range(5)]
    if any(len(group) != 28 for group in folds):
        raise ValueError("expected 28 objects in each of five folds")
    rng = np.random.default_rng(SEED)
    draws = np.concatenate([rng.choice(group, size=(BOOTSTRAPS, len(group))) for group in folds], axis=1)
    report = []
    for budget in source["budgets"]:
        count = str(budget["classical_starts"])
        if any(row["classical"][count] is None for row in rows):
            report.append({"classical_starts": int(count), "complete": False})
            continue
        error = np.array([
            row["classical"][count]["reference_error_degrees"] - row["delphi"]["reference_error_degrees"]
            for row in rows
        ])
        within = np.array([
            int(row["classical"][count]["reference_error_degrees"] <= 20)
            - int(row["delphi"]["reference_error_degrees"] <= 20)
            for row in rows
        ])
        log_rms = np.log(np.array([
            row["classical"][count]["final_relative_rms"] / row["delphi"]["final_relative_rms"]
            for row in rows
        ]))
        error_ci = np.percentile(error[draws].mean(axis=1), [2.5, 97.5]).tolist()
        within_ci = np.percentile(within[draws].mean(axis=1), [2.5, 97.5]).tolist()
        rms_ci = np.exp(np.percentile(log_rms[draws].mean(axis=1), [2.5, 97.5])).tolist()
        report.append({
            "classical_starts": int(count),
            "complete": True,
            "classical_minus_delphi_mean_error_95_interval_degrees": error_ci,
            "classical_minus_delphi_within20_95_interval_fraction": within_ci,
            "classical_over_delphi_rms_ratio_95_interval": rms_ci,
            "meets_original_quality_limits_descriptively": error_ci[1] <= 3 and within_ci[0] >= -0.05 and rms_ci[1] <= 1.01,
        })
    result = {
        "scope": "descriptive_post_hoc_scan_of_previously_opened_evaluation_objects",
        "bootstrap": {"unit": "asteroid_within_fold", "draws": BOOTSTRAPS, "seed": SEED},
        "original_quality_limits": {"mean_error_upper_degrees": 3, "within20_lower_fraction": -0.05, "rms_ratio_upper": 1.01},
        "budgets": report,
    }
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
