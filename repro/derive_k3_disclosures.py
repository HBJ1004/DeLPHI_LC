"""Derive manuscript disclosure statistics from the frozen K3 evidence.

This command does not train a model or rerun inversion. It exposes quantities
already present in the released per-object artifacts so that revisions remain
traceable to data rather than hand-transcribed calculations.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from lc_pipeline.k3.bundle import sha256
from lc_pipeline.k3.evaluation import paired_bootstrap_ci, summarize_errors

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260901


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _geometric_ratio_with_upper_ci(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float]:
    if (
        numerator.shape != denominator.shape
        or numerator.ndim != 1
        or numerator.size < 2
        or np.any(~np.isfinite(numerator))
        or np.any(~np.isfinite(denominator))
        or np.any(numerator <= 0)
        or np.any(denominator <= 0)
    ):
        raise ValueError("RMS vectors must be aligned, finite, positive, and nonempty")
    ratios = numerator / denominator
    point = float(np.exp(np.mean(np.log(ratios))))
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, ratios.size, size=(resamples, ratios.size))
    samples = np.exp(np.mean(np.log(ratios[indexes]), axis=1))
    return point, float(np.quantile(samples, 0.975))


def derive_disclosures(
    ensemble_path: Path, controls_path: Path, downstream_rows_path: Path
) -> dict[str, Any]:
    """Return publication disclosures after enforcing cross-artifact alignment."""
    with np.load(ensemble_path, allow_pickle=False) as ensemble:
        object_ids = ensemble["object_ids"].astype(str)
        refined = ensemble["oracle_errors_deg"].astype(np.float64)
        grid = ensemble["grid_oracle_errors_deg"].astype(np.float64)
        if str(ensemble["schema"]) != "delphi.k3-real-oof-ensemble.v1":
            raise ValueError("ensemble schema mismatch")
    with np.load(controls_path, allow_pickle=False) as controls:
        if str(controls["schema"]) != "delphi.k3-v1-comparators.v1":
            raise ValueError("control archive schema mismatch")
        if not np.array_equal(object_ids, controls["object_ids"].astype(str)):
            raise ValueError("ensemble and control object order differs")
        atlas = controls["atlas_errors_deg"].astype(np.float64)
        deranged = controls["deranged_errors_deg"].astype(np.float64)
    downstream = json.loads(downstream_rows_path.read_text())
    if downstream.get("schema") != "delphi.k3-fixed-period-rows.v1":
        raise ValueError("downstream schema mismatch")
    if downstream.get("object_ids") != object_ids.tolist():
        raise ValueError("ensemble and downstream object order differs")
    rows = downstream.get("rows")
    if not isinstance(rows, list) or len(rows) != len(object_ids):
        raise ValueError("downstream rows must contain one row per object")

    controls_output: dict[str, Any] = {}
    for name, values in (("atlas", atlas), ("derangement", deranged)):
        improvement = values - refined
        lower, upper = paired_bootstrap_ci(
            improvement, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED
        )
        controls_output[name] = {
            "mean_improvement_deg": float(np.mean(improvement)),
            "bootstrap_ci95_deg": [lower, upper],
        }

    baseline_wall = np.asarray(
        [_finite_number(row["baseline"]["wall_seconds"], "baseline wall time") for row in rows]
    )
    candidate_wall = np.asarray(
        [_finite_number(row["candidate"]["wall_seconds"], "candidate wall time") for row in rows]
    )
    neural_wall = np.asarray(
        [_finite_number(row["neural_inference_wall_seconds"], "neural wall time") for row in rows]
    )
    baseline_iterations = np.asarray([int(row["baseline"]["iterations"]) for row in rows])
    candidate_iterations = np.asarray([int(row["candidate"]["iterations"]) for row in rows])
    baseline_starts = np.asarray([int(row["baseline"]["valid_starts"]) for row in rows])
    candidate_starts = np.asarray([int(row["candidate"]["valid_starts"]) for row in rows])
    if (
        np.any(baseline_wall <= 0)
        or np.any(candidate_wall <= neural_wall)
        or not np.array_equal(baseline_iterations, candidate_iterations)
        or not np.array_equal(baseline_starts, candidate_starts)
    ):
        raise ValueError("fixed-work downstream invariants are violated")

    baseline_success = np.asarray([row["baseline"]["success"] for row in rows], dtype=bool)
    candidate_success = np.asarray([row["candidate"]["success"] for row in rows], dtype=bool)
    baseline_rms = np.asarray(
        [float(row["baseline"]["final_rms"]) if row["baseline"]["final_rms"] is not None else np.nan for row in rows]
    )
    candidate_rms = np.asarray(
        [float(row["candidate"]["final_rms"]) if row["candidate"]["final_rms"] is not None else np.nan for row in rows]
    )
    baseline_pole_error = np.asarray(
        [float(row["baseline"]["pole_error_deg"]) if row["baseline"]["pole_error_deg"] is not None else np.nan for row in rows]
    )
    candidate_pole_error = np.asarray(
        [float(row["candidate"]["pole_error_deg"]) if row["candidate"]["pole_error_deg"] is not None else np.nan for row in rows]
    )
    baseline_completed = np.isfinite(baseline_rms)
    candidate_completed = np.isfinite(candidate_rms)
    recovered_both = baseline_success & candidate_success
    completed_both = baseline_completed & candidate_completed
    recovered_ratio, recovered_upper = _geometric_ratio_with_upper_ci(
        candidate_rms[recovered_both], baseline_rms[recovered_both]
    )
    completed_ratio, completed_upper = _geometric_ratio_with_upper_ci(
        candidate_rms[completed_both], baseline_rms[completed_both]
    )
    wall_delta = float(np.sum(candidate_wall) - np.sum(baseline_wall))
    neural_total = float(np.sum(neural_wall))

    grid_summary = summarize_errors(grid)
    refined_summary = summarize_errors(refined)
    return {
        "schema": "delphi.k3-publication-disclosures.v1",
        "statistical_unit": "asteroid",
        "n_objects": int(len(object_ids)),
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "source_sha256": {
            "ensemble": sha256(ensemble_path),
            "controls": sha256(controls_path),
            "downstream_rows": sha256(downstream_rows_path),
        },
        "nonlearned_control_contrasts": controls_output,
        "downstream": {
            "iterations_per_arm": int(np.sum(baseline_iterations)),
            "completed_starts_per_arm": int(np.sum(baseline_starts)),
            "inversion_only_wall_time_ratio": float(
                np.sum(baseline_wall) / np.sum(candidate_wall - neural_wall)
            ),
            "wall_time_delta_seconds": wall_delta,
            "neural_wall_seconds": neural_total,
            "neural_fraction_of_wall_time_delta": neural_total / wall_delta,
            "baseline_recovered": int(np.sum(baseline_success)),
            "candidate_recovered": int(np.sum(candidate_success)),
            "baseline_recovery_rate": float(np.mean(baseline_success)),
            "candidate_recovery_rate": float(np.mean(candidate_success)),
            "baseline_completed": int(np.sum(baseline_completed)),
            "candidate_completed": int(np.sum(candidate_completed)),
            "recovered_in_both": {
                "n_objects": int(np.sum(recovered_both)),
                "geometric_mean_rms_ratio": recovered_ratio,
                "bootstrap_ci95_upper": recovered_upper,
            },
            "completed_in_both": {
                "n_objects": int(np.sum(completed_both)),
                "geometric_mean_rms_ratio": completed_ratio,
                "bootstrap_ci95_upper": completed_upper,
            },
            "median_pole_error_completed_deg": {
                "baseline": float(np.median(baseline_pole_error[baseline_completed])),
                "candidate": float(np.median(candidate_pole_error[candidate_completed])),
            },
        },
        "refinement": {
            "grid": grid_summary,
            "refined": refined_summary,
            "mean_change_refined_minus_grid_deg": float(np.mean(refined - grid)),
            "objects_improved": int(np.sum(refined < grid)),
            "objects_worsened": int(np.sum(refined > grid)),
            "objects_unchanged": int(np.sum(refined == grid)),
        },
    }


def render_tex(values: dict[str, Any]) -> str:
    """Render the revision quantities as LaTeX macros with stable rounding."""
    controls = values["nonlearned_control_contrasts"]
    downstream = values["downstream"]
    refinement = values["refinement"]
    rows = ["% Generated by repro.derive_k3_disclosures; do not edit by hand."]

    def add(name: str, value: str) -> None:
        rows.append(f"\\providecommand{{\\{name}}}{{{value}}}")

    for title, key in (("Atlas", "atlas"), ("Deranged", "derangement")):
        entry = controls[key]
        add(f"KThree{title}Improvement", f"{entry['mean_improvement_deg']:.2f}")
        add(f"KThree{title}ImprovementLow", f"{entry['bootstrap_ci95_deg'][0]:.2f}")
        add(f"KThree{title}ImprovementHigh", f"{entry['bootstrap_ci95_deg'][1]:.2f}")
    add("KThreeIterationsPerArm", f"{downstream['iterations_per_arm']:,}")
    add("KThreeCompletedStartsPerArm", f"{downstream['completed_starts_per_arm']:,}")
    add("KThreeInversionOnlyRatio", f"{downstream['inversion_only_wall_time_ratio']:.3f}")
    add("KThreeNeuralWallSeconds", f"{downstream['neural_wall_seconds']:.1f}")
    add("KThreeWallDeltaSeconds", f"{downstream['wall_time_delta_seconds']:.1f}")
    add("KThreeNeuralDeltaPercent", f"{100 * downstream['neural_fraction_of_wall_time_delta']:.1f}")
    add("KThreeBaselineRecoveredCount", str(downstream["baseline_recovered"]))
    add("KThreeCandidateRecoveredCount", str(downstream["candidate_recovered"]))
    add("KThreeBaselineRecoveryPercent", f"{100 * downstream['baseline_recovery_rate']:.1f}")
    add("KThreeCandidateRecoveryPercent", f"{100 * downstream['candidate_recovery_rate']:.1f}")
    add("KThreeRecoveredBothCount", str(downstream["recovered_in_both"]["n_objects"]))
    add("KThreeCompletedBothCount", str(downstream["completed_in_both"]["n_objects"]))
    add("KThreeCompletedBothRmsRatio", f"{downstream['completed_in_both']['geometric_mean_rms_ratio']:.3f}")
    add("KThreeCompletedBothRmsUpper", f"{downstream['completed_in_both']['bootstrap_ci95_upper']:.3f}")
    add("KThreeBaselineMedianPoleError", f"{downstream['median_pole_error_completed_deg']['baseline']:.2f}")
    add("KThreeCandidateMedianPoleError", f"{downstream['median_pole_error_completed_deg']['candidate']:.2f}")
    add("KThreeGridOracleMean", f"{refinement['grid']['mean_error_deg']:.2f}")
    add("KThreeRefinedOracleMean", f"{refinement['refined']['mean_error_deg']:.2f}")
    add("KThreeRefinementWorseCount", str(refinement["objects_worsened"]))
    add("KThreeRefinementBetterCount", str(refinement["objects_improved"]))
    return "\n".join(rows) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--downstream-rows", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--tex-output", type=Path)
    args = parser.parse_args(argv)
    if args.json_output.exists() or (args.tex_output is not None and args.tex_output.exists()):
        parser.error("output already exists; choose new output paths")
    result = derive_disclosures(args.ensemble, args.controls, args.downstream_rows)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.tex_output is not None:
        args.tex_output.write_text(render_tex(result))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
