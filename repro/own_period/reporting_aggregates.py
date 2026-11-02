"""Reproducible paired gains and refinement medians for the paper (CPU only)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from repro import network_root as network
from repro.own_period import DataPath as FixedDataPath
from repro.own_period.median_stats import Evidence, bootstrap_indices, compute_statistics
from repro.own_period.reporting_inputs import DataPath, load_npz, read_json, select

SEED_NAMES = dict(
    zip(
        network.SEEDS,
        ("Seventeen", "FortyTwo", "OneThirtySeven", "SevenSeventySeven", "TwentyTwentySeven"),
    )
)
BUDGET_NAMES = {6: "Six", 12: "Twelve", 18: "Eighteen", 36: "Thirtysix", 48: "Fortyeight"}


def paired_gains():
    evidence = Evidence()
    ensemble = evidence.ensemble
    ids = ensemble["object_ids"].astype(str).tolist()
    own = ensemble["oracle_errors_deg"]
    result, sources = {}, []
    for seed, name in SEED_NAMES.items():
        lookup = {}
        for fold in range(5):
            path = DataPath(f"maps/own-period/evaluations/real-fold-{fold}-seed-{seed}.npz")
            with load_npz(path, allow_pickle=False) as values:
                lookup.update(zip(values["object_ids"].astype(str), values["oracle_errors_deg"]))
            sources.append(path)
        if set(lookup) != set(ids):
            raise ValueError("single-seed/ensemble identities differ")
        result["TOneSeed" + name + "Gain"] = compute_statistics(
            [lookup[oid] for oid in ids], own, indices=bootstrap_indices(len(ids), 20260930)
        )
    # Fold summaries and their paired standard-minus-network gains use exactly
    # the ensemble order inside each held-out fold.
    baseline = dict(
        zip(evidence.baseline["object_ids"].astype(str), evidence.baseline["standard_errors_deg"])
    )
    folds = {}
    for fold in range(5):
        mask = ensemble["folds"] == fold
        fold_ids = np.asarray(ids)[mask].tolist()
        folds[str(fold)] = dict(
            object_ids=fold_ids,
            **compute_statistics(
                [baseline[oid] for oid in fold_ids],
                own[mask],
                indices=bootstrap_indices(len(fold_ids), 20260930),
            ),
        )
    path = DataPath("../direct-workflow-benchmark/classical-ladder-exploratory.json")
    ladder = read_json(path)["object_rows"]
    sources.append(path)
    for budget, name in BUDGET_NAMES.items():
        result["TOneDamit" + name] = compute_statistics(
            [r["classical"][str(budget)]["reference_error_degrees"] for r in ladder],
            [r["delphi"]["reference_error_degrees"] for r in ladder],
            indices=bootstrap_indices(
                len(ladder), 20260924, folds=[r["fold"] for r in ladder], order="numpy_fold_major"
            ),
        )
    path = DataPath("failed-excluded-derived/reliability-paired.json")
    sources.append(path)
    rows = read_json(path)["conditions"]["observation_cap-cap-50"]["objects"]
    result["TOneFifty"] = compute_statistics(
        [r["standard_error_deg"] for r in rows],
        [r["delphi_mean_deg"] for r in rows],
        indices=bootstrap_indices(len(rows), 20260930),
    )
    reverse = evidence.pair_from_score("FollowAlcdefAtlasDifference")
    result["TOneAlcdefFitted"] = dict(
        reverse,
        mean_recomputed=-reverse["mean_recomputed"],
        median=-reverse["median"],
        interval=[-v for v in reversed(reverse["interval"])],
        mean_interval_recomputed=[-v for v in reversed(reverse["mean_interval_recomputed"])],
    )
    sources.extend(reverse["source_files"])
    sources.append(DataPath("maps/own-period/evaluations/real-oof-ensemble.npz"))
    return dict(
        schema="delphi.paper1-paired-gains.v1",
        **network.json_metadata(),
        groups=result,
        fold_standard_minus_delphi=folds,
        object_order=ids,
        sources={str(select(p)): network.sha256(select(p)) for p in sources},
    )


def describe(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if not a.size:
        return dict(n=0, median=None, minimum=None, maximum=None, mean=None, total=0.0)
    return dict(
        n=len(a),
        median=float(np.median(a)),
        minimum=float(a.min()),
        maximum=float(a.max()),
        mean=float(a.mean()),
        total=float(a.sum()),
    )


def refine_medians(source):
    scores = read_json(source)
    rows = scores["objects"]
    seed = scores["bootstrap"]["seed"]
    count = scores["bootstrap"]["resamples"]
    if (seed, count) != (20261026, 10000):
        raise ValueError("refinement bootstrap settings changed")
    summary = {
        s: {
            a: dict(
                n=len(rows),
                failed=sum(r["scenarios"][s][a]["failed"] for r in rows),
                median_error_deg=float(
                    np.median([r["scenarios"][s][a]["error_deg"] for r in rows])
                ),
                within20=sum(r["scenarios"][s][a]["error_deg"] <= 20 for r in rows),
            )
            for a in ("delphi6", "classical12", "classical146")
        }
        for s in ("refined", "own", "published")
    }
    means, medians = {}, {}
    for label, stored in scores["paired_differences"].items():
        left, right = [part.split("/") for part in label.split(" minus ")]
        mean_checks = {}
        for mode in ("failures_at_90", "failures_excluded"):
            paired = [
                (r["scenarios"][left[0]][left[1]], r["scenarios"][right[0]][right[1]]) for r in rows
            ]
            if mode == "failures_excluded":
                paired = [(a, b) for a, b in paired if not a["failed"] and not b["failed"]]
            a, b = [np.array([p[i]["error_deg"] for p in paired]) for i in (0, 1)]
            if not paired:
                raise ValueError("refinement aggregation has no paired successes")
            draws = bootstrap_indices(len(a), seed, count)
            reproduced = np.percentile((a - b)[draws].mean(axis=1), [2.5, 97.5]).tolist()
            expected = stored[mode]["bootstrap_95"]["mean_error_difference_deg"]
            exact = reproduced == expected
            mean_checks[mode] = dict(stored=expected, reproduced=reproduced, exact_match=exact)
            if not exact:
                raise ValueError(f"refinement mean interval mismatch: {label}/{mode}")
            if mode == "failures_at_90":
                interval = np.percentile(
                    np.median(a[draws], axis=1) - np.median(b[draws], axis=1), [2.5, 97.5]
                ).tolist()
                medians[label] = dict(
                    n=len(a),
                    difference_of_medians_deg=float(np.median(a) - np.median(b)),
                    percentile_95_deg=interval,
                    stored_median_interval_exact_match=interval
                    == stored[mode]["bootstrap_95"]["median_error_difference_deg"],
                )
        means[label] = mean_checks
    keep = (
        "refined/delphi6 minus own/delphi6",
        "refined/delphi6 minus published/delphi6",
        "refined/classical12 minus own/classical12",
        "refined/classical12 minus published/classical12",
        "refined/classical12 minus refined/delphi6",
        "refined/classical146 minus refined/delphi6",
    )
    medians = {k: medians[k] for k in keep}
    per_object = [
        dict(
            object_id=r["object_id"],
            errors_deg={
                s: {a: v["error_deg"] for a, v in arms.items()}
                for s, arms in r["scenarios"].items()
            },
            refinement_P_core_CPU_hours=r["costs"]["refinement_P_core_cpu_seconds"] / 3600,
            inference_seconds=r["costs"]["inference_seconds"],
            solver_CPU_seconds=r["costs"]["solver_cpu_seconds"],
        )
        for r in rows
    ]
    outside = []
    for row in rows:
        period, ref = row["periods"]["refined"]["period_hours"], row["periods"]["DAMIT_hours"]
        if period is not None and abs(period / ref - 1) > 0.01:
            outside.append(
                dict(
                    object_id=row["object_id"],
                    refined_period_hours=period,
                    DAMIT_period_hours=ref,
                    ratio_to_DAMIT=period / ref,
                    relative_error=abs(period / ref - 1),
                )
            )
    return dict(
        source=dict(path=str(select(source)), sha256=network.sha256(select(source))),
        **network.json_metadata(),
        scope=scores["scope"],
        n=len(rows),
        object_order=[r["object_id"] for r in rows],
        bootstrap=dict(
            confidence_percent=95,
            seed=seed,
            resamples=count,
            unit="paired asteroid",
            object_order="scores.json objects order",
            generator="numpy.random.default_rng",
            sampling="integers(0, n, size=(10000, n))",
            interval="numpy.percentile at 2.5 and 97.5, default linear method",
            statistic="median(lhs resample) minus median(rhs resample), not median of paired differences",
        ),
        mean_interval_reproduction=dict(
            passed=True,
            comparison_count=len(means),
            failure_mode_count=2,
            exact_float_equality=True,
            checked_before_computing_medians=True,
            comparisons=means,
        ),
        summary=summary,
        paired_median_differences=medians,
        costs=dict(
            refinement_P_core_CPU_hours=describe(
                [r["refinement_P_core_CPU_hours"] for r in per_object]
            ),
            inference_seconds_excluding_model_loading=describe(
                [r["inference_seconds"] for r in per_object]
            ),
            solver_CPU_seconds={
                a: describe([r["solver_CPU_seconds"][a] for r in per_object])
                for a in ("delphi6", "classical12", "classical146")
            },
            note=scores["cost_note"],
        ),
        refined_periods_outside_one_percent_of_DAMIT=outside,
        per_object=per_object,
        protocol_settings_transcribed=dict(
            source="DeLPHI-followup/data/own-period-20261021/refine-chain/PROTOCOL.md",
            evaluation_sample_n=140,
            classical_comparison_starts=12,
            frequency_window_percent_each_side=1,
            frequency_spacing_coefficient_for_span_in_hours=0.4,
            frequency_spacing_note="0.5 / span_days / 24 * 0.8 = 0.4 / span_hours; same standard poles and distributed solver defaults; select lowest chi-squared row and round period to six decimals",
        ),
        runtime=dict(
            numpy_version=np.__version__,
            affinity=sorted(os.sched_getaffinity(0)),
            nice=os.getpriority(os.PRIO_PROCESS, 0),
            processes=1,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("paired-gains", "refine-medians"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=FixedDataPath("refine-chain/scores.json"))
    parser.add_argument("--check-published", type=Path)
    args = parser.parse_args()
    network.guard_output(args.output)
    result = paired_gains() if args.kind == "paired-gains" else refine_medians(args.source)
    if args.check_published:
        expected = json.loads(args.check_published.read_text())
        if result != expected:
            raise ValueError("aggregation differs from published values")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
