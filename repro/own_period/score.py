"""Scores of the own-period runs, with the comparisons fixed in PROTOCOL.md.

    python -m repro.own_period.score --output ../data/own-period-20261021/own-period-scores.json

Every number is computed here from object-level errors.  Failures (no period)
are reported at 90 deg and also excluded.  Fractions use <= with a tolerance of
1e-8 deg.  Paired differences are comparator minus own-period DeLPHI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from repro import compute_reference_baselines as crb
from repro.network_root import (
    NetworkPath,
    check_jsonl,
    check_npz,
    guard_output,
    network_metadata,
    switched,
)

from .search_external import FOLLOWUP, OUT

SEED = 20261021
RESAMPLES = 10_000
TOL = 1e-8
FAILED_DEG = 90.0
OOF = NetworkPath("evaluations", "real-oof-ensemble.npz")
COMPARATORS = FOLLOWUP / "inputs/frozen-artifacts/k3-2408c563/comparators/v1-comparators.npz"
BASELINES = FOLLOWUP / "source/repro/baselines_out/reference-baselines-per-object.npz"
SENSITIVITY = FOLLOWUP / "data/publication-revision-20260917-r2/error-channel-sensitivity.json"
ALCDEF_ANALYSIS = (
    FOLLOWUP
    / "data/generalization-20260910/locked-cohort/alcdef-gaia-transfer-analysis-30-20260911.json"
)
CATALOG = FOLLOWUP / "inputs/training-run/repro/data/damit-20250610T000301Z/catalog.jsonl"


def summary(errors) -> dict:
    e = np.asarray(errors, dtype=float)
    return {
        "n": int(e.size),
        "mean_deg": float(e.mean()),
        "median_deg": float(np.median(e)),
        "within_20_percent": float(100 * np.mean(e <= 20 + TOL)),
        "within_30_percent": float(100 * np.mean(e <= 30 + TOL)),
    }


def paired(first, second, seed: int) -> dict:
    d = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    g = np.random.default_rng(seed)
    means = np.array([d[g.integers(0, d.size, d.size)].mean() for _ in range(RESAMPLES)])
    low, high = np.percentile(means, (2.5, 97.5))
    return {
        "n": int(d.size),
        "mean_difference_deg": float(d.mean()),
        "ci95_low_deg": float(low),
        "ci95_high_deg": float(high),
        "fraction_positive": float(np.mean(d > 0)),
    }


def period_class(own: float | None, reference: float) -> str:
    if own is None or not np.isfinite(own):
        return "failed"
    for name, factor in (("P", 1.0), ("P/2", 0.5), ("2P", 2.0)):
        if abs(own / reference - factor) <= 0.01 * factor:
            return name
    return "other"


def axial_error(axes, references) -> float:
    a = np.asarray(axes, dtype=float).reshape(-1, 3)
    return crb.axial_error_deg(a / np.linalg.norm(a, axis=1, keepdims=True), references)


def cohort_block(ids, own, published, comparators, classes) -> dict:
    """own: errors with NaN for failures, published/comparators: finite errors."""
    own = np.asarray(own, dtype=float)
    ok = np.isfinite(own)
    at90 = np.where(ok, own, FAILED_DEG)
    block = {
        "n": len(ids),
        "n_failed": int((~ok).sum()),
        "own_period_failed_at_90": summary(at90),
        "own_period_failed_excluded": summary(own[ok]),
        "published_period": summary(published),
        "published_period_on_own_successes": summary(np.asarray(published)[ok]),
    }
    seed = SEED
    for name, values in [("published_period", published), *comparators.items()]:
        values = np.asarray(values, dtype=float)
        if name != "published_period":
            block[name] = summary(values)
        block[f"{name}_minus_own_failed_at_90"] = paired(values, at90, seed)
        block[f"{name}_minus_own_failed_excluded"] = paired(values[ok], own[ok], seed + 1)
        seed += 2
    counts = {}
    for c in ("P", "P/2", "2P", "other", "failed"):
        mask = np.asarray([k == c for k in classes])
        counts[c] = {"n": int(mask.sum())}
        if mask.any() and c != "failed":
            counts[c]["own_mean_deg"] = float(own[mask].mean())
            counts[c]["published_mean_deg"] = float(np.asarray(published)[mask].mean())
    block["period_classes"] = counts
    block["objects"] = [
        {
            "object_id": i,
            "period_class": k,
            "own_error_deg": None if not np.isfinite(o) else float(o),
            "published_error_deg": float(p),
        }
        for i, k, o, p in zip(ids, classes, own, published, strict=True)
    ]
    return block


def main_sample(own_path: Path) -> dict:
    check_npz(own_path)
    check_npz(OOF)
    own = np.load(own_path)
    pub = np.load(OOF)
    comp = np.load(COMPARATORS)
    base = np.load(BASELINES)
    ids = [str(v) for v in pub["object_ids"]]
    for other in (own, comp, base):
        if [str(v) for v in other["object_ids"]] != ids:
            raise ValueError("object order differs")
    catalog = {r["object_id"]: r for r in map(json.loads, CATALOG.read_text().splitlines())}
    reference = [float(catalog[i]["solutions"][0]["period_hours"]) for i in ids]
    periods = own["period_hours"].astype(float)
    classes = [
        period_class(p if np.isfinite(p) else None, r)
        for p, r in zip(periods, reference, strict=True)
    ]
    return cohort_block(
        ids,
        own["errors_deg"],
        pub["oracle_errors_deg"],
        {
            "standard": base["standard_errors_deg"],
            "random": base["random_errors_deg"],
            "fitted_axes": comp["atlas_errors_deg"],
        },
        classes,
    )


def external(cohort: str, predictions: Path, periods_path: Path) -> dict:
    check_jsonl(predictions)
    rows = {r["object_id"]: r for r in map(json.loads, predictions.read_text().splitlines())}
    periods = json.loads(periods_path.read_text())["periods_hours"]
    table, catalog, gaia = (
        crb.damit_table_poles(),
        crb.reference_poles(),
        crb.gaia_reference_poles(),
    )
    if cohort == "ztf":
        source = json.loads(SENSITIVITY.read_text())["ztf"]["objects"]
        published = {r["object_id"]: r["original_error_deg"] for r in source}
        fitted = {r["object_id"]: r["train_only_atlas_error_deg"] for r in source}
    else:
        source = json.loads(ALCDEF_ANALYSIS.read_text())["objects"]
        published = {r["object_id"]: r["oracle_at_3_error_deg"] for r in source}
        fitted = {r["object_id"]: r["atlas_oracle_at_3_error_deg"] for r in source}
    from .search_external import ready_objects

    if switched():
        path = NetworkPath("results", cohort, f"validation-{cohort}-published.jsonl")
        check_jsonl(path)
        fresh = {r["object_id"]: r for r in map(json.loads, path.read_text().splitlines())}

    ids, own, refs, classes, reference_periods = [], [], [], [], {}
    for object_id, path, _ in ready_objects(cohort):
        references = (
            gaia.get(object_id)
            if cohort == "alcdef"
            else crb.cohort_reference(object_id, table, catalog)
        )
        if references is None:
            raise ValueError(f"{object_id}: no reference poles")
        if switched():
            reference_row = fresh[object_id]
            if reference_row["status"] != "ok":
                raise ValueError(f"{object_id}: reference-period inference failed")
            published[object_id] = axial_error(reference_row["axes"], references)
        known = json.loads(path.read_text())["known_period_hours"]
        reference_periods[object_id] = float(known)
        row = rows[object_id]
        ids.append(object_id)
        refs.append(references)
        own.append(axial_error(row["axes"], references) if row["status"] == "ok" else np.nan)
        classes.append(period_class(periods.get(object_id), float(known)))
    standard_axes = crb.standard_axes()
    standard = [crb.axial_error_deg(standard_axes, r) for r in refs]
    draws = crb.RANDOM_DRAWS // 10
    random, _ = crb.random_reference_errors(refs, crb.RANDOM_SEED + len(cohort), draws)
    return cohort_block(
        ids,
        own,
        [published[i] for i in ids],
        {
            "standard": standard,
            "random": random,
            "fitted_axes": [fitted[i] for i in ids],
        },
        classes,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, default=OUT / "own-period-oof.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--main-only", action="store_true", help="headline sample only; external cohorts are separate rerun families")
    parser.add_argument("--cohort", choices=("ztf", "alcdef"), help="score one external family only")
    args = parser.parse_args()
    guard_output(args.output)
    if args.main_only and args.cohort:
        parser.error("--main-only and --cohort are mutually exclusive")
    if switched():
        crb.FOLLOWUP, crb.CATALOG = FOLLOWUP, CATALOG
        crb.ROOT = FOLLOWUP.parent
    result = {
        "schema": "delphi.own-period-scores.v1",
        "protocol": "PROTOCOL.md",
        "seed": SEED,
        "resamples": RESAMPLES,
    }
    if not args.cohort:
        result["main_sample"] = main_sample(args.oof)
    if switched():
        result["network_binding_sha256"] = str(network_metadata()["network_binding_sha256"])
    for cohort in ("ztf", "alcdef"):
        if args.main_only:
            break
        if args.cohort and args.cohort != cohort:
            continue
        predictions = (NetworkPath("results", cohort, f"own-period-{cohort}.jsonl")
                       if switched() else OUT / f"own-period-{cohort}.jsonl")
        if switched() or predictions.exists():
            result[cohort] = external(cohort, predictions, OUT / f"own-periods-{cohort}.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    for key in ("main_sample", "ztf", "alcdef"):
        if key in result:
            b = result[key]
            print(key, json.dumps({k: v for k, v in b.items() if k not in ("objects",)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
