#!/usr/bin/env python3
"""Stratify the frozen low-Q DAMIT census by lightcurve-source lineage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

SCHEMA = "delphi.k3-lowq-damit-lineage-analysis.v1"
BOOTSTRAP_SEED = 20260917
BOOTSTRAP_RESAMPLES = 10_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _source_records(snapshot: Path, asteroid_id: str) -> list[dict[str, str]]:
    path = snapshot / "files" / asteroid_id / "lc.ref.csv"
    if not path.is_file():
        return []
    records: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            bibcode = str(row.get("bibcode") or "").strip()
            label = str(row.get("display_label") or "").strip()
            raw = bibcode or label
            if not raw:
                continue
            key = raw.casefold()
            records[key] = {
                "key": key,
                "bibcode": bibcode,
                "display_label": label,
            }
    return [records[key] for key in sorted(records)]


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot summarize an empty lineage stratum")
    k3 = np.asarray([row["oracle_at_3_error_deg"] for row in rows], dtype=float)
    atlas = np.asarray([row["atlas_oracle_at_3_error_deg"] for row in rows], dtype=float)
    return {
        "n_objects": len(rows),
        "k3": {
            "mean_deg": float(np.mean(k3)),
            "median_deg": float(np.median(k3)),
            "within_20_fraction": float(np.mean(k3 <= 20.0)),
            "within_30_fraction": float(np.mean(k3 <= 30.0)),
        },
        "train_only_atlas": {
            "mean_deg": float(np.mean(atlas)),
            "median_deg": float(np.median(atlas)),
            "within_20_fraction": float(np.mean(atlas <= 20.0)),
            "within_30_fraction": float(np.mean(atlas <= 30.0)),
        },
        "paired_atlas_minus_k3_mean_deg": float(np.mean(atlas - k3)),
    }


def _bootstrap(rows: list[dict[str, object]], resamples: int) -> dict[str, object]:
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    k3 = np.asarray([row["oracle_at_3_error_deg"] for row in rows], dtype=float)
    atlas = np.asarray([row["atlas_oracle_at_3_error_deg"] for row in rows], dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty((resamples, 3), dtype=float)
    for draw in range(resamples):
        indices = rng.integers(0, len(rows), size=len(rows))
        estimates[draw] = (
            float(np.mean(k3[indices])),
            float(np.mean(atlas[indices])),
            float(np.mean(atlas[indices] - k3[indices])),
        )
    low, high = np.quantile(estimates, (0.025, 0.975), axis=0)
    return {
        "method": "identity_bootstrap_percentile",
        "seed": BOOTSTRAP_SEED,
        "resamples": resamples,
        "k3_mean_deg_95_interval": [float(low[0]), float(high[0])],
        "atlas_mean_deg_95_interval": [float(low[1]), float(high[1])],
        "paired_atlas_minus_k3_mean_deg_95_interval": [
            float(low[2]),
            float(high[2]),
        ],
    }


def analyze(
    snapshot: Path,
    split_path: Path,
    census_analysis_path: Path,
    output: Path,
    *,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, object]:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    split = _read_json(split_path)
    census = _read_json(census_analysis_path)
    folds = split.get("folds")
    rows = census.get("objects")
    if not isinstance(folds, list) or not isinstance(rows, list) or not rows:
        raise ValueError("split or census analysis has an invalid schema")

    training_ids = sorted(
        {
            identifier
            for fold in folds
            if isinstance(fold, dict)
            for identifier in fold.get("train_ids", [])
            if isinstance(identifier, str)
        }
    )
    training_sources: dict[str, dict[str, str]] = {}
    for asteroid_id in training_ids:
        for record in _source_records(snapshot, asteroid_id):
            training_sources[record["key"]] = record
    if not training_sources:
        raise ValueError("no training lightcurve-source records were found")

    classified: list[dict[str, object]] = []
    source_counts: Counter[str] = Counter()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("object_id"), str):
            raise ValueError("census object row is invalid")
        object_id = str(row["object_id"])
        if not object_id.startswith("damit:"):
            raise ValueError(f"unexpected census identity: {object_id}")
        asteroid_id = f"asteroid_{object_id.split(':', 1)[1]}"
        sources = _source_records(snapshot, asteroid_id)
        keys = [record["key"] for record in sources]
        source_counts.update(keys)
        overlaps = sorted(set(keys) & set(training_sources))
        status = "missing_source" if not keys else "source_overlap" if overlaps else "source_disjoint"
        classified.append(
            {
                **row,
                "lineage_status": status,
                "source_keys": keys,
                "overlapping_training_source_keys": overlaps,
            }
        )

    strata_rows = {
        status: [row for row in classified if row["lineage_status"] == status]
        for status in ("source_disjoint", "source_overlap", "missing_source")
    }
    if not strata_rows["source_disjoint"] or not strata_rows["source_overlap"]:
        raise ValueError("lineage analysis requires nonempty disjoint and overlap strata")

    all_source_keys = set(source_counts)
    result: dict[str, object] = {
        "schema": SCHEMA,
        "study_scope": "post_hoc_low_quality_DAMIT_literature_source_lineage",
        "lineage_definition": (
            "source_disjoint means that no case-folded exact bibcode, or display-label "
            "fallback when bibcode is absent, occurs among any real train-role identity "
            "in any of the five frozen folds"
        ),
        "reference_quality_warning": census.get("reference_quality_warning"),
        "inputs": {
            "snapshot_model_table_sha256": _sha256(snapshot / "tables" / "asteroid_models.csv"),
            "split_sha256": _sha256(split_path),
            "census_analysis_sha256": _sha256(census_analysis_path),
        },
        "training_identity_count": len(training_ids),
        "training_source_count": len(training_sources),
        "census_source_count": len(all_source_keys),
        "shared_source_count": len(all_source_keys & set(training_sources)),
        "new_source_count": len(all_source_keys - set(training_sources)),
        "lineage_counts": {status: len(values) for status, values in strata_rows.items()},
        "source_disjoint": {
            **_summary(strata_rows["source_disjoint"]),
            "bootstrap": _bootstrap(strata_rows["source_disjoint"], bootstrap_resamples),
        },
        "source_overlap": {
            **_summary(strata_rows["source_overlap"]),
            "bootstrap": _bootstrap(strata_rows["source_overlap"], bootstrap_resamples),
        },
        "source_frequency": [
            {"source_key": key, "object_count": count, "seen_in_training": key in training_sources}
            for key, count in sorted(source_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "objects": classified,
    }
    output.mkdir(parents=True)
    _write_json(output / "analysis.json", result)
    with (output / "objects.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            "object_id",
            "quality_flag",
            "lineage_status",
            "source_keys",
            "overlapping_training_source_keys",
            "oracle_at_3_error_deg",
            "atlas_oracle_at_3_error_deg",
            "observation_count",
            "epoch_count",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in classified:
            writer.writerow(
                {
                    **{name: row.get(name) for name in fieldnames},
                    "source_keys": "|".join(row["source_keys"]),
                    "overlapping_training_source_keys": "|".join(
                        row["overlapping_training_source_keys"]
                    ),
                }
            )
    disjoint = result["source_disjoint"]
    assert isinstance(disjoint, dict)
    k3 = disjoint["k3"]
    atlas = disjoint["train_only_atlas"]
    interval = disjoint["bootstrap"]["paired_atlas_minus_k3_mean_deg_95_interval"]
    summary = (
        "# Low-quality DAMIT literature-source analysis\n\n"
        "This post hoc analysis uses exact DAMIT lightcurve reference identifiers. "
        "It treats an object as source-disjoint only when none of its references occurs "
        "among any real training identity in any frozen fold.\n\n"
        f"There are {result['lineage_counts']['source_disjoint']:,} source-disjoint objects. "
        f"Their mean oracle@3 disagreement is {k3['mean_deg']:.2f} deg for K3 and "
        f"{atlas['mean_deg']:.2f} deg for the train-only atlas. The paired atlas-minus-K3 "
        f"difference is {disjoint['paired_atlas_minus_k3_mean_deg']:.2f} deg "
        f"(95% identity-bootstrap interval {interval[0]:.2f}--{interval[1]:.2f} deg).\n\n"
        "The references are lower-confidence Q1/Q2 model solutions. This is evidence of "
        "new-object and new-literature-source transfer within the DAMIT compilation, not "
        "cross-survey validation or physical ground truth.\n"
    )
    (output / "summary.md").write_text(summary, encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--census-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    args = parser.parse_args()
    try:
        result = analyze(
            args.snapshot,
            args.splits,
            args.census_analysis,
            args.output,
            bootstrap_resamples=args.bootstrap_resamples,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({"lineage_counts": result["lineage_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
