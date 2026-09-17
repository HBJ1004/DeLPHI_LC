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


def _source_file_sha256(snapshot: Path, asteroid_id: str) -> str | None:
    path = snapshot / "files" / asteroid_id / "lc.ref.csv"
    return _sha256(path) if path.is_file() else None


def _structured_reference_coverage(
    snapshot: Path, asteroid_ids: list[str]
) -> dict[str, int]:
    lightcurve_count = 0
    referenced_serials: set[tuple[str, int]] = set()
    identities_without_structured_references = 0
    for asteroid_id in asteroid_ids:
        lightcurve_path = snapshot / "files" / asteroid_id / "lc.txt"
        try:
            with lightcurve_path.open(encoding="ascii") as handle:
                count = int(handle.readline().strip())
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot count lightcurves for {asteroid_id}: {exc}") from exc
        lightcurve_count += count
        reference_path = snapshot / "files" / asteroid_id / "lc.ref.csv"
        serials: set[int] = set()
        if reference_path.is_file():
            with reference_path.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    try:
                        serials.add(int(str(row.get("light_curve_serial_number") or "")))
                    except ValueError:
                        continue
        if not serials:
            identities_without_structured_references += 1
        referenced_serials.update((asteroid_id, serial) for serial in serials)
    return {
        "training_lightcurve_count": lightcurve_count,
        "training_lightcurves_with_structured_reference": len(referenced_serials),
        "training_lightcurves_without_structured_reference": (
            lightcurve_count - len(referenced_serials)
        ),
        "training_identities_without_structured_reference": (
            identities_without_structured_references
        ),
    }


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
    for start in range(0, resamples, 64):
        stop = min(resamples, start + 64)
        indices = rng.integers(0, len(rows), size=(stop - start, len(rows)))
        estimates[start:stop, 0] = np.mean(k3[indices], axis=1)
        estimates[start:stop, 1] = np.mean(atlas[indices], axis=1)
        estimates[start:stop, 2] = np.mean((atlas - k3)[indices], axis=1)
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
    training_source_inventory: list[dict[str, object]] = []
    for asteroid_id in training_ids:
        records = _source_records(snapshot, asteroid_id)
        for record in records:
            training_sources[record["key"]] = record
        training_source_inventory.append(
            {
                "asteroid_id": asteroid_id,
                "structured_reference_file_sha256": _source_file_sha256(
                    snapshot, asteroid_id
                ),
                "records": records,
            }
        )
    if not training_sources:
        raise ValueError("no training lightcurve-source records were found")
    structured_coverage = _structured_reference_coverage(snapshot, training_ids)

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
        status = (
            "missing_structured_reference"
            if not keys
            else "structured_reference_overlap"
            if overlaps
            else "structured_reference_disjoint"
        )
        classified.append(
            {
                **row,
                "lineage_status": status,
                "source_keys": keys,
                "overlapping_training_source_keys": overlaps,
                "structured_reference_file_sha256": _source_file_sha256(
                    snapshot, asteroid_id
                ),
            }
        )

    strata_rows = {
        status: [row for row in classified if row["lineage_status"] == status]
        for status in (
            "structured_reference_disjoint",
            "structured_reference_overlap",
            "missing_structured_reference",
        )
    }
    if (
        not strata_rows["structured_reference_disjoint"]
        or not strata_rows["structured_reference_overlap"]
    ):
        raise ValueError("lineage analysis requires nonempty disjoint and overlap strata")

    all_source_keys = set(source_counts)
    source_groups = {
        "gaia_dr3_study": [
            row
            for row in classified
            if row["source_keys"] == ["2023arxiv230510798d"]
        ],
        "asas_sn_study": [
            row
            for row in classified
            if row["source_keys"] == ["2021a&a...654a..48h"]
        ],
    }
    if sum(len(rows) for rows in source_groups.values()) != len(
        strata_rows["structured_reference_disjoint"]
    ):
        raise ValueError("structured-reference-disjoint objects have an unknown source group")
    result: dict[str, object] = {
        "schema": SCHEMA,
        "study_scope": "post_hoc_low_quality_DAMIT_literature_source_lineage",
        "lineage_definition": (
            "structured_reference_disjoint means that no case-folded exact bibcode, or "
            "display-label fallback when bibcode is absent, occurs in the structured "
            "lc.ref.csv records of any real train-role identity in any frozen fold"
        ),
        "lineage_limitation": (
            "The structured reference table is incomplete and this classification does not "
            "establish observing-survey independence."
        ),
        "reference_quality_warning": census.get("reference_quality_warning"),
        "inputs": {
            "snapshot_model_table_sha256": _sha256(snapshot / "tables" / "asteroid_models.csv"),
            "split_sha256": _sha256(split_path),
            "census_analysis_sha256": _sha256(census_analysis_path),
        },
        "training_identity_count": len(training_ids),
        "training_source_count": len(training_sources),
        "training_source_inventory": training_source_inventory,
        "structured_reference_coverage": structured_coverage,
        "census_source_count": len(all_source_keys),
        "shared_source_count": len(all_source_keys & set(training_sources)),
        "new_source_count": len(all_source_keys - set(training_sources)),
        "lineage_counts": {status: len(values) for status, values in strata_rows.items()},
        "structured_reference_disjoint": {
            **_summary(strata_rows["structured_reference_disjoint"]),
            "bootstrap": _bootstrap(
                strata_rows["structured_reference_disjoint"], bootstrap_resamples
            ),
        },
        "structured_reference_overlap": {
            **_summary(strata_rows["structured_reference_overlap"]),
            "bootstrap": _bootstrap(
                strata_rows["structured_reference_overlap"], bootstrap_resamples
            ),
        },
        "structured_reference_disjoint_source_groups": {
            name: {
                **_summary(rows),
                "bootstrap": _bootstrap(rows, bootstrap_resamples),
            }
            for name, rows in source_groups.items()
            if rows
        },
        "source_frequency": [
            {
                "source_key": key,
                "object_count": count,
                "seen_in_training": key in training_sources,
            }
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
            "structured_reference_file_sha256",
            "oracle_at_3_error_deg",
            "atlas_oracle_at_3_error_deg",
            "observation_count",
            "epoch_count",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
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
    disjoint = result["structured_reference_disjoint"]
    assert isinstance(disjoint, dict)
    k3 = disjoint["k3"]
    atlas = disjoint["train_only_atlas"]
    interval = disjoint["bootstrap"]["paired_atlas_minus_k3_mean_deg_95_interval"]
    summary = (
        "# Low-quality DAMIT literature-source analysis\n\n"
        "This post hoc analysis uses exact DAMIT lightcurve reference identifiers. "
        "It treats an object as structured-reference-disjoint only when none of its "
        "structured references occurs among any real training identity in any frozen "
        "fold. The structured reference table is incomplete.\n\n"
        f"There are {result['lineage_counts']['structured_reference_disjoint']:,} "
        "structured-reference-disjoint objects. "
        f"Their mean oracle@3 disagreement is {k3['mean_deg']:.2f} deg for K3 and "
        f"{atlas['mean_deg']:.2f} deg for the train-only atlas. The paired atlas-minus-K3 "
        f"difference is {disjoint['paired_atlas_minus_k3_mean_deg']:.2f} deg "
        f"(95% identity-bootstrap interval {interval[0]:.2f}--{interval[1]:.2f} deg).\n\n"
        "The references are lower-confidence Q1/Q2 model solutions. This is evidence of "
        "new-object transfer within the DAMIT compilation, not cross-survey validation, "
        "source independence, or physical ground truth.\n"
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
