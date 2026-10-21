"""Check that ``lowq_score`` on the published predictions reproduces the published numbers.

    python -m repro.own_period.lowq_check_published --scores OP/lowq/scores-published-check --output X.json

Compares the recomputed values with the published files: ``lowq-analysis.json``
(census), ``lowq-lineage-analysis.json`` (lineage), the lower-quality cohorts of
``reference-baselines.json`` (standard poles, Table 8, Figure 7) and the
``lower_quality_direction`` block of ``mirror-symmetry.json`` (Figure 11,
latitude and longitude).  Exit code 1 if any value differs by more than 1e-9.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .lowq_search import FOLLOWUP

REVISION = FOLLOWUP / "data/publication-revision-20260917-r2"
BASELINES = FOLLOWUP / "source/repro/baselines_out/reference-baselines.json"
MIRROR = FOLLOWUP / "data/interpretation-20260926/mirror-symmetry.json"
TOLERANCE = 1e-9


def flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from flatten(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list) and value and not isinstance(value[0], (dict, list)):
        for index, item in enumerate(value):
            yield f"{prefix}[{index}]", item
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from flatten(item, f"{prefix}[{index}]")
    else:
        yield prefix, value


def compare(name, published, recomputed, rows):
    left, right = dict(flatten(published)), dict(flatten(recomputed))
    for key, value in left.items():
        other = right.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            ok = isinstance(other, (int, float)) and abs(float(value) - float(other)) <= TOLERANCE
        else:
            ok = value == other
        rows.append({"item": f"{name}:{key}", "published": value, "recomputed": other, "match": ok})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict] = []
    census_pub = json.loads((REVISION / "lowq-analysis.json").read_text())
    census_new = json.loads((args.scores / "census-analysis-excluded.json").read_text())
    for key in ("analyzed_denominator", "selected_denominator", "input_rejected_count",
                "primary", "quality_strata", "train_only_atlas", "paired_atlas_minus_k3"):
        compare(f"census.{key}", census_pub[key], census_new[key], rows)
    errors_pub = [o["oracle_at_3_error_deg"] for o in census_pub["objects"]]
    errors_new = [o["oracle_at_3_error_deg"] for o in census_new["objects"]]
    compare("census.object_errors", errors_pub, errors_new, rows)
    lineage_pub = json.loads((REVISION / "lowq-lineage-analysis.json").read_text())
    lineage_new = json.loads((args.scores / "lineage-excluded/analysis.json").read_text())
    for key in ("lineage_counts", "structured_reference_disjoint", "structured_reference_overlap",
                "structured_reference_disjoint_source_groups", "shared_source_count",
                "structured_reference_coverage"):
        compare(f"lineage.{key}", lineage_pub[key], lineage_new[key], rows)
    base_pub = json.loads(BASELINES.read_text())["cohorts"]
    base_new = json.loads((args.scores / "baselines-excluded.json").read_text())
    for cohort in ("lowq", "lowq_gaia", "lowq_asas", "lowq_shared", "lowq_unshared"):
        for key in ("object_count", "unmatched_count", "delphi", "standard", "standard_minus_delphi"):
            compare(f"baselines.{cohort}.{key}", base_pub[cohort][key], base_new[cohort][key], rows)
    mirror_pub = json.loads(MIRROR.read_text())["summary"]["lower_quality_direction"]
    mirror_new = json.loads((args.scores / "direction-excluded.json").read_text())
    for key in mirror_pub:
        compare(f"direction.{key}", mirror_pub[key], mirror_new[key], rows)
    mismatches = [row for row in rows if not row["match"]]
    report = {
        "schema": "delphi.own-period-lowq-published-check.v1",
        "tolerance": TOLERANCE,
        "compared_values": len(rows),
        "mismatches": mismatches,
        "passed": not mismatches,
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "mismatches"}), len(mismatches))
    for row in mismatches[:20]:
        print(row)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
