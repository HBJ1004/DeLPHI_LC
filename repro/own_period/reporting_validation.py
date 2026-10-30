"""Repeat the published 20-case CPU/GPU V7 comparison with selected networks.

Only the two descriptive maximum-angle diagnostics are consumed by the macro
ledger. Differences are recorded; they are never interpreted as a new accuracy
measurement or a reason to tune the networks or solver.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from repro import network_root as network
from repro.own_period import DataPath
from repro.own_period.v7_grid import _angle, audit_case


def build(candidate, reference, published_scope):
    scope = json.loads(Path(published_scope).read_text())
    rows = []
    for old in scope["rows"]:
        case = old["case"]
        folders = [Path(root) / "cases" / case for root in (candidate, reference)]
        if network.switched():
            for folder in folders:
                audit_case(folder)
        a, b = [json.loads((f / "timed-result.json").read_text()) for f in folders]
        predictions = [json.loads((f / "prediction.json").read_text())["axes"] for f in folders]
        if a["starts_requested"] != b["starts_requested"] or a["object_id"] != b["object_id"]:
            raise ValueError("CPU/GPU validation identities or budgets differ")
        prediction_max = max(
            _angle(x["axis_xyz"], y["axis_xyz"]) for x, y in zip(*predictions, strict=True)
        )
        fa, fb = a["selected_fit"], b["selected_fit"]
        rows.append(
            dict(
                case=case,
                prediction_max_angle_deg=prediction_max,
                selected_angle_deg=None
                if fa is None or fb is None
                else _angle(fa["axis"], fb["axis"]),
                failure_membership_changed=(fa is None) != (fb is None),
            )
        )
    return dict(
        cases=len(rows),
        max_prediction_angle_deg=max(r["prediction_max_angle_deg"] for r in rows),
        max_selected_angle_deg_atan2=max(
            r["selected_angle_deg"] for r in rows if r["selected_angle_deg"] is not None
        ),
        scope="Same 20 saved V7 validation cases, repeat 0, guided20, CPU versus GPU; descriptive hardware differences",
        rows=rows,
        sources=dict(
            candidate=str(candidate),
            reference=str(reference),
            fixed_scope_sha256=network.sha256(published_scope),
        ),
        **network.json_metadata(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--published-scope",
        type=Path,
        default=DataPath("v67/v7-validation-guided-gpu/validation-vs-published.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    network.guard_output(args.output)
    result = build(args.candidate, args.reference, args.published_scope)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
