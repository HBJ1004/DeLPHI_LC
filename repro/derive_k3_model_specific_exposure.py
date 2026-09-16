#!/usr/bin/env python3
"""Derive frozen-model training exposure separately from project-wide history.

This does not replace the conservative project-history audit.  It answers the
narrower reproducible question needed for a frozen-model transfer test: whether
an identity occurs in the released model's original real cohort or its
published synthetic-donor manifests.  A result marked unexposed here must not
be described as prospectively held out or unseen by every historical workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

CURRENT_MODEL_ROLES = {
    "original_170_publication_splits",
    "published_final_synthetic_96_shards",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derive(candidate_inventory: Path) -> dict[str, Any]:
    try:
        document = json.loads(candidate_inventory.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read candidate inventory: {exc}") from exc
    if document.get("schema") != "delphi.k3-alcdef-gaia-eligibility.v1":
        raise ValueError("candidate inventory schema mismatch")
    rows = document.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError("candidate inventory has no candidates")
    output = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("object_id"), str):
            raise ValueError("candidate inventory contains an invalid object")
        exposure = row.get("exposure")
        if not isinstance(exposure, dict) or not isinstance(exposure.get("reasons"), list):
            raise ValueError("candidate inventory lacks exposure reasons")
        direct_roles = sorted(
            reason.removeprefix("present_in:")
            for reason in exposure["reasons"]
            if isinstance(reason, str)
            and reason.startswith("present_in:")
            and reason.removeprefix("present_in:") in CURRENT_MODEL_ROLES
        )
        output.append({
            "object_id": row["object_id"],
            "status": "model_training_exposed" if direct_roles else "model_training_unexposed",
            "current_model_exposure_roles": direct_roles,
            "project_history_status": exposure.get("status"),
            "project_history_reasons": exposure["reasons"],
        })
    counts = {status: sum(item["status"] == status for item in output) for status in ("model_training_exposed", "model_training_unexposed")}
    return {
        "schema": "delphi.k3-model-specific-exposure-audit.v1",
        "purpose": "frozen-model training/donor exposure only; not a project-wide history or prospective-holdout attestation",
        "candidate_inventory": {"path": str(candidate_inventory), "sha256": _sha256(candidate_inventory)},
        "current_model_exposure_roles": sorted(CURRENT_MODEL_ROLES),
        "counts": counts,
        "objects": output,
        "permitted_description": "unseen by the frozen model's released real cohort and published synthetic donors",
        "forbidden_descriptions": ["prospective external validation", "unseen by all prior development", "project-wide unexposed"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        payload = derive(args.candidate_inventory)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
