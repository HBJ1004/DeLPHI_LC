#!/usr/bin/env python3
"""Lock a model-specific ALCDEF--Gaia transfer cohort before prediction.

Selection is deterministic over identities meeting the previously recorded
ALCDEF metadata screen and absent from the frozen model's released real cohort
and published synthetic donors. Gaia pole values, candidate axes, scores, and
errors are not read. The resulting cohort is retrospective and model-specific,
not a prospective or project-wide-history holdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SALT = "delphi-k3-alcdef-gaia-model-specific-cohort-20260911"
SCHEMA = "delphi.k3-alcdef-gaia-cohort-lock.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lock(candidate_inventory: Path, model_exposure: Path, count: int) -> dict[str, Any]:
    if count < 1:
        raise ValueError("cohort count must be positive")
    try:
        inventory = json.loads(candidate_inventory.read_text(encoding="utf-8"))
        exposure = json.loads(model_exposure.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read cohort inputs: {exc}") from exc
    if inventory.get("schema") != "delphi.k3-alcdef-gaia-eligibility.v1":
        raise ValueError("candidate inventory schema mismatch")
    if exposure.get("schema") != "delphi.k3-model-specific-exposure-audit.v1":
        raise ValueError("model-specific exposure schema mismatch")
    source_by_id = {
        str(row["object_id"]): row
        for row in inventory.get("candidates", [])
        if isinstance(row, dict) and isinstance(row.get("object_id"), str)
    }
    exposure_by_id = {
        str(row["object_id"]): row
        for row in exposure.get("objects", [])
        if isinstance(row, dict) and isinstance(row.get("object_id"), str)
    }
    if set(source_by_id) != set(exposure_by_id):
        raise ValueError("candidate inventory and exposure audit identities differ")
    eligible = []
    for object_id, row in source_by_id.items():
        if exposure_by_id[object_id].get("status") != "model_training_unexposed":
            continue
        alcdef = row.get("alcdef")
        if not isinstance(alcdef, dict) or not isinstance(alcdef.get("valid_point_count"), int):
            raise ValueError(f"candidate has invalid ALCDEF metadata: {object_id}")
        eligible.append((hashlib.sha256(f"{SALT}:{object_id}".encode()).hexdigest(), object_id, alcdef))
    eligible.sort()
    if len(eligible) < count:
        raise ValueError(f"only {len(eligible)} model-training-unexposed candidates for requested {count}")
    selected = [
        {
            "object_id": object_id,
            "selection_rank_sha256": digest,
            "alcdef_screen": {
                "valid_point_count": metadata["valid_point_count"],
                "session_count": metadata.get("session_count"),
                "distinct_session_years_approximation": metadata.get("distinct_session_years_approximation"),
                "filters": metadata.get("filters", []),
            },
        }
        for digest, object_id, metadata in eligible[:count]
    ]
    return {
        "schema": SCHEMA,
        "purpose": "pre-prediction frozen-model-specific ALCDEF-to-Gaia transfer cohort",
        "selection": {
            "salt": SALT,
            "rule": "ascending SHA-256(salt + ':' + MPC identity) among all model-training-unexposed metadata-screen candidates",
            "candidate_count": len(eligible),
            "selected_count": count,
            "forbidden_selection_inputs": ["Gaia pole values", "Gaia periods", "model predictions", "model scores", "reference errors"],
        },
        "sources": {
            "candidate_inventory": {"path": str(candidate_inventory), "sha256": _sha256(candidate_inventory)},
            "model_specific_exposure": {"path": str(model_exposure), "sha256": _sha256(model_exposure)},
        },
        "scope": {
            "model_training_unexposed": True,
            "project_wide_history_unexposed": False,
            "prospective": False,
            "permitted_description": "retrospective transfer cohort absent from the frozen model's released real cohort and published synthetic donors",
        },
        "objects": selected,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-inventory", type=Path, required=True)
    parser.add_argument("--model-exposure", type=Path, required=True)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        payload = lock(args.candidate_inventory, args.model_exposure, args.count)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
