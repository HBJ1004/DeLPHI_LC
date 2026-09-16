#!/usr/bin/env python3
"""Aggregate complete label-free ALCDEF predictions and seal a receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from lc_pipeline.k3.generalization_validation import create_prediction_receipt


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seal(prediction_directory: Path, atlas_path: Path, lock_path: Path, output: Path, receipt: Path) -> dict[str, object]:
    if output.exists() or receipt.exists():
        raise ValueError("aggregate and receipt outputs must both be new")
    try:
        manifest = json.loads((prediction_directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read prediction directory manifest: {exc}") from exc
    if manifest.get("schema") != "delphi.k3-alcdef-transfer-prediction-directory.v1" or not isinstance(manifest.get("objects"), list):
        raise ValueError("prediction directory schema mismatch")
    rows = manifest["objects"]
    if manifest.get("object_count") != 30 or len(rows) != 30:
        raise ValueError("prediction directory is incomplete")
    predictions = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("prediction directory record is invalid")
        source = prediction_directory / row["path"]
        if _sha256(source) != row.get("sha256"):
            raise ValueError("prediction file hash mismatch")
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("object_id") != row.get("object_id") or payload.get("model_count") != 25 or len(payload.get("axes", [])) != 3:
            raise ValueError("prediction content is incomplete")
        predictions.append({"object_id": payload["object_id"], "status": "ok", "axes": [item["axis_xyz"] for item in payload["axes"]]})
    aggregate = {"schema": "delphi.k3-alcdef-transfer-prediction-set.v1", "prediction_directory_manifest_sha256": _sha256(prediction_directory / "manifest.json"), "objects": sorted(predictions, key=lambda row: row["object_id"])}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = create_prediction_receipt(lock_path=lock_path, prediction_paths={"candidate": output, "atlas": atlas_path})
    receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-directory", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = seal(args.prediction_directory, args.atlas, args.lock, args.output, args.receipt)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({"prediction_roles": sorted(result["predictions"])}, sort_keys=True))


if __name__ == "__main__":
    main()
