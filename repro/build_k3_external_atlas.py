#!/usr/bin/env python3
"""Fit one external-evaluation atlas from the frozen original training labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from lc_pipeline.v2.atlas import fit_axis_atlas, write_axis_atlas
from lc_pipeline.v2.data import PreparedObject, load_catalog
from lc_pipeline.v2.preprocessing import TOKEN_FEATURE_NAMES


def _placeholder(object_id: str, vectors: tuple[tuple[float, float, float], ...], periods: tuple[float, ...]) -> PreparedObject:
    count = len(vectors)
    return PreparedObject(
        object_id=object_id,
        tokens=np.zeros((1, 1, len(TOKEN_FEATURE_NAMES)), dtype=np.float32),
        observation_mask=np.ones((1, 1), dtype=bool),
        epoch_descriptors=np.zeros((1, 3), dtype=np.float32),
        epoch_mask=np.ones(1, dtype=bool),
        period_values=np.zeros(2, dtype=np.float32),
        period_mask=np.zeros(2, dtype=bool),
        target_pixels=np.zeros(count, dtype=np.int64),
        target_vectors=np.asarray(vectors, dtype=np.float64),
        period_hours_targets=np.asarray(periods, dtype=np.float64),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fold_roles(splits_path: Path, fold: int) -> dict[str, list[str]]:
    payload = json.loads(splits_path.read_text(encoding="utf-8"))
    rows = payload.get("folds")
    if not isinstance(rows, list):
        raise ValueError("split document has no folds")
    matches = [row for row in rows if isinstance(row, dict) and row.get("fold") == fold]
    if len(matches) != 1:
        raise ValueError(f"split document has no unique fold {fold}")
    roles = matches[0]
    required = ("train_ids", "validation_ids", "calibration_ids", "test_ids")
    if any(not isinstance(roles.get(name), list) for name in required):
        raise ValueError("split fold has incomplete roles")
    return {name: [str(value) for value in roles[name]] for name in required}


def build(
    catalog_path: Path,
    output: Path,
    prediction_output: Path,
    *,
    splits_path: Path | None = None,
    fold: int | None = None,
) -> dict[str, object]:
    if output.exists() or prediction_output.exists():
        raise ValueError("atlas outputs must both be new")
    catalog = load_catalog(catalog_path)
    if (splits_path is None) != (fold is None):
        raise ValueError("--splits and --fold must be supplied together")
    if splits_path is None:
        train_ids = sorted(catalog)
        forbidden_ids: list[str] = []
    else:
        assert fold is not None
        roles = _fold_roles(splits_path, fold)
        train_ids = sorted(roles["train_ids"])
        forbidden_ids = sorted(
            identifier
            for role in ("validation_ids", "calibration_ids", "test_ids")
            for identifier in roles[role]
        )
        if set(train_ids) & set(forbidden_ids):
            raise ValueError("fold training identities overlap another role")
    missing = sorted(set(train_ids) - set(catalog))
    if missing:
        raise ValueError(f"atlas training identities are absent from the catalog: {missing[:10]}")
    values = [
        _placeholder(
            catalog[object_id].object_id,
            catalog[object_id].solution_vectors,
            catalog[object_id].solution_periods_hours,
        )
        for object_id in train_ids
    ]
    atlas = fit_axis_atlas(
        values,
        partition_role="train",
        forbidden_object_ids=forbidden_ids,
    )
    write_axis_atlas(output, atlas)
    source_ids = []
    for row in values:
        match = re.fullmatch(r"asteroid_([1-9][0-9]*)", row.object_id)
        if match is None:
            raise ValueError("frozen catalog identity cannot be namespaced")
        source_ids.append(f"damit:{match.group(1)}")
    payload = {
        "schema": "delphi.k3-external-train-only-atlas.v1",
        "train_only": True,
        "atlas_path": str(output),
        "atlas_sha256": atlas.atlas_sha256,
        "axes": [list(axis) for axis in atlas.axes],
        "source_identity_ids": sorted(source_ids),
        "source_catalog_path": str(catalog_path),
        "source_catalog_sha256": _sha256(catalog_path),
        "fold": fold,
        "split_path": None if splits_path is None else str(splits_path),
        "split_sha256": None if splits_path is None else _sha256(splits_path),
    }
    prediction_output.parent.mkdir(parents=True, exist_ok=True)
    prediction_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--atlas-output", type=Path, required=True)
    parser.add_argument("--prediction-output", type=Path, required=True)
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--fold", type=int, choices=range(5))
    args = parser.parse_args()
    try:
        result = build(
            args.catalog,
            args.atlas_output,
            args.prediction_output,
            splits_path=args.splits,
            fold=args.fold,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({"atlas_sha256": result["atlas_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
