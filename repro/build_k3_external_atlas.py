#!/usr/bin/env python3
"""Fit one external-evaluation atlas from the frozen original training labels."""

from __future__ import annotations

import argparse
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


def build(catalog_path: Path, output: Path, prediction_output: Path) -> dict[str, object]:
    if output.exists() or prediction_output.exists():
        raise ValueError("atlas outputs must both be new")
    catalog = load_catalog(catalog_path)
    values = [_placeholder(row.object_id, row.solution_vectors, row.solution_periods_hours) for row in catalog.values()]
    atlas = fit_axis_atlas(values, partition_role="train")
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
    }
    prediction_output.parent.mkdir(parents=True, exist_ok=True)
    prediction_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--atlas-output", type=Path, required=True)
    parser.add_argument("--prediction-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = build(args.catalog, args.atlas_output, args.prediction_output)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({"atlas_sha256": result["atlas_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
