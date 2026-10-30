"""Shared paths and helpers for the own-period network tests (package ``nettests``).

Every script of the package takes ``--periods damit`` (the published first
DAMIT period, used for the validation runs) or ``--periods <file>`` (a period
file with ``periods_hours`` and ``provenance``, normally ``own-periods-170.json``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from repro.network_root import NetworkPath, guard_output
from repro.own_period import DataPath

ROOT = DataPath("../../..")
FOLLOWUP = DataPath("../..")
REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / "repro/data/damit-20250610T000301Z/catalog.jsonl"
SPLITS = REPO / "repro/data/damit-20250610T000301Z/publication-splits-v2.3.json"
DUMP = ROOT / "damit-20250610T000301Z"
ARTIFACTS = NetworkPath("artifacts")
MODELS = NetworkPath("models")
EVALUATIONS = NetworkPath("evaluations")
MODEL_MANIFEST = NetworkPath("binding")
OP = DataPath()
OUT = OP / "nettests"
OWN_PERIODS_170 = OP / "own-periods-170.json"
SEEDS = (17, 42, 137, 777, 2027)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fold_rows() -> list[dict]:
    split = json.loads(SPLITS.read_text())
    return sorted(split["folds"], key=lambda row: int(row["fold"]))


def period_override(spec: str) -> tuple[dict[str, float] | None, str | None, str]:
    """Return (periods or None for the published path, provenance, label)."""
    if spec == "damit":
        return None, None, "damit"
    path = Path(spec)
    value = json.loads(path.read_text())
    periods = value["periods_hours"]
    missing = [oid for oid, p in periods.items() if p is None]
    if missing:
        raise ValueError(f"period file lacks periods for {len(missing)} objects: {missing[:5]}")
    return {k: float(v) for k, v in periods.items()}, value["provenance"], sha256(path)


def load_examples(object_ids, periods, provenance):
    from lc_pipeline.k3 import publication_run as run

    if periods is None:
        return run.load_real_examples(CATALOG, DUMP, tuple(object_ids))
    return run.load_real_examples(
        CATALOG, DUMP, tuple(object_ids), period_hours=periods, period_provenance=provenance
    )


def write_json_once(path: Path, value) -> None:
    guard_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=1, sort_keys=True, allow_nan=False)
        stream.write("\n")


def max_abs(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0
