from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lc_pipeline.k3.reliability_publication import (
    ReliabilityPublicationError,
    _tex_table,
    baseline_errors,
    latex_escape,
    recompute_two_d,
    unseal,
)


def seal(payload):
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    return {
        "seal_schema": "sha256-canonical-json-v1",
        "payload": payload,
        "payload_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }


def test_unseal_rejects_changed_payload(tmp_path: Path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps(seal({"a": 1})))
    assert unseal(p) == {"a": 1}
    p.write_text(json.dumps({**seal({"a": 1}), "payload": {"a": 2}}))
    with pytest.raises(ReliabilityPublicationError):
        unseal(p)


def test_two_d_recompute_uses_object_repeat_means_and_80_match(monkeypatch):
    conditions = [
        f"two_d-block_count-{b}-per_block_cap-{c}"
        for b in (1, 3, 5, 10, "None")
        for c in (5, 10, 20, 50, "None")
    ]
    rows = [
        {
            "condition": condition,
            "object_id": str(obj),
            "fold": obj % 5,
            "repeat": rep,
            "status": "ok",
            "error_deg": float(obj + rep),
            "metadata": {"n_points": 4},
        }
        for condition in conditions
        for obj in range(80)
        for rep in range(3)
    ]
    mean = sum(obj + 1 for obj in range(80)) / 80
    archived = {
        condition: {
            "mean_error_deg": str(mean),
            "mean_error_bootstrap_lower_95": "0",
            "mean_error_bootstrap_upper_95": "100",
            "n_matched": "80",
        }
        for condition in conditions
    }
    monkeypatch.setattr(
        "lc_pipeline.k3.reliability_publication.stratified_asteroid_bootstrap",
        lambda *_a, **_k: {"bootstrap_lower_95": 0.0, "bootstrap_upper_95": 100.0},
    )
    assert recompute_two_d(rows, archived)[0]["mean_error_deg"] == mean
    baseline = {str(obj): float(obj) for obj in range(80)}
    result = recompute_two_d(list(reversed(rows)), archived, baseline)
    assert all(cell["paired_change_deg"] == 1 for cell in result)
    assert all(
        cell["failed_predictions"] == 0 and cell["scheduled_predictions"] == 240 for cell in result
    )
    with pytest.raises(ReliabilityPublicationError, match="object sets differ"):
        recompute_two_d(rows, archived, {"unmatched": 0.0})
    with pytest.raises(ReliabilityPublicationError, match="missing repeat"):
        recompute_two_d(rows[1:], archived)


def test_baseline_requires_unique_objects_and_ignores_ineligible():
    rows = [
        {"condition": "full", "object_id": "a", "status": "ok", "error_deg": 12.0},
        {"condition": "full", "object_id": "b", "status": "ineligible", "error_deg": None},
    ]
    assert baseline_errors(rows, "full", 1) == {"a": 12.0}
    with pytest.raises(ReliabilityPublicationError, match="duplicated"):
        baseline_errors(rows + [rows[0]], "full", 1)


def test_text_is_latex_safe():
    assert latex_escape("period_hours & 95% < 30") == r"period\_hours \& 95\% \textless{} 30"


def test_tex_cells_are_sorted_numerically_not_lexically():
    rows = [
        {
            "block_count": b,
            "observation_cap_per_block": c,
            "mean_error_deg": 20,
            "ci95_low_deg": 18,
            "ci95_high_deg": 22,
            "retained_observations_min": 10,
            "retained_observations_max": 20,
        }
        for b, c in ((10, 5), (3, 20), (1, 10), (1, 5), ("all", "all"))
    ]
    text = _tex_table(rows)
    assert text.index("1 & 5") < text.index("1 & 10") < text.index("3 & 20")
    assert text.index("3 & 20") < text.index("10 & 5") < text.index("all & all")
