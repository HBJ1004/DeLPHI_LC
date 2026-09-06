"""Object-level OOF aggregation for the locked definitive V1 study.

This module deliberately refuses partial OOF output.  It is the only intended
bridge from checkpoint-level JSONL rows to manuscript/result-table values.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..v2.statistics import (
    aggregate_seed_rows,
    holm_adjust,
    object_pooled_summary,
    paired_object_bootstrap,
    paired_sign_flip_permutation,
)
from .artifacts import atomic_write_json
from .contract import (
    PUBLICATION_SPEC_SHA256,
    PublicationContractError,
    PublicationInputs,
    sha256_file,
)
from .runner import OOFExecutionPlan, _read_jsonl, validate_oof_completeness

OOF_SUMMARY_SCHEMA = "delphi.publication-oof-summary.v1"


def _select(
    rows: Sequence[Mapping[str, Any]], *, model_kind: str, condition: str
) -> list[Mapping[str, Any]]:
    selected = [
        row
        for row in rows
        if row.get("model_kind") == model_kind and row.get("condition") == condition
    ]
    if not selected:
        raise PublicationContractError(f"OOF result has no {model_kind}/{condition} rows")
    return selected


def aggregate_object_rows(
    rows: Sequence[Mapping[str, Any]], *, model_kind: str, condition: str
) -> tuple[list[str], list[float]]:
    """Mean the five frozen seed values before assigning one vote per object."""
    object_ids, errors = aggregate_seed_rows(
        _select(rows, model_kind=model_kind, condition=condition),
        "axis_oracle_at3_error_deg",
    )
    return object_ids, errors.tolist()


def summarize_oof_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    """Create all candidate-generator metrics without opening downstream data."""
    arms = ("v1_faithful", "v1_corrected", "amplitude_mlp")
    conditions = ("raw", "zero", "atlas", "exhaustive_deranged")
    metrics: dict[str, object] = {}
    values: dict[tuple[str, str], tuple[list[str], list[float]]] = {}
    for arm in arms:
        for condition in conditions:
            ids, errors = aggregate_object_rows(rows, model_kind=arm, condition=condition)
            values[(arm, condition)] = (ids, errors)
            metrics[f"{arm}.{condition}"] = object_pooled_summary(ids, errors)
    primary_ids, primary = values[("v1_faithful", "raw")]
    comparisons: dict[str, object] = {}
    p_values: dict[str, float] = {}
    for name, identity in {
        "v1_vs_corrected": ("v1_corrected", "raw"),
        "v1_vs_amplitude_mlp": ("amplitude_mlp", "raw"),
        "v1_vs_zero": ("v1_faithful", "zero"),
        "v1_vs_atlas": ("v1_faithful", "atlas"),
        "v1_vs_deranged": ("v1_faithful", "exhaustive_deranged"),
    }.items():
        ids, comparator = values[identity]
        if ids != primary_ids:
            raise PublicationContractError(f"OOF object membership mismatch: {name}")
        bootstrap = paired_object_bootstrap(primary_ids, primary, comparator)
        permutation = paired_sign_flip_permutation(primary_ids, primary, comparator)
        comparisons[name] = {
            "bootstrap": asdict(bootstrap),
            "permutation": asdict(permutation),
        }
        p_values[name] = permutation.one_sided_p_value
    adjusted = holm_adjust(p_values)
    for name, adjusted_p in adjusted.items():
        assert isinstance(comparisons[name], dict)
        comparisons[name]["holm_adjusted_permutation_p_value"] = adjusted_p
    return {
        "schema": OOF_SUMMARY_SCHEMA,
        "publication_spec_sha256": PUBLICATION_SPEC_SHA256,
        "aggregation": "five-seed mean; one vote per outer-test object",
        "metrics": metrics,
        "paired_comparisons": comparisons,
    }


def summarize_completed_oof(
    plan: OOFExecutionPlan, inputs: PublicationInputs, *, output_path: str | Path
) -> dict[str, object]:
    """Validate all 75 jobs before writing a single immutable derived summary."""
    # ``validate_oof_completeness`` performs input membership and artifact checks.
    validate_oof_completeness(plan, inputs)
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for job in plan.jobs:
        path = job.output_directory / "evaluation-rows.jsonl"
        rows.extend(_read_jsonl(path))
        hashes[job.job_id] = sha256_file(path)
    summary = summarize_oof_rows(rows)
    summary["evaluation_rows_sha256"] = hashes
    destination = Path(output_path)
    atomic_write_json(destination, summary, overwrite=False)
    return summary


def manuscript_macros(summary: Mapping[str, object]) -> str:
    """Render only axis-result macros; downstream macros remain unavailable."""
    if summary.get("schema") != OOF_SUMMARY_SCHEMA:
        raise PublicationContractError("cannot render macros from another summary schema")
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        raise PublicationContractError("OOF summary lacks metrics")
    primary = metrics.get("v1_faithful.raw")
    if not isinstance(primary, Mapping):
        raise PublicationContractError("OOF summary lacks V1 primary metrics")
    try:
        mean = float(primary["mean_error_deg"])
        median = float(primary["median_error_deg"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PublicationContractError("OOF primary metrics are invalid") from exc
    return (
        "% Generated from the hash-bound OOF summary; do not edit.\n"
        f"\\renewcommand{{\\PubOracleMean}}{{{mean:.2f}\\ensuremath{{^\\circ}}}}\n"
        f"\\renewcommand{{\\PubOracleMedian}}{{{median:.2f}\\ensuremath{{^\\circ}}}}\n"
    )


__all__ = [
    "OOF_SUMMARY_SCHEMA",
    "aggregate_object_rows",
    "manuscript_macros",
    "summarize_completed_oof",
    "summarize_oof_rows",
]
