"""Fail-closed publication summary and manuscript macros for the K3 study."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .evaluation import evaluate_oof_promotion_gate, summarize_errors
from .protocol import K3_PROTOCOL_SHA256
from .publication_run import K3_DIAGNOSTIC_CONTROLS, K3_OOF_SEEDS


class K3ReleaseError(ValueError):
    """Raised when definitive artifacts cannot support a release summary."""


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> str:
    if path.exists():
        raise K3ReleaseError(f"refusing to overwrite release output {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _sha256(path)


def _load_json(path: Path, schema: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise K3ReleaseError(f"cannot load {path.name}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise K3ReleaseError(f"{path.name} has the wrong schema")
    return payload


def _valid_errors(values: np.ndarray, count: int, description: str) -> np.ndarray:
    errors = np.asarray(values, dtype=np.float64)
    if (
        errors.shape != (count,)
        or not np.all(np.isfinite(errors))
        or np.any(errors < 0)
        or np.any(errors > 90)
    ):
        raise K3ReleaseError(f"{description} errors are invalid")
    return errors


def _summary(values: Sequence[float]) -> dict[str, float]:
    return summarize_errors(values)


def _latex_macros(summary: Mapping[str, object]) -> str:
    metrics = summary["metrics"]
    comparisons = summary["oof_gate"]["metrics"]
    p_values = summary["oof_gate"]["p_values_holm"]
    calibration = summary["calibration"]
    downstream = summary["downstream"]["verdict"]

    def deg(value: float) -> str:
        return f"{value:.2f}\\ensuremath{{^\\circ}}"

    def percent(value: float) -> str:
        return f"{100.0 * value:.1f}\\%"

    verdict = (
        "\\textsc{promote K3}"
        if summary["promotion_passed"]
        else "\\textsc{retain V1}"
    )
    lines = [
        "% Generated from publication-summary.json; do not edit measured values.",
        f"\\renewcommand{{\\KThreeImplementationCommit}}{{\\texttt{{{summary['implementation_commit'][:12]}}}}}",
        f"\\renewcommand{{\\KThreeSyntheticMean}}{{{deg(metrics['synthetic']['mean_error_deg'])}}}",
        f"\\renewcommand{{\\KThreeOofN}}{{{int(metrics['oof']['n_objects'])}}}",
        f"\\renewcommand{{\\KThreeOofMean}}{{{deg(metrics['oof']['mean_error_deg'])}}}",
        f"\\renewcommand{{\\KThreeOofMedian}}{{{deg(metrics['oof']['median_error_deg'])}}}",
        f"\\renewcommand{{\\KThreeOofWithinTwenty}}{{{percent(metrics['oof']['fraction_within_20_deg'])}}}",
        f"\\renewcommand{{\\KThreeOofWithinThirty}}{{{percent(metrics['oof']['fraction_within_30_deg'])}}}",
        f"\\renewcommand{{\\KThreeSwapGap}}{{{deg(metrics['mean_real_input_swap_gap_deg'])}}}",
        f"\\renewcommand{{\\KThreeLabelShuffleGap}}{{{deg(metrics['label_shuffle_gap_deg'])}}}",
        f"\\renewcommand{{\\KThreeVOneMean}}{{{deg(metrics['v1']['mean_error_deg'])}}}",
        f"\\renewcommand{{\\KThreeVOneMedian}}{{{deg(metrics['v1']['median_error_deg'])}}}",
        f"\\renewcommand{{\\KThreeVOneWithinTwenty}}{{{percent(metrics['v1']['fraction_within_20_deg'])}}}",
        f"\\renewcommand{{\\KThreeAtlasMean}}{{{deg(metrics['atlas']['mean_error_deg'])}}}",
        f"\\renewcommand{{\\KThreeAtlasMedian}}{{{deg(metrics['atlas']['median_error_deg'])}}}",
        f"\\renewcommand{{\\KThreeAtlasWithinTwenty}}{{{percent(metrics['atlas']['fraction_within_20_deg'])}}}",
        f"\\renewcommand{{\\KThreeDerangedMean}}{{{deg(metrics['deranged']['mean_error_deg'])}}}",
        f"\\renewcommand{{\\KThreeDerangedMedian}}{{{deg(metrics['deranged']['median_error_deg'])}}}",
        f"\\renewcommand{{\\KThreeDerangedWithinTwenty}}{{{percent(metrics['deranged']['fraction_within_20_deg'])}}}",
        f"\\renewcommand{{\\KThreeVOneImprovement}}{{{deg(comparisons['improvement_vs_v1_deg'])}}}",
        f"\\renewcommand{{\\KThreeVOneBootstrapLower}}{{{deg(comparisons['improvement_vs_v1_ci95_low_deg'])}}}",
        f"\\renewcommand{{\\KThreeHolmP}}{{{p_values['vs_v1']:.3g}}}",
        "\\renewcommand{\\KThreeConeNinetyRadiusRange}{"
        f"{calibration['cone90_min_deg']:.2f}--{calibration['cone90_max_deg']:.2f}"
        "\\ensuremath{^\\circ}}",
        f"\\renewcommand{{\\KThreeConeNinetyCoverage}}{{{percent(calibration['empirical_coverage90'])}}}",
        f"\\renewcommand{{\\KThreeConeNinetyFiveCoverage}}{{{percent(calibration['empirical_coverage95'])}}}",
        f"\\renewcommand{{\\KThreeDownstreamSpeedRatio}}{{{downstream['speed_ratio']:.2f}\\ensuremath{{\\times}}}}",
        f"\\renewcommand{{\\KThreeDownstreamSpeedRatioLower}}{{{downstream['speed_ratio_ci95_low']:.2f}\\ensuremath{{\\times}}}}",
        "\\renewcommand{\\KThreeDownstreamRecoveryDelta}{"
        f"{100.0 * downstream['recovery_rate_inferiority']:+.1f} percentage points}}",
        f"\\renewcommand{{\\KThreeDownstreamRmsUpper}}{{{downstream['geometric_mean_rms_ratio_ci95_high']:.3f}}}",
        f"\\renewcommand{{\\KThreePromotionVerdict}}{{{verdict}}}",
        "",
    ]
    return "\n".join(lines)


def build_publication_release(
    *,
    artifact_root: str | Path,
    comparator_path: str | Path,
    output_json: str | Path,
    output_tex: str | Path,
) -> dict[str, object]:
    """Validate every mandatory result and write one immutable promotion record."""
    root = Path(artifact_root)
    evaluations = root / "evaluations"
    audit_path = root / "audits" / "checkpoint-audit.json"
    downstream_path = root / "downstream" / "fixed-period" / "fixed-period-summary.json"
    audit = _load_json(audit_path, "delphi.k3-checkpoint-audit.v1")
    downstream = _load_json(
        downstream_path, "delphi.k3-fixed-period-summary.v1"
    )
    if (
        audit.get("passed") is not True
        or audit.get("protocol_sha256") != K3_PROTOCOL_SHA256
        or not isinstance(audit.get("implementation_commit"), str)
    ):
        raise K3ReleaseError("checkpoint audit did not pass the locked protocol")

    ensemble_path = evaluations / "real-oof-ensemble.npz"
    try:
        with np.load(ensemble_path, allow_pickle=False) as ensemble:
            if ensemble["schema"].item() != "delphi.k3-real-oof-ensemble.v1":
                raise K3ReleaseError("OOF ensemble has the wrong schema")
            object_ids = tuple(ensemble["object_ids"].astype(str))
            oof_errors = _valid_errors(
                ensemble["oracle_errors_deg"], 170, "OOF ensemble"
            )
            seeds = tuple(int(value) for value in ensemble["seeds"])
    except (OSError, ValueError, KeyError) as exc:
        raise K3ReleaseError(f"cannot load OOF ensemble: {exc}") from exc
    if seeds != K3_OOF_SEEDS or len(object_ids) != len(set(object_ids)):
        raise K3ReleaseError("OOF ensemble seed/object identity is invalid")

    comparator_source = Path(comparator_path)
    try:
        with np.load(comparator_source, allow_pickle=False) as comparator:
            if comparator["schema"].item() != "delphi.k3-v1-comparators.v1":
                raise K3ReleaseError("V1 comparator has the wrong schema")
            if tuple(comparator["object_ids"].astype(str)) != object_ids:
                raise K3ReleaseError("V1 comparator object order differs from K3")
            v1 = _valid_errors(comparator["v1_errors_deg"], 170, "V1")
            atlas = _valid_errors(comparator["atlas_errors_deg"], 170, "atlas")
            deranged = _valid_errors(
                comparator["deranged_errors_deg"], 170, "deranged"
            )
    except (OSError, ValueError, KeyError) as exc:
        raise K3ReleaseError(f"cannot load V1 comparator: {exc}") from exc

    seed_raw: dict[int, np.ndarray] = {}
    seed_swap: dict[int, np.ndarray] = {}
    real_swap_gaps: dict[str, float] = {}
    diagnostic_gaps: dict[str, dict[str, float]] = {}
    for seed in K3_OOF_SEEDS:
        raw_path = evaluations / f"real-oof-seed-{seed}.npz"
        swap_path = evaluations / f"real-oof-seed-{seed}-input-swap.npz"
        try:
            with np.load(raw_path, allow_pickle=False) as raw:
                if tuple(raw["object_ids"].astype(str)) != object_ids:
                    raise K3ReleaseError(f"seed {seed} raw OOF IDs differ")
                raw_errors = _valid_errors(raw["oracle_errors_deg"], 170, f"seed {seed} raw")
            with np.load(swap_path, allow_pickle=False) as swap:
                if (
                    swap["schema"].item() != "delphi.k3-real-oof-input-swap.v1"
                    or tuple(swap["object_ids"].astype(str)) != object_ids
                ):
                    raise K3ReleaseError(f"seed {seed} swap OOF IDs/schema differ")
                swap_errors = _valid_errors(
                    swap["oracle_errors_deg"], 170, f"seed {seed} swap"
                )
                bound_raw = str(swap["raw_artifact_sha256"].item())
        except (OSError, ValueError, KeyError) as exc:
            raise K3ReleaseError(f"cannot load seed {seed} OOF controls: {exc}") from exc
        if bound_raw != _sha256(raw_path):
            raise K3ReleaseError(f"seed {seed} input-swap does not bind its raw artifact")
        seed_raw[seed], seed_swap[seed] = raw_errors, swap_errors
        real_swap_gaps[str(seed)] = float(np.mean(swap_errors - raw_errors))
        diagnostic_gaps[str(seed)] = {}
        for control in K3_DIAGNOSTIC_CONTROLS:
            path = evaluations / f"real-oof-seed-{seed}-{control}.npz"
            try:
                with np.load(path, allow_pickle=False) as diagnostic:
                    if (
                        diagnostic["schema"].item()
                        != "delphi.k3-real-oof-diagnostic.v1"
                        or tuple(diagnostic["object_ids"].astype(str)) != object_ids
                        or diagnostic["control"].item() != control
                    ):
                        raise K3ReleaseError(
                            f"seed {seed} diagnostic {control} is misaligned"
                        )
                    degradation = np.asarray(
                        diagnostic["degradation_deg"], dtype=np.float64
                    )
            except (OSError, ValueError, KeyError) as exc:
                raise K3ReleaseError(
                    f"cannot load seed {seed} diagnostic {control}: {exc}"
                ) from exc
            if degradation.shape != (170,) or not np.all(np.isfinite(degradation)):
                raise K3ReleaseError(
                    f"seed {seed} diagnostic {control} degradation is invalid"
                )
            diagnostic_gaps[str(seed)][control] = float(np.mean(degradation))

    synthetic_raw: list[np.ndarray] = []
    synthetic_swap: list[np.ndarray] = []
    synthetic_ids: tuple[str, ...] | None = None
    synthetic_swap_gaps: dict[str, float] = {}
    for seed in K3_OOF_SEEDS:
        raw_path = evaluations / f"synthetic-test-seed-{seed}.npz"
        swap_path = evaluations / f"synthetic-test-seed-{seed}-input-swap.npz"
        try:
            with np.load(raw_path, allow_pickle=False) as raw:
                identifiers = tuple(raw["object_ids"].astype(str))
                raw_errors = _valid_errors(
                    raw["oracle_errors_deg"], 2000, f"synthetic seed {seed}"
                )
            with np.load(swap_path, allow_pickle=False) as swap:
                if tuple(swap["object_ids"].astype(str)) != identifiers:
                    raise K3ReleaseError(f"synthetic seed {seed} swap IDs differ")
                swap_errors = _valid_errors(
                    swap["oracle_errors_deg"], 2000, f"synthetic swap seed {seed}"
                )
                raw_hashes = np.asarray(swap["raw_artifact_sha256"]).astype(str)
        except (OSError, ValueError, KeyError) as exc:
            raise K3ReleaseError(
                f"cannot load synthetic seed {seed} controls: {exc}"
            ) from exc
        if (
            synthetic_ids is not None
            and identifiers != synthetic_ids
            or raw_hashes.shape != (2000,)
            or set(raw_hashes.tolist()) != {_sha256(raw_path)}
        ):
            raise K3ReleaseError(f"synthetic seed {seed} controls are not aligned/bound")
        synthetic_ids = identifiers
        synthetic_raw.append(raw_errors)
        synthetic_swap.append(swap_errors)
        synthetic_swap_gaps[str(seed)] = float(np.mean(swap_errors - raw_errors))
    synthetic_errors = np.mean(np.stack(synthetic_raw), axis=0)

    label_path = evaluations / "synthetic-label-shuffle-test-seed-17.npz"
    try:
        with np.load(label_path, allow_pickle=False) as label:
            if tuple(label["object_ids"].astype(str)) != synthetic_ids:
                raise K3ReleaseError("label-shuffle synthetic IDs differ")
            label_errors = _valid_errors(
                label["oracle_errors_deg"], 2000, "label-shuffle"
            )
    except (OSError, ValueError, KeyError) as exc:
        raise K3ReleaseError(f"cannot load label-shuffle control: {exc}") from exc
    label_gap = float(np.mean(label_errors - synthetic_raw[0]))
    synthetic_metrics = _summary(synthetic_errors)
    synthetic_failures: list[str] = []
    if synthetic_metrics["mean_error_deg"] > 20.0:
        synthetic_failures.append("synthetic mean error exceeds 20 degrees")
    if synthetic_metrics["median_error_deg"] > 18.0:
        synthetic_failures.append("synthetic median error exceeds 18 degrees")
    if min(synthetic_swap_gaps.values()) < 8.0:
        synthetic_failures.append("a synthetic seed input-swap gap is below 8 degrees")
    if label_gap < 0:
        synthetic_failures.append("label-shuffle training improves synthetic error")

    oof_verdict = evaluate_oof_promotion_gate(
        oof_errors,
        v1_errors=v1,
        atlas_errors=atlas,
        deranged_errors=deranged,
        seed_k3_errors=seed_raw,
        seed_input_swap_errors=seed_swap,
    )
    calibration_path = evaluations / "real-oof-ensemble-calibrated.npz"
    try:
        with np.load(calibration_path, allow_pickle=False) as calibration_artifact:
            if (
                calibration_artifact["schema"].item()
                != "delphi.k3-real-oof-ensemble-calibration.v1"
                or tuple(calibration_artifact["object_ids"].astype(str)) != object_ids
            ):
                raise K3ReleaseError("ensemble calibration is misaligned")
            cone90 = np.asarray(
                calibration_artifact["fold_cone90_deg"], dtype=np.float64
            )
            cone95 = np.asarray(
                calibration_artifact["fold_cone95_deg"], dtype=np.float64
            )
            covered90 = np.asarray(calibration_artifact["covered90"], dtype=bool)
            covered95 = np.asarray(calibration_artifact["covered95"], dtype=bool)
    except (OSError, ValueError, KeyError) as exc:
        raise K3ReleaseError(f"cannot load ensemble calibration: {exc}") from exc
    if (
        cone90.shape != (5,)
        or cone95.shape != (5,)
        or covered90.shape != (170,)
        or covered95.shape != (170,)
        or not np.all(np.isfinite(cone90))
        or not np.all(cone95 == 90.0)
    ):
        raise K3ReleaseError("ensemble calibration arrays are invalid")
    calibration = {
        "cone90_min_deg": float(np.min(cone90)),
        "cone90_max_deg": float(np.max(cone90)),
        "empirical_coverage90": float(np.mean(covered90)),
        "empirical_coverage95": float(np.mean(covered95)),
    }
    downstream_verdict = downstream.get("verdict")
    if not isinstance(downstream_verdict, dict):
        raise K3ReleaseError("downstream summary lacks a verdict")

    failures = [f"synthetic: {value}" for value in synthetic_failures]
    failures.extend(f"OOF: {value}" for value in oof_verdict.failures)
    if downstream_verdict.get("passed") is not True:
        downstream_failures = downstream_verdict.get("failures", [])
        failures.extend(f"downstream: {value}" for value in downstream_failures)
    metrics: dict[str, object] = {
        "synthetic": synthetic_metrics,
        "oof": _summary(oof_errors),
        "v1": _summary(v1),
        "atlas": _summary(atlas),
        "deranged": _summary(deranged),
        "synthetic_input_swap_gap_deg_by_seed": synthetic_swap_gaps,
        "real_input_swap_gap_deg_by_seed": real_swap_gaps,
        "mean_real_input_swap_gap_deg": float(np.mean(list(real_swap_gaps.values()))),
        "label_shuffle_gap_deg": label_gap,
        "diagnostic_gap_deg_by_seed": diagnostic_gaps,
    }
    source_hashes = {
        "checkpoint_audit": _sha256(audit_path),
        "oof_ensemble": _sha256(ensemble_path),
        "calibrated_ensemble": _sha256(calibration_path),
        "v1_comparator": _sha256(comparator_source),
        "downstream_summary": _sha256(downstream_path),
    }
    payload: dict[str, object] = {
        "schema": "delphi.k3-publication-summary.v1",
        "protocol_sha256": K3_PROTOCOL_SHA256,
        "implementation_commit": audit["implementation_commit"],
        "promotion_passed": not failures,
        "promotion_failures": failures,
        "metrics": metrics,
        "oof_gate": oof_verdict.as_mapping(),
        "calibration": calibration,
        "downstream": downstream,
        "source_sha256": source_hashes,
    }
    json_path = Path(output_json)
    tex_path = Path(output_tex)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    json_sha256 = _atomic_text(json_path, encoded)
    tex_sha256 = _atomic_text(tex_path, _latex_macros(payload))
    return {
        **payload,
        "publication_summary_sha256": json_sha256,
        "latex_macros_sha256": tex_sha256,
    }


__all__ = ["K3ReleaseError", "build_publication_release"]
