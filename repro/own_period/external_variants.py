"""ZTF preparation variants and the runs without reported uncertainties, at our own period.

    python -m repro.own_period.external_variants prepare-no-error
    python -m repro.own_period.external_variants search --set ztf-variants|no-error
    python -m repro.own_period.external_variants predict --set ztf-variants|no-error --periods published|own
    python -m repro.own_period.external_variants score --set ztf-variants|no-error --periods published|own

ZTF variants: the eight prepared inputs of ``run_k3_generalization_diagnostics.py``
(``mapped-ztf-diagnostics-current-endpoint``), predicted as in its ``predict``
(five networks of the held-out run, chunk size 256).

No uncertainties: the prepared ZTF and ALCDEF inputs with the reported errors
removed by ``run_k3_error_channel_sensitivity.prepare_directory`` (written anew
here and checked against the published hashes), predicted as in its
``predict_ztf`` / ``predict_alcdef`` (chunk size 1024).

``search`` applies ``estimate(curves, "fourier_gpu", resolution=1, timeout=180,
sparse_fallback=True)`` to each prepared variant as the networks receive it
(``search_external.curves_of``); no reference period is read.  ``predict`` with
``published`` uses the period stored in the prepared file (validation) and with
``own`` the period found by ``search``; an asteroid without a period is written
as a failure.  ``score`` reads the reference poles only afterwards.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from lc_pipeline.k3.bundle import _model_inputs
from lc_pipeline.k3.evaluation import ensemble_score_grids, modes_from_score_grid
from lc_pipeline.k3.inference import refine_ensemble_axes, score_axial_grid
from lc_pipeline.k3.ztf_prediction import _fold_policy, _load_prepared
from lc_pipeline.v2.preprocessing import KnownPeriod
from repro import network_root as network

from .nettests_common import FOLLOWUP, MODEL_MANIFEST, MODELS, OUT, SPLITS, sha256, write_json_once

GEN = FOLLOWUP / "data/generalization-20260910"
DIAG = GEN / "mapped-ztf-diagnostics-current-endpoint"
DIAG_PUBLISHED = GEN / "mapped-ztf-diagnostic-predictions-current-endpoint"
DIAG_SCORE = GEN / "mapped-ztf-development-diagnostic-score-current-endpoint.json"
ZTF_PREPARED = GEN / "mapped-ztf/prepared-current-endpoint"
ALCDEF_PREPARED = GEN / "locked-cohort/alcdef-prepared-30-20260911"
REVISION = FOLLOWUP / "data/publication-revision-20260917-r2"
FIXED_VARIANTS = OUT / 'ztf-variants'
FIXED_NOERR = OUT / 'no-uncertainty'
VARIANTS_OUT = network.NetworkPath('results', 'external-variants', 'ztf-variants', legacy_path=FIXED_VARIANTS)
NOERR_OUT = network.NetworkPath('results', 'external-variants', 'no-uncertainty', legacy_path=FIXED_NOERR)
PROVENANCE = "own-fourier-gpu-top1"
SEED = 20261021
FAILED_DEG = 90.0


def variant_items() -> dict[str, list[tuple[str, Path | None, str | None]]]:
    manifest = json.loads((DIAG / "manifest.json").read_text())
    items = defaultdict(list)
    for row in manifest["records"]:
        path = DIAG / row["path"] if row["status"] == "ready" else None
        items[row["variant"]].append((row["object_id"], path, row["sha256"]))
    return dict(items)


def no_error_items() -> dict[str, list[tuple[str, Path | None, str | None]]]:
    items = {}
    for cohort in ("ztf", "alcdef"):
        root = (FIXED_NOERR if network.switched() else NOERR_OUT) / f"prepared-{cohort}-no-error"
        manifest = json.loads((root / "manifest.json").read_text())
        items[cohort] = [
            (r["object_id"], root / r["path"] if r.get("path") else None, r.get("sha256"))
            for r in manifest["objects"]
        ]
    return items


def items_of(name: str):
    return variant_items() if name == "ztf-variants" else no_error_items()


def fixed_of(name: str):
    return FIXED_VARIANTS if name == 'ztf-variants' else FIXED_NOERR


def out_of(name: str) -> Path:
    return VARIANTS_OUT if name == "ztf-variants" else NOERR_OUT


def prepare_no_error(_args) -> int:
    if network.switched():
        raise ValueError('reuse the fixed prepared no-error inputs')
    from repro.run_k3_error_channel_sensitivity import prepare_directory

    report = {}
    for cohort, source, label, published in (
        ("ztf", ZTF_PREPARED, "mapped_ztf_same_identity", "error-channel-ztf-prepared-manifest.json"),
        ("alcdef", ALCDEF_PREPARED, "alcdef_gaia_new_identity", "error-channel-alcdef-prepared-manifest.json"),
    ):
        destination = NOERR_OUT / f"prepared-{cohort}-no-error"
        prepare_directory(source, destination, cohort=label)
        new = json.loads((destination / "manifest.json").read_text())
        old = json.loads((REVISION / published).read_text())
        new_hash = {r["object_id"]: r.get("sha256") for r in new["objects"]}
        old_hash = {r["object_id"]: r.get("sha256") for r in old["objects"]}
        report[cohort] = {
            "objects": len(new_hash),
            "ready": sum(v is not None for v in new_hash.values()),
            "identical_to_published_prepared_files": new_hash == old_hash,
        }
    write_json_once(NOERR_OUT / "prepare-check.json", report)
    print(json.dumps(report))
    return 0


def search(args) -> int:
    if network.switched():
        raise ValueError('reuse the fixed per-variant period searches')
    from repro.period_search.search import estimate

    from .search_external import curves_of

    out = out_of(args.set)
    for variant, items in sorted(items_of(args.set).items()):
        if args.only and variant not in args.only:
            continue
        destination = out / f"periods-{variant}.json"
        if destination.exists():
            continue
        periods, details = {}, {}
        for object_id, path, digest in items:
            if path is None:
                periods[object_id], details[object_id] = None, {"status": "input_unavailable"}
                continue
            if sha256(path) != digest:
                raise ValueError(f"{variant} {object_id}: prepared hash mismatch")
            started = time.perf_counter()
            try:
                curves = curves_of(path)
                result = estimate(curves, "fourier_gpu", resolution=1, timeout=180, sparse_fallback=True)
            except TimeoutError as error:
                result = {"status": "timeout", "reason": str(error), "ranked_periods": []}
            ranked = result.get("ranked_periods") or []
            periods[object_id] = float(ranked[0]) if result.get("status") == "ok" and ranked else None
            details[object_id] = {
                "status": result.get("status"),
                "reason": result.get("reason"),
                "input_regime": result.get("input_regime"),
                "ranked_periods": [float(v) for v in ranked[:5]],
                "seconds": time.perf_counter() - started,
            }
        write_json_once(
            destination,
            {
                "schema": "delphi.own-periods.v1",
                "provenance": PROVENANCE,
                "set": args.set,
                "variant": variant,
                "search_call": 'estimate(curves_of(prepared), "fourier_gpu", resolution=1, timeout=180, sparse_fallback=True)',
                "periods_hours": periods,
                "details": details,
            },
        )
        print(variant, sum(v is not None for v in periods.values()), "of", len(periods), flush=True)
    return 0


def _predict(models, epochs, period, device, chunk, sensitivity_style):
    inputs = _model_inputs(epochs, period, torch.device(device))
    if not sensitivity_style:
        grids = np.stack([score_axial_grid(m, inputs, chunk_size=chunk) for m in models])
        modes = modes_from_score_grid(ensemble_score_grids(grids[:, None, :])[0])
        axes, _ = refine_ensemble_axes(models, inputs, np.asarray([m.axis_xyz for m in modes]))
        return axes
    with torch.inference_mode():
        grids = np.stack([score_axial_grid(m, inputs, chunk_size=chunk) for m in models])
        modes = modes_from_score_grid(ensemble_score_grids(grids[:, None, :])[0])
    with torch.enable_grad():
        axes, _ = refine_ensemble_axes(models, inputs, np.asarray([m.axis_xyz for m in modes]))
    return axes


def predict(args) -> int:
    out = out_of(args.set)
    network.guard_output(out)
    for variant, items in sorted(items_of(args.set).items()):
        if args.only and variant not in args.only:
            continue
        destination = out / f"predictions-{args.periods}-{variant}.jsonl"
        if destination.exists():
            network.check_jsonl(destination)
            continue
        own = None
        if args.periods == "own":
            own = json.loads(((fixed_of(args.set) if network.switched() else out) /
                              f"periods-{variant}.json").read_text())["periods_hours"]
        if args.set == "ztf-variants" or variant == "ztf":
            groups = defaultdict(list)
            for item in items:
                groups[_fold_policy(item[0], SPLITS, "existing_identity")].append(item)
        else:
            groups = {tuple(range(5)): list(items)}
        chunk = 256 if args.set == "ztf-variants" else 1024
        sensitivity_style = args.set == "no-error"
        lines = []
        for folds, rows in sorted(groups.items()):
            models = None
            for object_id, path, digest in rows:
                record = {"object_id": object_id, "variant": variant, "folds": list(folds),
                          **network.json_metadata()}
                if path is None:
                    lines.append({**record, "status": "failed", "reason": "input_unavailable"})
                    continue
                if sha256(path) != digest:
                    raise ValueError(f"{variant} {object_id}: prepared hash mismatch")
                loaded_id, published, epochs = _load_prepared(path)
                if loaded_id != object_id:
                    raise ValueError("prepared identity mismatch")
                if own is None:
                    period = published
                elif own.get(object_id) is None:
                    lines.append({**record, "status": "failed", "reason": "no_period"})
                    continue
                else:
                    period = KnownPeriod(float(own[object_id]), PROVENANCE)
                if models is None:
                    models = network.load_historical_models(MODELS, MODEL_MANIFEST, folds, args.device)[0]
                try:
                    axes = _predict(models, epochs, period, args.device, chunk, sensitivity_style)
                    lines.append({**record, "status": "ok", "period_hours": period.hours,
                                  "axes": np.asarray(axes).tolist()})
                except (ValueError, RuntimeError) as exc:
                    lines.append({**record, "status": "failed", "reason": f"{type(exc).__name__}: {exc}",
                                  "period_hours": period.hours})
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            del models
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x") as stream:
            stream.writelines(json.dumps(line) + "\n" for line in lines)
        print(variant, args.periods, sum(line["status"] == "ok" for line in lines), "ok of", len(lines), flush=True)
    return 0


def _summary(e):
    e = np.asarray(e, dtype=float)
    return {"n": int(e.size), "mean_deg": float(e.mean()), "median_deg": float(np.median(e)),
            "within_20_percent": float(100 * np.mean(e <= 20 + 1e-8)),
            "within_30_percent": float(100 * np.mean(e <= 30 + 1e-8))}


def score(args) -> int:
    from repro import compute_reference_baselines as crb

    out = out_of(args.set)
    network.guard_output(out)
    if network.switched():
        crb.FOLLOWUP, crb.ROOT = FOLLOWUP, FOLLOWUP.parent
        crb.CATALOG = FOLLOWUP / 'inputs/training-run/repro/data/damit-20250610T000301Z/catalog.jsonl'
    table, catalog, gaia = crb.damit_table_poles(), crb.reference_poles(), crb.gaia_reference_poles()
    standard_axes = crb.standard_axes()
    result = {"set": args.set, "periods": args.periods, "variants": {}, **network.json_metadata()}
    published_checks = {}
    if args.set == "ztf-variants" and not network.switched():
        published = json.loads(DIAG_SCORE.read_text())["variants"]
    elif not network.switched():
        sensitivity = json.loads((REVISION / "error-channel-sensitivity.json").read_text())
    for variant in sorted(items_of(args.set)):
        network.check_jsonl(out / f'predictions-{args.periods}-{variant}.jsonl')
        lines = [json.loads(line) for line in (out / f"predictions-{args.periods}-{variant}.jsonl").read_text().splitlines()]
        alcdef = args.set == "no-error" and variant == "alcdef"
        ids, err, std, reasons = [], [], [], defaultdict(int)
        for line in lines:
            refs = gaia.get(line["object_id"]) if alcdef else crb.cohort_reference(line["object_id"], table, catalog)
            if refs is None:
                raise ValueError(f"no reference for {line['object_id']}")
            if line["status"] == "ok":
                a = np.asarray(line["axes"], dtype=float).reshape(-1, 3)
                e = crb.axial_error_deg(a / np.linalg.norm(a, axis=1, keepdims=True), refs)
            else:
                e = np.nan
                reasons[line["reason"].split(":")[0]] += 1
            ids.append(line["object_id"])
            err.append(e)
            std.append(crb.axial_error_deg(standard_axes, refs))
        err, std = np.asarray(err), np.asarray(std)
        available = np.asarray([line.get("reason") != "input_unavailable" for line in lines])
        ok = np.isfinite(err)
        at90 = np.where(ok, err, FAILED_DEG)
        a90, s_av = at90[available], std[available]
        block = {
            "n_intended": len(lines),
            "n_available_inputs": int(available.sum()),
            "n_ok": int(ok.sum()),
            "failure_reasons": dict(reasons),
            "available_failed_at_90": _summary(a90),
            "failed_excluded": _summary(err[ok]) if ok.any() else None,
            "standard_on_available": _summary(s_av),
            "standard_on_ok": _summary(std[ok]) if ok.any() else None,
            "objects": [{"object_id": i, "error_deg": None if not np.isfinite(e) else float(e),
                         "standard_error_deg": float(s)} for i, e, s in zip(ids, err, std)],
        }
        # own minus standard (protocol seed), and standard minus DeLPHI with the
        # seed of the published no-uncertainty comparison (20260924)
        block["minus_standard_failed_at_90"] = crb.paired_difference(a90, s_av, SEED)
        block["standard_minus_delphi_failed_at_90_seed20260924"] = crb.paired_difference(s_av, a90, 20260924)
        if ok.any():
            block["minus_standard_failed_excluded"] = crb.paired_difference(err[ok], std[ok], SEED + 1)
            block["standard_minus_delphi_failed_excluded_seed20260924"] = crb.paired_difference(
                std[ok], err[ok], 20260924
            )
        result["variants"][variant] = block
        if args.periods == "published" and not network.switched():
            if args.set == "ztf-variants":
                pv = published[variant]
                old = {r["object_id"]: r for r in pv["objects"]} if "objects" in pv else None
                check = {"published_fixed_available_mean": pv["fixed_available_input_cohort"]["mean_oracle_at_3_error_deg"],
                         "rerun_available_failed_at_90_mean": block["available_failed_at_90"]["mean_deg"],
                         "published_failures": pv["prediction_failure_count"],
                         "rerun_failures": len(lines) - int(ok.sum())}
                if old is not None:
                    diffs = [abs((old[i].get("oracle_at_3_error_deg") or 90.0) - (e if np.isfinite(e) else 90.0))
                             for i, e, av in zip(ids, err, available) if av]
                    check["max_abs_object_error_diff_deg"] = float(max(diffs))
            else:
                key = "ztf" if variant == "ztf" else "alcdef_gaia"
                check = {"published_without_error_mean": sensitivity[key]["without_error_channel_mean_deg"],
                         "rerun_mean": block["available_failed_at_90"]["mean_deg"]}
                pub_pred = json.loads((REVISION / f"error-channel-{variant}-predictions.json").read_text())["objects"]
                pa = {r["object_id"]: np.asarray(r["axes"]) for r in pub_pred}
                mine = {line["object_id"]: np.asarray(line["axes"]) for line in lines if line["status"] == "ok"}
                check["same_objects"] = sorted(pa) == sorted(mine)
                check["max_abs_axis_diff"] = float(max(np.max(np.abs(pa[k] - mine[k])) for k in pa))
            published_checks[variant] = check
    if published_checks:
        result["validation_against_published"] = published_checks
    destination = out / f"score-{args.periods}.json"
    write_json_once(destination, result)
    for variant, block in result["variants"].items():
        print(variant, json.dumps({k: block[k] for k in ("n_ok", "available_failed_at_90", "failed_excluded", "standard_on_available")}))
    if published_checks:
        print(json.dumps(published_checks, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare-no-error")
    for name in ("search", "predict", "score"):
        p = commands.add_parser(name)
        p.add_argument("--set", choices=("ztf-variants", "no-error"), required=True)
        p.add_argument("--only", nargs="+")
        if name in ("predict", "score"):
            p.add_argument("--periods", choices=("published", "own"), required=True)
        if name == "predict":
            p.add_argument("--device", default="cuda")
    args = parser.parse_args()
    return {"prepare-no-error": prepare_no_error, "search": search, "predict": predict, "score": score}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
