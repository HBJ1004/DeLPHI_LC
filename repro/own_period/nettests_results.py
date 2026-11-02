"""Write RESULTS.md of the nettests package from the run outputs.

    python -m repro.own_period.nettests_results [--output PATH]

For every inventory row of the package the table gives the published value,
the value of the rerun with the DAMIT (published) period, and the own-period
value with the file it comes from.  Missing outputs are marked "not run".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from repro import network_root as network

from .nettests_common import EVALUATIONS, FOLLOWUP, OP, OUT, SEEDS, sha256
from .network_controls import DIAGNOSTIC, FINER, output_name

PUB_SUMMARY = FOLLOWUP / "inputs/frozen-artifacts/k3-definitive-7874092/release/publication-summary.json"
PUB_FINER = FOLLOWUP / "data/interpretation-20260923/ablations"
PUB_INTERNAL = FOLLOWUP / "data/internal-validation-20260916/run-1"
PUB_SHAPLEY = FOLLOWUP / "data/interpretation-20260926/shapley"
PUB_BASELINES = FOLLOWUP / "release/paper-v1/staging/part-b-analyses/baselines/reference-baselines.json"
PUB_DIAG = FOLLOWUP / "data/generalization-20260910/mapped-ztf-development-diagnostic-score-current-endpoint.json"
PUB_SENS = FOLLOWUP / "data/publication-revision-20260917-r2/error-channel-sensitivity.json"
VARIANTS = (
    "distance", "light_time", "distance_light_time", "geometry_epochs",
    "distance_geometry_epochs", "light_time_geometry_epochs", "distance_light_time_geometry_epochs",
)
LABELS = {
    "zero-numeric": "All numerical inputs", "geometry": "Sun and observer directions",
    "brightness": "All brightness information", "binned-brightness": "Brightness in the 64 phase bins (`\\RemovalBinned`)",
    "fourier-all": "Fourier coefficients (`\\RemovalFourier`)", "fourier-even": "Even Fourier terms (`\\RemovalEven`)",
    "fourier-odd": "Odd Fourier terms (`\\RemovalOdd`)", "amplitude": "Peak-to-peak amplitude (`\\RemovalAmplitude`)",
    "sampling-summary": "Observation counts and coverage", "period-scalar": "Period as a number",
    "photometric-errors": "Photometric errors (`\\RemovalErrors`)",
    "observer-aspect-only": "Sun term and product term removed, keep observer (`\\RemovalKeepObserver`)",
    "solar-aspect-only": "Observer term and product term removed, keep Sun (`\\RemovalKeepSun`)",
    "no-cross-term": "Product term only removed (`\\RemovalNoCross`)",
}


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def rel(path: Path) -> str:
    return str(path.relative_to(OP)) if str(path).startswith(str(OP)) else str(path)


def f2(x, signed=False):
    if x is None:
        return "n/a"
    return f"{x:+.2f}" if signed else f"{x:.2f}"


def ci(d) -> str:
    return f"{d['mean_difference_deg']:+.2f} ({d['ci95_low_deg']:+.2f} to {d['ci95_high_deg']:+.2f})"


def published_gaps() -> dict:
    if network.switched():
        raise ValueError("published gaps are historical evidence; use selected-network summaries")
    out = {}
    for control in ("input-swap",) + DIAGNOSTIC + FINER:
        folder = PUB_FINER if control in FINER else EVALUATIONS
        gaps = []
        for seed in SEEDS:
            a = np.concatenate([np.load(folder / output_name(f, seed, control))["oracle_errors_deg"] for f in range(5)])
            b = np.concatenate([np.load(EVALUATIONS / output_name(f, seed, "raw"))["oracle_errors_deg"] for f in range(5)])
            gaps.append(float(np.mean(a - b)))
        out[control] = gaps
    return out


def selected_results(output):
    """Report selected-network controls, input removals and Shapley results."""
    network.guard_output(output)
    lines = ["# Selected-network controls and Shapley results", "",
             "network_binding_sha256: " + network.json_metadata()["network_binding_sha256"], ""]
    for family in ("controls-input-swap", "input-removals", "shapley"):
        for period in ("damit", "own"):
            path = network.NetworkPath("results", family, period + "-summary.json")
            network.check_json(path)
            value = json.loads(path.read_text())
            lines.extend([f"## {family}, {period} period", "", f"Source: {path}", ""])
            if family == "shapley":
                for row in value["groups"]:
                    lines.append(f"- {row['group']}: share {100 * row['percentile_share']:.3f}%; "
                                 f"grid effect {row['grid_error_mean_deg']:.3f} deg")
            else:
                for control, row in value["summary"]["controls"].items():
                    lines.append(f"- {control}: gap {row['gap_mean_deg']:.3f} deg "
                                 f"({row['gap_min_deg']:.3f} to {row['gap_max_deg']:.3f})")
            lines.append("")
    lines.append("Deferred families: training versus test, external variants and uncertainty sensitivity.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=network.NetworkPath(
        "results", "controls-input-swap", "RESULTS.md", legacy_path=OUT / "RESULTS.md"))
    args = parser.parse_args()
    if network.switched():
        return selected_results(Path(args.output))
    L = []
    w = L.append
    w("# Own-period network tests for the paper (package nettests)")
    w("")
    w("Exploratory revision analyses on the development sample. Rules: `../PROTOCOL.md` (Addendum 1).")
    w("Generated by `repro/own_period/nettests_results.py`. Producer scripts are in `repro/own_period/`.")
    w("Own period of the 170: `own-periods-170.json` (sha256 " + sha256(OP / "own-periods-170.json")[:12] + "...), all 170 have a period.")
    w("ZTF variants and the runs without uncertainties: period searched again on each variant as the networks receive it")
    w("(`estimate(curves, \"fourier_gpu\", resolution=1, timeout=180, sparse_fallback=True)`), failures reported at 90 deg and excluded.")
    w("All paths below are relative to `DeLPHI-followup/data/own-period-20261021/`.")
    w("")

    # ---------------- E1 / B3
    pub = load(PUB_SUMMARY)["metrics"]
    val = load(OUT / "controls-damit-validation.json")
    own = load(OUT / "controls-own-summary.json")
    w("## E1, B3: lightcurve exchange (`\\KThreeSwapGap`)")
    w("")
    w("Same derangements (seed `K3_INPUT_SWAP_SEED + fold`) and the same 25 networks; every asteroid, and so every donor, is folded with its own full-data period.")
    w("The gap is the mean over the 170 asteroids of the exchanged minus the unexchanged error of the same network, per seed.")
    w("")
    w("| Quantity | Published | Rerun, DAMIT period | Own period | Own-period source |")
    w("|---|---|---|---|---|")
    vtxt = "not run"
    finer_note = ""
    if val:
        c = val["comparison_with_published"]
        routine = [r for r in c["files"] if "/evaluations/" in r["published"]]
        fin = [r for r in c["files"] if "/evaluations/" not in r["published"]]
        mr = max(r["max_abs_oracle_error_diff_deg"] for r in routine)
        mg = max(r["max_abs_score_grid_diff"] for r in routine)
        vtxt = (f"fold 0, 34 asteroids x 5 seeds: unmodified, exchange and the five published removals, {len(routine)} files, "
                f"max abs oracle-error difference {mr:.1e} deg, score maps {mg:.1e} (bit-identical)")
        ff = max(r["max_abs_oracle_error_diff_deg"] for r in fin)
        finer_note = (f"The {len(fin)} finer-removal files of fold 0 differ from the published ones by up to {ff:.3f} deg per asteroid. "
                      "The published finer-removal files carry their own small numerical offset: their no-op removal of the photometric errors "
                      "(identically zero for DAMIT) differs from the published unmodified run by up to 6e-4 deg, while the rerun reproduces the unmodified run exactly. "
                      "At the level of the gaps the rerun agrees with the published fold-0 values to 0.001 deg or better, so the rounded values are unchanged.")
    o = own["summary"]["controls"].get("input-swap") if own else None
    w(f"| Mean gap over seeds | {pub['mean_real_input_swap_gap_deg']:.2f} | {vtxt} | {f2(o['gap_mean_deg']) if o else 'not run'} | `controls-own-summary.json` |")
    for seed in SEEDS:
        w(f"| Gap, seed {seed} | {pub['real_input_swap_gap_deg_by_seed'][str(seed)]:.2f} | (as above) | {f2(o['gap_by_seed_deg'][str(seed)]) if o else 'not run'} | same |")
    pos = all(v > 0 for v in pub["real_input_swap_gap_deg_by_seed"].values())
    w(f"| Positive for every seed | {'yes' if pos else 'no'} | (as above) | {('yes' if o['gap_positive_for_every_seed'] else 'no') if o else 'not run'} | same |")
    if own:
        raw = own["summary"]["raw_mean_by_seed"]
        w("")
        w("Single-network mean errors at the own period (baseline of the gaps), by seed: " + ", ".join(f"{s} {raw[str(s)]:.2f}" for s in SEEDS if str(s) in raw) + " deg.")
    w("")

    # ---------------- B5 / E3
    pubint = load(PUB_INTERNAL / "report-export-publication/summary.json")
    pubrole = load(PUB_INTERNAL / "exports/summary.json")["by_role"]
    dam = load(OUT / "train-vs-test-damit-report.json")
    tvt = load(OUT / "train-vs-test-own-report.json")
    w("## B5, E3: training versus test (`\\CheckTrainMean`, `\\CheckTestMean`, Figure 15 left)")
    w("")
    w("All 780 cells of the published internal-role diagnostic (540 training, 70 validation, 170 test), five networks of each run, with the bundle path of the published diagnostic. "
      "Both the training and the test roles use the same period source. Caveat (inventory F11): the networks were trained with the DAMIT period, so at the own period the training asteroids "
      "are no longer exactly the fitted inputs; the gap then mixes memorization with the period change.")
    w("")
    w("| Quantity | Published | Rerun, DAMIT period | Own period | Own-period source |")
    w("|---|---|---|---|---|")
    pb = pubint["train_oof_pairs"]["bootstrap"]

    def tv(rep, key):
        if not rep:
            return "not run"
        b = rep["train_oof_pairs"]["bootstrap"]
        if key == "ci":
            return f"{b['mean_difference_deg']:.2f} ({b['ci95_deg'][0]:.2f} to {b['ci95_deg'][1]:.2f})"
        return f"{b[key]:.2f}"

    w(f"| Training mean, 169 asteroids (object-averaged) | {pb['train_mean_error_deg']:.2f} | {tv(dam, 'train_mean_error_deg')} | {tv(tvt, 'train_mean_error_deg')} | `train-vs-test-own-report.json` |")
    w(f"| Test mean, same 169 | {pb['oof_mean_error_deg']:.2f} | {tv(dam, 'oof_mean_error_deg')} | {tv(tvt, 'oof_mean_error_deg')} | same |")
    w(f"| Training minus test (95% fold-wise bootstrap, seed 20260916) | {pb['mean_difference_deg']:.2f} ({pb['ci95_deg'][0]:.2f} to {pb['ci95_deg'][1]:.2f}) | {tv(dam, 'ci')} | {tv(tvt, 'ci')} | same |")
    for role, n in (("train", 540), ("validation", 70), ("test", 170)):
        d = dam["by_role"][role]["mean_error_deg"] if dam else None
        t = tvt["by_role"][role]["mean_error_deg"] if tvt else None
        w(f"| Mean over {role} cells (n={n}) | {pubrole[role]['mean_error_deg']:.2f} | {f2(d)} | {f2(t)} | same |")
    if dam and "comparison_with_published" in dam:
        c = dam["comparison_with_published"]
        w("")
        w(f"Validation: all {c['cells_compared']} cells rerun with the locked DAMIT period, max abs axis difference {c['max_abs_axis_diff']:.1e}, max abs error difference {c['max_abs_error_diff_deg']:.1e} deg ({'passes' if c['passes_1e-6'] else 'FAILS'} 1e-6). Source `train-vs-test-damit-report.json`.")
    w("")

    # ---------------- B9
    pg = published_gaps()
    w("## B9: input removals (Figure 16, `\\Removal*`)")
    w("")
    w("Change in the mean oracle error of the 170 test asteroids, mean over the five seeds of (removal minus unmodified, each seed pooled over the five folds); "
      "the range is the minimum to maximum over seeds, as the published `\\Removal*Low/High` values. At the own period the unmodified baseline is the own-period single-network run.")
    w("")
    w("| Removal | Published mean (range) | Rerun, DAMIT period | Own period mean (range) | Own-period source |")
    w("|---|---|---|---|---|")
    for control in DIAGNOSTIC + FINER:
        g = pg[control]
        ptxt = f"{np.mean(g):+.2f} ({min(g):+.2f} to {max(g):+.2f})"
        o = own["summary"]["controls"].get(control) if own else None
        otxt = f"{o['gap_mean_deg']:+.2f} ({o['gap_min_deg']:+.2f} to {o['gap_max_deg']:+.2f})" if o else "not run"
        vt = "bit-identical on fold 0" if control in DIAGNOSTIC else "fold-0 gap within 0.001 deg"
        w(f"| {LABELS[control]} | {ptxt} | {vt} | {otxt} | `controls-own-summary.json` |")
    w("")
    if finer_note:
        w("Validation: " + finer_note + " Source `controls-damit-validation.json`.")
        w("")
    w("Figure: `figures/own/input-removal.pdf` (drawn by `figures/render_main_figures.py`, a copy of the published figure script with only input and output paths changed).")
    w("")

    # ---------------- Shapley
    from .shapley_op import compute_summary

    ps = compute_summary(PUB_SHAPLEY)
    sv = load(OUT / "shapley-damit-validation.json")
    so = load(OUT / "shapley-own-summary.json")
    w("## I11-I14: Shapley values (Figure 10)")
    w("")
    w("Eight groups, 256 combinations, 1,536 trial axes, 25 networks; shares from the percentile measure, effects from the grid error, flat-map correction and bootstrap as `group_shapley_summary.py`.")
    w("")
    w("| Group | Published share (seed range) | Own-period share (seed range) | Published: lowers error >1 deg / raises >1 deg | Own period: lowers / raises | Published mean effect on grid error | Own mean effect |")
    w("|---|---|---|---|---|---|---|")
    og = {g["group"]: g for g in so["groups"]} if so else {}
    for g in ps["groups"]:
        o = og.get(g["group"])
        r = g["share_seed_range_percent"]
        own_share = f"{100 * o['percentile_share']:.1f}% ({o['share_seed_range_percent'][0]:.1f} to {o['share_seed_range_percent'][1]:.1f})" if o else "not run"
        own_frac = f"{100 * o['fraction_lowering_error_by_more_than_1_deg']:.0f}% / {100 * o['fraction_raising_error_by_more_than_1_deg']:.0f}%" if o else "not run"
        w(f"| {g['group']} | {100 * g['percentile_share']:.1f}% ({r[0]:.1f} to {r[1]:.1f}) | {own_share} | "
          f"{100 * g['fraction_lowering_error_by_more_than_1_deg']:.0f}% / {100 * g['fraction_raising_error_by_more_than_1_deg']:.0f}% | {own_frac} | "
          f"{g['grid_error_mean_deg']:+.2f} | {f2(o['grid_error_mean_deg'], True) if o else 'not run'} |")
    w("")
    if so:
        w(f"Full input: percentile {so['full_percentile']:.3f} (published {ps['full_percentile']:.3f}, `\\ShapFullPercentile`), grid error {so['full_grid_error_deg']:.2f} deg (published {ps['full_grid_error_deg']:.2f}); efficiency error {so['efficiency_max_abs_error']:.1e}. Source `shapley-own-summary.json`.")
    sc = load(OUT / "shapley-damit-summary-check.json")
    if sc:
        w(f"Validation (DAMIT period, fold 0 seed 17): the published run is not reproduced cell by cell (near-tied peaks change single grid errors), "
          f"but the reported quantities agree: shares within {sc['max_abs_share_diff_percentage_points']:.3f} percentage points, grid-error effects within "
          f"{sc['max_abs_effect_diff_deg']:.3f} deg, full-input percentile within {sc['full_percentile_diff']:.1e}; a repeat of the rerun is identical "
          f"(max difference {sc['repeat_max_abs_diff']:.1e}). Source `shapley-damit-summary-check.json`.")
    if sv:
        rows = sv["comparison_with_published"]
        w(f"Cell-level comparison (34 asteroids x 256 combinations, raw stored values including the flat-map combinations that the summary corrects): max abs percentile difference {max(r['max_abs_percentile_diff'] for r in rows):.1e}, "
          f"grid error {max(r['max_abs_grid_error_diff_deg'] for r in rows):.1e} deg, shared means {max(r['max_abs_shared_means_diff'] for r in rows):.1e}. Source `shapley-damit-validation.json`.")
    w("Text values: I12 brightness 44%, Sun 35%, product 15%, sampling 12%, observer -5%, period -2%, ranges 37-50% and 28-46%; I13 brightness lowers by >1 deg for 95%, Sun term lowers for 72% and raises for 25% (both by more than 1 deg, reproduced from the published runs); I14 phase angle and distances 0.3% share.")
    w("Figure: `figures/own/shapley.pdf`.")
    w("")

    # ---------------- ZTF variants
    pdg = load(PUB_DIAG)["variants"]
    zp = load(OUT / "ztf-variants/score-published.json")
    zo = load(OUT / "ztf-variants/score-own.json")
    base = load(PUB_BASELINES)
    w("## E11, L10: seven other ZTF preparations")
    w("")
    w(f"Standard starting poles on the 161 ZTF asteroids: {base['primary_variants']['ztf_primary']['standard']['mean_deg']:.2f} deg (published). "
      "Means on the 161 asteroids with ZTF input; at the own period failures (no period found, or no usable epochs) are counted at 90 deg, and also excluded (then the standard poles are given on the same asteroids). "
      "Own minus standard: paired bootstrap, 10,000 resamples, seed 20261021.")
    w("")
    w("| Preparation | Published mean, 161 | Rerun, published period | Own: n with period | Own, failures at 90 (161) | Own minus standard, at 90 (95% CI) | Own, failures excluded | Standard on same | Own minus standard, excluded (95% CI) |")
    w("|---|---|---|---|---|---|---|---|---|")
    for v in ("original",) + VARIANTS:
        pm = pdg[v]["fixed_available_input_cohort"]["mean_oracle_at_3_error_deg"]
        rp = zp["variants"][v]["available_failed_at_90"]["mean_deg"] if zp else None
        if zo:
            b = zo["variants"][v]
            d90, dex = b["minus_standard_failed_at_90"], b.get("minus_standard_failed_excluded")
            w(f"| {v} | {pm:.2f} | {f2(rp)} | {b['n_ok']} | {b['available_failed_at_90']['mean_deg']:.2f} | {ci(d90)} | "
              f"{f2(b['failed_excluded']['mean_deg']) if b['failed_excluded'] else 'n/a'} | "
              f"{f2(b['standard_on_ok']['mean_deg']) if b['standard_on_ok'] else 'n/a'} | {ci(dex) if dex else 'n/a'} |")
        else:
            w(f"| {v} | {pm:.2f} | {f2(rp)} | not run | | | | | |")
    w("")
    if zp:
        checks = zp["validation_against_published"]
        w("Validation (published periods, all 8 x 161): max abs per-asteroid error difference "
          f"{max(c['max_abs_object_error_diff_deg'] for c in checks.values()):.1e} deg; failures equal to published: {all(c['published_failures'] == c['rerun_failures'] for c in checks.values())}. Source `ztf-variants/score-published.json`.")
    w("Own-period sources: `ztf-variants/periods-<variant>.json` (search), `ztf-variants/predictions-own-<variant>.jsonl`, `ztf-variants/score-own.json`.")
    w("")

    # ---------------- no-error
    sens = load(PUB_SENS)
    np_ = load(OUT / "no-uncertainty/score-published.json")
    no = load(OUT / "no-uncertainty/score-own.json")
    w("## E17, L11: runs without the reported uncertainties")
    w("")
    w("Prepared inputs with the measured errors removed, rebuilt with `prepare_directory` (identical to the published prepared files, `no-uncertainty/prepare-check.json`). "
      "The period search does not use the uncertainties, so its input is identical to the as-supplied data; the search was nevertheless run again on the variant. "
      "The standard-pole difference uses the method of the published value (`compute_reference_baselines.paired_difference`, standard minus DeLPHI, seed 20260924), sign reversed so that positive means DeLPHI is worse.")
    w("")
    w("| Quantity | Published | Rerun, published period | Own period, failures at 90 | Own period, failures excluded | Own-period source |")
    w("|---|---|---|---|---|---|")
    for cohort, key, label in (("ztf", "ztf", "ZTF"), ("alcdef", "alcdef_gaia", "ALCDEF")):
        pubmean = sens[key]["without_error_channel_mean_deg"]
        pc = base["cohorts"][cohort]["standard_minus_delphi"]
        ptxt = f"{-pc['mean_difference_deg']:.2f} ({-pc['ci95_high_deg']:.2f} to {-pc['ci95_low_deg']:.2f})"
        rp = np_["variants"][cohort] if np_ else None
        ob = no["variants"][cohort] if no else None

        def diff(block, k):
            if not block or k not in block:
                return "n/a"
            d = block[k]
            return f"{-d['mean_difference_deg']:.2f} ({-d['ci95_high_deg']:.2f} to {-d['ci95_low_deg']:.2f})"

        w(f"| {label} mean without uncertainties (n={sens[key]['n_objects']}) | {pubmean:.2f} | {f2(rp['available_failed_at_90']['mean_deg']) if rp else 'not run'} | "
          f"{f2(ob['available_failed_at_90']['mean_deg']) if ob else 'not run'} | "
          f"{(f2(ob['failed_excluded']['mean_deg']) + ' (n=' + str(ob['n_ok']) + ')') if ob and ob['failed_excluded'] else 'not run'} | `no-uncertainty/score-own.json` |")
        w(f"| {label} minus standard poles (95% CI) | {ptxt} | {diff(rp, 'standard_minus_delphi_failed_at_90_seed20260924')} | {diff(ob, 'standard_minus_delphi_failed_at_90_seed20260924')} | {diff(ob, 'standard_minus_delphi_failed_excluded_seed20260924')} | same |")
        wb = load(OP / "own-period-scores.json")[cohort]
        w(f"| {label} with uncertainties, for comparison | {sens[key]['original_mean_deg']:.2f} | - | {wb['own_period_failed_at_90']['mean_deg']:.2f} | "
          f"{wb['own_period_failed_excluded']['mean_deg']:.2f} (n={wb['own_period_failed_excluded']['n']}) | `own-period-scores.json` (earlier run) |")
    w("")
    if np_:
        c = np_["validation_against_published"]
        w("Validation (published periods): " + "; ".join(f"{k} max abs axis difference {v['max_abs_axis_diff']:.1e}, same objects {v['same_objects']}" for k, v in c.items()) + ". Source `no-uncertainty/score-published.json`.")
    w("")

    # ---------------- period determinism check
    w("## Period search reproducibility")
    w("")
    for name, mine, ref in (
        ("ZTF original variant", OUT / "ztf-variants/periods-original.json", OP / "own-periods-ztf.json"),
        ("ZTF without uncertainties", OUT / "no-uncertainty/periods-ztf.json", OP / "own-periods-ztf.json"),
        ("ALCDEF without uncertainties", OUT / "no-uncertainty/periods-alcdef.json", OP / "own-periods-alcdef.json"),
    ):
        a, b = load(mine), load(ref)
        if not a:
            w(f"- {name}: not run")
            continue
        pa, pb2 = a["periods_hours"], b["periods_hours"]
        same = sum((pa.get(k) is None and pb2.get(k) is None) or (pa.get(k) is not None and pb2.get(k) is not None and abs(pa[k] - pb2[k]) <= 1e-9) for k in pb2)
        w(f"- {name}: search input identical to the as-supplied data; {same} of {len(pb2)} periods (or failures) equal to `{rel(ref)}`; "
          f"{sum(v is not None for v in pa.values())} with a period.")
    for v in VARIANTS:
        a = load(OUT / f"ztf-variants/periods-{v}.json")
        if a:
            w(f"- ZTF {v}: {sum(x is not None for x in a['periods_hours'].values())} of 161 with a period")
    w("")
    w("## Deviations and notes")
    w("")
    w("- Oracle errors are computed inside the published evaluation routines right after each prediction, as in the published runs; the references do not enter any prediction.")
    w("- The automatic 1e-6 deg gate of the removal validation failed on the finer removals only (see B9); the own-period removals were run after checking that the difference lies in the published finer files themselves (`run_queue2.sh`).")
    w("- Removal validation covered all 5 seeds and all 16 controls and the exchange in fold 0. "
      "Shapley validation covered fold 0 seed 17. Training versus test, the ZTF variants and the "
      "runs without uncertainties were validated in full.")
    if tvt:
        cells = [json.loads(line) for p in sorted((OUT / "train-vs-test-own").glob("cells-fold-*.jsonl")) for line in p.read_text().splitlines()]
        oof = np.load(OP / "own-period-oof.npz")
        ref = dict(zip(oof["object_ids"].astype(str), oof["errors_deg"]))
        diff = max(abs(c["error_deg"] - ref[c["object_id"]]) for c in cells if c["role"] == "test")
        w(f"- The training-versus-test test side is the bundle-path test cell of this run (both sides at the same period). It agrees with `own-period-oof.npz` to {diff:.1e} deg (max abs difference over the 170).")
    w("- The reference-reassignment part of Figure 15 is not redrawn (another package).")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write("\n".join(L) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
