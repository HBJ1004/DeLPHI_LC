"""Validation gate and RESULTS.md for the own-period reliability package.

    python -m repro.own_period.reliability_report gate
    python -m repro.own_period.reliability_report results

``gate`` exits with status 1 unless the DAMIT-period reruns reproduce the
published values (prediction axes within 1e-6 deg, scoring code identical,
withheld fits identical).  ``results`` writes ``RESULTS.md`` (write-once).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys

import numpy as np

from repro.own_period import reliability as R
from repro import network_root as network

PUB_GRID = R.FOLLOWUP / "data/reliability-publication-20260916-final/k3_data_degradation.csv"
PUB_TABLE = R.RUN / "sampling-table.csv"
PUB_BASE = R.FOLLOWUP / "data/interpretation-20260930/sampling-baselines.json"
PUB_SUPP = R.FOLLOWUP / "data/reliability-supplement-20260916-final-v8/k3_supplement_numbers.json"
from repro.own_period import DataPath

OWN_SCORES = DataPath('own-period-scores-main.json')
VALIDATION_IDS = ("asteroid_3521", "asteroid_1727")


def withheld_validation() -> dict:
    out = {}
    for oid in VALIDATION_IDS:
        for arm in R.WITHHELD_ARMS:
            pub = json.loads((R.RUN / "fits" / oid / "full" / arm / "result.json").read_text())["payload"]
            new_path = R.withheld_root("damit") / "fits" / oid / "full" / arm / "result.json"
            if not new_path.exists():
                out[f"{oid}/{arm}"] = {"status": "missing"}
                continue
            new = json.loads(new_path.read_text())["payload"]
            value = {"published_rms": pub["selected_fit"]["final_relative_rms"],
                     "rerun_rms": new["selected_fit"]["final_relative_rms"],
                     "published_start": pub["selected_fit"]["start_index"],
                     "rerun_start": new["selected_fit"]["start_index"],
                     "same_input": pub["input_sha256"] == new["input_sha256"],
                     "starts": [pub["starts_requested"], new["starts_requested"]]}
            if arm == "k3":
                a = json.loads((R.RUN / "fits" / oid / "full" / "k3" / "prediction.json").read_text())["payload"]["axes"]
                b = json.loads((new_path.parent / "prediction.json").read_text())["payload"]["axes"]
                value["axes_angle_deg"] = R.axial_angle_deg(a, b)
            value["identical"] = (value["published_rms"] == value["rerun_rms"] and value["same_input"]
                                  and value["published_start"] == value["rerun_start"]
                                  and value.get("axes_angle_deg", 0.0) <= 1e-6)
            out[f"{oid}/{arm}"] = value
    return out


def command_gate(args) -> int:
    if network.switched():
        raise ValueError('the publication parity gate belongs to the old networks; use Check 3 separately')
    prediction = json.loads((R.OUT / "validation-damit-period.json").read_text())
    scoring = json.loads((R.OUT / "validation-scoring.json").read_text())
    withheld = withheld_validation()
    checks = {
        # Bit-identical axes pass outright; validation-damit-period.json stored an arccos
        # angle whose rounding reaches ~1.7e-6 deg even for identical vectors.
        "prediction_axes_within_1e-6_deg": not prediction["status_or_input_mismatches"]
        and (prediction["bit_identical_axes"] == prediction["compared_cells"]
             or prediction["max_axis_angle_deg"] <= 1e-6),
        "sampling_table_identical": not scoring["sampling_table_mismatched_conditions"],
        "data_degradation_identical": scoring["data_degradation_max_abs_difference"] <= 1e-9,
        "sampling_baselines_identical": scoring["sampling_baselines_max_abs_difference"] <= 1e-9,
        "withheld_scoring_identical": scoring["withheld_table_max_abs_difference"] <= 1e-12,
        "withheld_fits_identical": all(v.get("identical") for v in withheld.values()),
    }
    record = {"checks": checks, "withheld_fit_validation": withheld}
    # The first gate record (arccos check, failed) is kept; this one is written once.
    target = R.OUT / "validation-gate-v2.json"
    if not target.exists():
        target.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record, indent=1))
    return 0 if all(checks.values()) else 1


# ----------------------------------------------------------------------------- RESULTS.md

def fmt(x, d=2):
    if x is None:
        return "n/a"
    if isinstance(x, str):
        return x
    return f"{x:.{d}f}"


def pct(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def ci(lo, hi, d=2):
    return f"({fmt(lo, d)} to {fmt(hi, d)})"


def command_results(args) -> int:
    network.guard_output(R.OUT)
    if network.switched():
        source = R.OUT / 'score/reliability-own-period.json'
        network.check_json(source)
        score = json.loads(source.read_text())
        lines = ['# Selected-network reliability rerun', '',
                 f"Network binding: `{network.json_metadata()['network_binding_sha256']}`.", '',
                 'Saved reduced/noisy inputs, seeds, periods, retained-data searches and classical20 fits are reused.',
                 'Five test-fold networks per object; guided fits rerun with the fixed 15.8 degree radius.',
                 'Historical validation and runtime receipts remain old evidence. No publication parity is claimed.', '',
                 '| Condition | Mean, failures at 90 deg | Mean, failed excluded |', '|---|---:|---:|']
        for condition, value in sorted(score['failed_at_90']['sampling_summary'].items()):
            excluded = score['failed_excluded']['conditions'][condition]
            lines.append(f"| {condition} | {fmt(value.get('mean_error_deg'))} | {fmt(excluded.get('mean_error_deg'))} |")
        lines.extend(['', 'Withheld comparisons (fresh guided arm, fixed classical arm):', '',
                      '```json', json.dumps(score['withheld'], indent=2), '```', '',
                      'Period-sensitivity results are rerun separately under `results/period-sensitivity`.'])
        target = R.OUT / 'RESULTS.md'
        with target.open('x') as stream:
            stream.write('\n'.join(lines) + '\n')
        return 0
    score = json.loads((R.OUT / "score" / "reliability-own-period.json").read_text())
    f90, fex = score["failed_at_90"], score["failed_excluded"]
    s90 = f90["sampling_summary"]
    ex = fex["conditions"]
    pub_table = {r["condition"]: r for r in csv.DictReader(PUB_TABLE.open())}
    pub_grid = {r["condition"]: r for r in csv.DictReader(PUB_GRID.open())}
    own_grid90 = {r["condition"]: r for r in f90["data_degradation"]}
    own_gridex = {r["condition"]: r for r in fex["data_degradation"]}
    pub_base = json.loads(PUB_BASE.read_text())
    b90, bex = f90["sampling_baselines"], fex["sampling_baselines"]
    periods = score["period_search"]
    validation = json.loads((R.OUT / "validation-damit-period.json").read_text())
    vscore = json.loads((R.OUT / "validation-scoring.json").read_text())
    gate = json.loads((R.OUT / "validation-gate-v2.json").read_text())
    commit = subprocess.run(["git", "-C", str(R.FOLLOWUP / "source-op-reliability"), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    L = []
    w = L.append
    w("# Reduced data, added noise and withheld lightcurves with our own period search\n")
    w("Package `reliability` of the own-period reproduction of Paper I (exploratory revision analysis on the "
      "development sample). Protocol: `../PROTOCOL.md`, Addendum 1. Code: worktree `DeLPHI-followup/source-op-reliability`, "
      f"branch `research/own-period-reliability` (commit {commit}), scripts `repro/own_period/reliability.py` (search, "
      "predictions, withheld fits), `reliability_score.py` (scoring), `reliability_report.py` (this file), "
      "`reliability_chunks.sh` (resumable GPU chunks).\n")
    w("## What was run\n")
    w("- The reduced and noisy data sets are the published ones: same sampling code, conditions and seeds "
      "(20260915, 20260916, 20260917), taken from the sealed lock `data/reliability-20260915/run-1/study.json`; "
      "same five networks per asteroid (the published fold bundles).")
    w("- On every reduced or noisy data set the period was searched again with "
      "`estimate(curves, \"fourier_gpu\", resolution=1, timeout=180, sparse_fallback=True)`, each retained lightcurve "
      "being one curve with magnitude -2.5 log10(flux), and the networks received the same data with that period. "
      "The complete data (`full`) use `own-periods-170.json`. Identical search inputs (for example `observation_cap-cap-None`, "
      "which equals the full data) reuse one search.")
    w("- A data set without a period is a failure: it is scored at 90 deg (published convention) and, separately, excluded.")
    w("- Withheld-lightcurve test: the period was searched on the retained data only (separately for all retained data and "
      "for the 10% thinning); the DeLPHI candidates and both convexinv arms (full 146-start grid and the search within "
      "15.8 deg of the candidates) use that period. The atlas arm of the published run is not in Table 12 and was not rerun.")
    w("- Period-sensitivity test (Figure 17): kept as published, not rerun (see the alias table below).\n")

    w("## Validation (DAMIT period)\n")
    w(f"- Predictions, fold 0 (all reduced-data and noise conditions, {validation['compared_cells']} cells): "
      f"{validation['bit_identical_axes']} bit-identical axes, largest axis difference {validation['max_axis_angle_deg']:.2e} deg as stored (an arccos rounding value; the axes are bit-identical, so the true difference is 0), "
      f"status or input mismatches: {len(validation['status_or_input_mismatches'])}. "
      "Per-condition fold-0 means: `validation-damit-period.json`.")
    w(f"- Scoring code on the published sealed rows: sampling table identical for {vscore['sampling_table_rows_compared']} "
      f"conditions (mismatches: {vscore['sampling_table_mismatched_conditions'] or 'none'}); Figure 8 table max difference "
      f"{vscore['data_degradation_max_abs_difference']:.1e}; sampling baselines max difference "
      f"{vscore['sampling_baselines_max_abs_difference']:.1e}; withheld table max difference "
      f"{vscore['withheld_table_max_abs_difference']:.1e} (`validation-scoring.json`).")
    w(f"- Withheld fits at the DAMIT period for {', '.join(VALIDATION_IDS)} (all retained data, both arms): "
      f"{'identical selected fits and candidates' if gate['checks']['withheld_fits_identical'] else 'NOT identical, see validation-gate-v2.json'}.")
    full = periods.get("_full_reproduces_own_periods_170", {})
    rel = full.get("max_relative_difference")
    w(f"- The search path on the complete data reproduces `own-periods-170.json`: {full.get('n_identical')} of {full.get('n')} "
      f"identical, largest relative difference {'n/a' if rel is None else f'{rel:.1e}'}, "
      f"missing {full.get('n_missing')}.\n")

    w("## Numbers printed in the manuscript\n")
    w("Own (90) = failures at 90 deg; Own (excl.) = failed data sets removed. Paired intervals are 95% bootstrap intervals "
      "with the published seeds.\n")
    w("| Item | Published | Own period (90) | Own period (excl.) | Source |")
    w("|---|---|---|---|---|")
    g = "two_d-block_count-{}-per_block_cap-{}"

    def grid_row(name, cond):
        p, o, e = pub_grid[cond], own_grid90[cond], own_gridex.get(cond, {})
        w(f"| {name} | {fmt(float(p['mean_error_deg']))} | {fmt(o['mean_error_deg'])} | {fmt(e.get('mean_error_deg'))} "
          f"(n={e.get('n_objects', 0)}) | `score/k3_data_degradation-own-*.csv` ({cond}) |")

    grid_row("S3 one group, mean (80)", g.format(1, "None"))
    w(f"| S3 all data, mean (80) | {fmt(float(pub_table['grid-full']['mean_error_deg']))} | "
      f"{fmt(s90['grid-full']['mean_error_deg'])} | {fmt(ex['grid-full'].get('mean_error_deg'))} "
      f"(n={ex['grid-full']['n_objects_with_success']}) | `score/sampling-table-own*.csv` (grid-full) |")
    for b in (1, 3, 5, 10):
        w(f"| S4 within 20 deg, {b} group(s) | {pct(pub_base['groups'][str(b)]['within20_mean'])} | "
          f"{pct(b90['groups'][str(b)]['within20_mean'])} | {pct(bex['groups'][str(b)]['within20_mean'])} | "
          "`score/reliability-own-period.json` sampling_baselines.groups |")
    for cap in (50, 20):
        grid_row(f"S6 ten groups, <= {cap} obs, mean", g.format(10, cap))
    c = g.format(10, 50)
    p, o, e = pub_grid[c], own_grid90[c], own_gridex[c]
    w(f"| S6 ten groups <= 50 obs, change vs all data | {fmt(float(p['paired_change_deg']))} "
      f"{ci(float(p['paired_ci95_low_deg']), float(p['paired_ci95_high_deg']))} | {fmt(o['paired_change_deg'])} "
      f"{ci(o['paired_ci95_low_deg'], o['paired_ci95_high_deg'])} | {fmt(e.get('paired_change_deg'))} "
      f"{ci(e.get('paired_ci95_low_deg'), e.get('paired_ci95_high_deg'))} | data-degradation tables |")
    for b in (5, 3, 1, 10):
        pv, ov, ev = (pub_base["standard_minus_delphi_deg"][str(b)], b90["standard_minus_delphi_deg"][str(b)],
                      bex["standard_minus_delphi_deg"][str(b)])
        w(f"| S7 standard minus DeLPHI, {b} group(s) | {fmt(pv[0])} {ci(pv[1], pv[2])} | {fmt(ov[0])} {ci(ov[1], ov[2])} | "
          f"{fmt(ev[0])} {ci(ev[1], ev[2])} (n={ev[3]}) | sampling_baselines.standard_minus_delphi_deg |")
    for cap in ("200", "500", "None"):
        pv, ov, ev = (pub_base["total_cap_standard_minus_delphi_deg"][cap], b90["total_cap_standard_minus_delphi_deg"][cap],
                      bex["total_cap_standard_minus_delphi_deg"][cap])
        w(f"| S8 standard minus DeLPHI, total <= {cap} obs (170) | {fmt(pv[1])} {ci(pv[2], pv[3])} | {fmt(ov[1])} "
          f"{ci(ov[2], ov[3])} | {fmt(ev[1])} {ci(ev[2], ev[3])} (n={ev[0]}) | sampling_baselines.total_cap_standard_minus_delphi_deg |")
    for sigma, name in (("0.03", "S9 noise 0.03 mag"), ("0.1", "S9 noise 0.1 mag")):
        cond = f"full-noise_sigma-{sigma}"
        w(f"| {name}, mean (170) | {fmt(float(pub_table[cond]['mean_error_deg']))} | {fmt(s90[cond]['mean_error_deg'])} | "
          f"{fmt(ex[cond].get('mean_error_deg'))} (n={ex[cond]['n_objects_with_success']}) | sampling-table ({cond}) |")
    w(f"| Complete data, mean (170) | {fmt(float(pub_table['full']['mean_error_deg']))} | {fmt(s90['full']['mean_error_deg'])} | "
      f"{fmt(ex['full'].get('mean_error_deg'))} | sampling-table (full) |")
    withheld = score.get("withheld") or {}
    pub_w = json.loads((R.RUN / "report.json").read_text())["payload"]["withheld"]
    for sampling, label in (("full", "all retained data"), ("thin10", "10% point thinning")):
        pc, pk = pub_w[f"{sampling}-classical20"]["withheld_rms"], pub_w[f"{sampling}-k3"]
        oc, ok = withheld.get(f"{sampling}-classical20", {}), withheld.get(f"{sampling}-k3", {})
        od = ok.get("candidate_minus_classical_rms") or {}
        w(f"| Table 12 {label}: classical / guided / difference | {fmt(pc['estimate'], 4)} / {fmt(pk['withheld_rms']['estimate'], 4)} / "
          f"{fmt(pk['candidate_minus_classical_rms']['estimate'], 4)} {ci(pk['candidate_minus_classical_rms']['bootstrap_lower_95'], pk['candidate_minus_classical_rms']['bootstrap_upper_95'], 4)} | "
          f"{fmt((oc.get('withheld_rms') or {}).get('estimate'), 4)} / {fmt((ok.get('withheld_rms') or {}).get('estimate'), 4)} / "
          f"{fmt(od.get('estimate'), 4)} {ci(od.get('bootstrap_lower_95'), od.get('bootstrap_upper_95'), 4)} "
          f"(n={oc.get('n_valid')}, pairs {ok.get('n_completed_pairs')}) | same (failed searches cannot be fitted) | "
          "`score/reliability-own-period.json` withheld |")
    w("")
    w("Published sources: Figure 8 cells `data/reliability-publication-20260916-final/k3_data_degradation.csv`; "
      "sampling table `data/reliability-20260915/run-1/sampling-table.csv`; S4, S7, S8 "
      "`data/interpretation-20260930/sampling-baselines.json`; Table 12 `data/reliability-20260915/run-1/report.json` (withheld). "
      "S5 (random 22%, standard 44%) is reference-only and unchanged.\n")

    w("## Figure 8 grid (80 asteroids, mean oracle error, deg)\n")
    w("Each cell: published / own (90) / own (excl.) [own failed predictions of 240].\n")
    caps = ("5", "10", "20", "50", "None")
    w("| groups \\ obs per group | " + " | ".join("all" if c == "None" else c for c in caps) + " |")
    w("|---|" + "---|" * len(caps))
    for b in ("1", "3", "5", "10", "None"):
        cells = []
        for cap in caps:
            cond = g.format(b, cap)
            cells.append(f"{fmt(float(pub_grid[cond]['mean_error_deg']), 1)} / {fmt(own_grid90[cond]['mean_error_deg'], 1)} / "
                         f"{fmt(own_gridex.get(cond, {}).get('mean_error_deg'), 1)} [{own_grid90[cond]['failed_predictions']}]")
        w(f"| {'all' if b == 'None' else b} | " + " | ".join(cells) + " |")
    w("\nFigures: `figures/data-degradation-own-failed90.pdf`, `figures/data-degradation-own-failed-excluded.pdf`, "
      "`figures/sampling-caps-own-failed90.pdf`, `figures/sampling-caps-own-failed-excluded.pdf` "
      "(drawn by the copy `figures/render_main_figures.py`).\n")

    w("## Figure 14: total observation caps (170 asteroids)\n")
    w("| Cap | Published mean (95%) | Own (90) mean (95%) | Own (excl.) mean (95%) | failed / cells |")
    w("|---|---|---|---|---|")
    for cap in ("10", "20", "50", "100", "200", "500", "None"):
        cond = f"observation_cap-cap-{cap}"
        p, o, e = pub_table[cond], s90[cond], ex[cond]
        w(f"| {'all' if cap == 'None' else cap} | {fmt(float(p['mean_error_deg']))} {ci(float(p['mean_error_bootstrap_lower_95']), float(p['mean_error_bootstrap_upper_95']))} | "
          f"{fmt(o['mean_error_deg'])} {ci(o['mean_error_bootstrap_lower_95'], o['mean_error_bootstrap_upper_95'])} | "
          f"{fmt(e.get('mean_error_deg'))} {ci(e.get('ci95_low_deg'), e.get('ci95_high_deg'))} | {e['n_failed_cells']} / {e['n_cells']} |")
    w("")

    w("## All conditions: published, own (90), own (excluded), period search\n")
    w("Period classes compare the period found with the DAMIT period (first solution): within 1% of P, P/2 or 2P, other, or failed. "
      "Search seconds are per distinct search (reused identical inputs not counted).\n")
    w("| Condition | Published | Own (90) | Own (excl.) | asteroids with a success | failed cells | period found | P | P/2 | 2P | other | failed | sparse fallback | mean s | max s |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cond in sorted(s90):
        p = pub_table.get(cond, {})
        e = ex.get(cond, {})
        ps = periods.get("full" if cond == "grid-full" else cond, {})
        cls = ps.get("classes", {})
        reg = ps.get("input_regimes", {})
        w(f"| {cond} | {fmt(float(p['mean_error_deg'])) if p else 'n/a'} | {fmt(s90[cond]['mean_error_deg'])} | "
          f"{fmt(e.get('mean_error_deg'))} | {e.get('n_objects_with_success')}/{e.get('n_eligible_objects')} | "
          f"{e.get('n_failed_cells')}/{e.get('n_cells')} | {pct(ps.get('success_rate'))} | {cls.get('P', 0)} | {cls.get('P/2', 0)} | "
          f"{cls.get('2P', 0)} | {cls.get('other', 0)} | {cls.get('failed', 0)} | {reg.get('sparse_survey_windows', 0)} | "
          f"{fmt(ps.get('search_seconds_mean'), 1)} | {fmt(ps.get('search_seconds_max'), 1)} |")
    w("")
    if (R.OUT / "search" / "periods-withheld.jsonl").exists():
        recs = R.read_jsonl(R.OUT / "search" / "periods-withheld.jsonl")
        from repro.own_period.reliability_score import damit_periods, period_class
        ref = damit_periods()
        for s in R.WITHHELD_SAMPLINGS:
            rs = [r for r in recs if r["condition"] == f"withheld-{s}"]
            counts = {}
            for r in rs:
                k = period_class(r["period_hours"], ref[r["object_id"]])
                counts[k] = counts.get(k, 0) + 1
            w(f"- Withheld test, retained data `{s}`: period found for {sum(r['period_hours'] is not None for r in rs)} of {len(rs)}; classes {counts}.")
        w("")

    w("## Period-sensitivity test (Figure 17): own-period result by alias class\n")
    w("Figure 17 is kept as published (it scales the DAMIT period by design). Beside it, the complete-data own-period "
      "result of the 170 asteroids split by how our period relates to the DAMIT period "
      "(`own-period-scores-main.json`, `main_sample.period_classes`):\n")
    classes = json.loads(OWN_SCORES.read_text())["main_sample"]["period_classes"]
    w("| Class of our period | Asteroids | Mean error, own period (deg) | Mean error, DAMIT period (deg) |")
    w("|---|---|---|---|")
    for k in ("P", "P/2", "2P", "other", "failed"):
        v = classes[k]
        w(f"| {k} | {v['n']} | {fmt(v.get('own_mean_deg'))} | {fmt(v.get('published_mean_deg'))} |")
    w("")
    w("## Compute cost (measured)\n")
    main_recs = R.read_jsonl(R.OUT / "search" / "periods-main.jsonl")
    distinct = [r for r in main_recs if r.get("reused_from") is None]
    pilot = main_recs[:400]
    pf = [r["seconds"] for r in pilot if r["condition"] == "full" and r.get("reused_from") is None]
    po = [r["seconds"] for r in pilot if r["condition"] != "full" and r.get("reused_from") is None]
    w(f"- Timing pilot (first 400 cells, one search at a time on the RTX 4070): complete data {fmt(np.mean(pf) if pf else None, 1)} s "
      f"per search (n={len(pf)}, max {fmt(max(pf) if pf else None, 1)} s); reduced or noisy data "
      f"{fmt(np.mean(po) if po else None, 1)} s (n={len(po)}).")
    w(f"- Main searches: {len(main_recs)} cells, {len(distinct)} distinct searches, "
      f"{sum(r['seconds'] for r in distinct) / 3600:.1f} h of summed search time "
      f"(run with 3 parallel workers after the pilot); timeouts: {sum(r['status'] == 'timeout' for r in main_recs)}.")
    own_pred = R.read_jsonl(R.OUT / "predictions" / "own.jsonl")
    w(f"- Network inference, own period: {len(own_pred)} cells, {sum(r['runtime_s'] for r in own_pred) / 3600:.2f} h.")
    fits_log = R.OUT / "withheld-fits-own.log"
    if fits_log.exists():
        secs = [float(x.split()[-1].rstrip("s")) for x in fits_log.read_text().splitlines()
                if x.startswith("asteroid_")]
        w(f"- Withheld convexinv fits, own period: {len(secs)} fit cells, {sum(secs) / 3600:.2f} h (6 workers).")
    w("")
    notes = (R.OUT / "NOTES.md")
    if notes.exists():
        w(notes.read_text())
    target = R.OUT / "RESULTS.md"
    with target.open("x") as stream:
        stream.write("\n".join(L) + "\n")
    print(target)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("gate")
    sub.add_parser("results")
    args = parser.parse_args()
    return {"gate": command_gate, "results": command_results}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
