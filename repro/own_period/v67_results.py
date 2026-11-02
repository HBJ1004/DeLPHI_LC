"""Write v67/RESULTS.md from the score files (run after v67_queue.sh has finished).

    python repro/own_period/v67_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from repro.own_period import DataPath

ROOT = DataPath("../../..")
OP = DataPath()
V = OP / "v67"
PUB_V6 = ROOT / "paper/k3-reproduction/publication-summary.json"
PUB_V7 = ROOT / "paper/k3-followup/transfer/followup-summary.json"


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def f(x, n=3):
    return "n/a" if x is None else f"{x:.{n}f}"


def main() -> int:
    pv6 = load(PUB_V6)["downstream"]["verdict"]
    pv7 = load(PUB_V7)["broad_grid"]
    p7 = pv7["analyses"]["cold_step20_primary"]
    pa7 = pv7["angular_comparison_step20"]
    v6val = load(V / "v6-validation-damit/validation-vs-published.json")
    v6chk = load(V / "v6-scorer-check-published.json")["verdict_published_convention"]
    v7c = load(V / "v7-validation-classical-cpu/validation-vs-published.json")
    v7g = load(V / "v7-validation-guided-gpu/validation-vs-published.json")
    v7chk = load(V / "v7-scorer-check-published.json")
    v6 = load(V / "v6-own-period-score.json")
    v7 = load(V / "v7-own-period-score.json")
    cross = load(V / "v7-own-period/crosscheck-own-period-full-grid.json")
    rounding = load(V / "own-periods-170-rounded6.json")["rounding"]
    L = []
    w = L.append
    w("# V6 and V7 at our own period (package v67)\n")
    w("Scripts `repro/own_period/v67_periods.py`, "
      "`v6_fixed_period.py`, `v7_grid.py`, `v67_queue.sh`, `v67_results.py`. Exploratory revision analysis on the development sample.\n")
    w("## Periods\n")
    w(f"Top `fourier_gpu` period of `own-periods-170.json`, rounded to six decimals for every run of both tests "
      f"(`v67/own-periods-170-rounded6.json`): {rounding['objects_changed']} of {rounding['objects']} changed, largest change "
      f"{rounding['largest_change_hours']:.2e} h ({rounding['largest_relative_change']:.1e} relative). Identical to the periods of "
      f"`benchmark/evaluation-lock.json` for its {rounding['matches_benchmark_lock_objects']} asteroids. Both arms of both tests use these "
      "periods; V7's DeLPHI networks also receive them. V6's DeLPHI starts are the refined axes of `own-period-oof.npz` "
      "(the protocol's own-period candidates, predicted at the unrounded period, which differs by < 5e-7 h).\n")

    w("## V6: six DeLPHI starts vs six standard starts (v1.0.0 downstream test, 170 asteroids)\n")
    w("Same code path as published (`lc_pipeline/k3/downstream.py` `_run_record`, unchanged): DAMIT convexinv 0.2.1 binary sha 7965542f "
      "(unpatched; storage limits make some asteroids fail in every arm, as published), 3600 s timeout, serial, same object and arm order; "
      "DeLPHI arm time includes the neural inference time stored in the published ensemble artifact (as published; the network cost does not depend on the period).\n")
    w("| | speed ratio (classical/DeLPHI) | lower 95% | geo-mean RMS ratio | RMS upper 95% | recovery difference | paired successes | source |")
    w("|---|---|---|---|---|---|---|---|")
    w(f"| published (DAMIT period) | {f(pv6['speed_ratio'],4)} | {f(pv6['speed_ratio_ci95_low'],4)} | {f(pv6['geometric_mean_rms_ratio'],4)} | "
      f"{f(pv6['geometric_mean_rms_ratio_ci95_high'],4)} | {f(pv6['recovery_rate_inferiority'],4)} | {pv6['paired_successful_objects']} | `paper/k3-reproduction/publication-summary.json` downstream.verdict |")
    w(f"| our scorer on the published records | {f(v6chk['speed_ratio'],4)} | {f(v6chk['speed_ratio_ci95_low'],4)} | {f(v6chk['geometric_mean_rms_ratio'],4)} | "
      f"{f(v6chk['geometric_mean_rms_ratio_ci95_high'],4)} | {f(v6chk['recovery_rate_inferiority'],4)} | {v6chk['paired_successful_objects']} | `v67/v6-scorer-check-published.json` |")
    c = load(V / "v6-scorer-check-published.json")["verdict_with_period_search_time_added_to_both_arms"]
    w(f"| published records + our period-search time (n={c['objects']}) | {f(c['speed_ratio'],4)} | {f(c['speed_ratio_ci95_low'],4)} | | | | | `v67/v6-scorer-check-published.json` |")
    if v6:
        for label, key in (("own period, published convention (no valid fit = not recovered)", "verdict_published_convention"),
                           ("own period, asteroids without a valid fit in an arm excluded", "verdict_objects_with_valid_fits_in_both_arms"),
                           ("own period, period-search time added to both arms", "verdict_with_period_search_time_added_to_both_arms")):
            v = v6[key]
            w(f"| {label} (n={v['objects']}) | {f(v['speed_ratio'],4)} | {f(v['speed_ratio_ci95_low'],4)} | {f(v['geometric_mean_rms_ratio'],4)} | "
              f"{f(v['geometric_mean_rms_ratio_ci95_high'],4)} | {f(v['recovery_rate_inferiority'],4)} | {v['paired_successful_objects']} | `v67/v6-own-period-score.json` |")
        w("")
        w(f"Verdict at own period: passed = {v6['verdict_published_convention']['passed']}; failures {v6['verdict_published_convention']['failures']}.")
        a = v6["arms"]
        w(f"Arms at own period: standard {a['baseline']['wall_seconds_mean']:.2f} s/asteroid, {a['baseline']['recovered_within_20']} recovered within 20 deg, "
          f"{a['baseline']['objects_without_valid_fit']} without a valid fit; DeLPHI {a['candidate']['wall_seconds_mean']:.2f} s/asteroid, "
          f"{a['candidate']['recovered_within_20']} recovered, {a['candidate']['objects_without_valid_fit']} without a valid fit. "
          f"Mean selected error (valid fits / failures at 90 deg): standard {f(a['baseline']['selected_error_mean_valid_deg'],2)} / "
          f"{f(a['baseline']['selected_error_mean_failures_at_90_deg'],2)}, DeLPHI {f(a['candidate']['selected_error_mean_valid_deg'],2)} / "
          f"{f(a['candidate']['selected_error_mean_failures_at_90_deg'],2)} deg.")
        ps = v6["period_search"]
        w(f"Period-search time: field `{ps['field']}` of `{Path(ps['file']).name}` (fourier_gpu, warm), mean {ps['mean_seconds']:.3f} s over {ps['objects_with_time']} asteroids, added per asteroid to both arms.\n")
    else:
        w("\nOwn-period run not finished yet (queued in `v67_queue.sh`).\n")
    w(f"Validation (DAMIT period, 20 asteroids: the first four test asteroids of each fold, untimed, CPU): {v6val['starts_compared']} starts, "
      f"{v6val['starts_bit_identical_outputs']} with bit-identical solver outputs, max angle {v6val['max_angle_deg_atan2']} deg (atan2), "
      f"max RMS difference {v6val['max_abs_rms_difference']}; passed = {v6val['passed']} (`v67/v6-validation-damit/validation-vs-published.json`).\n")

    w("## V7: about 20 starts near the DeLPHI axes vs the 146-start grid (170 asteroids, cold, three repeats)\n")
    w("Same worker code (`lc_pipeline/grid_benchmark.py`, unchanged), expanded solver sha 4df011a5, tolerance 0.0003, iteration cap 1000, 300 s per start, "
      "six solver threads, fresh process per case, published case order. Only the cold `guided20` and `classical20` cases were run "
      "(the published V7 numbers use only these); the published study also ran standard6, classical15, guided15 and a warm mode, interleaved, which were left out.\n")
    w("| | speed ratio | 95% interval | RMS ratio | 95% interval | guided minus classical mean error (deg) | recovered (classical/guided) | source |")
    w("|---|---|---|---|---|---|---|---|")
    w(f"| published (DAMIT period) | {f(p7['runtime_point_ratio'],4)} | {f(p7['runtime_interval_95']['lower'],3)}-{f(p7['runtime_interval_95']['upper'],3)} | "
      f"{f(p7['rms_point_ratio'],4)} | {f(p7['rms_interval_95']['lower'],4)}-{f(p7['rms_interval_95']['upper'],4)} | {f(pa7['guided_minus_classical_mean_error_deg'],3)} | "
      f"{p7['baseline_success_count']}/{p7['candidate_success_count']} | `paper/k3-followup/transfer/followup-summary.json` broad_grid |")
    c = v7chk["cold_step20_primary"]
    w(f"| our scorer on the published cases | {f(c['runtime_point_ratio'],4)} | {f(c['runtime_interval_95']['lower'],3)}-{f(c['runtime_interval_95']['upper'],3)} | "
      f"{f(c['rms_point_ratio'],4)} | {f(c['rms_interval_95']['lower'],4)}-{f(c['rms_interval_95']['upper'],4)} | "
      f"{f(v7chk['angular_comparison_step20']['guided_minus_classical_mean_error_deg'],3)} | {c['baseline_success_count']}/{c['candidate_success_count']} | `v67/v7-scorer-check-published.json` |")
    c = v7chk["cold_step20_with_period_search_time_added"]
    w(f"| published cases + our period-search time | {f(c['runtime_point_ratio'],4)} | {f(c['runtime_interval_95']['lower'],3)}-{f(c['runtime_interval_95']['upper'],3)} | | | | | `v67/v7-scorer-check-published.json` |")
    if v7:
        for label, key, ang in (("own period, published convention (angular: no fit = 90 deg)", "cold_step20_primary", "angular_comparison_step20"),
                                ("own period, asteroids without any fit in an arm excluded", "cold_step20_failures_excluded", "angular_comparison_step20_failures_excluded"),
                                ("own period, period-search time added to every case", "cold_step20_with_period_search_time_added", None)):
            c = v7[key]
            rms = c["rms_interval_95"]
            w(f"| {label} (n={c['objects']}) | {f(c['runtime_point_ratio'],4)} | {f(c['runtime_interval_95']['lower'],3)}-{f(c['runtime_interval_95']['upper'],3)} | "
              f"{f(c['rms_point_ratio'],4)} | {(f(rms['lower'],4)+'-'+f(rms['upper'],4)) if rms else 'n/a (support '+str(c['rms_object_support'])+')'} | "
              f"{f(v7[ang]['guided_minus_classical_mean_error_deg'],3) if ang else ''} | {c['baseline_success_count']}/{c['candidate_success_count']} | `v67/v7-own-period-score.json` |")
        a = v7["angular_comparison_step20"]
        w("")
        w(f"Own period: mean seconds classical {v7['cold_step20_primary']['classical_mean_seconds']:.2f}, guided {v7['cold_step20_primary']['guided_mean_seconds']:.2f}; "
          f"mean selected error classical {a['classical_mean_error_deg']:.2f}, guided {a['guided_mean_error_deg']:.2f} deg; "
          f"guided lower/equal/higher {a['guided_lower_equal_higher_error_counts']}; interval of the difference {a['interval_95_deg']}. "
          f"Asteroids without any fit in an arm: {v7['objects_without_any_fit_in_an_arm']}. Valid starts {v7['cold_step20_primary']['valid_starts']}.")
    else:
        w("\nOwn-period run not finished yet (queued in `v67_queue.sh`).\n")
    w("")
    w(f"Validation (published periods, 20 asteroids, repeat 0, untimed): classical20 on CPU with two solver threads, {v7c['cases']} cases, every start "
      f"identical to the published records (max angle {v7c['max_selected_angle_deg_atan2']} deg); passed = {v7c['passed']}. ")
    if v7g:
        w(f"guided20 on the GPU: {v7g['cases']} cases, max prediction angle {v7g['max_prediction_angle_deg']:.2e} deg, max selected angle "
          f"{v7g['max_selected_angle_deg_atan2']:.2e} deg, mismatches {v7g['mismatches']}; passed = {v7g['passed']}. ")
    w("(On the CPU the network axes differ from the GPU ones by 0.006 deg, so the guided validation runs on the GPU.)")
    if cross:
        w(f"\nCross-check: the classical20 fits of all three repeats equal the own-period full grid of `benchmark/full-grid` for its 140 asteroids "
          f"({cross['compared_starts']} start fits compared, max angle {cross['max_angle_deg_atan2']:.2e} deg, differing {len(cross['differing_starts'])}).")
    w("\nTimes differ from the published runs because they were measured on another day and machine state; "
      "only ratios within a run are compared. No reference pole was opened before the fits of a run were written.")
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else V / "RESULTS.md"
    with out.open("x") as stream:
        stream.write("\n".join(L) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
