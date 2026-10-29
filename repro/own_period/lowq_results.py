"""Write ``data/own-period-20261021/lowq/RESULTS.md`` from the lower-quality outputs.

    python -m repro.own_period.lowq_results

Published values, the DAMIT-period rerun of the first 500 asteroids, and the
own-period values (no period counted at 90 deg, and excluded) side by side,
with the source file of each number.  Written once.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from repro import compute_reference_baselines as base

from .lowq_search import FOLLOWUP

OUT = FOLLOWUP / "data/own-period-20261021/lowq"
REVISION = FOLLOWUP / "data/publication-revision-20260917-r2"
BASELINES = FOLLOWUP / "source/repro/baselines_out/reference-baselines.json"
MIRROR = FOLLOWUP / "data/interpretation-20260926/mirror-symmetry.json"
REL = "data/own-period-20261021/lowq"


def load(path: Path):
    return json.loads(path.read_text())


def f2(value, digits=2):
    return "-" if value is None else f"{value:.{digits}f}"


def interval(block):
    return (f"{block['mean_difference_deg']:.2f} [{block['ci95_low_deg']:.2f}, "
            f"{block['ci95_high_deg']:.2f}]")


def main() -> int:
    destination = OUT / "RESULTS.md"
    if destination.exists():
        raise FileExistsError(destination)
    own = load(OUT / "scores-own/summary.json")
    sub = load(OUT / "scores-damit-period-500/summary.json")
    check = load(OUT / "check-published-v2.json")
    validation = load(OUT / "validation-damit-period-500-check-v3.json")
    timing = load(OUT / "search-timing-sample.json")
    periods = load(OUT / "own-periods-lowq.json")
    pub_census = load(REVISION / "lowq-analysis.json")
    pub_lineage = load(REVISION / "lowq-lineage-analysis.json")
    pub_base = load(BASELINES)["cohorts"]
    pub_dir = load(MIRROR)["summary"]["lower_quality_direction"]
    exc = own["variants"]["failed_excluded"]
    at90 = own["variants"].get("failed_at_90", exc)
    bv = {"at90": at90["baselines"], "exc": exc["baselines"]}

    # Published values on the validation subset (same per-object definitions).
    subset_ids = [json.loads(line)["object_id"]
                  for line in (OUT / "validation-damit-period-500.jsonl").open()]
    series = load(OUT / "scores-published-check/direction-excluded.json")["series"]
    position = {name: i for i, name in enumerate(series["object_id"])}
    chosen = [position[i] for i in subset_ids if i in position]
    pub_d = np.asarray(series["delphi_error_deg"])[chosen]
    pub_s = np.asarray(series["standard_error_deg"])[chosen]
    pub_sub_diff = base.paired_difference(pub_s, pub_d, base.BOOTSTRAP_SEED + 2)
    sub_exc = sub["variants"]["failed_excluded"]

    n_fail = own["prediction_status_counts"].get("no_period", 0)
    lines: list[str] = []
    add = lines.append
    add("# Lower-quality DAMIT census at our own periods (package `lowq`)\n")
    add("Exploratory revision analysis (PROTOCOL.md, Addendum 1). The networks receive the top "
        "`fourier_gpu` period found by `estimate(curves, \"fourier_gpu\", resolution=1, timeout=180, "
        "sparse_fallback=True)` on exactly the observations of each census input (every source epoch of "
        "`lc.txt` one curve, magnitude -2.5 log10 of the relative brightness). Everything else is the "
        "published census: same 9,784 selected asteroids, same input parsing, the published **fold-0 "
        "five-network ensemble** (`run_k3_lowq_census.py`, `inference_policy: fixed_fold_five_seed_ensemble`), "
        "same scoring, lineage grouping, standard poles and latitude analysis, run with the published code. "
        "An asteroid without a period is a failure, given at 90 deg and also excluded.\n")
    add("Code: branch `research/own-period-lowq`, `repro/own_period/lowq_*.py`, chain "
        "`repro/own_period/run_lowq_chain.sh`, figure script `repro/own_period/lowq_figures/render_main_figures.py`.\n")

    add("## Validation\n")
    add(f"- DAMIT-period rerun of the first {validation['rows']} asteroids (published order) with the "
        f"prediction script: {validation['bit_identical_axis_sets']} of {validation['ok_rows']} axis sets "
        f"bit-identical, largest axis difference {validation['max_axis_angle_deg']:.3g} deg, "
        f"status and grid-index mismatches {len(validation['status_mismatches'])} and "
        f"{len(validation['grid_index_mismatches'])}; passed = {validation['passed']} "
        f"(`{REL}/validation-damit-period-500-check-v3.json`).")
    add(f"- Oracle errors of that rerun minus the published errors: largest absolute difference "
        f"{sub['published_minus_own']['max_abs_difference_deg']:.3g} deg "
        f"(`{REL}/scores-damit-period-500/summary.json`).")
    add(f"- The scoring chain (`lowq_score.py`, which calls the published census, lineage, baseline and "
        f"latitude code) run on the published prediction file reproduces {check['compared_values']} published "
        f"values with {len(check['mismatches'])} mismatches (tolerance {check['tolerance']}); passed = "
        f"{check['passed']} (`{REL}/check-published-v2.json`). The only differences "
        f"({len(check['standard_pole_20deg_tie_differences'])} values) are within-20 fractions of the six "
        "standard poles, caused by six asteroids lying exactly 20 deg from a standard pole (floating-point tie; "
        "reference-only, not printed). The first check file `check-published.json` reports these as failures.\n")

    add("## Period search\n")
    regimes = own.get("search_status_regime", {})
    add(f"- Asteroids searched: {len(periods['periods_hours'])} (one, damit:1015, keeps its published "
        f"status `input_rejected`). Period found: {sum(v is not None for v in periods['periods_hours'].values())}; "
        f"failures among the 9,783 scored asteroids: **{n_fail}**.")
    add("- Status and input regime (9,783 scored): " + ", ".join(
        f"{k} {v['n']}" for k, v in sorted(regimes.items(), key=lambda kv: -kv[1]["n"])) + ".")
    secs = own.get("search_seconds", {})
    add(f"- Search time: sum {secs.get('sum', 0) / 3600:.2f} h, mean {secs.get('mean', 0):.2f} s, median "
        f"{secs.get('median', 0):.2f} s, max {secs.get('max', 0):.1f} s, {secs.get('n_over_180')} over 180 s "
        f"(includes parsing). Timing sample of the first 200 before the run: mean {timing['mean_seconds']:.2f} s, "
        f"projected {timing['projected_hours_for_9784']:.2f} h for 9,784 (`{REL}/search-timing-sample.json`).")
    add("- Alias class of the own period against the DAMIT model-table period (within 1%), with the mean "
        "oracle error at the own period and at the DAMIT period (published):\n")
    add("| Class | n | Own-period mean (deg) | DAMIT-period mean (deg) |")
    add("|---|---:|---:|---:|")
    for name, block in own["period_classes"].items():
        add(f"| {name} | {block['n']:,} | {f2(block.get('own_mean_deg'))} | {f2(block.get('published_mean_deg'))} |")
    add(f"\nFailures are counted at 90 deg in the own-period column. Source: `{REL}/scores-own/summary.json`, "
        f"per asteroid `{REL}/scores-own/objects.csv`.\n")

    add("## Numbers the manuscript prints from the census\n")
    add("Columns: published value; own period with failures at 90 deg (n = 9,783); own period with failures "
        f"excluded (n = {exc['census_primary']['n_objects']:,}). Own-period sources are in `{REL}/scores-own/` "
        "(`census-analysis-{at90,excluded}.json`, `lineage-{at90,excluded}/analysis.json`, "
        "`baselines-{at90,excluded}.json`, `direction-{at90,excluded}.json`).\n")
    add("| Item (manuscript) | Published | Own, failures at 90 | Own, failures excluded | Published source |")
    add("|---|---:|---:|---:|---|")
    pc, a9, ex = pub_census["primary"], at90["census_primary"], exc["census_primary"]
    add(f"| Analyzed n (`FollowLowqN`) | {pc['n_objects']:,} | {a9['n_objects']:,} | {ex['n_objects']:,} | lowq-analysis.json |")
    add(f"| Mean oracle error, 95% interval (`FollowLowqMean`) | {pc['mean_oracle_at_3_error_deg']:.2f} "
        f"[{pc['mean_bootstrap_ci95_deg'][0]:.2f}, {pc['mean_bootstrap_ci95_deg'][1]:.2f}] | "
        f"{a9['mean_oracle_at_3_error_deg']:.2f} [{a9['mean_bootstrap_ci95_deg'][0]:.2f}, {a9['mean_bootstrap_ci95_deg'][1]:.2f}] | "
        f"{ex['mean_oracle_at_3_error_deg']:.2f} [{ex['mean_bootstrap_ci95_deg'][0]:.2f}, {ex['mean_bootstrap_ci95_deg'][1]:.2f}] | lowq-analysis.json |")
    add(f"| Median | {pc['median_oracle_at_3_error_deg']:.2f} | {a9['median_oracle_at_3_error_deg']:.2f} | {ex['median_oracle_at_3_error_deg']:.2f} | lowq-analysis.json |")
    add(f"| Within 20 deg | {100 * pc['within_20_deg']:.1f}% | {100 * a9['within_20_deg']:.1f}% | {100 * ex['within_20_deg']:.1f}% | lowq-analysis.json |")
    add(f"| Within 30 deg | {100 * pc['within_30_deg']:.1f}% | {100 * a9['within_30_deg']:.1f}% | {100 * ex['within_30_deg']:.1f}% | lowq-analysis.json |")
    add(f"| Standard poles mean (`FollowLowqStandardMean`) | {pub_base['lowq']['standard']['mean_deg']:.2f} | "
        f"{bv['at90']['lowq']['standard']['mean_deg']:.2f} | {bv['exc']['lowq']['standard']['mean_deg']:.2f} | reference-baselines.json |")
    add(f"| Standard minus DeLPHI, 95% interval (E6, line 864) | {interval(pub_base['lowq']['standard_minus_delphi'])} | "
        f"{interval(bv['at90']['lowq']['standard_minus_delphi'])} | {interval(bv['exc']['lowq']['standard_minus_delphi'])} | reference-baselines.json |")
    add(f"| ASAS-SN DeLPHI mean vs standard (E7, line 867) | {pub_base['lowq_asas']['delphi']['mean_deg']:.2f} vs "
        f"{pub_base['lowq_asas']['standard']['mean_deg']:.2f} | {bv['at90']['lowq_asas']['delphi']['mean_deg']:.2f} vs "
        f"{bv['at90']['lowq_asas']['standard']['mean_deg']:.2f} | {bv['exc']['lowq_asas']['delphi']['mean_deg']:.2f} vs "
        f"{bv['exc']['lowq_asas']['standard']['mean_deg']:.2f} (n {bv['exc']['lowq_asas']['object_count']}) | reference-baselines.json |")
    pl, ol, el = pub_dir, at90["direction"], exc["direction"]
    add(f"| Latitude 75-90 deg mean (`InsightLowqLatHigh`) | {pl['latitude_bins'][5]['mean_error_deg']:.2f} | "
        f"{ol['latitude_bins'][5]['mean_error_deg']:.2f} | {el['latitude_bins'][5]['mean_error_deg']:.2f} | mirror-symmetry.json |")
    add(f"| Latitude 0-15 deg mean (`InsightLowqLatLow`) | {pl['latitude_bins'][0]['mean_error_deg']:.2f} | "
        f"{ol['latitude_bins'][0]['mean_error_deg']:.2f} | {el['latitude_bins'][0]['mean_error_deg']:.2f} | mirror-symmetry.json |")
    add(f"| Standard poles, latitude 0-15 deg (`InsightLowqStandardLatLow`) | {pl['standard_latitude_bins'][0]['mean_error_deg']:.2f} | "
        f"{ol['standard_latitude_bins'][0]['mean_error_deg']:.2f} | {el['standard_latitude_bins'][0]['mean_error_deg']:.2f} | mirror-symmetry.json |")
    add(f"| Spearman, abs latitude vs error (`InsightLowqRhoLatitude`) | {pl['spearman_abs_latitude']:.2f} | "
        f"{ol['spearman_abs_latitude']:.2f} | {el['spearman_abs_latitude']:.2f} | mirror-symmetry.json |")
    add(f"| Longitude, Kruskal-Wallis p over six 60-deg bins (`InsightLowqLonP`, X4) | {pl['longitude_kruskal_p']:.2f} | "
        f"{ol['longitude_kruskal_p']:.2g} | {el['longitude_kruskal_p']:.2g} | mirror-symmetry.json |")
    add("")
    add("Latitude bins (Figure 11 lower-quality curve), mean oracle error in deg (n):\n")
    add("| Bin | Published | Own, at 90 | Own, excluded | Standard, published | Standard, own excluded set |")
    add("|---|---:|---:|---:|---:|---:|")
    for k in range(6):
        p, o, e = pl["latitude_bins"][k], ol["latitude_bins"][k], el["latitude_bins"][k]
        add(f"| {p['low']}-{p['high']} | {p['mean_error_deg']:.2f} ({p['n']}) | {o['mean_error_deg']:.2f} ({o['n']}) | "
            f"{e['mean_error_deg']:.2f} ({e['n']}) | {pl['standard_latitude_bins'][k]['mean_error_deg']:.2f} | "
            f"{el['standard_latitude_bins'][k]['mean_error_deg']:.2f} |")
    add("\nLongitude bins (60 deg), mean oracle error in deg: " + "; ".join(
        f"{p['low']}-{p['high']}: {p['mean_error_deg']:.2f} / {o['mean_error_deg']:.2f} / {e['mean_error_deg']:.2f}"
        for p, o, e in zip(pl["longitude_bins"], ol["longitude_bins"], el["longitude_bins"]))
        + " (published / own at 90 / own excluded).\n")

    add("## Table 8 (`tab:lowq`)\n")
    add("Each cell: published -> own with failures at 90 -> own with failures excluded. "
        "Standard poles are reference-only, so their mean changes only through the asteroid set.\n")
    add("| Group | n | DeLPHI mean | Median | Within 20 | Standard poles | Standard minus DeLPHI [95%] |")
    add("|---|---|---|---|---|---|---|")
    for label, key in (("All below quality 3", "lowq"), ("No shared source", "lowq_unshared"),
                       ("Gaia DR3", "lowq_gaia"), ("ASAS-SN", "lowq_asas"), ("Shared source", "lowq_shared")):
        cells = [pub_base[key], bv["at90"][key], bv["exc"][key]]
        add(f"| {label} | " + " -> ".join(f"{c['object_count']:,}" for c in cells)
            + " | " + " -> ".join(f"{c['delphi']['mean_deg']:.2f}" for c in cells)
            + " | " + " -> ".join(f"{c['delphi']['median_deg']:.2f}" for c in cells)
            + " | " + " -> ".join(f"{c['delphi']['within_twenty_percent']:.1f}%" for c in cells)
            + " | " + " -> ".join(f"{c['standard']['mean_deg']:.2f}" for c in cells)
            + " | " + " -> ".join(interval(c["standard_minus_delphi"]) for c in cells) + " |")
    add(f"\nSources: published `source/repro/baselines_out/reference-baselines.json` (cohorts lowq*), own "
        f"`{REL}/scores-own/baselines-{{at90,excluded}}.json` (same code and seeds as "
        "`compute_reference_baselines.py`; grouping from `lineage-*/objects.csv`).\n")

    add("## Figure 7 lower-quality bars (standard minus DeLPHI)\n")
    add("| Bar | Published | Own, at 90 | Own, excluded |")
    add("|---|---|---|---|")
    for label, key in (("All lower-quality DAMIT", "lowq"), ("Gaia DR3 lower-quality subset", "lowq_gaia"),
                       ("ASAS-SN lower-quality subset", "lowq_asas")):
        add(f"| {label} | {interval(pub_base[key]['standard_minus_delphi'])} | "
            f"{interval(bv['at90'][key]['standard_minus_delphi'])} | {interval(bv['exc'][key]['standard_minus_delphi'])} |")
    add(f"\nRedrawn: `{REL}/figures/generalization-checks-{{at90,excluded}}.pdf` (other rows published, labelled), "
        f"data `generalization-checks-*-rows.json`; Figure 11: `{REL}/figures/latitude-{{at90,excluded}}.pdf` "
        "(main-sample curve published; the main curve at own period is another package's item), binned series "
        "`latitude-*-series.json`, per-asteroid series in `scores-own/direction-*.json` (`series`).\n")

    add("## DAMIT-period (published) DeLPHI minus own-period DeLPHI\n")
    pm = own["published_minus_own"]
    for name in ("failed_at_90", "failed_excluded"):
        b = pm[name]
        add(f"- {name}: n {b['n']:,}, {b['mean_difference_deg']:.2f} deg [{b['ci95_low_deg']:.2f}, "
            f"{b['ci95_high_deg']:.2f}] (bootstrap 10,000, seed 20261021).")
    add(f"- Published mean on the own-period successes: {pm['published_on_own_successes']['mean_deg']:.2f} deg "
        f"(n {pm['published_on_own_successes']['n']:,}).\n")

    add("## Train-only atlas comparison (`k3-followup-results.tex`, not printed in the manuscript)\n")
    add("| Item | Published | Own, at 90 | Own, excluded |")
    add("|---|---|---|---|")
    pa, oa, ea = pub_census["paired_atlas_minus_k3"], at90["census_paired_atlas_minus_k3"], exc["census_paired_atlas_minus_k3"]
    fmt = lambda b: f"{b['mean_improvement_deg']:.2f} [{b['bootstrap_ci95_deg'][0]:.2f}, {b['bootstrap_ci95_deg'][1]:.2f}]"
    add(f"| Atlas minus DeLPHI, all | {fmt(pa)} | {fmt(oa)} | {fmt(ea)} |")
    for label, key in (("disjoint", "structured_reference_disjoint"), ("overlap", "structured_reference_overlap")):
        p, o, e = pub_lineage[key], at90[f"lineage_{label}"], exc[f"lineage_{label}"]
        add(f"| Lineage {label}: n, DeLPHI mean, atlas minus DeLPHI | {p['n_objects']}, {p['k3']['mean_deg']:.2f}, "
            f"{p['paired_atlas_minus_k3_mean_deg']:.2f} | {o['n_objects']}, {o['k3']['mean_deg']:.2f}, "
            f"{o['paired_atlas_minus_k3_mean_deg']:.2f} | {e['n_objects']}, {e['k3']['mean_deg']:.2f}, "
            f"{e['paired_atlas_minus_k3_mean_deg']:.2f} |")
    add("")

    add(f"## DAMIT-period rerun subset (first {len(subset_ids)} asteroids in the published order)\n")
    sp = sub_exc["census_primary"]
    add("| Item | Published, same asteroids | DAMIT-period rerun |")
    add("|---|---:|---:|")
    add(f"| n | {pub_d.size} | {sp['n_objects']} |")
    add(f"| Mean | {pub_d.mean():.4f} | {sp['mean_oracle_at_3_error_deg']:.4f} |")
    add(f"| Median | {np.median(pub_d):.4f} | {sp['median_oracle_at_3_error_deg']:.4f} |")
    add(f"| Within 20 deg | {100 * np.mean(pub_d <= 20):.1f}% | {100 * sp['within_20_deg']:.1f}% |")
    add(f"| Standard minus DeLPHI | {interval(pub_sub_diff)} | {interval(sub_exc['baselines']['lowq']['standard_minus_delphi'])} |")
    add(f"\nSource: `{REL}/scores-damit-period-500/` (rerun), published errors from `scores-published-check/direction-excluded.json`.\n")

    add("## Deviations and notes\n")
    add("- The brief speaks of the 25 networks; the published census (9,783) used the fixed fold-0 five-network "
        "ensemble (`predictions-fold0-five-seed/manifest.json`), so that is what is rerun here ('exactly as "
        "published'). The 25-network policy belongs to the earlier 169-asteroid low-quality cohort "
        "(`run_k3_lowq_damit.py`), which the manuscript does not print.")
    add("- Validation check files `-check.json` and `-check-v2.json` report `passed: false` only because of the "
        "angle formula (arccos of the dot product of float32-normalized axes gives 0.03 deg, and 1.7e-6 deg after "
        "normalizing, for bit-identical vectors). All 499 axis sets were bit-identical in every version; `-check-v3.json` "
        "uses the atan2 form and passes. The rerun itself was not repeated.")
    add("- damit:1015 (rejected before the networks in the published census) was written as `no_period` in "
        "`own-period-lowq-predictions.jsonl` (search status `input_rejected_published`). A first scoring pass counted "
        "it as an own-period failure at 90 deg (n 9,784); those outputs are kept in `superseded-1015-at90/`. "
        "The scoring here keeps its published rejected status, so the 90-deg variant has n 9,783.")
    add("- Census inputs are read from the raw `lc.txt` exactly as the published census does (there is no "
        "separate prepared-object directory for the 9,783); the search receives those same observations.")
    add("- Only the lower-quality rows are recomputed. Main-sample, ZTF and ALCDEF rows of Figure 7 and the "
        "main curve of Figure 11 are published values, labelled so.")
    add("- Random-axes values (`FollowLowqRandomMean`) and the synthetic-donor exposure (125) are "
        "period-independent and keep their published values.")
    add("- DAMIT Q1/Q2 poles were themselves derived with the DAMIT period from largely the same sparse "
        "photometry, so the own-period rows also measure the effect of our period differing from the period "
        "under which the reference poles were fitted.")
    destination.write_text("\n".join(lines) + "\n")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
