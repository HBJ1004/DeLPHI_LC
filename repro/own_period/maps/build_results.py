"""Collect the maps-package numbers into one table (printed as Markdown rows).

    python -m repro.own_period.maps.build_results OUT_DIR

OUT_DIR is data/own-period-20261021/maps.  Reads published-check/, damit-period/
and own-period/ (analyses/summary.json and the other analysis files).
"""
import json
import sys
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(Path(path).read_text())


def ecl(v):
    v = np.asarray(v, float) / np.linalg.norm(v)
    return np.degrees(np.arctan2(v[1], v[0])) % 360.0, np.degrees(np.arcsin(np.clip(v[2], -1, 1)))


def metis(run):
    e = np.load(run / "evaluations/real-oof-ensemble.npz")
    i = int(np.flatnonzero(e["object_ids"] == "asteroid_107")[0])
    parts = []
    for axis in e["refined_axes"][i]:
        if axis[2] < 0:
            axis = -axis
        lon, lat = ecl(axis)
        parts.append(f"({lon:.0f}, {lat:+.0f})")
    return " ".join(parts) + f"; oracle {e['oracle_errors_deg'][i]:.1f}"


def latitude_bins(mirror):
    rows = mirror["objects"]
    lat = np.array([abs(r["reference_latitude_deg"]) for r in rows])
    err = np.array([r["oracle_error_deg"] for r in rows])
    out = []
    for low in range(0, 90, 15):
        high = low + 15
        mask = (lat >= low) & ((lat < high) if high < 90 else (lat <= high))
        out.append(f"{err[mask].mean():.1f}")
    return "/".join(out)


def column(base: Path):
    s = load(base / "analyses/summary.json") if (base / "analyses/summary.json").exists() else load(base / "summary.json")
    a = base / "analyses" if (base / "analyses").exists() else base
    mirror = load(a / "mirror-symmetry.json")
    o, pi, ms = s["overall"], s["primary_intervals"], s["mirror_symmetry"]
    seeds = s["per_seed_mean_deg"]
    sym = ms["symmetry_by_observer_latitude"]
    pref = ms["reference_preferred_over_mirror"]
    pairs = ms["mirror_pairs_among_candidates"]["10"]
    aa, ap = s["amplitude_aspect"], s["amplitude_aspect_paired"]
    groups = {(g["low"], g["high"]): g for g in aa["agreement_groups"]}
    hi, lo = groups[(0.8, 1.0)], groups[(-1.0, 0.5)]
    strata = [r for r in s["reference_strata"] if r["property"] == "primary_reference_abs_latitude_deg"]
    sp = s["spearman_with_oracle_error"]
    others = ("log_observations", "native_sessions", "baseline_years", "period_hours", "reference_count")
    cc = {r["k"]: r for r in s["candidate_count"]["k"]}
    std = s["candidate_count"]["standard_poles"]
    rr = s["reference_reassignment"]
    folds = s["folds"]
    fmeans = [f["mean_deg"] for f in folds]
    r = {}
    r["M5"] = f"{s['phase_bin_occupancy']['median_filled_bins']:.0f} of 64 ({s['phase_bin_occupancy']['lightcurves']} lightcurves)"
    r["M6"] = f"{o['grid_mean_deg']:.2f} -> {o['adjusted_mean_deg']:.2f}"
    r["A2"] = r["M6"]
    r["M7"] = "Metis axes (lon, lat) " + metis(base if (base / "evaluations").exists() else Path(s["run"]).parents[1])
    r["R1"] = f"{s['phaethon_2514_oracle_deg']:.1f}"
    r["R4"] = f"{pi['mean_deg']:.2f} ({pi['mean_ci95_deg'][0]:.2f}-{pi['mean_ci95_deg'][1]:.2f}); median {pi['median_deg']:.2f}"
    r["R6"] = (f"{100 * pi['within20']:.1f}% ({100 * pi['within20_ci95'][0]:.1f}-{100 * pi['within20_ci95'][1]:.1f}%); "
               f"30 deg {100 * pi['within30']:.1f}%")
    r["R16"] = f"{min(fmeans):.2f}-{max(fmeans):.2f}"
    r["B1"] = "; ".join(f"{f['fold']}: {f['n']}, {f['mean_deg']:.2f}, {f['median_deg']:.2f}" for f in folds)
    r["R17"] = f"{min(seeds.values()):.2f}-{max(seeds.values()):.2f}" if seeds else "n/a"
    r["B2"] = "/".join(f"{seeds[k]:.2f}" for k in ("17", "42", "137", "777", "2027")) if seeds else "n/a"
    r["R18"] = ("yes" if s.get("ensemble_better_than_every_seed") else "no") if seeds else "n/a"
    r["R19"] = f"{o['min_deg']:.2f}-{o['max_deg']:.2f}, SD {o['sd_ddof1_deg']:.2f}"
    r["R20"] = f"median {s['network_seconds']['median']:.2f} s, mean {s['network_seconds']['mean']:.2f} s"
    r["I2"] = (f"{sym[0]['median_mirror_correlation']:.2f} (n={sym[0]['n']}), "
               f"{sym[3]['median_mirror_correlation']:.2f} (n={sym[3]['n']})")
    r["I3"] = f"{100 * pairs['observed']:.0f}% (null {100 * pairs['null_mean']:.0f}%, 97.5% {100 * pairs['null_97_5']:.0f}%)"
    r["I4"] = (f"{100 * pref[0]['fraction_reference_higher']:.0f}/{100 * pref[1]['fraction_reference_higher']:.0f}/"
               f"{100 * pref[2]['fraction_reference_higher']:.0f}% (n={sum(p['n'] for p in pref)})")
    r["I6"] = f"{aa['map_correlation_median']:.2f}, null {aa['cross_asteroid_null_median_correlation']['mean']:.2f}"
    r["I8"] = f"{aa['delphi_oracle_deg']['mean']:.2f} / {100 * aa['delphi_oracle_deg']['within20']:.0f}% (n={aa['n']})"
    r["I9"] = (f">0.8: {hi['delphi_mean_deg']:.1f} vs {hi['amplitude_aspect_mean_deg']:.1f} (n={hi['n']}); "
               f"<0.5: {lo['delphi_mean_deg']:.1f} vs {lo['amplitude_aspect_mean_deg']:.1f} (n={lo['n']})")
    g = ap["amplitude_aspect_minus_delphi_deg"]
    r["I10"] = f"{g[0]:.2f} ({g[1]:.2f}-{g[2]:.2f}); closer {ap['delphi_closer_count']} of {ap['n_objects']}"
    r["X1"] = "15-deg bin means " + latitude_bins(mirror)
    r["X2"] = (f"rho {strata[0]['spearman_rho']:.2f}; bins " + "/".join(f"{x['mean_error_deg']:.2f}" for x in strata)
               + " (n " + "/".join(str(x["n"]) for x in strata) + ")")
    r["X4"] = f"Kruskal p {ms['main_sample_direction']['longitude_kruskal_p']:.2f}"
    r["X5"] = f"{ms['coordinate_split']['mean_latitude_part_deg']:.2f} vs {ms['coordinate_split']['mean_longitude_part_deg']:.2f}"
    r["X6"] = ("max |rho| " + f"{max(abs(sp[k]) for k in others):.2f} ("
               + ", ".join(f"{k} {sp[k]:+.2f}" for k in others)
               + (f"; own period {sp['own_period_hours']:+.2f}" if "own_period_hours" in sp else "") + ")")
    r["C1"] = ("K=1..6 mean " + "/".join(f"{cc[k]['mean_error_deg']:.2f}" for k in range(1, 7))
               + "; within 20 " + "/".join(f"{100 * cc[k]['within_20']:.0f}" for k in range(1, 7)) + "%")
    r["C2"] = (f"3 candidates {100 * cc[3]['search_area_mean_radius']:.1f}% (p90 {100 * cc[3]['search_area_p90_radius']:.1f}%); "
               f"standard {100 * std['search_area_mean_radius']:.1f}% (p90 {100 * std['search_area_p90_radius']:.1f}%)")
    r["C3"] = ("areas K=1..6 " + "/".join(f"{100 * cc[k]['search_area_mean_radius']:.1f}" for k in range(1, 7))
               + f"%; smallest at K={s['candidate_count_smallest_area_k']}")
    r["B6"] = (f"{rr['null_mean_error_deg']:.1f} ({rr['null_q025_q975_deg'][0]:.1f}-{rr['null_q025_q975_deg'][1]:.1f}); "
               f"{rr['movable_objects']} of 170; observed {rr['observed_mean_error_deg']:.2f}")
    if "ztf_all_intended" in s:
        z = s["ztf_all_intended"]
        r["L7"] = (f"{z['both_failure_kinds_at_90_deg']:.2f} (n={z['n_with_both_at_90']}, period failures at 90); "
                   f"{z['period_failures_excluded_deg']:.2f} (n={z['n_with_period_failures_excluded']}, period failures excluded)")
    return r


def main():
    out = Path(sys.argv[1])
    cols = {"published-check": column(out / "published-check"),
            "damit": column(out / "damit-period"),
            "own": column(out / "own-period")}
    print(json.dumps(cols, indent=1))


if __name__ == "__main__":
    main()
