"""Delta-chi^2 uncertainties for every period in the frozen candidate lists.

Method (Harris et al. 1989; Polishook et al. 2012; Chang et al. 2015): with the
Fourier order fixed, the period interval is the connected range around the
minimum where chi^2 < chi^2_min + Delta chi^2.  DAMIT gives no photometric
errors, so chi^2 = RSS / s^2 with s^2 = RSS_min / (n - p), which sets the
reduced chi^2 to one at the best period.  Two thresholds are reported:

  * "polishook_3sigma": Delta chi^2 = chi2.ppf(0.9973, p) with
    p = 1 + sum over observing intervals of (2 N_k + N_s), the asteroid-survey
    convention (N_k harmonics, N_s lightcurve offsets);
  * "one_parameter_1sigma": Delta chi^2 = 1, the textbook interval for one
    parameter of interest.

The candidate periods themselves are read from end-to-end/candidates-v1.jsonl and
are not changed.  DAMIT periods are read only afterwards, for the calibration
summary.  Output files are never overwritten.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import chi2
from search import observing_groups, profile, read_curves, sparse_observing_groups

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CATALOG = ROOT / "DeLPHI-followup/inputs/training-run/repro/data/damit-20250610T000301Z/catalog.jsonl"
ORDERS = (2, 3, 4, 6)
MAX_RELATIVE_HALF_WIDTH = 0.1


def groups_for(path):
    curves = read_curves(path)
    groups = observing_groups(curves)
    return groups if groups else sparse_observing_groups(curves)


def rss(groups, frequency, order):
    n = sum(len(c.time) for g in groups for c in g)
    return float(profile(groups, [frequency], (order,))[1][0]) * n


def parameter_count(groups, order):
    count = 1
    for g in groups:
        points = sum(len(c.time) for c in g)
        hmax = min(order, max(1, (points - len(g) - 2) // 2))
        count += 2 * hmax + len(g)
    return count


def edge(objective, f0, direction, step, limit):
    """Walk outward from f0 until objective > 0, then locate the crossing."""
    previous, current = f0, f0 + direction * step
    while abs(current - f0) <= limit:
        if objective(current) > 0:
            return brentq(objective, min(previous, current), max(previous, current), xtol=1e-12)
        previous, current = current, current + direction * step
        step *= 1.25
    return None


def interval(groups, period, frequency_step):
    f = 24.0 / period
    scores = [float(profile(groups, [f], (o,))[0][0]) for o in ORDERS]
    order = ORDERS[int(np.argmin(scores))]
    fit = minimize_scalar(lambda x: rss(groups, x, order), bounds=(f - frequency_step, f + frequency_step),
                          method="bounded", options={"xatol": 1e-11})
    f_best, rss_best = float(fit.x), float(fit.fun)
    n = sum(len(c.time) for g in groups for c in g)
    p = parameter_count(groups, order)
    s2 = rss_best / (n - p)
    result = dict(order=order, n=n, parameters=p, frequency_best=f_best, period_best=24.0 / f_best,
                  reduced_rss=s2)
    for name, delta in (("polishook_3sigma", float(chi2.ppf(0.9973, p))), ("one_parameter_1sigma", 1.0)):
        threshold = rss_best + delta * s2
        def objective(x, threshold=threshold):
            return rss(groups, x, order) - threshold

        limit = MAX_RELATIVE_HALF_WIDTH * f_best
        lo = edge(objective, f_best, -1, frequency_step / 50, limit)
        hi = edge(objective, f_best, 1, frequency_step / 50, limit)
        result[name] = dict(delta_chi2=delta,
                            period_low=None if hi is None else 24.0 / hi,
                            period_high=None if lo is None else 24.0 / lo,
                            bounded=lo is not None and hi is not None)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default=HERE / "end-to-end/candidates-v1.jsonl", type=Path)
    parser.add_argument("--output", default=HERE / "end-to-end/period-uncertainty-v1.jsonl", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    catalog = {r["object_id"]: r for r in map(json.loads, CATALOG.read_text().splitlines())}
    rows = [json.loads(line) for line in args.candidates.read_text().splitlines()][:args.limit]
    with args.output.open("x", buffering=1) as stream:
        for row in rows:
            started = time.perf_counter()
            path = ROOT / "damit-20250610T000301Z" / catalog[row["object_id"]]["lightcurve"]["source_path"]
            groups = groups_for(path)
            out = []
            for c in row["candidates"]:
                out.append(dict(period_hours=c["period_hours"], candidate_index=c["candidate_index"],
                                **interval(groups, c["period_hours"], c["frequency_step_per_day"])))
            stream.write(json.dumps(dict(object_id=row["object_id"], candidates=out,
                                         seconds=time.perf_counter() - started)) + "\n")
            print(row["object_id"], len(out), f"{time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
