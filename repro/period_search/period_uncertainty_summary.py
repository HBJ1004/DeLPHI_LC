"""Compare the Delta-chi^2 period intervals with the DAMIT periods (after the fact)."""
import json

import numpy as np
from period_uncertainty import CATALOG, HERE

rows = [json.loads(line) for line in (HERE / "end-to-end/period-uncertainty-v1.jsonl").read_text().splitlines()]
catalog = {r["object_id"]: r for r in map(json.loads, CATALOG.read_text().splitlines())}
summary = dict(asteroids=len(rows), periods=sum(len(r["candidates"]) for r in rows),
               mean_seconds=float(np.mean([r["seconds"] for r in rows])))
for name in ("polishook_3sigma", "one_parameter_1sigma"):
    widths, inside, bounded, matched, pulls = [], 0, 0, 0, []
    all_bounded = 0
    for r in rows:
        reference = float(catalog[r["object_id"]]["solutions"][0]["period_hours"])
        for c in r["candidates"]:
            box = c[name]
            all_bounded += box["bounded"]
            if box["bounded"]:
                widths.append((box["period_high"] - box["period_low"]) / 2 / c["period_best"])
            if abs(c["period_best"] / reference - 1) > 0.01:
                continue
            matched += 1
            if box["bounded"]:
                bounded += 1
                inside += box["period_low"] <= reference <= box["period_high"]
                half = (box["period_high"] - box["period_low"]) / 2
                pulls.append(abs(reference - c["period_best"]) / half)
    summary[name] = dict(
        bounded_fraction_all=all_bounded / summary["periods"],
        median_relative_half_width=float(np.median(widths)),
        relative_half_width_percentiles_10_90=[float(v) for v in np.percentile(widths, (10, 90))],
        matched_periods=matched, matched_bounded=bounded, reference_inside=inside,
        coverage=inside / bounded if bounded else None,
        median_offset_over_half_width=float(np.median(pulls)))
out = HERE / "end-to-end/period-uncertainty-v1.summary.json"
out.write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps(summary, indent=1))
