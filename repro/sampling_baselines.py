"""Success fractions of the one- to ten-group sampling test and of the
standard poles and random axes on the same 80 asteroids (Section 6.1.3).

Reads reliability-scored-rows.json from release publication-evidence-2026-09-17-r4
(evidence/source-records/) and the per-object baselines of
compute_reference_baselines.py.  Writes data/interpretation-20260930/sampling-baselines.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import compute_reference_baselines as base  # noqa: E402

FOLLOWUP = HERE.parents[1]


def main(scored_rows):
    rows = [r for r in json.load(open(scored_rows))["payload"] if r.get("status") != "ineligible"]
    out = {"groups": {}}
    for blocks in (1, 3, 5, 10):
        cond = f"two_d-block_count-{blocks}-per_block_cap-None"
        chosen = [r for r in rows if r["condition"] == cond]
        per_repeat = [np.mean([r["error_deg"] < 20 for r in chosen if r["repeat"] == k]) for k in (0, 1, 2)]
        out["groups"][blocks] = {"within20_by_repeat": per_repeat, "within20_mean": float(np.mean(per_repeat)),
                                 "mean_error_deg": float(np.mean([r["error_deg"] for r in chosen]))}
    ids = sorted({r["object_id"] for r in rows if r["condition"] == "two_d-block_count-1-per_block_cap-None"})
    poles = base.reference_poles()
    standard = base.standard_axes()
    std = np.array([base.axial_error_deg(standard, poles[i]) for i in ids])
    generator = np.random.default_rng(1)
    fractions = []
    for _ in range(1000):
        axes = generator.normal(size=(len(ids), 3, 3))
        axes /= np.linalg.norm(axes, axis=2, keepdims=True)
        fractions.append(np.mean([base.axial_error_deg(axes[k], poles[i]) < 20 for k, i in enumerate(ids)]))
    # Paired standard-minus-DeLPHI difference, per-object DeLPHI error averaged over the three repeats.
    paired = {}
    for blocks in (1, 3, 5, 10):
        cond = f"two_d-block_count-{blocks}-per_block_cap-None"
        delphi = np.array([np.mean([r["error_deg"] for r in rows if r["condition"] == cond and r["object_id"] == i]) for i in ids])
        diff = std - delphi
        draws = np.random.default_rng(20260930).integers(0, len(ids), size=(10000, len(ids)))
        boot = diff[draws].mean(axis=1)
        paired[blocks] = [float(diff.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    out["standard_minus_delphi_deg"] = paired
    # Total-observation caps on all 170 asteroids, paired against the standard poles.
    per_object = np.load(HERE / "baselines_out/reference-baselines-per-object.npz")
    standard_all = dict(zip([str(v) for v in per_object["object_ids"]], per_object["standard_errors_deg"]))
    caps = {}
    for cap in ("10", "20", "50", "100", "200", "500", "None"):
        cond = f"observation_cap-cap-{cap}"
        errors = {}
        for r in rows:
            if r["condition"] == cond and r.get("status") == "ok":
                errors.setdefault(r["object_id"], []).append(r["error_deg"])
        diff = np.array([standard_all[i] - np.mean(v) for i, v in sorted(errors.items())])
        draws = np.random.default_rng(20260930).integers(0, len(diff), size=(10000, len(diff)))
        boot = diff[draws].mean(axis=1)
        caps[cap] = [int(len(diff)), float(diff.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    out["total_cap_standard_minus_delphi_deg"] = caps
    out.update(n_objects=len(ids), standard_mean_error_deg=float(std.mean()),
               standard_within20=float(np.mean(std < 20)), random_within20_1000_draws=float(np.mean(fractions)))
    target = FOLLOWUP / "data/interpretation-20260930/sampling-baselines.json"
    target.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
