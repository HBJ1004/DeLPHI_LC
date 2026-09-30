"""Paired comparison of DeLPHI, the amplitude-aspect map, and the standard poles
on the asteroids of the amplitude-aspect comparison (Section 6.2.2).

Reads data/interpretation-20260930/amplitude-aspect.json written by
amplitude_aspect_comparison.py and writes amplitude-aspect-paired.json next to it.
Intervals are 95% bootstrap intervals over asteroids (10,000 samples).
"""
import json
from pathlib import Path

import numpy as np

FOLLOWUP = Path(__file__).resolve().parents[2]
FOLDER = FOLLOWUP / "data/interpretation-20260930"


def main():
    rows = json.loads((FOLDER / "amplitude-aspect.json").read_text())
    delphi = np.array([row["delphi_oracle_deg"] for row in rows])
    amplitude = np.array([row["amplitude_aspect_oracle_deg"] for row in rows])
    standard = np.array([row["standard_oracle_deg"] for row in rows])
    draws = np.random.default_rng(20260930).integers(0, len(rows), size=(10000, len(rows)))

    def paired(difference):
        boot = difference[draws].mean(axis=1)
        return [float(difference.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

    out = {
        "n_objects": len(rows),
        "mean_error_deg": {"delphi": float(delphi.mean()), "amplitude_aspect": float(amplitude.mean()),
                           "standard": float(standard.mean())},
        "within20": {"delphi": float(np.mean(delphi < 20)), "amplitude_aspect": float(np.mean(amplitude < 20)),
                     "standard": float(np.mean(standard < 20))},
        "amplitude_aspect_minus_delphi_deg": paired(amplitude - delphi),
        "standard_minus_amplitude_aspect_deg": paired(standard - amplitude),
        "delphi_closer_count": int(np.sum(delphi < amplitude)),
    }
    (FOLDER / "amplitude-aspect-paired.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
