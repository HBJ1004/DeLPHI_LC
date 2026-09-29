"""Pole agreement and residual of the saved classical fits for each budget, against the six DeLPHI starts.

Post hoc descriptive analysis. Classical fits come from classical-ladder-exploratory.json (reconstructed
from the saved full-grid fits), and the DeLPHI fits from score-loaded.json, on the same 140 asteroids.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# Benchmark outputs (release paper-v1, part-b-analyses/workflow-benchmark, plus the raw fits).
DATA = HERE.parents[2] / "data/direct-workflow-benchmark"
ladder = json.load(open(DATA / "classical-ladder-exploratory.json"))["object_rows"]
delphi = {r["object_id"]: r["arms"]["delphi"] for r in json.load(open(DATA / "score-loaded.json"))["object_rows"]}
de = np.array([delphi[r["object_id"]]["reference_error_degrees"] for r in ladder])
dr = np.array([delphi[r["object_id"]]["final_relative_rms"] for r in ladder])
budgets = []
for b in sorted(ladder[0]["classical"], key=int):
    ce = np.array([r["classical"][b]["reference_error_degrees"] for r in ladder])
    cr = np.array([r["classical"][b]["final_relative_rms"] for r in ladder])
    budgets.append(dict(starts=int(b), mean_error_deg=float(ce.mean()), within20=int((ce <= 20).sum()),
                        rms_ratio_classical_over_delphi=float(np.exp(np.mean(np.log(cr / dr))))))
out = dict(scope="post_hoc_saved_fits_140_evaluation_asteroids", n=len(ladder),
           delphi=dict(starts=6, mean_error_deg=float(de.mean()), within20=int((de <= 20).sum())), budgets=budgets)
(DATA / "budget-equivalence.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
