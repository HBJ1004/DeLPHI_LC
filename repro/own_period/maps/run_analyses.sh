#!/usr/bin/env bash
# CPU analyses and figures of the maps package, after run_predictions.sh.
set -euo pipefail
WT="${DELPHI_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${DELPHI_DATA:?set DELPHI_DATA to the data folder}"
PY="${PYTHON:-python}"
OP="$DELPHI_DATA"
# The selected-network path never consults publication completion markers.
if [[ -n ${DELPHI_NETWORK_ROOT:-}${DELPHI_MODEL_BINDING:-}${DELPHI_BUNDLE_ROOT:-} ]]; then
  export PYTHONPATH="$WT"
  "$PY" -B - <<'PY_SWITCH'
import os
from pathlib import Path
import subprocess
import sys
from repro import network_root as network
out = Path(network.NetworkPath("results", "headline", "maps"))
network.guard_output(out)
for choice in ("damit-period", "own-period"):
    periods = "damit" if choice == "damit-period" else str(Path(os.environ["DELPHI_DATA"]) / "own-periods-170.json")
    subprocess.run([sys.executable, "-B", "-m", "repro.own_period.maps.analyses",
                    "--run", str(out / choice), "--periods", periods,
                    "--out", str(out / choice / "analyses")], check=True)
PY_SWITCH
  exit 0
fi
OUT="$OP/maps"
export PYTHONPATH="$WT" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "$WT"
until grep -q "exploratory timing done" "$OP/benchmark/explore-timed.log" 2>/dev/null; do sleep 300; done
until [ -f "$OUT/own-period/evaluations/score-summary.json" ]; do
  grep -q "FAILED" "$OUT/run-predictions-2.log" 2>/dev/null && { echo "prediction validation failed"; exit 1; }
  sleep 120
done
PUB="$OP/../../inputs/frozen-artifacts/k3-definitive-7874092"
[ -f "$OUT/published-check/summary.json" ] || \
  "$PY" -m repro.own_period.maps.analyses --run "$PUB" --periods damit --out "$OUT/published-check"
[ -f "$OUT/damit-period/analyses/summary.json" ] || \
  "$PY" -m repro.own_period.maps.analyses --run "$OUT/damit-period" --periods damit --out "$OUT/damit-period/analyses"
[ -f "$OUT/own-period/analyses/summary.json" ] || \
  "$PY" -m repro.own_period.maps.analyses --run "$OUT/own-period" --periods "$OP/own-periods-170.json" --out "$OUT/own-period/analyses"
[ -f "$OUT/damit-period/figures-check/worked-example.pdf" ] || \
  "$PY" -m repro.own_period.maps.render_figures --analyses "$OUT/damit-period/analyses" --run "$OUT/damit-period" \
    --periods damit --out "$OUT/damit-period/figures-check"
[ -f "$OUT/figures/worked-example.pdf" ] || \
  "$PY" -m repro.own_period.maps.render_figures --analyses "$OUT/own-period/analyses" --run "$OUT/own-period" \
    --periods "$OP/own-periods-170.json" --out "$OUT/figures"
echo "$(date -u +%FT%TZ) maps analyses done"
