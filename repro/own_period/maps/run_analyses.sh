#!/usr/bin/env bash
# CPU analyses and figures of the maps package, after run_predictions.sh.
set -euo pipefail
WT="/mnt/d/Downloads/Colab Notebooks/DeLPHI-followup/source-op-maps"
OP="/mnt/d/Downloads/Colab Notebooks/DeLPHI-followup/data/own-period-20261021"
OUT="$OP/maps"
PY=/home/onlyb/.venvs/delphi-reliability-20260915/bin/python
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
