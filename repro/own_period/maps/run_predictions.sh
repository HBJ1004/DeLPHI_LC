#!/usr/bin/env bash
# Queued GPU runs of the maps package: DAMIT-period validation, then own period.
# Each period choice takes the GPU lock once (expected well under one hour).
set -euo pipefail
WT="/mnt/d/Downloads/Colab Notebooks/DeLPHI-followup/source-op-maps"
OP="/mnt/d/Downloads/Colab Notebooks/DeLPHI-followup/data/own-period-20261021"
OUT="$OP/maps"
PY=/home/onlyb/.venvs/delphi-reliability-20260915/bin/python
MARK="$OP/benchmark/explore-timed.log"
export PYTHONPATH="$WT"
cd "$WT"
until grep -q "exploratory timing done" "$MARK" 2>/dev/null; do sleep 300; done
echo "$(date -u +%FT%TZ) marker seen"
for choice in damit-period own-period; do
  if [ "$choice" = damit-period ]; then periods=damit; else periods="$OP/own-periods-170.json"; fi
  if [ ! -f "$OUT/$choice/predictions/pred-ensemble-fold-4.npz" ]; then
    bash "/mnt/d/Downloads/Colab Notebooks/DeLPHI-followup/source-quality-ladder/repro/quality_ladder/gpu_wait.sh"
    echo "$(date -u +%FT%TZ) $choice predict start"
    flock /home/onlyb/MERCUDA-paper/.gpu.lock "$PY" -m repro.own_period.predict_oof_full \
      --output-dir "$OUT/$choice" predict --periods "$periods" --device cuda
    echo "$(date -u +%FT%TZ) $choice predict done"
  fi
  if [ ! -f "$OUT/$choice/evaluations/score-summary.json" ]; then
    "$PY" -m repro.own_period.predict_oof_full --output-dir "$OUT/$choice" score
  fi
  if [ "$choice" = damit-period ] && [ ! -f "$OUT/damit-period/validation.json" ]; then
    "$PY" -m repro.own_period.maps.validate_damit_run "$OUT/damit-period" "$OUT/damit-period/validation.json" \
      || true
  fi
  if [ "$choice" = damit-period ] && ! grep -q '"passed": true' "$OUT/damit-period/validation.json"; then
    echo "validation FAILED; own-period run not started"; exit 1
  fi
done
echo "$(date -u +%FT%TZ) maps predictions done"
