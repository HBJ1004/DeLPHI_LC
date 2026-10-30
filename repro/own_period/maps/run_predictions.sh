#!/usr/bin/env bash
# Queued GPU runs of the maps package: DAMIT-period validation, then own period.
# Each period choice takes the GPU lock once (expected well under one hour).
set -euo pipefail
WT="${DELPHI_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${DELPHI_DATA:?set DELPHI_DATA to the data folder}"
PY="${PYTHON:-python}"
OP="$DELPHI_DATA"
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
gpu = Path(network.NetworkPath("artifacts")).parent / "runners/gpu_run.sh"
for choice in ("damit-period", "own-period"):
    periods = "damit" if choice == "damit-period" else str(Path(os.environ["DELPHI_DATA"]) / "own-periods-170.json")
    command = [sys.executable, "-B", "-m", "repro.own_period.predict_oof_full", "--output-dir", str(out / choice)]
    subprocess.run([str(gpu), "--", *command, "predict", "--periods", periods, "--device", "cuda"], check=True)
    subprocess.run([*command, "score"], check=True)
    if choice == "damit-period":
        subprocess.run([sys.executable, "-B", "-m", "repro.own_period.maps.validate_damit_run",
                        str(out / choice), str(out / choice / "validation.json")], check=True)
PY_SWITCH
  exit 0
fi
OUT="$OP/maps"
MARK="$OP/benchmark/explore-timed.log"
export PYTHONPATH="$WT"
cd "$WT"
until grep -q "exploratory timing done" "$MARK" 2>/dev/null; do sleep 300; done
echo "$(date -u +%FT%TZ) marker seen"
for choice in damit-period own-period; do
  if [ "$choice" = damit-period ]; then periods=damit; else periods="$OP/own-periods-170.json"; fi
  if [ ! -f "$OUT/$choice/predictions/pred-ensemble-fold-4.npz" ]; then
    if [ -n "${GPU_WAIT:-}" ]; then bash "$GPU_WAIT"; fi
    echo "$(date -u +%FT%TZ) $choice predict start"
    gpu_command=("$PY" -m repro.own_period.predict_oof_full)
    if [ -n "${GPU_LOCK:-}" ]; then gpu_command=(flock "$GPU_LOCK" "${gpu_command[@]}"); fi
    "${gpu_command[@]}" \
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
