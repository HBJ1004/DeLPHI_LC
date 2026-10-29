#!/usr/bin/env bash
# Timed runs of packages V6 and V7 at our own period, queued behind phase B.
#   bash repro/own_period/v67_queue.sh  (run in the background; log in v67/queue.log)
# 1. wait until reliability/phaseB2.log has PHASE_B_DONE or PHASE_B_FAILED
# 2. V7 guided GPU validation at the published periods (20 asteroids, repeat 0, untimed)
# 3. V6 own-period timed run, chunks of ~30 min, each under gpu_wait + the GPU lock
# 4. V7 own-period timed run, chunks of ~50 min, each under gpu_wait + the GPU lock
# 5. scoring of both (only after each run is complete)
set -u
WT="${DELPHI_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${DELPHI_DATA:?set DELPHI_DATA to the data folder}"
PY="${PYTHON:-python}"
OP="$DELPHI_DATA"
V="$OP/v67"
PIN="${CONVEXINV_PYTHON:-$PY}"
IDS="asteroid_103 asteroid_111 asteroid_118 asteroid_120 asteroid_116 asteroid_125 asteroid_134 asteroid_153 asteroid_101 asteroid_104 asteroid_115 asteroid_117 asteroid_105 asteroid_106 asteroid_107 asteroid_121 asteroid_108 asteroid_109 asteroid_114 asteroid_123"
export PYTHONPATH="$WT"
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*"; }
gpu() {
  if [ -n "${GPU_WAIT:-}" ]; then bash "$GPU_WAIT"; fi
  if [ -n "${GPU_LOCK:-}" ]; then
    flock "$GPU_LOCK" "$@"
  else
    "$@"
  fi
}

log "waiting for phase B"
until grep -qE '^PHASE_B_(DONE|FAILED)' "$OP/reliability/phaseB2.log" 2>/dev/null; do sleep 300; done
log "phase B finished: $(grep -E '^PHASE_B_(DONE|FAILED)' "$OP/reliability/phaseB2.log" | tail -1)"

# 2. guided validation on the GPU (published periods)
if [ ! -f "$V/v7-validation-guided-gpu/validation-vs-published.json" ]; then
  gpu "$PIN" repro/own_period/v7_grid.py execute --lock "$V/v7-validation-published-gpu-lock.json" \
      --output "$V/v7-validation-guided-gpu" --ids $IDS --repeats 0 --arms guided20 || { log "V7 GPU validation run failed"; exit 1; }
  "$PIN" repro/own_period/v7_grid.py validate --execution "$V/v7-validation-guided-gpu" || { log "V7 GPU validation FAILED; stopping before timed runs"; exit 1; }
fi
log "V7 guided GPU validation passed"

# 3. V6 timed, chunks
while true; do
  gpu "$PIN" repro/own_period/v6_fixed_period.py run --periods "$V/own-periods-170-rounded6.json" \
      --axes "$OP/own-period-oof.npz" --output "$V/v6-own-period" --max-seconds 1800
  code=$?
  log "V6 chunk exit $code"
  [ "$code" -eq 0 ] && break
  [ "$code" -eq 3 ] || { log "V6 failed"; exit 1; }
done
"$PIN" repro/own_period/v6_fixed_period.py score --output "$V/v6-own-period" --result "$V/v6-own-period-score.json" || exit 1
log "V6 scored"

# 4. V7 timed, chunks
while true; do
  gpu "$PIN" repro/own_period/v7_grid.py execute --lock "$V/v7-own-period-lock.json" \
      --output "$V/v7-own-period" --max-seconds 3000
  code=$?
  log "V7 chunk exit $code"
  [ "$code" -eq 0 ] && break
  [ "$code" -eq 3 ] || { log "V7 failed"; exit 1; }
done
"$PIN" repro/own_period/v7_grid.py score --lock "$V/v7-own-period-lock.json" --execution "$V/v7-own-period" \
    --output "$V/v7-own-period-score.json" || exit 1
log "V67_QUEUE_DONE"
