#!/usr/bin/env bash
# Resumable driver for the own-period reliability runs on the shared machine.
#   bash repro/own_period/reliability_chunks.sh <stage>
# Every GPU chunk waits for an idle GPU, takes the shared lock, and stops after
# about 55 minutes; the next chunk resumes from the records already written.
set -euo pipefail
W="${DELPHI_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${DELPHI_DATA:?set DELPHI_DATA to the data folder}"
PY="${PYTHON:-python}"
OUT="$DELPHI_DATA/reliability"
MARK="$DELPHI_DATA/benchmark/explore-timed.log"
export PYTHONPATH="$W" OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${OMP_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS:-2}"
cd "$W"

if [[ -n "${DELPHI_NETWORK_ROOT:-}${DELPHI_MODEL_BINDING:-}${DELPHI_BUNDLE_ROOT:-}" ]]; then
  case "${1:-}" in pilot|search-main|search-withheld|validate) echo 'reuse fixed inputs; old publication validation is separate' >&2; exit 2;; esac
  OUT="$("$PY" -B -c 'from repro.own_period.reliability import OUT; print(OUT)')"
else
  until grep -q "exploratory timing done" "$MARK" 2>/dev/null; do sleep 300; done
fi

gpu() {  # one locked GPU chunk
  if [[ -n "${DELPHI_NETWORK_ROOT:-}" ]]; then
    "$(dirname "$DELPHI_NETWORK_ROOT")/runners/gpu_run.sh" -- "$PY" -B -m repro.own_period.reliability "$@"
    return
  fi
  if [ -n "${GPU_WAIT:-}" ]; then bash "$GPU_WAIT"; fi
  echo "$(date -u +%FT%TZ) chunk: $*"
  if [ -n "${GPU_LOCK:-}" ]; then
    flock "$GPU_LOCK" "$PY" -m repro.own_period.reliability "$@"
  else
    "$PY" -m repro.own_period.reliability "$@"
  fi
}

until_marker() {  # repeat a chunked GPU command until its marker exists
  local marker="$1"; shift
  while [ ! -e "$marker" ]; do
    local status=0
    gpu "$@" || status=$?
    if [[ $status != 0 && $status != 3 ]]; then return "$status"; fi
  done
}

stage="$1"; shift || true
case "$stage" in
  pilot)        gpu search --scope main --limit "${LIMIT:-400}" --workers "${WORKERS:-1}" --max-seconds 3300 ;;
  validate)     until_marker "$OUT/predictions/damit.folds-0.complete" predict --mode damit --folds 0 --max-seconds 3300
                "$PY" -m repro.own_period.reliability validate ;;
  search-main)  until_marker "$OUT/search/periods-main.complete" search --scope main --workers "${WORKERS:-3}" --max-seconds 3300 ;;
  search-withheld) until_marker "$OUT/search/periods-withheld.complete" search --scope withheld --workers "${WORKERS:-3}" --max-seconds 3300 ;;
  predict-own)  until_marker "$OUT/predictions/own.folds-01234.complete" predict --mode own --max-seconds 3300 ;;
  withheld-candidates) gpu withheld-candidates --mode "$1" "${@:2}" ;;
  withheld-fits) "$PY" -m repro.own_period.reliability withheld-fits --mode "$1" "${@:2}" ;;
  *) echo "unknown stage $stage" >&2; exit 2 ;;
esac
