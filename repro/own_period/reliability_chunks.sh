#!/usr/bin/env bash
# Resumable driver for the own-period reliability runs on the shared machine.
#   bash repro/own_period/reliability_chunks.sh <stage>
# Every GPU chunk waits for an idle GPU, takes the shared lock, and stops after
# about 55 minutes; the next chunk resumes from the records already written.
set -euo pipefail
W="$(cd "$(dirname "$0")/../.." && pwd)"
PY=/home/onlyb/.venvs/delphi-reliability-20260915/bin/python
OUT="$W/../data/own-period-20261021/reliability"
MARK="$W/../data/own-period-20261021/benchmark/explore-timed.log"
GPU_WAIT="$W/../source-quality-ladder/repro/quality_ladder/gpu_wait.sh"
LOCK=/home/onlyb/MERCUDA-paper/.gpu.lock
export PYTHONPATH="$W" OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${OMP_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS:-2}"
cd "$W"

until grep -q "exploratory timing done" "$MARK" 2>/dev/null; do sleep 300; done

gpu() {  # one locked GPU chunk
  bash "$GPU_WAIT"
  echo "$(date -u +%FT%TZ) chunk: $*"
  flock "$LOCK" "$PY" -m repro.own_period.reliability "$@"
}

until_marker() {  # repeat a chunked GPU command until its marker exists
  local marker="$1"; shift
  while [ ! -e "$marker" ]; do gpu "$@"; done
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
