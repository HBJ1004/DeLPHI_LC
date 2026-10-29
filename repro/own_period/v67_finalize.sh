#!/usr/bin/env bash
# After v67_queue.sh: full-grid cross-check and RESULTS.md.
set -u
WT="${DELPHI_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${DELPHI_DATA:?set DELPHI_DATA to the data folder}"
PY="${PYTHON:-python}"
V="$DELPHI_DATA/v67"
PIN="${CONVEXINV_PYTHON:-$PY}"
export PYTHONPATH="$WT"
cd "$WT" || exit 1
until grep -qE 'V67_QUEUE_DONE|failed|FAILED' "$V/queue.log"; do sleep 300; done
grep -q V67_QUEUE_DONE "$V/queue.log" || { echo "queue did not finish cleanly"; exit 1; }
"$PIN" repro/own_period/v7_grid.py crosscheck --execution "$V/v7-own-period"
"$PIN" repro/own_period/v67_results.py
echo V67_FINALIZE_DONE
