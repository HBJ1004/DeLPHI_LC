#!/usr/bin/env bash
# After v67_queue.sh: full-grid cross-check and RESULTS.md.
set -u
WT="/mnt/d/Downloads/Colab Notebooks/DeLPHI-followup/source-op-v67"
V="/mnt/d/Downloads/Colab Notebooks/DeLPHI-followup/data/own-period-20261021/v67"
PIN="/mnt/d/Downloads/Colab Notebooks/DeLPHI-followup/data/pole-grid-benchmark/runtime/python-venv-20260912/bin/python-pinned"
export PYTHONPATH="$WT"
cd "$WT" || exit 1
until grep -qE 'V67_QUEUE_DONE|failed|FAILED' "$V/queue.log"; do sleep 300; done
grep -q V67_QUEUE_DONE "$V/queue.log" || { echo "queue did not finish cleanly"; exit 1; }
"$PIN" repro/own_period/v7_grid.py crosscheck --execution "$V/v7-own-period"
"$PIN" repro/own_period/v67_results.py
echo V67_FINALIZE_DONE
