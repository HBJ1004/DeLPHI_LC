#!/usr/bin/env bash
# Lower-quality census at our own periods: validation, search, prediction,
# scoring, lineage, figures and RESULTS.md, chained and resumable.
#
#   bash repro/own_period/run_lowq_chain.sh >> OP/lowq/chain.log 2>&1
#
# Every GPU step runs in chunks of at most ~55 min, each chunk waiting for an
# idle GPU and taking the shared lock separately.  A restart skips finished
# asteroids (append-only JSONL) and finished steps (existing output files).
set -u
WT="/mnt/d/Downloads/Colab Notebooks/DeLPHI-followup/source-op-lowq"
PY=/home/onlyb/.venvs/delphi-reliability-20260915/bin/python
OUT="/mnt/d/Downloads/Colab Notebooks/DeLPHI-followup/data/own-period-20261021/lowq"
MARK="/mnt/d/Downloads/Colab Notebooks/DeLPHI-followup/data/own-period-20261021/benchmark/explore-timed.log"
GPUWAIT="/mnt/d/Downloads/Colab Notebooks/DeLPHI-followup/source-quality-ladder/repro/quality_ladder/gpu_wait.sh"
LOCK=/home/onlyb/MERCUDA-paper/.gpu.lock
export PYTHONPATH="$WT" OMP_NUM_THREADS=2
cd "$WT" || exit 1
mkdir -p "$OUT"
log() { echo "$(date -u +%FT%TZ) $*"; }

# Repeat a resumable chunked GPU command until it reports completion (exit 0).
# Exit code 3 means "more chunks needed".
gpu_chunks() {
  while true; do
    bash "$GPUWAIT"
    flock "$LOCK" "$@"
    rc=$?
    if [ "$rc" -eq 0 ]; then return 0; fi
    if [ "$rc" -ne 3 ]; then log "chunk failed with exit code $rc: $*"; return "$rc"; fi
  done
}

log "chain start, waiting for the timed benchmark marker"
until grep -q "exploratory timing done" "$MARK" 2>/dev/null; do sleep 300; done
log "marker found; lc_pipeline from $($PY -c 'import lc_pipeline; print(lc_pipeline.__file__)')"

# 1. Validation: published prediction path with the DAMIT period, first 500 asteroids.
V="$OUT/validation-damit-period-500.jsonl"
if [ ! -f "$V.manifest.json" ]; then
  gpu_chunks "$PY" -m repro.own_period.lowq_predict --periods published --limit 500 \
    --output "$V" --max-seconds 3000 || exit 1
  "$PY" -m repro.own_period.lowq_predict --periods published --limit 500 --output "$V" \
    --finalize "$V.manifest.json" || exit 1
fi
if [ ! -f "$OUT/validation-damit-period-500-check-v3.json" ]; then
  "$PY" -m repro.own_period.lowq_validate --rerun "$V" \
    --output "$OUT/validation-damit-period-500-check-v3.json" || { log "VALIDATION FAILED, stopping"; exit 1; }
fi
grep -q '"passed": true' "$OUT/validation-damit-period-500-check-v3.json" || { log "VALIDATION FAILED, stopping"; exit 1; }
log "validation passed"

# 2. Published scoring path reproduces every published census number (CPU only).
if [ ! -f "$OUT/check-published.json" ]; then
  [ -d "$OUT/scores-published-check" ] || "$PY" -m repro.own_period.lowq_score \
    --predictions published --output "$OUT/scores-published-check" > "$OUT/scores-published-check.log" 2>&1 || exit 1
  "$PY" -m repro.own_period.lowq_check_published --scores "$OUT/scores-published-check" \
    --output "$OUT/check-published.json" || { log "PUBLISHED SCORING CHECK FAILED, stopping"; exit 1; }
fi
log "published scoring check passed"

# 3. Timing sample: the first 200 census asteroids.
S="$OUT/search.jsonl"
if [ ! -f "$OUT/search-timing-sample.json" ]; then
  gpu_chunks "$PY" -m repro.own_period.lowq_search --output "$S" --limit 200 --max-seconds 3300 || exit 1
  "$PY" - "$S" "$OUT/search-timing-sample.json" <<'EOF'
import json, sys
import numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])][:200]
sec = np.array([r["seconds"] for r in rows])
status = {}
for r in rows:
    k = f'{r["status"]}|{r.get("input_regime")}'
    status[k] = status.get(k, 0) + 1
out = {"n": len(rows), "sum_seconds": float(sec.sum()), "mean_seconds": float(sec.mean()),
       "median_seconds": float(np.median(sec)), "max_seconds": float(sec.max()),
       "n_timeout": sum(r["status"] == "timeout" for r in rows),
       "projected_hours_for_9784": float(sec.mean() * 9784 / 3600), "status_regime": status}
open(sys.argv[2], "x").write(json.dumps(out, indent=1) + "\n")
print(out)
EOF
fi

# 4. Full search, then the period file (written once).
if [ ! -f "$OUT/own-periods-lowq.json" ]; then
  gpu_chunks "$PY" -m repro.own_period.lowq_search --output "$S" --max-seconds 3300 || exit 1
  "$PY" -m repro.own_period.lowq_search --output "$S" --finalize "$OUT/own-periods-lowq.json" || exit 1
fi
log "search finished"

# 5. Own-period predictions with the published census networks.
P="$OUT/own-period-lowq-predictions.jsonl"
if [ ! -f "$P.manifest.json" ]; then
  gpu_chunks "$PY" -m repro.own_period.lowq_predict --periods "$OUT/own-periods-lowq.json" \
    --output "$P" --max-seconds 3300 || exit 1
  "$PY" -m repro.own_period.lowq_predict --periods "$OUT/own-periods-lowq.json" --output "$P" \
    --finalize "$P.manifest.json" || exit 1
fi
log "predictions finished"

# 6. Scoring (reference poles and DAMIT periods are read only from here on).
[ -d "$OUT/scores-own" ] || "$PY" -m repro.own_period.lowq_score --predictions "$P" \
  --periods "$OUT/own-periods-lowq.json" --output "$OUT/scores-own" > "$OUT/scores-own.log" 2>&1 || exit 1
[ -d "$OUT/scores-damit-period-500" ] || "$PY" -m repro.own_period.lowq_score --predictions "$V" \
  --output "$OUT/scores-damit-period-500" > "$OUT/scores-damit-period-500.log" 2>&1 || exit 1
log "scoring finished"

# 7. Figures 7 and 11, and RESULTS.md.
[ -f "$OUT/figures/latitude-at90.pdf" ] || "$PY" repro/own_period/lowq_figures/render_main_figures.py \
  > "$OUT/figures.log" 2>&1 || exit 1
[ -f "$OUT/RESULTS.md" ] || "$PY" -m repro.own_period.lowq_results || exit 1
log "chain done"
