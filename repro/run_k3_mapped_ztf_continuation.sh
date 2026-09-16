#!/usr/bin/env bash
# Durable, label-blind continuation after the serial mapped Fink acquisition.
# It never starts Fink requests; every stage is append-only and resumable.
set -euo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT="$SOURCE_ROOT/../data/generalization-20260910"
PYTHON_BIN=${PYTHON_BIN:-python}
FETCH_ROOT="$DATA_ROOT/mapped-ztf/fetched-raw-current-endpoint"
CORRECTION_ROOT="$DATA_ROOT/mapped-ztf/fetched-raw-current-endpoint-correction"
INGEST_ROOT="$DATA_ROOT/mapped-ztf/ingest-current-endpoint"
HORIZONS_ROOT="$DATA_ROOT/mapped-ztf/horizons-current-endpoint"
PREPARED_ROOT="$DATA_ROOT/mapped-ztf/prepared-current-endpoint"
LOG_DIR="$DATA_ROOT/mapped-ztf/continuation-logs"
LOG_PATH=${MAPPED_ZTF_CONTINUATION_LOG:-"$LOG_DIR/continuation.log"}

mkdir -p "$LOG_DIR"
exec >>"$LOG_PATH" 2>&1
echo "$(date --iso-8601=seconds) mapped-ZTF continuation started"

receipt_overrides=()
raw_overrides=()
if [[ -d "$CORRECTION_ROOT/receipts" ]]; then
  receipt_overrides=(--fetch-receipt-override-directory "$CORRECTION_ROOT/receipts")
  if [[ -d "$CORRECTION_ROOT/raw" ]]; then
    raw_overrides=(--raw-override-directory "$CORRECTION_ROOT/raw")
  fi
elif [[ -e "$CORRECTION_ROOT" ]]; then
  echo "correction root lacks receipts: $CORRECTION_ROOT" >&2
  exit 2
fi

cd "$SOURCE_ROOT"
if [[ ! -f "$INGEST_ROOT/manifest.json" ]]; then
  "$PYTHON_BIN" -m repro.prepare_k3_mapped_ztf ingest \
    --identity-map "$DATA_ROOT/identity-audit/identity-map.json" \
    --period-manifest "$DATA_ROOT/identity-audit/mapped-periods.json" \
    --raw-directory "$FETCH_ROOT/raw" \
    --fetch-receipts-directory "$FETCH_ROOT/receipts" \
    "${receipt_overrides[@]}" "${raw_overrides[@]}" \
    --output "$INGEST_ROOT"
fi

if [[ ! -f "$HORIZONS_ROOT/manifest.json" ]]; then
  "$PYTHON_BIN" -m repro.prepare_k3_mapped_ztf fetch-horizons \
    --identity-map "$DATA_ROOT/identity-audit/identity-map.json" \
    --ingest-manifest "$INGEST_ROOT/manifest.json" \
    --reuse-horizons-root "$DATA_ROOT/mapped-ztf/horizons" \
    --reuse-horizons-root "$SOURCE_ROOT/../data/horizons-final" \
    --batch-size 20 --timeout-seconds 60 --request-interval-seconds 1 \
    --output "$HORIZONS_ROOT"
fi

if [[ ! -f "$PREPARED_ROOT/manifest.json" ]]; then
  "$PYTHON_BIN" -m repro.prepare_k3_mapped_ztf prepare \
    --identity-map "$DATA_ROOT/identity-audit/identity-map.json" \
    --period-manifest "$DATA_ROOT/identity-audit/mapped-periods.json" \
    --ingest-manifest "$INGEST_ROOT/manifest.json" \
    --horizons-manifest "$HORIZONS_ROOT/manifest.json" \
    --output "$PREPARED_ROOT"
fi
echo "$(date --iso-8601=seconds) mapped-ZTF continuation complete"
