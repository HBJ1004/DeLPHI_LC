# K3 follow-up external data

`repro/k3_followup_study_spec.yaml` freezes the cohort and label-access policy.
The commands below acquire or prepare inputs only. They do not run prediction,
fit selection, or reference scoring. Every output directory must be new.

## Existing Fink/ZTF cache

The offline ingest selects numbered JSON filenames exclusively from the 170
held-out identities in the frozen publication split. It never opens the
label-bearing DAMIT catalog. Historical Fink responses use both a numbered
`i:ssnamenr` and an asteroid name, sometimes within one file. Numeric values
must match the split-derived filename; a stable named value is checked against
`sso_name`, retained in each normalized row, and counted.

```bash
python -m repro.prepare_k3_followup_inputs ingest-fink \
  --raw-directory /path/to/existing/fink/data/raw \
  --splits repro/data/damit-20250610T000301Z/publication-splits-v2.3.json \
  --spec repro/k3_followup_study_spec.yaml \
  --output-directory /path/to/run/fink-normalized
```

Plan the Horizons acquisition before using the network:

```bash
python -m repro.prepare_k3_followup_inputs plan-horizons \
  --normalized-manifest /path/to/run/fink-normalized/manifest.json \
  --output-directory /path/to/run/horizons \
  --batch-size 20
```

The plan is read-only and reports validated cached objects, pending objects,
and estimated API requests. It fails on incomplete, stale, or hash-mismatched
caches. The default batch size is 20: this size succeeded against the official
API during the 2026-09-10 integration check, while a 200-epoch request received
HTTP 502. Each completed object directory is atomic, so rerunning the fetch
resumes at the next missing object.

```bash
python -m repro.prepare_k3_followup_inputs fetch-horizons \
  --normalized-manifest /path/to/run/fink-normalized/manifest.json \
  --output-directory /path/to/run/horizons \
  --batch-size 20
```

This queries only the official JPL Horizons API. Both raw JSON responses are
retained for every batch, with URLs, HTTP dates, content types, and SHA-256
hashes. The exact query requests uncorrected ecliptic-J2000 vectors at each
retained Fink `i:jd`. Horizons returns center-to-asteroid vectors; the cache
records their deterministic negation to asteroid-to-Sun and asteroid-to-Earth.

## Strict-temporal DAMIT objects

The frozen case series contains asteroids 49, 279, and 366, whose official DAMIT
records were created after the 2025-06-10 cutoff. Fetch a raw snapshot from the
official DAMIT export, lightcurve, and generated-file endpoints:

```bash
python -m repro.prepare_k3_followup_inputs fetch-temporal-damit \
  --spec repro/k3_followup_study_spec.yaml \
  --output-directory /path/to/run/temporal-damit-snapshot
```

The snapshot separates material by access role:

- `raw/input/asteroids.csv` and `raw/input/asteroid_N/lc.txt` retain source input bytes.
- `raw/reference/asteroid_models.csv` and `raw/reference/.../spin.txt` retain label bytes.
- `input-index.json` contains identity, creation date, hashes, lightcurve path,
  and the externally supplied fixed period, but no pole coordinates or axes.
- `reference/reference-manifest.json` contains the pole labels and must remain
  unavailable until all predictions or fit selections are saved.
- `fetch-receipt.json` binds every raw file to its endpoint, HTTP date, and hash.

DAMIT export timestamps do not include a timezone suffix. The receipt preserves
the source strings and the validator records its UTC interpretation. These three
records are more than a month after the cutoff, so ordinary timezone ambiguity
does not change their strict-temporal classification.

Create prediction-ready input documents in a separate offline step:

```bash
python -m repro.prepare_k3_followup_inputs prepare-temporal-damit \
  --snapshot-directory /path/to/run/temporal-damit-snapshot \
  --output-directory /path/to/run/temporal-prepared
```

Preparation opens only `input-index.json` and its hash-bound lightcurves. Tests
verify that it still succeeds when both reference directories are absent. The
result uses the existing external prediction loader but does not expose or score
any reference axis.
