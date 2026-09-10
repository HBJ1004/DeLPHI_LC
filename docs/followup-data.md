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
The observer vector uses the Earth center (`500@399`), not each exposure's ZTF
observatory location. This geocentric approximation is explicit in every
prepared object and must be disclosed when interpreting cross-survey results.

If the study specification is refrozen while a long Horizons acquisition is
already running, do not silently relabel its caches. `rebind-horizons` accepts a
complete source manifest only, verifies all 169 normalized objects are identical
apart from their study-spec hash, copies raw responses/vector rows unchanged,
and records both old and final hashes in every rebound cache.

Once all 169 cache directories and the final Horizons manifest exist, bind the
raw Horizons target headers to the Fink numeric/name designations and export
the externally supplied period only:

```bash
python -m repro.prepare_k3_followup_inputs audit-horizons \
  --normalized-manifest /path/to/run/fink-normalized/manifest.json \
  --horizons-manifest /path/to/run/horizons/manifest.json \
  --output /path/to/run/horizons-identity-audit.json

python -m repro.prepare_k3_followup_inputs build-ztf-period-manifest \
  --normalized-manifest /path/to/run/fink-normalized/manifest.json \
  --catalog repro/data/damit-20250610T000301Z/catalog.jsonl \
  --spec repro/k3_followup_study_spec.yaml \
  --output /path/to/run/ztf-periods.json
```

The period export chooses the first frozen catalog solution in catalog order,
records its model/hash provenance, and emits no pole coordinates, vectors, or
solution array. Bulk preparation refuses to run unless the normalized, final
Horizons, identity-audit, period, and current study-spec hashes agree and all
contain exactly the same 169 identities.

```bash
python -m repro.prepare_k3_followup_inputs prepare-ztf \
  --normalized-manifest /path/to/run/fink-normalized/manifest.json \
  --horizons-manifest /path/to/run/horizons/manifest.json \
  --horizons-identity-audit /path/to/run/horizons-identity-audit.json \
  --period-manifest /path/to/run/ztf-periods.json \
  --spec repro/k3_followup_study_spec.yaml \
  --output-directory /path/to/run/ztf-prepared
```

## Post-cutoff DAMIT records

The descriptive case series contains asteroids 49, 279, and 366, whose official
DAMIT records were created after the 2025-06-10 cutoff. This is a post-cutoff
DAMIT-record case series, not evidence that the asteroid identities themselves
were previously unseen. Fetch a raw snapshot from the
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
does not change their post-cutoff record classification.

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

Before any external prediction, verify all 25 publication models against both
the release archive index and checkpoint audit:

```bash
python -m repro.prepare_k3_followup_inputs bind-models \
  --artifact-root /path/to/k3-definitive-7874092 \
  --archive-index /path/to/k3-definitive-7874092/release/archive-index.json \
  --publication-package-manifest /path/to/publication-archive-manifest.json \
  --output /path/to/run/external-models.json
```

Prediction requires this manifest. Existing Fink/DAMIT identities receive only
the five seeds from their unique held-out fold; each post-cutoff DAMIT record
uses all five folds and five seeds (25 checkpoints). Checkpoint bytes, filename,
fold, seed, training commit, protocol, and synthetic-parent provenance are
revalidated before any model is deserialized for inference.
