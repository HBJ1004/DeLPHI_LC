# Generalization follow-up: implementation and current evidence

This work is separate from the released version 1.0 and the original manuscript
results. New code and data live in the follow-up workspace. Existing evidence,
release tags, models, and manuscript numbers are not overwritten.

## Identity correction discovered during implementation

The original ZTF follow-up is invalid as a cross-survey comparison. It parsed
MPC numbers from `asteroid_N`, but those names in the frozen publication split
refer to **internal DAMIT database IDs**. In the official DAMIT asteroid table,
database ID 101 is (2) Pallas, not (101) Helena. The former supplies the period
and reference; the latter supplied the old ZTF photons and Horizons vectors.
All 169 old prepared ZTF objects fail the physical-identity join.

The old 32.23-degree result therefore supplies no evidence for or against
generalization. Previously described geometry, brightness and sampling
differences remain properties of those ZTF inputs, but their angular errors
cannot diagnose transfer to the intended asteroid cohort. The newly started
factorial diagnostic prediction run was stopped before reference scoring.
Its partial outputs are retained with an `INVALIDATED.json` record.

The correction is an explicit join:

`DAMIT database ID -> official asteroid table -> MPC number -> Fink/Horizons target`

Known periods, reference axes and held-out models continue to join by internal
DAMIT ID. Photometry acquisition joins by MPC number. New prepared ZTF objects
must record both identities and hashes of the official mapping. The predictor
rejects old ZTF files lacking this binding. Unit tests specifically distinguish
database ID 101, MPC 2, and MPC 101.

This cross-survey defect does not itself invalidate the original within-DAMIT
evaluation, which uses the same internal database identity for input and label.
The three post-cutoff record cases must also pass physical-identity exposure
auditing before they could support any unseen-identity claim.

## Reproducible commands

Run from the source checkout with the supported Python environment. All output
paths below must be new; large data are outside the Git source tree.

```bash
python -m repro.fetch_k3_generalization_sources --source alcdef --output-directory ../data/generalization-20260910/sources/alcdef-pds-v1
python -m repro.fetch_k3_generalization_sources --source tess-metadata --output-directory ../data/generalization-20260910/sources/tess-dr1-metadata
python -m repro.fetch_k3_generalization_sources --source damit-identities --output-directory ../data/generalization-20260910/sources/damit-identities
python -m repro.fetch_k3_generalization_sources --source gaia-dr3-spins --output-directory ../data/generalization-20260910/sources/gaia-dr3-spins-20260911

python -m repro.audit_k3_survey_identity \
  --asteroid-table ../data/generalization-20260910/sources/damit-identities/asteroids.csv \
  --splits repro/data/damit-20250610T000301Z/publication-splits-v2.3.json \
  --prepared-manifest ../data/ztf-prepared-final/manifest.json \
  --output-directory ../data/generalization-20260910/identity-audit

python -m repro.export_k3_mapped_periods \
  --identity-map ../data/generalization-20260910/identity-audit/identity-map.json \
  --catalog repro/data/damit-20250610T000301Z/catalog.jsonl \
  --output ../data/generalization-20260910/identity-audit/mapped-periods.json

python -m repro.crossmatch_k3_alcdef_gaia \
  --alcdef-census ../data/generalization-20260910/census/alcdef-pds-v1-rev2.json \
  --gaia-table ../data/generalization-20260910/sources/gaia-dr3-spins-20260911/table3.dat \
  --exposure-audit ../data/generalization-20260910/exposure-audit/k3-census-exposure-audit-v3.json \
  --output ../data/generalization-20260910/census/alcdef-gaia-eligibility-20260911.json
```

The download receipts record source URLs, dates, byte counts and SHA-256.
Redirect query tokens are not needed for source identification. The identity
audit reads the official asteroid identity table, not a pole table. Period
export reads the previously exposed training catalog but writes no pole values.

The development design is in `repro/k3_generalization_study_spec.yaml`; the
subsequent required identity correction is recorded separately in
`repro/k3_generalization_identity_revision.yaml`. Neither is a claim that an
external confirmation test is already complete or that its final lock exists.

## Executed work and remaining checks

The public ALCDEF PDS archive and TSSYS DR1 metadata have been acquired with
checksums. The ALCDEF parser records 23,846 conservative candidate identities
and 9,944,099 positive-metadata-ID observations, of which 9,941,447 pass its
validity and duplicate checks. The source documentation gives 23,847 objects
and 9,944,193 observations. The census records the identity reconciliation
and the remaining 94-observation discrepancy; it does not silently replace
the source totals. TSSYS has 9,912 metadata identities in this inventory;
its lightcurve files have not yet been acquired.

The ALCDEF screen gives 12,174 identities with at least five metadata sessions
and 150 valid points. Of these, 2,713 span at least two session years, and
943 have an MPC-number match in the DAMIT identity table. Two session years
are not two verified apparitions. No pole values enter this census.

The published 1.58 GB synthetic archive matches the release manifest checksum.
Only its 96 JSON provenance manifests were extracted for this audit; training
arrays were not extracted. The older `k3-2408c563` donor manifests are
byte-identical to that set. The `k3-eef6d4ec` local directory does not contain
an auditable donor manifest. Other incomplete historical records remain
explicitly unknown. Among the 943 screened candidates, the currently
inventoried splits, donors and previous data queries identify 289 as exposed.
Adding the three already evaluated post-cutoff cases gives 292 exposed and
651 unresolved. Zero are certified as new independent test objects. The
reproducible inventory and separate case audit are under
`../data/generalization-20260910/exposure-audit/` (use the `-v3.json` records).
They include explicit original-fold and donor-membership annotations and
count only actually queried MPC identities for the legacy ZTF source role,
not the intended targets also listed in its mismatch audit.

A separate `original-cohort-synthetic-donor-audit.json` checks the original
170 identities against all published synthetic donor manifests. There is no
overlap in either the DAMIT or the mapped physical-identity namespace. The
unique shape-donor and geometry-donor counts are 411 each for training,
51 each for validation, and 52 each for testing. This checks the declared
manifest IDs, not the training arrays or a new training run.

The three previously reported post-cutoff cases (MPC 49, 279 and 366) do not
overlap the original 170 identities or the inventoried published synthetic
donors after the official namespace correction. They have already been
evaluated, however, so they remain development cases, not a prospective test.

The corrected local-cache ZTF manifest retains all 170 intended internal
identities: 16 have usable cached photometry and verified Horizons geometry;
154 lack local raw inputs. All eight preprocessing variants produced a
three-axis prediction for each of the 16 available objects (128 predictions).
Missing inputs remain separate failure records in every variant. The CUDA
diagnostic completed in about 84 seconds; this is an execution record, not a
timing benchmark. No reference scoring was performed for this subset.

The original Fink API host timed out. The current
[Fink/ZTF API](https://api.ztf.fink-portal.org/) and its Swagger definition were
checked directly. Pilot queries for physical MPC 2 and 5 returned Pallas and
Astraea with matching numeric identities. Full-cohort acquisition uses that
endpoint, immutable per-object receipts, and serial requests. Empty responses,
responses without retained r-band observations, HTTP failures and invalid
identities are distinct states. They must not be silently omitted.

The full acquisition and ingest finished with 164 ready objects, three
responses without retained observations, two empty responses, and one invalid
response. The latter is MPC 44612: the verified provisional-designation
spellings agree, but the source supplies null ephemeris fields. The original
rejection and separate correction-attempt receipt are both preserved. No
ephemeris values were invented to admit it. Horizons subsequently completed
for all 164 ready identities. Preparation retained 161 objects meeting the
full input contract and preserved nine failure records. The eight fixed-model
variants then produced all 1,360 required predictions. The development-only
score assigns 90 degrees to every input failure across all 170 intended
identities: the original convention has mean oracle-at-3 33.174 degrees
(29.997 degrees on the fixed 161-object prepared subset). Distance reduction
and first-order light-time subtraction changed that score by at most 0.06
degrees; geometry-bounded epoch grouping was worse. These variants are not
selected for a new model or manuscript claim. This is a same-identity,
cross-survey diagnostic and is not external validation.

The durable local continuation is `repro/run_k3_mapped_ztf_continuation.sh`;
its log is under `mapped-ztf/continuation-logs/`. It performs no new Fink
queries. Geometry is acquired serially, after which the prepared manifest is
written with all 170 rows. A separate 100-epoch GET pilot failed with HTTP 502;
the active run retains the existing 20-epoch batches. This is a transport
observation, not a scientific exclusion criterion.

The mapped pipeline is available through:

```bash
python -m repro.prepare_k3_mapped_ztf --help
python -m repro.prepare_k3_mapped_ztf fetch-fink --help
python -m repro.prepare_k3_mapped_ztf ingest --help
python -m repro.prepare_k3_mapped_ztf fetch-horizons --help
python -m repro.prepare_k3_mapped_ztf prepare --help
python -m repro.run_k3_generalization_diagnostics --help
python -m repro.score_k3_generalization_diagnostics --help
python -m repro.analyze_k3_generalization --help
```

For new downloads, `ingest --fetch-receipts-directory` is required by the
workflow: it checks raw bytes against the acquisition receipt and official
identity binding. The local-cache diagnostic and current-endpoint acquisition
are separate snapshots; neither is overwritten or silently merged.

Development scoring is separate from external scoring. It requires all eight
variants and every original split identity, checks each prediction and input
hash and the five held-out-fold model bindings, then opens the original
catalog. It reports all intended objects with a 90-degree failure penalty,
the same fixed available-input subset for every variant, paired changes, and
factorial average effects. These are descriptive, not simultaneous intervals
or a final-test selection rule. Neither the small local-cache subset nor a
favorable partial run is a basis for an external-validation claim.

The next scientific steps are to resolve candidate exposure histories,
apparitions and reference-photometry lineage; then lock an eligible external
cohort. The public CDS table for Durech and Hanus (2023), J/A+A/675/A24,
provides 8,596 Gaia DR3 model-derived spin solutions in ecliptic J2000
coordinates. Its initial intersection with the ALCDEF metadata screen contains
492 identities. A subsequent metadata-only screen of the historical phase-30
records found 23 of these candidates; it adds four previously unrecorded
exposures. The current inventory therefore has 95 known exposed identities and
397 unresolved identities. It is an eligibility inventory only. It does not
establish that any candidate is new, that the ALCDEF and Gaia reference photons
are independent in every case, or that ALCDEF point-level input and geometry
preparation will succeed. The phase-40 metadata identifies its synthetic
manifest, which contains 1,000 synthetic sample IDs but no real donor-identity
field. It therefore cannot clear any candidate. Different file hashes alone do
not prove that reference fitting used different photons.
The analysis CLI requires a conclusive exposure audit and checks the locked
data and sealed predictions before external scoring. These byte-level checks
are not a claim of independent human blinding.

No adaptation or 140-object acceleration test has run. No manuscript result,
release tag, published model or version number has been changed by this work.

## Completed model-specific ALCDEF--Gaia transfer check

A separate deterministic lock selected 30 ALCDEF objects absent from the
frozen model's released real cohort and published synthetic donor manifests.
This is a model-specific retrospective exposure check. It is not a prospective
test and it does not establish that the identities were absent from every
historical project experiment. The raw extract retained 926 native sessions
and 28,467 points; preparation retained 834 sessions and 28,375 finite
observations. JPL SBDB supplied the period inputs, and JPL Horizons supplied
asteroid-centric ecliptic-J2000 session-midpoint geometry. Neither input
required Gaia axes.

All 25 frozen models and the train-only three-axis atlas were predicted and
sealed before the Gaia DR3 spin table was opened for scoring. Against the
Gaia model-derived reference axes, K3's mean antipode-aware oracle@3 error was
37.25 degrees (asteroid bootstrap 95% interval 30.70--44.16; n=30), with a
median of 36.16 degrees and 6/30 within 20 degrees. The train-only atlas was
better by 13.19 degrees on average (paired bootstrap 95% interval -21.53 to
-5.26 when expressed as atlas minus K3). There were no prediction failures.

This is a negative transfer result. It provides no external-validation support
for the frozen model, and it must not be presented as a positive generalization
claim. The archive is retained because excluding an unfavorable locked result
would distort the record.

## Diagnostic and validation rules

After corrected input preparation, run the eight combinations of distance
reduction, light-time subtraction and geometry-bounded epoch grouping. The
weights remain fixed. Grouping is a disclosed new input intervention: maximum
30 days and 10 degrees pairwise geometry separation, without merging native
sessions. Singleton groups are counted and recorded; a resulting prediction
failure remains in the object denominator. No phase-law fitting or unverified
UTC/TDB conversion is silently applied. Light-time subtraction is a first-order
geocentric diagnostic; the geometry itself is not recomputed at retarded time.

DAMIT documents light-time-corrected dates and unit-distance reduction for
calibrated intensity measurements in its [lightcurve format](https://damit.cuni.cz/pages/documentation).
That supports checking the transport convention, not assuming correction will
improve the result. Corrected zero-shot results must be reported before deciding
whether adaptation is justified.

New-identity candidates come first from the [ALCDEF PDS snapshot](https://sbn.psi.edu/pds/resource/alcdef.html).
[TESS TSSYS](https://archive.konkoly.hu/pub/tssys/dr1/README), Gaia DR3 and ATLAS
provide separate instrument/sampling tests. A metadata crossmatch is not a
completed validation: periods, usable observations, observing apparitions,
reference provenance and all training/donor/history exposure must be audited.
Different calendar years are only an apparition-screening approximation.

Keep all epochs and synthetic derivatives of an asteroid in the same split.
Only the final untouched external cohort supplies confirmation after tuning;
development cross-validation is not an independent final test
([Cawley and Talbot 2010](https://jmlr.org/papers/v11/cawley10a.html)).
The primary endpoint remains mean antipodal oracle@3 for exactly three axes.
Report absolute errors, asteroid-level paired intervals, controls, reference
counts and failures. No ranking or unique-physical-pole claim is introduced.

Adaptation is conditional on the diagnostic and exposure audits. The approved
48 GPU-hour maximum is allocated as 2 hours pilot, 16 hours balanced screening,
and 30 hours for the selected complete ensemble. An incomplete screening matrix
does not authorize choosing a favorable partial run. No adaptation has been
authorized to use incorrectly paired ZTF examples.

## Acceleration is a separate experiment

The original convergence development design demanded complete RMS support even
though static solver limits excluded 5/30 development and 21/140 locked objects.
That is a feasibility failure, not evidence that no convergence tolerance works.
The capacity-expanded copy, provenance, parity check and actual support counts
are documented in [the solver report](solver-capacity-followup.md).

Capacity alone does not establish completed fits or acceleration. Both arms
must use the same verified binary, and the development study must be rerun under
a new execution lock before the 140-object timing evaluation is opened. Training
and benchmarking must not run concurrently. The final timing experiment remains
separate from new-identity external validation.
