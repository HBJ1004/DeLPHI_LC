# Metadata-only external eligibility

`lc_pipeline.k3.external_eligibility` defines the shared eligibility contract
for later dense ALCDEF, ATLAS SSCAT, and TESS TSSYS cohorts. It is an
inventory step only. It does not read predictions, pole directions, oracle
errors, model scores, or reference values, and it does not download or score a
cohort.

Each row must bind an explicit physical identity and both source receipts. The
row also records point and session counts, documented apparitions, observing
and photometric conventions, period provenance, reference provenance, and
three separate exposure fields. Missing history, periods, apparitions, or
reference provenance stays `unknown`; it cannot become a cleared external
identity by default.

The default metadata screen requires at least 150 valid points, five native
sessions, and two documented apparitions. These are eligibility thresholds, not
claims that the observations are sufficient for a reliable pole. A row with a
catalog-agreement reference is a `development_candidate`; a row with an
`independent_supported` reference and clear exposure history is a
`confirmatory_candidate`. Rows that fail a known quality threshold are
`excluded`; rows with unresolved provenance are `unknown`.

The output is sorted by `(physical_identity, input_source)` and contains a
hash of the policy. Duplicate identity/source rows, ambiguous identities,
invalid receipts, and fields that could encode an outcome fail closed. In
particular, changing a reference pole or model score cannot change eligibility,
because such fields are rejected rather than ignored.

Build a fixture or a reviewed metadata export with:

```bash
python -m repro.build_k3_external_eligibility \
  --input rows.json \
  --output eligibility.json
```

The resulting JSON is not a cohort lock and must not be described as external
validation. A later packet must separately verify raw observations, geometry,
period quality, reference-photometry overlap, full exposure history, and the
final sealed cohort before prediction or reference scoring.
