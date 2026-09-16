# Similarity and candidate-set stability results — 2026-09-16

The two CPU analyses are complete. They use existing DAMIT observations,
held-out folds, and saved predictions. There was no retraining, new neural
inference, inversion, or GPU use. No manuscript, Overleaf project, model
checkpoint, frozen publication result, or public version/release was changed.

The design is recorded in `similarity-stability-design-20260916.md`. It was
fixed before these new comparisons were scored, but after earlier cohort
results had been inspected. This is a retrospective diagnostic, not a new
preregistered or independent validation cohort.

## Similar-asteroid baseline

Each held-out asteroid is compared with its fold's 108 training asteroids
using fixed photometry, observing-geometry, sampling, and supplied-period
features. Scaling uses only training objects. The primary baseline takes
one first-listed catalog axis from each of three distinct nearest neighbours.
The sensitivity uses all catalog axes of the closest neighbour, repeating an
axis if necessary to fill three slots. Neither uses a previous-version model.
Both candidate sets and neighbour identities were sealed before scoring.

All methods are evaluated on the same 170 objects with the same antipodal
oracle@3 endpoint, not selected-pole accuracy.

| Method | Mean disagreement | Median | Within 20 degrees |
|---|---:|---:|---:|
| K3 ensemble | 15.77 deg | 12.09 deg | 74.1% |
| Three-neighbour first-axis baseline | 32.39 deg | 29.92 deg | 31.2% |
| One-neighbour all-axes sensitivity | 43.86 deg | 40.72 deg | 15.3% |
| Training-only atlas | 28.10 deg | 27.50 deg | 24.1% |

The paired three-neighbour-minus-K3 mean difference is **16.62 degrees**,
with a conditional 95% bootstrap interval of **13.40 to 19.76 degrees**.
This supports a benefit beyond this particular fixed similarity lookup.
The baselines were not optimized over feature choices or retrieval methods;
the result does not exclude all forms of memorization or replace the less
favourable cross-survey results. The one-neighbour sensitivity can contain
fewer than three distinct directions and should not be read as an equally
diverse three-proposal method.

## Candidate-set movement

Candidate ordering and axial sign are ignored. Match the three perturbed
axes one-to-one to the same object's full-input axes and take the smallest
mean angular displacement. Average sampling repeats within asteroid before
summarizing objects. Thus the movement metric concerns the entire three-axis
set; it is not displacement of a uniquely identified physical pole.

| Perturbation | Mean set movement | 90th percentile of object-average movement | Mean change in oracle@3 |
|---|---:|---:|---:|
| Retain half the points | 15.51 deg | 27.19 deg | +1.00 deg |
| Retain a quarter of the points | 23.00 deg | 37.71 deg | +4.97 deg |
| Retain half the native sessions | 14.05 deg | 24.42 deg | +1.29 deg |
| Retain a quarter of the native sessions | 19.09 deg | 33.20 deg | +3.67 deg |
| Add 3% full-input flux noise | 12.05 deg | 24.89 deg | +1.01 deg |
| Add 10% full-input flux noise | 27.18 deg | 54.70 deg | +11.52 deg |

Each displayed condition includes all 170 objects and three sampling repeats.
The small mean oracle changes under some perturbations do not imply that the
three individual proposals are stable. The minimum reference disagreement can
remain similar while other proposals, or the identity of the closest proposal,
change. These calculations do not establish the physical cause of the movement.

For users, the result supports retaining the unordered-set interpretation and
checking alternative starts and sensitivity to the available observations.
It does not provide a validated stability cutoff, confidence radius, or a
guarantee that stable candidates identify the physical pole.

## Statistical and execution details

- All **23,120** saved rows passed object/condition/repeat, eligibility,
  source-hash, axis, and fold checks. The report covers **50** perturbation
  conditions and **16,030** successful eligible perturbation repeats.
  There were no eligible prediction failures. Original ineligible rows remain
  counted rather than disappearing from denominators.
- Repeat means, minimum-maximum assignment distances, repeat-to-repeat
  movement, and the ten largest displacements per condition are retained.
  The complete per-object and per-repeat CSVs are included.
- Intervals use **10,000** asteroid resamples within the existing folds,
  seed **20260916**, conditional on the existing models and fixed feature
  rule. They do not include retraining or model-selection uncertainty.
- The atlas effect remains 12.33 degrees. Its interval in this *new*
  fold-stratified analysis is 9.60--15.00 degrees. This differs from the
  original unstratified, differently seeded 9.48--15.13 interval; the original
  manuscript result was not overwritten or relabelled.
- Independent oracle recomputation from all successful saved axes agreed
  with the saved errors to at most **6.81e-13 degrees**.
- A separate verifier imports no `lc_pipeline` code. It checks all input
  and output hashes, training-only scalers, all 170 neighbour selections,
  all 680 method/object errors, and all 16,030 successful perturbation
  assignments. It uses cross-product/atan2 angles and SciPy's assignment
  solver instead of the analysis's acos and six-permutation enumeration.
  Maximum differences were **2.50e-13 degrees** for baseline errors,
  **2.39e-6 degrees** for repeat-level assignment movement, and
  **5.94e-8 degrees** for condition-mean movement.
- **25 new tests passed; the full source suite passed 509 tests**. Scoped
  lint and whitespace checks passed. Tests include a complete synthetic
  170-object pipeline, changed-input/output rejection, and withheld-label
  isolation. No ongoing experiment remains.

## Artifacts and reproduction

Relative to the source checkout, the run directory is:

`../data/similarity-stability-20260916/run-1/`

It contains the sealed analysis lock, sealed neighbour predictions and raw
features/scalers, output manifest, and `report/` with the JSON summary,
three complete CSV tables, and a readable report. Local bindings include
absolute input paths and are not an anonymous submission package.

Analysis-lock SHA-256:
`b5a543fc9ec0c6851bfaea5dd47a590c43bbfb879afa6f7ba721de6d7b97f139`

Output-manifest SHA-256:
`cddefc2b33c67da3788b8a8f2702df052a994467b437717785224f264d4abe29`

From the source checkout, independently verify without changing any output:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
  python \
  -m repro.verify_k3_similarity_stability \
  --output ../data/similarity-stability-20260916/run-1
```

To reproduce, use `repro.run_k3_similarity_stability prepare` with
`--original-study ../data/reliability-20260915/run-1` and a **new** `--output`
directory, then `run --output` with that same new directory. Re-running `run`
on a completed directory verifies hashes and returns `already_complete`;
it does not overwrite evidence. The preparation timestamp and output path
binding make independent run receipts different even when numerical results
are identical. Preserve the existing run and protocol when making revisions.

These results are local follow-up evidence, not part of the existing public
v1.0/v1.0.0 assets. Manuscript and archival integration remain a separate step.
