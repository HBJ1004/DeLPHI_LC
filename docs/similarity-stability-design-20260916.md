# Similarity baseline and candidate-set stability: analysis specification

This is a retrospective addition to the completed DAMIT reliability study.
The earlier cohort results have already been inspected. The choices below
are recorded before computing these new comparisons; this is not an original
preregistration or a new independent test set. Positive, negative, and
inconclusive results will all be retained. No model, split, period, original
prediction, or published artifact will be changed.

## Similar-asteroid baseline

Use all 170 original objects and the existing five disjoint-role splits.
Only the 108 training objects in each fold may supply neighbours, labels,
feature means, or feature standard deviations. Validation, calibration, and
test objects cannot be neighbours. All choices are fixed, with no tuning on
test errors.

Extract three feature groups from undegraded native lightcurves and the
already supplied period, without reference poles:

1. Photometry: the median across valid native sessions of relative-flux
   standard deviation, the 95th-minus-5th percentile relative-flux range,
   and four harmonic magnitudes. Normalize each session by its mean flux.
   Harmonic h is twice the magnitude of the mean of
   `(relative_flux - 1) * exp(2 pi i h phase)`, h=1,...,4, where phase uses
   the supplied period and a session-local time origin. These are fixed
   summary features, not an irregular-sampling Fourier fit.
2. Geometry: session-equal means of unit Sun and observer vectors (six
   values), their symmetric second-moment upper triangles (six each), and
   the median and interquartile range of session-mean phase angles (two).
   This group has 20 values and retains the common coordinate frame.
3. Sampling/period: log(period in hours), log(point count), log(valid session
   count), log1p(time span in days), log(median points per valid session),
   and median fraction of 64 occupied phase bins within a session (six).

Standardize each feature with training-only mean and population standard
deviation. Exclude training-constant features (standard deviation <=1e-12).
Squared distance is the equal-weight mean of the three group-wise means of
squared standardized differences over active features. A group with no
active features contributes zero. Break distance ties by object ID.

Primary predictor: take the first archived reference axis from each of the
three nearest distinct training objects. The three axes are unordered; no
reference-dependent deduplication or reranking is done. The use of the first
catalog solution is a fixed convention, not a claim that it is physical truth.

Secondary sensitivity: use all one to three archived reference axes of the
single nearest training object, padding with repeats of its first axis to
three slots. Repeated axes do not improve oracle@3. This checks the effect of
discarding alternative training-object solutions in the primary predictor;
neither result will be selected as the preferred baseline after scoring.

Seal all neighbour identities, distances, axes, feature values and training
scalers before evaluating held-out reference poles. Compare both fixed
baselines, the existing training-only atlas, and the archived K3 ensemble on
the same object-level antipodal oracle@3 endpoint. Report means, medians,
fractions within 20 degrees, and paired baseline-minus-K3 mean differences
with 10,000 fold-stratified object bootstrap draws, seed 20260916. These
intervals condition on the fitted models and feature rule; they do not cover
model selection or retraining uncertainty. A stronger result than these
particular baselines would not exclude every form of memorization.

## Unordered candidate-set stability

Use the saved axes for every completed degradation condition, including the
80-object observing-block grid, without new neural inference. Validate the
complete original object/condition/repeat inventory and original input hashes.

For each successful perturbed prediction, form the 3-by-3 matrix of axial
angles to the same object's full-input prediction. Enumerate all six
one-to-one assignments. Primary displacement is the smallest assignment mean;
secondary displacement is the smallest assignment maximum (the bottleneck
distance). Both disregard signs and candidate ordering. Also compute the
mean assignment displacement between each pair of sampling repeats.

Average repeat-level quantities within an object before population summaries.
Report mean (with the same conditional bootstrap), median, 90th percentile,
and maximum object-average displacement, alongside the paired change in
oracle@3. Report complete scheduled/success/failure/ineligibility counts.
Stability is conditional on two successful predictions; a failure is not
assigned an invented angular displacement. Report any missing distance and
its denominator explicitly. Failed eligible predictions retain the original
90-degree oracle-error penalty.

The compact report highlights six fixed conditions: half and quarter of
points, half and quarter of sessions, and 3% and 10% full-input flux noise.
All other conditions remain in the machine-readable output. For each condition
report the ten objects with largest object-average mean-assignment displacement
(ties by object ID), plus the complete per-object table. Do not select a
stability cutoff, infer a pole confidence interval, or claim stable axes are
necessarily correct. Candidate movement and oracle disagreement measure
different properties.

## Execution and acceptance

CPU-only. No training, inversion, new data download, or public release change.
Create a new output directory; bind protocol, implementation, input catalog,
study, schedule, saved predictions, bundle-role manifests and raw lightcurves
by SHA-256. The execution lock must reject changed inputs or code. Keep
prediction and scoring stages separate and refuse differing output overwrites.
Tests must cover role isolation, training-only standardization, tie handling,
reference-blind neighbour selection, assignment permutation/sign invariance,
known angles, repeat aggregation, missing/duplicate rows, and changed bindings.

The output is a local analysis report and reproducible command. Manuscript,
Overleaf, and release updates are a separate step after inspecting the results.
