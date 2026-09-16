# DeLPHI K3 reliability study and manuscript handoff

This document defines what evidence is needed before making observational
reliability statements about DeLPHI K3. The machine-readable design is
`repro/k3_reliability_protocol.yaml`. This is a protocol and reporting
contract, not a completed result. No long experiment was run and no manuscript,
published artifact, model, or release was changed while preparing it.

## Evidence boundary

The study is a retrospective, same-identity, out-of-fold reliability analysis
of all 170 original objects. Each object is predicted only by the five models
from its held-out outer fold. The three returned axes are an unordered set;
there is no post-hoc candidate selection or ranking. Every fold's training-only
axis atlas uses only its 108 training labels.

Before inference, the physical-identity mapping and synthetic-donor exposure
must pass their checksum-bound audits. Reference axes cannot affect thinning,
eligibility, fitting, condition selection, or runtime sample selection. These
controls support a same-cohort robustness statement; they do not turn the study
into prospective or new-identity validation.

The historical `analysis/data_degradation_experiment.py` evaluated the top ten
objects with one subsample. It is useful motivation but cannot support a
population reliability boundary. In particular, its 50-point number is not a
threshold carried into this study. The historical snapshot is provenance, not
a previous-version comparator, and this study must not be presented as a model
ranking or version ranking.

## Frozen conditions

Three deterministic seeds (20260915, 20260916, and 20260917) are available for
sampling. The deterministic full baseline and period-sensitivity cells use
repeat 0 once; stochastic degradation cells use the frozen one-or-three-repeat
schedule selected by the runtime rule. The full input is compared with point
fractions 0.50, 0.25, 0.10, and 0.05; session fractions 0.50 and 0.25; absolute
point caps 10, 20, 50, 100, 200, 500, and all; and mean-preserving positive
noise at 3% and 10% on the full and 0.10-point inputs. Absolute caps use a
seeded session order, allocate one atomic two-point pair to every session in
the largest prefix that fits, and only after every valid session has received
its pair allocate individual observations round-robin. If the cap cannot seed
every valid session, its residual slot is left unused (for example, cap 3 with
at least two valid sessions retains two observations). There is no padding.

An observing block merges overlapping native sessions and sessions separated
by less than 30 days. These blocks are not verified apparitions. The block
grid is 1, 3, 5, 10, and all blocks crossed with 5, 10, 20, 50, and all points
per block. It is restricted to the frozen common cohort with at least ten full
input blocks (preliminary count: 80). Every cell is compared with the grid-full
condition on this same eligible cohort, rather than with a broader full-input
denominator.

The grid-full baseline is an explicit alias of the already computed full
prediction, rescoped to the grid-eligible objects; it does not trigger duplicate
inference or duplicate baseline failures.

Period sensitivity recomputes every feature at P/2, 2P, and offsets of
plus/minus 0.01%, 0.1%, and 1% around the supplied period. It produces three
unordered candidates at every period. This measures input sensitivity; it is
not period recovery or period validation.

The clustering comparison uses all three K3 candidates, three axes from the
fold's training-only atlas, and frozen-seed random axial candidates. All three
methods are scored against identical reference-solution sets. The primary
random comparator scores the first frozen three-axis draw per asteroid, which
is also the realization used for the illustrative clustering concentration. A
separate secondary expected-random control uses 1,000 three-axis draws per
asteroid: error and within-20 are averaged over draws within the asteroid before
population summaries. These draws are not treated as independent asteroids.
The 15.772555-degree K3 and atlas neighborhoods are geometric search or
assignment radii, not confidence bounds.

## Forward holdout check

For objects with at least three constructed blocks, retain the latest block as
the withheld block. Eligibility also requires at least five retained native
sessions, 150 retained training points, and 20 withheld points (preliminary
count: 157). Each object has six fit cells total: classical, K3-neighborhood,
and training-atlas-neighborhood arms for each of the full and 0.10-point
sampling conditions. Both sampling conditions use exactly the same withheld
observations.

Fit degree/order 6 spherical harmonics with eight Gaussian-curvature rows at
the supplied fixed period. The classical arm uses the complete 20-degree start
grid; 20 degrees is start-grid spacing, not spherical-harmonic degree. Select a
fit using training RMS only. The operation is strictly forward: no withheld
flux, reference axis, or test statistic may enter fitting or selection. Score
fixed predicted flux using

\[
  \operatorname{RMS}_{\rm heldout} =
  \sqrt{\operatorname{mean}_{\rm curves}\left[
    \operatorname{mean}_{\rm points}(y-\hat y)^2\right]}.
\]

Curves receive equal weight, and there is no test-time offset, amplitude, or
normalization fit. The supplied period remains fixed, so this check cannot be
described as validating the period.

## Sealing, runtime, and scoring

The 24-hour compute allowance excludes implementation and starts immediately
before preparation builds the training-only atlases. Its persisted wall-clock
deadline charges interruption downtime. Every phase is limited by the earlier
of its own deadline and this global deadline: one hour for a ten-object pilot,
ten hours for inference, eleven hours for fits, and two hours for scoring and
export. Within each fold of the frozen 30-object development subset, the
pilot takes the median- and maximum-observation-count objects with object ID as
the tie-break. It reports median and maximum observed runtime while scores
remain sealed. Use three repeats only if
the conservative projection fits in the inference allocation; otherwise use
one repeat uniformly across stochastic conditions and objects. If even one
uniform repeat is infeasible, report the study as infeasible instead of
dropping selected conditions.

The inference projection uses the maximum observed runtime separately for each
condition and the global maximum observed cell runtime for any unmeasured
condition. It multiplies these rates by the exact planned full-cohort cell
counts (using only the grid-eligible count for grid cells), then applies a 2x
factor and 300 seconds of fixed overhead. Any failed pilot prediction blocks
schedule selection.

For fitting, choose the largest of all, 125, 100, 75, 50, or 30 objects whose
projection fits the allocation, using twice the maximum pilot runtime for a
six-cell object. Object order is fixed-hash and fold-balanced. Failed objects
are never replaced.

Each score row is one ensemble object-condition-repeat result, with `object_id`,
`fold`, `condition`, `repeat`, sampling `seed`, and `status`; `model_id` defaults
to `k3`. Repeat indices 0, 1, and 2 bind to seeds 20260915, 20260916, and
20260917 respectively.

Successful rows contain a 3-by-3 `axes` array and `error_deg`. Failed eligible
rows receive the predeclared 90-degree angular error. Ineligible rows are kept
but are not attempted predictions. Scoring starts only after the complete
frozen condition-specific schedule—including explicit failures and ineligible
rows—has been sealed and checksum-verified.

Repeats are averaged within asteroid before population summaries and the
10,000-resample bootstrap samples asteroids within their original folds, with
seed 20260915. The mean is the mean of asteroid repeat-mean errors. The median
is the median of those object-level means. The within-20 result averages
repeat-level indicators within asteroid before averaging asteroids. Failure
counts, attempted-repeat counts, eligible counts, and unmatched counts are
reported beside every score. Intervals are descriptive bootstrap intervals,
not a tuned acceptance gate.

Runs live in separate immutable directories. Every sealed record is one atomic
canonical-JSON envelope containing its payload and payload SHA-256, avoiding a
payload/sidecar interruption window. Resume is allowed only when input,
schedule, code, and completed-cell checksums agree. Scoring produces new JSON,
CSV, TeX, per-object, and failure-ledger artifacts; it never overwrites a
published result.

## Required evidence and degradation table

The final report must materialize the following table from the sealed scoring
JSON. A row without its denominators is incomplete evidence. Values must not be
filled from a pilot, a partial run, or the historical ten-object analysis.

| Family | Condition | Scheduled objects | Eligible objects | Attempted repeats | Failed repeats | Failure rate | Mean error (fold-bootstrap interval) | Median object-repeat mean | Mean repeat within 20 deg | Held-out normalized-curve RMS | Matched baseline | Matched N | Mean degradation (fold-bootstrap interval) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| Full | full | generated | generated | generated | generated | generated | generated | generated | generated | generated | none | -- | -- |
| Point thinning | 0.50 / 0.25 / 0.10 / 0.05 | generated | generated | generated | generated | generated | generated | generated | generated | generated where registered | full, matched | generated | generated |
| Session thinning | 0.50 / 0.25 | generated | generated | generated | generated | generated | generated | generated | generated | -- | full, matched | generated | generated |
| Positive noise | 3% / 10%, full / 0.10 points | generated | generated | generated | generated | generated | generated | generated | generated | -- | matching no-noise input | generated | generated |
| Count cap | 10 / 20 / 50 / 100 / 200 / 500 / all | generated | generated | generated | generated | generated | generated | generated | generated | -- | cap-all, matched | generated | generated |
| Block grid | blocks x points per block | generated | generated | generated | generated | generated | generated | generated | generated | -- | grid-full on eligible 80 cohort | generated | generated |
| Period | P/2; offsets; P; 2P | generated | generated | generated | generated | generated | generated | generated | generated | -- | P, matched | generated | generated |
| Candidate control | K3 / train atlas / random | generated | generated | generated | generated | generated | generated | generated | generated | -- | declared matched control | generated | generated |
| Forward fit | full / 0.10 points | generated | generated | generated | generated | generated | as applicable | as applicable | as applicable | generated | full, same withheld block | generated | generated |

The supporting evidence bundle must also contain the frozen protocol and
condition schedule; identity/exposure audit receipts; exact input, split,
model, and code hashes; pilot timing report; prediction and fit receipts;
per-object repeat aggregates; all failure/ineligibility codes; matched-pair
membership; bootstrap seed; and the generated CSV and TeX tables. Preliminary
counts (80 grid-eligible and 157 holdout-eligible) are planning observations,
not final denominators.

## Observational guidance contract

Guidance is derived from the shape of the matched degradation curves and must
remain conditional on this cohort, preprocessing, supplied-period assumption,
and observing geometry. The report should say which tested regimes preserved
similar errors and failure rates, where degradation became material, and how
session/block coverage and point count interacted. It should separately discuss
period perturbation, noise sensitivity, candidate controls, and forward
held-out fit behavior.

Do not collapse the table into a universal cutoff such as "at least N points."
Point count cannot encode temporal leverage, geometry, session distribution,
noise, preprocessing, or period error. A useful observational statement has
the form: "Within the matched eligible retrospective cohort, condition X had
this change and failure rate relative to baseline; observations outside these
tested conditions remain uncharacterized." If strata are sparse or intervals
are wide, the correct guidance is inconclusive.

## Old-to-new manuscript checklist

The comparison uses the saved Overleaf `origin/master` snapshot, not a newly
fetched live project. Content carried forward or deliberately left separate:

| Earlier section | Treatment in this study |
|---|---|
| Absolute-count degradation and input requirements | New count-cap and block-grid measurements; separate software limits from measured guidance |
| Period estimation | Supplied-period sensitivity only; no period-estimator claim |
| Error versus latitude, period, amplitude, and observation count | Descriptive out-of-fold strata; latitude is not the viewing aspect angle |
| Preferred regions in candidate poles | All-three-candidate concentration and sky plots, with atlas and random references |
| Feature removal and attribution | Retain current published removal controls; do not transfer old attribution percentages |
| Variable candidate count, architecture changes, and smaller networks | Deferred because these require separate training and evaluation |
| Thousands of lower-quality DAMIT objects | Not relabelled as independent validation; exposure and reference quality need separate checks |
| Reference uncertainty and search-time arguments | Retain reference limitations and measured timing; do not substitute a geometric extrapolation for end-to-end timing |

The existing external survey results are not erased by this study. New-identity
validation remains a separate question even if these internal checks are favorable.

This checklist is a handoff for a later, separately reviewed manuscript change.
Completing it does not itself authorize editing the manuscript.

- [ ] Confirm that all 170 identities use only their held-out-fold five-model
  ensemble and that every atlas contains only 108 training-fold labels.
- [ ] Attach passing physical-identity and synthetic-exposure audit receipts
  created before inference.
- [ ] Verify the frozen expected-row schedule exactly, including explicit
  failures and ineligible cells, before reference scoring.
- [ ] Replace any 50-point or other universal observation-count claim with the
  complete matched degradation table and conditional observational guidance.
- [ ] Describe the old ten-object, one-subsample analysis as motivation only;
  do not reuse its number as a cutoff or portray it as population evidence.
- [ ] Report full denominators and the registered 90-degree failure convention
  alongside means, medians, within-20 fractions, and intervals.
- [ ] State that repeats were averaged within asteroid and bootstrapped within
  original outer folds; do not treat repeats as independent objects.
- [ ] Use each grid condition's own matched grid-full baseline and report the
  grid cohort denominator rather than comparing it with all 170 objects.
- [ ] Call merged sessions "observing blocks," not apparitions, unless a
  separate apparition verification is completed.
- [ ] State that P is supplied and fixed and that period perturbations measure
  sensitivity rather than period-estimation accuracy.
- [ ] Describe the 15.772555-degree neighborhood as a geometric radius, never
  as a confidence interval or calibrated uncertainty.
- [ ] Confirm that held-out flux and reference axes were unavailable to fitting
  and training-RMS fit selection; report the exact withheld metric.
- [ ] Keep the three candidates unordered and avoid candidate or model-version
  rankings.
- [ ] Label the evidence retrospective and same-identity; do not call it
  prospective, external, or new-identity validation.
- [ ] Present positive, negative, or inconclusive findings without a post-hoc
  promotion cutoff, and explicitly delimit untested observing regimes.
- [ ] Check that manuscript tables were generated from the sealed JSON/CSV/TeX
  outputs and that no pilot or incomplete-run values entered them.
