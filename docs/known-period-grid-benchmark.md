# Known-period pole-grid benchmark

This is a separate, retrospective follow-up experiment. It changes neither the
published model nor the frozen paper results. It is not external validation,
an unknown-period inversion benchmark, or a new untouched confirmation cohort.
The development data, original oracle errors, and earlier follow-up outcomes
have already been inspected. The tolerance was previously selected using a
solver with unsafe internal capacities; this experiment retains that tolerance
as a fixed follow-up choice, not as an independently validated optimum.

## Question and scope

Measure the elapsed time from input to the selected classical pole-and-shape
fit, with the period supplied and fixed in all arms. Compare a full pole grid
with a smaller set of starts constructed from the current model's three axes.
Selection is the minimum valid final relative RMS, with start index breaking
ties. Reference poles are not available to execution or selection. This is
automated fit selection, not validation of neural ranking or proof of a unique
physical pole. Human review time is not included or estimated.

| Arm | Initial signed directions | Purpose |
| --- | --- | --- |
| classical20 | 20-degree longitude/latitude mesh, poles once: 146 starts | Primary baseline |
| guided20 | Same mesh inside candidate caps, plus six signed centers | Primary guided arm |
| classical15 | 15-degree mesh, poles once: 266 starts | Sensitivity baseline |
| guided15 | Same mesh inside candidate caps, plus six signed centers | Sensitivity guided arm |
| standard6 | Six fixed DAMIT starting directions | Descriptive practical reference |

The mesh is an explicit latitude/longitude construction, not equal-area
sampling or a claim that all classical implementations use this many starts.
For guided starts, keep a mesh vector when its antipode-aware separation from
any candidate is at most 15.772555008039914 degrees. Add each candidate and its
negative and remove duplicate signed directions. The radius is the previously
reported mean oracle error on these same 170 objects. It is not a confidence
radius and does not guarantee containment. The optimizer may leave the caps;
these regions restrict initialization only. Both pole signs are tested.

All 170 original objects use their correct held-out-fold five-model ensemble.
Every object has all five arms, cold and warm modes, and three repeats. Models
are not retrained. Guided predictions are computed again inside each timed
case; archived prediction maps are not substituted for neural execution.

## Timing and solver validity

Every arm uses six single-threaded native solver workers, the same corrected
binary and parameter settings, convergence tolerance 0.0003, a 1,000-iteration
ceiling, and a 300-second per-start timeout. Nonzero exits, timeouts, missing or
nonfinite products, nonpositive iteration counts, and iteration-ceiling hits
are invalid fits. A case completes only after every requested start has been
attempted and at least one valid fit exists. Failed starts and failed objects
remain in the records and denominators. RMS uses per-lightcurve mean-normalized
observed and modelled brightness across all observations.

Cold timing starts before a fresh Python worker and includes imports, input
reading, model loading if required, prediction, search, classical selection,
and writing the result. This is process-cold latency, not cold filesystem or
GPU-driver caches. Classical cold workers do not import the neural stack.
Warm timing uses a persistent worker with a loaded ensemble and one synthetic
warmup input, never an evaluation input. Warm case timing includes IPC, input
reading, prediction if guided, native search, selection and result output;
worker setup is separately recorded. Models are not resident during cold
blocks. Fold/mode blocks and cases within each block have a fixed randomized
order. Parent-measured case wall time is the primary endpoint, not the sum of
overlapping worker times. Native child CPU time is recorded separately.

The lock binds source files, Python and dependency versions, CPU affinity and
model, GPU identity and driver, weights, raw inputs, supplied periods, splits,
and native source and executable hashes. Historical capacity-v1 receipts are
rejected: the corrected solver must allow its extra internal lightcurve and
three closure observations, with real boundary sanitizer checks and supported
input parity recorded. See `solver-internal-capacity-correction.md`.

## Pilot, budget, and scoring

Five predetermined development objects, one per outer fold, form the engineering
pilot. Each receives the full 30-case factorial. This pilot can estimate cost
and expose implementation failures but cannot authorize a cohort acceleration
claim. The conservative cost projection is twice pilot elapsed time multiplied
by the larger of the object-count and observation-count scale factors. This is
a planning heuristic, not a statistical upper confidence bound. The first
locked engineering pilot projected 88.4 hours and therefore stopped under its
24-hour ceiling. After that stop was retained, a separately identified v2
protocol set a seven-day ceiling. Its five-object pilot completed, but the
system Python executable changed before its full execution could start. A v3
environment-relocked protocol therefore repeats the engineering pilot with a
benchmark-owned interpreter. No scientific endpoint, comparison, or decision
criterion changed. A deadline interruption retains partial artifacts but cannot
produce a complete-cohort verdict.

After all fit selections are sealed and checked, the scorer opens the reference
catalog. Repeats remain clustered within their asteroid. The primary contrast
is classical20/guided20 process-cold wall time, using 10,000 fold-stratified
paired asteroid bootstrap resamples (seed 20260911, 2.5/97.5 percentiles).
Recovery means the selected axis is within 20 degrees of an admissible catalog
axis in at least two of three repeats. Completion likewise requires two of
three repeats. Conservative simultaneous exact binomial bounds for paired
favorable/adverse outcomes must establish both noninferiority differences at
or above -5 percentage points. The geometric mean RMS ratio is computed from
object mean log ratios, requires at least two jointly completed repeats for
every predeclared object, and must have a bootstrap upper bound below 1.01.
The runtime lower bound must exceed 1. All checks must pass for a scoped
acceleration statement. Warm20 is reported separately; the 15-degree
sensitivity and standard6 arm cannot rescue a failed primary contrast.

No result from this experiment establishes generalization to a new survey.
Existing negative external-transfer results remain part of the record.

## Commands

Use the same interpreter for every phase, from the source directory. Supply
existing paths; each output is create-once and must be new.

```sh
python -m repro.run_k3_grid_benchmark freeze \
  --blind-inputs repro/data/k3-followup-20260910/convergence-blind-inputs.json \
  --splits repro/data/damit-20250610T000301Z/publication-splits-v2.3.json \
  --development repro/data/k3-followup-20260910/development-objects.json \
  --dump-root ../../damit-20250610T000301Z \
  --bundle-root ../../DeLPHI-publication/DeLPHI-k3-release-assets-20260906 \
  --capacity-report ../data/solver-internal-capacity-corrected/20260911/solver-capacity-preflight.json \
  --reference-catalog repro/data/damit-20250610T000301Z/catalog.jsonl \
  --device cuda --output ../data/pole-grid-benchmark/run/lock.json
python -m repro.run_k3_grid_benchmark execute \
  --lock ../data/pole-grid-benchmark/run/lock.json \
  --role pilot --output ../data/pole-grid-benchmark/run/pilot
python -m repro.run_k3_grid_benchmark continue \
  --lock ../data/pole-grid-benchmark/run/lock.json \
  --pilot ../data/pole-grid-benchmark/run/pilot/execution.json \
  --root ../data/pole-grid-benchmark/run
```

The continuation performs the full blind execution before it opens the
reference catalog for scoring. It writes create-once start, completion, or stop
receipts. Scoring permits an environment-independent audit while still checking
the frozen source, inputs, weights, solver, and every retained product hash.

Do not edit runtime sources after freezing. A required correction needs a new
lock and new output directory; retain and label the superseded execution.
