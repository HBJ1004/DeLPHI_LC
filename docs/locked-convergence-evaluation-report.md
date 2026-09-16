# Locked convergence evaluation report

## Correction notice — 2026-09-11

**This execution is not valid evidence for a performance or quality claim.**
An independent AddressSanitizer/UndefinedBehaviorSanitizer check confirmed that
the expanded native solver did not reserve its internally appended three
regularization observations and one lightcurve. `asteroid_2512` exceeds the
observation allocation and `asteroid_109` exceeds the lightcurve allocation.
The latter was a development object, so the tolerance-selection run is also
affected. Passing Python tests did not establish native memory safety.

The original records and hashes below are retained without alteration. They
describe what ran, not a valid corrected experiment. A separately identified
capacity-safe solver and new execution are required. This notice does not
change the frozen neural oracle@3 predictions or any public release.

This report records the correction-bound DAMIT evaluation performed after the
solver-angle interpretation correction. It is a benchmark record, not a new
manuscript result.

## Inputs and scope

- Cohort: 140 locked DAMIT objects.
- Arms: baseline six-start classical inversion and guided six-start inversion.
- Repeats: three per object and arm (420 paired repeat cells; 5,040 starts).
- Selected convergence tolerance: `0.0003`.
- Study specification SHA-256: `ff97151e38584f5b2f5448e9b1f3ab7b6e9a6274eb51cf2a5007280937e971a3`.
- Revised lock SHA-256: `6388670f1b3c49862176ebcab5669c86d662e8854536b6328069213315fd9e8a`.
- Correction authorization SHA-256: `f44a5887c95b53ff3727aa987fd757e9756f3acd354e5793d2a9419c7991381d`.
- Frozen execution receipt SHA-256: `ab5b3286cadd8c41d9ac26fc8a1e4b974d4e5ba99b0c09adecb02bb00b1dee48`.
- Locked score SHA-256: `132a8dcc39f059e740b564032933a81d8fe201004ea9417b6b2f94eb468a6214`.

The execution receipt records `reference_catalog_opened: false`. Reference
axes were opened only by the separate scoring phase after execution completed.
All execution cells were retained and validated by the scorer.

The solver was invoked through the existing frozen-runner compatibility entry
point using the sealed `development-selection.json`; the separate authorization
file was independently verified but was not consumed by that invocation
(`solver_execution_started` remains false). The execution receipt itself binds
the same selected tolerance, lock, cohort, solver resources, and fit-selection
contract. This distinction is recorded so the authorization is not presented as
having controlled a run that had already started.

## Decision metrics

| Criterion | Result | Decision |
| --- | ---: | --- |
| Warm runtime ratio | 0.9040; empirical lower bound 0.8692 | Fail (required lower bound > 1) |
| Recovery difference | +11.43 percentage points; simultaneous lower bound −1.02 pp | Pass |
| Completion difference | 0; simultaneous lower bound −2.60 pp | Pass |
| RMS ratio | Not estimable; support 139/140 | Fail (full-support condition not met) |

The cold-runtime sensitivity was 0.7420 with lower bound 0.5753. It is not the
primary timing estimate.

One object, `asteroid_2512`, returned exit code `-11` (segmentation fault) in
all six starts of both arms across all three repeats. The failure was retained
as an execution outcome; it was not retried or removed from the denominator.
Because that object has no selectable fit, the prespecified RMS full-support
condition is not met.

## Interpretation

The four locked conditions are conjunctive. This evaluation therefore does not
demonstrate acceleration and does not establish RMS noninferiority. It provides
an observed higher recovery count at a matched six-start, convergence-stopped
budget, but that result is
separate from timing. A future acceleration study would require a new lock and
a work-reducing or time-to-quality design; these outcomes must not be reused as
an untouched confirmation set.
