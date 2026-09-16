# DeLPHI validation options and acceleration benchmark plan

## Assessment

External validation sources have not been exhausted. The completed work covers corrected ZTF transport on existing identities and a 30-object ALCDEF-to-Gaia transfer test on identities absent from the released model's real cohort and declared synthetic donors. TESS was inventoried, not evaluated. ATLAS, Gaia photometry as model input, additional dense-lightcurve identities, and independently supported reference subsets remain available routes. They address different questions and should not be treated as interchangeable tests.

A DAMIT benchmark can support an acceleration claim. It must compare the time needed for comparable scientific output under a defined solver, input contract, hardware configuration, and failure policy. An external survey is not a prerequisite for that limited claim. The current fixed-work result does not establish acceleration, and the convergence follow-up has not produced a locked-evaluation result.

There is also a newly identified defect in the convergence follow-up's output validation. All 105 start records classified as numerical-output failures across its five development settings fail solely because the printed latitude is outside the usual coordinate range. The solver does not constrain its angle parameters to that range. These outputs represent finite directions and need coordinate conversion, not automatic rejection. A read-only diagnostic changes the development eligibility assessment, but still does not produce an accepted acceleration result. This correction should precede another expensive experiment.

The recommended order is: correct and audit that interpretation; complete the existing benchmark if its unchanged selection rule authorizes it; establish a new-identity, dense-lightcurve test close to the intended input domain; and pursue one additional instrument-transfer test with explicit data and reference lineage. Any adaptation must use development data and a separate final cohort. The executable handoff is in [Terra/Luna implementation handoff](terra-luna-validation-handoff.md).

## Existing evidence

| Study | Executed evidence | Interpretation |
| --- | --- | --- |
| Original DAMIT out-of-fold evaluation | 170 objects; mean oracle@3 15.77 degrees | Evidence for three-axis candidate coverage on this cohort, conditional on the supplied period. Not unique-pole selection or evidence of transfer to every survey. |
| Original fixed-period benchmark | Equal inversion iteration totals; baseline/guided wall-time ratio about 0.947; recovery 106/170 versus 119/170 | Guided initialization improved observed recovery at the fixed iteration budget, but took longer. |
| Corrected ZTF transport | 161 usable objects; all 170 retained in the primary denominator; mean 30.00 degrees on usable inputs and 33.17 with input failures assigned 90 degrees | Negative same-identity instrument-transfer evidence. It is not a new-identity test. The older wrongly joined ZTF result is invalid and must remain excluded. |
| ALCDEF input, Gaia reference axes | 30 model-training-unexposed objects; K3 mean 37.25 degrees, atlas mean 24.07 degrees; atlas-minus-K3 difference −13.19 degrees, paired 95% interval [−21.53, −5.26] | A completed, negative retrospective external transfer test. It is incorrect to describe this as no external test having occurred. It does not support positive transfer. |
| Capacity-corrected convergence development | Five tolerances, 30 objects, three repeats, six starts per arm: 450 paired repeats and 5,400 solver starts | Existing selection file says no eligible tolerance. That assessment is affected by the angle-validation defect described below. |
| Locked convergence evaluation | The 140-object stage has not run | No confirmatory acceleration estimate exists. These are benchmark-held-out DAMIT objects, not a new astronomical population. |

The first two rows are defined by the frozen publication artifacts. The transport and convergence rows are defined by the follow-up artifacts listed under Local evidence below. The ALCDEF test uses all 25 released models; the original out-of-fold evaluation uses the five models from the object's held-out fold. Their aggregate errors should not be described as a controlled comparison of cohort effects alone.

The external result is a material limitation, not proof of a particular cause. Overfitting, survey/input mismatch, insufficient observing geometry, inaccurate supplied periods, and reference uncertainty remain distinguishable hypotheses. The result against the train-only atlas means that a successful transfer claim cannot currently be made. It cannot be dismissed merely by naming distribution shift.

## Remaining validation sources

### Dense lightcurves on new identities: first priority

The most informative first question is whether the frozen method works on previously unused identities when the inputs resemble its intended dense, multi-epoch setting. Changing asteroid identity, instrument, cadence, preprocessing, period quality, and reference method simultaneously makes a failure difficult to diagnose.

The local ALCDEF screen contains 943 physical-identity matches to the DAMIT identity table after preliminary session/point/year filtering. That number is not an eligible test-set size: training and donor overlap, unresolved history, actual apparitions, period provenance, and reference lineage remain to be resolved. A separate ALCDEF–Gaia screen identifies 407 identities absent from the released model's cohort and declared donors, of which 30 have already been scored. The remaining candidates must not be assumed to satisfy all scientific inclusion rules.

Use native DAMIT-format data for an initial same-format new-identity test, or independently acquired dense ALCDEF sessions for a stronger cross-source test. Freeze the model, admission rules, period source, and all admissible reference axes before prediction. An older observation is still a valid retrospective test of a frozen model if it was not used in its training or selection; it need not have been collected after publication. Conversely, a recent database entry is not automatically a new asteroid.

ALCDEF's PDS archive contains raw time-series photometry and metadata. Its live archive may provide additional sessions, but it also reports duplicate-submission issues; acquisition must preserve version and duplicate lineage. Multiple sessions or calendar years are screening variables, not proof of independent observing apparitions.[^1][^2]

### Additional instrument and reference routes

| Route | Available evidence or access | Suitable role | Main restriction | Priority |
| --- | --- | --- | --- | --- |
| ATLAS Solar System Catalog / PDS archive | The project advertises SSCAT v3 with over 200 million observations of approximately 700,000 small bodies. PDS provides an archive and moving-object search through CATCH.[^3][^4] | A second ground-based, multi-season transfer test, with independent reference axes | Use moving-object products, separate filters, and audit overlap with the observations used to obtain reference models. Catalog availability is verified; eligible DeLPHI overlap has not been counted. | First additional survey |
| TESS TSSYS-DR1 | Public release of 9,912 extracted lightcurves; local metadata already acquired. Archive documentation includes observing times, photometric errors, flags, and spacecraft/solar distances.[^5][^6] | Dense-photometry and period/preprocessing check; new-identity transfer where sufficient aspect coverage exists | A long, well-sampled lightcurve need not constrain a pole. Spacecraft geometry must replace the Earth-center approximation. Assess standalone and predeclared combined-data roles separately. | Low-cost feasibility census; secondary pole test |
| Gaia DR3 epoch photometry | SSO photometry is available in `gaiadr3.sso_observation`, including G flux, error, and magnitude.[^7] | Sparse space-survey transfer, preferably against references not fitted to those Gaia observations | Transit-level photometry must not be duplicated once per CCD. Time scale, light time, spacecraft geometry, and distance reduction need explicit handling. The existing Gaia spin table has been used as a reference, not as model-input photometry. | After sparse-input feasibility |
| Radar-supported spin/shape references | PDS lists several radar shape-model collections and object-specific products.[^8] | A small physically different reference check paired with optical inputs | Some models also use optical data. Audit that lineage, uncertainties, rotation state, and synthetic-donor exposure. Do not treat every radar product as an independent precise pole. | Parallel reference census |
| Resolved imaging / occultation-supported references | The VLT/SPHERE survey modeled 42 large main-belt asteroids using resolved imaging alongside other observations.[^9] | A stronger-reference subset for catalog-axis agreement | Joint inversion can reuse optical lightcurves. Large bright targets are a selected population and may already be training or donor identities. | Parallel reference census |
| New observing campaigns or genuinely later cohorts | New photons and independently determined references can be held out prospectively | Strongest temporal validation for a fixed method | Data access, observing geometry, and reference determination may take months. A later reference alone does not make all input data independent. | Longer-term, not a submission prerequisite |

ATLAS is not a speculative source: published inversion of its photometry produced thousands of models. Some are only partial pole solutions, so a longitude-unconstrained record cannot be used as a precise three-dimensional reference.[^10] The SSCAT README endpoint timed out during this assessment, while the official project and PDS access pages were available. Bulk acquisition is therefore not yet demonstrated end to end.

Gaia DR3 modeling explicitly includes stability tests, comparisons with other period/model catalogs, and correction of erroneous solutions. Its reference axes are model-derived, not direct physical truth. Its methods also show why observation count alone is insufficient and why period and photometric conventions need careful handling.[^11]

Other archive collections, including PDS optical lightcurves and SDSS moving-object data, can be considered if the first-priority sources do not yield enough eligible objects.[^12] A longer source list is not itself stronger evidence. Infrared photometry also requires a different physical input model and should not be passed directly into an optical-brightness predictor. Future releases are not dependencies of the immediate plan.

## Validation design and integrity

### Separate the questions

Maintain three distinct evaluations: new identities under comparable input conditions; a new instrument or cadence, with identity exposure stated; and agreement with more independent reference determinations. An object can belong to more than one descriptive evaluation, but those rows are not independent replications. The physical identity is the sampling unit.

An exposure ledger should distinguish direct training, validation/model selection, synthetic shape/geometry/noise donors, previously inspected outcomes, and unknown provenance. A conservative project-history audit is useful, but merely appearing in an old source inventory does not establish training leakage into the released model. Conversely, a clean real-training split does not clear synthetic-donor leakage. DAMIT's database identifier must always be joined explicitly to its MPC physical identity.[^13]

The 30 already scored ALCDEF objects cannot be reused as an untouched final test after development on their outcomes. Preserve their original result and use them only as declared diagnostics if needed. If adaptation is pursued, reserve separate identities before comparing variants, and seal both frozen-model and adapted-model predictions before final scoring.

### Diagnose the negative transfer before training another model

First audit the input contract on already exposed development objects. Check identity joins, time scales and light-time conventions, units, vector direction and coordinate frame, phase folding, filter handling, native session boundaries, observation counts, and normalization. For TESS and Gaia, use the actual spacecraft observer. DAMIT documents corrected observing times and a specific intensity/geometry convention; a file being syntactically valid does not establish equivalence to that convention.[^13]

Audit supplied periods back to the original measurement, not just the SBDB summary field. Record precision, ambiguity, and whether the period is sidereal or synodic. SBDB exposes physical-parameter references and notes that support this audit.[^14] A controlled development test should perturb otherwise correct DAMIT periods to the precision and uncertainty of the external inputs. This tests a causal explanation without selecting a better external period by its pole error.

The completed ZTF factorial already found little benefit from distance and first-order light-time changes, and worse results from the tested regrouping. Do not repeat that grid expecting a different conclusion. Additional ALCDEF checks should address its distinct preparation: session-midpoint geometry, raw-session normalization, external period quality, and documented observing coverage. None is yet an established explanation for its poor result.

Add a conventional-inversion information check on a small, predeclared development subset. Give the solver the same photometry and supplied period as DeLPHI, without reference poles in initialization or stopping. If classical inversion also cannot identify a stable pole, the inputs may be insufficient for the endpoint. If it succeeds while K3 fails, the evidence points more directly to the learned model or representation. Both outcomes are diagnostic; neither retrospectively changes the frozen external score.

### Strengthen validation without acquiring another survey

Use object-grouped validation, keeping all sessions and synthetic derivatives of an asteroid together. Add family- or orbit-blocked evaluation when the intended claim is transfer beyond those groups. Such blocking tests a harder generalization question and should not be substituted silently for ordinary object-level interpolation. The methodological reason for matching cross-validation structure to the dependence and intended prediction task is established; applying it to asteroid families is a study-design recommendation.[^15]

If retraining or selecting preprocessing, use nested development splits. Decisions based on cross-validation results can themselves overfit; an out-of-fold label does not make repeated model selection unbiased.[^16] Keep the final target identities outside all adaptation, normalization fitting, atlas fitting, early stopping, and hyperparameter selection.

Keep the current train-only three-axis atlas and add a reproducible uniform-random three-axis descriptive reference with the same number of admissible catalog solutions. Retain the real input-swap and synthetic label-shuffle controls as distinct domain-specific checks. A new full-pipeline label-permutation experiment would test a different null and requires a valid grouped randomization scheme and retraining; it is not equivalent to shuffling already generated predictions.[^17] No prior-version model or candidate-ranking result is needed.

Useful low-cost additions are prespecified performance strata by number of reference axes, observing geometry, point/session count, period quality, and reference tier; candidate-set stability under withheld sessions and realistic noise; and held-out-apparition photometric prediction after downstream fitting. Stability is not accuracy, and good held-out flux prediction is not proof of a unique pole. These checks strengthen different parts of the physical and statistical argument.

Report object-level intervals, failures, and the entire eligibility flow. Repeats, epochs, candidates, and ensemble members are not extra independent asteroids. Bootstrap intervals on fixed predictions are conditional on those trained models; they do not include all training and model-selection uncertainty. Select sample size by an explicit precision or paired-power calculation using development variability, rather than treating 30 or 100 as a universal adequacy threshold. New conformal calibration would need a separate target-domain calibration set; the original calibration does not automatically transfer to a new survey.

## DAMIT acceleration assessment

### Existing fixed-work result

Both original benchmark arms performed 43,200 inversion iterations. The guided arm incurred 103.9 seconds of neural work in a total wall-time increase of 106.9 seconds. Its observed end-to-end runtime is therefore not evidence of acceleration. The recovery increase from 62.4% to 70.0% remains a separate fixed-budget result.

One earlier audit statement needs qualification: equal iteration counts do not make every possible wall-time advantage mathematically impossible. Iteration cost can depend on the numerical path. The supported conclusion here is that the design did not reduce iteration work and the measured timing showed a slowdown. References explaining expensive pole or period searches motivate a new benchmark, but cannot supply DeLPHI's missing speed measurement.

### Angle-validation defect and diagnostic reanalysis

The native solver updates its angle parameters without a latitude constraint. `convexinv.c` prints `90 - theta` as latitude, and `blmatrix.c` uses the corresponding sine and cosine values. The follow-up checker in `lc_pipeline/k3/convergence_benchmark.py` rejects any printed latitude outside [−90, 90]. This confuses a nonstandard coordinate representation with a failed solution.

For a printed longitude `l` and latitude `b`, first form the direction

`p = (cos(b) cos(l), cos(b) sin(l), sin(b))`.

Then use `atan2(p_y, p_x)` and `atan2(p_z, hypot(p_x, p_y))` to obtain standard longitude and latitude. Preserve the raw values. This is interpretation of the reported axis, not a change to the solver's fitted shape, phase, objective, or iterations. Do not simply clip latitude; clipping changes the axis.

For example, a stored result has longitude 92.878146 and latitude 101.510344 degrees, return code zero, 68 iterations, finite positive RMS, and validated output files. Its standard axis coordinates are longitude 272.878146 and latitude 78.489656 degrees. The direction is unchanged. All 105 classified numerical-output failures in the grid violate only the latitude-range check; none is classified as a timeout or iteration-cap event. Across the grid, converting to a standard direction representation preserves vector components to approximately 1.1e−15 in a read-only calculation.

The following table is a **post hoc diagnostic**, not a replacement for the frozen score files. It includes those finite outputs, selects the minimum-RMS fit across all six starts again, and applies the existing development point thresholds without changing their values. All 30 objects would have joint completion support under that interpretation.

| Tolerance | Archived warm runtime ratio, baseline/guided | Reinterpreted baseline / guided recovery, out of 30 | Reinterpreted geometric RMS ratio, guided/baseline | Existing development point rules |
| --- | ---: | ---: | ---: | --- |
| 0.01 | 0.7027 | 9 / 16 | 0.8802 | Would pass |
| 0.003 | 0.7290 | 11 / 13 | 0.8905 | Would pass |
| 0.001 | 0.8459 | 16 / 21 | 0.9439 | Would pass |
| 0.0003 | 0.9841 | 18 / 17 | 0.9692 | Would pass |
| 0.0001 | 1.1589 | 24 / 22 | 0.9850 | Would fail recovery: −6.67 percentage points |

The raw timing measurements do not change in this diagnostic. Fresh operational timings would still need to include any changed interpretation overhead. Formal correction requires regression tests, verification of archived hashes, a new derived interpretation record, renewed phase separation, and a documented correction history. No archived output should be edited.

If verified, the existing selection rule would choose tolerance 0.0003 from the eligible settings: it has the largest archived runtime lower bound among them, about 0.936. Its development point ratio is about 0.984, so development still shows no positive speed advantage for the eligible choice. The faster 0.0001 setting cannot be promoted while ignoring its recovery loss. This finding changes the reason for the study's stop and may allow the planned holdout test; it does not justify a speed claim now.

### Defensible next benchmark

The first route is to finish the already specified study after a verified implementation correction. Retain its 30/140 split, six starts, tolerance grid, fit-selection rule, neural timing definitions, recovery/completion margins, RMS threshold, and acceptance-bound terminology. Use the original selection rule on corrected development results. If authorized by that rule, execute the 140-object stage once and report every criterion, including a null or negative conclusion. Those 140 identities have appeared in the original manuscript evaluation; they are untouched by this convergence experiment, not universally unseen.

A later work-reducing design can instead measure time to a common objective-quality target. Fixed-target optimization benchmarks and performance/data profiles provide a sound methodological basis for that question, including explicit treatment of unsolved problems.[^18][^19][^20] The objective should measure valid fit quality, with reference-pole recovery assessed separately and never used for stopping or start ordering.

For such a new design, define one primary quality target and a small secondary target grid before test results. An independent high-budget reference solve can define an offline target, but that target is a benchmark device: it is not known to a deployed pipeline. A deployed acceleration claim additionally needs a stopping rule available without that reference solve. Report those as separate experiments if both are attempted.

Compare standard signed starts with all three K3 axes expanded to six signs, preserving the unordered candidate interpretation. Begin with convergence stopping rather than discarding candidates by unvalidated neural scores. A staged scheduler or fallback search can be a later variant, chosen only on development data and fixed before evaluation. Its runtime must include unsuccessful starts and fallback work. A fast exit with no acceptable solution is not a speedup.

Count solver evaluations, iterations, actual elapsed time, inference, and loading/preparation costs under declared cold and warm conditions. The existing warm timing excludes model loading and lightcurve I/O; its cold sensitivity is model-load cold in an initialized process, not operating-system-cache cold. Keep CPU/GPU resources and concurrency explicit. A same-hardware CPU comparison and the intended heterogeneous deployment configuration answer different questions. Training cost is offline but should be disclosed, including break-even use volume if amortized savings are claimed.

Keep every object in time-to-target and success summaries. At a fixed cap, an unsolved object contributes capped time and a recorded target failure; it is not dropped from a successful-pairs mean. Use profiles and quality/recovery summaries together. For any noninferiority claim, choose margins from scientific tolerances before evaluation, document why they are acceptable, and require both adequate recovery and fit quality. Do not widen a margin to admit the observed 0.0001 result.

The maximum warranted conclusion, if the relevant criteria pass, is that K3 initialization reduced runtime for the specified known-period convex-inversion solver on a retrospective DAMIT out-of-fold benchmark. It would not establish acceleration of unknown-period global inversion, every classical solver, or ZTF/ALCDEF processing. Literature support for period-search cost does not bridge that gap.

## Implementation sequence and resource limits

| Stage | Deliverable | Release condition / stop condition |
| --- | --- | --- |
| 1. Correct output interpretation | Tested canonical-axis decoder; raw-output audit; new derived development records | Independent vector-equivalence and negative-case tests pass; raw hashes unchanged; existing numerical thresholds unchanged. Otherwise stop. |
| 2. Reassess original convergence study | Formal selection using all five corrected development settings | Follow the unchanged rule. If it selects no setting, retain that result; do not choose the fastest ineligible setting. |
| 3. Complete its holdout, if authorized | One sealed 140-object execution and full scoring report | Lock code, binary, inputs, interpretation revision, timing, and selected tolerance before execution. All four criteria must pass for the stated claim. |
| 4. Build validation eligibility matrix | Metadata-only identities, data/reference lineage, period quality, apparitions, exposure and exclusion reasons | Do not fetch or score large new cohorts until the intended test and selection rule are fixed. Unknowns remain unknown. |
| 5. Diagnose input/physical sufficiency | Small development comparison: native DAMIT preparation, external preparation, controlled degradation, classical information check | Choose a finite diagnostic set in advance. Record all results; do not use the final cohort to choose preprocessing. |
| 6. Lock primary new-identity test and one survey test | Frozen manifests, code/model hashes, atlas, random control, analysis, and precision calculation | Prefer dense new identities first; ATLAS is the first additional survey candidate. TESS remains a useful low-cost feasibility and period check. |
| 7. Optional adaptation | Survey-matched synthetic training, then a separately scored adapted model | Only after diagnostics; separate training identities and final test. Preserve the frozen model's external result. |

Stages 1–2 require no training and can primarily reuse completed solver outputs. A fresh five-tolerance development rerun would be a fallback if reliable replay cannot be implemented. Recorded solver time for the 30-object 0.0003 setting is approximately 1,366 seconds across both arms and repeats. Simple 140/30 scaling gives about 1.8 serial hours for the locked solver work, excluding overhead and allowing no guarantee about different object difficulty. This is a planning estimate, not an ETA or an acceleration result. The full five-setting development solver time was approximately 1.53 hours.

The earlier adaptation allowance of at most 48 GPU-hours remains a ceiling, not an instruction to spend it: two hours for a pilot, sixteen for a balanced screen, and thirty for a selected full ensemble. Require a development result and cost report before expanding a stage. Do not combine independently selected per-fold winners into an apparently prespecified ensemble. If final adaptation is contemplated, reserve its untouched cohort before any tuning.

Terra should own the numerical interpretation and provenance implementation. Luna should own an independent fixture-based audit and regression tests. Neither should edit the manuscript, Overleaf, version 1.0 release, model weights, frozen protocol, or archived outputs as part of these tasks. An implementation pass is not permission to make a scientific claim. The detailed handoff defines the boundaries and acceptance tests.

## Reporting and decision

The study does not need a favorable external result to have integrity. It needs accurate input/reference lineage, a valid measurement procedure, preserved negative results, and claims restricted to the evidence. Its existing out-of-fold coverage, input-dependence controls, reproducibility, and fixed-budget recovery result remain relevant. None alone establishes broad generalization.

Further work should answer a finite set of questions rather than continue until one data source produces a favorable mean. Report all predeclared external tests and all benchmark settings. Later discoveries that constrain the intended scope should be disclosed in the manuscript or an explicitly linked follow-up, rather than kept out solely because the original numbers are frozen.

The immediate decision is to audit and correct the angle interpretation, not to restore an earlier acceleration claim. In parallel, prepare the dense new-identity and ATLAS eligibility matrices. That plan addresses a concrete benchmark defect and the most informative remaining generalization question without requiring a new architecture or a return to prior-version comparisons.

## Local evidence

Paths below are relative to this report. They identify the local evidence inspected for this assessment; some large artifacts are not tracked in the source repository.

- [Original fixed-period rows](../../inputs/frozen-artifacts/k3-definitive-7874092/downstream/fixed-period/fixed-period-rows.json) and [publication summary](../../inputs/frozen-artifacts/k3-definitive-7874092/release/publication-summary.json).
- [Generalization execution record](generalization-study.md), including corrected ZTF results and the exposure/source inventory.
- [ALCDEF–Gaia analysis](../../data/generalization-20260910/locked-cohort/alcdef-gaia-transfer-analysis-30-20260911.json) and [its protocol](../repro/k3_alcdef_gaia_transfer_protocol.yaml).
- [Generalization study specification](../repro/k3_generalization_study_spec.yaml).
- [Convergence study specification](../repro/k3_followup_study_spec.yaml), [capacity-corrected selection](../../data/convergence-capacity-corrected/development/selection.json), and [capacity report and execution history](solver-capacity-followup.md).
- [Convergence output classifier](../lc_pipeline/k3/convergence_benchmark.py), [orchestration and scoring](../lc_pipeline/k3/convergence_study.py), [native angle output](../../data/solver-capacity-revision/expanded-solver/convexinv/convexinv.c), and [native rotation matrix](../../data/solver-capacity-revision/expanded-solver/convexinv/blmatrix.c).
- [Example rejected start](../../data/convergence-capacity-corrected/development/tolerance-0p0001/runs/asteroid_1729/repeat-0/candidate/start-2/result.json). The diagnostic traversed all five `blind-execution.json` files, their repeat markers, and the referenced individual start records; it did not change them or execute a locked object.

This assessment reflects the local records and sources inspected on 11 September 2026. The coordinate diagnostic is new to this report. The study implementation has not yet been corrected or re-certified. Source access is not equivalent to a completed usable-data download, and external-candidate counts are not promised final sample sizes.

## Sources

[^1]: Warner, B. D., editor. [Asteroid Lightcurve Data Exchange Format Database Bundle V1.0](https://sbn.psi.edu/pds/resource/alcdef.html). NASA PDS, 2021. DOI: 10.26033/b8cw-s522. Archived photometry and metadata.
[^2]: ALCDEF. [Database notices](https://www.alcdef.org/) and [ALCDEF format standard](https://www.alcdef.org/docs/ALCDEF_Standard.pdf). Duplicate-submission notice dated 2025; format conventions and metadata requirements.
[^3]: ATLAS Project. [Project and data-access page](https://atlas.fallingstar.com/). SSCAT v3 announcement and PDS access notice.
[^4]: NASA PDS Small Bodies Node. [ATLAS archive](https://pdssbn.astro.umd.edu/data_sb/missions/atlas/index.shtml). Page updated 26 February 2026; reduced images, detection catalogs, CATCH, and archive documentation.
[^5]: Pál, A., et al. [Solar System objects observed with TESS—First data release](https://arxiv.org/abs/2001.05822). ApJS 247, 26, 2020. DOI: 10.3847/1538-4365/ab64f0.
[^6]: TSSYS. [DR1 archive README](https://archive.konkoly.hu/pub/tssys/dr1/README). File layout, photometry columns, flags, and distances.
[^7]: Tanga, P., et al. [Gaia Data Release 3: The Solar System survey](https://doi.org/10.1051/0004-6361/202243796). A&A, 2023. SSO photometry fields and transit-level measurements.
[^8]: NASA PDS Small Bodies Node. [Asteroid radar data](https://sbn.psi.edu/pds/archive/radar.html). Radar data and shape-model collections.
[^9]: Vernazza, P., et al. [VLT/SPHERE imaging survey of the largest main-belt asteroids: Final results and synthesis](https://www.aanda.org/articles/aa/pdf/2021/10/aa41781-21.pdf). A&A 654, A56, 2021. DOI: 10.1051/0004-6361/202141781.
[^10]: Ďurech, J., et al. [Asteroid models reconstructed from ATLAS photometry](https://arxiv.org/abs/2010.01820). A&A 643, A59, 2020. DOI: 10.1051/0004-6361/202037729.
[^11]: Ďurech, J., and Hanuš, J. [Reconstruction of asteroid spin states from Gaia DR3 photometry](https://arxiv.org/html/2305.10798v1). A&A 675, A24, 2023. Sections 2–2.6: preparation, stability checks, reference comparisons, and model uncertainty.
[^12]: NASA PDS Small Bodies Node. [Asteroid datasets](https://sbn.psi.edu/pds/archive/asteroids.html). Optical lightcurves, moving-object photometry, and other collections.
[^13]: DAMIT. [Documentation](https://damit.cuni.cz/pages/documentation). Identity fields, lightcurve conventions, model provenance, and quality flags. The page was retrieved earlier in this assessment; subsequent attempts intermittently failed.
[^14]: NASA/JPL Solar System Dynamics. [SBDB API documentation](https://ssd-api.jpl.nasa.gov/doc/sbdb.html). Physical-parameter values, references, notes, and identity fields.
[^15]: Roberts, D. R., et al. [Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure](https://www.wsl.ch/lud/biodiversity_events/papers/Roberts_et_al-2017-Ecography.pdf). Ecography 40, 913–929, 2017. DOI: 10.1111/ecog.02881. Asteroid-specific application here is a recommendation, not a result of that paper.
[^16]: Cawley, G. C., and Talbot, N. L. C. [On over-fitting in model selection and subsequent selection bias in performance evaluation](https://jmlr.org/papers/v11/cawley10a.html). JMLR 11, 2079–2107, 2010.
[^17]: Ojala, M., and Garriga, G. C. [Permutation tests for studying classifier performance](https://jmlr.org/papers/v11/ojala10a.html). JMLR 11, 1833–1863, 2010. A pole-regression permutation procedure would need its own justified null and grouping.
[^18]: Dolan, E. D., and Moré, J. J. [Benchmarking optimization software with performance profiles](https://arxiv.org/pdf/cs/0102001). Mathematical Programming 91, 201–213, 2002. DOI: 10.1007/s101070100263.
[^19]: Moré, J. J., and Wild, S. M. [Benchmarking derivative-free optimization algorithms](https://www.mcs.anl.gov/~wild/papers/2009/JJMSMW07.html). SIAM Journal on Optimization 20, 172–191, 2009. DOI: 10.1137/080724083.
[^20]: Hansen, N., et al. [COCO: Performance Assessment](https://arxiv.org/pdf/1605.03560). 2016. Fixed-target performance, unsuccessful runs, target construction, and runtime aggregation. The proposed DeLPHI wall-time and scientific-recovery extensions are study-design recommendations.
