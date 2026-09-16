# Terra and Luna implementation handoff

## Purpose and limits

This is an implementation-ready plan, not a record of completed changes. Its scientific rationale and source review are in [Validation options and acceleration benchmark plan](validation-options-and-benchmark-plan.md). Work is confined to the follow-up workspace. The immediate objective is to correct a reproducible solver-output interpretation defect without changing the scientific thresholds, then determine whether the existing convergence holdout is authorized.

No procedure can guarantee zero mistakes. This handoff requires an independent test implementation, real-artifact checks, provenance checks, and explicit stop conditions. A passing toy fixture alone is insufficient.

Terra owns implementation and its unit tests. Luna independently checks the mathematics, writes adversarial/integration tests, and verifies the real-artifact report. To reduce cost, delegate bounded packets with the paths, contracts, and expected results below; do not fork the full conversation. Never have both agents edit the same file concurrently. If only one worker is used, separate implementation and verification into distinct passes with the independent verifier reading the contract before the patch.

Do not change version 1.0, weights, public releases, Git tags, the manuscript, Overleaf, author metadata, the frozen YAML/checksums, or archived study outputs. Do not launch training, download entire surveys, or run the 140-object stage as an incidental consequence of a test command. These are separately gated tasks after the initial implementation is accepted.

## Workspace and preflight

Source directory: the repository root.

Data and archived inputs are siblings of `source`: `../data` and `../inputs`. Large artifacts must remain outside the source repository. The working tree already contains follow-up changes. Record `git status --short` and the current commit, preserve those changes, and inspect any applicable `AGENTS.md` before editing. Do not reset, clean, or broadly delete directories.

Use the existing supported Python environment after recording Python, NumPy, Torch, and SciPy versions. If no environment is available, create a separate task environment; do not upgrade the environment that produced the frozen evidence. Identify the runnable interpreter explicitly in the completion report. Downloaded source data must keep their source receipts; do not print credentials or machine-specific secrets.

Read these files before implementation:

- `lc_pipeline/k3/convergence_benchmark.py`: `_completion`, `_selectable`, `_run_convergence_record`.
- `lc_pipeline/k3/convergence_study.py`: archived-result validation, `_selection_from_records`, `_arm_repeat_summary`, `_paired_rms_ratio`, `score_blind_execution`, and `_score_decision`.
- `lc_pipeline/v2/convexinv.py`: result parsing and output validation. Prefer a scoped follow-up adapter; do not change historical result semantics globally without a demonstrated need.
- `repro/k3_followup_study_spec.yaml` and its checksum; `docs/solver-capacity-followup.md`.
- Existing `tests/test_k3_convergence_study.py`, `tests/test_k3_convergence_timing.py`, and other convergence tests discovered with `rg --files tests`.
- Native source at `../data/solver-capacity-revision/expanded-solver/convexinv/{convexinv.c,blmatrix.c,mrqmin.c}`. Do not modify this solver for the angle correction.

Reserve the following file ownership before delegating. New names are proposed paths, not files that already implement these features; check for collisions first.

| Owner | File scope |
| --- | --- |
| Terra | New pure decoder `lc_pipeline/k3/convergence_axis.py`; required integration in `lc_pipeline/k3/convergence_benchmark.py` and `lc_pipeline/k3/convergence_study.py`; proposed replay entry point `repro/reinterpret_k3_convergence_development.py`. |
| Terra | Decoder unit tests in `tests/test_k3_convergence_axis.py`; correction explanation and runnable commands in a new `docs/convergence-angle-correction.md`. |
| Luna | Independent mathematical tests in `tests/test_k3_convergence_axis_independent.py`; provenance, tamper, and phase-separation tests in `tests/test_k3_convergence_reinterpretation.py`; a separate verification report. |
| Coordinator | Integration of changes to existing shared tests, final test run, lock approval, and progression to a subsequent packet. |

Luna may prepare independent fixtures while Terra implements, but integration assertions should wait for the decoder and replay interfaces to be agreed. Interface changes require an explicit handoff rather than simultaneous edits. Packet D gets a separate file allocation before it starts.

## Packet A — solver-output interpretation

Owner: Terra. Independent review: Luna. No new solver grid or training.

### Defect and invariant

The C solver uses unconstrained trigonometric angle parameters. It prints longitude and `90 - colatitude`; its reported latitude can legitimately lie outside [−90, 90]. The follow-up `_completion` currently calls that a numerical-output failure.

Represent the printed direction as

`p = (cos(b) * cos(l), cos(b) * sin(l), sin(b))`,

where the trigonometric arguments are radians. Obtain standard coordinates with

`l_standard = degrees(atan2(p_y, p_x)) mod 360`,

`b_standard = degrees(atan2(p_z, hypot(p_x, p_y)))`.

The invariant is preservation of the **directed** unit vector, not merely its antipodal equivalence. Latitude clipping is prohibited. At an exact pole, longitude is not identifiable; tests must compare vectors there. Preserve raw output coordinates and do not rewrite fitted shape/phase parameters or regenerate model flux from partially transformed parameters.

Keep the correction specific to trusted, finite solver-output interpretation. It must not relax the input schema for ordinary user-supplied coordinates. Preserve rejection of timeout, nonzero return, adapter errors, missing iteration logs, iteration cap, nonfinite values, nonpositive RMS, invalid period, invalid output products, and missing/invalid required hashes.

### Implementation contract

Use a small pure decoder returning the raw angles, standard angles, directed vector, and interpretation version. Avoid in-place mutation of the caller's record. Validate types and finiteness before conversion; explicitly reject booleans masquerading as numeric fields. Use a stable full-period reduction if handling large finite angles. Maintain existing result semantics for already standard valid outputs to numerical precision.

Both classification and selection/scoring must use the same validated interpretation. It is insufficient to remove the range check while leaving `_selectable` or archived revalidation bound to the old failure state. Inspect all callers of `_completion` and `_selectable` and all stored completion checks before patching. A correction must produce consistent statuses, selected fits, and axis interpretation from execution through scoring.

Write a small correction record that identifies the reason, decoder version/hash, changed source files, unchanged scientific thresholds, and parent artifacts. A dated explanation is required: this is a defect found after development scoring, not a new prespecified result.

### Required unit tests

| Test | Required outcome |
| --- | --- |
| Ordinary valid coordinates, including negative and wrapped longitude | Directed vector unchanged; returned longitude in [0, 360), latitude in [−90, 90]. |
| `(10, 100)` degrees | Equivalent to `(190, 80)`, not `(10, 90)` and not the antipode. |
| `(10, -100)` degrees | Equivalent to `(190, -80)`. |
| `(725, 0)` degrees | Equivalent to `(5, 0)`. |
| North/south poles and full rotations | Finite unit vector preserved; no fragile longitude assertion at poles. |
| Real example `(92.878146, 101.510344)` | Equivalent to `(272.878146, 78.489656)` degrees. |
| Fixed-seed random angles over several full turns | Native rotation-matrix third row and independently calculated decoded vector agree to a documented float64 tolerance, e.g. 1e−12 componentwise. |
| NaN, infinity, malformed strings, booleans, missing angles | Rejected with a specific validation reason, never converted to a successful result. |
| Valid angles plus timeout, nonzero exit, cap, or invalid RMS/period/output | Still rejected. Test each rejection independently. |
| Original record object reused after decode | Byte/content representation of the input object unchanged. |
| Swapping labels or references | No effect on decoding or label-blind minimum-RMS selection. |

Luna must implement vector checks independently of Terra's helper. Importing the helper twice and comparing its outputs is not an independent test. Read the C matrix convention rather than assuming latitude and colatitude are interchangeable.

## Packet B — archived development replay and provenance

Owner: Terra for replay; Luna for independent verification. Depends on Packet A acceptance.

The five parent directories are under `../data/convergence-capacity-corrected/development/`. Each contains `blind-execution.json`, `score.json`, and referenced repeat/start files. The parent selection has status `no_eligible_tolerance_stop` and SHA-256 `68bddcea938fdfef9a61347de7eac6611531e86a3122151ee3984d1e7e0b7d84`.

The parent study specification SHA-256 is `ff97151e38584f5b2f5448e9b1f3ab7b6e9a6274eb51cf2a5007280937e971a3`; the capacity-corrected lock SHA-256 is `6388670f1b3c49862176ebcab5669c86d662e8854536b6328069213315fd9e8a`. Resolve these paths and verify them, rather than using a pasted hash as proof that the actual inputs match.

### Replay contract

Add an explicitly named, create-once development reinterpretation command. The exact new module name is an implementation choice; document it and expose `--help`. Its input is the hash-verified old execution graph, not the old summary metrics. Its output must be a separate directory, for example `../data/convergence-angle-corrected/<unique-run-id>/`, with parent links and a new interpretation receipt. An existing output directory is a hard error.

Verify cohort identity, all 30 expected objects, three repeats, six starts per arm, raw record hashes, repeat-marker hashes, log/product hashes where the original contract requires them, binary/source identities, fixed periods, and saved timing provenance. Reject missing, duplicated, edited, mismatched, or out-of-root records. A pathname inside a marker is untrusted until resolved and checked against the intended artifact root.

Retain original raw statuses and new interpreted statuses separately. No entry in the original directory, including `result.json`, may be changed. All 5,400 start records must appear in the audit. The replay must not accept an arbitrary failed record solely because its latitude can be converted: all other original validity requirements still apply.

Complete direction interpretation and minimum-RMS selection without opening the reference catalog or old score files. Seal the resulting selection artifact and its hash. Only a separate scoring phase may read reference axes, verify that sealed hash, and calculate recovery. The human analyst has already seen development outcomes; the claim here is programmatic phase separation, not human blinding.

Keep the existing thresholds and repeat aggregation. Do not replace “all six starts completed” with “at least one fit available,” remove the all-object RMS support rule, relax the five-percentage-point margin, or pick a tolerance by an edited criterion. Such changes would be a different study, not this defect correction.

If reliable replay would require bypassing the frozen artifact checks, stop and recommend a fresh development execution in a new output directory. Do not weaken those checks to save runtime. The existing five-setting solver time is approximately 1.53 serial hours, so a full rerun is a bounded fallback rather than an excuse for unsafe replay.

### Real-artifact acceptance checks

1. Exactly 450 paired repeat cases and 5,400 individual starts; all raw inputs unchanged.
2. The old numerical-output-failure counts, in tolerance order 0.01, 0.003, 0.001, 0.0003, 0.0001, are 3, 9, 24, 30, 39. All 105 fail only the old latitude-range condition. Any additional reason must stop automatic reinterpretation for that record and be reported.
3. New interpretations preserve the raw directed axes within 1e−12 componentwise. All numerical validity checks remain active.
4. New selected fits are chosen by minimum final relative RMS, with the original deterministic start-index tie break. Restoring a previously rejected fit may change the chosen solution; do not keep the old selected pole while claiming to have corrected selection.
5. The diagnostic expectations below are independently reproduced. A discrepancy is a reason to investigate, not to change the code until it matches the table.

| Tolerance | Expected recovered objects: baseline / guided | Expected geometric RMS ratio | Expected development eligibility |
| --- | ---: | ---: | --- |
| 0.01 | 9 / 16 | 0.8801876033 | Eligible |
| 0.003 | 11 / 13 | 0.8904674014 | Eligible |
| 0.001 | 16 / 21 | 0.9438915829 | Eligible |
| 0.0003 | 18 / 17 | 0.9691672001 | Eligible |
| 0.0001 | 24 / 22 | 0.9849880678 | Ineligible: recovery difference −2/30 < −0.05 |

These values came from a read-only diagnostic and are not certified replacement results. Expected support is 30/30 at every setting. Recorded timing ratios remain unchanged when interpreting the same runs. The existing rule would choose 0.0003, not 0.0001. Its warm timing ratio is approximately 0.9841, and its empirical lower acceptance bound approximately 0.9358. Do not label that a demonstrated speedup.

### Adversarial integration tests

Luna should independently introduce one corruption at a time in disposable fixtures: modified raw bytes, stale hashes, changed object identity, missing repeat, duplicate start, unexpected tolerance, wrong period, altered binary identity, path traversal, and an existing destination. Each must fail closed without altering the input archive or leaving an apparently complete output.

Test that the selection phase raises if it tries to open a reference catalog or any old score file. Test that post-seal edits fail scoring. Test that resume behavior verifies previously written records rather than trusting the presence of a filename. Tests must use temporary directories and mocked small fixtures; normal test collection must never run the real solver grid, training, or network acquisition.

## Packet C — formal benchmark decision

Owner: primary coordinator after Terra and Luna reports. Depends on both earlier packets passing.

Prepare a new correction-bound execution lock and an explicit decision report. Keep the same capacity-corrected binary, source, fixed periods, original object split, six starts, three repeats, neural timing definitions, and scoring thresholds. Confirm that the 140 locked objects have not acquired new convergence results during testing. Their original DAMIT reference and publication outcomes are already known; call this an internally held-out benchmark, not an external or globally unseen cohort.

If the unchanged development rule selects a setting, the later authorized implementation phase can run that one setting on all 140 objects. Before doing so, publish the selected setting within the local lock and prevent changes after execution begins. Account for any new decoder overhead consistently. Do not reuse a cache and describe it as a fresh cold-start timing measurement.

Final acceptance remains conjunctive:

- Warm runtime ratio empirical lower acceptance bound strictly greater than 1.0.
- Recovery simultaneous exact lower bound at least −0.05.
- Completion simultaneous exact lower bound at least −0.05.
- Geometric RMS ratio empirical upper acceptance bound strictly less than 1.01, with the existing full-support condition.

Do not rename empirical acceptance bounds as confidence intervals. Do not turn a nonestimable endpoint into a passing endpoint. Publish per-object failures and all four outcomes. A failed or inconclusive holdout ends this protocol; a new scheduler or threshold needs a new, disclosed design and cannot recycle those outcomes as an untouched confirmation set.

A later time-to-quality study is optional and separate. Its implementation requires a common objective target, label-free stopping, actual elapsed time including neural/fallback work, explicit target failures, and a fresh analysis lock. Do not add it silently to the present correction.

## Packet D — external validation eligibility, not outcome shopping

Owner: Luna for metadata acquisition/census with fixtures; Terra for integration review. Can follow Packet A in parallel only if the primary coordinator explicitly delegates it and the file sets do not overlap.

Start from `repro/k3_generalization_study_spec.yaml`, `docs/generalization-study.md`, the ALCDEF census, the model-specific exposure ledger, and the separate 30-object transfer protocol. Preserve the original negative ALCDEF and corrected ZTF results. Do not rerun the old invalid ZTF physical-identity join.

Build one metadata-only eligibility table for dense new identities, ATLAS, and TESS. At minimum retain:

`physical_identity`, `source_database_id`, `alias_receipt`, `input_source`, `raw_source_version`, `input_receipt_hash`, `valid_points`, `native_sessions`, `documented_apparitions`, `apparition_evidence`, `observer`, `time_scale`, `photometric_convention`, `filter`, `period_source`, `period_precision`, `period_quality`, `reference_source`, `reference_quality_tier`, `reference_input_overlap`, `direct_training_exposure`, `synthetic_donor_exposure`, `selection_exposure`, `history_unknown`, `eligible_role`, and `exclusion_reasons`.

Use explicit unknown values rather than guessing. Period quality and apparition requirements must be fixed before outcome access; two calendar years do not certify two apparitions. Do not select by reference pole value, model score, oracle error, or a favorable preliminary result. Model-specific training clearance and whole-project-history clearance are distinct fields.

For ATLAS, use moving-object catalog/PDS access, not an assumed static-sky forced-photometry workflow. Record that the SSCAT README timed out during the research check; first prove a small data-access and format fixture works. For TESS, inspect the real archive column contract and use spacecraft geometry. Do not treat 9,912 metadata rows as 9,912 eligible spin-validation objects. Gaia photometry ingestion is a later packet and must collapse duplicate per-CCD representations of the same transit-level flux according to the source contract.

Deliver a census and a proposed locked cohort size with its precision calculation. Do not promise 100 eligible identities from the current screens. No final prediction or reference scoring belongs in this packet.

Required tests include the DAMIT-ID-101 versus MPC-2 versus MPC-101 distinction, aliases joining to exactly one physical object, duplicate observations across archives, uncertain identity remaining excluded, filter/time-unit parsing, missing period or uncertainty, unknown donor exposure, and deterministic metadata-only selection. Change a reference pole value in a fixture and prove that eligibility does not change. Reference quality metadata may influence a predeclared tier; its numerical pole direction may not.

## Packet E — diagnostics and optional adaptation

This packet is a later plan, not authority to start training. Its prerequisite is a fixed development list and a separate reserved final cohort. Use already exposed identities for period-precision perturbation, input-convention checks, and the classical-inversion information check described in the research report.

Before a diagnostic run, record which causal question it tests, the finite variants, expected input invariants, all failure reasons, and the aggregation rule. Keep held-out-fold bundles for known original identities; do not use a model that trained on the identity because it is convenient. For new identities, preserve the declared 25-model ensemble policy and train-only atlas construction.

The existing eight-factorial ZTF preprocessing results must be loaded as evidence, not repeated as an unbounded search. If adaptation is justified, preserve the prior 2/16/30 GPU-hour staged ceiling and complete balanced fold/seed stages. A failed or unfinished stage must not be replaced by a selectively reported partial ensemble. Final cohort predictions for both frozen and adapted models must be sealed before scoring.

## Verification commands and handoff record

After identifying the correct environment, the normal verification commands from the source directory are:

```bash
python -m pytest -q tests repro/tests
python -m ruff check lc_pipeline tests repro
git diff --check
git status --short
```

Run focused decoder/replay tests before the full suite. Add the tests to CI only after confirming they are deterministic, network-free, and small. Existing warning and minimum-dependency behavior must not be changed to conceal a failure. Use an available NumPy 1.26 environment for a focused compatibility check; if unavailable, record that limitation rather than claiming it passed. Do not install or upgrade dependencies in the frozen evidence environment.

Each worker's final handoff must list:

1. Exact files changed, diff summary, interpreter/dependency versions, and pre-existing changes preserved.
2. Commands actually run, exit codes, test counts, and skipped checks with reasons.
3. Parent and new artifact hashes, decoder version/hash, and proof that source artifacts were unchanged.
4. Numerical findings, clearly distinguishing diagnostic, development, and locked-evaluation results.
5. Any deviation from this contract, unresolved question, or blocked test.
6. A statement that no manuscript, release, weight, frozen threshold, or locked result was modified without its separate authorized step.

Terra's completion is provisional until Luna's independent checks agree. A green unit suite does not by itself authorize the scientific claim. The coordinator should release the next packet only after checking these handoff records.
