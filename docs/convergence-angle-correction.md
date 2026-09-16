# Convergence solver-angle interpretation correction

This note records a post-development interpretation defect in the convergence
follow-up. It does not alter the frozen study specification, its thresholds,
solver source, raw execution records, manuscript, model weights, or releases.

## Defect

The solver prints longitude and `90 - colatitude` after optimizing
unconstrained trigonometric parameters. A printed latitude outside the usual
`[-90, 90]` coordinate interval can therefore still represent a finite,
directed unit vector. The former convergence classifier treated that range
condition as a numerical-output failure.

`lc_pipeline.k3.convergence_axis.decode_convergence_axis` is the scoped
decoder for trusted solver output. It retains the raw angles and derives the
standard coordinates from the directed vector

`(cos(beta) cos(lambda), cos(beta) sin(lambda), sin(beta))`.

It uses interpretation version `delphi.k3-convergence-axis.v1`. It rejects
non-finite values, strings, and booleans. It does not clip latitude or relax
any other completion condition.

## Scope

The convergence classifier, fit selection, and reference scoring use the same
decoder. Selected fits retain standard coordinates and record the
interpretation version; raw solver output remains in the run record. The
decoder is not applied to public input-coordinate validation or to historical
inversion APIs.

This patch alone does not reinterpret archived development outputs. A separate
hash-verified replay under Packet B of
`docs/terra-luna-validation-handoff.md` is required before any result or
benchmark decision changes. That replay must write a new output directory and
retain the original statuses and files unchanged.

The replay command is `python -m repro.reinterpret_k3_convergence_development`.
Its `reinterpret` subcommand accepts only the parent execution root, the frozen
study specification, the revised study lock, the development manifest, and a
new output directory. It does not accept a score artifact or a reference
catalog. The separate `score` subcommand requires the sealed receipt hash and
is the first step permitted to open a reference catalog. Run `--help` before a
future authorized replay; this implementation step did not execute it on the
archived graph.

From the source checkout, the locked inputs for a future authorized run are:

```text
parent root:  ../data/convergence-capacity-corrected/development
study spec:   repro/k3_followup_study_spec.yaml
study lock:   ../data/solver-capacity-revision/revised-convergence-study-lock.json
development:  repro/data/k3-followup-20260910/development-objects.json
references:   repro/data/damit-20250610T000301Z/catalog.jsonl
```

Their required specification and lock digests are respectively
`ff97151e38584f5b2f5448e9b1f3ab7b6e9a6274eb51cf2a5007280937e971a3` and
`6388670f1b3c49862176ebcab5669c86d662e8854536b6328069213315fd9e8a`.
The command verifies these paths and hashes before it reads the execution
graph. It must use a new output directory outside the parent root, for example:

```bash
python repro/reinterpret_k3_convergence_development.py reinterpret \
  --parent-root ../data/convergence-capacity-corrected/development \
  --output ../data/convergence-angle-corrected/<new-run-id> \
  --study-spec repro/k3_followup_study_spec.yaml \
  --revised-study-lock ../data/solver-capacity-revision/revised-convergence-study-lock.json \
  --development-manifest repro/data/k3-followup-20260910/development-objects.json
```

After sealing, score only by passing the printed receipt digest to the separate
`score` command and the reference catalog above. The command never opens a
historical `score.json`. No such creation or scoring command has been run
during this implementation step.

## Correction-bound locked authorization

The interpreted development receipt and its separate score are now bound to a
locked-cohort authorization, not to the superseded five-score selection
format. The concrete correction inputs are:

```text
receipt:       ../data/convergence-angle-corrected/20260911-axis-v1/reinterpretation-receipt.json
score:         ../data/convergence-angle-corrected/20260911-axis-v1/reinterpreted-development-score.json
authorization: ../data/convergence-angle-corrected/20260911-axis-v1/locked-authorization.json
lock:          ../data/solver-capacity-revision/revised-convergence-study-lock.json
manifest:      repro/data/k3-followup-20260910/locked-evaluation-objects.json
```

`repro/authorize_k3_correction_locked_execution.py verify --help` exposes the
read-only authorization check. The new
`repro/run_k3_convergence_study.py execute-correction-locked` command accepts
that authorization instead of the old development score files. Before it can
begin solver work, it rechecks the receipt and score digests, the selected
`0.0003` tolerance, the revised lock, the unchanged 140-object manifest, six
starts per arm, three repeats, fixed 1,000-iteration/300-second conditions,
and warm/cold neural-timing definitions. It records a separate exclusive claim
immediately before execution. This implementation did not invoke that command
or create any locked execution result.

## Focused verification

Run from the source directory with the project-supported interpreter:

```bash
python -m pytest -q tests/test_k3_convergence_axis.py \
  tests/test_k3_convergence_study.py tests/test_k3_convergence_benchmark.py
python -m ruff check lc_pipeline/k3/convergence_axis.py \
  lc_pipeline/k3/convergence_benchmark.py \
  lc_pipeline/k3/convergence_study.py tests/test_k3_convergence_axis.py
git diff --check
```

The unit suite includes ordinary and wrapped coordinates, two out-of-range
latitude examples, poles, random angle checks against the C `blmatrix` third
row, invalid angles, retained completion failures, non-mutation, and
label-blind minimum-RMS selection.
