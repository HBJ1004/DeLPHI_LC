# Reliability study execution receipt — 2026-09-15

This records implementation and launch, not scientific results. The study is
retrospective, same-identity, out-of-fold evaluation. It does not replace the
reported external-data tests or establish new-identity generalization.

## Frozen run

- Directory: `../data/reliability-20260915/run-1` relative to the source checkout.
- Design: `repro/k3_reliability_protocol.yaml`; explanation and old-manuscript
  coverage checklist: `docs/reliability-study.md`.
- Sealed `study.json` file SHA-256:
  `08d27059b575af2ad5896d4718d4ba78e917def2affabbf10d8214870707d070`.
- Cohort: 170 original objects; 157 eligible for the latest-block holdout;
  80 eligible for the common observing-block grid; 51 prediction conditions.
- All 170 undegraded inputs have bit-identical model-input tensors under the
  native adapter and original adapter. Split and declared synthetic-donor
  checks passed. The donor check covers the published donor manifests, not
  an independent inspection of every synthetic training array.
- Five training-only atlases were rebuilt from the respective 108 training
  labels. No evaluation labels were used to fit them.
- The execution snapshot contains 128 Python files and matches the tested
  source. Model, input, source, solver, and environment bindings are sealed.
- Runtime: Python 3.12.3, NumPy 2.1.3, PyTorch 2.5.1+cu124, SciPy 1.14.1,
  Astropy 7.0.0, safetensors 0.4.5; CUDA inference, six runtime threads.
- The isolated solver's export-only precision patch preserved the tested
  native stdout, stderr, and modeled lightcurve. Direct-forward parity is
  also checked for each selected fit before its completion record is sealed.

## Verification before launch

`python-pinned -m pytest -o addopts='' -q tests repro/tests`: **465 passed**
in 90.39 seconds. These are the follow-up working tree's tests, not a new
published-release test count. Ruff and `git diff --check` also passed.

Tests include deterministic nested sampling, latest-block-only selection,
reference-blind prediction and fit selection, native-forward conventions,
complete condition/repeat denominators, failure accounting, scoring/export
integration, detached launch, and real process interruption/resumption without
resetting the persisted budget or duplicating completed cells.

## Launch and inspection

The detached worker was launched at 2026-09-15 16:42:08 KST with PID 45488.
The PID is only a launch receipt; query status to determine whether it is
still running. The worker runs from the sealed `runtime-source` directory,
not from the editable working checkout. Its output is `run.log` and its
latest operation is recorded in `progress.json` (or `pilot/progress.json`).

## Operational interruption

At 2026-09-15 18:07 KST, the worker was deliberately stopped at the user's
request to release the GPU. It had sealed 3,897 prediction cells and was in
the full inference phase. It was not stopped because of a prediction error.
The process released its CUDA context; `nvidia-smi` showed no DeLPHI compute
process afterwards. The runtime budget continues to count elapsed wall time,
so this interruption reduces the remaining time available under the registered
24-hour limit. Completed sealed cells remain available to a later verified
resume.

From the source checkout:

```sh
python \
  -m repro.run_k3_reliability_study status \
  --root ../data/reliability-20260915/run-1
```

The pilot measures runtime without opening scientific scores. If feasible,
the worker freezes a one- or three-repeat schedule and a fixed fitting cohort,
then continues through inference, fitting, scoring, and report export. It
stops with a recorded reason if checks fail or the budget is exhausted.

The total wall-clock deadline is **2026-09-16 16:28:31 KST**. This is the
24-hour limit, not an estimated completion time. Preparation and interruption
downtime count toward it. The pilot determines the usable runtime estimate.

If the worker has actually stopped, the same `start` command resumes verified
completed cells without resetting the deadline. Do not delete checkpoints,
edit the frozen snapshot, or change the scientific schedule to obtain a
preferred result. A technical correction requires an explicit recorded
revision; a changed experiment requires a new disclosed design.

No manuscript, Overleaf project, model weights, published evidence, or software
version was changed. Software remains version 1.0. Results and any later
manuscript additions need separate review after the full registered run.
