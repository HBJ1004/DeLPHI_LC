# Reliability and manuscript follow-up, 2026-09-16

The planned local analysis and manuscript integration are complete. Public
software remains version 1.0. No release, tag, remote branch, or Overleaf
project was changed. The original sampling study and publication evidence
were not overwritten.

## Sampling design and observing requirements

The earlier manuscript used groups separated by gaps longer than 30 days,
called apparitions, not observing nights. The current counterpart uses
native-session-preserving observing blocks. It reproduces the 5-by-5 design
with 1/3/5/10/all blocks and 5/10/20/50/all observations per block on the same
80 eligible asteroids, with three sampling repeats. It is not an exact rerun
of the earlier ten-object/model experiment.

All 6,000 eligible grid predictions completed. With no point cap, the mean
oracle@3 error is 34.13, 24.36, 19.88, and 16.45 degrees for one, three, five,
and ten blocks. The matched full-input mean is 15.34 degrees. Ten blocks
with a 50-point cap per block give 20.99 degrees. A block is not necessarily
one night or a physical apparition. These results do not establish a universal
minimum number of nights or observations per night.

The manuscript includes the heatmap, full cell table, actual retained counts,
conditional bootstrap intervals, paired changes, and an input-requirements
table separating software validity from measured performance.

## Internal checks and interpretation

The new diagnostic completed 780 fold/role/object evaluations: 170 held-out,
540 training, and 70 validation. Every held-out axis set matched the archived
prediction exactly (maximum component difference zero). Charged active run
time was 974.69 seconds, about 16.2 minutes; this includes CPU work and is not
a hardware GPU-busy-time measurement. No retraining or new inversion was run.

After averaging repeated training appearances within asteroid, 169 objects
have matched training/OOF results. Their mean errors are 9.47 and 15.82
degrees. The paired training-minus-OOF difference is -6.35 degrees, with a
conditional 95% interval of [-7.92, -4.87]. One object has no training-role
appearance in the retained splits.

Fixed-prediction reference-set reassignment gives mean error 34.78 degrees,
with the central 95% of 10,000 draws spanning 32.25--37.22 degrees, compared
with the observed 15.77 degrees. Reassignment is restricted within held-out
fold and reference-count strata; 167 of 170 objects can move. This is an
association diagnostic, not a label-permutation test with retraining.

The 25 archived training histories, true per-seed OOF evaluations, fold
summaries, and leave-one-fold-out descriptive summaries are included. These
checks support within-cohort object-specific learning but do not prove the
absence of overfitting or generalization to a new survey. Training losses use
real/synthetic mixtures whereas validation losses use real objects only.

The manuscript also reports the existing 75-object withheld-lightcurve check,
the broad-grid timing/fit trade-off, and the unfavorable corrected ZTF and
ALCDEF--Gaia transfer results. The withheld thinned arm retains 10% of points
within each native session (at least two), not ten points per session.

The statistical framing uses Cawley and Talbot (2010) for model-selection
bias, Ojala and Garriga (2010) to distinguish full retraining permutation tests,
and Bates et al. for the estimand and uncertainty of cross-validation. The
manuscript and machine-readable reports state the limits of each diagnostic.

## Deliverables

- `paper/sample701.tex` and its author/anonymous PDFs: 18 figures and 11 tables.
- `paper/k3-followup/`: 45 imported, checksum-verified files including source
  export manifests. The numerical imports are separate from the original
  frozen results.
- `paper/k3-followup-results.tex` and four companion tables: derived from
  imported JSON, with a separate derivation manifest.
- `docs/manuscript-reproduction-inventory.md`: all 12 earlier figures and
  seven earlier tables mapped to current counterparts or explicit exclusions.
  Historical period-estimation, period-fusion, and variable-head experiments
  are not presented as implemented by the fixed-period, fixed-K3 model.
- `DeLPHI-publication/DeLPHI-PSJ-followup-20260916/`: flat anonymous source
  package, machine-readable follow-up summaries, figures, and 26-page PDF.
- Corresponding ZIP SHA-256:
  `4bd0fc85763c4725b9de708b6f77c69bf8b6098ea63dcbdaa7a066790f5f216d`.

Canonical export directories relative to the follow-up project are:

- `data/reliability-publication-20260916-layout`
- `data/reliability-supplement-20260916-layout`
- `data/followup-transfer-publication-20260916-verified`
- `data/internal-validation-20260916/run-1/report-export-publication`

Older report-export directories are retained as intermediate provenance;
the manuscript imports only the four directories listed above.

## Verification and remaining submission work

The final pinned-environment full suite passed: 484 source tests and 36 paper
tool tests. Scoped lint passed. All 43 rendered citations passed the online
identifier audit bound to the final TeX and bibliography. Both PDFs compile
without undefined citations/references or stuck-float warnings. The anonymous
builder scans text and machine-readable exports for identifiers and private
paths. The post-results submission check passes.

An initially failing restart test had a 20-second limit that included a
measured 19.4-second cold dependency import. The test now allows startup
separately and retains a 20-second post-start recovery deadline, the same-budget
assertion, and checks for unchanged/nonduplicated cells. The runner and its
sealed protocol were not changed for this repair.

From the paper directory, verify offline with:

```sh
python3 tools/import_followup.py --verify
python3 tools/derive_followup.py --verify
python3 tools/check_submission.py --post-results
```

Run submission-check modes sequentially: each rebuilds the same PDFs.
The literal submission check still reports three expected items: the new
follow-up evidence is not publicly archived, the author metadata file is
missing, and its placeholder remains in the author PDF. The public frozen
release must not be described as containing these later experiments. A new
archival deposit and author-verified metadata are needed before submission.
