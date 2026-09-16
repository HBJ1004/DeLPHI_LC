# Earlier-manuscript figure and table inventory

The reference is `origin/master` at `9eecb63` in the **paper repository**,
`sample701.tex`. It contains 12 figures and 7 tables. The revised manuscript is
`paper/sample701.tex`. This inventory maps scientific purposes, not old model
values: results from a different architecture are not relabeled as current K3.

The earlier degradation grid selects 1, 3, 5, 10, or all time groups and caps
each at 5, 10, 20, 50, or all points. Those groups were called apparitions and
were based on gaps longer than 30 days; they were not observing nights. The
current sealed experiment retains native session identifiers and merges their
intervals into observing blocks. It reproduces the 5-by-5 design on 80 matched
objects, not the earlier ten-object sample or an exact historical grouping.

| Earlier object | Current counterpart in the manuscript | Disposition |
| --- | --- | --- |
| Fig. multiepoch | fig:multiepoch: verified (2) Pallas native sampling | Recreated with database-to-MPC identity checked |
| Table data_summary | tab:followup_data: 170-object sampling ranges | Recreated from the sealed cohort |
| Fig. pipeline | fig:k3_architecture | Current pipeline and architecture share one diagram |
| Fig. architecture | fig:k3_architecture | Current model, not the earlier direct-regression architecture |
| Table training_config | tab:configuration | Current frozen K3 settings |
| Table cv_results | tab:oof, tab:followup_folds, tab:followup_seeds | Ensemble, fold, and single-seed results kept distinct |
| Fig. cdf | fig:followup_cdf | Current 170-object ensemble oracle@3 CDF |
| Fig. ztf | fig:followup_transfer, tab:followup_transfer | Corrected ZTF diagnostic and ALCDEF--Gaia transfer; limitations retained |
| Fig. sky_distribution | fig:k3_candidate_calibration | All 510 current axes; no pole-population claim |
| Fig. degradation | fig:degradation, tab:degradation_cells | Matched 80-object, 25-cell, three-repeat block/point-cap grid |
| Table requirements | tab:requirements | Software validity separated from tested performance; no universal nights rule |
| Table ablation | tab:followup_ablation | Inference-only removal contrasts, not retrained-component ablations |
| Fig. feature_importance | fig:k3_control_sensitivity | Replaced by measured input-removal sensitivity, not causal feature importance |
| Fig. correlation_matrix | fig:followup_correlations | Tie-aware descriptive Spearman matrix on 170 OOF objects |
| Fig. error_analysis | fig:k3_error_properties; reference strata CSV | Current diagnostics and reference-count strata |
| Table period_ztf | tab:followup_transfer and fig:followup_period | Transfer and supplied-period sensitivity are separate |
| Table k_sweep | fig:followup_cap_area and cap-containment CSV | Variable-head experiment retired; fixed-three-axis radius/containment addresses its search-space question |
| Fig. period_fusion | Supplied-period path in fig:k3_architecture | Historical period-fusion module absent; not presented as implemented |
| Fig. period_accuracy | fig:followup_period | Supplied-period perturbation, not recovered-period accuracy |

Additional evidence includes the 169-object matched training/OOF comparison,
10,000 fixed-prediction reference-set reassignments, all 25 real fine-tuning
histories, leave-one-fold-out descriptive stability, and the 75-object
withheld-lightcurve experiment. They do not prove that overfitting is absent.

## Evidence and reproduction

- Original sealed sampling/withheld run: data/reliability-20260915/run-1.
  Original inputs, conditions, predictions, fits, and results are unchanged.
- Current manuscript imports: paper/k3-followup. The import manifest binds
  copied CSV, JSON, TeX, and PDF files and retains source export manifests.
  paper/tools/derive_followup.py generates manuscript numbers from these imports.
- Internal run: data/internal-validation-20260916/run-1. Fold/role/object
  checkpoints are separate. All 170 held-out axis sets match the original
  predictions exactly before training-role results are accepted.
- The 25 histories are artifact-root/models/real-fold-N-seed-S.jsonl in the
  original publication archive. Single-seed oracle errors come from its 25
  evaluation NPZs, not loss histories. Archive SHA-256:
  434480b07f7fd2ad948acd75fd2ecae351a5a026bd92ffbc7d52b64077eb1684.
- Corrected ZTF score:
  data/generalization-20260910/mapped-ztf-development-diagnostic-score-current-endpoint.json.
  Original preprocessing: 161 usable inputs, mean 30.00 degrees; 170 intended
  inputs with nine failures at 90 degrees, mean 33.17 degrees. All eight
  preprocessing variants remain visible. Incorrect-identity runs are not reused.
- ALCDEF--Gaia:
  data/generalization-20260910/locked-cohort/alcdef-gaia-transfer-analysis-30-20260911.json.
  Candidate mean 37.25 degrees versus fixed-atlas 24.07 degrees on 30 objects.
  Training-unexposed does not mean unexposed to all project development.
- Broad-grid benchmark:
  data/pole-grid-benchmark/20260912-environment-relocked-v3/full-score.json.
  Cold 20-degree ratio is 2.25 (classical/guided time); selected-fit RMS ratio
  is 1.019 (guided/classical). This is a cost/fit trade-off, not evidence of
  acceleration without loss of fit quality. It is not the original fixed-six
  benchmark or the 75-object withheld-lightcurve experiment.

Software remains version 1.0. This local revision does not move tags, change
original public assets, publish follow-up evidence, or update Overleaf. A
separate archival deposit of the new evidence and author-supplied metadata
are still required before submission.
