# Checking the results of the paper

This guide is for readers who want to check the numbers, tables and figures of
Jo, Ishiguro and Lee, "DeLPHI: Pole-Axis Candidates for Asteroid Lightcurve
Inversion" (submitted to PSJ). It explains what to download, how to recompute
the main results from the released predictions without rerunning any analysis,
which file and script lie behind each result, and how to arrange the files to
rerun the analyses.

**Shortest route.** To confirm the main numbers of the paper, you need only the
16 MB `paper-v1` archive, the code, and Check A: follow
[the minimal download](#minimal-download-for-check-a), [Install](#install), and
[Check A](#check-a-main-numbers-from-the-paper-v1-archive). The rest of this
guide is for checking individual results in detail.

A few terms are used throughout.

- **Oracle error**: for one asteroid, the smallest angle between any of the
  three DeLPHI candidate axes and any of its DAMIT reference poles, with the
  two directions along an axis treated as equal (paper Section 5.1).
- **Test asteroid**: each of the 170 asteroids is analyzed only with the
  networks of the one cross-validation run that held it out of training. File
  names call these results `oof` (out-of-fold).
- **K3** is the name of the DeLPHI model in the code and in file names (`k3-`,
  `delphi-k3`); it refers to the three candidate axes.

## 1. What to download

Two releases of the repository [HBJ1004/DeLPHI_LC](https://github.com/HBJ1004/DeLPHI_LC)
are needed, and the tag `paper-v1` marks the code that produced the results.
All other releases are earlier versions of this material and are superseded.

| Release | File | Size | Contents | Needed for |
|---|---|---|---|---|
| [`paper-v1`](https://github.com/HBJ1004/DeLPHI_LC/releases/tag/paper-v1) | `delphi-paper-v1-evidence.tar.gz` | 16.1 MB | Results of every analysis that is not in `v1.0.0`, with a `README.md` that maps each paper section to its files. `part-a-validation/` holds the reduced-data tests, other photometry, internal checks and withheld-lightcurve test; `part-b-analyses/` holds the reference baselines, interpretation, inversion comparison, period search and DAMIT census | Every check |
| | `SHA256SUMS` | 98 B | Checksum of the archive | |
| [`v1.0.0`](https://github.com/HBJ1004/DeLPHI_LC/releases/tag/v1.0.0) | `delphi-k3-artifacts-v1.0.0.tar.gz` | 0.93 GB | Extracts to `artifact-root/`: test predictions and score maps of the 170 asteroids (`evaluations/`), the 25 trained networks as PyTorch checkpoints with their training logs (`models/`), the control tests, and the summary of release `v1.0.0` (`release/`) | Main results, controls, reruns |
| | `delphi-k3-synthetic-data-v1.0.0.tar.gz` | 1.58 GB | Extracts to `synthetic-data/`: the 20,000 / 2,000 / 2,000 simulated training, validation and test asteroids | Retraining, DAMIT census, full verification |
| | `k3-oof-fold-0.tar.gz` ... `k3-oof-fold-4.tar.gz` | 12.2 MB each | The five networks of each cross-validation run, as used by `delphi-k3-predict` | Running DeLPHI, inversion comparison |
| | `delphi-k3-source-v1.0.0.tar.gz` | 0.35 MB | The code at release `v1.0.0` (commit `4d1bf13`) | Not needed with the `paper-v1` code |
| | `delphi-k3-v1-comparator-v1.0.0.npz` | 33 kB | Results of an earlier, different model, used only by the `v1.0.0` summary and its verification; not used in the paper | Check C, full verification |
| | `SHA256SUMS` | | Checksums of the five network archives | |
| | `publication-archive-manifest.json` | | Checksums and sizes of the four archives above and `VERIFY.md` | |
| | `VERIFY.md`, `release-manifest.json`, `parity.json`, `DeLPHI-v1.0-publication-*.json` | | Records of how the release was built and checked | |

Both releases contain a file called `SHA256SUMS`, so download them into separate
folders. The commands below work in a new folder, `delphi-check/`, and clone the
code at tag `paper-v1` as `DeLPHI_LC/`.

```bash
mkdir delphi-check && cd delphi-check
git clone --branch paper-v1 https://github.com/HBJ1004/DeLPHI_LC.git DeLPHI_LC

mkdir paper-v1 v1.0.0
URL=https://github.com/HBJ1004/DeLPHI_LC/releases/download
(cd paper-v1 && for f in delphi-paper-v1-evidence.tar.gz SHA256SUMS; do
   curl -L -O $URL/paper-v1/$f; done)
(cd v1.0.0 && for f in delphi-k3-artifacts-v1.0.0.tar.gz delphi-k3-synthetic-data-v1.0.0.tar.gz \
     delphi-k3-source-v1.0.0.tar.gz delphi-k3-v1-comparator-v1.0.0.npz VERIFY.md \
     publication-archive-manifest.json SHA256SUMS \
     k3-oof-fold-0.tar.gz k3-oof-fold-1.tar.gz k3-oof-fold-2.tar.gz k3-oof-fold-3.tar.gz k3-oof-fold-4.tar.gz; do
   curl -L -O $URL/v1.0.0/$f; done)
```

Leave out the archives you do not need; the checks below say which they use.

#### Minimal download for Check A

```bash
mkdir delphi-check && cd delphi-check
git clone --branch paper-v1 https://github.com/HBJ1004/DeLPHI_LC.git DeLPHI_LC
mkdir paper-v1 && cd paper-v1
curl -L -O https://github.com/HBJ1004/DeLPHI_LC/releases/download/paper-v1/delphi-paper-v1-evidence.tar.gz
curl -L -O https://github.com/HBJ1004/DeLPHI_LC/releases/download/paper-v1/SHA256SUMS
sha256sum -c SHA256SUMS && cd ..
tar xzf paper-v1/delphi-paper-v1-evidence.tar.gz
```

### Verify the downloads

```bash
# paper-v1 archive
(cd paper-v1 && sha256sum -c SHA256SUMS)
# v1.0.0 networks
(cd v1.0.0 && sha256sum -c SHA256SUMS)
# v1.0.0 archives, against the manifest
(cd v1.0.0 && python3 -c "import json; [print(f['sha256'] + '  ' + f['filename']) for f in json.load(open('publication-archive-manifest.json'))['files']]" | sha256sum -c -)
```

Each line should end in `OK`. With only some of the archives downloaded, add
`--ignore-missing` to the third command. On macOS, use `shasum -a 256 -c`
instead of `sha256sum -c`.

Then extract the archives into `delphi-check/` and check the files inside the
`paper-v1` archive (no output means that all files match):

```bash
tar xzf paper-v1/delphi-paper-v1-evidence.tar.gz
(cd delphi-paper-v1-evidence && sha256sum -c --quiet SHA256SUMS)
tar xzf v1.0.0/delphi-k3-artifacts-v1.0.0.tar.gz          # only for Checks B and C; creates artifact-root/
```

To check every one of the 9,797 files of release `v1.0.0` after extraction,
install the package (next section) and run, from `DeLPHI_LC/`,

```bash
python -m repro.verify_publication_package --package-directory ../v1.0.0 \
  --extract-to ../v1.0.0-verified --output ../v1.0.0-verification.json
```

This needs the four archives, the manifest and `VERIFY.md`, extracts them again
into the new folder `../v1.0.0-verified`, and prints a report with
`"file_count": 9797` and `"passed": true`.

### Install

Python 3.11 or 3.12 is needed. From `delphi-check/`:

```bash
python3.12 -m venv .venv && source .venv/bin/activate     # or python3.11
python -m pip install -e './DeLPHI_LC[test,plot]'
delphi-k3 validate-protocol
```

The last command checks that the fixed configuration file of the model,
`repro/k3_redesign_spec.yaml`, is the one used for the paper. It prints

```text
{"schema": "delphi.k3-redesign-spec.v1", "sha256": "84dee816d08a60b7780a6d1d400add16136fa96c0677e84929700cee8bf22a50"}
```

## 2. Quick checks without rerunning anything

Run the checks from `delphi-check/` inside the environment created under
[Install](#install), which provides NumPy and the package. Checks A and B need
nothing else; Checks C and D are described with their prerequisites.

### Check A: main numbers from the `paper-v1` archive

This recomputes Section 5.2 and Table 2 from the per-asteroid oracle errors of
DeLPHI, the six standard starting poles and three random axes (the random value
of each asteroid is already averaged over 10,000 draws), with the bootstrap
seeds used by the scripts.

```bash
python - <<'EOF'
import json
import numpy as np

catalog = [json.loads(line) for line in open("DeLPHI_LC/repro/data/damit-20250610T000301Z/catalog.jsonl")]
sample = [row for row in catalog if row["eligible"]]
print("sample: %d asteroids, %d pole solutions, %d observations" % (
    len(sample), sum(len(row["solutions"]) for row in sample),
    sum(row["lightcurve"]["n_observations"] for row in sample)))

d = np.load("delphi-paper-v1-evidence/part-b-analyses/baselines/reference-baselines-per-object.npz")
delphi, standard, random_axes = d["delphi_errors_deg"], d["standard_errors_deg"], d["random_errors_deg"]

def interval(x, seed):
    rng = np.random.default_rng(seed)
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(10000)]
    return "(%.2f to %.2f)" % tuple(np.percentile(means, [2.5, 97.5]))

print("DeLPHI: mean %.2f %s, median %.2f, within 20 deg %.1f%%, within 30 deg %.1f%%" % (
    delphi.mean(), interval(delphi, 20260930), np.median(delphi),
    100 * (delphi <= 20).mean(), 100 * (delphi <= 30).mean()))
print("six standard poles: mean %.2f, median %.2f" % (standard.mean(), np.median(standard)))
print("three random axes: mean %.2f" % random_axes.mean())
print("standard minus DeLPHI: %.2f %s" % ((standard - delphi).mean(), interval(standard - delphi, 20260922)))
print("random minus DeLPHI: %.2f %s" % ((random_axes - delphi).mean(), interval(random_axes - delphi, 20260923)))
EOF
```

Expected output:

```text
sample: 170 asteroids, 267 pole solutions, 652197 observations
DeLPHI: mean 15.77 (13.89 to 17.83), median 12.09, within 20 deg 74.1%, within 30 deg 89.4%
six standard poles: mean 24.42, median 24.13
three random axes: mean 33.84
standard minus DeLPHI: 8.65 (5.95 to 11.25)
random minus DeLPHI: 18.07 (16.00 to 20.00)
```

The median of the random axes in Table 2 (32.33°) is a median over the
individual draws and is in `part-b-analyses/baselines/reference-baselines.json`.

### Check B: oracle errors recomputed from the candidate axes (release `v1.0.0`)

This recomputes the oracle error of every test asteroid from its three stored
candidate axes and the DAMIT reference poles, and gives Tables 6 and 7 and the
network time of Section 5.3. It needs `artifact-root/` from
`delphi-k3-artifacts-v1.0.0.tar.gz`.

```bash
python - <<'EOF'
import json
import numpy as np

ensemble = np.load("artifact-root/evaluations/real-oof-ensemble.npz")
catalog = "DeLPHI_LC/repro/data/damit-20250610T000301Z/catalog.jsonl"
poles = {row["object_id"]: np.array([s["vector"] for s in row["solutions"]])
         for row in map(json.loads, open(catalog))}

errors = []
for name, axes in zip(ensemble["object_ids"], ensemble["refined_axes"]):
    axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
    ref = poles[str(name)] / np.linalg.norm(poles[str(name)], axis=1, keepdims=True)
    errors.append(np.degrees(np.arccos(min(1.0, np.abs(axes @ ref.T).max()))))
errors = np.array(errors)

print("asteroids %d: mean %.2f, median %.2f, within 20 deg %.1f%%" % (
    len(errors), errors.mean(), np.median(errors), 100 * (errors <= 20).mean()))
print("largest difference from the stored errors: %.0e deg" % np.abs(errors - ensemble["oracle_errors_deg"]).max())
for run in range(5):
    e = errors[ensemble["folds"] == run]
    print("run %d: n %d, mean %.2f, median %.2f" % (run, len(e), e.mean(), np.median(e)))
for seed in (17, 42, 137, 777, 2027):
    e = np.concatenate([np.load("artifact-root/evaluations/real-fold-%d-seed-%d.npz" % (run, seed))["oracle_errors_deg"]
                        for run in range(5)])
    print("seed %d alone: mean %.2f" % (seed, e.mean()))
t = ensemble["inference_wall_seconds"]
print("network time per asteroid: median %.2f s, mean %.2f s" % (np.median(t), t.mean()))
EOF
```

Expected output:

```text
asteroids 170: mean 15.77, median 12.09, within 20 deg 74.1%
largest difference from the stored errors: 2e-06 deg
run 0: n 34, mean 16.09, median 13.39
run 1: n 34, mean 19.17, median 15.79
run 2: n 34, mean 17.48, median 11.62
run 3: n 34, mean 11.43, median 10.47
run 4: n 34, mean 14.69, median 11.92
seed 17 alone: mean 20.93
seed 42 alone: mean 16.16
seed 137 alone: mean 19.81
seed 777 alone: mean 19.52
seed 2027 alone: mean 16.57
network time per asteroid: median 0.51 s, mean 0.61 s
```

These match Tables 6 and 7 of the paper. (The run-4 value 14.695 in
`part-a-validation/k3-followup/supplement/k3_fold_metrics.csv` is already
rounded; the unrounded mean is 14.6946°, which the paper gives as 14.69°.) The
same file, `real-oof-ensemble.npz`, also holds `grid_oracle_errors_deg`, whose mean
(15.32°) is the oracle error before the final small adjustment of the
candidates (Appendix A).

### Check C: rebuild the `v1.0.0` summary

With the package installed, this regenerates the summary file and LaTeX values
of release `v1.0.0` from the stored predictions and requires them to match the
released files byte for byte. It also recomputes every oracle error from the
axes, as in Check B. From `DeLPHI_LC/`:

```bash
python -m repro.reproduce_k3_paper --artifact-root ../artifact-root \
  --comparators ../v1.0.0/delphi-k3-v1-comparator-v1.0.0.npz \
  --catalog repro/data/damit-20250610T000301Z/catalog.jsonl \
  --output ../v1.0.0-summary
```

It prints one line of JSON containing `"frozen_summary_and_macros_match": true`.
The output folder then holds `k3-results.tex`, whose values include the
simulated-test mean (11.59°), the main results, the lightcurve-exchange gap
(17.31°) and the label-shuffle gap (37.99°) of Section 6.1.1, and
`oof-object-results.csv` with one row per test asteroid. The summary of release
`v1.0.0` also contains results that the paper does not use: a comparison with
the earlier model in the comparator file, an inversion benchmark with a fixed
amount of work per asteroid, and the pass or fail decisions that were set for
that release. The inversion comparison of the paper (Section 6.4.2) replaces
that benchmark. The plots it writes are those of release `v1.0.0`, not the
paper's figures.

### Check D: the Part A tables of the `paper-v1` archive

`part-a-validation/` carries its own checks (its `VERIFY.md`). From
`delphi-check/delphi-paper-v1-evidence/part-a-validation/`:

```bash
python manuscript-derived/import_followup.py --verify --output k3-followup
python manuscript-derived/derive_followup.py --verify
```

They print `{"files": 64, "passed": true}` and `{"passed": true, "files": 7}`.
The second command re-derives, from the Part A records, the DeLPHI values of
the reduced-data tests, the internal checks, the lower-quality models, ZTF and
ALCDEF (Tables 8 and 9) and the withheld-lightcurve test (Table 12), and checks
them against `k3-followup-*.tex`. Those files also list a comparison with a
fixed set of directions learned from the training poles (labelled "atlas"),
which the paper replaced by the six standard starting poles; the standard-pole
and random-axes values of Tables 8 and 9 are in
`part-b-analyses/baselines/reference-baselines.json`.

## 3. Where each result comes from

**Asteroids outside the sample.** For the ALCDEF comparison (Appendix D), the
paper scored each asteroid with the average score map of all 25 networks,
because none of them was trained on these asteroids
([repro/predict_k3_alcdef_transfer.py](../repro/predict_k3_alcdef_transfer.py),
policy `all_25_frozen_models_for_new_identity`). That script reads the 25
networks as PyTorch checkpoints from `artifact-root/models/` of release
`v1.0.0`, with the [model binding file](#model-binding-file) of Section 4. The
prediction command `delphi-k3-predict` uses one set of five networks instead,
which is the supported way to run DeLPHI on a new asteroid.

In this table `A/` and `B/` are `part-a-validation/` and `part-b-analyses/` of
the `paper-v1` archive, `v1:` is `artifact-root/` of `v1.0.0`, and scripts are
in the repository at tag `paper-v1`.
"DAMIT" means the June 2025 DAMIT lightcurves and tables (Section 4).

| Paper | Result files | Script | Needs |
|---|---|---|---|
| Section 2, Table 1, Figure 1 (sample, coverage) | `repro/data/damit-20250610T000301Z/catalog.jsonl` and `publication-splits-v2.3.json` (repository); `A/k3-followup/supplement/k3_data_summary.csv`, `k3_observing_structure.csv` | `repro/export_k3_reliability_supplement.py` | DAMIT |
| Section 5.2, Table 2 (DeLPHI), Tables 6 and 7, Section 5.3 | `v1:evaluations/real-oof-ensemble.npz`, `v1:evaluations/real-fold-<run>-seed-<seed>.npz`, `v1:release/publication-summary.json` | training sequence (Section 5); summary rebuilt by `repro/reproduce_k3_paper.py` | Checks B and C |
| Section 5.2 intervals, random-axes band of Figure 6 | `B/interpretation/primary-intervals.json`, `random-axes-cdf.npz` | `repro/primary_intervals.py` | `repro/baselines_out/` (repository); CPU |
| Section 5.2, Table 2 (standard poles, random axes); reference rows of Section 6.1.2, Table 9 and Figure 7 | `B/baselines/reference-baselines.json`, `reference-baselines-per-object.npz` (also in `repro/baselines_out/`) | `repro/compute_reference_baselines.py` | `v1:` ensemble, DAMIT model table, `A/k3-followup/revision/`, `A/evidence/source-records/alcdef-gaia-analysis.json`, Gaia DR3 spin table; CPU |
| Figures 3, 4 and 5 (worked example, oracle-error illustration, candidates on the sky) | `v1:evaluations/real-oof-ensemble.npz`, catalog | figure script of the manuscript (see Section 6) | |
| Section 6.1.1, lightcurve exchange and label shuffle | `v1:evaluations/*-input-swap.npz`, `synthetic-test-seed-17.npz`, `synthetic-label-shuffle-test-seed-17.npz`, `v1:release/publication-summary.json` | `delphi-k3` control commands (Section 5) | Check C |
| Section 6.1.1, candidate spread, random baseline | `B/interpretation/label-shuffle-spread.json` | `repro/label_shuffle_spread.py` | `v1:evaluations/`; CPU |
| Section 6.1.2, Appendix D, Tables 8 and 9, Figure 7 | `A/k3-followup/revision/` (`lowq-*`, `error-channel-*`), `A/k3-followup/transfer/`, `A/evidence/source-records/` (`mapped-ztf-score.json`, `alcdef-gaia-analysis.json`), `A/k3-followup-results.tex` | lower-quality models: `repro/run_k3_lowq_census.py`, `run_k3_lowq_damit.py`, `analyze_k3_lowq_lineage.py`, `audit_k3_lowq_exposure.py`; ZTF: `repro/prepare_k3_followup_inputs.py`, `prepare_k3_mapped_ztf.py`, `delphi-k3 predict-ztf-external` and `score-ztf-external`; ALCDEF: `repro/fetch_k3_alcdef_periods.py` to `score_k3_alcdef_gaia_transfer.py`; uncertainties removed: `repro/run_k3_error_channel_sensitivity.py`; collected by `repro/export_k3_revision_evidence.py` and `export_k3_followup_transfer.py` | DAMIT (all models), ZTF photometry from the Fink broker, JPL Horizons geometry, ALCDEF, Gaia DR3 tables (download addresses in `repro/fetch_k3_generalization_sources.py`); GPU for the predictions. Check D re-derives the tables |
| Section 6.1.3, Figure 8, Appendix B (noise), Figure 14 | `A/k3-followup/reliability/k3_data_degradation.csv`, `A/k3-followup/supplement/k3_supplement_numbers.json`, `A/evidence/source-records/reliability-scored-rows.json`, `reliability-sampling-table.csv`; `B/interpretation/sampling-baselines.json` | `repro/run_k3_reliability_study.py`, `export_k3_reliability_publication.py`, `export_k3_reliability_supplement.py`; `repro/sampling_baselines.py <reliability-scored-rows.json>` | DAMIT, `v1:` networks; `sampling_baselines.py` CPU only |
| Appendix B, training versus test, reference reassignment, Figure 15 | `A/k3-followup/internal/summary.json`, `train-oof-pairs.csv` | `repro/run_k3_internal_validation.py`, `export_k3_internal_validation_report.py` | DAMIT, `v1:` networks |
| Sections 6.2.1 and 6.3.1 (mirror ambiguity, latitude, longitude), Figures 9 and 11 | `B/interpretation/mirror-symmetry.json`; `A/k3-followup/revision/lowq-analysis.json`; `A/k3-followup/supplement/k3_reference_strata.csv`, `k3_descriptive_spearman_matrix.csv` | `repro/mirror_symmetry.py`; the two supplement files from `export_k3_reliability_supplement.py` | `v1:` ensemble, DAMIT; CPU |
| Section 6.2.2, amplitude-aspect relation, Figure 9 | `B/interpretation/amplitude-aspect.json`, `.npz`, `amplitude-aspect-paired.json` | `repro/amplitude_aspect_comparison.py`, then `amplitude_aspect_paired.py` | `v1:` ensemble, DAMIT; CPU |
| Section 6.2.3, Appendix B, Shapley values, Figure 10 | `B/interpretation/shapley/*.npz`, `shapley-summary.json` | `repro/group_shapley.py --output-dir <dir>`, then `group_shapley_summary.py` | `v1:models/*.pt`, DAMIT; GPU by default (`--device`) |
| Appendix B, input removals, Figure 16 | `B/interpretation/input-removal-errors.csv` (the per-asteroid errors of all runs in one table) | `repro/interpretation_ablations.py --output-dir <dir>` | `v1:models/*.pt`, DAMIT; GPU by default |
| Section 6.3.3, Appendix C, period errors, Figure 17 | `A/k3-followup/supplement/k3_supplement_numbers.json` | `repro/run_k3_reliability_study.py`, `export_k3_reliability_supplement.py` | DAMIT, `v1:` networks |
| Section 6.4.1, Figure 12 | `B/interpretation/candidate-count.json` | `repro/candidate_count.py` | `v1:` ensemble; CPU |
| Section 6.4.2, Appendix E, Tables 3, 4, 10, 11, Figure 13 | `B/workflow-benchmark/` | `repro/direct_workflow_benchmark/` (see its [README](../repro/direct_workflow_benchmark/README.md)) | DAMIT, patched `convexinv`, networks, GPU; raw fits on request |
| Appendix E.1, withheld-lightcurve test, Table 12 | `A/evidence/source-records/withheld-rows.json`, `A/k3-followup/supplement/k3_withheld_rms.csv` | `repro/run_k3_reliability_study.py`, `export_k3_reliability_supplement.py` | DAMIT, `v1:` networks, patched `convexinv` |
| Appendix C, experimental period search, Figure 18 | `B/period-search/results/` | `repro/period_search/` (see its [README](../repro/period_search/README.md)) | DAMIT; GPU; patched `convexinv` for the physical pilot |
| Section 6.5, DAMIT census of 2026-09-30 | `B/damit-census/` (`raw/` holds the DAMIT tables and lightcurves exported that day, `q3_current_classification.csv`, `candidate_lc_density.json`, and the input list of the lower-quality comparison, `lowq-input-index.json`) | `repro/damit_census/classify.py`, `lc_density.py` | DAMIT snapshot tables and lightcurves, `synthetic-data/` |
| Appendix A, Table 5 | `repro/k3_redesign_spec.yaml` | `delphi-k3 validate-protocol`; see [configuration.md](configuration.md) | |

## 4. Folder layout for rerunning the analyses

The analysis scripts under `repro/` locate their inputs and outputs from their
own position, so they need this layout. The name `DeLPHI-followup` of the top
folder is required, because several scripts contain it.

```text
<parent>/
  damit-20250610T000301Z/                  DAMIT snapshot (tables/, files/)
  paper/figures/                           output folder of Figure 13 (create it)
  DeLPHI-followup/
    source/                                this repository at tag paper-v1
    inputs/
      frozen-artifacts/k3-definitive-7874092/   v1.0.0 artifact-root/
      frozen-artifacts/k3-2408c563/synthetic/   v1.0.0 synthetic-data/ (census only)
      training-run/repro/data/damit-20250610T000301Z/
                                           copy of source/repro/data/damit-20250610T000301Z
    data/
      interpretation-20260926/             from B/interpretation/
      interpretation-20260930/             from B/interpretation/
      direct-workflow-benchmark/           B/workflow-benchmark/
      damit-postsnapshot-check/            B/damit-census/
      publication-revision-20260917-r2/    A/k3-followup/revision/
      generalization-20260910/             ALCDEF result and Gaia DR3 spin table
      external-models-final.json           written by bind-models (below)
      solver-internal-capacity-corrected/20260911/expanded-solver/convexinv/
                                           patched convexinv (below)
    experiments/period-refinement-20260921/
                                           repro/period_search/*.py and B/period-search/results/
```

### Arrange the releases

Work in an empty folder `<parent>`. Extract the `v1.0.0` archives there (they
create `artifact-root/` and `synthetic-data/`), extract the `paper-v1` archive
anywhere, and set `E` to its folder:

```bash
E=/path/to/delphi-paper-v1-evidence
T=DeLPHI-followup
git clone --branch paper-v1 https://github.com/HBJ1004/DeLPHI_LC.git $T/source
mkdir -p $T/inputs/frozen-artifacts/k3-2408c563 $T/inputs/training-run/repro/data \
         $T/data/interpretation-20260926 $T/data/interpretation-20260930 \
         $T/data/generalization-20260910/locked-cohort \
         $T/data/generalization-20260910/sources/gaia-dr3-spins-20260911 \
         $T/experiments/period-refinement-20260921/end-to-end paper/figures
mv artifact-root $T/inputs/frozen-artifacts/k3-definitive-7874092
mv synthetic-data $T/inputs/frozen-artifacts/k3-2408c563/synthetic
cp -r $T/source/repro/data/damit-20250610T000301Z $T/inputs/training-run/repro/data/

I=$E/part-b-analyses/interpretation
cp -r $I/primary-intervals.json $I/random-axes-cdf.npz $I/mirror-symmetry.json \
      $I/candidate-count.json $I/shapley $I/shapley-summary.json $T/data/interpretation-20260926/
cp $I/amplitude-aspect.json $I/amplitude-aspect.npz $I/amplitude-aspect-paired.json \
   $I/label-shuffle-spread.json $I/sampling-baselines.json $T/data/interpretation-20260930/
cp -r $E/part-b-analyses/workflow-benchmark $T/data/direct-workflow-benchmark
cp -r $E/part-b-analyses/damit-census $T/data/damit-postsnapshot-check
cp -r $E/part-a-validation/k3-followup/revision $T/data/publication-revision-20260917-r2
cp $E/part-a-validation/evidence/source-records/alcdef-gaia-analysis.json \
   $T/data/generalization-20260910/locked-cohort/alcdef-gaia-transfer-analysis-30-20260911.json
curl -L -o $T/data/generalization-20260910/sources/gaia-dr3-spins-20260911/table3.dat \
   https://cdsarc.cds.unistra.fr/ftp/J/A+A/675/A24/table3.dat

P=$T/experiments/period-refinement-20260921
cp $T/source/repro/period_search/*.py $P/
cp $E/part-b-analyses/period-search/results/* $P/end-to-end/
mv $P/end-to-end/warm170-first.* $P/
```

The Gaia DR3 spin table is that of Ďurech and Hanuš (2023, A&A 675, A24) at
CDS. Its SHA-256 on 2026-10-01 was
`e7f9743ba094860e553342eff83fe310f6dfcf7e6f740e0794f798a5e81ef865`, the same as
the copy used for the paper.

The copies of the released results in `data/` serve as inputs of later scripts
and are overwritten when a script is rerun, so compare a rerun with the copy in
`$E`. Install the repository (`python -m pip install -e "$T/source[test,plot]"`)
and run the scripts from `$T/source`, for example
`python repro/primary_intervals.py`. When we tested this layout, rerunning
`primary_intervals.py`, `candidate_count.py`, `label_shuffle_spread.py`,
`compute_reference_baselines.py`, `sampling_baselines.py`, `mirror_symmetry.py`,
`amplitude_aspect_comparison.py`, `amplitude_aspect_paired.py`,
`group_shapley_summary.py`, `damit_census/lc_density.py`,
`direct_workflow_benchmark/budget_equivalence.py` and
`direct_workflow_benchmark/score_classical_ladder.py` in this layout
reproduced the released files exactly or to the last digits of floating-point
round-off.

### DAMIT data

The DAMIT database (Ďurech et al. 2010, A&A 513, A46;
<https://damit.cuni.cz/projects/damit/>) is not redistributed here. The paper
used a copy retrieved in June 2025 and stored as `damit-20250610T000301Z/`.
DAMIT publishes monthly complete exports, and its export of 1 June 2025 is
identical to that copy for all tables and all lightcurve files and for the
spin files of the 170 asteroids:

```bash
curl -L -O https://damit.cuni.cz/projects/damit/exports/complete/damit-20250601T000301Z.tar.gz
tar xzf damit-20250601T000301Z.tar.gz
mv damit-20250601T000301Z damit-20250610T000301Z
```

The archive is about 1.4 GB. The repository lists the SHA-256 of the 442 files
of the main sample (model table, lightcurves and spin files) in
`repro/data/damit-20250610T000301Z/source_files.jsonl`. From `<parent>`:

```bash
python - <<'EOF'
import hashlib, json
bad = 0
for line in open("DeLPHI-followup/source/repro/data/damit-20250610T000301Z/source_files.jsonl"):
    row = json.loads(line)
    data = open("damit-20250610T000301Z/" + row["path"], "rb").read()
    bad += hashlib.sha256(data).hexdigest() != row["sha256"]
print("files differing from the June 2025 snapshot:", bad)
EOF
```

It should print 0. Single tables and lightcurves can also be exported from
DAMIT (`https://damit.cuni.cz/projects/damit/exports/table/asteroids`,
`.../exports/table/asteroid_models`,
`.../light_curves/exportAllForAsteroid/<DAMIT asteroid id>/plaintext` and
`.../generated_files/open/AsteroidModel/<model id>/spin.txt`), as
`lc_pipeline/k3/temporal_external.py` does. A current download differs from the
June 2025 snapshot: DAMIT has added models and lightcurves since then (Section
6.5 of the paper), and the current table exports also differ in format, so
current files will not match these checksums.

### Patched `convexinv`

The inversion comparison, the withheld-lightcurve test and the physical pilot
of the period search use `convexinv` from the DAMIT software, with three storage
limits raised so that the largest data sets of the sample fit (Appendix E). The
fitting algorithm is unchanged.

```bash
curl -L -O https://damit.cuni.cz/projects/damit/files/version_0.2.1.tar.gz
sha256sum version_0.2.1.tar.gz   # b1bc2402732e8fd4d1bfe56973a3693bf7ceb6f3bb31e8b5f03926fac59c410c
tar xzf version_0.2.1.tar.gz
cd version_0.2.1/convexinv
sed -i -e 's/^\(#define MAX_LC *\)[0-9]*/\1210/' \
       -e 's/^\(#define MAX_N_OBS *\)[0-9]*/\126612/' \
       -e 's/^\(#define POINTS_MAX *\)[0-9]*/\11549/' constants.h
make convexinv
```

This allows 210 lightcurves, 26,612 observations and 1,549 observations per
lightcurve. Copy the resulting `convexinv/` folder to
`DeLPHI-followup/data/solver-internal-capacity-corrected/20260911/expanded-solver/convexinv/`.

### Model binding file

The period-search score check, the ZTF predictions and the ALCDEF predictions load the 25 networks
through a binding file that records their checksums. Write it from
`DeLPHI-followup/source`:

```bash
python -m repro.prepare_k3_followup_inputs bind-models \
  --artifact-root ../inputs/frozen-artifacts/k3-definitive-7874092 \
  --archive-index ../inputs/frozen-artifacts/k3-definitive-7874092/release/archive-index.json \
  --publication-package-manifest /path/to/v1.0.0/publication-archive-manifest.json \
  --output ../data/external-models-final.json
```

This reproduced the authors' file byte for byte.

### Scripts that cannot run from the released files alone

- The run scripts of the inversion comparison (steps 1 to 7 and 9 in its
  [README](../repro/direct_workflow_benchmark/README.md)) check lock files that
  store the authors' paths and computer, and need saved fits that are part of
  the raw fits. `candidate_accuracy_cost.py` and `score_exploratory_timing.py`
  also read the raw per-search records.
- The reruns of the lower-quality, ZTF and ALCDEF predictions need the
  external photometry (Fink, ALCDEF, Gaia DR3) and Horizons geometry, which are
  not redistributed.

## 5. Retraining the networks

The training sequence is run with the `delphi-k3` command (each subcommand has
`--help`) on a clean Git checkout, whose commit it records. It needs
`synthetic-data/` from `v1.0.0`, the DAMIT snapshot, the catalog and the
cross-validation assignment of `repro/data/damit-20250610T000301Z/`
(`catalog.jsonl`, `publication-splits-v2.3.json`), and a CUDA GPU.

1. `delphi-k3 train-publication-synthetic --train-directory synthetic-data/train
   --validation-directory synthetic-data/validation --seed <seed> ...` for the
   seeds 17, 42, 137, 777 and 2027.
2. `delphi-k3 train-publication-real-oof --pretrained-checkpoint <network of step 1>
   --synthetic-train-directory synthetic-data/train --catalog ... --dump-root
   damit-20250610T000301Z --splits ... --fold <run> --seed <seed> ...` for each of
   the five cross-validation runs and five seeds.
3. `evaluate-publication-real-oof` for each network, then
   `evaluate-publication-real-oof-ensemble --seeds 17 42 137 777 2027` for the
   combined five networks of each run.
4. The controls: `evaluate-publication-real-oof-input-swap`,
   `train-publication-synthetic-label-shuffle` with
   `evaluate-publication-synthetic-label-shuffle`, and
   `evaluate-publication-real-oof-diagnostic --control <name>`.

`python -m repro.export_k3_bundles` turns trained checkpoints into network
archives like `k3-oof-fold-*.tar.gz`. `delphi-k3 train-smoke --objects 8
--output <dir> --allow-dirty` exercises the code path on a few asteroids; its
numbers mean nothing.

The computing environment recorded with the trained networks
(`artifact-root/release/environment/environment.json` and `pip-freeze.txt`) is
an Intel Core i5-13600KF CPU under WSL2, an NVIDIA RTX 4070 GPU (12 GB),
Python 3.12 and PyTorch 2.5.1 with CUDA 12.4. The per-epoch training losses are in
`artifact-root/models/*.jsonl`. The total training time was not recorded.

## 6. What cannot be reproduced exactly

- **Training on a GPU is not bit-for-bit repeatable.** Retraining with the same
  seeds gives slightly different networks and results (paper Appendix A).
  The paper therefore reports the spread over the five seeds (Table 7) and
  releases the trained networks and their predictions, which reproduce every
  reported value.
- **Run times** depend on the computer. The paper's times were measured on the
  workstation described in Appendix E (Intel Core i5-13600KF under WSL2, RTX
  4070). Bootstrap intervals of times describe the asteroids on that computer
  only.
- **Raw convex-inversion fits** of the inversion comparison (2.6 GB) are not in
  the releases and are available from the authors on request. The released
  files hold the selected fits, scores and times of every asteroid.
- **DAMIT data** are not redistributed. The DAMIT export of 1 June 2025 matches
  the files used (Section 4), but later DAMIT versions differ.
- **External photometry** (ZTF through Fink, ALCDEF, Gaia DR3) and Horizons
  geometry are not redistributed. The releases hold the prepared inputs'
  checksums, the predictions and the scores.
- **Figures.** Figures 13 and 18 are drawn by
  `repro/direct_workflow_benchmark/plot_direct_workflow.py` and
  `repro/period_search/benchmark_end_to_end.py`. The other figures were drawn
  by the manuscript's plotting script, which is kept with the manuscript source
  and included unchanged in the `paper-v1` archive as
  `B/figures-code/render_main_figures.py`. It reads the result files listed in
  Section 3 from the folder layout of Section 4 and writes PDF files; it is
  provided as a record of how each figure was drawn rather than as a supported
  tool.
- Several period-search scripts were edited after their runs, and their
  checksums differ from those stored with the results; the
  [period-search README](../repro/period_search/README.md) describes the
  reruns that show the same results.
