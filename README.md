# DeLPHI

DeLPHI (Deep Learning Photometry-based Hypothesis Inference) proposes three
candidate spin axes of an asteroid from its lightcurves, the observing
geometry, and a rotation period that you supply. The candidates are meant as
starting points for classical convex lightcurve inversion, not as final poles.

It scores 6,144 trial axes with five trained networks, averages their score
maps, and returns the three highest well-separated peaks. Each candidate is an
**axis**: it stands for both of its ends, which correspond to prograde and
retrograde rotation, so an inversion should start from all six ends. The three
candidates are **unranked**.

The method and its tests are described in Jo, Ishiguro and Lee,
"DeLPHI: Pole-Axis Candidates for Asteroid Lightcurve Inversion", submitted to
The Planetary Science Journal. "K3", which appears in the code and file names,
means "three candidates".

## Is DeLPHI suitable for my data?

DeLPHI was trained and tested on asteroids with high-quality models in the
Database of Asteroid Models from Inversion Techniques (DAMIT). On those 170
asteroids, using periods estimated from their lightcurves, the closest of the
three candidates has a median disagreement of 13.58° with a published pole
(71.8% within 20°), against 24.13° for the six standard starting poles of a
classical search.

It works best for **dense lightcurves from several well-separated observing
geometries, with a well-constrained period**. For sparse survey photometry, a
single short observing period, or a few hundred observations, it did worse than
the standard starting poles in the paper; see
[when not to use DeLPHI](docs/usage.md#when-not-to-use-delphi) before you start.

## 1. Install

You need Python 3.11 or 3.12, about 4 GB of free disk space for the
dependencies, and a few minutes. A GPU is optional; predictions also run on a
CPU.

```bash
git clone --branch v1.0.0 https://github.com/HBJ1004/DeLPHI_LC.git
cd DeLPHI_LC
python3.12 -m venv .venv          # or python3.11
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
```

PyTorch is installed automatically in the next step. On Linux the default
download includes the NVIDIA CUDA libraries (about 2–3 GB). If you have no
NVIDIA GPU, or need a build for an older driver, install PyTorch first with the
command from [pytorch.org/get-started](https://pytorch.org/get-started/locally/),
for example the CPU-only build:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then install DeLPHI and run the checks:

```bash
python -m pip install -e '.[test,plot]'
delphi-k3 validate-protocol          # one line of JSON with "sha256": "84dee816d08a..."
delphi-k3 renderer-smoke --cases 8   # one line of JSON ending in "passed": true}
python -m pytest -q tests
```

The first check confirms that the fixed settings of the paper
(`repro/k3_redesign_spec.yaml`) are intact. The second compares the brightness
simulator that made the simulated training lightcurves with an independent
reference calculation. The last runs the automated tests.

Tests that need stored research data or a solver are skipped when those inputs
are unavailable; the count depends on the local data. Set `DELPHI_DATA` to the
extracted `own-period-20261021/` folder to enable the stored-data checks (see the
[reproduction guide](docs/reproduction.md#paper-i-results)). The training smoke
test records the current Git commit and needs a Git checkout; a source archive
without Git metadata cannot pass that provenance check. To install the exact
package versions of our test environment, add `-c constraints/ci.txt` to the install command.

## 2. Make your first prediction

Follow the [prediction guide](docs/usage.md). In a few minutes it downloads one
set of trained networks (12 MB, from the
[Zenodo data record](https://doi.org/10.5281/zenodo.23757942)), runs a
worked example, the asteroid (5) Astraea, against its expected output, explains
every field of the result, and turns the candidate axes into the six starting
poles for an inversion. It also explains which network set to use, which
matters for the 170 asteroids of the paper.

To prepare your own asteroid, see the [data-format guide](docs/data-format.md):
it converts DAMIT lightcurve files, or takes your own photometry with geometry
from JPL Horizons.

## 3. Other things you can do

- **Train on your own labelled asteroids:** see
  [training on your data](docs/training-your-data.md). The command trains a
  single network on your data; it is not the full procedure of the paper.
- **Check the results of the paper:** see the
  [reproduction guide](docs/reproduction.md).
- **Change settings:** all model and training settings, and which of them can
  be changed without retraining, are listed in the
  [configuration guide](docs/configuration.md).

## Code and data

The code is this repository at tag
[`v1.0.0`](https://github.com/HBJ1004/DeLPHI_LC/releases/tag/v1.0.0), software
version 1.0.0. All data are in one [Zenodo data record,
DOI 10.5281/zenodo.23757942](https://doi.org/10.5281/zenodo.23757942).

| To… | Use from the data record |
|---|---|
| Predict from your lightcurves | `k3-oof-fold-0.tar.gz` through `k3-oof-fold-4.tar.gz`: one five-network bundle per test fold |
| Check Paper I results | `delphi-own-period-results-v1.0.0.tar.gz`, with `delphi-benchmark-fits-v1.0.0.tar.gz` for the saved inversion fits; both extract into `own-period-20261021/` |
| Check the supporting comparisons | `delphi-paper-v1-evidence.tar.gz` (validation and analysis records), `delphi-supporting-results-v1.0.0.tar.gz` (sibling data folders), and `delphi-k3-artifacts-v1.0.0.tar.gz` (`artifact-root/`, with checkpoints and predictions) |
| Retrain the networks | `delphi-k3-synthetic-data-v1.0.0.tar.gz`, which extracts to `synthetic-data/`, plus the DAMIT inputs |

The [download and verification instructions](docs/reproduction.md#1-what-to-download)
list every file, including the verification JSON files, `README.md` and
`SHA256SUMS`, and describe their extraction layout.

## Paper I results

The code includes the own-period analyses, inversion benchmarks, reliability
and network tests, and median statistics and figures. See
[Paper I results](docs/reproduction.md#paper-i-results) for the settings,
section-to-file maps and data layout. Set `DELPHI_DATA` to the extracted
`own-period-20261021/` folder.

## Results in the paper

All numbers are for the 170 DAMIT asteroids, each analyzed by the networks of
the cross-validation run in which it was not used for training. The **oracle
error** of an asteroid is the smallest angle between any of its three candidate
axes and any of its published DAMIT poles. It measures how well the candidates
cover the published solutions; it uses the published poles after the
prediction and is not the error of a single pole chosen by DeLPHI.

The revised analysis uses periods estimated from the lightcurves, with frozen
networks and no choice between P/2, P and 2P after scoring. The saved
`own-period-scores.json`, `median-stats.json` and
`repro/baselines_out/reference-baselines.json` give:

| Directions | Median oracle error | Within 20° |
|---|---:|---:|
| DeLPHI, three candidate axes | 13.58° | 71.8% |
| Six standard starting poles | 24.13° | 39.4% |
| Three random axes | 32.33° | 23.3% |

The random-axis median is averaged over randomization draws, rather than
computed from each asteroid's mean over draws. With published DAMIT periods,
DeLPHI's median is 12.09°; those predictions remain available as comparison
evidence in `artifact-root/`.

In complete pole searches with `convexinv` on 140 evaluation asteroids,
six inversions started from the candidates have a median disagreement of
27.44° with DAMIT poles, against 45.58° for 12 classical starts. With the
networks kept loaded for a batch, the classical-to-DeLPHI time ratio is 1.29;
with networks loaded anew for each asteroid it is 0.40. The classical search
reaches lower residuals, so these timings do not establish a saving at equal
solution quality. The result files and saved fits are mapped in the
[reproduction guide](docs/reproduction.md).

## Citation

Please cite the paper, the software using the Zenodo DOI shown on the v1.0.0 release page,
and the [data DOI 10.5281/zenodo.23757942](https://doi.org/10.5281/zenodo.23757942).
See [CITATION.cff](CITATION.cff) and the [release notes](CHANGELOG.md).

## Use of AI assistants

As stated in the paper, we used the AI assistants Claude/Claude Code
(Anthropic) and GPT/Codex (OpenAI) to help build the DeLPHI software, to search
the literature, and to edit the language of the manuscript for grammar and
vocabulary. The authors checked all code, analyses, and text and take full
responsibility for the content.

## Source layout

- `lc_pipeline/k3/`: the method, from the lightcurve description to the
  networks, training, prediction and evaluation.
- `lc_pipeline/v2/`: shared data and geometry utilities; the name does not
  denote a separate model.
- `lc_pipeline/grid_benchmark.py` and `lc_pipeline/workflow_benchmark.py`: inversion
  benchmark execution and scoring.
- `repro/`: the frozen data splits and the scripts behind the paper's analyses.
- `examples/`: example inputs, the DAMIT converter, and the worked example.
- `tests/`: automated tests.

The software license does not cover third-party photometry or inversion
software; obtain those from their providers under their terms.
