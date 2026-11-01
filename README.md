# DeLPHI

DeLPHI (Deep Learning Photometry-based Hypothesis Inference) estimates an
asteroid's rotation period with a Fourier search. It then proposes three
candidate spin axes from the lightcurves, observing geometry and that period.
The candidates are meant as starting points for classical convex lightcurve
inversion, not as final poles.

It scores 6,144 trial axes with five trained networks, averages their score
maps, and returns the three highest well-separated peaks. Each candidate is an
**axis**: it stands for both of its ends, which correspond to prograde and
retrograde rotation, so an inversion should start from all six ends. The three
candidates are **unranked**.

The method and its tests are described in Jo, Ishiguro and Lee,
"DeLPHI: Neural-Network Pole Candidates for Asteroid Lightcurve Inversion", submitted to
The Planetary Science Journal. "K3", which appears in the code and file names,
means "three candidates".

## Is DeLPHI suitable for my data?

DeLPHI was trained and tested on asteroids with high-quality models in the
Database of Asteroid Models from Inversion Techniques (DAMIT). On those 170
asteroids, using periods estimated from their lightcurves, the closest of the
three candidates has a median disagreement of 10.70° with a published pole
(77.1% within 20°), against 24.13° for the six standard starting poles of a
classical search.

It works best for **dense lightcurves from several well-separated observing
geometries, with a well-constrained period**. It did worse than the standard
starting poles for the tested ZTF and ALCDEF photometry and for a single
30-day group of lightcurves. The observation-cap tests do not define a strict
minimum. On asteroids with a period, no median difference was detected at
caps of 50 or 100 observations, while DeLPHI did better at caps of 200 and
500. See
[when not to use DeLPHI](docs/usage.md#when-not-to-use-delphi) before you start.

## 1. Install

You need Python 3.11 or 3.12, about 4 GB of free disk space for the
dependencies, and a few minutes. A GPU is optional; predictions also run on a
CPU.

```bash
git clone --branch v1.1.0 https://github.com/HBJ1004/DeLPHI_LC.git
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
test and the two local-binding provenance tests,
`test_local_binding_audits_graph_without_inventing_publication_evidence` and
`test_local_binding_accepts_native_training_cli_receipts` in `tests/test_network_root.py`,
record the current Git commit and need a Git checkout; a source archive
without Git metadata cannot pass those provenance checks. To install the exact
package versions of our test environment, add `-c constraints/ci.txt` to the install command.

## 2. Make your first prediction

Download the retrained run-0 networks (about 12 MB) from the
[v1.1.0 data record](https://doi.org/10.5281/zenodo.23857881) and check their checksum:

```bash
mkdir -p networks && cd networks
curl --fail -LO https://zenodo.org/records/23857881/files/k3-oof-fold-0-v1.1.0.tar.gz
curl --fail -LO https://zenodo.org/records/23857881/files/SHA256SUMS
sha256sum --check SHA256SUMS --ignore-missing     # must print "k3-oof-fold-0-v1.1.0.tar.gz: OK"
tar -xzf k3-oof-fold-0-v1.1.0.tar.gz
cd ..
```

On macOS use `shasum -a 256 --check SHA256SUMS --ignore-missing`.
Then follow the [prediction guide](docs/usage.md#4-estimate-the-period) to
estimate the rotation period and predict three candidate axes for (5) Astraea,
using `--bundle networks/k3-oof-fold-0`. The guide also shows a supplied-period
example, explains every field of the result, and turns the candidate axes into
the six starting poles for an inversion. The saved expected output uses these
retrained networks. Check
[which network set to use](docs/usage.md#3-choose-the-network-set) for the
170 asteroids of the paper.

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
[`v1.1.0`](https://github.com/HBJ1004/DeLPHI_LC/releases/tag/v1.1.0), software
DOI `10.5281/zenodo.23787947`. The revised paper uses 25 networks retrained on corrected
simulations and timings measured on one PC. Data are in the
[v1.1.0 data record, DOI 10.5281/zenodo.23857881](https://doi.org/10.5281/zenodo.23857881); downloads use
`https://zenodo.org/records/23857881/files/<name>`.

| To… | Use from the data record |
|---|---|
| Predict Astraea or use test run 0 | `k3-oof-fold-0-v1.1.0.tar.gz` (about 12 MB), extracts to `k3-oof-fold-0/` |
| Use test run 1 | `k3-oof-fold-1-v1.1.0.tar.gz` (about 12 MB), extracts to `k3-oof-fold-1/` |
| Use test run 2 | `k3-oof-fold-2-v1.1.0.tar.gz` (about 12 MB), extracts to `k3-oof-fold-2/` |
| Use test run 3 | `k3-oof-fold-3-v1.1.0.tar.gz` (about 12 MB), extracts to `k3-oof-fold-3/` |
| Use test run 4 | `k3-oof-fold-4-v1.1.0.tar.gz` (about 12 MB), extracts to `k3-oof-fold-4/` |
| Reproduce network evaluations | `delphi-k3-artifacts-v1.1.0.tar.gz` (648 MB): retrained checkpoints, evaluations, model binding and five-network fold bundles in `retrain-20261030/artifact-root/` |
| Check Paper I results | `delphi-retrain-results-v1.1.0.tar.gz`: revised results, protocols and same-PC timing receipts under `retrain-20261030/`; newly written solver fits are in `delphi-retrain-benchmark-fits-v1.1.0.tar.gz` |
| Use retained inputs and comparisons | Unchanged `delphi-own-period-results-v1.0.0.tar.gz`, `delphi-benchmark-fits-v1.0.0.tar.gz`, `delphi-supporting-results-v1.0.0.tar.gz` and `delphi-paper-v1-evidence.tar.gz`; their older network scores are historical |
| Retrain the networks | `delphi-k3-synthetic-data-v1.1.0.tar.gz`, which extracts to `retrain-20261030/synthetic/`, plus the DAMIT inputs and retraining protocol |

The [download and verification instructions](docs/reproduction.md#1-what-to-download)
list every file, including `README.md` and
`SHA256SUMS`, and describe their extraction layout.
Superseded network archives remain in the
[v1.0.0 data record, DOI 10.5281/zenodo.23757942](https://doi.org/10.5281/zenodo.23757942).

## Paper I results

The code includes the own-period analyses, inversion benchmarks, reliability
and network tests, and median statistics and figures. See
[Paper I results](docs/reproduction.md#paper-i-results) for the settings,
section-to-file maps and data layout. Set `DELPHI_DATA` to the extracted
`own-period-20261021/` folder for unchanged inputs, and select the retrained
networks and results with:

```bash
export DELPHI_NETWORK_ROOT=/path/to/data/retrain-20261030/artifact-root
export DELPHI_MODEL_BINDING="$DELPHI_NETWORK_ROOT/model-binding.json"
export DELPHI_BUNDLE_ROOT="$DELPHI_NETWORK_ROOT/bundles"
```

## Results in the paper

The candidate results below are for the 170 DAMIT asteroids, each analyzed by the networks of
the cross-validation run in which it was not used for training. The **oracle
error** of an asteroid is the smallest angle between any of its three candidate
axes and any of its published DAMIT poles. It measures how well the candidates
cover the published solutions; it uses the published poles after the
prediction and is not the error of a single pole chosen by DeLPHI.

The revised analysis uses periods estimated from the lightcurves, with the
retrained networks held fixed and no choice between P/2, P and 2P after scoring.
The saved `retrain-20261030/results/headline/own-period-scores-main.json`,
`retrain-20261030/results/median-stats.json` and
`repro/baselines_out/reference-baselines.json` give:

| Directions | Median oracle error | Within 20° |
|---|---:|---:|
| DeLPHI, three candidate axes | 10.70° | 77.1% |
| Six standard starting poles | 24.13° | 39.4% |
| Three random axes | 32.33° | 23.3% |

The random-axis median is averaged over randomization draws, rather than
computed from each asteroid's mean over draws. With published DAMIT periods,
DeLPHI's median is 10.48°; those predictions are in
`retrain-20261030/artifact-root/evaluations/`.

In complete pole searches with `convexinv` on 140 evaluation asteroids,
six inversions started from the candidates at our rough period have a median
disagreement of 23.55° with DAMIT poles, against 45.58° for 12 classical starts.
Even 146 classical starts did not reach DeLPHI's accuracy at this period.
With a refined period, the medians in an exploratory test of 20 asteroids
were 12.05° for DeLPHI and 8.37° for the classical search, with no detected
gain for DeLPHI. A precise pole still needs a refined period from a classical
scan before inversion. With the networks kept loaded for a batch, the
classical-to-DeLPHI pole-fit time ratio is 1.39; with networks loaded anew for
each asteroid it is 0.81. These times were measured on one Intel Core Ultra 7
270K Plus PC with an NVIDIA RTX 5070 Ti. The classical search
reaches lower residuals, so these timings do not establish a saving at equal
solution quality. The result files and saved fits are mapped in the
[reproduction guide](docs/reproduction.md).

## Citation

Please cite the paper, software v1.1.0 using DOI `10.5281/zenodo.23787947`,
and the [v1.1.0 data DOI 10.5281/zenodo.23857881](https://doi.org/10.5281/zenodo.23857881).
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
