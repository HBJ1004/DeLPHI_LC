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
asteroids, the closest of the three candidates lies on average 15.77° from a
published pole (median 12.09°, 74.1% within 20°), against 24.42° for the six
standard starting poles of a classical search.

It works best for **dense lightcurves from several well-separated observing
geometries, with a well-constrained period**. In the paper it was **worse than
the standard starting poles** for

- sparse survey photometry (ZTF) and the ALCDEF lightcurves tested,
- lightcurves from a single 30-day period,
- 200 or fewer observations in total,

and the period must not be wrong by a factor of two for the best results. For
such data, use the standard starting poles of classical inversion instead.

## 1. Install

You need Python 3.11 or 3.12, about 4 GB of free disk space for the
dependencies, and a few minutes. A GPU is optional; predictions also run on a
CPU.

```bash
git clone https://github.com/HBJ1004/DeLPHI_LC.git
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
python -m pytest -q tests repro/tests
```

The test suite takes about half a minute and should end with all tests passed
and two skipped. The skipped tests need local research data that is not part of
the repository. To install the exact package versions of our test environment,
add `-c constraints/ci.txt` to the install command.

## 2. Make your first prediction

The trained networks are not stored in git. They are in release
[`v1.0.0`](https://github.com/HBJ1004/DeLPHI_LC/releases/tag/v1.0.0), as five
sets of five networks, one set per cross-validation run (`k3-oof-fold-0` to
`k3-oof-fold-4`, about 12 MB each). Download one set and check it:

```bash
mkdir -p networks && cd networks
curl -LO https://github.com/HBJ1004/DeLPHI_LC/releases/download/v1.0.0/k3-oof-fold-0.tar.gz
curl -LO https://github.com/HBJ1004/DeLPHI_LC/releases/download/v1.0.0/SHA256SUMS
sha256sum --check SHA256SUMS --ignore-missing     # must print "k3-oof-fold-0.tar.gz: OK"
tar -xzf k3-oof-fold-0.tar.gz
cd ..
```

Then follow the [step-by-step prediction guide](docs/usage.md). It runs a
worked example, the asteroid (5) Astraea, whose result you can compare with the
expected output in [examples/astraea](examples/astraea), and explains every
field of the output, how to convert the candidate axes to ecliptic longitude
and latitude, and how to pass the six starting poles to an inversion.

To prepare your own data, see the [data-format guide](docs/data-format.md). It
explains the input file, how to obtain the Sun and observer directions from JPL
Horizons, and how to convert DAMIT lightcurve files with
[examples/damit_to_observations.py](examples/damit_to_observations.py).

**Which network set to use.**

- For a **new asteroid**, any one set is the supported choice. Choose it before
  looking at results. (The paper's tests on new asteroids averaged all 25
  networks; see the [prediction guide](docs/usage.md).)
- For one of the **170 benchmark asteroids**, use only the set of the
  cross-validation run in which it is a test asteroid, listed in
  [repro/data/benchmark-asteroids.csv](repro/data/benchmark-asteroids.csv). The
  other sets were trained on it. The software refuses such a combination, but
  only if you name the asteroid by its benchmark ID, `asteroid_<DAMIT id>`; for
  example (5) Astraea is `asteroid_103`.

## 3. Other things you can do

- **Train on your own labelled asteroids:** see
  [training on your data](docs/training-your-data.md). The command trains a
  single network on your data; it is not the full procedure of the paper.
- **Check the results of the paper:** see the
  [reproduction guide](docs/reproduction.md).
- **Change settings:** all model and training settings, and which of them can
  be changed without retraining, are listed in the
  [configuration guide](docs/configuration.md).

## Which release do I need?

| To... | Use |
|---|---|
| Run DeLPHI on your own lightcurves | The trained networks in release [`v1.0.0`](https://github.com/HBJ1004/DeLPHI_LC/releases/tag/v1.0.0) and this repository |
| Check the numbers of the paper | Release [`paper-v1`](https://github.com/HBJ1004/DeLPHI_LC/releases/tag/paper-v1) together with `v1.0.0`, and the code at tag `paper-v1` |

The README inside the `paper-v1` archive lists the files behind each section of
the paper. All other releases (`v1.0`, `v1.0.1`, and the
`publication-evidence-*` releases) are earlier versions and are superseded.

## Results in the paper

All numbers are for the 170 DAMIT asteroids, each analyzed by the networks of
the cross-validation run in which it was not used for training. The **oracle
error** of an asteroid is the smallest angle between any of its three candidate
axes and any of its published DAMIT poles. It measures how well the candidates
cover the published solutions; it uses the published poles after the
prediction and is not the error of a single pole chosen by DeLPHI.

| Directions | Mean oracle error | Median | Within 20° |
|---|---:|---:|---:|
| DeLPHI, three candidate axes | 15.77° | 12.09° | 74.1% |
| Six standard starting poles | 24.42° | 24.13° | 39.4% |
| Three random axes | 33.84° | 32.33° | 23.3% |

In complete pole searches with the program `convexinv` on 140 of these
asteroids (Section 6.4.2 of the paper), six inversions started from the
candidates agreed better with the DAMIT poles than 12 classical starts (mean
disagreement 17.32° against 22.52°). With the networks kept loaded for a batch
of asteroids they were 1.29 times faster; with the networks loaded anew for each
asteroid they were slower (ratio 0.30). The classical search reached slightly
lower residuals, so this is not a saving at equal solution quality. Scoring and
selecting the candidates takes a median of 0.51 s per asteroid on an NVIDIA
RTX 4070, without reading the data and loading the networks.

## Citation

Please cite the paper and the release you used; see [CITATION.cff](CITATION.cff).
Changes between releases are listed in the [release notes](CHANGELOG.md).

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
- `lc_pipeline/publication/`: benchmark definitions and result aggregation.
- `repro/`: the frozen data splits and the scripts behind the paper's analyses.
- `examples/`: example inputs, the DAMIT converter, and the worked example.
- `tests/`: automated tests.

The software license does not cover third-party photometry or inversion
software; obtain those from their providers under their terms.
