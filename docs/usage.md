# Make your first prediction

This guide takes you from an installed copy of DeLPHI to three candidate spin
axes and the six starting poles they give for a lightcurve inversion. It first
runs a worked example, (5) Astraea, using the input in the repository.
The printed example outputs below and `expected_prediction.json` use the
retrained v1.1.0 networks.

DeLPHI needs the photometry of one asteroid, the directions from the asteroid
to the Sun and to the observer at every observation, and a rotation period.
It estimates the period with a Fourier search, or accepts a reliable period
that you already have. It does not compute ephemerides. It
returns three **unranked candidate axes**; each axis has two ends, so an
inversion should start from all six.

## 1. Before you start

- Install DeLPHI from a source checkout as described in the
  [README](../README.md#1-install). Run all
  commands below from the repository root. The prediction checks the network
  files against the frozen cross-validation split in `repro/data`, so it needs
  the checkout, not only an installed package.
- Allow a few minutes for installation and download, then seconds for one
  prediction. Times depend on your computer.
- A GPU is optional.

## 2. Download and check one network set

Use the code at tag `v1.1.0`. The retrained networks are in the
[Zenodo data record v1.1.0, DOI {DATA_DOI}](https://doi.org/{DATA_DOI}) as five
small archives, `k3-oof-fold-0-v1.1.0.tar.gz` to
`k3-oof-fold-4-v1.1.0.tar.gz`, about 12 MB each. Each holds the five networks
of one cross-validation run (the `fold` in the name) and extracts to
`k3-oof-fold-N/`. They are not in the Git source tree. For Astraea, download
only run 0 and the checksum file. DOI and record placeholders will be filled
before publication.

```bash
mkdir -p networks && cd networks
curl --fail -LO https://zenodo.org/records/{DATA_RECORD}/files/k3-oof-fold-0-v1.1.0.tar.gz
curl --fail -LO https://zenodo.org/records/{DATA_RECORD}/files/SHA256SUMS
sha256sum --check SHA256SUMS --ignore-missing     # must print "k3-oof-fold-0-v1.1.0.tar.gz: OK"
tar -xzf k3-oof-fold-0-v1.1.0.tar.gz
cd ..
```

On macOS use `shasum -a 256 --check SHA256SUMS --ignore-missing`. The folder
`networks/k3-oof-fold-0` now contains `bundle.json` and five
`seed-*.safetensors` files. DeLPHI checks the checksum of every network file
again each time it loads the set.

The full `delphi-k3-artifacts-v1.1.0.tar.gz` archive (648 MB) also includes
training histories and evaluations; it is for
[reproduction](reproduction.md#1-what-to-download).

## 3. Choose the network set

**Benchmark asteroids.** The 170 asteroids of the paper were each a test
asteroid in exactly one cross-validation run, and the networks of the other
four runs were trained on them. Use only the set of the run in which the
asteroid is a test asteroid. Look it up in
[repro/data/benchmark-asteroids.csv](../repro/data/benchmark-asteroids.csv),
which lists the benchmark ID, DAMIT id, number, name, test run and network set:

```bash
grep -i astraea repro/data/benchmark-asteroids.csv
# asteroid_103,103,5,Astraea,,0,k3-oof-fold-0
```

Set `object_id` in your observations file to the benchmark ID,
`asteroid_<DAMIT id>` (for Astraea `asteroid_103`;
[examples/damit_to_observations.py](../examples/damit_to_observations.py) does
this automatically). The program then refuses a set that was trained on the
asteroid. It recognizes the asteroid **only by this exact ID**: if you call it
`Astraea` or `5`, nothing stops you from using a contaminated set, and the
result is not a test result.

**New asteroids.** Use any one set. Choose it before you look at any output,
and record which one you used. (How the paper scored asteroids outside its
sample is described in the [reproduction guide](reproduction.md).)

## 4. Estimate the period

Run the paper's Fourier search on the observations:

```bash
delphi-k3 estimate-period examples/astraea/observations.json --output period.json
```

The default `--device auto` uses CUDA when available and otherwise uses the
CPU. You can choose `--device cpu` or `--device cuda` explicitly. Without
`--output`, the command prints the JSON to the terminal. An existing output
file is never overwritten. A failed or timed-out search exits with an error
and writes no period file.

For Astraea, the result includes these fields (numbers shortened):

```json
{
  "schema": "delphi.k3-period.v1",
  "status": "ok",
  "object_id": "asteroid_103",
  "period_hours": 16.79973752943218,
  "ranked_periods": [16.79973752943218, 33.642835943202954, 42.047203061964524,
                     8.400876432622336, 25.228764774553298, 52.3936664837061,
                     50.47280230396282],
  "method": "fourier_cpu",
  "device": "cpu",
  "provenance": "DeLPHI Fourier search, rank 1"
}
```

The whole command took about 3.5 s on our Intel Core Ultra 7 270K Plus CPU,
with one BLAS thread, and 3.4 s on our NVIDIA RTX 5070 Ti GPU. Prediction
timings below use the same PC with 12 CPU cores. Times depend on your computer
and its load. On CUDA, `method` is
`fourier_gpu` and `device` is `cuda`. Both backends use the same float64
Fourier computation; roundoff can change candidate selection or ranking near
tied minima, so identical periods are not guaranteed for every input. The
JSON records this limit in `backend_note`.

Each input epoch stays one lightcurve. The search sorts its observations by
time and converts linear brightness to magnitude with `-2.5 log10(flux)`.
Times are used as supplied; DAMIT times already include the light-time
correction. The paper's settings are fixed: resolution 1, a 180 s search
budget, and the 30-day-window fallback when no short observing interval is
usable. The search ignores `known_period`, which may be omitted from the
input when estimating a period or predicting with `--period-json`.

The paper's main results use this rank-1 period without choosing P/2 or 2P
after scoring. It is a rough period: the median difference from the DAMIT
period was 0.0162% over the 140 evaluation asteroids. Even this small
difference can shift rotational phase over observations spanning many years. A precise pole
needs a refined period from a classical period scan before inversion.

Pass the saved estimate to the existing prediction command:

```bash
delphi-k3-predict \
  --bundle networks/k3-oof-fold-0 \
  --input examples/astraea/observations.json \
  --period-json period.json \
  --output prediction-estimated-period.json
python examples/prediction_to_poles.py prediction-estimated-period.json
```

```text
object asteroid_103  bundle k3-oof-fold-0
cand end  lambda(deg)  beta(deg)
   1  +       117.4      +63.6
   1  -       297.4      -63.6
   2  +       320.2      +67.6
   2  -       140.2      -67.6
   3  +        93.9      +65.4
   3  -       273.9      -65.4
```

These are the candidate axes at the estimated period. The supplied-period
example below gives different candidates because its period differs.

`--period-json` uses the saved `period_hours` and `provenance` in place of
`known_period`. The file must contain a successful search for the same
`object_id`. Record the period file alongside your prediction. Add
`--device cuda` to the prediction command to run the networks on a GPU.

## 5. If you already have a reliable period

Keep it in `known_period` and omit `--period-json`. The supplied-period
Astraea example uses the DAMIT period of 16.80059 hours. Run this alternative
with the retrained networks:

```bash
delphi-k3-predict \
  --bundle networks/k3-oof-fold-0 \
  --input examples/astraea/observations.json \
  --output prediction.json
```

Add `--device cuda` to use an NVIDIA GPU. The command prints nothing when it
succeeds; check that `prediction.json` was written. The output file is never
overwritten, so choose a new name for a rerun. Then list the six starting poles
with this command:

```bash
python examples/prediction_to_poles.py prediction.json \
  --compare examples/astraea/expected_prediction.json
```

The output should be:

```text
object asteroid_103  bundle k3-oof-fold-0
cand end  lambda(deg)  beta(deg)
   1  +       112.6      +66.9
   1  -       292.6      -66.9
   2  +       319.3      +70.7
   2  -       139.3      -70.7
   3  +        91.8      +58.7
   3  -       271.8      -58.7
largest axis difference 0.0000 deg: matches examples/astraea/expected_prediction.json
```

Small differences are normal between computers and between a CPU and a GPU.
The comparison reports "matches" up to 0.1° by default
(`--tolerance-deg` changes this).
Nearly tied candidates can change between GPU runs; see
[Run-to-run GPU differences](reproduction.md#run-to-run-gpu-differences). The
[example's README](../examples/astraea/README.md) gives
the full expected result, the angle of each candidate to Astraea's DAMIT pole,
and measured timings.

To run your own asteroid, make `observations.json` as described in the
[data-format guide](data-format.md) (from your photometry with JPL Horizons
geometry, or from a DAMIT `lc.txt`). Estimate its period and predict as above,
or use this alternative if you already have a reliable period.

## 6. Read the output

`prediction.json` from the supplied-period Astraea example (numbers shortened):

```json
{
  "schema": "delphi.k3-ensemble-prediction.v1",
  "status": "ok",
  "object_id": "asteroid_103",
  "fold": 0,
  "bundle_id": "k3-oof-fold-0",
  "tokenizer_sha256": "04fdcf84...",
  "period_provenance": "DAMIT model 1816 of (5) Astraea (DAMIT asteroid 103), ...",
  "axes": [
    {"axis_xyz": [-0.1505, 0.3621, 0.9199], "score": -1.526, "grid_index": 1931},
    {"axis_xyz": [0.2504, -0.2152, 0.9439], "score": -1.631, "grid_index": 3989},
    {"axis_xyz": [-0.0164, 0.5197, 0.8542], "score": -1.593, "grid_index": 476}
  ],
  "calibration": {"cone90_deg": 44.70, "cone95_deg": 90.0, "source_sha256": "15bb65d6...",
                  "scope": "archived DAMIT fold; transfer coverage is not established"},
  "risk_deg": null
}
```

| Field | Meaning |
|---|---|
| `schema`, `status` | Format of this file; `status` is `ok` for every written result (invalid input stops the program with an error message instead). |
| `object_id` | Copied from the input. |
| `fold`, `bundle_id` | The cross-validation run and network set used. Record them with your result. |
| `tokenizer_sha256` | Identifies the version of the input preparation the networks expect. |
| `period_provenance` | Your `known_period.provenance`, copied so the result records which period it used. |
| `axes[].axis_xyz` | A candidate axis: a unit vector (x, y, z) in ecliptic J2000. It stands for both ends, `(x, y, z)` and `(-x, -y, -z)`. The three axes are **unranked**, and their order means nothing. |
| `axes[].score` | The averaged network score at the axis. Only differences between axes of the same prediction have any meaning. It is not a probability or an uncertainty, and the paper did not test whether the highest score marks the best axis (keeping all three axes gave a much lower oracle error than the highest-scoring one alone). Do not use it to select a pole. |
| `axes[].grid_index` | Number (0–6143) of the trial axis on DeLPHI's fixed grid of 6,144 axes where this candidate was found, before the small refinement that gives `axis_xyz`. For bookkeeping only. |
| `calibration.cone90_deg`, `cone95_deg` | Split-conformal 90% and 95% radii for the oracle error, computed from the 14 calibration asteroids of this run (with 14 asteroids the 90% radius is the largest of their oracle errors). They describe how close the nearest of the three axes came to the reference pole for that group of DAMIT asteroids. The 95% radius is always 90°, the largest possible angle between axes, so it carries no information. They are **not uncertainties of the individual candidates**, their validity for other data is not established (`scope`), and the paper does not use them. |
| `calibration.source_sha256` | Checksum of the file the radii came from. |
| `risk_deg` | Always `null`; DeLPHI gives no individual error estimate. |

## 7. From axes to six starting poles

For an axis `(x, y, z)`, the ecliptic longitude and latitude of its first end
are

```text
lambda = atan2(y, x) mod 360°
beta   = asin(z)
```

and its opposite end is `(lambda + 180° mod 360°, -beta)`. The three axes
therefore give six starting poles. `examples/prediction_to_poles.py` prints
them, as above; in Python, `lc_pipeline.v2.convexinv.axis_to_six_pole_starts`
returns the same six `(lambda, beta)` pairs.

Run your inversion once from each of the six poles, with everything else
(period, shape and scattering settings) as you would set it for a classical
search, and keep the fit with the **lowest residual**. In DAMIT's `convexinv`,
the starting pole is set on the first two lines of the parameter file
(initial λ and β, each followed by 1 to let it vary);
`lc_pipeline.v2.convexinv.ConvexinvParameters` and
`write_convexinv_parameters` write that file. `convexinv` reads lightcurves in
the DAMIT `lc.txt` layout described in the
[data-format guide](data-format.md#damit-lightcurve-files-lctxt), which holds
the same numbers as the observations file.

When two candidates are near mirror images of each other, (λ, β) and
(λ + 180°, β), as candidates 1 and 2 of Astraea are, treat the choice between
them with the same caution as a mirror pair in classical inversion. If a
DeLPHI-guided search and a classical search select different poles, examine
both fits and use a wider search or independent information where possible.

Do not select a candidate by its closeness to a published pole. That angle,
the **oracle error** of the paper, needs a reference pole and is an evaluation
quantity for the set of three axes; it is not the accuracy of any single axis
and is not available for a new asteroid.

## When not to use DeLPHI

From the paper (Sections 6.1.2, 6.1.4 and 6.4.3), DeLPHI is best for dense
lightcurves from several well-separated epochs with a well-constrained
rotation period. In the paper it was **worse than the six standard starting
poles** of a classical search for

- the tested ZTF and ALCDEF photometry. For ALCDEF,
  standard poles gave detectably lower errors by both the median and the
  pre-specified mean; no difference from random axes was detected;
- data from a single 30-day group of lightcurves in the tested reductions.

The observation-cap tests do not define a strict minimum. On asteroids with
a period, caps of 50 or 100 observations gave no detectable median difference
from standard poles; DeLPHI did better at caps of 200 and 500 observations.
At smaller caps, failed period searches further limited the usable sample.

For survey or single-group data, start the inversion from the standard poles.
The six standard starting poles used in the paper (Appendix E) are, in ecliptic
longitude and latitude, (0°, 0°), (180°, 0°), (90°, 60°), (240°, 60°),
(90°, −60°) and (240°, −60°).

The period must be well constrained. In the paper's DAMIT-period
sensitivity check (Section 6.3.3), errors of up to 1% changed the median
oracle error by about 1.02° or less, but half or twice the period raised it from
10.48° to 15.31° and 15.21° (still below the 24.13° standard-pole median on these
asteroids), and choosing between a period and its half or double is the main
difficulty in practice. When alternative poles or the
best attainable fit matter, for example in detailed shape modelling, a wider
search remains useful, since DeLPHI does not tell when its short search is
sufficient.

## Python API

```python
from pathlib import Path
from lc_pipeline.k3.bundle import load_predictor
from lc_pipeline.k3.predict import read_observations

object_id, period, epochs = read_observations(Path("examples/astraea/observations.json"))
model = load_predictor(Path("networks/k3-oof-fold-0"), device="cpu")   # load once
result = model.predict(epochs, known_period=period, object_id=object_id)
for candidate in result["axes"]:
    print(candidate["axis_xyz"])
```

On the same PC, the whole `delphi-k3-predict` command took about 4.3 s on the
CPU and 2.6 s on the GPU. Loading the networks once and predicting many
asteroids in one session is faster: the second API call took 2.7 s on the CPU
and 0.25 s on the GPU. See the
[example's timing table](../examples/astraea/README.md#timing).

For the scientific method and its interpretation, cite Jo, Ishiguro and Lee,
"DeLPHI: Neural-Network Pole Candidates for Asteroid Lightcurve Inversion", submitted to
The Planetary Science Journal.
