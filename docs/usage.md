# Make your first prediction

This guide takes you from an installed copy of DeLPHI to three candidate spin
axes and the six starting poles they give for a lightcurve inversion. It first
runs a worked example, (5) Astraea, whose expected result is in the
repository, so you can check every step before you use your own data.

DeLPHI needs the photometry of one asteroid, the directions from the asteroid
to the Sun and to the observer at every observation, and a rotation period
that you supply. It does not compute ephemerides or determine the period. It
returns three **unranked candidate axes**; each axis has two ends, so an
inversion should start from all six.

## 1. Before you start

- Install DeLPHI from a source checkout as described in the
  [README](../README.md#1-install) (`python -m pip install -e .`). Run all
  commands below from the repository root. The prediction checks the network
  files against the frozen cross-validation split in `repro/data`, so it needs
  the checkout, not only an installed package.
- Expected time: a few minutes for the installation, a 12 MB download, and
  about 6 s for one prediction on a CPU (4 s with `--device cuda`).
- A GPU is optional.

## 2. Download and check one network set

The trained networks are in release
[`v1.0.0`](https://github.com/HBJ1004/DeLPHI_LC/releases/tag/v1.0.0) as five
files `k3-oof-fold-0.tar.gz` to `k3-oof-fold-4.tar.gz`. Each holds the five
networks trained in one of the five cross-validation runs of the paper
(the `fold` in the name is the run). They are not in the Git source tree.

```bash
mkdir -p networks && cd networks
curl -LO https://github.com/HBJ1004/DeLPHI_LC/releases/download/v1.0.0/k3-oof-fold-0.tar.gz
curl -LO https://github.com/HBJ1004/DeLPHI_LC/releases/download/v1.0.0/SHA256SUMS
sha256sum --check SHA256SUMS --ignore-missing     # must print "k3-oof-fold-0.tar.gz: OK"
tar -xzf k3-oof-fold-0.tar.gz
cd ..
```

On macOS use `shasum -a 256 --check SHA256SUMS --ignore-missing`. The folder
`networks/k3-oof-fold-0` now contains `bundle.json` and five
`seed-*.safetensors` files. DeLPHI checks the checksum of every network file
again each time it loads the set.

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

**New asteroids.** Any one set is the supported choice. Choose it before you
look at any output, and record which one you used. Note that the paper's tests
on asteroids outside its sample (the ALCDEF comparison, Appendix D) averaged
the score maps of all 25 networks, with
[repro/predict_k3_alcdef_transfer.py](../repro/predict_k3_alcdef_transfer.py)
(policy `all_25_frozen_models_for_new_identity`). That script needs the
research checkpoints and their model manifest rather than the `v1.0.0` sets,
and `delphi-k3-predict` does not offer it. Each set used alone is what the
paper evaluated on the benchmark asteroids.

## 4. Run the Astraea example

```bash
delphi-k3-predict \
  --bundle networks/k3-oof-fold-0 \
  --input examples/astraea/observations.json \
  --output prediction.json
```

Add `--device cuda` to use an NVIDIA GPU. The output file is never
overwritten; choose a new name for a rerun. Then list the six starting poles
and compare with the expected output:

```bash
python examples/prediction_to_poles.py prediction.json \
  --compare examples/astraea/expected_prediction.json
```

```text
object asteroid_103  bundle k3-oof-fold-0
cand end  lambda(deg)  beta(deg)
   1  +       124.4      +50.9
   1  -       304.4      -50.9
   2  +       320.4      +52.4
   2  -       140.4      -52.4
   3  +       106.6      +51.8
   3  -       286.6      -51.8
largest axis difference 0.0000 deg: matches examples/astraea/expected_prediction.json
```

Differences of up to about 0.01° between computers, or between a CPU and a
GPU, are normal. The [example's README](../examples/astraea/README.md) gives
the full expected result, the angle of each candidate to Astraea's DAMIT pole,
and measured timings.

To run your own asteroid, make `observations.json` as described in the
[data-format guide](data-format.md) (from your photometry with JPL Horizons
geometry, or from a DAMIT `lc.txt`) and use the same two commands.

## 5. Read the output

`prediction.json` from the Astraea example (numbers shortened):

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
    {"axis_xyz": [-0.3558, 0.5205, 0.7762], "score": -1.069, "grid_index": 1835},
    {"axis_xyz": [0.4701, -0.3886, 0.7925], "score": -1.208, "grid_index": 3870},
    {"axis_xyz": [-0.1773, 0.5927, 0.7856], "score": -1.113, "grid_index": 1762}
  ],
  "calibration": {"cone90_deg": 46.60, "cone95_deg": 90.0, "source_sha256": "7b73c293...",
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

## 6. From axes to six starting poles

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

From the paper (Sections 6.1.2, 6.1.3 and 6.4.3), DeLPHI is best for dense
lightcurves from several well-separated epochs with a well-constrained
rotation period. In the paper it was **worse than the six standard starting
poles** of a classical search for

- sparse survey photometry such as ZTF, and on the ALCDEF test, where it was
  not distinguishable from random axes;
- data from a single 30-day group of lightcurves, however many observations
  that group contains;
- 200 or fewer observations in total (with up to 500 it was not
  distinguishable from the standard poles).

For such data, start the inversion from the standard poles instead.

The period must be well constrained (Section 6.3.3): errors of up to 1% changed the mean
oracle error by about 0.4° or less, but half or twice the period raised it from
15.8° to 19.3° and 17.7° (still better than the standard poles on these
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
model = load_predictor("networks/k3-oof-fold-0", device="cpu")   # load once
result = model.predict(epochs, known_period=period, object_id=object_id)
for candidate in result["axes"]:
    print(candidate["axis_xyz"])
```

Loading the networks once and predicting many asteroids in one session is
much faster than starting the command for each: with the networks loaded, the
Astraea prediction takes about 4 s on our CPU and 0.7 s on our GPU.

For the scientific method and its interpretation, cite Jo, Ishiguro and Lee,
"DeLPHI: Pole-Axis Candidates for Asteroid Lightcurve Inversion", submitted to
The Planetary Science Journal.
