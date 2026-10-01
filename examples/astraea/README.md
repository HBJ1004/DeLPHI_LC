# Worked example: (5) Astraea

This folder holds a complete DeLPHI input and the output you should get from
it. Use it to check your installation before you prepare your own data.

| | |
|---|---|
| Asteroid | (5) Astraea, DAMIT asteroid 103, benchmark ID `asteroid_103` |
| Input | [`observations.json`](observations.json): 25 DAMIT lightcurves, 881 observations, 1958–2004 (193 KB) |
| Period | 16.80059 h, from DAMIT model 1816 (quality flag 4), the first accepted DAMIT solution, as in the paper |
| Network set | `k3-oof-fold-0`: Astraea is a test asteroid of cross-validation run 0 |
| Expected output | [`expected_prediction.json`](expected_prediction.json), computed on a CPU |

## Run it

From the repository root, with the run-0 networks extracted to
`networks/k3-oof-fold-0` as in the [prediction guide](../../docs/usage.md):

```bash
delphi-k3-predict \
  --bundle networks/k3-oof-fold-0 \
  --input examples/astraea/observations.json \
  --output prediction.json

python examples/prediction_to_poles.py prediction.json \
  --compare examples/astraea/expected_prediction.json
```

The last line of the second command should read
`largest axis difference ... deg: matches examples/astraea/expected_prediction.json`.
Small differences are normal between computers and between a CPU and a GPU.
On our GPU the axes differed from the CPU result by at most 0.007° and the
scores by about 0.003. The comparison accepts up to 0.1°.

Do not run this example with another network set. Runs 1–4 trained on
Astraea, and the program stops with an error saying that `asteroid_103` is a
benchmark asteroid used in training that network set.

## Expected candidates

Each candidate axis has two ends, which give six starting poles
(ecliptic J2000, degrees):

| Candidate | Axis (x, y, z) | End 1 (λ, β) | End 2 (λ, β) | Angle to the DAMIT pole |
|---|---|---|---|---|
| 1 | (−0.3558, 0.5205, 0.7762) | (124.4, +50.9) | (304.4, −50.9) | 11.9° |
| 2 | (0.4701, −0.3886, 0.7925) | (320.4, +52.4) | (140.4, −52.4) | 87.5° |
| 3 | (−0.1773, 0.5927, 0.7856) | (106.6, +51.8) | (286.6, −51.8) | 17.6° |

The list order means nothing; the candidates are unranked.

The last column is the angle between each axis and the reference pole of
DAMIT model 1816, λ = 124°, β = +39°, counting an axis and its opposite end as
the same. The smallest of the three, 11.9°, is the **oracle error** of
Astraea. It agrees with the value recorded for Astraea in the paper's
cross-validation run 0 (11.918°). The oracle error needs a known reference pole, so you cannot compute
it for a new asteroid, and it does not tell how close any single candidate is
to the true pole. In real use you would not know which candidate is the good
one: start an inversion from all six poles and keep the fit with the lowest
residual.

The example also shows the mirror ambiguity (paper Section 6.2.1).
Candidate 2 lies about 10° from the mirror of candidate 1, the pole at
(λ + 180°, β). Treat such a pair with the same caution as a mirror pair in
classical inversion.

Other fields of `expected_prediction.json`: `fold` 0 and `bundle_id`
`k3-oof-fold-0`; `calibration.cone90_deg` 46.6 and `cone95_deg` 90.0, which
apply to the run-0 benchmark asteroids as a group and are not uncertainties of
these candidates; `risk_deg` `null`. All fields are explained in the
[prediction guide](../../docs/usage.md#5-read-the-output).

## Timing

Measured with Python 3.12, PyTorch 2.3.1, an Intel Core i5-13600KF CPU and an
NVIDIA GeForce RTX 4070 GPU:

| | CPU | GPU (`--device cuda`) |
|---|---|---|
| Whole `delphi-k3-predict` command, including start-up and loading the five networks | 6.0 s | 4.0 s |
| Prediction only, networks already loaded (Python API, second call) | 4.0 s | 0.7 s |

## How the input was made

`observations.json` was converted from the June 2025 DAMIT snapshot used in the
paper with [`../damit_to_observations.py`](../damit_to_observations.py):

```bash
python examples/damit_to_observations.py --damit-id 103 \
  --lc damit-20250610T000301Z/files/asteroid_103/lc.txt \
  --models-table damit-20250610T000301Z/tables/asteroid_models.csv \
  --asteroids-table damit-20250610T000301Z/tables/asteroids.csv \
  --output observations.json
```

Without a local DAMIT copy, download the same data instead:

```bash
python examples/damit_to_observations.py --damit-id 103 --output my_astraea.json
```

On 1 October 2026 the downloaded `lc.txt` was identical to the snapshot. Only
the period source text differs (it names the download address and date), and
that text does not affect the prediction.

DAMIT lists a second model of Astraea, model 104 (λ = 126°, β = +40°,
16.80061 h), without a quality flag. The paper used only models with quality
flag 3 or higher, so the converter takes the period of model 1816.

## Data source

The photometry and geometry in `observations.json` are the lightcurves of
(5) Astraea as compiled in DAMIT (Ďurech, Sidorin & Kaasalainen 2010, A&A 513,
A46; https://damit.cuni.cz/), converted with
[`examples/damit_to_observations.py`](../damit_to_observations.py). They are
included only as an example. The lightcurves come from the original
publications listed for this asteroid in DAMIT, which should be cited if you use
the data scientifically; the DeLPHI software license does not cover them.
