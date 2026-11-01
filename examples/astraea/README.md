# Worked example: (5) Astraea

This folder holds a complete DeLPHI input and the output you should get with
the retrained v1.1.0 networks. Use the
[prediction guide](../../docs/usage.md) for downloads and commands.

| | |
|---|---|
| Asteroid | (5) Astraea, DAMIT asteroid 103, benchmark ID `asteroid_103` |
| Input | [`observations.json`](observations.json): 25 DAMIT lightcurves, 881 observations, 1958–2004 (193 KB) |
| Period | 16.80059 h, from DAMIT model 1816 (quality flag 4), the first accepted DAMIT solution, as in the paper |
| Network set | `k3-oof-fold-0`: Astraea is a test asteroid of cross-validation run 0 |
| Expected output | [`expected_prediction.json`](expected_prediction.json), computed on a CPU |

## Run it

Download `k3-oof-fold-0-v1.1.0.tar.gz` and check its checksum as described in
the [prediction guide](../../docs/usage.md#2-download-and-check-one-network-set).
Extract it into `networks/`, then run from the repository root with the
retrained run-0 networks in `networks/k3-oof-fold-0`:

```bash
delphi-k3-predict \
  --bundle networks/k3-oof-fold-0 \
  --input examples/astraea/observations.json \
  --output prediction.json

python examples/prediction_to_poles.py prediction.json \
  --compare examples/astraea/expected_prediction.json
```

The last line of the second command should read:

```text
largest axis difference 0.0000 deg: matches examples/astraea/expected_prediction.json
```

Small differences are normal between computers and between a CPU and a GPU.
On our GPU the axes differed from the CPU result by at most 0.0087° and the
scores by about 0.0015. The comparison accepts up to 0.1°.
Nearly tied candidates can change between GPU runs; see
[Run-to-run GPU differences](../../docs/reproduction.md#run-to-run-gpu-differences).

Do not run this example with another network set. Runs 1–4 trained on
Astraea, and the program stops with an error saying that `asteroid_103` is a
benchmark asteroid used in training that network set.

## Expected candidates

Each candidate axis has two ends, which give six starting poles
(ecliptic J2000, degrees):

| Candidate | Axis (x, y, z) | End 1 (λ, β) | End 2 (λ, β) | Angle to the DAMIT pole |
|---|---|---|---|---|
| 1 | (−0.1505, 0.3621, 0.9199) | (112.6, +66.9) | (292.6, −66.9) | 28.6° |
| 2 | (0.2504, −0.2152, 0.9439) | (319.3, +70.7) | (139.3, −70.7) | 69.7° |
| 3 | (−0.0164, 0.5197, 0.8542) | (91.8, +58.7) | (271.8, −58.7) | 28.4° |

The list order means nothing; the candidates are unranked.

The last column is the angle between each axis and the reference pole of
DAMIT model 1816, λ = 124°, β = +39°, counting an axis and its opposite end as
the same. The smallest of the three, 28.42°, is the **oracle error** of
Astraea. It agrees to two decimal places with the value recorded for Astraea
in the paper's retrained cross-validation run 0 (28.4176°). Using the estimated
period instead gives 24.91°. The oracle error needs a known reference pole,
so you cannot compute it for a new asteroid, and it does not tell how close
any single candidate is to the true pole. In real use you would not know
which candidate is the good one: start an inversion from all six poles and
keep the fit with the lowest residual.

The example also shows the mirror ambiguity (paper Section 6.2.1).
Candidate 2 lies about 10° from the mirror of candidate 1, the pole at
(λ + 180°, β). Treat such a pair with the same caution as a mirror pair in
classical inversion.

Other fields of `expected_prediction.json`: `fold` 0 and `bundle_id`
`k3-oof-fold-0`; `calibration.cone90_deg` 44.7 and `cone95_deg` 90.0, which
apply to the run-0 benchmark asteroids as a group and are not uncertainties of
these candidates; `risk_deg` `null`. All fields are explained in the
[prediction guide](../../docs/usage.md#6-read-the-output).

## Timing

Measured on an Intel Core Ultra 7 270K Plus CPU and an NVIDIA RTX 5070 Ti GPU,
using 12 CPU cores for prediction:

| | CPU | GPU (`--device cuda`) |
|---|---|---|
| Whole `delphi-k3 estimate-period` command (one BLAS thread on CPU) | 3.5 s | 3.4 s |
| Whole `delphi-k3-predict` command, including start-up and loading the five networks | about 4.3 s | 2.6 s |
| Prediction only, networks already loaded (Python API, second call) | 2.7 s | 0.25 s |

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
