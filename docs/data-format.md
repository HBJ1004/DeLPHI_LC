# Prepare compatible data

DeLPHI expects one object per JSON file for prediction, or one object per line
in a JSONL file for training. The observation part has the same shape in both.

The code checks the **structure** of the file: required fields, positive
times, brightnesses and periods, finite nonzero three-component vectors, and
at least two observations per epoch. It **cannot check units, signs, or
frames**. Magnitudes such as `12.3` are positive numbers and are accepted
silently as if they were fluxes; vectors pointing from the Earth to the
asteroid, positions in km, or equatorial coordinates are accepted too. All of
these give wrong candidates without any error message, so check them yourself
against the conventions below.

## Prediction JSON

```json
{
  "schema": "delphi.k3-observations.v1",
  "object_id": "my-asteroid-001",
  "known_period": {"hours": 7.25, "provenance": "published-period-source"},
  "epochs": [{
    "epoch_id": "2026-01-12-site-filter",
    "observations": [{
      "time_jd": 2461052.125,
      "relative_brightness": 1.034,
      "sun_asteroid_ecliptic_j2000_au": [1.42, -0.31, 0.04],
      "observer_asteroid_ecliptic_j2000_au": [0.77, -0.68, 0.03],
      "measured_error": 0.012
    }, {
      "time_jd": 2461052.132,
      "relative_brightness": 1.018,
      "sun_asteroid_ecliptic_j2000_au": [1.42, -0.31, 0.04],
      "observer_asteroid_ecliptic_j2000_au": [0.77, -0.68, 0.03],
      "measured_error": 0.012
    }]
  }]
}
```

The values are illustrative, not observations of a real object. A real file,
converted from DAMIT, is [examples/astraea/observations.json](../examples/astraea/observations.json).

| Field | Required form |
|---|---|
| `object_id` | Nonempty stable identifier; use the same ID throughout a training project. For a benchmark asteroid use its benchmark ID `asteroid_<DAMIT id>` (see the [prediction guide](usage.md#3-choose-the-network-set)). |
| `known_period.hours` | Positive rotation period in hours. |
| `known_period.provenance` | Nonempty source description, such as a DOI, catalogue version, or analysis ID. It is copied to the output as `period_provenance`. |
| `known_period.uncertainty_hours` | Optional. If present, it must be a positive number. It is checked but **not used**: the networks fold the lightcurves with `hours` alone, and the value is not copied to the output. |
| `epoch_id` | Unique nonempty identifier of one lightcurve (see below). |
| `time_jd` | Positive Julian date. Use one time scale consistently; DAMIT uses light-time-corrected times (see below). |
| `relative_brightness` | Positive, dimensionless **linear** flux. Not magnitude. |
| geometry vectors | Three finite nonzero values: **asteroid → Sun** and **asteroid → observer**, ecliptic J2000, au. |
| `measured_error` | Optional positive uncertainty in the same relative-flux units. |

Do not substitute Earth-to-asteroid vectors, RA/Dec, heliocentric positions, or
zero vectors. Convert them first to the stated asteroid-centred vectors. If you
use an ephemeris service, save its source, time scale, frame, and conversion
formula with your data.

## Epochs are lightcurves

An **epoch** in this file format is what the paper calls one **lightcurve**: a
source-defined set of observations, for example one night with one telescope
or one published lightcurve segment. DeLPHI keeps these boundaries and never
regroups observations by time gaps, so do not reconstruct epochs from
arbitrary time gaps either. Each epoch must contain at least two
observations; longer, well-sampled lightcurves are generally more informative.
Keep excluded lightcurves in your project log with their exclusion reason.

## Brightness: linear flux, normalized per lightcurve

`relative_brightness` must be linear flux. If your measurement is magnitude
`m`, convert it with any reference magnitude `m0`:

```text
relative_brightness = 10 ** (-0.4 * (m - m0))
```

In Python with NumPy: `flux = 10 ** (-0.4 * (mag - np.median(mag)))`. A
magnitude uncertainty `sigma_m` becomes
`measured_error = 0.4 * ln(10) * flux * sigma_m`, about `0.921 * flux * sigma_m`.

DeLPHI normalizes each lightcurve separately. Within every epoch it takes the
natural logarithm of the flux and subtracts the mean over that epoch, without
dividing by the amplitude (paper Section 3.2). Consequently:

- a constant factor on the flux of one lightcurve, and so the choice of `m0`,
  does not change the result, and lightcurves need not share a calibration;
- within one lightcurve, all points must share one reference, so do not mix
  filters or zero points inside an epoch, and split the data into separate
  epochs where the calibration changes;
- the relative amplitudes of different lightcurves are kept, so do not rescale
  each lightcurve to a common amplitude.

Apply quality filtering, filter handling, and any phase or distance
corrections consistently before the conversion. DeLPHI does not make
incompatible photometry physically equivalent.

## Measurement uncertainties

`measured_error` is optional. If you have uncertainties, supply them, and
record any decision to omit them. The DAMIT lightcurves used to train and test
the networks carry none, so your uncertainties are one way your data differ from
the training data; the paper (Section 6.1.2) found no general benefit in
removing them.

## Geometry from JPL Horizons

For your own photometry you need, for every observation, the vector from the
asteroid to the Sun and from the asteroid to the observer. Compute them as the
DAMIT lightcurve files do:

1. Find the light time `LT` from the asteroid to the observer at the
   observation time `jd`, and use the light-time-corrected time
   `t = jd - LT` as `time_jd`.
2. At time `t`, get the geometric heliocentric position of the asteroid
   `r_helio` and its geometric position relative to the observer `r_obs`, both
   in the ecliptic J2000 frame, in au.
3. Write `sun_asteroid_ecliptic_j2000_au = -r_helio` (minus the heliocentric
   asteroid vector) and `observer_asteroid_ecliptic_j2000_au = -r_obs`
   (observer position minus asteroid position).

Horizons gives the position of the target relative to the centre you choose,
so both vectors must be negated.

With [astroquery](https://astroquery.readthedocs.io) (`pip install astroquery`):

```python
import numpy as np
from astroquery.jplhorizons import Horizons

def vectors(target, location, epochs, aberrations):
    table = Horizons(id=target, id_type="smallbody", location=location,
                     epochs=epochs).vectors(refplane="ecliptic", aberrations=aberrations)
    xyz = np.column_stack([np.asarray(table[c], dtype=float) for c in ("x", "y", "z")])
    return xyz, np.asarray(table["lighttime"], dtype=float)

# Observation times (at most ~50 per query). These two give, within 1 s and
# 0.002 deg, the first two rows of examples/astraea/observations.json.
jd = [2436250.5703, 2436250.5750]
_, light_time = vectors("5", "500@399", jd, "astrometric")
t = (np.asarray(jd) - light_time).tolist()  # time_jd
r_helio, _ = vectors("5", "500@10", t, "geometric")
r_obs, _ = vectors("5", "500@399", t, "geometric")
sun_asteroid = -r_helio
observer_asteroid = -r_obs
```

`"500@399"` is the geocentre, as in DAMIT; an observatory code such as `"568"`
also works, and the difference is negligible for the pole. Horizons treats the
epochs of a vector query as TDB; for UTC times the offset of about one minute
moves the geometry negligibly, and `time_jd` stays in your own time scale
because only `LT` is subtracted.

[examples/horizons_geometry.py](../examples/horizons_geometry.py) does all of
this for a CSV file with the columns `epoch_id, jd, mag` (or `flux`, with
optional `mag_err` or `flux_err`) and writes a complete observations file:

```bash
python examples/horizons_geometry.py --target 5 \
  --photometry my_photometry.csv --object-id my-astraea \
  --period-hours 16.80059 --period-provenance "DAMIT model 1816" \
  --output observations.json
```

We checked this recipe against DAMIT. Recomputing all 881 Astraea observations
from Horizons reproduced the DAMIT times within 0.15 s and the Sun and
observer directions within 0.002°, and the prediction from the recomputed file
agreed with the [Astraea example](../examples/astraea) within 0.001°.

In the Horizons web interface the same settings are: ephemeris type *Vector
Table*, coordinate centre *Sun (body center) [500@10]* for the Sun vector and
*Geocentric [500@399]* (or your site) for the observer vector, reference frame
*ICRF*, reference plane *ecliptic x-y plane*, output units *au and days*, and
the vector correction *none (geometric states)* at the light-time-corrected
times. To get the light time, first request the observer-centred vectors at
your observation times with the output option that adds light time, range and
range rate, and read the `LT` column (days).

## DAMIT lightcurve files (lc.txt)

DAMIT lightcurve files already follow DeLPHI's conventions. We confirmed this
with the loader DeLPHI used for the paper (`lc_pipeline/v2/data.py` and
`lc_pipeline/k3/damit.py`) and with JPL Horizons, as described above. An
`lc.txt` file contains the number of lightcurves, then for each lightcurve a
header with the number of points and a calibration flag, followed by one row
per point:

```text
JD   intensity   Sx Sy Sz   Ex Ey Ez
```

with the light-time-corrected Julian date, the **linear** relative intensity,
and the asteroid-centred vectors to the Sun (`S`) and to the Earth (`E`) in
ecliptic J2000, in au. Each DAMIT lightcurve becomes one epoch, and DAMIT gives
no per-point uncertainties.

[examples/damit_to_observations.py](../examples/damit_to_observations.py)
converts one asteroid, either from a local `lc.txt` with a period you supply,
or by downloading the lightcurves and the period from DAMIT:

```bash
# local lc.txt and your own period
python examples/damit_to_observations.py --lc lc.txt --object-id my-asteroid \
  --period-hours 16.80059 --period-provenance "DAMIT model 1816" \
  --output observations.json

# download DAMIT asteroid 103, (5) Astraea; the object_id becomes asteroid_103
python examples/damit_to_observations.py --damit-id 103 --output observations.json
```

With `--damit-id` the period is that of the first DAMIT model of quality flag
3 or higher, as in the paper; `--model-id` chooses another model. The source
of the period, including the DAMIT model number, is written to
`known_period.provenance`. When the converter downloads the lightcurves, it
also keeps them as `<output>.lc.txt` next to the output file (choose another
name with `--save-lc`); this is the file an inversion program such as
`convexinv` reads.

**Finding the DAMIT id.** The DAMIT asteroid id is not the asteroid number:
(5) Astraea has DAMIT id 103, which is also the number in its DAMIT web address
(`https://damit.cuni.cz/projects/damit/asteroids/view/103`). For the 170
asteroids of the paper, look it up in the benchmark table:

```bash
grep -i phaethon repro/data/benchmark-asteroids.csv
# asteroid_2514,2514,3200,Phaethon,,2,k3-oof-fold-2
```

For any other asteroid in DAMIT, search DAMIT's table of asteroids; the first
column is the DAMIT id:

```bash
curl -s https://damit.cuni.cz/projects/damit/exports/table/asteroids | grep -i '"phaethon"'
# 2514,3200,"Phaethon",,,2020-02-13 08:54:32,2020-02-13 08:54:32
```

**Next step.** Choose the network set before you predict. If the asteroid is
one of the 170 of the paper, you must use the set of the run in which it is a
test asteroid (the last column of the benchmark table above); see
[choosing the network set](usage.md#3-choose-the-network-set).

## Pole labels for custom training

Training requires at least one reference axis per object. Supply unit Cartesian
vectors in the same ecliptic J2000 frame as geometry. If your label is ecliptic
longitude `lambda_deg` and latitude `beta_deg`, convert it as follows (angles in
radians for trigonometric functions):

```text
x = cos(beta) * cos(lambda)
y = cos(beta) * sin(lambda)
z = sin(beta)
```

A label and its antipode are equivalent to K3; include genuinely alternative
solutions when valid. Do not invent an antipode merely to double the sample.
The [training guide](training-your-data.md) gives the JSONL wrapper.
