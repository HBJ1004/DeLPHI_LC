# Prepare compatible data

DeLPHI expects one object per JSON file for prediction, or one object per line
in a JSONL file for training. The observation part has the same shape in both.
The code validates these fields rather than guessing units or coordinate frames.

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

The values are illustrative, not observations of a real object.

| Field | Required form |
|---|---|
| `object_id` | Nonempty stable identifier; use the same ID throughout a training project. |
| `known_period.hours` | Positive rotation period in hours. |
| `known_period.provenance` | Nonempty source description, such as a DOI, catalogue version, or analysis ID. |
| `epoch_id` | Unique nonempty identifier. Preserve your observing-session/group identity. |
| `time_jd` | Positive Julian date. Use one time scale consistently. |
| `relative_brightness` | Positive, dimensionless linear flux. |
| geometry vectors | Three finite nonzero values: **asteroid → Sun** and **asteroid → observer**, ecliptic J2000, AU. |
| `measured_error` | Optional positive uncertainty in the same relative-flux units. |

Do not substitute Earth-to-asteroid vectors, RA/Dec, heliocentric positions, or
zero vectors. Convert them first to the stated asteroid-centred vectors. If you
use an ephemeris service, save its source, time scale, frame, and conversion
formula with your data.

## From magnitudes to relative flux

If your measurement is magnitude `m`, choose a documented reference magnitude
`m0` for the same object/preprocessing stream and calculate:

```text
relative_brightness = 10 ** (-0.4 * (m - m0))
```

Apply quality filtering, filter handling, and phase/scattering corrections
consistently before this conversion. Do not mix filters or surveys without a
documented calibration decision. DeLPHI's tokenizer normalizes features within
an object; it does not make incompatible photometry physically equivalent.

## Choosing epochs

An epoch is a source-defined observing group, for example one calibrated night
or one lightcurve segment. Do not reconstruct epochs from arbitrary time gaps.
Each epoch must contain at least two observations; longer, well-sampled epochs
are generally more informative. Keep excluded epochs in your project log with
their exclusion reason.

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
