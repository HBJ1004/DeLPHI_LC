# Running K3 on observations

Input is JSON with schema `delphi.k3-observations.v1`, a nonempty `object_id`,
`known_period` containing positive `hours` and nonempty `provenance`, and an
`epochs` array. Every epoch has an explicit `epoch_id` and `observations` array.
Each observation has:

| Field | Meaning |
|---|---|
| `time_jd` | Observation Julian date, consistently defined across epochs |
| `relative_brightness` | Positive linear relative flux, not magnitude |
| `sun_asteroid_ecliptic_j2000_au` | Three Cartesian components, asteroid to Sun, ecliptic J2000, AU |
| `observer_asteroid_ecliptic_j2000_au` | Three components, asteroid to observer, same frame and units |
| `measured_error` | Optional positive flux uncertainty in the same units |

Preserve original observing epochs; do not infer epoch boundaries from arbitrary
time gaps. The tokenizer requires at least two observations per epoch. Geometry
must be real, finite and nonzero; zero-filled placeholders are rejected. Use a
documented magnitude-to-flux conversion if starting from magnitudes. Do not mix
observation conventions or silently invent uncertainties. The package does not
query ephemerides or estimate the period for this inference path.

The Python API is:

```python
from lc_pipeline.k3.bundle import K3EnsemblePredictor
from lc_pipeline.k3.predict import read_observations
from pathlib import Path

object_id, period, epochs = read_observations(Path("observations.json"))
model = K3EnsemblePredictor("k3-oof-fold-0", device="cpu")
prediction = model.predict(epochs, known_period=period, object_id=object_id)
```

Output schema `delphi.k3-ensemble-prediction.v1` contains three `axes` with unit
`axis_xyz`, compatibility `score`, and originating `grid_index`. Scores are not
probabilities. The refinement can change peak order; do not treat the first axis
as a calibrated best pole. The output also records bundle/fold identity, period
provenance and archived fold containment radii. `risk_deg` is intentionally null.

For convex inversion, expand each axis to both signs, producing six initial pole
directions. Evaluate solutions by the inversion objective and physical checks,
not proximity to a reference pole. The supplied period is fixed in the published
benchmark. Accuracy for uncertain periods or unseen surveys is not established.

The public loader uses JSON and safetensors, validates all weight hashes and
checks fold membership against the shipped split. Download bundles and checksum
files through a trusted release channel. Training `.pt` checkpoints use Python
pickle and must never be loaded from an untrusted source.

For scientific details, cite Jo, Ishiguro and Lee, in prep.
