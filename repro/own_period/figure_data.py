"""Save the numerical Matplotlib artists next to each rendered PDF."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from repro import network_root as network


def clean(values):
    array = np.ma.asarray(values)
    if array.dtype.kind in "fc":
        return np.where(np.isfinite(array.filled(np.nan)), array.filled(np.nan), None).tolist()
    return array.tolist()


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def plotted_arrays(fig):
    return dict(
        size_inches=fig.get_size_inches().tolist(),
        axes=[
            dict(
                xlim=list(ax.get_xlim()),
                ylim=list(ax.get_ylim()),
                lines=[
                    dict(x=clean(line.get_xdata()), y=clean(line.get_ydata())) for line in ax.lines
                ],
                collections=[
                    dict(
                        offsets=clean(c.get_offsets()),
                        array=None if c.get_array() is None else clean(c.get_array()),
                        paths=[clean(p.vertices) for p in c.get_paths()],
                    )
                    for c in ax.collections
                ],
                images=[clean(im.get_array()) for im in ax.images],
                patches=[
                    dict(
                        path=clean(p.get_path().vertices),
                        transform=clean(p.get_patch_transform().get_matrix()),
                    )
                    for p in ax.patches
                ],
            )
            for ax in fig.axes
        ],
    )


@contextmanager
def capture_figures():
    from matplotlib.figure import Figure

    original = Figure.savefig

    def save(fig, filename, *args, **kwargs):
        if not isinstance(filename, (str, Path)):
            return original(fig, filename, *args, **kwargs)
        path = Path(filename)
        network.guard_output(path)
        if "paper-rev-20261021" in path.resolve().parts:
            raise ValueError("figure output must be outside the manuscript directory")
        result = original(fig, filename, *args, **kwargs)
        if path.suffix == ".pdf":
            sidecar = path.with_suffix(".arrays.json")
            sidecar.write_text(
                json.dumps(
                    json_safe(dict(arrays=plotted_arrays(fig), **network.json_metadata())),
                    indent=1,
                    allow_nan=False,
                )
                + "\n"
            )
        return result

    Figure.savefig = save
    try:
        yield
    finally:
        Figure.savefig = original
