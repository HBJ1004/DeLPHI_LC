import glob
import hashlib
import json
from pathlib import Path

"""Lightcurve density of the census candidates, current DAMIT against the snapshot."""
FU = Path(__file__).resolve().parents[3]
OUT = FU / "data/damit-postsnapshot-check"
SNAP = FU.parent / "damit-20250610T000301Z/files"
donors = set()
for f in glob.glob(
    str(FU / "inputs/frozen-artifacts/k3-2408c563/synthetic/**/*.npz.manifest.json"), recursive=True
):
    donors |= set(json.load(open(f))["geometry_donor_ids"])


def parse(path):
    lines = Path(path).read_text().split("\n")
    n = int(lines[0])
    i = 1
    lcs = []
    for _ in range(n):
        npts, cal = map(int, lines[i].split()[:2])
        i += 1
        ts = [float(lines[i + k].split()[0]) for k in range(npts)]
        i += npts
        lcs.append((npts, cal, min(ts), max(ts)))
    return lcs


def summarize(lcs):
    dense = [lc for lc in lcs if lc[1] == 0 and lc[0] >= 10]
    starts = sorted(lc[2] for lc in dense)
    groups = 0
    prev = None
    for t in starts:
        if prev is None or t - prev > 30:
            groups += 1
        prev = t
    return dict(
        n_lc=len(lcs),
        n_obs=sum(lc[0] for lc in lcs),
        n_dense_lc=len(dense),
        n_dense_obs=sum(lc[0] for lc in dense),
        n_sparse_or_calibrated_lc=sum(1 for lc in lcs if lc[1] == 1),
        dense_30d_groups=groups,
        span_yr=round((max(lc[3] for lc in lcs) - min(lc[2] for lc in lcs)) / 365.25, 1),
    )


rows = {}
for f in sorted(glob.glob(str(OUT / "raw/lc/*.lc.txt"))):
    oid = Path(f).name.split(".")[0]
    cur = summarize(parse(f))
    snap = SNAP / oid / "lc.txt"
    s = summarize(parse(snap)) if snap.exists() else None
    same = (
        snap.exists()
        and hashlib.sha256(snap.read_bytes()).hexdigest()
        == hashlib.sha256(Path(f).read_bytes()).hexdigest()
    )
    rows[oid] = dict(
        current=cur, snapshot=s, lc_bytes_identical_to_snapshot=same, is_donor=oid in donors
    )
    print(oid, rows[oid])
json.dump(rows, open(OUT / "candidate_lc_density.json", "w"), indent=1)
