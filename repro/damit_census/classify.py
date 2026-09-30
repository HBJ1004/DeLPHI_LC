"""DAMIT census of 2026-09-30 for the Future work section of the paper.

Compares the current DAMIT tables (exported from
https://damit.cuni.cz/projects/damit/exports/table/{asteroids,asteroid_models,tumblers}
on 2026-09-30 and kept in data/damit-postsnapshot-check/raw) with the snapshot
damit-20250610T000301Z, and classifies every asteroid with a non-tumbler model
of quality 3 or higher as main sample, simulation donor, set aside from the
original list of 174, or new to DAMIT.  Asteroids are matched on the internal
DAMIT id, which is stable between the two versions.
"""
import csv, glob, json, io
from pathlib import Path
import numpy as np

FU = Path(__file__).resolve().parents[3]
ROOT = FU.parent
OUT = FU / "data/damit-postsnapshot-check"
SNAP = ROOT / "damit-20250610T000301Z/tables"

def read(path):
    txt = Path(path).read_text(encoding="utf-8-sig")
    return list(csv.DictReader(io.StringIO(txt)))

def q(v):
    try: return float(v)
    except (TypeError, ValueError): return None

snap_ast = {int(r["id"]): r for r in read(SNAP / "asteroids.csv")}
snap_mod = read(SNAP / "asteroid_models.csv")
snap_tum = read(SNAP / "tumblers.csv")
cur_ast = {int(r["id"]): r for r in read(OUT / "raw/asteroids.csv")}
cur_mod = read(OUT / "raw/asteroid_models.csv")
cur_tum = read(OUT / "raw/tumblers.csv")

def per_ast(models):
    d = {}
    for m in models:
        d.setdefault(int(m["asteroid_id"]), []).append(m)
    return d
snap_by, cur_by = per_ast(snap_mod), per_ast(cur_mod)
snap_tumblers = {int(r["asteroid_id"]) for r in snap_tum}
cur_tumblers = {int(r["asteroid_id"]) for r in cur_tum}

# identity consistency check on shared internal ids
shared = set(snap_ast) & set(cur_ast)
mismatch = [i for i in shared if (snap_ast[i]["number"] or "") != (cur_ast[i]["number"] or "")
            or (snap_ast[i]["name"] or "") != (cur_ast[i]["name"] or "")
            or (snap_ast[i]["designation"] or "") != (cur_ast[i]["designation"] or "")]
def key(r):  # MPC number if numbered, else designation/name
    return ("num", r["number"]) if r["number"] else ("des", (r["designation"] or r["name"]).strip())
snap_keys = {key(r): i for i, r in snap_ast.items()}

main = set(str(x) for x in np.load(FU / "inputs/frozen-artifacts/k3-definitive-7874092/evaluations/real-oof-ensemble.npz", allow_pickle=True)["object_ids"])
donors = set()
for f in glob.glob(str(FU / "inputs/frozen-artifacts/k3-2408c563/synthetic/**/*.npz.manifest.json"), recursive=True):
    donors |= set(json.load(open(f))["geometry_donor_ids"])
catalog = [json.loads(l) for l in open(FU / "inputs/training-run/repro/data/damit-20250610T000301Z/catalog.jsonl")]
requested174 = {c["object_id"] for c in catalog}
lowq = {o["object_id"].replace("damit:", "asteroid_") for o in json.load(open(FU / "source/experiments/lowq_damit_census_20260917/input-index.json"))["objects"]}

def qmax(models):
    vals = [q(m["quality_flag"]) for m in models]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None

cands = []
for i, a in cur_ast.items():
    ms = cur_by.get(i, [])
    q3 = [m for m in ms if (q(m["quality_flag"]) or 0) >= 3]
    if not q3 or i in cur_tumblers:
        continue
    oid = f"asteroid_{i}"
    snap_id = i if i in snap_ast else snap_keys.get(key(a))
    in_snap = snap_id is not None
    sms = snap_by.get(snap_id, []) if in_snap else []
    snap_q = qmax(sms)
    snap_mids = {m["id"] for m in sms}
    new_q3 = [m for m in q3 if m["id"] not in snap_mids]
    if oid in main: cat = "A_main170"
    elif oid in requested174: cat = "A2_requested174_quarantined"
    elif oid in donors: cat = "B_sim_donor"
    elif not in_snap: cat = "C_new_to_damit"
    elif not sms: cat = "D_in_snapshot_without_models"
    elif snap_q is not None and snap_q >= 3: cat = "E_snapshot_already_q3_not_used"
    else: cat = "F_snapshot_only_q<3_now_q3"
    cands.append(dict(object_id=oid, damit_id=i, snapshot_id=snap_id, number=a["number"], name=a["name"],
        designation=a["designation"], created=a["created"], modified=a["modified"], category=cat,
        in_lowq_test=f"asteroid_{snap_id}" in lowq if in_snap else False,
        snapshot_max_q=snap_q, snapshot_n_models=len(sms), current_n_q3=len(q3),
        n_new_q3_models=len(new_q3),
        new_q3_model_created=";".join(sorted({m["created"] for m in new_q3})),
        q3_versions=";".join(sorted({m["version"] for m in q3})),
        periods=";".join(sorted({m["period"] for m in q3}))))

from collections import Counter
print("snapshot asteroids", len(snap_ast), "current", len(cur_ast))
print("ids only in current", sorted(set(cur_ast) - set(snap_ast)))
print("ids only in snapshot", sorted(set(snap_ast) - set(cur_ast))[:20], len(set(snap_ast) - set(cur_ast)))
print("shared-id identity mismatches", len(mismatch), mismatch[:10])
print("snapshot models", len(snap_mod), "current", len(cur_mod))
smid = {m["id"] for m in snap_mod}; cmid = {m["id"] for m in cur_mod}
print("model ids new", sorted(cmid - smid, key=int), "removed", len(smid - cmid))
# changed quality flags on shared models
sq = {m["id"]: m["quality_flag"] for m in snap_mod}
chg = [(m["id"], m["asteroid_id"], sq[m["id"]], m["quality_flag"]) for m in cur_mod if m["id"] in sq and sq[m["id"]] != m["quality_flag"]]
print("models with changed quality_flag", len(chg), chg[:20])
print("main", len(main), "requested", len(requested174), "donors", len(donors), "main&donors", len(main & donors), "lowq", len(lowq))
print("current Q>=3 non-tumbler asteroids:", len(cands))
print(Counter(c["category"] for c in cands))
print("cats with new q3 models:", Counter(c["category"] for c in cands if c["n_new_q3_models"]))
with open(OUT / "q3_current_classification.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(cands[0])); w.writeheader(); w.writerows(cands)
for c in cands:
    if c["category"] not in ("A_main170", "B_sim_donor", "A2_requested174_quarantined"):
        print(c)
