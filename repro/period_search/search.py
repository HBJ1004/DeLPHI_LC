"""Photometric period searches. No reference period or object identity is accepted."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks

METHODS = ("legacy4", "lomb_scargle", "fourier_cpu", "fourier_gpu", "fourier_cv", "pdm", "entropy")


@dataclass
class Curve:
    time: np.ndarray  # light-time corrected JD, as supplied by DAMIT
    magnitude: np.ndarray


def read_curves(path: Path) -> list[Curve]:
    tokens = path.read_text().split()
    pos, curves = 1, []
    for _ in range(int(tokens[0])):
        n = int(tokens[pos]); pos += 2  # preserve relative/calibrated input as provided
        a = np.asarray(tokens[pos:pos + 8*n], float).reshape(n, 8); pos += 8*n
        a = a[np.isfinite(a[:, :2]).all(axis=1) & (a[:, 1] > 0)]
        a = a[np.argsort(a[:, 0])]
        if len(a) >= 2:
            curves.append(Curve(a[:, 0], -2.5*np.log10(a[:, 1])))
    if pos != len(tokens):
        raise ValueError("Unexpected trailing DAMIT fields")
    return curves


def observing_groups(curves):
    groups = []
    for c in sorted(curves, key=lambda c: c.time[0]):
        if not (0 < np.ptp(c.time) <= 30):
            continue
        if not groups or c.time[-1] - groups[-1][0].time[0] > 30:
            groups.append([])
        groups[-1].append(c)
    return [g for g in groups if sum(len(c.time) for c in g) >= 15]


def sparse_observing_groups(curves, window_days=30.0):
    """Build short windows from long survey lightcurves.

    Survey exports can place several years of measurements in one source
    lightcurve.  ``observing_groups`` intentionally rejects those records.
    This fallback cuts them into fixed time windows while preserving the
    source-lightcurve identity through separate magnitude offsets.  It is used
    only when the ordinary short-interval search has no usable group.
    """
    if not np.isfinite(window_days) or window_days <= 0:
        raise ValueError("window_days must be finite and positive")
    if not curves:
        return []
    origin = min(float(c.time[0]) for c in curves)
    pieces = []
    for source_index, curve in enumerate(curves):
        window_index = np.floor((curve.time - origin) / window_days).astype(int)
        for value in np.unique(window_index):
            selected = window_index == value
            if int(selected.sum()) >= 2:
                # Keep each source/window combination separate so its mean is
                # removed independently in the Fourier and PDM calculations.
                pieces.append((source_index, Curve(curve.time[selected], curve.magnitude[selected])))
    grouped = {}
    for source_index, curve in pieces:
        key = int(np.floor((float(curve.time[0]) - origin) / window_days))
        grouped.setdefault(key, []).append((source_index, curve))
    return [
        [curve for _, curve in sorted(group, key=lambda item: item[0])]
        for _, group in sorted(grouped.items())
        if sum(len(curve.time) for _, curve in group) >= 15
    ]


def grid(group, harmonics=6, resolution=1):
    span = max(c.time[-1] for c in group) - min(c.time[0] for c in group)
    step = min(.01, 1/(5*harmonics*span))/resolution
    freq = np.linspace(.12, 12., int(np.ceil(11.88/step))+1)
    return freq, float(freq[1]-freq[0])


def _arrays(group, h):
    origin = min(c.time[0] for c in group)
    t = np.concatenate([c.time-origin for c in group])
    y = np.concatenate([c.magnitude-c.magnitude.mean() for c in group])
    lengths = [len(c.time) for c in group]
    h = min(h, max(1, (len(t)-len(group)-2)//2))
    return t, y, lengths, h


def fourier_rss(group, frequencies, h=4, backend="cpu"):
    """Profile out an independent offset for each original lightcurve."""
    t, y, lengths, h = _arrays(group, h)
    frequencies = np.asarray(frequencies, float)
    if backend == "gpu":
        import torch
        device = "cuda"
        tt = torch.as_tensor(t, dtype=torch.float64, device=device)
        yy = torch.as_tensor(y, dtype=torch.float64, device=device)
        hh = torch.arange(1, h+1, dtype=torch.float64, device=device)
    values = []
    for start in range(0, len(frequencies), 64):
        f = frequencies[start:start+64]
        if backend == "gpu":
            ff = torch.as_tensor(f, dtype=torch.float64, device=device)
            phase = 2*np.pi*ff[:, None, None]*tt[None, :, None]*hh[None, None, :]
            x = torch.cat((phase.sin(), phase.cos()), dim=2)
            left = 0
            for n in lengths:
                x[:, left:left+n] -= x[:, left:left+n].mean(dim=1, keepdim=True)
                left += n
            gram = x.transpose(1, 2) @ x
            scale = gram.diagonal(dim1=1, dim2=2).sum(1).clamp(min=1.)
            gram += (1e-10*scale)[:, None, None]*torch.eye(2*h, dtype=torch.float64, device=device)
            beta = torch.linalg.solve(gram, (x.transpose(1, 2) @ yy).unsqueeze(-1)).squeeze(-1)
            rss = ((yy[None, :] - (x @ beta.unsqueeze(-1)).squeeze(-1))**2).sum(1)
            values.extend(rss.cpu().numpy())
        else:
            phase = 2*np.pi*f[:, None, None]*t[None, :, None]*np.arange(1, h+1)[None, None, :]
            x = np.concatenate((np.sin(phase), np.cos(phase)), axis=2)
            left = 0
            for n in lengths:
                x[:, left:left+n] -= x[:, left:left+n].mean(axis=1, keepdims=True)
                left += n
            gram = x.transpose(0, 2, 1) @ x
            scale = np.maximum(np.trace(gram, axis1=1, axis2=2), 1.)
            gram += (1e-10*scale)[:, None, None]*np.eye(2*h)
            rhs = x.transpose(0, 2, 1) @ y
            beta = np.linalg.solve(gram, rhs[..., None])[..., 0]
            rss = np.sum((y[None, :]-(x @ beta[..., None])[..., 0])**2, axis=1)
            values.extend(rss)
    return np.asarray(values), 2*h+len(group)


def binned_score(group, frequencies, method):
    t, y, _, _ = _arrays(group, 1)
    # Independent magnitude offsets removed before a shape-free phase comparison.
    if np.ptp(y) < 1e-12:
        return np.ones(len(frequencies))
    yy = np.clip(((y-y.min())/np.ptp(y)*5).astype(int), 0, 4)
    scores = []
    for f in frequencies:
        # Two shifted binnings reduce accidental placement at bin edges.
        vals = []
        for shift in (0., .05):
            bins = np.floor(((t*f+shift) % 1)*10).astype(int)
            count = np.bincount(bins, minlength=10)
            if method == "pdm":
                sums = np.bincount(bins, weights=y, minlength=10)
                rss = float(y@y - np.sum(sums*sums/np.maximum(count, 1)))
                dof = len(y)-np.count_nonzero(count)
                vals.append(rss/max(dof, 1))
            else:
                counts = np.bincount(5*bins+yy, minlength=50).reshape(10, 5)
                prob = counts / len(y)
                conditional = counts / np.maximum(count[:, None], 1)
                vals.append(float(-np.sum(prob*np.log(np.maximum(conditional, 1e-300)))))
        scores.append(np.mean(vals))
    return np.asarray(scores)


def profile(groups, frequencies, orders=(2, 3, 4, 6)):
    n = sum(len(c.time) for g in groups for c in g)
    all_rss = np.zeros((len(orders),len(frequencies)))
    parameters = np.zeros(len(orders))
    for g in groups:
        t,y,lengths,hmax=_arrays(g,max(orders))
        for oi,order in enumerate(orders):
            parameters[oi] += 2*min(order,hmax)+len(g)
        for left in range(0,len(frequencies),64):
            f=np.asarray(frequencies[left:left+64])
            phase=2*np.pi*f[:,None,None]*t[None,:,None]*np.arange(1,hmax+1)[None,None,:]
            x=np.concatenate((np.sin(phase),np.cos(phase)),axis=2)
            pos=0
            for size in lengths:
                x[:,pos:pos+size] -= x[:,pos:pos+size].mean(axis=1,keepdims=True)
                pos+=size
            gram=x.transpose(0,2,1)@x; rhs=x.transpose(0,2,1)@y
            for oi,order in enumerate(orders):
                h=min(order,hmax); indices=np.r_[np.arange(h),np.arange(hmax,hmax+h)]
                a=gram[:,indices][:,:,indices]; b=rhs[:,indices]
                ridge=1e-10*np.maximum(np.trace(a,axis1=1,axis2=2),1.)
                beta=np.linalg.solve(a+ridge[:,None,None]*np.eye(2*h),b[...,None])[...,0]
                rss=y@y-2*np.sum(beta*b,axis=1)+np.sum(beta*(a@beta[...,None])[...,0],axis=1)
                all_rss[oi,left:left+len(f)] += np.maximum(rss,0)
    all_rss /= n
    scores=n*np.log(np.maximum(all_rss,1e-30))+parameters[:,None]*np.log(n)
    chosen = np.argmin(scores, axis=0)
    return scores[chosen, np.arange(len(frequencies))], all_rss[chosen, np.arange(len(frequencies))]


def distinct_frequencies(values, spacing):
    retained=[]
    for f in sorted(set(values)):
        if not retained or f-retained[-1]>spacing:
            retained.append(f)
    return np.asarray(retained)


def refine_minimum(objective, frequency, step):
    """Expand to bracket a local minimum; a bracket edge is not a resolved peak."""
    radius=8*step
    for _ in range(5):
        lo,hi=max(.12,frequency-radius),min(12.,frequency+radius)
        fit=minimize_scalar(objective,bounds=(lo,hi),method='bounded',options={'xatol':1e-9})
        if min(fit.x-lo,hi-fit.x)>.05*(hi-lo):
            return float(fit.x),False
        radius*=2
    return float(fit.x),True


def odd_even_ratio(group, frequency):
    t,y,lengths,h=_arrays(group,4)
    phase=2*np.pi*frequency*t[:,None]*np.arange(1,h+1)
    x=np.concatenate((np.sin(phase),np.cos(phase)),axis=1)
    pos=0
    for size in lengths:
        x[pos:pos+size]-=x[pos:pos+size].mean(axis=0); pos+=size
    beta=np.linalg.lstsq(x,y,rcond=None)[0]
    power=beta[:h]**2+beta[h:]**2
    return float(power[::2].sum()/max(power[1::2].sum(),1e-20))


def cv_loss(train_groups, test_groups, frequency, h=4):
    total, count = 0., 0
    for train, test in zip(train_groups, test_groups):
        origin = min(c.time[0] for c in train)
        xs, ys, hold = [], [], []
        for c, d in zip(train, test):
            def design(t, reference_time=origin):
                phase = 2*np.pi*frequency*(t-reference_time)[:, None]*np.arange(1, h+1)
                return np.concatenate((np.sin(phase), np.cos(phase)), axis=1)
            x, z = design(c.time), design(d.time)
            xs.append(x-x.mean(0)); ys.append(c.magnitude-c.magnitude.mean())
            hold.append((z-x.mean(0), d.magnitude-c.magnitude.mean()))
        beta = np.linalg.lstsq(np.vstack(xs), np.concatenate(ys), rcond=None)[0]
        for x, y in hold:
            total += np.sum((y-x@beta)**2); count += len(y)
    return float(total/count)


def estimate(curves, method, resolution=1, timeout=180, sparse_fallback=False):
    started = time.monotonic()
    def check():
        if time.monotonic()-started > timeout:
            raise TimeoutError("photometric search budget exhausted")
    groups = observing_groups(curves)
    input_regime = "short_intervals"
    if not groups and sparse_fallback:
        groups = sparse_observing_groups(curves)
        input_regime = "sparse_survey_windows"
    if not groups:
        return {"status": "unresolved", "reason": "no short observing interval with 15 points", "ranked_periods": [], "candidates": []}
    if max(np.ptp(c.magnitude) for g in groups for c in g) < 1e-10:
        return {"status": "unresolved", "reason": "constant brightness", "ranked_periods": [], "candidates": []}
    if method == "legacy4":
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'e2e-period-20260921'))
        from fourier4 import estimate as old_estimate
        from period_pilot import stronger
        pairs = [(c.time, c.magnitude) for c in curves]
        initial = stronger(pairs, True)
        if not initial['success']:
            return {"status": "unresolved", "reason": initial['reason'], "ranked_periods": [], "candidates": []}
        periods = old_estimate(pairs, initial['ranked_periods'])
        step = min(grid(g)[1] for g in groups)
        return {
            "status": "ok",
            "ranked_periods": periods[:8],
            "candidates": [
                {"period": p, "score": float(i), "frequency_step": step}
                for i, p in enumerate(periods[:8])
            ],
        }
    train_groups, tests = groups, None
    if method == "fourier_cv":
        train_groups, tests = [], []
        for g in groups:
            pairs = [(Curve(c.time[:max(2, int(.8*len(c.time)))], c.magnitude[:max(2,int(.8*len(c.time)))]),
                      Curve(c.time[max(2,int(.8*len(c.time))):], c.magnitude[max(2,int(.8*len(c.time))):])) for c in g if len(c.time)>=5]
            if pairs and sum(len(a.time) for a,b in pairs)>=15:
                train_groups.append([a for a,b in pairs]); tests.append([b for a,b in pairs])
        if not train_groups:
            return {"status": "unresolved", "reason": "insufficient data for blocked validation", "ranked_periods": [], "candidates": []}
    orders = (1,) if method == "lomb_scargle" else (2, 3, 4, 6)
    candidate_freqs = []
    step = min(grid(g, resolution=resolution)[1] for g in train_groups)
    for g in train_groups:
        freq, _ = grid(g, resolution=resolution)
        search_orders = orders if method not in ("pdm", "entropy") else (1,)
        for h in search_orders:
            check()
            if method in ("pdm", "entropy"):
                score = binned_score(g, freq, method)
            else:
                score, _ = fourier_rss(g, freq, h, "gpu" if method == "fourier_gpu" else "cpu")
            peaks = np.unique(np.r_[0, find_peaks(-score)[0], len(freq)-1])
            selected = sorted(peaks, key=lambda i: score[i])[:8]
            for i in selected:
                for factor in (.5, 1, 2):
                    candidate_freqs.extend([freq[i]*factor+d for d in (0, -1, 1, -2, 2)])
    fs = [float(f) for f in candidate_freqs if .12 <= f <= 12]
    # Only merge candidates inside one fine search cell, never by a period percentage.
    fs = distinct_frequencies(fs,step*.25)
    check()
    if method in ("pdm", "entropy"):
        scores = sum(binned_score(g, fs, method) for g in train_groups)/len(train_groups)
    else:
        scores, _ = profile(train_groups, fs, orders)
    best = np.argsort(scores)[:24]
    refined = []
    for i in best:
        check()
        f = fs[i]
        edge=False
        if method not in ("pdm", "entropy"):
            f,edge=refine_minimum(lambda x: float(profile(train_groups,[x],orders)[0][0]),f,step)
        bic, mse = profile(train_groups, [f], orders)
        if tests is not None:
            score = cv_loss(train_groups, tests, f)
        elif method in ("pdm", "entropy"):
            score = float(np.mean([binned_score(g, [f], method)[0] for g in train_groups]))
        else:
            score = float(bic[0])
        refined.append({
            "period": 24/f,
            "score": score,
            "bic": float(bic[0]),
            "mse": float(mse[0]),
            "amplitude": float(np.median([
                np.ptp(c.magnitude) for g in train_groups for c in g
            ])),
            "frequency_step": step,
            "refinement_edge": edge,
        })
    candidates = []
    for c in sorted(refined, key=lambda c:(c['score'],c['period'])):
        if all(abs(24/c['period']-24/d['period'])>step for d in candidates):
            candidates.append(c)
        if len(candidates)==8: break
    largest=max(train_groups,key=lambda g:sum(len(c.time) for c in g))
    for c in candidates:
        c['odd_even_ratio']=odd_even_ratio(largest,24/c['period'])
    return {"status": "ok", "ranked_periods": [c['period'] for c in candidates], "candidates": candidates,
            "lightcurves_used": sum(map(len, train_groups)), "observing_intervals": len(train_groups),
            "candidate_count": len(fs), "ambiguity": "not_calibrated",
            "ranking": "blocked_holdout_four_harmonic" if tests is not None else method,
            "input_regime": input_regime}
