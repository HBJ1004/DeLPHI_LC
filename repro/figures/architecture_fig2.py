"""Draw the DeLPHI architecture (design B4), Fig. 2 of Paper I."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from pathlib import Path

INK = "#1f2630"
MUTED = "#5f6876"
LINE = "#b9c0ca"
FAINT = "#eef1f4"
ACC = "#1f5fae"
ACCL = "#e4eefa"
PEAK = "#c4472d"
DEFS = f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker>\n<radialGradient id="blob"><stop offset="0" stop-color="{ACC}" stop-opacity="0.95"/><stop offset="1" stop-color="{ACC}" stop-opacity="0"/></radialGradient></defs>'
PHOT = "#5b7fa8"
GEOM = "#c98b2b"
SUMM = "#6a9a6e"
STG = "#f6f8fa"
NOTE = 15
LAB = 15.5
PT = 18
CT = 17.5
VAR = 17
FONT = "font-family:'Nimbus Sans','Helvetica Neue',Helvetica,Arial,'Liberation Sans',sans-serif"
W, H = (1000, 950)
PY = 158


def _polyline(pts, **kw):
    a = " ".join((f"{x:.1f},{y:.1f}" for x, y in pts))
    attrs = " ".join((f'''{k.replace("_", "-")}="{v}"''' for k, v in kw.items()))
    return f'<polyline points="{a}" fill="none" {attrs}/>'


def _text(x, y, s, size=13, col=INK, anchor="middle", weight=400, italic=False):
    st = ' font-style="italic"' if italic else ""
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{col}" text-anchor="{anchor}" font-weight="{weight}"{st}>{s}</text>'


def _arrow(x1, y1, x2, y2, col=MUTED):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="1.6" marker-end="url(#ah)"/>'


def _lightcurves(cx, cy):
    out = []
    for dx, dy, ph in [(-30, -20, 0), (0, 0, 1.3), (30, 20, 2.4)]:
        pts = [
            (
                cx + dx - 22 + i * 2.2,
                cy + dy - 6 * math.sin(i * 0.55 + ph) - 2.5 * math.sin(i * 1.1 + ph),
            )
            for i in range(21)
        ]
        out.append(_polyline(pts, stroke=MUTED, stroke_width=1.3))
        for x, y in pts[::3]:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.7" fill="{INK}"/>')
    return "".join(out)


def _period(cx, cy):
    out = [
        f'<line x1="{cx - 50}" y1="{cy + 28}" x2="{cx + 50}" y2="{cy + 28}" stroke="{LINE}" stroke-width="1.2"/>'
    ]
    pts = []
    for i in range(101):
        x = cx - 50 + i
        f = i / 100
        y = 4 + 3 * abs(math.sin(i * 0.9)) + 2 * abs(math.sin(i * 2.3))
        y += 48 * math.exp(-(((f - 0.58) / 0.018) ** 2)) + 16 * math.exp(
            -(((f - 0.29) / 0.018) ** 2)
        )
        pts.append((x, cy + 28 - y))
    out.append(_polyline(pts, stroke=INK, stroke_width=1.3))
    out.append(
        f'<circle cx="{cx + 8}" cy="{cy + 28 - 52}" r="3" fill="none" stroke="{PEAK}" stroke-width="1.5"/>'
    )
    return "".join(out)


def _fold(cx, cy):
    out = [
        f'<rect x="{cx - 50}" y="{cy - 30}" width="100" height="60" fill="none" stroke="{LINE}" stroke-width="1"/>'
    ]
    for i in range(1, 16):
        out.append(
            f'<line x1="{cx - 50 + i * 100 / 16:.1f}" y1="{cy - 30}" x2="{cx - 50 + i * 100 / 16:.1f}" y2="{cy + 30}" stroke="{FAINT}" stroke-width="1"/>'
        )
    pts = [
        (
            cx - 50 + i * 2,
            cy
            - 15 * math.cos(i * 2 / 100 * 4 * math.pi) * 0.9
            - 4 * math.sin(i * 2 / 100 * 2 * math.pi),
        )
        for i in range(51)
    ]
    out.append(_polyline(pts, stroke=MUTED, stroke_width=1.3))
    for i in range(0, 51, 2):
        x, y = pts[i]
        out.append(
            f'<circle cx="{x:.1f}" cy="{y + (2.2 if i % 4 else -2.0):.1f}" r="1.6" fill="{INK}"/>'
        )
    out.append(
        f'<rect x="{cx - 50}" y="{cy - 30}" width="100" height="60" fill="none" stroke="{LINE}" stroke-width="1"/>'
    )
    return "".join(out)


def _sky(cx, cy):
    out = [
        f'<ellipse cx="{cx}" cy="{cy}" rx="52" ry="30" fill="{FAINT}" stroke="{LINE}" stroke-width="1"/>'
    ]
    for dx, dy, r in [(-22, -6, 15), (18, 8, 13), (30, -12, 10), (-5, 14, 9)]:
        out.append(
            f'<circle cx="{cx + dx}" cy="{cy + dy}" r="{r}" fill="url(#blob)" opacity="0.8"/>'
        )
    for dx, dy in [(-22, -6), (18, 8), (30, -12)]:
        out.append(
            f'<path d="M{cx + dx},{cy + dy - 4.5} L{cx + dx + 1.3},{cy + dy - 1.3} L{cx + dx + 4.5},{cy + dy - 1.3} L{cx + dx + 2},{cy + dy + 0.9} L{cx + dx + 2.9},{cy + dy + 4.3} L{cx + dx},{cy + dy + 2.3} L{cx + dx - 2.9},{cy + dy + 4.3} L{cx + dx - 2},{cy + dy + 0.9} L{cx + dx - 4.5},{cy + dy - 1.3} L{cx + dx - 1.3},{cy + dy - 1.3} z" fill="#ffffff" stroke="{PEAK}" stroke-width="1"/>'
        )
    return "".join(out)


def _sphere(cx, cy, r=32):
    out = [
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{LINE}" stroke-width="1.2"/>',
        f'<ellipse cx="{cx}" cy="{cy}" rx="{r}" ry="{r * 0.3}" fill="none" stroke="{LINE}" stroke-width="1"/>',
    ]
    for ang in (70, -25, 20):
        a = math.radians(ang)
        dx, dy = (r * 1.15 * math.cos(a), -r * 1.15 * math.sin(a) * 0.85)
        out.append(
            f'<line x1="{cx - dx:.1f}" y1="{cy - dy:.1f}" x2="{cx + dx:.1f}" y2="{cy + dy:.1f}" stroke="{INK}" stroke-width="1.4" stroke-dasharray="4 3"/>'
        )
        out.append(
            f'<circle cx="{cx + dx:.1f}" cy="{cy + dy:.1f}" r="3.2" fill="{PEAK}"/><circle cx="{cx - dx:.1f}" cy="{cy - dy:.1f}" r="3.2" fill="#ffffff" stroke="{PEAK}" stroke-width="1.3"/>'
        )
    return "".join(out)


def _scaled(fn, cx, cy, k):
    return f'<g transform="translate({cx},{cy}) scale({k}) translate({-cx},{-cy})">{fn(cx, cy)}</g>'


def _cells(x, y, n, col, w=9, h=12, gap=2):
    return "".join(
        (
            f'<rect x="{x + i * (w + gap)}" y="{y}" width="{w}" height="{h}" rx="1.5" fill="{col}" opacity="{0.45 + 0.55 * (i * 37 % 10) / 10:.2f}"/>'
            for i in range(n)
        )
    )


def _svg(w, h, body):
    return f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="{FONT}" role="img">{DEFS}{body}</svg>'


def _note(x, y, s, col=MUTED, anchor="start", size=NOTE):
    return _text(x, y, s, size, col, anchor)


def _stage(x, y, h, label, learned=False):
    return (
        f'<rect x="{x}" y="{y}" width="296" height="{h}" rx="10" fill="{(ACCL if learned else STG)}" stroke="{(ACC if learned else LINE)}" stroke-width="{(1.4 if learned else 1.1)}"/>'
        + _text(x + 12, y + 23, label, LAB, ACC if learned else INK, "start", 700)
    )


def _down(xc, y1, y2):
    return _arrow(xc, y1 + 1, xc, y2 - 1)


def _feature_cells(x, y, n, col, w=11, h=15, gap=3):
    return _cells(x, y, n, col, w, h, gap)


def _encoder(cx, cy):
    out = []
    for i in range(5):
        x = cx - 46 + i * 19
        out.append(
            f'<rect x="{x}" y="{cy - 30}" width="15" height="22" rx="3" fill="{ACCL}" stroke="{ACC}" stroke-width="1.2"/>'
        )
        out.append(
            f'<line x1="{x + 7.5}" y1="{cy - 7}" x2="{x + 7.5}" y2="{cy + 3}" stroke="{ACC}" stroke-width="1.1"/>'
        )
    out.append(f'<rect x="{cx - 48}" y="{cy + 4}" width="96" height="24" rx="4" fill="{ACC}"/>')
    return "".join(out)


def _chip(x, y, l1, l2, col):
    return (
        f'<rect x="{x}" y="{y}" width="272" height="44" rx="8" fill="#ffffff" stroke="{col}" stroke-width="1.3"/>'
        + _text(x + 136, y + 19, l1, NOTE, INK, weight=700)
        + _text(x + 136, y + 37, l2, NOTE, MUTED)
    )


def _overview() -> str:
    b = []
    cards = [
        ("Lightcurves", "Several epochs", _lightcurves, False),
        ("Period", "Fourier search", _period, False),
        ("Fold", "64 phase bins", _fold, False),
        ("Encoder", "Trained network", _encoder, True),
        ("Scorer", "6,144 trial axes", _sky, True),
        ("Starting poles", "3 axes, 6 poles", _sphere, False),
    ]
    cx = [82 + i * 167 for i in range(6)]
    for i, (n, sub, g, lr) in enumerate(cards):
        b.append(
            f'''<rect x="{cx[i] - 74}" y="8" width="148" height="134" rx="12" fill="{(ACCL if lr else "#ffffff")}" stroke="{(ACC if lr else LINE)}" stroke-width="{(1.6 if lr else 1.1)}"/>'''
        )
        b.append(_scaled(g, cx[i], 52, 0.7))
        b.append(_text(cx[i], 108, n, CT, ACC if lr else INK, weight=700))
        b.append(_text(cx[i], 129, sub, NOTE, MUTED))
        if i < 5:
            b.append(_arrow(cx[i] + 76, 75, cx[i + 1] - 77, 75))
    for ci, L in [(3, "A"), (4, "B"), (5, "C")]:
        b.append(
            f'<circle cx="{cx[ci] + 56}" cy="25" r="11" fill="{INK}"/>'
            + _text(cx[ci] + 56, 30.5, L, 14, "#ffffff", weight=700)
        )
    for px, L, ti, sub in [
        (8, "A", "Encoder", "Each lightcurve becomes one vector"),
        (342, "B", "Scorer", "How well one trial axis fits"),
        (676, "C", "Starting poles", "From score maps to six starts"),
    ]:
        b.append(
            f'<rect x="{px}" y="{PY}" width="316" height="{H - PY - 8}" rx="12" fill="#ffffff" stroke="{LINE}"/>'
        )
        b.append(
            _text(px + 14, PY + 28, L, PT, INK, "start", 700)
            + _text(px + 34, PY + 28, ti, PT, INK, "start", 700)
            + _text(px + 14, PY + 50, sub, NOTE, MUTED, "start")
        )
    return "".join(b)


def _encoder_panel() -> str:
    b = []
    X = 18
    y = PY + 64
    b.append(_stage(X, y, 108, "Folded lightcurve, 64 phase bins"))
    fx, fy = (X + 12, y + 36)
    b.append(
        f'<rect x="{fx}" y="{fy}" width="170" height="56" rx="3" fill="#ffffff" stroke="{LINE}"/>'
    )
    for i in range(1, 16):
        b.append(
            f'<line x1="{fx + i * 170 / 16:.1f}" y1="{fy}" x2="{fx + i * 170 / 16:.1f}" y2="{fy + 56}" stroke="{FAINT}"/>'
        )
    b.append(
        f'<rect x="{fx + 6 * 170 / 16:.1f}" y="{fy}" width="{170 / 16:.1f}" height="56" fill="{ACC}" opacity="0.22"/>'
    )
    pts = [
        (
            fx + i * 3.4,
            fy
            + 28
            - 17 * math.cos(i * 3.4 / 170 * 4 * math.pi)
            - 4 * math.sin(i * 3.4 / 170 * 2 * math.pi),
        )
        for i in range(51)
    ]
    b.append(_polyline(pts, stroke=MUTED, stroke_width=1.3))
    for i in range(0, 51, 2):
        b.append(
            f'<circle cx="{pts[i][0]:.1f}" cy="{pts[i][1] + (1.8 if i % 4 else -1.8):.1f}" r="1.8" fill="{INK}"/>'
        )
    b.append(_note(X + 194, y + 60, "One occupied") + _note(X + 194, y + 78, "phase bin"))
    b.append(_down(X + 80, y + 108, y + 126))
    y = PY + 190
    b.append(_stage(X, y, 96, "Numbers for each occupied bin"))
    b.append(_feature_cells(X + 12, y + 36, 6, PHOT) + _feature_cells(X + 128, y + 36, 8, GEOM))
    b.append(
        _note(X + 12, y + 76, "6 photometric", PHOT) + _note(X + 128, y + 76, "8 geometric", GEOM)
    )
    b.append(
        f'<path d="M{X + 244},{y + 44} L{X + 322},{y + 44}" stroke="{GEOM}" stroke-width="1.5" marker-end="url(#ah)"/>'
        + _text(X + 262, y + 36, "To B", NOTE, GEOM, "start", 700)
    )
    b.append(_down(X + 80, y + 96, y + 114))
    y = PY + 304
    b.append(_stage(X, y, 112, "Convolution over the bins", True))
    for r in range(6):
        for c in range(24):
            b.append(
                f'<rect x="{X + 12 + c * 7}" y="{y + 34 + r * 7}" width="6" height="6" rx="1" fill="{PHOT}" opacity="{0.25 + 0.6 * abs(math.sin(c * 0.5 + r)):.2f}"/>'
            )
    b.append(
        f'<rect x="{X + 12 + 5 * 7 - 2}" y="{y + 32}" width="{3 * 7 + 3}" height="{6 * 7 + 3}" fill="none" stroke="{ACC}" stroke-width="1.8" rx="3"/>'
    )
    b.append(_note(X + 12, y + 100, "Then the mean over occupied bins", ACC))
    b.append(_down(X + 80, y + 112, y + 130))
    y = PY + 434
    b.append(_stage(X, y, 100, "One vector per lightcurve"))
    b.append(
        _feature_cells(X + 12, y + 38, 10, ACC)
        + _text(X + 162, y + 51, "+", 18, MUTED)
        + _feature_cells(X + 178, y + 38, 5, SUMM)
    )
    b.append(
        _note(X + 12, y + 80, "128 from the bins", ACC)
        + _note(X + 178, y + 80, "22 per curve", SUMM)
    )
    b.append(_down(X + 80, y + 100, y + 118))
    y = PY + 552
    b.append(_stage(X, y, 132, "Transformer over the lightcurves", True))
    tokx = [X + 16, X + 66, X + 116]
    for a_, c_ in [(0, 1), (1, 2), (0, 2)]:
        xa, xc = (tokx[a_] + 15, tokx[c_] + 15)
        b.append(
            f'<path d="M{xa},{y + 58} Q{(xa + xc) / 2},{y + 40 - 6 * (c_ - a_)} {xc},{y + 58}" fill="none" stroke="{ACC}" stroke-width="1.2" opacity="0.8"/>'
        )
    for j, tx in enumerate(tokx):
        for k in range(5):
            b.append(
                f'<rect x="{tx}" y="{y + 62 + k * 12}" width="30" height="9" rx="2" fill="{ACC}" opacity="{0.4 + 0.12 * ((j + k) % 5):.2f}"/>'
            )
    b.append(
        _note(X + 164, y + 72, "Lightcurves", ACC)
        + _note(X + 164, y + 91, "inform each other", ACC)
        + _note(X + 164, y + 114, "2 layers, 4 heads")
    )
    b.append(_down(X + 80, y + 132, y + 150))
    y = PY + 702
    b.append(_stage(X, y, 80, "Encoded lightcurves"))
    for j, tx in enumerate([X + 12, X + 68, X + 124]):
        b.append(
            _cells(tx, y + 40, 3, INK, 8, 14, 2)
            + _text(tx + 33, y + 53, f"h{'₁₂₃'[j]}", VAR, INK, "start", italic=True)
        )
    b.append(_note(X + 184, y + 53, "128 numbers"))
    b.append(_note(X + 184, y + 71, "each"))
    return "".join(b)


def _scorer_panel() -> str:
    b = []
    X = 352
    y = PY + 64
    b.append(_stage(X, y, 176, "Trial axis and viewing geometry"))
    sx, sy = (X + 64, y + 108)
    b.append(
        f'<circle cx="{sx}" cy="{sy}" r="38" fill="#ffffff" stroke="{LINE}"/><ellipse cx="{sx}" cy="{sy}" rx="38" ry="11" fill="none" stroke="{LINE}"/><circle cx="{sx}" cy="{sy}" r="10" fill="#d7dbe0"/>'
    )
    b.append(
        f'<line x1="{sx - 18}" y1="{sy + 52}" x2="{sx + 18}" y2="{sy - 52}" stroke="{INK}" stroke-width="2" stroke-dasharray="5 3"/>'
        + _text(sx + 23, sy - 42, "p", VAR, INK, "start", 700, True)
    )
    b.append(
        f'<line x1="{sx}" y1="{sy}" x2="{sx - 40}" y2="{sy - 20}" stroke="{GEOM}" stroke-width="1.8" marker-end="url(#ah)"/>'
        + _text(sx - 44, sy - 26, "s", VAR, GEOM, "end", 700, True)
    )
    b.append(
        f'<line x1="{sx}" y1="{sy}" x2="{sx + 50}" y2="{sy + 20}" stroke="{GEOM}" stroke-width="1.8" marker-end="url(#ah)"/>'
        + _text(sx + 50, sy + 42, "o", VAR, GEOM, "middle", 700, True)
    )
    for k, (s_, c_) in enumerate(
        [
            ("Trial axis p", INK),
            ("Sun direction s", GEOM),
            ("Observer direction o", GEOM),
            ("Both at each bin", GEOM),
        ]
    ):
        b.append(_note(X + 138, y + 72 + k * 21, s_, c_))
    b.append(_down(X + 148, y + 176, y + 194))
    y = PY + 258
    b.append(_stage(X, y, 140, "Geometric terms for each bin"))
    b.append(_chip(X + 12, y + 36, "3 axis terms", "(p·s)², (p·o)², (p·s)(p·o)", INK))
    b.append(_chip(X + 12, y + 88, "4 shared terms", "Sun–observer angle, distances", GEOM))
    b.append(_down(X + 148, y + 140, y + 158))
    y = PY + 416
    b.append(_stage(X, y, 104, "Combined with each bin", True))
    b.append(
        _feature_cells(X + 56, y + 38, 6, PHOT)
        + _text(X + 148, y + 51, "×", 17, MUTED)
        + _feature_cells(X + 166, y + 38, 4, INK)
        + _text(X + 226, y + 51, "hₙ", VAR, INK, "start", italic=True)
    )
    b.append(_note(X + 148, y + 88, "Averaged over bins and lightcurves", anchor="middle"))
    b.append(_down(X + 148, y + 104, y + 122))
    y = PY + 538
    b.append(_stage(X, y, 72, "Score of axis p"))
    b.append(
        f'<rect x="{X + 30}" y="{y + 40}" width="236" height="14" rx="7" fill="#ffffff" stroke="{LINE}"/><rect x="{X + 30}" y="{y + 40}" width="176" height="14" rx="7" fill="{ACC}"/>'
    )
    b.append(_down(X + 148, y + 72, y + 90))
    y = PY + 628
    b.append(_stage(X, y, 154, "Repeated for all 6,144 axes"))
    mx, my2 = (X + 148, y + 94)
    b.append(f'<ellipse cx="{mx}" cy="{my2}" rx="124" ry="50" fill="#ffffff" stroke="{LINE}"/>')
    for gi in range(-16, 17):
        for gj in range(-6, 7):
            u, v = (gi / 16.5, gj / 6.5)
            if u * u + v * v > 0.95:
                continue
            val = (
                math.exp(-((u + 0.4) ** 2 + (v + 0.1) ** 2) / 0.03)
                + 0.85 * math.exp(-((u - 0.35) ** 2 + (v - 0.35) ** 2) / 0.025)
                + 0.7 * math.exp(-((u - 0.55) ** 2 + (v + 0.4) ** 2) / 0.02)
            )
            b.append(
                f'<circle cx="{mx + u * 118:.1f}" cy="{my2 + v * 46:.1f}" r="2.9" fill="{ACC}" opacity="{0.08 + 0.9 * min(val, 1):.2f}"/>'
            )
    b.append(
        f'<circle cx="{mx - 0.4 * 118:.1f}" cy="{my2 - 0.1 * 46:.1f}" r="6" fill="none" stroke="{INK}" stroke-width="1.6"/>'
    )
    return "".join(b)


def _starting_poles_panel() -> str:
    b = []
    X = 686
    y = PY + 64
    b.append(_stage(X, y, 130, "Five networks, one map each"))
    for k, sd in enumerate(["17", "42", "137", "777", "2027"]):
        ex, ey = (X + 30 + k * 59, y + 62)
        b.append(f'<ellipse cx="{ex}" cy="{ey}" rx="25" ry="15" fill="#ffffff" stroke="{LINE}"/>')
        for dx, dy, r in [(-8 + k % 2 * 2, -3, 7), (8, 4 - k % 3, 6), (12, -5, 5)]:
            b.append(
                f'<circle cx="{ex + dx}" cy="{ey + dy}" r="{r}" fill="url(#blob)" opacity="0.75"/>'
            )
    b.append(_note(X + 148, y + 104, "Seeds 17, 42, 137, 777, 2027", anchor="middle"))
    b.append(_down(X + 148, y + 130, y + 148) + _note(X + 160, y + 144, "Average"))
    y = PY + 212
    b.append(_stage(X, y, 220, "Averaged map, three highest peaks"))
    b.append(_scaled(_sky, X + 148, y + 110, 1.85))
    b.append(_note(X + 148, y + 202, "At least 15° apart", PEAK, "middle"))
    b.append(_down(X + 148, y + 220, y + 238))
    y = PY + 450
    b.append(_stage(X, y, 200, "Three axes, six starting poles"))
    b.append(_scaled(_sphere, X + 76, y + 108, 1.35))
    b.append(
        _note(X + 150, y + 92, "Each axis gives") + _note(X + 150, y + 112, "two starting poles")
    )
    b.append(
        f'<circle cx="{X + 156}" cy="{y + 138}" r="4" fill="{PEAK}"/><circle cx="{X + 170}" cy="{y + 138}" r="4" fill="#fff" stroke="{PEAK}" stroke-width="1.5"/>'
        + _note(X + 182, y + 143, "Both ends")
    )
    b.append(_down(X + 148, y + 200, y + 218))
    y = PY + 668
    b.append(_stage(X, y, 114, "Classical inversion"))
    b.append(
        _note(X + 12, y + 58, "Started from all six poles,")
        + _note(X + 12, y + 78, "the period held fixed")
    )
    return "".join(b)


def build_svg() -> str:
    """Return the 1000 by 950 SVG with the original B4 drawing order."""
    body = _overview() + _encoder_panel() + _scorer_panel() + _starting_poles_panel()
    return _svg(W, H, body)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True, help="Figure output directory")
    parser.add_argument("--pdf", action="store_true", help="Also export PDF using Inkscape")
    args = parser.parse_args(argv)
    inkscape = shutil.which("inkscape") if args.pdf else None
    if args.pdf and inkscape is None:
        parser.error("--pdf requires Inkscape; install it and put 'inkscape' on PATH")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = args.out_dir / "architecture.svg"
    svg_path.write_bytes(build_svg().encode("utf-8"))
    if args.pdf:
        subprocess.run(
            [
                inkscape,
                str(svg_path),
                "--export-type=pdf",
                f"--export-filename={args.out_dir / 'architecture.pdf'}",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
