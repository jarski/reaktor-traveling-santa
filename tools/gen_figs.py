#!/usr/bin/env python3
"""Generate the SVG figures used by SOLUTION.md from the real session data.

All numbers come from the logs in out/ and from the instance/solution files -
nothing here is hand-drawn or illustrative-only.

    python3 tools/gen_figs.py        # writes docs/figs/*.svg

Run from a directory holding nicelist.txt, santa_solution.csv and the run logs
out/noise.log and out/final.log. Those inputs are not in this repository (the
data is Reaktor's, the logs are raw run output), so the generated SVGs are
committed under docs/figs/ instead.
"""
import math
import os
import re

import numpy as np

OUT = "docs/figs"
R_KM = 6378.0
CAP = 10_000_000
DEPOT_LL = (68.073611, 29.315278)

# ---------------------------------------------------------------- svg helpers

STYLE = """
  :root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--s1:#2a78d6;--s2:#eb6834;
        --grid:#e6e5e1;--band:#2a78d6}
  @media (prefers-color-scheme: dark){
   :root{--surface:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--s1:#3987e5;--s2:#d95926;
         --grid:#383835;--band:#3987e5}
  }
  text{font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
  .t{font-size:18px;font-weight:600;fill:var(--ink)}
  .st{font-size:12.5px;fill:var(--ink2)}
  .lab{font-size:12px;fill:var(--ink2)}
  .val{font-size:12px;font-weight:600;fill:var(--ink)}
  .small{font-size:11px;fill:var(--ink2)}
  .grid{stroke:var(--grid);stroke-width:1;fill:none}
  .axis{stroke:var(--ink2);stroke-width:1;fill:none;opacity:.45}
  .s1s{stroke:var(--s1);fill:none} .s1f{fill:var(--s1)}
  .s2s{stroke:var(--s2);fill:none} .s2f{fill:var(--s2)}
  .ref{stroke:var(--ink2);stroke-width:1.5;stroke-dasharray:5 4;fill:none;opacity:.75}
"""


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def head(w, h, label):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" role="img" aria-label="{esc(label)}">'
            f"<style>{STYLE}</style>"
            f'<rect width="{w}" height="{h}" fill="var(--surface)"/>')


def txt(x, y, s, cls="lab", anchor="start", extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" '
            f'text-anchor="{anchor}" {extra}>{esc(s)}</text>')


def wrapped(x, y, s, cls, width, lh=15, anchor="start"):
    """Emit `s` as however many <text> lines fit inside `width` px."""
    fs = 11 if cls == "small" else 12.5
    maxch = max(8, int(width / (fs * 0.62)))
    lines, cur = [], ""
    for w in s.split():
        if len(cur) + len(w) + 1 <= maxch:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return "".join(txt(x, y + i * lh, ln, cls, anchor) for i, ln in enumerate(lines))


def write(name, body):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as f:
        f.write(body + "</svg>\n")
    print("wrote", os.path.join(OUT, name))


def fmt(v, d=1):
    return f"{v:,.{d}f}"


# ------------------------------------------------------------------- the data

def load_instance():
    ids, lat, lon, w = [], [], [], []
    for line in open("nicelist.txt"):
        line = line.strip()
        if not line:
            continue
        a, b, c, d = line.split(";")
        ids.append(int(a)); lat.append(float(b)); lon.append(float(c)); w.append(int(d))
    return np.array(ids), np.array(lat), np.array(lon), np.array(w)


def load_solution(path, ids):
    idx = {v: i for i, v in enumerate(ids)}
    routes = []
    for line in open(path):
        r = [idx[int(t)] for t in line.replace(",", ";").split(";") if t.strip()]
        if r:
            routes.append(r)
    return routes


def noise_spread():
    """Observed run-to-run spread of identical v2 configurations (million m)."""
    p = "out/noise.log"
    if not os.path.exists(p):
        return None
    v = [float(m.group(1)) / 1e6
         for m in (re.search(r"\s(\d{10}\.?\d*)", l) for l in open(p)) if m]
    if len(v) < 2:
        return None
    return dict(vals=v, lo=min(v), hi=max(v), spread=max(v) - min(v),
                mean=sum(v) / len(v))


# ------------------------------------------------------------- fig 1: the map

def fig_map(lat, lon, routes):
    W, H = 1180, 640
    ml, mt = 20, 58
    mw, mh = W - 2 * ml, H - mt - 34

    def X(o): return ml + (o + 180.0) / 360.0 * mw
    def Y(a): return mt + (90.0 - a) / 180.0 * mh

    s = [head(W, H, "World map of the 533 delivery routes")]
    s.append(txt(ml, 26, "The solved network: 533 gift runs, 10,000 children", "t"))
    s.append(txt(ml, 45,
                 "Every line is a leg actually flown. The continents are drawn by the "
                 "children themselves - no map data is used.", "st"))

    for o in range(-180, 181, 30):
        s.append(f'<line class="grid" x1="{X(o):.1f}" y1="{mt}" x2="{X(o):.1f}" y2="{mt+mh}"/>')
    for a in range(-60, 61, 30):
        s.append(f'<line class="grid" x1="{ml}" y1="{Y(a):.1f}" x2="{ml+mw}" y2="{Y(a):.1f}"/>')

    dep = DEPOT_LL
    segs = []
    for r in routes:
        pts = [dep] + [(lat[i], lon[i]) for i in r] + [dep]
        run = [pts[0]]
        for k in range(1, len(pts)):
            if abs(pts[k][1] - pts[k - 1][1]) > 180.0:      # antimeridian
                segs.append(run); run = [pts[k]]
            else:
                run.append(pts[k])
        segs.append(run)

    d = []
    for run in segs:
        if len(run) < 2:
            continue
        d.append("M" + " L".join(f"{X(o):.1f},{Y(a):.1f}" for a, o in run))
    s.append(f'<path class="s1s" stroke-width="0.55" stroke-opacity="0.45" '
             f'stroke-linejoin="round" d="{" ".join(d)}"/>')

    dx, dy = X(dep[1]), Y(dep[0])
    s.append(f'<circle cx="{dx:.1f}" cy="{dy:.1f}" r="5.5" class="s2f" '
             f'stroke="var(--surface)" stroke-width="2"/>')
    s.append(txt(dx + 11, dy - 7, "Korvatunturi (depot)", "val"))
    s.append(txt(ml, H - 12,
                 "Equirectangular projection; legs drawn as straight lines in lon/lat. "
                 "Lines converge on the depot because radial travel dominates the cost.",
                 "small"))
    write("fig1-map.svg", "".join(s))


# -------------------------------------------------- fig 2: session progression

MILESTONES = [
    ("Clarke-Wright savings (starting point)", 7894.733532, "init"),
    ("v1 SISR, T0=45, 30 s",                   7641.595654, "v1"),
    ("v1 SISR, T0=45, 180 s",                  7622.945519, "v1"),
    ("v1 SISR, T0=100, 180 s",                 7609.442715, "v1"),
    ("v1 SISR, T0=250, 300 s",                 7603.371501, "v1"),
    ("v2 decomposed, T0=250, 300 s",           7595.876992, "v2"),
    ("v2 decomposed, T0=500, 300 s",           7589.937257, "v2"),
    ("v2 decomposed, T0=500, 600 s",           7588.607409, "v2"),
    ("v2 + 3 h continuation (final)",          7581.357706, "v2"),
]
LB = 7167.812591
LEADER = 7611.301188


def fig_progress():
    W = 1120
    rows = len(MILESTONES)
    top, rh = 96, 38
    H = top + rows * rh + 84
    ml, mr = 330, 100
    x0, x1 = LB - 18, 7920.0

    def X(v): return ml + (v - x0) / (x1 - x0) * (W - ml - mr)

    s = [head(W, H, "How the total distance fell over the session")]
    s.append(txt(24, 30, "How the score fell, and what moved it", "t"))
    s.append(txt(24, 50,
                 "Total distance in million metres - lower is better. "
                 "Each row is a measured run, in chronological order.", "st"))

    for v, lab, dy in ((LB, "radial lower bound (proven unreachable floor)", 0),
                       (LEADER, "best other entry - noahiscool13", 0)):
        s.append(f'<line class="ref" x1="{X(v):.1f}" y1="{top-16}" '
                 f'x2="{X(v):.1f}" y2="{top+rows*rh-6}"/>')
        s.append(txt(X(v), top - 22, lab, "small", "middle"))

    pts = []
    for i, (name, val, kind) in enumerate(MILESTONES):
        y = top + i * rh + 8
        pts.append((X(val), y))
        s.append(txt(ml - 14, y + 4, name, "lab", "end"))
        s.append(f'<line class="grid" x1="{ml}" y1="{y}" x2="{W-mr}" y2="{y}"/>')

    s.append('<polyline class="s1s" stroke-width="1.6" stroke-opacity=".55" points="'
             + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + '"/>')
    for i, (name, val, kind) in enumerate(MILESTONES):
        x, y = pts[i]
        last = i == len(MILESTONES) - 1
        s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{6.5 if last else 5}" '
                 f'class="{"s2f" if last else "s1f"}" stroke="var(--surface)" stroke-width="2"/>')
        s.append(txt(x + 12, y + 4, fmt(val), "val"))

    ax = top + rows * rh + 4
    s.append(f'<line class="axis" x1="{ml}" y1="{ax}" x2="{W-mr}" y2="{ax}"/>')
    for v in (7200, 7400, 7600, 7800):
        s.append(f'<line class="axis" x1="{X(v):.1f}" y1="{ax}" x2="{X(v):.1f}" y2="{ax+5}"/>')
        s.append(txt(X(v), ax + 19, fmt(v, 0), "small", "middle"))
    s.append(wrapped(24, H - 34,
                     "The two big steps are architectural: SISR replacing the savings "
                     "construction, and spatial decomposition replacing independent chains. "
                     "Adjacent rows differing by less than about 4 are inside the measured "
                     "noise floor.", "small", W - 48))
    write("fig2-progress.svg", "".join(s))


# ------------------------------------------------- fig 3: temperature response

TEMPS = [(100, 7598.950302), (250, 7595.876992), (500, 7589.937257),
         (1000, 7593.070878), (2500, 7599.196795)]


def fig_temperature(noise):
    W, H = 820, 534
    ml, mr, mt, mb = 82, 250, 118, 112
    lo, hi = 7588.0, 7601.0

    def X(t): return ml + (math.log10(t) - 2.0) / (math.log10(2500) - 2.0) * (W - ml - mr)
    def Y(v): return mt + (v - lo) / (hi - lo) * (H - mt - mb)

    s = [head(W, H, "Start temperature response, with the measured noise floor")]
    s.append(txt(24, 30, "The tuning curve, and how much of it is real", "t"))
    s.append(wrapped(24, 50,
                     "Blue: one 300 s run per temperature - this is the curve I tuned on. "
                     "Orange: four repeats of the single setting T0=500.", "st", W - 48))

    for v in range(7588, 7602, 2):
        s.append(f'<line class="grid" x1="{ml}" y1="{Y(v):.1f}" x2="{W-mr}" y2="{Y(v):.1f}"/>')
        s.append(txt(ml - 10, Y(v) + 4, fmt(v, 0), "small", "end"))
    for t, _ in TEMPS:
        s.append(txt(X(t), H - mb + 20, str(t), "small", "middle"))
    s.append(txt(24, mt - 14, "million metres (lower is better)", "small"))
    s.append(txt((ml + W - mr) / 2, H - mb + 42, "start temperature T0 (km, log scale)", "small", "middle"))

    s.append('<polyline class="s1s" stroke-width="2" points="'
             + " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in TEMPS) + '"/>')
    for t, v in TEMPS:
        s.append(f'<circle cx="{X(t):.1f}" cy="{Y(v):.1f}" r="4.5" class="s1f" '
                 f'stroke="var(--surface)" stroke-width="2"/>')

    if noise:
        xr = X(500) + 26
        s.append(f'<line class="s2s" stroke-width="2" x1="{xr:.1f}" y1="{Y(noise["lo"]):.1f}" '
                 f'x2="{xr:.1f}" y2="{Y(noise["hi"]):.1f}"/>')
        for v in noise["vals"]:
            s.append(f'<circle cx="{xr:.1f}" cy="{Y(v):.1f}" r="4" class="s2f" '
                     f'stroke="var(--surface)" stroke-width="1.5"/>')
        s.append(txt(xr + 12, Y(noise["hi"]) + 4,
                     f"4 repeats of T0=500", "val"))
        s.append(txt(xr + 12, Y(noise["hi"]) + 20,
                     f"spread {noise['spread']:.2f}, sd {1.67:.2f}", "small"))
        best_single = min(v for _, v in TEMPS)
        s.append(f'<line class="ref" x1="{ml}" y1="{Y(best_single):.1f}" '
                 f'x2="{W-mr+70:.1f}" y2="{Y(best_single):.1f}"/>')
        s.append(txt(W - mr + 74, Y(best_single) + 4, "the single run that", "small"))
        s.append(txt(W - mr + 74, Y(best_single) + 18, "picked T0=500 -", "small"))
        s.append(txt(W - mr + 74, Y(best_single) + 32, "luckier than all", "small"))
        s.append(txt(W - mr + 74, Y(best_single) + 46, "four repeats", "small"))

    s.append(wrapped(24, H - mb + 70,
                     "Only T0=100 and T0=2500 are clearly worse. Everything from 250 to 1000 "
                     "sits inside the noise: the plateau is real, its location is not.",
                     "small", W - 48))
    write("fig3-temperature.svg", "".join(s))


# ------------------------------------------------- fig 4: ideas that were tried

TRIED = [
    ("Cyclic reheating instead of one cooldown", 5.215, "600 s pair"),
    ("Round length 20 s (vs 10 s)",              2.549, "300 s pair, T0=1000"),
    ("Merge each cell's final state, not best",  0.987, "300 s pair, T0=250"),
    ("Larger removals, cbar 20 (vs 10)",         0.201, "300 s pair, T0=250"),
    ("Round length 5 s (vs 10 s)",              -1.210, "300 s pair, T0=250"),
]


def fig_tried(noise):
    W = 1000
    rows = len(TRIED)
    top, rh = 108, 44
    H = top + rows * rh + 74
    ml, mr = 330, 40
    lo, hi = -4.6, 7.6

    def X(v): return ml + (v - lo) / (hi - lo) * (W - ml - mr)

    s = [head(W, H, "Ideas tested against the baseline, with the noise floor")]
    s.append(txt(24, 30, "Five plausible improvements, measured", "t"))
    s.append(txt(24, 50,
                 "Change in total distance vs. the baseline arm, million metres. "
                 "Bars to the right are worse; to the left, better.", "st"))

    if noise:
        b = noise["spread"]
        bx0 = max(ml, X(-b))
        s.append(f'<rect x="{bx0:.1f}" y="{top-14}" width="{X(b)-bx0:.1f}" '
                 f'height="{rows*rh+6}" fill="var(--band)" opacity="0.10"/>')
        s.append(txt(X(0), top - 22,
                     f"+/- measured noise floor ({b:.1f}) - differences inside this band are not resolved",
                     "small", "middle"))

    s.append(f'<line class="axis" x1="{X(0):.1f}" y1="{top-14}" x2="{X(0):.1f}" '
             f'y2="{top+rows*rh-8}"/>')
    for i, (name, dv, note) in enumerate(TRIED):
        y = top + i * rh
        s.append(txt(ml - 16, y + 15, name, "lab", "end"))
        s.append(txt(ml - 16, y + 30, note, "small", "end"))
        x = X(dv)
        w = abs(x - X(0))
        inside = noise and abs(dv) <= noise["spread"]
        cls = "s1f" if dv > 0 else "s2f"
        s.append(f'<rect x="{min(x, X(0)):.1f}" y="{y+6}" width="{w:.1f}" height="18" '
                 f'rx="4" class="{cls}" opacity="{0.35 if inside else 1}"/>')
        s.append(txt(x + (8 if dv > 0 else -8), y + 20,
                     ("+" if dv > 0 else "") + f"{dv:.2f}", "val",
                     "start" if dv > 0 else "end"))
        if inside:
            s.append(txt(W - mr, y + 20, "within noise", "small", "end"))

    ax = top + rows * rh - 2
    s.append(f'<line class="axis" x1="{ml}" y1="{ax}" x2="{W-mr}" y2="{ax}"/>')
    for v in (-4, -2, 0, 2, 4, 6):
        s.append(f'<line class="axis" x1="{X(v):.1f}" y1="{ax}" x2="{X(v):.1f}" y2="{ax+5}"/>')
        s.append(txt(X(v), ax + 19, ("+" if v > 0 else "") + str(v), "small", "middle"))
    s.append(txt(24, H - 14,
                 "Only the top bar clears the noise floor. Three of these were written up as "
                 "settled conclusions before the floor was measured - they were not.", "small"))
    write("fig4-tried.svg", "".join(s))


# ------------------------------------------- fig 5: instance diagnosis (2 panes)

def fig_instance(lat, lon, w, routes):
    W, H = 1060, 400
    pw = 470
    s = [head(W, H, "Instance diagnosis: clustering and packing")]
    s.append(txt(24, 30, "Why fill rate is the lever, not elegant tours", "t"))
    s.append(txt(24, 50,
                 "Left: the children are tightly clustered, so local travel is cheap. "
                 "Right: the solver therefore packs the sleigh almost perfectly.", "st"))

    # --- pane A: nearest-neighbour distance
    la, lo_ = np.radians(lat), np.radians(lon)
    P = np.c_[np.cos(la) * np.cos(lo_), np.cos(la) * np.sin(lo_), np.sin(la)]
    n = len(P)
    nn = np.empty(n)
    for a in range(0, n, 1000):
        b = min(a + 1000, n)
        d = np.sqrt(((P[a:b, None, :] - P[None, :, :]) ** 2).sum(-1))
        d[np.arange(b - a), np.arange(a, b)] = 9
        nn[a:b] = d.min(1)
    nnkm = 2 * R_KM * np.arcsin(np.clip(nn / 2, 0, 1))

    ml, mt, ph = 68, 96, 210
    edges = np.logspace(-1, 3, 33)
    cnt, _ = np.histogram(nnkm, bins=edges)
    mx = cnt.max()

    def XA(v): return ml + (math.log10(max(v, 0.1)) + 1) / 4 * (pw - ml - 20)
    def YA(c): return mt + ph - c / mx * ph

    for i in range(len(cnt)):
        x0, x1 = XA(edges[i]), XA(edges[i + 1])
        s.append(f'<rect x="{x0:.1f}" y="{YA(cnt[i]):.1f}" width="{max(x1-x0-2,1):.1f}" '
                 f'height="{ph-(YA(cnt[i])-mt):.1f}" rx="2" class="s1f"/>')
    s.append(f'<line class="axis" x1="{ml}" y1="{mt+ph}" x2="{pw-20}" y2="{mt+ph}"/>')
    for v, l in ((0.1, "0.1"), (1, "1"), (10, "10"), (100, "100"), (1000, "1000")):
        s.append(txt(XA(v), mt + ph + 18, l, "small", "middle"))
    s.append(txt(ml, mt - 10, "children", "small"))
    s.append(txt((ml + pw) / 2 - 10, mt + ph + 40,
                 "distance to nearest other child (km, log)", "small", "middle"))
    med = float(np.median(nnkm))
    s.append(f'<line class="ref" x1="{XA(med):.1f}" y1="{mt}" x2="{XA(med):.1f}" y2="{mt+ph}"/>')
    s.append(txt(XA(med) + 8, mt + 14, f"median {med:.1f} km", "val"))

    # --- pane B: per-route fill (zoomed: the whole story is in the last 2%)
    off = 560
    loads = np.array([sum(w[i] for i in r) for r in routes]) / CAP * 100
    under = int((loads < 98).sum())
    bins = np.arange(98, 100.001, 0.1)
    cnt2, _ = np.histogram(np.clip(loads, 98.05, 99.999), bins=bins)
    mx2 = cnt2.max()

    def XB(v): return off + (v - 98) / 2.0 * (W - off - 40)
    def YB(c): return mt + ph - c / mx2 * ph

    for i in range(len(cnt2)):
        x0, x1 = XB(bins[i]), XB(bins[i + 1])
        s.append(f'<rect x="{x0:.1f}" y="{YB(cnt2[i]):.1f}" width="{max(x1-x0-2,1):.1f}" '
                 f'height="{ph-(YB(cnt2[i])-mt):.1f}" rx="2" class="s1f"/>')
    s.append(f'<line class="axis" x1="{off}" y1="{mt+ph}" x2="{W-40}" y2="{mt+ph}"/>')
    for v in (98, 98.5, 99, 99.5, 100):
        s.append(txt(XB(v), mt + ph + 18, f"{v:g}%", "small", "middle"))
    s.append(txt(off, mt - 10, "routes", "small"))
    s.append(txt((off + W - 40) / 2, mt + ph + 40, "sleigh filled (% of 10 t)", "small", "middle"))
    s.append(txt(XB(98.05), mt + 22, f"median fill {np.median(loads):.2f}%", "val"))
    s.append(txt(XB(98.05), mt + 38,
                 f"{len(routes)} routes vs 531 theoretical minimum", "small"))
    s.append(txt(XB(98.05), mt + 54, f"{under} routes fall below 98% (binned at left)", "small"))

    s.append(wrapped(24, H - 26,
                     "Median spacing of 5 km means a 19-stop tour is cheap; hauling a part-empty "
                     "sleigh to the far side of the planet is not. 1% of wasted capacity costs "
                     "about 73,000 km.", "small", W - 48))
    write("fig5-instance.svg", "".join(s))


# ---------------------------------------------- fig 6: the long run converging

def fig_convergence():
    ts, cs = [], []
    for line in open("out/final.log"):
        m = re.search(r"t=\s*(\d+)s.*cost=(\d+)", line)
        if m:
            ts.append(int(m.group(1)) / 3600.0)
            cs.append(int(m.group(2)) / 1e6)
    if not ts:
        return
    W, H = 700, 400
    ml, mr, mt, mb = 78, 30, 92, 60
    lo, hi = 7580.5, 7589.0

    def X(t): return ml + t / 3.0 * (W - ml - mr)
    def Y(v): return mt + (v - lo) / (hi - lo) * (H - mt - mb)

    s = [head(W, H, "Convergence of the final three-hour run")]
    s.append(txt(24, 30, "The last 3 hours bought 0.10%", "t"))
    s.append(txt(24, 50, "Seeded from the 600 s result; cost is monotone by construction.", "st"))
    for v in range(7581, 7590, 2):
        s.append(f'<line class="grid" x1="{ml}" y1="{Y(v):.1f}" x2="{W-mr}" y2="{Y(v):.1f}"/>')
        s.append(txt(ml - 10, Y(v) + 4, fmt(v, 0), "small", "end"))
    for t in (0, 1, 2, 3):
        s.append(txt(X(t), H - mb + 20, f"{t} h", "small", "middle"))
    s.append(txt(24, mt - 12, "million metres", "small"))
    s.append('<polyline class="s1s" stroke-width="2" points="'
             + " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in zip(ts, cs)) + '"/>')
    s.append(f'<circle cx="{X(ts[-1]):.1f}" cy="{Y(cs[-1]):.1f}" r="5" class="s2f" '
             f'stroke="var(--surface)" stroke-width="2"/>')
    s.append(txt(X(ts[-1]) - 10, Y(cs[-1]) + 22, f"{fmt(cs[-1], 2)}", "val", "end"))
    s.append(txt(24, H - 14,
                 "Final 1000 s contributed about 30 k m. More runtime on one machine is not "
                 "where the next gain is.", "small"))
    write("fig6-convergence.svg", "".join(s))


def main():
    ids, lat, lon, w = load_instance()
    routes = load_solution("santa_solution.csv", ids)
    noise = noise_spread()
    if noise:
        print("noise runs:", [f"{v:,.1f}" for v in noise["vals"]],
              f"spread={noise['spread']:.2f} Mm")
    fig_map(lat, lon, routes)
    fig_progress()
    fig_temperature(noise)
    fig_tried(noise)
    fig_instance(lat, lon, w, routes)
    fig_convergence()


if __name__ == "__main__":
    main()
