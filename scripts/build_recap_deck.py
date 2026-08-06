#!/usr/bin/env python3
"""Build the visual-latents recap deck from docs/RECAP_2026_08_06.md.

Markdown is canonical; this HTML is disposable — regenerate, never hand-edit.
Every number in the deck is lifted verbatim from the markdown; the figures
re-shape those numbers, they never recompute them. Each slide links to a
faithful conversion of the source section it was distilled from.
"""
import hashlib
import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # repo root (this file lives in scripts/)
SRC = ROOT / "docs" / "RECAP_2026_08_06.md"
OUT = ROOT / "docs" / "RECAP_2026_08_06.html"

# ---------------------------------------------------------------- palette
# dataviz reference palette, dark column (validated: node validate_palette.js
# "#3987e5,#d95926,#199e70" --mode dark --pairs all -> ALL CHECKS PASS)
C = dict(
    page="#0d0d0d", surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7",
    muted="#898781", grid="#2c2c2a", axis="#383835",
    s1="#3987e5", s2="#d95926", s3="#199e70",
    good="#0ca30c", warn="#fab219", serious="#ec835a", crit="#d03b3b",
)

# ------------------------------------------------------- markdown -> html
def md_inline(t: str) -> str:
    t = html.escape(t, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![\w*])\*([^*]+)\*(?![\w*])", r"<em>\1</em>", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
    t = t.replace(r"\|", "|").replace(r"\*", "*")
    return t


def md_to_html(md: str) -> str:
    """Small faithful converter: headings, tables, lists, fences, rules, paras."""
    lines = md.split("\n")
    out, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            j = i + 1
            buf = []
            while j < len(lines) and not lines[j].startswith("```"):
                buf.append(lines[j]); j += 1
            out.append("<pre><code>" + html.escape("\n".join(buf)) + "</code></pre>")
            i = j + 1; continue
        if re.match(r"^\s*---+\s*$", ln):
            out.append("<hr>"); i += 1; continue
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{md_inline(m.group(2))}</h{lvl}>"); i += 1; continue
        if ln.lstrip().startswith("|") and i + 1 < len(lines) and re.match(
                r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            def cells(row):
                return [c.strip() for c in row.strip().strip("|").split("|")]
            head = cells(ln)
            aligns = []
            for spec in cells(lines[i + 1]):
                if spec.endswith(":") and spec.startswith(":"):
                    aligns.append("center")
                elif spec.endswith(":"):
                    aligns.append("right")
                else:
                    aligns.append("left")
            body, j = [], i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                body.append(cells(lines[j])); j += 1
            th = "".join(f'<th style="text-align:{a}">{md_inline(c)}</th>'
                         for c, a in zip(head, aligns))
            trs = []
            for row in body:
                tds = "".join(f'<td style="text-align:{a}">{md_inline(c)}</td>'
                              for c, a in zip(row, aligns + ["left"] * 8))
                trs.append(f"<tr>{tds}</tr>")
            out.append(f"<table><thead><tr>{th}</tr></thead><tbody>"
                       + "".join(trs) + "</tbody></table>")
            i = j; continue
        if re.match(r"^\s*(\d+\.|[-*])\s+", ln):
            ordered = bool(re.match(r"^\s*\d+\.", ln))
            items, j = [], i
            while j < len(lines) and (re.match(r"^\s*(\d+\.|[-*])\s+", lines[j])
                                      or (lines[j].startswith("   ") and lines[j].strip())):
                if re.match(r"^\s*(\d+\.|[-*])\s+", lines[j]):
                    items.append(re.sub(r"^\s*(\d+\.|[-*])\s+", "", lines[j]))
                else:
                    items[-1] += " " + lines[j].strip()
                j += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{md_inline(x)}</li>" for x in items)
                       + f"</{tag}>")
            i = j; continue
        if not ln.strip():
            i += 1; continue
        para, j = [ln], i + 1
        while (j < len(lines) and lines[j].strip()
               and not lines[j].lstrip().startswith(("|", "#", "```"))
               and not re.match(r"^\s*([-*]|\d+\.)\s+", lines[j])
               and not re.match(r"^\s*---+\s*$", lines[j])):
            para.append(lines[j]); j += 1
        out.append("<p>" + md_inline(" ".join(x.strip() for x in para)) + "</p>")
        i = j
    return "\n".join(out)


def split_sections(md: str):
    """Return {number: (title, raw_markdown_of_that_section)}."""
    parts = re.split(r"^## ", md, flags=re.M)[1:]
    secs = {}
    for p in parts:
        title_line = p.split("\n", 1)[0].strip()
        m = re.match(r"^(\d+)\.\s*(.*)$", title_line)
        if not m:
            continue
        secs[int(m.group(1))] = (m.group(2), "## " + p.rstrip())
    return secs


# ------------------------------------------------------------- svg atoms
def svg_open(w, h, extra=""):
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" '
            f'preserveAspectRatio="xMidYMid meet" {extra}>')


def txt(x, y, s, fill=C["ink2"], size=13, anchor="start", weight="400", extra=""):
    return (f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" '
            f'text-anchor="{anchor}" font-weight="{weight}" {extra}>{html.escape(s)}</text>')


def bar(x, y, w, h, fill, r=4, title=None, opacity=1.0):
    """Rounded data-end bar (rounding on the value end only via path)."""
    r = min(r, h / 2, max(w, 0.1))
    if w <= 0:
        return ""
    d = (f"M{x},{y} H{x+w-r} a{r},{r} 0 0 1 {r},{r} V{y+h-r} "
         f"a{r},{r} 0 0 1 -{r},{r} H{x} Z")
    t = f"<title>{html.escape(title)}</title>" if title else ""
    return f'<path d="{d}" fill="{fill}" opacity="{opacity}">{t}</path>'


def hgrid(x0, x1, ys, stroke=C["grid"]):
    return "".join(f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="{stroke}" '
                   f'stroke-width="1"/>' for y in ys)


# ------------------------------------------------------------- figures
def fig_tracks():
    """Two parallel upstream tracks on a linear date axis, and where each stopped."""
    W, H = 720, 230
    x0, x1 = 106, 660          # 2026-05-01 .. 2026-08-06
    span = 97.0                # days
    def X(day):                # day = days since 2026-05-01
        return x0 + (x1 - x0) * day / span
    s = [svg_open(W, H)]
    for name, day in (("May", 0), ("Jun", 31), ("Jul", 61), ("Aug", 92)):
        s.append(f'<line x1="{X(day)}" y1="34" x2="{X(day)}" y2="206" stroke="{C["grid"]}" stroke-width="1"/>')
        s.append(txt(X(day), 26, name, C["muted"], 12, "middle"))
    rows = [
        ("Monet track", 62, C["s1"], 1, 35, "complete", C["good"],
         "probe → pivot A → faithful repro → benchmarks"),
        ("LVR track", 122, C["s2"], 35, 46, "blocked", C["crit"],
         "vendor → env → 142 GB data → Stage-1 SFT"),
    ]
    for label, y, col, d0, d1, state, scol, sub in rows:
        s.append(txt(x0 - 12, y + 11, label, C["ink"], 13.5, "end", "600"))
        s.append(f'<rect x="{X(d0)}" y="{y}" width="{X(d1)-X(d0)}" height="14" rx="7" fill="{col}">'
                 f'<title>{html.escape(label)}: {html.escape(sub)}</title></rect>')
        s.append(f'<circle cx="{X(d1)}" cy="{y+7}" r="7" fill="{scol}" stroke="{C["surface"]}" stroke-width="2"/>')
        s.append(txt(X(d1) + 16, y + 12, state, scol, 13, "start", "700"))
        s.append(txt(X(d0), y + 32, sub, C["muted"], 11.5))   # under the bar: never clips
    # the idle span gets its own row so it overlaps nothing
    s.append(txt(x0 - 12, 191, "idle", C["muted"], 13.5, "end", "600"))
    s.append(f'<rect x="{X(46)}" y="180" width="{X(97)-X(46)}" height="14" rx="7" fill="{C["axis"]}"/>')
    s.append(txt(X(46) + 12, 172, "last commit 16 Jun", C["muted"], 11.5))
    s.append(f'<line x1="{X(97)}" y1="34" x2="{X(97)}" y2="206" stroke="{C["muted"]}" stroke-width="1"/>')
    s.append(txt(X(97), 222, "today", C["muted"], 11.5, "middle"))
    s.append("</svg>")
    return "".join(s)


def fig_cos_line(items, caption_left="distributed", caption_right="collapsed"):
    """Dot plot on the mean off-diagonal cosine axis, 0 -> 1."""
    W, H = 720, 64 + 38 * len(items)
    x0, x1 = 250, 620          # inset leaves room for the longest row label
    def X(v):
        return x0 + (x1 - x0) * v
    s = [svg_open(W, H)]
    s.append(f'<line x1="{x0}" y1="34" x2="{x1}" y2="34" stroke="{C["axis"]}" stroke-width="1"/>')
    for v in [0, 0.25, 0.5, 0.75, 1.0]:
        s.append(f'<line x1="{X(v)}" y1="30" x2="{X(v)}" y2="38" stroke="{C["axis"]}" stroke-width="1"/>')
        s.append(txt(X(v), 24, f"{v:.2f}", C["muted"], 11, "middle"))
    s.append(txt(x0, 12, caption_left, C["good"], 12))
    s.append(txt(x1, 12, caption_right, C["crit"], 12, "middle"))
    for k, (label, val, disp, col, note) in enumerate(items):
        y = 66 + 38 * k
        # val may be a scalar or a (lo, hi) range — the source sometimes reports a range,
        # and a range is drawn as a band rather than collapsed to an invented midpoint.
        rng = isinstance(val, tuple)
        lo, hi = val if rng else (val, val)
        mid = (lo + hi) / 2
        s.append(f'<line x1="{X(mid)}" y1="38" x2="{X(mid)}" y2="{y}" stroke="{C["grid"]}" stroke-width="1"/>')
        if rng:
            s.append(f'<rect x="{X(lo)-7}" y="{y-7}" width="{X(hi)-X(lo)+14}" height="14" rx="7" '
                     f'fill="{col}" opacity="0.35"/>')
            for e in (lo, hi):
                s.append(f'<circle cx="{X(e)}" cy="{y}" r="7" fill="{col}" stroke="{C["surface"]}" '
                         f'stroke-width="2"><title>{html.escape(label)}: cos {e}</title></circle>')
        else:
            s.append(f'<circle cx="{X(mid)}" cy="{y}" r="7" fill="{col}" stroke="{C["surface"]}" '
                     f'stroke-width="2"><title>{html.escape(label)}: mean off-diag cos {disp}</title></circle>')
        s.append(txt(X(hi) + 14, y + 5, disp, C["ink"], 14, "start", "700"))
        # label + note both live in the left column, so nothing can run off the right edge
        s.append(txt(x0 - 16, y + 1, label, C["ink2"], 13, "end"))
        if note:
            s.append(txt(x0 - 16, y + 17, note, C["muted"], 11.5, "end"))
    s.append("</svg>")
    return "".join(s)


def fig_pivot():
    """Two small multiples over the same variant order: cosine, then utility."""
    variants = [
        ("C1", "cos hinge", 0.725, None),
        ("C2", "VICReg λ=1", 0.441, None),
        ("D1", "+ sum-MSE", 0.987, None),
        ("D2", "VICReg λ=2", 0.341, None),
        ("E", "D2 ×2 steps", 0.389, 0.088),
        ("F", "D2 @ K=4", 0.311, 0.132),
        ("G", "F ×2 steps", 0.372, 0.222),
    ]
    W, H = 720, 316
    s = [svg_open(W, H)]
    lx, lw = 128, 220          # left panel (cosine)
    rx, rw = 430, 190          # right panel (utility)
    s.append(txt(lx, 18, "mean off-diagonal cosine", C["ink2"], 13, "start", "600"))
    s.append(txt(lx, 34, "lower = better", C["muted"], 11))
    s.append(txt(rx, 18, "qwen_base transfer utility (nat)", C["ink2"], 13, "start", "600"))
    s.append(txt(rx, 34, "higher = better", C["muted"], 11))
    # Monet stage-2 reference line on the cosine panel
    ref = lx + lw * 0.375
    s.append(f'<line x1="{ref}" y1="46" x2="{ref}" y2="292" stroke="{C["s3"]}" stroke-width="1"/>')
    s.append(txt(ref + 6, 58, "Monet stage 2 = 0.375", C["s3"], 11))
    for k, (code, name, cos, util) in enumerate(variants):
        y = 68 + 32 * k
        best = code == "G"
        op = 1.0 if best else 0.45
        s.append(txt(lx - 106, y + 12, code, C["ink"] if best else C["muted"], 12.5, "start",
                     "700" if best else "600"))
        s.append(txt(lx - 12, y + 12, name, C["ink"] if best else C["ink2"], 12.5, "end",
                     "700" if best else "400"))
        s.append(bar(lx, y, lw * cos, 15, C["s1"], title=f"{name}: cos {cos}", opacity=op))
        s.append(txt(lx + lw * cos + 8, y + 12, f"{cos:.3f}",
                     C["ink"] if best else C["muted"], 12, "start", "700" if best else "400"))
        if util is not None:
            s.append(bar(rx, y, rw * (util / 0.25), 15, C["s3"], title=f"{name}: utility +{util}", opacity=op))
            s.append(txt(rx + rw * (util / 0.25) + 8, y + 12, f"+{util:.3f}",
                         C["ink"] if best else C["muted"], 12, "start", "700" if best else "400"))
        else:
            s.append(txt(rx, y + 12, "not measured", C["muted"], 11.5))
    s.append(f'<line x1="{rx}" y1="62" x2="{rx}" y2="292" stroke="{C["axis"]}" stroke-width="1"/>')
    s.append(f'<line x1="{lx}" y1="62" x2="{lx}" y2="292" stroke="{C["axis"]}" stroke-width="1"/>')
    s.append(txt(rx, 308, "Monet stage 2 sits at +2.19 — off this scale, ~10× G", C["muted"], 11.5))
    s.append("</svg>")
    return "".join(s)


def fig_dissociation():
    """Before/after on two panels for stage 3 — one series each, so nothing overlaps.

    rl moves in lockstep (0.866 -> 0.323, utility -0.117 -> -0.137); plotting it too
    would put two coincident lines on top of each other, so it is stated in the caption
    instead of drawn.
    """
    W, H = 720, 258
    top, bot = 66, 196
    s = [svg_open(W, H)]
    panels = [
        (96, 216, "mean off-diagonal cosine", 0.0, 1.0, 0.864, 0.329, "{:.3f}",
         [0, 0.25, 0.5, 0.75, 1.0], "{:.2f}", C["crit"], "moves 0.535"),
        (452, 216, "qwen_base utility (nat)", -0.25, 0.0, -0.115, -0.117, "{:+.3f}",
         [-0.25, -0.20, -0.15, -0.10, -0.05, 0.0], "{:+.2f}", C["good"], "flat: 0.002"),
    ]
    for px, pw, title, lo, hi, a, b, fmt, ticks, tfmt, dcol, delta in panels:
        def Y(v):
            return bot - (bot - top) * (v - lo) / (hi - lo)
        s.append(txt(px, 30, title, C["ink2"], 13, "start", "600"))
        s.append(hgrid(px, px + pw, [Y(t) for t in ticks]))
        for t in ticks:
            s.append(txt(px - 10, Y(t) + 4, tfmt.format(t), C["muted"], 11, "end"))
        s.append(f'<line x1="{px}" y1="{top}" x2="{px}" y2="{bot}" stroke="{C["axis"]}" stroke-width="1"/>')
        xa, xb = px + 44, px + pw - 40
        for lab, x in (("unmodified", xa), ("+ vicreg_project", xb)):
            s.append(txt(x, 218, lab, C["muted"], 11.5, "middle"))
        s.append(f'<line x1="{xa}" y1="{Y(a)}" x2="{xb}" y2="{Y(b)}" stroke="{C["s1"]}" stroke-width="2"/>')
        for xx, vv in ((xa, a), (xb, b)):
            s.append(f'<circle cx="{xx}" cy="{Y(vv)}" r="5.5" fill="{C["s1"]}" stroke="{C["surface"]}" '
                     f'stroke-width="2"><title>stage 3: {fmt.format(vv)}</title></circle>')
        s.append(txt(xa, Y(a) - 14, fmt.format(a), C["ink"], 13, "middle", "700"))
        s.append(txt(xb, Y(b) - 14, fmt.format(b), C["ink"], 13, "middle", "700"))
        s.append(txt(px + pw * 0.42, Y((a + b) / 2) + (34 if a > b else -24), delta, dcol, 12.5,
                     "middle", "700"))
    s.append(txt(96, 246, "stage 3, n=500. rl moves in lockstep: cos 0.866 → 0.323, "
                          "utility −0.117 → −0.137.", C["muted"], 11.5))
    s.append("</svg>")
    return "".join(s)


def fig_ksweep():
    """Two series over test-time latent budget K. Legend + direct endpoint labels."""
    W, H = 720, 290
    x0, x1, top, bot = 100, 560, 50, 220
    ks = [8, 10, 12, 16]
    lo, hi = 74.0, 82.0
    def X(k):
        return x0 + (x1 - x0) * (k - 8) / 8
    def Y(v):
        return bot - (bot - top) * (v - lo) / (hi - lo)
    ours = [77.49, 77.49, 75.92, 77.49]
    rel = [77.49, 78.53, 77.49, 80.63]
    s = [svg_open(W, H)]
    for v in [74, 76, 78, 80, 82]:
        s.append(f'<line x1="{x0}" y1="{Y(v)}" x2="{x1}" y2="{Y(v)}" stroke="{C["grid"]}" stroke-width="1"/>')
        s.append(txt(x0 - 12, Y(v) + 4, f"{v}", C["muted"], 11.5, "end"))
    s.append(txt(x0 - 12, 32, "V* %", C["muted"], 11.5, "end"))
    for k in ks:
        s.append(txt(X(k), 244, f"K={k}", C["muted"], 12, "middle"))
    s.append(f'<line x1="{x0}" y1="{bot}" x2="{x1}" y2="{bot}" stroke="{C["axis"]}" stroke-width="1"/>')
    for vals, col, name in ((ours, C["s1"], "ours ep3  (cos 0.40)"), (rel, C["s2"], "released SFT  (cos 0.87)")):
        pts = " ".join(f"{X(k)},{Y(v)}" for k, v in zip(ks, vals))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2" '
                 f'stroke-linejoin="round"/>')
        for k, v in zip(ks, vals):
            s.append(f'<circle cx="{X(k)}" cy="{Y(v)}" r="5" fill="{col}" stroke="{C["surface"]}" '
                     f'stroke-width="2"><title>{html.escape(name)} @ K={k}: {v}</title></circle>')
        s.append(txt(X(16) + 14, Y(vals[-1]) + 4, f"{vals[-1]}", col, 13.5, "start", "700"))
    s.append(txt(X(16) + 14, Y(rel[-1]) - 14, "+3.1 with K", C["s2"], 11.5))
    s.append(txt(X(16) + 14, Y(ours[-1]) + 20, "K-flat", C["s1"], 11.5))
    for k, (name, col) in enumerate([("ours ep3 — healthy, cos 0.40", C["s1"]),
                                     ("released SFT — collapsed, cos 0.87", C["s2"])]):
        s.append(f'<rect x="{100 + k*270}" y="266" width="10" height="10" rx="2" fill="{col}"/>')
        s.append(txt(116 + k * 270, 275, name, C["ink2"], 12))
    s.append("</svg>")
    return "".join(s)


def fig_mechanism():
    """Where the collapse lives: the recurrence loop, plus the refuted hypothesis."""
    W, H = 720, 296
    s = [svg_open(W, H)]
    s.append(txt(0, 16, "the recurrence loop (output side)", C["ink2"], 13, "start", "600"))
    slots = [("slot k", 40), ("slot k+1", 250), ("slot k+2", 460)]
    for name, x in slots:
        s.append(f'<rect x="{x}" y="40" width="150" height="66" rx="10" fill="{C["surface"]}" '
                 f'stroke="{C["axis"]}" stroke-width="1"/>')
        s.append(txt(x + 75, 66, name, C["ink"], 13, "middle", "600"))
        s.append(txt(x + 75, 86, "seed → 28 layers → out", C["muted"], 11, "middle"))
    for x in (190, 400):
        s.append(f'<path d="M{x},73 h48" stroke="{C["crit"]}" stroke-width="2" marker-end="url(#ah)"/>')
    s.append(f'<defs><marker id="ah" markerWidth="8" markerHeight="8" refX="7" refY="4" '
             f'orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{C["crit"]}"/></marker></defs>')
    s.append(txt(40, 126, "each slot's output becomes the next slot's seed — the redundancy "
                          "self-amplifies", C["crit"], 11.5))
    # seed vs depth split, as a dumbbell on the same cosine axis used elsewhere
    s.append(txt(0, 160, "cosine at the seed (L0) → at the last layer (L27)", C["ink2"], 13, "start", "600"))
    ax0, ax1 = 130, 560
    def X(v):
        return ax0 + (ax1 - ax0) * v
    s.append(f'<line x1="{ax0}" y1="182" x2="{ax1}" y2="182" stroke="{C["axis"]}" stroke-width="1"/>')
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        s.append(f'<line x1="{X(v)}" y1="178" x2="{X(v)}" y2="186" stroke="{C["axis"]}" stroke-width="1"/>')
        s.append(txt(X(v), 174, f"{v:.2f}", C["muted"], 10.5, "middle"))
    rows = [("stage 2", 0.32, 0.37, C["good"]), ("stage 3 / rl", 0.72, 0.88, C["crit"])]
    for k, (name, l0, l27, col) in enumerate(rows):
        y = 210 + 32 * k
        s.append(txt(ax0 - 12, y + 5, name, C["ink2"], 12.5, "end"))
        s.append(f'<line x1="{X(l0)}" y1="{y}" x2="{X(l27)}" y2="{y}" stroke="{col}" stroke-width="2"/>')
        s.append(f'<circle cx="{X(l0)}" cy="{y}" r="6" fill="{C["surface"]}" stroke="{col}" '
                 f'stroke-width="2"><title>{html.escape(name)} L0 (seed) cos {l0}</title></circle>')
        s.append(f'<circle cx="{X(l27)}" cy="{y}" r="6" fill="{col}" stroke="{C["surface"]}" '
                 f'stroke-width="2"><title>{html.escape(name)} L27 cos {l27}</title></circle>')
        s.append(txt(X(l27) + 16, y + 5, f"{l0} → {l27}", C["ink"], 12.5, "start", "700"))
    s.append(f'<circle cx="{ax0}" cy="278" r="6" fill="{C["surface"]}" stroke="{C["muted"]}" stroke-width="2"/>')
    s.append(txt(ax0 + 14, 282, "seed (L0)", C["muted"], 11.5))
    s.append(f'<circle cx="{ax0+110}" cy="278" r="6" fill="{C["muted"]}"/>')
    s.append(txt(ax0 + 124, 282, "last layer (L27)", C["muted"], 11.5))
    s.append(txt(ax0 + 250, 282, "≈ ⅔ of the gap is set by the seed alone, ⅓ by depth", C["muted"], 11.5))
    s.append("</svg>")
    return "".join(s)


def fig_attention():
    """The refuted hypothesis, as its own small panel."""
    W, H = 720, 118
    s = [svg_open(W, H)]
    s.append(txt(0, 18, "peak mean attention onto prior latent slots", C["ink2"], 13, "start", "600"))
    for k, (name, v, col) in enumerate([("stage 2", 0.46, C["good"]), ("stage 3 / rl", 0.06, C["crit"])]):
        y = 40 + 34 * k
        s.append(txt(110, y + 12, name, C["ink2"], 12.5, "end"))
        s.append(bar(120, y, 460 * (v / 0.5), 15, col,
                     title=f"{name}: peak latent_prior attention {v} at L2"))
        s.append(txt(120 + 460 * (v / 0.5) + 10, y + 12, f"{v}", C["ink"], 13, "start", "700"))
    s.append(txt(120, 112, "the collapsed model attends to prior slots LESS — attention copying is refuted",
                 C["muted"], 11.5))
    s.append("</svg>")
    return "".join(s)


def fig_pipeline():
    """LVR pipeline status strip."""
    steps = [("0", "vendor", "done"), ("1", "build env", "done"),
             ("2", "142 GB data", "done"), ("3", "Stage-1 SFT", "done"),
             ("4", "eval SFT", "done"), ("5", "Stage-2 RL", "blocked"),
             ("6", "eval RL gain", "pending")]
    W, H = 720, 116
    s = [svg_open(W, H)]
    x = 0
    wpx = 96
    for num, name, state in steps:
        col = {"done": C["good"], "blocked": C["crit"], "pending": C["axis"]}[state]
        fill = {"done": C["good"], "blocked": C["crit"], "pending": "none"}[state]
        s.append(f'<rect x="{x}" y="30" width="{wpx-8}" height="34" rx="8" fill="{fill}" '
                 f'opacity="{0.18 if state!="blocked" else 0.3}" stroke="{col}" stroke-width="1.5"/>')
        s.append(txt(x + (wpx - 8) / 2, 52, name, C["ink"] if state != "pending" else C["muted"],
                     12, "middle", "600" if state == "blocked" else "400"))
        s.append(txt(x + (wpx - 8) / 2, 22, num, C["muted"], 11, "middle"))
        mark = {"done": "✓", "blocked": "✕", "pending": "·"}[state]
        s.append(txt(x + (wpx - 8) / 2, 84, mark, col, 15, "middle", "700"))
        if state == "blocked":
            s.append(txt(x + (wpx - 8) / 2, 104, "reward ≡ 0", C["crit"], 11.5, "middle", "600"))
        x += wpx
    s.append("</svg>")
    return "".join(s)


def fig_anchor():
    """Released LVR-7B through our harness vs the paper, at each decoding budget."""
    W, H = 720, 210
    x0, x1 = 180, 620
    lo, hi = 68, 84
    def X(v):
        return x0 + (x1 - x0) * (v - lo) / (hi - lo)
    s = [svg_open(W, H)]
    for v in range(68, 85, 2):
        s.append(f'<line x1="{X(v)}" y1="48" x2="{X(v)}" y2="164" stroke="{C["grid"]}" stroke-width="1"/>')
        s.append(txt(X(v), 40, str(v), C["muted"], 11, "middle"))
    rows = [("V*", [(4, 80.63, 81.2), (8, 81.68, 81.7), (16, 80.63, 80.6)], 74),
            ("MMVP", [(4, 72.00, 72.0), (8, 72.00, 71.7), (16, 71.67, 71.7)], 130)]
    for name, pts, y0 in rows:
        s.append(txt(x0 - 16, y0 + 20, name, C["ink"], 13.5, "end", "700"))
        for k, (step, ours, paper) in enumerate(pts):
            y = y0 + 18 * k
            s.append(txt(x0 - 16, y + 4, f"s{step}", C["muted"], 11, "end") if False else "")
            s.append(f'<circle cx="{X(paper)}" cy="{y}" r="8" fill="none" stroke="{C["s2"]}" '
                     f'stroke-width="2"><title>paper s{step}: {paper}</title></circle>')
            s.append(f'<circle cx="{X(ours)}" cy="{y}" r="5" fill="{C["s1"]}" stroke="{C["surface"]}" '
                     f'stroke-width="2"><title>ours s{step}: {ours}</title></circle>')
            s.append(txt(X(max(ours, paper)) + 16, y + 4, f"s{step}   {ours}", C["ink2"], 12))
    s.append(f'<circle cx="186" cy="196" r="5" fill="{C["s1"]}"/>')
    s.append(txt(200, 200, "ours", C["ink2"], 12))
    s.append(f'<circle cx="266" cy="196" r="8" fill="none" stroke="{C["s2"]}" stroke-width="2"/>')
    s.append(txt(282, 200, "paper", C["ink2"], 12))
    s.append(txt(360, 200, "every pair within 0.6 pt of the paper", C["muted"], 11.5))
    s.append(txt(x0 - 16, 20, "released LVR-7B", C["ink2"], 12.5, "end", "600"))
    s.append("</svg>")
    return "".join(s)


def fig_sft_baseline():
    """Our 3B Stage-1 SFT across decoding budgets, against the paper's 3B RL target band."""
    W, H = 720, 188
    x0, x1 = 180, 620
    lo, hi = 50, 75
    def X(v):
        return x0 + (x1 - x0) * (v - lo) / (hi - lo)
    s = [svg_open(W, H)]
    for v in range(50, 76, 5):
        s.append(f'<line x1="{X(v)}" y1="46" x2="{X(v)}" y2="146" stroke="{C["grid"]}" stroke-width="1"/>')
        s.append(txt(X(v), 38, str(v), C["muted"], 11, "middle"))
    rows = [("V*", [65.97, 65.45, 65.45], (65, 67), 74),
            ("MMVP", [55.67, 56.33, 57.33], (55, 58), 116)]
    for name, vals, band, y in rows:
        s.append(txt(x0 - 16, y + 5, name, C["ink"], 13.5, "end", "700"))
        s.append(f'<rect x="{X(band[0])}" y="{y-16}" width="{X(band[1])-X(band[0])}" height="32" '
                 f'rx="5" fill="{C["s3"]}" opacity="0.18"><title>paper 3B RL target '
                 f'~{band[0]}–{band[1]}</title></rect>')
        for step, v in zip((4, 8, 16), vals):
            s.append(f'<circle cx="{X(v)}" cy="{y}" r="5.5" fill="{C["s1"]}" stroke="{C["surface"]}" '
                     f'stroke-width="2"><title>our 3B SFT s{step}: {v}</title></circle>')
        s.append(txt(X(max(vals)) + 16, y + 5, " / ".join(str(v) for v in vals), C["ink2"], 12))
    s.append(f'<circle cx="186" cy="172" r="5.5" fill="{C["s1"]}"/>')
    s.append(txt(200, 176, "our 3B SFT (s4 / s8 / s16)", C["ink2"], 12))
    s.append(f'<rect x="380" y="166" width="18" height="12" rx="3" fill="{C["s3"]}" opacity="0.3"/>')
    s.append(txt(404, 176, "paper 3B RL target — what Stage-2 has to beat", C["ink2"], 12))
    s.append("</svg>")
    return "".join(s)


def fig_papersuite():
    """Our ep3 at fixed K=8 against the paper's reported rows, per benchmark."""
    W, H = 720, 260
    x0, x1 = 190, 610
    lo, hi = 25, 90
    def X(v):
        return x0 + (x1 - x0) * (v - lo) / (hi - lo)
    rows = [
        ("V*", 77.49, None, 82.20, 83.25, 76.44),
        ("HRBench4K", 62.1, None, 68.50, 71.00, 68.00),
        ("HRBench8K", 59.1, None, 66.00, 68.00, 63.75),
        ("MME-RW-Lite", 30.0, 41.3, 52.68, 55.50, 45.75),
    ]
    s = [svg_open(W, H)]
    for v in (30, 40, 50, 60, 70, 80, 90):
        s.append(f'<line x1="{X(v)}" y1="46" x2="{X(v)}" y2="196" stroke="{C["grid"]}" stroke-width="1"/>')
        s.append(txt(X(v), 38, str(v), C["muted"], 11, "middle"))
    for k, (name, ours, ours_hi, sft, rl, qwen) in enumerate(rows):
        y = 70 + 40 * k
        s.append(txt(x0 - 16, y + 5, name, C["ink"], 12.5, "end", "600"))
        # Qwen baseline is a reference tick, not a series
        s.append(f'<line x1="{X(qwen)}" y1="{y-12}" x2="{X(qwen)}" y2="{y+12}" stroke="{C["muted"]}" '
                 f'stroke-width="2"><title>paper Qwen2.5-VL-7B: {qwen}</title></line>')
        s.append(f'<circle cx="{X(sft)}" cy="{y}" r="7" fill="none" stroke="{C["s2"]}" stroke-width="2">'
                 f'<title>paper Monet-SFT best-K: {sft}</title></circle>')
        s.append(f'<circle cx="{X(rl)}" cy="{y}" r="7" fill="none" stroke="{C["s3"]}" stroke-width="2">'
                 f'<title>paper Monet-7B RL: {rl}</title></circle>')
        if ours_hi is not None:
            s.append(f'<rect x="{X(ours)-6}" y="{y-6}" width="{X(ours_hi)-X(ours)+12}" height="12" rx="6" '
                     f'fill="{C["s1"]}" opacity="0.4"/>')
            for e in (ours, ours_hi):
                s.append(f'<circle cx="{X(e)}" cy="{y}" r="6" fill="{C["s1"]}" stroke="{C["surface"]}" '
                         f'stroke-width="2"><title>ours K=8: {e}</title></circle>')
            # this row is crowded on both sides — label the band underneath it
            s.append(txt((X(ours) + X(ours_hi)) / 2, y + 23, "30.0 – 41.3", C["ink"], 12,
                         "middle", "700"))
        else:
            s.append(f'<circle cx="{X(ours)}" cy="{y}" r="6" fill="{C["s1"]}" stroke="{C["surface"]}" '
                     f'stroke-width="2"><title>ours ep3 K=8: {ours}</title></circle>')
            s.append(txt(X(ours) - 14, y + 5, f"{ours}", C["ink"], 12.5, "end", "700"))
    leg = [("ours ep3, K=8", C["s1"], "dot"), ("paper Monet-SFT", C["s2"], "ring"),
           ("paper Monet-7B RL", C["s3"], "ring"), ("paper Qwen2.5-VL-7B", C["muted"], "tick")]
    lx = 60
    for label, col, kind in leg:
        if kind == "dot":
            s.append(f'<circle cx="{lx}" cy="228" r="5.5" fill="{col}"/>')
        elif kind == "ring":
            s.append(f'<circle cx="{lx}" cy="228" r="7" fill="none" stroke="{col}" stroke-width="2"/>')
        else:
            s.append(f'<line x1="{lx}" y1="220" x2="{lx}" y2="236" stroke="{col}" stroke-width="2"/>')
        s.append(txt(lx + 13, 232, label, C["ink2"], 11.5))
        lx += 13 + 6.6 * len(label) + 22
    s.append(txt(60, 254, "MME-RW-Lite carries both its strict-scorer number and the manual "
                          "re-extraction — ep3 emits 0% \\boxed{}", C["muted"], 11))
    s.append("</svg>")
    return "".join(s)


def fig_blocker():
    """What is eliminated, what remains."""
    W, H = 720, 316
    s = [svg_open(W, H)]
    elim = [
        ("token stripping", "lvr tokens are non-special → skip_special_tokens does not strip"),
        ("the reward functions", "CPU fp32 run: format_reward = 1.0"),
        ("the format + the SFT model", "well-formed completions, accuracy fires 1 of 4"),
        ("sampling config", "matches HF defaults (top_p 1, top_k 50, rep 1)"),
        ("dtype (bf16)", "mild degradation only — 1 of 16 at n=16"),
    ]
    y = 34
    s.append(txt(0, 16, "ruled out on cpu-short", C["ink2"], 13, "start", "600"))
    for name, why in elim:
        s.append(f'<rect x="0" y="{y}" width="700" height="30" rx="7" fill="{C["surface"]}" '
                 f'stroke="{C["axis"]}" stroke-width="1"/>')
        s.append(txt(16, y + 20, "✓", C["good"], 14, "start", "700"))
        s.append(txt(38, y + 20, name, C["muted"], 12.5, "start", "600",
                     'style="text-decoration:line-through"'))
        s.append(txt(258, y + 20, why, C["muted"], 12))
        y += 36
    s.append(f'<rect x="0" y="{y+8}" width="700" height="68" rx="8" fill="{C["crit"]}" '
             f'opacity="0.14" stroke="{C["crit"]}" stroke-width="1.5"/>')
    s.append(txt(16, y + 32, "✕", C["crit"], 15, "start", "700"))
    s.append(txt(38, y + 32, "what remains: something DeepSpeed / multi-GPU specific",
                 C["ink"], 13.5, "start", "700"))
    s.append(txt(38, y + 51, "systematic — format_reward = 0 on every completion, at every step",
                 C["ink2"], 12))
    s.append(txt(38, y + 68, "GPU mean completion length 45  vs  22 on CPU", C["ink2"], 12))
    s.append("</svg>")
    return "".join(s)


# ------------------------------------------------------------- slides
def tile(value, label, tone="ink"):
    col = {"ink": C["ink"], "crit": C["crit"], "good": C["good"], "warn": C["warn"]}[tone]
    return (f'<div class="tile"><div class="tile-v" style="color:{col}">{value}</div>'
            f'<div class="tile-l">{label}</div></div>')


def build_slides():
    S = []

    S.append(dict(sec=0, kicker="state of play", head="Two upstreams reproduced. One bug in the way.",
        body=f'''<div class="tiles">{tile("1", "open blocker", "crit")}
        {tile("2", "faithful upstream reproductions", "good")}
        {tile("7wk", "since the last commit")}</div>
        <figure>{fig_tracks()}</figure>''',
        sowhat="LVR Stage-2 GRPO trains with reward ≡ 0 — job 221059 ran 319 steps with no RL signal. "
               "The next diagnostic probe is committed but was never run."))

    S.append(dict(sec=1, kicker="the original thesis", head="Latents a frozen reader can actually use.",
        body=f'''<figure>{fig_goal()}</figure>
        <p class="lede">The generator emits <code>h ∈ R^(K×D)</code> from an image. Frozen anchor
        models — siblings of the generator, never trained on it — consume <code>h</code> spliced into
        their own vision-token positions. The score is <strong>transfer utility in nats</strong>: how
        much a frozen reader's answer likelihood improves when handed <code>h</code>.</p>''',
        sowhat="If only the training decoder can read h, nothing has been learned that generalises. "
               "Utility, not reconstruction, is the target."))

    S.append(dict(sec=2, kicker="phase 0 — probing released Monet", head="The K latents converge to one vector.",
        body=f'''<figure>{fig_cos_line([
            ("Monet stage 2", 0.38, "0.38", C["good"], "utility +2.19 nat"),
            ("Monet stage 3", (0.85, 0.89), "0.85 – 0.89", C["crit"], "utility −0.13 nat"),
        ])}</figure>
        <p>Phase 0, n=200 held-out Visual_CoT. Stage 3 self-distills onto stage-2 targets and lands with
        positions 4–7 collinear at cos ≈ 1.00 — and frozen-reader transfer goes <em>negative</em>.</p>''',
        sowhat="compression_ratio is blind to this. The pairwise cosine matrix is the load-bearing "
               "diagnostic, and has been logged automatically since 2026-05-07."))

    S.append(dict(sec=3, kicker="phases 1 / 1.5 / 1.5b / 2", head="Every phase eliminated a suspect.",
        body=f'''<figure>{fig_ruledout()}</figure>''',
        sowhat="The loss is exonerated and attention isolation is not the mechanism — which is exactly "
               "what pointed at direct representation-level regularization."))

    S.append(dict(sec=4, kicker="pivot A — the breakthrough", head="VICReg breaks the collapse at 3B/5K.",
        body=f'''<figure>{fig_pivot()}</figure>''',
        sowhat="Recipe G — VICReg λ_reg=2, mean-MSE LVR, K=4, 2000 steps — matches Monet stage 2's cosine "
               "and became the validated cluster starting point. Utility still ~10× short."))

    S.append(dict(sec=5, kicker="mechanism", head="It is the output side, not attention.",
        body=f'''<figure>{fig_mechanism()}</figure>
        <figure>{fig_attention()}</figure>''',
        sowhat="Post-RL ≡ stage 3 at every diagnostic (CKA 0.938) — RL did not recover reader transfer. "
               "The collapsed model writes near-identical residuals per slot whatever its attention does."))

    S.append(dict(sec=6, kicker="dissociation 1", head="The cosine signature moves. Utility does not.",
        body=f'''<figure>{fig_dissociation()}</figure>
        <p>Inference-time <code>vicreg_project</code> orthogonalises the K recurrence seeds and re-injects
        them. n=500 per condition. CKA between the two stage-3 conditions is 0.518 — the representation
        genuinely rotates.</p>''',
        sowhat="Matching stage 2's geometry is necessary but not sufficient. Cosine is the trace of "
               "utility, not its cause — so no success criterion should rest on it alone."))

    S.append(dict(sec=7, kicker="finding 2", head="The faithful stage-3 recipe never collapses.",
        body=f'''<figure>{fig_cos_line([
            ("released stage 2 (teacher)", 0.377, "0.377", C["good"], "util +2.05"),
            ("ours, epochs=3", 0.401, "0.401", C["s1"], "util +1.02"),
            ("ours, epochs=2", 0.420, "0.420", C["s1"], "util +0.90"),
            ("released stage 3", 0.871, "0.871", C["crit"], "util −0.05"),
        ])}</figure>
        <p>The epochs hypothesis was tested and killed: released = 3 epochs, we ran 2 <em>and</em> 3, and all
        training code is <strong>byte-identical</strong> between the upload commit and HEAD — a 0-line diff.
        A faithful stage 3 tracks its healthy teacher to ~0.40 and stays useful.</p>''',
        sowhat="The released collapse is unreproducible from the public recipe. \"VICReg fixes Monet's "
               "collapse\" is not viable as posed — there is no collapse in the faithful baseline to fix."))

    S.append(dict(sec=8, kicker="finding 3", head="Same score at K=8. Only one of them scales with K.",
        body=f'''<div class="tiles">{tile("148/191", "ours ep3 — cos 0.401")}
        {tile("148/191", "released — cos 0.871")}
        {tile("0.0", "net V* difference", "warn")}</div>
        <figure>{fig_ksweep()}</figure>''',
        sowhat="Whatever produced the released collapse also gave it test-time-K extrapolation. Ours has "
               "neither — and since K=8 is optimal for our checkpoint, the paper-suite gap is real, not a "
               "K-protocol artifact."))

    S.append(dict(sec=9, kicker="monet paper suite", head="Below the paper on every reported benchmark.",
        body=f'''<figure>{fig_papersuite()}</figure>''',
        sowhat="At the trained K=8 on the one parser-clean benchmark, ep3 matches released SFT exactly "
               "(77.49). The paper's headline gains come from RL — which our checkpoint does not have."))

    S.append(dict(sec=10, kicker="second upstream", head="LVR: the harness was proven before training.",
        body=f'''<figure>{fig_anchor()}</figure>
        <p>Qwen2.5-VL base, vision encoder and merger frozen. <code>&lt;lvr&gt;</code> placeholders are
        filled with ROI visual-patch embeddings and the hidden state before each is regressed toward
        them — <code>L = L_NTP + 0.1·MSE</code>, no projection head. Stage-1 SFT on Visual-CoT, then
        Stage-2 GRPO_latent RL on ViRL39K.</p>''',
        sowhat="Running the released LVR-7B through our own harness first means any later gap is "
               "attributable to training, not to measurement."))

    S.append(dict(sec=10, kicker="reproduction cost", head="The published repo does not run as-is.",
        body=f'''<div class="ledger">{ledger_cards()}</div>''',
        sowhat="Method, hyperparameters and data were fully specified; the environment was not. Ten distinct "
               "repairs stand between the published recipe and a run — all documented in docs/lvr/."))

    S.append(dict(sec=11, kicker="LVR pipeline", head="Five of six steps done. Stage-2 is the wall.",
        body=f'''<figure>{fig_pipeline()}</figure>
        <figure>{fig_sft_baseline()}</figure>''',
        sowhat="Stage-1 converged cleanly (loss_total 7.5 → 0.37) with no collapse, and already sits in "
               "the paper's 3B RL target band — which is exactly what the blocked stage must improve on."))

    S.append(dict(sec=12, kicker="the blocker", head="Reward ≡ 0 — and it is not the reward code.",
        body=f'''<figure>{fig_blocker()}</figure>''',
        sowhat="The CPU path proves every component works in isolation. The failure only exists on the GPU "
               "rollout, so the next evidence has to come from there."))

    S.append(dict(sec=13, kicker="housekeeping", head="Results that exist in exactly one place.",
        body=f'''<div class="risks">
        <div class="risk"><div class="risk-h">Untracked on disk only — no backup</div>
        <p><code>docs/overnight_2026_05_{{20,22,24}}/</code>, <code>phase0_monet_probe/mech{{,2,3}}/</code>,
        <code>eval_local/</code>, <code>lvr_eval/</code>, the <code>configs/interleaved_*</code> sweep,
        <code>scripts/overnight_*.sh</code>, <code>docs/lvr/</code>.</p></div>
        <div class="risk"><div class="risk-h">Branch drift</div>
        <p><code>origin/main</code> = <code>b81ded9</code>; local <code>main</code> is 16 commits behind.
        The real HEAD is in the <code>jobA2-ep3</code> worktree.</p></div>
        <div class="risk"><div class="risk-h">Stale README</div>
        <p>Still reads "v0.1.0 — scaffold," about three months out of date.</p></div>
        </div>''',
        sowhat="Every finding in this deck is reconstructable from those directories. Committing them is "
               "cheap insurance."))

    S.append(dict(sec=14, kicker="next actions", head="Three, in order.",
        body=f'''<div class="acts">
        <div class="act"><span class="an">1</span><div>
          <div class="ah">Unblock Stage-2</div>
          <pre class="cmd"><code>STEPS=2 DEBUG_COMPLETIONS=1 sb lvr_stage2_3b</code></pre>
          <p>On the cluster, user-submitted, from <code>upstreams/lvr</code>. Read the dumped GPU
          rollout text — the one piece of evidence the CPU path cannot produce.</p></div></div>
        <div class="act"><span class="an">2</span><div>
          <div class="ah">Decide the thesis framing</div>
          <p>"VICReg rescues Monet stage-3 collapse" needs restating, because the faithful baseline does
          not collapse. The live alternatives: the K-scaling result as the real phenomenon, or LVR
          SFT→RL as the primary contribution.</p></div></div>
        <div class="act"><span class="an">3</span><div>
          <div class="ah">Back up the untracked results</div>
          <p>Commit or copy the directories listed in §13 before anything else touches that disk.</p>
          </div></div></div>''',
        sowhat="Standing rule: cluster jobs are never agent-submitted."))
    return S


def fig_goal():
    W, H = 720, 196
    s = [svg_open(W, H)]
    s.append(f'<rect x="0" y="56" width="150" height="70" rx="10" fill="{C["surface"]}" '
             f'stroke="{C["axis"]}"/>')
    s.append(txt(75, 86, "generator VLM", C["ink"], 13, "middle", "600"))
    s.append(txt(75, 106, "image → h", C["muted"], 11.5, "middle"))
    for k in range(4):
        s.append(f'<rect x="{206 + k*30}" y="{74}" width="22" height="34" rx="5" fill="{C["s1"]}" '
                 f'opacity="{0.55 + 0.15*k}"><title>latent slot {k+1}</title></rect>')
    s.append(txt(266, 66, "K latents  h ∈ R^(K×D)", C["ink2"], 12, "middle", "600"))
    s.append(f'<path d="M158,91 h40" stroke="{C["muted"]}" stroke-width="2" marker-end="url(#a2)"/>')
    s.append(f'<path d="M338,91 h44" stroke="{C["muted"]}" stroke-width="2" marker-end="url(#a2)"/>')
    s.append(f'<defs><marker id="a2" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
             f'<path d="M0,0 L8,4 L0,8 z" fill="{C["muted"]}"/></marker></defs>')
    for k, name in enumerate(["frozen anchor A", "frozen anchor B", "frozen anchor C"]):
        y = 30 + k * 52
        s.append(f'<rect x="392" y="{y}" width="180" height="42" rx="9" fill="none" '
                 f'stroke="{C["s3"]}" stroke-width="1.5" stroke-dasharray="0"/>')
        s.append(txt(482, y + 26, name, C["s3"], 12.5, "middle", "600"))
    s.append(txt(596, 82, "utility", C["ink"], 13, "start", "700"))
    s.append(txt(596, 100, "in nats", C["muted"], 11.5))
    s.append(txt(0, 186, "readers stay frozen — h is spliced into their vision-token positions",
                 C["muted"], 11.5))
    s.append("</svg>")
    return "".join(s)


def fig_ruledout():
    rows = [
        ("mean-MSE LVR loss", "0.851", "collapse", C["crit"]),
        ("sum-MSE LVR loss  (~2048× stronger)", "0.987", "collapse is worse", C["crit"]),
        ("the ROI targets themselves  (v_roi)", "0.465", "targets are fine → loss exonerated", C["good"]),
        ("Monet model class, no 4D mask", "0.959", "collapse", C["crit"]),
        ("+ attention_mask_4d injected", "0.961", "identical → attention is not it", C["good"]),
        ("full stage-2 recipe, in-distribution", "0.704", "genuine, not an OOD artifact", C["crit"]),
    ]
    W, H = 720, 40 + 38 * len(rows)
    s = [svg_open(W, H)]
    s.append(txt(0, 16, "mean off-diagonal cosine", C["ink2"], 12.5, "start", "600"))
    for k, (name, val, verdict, col) in enumerate(rows):
        y = 30 + 38 * k
        s.append(f'<rect x="0" y="{y}" width="700" height="30" rx="7" fill="{C["surface"]}" '
                 f'stroke="{C["axis"]}" stroke-width="1"/>')
        s.append(txt(16, y + 20, name, C["ink2"], 12.5))
        s.append(txt(348, y + 20, val, C["ink"], 13.5, "end", "700"))
        s.append(f'<circle cx="368" cy="{y+15}" r="4.5" fill="{col}"/>')
        s.append(txt(382, y + 20, verdict, col if col == C["good"] else C["muted"], 12))
    s.append("</svg>")
    return "".join(s)


LEDGER = [
    ("1", "flash-attn unversioned → 2.8.3 ABI mismatch", "version drift",
     "inferred 2.7.4.post1 — shipped the same day as torch 2.6.0"),
    ("2", "triton JIT needs Python.h", "unpublished env", "venv on uv-managed standalone CPython"),
    ("3", "code mixes from src. and from train.", "unpublished env", "PYTHONPATH=$REPO:$REPO/src"),
    ("4", "qwen-vl-utils imported, not in requirements", "version drift", "added to env build"),
    ("5", "av yanked; hub pin contradicts transformers", "inconsistent pins", "uv --override, requirements pristine"),
    ("6", "packed_fixedToken dataset never committed", "missing file", "stub that raises if called"),
    ("7", "s3_checkpoints_lvr.py truncated", "upstream defect", "appended pass"),
    ("8", "hard-coded Oracle Cloud credentials", "security", "removed; no code edit needed"),
    ("9", "script assumes 8 GPUs", "cluster adapt", "grad_accum 16 → effective batch 64"),
    ("10", "rsync-based transfer", "cluster adapt", "vendored; path-independent sbatch"),
]


def ledger_cards():
    out = []
    tone = {"version drift": C["warn"], "unpublished env": C["s1"], "missing file": C["crit"],
            "upstream defect": C["crit"], "inconsistent pins": C["warn"],
            "cluster adapt": C["s3"], "security": C["serious"]}
    for num, prob, cls, fix in LEDGER:
        out.append(
            f'<div class="lcard"><div class="lhead"><span class="lnum">{num}</span>'
            f'<span class="ltag" style="color:{tone[cls]};border-color:{tone[cls]}55">{cls}</span></div>'
            f'<div class="lprob">{html.escape(prob)}</div>'
            f'<div class="lfix">{html.escape(fix)}</div></div>')
    return "".join(out)


# ------------------------------------------------------------- assemble
def main():
    md = SRC.read_text()
    sha = hashlib.sha256(md.encode()).hexdigest()
    secs = split_sections(md)
    slides = build_slides()

    panes = []
    for n, (title, raw) in sorted(secs.items()):
        panes.append(f'<section class="srcsec" id="src-{n}">'
                     f'<div class="srchead">source § {n} — {html.escape(title)}</div>'
                     f'<div class="srcbody">{md_to_html(raw)}</div></section>')

    cards = []
    for idx, sl in enumerate(slides):
        sec = sl["sec"]
        cards.append(f'''<section class="slide" data-i="{idx}" id="s{idx+1}">
  <div class="slide-in">
    <div class="kicker">{html.escape(sl["kicker"])}</div>
    <h2 class="head">{html.escape(sl["head"])}</h2>
    <div class="body">{sl["body"]}</div>
    <div class="sowhat"><span>so what</span><p>{html.escape(sl["sowhat"])}</p></div>
    <button class="srcbtn" data-src="{sec}">source § {sec} ·  verbatim table</button>
  </div>
</section>''')

    nav = "".join(
        f'<button class="ovcard" data-go="{i}"><span class="ovn">{i+1}</span>'
        f'<span class="ovt">{html.escape(s["head"])}</span></button>' for i, s in enumerate(slides))

    doc = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>visual-latents — recap 2026-08-06</title>
<style>
:root{{
  --page:{C['page']}; --surface:{C['surface']}; --ink:{C['ink']}; --ink2:{C['ink2']};
  --muted:{C['muted']}; --grid:{C['grid']}; --axis:{C['axis']};
  --s1:{C['s1']}; --s2:{C['s2']}; --s3:{C['s3']};
  --good:{C['good']}; --warn:{C['warn']}; --crit:{C['crit']};
  color-scheme: dark;
}}
*{{box-sizing:border-box}}
html,body{{margin:0;padding:0;background:var(--page);color:var(--ink);
 font:16px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
 -webkit-text-size-adjust:100%}}
body{{overflow:hidden}}
body.overview,body.srcopen{{overflow:auto}}

/* progress */
#bar{{position:fixed;top:0;left:0;height:2px;background:var(--s1);z-index:60;transition:width .25s}}

/* deck */
#deck{{height:100vh;height:100dvh;overflow:hidden;position:relative}}
.slide{{position:absolute;inset:0;display:none;overflow-y:auto;padding:56px 28px 96px}}
.slide.on{{display:block}}
.slide-in{{max-width:860px;margin:0 auto}}
.kicker{{font-size:12.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);
 font-weight:700;margin-bottom:10px}}
.head{{font-size:30px;line-height:1.22;letter-spacing:-.015em;margin:0 0 22px;font-weight:700}}
.body p{{color:var(--ink2);font-size:15.5px}}
.body p.lede{{font-size:16.5px;color:var(--ink)}}
figure{{margin:18px 0;background:var(--surface);border:1px solid var(--axis);border-radius:12px;
 padding:16px 18px}}
code{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.92em;
 background:#20201e;color:#e5c07b;padding:1px 5px;border-radius:4px}}
pre.cmd{{background:var(--surface);border:1px solid var(--s3);border-radius:10px;padding:16px 18px;
 overflow-x:auto}}
pre.cmd code{{background:none;color:var(--s3);font-size:16px;padding:0}}

/* tiles */
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:4px 0 18px}}
.tile{{background:var(--surface);border:1px solid var(--axis);border-radius:12px;padding:14px 16px}}
.tile-v{{font-size:30px;font-weight:800;letter-spacing:-.02em;line-height:1.1}}
.tile-l{{color:var(--muted);font-size:12.5px;margin-top:4px}}

/* so-what */
.sowhat{{margin:22px 0 8px;border-left:2px solid var(--s1);padding:2px 0 2px 16px}}
.sowhat span{{display:block;font-size:11.5px;letter-spacing:.09em;text-transform:uppercase;
 color:var(--muted);font-weight:700;margin-bottom:4px}}
.sowhat p{{margin:0;font-size:16px;color:var(--ink)}}

/* ledger */
.ledger{{display:grid;grid-template-columns:repeat(auto-fit,minmax(232px,1fr));gap:10px}}
.lcard{{background:var(--surface);border:1px solid var(--axis);border-radius:10px;padding:11px 13px}}
.lhead{{display:flex;align-items:center;gap:8px;margin-bottom:6px}}
.lnum{{display:inline-grid;place-items:center;width:20px;height:20px;border-radius:5px;
 background:#232322;color:var(--muted);font-size:11.5px;font-weight:700}}
.ltag{{font-size:10.5px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;
 border:1px solid;border-radius:999px;padding:2px 7px}}
.lprob{{font-size:13px;color:var(--ink);margin-bottom:5px}}
.lfix{{font-size:12.5px;color:var(--muted)}}

/* next actions */
.acts{{display:grid;gap:14px}}
.act{{display:flex;gap:12px;align-items:flex-start;background:var(--surface);
 border:1px solid var(--axis);border-radius:10px;padding:14px 16px}}
.act .an{{display:inline-grid;place-items:center;width:24px;height:24px;border-radius:6px;
 background:#232322;color:var(--s1);font-size:13px;font-weight:700;flex:0 0 auto}}
.ah{{font-weight:700;font-size:15px;margin-bottom:6px}}
.act p{{margin:8px 0 0;font-size:14px}}
.act pre.cmd{{padding:11px 13px;margin:0}}
.act pre.cmd code{{font-size:14.5px}}

/* risks */
.risks{{display:grid;gap:12px}}
.risk{{background:var(--surface);border:1px solid var(--axis);border-left:2px solid var(--warn);
 border-radius:10px;padding:12px 15px}}
.risk-h{{font-weight:700;font-size:14.5px;margin-bottom:4px}}
.risk p{{margin:0;font-size:13.5px;color:var(--ink2)}}

/* source button + pane */
.srcbtn{{margin-top:10px;background:none;border:1px solid var(--axis);color:var(--muted);
 border-radius:999px;padding:7px 14px;font:inherit;font-size:12.5px;cursor:pointer}}
.srcbtn:hover{{color:var(--ink);border-color:var(--muted)}}
#src{{position:fixed;inset:0;background:var(--page);z-index:70;display:none;overflow-y:auto;
 padding:64px 22px 80px}}
body.srcopen #src{{display:block}}
#src .wrap{{max-width:820px;margin:0 auto}}
.srcsec{{display:none}} .srcsec.on{{display:block}}
.srchead{{font-size:12px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);
 font-weight:700;margin-bottom:14px}}
.srcbody h2{{font-size:23px;margin:0 0 14px}} .srcbody h3{{font-size:17px;margin:18px 0 6px}}
.srcbody p,.srcbody li{{font-size:15px;color:var(--ink2)}}
.srcbody table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:13.5px;
 font-variant-numeric:tabular-nums;display:block;overflow-x:auto}}
.srcbody th,.srcbody td{{border:1px solid var(--axis);padding:7px 10px;vertical-align:top}}
.srcbody th{{background:#1f1f1e;color:var(--ink);font-weight:600;text-align:left}}
.srcbody pre{{background:var(--surface);border:1px solid var(--axis);border-radius:8px;
 padding:12px 14px;overflow-x:auto;font-size:13px}}
.srcbody hr{{border:none;border-top:1px solid var(--axis);margin:20px 0}}
.srcbody strong{{color:var(--ink)}}

/* overview */
#ov{{position:fixed;inset:0;background:var(--page);z-index:65;display:none;overflow-y:auto;
 padding:64px 22px 60px}}
body.overview #ov{{display:block}}
#ov .grid{{max-width:900px;margin:0 auto;display:grid;
 grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:10px}}
.ovcard{{text-align:left;background:var(--surface);border:1px solid var(--axis);border-radius:10px;
 padding:12px 14px;color:var(--ink2);font:inherit;cursor:pointer;display:flex;gap:10px;align-items:flex-start}}
.ovcard:hover{{border-color:var(--s1);color:var(--ink)}}
.ovn{{color:var(--muted);font-size:12px;font-weight:700;min-width:18px}}
.ovt{{font-size:13.5px;line-height:1.35}}

/* chrome */
#chrome{{position:fixed;bottom:0;left:0;right:0;height:54px;display:flex;align-items:center;
 justify-content:space-between;padding:0 16px;background:linear-gradient(transparent,var(--page) 46%);
 z-index:50;gap:10px}}
#chrome button{{background:var(--surface);border:1px solid var(--axis);color:var(--ink2);
 border-radius:9px;min-width:44px;height:38px;font:inherit;font-size:13px;cursor:pointer;padding:0 12px}}
#chrome button:hover{{color:var(--ink);border-color:var(--muted)}}
#pos{{color:var(--muted);font-size:12.5px;font-variant-numeric:tabular-nums}}
#stale{{position:fixed;top:0;left:0;right:0;background:var(--warn);color:#000;font-size:13px;
 padding:8px 14px;z-index:90;display:none;font-weight:600}}
#hint{{position:fixed;top:12px;right:16px;color:var(--muted);font-size:11.5px;z-index:55}}
@media (max-width:760px){{
  .head{{font-size:24px}} .slide{{padding:44px 16px 110px}} #hint{{display:none}}
  .tile-v{{font-size:25px}}
  /* keep figure type legible: scroll the figure instead of shrinking its text */
  figure{{overflow-x:auto;-webkit-overflow-scrolling:touch;padding:14px}}
  figure svg{{min-width:600px}}
  figure::after{{content:"swipe the figure to see it all";display:block;color:var(--muted);
   font-size:11px;margin-top:8px;position:sticky;left:0}}
}}
</style></head><body>
<div id="stale"></div>
<div id="bar"></div>
<div id="hint">← → move · O overview · S source</div>
<div id="deck">{''.join(cards)}</div>
<div id="ov"><div class="grid">{nav}</div></div>
<div id="src"><div class="wrap">{''.join(panes)}</div></div>
<div id="chrome">
  <button id="prev" aria-label="previous">←</button>
  <button id="ovbtn">overview</button>
  <span id="pos"></span>
  <button id="srcclose" style="display:none">close source</button>
  <button id="next" aria-label="next">→</button>
</div>
<script>
const SHA = "{sha}";
const slides=[...document.querySelectorAll('.slide')];
const N=slides.length; let cur=0;
function show(i,push){{
  cur=Math.max(0,Math.min(N-1,i));
  slides.forEach((s,k)=>s.classList.toggle('on',k===cur));
  document.getElementById('bar').style.width=((cur+1)/N*100)+'%';
  document.getElementById('pos').textContent=(cur+1)+' / '+N;
  document.body.classList.remove('overview');
  slides[cur].scrollTop=0;
  if(push!==false) history.replaceState(null,'','#s'+(cur+1));
}}
function closeSrc(){{document.body.classList.remove('srcopen');
  document.getElementById('srcclose').style.display='none';}}
function openSrc(n){{
  document.querySelectorAll('.srcsec').forEach(s=>s.classList.remove('on'));
  const t=document.getElementById('src-'+n); if(t)t.classList.add('on');
  document.body.classList.add('srcopen'); window.scrollTo(0,0);
  document.getElementById('srcclose').style.display='';
}}
document.querySelectorAll('.srcbtn').forEach(b=>b.onclick=()=>openSrc(b.dataset.src));
document.querySelectorAll('.ovcard').forEach(b=>b.onclick=()=>show(+b.dataset.go));
document.getElementById('next').onclick=()=>{{closeSrc();show(cur+1);}};
document.getElementById('prev').onclick=()=>{{closeSrc();show(cur-1);}};
document.getElementById('srcclose').onclick=closeSrc;
document.getElementById('ovbtn').onclick=()=>{{closeSrc();
  document.body.classList.toggle('overview');window.scrollTo(0,0);}};
addEventListener('keydown',e=>{{
  if(e.key==='ArrowRight'||e.key===' '||e.key==='PageDown'){{closeSrc();show(cur+1);e.preventDefault();}}
  else if(e.key==='ArrowLeft'||e.key==='PageUp'){{closeSrc();show(cur-1);e.preventDefault();}}
  else if(e.key==='o'||e.key==='O'){{closeSrc();document.body.classList.toggle('overview');window.scrollTo(0,0);}}
  else if(e.key==='s'||e.key==='S'){{document.body.classList.contains('srcopen')?closeSrc():openSrc(slides[cur].querySelector('.srcbtn').dataset.src);}}
  else if(e.key==='Escape'){{closeSrc();document.body.classList.remove('overview');}}
  else if(e.key==='Home'){{show(0);}} else if(e.key==='End'){{show(N-1);}}
}});
let tx=0,ty=0,inFig=false;
addEventListener('touchstart',e=>{{tx=e.changedTouches[0].clientX;ty=e.changedTouches[0].clientY;
  inFig=!!(e.target.closest&&e.target.closest('figure'));}},{{passive:true}});
addEventListener('touchend',e=>{{
  if(inFig)return;   // a horizontal drag inside a figure scrolls it, never changes slide
  if(document.body.classList.contains('srcopen')||document.body.classList.contains('overview'))return;
  const dx=e.changedTouches[0].clientX-tx, dy=e.changedTouches[0].clientY-ty;
  if(Math.abs(dx)>60&&Math.abs(dx)>Math.abs(dy)*1.6) show(cur+(dx<0?1:-1));
}},{{passive:true}});
const h=location.hash.match(/^#s(\\d+)$/); show(h?(+h[1]-1):0,false);
// stale-source detector: the markdown is canonical; announce if it moved on.
fetch('RECAP_2026_08_06.md',{{cache:'no-store'}}).then(r=>r.ok?r.arrayBuffer():null).then(b=>{{
  if(!b)return; return crypto.subtle.digest('SHA-256',b).then(d=>{{
    const hex=[...new Uint8Array(d)].map(x=>x.toString(16).padStart(2,'0')).join('');
    if(hex!==SHA){{const el=document.getElementById('stale');
      el.textContent='⚠ This page was built from an older RECAP_2026_08_06.md — regenerate it.';
      el.style.display='block';}}
  }});
}}).catch(()=>{{}});
</script></body></html>"""
    OUT.write_text(doc)
    print(f"wrote {OUT}  ({len(doc)//1024} KB)")
    print(f"source sha256 {sha[:16]}…  sections {sorted(secs)}  slides {len(slides)}")


if __name__ == "__main__":
    main()
