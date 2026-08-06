#!/usr/bin/env python3
"""Build the visual-latents recap deck.

Canonical source: docs/RECAP_2026_08_06.md  (markdown is authoritative)
Output:           docs/RECAP_2026_08_06.html (disposable — regenerate, never hand-edit)

Every slide is a distilled view; every slide links to a faithful conversion of the
exact source section it came from, embedded in the same file. Numbers are the
source's, verbatim.
"""
import hashlib
import html
import re
import sys
from pathlib import Path

DOCS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "docs"
SRC = DOCS / "RECAP_2026_08_06.md"
OUT = DOCS / "RECAP_2026_08_06.html"

INK = "#e6edf3"
MUTED = "#9aa7b4"
LINE = "#2a3441"
OK = "#56d364"
WARN = "#e5c07b"
BAD = "#ff7b72"
BLUE = "#79c0ff"
VIO = "#d2a8ff"

# ─────────────────────────────────────────────────────────── markdown → html


def md_inline(s):
    s = html.escape(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*([^*]+)\*(?![\w*])", r"<em>\1</em>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    s = s.replace("\\|", "|").replace("\\*", "*")
    return s


def md_table(lines):
    head = [c.strip() for c in lines[0].strip().strip("|").split("|")]
    align = []
    for c in lines[1].strip().strip("|").split("|"):
        c = c.strip()
        align.append("right" if c.endswith(":") and not c.startswith(":")
                     else "center" if c.startswith(":") and c.endswith(":") else "left")
    out = ["<table><thead><tr>"]
    for i, h in enumerate(head):
        out.append(f'<th style="text-align:{align[i]}">{md_inline(h)}</th>')
    out.append("</tr></thead><tbody>")
    for row in lines[2:]:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        out.append("<tr>")
        for i, c in enumerate(cells):
            a = align[i] if i < len(align) else "left"
            out.append(f'<td style="text-align:{a}">{md_inline(c)}</td>')
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def md_to_html(md):
    """Small faithful converter — headings, tables, lists, quotes, rules, paras."""
    lines = md.split("\n")
    out, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
            continue
        if ln.startswith("---") and set(ln.strip()) == {"-"}:
            out.append("<hr>")
            i += 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            lv = len(m.group(1))
            out.append(f"<h{lv}>{md_inline(m.group(2))}</h{lv}>")
            i += 1
            continue
        if ln.lstrip().startswith("|") and i + 1 < len(lines) and re.match(
                r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            blk = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                blk.append(lines[i])
                i += 1
            out.append(md_table(blk))
            continue
        if re.match(r"^\s*[-*]\s+", ln):
            items, cur = [], None
            while i < len(lines) and (re.match(r"^\s*[-*]\s+", lines[i])
                                      or (lines[i].startswith("  ") and lines[i].strip() and cur is not None)):
                m2 = re.match(r"^\s*[-*]\s+(.*)$", lines[i])
                if m2:
                    if cur is not None:
                        items.append(cur)
                    cur = m2.group(1)
                else:
                    cur += " " + lines[i].strip()
                i += 1
            if cur is not None:
                items.append(cur)
            out.append("<ul>" + "".join(f"<li>{md_inline(x)}</li>" for x in items) + "</ul>")
            continue
        if re.match(r"^\s*\d+\.\s+", ln):
            items, cur = [], None
            while i < len(lines) and (re.match(r"^\s*\d+\.\s+", lines[i])
                                      or (lines[i].startswith("   ") and lines[i].strip() and cur is not None)):
                m2 = re.match(r"^\s*\d+\.\s+(.*)$", lines[i])
                if m2:
                    if cur is not None:
                        items.append(cur)
                    cur = m2.group(1)
                else:
                    cur += " " + lines[i].strip()
                i += 1
            if cur is not None:
                items.append(cur)
            out.append("<ol>" + "".join(f"<li>{md_inline(x)}</li>" for x in items) + "</ol>")
            continue
        if ln.startswith(">"):
            blk = []
            while i < len(lines) and lines[i].startswith(">"):
                blk.append(lines[i].lstrip("> ").rstrip())
                i += 1
            out.append(f"<blockquote>{md_inline(' '.join(blk))}</blockquote>")
            continue
        para = []
        while i < len(lines) and lines[i].strip() and not lines[i].lstrip().startswith(("|", "#", ">", "- ", "* ")) \
                and not re.match(r"^\s*\d+\.\s+", lines[i]) and not (lines[i].startswith("---") and set(lines[i].strip()) == {"-"}):
            para.append(lines[i].strip())
            i += 1
        if para:
            out.append(f"<p>{md_inline(' '.join(para))}</p>")
    return "\n".join(out)


def split_sections(md):
    """Split canonical md into '## N. Title' sections, keyed by number."""
    secs = {}
    parts = re.split(r"^## ", md, flags=re.M)
    for p in parts[1:]:
        title = p.split("\n", 1)[0].strip()
        body = p.split("\n", 1)[1] if "\n" in p else ""
        body = re.sub(r"\n---\s*$", "", body.rstrip())
        m = re.match(r"^(\d+)\.", title)
        if m:
            secs[int(m.group(1))] = (title, md_to_html(body))
    return secs


# ─────────────────────────────────────────────────────────── svg chart kit

def _t(x, y, s, fill=INK, size=13, anchor="start", weight="400", mono=False, op=1):
    fam = 'ui-monospace,SFMono-Regular,Menlo,monospace' if mono else 'inherit'
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
            f'text-anchor="{anchor}" font-weight="{weight}" font-family="{fam}" '
            f'opacity="{op}">{html.escape(str(s))}</text>')


def svg(w, h, body, cls="chart"):
    return (f'<svg class="{cls}" viewBox="0 0 {w} {h}" width="100%" '
            f'preserveAspectRatio="xMidYMid meet" role="img">{body}</svg>')


def bars_signed(rows, vmin, vmax, w=780, rowh=42, labw=210, fmt="{:+.3f}", title=None, ref=None):
    """rows: (label, value, color, note). Handles negative values with a zero line."""
    pad_t = 34 if title else 12
    h = pad_t + rowh * len(rows) + 30
    plot_x, plot_w = labw, w - labw - 96
    span = vmax - vmin

    def X(v):
        return plot_x + (v - vmin) / span * plot_w
    b = []
    if title:
        b.append(_t(0, 18, title, MUTED, 13, weight="600"))
    zx = X(0) if vmin <= 0 <= vmax else plot_x
    b.append(f'<line x1="{zx:.1f}" y1="{pad_t-4}" x2="{zx:.1f}" y2="{pad_t+rowh*len(rows)}" stroke="{LINE}" stroke-width="1.5"/>')
    if ref is not None:
        rx = X(ref[0])
        b.append(f'<line x1="{rx:.1f}" y1="{pad_t-4}" x2="{rx:.1f}" y2="{pad_t+rowh*len(rows)}" stroke="{WARN}" stroke-width="1" stroke-dasharray="4 4" opacity=".8"/>')
        b.append(_t(rx, pad_t + rowh * len(rows) + 18, ref[1], WARN, 12, anchor="middle"))
    for i, (lab, val, col, note) in enumerate(rows):
        y = pad_t + i * rowh
        cy = y + rowh / 2
        b.append(_t(labw - 12, cy + 5, lab, INK, 14, anchor="end"))
        x0, x1 = (zx, X(val)) if val >= 0 else (X(val), zx)
        b.append(f'<rect x="{x0:.1f}" y="{y+8:.1f}" width="{max(1.5,x1-x0):.1f}" height="{rowh-18}" fill="{col}" rx="3" opacity=".92"/>')
        vx = x1 + 8 if val >= 0 else x0 - 8
        an = "start" if val >= 0 else "end"
        b.append(_t(vx, cy + 5, fmt.format(val), col, 14, anchor=an, weight="700", mono=True))
        if note:
            b.append(_t(w - 2, cy + 5, note, MUTED, 12, anchor="end"))
    return svg(w, h, "".join(b))


def dumbbell(rows, vmin, vmax, w=780, rowh=54, labw=170, fmt="{:.3f}", title=None,
             c_from=BAD, c_to=BLUE):
    pad_t = 40 if title else 14
    h = pad_t + rowh * len(rows) + 16
    plot_x, plot_w = labw, w - labw - 70
    span = vmax - vmin

    def X(v):
        return plot_x + (v - vmin) / span * plot_w
    b = []
    if title:
        b.append(_t(0, 18, title, MUTED, 13, weight="600"))
    for i, (lab, v1, v2) in enumerate(rows):
        y = pad_t + i * rowh + rowh / 2
        b.append(_t(labw - 14, y + 5, lab, INK, 14, anchor="end"))
        x1, x2 = X(v1), X(v2)
        b.append(f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" stroke="{LINE}" stroke-width="3"/>')
        mid = (x1 + x2) / 2
        d = abs(v2 - v1)
        if abs(x2 - x1) > 46:
            b.append(f'<path d="M{mid-9:.1f} {y-11:.1f} L{mid+2:.1f} {y-11:.1f}" stroke="{MUTED}" stroke-width="1.4" marker-end="url(#ah)"/>')
            b.append(_t(mid, y - 17, f"Δ {d:.3f}", MUTED, 11.5, anchor="middle", mono=True))
        b.append(f'<circle cx="{x1:.1f}" cy="{y:.1f}" r="7.5" fill="{c_from}"/>')
        b.append(f'<circle cx="{x2:.1f}" cy="{y:.1f}" r="7.5" fill="{c_to}"/>')
        b.append(_t(x1, y + 26, fmt.format(v1), c_from, 12.5, anchor="middle", mono=True))
        b.append(_t(x2, y + 26, fmt.format(v2), c_to, 12.5, anchor="middle", weight="700", mono=True))
    defs = f'<defs><marker id="ah" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="{MUTED}"/></marker></defs>'
    return svg(w, h, defs + "".join(b))


def linechart(xs, series, ylim, w=760, h=330, xlab="", ylab="", fmt="{:.2f}", ref=None):
    pl, pr, pt, pb = 56, 118, 24, 46
    y0, y1 = ylim

    def X(i):
        return pl + i / max(1, len(xs) - 1) * (w - pl - pr)

    def Y(v):
        return h - pb - (v - y0) / (y1 - y0) * (h - pt - pb)
    b = []
    steps = 5
    for k in range(steps + 1):
        v = y0 + (y1 - y0) * k / steps
        yy = Y(v)
        b.append(f'<line x1="{pl}" y1="{yy:.1f}" x2="{w-pr}" y2="{yy:.1f}" stroke="{LINE}" stroke-width="1" opacity=".6"/>')
        b.append(_t(pl - 10, yy + 4, fmt.format(v), MUTED, 12, anchor="end", mono=True))
    for i, x in enumerate(xs):
        b.append(_t(X(i), h - pb + 22, x, INK, 13, anchor="middle", mono=True))
    if xlab:
        b.append(_t((pl + w - pr) / 2, h - 6, xlab, MUTED, 12.5, anchor="middle"))
    if ylab:
        b.append(_t(10, 14, ylab, MUTED, 12.5))
    if ref:
        rv, rl, rc = ref
        yy = Y(rv)
        b.append(f'<line x1="{pl}" y1="{yy:.1f}" x2="{w-pr}" y2="{yy:.1f}" stroke="{rc}" stroke-width="1.4" stroke-dasharray="5 5" opacity=".9"/>')
        b.append(_t(w - pr + 8, yy + 4, rl, rc, 12, weight="600"))
    for name, vals, col in series:
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))
        b.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="3" stroke-linejoin="round"/>')
        for i, v in enumerate(vals):
            b.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="5.5" fill="#0d1117" stroke="{col}" stroke-width="3"/>')
            b.append(_t(X(i), Y(v) - 14, f"{v:.2f}", col, 12, anchor="middle", mono=True))
        b.append(_t(w - pr + 8, Y(vals[-1]) + 5, name, col, 13, weight="700"))
    return svg(w, h, "".join(b))


def groupbars(cats, series, ymax, w=780, h=340, fmt="{:.1f}", ylab=""):
    pl, pr, pt, pb = 50, 12, 30, 64
    gw = (w - pl - pr) / len(cats)
    n = len(series)
    bw = min(30, (gw - 26) / n)

    def Y(v):
        return h - pb - v / ymax * (h - pt - pb)
    b = []
    for k in range(5):
        v = ymax * k / 4
        yy = Y(v)
        b.append(f'<line x1="{pl}" y1="{yy:.1f}" x2="{w-pr}" y2="{yy:.1f}" stroke="{LINE}" stroke-width="1" opacity=".55"/>')
        b.append(_t(pl - 8, yy + 4, f"{v:.0f}", MUTED, 11.5, anchor="end", mono=True))
    if ylab:
        b.append(_t(0, 14, ylab, MUTED, 12))
    for ci, c in enumerate(cats):
        gx = pl + ci * gw
        b.append(_t(gx + gw / 2, h - pb + 22, c, INK, 13, anchor="middle", weight="600"))
        for si, (name, vals, col) in enumerate(series):
            v = vals[ci]
            if v is None:
                continue
            x = gx + (gw - bw * n) / 2 + si * bw
            b.append(f'<rect x="{x:.1f}" y="{Y(v):.1f}" width="{bw-3:.1f}" height="{h-pb-Y(v):.1f}" fill="{col}" rx="2.5" opacity=".93"/>')
            b.append(_t(x + (bw - 3) / 2, Y(v) - 6, fmt.format(v), col, 11.5, anchor="middle", weight="700", mono=True))
    lx = pl
    for name, vals, col in series:
        b.append(f'<rect x="{lx}" y="{h-24}" width="11" height="11" fill="{col}" rx="2"/>')
        b.append(_t(lx + 16, h - 14, name, MUTED, 12))
        lx += 20 + len(name) * 6.6
    return svg(w, h, "".join(b))


# ─────────────────────────────────────────────────────────── slide helpers

def cards(items):
    out = ['<div class="cards">']
    for kicker, big, cap, tone in items:
        out.append(f'<div class="card t-{tone}"><div class="k">{html.escape(kicker)}</div>'
                   f'<div class="big">{big}</div><div class="cap">{cap}</div></div>')
    out.append("</div>")
    return "".join(out)


def ledger(rows):
    out = ['<table class="lg"><thead><tr><th>hypothesis</th><th>test</th><th>result</th></tr></thead><tbody>']
    for hyp, test, res, tone in rows:
        out.append(f'<tr class="r-{tone}"><td>{hyp}</td><td class="mut">{test}</td><td class="res">{res}</td></tr>')
    out.append("</tbody></table>")
    return "".join(out)


def stepper(steps):
    out = ['<div class="stepper">']
    for label, where, state in steps:
        icon = {"done": "✓", "blocked": "✕", "todo": "○"}[state]
        out.append(f'<div class="step s-{state}"><div class="ic">{icon}</div>'
                   f'<div class="sl">{label}<span class="sw">{where}</span></div></div>')
    out.append("</div>")
    return "".join(out)


SLIDES = []


def slide(kicker, headline, body, sowhat=None, src=None, note=None):
    SLIDES.append(dict(kicker=kicker, headline=headline, body=body,
                       sowhat=sowhat, src=src, note=note))


# ── 1 ─────────────────────────────────────────────────────────────────────
slide("status · 2026-08-06", "One bug is holding the whole project: Stage-2 GRPO trains with reward ≡ 0.",
      cards([
          ("blocked at", 'reward&nbsp;≡&nbsp;0', "LVR Stage-2 GRPO, job 221059 — 319 steps, no RL signal", "bad"),
          ("idle since", "2026-06-16", "~7 weeks. Debug knob committed, never run.", "warn"),
          ("repo tip", "b81ded9", "origin/main = branch <code>jobA2-ep3</code>; local main 16 behind", "blue"),
      ]) + '<p class="lede">Everything upstream of the blocker is complete and documented. '
      'Separately, three findings have <strong>reframed the thesis</strong> — the faithful baseline '
      'does not collapse, so “VICReg fixes latent collapse” is not viable as originally posed.</p>',
      sowhat="One cluster job — <code>STEPS=2 DEBUG_COMPLETIONS=1</code> — produces the one piece of evidence the CPU path cannot.",
      src=0)

# ── 2 ─────────────────────────────────────────────────────────────────────
TIMELINE = [
    ("05-02", "scaffold", "v0.1.0", OK),
    ("05-07", "Phase 0", "Monet probed: stage2 healthy, stage3 collapsed", BLUE),
    ("05-08", "Pivot A", "VICReg breaks collapse — recipe G", OK),
    ("05-20", "mechanism", "collapse = recurrence seed, not attention", BLUE),
    ("05-22", "dissociation 1", "geometry ≠ function", WARN),
    ("06-04", "dissociation 2", "faithful stage-3 doesn't collapse", WARN),
    ("06-05", "LVR added", "released 7B anchor reproduces paper", OK),
    ("06-09", "LVR Stage-1", "3B SFT done, eval'd", OK),
    ("06-16", "blocked", "Stage-2 GRPO reward ≡ 0", BAD),
]


def timeline_svg():
    w, h = 800, 250
    b = []
    y = 108
    b.append(f'<line x1="24" y1="{y}" x2="{w-24}" y2="{y}" stroke="{LINE}" stroke-width="2"/>')
    n = len(TIMELINE)
    for i, (d, t, desc, col) in enumerate(TIMELINE):
        x = 34 + i * (w - 68) / (n - 1)
        up = i % 2 == 0
        ty = y - 20 if up else y + 20
        b.append(f'<line x1="{x:.1f}" y1="{y}" x2="{x:.1f}" y2="{ty:.1f}" stroke="{col}" stroke-width="1.6" opacity=".7"/>')
        b.append(f'<circle cx="{x:.1f}" cy="{y}" r="6" fill="#0d1117" stroke="{col}" stroke-width="3"/>')
        ly = ty - 8 if up else ty + 16
        b.append(_t(x, ly, t, col, 13, anchor="middle", weight="700"))
        b.append(_t(x, ly + (-15 if up else 15), d, MUTED, 11.5, anchor="middle", mono=True))
        words, cur, ls = desc.split(), "", []
        for wd in words:
            if len(cur) + len(wd) > 17:
                ls.append(cur)
                cur = wd
            else:
                cur = (cur + " " + wd).strip()
        ls.append(cur)
        for j, l in enumerate(ls):
            yy = (ly - 30 - (len(ls) - 1 - j) * 13) if up else (ly + 28 + j * 13)
            b.append(_t(x, yy, l, MUTED, 11, anchor="middle"))
    return svg(w, h, "".join(b))


slide("orientation", "The whole project in one line — nine beats, four months.",
      timeline_svg() +
      '<p class="lede">Two faithful upstream reproductions run in parallel: <strong>Monet</strong> '
      '(arXiv 2511.21395) since May, <strong>LVR</strong> (arXiv 2509.24251) since June. '
      'The amber beats are where the science turned.</p>',
      sowhat="The May work answered <em>what causes collapse</em>; the June work found that collapse may not be the right target.",
      src=1)

# ── 3 ─────────────────────────────────────────────────────────────────────
def thesis_svg():
    w, h = 800, 250
    b = []

    def box(x, y, ww, hh, label, sub, col, dash=False):
        d = ' stroke-dasharray="5 4"' if dash else ""
        b.append(f'<rect x="{x}" y="{y}" width="{ww}" height="{hh}" rx="10" fill="#141b24" stroke="{col}" stroke-width="1.6"{d}/>')
        b.append(_t(x + ww / 2, y + hh / 2 - 2, label, INK, 13.5, anchor="middle", weight="700"))
        b.append(_t(x + ww / 2, y + hh / 2 + 16, sub, MUTED, 11.5, anchor="middle"))

    def arrow(x1, y1, x2, y2, col=MUTED):
        b.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="1.8" marker-end="url(#ar)"/>')
    b.append(f'<defs><marker id="ar" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="{MUTED}"/></marker></defs>')
    box(10, 60, 108, 62, "image", "", BLUE)
    arrow(122, 91, 154, 91)
    box(158, 60, 128, 62, "generator VLM", "Qwen2.5-VL, trained", VIO)
    arrow(290, 91, 322, 91)
    for k in range(4):
        yy = 46 + k * 26
        b.append(f'<rect x="330" y="{yy}" width="66" height="18" rx="4" fill="{OK}" opacity="{0.9-k*0.06:.2f}"/>')
    b.append(_t(363, 168, "K latents  h ∈ R^{K×D}", OK, 12.5, anchor="middle", mono=True))
    arrow(400, 91, 434, 91)
    box(438, 44, 150, 44, "frozen anchor A", "never trained on h", WARN, dash=True)
    box(438, 96, 150, 44, "frozen anchor B", "never trained on h", WARN, dash=True)
    arrow(592, 91, 624, 91)
    box(628, 60, 108, 62, "answer", "NLL measured", BLUE)
    b.append(_t(10, 206, "COLLAPSE — the failure mode across every phase:", BAD, 13, weight="700"))
    b.append(_t(10, 228, "all K slots become near-identical  →  mean off-diag cos → 0.85–0.99  →  the anchors learn nothing new", MUTED, 12.5))
    b.append(f'<rect x="600" y="192" width="196" height="44" rx="8" fill="#241417" stroke="{BAD}" stroke-width="1"/>')
    b.append(_t(698, 210, "compression_ratio is BLIND", BAD, 11.5, anchor="middle", weight="700"))
    b.append(_t(698, 226, "mean_off_diag_cos is the metric", MUTED, 11, anchor="middle"))
    return svg(w, h, "".join(b))


slide("the thesis", "Latents that <em>any</em> frozen reader can decode — that is the whole bet.",
      thesis_svg(),
      sowhat="If the K slots collapse into one, the objective is satisfiable without encoding anything transferable.",
      src=1)

# ── 4 ─────────────────────────────────────────────────────────────────────
slide("phase 0 · n=200", "Monet stage 2 is healthy. Stage 3 destroys reader-transfer.",
      '<div class="two">' +
      bars_signed([("stage 2  cos", 0.38, OK, "encoder-grounded"),
                   ("stage 3  cos", 0.87, BAD, "positions 4–7 collinear (≈1.00)")],
                  0, 1.0, w=760, fmt="{:.2f}", title="mean off-diagonal cosine  (lower = more diverse)") +
      bars_signed([("stage 2  utility", 2.19, OK, "n_helpful = 4"),
                   ("stage 3  utility", -0.13, BAD, "transfer fails")],
                  -0.6, 2.6, w=760, title="frozen-Qwen transfer utility (nat)") + "</div>",
      sowhat="Import the stage-2 pattern; skip stage-3 self-distillation. The student-path replication rules out an extraction artifact.",
      src=2)

# ── 5 ─────────────────────────────────────────────────────────────────────
slide("phases 1 – 2 · four negatives", "Four failed runs each removed a suspect — the loss and the attention mask were both exonerated.",
      bars_signed([("Ph1 · mean-MSE λ=1", 0.851, BAD, "MARGINAL 2/4"),
                   ("Ph1 · sum-MSE λ=1", 0.987, BAD, "~2048× stronger signal → WORSE"),
                   ("Ph1.5 · no 4D mask", 0.959, BAD, "Monet vendored class"),
                   ("Ph1.5b · +4D mask", 0.961, BAD, "identical to no-mask"),
                   ("Ph2 · full stage-2 recipe", 0.704, WARN, "in-dist; utility −5.63")],
                  0, 1.05, w=780, fmt="{:.3f}", title="mean off-diagonal cosine, 3B / 5K examples") +
      '<div class="two callouts">'
      f'<div class="co ok"><h4>The loss is exonerated</h4><p><code>v_roi</code> target cosine = <strong>0.465</strong> in both Phase-1 runs — '
      'the ROI targets are well-distributed, so <code>h</code>\'s collapse is not a property of the gather rule.</p></div>'
      f'<div class="co ok"><h4>Attention isolation is exonerated</h4><p>Injecting <code>attention_mask_4d</code> moved cos '
      '<strong>0.959 → 0.961</strong> — statistically identical. Cross-slot attention is not the load-bearing mechanism at 3B/5K.</p></div>'
      "</div>",
      sowhat="By elimination, the cause is representational, not architectural — which is what Pivot A then exploited.",
      src=3)

# ── 6 ─────────────────────────────────────────────────────────────────────
slide("pivot A · 2026-05-08", "Direct representation-level regularization breaks the collapse.",
      bars_signed([("C1  cos hinge τ=0.5", 0.725, BAD, "MARGINAL — hinge too weak"),
                   ("D1  VICReg + sum-MSE", 0.987, BAD, "MARGINAL — NTP starves at 3.21"),
                   ("C2  VICReg λ=1", 0.441, WARN, "PASS 3/4"),
                   ("E   D2 @ 2000 steps", 0.389, OK, "utility +0.088"),
                   ("G   K=4, 2000 steps", 0.372, OK, "utility +0.222  ← best"),
                   ("D2  VICReg λ=2", 0.341, OK, "PASS 3/4"),
                   ("F   D2 @ K=4", 0.311, OK, "utility +0.132 · tightest"),
                   ],
                  0, 1.05, w=780, fmt="{:.3f}",
                  title="mean off-diagonal cosine by variant",
                  ref=(0.375, "Monet stage 2 = 0.375")) +
      '<p class="lede">VICReg variance-hinge + dimension-decorrelation is the mechanism; <strong>mean</strong>-MSE '
      '(not sum-MSE) is the right operating form; λ_reg dose-responds smoothly. Recipe <strong>G</strong> lands at '
      'cos 0.372 — Monet stage 2\'s neighbourhood.</p>'
      f'<div class="co warn wide"><h4>The gap that motivated everything after</h4><p>G\'s transfer utility is '
      '<strong>+0.222</strong> against Monet stage 2\'s <strong>+2.19</strong> — roughly <strong>10×</strong>. '
      'Attributed to scale and data volume (3B/5K vs 7B/125K); closing it was the point of the cluster phase.</p></div>',
      sowhat="Geometry was solved at 3B. Utility was not — and §6–§8 explain why those are not the same problem.",
      src=4)

# ── 7 ─────────────────────────────────────────────────────────────────────
slide("mechanism · n=200", "Post-RL is stage 3, bit for bit. RL recovers nothing.",
      '<table class="lg mech"><thead><tr><th>stage</th><th>last-layer cos</th><th>util (monet_self)</th><th>util (qwen_base)</th></tr></thead><tbody>'
      f'<tr><td>stage 2</td><td class="num ok">0.369</td><td class="num ok">+2.717</td><td class="num ok">+2.145</td></tr>'
      f'<tr><td>stage 3</td><td class="num bad">0.860</td><td class="num">+0.234</td><td class="num bad">−0.128</td></tr>'
      f'<tr><td>rl</td><td class="num bad">0.861</td><td class="num">+0.118</td><td class="num bad">−0.131</td></tr>'
      "</tbody></table>"
      '<p class="lede">Identical at <em>every</em> diagnostic: cos curve ±0.005/layer, participation-ratio curve '
      '±0.005/layer, attention buckets ±0.02/layer, and the K×K cosine matrix <strong>bit-for-bit</strong>. '
      'A later CKA measurement puts rl ↔ stage3 at <strong>0.938</strong>.</p>',
      sowhat="Whatever stage 3 does to the latents, VLPO does not undo it — so the RL stage is not a fix for collapse.",
      src=5)

# ── 8 ─────────────────────────────────────────────────────────────────────
def seed_svg():
    w, h = 780, 240
    pl, pr, pt, pb = 60, 150, 30, 40

    def X(i):
        return pl + i * (w - pl - pr)

    def Y(v):
        return h - pb - v / 1.0 * (h - pt - pb)
    b = []
    for k in range(5):
        v = k / 4
        yy = Y(v)
        b.append(f'<line x1="{pl}" y1="{yy:.1f}" x2="{w-pr}" y2="{yy:.1f}" stroke="{LINE}" stroke-width="1" opacity=".55"/>')
        b.append(_t(pl - 10, yy + 4, f"{v:.2f}", MUTED, 11.5, anchor="end", mono=True))
    b.append(_t(pl, h - pb + 22, "L0  (recurrence seed)", MUTED, 12.5, anchor="middle"))
    b.append(_t(w - pr, h - pb + 22, "L27  (last layer)", MUTED, 12.5, anchor="middle"))
    b.append(_t(6, 16, "mean off-diag cos", MUTED, 12))
    for name, v0, v1, col in [("stage 3 / rl", 0.72, 0.88, BAD), ("stage 2", 0.32, 0.369, OK)]:
        b.append(f'<line x1="{X(0):.1f}" y1="{Y(v0):.1f}" x2="{X(1):.1f}" y2="{Y(v1):.1f}" stroke="{col}" stroke-width="3"/>')
        for i, v in ((0, v0), (1, v1)):
            b.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="6" fill="#0d1117" stroke="{col}" stroke-width="3"/>')
            b.append(_t(X(i) + (-12 if i == 0 else 12), Y(v) - 12, f"{v:.2f}", col, 12.5,
                        anchor="end" if i == 0 else "start", mono=True, weight="700"))
        b.append(_t(w - pr + 12, Y(v1) + 5, name, col, 13, weight="700"))
    b.append(f'<rect x="{X(0)-4:.1f}" y="{Y(0.72):.1f}" width="8" height="{Y(0.32)-Y(0.72):.1f}" fill="{WARN}" opacity=".28"/>')
    b.append(_t(X(0) + 14, (Y(0.72) + Y(0.32)) / 2 + 4, "≈ 2/3 of the gap is already set at L0", WARN, 12.5, weight="600"))
    return svg(w, h, "".join(b))


slide("mechanism · locus", "The collapse is written at the <em>seed</em>, then amplified by depth — it is not attention copying.",
      seed_svg() +
      '<div class="two callouts">'
      f'<div class="co bad"><h4>Refuted: attention copying</h4><p>The naive hypothesis fails outright — stage 2 has '
      '<strong>higher</strong> mean <code>latent_prior</code> attention than stage3/rl at most layers '
      '(peak <strong>0.46 vs 0.06</strong> at L2).</p></div>'
      f'<div class="co blue"><h4>The real locus: the output side</h4><p>stage3/rl write nearly identical residual updates at '
      'every slot <em>regardless of attention pattern</em>. Each slot\'s output becomes the next slot\'s seed — '
      'self-reinforcing recurrence at the weight level.</p></div>'
      "</div>",
      sowhat="This is exactly what a VICReg term acts on — output diversity — which is why Pivot A was the right lever.",
      src=5)

# ── 9 ─────────────────────────────────────────────────────────────────────
slide("dissociation 1 · n=500", "The cosine signature is causally movable — and moving it changes nothing.",
      '<div class="two">' +
      dumbbell([("stage 3", 0.864, 0.329), ("rl", 0.866, 0.323)], 0.0, 1.0, w=380, labw=88,
               title="mean off-diag cos — before → after vicreg_project", c_from=BAD, c_to=BLUE) +
      dumbbell([("stage 3", -0.115, -0.117), ("rl", -0.117, -0.137)], -0.30, 0.10, w=380, labw=88,
               fmt="{:+.3f}", title="qwen_base transfer utility (nat) — same runs", c_from=BAD, c_to=BLUE) +
      "</div>"
      '<p class="lede">SVD-orthogonalising the K recurrence seeds at inference time and re-injecting them '
      'halves the cosine. The representation genuinely rotates — CKA(stage3, projected) = <strong>0.518</strong> — '
      'yet the trained model writes the same per-slot residuals anyway. <span class="mut">CKA rl ↔ stage3 = 0.938 · stage2 ↔ stage3 = 0.383.</span></p>',
      sowhat="Matching stage 2's geometry is <strong>necessary but not sufficient</strong>. Cosine is the trace, not the cause, of utility.",
      src=6)

# ── 10 ────────────────────────────────────────────────────────────────────
slide("dissociation 2 · 2026-06-04", "The collapsed baseline the whole A/B needed <em>does not exist</em>.",
      '<div class="two">' +
      bars_signed([("released stage2 (teacher)", 0.377, OK, ""),
                   ("ours, epochs=2 (step1500)", 0.420, OK, ""),
                   ("ours, epochs=3 (step2913)", 0.401, OK, ""),
                   ("released stage3 (target)", 0.871, BAD, "")],
                  0, 1.0, w=760, fmt="{:.3f}", title="mean off-diag cos") +
      bars_signed([("released stage2", 2.05, OK, ""),
                   ("ours, epochs=2", 0.90, OK, ""),
                   ("ours, epochs=3", 1.02, OK, ""),
                   ("released stage3", -0.05, BAD, "")],
                  -0.4, 2.4, w=760, fmt="{:+.2f}", title="qwen_base utility (nat)") + "</div>"
      '<p class="lede">Git archaeology pointed at epochs (models uploaded at <code>f49f382</code> when the script said '
      '<code>--epochs 3</code>) and <strong>all training code is byte-identical to HEAD — a 0-line diff</strong>. '
      'The epochs hypothesis was then run directly and <strong>also falsified</strong>: epochs 2 <em>and</em> 3 both land at ~0.40.</p>'
      f'<div class="co warn wide"><h4>Mechanism, now self-consistent</h4><p>Stage 3\'s all-layer alignment loss makes the student match '
      'the <em>teacher\'s</em> geometry. The released stage-2 teacher is healthy (0.377), so a faithful stage 3 converges to ~0.40 and '
      '<strong>stays healthy and useful</strong>. The released collapse is unreproducible from the public recipe + public checkpoints.</p></div>',
      sowhat="“VICReg rescues Monet's collapse” is not viable as posed. Collapse is real but <em>config-dependent</em> — our deviated stage-2 repro hit 0.840 / −5.35.",
      src=7)

# ── 11 ────────────────────────────────────────────────────────────────────
def equals_svg():
    w, h = 800, 210
    b = []

    def panel(x, title, cos, cosc, score, sub):
        b.append(f'<rect x="{x}" y="16" width="300" height="150" rx="12" fill="#141b24" stroke="{LINE}"/>')
        b.append(_t(x + 150, 44, title, INK, 14, anchor="middle", weight="700"))
        b.append(_t(x + 150, 70, f"latent cos {cos}", cosc, 13, anchor="middle", mono=True))
        b.append(_t(x + 150, 118, score, INK, 38, anchor="middle", weight="800", mono=True))
        b.append(_t(x + 150, 144, sub, MUTED, 12, anchor="middle"))
    panel(20, "our ep3 stage 3  (healthy)", "0.401", OK, "148 / 191", "direct_attr .800 · rel_position .737")
    panel(480, "released stage 3  (collapsed)", "0.871", BAD, "148 / 191", "direct_attr .791 · rel_position .750")
    b.append(_t(400, 104, "=", WARN, 56, anchor="middle", weight="800"))
    b.append(_t(400, 138, "V*Bench, K=8", MUTED, 12, anchor="middle"))
    b.append(_t(400, 190, "a 0.47 difference in internal geometry buys exactly zero benchmark points", MUTED, 12.5, anchor="middle"))
    return svg(w, h, "".join(b))


slide("dissociation 3a", "Opposite geometry, identical score — 148/191 both.",
      equals_svg(),
      sowhat="Only one question shifts between splits. On a real downstream task, the collapse is invisible at the trained K.",
      src=8)

# ── 12 ────────────────────────────────────────────────────────────────────
slide("dissociation 3b", "…but the collapsed model is the only one that scales with test-time K.",
      linechart(["K=8", "K=10", "K=12", "K=16"],
                [("released SFT\n(collapsed)", [77.49, 78.53, 77.49, 80.63], BAD),
                 ("our ep3\n(healthy)", [77.49, 77.49, 75.92, 77.49], OK)],
                (74, 84), w=760, h=330, ylab="V* Overall (%)", fmt="{:.0f}",
                ref=(82.20, "paper best-K 82.20", WARN)) +
      '<p class="lede">Our repro is <strong>K-flat</strong> — pinned at 148/191, no benefit from extra latent slots. '
      'The released collapsed model gains <strong>+3.1</strong> at K=16. Whatever produced the released collapse '
      '<em>also</em> gave it K-extrapolation; our faithful repro has neither.</p>',
      sowhat="Since K=8 is optimal for our checkpoint, the paper-suite gap on the next slide is real — not a K-protocol artifact.",
      src=8)

# ── 13 ────────────────────────────────────────────────────────────────────
slide("monet paper suite · K=8", "On the paper's own suite we sit at the Qwen baseline, below reported SFT.",
      groupbars(["V*", "HRBench4K", "HRBench8K", "MME-RW-Lite"],
                [("ours ep3 K=8", [77.49, 62.1, 59.1, 41.3], BLUE),
                 ("paper Monet-SFT", [82.20, 68.50, 66.00, 52.68], WARN),
                 ("paper Monet-7B RL", [83.25, 71.00, 68.00, 55.50], VIO),
                 ("paper Qwen2.5-VL-7B", [76.44, 68.00, 63.75, 45.75], MUTED)],
                90, w=790, h=340, ylab="score") +
      '<div class="two callouts">'
      f'<div class="co ok"><h4>The clean result</h4><p>On the one parser-clean benchmark, ep3 equals released SFT '
      '<em>exactly</em> at K=8 — V* <strong>77.49</strong>. Faithful repro confirmed.</p></div>'
      f'<div class="co warn"><h4>Three confounds before calling it a gap</h4><p>(1) K — ours fixed 8, paper best-K. '
      '(2) harness residual ~1.6 pt. (3) parser — ep3 emits <strong>0% <code>\\boxed{}</code></strong>; MME-RW scored '
      '30.0 by VLMEvalKit vs <strong>41.3 manual</strong> (16.8% unparseable; the bar above uses the manual figure).</p></div>'
      "</div>",
      sowhat="The paper's headline gains come from RL (VLPO), which our checkpoint does not have.",
      src=9)

# ── 14 ────────────────────────────────────────────────────────────────────
slide("LVR · anchor eval", "Before training anything, the released LVR-7B reproduced the paper to ≤0.6 pt.",
      '<table class="lg anchor"><thead><tr><th>bench</th><th>steps</th><th>ours</th><th>paper</th><th>Δ</th></tr></thead><tbody>'
      '<tr><td rowspan="3">V*</td><td class="mono">4</td><td class="num">80.63</td><td class="num mut">81.2</td><td class="num ok">−0.57</td></tr>'
      '<tr><td class="mono">8</td><td class="num strong">81.68</td><td class="num mut">81.7</td><td class="num ok">−0.02</td></tr>'
      '<tr><td class="mono">16</td><td class="num">80.63</td><td class="num mut">80.6</td><td class="num ok">+0.03</td></tr>'
      '<tr><td rowspan="3">MMVP</td><td class="mono">4</td><td class="num">72.00</td><td class="num mut">72.0</td><td class="num ok">0.00</td></tr>'
      '<tr><td class="mono">8</td><td class="num strong">72.00</td><td class="num mut">71.7</td><td class="num ok">+0.30</td></tr>'
      '<tr><td class="mono">16</td><td class="num">71.67</td><td class="num mut">71.7</td><td class="num ok">−0.03</td></tr>'
      "</tbody></table>"
      '<p class="lede">The eval harness and the custom steps-decoding path are faithful <em>before</em> any of our own '
      'training entered the picture. <span class="mut">BLINK failed on a <code>datasets==3.5.1</code> feature-type '
      'incompatibility — a loader-version issue, not LVR.</span></p>',
      sowhat="Same discipline paid off on Monet: released RL @K=10 → 81.68 and released SFT @K=16 → 80.63, both within ~1.6 pt of the paper.",
      src=10)

# ── 15 ────────────────────────────────────────────────────────────────────
DEFECTS = [
    ("flash-attn version", "version drift", "README installs unversioned → resolves to 2.8.3 (new C++ ABI), won't link torch 2.6.0",
     "Inferred <strong>2.7.4.post1</strong> — shipped the same day as torch 2.6.0 (2025-01-29). Pinned; verified working.", BAD),
    ("Python dev headers", "unpublished env", "triton's runtime JIT needs <code>Python.h</code>; cluster python3.11 has none",
     "Build the venv on a uv-managed standalone CPython.", WARN),
    ("PYTHONPATH", "unpublished env", "code mixes <code>from src…</code> and <code>from train…</code>; neither is on the path",
     "<code>PYTHONPATH=$REPO:$REPO/src</code> in both stage launchers.", WARN),
    ("lvr_sft_dataset_packed_fixedToken.py", "missing file", "imported by <code>src/dataset/__init__.py</code> but <strong>never committed</strong> (404) — the repo can't even import",
     "Stub that provides the symbol and raises if actually invoked (the faithful recipe never calls it).", BAD),
    ("s3_checkpoints_lvr.py", "upstream defect", "truncated — ends on a bare <code>if __name__ == \"__main__\":</code> with no body",
     "Appended <code>pass</code>.", BAD),
    ("av / huggingface-hub pins", "inconsistent pins", "<code>av==14.3.0</code> yanked &amp; sdist-only; <code>hub==0.30.2</code> contradicts <code>transformers==4.54.0</code>",
     "uv <code>--override</code> → av 17.0.1, hub ≥0.34; requirements.txt kept byte-identical.", WARN),
    ("Oracle-Cloud credentials", "security", "hard-coded creds shipped in the repo",
     "Removed; <code>--online_checkpoint False</code> + boto3 needs no code edit.", VIO),
    ("qwen-vl-utils", "omitted dep", "imported by the trainer, absent from requirements.txt",
     "Added to the env build.", WARN),
]


def defect_grid():
    out = ['<div class="dgrid">']
    for i, (name, tag, prob, fix, col) in enumerate(DEFECTS):
        out.append(
            f'<div class="dc"><div class="dh"><span class="dn">{i+1}</span>'
            f'<span class="dname">{name}</span><span class="dtag" style="color:{col};border-color:{col}55">{tag}</span></div>'
            f'<p class="dp"><span class="lbl bad">problem</span>{prob}</p>'
            f'<p class="dp"><span class="lbl ok">fix</span>{fix}</p></div>')
    out.append("</div>")
    return "".join(out)


slide("LVR · reproducibility", "The published repo does not run as-is — ten distinct defects stood between the paper and execution.",
      defect_grid() +
      '<p class="lede mut">Plus grad_accum 8→16 to hold effective batch 64 on 4 GPUs, and git-based transfer '
      '(vendored into <code>upstreams/lvr/</code>, no rsync). The method, hyperparameters and data were all '
      '<em>explicit</em>; only the environment was not.</p>',
      sowhat="This ledger is the reusable artifact — <code>docs/lvr/lvr_reproduction.md</code> separates what the paper specified from what we had to infer.",
      src=10)

# ── 16 ────────────────────────────────────────────────────────────────────
slide("LVR · pipeline", "Five of seven steps done. Step 5 is where everything stopped.",
      stepper([("0 · vendor repo + git transfer", "local → git pull", "done"),
               ("1 · build training env", "cpu-short", "done"),
               ("2 · stage data — Visual-CoT 142 GB + ViRL39K", "cpu-short", "done"),
               ("3 · Stage-1 SFT, 2500 steps", "gpu-4farm 4×H100 · job 219711 · ~16h", "done"),
               ("4 · upload ckpt + eval = RL baseline", "local A6000", "done"),
               ("5 · Stage-2 GRPO_latent RL", "gpu-4farm · job 221059", "blocked"),
               ("6 · upload RL ckpt + eval; report SFT→RL gain", "local A6000", "todo")]),
      sowhat="The pipeline is otherwise proven end-to-end; only the RL reward path is unresolved.",
      src=11)

# ── 17 ────────────────────────────────────────────────────────────────────
slide("LVR · Stage-1 SFT", "The 3B SFT baseline trained clean and landed in the paper's 3B-RL target band.",
      '<div class="two">' +
      groupbars(["steps 4", "steps 8", "steps 16"],
                [("our 3B SFT — V*", [65.97, 65.45, 65.45], BLUE),
                 ("our 3B SFT — MMVP", [55.67, 56.33, 57.33], VIO)],
                90, w=390, h=300, ylab="score") +
      f'<div class="statcol">'
      f'<div class="co ok"><h4>Training</h4><p>job 219711, 4×H100, ~16 h, clean exit on all 4 ranks. 2500 steps, epoch 1.0.<br>'
      f'<code>loss_total</code> 7.5 → <strong>0.37</strong><br><code>loss_ce</code> 5.5 → <strong>0.21</strong><br>'
      f'<code>loss_lvr</code> ~20 → <strong>1.65</strong></p></div>'
      f'<div class="co blue"><h4>Where it sits</h4><p>Paper 3B-RL targets are V* <strong>~65–67</strong>, MMVP <strong>~55–58</strong> — '
      'the SFT baseline is already inside both bands. Released 7B anchor is 81.68 / 72.00; 3B &lt; 7B as expected, '
      '<strong>no collapse</strong>. V* is K-flat at ~66; MMVP edges up with K.</p></div></div>'
      "</div>",
      sowhat="This is the number Stage-2 RL has to beat — and the reason the reward bug matters rather than being cosmetic.",
      src=11)

# ── 18 ────────────────────────────────────────────────────────────────────
slide("the blocker", "Every cheap explanation is eliminated. What remains is GPU-only — and unobserved.",
      ledger([
          ("<code>skip_special_tokens=True</code> strips the <code>&lt;lvr&gt;</code> markers",
           "checked token flags on checkpoint-2500", "<b>special=False</b> → not stripped", "ok"),
          ("the reward functions / regex are broken",
           "CPU fp32 diagnosis, job 221327", "<b>format_reward = 1.0</b>, accuracy 1/4", "ok"),
          ("the SFT model emits malformed completions",
           "same CPU run", "well-formed → <b>model fine</b>", "ok"),
          ("bf16 (the rollout dtype) breaks the format",
           "<code>--dtype bf16</code> at n=16", "only <b>1/16</b> degraded → mild", "ok"),
          ("sampling config drift",
           "compared against HF defaults", "top_p=1, top_k=50, rep_pen=1 → <b>matches</b>", "ok"),
          ("DeepSpeed / multi-GPU specific",
           "cannot be reproduced on CPU", "<b>UNTESTED — the open lead</b>", "bad"),
      ]) +
      '<div class="two callouts">'
      f'<div class="co bad"><h4>The signature</h4><p><code>format_reward = 0</code> for <em>all</em> completions at '
      '<em>every</em> step — systematic, not random. Mean completion length <strong>45 on GPU vs 22 on CPU</strong>. '
      'The regex is <code>^&lt;|lvr_start|&gt;.*?&lt;|lvr_end|&gt;\\s*&lt;answer&gt;.*?&lt;/answer&gt;$</code>.</p></div>'
      f'<div class="co ok"><h4>The instrument already exists</h4><p>Commit <code>b81ded9</code> added '
      '<code>DEBUG_COMPLETIONS=1</code>: a short full-init run that sets <code>DEBUG_MODE=true</code> + '
      '<code>LOG_PATH</code> so <code>accuracy_reward</code> dumps every completion and solution to a file. '
      '<strong>It has never been run.</strong></p></div>'
      "</div>",
      sowhat="Six transformers-4.54.0 drift fixes already went in around this path — the remaining failure is not a version issue.",
      src=12)

# ── 19 ────────────────────────────────────────────────────────────────────
slide("hygiene", "A lot of finished science exists only on one disk.",
      '<div class="two">'
      f'<div class="co bad"><h4>Untracked — no backup anywhere</h4><ul class="tight">'
      '<li><code>docs/overnight_2026_05_{20,22,24}/</code> — the mechanism + dissociation reports</li>'
      '<li><code>phase0_monet_probe/mech{,2,3}/</code> — probe harnesses and raw latents</li>'
      '<li><code>eval_local/</code>, <code>lvr_eval/</code> — every benchmark result</li>'
      '<li><code>configs/interleaved_*</code>, <code>scripts/overnight_*.sh</code>, <code>docs/lvr/</code></li>'
      "</ul></div>"
      f'<div class="co warn"><h4>Branch state</h4><p><code>origin/main</code> = <strong>b81ded9</strong> (branch '
      '<code>jobA2-ep3</code>, pushed). Local <code>main</code> is <strong>16 commits behind</strong>; the live work '
      'sits in the worktree.</p><p><code>README.md</code> still reads “v0.1.0 — scaffold”, ~3 months stale.</p>'
      '<p class="mut">Standing rules: cluster jobs are never agent-submitted; no login-node compute; CPU preflight before a GPU slot.</p></div>'
      "</div>",
      sowhat="Three of the four headline findings live in files that git has never seen.",
      src=13)

# ── 20 ────────────────────────────────────────────────────────────────────
slide("next", "Three moves, in order.",
      '<div class="nextlist">'
      '<div class="nx"><span class="n">1</span><div><h4>Unblock Stage-2</h4>'
      '<p>Run <code>STEPS=2 DEBUG_COMPLETIONS=1 sb lvr_stage2_3b</code> on the cluster (user-submitted) and read the '
      'dumped GPU rollout text — the one piece of evidence the CPU path cannot produce.</p></div></div>'
      '<div class="nx"><span class="n">2</span><div><h4>Decide the thesis framing</h4>'
      '<p>Given that the faithful baseline does not collapse, “VICReg rescues Monet stage-3 collapse” needs restating. '
      'Live alternatives: the <strong>K-scaling result</strong> as the real phenomenon, or <strong>LVR SFT→RL</strong> '
      'as the primary contribution.</p></div></div>'
      '<div class="nx"><span class="n">3</span><div><h4>Back up the untracked results</h4>'
      '<p>Commit or archive the directories in §13 before anything else touches that disk.</p></div></div>'
      "</div>",
      sowhat="Move 1 is a single job. Move 3 is a single commit. Move 2 is the only one that needs you.",
      src=14)


# ─────────────────────────────────────────────────────────── assemble

CSS = """
*{box-sizing:border-box}
html,body{margin:0;height:100%}
body{background:#0d1117;color:#e6edf3;font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;overflow:hidden}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.88em;background:#1b2230;color:#e5c07b;padding:1px 5px;border-radius:5px}
strong{color:#fff;font-weight:700}
em{color:#e5c07b;font-style:normal}
.mut,.mut *{color:#9aa7b4}
a{color:#79c0ff}

/* ── deck chrome ── */
#deck{height:100vh;display:flex;flex-direction:column}
#bar{display:flex;align-items:center;gap:14px;padding:9px 18px;border-bottom:1px solid #2a3441;background:#0f1620;flex:0 0 auto}
#bar .title{font-weight:700;font-size:14px;color:#e5c07b;letter-spacing:.01em}
#bar .sub{font-size:12.5px;color:#9aa7b4}
#bar .spacer{flex:1}
#bar button{background:#161d27;color:#cdd6e0;border:1px solid #2a3441;border-radius:7px;padding:5px 11px;font-size:12.5px;cursor:pointer;font-family:inherit}
#bar button:hover{background:#1d2632;color:#fff}
#bar .cnt{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;color:#9aa7b4;min-width:52px;text-align:right}
#prog{height:2px;background:#2a3441;flex:0 0 auto}
#prog i{display:block;height:100%;background:#e5c07b;transition:width .25s}
#stage{flex:1;overflow:auto;scroll-behavior:smooth}
.slide{min-height:100%;padding:26px 40px 60px;max-width:1080px;margin:0 auto;display:none;animation:fade .25s ease}
.slide.on{display:block}
@keyframes fade{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.kicker{font-size:12px;letter-spacing:.09em;text-transform:uppercase;color:#e5c07b;font-weight:700;margin-bottom:7px}
h1.head{font-size:29px;line-height:1.24;margin:0 0 20px;letter-spacing:-.015em;font-weight:750;max-width:26ch;max-width:none}
.lede{font-size:16.5px;color:#cdd6e0;margin:18px 0 0;max-width:88ch}
.sowhat{margin-top:24px;padding:13px 16px 13px 15px;border-left:3px solid #e5c07b;background:#1a1710;border-radius:0 9px 9px 0;font-size:15.5px}
.sowhat b{color:#e5c07b;font-size:11.5px;letter-spacing:.09em;text-transform:uppercase;display:block;margin-bottom:3px;font-weight:700}
.srcbtn{margin-top:16px;display:inline-flex;align-items:center;gap:7px;font-size:12.5px;color:#9aa7b4;background:#141b24;border:1px solid #2a3441;border-radius:999px;padding:5px 13px;cursor:pointer;font-family:inherit}
.srcbtn:hover{color:#e6edf3;border-color:#3d4a5c}

/* ── content bits ── */
.chart{display:block;margin:6px 0 4px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:22px;align-items:start;margin-top:4px}
.two.callouts{margin-top:20px}
.co{background:#141b24;border:1px solid #2a3441;border-left-width:3px;border-radius:9px;padding:13px 16px}
.co.wide{margin-top:18px}
.co h4{margin:0 0 6px;font-size:14.5px}
.co p{margin:5px 0;font-size:14.5px;color:#cdd6e0}
.co.ok{border-left-color:#56d364}.co.ok h4{color:#56d364}
.co.bad{border-left-color:#ff7b72}.co.bad h4{color:#ff7b72}
.co.warn{border-left-color:#e5c07b}.co.warn h4{color:#e5c07b}
.co.blue{border-left-color:#79c0ff}.co.blue h4{color:#79c0ff}
.statcol{display:flex;flex-direction:column;gap:14px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px;margin:8px 0 4px}
.card{background:#141b24;border:1px solid #2a3441;border-radius:12px;padding:16px 18px}
.card .k{font-size:11.5px;letter-spacing:.08em;text-transform:uppercase;color:#9aa7b4;font-weight:700}
.card .big{font-size:33px;font-weight:800;letter-spacing:-.02em;margin:6px 0 4px;font-family:ui-monospace,Menlo,monospace}
.card .cap{font-size:13.5px;color:#cdd6e0;line-height:1.45}
.card.t-bad{border-color:#ff7b7255;background:#1c1416}.card.t-bad .big{color:#ff7b72}
.card.t-warn{border-color:#e5c07b55;background:#1c1810}.card.t-warn .big{color:#e5c07b}
.card.t-blue{border-color:#79c0ff44;background:#101821}.card.t-blue .big{color:#79c0ff}
table.lg{border-collapse:collapse;width:100%;margin:10px 0;font-size:14.5px}
table.lg th{text-align:left;padding:9px 12px;border-bottom:1px solid #2a3441;color:#9aa7b4;font-weight:600;font-size:12.5px;letter-spacing:.05em;text-transform:uppercase}
table.lg td{padding:9px 12px;border-bottom:1px solid #1c242e;vertical-align:top}
table.lg .num{font-family:ui-monospace,Menlo,monospace;text-align:right;font-weight:700}
table.lg .mono{font-family:ui-monospace,Menlo,monospace}
table.lg .num.ok{color:#56d364}table.lg .num.bad{color:#ff7b72}table.lg .num.strong{color:#e5c07b}
table.lg .res{font-weight:600}
table.lg tr.r-ok .res{color:#56d364}table.lg tr.r-bad .res{color:#ff7b72}
table.lg td.mut{color:#9aa7b4}
table.lg.mech td:first-child{color:#e5c07b;font-family:ui-monospace,Menlo,monospace}
table.lg.anchor td{border-bottom:1px solid #1c242e}
.dgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:12px;margin-top:6px}
.dc{background:#141b24;border:1px solid #2a3441;border-radius:10px;padding:11px 13px}
.dh{display:flex;align-items:center;gap:8px;margin-bottom:6px;flex-wrap:wrap}
.dn{display:inline-grid;place-items:center;width:21px;height:21px;border-radius:6px;background:#1c2533;color:#e5c07b;font-size:12px;font-weight:800;flex:0 0 auto}
.dname{font-family:ui-monospace,Menlo,monospace;font-size:13px;color:#e6edf3;font-weight:600;flex:1}
.dtag{font-size:10.5px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;border:1px solid;border-radius:999px;padding:2px 7px}
.dp{margin:4px 0;font-size:13.4px;color:#cdd6e0;line-height:1.5}
.lbl{display:inline-block;font-size:10px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;padding:1px 6px;border-radius:4px;margin-right:7px;vertical-align:1px}
.lbl.bad{background:#3a1d1d;color:#ff9a9a}.lbl.ok{background:#16301f;color:#56d364}
.stepper{margin-top:6px}
.step{display:flex;align-items:center;gap:14px;padding:11px 14px;border:1px solid #2a3441;border-radius:10px;margin-bottom:8px;background:#141b24}
.step .ic{width:26px;height:26px;border-radius:50%;display:grid;place-items:center;font-weight:800;font-size:14px;flex:0 0 auto}
.step .sl{font-size:15px;font-weight:600}
.step .sw{display:block;font-size:12.5px;color:#9aa7b4;font-weight:400;font-family:ui-monospace,Menlo,monospace}
.step.s-done{opacity:.72}.step.s-done .ic{background:#16301f;color:#56d364}
.step.s-blocked{border-color:#ff7b7288;background:#1c1416}.step.s-blocked .ic{background:#3a1d1d;color:#ff7b72}
.step.s-blocked .sl{color:#ff7b72}
.step.s-todo .ic{background:#1c2533;color:#9aa7b4}.step.s-todo{opacity:.62}
.nextlist{margin-top:10px}
.nx{display:flex;gap:16px;padding:16px 18px;background:#141b24;border:1px solid #2a3441;border-radius:11px;margin-bottom:12px}
.nx .n{display:inline-grid;place-items:center;width:30px;height:30px;border-radius:9px;background:#1c2533;color:#e5c07b;font-weight:800;flex:0 0 auto}
.nx h4{margin:2px 0 5px;font-size:16px}
.nx p{margin:0;font-size:14.8px;color:#cdd6e0}
ul.tight{margin:6px 0;padding-left:19px}ul.tight li{margin:3px 0;font-size:14px}

/* ── overview ── */
#ov{position:fixed;inset:0;background:#0d1117f2;backdrop-filter:blur(3px);z-index:40;overflow:auto;padding:26px;display:none}
#ov.on{display:block}
#ov h3{margin:0 0 16px;font-size:15px;color:#e5c07b;letter-spacing:.05em;text-transform:uppercase}
.ovg{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px;max-width:1180px;margin:0 auto}
.ovc{background:#141b24;border:1px solid #2a3441;border-radius:10px;padding:12px 14px;cursor:pointer;transition:.15s}
.ovc:hover{border-color:#e5c07b;transform:translateY(-2px)}
.ovc .i{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:#9aa7b4}
.ovc .k{font-size:10.5px;letter-spacing:.07em;text-transform:uppercase;color:#e5c07b;font-weight:700;margin:4px 0 5px}
.ovc .h{font-size:13.5px;line-height:1.4;color:#e6edf3}
.ovc.cur{border-color:#e5c07b;background:#1a1710}

/* ── source pane ── */
#src{position:fixed;top:0;right:0;bottom:0;width:min(720px,92vw);background:#0f1620;border-left:1px solid #2a3441;z-index:50;transform:translateX(101%);transition:transform .22s ease;display:flex;flex-direction:column}
#src.on{transform:none}
#srch{display:flex;align-items:center;gap:12px;padding:12px 18px;border-bottom:1px solid #2a3441;flex:0 0 auto}
#srch .t{font-size:13px;color:#e5c07b;font-weight:700;letter-spacing:.04em;text-transform:uppercase}
#srch .f{font-size:12px;color:#9aa7b4;font-family:ui-monospace,Menlo,monospace}
#srch .spacer{flex:1}
#srch button{background:#161d27;color:#cdd6e0;border:1px solid #2a3441;border-radius:7px;padding:4px 10px;cursor:pointer;font-size:12.5px;font-family:inherit}
#srcb{overflow:auto;padding:8px 24px 60px;font-size:15px}
#srcb h2,#srcb h3{color:#e5c07b;margin:18px 0 8px;font-size:16.5px}
#srcb h3{font-size:15px;color:#cdd6e0}
#srcb p,#srcb li{color:#cdd6e0;font-size:14.6px}
#srcb table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13.6px}
#srcb th,#srcb td{border:1px solid #2a3441;padding:6px 9px}
#srcb th{background:#161d27;color:#e6edf3}
#srcb blockquote{border-left:3px solid #e5c07b;margin:10px 0;padding:4px 14px;color:#cdd6e0;background:#161d27}
#srcb hr{border:none;border-top:1px solid #2a3441;margin:18px 0}
.veil{position:fixed;inset:0;background:#0006;z-index:45;display:none}
.veil.on{display:block}

/* ── staleness banner ── */
#stale{display:none;position:fixed;left:0;right:0;bottom:0;z-index:60;background:#3a1d1d;border-top:1px solid #ff7b72;color:#ffd7d5;padding:10px 18px;font-size:13.5px;text-align:center}
#stale.on{display:block}
#stale b{color:#ff7b72}

@media (max-width:820px){
 .slide{padding:20px 18px 70px}
 h1.head{font-size:23px}
 .two{grid-template-columns:1fr;gap:14px}
 #bar{gap:9px;padding:8px 12px}
 #bar .sub,#bar .title{display:none}
 .cards{grid-template-columns:1fr}
 .lede{font-size:15.5px}
}
"""

JS_TMPL = """
const N = %(n)d;
let cur = 0, ovOpen = false, srcOpen = false;
const slides = [...document.querySelectorAll('.slide')];
const stage = document.getElementById('stage');
const SRC = %(srcmap)s;

function go(i, push=true){
  cur = Math.max(0, Math.min(N-1, i));
  slides.forEach((s,k)=>s.classList.toggle('on', k===cur));
  document.getElementById('cnt').textContent = (cur+1)+' / '+N;
  document.querySelector('#prog i').style.width = ((cur+1)/N*100)+'%%';
  stage.scrollTop = 0;
  [...document.querySelectorAll('.ovc')].forEach((c,k)=>c.classList.toggle('cur',k===cur));
  if(push) history.replaceState(null,'','#/'+(cur+1));
  if(srcOpen) loadSrc();
}
function loadSrc(){
  const id = slides[cur].dataset.src;
  const s = SRC[id] || {title:'—', html:'<p class="mut">No source section for this view.</p>'};
  document.getElementById('srct').textContent = s.title;
  document.getElementById('srcb').innerHTML = s.html;
  document.getElementById('srcb').scrollTop = 0;
}
function toggleSrc(force){
  srcOpen = (force===undefined) ? !srcOpen : force;
  document.getElementById('src').classList.toggle('on', srcOpen);
  document.getElementById('veil').classList.toggle('on', srcOpen);
  if(srcOpen) loadSrc();
}
function toggleOv(force){
  ovOpen = (force===undefined) ? !ovOpen : force;
  document.getElementById('ov').classList.toggle('on', ovOpen);
}
document.addEventListener('keydown', e=>{
  if(e.metaKey||e.ctrlKey||e.altKey) return;
  const k = e.key;
  if(k==='Escape'){ if(srcOpen) return toggleSrc(false); if(ovOpen) return toggleOv(false); return toggleOv(true); }
  if(k==='ArrowRight'||k===' '||k==='PageDown'||k==='j'){ e.preventDefault(); toggleOv(false); go(cur+1); }
  else if(k==='ArrowLeft'||k==='PageUp'||k==='k'){ e.preventDefault(); toggleOv(false); go(cur-1); }
  else if(k==='Home'){ go(0); } else if(k==='End'){ go(N-1); }
  else if(k==='s'||k==='S'){ toggleSrc(); }
  else if(k==='o'||k==='O'){ toggleOv(); }
  else if(/^[0-9]$/.test(k)){ }
});
document.getElementById('veil').onclick = ()=>toggleSrc(false);
const m = location.hash.match(/#\\/(\\d+)/);
go(m ? parseInt(m[1])-1 : 0, false);

// staleness: re-hash the served markdown and warn if the deck is out of date
(async ()=>{
  try{
    const r = await fetch('RECAP_2026_08_06.md?ts='+Date.now(), {cache:'no-store'});
    if(!r.ok) return;
    const buf = await r.arrayBuffer();
    const d = await crypto.subtle.digest('SHA-256', buf);
    const hex = [...new Uint8Array(d)].map(b=>b.toString(16).padStart(2,'0')).join('');
    if(hex.slice(0,12) !== '%(hash)s'){
      const el = document.getElementById('stale');
      el.innerHTML = '<b>STALE.</b> The source markdown has changed since this deck was built ('
        + '%(hash)s → ' + hex.slice(0,12) + '). Re-run the build script.';
      el.classList.add('on');
    }
  }catch(e){}
})();
"""


def build():
    md = SRC.read_text()
    digest = hashlib.sha256(md.encode()).hexdigest()[:12]
    secs = split_sections(md)

    srcmap = {}
    for k, (title, body) in secs.items():
        srcmap[str(k)] = {"title": title, "html": body}

    def jsobj(d):
        import json
        return json.dumps(d)

    body = []
    for i, s in enumerate(SLIDES):
        sw = (f'<div class="sowhat"><b>so what</b>{s["sowhat"]}</div>' if s["sowhat"] else "")
        sb = ""
        if s["src"] is not None and str(s["src"]) in srcmap:
            sb = (f'<button class="srcbtn" onclick="toggleSrc(true)">◧ verbatim source — '
                  f'§{s["src"]}. {html.escape(srcmap[str(s["src"])]["title"].split(". ",1)[-1])}</button>')
        body.append(
            f'<section class="slide" data-src="{s["src"]}">'
            f'<div class="kicker">{s["kicker"]}</div>'
            f'<h1 class="head">{s["headline"]}</h1>'
            f'{s["body"]}{sw}{sb}</section>')

    ov = ['<div id="ov"><h3>overview — click a view, or press O / Esc</h3><div class="ovg">']
    for i, s in enumerate(SLIDES):
        plain = re.sub(r"<[^>]+>", "", s["headline"])
        ov.append(f'<div class="ovc" onclick="toggleOv(false);go({i})"><div class="i">{i+1:02d}</div>'
                  f'<div class="k">{html.escape(s["kicker"])}</div><div class="h">{html.escape(plain)}</div></div>')
    ov.append("</div></div>")

    js = JS_TMPL % {"n": len(SLIDES), "srcmap": jsobj(srcmap), "hash": digest}

    doc = f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>visual-latents — recap 2026-08-06</title>
<style>{CSS}</style></head><body>
<div id="deck">
  <div id="bar">
    <span class="title">visual-latents</span>
    <span class="sub">project recap · 2026-08-06 · build {digest}</span>
    <span class="spacer"></span>
    <button onclick="toggleOv()">overview <span class="mut">O</span></button>
    <button onclick="toggleSrc()">source <span class="mut">S</span></button>
    <button onclick="go(cur-1)">←</button><button onclick="go(cur+1)">→</button>
    <span class="cnt" id="cnt"></span>
  </div>
  <div id="prog"><i></i></div>
  <div id="stage">{''.join(body)}</div>
</div>
{''.join(ov)}
<div class="veil" id="veil"></div>
<aside id="src"><div id="srch"><span class="t">verbatim source</span><span class="t" id="srct"></span>
 <span class="spacer"></span><span class="f">RECAP_2026_08_06.md</span>
 <button onclick="toggleSrc(false)">close ✕</button></div><div id="srcb"></div></aside>
<div id="stale"></div>
<script>{js}</script></body></html>"""

    OUT.write_text(doc)
    print(f"built {OUT}  ({len(doc)//1024} KB, {len(SLIDES)} views, source hash {digest}, "
          f"{len(srcmap)} source sections embedded)")


if __name__ == "__main__":
    build()
