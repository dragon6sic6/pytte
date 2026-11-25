"""Draw the paper's figures as SVG from paper/results.json. Plain Python, no dependencies.

    python paper/make_figures.py        # writes paper/figures/*.svg

Conventions: 2 px lines, 8 px markers with a surface ring, hairline grid, direct labels at the
line ends plus a legend, text in ink (never in the series colour), one y-axis per panel.
Colours: the first three slots of a colour-blind-validated categorical palette, in fixed order.
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

SIZES = ["0.3m", "1m", "2m", "3m", "7m", "15m"]
LABEL = {"0.3m": "0.3M", "1m": "1M", "2m": "2M", "3m": "3M", "7m": "7M", "15m": "15M"}
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e2dc", "#ffffff"
FONT = "font-family=\"Helvetica Neue, Helvetica, Arial, sans-serif\""


def params(s):
    return R["families"]["sv3"][s]["params"]


class Panel:
    """One plotting area with a log-x axis over the six model sizes and a linear y axis."""

    def __init__(self, x, y, w, h, ymin, ymax, yticks, yfmt=lambda v: f"{v:g}", title=None, ylabel=None, xlog=True):
        self.x, self.y, self.w, self.h, self.ymin, self.ymax = x, y, w, h, ymin, ymax
        self.yticks, self.yfmt, self.title, self.ylabel = yticks, yfmt, title, ylabel
        self.lo, self.hi = math.log10(params("0.3m")), math.log10(params("15m"))
        self.el = []

    def px(self, s):
        return self.x + (math.log10(params(s)) - self.lo) / (self.hi - self.lo) * self.w

    def py(self, v):
        return self.y + self.h - (v - self.ymin) / (self.ymax - self.ymin) * self.h

    def frame(self, xlabel="Parameters"):
        e = self.el
        if self.title:
            e.append(f'<text x="{self.x}" y="{self.y - 14}" font-size="12.5" font-weight="600" fill="{INK}">{self.title}</text>')
        for t in self.yticks:
            yy = self.py(t)
            e.append(f'<line x1="{self.x}" x2="{self.x + self.w}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="1"/>')
            e.append(f'<text x="{self.x - 8}" y="{yy + 4:.1f}" font-size="11" text-anchor="end" fill="{INK2}">{self.yfmt(t)}</text>')
        for s in SIZES:
            xx = self.px(s)
            e.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{self.y + self.h}" y2="{self.y + self.h + 4}" stroke="{INK2}" stroke-width="1"/>')
            e.append(f'<text x="{xx:.1f}" y="{self.y + self.h + 18}" font-size="11" text-anchor="middle" fill="{INK2}">{LABEL[s]}</text>')
        e.append(f'<line x1="{self.x}" x2="{self.x + self.w}" y1="{self.y + self.h}" y2="{self.y + self.h}" stroke="{INK2}" stroke-width="1"/>')
        if xlabel:
            e.append(f'<text x="{self.x + self.w / 2}" y="{self.y + self.h + 36}" font-size="11" text-anchor="middle" fill="{INK2}">{xlabel}</text>')
        if self.ylabel:
            e.append(f'<text x="{self.x - 40}" y="{self.y + self.h / 2}" font-size="11" text-anchor="middle" fill="{INK2}" '
                     f'transform="rotate(-90 {self.x - 40} {self.y + self.h / 2})">{self.ylabel}</text>')

    def line(self, values, color, width=2, dash=None):
        pts = [(self.px(s), self.py(v)) for s, v in zip(SIZES, values) if v is not None]
        d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.el.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{da}/>')
        for x, y in pts:
            self.el.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" stroke="{SURF}" stroke-width="2"/>')
        return pts[-1]

    def end_labels(self, items, dx=10, min_gap=14):
        """items: [(y_px, text)] -> labels to the right of the last point, nudged apart."""
        items = sorted(items)
        ys = [y for y, _ in items]
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] < min_gap:
                ys[i] = ys[i - 1] + min_gap
        shift = max(0, ys[-1] - (self.y + self.h)) if ys else 0
        for (y0, text), yy in zip(items, ys):
            yy -= shift
            self.el.append(f'<text x="{self.x + self.w + dx}" y="{yy + 4:.1f}" font-size="11" fill="{INK}">{text}</text>')


def legend(x, y, entries, gap=150):
    out = []
    for i, (color, text) in enumerate(entries):
        xx = x + i * gap
        out.append(f'<line x1="{xx}" x2="{xx + 22}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="2" stroke-linecap="round"/>')
        out.append(f'<circle cx="{xx + 11}" cy="{y}" r="4" fill="{color}" stroke="{SURF}" stroke-width="2"/>')
        out.append(f'<text x="{xx + 30}" y="{y + 4}" font-size="11" fill="{INK}">{text}</text>')
    return out


def svg(name, w, h, parts):
    body = "\n".join(parts)
    (OUT / name).write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" {FONT}>\n'
        f'<rect width="{w}" height="{h}" fill="{SURF}"/>\n{body}\n</svg>\n')
    print(f"figures/{name}")


def fam(f, key, sub=None, scale=1.0):
    vals = []
    for s in SIZES:
        v = R["families"][f][s]
        x = v[key] if sub is None else (v[key].get(sub[0], v[key].get(sub[1])) if v[key] else None)
        vals.append(None if x is None else x * scale)
    return vals


# ------------------------------------------------------------------ Figure 1: parameter budget
def fig_params():
    W, H, x0, bw, rh = 680, 250, 92, 470, 22
    parts = []
    names = [(C1, "Embedding table (tied with the output layer)"), (C2, "Attention"), (C3, "Feed-forward (SwiGLU)")]
    for i, (c, t) in enumerate(names):
        xx = x0 + [0, 250, 340][i]
        parts.append(f'<rect x="{xx}" y="14" width="12" height="12" rx="2" fill="{c}"/>')
        parts.append(f'<text x="{xx + 18}" y="24" font-size="11" fill="{INK}">{t}</text>')
    for r, s in enumerate(SIZES):
        c = R["families"]["sv3"][s]["config"]
        D, L, Hh, KV, FF, V = c["dim"], c["n_layers"], c["n_heads"], c["n_kv_heads"], c["ff_dim"], c["vocab_size"]
        hd = D // Hh
        emb = V * D
        att = L * (D * Hh * hd + 2 * D * KV * hd + Hh * hd * D)
        mlp = L * 3 * D * FF
        tot = emb + att + mlp
        total_params = R["families"]["sv3"][s]["params"]
        y = 44 + r * (rh + 10)
        parts.append(f'<text x="{x0 - 10}" y="{y + 15}" font-size="11.5" text-anchor="end" fill="{INK}">pytte-{s}</text>')
        xx = x0
        for val, col in ((emb, C1), (att, C2), (mlp, C3)):
            wseg = val / tot * bw
            parts.append(f'<rect x="{xx + 1:.1f}" y="{y}" width="{max(wseg - 2, 0):.1f}" height="{rh}" rx="3" fill="{col}"/>')
            if wseg > 34:
                parts.append(f'<text x="{xx + wseg / 2:.1f}" y="{y + 15}" font-size="11" text-anchor="middle" fill="#ffffff" font-weight="600">{100 * val / tot:.0f}%</text>')
            xx += wseg
        parts.append(f'<text x="{x0 + bw + 10}" y="{y + 15}" font-size="11" fill="{INK2}">{total_params / 1e6:.2f}M parameters</text>')
    svg("params.svg", W, H, parts)


# ------------------------------------------------------------------ Figure 2: scaling, three data sets
def fig_scaling():
    W, H = 680, 330
    p = Panel(70, 50, 430, 220, 0.35, 0.90, [0.4, 0.5, 0.6, 0.7, 0.8, 0.9], lambda v: f"{v:.1f}",
              ylabel="Bits per byte on held-out stories (lower is better)")
    p.frame()
    ends = []
    for f, col, text in (("en8m", C3, "English, 8M tokens"), ("en", C2, "English, 553M tokens"), ("sv", C1, "Swedish v1.1, 8M tokens")):
        x, y = p.line(fam(f, "bpb"), col)
        ends.append((y, text))
    p.end_labels(ends)
    parts = legend(70, 22, [(C1, "Swedish v1.1 (8M tokens)"), (C2, "English (553M tokens)"), (C3, "English (8M tokens)")], gap=180) + p.el
    svg("scaling.svg", W, H, parts)


# ------------------------------------------------------------------ Figure 3: three data mixes
def fig_datamix():
    W, H = 680, 320
    a = Panel(58, 62, 240, 190, 60, 100, [60, 70, 80, 90, 100], lambda v: f"{v:g}%", title="Grammar: minimal pairs correct")
    b = Panel(388, 62, 240, 190, 4, 10, [4, 6, 8, 10], lambda v: f"{v:g}", title="Coherence: judge score, 1 to 10")
    a.frame(); b.frame()
    ea, eb = [], []
    for f, col, short in (("sv", C1, "v1.1"), ("sv2", C2, "+ v2"), ("sv3", C3, "mix1")):
        _, ya = a.line(fam(f, "pairs", scale=100), col)
        _, yb = b.line(fam(f, "judge", ("sammanhang", "coherence")), col)
        ea.append((ya, short)); eb.append((yb, short))
    a.end_labels(ea, dx=9, min_gap=13); b.end_labels(eb, dx=9, min_gap=13)
    parts = legend(58, 22, [(C1, "v1.1 (template stories)"), (C2, "v1.1 + v2 (all varied stories)"), (C3, "mix1: v1.1 + one-child v2 (release)")], gap=195) + a.el + b.el
    svg("datamix.svg", W, H, parts)


# ------------------------------------------------------------------ Figure 4: categories, v1.1 vs mix1
def fig_categories():
    cats = [("plural", "Plural agreement"), ("adjective_agreement", "Adjective agreement"), ("word_order_v2", "Verb-second word order"),
            ("past_tense", "Past tense"), ("verb_present", "Present tense"), ("pronoun_case", "Pronoun case"),
            ("definiteness", "Definiteness"), ("modal_infinitive", "Modal + infinitive"), ("gender", "Gender (en/ett)"),
            ("subordinate_negation", "Negation in subordinate clause")]
    big = SIZES[1:]

    def mean(f, c):
        return 100 * sum(R["families"][f][s]["pairs_cat"][c] for s in big) / len(big)

    W, H, x0, w, rh, y0 = 680, 60 + len(cats) * 24 + 40, 222, 380, 24, 52
    X = lambda v: x0 + (v - 30) / 70 * w
    parts = legend(x0, 20, [(C1, "v1.1"), (C3, "mix1 (release)")], gap=90)
    for t in (40, 50, 60, 70, 80, 90, 100):
        parts.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{y0 - 6}" y2="{y0 + len(cats) * rh - 6}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{X(t):.1f}" y="{y0 + len(cats) * rh + 10}" font-size="11" text-anchor="middle" fill="{INK2}">{t}%</text>')
    parts.append(f'<line x1="{X(50):.1f}" x2="{X(50):.1f}" y1="{y0 - 6}" y2="{y0 + len(cats) * rh - 6}" stroke="{INK2}" stroke-width="1"/>')
    parts.append(f'<text x="{X(50):.1f}" y="{y0 - 12}" font-size="10.5" text-anchor="middle" fill="{INK2}">chance</text>')
    parts.append(f'<text x="{x0 + w / 2}" y="{y0 + len(cats) * rh + 28}" font-size="11" text-anchor="middle" fill="{INK2}">Minimal pairs correct, mean of the five models from 1M to 15M (20 pairs per category)</text>')
    for i, (c, name) in enumerate(cats):
        y = y0 + i * rh + 6
        a, b = mean("sv", c), mean("sv3", c)
        parts.append(f'<text x="{x0 - 12}" y="{y + 4}" font-size="11.5" text-anchor="end" fill="{INK}">{name}</text>')
        parts.append(f'<line x1="{X(a):.1f}" x2="{X(b):.1f}" y1="{y}" y2="{y}" stroke="{INK2}" stroke-width="1.5"/>')
        parts.append(f'<circle cx="{X(a):.1f}" cy="{y}" r="5" fill="{C1}" stroke="{SURF}" stroke-width="2"/>')
        parts.append(f'<circle cx="{X(b):.1f}" cy="{y}" r="5" fill="{C3}" stroke="{SURF}" stroke-width="2"/>')
        parts.append(f'<text x="{X(max(a, b)) + 12:.1f}" y="{y + 4}" font-size="10.5" fill="{INK2}">{a:.0f} → {b:.0f}</text>')
    svg("categories.svg", W, H, parts)


# ------------------------------------------------------------------ Figure 5: instruction following
def fig_chat():
    W, H = 680, 330
    p = Panel(70, 50, 430, 220, 0, 100, [0, 20, 40, 60, 80, 100], lambda v: f"{v:g}%",
              ylabel="Requests in which the ingredient appears in the story")
    p.frame()
    ch = R["chat"]
    ends = []
    for key, col, text in (("name", C1, "Name"), ("noun", C2, "Object"), ("adj", C3, "Adjective")):
        vals = [100 * ch[s]["per"].get(key, 0) for s in SIZES]
        x, y = p.line(vals, col)
        ends.append((y, f"{text} ({vals[-1]:.0f}%)"))
    vals = [100 * ch[s]["hit"] for s in SIZES]
    x, y = p.line(vals, INK2, dash="5 4")
    ends.append((y, f"All ingredients ({vals[-1]:.0f}%)"))
    p.end_labels(ends)
    parts = legend(70, 22, [(C1, "Name"), (C2, "Object (noun)"), (C3, "Adjective"), (INK2, "All requested ingredients")], gap=125) + p.el
    svg("chat.svg", W, H, parts)


# ------------------------------------------------------------------ Figure 6: 15M overfits the chat data
def fig_overfit():
    W, H, x0, y0, w, h = 680, 310, 70, 50, 430, 200
    c7, c15 = R["chat"]["7m"], R["chat"]["15m"]
    xmax, ymin, ymax = 18500, 1.30, 1.46
    X = lambda s: x0 + s / xmax * w
    Y = lambda v: y0 + h - (min(max(v, ymin), ymax) - ymin) / (ymax - ymin) * h
    parts = legend(70, 22, [(C1, "pytte-7m-chat (50 min)"), (C2, "pytte-15m-chat (75 min)")], gap=190)
    for t in (1.30, 1.34, 1.38, 1.42, 1.46):
        parts.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{x0 - 8}" y="{Y(t) + 4:.1f}" font-size="11" text-anchor="end" fill="{INK2}">{t:.2f}</text>')
    for t in (0, 4000, 8000, 12000, 16000):
        parts.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{y0 + h}" y2="{y0 + h + 4}" stroke="{INK2}" stroke-width="1"/>')
        parts.append(f'<text x="{X(t):.1f}" y="{y0 + h + 18}" font-size="11" text-anchor="middle" fill="{INK2}">{t:,}</text>'.replace(",", " "))
    parts.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{y0 + h}" y2="{y0 + h}" stroke="{INK2}" stroke-width="1"/>')
    parts.append(f'<text x="{x0 + w / 2}" y="{y0 + h + 36}" font-size="11" text-anchor="middle" fill="{INK2}">Fine-tuning step (8 192 tokens per step)</text>')
    parts.append(f'<text x="{x0 - 46}" y="{y0 + h / 2}" font-size="11" text-anchor="middle" fill="{INK2}" transform="rotate(-90 {x0 - 46} {y0 + h / 2})">Validation loss on replies (smoothed)</text>')

    def smooth(curve, k=5):
        out = []
        for i in range(len(curve)):
            win = curve[max(0, i - k // 2): i + k // 2 + 1]
            out.append((curve[i][0], sum(v for _, v in win) / len(win)))
        return out

    ends = []
    for c, col, text in ((c7, C1, "7M"), (c15, C2, "15M")):
        sm = smooth(c["val_curve"])
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(s):.1f},{Y(v):.1f}" for i, (s, v) in enumerate(sm))
        parts.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        bs = c["best_step"]
        bv = next(v for s, v in sm if s == bs)
        parts.append(f'<circle cx="{X(bs):.1f}" cy="{Y(bv):.1f}" r="5" fill="{col}" stroke="{SURF}" stroke-width="2"/>')
        ly = Y(bv) + 38
        parts.append(f'<line x1="{X(bs):.1f}" x2="{X(bs):.1f}" y1="{Y(bv) + 7:.1f}" y2="{ly - 12:.1f}" stroke="{INK2}" stroke-width="1"/>')
        parts.append(f'<text x="{X(bs):.1f}" y="{ly:.1f}" font-size="10.5" text-anchor="middle" fill="{INK}">best checkpoint: step {bs:,}</text>'.replace(",", " "))
        ends.append((Y(sm[-1][1]), sm[-1][0], text))
    for yy, s, text in ends:
        parts.append(f'<text x="{X(s) + 10:.1f}" y="{yy + 4:.1f}" font-size="11" fill="{INK}">{text}</text>')
    svg("overfit.svg", W, H, parts)


if __name__ == "__main__":
    fig_params(); fig_scaling(); fig_datamix(); fig_categories(); fig_chat(); fig_overfit()
