"""Build the paper: fill the tables in pytte.src.html from results.json, write pytte.html, print pytte.pdf.

    python paper/collect_results.py     # needs the checkpoints; writes results.json
    python paper/make_figures.py        # figures/*.svg from results.json
    python paper/build.py               # pytte.html and pytte.pdf (needs Google Chrome for the PDF)

Every number in a generated table comes from results.json; the prose quotes the lab log.
"""
import html
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
SIZES = ["0.3m", "1m", "2m", "3m", "7m", "15m"]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def F(f, s):
    return R["families"][f][s]


def judge(v, key):
    j = v["judge"]
    alt = {"grammatik": "grammar", "sammanhang": "coherence", "kreativitet": "creativity", "sprakfel": "errors"}[key]
    return j.get(key, j.get(alt))


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def table(head, rows, cls="", groups=None):
    out = [f'<table class="{cls}">', "<thead>"]
    if groups:
        out.append("<tr>" + "".join(f'<th colspan="{n}" class="grp">{t}</th>' for t, n in groups) + "</tr>")
    out.append("<tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>")
    for r in rows:
        out.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def t_family():
    rows = []
    for s in SIZES + ["40m"]:
        if s == "40m":
            rows.append(["pytte-40m (defined, not trained)", 512, 12, "8", 1408, "4 096", "40.65M", "38.55M", "5%", "–", "–"])
            continue
        v = F("sv3", s); c = v["config"]
        heads = f'{c["n_heads"]}' + (f' ({c["n_kv_heads"]} kv)' if c["n_kv_heads"] != c["n_heads"] else "")
        rows.append([f"pytte-{s}", c["dim"], c["n_layers"], heads, c["ff_dim"], f'{c["vocab_size"]:,}'.replace(",", " "),
                     f'{v["params"] / 1e6:.2f}M', f'{v["params_no_embed"] / 1e6:.2f}M',
                     f'{100 * (v["params"] - v["params_no_embed"]) / v["params"]:.0f}%', f'{v["minutes"]:.0f}', f'{v["lr"]:g}'])
    return table(["Model", "dim", "layers", "heads", "ff", "vocab", "params", "w/o embed.", "embed. share", "minutes", "peak lr"], rows)


def t_twins():
    rows = []
    for s in SIZES:
        r = [f"pytte-{s}"]
        for key in ("bpb", "pairs", "coh", "ended"):
            for f in ("sv", "en", "en8m"):
                v = F(f, s)
                r.append({"bpb": f'{v["bpb"]:.3f}', "pairs": pct(v["pairs"]), "coh": f'{judge(v, "sammanhang"):.1f}', "ended": pct(v["ended"], 0)}[key])
        rows.append(r)
    head = ["Model"] + ["sv", "en", "en8m"] * 4
    return table(head, rows, groups=[("", 1), ("Bits per byte", 3), ("Minimal pairs, %", 3), ("Judge coherence", 3), ("Ends on its own, %", 3)])


def t_mix():
    rows = []
    for s in SIZES:
        r = [f"pytte-{s}"]
        for fn in (lambda v: pct(v["pairs"]), lambda v: pct(v["pairs_cat"]["plural"], 0), lambda v: f'{judge(v, "sammanhang"):.1f}',
                   lambda v: pct(v["novel7"], 0)):
            for f in ("sv", "sv2", "sv3"):
                r.append(fn(F(f, s)))
        rows.append(r)
    head = ["Model"] + ["v1.1", "+v2", "mix1"] * 4
    return table(head, rows, groups=[("", 1), ("Minimal pairs, %", 3), ("Plural pairs, %", 3), ("Judge coherence", 3), ("Novel 7-grams, %", 3)])


def t_release():
    rows = []
    for s in SIZES:
        v = F("sv3", s)
        rows.append([f"pytte-{s}", f'{v["minutes"]:.0f}', f'{v["tokens"] / 1e6:.0f}M', f'{v["bpb"]:.3f}', pct(v["pairs"]),
                     f'{judge(v, "grammatik"):.1f}', f'{judge(v, "sammanhang"):.1f}', f'{judge(v, "kreativitet"):.1f}',
                     f'{judge(v, "sprakfel"):.2f}', pct(v["ended"], 0), pct(v["novel7"], 0), f'{v["avg_words"]:.0f}'])
    return table(["Model", "min", "tokens seen", "bits/byte", "min. pairs, %", "gram.", "coh.", "creat.", "errors/story", "ends, %", "novel 7-gr., %", "words"],
                 rows, groups=[("", 5), ("Judge, 1–10", 3), ("", 4)])


def chat_row(name, c, extra=None):
    st = c["small_talk"]
    short = sum(round(v[0] * v[2]) for v in st.values()); n = sum(v[2] for v in st.values())
    per = c["per"]
    j = c["judge"] or {}
    r = [name]
    if extra is not None:
        r.append(extra)
    r += [f'{c["minutes"]:.0f}' if c["minutes"] else "25 of 30", f'{c["tokens"] / 15156937:.1f}' if extra is None else f'{c["tokens"] / (45.21e6 if "x3" in name else 15156937):.1f}',
          pct(c["hit"], 0), pct(c["all"], 0), pct(per.get("name", 0), 0), pct(per.get("place", 0), 0), pct(per.get("noun", 0), 0),
          pct(per.get("verb", 0), 0), pct(per.get("adj", 0), 0), pct(c["follow_name"], 0), f"{short}/{n}",
          pct(c["pairs"]) if c["pairs"] is not None else "–",
          f'{j["grammatik"]:.1f} / {j["sammanhang"]:.1f}' if j else "–"]
    return r


def t_chat():
    rows = [chat_row(f"pytte-{s}-chat", R["chat"][s]) for s in SIZES]
    return table(["Model", "min", "epochs", "followed", "all", "name", "place", "object", "verb", "adj.", "same child, %", "short", "pairs, %", "judge g. / c."],
                 rows, cls="wide", groups=[("", 3), ("Requested ingredients used, %", 7), ("Follow-up", 1), ("Small talk", 1), ("Story quality", 2)])


def t_chat_x3():
    V = R["chat_variants"]
    items = [("pytte-7m-chat", R["chat"]["7m"], "1 pass, 15M tok"), ("pytte-7m-chat-x3", V["pytte-7m-sv3-chat-x3"], "3 passes, 45M tok"),
             ("pytte-15m-chat", R["chat"]["15m"], "1 pass, 15M tok"), ("pytte-15m-chat-x3", V["pytte-15m-sv3-chat-x3"], "3 passes, 45M tok")]
    rows = []
    for name, c, data in items:
        rows.append([name, data, f'{c["tokens"] / (45.21e6 if "x3" in name else 15.157e6):.1f}', f'{c["best_val_loss"]:.3f}',
                     f'{c["best_step"]:,}'.replace(",", " "), f'{c["last_val_loss"]:.3f}', pct(c["hit"], 0), pct(c["per"]["noun"], 0),
                     pct(c["per"]["adj"], 0), pct(c["pairs"])])
    return table(["Model", "Chat data", "epochs", "best val. loss", "at step", "final val. loss", "followed, %", "object, %", "adj., %", "min. pairs, %"], rows, cls="wide")


def t_categories():
    cats = [("gender", "Gender (en/ett)"), ("definiteness", "Definiteness"), ("modal_infinitive", "Modal + infinitive"), ("past_tense", "Past tense"),
            ("verb_present", "Present tense"), ("word_order_v2", "Verb-second word order"), ("pronoun_case", "Pronoun case"),
            ("adjective_agreement", "Adjective agreement"), ("plural", "Plural agreement"), ("subordinate_negation", "Negation in subordinate clause")]
    rows = []
    for key, name in cats:
        rows.append([name] + [f'{100 * F("sv", s)["pairs_cat"][key]:.0f} → {100 * F("sv3", s)["pairs_cat"][key]:.0f}' for s in SIZES])
    rows.append(["<b>All 200 pairs</b>"] + [f'<b>{pct(F("sv", s)["pairs"])} → {pct(F("sv3", s)["pairs"])}</b>' for s in SIZES])
    return table(["Category (20 pairs each)"] + [f"pytte-{s}" for s in SIZES], rows)


def samples():
    out = []
    for s in SIZES:
        t = html.escape(F("sv3", s)["sample0"].strip()).replace("\n\n", "</p><p>").replace("\n", " ")
        out.append(f'<div class="sample"><div class="who">pytte-{s} · first of its 100 evaluation samples, unprompted</div><p>{t}</p></div>')
    c = R["chat"]["7m"]
    req = c["request0"]
    out.append(f'<div class="sample"><div class="who">pytte-7m-chat · first of its 100 evaluation requests</div>'
               f'<p><b>Du:</b> {html.escape(req["request"])}</p><p><b>Pytte:</b> ' + html.escape(req["reply"].strip()).replace("\n\n", "</p><p>") + "</p></div>")
    seen, st = set(), []
    for e in c["small_talk_examples"]:
        if e["group"] not in seen:
            seen.add(e["group"]); st.append(f'<p><b>Du:</b> {html.escape(e["request"])}<br><b>Pytte:</b> {html.escape(e["reply"].strip())}</p>')
    n_short = sum(1 for e in c["small_talk_examples"] if e["short"])
    out.append(f'<div class="sample"><div class="who">pytte-7m-chat · the first small-talk example of each kind '
               f'({n_short} of the {len(c["small_talk_examples"])} small-talk replies were short)</div>' + "".join(st) + "</div>")
    return "\n".join(out)


TABLES = {"family": t_family, "twins": t_twins, "mix": t_mix, "release": t_release, "chat": t_chat, "chat_x3": t_chat_x3,
          "categories": t_categories, "samples": samples}


def main():
    src = (HERE / "pytte.src.html").read_text()
    for name, fn in TABLES.items():
        tag = f"<!--GEN:{name}-->"
        assert tag in src, tag
        src = src.replace(tag, fn())
    assert "<!--GEN:" not in src, "unfilled placeholder"
    head, body = src.split("</style>", 1)
    body = re.sub(r"(?<=\d) (?=\d{3}(?!\d))", "&nbsp;", body)     # keep 59 224 on one line
    src = head + "</style>" + body
    (HERE / "pytte.html").write_text(src)
    print("paper/pytte.html")
    if "--no-pdf" in sys.argv:
        return
    pdf = HERE / "pytte.pdf"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                    f"file://{HERE / 'pytte.html'}"], check=True, capture_output=True)
    print(f"paper/pytte.pdf ({pdf.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
