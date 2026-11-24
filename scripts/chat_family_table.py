"""Print the chat (SFT) family table from <root>/pytte-*-sv3-chat/{eval,eval_pairs,judge,summary}.json.

    python scripts/chat_family_table.py            # markdown table, Swedish headers
"""
import json, sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/Volumes/Modelltest/pytte/checkpoints")
SIZES = ["0.3m", "1m", "2m", "3m", "7m", "15m"]

def load(p):
    return json.loads(p.read_text()) if p.exists() else None

def pct(x):
    return "–" if x is None else f"{100*x:.0f} %"

rows = []
for s in SIZES:
    d = ROOT / f"pytte-{s}-sv3-chat"
    ev, pairs, jd, sm = (load(d / f) for f in ("eval.json", "eval_pairs.json", "judge.json", "summary.json"))
    if ev is None:
        continue
    inf, fu, st = ev["instruction_following"], ev["follow_ups"], ev["small_talk"]
    per = inf["per_ingredient"]
    mp = pairs["minimal_pairs:minimal_pairs_sv"]["accuracy"] if pairs else None
    js = jd["summary"] if jd else {}
    minutes = sm.get("train_minutes") if sm else None
    st_short = sum(st[k]["short"] for k in st) / len(st)
    st_offer = sum(st[k]["offers_story"] for k in st) / len(st)
    rows.append([
        f"pytte-{s}-chat",
        f"{minutes:.0f}" if minutes else "–",
        pct(inf["ingredient_hit_rate"]), pct(per.get("name")), pct(per.get("noun")), pct(per.get("adj")),
        pct(per.get("verb")), pct(per.get("place")), pct(inf["all_ingredients"]), pct(inf["ended_naturally"]),
        pct(fu["keeps_name"]), pct(st_short), pct(st_offer), pct(mp),
        f"{js['grammatik']:.1f}" if js else "–", f"{js['sammanhang']:.1f}" if js else "–",
        f"{js['kreativitet']:.1f}" if js else "–", f"{js['sprakfel']:.2f}" if js else "–",
    ])

hdr = ["Modell", "min", "Träff", "namn", "subst", "adj", "verb", "plats", "Alla", "Slut",
       "Uppf. namn", "Småprat kort", "Erbj. saga", "Min. par", "Gr", "Sam", "Kre", "Fel"]
print("| " + " | ".join(hdr) + " |")
print("|" + "|".join(["---"] + ["---:"] * (len(hdr) - 1)) + "|")
for r in rows:
    print("| " + " | ".join(r) + " |")
