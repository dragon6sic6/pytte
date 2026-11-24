"""Swedish/English twin comparison: same architecture, same time budget, different data.

    python scripts/twin_table.py [--root /Volumes/Modelltest/pytte/checkpoints]

Prints a markdown table per size with tokens seen, epochs, bits/byte at the end and at
matched token counts (from log.jsonl), minimal-pair accuracy and judge scores.
"""
import argparse
import json
import math
from pathlib import Path

SIZES = ["0.3m", "1m", "2m", "3m", "7m", "15m"]
TRAIN_TOKENS = {"sv": {2048: 8_676_373, 4096: 8_137_315}, "en": {2048: 552_572_796, 4096: 552_572_796},
                "en8m": {}}  # filled from tokens-en8m-*/meta.json at run time
MATCH_MTOK = (10, 20, 40)


def load(d):
    s = json.loads((d / "summary.json").read_text())
    e = json.loads((d / "eval.json").read_text()) if (d / "eval.json").exists() else {}
    j = json.loads((d / "judge.json").read_text())["summary"] if (d / "judge.json").exists() else {}
    bpt = e.get("perplexity", {}).get("bytes_per_token") or s["bytes_per_token"]
    tps = s["tokens"] / s["steps"]
    evals = [r for r in map(json.loads, open(d / "log.jsonl")) if "val_loss" in r]
    for r in evals:
        r["tokens"] = r["step"] * tps
    mp = next((v for k, v in e.items() if k.startswith("minimal_pairs:")), {})
    return {"s": s, "e": e, "j": j, "bpt": bpt, "evals": evals, "mp": mp, "g": e.get("generation", {})}


def bpb_at(r, mtok):
    best = min(r["evals"], key=lambda x: abs(x["tokens"] / 1e6 - mtok))
    if abs(best["tokens"] / 1e6 - mtok) > 0.2 * mtok:
        return None
    return best["val_loss"] / math.log(2) / r["bpt"]


def f(v, spec="{:.3f}"):
    return "–" if v is None else spec.format(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/Volumes/Modelltest/pytte/checkpoints")
    a = p.parse_args()
    root = Path(a.root)
    for mp in root.parent.glob("data/tokens-en8m-*/meta.json"):
        m = json.loads(mp.read_text()); TRAIN_TOKENS["en8m"][m["vocab_size"]] = m["train_tokens"]
    cols = ["Modell", "Språk", "Mtoken", "Epoker", "Bits/byte", *[f"Bpb @{m}M" for m in MATCH_MTOK],
            "Minimala par", "Per token", "Nya 7-gram", "Domare G/S/K"]
    print("| " + " | ".join(cols) + " |")
    print("|---|---|" + "---:|" * (len(cols) - 2))
    for size in SIZES:
        for lang in ("sv", "en", "en8m"):
            d = root / f"pytte-{size}-{lang}"
            if not (d / "summary.json").exists():
                continue
            r = load(d)
            s, j = r["s"], r["j"]
            vocab = s["config"]["vocab_size"]
            epochs = s["tokens"] / TRAIN_TOKENS[lang][vocab]
            judge = "–" if not j else f"{j.get('grammatik', j.get('grammar', 0)):.1f}/" \
                                    f"{j.get('sammanhang', j.get('coherence', 0)):.1f}/" \
                                    f"{j.get('kreativitet', j.get('creativity', 0)):.1f}"
            row = [f"pytte-{size}", lang, f"{s['tokens']/1e6:.0f}", f"{epochs:.1f}" if epochs >= 1 else f"{epochs:.2f}",
                   f"{s['best_val_loss']/math.log(2)/r['bpt']:.3f}",
                   *[f(bpb_at(r, m)) for m in MATCH_MTOK],
                   f(r["mp"].get("accuracy"), "{:.1%}"), f(r["mp"].get("accuracy_per_token"), "{:.1%}"),
                   f(r["g"].get("novel_7grams"), "{:.0%}"), judge]
            print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
