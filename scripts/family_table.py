"""Print a markdown table of every trained Pytte checkpoint (summary.json + eval.json).

    python scripts/family_table.py [--root /Volumes/Modelltest/pytte/checkpoints]
"""
import argparse
import json
from pathlib import Path


def row(d):
    s = json.loads((d / "summary.json").read_text())
    e = json.loads((d / "eval.json").read_text()) if (d / "eval.json").exists() else {}
    mp = next((v for k, v in e.items() if k.startswith("minimal_pairs:")), {})
    g = e.get("generation", {})
    j = json.loads((d / "judge.json").read_text())["summary"] if (d / "judge.json").exists() else {}
    return {
        "model": d.name,
        "params": s["params"] / 1e6,
        "min": s["train_minutes"],
        "tokens": s["tokens"] / 1e6,
        "epochs": s["tokens"] / s.get("train_tokens", float("nan")) if s.get("train_tokens") else None,
        "ppl": s["best_val_ppl"],
        "bpb": s["best_val_bpb"],
        "pairs": mp.get("accuracy"),
        "pairs_tok": mp.get("accuracy_per_token"),
        "n_pairs": mp.get("n"),
        "grammar": j.get("grammatik", j.get("grammar")),
        "coherence": j.get("sammanhang", j.get("coherence")),
        "creativity": j.get("kreativitet", j.get("creativity")),
        "novel7": g.get("novel_7grams"),
        "rep4": g.get("repeated_4grams"),
        "ended": g.get("ended_naturally"),
        "sample": (e.get("samples") or [""])[0].replace("\n", " ")[:160],
    }


def fmt(v, spec):
    return "–" if v is None else format(v, spec)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/Volumes/Modelltest/pytte/checkpoints")
    p.add_argument("--samples", action="store_true", help="also print one sample per model")
    a = p.parse_args()
    rows = [row(d) for d in sorted(Path(a.root).iterdir()) if (d / "summary.json").exists()]
    rows.sort(key=lambda r: (r["model"].rsplit("-", 1)[-1], r["params"]))
    print("| Modell | Param | Min | Mtoken | Ppl | Bits/byte | Minimala par | Nya 7-gram | Upprepn. 4-gram | Slutade själv | Domare G/S/K |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        judge = "–" if r["grammar"] is None else f"{r['grammar']:.1f}/{r['coherence']:.1f}/{r['creativity']:.1f}"
        print(f"| {r['model']} | {r['params']:.2f}M | {r['min']:.0f} | {r['tokens']:.0f} | {r['ppl']:.2f} | {r['bpb']:.3f} | "
              f"{fmt(r['pairs'], '.1%')} | {fmt(r['novel7'], '.0%')} | {fmt(r['rep4'], '.1%')} | {fmt(r['ended'], '.0%')} | {judge} |")
    if a.samples:
        print()
        for r in rows:
            print(f"**{r['model']}:** {r['sample']}\n")


if __name__ == "__main__":
    main()
