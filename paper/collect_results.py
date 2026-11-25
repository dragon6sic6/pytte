"""Collect every number the paper uses from the checkpoint directories into paper/results.json.

    python paper/collect_results.py [--root /Volumes/Modelltest/pytte/checkpoints]

The paper's tables and figures are generated from results.json only (paper/make_figures.py,
paper/build.py), so they can be rebuilt without the checkpoints.
"""
import argparse
import json
from pathlib import Path

SIZES = ["0.3m", "1m", "2m", "3m", "7m", "15m"]
FAMILIES = ["sv", "en", "en8m", "sv2", "sv3"]


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def base(d):
    sm, ev, jd = load(d / "summary.json"), load(d / "eval.json"), load(d / "judge.json")
    if not sm or not ev:
        return None
    pairs = next((v for k, v in ev.items() if k.startswith("minimal_pairs")), None)
    gen = ev.get("generation", {})
    return {
        "params": sm["params"], "params_no_embed": sm["params_no_embed"], "minutes": sm["train_minutes"],
        "tokens": sm["tokens"], "steps": sm["steps"], "tok_s": sm["tok_s"], "bpb": sm["best_val_bpb"],
        "val_loss": sm["best_val_loss"], "bytes_per_token": sm["bytes_per_token"], "lr": sm["args"]["lr"],
        "pairs": pairs["accuracy"] if pairs else None, "pairs_per_token": pairs.get("accuracy_per_token") if pairs else None,
        "pairs_cat": pairs["per_category"] if pairs else None,
        "ended": gen.get("ended_naturally"), "novel7": gen.get("novel_7grams"), "rep4": gen.get("repeated_4grams"),
        "avg_words": gen.get("avg_words"),
        "judge": jd["summary"] if jd else None,
        "judge_scores": [r.get("sammanhang") for r in jd["results"]] if jd and isinstance(jd["results"][0], dict) else None,
        "sample0": ev["samples"][0] if ev.get("samples") else None,
        "config": sm["config"],
    }


def chat(d):
    ev, jd, pr, sm = load(d / "eval.json"), load(d / "judge.json"), load(d / "eval_pairs.json"), load(d / "summary.json")
    if not ev:
        return None
    evs = [json.loads(l) for l in (d / "log.jsonl").read_text().splitlines() if '"val_loss"' in l]
    best = min(evs, key=lambda e: e["val_loss"])
    inf, fu, st = ev["instruction_following"], ev["follow_ups"], ev["small_talk"]
    pairs = pr["minimal_pairs:minimal_pairs_sv"] if pr else None
    return {
        "minutes": sm["train_minutes"] if sm else None, "steps": sm["steps"] if sm else evs[-1]["step"],
        "tokens": sm["tokens"] if sm else evs[-1]["step"] * 8192,
        "best_val_loss": best["val_loss"], "best_step": best["step"], "last_val_loss": evs[-1]["val_loss"],
        "val_curve": [[e["step"], round(e["val_loss"], 4)] for e in evs],
        "hit": inf["ingredient_hit_rate"], "all": inf["all_ingredients"], "per": inf["per_ingredient"],
        "per_n": inf["per_ingredient_n"], "ended": inf["ended_naturally"], "avg_words": inf["avg_words"],
        "follow_nonempty": fu["non_empty"], "follow_name": fu["keeps_name"], "follow_overlap": fu["avg_overlap_4gram"],
        "small_talk": {k: [v["short"], v["offers_story"], v["n"]] for k, v in st.items()},
        "pairs": pairs["accuracy"] if pairs else None, "pairs_cat": pairs["per_category"] if pairs else None,
        "judge": jd["summary"] if jd else None,
        "request0": ev["requests"][0] if ev.get("requests") else None, "sample0": ev["samples"][0] if ev.get("samples") else None,
        "follow_example0": ev["follow_up_examples"][0] if ev.get("follow_up_examples") else None,
        "small_talk_examples": ev.get("small_talk_examples", [])[:24],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/Volumes/Modelltest/pytte/checkpoints")
    args = p.parse_args()
    root = Path(args.root)
    out = {"families": {}, "chat": {}, "chat_variants": {}, "extra": {}}
    for fam in FAMILIES:
        out["families"][fam] = {s: base(root / f"pytte-{s}-{fam}") for s in SIZES}
    for s in SIZES:
        out["chat"][s] = chat(root / f"pytte-{s}-sv3-chat")
    for name in ("pytte-7m-sv3-chat-x3", "pytte-15m-sv3-chat-x3", "pytte-15m-sv3-chat-27min", "pytte-7m-sv3-chat-q8"):
        if (root / name / "eval.json").exists():
            out["chat_variants"][name] = chat(root / name) if (root / name / "log.jsonl").exists() else None
    for sub in ("seedcheck", "sv2long", "sv3mix", "sftsweep"):
        d = root / sub
        if d.exists():
            out["extra"][sub] = sorted(x.name for x in d.iterdir() if x.is_dir())
    dst = Path(__file__).resolve().parent / "results.json"
    dst.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    n = sum(1 for f in out["families"].values() for v in f.values() if v) + sum(1 for v in out["chat"].values() if v)
    print(f"{dst}: {n} models, {dst.stat().st_size/1e3:.0f} kB")


if __name__ == "__main__":
    main()
