"""Evaluate a chat-tuned Pytte: does it do what the request asks?

Builds fresh requests from the validation stories' recipes (same generator as the training
data, different stories), lets the model answer, and checks that each ingredient the request
mentioned appears in the reply. Also tests follow-ups ("Vad händer sedan?"), greetings and
off-topic questions. Writes <model>/eval.json with the replies as "samples" so
scripts/judge_family.py can grade them.

    python scripts/eval_sft.py --model checkpoints/pytte-3m-sv3-chat --tok data/tok-sv3-4096 \
        --data data/sagosvenska-mix1/clean --lexicon data/sft-mix1/lexicon.json --n 100
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mlx.core as mx  # noqa: E402
from pytte.chat import reply  # noqa: E402
from pytte.model import Pytte  # noqa: E402
from pytte.prepare_sft import FOLLOW_UPS, GREETINGS, OFFTOPIC, WHO, make_request  # noqa: E402
from pytte.tokenizer import load as load_tok  # noqa: E402

ADJ_IRREG = {"liten": {"liten", "litet", "lilla", "små"}, "gammal": {"gammal", "gammalt", "gamla"},
             "vacker": {"vacker", "vackert", "vackra"}}


def words(t):
    return re.findall(r"[a-zåäöé]+", t.lower())


def adj_forms(a):
    if a in ADJ_IRREG:
        return ADJ_IRREG[a]
    f = {a, a + "a", a + "t", a + "e", a + "are"}
    if a.endswith("d"):
        f.add(a[:-1] + "tt")
    if a.endswith("t"):
        f.add(a + "t")
    if a.endswith("en"):
        f.add(a[:-2] + "na")
    return f


def verb_forms(v, present):
    head, _, tail = v.partition(" ")
    stem = head[:-1] if head.endswith("a") else head
    f = {head, stem + "ade", stem + "de", stem + "te", stem + "at", stem + "t", stem + "er", stem + "ar", stem + "r"}
    if present:
        f.add(present.split()[0])
    return f


def noun_hit(n, ws):
    stem = n[:-1] if (n.endswith(("a", "e")) and len(n) > 4) else n
    return any(w.startswith(stem) for w in ws)


def check(used, text, lex):
    ws = set(words(text))
    hits = {}
    if "name" in used:
        hits["name"] = used["name"].lower() in ws
    if "noun" in used:
        hits["noun"] = noun_hit(used["noun"], ws)
    if "adj" in used:
        hits["adj"] = bool(adj_forms(used["adj"]) & ws)
    if "verb" in used:
        hits["verb"] = bool(verb_forms(used["verb"], lex["present"].get(used["verb"])) & ws)
    if "place" in used:
        last = words(used["place"])[-1]
        hits["place"] = any(w.startswith(last[:-1] if len(last) > 4 else last) for w in ws)
    if used.get("plural"):
        hits["plural"] = used["plural"] in ws
    if used.get("first_person"):
        hits["first_person"] = "jag" in ws
    if used.get("tense") == "preteritum":
        past = sum(w in ("var", "gick", "sa", "hade", "kom", "såg", "blev", "tittade", "sprang") for w in words(text))
        pres = sum(w in ("är", "går", "säger", "har", "kommer", "ser", "blir", "tittar", "springer") for w in words(text))
        hits["tense"] = past > pres
    return hits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tok", required=True)
    p.add_argument("--data", required=True, help="clean dir (val.jsonl is used)")
    p.add_argument("--lexicon", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--n-follow", type=int, default=30)
    p.add_argument("--max-new", type=int, default=300)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    mx.random.seed(args.seed)
    rng = random.Random(args.seed)
    model = Pytte.load(args.model)
    tok = load_tok(args.tok)
    lex = json.loads(Path(args.lexicon).read_text())
    val = [json.loads(l) for l in (Path(args.data) / "val.jsonl").open()]
    rng.shuffle(val)
    t0 = time.time()

    # 1. requests from recipes; force at least one ingredient so there is something to check
    items = []
    for rec in val:
        req, used = make_request(rng, rec, lex)
        if used:
            items.append((req, used))
        if len(items) == args.n:
            break
    results, totals, hits_by = [], {}, {}
    for req, used in items:
        ans = reply(model, tok, [{"role": "user", "content": req}], args.max_new, args.temp)
        n_tok = len(tok.encode(" " + ans).ids)
        h = check(used, ans, lex)
        for k, v in h.items():
            totals[k] = totals.get(k, 0) + 1
            hits_by[k] = hits_by.get(k, 0) + int(v)
        results.append({"request": req, "used": used, "hits": h, "reply": ans, "tokens": n_tok,
                        "ended": n_tok < args.max_new - 1, "words": len(words(ans))})
    follow = {k: hits_by[k] / totals[k] for k in totals}
    all_hits = sum(hits_by.values()) / max(1, sum(totals.values()))
    full = sum(all(r["hits"].values()) for r in results) / len(results)
    print(f"requests: {len(results)} | ingredient hit rate {all_hits:.1%} | all ingredients in {full:.1%} of replies | "
          + " ".join(f"{k} {v:.0%} (n={totals[k]})" for k, v in follow.items()), flush=True)
    print(f"  ended on its own {sum(r['ended'] for r in results)/len(results):.0%} | avg {sum(r['words'] for r in results)/len(results):.0f} words",
          flush=True)

    # 2. follow-ups: story, then "Vad händer sedan?"; the continuation should be new text that keeps the name
    fu = []
    for req, used in items[:args.n_follow]:
        first = reply(model, tok, [{"role": "user", "content": req}], args.max_new, args.temp)
        q2 = rng.choice(FOLLOW_UPS)
        second = reply(model, tok, [{"role": "user", "content": req}, {"role": "assistant", "content": first},
                                    {"role": "user", "content": q2}], args.max_new, args.temp)
        w1, w2 = words(first), words(second)
        g1 = {tuple(w1[i:i + 4]) for i in range(len(w1) - 3)}
        g2 = [tuple(w2[i:i + 4]) for i in range(len(w2) - 3)]
        overlap = sum(g in g1 for g in g2) / max(1, len(g2))
        keeps = ("name" not in used) or (used["name"].lower() in set(w2))
        fu.append({"request": req, "first": first, "follow_up": q2, "second": second, "overlap_4gram": overlap,
                   "keeps_name": keeps, "words": len(w2)})
    fu_summary = {"n": len(fu), "non_empty": sum(f["words"] >= 10 for f in fu) / max(1, len(fu)),
                  "keeps_name": sum(f["keeps_name"] for f in fu) / max(1, len(fu)),
                  "avg_overlap_4gram": sum(f["overlap_4gram"] for f in fu) / max(1, len(fu)),
                  "avg_words": sum(f["words"] for f in fu) / max(1, len(fu))}
    print("follow-ups:", json.dumps(fu_summary), flush=True)

    # 3. greetings / identity / off-topic: short replies that offer a story, not a story
    small = []
    for group, qs in (("greet", GREETINGS), ("who", WHO), ("off", OFFTOPIC)):
        for q in rng.sample(qs, min(8, len(qs))):
            a = reply(model, tok, [{"role": "user", "content": q}], 120, args.temp)
            small.append({"group": group, "request": q, "reply": a, "words": len(words(a)),
                          "offers_story": "saga" in a.lower(), "short": len(words(a)) <= 40})
    small_summary = {}
    for group in ("greet", "who", "off"):
        g = [x for x in small if x["group"] == group]
        small_summary[group] = {"n": len(g), "short": sum(x["short"] for x in g) / len(g),
                                "offers_story": sum(x["offers_story"] for x in g) / len(g)}
    print("small talk:", json.dumps(small_summary), flush=True)

    out = {"model": args.model, "params": model.n_params(True), "params_no_embed": model.n_params(False),
           "instruction_following": {"n": len(results), "ingredient_hit_rate": all_hits, "all_ingredients": full,
                                     "per_ingredient": follow, "per_ingredient_n": totals,
                                     "ended_naturally": sum(r["ended"] for r in results) / len(results),
                                     "avg_words": sum(r["words"] for r in results) / len(results)},
           "follow_ups": fu_summary, "small_talk": small_summary,
           "samples": [r["reply"] for r in results], "requests": results, "follow_up_examples": fu, "small_talk_examples": small,
           "seconds": time.time() - t0}
    out_path = Path(args.out) if args.out else Path(args.model) / "eval.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {out_path} ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
