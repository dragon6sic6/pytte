"""Evaluate a Pytte checkpoint: perplexity, bits/byte, minimal-pair accuracy,
generation statistics and novelty against the training data.

    python -m pytte.eval --model ckpt/pytte-1m-en --tok tok-en-4096 --data tokens-en-4096 \
        --pairs eval/minimal_pairs_en.json --train-text data/TinyStoriesV2-GPT4-train.txt --n-samples 100
"""
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import mlx.core as mx
import numpy as np

from .model import Pytte
from .prepare_data import iter_docs
from .sample import generate
from .tokenizer import EOT, load as load_tok


def sequence_logprob(model, tok, text):
    """(total log p(text), number of predicted tokens) with <|endoftext|> as the start token."""
    eot = tok.token_to_id(EOT)
    ids = [eot] + tok.encode(text).ids
    x = mx.array([ids[:-1]])
    logp = mx.log(mx.softmax(model(x)[0], axis=-1))
    tgt = mx.array(ids[1:])
    return float(mx.take_along_axis(logp, tgt[:, None], axis=1).sum()), len(ids) - 1


def minimal_pairs(model, tok, pairs):
    """BLiMP-style: the model is right when it assigns the grammatical sentence a higher total
    log-probability. `accuracy_per_token` compares mean log-probability per token instead, which
    removes the advantage a shorter string gets when the two forms tokenize to different lengths."""
    per_cat = {}
    hits_norm = 0
    for p in pairs:
        (g, ng), (b, nb) = sequence_logprob(model, tok, p["good"]), sequence_logprob(model, tok, p["bad"])
        c = per_cat.setdefault(p["cat"], [0, 0])
        c[0] += int(g > b)
        c[1] += 1
        hits_norm += int(g / ng > b / nb)
    total = sum(c[0] for c in per_cat.values()), sum(c[1] for c in per_cat.values())
    return {"accuracy": total[0] / total[1], "accuracy_per_token": hits_norm / total[1], "n": total[1],
            "per_category": {k: round(v[0] / v[1], 3) for k, v in sorted(per_cat.items())}}


def perplexity(model, tok, data_dir, seq=256, batch=32, n_batches=50, seed=1):
    data = np.memmap(Path(data_dir) / "val.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(n_batches):
        idx = rng.integers(0, len(data) - seq - 1, batch)
        x = mx.array(np.stack([data[i:i + seq] for i in idx]).astype(np.int32))
        y = mx.array(np.stack([data[i + 1:i + seq + 1] for i in idx]).astype(np.int32))
        logp = mx.log(mx.softmax(model(x), axis=-1))
        losses.append(-mx.take_along_axis(logp, y[..., None], axis=-1).mean())
    loss = float(mx.mean(mx.stack(losses)))
    sample = np.array(data[:200_000]).tolist()
    bpt = len(tok.decode(sample).encode("utf-8")) / len(sample)
    return {"val_loss": loss, "val_ppl": math.exp(loss), "val_bpb": loss / math.log(2) / bpt,
            "bytes_per_token": bpt}


def words(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())


def ngrams(ws, n):
    return [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)]


def generation_stats(model, tok, n_samples, max_new, temp, top_p, train_ngrams=None, n=7):
    texts = [generate(model, tok, "", max_new=max_new, temp=temp, top_p=top_p) for _ in range(n_samples)]
    lens, rep, dist2, dist3, novel = [], [], [], [], []
    for t in texts:
        ws = words(t)
        lens.append(len(ws))
        g4 = ngrams(ws, 4)
        rep.append(1 - len(set(g4)) / max(1, len(g4)))          # repeated 4-grams within the sample
        dist2.append(len(set(ngrams(ws, 2))) / max(1, len(ws)))
        dist3.append(len(set(ngrams(ws, 3))) / max(1, len(ws)))
        if train_ngrams is not None:
            gn = ngrams(ws, n)
            if gn:
                h = np.fromiter((hash(g) & 0x7FFFFFFFFFFFFFFF for g in gn), dtype=np.int64, count=len(gn))
                novel.append(float(np.mean(~np.isin(h, train_ngrams))))
    out = {"n_samples": n_samples, "avg_words": float(np.mean(lens)),
           "ended_naturally": sum(len(tok.encode(t).ids) < max_new for t in texts) / n_samples,
           "repeated_4grams": float(np.mean(rep)), "distinct_2": float(np.mean(dist2)),
           "distinct_3": float(np.mean(dist3))}
    if novel:
        out[f"novel_{n}grams"] = float(np.mean(novel))
    return out, texts


def load_train_ngrams(path, n=7, max_docs=50_000):
    """Sorted array of hashed training n-grams. ~50k stories -> ~10M hashes -> 80 MB.
    Kept small on purpose: the evaluation must not compete for memory with other jobs."""
    chunks = []
    for i, doc in enumerate(iter_docs(path)):
        if i >= max_docs:
            break
        gn = ngrams(words(doc), n)
        if gn:
            chunks.append(np.fromiter((hash(g) & 0x7FFFFFFFFFFFFFFF for g in gn), dtype=np.int64, count=len(gn)))
    return np.unique(np.concatenate(chunks)) if chunks else np.zeros(0, dtype=np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tok", required=True)
    p.add_argument("--data", default=None, help="token dir with val.bin for perplexity")
    p.add_argument("--pairs", nargs="*", default=[], help="minimal pair json files")
    p.add_argument("--train-text", default=None, help="training text/jsonl for novelty")
    p.add_argument("--n-samples", type=int, default=50)
    p.add_argument("--max-new", type=int, default=200)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="defaults to <model>/eval.json")
    args = p.parse_args()

    mx.random.seed(args.seed)
    model = Pytte.load(args.model)
    tok = load_tok(args.tok)
    result = {"model": args.model, "params": model.n_params(True), "params_no_embed": model.n_params(False)}

    if args.data:
        result["perplexity"] = perplexity(model, tok, args.data)
        print("perplexity:", json.dumps(result["perplexity"]), flush=True)
    for pf in args.pairs:
        spec = json.loads(Path(pf).read_text())
        r = minimal_pairs(model, tok, spec["pairs"])
        result[f"minimal_pairs:{Path(pf).stem}"] = r
        print(f"{Path(pf).stem}: {r['accuracy']:.1%} of {r['n']} | {r['per_category']}", flush=True)
    if args.n_samples:
        train_ngrams = load_train_ngrams(args.train_text) if args.train_text else None
        stats, texts = generation_stats(model, tok, args.n_samples, args.max_new, args.temp, args.top_p, train_ngrams)
        result["generation"] = stats
        result["samples"] = texts  # all of them: the judge (pytte.judge) reads these
        print("generation:", json.dumps(stats), flush=True)

    out = Path(args.out) if args.out else Path(args.model) / "eval.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
