"""Autoregressive sampling from a Pytte checkpoint with a KV cache.

    python -m pytte.sample --model ckpt/pytte-1m --tok tok-en-4096 --prompt "Once upon a time"
"""
import argparse
import sys
import time

import mlx.core as mx

from .model import Pytte
from .tokenizer import EOT, load as load_tok


def sample_token(logits, temp=1.0, top_k=0, top_p=1.0):
    """logits: (vocab,) -> int token id."""
    if temp <= 0:
        return int(mx.argmax(logits))
    logits = logits / temp
    if top_k > 0 and top_k < logits.shape[-1]:
        kth = mx.sort(logits)[-top_k]
        logits = mx.where(logits < kth, -mx.inf, logits)
    if top_p < 1.0:
        order = mx.argsort(-logits)
        probs = mx.softmax(logits[order])
        cum = mx.cumsum(probs)
        keep = cum - probs < top_p          # keep tokens until the mass passes top_p
        masked = mx.where(keep, logits[order], -mx.inf)
        logits = mx.full(logits.shape, -mx.inf)
        logits[order] = masked                # scatter back to vocabulary order
    return int(mx.random.categorical(logits))


def generate(model, tok, prompt, max_new=200, temp=0.8, top_k=0, top_p=1.0, stream=None):
    eot = tok.token_to_id(EOT)
    ids = tok.encode(prompt).ids if prompt else [eot]
    return generate_ids(model, tok, ids, max_new, temp, top_k, top_p, stream)


def generate_ids(model, tok, ids, max_new=200, temp=0.8, top_k=0, top_p=1.0, stream=None):
    """Continue a list of token ids; stops at <|endoftext|>. Returns the decoded continuation."""
    eot = tok.token_to_id(EOT)
    caches = model.new_caches()
    logits = model(mx.array([ids]), caches)[0, -1]
    out = []
    for _ in range(max_new):
        nxt = sample_token(logits, temp, top_k, top_p)
        if nxt == eot:
            break
        out.append(nxt)
        if stream:
            stream(tok.decode([nxt]))
        logits = model(mx.array([[nxt]]), caches)[0, -1]
    return tok.decode(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tok", required=True)
    p.add_argument("--prompt", default="")
    p.add_argument("--max-new", type=int, default=200)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        mx.random.seed(args.seed)
    model = Pytte.load(args.model)
    tok = load_tok(args.tok)
    for i in range(args.n):
        print(f"--- sample {i+1} ---\n{args.prompt}", end="", flush=True)
        t0 = time.time()
        text = generate(model, tok, args.prompt, args.max_new, args.temp, args.top_k, args.top_p,
                        stream=lambda s: (sys.stdout.write(s), sys.stdout.flush()))
        n = len(tok.encode(text).ids)
        print(f"\n[{n} tokens, {n/(time.time()-t0):.0f} tok/s]\n")


if __name__ == "__main__":
    main()
