"""Train a Pytte model from scratch with AdamW, warmup + cosine schedule, grad clipping.

Either a fixed number of steps or a wall-clock budget (the schedule then follows
elapsed time, so throughput changes do not break it).

    python -m pytte.train --config pytte-1m --data tokens-en-4096 --out ckpt/pytte-1m-en --minutes 5
"""
import argparse
import json
import math
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .model import FAMILY, Pytte, PytteConfig
from .sample import generate
from .tokenizer import load as load_tok


class TokenData:
    """Random fixed-length windows from a flat uint16 token file."""

    def __init__(self, path, seq, seed=0, max_ram_bytes=4 << 30):
        # Random windows over a memmap on a USB disk are I/O-bound (~30k tok/s measured on
        # TinyStories' 1.1 GB file); read the whole file into RAM when it fits.
        path = Path(path)
        if path.stat().st_size <= max_ram_bytes:
            self.data = np.fromfile(path, dtype=np.uint16)
        else:
            self.data = np.memmap(path, dtype=np.uint16, mode="r")
        # Optional loss mask (chat data from pytte.prepare_sft): <name>.mask.bin, one uint8 per
        # token, 1 where the loss is computed (assistant tokens). Absent = plain pretraining.
        mask_path = path.with_name(path.stem + ".mask.bin")
        self.mask = np.fromfile(mask_path, dtype=np.uint8) if mask_path.exists() else None
        self.seq = seq
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.data)

    def batch(self, batch_size):
        idx = self.rng.integers(0, len(self.data) - self.seq - 1, batch_size)
        x = np.stack([self.data[i:i + self.seq] for i in idx]).astype(np.int32)
        y = np.stack([self.data[i + 1:i + self.seq + 1] for i in idx]).astype(np.int32)
        if self.mask is None:
            return mx.array(x), mx.array(y), None
        m = np.stack([self.mask[i + 1:i + self.seq + 1] for i in idx]).astype(np.float32)
        return mx.array(x), mx.array(y), mx.array(m)


def loss_fn(model, x, y, m=None):
    if m is None:
        return nn.losses.cross_entropy(model(x), y, reduction="mean")
    ce = nn.losses.cross_entropy(model(x), y, reduction="none")
    return (ce * m).sum() / mx.maximum(m.sum(), 1.0)


def evaluate(model, data, batch_size, n_batches):
    losses = [loss_fn(model, *data.batch(batch_size)) for _ in range(n_batches)]
    return float(mx.mean(mx.stack(losses)))


def bytes_per_token(tok, data, n=200_000):
    # Divide by the number of tokens actually decoded: a val set smaller than n
    # (SagoSvenska has 85k val tokens) otherwise gives a bytes/token that is too low.
    ids = np.array(data.data[:n]).tolist()
    return len(tok.decode(ids).encode("utf-8")) / len(ids)


def cosine_lr(frac, lr, min_lr, warm_frac):
    """Learning rate at progress frac in [0, 1]: linear warmup, then cosine to min_lr."""
    if frac < warm_frac:
        return lr * frac / warm_frac
    f = min(1.0, (frac - warm_frac) / max(1e-9, 1 - warm_frac))
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="pytte-1m", help="name in FAMILY or path to config.json")
    p.add_argument("--data", required=True, help="dir with train.bin, val.bin, meta.json")
    p.add_argument("--tok", default=None, help="tokenizer dir, for samples and bits/byte")
    p.add_argument("--out", required=True)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--minutes", type=float, default=None, help="wall-clock budget instead of steps")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=float, default=0.05, help="fraction of training used for warmup")
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--snapshots", default=None, help="comma-separated steps at which a text sample is written to "
                   "snapshots.jsonl (0 = before training); the last step is always added. Needs --tok")
    p.add_argument("--snap-prompt", default="", help="the text the snapshot samples continue")
    p.add_argument("--init", default=None, help="checkpoint dir to start from (fine-tuning); its config.json is used")
    args = p.parse_args()

    if (args.steps is None) == (args.minutes is None):
        p.error("give exactly one of --steps or --minutes")
    mx.random.seed(args.seed)
    data_dir = Path(args.data)
    meta = json.loads((data_dir / "meta.json").read_text())

    if args.init:
        model = Pytte.load(args.init)
        cfg = model.cfg
        assert cfg.vocab_size == meta["vocab_size"], "init checkpoint and data use different vocabularies"
        cfg.max_seq = max(cfg.max_seq, args.seq)
    else:
        cfg = FAMILY[args.config] if args.config in FAMILY else \
            PytteConfig.from_dict(json.loads(Path(args.config).read_text()))
        cfg.vocab_size = meta["vocab_size"]
        cfg.max_seq = max(cfg.max_seq, args.seq)
        model = Pytte(cfg)
    mx.eval(model.parameters())
    n_total, n_body = model.n_params(True), model.n_params(False)

    train = TokenData(data_dir / "train.bin", args.seq, args.seed)
    val = TokenData(data_dir / "val.bin", args.seq, args.seed + 1)
    tok = load_tok(args.tok) if args.tok else None
    bpt = bytes_per_token(tok, val) if tok else None

    tokens_per_step = args.batch * args.seq
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "log.jsonl").open("a")
    budget = args.minutes * 60 if args.minutes else None

    print(f"{cfg.name}: {n_total/1e6:.2f}M params ({n_body/1e6:.2f}M without embeddings) | "
          f"train {len(train)/1e6:.1f}M tokens, val {len(val)/1e6:.1f}M | batch {args.batch}x{args.seq} "
          f"= {tokens_per_step} tokens/step" + (f" | {bpt:.2f} bytes/token" if bpt else "")
          + (f" | budget {args.minutes} min" if budget else f" | {args.steps} steps")
          + (f" | init {args.init}" if args.init else "") + (" | loss on masked tokens" if train.mask is not None else ""),
          flush=True)

    opt = optim.AdamW(learning_rate=args.lr, betas=[0.9, 0.95], weight_decay=args.wd)
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, opt.state]

    def raw_step(x, y, m=None):
        loss, grads = loss_and_grad(model, x, y, m)
        if args.grad_clip > 0:
            grads, _ = optim.clip_grad_norm(grads, args.grad_clip)
        opt.update(model, grads)
        return loss

    step = raw_step if args.no_compile else partial(mx.compile, inputs=state, outputs=state)(raw_step)

    def progress(it, elapsed):
        return elapsed / budget if budget else it / args.steps

    def do_eval(it):
        nonlocal best_val
        vl = evaluate(model, val, args.batch, args.eval_batches)
        rec = {"step": it, "val_loss": vl, "val_ppl": math.exp(vl)}
        if bpt:
            rec["val_bpb"] = vl / math.log(2) / bpt
        print(f"  eval step {it}: val loss {vl:.3f} | ppl {math.exp(vl):.2f}"
              + (f" | {rec['val_bpb']:.3f} bits/byte" if bpt else ""), flush=True)
        log.write(json.dumps(rec) + "\n"); log.flush()
        if vl < best_val:
            best_val = vl
            model.save(out)

    t_start = time.monotonic()
    t_log = t_start
    snaps = {int(v) for v in args.snapshots.split(",")} if args.snapshots else set()

    def do_snap(it, loss=None):
        text = generate(model, tok, args.snap_prompt, max_new=70, temp=0.8, top_p=0.95)
        rec = {"step": it, "tokens": it * tokens_per_step, "elapsed": time.monotonic() - t_start, "loss": loss, "text": text}
        with (out / "snapshots.jsonl").open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if 0 in snaps:
        do_snap(0)
    best_val = math.inf
    last_eval = 0
    it = 0
    while True:
        it += 1
        elapsed = time.monotonic() - t_start
        frac = progress(it, elapsed)
        if frac >= 1.0:
            it -= 1
            break
        opt.learning_rate = mx.array(cosine_lr(frac, args.lr, args.min_lr, args.warmup))
        x, y, m = train.batch(args.batch)
        loss = step(x, y) if m is None else step(x, y, m)
        mx.eval(loss, state)

        if it % args.log_every == 0:
            now = time.monotonic()
            tps = args.log_every * tokens_per_step / (now - t_log)
            t_log = now
            rec = {"step": it, "loss": float(loss), "lr": float(opt.learning_rate),
                   "tok_s": tps, "tokens": it * tokens_per_step, "elapsed": now - t_start}
            total = f"/{args.steps}" if args.steps else ""
            print(f"step {it:6d}{total} | loss {rec['loss']:.3f} | lr {rec['lr']:.2e} | "
                  f"{tps:,.0f} tok/s | {rec['elapsed']/60:.1f} min", flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
        if it in snaps:
            do_snap(it, float(loss))
        if it % args.eval_every == 0:
            do_eval(it)
            last_eval = it

    if last_eval != it:
        do_eval(it)
    if snaps and it not in snaps:
        do_snap(it, float(loss))
    total_time = time.monotonic() - t_start
    summary = {"name": cfg.name, "params": n_total, "params_no_embed": n_body, "steps": it,
               "tokens": it * tokens_per_step, "train_minutes": total_time / 60,
               "tok_s": it * tokens_per_step / total_time,
               "best_val_loss": best_val, "best_val_ppl": math.exp(best_val),
               "bytes_per_token": bpt, "best_val_bpb": (best_val / math.log(2) / bpt) if bpt else None,
               "config": cfg.__dict__, "args": vars(args)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"done: {it} steps, {it*tokens_per_step/1e6:.1f}M tokens, {total_time/60:.1f} min, "
          f"{summary['tok_s']:,.0f} tok/s, best val loss {best_val:.3f} (ppl {math.exp(best_val):.2f})",
          flush=True)

    if tok:
        best = Pytte.load(out)
        if train.mask is not None:
            from .chat import reply
            for q in ["Berätta en saga om en älg.", "Hej!", "Vad händer sedan?"]:
                print(f"\n> {q}", reply(best, tok, [{"role": "user", "content": q}], max_new=150), flush=True)
        else:
            for prompt in ["Once upon a time", "The little dog", "Lily wanted to"]:
                print(f"\n> {prompt}", generate(best, tok, prompt, max_new=150, temp=0.8, top_p=0.95), flush=True)


if __name__ == "__main__":
    main()
