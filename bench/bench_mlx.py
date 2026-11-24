"""Training-throughput benchmark: Llama-style decoder in MLX.

Same architecture as bench_torch.py (RMSNorm, RoPE, causal MHA, SwiGLU,
tied embeddings, AdamW). Random tokens; we only measure tokens/second.
"""
import argparse
import math
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

CONFIGS = {
    "small": (4096, 128, 4, 4, 352),   # ~1.3M params
    "mid": (4096, 288, 6, 6, 768),     # ~7M params
}


class Block(nn.Module):
    def __init__(self, d, h, ff):
        super().__init__()
        self.h = h
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.w1 = nn.Linear(d, ff, bias=False)
        self.w2 = nn.Linear(ff, d, bias=False)
        self.w3 = nn.Linear(d, ff, bias=False)
        self.rope = nn.RoPE(d // h)
        self.scale = 1.0 / math.sqrt(d // h)

    def __call__(self, x):
        B, T, D = x.shape
        q, k, v = mx.split(self.qkv(self.n1(x)), 3, axis=-1)
        q = q.reshape(B, T, self.h, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.h, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.h, -1).transpose(0, 2, 1, 3)
        q, k = self.rope(q), self.rope(k)
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        x = x + self.o(a.transpose(0, 2, 1, 3).reshape(B, T, D))
        hn = self.n2(x)
        return x + self.w2(nn.silu(self.w1(hn)) * self.w3(hn))


class GPT(nn.Module):
    def __init__(self, V, d, L, h, ff):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.blocks = [Block(d, h, ff) for _ in range(L)]
        self.nf = nn.RMSNorm(d)

    def __call__(self, idx):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x)
        return self.emb.as_linear(self.nf(x))


def n_params(model):
    return sum(v.size for _, v in tree_flatten(model.parameters()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="small", choices=CONFIGS)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--no-compile", action="store_true")
    args = p.parse_args()

    V, d, L, h, ff = CONFIGS[args.config]
    mx.random.seed(0)
    model = GPT(V, d, L, h, ff)
    mx.eval(model.parameters())
    opt = optim.AdamW(learning_rate=1e-3, betas=[0.9, 0.95], weight_decay=0.1)

    def loss_fn(model, x, y):
        return nn.losses.cross_entropy(model(x), y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, opt.state]

    def raw_step(x, y):
        loss, grads = loss_and_grad(model, x, y)
        opt.update(model, grads)
        return loss

    if args.no_compile:
        step = raw_step
    else:
        step = partial(mx.compile, inputs=state, outputs=state)(raw_step)

    x = mx.random.randint(0, V, (args.batch, args.seq))
    y = mx.random.randint(0, V, (args.batch, args.seq))

    for _ in range(5):
        loss = step(x, y)
        mx.eval(loss, state)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        loss = step(x, y)
        mx.eval(loss, state)
    dt = time.perf_counter() - t0
    toks = args.batch * args.seq * args.steps
    mode = "nocompile" if args.no_compile else "compiled"
    print(
        f"mlx/{mode} config={args.config} params={n_params(model)/1e6:.2f}M "
        f"batch={args.batch} seq={args.seq} | {toks/dt:,.0f} tok/s | "
        f"{1000*dt/args.steps:.0f} ms/step | loss={loss.item():.3f}"
    )


if __name__ == "__main__":
    main()
