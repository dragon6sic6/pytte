"""Training-throughput benchmark: Llama-style decoder in PyTorch on MPS.

Same architecture as bench_mlx.py (RMSNorm, RoPE, causal MHA, SwiGLU,
tied embeddings, AdamW). Random tokens; we only measure tokens/second.
"""
import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIGS = {
    # name: (vocab, dim, layers, heads, ff)
    "small": (4096, 128, 4, 4, 352),   # ~1.3M params
    "mid": (4096, 288, 6, 6, 768),     # ~7M params
}


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


def rope_cache(T, hd, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2).float() / hd))
    t = torch.arange(T).float()
    f = torch.outer(t, inv)
    return torch.cos(f), torch.sin(f)


def apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d, h, ff):
        super().__init__()
        self.h = h
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.w1 = nn.Linear(d, ff, bias=False)
        self.w2 = nn.Linear(ff, d, bias=False)
        self.w3 = nn.Linear(d, ff, bias=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).split(D, -1)
        q = q.view(B, T, self.h, -1).transpose(1, 2)
        k = k.view(B, T, self.h, -1).transpose(1, 2)
        v = v.view(B, T, self.h, -1).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        hn = self.n2(x)
        return x + self.w2(F.silu(self.w1(hn)) * self.w3(hn))


class GPT(nn.Module):
    def __init__(self, V, d, L, h, ff, T):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.blocks = nn.ModuleList([Block(d, h, ff) for _ in range(L)])
        self.nf = RMSNorm(d)
        cos, sin = rope_cache(T, d // h)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, idx, targets):
        T = idx.shape[1]
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x, self.cos[:T], self.sin[:T])
        logits = self.nf(x) @ self.emb.weight.T
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="small", choices=CONFIGS)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--device", default="mps")
    args = p.parse_args()

    V, d, L, h, ff = CONFIGS[args.config]
    dev = torch.device(args.device)
    torch.manual_seed(0)
    model = GPT(V, d, L, h, ff, args.seq).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)

    x = torch.randint(0, V, (args.batch, args.seq), device=dev)
    y = torch.randint(0, V, (args.batch, args.seq), device=dev)

    def step():
        loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss

    def sync():
        if dev.type == "mps":
            torch.mps.synchronize()

    for _ in range(5):
        step()
    sync()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        loss = step()
    sync()
    dt = time.perf_counter() - t0
    toks = args.batch * args.seq * args.steps
    print(
        f"torch/{dev.type} config={args.config} params={n_params/1e6:.2f}M "
        f"batch={args.batch} seq={args.seq} | {toks/dt:,.0f} tok/s | "
        f"{1000*dt/args.steps:.0f} ms/step | loss={loss.item():.3f}"
    )


if __name__ == "__main__":
    main()
