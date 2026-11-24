"""Pytte: a decoder-only Transformer in the Llama style, written in MLX.

Every part of a modern LLM is here and nothing else:
  token embedding -> N x [RMSNorm -> causal attention with RoPE (+GQA) -> residual
                          -> RMSNorm -> SwiGLU MLP -> residual] -> RMSNorm -> tied LM head
plus a KV cache for fast autoregressive generation.
"""
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten


@dataclass
class PytteConfig:
    vocab_size: int = 4096
    dim: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 4          # < n_heads gives grouped-query attention
    ff_dim: int = 352            # SwiGLU hidden size, ~8/3 * dim rounded to a multiple of 32
    max_seq: int = 512
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    name: str = "pytte"

    @property
    def head_dim(self):
        return self.dim // self.n_heads

    @staticmethod
    def from_dict(d):
        return PytteConfig(**{k: v for k, v in d.items() if k in PytteConfig.__dataclass_fields__})


class KVCache:
    """Grows along the sequence axis. Simple and fine for small models."""

    def __init__(self):
        self.keys = None
        self.values = None

    @property
    def offset(self):
        return 0 if self.keys is None else self.keys.shape[2]

    def update(self, k, v):
        if self.keys is None:
            self.keys, self.values = k, v
        else:
            self.keys = mx.concatenate([self.keys, k], axis=2)
            self.values = mx.concatenate([self.values, v], axis=2)
        return self.keys, self.values


class Attention(nn.Module):
    def __init__(self, cfg: PytteConfig):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.scale = 1.0 / math.sqrt(cfg.head_dim)
        self.wq = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)
        self.rope = nn.RoPE(cfg.head_dim, traditional=False, base=cfg.rope_theta)

    def __call__(self, x, cache: KVCache | None = None):
        B, T, _ = x.shape
        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        offset = cache.offset if cache is not None else 0
        q, k = self.rope(q, offset=offset), self.rope(k, offset=offset)
        if cache is not None:
            k, v = cache.update(k, v)
        mask = "causal" if T > 1 else None
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.wo(out.transpose(0, 2, 1, 3).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg: PytteConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.dim, cfg.ff_dim, bias=False)   # gate
        self.w3 = nn.Linear(cfg.dim, cfg.ff_dim, bias=False)   # up
        self.w2 = nn.Linear(cfg.ff_dim, cfg.dim, bias=False)   # down

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: PytteConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = nn.RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def __call__(self, x, cache=None):
        x = x + self.attn(self.attn_norm(x), cache)
        return x + self.mlp(self.mlp_norm(x))


class Pytte(nn.Module):
    def __init__(self, cfg: PytteConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = [Block(cfg) for _ in range(cfg.n_layers)]
        self.norm = nn.RMSNorm(cfg.dim, eps=cfg.norm_eps)
        if not cfg.tie_embeddings:
            self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        """GPT-2 style: N(0, 0.02), residual projections scaled by 1/sqrt(2L)."""
        std = 0.02
        res_std = std / math.sqrt(2 * self.cfg.n_layers)
        new = {}
        for name, p in tree_flatten(self.parameters()):
            if name.endswith("weight") and ("norm" not in name):
                s = res_std if (".wo." in name or ".w2." in name) else std
                new[name] = mx.random.normal(p.shape, scale=s)
        self.update(tree_unflatten(list(new.items())))

    def __call__(self, tokens, caches=None):
        x = self.embed(tokens)
        for i, layer in enumerate(self.layers):
            x = layer(x, caches[i] if caches is not None else None)
        x = self.norm(x)
        if self.cfg.tie_embeddings:
            return self.embed.as_linear(x)
        return self.lm_head(x)

    def new_caches(self):
        return [KVCache() for _ in self.layers]

    # ---- parameter counting ----
    def n_params(self, include_embeddings=True):
        total = sum(v.size for k, v in tree_flatten(self.parameters()))
        if include_embeddings:
            return total
        return total - self.embed.weight.size

    # ---- save / load ----
    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(path / "model.safetensors"), dict(tree_flatten(self.parameters())))
        (path / "config.json").write_text(json.dumps(asdict(self.cfg), indent=2))

    @staticmethod
    def load(path):
        path = Path(path)
        cfg = PytteConfig.from_dict(json.loads((path / "config.json").read_text()))
        model = Pytte(cfg)
        weights = mx.load(str(path / "model.safetensors"))
        model.update(tree_unflatten(list(weights.items())))
        mx.eval(model.parameters())
        return model


# Model family. ff_dim ~ 8/3 * dim rounded to a multiple of 32.
FAMILY = {
    "pytte-0.3m": PytteConfig(dim=64, n_layers=4, n_heads=4, n_kv_heads=2, ff_dim=192, vocab_size=2048, name="pytte-0.3m"),
    "pytte-1m":   PytteConfig(dim=128, n_layers=4, n_heads=4, n_kv_heads=4, ff_dim=352, name="pytte-1m"),
    "pytte-2m":   PytteConfig(dim=160, n_layers=5, n_heads=5, n_kv_heads=5, ff_dim=448, name="pytte-2m"),
    "pytte-3m":   PytteConfig(dim=192, n_layers=6, n_heads=6, n_kv_heads=6, ff_dim=512, name="pytte-3m"),
    "pytte-7m":   PytteConfig(dim=288, n_layers=6, n_heads=6, n_kv_heads=6, ff_dim=768, name="pytte-7m"),
    "pytte-15m":  PytteConfig(dim=384, n_layers=8, n_heads=6, n_kv_heads=6, ff_dim=1024, name="pytte-15m"),
    "pytte-40m":  PytteConfig(dim=512, n_layers=12, n_heads=8, n_kv_heads=8, ff_dim=1408, name="pytte-40m"),
}


if __name__ == "__main__":
    for name, cfg in FAMILY.items():
        m = Pytte(cfg)
        tot, body = m.n_params(True), m.n_params(False)
        print(f"{name:11s} dim={cfg.dim:3d} L={cfg.n_layers:2d} H={cfg.n_heads} ff={cfg.ff_dim:4d} "
              f"vocab={cfg.vocab_size} | {tot/1e6:5.2f}M total, {body/1e6:5.2f}M without embeddings "
              f"({100*(tot-body)/tot:.0f}% embeddings)")
