"""Export a Pytte checkpoint + tokenizer to one self-contained binary for run.c and the web demo.

    python -m pytte.export --model ckpt/pytte-7m-sv3-chat --tok data/tok-sv3-4096 --out pytte-7m-chat.bin
    python -m pytte.export ... --f32          # no quantisation (for checking run.c against Python)

File layout (little-endian), version 1:

    magic "PYTT", i32 version=1
    i32 dim, n_layers, n_heads, n_kv_heads, ff_dim, vocab_size, max_seq, quant (0=f32, 1=int8), tie_embeddings
    f32 rope_theta, norm_eps
    --- tokenizer (byte-level BPE, the only special token is <|endoftext|>) ---
    i32 n_vocab; per token: u8 length, raw bytes
    i32 eot_id
    i32 n_merges; per merge (in rank order): i32 left_id, right_id, result_id
    --- weights, in forward order; every Linear is [out, in] row-major ---
    embed [vocab, dim]
    per layer: attn_norm[dim] f32, wq, wk, wv, wo, mlp_norm[dim] f32, w1 (gate), w3 (up), w2 (down)
    norm[dim] f32
    An int8 matrix is stored as int8 values [rows*cols] followed by f32 scales[rows]
    (symmetric per-row: w = q * scale). Norm vectors are always f32.
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten

from .model import Pytte
from .tokenizer import EOT

MAGIC = b"PYTT"
VERSION = 1


def bytes_to_unicode():
    """GPT-2's reversible byte -> printable unicode map (what the ByteLevel pre-tokenizer uses)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def load_tokenizer_tables(tok_path):
    """vocab as list of byte strings (index = id), eot id, merges as (left, right, result) ids."""
    p = Path(tok_path)
    if p.is_dir():
        p = p / "tokenizer.json"
    j = json.loads(p.read_text())
    assert j["model"]["type"] == "BPE"
    pre = j.get("pre_tokenizer") or {}
    assert pre.get("type") == "ByteLevel" and not pre.get("add_prefix_space", False), pre
    u2b = {v: k for k, v in bytes_to_unicode().items()}
    vocab = j["model"]["vocab"]
    specials = {t["content"]: t["id"] for t in j.get("added_tokens", [])}
    n = max(max(vocab.values()), max(specials.values(), default=-1)) + 1
    table = [b""] * n
    for s, i in vocab.items():
        if s in specials:
            continue
        table[i] = bytes(u2b[c] for c in s)
    for s, i in specials.items():
        table[i] = s.encode()
    merges = []
    for m in j["model"]["merges"]:
        a, b = m if isinstance(m, list) else m.split(" ")
        merges.append((vocab[a], vocab[b], vocab[a + b]))
    assert EOT in specials, "tokenizer has no <|endoftext|>"
    return table, specials[EOT], merges


def quantize_rows(w):
    """Symmetric int8 per output row. Returns (int8 [rows, cols], f32 scales [rows])."""
    w = np.asarray(w, dtype=np.float32)
    amax = np.abs(w).max(axis=1)
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.rint(w / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale


def export(model_dir, tok_dir, out_path, f32=False):
    model = Pytte.load(model_dir)
    cfg = model.cfg
    params = dict(tree_flatten(model.parameters()))
    w = {k: np.array(v.astype(mx.float32)) for k, v in params.items()}
    table, eot, merges = load_tokenizer_tables(tok_dir)
    assert len(table) == cfg.vocab_size, (len(table), cfg.vocab_size)
    assert cfg.tie_embeddings, "run.c assumes tied embeddings"
    quant = 0 if f32 else 1

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_q8 = n_f32 = 0
    with out.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<10i", VERSION, cfg.dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.ff_dim,
                            cfg.vocab_size, cfg.max_seq, quant, 1))
        f.write(struct.pack("<2f", cfg.rope_theta, cfg.norm_eps))
        # tokenizer
        f.write(struct.pack("<i", len(table)))
        for t in table:
            assert len(t) < 256
            f.write(struct.pack("<B", len(t)))
            f.write(t)
        f.write(struct.pack("<i", eot))
        f.write(struct.pack("<i", len(merges)))
        for a, b, c in merges:
            f.write(struct.pack("<3i", a, b, c))

        def mat(name):
            nonlocal n_q8, n_f32
            m = w[name]
            assert m.ndim == 2, name
            if quant:
                q, s = quantize_rows(m)
                f.write(q.tobytes()); f.write(s.tobytes())
                n_q8 += m.size
            else:
                f.write(m.astype(np.float32).tobytes())
                n_f32 += m.size

        def vec(name):
            nonlocal n_f32
            v = w[name]
            assert v.ndim == 1, name
            f.write(v.astype(np.float32).tobytes())
            n_f32 += v.size

        mat("embed.weight")
        for i in range(cfg.n_layers):
            L = f"layers.{i}."
            vec(L + "attn_norm.weight")
            for n in ("wq", "wk", "wv", "wo"):
                mat(L + f"attn.{n}.weight")
            vec(L + "mlp_norm.weight")
            for n in ("w1", "w3", "w2"):
                mat(L + f"mlp.{n}.weight")
        vec("norm.weight")
    size = out.stat().st_size
    print(f"{out}: {size/1e6:.2f} MB ({'f32' if f32 else 'int8'}; {n_q8/1e6:.2f}M int8 + {n_f32/1e6:.3f}M f32 params, "
          f"{len(table)} tokens, {len(merges)} merges, eot={eot})")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tok", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--f32", action="store_true", help="export float32 weights instead of int8")
    args = p.parse_args()
    export(args.model, args.tok, args.out, args.f32)


if __name__ == "__main__":
    main()
