"""Measure what int8 export costs: write a fake-quantised checkpoint (int8 round-trip of every
matrix, exactly what run.c computes with) and run the usual tests on it.

    python scripts/q8_eval.py --model ckpt/pytte-7m-sv3-chat --tok data/tok-sv3-4096 \
        --data data/sagosvenska-mix1/clean --lexicon data/sft-mix1/lexicon.json

Writes <model>-q8/ (model.safetensors, config.json) next to the original, then
eval_pairs.json (minimal pairs) and, for chat models, eval.json via scripts/eval_sft.py.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pytte.export import quantize_rows  # noqa: E402
from pytte.model import Pytte  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tok", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--lexicon", default=None, help="run eval_sft too (chat models)")
    p.add_argument("--n", type=int, default=100)
    args = p.parse_args()
    src = Path(args.model)
    dst = src.parent / (src.name + "-q8")
    model = Pytte.load(src)
    params = dict(tree_flatten(model.parameters()))
    out = {}
    err = []
    for k, v in params.items():
        a = np.array(v.astype(mx.float32))
        if a.ndim == 2:
            q, s = quantize_rows(a)
            deq = (q.astype(np.float32) * s[:, None]).astype(np.float32)
            err.append(float(np.abs(deq - a).mean() / (np.abs(a).mean() + 1e-12)))
            out[k] = mx.array(deq)
        else:
            out[k] = v
    dst.mkdir(exist_ok=True)
    mx.save_safetensors(str(dst / "model.safetensors"), out)
    (dst / "config.json").write_text((src / "config.json").read_text())
    print(f"{dst}: mean relative weight error {np.mean(err):.4f} (max over matrices {np.max(err):.4f})")
    py = sys.executable
    subprocess.run([py, "-m", "pytte.eval", "--model", str(dst), "--tok", args.tok, "--pairs", "eval/minimal_pairs_sv.json",
                    "--n-samples", "0", "--out", str(dst / "eval_pairs.json")], check=True)
    if args.lexicon:
        subprocess.run([py, "scripts/eval_sft.py", "--model", str(dst), "--tok", args.tok, "--data", args.data,
                        "--lexicon", args.lexicon, "--n", str(args.n)], check=True)
    for name in ("eval_pairs.json", "eval.json"):
        for d in (src, dst):
            f = d / name
            if f.exists():
                j = json.loads(f.read_text())
                if name == "eval_pairs.json":
                    print(f"{d.name:28s} minimal pairs {j['minimal_pairs:minimal_pairs_sv']['accuracy']:.3f}")
                else:
                    i = j["instruction_following"]
                    print(f"{d.name:28s} ingredients followed {i['ingredient_hit_rate']:.3f}, all {i['all_ingredients']:.2f}, "
                          f"ended {i['ended_naturally']:.2f}, follow-ups keep name {j['follow_ups']['keeps_name']:.2f}")


if __name__ == "__main__":
    main()
