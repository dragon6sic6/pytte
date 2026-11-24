"""Judge the generated samples of every trained checkpoint with one load of the judge model.

    python scripts/judge_family.py --n 50            # all <root>/pytte-*-{sv,en} with eval.json
    python scripts/judge_family.py --only pytte-1m-sv pytte-1m-en

Writes <ckpt>/judge.json (same format as pytte.judge). Needs the GPU alone.
"""
import argparse
import json
import os
import sys

# The judge model lives in the project's cache on the external disk; without this,
# mlx-lm downloaded a second 7 GB copy to ~/.cache/huggingface (2026-11-09).
os.environ.setdefault("HF_HOME", "/Volumes/Modelltest/pytte/hf-cache")
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pytte.judge import JUDGE_MODEL, judge, summarize  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--root", default="/Volumes/Modelltest/pytte/checkpoints")
p.add_argument("--n", type=int, default=50)
p.add_argument("--only", nargs="*", default=None)
p.add_argument("--force", action="store_true", help="re-judge even if judge.json exists")
p.add_argument("--batch", type=int, default=8, help="stories per generation batch (16 ran 95 min, then the Mac kernel-panicked)")
args = p.parse_args()

from mlx_lm import load  # noqa: E402
import mlx.core as mx  # noqa: E402
mx.set_cache_limit(2 * 1024**3)
mx.set_memory_limit(20 * 1024**3)

dirs = sorted(d for d in Path(args.root).iterdir() if (d / "eval.json").exists()
              and d.name.rsplit("-", 1)[1] in ("sv", "en", "en8m", "sv2", "sv3", "chat")
              and (not args.only or d.name in args.only))
print(f"[judge] {time.strftime('%FT%T')} loading {JUDGE_MODEL}", flush=True)
loaded = load(JUDGE_MODEL)
for d in dirs:
    if (d / "judge.json").exists() and not args.force:
        print(f"[judge] skip {d.name} (judge.json exists)", flush=True)
        continue
    lang = "en" if "-en" in d.name else "sv"
    texts = json.loads((d / "eval.json").read_text())["samples"][:args.n]
    print(f"[judge] {time.strftime('%FT%T')} {d.name}: {len(texts)} samples ({lang})", flush=True)
    results = judge(texts, lang, loaded=loaded, batch=args.batch)
    summary = summarize(results, lang)
    (d / "judge.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False))
    print(f"[judge] {d.name}: " + json.dumps(summary, ensure_ascii=False), flush=True)
    mx.clear_cache()
print(f"[judge] {time.strftime('%FT%T')} done", flush=True)
