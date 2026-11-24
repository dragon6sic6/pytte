"""Judge story sets (generator pilots) with one load of a judge model.

    python scripts/judge_pilots.py --judge mlx-community/gemma-4-12B-it-qat-4bit \
        v1=/Volumes/Modelltest/pytte/data/sagosvenska/raw \
        12b-t085=/Volumes/Modelltest/pytte/data/pilot-v2/12b-t085 ...

Each set is a directory of stories-*.jsonl (or one jsonl). The same --n random stories
(fixed seed) are judged per set. Writes <out>/judge-<judge short name>.json. Needs the GPU alone.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/Volumes/Modelltest/pytte/hf-cache")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pytte.judge import judge, summarize  # noqa: E402


def load_texts(path):
    p = Path(path)
    files = sorted(p.glob("stories-*.jsonl")) if p.is_dir() else [p]
    out = []
    for f in files:
        for line in open(f, encoding="utf-8"):
            if line.strip():
                out.append(json.loads(line)["text"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+", help="NAME=path")
    ap.add_argument("--judge", default="mlx-community/gemma-4-12B-it-qat-4bit")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="/Volumes/Modelltest/pytte/data/pilot-v2")
    a = ap.parse_args()

    from mlx_lm import load
    import mlx.core as mx
    mx.set_cache_limit(2 * 1024**3)
    mx.set_memory_limit(20 * 1024**3)
    print(f"[judge] {time.strftime('%FT%T')} loading {a.judge}", flush=True)
    loaded = load(a.judge)
    short = a.judge.rsplit("/", 1)[-1].split("-it")[0].lower()   # gemma-4-12b / gemma-4-26b-a4b
    report = {"judge": a.judge, "n": a.n, "seed": a.seed, "sets": {}}
    for s in a.sets:
        name, path = s.split("=", 1)
        texts = load_texts(path)
        rng = random.Random(a.seed)
        if len(texts) > a.n:
            texts = rng.sample(texts, a.n)
        print(f"[judge] {time.strftime('%FT%T')} {name}: {len(texts)} stories", flush=True)
        results = judge(texts, "sv", loaded=loaded, batch=a.batch)
        summary = summarize(results, "sv")
        report["sets"][name] = {"summary": summary, "results": results}
        print(f"[judge] {name}: " + json.dumps(summary, ensure_ascii=False), flush=True)
        mx.clear_cache()
    out = Path(a.out) / f"judge-{short}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[judge] {time.strftime('%FT%T')} wrote {out}", flush=True)


if __name__ == "__main__":
    main()
