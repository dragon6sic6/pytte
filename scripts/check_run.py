"""Check run.c against the Python implementation.

    python scripts/check_run.py --model ckpt/pytte-7m-sv3-chat --tok data/tok-sv3-4096 --data data/sagosvenska-mix1/clean

1. Tokenizer: encodes N validation stories and the SFT-style prompts with both and compares ids.
2. Forward pass: exports f32, generates greedily (temp 0) from several prompts in both, compares tokens.
3. int8: exports int8, same greedy prompts, reports how many tokens agree before the first divergence,
   and the speed of run.c for both files.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pytte.export import export  # noqa: E402
from pytte.model import Pytte  # noqa: E402
from pytte.sample import generate_ids  # noqa: E402
from pytte.tokenizer import EOT, load as load_tok  # noqa: E402

PROMPTS = ["Det var en gång en liten älg som", "Saga bor i en röd stuga.", "", "Hej!", "Plask! Vattnet var kallt",
           "Du: Berätta en saga om en räv.\nPytte:", "Du: Vem är du?\nPytte:", "Lisa gick till skogen 3 gånger, sa hon."]


def run_c(run, binf, *args):
    r = subprocess.run([run, binf, *args], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode(errors="replace"))
    return r.stdout.decode(errors="replace"), r.stderr.decode(errors="replace")


def c_tokens(run, binf, text):
    out, _ = run_c(run, binf, "-tok", text)
    first = out.splitlines()[0].strip()
    return [int(x) for x in first.split()] if first else []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tok", required=True)
    p.add_argument("--data", required=True, help="clean dir with val.jsonl")
    p.add_argument("--run", default="./run")
    p.add_argument("--n-stories", type=int, default=300)
    p.add_argument("--max-new", type=int, default=80)
    p.add_argument("--out-dir", default="/tmp/pytte-check")
    args = p.parse_args()
    tok = load_tok(args.tok)
    eot = tok.token_to_id(EOT)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    f32 = export(args.model, args.tok, out_dir / "f32.bin", f32=True)
    q8 = export(args.model, args.tok, out_dir / "q8.bin", f32=False)

    # 1. tokenizer
    texts = [json.loads(l)["text"] for l, _ in zip((args.data and open(Path(args.data) / "val.jsonl")), range(args.n_stories))]
    texts += [p for p in PROMPTS if p] + ["Du: Berätta en lång saga om Ebba och ett äpple.\nPytte: Ebba är glad.",
                                           "midsommarafton", "  två  mellanslag ", "Hej\n\nhej", "it's 7 o'clock", "räv–älg… 'ok' \"ja\""]
    bad = 0
    for t in texts:
        py = tok.encode(t).ids
        c = c_tokens(args.run, str(f32), t)
        if py != c:
            bad += 1
            if bad <= 5:
                print(f"TOKENIZER MISMATCH: {t[:60]!r}\n  py {py[:20]}\n  c  {c[:20]}")
    print(f"tokenizer: {len(texts) - bad}/{len(texts)} texts identical")

    # 2 + 3. greedy generation
    model = Pytte.load(args.model)
    for name, binf in (("f32", f32), ("int8", q8)):
        agree_total = 0; n_total = 0; exact = 0; speeds = []
        for pr in PROMPTS:
            ids = tok.encode(pr).ids if pr else [eot]
            py_text = generate_ids(model, tok, ids, args.max_new, temp=0)
            py_ids = tok.encode(py_text).ids if py_text else []
            out, err = run_c(args.run, str(binf), "-p", pr, "-n", str(args.max_new), "-t", "0", "-v")
            c_text = out[len(pr):] if pr else out
            c_text = c_text[:-1] if c_text.endswith("\n") else c_text
            c_ids = tok.encode(c_text).ids if c_text else []
            k = 0
            while k < min(len(py_ids), len(c_ids)) and py_ids[k] == c_ids[k]:
                k += 1
            agree_total += k; n_total += max(len(py_ids), len(c_ids))
            exact += (py_ids == c_ids)
            if "tok/s" in err:
                speeds.append(float(err.split(",")[1].split()[0]))
            if name == "f32" and py_ids != c_ids:
                print(f"  {name} differs after {k} tokens for {pr!r}:\n    py: {py_text[:120]!r}\n    c : {c_text[:120]!r}")
        print(f"{name}: {exact}/{len(PROMPTS)} greedy continuations identical to Python, "
              f"{agree_total}/{n_total} tokens agree before first divergence, run.c {sum(speeds)/len(speeds):.0f} tok/s")


if __name__ == "__main__":
    main()
