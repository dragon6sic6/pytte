"""Run the minimal-pair grammar test through run.c (its -score mode), so the exported int8 file
and the C forward pass are tested end to end with the same test as the Python models.

    python scripts/pairs_run.py bin/pytte-7m.bin [more .bin files]  [--pairs eval/minimal_pairs_sv.json]
"""
import argparse
import json
import subprocess
from pathlib import Path


def score(run, binf, text):
    out = subprocess.run([run, binf, "-score", text], capture_output=True, text=True, check=True).stdout
    lp, n = out.split()
    return float(lp), int(n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bins", nargs="+")
    p.add_argument("--pairs", default="eval/minimal_pairs_sv.json")
    p.add_argument("--run", default="./run")
    args = p.parse_args()
    pairs = json.loads(Path(args.pairs).read_text())["pairs"]
    for binf in args.bins:
        per = {}
        for pr in pairs:
            g, _ = score(args.run, binf, pr["good"])
            b, _ = score(args.run, binf, pr["bad"])
            c = per.setdefault(pr["cat"], [0, 0])
            c[0] += int(g > b); c[1] += 1
        hits = sum(c[0] for c in per.values()); n = sum(c[1] for c in per.values())
        print(f"{Path(binf).name:22s} {100*hits/n:5.1f} % of {n} | " +
              " ".join(f"{k}={v[0]/v[1]:.2f}" for k, v in sorted(per.items())))


if __name__ == "__main__":
    main()
