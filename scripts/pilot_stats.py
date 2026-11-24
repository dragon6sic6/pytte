"""Compare story sets without a GPU: how varied and how template-like they are.

Usage: pilot_stats.py NAME=path/to/dir_or_jsonl ... [--n 500] [--seed 1]

Each set is sampled down to --n stories (fixed seed) so the vocabulary numbers are
comparable. Reads every stories-*.jsonl in a directory, or one jsonl file.
"""
import argparse, json, random, re, sys
from collections import Counter
from pathlib import Path


def load(path):
    p = Path(path)
    files = sorted(p.glob("stories-*.jsonl")) if p.is_dir() else [p]
    out = []
    for f in files:
        for line in open(f, encoding="utf-8"):
            if line.strip():
                out.append(json.loads(line)["text"])
    return out


def words_of(t):
    return re.findall(r"\w+", t.lower())


def first_sentence(t):
    return re.split(r"(?<=[.!?])\s", t.strip(), 1)[0]


def stats(texts):
    W = [words_of(t) for t in texts]
    n_words = sum(len(w) for w in W)
    vocab = Counter(w for ws in W for w in ws)
    cum, k90 = 0, 0
    for i, (_, c) in enumerate(vocab.most_common(), 1):
        cum += c
        if cum >= 0.9 * n_words:
            k90 = i
            break
    openings = Counter(" ".join(ws[:4]) for ws in W)
    first = Counter(first_sentence(t) for t in texts)
    # Cross-story reuse: share of a story's 4-grams that also occur in another story of the set.
    grams = [set(zip(ws, ws[1:], ws[2:], ws[3:])) for ws in W]
    count = Counter(g for gs in grams for g in gs)
    reuse = sum(sum(1 for g in gs if count[g] > 1) for gs in grams) / max(1, sum(len(gs) for gs in grams))
    # Dialogue share and average sentence length.
    n_sent = sum(len(re.findall(r"[.!?]", t)) for t in texts)
    quotes = sum(1 for t in texts if '"' in t or "”" in t) / len(texts)
    return {
        "n": len(texts),
        "avg_words": n_words / len(texts),
        "avg_sent_len": n_words / max(1, n_sent),
        "unique_words": len(vocab),
        "hapax_share": sum(1 for c in vocab.values() if c == 1) / len(vocab),
        "words_90pct": k90,
        "distinct_openings": len(openings) / len(texts),
        "top_opening_share": openings.most_common(1)[0][1] / len(texts),
        "top_first_sentence": first.most_common(1)[0][1],
        "cross_story_4gram_reuse": reuse,
        "dialogue_share": quotes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+", help="NAME=path")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rows = {}
    for s in a.sets:
        name, path = s.split("=", 1)
        texts = load(path)
        rng = random.Random(a.seed)
        if len(texts) > a.n:
            texts = rng.sample(texts, a.n)
        rows[name] = stats(texts)
    keys = [("n", "Sagor", "{:d}"), ("avg_words", "Ord/saga", "{:.0f}"), ("avg_sent_len", "Ord/mening", "{:.1f}"),
            ("unique_words", "Unika ord", "{:d}"), ("hapax_share", "Andel engångsord", "{:.1%}"),
            ("words_90pct", "Ord som täcker 90 %", "{:d}"), ("distinct_openings", "Olika öppningar (4 ord)", "{:.1%}"),
            ("top_opening_share", "Vanligaste öppningen", "{:.1%}"), ("top_first_sentence", "Vanligaste första meningen (antal)", "{:d}"),
            ("cross_story_4gram_reuse", "4-gram som återkommer i andra sagor", "{:.1%}"), ("dialogue_share", "Sagor med dialog", "{:.0%}")]
    print("| Mått | " + " | ".join(rows) + " |")
    print("|---|" + "---:|" * len(rows))
    for k, label, fmt in keys:
        print(f"| {label} | " + " | ".join(fmt.format(r[k]) for r in rows.values()) + " |")


if __name__ == "__main__":
    main()
