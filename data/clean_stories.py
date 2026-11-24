"""Clean, filter, deduplicate and split the raw SagoSvenska stories.

    python clean_stories.py --raw raw/ --out clean/

Steps: normalize text -> drop junk (model commentary, markdown, English, too short/long,
unsafe words) -> exact dedup -> near-dup dedup (MinHash over word 3-grams) ->
deterministic train/val/test split by content hash -> stats.json.
"""
import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

SV_STOP = set("och att det en ett är som på i var hon han de den jag du vi ni inte med för till av om "
              "så men här där nu då när hade har kan ska vill sa säger gick går kom kommer".split())
EN_WORDS = re.compile(r"\b(the|and|with|was|were|she|he|they|said|very|little|happy|then|there)\b")
UNSAFE = re.compile(
    r"\b(dö|dör|död|döda|dog|drunkna\w*|blod\w*|skadad\w*|skjut\w*|mörda\w*|kniv\w*|vapen|"
    r"pistol\w*|krig\w*|misshandl\w*|självmord|begravn\w*|kista|sjukhus\w*|ambulans\w*)\b", re.I)
JUNK_LINE = re.compile(r"^(här är|saga:|rubrik|titel|\*\*|#|okej|visst|självklart)", re.I)
MARKDOWN = re.compile(r"[*_#`>]+")
# Gemma 4 (4-bit) sometimes emits one of its own special tokens, nearly always the end-of-image
# marker <image|>, in the middle of a word ("gung<image|>a", "l<image|>åda") or right after a
# full stop. 12 % of v1 raw stories had one. Removing the token restores the word in almost
# every case; after punctuation it becomes a space. v1.0 only stripped the ">" (MARKDOWN above),
# which left "<image|" in 7 097 training stories and taught the tokenizer an "image" token.
SPECIAL_TOKEN = re.compile(r"<\|?[A-Za-z_\"]{0,20}\|?>")


def repair_special_tokens(text):
    """Remove leaked generator special tokens. Returns (text, number removed)."""
    n = 0

    def sub(m):
        nonlocal n
        n += 1
        before = text[m.start() - 1] if m.start() > 0 else " "
        after = text[m.end()] if m.end() < len(text) else " "
        return "" if before.isalpha() and after.isalpha() else " "

    return SPECIAL_TOKEN.sub(sub, text), n


def normalize(text):
    text = unicodedata.normalize("NFC", text)
    text = text.replace("”", '"').replace("“", '"').replace("„", '"').replace("’", "'").replace("‘", "'")
    text = text.replace("–", "-").replace("—", "-").replace("…", "...")
    text = MARKDOWN.sub("", text)
    lines = [l.strip() for l in text.split("\n")]
    lines = [l for l in lines if not JUNK_LINE.match(l)]
    text = "\n".join(lines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def swedish_ratio(words):
    return sum(w in SV_STOP for w in words) / max(1, len(words))


def shingles(words, n=3):
    return {hash(" ".join(words[i:i + n])) for i in range(max(1, len(words) - n + 1))}


def minhash(sh, seeds):
    return [min((s ^ seed) & 0xFFFFFFFFFFFF for s in sh) for seed in seeds]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="/Volumes/Modelltest/pytte/data/sagosvenska/raw")
    p.add_argument("--out", default="/Volumes/Modelltest/pytte/data/sagosvenska/clean")
    p.add_argument("--min-words", type=int, default=40)
    p.add_argument("--max-words", type=int, default=400)
    p.add_argument("--jaccard", type=float, default=0.7, help="near-duplicate threshold")
    p.add_argument("--val-frac", type=float, default=0.01)
    args = p.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for f in sorted(raw.glob("stories-*.jsonl")):
        with f.open() as fh:
            records.extend(json.loads(l) for l in fh if l.strip())
    n_main = len(records)
    # Stories the generator flagged as "unsafe" used a broader word list (e.g. "slår" as in
    # "hjärtat slår"); re-test them against the stricter UNSAFE below instead of dropping blindly.
    for f in sorted(raw.glob("review-*.jsonl")):
        with f.open() as fh:
            records.extend(r for r in (json.loads(l) for l in fh if l.strip()) if r.get("reason") == "unsafe")
    print(f"read {n_main} raw stories + {len(records) - n_main} flagged for re-test")

    drops = Counter()
    kept = []
    seen_exact = set()
    repaired = 0
    for r in records:
        text, n_rep = repair_special_tokens(r["text"])
        repaired += n_rep > 0
        text = normalize(text)
        if "<" in text or "|" in text:
            drops["leftover_special_token"] += 1; continue
        words = re.findall(r"\w+", text.lower())
        if len(words) < args.min_words:
            drops["short"] += 1; continue
        if len(words) > args.max_words:
            drops["long"] += 1; continue
        if swedish_ratio(words) < 0.18:
            drops["not_swedish"] += 1; continue
        if EN_WORDS.search(text.lower()):
            drops["english_words"] += 1; continue
        if UNSAFE.search(text):
            drops["unsafe"] += 1; continue
        if text.count('"') % 2 == 1:
            drops["unbalanced_quotes"] += 1; continue
        key = hashlib.sha1(re.sub(r"\W+", "", text.lower()).encode()).hexdigest()
        if key in seen_exact:
            drops["exact_dup"] += 1; continue
        seen_exact.add(key)
        r = dict(r); r["text"] = text; r["n_words"] = len(words); r["_words"] = words; r["_key"] = key
        kept.append(r)

    # Near-duplicate removal with MinHash LSH (32 hashes, 8 bands of 4).
    seeds = [int.from_bytes(hashlib.sha1(str(i).encode()).digest()[:6], "big") for i in range(32)]
    buckets = {}
    sh_cache = []
    final = []
    for r in kept:
        sh = shingles(r["_words"])
        mh = minhash(sh, seeds)
        dup = False
        cand = set()
        for b in range(8):
            band = tuple(mh[b * 4:(b + 1) * 4])
            cand.update(buckets.get((b, band), ()))
        for j in cand:
            other = sh_cache[j]
            jac = len(sh & other) / len(sh | other)
            if jac >= args.jaccard:
                dup = True
                break
        if dup:
            drops["near_dup"] += 1
            continue
        idx = len(sh_cache)
        sh_cache.append(sh)
        for b in range(8):
            buckets.setdefault((b, tuple(mh[b * 4:(b + 1) * 4])), []).append(idx)
        final.append(r)

    # Deterministic split by content hash.
    splits = {"train": [], "val": [], "test": []}
    for r in final:
        h = int(r["_key"][:8], 16) / 0xFFFFFFFF
        name = "val" if h < args.val_frac else "test" if h < 2 * args.val_frac else "train"
        splits[name].append(r)

    for name, rows in splits.items():
        with (out / f"{name}.jsonl").open("w") as fh:
            for r in rows:
                clean = {k: v for k, v in r.items() if not k.startswith("_")}
                fh.write(json.dumps(clean, ensure_ascii=False) + "\n")

    n_words = sum(r["n_words"] for r in final)
    vocab = Counter(w for r in final for w in r["_words"])
    stats = {
        "raw": len(records), "kept": len(final), "dropped": dict(drops),
        "repaired_special_token_stories": repaired,
        "split": {k: len(v) for k, v in splits.items()},
        "words": n_words, "avg_words": n_words / max(1, len(final)),
        "unique_words": len(vocab),
        "words_covering_90pct": next(i for i, c in enumerate(_cum(vocab), 1) if c >= 0.9 * n_words),
        "names": Counter(r["name"] for r in final).most_common(5),
        "features": Counter(r["feature"] for r in final).most_common(3),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


def _cum(counter):
    tot = 0
    for _, c in counter.most_common():
        tot += c
        yield tot


if __name__ == "__main__":
    main()
