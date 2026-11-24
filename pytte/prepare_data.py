"""Tokenize a text corpus into flat uint16 token files for training.

Input: text files where documents are separated by <|endoftext|> lines
(TinyStories format) or JSONL files with a "text" field (SagoSvenska).
Output: <out>/train.bin and <out>/val.bin, plus meta.json.

    python -m pytte.prepare_data --tok tok/ --train train.txt --val val.txt --out tokens/
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .tokenizer import EOT, load

CHUNK_DOCS = 20000


def iter_docs(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open() as f:
            for line in f:
                yield json.loads(line)["text"].strip()
        return
    buf = []
    with path.open() as f:
        for line in f:
            if line.strip() == EOT:
                if buf:
                    yield "".join(buf).strip()
                buf = []
            else:
                buf.append(line)
    if buf:
        yield "".join(buf).strip()


def tokenize_file(tok, paths, out_path):
    eot_id = tok.token_to_id(EOT)
    n_docs = n_tokens = 0
    with open(out_path, "wb") as out:
        for path in paths:
            docs = []
            for doc in iter_docs(path):
                docs.append(doc)
                if len(docs) == CHUNK_DOCS:
                    n_tokens += flush(tok, docs, eot_id, out)
                    n_docs += len(docs)
                    docs = []
                    print(f"  {path}: {n_docs} docs, {n_tokens/1e6:.1f}M tokens", end="\r", flush=True)
            if docs:
                n_tokens += flush(tok, docs, eot_id, out)
                n_docs += len(docs)
    print(f"  {out_path}: {n_docs} docs, {n_tokens/1e6:.2f}M tokens")
    return n_docs, n_tokens


def flush(tok, docs, eot_id, out):
    encs = tok.encode_batch(docs)
    ids = np.concatenate([np.array(e.ids + [eot_id], dtype=np.uint16) for e in encs])
    assert ids.max() < 65536
    ids.tofile(out)
    return len(ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tok", required=True)
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--val", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    tok = load(args.tok)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"vocab_size": tok.get_vocab_size(), "tokenizer": str(args.tok)}
    meta["train_docs"], meta["train_tokens"] = tokenize_file(tok, args.train, out / "train.bin")
    meta["val_docs"], meta["val_tokens"] = tokenize_file(tok, args.val, out / "val.bin")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
