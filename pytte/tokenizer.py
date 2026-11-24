"""Train and use Pytte's byte-level BPE tokenizer.

A small vocabulary (default 4096) keeps the embedding table from eating the
parameter budget of a tiny model. Byte-level BPE means every character
(å, ä, ö, emoji) is always representable.

    python -m pytte.tokenizer train --input data.txt --vocab 4096 --out tok/
"""
import argparse
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOT = "<|endoftext|>"


def train(files, vocab_size, out_dir):
    tok = Tokenizer(models.BPE(byte_fallback=False))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    from .prepare_data import iter_docs   # local import: prepare_data imports this module
    # Accepts plain text (documents separated by <|endoftext|> lines) and jsonl with a "text" field.
    tok.train_from_iterator((doc for f in files for doc in iter_docs(f)), trainer)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_dir / "tokenizer.json"))
    return tok


def load(path):
    path = Path(path)
    if path.is_dir():
        path = path / "tokenizer.json"
    return Tokenizer.from_file(str(path))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--input", nargs="+", required=True)
    t.add_argument("--vocab", type=int, default=4096)
    t.add_argument("--out", required=True)
    s = sub.add_parser("show")
    s.add_argument("--tok", required=True)
    s.add_argument("text", nargs="+")
    args = p.parse_args()

    if args.cmd == "train":
        tok = train(args.input, args.vocab, args.out)
        print(f"vocab={tok.get_vocab_size()} saved to {args.out}")
    else:
        tok = load(args.tok)
        for text in args.text:
            enc = tok.encode(text)
            print(f"{len(enc.ids):3d} tokens | {text}")
            print("    ", [tok.decode([i]) for i in enc.ids])


if __name__ == "__main__":
    main()
