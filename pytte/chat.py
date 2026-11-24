"""Chat with a Pytte model fine-tuned by pytte.train on pytte.prepare_sft data.

The format is plain text, so it also works in run.c and the web demo:

    <|endoftext|>Du: Berätta en saga om en älg.
    Pytte: <story><|endoftext|>

Every assistant turn ends with <|endoftext|>; a follow-up ("Vad händer sedan?") is appended
after it and the model continues.

    python -m pytte.chat --model ckpt/pytte-3m-sv3-chat --tok data/tok-sv3-4096 --prompt "Berätta en saga om en räv."
    python -m pytte.chat --model ... --tok ...                 # interactive
    python -m pytte.chat --model ... --tok ... --requests requests.json --out replies.json
"""
import argparse
import json
import sys

import mlx.core as mx

from .model import Pytte
from .prepare_sft import prompt_ids, USER
from .sample import generate_ids
from .tokenizer import load as load_tok

STOP = f"\n{USER}"   # safety net: a reply that drifts into a new user turn is cut there


def reply(model, tok, messages, max_new=300, temp=0.8, top_p=0.95, stream=None):
    """messages: [{"role": "user"|"assistant", "content": ...}], ending with a user turn."""
    text = generate_ids(model, tok, prompt_ids(tok, messages), max_new, temp, 0, top_p, stream)
    if STOP in text:
        text = text.split(STOP)[0]
    return text.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tok", required=True)
    p.add_argument("--prompt", default=None, help="one request; omit for an interactive session")
    p.add_argument("--requests", default=None, help="JSON list of requests (strings) to answer in batch")
    p.add_argument("--out", default=None)
    p.add_argument("--max-new", type=int, default=300)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    if args.seed is not None:
        mx.random.seed(args.seed)
    model = Pytte.load(args.model)
    tok = load_tok(args.tok)

    if args.requests:
        reqs = json.loads(open(args.requests).read())
        out = [{"request": q, "reply": reply(model, tok, [{"role": "user", "content": q}], args.max_new, args.temp, args.top_p)}
               for q in reqs]
        if args.out:
            open(args.out, "w").write(json.dumps(out, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(out, indent=2, ensure_ascii=False))
        return
    if args.prompt:
        print(reply(model, tok, [{"role": "user", "content": args.prompt}], args.max_new, args.temp, args.top_p,
                    stream=lambda s: (sys.stdout.write(s), sys.stdout.flush())))
        return
    history = []
    print("Pytte: Hej! Skriv något (tom rad avslutar, /ny börjar om).")
    while True:
        try:
            q = input("Du: ").strip()
        except EOFError:
            break
        if not q:
            break
        if q == "/ny":
            history = []
            continue
        history.append({"role": "user", "content": q})
        sys.stdout.write("Pytte: ")
        a = reply(model, tok, history, args.max_new, args.temp, args.top_p,
                  stream=lambda s: (sys.stdout.write(s), sys.stdout.flush()))
        print()
        history.append({"role": "assistant", "content": a})


if __name__ == "__main__":
    main()
