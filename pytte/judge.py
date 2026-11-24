"""LLM judge for generated stories (TinyStories-style grading), run locally with mlx-lm.

Grades grammar, coherence and creativity 1-10 and counts language errors, using a fixed
prompt so results are reproducible. Needs the GPU, so run it when nothing else does.

    python -m pytte.judge --samples ckpt/pytte-1m-sv/eval.json --lang sv
    python -m pytte.judge --texts stories.jsonl --lang sv --n 100
"""
import argparse
import json
import re
from pathlib import Path

from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

JUDGE_MODEL = "mlx-community/gemma-4-12B-it-qat-4bit"

PROMPT = {
    "sv": (
        "Du är en noggrann svensklärare. Nedan är en kort saga skriven av en mycket liten AI-modell. "
        "Bedöm den på tre skalor från 1 till 10:\n"
        "- grammatik: stavning, böjning, kongruens (en/ett, adjektiv, verbformer), ordföljd\n"
        "- sammanhang: håller sagan ihop? Samma figurer, samma plats, logisk följd?\n"
        "- kreativitet: finns det en idé, en händelse, något oväntat?\n"
        "Räkna också antalet tydliga språkfel (ord som inte finns, fel böjning, fel ordföljd).\n"
        "Svara ENDAST med JSON på formen "
        '{"grammatik": N, "sammanhang": N, "kreativitet": N, "sprakfel": N, "kommentar": "en kort mening"}.\n\n'
        "SAGA:\n{story}"
    ),
    "en": (
        "You are a careful English teacher. Below is a short story written by a very small AI model. "
        "Grade it on three scales from 1 to 10:\n"
        "- grammar: spelling, inflection, agreement, word order\n"
        "- coherence: does the story hold together? Same characters, same place, logical sequence?\n"
        "- creativity: is there an idea, an event, something unexpected?\n"
        "Also count the number of clear language errors (non-words, wrong inflection, wrong word order).\n"
        "Answer ONLY with JSON of the form "
        '{"grammar": N, "coherence": N, "creativity": N, "errors": N, "comment": "one short sentence"}.\n\n'
        "STORY:\n{story}"
    ),
}
KEYS = {"sv": ["grammatik", "sammanhang", "kreativitet", "sprakfel"],
        "en": ["grammar", "coherence", "creativity", "errors"]}


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def judge(texts, lang, model_name=JUDGE_MODEL, batch=16, loaded=None):
    """loaded=(model, tokenizer) lets a caller judge several sets with one model load."""
    model, tokenizer = loaded if loaded else load(model_name)
    sampler = make_sampler(temp=0.0)
    results = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT[lang].replace("{story}", t)}],
            add_generation_prompt=True, enable_thinking=False) for t in chunk]
        resp = batch_generate(model, tokenizer, prompts, max_tokens=160, sampler=sampler)
        for t, out in zip(chunk, resp.texts):
            results.append({"text": t, "judge": parse_json(out), "raw": out.strip()})
        print(f"judged {min(i + batch, len(texts))}/{len(texts)}", flush=True)
    return results


def summarize(results, lang):
    ok = [r["judge"] for r in results if r["judge"]]
    summary = {"n": len(results), "parsed": len(ok)}
    for k in KEYS[lang]:
        vals = [float(j[k]) for j in ok if isinstance(j.get(k), (int, float))]
        if vals:
            summary[k] = sum(vals) / len(vals)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", help="eval.json from pytte.eval (uses its 'samples')")
    p.add_argument("--texts", help="jsonl with a 'text' field, or a txt with <|endoftext|> separators")
    p.add_argument("--lang", choices=["sv", "en"], required=True)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--judge-model", default=JUDGE_MODEL)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.samples:
        texts = json.loads(Path(args.samples).read_text())["samples"]
        default_out = Path(args.samples).with_name("judge.json")
    else:
        from .prepare_data import iter_docs
        texts = []
        for d in iter_docs(args.texts):
            texts.append(d)
            if len(texts) >= args.n:
                break
        default_out = Path(args.texts).with_suffix(".judge.json")
    texts = texts[:args.n]
    results = judge(texts, args.lang, args.judge_model)
    summary = summarize(results, args.lang)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    out = Path(args.out) if args.out else default_out
    out.write_text(json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
