"""Not used: Pytte is published on GitHub and on mathiaskarlsson.se/pytte instead. Kept as a record.

Build the Hugging Face upload folders: one per model with the safetensors checkpoint,
config, tokenizer, the int8 .bin and a model card generated from the measurements.

    python scripts/make_model_cards.py --out /Volumes/Modelltest/pytte/hf

Reads <ckpt>/{summary,eval,judge,eval_pairs}.json for pytte-<size>-sv3 and -sv3-chat.
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path("/Volumes/Modelltest/pytte")
SIZES = ["0.3m", "1m", "2m", "3m", "7m", "15m"]
ARCH = {"0.3m": (64, 4, 4, 2, 192, 2048), "1m": (128, 4, 4, 4, 352, 4096), "2m": (160, 5, 5, 5, 448, 4096),
        "3m": (192, 6, 6, 6, 512, 4096), "7m": (288, 6, 6, 6, 768, 4096), "15m": (384, 8, 6, 6, 1024, 4096)}
SPEED = {"0.3m": 38000, "1m": 15500, "2m": 10100, "3m": 6800, "7m": 3900, "15m": 1900}


def load(p):
    return json.loads(p.read_text()) if p.exists() else {}


def card(size, chat, src, bin_mb):
    name = f"pytte-{size}{'-chat' if chat else ''}"
    dim, L, H, KV, FF, V = ARCH[size]
    sm, ev, jd, pr = (load(src / f) for f in ("summary.json", "eval.json", "judge.json", "eval_pairs.json"))
    js = jd.get("summary", {})
    pairs = (pr.get("minimal_pairs:minimal_pairs_sv") or ev.get("minimal_pairs:minimal_pairs_sv") or ev.get("minimal_pairs") or {})
    acc = pairs.get("accuracy")
    params = sm.get("params", 0)
    minutes = sm.get("train_minutes")
    lines = [
        "---", "language:", "- sv", "license: apache-2.0", "library_name: mlx", "pipeline_tag: text-generation",
        "tags:", "- pytte", "- tiny", "- swedish", "- from-scratch", "- tinystories", "- mlx",
        "datasets:", "- SagoSvenska", "---", "",
        f"# {name}", "",
        f"{'An instruction-tuned' if chat else 'A'} Swedish language model with **{params/1e6:.2f} million parameters** "
        f"({sm.get('params_no_embed', 0)/1e6:.2f}M without the embedding table), written and trained from scratch on a "
        f"MacBook Pro (M2 Pro) as part of [Pytte](https://github.com/dragon6sic6/pytte), a family of small models that write "
        f"Swedish children's stories. It is not a fine-tune of an existing model.", "",
        "## Architecture", "",
        f"Llama-style decoder: RMSNorm, rotary positions, {'grouped-query ' if KV < H else ''}attention, SwiGLU, tied "
        f"embeddings. dim {dim}, {L} layers, {H} heads ({KV} kv heads), feed-forward {FF}, vocabulary {V} "
        f"(byte-level BPE trained on the stories), context 512.", "",
        "## Training", "",
    ]
    if chat:
        lines += [
            f"Fine-tuned from `pytte-{size}` on request → story conversations built from the recipes of the "
            f"training stories (name, object, adjective, verb, place, plural, tense, first person, features, ending), "
            f"with half-story follow-ups, new episodes about the same child and a little small talk. Plain-text format, "
            f"loss on the replies only" + (f", {minutes:.0f} minutes" if minutes else "") + ".", "",
            "```", "<|endoftext|>Du: Berätta en saga om en älg.", "Pytte: <story><|endoftext|>", "```", "",
        ]
    else:
        lines += [
            f"Trained from random initialisation on SagoSvenska (85 764 Swedish children's stories, 12M tokens, "
            f"generated locally with Gemma 4 12B under Apache-2.0)" + (f" for {minutes:.0f} minutes" if minutes else "") +
            f" with AdamW and a cosine schedule; the best-validation checkpoint is kept.", "",
        ]
    lines += ["## Results", "", "| Measure | Value |", "|---|---:|"]
    if sm.get("best_val_bpb"):
        lines.append(f"| Bits per byte on held-out stories | {sm['best_val_bpb']:.3f} |")
    if acc is not None:
        lines.append(f"| Swedish minimal-pair grammar test (200 pairs) | {100*acc:.1f}% |")
    if js:
        lines.append(f"| Judge (Gemma 4 12B) grammar / coherence, 1–10 | {js['grammatik']:.1f} / {js['sammanhang']:.1f} |")
        lines.append(f"| Language errors per story (judge) | {js['sprakfel']:.2f} |")
    inf = ev.get("instruction_following")
    if chat and inf:
        per = inf["per_ingredient"]
        lines.append(f"| Requested ingredients used in the story | {100*inf['ingredient_hit_rate']:.0f}% "
                     f"(name {100*per.get('name',0):.0f}%, object {100*per.get('noun',0):.0f}%, adjective {100*per.get('adj',0):.0f}%) |")
        lines.append(f"| Follow-up keeps the same child | {100*ev['follow_ups']['keeps_name']:.0f}% |")
    elif ev.get("ended_naturally") is not None:
        lines.append(f"| Stories that end on their own | {100*ev['ended_naturally']:.0f}% |")
    lines += [
        f"| int8 export | {bin_mb:.2f} MB, {SPEED[size]:,} tokens/s on one M2 Pro core (run.c) |".replace(",0", " 0").replace(",5", " 5").replace(",9", " 9").replace(",1", " 1").replace(",8", " 8"), "",
        "The report, [Pytte: Tiny Language Models That Write Coherent Swedish, Trained from Scratch on a Laptop]"
        "(https://github.com/dragon6sic6/pytte/blob/main/paper/pytte.pdf), describes the data, the training and every "
        "measurement. The minimal-pair test and the judge prompts are in the repository; the lab log "
        "(`FÖRSLAG-OCH-PLAN.md`, Swedish) records every run, including the ones that failed.", "",
        "## Files", "",
        "- `model.safetensors`, `config.json`: the MLX checkpoint (`pytte.model.Pytte.load`)",
        "- `tokenizer.json`: the byte-level BPE tokenizer (Hugging Face `tokenizers`)",
        f"- `{name}.bin`: model and tokenizer in one int8 file for `run.c` and the browser demo", "",
        "## Use", "", "```bash",
        "git clone https://github.com/dragon6sic6/pytte && cd pytte",
        "cc -O3 -o run run.c -lm",
        f"./run {name}.bin" + (" -chat" if chat else ' -p "Det var en gång"'),
        "```", "",
        "```python",
        "from pytte.model import Pytte",
        "from pytte.tokenizer import load as load_tok",
        "from pytte." + ("chat import reply" if chat else "sample import generate"), "",
        f"model, tok = Pytte.load('.'), load_tok('tokenizer.json')",
        ("print(reply(model, tok, [{'role': 'user', 'content': 'Berätta en saga om en älg.'}]))" if chat
         else "print(generate(model, tok, 'Det var en gång', max_new=200))"),
        "```", "",
        "## Limitations", "",
        "Children's-story Swedish only: a few thousand words, short sentences, no facts about the world. "
        + ("It follows the name and the place in a request far more reliably than an object or an adjective, and "
           "small talk is a handful of fixed replies. " if chat else "")
        + ("At this size the model writes Swedish sentences rather than coherent stories. " if size == "0.3m" else "")
        + "The training data is synthetic (generated by Gemma 4 12B) and the judge is the same model family, which "
        "the report discusses.", "",
        "## License", "", "Apache-2.0 (code, models and weights). SagoSvenska is released under CC BY 4.0.", "",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "hf"))
    args = p.parse_args()
    out = Path(args.out)
    for size in SIZES:
        for chat in (False, True):
            name = f"pytte-{size}{'-chat' if chat else ''}"
            src = ROOT / "checkpoints" / f"pytte-{size}-sv3{'-chat' if chat else ''}"
            d = out / name
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy(src / "model.safetensors", d / "model.safetensors")
            shutil.copy(src / "config.json", d / "config.json")
            shutil.copy(ROOT / "data" / f"tok-sv3-{ARCH[size][5]}" / "tokenizer.json", d / "tokenizer.json")
            binf = ROOT / "bin" / f"{name}.bin"
            shutil.copy(binf, d / f"{name}.bin")
            (d / "README.md").write_text(card(size, chat, src, binf.stat().st_size / 1e6))
            print(f"{d}: {sum(f.stat().st_size for f in d.iterdir())/1e6:.1f} MB")


if __name__ == "__main__":
    main()
