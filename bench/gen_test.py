"""Test local Swedish story generation with an open model via mlx-lm.

Measures throughput at several batch sizes and saves the stories so a human
can judge the Swedish. Usage:

    python gen_test.py mlx-community/Qwen3-4B-4bit --batches 1 16 32
"""
import argparse
import dataclasses
import random
import time
from pathlib import Path

from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

NOUNS = ["älg", "båt", "kanelbulle", "skog", "sjö", "traktor", "katt", "snögubbe",
         "cykel", "midsommarstång", "räv", "stuga", "regnjacka", "igelkott", "fika",
         "lingon", "drake", "brygga", "skridskor", "mormor"]
VERBS = ["hoppa", "leta", "bygga", "simma", "dela", "gömma", "skratta", "hjälpa",
         "tappa", "vänta", "klättra", "baka", "springa", "sjunga", "låna"]
ADJS = ["röd", "blyg", "modig", "liten", "glad", "sömnig", "nyfiken", "kall",
        "snäll", "busig", "ledsen", "varm", "stor", "hungrig", "tyst"]
FEATURES = ["en dialog mellan två figurer", "ett lite sorgligt men tryggt slut där ingen skadas",
            "en oväntad men ofarlig vändning", "en enkel sensmoral på slutet", "ett lyckligt slut",
            "en liten konflikt som löses på ett snällt sätt"]
NAMES = ["Elsa", "Noah", "Alma", "Hugo", "Maja", "Liam", "Vera", "Oskar", "Selma", "Elias",
         "Astrid", "Viggo", "Ebba", "Sixten", "Saga", "Melvin", "Tuva", "Arvid", "Lo", "Ivar",
         "Yusuf", "Amira", "Nour", "Leon", "Ida", "Frans", "Juni", "Otto", "Stella", "Malte"]
PLACES = ["i skärgården", "på landet hos mormor", "i en liten by i Norrland", "på en förskola",
          "vid en skog med höga tallar", "i en lägenhet i stan", "på en bondgård", "vid en sjö",
          "i ett rött hus med vita knutar", "på ett tåg", "i en snöig park", "på ett bibliotek"]

SYSTEM = (
    "Du skriver korta sagor på svenska för barn som är 3 till 4 år gamla. "
    "Använd bara enkla, vanliga ord som ett litet barn förstår. Korta meningar. "
    "Skriv bara själva sagan, ingen rubrik och inga kommentarer."
)


def make_prompt(rng):
    n, v, a, f = rng.choice(NOUNS), rng.choice(VERBS), rng.choice(ADJS), rng.choice(FEATURES)
    name, place = rng.choice(NAMES), rng.choice(PLACES)
    return (
        f"Skriv en saga på 100 till 150 ord. Huvudpersonen heter {name} och sagan utspelar sig {place}. "
        f"Sagan ska innehålla ordet \"{n}\", verbet \"{v}\" och adjektivet \"{a}\", och den ska ha {f}."
    )


def encode(tokenizer, user, model_name):
    msgs = [{"role": "user", "content": SYSTEM + "\n\n" + user}]
    kwargs = {}
    if "qwen3" in model_name.lower() or "gemma-4" in model_name.lower():
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, **kwargs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--batches", type=int, nargs="+", default=[1, 16])
    p.add_argument("--max-tokens", type=int, default=320)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--tag", default="", help="suffix for the output file name")
    p.add_argument("--out", default="/Volumes/Modelltest/pytte/data/gen-test")
    args = p.parse_args()

    model, tokenizer = load(args.model)
    sampler = make_sampler(temp=args.temp, top_p=0.95)
    rng = random.Random(0)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.model.split("/")[-1] + args.tag
    out_file = out_dir / f"{tag}.txt"
    stories = []

    for bs in args.batches:
        prompts = [encode(tokenizer, make_prompt(rng), args.model) for _ in range(bs)]
        t0 = time.perf_counter()
        resp = batch_generate(model, tokenizer, prompts, max_tokens=args.max_tokens,
                              sampler=sampler, return_token_ids=True)
        dt = time.perf_counter() - t0
        n_out = sum(len(t) for t in resp.token_ids)
        print(f"{tag} batch={bs:>3} | {n_out:>6} output tokens in {dt:6.1f}s | "
              f"{n_out/dt:7.1f} tok/s | {bs/dt*3600:6.0f} stories/hour")
        stats = {k: v for k, v in dataclasses.asdict(resp.stats).items()
                 if isinstance(v, (int, float))}
        print("   stats:", stats)
        stories.extend(resp.texts)

    with out_file.open("w") as f:
        for i, s in enumerate(stories, 1):
            f.write(f"===== Saga {i}\n{s.strip()}\n\n")
    print(f"Sparade {len(stories)} sagor till {out_file}")
    print("\n--- Exempel (saga 1) ---\n" + stories[0].strip()[:800])


if __name__ == "__main__":
    main()
