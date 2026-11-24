"""SagoSvenska: generate Swedish children's stories locally with an open model.

Resumable, batched generation with mlx-lm. Each story is written as one JSON
line with its prompt ingredients so the dataset card can describe exactly how
it was made. Stories that trip the child-safety word filter go to a separate
file for manual review instead of the main set.

    python gen_stories.py --target 60000

Stop gracefully by creating the file <out>/STOP; pause with <out>/PAUSE (removed = resume).
"""
import argparse
import copy
import datetime as dt
import json
import random
import re
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import batch_generate, load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

MODEL = "mlx-community/gemma-4-12B-it-qat-4bit"

NAMES = [
    "Elsa", "Noah", "Alma", "Hugo", "Maja", "Liam", "Vera", "Oskar", "Selma", "Elias",
    "Astrid", "Viggo", "Ebba", "Sixten", "Saga", "Melvin", "Tuva", "Arvid", "Lo", "Ivar",
    "Yusuf", "Amira", "Nour", "Leon", "Ida", "Frans", "Juni", "Otto", "Stella", "Malte",
    "Lilly", "Adam", "Wilma", "Vincent", "Agnes", "Ali", "Olivia", "Nils", "Freja", "Ture",
    "Lova", "Alfred", "Signe", "Charlie", "Ellen", "Omar", "Edith", "Harry", "Linnea", "Isak",
    "Sara", "Kalle", "Emma", "Erik", "Lisa", "Anton", "Klara", "Felix", "Hanna", "Axel",
]
PLACES = [
    "i skärgården", "på landet hos mormor", "i en liten by i Norrland", "på en förskola",
    "i en skog med höga tallar", "i en lägenhet i stan", "på en bondgård", "vid en sjö",
    "i ett rött hus med vita knutar", "på ett tåg", "i en snöig park", "på ett bibliotek",
    "på en lekplats", "vid havet", "i ett sommarhus", "på en båt", "i en trädgård",
    "på en äng full av blommor", "i ett växthus", "på ett torg", "i en stuga i fjällen",
    "på en strand", "i en tunnelbana", "på ett museum", "i ett badhus", "på en marknad",
    "hos farfar", "i ett koloniområde", "på en gård med hästar", "i en liten stad",
]
NOUNS = [
    "älg", "båt", "kanelbulle", "skog", "sjö", "traktor", "katt", "snögubbe", "cykel",
    "midsommarstång", "räv", "stuga", "regnjacka", "igelkott", "fika", "lingon", "drake",
    "brygga", "skridskor", "mormor", "hund", "boll", "ballong", "kaka", "äpple", "nalle",
    "bil", "buss", "tåg", "flygplan", "kanin", "fågel", "fisk", "groda", "ekorre", "björn",
    "varg", "ko", "häst", "gris", "får", "höna", "anka", "mus", "nyckelpiga", "fjäril", "bi",
    "mask", "blomma", "träd", "sten", "pinne", "spade", "hink", "sandlåda", "gunga",
    "rutschkana", "trampolin", "pussel", "bok", "penna", "papper", "låda", "nyckel", "lampa",
    "stol", "säng", "kudde", "filt", "mössa", "vante", "stövel", "paraply", "ryggsäck",
    "smörgås", "pannkaka", "glass", "saft", "mjölk", "ost", "bulle", "soppa", "köttbulle",
    "potatis", "morot", "jordgubbe", "blåbär", "svamp", "kotte", "löv", "snö", "regn", "is",
    "moln", "sol", "måne", "stjärna", "vind", "pöl", "bro", "väg", "staket", "dörr",
    "fönster", "trappa", "tak", "koja", "tält", "eld", "lykta", "klocka", "spegel", "tavla",
    "trumma", "gitarr", "sång", "dans", "kalas", "present", "tårta", "ljus", "brev", "karta",
]
VERBS = [
    "hoppa", "leta", "bygga", "simma", "dela", "gömma", "skratta", "hjälpa", "tappa",
    "vänta", "klättra", "baka", "springa", "sjunga", "låna", "rita", "måla", "läsa",
    "sova", "vakna", "äta", "dricka", "plocka", "kasta", "fånga", "cykla", "åka", "flyga",
    "gräva", "plantera", "vattna", "tvätta", "städa", "laga", "sy", "sticka", "fiska",
    "paddla", "ro", "glida", "halka", "ramla", "resa sig", "trösta", "krama", "viska",
    "ropa", "lyssna", "titta", "smyga", "räkna", "mäta", "väga", "packa", "öppna",
    "stänga", "låsa", "knacka", "ringa", "skriva", "drömma", "önska", "fira", "bjuda",
]
ADJS = [
    "röd", "blyg", "modig", "liten", "glad", "sömnig", "nyfiken", "kall", "snäll", "busig",
    "ledsen", "varm", "stor", "hungrig", "tyst", "blå", "gul", "grön", "vit", "svart",
    "brun", "rosa", "mjuk", "hård", "blöt", "torr", "snabb", "långsam", "hög", "låg",
    "tung", "lätt", "ny", "gammal", "ren", "smutsig", "söt", "sur", "salt", "arg", "rädd",
    "trött", "pigg", "stolt", "ensam", "lycklig", "orolig", "tapper", "rolig", "klok",
    "envis", "försiktig", "noggrann", "slarvig", "generös", "tålmodig", "tjock", "smal",
    "randig", "prickig", "glittrig", "mysig", "läskig", "konstig", "vacker", "fin",
]
FEATURES = [
    "en dialog mellan två figurer",
    "ett lite sorgligt men tryggt slut där ingen skadas",
    "en oväntad men ofarlig vändning",
    "en enkel sensmoral på slutet",
    "ett lyckligt slut",
    "en liten konflikt som löses på ett snällt sätt",
    "ett missförstånd som reds ut",
    "en figur som lär sig något nytt",
    "en figur som är rädd men blir modig",
    "ett löfte som hålls",
    "något som försvinner och hittas igen",
    "två vänner som turas om",
]

SYSTEM = (
    "Du skriver korta sagor på svenska för barn som är 3 till 4 år gamla. "
    "Använd bara enkla, vanliga svenska ord som ett litet barn förstår. Korta meningar. "
    "Sagorna ska vara trygga och snälla: ingen dör, ingen skadas allvarligt, inget skrämmande. "
    "Skriv bara själva sagan på svenska, utan rubrik, utan kommentarer, utan engelska ord."
)

# Words that send a story to review instead of the main set.
UNSAFE = re.compile(
    r"\b(dö|dör|död|döda|dog|drunkna\w*|blod\w*|skadad\w*|skjut\w*|mörda\w*|kniv\w*|"
    r"vapen|pistol\w*|krig\w*|slår|slog|misshandl\w*|självmord|försvann för alltid|"
    r"aldrig mer|begravn\w*|kista|sjukhus\w*|ambulans\w*)\b",
    re.IGNORECASE,
)


def make_prompt(rng):
    ing = {
        "name": rng.choice(NAMES),
        "place": rng.choice(PLACES),
        "noun": rng.choice(NOUNS),
        "verb": rng.choice(VERBS),
        "adj": rng.choice(ADJS),
        "feature": rng.choice(FEATURES),
        "words": rng.choice([80, 100, 120, 150, 180]),
    }
    text = (
        f"Skriv en saga på ungefär {ing['words']} ord. Huvudpersonen heter {ing['name']} "
        f"och sagan utspelar sig {ing['place']}. Sagan ska innehålla ordet \"{ing['noun']}\", "
        f"verbet \"{ing['verb']}\" och adjektivet \"{ing['adj']}\", och den ska ha {ing['feature']}."
    )
    return ing, text


# ---------------------------------------------------------------------------------------
# Recipe v2 (2026-11-10). The v1 pilot showed that temperature alone does not break the
# template (80 % of stories still opened "Namn bor/är ..."): the prompt decides the shape.
# v2 randomises what v1 held fixed: opening type, tense, number of characters, plural
# objects, perspective, ending, sentence style, one or two features, and a wider length
# range. It also bans the three most common v1 sentences outright.
# ---------------------------------------------------------------------------------------
OPENINGS = [
    ("dialog", "med en replik, alltså att någon säger något"),
    ("ljud", "med ett ljud, till exempel \"{s1}\" eller \"{s2}\""),
    ("fråga", "med en fråga"),
    ("väder", "med vädret eller årstiden"),
    ("handling", "mitt i en handling, där något redan händer"),
    ("känsla", "med hur huvudpersonen känner sig"),
    ("tid", "med tiden på dygnet, till exempel morgon, kväll eller natt"),
    ("klassisk", "med \"Det var en gång\""),
    ("räkning", "med att räkna något, till exempel \"{count} {things} ...\""),
    ("sak", "med en sak eller ett djur, inte med huvudpersonen"),
    ("plats", "med en kort beskrivning av platsen"),
    ("rop", "med att någon ropar på huvudpersonen"),
    ("inte", "med något som huvudpersonen inte vill eller inte kan"),
    ("besök", "med att någon kommer på besök"),
]
TENSES = [("presens", "i presens, alltså nutid"), ("preteritum", "i preteritum, alltså dåtid med ord som gick, sa och var")]
CASTS = [  # (key, weight, template)
    ("en", 45, "Huvudpersonen heter {n1}."),
    ("två", 35, "Huvudpersonerna är {n1} och {n2}, som gör saker tillsammans."),
    ("tre", 10, "Sagan handlar om tre barn: {n1}, {n2} och {n3}."),
    ("vuxen", 10, "Huvudpersonen heter {n1} och är tillsammans med en vuxen, till exempel mamma, pappa, mormor eller farfar."),
]
PLURALS = [
    "katter", "hundar", "bollar", "ankor", "kaniner", "fåglar", "fiskar", "grodor", "ekorrar", "kor",
    "hästar", "grisar", "får", "höns", "möss", "fjärilar", "blommor", "träd", "stenar", "pinnar",
    "äpplen", "kakor", "bullar", "pannkakor", "jordgubbar", "blåbär", "svampar", "kottar", "löv", "moln",
    "stjärnor", "stövlar", "vantar", "böcker", "pennor", "lådor", "ballonger", "skor", "strumpor", "kuddar",
    "vänner", "barn", "ankungar", "kycklingar", "båtar", "bilar", "tåg", "cyklar", "snögubbar", "fönster",
    "dörrar", "ljus", "presenter", "hinkar", "spadar", "nycklar", "mössor", "päron", "bananer", "ägg",
]
ENDINGS = [
    "ett lyckligt slut", "ett lugnt slut där alla somnar", "ett roligt slut", "ett slut med en fråga till den som lyssnar",
    "ett slut där en mening från början kommer tillbaka", "ett slut med en liten överraskning",
    "ett slut där figurerna säger god natt", "ett slut där de delar med sig", "ett slut som är lite sorgligt men tryggt",
    "ett slut där huvudpersonen berättar för någon vad som hänt", "ett slut där de planerar vad de ska göra i morgon",
]
FEATURES_V2 = [f for f in FEATURES if "slut" not in f] + [   # endings are drawn from ENDINGS instead
    "ett djur som pratar", "ett räkneord, som två, tre eller fyra", "en sång eller ramsa som upprepas",
    "något som är för stort eller för litet", "en figur som hjälper en annan", "något som går sönder och lagas",
    "en överraskning i en låda eller ett paket", "ett väder som ändras mitt i sagan", "en saga som utspelar sig på natten",
    "någon som väntar på något", "en resa fram och tillbaka", "något som smakar gott", "en figur som gömmer sig",
    "en lek med regler", "en vuxen som tröstar", "en figur som är envis och ändrar sig", "något som tappas bort",
    "ett ljud som ingen först förstår", "en present", "en figur som säger förlåt",
]
SOUNDS = ["Plask!", "Dunk!", "Pip pip!", "Knack knack!", "Vroom!", "Bom!", "Mjau!", "Voff!", "Plopp!", "Ding dong!",
          "Kling!", "Tut tut!", "Hopp!", "Krasch!", "Sch!", "Hoppsan!", "Brum brum!", "Tick tack!", "Svisch!", "Kvack!"]
COUNTS = ["Två", "Tre", "Fyra", "Fem"]
STYLES = [(70, "Korta meningar."), (30, "Blanda korta och lite längre meningar.")]
WORDS_V2 = [(60, 1), (80, 2), (100, 3), (120, 3), (150, 3), (180, 2), (220, 1)]
BAN = ("Börja inte med huvudpersonens namn följt av bor, är eller går, och använd inte meningarna "
       "\"Det är en fin dag\", \"Det är en solig dag\" eller \"Det är vinter i fjällen\".")


def _weighted(rng, items):
    return rng.choices([x[0] if len(x) == 2 else x for x in items], weights=[x[1] if len(x) == 2 else 1 for x in items])[0]


def make_prompt_v2(rng):
    names = rng.sample(NAMES, 3)
    cast = rng.choices(CASTS, weights=[c[1] for c in CASTS])[0]
    opening = rng.choice(OPENINGS)
    s1, s2 = rng.sample(SOUNDS, 2)
    opening = (opening[0], opening[1].format(s1=s1, s2=s2, count=rng.choice(COUNTS), things=rng.choice(PLURALS)))
    tense = rng.choice(TENSES)
    n_feat = 1 if rng.random() < 0.6 else 2
    features = rng.sample(FEATURES_V2, n_feat)
    plural = rng.choice(PLURALS) if rng.random() < 0.4 else None
    first_person = cast[0] == "en" and rng.random() < 0.12
    style = rng.choices([x[1] for x in STYLES], weights=[x[0] for x in STYLES])[0]
    words = rng.choices([w for w, _ in WORDS_V2], weights=[k for _, k in WORDS_V2])[0]
    ending = rng.choice(ENDINGS)
    ing = {
        "name": names[0], "names": names[:{"en": 1, "två": 2, "tre": 3, "vuxen": 1}[cast[0]]], "cast": cast[0],
        "place": rng.choice(PLACES), "noun": rng.choice(NOUNS), "verb": rng.choice(VERBS), "adj": rng.choice(ADJS),
        "plural": plural, "opening": opening[0], "tense": tense[0], "first_person": first_person,
        "features": features, "feature": features[0], "ending": ending, "style": style, "words": words, "recipe": "v2",
    }
    text = (
        f"Skriv en saga på ungefär {words} ord, {tense[1]}. "
        + cast[2].format(n1=names[0], n2=names[1], n3=names[2]) + " "
        + (f"Låt {names[0]} själv berätta sagan i jag-form. " if first_person else "")
        + f"Sagan utspelar sig {ing['place']}. Börja sagan {opening[1]}. "
        + f"Sagan ska innehålla ordet \"{ing['noun']}\", verbet \"{ing['verb']}\" och adjektivet \"{ing['adj']}\""
        + (f", och flera {plural} ska vara med, i pluralis, så att böjningen syns (som \"{plural} är\" och \"de\")" if plural else "")
        + ". Den ska ha " + " och ".join(features) + f", och {ending}. {style} {BAN}"
    )
    return ing, text


RECIPES = {"v1": make_prompt, "v2": make_prompt_v2}


def encode(tokenizer, user):
    msgs = [{"role": "user", "content": SYSTEM + "\n\n" + user}]
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)


def common_prefix(id_lists):
    """Longest shared token prefix across prompts, minus one token for BPE-boundary safety."""
    first = id_lists[0]
    n = len(first)
    for ids in id_lists[1:]:
        k = 0
        while k < min(n, len(ids)) and ids[k] == first[k]:
            k += 1
        n = k
    return first[:max(0, n - 1)]


def build_prefix_cache(model, prefix_ids):
    """Run the shared prompt prefix once and keep its KV cache."""
    cache = make_prompt_cache(model)
    model(mx.array(prefix_ids)[None], cache=cache)
    mx.eval([c.state for c in cache])
    return cache


def split_with_cache(prompts, prefix_ids, prefix_cache):
    """Per prompt: (remaining tokens, cache copy) or (full prompt, None) if it does not share the prefix."""
    n = len(prefix_ids)
    rest, caches = [], []
    for ids in prompts:
        if n and ids[:n] == prefix_ids and len(ids) > n:
            rest.append(ids[n:])
            caches.append(copy.deepcopy(prefix_cache))
        else:
            rest.append(ids)
            caches.append(None)
    return rest, caches


SPECIAL_TOKEN = re.compile(r"<\|?[A-Za-z_\"]{0,20}\|?>")


def clean(text):
    """Strip leaked special tokens (see ban_special_tokens) and tidy whitespace.
    Returns (text, n_special) so the caller can send affected stories to review."""
    n = 0

    def sub(m):
        nonlocal n
        n += 1
        before = text[m.start() - 1] if m.start() > 0 else " "
        after = text[m.end()] if m.end() < len(text) else " "
        return "" if before.isalpha() and after.isalpha() else " "

    text = SPECIAL_TOKEN.sub(sub, text).strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text, n


def ban_special_tokens(tokenizer):
    """Logits processor that forbids every special token except end-of-turn/EOS.

    The 4-bit Gemma 4 models emit <image|> (end-of-image) mid-word in ~12 % of stories at
    temp 0.6 and ~20 % at 0.85. Setting those logits to -inf removes the glitch at the source."""
    keep = set(tokenizer.eos_token_ids)
    for t in ("<eos>", "<turn|>", "<end_of_turn>"):   # end-of-turn must stay allowed or stories never stop
        tid = tokenizer.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid >= 0 and tid != getattr(tokenizer, "unk_token_id", None):
            keep.add(tid)
    ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    added = getattr(tokenizer, "added_tokens_decoder", {}) or {}
    ids.update(added.keys())
    banned = sorted(i for i in ids if i not in keep)
    vocab = max(banned) + 1 if banned else 0
    mask = mx.zeros((vocab,), dtype=mx.float32)
    if banned:
        mask[mx.array(banned)] = -float("inf")

    def processor(tokens, logits):
        v = logits.shape[-1]
        m = mask[:v] if v <= vocab else mx.concatenate([mask, mx.zeros((v - vocab,), dtype=mx.float32)])
        return logits + m.astype(logits.dtype)

    print(f"Banned {len(banned)} special tokens during sampling (kept EOS ids {sorted(keep)})", flush=True)
    return processor


def count_existing(out_dir):
    n = 0
    for f in out_dir.glob("stories-*.jsonl"):
        with f.open() as fh:
            n += sum(1 for _ in fh)
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    p.add_argument("--target", type=int, default=60000, help="total stories wanted in out dir")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--temp", type=float, default=0.6)
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--out", default="/Volumes/Modelltest/pytte/data/sagosvenska/raw")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-prefix-cache", action="store_true", help="recompute the shared system prompt every time")
    p.add_argument("--recipe", choices=sorted(RECIPES), default="v2", help="prompt recipe (v1 = SagoSvenska v1, 2026-11-06)")
    args = p.parse_args()
    make = RECIPES[args.recipe]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stop_file = out_dir.parent / "STOP"
    have = count_existing(out_dir)
    if have >= args.target:
        print(f"Already have {have} stories (target {args.target}). Nothing to do.")
        return

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_file = out_dir / f"stories-{stamp}.jsonl"
    review_file = out_dir / f"review-{stamp}.jsonl"
    seed = args.seed if args.seed is not None else int(time.time())
    seed = seed + have          # a restart continues with new prompts instead of replaying the first ones
    rng = random.Random(seed)

    # Hard caps so a slow buffer-cache creep cannot end in a Metal OOM over a multi-day run.
    mx.set_cache_limit(2 * 1024**3)       # at most 2 GB of recycled buffers kept between ops
    mx.set_memory_limit(22 * 1024**3)     # allocations above this wait for the GPU queue to drain
    print(f"Loading {args.model} ...", flush=True)
    model, tokenizer = load(args.model)
    sampler = make_sampler(temp=args.temp, top_p=0.95)
    logits_processors = [ban_special_tokens(tokenizer)]
    print(f"Have {have}, target {args.target}, batch {args.batch}, temp {args.temp}, seed {seed}, recipe {args.recipe}", flush=True)
    print(f"Writing to {out_file}. Create {stop_file} to stop.", flush=True)

    prefix_ids, prefix_cache = [], None
    if not args.no_prefix_cache:
        probe = [encode(tokenizer, make(random.Random(i))[1]) for i in range(8)]
        prefix_ids = common_prefix(probe)
        prefix_cache = build_prefix_cache(model, prefix_ids)
        print(f"Prefix cache: {len(prefix_ids)} shared tokens of ~{len(probe[0])} per prompt", flush=True)

    t_start = time.time()
    made = flagged = 0
    paused = 0.0   # seconds spent waiting on PAUSE; excluded from the rate/ETA
    with out_file.open("a") as fo, review_file.open("a") as fr:
        while have + made < args.target:
            pause_file = out_dir.parent / "PAUSE"
            if pause_file.exists():
                print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] PAUSE file found, waiting ...", flush=True)
                t_pause = time.time()
                while pause_file.exists() and not stop_file.exists():
                    time.sleep(30)
                paused += time.time() - t_pause
                print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] resuming", flush=True)
            if stop_file.exists():
                print("STOP file found, exiting.", flush=True)
                break
            n = min(args.batch, args.target - have - made)
            batch = [make(rng) for _ in range(n)]
            prompts = [encode(tokenizer, text) for _, text in batch]
            prompts, caches = split_with_cache(prompts, prefix_ids, prefix_cache)
            t0 = time.time()
            resp = batch_generate(model, tokenizer, prompts, prompt_caches=caches if prefix_cache else None,
                                  max_tokens=args.max_tokens, sampler=sampler, return_token_ids=True,
                                  logits_processors=logits_processors)
            dt_batch = time.time() - t0
            now = dt.datetime.now().isoformat(timespec="seconds")
            for (ing, _), text, ids in zip(batch, resp.texts, resp.token_ids):
                story, n_special = clean(text)
                rec = {**ing, "text": story, "n_tokens": len(ids), "model": args.model,
                       "temp": args.temp, "seed": seed, "created": now, "recipe": args.recipe}
                truncated = len(ids) >= args.max_tokens
                if truncated or len(story) < 150 or UNSAFE.search(story) or n_special:
                    rec["reason"] = ("truncated" if truncated else "short" if len(story) < 150
                                     else "unsafe" if UNSAFE.search(story) else "special_token")
                    fr.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    flagged += 1
                else:
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    made += 1
            fo.flush(); fr.flush()
            mx.clear_cache()   # return MLX's buffer cache to the system between batches; GPU OOM otherwise over long runs
            mem_gb = mx.get_active_memory() / 1024**3
            peak_gb = mx.get_peak_memory() / 1024**3
            elapsed = time.time() - t_start - paused
            rate = (made + flagged) / elapsed * 3600
            left = args.target - have - made
            eta_h = left / rate if rate else float("inf")
            print(f"[{now}] total {have + made:>6}/{args.target} | +{n} in {dt_batch:5.1f}s | "
                  f"{rate:6.0f} stories/h | flagged {flagged} | mem {mem_gb:4.1f}/{peak_gb:4.1f} GB | ETA {eta_h:5.1f} h", flush=True)

    print(f"Done. Wrote {made} stories, {flagged} to review, in {(time.time()-t_start)/3600:.2f} h.")


if __name__ == "__main__":
    main()
