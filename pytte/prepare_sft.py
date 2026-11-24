"""Build Pytte's chat (SFT) data from SagoSvenska's own recipes, no new generation needed.

Every story in the dataset carries the ingredients its prompt asked for (name, place,
noun, verb, adjective, plural word, tense, perspective, features). A short Swedish
request is written from a random subset of them and the story becomes the answer:

    Du: Berätta en saga om en älg som är blyg.
    Pytte: <the story><|endoftext|>

Also built: two-turn continuations ("Vad händer sedan?"), "en till", a few hundred
greetings / "who are you" / "I can only tell stories" exchanges, and a validation set
from val.jsonl. Output: readable JSONL conversations plus the flat token files the
trainer reads (train.bin + train.mask.bin, where the mask marks the assistant tokens
the loss is computed on).

    python -m pytte.prepare_sft --data data/sagosvenska-mix1/clean --tok data/tok-sv3-4096 \
        --out data/tokens-sft-sv3-4096 --jsonl-out data/sft-mix1
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .tokenizer import EOT, load as load_tok

USER, BOT = "Du:", "Pytte:"

OPENERS = [  # (weight, text); {kort} is replaced by "kort "/"lång "/""
    (30, "Berätta en {kort}saga"), (12, "Kan du berätta en {kort}saga"), (8, "Jag vill höra en {kort}saga"),
    (8, "Skriv en {kort}saga"), (6, "Hitta på en {kort}saga"), (5, "En {kort}saga"), (4, "Kan jag få en {kort}saga"),
    (4, "Snälla, berätta en {kort}saga"), (4, "Berätta en {kort}godnattsaga"), (3, "Jag vill ha en {kort}saga"),
    (3, "Kan du hitta på en {kort}saga"), (3, "Säg en {kort}saga"), (2, "Berätta något"), (2, "{Kort}saga"),
]
FOLLOW_UPS = ["Vad händer sedan?", "Fortsätt.", "Och sedan?", "Berätta mer.", "Vad hände sen?", "Fortsätt berätta!",
              "Och sen då?", "Mer!", "Vad händer sen?", "Fortsätt, snälla.", "Hur går det sedan?", "Sen då?"]
ANOTHER = ["En till!", "Berätta en till.", "En saga till, tack.", "Kan du berätta en till?", "Igen!", "En till saga.",
           "Berätta en saga till.", "Mer sagor!", "Kan jag få en till?"]
# Recipe features/endings that make sense in a short request (the rest are too abstract to ask for).
FEATURE_PHRASES = {
    "ett djur som pratar": "ett djur som pratar", "en present": "en present", "något som smakar gott": "något som smakar gott",
    "en figur som gömmer sig": "någon som gömmer sig", "en saga som utspelar sig på natten": "natten",
    "något som går sönder och lagas": "något som går sönder", "en sång eller ramsa som upprepas": "en sång",
    "något som försvinner och hittas igen": "något som försvinner och hittas igen", "en resa fram och tillbaka": "en resa",
    "en figur som är rädd men blir modig": "någon som är rädd men blir modig", "en figur som säger förlåt": "någon som säger förlåt",
    "en överraskning i en låda eller ett paket": "en överraskning", "ett väder som ändras mitt i sagan": "väder som ändras",
    "någon som väntar på något": "någon som väntar", "en lek med regler": "en lek", "en vuxen som tröstar": "någon som tröstar",
    "en figur som hjälper en annan": "någon som hjälper", "ett lyckligt slut": "ett lyckligt slut",
}
ENDING_PHRASES = {
    "ett lyckligt slut": "ett lyckligt slut", "ett lugnt slut där alla somnar": "ett slut där alla somnar",
    "ett roligt slut": "ett roligt slut", "ett slut där figurerna säger god natt": "ett slut där de säger god natt",
    "ett slut där de delar med sig": "ett slut där de delar med sig", "ett slut med en liten överraskning": "en överraskning på slutet",
}

GREETINGS = ["Hej", "Hej!", "Hejsan", "Hallå", "Hej hej", "God morgon", "God kväll", "Tjena", "Hej Pytte", "Hallå där", "Hej på dig"]
GREETING_REPLIES = ["Hej! Jag heter Pytte. Vill du höra en saga?", "Hej! Vill du höra en saga?",
                    "Hej hej! Jag kan berätta en saga. Vad ska den handla om?", "Hej! Jag är Pytte. Säg vad sagan ska handla om."]
WHO = ["Vem är du?", "Vad heter du?", "Vad är du för något?", "Vad är du?", "Är du en robot?", "Vem pratar jag med?",
       "Berätta om dig själv.", "Vad är Pytte?", "Är du en människa?", "Hur stor är du?"]
WHO_REPLIES = ["Jag heter Pytte. Jag är en pytteliten språkmodell som har lärt sig svenska av sagor. Jag kan berätta en saga om du vill.",
               "Jag är Pytte, en mycket liten språkmodell. Jag kan bara en sak: berätta korta sagor på svenska. Vad ska sagan handla om?",
               "Jag heter Pytte och jag är en liten dator som kan berätta sagor. Vill du höra en?"]
CAN = ["Vad kan du göra?", "Vad kan du?", "Hjälp", "Vad gör du?", "Hur funkar du?", "Vad ska jag säga?", "Vad kan du berätta?"]
CAN_REPLIES = ["Jag kan berätta korta sagor på svenska. Säg vad sagan ska handla om, till exempel en katt, en båt eller ett barn som heter Elsa.",
               "Jag berättar sagor. Säg till exempel: Berätta en saga om en liten räv.",
               "Bara en sak: sagor. Säg vem sagan ska handla om, så börjar jag."]
THANKS = ["Tack", "Tack!", "Tack så mycket", "Tack snälla", "Bra saga!", "Fin saga", "Vilken bra saga!", "Jag gillade den", "Den var rolig"]
THANKS_REPLIES = ["Varsågod! Vill du höra en saga till?", "Tack! Säg till om du vill ha en till.", "Vad roligt! Ska jag berätta en till?"]
BYE = ["Hej då", "Hej då!", "God natt", "Nu ska jag sova", "Vi ses", "Godnatt Pytte", "Jag måste gå"]
BYE_REPLIES = ["Hej då! Sov gott.", "God natt! Dröm något fint.", "Hej då! Kom tillbaka om du vill höra fler sagor."]
OFFTOPIC = ["Vad är huvudstaden i Frankrike?", "Hur mycket är 7 gånger 8?", "Vad blir det för väder i morgon?", "Skriv ett program i Python.",
            "Vem vann valet?", "Vad är klockan?", "Översätt det här till engelska.", "Vad betyder fotosyntes?", "Hur många planeter finns det?",
            "Vem är Sveriges statsminister?", "Vad kostar en bil?", "Kan du hjälpa mig med matteläxan?", "Vad heter du på engelska?",
            "Räkna till hundra.", "Vad är meningen med livet?", "Hur lagar man pannkakor?", "Vilken dag är det i dag?", "Berätta ett skämt.",
            "Vad är 2 plus 2?", "Hur gammal är du?", "Var bor du?", "Vad tycker du om fotboll?", "Vilket år är det?", "Hur långt är det till månen?",
            "Kan du sjunga?", "Skriv en dikt om kärlek.", "Vad händer i nyheterna?", "Vem uppfann telefonen?", "Hur stavas giraff?",
            "Vad ska jag äta till middag?", "Rätta min uppsats.", "Vad är en algoritm?", "Ge mig ett recept på kakor.", "Hur mår du?"]
OFFTOPIC_REPLIES = ["Det vet jag inte. Jag är en pytteliten modell och kan bara berätta sagor. Vill du höra en saga?",
                    "Det kan jag tyvärr inte. Jag kan bara en sak: berätta sagor på svenska. Vad ska sagan handla om?",
                    "Jag vet inte. Jag är för liten för sådant, men jag kan berätta en saga om du vill.",
                    "Det klarar jag inte. Fråga mig om en saga i stället, till exempel om en katt eller en båt."]

WORD = re.compile(r"[a-zåäöé]+")
NO_ARTICLE = {"saft", "mjölk", "soppa", "snö", "is", "regn", "skridskor"}   # mass nouns and a plural
PRESENT_OVERRIDE = {"tappa": "tappar"}   # the corpus count picks the adjective "tapper"


# ------------------------------------------------------------------ lexicon from the corpus
def build_lexicon(texts, nouns, verbs):
    """Data-driven word forms: noun article (en/ett/none for mass nouns) and verb present tense,
    chosen by corpus frequency so the requests use the forms the stories actually use."""
    uni, art = Counter(), Counter()
    for t in texts:
        ws = WORD.findall(t.lower())
        uni.update(ws)
        for a, b in zip(ws, ws[1:]):
            if a in ("en", "ett"):
                art[(a, b)] += 1
    article = {}
    for n in nouns:
        en, ett = art[("en", n)], art[("ett", n)]
        if n in NO_ARTICLE:
            article[n] = None
        else:
            article[n] = ("en" if en >= ett else "ett") if max(en, ett) >= 5 else "en"
    present = {}
    for v in verbs:
        head, _, tail = v.partition(" ")           # "resa sig" -> "reser sig"
        stem = head[:-1] if head.endswith("a") else head
        cands = [stem + "ar", stem + "er"] if head.endswith("a") else [head + "r"]
        best = max(cands, key=lambda c: uni[c])
        present[v] = PRESENT_OVERRIDE.get(v, (best + (" " + tail if tail else "")) if uni[best] >= 5 else None)
    return {"article": article, "present": present, "unigram": uni}


# ------------------------------------------------------------------ requests
def _pick(rng, weighted):
    return rng.choices([w[1] for w in weighted], weights=[w[0] for w in weighted])[0]


DENSE = False   # --dense: requests mention 2-3 ingredients and the noun more often (stronger conditioning signal)


def make_request(rng, rec, lex):
    """Return (request text, dict of ingredients the request mentions)."""
    name, noun, adj, verb, place = rec["name"], rec["noun"], rec["adj"], rec["verb"], rec["place"]
    art = lex["article"].get(noun)
    noun_phrase = f"{art} {noun}" if art else noun
    used = {}
    slots = []  # (weight, key, phrase) ; phrases that start with "om" are the topic, others are trailers
    slots.append((40, "name", f"om {name}"))
    slots.append((70 if DENSE else 40, "noun", f"om {noun_phrase}"))
    if art == "en":
        slots.append((35 if DENSE else 20, "adj_noun", f"om en {adj} {noun}"))
    slots.append((20 if DENSE else 12, "adj", f"där {name} är {adj}"))
    if lex["present"].get(verb):
        slots.append((14 if DENSE else 8, "verb", f"där {name} {lex['present'][verb]}"))
    slots.append((14 if DENSE else 8, "verb_inf", f"om {name} som vill {verb}"))
    slots.append((14, "place", f"som utspelar sig {place}"))
    if rec.get("plural"):
        slots.append((20, "plural", f"med många {rec['plural']}"))
    if rec.get("tense") == "preteritum":
        slots.append((14, "tense", "i dåtid"))
    if rec.get("first_person"):
        slots.append((30, "first_person", f"där {name} berättar själv"))
    feats = [FEATURE_PHRASES[f] for f in rec.get("features") or [rec.get("feature")] if f in FEATURE_PHRASES]
    if feats:
        slots.append((8, "feature", f"med {rng.choice(feats)}"))
    if rec.get("ending") in ENDING_PHRASES:
        slots.append((6, "ending", f"med {ENDING_PHRASES[rec['ending']]}"))
    if rec.get("opening") == "klassisk":
        slots.append((10, "opening", "som börjar med Det var en gång"))

    # which ingredients each phrase mentions; one phrase per ingredient
    COVERS = {"name": {"name"}, "noun": {"noun"}, "adj_noun": {"adj", "noun"}, "adj": {"adj", "name"},
              "verb": {"verb", "name"}, "verb_inf": {"verb", "name"}, "place": {"place"}, "plural": {"plural"},
              "tense": {"tense"}, "first_person": {"first_person", "name"}, "feature": {"feature"},
              "ending": {"ending"}, "opening": {"opening"}}
    k = rng.choices([0, 1, 2, 3], weights=[3, 25, 42, 30] if DENSE else [6, 42, 36, 16])[0]
    chosen = []
    covered = set()
    pool = list(slots)
    while pool and len(chosen) < k:
        w, key, phrase = _pick(rng, [(s[0], s) for s in pool])
        pool.remove((w, key, phrase))
        if COVERS[key] & covered:
            continue
        covered |= COVERS[key]
        chosen.append((key, phrase))
    # topic phrases ("om ...") first (the name before a thing), at most two joined by "och"
    order = {"name": 0, "adj": 1, "verb": 1, "verb_inf": 1, "first_person": 1, "noun": 2, "adj_noun": 2}
    chosen.sort(key=lambda kp: order.get(kp[0], 3))
    topics = [p for k_, p in chosen if p.startswith("om ")]
    trailers = [p for k_, p in chosen if not p.startswith("om ")]
    if len(topics) > 2:
        trailers = [p for p in topics[2:]] + trailers
        topics = topics[:2]
    if len(topics) == 2:
        topics = [topics[0] + " och " + topics[1][3:]]
    vals = {"name": name, "noun": noun, "adj": adj, "verb": verb, "place": place, "plural": rec.get("plural"),
            "tense": rec.get("tense"), "first_person": rec.get("first_person")}
    for ing in covered:
        if ing in vals:
            used[ing] = vals[ing]
    words = rec.get("words", 100)
    kort = ""
    if words <= 80 and rng.random() < 0.35:
        kort = "kort "
    elif words >= 150 and rng.random() < 0.35:
        kort = "lång "
    opener = _pick(rng, OPENERS)
    if "{Kort}" in opener:
        opener = (kort.capitalize() + "saga") if kort else "Saga"
    opener = opener.replace("{kort}", kort)
    text = " ".join([opener] + topics + trailers)
    if rng.random() < 0.10:
        text += ", tack"
    text += "?" if opener.startswith("Kan") else ("!" if rng.random() < 0.12 else ".")
    if rng.random() < 0.15:
        text = text.lower()
    if rng.random() < 0.15:
        text = text.rstrip(".!?")
    return text, used


def split_story(text, rng):
    """Split at a sentence boundary near the middle; None if too short."""
    sents = re.findall(r"[^.!?]+[.!?]+(?:\"|”)?", text)
    if len(sents) < 6:
        return None
    lo, hi = max(3, int(len(sents) * 0.35)), min(len(sents) - 3, int(len(sents) * 0.65))
    if lo > hi:
        return None
    cut = rng.randint(lo, hi)
    a, b = "".join(sents[:cut]).strip(), "".join(sents[cut:]).strip()
    return (a, b) if a and b else None


def build_conversations(recs, lex, rng, cont_frac=0.35, another_frac=0.08, extras=0.0, episode_frac=0.10):
    convs = []
    by_name = {}
    for j, r in enumerate(recs):
        by_name.setdefault(r["name"], []).append(j)
    i = 0
    while i < len(recs):
        rec = recs[i]
        req, used = make_request(rng, rec, lex)
        text = rec["text"].strip()
        turns = [{"role": "user", "content": req}]
        r = rng.random()
        parts = split_story(text, rng) if r < cont_frac else None
        if parts:
            turns += [{"role": "assistant", "content": parts[0]},
                      {"role": "user", "content": rng.choice(FOLLOW_UPS)},
                      {"role": "assistant", "content": parts[1]}]
            kind = "continue"
        else:
            turns.append({"role": "assistant", "content": text})
            kind = "single"
        i += 1
        if kind == "single" and i < len(recs) and rng.random() < another_frac:
            turns += [{"role": "user", "content": rng.choice(ANOTHER)},
                      {"role": "assistant", "content": recs[i]["text"].strip()}]
            kind = "another"
            i += 1
        elif kind == "single" and rng.random() < episode_frac and len(by_name[rec["name"]]) > 1:
            # "Vad händer sedan?" after a complete story: a new story about the same child.
            # (The pilot model ended the tale in one sentence here, because training only had
            # follow-ups after half a story.)
            j = rng.choice(by_name[rec["name"]])
            if recs[j] is not rec:
                turns += [{"role": "user", "content": rng.choice(FOLLOW_UPS)},
                          {"role": "assistant", "content": recs[j]["text"].strip()}]
                kind = "episode"
        convs.append({"kind": kind, "recipe": rec.get("recipe", "v1"), "used": used, "messages": turns})
    n_extra = int(len(convs) * extras)
    for _ in range(n_extra):
        group = rng.choices(["greet", "who", "can", "thanks", "bye", "off"], weights=[20, 15, 10, 15, 10, 30])[0]
        q, a = {"greet": (GREETINGS, GREETING_REPLIES), "who": (WHO, WHO_REPLIES), "can": (CAN, CAN_REPLIES),
                "thanks": (THANKS, THANKS_REPLIES), "bye": (BYE, BYE_REPLIES), "off": (OFFTOPIC, OFFTOPIC_REPLIES)}[group]
        convs.append({"kind": group, "recipe": None, "used": {},
                      "messages": [{"role": "user", "content": rng.choice(q)}, {"role": "assistant", "content": rng.choice(a)}]})
    rng.shuffle(convs)
    return convs


# ------------------------------------------------------------------ tokens
def encode_conversation(tok, messages, eot):
    """Token ids + mask (1 = assistant token or its closing EOT). Each assistant turn ends with EOT."""
    ids, mask = [eot], [0]
    for m in messages:
        if m["role"] == "user":
            seg = tok.encode(f"{USER} {m['content']}\n{BOT}").ids
            ids += seg; mask += [0] * len(seg)
        else:
            seg = tok.encode(" " + m["content"]).ids + [eot]
            ids += seg; mask += [1] * len(seg)
    return ids, mask


def prompt_ids(tok, messages):
    """Ids for generation: the conversation so far, ending with 'Pytte:' (used by pytte.chat)."""
    eot = tok.token_to_id(EOT)
    ids = [eot]
    for m in messages:
        if m["role"] == "user":
            ids += tok.encode(f"{USER} {m['content']}\n{BOT}").ids
        else:
            ids += tok.encode(" " + m["content"]).ids + [eot]
    return ids


def write_tokens(tok, convs, out_dir, split):
    eot = tok.token_to_id(EOT)
    all_ids, all_mask = [], []
    for c in convs:
        ids, mask = encode_conversation(tok, c["messages"], eot)
        all_ids += ids; all_mask += mask
    ids = np.array(all_ids, dtype=np.uint16); mask = np.array(all_mask, dtype=np.uint8)
    ids.tofile(out_dir / f"{split}.bin"); mask.tofile(out_dir / f"{split}.mask.bin")
    return len(ids), int(mask.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="clean dir with train.jsonl and val.jsonl")
    p.add_argument("--tok", required=True)
    p.add_argument("--out", required=True, help="token dir")
    p.add_argument("--jsonl-out", required=True, help="readable conversations")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cont-frac", type=float, default=0.35)
    p.add_argument("--another-frac", type=float, default=0.08)
    p.add_argument("--extras", type=float, default=0.025, help="greetings/identity/off-topic, fraction of conversations")
    p.add_argument("--episode-frac", type=float, default=0.10, help="follow-up after a complete story -> new story, same child")
    p.add_argument("--dense", action="store_true", help="2-3 ingredients per request, noun/adjective/verb more often")
    p.add_argument("--per-story", type=int, default=1,
                   help="passes over the training stories; each pass draws new requests, split points and kinds "
                        "for the same stories (more variety at zero generation cost; duplicates are dropped)")
    args = p.parse_args()
    global DENSE
    DENSE = args.dense

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
    from gen_stories import NOUNS, VERBS  # noqa: E402  (the recipe lists)

    data = Path(args.data)
    train = [json.loads(l) for l in (data / "train.jsonl").open()]
    val = [json.loads(l) for l in (data / "val.jsonl").open()]
    lex = build_lexicon((r["text"] for r in train), NOUNS, VERBS)
    rng = random.Random(args.seed)
    convs_train = []
    seen = set()
    for k in range(args.per_story):
        pass_rng = random.Random(args.seed + 1000 * k)
        for c in build_conversations(train, lex, pass_rng, args.cont_frac, args.another_frac, args.extras, args.episode_frac):
            # Drop exact repeats of story conversations only: small talk comes from short fixed
            # lists and *needs* its repeats (the first x3 build deduped it down to ~50 lines and
            # the models lost small talk entirely).
            key = json.dumps(c["messages"], ensure_ascii=False) if c["recipe"] is not None else None
            if key is None or key not in seen:
                if key is not None:
                    seen.add(key)
                convs_train.append(c)
    rng.shuffle(convs_train)
    if args.per_story > 1:
        print(f"{args.per_story} passes over {len(train)} stories -> {len(convs_train)} unique conversations")
    convs_val = build_conversations(val, lex, random.Random(args.seed + 1), args.cont_frac, args.another_frac, args.extras, args.episode_frac)

    jout = Path(args.jsonl_out); jout.mkdir(parents=True, exist_ok=True)
    for split, convs in (("train", convs_train), ("val", convs_val)):
        with (jout / f"{split}.jsonl").open("w") as f:
            for c in convs:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    lexout = {"article": lex["article"], "present": lex["present"]}
    (jout / "lexicon.json").write_text(json.dumps(lexout, indent=1, ensure_ascii=False))

    tok = load_tok(args.tok)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    meta = {"vocab_size": tok.get_vocab_size(), "tokenizer": str(args.tok), "format": f"{USER} ... \\n{BOT} ...<|endoftext|>",
            "source": str(data), "seed": args.seed, "per_story": args.per_story, "dense": args.dense}
    for split, convs in (("train", convs_train), ("val", convs_val)):
        n_tok, n_loss = write_tokens(tok, convs, out, split)
        kinds = Counter(c["kind"] for c in convs)
        meta[split] = {"conversations": len(convs), "tokens": n_tok, "loss_tokens": n_loss, "kinds": dict(kinds)}
        print(f"{split}: {len(convs)} conversations, {n_tok/1e6:.2f}M tokens, {n_loss/1e6:.2f}M in the loss, {dict(kinds)}")
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    for c in convs_train[:8]:
        print("---", c["kind"], c["used"])
        for m in c["messages"]:
            print(f"  {m['role']:9s} {m['content'][:110]}")


if __name__ == "__main__":
    main()
