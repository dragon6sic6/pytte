# Pytte

**How small can a language model be and still write coherent Swedish?**

Pytte (Swedish for *tiny*) is a family of Llama-style language models, from 0.3M to 40M
parameters, trained from scratch on a MacBook. Every part of a modern LLM is here and
written in plain, readable code: byte-level BPE tokenizer, RoPE, RMSNorm, grouped-query
attention, SwiGLU, tied embeddings, KV cache, AdamW with warmup and cosine decay. Nothing
is fine-tuned, distilled or wrapped. The weights start random.

The same code trains a Swedish and an English twin on matched data, so we can ask:
*how many parameters does Swedish need compared to English?*

> **Status (25 November 2026).** The work is done and written up: **the report is
> [`paper/pytte.pdf`](paper/pytte.pdf)** (19 pages, English with a Swedish summary). SagoSvenska
> has 120 649 stories in three sets, six model sizes (0.3M to 15.7M parameters) have been trained
> on three corpus mixes plus two English controls, the release family is instruction-tuned,
> exported to one int8 file per model, and runs in `run.c` and in the browser (`web/`). A 0.33M
> model writes grammatical Swedish sentences; coherent stories need about 1M parameters. The lab
> log with every run is `FÖRSLAG-OCH-PLAN.md` (Swedish). This is a hobby project by one person and
> has not been reviewed by anyone else.

## What this is not

- Not "the world's smallest LLM". Transformers with 25k parameters exist on a Commodore 64.
  A size claim without a quality floor is meaningless. Pytte defines the floor first
  (see `eval/`) and then asks how small a model can be while clearing it.
- Not large. It is a small language model with the same architecture and training method
  as large ones.
- Not the first Swedish TinyStories. A Linköping University course published a
  machine-translated TinyStories and 15 Swedish models in December 2024
  (`tdde19-llm-from-scratch` on Hugging Face). That data has no license, so it could
  not be reused here.

## Layout

```
pytte/            the model: tokenizer.py, prepare_data.py, model.py, train.py, sample.py, eval.py, judge.py,
                  prepare_sft.py (chat data from story recipes), chat.py (chat loop), export.py (one .bin file)
run.c             runs a .bin without Python: tokenizer, forward pass, sampling, chat; no dependencies
web/              the browser demo: pytte.js (run.c in JavaScript), worker.js, index.html, app.js, models/
data/             dataset tooling: gen_stories.py (SagoSvenska generation), clean_stories.py
eval/             minimal_pairs_sv.json, minimal_pairs_en.json (200 length-matched pairs each)
scripts/          train_family.sh, reeval_family.sh, judge_family.py, family_table.py,
                  chain_sft_family.sh, eval_sft.py, chat_family_table.py (the chat family),
                  check_run.py, pairs_run.py, q8_eval.py (run.c and int8 against Python)
bench/            MLX vs PyTorch throughput and generation-model tests
paper/            the report: pytte.pdf, its source, results.json, and the scripts that build tables and figures
FÖRSLAG-OCH-PLAN.md   the plan and the measurement log (Swedish)
```

Large files (datasets, tokens, checkpoints, generation models) live outside the repo.

## Quick start (Apple Silicon)

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python mlx mlx-lm tokenizers datasets numpy
.venv/bin/python -m pytte.tokenizer train --input train.txt --vocab 4096 --out tok/
.venv/bin/python -m pytte.prepare_data --tok tok/ --train train.txt --val valid.txt --out tokens/
.venv/bin/python -m pytte.train --config pytte-1m --data tokens/ --tok tok/ --out ckpt/pytte-1m --minutes 5
.venv/bin/python -m pytte.sample --model ckpt/pytte-1m --tok tok/ --prompt "Once upon a time"
.venv/bin/python -m pytte.eval --model ckpt/pytte-1m --tok tok/ --data tokens/ --pairs eval/minimal_pairs_en.json
# chat tuning: build request/story conversations from the story recipes, fine-tune, talk to it
.venv/bin/python -m pytte.prepare_sft --data stories/clean --tok tok/ --out tokens-sft/ --jsonl-out sft/ --dense
.venv/bin/python -m pytte.train --init ckpt/pytte-1m --data tokens-sft/ --tok tok/ --out ckpt/pytte-1m-chat --minutes 20 --lr 3e-4
.venv/bin/python -m pytte.chat --model ckpt/pytte-1m-chat --tok tok/
# without Python: export to one int8 file and run it in C
.venv/bin/python -m pytte.export --model ckpt/pytte-1m-chat --tok tok/ --out pytte-1m-chat.bin
cc -O3 -o run run.c -lm
./run pytte-1m-chat.bin -chat
```

## Model family

| Name | dim | layers | heads | ff | params | without embeddings |
|---|---:|---:|---:|---:|---:|---:|
| pytte-0.3m | 64 | 4 | 4 (2 kv) | 192 | 0.33M | 0.20M |
| pytte-1m | 128 | 4 | 4 | 352 | 1.33M | 0.80M |
| pytte-2m | 160 | 5 | 5 | 448 | 2.24M | 1.59M |
| pytte-3m | 192 | 6 | 6 | 512 | 3.44M | 2.66M |
| pytte-7m | 288 | 6 | 6 | 768 | 7.16M | 5.98M |
| pytte-15m | 384 | 8 | 6 | 1024 | 15.74M | 14.16M |
| pytte-40m | 512 | 12 | 8 | 1408 | 40.65M | 38.55M |

Vocabulary 4096 (2048 for pytte-0.3m). Parameter counts are always given with and without
the embedding table; in the smallest model the table is 40% of the parameters.

## Results so far (13 November 2026)

Same code, same six configurations, same wall-clock budgets on an M2 Pro; only the data differs.
`sv` is SagoSvenska (8.1M tokens), `en` is all of TinyStories V2 (553M tokens) and `en8m` is a
random TinyStories subset of exactly SagoSvenska's size (40 027 stories, 8.1M tokens), so that
`sv` and `en8m` see the same amount of data and the same 6–9 epochs. Bits/byte is the cross-vocabulary
measure; the grammar test is 200 length-matched minimal pairs per language; the judge is
Gemma 4 12B grading grammar / coherence 1–10 on 50 random samples per model. `sv` rows are
trained on SagoSvenska v1.1 (8.1M tokens), `sv2` rows on v1.1 + v2 (118 269 stories, 17M tokens)
and **`sv3` rows on the release mix** (v1.1 plus the one-child v2 stories, 85 764 stories, 12M
tokens), all with the same budgets; see the paragraphs after the table before comparing their
bits/byte.

| Model | Lang | Min | Tokens | Bits/byte | Minimal pairs | Judge gram. / coh. | Ends on its own |
|---|---|---:|---:|---:|---:|---:|---:|
| pytte-0.3m | sv | 5 | 82M | 0.538 | 67.5% | 8.7 / 5.3 | 86% |
| pytte-0.3m | en | 5 | 86M | 0.842 | 83.5% | 7.2 / 2.9 | 67% |
| pytte-0.3m | en8m | 5 | 79M | 0.859 | 79.0% | 7.2 / 2.8 | 55% |
| pytte-0.3m | sv2 | 5 | 87M | 0.654 | 82.5% | 8.1 / 4.7 | 94% |
| pytte-0.3m | **sv3** | 5 | 87M | 0.619 | 82.0% | 8.7 / 5.3 | 92% |
| pytte-1m | sv | 8 | 80M | 0.433 | 76.0% | 9.5 / 8.8 | 95% |
| pytte-1m | en | 8 | 81M | 0.650 | 90.0% | 9.3 / 4.5 | 73% |
| pytte-1m | en8m | 8 | 74M | 0.690 | 88.0% | 8.8 / 4.9 | 77% |
| pytte-1m | sv2 | 8 | 85M | 0.525 | 91.5% | 9.2 / 6.9 | 85% |
| pytte-1m | **sv3** | 8 | 83M | 0.493 | 91.5% | 9.5 / 8.3 | 95% |
| pytte-2m | sv | 10 | 60M | 0.417 | 77.0% | 9.5 / 9.0 | 93% |
| pytte-2m | en | 10 | 60M | 0.634 | 90.5% | 9.3 / 4.9 | 83% |
| pytte-2m | en8m | 10 | 59M | 0.657 | 91.5% | 9.0 / 4.9 | 72% |
| pytte-2m | sv2 | 10 | 57M | 0.506 | 89.0% | 9.6 / 7.0 | 91% |
| pytte-2m | **sv3** | 10 | 54M | 0.478 | 88.5% | 9.6 / 8.4 | 97% |
| pytte-3m | sv | 12 | 54M | 0.409 | 79.0% | 9.7 / 9.1 | 97% |
| pytte-3m | en | 12 | 56M | 0.606 | 91.5% | 9.5 / 6.1 | 71% |
| pytte-3m | en8m | 12 | 52M | 0.643 | 90.0% | 9.6 / 5.9 | 77% |
| pytte-3m | sv2 | 12 | 56M | 0.491 | 90.0% | 9.5 / 7.8 | 89% |
| pytte-3m | **sv3** | 12 | 54M | 0.466 | 94.0% | 9.4 / 8.4 | 94% |
| pytte-7m | sv | 20 | 55M | 0.403 | 80.0% | 9.5 / 9.1 | 93% |
| pytte-7m | en | 20 | 57M | 0.576 | 90.5% | 9.5 / 5.6 | 82% |
| pytte-7m | en8m | 20 | 59M | 0.623 | 92.5% | 9.2 / 5.1 | 72% |
| pytte-7m | sv2 | 20 | 59M | 0.472 | 95.0% | 9.5 / 8.0 | 91% |
| pytte-7m | **sv3** | 20 | 55M | 0.451 | 94.5% | 9.6 / 9.0 | 99% |
| pytte-15m | sv | 30 | 52M | 0.405 | 77.0% | 9.7 / 9.4 | 98% |
| pytte-15m | en | 30 | 51M | 0.551 | 94.0% | 9.8 / 7.3 | 80% |
| pytte-15m | en8m | 30 | 53M | 0.620 | 93.0% | 9.5 / 6.0 | 88% |
| pytte-15m | sv2 | 30 | 53M | 0.461 | 90.0% | 9.5 / 8.0 | 96% |
| pytte-15m | **sv3** | 30 | 47M | 0.445 | 93.0% | 9.5 / 8.9 | 96% |

Four things stand out. Pytte-0.3M already writes coherent Swedish (judge coherence 5.3 vs
2.8–2.9 for both English twins). The Swedish family plateaus at 3M parameters and the
size-matched English one at 7M, while English on the full corpus keeps improving to 15M: with
8M tokens the data, not the model, is the ceiling, and the gap between `en` and `en8m` grows
with model size (0.02 bits/byte at 0.3M, 0.07 at 15M). The Swedish models' lower bits/byte
is a property of the data, not the language: on the same 8M tokens English sits 0.2 bits/byte
higher, because SagoSvenska (generated at temperature 0.6) is more predictable than
TinyStories. And number agreement (Swedish plurals, English subject–verb) is the last
grammatical skill to arrive in both languages; it is not a data-size effect, since `en8m`
passes the grammar test as well as `en`. The Swedish grammar score should be read with care:
retraining on the repaired data moved it by up to 10 points for the same configuration while
every other metric stayed put, and three seeds of pytte-0.3m span 67.5–80.0% at identical
bits/byte (pytte-1m: 76–77%). The test is stable from 1M; at 0.3M it must be read as a range.

Retraining on v1.1 + v2 (the `sv2` rows, 12 November) changed three things at once. Grammar
rose 12–15 points at every size (1M: 76 → 91.5%, 7M: 80 → 95%), far above the test's noise,
and plural agreement went from chance (35–55%) to 85–100% because 40% of the v2 prompts ask
for plural objects; adjective agreement and verb-second word order rose similarly, while
subordinate-clause negation, which no prompt asks for, did not move. The `sv` models turn out
to have learned the v1 template rather than Swedish: on v2 text they cost 1.06–1.08 bits/byte
against 0.41–0.44 on their own validation set, whereas the `sv2` models cost 0.52–0.58 on v2 text
and 0.00–0.02 more on v1 text (so the `sv` and `sv2` bits/byte columns above, each measured on
its own validation set, are not comparable). The price is coherence: the judge gives the `sv2`
models 1.1–2.0 points less from 1M up, with grammar unchanged, and its comments point at
character names being swapped mid-story in the stories with more than one character that v2 added.
The `sv3` rows (13 November) are the resolution: all of v1.1 plus only the one-child v2 stories.
Grammar stays where `sv2` put it (88.5–94.5% from 1M, plurals 90–95%), coherence comes back to
8.3–8.4 at 1M–3M and to 9.0 / 8.9 at 7M / 15M (against 9.1 / 9.4 for `sv`), language errors are
back at 0.3–0.5 per story, and on v2 text the models cost 0.60–0.65 bits/byte instead of the
`sv` models' 1.06–1.08. From 7M the model has the capacity to hold a freer story together; below
that it still pays about half a point of coherence for not having memorised a template.
Doubling the training time on the merged data (pytte-1m 16 min, pytte-3m 24 min) did not bring
coherence back (6.9 → 7.0 and 7.8 → 7.9), so the loss is a property of the data mix, not of
under-training.
A third mix, all of v1.1 plus only the one-child v2 stories (85 764 stories, `sv3`), keeps
the grammar gain and recovers most of the coherence, first on 1M and 3M and then across the whole
family on 13 November. **The `sv3` family is the release family**; `sv` and `sv2` are kept as
comparisons (template vs variety vs mix is itself a result).
Full analysis and the lab log: `FÖRSLAG-OCH-PLAN.md`, sections 13–17.

### Chat models (instruction tuning, 13 November 2026)

Each `sv3` model was then fine-tuned on request → story conversations so that it can be asked
for a story instead of continuing one. No new text was generated: the requests are built from
the recipe that produced each training story (name, object, adjective, verb, place, plural,
tense, first person, feature, ending), so the data costs nothing and the right answer is known.
83 474 conversations, 15.2M tokens: 52% single requests, 34% half a story plus "what happens
next?", 6% a new story about the same child, 5% "another one!", 2% small talk (greetings,
"who are you?", "what is 7 times 8?") with fixed short replies. Plain-text format,
`Du: … / Pytte: …`, every reply ends with the end-of-text token; the loss is taken on the
replies only. Fine-tuning starts from the `sv3` checkpoint with 2.5× the pretraining budget,
lr 3e-4, cosine to 2e-5. Instruction following is measured on 100 held-out requests with at
least one ingredient (did the story use the requested name, object, adjective, verb, place),
follow-ups on 30 half stories (new text, same child), small talk on 24 lines (short reply,
offers a story), and story quality with the same minimal pairs and judge as above.

| Model | Min | Epochs | Ingredients followed | All | Name | Object | Adjective | Verb | Place | Follow-up keeps child | Small talk short | Minimal pairs | Judge gram. / coh. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pytte-0.3m-chat | 12.5 | 15 | 51% | 31% | 90% | 29% | 8% | 24% | 67% | 87% | 25% | 81.5% | 8.8 / 6.0 |
| pytte-1m-chat | 20 | 14 | 70% | 54% | 100% | 47% | 35% | 53% | 92% | 97% | 83% | 93.0% | 9.7 / 8.3 |
| pytte-2m-chat | 25 | 10 | 74% | 60% | 100% | 66% | 19% | 41% | 92% | 100% | 67% | 86.5% | 9.7 / 8.1 |
| pytte-3m-chat | 25 of 30 | 8 | 76% | 57% | 100% | 56% | 35% | 71% | 100% | 100% | 67% | 91.5% | 9.7 / 8.7 |
| pytte-7m-chat | 50 | 10 | **79%** | **66%** | 99% | **74%** | **42%** | 53% | 92% | 100% | 92% | 91.5% | 9.6 / **9.3** |
| pytte-15m-chat | 75 | 9 | 72% | 56% | 100% | 50% | **46%** | 41% | 96% | 97% | 96% | 88.5% | 9.7 / 8.9 |

Three things stand out. Instruction following grows with capacity at equal epochs (7M vs 2M,
both 10 epochs: 79 vs 74%, object 74 vs 66%, adjective 42 vs 19%), and the 0.3M model cannot
do it: it keeps the name but not an object it has to bring in mid-story. The order of
difficulty is the same at every size: the name and the place, which open the story, come
first; the object and the verb next; the adjective, which must land before a noun the model
chose itself, last. And the story does not suffer: judge grammar 9.6–9.7 at every size (base
9.4–9.6), coherence up from 3M (a given child and place make the story easier to hold
together), language errors 0.2–0.3 per story, minimal pairs within the test's noise of the
base. The 3M run was cut at 25 of 30 minutes (the learning rate was already at 4e-5). The 15M
model overfits the chat data: its validation loss bottoms out after 3 epochs (24 minutes) and
rises for the remaining 51, so the saved best-validation checkpoint is the early one, and it
follows 72% of the ingredients against 7M's 79%. Fifteen million parameters memorise 12.7M
loss tokens in three epochs; seven million do not in ten. Tripling the variety at no cost
(three request passes per story, 45M tokens) was tried on 14 November: 15M gained 3 points
(72 → 75%) and overfits later but still overfits, while 7M lost 7 points (79 → 72%) because
it now saw each story–request pair three times instead of ten. At a fixed time budget it is
repetition, not variety, that teaches a small model to carry a word from the request into
the story, and validation loss does not measure it (the retrained models have the lowest loss
and the weakest following). The one-pass models are the chat family; the retrained ones are
kept as a comparison. Section 18 of the lab log has the pilot, the two sweeps and both tables.

## Running without Python

`pytte/export.py` writes a model and its tokenizer into one file (int8 weights per row, f32
norms, the token bytes and the BPE merge rules), and `run.c` runs it: byte-level BPE with GPT-2's
pre-tokenization, the full Llama-style forward pass with a KV cache, sampling, and the chat
format above. One file, no dependencies, `cc -O3 -o run run.c -lm`.

| Model | .bin | tok/s, one M2 Pro core | Minimal pairs through run.c (Python f32) |
|---|---:|---:|---:|
| pytte-0.3m | 0.38 MB | 38 000 | 81.5% (82.0%) |
| pytte-1m | 1.44 MB | 15 500 | 90.5% (91.5%) |
| pytte-2m | 2.37 MB | 10 100 | 89.5% (88.5%) |
| pytte-3m | 3.59 MB | 6 800 | 94.0% (94.0%) |
| pytte-7m | 7.33 MB | 3 900 | 94.5% (94.5%) |
| pytte-15m | 15.97 MB | 1 900 | 92.5% (93.0%) |

The C implementation is checked against Python three ways (`scripts/check_run.py`,
`scripts/pairs_run.py`): identical token ids on 300 validation stories and edge cases, identical
greedy continuations from an f32 export for 0.3M, 7M and 15M, and the grammar test run through
`run.c -score`, which reproduces Python's 91.5% exactly for the f32 file. With int8 weights and
int8 activations (one scale per vector, int32 accumulation; the float loop did not vectorise and
was 14–19× slower) every model stays within one point of its f32 grammar score, and the 7M chat
model's instruction following is unchanged within noise. Lab log section 19.

## Web demo

`web/` is the same thing in the browser: `pytte.js` loads a `.bin` and runs it in plain
JavaScript (no WebAssembly, no WebGPU, no server), `worker.js` keeps it off the UI thread, and
`index.html` has three panels: a size slider that regenerates the same story with the same seed
at every size, a chat with the instruction-tuned models, and a view of the tokens and the
next-token probabilities. With the same seed the browser produces byte-for-byte the same story
as `run.c`. Speed in Chromium on an M2 Pro: 0.3M 640 tokens/s, 2M 220, 7M 43, 15M 55. Serve the
folder statically (`python3 -m http.server 8765 --directory web`); the model files go in
`web/models/`.

## Data

**Download:** the three sets described below (v1.1, v2 and mix1, each with train/val/test
splits and the recipe behind every story; JSON Lines, 49 MB zipped) are in the GitHub release
[`sagosvenska-1.1`](https://github.com/dragon6sic6/pytte/releases/tag/sagosvenska-1.1), under CC BY 4.0.

**SagoSvenska** (v1.1, 2026-11-09): 60 426 short Swedish stories for 3-4 year olds,
6.7M words, generated locally in 54 hours with Gemma 4 12B (Apache-2.0) using the
TinyStories recipe (random name, place, noun, verb, adjective and story feature per prompt),
then filtered, deduplicated (MinHash, zero near-duplicates) and split deterministically
(59 224 / 599 / 603). Every story carries its prompt ingredients. 8.1M tokens with the
4096 vocabulary, 68x smaller than TinyStories, and template-heavy (generated at temperature
0.6). **SagoSvenska v2** (2026-11-12): 60 223 further stories (59 045 / 611 / 567), 7.0M words,
generated in 40 hours with Gemma 4 26B A4B at temperature 0.85 and a varied prompt recipe
(14 opening types, both tenses, 1-3 characters, plural objects, first person, 29 features,
11 ending types, 60-220 words). Against v1: 13 529 unique words instead of 11 619, cross-story
4-gram reuse 37 % instead of 51 %, and the most common first sentence covers 2 % of stories
instead of 16 % (FÖRSLAG section 16). The release models are trained on **mix1**:
all of v1.1 plus the 26 540 one-child v2 stories (85 764 training stories, 9.6M words); the
`sv2` rows above use the full merge (118 269 stories, 13.6M words). English twin: TinyStories V2
(553M tokens), plus a size-matched random subset of it (40 027 stories, 8.1M tokens) used as
the control family `en8m`.

## Honesty notes

The code in this repository was written with Claude Code, directed by a human who made the
decisions, read the stories and checked the numbers. The report will say exactly what the
AI did and what the human did. Quality claims use random samples, never hand-picked ones,
and the evaluation prompts are published.

Known mistake, fixed: the 4-bit Gemma 4 12B writer sometimes emitted its own end-of-image
special token mid-word, and the v1.0 cleaner stripped only its last character, so 12% of the
training stories contained the string `<image|` and 5–14% of the Swedish models' samples
contained it too. The generator now blocks every special token at sampling time, the cleaner
repairs the word, the data was re-cleaned (8 835 stories repaired, v1.1) and all six Swedish
models were retrained. Bits/byte, novelty and judge scores changed by less than the noise;
the grammar test moved by up to 10 points, which is now reported as the test's noise floor.
The v1.0 numbers are kept in the lab log (section 15).

Which data mix ships was settled by experiment, not taste. v1.1 alone gives the most coherent
stories but a model that has memorised its template (1.07 bits/byte on unseen-style text);
v1.1 + v2 gives much better grammar and generalisation but loses 1–2 points of coherence,
and doubling the training time did not bring it back. v1.1 plus the one-child v2 stories keeps
the grammar and most of the coherence (all of it from 7M), and is the release mix (`sv3`).
All three families are kept and reported; every released model states which data it was
trained on.

## License

Code and models: Apache-2.0 (see [`LICENSE`](LICENSE)). SagoSvenska, the training data: CC BY 4.0.
