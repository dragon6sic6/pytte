# Pytte on mathiaskarlsson.se: handoff brief

*Status: done. The page was built from this brief on 2026-11-25 on the site's `pytte` branch. The brief is kept as a record; the repository has since been made public, so the notes below about a private repository are out of date, and the training data now has a download: the GitHub release `sagosvenska-1.1` (49 MB zipped).*

Written 2026-11-25 in the smallestllm session ("Världens minsta LLM-projekt") for the session
that owns the site (`mathiaskarlsson2026final`). Mathias decided that Pytte is published as a
**hobby project on his own site**, at `/pytte` (and `/en/pytte`), with a card on `/sidoprojekt`,
instead of as a model release on Hugging Face. He wants to be open that he does not know how
good it is. **Nothing is to be published until he has seen the page.**

## What the page is

One page, both languages, in the site's own design and structure (header, footer, language
switch, content files, OG image, sitemap), with this order:

1. **Hero**: eyebrow "Hobbyprojekt", the name, one-sentence lead, the note that everything runs
   in the browser. The mark is `brand/pytte-mark.svg`.
2. **The demo**, immediately, three tabs: the size slider that regenerates one story with the
   same seed at each of six model sizes, the chat, and the token view. This is the point of the
   page; it should be usable without scrolling far on a phone.
3. **Why I did this** (three short paragraphs, Mathias's voice).
4. **Pytte in numbers** (five facts).
5. **What I learned** (five items).
6. **What I do not know** (six items). This section matters as much as the demo: it is the
   honest framing he asked for. Do not soften or shorten it.
7. **Download**: the report (PDF), the twelve model files as a table (base and chat per size),
   how to run it with `run.c`, the code on GitHub, the data, the licences.
8. Back link to `/sidoprojekt`.

All copy for both languages is in `site/content.pytte.ts` (plain data, adapt it to your schema).
Every number in it comes from the report; please do not change numbers or add claims. If
something reads badly in the site's voice, rewrite the sentence but keep the fact.

## Files (absolute paths)

| What | Where | Notes |
|---|---|---|
| Copy, sv + en | `/Users/mathiaskarlsson/Projects/smallestllm/site/content.pytte.ts` | includes the card for `/sidoprojekt` |
| Runtime | `/Users/mathiaskarlsson/Projects/smallestllm/web/pytte.js` | ES module, no dependencies: loads a `.bin`, tokenizer, forward pass, sampling |
| Worker | `/Users/mathiaskarlsson/Projects/smallestllm/web/worker.js` | Web Worker protocol documented at the top of the file; fetches `models/<file>` relative to itself |
| Reference UI | `/Users/mathiaskarlsson/Projects/smallestllm/web/{index.html,app.js,style.css}` | a working plain-JS version of the three tabs; use it as the behaviour spec, not as the design |
| Models | `/Users/mathiaskarlsson/Projects/smallestllm/web/models/*.bin` | 12 files, 59 MB in total, 0.38 to 15.97 MB each |
| Report | `/Users/mathiaskarlsson/Projects/smallestllm/paper/pytte.pdf` | 19 pages, about 1.4 MB |
| Logo | `/Users/mathiaskarlsson/Projects/smallestllm/brand/pytte-mark.svg`, `pytte-card.png` | colours #F6F1E7, #1E2A4A, #F2B632 |
| C runtime | `/Users/mathiaskarlsson/Projects/smallestllm/run.c` | linked from the downloads section via GitHub once the repo is public |

To see the reference demo: `python3 -m http.server 8765 --directory /Users/mathiaskarlsson/Projects/smallestllm/web`.

## How the demo works (for a React port)

- `new Worker(".../worker.js", { type: "module" })`. Messages in: `{type:"generate", id, file,
  prompt | turns, seed, temp, topP, maxNew, stop}`, `{type:"probs", id, file, text}`,
  `{type:"cancel"}`. Messages out: `progress` (download), `loaded`, `token` (streamed text),
  `done` (`tokens`, `secs`), `cancelled`, `error`, `probs` (`tokens`, `top`).
- Slider: file `pytte-<size>.bin`, `prompt` = the text field, same `seed` for every size, temp
  0.8, topP 0.95, maxNew 250. Changing the slider cancels and regenerates.
- Chat: file `pytte-<size>-chat.bin` (default 7m), `turns` = alternating user and assistant
  strings, `stop: "\nDu:"`, maxNew 300. The chat format is built inside the worker.
- Token view: `probs` with the base model, debounced while typing.
- Models are cached in the worker after the first load. Nothing else is stored or sent.
- It is a client-only feature: no SSR for the demo component.

## Things to decide on your side

- **Where the `.bin` files are served from.** Simplest is `public/pytte/models/` with long
  immutable cache headers. They are 59 MB of static assets and a visitor who drags the slider
  through all sizes downloads 31 MB, so check what that means for the hosting plan's bandwidth
  before launch day. A CDN in front of the public GitHub repo is a possible alternative once
  the repo is public; verify file-size limits and CORS before relying on it.
- **Download links** for the model table can point at the same files.
- **GitHub link**: `https://github.com/dragon6sic6/pytte` is private today. Show the link only
  when Mathias has made it public, or mark it "kommer snart".
- **The data section** has no download yet (211 MB); it will be a GitHub release. Until then
  show the text without a button.
- **OG image and the `/sidoprojekt` card** in the site's style; `brand/pytte-card.png` exists as
  raw material.

## Verification before showing Mathias

- Same seed gives the same story as the reference demo (it is deterministic): prompt
  "Det var en gång en liten älg som", size 2M, seed 624419 must begin
  "heter Alfred. Han går ut i den stora trädgården."
- The chat answers "Hej!" with one short line that offers a story, on the 7M model.
- Works on a phone-width viewport and in both themes.
- Both languages are complete; the demo's inputs stay Swedish in the English version (the
  model only knows Swedish) and the English copy says so.

## Not in scope

Hugging Face, Zenodo, press. No analytics or tracking added for this page beyond what the site
already has.
