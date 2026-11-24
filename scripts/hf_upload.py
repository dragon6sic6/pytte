"""Upload the models and the dataset to Hugging Face as private repos (made public at launch).

    hf auth login                                   # once, with a write token
    python scripts/hf_upload.py --models            # <user>/pytte-<size>[-chat] from <root>/hf/
    python scripts/hf_upload.py --dataset           # <user>/sagosvenska from the clean story sets
    python scripts/hf_upload.py --models --dataset --public   # at launch

The namespace defaults to the logged-in user; pass --org to use an organisation.
"""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path("/Volumes/Modelltest/pytte")
SIZES = ["0.3m", "1m", "2m", "3m", "7m", "15m"]

DATASET_CARD = """---
language:
- sv
license: cc-by-4.0
task_categories:
- text-generation
tags:
- tinystories
- swedish
- synthetic
- children-stories
size_categories:
- 100K<n<1M
configs:
- config_name: mix1
  data_files:
  - split: train
    path: mix1/train.jsonl
  - split: validation
    path: mix1/val.jsonl
  - split: test
    path: mix1/test.jsonl
---

# SagoSvenska

Swedish children's stories in the spirit of TinyStories, generated locally on a MacBook for the
[Pytte](https://github.com/dragon6sic6/pytte) project: the smallest open, documented language models
that write coherent Swedish. Every story was written by Gemma 4 (Apache-2.0) from a recipe: a child's
name, a place, an object, a verb, an adjective, one or two story features and an ending type, in a
vocabulary a small child knows. No story was written by a person and none comes from a book.

## Sets

| Folder | Stories (train / val / test) | Generated with | Recipe |
|---|---:|---|---|
| `v1.1/` | 59 224 / 599 / 603 | Gemma 4 12B-it (4-bit), temperature 0.6 | v1: one child, fixed opening, 60–200 words |
| `v2/` | 59 045 / 611 / 567 | Gemma 4 26B-A4B (4-bit), temperature 0.85 | v2: 14 opening types, past or present tense, 1–3 characters, 40% plural objects, 12% first person, 29 features, 11 endings |
| `mix1/` | 85 764 / 853 / 865 | v1.1 + the one-child v2 stories | the release mix (what the Pytte models are trained on) |

`mix1` is the default configuration. Each line is a JSON object with the story under `text` and
its full recipe (`name`, `place`, `noun`, `verb`, `adj`, `features`, `ending`, `tense`, …), so
instruction data can be built from it without generating anything new.

## Why three sets

Training on v1.1 alone gives the most coherent stories but a model that has memorised the
template; v1.1 + v2 gives better grammar and generalisation but less coherent two- and
three-child tales; v1.1 plus the one-child v2 stories keeps the grammar and most of the
coherence. The comparison is a result in itself and is reported in the Pytte lab log.

## Cleaning

Markdown and quotes normalised; stories that were too short, in English or another language, or
flagged as unsafe were dropped; exact duplicates removed; a special token (`<image|>`) leaked by
the 4-bit generator was repaired (documented in the lab log; v1.0 had it in 12% of stories).

## License

CC BY 4.0. Please cite the Pytte repository.
"""


def upload_models(api, ns, public):
    for size in SIZES:
        for chat in (False, True):
            name = f"pytte-{size}{'-chat' if chat else ''}"
            repo = f"{ns}/{name}"
            api.create_repo(repo, repo_type="model", private=not public, exist_ok=True)
            api.upload_folder(folder_path=str(ROOT / "hf" / name), repo_id=repo, repo_type="model",
                              commit_message=f"{name}: weights, tokenizer, int8 bin, model card")
            print(f"uploaded {repo}")


def upload_dataset(api, ns, public):
    repo = f"{ns}/sagosvenska"
    api.create_repo(repo, repo_type="dataset", private=not public, exist_ok=True)
    tmp = ROOT / "hf" / "sagosvenska"
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "README.md").write_text(DATASET_CARD)
    api.upload_file(path_or_fileobj=str(tmp / "README.md"), path_in_repo="README.md", repo_id=repo, repo_type="dataset")
    for folder, src in (("v1.1", "sagosvenska"), ("v2", "sagosvenska-v2"), ("mix1", "sagosvenska-mix1")):
        api.upload_folder(folder_path=str(ROOT / "data" / src / "clean"), path_in_repo=folder, repo_id=repo,
                          repo_type="dataset", allow_patterns=["*.jsonl", "stats.json"],
                          commit_message=f"SagoSvenska {folder}")
        print(f"uploaded {repo}/{folder}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", action="store_true")
    p.add_argument("--dataset", action="store_true")
    p.add_argument("--public", action="store_true", help="create as public (launch day)")
    p.add_argument("--org", default=None, help="namespace; defaults to the logged-in user")
    args = p.parse_args()
    api = HfApi()
    ns = args.org or api.whoami()["name"]
    print(f"namespace: {ns} ({'public' if args.public else 'private'})")
    if args.models:
        upload_models(api, ns, args.public)
    if args.dataset:
        upload_dataset(api, ns, args.public)
    if not (args.models or args.dataset):
        print("nothing to do: pass --models and/or --dataset")


if __name__ == "__main__":
    main()
