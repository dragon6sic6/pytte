# Pytte convenience targets. Paths default to the external disk layout used in development.
ROOT ?= /Volumes/Modelltest/pytte
PY   ?= $(ROOT)/.venv/bin/python
export HF_HOME ?= $(ROOT)/hf-cache

LANG_   ?= en
CONFIG  ?= pytte-1m
MINUTES ?= 5
TOK     = $(ROOT)/data/tok-$(LANG_)-4096
TOKENS  = $(ROOT)/data/tokens-$(LANG_)-4096
CKPT    = $(ROOT)/checkpoints/$(CONFIG)-$(LANG_)

.PHONY: bench train sample eval gen clean-stories family run export

family:        ## print the model family table
	$(PY) -m pytte.model

bench:         ## MLX vs PyTorch throughput
	bench/run_bench.sh

train:         ## make train CONFIG=pytte-1m LANG_=en MINUTES=5
	$(PY) -m pytte.train --config $(CONFIG) --data $(TOKENS) --tok $(TOK) --out $(CKPT) --minutes $(MINUTES)

sample:        ## make sample CONFIG=pytte-1m LANG_=en PROMPT="Once upon a time"
	$(PY) -m pytte.sample --model $(CKPT) --tok $(TOK) --prompt "$(PROMPT)" --n 3

eval:          ## perplexity, minimal pairs, generation stats
	$(PY) -m pytte.eval --model $(CKPT) --tok $(TOK) --data $(TOKENS) --pairs eval/minimal_pairs_$(LANG_).json

gen:           ## generate Swedish stories (long-running, owns the GPU)
	cd $(ROOT)/data/sagosvenska && nohup caffeinate -is $(PY) $(CURDIR)/data/gen_stories.py --target 60000 >> gen.log 2>&1 &

gen-stop:      ## stop generation gracefully after the current batch
	touch $(ROOT)/data/sagosvenska/STOP

clean-stories: ## filter, dedup and split raw stories
	$(PY) data/clean_stories.py --raw $(ROOT)/data/sagosvenska/raw --out $(ROOT)/data/sagosvenska/clean

# ---- export and run.c ----
run:           ## compile run.c
	cc -O3 -o run run.c -lm

export:        ## export CKPT + TOK to one int8 .bin (make export CONFIG=pytte-7m LANG_=sv3-chat)
	$(PY) -m pytte.export --model $(CKPT) --tok $(TOK) --out $(ROOT)/bin/$(CONFIG)-$(LANG_).bin
