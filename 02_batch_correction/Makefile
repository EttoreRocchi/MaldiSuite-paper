PYTHON ?= python
SCRIPTS := scripts

REFRESH_FLAG := $(if $(REFRESH),--refresh,)

.DEFAULT_GOAL := all
.PHONY: help all load correct metrics embed figures clean clean-cache clean-results

help:  ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?##' $(MAKEFILE_LIST) | \
	  awk -F':.*?## ' '{printf "  make %-20s %s\n", $$1, $$2}'

all: figures   ## Run the full pipeline (depends on every previous step)

load:          ## Step 1: DRIAMS pair caches under cache/data/
	$(PYTHON) $(SCRIPTS)/01_load_data.py $(REFRESH_FLAG)

correct: load  ## Step 2: 11 corrections per experiment (results/corrected/)
	$(PYTHON) $(SCRIPTS)/02_run_corrections.py $(REFRESH_FLAG)

metrics: correct  ## Step 3: batch-effect metrics behind Tables S4-S7 and Figure S4 (results/metrics/)
	$(PYTHON) $(SCRIPTS)/03_compute_metrics.py

embed: correct  ## Step 4: t-SNE + UMAP coords behind Figures 1c, S2 and S3 (results/embeddings/)
	$(PYTHON) $(SCRIPTS)/04_compute_embeddings.py $(REFRESH_FLAG)

figures: metrics embed  ## Step 5: Supplementary Figure S2 (figures/)
	$(PYTHON) $(SCRIPTS)/05_make_figures.py

clean: clean-cache clean-results  ## Wipe every artefact (caches + results + figures)

clean-cache:    ## Drop cache/data and cache/estimators
	rm -rf cache/data/* cache/estimators/*

clean-results:  ## Drop results/* and figures/*
	rm -rf results/corrected/* results/embeddings/* results/metrics/* figures/* logs/*
