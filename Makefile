PYTHON ?= python3.10

VENV_CLS := .venv-classification
VENV_BATCH := .venv-batch
PY_CLS := $(CURDIR)/$(VENV_CLS)/bin/python
PY_BATCH := $(CURDIR)/$(VENV_BATCH)/bin/python

.DEFAULT_GOAL := help
.PHONY: help venvs preprocessing batch-correction classification leakage figures all

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  make %-18s %s\n", $$1, $$2}'

venvs: $(VENV_CLS)/.installed $(VENV_BATCH)/.installed  ## Create the two locked environments

$(VENV_CLS)/.installed: requirements/classification-leakage.txt
	$(PYTHON) -m venv $(VENV_CLS)
	$(VENV_CLS)/bin/pip install -r $<
	@touch $@

$(VENV_BATCH)/.installed: requirements/batch-correction.txt
	$(PYTHON) -m venv $(VENV_BATCH)
	$(VENV_BATCH)/bin/pip install -r $<
	@touch $@

preprocessing: $(VENV_CLS)/.installed  ## Figure 1b, one DRIAMS-A spectrum
	$(PY_CLS) 01_preprocessing/make_figure1_panelB.py

batch-correction: $(VENV_BATCH)/.installed  ## Eleven correctors on DRIAMS-A/B/C/D (02_batch_correction)
	$(MAKE) -C 02_batch_correction PYTHON=$(PY_BATCH) all

classification: $(VENV_CLS)/.installed  ## Five classifiers on DRIAMS-A/B/C/D (03_classification)
	$(MAKE) -C 03_classification PYTHON=$(PY_CLS) benchmark-all figures

leakage: $(VENV_CLS)/.installed  ## Leakage protocols and bin-width sweep (04_leakage_and_bin_width)
	cd 04_leakage_and_bin_width && PYTHON=$(PY_CLS) ./run_all.sh && PYTHON=$(PY_CLS) ./run_curves.sh

figures: $(VENV_CLS)/.installed  ## Figures 1c-d, S3-S5 and Tables S9-S10 under output/ (t-SNE panels need batch-correction first)
	$(PY_CLS) figures/make_figure1.py
	$(PY_CLS) figures/make_supp_figures.py
	$(PY_CLS) figures/make_supp_tables.py

all: preprocessing batch-correction classification leakage figures  ## Every step above, in order
