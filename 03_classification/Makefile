PYTHON ?= python

RESULTS_A := results_A
DATA_CACHE_A := .data_cache_A
RESULTS_B := results_B
DATA_CACHE_B := .data_cache_B
RESULTS_C := results_C
DATA_CACHE_C := .data_cache_C
RESULTS_D := results_D
DATA_CACHE_D := .data_cache_D
RESULTS_QUICK := results_quick
RESULTS_MULTISITE := results_multisite

.DEFAULT_GOAL := help
.PHONY: help quick benchmark-A benchmark-B benchmark-C benchmark-D benchmark-all \
        analyze-multisite figures clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  make %-20s %s\n", $$1, $$2}'

quick:  ## Smoke-test the setup in a few minutes (DRIAMS-A, 1 classifier, 1 pair, reduced epochs)
	@$(PYTHON) run_benchmark.py --quick --sites DRIAMS-A --no-warping \
		--classifiers mlp_attn --species "Staphylococcus aureus" --n-folds 2 \
		--data-cache $(DATA_CACHE_A) \
		--results-dir $(RESULTS_QUICK)

benchmark-A:  ## DRIAMS-A: 5 classifiers x 3 pairs x (3 repeats x 5 folds)
	@$(PYTHON) run_benchmark.py --sites DRIAMS-A --no-warping \
		--data-cache $(DATA_CACHE_A) \
		--results-dir $(RESULTS_A)

benchmark-B:  ## DRIAMS-B: same protocol as benchmark-A
	@$(PYTHON) run_benchmark.py --sites DRIAMS-B --no-warping \
		--data-cache $(DATA_CACHE_B) \
		--results-dir $(RESULTS_B)

benchmark-C:  ## DRIAMS-C: same protocol as benchmark-A
	@$(PYTHON) run_benchmark.py --sites DRIAMS-C --no-warping \
		--data-cache $(DATA_CACHE_C) \
		--results-dir $(RESULTS_C)

benchmark-D:  ## DRIAMS-D: same protocol as benchmark-A
	@$(PYTHON) run_benchmark.py --sites DRIAMS-D --no-warping \
		--data-cache $(DATA_CACHE_D) \
		--results-dir $(RESULTS_D)

benchmark-all: benchmark-A benchmark-B benchmark-C benchmark-D  ## Run the four site benchmarks back-to-back (resumes cached cells)

analyze-multisite:  ## Aggregate results_{A,B,C,D}/benchmark.csv into results_multisite/
	@$(PYTHON) analyze_multisite.py

figures: analyze-multisite  ## Render Supplementary Figure S1 under results_multisite/figures/
	@$(PYTHON) make_paper_figures.py

clean:  ## Remove per-site results, spectra caches and aggregated tables
	@rm -rf $(RESULTS_A) $(DATA_CACHE_A) $(RESULTS_B) $(DATA_CACHE_B) \
		$(RESULTS_C) $(DATA_CACHE_C) $(RESULTS_D) $(DATA_CACHE_D) \
		$(RESULTS_QUICK) $(RESULTS_MULTISITE)
