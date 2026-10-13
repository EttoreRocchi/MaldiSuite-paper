# MaldiSuite-paper

Analysis code for "MaldiSuite: a Python ecosystem for MALDI-TOF spectral processing and analysis in antimicrobial resistance research".

The three packages described in the paper live in their own repositories ([MaldiAMRKit](https://github.com/EttoreRocchi/MaldiAMRKit), [MaldiBatchKit](https://github.com/EttoreRocchi/MaldiBatchKit), [MaldiDeepKit](https://github.com/EttoreRocchi/MaldiDeepKit), bundled by [MaldiSuite](https://github.com/EttoreRocchi/MaldiSuite)). This repository contains what is needed to reproduce the results of the paper: the scripts behind its figures and tables, and the aggregated results they are computed from.

## What produces what

| Item in the paper | Script | Reads |
|---|---|---|
| Figure 1b, preprocessing of one spectrum | `01_preprocessing/make_figure1_panelB.py` | one raw DRIAMS-A spectrum |
| Figure 1c, t-SNE before and after harmonisation | `figures/make_figure1.py` | t-SNE embeddings from `02_batch_correction` |
| Figure 1d, AUROC heat maps | `figures/make_figure1.py` | `results/classification/summary.csv` |
| Figure S1, per-architecture benchmark | `03_classification/make_paper_figures.py` | `results/classification/summary.csv` |
| Figure S2, t-SNE and UMAP per corrector | `02_batch_correction/scripts/05_make_figures.py` | t-SNE and UMAP embeddings from `02_batch_correction` |
| Figure S3, per-species t-SNE per corrector | `figures/make_supp_figures.py` | t-SNE embeddings from `02_batch_correction` |
| Figure S4, batch-effect metrics per corrector | `figures/make_supp_figures.py` | `results/batch_correction/supp_table_s3.csv` |
| Figure S5, bin-width sensitivity | `figures/make_supp_figures.py` | `results/leakage_and_bin_width/bin_width_sensitivity.csv` |
| Tables S4-S7, batch-effect metrics | `02_batch_correction/scripts/03_compute_metrics.py` | corrected DRIAMS-A/B/C/D spectra from `02_batch_correction` |
| Table S9, leakage experiment | `figures/make_supp_tables.py` | `results/leakage_and_bin_width/leakage_*.csv` |
| Table S10, bin-width sensitivity | `figures/make_supp_tables.py` | `results/leakage_and_bin_width/bin_width_sensitivity.csv` |
| Note S3, leakage experiment | `04_leakage_and_bin_width/leakage_ladder.py`, `leakage_curves.py` | DRIAMS-A raw spectra |
| Note S4, bin-width sensitivity | `04_leakage_and_bin_width/bin_width_sensitivity.py` | DRIAMS-A raw spectra |

The values of Tables S4-S7 are the columns of `results/batch_correction/supp_table_s3.csv`, rounded to three decimals. Tables S9 and S10 are computed from the per-fold results and written as CSV. The numbers quoted in the text come from the same files.

The scripts in `figures/` write under `output/` (override with the `OUTPUT_DIR` environment variable); Figure 1b is written next to its script, Figure S1 under `03_classification/results_multisite/figures/` and Figure S2 under `02_batch_correction/figures/`.

## Layout

```
01_preprocessing/           one spectrum, raw against preprocessed and binned
02_batch_correction/        eleven correctors on DRIAMS-A/B/C/D (scripts/, Makefile)
03_classification/          five classifier configurations on four sites (Makefile)
04_leakage_and_bin_width/   leakage protocols and bin-width sweep (run_all.sh, run_curves.sh)
figures/                    figures and tables of the paper built from results/
results/                    aggregated results the figures and tables are built from
requirements/               frozen environments
Makefile                    entry point for every step (make help)
```

## Data

The experiments read the public DRIAMS dataset (Weis *et al.*, *Nature Medicine* 2022, [doi:10.5061/dryad.bzkh1899q](https://doi.org/10.5061/dryad.bzkh1899q)), which is not redistributed here. Every script that reads spectra requires the `DRIAMS_ROOT` environment variable, pointing at a directory that holds `DRIAMS-A/`, `DRIAMS-B/`, `DRIAMS-C/` and `DRIAMS-D/`:

```bash
export DRIAMS_ROOT=/path/to/MALDI-data
```

`results/` contains only aggregated outputs: per-fold metrics, per-corrector batch-effect metrics and cohort summaries. Spectra and caches are not included; they are rebuilt by the scripts.

## Environment

The two lock files in `requirements/` record the environments actually used (Python 3.10, Linux), with the three MaldiSuite packages pinned to the releases of the paper: MaldiAMRKit 0.20.0, MaldiBatchKit 0.2.0 and MaldiDeepKit 0.3.0. They are not compatible with each other, since they pin different versions of `numpy` and `torch`, so each needs its own virtual environment:

```bash
make venvs
```

This creates `.venv-classification` from `requirements/classification-leakage.txt` (used by `01_preprocessing`, `03_classification`, `04_leakage_and_bin_width` and `figures/`) and `.venv-batch` from `requirements/batch-correction.txt` (used by `02_batch_correction`). Pass `PYTHON=/path/to/python3.10` if `python3.10` is not on your `PATH`. `torch` is pinned without the local CUDA suffix and the CUDA libraries it depends on are not pinned, so pip installs the build that matches your platform; a GPU is recommended for `03_classification` and `04_leakage_and_bin_width` and unnecessary elsewhere. Installing the meta-package (`pip install maldisuite`) pulls the same three packages, but resolves their dependencies freely instead of to the locked versions.

## Running

Every step is a target of the `Makefile` at the repository root (`make help` lists them) and runs in the right environment:

| Target                  | Step                                                                 |
| ----------------------- | -------------------------------------------------------------------- |
| `make preprocessing`    | Figure 1b, one DRIAMS-A spectrum                                     |
| `make batch-correction` | Eleven correctors on DRIAMS-A/B/C/D (`02_batch_correction`)          |
| `make classification`   | Five classifiers on DRIAMS-A/B/C/D (`03_classification`)             |
| `make leakage`          | Leakage protocols and bin-width sweep (`04_leakage_and_bin_width`)   |
| `make figures`          | Figures 1c-d, S3-S5 and Tables S9-S10 under `output/`               |
| `make all`              | Every step above, in order                                           |

`make figures` reads the aggregated results in `results/` and the t-SNE embeddings written by `make batch-correction`, so run the latter first. Every experiment caches its intermediate results to disk, so an interrupted step resumes where it stopped when the same target is run again. The `Makefile` in `02_batch_correction` and `03_classification` exposes finer-grained targets, documented in the README of each directory.

Every randomised step uses a fixed seed (`SEED = 42`); subsample, split and initialisation seeds all derive from it, so reruns reproduce the published numbers.

## Citation

Available soon.

## Licence

MIT, as the packages of the suite. See [LICENSE](LICENSE).
