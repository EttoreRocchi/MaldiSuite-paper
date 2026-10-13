# Classification benchmark

Scripts for the classification experiment of the MaldiSuite paper: five MaldiDeepKit classifier configurations, three DRIAMS species-drug resistance-prediction tasks, run once per DRIAMS site (A, B, C, D). They produce `results/classification/` at the repository root, the input of Figure 1d and Supplementary Figure S1.

The five classifier configurations are:

| Key                | Classifier                                       |
| ------------------ | ------------------------------------------------ |
| `mlp_attn`         | `MaldiMLPClassifier` (attention on)              |
| `cnn`              | `MaldiCNNClassifier`                             |
| `resnet`           | `MaldiResNetClassifier`                          |
| `transformer`      | `MaldiTransformerClassifier` (1-D ViT)           |
| `ensemble_mlp_cnn` | `SpectralEnsemble` of `mlp_attn` and `cnn`       |

ResNet and the Transformer use a linear learning-rate warmup (see `build_classifier` in `_shared.py`); without it they diverged at `lr=1e-3` on this data.

The three species-drug pairs are:

- *Staphylococcus aureus* / oxacillin
- *Escherichia coli* / ceftriaxone
- *Klebsiella pneumoniae* / ceftriaxone

Every (pair, classifier) cell is evaluated with 3 repeats of stratified 5-fold cross-validation (15 fits). Repeat `r` splits with seed `SEED + 100 r` and fit `k` is initialised with seed `SEED + k`, with `SEED = 42`.

## Requirements

The loader reads `DRIAMS-A/`, `DRIAMS-B/`, `DRIAMS-C/` and `DRIAMS-D/` through `maldiamrkit.data.DRIAMSLayout`, under the path in the `DRIAMS_ROOT` environment variable, which must be set. MaldiDeepKit and MaldiAMRKit must be installed; `requirements/classification-leakage.txt` at the repository root pins the versions used for the paper. A GPU is strongly recommended: the smoke test runs on CPU, the full benchmark realistically needs CUDA.

## Running

All entry points are wired up through the `Makefile` (`make help` lists them):

| Target                  | What it does                                                                 |
| ----------------------- | ---------------------------------------------------------------------------- |
| `make quick`            | Smoke-test the setup in a few minutes: DRIAMS-A, 1 classifier, 1 pair         |
| `make benchmark-A`      | DRIAMS-A: 5 classifiers x 3 pairs x (3 repeats x 5 folds)                    |
| `make benchmark-B`      | Same protocol on DRIAMS-B                                                    |
| `make benchmark-C`      | Same protocol on DRIAMS-C                                                    |
| `make benchmark-D`      | Same protocol on DRIAMS-D                                                    |
| `make benchmark-all`    | The four site benchmarks back-to-back                                        |
| `make analyze-multisite`| Aggregate `results_{A,B,C,D}/benchmark.csv` into `results_multisite/`        |
| `make figures`          | Aggregate, then render Supplementary Figure S1 under `results_multisite/figures/` |
| `make clean`            | Remove per-site results, spectra caches and aggregated tables                |

To reproduce the results of the paper:

```bash
make benchmark-all
make figures
```

Each site runs without spectral warping, which corrects calibration drift between sites and has nothing to correct within one site, and writes to its own results and cache directories.

A `(pair, classifier)` cell whose `fold_metrics.csv` is already on disk with the expected number of folds is skipped, so an interrupted `make benchmark-all` resumes when run again. `run_benchmark.py --force` retrains every cell.

Supplementary Figure S1 can also be rendered directly from the results shipped with the repository:

```bash
python make_paper_figures.py --summary-csv ../results/classification/summary.csv
```

## Layout

| Path                   | Contents                                                           |
| ---------------------- | ------------------------------------------------------------------ |
| `_shared.py`           | DRIAMS loader, classifier factory, metrics, CV runner              |
| `run_benchmark.py`     | Main experiment: 3 pairs x 5 classifiers x (3 repeats x 5 folds)   |
| `analyze_multisite.py` | Concatenates the four per-site `benchmark.csv` and summarises them |
| `make_paper_figures.py`| Renders Supplementary Figure S1 from `summary.csv`                 |
| `Makefile`             | Entry points (`make help`)                                         |

`results_{A,B,C,D}/`, `results_quick/`, `.data_cache_{A,B,C,D}/` and `results_multisite/` are produced by the runs and are not tracked in the repository.

## Outputs

Each `make benchmark-<site>` writes, under `results_<site>/`:

```
benchmark.csv                              raw, one row per (pair, classifier, repeat, fold)
benchmark.log
<pair-slug>/<classifier>/fold_metrics.csv  per-cell fold metrics
```

`make figures` then writes, under `results_multisite/`:

```
benchmark_multisite.csv   the four per-site benchmark.csv concatenated
summary.csv               mean/std per (site, pair, classifier)
figures/                  figure_s1_benchmark.{pdf,png}
```

The copies used in the paper are shipped under `results/classification/` at the repository root.
