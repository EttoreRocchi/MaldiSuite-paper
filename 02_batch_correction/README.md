# Batch correction benchmark

The 11 MaldiBatchKit correctors on the combined DRIAMS-A/B/C/D data, as reported in the MaldiSuite paper. Spectra are preprocessed with the MaldiAMRKit default pipeline (clip negatives, square root, Savitzky-Golay smoothing, SNIP baseline, trimming to 2000-20000 Da, TIC normalisation) and binned at 3 Da. The correctors are:

- ComBat in three variants: Johnson, Fortin, and Chen (CovBat)
- Limma `removeBatchEffect`
- Harmony, on PCA-reduced data
- median centering, z-score per batch and reference scaling
- batch-aware warping
- quality-weighted ComBat, with SNR weights
- species-aware ComBat (Fortin preset)

## Experiments

| Key                    | Pairs                                        | Sites | Species covariate |
|------------------------|----------------------------------------------|-------|-------------------|
| `exp1_saureus`         | S. aureus / oxacillin                        | A,B,C,D | no              |
| `exp1_ecoli`           | E. coli / ceftriaxone                        | A,B,C,D | no              |
| `exp1_kpneumoniae`     | K. pneumoniae / ceftriaxone                  | A,B,C,D | no              |
| `exp3_multi_species`   | E. coli + K. pneumoniae (both ceftriaxone)   | A,B,C,D | yes (2 species) |

Experiment 3 provides Figure 1c and Supplementary Figure S2; experiments 1a-c provide the per-species panels of Supplementary Figure S3.

## Outputs

- `results/metrics/supp_table_s3.csv`: silhouette, Davies-Bouldin, kBET, LISI, variance ratio, k-NN preservation, distance correlation and species preservation for each experiment, method and stage. These are the values of Supplementary Tables S4-S7 and the input of Figure S4; the file is shipped as `results/batch_correction/supp_table_s3.csv` at the repository root.
- `results/embeddings/`: t-SNE and UMAP coordinates for every experiment and corrector, used for Figure 1c and Supplementary Figures S2 and S3.
- `figures/figure_s2a_tsne_site.{png,pdf}` and `figures/figure_s2b_umap_site.{png,pdf}`: Supplementary Figure S2. Figure 1c and Supplementary Figure S3 are drawn from the same embeddings by `figures/make_figure1.py` and `figures/make_supp_figures.py` at the repository root.

## Data path

DRIAMS data is read from the path given by the `DRIAMS_ROOT` environment variable, which must be set and contain `DRIAMS-{A,B,C,D}`.

## Run order

From this directory, the full pipeline (it uses the caches) is:

```bash
make all
```

Add `REFRESH=1` to recompute every step even when caches are present:

```bash
make REFRESH=1 all
```

The steps can also be run one at a time, in this order:

| Target         | Step                                          |
| -------------- | --------------------------------------------- |
| `make load`    | 1. preprocess + bin + SNR per pair            |
| `make correct` | 2. fit every corrector on every experiment    |
| `make metrics` | 3. batch-effect metrics (Tables S4-S7)        |
| `make embed`   | 4. t-SNE / UMAP coords                        |
| `make figures` | 5. Supplementary Figure S2                    |

The full pipeline takes several hours (Harmony + ComBat-Chen + t-SNE x 4 experiments x 11 methods). Every step caches to disk, so reruns without `REFRESH=1` skip already-completed cells.

## Layout

| Path                   | Contents                                              |
| ---------------------- | ----------------------------------------------------- |
| `scripts/_shared.py`   | paths, experiment definitions, DRIAMS loaders         |
| `scripts/01_load_data.py` | step 1                                             |
| `scripts/02_run_corrections.py` | step 2                                       |
| `scripts/03_compute_metrics.py` | step 3                                       |
| `scripts/04_compute_embeddings.py` | step 4                                    |
| `scripts/05_make_figures.py` | step 5                                          |
| `cache/data/`          | cached DRIAMS pair matrices + meta                    |
| `cache/estimators/`    | joblib-pickled fitted correctors                      |
| `results/corrected/`   | per-experiment, per-method `X_corrected.npz`          |
| `results/embeddings/`  | tSNE / UMAP coords + subsample indices                |
| `results/metrics/`     | `supp_table_s3.csv`                                   |
| `figures/`             | PNG + PDF of Supplementary Figure S2                  |
| `logs/`                | one `.log` file per script run                        |

Everything below `cache/`, `results/`, `figures/` and `logs/` is produced by the runs and is not tracked in the repository; the aggregated tables used in the paper are shipped under `results/batch_correction/` at the repository root.

## Reproducibility

Every randomized step uses `_shared.SEED = 42`. The stratified subsample for embeddings is cached as `<exp_key>__indices.npy` so pre / post t-SNE plots show the same spectra.
