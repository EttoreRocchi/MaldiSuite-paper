"""Compute t-SNE and UMAP embeddings of the uncorrected and corrected matrices.

The matrices are too large to embed in full, so one subsample per
experiment, stratified by site and species, is drawn once and reused
for every method: the pre and post panels show the same spectra.

Every matrix is standardised and reduced to its own 50 principal
components before t-SNE and UMAP; a basis fitted on the uncorrected
data would misrepresent corrections that change the principal axes.

Outputs (per experiment) under ``results/embeddings/``::

    <exp_key>__indices.npy            subsample indices into meta
    <exp_key>__meta.csv               site / species / rs labels of those rows
    <exp_key>__pre__tsne.npy
    <exp_key>__pre__umap.npy
    <exp_key>__<method>__tsne.npy
    <exp_key>__<method>__umap.npy

Run::

    python scripts/04_compute_embeddings.py
    python scripts/04_compute_embeddings.py --experiment exp3_multi_species
    python scripts/04_compute_embeddings.py --refresh
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from _shared import (
    CORRECTED_DIR,
    EMBEDDINGS_DIR,
    EXPERIMENTS,
    LOGS_DIR,
    METHOD_DISPLAY,
    METHOD_KEYS,
    SEED,
    Experiment,
    load_experiment,
    setup_logging,
    stratified_subsample,
)


def _tsne(X_pca: np.ndarray, *, seed: int) -> np.ndarray:
    return TSNE(
        n_components=2,
        perplexity=30.0,
        learning_rate="auto",
        init="pca",
        random_state=seed,
        n_jobs=-1,
    ).fit_transform(X_pca)


def _umap(X_pca: np.ndarray, *, seed: int) -> np.ndarray:
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "umap-learn is required for the UMAP plots. Install with "
            "`pip install maldibatchkit[viz]` (or `pip install umap-learn`)."
        ) from exc
    return umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.3,
        metric="euclidean",
        random_state=seed,
    ).fit_transform(X_pca)


def _project(X: np.ndarray, *, n_components: int = 50, seed: int = SEED) -> np.ndarray:
    """Standardise, then project on the first 50 principal components.

    Standardising gives every m/z bin the same weight in the principal
    components, as in batch-correction benchmarks (Tran 2020, Luecken
    2022). Otherwise a few high-intensity peaks dominate, and methods that
    change the scale of the data (``ZScorePerBatch``,
    ``QualityWeightedComBat``) would look different for that reason alone.
    """
    n_components = int(min(n_components, X.shape[1] - 1, X.shape[0] - 1))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.astype(np.float32))
    pca = PCA(n_components=n_components, random_state=seed)
    return pca.fit_transform(X_scaled)


def _resolve_path(exp_key: str, method: str, kind: str) -> Path:
    """``method='pre'`` for the uncorrected baseline."""
    return EMBEDDINGS_DIR / f"{exp_key}__{method}__{kind}.npy"


def compute_for_experiment(
    exp: Experiment,
    *,
    target: int,
    refresh: bool,
    log,
    seed: int = SEED,
) -> None:
    indices_path = EMBEDDINGS_DIR / f"{exp.key}__indices.npy"
    meta_path = EMBEDDINGS_DIR / f"{exp.key}__meta.csv"

    X_df, meta = load_experiment(exp)
    X_pre = X_df.to_numpy(dtype=np.float32)

    if not refresh and indices_path.exists():
        sub_idx = np.load(indices_path)
    else:
        sub_idx = stratified_subsample(
            meta,
            target=target,
            stratify_cols=("site", "species_canonical")
            if exp.species_covariate
            else ("site",),
            seed=seed,
        )
        np.save(indices_path, sub_idx)
    meta_sub = meta.iloc[sub_idx].reset_index(drop=True)
    if refresh or not meta_path.exists():
        meta_sub.to_csv(meta_path, index=False)
    log.info(
        "[%s] subsample n=%d (target %d), sites=%d, species=%d",
        exp.key,
        len(sub_idx),
        target,
        meta_sub["site"].nunique(),
        meta_sub["species_canonical"].nunique(),
    )

    if (
        refresh
        or not _resolve_path(exp.key, "pre", "tsne").exists()
        or not _resolve_path(exp.key, "pre", "umap").exists()
    ):
        log.info("[%s / pre] PCA -> t-SNE / UMAP", exp.key)
        Z_pre = _project(X_pre[sub_idx], seed=seed)
        t0 = time.time()
        np.save(_resolve_path(exp.key, "pre", "tsne"), _tsne(Z_pre, seed=seed))
        log.info("  pre tSNE %.1fs", time.time() - t0)
        t0 = time.time()
        np.save(_resolve_path(exp.key, "pre", "umap"), _umap(Z_pre, seed=seed))
        log.info("  pre UMAP %.1fs", time.time() - t0)

    for method in METHOD_KEYS:
        corrected_path = CORRECTED_DIR / f"{exp.key}__{method}.npz"
        if not corrected_path.exists():
            log.warning("[%s / %s] missing corrected matrix, skipping", exp.key, method)
            continue
        tsne_path = _resolve_path(exp.key, method, "tsne")
        umap_path = _resolve_path(exp.key, method, "umap")
        if not refresh and tsne_path.exists() and umap_path.exists():
            log.info("[%s / %s] cached, skipping", exp.key, method)
            continue

        log.info(
            "[%s / %s] (%s) PCA -> t-SNE / UMAP",
            exp.key,
            method,
            METHOD_DISPLAY[method],
        )
        X_post = np.load(corrected_path)["X"].astype(np.float32)
        Z_post = _project(X_post[sub_idx], seed=seed)
        if refresh or not tsne_path.exists():
            t0 = time.time()
            np.save(tsne_path, _tsne(Z_post, seed=seed))
            log.info("  tSNE %.1fs", time.time() - t0)
        if refresh or not umap_path.exists():
            t0 = time.time()
            np.save(umap_path, _umap(Z_post, seed=seed))
            log.info("  UMAP %.1fs", time.time() - t0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--experiment",
        choices=[e.key for e in EXPERIMENTS] + ["all"],
        default="all",
    )
    p.add_argument("--target", type=int, default=5000)
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()

    log = setup_logging("04_compute_embeddings", LOGS_DIR / "04_compute_embeddings.log")

    experiments = (
        list(EXPERIMENTS)
        if args.experiment == "all"
        else [e for e in EXPERIMENTS if e.key == args.experiment]
    )

    for exp in experiments:
        log.info("=== experiment %s :: %s ===", exp.key, exp.title)
        compute_for_experiment(
            exp,
            target=args.target,
            refresh=args.refresh,
            log=log,
        )


if __name__ == "__main__":
    main()
