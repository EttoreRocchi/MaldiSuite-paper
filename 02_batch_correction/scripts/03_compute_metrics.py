"""Compute the batch-effect and signal-preservation metrics behind Supplementary Tables S4-S7.

Batch-effect metrics:
    - silhouette by batch (closer to 0 = better mixing)
    - Davies-Bouldin index by batch (lower = better)
    - kBET acceptance rate (higher = better mixing) and mean chi^2
    - LISI by batch (closer to the number of batches = better mixing)
    - variance ratio, between-batch SS over total SS (lower = less batch
      structure)

Signal-preservation metrics (higher = more of the original structure kept):
    - k-NN preservation: mean Jaccard similarity of each sample's k-NN
      set before and after correction (k=15)
    - distance correlation: Pearson correlation of pairwise distances
      before and after correction, on a fixed random subset of pairs

For the multi-species experiment, which has a species covariate, also:
    - LISI by species (close to 1 = species kept apart)
    - silhouette by species (higher = species better separated)
    - species preservation

Output: ``results/metrics/supp_table_s3.csv``, one row per
(experiment, method, stage). Stage ``pre`` is the uncorrected data
(one row per experiment, method ``__pre__``), stage ``post`` the data
after correction.

Run::

    python scripts/03_compute_metrics.py
    python scripts/03_compute_metrics.py --experiment exp3_multi_species
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

from _shared import (
    CORRECTED_DIR,
    EXPERIMENTS,
    LOGS_DIR,
    METHOD_DISPLAY,
    METHOD_KEYS,
    METRICS_DIR,
    SEED,
    Experiment,
    load_experiment,
    setup_logging,
    stratified_subsample,
)


def _safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    from maldibatchkit.diagnostics import silhouette_batch

    return float(silhouette_batch(X, labels))


def _safe_davies_bouldin(X: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    try:
        return float(davies_bouldin_score(X, labels))
    except Exception:
        return float("nan")


def _kbet(X: np.ndarray, batch: np.ndarray, k: int) -> tuple[float, float]:
    from maldibatchkit.diagnostics import kbet

    res = kbet(X, batch, k=k)
    return float(res["acceptance_rate"]), float(res["mean_chi2"])


def _lisi(X: np.ndarray, labels: np.ndarray) -> float:
    from maldibatchkit.diagnostics import lisi

    return float(lisi(X, labels))


def _variance_ratio(X: np.ndarray, batch: np.ndarray) -> float:
    """Between-batch over total sum of squares (one-way ANOVA R^2), averaged over features.

    Being a ratio per feature, it does not depend on the scale of the data.
    """
    levels = np.unique(batch)
    if len(levels) < 2:
        return 0.0
    grand = X.mean(axis=0)
    ss_total = np.sum((X - grand) ** 2, axis=0)
    ss_between = np.zeros(X.shape[1])
    for lvl in levels:
        mask = batch == lvl
        if not np.any(mask):
            continue
        diff = X[mask].mean(axis=0) - grand
        ss_between += int(mask.sum()) * (diff**2)
    valid = ss_total > 1e-12
    if not np.any(valid):
        return 0.0
    return float(np.mean(ss_between[valid] / ss_total[valid]))


def _knn_preservation(
    X_pre: np.ndarray, X_post: np.ndarray, k: int = 15
) -> float:
    """Mean Jaccard similarity between pre- and post-correction k-NN sets."""
    nn_pre = NearestNeighbors(n_neighbors=k + 1).fit(X_pre)
    nn_post = NearestNeighbors(n_neighbors=k + 1).fit(X_post)
    _, idx_pre = nn_pre.kneighbors(X_pre)
    _, idx_post = nn_post.kneighbors(X_post)
    idx_pre = idx_pre[:, 1:]
    idx_post = idx_post[:, 1:]
    sims = []
    for i in range(X_pre.shape[0]):
        s_pre = set(idx_pre[i])
        s_post = set(idx_post[i])
        union = len(s_pre | s_post)
        sims.append(len(s_pre & s_post) / union if union > 0 else 0.0)
    return float(np.mean(sims))


def _distance_correlation(
    X_pre: np.ndarray,
    X_post: np.ndarray,
    *,
    n_pairs: int = 200_000,
    seed: int = SEED,
) -> float:
    """Pearson correlation of pairwise distances pre vs post on random pairs."""
    rng = np.random.default_rng(seed)
    n = X_pre.shape[0]
    n_pairs = min(n_pairs, n * (n - 1) // 2)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    d_pre = np.linalg.norm(X_pre[i] - X_pre[j], axis=1)
    d_post = np.linalg.norm(X_post[i] - X_post[j], axis=1)
    if np.std(d_pre) < 1e-12 or np.std(d_post) < 1e-12:
        return float("nan")
    return float(np.corrcoef(d_pre, d_post)[0, 1])


def _evaluate_stage(
    X: np.ndarray,
    meta: pd.DataFrame,
    *,
    has_species: bool,
    k_kbet: int,
) -> dict[str, float]:
    from maldibatchkit.diagnostics import lisi_max, species_preservation

    site = meta["site"].to_numpy()
    lisi_b = _lisi(X, site)
    lisi_b_max = lisi_max(site)
    out: dict[str, float] = {
        "silhouette_batch": _safe_silhouette(X, site),
        "davies_bouldin_batch": _safe_davies_bouldin(X, site),
        "lisi_batch": lisi_b,
        "lisi_batch_max": lisi_b_max,
        "lisi_batch_normalized": (
            lisi_b / lisi_b_max if lisi_b_max > 1.0 else float("nan")
        ),
        "variance_ratio_batch": _variance_ratio(X, site),
    }
    accept, chi2 = _kbet(X, site, k=k_kbet)
    out["kbet_acceptance"] = accept
    out["kbet_mean_chi2"] = chi2

    if has_species:
        species = meta["species_canonical"].to_numpy()
        lisi_s = _lisi(X, species)
        lisi_s_max = lisi_max(species)
        out["silhouette_species"] = _safe_silhouette(X, species)
        out["lisi_species"] = lisi_s
        out["lisi_species_max"] = lisi_s_max
        out["species_preservation"] = species_preservation(X, species)
    return out


def evaluate_experiment(
    exp: Experiment,
    *,
    log,
    eval_subsample: int = 8000,
    k_kbet: int = 30,
) -> pd.DataFrame:
    """Compute every metric for one experiment, before and after each correction."""
    X_df, meta = load_experiment(exp)
    X_pre_full = X_df.to_numpy(dtype=np.float32)

    sub_idx = stratified_subsample(
        meta, target=eval_subsample, stratify_cols=("site",)
    )
    log.info(
        "[%s] eval-subsample %d/%d (stratified by site)",
        exp.key,
        len(sub_idx),
        len(meta),
    )
    meta_sub = meta.iloc[sub_idx].reset_index(drop=True)
    X_pre_sub = X_pre_full[sub_idx]
    has_species = bool(exp.species_covariate)

    rows: list[dict[str, Any]] = []
    log.info("[%s] evaluating the uncorrected data", exp.key)
    pre_metrics = _evaluate_stage(
        X_pre_sub, meta_sub, has_species=has_species, k_kbet=k_kbet
    )
    pre_row = {
        "experiment": exp.key,
        "method": "__pre__",
        "method_display": "(uncorrected)",
        "stage": "pre",
        **pre_metrics,
        "knn_preservation": float("nan"),
        "distance_correlation": float("nan"),
    }
    rows.append(pre_row)

    for method in METHOD_KEYS:
        path = CORRECTED_DIR / f"{exp.key}__{method}.npz"
        if not path.exists():
            log.warning("[%s / %s] missing corrected matrix, skipping", exp.key, method)
            continue
        X_post_full = np.load(path)["X"].astype(np.float32)
        if X_post_full.shape != X_pre_full.shape:
            log.warning(
                "[%s / %s] shape mismatch %s vs %s, skipping",
                exp.key,
                method,
                X_post_full.shape,
                X_pre_full.shape,
            )
            continue
        X_post_sub = X_post_full[sub_idx]

        t0 = time.time()
        post_metrics = _evaluate_stage(
            X_post_sub, meta_sub, has_species=has_species, k_kbet=k_kbet
        )
        knn_pres = _knn_preservation(X_pre_sub, X_post_sub, k=15)
        dist_corr = _distance_correlation(X_pre_sub, X_post_sub)
        elapsed = time.time() - t0

        rows.append(
            {
                "experiment": exp.key,
                "method": method,
                "method_display": METHOD_DISPLAY[method],
                "stage": "post",
                **post_metrics,
                "knn_preservation": knn_pres,
                "distance_correlation": dist_corr,
            }
        )
        log.info(
            "[%s / %s] silh=%.3f -> %.3f, lisi=%.2f -> %.2f, kbet=%.2f -> %.2f, "
            "knn_pres=%.2f, dist_r=%.2f, %.1fs",
            exp.key,
            method,
            pre_metrics["silhouette_batch"],
            post_metrics["silhouette_batch"],
            pre_metrics["lisi_batch"],
            post_metrics["lisi_batch"],
            pre_metrics["kbet_acceptance"],
            post_metrics["kbet_acceptance"],
            knn_pres,
            dist_corr,
            elapsed,
        )

    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--experiment",
        choices=[e.key for e in EXPERIMENTS] + ["all"],
        default="all",
    )
    p.add_argument(
        "--eval-subsample",
        type=int,
        default=8000,
        help="Stratified subsample size used for batch-level metrics.",
    )
    p.add_argument(
        "--k-kbet",
        type=int,
        default=30,
        help="k for kBET (also affects k-NN preservation indirectly).",
    )
    args = p.parse_args()

    log = setup_logging("03_compute_metrics", LOGS_DIR / "03_compute_metrics.log")

    experiments = (
        list(EXPERIMENTS)
        if args.experiment == "all"
        else [e for e in EXPERIMENTS if e.key == args.experiment]
    )

    frames: list[pd.DataFrame] = []
    for exp in experiments:
        log.info("=== experiment %s ===", exp.key)
        frames.append(
            evaluate_experiment(
                exp,
                log=log,
                eval_subsample=args.eval_subsample,
                k_kbet=args.k_kbet,
            )
        )
    if not frames:
        log.warning("Nothing evaluated.")
        return
    full = pd.concat(frames, ignore_index=True)

    csv_path = METRICS_DIR / "supp_table_s3.csv"
    full.to_csv(csv_path, index=False)
    log.info("Wrote %s (%d rows)", csv_path, len(full))


if __name__ == "__main__":
    main()
