"""Leakage ladder: how much do common train/test leakage modes inflate CV estimates?

On DRIAMS-A, for each species-drug pair, the same MLP+attention classifier is
run under one leak-safe reference protocol and several deliberately leaky ones
(Supplementary Note S3). All protocols share the same isolate-grouped,
stratified 3 x 5-fold splits (seeded), so the comparisons are paired.

Protocols
---------
reference
    The MaldiSuite default: spectral warping (median-shift alignment) and
    per-bin standardisation are fitted inside the estimator on the training
    fold only; the decision threshold is tuned on an inner validation split
    of the training fold; no feature selection.
reference_nowarp
    As ``reference`` without warping (the configuration of the same-site
    benchmark in Supplementary Figure S1).
transductive_preproc
    Warping and standardisation are fitted once on the *whole* cohort before
    cross-validation (the "normalise everything, then split" anti-pattern).
test_fold_threshold
    As ``reference``, but the decision threshold is chosen to maximise
    balanced accuracy on the *test* fold itself (evaluation leakage; AUROC is
    unaffected by construction, thresholded metrics are).
feature_selection_safe
    Top-k discriminative bins (Mann-Whitney U) selected on the training fold
    only, then classifier on those k bins. No warping (a k-bin panel cannot
    be aligned).
feature_selection_leak
    Same k, but the bins are selected once on the whole cohort before
    cross-validation (the classic feature-selection leak).
replicate_leak
    As ``reference``, but folds are drawn with a plain ``StratifiedKFold``
    that ignores isolate identity, so the technical replicates of one
    isolate can land on both sides of the split. Only ~2% of DRIAMS-A
    isolates carry a replicate, so the effect is expected to be small; the
    replicate count is logged and saved next to the results.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import _common as C

K_BINS = 100
BIN_WIDTH = 3


def _select_bins(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k bins with the smallest Mann-Whitney U p-value (R vs S)."""
    from scipy.stats import mannwhitneyu

    res = mannwhitneyu(X[y == 1], X[y == 0], axis=0, alternative="two-sided")
    p = np.asarray(res.pvalue, dtype=float)
    p = np.where(np.isfinite(p), p, 1.0)
    return np.argsort(p, kind="stable")[:k]


def _transductive_preprocess(X: np.ndarray) -> np.ndarray:
    """Warping + standardisation fitted on the full matrix (deliberately leaky)."""
    from maldiamrkit.alignment import Warping

    warper = Warping(method="shift", n_jobs=-1)
    df = pd.DataFrame(X)
    Xw = warper.fit(df).transform(df)
    Xw = Xw.to_numpy(dtype=np.float32) if hasattr(Xw, "to_numpy") else np.asarray(Xw, dtype=np.float32)
    mean = Xw.mean(axis=0)
    std = np.maximum(Xw.std(axis=0), 1e-7)
    return ((Xw - mean) / std).astype(np.float32)


def run_pair(species: str, drug: str, *, quick: bool, log) -> pd.DataFrame:
    X_by_w, y, meta = C.load_pair(species, drug, bin_widths=(BIN_WIDTH,), logger=log)
    X = X_by_w[BIN_WIDTH]
    groups = meta["group"].to_numpy()
    summary = C.dataset_summary(meta)
    log.info("%s / %s dataset: %s", species, drug, summary)
    C.write_json(summary, C.RESULTS_DIR / f"leakage_ladder__{C.slug(species, drug)}__dataset.json")

    n_repeats = 1 if quick else C.N_REPEATS
    X_trans = _transductive_preprocess(X)
    leaky_bins = _select_bins(X, y, K_BINS)

    rows: list[dict] = []
    for repeat in range(n_repeats):
        folds = C.grouped_folds(y, groups, repeat)
        strat_folds = C.stratified_folds(y, repeat)
        for fold, (tr, te) in enumerate(folds):
            run_idx = repeat * C.N_FOLDS + fold
            y_tr, y_te = y[tr], y[te]

            t0 = time.time()
            clf = C.make_mlp(X.shape[1], BIN_WIDTH, run_idx, quick=quick)
            clf.fit(X[tr], y_tr)
            m_ref, proba_ref = C.score(clf, X[te], y_te)
            rows.append({**m_ref, "protocol": "reference"})

            from maldideepkit.utils.calibration import tune_threshold

            t_leak = tune_threshold(y_te, proba_ref, metric="balanced_accuracy")
            y_pred_leak = (proba_ref >= t_leak).astype(int)
            rows.append({**C._shared.compute_fold_metrics(y_te, y_pred_leak, proba_ref),
                         "protocol": "test_fold_threshold"})

            tr_l, te_l = strat_folds[fold]
            clf = C.make_mlp(X.shape[1], BIN_WIDTH, run_idx, quick=quick)
            clf.fit(X[tr_l], y[tr_l])
            m, _ = C.score(clf, X[te_l], y[te_l])
            rows.append({**m, "protocol": "replicate_leak"})

            clf = C.make_mlp(X.shape[1], BIN_WIDTH, run_idx, quick=quick, disable_warping=True)
            clf.fit(X[tr], y_tr)
            m, _ = C.score(clf, X[te], y_te)
            rows.append({**m, "protocol": "reference_nowarp"})

            clf = C.make_mlp(X.shape[1], BIN_WIDTH, run_idx, quick=quick,
                             disable_warping=True, input_transform="none")
            clf.fit(X_trans[tr], y_tr)
            m, _ = C.score(clf, X_trans[te], y_te)
            rows.append({**m, "protocol": "transductive_preproc"})

            safe_bins = _select_bins(X[tr], y_tr, K_BINS)
            clf = C.make_mlp(K_BINS, BIN_WIDTH, run_idx, quick=quick, disable_warping=True)
            clf.fit(X[tr][:, safe_bins], y_tr)
            m, _ = C.score(clf, X[te][:, safe_bins], y_te)
            rows.append({**m, "protocol": "feature_selection_safe"})

            clf = C.make_mlp(K_BINS, BIN_WIDTH, run_idx, quick=quick, disable_warping=True)
            clf.fit(X[tr][:, leaky_bins], y_tr)
            m, _ = C.score(clf, X[te][:, leaky_bins], y_te)
            rows.append({**m, "protocol": "feature_selection_leak"})

            for r in rows[-7:]:
                r.update({"species": species, "drug": drug, "repeat": repeat, "fold": fold,
                          "k_bins": K_BINS, "bin_width": BIN_WIDTH})
            log.info("  repeat %d fold %d done in %.0f s: " + ", ".join(
                f"{r['protocol']}={r['auroc']:.3f}/{r['mcc']:.3f}" for r in rows[-7:]),
                repeat, fold, time.time() - t0)

    df = pd.DataFrame(rows)
    lead = ["species", "drug", "protocol", "repeat", "fold", "k_bins", "bin_width"]
    return df[lead + [c for c in df.columns if c not in lead]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="1 repeat, short training (smoke test)")
    ap.add_argument("--pairs", nargs="*", default=None, help="subset of pair slugs to run")
    ap.add_argument("--force", action="store_true", help="re-run pairs whose results already exist")
    args = ap.parse_args()
    suffix = "_quick" if args.quick else ""
    log = C.setup_logging("leakage_ladder" + suffix)

    for species, drug in C.PAIRS:
        if args.pairs and C.slug(species, drug) not in args.pairs:
            continue
        per_pair = C.RESULTS_DIR / f"leakage_ladder__{C.slug(species, drug)}{suffix}.csv"
        if per_pair.exists() and not args.force:
            log.info("=== %s / %s: results exist (%s), skipping ===", species, drug, per_pair.name)
            continue
        log.info("=== %s / %s ===", species, drug)
        df = run_pair(species, drug, quick=args.quick, log=log)
        C._shared.save_dataframe(df, per_pair)
        _merge(suffix, log)
    _merge(suffix, log)


def _merge(suffix: str, log) -> None:
    """Rebuild the combined CSV from every per-pair CSV present (resumable runs)."""
    frames = []
    for species, drug in C.PAIRS:
        per_pair = C.RESULTS_DIR / f"leakage_ladder__{C.slug(species, drug)}{suffix}.csv"
        if per_pair.exists():
            frames.append(pd.read_csv(per_pair))
    if frames:
        out_all = C.RESULTS_DIR / f"leakage_ladder{suffix}.csv"
        C._shared.save_dataframe(pd.concat(frames, ignore_index=True), out_all)
        log.info("Combined results: %s", out_all)


if __name__ == "__main__":
    main()
