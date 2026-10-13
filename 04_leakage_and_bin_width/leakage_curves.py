"""Leakage as a function of cohort size and of replicate dose (Supplementary Note S3).

``leakage_ladder.py`` works on the full DRIAMS-A cohorts (3-5 thousand spectra per
species-drug pair), where leakage biases are diluted and only 2-4% of isolates carry a
technical replicate. This script repeats the comparisons on smaller cohorts and with
more replicates:

size curve
    Isolate-level subsamples of n = 100, 200, 400, 800 and 1600 isolates, stratified by
    phenotype, 5 subsamples per size, each evaluated with 3 x 5 isolate-grouped folds.
    Leak-safe vs leaky pairs, as in ``leakage_ladder.py``:
      * preprocessing: warping + per-bin standardisation fitted on the training fold
        (``reference``) vs on the whole subsample (``transductive_preproc``);
      * feature selection: 100 Mann-Whitney U bins selected on the training fold vs on the
        whole subsample;
      * decision threshold: inner validation split (``reference``) vs test fold
        (``test_fold_threshold``, same fitted model).
    The full-cohort point is taken from ``leakage_ladder.csv``.

replicate dose
    Subsamples of 300 isolates in which 0, 5, 12 or 25% of isolates carry a technical
    replicate (both spectra kept), 5 subsamples per dose; isolate-grouped vs ungrouped
    stratified folds, 3 x 5 folds, reference model.

Every (pair, curve, level, subsample) cell is written to its own CSV, so the run is
resumable; ``--merge`` rebuilds the combined CSVs.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import _common as C
from leakage_ladder import BIN_WIDTH, K_BINS, _select_bins, _transductive_preprocess

SIZES: tuple[int, ...] = (100, 200, 400, 800, 1600)
DOSES: tuple[float, ...] = (0.0, 0.05, 0.12, 0.25)
DOSE_N = 300
N_SUB = 5
OUT = C.RESULTS_DIR / "leakage_curves"


def _isolates(meta: pd.DataFrame) -> pd.DataFrame:
    iso = meta.groupby("group").agg(label=("rs_label", "max"), label_min=("rs_label", "min"),
                                    n_spectra=("spectrum_id", "size"))
    assert (iso.label == iso.label_min).all(), "replicates of one isolate disagree on the phenotype"
    return iso.drop(columns="label_min")


def _draw(pool: pd.DataFrame, n_r: int, n_s: int, rng: np.random.Generator) -> np.ndarray:
    r = pool.index[pool.label == 1].to_numpy()
    s = pool.index[pool.label == 0].to_numpy()
    if n_r > len(r) or n_s > len(s):
        raise ValueError(f"cannot draw {n_r} R / {n_s} S from {len(r)} R / {len(s)} S")
    return np.concatenate([rng.choice(r, n_r, replace=False), rng.choice(s, n_s, replace=False)])


def size_subsample(iso: pd.DataFrame, n: int, rng: np.random.Generator) -> np.ndarray:
    n_r = int(round(n * iso.label.mean()))
    return _draw(iso, n_r, n - n_r, rng)


def dose_subsample(iso: pd.DataFrame, n: int, frac: float, rng: np.random.Generator) -> np.ndarray:
    """n isolates, a fraction ``frac`` of which carry a technical replicate; overall prevalence kept."""
    rep, single = iso[iso.n_spectra > 1], iso[iso.n_spectra == 1]
    prev = iso.label.mean()
    m = int(round(frac * n))
    m_r = min(int(round(m * prev)), int((rep.label == 1).sum()))
    m_s = m - m_r
    if m_s > (rep.label == 0).sum():
        m_s = int((rep.label == 0).sum())
        m_r = m - m_s
    picked_rep = _draw(rep, m_r, m_s, rng) if m else np.array([], dtype=object)
    n_r_total = int(round(n * prev))
    return np.concatenate([picked_rep, _draw(single, n_r_total - m_r, (n - m) - (n_r_total - m_r), rng)])


def _rows_of(meta: pd.DataFrame, isolates: np.ndarray) -> np.ndarray:
    return np.flatnonzero(meta["group"].isin(set(isolates)).to_numpy())


def _tag(rows: list[dict], **kw) -> list[dict]:
    for r in rows:
        r.update(kw)
    return rows


def run_size_cell(X, y, groups, *, sub: int, log) -> list[dict]:
    from maldideepkit.utils.calibration import tune_threshold

    X_trans = _transductive_preprocess(X)
    leaky_bins = _select_bins(X, y, K_BINS)
    rows: list[dict] = []
    for repeat in range(C.N_REPEATS):
        for fold, (tr, te) in enumerate(C.grouped_folds(y, groups, repeat)):
            run_idx = 1000 * (sub + 1) + repeat * C.N_FOLDS + fold
            y_tr, y_te = y[tr], y[te]
            base = {"repeat": repeat, "fold": fold, "n_train": len(tr), "n_test": len(te),
                    "n_test_resistant": int(y_te.sum())}

            clf = C.make_mlp(X.shape[1], BIN_WIDTH, run_idx)
            clf.fit(X[tr], y_tr)
            m_ref, proba = C.score(clf, X[te], y_te)
            rows.append({**base, **m_ref, "protocol": "reference"})
            t_leak = tune_threshold(y_te, proba, metric="balanced_accuracy")
            rows.append({**base, **C._shared.compute_fold_metrics(y_te, (proba >= t_leak).astype(int), proba),
                         "protocol": "test_fold_threshold"})

            clf = C.make_mlp(X.shape[1], BIN_WIDTH, run_idx, disable_warping=True, input_transform="none")
            clf.fit(X_trans[tr], y_tr)
            rows.append({**base, **C.score(clf, X_trans[te], y_te)[0], "protocol": "transductive_preproc"})

            safe_bins = _select_bins(X[tr], y_tr, K_BINS)
            clf = C.make_mlp(K_BINS, BIN_WIDTH, run_idx, disable_warping=True)
            clf.fit(X[tr][:, safe_bins], y_tr)
            rows.append({**base, **C.score(clf, X[te][:, safe_bins], y_te)[0], "protocol": "feature_selection_safe"})

            clf = C.make_mlp(K_BINS, BIN_WIDTH, run_idx, disable_warping=True)
            clf.fit(X[tr][:, leaky_bins], y_tr)
            rows.append({**base, **C.score(clf, X[te][:, leaky_bins], y_te)[0], "protocol": "feature_selection_leak"})
    return rows


def run_dose_cell(X, y, groups, *, sub: int, log) -> list[dict]:
    rows: list[dict] = []
    for repeat in range(C.N_REPEATS):
        grouped = C.grouped_folds(y, groups, repeat)
        ungrouped = C.stratified_folds(y, repeat)
        for fold in range(C.N_FOLDS):
            run_idx = 1000 * (sub + 1) + repeat * C.N_FOLDS + fold
            for protocol, (tr, te) in (("grouped", grouped[fold]), ("ungrouped", ungrouped[fold])):
                clf = C.make_mlp(X.shape[1], BIN_WIDTH, run_idx)
                clf.fit(X[tr], y[tr])
                m, _ = C.score(clf, X[te], y[te])
                twin = np.isin(groups[te], groups[tr]).mean()
                rows.append({"repeat": repeat, "fold": fold, "protocol": protocol, "n_train": len(tr),
                             "n_test": len(te), "test_share_with_twin_in_train": float(twin), **m})
    return rows


def run_pair(species: str, drug: str, curves: list[str], log) -> None:
    X_by_w, y_all, meta = C.load_pair(species, drug, bin_widths=(BIN_WIDTH,), logger=log)
    X_all = X_by_w[BIN_WIDTH]
    iso = _isolates(meta)
    slug = C.slug(species, drug)
    log.info("%s: %d isolates (%d resistant), %d with a replicate", slug, len(iso), int(iso.label.sum()),
             int((iso.n_spectra > 1).sum()))
    plan = []
    if "size" in curves:
        plan += [("size", n, sub) for n in SIZES if n < len(iso) for sub in range(N_SUB)]
    if "dose" in curves:
        plan += [("dose", d, sub) for d in DOSES for sub in range(N_SUB)]
    for curve, level, sub in plan:
        cell = OUT / slug / f"{curve}_{level:g}_sub{sub}.csv"
        if cell.exists():
            continue
        rng = np.random.default_rng(C.SEED + 7919 * sub + int(1000 * level))
        picked = size_subsample(iso, int(level), rng) if curve == "size" else dose_subsample(iso, DOSE_N, level, rng)
        idx = _rows_of(meta, picked)
        X, y, groups = X_all[idx], y_all[idx], meta["group"].to_numpy()[idx]
        t0 = time.time()
        rows = (run_size_cell if curve == "size" else run_dose_cell)(X, y, groups, sub=sub, log=log)
        _tag(rows, species=species, drug=drug, curve=curve, level=level, subsample=sub,
             n_isolates=len(picked), n_spectra=len(idx), n_resistant_isolates=int(iso.loc[picked, "label"].sum()),
             n_replicated_isolates=int((iso.loc[picked, "n_spectra"] > 1).sum()))
        cell.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(cell, index=False)
        df = pd.DataFrame(rows)
        summ = df.groupby("protocol")[["auroc", "balanced_accuracy"]].mean().round(3).to_dict("index")
        log.info("%s %s=%g sub%d (%d isolates, %d spectra) %.0f s: %s", slug, curve, level, sub, len(picked),
                 len(idx), time.time() - t0, summ)


def merge() -> None:
    for curve in ("size", "dose"):
        files = sorted(OUT.glob(f"*/{curve}_*_sub*.csv"))
        if files:
            out = C.RESULTS_DIR / f"leakage_{curve}_curve.csv"
            pd.concat([pd.read_csv(f) for f in files], ignore_index=True).to_csv(out, index=False)
            print(f"merged {len(files)} cells -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--curves", nargs="*", default=["size", "dose"], choices=["size", "dose"])
    ap.add_argument("--pairs", nargs="*", default=None)
    ap.add_argument("--merge", action="store_true", help="only merge existing cells")
    args = ap.parse_args()
    if not args.merge:
        log = C.setup_logging("leakage_curves")
        for species, drug in C.PAIRS:
            if args.pairs and C.slug(species, drug) not in args.pairs:
                continue
            run_pair(species, drug, args.curves, log)
    merge()


if __name__ == "__main__":
    main()
