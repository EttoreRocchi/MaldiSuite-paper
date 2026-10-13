"""Bin-width sensitivity of the DRIAMS-A classification benchmark (Supplementary Note S4).

Checks whether the results change with the bin width, i.e. whether a
resistance-specific peak could be merged with a susceptibility-specific
one at the default 3 Da. The raw DRIAMS-A spectra go through the
MaldiAMRKit default preprocessing, are binned at 1, 2, 3, 5 and 10 Da
and are classified with MLP+attention under the same isolate-grouped,
stratified 3 x 5-fold CV, without warping as in the single-site
benchmark of Supplementary Figure S1. Architecture defaults rescale with
the input through ``from_spectrum``.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

import _common as C

BIN_WIDTHS: tuple[float, ...] = (1, 2, 3, 5, 10)


def run_pair(species: str, drug: str, *, quick: bool, log) -> pd.DataFrame:
    X_by_w, y, meta = C.load_pair(species, drug, bin_widths=BIN_WIDTHS, logger=log)
    groups = meta["group"].to_numpy()
    n_repeats = 1 if quick else C.N_REPEATS
    rows: list[dict] = []
    for w in BIN_WIDTHS:
        X = X_by_w[w]
        log.info("bin width %g Da -> %d bins", w, X.shape[1])
        for repeat in range(n_repeats):
            for fold, (tr, te) in enumerate(C.grouped_folds(y, groups, repeat)):
                run_idx = repeat * C.N_FOLDS + fold
                t0 = time.time()
                clf = C.make_mlp(X.shape[1], w, run_idx, quick=quick, disable_warping=True)
                clf.fit(X[tr], y[tr])
                m, _ = C.score(clf, X[te], y[te])
                m.update({"species": species, "drug": drug, "bin_width": w, "n_bins": X.shape[1],
                          "repeat": repeat, "fold": fold})
                rows.append(m)
                log.info("  bw=%g repeat %d fold %d: auroc=%.3f mcc=%.3f (%.0f s)",
                         w, repeat, fold, m["auroc"], m["mcc"], time.time() - t0)
    df = pd.DataFrame(rows)
    lead = ["species", "drug", "bin_width", "n_bins", "repeat", "fold"]
    return df[lead + [c for c in df.columns if c not in lead]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="1 repeat, short training (smoke test)")
    ap.add_argument("--pairs", nargs="*", default=None, help="subset of pair slugs to run")
    ap.add_argument("--force", action="store_true", help="re-run pairs whose results already exist")
    args = ap.parse_args()
    suffix = "_quick" if args.quick else ""
    log = C.setup_logging("bin_width_sensitivity" + suffix)

    for species, drug in C.PAIRS:
        if args.pairs and C.slug(species, drug) not in args.pairs:
            continue
        per_pair = C.RESULTS_DIR / f"bin_width_sensitivity__{C.slug(species, drug)}{suffix}.csv"
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
        per_pair = C.RESULTS_DIR / f"bin_width_sensitivity__{C.slug(species, drug)}{suffix}.csv"
        if per_pair.exists():
            frames.append(pd.read_csv(per_pair))
    if frames:
        out_all = C.RESULTS_DIR / f"bin_width_sensitivity{suffix}.csv"
        C._shared.save_dataframe(pd.concat(frames, ignore_index=True), out_all)
        log.info("Combined results: %s", out_all)


if __name__ == "__main__":
    main()
