"""Classification benchmark: five MaldiDeepKit classifier configurations on three DRIAMS AMR tasks.

Runs repeated stratified K-fold CV of each configuration (four
architectures and an ensemble of the MLP and the CNN) on three
species-drug resistance-prediction tasks from DRIAMS A-D:

- *Staphylococcus aureus* / oxacillin
- *Escherichia coli* / ceftriaxone
- *Klebsiella pneumoniae* / ceftriaxone

Outputs
-------
``<results-dir>/benchmark.csv``
    One row per (pair, classifier, repeat, fold) with all metrics.
``<results-dir>/<pair-slug>/<classifier>/fold_metrics.csv``
    Per-pair / per-classifier metric snapshot.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
from _shared import (
    CLASSIFIER_DISPLAY,
    CLASSIFIERS,
    DATA_CACHE_DIR,
    DEFAULT_BIN_WIDTH,
    DRIAMS_SITES,
    N_FOLDS,
    N_REPEATS,
    RESULTS_DIR,
    SEED,
    SPECIES_DRUG_PAIRS,
    build_classifier,
    load_pair,
    run_cv,
    save_dataframe,
    setup_logging,
    slug,
)

logger = logging.getLogger("benchmark")


def _run_one(
    species: str,
    drug: str,
    classifier_name: str,
    *,
    n_folds: int,
    n_repeats: int,
    random_state: int,
    quick: bool,
    results_dir: Path,
    sites: tuple[str, ...],
    data_cache_dir: Path,
    disable_warping: bool,
    force: bool = False,
) -> pd.DataFrame:
    """Run repeated CV for one (pair, classifier) cell and persist outputs.

    If ``fold_metrics.csv`` already exists for this cell with the
    expected ``n_repeats * n_folds`` rows and ``force=False``, the
    existing file is loaded and returned without retraining. A wrong
    row count is treated as a stale cache and triggers retraining.
    Use ``--force`` on the CLI to recompute unconditionally.
    """
    pair_slug = slug(species, drug)
    cell_dir = results_dir / pair_slug / classifier_name
    metrics_path = cell_dir / "fold_metrics.csv"

    expected_rows = n_repeats * n_folds
    logger.info("=== %s / %s | classifier=%s ===", species, drug, classifier_name)

    if metrics_path.exists() and not force:
        cached = pd.read_csv(metrics_path)
        if len(cached) == expected_rows:
            logger.info("  cached -> skipping (found %s)", metrics_path)
            return cached
        logger.info(
            "  stale cache (%d rows != %d = %d repeats * %d folds) -> retraining",
            len(cached),
            expected_rows,
            n_repeats,
            n_folds,
        )

    try:
        X, y, _meta = load_pair(
            species,
            drug,
            use_cache=True,
            sites=sites,
            cache_dir=data_cache_dir,
        )
    except FileNotFoundError as exc:
        logger.warning(
            "  skipping %s / %s on sites=%s: %s",
            species,
            drug,
            list(sites),
            exc,
        )
        return pd.DataFrame()
    input_dim = int(X.shape[1])

    def factory(run_idx: int):
        return build_classifier(
            classifier_name,
            bin_width=DEFAULT_BIN_WIDTH,
            input_dim=input_dim,
            random_state=random_state + run_idx,
            quick=quick,
            disable_warping=disable_warping,
        )

    t0 = time.time()
    df = run_cv(
        factory,
        X,
        y,
        n_folds=n_folds,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    elapsed = time.time() - t0

    df.insert(0, "classifier", classifier_name)
    df.insert(0, "drug", drug)
    df.insert(0, "species", species)

    save_dataframe(df, metrics_path)

    logger.info(
        "  done in %.1f s -> mean auroc=%.3f, mean mcc=%.3f (%d repeats * %d folds)",
        elapsed,
        df["auroc"].mean(),
        df["mcc"].mean(),
        n_repeats,
        n_folds,
    )
    return df


def main(
    pairs: list[tuple[str, str]],
    classifiers: list[str],
    n_folds: int,
    n_repeats: int,
    random_state: int,
    quick: bool,
    results_dir: Path,
    sites: tuple[str, ...],
    data_cache_dir: Path,
    disable_warping: bool,
    force: bool = False,
) -> None:
    """Run the full benchmark over every (pair, classifier) cell.

    Cells with an existing ``fold_metrics.csv`` of the expected size
    (``n_repeats * n_folds`` rows) are skipped unless ``force=True``.
    An interrupted ``make benchmark-<site>`` therefore resumes where it
    stopped.
    """
    setup_logging("benchmark", results_dir / "benchmark.log")
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("MaldiDeepKit paper benchmark")
    logger.info("Pairs: %s", pairs)
    logger.info("Classifiers: %s", classifiers)
    logger.info("Sites: %s", list(sites))
    logger.info("Data cache: %s", data_cache_dir)
    logger.info("Warping enabled: %s", not disable_warping)
    logger.info(
        "Repeats: %d, folds: %d (%d fits per cell), seed: %d, quick: %s, force: %s",
        n_repeats,
        n_folds,
        n_repeats * n_folds,
        random_state,
        quick,
        force,
    )

    t_start = time.time()
    all_rows: list[pd.DataFrame] = []
    for species, drug in pairs:
        for clf_name in classifiers:
            df = _run_one(
                species,
                drug,
                clf_name,
                n_folds=n_folds,
                n_repeats=n_repeats,
                random_state=random_state,
                quick=quick,
                results_dir=results_dir,
                sites=sites,
                data_cache_dir=data_cache_dir,
                disable_warping=disable_warping,
                force=force,
            )
            if not df.empty:
                all_rows.append(df)

    if not all_rows:
        logger.warning("No (pair, classifier) cells produced any rows.")
        return
    combined = pd.concat(all_rows, ignore_index=True)
    save_dataframe(combined, results_dir / "benchmark.csv")

    total = time.time() - t_start
    logger.info("Benchmark complete in %.1f s (%.1f min)", total, total / 60)
    print(f"Total elapsed: {total / 60:.1f} min")
    print(f"Raw results: {results_dir / 'benchmark.csv'}")
    print("Run `make figures` once all four sites are done to build the paper tables.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--n-folds",
        type=int,
        default=N_FOLDS,
        help=f"Stratified K-fold CV splits per repeat (default: {N_FOLDS}).",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=N_REPEATS,
        help=(
            f"Number of independent outer shuffles (default: {N_REPEATS}). "
            f"Total fits per (pair, classifier) = n_repeats * n_folds."
        ),
    )
    parser.add_argument(
        "--species",
        type=str,
        default=None,
        help=(
            "Restrict to one species (matches the species field of the "
            "species-drug pairs)."
        ),
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=None,
        help=(
            f"Subset of {sorted(CLASSIFIER_DISPLAY)} "
            f"(default paper set: {CLASSIFIERS})."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Seed for CV split + classifier init (default: {SEED}).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Shrink epoch counts for a smoke-test run (non-paper quality).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Retrain every (pair, classifier) cell even if a cached "
            "fold_metrics.csv exists. Use this after changing "
            "hyperparameters, the data cache, or the classifier factory."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"Where to write CSVs and logs (default: {RESULTS_DIR}).",
    )
    parser.add_argument(
        "--sites",
        nargs="+",
        default=None,
        help=(
            f"Subset of DRIAMS sites to union (default all four: {list(DRIAMS_SITES)}). "
            "Pass e.g. 'DRIAMS-A' for a single-site benchmark."
        ),
    )
    parser.add_argument(
        "--data-cache",
        type=Path,
        default=DATA_CACHE_DIR,
        help=(
            f"Directory for the cached per-pair spectra (default: {DATA_CACHE_DIR}). "
            "Use a dedicated path (e.g. .data_cache_A) when running on a site "
            "subset, otherwise the cached matrix from a previous A-D run would "
            "silently be reused."
        ),
    )
    parser.add_argument(
        "--no-warping",
        action="store_true",
        help=(
            "Disable the Warping preprocessor. Recommended for same-site "
            "runs (e.g. DRIAMS-A only) where cross-site calibration drift "
            "is not present."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.species is not None:
        pairs = [p for p in SPECIES_DRUG_PAIRS if p[0] == args.species]
        if not pairs:
            raise SystemExit(
                f"No pair matches species={args.species!r}. "
                f"Known: {sorted({p[0] for p in SPECIES_DRUG_PAIRS})}"
            )
    else:
        pairs = SPECIES_DRUG_PAIRS

    selected = args.classifiers if args.classifiers is not None else CLASSIFIERS
    unknown = [c for c in selected if c not in CLASSIFIER_DISPLAY]
    if unknown:
        raise SystemExit(
            f"Unknown classifier(s): {unknown}. Known: {sorted(CLASSIFIER_DISPLAY)}"
        )

    sites = tuple(args.sites) if args.sites is not None else DRIAMS_SITES
    allowed_sites = set(DRIAMS_SITES)
    unknown_sites = [s for s in sites if s not in allowed_sites]
    if unknown_sites:
        raise SystemExit(
            f"Unknown site(s): {unknown_sites}. Known: {sorted(allowed_sites)}"
        )

    main(
        pairs=pairs,
        classifiers=selected,
        n_folds=args.n_folds,
        n_repeats=args.n_repeats,
        random_state=args.seed,
        quick=args.quick,
        results_dir=args.results_dir,
        sites=sites,
        data_cache_dir=args.data_cache,
        disable_warping=args.no_warping,
        force=args.force,
    )
