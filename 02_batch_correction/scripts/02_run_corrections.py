"""Apply the 11 MaldiBatchKit correctors to each experiment.

Every corrector is fitted on the whole experiment matrix with
``fit_transform``. The corrected matrices feed the embeddings and the
batch-effect metrics, not a generalisation estimate, so there is no
train/test split. The corrected matrix and the fitted estimator are
saved for each (experiment, method).

Method keys (display names in ``_shared.METHODS``):
    combat_johnson, combat_fortin, combat_chen,
    limma, harmony,
    median_centering, zscore_per_batch, reference_scaling,
    batch_aware_warping, quality_weighted_combat, species_aware_combat

Outputs::

    results/corrected/<exp_key>__<method>.npz       corrected matrix (float32)
    cache/estimators/<exp_key>__<method>.joblib     fitted estimator

Run::

    python scripts/02_run_corrections.py
    python scripts/02_run_corrections.py --experiment exp3_multi_species
    python scripts/02_run_corrections.py --method harmony --refresh
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from _shared import (
    CORRECTED_DIR,
    ESTIMATOR_CACHE,
    EXPERIMENTS,
    LOGS_DIR,
    METHOD_DISPLAY,
    METHOD_KEYS,
    Experiment,
    build_method,
    load_experiment,
    setup_logging,
)


def _paths(exp_key: str, method: str) -> tuple[Path, Path]:
    corrected = CORRECTED_DIR / f"{exp_key}__{method}.npz"
    estimator = ESTIMATOR_CACHE / f"{exp_key}__{method}.joblib"
    return corrected, estimator


def run_one(
    exp: Experiment,
    method: str,
    *,
    refresh: bool,
    log,
) -> None:
    """Fit one corrector on one experiment and persist outputs."""
    corrected_path, estimator_path = _paths(exp.key, method)
    if not refresh and corrected_path.exists() and estimator_path.exists():
        log.info("[%s / %s] cached, skipping", exp.key, method)
        return

    X_df, meta = load_experiment(exp)
    batch = meta["site"]
    species = meta["species_canonical"] if exp.species_covariate else None
    snr = meta["snr"].astype(float).fillna(meta["snr"].astype(float).median())

    log.info(
        "[%s / %s] fitting on X=%s, n_sites=%d, n_species=%d",
        exp.key,
        method,
        X_df.shape,
        meta["site"].nunique(),
        meta["species_canonical"].nunique(),
    )
    estimator = build_method(
        method,
        batch=batch,
        species=species,
        snr=snr,
    )

    t0 = time.time()
    X_corrected = estimator.fit_transform(X_df)
    dt = time.time() - t0

    if isinstance(X_corrected, pd.DataFrame):
        arr = X_corrected.to_numpy(dtype=np.float32)
    else:
        arr = np.asarray(X_corrected, dtype=np.float32)
    np.savez_compressed(corrected_path, X=arr)
    joblib.dump(estimator, estimator_path)
    log.info("[%s / %s] done in %.1fs -> %s", exp.key, method, dt, corrected_path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--experiment",
        choices=[e.key for e in EXPERIMENTS] + ["all"],
        default="all",
    )
    p.add_argument(
        "--method",
        choices=METHOD_KEYS + ["all"],
        default="all",
    )
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()

    log = setup_logging("02_run_corrections", LOGS_DIR / "02_run_corrections.log")

    experiments = (
        list(EXPERIMENTS)
        if args.experiment == "all"
        else [e for e in EXPERIMENTS if e.key == args.experiment]
    )
    methods = METHOD_KEYS if args.method == "all" else [args.method]

    for exp in experiments:
        log.info("=== experiment %s :: %s ===", exp.key, exp.title)
        for method in methods:
            try:
                run_one(exp, method, refresh=args.refresh, log=log)
            except Exception as exc:
                log.exception(
                    "[%s / %s] FAILED: %s (%s)",
                    exp.key,
                    method,
                    METHOD_DISPLAY.get(method, method),
                    exc,
                )


if __name__ == "__main__":
    main()
