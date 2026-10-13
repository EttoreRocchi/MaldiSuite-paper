"""Load and cache the DRIAMS matrices of the three species-drug pairs.

Each pair is loaded from the four DRIAMS sites, preprocessed with the
MaldiAMRKit default pipeline (clip negatives, square root,
Savitzky-Golay, SNIP baseline, trimming to 2000-20000 Da, TIC
normalisation) and binned to 6000 bins of 3 Da; the SNR is computed on
the preprocessed spectrum.

Outputs (per pair) under ``cache/data/``:
    <pair_slug>.npz                X matrix (float32)
    <pair_slug>_meta.csv           site, species_canonical, drug_canonical,
                                   rs_label, snr

Run::

    python scripts/01_load_data.py
    python scripts/01_load_data.py --refresh

``--refresh`` rebuilds the cache.
"""

from __future__ import annotations

import argparse
import time

from _shared import LOGS_DIR, PAIRS, load_pair, setup_logging


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Re-load and overwrite caches even if they already exist.",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Workers for DatasetLoader (default: all cores).",
    )
    args = p.parse_args()

    log = setup_logging("01_load_data", LOGS_DIR / "01_load_data.log")

    for species, drug in PAIRS:
        t0 = time.time()
        X, meta = load_pair(
            species, drug, n_jobs=args.n_jobs, use_cache=not args.refresh
        )
        elapsed = time.time() - t0
        log.info(
            "%s / %s -> X=%s, n_sites=%d, time=%.1fs",
            species,
            drug,
            X.shape,
            meta["site"].nunique(),
            elapsed,
        )
        log.info(
            "  site counts: %s",
            meta["site"].value_counts().sort_index().to_dict(),
        )
        log.info(
            "  R/S counts: %s",
            meta["rs_label"].value_counts().sort_index().to_dict(),
        )


if __name__ == "__main__":
    main()
