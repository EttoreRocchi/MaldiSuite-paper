"""Aggregate the per-site benchmark CSVs into the multi-site results of the paper.

Reads ``results_{A,B,C,D}/benchmark.csv`` (the outputs of
``make benchmark-A`` / ``-B`` / ``-C`` / ``-D``), tags each row with its
site and concatenates them.

Outputs (under ``results_multisite/`` by default):

- ``benchmark_multisite.csv``: the four per-site CSVs concatenated, one
  row per fold, with a ``site`` column.
- ``summary.csv``: mean and std per ``(site, species, drug, classifier)``
  cell, the input of Figure 1d and Supplementary Figure S1.

Sites that have not produced a ``benchmark.csv`` are skipped with a
warning. Pairs missing on a given site (e.g. DRIAMS-D has no oxacillin
annotations) appear only in the rows of sites that have them.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from _shared import (
    HERE,
    save_dataframe,
    setup_logging,
)

logger = logging.getLogger("analyze_multisite")

METRICS_SHOWN = [
    "auroc",
    "mcc",
    "balanced_accuracy",
    "f1_r",
    "f1_s",
    "recall_r",
    "recall_s",
    "vme",
    "me",
]

SITE_INPUTS: tuple[tuple[str, Path], ...] = (
    ("DRIAMS-A", HERE / "results_A" / "benchmark.csv"),
    ("DRIAMS-B", HERE / "results_B" / "benchmark.csv"),
    ("DRIAMS-C", HERE / "results_C" / "benchmark.csv"),
    ("DRIAMS-D", HERE / "results_D" / "benchmark.csv"),
)


def load_concat(inputs: tuple[tuple[str, Path], ...]) -> pd.DataFrame:
    """Load each per-site CSV, tag with ``site``, return concatenated."""
    blocks: list[pd.DataFrame] = []
    for site, path in inputs:
        if not path.exists():
            logger.warning("skipping %s: %s not found", site, path)
            continue
        df = pd.read_csv(path)
        df = df.copy()
        df["site"] = site
        blocks.append(df)
        logger.info("loaded %d rows from %s", len(df), path)
    if not blocks:
        raise SystemExit(
            "No per-site benchmark.csv found. Run `make benchmark-A/B/C/D` first."
        )
    return pd.concat(blocks, ignore_index=True)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse fold rows to ``mean / std`` per ``(site, species, drug, classifier)``."""
    group_cols = ["site", "species", "drug", "classifier"]
    agg = df.groupby(group_cols)[METRICS_SHOWN].agg(["mean", "std"]).reset_index()
    agg.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col
        for col in agg.columns
    ]
    n_fits = (
        df.groupby(group_cols).size().rename("n_fits").reset_index()
    )
    agg = agg.merge(n_fits, on=group_cols)
    return agg


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging("analyze_multisite", out_dir / "analyze_multisite.log")

    long_df = load_concat(SITE_INPUTS)
    save_dataframe(long_df, out_dir / "benchmark_multisite.csv")

    agg = aggregate(long_df)
    save_dataframe(agg, out_dir / "summary.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=HERE / "results_multisite",
        help="Where to write benchmark_multisite.csv and summary.csv.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.out_dir)
