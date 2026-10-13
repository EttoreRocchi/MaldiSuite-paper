"""Render Supplementary Figure S1 from the multi-site benchmark.

Reads ``results_multisite/summary.csv`` (produced by
``analyze_multisite.py``) and writes ``figure_s1_benchmark.{pdf,png}``
under ``results_multisite/figures/``: three panels, one per
species-drug pair. X-axis: DRIAMS site (A to D). Grouped bars:
classifier (5 colours). Y-axis: mean AUROC +/- std across the 15 fits
(3 repeats x 5 folds). The winning classifier in each (site, pair)
cell is marked with a star above its bar.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _shared import (
    CLASSIFIER_DISPLAY,
    HERE,
    setup_logging,
)
from matplotlib.patches import Patch

logger = logging.getLogger("make_paper_figures")

CLASSIFIER_ORDER: tuple[str, ...] = (
    "mlp_attn",
    "cnn",
    "resnet",
    "transformer",
    "ensemble_mlp_cnn",
)

CLASSIFIER_COLORS: dict[str, str] = {
    "mlp_attn": "#0072B2",
    "cnn": "#009E73",
    "resnet": "#D55E00",
    "transformer": "#CC79A7",
    "ensemble_mlp_cnn": "#E69F00",
}

SITE_ORDER: tuple[str, ...] = ("DRIAMS-A", "DRIAMS-B", "DRIAMS-C", "DRIAMS-D")

PAIR_ORDER: tuple[tuple[str, str], ...] = (
    ("Staphylococcus aureus", "oxacillin"),
    ("Escherichia coli", "ceftriaxone"),
    ("Klebsiella pneumoniae", "ceftriaxone"),
)

def _set_publication_style() -> None:
    """Matplotlib settings for the figure."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 1.2,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _load_summary(summary_csv: Path) -> pd.DataFrame:
    if not summary_csv.exists():
        raise SystemExit(
            f"{summary_csv} not found. Run `make analyze-multisite` first."
        )
    df = pd.read_csv(summary_csv)
    needed = {"site", "species", "drug", "classifier", "auroc_mean", "auroc_std"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"summary.csv missing columns: {sorted(missing)}")
    return df


def _save_both(fig: mpl.figure.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    logger.info("wrote %s + %s", pdf_path.name, png_path.name)


def fig_benchmark(df: pd.DataFrame, out_dir: Path) -> None:
    """Three panels (one per pair); grouped bars per (site, classifier)."""
    n_pairs = len(PAIR_ORDER)
    n_clf = len(CLASSIFIER_ORDER)
    bar_width = 0.16

    fig, axes = plt.subplots(
        nrows=1, ncols=n_pairs, figsize=(11.0, 4.2), sharey=True
    )
    if n_pairs == 1:
        axes = [axes]

    site_x = np.arange(len(SITE_ORDER))

    for ax, (species, drug) in zip(axes, PAIR_ORDER, strict=True):
        ax.set_title(
            f"$\\it{{{species.split()[0][0]}.\\,{species.split()[1]}}}$ / {drug}",
            pad=8,
        )
        sub = df[(df["species"] == species) & (df["drug"] == drug)]
        for j, clf in enumerate(CLASSIFIER_ORDER):
            offset = (j - (n_clf - 1) / 2) * bar_width
            means: list[float] = []
            stds: list[float] = []
            for site in SITE_ORDER:
                cell = sub[(sub["site"] == site) & (sub["classifier"] == clf)]
                if cell.empty:
                    means.append(np.nan)
                    stds.append(np.nan)
                else:
                    means.append(float(cell["auroc_mean"].iloc[0]))
                    stds.append(float(cell["auroc_std"].iloc[0]))
            color = CLASSIFIER_COLORS[clf]
            ax.bar(
                site_x + offset,
                means,
                width=bar_width,
                color=color,
                yerr=stds,
                error_kw=dict(ecolor="black", lw=0.7, capsize=2),
                edgecolor="none",
            )
        for k, site in enumerate(SITE_ORDER):
            cell = sub[sub["site"] == site]
            if cell.empty:
                continue
            winner = cell.sort_values("auroc_mean", ascending=False).iloc[0]
            j = CLASSIFIER_ORDER.index(winner["classifier"])
            offset = (j - (n_clf - 1) / 2) * bar_width
            top = winner["auroc_mean"] + winner["auroc_std"] + 0.02
            ax.text(
                k + offset,
                min(top, 1.03),
                "*",
                ha="center",
                va="bottom",
                fontsize=10,
                color="black",
            )

        ax.set_xticks(site_x)
        ax.set_xticklabels([s.replace("DRIAMS-", "") for s in SITE_ORDER])
        ax.set_xlabel("DRIAMS site")
        if ax is axes[0]:
            ax.set_ylabel("Mean AUROC")
        ax.set_ylim(0.45, 1.05)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.axhline(0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.6)
        ax.grid(axis="y", linewidth=0.4, alpha=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", pad=2)

    handles = [
        Patch(color=CLASSIFIER_COLORS[c], label=CLASSIFIER_DISPLAY[c])
        for c in CLASSIFIER_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(CLASSIFIER_ORDER),
        frameon=False,
        columnspacing=2.0,
        handlelength=1.6,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save_both(fig, out_dir, "figure_s1_benchmark")


def main(summary_csv: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging("make_paper_figures", out_dir / "make_paper_figures.log")

    df = _load_summary(summary_csv)
    logger.info("loaded %d aggregated cells from %s", len(df), summary_csv)

    _set_publication_style()
    fig_benchmark(df, out_dir)
    logger.info("done; figures under %s", out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=HERE / "results_multisite" / "summary.csv",
        help="Path to summary.csv emitted by analyze_multisite.py.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=HERE / "results_multisite" / "figures",
        help="Where to write figure_s1_benchmark.{pdf,png}.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.summary_csv, args.out_dir)
