"""Render Supplementary Figure S2 from the cached embeddings of Experiment 3.

Writes ``figures/figure_s2a_tsne_site.{png,pdf}`` and
``figures/figure_s2b_umap_site.{png,pdf}``: uncorrected + 11 correctors
on the multi-species, multi-site cohort, coloured by DRIAMS site.
Figure 1c and Supplementary Figure S3 are rendered from the same
embeddings by ``figures/make_figure1.py`` and
``figures/make_supp_figures.py`` at the repository root.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from _shared import (
    EMBEDDINGS_DIR,
    EXPERIMENTS,
    FIGURES_DIR,
    LOGS_DIR,
    METHOD_DISPLAY,
    METHOD_KEYS,
    Experiment,
    setup_logging,
)


SITE_COLORS = {
    "DRIAMS-A": "#1f77b4",
    "DRIAMS-B": "#ff7f0e",
    "DRIAMS-C": "#2ca02c",
    "DRIAMS-D": "#d62728",
}

FIGURE_S2: tuple[tuple[str, str], ...] = (
    ("tsne", "figure_s2a_tsne_site"),
    ("umap", "figure_s2b_umap_site"),
)


def _embed_path(exp_key: str, method: str, kind: str) -> Path:
    return EMBEDDINGS_DIR / f"{exp_key}__{method}__{kind}.npy"


def _scatter(ax, coords: np.ndarray, labels: np.ndarray, color_map: dict) -> None:
    for value, color in color_map.items():
        mask = labels == value
        if not np.any(mask):
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=4,
            alpha=0.45,
            c=color,
            linewidths=0,
            rasterized=True,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.4)
        spine.set_color("0.6")


def _legend_handles(color_map: dict) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            markerfacecolor=color,
            markeredgecolor="none",
            markersize=8,
            label=value,
        )
        for value, color in color_map.items()
    ]


def render_site_grid(
    exp: Experiment,
    kind: str,
    out_stub: str,
    log,
    *,
    n_cols: int = 4,
) -> None:
    """Twelve ``kind`` embeddings of ``exp`` (uncorrected + 11 correctors) coloured by site.

    ``kind``: ``"tsne"`` or ``"umap"``. The panels are arranged
    ``ceil(12 / n_cols)`` rows by ``n_cols`` columns.
    """
    meta_path = EMBEDDINGS_DIR / f"{exp.key}__meta.csv"
    if not meta_path.exists():
        log.warning("[%s] no embeddings/meta; skipping", exp.key)
        return
    meta = pd.read_csv(meta_path)
    labels = meta["site"].to_numpy()

    panels: list[tuple[str, str, str]] = [("__pre__", "Uncorrected", "pre")]
    panels.extend((m, METHOD_DISPLAY[m], m) for m in METHOD_KEYS)

    n_total = len(panels)
    n_rows = (n_total + n_cols - 1) // n_cols

    cell_w = 3.4
    cell_h = 3.4
    fig_w = cell_w * n_cols + 0.8
    fig_h = cell_h * n_rows + 1.6
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h), squeeze=False,
        gridspec_kw={
            "left": 0.04,
            "right": 0.99,
            "top": 0.93,
            "bottom": 0.07,
            "hspace": 0.22,
            "wspace": 0.06,
        },
    )

    for idx, (key, display, file_token) in enumerate(panels):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]
        method_for_path = "pre" if key == "__pre__" else file_token
        embed_path = _embed_path(exp.key, method_for_path, kind)
        if not embed_path.exists():
            ax.text(0.5, 0.5, "missing",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        Z = np.load(embed_path)
        _scatter(ax, Z, labels, SITE_COLORS)
        weight = "bold" if key == "__pre__" else "normal"
        ax.set_title(display, fontsize=12, fontweight=weight, pad=4)

    for idx in range(n_total, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r, c].set_visible(False)

    fig.legend(
        handles=_legend_handles(SITE_COLORS),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=4, frameon=False, fontsize=12,
        title="Site", title_fontsize=12,
    )

    out_png = FIGURES_DIR / f"{out_stub}.png"
    out_pdf = FIGURES_DIR / f"{out_stub}.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s and %s", out_png, out_pdf)


def main() -> None:
    log = setup_logging("05_make_figures", LOGS_DIR / "05_make_figures.log")
    exp = next(e for e in EXPERIMENTS if e.key == "exp3_multi_species")
    for kind, out_stub in FIGURE_S2:
        render_site_grid(exp, kind, out_stub, log)


if __name__ == "__main__":
    main()
