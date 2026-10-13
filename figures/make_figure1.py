"""Figure 1 of the MaldiSuite paper.

Renders the vector panels of Figure 1:

* ``figure1_panelC_batch.pdf``: panel c, t-SNE before and after harmonisation, coloured by site
* ``figure1_panelE_auroc.pdf``: panel d, AUROC per classifier, site and species-drug pair

Panel a is a TikZ drawing in the manuscript sources; panel b
(preprocessing) is rendered by ``01_preprocessing/make_figure1_panelB.py``.

Run from the repository root:

    python figures/make_figure1.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FIG_DIR = Path(os.environ.get("OUTPUT_DIR", REPO / "output")) / "figures"
DATA_DIR = REPO / "results" / "classification"
FIG_DIR.mkdir(parents=True, exist_ok=True)
BATCHKIT_EMB = Path(os.environ.get("BATCHKIT_EMBEDDINGS", REPO / "02_batch_correction" / "results" / "embeddings"))

CLASSIFIER_ORDER = ["mlp_attn", "cnn", "resnet", "transformer", "ensemble_mlp_cnn"]
CLASSIFIER_DISPLAY = {
    "mlp_attn": "MLP + attention",
    "cnn": "1-D CNN",
    "resnet": "ResNet",
    "transformer": "1-D ViT",
    "ensemble_mlp_cnn": "Ensemble (MLP+CNN)",
}
SITE_COLORS = {"DRIAMS-A": "#1f77b4", "DRIAMS-B": "#ff7f0e", "DRIAMS-C": "#2ca02c", "DRIAMS-D": "#d62728"}
PAIR_ORDER = [("Staphylococcus aureus", "oxacillin"), ("Escherichia coli", "ceftriaxone"), ("Klebsiella pneumoniae", "ceftriaxone")]
PAIR_DISPLAY = {
    ("Staphylococcus aureus", "oxacillin"): "$S.\\,aureus$ / oxacillin",
    ("Escherichia coli", "ceftriaxone"): "$E.\\,coli$ / ceftriaxone",
    ("Klebsiella pneumoniae", "ceftriaxone"): "$K.\\,pneumoniae$ / ceftriaxone",
}

INK = "#222222"
INK_2 = "#555555"


def _style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    })


def make_batch_panel(out: Path, *, method: str = "harmony", method_display: str = "Harmony") -> None:
    meta = pd.read_csv(BATCHKIT_EMB / "exp3_multi_species__meta.csv")
    pre = np.load(BATCHKIT_EMB / "exp3_multi_species__pre__tsne.npy")
    post = np.load(BATCHKIT_EMB / f"exp3_multi_species__{method}__tsne.npy")
    assert len(meta) == len(pre) == len(post)
    sites = meta["site"].to_numpy()
    rng = np.random.default_rng(0)
    order = rng.permutation(len(meta))

    fig, axes = plt.subplots(1, 2, figsize=(3.6, 2.2))
    for ax, Z, title in zip(axes, (pre, post), ("Uncorrected", f"After {method_display}")):
        colors = np.array([SITE_COLORS[s] for s in sites])
        ax.scatter(Z[order, 0], Z[order, 1], s=1.6, c=colors[order], alpha=0.55, linewidths=0, rasterized=True)
        ax.set_title(title, fontsize=8, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
            sp.set_color(INK_2)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=4, color=SITE_COLORS[s], label=s.replace("DRIAMS-", "site "))
               for s in SITE_COLORS]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7,
               handletextpad=0.3, columnspacing=1.2, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.12, wspace=0.08)
    fig.savefig(out, format="pdf", dpi=350)
    fig.savefig(out.with_suffix(".png"), dpi=350)
    plt.close(fig)


def make_auroc_panel(out: Path) -> None:
    """Three heat maps (one per species-drug pair): classifiers x DRIAMS sites, mean AUROC."""
    summ = pd.read_csv(DATA_DIR / "summary.csv")
    sites = ["DRIAMS-A", "DRIAMS-B", "DRIAMS-C", "DRIAMS-D"]
    cmap = mpl.colormaps["Blues"]
    norm = mpl.colors.Normalize(vmin=0.5, vmax=1.0)

    fig, axes = plt.subplots(1, len(PAIR_ORDER), figsize=(7.2, 2.0), sharey=True,
                             gridspec_kw={"wspace": 0.10})
    im = None
    for j, ((sp, dr), ax) in enumerate(zip(PAIR_ORDER, axes)):
        M = np.full((len(CLASSIFIER_ORDER), len(sites)), np.nan)
        S = np.full_like(M, np.nan)
        for k, site in enumerate(sites):
            for i, clf in enumerate(CLASSIFIER_ORDER):
                row = summ[(summ.species == sp) & (summ.drug == dr) & (summ.site == site) & (summ.classifier == clf)]
                if len(row):
                    M[i, k] = float(row["auroc_mean"].iloc[0])
                    S[i, k] = float(row["auroc_std"].iloc[0])
        im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
        for i in range(M.shape[0]):
            for k in range(M.shape[1]):
                if np.isnan(M[i, k]):
                    ax.text(k, i, "n/a", ha="center", va="center", fontsize=6.5, color=INK_2)
                    continue
                dark = M[i, k] > 0.78
                ax.text(k, i - 0.10, f"{M[i, k]:.2f}", ha="center", va="center", fontsize=7.2,
                        color="white" if dark else INK)
                ax.text(k, i + 0.27, f"$\\pm${S[i, k]:.2f}", ha="center", va="center", fontsize=5.2,
                        color="#DDE7F3" if dark else INK_2)
        for k in range(M.shape[1]):
            col = M[:, k]
            if np.all(np.isnan(col)):
                continue
            i = int(np.nanargmax(col))
            ax.text(k - 0.44, i - 0.32, "\u2605", ha="left", va="center", fontsize=6,
                    color="white" if M[i, k] > 0.78 else INK)
        for k in range(1, len(sites)):
            ax.axvline(k - 0.5, color="white", lw=1.5)
        for i in range(1, len(CLASSIFIER_ORDER)):
            ax.axhline(i - 0.5, color="white", lw=1.5)
        ax.set_title(PAIR_DISPLAY[(sp, dr)], fontsize=7.5, pad=5)
        ax.set_xticks(range(len(sites)))
        ax.set_xticklabels([s.replace("DRIAMS-", "") for s in sites], fontsize=7)
        ax.set_xlabel("DRIAMS site", fontsize=7, labelpad=2)
        ax.tick_params(length=0, pad=2)
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
    axes[0].set_yticks(range(len(CLASSIFIER_ORDER)))
    axes[0].set_yticklabels([CLASSIFIER_DISPLAY[c] for c in CLASSIFIER_ORDER], fontsize=7)
    fig.subplots_adjust(left=0.165, right=0.925, top=0.86, bottom=0.19)
    cax = fig.add_axes([0.945, 0.19, 0.012, 0.67])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("mean AUROC", fontsize=6.5, labelpad=2)
    cbar.ax.tick_params(labelsize=6, length=2, width=0.5)
    cbar.outline.set_linewidth(0.5)
    fig.savefig(out, format="pdf")
    fig.savefig(out.with_suffix(".png"), dpi=350)
    plt.close(fig)


def main() -> None:
    """Render the data panels c and d of Figure 1; panel a is a TikZ drawing in the
    manuscript sources and panel b is rendered by ``01_preprocessing/make_figure1_panelB.py``."""
    _style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    make_batch_panel(FIG_DIR / "figure1_panelC_batch.pdf")
    make_auroc_panel(FIG_DIR / "figure1_panelE_auroc.pdf")
    print("wrote panels to", FIG_DIR)


if __name__ == "__main__":
    main()
