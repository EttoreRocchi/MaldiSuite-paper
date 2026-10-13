"""Supplementary Figures S3, S4 and S5 of the MaldiSuite paper.

* Figure S3: per-species t-SNE of the eleven MaldiBatchKit correctors, 6 x 6 grid
  (embeddings written by ``02_batch_correction``).
* Figure S4: batch-effect metrics per corrector, from
  ``results/batch_correction/supp_table_s3.csv`` (the values of Supplementary Tables S4-S7).
* Figure S5: bin-width sensitivity of the DRIAMS-A benchmark.

The leakage experiment has no figure; its results are Supplementary Table S9
(``make_supp_tables.py``).

Run from the repository root:

    python figures/make_supp_figures.py
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
BATCH_RESULTS = REPO / "results" / "batch_correction"
EXP_RESULTS = REPO / "results" / "leakage_and_bin_width"
FIG_DIR.mkdir(parents=True, exist_ok=True)
BATCHKIT_EMB = Path(os.environ.get("BATCHKIT_EMBEDDINGS", REPO / "02_batch_correction" / "results" / "embeddings"))

INK, INK_2 = "#222222", "#555555"
POINT = "#0072B2"
PAIR_ORDER = [("Staphylococcus aureus", "oxacillin"), ("Escherichia coli", "ceftriaxone"),
              ("Klebsiella pneumoniae", "ceftriaxone")]
PAIR_DISPLAY = {
    ("Staphylococcus aureus", "oxacillin"): "$S.\\,aureus$ / oxacillin",
    ("Escherichia coli", "ceftriaxone"): "$E.\\,coli$ / ceftriaxone",
    ("Klebsiella pneumoniae", "ceftriaxone"): "$K.\\,pneumoniae$ / ceftriaxone",
}
PAIR_COLORS = {
    ("Staphylococcus aureus", "oxacillin"): "#0072B2",
    ("Escherichia coli", "ceftriaxone"): "#009E73",
    ("Klebsiella pneumoniae", "ceftriaxone"): "#D55E00",
}
SITE_COLORS = {"DRIAMS-A": "#1f77b4", "DRIAMS-B": "#ff7f0e", "DRIAMS-C": "#2ca02c", "DRIAMS-D": "#d62728"}

EXP_DISPLAY = {
    "exp1_saureus": "$S.\\,aureus$ / oxacillin",
    "exp1_ecoli": "$E.\\,coli$ / ceftriaxone",
    "exp1_kpneumoniae": "$K.\\,pneumoniae$ / ceftriaxone",
    "exp3_multi_species": "$E.\\,coli$ + $K.\\,pneumoniae$ / ceftriaxone (species protected)",
}
SPECIES_ROWS = [("exp1_saureus", "$S.\\,aureus$"), ("exp1_ecoli", "$E.\\,coli$"),
                ("exp1_kpneumoniae", "$K.\\,pneumoniae$")]
EXP_ORDER = ["exp1_saureus", "exp1_ecoli", "exp1_kpneumoniae", "exp3_multi_species"]
METHOD_ORDER = ["combat_johnson", "combat_fortin", "combat_chen", "limma", "harmony", "median_centering",
                "zscore_per_batch", "reference_scaling", "batch_aware_warping", "quality_weighted_combat",
                "species_aware_combat"]
METHOD_SHORT = {
    "__pre__": "Uncorrected\n ",
    "combat_johnson": "ComBat\n(Johnson)",
    "combat_fortin": "ComBat\n(Fortin)",
    "combat_chen": "CovBat\n(Chen)",
    "limma": "Limma\n ",
    "harmony": "Harmony\n ",
    "median_centering": "Median\ncentering",
    "zscore_per_batch": "Z-score\nper batch",
    "reference_scaling": "Reference\nscaling",
    "batch_aware_warping": "Batch-aware\nwarping",
    "quality_weighted_combat": "Quality-weighted\nComBat",
    "species_aware_combat": "Species-aware\nComBat",
}


def _style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7.5, "axes.linewidth": 0.6, "pdf.fonttype": 42,
        "ps.fonttype": 42, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    })


def _site_handles():
    return [plt.Line2D([], [], marker="o", ls="", ms=5, color=c, label=s) for s, c in SITE_COLORS.items()]


def make_per_species_tsne(out: Path) -> None:
    columns = ["__pre__"] + METHOD_ORDER
    blocks = [columns[:6], columns[6:]]
    fig = plt.figure(figsize=(7.2, 8.6))
    grids = [fig.add_gridspec(3, 6, left=0.10, right=0.99, top=0.905, bottom=0.525, wspace=0.08, hspace=0.08),
             fig.add_gridspec(3, 6, left=0.10, right=0.99, top=0.445, bottom=0.065, wspace=0.08, hspace=0.08)]
    for gs, block in zip(grids, blocks):
        for r, (exp_key, species_label) in enumerate(SPECIES_ROWS):
            meta = pd.read_csv(BATCHKIT_EMB / f"{exp_key}__meta.csv")
            colors = np.array([SITE_COLORS[s] for s in meta["site"]])
            order = np.random.default_rng(0).permutation(len(meta))
            for c, key in enumerate(block):
                ax = fig.add_subplot(gs[r, c])
                token = "pre" if key == "__pre__" else key
                Z = np.load(BATCHKIT_EMB / f"{exp_key}__{token}__tsne.npy")
                ax.scatter(Z[order, 0], Z[order, 1], s=0.9, c=colors[order], alpha=0.5, linewidths=0,
                           rasterized=True)
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_linewidth(0.5)
                    sp.set_color("#888888")
                if r == 0:
                    ax.set_title(METHOD_SHORT[key], fontsize=7.2, pad=3,
                                 fontweight="bold" if key == "__pre__" else "normal")
                if c == 0:
                    ax.set_ylabel(species_label, fontsize=8, labelpad=4)
    fig.legend(handles=_site_handles(), loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.545, 0.995), columnspacing=1.6, handletextpad=0.3)
    fig.savefig(out, format="pdf", dpi=400, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def make_batch_metrics(out: Path) -> None:
    df = pd.read_csv(BATCH_RESULTS / "supp_table_s3.csv")
    metrics = [
        ("silhouette_batch", "silhouette on batch labels\n(closer to 0 is better)", True),
        ("kbet_acceptance", "kBET acceptance rate\n(higher is better)", False),
        ("lisi_batch_normalized", "normalised batch LISI\n(higher is better)", False),
    ]
    n_rows = len(EXP_ORDER)
    fig, axes = plt.subplots(n_rows, len(metrics), figsize=(7.2, 10.2), sharey="row")
    fig.subplots_adjust(left=0.215, right=0.985, top=0.935, bottom=0.055, hspace=0.72, wspace=0.18)
    for r, exp in enumerate(EXP_ORDER):
        sub = df[df.experiment == exp]
        pre = sub[sub.stage == "pre"].iloc[0]
        post = sub[sub.stage == "post"].set_index("method").reindex(METHOD_ORDER)
        labels = post["method_display"].fillna("").tolist()
        ypos = np.arange(len(METHOD_ORDER))[::-1]
        for c, (col, xlabel, abs_zero) in enumerate(metrics):
            ax = axes[r, c]
            vals = post[col].to_numpy(dtype=float)
            ax.axvline(pre[col], color=INK_2, lw=0.8, ls="--", zorder=1)
            if abs_zero:
                ax.axvline(0, color="#AAAAAA", lw=0.5, zorder=0)
            ax.scatter(vals, ypos, s=20, color=POINT, zorder=3, linewidths=0)
            ax.set_yticks(ypos)
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_ylim(-0.8, len(METHOD_ORDER) - 0.2)
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
            ax.tick_params(axis="x", labelsize=6.8, pad=2)
            ax.set_xlabel(xlabel, fontsize=6.8, labelpad=3, color=INK_2)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.grid(axis="x", lw=0.3, color="#DDDDDD")
            ax.set_axisbelow(True)
        pos = axes[r, 0].get_position()
        fig.text(0.012, pos.y1 + 0.012, EXP_DISPLAY[exp], ha="left", va="bottom", fontsize=9)
    handles = [plt.Line2D([], [], ls="--", color=INK_2, lw=0.8, label="uncorrected cohort"),
               plt.Line2D([], [], marker="o", ls="", color=POINT, ms=4.5, label="after correction")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.6, 0.995),
               columnspacing=2.0)
    fig.savefig(out, format="pdf")
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)


def make_bin_width(out: Path, results_csv: Path) -> None:
    df = pd.read_csv(results_csv)
    metrics = [("auroc", "AUROC"), ("balanced_accuracy", "Balanced accuracy"), ("mcc", "MCC"), ("vme", "VME")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.2, 2.0))
    widths = sorted(df.bin_width.unique())
    for ax, (metric, title) in zip(axes, metrics):
        for (sp, dr) in PAIR_ORDER:
            sub = df[(df.species == sp) & (df.drug == dr)]
            if sub.empty:
                continue
            g = sub.groupby("bin_width")[metric]
            m, s = g.mean().reindex(widths), g.std().reindex(widths)
            col = PAIR_COLORS[(sp, dr)]
            ax.errorbar(widths, m, yerr=s, fmt="-o", ms=3.5, lw=1.0, color=col, ecolor=col, elinewidth=0.7,
                        capsize=1.8, label=PAIR_DISPLAY[(sp, dr)])
        ax.set_xscale("log")
        ax.set_xticks(widths)
        ax.set_xticklabels([f"{w:g}" for w in widths])
        ax.minorticks_off()
        ax.axvline(3, color="#AAAAAA", lw=0.6, ls=":")
        ax.set_xlabel("bin width (Da)")
        ax.set_title(title, fontsize=8, pad=4)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        ax.grid(axis="y", lw=0.3, color="#DDDDDD")
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.30, wspace=0.28)
    fig.savefig(out, format="pdf")
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)


def _pick(name: str) -> Path | None:
    for cand in (f"{name}.csv", f"{name}_quick.csv"):
        p = EXP_RESULTS / cand
        if p.exists():
            if cand != f"{name}.csv":
                print(f"WARNING: {name}.csv missing, using {cand}")
            return p
    print(f"WARNING: no results for {name}; figure skipped")
    return None


def main() -> None:
    _style()
    make_per_species_tsne(FIG_DIR / "figure_s3_per_species_tsne.pdf")
    make_batch_metrics(FIG_DIR / "figure_s4_batch_metrics.pdf")
    if (p := _pick("bin_width_sensitivity")) is not None:
        make_bin_width(FIG_DIR / "figure_s5_bin_width.pdf", p)
    print("wrote supplementary figures to", FIG_DIR)


if __name__ == "__main__":
    main()
