"""Figure 1b: one DRIAMS-A S. aureus spectrum (oxacillin-resistant) before and after preprocessing.

Top panel: the raw spectrum as stored in DRIAMS-A, with its baseline drift.
Bottom panel: the same spectrum after the MaldiAMRKit default pipeline
(clip negatives, square-root variance stabilisation, Savitzky-Golay
smoothing, SNIP baseline removal, trimming to 2-20 kDa, TIC
normalisation) and 3-Da binning, i.e. the 6000-bin vector that the
MaldiDeepKit classifiers take as input.

Both panels carry a zoom inset on 4000-4200 Da: a continuous trace in
the raw panel, one stick per bin in the binned one.

Output: ``figure1_panelA_preprocessing.{pdf,png}`` next to this script.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

from maldiamrkit import MaldiSpectrum
from maldiamrkit.preprocessing import PreprocessingPipeline, bin_spectrum

HERE = Path(__file__).resolve().parent

SPECTRUM = "DRIAMS-A/raw/2017/5c4b911b-44b1-42ee-a021-6cc658d69b94_MALDI1.txt"

RAW_COLOR = "#1f4e79"
FINAL_COLOR = "#d97706"

INSET_LO, INSET_HI = 4000, 4200


def _set_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 0.5,
            "savefig.dpi": 350,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _add_inset(parent_ax, *, kind: str, x, y, color: str) -> None:
    """Attach a zoom inset to the upper right of ``parent_ax``.

    ``kind="line"`` draws a continuous trace (raw), ``kind="stems"`` one
    stick per bin (binned). The source window is shaded over the full
    height of the parent panel and joined to the inset by two lines.
    """
    axins = parent_ax.inset_axes([0.72, 0.30, 0.25, 0.55])
    if kind == "line":
        axins.plot(x, y, color=color, linewidth=0.6)
    elif kind == "stems":
        axins.vlines(x, 0.0, y, color=color, linewidth=0.9)
    else:
        raise ValueError(f"unknown inset kind: {kind}")
    axins.set_xlim(INSET_LO - 3, INSET_HI + 3)
    axins.set_ylim(0.0, float(y.max()) * 1.10)
    axins.set_xticks([INSET_LO, INSET_HI])
    axins.set_xticklabels([str(INSET_LO), str(INSET_HI)], fontsize=6)
    axins.set_yticks([])
    axins.tick_params(axis="x", length=2, pad=1)
    for spine in axins.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#444444")
    parent_ymin, parent_ymax = parent_ax.get_ylim()
    indicator = parent_ax.indicate_inset(
        bounds=(INSET_LO, parent_ymin, INSET_HI - INSET_LO, parent_ymax - parent_ymin),
        inset_ax=axins,
        edgecolor="#444444",
        linewidth=0.6,
        alpha=0.7,
    )
    for conn in indicator.connectors:
        conn.set_visible(False)
    for inset_corner, source_corner in [
        ((0.0, 0.0), (INSET_HI, parent_ymin)),
        ((0.0, 1.0), (INSET_HI, parent_ymax)),
    ]:
        cp = ConnectionPatch(
            xyA=inset_corner, coordsA=axins.transAxes,
            xyB=source_corner, coordsB=parent_ax.transData,
            arrowstyle="-",
            edgecolor="#444444",
            linewidth=0.6,
            alpha=0.7,
            zorder=4.99,
        )
        parent_ax.add_artist(cp)


def main() -> None:
    root = os.environ.get("DRIAMS_ROOT")
    if not root:
        raise SystemExit("Set the DRIAMS_ROOT environment variable to the directory that holds DRIAMS-A/.")
    spectrum_path = Path(root) / SPECTRUM
    if not spectrum_path.exists():
        raise SystemExit(f"DRIAMS spectrum not found at {spectrum_path}")

    spec = MaldiSpectrum(spectrum_path)
    raw_df = spec._raw

    full = PreprocessingPipeline.default()
    prep_df = full(raw_df.copy())
    binned_df, _ = bin_spectrum(
        prep_df, mz_min=2000, mz_max=20000, bin_width=3, method="uniform"
    )
    bin_centres = binned_df["mass"].astype(float).to_numpy() + 1.5
    bin_intens = binned_df["intensity"].astype(float).to_numpy()

    _set_publication_style()
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(3.5, 2.5), sharex=True)

    raw_x = raw_df["mass"].to_numpy()
    raw_y = raw_df["intensity"].to_numpy()
    axes[0].plot(raw_x, raw_y, color=RAW_COLOR)
    axes[0].set_ylabel("Raw")
    axes[0].set_title(
        r"DRIAMS-A / $\it{S.\,aureus}$ (oxacillin-R)", pad=4
    )
    axes[0].set_yticks([])
    axes[0].set_ylim(0.0, float(raw_y.max()) * 1.04)

    axes[1].plot(bin_centres, bin_intens, color=FINAL_COLOR)
    axes[1].set_ylabel("Binned\n(6000 bins)")
    axes[1].set_xlabel("m/z (Da)")
    axes[1].set_yticks([])
    axes[1].set_ylim(0.0, float(bin_intens.max()) * 1.04)

    axes[1].set_xlim(2000, 20000)
    axes[1].set_xticks([2000, 5000, 10000, 15000, 20000])
    for ax in axes:
        ax.tick_params(axis="x", pad=2)

    raw_mask = (raw_x >= INSET_LO) & (raw_x <= INSET_HI)
    bin_mask = (bin_centres >= INSET_LO) & (bin_centres <= INSET_HI)
    _add_inset(
        axes[0], kind="line",
        x=raw_x[raw_mask], y=raw_y[raw_mask],
        color=RAW_COLOR,
    )
    _add_inset(
        axes[1], kind="stems",
        x=bin_centres[bin_mask], y=bin_intens[bin_mask],
        color=FINAL_COLOR,
    )

    fig.tight_layout()
    pdf = HERE / "figure1_panelA_preprocessing.pdf"
    png = HERE / "figure1_panelA_preprocessing.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print(f"wrote {pdf.name} and {png.name}")


if __name__ == "__main__":
    main()
