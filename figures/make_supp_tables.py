"""Compute Supplementary Tables S9 (leakage experiment, paired protocols) and S10 (bin-width sensitivity)
from the per-fold results in ``results/leakage_and_bin_width/``.

Writes ``table_s9_leakage.csv`` and ``table_s10_binwidth.csv`` under ``output/tables/`` (override the
directory with the ``OUTPUT_DIR`` environment variable).

Table S9 has one row per (section, level, species, drug, metric):

* ``reference``: mean performance of the leak-safe protocol (``value``);
* ``feature_selection``, ``preprocessing``, ``decision_threshold``: mean difference, leaky minus leak-safe
  protocol, over fits that share subsample, repeat and fold (``value``), and the number of those fits in
  which the leaky protocol scored higher (``n_higher`` out of ``n``);
* ``replicates``: ungrouped minus isolate-grouped folds on subsamples of 300 isolates, averaged per
  subsample, so ``n_higher`` and ``n`` count subsamples; ``test_share_with_twin`` is the share of test
  spectra whose replicate sits in the training split.

``level`` is the number of isolates, or ``full`` for the whole cohort (its size in ``n_isolates``); in the
``replicates`` section it is the share of isolates carrying a technical replicate.

    python figures/make_supp_tables.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("OUTPUT_DIR", REPO / "output")) / "tables"
EXP_RESULTS = REPO / "results" / "leakage_and_bin_width"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PAIR_ORDER = [("Staphylococcus aureus", "oxacillin"), ("Escherichia coli", "ceftriaxone"),
              ("Klebsiella pneumoniae", "ceftriaxone")]
METRICS = ("auroc", "balanced_accuracy")
LEAK_PAIRS = [
    ("feature_selection", "feature_selection_safe", "feature_selection_leak", METRICS),
    ("preprocessing", "reference", "transductive_preproc", METRICS),
    ("decision_threshold", "reference", "test_fold_threshold", ("balanced_accuracy",)),
]
SIZES_ORDER = (100, 200, 400, 800, 1600)
BINWIDTH_METRICS = ("auroc", "balanced_accuracy", "mcc", "vme")


def _paired(df: pd.DataFrame, safe: str, leak: str, metric: str, keys: list[str]) -> tuple[float, int, int]:
    a = df[df.protocol == safe].set_index(keys)[metric]
    b = df[df.protocol == leak].set_index(keys)[metric]
    d = (b - a.reindex(b.index)).dropna()
    return float(d.mean()), int((d > 0).sum()), int(len(d))


def _n_isolates(species: str, drug: str) -> int | None:
    info = EXP_RESULTS / f"leakage_ladder__{species.replace(' ', '_')}__{drug}__dataset.json"
    return json.loads(info.read_text())["n_isolates"] if info.exists() else None


def leakage_table(size_csv: Path, dose_csv: Path | None, full_csv: Path) -> pd.DataFrame:
    size = pd.read_csv(size_csv)
    full = pd.read_csv(full_csv)
    dose = pd.read_csv(dose_csv) if dose_csv is not None and dose_csv.exists() else None
    n_full = {pair: _n_isolates(*pair) for pair in PAIR_ORDER}
    rows: list[dict] = []

    def add(section: str, level: int | float | str, pair: tuple[str, str], metric: str, value: float,
            n_higher: int | None = None, n: int | None = None, **extra: object) -> None:
        rows.append({"section": section, "level": level, "species": pair[0], "drug": pair[1],
                     "metric": metric, "value": value, "n_higher": n_higher, "n": n, **extra})

    def sub(pair: tuple[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        sp, dr = pair
        return size[(size.species == sp) & (size.drug == dr)], full[(full.species == sp) & (full.drug == dr)]

    for n in SIZES_ORDER:
        for pair in PAIR_ORDER:
            s, _ = sub(pair)
            g = s[(s.level == n) & (s.protocol == "reference")]
            for m in METRICS if len(g) else ():
                add("reference", n, pair, m, float(g[m].mean()))
    for pair in PAIR_ORDER:
        _, f = sub(pair)
        g = f[f.protocol == "reference"]
        for m in METRICS:
            add("reference", "full", pair, m, float(g[m].mean()), n_isolates=n_full[pair])

    for section, safe, leak, metrics in LEAK_PAIRS:
        for n in SIZES_ORDER:
            for pair in PAIR_ORDER:
                s, _ = sub(pair)
                g = s[s.level == n]
                for m in metrics if len(g) else ():
                    add(section, n, pair, m, *_paired(g, safe, leak, m, ["subsample", "repeat", "fold"]))
        for pair in PAIR_ORDER:
            _, f = sub(pair)
            for m in metrics:
                add(section, "full", pair, m, *_paired(f, safe, leak, m, ["repeat", "fold"]),
                    n_isolates=n_full[pair])

    if dose is not None:
        for lvl in sorted(dose.level.unique()):
            for pair in PAIR_ORDER:
                g = dose[(dose.level == lvl) & (dose.species == pair[0]) & (dose.drug == pair[1])]
                if len(g) == 0:
                    continue
                twin = float(g[g.protocol == "ungrouped"]["test_share_with_twin_in_train"].mean())
                for m in METRICS:
                    per_sub = g.groupby(["subsample", "protocol"])[m].mean().unstack()
                    d = per_sub["ungrouped"] - per_sub["grouped"]
                    add("replicates", lvl, pair, m, float(d.mean()), int((d > 0).sum()), int(len(d)),
                        test_share_with_twin=twin)
        for pair in PAIR_ORDER:
            _, f = sub(pair)
            for m in METRICS:
                d = f[f.protocol == "replicate_leak"][m].mean() - f[f.protocol == "reference"][m].mean()
                add("replicates", "full", pair, m, float(d), n_isolates=n_full[pair])

    out = pd.DataFrame(rows)
    for col in ("n_higher", "n", "n_isolates"):
        out[col] = out[col].astype("Int64")
    return out


def binwidth_table(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    rows: list[dict] = []
    for sp, dr in PAIR_ORDER:
        sub = df[(df.species == sp) & (df.drug == dr)]
        for w in sorted(sub.bin_width.unique()):
            g = sub[sub.bin_width == w]
            row = {"species": sp, "drug": dr, "bin_width": w, "n_bins": int(g["n_bins"].iloc[0])}
            for m in BINWIDTH_METRICS:
                row[f"{m}_mean"] = float(g[m].mean())
                row[f"{m}_std"] = float(g[m].std())
            rows.append(row)
    return pd.DataFrame(rows)


def _pick(name: str) -> Path | None:
    for cand in (f"{name}.csv", f"{name}_quick.csv"):
        if (EXP_RESULTS / cand).exists():
            if cand.endswith("_quick.csv"):
                print(f"WARNING: using {cand}")
            return EXP_RESULTS / cand
    return None


def main() -> None:
    size_csv = EXP_RESULTS / "leakage_size_curve.csv"
    full_csv = EXP_RESULTS / "leakage_ladder.csv"
    if size_csv.exists() and full_csv.exists():
        table = leakage_table(size_csv, EXP_RESULTS / "leakage_dose_curve.csv", full_csv)
        table.to_csv(DATA_DIR / "table_s9_leakage.csv", index=False)
    else:
        print("WARNING: leakage curve results missing; Table S9 not rebuilt")
    if (p := _pick("bin_width_sensitivity")) is not None:
        binwidth_table(p).to_csv(DATA_DIR / "table_s10_binwidth.csv", index=False)
    print("tables written to", DATA_DIR)


if __name__ == "__main__":
    main()
