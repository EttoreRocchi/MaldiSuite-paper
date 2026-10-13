"""Shared helpers for the leakage and bin-width experiments of the MaldiSuite paper.

Used by the leakage experiment (``leakage_ladder.py`` and ``leakage_curves.py``,
Supplementary Note S3), which measures how much several train/test leakage modes
inflate cross-validated performance compared with the leak-safe protocol of
MaldiSuite, and by the bin-width experiment (``bin_width_sensitivity.py``,
Supplementary Note S4), which repeats the DRIAMS-A classification benchmark at
several bin widths.

Both reuse the benchmark code of ``03_classification/_shared.py`` (classifier
factory, metrics, seeds), so their numbers are comparable with Supplementary
Figure S1. The only difference is the loader below, which
runs MaldiAMRKit's default preprocessing on the raw DRIAMS spectra and bins
them at an arbitrary width, keeping the isolate identifier as a CV group.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEEPKIT_PAPER = REPO / "03_classification"
sys.path.insert(0, str(DEEPKIT_PAPER))

import _shared

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "cache"
RESULTS_DIR = HERE / "results"
LOG_DIR = HERE / "logs"

SEED = _shared.SEED
N_FOLDS = 5
N_REPEATS = 3
MZ_MIN, MZ_MAX = 2000, 20000
SITE = "DRIAMS-A"

PAIRS: list[tuple[str, str]] = [
    ("Staphylococcus aureus", "oxacillin"),
    ("Escherichia coli", "ceftriaxone"),
    ("Klebsiella pneumoniae", "ceftriaxone"),
]

_REPLICATE_RE = re.compile(r"_MALDI\d+$")


def setup_logging(name: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / f"{name}.log"),
        ],
    )
    logging.getLogger("maldiamrkit").setLevel(logging.WARNING)
    return logging.getLogger(name)


def slug(species: str, drug: str) -> str:
    return _shared.slug(species, drug)


def _cache_paths(species: str, drug: str, bin_width: float, site: str) -> tuple[Path, Path]:
    key = f"{site}__{slug(species, drug)}"
    return CACHE_DIR / f"{key}__bw{bin_width:g}.npz", CACHE_DIR / f"{key}__meta.csv"


def _preprocess_and_bin(spec, bin_widths: tuple[float, ...]) -> list[np.ndarray]:
    from maldiamrkit.preprocessing import bin_spectrum, preprocess

    processed = preprocess(spec.raw)
    out = []
    for w in bin_widths:
        binned, _ = bin_spectrum(processed, bin_width=w, mz_min=MZ_MIN, mz_max=MZ_MAX)
        out.append(binned["intensity"].to_numpy(dtype=np.float32))
    return out


def load_pair(
    species: str,
    drug: str,
    *,
    bin_widths: tuple[float, ...] = (3,),
    site: str = SITE,
    n_jobs: int = 8,
    logger: logging.Logger | None = None,
) -> tuple[dict[float, np.ndarray], np.ndarray, pd.DataFrame]:
    """Load one species-drug pair of one DRIAMS site at the requested bin widths.

    Returns ``(X_by_width, y, meta)``. ``meta`` carries ``group`` (isolate
    identifier with any ``_MALDI<n>`` replicate suffix stripped) and
    ``replicate`` columns so that grouped cross-validation can be applied.
    Everything is cached under ``cache/`` after the first call.
    """
    log = logger or logging.getLogger("common")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    missing = [w for w in bin_widths if not _cache_paths(species, drug, w, site)[0].exists()]
    meta_path = _cache_paths(species, drug, bin_widths[0], site)[1]

    if not missing and meta_path.exists():
        meta = pd.read_csv(meta_path)
        X = {w: np.load(_cache_paths(species, drug, w, site)[0])["X"] for w in bin_widths}
        log.info("Loaded %s / %s from cache (%d spectra)", species, drug, len(meta))
        return X, meta["rs_label"].to_numpy(dtype=np.int64), meta

    from joblib import Parallel, delayed
    from maldiamrkit.data import DatasetLoader, DRIAMSLayout

    site_dir = _shared.driams_root() / site
    layout = DRIAMSLayout(site_dir, id_transform=str)
    raw_meta = layout.discover_metadata()
    raw_species, raw_drug = _shared._resolve_raw_names(raw_meta, species, drug)
    loader = DatasetLoader(layout, stage="raw", n_jobs=n_jobs)
    t0 = time.time()
    ds = loader.load(aggregate_by={"antibiotics": raw_drug, "species": raw_species})
    log.info("Loaded %d raw spectra for %s / %s in %.0f s", len(ds.spectra), species, drug, time.time() - t0)

    ids = [s.id for s in ds.spectra]
    meta = ds.meta.loc[ids].copy()
    drug_values = meta[raw_drug].astype(str).str.strip().str.upper()
    keep = drug_values.isin(["R", "S"]).to_numpy()
    spectra = [s for s, k in zip(ds.spectra, keep) if k]
    meta = meta.loc[keep].copy()
    meta["site"] = site
    meta["rs_label"] = (drug_values[keep] == "R").astype(int).to_numpy()
    meta["spectrum_id"] = meta.index.astype(str)
    meta["group"] = [_REPLICATE_RE.sub("", i) for i in meta["spectrum_id"]]
    meta["replicate"] = [
        (m.group(0)[1:] if (m := _REPLICATE_RE.search(i)) else "") for i in meta["spectrum_id"]
    ]
    meta = meta[["spectrum_id", "group", "replicate", "site", "rs_label"]].reset_index(drop=True)

    t0 = time.time()
    all_widths = tuple(sorted(set(bin_widths) | set(missing)))
    rows = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_preprocess_and_bin)(s, all_widths) for s in spectra
    )
    log.info("Preprocessed + binned %d spectra at widths %s in %.0f s", len(rows), all_widths, time.time() - t0)

    X: dict[float, np.ndarray] = {}
    for j, w in enumerate(all_widths):
        X[w] = np.vstack([r[j] for r in rows]).astype(np.float32)
        np.savez_compressed(_cache_paths(species, drug, w, site)[0], X=X[w])
    meta.to_csv(meta_path, index=False)
    n_groups = meta["group"].nunique()
    n_rep = int((meta.groupby("group").size() > 1).sum())
    log.info(
        "%s / %s: %d spectra, %d isolates, %d isolates with >1 replicate (%.1f%%), R=%d S=%d",
        species, drug, len(meta), n_groups, n_rep, 100 * n_rep / max(1, n_groups),
        int(meta["rs_label"].sum()), int((meta["rs_label"] == 0).sum()),
    )
    return {w: X[w] for w in bin_widths}, meta["rs_label"].to_numpy(dtype=np.int64), meta


def grouped_folds(y: np.ndarray, groups: np.ndarray, repeat: int, n_folds: int = N_FOLDS):
    """Stratified, isolate-grouped folds (what ``CaseGroupedKFold`` wraps)."""
    from sklearn.model_selection import StratifiedGroupKFold

    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED + repeat * 100)
    return list(cv.split(np.zeros(len(y)), y, groups=groups))


def stratified_folds(y: np.ndarray, repeat: int, n_folds: int = N_FOLDS):
    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED + repeat * 100)
    return list(cv.split(np.zeros(len(y)), y))


def make_mlp(input_dim: int, bin_width: float, run_index: int, *, quick: bool = False,
             disable_warping: bool = False, input_transform: str | None = None):
    """MLP+attention with the paper's training recipe (see ``_shared.build_classifier``)."""
    return _shared.build_classifier(
        "mlp_attn",
        bin_width=int(round(bin_width)),
        input_dim=int(input_dim),
        random_state=SEED + run_index,
        quick=quick,
        disable_warping=disable_warping,
        input_transform=input_transform,
    )


def score(clf, X_te: np.ndarray, y_te: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    proba = clf.predict_proba(X_te)
    proba_r = proba[:, 1] if proba.shape[1] == 2 else proba.max(axis=1)
    y_pred = clf.predict(X_te)
    return _shared.compute_fold_metrics(y_te, y_pred, proba_r), proba_r


def dataset_summary(meta: pd.DataFrame) -> dict:
    g = meta.groupby("group").size()
    return {
        "n_spectra": int(len(meta)),
        "n_isolates": int(len(g)),
        "n_isolates_with_replicates": int((g > 1).sum()),
        "n_resistant": int(meta["rs_label"].sum()),
        "n_susceptible": int((meta["rs_label"] == 0).sum()),
    }


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
