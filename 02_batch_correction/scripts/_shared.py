"""Paths, constants and DRIAMS loaders for the batch-correction benchmark.

The benchmark runs the 11 MaldiBatchKit correctors on DRIAMS-A/B/C/D
spectra preprocessed with the MaldiAMRKit default pipeline. Experiments
(see :data:`EXPERIMENTS`):

- ``exp1_<species>``: one species-drug pair at a time across the four
  DRIAMS sites, so the site effect is the only structure to remove
  (S. aureus / oxacillin, E. coli / ceftriaxone,
  K. pneumoniae / ceftriaxone).
- ``exp3_multi_species``: E. coli + K. pneumoniae (both ceftriaxone)
  across the four sites, with species as a biological covariate for the
  correctors that accept one.
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PAPER_ROOT = HERE.parent

CACHE_DIR = PAPER_ROOT / "cache"
DATA_CACHE = CACHE_DIR / "data"
ESTIMATOR_CACHE = CACHE_DIR / "estimators"
RESULTS_DIR = PAPER_ROOT / "results"
CORRECTED_DIR = RESULTS_DIR / "corrected"
EMBEDDINGS_DIR = RESULTS_DIR / "embeddings"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = PAPER_ROOT / "figures"
LOGS_DIR = PAPER_ROOT / "logs"

for _d in (
    DATA_CACHE,
    ESTIMATOR_CACHE,
    CORRECTED_DIR,
    EMBEDDINGS_DIR,
    METRICS_DIR,
    FIGURES_DIR,
    LOGS_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)


DRIAMS_SITES: tuple[str, ...] = ("DRIAMS-A", "DRIAMS-B", "DRIAMS-C", "DRIAMS-D")
SEED = 42

BIN_WIDTH = 3
MZ_MIN = 2000
MZ_MAX = 20000
N_BINS = (MZ_MAX - MZ_MIN) // BIN_WIDTH

PAIRS: list[tuple[str, str]] = [
    ("Staphylococcus aureus", "oxacillin"),
    ("Escherichia coli", "ceftriaxone"),
    ("Klebsiella pneumoniae", "ceftriaxone"),
]

def slug(species: str, drug: str) -> str:
    """Filesystem-safe identifier for a species-drug pair."""
    return f"{species.replace(' ', '_')}__{drug}"


def driams_root() -> Path:
    """Directory holding DRIAMS-A/, DRIAMS-B/, DRIAMS-C/ and DRIAMS-D/, from ``DRIAMS_ROOT``."""
    root = os.environ.get("DRIAMS_ROOT")
    if not root:
        raise SystemExit(
            "Set the DRIAMS_ROOT environment variable to the directory that "
            "holds DRIAMS-A/, DRIAMS-B/, DRIAMS-C/ and DRIAMS-D/."
        )
    return Path(root)


@dataclass(frozen=True)
class Experiment:
    """One benchmark dataset: which DRIAMS pairs / sites we union together.

    Attributes
    ----------
    key : str
        Short identifier used in filenames.
    title : str
        Human-readable description for plots / captions.
    pairs : tuple[tuple[str, str], ...]
        Species-drug pairs to union into the dataset.
    sites : tuple[str, ...]
        DRIAMS sites to union.
    species_covariate : bool
        Whether SpeciesAwareComBat / Limma-with-design should pass the
        species column. Pure single-species experiments set this to False.
    """

    key: str
    title: str
    pairs: tuple[tuple[str, str], ...]
    sites: tuple[str, ...]
    species_covariate: bool


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        key="exp1_saureus",
        title="S. aureus (oxacillin), DRIAMS A+B+C+D",
        pairs=(("Staphylococcus aureus", "oxacillin"),),
        sites=DRIAMS_SITES,
        species_covariate=False,
    ),
    Experiment(
        key="exp1_ecoli",
        title="E. coli (ceftriaxone), DRIAMS A+B+C+D",
        pairs=(("Escherichia coli", "ceftriaxone"),),
        sites=DRIAMS_SITES,
        species_covariate=False,
    ),
    Experiment(
        key="exp1_kpneumoniae",
        title="K. pneumoniae (ceftriaxone), DRIAMS A+B+C+D",
        pairs=(("Klebsiella pneumoniae", "ceftriaxone"),),
        sites=DRIAMS_SITES,
        species_covariate=False,
    ),
    Experiment(
        key="exp3_multi_species",
        title="E. coli + K. pneumoniae (ceftriaxone), DRIAMS A+B+C+D",
        pairs=(
            ("Escherichia coli", "ceftriaxone"),
            ("Klebsiella pneumoniae", "ceftriaxone"),
        ),
        sites=DRIAMS_SITES,
        species_covariate=True,
    ),
)


METHODS: list[tuple[str, str]] = [
    ("combat_johnson", "ComBat (Johnson)"),
    ("combat_fortin", "ComBat (Fortin)"),
    ("combat_chen", "ComBat / CovBat (Chen)"),
    ("limma", "Limma"),
    ("harmony", "Harmony"),
    ("median_centering", "Median centering"),
    ("zscore_per_batch", "Z-score per batch"),
    ("reference_scaling", "Reference scaling"),
    ("batch_aware_warping", "Batch-aware warping"),
    ("quality_weighted_combat", "Quality-weighted ComBat"),
    ("species_aware_combat", "Species-aware ComBat"),
]
METHOD_KEYS = [m[0] for m in METHODS]
METHOD_DISPLAY = dict(METHODS)


def setup_logging(name: str, log_file: Path | None = None) -> logging.Logger:
    """Configure root + named logger; INFO to stdout, DEBUG to optional file."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    )
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
        )
        root.addHandler(fh)

    return logging.getLogger(name)


def _resolve_raw_names(
    raw_meta: pd.DataFrame, canonical_species: str, canonical_drug: str
) -> tuple[str, str]:
    """Resolve canonical species/drug names to whatever the DRIAMS site uses."""
    drug_lower = canonical_drug.lower()
    raw_drug = next((c for c in raw_meta.columns if c.lower() == drug_lower), None)
    if raw_drug is None:
        raise KeyError(
            f"No drug column matching {canonical_drug!r} in metadata; "
            f"first 20 columns: {list(raw_meta.columns)[:20]}..."
        )
    if "Species" not in raw_meta.columns:
        raise KeyError("Expected a 'Species' column after DRIAMSLayout normalization.")
    species_lower = canonical_species.lower()
    raw_species = next(
        (
            v
            for v in raw_meta["Species"].dropna().astype(str).str.strip().unique()
            if v.lower() == species_lower
        ),
        None,
    )
    if raw_species is None:
        raise KeyError(
            f"No species value matching {canonical_species!r} found; "
            f"first 10 seen: "
            f"{sorted(raw_meta['Species'].dropna().astype(str).unique())[:10]}"
        )
    return raw_species, raw_drug


def _preprocess_bin_and_score(spectra: list) -> tuple[np.ndarray, list[float]]:
    """Run MaldiAMRKit default preprocessing + 3-Da binning + SNR per spectrum.

    Returns
    -------
    X : ndarray (n_spectra, N_BINS), float32
        Binned intensity matrix.
    snr : list[float]
        SNR computed by ``estimate_snr`` (median-peaks signal / MAD noise)
        on the preprocessed spectrum. ``np.nan`` if estimation fails.
    """
    from maldiamrkit.preprocessing import bin_spectrum, preprocess
    from maldiamrkit.preprocessing.quality import estimate_snr
    from maldiamrkit.spectrum import MaldiSpectrum

    rows: list[np.ndarray] = []
    snr_values: list[float] = []
    for spec in spectra:
        processed = preprocess(spec.raw)
        try:
            snr = float(
                estimate_snr(
                    MaldiSpectrum(processed),
                    noise_region=(19500, 20000),
                    signal_method="median_peaks",
                    n_top_peaks=10,
                )
            )
        except Exception:
            snr = float("nan")
        snr_values.append(snr)

        binned, _ = bin_spectrum(
            processed, bin_width=BIN_WIDTH, mz_min=MZ_MIN, mz_max=MZ_MAX
        )
        rows.append(binned["intensity"].to_numpy(dtype=np.float32))
    if not rows:
        return np.empty((0, N_BINS), dtype=np.float32), []
    return np.vstack(rows), snr_values


def _load_one_pair_one_site(
    site: str, species: str, drug: str, n_jobs: int
) -> tuple[np.ndarray, pd.DataFrame] | None:
    """Load (X, meta) for one (species, drug, site). Returns None if site missing."""
    from maldiamrkit.data import DatasetLoader, DRIAMSLayout

    site_dir = driams_root() / site
    if not site_dir.is_dir():
        return None

    layout = DRIAMSLayout(
        site_dir,
        id_transform=lambda s: re.sub(r"_MALDI\d+$", "", s),
        duplicate_strategy="first",
    )
    raw_meta = layout.discover_metadata()
    raw_species, raw_drug = _resolve_raw_names(raw_meta, species, drug)

    loader = DatasetLoader(layout, stage="raw", n_jobs=n_jobs)
    ds = loader.load(aggregate_by={"antibiotics": raw_drug, "species": raw_species})

    X, snr = _preprocess_bin_and_score(ds.spectra)
    ids = [s.id for s in ds.spectra]
    meta = ds.meta.loc[ids].copy()

    drug_values = meta[raw_drug].astype(str).str.strip().str.upper()
    keep = drug_values.isin(["R", "S"])
    meta = meta.loc[keep].copy()
    X = X[keep.to_numpy()]
    snr_kept = [s for s, k in zip(snr, keep.tolist(), strict=True) if k]

    meta["site"] = site
    meta["species_canonical"] = species
    meta["drug_canonical"] = drug
    meta["rs_label"] = (
        meta[raw_drug].astype(str).str.strip().str.upper() == "R"
    ).astype(int)
    meta["snr"] = snr_kept
    return X, meta


def load_pair(
    species: str,
    drug: str,
    *,
    sites: tuple[str, ...] = DRIAMS_SITES,
    n_jobs: int = -1,
    use_cache: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load binned spectra + metadata + SNR for one species-drug pair.

    Caches to ``cache/data/<pair_slug>.npz`` (X) and
    ``cache/data/<pair_slug>_meta.csv`` (metadata, includes SNR + site).
    """
    logger = logging.getLogger("data")
    pair_slug = slug(species, drug)
    x_cache = DATA_CACHE / f"{pair_slug}.npz"
    meta_cache = DATA_CACHE / f"{pair_slug}_meta.csv"

    if use_cache and x_cache.exists() and meta_cache.exists():
        logger.info("Loading %s / %s from cache", species, drug)
        X = np.load(x_cache)["X"].astype(np.float32)
        meta = pd.read_csv(meta_cache)
        return X, meta

    blocks: list[tuple[np.ndarray, pd.DataFrame]] = []
    for site in sites:
        try:
            res = _load_one_pair_one_site(site, species, drug, n_jobs=n_jobs)
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping %s / %s @ %s: %s", species, drug, site, exc)
            continue
        if res is None:
            logger.info("Skipping %s (directory not found)", site)
            continue
        X_site, meta_site = res
        if len(meta_site) == 0:
            logger.info("%s yielded no R/S rows for %s / %s", site, species, drug)
            continue
        logger.info("%s / %s @ %s -> %d spectra", species, drug, site, len(meta_site))
        blocks.append((X_site, meta_site))

    if not blocks:
        raise FileNotFoundError(
            f"No DRIAMS site yielded data for {species} / {drug} under {driams_root()}."
        )

    X = np.vstack([b[0] for b in blocks]).astype(np.float32)
    meta = pd.concat([b[1] for b in blocks], ignore_index=True)
    meta = meta[
        ["site", "species_canonical", "drug_canonical", "rs_label", "snr"]
    ].copy()

    if use_cache:
        np.savez_compressed(x_cache, X=X)
        meta.to_csv(meta_cache, index=False)
        logger.info("Cached %s / %s -> %s", species, drug, x_cache)

    return X, meta


def load_experiment(exp: Experiment) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble one experiment's matrix + metadata from cached pair loads.

    Returns
    -------
    X_df : pd.DataFrame
        ``(n_samples, N_BINS)`` indexed by a stringified spectrum id;
        column names are ``mz_<centre>`` for traceability.
    meta : pd.DataFrame
        Aligned to ``X_df.index``. Columns: site, species_canonical,
        drug_canonical, rs_label, snr, sample_id.
    """
    blocks_X: list[np.ndarray] = []
    blocks_meta: list[pd.DataFrame] = []
    for species, drug in exp.pairs:
        X_pair, meta_pair = load_pair(species, drug, sites=exp.sites)
        blocks_X.append(X_pair)
        blocks_meta.append(meta_pair)

    X = np.vstack(blocks_X).astype(np.float32)
    meta = pd.concat(blocks_meta, ignore_index=True)

    keep = meta["site"].isin(exp.sites)
    X = X[keep.to_numpy()]
    meta = meta.loc[keep].reset_index(drop=True)

    sample_ids = [f"{exp.key}_{i:06d}" for i in range(len(meta))]
    meta = meta.assign(sample_id=sample_ids).set_index("sample_id", drop=False)
    bin_centres = MZ_MIN + (np.arange(N_BINS) + 0.5) * BIN_WIDTH
    columns = [f"mz_{c:.0f}" for c in bin_centres]
    X_df = pd.DataFrame(X, index=meta.index, columns=columns)
    return X_df, meta


def build_method(
    key: str,
    *,
    batch: pd.Series,
    species: pd.Series | None,
    snr: pd.Series,
    reference_batch: str = "DRIAMS-A",
    seed: int = SEED,
) -> Any:
    """Instantiate one MaldiBatchKit corrector by short key.

    All correctors share the same ``batch`` index (the DataFrame index of
    ``X``) so the base class can align them at fit / transform time.

    Parameters
    ----------
    key : str
        One of :data:`METHOD_KEYS`.
    batch : pd.Series
        Site labels indexed by sample_id.
    species : pd.Series, optional
        Canonical species labels indexed by sample_id. Only used by methods
        that take a biological covariate (combat_fortin, combat_chen,
        species_aware_combat, limma) and only when not ``None``. Single-
        species experiments pass ``None`` here so those methods reduce to
        their no-covariate equivalent.
    snr : pd.Series
        SNR per spectrum, used by quality_weighted_combat.
    reference_batch : str
        Site to leave fixed for ReferenceScaling (default DRIAMS-A).
    seed : int
        Random seed for stochastic methods (Harmony only).
    """
    from maldibatchkit import (
        BatchAwareWarping,
        ComBat,
        Harmony,
        Limma,
        MedianCentering,
        QualityWeightedComBat,
        ReferenceScaling,
        SpeciesAwareComBat,
        ZScorePerBatch,
    )

    if key == "combat_johnson":
        return ComBat(batch=batch, method="johnson")
    if key == "combat_fortin":
        if species is None:
            return ComBat(batch=batch, method="johnson")
        return ComBat(batch=batch, method="fortin", discrete_covariates=species)
    if key == "combat_chen":
        if species is None:
            return ComBat(batch=batch, method="johnson")
        return ComBat(batch=batch, method="chen", discrete_covariates=species)
    if key == "limma":
        if species is None:
            return Limma(batch=batch)
        species_dummies = pd.get_dummies(species, drop_first=True).astype(float)
        return Limma(batch=batch, design=species_dummies)
    if key == "harmony":
        return Harmony(batch=batch, random_state=seed)
    if key == "median_centering":
        return MedianCentering(batch=batch)
    if key == "zscore_per_batch":
        return ZScorePerBatch(batch=batch)
    if key == "reference_scaling":
        return ReferenceScaling(batch=batch, reference_batch=reference_batch)
    if key == "batch_aware_warping":
        return BatchAwareWarping(batch=batch, method="shift", n_jobs=1)
    if key == "quality_weighted_combat":
        return QualityWeightedComBat(batch=batch, quality=snr)
    if key == "species_aware_combat":
        if species is None:
            return ComBat(batch=batch, method="johnson")
        return SpeciesAwareComBat(batch=batch, species=species)
    raise KeyError(f"Unknown method key {key!r}; expected one of {METHOD_KEYS}")


def stratified_subsample(
    meta: pd.DataFrame,
    *,
    target: int = 5000,
    stratify_cols: tuple[str, ...] = ("site",),
    seed: int = SEED,
) -> np.ndarray:
    """Return positional indices into ``meta`` for a stratified subsample.

    Strata are the unique combinations of ``stratify_cols``. Each stratum
    contributes a count proportional to its size, so site/species ratios
    are preserved. If the input is already smaller than ``target``, all
    rows are kept.
    """
    n = len(meta)
    if n <= target:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    keys = meta[list(stratify_cols)].astype(str).agg("|".join, axis=1).to_numpy()
    positions = np.arange(n)
    out: list[int] = []
    for key in np.unique(keys):
        idx_pos = positions[keys == key]
        share = max(1, int(round(target * len(idx_pos) / n)))
        share = min(share, len(idx_pos))
        chosen = rng.choice(idx_pos, size=share, replace=False)
        out.extend(int(p) for p in chosen)
    out_arr = np.asarray(sorted(out))
    if len(out_arr) > target:
        out_arr = rng.choice(out_arr, size=target, replace=False)
        out_arr.sort()
    return out_arr
