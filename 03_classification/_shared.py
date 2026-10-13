"""Constants and helpers shared by the classification benchmark.

The benchmark evaluates five MaldiDeepKit classifier configurations
(four architectures and an ensemble of the MLP and the CNN) on three
species-drug resistance-prediction tasks from DRIAMS A-D. DRIAMS is
read with MaldiAMRKit's :class:`~maldiamrkit.data.DRIAMSLayout`.
"""

from __future__ import annotations

import logging
import sys
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
DATA_CACHE_DIR = HERE / ".data_cache"

SPECIES_DRUG_PAIRS: list[tuple[str, str]] = [
    ("Staphylococcus aureus", "oxacillin"),
    ("Escherichia coli", "ceftriaxone"),
    ("Klebsiella pneumoniae", "ceftriaxone"),
]

SEED = 42
N_FOLDS = 5
N_REPEATS = 3

CLASSIFIERS: list[str] = [
    "mlp_attn",
    "cnn",
    "resnet",
    "transformer",
    "ensemble_mlp_cnn",
]

CLASSIFIER_DISPLAY = {
    "mlp_attn": "MLP + attention",
    "cnn": "CNN",
    "resnet": "ResNet",
    "transformer": "Transformer (1-D ViT)",
    "ensemble_mlp_cnn": "Ensemble (MLP + CNN)",
}


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


def setup_logging(
    name: str = "maldideepkit-paper", log_file: Path | None = None
) -> None:
    """Configure logging to console (INFO) and optionally to a file (DEBUG)."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
        )
        root.addHandler(fh)


DRIAMS_SITES = ("DRIAMS-A", "DRIAMS-B", "DRIAMS-C", "DRIAMS-D")


def _resolve_raw_names(
    raw_meta: pd.DataFrame, canonical_species: str, canonical_drug: str
) -> tuple[str, str]:
    """Resolve canonical species / drug names to the raw DRIAMS strings.

    DRIAMS CSVs use capitalized drug column names (``"Oxacillin"``,
    ``"Ceftriaxone"``) and can vary in species string formatting across
    sites. This helper case-insensitively maps the paper's lowercase
    canonical names to whatever the raw metadata actually contains, so
    ``aggregate_by`` filters compare the right strings.
    """
    drug_lower = canonical_drug.lower()
    raw_drug = next((c for c in raw_meta.columns if c.lower() == drug_lower), None)
    if raw_drug is None:
        raise KeyError(
            f"No drug column matching {canonical_drug!r} in metadata; "
            f"available drug-like columns: {list(raw_meta.columns)[:20]}..."
        )

    if "Species" not in raw_meta.columns:
        raise KeyError(
            "Expected a 'Species' column in DRIAMS metadata after "
            "DRIAMSLayout normalization."
        )
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
            f"No species value matching {canonical_species!r} found in "
            f"metadata; first 10 seen: "
            f"{sorted(raw_meta['Species'].dropna().astype(str).unique())[:10]}"
        )
    return raw_species, raw_drug


def _load_one_site(
    site: str,
    species: str,
    drug: str,
    n_jobs: int,
) -> tuple[np.ndarray, pd.DataFrame] | None:
    """Load one DRIAMS site for the given species-drug pair.

    Returns ``(X, meta)`` for the subset of that site that has both the
    species and an R or S (non-intermediate) label for the drug, or
    ``None`` if the site directory is missing.

    DRIAMS CSVs use capitalized drug column names and slightly varying
    species spellings; we resolve the canonical lowercase inputs to the
    raw strings first, so :meth:`DatasetLoader.load`'s ``aggregate_by``
    filter compares the right values.

    Raw spectra are read and run through MaldiAMRKit's default
    preprocessing + 3-Da uniform binning, each independently.
    """
    import re as _re

    from maldiamrkit.data import DatasetLoader, DRIAMSLayout

    site_dir = driams_root() / site
    if not site_dir.is_dir():
        return None

    layout = DRIAMSLayout(
        site_dir,
        id_transform=lambda s: _re.sub(r"_MALDI\d+$", "", s),
        duplicate_strategy="first",
    )
    raw_meta = layout.discover_metadata()
    raw_species, raw_drug = _resolve_raw_names(raw_meta, species, drug)

    loader = DatasetLoader(layout, stage="raw", n_jobs=n_jobs)
    ds = loader.load(aggregate_by={"antibiotics": raw_drug, "species": raw_species})

    X_np = _preprocess_and_bin_spectra(ds.spectra)
    ids = [s.id for s in ds.spectra]
    meta = ds.meta.loc[ids].copy()

    drug_values = meta[raw_drug].astype(str).str.strip().str.upper()
    keep_mask = drug_values.isin(["R", "S"])
    meta = meta.loc[keep_mask].copy()

    meta["site"] = site
    meta["rs_label"] = (
        meta[raw_drug].astype(str).str.strip().str.upper() == "R"
    ).astype(int)

    X = X_np[keep_mask.to_numpy()].astype(np.float32)
    return X, meta


def _preprocess_and_bin_spectra(spectra: list) -> np.ndarray:
    """Run MaldiAMRKit's default pipeline + 3-Da uniform binning on each spectrum.

    Each ``MaldiSpectrum`` in ``spectra`` gives one row of the
    ``(n_spectra, 6000)`` output. The loop is single-threaded; ``load_pair``
    caches the result, so it runs once per ``(species, drug, site)``.

    The bins follow the DRIAMS layout (3 Da over 2000-20000 Da), which the
    classifier configs built with ``DEFAULT_BIN_WIDTH=3`` assume.
    """
    from maldiamrkit.preprocessing import bin_spectrum, preprocess

    rows: list[np.ndarray] = []
    for spec in spectra:
        processed = preprocess(spec.raw)
        binned, _bin_meta = bin_spectrum(
            processed, bin_width=3, mz_min=2000, mz_max=20000
        )
        rows.append(binned["intensity"].to_numpy(dtype=np.float32))
    if not rows:
        return np.empty((0, 6000), dtype=np.float32)
    return np.vstack(rows)


def load_pair(
    species: str,
    drug: str,
    *,
    use_cache: bool = True,
    n_jobs: int = -1,
    sites: tuple[str, ...] | None = None,
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load binned DRIAMS spectra for one species-drug pair.

    Reads each requested DRIAMS site under ``DRIAMS_ROOT`` with
    :class:`maldiamrkit.data.DRIAMSLayout` and :class:`DatasetLoader`,
    runs the MaldiAMRKit default preprocessing (clip negatives, square
    root, Savitzky-Golay, SNIP baseline, trimming to 2000-20000 Da, TIC
    normalisation) and 3-Da binning on every raw spectrum, without
    merging replicates, keeps the resistant and susceptible isolates and
    caches the result.

    Parameters
    ----------
    species, drug : str
        Canonical lowercase species and drug names.
    use_cache : bool, default=True
        If True, read / write an npz + csv cache under ``cache_dir``.
    n_jobs : int, default=-1
        Forwarded to :class:`maldiamrkit.data.DatasetLoader`.
    sites : tuple of str, optional
        Which DRIAMS sites to union. Defaults to all four
        (``DRIAMS-A``, ``DRIAMS-B``, ``DRIAMS-C``, ``DRIAMS-D``). Pass a
        subset (e.g. ``("DRIAMS-A",)``) to restrict.
    cache_dir : Path, optional
        Directory for the npz + csv cache. Defaults to
        :data:`DATA_CACHE_DIR`. Use a site-specific directory (e.g.
        ``.data_cache_A``) when running on a site subset so the
        cache does not collide with the full-union cache.

    Returns
    -------
    X : ndarray of shape (n_samples, n_bins), float32
    y : ndarray of shape (n_samples,), int64 (0 = S, 1 = R)
    metadata : pd.DataFrame with at least ``site`` and ``rs_label`` columns.
    """
    logger = logging.getLogger("shared")
    if sites is None:
        sites = DRIAMS_SITES
    if cache_dir is None:
        cache_dir = DATA_CACHE_DIR
    pair_slug = slug(species, drug)
    x_cache = cache_dir / f"{pair_slug}.npz"
    meta_cache = cache_dir / f"{pair_slug}_meta.csv"

    if use_cache and x_cache.exists() and meta_cache.exists():
        logger.info("Loading %s / %s from cache (%s)", species, drug, cache_dir)
        X = np.load(x_cache)["X"].astype(np.float32)
        metadata = pd.read_csv(meta_cache)
    else:
        blocks: list[tuple[np.ndarray, pd.DataFrame]] = []
        for site in sites:
            try:
                result = _load_one_site(site, species, drug, n_jobs=n_jobs)
            except (KeyError, ValueError) as exc:
                logger.warning("Skipping %s for %s / %s: %s", site, species, drug, exc)
                continue
            if result is None:
                logger.info("Skipping %s (directory not found)", site)
                continue
            X_site, meta_site = result
            if len(meta_site) == 0:
                logger.info("Skipping %s (no R/S rows for this pair)", site)
                continue
            logger.info("%s -> %d spectra", site, len(meta_site))
            blocks.append((X_site, meta_site))

        if not blocks:
            raise FileNotFoundError(
                f"No DRIAMS site yielded data for {species} / {drug} "
                f"under {driams_root()} (sites={list(sites)})."
            )

        X = np.vstack([b[0] for b in blocks]).astype(np.float32)
        metadata = pd.concat([b[1] for b in blocks], ignore_index=True)

        if use_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(x_cache, X=X)
            metadata.to_csv(meta_cache, index=False)
            logger.info("Cached %s / %s to %s", species, drug, x_cache)

    if "rs_label" not in metadata.columns:
        raise KeyError("Expected 'rs_label' column in cached DRIAMS metadata.")
    y = metadata["rs_label"].to_numpy().astype(np.int64)

    n_r = int((y == 1).sum())
    n_s = int((y == 0).sum())
    logger.info(
        "Loaded %s / %s: %d spectra (%d R / %d S), %d bins",
        species,
        drug,
        len(y),
        n_r,
        n_s,
        X.shape[1],
    )
    return X, y, metadata


DEFAULT_BIN_WIDTH = 3


def build_classifier(
    name: str,
    *,
    bin_width: int,
    input_dim: int,
    random_state: int = 0,
    quick: bool = False,
    disable_warping: bool = False,
    input_transform: str | None = None,
) -> Any:
    """Instantiate a MaldiDeepKit classifier config by name.

    Accepts any key from :data:`CLASSIFIER_DISPLAY`.

    Each classifier is built with its :meth:`from_spectrum` factory,
    which scales kernel and patch sizes to the bin layout. At the DRIAMS
    layout used here (``bin_width=3``, ``input_dim=6000``) it returns the
    package defaults.

    Parameters
    ----------
    name : str
        One of the entries in :data:`CLASSIFIERS`.
    bin_width : int
        Bin width in Daltons of the spectra. DRIAMS defaults to 3.
    input_dim : int
        Number of bins in each spectrum. DRIAMS at the full 2000-20000
        Da range gives 6000.
    random_state : int, default=0
        Passed to the classifier's ``random_state``.
    quick : bool, default=False
        If True, shrink ``epochs`` and ``early_stopping_patience`` for a
        smoke test. The paper uses ``quick=False``.
    disable_warping : bool, default=False
        If True, force ``warping=None`` on the built classifier, as in
        the single-site runs: the calibration drift that warping
        corrects is a cross-site effect.
    input_transform : str, optional
        Override the input-transform mode on all classifiers. One of
        ``{"none", "standardize", "log1p", "robust",
        "log1p+standardize"}``. When ``None`` (default), the paper
        factory leaves the classifier's default in place
        (``standardize=True``).

    Returns
    -------
    BaseSpectralClassifier
        An unfitted estimator.
    """
    from maldiamrkit.alignment import Warping

    from maldideepkit import (
        MaldiCNNClassifier,
        MaldiMLPClassifier,
        MaldiResNetClassifier,
        MaldiTransformerClassifier,
    )

    common = dict(
        epochs=5 if quick else 100,
        early_stopping_patience=3 if quick else 10,
        batch_size=32,
        random_state=random_state,
        class_weight="balanced",
        standardize=True,
        device="auto",
        verbose=False,
        use_amp=True,
        tune_threshold=True,
        threshold_metric="balanced_accuracy",
        label_smoothing=0.05,
        warping=(
            None if quick or disable_warping else Warping(method="shift", n_jobs=-1)
        ),
    )

    if input_transform is not None:
        common["input_transform"] = input_transform

    if name == "mlp_attn":
        cls = MaldiMLPClassifier
        common["use_attention"] = True
    elif name == "cnn":
        cls = MaldiCNNClassifier
    elif name == "resnet":
        cls = MaldiResNetClassifier
        common["stage_channels"] = (32, 64, 128, 256)
        if not quick:
            common["epochs"] = 200
            common["early_stopping_patience"] = 20
    elif name == "transformer":
        cls = MaldiTransformerClassifier
        common["layerscale_init"] = None
        common["depth"] = 4
        common["num_heads"] = 2
        common["attention_dropout"] = 0.1
        common["retry_on_val_auroc_below"] = 0.60
        common["max_retries"] = 3
    elif name == "ensemble_mlp_cnn":
        from maldideepkit.utils import SpectralEnsemble

        member_mlp = build_classifier(
            "mlp_attn",
            bin_width=bin_width,
            input_dim=input_dim,
            random_state=random_state,
            quick=quick,
            disable_warping=disable_warping,
            input_transform=input_transform,
        )
        member_cnn = build_classifier(
            "cnn",
            bin_width=bin_width,
            input_dim=input_dim,
            random_state=random_state,
            quick=quick,
            disable_warping=disable_warping,
            input_transform=input_transform,
        )
        return SpectralEnsemble([member_mlp, member_cnn])
    else:
        raise ValueError(f"Unknown classifier name: {name!r}. Known: {CLASSIFIERS}")

    if quick and name in {"resnet", "transformer"}:
        common["warmup_epochs"] = 1
    return cls.from_spectrum(bin_width=bin_width, input_dim=input_dim, **common)


def compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """Compute per-class and overall metrics for one CV fold.

    Labels: 0 = Susceptible (S, usually majority), 1 = Resistant
    (R, usually minority).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    recall_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    recall_s = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision_r = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    precision_s = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    def _f1(p: float, r: float) -> float:
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    f1_r = _f1(precision_r, recall_r)
    f1_s = _f1(precision_s, recall_s)

    vme = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    me = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    mcc = float(matthews_corrcoef(y_true, y_pred))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))

    if len(np.unique(y_true)) < 2:
        auroc = float("nan")
    else:
        auroc = float(roc_auc_score(y_true, y_proba))

    return {
        "auroc": auroc,
        "mcc": mcc,
        "balanced_accuracy": balanced_acc,
        "recall_r": recall_r,
        "recall_s": recall_s,
        "precision_r": precision_r,
        "precision_s": precision_s,
        "f1_r": f1_r,
        "f1_s": f1_s,
        "vme": vme,
        "me": me,
    }


def run_cv(
    classifier_factory: Callable[[int], Any],
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_folds: int = N_FOLDS,
    n_repeats: int = N_REPEATS,
    random_state: int = SEED,
) -> pd.DataFrame:
    """Run repeated stratified K-fold CV of ``classifier_factory`` on ``(X, y)``.

    Each of the ``n_repeats`` repeats draws a ``StratifiedKFold`` split
    with seed ``random_state + repeat * 100``, for
    ``n_repeats x n_folds`` fits in total.

    ``classifier_factory(run_index)`` must return a fresh unfitted
    estimator on each call, where ``run_index = repeat * n_folds + fold``
    ranges over ``[0, n_repeats * n_folds)`` so the caller can derive a
    unique seed per fit.

    Returns
    -------
    pd.DataFrame
        One row per (repeat, fold), with columns ``repeat``, ``fold``,
        then every metric from :func:`compute_fold_metrics`.
    """
    from sklearn.model_selection import StratifiedKFold

    logger = logging.getLogger("shared")

    rows: list[dict[str, float]] = []
    total_fits = n_repeats * n_folds
    for repeat in range(n_repeats):
        kf = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_state + repeat * 100,
        )
        for fold, (tr_idx, te_idx) in enumerate(kf.split(X, y)):
            run_idx = repeat * n_folds + fold
            clf = classifier_factory(run_idx)
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            clf.fit(X_tr, y_tr)
            proba = clf.predict_proba(X_te)
            proba_r = proba[:, 1] if proba.shape[1] == 2 else proba.max(axis=1)
            y_pred = clf.predict(X_te)

            metrics = compute_fold_metrics(y_te, y_pred, proba_r)
            metrics["repeat"] = repeat
            metrics["fold"] = fold
            rows.append(metrics)

            logger.info(
                "  run %d/%d (repeat %d, fold %d/%d)  "
                "auroc=%.3f  mcc=%.3f  bal_acc=%.3f",
                run_idx + 1,
                total_fits,
                repeat,
                fold + 1,
                n_folds,
                metrics["auroc"],
                metrics["mcc"],
                metrics["balanced_accuracy"],
            )

    df = pd.DataFrame(rows)
    lead = ["repeat", "fold"]
    cols = lead + [c for c in df.columns if c not in lead]
    return df[cols]


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to CSV, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logging.getLogger("shared").info("Saved %s (%d rows)", path, len(df))
