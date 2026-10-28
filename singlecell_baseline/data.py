"""PBMC3k QC and train-only feature selection; never use exploration outputs to fit."""

from dataclasses import asdict, dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse
from sklearn.model_selection import train_test_split


@dataclass
class DataConfig:
    n_genes: int = 128
    seed: int = 42
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    min_genes: int = 200
    max_genes: int = 2500
    max_pct_mt: float = 5.0
    min_train_cells: int = 3
    target_sum: float = 10_000.0


@dataclass
class PreparedData:
    adata: ad.AnnData
    indices: dict[str, np.ndarray]
    report: dict

    def values(self, split: str) -> np.ndarray:
        return np.asarray(self.adata.X[self.indices[split]], dtype=np.float32).copy()


def load_pbmc3k(data_dir: str | Path = "data/raw") -> ad.AnnData:
    """Scanpy caches the public raw-count dataset locally on its first call."""
    directory = Path(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    previous = sc.settings.datasetdir
    try:
        sc.settings.datasetdir = directory
        adata = sc.datasets.pbmc3k()
    finally:
        sc.settings.datasetdir = previous
    adata.var_names_make_unique()
    return adata


def quality_control(raw: ad.AnnData, config: DataConfig) -> ad.AnnData:
    """Fixed teaching cutoffs, applied independently to each cell, without labels."""
    adata = raw.copy()
    values = adata.X.data if sparse.issparse(adata.X) else np.asarray(adata.X)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Expected finite, nonnegative raw counts.")
    if not np.allclose(values, np.round(values)):
        raise ValueError("Expected raw integer counts, not normalized expression.")
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, inplace=True)
    keep = (
        (adata.obs.n_genes_by_counts >= config.min_genes)
        & (adata.obs.n_genes_by_counts < config.max_genes)
        & (adata.obs.pct_counts_mt < config.max_pct_mt)
        & (adata.obs.total_counts > 0)
    )
    return adata[keep].copy()


def prepare_data(raw: ad.AnnData, config: DataConfig | None = None) -> PreparedData:
    """Split cells before fitting gene filters, HVGs, or gene-mean statistics."""
    config = config or DataConfig()
    if config.n_genes < 2 or config.min_train_cells < 1 or config.target_sum <= 0:
        raise ValueError("Need >=2 genes, >=1 train cell per gene, and target_sum >0.")
    if not (0 < config.val_fraction < 1 and 0 < config.test_fraction < 1
            and config.val_fraction + config.test_fraction < 1):
        raise ValueError("Validation/test fractions must be positive and sum to <1.")
    adata = quality_control(raw, config)
    if adata.n_obs < 10:
        raise ValueError("Too few cells after QC for train/validation/test splits.")
    train_val, test = train_test_split(
        np.arange(adata.n_obs), test_size=config.test_fraction, random_state=config.seed
    )
    train, val = train_test_split(
        train_val, test_size=config.val_fraction / (1 - config.test_fraction),
        random_state=config.seed,
    )
    indices = {"train": np.sort(train), "validation": np.sort(val), "test": np.sort(test)}
    adata.obs["split"] = ""
    for split, rows in indices.items():
        adata.obs.loc[adata.obs_names[rows], "split"] = split

    # Each cell uses its own full library size and a fixed target, no fitted scaler.
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=config.target_sum)
    sc.pp.log1p(adata)
    train_counts = adata.layers["counts"][indices["train"]]
    detected = np.asarray((train_counts > 0).sum(axis=0)).ravel()
    eligible = detected >= config.min_train_cells
    training = adata[indices["train"], eligible].copy()
    if training.n_vars < config.n_genes:
        raise ValueError(f"Only {training.n_vars} eligible genes; requested {config.n_genes}.")
    sc.pp.highly_variable_genes(training, flavor="seurat", n_top_genes=config.n_genes)
    # Stable rank enforces an exact vocabulary size even if HVG scores tie.
    scores = training.var["dispersions_norm"].to_numpy()
    ranking = np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind="stable")
    selected = training.var_names[ranking[:config.n_genes]]
    adata = adata[:, selected].copy()
    adata.var["train_dispersion_norm"] = training.var.loc[selected, "dispersions_norm"]
    adata.var["train_detected_cells"] = detected[raw.var_names.get_indexer(selected)]
    # Only the small chosen gene matrix becomes dense, never the entire raw matrix.
    adata.X = (adata.X.toarray() if sparse.issparse(adata.X) else adata.X).astype(np.float32)
    adata.var["train_mean"] = adata.X[indices["train"]].mean(axis=0)
    report = {
        "config": asdict(config),
        "steps": [
            {"stage": "raw counts", "cells": raw.n_obs, "genes": raw.n_vars},
            {"stage": "fixed cell QC", "cells": adata.n_obs, "genes": raw.n_vars},
            {"stage": "train-only detected gene filter", "cells": len(train),
             "genes": int(eligible.sum())},
            {"stage": "train-only HVGs / log1p target", "cells": adata.n_obs,
             "genes": adata.n_vars},
        ],
        "split_sizes": {name: len(rows) for name, rows in indices.items()},
        "normalization": "Per-cell full-library total to 10000 (configurable), then log1p; before gene subset.",
        "feature_selection": "Seurat normalized dispersion fitted only on training cells; stable top-k.",
        "qc_note": "Preset PBMC tutorial cutoffs; not learned from validation/test; not universal biological rules.",
    }
    return PreparedData(adata, indices, report)


def save_prepared(prepared: PreparedData, output: Path, processed_dir: Path) -> None:
    import json

    metrics = output / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    prepared.adata.write_h5ad(processed_dir / "baseline.h5ad", compression="gzip")
    prepared.adata.obs[["split"]].to_csv(metrics / "cell_splits.csv", index_label="cell_id")
    prepared.adata.var.to_csv(metrics / "selected_genes.csv", index_label="gene")
    (metrics / "preprocessing.json").write_text(
        json.dumps(prepared.report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
