"""Descriptive PBMC3k exploration, kept separate from model feature fitting."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import scanpy as sc

from .data import DataConfig, load_pbmc3k, quality_control


def run_exploration(data_dir: str | Path = "data", output: str | Path = "outputs"):
    data_dir, output = Path(data_dir), Path(output)
    for name in ("metrics", "figures"):
        (output / name).mkdir(parents=True, exist_ok=True)
    config = DataConfig()
    raw = load_pbmc3k(data_dir / "raw")
    print(raw)
    print(f"X: {type(raw.X).__name__}; shape={raw.shape}; dtype={raw.X.dtype}")
    adata = quality_control(raw, config)
    # QC plotting uses the pre-filter distribution to show the actual cutoffs.
    before = quality_control(raw, DataConfig(min_genes=0, max_genes=10**9, max_pct_mt=101))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for ax, key, label in zip(axes, ("total_counts", "n_genes_by_counts", "pct_counts_mt"),
                              ("Total counts", "Detected genes", "Mitochondrial counts (%)")):
        ax.hist(before.obs[key], bins=50, color="#487b9e")
        ax.set(xlabel=label, ylabel="Cells")
    axes[1].axvline(config.min_genes, color="red", linestyle="--")
    axes[1].axvline(config.max_genes, color="red", linestyle="--")
    axes[2].axvline(config.max_pct_mt, color="red", linestyle="--")
    fig.tight_layout()
    fig.savefig(output / "figures" / "qc.png", dpi=150)
    plt.close(fig)
    steps = [{"stage": "raw", "cells": raw.n_obs, "genes": raw.n_vars},
             {"stage": "cell QC", "cells": adata.n_obs, "genes": adata.n_vars}]
    sc.pp.filter_genes(adata, min_cells=3)
    steps.append({"stage": "exploratory gene filter", "cells": adata.n_obs, "genes": adata.n_vars})
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=10_000)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
    sc.pp.pca(adata, n_comps=30, mask_var="highly_variable", random_state=config.seed)
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30, random_state=config.seed)
    sc.tl.umap(adata, random_state=config.seed)
    for basis in ("pca", "umap"):
        fig = getattr(sc.pl, basis)(adata, color="pct_counts_mt", show=False, return_fig=True)
        fig.savefig(output / "figures" / f"{basis}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(processed_dir / "exploration.h5ad", compression="gzip")
    raw_path = data_dir / "raw" / "pbmc3k_raw.h5ad"
    report = {
        "source": "scanpy.datasets.pbmc3k() / 10x Genomics PBMC3k",
        "documentation": "https://scanpy.scverse.org/en/stable/generated/scanpy.datasets.pbmc3k.html",
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "raw_shape": list(raw.shape), "raw_matrix_type": type(raw.X).__name__,
        "raw_obs_columns": raw.obs.columns.tolist(), "raw_var_columns": raw.var.columns.tolist(),
        "qc_config": asdict(config), "steps": steps,
        "qc_quantiles_before_filter": before.obs[
            ["total_counts", "n_genes_by_counts", "pct_counts_mt"]
        ].quantile([0, .01, .5, .99, 1]).to_dict(),
        "pca_components": 30, "umap_seed": config.seed,
        "note": "Full-data exploration only. Training reloads raw counts and refits features on train cells.",
    }
    (output / "metrics" / "exploration.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return adata, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    _, report = run_exploration(args.data_dir, args.output)
    print(json.dumps(report["steps"], indent=2))


if __name__ == "__main__":
    main()
