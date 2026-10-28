# Single-cell Transformer

English | [繁體中文](README_TW.md)

A small project for learning single-cell data processing and Transformers with **PBMC3k + Scanpy + PyTorch**: artificially mask known gene expression values, reconstruct them from the other genes in the same cell, and compare the predictions with **per-gene means computed from the training set**.

The first version covers data exploration, training, evaluation with fixed masks, and reloading model weights. It does not yet include scGPT, cell-type annotation, or multi-omics. The original learning plan is referenced at [docs/learning_plan.md](docs/learning_plan.md).

Start with [readme_project.md](readme_project.md) for concrete examples of the model inputs, prediction targets, code workflow, and outputs.

## Quick start

This version was run on **macOS arm64, Python 3.11.2, CPU**. `requirements.txt` pins the direct dependency versions used for that run; compatible version ranges are defined in `pyproject.toml`.

Run these commands from the project root:

```bash
python3 -m venv singlecell-venv
source singlecell-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
jupyter lab
```

Open and run the notebooks in this order:

1. [01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb): matrix structure and metadata, QC, counts versus log1p values, and PCA/UMAP.
2. [02_masked_expression_transformer.ipynb](notebooks/02_masked_expression_transformer.ipynb): data splits, tensor shapes, masking, forward/backward passes, full training, baseline comparison, and loading weights.

Both notebooks contain explanations in Traditional Chinese and actual execution outputs. In Jupyter, select the Python 3 kernel for the environment created above. The first run requires an internet connection to download approximately 6 MB of PBMC3k data; subsequent runs use the local cache.

You can also run the modules directly:

```bash
python -m singlecell_baseline.explore
python -m singlecell_baseline.train
python -m pytest -q
```

The training defaults are 5 epochs, 128 genes, batch size 16, and seed 42. Save other experiments to a separate directory to avoid overwriting this version's results:

```bash
python -m singlecell_baseline.train --epochs 10 --output outputs/experiment_02
```

You can explicitly request `--device mps`; availability is checked before training starts. This version has only been verified on CPU; MPS training has not been verified.

## Results from the first version

Run date: **2026-10-23**. The raw dataset contains **2,700 cells × 32,738 genes**. Fixed QC thresholds retain **2,638 cells**, split into **1,846 training**, **396 validation**, and **396 test** cells. The model uses 128 genes selected from the training set and has **12,769 trainable parameters**.

None of the 5 training epochs improved on the gene-mean baseline's validation MSE, so **epoch 0** was selected. The model starts from the training-set gene means with a zero-initialized residual head, making epoch 0 exactly equal to the baseline. The table below reports the selected checkpoint; **it is not evidence that training made the model outperform the baseline**.

| Masked test positions | Count | Gene mean MAE | Transformer MAE | Gene mean RMSE | Transformer RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 7,524 | 0.066361 | 0.066361 | 0.250510 | 0.250510 |
| Originally zero | 7,376 | 0.033832 | 0.033832 | 0.039179 | 0.039179 |
| Originally nonzero | 148 | 1.687506 | 1.687506 | 1.764610 | 1.764610 |

Errors are measured in processed log1p expression values. Approximately 98% of masked positions are zero, so the low overall error hides the difficulty of reconstructing nonzero values. The short training run and current highly variable gene (HVG) selection have not improved validation performance. Future work should first examine features and settings on the training and validation sets before conducting a new evaluation.

![Learning curve](outputs/figures/learning_curve.png)

Full results: [evaluation.csv](outputs/metrics/evaluation.csv), [results.json](outputs/metrics/results.json), and [history.csv](outputs/metrics/history.csv). The training function verifies that test predictions are exactly identical before saving and after reloading. Tests cover splitting and fitting on training data only, preventing masked-value leakage, grouped metrics, a small offline training run, and checkpoint loading.

## Data and evaluation design

- **QC:** Retain cells with at least 200 and fewer than 2,500 detected genes, and a mitochondrial count fraction below 5%. These introductory PBMC tutorial thresholds are applied per cell and are not derived from held-out statistics; they are not universal quality rules.
- **Split before fitting:** Require a gene to be detected in at least 3 training cells, then compute Seurat normalized dispersion using only training data and select the top 128 genes with a stable sort. Gene means are also computed only from training data. Validation and test data use the same gene vocabulary and order.
- **Value processing:** Preserve raw counts. Scale each cell to a total of 10,000 using its full library size, apply `log1p`, and then select the model's gene subset. This fixed per-cell transformation does not require fitting a scaler.
- **Masking:** Randomly mask 19/128 positions per cell; both zero and nonzero values can be selected. The model clears masked values at its input and receives a separate boolean mask. Original targets are stored separately, and MSE is computed only at masked positions.
- **Fair comparison:** Resample training masks for each batch; fix validation/test masks with seeds 43/44. The Transformer and gene-mean baseline use identical targets and masks. Validation MSE selects the weights, and test evaluation takes place after selection.

The model uses `gene embedding + value/mask encoding → Transformer Encoder → scalar head`, then adds the training-set gene mean to form a residual prediction. Defaults are embedding dimension 32, 4 heads, 1 layer, dropout 0.1, and AdamW learning rate 0.001. There is no positional encoding, padding, or causal mask.

| Tensor | Shape (B: batch; G: genes; D: embedding) |
| --- | --- |
| Expression values / targets / boolean mask | `(B, G)`, default `(16, 128)` |
| Values and mask indicators | `(B, G, 2)` |
| Sum of gene/value embeddings; Encoder output | `(B, G, D)`, default `(16, 128, 32)` |
| Predictions | `(B, G)` |

The full raw dataset stays sparse; only the selected 128 genes are converted to dense arrays for the model. Data sources, SHA-256 hashes, QC settings, gene lists, cell splits, hyperparameters, and package versions are saved in `outputs/metrics/`.

## Files and artifacts

```text
notebooks/                 # Two tutorial notebooks with actual execution outputs
singlecell_baseline/        # data.py / model.py / train.py / explore.py
tests/                     # Correctness tests that require no data download
data/                      # Source notes; raw/ and processed/ are excluded from Git
outputs/figures/            # QC, PCA, UMAP, and learning curve
outputs/metrics/            # JSON/CSV settings and actual results
outputs/checkpoints/        # Local best.pt, excluded from Git
docs/learning_plan.md       # Original learning plan
.github/workflows/          # GitHub Actions: offline CPU tests
```

`outputs/checkpoints/best.pt` contains the state dict, ordered genes, preprocessing settings, and model configuration. Rerun training to recreate it; there is no need to upload weights or data to GitHub. `.gitignore` also excludes virtual environments, caches, and secrets. Figures, small evaluation files, and notebooks are kept for convenient browsing.

```python
from singlecell_baseline.train import load_checkpoint

model, checkpoint = load_checkpoint("outputs/checkpoints/best.pt")
genes = checkpoint["genes"]  # New inputs must use exactly the same gene order and preprocessing.
```

## Scope and next steps

This task artificially masks values **after preprocessing** a fully measured dataset. Normalization happens before masking, so these results do not demonstrate recovery of true signals from missing raw counts. No independent cell-type labels are used for PBMC3k here; PCA/UMAP and attention alone cannot establish biological mechanisms. Random cell splits from a single donor also do not demonstrate generalization across donors or datasets.

Next steps can compare feature selection, training duration, and performance at nonzero positions on the validation set, followed by scGPT embeddings/fine-tuning and multi-omics. This version keeps the workflow small, readable, and reproducible.

Data and API references: [Scanpy PBMC3k](https://scanpy.scverse.org/en/stable/generated/scanpy.datasets.pbmc3k.html), [Scanpy HVG](https://scanpy.scverse.org/en/stable/generated/scanpy.pp.highly_variable_genes.html), and [PyTorch TransformerEncoder](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html). Code and documentation are licensed under MIT. Data remain subject to the original provider's terms; see [data/README.md](data/README.md).
