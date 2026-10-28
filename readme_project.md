# Project overview: Single-cell Transformer

English | [繁體中文](readme_project_TW.md)

## What does this project do?

This is a small project for learning **Scanpy, PyTorch, and Transformers** with real single-cell RNA data.

The core task is to **artificially mask some gene expression values in a cell and have the model predict the original values from the other visible genes**.

The first version aims to run the complete workflow—loading data, preprocessing, training, evaluating, saving, and reloading a model—and use a simple baseline to check whether the Transformer improves performance. It is a baseline for learning and experimentation; pretrained models and multi-omics have not yet been added.

## What does the data look like?

The project uses the public PBMC3k single-cell RNA-seq dataset. See [data/README.md](data/README.md) for data sources and download instructions.

| Data element | Meaning in this project |
| --- | --- |
| One row | One cell, or one sample |
| One column | One gene, or one feature |
| A value in the raw matrix | RNA counts measured for that gene in that cell |
| A value used by the model | Expression after count normalization and `log1p` |

The raw dataset has **2,700 cells and 32,738 genes** and uses a sparse matrix to store its many zero values. A zero means no counts were detected in that measurement; it does not establish that the gene has no biological expression.

This version's fixed quality control (QC) retains **2,638 cells**, and **128 genes** are selected as model inputs. Raw counts are preserved in the data layer and are not overwritten by processed values.

## What exactly does the model predict?

The following is an illustration, not an experimental result. Suppose a cell has four processed gene expression values:

| Gene | Preserved target | Value seen by the model | Mask indicator |
| --- | ---: | ---: | --- |
| A | 0.0 | 0.0 | False |
| B | 1.8 | 0.0 | True |
| C | 0.7 | 0.7 | False |
| D | 2.1 | 0.0 | True |

The model uses gene identities, the visible values for A/C, and mask indicators to predict the original values for B/D. A's zero is an observed value; the zeros for B/D are placeholders introduced by masking. The mask distinguishes them.

Original targets are kept separately, and the model clears masked positions to zero at its input. Training computes errors only for B/D, so copying other visible values cannot reduce the loss.

The actual configuration masks **19/128** positions per cell (about 15%). Both originally zero and nonzero positions can be selected.

## Complete workflow

1. **Load and explore:** Inspect the AnnData matrix, cell metadata, and gene metadata to understand the data axes.
2. **Quality control:** Retain cells with at least 200 and fewer than 2,500 detected genes, and a mitochondrial count fraction below 5%. These are tutorial thresholds for this version, not universal standards for every dataset.
3. **Split cells:** Use 1,846 training, 396 validation, and 396 test cells, with random seed 42.
4. **Preprocess and select genes:** Scale each cell to a total of 10,000 using its own full library size, then apply `log1p`. Use only training data to determine gene filtering, HVG rankings, and per-gene means; select 128 genes.
5. **Train:** Resample masked positions for each batch, run the forward pass, compute the loss, backpropagate, and update weights.
6. **Select a model:** Use fixed validation masks and choose the checkpoint with the lowest validation MSE, including the initial epoch 0 checkpoint.
7. **Final evaluation:** After selecting weights, compare the Transformer with the gene-mean baseline using fixed test masks, and verify that predictions are identical before saving and after reloading.

HVG, PCA, and UMAP results in the data exploration notebook use the full dataset for exploration. Model training starts again from raw counts to avoid using statistics fitted to the full dataset during evaluation.

## How is the Transformer built?

```text
Gene ID ─────────────────→ Gene embedding ──────┐
                                               ├→ Sum → Transformer Encoder → Prediction head
Visible expression + mask → Value/mask encoding ┘                                     │
                                                                                     ↓
                                                            Add train gene mean → Predicted expression
```

- **Gene embedding:** Identifies the gene at each position.
- **Value/mask encoding:** Converts the visible expression value and masked/unmasked indicator into a vector.
- **Transformer Encoder:** Uses bidirectional self-attention to model relationships between gene inputs within the same cell.
- **Prediction head:** Outputs a correction for each gene position, then adds the training-set gene mean to obtain the prediction.

This version uses 1 Encoder layer, embedding dimension 32, and 4 attention heads, for **12,769 trainable parameters**. Gene identity is represented by embeddings; no positional encoding is added for the arbitrary gene column order, and no causal mask is used.

Expression values and masks in one batch have shape `(16, 128)`. Intermediate token vectors have shape `(16, 128, 32)`, and final predictions return to `(16, 128)`.

## How do we tell whether the model helps?

The comparison is a **gene-mean baseline**: predict each gene's mean in the training set without looking at other genes in the same cell.

Both methods use exactly the same data splits, targets, and masked positions. The Transformer's head is initialized to zero, so its epoch 0 output is exactly equal to the gene-mean baseline. Subsequent training attempts to learn corrections that depend on each cell's contents.

| Metric | Interpretation |
| --- | --- |
| Masked MSE | Mean squared error at masked positions only, used for training and model selection |
| MAE | Mean absolute error at masked positions; lower is better |
| RMSE | Root mean squared error at masked positions, giving larger errors more weight; lower is better |

All three measure **normalize-total + log1p expression values**, not raw counts. Evaluation reports all, originally zero, and originally nonzero positions separately so that the many zeros do not hide reconstruction difficulties.

## How do I run it?

See the [README quick start](README.md#quick-start) for initial environment setup and installation. Once the environment exists, run these commands from the project root:

```bash
source singlecell-venv/bin/activate

# Run correctness tests that require no data download.
python -m pytest -q

# Real PBMC3k: preprocessing, 5 training epochs, validation selection, and test evaluation.
python -m singlecell_baseline.train

# Exploration and QC/PCA/UMAP figures.
python -m singlecell_baseline.explore
```

You can also run `jupyter lab`, then read and execute the two notebooks in order.

## Recorded run results (2026-10-23)

Tests and full PBMC3k training were rerun in the existing Python 3.11.2 virtual environment on CPU. Both commands exited with code 0.

### Correctness tests

```text
.......                                                                  [100%]
7 passed in 4.63s
```

Tests use small synthetic counts and require no data download. They cover held-out values not affecting statistics fitted on training data, non-overlapping splits, preventing masked-target leakage, zero/nonzero metrics, a training run and checkpoint loading, and invalid masking ratios. The 7 cases include 3 parameterized cases for masking-ratio validation.

Full raw output: [test_output.txt](outputs/metrics/test_output.txt).

### Training on real PBMC3k data

```text
Epoch 01/5: train MSE=0.0684, val MSE=0.0559
Epoch 02/5: train MSE=0.0683, val MSE=0.0559
Epoch 03/5: train MSE=0.0705, val MSE=0.0559
Epoch 04/5: train MSE=0.0732, val MSE=0.0571
Epoch 05/5: train MSE=0.0646, val MSE=0.0561
Selected epoch: 0; artifacts: outputs
```

Epoch 0 has a validation MSE of **0.05578436**. None of the 5 training epochs achieved a lower value, so selection returned to epoch 0, which equals the gene-mean baseline; the last epoch was not selected.

| Masked test positions | Count | Gene mean MAE | Transformer MAE | Gene mean RMSE | Transformer RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 7,524 | 0.066361 | 0.066361 | 0.250510 | 0.250510 |
| Originally zero | 7,376 | 0.033832 | 0.033832 | 0.039179 | 0.039179 |
| Originally nonzero | 148 | 1.687506 | 1.687506 | 1.764610 | 1.764610 |

Approximately 98% of masked positions were originally zero, so overall error alone cannot establish that the model helps. The Transformer currently **does not outperform the gene-mean baseline**, and reconstruction errors for nonzero values are substantially larger. Test predictions are exactly identical before saving and after reloading; `checkpoint_reload_identical` is `true`.

Full training output: [baseline_output.txt](outputs/metrics/baseline_output.txt).

![Learning curve from this training run](outputs/figures/learning_curve.png)

## Where can I find the code and outputs?

| File / directory | Contents |
| --- | --- |
| [notebooks/01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb) | Data exploration, QC, counts versus processed values, and PCA/UMAP |
| [notebooks/02_masked_expression_transformer.ipynb](notebooks/02_masked_expression_transformer.ipynb) | Tensors, masking, gradient updates, full training, and evaluation |
| [singlecell_baseline/data.py](singlecell_baseline/data.py) | QC, splitting, feature selection fitted on training data only, and normalization |
| [singlecell_baseline/model.py](singlecell_baseline/model.py) | Masking, model architecture, and masked MSE |
| [singlecell_baseline/train.py](singlecell_baseline/train.py) | Training, baseline comparison, evaluation, and checkpoint reading/writing |
| [tests/test_baseline.py](tests/test_baseline.py) | Tests for preprocessing without data leakage, masking, metrics, training, and loading |
| [outputs/metrics/evaluation.csv](outputs/metrics/evaluation.csv) | MAE/RMSE by model, split, and zero/nonzero group |
| [outputs/metrics/history.csv](outputs/metrics/history.csv) | Training/validation loss for each epoch |
| [outputs/metrics/results.json](outputs/metrics/results.json) | Model configuration, selected epoch, versions, and full metrics |
| `outputs/figures/` | QC, PCA, UMAP, and learning curve |
| `outputs/checkpoints/best.pt` | Local checkpoint containing weights and an ordered gene list |

## What can these results establish, and what are the limits?

This project demonstrates an implementation workflow for single-cell data processing, a PyTorch Transformer, and reproducible evaluation. Passing tests means the code's key constraints behave as expected; it does not mean the model has outperformed the baseline.

The current task artificially masks values after normalizing a fully measured dataset. It therefore cannot establish recovery of genuinely missing RNA counts. UMAP clusters are not independent cell-type labels. Splitting cells from a single donor supports only a limited evaluation within the dataset.
