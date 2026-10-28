"""A short CPU-friendly training loop with fixed held-out masks and a mean baseline."""

import argparse
import copy
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import DataConfig, PreparedData, load_pbmc3k, prepare_data, save_prepared
from .model import ExpressionTransformer, make_mask, masked_mse


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 0.001
    mask_fraction: float = 0.15
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 1
    dropout: float = 0.1
    seed: int = 42
    device: str = "cpu"
    threads: int = 4


def error_metrics(prediction: torch.Tensor, target: torch.Tensor,
                  mask: torch.Tensor) -> dict:
    result = {}
    for name, positions in {
        "all": mask, "zero": mask & (target == 0), "nonzero": mask & (target != 0)
    }.items():
        error = (prediction[positions] - target[positions]).double()
        result[name] = {
            "n": error.numel(),
            "mae": error.abs().mean().item() if error.numel() else None,
            "rmse": error.square().mean().sqrt().item() if error.numel() else None,
        }
    return result


@torch.no_grad()
def predict(model: ExpressionTransformer, target: torch.Tensor, mask: torch.Tensor,
            batch_size: int, device: str) -> torch.Tensor:
    model.eval()
    return torch.cat([
        model(target[start:start + batch_size].to(device),
              mask[start:start + batch_size].to(device)).cpu()
        for start in range(0, len(target), batch_size)
    ])


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[ExpressionTransformer, dict]:
    """Returns a model plus the ordered gene vocabulary and preprocessing metadata."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = ExpressionTransformer(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def run_training(prepared: PreparedData, config: TrainConfig,
                 output: str | Path = "outputs") -> dict:
    if config.epochs < 1 or config.batch_size < 1 or config.threads < 1 or config.learning_rate <= 0:
        raise ValueError("epochs, batch_size, threads and learning_rate must be positive.")
    if config.device not in ("cpu", "mps"):
        raise ValueError("This baseline supports cpu or explicit mps.")
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable on this machine; use --device cpu.")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    output = Path(output)
    for name in ("metrics", "figures", "checkpoints"):
        (output / name).mkdir(parents=True, exist_ok=True)
    targets = {split: torch.from_numpy(prepared.values(split)) for split in prepared.indices}
    masks = {
        split: make_mask(tuple(targets[split].shape), config.mask_fraction, config.seed + offset)
        for split, offset in (("validation", 1), ("test", 2))
    }
    np.savez_compressed(output / "masks.npz", **{k: v.numpy() for k, v in masks.items()})
    model_config = {
        "n_genes": prepared.adata.n_vars, "d_model": config.d_model,
        "n_heads": config.n_heads, "n_layers": config.n_layers, "dropout": config.dropout,
    }
    model = ExpressionTransformer(**model_config).to(config.device)
    gene_mean = targets["train"].mean(dim=0)
    model.gene_mean.copy_(gene_mean.to(config.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    loader = DataLoader(
        TensorDataset(targets["train"]), batch_size=config.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(config.seed), num_workers=0,
    )
    baseline_val = gene_mean.expand_as(targets["validation"])
    baseline_val_mse = masked_mse(baseline_val, targets["validation"], masks["validation"]).item()
    best_mse, best_epoch = baseline_val_mse, 0
    best_state = copy.deepcopy(model.state_dict())
    history = [{"epoch": 0, "train_mse": None, "validation_mse": best_mse,
                "gene_mean_validation_mse": baseline_val_mse}]
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        squared_error, masked_count = 0.0, 0
        for batch_index, (target,) in enumerate(loader):
            mask = make_mask(tuple(target.shape), config.mask_fraction,
                             config.seed + 10_000 * epoch + batch_index)
            target, mask = target.to(config.device), mask.to(config.device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_mse(model(target, mask), target, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            count = mask.sum().item()
            squared_error += loss.item() * count
            masked_count += count
        validation_prediction = predict(
            model, targets["validation"], masks["validation"], config.batch_size, config.device
        )
        val_mse = masked_mse(validation_prediction, targets["validation"], masks["validation"]).item()
        train_mse = squared_error / masked_count
        if not np.isfinite([train_mse, val_mse]).all():
            raise RuntimeError("Non-finite loss; inspect inputs and learning rate.")
        history.append({"epoch": epoch, "train_mse": train_mse, "validation_mse": val_mse,
                        "gene_mean_validation_mse": baseline_val_mse})
        if val_mse < best_mse:
            best_mse, best_epoch = val_mse, epoch
            best_state = copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch:02d}/{config.epochs}: train MSE={train_mse:.4f}, val MSE={val_mse:.4f}",
              flush=True)
    model.load_state_dict(best_state)
    model.eval()
    checkpoint_path = output / "checkpoints" / "best.pt"
    torch.save({
        "state_dict": {k: v.cpu() for k, v in best_state.items()},
        "model_config": model_config, "training_config": asdict(config),
        "genes": prepared.adata.var_names.tolist(), "preprocessing": prepared.report,
        "best_epoch": best_epoch, "best_validation_mse": best_mse,
    }, checkpoint_path)
    # Test is evaluated only after model selection has finished.
    reloaded, _ = load_checkpoint(checkpoint_path, config.device)
    results, rows = {}, []
    for split in ("validation", "test"):
        target, mask = targets[split], masks[split]
        prediction = predict(model, target, mask, config.batch_size, config.device)
        if split == "test":
            restored = predict(reloaded, target, mask, config.batch_size, config.device)
            torch.testing.assert_close(prediction, restored, rtol=0, atol=0)
        results[split] = {
            "gene_mean": error_metrics(gene_mean.expand_as(target), target, mask),
            "transformer": error_metrics(prediction, target, mask),
        }
        for name, subsets in results[split].items():
            for subset, metrics in subsets.items():
                rows.append({"split": split, "model": name, "subset": subset, **metrics})
    report = {
        "training_config": asdict(config), "model_config": model_config,
        "best_epoch": best_epoch, "best_validation_mse": best_mse,
        "parameters": sum(p.numel() for p in model.parameters()),
        "training_and_evaluation_seconds": round(time.perf_counter() - started, 2),
        "checkpoint_reload_identical": True,
        "mask_seeds": {"validation": config.seed + 1, "test": config.seed + 2},
        "masked_genes_per_cell": int(masks["test"][0].sum()),
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            **{name: version(name) for name in ("torch", "scanpy", "anndata", "numpy", "scikit-learn")},
        },
        "metrics": results,
    }
    (output / "metrics" / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(output / "metrics" / "evaluation.csv", index=False)
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output / "metrics" / "history.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history_frame.epoch, history_frame.train_mse, label="Train (resampled masks)")
    ax.plot(history_frame.epoch, history_frame.validation_mse, label="Validation (fixed masks)")
    ax.axhline(baseline_val_mse, color="gray", linestyle="--", label="Gene-mean validation")
    ax.axvline(best_epoch, color="black", alpha=0.2, label=f"Selected epoch {best_epoch}")
    ax.set(xlabel="Epoch", ylabel="Masked MSE (log1p expression)", title="PBMC3k learning curve")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "figures" / "learning_curve.png", dpi=150)
    plt.close(fig)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--n-genes", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    prepared = prepare_data(load_pbmc3k(args.data_dir / "raw"),
                            DataConfig(n_genes=args.n_genes, seed=args.seed))
    save_prepared(prepared, args.output, args.data_dir / "processed")
    report = run_training(prepared, TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
        device=args.device, threads=args.threads,
    ), args.output)
    print(pd.read_csv(args.output / "metrics" / "evaluation.csv").to_string(index=False))
    print(f"Selected epoch: {report['best_epoch']}; artifacts: {args.output}")


if __name__ == "__main__":
    main()
