"""Checks for evaluation validity and a tiny offline end-to-end training run."""

import anndata as ad
import numpy as np
import pytest
from scipy import sparse
import torch

from singlecell_baseline.data import DataConfig, prepare_data
from singlecell_baseline.model import ExpressionTransformer, make_mask, masked_mse
from singlecell_baseline.train import TrainConfig, error_metrics, load_checkpoint, run_training


def toy_counts():
    rng = np.random.default_rng(7)
    counts = rng.poisson(rng.uniform(.2, 2, size=(80, 64))).astype(np.float32)
    raw = ad.AnnData(sparse.csr_matrix(counts))
    raw.obs_names = [f"cell_{i}" for i in range(raw.n_obs)]
    raw.var_names = [f"gene_{i}" for i in range(raw.n_vars)]
    return raw


def toy_config():
    return DataConfig(n_genes=16, min_genes=0, max_genes=1000, max_pct_mt=100)


def test_held_out_values_cannot_change_fitted_features_or_train_targets():
    raw = toy_counts()
    prepared = prepare_data(raw, toy_config())
    altered = raw.copy()
    altered.X = altered.X.toarray()
    held_out = np.concatenate([prepared.indices["validation"], prepared.indices["test"]])
    altered.X[held_out] *= 100
    second = prepare_data(altered, toy_config())
    assert prepared.adata.var_names.tolist() == second.adata.var_names.tolist()
    np.testing.assert_array_equal(prepared.values("train"), second.values("train"))
    np.testing.assert_array_equal(prepared.adata.var.train_mean, second.adata.var.train_mean)
    splits = [set(rows) for rows in prepared.indices.values()]
    assert not (splits[0] & splits[1] or splits[0] & splits[2] or splits[1] & splits[2])
    assert set.union(*splits) == set(range(raw.n_obs))
    assert sparse.issparse(prepared.adata.layers["counts"])


def test_masks_hide_targets_but_keep_visible_zeros_distinguishable():
    mask = make_mask((4, 16), .15, seed=42)
    assert torch.equal(mask, make_mask((4, 16), .15, seed=42))
    assert torch.all(mask.sum(dim=1) == 2)
    model = ExpressionTransformer(16, d_model=8, n_heads=2).eval()
    # Activate the head so leakage checks are not vacuous at zero initialization.
    torch.nn.init.normal_(model.head.weight)
    target = torch.rand(4, 16)
    changed = target.clone()
    changed[mask] = 9999
    with torch.no_grad():
        torch.testing.assert_close(model(target, mask), model(changed, mask), rtol=0, atol=0)
        assert model(target, mask).shape == target.shape
    loss = masked_mse(torch.tensor([[2., 100.]]), torch.tensor([[1., 0.]]),
                      torch.tensor([[True, False]]))
    assert loss.item() == 1
    assert not torch.equal(model.value_mask_encoding(torch.tensor([0., 0.])),
                           model.value_mask_encoding(torch.tensor([0., 1.])))


def test_zero_and_nonzero_metrics_and_empty_subsets():
    result = error_metrics(torch.tensor([[1., 1., 999.]]), torch.tensor([[0., 3., 1.]]),
                           torch.tensor([[True, True, False]]))
    assert result["all"]["n"] == 2
    assert result["all"]["mae"] == 1.5
    assert result["zero"]["rmse"] == 1
    assert result["nonzero"]["rmse"] == 2
    empty = error_metrics(torch.zeros(1, 2), torch.ones(1, 2), torch.ones(1, 2, dtype=torch.bool))
    assert empty["zero"] == {"n": 0, "mae": None, "rmse": None}


def test_offline_training_and_checkpoint_round_trip(tmp_path):
    prepared = prepare_data(toy_counts(), toy_config())
    report = run_training(prepared, TrainConfig(epochs=1, batch_size=8, d_model=8,
                          n_heads=2, threads=1), tmp_path)
    model, checkpoint = load_checkpoint(tmp_path / "checkpoints" / "best.pt")
    assert checkpoint["genes"] == prepared.adata.var_names.tolist()
    assert report["checkpoint_reload_identical"]
    assert report["metrics"]["test"]["transformer"]["all"]["n"] == 24
    assert model.gene_mean.shape == (16,)
    assert (tmp_path / "figures" / "learning_curve.png").is_file()


@pytest.mark.parametrize("fraction", [0, 1, -0.1])
def test_invalid_mask_fractions(fraction):
    with pytest.raises(ValueError):
        make_mask((2, 16), fraction, seed=42)
