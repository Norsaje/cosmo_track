# ruff: noqa: E402
# Optional dependency skip должен произойти до импорта torch-моделей.
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from veg_recovery.dl.artifacts import load_research_checkpoint, save_research_checkpoint
from veg_recovery.dl.fixtures import fixture_datasets
from veg_recovery.dl.models.tcn import ResidualTCN, TCNConfig, masked_loss
from veg_recovery.dl.training import (
    TrainingConfig,
    fit_tcn,
    predict_dataset,
    seed_everything,
)


def train_once(seed=42):
    train, inner = fixture_datasets()
    seed_everything(seed)
    model = ResidualTCN(
        TCNConfig(
            train.input_features,
            len(train.preprocessor.crops) + 1,
            hidden_size=16,
            layers=2,
        )
    )
    result = fit_tcn(
        model,
        train,
        inner,
        TrainingConfig(seed=seed, epochs=3, patience=2, batch_size=8),
    )
    predictions, _ = predict_dataset(model, inner)
    return model, train, inner, result, predictions


def test_cpu_deterministic_training_and_reload(tmp_path):
    model, train, inner, result, predictions = train_once()
    again = train_once()
    np.testing.assert_allclose(
        predictions.primary_ndvi_pred, again[-1].primary_ndvi_pred, rtol=0, atol=1e-7
    )
    assert result["best_epoch"] == again[-2]["best_epoch"]
    assert result["parameters"] < 2_000_000
    path = tmp_path / "research"
    manifest = save_research_checkpoint(
        path, model, train.preprocessor, {"kind": "synthetic_smoke"}
    )
    assert manifest["production_approved"] is False
    restored, preprocessor, _ = load_research_checkpoint(path)
    assert preprocessor.fingerprint == train.preprocessor.fingerprint
    after, _ = predict_dataset(restored, inner)
    np.testing.assert_array_equal(
        predictions.primary_ndvi_pred, after.primary_ndvi_pred
    )
    (path / "weights.pt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="SHA256"):
        load_research_checkpoint(path)


def test_loss_never_touches_natural_missing_or_padding():
    prediction = torch.tensor([[0.4, 0.5, 1e9]], requires_grad=True)
    labels = torch.tensor([[float("nan"), 0.6, float("nan")]])
    mask = torch.tensor([[False, True, False]])
    loss = masked_loss(prediction, labels, mask, kind="mse")
    loss.backward()
    assert loss.item() == pytest.approx(0.01)
    np.testing.assert_allclose(prediction.grad.numpy(), [[0, -0.2, 0]], atol=1e-6)
    with pytest.raises(ValueError, match="nonempty"):
        masked_loss(prediction, labels, torch.zeros_like(mask))


def test_no_train_inner_overlap():
    train, _ = fixture_datasets()
    model = ResidualTCN(
        TCNConfig(train.input_features, len(train.preprocessor.crops) + 1)
    )
    with pytest.raises(ValueError, match="overlap"):
        fit_tcn(model, train, train, TrainingConfig(epochs=1))


def test_initial_prediction_is_exact_linear_base():
    train, _ = fixture_datasets()
    model = ResidualTCN(
        TCNConfig(train.input_features, len(train.preprocessor.crops) + 1)
    )
    predictions, _ = predict_dataset(model, train)
    bases = [train[i]["base"][train.center_index] for i in range(len(train))]
    np.testing.assert_array_equal(predictions.primary_ndvi_pred, bases)
