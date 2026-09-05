"""Общий CPU/CUDA цикл; caller обязан передать train и inner early-stop данные."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .models.tcn import masked_loss


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 40
    patience: int = 6
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.001
    gradient_clip: float = 1.0
    loss: str = "huber"
    device: str = "cpu"
    cpu_threads: int = 2
    epoch_policy: str = "early_stop"

    def __post_init__(self):
        if (
            min(self.epochs, self.patience, self.batch_size, self.cpu_threads) < 1
            or self.learning_rate <= 0
            or self.weight_decay < 0
            or self.gradient_clip <= 0
            or self.loss not in {"huber", "mse"}
            or self.epoch_policy not in {"early_stop", "fixed"}
        ):
            raise ValueError("Invalid training configuration")


def seed_everything(seed: int, cpu_threads: int = 2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _collate(samples):
    fields = (
        "features",
        "crop_ids",
        "base",
        "valid_calendar_mask",
        "labels",
        "loss_mask",
    )
    return {
        name: torch.from_numpy(np.stack([s[name] for s in samples])) for name in fields
    }


def _forward(model, batch):
    return model(
        batch["features"],
        batch["crop_ids"],
        batch["base"],
        batch["valid_calendar_mask"],
    )


def predict_dataset(model, dataset, *, device="cpu", batch_size=64):
    model = model.to(device)
    model.eval()
    outputs = []
    started = time.perf_counter()
    with torch.no_grad():
        for batch in DataLoader(
            dataset, batch_size=batch_size, collate_fn=_collate, shuffle=False
        ):
            batch = {key: value.to(device) for key, value in batch.items()}
            prediction = _forward(model, batch)[:, dataset.center_index]
            outputs.append(prediction.cpu().numpy())
    values = np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)
    return dataset.predictions_frame(values), time.perf_counter() - started


def fit_tcn(model, train_dataset, early_stop_dataset, config: TrainingConfig):
    """`fixed` не выбирает epoch: inner RMSE только логируется как монитор.

    Это политика по умолчанию для конкурсной CV, пока ML не передал inner keys.
    """
    from .data import key_index
    from torch.utils.data import ConcatDataset

    parts = (
        list(train_dataset.datasets)
        if isinstance(train_dataset, ConcatDataset)
        else [train_dataset]
    )
    reference = parts[0]
    for part in parts:
        if key_index(part.keys).isin(key_index(early_stop_dataset.keys)).any():
            raise ValueError("Training and early-stop labels overlap")
        if part.preprocessor.fingerprint != reference.preprocessor.fingerprint:
            raise ValueError("All training blocks must share one fitted preprocessor")
        if not np.isfinite(part.y).all():
            raise ValueError("Supervised train labels are required")
    if len(train_dataset) == 0 or len(early_stop_dataset) == 0:
        raise ValueError("Training and early-stop datasets must be nonempty")
    if reference.preprocessor.fingerprint != early_stop_dataset.preprocessor.fingerprint:
        raise ValueError(
            "Train and validation must share the train-fitted preprocessor"
        )
    if not np.isfinite(early_stop_dataset.y).all():
        raise ValueError("Supervised early-stop labels are required")
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable: use --device cpu or the documented Kaggle GPU runner"
        )
    seed_everything(config.seed, config.cpu_threads)
    model.to(config.device)
    if sum(p.numel() for p in model.parameters()) >= 2_000_000:
        raise ValueError("P0 model budget is < 2 million parameters")
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=_collate,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best, best_epoch, best_state, stale = np.inf, 0, None, 0
    history = []
    started = time.perf_counter()
    if config.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss, count = 0.0, 0
        for batch in loader:
            batch = {key: value.to(config.device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(
                _forward(model, batch),
                batch["labels"],
                batch["loss_mask"],
                kind=config.loss,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
            n = int(batch["loss_mask"].sum())
            train_loss += float(loss.detach()) * n
            count += n
        predictions, _ = predict_dataset(
            model,
            early_stop_dataset,
            device=config.device,
            batch_size=config.batch_size,
        )
        error = float(
            np.sqrt(
                np.mean(
                    (predictions.primary_ndvi_pred.to_numpy() - early_stop_dataset.y)
                    ** 2
                )
            )
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss / count, "inner_rmse": error}
        )
        if config.epoch_policy == "fixed":
            best, best_epoch, stale = error, epoch, 0
            best_state = copy.deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
            continue
        if error < best - 1e-8:
            best, best_epoch, stale = error, epoch, 0
            best_state = copy.deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
        else:
            stale += 1
        if stale >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("No finite checkpoint")
    model.load_state_dict(best_state)
    peak = (
        torch.cuda.max_memory_allocated() / 2**20
        if config.device.startswith("cuda")
        else 0.0
    )
    return {
        "epoch_policy": config.epoch_policy,
        "best_epoch": best_epoch,
        "inner_rmse": best,
        "history": history,
        "train_sec": time.perf_counter() - started,
        "peak_vram_mb": peak,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
