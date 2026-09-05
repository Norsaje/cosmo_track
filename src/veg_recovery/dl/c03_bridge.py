"""Строит DL training manifest напрямую из опубликованных артефактов C-03.

Это **не** новая схема фолдов. Мост исполняет producer-код ML (`read_dataset`,
`MaskSpec.from_test`, `load_folds`, `split_fold`) из указанного корня проекта и
экспортирует те же ключи. Файлы ML не изменяются и не копируются.

Собственный вклад DL ограничен обучающей выборкой внутри разрешённого
`fit_frame`: train-блоки и inner-монитор. Outer keys, censoring и MaskSpec
берутся из зафиксированных ML-артефактов. `review_status` сохраняет происхождение
`derived_from_producer_artifacts`.
"""

from __future__ import annotations

import argparse
from functools import partial
import gzip
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from .artifacts import file_sha256
from .data import KEY, canonical_keys, key_index
from .evaluation import metrics
from .ml_handoff import MODE_MAP, adapt_baseline_oof

SCHEMA_VERSION = "dl-c03-consumer-0.2"
POLICY_MAP = {
    "transductive": "interpolation",
    "past_only": "causal",
    "one_sided_hard": "causal",
}
# Длины серий синтетических пропусков в реальном test_data.csv (3 112 ключей).
TEST_RUN_WEIGHTS = {1: 2827, 2: 136, 3: 3, 4: 1}


def censored_keys(raw: pd.DataFrame, context: pd.DataFrame, columns) -> pd.DataFrame:
    """Строки, которые producer скрыл: раньше значение было, теперь нет.

    Разностный признак не зависит от того, какие поля есть у естественно
    отсутствующих наблюдений, и не восстанавливает ни одного скрытого значения.
    """
    present = [c for c in columns if c in raw and c in context]
    if not present:
        raise ValueError("MaskSpec columns absent from producer frames")
    left = raw[present].reset_index(drop=True)
    right = context[present].reset_index(drop=True)
    removed = (left.notna().to_numpy() & right.isna().to_numpy()).any(axis=1)
    return canonical_keys(raw.loc[removed])


def _runs(frame: pd.DataFrame, rng, weights=TEST_RUN_WEIGHTS):
    """Серии подряд идущих наблюдений внутри polygon-year, как в test."""
    lengths = np.array(sorted(weights))
    probabilities = np.array([weights[k] for k in lengths], dtype=float)
    probabilities /= probabilities.sum()
    out = []
    ordered = frame.sort_values(KEY).reset_index(drop=True)
    for _, group in ordered.groupby(
        ["anon_polygon_id", ordered.date.dt.year], sort=True
    ):
        positions = group.index.to_numpy()
        start = 0
        while start < len(positions):
            length = min(int(rng.choice(lengths, p=probabilities)), len(positions) - start)
            out.append(positions[start : start + length])
            # Один видимый разделитель: серии не сливаются в один длинный пропуск.
            start += length + 1
    rng.shuffle(out)
    return ordered, out


def training_targets(
    fit_frame: pd.DataFrame,
    *,
    seed: int,
    blocks: int,
    inner_fraction: float,
    max_targets: int,
):
    """Train-блоки и inner-монитор строго внутри разрешённого fit_frame."""
    if blocks < 1 or not 0 < inner_fraction < 0.5 or max_targets < blocks:
        raise ValueError("Invalid training target configuration")
    values = pd.to_numeric(fit_frame.primary_ndvi, errors="raise").to_numpy(float)
    observed = fit_frame.loc[np.isfinite(values)]
    if observed.empty:
        raise ValueError("Fit frame has no supervised labels")
    rng = np.random.default_rng(seed)
    ordered, runs = _runs(observed, rng)
    n_inner = max(1, int(round(len(runs) * inner_fraction)))
    inner_runs, train_runs = runs[:n_inner], runs[n_inner:]
    if not train_runs:
        raise ValueError("No training runs left after the inner monitor split")
    budget, selected = max_targets, []
    for run in train_runs:
        if budget <= 0:
            break
        selected.append(run[:budget])
        budget -= len(run)
    train = pd.concat(
        [
            ordered.iloc[run][KEY].assign(block=int(i % blocks))
            for i, run in enumerate(selected)
            if len(run)
        ],
        ignore_index=True,
    )
    inner = pd.concat([ordered.iloc[run][KEY] for run in inner_runs], ignore_index=True)
    blocks_column = train.block.to_numpy()
    train, inner = canonical_keys(train), canonical_keys(inner)
    train["block"] = blocks_column
    if key_index(train).isin(key_index(inner)).any():
        raise ValueError("Train and inner monitor targets overlap")
    return train, inner


def _compress(path: Path, payload: bytes) -> dict:
    """mtime=0: архив байт-стабилен, поэтому SHA256 воспроизводим при перезапуске."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, compresslevel=9) as stream:
            stream.write(payload)
    return {"path": path.name, "sha256": file_sha256(path)}


def _write(path: Path, frame: pd.DataFrame) -> dict:
    frame = frame.copy()
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
    return _compress(path, frame.to_csv(index=False, lineterminator=chr(10)).encode())


def build(
    ml_root,
    output,
    *,
    blocks=6,
    inner_fraction=0.08,
    max_targets=12000,
    baseline_model="mean_neighbors",
    limit_folds=None,
):
    from veg_recovery.data.io import read_dataset
    from veg_recovery.validation.folds import FOLD_VERSION, load_folds, split_fold
    from veg_recovery.validation.masking import MaskSpec, apply_mask

    root, output = Path(ml_root).resolve(), Path(output)
    producer = Path(
        __import__(split_fold.__module__, fromlist=["__file__"]).__file__
    ).resolve()
    if not producer.is_relative_to(root):
        raise ValueError("Imported ML implementation is not the requested worktree")
    output.mkdir(parents=True, exist_ok=True)
    inputs = {
        "train": root / "data/train_dataset.csv",
        "test": root / "data/test_data.csv",
        "folds": root / "configs/ml/folds_v1.csv",
        "oof": root / "artifacts/ml/baseline_v1/oof_predictions.csv.gz",
    }
    train = read_dataset(inputs["train"], "train", strict_current=True)
    test = read_dataset(inputs["test"], "test", strict_current=True)
    folds = load_folds(inputs["folds"])
    spec = MaskSpec.from_test(test)
    mask = partial(apply_mask, spec=spec)
    baseline = adapt_baseline_oof(
        pd.read_csv(inputs["oof"], parse_dates=["date"]), baseline_model
    )
    baseline_metrics = metrics(baseline)

    data_entry = _compress(output / "frame.csv.gz", inputs["train"].read_bytes())
    baseline_entry = _write(output / "baseline_oof.csv.gz", baseline)
    context_entry = _write(output / "context_keys.csv.gz", canonical_keys(train))

    combinations = list(
        folds[["mode", "repeat", "fold"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if limit_folds:
        combinations = combinations[:limit_folds]
    exported = []
    for mode, repeat, fold in combinations:
        part = split_fold(train, folds, mode, int(repeat), int(fold), spec)
        hidden = censored_keys(train, part.context_frame, spec.columns)
        # Round-trip: наш censoring обязан воспроизвести producer context точно.
        replay = mask(train.copy(deep=True), hidden.copy())
        expected = part.context_frame.drop(
            columns=["is_synthetic_gap"], errors="ignore"
        )
        actual = replay.drop(columns=["is_synthetic_gap"], errors="ignore")
        pd.testing.assert_frame_equal(
            actual[expected.columns].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )
        if not key_index(part.gap_keys).isin(key_index(hidden)).all():
            raise ValueError("Producer censoring must contain every evaluation key")
        seed = int(part.metadata["seed"])
        targets, inner = training_targets(
            part.fit_frame,
            seed=seed + 9973,
            blocks=blocks,
            inner_fraction=inner_fraction,
            max_targets=max_targets,
        )
        fit_keys = canonical_keys(part.fit_frame)
        fit_keys = fit_keys.loc[~key_index(fit_keys).isin(key_index(inner))]
        if not key_index(targets).isin(key_index(fit_keys)).all():
            raise ValueError("Train targets must remain inside the allowed fit context")
        if key_index(fit_keys).isin(key_index(hidden)).any():
            raise ValueError("Fit context must not contain producer-censored rows")
        name = f"{MODE_MAP[mode]}_r{int(repeat)}_f{int(fold)}"
        entry = {
            "split": MODE_MAP[mode],
            "fold": f"r{int(repeat)}_f{int(fold)}",
            "producer_mode": mode,
            "producer_repeat": int(repeat),
            "producer_fold": int(fold),
            "producer_context_policy": part.metadata["context_policy"],
            "context_policy": POLICY_MAP[part.metadata["context_policy"]],
            "seed": seed,
            "train_blocks": blocks,
            "fit_context_keys": _write(output / f"{name}_fit.csv.gz", fit_keys),
            "inference_context_keys": dict(context_entry),
            "censored_context_keys": _write(output / f"{name}_censored.csv.gz", hidden),
            "train_target_keys": _write(output / f"{name}_train.csv.gz", targets),
            "inner_target_keys": _write(output / f"{name}_inner.csv.gz", inner),
            "evaluation_keys": _write(
                output / f"{name}_eval.csv.gz", canonical_keys(part.gap_keys)
            ),
            "counts": {
                "fit_rows": len(fit_keys),
                "censored_rows": len(hidden),
                "train_targets": len(targets),
                "inner_targets": len(inner),
                "evaluation": len(part.gap_keys),
            },
        }
        exported.append(entry)
        print(json.dumps({"fold": name, **entry["counts"]}), flush=True)

    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "producer": "ML",
        "review_status": "derived_from_producer_artifacts",
        "derivation": (
            "DL executed the producer split_fold/MaskSpec read-only. Outer keys and "
            "censoring are ML-owned; only train blocks and the inner monitor are DL-side."
        ),
        "producer_evidence": {
            "commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "input_sha256": {k: file_sha256(v) for k, v in inputs.items()},
            "producer_module": producer.name,
            "ml_acknowledgement": "pending",
        },
        "fold_version": FOLD_VERSION,
        "mask_version": spec.version,
        "mask_columns": list(spec.columns),
        "mask_callable": "veg_recovery.validation.masking:apply_mask",
        "baseline_model": baseline_model,
        "data": data_entry,
        "baseline_oof": baseline_entry,
        "baseline_metrics": {
            "overall_rmse": baseline_metrics["overall_rmse"],
            "gap_score": baseline_metrics["gap_score"],
            "split_rmse": baseline_metrics["split_rmse"],
            "composite_rmse": baseline_metrics["composite_rmse"],
        },
        "folds": exported,
    }
    (output / "dl_c03.json").write_text(
        json.dumps(
            manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ml-root", required=True)
    parser.add_argument("--output", default="artifacts/dl/c03_derived")
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--inner-fraction", type=float, default=0.08)
    parser.add_argument("--max-targets", type=int, default=12000)
    parser.add_argument("--baseline-model", default="mean_neighbors")
    parser.add_argument("--limit-folds", type=int, default=None)
    args = parser.parse_args(argv)
    manifest = build(
        args.ml_root,
        args.output,
        blocks=args.blocks,
        inner_fraction=args.inner_fraction,
        max_targets=args.max_targets,
        baseline_model=args.baseline_model,
        limit_folds=args.limit_folds,
    )
    print(
        json.dumps(
            {
                "folds": len(manifest["folds"]),
                "output": str(Path(args.output) / "dl_c03.json"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
