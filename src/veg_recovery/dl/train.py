"""CPU/CUDA runner. Реальная CV разрешена только с проверенным handoff C-03.

Example: python -m veg_recovery.dl.train --smoke --output artifacts/dl/smoke
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .artifacts import environment_info, file_sha256, save_research_checkpoint
from .data import (
    KEY,
    WindowDatasetAdapter,
    WindowPreprocessor,
    canonical_keys,
    key_index,
    labels_for,
    prepare_context,
)
from .evaluation import (
    align_oof,
    assert_metric_equality,
    metrics,
    paired_polygon_bootstrap,
    validate_oof,
)


def _read_artifact(root, spec):
    if not isinstance(spec, dict) or set(spec) != {"path", "sha256"}:
        raise ValueError("Each C-03 input needs explicit path and sha256")
    path = (root / spec["path"]).resolve()
    if not path.is_file():
        raise ValueError(f"Missing ML handoff artifact: {path}")
    if file_sha256(path) != spec["sha256"]:
        raise ValueError(f"ML input fingerprint mismatch: {path}")
    return pd.read_csv(path)


def preflight(manifest_path):
    path = Path(manifest_path)
    if not path.is_file():
        raise ValueError(
            "C-03 handoff is missing. Publish ML folds/MaskSpec/OOF; do not generate a DL split."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "dl-c03-consumer-0.1"
        or manifest.get("producer") != "ML"
        or manifest.get("review_status") != "accepted"
    ):
        raise ValueError(
            "C-03 consumer manifest must be reviewed by ML; draft is not a training input"
        )
    for field in (
        "fold_version",
        "mask_version",
        "mask_callable",
        "baseline_metrics",
        "folds",
    ):
        if not manifest.get(field):
            raise ValueError(f"Missing C-03 field: {field}")
    frame = _read_artifact(path.parent, manifest["data"])
    frame[KEY] = canonical_keys(frame)
    baseline = validate_oof(_read_artifact(path.parent, manifest["baseline_oof"]))
    assert_metric_equality(baseline, manifest["baseline_metrics"])
    module, function = manifest["mask_callable"].split(":", 1)
    if not module.startswith("veg_recovery.validation."):
        raise ValueError(
            "Mask callable must belong to ML-owned veg_recovery.validation"
        )
    try:
        masking = getattr(importlib.import_module(module), function)
    except (ImportError, AttributeError) as exc:
        raise ValueError(
            "Shared ML MaskSpec/apply_mask adapter is unavailable; C-03 is not ready"
        ) from exc
    folds, eval_key_frames = [], []
    seen_folds = set()
    for config in manifest["folds"]:
        fold_id = (str(config["split"]), str(config["fold"]))
        if fold_id in seen_folds:
            raise ValueError("Duplicate split/fold in C-03 consumer manifest")
        seen_folds.add(fold_id)
        if config.get("context_policy") not in {"interpolation", "causal"}:
            raise ValueError("ML must explicitly state inference context_policy")
        keys = {
            name: canonical_keys(_read_artifact(path.parent, config[name]))
            for name in (
                "fit_context_keys",
                "inference_context_keys",
                "train_target_keys",
                "inner_target_keys",
                "evaluation_keys",
            )
        }
        for name, value in keys.items():
            if value.empty or not key_index(value).isin(key_index(frame)).all():
                raise ValueError(f"Empty or absent {name} for {fold_id}")
        train, inner, evaluation = [
            key_index(keys[k])
            for k in ("train_target_keys", "inner_target_keys", "evaluation_keys")
        ]
        fit, inference = (
            key_index(keys["fit_context_keys"]),
            key_index(keys["inference_context_keys"]),
        )
        if (
            not train.isin(fit).all()
            or inner.isin(fit).any()
            or evaluation.isin(fit).any()
        ):
            raise ValueError(
                "Fit context must include training targets and exclude inner/outer labels"
            )
        if (
            inner.isin(evaluation).any()
            or train.isin(inner).any()
            or train.isin(evaluation).any()
        ):
            raise ValueError("Train/inner/outer target keys must be disjoint")
        if not inner.isin(inference).all() or not evaluation.isin(inference).all():
            raise ValueError(
                "Inference context must contain all inner and evaluation keys"
            )
        if fold_id[0] == "unseen":
            if (
                keys["fit_context_keys"]
                .anon_polygon_id.isin(keys["evaluation_keys"].anon_polygon_id)
                .any()
            ):
                raise ValueError("Unseen fold fit context contains evaluation polygons")
        if config["context_policy"] == "causal":
            # Консервативный общий контекст: не позже первой query date каждого polygon.
            limits = (
                pd.concat([keys["inner_target_keys"], keys["evaluation_keys"]])
                .groupby("anon_polygon_id")
                .date.min()
            )
            context_dates = keys["inference_context_keys"]
            limit = context_dates.anon_polygon_id.map(limits)
            is_target = key_index(context_dates).isin(inner.union(evaluation))
            if (context_dates.date.gt(limit) & ~is_target).any():
                raise ValueError("Causal context contains future observations")
        eval_key_frames.append(
            keys["evaluation_keys"].assign(split=config["split"], fold=config["fold"])
        )
        folds.append((config, keys))
    eval_keys = pd.concat(eval_key_frames, ignore_index=True)
    # Equality проверяет и точный набор keys, и labels; до создания модели и optimizer.
    labels = frame[KEY + ["primary_ndvi"]].rename(columns={"primary_ndvi": "y_true"})
    candidate = eval_keys.merge(labels, on=KEY, validate="many_to_one")
    candidate["primary_ndvi_pred"] = candidate.y_true
    align_oof(baseline, candidate)
    return manifest, frame, baseline, folds, masking


def _model(dataset, args, seed):
    from .models.tcn import ResidualTCN, TCNConfig
    from .training import seed_everything

    seed_everything(seed, args.cpu_threads)
    return ResidualTCN(
        TCNConfig(
            dataset.input_features,
            len(dataset.preprocessor.crops) + 1,
            hidden_size=args.hidden_size,
            layers=args.layers,
            residual_bound=args.residual_bound,
        )
    )


def _train_config(args, seed):
    from .training import TrainingConfig

    return TrainingConfig(
        seed=seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        loss=args.loss,
        device=args.device,
        cpu_threads=args.cpu_threads,
    )


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(
            value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        ),
        encoding="utf-8",
    )


def run_smoke(args):
    from .fixtures import fixture_datasets
    from .training import fit_tcn, predict_dataset
    from .artifacts import load_research_checkpoint

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for seed in args.seeds:
        train, inner = fixture_datasets(window_days=args.window)
        model = _model(train, args, seed)
        result = fit_tcn(model, train, inner, _train_config(args, seed))
        predictions, infer_sec = predict_dataset(model, inner, device=args.device)
        checkpoint = output / f"seed_{seed}"
        save_research_checkpoint(
            checkpoint,
            model,
            train.preprocessor,
            {
                "kind": "synthetic_smoke_not_competition_CV",
                "seed": seed,
                "window": args.window,
            },
        )
        restored, _, _ = load_research_checkpoint(checkpoint, device=args.device)
        rerun, _ = predict_dataset(restored, inner, device=args.device)
        np.testing.assert_allclose(
            predictions.primary_ndvi_pred, rerun.primary_ndvi_pred, atol=1e-7, rtol=0
        )
        predictions.to_csv(checkpoint / "fixture_predictions.csv", index=False)
        results.append(
            {
                "seed": seed,
                **result,
                "infer_sec": infer_sec,
                "checkpoint_bytes": (checkpoint / "weights.pt").stat().st_size,
                "reload_max_abs_error": float(
                    np.max(
                        np.abs(predictions.primary_ndvi_pred - rerun.primary_ndvi_pred)
                    )
                ),
            }
        )
    report = {
        "kind": "synthetic_smoke_not_competition_CV",
        "decision": "PENDING_EVALUATION",
        "environment": environment_info(),
        "results": results,
    }
    _write_json(output / "smoke_report.json", report)
    print(
        json.dumps(
            {
                "kind": report["kind"],
                "output": str(output),
                "seeds": args.seeds,
                "reload": "passed",
            }
        )
    )
    return 0


def run_cv(args):
    manifest, frame, baseline, folds, apply_mask = preflight(args.fold_manifest)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "C03_PREFLIGHT_PASSED",
                    "folds": len(folds),
                    "fold_version": manifest["fold_version"],
                }
            )
        )
        return 0
    from .training import fit_tcn, predict_dataset

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "input_manifest.json", manifest)
    _write_json(output / "environment.json", environment_info())
    all_summaries = []
    indexed = frame.set_index(KEY)
    for seed in args.seeds:
        seed_oof = []
        for ordinal, (config, keys) in enumerate(folds):
            fold, split = config["fold"], config["split"]
            fit_frame = indexed.loc[key_index(keys["fit_context_keys"])].reset_index()
            infer_frame = indexed.loc[
                key_index(keys["inference_context_keys"])
            ].reset_index()
            train_context = prepare_context(
                fit_frame, keys["train_target_keys"], apply_mask=apply_mask
            )
            # Inner и outer скрыты вместе: early stopping никогда не видит outer labels в контексте.
            infer_hidden = pd.concat(
                [keys["inner_target_keys"], keys["evaluation_keys"]]
            )
            inner_context = prepare_context(
                infer_frame, infer_hidden, apply_mask=apply_mask
            )
            outer_context = prepare_context(
                infer_frame, keys["evaluation_keys"], apply_mask=apply_mask
            )
            prep = WindowPreprocessor.fit(train_context, keys["fit_context_keys"])
            train = WindowDatasetAdapter(
                train_context,
                keys["train_target_keys"],
                prep,
                window_days=args.window,
                labels=labels_for(frame, keys["train_target_keys"]),
            )
            inner = WindowDatasetAdapter(
                inner_context,
                keys["inner_target_keys"],
                prep,
                window_days=args.window,
                labels=labels_for(frame, keys["inner_target_keys"]),
            )
            outer = WindowDatasetAdapter(
                outer_context, keys["evaluation_keys"], prep, window_days=args.window
            )
            model = _model(train, args, seed)
            result = fit_tcn(model, train, inner, _train_config(args, seed))
            predictions, infer_sec = predict_dataset(model, outer, device=args.device)
            reference = baseline.loc[
                baseline["split"].eq(split) & baseline.fold.eq(fold)
            ]
            metadata = reference.drop(columns="primary_ndvi_pred")
            oof = metadata.merge(predictions, on=KEY, validate="one_to_one")
            align_oof(reference, oof)
            checkpoint = output / f"seed_{seed}_fold_{ordinal}"
            save_research_checkpoint(
                checkpoint,
                model,
                prep,
                {
                    "kind": "C03_OOF_research",
                    "seed": seed,
                    "fold": fold,
                    "split": split,
                    "window": args.window,
                    "data_fingerprint": manifest["data"]["sha256"],
                    "fold_version": manifest["fold_version"],
                    "mask_version": manifest["mask_version"],
                    "training": asdict(_train_config(args, seed)),
                    "result": result,
                    "infer_sec": infer_sec,
                },
            )
            oof.to_csv(checkpoint / "oof.csv", index=False)
            seed_oof.append(oof)
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "split": split,
                        "fold": fold,
                        "best_epoch": result["best_epoch"],
                        "rmse": metrics(oof)["overall_rmse"],
                    }
                ),
                flush=True,
            )
        oof = pd.concat(seed_oof, ignore_index=True)
        aligned = align_oof(baseline, oof)
        oof.to_csv(output / f"oof_seed_{seed}.csv", index=False)
        summary = {
            "seed": seed,
            "ml": metrics(baseline),
            "dl": metrics(oof),
            "polygon_bootstrap": paired_polygon_bootstrap(aligned),
        }
        all_summaries.append(summary)
    _write_json(
        output / "cv_report.json",
        {
            "decision": "PENDING_REVIEW",
            "environment": environment_info(),
            "seeds": all_summaries,
            "note": "No candidate export before subgroup/ensemble/integration review",
        },
    )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--smoke",
        action="store_true",
        help="Artificial fixture only; no competition score",
    )
    group.add_argument("--fold-manifest", help="ML-reviewed C-03 consumer manifest")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output", default="artifacts/dl/research_run")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 73])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--window", type=int, default=61)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--residual-bound", type=float, default=0.3)
    parser.add_argument("--loss", choices=["mse", "huber"], default="huber")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct")
    if args.smoke and args.preflight_only:
        parser.error("--preflight-only requires --fold-manifest")
    try:
        return run_smoke(args) if args.smoke else run_cv(args)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"DL runner: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
