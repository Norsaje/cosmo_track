"""Read-only review of published ML folds/OOF; does not invent DL training splits."""

import argparse
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from .artifacts import file_sha256
from .data import (
    KEY,
    WindowDatasetAdapter,
    WindowPreprocessor,
    key_index,
    prepare_context,
)
from .evaluation import align_oof, metrics, validate_oof

MODE_MAP = {"A": "matched", "B": "unseen", "C": "temporal", "D": "hard"}


def bundle_integrity(directory):
    """Diagnose EOL transport damage, without bypassing producer hash validation."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    results = []
    for entry in manifest["model_files"]:
        path = (directory / entry["path"]).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Bundle member escapes artifact directory")
        data = path.read_bytes()
        actual = sha256(data).hexdigest()
        results.append(
            {
                "path": entry["path"],
                "expected_sha256": entry["sha256"],
                "checkout_sha256": actual,
                "matches": actual == entry["sha256"],
                "lf_normalized_matches": sha256(
                    data.replace(b"\r\n", b"\n")
                ).hexdigest()
                == entry["sha256"],
            }
        )
    return {
        "loadable_hashes": bool(results) and all(item["matches"] for item in results),
        "files": results,
        "repair_applied": False,
    }


def adapt_baseline_oof(frame, model="mean_neighbors"):
    """Preserve repeat/fold identity and labels from the producer, without joins."""
    required = {
        *KEY,
        "mode",
        "repeat",
        "fold",
        "primary_ndvi",
        "seen_polygon",
        f"pred_{model}",
    }
    if not required.issubset(frame):
        raise ValueError(f"ML OOF missing columns: {sorted(required - set(frame))}")
    if not frame["mode"].isin(MODE_MAP).all():
        raise ValueError("Unknown ML CV mode")
    for column in ("repeat", "fold"):
        numbers = pd.to_numeric(frame[column], errors="raise")
        if (
            not np.isfinite(numbers).all()
            or not (numbers >= 0).all()
            or not numbers.eq(numbers.astype(int)).all()
        ):
            raise ValueError("ML repeat/fold must be nonnegative integers")
    if (
        frame.seen_polygon.isna().any()
        or not frame.seen_polygon.isin([True, False]).all()
    ):
        raise ValueError("seen_polygon must be boolean")
    out = frame.rename(
        columns={"primary_ndvi": "y_true", "source_label": "source"}
    ).copy()
    out["split"] = out["mode"].map(MODE_MAP)
    out["fold"] = (
        "r"
        + frame["repeat"].astype(int).astype(str)
        + "_f"
        + frame["fold"].astype(int).astype(str)
    )
    out["primary_ndvi_pred"] = frame[f"pred_{model}"]
    out["is_unseen"] = ~frame.seen_polygon.astype(bool)
    out["is_hard"] = frame["mode"].eq("D")
    return validate_oof(out)


def audit_handoff(root, output):
    # Import the actual producer through PYTHONPATH; never vendor its implementation.
    from veg_recovery.data.io import read_dataset
    from veg_recovery.validation.folds import load_folds, split_fold
    from veg_recovery.validation.masking import MaskSpec, apply_mask

    root, output = Path(root).resolve(), Path(output)
    producer_source = Path(
        __import__(split_fold.__module__, fromlist=["__file__"]).__file__
    ).resolve()
    if not producer_source.is_relative_to(root):
        raise ValueError(
            "Imported ML implementation differs from the requested worktree"
        )
    inputs = {
        "train": root / "data/train_dataset.csv",
        "test": root / "data/test_data.csv",
        "folds": root / "configs/ml/folds_v1.csv",
        "oof": root / "artifacts/ml/baseline_v1/oof_predictions.csv.gz",
        "metrics": root / "artifacts/ml/baseline_v1/cv_summary.csv",
        "decisions": root / "artifacts/ml/baseline_v1/baseline_decisions.json",
    }
    train = read_dataset(inputs["train"], "train", strict_current=True)
    test = read_dataset(inputs["test"], "test", strict_current=True)
    folds = load_folds(inputs["folds"])
    spec = MaskSpec.from_test(test)
    mask = partial(apply_mask, spec=spec)
    raw_oof = pd.read_csv(inputs["oof"], parse_dates=["date"])
    baseline = adapt_baseline_oof(raw_oof)
    actual = metrics(baseline)
    published = pd.read_csv(inputs["metrics"])
    for row in published.loc[published.model.eq("mean_neighbors")].itertuples():
        measured = metrics(baseline.loc[baseline.split.eq(MODE_MAP[row.mode])])
        np.testing.assert_allclose(
            measured["overall_rmse"], row.rmse, atol=1e-12, rtol=0
        )
        assert measured["gap_score"] == row.gap_score
    decisions = json.loads(inputs["decisions"].read_text(encoding="utf-8"))
    published_composite = next(
        item["composite_rmse"]
        for item in decisions
        if item["model"] == "mean_neighbors"
    )
    np.testing.assert_allclose(
        actual["composite_rmse"], published_composite, atol=1e-12, rtol=0
    )
    summaries = []
    for mode, repeat, fold in (
        folds[["mode", "repeat", "fold"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ):
        split = split_fold(train, folds, mode, repeat, fold, spec)
        expected = baseline.loc[
            baseline.split.eq(MODE_MAP[mode]) & baseline.fold.eq(f"r{repeat}_f{fold}")
        ]
        labels = split.targets[KEY + ["primary_ndvi"]].rename(
            columns={"primary_ndvi": "y_true"}
        )
        labels = labels.assign(
            split=MODE_MAP[mode],
            fold=f"r{repeat}_f{fold}",
            primary_ndvi_pred=labels.y_true,
        )
        align_oof(expected, labels)
        assert not key_index(split.gap_keys).isin(key_index(split.fit_frame)).any()
        assert not split.fit_frame.anon_polygon_id.isin(
            split.metadata["fit_excluded_polygons"]
        ).any()
        context = prepare_context(split.context_frame, split.gap_keys, apply_mask=mask)
        # Producer context is already censored. Restore only mask provenance from raw,
        # never raw channel values. This object is inference-only (labels absent).
        original = train.set_index(KEY).loc[key_index(context.frame)]
        context.natural_missing = pd.Series(
            ~np.isfinite(original.primary_ndvi.to_numpy())
        )
        prep = WindowPreprocessor.fit(context, split.fit_frame[KEY])
        probe_keys = split.gap_keys.iloc[:8]
        windows = WindowDatasetAdapter(context, probe_keys, prep)
        for i in range(len(windows)):
            item = windows[i]
            assert np.isfinite(item["features"]).all()
            assert not item["observation_mask"][windows.center_index].any()
            assert not item["loss_mask"].any()
        policy = split.metadata["context_policy"]
        if policy == "past_only":
            cutoff = pd.Timestamp(
                folds.loc[
                    folds["mode"].eq(mode)
                    & folds["repeat"].eq(repeat)
                    & folds["fold"].eq(fold),
                    "cutoff_date",
                ].iloc[0]
            )
            assert (
                not split.context_frame.loc[
                    split.context_frame.date.ge(cutoff), "primary_ndvi"
                ]
                .notna()
                .any()
            )
        if policy == "one_sided_hard":
            for polygon in split.targets.anon_polygon_id.unique():
                visible = split.context_frame.loc[
                    split.context_frame.anon_polygon_id.eq(polygon)
                    & split.context_frame.primary_ndvi.notna()
                ]
                assert len(visible) == 1
                assert (
                    split.targets.loc[
                        split.targets.anon_polygon_id.eq(polygon), "date"
                    ].min()
                    - visible.date.iloc[0]
                ).days > 15
        summary = {
            **split.metadata,
            "window_probes": len(windows),
            "fit_rows": len(split.fit_frame),
            "oof_keys_labels_equal": True,
            "hidden_channels_empty": True,
        }
        summaries.append(summary)
        print(
            json.dumps(
                {
                    "mode": mode,
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "status": "passed",
                }
            ),
            flush=True,
        )
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "kind": "ML_producer_compatibility_review_not_DL_experiment",
        "producer_commit": commit,
        "input_sha256": {key: file_sha256(path) for key, path in inputs.items()},
        "mask_version": spec.version,
        "mask_columns": list(spec.columns),
        "mode_mapping": MODE_MAP,
        "baseline_metrics": actual,
        "baseline_bundle_integrity": bundle_integrity(
            root / "artifacts/ml/baseline_v1/bundle"
        ),
        "oof_rows": len(baseline),
        "folds": summaries,
        "trained_gpu_bundle_available": (
            root / "artifacts/ml/trained_gpu_v1/bundle/manifest.json"
        ).is_file(),
        "remaining": [
            "ML review/acknowledgement in main",
            "DL training-target and inner-stop mask policy",
            "Producer-context consumer runner integration (past_only/one_sided_hard)",
            "Published final ML ensemble OOF plus hashes; report-only metrics insufficient",
        ],
        "dl_decision": "PENDING_EVALUATION",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ml-root", required=True)
    parser.add_argument("--output", default="artifacts/dl/ml_handoff_review.json")
    args = parser.parse_args(argv)
    audit_handoff(args.ml_root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
