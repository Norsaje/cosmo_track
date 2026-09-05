"""Сводит прогон CV в таблицы отчёта; сам ничего не обучает и не выбирает.

Читает выход `veg_recovery.dl.train --fold-manifest ...` и публикует
`reports/dl_experiments.csv` и краткую текстовую сводку метрик.

Вес ансамбля подбирается leave-one-fold-out: для каждого фолда вес взят на
остальных фолдах, поэтому бленд оценивается не на тех же строках.

    PYTHONPATH=src python -m veg_recovery.dl.report --run artifacts/dl/cv_tcn_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import (
    COMPOSITE_WEIGHTS,
    adoption_decision,
    align_oof,
    gap_score,
    rmse,
    validate_oof,
)

SPLIT_LABEL = {
    "matched": "A matched-mask",
    "unseen": "B unseen-polygon",
    "temporal": "C past-only",
    "hard": "D hard one-sided",
}
EXPERIMENT_COLUMNS = [
    "experiment_id", "model", "commit", "data_fingerprint", "fold_version",
    "mask_version", "seed", "window", "channels", "params_json", "epoch", "rmse",
    "gap_score", "unseen_rmse", "hard_rmse", "source_s2_rmse", "source_landsat_rmse",
    "source_modis_rmse", "coverage_80", "width_80", "train_sec", "infer_sec",
    "peak_vram_mb", "artifact_mb", "decision", "notes",
]


def _composite(frame: pd.DataFrame, column: str) -> float | None:
    per_split = {str(k): rmse(g.y_true, g[column]) for k, g in frame.groupby("split")}
    if not set(COMPOSITE_WEIGHTS).issubset(per_split):
        return None
    return float(sum(COMPOSITE_WEIGHTS[k] * per_split[k] for k in COMPOSITE_WEIGHTS))


def _subgroup(result: dict, field: str, key: str) -> float | None:
    entry = result["subgroups"].get(field, {}).get(key)
    return None if entry is None else entry["rmse"]


def blend_out_of_sample(aligned: pd.DataFrame, grid=np.linspace(0, 1, 101)) -> dict:
    """Вес выбирается на остальных фолдах и применяется к отложенному фолду."""
    folds = sorted({(a, b) for a, b in zip(aligned["split"], aligned["fold"])})
    if len(folds) < 2:
        raise ValueError("Out-of-sample blending needs at least two folds")
    parts, weights = [], {}
    for split, fold in folds:
        held = aligned["split"].eq(split) & aligned.fold.eq(fold)
        other = aligned.loc[~held]
        errors = [
            rmse(
                other.y_true,
                (1 - w) * other.primary_ndvi_pred + w * other.candidate_pred,
            )
            for w in grid
        ]
        weight = float(grid[int(np.argmin(errors))])
        weights[f"{split}/{fold}"] = weight
        block = aligned.loc[held].copy()
        block["blend_pred"] = (
            1 - weight
        ) * block.primary_ndvi_pred + weight * block.candidate_pred
        parts.append(block)
    blended = pd.concat(parts, ignore_index=True)
    unseen = blended.loc[blended.is_unseen.astype(bool)] if "is_unseen" in blended else None
    return {
        "weights": weights,
        "mean_weight": float(np.mean(list(weights.values()))),
        "overall_rmse": rmse(blended.y_true, blended.blend_pred),
        "composite_rmse": _composite(blended, "blend_pred"),
        "unseen_rmse": None
        if unseen is None or unseen.empty
        else rmse(unseen.y_true, unseen.blend_pred),
        "ml_unseen_rmse": None
        if unseen is None or unseen.empty
        else rmse(unseen.y_true, unseen.primary_ndvi_pred),
    }


def learning_curves(run: Path) -> dict:
    """Собирает кривые обучения из per-fold checkpoints в один небольшой файл.

    Сами checkpoints остаются локальными: веса не публикуются, а числа, на
    которых стоит отчёт, сохраняются вместе с их SHA256.
    """
    out = {}
    for path in sorted(run.glob("seed_*_fold_*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        metadata = manifest["metadata"]
        out[path.parent.name] = {
            "split": metadata["split"],
            "fold": metadata["fold"],
            "seed": metadata["seed"],
            "epoch_policy": metadata["result"]["epoch_policy"],
            "best_epoch": metadata["result"]["best_epoch"],
            "parameters": metadata["result"]["parameters"],
            "weights_sha256": manifest["files"]["weights.pt"],
            "history": metadata["history"],
        }
    return out


def consolidate_partial(run: Path) -> dict:
    """Сводит прерванный прогон: только фолды, где посчитаны все запрошенные seed.

    Полный `cv_report.json` пишется в самом конце, поэтому остановленный прогон
    иначе не оставил бы ничего, кроме локальных checkpoints. Режим засчитывается
    завершённым только если в нём завершены все его фолды.
    """
    frames = []
    for path in sorted(run.glob("seed_*_fold_*/oof.csv")):
        seed = int(path.parent.name.split("seed_")[1].split("_fold")[0])
        frames.append(pd.read_csv(path, parse_dates=["date"]).assign(seed_run=seed))
    if not frames:
        raise ValueError("Прогон не оставил ни одного завершённого фолда")
    oof = pd.concat(frames, ignore_index=True)
    seeds = sorted(int(v) for v in oof.seed_run.unique())
    complete = [
        fold
        for fold, group in oof.groupby(["split", "fold"])
        if sorted(int(v) for v in group.seed_run.unique()) == seeds
    ]
    kept = oof.loc[
        pd.MultiIndex.from_frame(oof[["split", "fold"]]).isin(complete)
    ].copy()
    manifest = json.loads((run / "input_manifest.json").read_text(encoding="utf-8"))
    planned = {(f["split"], f["fold"]) for f in manifest["folds"]}
    per_split = {}
    for split, group in kept.groupby("split"):
        folds_done = {(split, f) for f in group.fold.unique()}
        folds_planned = {f for f in planned if f[0] == split}
        values = [
            rmse(part.y_true, part.primary_ndvi_pred)
            for _, part in group.groupby("seed_run")
        ]
        base = [
            rmse(part.y_true, part.base_pred)
            for _, part in group.groupby("seed_run")
        ]
        per_split[split] = {
            "folds_complete": len(folds_done),
            "folds_planned": len(folds_planned),
            "split_complete": folds_done == folds_planned,
            "rows_per_seed": int(len(group) / len(seeds)),
            "dl_rmse_mean": float(np.mean(values)),
            "dl_rmse_std": float(np.std(values)),
            "dl_gap_score": gap_score(float(np.mean(values))),
            "base_rmse_mean": float(np.mean(base)),
            "ml_rmse": rmse(group.y_true, group.pred_ml)
            if "pred_ml" in group
            else None,
        }
    status = {
        "kind": "interrupted_run_partial_summary",
        "complete_cv": False,
        "seeds": seeds,
        "folds_complete": len(complete),
        "folds_planned": len(planned),
        "composite_rmse": None,
        "composite_reason": "composite требует всех четырёх режимов A/B/C/D",
        "splits": per_split,
    }
    for seed in seeds:
        part = kept.loc[kept.seed_run.eq(seed)].drop(columns="seed_run")
        part.to_csv(run / f"oof_partial_seed_{seed}.csv", index=False,
                    lineterminator=chr(10))
    (run / "partial_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + chr(10),
        encoding="utf-8",
    )
    return status


def summarise(run: Path) -> dict:
    report = json.loads((run / "cv_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "input_manifest.json").read_text(encoding="utf-8"))
    experiments = pd.read_csv(run / "experiments.csv")
    environment = report["environment"]
    rows, seeds = [], []
    for entry in report["seeds"]:
        seed = entry["seed"]
        detail = experiments.loc[experiments.seed.eq(seed)]
        shared = {
            "commit": environment.get("git_commit"),
            "data_fingerprint": manifest["data"]["sha256"],
            "fold_version": manifest["fold_version"],
            "mask_version": manifest["mask_version"],
            "seed": seed,
            "window": int(detail.window.iloc[0]),
            "channels": 11,
            "coverage_80": "",
            "width_80": "",
        }
        variants = (
            (
                "residual_tcn",
                entry["dl"],
                {
                    "epoch_policy": str(detail.epoch_policy.iloc[0]),
                    "epochs": int(detail.epoch.max()),
                    "base_mode": report.get("base_mode"),
                    "parameters": int(detail.parameters.iloc[0]),
                },
            ),
            ("anchored_base_only", entry["dl_base_only"], {"model": "residual base, no network"}),
            ("ml_mean_neighbors", entry["ml"], {"model": "published ML baseline OOF"}),
        )
        for name, result, params in variants:
            trained = name == "residual_tcn"
            rows.append(
                {
                    **shared,
                    "experiment_id": f"{name}_s{seed}",
                    "model": name,
                    "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
                    "epoch": int(detail.epoch.max()) if trained else "",
                    "rmse": result["overall_rmse"],
                    "gap_score": result["gap_score"],
                    "unseen_rmse": _subgroup(result, "is_unseen", "True"),
                    "hard_rmse": _subgroup(result, "is_hard", "True"),
                    "source_s2_rmse": _subgroup(result, "source", "s2"),
                    "source_landsat_rmse": _subgroup(result, "source", "landsat"),
                    "source_modis_rmse": _subgroup(result, "source", "modis"),
                    "train_sec": float(detail.train_sec.sum()) if trained else "",
                    "infer_sec": float(detail.infer_sec.sum()) if trained else "",
                    "peak_vram_mb": float(detail.peak_vram_mb.max()) if trained else "",
                    "artifact_mb": round(
                        sum(
                            p.stat().st_size
                            for p in run.glob(f"seed_{seed}_fold_*/weights.pt")
                        )
                        / 2**20,
                        4,
                    )
                    if trained
                    else "",
                    "decision": "PENDING_EVALUATION",
                    "notes": f"composite={result['composite_rmse']}",
                }
            )
        seeds.append(
            {
                "seed": seed,
                "ml": entry["ml"],
                "dl": entry["dl"],
                "base": entry["dl_base_only"],
                "bootstrap": entry["polygon_bootstrap"],
                "complete_cv": entry.get("complete_cv"),
            }
        )
    return {
        "report": report,
        "manifest": manifest,
        "seeds": seeds,
        "table": pd.DataFrame(rows, columns=EXPERIMENT_COLUMNS),
        "environment": environment,
    }


def gate_inputs(summary: dict, blend: dict | None = None) -> list[dict]:
    out = []
    for entry in summary["seeds"]:
        ml, dl = entry["ml"], entry["dl"]
        row = {
            "seed": entry["seed"],
            "ml_composite": ml["composite_rmse"],
            "dl_composite": dl["composite_rmse"],
            "ml_unseen": _subgroup(ml, "is_unseen", "True"),
            "dl_unseen": _subgroup(dl, "is_unseen", "True"),
            # Ни один режим не должен просесть больше чем на 0.003 против ML.
            "subgroups_passed": all(
                dl["split_rmse"][k] <= ml["split_rmse"][k] + 0.003
                for k in dl["split_rmse"]
            ),
        }
        value = (blend or {}).get(entry["seed"])
        if value and value["composite_rmse"] is not None:
            row.update(
                {
                    "blend_evaluated_out_of_sample": True,
                    "blend_gain": ml["composite_rmse"] - value["composite_rmse"],
                    "blend_unseen_degradation": (value["unseen_rmse"] or 0.0)
                    - (value["ml_unseen_rmse"] or 0.0),
                }
            )
        out.append(row)
    return out


def markdown(summary: dict, blend: dict | None = None) -> str:
    seeds = summary["seeds"]
    lines = [
        "| Режим | Вес | ML baseline | Только anchored base | DL TCN |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, weight in COMPOSITE_WEIGHTS.items():
        values = [
            float(np.mean([s[k]["split_rmse"][split] for s in seeds]))
            for k in ("ml", "base", "dl")
        ]
        lines.append(
            f"| {SPLIT_LABEL[split]} | {weight:.2f} | "
            + " | ".join(f"{v:.6f}" for v in values)
            + " |"
        )
    for label, key in (
        ("Composite", "composite_rmse"),
        ("Row-weighted RMSE", "overall_rmse"),
    ):
        values = [
            float(np.mean([s[k][key] for s in seeds])) for k in ("ml", "base", "dl")
        ]
        lines.append(
            f"| **{label}** | — | " + " | ".join(f"{v:.6f}" for v in values) + " |"
        )
    overall = [s["dl"]["overall_rmse"] for s in seeds]
    ml_overall = float(np.mean([s["ml"]["overall_rmse"] for s in seeds]))
    lines += [
        "",
        f"Seeds: {[s['seed'] for s in seeds]}. Общий RMSE DL "
        f"{np.mean(overall):.6f} ± {np.std(overall):.6f}; GapScore "
        f"{gap_score(float(np.mean(overall))):.2f} против {gap_score(ml_overall):.2f} "
        "у опубликованного baseline на тех же строках.",
        "",
        "| Seed | Разница общего RMSE (ML − DL) | 95% CI, bootstrap по полигонам |",
        "|---|---:|---|",
    ]
    for entry in seeds:
        boot = entry["bootstrap"]
        lines.append(
            f"| {entry['seed']} | {boot['gain_rmse']:.6f} | "
            f"[{boot['ci95_low']:.6f}; {boot['ci95_high']:.6f}] |"
        )
    if blend:
        lines += [
            "",
            "| Seed | Средний вес DL | Composite бленда | Общий RMSE бленда |",
            "|---|---:|---:|---:|",
        ]
        for seed, value in blend.items():
            composite = value["composite_rmse"]
            lines.append(
                f"| {seed} | {value['mean_weight']:.3f} | "
                f"{composite:.6f} | {value['overall_rmse']:.6f} |"
            )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--baseline-oof", default="artifacts/dl/c03_derived/baseline_oof.csv.gz")
    parser.add_argument("--experiments", default="reports/dl_experiments.csv")
    parser.add_argument("--markdown", default=None)
    parser.add_argument(
        "--partial-only",
        action="store_true",
        help="Свести прерванный прогон: только фолды со всеми seed",
    )
    args = parser.parse_args(argv)
    run = Path(args.run)
    if args.partial_only or not (run / "cv_report.json").is_file():
        status = consolidate_partial(run)
        curves = learning_curves(run)
        if curves:
            (run / "learning_curves.json").write_text(
                json.dumps(curves, ensure_ascii=False, indent=2, sort_keys=True)
                + chr(10),
                encoding="utf-8",
            )
        print(json.dumps(status, ensure_ascii=False, indent=2, default=float))
        return 0
    summary = summarise(run)
    blend = {}
    baseline_path = Path(args.baseline_oof)
    if baseline_path.is_file():
        reference = validate_oof(pd.read_csv(baseline_path, parse_dates=["date"]))
        for entry in summary["seeds"]:
            oof = validate_oof(
                pd.read_csv(run / f"oof_seed_{entry['seed']}.csv", parse_dates=["date"])
            )
            blend[entry["seed"]] = blend_out_of_sample(align_oof(reference, oof))
    inputs = gate_inputs(summary, blend)
    decision = adoption_decision(inputs, integration_passed=False)
    Path(args.experiments).parent.mkdir(parents=True, exist_ok=True)
    summary["table"].to_csv(args.experiments, index=False, lineterminator="\n")
    curves = learning_curves(run)
    if curves:
        (run / "learning_curves.json").write_text(
            json.dumps(curves, ensure_ascii=False, indent=2, sort_keys=True) + chr(10),
            encoding="utf-8",
        )
    text = markdown(summary, blend)
    if args.markdown:
        Path(args.markdown).write_text(text + "\n", encoding="utf-8")
    print(text)
    print()
    print(
        json.dumps(
            {"decision": decision, "gate_inputs": inputs},
            ensure_ascii=False,
            indent=2,
            default=float,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
