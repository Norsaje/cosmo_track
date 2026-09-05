# DL consumer C-03 — интерфейс `dl-c03-consumer-0.2`

Это **не** новая схема фолдов и не собственный MaskSpec. DL runner принимает
обёртку над артефактами ML. Если фактический handoff отличается, меняется DL
consumer, а не ML-owned folds.

## Как manifest получается сегодня

`veg_recovery.dl.c03_bridge` исполняет **producer-код ML** из отдельного worktree
(`read_dataset`, `MaskSpec.from_test`, `load_folds`, `split_fold`) и записывает
именно те ключи, которые вернул producer. Реализация ML не копируется и не
изменяется; мост падает, если импортированный модуль лежит вне указанного
`--ml-root`.

```powershell
git worktree add --detach tmp/dl-review-ml origin/ML
$env:PYTHONPATH='src;tmp/dl-review-ml/src'
python -m veg_recovery.dl.c03_bridge --ml-root tmp/dl-review-ml `
  --output artifacts/dl/c03_derived
python -m veg_recovery.dl.train --fold-manifest artifacts/dl/c03_derived/dl_c03.json `
  --preflight-only
```

`review_status` производного manifest — `derived_from_producer_artifacts`.
Runner принимает его наравне с `accepted`, но требует `producer_evidence`
(commit ML и SHA256 всех входов) и записывает статус в каждый checkpoint и
отчёт. **Это не подтверждение ML.** Задача не может стать DONE, пока владелец
C-03 не поставил acknowledgement.

## Что берётся у ML и что добавляет DL

| Часть | Владелец | Источник |
|---|---|---|
| outer evaluation keys | ML | `split_fold(...).gap_keys` |
| policy censoring (C/D) | ML | разность `train` и `split_fold(...).context_frame` |
| разрешённый fit context | ML | `split_fold(...).fit_frame` |
| MaskSpec и `apply_mask` | ML | `veg_recovery.validation.masking` |
| baseline OOF и метрики | ML | `artifacts/ml/baseline_v1/oof_predictions.csv.gz` |
| train-блоки, inner-монитор | **DL** | серии внутри `fit_frame`, seed от ML fold |

Разностный censoring проверяется round-trip: `apply_mask(train, censored_keys)`
обязан побайтово воспроизвести `split_fold(...).context_frame`, иначе мост падает.
Без этого шага inference на CV-C/CV-D увидел бы сырые значения, скрытые policy.

## Поля manifest

```json
{
  "schema_version": "dl-c03-consumer-0.2",
  "producer": "ML",
  "review_status": "derived_from_producer_artifacts",
  "producer_evidence": {"commit": "...", "input_sha256": {"train": "...", "folds": "..."},
                        "ml_acknowledgement": "pending"},
  "fold_version": "folds_v1",
  "mask_version": "real_test_v1",
  "mask_callable": "veg_recovery.validation.masking:apply_mask",
  "data": {"path": "frame.csv.gz", "sha256": "..."},
  "baseline_oof": {"path": "baseline_oof.csv.gz", "sha256": "..."},
  "baseline_metrics": {"overall_rmse": 0.0, "gap_score": 0.0},
  "folds": [{
    "split": "temporal", "fold": "r0_f0",
    "producer_mode": "C", "producer_context_policy": "past_only",
    "context_policy": "causal",
    "fit_context_keys": {"path": "...", "sha256": "..."},
    "inference_context_keys": {"path": "...", "sha256": "..."},
    "censored_context_keys": {"path": "...", "sha256": "..."},
    "train_target_keys": {"path": "...", "sha256": "..."},
    "inner_target_keys": {"path": "...", "sha256": "..."},
    "evaluation_keys": {"path": "...", "sha256": "..."}
  }]
}
```

`fit_context_keys` — разрешённые обучающие строки, включая train targets, но без
inner/outer target rows; в unseen CV без всех evaluation polygons.
`inference_context_keys` — все ключи кадра; значения скрываются
`censored_context_keys`. `censored_context_keys` обязан содержать все
evaluation keys и не пересекаться с fit/inner. `train_target_keys` несёт
колонку `block`: каждый блок маскируется отдельно, поэтому плотность
искусственных пропусков в обучении близка к тестовой, а не в разы выше.

Все key CSV имеют `anon_polygon_id,date` и хранятся gzip с `mtime=0`, поэтому
SHA256 воспроизводим. OOF имеет уникальный ключ `split,fold,anon_polygon_id,date`
и поля `y_true,primary_ndvi_pred`; optional `is_unseen,is_hard,source,gap_length,
crop_type` переносятся в DL OOF. Split-имена consumer: `matched`, `unseen`,
`temporal`, `hard`; mapping CV-A/B/C/D задан явно. Повторы — разные `fold`.

Для composite нужны все четыре режима с весами 0.50/0.25/0.15/0.10. Отдельно
публикуется row-weighted RMSE/GapScore; путать их нельзя. Bootstrap по polygon
оценивает разницу общего RMSE, это не CI composite.

## Что ещё нужно от ML

1. **Acknowledgement C-03** в `main:00_team_coordination.md`.
2. **Per-row OOF сильной модели.** В опубликованном `oof_predictions.csv.gz`
   есть только `pred_nearest_left/right, pred_mean_neighbors, pred_linear,
   pred_seasonal_crop, pred_pchip, pred_akima`. Колонок CatBoost/ансамбля нет,
   а `/artifacts/ml/trained_gpu_v1/` исключён их же `.gitignore`. Достаточно
   добавить `pred_catboost` и `pred_ensemble` в тот же файл (или отдельный OOF
   CSV с тем же ключом) — bundle публиковать не требуется. Без этого adoption
   gate на composite не имеет вычислимого знаменателя.
3. **Inner keys или согласованный epoch budget.** Пока ML не передал inner keys,
   DL использует `--epoch-policy fixed`: epoch не выбирается ни по outer OOF, ни
   по DL-inner. Inner-монитор только логируется.
