# DL consumer C-03: предложение интерфейса, требует review ML

Это **не** новая схема фолдов или собственный MaskSpec. DL runner принимает
обёртку над артефактами ML. Если фактический handoff отличается, меняется DL
consumer, а не ML-owned folds. Сейчас реальные C-03 файлы ещё не опубликованы.

JSON `dl_c03.json`:

```json
{
  "schema_version": "dl-c03-consumer-0.1",
  "producer": "ML",
  "review_status": "draft",
  "fold_version": "REPLACE_WITH_ML_VERSION",
  "mask_version": "REPLACE_WITH_ML_VERSION",
  "mask_callable": "veg_recovery.validation.masking:apply_mask",
  "data": {"path": "frame.csv", "sha256": "REPLACE_WITH_SHA256"},
  "baseline_oof": {"path": "baseline_oof.csv", "sha256": "REPLACE_WITH_SHA256"},
  "baseline_metrics": {"overall_rmse": 0.0, "gap_score": 30.0},
  "folds": [{
    "split": "matched",
    "fold": "f0",
    "context_policy": "interpolation",
    "fit_context_keys": {"path": "f0_fit_context.csv", "sha256": "REPLACE_WITH_SHA256"},
    "inference_context_keys": {"path": "f0_infer_context.csv", "sha256": "REPLACE_WITH_SHA256"},
    "train_target_keys": {"path": "f0_train_targets.csv", "sha256": "REPLACE_WITH_SHA256"},
    "inner_target_keys": {"path": "f0_inner_targets.csv", "sha256": "REPLACE_WITH_SHA256"},
    "evaluation_keys": {"path": "f0_eval.csv", "sha256": "REPLACE_WITH_SHA256"}
  }]
}
```

Числа выше — placeholders, **не результат эксперимента**. `draft` отклоняется.
`accepted` выставляет ML после проверки exported keys/metrics/context policy.
Все пути разрешаются относительно JSON. В Kaggle Dataset сохраняется та же структура.

`fit_context_keys` содержит разрешённые обучающие строки, включая train targets,
но исключает inner/outer target rows. В unseen CV исключает все evaluation polygons.
`inference_context_keys` — разрешённый контекст plus все inner/evaluation target keys.
Для forecasting/causal контекста будущие наблюдения запрещены. Для interpolation
двусторонний контекст разрешён явно. `inner_target_keys` — предоставленная ML
внутренняя выборка early stopping; outer OOF не используется для выбора epoch.

Все key CSV имеют `anon_polygon_id,date`. OOF имеет уникальный ключ
`split,fold,anon_polygon_id,date` и поля `y_true,primary_ndvi_pred`;
optional `is_unseen,is_hard,source,gap_length,crop_type` переносятся в DL OOF.
Split имена consumer: `matched`, `unseen`, `temporal`, `hard`; mapping CV-A/B/C/D
должен быть явным на границе consumer. Повторы хранятся как разные `fold`.

Для composite требуются все четыре режима с весами 0.50/0.25/0.15/0.10.
Публикуется также общий row-weighted RMSE/GapScore. Их нельзя путать.
Bootstrap по polygon оценивает разницу общего RMSE; это не CI composite.

`mask_callable` — ML-owned адаптер `apply_mask(frame, selected_keys) -> frame`,
внутри которого применяется официальный MaskSpec. Поля всех скрытых строк
проверяются, ключи/порядок не меняются. Stock PyPOTS MCAR не заменяет этот шаг.

Запуск проверки не создаёт весов и не импортирует torch:

```bash
PYTHONPATH=src python -m veg_recovery.dl.train --fold-manifest inputs/dl_c03.json --preflight-only
```
