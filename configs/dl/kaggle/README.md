# Запуск DL-обучения на Kaggle

Полный прогон — 16 фолдов ML × 3 seed. На CPU это часы, на одной Kaggle GPU
существенно быстрее. Notebook ничего не скачивает, не создаёт свои folds и не
видит скрытые test labels: он работает с уже опубликованным `train_dataset.csv`
и производным C-03 manifest.

## Шаг 1. Собрать входы локально

**После слияния DL и ML** код лежит в одном дереве:

```powershell
$env:PYTHONPATH='src'
python -m veg_recovery.dl.c03_bridge --ml-root . --output artifacts/dl/c03_derived
python -m veg_recovery.dl.train --fold-manifest artifacts/dl/c03_derived/dl_c03.json --preflight-only
python configs/dl/kaggle/package_dataset.py --ml-root .
```

**До слияния** источником C-01/C-03 служит отдельный read-only worktree ML:

```powershell
git worktree add --detach tmp/dl-review-ml origin/ML
$env:PYTHONPATH='src;tmp/dl-review-ml/src'
python -m veg_recovery.dl.c03_bridge --ml-root tmp/dl-review-ml --output artifacts/dl/c03_derived
python -m veg_recovery.dl.train --fold-manifest artifacts/dl/c03_derived/dl_c03.json --preflight-only
python configs/dl/kaggle/package_dataset.py --ml-root tmp/dl-review-ml
```

Готово: `artifacts/dl/kaggle_bundle.zip`. Внутри `tests/`, `inputs/` и
`BUNDLE_MANIFEST.json` с SHA256 каждого файла, полем `layout` и точной командой
прогона. Код лежит в `src/` для объединённого дерева либо в `src_dl/` и `src_ml/`
до слияния; notebook читает `layout` и сам собирает PYTHONPATH.

## Шаг 2. Создать Kaggle Dataset

1. kaggle.com → **Datasets** → **New Dataset** → **Upload**.
2. Загрузить `artifacts/dl/kaggle_bundle.zip`; Kaggle распакует архив.
3. Название задать так, чтобы slug получился `cosmo-dl-bundle`. Видимость —
   Private. Дождаться окончания обработки.

Если slug вышел другим, поправьте `BUNDLE` в первой ячейке notebook.

## Шаг 3. Notebook

1. **Code** → **New Notebook** → **File → Import Notebook** →
   `configs/dl/kaggle/run_tcn.ipynb`.
2. **Add Input** → добавить созданный Dataset.
3. Справа: **Accelerator = GPU** (T4/P100 достаточно), **Internet = Off**.
   Интернет не нужен: torch уже в образе Kaggle, PyPOTS в P0 не используется.
4. **Run All**. Порядок ячеек: контроль bundle → preflight → tests и GPU smoke →
   полная CV → сводка и архив → ablation с линейной base.

Ячейка полной CV выполняет:

```bash
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m veg_recovery.dl.train   --fold-manifest inputs/dl_c03.json --device cuda   --seeds 17 42 73 --window 61 --epochs 16 --epoch-policy fixed   --base-mode anchored --batch-size 64 --output /kaggle/working/dl_tcn_run
```

Бюджет 16 эпох закреплён заранее по inner-кривой пилотного фолда: плато на
12–17 эпохе в matched, unseen и temporal. По outer OOF epoch не выбирается.
До слияния веток PYTHONPATH будет `src_dl:src_ml`; notebook подставляет его сам.

## Шаг 4. Забрать результаты

Скачать `/kaggle/working/dl_tcn_run.zip` и положить рядом с отчётом. Внутри:

- `oof_seed_*.csv` — OOF на тех же ключах, что у ML, плюс `base_pred`;
- `experiments.csv` — строка на каждый (fold, seed) с RMSE, временем и памятью;
- `cv_report.json` — метрики по режимам, composite, bootstrap по полигонам;
- `seed_*_fold_*/` — checkpoints с SHA256, preprocessor и кривой обучения;
- `input_manifest.json`, `environment.json` — provenance, версии, CUDA/cuDNN.

## Что notebook сделать не может

- Обучение упадёт, если SHA256 входов не совпали, метрики ML baseline не
  воспроизвелись, пересеклись train/inner/outer ключи, отсутствует producer
  censoring или недоступен ML-owned `apply_mask`. Это намеренно.
- `review_status` производного manifest — `derived_from_producer_artifacts`.
  Подтверждение ML всё ещё требуется; результаты нельзя объявлять принятыми.
- Веса из прогона не становятся production-кандидатом: экспорт C-14 разрешён
  только после adoption gate.
- Общий `uv.lock` принадлежит Backend (SH-002); образ Kaggle его не заменяет.
  Фактические версии сохраняются в `environment.json` каждого прогона.
