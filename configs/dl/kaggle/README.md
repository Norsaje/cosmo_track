# Запуск DL на Kaggle

GPU понадобится для полного C-03 прогона и следующих SAITS/BRITS экспериментов.
Локальные smoke tests и anomaly pipeline работают на CPU. GPU сейчас не блокер:
сначала ML должен опубликовать frozen folds, MaskSpec и baseline/HGB OOF.

1. Загрузить `run_tcn.ipynb` в Kaggle Notebook. Добавить Dataset с кодом ветки DL
   (включая `src/`), исходными CSV и принятым C-03 consumer manifest вместе с его
   файлами. Версию кода и данные зафиксировать; не брать движущийся `main` в runtime.
2. В настройках Notebook выбрать доступный GPU. Исправить только `REPO` и
   `FOLD_MANIFEST` в первой ячейке. Manifest содержит относительные пути и SHA256.
3. Выполнить preflight, затем smoke. Обучение запустится только если есть CUDA,
   совпадают SHA256/ключи/метрики ML, нет пересечения fit/inner/outer labels и
   доступен ML-owned mask adapter. Устанавливать torch поверх Kaggle image не нужно.
4. Запустить ячейку полного CV. Сохранить версию Notebook и скачать
   `/kaggle/working/dl_tcn_run.zip`: OOF по каждому seed, CV/подгруппы/bootstrap,
   checkpoints, preprocessing, manifest, конфигурация и версии Python/CUDA/cuDNN.

Notebook не обращается к внешней сети, не получает скрытые test labels и не
генерирует собственные folds. Готовые веса не становятся production автоматически.
Если пакеты Kaggle отличаются от локальных, сначала должны пройти smoke tests;
фактические версии сохраняются в каждом manifest. Clean-install/uv.lock — SH-002,
владелец Backend. Этот notebook не подменяет командный lock.

Команда полного прогона после handoff:

```bash
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m veg_recovery.dl.train \
  --fold-manifest /kaggle/input/cosmo-dl/inputs/dl_c03.json \
  --device cuda --seeds 17 42 73 --window 61 --epochs 40 --patience 6 \
  --output /kaggle/working/dl_tcn_run
```

Бюджет P0: одна TCN-конфигурация, 3 seed, без sweep. Время/VRAM измеряются по факту;
оценку времени полного прогона нельзя дать до числа фолдов и train targets из C-03.
