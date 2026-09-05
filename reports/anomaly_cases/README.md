# Anomaly cases

`synthetic_v1/` содержит семь **искусственных** сценариев, общий reference.csv,
config.json, per-case scored CSV, JSON contract, PNG и summary.json.

Воспроизведение: `PYTHONPATH=src python -m veg_recovery.dl.anomaly_cases`.

| Сценарий | Ожидание | Назначение |
|---|---|---|
| mild_pulse | suppression | слабое устойчивое отрицательное отклонение |
| medium_pulse | critical | сильное устойчивое отклонение |
| strong_pulse | critical | проверка монотонности magnitude/score |
| source_switch | нет event | raw меняется, harmonized остаётся согласованным |
| single_outlier | нет event | одиночный провал не означает устойчивую аномалию |
| wide_uncertainty | нет event | реконструкция без достаточного подтверждения |
| normal | нет event | сезонная динамика относительно нормы |

IoU=1 на внедрённых прямоугольных pulses — инженерная проверка границ,
не утверждение о реальной точности. Earliest support delay (2 дня) — минимальный
интервал до второго поддерживающего наблюдения в fixture, не измеренный online
detection delay. Ложные тревоги приведены как счётчик на четырёх отрицательных
контролях, не как оценка false alerts per real season.

`real_2024/` содержит диагностический проход по 39 реальным train polygons,
3 сильных/3 сомнительных случая с CSV/JSON/PNG и визуальным review агента.
Калибровка/reference — 2010–2023, query — 2024; это не конкурсная CV и не ground truth.
Детали: `real_2024/review.md`. Экспертное принятие Backend/ML и интеграция с
официальным ML-014 остаются pending. Машинные тесты не заменяют это принятие.
