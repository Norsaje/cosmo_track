"""Объяснения описывают совпадение сигналов, не устанавливают причинность."""

from .events import REASON_CODES

_TEXT = {
    "LOW_PRECIPITATION": "осадками ниже сезонной нормы",
    "HIGH_TEMPERATURE": "температурой выше сезонной нормы",
    "LOW_NDWI": "пониженным NDWI относительно его исторического уровня",
    "MULTISENSOR_CONFIRMATION": "подтверждением снижения независимыми датчиками",
    "SOURCE_SWITCH_RISK": "сменой источника; возможен остаточный сдвиг между датчиками",
    "LOW_DATA_COVERAGE": "недостатком надёжных наблюдений или широкой неопределённостью",
    "RAPID_NEGATIVE_CHANGE": "быстрым отрицательным изменением ряда",
    "PROLONGED_SUPPRESSION": "продолжительным отклонением от сезонной нормы",
    "PHENOLOGY_SHIFT": "признаками сдвига фазы вегетации",
}


def explain_ru(reason_codes: tuple[str, ...]) -> str:
    if not set(reason_codes).issubset(REASON_CODES):
        raise ValueError("Unknown anomaly reason code")
    if not reason_codes:
        return (
            "NDVI устойчиво ниже исторической сезонной нормы. "
            "Данных для интерпретации причины недостаточно; причинность не доказана."
        )
    text = (
        "Снижение NDVI совпадает с " + "; ".join(_TEXT[c] for c in reason_codes) + ". "
    )
    if {"LOW_PRECIPITATION", "HIGH_TEMPERATURE"}.issubset(reason_codes):
        text += "Сочетание сигналов согласуется с водным стрессом. "
    return (
        text
        + "Причинность не доказана; confidence отражает поддержку данными, а не вероятность причины."
    )
