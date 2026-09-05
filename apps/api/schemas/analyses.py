"""Схемы анализа, ряда, аномалий и provenance — часть контракта C-07 v0.1."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .jobs import JobState

#: Максимальный запрашиваемый диапазон. Ограничение из §8.12 ТЗ («date-range limits»):
#: без него запрос на сорок лет превращается в тысячи обращений к провайдеру.
MAX_ANALYSIS_DAYS = 366 * 5


class AnalysisCreate(BaseModel):
    """Тело `POST /api/v1/analyses`. Отвечаем 202 + job_id, синхронно не считаем:
    растровый fetch внутри HTTP-запроса запрещён (инвариант 1)."""

    #: extra="forbid": опечатка вроде `dateFrom` обязана давать 422, а не молча
    #: превращаться в анализ за диапазон по умолчанию.
    model_config = ConfigDict(extra="forbid")

    polygon_id: str
    date_from: date
    date_to: date
    #: Предпочтительный провайдер (§8.4 ТЗ). Пусто — выбирает orchestrator по приоритету.
    provider_preference: str | None = None

    @model_validator(mode="after")
    def _check_range(self) -> AnalysisCreate:
        if self.date_to < self.date_from:
            raise ValueError("date_to не может быть раньше date_from")
        if (self.date_to - self.date_from).days > MAX_ANALYSIS_DAYS:
            raise ValueError(f"диапазон дат больше {MAX_ANALYSIS_DAYS} дней")
        return self


class AnalysisAccepted(BaseModel):
    """Ответ 202. Повторный запрос с тем же `geometry_hash + date_range + pipeline_version`
    обязан вернуть тот же `analysis_id` и не создавать дубликат джобы (инвариант 7)."""

    job_id: str
    analysis_id: str
    state: JobState = JobState.QUEUED


class SeriesPoint(BaseModel):
    """Одна суточная точка ряда для графика (`GET /analyses/{id}/series`)."""

    date: date
    #: Сырой конкурсный ряд. Иерархия источников S2 → Landsat → MODIS.
    primary_ndvi: float | None = None
    #: Восстановленное значение. Заполнено только там, где `is_reconstructed = true`.
    primary_ndvi_reconstructed: float | None = None
    #: Гармонизированный продуктовый ряд. Отдельное поле, не target (решение D-003).
    ndvi_harmonized: float | None = None
    is_observed: bool
    is_reconstructed: bool
    lower: float | None = None
    upper: float | None = None
    #: Номинальное покрытие интервала [lower, upper], например 0.8. Без него UI
    #: не имеет права подписывать ленту процентом — §8.9 запрещает «точность 95 %».
    interval_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    #: Какой сенсор дал значение. Показывается в tooltip графика.
    selected_source: str | None = None
    method: str | None = None
    #: Доля валидных пикселей в полигоне на эту дату. Низкое значение — повод
    #: пометить точку как low_support, а не молча усреднить облако как ноль (инвариант 3).
    valid_pixel_fraction: float | None = None
    #: Абсолютное число пикселей. Инвариант 3 требует хранить его вместе с долей:
    #: поле в 3 пикселя и поле в 3000 при одинаковой доле — разные по достоверности.
    pixel_count: int | None = None
    #: Сырые QA-флаги источника (§8.3, `qa_flags jsonb`). Нужны tooltip'у по §8.9.
    qa_flags: dict[str, Any] | None = None
    #: Сопутствующие индексы. Выводятся отдельным слоем, не перегружая основной график.
    evi: float | None = None
    ndwi: float | None = None
    #: Погодный контекст: без него утверждение аномалии про засуху нечем подтвердить.
    temp_c: float | None = None
    precip_mm: float | None = None
    #: Климатическая норма — нейтральная линия графика (§8.9). Без неё `min_robust_z`
    #: из события аномалии не с чем сопоставить визуально.
    ndvi_climatology_mean: float | None = None
    ndvi_climatology_std: float | None = None


class SeriesResponse(BaseModel):
    analysis_id: str
    points: list[SeriesPoint]
    #: Честная маркировка кэша прямо там, где фронт берёт данные графика.
    #: Инвариант 10 и решение D-005: выдавать кэш за live запрещено, а требовать
    #: ради этого отдельный запрос к /provenance — значит гарантировать, что забудут.
    cached: bool = False
    data_retrieved_at: datetime | None = None


class AnomalyOut(BaseModel):
    """Событие аномалии в терминах контракта C-09."""

    #: Стабильный идентификатор для перехода «список → карточка события».
    id: str | None = None
    start_date: date
    end_date: date
    severity: str
    score: float
    confidence: float
    min_robust_z: float
    negative_area: float
    observed_points: int
    reconstructed_points: int
    #: ОТКРЫТЫЙ список строк: контракт C-09 версии 0.1 допускает расширение кодов.
    #: Поэтому здесь запрещены Literal/Enum, а в БД —
    #: CHECK и enum-тип: неизвестный код обязан дойти до UI как есть, без 500.
    reason_codes: list[str]
    #: Объяснение на русском. Формулировки без утверждений о причинности.
    explanation_ru: str
    algorithm_version: str
    #: Оба поля производит детектор (`to_dict`); без них потребитель не отличит
    #: событие C-09 0.1 от будущих версий и посчитает длительность по-своему.
    duration_days: int | None = None
    schema_version: str | None = None


class AnomaliesResponse(BaseModel):
    """Конверт детектора целиком, а не только события.

    `DetectionResult` несёт warnings отдельно от событий. Без них пустой `items`
    означал бы одновременно «поле в норме» и «данных не было»; при недостатке
    данных отсутствие события не считается подтверждённой нормой.
    """

    analysis_id: str
    #: Пустой список — валидный результат «аномалий нет», а не ошибка.
    items: list[AnomalyOut]
    #: Диагностические предупреждения детектора: EMPTY_SERIES,
    #: INSUFFICIENT_REFERENCE_YEARS, MISSING_HARMONIZED_VALUES, LOW_RECONSTRUCTION_SUPPORT.
    #: Список открытый по той же причине, что и reason_codes.
    warnings: list[str] = Field(default_factory=list)
    schema_version: str | None = None
    algorithm_version: str | None = None
    #: Семантика confidence, как её объявляет производитель. C-09 присылает
    #: "heuristic_support_not_calibrated_probability" — UI не имеет права
    #: показывать это значение как вероятность аномалии.
    confidence_semantics: str | None = None


class ProvenanceEntry(BaseModel):
    """Происхождение данных (`GET /analyses/{id}/provenance`)."""

    provider: str
    collection_id: str
    collection_version: str | None = None
    item_ids: list[str] = Field(default_factory=list)
    queried_at: datetime
    crs: str | None = None
    resolution_m: float | None = None
    qa_definition: str | None = None
    processing_params: dict[str, Any] = Field(default_factory=dict)
    license_url: str | None = None
    #: Честная маркировка кэша. Выдавать кэшированный результат за live запрещено
    #: (инвариант 10, решение D-005); UI показывает этот флаг и дату извлечения.
    cached: bool = False


class ProvenanceResponse(BaseModel):
    analysis_id: str
    entries: list[ProvenanceEntry]


class AnalysisOut(BaseModel):
    id: str
    polygon_id: str
    date_from: date
    date_to: date
    state: JobState
    pipeline_version: str | None = None
    model_version: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    #: True, если хотя бы один провайдер отдал неполные данные (стадия PARTIAL).
    partial: bool = False
    #: Дублирует маркировку кэша на уровне анализа — см. комментарий в SeriesResponse.
    cached: bool = False
    data_retrieved_at: datetime | None = None
