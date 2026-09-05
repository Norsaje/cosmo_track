"""Проверки фронтенда (BE-005/BE-013) на уровне разметки и скрипта.

Полноценного браузерного теста здесь нет — он потребовал бы headless-браузера,
которого нет в offline CI. Поэтому проверяется то, что проверяемо статически и
что легче всего потерять при правке: обязательные по ТЗ элементы интерфейса и
формулировки, за которыми стоят прямые запреты.

Каждый тест ниже соответствует пункту, где ошибка UI означает не косметику,
а неверное утверждение о данных.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

INDEX = Path(__file__).resolve().parents[2] / "apps/web/index.html"

pytestmark = pytest.mark.skipif(not INDEX.is_file(), reason="нет apps/web/index.html")


@pytest.fixture(scope="module")
def page() -> str:
    """Разметка с нормализованными пробелами.

    Проверяются формулировки, а не вёрстка: одна и та же фраза в HTML легко
    оказывается разбитой переносом строки при переформатировании. Схлопывание
    пробелов оставляет тест про смысл — иначе он ломается от любой правки отступов
    и вынуждает верстать под тест вместо того, чтобы верстать читаемо.
    """
    raw = INDEX.read_text(encoding="utf-8")
    return re.sub(r"\s+", " ", raw)


@pytest.fixture(scope="module")
def visible() -> str:
    """Страница без JS-комментариев.

    Запреты вида «не подписывать интервал процентом» проверяются по тому, что
    увидит пользователь. В комментариях запрещённая формулировка встречается
    намеренно — как пример того, чего делать нельзя, — и считать её нарушением
    значит заставить документацию коверкать собственные слова.
    """
    raw = INDEX.read_text(encoding="utf-8")
    return re.sub(r"\s+", " ", re.sub(r"^\s*//.*$", "", raw, flags=re.MULTILINE))


def test_required_controls_exist(page: str) -> None:
    """CP-4: «Полигон можно найти или нарисовать» — значит в UI есть чем рисовать."""
    markers = ('id="map"', 'id="btn-draw"', 'id="btn-finish"', 'id="btn-analyse"', 'id="chart"')
    for marker in markers:
        assert marker in page, marker


def test_series_distinguishes_observed_from_reconstructed(page: str) -> None:
    """CP-4: «UI различает observed/reconstructed/uncertainty/anomaly».

    Проверяется, что скрипт вообще читает эти поля: без них график нарисовал бы
    одну линию, где измерение и модельная оценка неразличимы.
    """
    fields = ("is_observed", "is_reconstructed", "primary_ndvi_reconstructed", "lower", "upper")
    for field in fields:
        assert field in page, field
    # Легенда обязана называть обе сущности словами, а не только цветом (§8.9 ТЗ).
    assert "наблюдение" in page
    assert "восстановлено моделью" in page


def test_uncertainty_is_not_labelled_as_a_percentage(page: str, visible: str) -> None:
    """Интервал не подписывается процентом уверенности.

    `interval_status` у producer прямо говорит, что интервалы эмпирические и
    формально не сертифицированы, поэтому «95 % уверенности» было бы неверным
    утверждением о точности.
    """
    assert "не сертифицирован" in page
    # Ищем подписи вида «95 %» рядом со словом «уверенн»: именно так выглядит ошибка.
    assert not re.search(r"9[05]\s*%\s*уверенн", visible, re.IGNORECASE)


def test_confidence_is_not_called_a_probability(page: str) -> None:
    """C-09: confidence — эвристическая поддержка, а не вероятность аномалии."""
    assert "эвристическая" in page
    assert "не вероятность" in page


def test_absence_of_events_is_not_shown_as_normality(page: str) -> None:
    """«Событий нет» и «сравнивать не с чем» — разные состояния.

    Отсутствие события при недостатке истории нельзя показывать как подтверждённую
    норму: это требование семантики C-09.
    """
    assert "INSUFFICIENT_REFERENCE_YEARS" in page
    assert "не подтверждение нормы" in page


def test_causality_is_not_asserted(page: str) -> None:
    """Red-team перед CP-4: «Anomaly explanation не утверждает причинность»."""
    assert "не означает причину" in page


def test_cached_result_is_marked(page: str) -> None:
    """Инвариант 10 и решение D-005: выдавать кэш за live запрещено."""
    assert "cached" in page
    assert "из кэша" in page


def test_temporal_transfer_limitation_is_visible(page: str) -> None:
    """Ограничение модели проговаривается в интерфейсе, а не только в README.

    Владелец C-04 предупреждает прямым текстом: на переносе в будущий сезон
    ансамбль уступает простому baseline. Умолчать об этом на демо нельзя.
    """
    assert "Перенос на будущий сезон" in page
    assert "уступает простому baseline" in page


def test_chart_axis_is_not_clamped_to_physical_range(page: str) -> None:
    """Ось Y строится по данным, а не по [-1, 1].

    Реальный target выходит за физический диапазон (train до −2.13, test до 1.84);
    жёсткая обрезка спрятала бы часть ряда. Проверяется поведение кода, а не
    формулировка комментария: комментарий можно переписать, а зажатая ось —
    это молчаливая потеря данных на графике.
    """
    assert "minValue" in page and "maxValue" in page
    # Границы вычисляются из значений ряда.
    assert "Math.min.apply" in page and "Math.max.apply" in page
    # И нигде не подставляются константы физического диапазона.
    assert not re.search(r"minValue\s*=\s*-?1(\.0)?\b", page)
    assert not re.search(r"maxValue\s*=\s*1(\.0)?\b", page)


def test_map_failure_degrades_without_breaking_analysis(page: str) -> None:
    """Отсутствие сети ломает подложку, но не сервис.

    MapLibre грузится из CDN; если инициализация карты уронит скрипт, страница
    будет выглядеть мёртвой при полностью живом API.
    """
    assert 'typeof maplibregl === "undefined"' in page
    assert "Это не отказ сервиса" in page


def test_idempotent_response_is_explained(page: str) -> None:
    """Идемпотентный ответ 200 отличается от нового запуска 202 и объясняется."""
    assert "response.status === 200" in page
    assert "уже выполнялся" in page


def test_script_has_no_obvious_syntax_break(page: str) -> None:
    """Грубая проверка целостности: баланс скобок в inline-скрипте.

    Не заменяет разбор JS, но ловит обрыв файла и незакрытый блок — самую частую
    поломку при ручной правке большого inline-скрипта.
    """
    raw = INDEX.read_text(encoding="utf-8")
    script = raw.split("<script>")[-1].rsplit("</script>", 1)[0]
    assert script.count("{") == script.count("}")
    assert script.count("(") == script.count(")")


def test_data_source_is_chosen_before_saving_a_field(page: str) -> None:
    """Ряд наблюдений выбирается при создании поля.

    Геометрий в конкурсных CSV нет, автоматически связать нарисованный контур
    с рядом невозможно, а живых провайдеров ещё нет. Значит выбор делает человек —
    и делает его до сохранения, а не после падения анализа.
    """
    assert 'id="field-series"' in page
    assert "/reference-polygons" in page
    assert "Ряд наблюдений" in page


def test_field_without_data_source_cannot_start_analysis(page: str) -> None:
    """Кнопка анализа заблокирована, пока у поля нет источника данных.

    Активная кнопка, ведущая к гарантированной ошибке через несколько секунд, —
    это обещание, которое интерфейс не может выполнить.
    """
    assert "$(\"btn-analyse\").disabled = !series" in page
    assert "анализ недоступен" in page
    # В списке полей отсутствие источника видно сразу, а не только при выборе.
    assert "без данных" in page


def test_service_indicator_appears_only_on_trouble(page: str) -> None:
    """В норме индикатор состояния скрыт.

    Постоянно зелёный значок не несёт информации: глаз перестаёт его замечать
    ровно к тому моменту, когда он впервые становится важным. Уведомление должно
    появляться как отклонение, а не висеть всегда.
    """
    assert "chip.hidden = true" in page
    assert "нет соединения с сервисом" in page
    # Позитивного состояния в интерфейсе больше нет.
    assert "сервис готов" not in page


def test_service_indicator_is_announced_to_screen_readers(page: str) -> None:
    """Появление уведомления должно быть озвучено, а не только показано.

    Индикатор возникает без действия пользователя, поэтому без aria-live человек,
    работающий со скринридером, о потере связи просто не узнает.
    """
    assert 'aria-live="polite"' in page
    assert 'role="status"' in page
