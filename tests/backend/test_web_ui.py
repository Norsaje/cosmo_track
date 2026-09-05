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
    markers = ('id="map"', 'id="btn-draw"', 'id="btn-finish"', 'id="chart"')
    for marker in markers:
        assert marker in page, marker
    # Кнопка запуска создаётся вместе с развёрнутой плиткой, а не лежит в
    # статической разметке, поэтому проверяется её создание.
    assert 'run.id = "btn-analyse"' in page


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


def test_model_limitation_is_visible(page: str) -> None:
    """Ограничение модели проговаривается в интерфейсе, а не только в README.

    Скрытые ответы организаторов недоступны: всё измеренное качество — локальное,
    на двадцати полигонах нового test. Показать число без этой оговорки значит
    выдать локальную оценку за официальный балл.
    """
    assert "измерено локально" in page
    assert "официальный балл не" in page
    # Ограничение прежнего бандла C-04 к работающей модели не относится: на её
    # audit смесь baseline не уступает. Оставить старую фразу значит приписать
    # одной модели слабое место другой.
    assert "уступает простому baseline" not in page


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
    """Неработающая карта не выдаётся за отказ сервиса.

    Если инициализация карты уронит скрипт, страница будет выглядеть мёртвой при
    полностью живом API, поэтому отсутствие библиотеки проверяется явно. И оба
    сообщения о проблемах с картой обязаны сказать, что остальное работает —
    иначе пользователь решит, что сломан весь сервис, и не станет ничего делать.
    """
    assert 'typeof maplibregl === "undefined"' in page
    assert "недоступна только карта" in page
    assert "не видно только фон карты" in page


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


def test_page_loads_nothing_from_external_cdn(visible: str) -> None:
    """Страница не должна зависеть от доступности чужих доменов.

    Реальный случай с демо: стенд открыли с другой машины, у браузера не было
    доступа к unpkg.com и tile.openstreetmap.org — карта не появилась при
    полностью исправном сервере, и в логах не было ни одной ошибки, потому что
    браузер ходил за этими файлами мимо нас.

    Теперь и библиотека карты, и тайлы отдаются нашим сервером. Проверяются
    только реальные ссылки (src/href), комментарии из проверки исключены.
    """
    external = re.findall(r'(?:src|href)="(https?://[^"]+)"', visible)
    assert external == [], f"страница тянет внешние ресурсы: {external}"
    assert '"/vendor/maplibre-gl.js"' in visible
    assert "/tiles/{z}/{x}/{y}.png" in visible


def test_map_problems_are_distinguished(page: str) -> None:
    """Причины неработающей карты различаются.

    Одна плашка на все беды скрывала причину: «нет доступа к сети» показывалось
    и когда не загрузилась библиотека, и когда не пришёл один тайл. Чинятся эти
    случаи по-разному, значит и называться должны по-разному.
    """
    assert "Библиотека карты не загрузилась" in page
    assert "Подложка карты не загружается" in page
    # Ошибка одного тайла не должна поднимать плашку: MapLibre шлёт error и на
    # отменённых при зуме запросах, а пугать сообщением поверх работающей карты нельзя.
    assert "tilesLoaded === 0 && tileErrors >= 3" in page


def test_hidden_attribute_actually_hides(page: str) -> None:
    """`hidden` обязан скрывать, а не просто присутствовать в разметке.

    Браузер реализует его как `[hidden]{display:none}` в собственном стиле, и любое
    авторское правило с display перебивает его по специфичности. Здесь это ловилось
    на живом стенде: `.chip{display:inline-flex}` и `.alert{display:flex}` показывали
    элементы с hidden — в шапке висел пустой красный индикатор, а плашка о подложке
    карты была видна независимо от того, загрузились тайлы или нет.

    Правило проверяется явно, потому что дефект не виден ни в разметке, ни в логах:
    атрибут на месте, JS отрабатывает верно, а элемент всё равно на экране.
    """
    assert re.search(r"\[hidden\]\s*\{[^}]*display:\s*none\s*!important", page)


def test_hidden_elements_use_classes_that_would_override_it(page: str) -> None:
    """Проверка осмысленности предыдущего теста.

    Если однажды ни один скрываемый элемент не будет иметь класса с display,
    правило `[hidden]` перестанет быть нужным — и тест выше станет пустым.
    Пока такие элементы есть, защита обязательна.
    """
    assert "display: inline-flex" in page or "display: flex" in page
    assert re.search(r'id="tiles-note"[^>]*hidden', page)
    assert re.search(r'id="health-chip"[^>]*hidden', page)


def test_status_indicator_starts_neutral(page: str) -> None:
    """До первой проверки состояние неизвестно, и красный цвет сообщал бы неправду."""
    assert re.search(r'id="health-chip" class="chip chip-default"', page)


def test_map_takes_the_whole_screen(page: str) -> None:
    """Карта — главный объект работы, а не полоска сверху.

    `position:absolute; inset:0` вместо строки грида: панель лежит поверх карты,
    а не отнимает у неё высоту.
    """
    assert re.search(r"#map\s*\{[^}]*position:\s*absolute[^}]*inset:\s*0", page)


def test_sheet_can_be_dragged(page: str) -> None:
    """Меню открывается свайпом справа налево.

    Pointer-события, а не touch: та же логика работает мышью и стилусом и не
    требует отдельной ветки кода. Жест начинается только от кромки экрана —
    иначе он отбирал бы у карты обычное перетаскивание пальцем.
    """
    for handler in ("pointerdown", "pointermove", "pointerup", "pointercancel"):
        assert handler in page, handler
    assert "EDGE_ZONE" in page
    assert "event.clientX >= width - EDGE_ZONE" in page
    # Панель либо закрыта, либо открыта: промежуточных положений у боковой нет.
    assert 'data-state="closed"' in page
    assert 'sheet[data-state="open"]' in page


def test_sheet_shows_it_can_be_dragged(page: str) -> None:
    """Пользователь должен видеть, что панель вытягивается, не пробуя жест наугад.

    Жест, о котором нельзя догадаться, для большинства не существует, поэтому
    подсказок две и они разного рода: язычок у той самой кромки, откуда тянуть,
    и мини-меню внизу, которое вообще не требует жеста.
    """
    assert 'id="edge-hint"' in page
    assert "потяните влево" in page
    assert ".edge-hint::after" in page  # полоска-захват, как у ручки
    assert 'class="dock"' in page


def test_sheet_is_operable_without_gestures(page: str) -> None:
    """Жест не должен быть единственным способом.

    Свайп недоступен при работе со скринридером и неудобен мышью, поэтому то же
    состояние достигается кнопками мини-меню, язычком, крестиком и касанием по
    свободной карте.
    """
    assert '$("edge-hint").addEventListener("click", openSheet)' in page
    assert '$("dock-fields").addEventListener("click", openSheet)' in page
    assert '$("dock-map").addEventListener("click", closeSheet)' in page
    assert '$("sheet-close").addEventListener("click", closeSheet)' in page
    # Касание мимо панели закрывает её через саму карту, а не через затемнение:
    # см. test_open_panel_does_not_take_the_map_away.
    assert re.search(
        r'queryRenderedFeatures.{0,900}?isMobile\(\) && state\.sheet === "open"\)'
        r" closeSheet\(\)",
        page,
    )


def test_open_panel_does_not_take_the_map_away(page: str) -> None:
    """Открытая панель не имеет права выключать карту.

    Затемнение растянуто на `inset: 0`, то есть и на полосу карты, которая
    остаётся видимой слева от панели. Пока оно ловило указатель, карта не
    получала ни одного касания: её нельзя было ни сдвинуть, ни свести пальцами,
    ни нажать кнопки масштаба — панель просто захлопывалась в ответ на любое
    прикосновение. Затемнение обязано остаться чисто декоративным.
    """
    rules = re.findall(r"\.backdrop[^{]*\{[^}]*\}", page)
    assert rules, "правила .backdrop не найдены — тест потерял предмет проверки"
    for rule in rules:
        assert "pointer-events: auto" not in rule, rule
    # Кнопки масштаба уходят из-под панели: без этого «увеличить/уменьшить»
    # пропадали ровно при открытом списке полей.
    assert 'body[data-sheet="open"] .maplibregl-ctrl-top-right' in page
    assert "document.body.dataset.sheet = name" in page


def test_reverse_swipe_closes_the_panel(page: str) -> None:
    """Панель закрывается тем же жестом, которым открылась, только обратным.

    Ось X внутри панели обязана принадлежать жесту, а не браузеру. Без явного
    `touch-action: pan-y` на прокручиваемом теле панели браузер считал
    горизонтальное движение своим и после второго `pointermove` присылал
    `pointercancel`: свайп вправо по списку не закрывал ничего. `preventDefault`
    здесь бессилен — для pointer-событий прокрутку решает только `touch-action`.
    """
    assert re.search(r"\.sheet-body \{[^}]*touch-action: pan-y", page)
    # Направление обязательно: `decisive` считает модуль пути, и без проверки
    # знака быстрое движение влево внутри открытой панели её закрывало.
    assert "(decisive && dx > 0) || dx > width * OPEN_THRESHOLD" in page


def test_hint_opens_by_a_plain_tap(page: str) -> None:
    """По подсказке «потяните влево» сначала нажимают, а потом уже тянут.

    Жест, о котором написано словами, всё равно проверяют нажатием, и отказ
    читается как поломка. Касанием считается жест, никуда не уехавший: сравнение
    `dx <= 0` касанием не считало ничего, потому что палец на язычке сползает
    вправо на пару пикселей.
    """
    assert "TAP_SLOP" in page
    assert "const tapped = moved <= TAP_SLOP" in page
    assert "(fromHint && (tapped || dx <= 0))" in page


def test_synthetic_click_after_a_gesture_is_swallowed(page: str) -> None:
    """Клик, синтезированный из того же касания, не должен отменять жест.

    Браузер присылает click следом за pointerup — уже в новую раскладку. После
    нажатия на язычок он попадал в карту (язычок к тому моменту скрыт открытой
    панелью), а карта по касанию панель закрывает: меню открывалось и тут же
    захлопывалось. После свайпа тот же клик попадал в плитку под пальцем и
    выбирал поле. Глотается ровно один клик и только сразу за жестом — иначе
    перестали бы работать крестик и выбор поля касанием.
    """
    assert "swallowClickUntil" in page
    assert "swallowClickUntil = (!tapped || (opening && fromHint)) ? Date.now() + 700 : 0" in page
    assert re.search(r'addEventListener\("click", \(event\) => \{ if \(Date\.now\(\) > '
                     r"swallowClickUntil\) return;", page)


def test_browser_gestures_do_not_override_page_gestures(page: str) -> None:
    """Свайпы браузера не должны конкурировать со свайпами страницы.

    Мета-тега для этого нет — работает `overscroll-behavior`: он выключает
    навигацию «назад/вперёд» перетягиванием содержимого и перезагрузку рывком
    вниз. Системный жест от кромки экрана странице не подчиняется вообще,
    поэтому панель обязана открываться и без жеста — это проверяет
    test_sheet_is_operable_without_gestures.
    """
    assert re.search(r"html, body \{ overscroll-behavior: none; \}", page)


def test_severity_categories_are_distinguished(page: str) -> None:
    """UI различает три категории C-09, а не красит всё красным.

    `critical` и `biomass_suppression` — разные состояния поля, и одинаковый
    красный чип уравнивал бы их. Категория `baseline_unranked` у fallback-а не
    красная намеренно: baseline тяжесть не оценивает, и красный цвет приписал бы
    событию оценку, которой никто не давал.
    """
    assert "const SEVERITY = {" in page
    for name in ("critical", "biomass_suppression", "normal", "baseline_unranked"):
        assert name + ":" in page, name
    # Неизвестная категория обязана дойти до экрана как есть: контракт 0.1,
    # закрывать множество на нашей стороне запрещено (§4.1).
    assert 'SEVERITY[severity] || { chip: "chip-default"' in page


def test_producer_explanation_and_algorithm_version_are_shown(page: str) -> None:
    """Пояснение пишет производитель детектора, версия алгоритма видна.

    Пересказывать вывод чужого алгоритма своими словами — значит отвечать за
    формулировку, которой он не давал. Версия обязательна: результаты 0.1.0 и
    0.1.1 различаются, и без неё непонятно, чем получены события на экране.
    """
    assert "event.explanation_ru" in page
    assert "e.algorithm_version" in page
    assert "Детектор:" in page


def test_vertical_movement_belongs_to_the_content(page: str) -> None:
    """Прокрутка внутри панели не должна её закрывать.

    У боковой панели жест горизонтальный, а список внутри листается вертикально.
    Без разделения по преобладающей оси любое движение пальца вверх по таблице
    аномалий уводило бы панель за экран.
    """
    assert "Math.abs(dy) > Math.abs(dx)" in page


def test_analysis_lives_inside_the_field_tile(page: str) -> None:
    """Анализ открывается внутри плитки поля, а не отдельной карточкой ниже.

    Отдельная карточка заставляла соотносить её с выбранной плиткой глазами и
    прокруткой: на телефоне список и форма не помещались на экран одновременно.
    Теперь плитка разворачивается на месте и содержит всё, что относится к полю.
    """
    assert "function buildDetails(polygon)" in page
    assert 'box.className = "tile-details"' in page
    # Порядок частей развёрнутой плитки: координаты, период, запуск, ход
    # выполнения. Удаление стоит выше, в шапке плитки, — оно про поле целиком.
    assert "box.append(coordsTitle, coords, periodLabel, dates, run, job)" in page
    # Повторный клик по плитке сворачивает её.
    assert "select(polygon.id === state.selected ? null : polygon.id)" in page


def test_tile_shows_delete_and_vertex_coordinates(page: str) -> None:
    """В развёрнутой плитке — удаление в правом верхнем углу и координаты вершин.

    Удаление относится к полю целиком, поэтому стоит отдельно от кнопки запуска:
    рядом с ней легко промахнуться после долгой настройки периода.
    """
    assert 'remove.className = "tile-delete btn-danger"' in page
    # Кнопка делит строку с текстом как сосед по flex, а не лежит поверх него.
    # Абсолютное позиционирование резервировало место отступом у заголовка, и
    # резерв переставал работать при другой длине названия — кнопка наезжала
    # на вторую строку. Проверяется именно отсутствие наложения по построению.
    assert re.search(r"\.tile-head\s*\{[^}]*display:\s*flex", page)
    assert re.search(r"\.tile-main\s*\{[^}]*flex:\s*1[^}]*min-width:\s*0", page)
    assert not re.search(r"\.tile-delete\s*\{[^}]*position:\s*absolute", page)
    assert "head.appendChild(buildDeleteButton(polygon))" in page
    assert "Вершины контура (широта, долгота)" in page
    # Замыкающая точка кольца GeoJSON дублирует первую — человеку её не показываем.
    assert "ring.slice(0, -1)" in page


def test_tile_survives_being_rerendered(page: str) -> None:
    """Введённый период не теряется при перерисовке списка.

    Плитка пересоздаётся на каждый выбор, вместе с ней и поля ввода. Без хранения
    периода в состоянии выбранные даты молча возвращались бы к значениям
    по умолчанию — а человек уже нажал бы «Запустить».
    """
    assert "dateFrom:" in page and "dateTo:" in page
    assert "date_from: state.dateFrom, date_to: state.dateTo" in page


def test_dock_hides_when_the_panel_is_open(page: str) -> None:
    """Мини-меню уступает место открытой панели.

    Оно плавает поверх и закрывало кнопку «Запустить анализ» ровно тогда, когда
    до неё добирались. Вернуться к карте можно крестиком в шапке.
    """
    assert '$("dock").hidden = name === "open"' in page


def test_panel_and_field_are_linkable(page: str) -> None:
    """Ссылка открывает панель и разворачивает нужное поле.

    Коллеге отправляют ссылку на конкретный участок, а не на пустую карту
    с просьбой найти его самому.
    """
    assert 'hash.indexOf("#field=") === 0' in page
    assert 'hash === "#menu"' in page
    assert 'window.addEventListener("hashchange", applyHash)' in page


def test_drawing_controls_live_over_the_map(page: str) -> None:
    """Кнопки рисования — поверх карты, а не в панели.

    Рисуют по карте; уходить за кнопкой в панель значило бы закрывать то,
    по чему собираешься кликать.
    """
    assert 'class="map-actions"' in page
    assert re.search(r'\.map-actions\s*\{[^}]*position:\s*absolute', page)


def test_closed_sheet_is_fully_off_screen(page: str) -> None:
    """Закрытая панель уходит за экран целиком.

    Прежняя, нижняя, оставляла видимой шапку — и из-под неё выглядывала
    обрезанная карточка, что читалось как недогрузившийся экран. У боковой
    панели такого состояния нет вовсе: она либо открыта, либо её не видно,
    а роль приглашения играет отдельный язычок.
    """
    assert re.search(r"\.sheet\s*\{[^}]*transform:\s*translateX\(100%\)", page)
    assert re.search(r'\.sheet\[data-state="open"\]\s*\{\s*transform:\s*translateX\(0\)', page)


def test_bottom_dock_offers_both_sections(page: str) -> None:
    """Мини-меню внизу — вторая, безжестовая дорога к тем же состояниям.

    Две вкладки, обе с иконкой и подписью: иконка без текста опознаётся не всеми,
    а подпись без иконки хуже находится боковым зрением.
    """
    assert 'id="dock-map"' in page and 'id="dock-fields"' in page
    # Подпись ищется внутри самой кнопки, а не где угодно на странице: слово
    # «Поля» встречается и в заголовке карточки, и совпадение там ничего не значит.
    dock_map = re.search(r'id="dock-map".*?</button>', page)
    dock_fields = re.search(r'id="dock-fields".*?</button>', page)
    assert dock_map and "Карта" in dock_map.group(0)
    assert dock_fields and "Поля" in dock_fields.group(0)
    assert "<svg" in dock_map.group(0) and "<svg" in dock_fields.group(0)
    # Текущий раздел помечен состоянием, а не только цветом.
    assert 'aria-selected' in page


def test_offscreen_panel_does_not_widen_the_document(page: str) -> None:
    """Закрытая панель не имеет права расширять страницу.

    Она стоит `translateX(100%)`, то есть физически лежит справа за экраном и
    добавляет к ширине прокрутки свои 335 px. `overflow` у `body` не помогает:
    при `visible` у `html` браузер поднимает свойство на область просмотра, а
    сам body остаётся `visible`. Телефон видит документ шире экрана, уменьшает
    масштаб — и справа появляется белая полоса в ширину панели. Клип ставится
    на `main`, внутри которого панель и лежит.
    """
    assert re.search(r"main\s*\{[^}]*overflow:\s*hidden", page)


def test_layout_switches_by_height_too(page: str) -> None:
    """Телефон в альбомной ориентации остаётся с накладной панелью.

    Порог только по ширине отправлял экран 844x390 в «десктоп»: панель на
    380 px намертво занимала половину экрана, а карте оставалась полоса.
    Условие в CSS и в JS обязано быть одним и тем же — иначе CSS нарисует
    накладную панель, а скрипт будет считать её пристыкованной и перестанет
    открывать.
    """
    assert "(min-width: 768px) and (min-height: 600px)" in page
    assert "@media (max-width: 767px), (max-height: 599px)" in page
    assert 'matchMedia("(max-width: 767px), (max-height: 599px)")' in page
    # Точки переключения без учёта высоты остаться не должно.
    assert "@media (min-width: 768px) {" not in page


def test_default_period_covers_the_year_with_data(page: str) -> None:
    """Период по умолчанию — 2024 год целиком.

    Наблюдения в наборе заканчиваются 30 октября 2024-го. Период по текущей
    дате не содержал бы ни одной строки, и первый же запуск анализа падал бы
    с «нет строк в диапазоне» на поле, у которого данные есть.
    """
    assert 'dateFrom: "2024-01-01", dateTo: "2024-12-31"' in page


def test_brand_is_braining_space(page: str) -> None:
    """Название и слоган продукта."""
    assert "Braining Space" in page
    assert "Думаем о вашем хозяйстве за вас" in page
    assert "cosmo_track" not in page


NGINX = INDEX.parent / "nginx.conf"


def test_satellite_basemap_is_available(page: str) -> None:
    """Есть вторая подложка — снимок.

    По схеме границы участков не нарисованы, и обвести контур поля по ней
    невозможно: ориентиры есть, самого поля нет.
    """
    assert "satellite" in page
    assert "Спутник" in page and "Схема" in page
    assert 'id="btn-basemap"' in page


def test_esri_tiles_use_zyx_order(page: str) -> None:
    """У Esri порядок координат {z}/{y}/{x}, а не {z}/{x}/{y}.

    Это не педантизм: перепутанные оси дают мозаику из чужих мест, которая
    выглядит правдоподобно — человек обведёт «своё» поле в другой области и
    не заметит подмены.
    """
    assert "/tiles-sat/{z}/{y}/{x}" in page
    # У схемы порядок обычный, и они не должны совпасть по недосмотру.
    assert "/tiles/{z}/{x}/{y}.png" in page


def test_satellite_attribution_is_kept(page: str) -> None:
    """Атрибуция Esri обязательна по условиям использования их слоя."""
    assert "Esri, Maxar, Earthstar Geographics" in page


def test_basemap_switch_keeps_layers(page: str) -> None:
    """Переключение меняет прозрачность, а не пересоздаёт стиль.

    Пересоздание стиля выбрасывало бы уже загруженные тайлы и слои с контурами
    полей — карта моргала бы и теряла выделение при каждом переключении.
    """
    assert 'setPaintProperty("satellite", "raster-opacity"' in page
    # На снимке синяя заливка контуров читается хуже, чем на светлой схеме.
    assert 'setPaintProperty("fields-line", "line-width"' in page


def test_satellite_tiles_are_proxied_and_cached() -> None:
    """Снимки идут через наш сервер, как и схема.

    У браузера пользователя может не быть доступа наружу — на этом уже один раз
    пропала подложка. Кэш отдельный: снимки в разы тяжелее схемы и в общей зоне
    вытесняли бы её как раз тогда, когда нужны оба слоя.
    """
    config = NGINX.read_text(encoding="utf-8")
    assert "location /tiles-sat/" in config
    assert "proxy_cache tiles_sat;" in config
    assert "keys_zone=tiles_sat" in config
    assert "server.arcgisonline.com" in config
