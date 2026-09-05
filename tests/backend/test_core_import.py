"""Компонент «core» задачи SH-005: пакет обязан импортироваться в лёгком окружении.

Этот модуль намеренно не требует extra `web`: его гоняет и Разработчик 4 командой
`uv sync --extra dev && uv run pytest -q`. Требование дословно — `docs/ed_part.md:358`:
«import базового пакета не требует torch».
"""

from __future__ import annotations

import importlib.util
import sys


def test_package_imports() -> None:
    import veg_recovery

    assert veg_recovery.__version__


def test_import_does_not_require_torch() -> None:
    """Импорт пакета не должен затягивать torch: он живёт только в extra `dl`,
    а offline CI идёт без GPU (инвариант 13)."""
    sys.modules.pop("veg_recovery", None)
    import veg_recovery  # noqa: F401

    assert "torch" not in sys.modules


def test_our_subpackages_are_importable() -> None:
    for name in ("veg_recovery.service", "veg_recovery.providers", "veg_recovery.geospatial"):
        assert importlib.util.find_spec(name) is not None, name
