"""Smoke-проверки scripts/validate_submission.py.

Покрывает:
- CLI --help exit 0;
- integration на реальном test_data.csv + полноразмерный valid submission;
- все bad-fixtures возвращают non-zero (exit 1 или 2);
- valid fixture возвращает 0;
- --json-report детерминирован;
- --warn-outside не ломает валидный submission;
- абсолютные пути → exit 2.

Тесты используют main() напрямую через monkeypatch sys.argv (не
subprocess), плюс один subprocess-тест на --help.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts.validate_submission import (
    EXPECTED_ROW_COUNT,
    extract_test_gap_keys,
)
from scripts.validate_submission import (
    main as vs_main,
)

ROOT = Path(__file__).resolve().parents[2]
TMP = "tests/smoke/_tmp"
TEST = "data/test_data.csv"


# ───────────────────────────── helpers ─────────────────────────────

@pytest.fixture
def smoke_tmp(monkeypatch):
    monkeypatch.chdir(ROOT)
    tmp_path = ROOT / TMP
    tmp_path.mkdir(parents=True, exist_ok=True)
    yield tmp_path


@pytest.fixture
def real_gap_keys() -> set[tuple[str, str]]:
    test_df = pd.read_csv(ROOT / TEST)
    return extract_test_gap_keys(test_df)


@pytest.fixture
def full_valid_submission(smoke_tmp, real_gap_keys) -> Path:
    """Генерирует полноразмерный валидный submission (3112 строк).

    Файл кладётся в smoke_tmp (tests/smoke/_tmp/), чтобы relative_to(ROOT)
    работал корректно на Windows.
    """
    rows = sorted(real_gap_keys)
    df = pd.DataFrame({
        "anon_polygon_id": [r[0] for r in rows],
        "date": [r[1] for r in rows],
        "primary_ndvi_pred": [0.5] * len(rows),
    })
    path = smoke_tmp / "submission_full_valid.csv"
    df.to_csv(path, index=False)
    return path


# ───────────────────────────── CLI --help ─────────────────────────────

class TestCliHelp:
    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.validate_submission", "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--test" in result.stdout
        assert "--submission" in result.stdout
        assert "--json-report" in result.stdout
        assert "--warn-outside" in result.stdout


# ───────────────────────────── integration ─────────────────────────────

class TestIntegrationReal:
    def test_full_valid_submission_passes(self, smoke_tmp, full_valid_submission):
        sys.argv = [
            "validate_submission",
            "--test", TEST,
            "--submission", "tests/smoke/_tmp/submission_full_valid.csv",
        ]
        assert vs_main() == 0

    def test_full_valid_with_json_report(self, smoke_tmp, full_valid_submission):
        report = f"{TMP}/vs_report.json"
        sys.argv = [
            "validate_submission",
            "--test", TEST,
            "--submission", "tests/smoke/_tmp/submission_full_valid.csv",
            "--json-report", report,
        ]
        assert vs_main() == 0
        payload = json.loads((ROOT / report).read_text(encoding="utf-8"))
        assert payload["is_valid"] is True
        assert payload["summary"]["rows"] == EXPECTED_ROW_COUNT
        assert payload["summary"]["key_set_match"] is True
        assert payload["summary"]["finite_predictions"] == EXPECTED_ROW_COUNT

    def test_warn_outside_does_not_break_valid(self, smoke_tmp, full_valid_submission):
        sys.argv = [
            "validate_submission",
            "--test", TEST,
            "--submission", "tests/smoke/_tmp/submission_full_valid.csv",
            "--warn-outside", "0.0", "1.0",
        ]
        # У нас все predictions = 0.5 → внутри диапазона, warnings пусты.
        assert vs_main() == 0

    def test_short_submission_fails(self, smoke_tmp):
        # Только 5 строк → row_count_mismatch и missing_key.
        df = pd.DataFrame({
            "anon_polygon_id": ["AOI-0001"] * 5,
            "date": ["2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04", "2020-04-05"],
            "primary_ndvi_pred": [0.5] * 5,
        })
        path = smoke_tmp / "short.csv"
        df.to_csv(path, index=False)
        sys.argv = [
            "validate_submission",
            "--test", TEST,
            "--submission", f"{TMP}/short.csv",
        ]
        assert vs_main() == 1


# ───────────────────────────── exit codes ─────────────────────────────

class TestExitCodes:
    def test_absolute_path_returns_2(self, smoke_tmp, full_valid_submission):
        sys.argv = [
            "validate_submission",
            "--test", "C:/absolute/test.csv",
            "--submission", "tests/smoke/_tmp/submission_full_valid.csv",
        ]
        with pytest.raises(SystemExit) as exc_info:
            vs_main()
        assert exc_info.value.code == 2

    def test_missing_test_returns_2(self, smoke_tmp):
        sys.argv = [
            "validate_submission",
            "--test", "nonexistent_test.csv",
            "--submission", "x.csv",
        ]
        assert vs_main() == 2

    def test_missing_submission_returns_2(self, smoke_tmp):
        sys.argv = [
            "validate_submission",
            "--test", TEST,
            "--submission", "nonexistent_sub.csv",
        ]
        assert vs_main() == 2

    def test_semicolon_returns_2(self, smoke_tmp):
        # Создадим semicolon-delimited файл вручную.
        text = "anon_polygon_id;date;primary_ndvi_pred\nAOI-0001;2020-04-01;0.5\n"
        (smoke_tmp / "sem.csv").write_text(text, encoding="utf-8")
        sys.argv = [
            "validate_submission",
            "--test", TEST,
            "--submission", f"{TMP}/sem.csv",
        ]
        assert vs_main() == 2


# ───────────────────────────── bad fixtures run end-to-end ─────────────────────────────

class TestBadFixturesEndToEnd:
    """Прогон каждого bad-fixture через main() — должен вернуть non-zero."""

    @pytest.mark.parametrize("fixture_name,expected_exit", [
        ("submission_invalid_wrong_columns.csv", 1),
        ("submission_invalid_wrong_order.csv", 1),
        ("submission_invalid_duplicate.csv", 1),
        ("submission_invalid_missing.csv", 1),
        ("submission_invalid_extra.csv", 1),
        ("submission_invalid_date.csv", 1),
        ("submission_invalid_nan.csv", 1),
        ("submission_invalid_inf.csv", 1),
        ("submission_invalid_string.csv", 1),
        ("submission_invalid_pandas_index.csv", 1),
        ("submission_invalid_short.csv", 1),
    ])
    def test_bad_fixture_returns_expected_exit(self, smoke_tmp, fixture_name, expected_exit):
        sys.argv = [
            "validate_submission",
            "--test", TEST,
            "--submission", f"tests/fixtures/{fixture_name}",
        ]
        assert vs_main() == expected_exit

    def test_valid_bom_returns_zero(self, smoke_tmp):
        sys.argv = [
            "validate_submission",
            "--test", TEST,
            "--submission", "tests/fixtures/submission_valid_bom.csv",
        ]
        # Но! valid_bom содержит только 20 ключей, а не 3112.
        # Поэтому он INVALID по key set (missing 3092 keys). Ожидаем exit 1.
        assert vs_main() == 1

    def test_valid_tiny_returns_one(self, smoke_tmp):
        # valid_tiny содержит 20 реальных gap-ключей, но не все 3112.
        # Это INVALID по row count + missing key. Ожидаем exit 1.
        sys.argv = [
            "validate_submission",
            "--test", TEST,
            "--submission", "tests/fixtures/submission_valid_tiny.csv",
        ]
        assert vs_main() == 1