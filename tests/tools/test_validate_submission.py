"""Unit-тесты для scripts.validate_submission.

Покрывает:
- чистые функции: extract_test_gap_keys, validate_submission, write_json_report;
- pydantic модели: ValidationError/Warning/Result;
- все предусмотренные ошибочные случаи (через fixtures);
- BOM handling (UTF-8-sig);
- pandas index rejection;
- --warn-outside (не отклоняет, только warning);
- детерминизм JSON-отчёта.

Все тесты работают без subprocess.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.validate_submission import (
    EXPECTED_ROW_COUNT,
    REPORT_SCHEMA_VERSION,
    SUBMISSION_HEADER,
    ValidationResult,
    extract_test_gap_keys,
    format_stderr_report,
    validate_submission,
    write_json_report,
)

FIX_DIR = Path(__file__).resolve().parents[1] / "fixtures"


# ───────────────────────────── helpers ─────────────────────────────

@pytest.fixture
def real_test_gap_keys() -> set[tuple[str, str]]:
    """Реальные gap-keys из data/test_data.csv (3 112 штук)."""
    test_df = pd.read_csv(FIX_DIR.parent.parent / "data" / "test_data.csv")
    return extract_test_gap_keys(test_df)


@pytest.fixture
def tiny_test_gap_keys() -> set[tuple[str, str]]:
    """Маленький набор gap-keys, соответствующий submission_valid_tiny.csv.

    Для проверки используем первые 20 реальных строк
    gap-ключей из data/test_data.csv — это гарантирует корректную
    семантику key set match.
    """
    test_df = pd.read_csv(FIX_DIR.parent.parent / "data" / "test_data.csv")
    all_gaps = sorted(extract_test_gap_keys(test_df))
    return set(all_gaps[:20])


def _load(name: str) -> pd.DataFrame:
    return pd.read_csv(FIX_DIR / name)


# ───────────────────────────── pydantic models ─────────────────────────────

class TestPydanticModels:
    def test_validation_result_defaults(self):
        from scripts.validate_submission import ValidationSummary
        r = ValidationResult(
            is_valid=True,
            summary=ValidationSummary(
                rows=0, expected_rows=0, key_set_match=True, finite_predictions=0,
            ),
        )
        assert r.errors == []
        assert r.warnings == []
        assert r.is_valid is True

    def test_validation_result_dump(self):
        from scripts.validate_submission import ValidationError, ValidationSummary
        r = ValidationResult(
            is_valid=False,
            errors=[ValidationError(type="x", count=1, examples=["a"], hint="h")],
            summary=ValidationSummary(
                rows=5, expected_rows=10, key_set_match=False, finite_predictions=3,
            ),
        )
        d = r.model_dump()
        assert d["is_valid"] is False
        assert d["errors"][0]["type"] == "x"
        assert d["summary"]["rows"] == 5


# ───────────────────────────── extract_test_gap_keys ─────────────────────────────

class TestExtractTestGapKeys:
    def test_extracts_from_toy(self, tiny_test_gap_keys):
        # Valid fixture имеет 20 строк → 20 gap-keys.
        assert len(tiny_test_gap_keys) == 20

    def test_real_test_has_3112_gaps(self, real_test_gap_keys):
        assert len(real_test_gap_keys) == 3_112
        assert EXPECTED_ROW_COUNT == 3_112


# ───────────────────────────── validate_submission — valid cases ─────────────────────────────

class TestValidateValid:
    def test_valid_tiny_passes(self, tiny_test_gap_keys):
        sub = _load("submission_valid_tiny.csv")
        result = validate_submission(sub, tiny_test_gap_keys)
        assert result.is_valid, result.errors
        assert result.summary.rows == 20
        assert result.summary.key_set_match is True
        assert result.summary.finite_predictions == 20

    def test_valid_bom_passes(self, tiny_test_gap_keys):
        # BOM-prefixed CSV читается как UTF-8-sig.
        raw = (FIX_DIR / "submission_valid_bom.csv").read_bytes()
        assert raw.startswith(b"\xef\xbb\xbf")  # убедимся, что BOM есть
        sub = pd.read_csv(FIX_DIR / "submission_valid_bom.csv")
        result = validate_submission(sub, tiny_test_gap_keys)
        assert result.is_valid, result.errors


# ───────────────────────────── validate_submission — bad cases ─────────────────────────────

class TestValidateBadCases:
    def test_wrong_columns(self, tiny_test_gap_keys):
        sub = _load("submission_invalid_wrong_columns.csv")
        result = validate_submission(sub, tiny_test_gap_keys)
        assert not result.is_valid
        assert any(e.type == "wrong_columns" for e in result.errors)

    def test_wrong_column_order(self, tiny_test_gap_keys):
        sub = _load("submission_invalid_wrong_order.csv")
        result = validate_submission(sub, tiny_test_gap_keys)
        assert not result.is_valid
        # Должна быть ошибка wrong_column_order ИЛИ chain после неё.
        assert any(e.type == "wrong_column_order" for e in result.errors)

    def test_duplicate_key(self, tiny_test_gap_keys):
        sub = _load("submission_invalid_duplicate.csv")
        result = validate_submission(sub, tiny_test_gap_keys)
        assert not result.is_valid
        assert any(e.type == "duplicate_key" for e in result.errors)

    def test_missing_key(self, tiny_test_gap_keys):
        sub = _load("submission_invalid_missing.csv")
        result = validate_submission(sub, tiny_test_gap_keys)
        assert not result.is_valid
        # Должна быть missing_key (и возможно row_count_mismatch).
        assert any(e.type == "missing_key" for e in result.errors)

    def test_extra_key(self, tiny_test_gap_keys):
        sub = _load("submission_invalid_extra.csv")
        result = validate_submission(sub, tiny_test_gap_keys)
        assert not result.is_valid
        assert any(e.type == "extra_key" for e in result.errors)
        # Examples должны содержать AOI-NONEXISTENT.
        ex_err = next(e for e in result.errors if e.type == "extra_key")
        assert any("AOI-NONEXISTENT" in ex for ex in ex_err.examples)

    def test_invalid_date(self, tiny_test_gap_keys):
        sub = _load("submission_invalid_date.csv")
        result = validate_submission(sub, tiny_test_gap_keys)
        assert not result.is_valid
        assert any(e.type == "invalid_date" for e in result.errors)

    def test_nan_prediction(self, tiny_test_gap_keys):
        sub = _load("submission_invalid_nan.csv")
        result = validate_submission(sub, tiny_test_gap_keys)
        assert not result.is_valid
        # NaN попадает в non_finite_prediction / nan_prediction.
        types = {e.type for e in result.errors}
        assert any("nan" in t or "non_finite" in t for t in types)

    def test_inf_prediction(self, tiny_test_gap_keys):
        sub = _load("submission_invalid_inf.csv")
        result = validate_submission(sub, tiny_test_gap_keys)
        assert not result.is_valid
        types = {e.type for e in result.errors}
        assert any("inf" in t or "non_finite" in t for t in types)

    def test_string_prediction(self, tiny_test_gap_keys):
        sub = _load("submission_invalid_string.csv")
        result = validate_submission(sub, tiny_test_gap_keys)
        assert not result.is_valid
        types = {e.type for e in result.errors}
        assert any("string" in t or "non_finite" in t for t in types)

    def test_short_row_count(self, tiny_test_gap_keys):
        sub = _load("submission_invalid_short.csv")
        result = validate_submission(sub, tiny_test_gap_keys)
        assert not result.is_valid
        # row_count_mismatch + missing_key (так как строк мало).
        types = {e.type for e in result.errors}
        assert "row_count_mismatch" in types
        assert "missing_key" in types


class TestValidateSemicolonAndIndex:
    """Кейсы, требующие IO (semicolon delimiter, pandas index)."""

    def test_semicolon_delimiter_rejected_at_io(self, tmp_path, tiny_test_gap_keys):
        """Semicolon-delimited CSV — это IO-уровневая ошибка.
        Тестируем через _read_submission, не через validate_submission."""
        from scripts.validate_submission import _read_submission
        sem_path = FIX_DIR / "submission_invalid_semicolon.csv"
        with pytest.raises(ValueError, match="semicolon"):
            _read_submission(sem_path)

    def test_pandas_index_rejected(self, tiny_test_gap_keys):
        sub = _load("submission_invalid_pandas_index.csv")
        cols = list(sub.columns)
        # Убедимся, что есть Unnamed:0.
        assert any(str(c).startswith("Unnamed") for c in cols)
        result = validate_submission(sub, tiny_test_gap_keys)
        assert not result.is_valid
        types = {e.type for e in result.errors}
        assert "accidental_pandas_index" in types or "wrong_columns" in types


# ───────────────────────────── warn_outside ─────────────────────────────

class TestWarnOutside:
    def test_outside_range_only_warns(self, tiny_test_gap_keys):
        sub = _load("submission_valid_tiny.csv")
        # Принудительно выкрутим одну prediction в 1.5 (NDVI >1 — необычно).
        sub_mod = sub.copy()
        sub_mod.loc[0, "primary_ndvi_pred"] = 1.5
        sub_mod.loc[1, "primary_ndvi_pred"] = -0.5

        result = validate_submission(sub_mod, tiny_test_gap_keys, warn_outside=(0.0, 1.0))
        # submission всё равно валиден — warn_outside не ломает.
        assert result.is_valid
        assert len(result.warnings) == 1
        w = result.warnings[0]
        assert w.type == "prediction_outside_warn_range"
        assert w.count == 2

    def test_no_warn_when_in_range(self, tiny_test_gap_keys):
        sub = _load("submission_valid_tiny.csv")
        result = validate_submission(sub, tiny_test_gap_keys, warn_outside=(0.0, 1.0))
        assert result.warnings == []


# ───────────────────────────── JSON report ─────────────────────────────

class TestWriteJsonReport:
    def test_determinism_and_no_timestamp(self, tmp_path, tiny_test_gap_keys):
        sub = _load("submission_valid_tiny.csv")
        result = validate_submission(sub, tiny_test_gap_keys)
        path1 = tmp_path / "r1.json"
        path2 = tmp_path / "r2.json"
        write_json_report(result, path1)
        write_json_report(result, path2)
        assert path1.read_bytes() == path2.read_bytes()
        payload = json.loads(path1.read_text(encoding="utf-8"))
        assert "timestamp" not in payload
        assert payload["report_schema_version"] == REPORT_SCHEMA_VERSION

    def test_payload_includes_errors(self, tmp_path, tiny_test_gap_keys):
        sub = _load("submission_invalid_nan.csv")
        result = validate_submission(sub, tiny_test_gap_keys)
        path = tmp_path / "r.json"
        write_json_report(result, path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["is_valid"] is False
        assert len(payload["errors"]) >= 1
        # Каждая ошибка имеет type, count, examples, hint.
        for e in payload["errors"]:
            assert "type" in e
            assert "count" in e
            assert "examples" in e
            assert "hint" in e
            assert len(e["examples"]) <= 10


# ───────────────────────────── format_stderr_report ─────────────────────────────

class TestFormatStderrReport:
    def test_valid_short_ok_message(self, tiny_test_gap_keys):
        sub = _load("submission_valid_tiny.csv")
        result = validate_submission(sub, tiny_test_gap_keys)
        text = format_stderr_report(result)
        assert "[ok]" in text
        assert "submission valid" in text
        assert "key_set_match=True" in text

    def test_invalid_message_includes_type_count_hint(self, tiny_test_gap_keys):
        sub = _load("submission_invalid_extra.csv")
        result = validate_submission(sub, tiny_test_gap_keys)
        text = format_stderr_report(result)
        assert "[error]" in text
        assert "extra_key" in text
        assert "hint:" in text


# ───────────────────────────── constants ─────────────────────────────

class TestConstants:
    def test_submission_header_exact(self):
        assert SUBMISSION_HEADER == ("anon_polygon_id", "date", "primary_ndvi_pred")

    def test_expected_row_count_matches_ed_part(self):
        assert EXPECTED_ROW_COUNT == 3_112
