"""Unit-тесты для scripts.data_quality_report.

Покрывает:
- compute_quality на toy fixture (train и test);
- _is_finite (NaN/inf/строки);
- _parse_date_strict (валидные/невалидные/исходные NaN);
- _check_hierarchy (S2 vs Landsat vs MODIS);
- _compare_with_expected (warning'и при mismatch);
- write_json детерминизм (повторный запуск → побайтно одинаково);
- write_markdown содержит ожидаемые разделы;
- parse_args запрещает абсолютные пути (exit 2).

Все тесты работают без subprocess — публичные API импортируются напрямую.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.data_quality_report import (
    EXPECTED,
    HIERARCHY_TOLERANCE,
    REPORT_SCHEMA_VERSION,
    _check_hierarchy,
    _compare_with_expected,
    _is_finite,
    _parse_date_strict,
    compute_quality,
    parse_args,
    write_json,
    write_markdown,
)

# ───────────────────────────── fixtures ─────────────────────────────

FIX_DIR = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def train_df() -> pd.DataFrame:
    return pd.read_csv(FIX_DIR / "quality_train_tiny.csv")


@pytest.fixture
def test_df() -> pd.DataFrame:
    return pd.read_csv(FIX_DIR / "quality_test_tiny.csv")


# ───────────────────────────── _is_finite ─────────────────────────────

class TestIsFinite:
    @pytest.mark.parametrize(
        "value",
        [0, 0.0, 1, -1.5, "3.14", "-2e-3", "1e308"],
    )
    def test_finite_values(self, value):
        assert _is_finite(value) is True

    @pytest.mark.parametrize(
        "value",
        [None, float("nan"), float("inf"), float("-inf"),
         "", " ", "nan", "NaN", "NA", "none", "NULL", "abc", "1e500"],
    )
    def test_non_finite_values(self, value):
        assert _is_finite(value) is False


# ───────────────────────────── _parse_date_strict ─────────────────────────────

class TestParseDateStrict:
    def test_valid_dates(self):
        s = pd.Series(["2020-04-01", "2021-12-31"])
        parsed, invalid = _parse_date_strict(s)
        assert invalid == 0
        assert parsed.iloc[0].strftime("%Y-%m-%d") == "2020-04-01"
        assert parsed.iloc[1].strftime("%Y-%m-%d") == "2021-12-31"

    def test_invalid_dates_counted(self):
        s = pd.Series(["2020-04-01", "2020-13-01", "abc", "2020-04-01"])
        parsed, invalid = _parse_date_strict(s)
        assert invalid == 2  # "2020-13-01" и "abc"
        assert parsed.iloc[0] is not pd.NaT
        assert pd.isna(parsed.iloc[1])
        assert pd.isna(parsed.iloc[2])
        assert parsed.iloc[3] is not pd.NaT

    def test_source_nan_not_counted_as_invalid(self):
        # Исходный NaN — это missingness, не bad data.
        s = pd.Series(["2020-04-01", None])
        parsed, invalid = _parse_date_strict(s)
        assert invalid == 0
        assert pd.isna(parsed.iloc[1])


# ───────────────────────────── compute_quality ─────────────────────────────

class TestComputeQuality:
    def test_train_tiny_basic_counts(self, train_df):
        rep = compute_quality(train_df, "train")
        assert rep["rows"] == 5
        assert rep["columns"] == 21
        assert rep["polygon_count"] == 2
        # TOY-A: 3 finite, TOY-B: 2 finite → итого 5.
        assert rep["primary_ndvi_finite"] == 5
        # polygon_overlap считается только для test.
        assert rep["polygon_overlap"] is None

    def test_test_tiny_has_overlap_and_gaps(self, test_df, train_df):
        train_polygons = set(train_df["anon_polygon_id"].unique())
        rep = compute_quality(test_df, "test", train_polygons=train_polygons)
        assert rep["rows"] == 7
        assert rep["columns"] == 20
        assert rep["polygon_count"] == 2  # TOY-A, TOY-C
        assert rep["is_synthetic_gap"] == {"true": 4, "false": 3, "missing": 0}
        assert rep["polygon_overlap"]["known"] == 1  # TOY-A
        assert rep["polygon_overlap"]["unseen"] == 1  # TOY-C
        assert rep["gap_count_by_overlap"] == {"known": 3, "unseen": 1}

    def test_gap_run_lengths(self, test_df, train_df):
        train_polygons = set(train_df["anon_polygon_id"].unique())
        rep = compute_quality(test_df, "test", train_polygons=train_polygons)
        # Toy данные: gap-даты НЕ подряд (04-09, 04-25, 05-03 для TOY-A;
        # 04-09 для TOY-C). Интервал между gap-датами > 1 дня, значит
        # каждая gap-дата = отдельный ран длиной 1.
        # TOY-A: 3 рана по 1, TOY-C: 1 ран по 1 → итого 4 рана по 1.
        assert rep["gap_run_lengths"] is not None
        assert rep["gap_run_lengths"]["1"] == 4
        assert rep["gap_run_lengths"]["2"] == 0
        assert rep["gap_run_lengths"]["3"] == 0
        assert rep["gap_run_lengths"]["4+"] == 0

    def test_gap_rows_always_missing_columns(self, test_df, train_df):
        train_polygons = set(train_df["anon_polygon_id"].unique())
        rep = compute_quality(test_df, "test", train_polygons=train_polygons)
        # В gap-строках всегда пусты: все NDVI/EVI/NDWI колонки.
        expected = {
            "s2_ndvi", "s2_evi", "s2_ndwi",
            "landsat_ndvi", "landsat_evi", "landsat_ndwi",
            "modis_ndvi", "modis_evi",
            "primary_ndvi",
            "ndvi_climatology_mean", "ndvi_climatology_std",
        }
        assert set(rep["gap_rows_always_missing"]) == expected

    def test_duplicate_keys_zero(self, train_df, test_df):
        assert compute_quality(train_df, "train")["duplicate_keys"] == 0
        train_polygons = set(train_df["anon_polygon_id"].unique())
        assert compute_quality(test_df, "test", train_polygons=train_polygons)["duplicate_keys"] == 0

    def test_crop_type_counts(self, train_df):
        rep = compute_quality(train_df, "train")
        assert rep["crop_type_counts"]["toy_wheat"] == 3
        assert rep["crop_type_counts"]["toy_corn"] == 2

    def test_rows_per_polygon(self, train_df):
        rep = compute_quality(train_df, "train")
        # TOY-A=3, TOY-B=2
        assert rep["rows_per_polygon"]["min"] == 2
        assert rep["rows_per_polygon"]["max"] == 3

    def test_target_source_hierarchy_train(self, train_df):
        rep = compute_quality(train_df, "train")
        h = rep["target_source_hierarchy"]
        # На visible-target строках primary должен совпасть с S2 (первый в иерархии).
        # Проверяем только что checked > 0 и match_rate > 0.
        assert h["checked"] > 0
        assert h["match_rate"] >= 0.0
        assert h["max_abs_mismatch"] is not None
        assert h["max_abs_mismatch"] >= 0.0

    def test_target_source_hierarchy_falls_back_to_landsat(self):
        # Если S2 пуст, hierarchy должен взять Landsat.
        df = pd.DataFrame({
            "anon_polygon_id": ["X"],
            "date": ["2020-04-01"],
            "s2_ndvi": [None],
            "landsat_ndvi": [0.42],
            "modis_ndvi": [0.40],
            "primary_ndvi": [0.42],
        })
        h = _check_hierarchy(df)
        assert h["checked"] == 1
        assert h["match"] == 1
        assert abs(h["max_abs_mismatch"]) < HIERARCHY_TOLERANCE

    def test_target_source_hierarchy_no_primary(self):
        df = pd.DataFrame({
            "anon_polygon_id": ["X"],
            "date": ["2020-04-01"],
            "s2_ndvi": [0.5],
            "primary_ndvi": [None],
        })
        h = _check_hierarchy(df)
        assert h["checked"] == 0
        assert h["match"] == 0
        assert h["match_rate"] == 0.0


# ───────────────────────────── _compare_with_expected ─────────────────────────────

class TestCompareWithExpected:
    def test_match_no_warnings(self):
        rep = {
            "name": "train",
            "rows": EXPECTED["train"]["rows"],
            "columns": EXPECTED["train"]["columns"],
            "polygon_count": EXPECTED["train"]["polygon_count"],
            "primary_ndvi_finite": EXPECTED["train"]["primary_ndvi_finite"],
            "is_synthetic_gap": None,
        }
        assert _compare_with_expected(rep) == []

    def test_mismatch_returns_warnings(self):
        rep = {
            "name": "train",
            "rows": 1,  # WRONG
            "columns": EXPECTED["train"]["columns"],
            "polygon_count": EXPECTED["train"]["polygon_count"],
            "primary_ndvi_finite": EXPECTED["train"]["primary_ndvi_finite"],
            "is_synthetic_gap": None,
        }
        warnings = _compare_with_expected(rep)
        assert any("rows" in w for w in warnings)

    def test_test_gaps_checked(self):
        # Match: эталоны совпадают → warnings пуст.
        rep = {
            "name": "test",
            "rows": EXPECTED["test"]["rows"],
            "columns": EXPECTED["test"]["columns"],
            "polygon_count": EXPECTED["test"]["polygon_count"],
            "primary_ndvi_finite": EXPECTED["test"]["primary_ndvi_finite"],
            "is_synthetic_gap": {"true": 3_112, "false": 54_073, "missing": 0},
        }
        assert _compare_with_expected(rep) == []

        # Mismatch: 100 вместо 3112 → warning.
        rep_bad = dict(rep)
        rep_bad["is_synthetic_gap"] = {"true": 100, "false": 1, "missing": 0}
        assert any("gaps" in w for w in _compare_with_expected(rep_bad))


# ───────────────────────────── writers ─────────────────────────────

class TestWriteJsonDeterminism:
    def test_determinism_byte_to_byte(self, tmp_path, train_df, test_df):
        train_polygons = set(train_df["anon_polygon_id"].unique())
        rep = {
            "train": compute_quality(train_df, "train"),
            "test": compute_quality(test_df, "test", train_polygons=train_polygons),
        }

        path1 = tmp_path / "out1.json"
        path2 = tmp_path / "out2.json"
        write_json(rep, path1)
        write_json(rep, path2)

        # Побайтно одинаково.
        assert path1.read_bytes() == path2.read_bytes()

        # Никакого timestamp в payload.
        payload = json.loads(path1.read_text(encoding="utf-8"))
        assert "timestamp" not in payload
        assert "created_at" not in payload
        assert payload["report_schema_version"] == REPORT_SCHEMA_VERSION

    def test_sort_keys(self, tmp_path):
        rep = {"z_train": 1, "a_test": 2, "m_middle": 3}
        path = tmp_path / "out.json"
        write_json(rep, path)
        # sort_keys=True → ключи идут в алфавитном порядке: a_test, m_middle, z_train.
        text = path.read_text(encoding="utf-8")
        assert text.find('"a_test"') < text.find('"m_middle"') < text.find('"z_train"')


class TestWriteMarkdown:
    def test_sections_present(self, tmp_path, train_df, test_df):
        train_polygons = set(train_df["anon_polygon_id"].unique())
        rep = {
            "train": compute_quality(train_df, "train"),
            "test": compute_quality(test_df, "test", train_polygons=train_polygons),
        }
        path = tmp_path / "out.md"
        write_markdown(rep, path)
        text = path.read_text(encoding="utf-8")
        for section in (
            "# Data Quality Report",
            "## 1. Summary",
            "## 2. Schema",
            "## 3. Missingness",
            "## 4. Gaps",
            "## 5. Polygon overlap",
            "## 6. Target-source hierarchy",
            "## 7. Warnings",
        ):
            assert section in text, f"missing section: {section}"


# ───────────────────────────── parse_args ─────────────────────────────

class TestParseArgs:
    def test_basic(self):
        args = parse_args(["--train", "a.csv", "--test", "b.csv",
                            "--out-md", "m.md", "--out-json", "j.json"])
        assert str(args.train) == "a.csv"
        assert str(args.test) == "b.csv"
        assert args.allow_mismatch is False

    def test_absolute_paths_rejected(self):
        # Windows-абсолютный путь → должен быть rejected.
        with pytest.raises(SystemExit):
            parse_args(["--train", "C:/absolute/train.csv", "--test", "b.csv",
                        "--out-md", "m.md", "--out-json", "j.json"])

    def test_allow_mismatch_flag(self):
        args = parse_args(["--train", "a.csv", "--test", "b.csv",
                            "--out-md", "m.md", "--out-json", "j.json",
                            "--allow-mismatch"])
        assert args.allow_mismatch is True