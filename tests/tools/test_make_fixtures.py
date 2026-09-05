"""Unit-тесты для scripts.make_test_fixtures.

Покрывает:
- select_known_polygon / select_unseen_polygon (алфавитный порядок);
- find_gap_run (поиск последовательных gap-ранов);
- build_train_tiny / build_test_tiny (лимит строк, сортировка);
- build_submission_valid (только gap-строки);
- build_submission_invalid_duplicate (1 дубликат);
- make_synthetic_gap_run / make_synthetic_one_sided_context (TOY-* префикс,
  нет primary_ndvi для gaps);
- build_fixtures (целостность manifest);
- sha256_file и write_manifest (детерминизм, без timestamp);
- CLI --help exit 0.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts.make_test_fixtures import (
    DEFAULT_PREDICTION,
    MANIFEST_SCHEMA_VERSION,
    SCRIPT_VERSION,
    SUBMISSION_HEADER,
    TOY_PREFIX,
    build_fixtures,
    build_submission_invalid_duplicate,
    build_submission_valid,
    build_test_tiny,
    build_train_tiny,
    find_gap_run,
    make_synthetic_gap_run,
    make_synthetic_one_sided_context,
    select_known_polygon,
    select_unseen_polygon,
    sha256_file,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[2]


# ───────────────────────────── fixtures ─────────────────────────────

@pytest.fixture(scope="module")
def train_df() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "train_dataset.csv")


@pytest.fixture(scope="module")
def test_df() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "test_data.csv")


@pytest.fixture
def small_train() -> pd.DataFrame:
    """Маленький train CSV для быстрых unit-тестов."""
    return pd.DataFrame({
        "anon_polygon_id": ["P1"] * 5 + ["P2"] * 3,
        "date": ["2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04", "2020-04-05",
                 "2020-04-01", "2020-04-02", "2020-04-03"],
        "primary_ndvi": [0.5] * 8,
    })


@pytest.fixture
def small_test() -> pd.DataFrame:
    """Маленький test CSV с known/unseen polygon-ами и gap-ами."""
    rows: list[dict[str, object]] = []
    # P1 (known): 5 строк, 2 gap-а (04-03, 04-04 = рана длиной 2).
    for d, gap in [
        ("2020-04-01", False),
        ("2020-04-02", False),
        ("2020-04-03", True),
        ("2020-04-04", True),
        ("2020-04-05", False),
    ]:
        rows.append({"anon_polygon_id": "P1", "date": d,
                     "is_synthetic_gap": gap, "primary_ndvi": 0.5 if not gap else None})
    # P3 (unseen): 3 строки, 1 gap.
    for d, gap in [("2020-04-01", False), ("2020-04-02", True), ("2020-04-03", False)]:
        rows.append({"anon_polygon_id": "P3", "date": d,
                     "is_synthetic_gap": gap, "primary_ndvi": 0.5 if not gap else None})
    return pd.DataFrame(rows)


# ───────────────────────────── select_known / select_unseen ─────────────────────────────

class TestSelectors:
    def test_known_first_alphabetically(self):
        train_polys = {"AOI-A", "AOI-B", "AOI-C"}
        test_polys = {"AOI-B", "AOI-X", "AOI-Z"}
        assert select_known_polygon(train_polys, test_polys) == "AOI-B"

    def test_known_none_if_no_intersection(self):
        assert select_known_polygon({"A", "B"}, {"C", "D"}) is None

    def test_unseen_first_alphabetically(self):
        train_polys = {"AOI-A", "AOI-B"}
        test_polys = {"AOI-A", "AOI-B", "AOI-Z", "AOI-X"}
        # Exclude train → {AOI-Z, AOI-X}; first alphabetically = AOI-X.
        assert select_unseen_polygon(train_polys, test_polys) == "AOI-X"

    def test_unseen_none_if_all_known(self):
        assert select_unseen_polygon({"A", "B"}, {"A", "B"}) is None


# ───────────────────────────── find_gap_run ─────────────────────────────

class TestFindGapRun:
    def test_finds_run_of_two(self, small_test):
        run = find_gap_run(small_test, "P1", min_len=2, max_len=4)
        assert run is not None
        assert len(run) == 2
        assert run[0].strftime("%Y-%m-%d") == "2020-04-03"

    def test_no_run_when_min_len_too_high(self, small_test):
        # P3 имеет только 1 gap → не найдёт с min_len=2.
        assert find_gap_run(small_test, "P3", min_len=2, max_len=4) is None

    def test_no_run_for_unknown_polygon(self, small_test):
        assert find_gap_run(small_test, "MISSING", min_len=2, max_len=4) is None

    def test_no_is_synthetic_gap_column(self):
        df = pd.DataFrame({"anon_polygon_id": ["P1"], "date": ["2020-04-01"]})
        assert find_gap_run(df, "P1") is None


# ───────────────────────────── build_train_tiny / build_test_tiny ─────────────────────────────

class TestBuildTiny:
    def test_train_tiny_limits_rows(self, train_df):
        tiny = build_train_tiny(train_df, "AOI-0001", max_rows=10)
        assert len(tiny) <= 10

    def test_train_tiny_sorted_by_date(self, train_df):
        tiny = build_train_tiny(train_df, "AOI-0001", max_rows=20)
        dates = pd.to_datetime(tiny["date"], errors="coerce")
        assert dates.is_monotonic_increasing

    def test_test_tiny_only_polygon_rows(self, train_df):
        # Используем train_df как test_df для скорости (только проверяем фильтрацию).
        tiny = build_test_tiny(train_df, "AOI-0001", max_rows=5)
        assert (tiny["anon_polygon_id"] == "AOI-0001").all()
        assert len(tiny) <= 5


# ───────────────────────────── build_submission_valid ─────────────────────────────

class TestBuildSubmissionValid:
    def test_only_gap_keys(self, small_test):
        sub = build_submission_valid(small_test)
        # P1: 2 gap, P3: 1 gap → 3 строки.
        assert len(sub) == 3
        assert list(sub.columns) == list(SUBMISSION_HEADER)
        # Все key соответствуют gap-строкам test.
        gap_keys = {
            ("P1", "2020-04-03"), ("P1", "2020-04-04"), ("P3", "2020-04-02")
        }
        sub_keys = {(r["anon_polygon_id"], r["date"]) for _, r in sub.iterrows()}
        assert sub_keys == gap_keys

    def test_default_prediction(self, small_test):
        sub = build_submission_valid(small_test)
        assert (sub["primary_ndvi_pred"] == DEFAULT_PREDICTION).all()

    def test_custom_prediction(self, small_test):
        sub = build_submission_valid(small_test, prediction=0.7)
        assert (sub["primary_ndvi_pred"] == 0.7).all()


# ───────────────────────────── build_submission_invalid_duplicate ─────────────────────────────

class TestBuildSubmissionInvalidDup:
    def test_one_duplicate(self, small_test):
        valid = build_submission_valid(small_test)
        invalid = build_submission_invalid_duplicate(valid)
        assert len(invalid) == len(valid) + 1
        # Первая строка дублируется.
        assert invalid.iloc[0].equals(invalid.iloc[1])

    def test_empty_raises(self):
        empty = pd.DataFrame(columns=list(SUBMISSION_HEADER))
        with pytest.raises(ValueError):
            build_submission_invalid_duplicate(empty)


# ───────────────────────────── make_synthetic_* ─────────────────────────────

class TestMakeSynthetic:
    def test_gap_run_toy_prefix(self):
        run = make_synthetic_gap_run(f"{TOY_PREFIX}RUN-X", pd.Timestamp("2020-04-09"), 3)
        assert len(run) == 3
        assert (run["anon_polygon_id"] == f"{TOY_PREFIX}RUN-X").all()
        # Все NDVI пустые (нет ground truth для gaps).
        for col in ["s2_ndvi", "landsat_ndvi", "modis_ndvi", "primary_ndvi"]:
            assert run[col].isna().all()
        assert run["is_synthetic_gap"].all()
        # Даты последовательные.
        dates = pd.to_datetime(run["date"])
        assert (dates.diff().dropna().dt.days == 1).all()

    def test_one_sided_context_has_s2_only(self):
        ctx = make_synthetic_one_sided_context(
            f"{TOY_PREFIX}CTX-X", pd.Timestamp("2020-04-09"),
        )
        assert len(ctx) == 1
        # S2 present, Landsat/MODIS пустые.
        assert pd.notna(ctx["s2_ndvi"].iloc[0])
        assert pd.isna(ctx["landsat_ndvi"].iloc[0])
        assert pd.isna(ctx["modis_ndvi"].iloc[0])
        # primary_ndvi = s2_ndvi (иерархия S2 первая).
        assert ctx["primary_ndvi"].iloc[0] == ctx["s2_ndvi"].iloc[0]
        # Не gap.
        assert ctx["is_synthetic_gap"].iloc[0] == False  # noqa: E712


# ───────────────────────────── build_fixtures (integration) ─────────────────────────────

class TestBuildFixtures:
    def test_basic_invariants(self, train_df, test_df):
        result = build_fixtures(train_df, test_df, seed=42)
        # train_tiny: <= 60 строк, 21 колонка.
        assert len(result.train_tiny) > 0
        assert len(result.train_tiny) <= 60
        assert len(result.train_tiny.columns) == 21
        # test_tiny: >= 60 строк (есть unseen), 20 колонок.
        assert len(result.test_tiny.columns) == 20
        # Submission valid: только gap-строки test_tiny.
        assert len(result.submission_valid) == int(
            (result.test_tiny["is_synthetic_gap"] == True).sum()  # noqa: E712
        )
        # Submission invalid: +1 строка.
        assert len(result.submission_invalid_duplicate) == len(result.submission_valid) + 1
        # Manifest: все поля.
        assert "selected_keys" in result.manifest
        assert "files" in result.manifest
        assert "design_assumptions" in result.manifest
        assert result.manifest["design_assumptions_requires_ml_confirmation"] is True

    def test_no_duplicate_keys(self, train_df, test_df):
        result = build_fixtures(train_df, test_df, seed=42)
        for name, df in (
            ("train_tiny", result.train_tiny),
            ("test_tiny", result.test_tiny),
            ("submission_valid", result.submission_valid),
        ):
            keys = list(zip(
                df["anon_polygon_id"].astype(str),
                df["date"].astype(str),
                strict=True,
            ))
            assert len(keys) == len(set(keys)), f"duplicate keys in {name}"

    def test_synthetic_rows_have_toy_prefix(self, train_df, test_df):
        result = build_fixtures(train_df, test_df, seed=42)
        toy_rows = result.test_tiny[
            result.test_tiny["anon_polygon_id"].astype(str).str.startswith(TOY_PREFIX)
        ]
        assert len(toy_rows) > 0, "expected at least some TOY-* rows"
        # Synthetic gap rows не имеют primary_ndvi.
        toy_gap = toy_rows[toy_rows["is_synthetic_gap"] == True]  # noqa: E712
        if not toy_gap.empty:
            assert toy_gap["primary_ndvi"].isna().all()

    def test_determinism_same_seed_same_output(self, train_df, test_df):
        r1 = build_fixtures(train_df, test_df, seed=42)
        r2 = build_fixtures(train_df, test_df, seed=42)
        # Совпадают все DataFrame.
        pd.testing.assert_frame_equal(r1.train_tiny, r2.train_tiny)
        pd.testing.assert_frame_equal(r1.test_tiny, r2.test_tiny)
        pd.testing.assert_frame_equal(r1.submission_valid, r2.submission_valid)
        pd.testing.assert_frame_equal(
            r1.submission_invalid_duplicate, r2.submission_invalid_duplicate,
        )
        # Manifest совпадает (без source_fingerprints — они заполняются в main).
        for k in ("selection_rule", "selected_keys", "files", "design_assumptions"):
            assert r1.manifest[k] == r2.manifest[k]

    def test_different_seeds_yield_different_synthetic(self, train_df, test_df):
        r1 = build_fixtures(train_df, test_df, seed=42)
        r2 = build_fixtures(train_df, test_df, seed=43)
        # Real части одинаковы, synthetic — разные (era5 values зависят от seed).
        pd.testing.assert_frame_equal(r1.train_tiny, r2.train_tiny)
        # Хотя бы один numeric столбец в synthetic отличается.
        toy1 = r1.test_tiny[
            r1.test_tiny["anon_polygon_id"].astype(str).str.startswith(TOY_PREFIX)
        ]
        toy2 = r2.test_tiny[
            r2.test_tiny["anon_polygon_id"].astype(str).str.startswith(TOY_PREFIX)
        ]
        if not toy1.empty and not toy2.empty:
            # Если оба не пусты, проверим что хотя бы одно значение отличается.
            assert not toy1["era5_temp_c"].equals(toy2["era5_temp_c"]) \
                or not toy1["era5_precip_mm"].equals(toy2["era5_precip_mm"])


# ───────────────────────────── sha256_file / write_manifest ─────────────────────────────

class TestSha256AndManifest:
    def test_sha256_stable(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("hello", encoding="utf-8")
        h1 = sha256_file(f)
        h2 = sha256_file(f)
        assert h1 == h2
        assert len(h1) == 64

    def test_manifest_deterministic_no_timestamp(self, tmp_path):
        manifest = {
            "selected_keys": {"a": 1},
            "files": {"x": 2},
        }
        path1 = tmp_path / "m1.json"
        path2 = tmp_path / "m2.json"
        write_manifest(manifest, path1)
        write_manifest(manifest, path2)
        assert path1.read_bytes() == path2.read_bytes()
        payload = json.loads(path1.read_text(encoding="utf-8"))
        assert "timestamp" not in payload
        assert "created_at" not in payload
        assert payload["manifest_schema_version"] == MANIFEST_SCHEMA_VERSION
        assert payload["script_version"] == SCRIPT_VERSION


# ───────────────────────────── CLI --help ─────────────────────────────

class TestCliHelp:
    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.make_test_fixtures", "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--out" in result.stdout
        assert "--manifest" in result.stdout
        assert "--train-source" in result.stdout
        assert "--test-source" in result.stdout
        assert "--seed" in result.stdout