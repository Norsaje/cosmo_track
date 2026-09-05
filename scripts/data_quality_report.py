"""Data quality report for train/test CSVs.

Читает train и test, считает метрики качества данных и пишет отчёты в
Markdown и JSON. Ничего не пишет в исходные CSV — это страховка от
непреднамеренного повреждения данных.

Схема результата версионируется константой REPORT_SCHEMA_VERSION.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

import pandas as pd

# Версия схемы отчёта. Любое изменение структуры JSON → bump major.
REPORT_SCHEMA_VERSION = "1.0.0"
SCRIPT_VERSION = "1.0.0"

# Иерархия первичных источников primary_ndvi (от лучшего к худшему).
# Используется ТОЛЬКО для отчётной проверки «совпадает ли primary_ndvi
# с первым доступным источником», модель не строится.
SOURCE_HIERARCHY = ("s2_ndvi", "landsat_ndvi", "modis_ndvi")
HIERARCHY_TOLERANCE = 1e-10

# Целевые эталоны набора данных.
# При несовпадении скрипт не «чинит» данные, а падает с понятной ошибкой.
EXPECTED: dict[str, dict[str, int]] = {
    "train": {
        "rows": 99_955,
        "columns": 21,
        "polygon_count": 39,
        "primary_ndvi_finite": 30_520,
    },
    "test": {
        "rows": 57_185,
        "columns": 20,
        "polygon_count": 78,
        "primary_ndvi_finite": 17_641,
        "gaps": 3_112,
    },
}


# ───────────────────────────── helpers ─────────────────────────────

def _is_finite(value: Any) -> bool:
    """Является ли значение конечным числом.

    NaN/None/пустая строка → False. inf/-inf → False.
    Используем именно math.isfinite для строгой проверки.
    """
    if value is None:
        return False
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, int):
        return True
    if isinstance(value, str):
        s = value.strip()
        if not s or s.lower() in {"nan", "na", "none", "null"}:
            return False
        try:
            return math.isfinite(float(s))
        except ValueError:
            return False
    return False


def _parse_date_strict(series: pd.Series) -> tuple[pd.Series, int]:
    """Строгий парсинг ISO-дат (YYYY-MM-DD).

    Возвращает (datetime_series, invalid_count). invalid_count — число
    элементов, которые не распарсились. NaT считается невалидным.
    """
    parsed = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
    invalid = int(parsed.isna().sum() - series.isna().sum())
    # Не считаем исходные NaN за invalid (это missingness, не bad data).
    invalid = max(invalid, 0)
    return parsed, invalid


def _gap_run_lengths(gaps_by_polygon: dict[str, list[pd.Timestamp]]) -> dict[str, int]:
    """Распределение длин последовательных gap-ранов по полигону.

    gaps_by_polygon[pid] — отсортированный список дат, где есть gap.
    Соседние даты считаются «раном», если разница ровно 1 день.
    """
    buckets: Counter[str, int] = Counter()
    for dates in gaps_by_polygon.values():
        if not dates:
            continue
        run = 1
        prev = dates[0]
        for cur in dates[1:]:
            if (cur - prev).days == 1:
                run += 1
            else:
                buckets[_run_bucket(run)] += 1
                run = 1
            prev = cur
        buckets[_run_bucket(run)] += 1
    # Заполняем пропуски единицами, чтобы ключи всегда были в JSON.
    out = {b: buckets.get(b, 0) for b in ("1", "2", "3", "4+")}
    return out


def _run_bucket(length: int) -> str:
    if length <= 3:
        return str(length)
    return "4+"


# ───────────────────────────── core ─────────────────────────────

def compute_quality(
    df: pd.DataFrame,
    name: str,
    *,
    train_polygons: set[str] | None = None,
) -> dict[str, Any]:
    """Считает quality-метрики для одного DataFrame.

    Параметры:
        df: уже распарсенный CSV (через pandas.read_csv).
        name: "train" или "test" — влияет на ожидаемые колонки и эталон.
        train_polygons: для test — set полигонов train, чтобы посчитать
            known/unseen overlap. Для train можно не передавать.

    Возвращает словарь, сериализуемый в JSON. Никакого timestamp —
    иначе нарушается детерминизм.
    """
    if name not in EXPECTED:
        raise ValueError(f"unknown dataset name: {name!r}")

    # 1) Схема: колонки и dtype.
    schema = [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns]

    # 2) Базовые counts.
    rows = len(df)
    cols = len(df.columns)
    polygon_col = df["anon_polygon_id"].astype(str)
    polygons = set(polygon_col.unique())

    # 3) Duplicate keys.
    duplicate_keys = int(df.duplicated(subset=["anon_polygon_id", "date"]).sum())

    # 4) Даты: строгий парсинг + min/max.
    parsed_dates, invalid_dates = _parse_date_strict(df["date"])
    valid_dates = parsed_dates.dropna()
    date_min = valid_dates.min().strftime("%Y-%m-%d") if len(valid_dates) else None
    date_max = valid_dates.max().strftime("%Y-%m-%d") if len(valid_dates) else None

    # 5) Missingness по каждой колонке + finite для numeric.
    missingness: list[dict[str, Any]] = []
    for col in df.columns:
        total = len(df)
        missing = int(df[col].isna().sum())
        finite = 0
        if df[col].dtype.kind in "fiub":
            finite = int(df[col].apply(_is_finite).sum())
        else:
            # Для object/string колонок считаем «finite» как непустое значение,
            # парсящееся в число (если парсится).
            finite = int(df[col].apply(_is_finite).sum())
        missingness.append(
            {
                "column": col,
                "total": total,
                "missing": missing,
                "finite": finite,
                "missing_rate": missing / total if total else 0.0,
                "finite_rate": finite / total if total else 0.0,
            }
        )

    # 6) Finite primary_ndvi.
    primary_ndvi_finite = 0
    if "primary_ndvi" in df.columns:
        primary_ndvi_finite = int(df["primary_ndvi"].apply(_is_finite).sum())

    # 7) rows per polygon.
    rows_per_polygon_series = polygon_col.value_counts()
    rows_per_polygon = {
        "min": int(rows_per_polygon_series.min()) if len(rows_per_polygon_series) else 0,
        "median": float(rows_per_polygon_series.median()) if len(rows_per_polygon_series) else 0.0,
        "max": int(rows_per_polygon_series.max()) if len(rows_per_polygon_series) else 0,
    }

    # 8) Crop types → counts.
    crop_type_counts: dict[str, int] = {}
    if "crop_type" in df.columns:
        crop_type_counts = {
            str(k): int(v) for k, v in df["crop_type"].value_counts(dropna=True).items()
        }

    # 9) years per polygon + date range per polygon.
    years_per_polygon: dict[str, dict[str, Any]] = {}
    if "year" in df.columns:
        tmp = df[["anon_polygon_id", "year", "date"]].copy()
        tmp["date_parsed"] = parsed_dates
        for pid, sub in tmp.groupby("anon_polygon_id"):
            years = sorted({int(y) for y in sub["year"].dropna().unique()})
            d_valid = sub["date_parsed"].dropna()
            d_min = d_valid.min().strftime("%Y-%m-%d") if len(d_valid) else None
            d_max = d_valid.max().strftime("%Y-%m-%d") if len(d_valid) else None
            years_per_polygon[str(pid)] = {
                "years": years,
                "date_range": [d_min, d_max],
            }

    # 10) test-only: is_synthetic_gap counts.
    is_synthetic_gap: dict[str, int] | None = None
    gaps_total = 0
    gap_rows_always_missing: list[str] = []
    if "is_synthetic_gap" in df.columns:
        gap_bool = df["is_synthetic_gap"]
        # Приводим к bool аккуратно: True / False / NaN.
        gap_true = int((gap_bool == True).sum())  # noqa: E712 — pandas nullable bool
        gap_false = int((gap_bool == False).sum())  # noqa: E712
        gap_missing = int(gap_bool.isna().sum())
        is_synthetic_gap = {"true": gap_true, "false": gap_false, "missing": gap_missing}
        gaps_total = gap_true

        # Какие колонки всегда missing в gap-строках.
        gap_df = df[gap_bool == True]  # noqa: E712
        if len(gap_df):
            for col in df.columns:
                if col in {"anon_polygon_id", "date", "is_synthetic_gap"}:
                    continue
                if gap_df[col].isna().all():
                    gap_rows_always_missing.append(col)

    # 11) Gap-run length distribution (только если есть is_synthetic_gap).
    gap_run_lengths: dict[str, int] | None = None
    if is_synthetic_gap is not None and gaps_total:
        tmp = df[["anon_polygon_id", "date", "is_synthetic_gap"]].copy()
        tmp["date_parsed"] = parsed_dates
        # Собираем список дат per polygon через pair-iteration, чтобы не
        # зависеть от того, что pandas возвращает после groupby (multi-index,
        # нестабильный порядок). Сортируем в Python по значению.
        per_poly: dict[str, list[pd.Timestamp]] = {}
        for i in range(len(tmp)):
            row = tmp.iloc[i]
            if bool(row["is_synthetic_gap"]) is not True:  # noqa: E712
                continue
            d = row["date_parsed"]
            if pd.isna(d):
                continue
            per_poly.setdefault(str(row["anon_polygon_id"]), []).append(d)
        for pid, dates in per_poly.items():
            per_poly[pid] = sorted(dates)
        gap_run_lengths = _gap_run_lengths(per_poly)

    # 12) Known vs unseen polygons (для test).
    polygon_overlap: dict[str, int] | None = None
    gap_count_by_overlap: dict[str, int] | None = None
    if name == "test" and train_polygons is not None:
        known = polygons & train_polygons
        unseen = polygons - train_polygons
        polygon_overlap = {
            "known": len(known),
            "unseen": len(unseen),
            "train_polygon_count": len(train_polygons),
        }
        if "is_synthetic_gap" in df.columns:
            gap_df = df[df["is_synthetic_gap"] == True]  # noqa: E712
            gap_count_by_overlap = {
                "known": int(gap_df["anon_polygon_id"].isin(known).sum()),
                "unseen": int(gap_df["anon_polygon_id"].isin(unseen).sum()),
            }

    # 13) Target-source hierarchy check (отчётная проверка).
    target_source_hierarchy = _check_hierarchy(df)

    report: dict[str, Any] = {
        "name": name,
        "rows": rows,
        "columns": cols,
        "schema": schema,
        "duplicate_keys": duplicate_keys,
        "invalid_dates": invalid_dates,
        "date_min": date_min,
        "date_max": date_max,
        "polygon_count": len(polygons),
        "polygons": sorted(polygons),
        "crop_type_counts": crop_type_counts,
        "missingness": missingness,
        "primary_ndvi_finite": primary_ndvi_finite,
        "rows_per_polygon": rows_per_polygon,
        "years_per_polygon": years_per_polygon,
        "is_synthetic_gap": is_synthetic_gap,
        "gap_run_lengths": gap_run_lengths,
        "gap_rows_always_missing": gap_rows_always_missing,
        "polygon_overlap": polygon_overlap,
        "gap_count_by_overlap": gap_count_by_overlap,
        "target_source_hierarchy": target_source_hierarchy,
        "warnings": [],
    }
    return report


def _check_hierarchy(df: pd.DataFrame) -> dict[str, Any]:
    """На видимых target-строках проверяет, совпадает ли primary_ndvi
    с первым доступным источником по иерархии S2 → Landsat → MODIS.

    Только отчётная проверка; модель не строится. tolerance = 1e-10.

    Реализация намеренно на чистых списках (не Series), чтобы избежать
    проблем с разными типами индекса в pandas при смешанных dtype.
    """
    if "primary_ndvi" not in df.columns:
        return {"checked": 0, "match": 0, "match_rate": 0.0, "max_abs_mismatch": None}

    n = len(df)
    primary_vals: list[Any] = [df["primary_ndvi"].iloc[i] for i in range(n)]
    primary_finite_mask = [_is_finite(v) for v in primary_vals]

    # Для каждой строки выбираем первый доступный источник по иерархии.
    chosen_vals: list[float | None] = [None] * n
    for i in range(n):
        if not primary_finite_mask[i]:
            continue
        for col in SOURCE_HIERARCHY:
            if col not in df.columns:
                continue
            v = df[col].iloc[i]
            if _is_finite(v):
                chosen_vals[i] = float(v)
                break

    comparable_mask = [
        primary_finite_mask[i] and chosen_vals[i] is not None
        for i in range(n)
    ]
    comparable_count = sum(comparable_mask)
    if not comparable_count:
        return {"checked": 0, "match": 0, "match_rate": 0.0, "max_abs_mismatch": None}

    match_count = 0
    max_mismatch = 0.0
    for i in range(n):
        if not comparable_mask[i]:
            continue
        diff = abs(float(primary_vals[i]) - float(chosen_vals[i]))
        if diff <= HIERARCHY_TOLERANCE:
            match_count += 1
        if diff > max_mismatch:
            max_mismatch = diff

    return {
        "checked": comparable_count,
        "match": match_count,
        "match_rate": match_count / comparable_count,
        "max_abs_mismatch": max_mismatch,
    }


def _compare_with_expected(report: dict[str, Any]) -> list[str]:
    """Возвращает список warning'ов для метрик, не совпавших с эталоном.

    Не «чинит» данные — только сообщает.
    """
    warnings: list[str] = []
    name = report["name"]
    expected = EXPECTED[name]
    checks = {
        "rows": expected["rows"],
        "columns": expected["columns"],
        "polygon_count": expected["polygon_count"],
        "primary_ndvi_finite": expected["primary_ndvi_finite"],
    }

    # Для test дополнительно проверяем gaps. is_synthetic_gap в отчёте —
    # это dict {true, false, missing}; берём .true для сравнения.
    if "gaps" in expected:
        gap_obj = report.get("is_synthetic_gap")
        if gap_obj is None:
            warnings.append(f"{name}.gaps: missing (expected {expected['gaps']})")
        else:
            if isinstance(gap_obj, dict):
                got_gaps = int(gap_obj.get("true", 0))
            else:
                got_gaps = int(gap_obj)
            if got_gaps != expected["gaps"]:
                warnings.append(
                    f"{name}.gaps: got {got_gaps}, expected {expected['gaps']} "
                    f"(difference {got_gaps - expected['gaps']:+d})"
                )

    for metric, want in checks.items():
        got = report.get(metric)
        if got is None:
            warnings.append(f"{name}.{metric}: missing (expected {want})")
            continue
        if got != want:
            warnings.append(
                f"{name}.{metric}: got {got}, expected {want} "
                f"(difference {got - want:+d})"
            )
    return warnings


# ───────────────────────────── writers ─────────────────────────────

def write_json(report: dict[str, Any], path: Path) -> None:
    """Пишет JSON. Детерминированно: sort_keys, без timestamp.

    ВАЖНО: никакого datetime.now() — иначе нарушается детерминизм
    согласно правилам проверки данных.

    Если в report ровно два ключа "train" и "test" — оборачиваем в
    envelope c report_schema_version и script_version. Иначе пишем как
    есть (для unit-тестов с произвольной структурой).
    """
    if set(report.keys()) >= {"train", "test"}:
        payload = {
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "script_version": SCRIPT_VERSION,
            "train": report["train"],
            "test": report["test"],
        }
    else:
        payload = {
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "script_version": SCRIPT_VERSION,
            **report,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2)
    # Финальный newline для совместимости с POSIX-текст-конвенциями.
    path.write_text(text + "\n", encoding="utf-8")


def write_markdown(report: dict[str, Any], path: Path) -> None:
    """Пишет Markdown-отчёт по семи разделам проверки данных."""
    train = report["train"]
    test = report["test"]
    lines: list[str] = []
    lines.append("# Data Quality Report")
    lines.append("")
    lines.append(
        f"_report_schema_version: `{REPORT_SCHEMA_VERSION}`, "
        f"script_version: `{SCRIPT_VERSION}`_"
    )
    lines.append("")

    # 1) Summary.
    lines.append("## 1. Summary")
    lines.append("")
    lines.append("| Metric | train | test |")
    lines.append("|---|---:|---:|")
    for m in ("rows", "columns", "polygon_count", "primary_ndvi_finite"):
        lines.append(f"| {m} | {train[m]} | {test[m]} |")
    lines.append(f"| duplicate_keys | {train['duplicate_keys']} | {test['duplicate_keys']} |")
    if test.get("is_synthetic_gap") is not None:
        gaps = test["is_synthetic_gap"]["true"]
        lines.append(f"| gaps (is_synthetic_gap==True) | — | {gaps} |")
    lines.append("")

    # 2) Schema.
    lines.append("## 2. Schema")
    lines.append("")
    lines.append("| file | column | dtype |")
    lines.append("|---|---|---|")
    for col in train["schema"]:
        lines.append(f"| train | `{col['name']}` | `{col['dtype']}` |")
    for col in test["schema"]:
        lines.append(f"| test | `{col['name']}` | `{col['dtype']}` |")
    lines.append("")

    # 3) Missingness.
    lines.append("## 3. Missingness")
    lines.append("")
    lines.append("Top-15 columns by missing rate (test):")
    lines.append("")
    lines.append("| column | missing | missing_rate | finite | finite_rate |")
    lines.append("|---|---:|---:|---:|---:|")
    miss_sorted = sorted(test["missingness"], key=lambda r: r["missing_rate"], reverse=True)
    for r in miss_sorted[:15]:
        lines.append(
            f"| `{r['column']}` | {r['missing']} | {r['missing_rate']:.4f} "
            f"| {r['finite']} | {r['finite_rate']:.4f} |"
        )
    lines.append("")

    # 4) Gaps.
    lines.append("## 4. Gaps")
    lines.append("")
    if test.get("gap_run_lengths") is not None:
        lines.append("| run length | count |")
        lines.append("|---|---:|")
        for k, v in test["gap_run_lengths"].items():
            lines.append(f"| {k} | {v} |")
        lines.append("")
    if test.get("gap_rows_always_missing"):
        lines.append(
            "Columns **always missing** in gap rows: "
            + ", ".join(f"`{c}`" for c in test["gap_rows_always_missing"])
        )
        lines.append("")
    else:
        lines.append("_No gap-only always-missing columns detected._")
        lines.append("")

    # 5) Polygon overlap.
    lines.append("## 5. Polygon overlap")
    lines.append("")
    if test.get("polygon_overlap"):
        ov = test["polygon_overlap"]
        lines.append(f"- train polygons: **{ov['train_polygon_count']}**")
        lines.append(f"- test polygons (known): **{ov['known']}**")
        lines.append(f"- test polygons (unseen): **{ov['unseen']}**")
        if test.get("gap_count_by_overlap"):
            g = test["gap_count_by_overlap"]
            lines.append(f"- gaps on known polygons: **{g['known']}**")
            lines.append(f"- gaps on unseen polygons: **{g['unseen']}**")
    else:
        lines.append("_N/A (computed only for test against train polygons)._")
    lines.append("")

    # 6) Target-source hierarchy.
    lines.append("## 6. Target-source hierarchy (S2 → Landsat → MODIS)")
    lines.append("")
    for label, rep in (("train", train), ("test", test)):
        h = rep["target_source_hierarchy"]
        lines.append(f"**{label}**: checked={h['checked']}, match={h['match']}, "
                     f"match_rate={h['match_rate']:.4f}, "
                     f"max_abs_mismatch={h['max_abs_mismatch']}")
    lines.append("")

    # 7) Warnings.
    lines.append("## 7. Warnings")
    lines.append("")
    all_warnings: list[str] = []
    all_warnings.extend(train.get("warnings", []))
    all_warnings.extend(test.get("warnings", []))
    if all_warnings:
        for w in all_warnings:
            lines.append(f"- ⚠️ {w}")
    else:
        lines.append("_No warnings. All counts match the reference._")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ───────────────────────────── IO wrappers ─────────────────────────────

def _read_csv(path: Path) -> pd.DataFrame:
    """Читает CSV со строгим dtype='str' для anon_polygon_id/date.

    Другие колонки оставляем на авто-detect pandas. Это:
      - защищает anon_polygon_id от превращения в число (AOI-0001 → 1);
      - сохраняет date как строку для последующего строгого парсинга.
    """
    head = pd.read_csv(path, nrows=0)
    dtype: dict[str, str] = {}
    for col in ("anon_polygon_id", "date", "crop_type"):
        if col in head.columns:
            dtype[col] = "string"
    return pd.read_csv(path, dtype=dtype, encoding="utf-8")


def _warn_and_stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


# ───────────────────────────── CLI ─────────────────────────────

@dataclass
class Args:
    train: Path
    test: Path
    out_md: Path
    out_json: Path
    allow_mismatch: bool = False


def parse_args(argv: list[str] | None = None) -> Args:
    parser = argparse.ArgumentParser(
        prog="data_quality_report",
        description="Считает quality-метрики train/test и пишет отчёты.",
    )
    parser.add_argument("--train", required=True, type=Path, help="path to train CSV (relative)")
    parser.add_argument("--test", required=True, type=Path, help="path to test CSV (relative)")
    parser.add_argument("--out-md", required=True, type=Path, help="output markdown path")
    parser.add_argument("--out-json", required=True, type=Path, help="output JSON path")
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="не падать, если counts отличаются от эталона (только предупреждение)",
    )
    ns = parser.parse_args(argv)

    # Абсолютные пути запрещены для воспроизводимости запуска.
    for label, p in (("train", ns.train), ("test", ns.test), ("out-md", ns.out_md), ("out-json", ns.out_json)):
        if p.is_absolute() or PureWindowsPath(str(p)).is_absolute():
            parser.error(f"--{label}={p} is absolute; only relative paths are allowed")

    return Args(
        train=ns.train,
        test=ns.test,
        out_md=ns.out_md,
        out_json=ns.out_json,
        allow_mismatch=ns.allow_mismatch,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # IO.
    for label, path in (("train", args.train), ("test", args.test)):
        if not path.exists():
            _warn_and_stderr(
                f"[error] {label} CSV not found: {path}\n"
                f"  как исправить: передайте существующий относительный путь через --{label}"
            )
            return 2

    try:
        train_df = _read_csv(args.train)
        test_df = _read_csv(args.test)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        _warn_and_stderr(
            f"[error] failed to parse CSV: {type(exc).__name__}: {exc}\n"
            f"  как исправить: проверьте, что файл — валидный UTF-8 CSV с запятой-разделителем"
        )
        return 2

    # Schema check: обязательные колонки.
    for col in ("anon_polygon_id", "date"):
        if col not in train_df.columns:
            _warn_and_stderr(
                f"[error] train CSV missing required column: {col!r}\n"
                "  как исправить: добавьте обязательные колонки anon_polygon_id и date"
            )
            return 2
        if col not in test_df.columns:
            _warn_and_stderr(
                f"[error] test CSV missing required column: {col!r}\n"
                "  как исправить: добавьте обязательные колонки anon_polygon_id и date"
            )
            return 2

    # Train polygons (нужны для overlap в test).
    train_polygons = set(train_df["anon_polygon_id"].astype(str).unique())

    train_report = compute_quality(train_df, "train")
    test_report = compute_quality(test_df, "test", train_polygons=train_polygons)

    # Сравнение с эталоном.
    warnings_train = _compare_with_expected(train_report)
    warnings_test = _compare_with_expected(test_report)
    train_report["warnings"] = warnings_train
    test_report["warnings"] = warnings_test

    has_mismatch = bool(warnings_train or warnings_test)

    # Пишем отчёты (всегда, даже при mismatch — это полезно для ревьюера).
    report = {"train": train_report, "test": test_report}
    try:
        write_json(report, args.out_json)
        write_markdown(report, args.out_md)
    except OSError as exc:
        _warn_and_stderr(
            f"[error] failed to write report: {type(exc).__name__}: {exc}\n"
            f"  как исправить: проверьте права на запись в директорию"
        )
        return 1

    # Итог.
    if has_mismatch and not args.allow_mismatch:
        all_w = warnings_train + warnings_test
        _warn_and_stderr(
            "[error] counts do not match the reference:\n"
            + "\n".join(f"  - {w}" for w in all_w)
            + "\n  как исправить: проверьте входные CSV и ожидаемые контрольные значения; "
            "исходные данные автоматически не изменяются.\n"
            f"  отчёты записаны в {args.out_json} и {args.out_md} для анализа"
        )
        return 1

    if has_mismatch and args.allow_mismatch:
        # Только предупреждение в stderr.
        all_w = warnings_train + warnings_test
        _warn_and_stderr("[warn] counts mismatch (allowed by --allow-mismatch):")
        for w in all_w:
            _warn_and_stderr(f"  - {w}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
