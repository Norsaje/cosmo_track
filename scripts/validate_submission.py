"""Strict submission validator.

Независимая страховка для submission.csv. НЕ редактирует submission,
НЕ авто-фиксит неверные prediction. Проверяет встроенный набор правил формата
и формирует понятный отчёт.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable

import pandas as pd
from pydantic import BaseModel, Field

REPORT_SCHEMA_VERSION = "1.0.0"
SCRIPT_VERSION = "1.0.0"

# Ожидаемая submission schema — заголовок и порядок строго зафиксированы.
SUBMISSION_HEADER: tuple[str, ...] = ("anon_polygon_id", "date", "primary_ndvi_pred")
EXPECTED_ROW_COUNT = 3_112  # число gaps в текущем test

# Регулярка для валидации ISO-даты YYYY-MM-DD (формальная — реальную
# валидность проверит datetime).
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Максимальное число примеров, которое мы печатаем в сообщениях.
MAX_EXAMPLES = 10


# ───────────────────────────── pydantic models ─────────────────────────────

class ValidationError(BaseModel):
    """Одна ошибка валидации submission.

    type — короткий стабильный идентификатор (для машинного парсинга).
    count — сколько таких проблем найдено.
    examples — первые до 10 конкретных примеров.
    hint — как исправить (человеко-читаемо).
    """
    type: str
    count: int
    examples: list[str] = Field(default_factory=list)
    hint: str


class ValidationWarning(BaseModel):
    """Некритичное предупреждение (например, значение вне диапазона)."""
    type: str
    count: int
    examples: list[str] = Field(default_factory=list)
    hint: str


class ValidationSummary(BaseModel):
    """Краткая сводка по submission."""
    rows: int
    expected_rows: int
    key_set_match: bool
    finite_predictions: int


class ValidationResult(BaseModel):
    """Результат валидации.

    is_valid=True ТОЛЬКО если errors пуст. warnings не влияют на is_valid.
    """
    is_valid: bool
    errors: list[ValidationError] = Field(default_factory=list)
    warnings: list[ValidationWarning] = Field(default_factory=list)
    summary: ValidationSummary


# ───────────────────────────── helpers ─────────────────────────────

def _is_finite(value: Any) -> bool:
    """Является ли значение конечным числом."""
    if value is None:
        return False
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, int):
        return True
    if isinstance(value, str):
        s = value.strip()
        if not s or s.lower() in {"nan", "na", "none", "null", "inf", "-inf", "infinity"}:
            return False
        try:
            return math.isfinite(float(s))
        except ValueError:
            return False
    return False


def _format_key(pid: Any, date: Any) -> str:
    """Ключ строки submission в формате, понятном человеку и машине."""
    return f"{pid}|{date}"


def _examples(items: Iterable[Any], limit: int = MAX_EXAMPLES) -> list[str]:
    """Первые ≤limit элементов как строки."""
    out: list[str] = []
    for x in items:
        out.append(str(x))
        if len(out) >= limit:
            break
    return out


# ───────────────────────────── core ─────────────────────────────

def _check_columns(df: pd.DataFrame, errors: list[ValidationError]) -> bool:
    """Проверка заголовка: ровно ожидаемые колонки и в нужном порядке.

    Возвращает True если заголовок OK. Иначе добавляет ошибки и
    возвращает False (дальнейшие проверки бессмысленны).
    """
    cols = list(df.columns)
    if cols == list(SUBMISSION_HEADER):
        return True

    # Дополнительно: если в начале стоит unnamed index — отдельная ошибка.
    has_extra_index = any(str(c).lower().startswith("unnamed") for c in cols)

    expected_str = ",".join(SUBMISSION_HEADER)
    actual_str = ",".join(map(str, cols))
    if set(cols) == set(SUBMISSION_HEADER) and cols != list(SUBMISSION_HEADER):
        # Колонки те же, но порядок другой.
        errors.append(ValidationError(
            type="wrong_column_order",
            count=1,
            examples=[f"got: {actual_str}"],
            hint=(
                "Колонки должны идти строго в порядке: "
                f"{expected_str}"
            ),
        ))
    elif has_extra_index:
        errors.append(ValidationError(
            type="accidental_pandas_index",
            count=1,
            examples=[f"columns: {actual_str}"],
            hint=(
                "В CSV попал служебный index-столбец (Unnamed: 0). "
                "Сохраняйте через pandas.to_csv(..., index=False)."
            ),
        ))
    else:
        errors.append(ValidationError(
            type="wrong_columns",
            count=1,
            examples=[f"got: {actual_str}"],
            hint=(
                "Ожидаются ровно 3 колонки в порядке: "
                f"{expected_str}. Лишние или отсутствующие колонки недопустимы."
            ),
        ))
    return False


def _check_dates(
    df: pd.DataFrame, errors: list[ValidationError]
) -> list[pd.Timestamp]:
    """Строгий парсинг ISO-дат (YYYY-MM-DD). Возвращает список Timestamp'ов."""
    dates = df["date"].astype(str)
    # Регулярка отсекает мусор; pd.to_datetime ловит невалидные комбинации
    # типа "2020-02-30".
    valid_mask = dates.str.match(_DATE_RE)
    parsed = pd.to_datetime(dates, format="%Y-%m-%d", errors="coerce")
    invalid_mask = parsed.isna() | ~valid_mask
    if invalid_mask.any():
        bad_idx = df.index[invalid_mask].tolist()
        bad_examples = _examples(
            _format_key(df.loc[i, "anon_polygon_id"], df.loc[i, "date"])
            for i in bad_idx
        )
        errors.append(ValidationError(
            type="invalid_date",
            count=int(invalid_mask.sum()),
            examples=bad_examples,
            hint=(
                "Каждая дата должна быть валидной YYYY-MM-DD. "
                "Проверьте формат строк; '2020-13-01' или 'abc' — недопустимы."
            ),
        ))
    return parsed.tolist()


def _check_predictions_finite(
    df: pd.DataFrame, errors: list[ValidationError]
) -> list[float | None]:
    """Проверка, что primary_ndvi_pred парсится в float и finite."""
    col = df["primary_ndvi_pred"]
    parsed: list[float | None] = []
    for i in range(len(col)):
        v = col.iloc[i]
        if _is_finite(v):
            parsed.append(float(v))
        else:
            parsed.append(None)
    non_finite = sum(1 for p in parsed if p is None)
    if non_finite:
        # Собираем типы проблем: NaN, inf, string — отдельными bucket'ами.
        bad_nan: list[str] = []
        bad_inf: list[str] = []
        bad_str: list[str] = []
        for i in range(len(col)):
            if parsed[i] is not None:
                continue
            v = col.iloc[i]
            key = _format_key(df.loc[i, "anon_polygon_id"], df.loc[i, "date"])
            if isinstance(v, str):
                s = v.strip().lower()
                if s.startswith("inf") or s.startswith("+inf"):
                    bad_inf.append(key + f" ({v})")
                elif s in {"nan", "na", "none", "null"}:
                    bad_nan.append(key + f" ({v!r})")
                else:
                    bad_str.append(key + f" ({v!r})")
            elif isinstance(v, float):
                # NB: NaN попадает сюда. math.isfinite(NaN)=False.
                if math.isnan(v):
                    bad_nan.append(key + " (NaN)")
                elif math.isinf(v):
                    bad_inf.append(key + f" ({v})")
                else:
                    bad_str.append(key + f" ({v!r})")
            elif v is None:
                bad_nan.append(key + " (None)")
            else:
                # numpy.float64(Nan) и другие числовые типы.
                f = float(v)
                if math.isnan(f):
                    bad_nan.append(key + " (NaN)")
                elif math.isinf(f):
                    bad_inf.append(key + f" ({f})")
                else:
                    bad_str.append(key + f" ({v!r})")
        # Подтип — самый специфичный из встретившихся.
        examples: list[str] = []
        for bucket in (bad_nan, bad_inf, bad_str):
            examples.extend(bucket)
        if bad_str:
            sub_type = "string_prediction"
        elif bad_inf and not bad_nan:
            sub_type = "inf_prediction"
        elif bad_nan and not bad_inf:
            sub_type = "nan_prediction"
        else:
            sub_type = "non_finite_prediction"
        errors.append(ValidationError(
            type=sub_type,
            count=non_finite,
            examples=_examples(examples),
            hint=(
                "primary_ndvi_pred должен быть парсируемым в float "
                "и конечным (не NaN, не ±inf). Замените значения на вещественные числа."
            ),
        ))
    return parsed


def _check_duplicate_keys(
    df: pd.DataFrame, errors: list[ValidationError]
) -> None:
    """Проверка уникальности ключей anon_polygon_id+date."""
    keys = [_format_key(df["anon_polygon_id"].iloc[i], df["date"].iloc[i])
            for i in range(len(df))]
    seen: set[str] = set()
    dups: list[str] = []
    for k in keys:
        if k in seen:
            dups.append(k)
        else:
            seen.add(k)
    if dups:
        errors.append(ValidationError(
            type="duplicate_key",
            count=len(dups),
            examples=_examples(dups),
            hint=(
                "Каждый ключ anon_polygon_id+date должен встречаться ровно один раз. "
                "Удалите дубликаты строк."
            ),
        ))


def _check_row_count(
    df: pd.DataFrame,
    test_gap_keys: set[tuple[str, str]],
    errors: list[ValidationError],
) -> None:
    """Число строк должно совпадать с числом gap-ключей test."""
    expected = len(test_gap_keys)
    actual = len(df)
    if actual != expected:
        errors.append(ValidationError(
            type="row_count_mismatch",
            count=1,
            examples=[f"got {actual}, expected {expected}"],
            hint=(
                f"submission должен содержать ровно {expected} строк "
                "(по числу gap-ключей в test)."
            ),
        ))


def _check_key_set(
    df: pd.DataFrame,
    parsed_dates: list[pd.Timestamp],
    test_gap_keys: set[tuple[str, str]],
    errors: list[ValidationError],
) -> None:
    """Key set submission должен быть равен test gap keys."""
    sub_keys: set[tuple[str, str]] = set()
    for i in range(len(df)):
        pid = str(df["anon_polygon_id"].iloc[i])
        d = parsed_dates[i]
        if pd.isna(d):
            continue
        sub_keys.add((pid, d.strftime("%Y-%m-%d")))

    missing = test_gap_keys - sub_keys
    extra = sub_keys - test_gap_keys
    if missing:
        errors.append(ValidationError(
            type="missing_key",
            count=len(missing),
            examples=_examples(f"{pid}|{date}" for pid, date in missing),
            hint=(
                "Эти gap-ключи из test отсутствуют в submission. "
                "Добавьте строки с предсказаниями для них."
            ),
        ))
    if extra:
        errors.append(ValidationError(
            type="extra_key",
            count=len(extra),
            examples=_examples(f"{pid}|{date}" for pid, date in extra),
            hint=(
                "Эти ключи есть в submission, но их нет среди test gaps. "
                "Удалите лишние строки (submission должен покрывать только gaps)."
            ),
        ))


def _check_predictions_range(
    parsed: list[float | None],
    df: pd.DataFrame,
    warn_outside: tuple[float, float] | None,
    warnings: list[ValidationWarning],
) -> None:
    """Опциональный warning о значениях вне физического диапазона.

    Не отклоняет submission, только предупреждает.
    """
    if warn_outside is None:
        return
    lo, hi = warn_outside
    if lo > hi:
        # Некорректный диапазон — игнорируем, чтобы не шуметь.
        return
    bad: list[str] = []
    for i, p in enumerate(parsed):
        if p is None:
            continue
        if p < lo or p > hi:
            bad.append(_format_key(
                df["anon_polygon_id"].iloc[i],
                df["date"].iloc[i],
            ) + f" ({p})")
    if bad:
        warnings.append(ValidationWarning(
            type="prediction_outside_warn_range",
            count=len(bad),
            examples=_examples(bad),
            hint=(
                f"Значения primary_ndvi_pred вне [{lo}, {hi}]. "
                "Это только предупреждение; submission всё равно валиден, "
                "если прошёл остальные проверки."
            ),
        ))


# ───────────────────────────── public API ─────────────────────────────

def extract_test_gap_keys(test_df: pd.DataFrame) -> set[tuple[str, str]]:
    """Извлекает set (anon_polygon_id, date_str) для всех gap-строк test.

    Это «эталонный» набор ключей, относительно которого валидируется
    submission.
    """
    if "is_synthetic_gap" not in test_df.columns:
        return set()
    gap_mask = test_df["is_synthetic_gap"] == True  # noqa: E712
    if not gap_mask.any():
        return set()
    gap_df = test_df[gap_mask]
    return {
        (str(row["anon_polygon_id"]), str(row["date"]))
        for _, row in gap_df.iterrows()
    }


def validate_submission(
    submission_df: pd.DataFrame,
    test_gap_keys: set[tuple[str, str]],
    *,
    warn_outside: tuple[float, float] | None = None,
) -> ValidationResult:
    """Главная точка входа. Чистая функция — тестируется без subprocess.

    submission_df: распарсенный CSV (см. _read_submission).
    test_gap_keys: set (anon_polygon_id, date_str) из test gaps.
    warn_outside: (lo, hi) — если задано, значения вне диапазона дают warning,
        но НЕ ломают валидацию.
    """
    errors: list[ValidationError] = []
    warnings: list[ValidationWarning] = []

    # 1) Заголовок.
    if not _check_columns(submission_df, errors):
        # Без правильных колонок дальше проверять бессмысленно.
        return ValidationResult(
            is_valid=False,
            errors=errors,
            warnings=warnings,
            summary=ValidationSummary(
                rows=len(submission_df),
                expected_rows=len(test_gap_keys),
                key_set_match=False,
                finite_predictions=0,
            ),
        )

    # 2) Даты.
    parsed_dates = _check_dates(submission_df, errors)

    # 3) primary_ndvi_pred finite.
    parsed_preds = _check_predictions_finite(submission_df, errors)

    # 4) Дубликаты ключей.
    _check_duplicate_keys(submission_df, errors)

    # 5) Row count.
    _check_row_count(submission_df, test_gap_keys, errors)

    # 6) Key set match.
    _check_key_set(submission_df, parsed_dates, test_gap_keys, errors)

    # 7) Опциональный range warning.
    _check_predictions_range(parsed_preds, submission_df, warn_outside, warnings)

    finite_count = sum(1 for p in parsed_preds if p is not None)
    key_set_match = (
        len(errors) == 0  # грубая проверка; точная — в _check_key_set
    )
    # Точная key_set_match: errors нет missing и нет extra.
    has_missing = any(e.type == "missing_key" for e in errors)
    has_extra = any(e.type == "extra_key" for e in errors)
    key_set_match = not has_missing and not has_extra

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        summary=ValidationSummary(
            rows=len(submission_df),
            expected_rows=len(test_gap_keys),
            key_set_match=key_set_match,
            finite_predictions=finite_count,
        ),
    )


# ───────────────────────────── IO ─────────────────────────────

def _read_submission(path: Path) -> tuple[pd.DataFrame, str | None]:
    """Читает submission CSV с поддержкой UTF-8-sig (BOM).

    Возвращает (df, encoding_used) — encoding полезен для отчёта.
    Бросает ValueError на структурные проблемы.
    """
    # 1) Декодируем файл.
    raw_bytes = path.read_bytes()
    # Пробуем UTF-8-sig (с BOM), затем обычный UTF-8.
    encoding_used: str | None = None
    text: str | None = None
    for enc in ("utf-8-sig", "utf-8"):
        try:
            text = raw_bytes.decode(enc)
            encoding_used = enc
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError("file is not valid UTF-8")

    # 2) Проверяем delimiter: разрешена только запятая.
    # Если первая непустая строка содержит ';' и не содержит ',',
    # это semicolon-delimited CSV → ошибка.
    first_line = text.splitlines()[0] if text else ""
    if ";" in first_line and "," not in first_line:
        raise ValueError("file uses semicolon delimiter; expected comma")

    # 3) Парсим через pandas — он сам разберётся с заголовком.
    from io import StringIO
    df = pd.read_csv(StringIO(text))
    return df, encoding_used


def _read_test(path: Path) -> pd.DataFrame:
    """Читает test CSV. anon_polygon_id и is_synthetic_gap — специальные dtype."""
    head = pd.read_csv(path, nrows=0)
    dtype: dict[str, str] = {}
    for col in ("anon_polygon_id", "date"):
        if col in head.columns:
            dtype[col] = "string"
    df = pd.read_csv(path, dtype=dtype, encoding="utf-8")
    # is_synthetic_gap из CSV может прийти как string ("True"/"False").
    # Приводим к bool: True / False / NaN. Используем pandas nullable Bool
    # (через .astype("boolean")), чтобы корректно работать с пропусками.
    if "is_synthetic_gap" in df.columns:
        # Сначала нормализуем значения: map для обычных строк, для Int64 —
        # прямое приведение.
        if df["is_synthetic_gap"].dtype == object or str(df["is_synthetic_gap"].dtype) == "string":
            mapping = {
                "True": True, "False": False, "true": True, "false": False,
                "TRUE": True, "FALSE": False,
            }
            converted = df["is_synthetic_gap"].map(mapping)
            # Преобразуем в boolean (nullable), NaN остаются NaN.
            df["is_synthetic_gap"] = converted.astype("boolean")
        elif str(df["is_synthetic_gap"].dtype).startswith("Int"):
            df["is_synthetic_gap"] = df["is_synthetic_gap"].astype("boolean")
        # Если dtype уже bool — оставляем как есть.
    return df


# ───────────────────────────── writers ─────────────────────────────

def write_json_report(result: ValidationResult, path: Path) -> None:
    """Пишет JSON-отчёт. Детерминированно (sort_keys, без timestamp)."""
    payload = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "is_valid": result.is_valid,
        "summary": result.summary.model_dump(),
        "errors": [e.model_dump() for e in result.errors],
        "warnings": [w.model_dump() for w in result.warnings],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2)
    path.write_text(text + "\n", encoding="utf-8")


def format_stderr_report(result: ValidationResult) -> str:
    """Компактный текст для stderr (если --json-report не указан)."""
    if result.is_valid and not result.warnings:
        return f"[ok] submission valid; {result.summary.rows} rows, " \
               f"key_set_match={result.summary.key_set_match}"
    lines: list[str] = []
    if result.is_valid:
        lines.append("[ok] submission valid")
    else:
        lines.append(f"[error] submission INVALID ({len(result.errors)} problem type(s)):")
    for e in result.errors:
        ex = ", ".join(e.examples) if e.examples else "(no examples)"
        lines.append(f"  - {e.type}: {e.count} | examples: {ex}")
        lines.append(f"      hint: {e.hint}")
    for w in result.warnings:
        ex = ", ".join(w.examples) if w.examples else "(no examples)"
        lines.append(f"  - [warn] {w.type}: {w.count} | examples: {ex}")
        lines.append(f"      hint: {w.hint}")
    lines.append(
        f"  summary: rows={result.summary.rows}, "
        f"expected={result.summary.expected_rows}, "
        f"key_set_match={result.summary.key_set_match}, "
        f"finite={result.summary.finite_predictions}"
    )
    return "\n".join(lines)


# ───────────────────────────── CLI ─────────────────────────────

@dataclass
class Args:
    test: Path
    submission: Path
    json_report: Path | None = None
    warn_outside: tuple[float, float] | None = None


def parse_args(argv: list[str] | None = None) -> Args:
    parser = argparse.ArgumentParser(
        prog="validate_submission",
        description="Строгий валидатор submission.csv (без auto-fix).",
    )
    parser.add_argument("--test", required=True, type=Path,
                        help="path to test CSV (relative)")
    parser.add_argument("--submission", required=True, type=Path,
                        help="path to submission CSV (relative)")
    parser.add_argument("--json-report", type=Path, default=None,
                        help="optional JSON report path")
    parser.add_argument("--warn-outside", nargs=2, type=float, default=None,
                        metavar=("MIN", "MAX"),
                        help="warn if prediction outside [MIN, MAX] (no rejection)")
    ns = parser.parse_args(argv)

    for label, p in (("test", ns.test), ("submission", ns.submission),
                     ("json-report", ns.json_report)):
        if p is None:
            continue
        if p.is_absolute() or PureWindowsPath(str(p)).is_absolute():
            parser.error(f"--{label}={p} is absolute; only relative paths are allowed")

    return Args(
        test=ns.test,
        submission=ns.submission,
        json_report=ns.json_report,
        warn_outside=tuple(ns.warn_outside) if ns.warn_outside else None,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # IO.
    for label, path in (("test", args.test), ("submission", args.submission)):
        if not path.exists():
            print(
                f"[error] {label} CSV not found: {path}\n"
                f"  как исправить: передайте существующий относительный путь через --{label}",
                file=sys.stderr,
            )
            return 2

    try:
        test_df = _read_test(args.test)
    except (OSError, UnicodeDecodeError, ValueError, pd.errors.ParserError) as exc:
        print(
            f"[error] failed to parse test CSV: {type(exc).__name__}: {exc}\n"
            f"  как исправить: проверьте, что test — валидный UTF-8 CSV с запятой-разделителем",
            file=sys.stderr,
        )
        return 2

    if "anon_polygon_id" not in test_df.columns or "is_synthetic_gap" not in test_df.columns:
        print(
            "[error] test CSV missing required columns: anon_polygon_id, is_synthetic_gap\n"
            "  как исправить: используйте исходный test_data.csv",
            file=sys.stderr,
        )
        return 2

    try:
        sub_df, encoding_used = _read_submission(args.submission)
    except FileNotFoundError:
        print(
            f"[error] submission not found: {args.submission}",
            file=sys.stderr,
        )
        return 2
    except (OSError, UnicodeDecodeError, ValueError, pd.errors.ParserError) as exc:
        print(
            f"[error] failed to parse submission: {type(exc).__name__}: {exc}\n"
            f"  как исправить: проверьте, что submission — валидный UTF-8 CSV с запятой-разделителем",
            file=sys.stderr,
        )
        return 2

    test_gap_keys = extract_test_gap_keys(test_df)
    result = validate_submission(
        sub_df,
        test_gap_keys,
        warn_outside=args.warn_outside,
    )

    # Пишем JSON-отчёт если просили.
    if args.json_report is not None:
        try:
            write_json_report(result, args.json_report)
        except OSError as exc:
            print(
                f"[error] failed to write JSON report: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 2

    # Печатаем компактный отчёт в stderr.
    print(format_stderr_report(result), file=sys.stderr)

    return 0 if result.is_valid else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
