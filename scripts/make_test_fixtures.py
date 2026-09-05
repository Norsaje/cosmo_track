"""Safe fixture builder for tests and demo.

Создаёт маленькие детерминированные CSV из реальных train/test для
offline unit/integration тестов. Не раскрывает скрытый ground truth
(не придумывает primary_ndvi для test gap).

Формат созданных файлов фиксируется в `tests/fixtures/manifest.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

import pandas as pd

MANIFEST_SCHEMA_VERSION = "1.0.0"
SCRIPT_VERSION = "1.0.0"

# Идентификаторы TOY-* используются для полностью синтетических строк,
# которые НЕ смешиваются с копиями реальных данных.
TOY_PREFIX = "TOY-"

# Дефолтные значения для synthetic строк (NDVI-диапазон реалистичный,
# но не претендует на ground truth).
DEFAULT_PREDICTION = 0.5
SYNTHETIC_ERA5_TEMP = 12.0
SYNTHETIC_ERA5_PRECIP = 1.0
SYNTHETIC_REFERENCE_YEARS = 5

# Минимальные/максимальные длины gap-рана, которые мы ищем в данных.
GAP_RUN_MIN = 2
GAP_RUN_MAX = 4

# Лимит строк в выходных файлах (privacy + размер).
TRAIN_TINY_MAX_ROWS = 60
TEST_TINY_MAX_ROWS = 60

# Все выходные колонки — те же, что в исходных CSV.
TRAIN_HEADER_DEFAULT = [
    "anon_polygon_id", "date",
    "s2_ndvi", "s2_evi", "s2_ndwi",
    "landsat_ndvi", "landsat_evi", "landsat_ndwi",
    "modis_ndvi", "modis_evi",
    "era5_temp_c", "era5_precip_mm",
    "year", "primary_ndvi", "doy",
    "ndvi_climatology_mean", "ndvi_climatology_std",
    "ndvi_zscore", "n_reference_years",
    "status", "crop_type",
]
TEST_HEADER_DEFAULT = [
    "anon_polygon_id", "date",
    "s2_ndvi", "s2_evi", "s2_ndwi",
    "landsat_ndvi", "landsat_evi", "landsat_ndwi",
    "modis_ndvi", "modis_evi",
    "era5_temp_c", "era5_precip_mm",
    "year", "primary_ndvi", "doy",
    "ndvi_climatology_mean", "ndvi_climatology_std",
    "n_reference_years", "is_synthetic_gap",
    "crop_type",
]
SUBMISSION_HEADER = ("anon_polygon_id", "date", "primary_ndvi_pred")


# ───────────────────────────── public API ─────────────────────────────

def select_known_polygon(
    train_polys: set[str], test_polys: set[str]
) -> str | None:
    """Первый по алфавиту polygon, присутствующий и в train, и в test."""
    common = sorted(train_polys & test_polys)
    return common[0] if common else None


def select_unseen_polygon(
    train_polys: set[str], test_polys: set[str]
) -> str | None:
    """Первый по алфавиту polygon, присутствующий только в test."""
    unseen = sorted(test_polys - train_polys)
    return unseen[0] if unseen else None


def find_gap_run(
    df: pd.DataFrame,
    polygon: str,
    *,
    min_len: int = GAP_RUN_MIN,
    max_len: int = GAP_RUN_MAX,
) -> list[pd.Timestamp] | None:
    """Ищет последовательный gap-ран для polygon длиной [min_len, max_len].

    Последовательность = соседние gap-даты с разницей ровно 1 день.
    Возвращает None если не нашли.
    """
    if "is_synthetic_gap" not in df.columns:
        return None
    sub = df[(df["anon_polygon_id"] == polygon) & (df["is_synthetic_gap"] == True)].copy()  # noqa: E712
    if sub.empty:
        return None
    sub["date_parsed"] = pd.to_datetime(sub["date"], format="%Y-%m-%d", errors="coerce")
    sub = sub.dropna(subset=["date_parsed"]).sort_values("date_parsed")
    if sub.empty:
        return None

    dates: list[pd.Timestamp] = sub["date_parsed"].tolist()
    # Ищем раны.
    for i in range(len(dates)):
        run = [dates[i]]
        for j in range(i + 1, len(dates)):
            if (dates[j] - run[-1]).days == 1:
                run.append(dates[j])
                if len(run) >= max_len:
                    break
            else:
                break
        if min_len <= len(run) <= max_len:
            return run
    return None


def build_train_tiny(
    df: pd.DataFrame,
    polygon: str,
    *,
    max_rows: int = TRAIN_TINY_MAX_ROWS,
) -> pd.DataFrame:
    """Берёт строки polygon из train_df. Первые max_rows по дате.

    Сортировка по дате важна для воспроизводимости.
    """
    sub = df[df["anon_polygon_id"] == polygon].copy()
    sub["date_parsed"] = pd.to_datetime(sub["date"], format="%Y-%m-%d", errors="coerce")
    sub = sub.dropna(subset=["date_parsed"]).sort_values("date_parsed").head(max_rows)
    return sub.drop(columns=["date_parsed"]).reset_index(drop=True)


def build_test_tiny(
    df: pd.DataFrame,
    polygon: str,
    *,
    max_rows: int = TEST_TINY_MAX_ROWS,
) -> pd.DataFrame:
    """Берёт строки polygon из test_df. Первые max_rows по дате."""
    sub = df[df["anon_polygon_id"] == polygon].copy()
    sub["date_parsed"] = pd.to_datetime(sub["date"], format="%Y-%m-%d", errors="coerce")
    sub = sub.dropna(subset=["date_parsed"]).sort_values("date_parsed").head(max_rows)
    return sub.drop(columns=["date_parsed"]).reset_index(drop=True)


def build_submission_valid(
    test_tiny: pd.DataFrame,
    *,
    prediction: float = DEFAULT_PREDICTION,
) -> pd.DataFrame:
    """Создаёт валидный submission для test_tiny: только gap-строки."""
    if "is_synthetic_gap" not in test_tiny.columns:
        raise ValueError("test_tiny must have is_synthetic_gap column")
    gap = test_tiny[test_tiny["is_synthetic_gap"] == True].copy()  # noqa: E712
    return pd.DataFrame({
        "anon_polygon_id": gap["anon_polygon_id"].values,
        "date": gap["date"].values,
        "primary_ndvi_pred": prediction,
    })


def build_submission_invalid_duplicate(submission: pd.DataFrame) -> pd.DataFrame:
    """Submission с 1 дубликатом (первая строка повторена)."""
    if submission.empty:
        raise ValueError("submission is empty; cannot duplicate")
    return pd.concat([submission.iloc[[0]], submission], ignore_index=True)


def make_synthetic_gap_run(
    polygon: str,
    start_date: pd.Timestamp,
    length: int,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Создаёт полностью synthetic gap-run с идентификатором TOY-*.

    Используется только когда в реальных данных нет нужного рана.
    Все NDVI-колонки пустые (нет ground truth).
    """
    import random
    rng = random.Random(seed + len(polygon))
    rows: list[dict[str, Any]] = []
    for i in range(length):
        d = start_date + pd.Timedelta(days=i)
        rows.append({
            "anon_polygon_id": polygon,
            "date": d.strftime("%Y-%m-%d"),
            "s2_ndvi": None, "s2_evi": None, "s2_ndwi": None,
            "landsat_ndvi": None, "landsat_evi": None, "landsat_ndwi": None,
            "modis_ndvi": None, "modis_evi": None,
            "era5_temp_c": SYNTHETIC_ERA5_TEMP + rng.uniform(-2, 2),
            "era5_precip_mm": SYNTHETIC_ERA5_PRECIP + rng.uniform(-0.5, 0.5),
            "year": float(d.year),
            "primary_ndvi": None,
            "doy": float(d.timetuple().tm_yday),
            "ndvi_climatology_mean": None,
            "ndvi_climatology_std": None,
            "n_reference_years": float(SYNTHETIC_REFERENCE_YEARS),
            "is_synthetic_gap": True,
            "crop_type": "synthetic",
        })
    return pd.DataFrame(rows, columns=TEST_HEADER_DEFAULT)


def make_synthetic_one_sided_context(
    polygon: str,
    anchor_date: pd.Timestamp,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic строка, где только S2 есть, а Landsat/MODIS пустые."""
    import random
    rng = random.Random(seed)
    s2 = round(0.4 + rng.uniform(-0.1, 0.3), 4)
    return pd.DataFrame([{
        "anon_polygon_id": polygon,
        "date": anchor_date.strftime("%Y-%m-%d"),
        "s2_ndvi": s2, "s2_evi": round(s2 * 0.55, 4), "s2_ndwi": round(s2 * 0.18, 4),
        "landsat_ndvi": None, "landsat_evi": None, "landsat_ndwi": None,
        "modis_ndvi": None, "modis_evi": None,
        "era5_temp_c": SYNTHETIC_ERA5_TEMP, "era5_precip_mm": SYNTHETIC_ERA5_PRECIP,
        "year": float(anchor_date.year),
        "primary_ndvi": s2,  # S2 present → иерархия S2 → Landsat → MODIS даёт S2.
        "doy": float(anchor_date.timetuple().tm_yday),
        "ndvi_climatology_mean": None, "ndvi_climatology_std": None,
        "n_reference_years": float(SYNTHETIC_REFERENCE_YEARS),
        "is_synthetic_gap": False,
        "crop_type": "synthetic",
    }], columns=TEST_HEADER_DEFAULT)


def sha256_file(path: Path) -> str:
    """SHA256 файла (для source fingerprints в manifest)."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    """Пишет manifest JSON. Детерминированно (sort_keys, без timestamp)."""
    payload = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        **manifest,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2)
    path.write_text(text + "\n", encoding="utf-8")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Пишет CSV без служебного pandas-индекса."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ───────────────────────────── main builder ─────────────────────────────

@dataclass
class BuildResult:
    """Результат сборки fixtures. Тестируется без subprocess."""
    train_tiny: pd.DataFrame
    test_tiny: pd.DataFrame
    submission_valid: pd.DataFrame
    submission_invalid_duplicate: pd.DataFrame
    manifest: dict[str, Any]


def build_fixtures(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    seed: int = 42,
) -> BuildResult:
    """Главная точка сборки. Чистая функция — тестируется без subprocess.

    Логика выбора:
    - known: первый по алфавиту в пересечении train ∩ test;
    - unseen: первый по алфавиту в test \\ train;
    - gap run длиной [2,4]: ищем в данных; если нет — synthetic TOY-RUN-A;
    - one-sided context: берём строку из known; если нет — synthetic TOY-CTX-A;
    - synthetic строки помечены TOY-* и НЕ смешиваются с копиями реальных.
    """
    train_polys = set(train_df["anon_polygon_id"].astype(str).unique())
    test_polys = set(test_df["anon_polygon_id"].astype(str).unique())

    known = select_known_polygon(train_polys, test_polys)
    unseen = select_unseen_polygon(train_polys, test_polys)

    # 1) train_tiny: берём known polygon из train (если есть).
    if known is not None:
        train_tiny = build_train_tiny(train_df, known)
        train_source = f"real:{known}"
    else:
        # Fallback: пустой DataFrame с правильной схемой.
        train_tiny = pd.DataFrame(columns=TRAIN_HEADER_DEFAULT)
        train_source = "synthetic:empty"

    # 2) test_tiny: known polygon (с gap-run если нашли) + unseen + synthetic добавки.
    test_tiny_parts: list[pd.DataFrame] = []
    synthetic_added: list[str] = []
    gap_run_polygon_used: str | None = None

    if known is not None:
        known_test_rows = build_test_tiny(test_df, known)
        test_tiny_parts.append(known_test_rows)

        # Gap run для known.
        gap_run = find_gap_run(test_df, known)
        if gap_run is not None:
            gap_run_polygon_used = known
        else:
            # Synthetic fallback для gap run длиной GAP_RUN_MAX.
            anchor = pd.Timestamp("2020-04-09")
            synthetic_run = make_synthetic_gap_run(
                f"{TOY_PREFIX}RUN-A", anchor, GAP_RUN_MAX, seed=seed,
            )
            test_tiny_parts.append(synthetic_run)
            synthetic_added.append(f"{TOY_PREFIX}RUN-A")
            gap_run_polygon_used = f"{TOY_PREFIX}RUN-A"

        # One-sided context для known.
        if not known_test_rows.empty:
            test_tiny_parts.append(make_synthetic_one_sided_context(
                f"{TOY_PREFIX}CTX-A",
                pd.to_datetime(known_test_rows["date"].iloc[-1])
                + pd.Timedelta(days=1),
                seed=seed,
            ))
            synthetic_added.append(f"{TOY_PREFIX}CTX-A")

    if unseen is not None:
        unseen_rows = build_test_tiny(test_df, unseen)
        test_tiny_parts.append(unseen_rows)

    if not test_tiny_parts:
        # Совсем нет данных → пустой test_tiny с правильной схемой.
        test_tiny = pd.DataFrame(columns=TEST_HEADER_DEFAULT)
    else:
        # FutureWarning про concat с empty/all-NA колонками (GH#39122):
        # возникает, когда synthetic-строки имеют полностью пустые NDVI-
        # колонки. Это ожидаемое поведение, шум подавляем локально.
        import warnings as _w
        non_empty = [p for p in test_tiny_parts if not p.empty]
        if non_empty:
            with _w.catch_warnings():
                _w.filterwarnings(
                    "ignore",
                    message="The behavior of DataFrame concatenation with empty.*",
                    category=FutureWarning,
                )
                test_tiny = pd.concat(non_empty, ignore_index=True)
        else:
            test_tiny = pd.DataFrame(columns=TEST_HEADER_DEFAULT)

    # 3) Валидный submission для test_tiny.
    submission_valid = build_submission_valid(test_tiny)

    # 4) Submission с дубликатом.
    submission_invalid_dup = build_submission_invalid_duplicate(submission_valid)

    # 5) Manifest.
    selected_keys: dict[str, Any] = {
        "known_polygon": known,
        "unseen_polygon": unseen,
        "gap_run_polygon": gap_run_polygon_used,
        "synthetic_added": synthetic_added,
    }
    design_assumptions = [
        "Берём ровно 1 known polygon (первый по алфавиту в train ∩ test).",
        "Берём ровно 1 unseen polygon (первый по алфавиту в test \\ train).",
        "Gap run длиной [2,4] ищем в реальных данных; если не нашли — synthetic TOY-RUN-A.",
        "One-sided context: synthetic TOY-CTX-A (только S2 present).",
        "Submission valid = только gap-строки test_tiny с prediction=0.5.",
        "Submission invalid duplicate = submission valid + 1 дубликат первой строки.",
    ]
    manifest = {
        "selection_rule": {
            "criteria": "first alphabetically for known/unseen, first gap-run [2,4] found, fallback TOY-* if missing",
            "seed": seed,
            "max_rows_train_tiny": TRAIN_TINY_MAX_ROWS,
            "max_rows_test_tiny": TEST_TINY_MAX_ROWS,
        },
        "source_fingerprints": {
            # sha256 вычислим позже в main (когда знаем пути), здесь заглушки.
            "train_dataset.csv": None,
            "test_data.csv": None,
        },
        "selected_keys": selected_keys,
        "files": {
            "train_tiny.csv": {
                "rows": len(train_tiny),
                "columns": len(train_tiny.columns),
                "source": train_source,
            },
            "test_tiny.csv": {
                "rows": len(test_tiny),
                "columns": len(test_tiny.columns),
                "gaps": int((test_tiny.get("is_synthetic_gap", pd.Series(dtype=bool)) == True).sum()),  # noqa: E712
            },
            "submission_for_test_tiny.csv": {
                "rows": len(submission_valid),
                "columns": len(submission_valid.columns),
            },
            "submission_invalid_duplicate.csv": {
                "rows": len(submission_invalid_dup),
                "columns": len(submission_invalid_dup.columns),
            },
        },
        "design_assumptions": design_assumptions,
        "design_assumptions_requires_ml_confirmation": True,
        "created_by_script_version": SCRIPT_VERSION,
    }
    return BuildResult(
        train_tiny=train_tiny,
        test_tiny=test_tiny,
        submission_valid=submission_valid,
        submission_invalid_duplicate=submission_invalid_dup,
        manifest=manifest,
    )


# ───────────────────────────── IO ─────────────────────────────

def _read_csv(path: Path) -> pd.DataFrame:
    """Читает CSV. anon_polygon_id и date — string (защита от float coercion)."""
    head = pd.read_csv(path, nrows=0)
    dtype: dict[str, str] = {}
    for col in ("anon_polygon_id", "date"):
        if col in head.columns:
            dtype[col] = "string"
    return pd.read_csv(path, dtype=dtype, encoding="utf-8")


# ───────────────────────────── CLI ─────────────────────────────

@dataclass
class Args:
    out: Path
    manifest: Path
    train_source: Path
    test_source: Path
    seed: int = 42


def parse_args(argv: list[str] | None = None) -> Args:
    parser = argparse.ArgumentParser(
        prog="make_test_fixtures",
        description="Создаёт детерминированные маленькие fixtures из train/test.",
    )
    parser.add_argument("--out", required=True, type=Path,
                        help="output directory for fixtures (relative)")
    parser.add_argument("--manifest", required=True, type=Path,
                        help="output manifest.json path (relative)")
    parser.add_argument("--train-source", type=Path,
                        default=Path("data/train_dataset.csv"),
                        help="source train CSV (relative)")
    parser.add_argument("--test-source", type=Path,
                        default=Path("data/test_data.csv"),
                        help="source test CSV (relative)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for synthetic generators (default: 42)")
    ns = parser.parse_args(argv)

    for label, p in (
        ("out", ns.out), ("manifest", ns.manifest),
        ("train-source", ns.train_source), ("test-source", ns.test_source),
    ):
        if p.is_absolute() or PureWindowsPath(str(p)).is_absolute():
            parser.error(f"--{label}={p} is absolute; only relative paths are allowed")

    return Args(
        out=ns.out,
        manifest=ns.manifest,
        train_source=ns.train_source,
        test_source=ns.test_source,
        seed=ns.seed,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # IO sources.
    for label, path in (("train-source", args.train_source), ("test-source", args.test_source)):
        if not path.exists():
            print(
                f"[error] {label} not found: {path}\n"
                f"  как исправить: передайте существующий относительный путь",
                file=sys.stderr,
            )
            return 2

    try:
        train_df = _read_csv(args.train_source)
        test_df = _read_csv(args.test_source)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        print(
            f"[error] failed to parse source CSV: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    # Сборка fixtures.
    result = build_fixtures(train_df, test_df, seed=args.seed)

    # Source fingerprints.
    result.manifest["source_fingerprints"]["train_dataset.csv"] = {
        "sha256": sha256_file(args.train_source),
        "rows": len(train_df),
        "columns": len(train_df.columns),
    }
    result.manifest["source_fingerprints"]["test_data.csv"] = {
        "sha256": sha256_file(args.test_source),
        "rows": len(test_df),
        "columns": len(test_df.columns),
    }

    # Пишем файлы.
    try:
        args.out.mkdir(parents=True, exist_ok=True)
        write_csv(result.train_tiny, args.out / "train_tiny.csv")
        write_csv(result.test_tiny, args.out / "test_tiny.csv")
        write_csv(result.submission_valid, args.out / "submission_for_test_tiny.csv")
        write_csv(result.submission_invalid_duplicate, args.out / "submission_invalid_duplicate.csv")
        write_manifest(result.manifest, args.manifest)
    except OSError as exc:
        print(
            f"[error] failed to write fixtures: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    print(
        f"[ok] fixtures written to {args.out}/\n"
        f"  - train_tiny.csv ({len(result.train_tiny)} rows)\n"
        f"  - test_tiny.csv ({len(result.test_tiny)} rows)\n"
        f"  - submission_for_test_tiny.csv ({len(result.submission_valid)} rows)\n"
        f"  - submission_invalid_duplicate.csv ({len(result.submission_invalid_duplicate)} rows)\n"
        f"  - {args.manifest} (manifest)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
