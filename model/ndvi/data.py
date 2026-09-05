from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["anon_polygon_id", "date"]
SENSORS = ["s2", "landsat", "modis"]
VALUES = ["primary_ndvi", "s2_ndvi", "landsat_ndvi", "modis_ndvi",
          "s2_evi", "s2_ndwi", "landsat_evi", "landsat_ndwi", "modis_evi",
          "era5_temp_c", "era5_precip_mm"]
CROPS = ["зерновые", "озимая пшеница", "пастбища/зерновые", "подсолнечник"]
EXPECTED = {"train": "a75e530d0fb51581ad6800f84b3875233778801491f02236917862faf9b424ec",
            "test": "f7ba087818175ea644fc9fe652c6a3874b5bfcde7df197d3e7541cb95876f855"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_data(train_path, test_path, strict=True):
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    if strict:
        for name, path in [("train", train_path), ("test", test_path)]:
            if sha256(path) != EXPECTED[name]:
                raise ValueError(f"{name}: SHA256 отличается от исследованного набора. "
                                 "Используйте файлы data из архива; для осознанно новых данных --allow-new-data.")
    for name, frame in [("train", train), ("test", test)]:
        if not set(KEYS + VALUES + ["crop_type"]).issubset(frame.columns):
            raise ValueError(f"{name}: неполная схема входных данных")
        frame["date"] = pd.to_datetime(frame.date, errors="raise")
        if frame.duplicated(KEYS).any():
            raise ValueError(f"{name}: повтор ключа polygon/date")
        if np.isinf(frame[VALUES].to_numpy(float)).any():
            raise ValueError(f"{name}: бесконечные значения")
    gap = parse_gap(test.is_synthetic_gap)
    if test.loc[gap, VALUES].notna().any().any():
        raise ValueError("У контрольных пропусков есть динамические значения: проверьте версию test")
    train["dataset"] = "train"
    test["dataset"] = "test"
    train["is_synthetic_gap"] = False
    test["is_synthetic_gap"] = gap
    # Производные организаторов исключены: они могут содержать информацию о скрытой цели.
    df = pd.concat([train, test], ignore_index=True)[KEYS + ["crop_type", "dataset", "is_synthetic_gap"] + VALUES]
    df["year"] = df.date.dt.year
    df["doy"] = df.date.dt.dayofyear
    df["day"] = (df.date - pd.Timestamp("2000-01-01")).dt.days
    df["row_id"] = np.arange(len(df))
    df = df.set_index("row_id", drop=False)
    if df.duplicated(KEYS).any():
        raise ValueError("Train/test содержат один и тот же ключ")
    return df


def parse_gap(s):
    x = s.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if x.isna().any():
        raise ValueError("Некорректный is_synthetic_gap")
    return x.astype(bool)


def source_of(df):
    return np.select([df[f"{s}_ndvi"].notna() for s in SENSORS], [0, 1, 2], default=-1)


def hide(df, ids):
    out = df.copy()
    out.loc[out.index.intersection(ids), VALUES] = np.nan
    return out


def query_only(df):
    # Ни один динамический столбец query не доступен построителю признаков.
    return df[["row_id", *KEYS, "crop_type", "year", "doy", "day"]].copy()


def mask_partition(df, seed, n_parts=7):
    rng = np.random.default_rng(seed)
    out = pd.Series(index=df.index, dtype=int)
    for _, g in df.groupby(["anon_polygon_id", "year"], sort=True):
        ids = rng.permutation(g.index.to_numpy())
        # Начальная фаза случайна, чтобы малые сезоны не попадали всегда в fold 0.
        out.loc[ids] = (np.arange(len(ids)) + rng.integers(n_parts)) % n_parts
    return out.astype(int)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Нефинитные значения при оценке")
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    return {"n": len(y), "rmse": rmse, "mae": float(np.mean(abs(y - p))),
            "gapscore_local": round(30 * max(0, 1 - rmse / .10), 2)}
