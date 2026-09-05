"""Только искусственные данные для offline тестов; не конкурсная CV."""

import numpy as np
import pandas as pd

from .data import (
    CHANNELS,
    KEY,
    WindowDatasetAdapter,
    WindowPreprocessor,
    labels_for,
    prepare_context,
)


def training_fixture():
    frames = []
    for polygon, crop in (("TOY-A", "wheat"), ("TOY-B", "barley")):
        dates = pd.date_range("2020-04-01", periods=61)
        phase = np.arange(len(dates))
        target = 0.4 + 0.15 * np.sin(phase / 15)
        frame = pd.DataFrame(
            {
                "anon_polygon_id": polygon,
                "date": dates,
                "crop_type": crop,
                "primary_ndvi": target,
            }
        )
        for col in CHANNELS[1:]:
            frame[col] = target + 0.01 if "ndvi" in col else 0.2
        frame["era5_temp_c"] = 15 + np.sin(phase / 10)
        frame.loc[phase % 3 != 0, ["s2_ndvi", "s2_evi", "s2_ndwi"]] = np.nan
        frame.loc[phase % 4 == 0, "primary_ndvi"] = np.nan
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def fixture_datasets(window_days=15):
    frame = training_fixture()
    observed = frame.loc[frame.primary_ndvi.notna(), KEY]
    train_keys = observed.iloc[2::7].reset_index(drop=True)
    inner_keys = observed.iloc[4::13].reset_index(drop=True)
    inner_keys = inner_keys.merge(train_keys.assign(_train=True), on=KEY, how="left")
    inner_keys = inner_keys.loc[inner_keys._train.isna(), KEY]
    hidden = pd.concat([train_keys, inner_keys]).drop_duplicates(KEY)
    context = prepare_context(frame, hidden)
    fit_keys = frame[KEY].merge(inner_keys.assign(_inner=True), on=KEY, how="left")
    fit_keys = fit_keys.loc[fit_keys._inner.isna(), KEY]
    prep = WindowPreprocessor.fit(context, fit_keys)
    train = WindowDatasetAdapter(
        context,
        train_keys,
        prep,
        window_days=window_days,
        labels=labels_for(frame, train_keys),
    )
    inner = WindowDatasetAdapter(
        context,
        inner_keys,
        prep,
        window_days=window_days,
        labels=labels_for(frame, inner_keys),
    )
    return train, inner


def anomaly_reference():
    frames = []
    for year, offset in zip(range(2015, 2020), (-0.008, 0.006, 0.0, 0.004, -0.004)):
        dates = pd.date_range(f"{year}-04-01", periods=75, freq="2D")
        values = 0.5 + 0.15 * np.sin(np.arange(len(dates)) / 28) + offset
        frames.append(
            pd.DataFrame(
                {
                    "anon_polygon_id": "TOY-FIELD",
                    "date": dates,
                    "crop_type": "wheat",
                    "ndvi_harmonized": values,
                    "is_observed": True,
                    "quality": 1.0,
                    "selected_source": "s2",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def anomaly_case(name):
    query = (
        anomaly_reference().query("date >= '2019-01-01'").copy().reset_index(drop=True)
    )
    query["date"] = query.date + pd.DateOffset(years=1)
    query["ndvi_harmonized"] += 0.004
    query["primary_ndvi_raw"] = query.ndvi_harmonized
    truth = None
    pulse = query.index.to_series().between(25, 36)
    if name in {"mild_pulse", "medium_pulse", "strong_pulse"}:
        magnitude = {"mild_pulse": 0.065, "medium_pulse": 0.13, "strong_pulse": 0.25}[
            name
        ]
        query.loc[pulse, "ndvi_harmonized"] -= magnitude
        query.loc[pulse, "primary_ndvi_raw"] -= magnitude
        query.loc[pulse, "precip_30d_ratio"] = 0.3
        query.loc[pulse, "temp_anomaly_c"] = 4.5
        query.loc[pulse, "sensor_negative_count"] = 2
        truth = (
            query.loc[pulse, "date"].min().date(),
            query.loc[pulse, "date"].max().date(),
        )
    elif name == "source_switch":
        query.loc[query.index >= 25, "selected_source"] = "modis"
        query.loc[query.index >= 25, "primary_ndvi_raw"] -= 0.3
    elif name == "single_outlier":
        query.loc[30, "ndvi_harmonized"] -= 0.45
    elif name == "wide_uncertainty":
        query.loc[pulse, "ndvi_harmonized"] -= 0.15
        query.loc[pulse, "is_observed"] = False
        query.loc[pulse, "uncertainty_std"] = 0.4
    elif name != "normal":
        raise ValueError(f"Unknown synthetic case: {name}")
    return query, truth
