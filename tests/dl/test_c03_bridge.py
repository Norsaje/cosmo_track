"""Мост к опубликованному C-03 и сезонный prior: только offline fixtures."""

import numpy as np
import pandas as pd
import pytest

from veg_recovery.dl.c03_bridge import censored_keys, training_targets
from veg_recovery.dl.data import (
    KEY,
    SeasonalPrior,
    WindowDatasetAdapter,
    WindowPreprocessor,
    key_index,
    prepare_context,
)
from veg_recovery.dl.fixtures import training_fixture

MASKED = ["primary_ndvi", "s2_ndvi", "era5_temp_c"]


def test_censored_keys_ignore_natural_missing_and_find_producer_censoring():
    raw = training_fixture()
    context = raw.copy(deep=True)
    hidden = raw.index[[3, 40, 77]]
    context.loc[hidden, MASKED] = np.nan
    found = censored_keys(raw, context, MASKED)
    assert len(found) == 3
    assert key_index(found).equals(key_index(raw.loc[hidden]))
    # Естественно отсутствующий target уже был NaN и не считается censoring.
    assert not raw.primary_ndvi.isna().sum() == 0
    assert censored_keys(raw, raw.copy(deep=True), MASKED).empty


def test_training_targets_stay_inside_fit_and_stay_disjoint():
    fit = training_fixture()
    train, inner = training_targets(
        fit, seed=17, blocks=3, inner_fraction=0.1, max_targets=40
    )
    assert set(train.block) == {0, 1, 2}
    assert not key_index(train).isin(key_index(inner)).any()
    allowed = fit.loc[fit.primary_ndvi.notna()]
    assert key_index(train).isin(key_index(allowed)).all()
    assert key_index(inner).isin(key_index(allowed)).all()
    assert len(train) <= 40
    repeat, _ = training_targets(
        fit, seed=17, blocks=3, inner_fraction=0.1, max_targets=40
    )
    pd.testing.assert_frame_equal(train, repeat)


def test_training_targets_reject_invalid_configuration():
    with pytest.raises(ValueError):
        training_targets(
            training_fixture(), seed=1, blocks=3, inner_fraction=0.9, max_targets=40
        )


def _prior_frame():
    rows = []
    for year in range(2018, 2023):
        for day in range(1, 300, 4):
            date = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=day - 1)
            for polygon, crop, level in (("P1", "wheat", 0.6), ("P2", "wheat", 0.3)):
                rows.append(
                    {
                        "anon_polygon_id": polygon,
                        "date": date,
                        "crop_type": crop,
                        "primary_ndvi": level + 0.1 * np.sin(day / 40),
                    }
                )
    return pd.DataFrame(rows)


def test_prior_is_polygon_specific_and_never_reads_a_hidden_value():
    frame = _prior_frame()
    hidden = frame.loc[frame.anon_polygon_id.eq("P1")].iloc[[20]][KEY]
    context = prepare_context(frame, hidden)
    prior = SeasonalPrior.fit(context, frame[KEY])
    doys = np.array([frame.loc[20, "date"].dayofyear])
    p1, level1 = prior.series("P1", "wheat", doys)
    p2, _ = prior.series("P2", "wheat", doys)
    assert level1[0] == 2.0
    assert abs(float(p1[0]) - float(p2[0])) > 0.2
    # Скрытая строка удалена из контекста, поэтому не может задать собственный prior.
    poisoned = frame.copy()
    poisoned.loc[20, "primary_ndvi"] = 5.0
    poisoned_prior = SeasonalPrior.fit(prepare_context(poisoned, hidden), frame[KEY])
    np.testing.assert_allclose(
        poisoned_prior.series("P1", "wheat", doys)[0], p1, rtol=0, atol=1e-12
    )
    assert poisoned_prior.fingerprint == prior.fingerprint


def test_prior_falls_back_to_crop_then_global_for_unknown_groups():
    frame = _prior_frame()
    context = prepare_context(frame, frame[KEY].iloc[:0])
    prior = SeasonalPrior.fit(context, frame[KEY])
    doys = np.array([120, 200])
    _, unseen_polygon = prior.series("NEW", "wheat", doys)
    _, unseen_crop = prior.series("NEW", "NEW_CROP", doys)
    assert list(unseen_polygon) == [1.0, 1.0]
    assert list(unseen_crop) == [0.0, 0.0]
    assert np.isfinite(prior.series("NEW", "NEW_CROP", np.arange(1, 367))[0]).all()


def test_anchored_base_uses_prior_when_the_polygon_year_has_no_observation():
    frame = _prior_frame()
    forecast_year = frame.date.dt.year.eq(2022)
    hidden = frame.loc[forecast_year & frame.anon_polygon_id.eq("P1")][KEY]
    context = prepare_context(frame, hidden)
    fit_keys = frame.loc[~key_index(frame).isin(key_index(hidden))][KEY]
    prep = WindowPreprocessor.fit(context, fit_keys, channels=("primary_ndvi",))
    prior = SeasonalPrior.fit(context, fit_keys)
    targets = hidden.iloc[[10]].reset_index(drop=True)
    dataset = WindowDatasetAdapter(
        context, targets, prep, window_days=31, prior=prior, base_mode="anchored"
    )
    item = dataset[0]
    center = dataset.center_index
    expected = prior.series(
        "P1", "wheat", np.array([targets.date.iloc[0].dayofyear])
    )[0][0]
    assert not item["observation_mask"][center].any()
    np.testing.assert_allclose(item["base"][center], expected, rtol=0, atol=1e-6)
    assert np.isfinite(item["features"]).all()
