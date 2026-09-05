import numpy as np
import pandas as pd
import pytest

from veg_recovery.dl.data import (
    CHANNELS,
    KEY,
    WindowDatasetAdapter,
    WindowPreprocessor,
    canonical_keys,
    labels_for,
    prepare_context,
)
from veg_recovery.dl.fixtures import training_fixture


def setup_frame():
    frame = training_fixture()
    keys = (
        frame.loc[frame.primary_ndvi.notna(), KEY]
        .iloc[[1, 2, 10]]
        .reset_index(drop=True)
    )
    return frame, keys


def test_all_dynamic_fields_hidden_before_overlapping_windows():
    frame, keys = setup_frame()
    frame["future_unknown_feature"] = 100.0
    frame["ndvi_climatology_mean"] = frame.primary_ndvi
    frame["status"] = "critical"
    a = prepare_context(frame, keys)
    modified = frame.copy()
    hidden = pd.MultiIndex.from_frame(modified[KEY]).isin(
        pd.MultiIndex.from_frame(keys)
    )
    modified.loc[hidden, [*CHANNELS, "ndvi_climatology_mean"]] = 999999.0
    b = prepare_context(modified, keys)
    pa = WindowPreprocessor.fit(a, frame[KEY])
    pb = WindowPreprocessor.fit(b, frame[KEY])
    assert pa.fingerprint == pb.fingerprint
    da, db = [
        WindowDatasetAdapter(c, keys, p, window_days=15) for c, p in [(a, pa), (b, pb)]
    ]
    for i in range(len(da)):
        for field in ("features", "values", "observation_mask", "base", "crop_ids"):
            np.testing.assert_array_equal(da[i][field], db[i][field])
    assert a.frame.loc[hidden, "future_unknown_feature"].isna().all()
    assert frame.loc[hidden, "future_unknown_feature"].eq(100).all()


def test_masks_padding_calendar_and_loss():
    frame, _ = setup_frame()
    keys = frame.loc[frame.primary_ndvi.notna(), KEY].iloc[[0]].reset_index(drop=True)
    ctx = prepare_context(frame, keys)
    prep = WindowPreprocessor.fit(ctx, frame[KEY])
    sample = WindowDatasetAdapter(
        ctx, keys, prep, window_days=15, labels=labels_for(frame, keys)
    )[0]
    assert sample["valid_calendar_mask"].sum() == 9
    assert sample["synthetic_gap_mask"][7]
    assert not sample["natural_missing_mask"][7]
    assert sample["natural_missing_mask"][6]
    assert sample["loss_mask"].sum() == 1
    assert sample["loss_mask"][7]
    assert np.isnan(sample["labels"][~sample["loss_mask"]]).all()
    assert not sample["observation_mask"][7].any()
    assert not sample["features"][~sample["valid_calendar_mask"]].any()
    assert np.isnat(sample["dates"][~sample["valid_calendar_mask"]]).all()


def test_scaler_and_categories_use_only_explicit_fit_rows():
    frame, keys = setup_frame()
    ctx = prepare_context(frame, keys)
    fit_keys = frame.loc[frame.anon_polygon_id.eq("TOY-A"), KEY]
    prep = WindowPreprocessor.fit(ctx, fit_keys)
    modified = frame.copy()
    modified.loc[modified.anon_polygon_id.eq("TOY-B"), "era5_temp_c"] = 1e8
    other = WindowPreprocessor.fit(prepare_context(modified, keys), fit_keys)
    assert prep.fingerprint == other.fingerprint
    test_keys = frame.loc[frame.anon_polygon_id.eq("TOY-B"), KEY].iloc[[1]]
    ctx = prepare_context(frame, test_keys)
    assert not WindowDatasetAdapter(ctx, test_keys, prep, window_days=7)[0][
        "crop_ids"
    ].any()
    assert WindowPreprocessor.from_dict(prep.to_dict()).fingerprint == prep.fingerprint


def test_delta_uses_calendar_days_and_no_other_polygon():
    frame, keys = setup_frame()
    frame.loc[frame.anon_polygon_id.eq("TOY-A"), list(CHANNELS)] = np.nan
    ctx = prepare_context(frame, keys)
    prep = WindowPreprocessor.fit(ctx, frame[KEY])
    sample = WindowDatasetAdapter(ctx, keys, prep, window_days=15)[0]
    assert not sample["observation_mask"].any()
    assert sample["delta_since"][7, 0] == 366
    assert sample["delta_until"][7, 0] == 366
    assert sample["base"][7] == pytest.approx(prep.center[0])


def test_invalid_context_flag_and_target_not_clipped():
    frame, keys = setup_frame()
    frame.loc[5, "s2_evi"] = 500
    frame.loc[5, "primary_ndvi"] = 1.2
    ctx = prepare_context(frame, keys)
    prep = WindowPreprocessor.fit(ctx, frame[KEY])
    sample = WindowDatasetAdapter(ctx, keys, prep, window_days=15)[0]
    pos = np.flatnonzero(sample["dates"] == np.datetime64(frame.loc[5, "date"], "D"))[0]
    channel = CHANNELS.index("s2_evi")
    assert sample["invalid_mask"][pos, channel]
    assert not sample["observation_mask"][pos, channel]
    assert sample["observation_mask"][pos, 0]


def test_bad_mask_function_is_rejected():
    frame, keys = setup_frame()
    with pytest.raises(ValueError, match="leaked"):
        prepare_context(frame, keys, apply_mask=lambda f, k: f)


@pytest.mark.parametrize("bad", ["2020/04/01", "2020-04-01T00:00:00Z", "2020-02-30"])
def test_strict_dates(bad):
    with pytest.raises((ValueError, TypeError)):
        canonical_keys(pd.DataFrame({"anon_polygon_id": ["P"], "date": [bad]}))


def test_duplicate_and_absent_keys_rejected():
    frame, keys = setup_frame()
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_context(pd.concat([frame, frame.iloc[[0]]]), keys)
    bad = keys.copy()
    bad["anon_polygon_id"] = "ABSENT"
    with pytest.raises(ValueError, match="absent"):
        prepare_context(frame, bad)


def test_natural_missing_cannot_be_supervised():
    frame, _ = setup_frame()
    keys = frame.iloc[[0]][KEY]
    with pytest.raises(ValueError, match="naturally missing"):
        labels_for(frame, keys)
    ctx = prepare_context(frame, keys)
    prep = WindowPreprocessor.fit(ctx, frame[KEY])
    with pytest.raises(ValueError, match="Natural missing"):
        WindowDatasetAdapter(ctx, keys, prep, labels=keys.assign(primary_ndvi=0.5))


def test_inference_preserves_key_order_and_has_no_loss():
    frame, keys = setup_frame()
    keys = keys.iloc[::-1]
    ctx = prepare_context(frame, keys)
    data = WindowDatasetAdapter(ctx, keys, WindowPreprocessor.fit(ctx, frame[KEY]))
    out = data.predictions_frame([0.1, 0.2, 0.3])
    assert out.date.tolist() == keys.date.tolist()
    assert all(not data[i]["loss_mask"].any() for i in range(len(data)))


def test_supplied_fake_labels_do_not_turn_test_gaps_into_supervision():
    frame, keys = setup_frame()
    frame["is_synthetic_gap"] = False
    hidden = pd.MultiIndex.from_frame(frame[KEY]).isin(pd.MultiIndex.from_frame(keys))
    frame.loc[hidden, "is_synthetic_gap"] = True
    frame.loc[hidden, "primary_ndvi"] = np.nan
    ctx = prepare_context(frame, keys)
    prep = WindowPreprocessor.fit(ctx, frame[KEY])
    with pytest.raises(ValueError, match="hidden test"):
        WindowDatasetAdapter(ctx, keys, prep, labels=keys.assign(primary_ndvi=0.5))


def test_real_test_gap_alignment():
    from pathlib import Path

    path = Path("data/test_data.csv")
    if not path.exists():
        pytest.skip("Competition CSV not available in this checkout")
    frame = pd.read_csv(path)
    gap_keys = frame.loc[frame.is_synthetic_gap, KEY]
    assert len(gap_keys) == 3112
    ctx = prepare_context(frame, gap_keys)
    # Smoke only: не фиттим scaler на test и не вычисляем test RMSE.
    train = training_fixture()
    prep = WindowPreprocessor.fit(
        prepare_context(train, train[KEY].iloc[:0]), train[KEY]
    )
    data = WindowDatasetAdapter(ctx, gap_keys.iloc[:12], prep)
    assert len(data) == 12
    for sample in data:
        assert sample["synthetic_gap_mask"][30]
        assert not sample["observation_mask"][30].any()
        assert not sample["loss_mask"].any()
