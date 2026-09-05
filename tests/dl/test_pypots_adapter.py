import numpy as np
import pytest

from veg_recovery.dl.fixtures import fixture_datasets
from veg_recovery.dl.models.pypots import PyPOTSInferenceAdapter, to_pypots_arrays


def test_only_target_is_revealed_to_validation_metric():
    _, validation = fixture_datasets()
    arrays = to_pypots_arrays(validation, include_validation_labels=True)
    indicating = np.isfinite(arrays.X_ori) & ~np.isfinite(arrays.X)
    assert indicating.sum() == len(validation)
    assert indicating[:, validation.center_index, 0].all()
    assert np.isnan(arrays.X[:, validation.center_index]).all()
    assert np.isfinite(arrays.X_ori[:, validation.center_index, 0]).all()
    for i in range(len(validation)):
        assert np.isnan(arrays.X_ori[i, ~validation[i]["valid_calendar_mask"]]).all()


def test_public_impute_output_alignment():
    _, dataset = fixture_datasets()

    class PublicAPIFixture:
        def impute(self, inputs):
            assert set(inputs) == {"X"}
            output = np.zeros_like(inputs["X"])
            output[:, dataset.center_index, 0] = np.arange(len(dataset))
            return output

    out = PyPOTSInferenceAdapter(PublicAPIFixture(), model_name="saits").predict(
        dataset
    )
    expected = (
        np.arange(len(dataset)) * dataset.preprocessor.scale[0]
        + dataset.preprocessor.center[0]
    )
    np.testing.assert_allclose(out.primary_ndvi_pred, expected)
    assert out.date.tolist() == dataset.keys.date.tolist()


def test_malformed_output_is_not_silently_flattened():
    _, dataset = fixture_datasets()

    class WrongShape:
        def impute(self, inputs):
            return np.zeros(10)

    with pytest.raises(ValueError, match="shape"):
        PyPOTSInferenceAdapter(WrongShape(), model_name="brits").predict(dataset)
