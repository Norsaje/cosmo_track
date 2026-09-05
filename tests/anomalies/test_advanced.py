from dataclasses import replace
from datetime import date
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from veg_recovery.anomalies.advanced import (
    AdvancedAnomalyDetector,
    AnomalyConfig,
    SensorHarmonizer,
)
from veg_recovery.anomalies.events import AnomalyEvent
from veg_recovery.anomalies.explain import explain_ru
from veg_recovery.dl.fixtures import anomaly_case, anomaly_reference


@pytest.fixture(scope="module")
def detector():
    return AdvancedAnomalyDetector().fit(anomaly_reference())


@pytest.mark.parametrize(
    "case", ["normal", "source_switch", "single_outlier", "wide_uncertainty"]
)
def test_false_positive_proxies(detector, case):
    frame, _ = anomaly_case(case)
    assert detector.detect(frame).events == ()


def test_pulse_magnitude_severity_and_area_monotonic(detector):
    events = []
    for name in ("mild_pulse", "medium_pulse", "strong_pulse"):
        frame, truth = anomaly_case(name)
        result = detector.detect(frame)
        assert len(result.events) == 1
        event = result.events[0]
        assert (event.start_date, event.end_date) == truth
        events.append(event)
    assert events[0].severity == "biomass_suppression"
    assert events[-1].severity == "critical"
    assert events[0].negative_area < events[1].negative_area < events[2].negative_area
    assert events[0].score < events[1].score < events[2].score


def test_current_year_never_enters_own_climatology():
    query, _ = anomaly_case("strong_pulse")
    ref = pd.concat([anomaly_reference(), query], ignore_index=True)
    a = AdvancedAnomalyDetector().fit(ref).score_points(query)
    ref.loc[ref.date.dt.year.eq(2020), "ndvi_harmonized"] = 1e6
    b = AdvancedAnomalyDetector().fit(ref).score_points(query)
    np.testing.assert_array_equal(a.expected_ndvi, b.expected_ndvi)
    np.testing.assert_array_equal(a.climatology_scale, b.climatology_scale)


def test_no_history_reports_unknown_instead_of_normal():
    query, _ = anomaly_case("strong_pulse")
    detector = AdvancedAnomalyDetector().fit(query)
    result = detector.detect(query)
    assert not result.events
    assert "INSUFFICIENT_REFERENCE_YEARS" in result.warnings


def test_wide_uncertainty_lowers_support(detector):
    query, _ = anomaly_case("strong_pulse")
    narrow = query.copy()
    narrow["is_observed"] = False
    narrow["uncertainty_std"] = 0.01
    wide = narrow.assign(uncertainty_std=0.3)
    a, b = detector.score_points(narrow), detector.score_points(wide)
    assert (a.confidence > b.confidence).all()
    assert not any(e.severity == "critical" for e in detector.detect(narrow).events)
    unknown = narrow.drop(columns="uncertainty_std")
    assert not detector.detect(unknown).events


def test_normal_point_splits_events(detector):
    query, _ = anomaly_case("strong_pulse")
    query.loc[30, "ndvi_harmonized"] += 0.25
    result = detector.detect(query)
    assert len(result.events) == 2
    assert result.events[0].end_date < query.loc[30, "date"].date()
    assert result.events[1].start_date > query.loc[30, "date"].date()


def test_cloud_bad_quality_not_a_critical_event(detector):
    query, _ = anomaly_case("strong_pulse")
    query["quality"] = 0.1
    assert not detector.detect(query).events


def test_contract_json_and_noncausal_explanation(detector):
    query, _ = anomaly_case("strong_pulse")
    result = detector.detect(query)
    document = json.loads(
        json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False)
    )
    event = document["events"][0]
    assert event["schema_version"] == "0.1"
    assert "LOW_PRECIPITATION" in event["reason_codes"]
    assert "HIGH_TEMPERATURE" in event["reason_codes"]
    assert "Причинность не доказана" in event["explanation_ru"]
    assert "вызвала" not in event["explanation_ru"]
    assert "причинность не доказана" in explain_ru(())


def test_invalid_payload_rejected(detector):
    query, _ = anomaly_case("normal")
    with pytest.raises(ValueError, match="Required columns"):
        detector.detect(query.drop(columns="ndvi_harmonized"))
    with pytest.raises(ValueError, match="boolean"):
        detector.detect(query.assign(is_observed="False"))
    with pytest.raises(ValueError, match="nonnegative"):
        detector.detect(query.assign(uncertainty_std=-1))
    with pytest.raises(ValueError, match="exactly one"):
        detector.detect(pd.concat([query, query.assign(anon_polygon_id="ANOTHER")]))
    with pytest.raises(ValueError):
        replace(AnomalyConfig(), mad_floor=0)
    with pytest.raises(ValueError):
        AnomalyEvent(
            date(2020, 1, 1),
            date(2020, 1, 2),
            "critical",
            1,
            np.nan,
            -3,
            1,
            2,
            0,
            (),
            "test",
        )


def test_sensor_mapping_preserves_raw_and_handles_unseen():
    n = 120
    target = np.linspace(0.2, 0.8, n)
    reference = pd.DataFrame(
        {
            "anon_polygon_id": "P",
            "crop_type": "wheat",
            "date": pd.date_range("2019-01-01", periods=n),
            "s2_ndvi": target,
            "modis_ndvi": (target - 0.1) / 0.8,
        }
    )
    reference.loc[0, "s2_ndvi"] = -0.8
    harmonizer = SensorHarmonizer().fit(reference)
    query = pd.DataFrame(
        {
            "anon_polygon_id": ["UNSEEN"] * 3,
            "crop_type": ["unknown"] * 3,
            "date": pd.date_range("2020-01-01", periods=3),
            "primary_ndvi": [0.5, 0.6, 0.7],
            "selected_source": ["modis", "s2", "landsat"],
        }
    )
    out = harmonizer.transform(query)
    assert out.ndvi_harmonized.iloc[0] == pytest.approx(0.5, abs=0.002)
    assert out.ndvi_harmonized.iloc[1] == 0.6
    assert np.isnan(out.ndvi_harmonized.iloc[2])
    assert not out.calibration_supported.iloc[2]
    np.testing.assert_array_equal(out.primary_ndvi_raw, query.primary_ndvi)
    json.dumps(harmonizer.to_dict(), allow_nan=False)
    restored = SensorHarmonizer.from_dict(harmonizer.to_dict())
    pd.testing.assert_frame_equal(restored.transform(query), out)
    uncertainty = query.assign(uncertainty_std=0.1, lower=0.4, upper=0.6)
    corrected = restored.transform(uncertainty)
    assert corrected.uncertainty_std.iloc[0] == pytest.approx(0.08, abs=0.002)
    assert corrected.lower.iloc[0] == pytest.approx(0.42, abs=0.002)
    assert corrected.raw_uncertainty_std.iloc[0] == 0.1


def test_unknown_crop_global_reference_cannot_alone_be_critical(detector):
    query, _ = anomaly_case("strong_pulse")
    result = detector.detect(
        query.assign(crop_type="UNKNOWN", anon_polygon_id="UNSEEN")
    )
    assert result.events
    assert all(event.severity != "critical" for event in result.events)


def test_natural_calendar_gaps_are_not_source_switches_or_reconstructions(detector):
    query, _ = anomaly_case("normal")
    missing = query.index % 3 != 0
    query.loc[missing, "ndvi_harmonized"] = np.nan
    query.loc[missing, "is_observed"] = False
    query.loc[missing, "selected_source"] = "unknown"
    points, result = detector.analyze(query)
    assert not points.source_switch_risk.any()
    assert "LOW_RECONSTRUCTION_SUPPORT" not in result.warnings
    query.loc[query.index >= 36, "selected_source"] = "modis"
    points = detector.score_points(query)
    assert points.source_switch_risk.sum() == 1
    assert points.loc[36, "source_switch_risk"]


def test_climatology_cache_is_reset_on_refit():
    query, _ = anomaly_case("normal")
    detector = AdvancedAnomalyDetector().fit(anomaly_reference())
    first = detector.score_points(query)
    again = detector.score_points(query)
    np.testing.assert_array_equal(first.expected_ndvi, again.expected_ndvi)
    reference = anomaly_reference()
    reference["ndvi_harmonized"] += 0.2
    detector.fit(reference)
    changed = detector.score_points(query)
    np.testing.assert_allclose(changed.expected_ndvi - first.expected_ndvi, 0.2)


def test_events_invariant_to_empty_calendar_rows(detector):
    query, _ = anomaly_case("strong_pulse")
    sparse = query.iloc[::3].copy()
    dense = query.copy()
    missing = ~dense.index.isin(sparse.index)
    dense.loc[missing, "ndvi_harmonized"] = np.nan
    dense.loc[missing, "is_observed"] = False
    dense.loc[missing, "selected_source"] = "unknown"
    a = detector.detect(sparse)
    b = detector.detect(dense)
    assert [event.to_dict() for event in a.events] == [
        event.to_dict() for event in b.events
    ]


def test_core_import_does_not_import_torch():
    code = """
import sys
class BlockTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'pypots'}:
            raise ImportError('forbidden optional dependency')
sys.meta_path.insert(0, BlockTorch())
from veg_recovery.anomalies.advanced import AdvancedAnomalyDetector
from veg_recovery.anomalies.events import AnomalyEvent
from veg_recovery.dl.data import WindowDatasetAdapter
assert 'torch' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
