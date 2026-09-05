import numpy as np
import pandas as pd
import pytest

from veg_recovery.dl.evaluation import (
    adoption_decision,
    align_oof,
    assert_metric_equality,
    gap_score,
    metrics,
    paired_polygon_bootstrap,
    rmse,
    validate_oof,
)


def example():
    return pd.DataFrame(
        {
            "split": ["matched"] * 4,
            "fold": ["f0"] * 4,
            "anon_polygon_id": ["P", "P", "Q", "Q"],
            "date": ["2020-01-01", "2020-01-02"] * 2,
            "y_true": [0.5] * 4,
            "primary_ndvi_pred": [0.6] * 4,
        }
    )


def test_metrics_exact_and_incomplete_composite_unknown():
    frame = example()
    result = assert_metric_equality(frame, {"overall_rmse": 0.1, "gap_score": 0})
    assert result["composite_rmse"] is None
    assert gap_score(0.05) == 15
    with pytest.raises(ValueError, match="equality"):
        assert_metric_equality(frame, {"overall_rmse": 0.06, "gap_score": 12})
    full = pd.concat(
        [frame.assign(split=s) for s in ["matched", "unseen", "temporal", "hard"]]
    )
    assert metrics(validate_oof(full))["composite_rmse"] == pytest.approx(0.1)


def test_alignment_key_order_and_block_bootstrap():
    baseline = example()
    candidate = baseline.iloc[::-1].assign(primary_ndvi_pred=0.55)
    aligned = align_oof(baseline, candidate)
    result = paired_polygon_bootstrap(aligned, repeats=100)
    assert result["unit"] == "polygon"
    assert result["gain_rmse"] == pytest.approx(0.05)
    assert result["ci95_low"] == pytest.approx(0.05)
    assert result == paired_polygon_bootstrap(aligned, repeats=100)


def test_oof_rejects_missing_extra_duplicate_or_changed_labels():
    baseline = example()
    for bad in [
        baseline.iloc[:3],
        pd.concat([baseline, baseline.iloc[:1]]),
        baseline.assign(y_true=0.7),
        baseline.assign(primary_ndvi_pred=np.nan),
    ]:
        with pytest.raises(ValueError):
            align_oof(baseline, bad)
    with pytest.raises(ValueError):
        rmse([], [])


def test_gate_never_adopts_without_evidence():
    assert adoption_decision([])["decision"] == "PENDING_EVALUATION"
    evidence = [
        {
            "seed": s,
            "ml_composite": 0.06,
            "dl_composite": 0.057,
            "ml_unseen": 0.07,
            "dl_unseen": 0.071,
            "subgroups_passed": True,
        }
        for s in (17, 42, 73)
    ]
    assert adoption_decision(evidence)["decision"] == "PENDING_INTEGRATION"
    assert adoption_decision(evidence, integration_passed=True)["decision"] == "ADOPT"
    evidence[0]["dl_unseen"] = 0.1
    assert adoption_decision(evidence, integration_passed=True)["decision"] == "REJECT"
    evidence[0]["ml_composite"] = None
    assert adoption_decision(evidence)["decision"] == "PENDING_EVALUATION"
