"""Отчётность: бленд выбирается вне оцениваемого фолда, gate остаётся fail-closed."""

import numpy as np
import pandas as pd
import pytest

from veg_recovery.dl.report import blend_out_of_sample, gate_inputs, markdown

SPLITS = ("matched", "unseen", "temporal", "hard")


def _aligned(seed=0, n=60):
    rng = np.random.default_rng(seed)
    rows = []
    for split in SPLITS:
        for fold in ("f0", "f1"):
            truth = rng.normal(0.5, 0.1, n)
            rows.append(
                pd.DataFrame(
                    {
                        "split": split,
                        "fold": fold,
                        "anon_polygon_id": [f"P{i % 7}" for i in range(n)],
                        "date": pd.date_range("2020-04-01", periods=n),
                        "y_true": truth,
                        # ML шумит сильнее, кандидат точнее: оптимальный вес близок к 1.
                        "primary_ndvi_pred": truth + rng.normal(0, 0.05, n),
                        "candidate_pred": truth + rng.normal(0, 0.02, n),
                        "is_unseen": True,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def test_blend_weight_is_chosen_outside_the_scored_fold():
    aligned = _aligned()
    result = blend_out_of_sample(aligned)
    assert set(result["weights"]) == {f"{s}/{f}" for s in SPLITS for f in ("f0", "f1")}
    assert result["mean_weight"] > 0.7
    assert result["composite_rmse"] is not None
    # Ошибки слагаемых независимы, поэтому бленд бьёт оба, а не только слабое.
    ml = float(np.sqrt(np.mean((aligned.y_true - aligned.primary_ndvi_pred) ** 2)))
    candidate = float(np.sqrt(np.mean((aligned.y_true - aligned.candidate_pred) ** 2)))
    assert result["overall_rmse"] < candidate < ml


def test_blend_needs_more_than_one_fold():
    single = _aligned().query("split == 'matched' and fold == 'f0'")
    with pytest.raises(ValueError):
        blend_out_of_sample(single)


def _summary(dl_composite, dl_unseen, seeds=(17, 42, 73)):
    def result(composite, unseen, splits):
        return {
            "overall_rmse": composite,
            "gap_score": 0.0,
            "composite_rmse": composite,
            "split_rmse": splits,
            "subgroups": {"is_unseen": {"True": {"n": 10, "rmse": unseen}}},
        }

    ml_splits = dict.fromkeys(SPLITS, 0.1)
    dl_splits = dict.fromkeys(SPLITS, 0.09)
    return {
        "seeds": [
            {
                "seed": seed,
                "ml": result(0.1, 0.11, ml_splits),
                "dl": result(dl_composite, dl_unseen, dl_splits),
                "base": result(0.095, 0.1, dl_splits),
                "bootstrap": {
                    "gain_rmse": 0.01,
                    "ci95_low": 0.005,
                    "ci95_high": 0.015,
                },
            }
            for seed in seeds
        ]
    }


def test_gate_inputs_flag_a_collapsed_subgroup():
    summary = _summary(0.09, 0.11)
    summary["seeds"][0]["dl"]["split_rmse"] = {**dict.fromkeys(SPLITS, 0.09), "hard": 0.2}
    rows = gate_inputs(summary)
    assert rows[0]["subgroups_passed"] is False
    assert all(r["subgroups_passed"] for r in rows[1:])


def test_gate_inputs_carry_composite_and_unseen():
    rows = gate_inputs(_summary(0.09, 0.108))
    assert [r["seed"] for r in rows] == [17, 42, 73]
    assert all(r["ml_composite"] == 0.1 and r["dl_composite"] == 0.09 for r in rows)
    assert all(r["dl_unseen"] == 0.108 for r in rows)
    assert "blend_gain" not in rows[0]


def test_markdown_reports_every_mode_and_seed():
    text = markdown(_summary(0.09, 0.108))
    assert text.count("|---") >= 2
    for label in ("A matched-mask", "B unseen-polygon", "C past-only", "D hard one-sided"):
        assert label in text
    assert "Composite" in text and "GapScore" in text
    for seed in (17, 42, 73):
        assert f"| {seed} |" in text
