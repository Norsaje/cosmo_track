"""Строгое сравнение OOF: никаких inner joins с молчаливой потерей ключей."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import canonical_keys

OOF_KEY = ["split", "fold", "anon_polygon_id", "date"]
COMPOSITE_WEIGHTS = {"matched": 0.5, "unseen": 0.25, "temporal": 0.15, "hard": 0.1}


def rmse(y_true, y_pred) -> float:
    truth, pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if truth.ndim != 1 or truth.size == 0 or truth.shape != pred.shape:
        raise ValueError("RMSE requires nonempty aligned 1D arrays")
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise ValueError("RMSE rejects missing/infinite values")
    return float(np.sqrt(np.mean(np.square(truth - pred))))


def gap_score(error: float) -> float:
    if not np.isfinite(error) or error < 0:
        raise ValueError("RMSE must be finite and nonnegative")
    return round(30 * max(0.0, 1 - error / 0.10), 2)


def validate_oof(frame: pd.DataFrame) -> pd.DataFrame:
    if not set(OOF_KEY + ["y_true", "primary_ndvi_pred"]).issubset(frame):
        raise ValueError("OOF needs split/fold/polygon/date/y_true/primary_ndvi_pred")
    out = frame.copy().reset_index(drop=True)
    if out.empty or out[OOF_KEY].isna().any().any():
        raise ValueError("Empty OOF or missing OOF keys")
    if out.duplicated(OOF_KEY).any():
        raise ValueError("Duplicate OOF keys")
    for _, group in out.groupby(["split", "fold"], sort=False):
        parsed = canonical_keys(group)
        out.loc[group.index, "date"] = parsed.date.to_numpy()
    out["date"] = pd.to_datetime(out.date)
    for col in ["y_true", "primary_ndvi_pred"]:
        out[col] = pd.to_numeric(out[col], errors="raise")
        if not np.isfinite(out[col]).all():
            raise ValueError(f"Nonfinite OOF {col}")
    return out


def align_oof(reference: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    left, right = validate_oof(reference), validate_oof(candidate)
    lk, rk = (
        pd.MultiIndex.from_frame(left[OOF_KEY]),
        pd.MultiIndex.from_frame(right[OOF_KEY]),
    )
    if len(lk) != len(rk) or not lk.isin(rk).all():
        raise ValueError("Candidate OOF key set differs from immutable ML baseline")
    right = right.set_index(OOF_KEY).loc[lk]
    if not np.allclose(left.y_true, right.y_true, rtol=0, atol=1e-12):
        raise ValueError("Candidate labels differ from ML OOF labels")
    return left.assign(candidate_pred=right.primary_ndvi_pred.to_numpy())


def metrics(frame: pd.DataFrame, prediction: str = "primary_ndvi_pred") -> dict:
    result = {"overall_rmse": rmse(frame.y_true, frame[prediction])}
    result["gap_score"] = gap_score(result["overall_rmse"])
    splits = {str(k): rmse(g.y_true, g[prediction]) for k, g in frame.groupby("split")}
    result["split_rmse"] = splits
    # Отсутствующий CV не получает вес 0: composite пока не определён.
    result["composite_rmse"] = (
        sum(COMPOSITE_WEIGHTS[k] * splits[k] for k in COMPOSITE_WEIGHTS)
        if set(COMPOSITE_WEIGHTS).issubset(splits)
        else None
    )
    subgroups = {}
    for field in ("is_unseen", "is_hard", "source", "gap_length", "crop_type", "fold"):
        if field in frame:
            subgroups[field] = {
                str(k): {"n": len(g), "rmse": rmse(g.y_true, g[prediction])}
                for k, g in frame.groupby(field, dropna=False)
            }
    result["subgroups"] = subgroups
    return result


def assert_metric_equality(
    baseline: pd.DataFrame, expected: dict, *, atol: float = 1e-10
):
    actual = metrics(validate_oof(baseline))
    if not {"overall_rmse", "gap_score"}.issubset(expected):
        raise ValueError("ML handoff must publish RMSE and GapScore for equality check")
    for key in ("overall_rmse", "gap_score"):
        if not np.isclose(actual[key], expected[key], rtol=0, atol=atol):
            raise ValueError(
                f"ML metric equality failed for {key}: {actual[key]} != {expected[key]}"
            )
    return actual


def paired_polygon_bootstrap(
    aligned: pd.DataFrame, *, repeats: int = 2000, seed: int = 42
) -> dict:
    if repeats < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    polygons = aligned.anon_polygon_id.unique()
    if len(polygons) < 2:
        raise ValueError("Polygon bootstrap needs >= 2 independent polygons")
    grouped = []
    for _, group in aligned.groupby("anon_polygon_id", sort=False):
        grouped.append(
            (
                np.square(group.y_true - group.primary_ndvi_pred).sum(),
                np.square(group.y_true - group.candidate_pred).sum(),
                len(group),
            )
        )
    totals = np.asarray(grouped, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(totals), size=(repeats, len(totals)))
    sums = totals[draws].sum(axis=1)
    gain = np.sqrt(sums[:, 0] / sums[:, 2]) - np.sqrt(sums[:, 1] / sums[:, 2])
    low, high = np.quantile(gain, [0.025, 0.975])
    return {
        "unit": "polygon",
        "repeats": repeats,
        "seed": seed,
        "gain_rmse": rmse(aligned.y_true, aligned.primary_ndvi_pred)
        - rmse(aligned.y_true, aligned.candidate_pred),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def adoption_decision(
    seed_results: list[dict], *, integration_passed: bool = False
) -> dict:
    """Fail-closed gate; uncertainty adoption требует отдельного отчёта calibration."""
    if len({r.get("seed") for r in seed_results}) < 3:
        return {
            "decision": "PENDING_EVALUATION",
            "reason": "At least three distinct seeds required",
        }
    required = {
        "ml_composite",
        "dl_composite",
        "ml_unseen",
        "dl_unseen",
        "subgroups_passed",
        "seed",
    }
    if any(not required.issubset(r) for r in seed_results):
        return {
            "decision": "PENDING_EVALUATION",
            "reason": "Missing CV/subgroup evidence",
        }
    for result in seed_results:
        numeric = [
            result[k]
            for k in ("ml_composite", "dl_composite", "ml_unseen", "dl_unseen")
        ]
        if any(v is None or not np.isfinite(v) or v < 0 for v in numeric):
            return {"decision": "PENDING_EVALUATION", "reason": "Invalid CV evidence"}
    passes = [
        r["ml_composite"] - r["dl_composite"] >= 0.002
        and r["dl_unseen"] - r["ml_unseen"] <= 0.003
        and r["subgroups_passed"] is True
        for r in seed_results
    ]
    if all(passes):
        return {
            "decision": "ADOPT" if integration_passed else "PENDING_INTEGRATION",
            "reason": "Point gate passed on all seeds; integration, manifest and latency also required",
        }
    # Вес ансамбля должен быть выбран на отдельной calibration части, не оценочных OOF labels.
    blend_pass = all(
        r.get("blend_evaluated_out_of_sample") is True
        and r.get("blend_gain", -np.inf) >= 0.001
        and r.get("blend_unseen_degradation", np.inf) <= 0.003
        and r.get("subgroups_passed") is True
        for r in seed_results
    )
    if blend_pass:
        return {
            "decision": "ENSEMBLE_ONLY"
            if integration_passed
            else "PENDING_INTEGRATION",
            "reason": "Independent ensemble gate passed",
        }
    return {
        "decision": "REJECT",
        "reason": "Point and independent ensemble gates not met",
    }
