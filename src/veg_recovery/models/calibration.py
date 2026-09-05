"""OOF-only blend, conservative gating and empirical interval choices."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

COMPOSITE_WEIGHTS = {"A": 0.50, "B": 0.25, "C": 0.15, "D": 0.10}


def source_reliability_table(source_oof):
    """Reliability by class and confidence bin, using only held-out source labels."""
    records=[]
    for mode,group in source_oof.groupby('mode'):
        for source in ('s2','landsat','modis','unknown'):
            probability=group['p_'+source]
            bins=pd.cut(probability,[-.001,.2,.4,.6,.8,1.],include_lowest=True)
            for label,part in group.groupby(bins,observed=True):
                records.append(dict(mode=mode,source=source,confidence_bin=str(label),n=len(part),
                                    mean_probability=float(part['p_'+source].mean()),
                                    observed_frequency=float(part.source_label.eq(source).mean())))
    return pd.DataFrame(records)


def rmse(truth, prediction):
    return float(np.sqrt(np.mean((np.asarray(truth) - np.asarray(prediction)) ** 2)))


def composite_rmse(oof, prediction):
    prediction = np.asarray(prediction)
    mode_scores = {}
    for mode in COMPOSITE_WEIGHTS:
        selection = oof["mode"].eq(mode).to_numpy()
        if selection.any():
            mode_scores[mode] = rmse(oof.loc[selection, "primary_ndvi"], prediction[selection])
    if set(mode_scores) != set(COMPOSITE_WEIGHTS):
        raise ValueError("Model selection requires every CV mode A/B/C/D")
    return sum(COMPOSITE_WEIGHTS[mode] * score for mode, score in mode_scores.items())


def select_blend(oof, members):
    """Constrained composite RMSE search, always retain best feasible member."""
    from scipy.optimize import minimize
    matrix = oof[[f"pred_{name}" for name in members]].to_numpy(float)
    if not np.isfinite(matrix).all():
        raise ValueError("OOF predictions must all be finite")
    objective = lambda weights: composite_rmse(oof, matrix @ weights)
    candidates = [(f"single:{name}", np.eye(len(members))[i]) for i, name in enumerate(members)]
    if "hgb" in members and "extra_trees" in members:
        weights = np.zeros(len(members))
        weights[members.index("hgb")] = weights[members.index("extra_trees")] = 0.5
        candidates.append(("hgb_extra_trees_50_50", weights))
    fit = minimize(objective, np.full(len(members), 1 / len(members)), method="SLSQP",
                   bounds=[(0.0, 1.0)] * len(members),
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
                   options={"maxiter": 300, "ftol": 1e-10})
    if fit.success and np.isfinite(fit.x).all():
        weights = np.maximum(fit.x, 0)
        candidates.append(("optimized_nonnegative", weights / weights.sum()))
    selected_name, selected = min(candidates, key=lambda item: objective(item[1]))
    comparisons = [{"candidate": name, "composite_rmse": objective(weights),
                    "weights": dict(zip(members, map(float, weights)))} for name, weights in candidates]
    return dict(zip(members, map(float, selected))), comparisons, matrix @ selected, selected_name


def _values(frame, names, default=np.nan):
    for name in names:
        if name in frame:
            return pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
    return np.full(len(frame), default, dtype=float)


def _gate_mask(features, diagnostics, gate):
    n = len(features)
    if not gate:
        return np.zeros(n, dtype=bool)
    left = _values(features, ["left_days_1", "prev_days_1", "left_distance_days"])
    right = _values(features, ["right_days_1", "next_days_1", "right_distance_days"])
    if gate["kind"] == "short_two_sided":
        return (left <= 15) & (right <= 15)
    if gate["kind"] == "long_one_sided":
        return ~np.isfinite(left) | ~np.isfinite(right) | (np.minimum(left, right) > 15)
    if gate["kind"] == "low_source_confidence":
        return np.max(np.column_stack([diagnostics[f"p_{label}"] for label in ("s2", "landsat", "modis")]), axis=1) < gate["threshold"]
    if gate["kind"] == "unseen":
        return _values(features, ["seen_polygon"], 0) == 0
    if gate["kind"] == "high_disagreement":
        return np.asarray(diagnostics["model_disagreement"]) >= gate["threshold"]
    if gate["kind"] == "low_context_quality":
        return _values(features, ["context_quality"], 0) <= gate["threshold"]
    raise ValueError(f"Unknown gate: {gate['kind']}")


def apply_gate(prediction, baseline, features, diagnostics, gate):
    mask = _gate_mask(features, diagnostics, gate)
    out = np.asarray(prediction, dtype=float).copy()
    if gate:
        weight = float(gate.get("baseline_weight", 0.5))
        out[mask] = (1 - weight) * out[mask] + weight * np.asarray(baseline)[mask]
    return out, mask


def select_gate(oof, prediction):
    diagnostics = {c: oof[c].to_numpy() for c in oof if c.startswith("p_") or c == "model_disagreement"}
    candidates = [{"kind": name, "baseline_weight": 0.5} for name in
                  ("short_two_sided", "long_one_sided", "unseen")]
    candidates += [{"kind": "low_source_confidence", "threshold": 0.65, "baseline_weight": 0.5},
                   {"kind": "high_disagreement", "threshold": float(oof.model_disagreement.quantile(0.8)), "baseline_weight": 0.5},
                   {"kind": "low_context_quality", "threshold": float(np.nanquantile(_values(oof, ["context_quality"], 0), 0.2)), "baseline_weight": 0.5}]
    unseen = oof["seen_polygon"].eq(0).to_numpy()
    baseline_score = composite_rmse(oof, prediction)
    experiments = []
    best = (baseline_score, None, np.asarray(prediction))
    for gate in candidates:
        proposed, selected = apply_gate(prediction, oof["pred_baseline"], oof, diagnostics, gate)
        improvements = []
        for _, group in oof.groupby(["mode", "repeat"]):
            idx = group.index.to_numpy()
            improvements.append(rmse(group.primary_ndvi, proposed[idx]) - rmse(group.primary_ndvi, np.asarray(prediction)[idx]))
        unseen_delta = (rmse(oof.loc[unseen, "primary_ndvi"], proposed[unseen]) -
                        rmse(oof.loc[unseen, "primary_ndvi"], np.asarray(prediction)[unseen])) if unseen.any() else 0.0
        score = composite_rmse(oof, proposed)
        accepted = bool(selected.any() and np.median(improvements) < 0 and unseen_delta <= 0.003 and score < baseline_score)
        experiments.append({"gate": gate, "composite_rmse": score, "median_repeat_delta": float(np.median(improvements)),
                            "unseen_rmse_delta": unseen_delta, "accepted": accepted,
                            "reason": "OOF median improves; unseen degradation <=0.003" if accepted else "OOF acceptance criteria failed"})
        if accepted and score < best[0]:
            best = (score, gate, proposed)
    return best[1], experiments, best[2]


def _bin_labels(features, diagnostics):
    left = _values(features, ["left_days_1", "prev_days_1", "left_distance_days"])
    right = _values(features, ["right_days_1", "next_days_1", "right_distance_days"])
    context = np.where(np.isfinite(left) & np.isfinite(right) & (np.maximum(left, right) <= 15), "short_two_sided", "hard_context")
    probs = np.column_stack([diagnostics.get(f"p_{label}", np.zeros(len(features))) for label in ("s2", "landsat", "modis")])
    source = np.asarray(["s2", "landsat", "modis"])[np.argmax(probs, axis=1)]
    source = np.where(np.max(probs, axis=1) >= 0.65, source, "uncertain")
    return np.asarray([f"{c}:{s}" for c, s in zip(context, source)])


def _quantile(errors, level):
    count = len(errors)
    adjusted = min(1.0, math.ceil((count + 1) * level) / count)
    return float(np.quantile(errors, adjusted, method="higher"))


def fit_uncertainty(oof, prediction, min_bin_size=60):
    errors = np.abs(oof.primary_ndvi.to_numpy(float) - np.asarray(prediction))
    if not len(errors) or not np.isfinite(errors).all():
        raise ValueError("Finite OOF residuals required for empirical intervals")
    labels = _bin_labels(oof, {c: oof[c].to_numpy() for c in oof if c.startswith("p_")})
    result = {"method": "empirical_oof_absolute_residual", "levels": [0.8, 0.95],
              "min_bin_size": min_bin_size, "calibrated": False, "n": len(errors), "bins": {},
              "global": {str(level): _quantile(errors, level) for level in (0.8, 0.95)},
              "assessment": "Empirical OOF intervals; inspect cross-fold subgroup coverage before any calibration claim."}
    for label in sorted(set(labels)):
        selection = labels == label
        if selection.sum() >= min_bin_size:
            result["bins"][label] = {str(level): _quantile(errors[selection], level) for level in (0.8, 0.95)}
    return result


def interval_bounds(prediction, features, diagnostics, config, level=0.95):
    config = config.get("uncertainty", config)
    if config.get("method") != "empirical_oof_absolute_residual":
        raise ValueError("No empirical OOF interval calibration in bundle")
    key = str(float(level))
    if key not in config["global"]:
        raise ValueError(f"Unsupported interval level: {level}")
    labels = _bin_labels(features, diagnostics)
    radius = np.asarray([config["bins"].get(label, config["global"])[key] for label in labels])
    return np.asarray(prediction) - radius, np.asarray(prediction) + radius


def evaluate_uncertainty(oof, prediction):
    """Leave-fold calibration also removes repeated physical keys from residual pool."""
    records = []
    for (mode, repeat, fold), group in oof.groupby(["mode", "repeat", "fold"]):
        held_keys = pd.MultiIndex.from_frame(group[["anon_polygon_id", "date"]])
        all_keys = pd.MultiIndex.from_frame(oof[["anon_polygon_id", "date"]])
        calibration = ~all_keys.isin(held_keys)
        if calibration.sum() < 60:
            continue
        config = fit_uncertainty(oof.loc[calibration], np.asarray(prediction)[calibration])
        indices = group.index.to_numpy()
        diag = {c: group[c].to_numpy() for c in group if c.startswith("p_")}
        for level in (0.8, 0.95):
            lower, upper = interval_bounds(np.asarray(prediction)[indices], group, diag, config, level)
            covered = (group.primary_ndvi.to_numpy() >= lower) & (group.primary_ndvi.to_numpy() <= upper)
            for name, selection in {"overall": np.ones(len(group), bool), "unseen": group.seen_polygon.eq(0).to_numpy(),
                                    "hard": np.full(len(group), mode == "D")}.items():
                if selection.any():
                    records.append(dict(mode=mode, repeat=int(repeat), fold=int(fold), subgroup=name, level=level,
                                        n=int(selection.sum()), coverage=float(covered[selection].mean()),
                                        mean_width=float((upper - lower)[selection].mean())))
    return pd.DataFrame(records)
