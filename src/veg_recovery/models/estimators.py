"""Tabular estimators. Calling factories never starts training."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

KEY_COLUMNS = {"anon_polygon_id", "date"}
SOURCE_CLASSES = ("s2", "landsat", "modis", "unknown")


def _preprocessor(features):
    columns = [c for c in features if c != "date"]
    numeric = [c for c in columns if pd.api.types.is_numeric_dtype(features[c])]
    categorical = [c for c in columns if c not in numeric]
    return ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), numeric),
        ("categorical", Pipeline([
            ("missing", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                       encoded_missing_value=-2)),
        ]), categorical),
    ], remainder="drop", verbose_feature_names_out=False)


def make_regressor(name, features, seed=17, params=None, n_jobs=-1, device="cpu"):
    params = params or {}
    if device not in {"cpu", "gpu"}:
        raise ValueError(f"Unknown training device: {device}")
    if name == "hgb":
        if device != "cpu":
            raise ValueError("HistGradientBoostingRegressor does not support GPU training")
        defaults = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                        l2_regularization=2.0, min_samples_leaf=30, early_stopping=False)
        estimator = HistGradientBoostingRegressor(random_state=seed, **(defaults | params))
    elif name == "extra_trees":
        if device != "cpu":
            raise ValueError("ExtraTreesRegressor does not support GPU training")
        defaults = dict(n_estimators=400, min_samples_leaf=3, max_features=0.85)
        estimator = ExtraTreesRegressor(random_state=seed, n_jobs=n_jobs, **(defaults | params))
    elif name in {"catboost", "lightgbm"}:
        return NativeCategoricalRegressor(name=name, seed=seed, params=params, n_jobs=n_jobs, device=device)
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("prepare", _preprocessor(features)), ("model", estimator)])


def make_source_classifier(features, seed=17, n_jobs=-1):
    return Pipeline([
        ("prepare", _preprocessor(features)),
        ("model", RandomForestClassifier(n_estimators=200, min_samples_leaf=8,
                                          class_weight="balanced_subsample", random_state=seed, n_jobs=n_jobs)),
    ])


def source_probabilities(estimator, features):
    out = np.zeros((len(features), len(SOURCE_CLASSES)), dtype=float)
    if estimator is None:
        out[:, 3] = 1.0
        return out
    probabilities = estimator.predict_proba(features)
    for i, label in enumerate(estimator.classes_):
        out[:, SOURCE_CLASSES.index(str(label))] = probabilities[:, i]
    return out


class NativeCategoricalRegressor(RegressorMixin, BaseEstimator):
    """Preserve native categories without one-hot polygon expansion."""
    def __init__(self, name="catboost", seed=17, params=None, n_jobs=-1, device="cpu"):
        self.name, self.seed, self.params, self.n_jobs, self.device = name, seed, params, n_jobs, device

    def _prepare(self, features):
        out = features[self.columns_].copy()
        for column in self.categorical_:
            values = out[column].fillna("__missing__").astype(str)
            values = values.where(values.isin(self.categories_[column]), "__unknown__")
            out[column] = (pd.Categorical(values, categories=self.categories_[column] + ["__unknown__"])
                           if self.name == "lightgbm" else values)
        for column in out:
            if column not in self.categorical_:
                out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        return out

    def fit(self, features, target):
        self.columns_ = [c for c in features if c != "date"]
        self.categorical_ = [c for c in self.columns_ if not pd.api.types.is_numeric_dtype(features[c])]
        self.categories_ = {c: sorted(set(features[c].fillna("__missing__").astype(str)) - {"__unknown__"})
                            for c in self.categorical_}
        params = self.params or {}
        if self.name == "catboost":
            from catboost import CatBoostRegressor
            defaults = dict(iterations=700, depth=7, learning_rate=0.05, loss_function="RMSE",
                            verbose=False, allow_writing_files=False, task_type=self.device.upper())
            if self.device == "gpu":
                defaults["devices"] = "0"
            self.model_ = CatBoostRegressor(random_seed=self.seed, thread_count=self.n_jobs, **(defaults | params))
            self.model_.fit(self._prepare(features), target, cat_features=self.categorical_)
        else:
            from lightgbm import LGBMRegressor
            defaults = dict(n_estimators=500, num_leaves=31, learning_rate=0.05, verbosity=-1,
                            device_type=self.device)
            self.model_ = LGBMRegressor(random_state=self.seed, n_jobs=self.n_jobs, **(defaults | params))
            self.model_.fit(self._prepare(features), target, categorical_feature=self.categorical_)
        return self

    def predict(self, features):
        return self.model_.predict(self._prepare(features))


def predict_bundle_features(bundle, features):
    """Return prediction array and diagnostic arrays using saved OOF choices."""
    from .baselines import BaselineModel
    data = features.copy()
    models = bundle.estimators or {}
    source = models.get("source_classifier")
    probabilities = source_probabilities(source, data[models.get("source_columns", [])])
    for i, label in enumerate(SOURCE_CLASSES):
        data[f"p_{label}"] = probabilities[:, i]
    data['source_confidence'] = probabilities.max(axis=1)
    expected = bundle.config.get('feature_columns')
    if expected is not None and list(data.columns) != expected:
        raise ValueError('Feature column/order mismatch with trained bundle')
    members = {}
    baseline = BaselineModel(method=bundle.config.get("baseline_method", "mean_neighbors"), state=bundle.state)
    members["baseline"] = baseline.predict_from_features(features)["primary_ndvi_pred"].to_numpy(float)
    for name, estimator in models.get("regressors", {}).items():
        members[name] = np.asarray(estimator.predict(data), dtype=float)
    invalid = np.zeros(len(data), dtype=bool)
    for name in members:
        if name == 'baseline': continue
        bad = ~np.isfinite(members[name])
        invalid |= bad
        members[name] = np.where(bad, members['baseline'], members[name])
    weights = bundle.config.get("weights", {"baseline": 1.0})
    if any(name not in members for name in weights):
        raise ValueError("Bundle blend references an absent model")
    if any(weight < 0 for weight in weights.values()) or not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("Bundle blend weights must be nonnegative and sum to one")
    prediction = sum(weights[name] * members[name] for name in weights)
    disagreement = np.std(np.column_stack(list(members.values())), axis=1)
    diagnostics = {f"p_{label}": probabilities[:, i] for i, label in enumerate(SOURCE_CLASSES)}
    diagnostics.update(model_disagreement=disagreement, **{f"pred_{name}": value for name, value in members.items()})
    diagnostics['model_nonfinite_replaced'] = invalid
    from .calibration import apply_gate
    prediction, gated = apply_gate(prediction, members["baseline"], features, diagnostics,
                                    bundle.config.get("gate"))
    diagnostics["conservative_gate"] = gated
    clip = bundle.config.get("clip")
    if clip is not None:
        prediction = np.clip(prediction, *clip)
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite estimator output")
    return prediction, diagnostics
