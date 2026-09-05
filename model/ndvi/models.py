from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMRegressor

from .features import baseline
from .data import SENSORS


def fit_predict(name, x, y, source, queries, seed=2026, device="CPU", iterations=1400, threads=4):
    """Все бюджеты фиксированы до outer audit; его y сюда не передается."""
    params = dict(iterations=iterations, depth=7, learning_rate=.04, l2_leaf_reg=8,
                  loss_function="RMSE", thread_count=threads, random_seed=seed,
                  task_type=device, verbose=False, allow_writing_files=False,
                  border_count=128)
    if device == "GPU":
        params["devices"] = "0"
    result, models = {}, {}
    if name.startswith("cat"):
        residual = name == "cat_residual"
        if residual:
            params.update(depth=6, l2_leaf_reg=12)
        model = CatBoostRegressor(**params)
        target = y - baseline(x) if residual else y
        model.fit(x, target)
        for k, q in queries.items():
            result[k] = model.predict(q) + (baseline(q) if residual else 0)
        models["model"] = model
    elif name in {"lgb", "lgb_nodonor", "lgb_residual"}:
        columns = [c for c in x if name != "lgb_nodonor" or "donor" not in c]
        model = LGBMRegressor(n_estimators=iterations, learning_rate=.025, num_leaves=31,
                              max_depth=-1, min_child_samples=45, reg_lambda=8,
                              colsample_bytree=.85, subsample=.85, subsample_freq=1,
                              n_jobs=threads, random_state=seed, verbosity=-1,
                              deterministic=True, force_col_wise=True)
        residual = name == "lgb_residual"
        model.fit(x[columns], y-baseline(x) if residual else y)
        for k, q in queries.items():
            result[k] = model.predict(q[columns]) + (baseline(q) if residual else 0)
        models.update(model=model, columns=columns)
    elif name == "source_experts":
        # Идентичность источника доступна только у тренировочной цели.
        # Классификатор не получает текущие спутниковые значения запроса.
        clf_params = {k:v for k,v in params.items() if k != "loss_function"}
        clf_params.update(iterations=min(iterations,700), depth=6, loss_function="MultiClass")
        clf = CatBoostClassifier(**clf_params)
        clf.fit(x, source)
        models["classifier"] = clf
        sensor_models = {}
        for i, s in enumerate(SENSORS):
            mask = source == i
            if mask.sum() < 50:
                sensor_models[s] = None
                continue
            anchor = x[f"{s}_linear"].fillna(x.primary_linear).fillna(.5).to_numpy()
            sp = dict(params)
            sp.update(depth=6, iterations=min(iterations,1200), l2_leaf_reg=12)
            reg = CatBoostRegressor(**sp)
            reg.fit(x.loc[mask], y[mask]-anchor[mask])
            sensor_models[s] = reg
        models["sensor_models"] = sensor_models
        for k,q in queries.items():
            proba = clf.predict_proba(q)
            mat = np.zeros((len(q),3))
            for i,s in enumerate(SENSORS):
                anchor = q[f"{s}_linear"].fillna(q.primary_linear).fillna(.5).to_numpy()
                reg = sensor_models[s]
                mat[:,i] = anchor + (reg.predict(q) if reg is not None else 0)
            probabilities = np.zeros_like(mat)
            probabilities[:,clf.classes_.astype(int)] = proba
            result[k] = (probabilities*mat).sum(axis=1)
            result[k+"_source_probability"] = probabilities
    else:
        raise ValueError(name)
    return result, models


def predict_fitted(name, models, q):
    if name.startswith("cat"):
        return models["model"].predict(q) + (baseline(q) if name=="cat_residual" else 0)
    if name.startswith("lgb"):
        return models["model"].predict(q[models["columns"]]) + (baseline(q) if name=="lgb_residual" else 0)
    if name == "source_experts":
        clf = models["classifier"]
        proba = np.zeros((len(q),3))
        proba[:,clf.classes_.astype(int)] = clf.predict_proba(q)
        mat=[]
        for s in SENSORS:
            anchor=q[f"{s}_linear"].fillna(q.primary_linear).fillna(.5).to_numpy()
            reg=models["sensor_models"][s]
            mat.append(anchor + (reg.predict(q) if reg is not None else 0))
        return (proba*np.asarray(mat).T).sum(axis=1)
    raise ValueError(name)
