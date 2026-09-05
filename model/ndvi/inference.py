"""Пакетное применение модели. Модуль не содержит сервера или сетевых запросов."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import read_data, KEYS, hide
from .features import build_features,baseline
from .models import predict_fitted
from .pipeline import load,validate_submission


def predict_queries(df,queries,run_dir):
    run_dir=Path(run_dir)
    bundle=load(run_dir/"artifacts/model_bundle.pkl")
    x=build_features(queries,hide(df,queries.index))
    if x.columns.tolist()!=bundle["feature_columns"]:
        raise ValueError("Несовместимая схема признаков")
    pred=np.zeros(len(x))
    for name,w in bundle["weights"].items():
        if w<=0:continue
        if name in {"linear","midpoint"}:p=baseline(x,name)
        elif name=="tabicl":
            from tabicl import TabICLRegressor
            model=bundle["models"][name]
            # Не полагаемся на абсолютный путь исходной Kaggle-сессии.
            fitted=run_dir/"artifacts/tabicl_final/fitted.pkl"
            reg=TabICLRegressor.load(fitted,device="cpu")
            p=reg.predict(x[model["columns"]].to_numpy())
        else:p=predict_fitted(name,bundle["models"][name],x)
        pred+=w*np.asarray(p).reshape(-1)
    if not np.isfinite(pred).all():raise ValueError("Нефинитный прогноз")
    return pred,x


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--train",default="data/train.csv");a.add_argument("--test",default="data/test_features.csv")
    a.add_argument("--run",default="runs/local");a.add_argument("--output",default="submission_reproduced.csv")
    a.add_argument("--allow-new-data",action="store_true");a.add_argument("--all-missing",action="store_true")
    args=a.parse_args()
    d=read_data(args.train,args.test,strict=not args.allow_new_data)
    query=d[(d.dataset=="test") & d.primary_ndvi.isna()] if args.all_missing else d[d.is_synthetic_gap]
    p,x=predict_queries(d,query,args.run)
    if args.all_missing:
        z=query[KEYS].copy();z["primary_ndvi_pred"]=p
        z.to_csv(args.output,index=False)
    else:validate_submission(d,p,args.output)
    print(f"Сохранено {len(p)} строк: {args.output}")


if __name__=="__main__":main()
