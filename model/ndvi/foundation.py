from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

CHECKPOINT = "tabicl-regressor-v2-20260212.ckpt"


def select_columns(x,y):
    """Отбор выполняется заново только на train соответствующего outer fold."""
    from lightgbm import LGBMRegressor
    selector=LGBMRegressor(n_estimators=180,num_leaves=15,min_child_samples=50,
                           n_jobs=2,verbosity=-1,random_state=417)
    selector.fit(x,y)
    priority=pd.Series(selector.feature_importances_,index=x.columns).sort_values(ascending=False)
    mandatory=["year","doy","crop","primary_linear","primary_mean21","one_sided",
               "source_spread","s2_linear","landsat_linear","modis_linear"]
    return list(dict.fromkeys(mandatory+priority.index.tolist()))[:96]


def sample_context(meta, maximum, seed=981):
    if len(meta)<=maximum:return np.arange(len(meta))
    # Пропорциональная стратификация сохраняет распределение источников, лет и культур.
    from sklearn.model_selection import train_test_split
    strata=meta.source.astype(str)+"_"+(meta.year//3).astype(str)+"_"+meta.crop_type.astype(str)
    counts=strata.map(strata.value_counts())
    strata=strata.where(counts>=2,"rare")
    if (strata.value_counts()<2).any():strata=meta.source.astype(str)
    selected,_=train_test_split(np.arange(len(meta)),train_size=maximum,
                                stratify=strata,random_state=seed)
    return np.sort(selected)


def fit_predict_tabicl(x,meta,q,save_dir,args):
    import torch
    from tabicl import TabICLRegressor
    from .pipeline import write_json
    from .data import sha256
    device="cuda" if args.device=="GPU" else "cpu"
    if device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна: включите GPU в Kaggle settings")
    y=meta.primary_ndvi.to_numpy(float)
    columns=select_columns(x,y)
    keep=sample_context(meta,args.tabicl_context)
    reg=TabICLRegressor(n_estimators=args.tabicl_estimators,batch_size=1,
                        checkpoint_version=CHECKPOINT,model_path=args.tabicl_model_path,
                        allow_auto_download=args.tabicl_model_path is None,
                        device=device,use_amp="auto",use_fa3=False,kv_cache=False,
                        offload_mode="auto",random_state=981,n_jobs=args.threads)
    # Нормализация, обработка NaN и feature permutations принадлежат самому TabICL.
    reg.fit(x.iloc[keep][columns].to_numpy(),y[keep])
    pred=np.asarray(reg.predict(q[columns].to_numpy()),float).reshape(-1)
    if not np.isfinite(pred).all():raise RuntimeError("TabICL вернул нефинитные значения")
    save_dir=Path(save_dir);save_dir.mkdir(parents=True,exist_ok=True)
    reg.save(save_dir/"fitted.pkl",save_model_weights=True,save_training_data=True,save_kv_cache=False)
    info={"package":"tabicl==2.1.1","repo_id":"jingang/TabICL","checkpoint":CHECKPOINT,
          "checkpoint_sha256":sha256(reg.model_path_),"license":"BSD-3-Clause",
          "n_context":len(keep),"n_features":len(columns),"estimators":args.tabicl_estimators,
          "training_indices":keep.tolist(),"columns":columns,"device":device,
          "reference":"https://arxiv.org/abs/2602.11139"}
    write_json(save_dir/"manifest.json",info)
    # Путь относителен к run directory; bundle переносим вместе с artifacts.
    model={"fitted_path":str(save_dir/"fitted.pkl"),"columns":columns,
           "manifest":info}
    del reg
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return pred,model


def run_foundation_cv(args,df,out,deadline):
    from .pipeline import prepare_fold,load,dump,log,write_json
    from .data import metrics
    config={"model":"tabicl","checkpoint":CHECKPOINT,"context":args.tabicl_context,
            "estimators":args.tabicl_estimators,"columns":96,"selection":"train-only LightGBM"}
    path=out/"artifacts/foundation_config.json"
    if path.exists() and json.loads(path.read_text())!=config:
        raise ValueError("Изменились параметры TabICL: нужен новый --out для честного сравнения")
    write_json(path,config)
    for fold in args.folds:
        if time.monotonic()>deadline-120:break
        pp=out/f"predictions/fold_{fold}.pkl"
        if not pp.exists():raise ValueError("Сначала выполните tree CV")
        pack=load(pp)
        if "tabicl" in pack["pred"]:continue
        xt,yt,xv,yv=prepare_fold(df,fold,out,args.rounds)
        log(f"TabICLv2 fold {fold}: начало")
        pred,_=fit_predict_tabicl(xt,yt,xv,out/f"artifacts/tabicl_fold_{fold}",args)
        pack["pred"]["tabicl"]=pred;dump(pp,pack)
        log(f"TabICLv2 fold {fold}: {metrics(yv.primary_ndvi,pred)}")
