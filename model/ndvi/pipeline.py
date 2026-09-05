from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .data import KEYS, VALUES, hide, read_data, mask_partition, metrics, source_of, sha256
from .features import build_features, baseline


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def write_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    tmp.replace(path)


def dump(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("wb") as f: pickle.dump(obj,f,protocol=5)
    tmp.replace(path)


def load(path):
    with Path(path).open("rb") as f: return pickle.load(f)


def folds_for(df, seed=405):
    polygons=np.array(sorted(df.anon_polygon_id.unique()))
    polygons=np.random.default_rng(seed).permutation(polygons)
    return {p:i%4 for i,p in enumerate(polygons)}


def validation_ids(observed, seed):
    rng=np.random.default_rng(seed); ids=[]
    for _,g in observed.groupby(["anon_polygon_id","year"],sort=True):
        ids.extend(rng.choice(g.index, max(1,round(.15*len(g))), replace=False))
    return np.asarray(sorted(ids),dtype=int)


def training_features(pool, context, seed, rounds=1):
    frames=[]; targets=[]
    for r in range(rounds):
        parts=mask_partition(pool,seed+1009*r)
        for part in range(7):
            ids=parts[parts==part].index
            if len(ids)==0: continue
            q=pool.loc[ids]
            x=build_features(q,hide(context,ids))
            frames.append(x)
            targets.append(q[["row_id",*KEYS,"primary_ndvi","year","crop_type"]].assign(source=source_of(q)))
        log(f"Маски: круг {r+1}/{rounds}, целей {sum(len(x) for x in frames)}")
    return pd.concat(frames,ignore_index=True),pd.concat(targets,ignore_index=True)


def prepare_fold(df, fold, out, rounds):
    cache=out/f"cache/fold_{fold}.pkl"
    if cache.exists(): return load(cache)
    tr=df[df.dataset=="train"].copy()
    assignment=folds_for(tr)
    held=tr.anon_polygon_id.map(assignment)==fold
    observed=tr.primary_ndvi.notna()
    vid=validation_ids(tr[held & observed],9100+fold)
    context=hide(tr,vid)
    pool=tr[~held & observed]
    xt,yt=training_features(pool,context,12000+fold,rounds)
    xv=build_features(tr.loc[vid],context)
    yv=tr.loc[vid,["row_id",*KEYS,"primary_ndvi","crop_type","year"]].assign(source=source_of(tr.loc[vid]))
    pack=(xt,yt,xv,yv)
    dump(cache,pack)
    manifest={"fold":fold,"role":"audit" if fold==3 else "development",
              "mask_seed":9100+fold,"training_polygons":sorted(pool.anon_polygon_id.unique()),
              "validation_polygons":sorted(tr.loc[vid].anon_polygon_id.unique()),
              "hidden_row_ids":vid.tolist(),"n_train_examples":len(xt),"n_validation":len(xv),
              "context_policy":"All query dynamic values hidden globally; held-out polygon visible history allowed; no test context in CV."}
    write_json(out/f"manifests/fold_{fold}.json",manifest)
    return pack


def fit_weights(y, pred):
    """Выпуклая смесь, выбранная только на development OOF."""
    names=list(pred); p=np.column_stack([pred[n] for n in names]); y=np.asarray(y)
    rmse=np.sqrt(((p-y[:,None])**2).mean(axis=0))
    best=int(rmse.argmin()); start=np.zeros(len(names));start[best]=1
    def objective(w):
        return np.mean((p@w-y)**2) + 2e-6*np.sum(w*w)
    opt=minimize(objective,start,method="SLSQP",bounds=[(0,1)]*len(names),
                 constraints={"type":"eq","fun":lambda w:w.sum()-1},
                 options={"maxiter":1000,"ftol":1e-12})
    w=opt.x if opt.success else start
    # Удаляем ничтожные компоненты и уменьшаем размер финального bundle.
    w[w<.01]=0
    if w.sum()==0:w[best]=1
    w/=w.sum()
    return dict(zip(names,map(float,w)))


def blend(pred,weights):
    return sum(w*np.asarray(pred[n]) for n,w in weights.items() if w>0)


def validate_submission(df,pred,path):
    gap=df[df.is_synthetic_gap]
    p=np.asarray(pred,float)
    if len(p)!=len(gap) or not np.isfinite(p).all():
        raise ValueError("Неверная длина/значения submission")
    sub=gap[KEYS].copy();sub["date"]=sub.date.dt.strftime("%Y-%m-%d")
    sub["primary_ndvi_pred"]=p
    if sub.duplicated(KEYS).any():raise ValueError("Дубликаты submission")
    tmp=Path(path).with_suffix(".tmp.csv")
    sub.to_csv(tmp,index=False,float_format="%.10f");tmp.replace(path)
    reread=pd.read_csv(path)
    assert list(reread.columns)==[*KEYS,"primary_ndvi_pred"]
    assert len(reread)==int(df.is_synthetic_gap.sum())
    return sub


def run_cv(args,df,out,deadline):
    from .models import fit_predict
    model_names=args.models.split(",")
    for fold in args.folds:
        if time.monotonic()>deadline-60:break
        log(f"Fold {fold}: подготовка")
        xt,yt,xv,yv=prepare_fold(df,fold,out,args.rounds)
        path=out/f"predictions/fold_{fold}.pkl"
        existing=load(path) if path.exists() else {"meta":yv,"pred":{}}
        existing["pred"].update(linear=baseline(xv),midpoint=baseline(xv,"midpoint"))
        for name in model_names:
            if name in existing["pred"]:continue
            if time.monotonic()>deadline-60:break
            started=time.monotonic();log(f"Fold {fold}: {name}")
            preds,model=fit_predict(name,xt,yt.primary_ndvi.to_numpy(),yt.source.to_numpy(),
                                    {"validation":xv},seed=2026,device=args.device,
                                    iterations=args.iterations,threads=args.threads)
            existing["pred"][name]=preds["validation"]
            if "validation_source_probability" in preds:
                existing["source_probability"]=preds["validation_source_probability"]
            dump(path,existing)
            score=metrics(yv.primary_ndvi,preds["validation"])
            log(f"Fold {fold} {name}: {score}, {time.monotonic()-started:.1f}s")
            # Только важности и прогнозы CV; веса моделей будут получены при final refit.
            if name=="cat":
                pd.DataFrame({"feature":xt.columns,"importance":model["model"].feature_importances_}).sort_values(
                    "importance",ascending=False).to_csv(out/f"reports/importance_fold_{fold}.csv",index=False)
        dump(path,existing)
    summarize(out)


def summarize(out):
    packs={f:load(out/f"predictions/fold_{f}.pkl") for f in range(4) if (out/f"predictions/fold_{f}.pkl").exists()}
    rows=[]
    for f,p in packs.items():
        for name,pr in p["pred"].items():
            rows.append({"fold":f,"role":"audit" if f==3 else "development","model":name,**metrics(p["meta"].primary_ndvi,pr)})
    pd.DataFrame(rows).to_csv(out/"reports/experiments.csv",index=False)
    dev=[packs[f] for f in [0,1,2] if f in packs]
    if not dev:return
    common=set.intersection(*[set(p["pred"]) for p in dev])
    if 3 in packs:
        # Незавершенный дорогой эксперт не попадает в смесь без прогнозов audit.
        # Проверяется наличие файла, значения audit y не участвуют в выборе.
        common &= set(packs[3]["pred"])
    names=sorted(common)
    y=np.concatenate([p["meta"].primary_ndvi.to_numpy() for p in dev])
    pred={n:np.concatenate([p["pred"][n] for p in dev]) for n in names}
    weights=fit_weights(y,pred)
    result={"development_folds":[f for f in [0,1,2] if f in packs],"weights":weights,
            "development":{n:metrics(y,pr) for n,pr in pred.items()},
            "development_ensemble":metrics(y,blend(pred,weights)),
            "audit_note":"Fold 3 is never used for weights or model hyperparameters."}
    if 3 in packs and all(n in packs[3]["pred"] for n,w in weights.items() if w>0):
        a=packs[3];pa=blend(a["pred"],weights)
        result["audit"]={n:metrics(a["meta"].primary_ndvi,pr) for n,pr in a["pred"].items()}
        result["audit_ensemble"]=metrics(a["meta"].primary_ndvi,pa)
        result["audit_by_source"]={str(s):metrics(a["meta"].loc[a["meta"].source==s,"primary_ndvi"],pa[a["meta"].source.to_numpy()==s])
                                   for s in range(3) if (a["meta"].source==s).any()}
    # OOF сохраняются построчно, чтобы последующие модели сравнивались по тем же ключам.
    export=[]
    for f,p in packs.items():
        z=p["meta"].copy().assign(fold=f,role="audit" if f==3 else "development")
        for n,pr in p["pred"].items():z[n]=pr
        if all(n in p["pred"] for n,w in weights.items() if w>0):z["ensemble"]=blend(p["pred"],weights)
        export.append(z)
    pd.concat(export).to_csv(out/"reports/oof_predictions.csv",index=False)
    write_json(out/"reports/metrics.json",result)
    write_json(out/"artifacts/ensemble_weights.json",weights)
    log("Development blend: "+str(result["development_ensemble"])+"; weights="+str(weights))
    if "audit_ensemble" in result:log("Audit: "+str(result["audit_ensemble"]))


def final_fit(args,df,out,deadline):
    from .models import fit_predict
    weights_path=out/"artifacts/ensemble_weights.json"
    if not weights_path.exists():raise ValueError("Сначала запустите CV, чтобы выбрать смесь")
    report=json.loads((out/"reports/metrics.json").read_text())
    if len(report["development_folds"])!=3 or "audit_ensemble" not in report:
        raise ValueError("Final refit требует завершенных трех development folds и audit fold 3")
    weights=json.loads(weights_path.read_text())
    gap=df[df.is_synthetic_gap]
    xg=build_features(gap,df)
    validate_submission(df,baseline(xg),out/"submission_baseline.csv")
    cache=out/"cache/final_training.pkl"
    if cache.exists():xt,yt=load(cache)
    else:
        # Глобальный supervised fit использует только официальный train.
        # Видимые значения test используются как контекст реконструкции.
        pool=df[(df.dataset=="train") & df.primary_ndvi.notna()]
        xt,yt=training_features(pool,df,27001,args.rounds)
        dump(cache,(xt,yt))
    dump(out/"cache/final_queries.pkl",xg)
    pp=out/"artifacts/final_predictions.pkl"
    pred=load(pp) if pp.exists() else {"linear":baseline(xg),"midpoint":baseline(xg,"midpoint")}
    bundle={"feature_columns":xt.columns.tolist(),"weights":weights,"models":{},"version":"1.0.0"}
    bundle_path=out/"artifacts/model_bundle.pkl"
    if bundle_path.exists():bundle=load(bundle_path)
    bundle["weights"]=weights
    for name,w in sorted(weights.items(),key=lambda kv:-kv[1]):
        if w<=0 or name in pred:continue
        if time.monotonic()>deadline-60:
            log("Бюджет исчерпан, завершенные модели сохранены. Запустите --stage final для продолжения.")
            break
        log(f"Final fit: {name}")
        if name=="tabicl":
            from .foundation import fit_predict_tabicl
            pr,model=fit_predict_tabicl(xt,yt,xg,out/"artifacts/tabicl_final",args)
            pred[name]=pr
            bundle["models"][name]=model
        else:
            pr,model=fit_predict(name,xt,yt.primary_ndvi.to_numpy(),yt.source.to_numpy(),{"test":xg},
                                seed=2026,device=args.device,iterations=args.iterations,threads=args.threads)
            pred[name]=pr["test"];bundle["models"][name]=model
        dump(pp,pred);dump(bundle_path,bundle)
        # Промежуточный файл помечен отдельно: это еще не полная выбранная смесь.
        available={n:w for n,w in weights.items() if w>0 and n in pred}
        total=sum(available.values())
        partial=blend(pred,{n:w/total for n,w in available.items()})
        validate_submission(df,partial,out/"submission_checkpoint.csv")
    if all(n in pred for n,w in weights.items() if w>0):
        dump(bundle_path,bundle)
        p=blend(pred,weights)
        validate_submission(df,p,out/"submission.csv")
        write_json(out/"artifacts/submission_manifest.json",{
            "status":"complete","n_rows":len(gap),"weights":weights,"test_sha256":sha256(args.test),
            "train_sha256":sha256(args.train),"submission_sha256":sha256(out/"submission.csv"),
            "features":xt.columns.tolist(),"label_policy":"official train only; observed new test is context only"})
        log(f"Готово: {out/'submission.csv'}, строк {len(gap)}")
    else:
        write_json(out/"artifacts/submission_manifest.json",{"status":"incomplete","missing":[n for n,w in weights.items() if w>0 and n not in pred]})


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",default="data/train.csv");p.add_argument("--test",default="data/test_features.csv")
    p.add_argument("--out",default="runs/main")
    p.add_argument("--stage",choices=["cv","final","all","summary","foundation"],default="all")
    p.add_argument("--device",choices=["CPU","GPU"],default="CPU")
    p.add_argument("--models",default="lgb,lgb_nodonor,cat,cat_residual,source_experts")
    p.add_argument("--iterations",type=int,default=1400)
    p.add_argument("--rounds",type=int,default=1)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--budget-minutes",type=float,default=360)
    p.add_argument("--folds",type=int,nargs="+",default=[0,1,2,3])
    p.add_argument("--allow-new-data",action="store_true")
    p.add_argument("--tabicl-context",type=int,default=12000)
    p.add_argument("--tabicl-estimators",type=int,default=4)
    p.add_argument("--tabicl-model-path",default=None)
    args=p.parse_args()
    deadline=time.monotonic()+args.budget_minutes*60
    out=Path(args.out).resolve()
    for d in ["artifacts","reports","cache","predictions","manifests"]:(out/d).mkdir(parents=True,exist_ok=True)
    df=read_data(args.train,args.test,strict=not args.allow_new_data)
    config={"train_sha256":sha256(args.train),"test_sha256":sha256(args.test),
            "iterations":args.iterations,"rounds":args.rounds,"device":args.device,
            "feature_version":"1.0.0","source_hash":hashlib.sha256(Path(__file__).with_name("features.py").read_bytes()).hexdigest()}
    cp=out/"artifacts/run_config.json"
    if cp.exists() and json.loads(cp.read_text())!=config:
        raise ValueError("Параметры/код/данные изменились: создайте новый --out, чтобы не смешивать cache/OOF.")
    write_json(cp,config)
    if args.stage in {"cv","all"}:run_cv(args,df,out,deadline)
    if args.stage=="summary":summarize(out)
    if args.stage=="foundation":
        from .foundation import run_foundation_cv
        run_foundation_cv(args,df,out,deadline)
        summarize(out)
    if args.stage in {"final","all"}:final_fit(args,df,out,deadline)


if __name__=="__main__":main()
