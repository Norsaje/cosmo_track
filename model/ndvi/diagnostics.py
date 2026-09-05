"""Отчёт об ошибках, эмпирические интервалы и кандидаты аномальных периодов."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import KEYS,SENSORS,read_data,source_of,metrics
from .features import seasonal
from .pipeline import write_json,load,blend


def residual_analysis(run):
    oof=pd.read_csv(run/"reports/oof_predictions.csv",parse_dates=["date"])
    dev=oof[oof.role=="development"].copy();audit=oof[oof.role=="audit"].copy()
    dev["abs_error"]=(dev.primary_ndvi-dev.ensemble).abs()
    q90=float(dev.abs_error.quantile(.9,interpolation="higher"))
    result={"interval_kind":"empirical; not a guaranteed conformal interval for dependent time series",
            "development_absolute_error_q90":q90,
            "audit_coverage":float(((audit.primary_ndvi-audit.ensemble).abs()<=q90).mean())}
    rng=np.random.default_rng(1981)
    groups=[g for _,g in audit.groupby("anon_polygon_id")]
    bootstrap=[]
    for _ in range(1500):
        selected=rng.integers(len(groups),size=len(groups))
        d=pd.concat([groups[i] for i in selected])
        bootstrap.append(np.sqrt(np.mean((d.primary_ndvi-d.ensemble)**2)))
    result["audit_polygon_bootstrap_rmse_interval_95"]=[float(x) for x in np.quantile(bootstrap,[.025,.975])]
    rows=[]
    for role,d in [("development",dev),("audit",audit)]:
        for field in ["year","source","crop_type","anon_polygon_id"]:
            for key,g in d.groupby(field):
                rows.append({"role":role,"slice":field,"value":str(key),**metrics(g.primary_ndvi,g.ensemble)})
    pd.DataFrame(rows).to_csv(run/"reports/error_slices.csv",index=False)
    write_json(run/"reports/uncertainty.json",result)
    return result


def anomaly_candidates(df,submission):
    t=df[df.dataset=="test"].copy()
    # Только реальные наблюдения создают сезонную норму. Восстановления в нее не возвращаются.
    clim=seasonal(t,df)
    source=source_of(t)
    mu=np.full(len(t),np.nan);std=np.full(len(t),np.nan);n=np.zeros(len(t))
    for i,s in enumerate(SENSORS):
        m=source==i
        mu[m]=clim.loc[m,f"clim_{s}_mean"]
        std[m]=clim.loc[m,f"clim_{s}_std"]
        n[m]=clim.loc[m,f"clim_{s}_n"].fillna(0)
    value=t.primary_ndvi.to_numpy()
    valid=np.isfinite(value)&(value>=-1)&(value<=1)&(n>=3)&np.isfinite(mu)
    z=np.divide(value-mu,np.maximum(std,.03),out=np.full(len(t),np.nan),where=valid)
    result=t[KEYS+ ["primary_ndvi"]].copy()
    result["source"]=np.array([*SENSORS,"missing"])[np.where(source>=0,source,3)]
    result["reference_mean"]=mu;result["reference_std"]=std
    result["reference_years"]=n;result["z_score"]=z
    result["quality_issue"]=np.isfinite(value)&((value < -1)|(value > 1))
    result["status"]=np.select([~valid,z < -2,z < -1],["insufficient_or_invalid","strong_negative_candidate","negative_candidate"],default="usual")
    # Предсказания сохраняются как отдельный слой и не объявляются подтвержденным стрессом.
    sub=submission.copy();sub["date"]=pd.to_datetime(sub.date)
    result=result.merge(sub,on=KEYS,how="left")
    events=[]
    for p,g in result[result.primary_ndvi.notna()].groupby("anon_polygon_id"):
        g=g.sort_values("date")
        negative=g.z_score < -1
        breaks=(~negative | ~negative.shift(fill_value=False) | g.date.diff().dt.days.fillna(999).gt(16)).cumsum()
        for _,segment in g[negative].groupby(breaks[negative]):
            duration=(segment.date.max()-segment.date.min()).days
            if len(segment)<2 or duration<8:continue
            events.append({"anon_polygon_id":p,"start_date":segment.date.min(),"end_date":segment.date.max(),
                           "observed_points":len(segment),"duration_days":duration,"min_z":segment.z_score.min(),
                           "interpretation":"Устойчивое отрицательное отклонение. Причина не установлена; проверить облачность, уборку, смену культуры и погоду.",
                           "status":"candidate_for_review"})
    return result,pd.DataFrame(events,columns=["anon_polygon_id","start_date","end_date","observed_points","duration_days","min_z","interpretation","status"])


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",default="runs/local")
    p.add_argument("--train",default="data/train.csv");p.add_argument("--test",default="data/test_features.csv")
    args=p.parse_args();run=Path(args.run)
    uncertainty=residual_analysis(run)
    sub=pd.read_csv(run/"submission.csv")
    q=uncertainty["development_absolute_error_q90"]
    intervals=sub.copy();intervals["lower_empirical90"]=intervals.primary_ndvi_pred-q
    intervals["upper_empirical90"]=intervals.primary_ndvi_pred+q
    intervals.to_csv(run/"reports/gap_uncertainty.csv",index=False)
    d=read_data(args.train,args.test)
    points,events=anomaly_candidates(d,sub)
    points.to_csv(run/"reports/anomaly_points.csv",index=False)
    events.to_csv(run/"reports/anomaly_candidates.csv",index=False)
    print(json.dumps(uncertainty,indent=2));print("Anomaly candidates:",len(events))


if __name__=="__main__":main()
