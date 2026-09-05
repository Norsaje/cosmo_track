"""Reproduce data audit, all baseline CVs and a checked inference bundle. No ML fit."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from veg_recovery.data import read_dataset, summarize_frame, fingerprint
from veg_recovery.features import FEATURE_VERSION, build_features, fit_feature_state, infer_source_labels
from veg_recovery.validation import (generate_folds, save_folds, load_folds, split_fold, MaskSpec, context_diagnostics,
                                     metric_table, gap_score, rmse, composite_score, test_mask_profile)
from veg_recovery.models.baselines import BaselineModel, BASELINE_METHODS
from veg_recovery.models.bundle import save_baseline_bundle
from veg_recovery.models.manifest import git_commit, package_versions
from veg_recovery.models.calibration import fit_uncertainty, evaluate_uncertainty, source_reliability_table
from veg_recovery.anomalies.baseline import fit_harmonization
from veg_recovery.inference import load_reconstructor
from veg_recovery.contracts import ReconstructionRequest, ReconstructionPayload
from veg_recovery.cli.batch import write_submission


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str)+'\n', encoding='utf-8')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',default='data/train_dataset.csv'); parser.add_argument('--test',default='data/test_data.csv')
    parser.add_argument('--folds',default='configs/ml/folds_v1.csv'); parser.add_argument('--output',default='artifacts/ml/baseline_v1')
    parser.add_argument('--max-gaps',type=int,default=1200)
    args=parser.parse_args(argv)
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    train=read_dataset(args.train,'train',strict_current=True); test=read_dataset(args.test,'test',strict_current=True)
    fingerprints={'train':fingerprint(args.train),'test':fingerprint(args.test)}
    dump(output/'data_summary.json',{'train':summarize_frame(train),'test':summarize_frame(test),'fingerprints':fingerprints,
                                    'train_target_range':[float(train.primary_ndvi.min()),float(train.primary_ndvi.max())],
                                    'test_visible_target_range':[float(test.primary_ndvi.min()),float(test.primary_ndvi.max())]})
    dump(output/'test_mask_profile.json',test_mask_profile(test,train))
    source_report={}
    for name,frame in [('train',train),('test_visible',test)]:
        visible=frame.loc[np.isfinite(frame.primary_ndvi)]
        labels=infer_source_labels(visible)
        source_report[name]={'n':len(visible),'label_counts':labels.value_counts().to_dict(),
                             'unknown_keys':visible.loc[labels.eq('unknown'),['anon_polygon_id','date']].to_dict('records')}
    dump(output/'source_hierarchy.json',source_report)
    folds=load_folds(args.folds) if Path(args.folds).exists() else generate_folds(train,test,max_gaps_per_split=args.max_gaps)
    if not Path(args.folds).exists(): save_folds(folds,args.folds)
    spec=MaskSpec.from_test(test); all_oof=[]; all_metrics=[]; experiments=[]; mask_matches=[]
    commit=git_commit(); stamp=datetime.now(timezone.utc).isoformat()
    for mode,repeat,fold in folds[['mode','repeat','fold']].drop_duplicates().itertuples(index=False,name=None):
        start=time.perf_counter()
        part=split_fold(train,folds,mode,int(repeat),int(fold),spec)
        state=fit_feature_state(part.fit_frame)
        features=build_features(part.context_frame,part.gap_keys,state)
        detail=context_diagnostics(part.context_frame,part.gap_keys,state.seen_polygons)
        detail=detail.merge(features[['anon_polygon_id','date','p_s2','p_landsat','p_modis','p_unknown','context_quality']],on=['anon_polygon_id','date'],validate='one_to_one')
        base=part.targets[['anon_polygon_id','date','primary_ndvi','source_label']].merge(detail,on=['anon_polygon_id','date'],validate='one_to_one')
        base['left_days_1']=features.days_left_1.to_numpy(); base['right_days_1']=features.days_right_1.to_numpy()
        base['mode']=mode; base['repeat']=repeat; base['fold']=fold; base['seed']=part.metadata['seed']
        feature_hash=hashlib.sha256(json.dumps([(c,str(features[c].dtype)) for c in features]).encode()).hexdigest()
        shared_runtime=time.perf_counter()-start
        for method in BASELINE_METHODS:
            method_start=time.perf_counter()
            prediction=BaselineModel(method,state).predict_from_features(features)
            base['pred_'+method]=prediction.primary_ndvi_pred.to_numpy()
            base['error_'+method]=base['pred_'+method]-base.primary_ndvi
            table=metric_table(part.targets,prediction,detail)
            table['mode']=mode; table['repeat']=repeat; table['fold']=fold; table['model']=method
            all_metrics.append(table)
            value=rmse(base.primary_ndvi,base['pred_'+method]); runtime=shared_runtime+time.perf_counter()-method_start
            def subgroup(mask): return rmse(base.loc[mask,'primary_ndvi'],base.loc[mask,'pred_'+method]) if mask.any() else None
            experiments.append(dict(experiment_id=f'baseline_v1_{method}_{mode}_{repeat}_{fold}',timestamp=stamp,git_commit=commit,
                                    data_fingerprint=fingerprints['train'],fold_version='folds_v1',mask_version=spec.version,
                                    feature_version=FEATURE_VERSION,feature_set_hash=feature_hash,model=method,params_json='{}',seed=part.metadata['seed'],
                                    split=f'CV-{mode}/{repeat}/{fold}',rmse=value,gap_score=gap_score(value),
                                    known_rmse=subgroup(base.seen_polygon),unseen_rmse=subgroup(~base.seen_polygon),
                                    hard_rmse=value if mode=='D' else None,runtime_sec=runtime,
                                    peak_memory_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024 if __import__('sys').platform=='darwin' else 1024),
                                    artifact_uri=str(output/'oof_predictions.csv.gz'),conclusion='measured; selection uses four-mode composite'))
        all_oof.append(base)
        mask_matches.append(dict(mode=mode,repeat=int(repeat),fold=int(fold),n=len(base),known_fraction=float(base.seen_polygon.mean()),
                                  gap_length_counts={str(k):int(v) for k,v in base.gap_length.value_counts().items()},
                                  source_counts=base.source_label.value_counts().to_dict(),
                                  month_counts={str(k):int(v) for k,v in base.month.value_counts().items()},
                                  left_distance_counts=base.left_distance_bin.value_counts().to_dict(),
                                  right_distance_counts=base.right_distance_bin.value_counts().to_dict()))
        print(f'CV-{mode}/{repeat}/{fold}: n={len(base)}, mean={rmse(base.primary_ndvi,base.pred_mean_neighbors):.6f}, linear={rmse(base.primary_ndvi,base.pred_linear):.6f}, {time.perf_counter()-start:.1f}s',flush=True)
    oof=pd.concat(all_oof,ignore_index=True); oof.to_csv(output/'oof_predictions.csv.gz',index=False)
    pd.concat(all_metrics,ignore_index=True).to_csv(output/'subgroup_metrics.csv',index=False)
    dump(output/'mask_match_by_fold.json',mask_matches)
    summaries=[]; decisions=[]
    for method in BASELINE_METHODS:
        values={}
        for mode, group in oof.groupby('mode'):
            value=rmse(group.primary_ndvi,group['pred_'+method]); values[mode]=value
            summaries.append(dict(model=method,mode=mode,n=len(group),rmse=value,gap_score=gap_score(value)))
        decisions.append(dict(model=method,composite_rmse=composite_score(values)))
    selected=min(decisions,key=lambda d:d['composite_rmse'])['model']
    for record in decisions: record['decision']='keep' if record['model']==selected else 'reject'; record['reason']='four-mode weighted RMSE, weights 0.50/0.25/0.15/0.10'
    pd.DataFrame(summaries).to_csv(output/'cv_summary.csv',index=False); dump(output/'baseline_decisions.json',decisions)
    prediction=oof['pred_'+selected].to_numpy()
    uncertainty=fit_uncertainty(oof,prediction); uncertainty['status']='empirical_oof_not_certified'
    coverage=evaluate_uncertainty(oof,prediction); coverage.to_csv(output/'interval_coverage.csv',index=False)
    dump(output/'uncertainty.json',uncertainty)
    labels=np.array(['s2','landsat','modis','unknown']); probs=oof[['p_'+s for s in labels]].to_numpy()
    source_oof=oof[['anon_polygon_id','date','mode','repeat','fold','source_label']].copy()
    source_oof['source_prediction']=labels[probs.argmax(axis=1)]
    for s in labels: source_oof['p_'+s]=oof['p_'+s]
    source_oof.to_csv(output/'source_oof.csv.gz',index=False)
    source_reliability_table(source_oof).to_csv(output/'source_calibration.csv',index=False)
    source_summary=[]
    for (mode,source),group in source_oof.groupby(['mode','source_label']):
        source_summary.append(dict(mode=mode,source=source,n=len(group),accuracy=float(group.source_label.eq(group.source_prediction).mean()),method='empirical_context_uncalibrated'))
    pd.DataFrame(source_summary).to_csv(output/'source_accuracy.csv',index=False)
    config=dict(method=selected,clip=None,interval_level=.95,uncertainty=uncertainty,harmonization=fit_harmonization(train),
                selection={'rule':'minimum composite baseline RMSE','weights':{'A':.5,'B':.25,'C':.15,'D':.1}},
                training_status='no ML estimators trained; baseline statistics only')
    bundle=output/'bundle'
    if not (bundle/'manifest.json').exists():
        save_baseline_bundle(bundle,fit_feature_state(train),config,dict(model_version='baseline-v1-'+selected,train_fingerprints=fingerprints,
            seeds=[17,29,43,71,101],cv_summary={'metrics':summaries,'selection':decisions}))
    model=load_reconstructor(bundle)
    result=model.predict(ReconstructionRequest(test,test.is_synthetic_gap,'competition'))
    write_submission(result.predictions,test.loc[test.is_synthetic_gap,['anon_polygon_id','date']],output/'submission.csv')
    result.diagnostics.to_csv(output/'diagnostics.csv',index=False)
    # Real backend-boundary smoke: Pydantic payload validates every exported row.
    payload=ReconstructionPayload.from_result(result)
    dump(output/'api_smoke.json',dict(model_version=payload.model_version,n_predictions=len(payload.predictions),n_diagnostics=len(payload.diagnostics),schema_version=payload.schema_version))
    assert fingerprints=={'train':fingerprint(args.train),'test':fingerprint(args.test)},'Source CSV was modified'
    report=Path('reports'); report.mkdir(exist_ok=True)
    experiment_path=report/'experiments.csv'; new=pd.DataFrame(experiments)
    if experiment_path.exists():
        # A rerun is a separate measurement; preserve negative/earlier results.
        old=pd.read_csv(experiment_path)
        collision=new.experiment_id.isin(old.experiment_id)
        new.loc[collision,'experiment_id']=new.loc[collision,'experiment_id']+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
        new=pd.concat([old,new],ignore_index=True)
    new.to_csv(experiment_path,index=False)
    dump(output/'environment.json',package_versions())
    print(json.dumps(dict(selected=selected,bundle=str(bundle),submission=str(output/'submission.csv'),decisions=decisions)),flush=True)


if __name__=='__main__': main()
