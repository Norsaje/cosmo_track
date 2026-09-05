"""Explicitly authorized offline training entry point intended for Kaggle GPU.

Nothing fits on import. Real estimator training requires --allow-training.
Validation keys, excluded polygons and forecasting context are shared with CV.
"""
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

from veg_recovery.data import fingerprint, read_dataset
from veg_recovery.features import (FEATURE_VERSION, build_features, fit_feature_state,
                                   infer_source_labels, source_feature_columns)
from veg_recovery.validation import (MaskSpec, apply_mask, load_folds, split_fold,
                                     context_diagnostics, metric_table, rmse, gap_score)
from .baselines import BaselineModel
from .bundle import save_trained_bundle
from .manifest import git_commit, package_versions, sha256_file
from .estimators import make_regressor, make_source_classifier, source_probabilities, SOURCE_CLASSES
from .calibration import select_blend, select_gate, fit_uncertainty, evaluate_uncertainty, composite_rmse, source_reliability_table


def _json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=str)+'\n', encoding='utf-8')


def prepare_pseudo_training(frame, *, seed=17, max_samples=6000, blocks=5):
    """Key/polygon cross-masking before priors; no current target in any feature.

    Four of five blocks also omit their polygons from target priors, approximating
    the large unseen fraction. Source features never read the pseudo row's sensor.
    This function computes deterministic statistics only; it trains no estimators.
    """
    visible=frame.loc[np.isfinite(frame.primary_ndvi)].sort_values(['anon_polygon_id','date'])
    if visible.empty: raise ValueError('No finite fit labels')
    rng=np.random.default_rng(seed)
    polygons=np.array(sorted(visible.anon_polygon_id.unique()))
    partitions=np.array_split(rng.permutation(polygons),min(blocks,len(polygons)))
    features=[]; targets=[]
    for block,group in enumerate(partitions):
        pool=visible.loc[visible.anon_polygon_id.isin(group)]
        n=min(len(pool),max(1,max_samples//len(partitions)))
        selected=pool.iloc[np.sort(rng.choice(len(pool),n,replace=False))]
        keys=selected[['anon_polygon_id','date']]
        state=fit_feature_state(frame,excluded_keys=keys,excluded_polygons=group if block else ())
        features.append(build_features(apply_mask(frame,keys),keys,state))
        truth=selected[['anon_polygon_id','date','primary_ndvi']].copy()
        truth['source_label']=infer_source_labels(selected).to_numpy()
        truth['pseudo_block']=block; targets.append(truth)
    return pd.concat(features,ignore_index=True),pd.concat(targets,ignore_index=True)


def crossfit_source(features, targets, *, seed=17, n_jobs=2):
    """Group OOF source probabilities for regressor fitting, final classifier for inference."""
    from sklearn.model_selection import GroupKFold
    cols=source_feature_columns(features); groups=features.anon_polygon_id
    if groups.nunique()<2: raise ValueError('Source cross-fit needs at least two fit polygons')
    out=np.zeros((len(features),4))
    for k,(fit,validation) in enumerate(GroupKFold(min(5,groups.nunique())).split(features,groups=groups)):
        source=make_source_classifier(features[cols],seed+k,n_jobs)
        source.fit(features.iloc[fit][cols],targets.iloc[fit].source_label)
        out[validation]=source_probabilities(source,features.iloc[validation][cols])
    final=make_source_classifier(features[cols],seed,n_jobs)
    final.fit(features[cols],targets.source_label)
    return out,final,cols


def _with_source(features, probabilities):
    result=features.copy()
    for i,source in enumerate(SOURCE_CLASSES): result['p_'+source]=probabilities[:,i]
    result['source_confidence']=probabilities.max(axis=1)
    return result


def _peak_mb():
    import sys
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024 if sys.platform=='darwin' else 1024)


def prepare_folds(train,test,folds,output,max_samples,n_jobs):
    cache=[]; spec=MaskSpec.from_test(test)
    for mode,repeat,fold in folds[['mode','repeat','fold']].drop_duplicates().itertuples(index=False,name=None):
        start=time.perf_counter(); part=split_fold(train,folds,mode,int(repeat),int(fold),spec)
        x,y=prepare_pseudo_training(part.fit_frame,seed=part.metadata['seed'],max_samples=max_samples)
        p,source,cols=crossfit_source(x,y,seed=part.metadata['seed'],n_jobs=n_jobs)
        x=_with_source(x,p)
        state=fit_feature_state(part.fit_frame)
        validation=build_features(part.context_frame,part.gap_keys,state)
        validation=_with_source(validation,source_probabilities(source,validation[cols]))
        detail=context_diagnostics(part.context_frame,part.gap_keys,state.seen_polygons)
        for col in ['p_'+s for s in SOURCE_CLASSES]+['context_quality','left_days_1','right_days_1']:
            detail[col]=validation[col].to_numpy()
        truth=part.targets[['anon_polygon_id','date','primary_ndvi','source_label']].merge(detail,on=['anon_polygon_id','date'],validate='one_to_one')
        for col,value in [('mode',mode),('repeat',repeat),('fold',fold),('seed',part.metadata['seed'])]: truth[col]=value
        truth['pred_baseline']=BaselineModel(state=state).predict_from_features(validation).primary_ndvi_pred.to_numpy()
        truth['source_prediction']=np.array(SOURCE_CLASSES)[validation[['p_'+s for s in SOURCE_CLASSES]].to_numpy().argmax(axis=1)]
        path=output/'cache'/f'{mode}_{repeat}_{fold}'; path.mkdir(parents=True,exist_ok=True)
        files={}
        for name,table in [('x_train',x),('y_train',y),('x_validation',validation),('y_validation',truth)]:
            dest=path/(name+'.csv.gz'); table.to_csv(dest,index=False); files[dest.name]=sha256_file(dest)
        _json(path/'manifest.json',dict(files=files,feature_version=FEATURE_VERSION,metadata=part.metadata))
        cache.append(dict(path=path,mode=mode,repeat=int(repeat),fold=int(fold),seed=part.metadata['seed']))
        print(f'prepared {mode}/{repeat}/{fold}: {len(x)} pseudo train, {len(validation)} validation, {time.perf_counter()-start:.1f}s',flush=True)
    return cache


def read_cache(entry):
    root=entry['path']; manifest=json.loads((root/'manifest.json').read_text())
    if manifest['feature_version']!=FEATURE_VERSION: raise ValueError('Incompatible cached features')
    for name,digest in manifest['files'].items():
        if sha256_file(root/name)!=digest: raise ValueError('Corrupt feature cache')
    result=[]
    for name in ['x_train','y_train','x_validation','y_validation']:
        table=pd.read_csv(root/(name+'.csv.gz'),parse_dates=['date'])
        for col in ['anon_polygon_id','crop_type','seasonal_prior_level']:
            if col in table: table[col]=table[col].fillna('__unknown__').astype(str)
        result.append(table)
    return result


def tune_parameters(name,cache,output,trials,n_jobs,device):
    if trials==0: return {}
    if not 30<=trials<=60: raise ValueError('Initial Optuna budget must be 30–60 trials, or 0 for fixed-parameter reproduction')
    import optuna
    def objective(trial):
        if name=='hgb':
            params={'learning_rate':trial.suggest_float('learning_rate',.025,.12,log=True),
                    'max_leaf_nodes':trial.suggest_int('max_leaf_nodes',15,63),
                    'l2_regularization':trial.suggest_float('l2_regularization',.1,20,log=True),
                    'min_samples_leaf':trial.suggest_int('min_samples_leaf',15,60),
                    'max_iter':trial.suggest_int('max_iter',150,450,step=100)}
        elif name=='extra_trees':
            params={'n_estimators':400,'min_samples_leaf':trial.suggest_int('min_samples_leaf',2,12),
                    'max_features':trial.suggest_float('max_features',.4,1.),
                    'max_depth':trial.suggest_categorical('max_depth',[None,12,20,32])}
        elif name=='catboost':
            params={'iterations':trial.suggest_int('iterations',400,1000,step=200),
                    'depth':trial.suggest_int('depth',5,9),
                    'learning_rate':trial.suggest_float('learning_rate',.025,.12,log=True),
                    'l2_leaf_reg':trial.suggest_float('l2_leaf_reg',.5,20,log=True),
                    'loss_function':'RMSE'}
        else:
            params={'num_leaves':trial.suggest_int('num_leaves',15,63),'learning_rate':trial.suggest_float('learning_rate',.025,.1,log=True),
                    'objective':trial.suggest_categorical('objective',['regression','huber'])}
        completed=[]
        for step,entry in enumerate(cache):
            x,y,v,t=read_cache(entry); estimator=make_regressor(name,x,entry['seed'],params,n_jobs,device)
            estimator.fit(x,y.primary_ndvi); t['prediction']=estimator.predict(v); completed.append(t)
            trial.report(float(np.mean([rmse(c.primary_ndvi,c.prediction) for c in completed])),step)
            if trial.should_prune(): raise optuna.TrialPruned()
        result=pd.concat(completed,ignore_index=True)
        trial.set_user_attr('full_parameters',params)
        return composite_rmse(result,result.prediction)
    study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=17),
                              pruner=optuna.pruners.MedianPruner(n_startup_trials=5,n_warmup_steps=5),
                              study_name=name,storage=f'sqlite:///{output.resolve()}/{name}_optuna.sqlite3',load_if_exists=True)
    remaining=max(0,trials-len(study.trials))
    if remaining: study.optimize(objective,n_trials=remaining)
    study.trials_dataframe().to_csv(output/f'{name}_trials.csv',index=False)
    return study.best_trial.user_attrs['full_parameters']


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-training',action='store_true')
    parser.add_argument('--train',default='data/train_dataset.csv'); parser.add_argument('--test',default='data/test_data.csv')
    parser.add_argument('--folds',default='configs/ml/folds_v1.csv'); parser.add_argument('--output',default='artifacts/ml/trained_gpu_v1')
    parser.add_argument('--models',nargs='+',choices=['hgb','extra_trees','catboost','lightgbm'],default=['catboost'])
    parser.add_argument('--device',choices=['cpu','gpu'],default='gpu')
    parser.add_argument('--trials',type=int,default=30); parser.add_argument('--max-train-gaps',type=int,default=6000)
    parser.add_argument('--jobs',type=int,default=2); parser.add_argument('--resume-cache',action='store_true')
    args=parser.parse_args(argv)
    if not args.allow_training: parser.error('Estimator training disabled. Run on Kaggle with --allow-training after authorization.')
    if args.trials!=0 and not 30<=args.trials<=60: parser.error('--trials must be 0 or 30–60')
    if args.device=='gpu' and any(name!='catboost' for name in args.models):
        parser.error('GPU mode currently supports --models catboost only')
    if args.device=='gpu':
        try:
            from catboost.utils import get_gpu_device_count
            gpu_count=get_gpu_device_count()
        except ImportError as error:
            parser.error(f'CatBoost GPU dependency is unavailable: {error}')
        if gpu_count<1:
            parser.error('No CUDA GPU detected; this profile requires a GPU runtime.')
        print(f'GPU training enabled: CatBoost sees {gpu_count} CUDA device(s)',flush=True)
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    if (output/'bundle/manifest.json').exists(): parser.error('Completed bundle exists: use a new output path')
    train=read_dataset(args.train,'train'); test=read_dataset(args.test,'test'); folds=load_folds(args.folds)
    fingerprints={'train':fingerprint(args.train),'test':fingerprint(args.test),'folds':fingerprint(args.folds)}
    cache_spec={'fingerprints':fingerprints,'max_train_gaps':args.max_train_gaps,'feature_version':FEATURE_VERSION}
    if args.resume_cache:
        saved=json.loads((output/'cache_manifest.json').read_text())
        if saved['spec']!=cache_spec: raise ValueError('Resume cache input/config mismatch')
        cache=[{**entry,'path':output/entry['path']} for entry in saved['entries']]
    else:
        cache=prepare_folds(train,test,folds,output,args.max_train_gaps,args.jobs)
        _json(output/'cache_manifest.json',{'spec':cache_spec,'entries':[{**e,'path':str(e['path'].relative_to(output))} for e in cache]})
    params={name:tune_parameters(name,cache,output,args.trials,args.jobs,args.device) for name in args.models}
    _json(output/'hyperparameters.json',params)
    all_oof=[]; tables=[]; experiments=[]; importance=[]
    stamp=datetime.now(timezone.utc).isoformat(); commit=git_commit()
    for entry in cache:
        x,y,v,truth=read_cache(entry)
        for name in args.models:
            start=time.perf_counter(); estimator=make_regressor(name,x,entry['seed'],params[name],args.jobs,args.device)
            estimator.fit(x,y.primary_ndvi); predicted=np.asarray(estimator.predict(v),float)
            truth['pred_'+name]=predicted
            table=metric_table(truth,truth[['anon_polygon_id','date']].assign(primary_ndvi_pred=predicted))
            for col in ['mode','repeat','fold']: table[col]=entry[col]
            table['model']=name; tables.append(table)
            # Real held-out permutation importance; no training targets reused.
            from sklearn.inspection import permutation_importance
            sample=np.linspace(0,len(v)-1,min(250,len(v)),dtype=int)
            pi=permutation_importance(estimator,v.iloc[sample],truth.primary_ndvi.iloc[sample],
                                      scoring='neg_root_mean_squared_error',n_repeats=2,random_state=entry['seed'],n_jobs=1)
            for column,value,std in zip(v.columns,pi.importances_mean,pi.importances_std):
                importance.append(dict(model=name,mode=entry['mode'],repeat=entry['repeat'],fold=entry['fold'],feature=column,importance=float(value),std=float(std)))
            def sub(mask): return rmse(truth.loc[mask,'primary_ndvi'],predicted[mask]) if mask.any() else None
            value=rmse(truth.primary_ndvi,predicted)
            experiments.append(dict(experiment_id=f'{name}_{entry["mode"]}_{entry["repeat"]}_{entry["fold"]}',timestamp=stamp,git_commit=commit,
                data_fingerprint=fingerprints['train'],fold_version='folds_v1',mask_version='real_test_v1',feature_version=FEATURE_VERSION,
                feature_set_hash=hashlib.sha256(json.dumps(list(x.columns)).encode()).hexdigest(),model=name,params_json=json.dumps(params[name]),
                seed=entry['seed'],split=entry['mode'],rmse=value,gap_score=gap_score(value),known_rmse=sub(truth.seen_polygon),
                unseen_rmse=sub(~truth.seen_polygon),hard_rmse=value if entry['mode']=='D' else None,runtime_sec=time.perf_counter()-start,
                peak_memory_mb=_peak_mb(),artifact_uri=str(output/'oof_predictions.csv.gz'),conclusion='OOF evaluated; see decisions.json'))
        truth['model_disagreement']=truth[['pred_baseline']+['pred_'+m for m in args.models]].std(axis=1,ddof=0)
        all_oof.append(truth)
    oof=pd.concat(all_oof,ignore_index=True)
    weights,comparisons,blend,choice=select_blend(oof,['baseline']+args.models)
    gate,gate_experiments,prediction=select_gate(oof,blend)
    oof['pred_global_blend']=blend; oof['pred_ensemble']=prediction
    for (mode,repeat,fold),group in oof.groupby(['mode','repeat','fold']):
        for name in ['baseline','global_blend','ensemble']:
            table=metric_table(group,group[['anon_polygon_id','date']].assign(primary_ndvi_pred=group['pred_'+name]))
            table['mode']=mode; table['repeat']=repeat; table['fold']=fold; table['model']=name
            tables.append(table)
    oof.to_csv(output/'oof_predictions.csv.gz',index=False)
    pd.concat(tables,ignore_index=True).to_csv(output/'subgroup_metrics.csv',index=False)
    pd.DataFrame(experiments).to_csv(output/'experiments.csv',index=False)
    pd.DataFrame(importance).to_csv(output/'feature_importance.csv',index=False)
    evaluate_uncertainty(oof,prediction).to_csv(output/'interval_coverage.csv',index=False)
    uncertainty=fit_uncertainty(oof,prediction); uncertainty['status']='empirical_oof_not_certified'
    decisions={'blend':comparisons,'choice':choice,'gates':gate_experiments,
               'clip':{'decision':'reject','reason':'No OOF clipping experiment conducted; raw target preserved'},
               'selection_bias_warning':'OOF used for tuning and selection; not independent final generalization estimate'}
    _json(output/'decisions.json',decisions)
    source=oof[['anon_polygon_id','date','mode','repeat','fold','source_label','source_prediction']+['p_'+s for s in SOURCE_CLASSES]]
    source.to_csv(output/'source_oof.csv.gz',index=False)
    source_reliability_table(source).to_csv(output/'source_calibration.csv',index=False)
    summary=[]
    for (mode,label),group in source.groupby(['mode','source_label']):
        summary.append(dict(mode=mode,source=label,n=len(group),accuracy=float(group.source_label.eq(group.source_prediction).mean()),
                            brier=float(np.mean(np.sum((group[['p_'+s for s in SOURCE_CLASSES]].to_numpy()-np.eye(4)[[SOURCE_CLASSES.index(s) for s in group.source_label]])**2,axis=1)))))
    pd.DataFrame(summary).to_csv(output/'source_accuracy.csv',index=False)
    # Final fit uses all train labels, with the same pseudo-gap procedure. Test
    # visible rows remain inference context; test predictions never select params.
    x,y=prepare_pseudo_training(train,seed=17,max_samples=args.max_train_gaps)
    p,source,source_cols=crossfit_source(x,y,seed=17,n_jobs=args.jobs); x=_with_source(x,p)
    regressors={}
    for name in args.models:
        regressors[name]=make_regressor(name,x,17,params[name],args.jobs,args.device)
        regressors[name].fit(x,y.primary_ndvi)
    from veg_recovery.anomalies.baseline import fit_harmonization
    config=dict(baseline_method='mean_neighbors',method='mean_neighbors',weights=weights,gate=gate,clip=None,
                uncertainty=uncertainty,interval_level=.95,feature_columns=list(x.columns),hyperparameters=params,
                harmonization=fit_harmonization(train),composite_weights={'A':.5,'B':.25,'C':.15,'D':.1},
                execution_device=args.device)
    cv_summary=[]
    for mode,group in oof.groupby('mode'):
        for name in ['baseline']+args.models+['ensemble']:
            value=rmse(group.primary_ndvi,group['pred_'+name]); cv_summary.append(dict(mode=mode,model=name,n=len(group),rmse=value,gap_score=gap_score(value)))
    pd.DataFrame(cv_summary).to_csv(output/'cv_summary.csv',index=False)
    save_trained_bundle(output/'bundle',fit_feature_state(train),config,
        {'regressors':regressors,'source_classifier':source,'source_columns':source_cols},
        dict(model_version=f'p0-{"-".join(args.models)}-{args.device}-v1',train_fingerprints=fingerprints,
             seeds=[17,29,43,71,101],cv_summary={'metrics':cv_summary,'decisions':decisions}))
    from veg_recovery.cli.batch import main as batch
    code=batch(['--input',args.test,'--bundle',str(output/'bundle'),'--trusted-bundle','--output',str(output/'submission.csv'),
                '--diagnostics',str(output/'diagnostics.csv'),'--expected-count','3112'])
    if code: raise RuntimeError('Post-training batch validation failed')
    _json(output/'run_complete.json',dict(timestamp=datetime.now(timezone.utc),packages=package_versions(),
          fingerprints=fingerprints,models=args.models,device=args.device))


if __name__=='__main__': main()
