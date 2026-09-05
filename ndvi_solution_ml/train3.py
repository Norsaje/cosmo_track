import warnings, os, json, argparse, time, pickle
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold

ap = argparse.ArgumentParser()
ap.add_argument('--dir', default='work/cv2')
ap.add_argument('--models', default='lgb_c,lgb_d,cat,mlp')
ap.add_argument('--final', type=int, default=0)
a = ap.parse_args()
D = a.dir
Xtr = pd.read_pickle(f'{D}/Xtr.pkl'); ytr = np.load(f'{D}/ytr.npy'); mtr = pd.read_pickle(f'{D}/mtr.pkl')
QF = 'Xq.pkl' if a.final else 'Xev.pkl'
Xev = pd.read_pickle(f'{D}/{QF}')
yev = None if a.final else np.load(f'{D}/yev.npy')
mev = pd.read_pickle(f'{D}/mq.pkl' if a.final else f'{D}/mev.pkl')
for c in Xtr.columns:
    if Xtr[c].dtype == np.float64: Xtr[c] = Xtr[c].astype(np.float32)
    if Xev[c].dtype == np.float64: Xev[c] = Xev[c].astype(np.float32)
gtr = mtr['anon_polygon_id'].values
def rmse(p): return float('nan') if yev is None else float(np.sqrt(np.mean((yev - p) ** 2)))

SRCF = f'{D}/srcprob.pkl'
if os.path.exists(SRCF):
    oof_p, ev_p = pickle.load(open(SRCF, 'rb'))
else:
    SRC = [c for c in Xtr.columns if any(k in c for k in
           ['ph5','ph10','ph16','ph_modis_ok','rate_','av_','ds_','dn_','doy','year','tt','aoi_code',
            'sin','cos','yfrac','crop_code','n_obs','p_cal_','cm_harm_n','cm_s2_n','cm_landsat_n','cm_modis_n',
            'h_last_src','h_next_src','h_pd1','h_nd1','gap_'])]
    pc = {'objective':'multiclass','num_class':3,'learning_rate':0.05,'num_leaves':63,
          'min_data_in_leaf':40,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':1,
          'lambda_l2':3.0,'verbose':-1,'num_threads':2,'deterministic':True,'force_col_wise':True}
    str_ = mtr['source'].values.astype(int); mok = str_ >= 0
    oof_p = np.full((len(Xtr), 3), np.nan)
    for tr_i, va_i in GroupKFold(n_splits=4).split(Xtr[mok], str_[mok], groups=gtr[mok]):
        ii = np.flatnonzero(mok)
        m = lgb.train(pc, lgb.Dataset(Xtr.iloc[ii[tr_i]][SRC], str_[ii[tr_i]],
                      categorical_feature=['aoi_code','crop_code']), num_boost_round=350)
        oof_p[ii[va_i]] = m.predict(Xtr.iloc[ii[va_i]][SRC])
    sf = lgb.train(pc, lgb.Dataset(Xtr[mok][SRC], str_[mok],
                   categorical_feature=['aoi_code','crop_code']), num_boost_round=350)
    ev_p = sf.predict(Xev[SRC])
    pickle.dump((oof_p, ev_p), open(SRCF, 'wb'))
    if not a.final:
        print('source acc eval %.4f' % (ev_p.argmax(1) == mev['source'].values).mean(), flush=True)
for k, nm in enumerate(['p_s2','p_ls','p_md']):
    Xtr[nm] = oof_p[:, k]; Xev[nm] = ev_p[:, k]
Xtr['p_max'] = np.nanmax(oof_p, 1); Xev['p_max'] = np.nanmax(ev_p, 1)
F = list(Xtr.columns); CAT = ['aoi_code','crop_code']
np.save(f'{D}/evp.npy', ev_p)

B = {'objective':'l2','learning_rate':0.03,'num_leaves':95,'min_data_in_leaf':30,
     'feature_fraction':0.5,'bagging_fraction':0.8,'bagging_freq':1,'lambda_l2':6.0,
     'cat_smooth':20,'verbose':-1,'num_threads':2,'deterministic':True,'force_col_wise':True,'seed':17}
CFG = {
 'lgb_c': (dict(B, num_leaves=255, min_data_in_leaf=60, feature_fraction=0.35, lambda_l2=20.0,
                learning_rate=0.025, seed=101), 2600),
 'lgb_d': (dict(B, extra_trees=True, num_leaves=180, min_data_in_leaf=25, feature_fraction=0.45,
                lambda_l2=3.0, learning_rate=0.035, seed=202), 2400),
 'lgb_e': (dict(B, objective='huber', alpha=0.12, num_leaves=95, feature_fraction=0.5, seed=303), 2200),
 'lgb_a2': (dict(B, seed=909), 2200),
 'lgb_a': (dict(B), 2200),
 'lgb_b': (dict(B, num_leaves=47, learning_rate=0.025, feature_fraction=0.35,
                lambda_l2=12.0, min_data_in_leaf=60, seed=71), 2860),
}
EXP = dict(B, num_leaves=47, min_data_in_leaf=40, feature_fraction=0.45)
want = a.models.split(',')
for nm in want:
    t0 = time.time()
    if nm in CFG:
        params, rounds = CFG[nm]
        m = lgb.train(params, lgb.Dataset(Xtr[F], ytr, categorical_feature=CAT), num_boost_round=rounds)
        p = m.predict(Xev[F])
    elif nm == 'experts':
        # три спутниковых эксперта: каждый учится на ВСЕХ строках, где его сенсор наблюдался
        SCH = ['s2_ndvi', 'landsat_ndvi', 'modis_ndvi']; BCOL = ['base_s2_adj', 'base_ls_adj', 'base_md_adj']
        import gc
        expm = np.zeros((len(Xev), 3))
        Xall = Xtr[F].to_numpy(np.float32); Xevn = Xev[F].to_numpy(np.float32)
        cat_idx = [F.index(c) for c in CAT]
        md = float(np.nanmedian(ytr))
        for k in range(3):
            yv = mtr[SCH[k]].values.astype(np.float64); msk = np.isfinite(yv)
            b_tr = Xtr[BCOL[k]].to_numpy(np.float64); b_ev = Xev[BCOL[k]].to_numpy(np.float64)
            b_tr = np.where(np.isfinite(b_tr), b_tr, md); b_ev = np.where(np.isfinite(b_ev), b_ev, md)
            ds = lgb.Dataset(Xall[msk], (yv - b_tr)[msk], feature_name=F,
                             categorical_feature=CAT, free_raw_data=True)
            mk = lgb.train(EXP, ds, num_boost_round=1500)
            expm[:, k] = b_ev + mk.predict(Xevn)
            del ds, mk; gc.collect()
            print('  expert %d done (n=%d)' % (k, int(msk.sum())), flush=True)
        del Xall, Xevn; gc.collect()
        np.save(f'{D}/expmat.npy', expm)
        p = (expm * ev_p).sum(1)
    elif nm == 'cat':
        from catboost import CatBoostRegressor, Pool
        Xc = Xtr[F].copy(); Xq = Xev[F].copy()
        for c in CAT:
            Xc[c] = Xc[c].fillna(-1).astype(int).astype(str); Xq[c] = Xq[c].fillna(-1).astype(int).astype(str)
        cb = CatBoostRegressor(iterations=1400, depth=7, learning_rate=0.045, l2_leaf_reg=6,
                              loss_function='RMSE', random_seed=17, verbose=200, thread_count=2,
                              border_count=96, rsm=0.5, boosting_type='Plain')
        cb.fit(Pool(Xc, ytr, cat_features=CAT)); p = cb.predict(Xq)
        del Xc, Xq
    elif nm == 'mlp':
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import QuantileTransformer
        Z = Xtr[F].to_numpy(np.float32); Ze = Xev[F].to_numpy(np.float32)
        med = np.nanmedian(Z, 0); Z = np.where(np.isfinite(Z), Z, med); Ze = np.where(np.isfinite(Ze), Ze, med)
        qt = QuantileTransformer(n_quantiles=500, output_distribution='normal', subsample=60000, random_state=0)
        Z = qt.fit_transform(Z).astype(np.float32); Ze = qt.transform(Ze).astype(np.float32)
        base = Xtr['h_rsm'].to_numpy(np.float64); base = np.where(np.isfinite(base), base, np.nanmedian(ytr))
        base_e = Xev['h_rsm'].to_numpy(np.float64); base_e = np.where(np.isfinite(base_e), base_e, np.nanmedian(ytr))
        acc = np.zeros(len(Ze))
        for sd in [0, 1]:
            mlp = MLPRegressor(hidden_layer_sizes=(384, 192, 96), activation='relu', alpha=3e-4,
                              batch_size=1024, learning_rate_init=1.2e-3, max_iter=45,
                              early_stopping=True, n_iter_no_change=6, validation_fraction=0.06,
                              random_state=sd, verbose=False)
            mlp.fit(Z, ytr - base)
            acc += base_e + mlp.predict(Ze)
        p = acc / 2
        del Z, Ze
    else:
        continue
    np.save(f'{D}/pred_{nm}.npy', p)
    print('%s %.5f (%.0fs)' % (nm, rmse(p), time.time() - t0), flush=True)
