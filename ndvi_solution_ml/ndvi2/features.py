import numpy as np, pandas as pd
from .core import FEATCH, NDVI_CH, SENSORS
from .smooth import local_poly, robust_local_poly

TMAX = 5600
FULL = ['primary_ndvi', 's2_ndvi', 'landsat_ndvi', 'modis_ndvi']
RED = ['s2_evi', 'landsat_evi', 'modis_evi', 's2_ndwi', 'landsat_ndwi']
WEA = ['era5_temp_c', 'era5_precip_mm']
SENS_CH = {'s2': 's2_ndvi', 'landsat': 'landsat_ndvi', 'modis': 'modis_ndvi'}


def _cumstats(T, V):
    cs = np.concatenate([[0.0], np.cumsum(V)])
    cs2 = np.concatenate([[0.0], np.cumsum(V * V)])
    return cs, cs2


def _window(T, V, cs, cs2, q, W):
    lo = np.searchsorted(T, q - W, 'left'); hi = np.searchsorted(T, q + W, 'right')
    n = (hi - lo).astype(np.float64)
    s = cs[hi] - cs[lo]; s2 = cs2[hi] - cs2[lo]
    mean = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    var = np.where(n > 1, s2 / np.maximum(n, 1) - mean ** 2, np.nan)
    return mean, np.sqrt(np.maximum(var, 0)), n


def _neigh(T, V, q, k=3):
    n = len(T)
    idx = np.searchsorted(T, q, 'left')
    out = {}
    for j in range(1, k + 1):
        pi = idx - j
        ok = pi >= 0
        pv = np.where(ok, V[np.clip(pi, 0, max(n - 1, 0))] if n else np.nan, np.nan)
        pd_ = np.where(ok, q - T[np.clip(pi, 0, max(n - 1, 0))] if n else np.nan, np.nan)
        ni = idx + j - 1
        ok2 = ni < n
        nv = np.where(ok2, V[np.clip(ni, 0, max(n - 1, 0))] if n else np.nan, np.nan)
        nd = np.where(ok2, T[np.clip(ni, 0, max(n - 1, 0))] - q if n else np.nan, np.nan)
        out[f'p{j}'] = pv.astype(np.float64); out[f'pd{j}'] = pd_.astype(np.float64)
        out[f'n{j}'] = nv.astype(np.float64); out[f'nd{j}'] = nd.astype(np.float64)
    return out


def _calib(x, y, n0=25.0):
    """robust affine y ~ a + b x, shrunk to identity"""
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 6:
        return 0.0, 1.0, n
    x1, y1 = x[m], y[m]
    for _ in range(3):
        b = np.polyfit(x1, y1, 1)
        r = y1 - (b[0] * x1 + b[1])
        s = max(np.median(np.abs(r)) * 1.4826, 0.02)
        keep = np.abs(r) < 3 * s
        if keep.sum() < 6: break
        x1, y1 = x1[keep], y1[keep]
    slope, inter = float(b[0]), float(b[1])
    slope = float(np.clip(slope, 0.4, 2.2)); inter = float(np.clip(inter, -0.4, 0.4))
    w = n / (n + n0)
    return inter * w, 1.0 + (slope - 1.0) * w, n


class Prep:
    """Context-level precomputation shared by all query rows."""

    def __init__(self, ctx, wgroups=None):
        self.ctx = ctx
        A = ctx.n_aoi
        self.A = A
        self.wgroups = wgroups if wgroups is not None else {a: [a] for a in range(A)}
        # ---- cross-sensor calibration to S2 scale, per AOI ----
        self.cal = {}
        gl = {}
        for s in ['landsat', 'modis']:
            xs, ys = [], []
            for a in range(A):
                Ts, Vs, _ = ctx.obs['s2_ndvi'][a]
                To, Vo, _ = ctx.obs[SENS_CH[s]][a]
                if len(Ts) == 0 or len(To) == 0: continue
                i = np.searchsorted(Ts, To)
                ok = (i < len(Ts))
                ok &= np.where(ok, Ts[np.clip(i, 0, len(Ts) - 1)] == To, False)
                if ok.sum() == 0: continue
                xs.append(Vo[ok]); ys.append(Vs[np.clip(i[ok], 0, len(Ts) - 1)])
            if xs:
                gl[s] = _calib(np.concatenate(xs), np.concatenate(ys), n0=0.0)[:2]
            else:
                gl[s] = (0.0, 1.0)
        self.cal_global = gl
        for a in range(A):
            self.cal[a] = {'s2': (0.0, 1.0)}
            for s in ['landsat', 'modis']:
                Ts, Vs, _ = ctx.obs['s2_ndvi'][a]
                To, Vo, _ = ctx.obs[SENS_CH[s]][a]
                a0, b0 = gl[s]
                if len(Ts) and len(To):
                    i = np.searchsorted(Ts, To)
                    ok = (i < len(Ts))
                    ok &= np.where(ok, Ts[np.clip(i, 0, len(Ts) - 1)] == To, False)
                    if ok.sum() >= 8:
                        ai, bi, n = _calib(Vo[ok], Vs[np.clip(i[ok], 0, len(Ts) - 1)], n0=30.0)
                        w = ok.sum() / (ok.sum() + 30.0)
                        a0 = a0 * (1 - w) + ai * w; b0 = b0 * (1 - w) + bi * w
                self.cal[a][s] = (a0, b0)
        # ---- harmonized series per AOI ----
        self.harm = []
        for a in range(A):
            Ts, Vs, Ss = [], [], []
            for k, s in enumerate(SENSORS):
                T, V, _ = ctx.obs[SENS_CH[s]][a]
                if len(T) == 0: continue
                a0, b0 = self.cal[a][s]
                Ts.append(T); Vs.append(a0 + b0 * V); Ss.append(np.full(len(T), k, np.int8))
            if Ts:
                T = np.concatenate(Ts); V = np.concatenate(Vs); S = np.concatenate(Ss)
                o = np.argsort(T, kind='mergesort')
                self.harm.append((T[o], V[o], S[o]))
            else:
                self.harm.append((np.zeros(0, np.int64), np.zeros(0), np.zeros(0, np.int8)))
        # ---- residuals of visible observations -> per-date common mode ----
        self.res_sum = {}; self.res_cnt = {}
        self.res_med = {}; self.res_grp = {}; self.res_crop = {}
        self.harm_res = [None] * A
        base = ctx.base
        self._crop_of = base.groupby(ctx.aoi_codes)['crop_type'].first().reindex(range(A)).values
        self._gid = np.zeros(A, np.int32)
        seen = {}
        for a in range(A):
            k = tuple(sorted(self.wgroups.get(a, [a])))
            if k not in seen: seen[k] = len(seen)
            self._gid[a] = seen[k]
        self._ncrop = pd.factorize(pd.Series(self._crop_of))[0]
        for name, getter in [('harm', None)] + [(s, SENS_CH[s]) for s in SENSORS]:
            rs = np.zeros(TMAX); rc = np.zeros(TMAX)
            acc_t, acc_r, acc_a = [], [], []
            for a in range(A):
                if name == 'harm':
                    T, V, _ = self.harm[a]
                else:
                    T, V, _ = ctx.obs[getter][a]
                if len(T) < 8: continue
                fit = robust_local_poly(T, V, T.astype(np.float64), h=16.0, K=8, deg=2,
                                        exclude_self=True, iters=2)[:, 0]
                r = V - fit
                m = np.isfinite(r)
                np.add.at(rs, T[m], r[m]); np.add.at(rc, T[m], 1.0)
                acc_t.append(T[m]); acc_r.append(r[m]); acc_a.append(np.full(int(m.sum()), a))
                if name == 'harm':
                    self.harm_res[a] = (T[m], r[m])
            self.res_sum[name] = rs; self.res_cnt[name] = rc
            if acc_t:
                tt = np.concatenate(acc_t); rr = np.concatenate(acc_r); aa = np.concatenate(acc_a)
                df = pd.DataFrame({'t': tt, 'r': rr, 'g': self._gid[aa], 'c': self._ncrop[aa]})
                med = np.full(TMAX, np.nan)
                gm = df.groupby('t')['r'].median()
                med[gm.index.values] = gm.values
                self.res_med[name] = med
                self.res_grp[name] = df.groupby(['g', 't'])['r'].mean().to_dict()
                self.res_crop[name] = df.groupby(['c', 't'])['r'].mean().to_dict()
            else:
                self.res_med[name] = np.full(TMAX, np.nan)
                self.res_grp[name] = {}; self.res_crop[name] = {}
        # ---- систематическое смещение сенсора по фазе орбиты / бину композита ----
        # Физика: Landsat 7/8/9 и разные пути дают разное NDVI на одном поле;
        # у MODIS 16-дневный композит смещён относительно мгновенной кривой.
        # Оцениваем среднее отклонение наблюдений сенсора от гармонизованной кривой
        # внутри фазы, со сжатием к нулю (эмпирический Байес, k = sigma_res^2/sigma_bias^2).
        self.PHMOD = {'s2': 10, 'landsat': 16, 'modis': 13}
        self.phb = {}
        SHRINK = 10.0
        for s_ in SENSORS:
            nph = self.PHMOD[s_]
            bias = np.zeros((A, nph)); cnt = np.zeros((A, nph))
            for a in range(A):
                Th, Vh, _ = self.harm[a]
                T, V, _ = ctx.obs[SENS_CH[s_]][a]
                if len(Th) < 20 or len(T) < 10: continue
                a0, b0 = self.cal[a][s_]
                fit = robust_local_poly(Th, Vh, T.astype(np.float64), h=16.0, K=9, deg=2, iters=1)[:, 0]
                r = (a0 + b0 * V) - fit
                ph = (T % nph) if s_ != 'modis' else np.clip(((T % 365.25).astype(int) - 97) // 16, 0, nph - 1)
                m = np.isfinite(r)
                np.add.at(bias[a], ph[m], r[m]); np.add.at(cnt[a], ph[m], 1.0)
            self.phb[s_] = (bias / (cnt + SHRINK), cnt)
        # ---- per-date sensor availability counts ----
        self.av_cnt = {}
        for s in SENSORS:
            c = np.zeros(TMAX)
            for a in range(A):
                T, V, _ = ctx.obs[SENS_CH[s]][a]
                np.add.at(c, T, 1.0)
            self.av_cnt[s] = c
        self.aoi_alive = np.zeros(TMAX)
        for a in range(A):
            idx = ctx._order[ctx.aoi_start[a]:ctx.aoi_end[a]]
            np.add.at(self.aoi_alive, ctx.t[idx], 1.0)
        # ---- weather per-date per-aoi ----
        self.wt = np.full((A, TMAX), np.nan); self.wp = np.full((A, TMAX), np.nan)
        for a in range(A):
            for arr, ch in [(self.wt, 'era5_temp_c'), (self.wp, 'era5_precip_mm')]:
                T, V, _ = ctx.obs[ch][a]
                arr[a, T] = V
        self.wt_med = np.nanmedian(self.wt, 0); self.wp_med = np.nanmedian(self.wp, 0)
        # ---- phase observation rates per aoi/sensor/era/phase ----
        self.phase = {}
        eras = self._era(np.arange(TMAX))
        for s, mod in [('s2', 5), ('s2', 10), ('landsat', 16), ('modis', 16)]:
            ph = np.arange(TMAX) % mod
            key = (s, mod)
            num = np.zeros((A, 3, mod)); den = np.zeros((A, 3, mod))
            for a in range(A):
                idx = ctx._order[ctx.aoi_start[a]:ctx.aoi_end[a]]
                tt = ctx.t[idx]
                np.add.at(den, (a, eras[tt], ph[tt]), 1.0)
                T, V, _ = ctx.obs[SENS_CH[s]][a]
                np.add.at(num, (a, eras[T], ph[T]), 1.0)
            self.phase[key] = (num, den)
        # ---- climatology: per aoi x sensor x doy (excluding own year) ----
        self.clim = {}
        for s in SENSORS:
            tab = {}
            for a in range(A):
                T, V, _ = ctx.obs[SENS_CH[s]][a]
                tab[a] = (T, V)
            self.clim[s] = tab
        self.harm_clim = {a: (self.harm[a][0], self.harm[a][1]) for a in range(A)}
        self.crop = base.groupby(ctx.aoi_codes)['crop_type'].first().reindex(range(A)).values
        # crop x doy kernel grids (harmonised), plus per-AOI grids for exclusion
        DG = 366
        dd = np.arange(DG)
        K = np.exp(-0.5 * (np.minimum(np.abs(dd[:, None] - dd[None, :]),
                                      365 - np.abs(dd[:, None] - dd[None, :])) / 12.0) ** 2)
        K[K < 1e-3] = 0.0
        self.aoi_S = np.zeros((A, DG)); self.aoi_W = np.zeros((A, DG))
        for a in range(A):
            T, V, _ = self.harm[a]
            if len(T) == 0: continue
            d = (T % 365.25).astype(int) % DG
            cnt = np.bincount(d, minlength=DG).astype(float)
            val = np.bincount(d, weights=V, minlength=DG)
            self.aoi_S[a] = K @ val; self.aoi_W[a] = K @ cnt
        crops = pd.Series(self.crop)
        self.crop_S = {}; self.crop_W = {}
        for cname, ids in crops.groupby(crops).groups.items():
            ids = np.asarray(list(ids))
            self.crop_S[cname] = self.aoi_S[ids].sum(0); self.crop_W[cname] = self.aoi_W[ids].sum(0)

    @staticmethod
    def _era(t):
        # 0: <2013(-04), 1: 2013..2020, 2: 2021+
        yr = 2010 + (t // 365.25).astype(int)
        return np.clip(np.where(yr <= 2012, 0, np.where(yr <= 2020, 1, 2)), 0, 2)


def build_features(ctx, prep, qa, qt, qdoy, qyear):
    """qa: aoi codes, qt: day index. Returns DataFrame."""
    nq = len(qa)
    F = {}
    order = np.argsort(qa, kind='mergesort')
    starts = np.searchsorted(qa[order], np.arange(prep.A))
    ends = np.searchsorted(qa[order], np.arange(prep.A), side='right')

    def alloc(name):
        F[name] = np.full(nq, np.nan)

    names_full = []
    for c in FULL:
        for k in ['p1','pd1','p2','pd2','p3','pd3','n1','nd1','n2','nd2','n3','nd3',
                  'lin','mid','sm','slope','curv','eff','w21m','w21s','w21n','w45m','w45n','rsm']:
            names_full.append(f'{c}__{k}')
    names_red = []
    for c in RED + WEA:
        for k in ['p1','pd1','n1','nd1','lin','sm','w45m','w45n']:
            names_red.append(f'{c}__{k}')
    extra = ['res_prev','res_next','res_prev_dt','res_next_dt','res_mean2',
             'phb_s2','phb_ls','phb_md','phb_n_s2','phb_n_ls','phb_n_md',
             'base_s2_adj','base_ls_adj','base_md_adj','phb_exp',
             'aoi_code','cm_harm_med','cm_wgrp','cm_crop','cm_m1','cm_p1','cm_m2','cm_p2',
             'cm_win5','cm_win5_n','p_cal_s2','p_cal_ls','p_cal_md',
             'base_s2','base_ls','base_md','h_sm','h_slope','h_curv','h_eff','h_rsm','h_p1','h_pd1','h_n1','h_nd1','h_lin','h_mid',
             'h_w21m','h_w21s','h_w21n','h_w45m','h_w45n','h_last_src','h_next_src',
             'cm_harm','cm_harm_n','cm_s2','cm_s2_n','cm_landsat','cm_landsat_n','cm_modis','cm_modis_n',
             'lvl_harm','lvl_harm_n',
             'av_s2','av_landsat','av_modis','av_any','av_s2_f','av_landsat_f','av_modis_f',
             'ph5','ph10','ph16','ph_modis_ok','ph16_8',
             'rate_s2_5','rate_s2_10','rate_landsat_16','rate_modis_16',
             'ds_s2','dn_s2','ds_landsat','dn_landsat','ds_modis','dn_modis',
             'wt_grp','wp_grp','wt_med','wp_med','wp_c7','wp_c15','wp_c30','wt_m15','wt_m30',
             'clim_s2','clim_landsat','clim_modis','clim_harm','clim_harm_n','clim_crop',
             'cal_ls_a','cal_ls_b','cal_md_a','cal_md_b',
             'sm_s2','sm_landsat','sm_modis','sm_s2_eff','sm_landsat_eff','sm_modis_eff',
             'doy','year','tt','sin1','cos1','sin2','cos2','yfrac',
             'n_obs_year','n_obs_aoi','gap_left','gap_right','gap_span','is_edge']
    for n in names_full + names_red + extra:
        alloc(n)

    eras = Prep._era(qt)
    for a in range(prep.A):
        sl = order[starts[a]:ends[a]]
        if len(sl) == 0: continue
        q = qt[sl].astype(np.float64)
        qi = qt[sl].astype(np.int64)
        # ---- per-channel ----
        for c in FULL + RED + WEA:
            T, V, _ = ctx.obs[c][a]
            red = c in RED + WEA
            nb = _neigh(T, V, qi, k=1 if red else 3)
            if len(T):
                cs, cs2 = _cumstats(T, V)
                w45 = _window(T, V, cs, cs2, qi, 45)
                w21 = _window(T, V, cs, cs2, qi, 21)
                sm = local_poly(T, V, q, h=13.0, K=7, deg=2)
                if not red:
                    rsm = robust_local_poly(T, V, q, h=16.0, K=8, deg=2, iters=2)[:, 0]
            else:
                w45 = (np.full(len(sl), np.nan),) * 3; w21 = w45
                sm = np.full((len(sl), 5), np.nan); rsm = np.full(len(sl), np.nan)
            p1, n1 = nb['p1'], nb['n1']; pd1, nd1 = nb['pd1'], nb['nd1']
            tot = pd1 + nd1
            lin = np.where(np.isfinite(p1) & np.isfinite(n1) & (tot > 0),
                           p1 + (n1 - p1) * (pd1 / np.maximum(tot, 1e-9)), np.nan)
            lin = np.where(np.isfinite(lin), lin, np.where(np.isfinite(p1), p1, n1))
            mid = np.where(np.isfinite(p1) & np.isfinite(n1), 0.5 * (p1 + n1),
                           np.where(np.isfinite(p1), p1, n1))
            F[f'{c}__p1'][sl] = p1; F[f'{c}__pd1'][sl] = pd1
            F[f'{c}__n1'][sl] = n1; F[f'{c}__nd1'][sl] = nd1
            F[f'{c}__lin'][sl] = lin
            F[f'{c}__sm'][sl] = sm[:, 0]
            F[f'{c}__w45m'][sl] = w45[0]; F[f'{c}__w45n'][sl] = w45[2]
            if not red:
                for j in [2, 3]:
                    F[f'{c}__p{j}'][sl] = nb[f'p{j}']; F[f'{c}__pd{j}'][sl] = nb[f'pd{j}']
                    F[f'{c}__n{j}'][sl] = nb[f'n{j}']; F[f'{c}__nd{j}'][sl] = nb[f'nd{j}']
                F[f'{c}__mid'][sl] = mid
                F[f'{c}__slope'][sl] = sm[:, 1]; F[f'{c}__curv'][sl] = sm[:, 2]
                F[f'{c}__eff'][sl] = sm[:, 3]
                F[f'{c}__w21m'][sl] = w21[0]; F[f'{c}__w21s'][sl] = w21[1]; F[f'{c}__w21n'][sl] = w21[2]
                F[f'{c}__rsm'][sl] = rsm
        # ---- harmonized ----
        T, V, S = prep.harm[a]
        nb = _neigh(T, V, qi, k=1)
        if len(T):
            cs, cs2 = _cumstats(T, V)
            w45 = _window(T, V, cs, cs2, qi, 45); w21 = _window(T, V, cs, cs2, qi, 21)
            sm = local_poly(T, V, q, h=13.0, K=8, deg=2)
            rsm = robust_local_poly(T, V, q, h=16.0, K=9, deg=2, iters=2)[:, 0]
            idxp = np.clip(np.searchsorted(T, qi, 'left') - 1, 0, len(T) - 1)
            idxn = np.clip(np.searchsorted(T, qi, 'left'), 0, len(T) - 1)
            F['h_last_src'][sl] = S[idxp]; F['h_next_src'][sl] = S[idxn]
            F['h_w21m'][sl] = w21[0]; F['h_w21s'][sl] = w21[1]; F['h_w21n'][sl] = w21[2]
            F['h_w45m'][sl] = w45[0]; F['h_w45n'][sl] = w45[2]
            F['h_sm'][sl] = sm[:, 0]; F['h_slope'][sl] = sm[:, 1]
            F['h_curv'][sl] = sm[:, 2]; F['h_eff'][sl] = sm[:, 3]; F['h_rsm'][sl] = rsm
            p1, n1, pd1, nd1 = nb['p1'], nb['n1'], nb['pd1'], nb['nd1']
            tot = pd1 + nd1
            lin = np.where(np.isfinite(p1) & np.isfinite(n1) & (tot > 0),
                           p1 + (n1 - p1) * (pd1 / np.maximum(tot, 1e-9)), np.nan)
            F['h_p1'][sl] = p1; F['h_pd1'][sl] = pd1; F['h_n1'][sl] = n1; F['h_nd1'][sl] = nd1
            F['h_lin'][sl] = lin
            F['h_mid'][sl] = np.where(np.isfinite(p1) & np.isfinite(n1), 0.5 * (p1 + n1), np.nan)
            F['gap_left'][sl] = pd1; F['gap_right'][sl] = nd1
            F['gap_span'][sl] = pd1 + nd1
            F['is_edge'][sl] = (~np.isfinite(p1)).astype(float) + (~np.isfinite(n1)).astype(float)
            F['n_obs_aoi'][sl] = len(T)
        # per-sensor smooth on own scale
        for s in SENSORS:
            Ts, Vs, _ = ctx.obs[SENS_CH[s]][a]
            if len(Ts) >= 4:
                r = robust_local_poly(Ts, Vs, q, h=18.0, K=8, deg=2, iters=1)
                F[f'sm_{s}'][sl] = r[:, 0]; F[f'sm_{s}_eff'][sl] = r[:, 3]
            ds = np.full(len(sl), np.nan); dn = np.full(len(sl), np.nan)
            if len(Ts):
                i = np.searchsorted(Ts, qi, 'left')
                okp = i > 0; okn = i < len(Ts)
                ds[okp] = qi[okp] - Ts[np.clip(i[okp] - 1, 0, len(Ts) - 1)]
                dn[okn] = Ts[np.clip(i[okn], 0, len(Ts) - 1)] - qi[okn]
            F[f'ds_{s}'][sl] = ds; F[f'dn_{s}'][sl] = dn
        # ---- common mode / availability / weather ----
        for name in ['harm'] + SENSORS:
            rs = prep.res_sum[name][qi]; rc = prep.res_cnt[name][qi]
            F[f'cm_{name}'][sl] = np.where(rc > 0, rs / np.maximum(rc, 1), np.nan)
            F[f'cm_{name}_n'][sl] = rc
        for s in SENSORS:
            F[f'av_{s}'][sl] = prep.av_cnt[s][qi]
        alive = prep.aoi_alive[qi]
        F['av_any'][sl] = alive
        for s in SENSORS:
            F[f'av_{s}_f'][sl] = prep.av_cnt[s][qi] / np.maximum(alive, 1)
        F['ph5'][sl] = qi % 5; F['ph10'][sl] = qi % 10; F['ph16'][sl] = qi % 16
        F['ph16_8'][sl] = qi % 8
        F['ph_modis_ok'][sl] = ((qdoy[sl] - 1) % 16 == 0).astype(float)
        e = eras[sl]
        for s, mod, key in [('s2', 5, 'rate_s2_5'), ('s2', 10, 'rate_s2_10'),
                            ('landsat', 16, 'rate_landsat_16'), ('modis', 16, 'rate_modis_16')]:
            num, den = prep.phase[(s, mod)]
            F[key][sl] = num[a, e, qi % mod] / np.maximum(den[a, e, qi % mod], 1)
        grp = prep.wgroups.get(a, [a])
        wt = np.nanmean(prep.wt[grp][:, qi], 0); wp = np.nanmean(prep.wp[grp][:, qi], 0)
        F['wt_grp'][sl] = wt; F['wp_grp'][sl] = wp
        F['wt_med'][sl] = prep.wt_med[qi]; F['wp_med'][sl] = prep.wp_med[qi]
        for W, key in [(7, 'wp_c7'), (15, 'wp_c15'), (30, 'wp_c30')]:
            lo = np.maximum(qi - W, 0)
            seg = np.array([np.nanmean(prep.wp_med[l:h + 1]) if h >= l else np.nan
                            for l, h in zip(lo, qi)])
            F[key][sl] = seg * (W + 1)
        for W, key in [(15, 'wt_m15'), (30, 'wt_m30')]:
            lo = np.maximum(qi - W, 0)
            F[key][sl] = np.array([np.nanmean(prep.wt_med[l:h + 1]) if h >= l else np.nan
                                   for l, h in zip(lo, qi)])
        # ---- climatology excluding current year ----
        for s in SENSORS:
            Ts, Vs = prep.clim[s][a]
            F[f'clim_{s}'][sl] = _clim(Ts, Vs, qi, qyear[sl])
        Th, Vh = prep.harm_clim[a]
        cv, cn = _clim(Th, Vh, qi, qyear[sl], with_n=True)
        F['clim_harm'][sl] = cv; F['clim_harm_n'][sl] = cn
        F['cal_ls_a'][sl] = prep.cal[a]['landsat'][0]; F['cal_ls_b'][sl] = prep.cal[a]['landsat'][1]
        F['cal_md_a'][sl] = prep.cal[a]['modis'][0]; F['cal_md_b'][sl] = prep.cal[a]['modis'][1]
        if len(Th):
            yrs = 2010 + ((Th) // 365.25).astype(int)
            for yv in np.unique(qyear[sl]):
                mm = qyear[sl] == yv
                F['n_obs_year'][sl[mm]] = (yrs == yv).sum()
        F['doy'][sl] = qdoy[sl]; F['year'][sl] = qyear[sl]; F['tt'][sl] = qi
        ang = 2 * np.pi * qdoy[sl] / 365.25
        F['sin1'][sl] = np.sin(ang); F['cos1'][sl] = np.cos(ang)
        F['sin2'][sl] = np.sin(2 * ang); F['cos2'][sl] = np.cos(2 * ang)
        F['yfrac'][sl] = qdoy[sl] / 365.25
        F['aoi_code'][sl] = a
        hr = prep.harm_res[a]
        if hr is not None and len(hr[0]):
            Tr, Rr = hr
            nb2 = _neigh(Tr, Rr, qi, k=1)
            F['res_prev'][sl] = nb2['p1']; F['res_next'][sl] = nb2['n1']
            F['res_prev_dt'][sl] = nb2['pd1']; F['res_next_dt'][sl] = nb2['nd1']
            F['res_mean2'][sl] = np.nanmean(np.stack([nb2['p1'], nb2['n1']]), 0)
        rs_ = prep.res_sum['harm']; rc_ = prep.res_cnt['harm']
        F['cm_harm_med'][sl] = prep.res_med['harm'][qi]
        gid = int(prep._gid[a]); cid = int(prep._ncrop[a])
        F['cm_wgrp'][sl] = [prep.res_grp['harm'].get((gid, int(x)), np.nan) for x in qi]
        F['cm_crop'][sl] = [prep.res_crop['harm'].get((cid, int(x)), np.nan) for x in qi]
        for off, key in [(-1, 'cm_m1'), (1, 'cm_p1'), (-2, 'cm_m2'), (2, 'cm_p2')]:
            j = np.clip(qi + off, 0, TMAX - 1)
            F[key][sl] = np.where(rc_[j] > 0, rs_[j] / np.maximum(rc_[j], 1), np.nan)
        cs_ = np.concatenate([[0.0], np.cumsum(rs_)]); cc_ = np.concatenate([[0.0], np.cumsum(rc_)])
        lo_ = np.clip(qi - 2, 0, TMAX - 1); hi_ = np.clip(qi + 3, 0, TMAX)
        num_ = cs_[hi_] - cs_[lo_]; den_ = cc_[hi_] - cc_[lo_]
        F['cm_win5'][sl] = np.where(den_ > 0, num_ / np.maximum(den_, 1), np.nan)
        F['cm_win5_n'][sl] = den_
        r5 = prep.phase[('s2', 5)][0][a, e, qi % 5] / np.maximum(prep.phase[('s2', 5)][1][a, e, qi % 5], 1)
        rl = prep.phase[('landsat', 16)][0][a, e, qi % 16] / np.maximum(prep.phase[('landsat', 16)][1][a, e, qi % 16], 1)
        rmd = prep.phase[('modis', 16)][0][a, e, qi % 16] / np.maximum(prep.phase[('modis', 16)][1][a, e, qi % 16], 1)
        tot_ = r5 + (1 - r5) * rl + (1 - r5) * (1 - rl) * rmd
        F['p_cal_s2'][sl] = r5 / np.maximum(tot_, 1e-6)
        F['p_cal_ls'][sl] = (1 - r5) * rl / np.maximum(tot_, 1e-6)
        F['p_cal_md'][sl] = (1 - r5) * (1 - rl) * rmd / np.maximum(tot_, 1e-6)
        hb = F['h_rsm'][sl]
        phbv = {}
        for s_, kb, kn, ka in [('s2', 'phb_s2', 'phb_n_s2', 'base_s2_adj'),
                               ('landsat', 'phb_ls', 'phb_n_ls', 'base_ls_adj'),
                               ('modis', 'phb_md', 'phb_n_md', 'base_md_adj')]:
            nph = prep.PHMOD[s_]
            ph = (qi % nph) if s_ != 'modis' else np.clip((qdoy[sl] - 97) // 16, 0, nph - 1)
            b_, c_ = prep.phb[s_]
            phbv[s_] = b_[a, ph]
            F[kb][sl] = b_[a, ph]; F[kn][sl] = c_[a, ph]
            a0, b0 = prep.cal[a][s_]
            F[ka][sl] = (hb + b_[a, ph] - a0) / max(b0, 1e-6)
        F['phb_exp'][sl] = (F['p_cal_s2'][sl] * phbv['s2'] + F['p_cal_ls'][sl] * phbv['landsat']
                            + F['p_cal_md'][sl] * phbv['modis'])
        for s_, key in [('s2', 'base_s2'), ('landsat', 'base_ls'), ('modis', 'base_md')]:
            a0, b0 = prep.cal[a][s_]
            F[key][sl] = (hb - a0) / max(b0, 1e-6)
        cn = prep.crop[a]
        if cn in prep.crop_S:
            dgi = (qi % 365.25).astype(int) % 366
            S = prep.crop_S[cn][dgi] - prep.aoi_S[a][dgi]
            Wq = prep.crop_W[cn][dgi] - prep.aoi_W[a][dgi]
            F['clim_crop'][sl] = np.where(Wq > 1e-6, S / np.maximum(Wq, 1e-9), np.nan)
        F['lvl_harm'][sl] = np.nan
    # cross-AOI harmonized level at exact date
    lv_s = np.zeros(TMAX); lv_c = np.zeros(TMAX)
    for a in range(prep.A):
        T, V, _ = prep.harm[a]
        np.add.at(lv_s, T, V); np.add.at(lv_c, T, 1.0)
    F['lvl_harm'] = np.where(lv_c[qt] > 0, lv_s[qt] / np.maximum(lv_c[qt], 1), np.nan)
    F['lvl_harm_n'] = lv_c[qt]
    X = pd.DataFrame(F)
    # engineered combinations
    X['base_cm'] = X['h_rsm'] + X['cm_harm'].fillna(0.0)
    X['base_cm_phb'] = X['base_cm'] + X['phb_exp'].fillna(0.0)
    X['dev_lin_sm'] = X['h_lin'] - X['h_sm']
    X['dev_clim'] = X['h_rsm'] - X['clim_harm']
    X['prim_minus_h'] = X['primary_ndvi__lin'] - X['h_lin']
    X['s2_minus_ls'] = X['s2_ndvi__sm'] - X['landsat_ndvi__sm']
    X['s2_minus_md'] = X['s2_ndvi__sm'] - X['modis_ndvi__sm']
    return X


def _clim(T, V, qi, qyear, with_n=False, bw=12.0, rad=32):
    """Gaussian-kernel DOY climatology of series (T,V) excluding the query year."""
    nq = len(qi)
    out = np.full(nq, np.nan); cnt = np.zeros(nq)
    if len(T) == 0:
        return (out, cnt) if with_n else out
    doy = ((T % 365.25).astype(int))
    yr = 2010 + (T // 365.25).astype(int)
    qd = ((qi % 365.25).astype(int))
    d = np.abs(qd[:, None] - doy[None, :])
    d = np.minimum(d, 365 - d)
    w = np.where(d <= rad, np.exp(-0.5 * (d / bw) ** 2), 0.0)
    w = np.where(yr[None, :] == qyear[:, None], 0.0, w)
    s = w.sum(1)
    out = np.where(s > 1e-6, (w * V[None, :]).sum(1) / np.maximum(s, 1e-9), np.nan)
    cnt = (w > 1e-3).sum(1).astype(float)
    return (out, cnt) if with_n else out
