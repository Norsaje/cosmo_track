from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from .data import CROPS, SENSORS, VALUES, query_only


def _nanmean(a, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def temporal(query, context, column, prefix, k=3):
    """Только строго соседние даты в том же сезоне; query уже скрыт глобально."""
    names = []
    for side in ["prev", "next"]:
        for j in range(k):
            names += [f"{prefix}_{side}{j+1}", f"{prefix}_{side}{j+1}_days"]
    names += [f"{prefix}_{s}" for s in ["linear", "midpoint", "nearest", "span", "slope", "count",
                                       "mean21", "std21", "mean45", "robust_linear"]]
    out = pd.DataFrame(np.nan, index=query.index, columns=names)
    groups = {p: g.sort_values("day") for p, g in context[context[column].notna()].groupby("anon_polygon_id")}
    for p, q in query.groupby("anon_polygon_id", sort=False):
        b = groups.get(p)
        if b is None or b.empty:
            continue
        t, v, by = b.day.to_numpy(float), b[column].to_numpy(float), b.year.to_numpy(int)
        tq, qy = q.day.to_numpy(float), q.year.to_numpy(int)
        left = np.searchsorted(t, tq, "left")
        right = np.searchsorted(t, tq, "right")
        vals, ds = [], []
        for sign, starts, side in [(-1, left - 1, "prev"), (1, right, "next")]:
            for j in range(k):
                pos = starts + sign*j
                safe = np.clip(pos, 0, len(t)-1)
                ok = (pos >= 0) & (pos < len(t)) & (by[safe] == qy)
                vv = np.where(ok, v[safe], np.nan)
                dd = np.where(ok, np.abs(t[safe]-tq), np.nan)
                out.loc[q.index, f"{prefix}_{side}{j+1}"] = vv
                out.loc[q.index, f"{prefix}_{side}{j+1}_days"] = dd
                vals.append(vv); ds.append(dd)
        prev, nxt, dl, dr = vals[0], vals[k], ds[0], ds[k]
        mid = _nanmean(np.array([prev, nxt]))
        span = dl + dr
        linear = np.where(np.isfinite(span) & (span > 0), prev + (nxt-prev)*dl/span, mid)
        near = np.where(np.isnan(nxt) | (dl <= dr), prev, nxt)
        out.loc[q.index, f"{prefix}_linear"] = linear
        out.loc[q.index, f"{prefix}_midpoint"] = mid
        out.loc[q.index, f"{prefix}_nearest"] = near
        out.loc[q.index, f"{prefix}_span"] = span
        out.loc[q.index, f"{prefix}_slope"] = (nxt-prev)/span
        counts = b.groupby("year").size()
        out.loc[q.index, f"{prefix}_count"] = q.year.map(counts).fillna(0).to_numpy()
        vmat, dmat = np.asarray(vals), np.asarray(ds)
        for radius in [21, 45]:
            nearv = np.where(dmat <= radius, vmat, np.nan)
            out.loc[q.index, f"{prefix}_mean{radius}"] = _nanmean(nearv)
            if radius == 21:
                mu = _nanmean(nearv)
                out.loc[q.index, f"{prefix}_std21"] = np.sqrt(_nanmean((nearv-mu)**2))
        # Физический диапазон применяется только к дополнительному признаку, не к y.
        if "ndvi" in column:
            pc, nc = np.clip(prev, -1, 1), np.clip(nxt, -1, 1)
            out.loc[q.index, f"{prefix}_robust_linear"] = np.where(
                np.isfinite(span) & (span > 0), pc+(nc-pc)*dl/span, _nanmean(np.array([pc,nc])))
        else:
            out.loc[q.index, f"{prefix}_robust_linear"] = linear
    return out


def seasonal(query, context):
    names = [f"clim_{s}_{v}" for s in ["primary", *SENSORS] for v in ["mean", "std", "n"]]
    out = pd.DataFrame(np.nan, index=query.index, columns=names)
    groups = dict(tuple(context.groupby("anon_polygon_id")))
    for p, q in query.groupby("anon_polygon_id", sort=False):
        b = groups.get(p)
        if b is None:
            continue
        for s in ["primary", *SENSORS]:
            col = f"{s}_ndvi"
            bb = b[b[col].notna() & b[col].between(-1, 1)]
            if bb.empty:
                continue
            # Усреднение отдельно по годам не позволяет плотному S2-сезону подавить остальные.
            years = []
            for y, by in bb.groupby("year"):
                dist = abs(q.doy.to_numpy()[:,None] - by.doy.to_numpy()[None,:])
                w = np.exp(-.5*(dist/12)**2) * (dist <= 30)
                w[q.year.to_numpy() == y, :] = 0
                denom = w.sum(axis=1)
                years.append(np.divide(w @ by[col].to_numpy(), denom,
                                       out=np.full(len(q), np.nan), where=denom>0))
            mat = np.asarray(years)
            mu = _nanmean(mat)
            out.loc[q.index, f"clim_{s}_mean"] = mu
            out.loc[q.index, f"clim_{s}_std"] = np.sqrt(_nanmean((mat-mu)**2))
            out.loc[q.index, f"clim_{s}_n"] = np.isfinite(mat).sum(axis=0)
    return out


def acquisition(query, context):
    """Наблюдения других AOI на ту же дату: свидетельство источника, не географии."""
    out = pd.DataFrame(index=query.index)
    for s in ["primary", *SENSORS]:
        col = f"{s}_ndvi"
        b = context[context[col].notna()]
        for label, keys in [("all", ["date"]), ("crop", ["date", "crop_type"])]:
            ag = b.groupby(keys)[col].agg(["mean", "std", "count", "median"])
            merged = query[keys].merge(ag, on=keys, how="left", sort=False)
            for stat in ag.columns:
                out[f"date_{label}_{s}_{stat}"] = merged[stat].to_numpy()
    return out


def donors(query, context):
    """Взвешенная реконструкция по реально наблюдаемым синхронным рядам.

    Координаты не известны; корреляция рядов не интерпретируется как близость полей.
    Парные калибровки вычисляются заново после глобального удаления маски.
    """
    out = pd.DataFrame(index=query.index)
    for s in SENSORS:
        for n in [3, 10]:
            out[f"donor_{s}_{n}"] = np.nan
            out[f"donor_{s}_{n}_n"] = 0.
        out[f"donor_{s}_confidence"] = np.nan
        b = context[context[f"{s}_ndvi"].between(-1, 1)]
        if b.empty:
            continue
        pivot = b.pivot(index="date", columns="anon_polygon_id", values=f"{s}_ndvi")
        arr = pivot.to_numpy(float)
        for p, q in query.groupby("anon_polygon_id", sort=False):
            if p not in pivot.columns:
                continue
            target = arr[:, pivot.columns.get_loc(p)]
            weights, predictions = [], []
            dates = pivot.index.get_indexer(q.date)
            ok_date = dates >= 0
            for j, other in enumerate(pivot.columns):
                if other == p:
                    continue
                donor = arr[:, j]
                good = np.isfinite(target) & np.isfinite(donor)
                n = int(good.sum())
                if n < 12:
                    continue
                x, y = donor[good], target[good]
                dx, dy = x-x.mean(), y-y.mean()
                xx, yy, xy = dx@dx, dy@dy, dx@dy
                corr = xy/np.sqrt(max(xx*yy, 1e-10))
                if corr < .2:
                    continue
                # Слабые калибровки сжимаем к тождественному преобразованию.
                slope = np.clip((xy + .15)/(xx + .15), .2, 1.8)
                intercept = y.mean() - slope*x.mean()
                pred = np.full(len(q), np.nan)
                pred[ok_date] = intercept + slope*donor[dates[ok_date]]
                predictions.append(pred)
                weights.append(max(corr, 0)**4 * n/(n+30))
            if not weights:
                continue
            order = np.argsort(weights)[::-1]
            pm, ws = np.asarray(predictions)[order], np.asarray(weights)[order]
            for top in [3, 10]:
                w = ws[:top,None] * np.isfinite(pm[:top])
                denom = w.sum(axis=0)
                result = np.divide((np.nan_to_num(pm[:top])*w).sum(axis=0), denom,
                                   out=np.full(len(q), np.nan), where=denom>0)
                out.loc[q.index, f"donor_{s}_{top}"] = result
                out.loc[q.index, f"donor_{s}_{top}_n"] = np.isfinite(pm[:top]).sum(axis=0)
            out.loc[q.index, f"donor_{s}_confidence"] = ws[:3].mean()
    return out


def build_features(queries, context, use_donors=True):
    q = query_only(queries)
    # Это ключевой барьер от утечки: даже забытая вызывающим кодом маска остановит запуск.
    if context.loc[context.index.intersection(q.index), VALUES].notna().any().any():
        raise ValueError("Query не скрыт во всех динамических каналах контекста")
    f = pd.DataFrame(index=q.index)
    f["year"], f["doy"] = q.year, q.doy
    f["crop"] = q.crop_type.map({x:i for i,x in enumerate(CROPS)}).fillna(-1)
    for period in [5, 8, 10, 16, 32, 365.25, 182.625]:
        # Абсолютная дата сохраняет орбитальную фазу на границе года.
        for fun, name in [(np.sin,"sin"), (np.cos,"cos")]:
            f[f"calendar_{name}_{period}"] = fun(2*np.pi*q.day/period)
    chunks = [f]
    for col in VALUES:
        prefix = col.removesuffix("_ndvi")
        chunks.append(temporal(q, context, col, prefix, 4 if col.endswith("ndvi") else 2))
    chunks += [seasonal(q,context), acquisition(q,context)]
    if use_donors:
        chunks.append(donors(q,context))
    f = pd.concat(chunks,axis=1)
    for s in SENSORS:
        f[f"{s}_minus_primary"] = f[f"{s}_linear"] - f.primary_linear
        f[f"{s}_minus_clim"] = f[f"{s}_linear"] - f[f"clim_{s}_mean"]
        f[f"{s}_phase16_prev"] = np.cos(2*np.pi*f[f"{s}_prev1_days"]/16)
        f[f"{s}_phase16_next"] = np.cos(2*np.pi*f[f"{s}_next1_days"]/16)
        if use_donors:
            f[f"{s}_donor_disagreement"] = f[f"donor_{s}_3"] - f[f"{s}_linear"]
    f["primary_minus_clim"] = f.primary_linear - f.clim_primary_mean
    f["one_sided"] = (f.primary_prev1.isna() | f.primary_next1.isna()).astype(int)
    f["source_spread"] = f[[f"{s}_linear" for s in SENSORS]].std(axis=1)
    f = f.replace([np.inf,-np.inf], np.nan).astype(np.float32)
    if f.columns.duplicated().any():
        raise ValueError("Повтор имени признака")
    return f


def baseline(x, kind="linear"):
    return x[f"primary_{kind}"].fillna(x.clim_primary_mean).fillna(.5).to_numpy(float)
