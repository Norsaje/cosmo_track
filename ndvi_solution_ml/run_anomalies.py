"""Строит гармонизованный ряд по всем полям и выделяет негативные аномальные периоды."""
import warnings, os
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from ndvi2.core import Ctx
from ndvi2.features import Prep
from ndvi2.pipeline import weather_groups
from ndvi2.anomalies import score_series, detect_events

base = pd.read_pickle('base.pkl')
wg, names = weather_groups(base)
ctx = Ctx(base, np.array([], int))
prep = Prep(ctx, wg)
# восстановленные значения контрольных точек подмешиваем как is_observed=False
sub = pd.read_csv('outputs/submission.csv', parse_dates=['date'])
# восстановленное значение дано в шкале primary; переводим в шкалу S2 ожиданием
# по вероятностям источника, иначе точка сместится относительно нормы
evp = np.load('work/final/evp.npy')
mq = pd.read_pickle('work/final/mq.pkl')
sub = sub.merge(mq.assign(_i=np.arange(len(mq))), on=['anon_polygon_id'], how='left') \
         .drop_duplicates(['anon_polygon_id', 'date']) if False else sub
pmap = {(r.anon_polygon_id, pd.Timestamp(r.date)): evp[i] for i, r in enumerate(mq.itertuples())}
rec = {}
for r in sub.itertuples():
    rec[(r.anon_polygon_id, r.date)] = (r.primary_ndvi_pred, pmap.get((r.anon_polygon_id, r.date)))

pts_all, ev_all = [], []
for a in range(prep.A):
    aoi = ctx.aoi_names[a]
    T, V, S = prep.harm[a]
    if len(T) < 40: continue
    d = pd.DataFrame({'anon_polygon_id': aoi,
                      'date': pd.Timestamp('2010-01-01') + pd.to_timedelta(T, 'D'),
                      'value': V, 'is_observed': True})
    cal = prep.cal[a]
    add = []
    for k, (v, pr) in rec.items():
        if k[0] != aoi: continue
        if pr is None:
            vv = v
        else:
            vv = sum(pr[j] * (cal[s_][0] + cal[s_][1] * v)
                     for j, s_ in enumerate(['s2', 'landsat', 'modis']))
        add.append((aoi, k[1], vv, False))
    if add:
        d2 = pd.DataFrame(add, columns=['anon_polygon_id', 'date', 'value', 'is_observed'])
        d = pd.concat([d, d2], ignore_index=True)
    d = d.sort_values('date')
    p = score_series(d)
    w = base[base.anon_polygon_id == aoi][['anon_polygon_id', 'date', 'era5_temp_c', 'era5_precip_mm']]
    e = detect_events(p, w)
    pts_all.append(p)
    if len(e): ev_all.append(e)
pts = pd.concat(pts_all, ignore_index=True)
ev = pd.concat(ev_all, ignore_index=True) if ev_all else pd.DataFrame()
os.makedirs('outputs', exist_ok=True)
pts.to_csv('outputs/anomaly_points.csv', index=False)
ev.sort_values('severity', ascending=False).to_csv('outputs/anomaly_events.csv', index=False)
print('точек:', len(pts), ' статусы:', pts.status.value_counts().to_dict())
print('событий:', len(ev))
if len(ev):
    print(ev.level.value_counts().to_dict())
    print('\nТОП-5 по severity:')
    print(ev.sort_values('severity', ascending=False)
            .head(5)[['anon_polygon_id','start','end','duration_days','min_z','severity','reason_codes']]
            .to_string(index=False))
