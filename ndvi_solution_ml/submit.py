"""Собирает submission.csv из сохранённых предсказаний финальных моделей и весов смеси."""
import warnings, json, os, glob, hashlib
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd

CV, FIN = 'work/cv3', 'work/final'
bl = json.load(open(f'{CV}/blend.json'))
names, w = bl['names'], np.array(bl['w'])
mq = pd.read_pickle(f'{FIN}/mq.pkl')
P, W, used = [], [], []
for n, wi in zip(names, w):
    if wi <= 1e-6: continue
    f = f'{FIN}/pred_{n}.npy'
    if not os.path.exists(f):
        print(f'  пропускаю {n}: финальные предсказания ещё не готовы'); continue
    P.append(np.load(f)); W.append(wi); used.append(n)
W = np.array(W); W = W / W.sum()
pred = np.stack(P, 1) @ W
print('компоненты:', {n: round(float(x), 4) for n, x in zip(used, W)})
assert len(pred) == 2323 and np.isfinite(pred).all()
sub = pd.DataFrame({'anon_polygon_id': mq.anon_polygon_id.values,
                    'date': pd.to_datetime(mq.date).dt.strftime('%Y-%m-%d'),
                    'primary_ndvi_pred': pred})
# строгая проверка формата
te = pd.read_csv('data/test.csv', parse_dates=['date'])
g = te[te.is_synthetic_gap.astype(str).str.lower().eq('true')][['anon_polygon_id', 'date']]
g['date'] = g.date.dt.strftime('%Y-%m-%d')
assert list(sub.columns) == ['anon_polygon_id', 'date', 'primary_ndvi_pred']
assert len(sub) == 2323 == len(g)
assert not sub.duplicated(['anon_polygon_id', 'date']).any()
m = sub.merge(g, on=['anon_polygon_id', 'date'], how='outer', indicator=True, validate='one_to_one')
assert (m._merge == 'both').all(), 'ключи не совпадают с контрольными строками теста'
os.makedirs('outputs', exist_ok=True)
sub.to_csv('outputs/submission.csv', index=False, encoding='utf-8', lineterminator='\n')
h = hashlib.sha256(open('outputs/submission.csv', 'rb').read()).hexdigest()
print('outputs/submission.csv  строк:', len(sub), ' SHA256:', h)
print(sub.primary_ndvi_pred.describe().round(4).to_string())
