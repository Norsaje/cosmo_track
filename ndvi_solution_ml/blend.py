import warnings, json, glob, os
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.optimize import nnls
import sys
D = sys.argv[1] if len(sys.argv)>1 else 'work/cv3'
y = np.load(f'{D}/yev.npy'); mev = pd.read_pickle(f'{D}/mev.pkl'); X = pd.read_pickle(f'{D}/Xev.pkl')
P = {}
if os.path.exists(f'{D}/preds_m2.npy'):
    old = np.load(f'{D}/preds_m2.npy'); comp_old = json.load(open(f'{D}/res_m2.json'))['comp']
    P = {c: old[:, i] for i, c in enumerate(comp_old)}
for f in glob.glob(f'{D}/pred_*.npy'):
    P[os.path.basename(f)[5:-4]] = np.load(f)
names = [c for c in P if c not in ('midpoint','base_cm')]
A = np.stack([P[c] for c in names], 1)
def R(p, m=None): 
    m = slice(None) if m is None else m
    return float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))
print('individual:')
for c in names: print('  %-10s %.5f' % (c, R(P[c])))
aoi = mev.anon_polygon_id.values; ua = np.sort(np.unique(aoi))
rng = np.random.default_rng(0); fold = {a: i % 5 for i, a in enumerate(rng.permutation(ua))}
fa = np.array([fold[a] for a in aoi])
oof = np.zeros(len(y))
for k in range(5):
    tr = fa != k
    w, _ = nnls(A[tr], y[tr]); w = w / max(w.sum(), 1e-9)
    oof[~tr] = A[~tr] @ w
print('\nhonest 5-fold-AOI blend RMSE %.5f  gapscore %.2f' % (R(oof), 30 * max(0, 1 - R(oof) / 0.10)))
w, _ = nnls(A, y); w = w / max(w.sum(), 1e-9)
print('final weights:', {c: round(float(v), 4) for c, v in zip(names, w) if v > 1e-4})
print('in-sample blend %.5f' % R(A @ w))
# simple average of top models as robustness check
top = [c for c in ['lgb_a','lgb_c','lgb_d','experts','cat','mlp','lgb_b','lgb_e'] if c in P]
print('equal-average of %s: %.5f' % (top, R(np.mean([P[c] for c in top], 0))))
json.dump({'names': names, 'w': w.tolist(), 'honest': R(oof)}, open(f'{D}/blend.json', 'w'), indent=1)
