"""Build feature matrices for the real submission: train on everything visible,
predict the 2323 rows with is_synthetic_gap == True."""
import warnings, os, time, argparse
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from ndvi2.pipeline import weather_groups, featurize, build_training_set

ap = argparse.ArgumentParser()
ap.add_argument('--rounds', type=int, default=5)
ap.add_argument('--nblocks', type=int, default=7)
ap.add_argument('--seed', type=int, default=17)
ap.add_argument('--out', default='work/final')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
base = pd.read_pickle('base.pkl')
wg, names = weather_groups(base)
gap_rows = np.flatnonzero(base['is_synthetic_gap'].values)
print('gap rows', len(gap_rows))
assert len(gap_rows) == 2323
t0 = time.time()
Xtr, ytr, mtr = build_training_set(base, np.array([], int), a.nblocks,
                                   list(range(a.seed, a.seed + a.rounds)), wg)
print('train', Xtr.shape, round(time.time()-t0, 1), 's', flush=True)
Xq = featurize(base, np.array([], int), gap_rows, wg)
mq = base.loc[gap_rows, ['anon_polygon_id', 'date', 'year']].reset_index(drop=True)
Xtr.to_pickle(f'{a.out}/Xtr.pkl'); np.save(f'{a.out}/ytr.npy', ytr); mtr.to_pickle(f'{a.out}/mtr.pkl')
Xq.to_pickle(f'{a.out}/Xq.pkl'); mq.to_pickle(f'{a.out}/mq.pkl')
np.save(f'{a.out}/gap_rows.npy', gap_rows)
print('saved', a.out)
