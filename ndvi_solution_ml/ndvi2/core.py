"""Core data structures: context with hidden rows, per-AOI channel arrays."""
import numpy as np, pandas as pd

DAY0 = pd.Timestamp('2010-01-01')

NDVI_CH = ['s2_ndvi', 'landsat_ndvi', 'modis_ndvi']
SENSORS = ['s2', 'landsat', 'modis']
DYNAMIC = ['s2_ndvi','s2_evi','s2_ndwi','landsat_ndvi','landsat_evi','landsat_ndwi',
           'modis_ndvi','modis_evi','era5_temp_c','era5_precip_mm','primary_ndvi']
# channels used for neighbour/window features
FEATCH = ['primary_ndvi','s2_ndvi','landsat_ndvi','modis_ndvi',
          's2_evi','landsat_evi','modis_evi','s2_ndwi','landsat_ndwi',
          'era5_temp_c','era5_precip_mm']


def load_raw(train_path, test_path):
    tr = pd.read_csv(train_path, parse_dates=['date'])
    te = pd.read_csv(test_path, parse_dates=['date'])
    tr['is_synthetic_gap'] = False
    tr['split'] = 'train'
    te['split'] = 'test'
    te['is_synthetic_gap'] = te['is_synthetic_gap'].astype(str).str.lower().eq('true')
    keep = ['anon_polygon_id','date','crop_type','split','is_synthetic_gap'] + DYNAMIC
    al = pd.concat([tr[keep], te[keep]], ignore_index=True)
    al['t'] = (al['date'] - DAY0).dt.days.astype(np.int32)
    al['doy'] = al['date'].dt.dayofyear.astype(np.int16)
    al['year'] = al['date'].dt.year.astype(np.int16)
    al = al.sort_values(['anon_polygon_id','t'], kind='mergesort').reset_index(drop=True)
    al['row_id'] = np.arange(len(al), dtype=np.int64)
    # true source of every observed primary value
    src = np.full(len(al), -1, np.int8)
    p = al['primary_ndvi'].values
    for k, c in enumerate(NDVI_CH):
        v = al[c].values
        m = (src < 0) & np.isfinite(p) & np.isfinite(v) & (np.abs(v - p) <= 1e-7)
        src[m] = k
    al['source'] = src
    return al


class Ctx:
    """Observation context: everything visible after hiding `hidden` row_ids."""

    def __init__(self, base: pd.DataFrame, hidden_rows: np.ndarray):
        self.base = base
        n = len(base)
        h = np.zeros(n, bool)
        if hidden_rows is not None and len(hidden_rows):
            h[hidden_rows] = True
        self.hidden = h
        self.aoi_codes, self.aoi_names = pd.factorize(base['anon_polygon_id'], sort=True)
        self.n_aoi = len(self.aoi_names)
        self.t = base['t'].values.astype(np.int64)
        # per-aoi row slices (base is sorted by aoi,t)
        order = np.argsort(self.aoi_codes, kind='mergesort')
        assert (np.diff(self.aoi_codes[order]) >= 0).all()
        self.aoi_start = np.searchsorted(self.aoi_codes[order], np.arange(self.n_aoi))
        self.aoi_end = np.searchsorted(self.aoi_codes[order], np.arange(self.n_aoi), side='right')
        self._order = order
        # per aoi/channel observed arrays
        self.obs = {}
        vals = {c: base[c].values.astype(np.float64) for c in FEATCH}
        for c in FEATCH:
            v = vals[c].copy()
            v[h] = np.nan
            self.obs[c] = []
            for a in range(self.n_aoi):
                idx = order[self.aoi_start[a]:self.aoi_end[a]]
                tt = self.t[idx]; vv = v[idx]
                m = np.isfinite(vv)
                self.obs[c].append((tt[m].astype(np.int64), vv[m], idx[m]))
        self.vals_masked = {c: np.where(h, np.nan, vals[c]) for c in FEATCH}
        self.source_masked = np.where(h, -1, base['source'].values)

    def aoi_index(self, names):
        lut = {n: i for i, n in enumerate(self.aoi_names)}
        return np.array([lut[x] for x in names], np.int32)
