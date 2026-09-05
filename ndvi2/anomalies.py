"""Детекция негативных аномалий вегетации.

Логика:
  1. Ряд приводится к единой шкале Sentinel-2 (та же кросс-сенсорная калибровка,
     что и в модели восстановления) - иначе смена спутника выглядит как «аномалия».
  2. Норма считается только по ПРЕДЫДУЩИМ годам того же поля: робастная медиана и
     MAD в окне +-15 дней года, минимум 3 референсных года.
  3. Робастный z = (x - median) / max(1.4826*MAD, 0.03). Порог -1 - угнетение,
     -2 - критическая аномалия (соответствует шкале из постановки задачи).
  4. Событие требует устойчивости: >=2 точек подряд, длительность >=8 дней,
     разрыв между точками <=16 дней. Одиночный выброс событием не считается.
  5. severity = площадь отрицательного отклонения (NDVI-дни) * confidence.
  6. Reason codes сопоставляют событие с погодой и другими индексами. Причинность
     НЕ утверждается: коды описывают совпадение, а не доказанную причину.
"""
import numpy as np, pandas as pd

Z_WARN, Z_CRIT = -1.0, -2.0
MIN_REF_YEARS = 3
DOY_HALFWIN = 15
MIN_POINTS = 2
MIN_DURATION_DAYS = 8
MAX_INNER_GAP = 16
MAD_FLOOR = 0.03


def robust_climatology(hist: pd.DataFrame, doy: int, year: int):
    """hist: колонки doy, year, value (только прошлые годы того же поля)."""
    d = np.abs(hist['doy'].values - doy)
    d = np.minimum(d, 365 - d)
    m = (d <= DOY_HALFWIN) & (hist['year'].values < year)
    if m.sum() == 0:
        return np.nan, np.nan, 0
    v = hist['value'].values[m]
    nyears = len(np.unique(hist['year'].values[m]))
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    return med, max(mad, MAD_FLOOR), nyears


def score_series(series: pd.DataFrame):
    """series: anon_polygon_id, date, value (гармонизованный NDVI), is_observed.
    Возвращает точки со z-score и статусом."""
    s = series.sort_values('date').copy()
    s['doy'] = s['date'].dt.dayofyear
    s['year'] = s['date'].dt.year
    hist = s.loc[s['is_observed'], ['doy', 'year', 'value']].dropna()
    out = []
    for r in s.itertuples():
        med, mad, ny = robust_climatology(hist, r.doy, r.year)
        z = (r.value - med) / mad if ny >= MIN_REF_YEARS and np.isfinite(med) else np.nan
        out.append((med, mad, ny, z))
    s[['clim_med', 'clim_mad', 'n_ref_years', 'robust_z']] = pd.DataFrame(out, index=s.index)
    s['status'] = np.where(s.robust_z.isna(), 'нет нормы',
                    np.where(s.robust_z < Z_CRIT, 'критическая аномалия',
                      np.where(s.robust_z < Z_WARN, 'угнетение биомассы', 'штатное развитие')))
    return s


def detect_events(points: pd.DataFrame, weather: pd.DataFrame = None):
    """Склеивает устойчивые отрицательные периоды в события."""
    p = points.dropna(subset=['robust_z']).sort_values('date')
    ev, cur = [], []
    prev_date = None
    for r in p.itertuples():
        neg = r.robust_z < Z_WARN
        far = prev_date is not None and (r.date - prev_date).days > MAX_INNER_GAP
        same_year = prev_date is not None and r.date.year == prev_date.year
        if neg and cur and not far and same_year:
            cur.append(r)
        elif neg:
            if cur: ev.append(cur)
            cur = [r]
        else:
            if cur: ev.append(cur); cur = []
        prev_date = r.date
    if cur: ev.append(cur)
    rows = []
    for g in ev:
        if len(g) < MIN_POINTS: continue
        d0, d1 = g[0].date, g[-1].date
        dur = (d1 - d0).days
        if dur < MIN_DURATION_DAYS: continue
        z = np.array([x.robust_z for x in g])
        dev = np.array([x.value - x.clim_med for x in g])
        days = np.array([(x.date - d0).days for x in g], float)
        area = float(-np.trapezoid(np.minimum(dev, 0), days))
        obs = int(sum(bool(x.is_observed) for x in g))
        conf = min(1.0, 0.35 + 0.15 * obs + 0.05 * len(g))
        rows.append(dict(anon_polygon_id=g[0].anon_polygon_id, start=d0, end=d1,
                         duration_days=dur, n_points=len(g), n_observed=obs,
                         min_z=float(z.min()), mean_z=float(z.mean()),
                         negative_area=area, confidence=round(conf, 3),
                         severity=round(area * conf, 4),
                         level='критическая' if z.min() < Z_CRIT else 'угнетение'))
    df = pd.DataFrame(rows)
    if len(df) and weather is not None:
        df = add_reasons(df, weather)
    return df


def add_reasons(events: pd.DataFrame, weather: pd.DataFrame):
    """Reason codes: совпадение с погодной аномалией. Причинность не утверждается."""
    w = weather.copy()
    w['doy'] = w['date'].dt.dayofyear; w['year'] = w['date'].dt.year
    reasons = []
    for r in events.itertuples():
        sub = w[(w.anon_polygon_id == r.anon_polygon_id) &
                (w.date > r.start - pd.Timedelta(days=30)) & (w.date <= r.end)]
        codes = []
        if len(sub):
            ref = w[(w.anon_polygon_id == r.anon_polygon_id) & (w.year < r.start.year) &
                    (w.doy.between(max(sub.doy.min(), 1), sub.doy.max()))]
            if len(ref) > 30:
                pr = sub.era5_precip_mm.mean(); prr = ref.groupby('year').era5_precip_mm.mean()
                if np.isfinite(pr) and len(prr) >= 3 and pr < np.percentile(prr, 20):
                    codes.append('дефицит осадков (ниже 20-го перцентиля прошлых лет)')
                tm = sub.era5_temp_c.mean(); tmr = ref.groupby('year').era5_temp_c.mean()
                if np.isfinite(tm) and len(tmr) >= 3 and tm > np.percentile(tmr, 80):
                    codes.append('повышенная температура (выше 80-го перцентиля)')
        # Отличаем стресс от смены культуры: если отклонение держится почти весь
        # сезон и начинается в его первой трети - это скорее севооборот или пар.
        if r.duration_days >= 60 and r.start.dayofyear <= 150:
            codes.append('вероятна смена культуры или пар: отклонение охватывает почти весь сезон')
        elif r.duration_days >= 30:
            codes.append('длительное отклонение')
        if r.n_observed == 0: codes.append('период построен только на восстановленных точках')
        if not codes: codes.append('погодная аномалия не обнаружена; возможны уборка, смена культуры или качество съёмки')
        reasons.append('; '.join(codes))
    events = events.copy(); events['reason_codes'] = reasons
    events['explanation'] = [
        f"С {r.start:%Y-%m-%d} по {r.end:%Y-%m-%d} ({r.duration_days} дн.) NDVI поля {r.anon_polygon_id} "
        f"был ниже собственной исторической нормы, минимум z = {r.min_z:.2f}. Совпадения: {c}. "
        f"Это описание отклонения, а не установленная причина."
        for r, c in zip(events.itertuples(), reasons)]
    return events
