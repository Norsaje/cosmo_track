import numpy as np

def local_poly(T, V, q, h=14.0, K=9, deg=2, exclude_self=False, W=None):
    """Local polynomial fit of V(T) evaluated at q.
    Returns (value, slope, curv, eff_n, wsum)."""
    nq = len(q)
    out = np.full((nq, 5), np.nan)
    n = len(T)
    if n == 0 or nq == 0:
        return out
    idx = np.searchsorted(T, q)
    half = K
    lo = np.clip(idx - half, 0, max(n - 1, 0))
    off = np.arange(-half, half + 1)
    cols = lo[:, None] + (off + half)[None, :]
    valid = (cols >= 0) & (cols < n)
    cols = np.clip(cols, 0, n - 1)
    tt = T[cols].astype(np.float64)
    vv = V[cols]
    dx = tt - q[:, None]
    w = np.exp(-0.5 * (dx / h) ** 2)
    if W is not None:
        w = w * W[cols]
    w = np.where(valid, w, 0.0)
    if exclude_self:
        w = np.where(np.abs(dx) < 1e-9, 0.0, w)
    # deduplicate columns clipped to the same index
    first = np.zeros_like(valid)
    first[:, 0] = True
    first[:, 1:] = cols[:, 1:] != cols[:, :-1]
    w = np.where(first, w, 0.0)
    wsum = w.sum(1)
    eff = (w.sum(1) ** 2) / np.maximum((w ** 2).sum(1), 1e-12)
    p = deg + 1
    X = np.stack([dx ** k for k in range(p)], -1)          # (nq,K2,p)
    XW = X * w[:, :, None]
    A = np.einsum('nkp,nkq->npq', XW, X)
    b = np.einsum('nkp,nk->np', XW, vv)
    A = A + np.eye(p)[None] * 1e-6 * np.maximum(wsum, 1e-9)[:, None, None]
    ok = wsum > 1e-8
    beta = np.full((nq, p), np.nan)
    if ok.any():
        try:
            beta[ok] = np.linalg.solve(A[ok], b[ok][:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            for i in np.nonzero(ok)[0]:
                try:
                    beta[i] = np.linalg.lstsq(A[i], b[i], rcond=None)[0]
                except Exception:
                    pass
    bad = ~np.isfinite(beta[:, 0])
    if bad.any():
        wm = np.where(wsum[bad] > 1e-8, (w[bad] * vv[bad]).sum(1) / np.maximum(wsum[bad], 1e-12), np.nan)
        beta[bad, 0] = wm
    out[:, 0] = beta[:, 0]
    out[:, 1] = beta[:, 1] if p > 1 else np.nan
    out[:, 2] = beta[:, 2] if p > 2 else np.nan
    out[:, 3] = eff
    out[:, 4] = wsum
    return out


def robust_local_poly(T, V, q, h=14.0, K=9, deg=2, exclude_self=False, iters=2, c=0.06):
    W = np.ones(len(T))
    res = local_poly(T, V, q, h, K, deg, exclude_self, W)
    if iters <= 0 or len(T) < 6:
        return res
    fit = local_poly(T, V, T.astype(np.float64), h, K, deg, exclude_self=False, W=W)[:, 0]
    for _ in range(iters):
        r = V - fit
        s = np.nanmedian(np.abs(r)) * 1.4826
        s = max(s, c)
        u = np.clip(np.abs(r) / (4.0 * s), 0, 1)
        W = (1 - u ** 2) ** 2
        fit = local_poly(T, V, T.astype(np.float64), h, K, deg, exclude_self=False, W=W)[:, 0]
    return local_poly(T, V, q, h, K, deg, exclude_self, W)
