from libraries import np, Lasso, LassoCV, LCV_KW, smatrix, ledoit_wolf_shrinkage
from helpers import alphakn_threshold, stop_early, ebh, _mask, _scale


# Model-X knockoffs as in the R package `knockoff` (knockoff.filter defaults)

def corpcor_cov_shrink(X):
    n, p = X.shape
    h1, h1w2 = n / (n - 1.0), 1.0 / (n - 1.0)
    xc = X - X.mean(axis=0)
    v = h1 * np.mean(xc ** 2, axis=0)
    zz = xc ** 2
    q1 = zz.mean(axis=0); q2 = (zz ** 2).mean(axis=0) - q1 ** 2
    target = np.median(v)
    den = np.sum((q1 - target / h1) ** 2)
    lam_var = 1.0 if den == 0 else float(np.clip(q2.sum() / den * h1w2, 0, 1))
    v_s = lam_var * target + (1 - lam_var) * v
    xs = xc / np.sqrt(v)
    xsw = xs / np.sqrt(n)
    G = xsw @ xsw.T
    colsq = np.sum(xsw ** 2, axis=0)
    sE2R = np.sum(G * G) - np.sum(colsq ** 2)
    B = xs ** 2 / np.sqrt(n)
    sER2 = np.sum(B.sum(axis=1) ** 2) - np.sum(B ** 2)
    lam = 1.0 if sE2R == 0 else float(np.clip((sER2 - sE2R) / sE2R * h1w2, 0, 1))
    R = (1 - lam) * h1 * (xsw.T @ xsw)
    np.fill_diagonal(R, 1.0)
    sd = np.sqrt(v_s)
    return R * np.outer(sd, sd), lam, lam_var


def r_second_order_moments(X):
    mu = X.mean(axis=0)
    Sigma = np.cov(X, rowvar=False, ddof=1)
    if (X.shape[0] <= X.shape[1]
            or np.linalg.eigvalsh(Sigma)[0] <= 1e-8 * np.trace(Sigma) / Sigma.shape[0]):
        Sigma = corpcor_cov_shrink(X)[0]
    return mu, Sigma


def r_asdp_smatrix(Sigma, max_block=500):
    return smatrix.compute_smatrix(Sigma, method="sdp", max_block=max_block,
                                   dsdp_warning=False)


def stat_glmnet_coefdiff(X, Xk, y, rng, nlambda=100, cv_folds=10, lam=None):
    n, p = X.shape
    swap = rng.binomial(1, 0.5, p).astype(bool)
    Xs = np.where(swap, Xk, X); Xks = np.where(swap, X, Xk)
    Z = _scale(np.hstack([Xs, Xks]))
    if lam is None:
        lam_max = np.abs(Z.T @ y).max() / n
        grid = lam_max * (1.0 / 2000.0) ** (np.arange(nlambda) / nlambda)
        fit = LassoCV(alphas=grid, cv=cv_folds, fit_intercept=True,
                      max_iter=20_000, tol=1e-4).fit(Z, y)
        lam, coef = float(fit.alpha_), fit.coef_
    else:
        coef = Lasso(alpha=lam, fit_intercept=True, max_iter=20_000,
                     tol=1e-4).fit(Z, y).coef_
    W = np.abs(coef[:p]) - np.abs(coef[p:])
    return W * (1 - 2 * swap), lam


def r_gaussian_knockoff_sampler(X, mu, Sigma, S):
    n, p = X.shape
    d = np.diag(S)
    SigInv_s = np.linalg.solve(Sigma, np.diag(d))
    mu_k = X - (X - mu) @ SigInv_s
    Sk = 2 * np.diag(d) - np.diag(d) @ SigInv_s
    w, V = np.linalg.eigh((Sk + Sk.T) / 2)
    L = V * np.sqrt(np.clip(w, 0, None))          # Sk = L L'
    return lambda rng: mu_k + rng.standard_normal((n, p)) @ L.T


def knockoff_arms_R(X, y, q_star, gamma, M, seed, nlambda=100, offset=1):
    p = X.shape[1]
    mu, Sigma = r_second_order_moments(X)
    S = r_asdp_smatrix(Sigma)
    draw = r_gaussian_knockoff_sampler(X, mu, Sigma, S)
    rng = np.random.default_rng(seed)
    W, lam = stat_glmnet_coefdiff(X, draw(rng), y, rng, nlambda=nlambda)
    T = alphakn_threshold(W, q_star, offset)
    ko = W >= T if np.isfinite(T) else np.zeros(p, dtype=bool)
    E = np.zeros((M, p))
    for m in range(M):
        Wm, _ = stat_glmnet_coefdiff(X, draw(rng), y, rng, lam=lam)
        t = stop_early(Wm, gamma, offset)
        E[m] = (Wm >= t) / (1.0 + np.sum(Wm <= -t))
    return ko, _mask(ebh(p * E.mean(axis=0), q_star), p)



class LowRankKnockoffs:
    def __init__(self, X, tol=1e-10):
        n, p = X.shape
        mu = X.mean(axis=0)
        sd = X.std(axis=0); sd[sd == 0] = 1.0
        Z = (X - mu) / sd
        self.Z = Z
        self.delta = float(ledoit_wolf_shrinkage(Z, assume_centered=True))
        _, d, Vt = np.linalg.svd(Z / np.sqrt(n), full_matrices=False)
        keep = d ** 2 > tol * max(1.0, d[0] ** 2)
        self.Vt = Vt[keep]
        e = self.delta + (1.0 - self.delta) * d[keep] ** 2
        full_rank = self.Vt.shape[0] >= p
        lam_min = float(e.min()) if full_rank else self.delta
        s = min(1.0, 2.0 * lam_min)
        a = 2.0 * s - s ** 2 / self.delta
        b = s ** 2 * (1.0 / self.delta - 1.0 / e)
        if full_rank:                      
            self.a, self.b = 0.0, np.maximum(a + b, 0.0)
        else:                               
            self.a, self.b = max(a, 0.0), np.maximum(b, 0.0)
        self.s = s
        ZV = Z @ self.Vt.T
        self.mean = Z - s * (Z / self.delta + (ZV * (1.0 / e - 1.0 / self.delta)) @ self.Vt)

    def sample(self, rng):
        n, p = self.Z.shape
        out = self.mean + (rng.standard_normal((n, self.Vt.shape[0]))
                           * np.sqrt(self.b)) @ self.Vt
        if self.a > 0:
            out += np.sqrt(self.a) * rng.standard_normal((n, p))
        return out


def _W(X, Xk, y, alpha=None):
    p = X.shape[1]; Z = np.hstack([X, Xk])
    if alpha is None:
        f = LassoCV(cv=3, fit_intercept=False, tol=1e-3, max_iter=20_000,
                    **LCV_KW).fit(Z, y)
        return np.abs(f.coef_[:p]) - np.abs(f.coef_[p:]), float(f.alpha_)
    c = Lasso(alpha=alpha, fit_intercept=False, tol=1e-3,
              max_iter=20_000).fit(Z, y).coef_
    return np.abs(c[:p]) - np.abs(c[p:]), alpha


def knockoff_arms_lowrank(X, y, q_star, gamma, M, seed, offset=1):
    p = X.shape[1]
    K = LowRankKnockoffs(X)
    draw = lambda s: K.sample(np.random.default_rng(s))
    W, alpha = _W(K.Z, draw(seed), y)
    tau = alphakn_threshold(W, q_star, offset)
    ko = W >= tau if np.isfinite(tau) else np.zeros(p, dtype=bool)
    E = np.zeros((M, p))
    for m in range(M):
        Wm, _ = _W(K.Z, draw(seed + 1 + m), y, alpha=alpha)
        t = stop_early(Wm, gamma, offset)
        E[m] = (Wm >= t) / (1.0 + np.sum(Wm <= -t))
    return ko, _mask(ebh(p * E.mean(axis=0), q_star), p)


# Gaussian Mirror: port of gm() from the Xing, Zhao and Liu (2023) supplement

def gm_reject(w, q):
    p = w.size
    sel = np.zeros(p, dtype=bool)
    nz = np.flatnonzero(w != 0)
    if nz.size == 0:
        return sel
    wnz = w[nz]; order = np.argsort(-wnz)
    n_pos = int((w > 0).sum())
    if n_pos == 0:
        return sel
    est = np.array([np.sum(wnz < -wnz[order[i]]) / (i + 2) for i in range(n_pos)])
    over = np.flatnonzero(est - q > 0)
    k = n_pos if over.size == 0 else int(over[0])
    if k > 0:
        sel[nz[order[:k]]] = True
    return sel


def _gm_scale_factors(X, nz):
    n, p = X.shape
    pp = np.ones(p)
    if nz.size == 0:
        return pp
    U0 = np.linalg.svd(X[:, nz], full_matrices=False)[0]
    P0 = U0[:, : int(min(0.95 * nz.size, n))]
    out = np.setdiff1d(np.arange(p), nz)
    pp[out] = 1.0 - np.sum((P0.T @ X[:, out]) ** 2, axis=0) / n
    low = out[pp[out] < 0.05]                  # gm()'s fallback
    if low.size:
        pp[low] = 1.0 - np.sum((U0[:, :10].T @ X[:, low]) ** 2, axis=0) / n
    for j in nz:
        jidx = nz[nz != j]; xj = X[:, j]
        if jidx.size == 0:
            pp[j] = 1.0; continue
        U = np.linalg.svd(X[:, jidx], full_matrices=False)[0]
        P = U[:, : int(min(0.95 * jidx.size, n))]
        pp[j] = 1.0 - float(np.sum((P.T @ xj) ** 2)) / n
        if pp[j] < 0.05:
            pp[j] = 1.0 - float(np.sum((U[:, :10].T @ xj) ** 2)) / n
    return pp


def gaussian_mirror_select(X, y, q, rng, nlambda=100, cv_folds=10, precompute=True):
    n, p = X.shape
    X = _scale(X)
    lam_max = np.abs(X.T @ y).max() / n ** 1.5
    k = np.arange(nlambda) / nlambda
    lam_grid = lam_max * (1e-2) ** k
    cv = LassoCV(alphas=lam_grid, cv=cv_folds, fit_intercept=False,
                 max_iter=100_000, tol=1e-4).fit(X, y)
    lam_ast = float(cv.alpha_)
    coef_cv = cv.coef_.copy()                 
    coef = coef_cv.copy()

    nz = np.flatnonzero(coef)
    if nz.size > 0.9 * n:                      
        drop = nz[np.argsort(np.abs(coef[nz]))[: int(nz.size - 0.9 * n)]]
        coef[drop] = 0.0; nz = np.flatnonzero(coef)
    s_hat = nz.size

    pp = _gm_scale_factors(X, nz)
    Z = np.zeros((n, p))
    for j in range(p):
        Z[:, j] = rng.normal(0.0, np.sqrt(max(pp[j], 1e-12) / (1.0 - s_hat / n)), n)

    w = np.zeros(p)
    fit = Lasso(alpha=lam_ast, fit_intercept=False, max_iter=100_000, tol=1e-4,
                warm_start=True, precompute=precompute)
    for j in range(p):
        rest = np.concatenate([np.arange(j), np.arange(j + 1, p)])
        Xnew = np.column_stack([X[:, j] + Z[:, j], X[:, j] - Z[:, j], X[:, rest]])
        fit.coef_ = np.r_[coef_cv[j] / 2, coef_cv[j] / 2, coef_cv[rest]]
        b = fit.fit(Xnew, y).coef_
        w[j] = abs(b[0] + b[1]) - abs(b[0] - b[1])
    return gm_reject(w, q)


# Data splitting: port of DS() from the Dai, Lin, Xing and Liu (2023) repo

def analys(M, q):
    cand = np.sort(np.unique(np.abs(M)))
    srt = np.sort(M)
    n_pos = srt.size - np.searchsorted(srt, cand, side="right")
    n_neg = np.searchsorted(srt, -cand, side="left")
    ok = (1 + n_neg) <= q * np.maximum(n_pos, 1)
    if not ok.any():
        return np.zeros(M.size, dtype=bool)
    return M > cand[int(np.argmax(ok))]


def ds_mds_select(X, y, q, num_split, rng, cv_folds=10, precompute=True):
    n, p = X.shape
    Xs = _scale(X)
    incl = np.zeros(p); ds_sel = np.zeros(p, dtype=bool)
    for _ in range(num_split):
        i1 = rng.choice(n, size=n // 2, replace=False)
        i2 = np.setdiff1d(np.arange(n), i1)
        cv = LassoCV(cv=cv_folds, fit_intercept=True, max_iter=100_000,
                     tol=1e-4, precompute=precompute).fit(Xs[i1], y[i1])
        b1 = Lasso(alpha=cv.alpha_, fit_intercept=True, max_iter=100_000,
                   tol=1e-4, precompute=precompute).fit(Xs[i1], y[i1]).coef_
        S = np.flatnonzero(b1)
        ds_sel = np.zeros(p, dtype=bool)
        if S.size == 0 or S.size >= i2.size:
            continue
        b2 = np.zeros(p)
        b2[S] = np.linalg.lstsq(Xs[np.ix_(i2, S)], y[i2], rcond=None)[0]
        M = np.sign(b1 * b2) * (np.abs(b1) + np.abs(b2))
        ds_sel = analys(M, q)
        k = int(ds_sel.sum())
        if k:
            incl[ds_sel] += 1.0 / k

    incl /= num_split
    mds_sel = np.zeros(p, dtype=bool)
    nz = np.flatnonzero(incl > 0)
    if nz.size:
        order = nz[np.argsort(incl[nz])]
        jp = int(np.searchsorted(np.cumsum(incl[order]), q, side="right"))
        mds_sel = incl > incl[order[jp - 1]] if jp > 0 else _mask(order, p)
    return ds_sel, mds_sel