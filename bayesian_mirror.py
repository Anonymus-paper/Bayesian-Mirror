from libraries import (np, cho_factor, cho_solve, gammaln, LassoCV, LCV_KW,
                       dataclass)
from helpers import alphakn_threshold, stop_early, ebh, _mask, _scale


@dataclass
class DesignCache:
    p: int; r: int
    alpha: np.ndarray       
    qy: np.ndarray          
    Hy_norm: float
    Hy2: float


def precompute(X, y):
    n, p = X.shape
    if n <= p:
        raise ValueError(f"precompute requires n > p; got n={n}, p={p}.")
    XtX = X.T @ X; Xty = X.T @ y
    Omega = cho_solve(cho_factor(XtX, lower=True), np.eye(p))
    Od = np.diag(Omega).copy()
    if np.any(Od <= 0):
        raise np.linalg.LinAlgError("X'X is numerically singular.")
    beta_ls = Omega @ Xty
    Hy2 = max(float(y @ y - Xty @ beta_ls), 0.0)
    return DesignCache(p=p, r=n - p + 1, alpha=1.0 / Od,
                       qy=beta_ls / np.sqrt(Od),
                       Hy_norm=float(np.sqrt(Hy2)), Hy2=Hy2)


def _perturbation(cache, T, rng):
    df = cache.r - 2
    g = rng.standard_normal((T, cache.p))
    zeta = rng.standard_normal((T, cache.p))
    U = rng.chisquare(df, size=(T, cache.p)) if df > 0 else np.zeros((T, cache.p))
    S = np.sqrt(g * g + U + zeta * zeta)
    return zeta / S, np.sqrt(cache.alpha) * (cache.Hy_norm * g + zeta * cache.qy) / S


def posterior_params(cache, eta, wy, psi2, a0, b0):
    alpha, qy = cache.alpha, cache.qy
    a_p = alpha * (1.0 + eta); a_m = alpha * (1.0 - eta)
    rho_p = 2.0 * a_p + 1.0 / psi2; rho_m = 2.0 * a_m + 1.0 / psi2
    raqy = np.sqrt(alpha) * qy
    d_p, d_m = raqy + wy, raqy - wy
    rate = b0 + 0.5 * (cache.Hy2 + qy * qy - d_p * d_p / rho_p - d_m * d_m / rho_m)
    return {"rho_plus": rho_p, "rho_minus": rho_m, "d_plus": d_p, "d_minus": d_m,
            "shape": a0 + 0.5 * cache.r,
            "rate": np.maximum(rate, np.finfo(float).tiny)}


def draw_statistics(cache, T, psi2, a0, b0, rng, posterior_draw=False,
                    scale="mean"):
    eta, wy = _perturbation(cache, T, rng)
    pp = posterior_params(cache, eta, wy, psi2, a0, b0)
    rp, rm = pp["rho_plus"], pp["rho_minus"]
    a, rate = pp["shape"], pp["rate"]
    bp, bm = pp["d_plus"] / rp, pp["d_minus"] / rm

    if posterior_draw or scale == "draw":
        sigma_draw = np.sqrt(rate / rng.gamma(a, 1.0, size=rate.shape))
    if posterior_draw:
        bp = bp + sigma_draw * rng.standard_normal(rate.shape) / np.sqrt(rp)
        bm = bm + sigma_draw * rng.standard_normal(rate.shape) / np.sqrt(rm)

    if scale == "mean":
        sigma = np.sqrt(rate) * np.exp(gammaln(a - 0.5) - gammaln(a))
    elif scale == "draw":
        sigma = sigma_draw
    else:
        raise ValueError(f"scale must be 'mean' or 'draw', got {scale!r}")

    raw = np.abs(bp + bm) - np.abs(bp - bm)                       
    return raw / (sigma * np.sqrt(0.5 * (1.0 / rp + 1.0 / rm)))   


def evalues_from_statistics(M, gamma, early=True):
    T, p = M.shape
    E = np.zeros((T, p))
    for t in range(T):
        tau = (stop_early(M[t], gamma, 1) if early
               else alphakn_threshold(M[t], gamma, 1))
        if np.isfinite(tau):
            E[t] = (M[t] >= tau) / (1.0 + np.sum(M[t] <= -tau))
    return p * E.mean(axis=0)


def bm_evalues(X, y, gamma, T, rng, psi2=None, a0=0.01, b0=0.01, early=True,
               posterior_draw=False, scale="mean"):
    n, p = X.shape
    M = draw_statistics(precompute(X, y), T, float(n) if psi2 is None else psi2,
                        a0, b0, rng, posterior_draw=posterior_draw, scale=scale)
    return evalues_from_statistics(M, gamma, early)


def bayesian_mirror_select(X, y, q_star, gamma, T, rng, psi2=None, a0=0.01,
                           b0=0.01, early=False, posterior_draw=False, scale="mean"):
    e = bm_evalues(X, y, gamma, T, rng, psi2, a0, b0, early, posterior_draw, scale)
    return _mask(ebh(e, q_star), X.shape[1])


# Multi-split Bayesian Mirror (Algorithm 1)

def lasso_screen(X, y, rng, split=0.5, cv_folds=5, cap_frac=0.5):
    n = X.shape[0]
    perm = rng.permutation(n)
    n1 = int(round(split * n))
    i1, i2 = perm[:n1], perm[n1:]
    cv = LassoCV(cv=cv_folds, fit_intercept=True, max_iter=50_000, tol=1e-3,
                 **LCV_KW).fit(_scale(X[i1]), y[i1])
    S = np.flatnonzero(cv.coef_)
    cap = max(2, int(cap_frac * i2.size) - 1)
    if S.size > cap:
        S = np.sort(S[np.argsort(-np.abs(cv.coef_[S]))[:cap]])
    return S, i2


def bayesian_mirror_multisplit(X, y, q_star, gamma, T, rng, n_splits=10,
                               split=0.4, cap_frac=0.5, psi2=None, a0=0.01,
                               b0=0.01, early=True, posterior_draw=False,
                               scale="mean"):
    n, p = X.shape
    e = np.zeros(p)
    T_k = max(1, T // n_splits)
    for _ in range(n_splits):
        S, i2 = lasso_screen(X, y, rng, split=split, cap_frac=cap_frac)
        if S.size < 2:
            continue
        Xs, ys = X[np.ix_(i2, S)], y[i2]
        eS = bm_evalues(Xs, ys, gamma, T_k, rng,
                        float(Xs.shape[0]) if psi2 is None else psi2,
                        a0, b0, early, posterior_draw, scale)
        e[S] += eS * (p / S.size)
    return _mask(ebh(e / n_splits, q_star), p)