from libraries import (argparse, os, np, pd, time, cholesky, Parallel, delayed,
                       parallel_config)
from helpers import sigma_ar1, gen_data, fdp_power, ebh, _mask
from bayesian_mirror import (precompute, draw_statistics, lasso_screen,
                             evalues_from_statistics)

SCENARIOS = {"low": dict(n=1000, p=500), "high": dict(n=800, p=2000)}
SPLITS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
GAMMAS = (0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08)


def ablation_rep(rep, Sigma_chol, cfg):
    n, p, K, seed = cfg["n"], cfg["p"], cfg["K"], cfg["seed"]
    rng = np.random.default_rng(seed + 1884 * rep)
    X, y, sup = gen_data(n, p, Sigma_chol, cfg["s0"], rng, sigma0=cfg["sigma0"])
    T_k = max(1, cfg["T"] // K)

    rows = []
    for si, split in enumerate(cfg["splits"]):
        rs = np.random.default_rng(seed + 7 + 31 * si + 1884 * rep)    
        rd = np.random.default_rng(seed + 13 + 31 * si + 1884 * rep)   
        e = {g: np.zeros(p) for g in cfg["gammas"]}
        for _ in range(K):
            S, i2 = lasso_screen(X, y, rs, split=split)
            if S.size < 2:
                continue
            Xs, ys = X[np.ix_(i2, S)], y[i2]
            M = draw_statistics(precompute(Xs, ys), T_k, float(i2.size),
                                0.01, 0.01, rd, scale="draw")
            for g in cfg["gammas"]:
                e[g][S] += evalues_from_statistics(M, g) * (p / S.size)
        for g in cfg["gammas"]:
            fdp, power = fdp_power(_mask(ebh(e[g] / K, cfg["q_star"]), p), sup, p)
            rows.append({"rep": rep, "split": split, "gamma": g,
                         "FDP": fdp, "power": power})
    return rows


def run_ablation(label, n, p, s0=70, rho=0.4, splits=SPLITS, gammas=GAMMAS, K=20,
                 T=1000, sigma0=1.0, q_star=0.1, reps=100, seed=1924, n_jobs=1,
                 out_csv=None):
    bad = [g for g in gammas if not q_star > g / (1 - g)]
    if bad:
        raise ValueError(f"q* > q/(1-q) fails for q in {bad}")
    cfg = dict(n=n, p=p, s0=s0, splits=tuple(splits), gammas=tuple(gammas), K=K,
               T=T, sigma0=sigma0, q_star=q_star, seed=seed)
    Lc = cholesky(sigma_ar1(p, rho), lower=False)
    t0 = time.time()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        res = Parallel(n_jobs=n_jobs)(
            delayed(ablation_rep)(r, Lc, cfg) for r in range(reps))
    df = pd.DataFrame([row for rows in res for row in rows])
    df.insert(0, "scenario", label)

    print(f"\n{label}: n={n}, p={p}, AR(1) rho={rho}, s0={s0}, L={K}, T={T}, "
          f"q*={q_star}, {reps} reps  ({time.time() - t0:.0f}s)")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        for metric in ("power", "FDP"):
            tab = df.pivot_table(index="split", columns="gamma", values=metric)
            tab.index.name, tab.columns.name = "screening fraction", "q"
            print(f"\nEmpirical {'power' if metric == 'power' else 'FDR'}:")
            print(tab)
    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"\nPer-replicate results written to {out_csv}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=("low", "high", "both"), default="both")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=min(48, os.cpu_count() or 1),
                    help="parallel workers; results do not depend on it")
    ap.add_argument("--seed", type=int, default=1924)
    ap.add_argument("--out-dir", default="results_ablation")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    names = ("low", "high") if args.scenario == "both" else (args.scenario,)
    for name in names:
        run_ablation(f"{name.capitalize()}-dimensional", **SCENARIOS[name],
                     reps=args.reps, seed=args.seed, n_jobs=args.n_jobs,
                     out_csv=os.path.join(args.out_dir, f"ablation_{name}.csv"))