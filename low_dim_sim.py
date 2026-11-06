from libraries import (argparse, os, np, pd, plt, time, cholesky, Parallel,
                       delayed, parallel_config)
from helpers import (sigma_ar1, sigma_constant_partial, gen_data, fdp_power,
                     plot_all, plot_design, fdr_exceedances, save_file_name, Q_STAR)
from bayesian_mirror import bayesian_mirror_multisplit
from competing_methods import (knockoff_arms_R, gaussian_mirror_select,
                               ds_mds_select)



METHODS = ["Model-X knockoff", "Derand. knockoff", "Gaussian Mirror",
           "Data splitting", "Multiple DS", "Bayesian Mirror"]


def one_rep(rep, Sigma_chol, cfg):
    rng = np.random.default_rng(cfg["seed"] + 1884 * rep)
    X, y, supp = gen_data(cfg["n"], cfg["p"], Sigma_chol, cfg["s0"], rng,
                          sigma0=cfg["sigma0"])
    p, q, gamma = cfg["p"], cfg["q_star"], cfg["gamma"]

    ko, dk = knockoff_arms_R(X, y, q, gamma, cfg["n_derand"],
                             int(rng.integers(2 ** 31)))
    ds, mds = ds_mds_select(X, y, q, cfg["n_mds_splits"], rng)
    bm = bayesian_mirror_multisplit(X, y, q, gamma, cfg["T"], rng,
                                    n_splits=cfg["bm_splits"],
                                    split=cfg["bm_split"], early=True)
    out = {
        "Model-X knockoff": ko,
        "Derand. knockoff": dk,
        "Gaussian Mirror": gaussian_mirror_select(X, y, q, rng),
        "Data splitting": ds,
        "Multiple DS": mds,
        "Bayesian Mirror": bm,
    }
    return {m: fdp_power(s, supp, p) for m, s in out.items()}


def run_study(designs, n, p, s0, T, n_derand, n_mds_splits, bm_splits, bm_split,
              reps, sigma0=1.0, q_star=0.1, gamma=0.03, seed=1924, n_jobs=1,
              out_csv=None):
    if not (q_star > gamma / (1.0 - gamma)):
        raise ValueError(f"need q* > gamma/(1-gamma) = "
                         f"{gamma/(1-gamma):.4f}, got {q_star}.")
    cfg = dict(n=n, p=p, s0=s0, sigma0=sigma0, q_star=q_star, gamma=gamma, T=T,
               n_derand=n_derand, n_mds_splits=n_mds_splits, seed=seed,
               bm_splits=bm_splits, bm_split=bm_split)

    table, rows = {}, []
    for label, Sigma in designs:
        Lc = cholesky(Sigma, lower=False)
        t0 = time.time()
        with parallel_config(backend="loky", inner_max_num_threads=1):
            res = Parallel(n_jobs=n_jobs)(
                delayed(one_rep)(r, Lc, cfg) for r in range(reps))
        for m in METHODS:
            a = np.array([r[m] for r in res])
            se = (a.std(axis=0, ddof=1) / np.sqrt(reps) if reps > 1
                  else np.full(2, np.nan))
            table[(label, m)] = (a[:, 0].mean(), se[0], a[:, 1].mean(), se[1])
            rows += [{"design": label, "rep": r, "method": m,
                      "FDP": a[r, 0], "power": a[r, 1]} for r in range(reps)]
        print(f"{label:<28} ({time.time() - t0:.0f}s)", flush=True)

    labels = [d[0] for d in designs]
    print(f"\nn={n}, p={p}, s0={s0}, sigma0={sigma0:g}, q*={q_star}, "
          f"q={gamma}, T={T}, {reps} reps, Bayesian Mirror with {bm_splits} "
          f"partitions ({100*bm_split:.0f}% screen / "
          f"{100*(1-bm_split):.0f}% inference)\n")
    print(f"{'Method':<20}" + "".join(f"{l:>32}" for l in labels))
    print(f"{'':<20}" + "".join(f"{'FDR (SE)':>16}{'Power (SE)':>16}" for _ in labels))
    for m in METHODS:
        row = f"{m:<20}"
        for l in labels:
            f_, fse, pw, pse = table[(l, m)]
            row += f"{f_:.3f} ({fse:.3f})".rjust(16) + f"{pw:.3f} ({pse:.3f})".rjust(16)
        print(row)

    rep_df = pd.DataFrame(rows)
    if out_csv:
        rep_df.to_csv(out_csv, index=False)
        print(f"\nPer-replicate results written to {out_csv}")
    return rep_df


def make_designs(p, grid):
    ar = [f"AR(1) rho={r}" for r in grid]
    cp = [f"constant partial {r}" for r in grid]
    designs = ([(lab, sigma_ar1(p, r)) for lab, r in zip(ar, grid)]
               + [(lab, sigma_constant_partial(p, r)) for lab, r in zip(cp, grid)])
    return designs, ar, cp


def report(rep_df, designs, ar, cp, out_dir, show=False):
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(f"\nReplicates with FDP > q* = {Q_STAR}:")
        print(fdr_exceedances(rep_df, [d[0] for d in designs]))
    plot_all(rep_df, ar, fname=os.path.join(out_dir, "boxplots_ar1"))
    plot_all(rep_df, cp, fname=os.path.join(out_dir, "boxplots_constant_partial"))
    for des, _ in designs:
        plot_design(rep_df, des, fname=os.path.join(out_dir, save_file_name(des)))
    if show:
        plt.show()
    plt.close("all")


def parse_args(description, default_out):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=min(48, os.cpu_count() or 1),
                    help="parallel workers; results do not depend on it")
    ap.add_argument("--seed", type=int, default=1924)
    ap.add_argument("--out-dir", default=default_out)
    ap.add_argument("--quick", action="store_true",
                    help="small configuration to check that everything runs")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--show", action="store_true", help="display the figures")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args(__doc__, "results_low_dim")
    os.makedirs(args.out_dir, exist_ok=True)
    
    if args.quick:
        P, GRID = 200, (0, 0.6)
        settings = dict(n=500, p=P, s0=40, T=100, n_derand=5, n_mds_splits=5,
                        bm_splits=5, reps=min(args.reps, 2))
    else:
        P, GRID = 500, (0, 0.2, 0.4, 0.6, 0.8)
        settings = dict(n=1000, p=P, s0=70, T=1000, n_derand=50,
                        n_mds_splits=50, bm_splits=20, reps=args.reps)

    designs, ar, cp = make_designs(P, GRID)
    rep_df = run_study(
        designs, sigma0=1.0, q_star=0.1, gamma=0.03, bm_split=0.4,
        seed=args.seed, n_jobs=args.n_jobs,
        out_csv=os.path.join(args.out_dir, "low_dim_replicates.csv"), **settings)
    report(rep_df, designs, ar, cp, args.out_dir, show=args.show)