from libraries import (argparse, os, np, pd, plt, time, Parallel, delayed,
                       parallel_config, StandardScaler)
from helpers import fdp_power, plot_design, fdr_exceedances, Q_STAR
from bayesian_mirror import bayesian_mirror_multisplit
from competing_methods import (knockoff_arms_lowrank, gaussian_mirror_select,
                               ds_mds_select)
from low_dim_sim import METHODS

GBM_FILE = ("GBM.transcriptome__ht_hg_u133a__broad_mit_edu__Level_3__"
            "gene_rma__data.data.txt")



def load_gbm(path):
    df = pd.read_csv(path, sep="\t", index_col=0, skiprows=[1])
    X = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float).T
    keep = np.isfinite(X).all(axis=0) & (np.nanstd(X, axis=0) > 0)
    print(f"GBM design: {X.shape[0]} samples x {keep.sum()} genes "
          f"({(~keep).sum()} genes dropped)")
    return StandardScaler().fit_transform(X[:, keep])


def gen_response(X, s0, rng, sigma0=1.0):
    n, p = X.shape
    support = np.sort(rng.choice(p, s0, replace=False))
    b0 = np.zeros(p); b0[support] = rng.standard_normal(s0)
    return X @ b0 + sigma0 * rng.standard_normal(n), support


def one_rep_semi(rep, X, cfg):
    rng = np.random.default_rng(cfg["seed"] + 10_000 * rep)
    y, supp = gen_response(X, cfg["s0"], rng, cfg["sigma0"])
    p, q, gamma = X.shape[1], cfg["q_star"], cfg["gamma"]

    ko, dk = knockoff_arms_lowrank(X, y, q, gamma, cfg["n_derand"],
                                   int(rng.integers(2 ** 31)))
    ds, mds = ds_mds_select(X, y, q, cfg["n_mds_splits"], rng, precompute=False)
    bm = bayesian_mirror_multisplit(X, y, q, gamma, cfg["T"], rng,
                                    n_splits=cfg["bm_splits"],
                                    split=cfg["bm_split"],
                                    cap_frac=cfg["bm_cap_frac"],
                                    early=True, scale="draw")
    out = {
        "Model-X knockoff": ko,
        "Derand. knockoff": dk,
        "Gaussian Mirror": gaussian_mirror_select(X, y, q, rng, precompute=False),
        "Data splitting": ds,
        "Multiple DS": mds,
        "Bayesian Mirror": bm,
    }
    return {m: fdp_power(s, supp, p) for m, s in out.items()}


def run_semisynthetic(X, s0=30, sigma0=1.0, q_star=0.1, gamma=0.07, T=1000,
                      n_derand=30, n_mds_splits=50, bm_splits=20, bm_split=0.8,
                      bm_cap_frac=0.3, reps=100, seed=2026, n_jobs=1,
                      label="RNA-seq", out_csv=None):
    if not (q_star > gamma / (1.0 - gamma)):
        raise ValueError(f"need q* > gamma/(1-gamma) = "
                         f"{gamma/(1-gamma):.4f}, got {q_star}.")
    n, p = X.shape
    cfg = dict(s0=s0, sigma0=sigma0, q_star=q_star, gamma=gamma, T=T,
               n_derand=n_derand, n_mds_splits=n_mds_splits, seed=seed,
               bm_splits=bm_splits, bm_split=bm_split, bm_cap_frac=bm_cap_frac)
    t0 = time.time()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        res = Parallel(n_jobs=n_jobs)(
            delayed(one_rep_semi)(r, X, cfg) for r in range(reps))

    print(f"\n{label}: n={n}, p={p}, s0={s0}, sigma0={sigma0:g}, q*={q_star}, "
          f"q={gamma}, T={T}, {reps} reps, Bayesian Mirror with {bm_splits} "
          f"partitions ({100*bm_split:.0f}% screen / {100*(1-bm_split):.0f}% "
          f"inference)  ({time.time() - t0:.0f}s)\n")
    print(f"{'Method':<20}{'FDR (SE)':>16}{'Power (SE)':>16}{'FDP > q*':>10}")
    rows = []
    for m in METHODS:
        a = np.array([r[m] for r in res])
        se = (a.std(axis=0, ddof=1) / np.sqrt(reps) if reps > 1
              else np.full(2, np.nan))
        print(f"{m:<20}" + f"{a[:, 0].mean():.3f} ({se[0]:.3f})".rjust(16)
              + f"{a[:, 1].mean():.3f} ({se[1]:.3f})".rjust(16)
              + f"{int((a[:, 0] > q_star).sum())}/{reps}".rjust(10))
        rows += [{"design": label, "rep": r, "method": m,
                  "FDP": a[r, 0], "power": a[r, 1]} for r in range(reps)]

    rep_df = pd.DataFrame(rows)
    if out_csv:
        rep_df.to_csv(out_csv, index=False)
        print(f"\nPer-replicate results written to {out_csv}")
    return rep_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=GBM_FILE, help="path to the GBM data file")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=min(48, os.cpu_count() or 1),
                    help="parallel workers; results do not depend on it")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-dir", default="results_real_data")
    ap.add_argument("--show", action="store_true", help="display the figure")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    X = load_gbm(args.data)
    rep_df = run_semisynthetic(
        X, reps=args.reps, seed=args.seed, n_jobs=args.n_jobs,
        out_csv=os.path.join(args.out_dir, "real_data_replicates.csv"))

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(f"\nReplicates with FDP > q* = {Q_STAR}:")
        print(fdr_exceedances(rep_df, ["RNA-seq"]))
    plot_design(rep_df, "RNA-seq",
                fname=os.path.join(args.out_dir, "boxplot_real_data"))
    if args.show:
        plt.show()