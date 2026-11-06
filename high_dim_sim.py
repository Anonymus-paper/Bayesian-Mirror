from libraries import os
from low_dim_sim import run_study, make_designs, report, parse_args


if __name__ == "__main__":
    args = parse_args(__doc__, "results_high_dim")
    os.makedirs(args.out_dir, exist_ok=True)
   
    if args.quick:
        P, GRID = 600, (0, 0.6)
        settings = dict(n=400, p=P, s0=40, T=100, n_derand=5, n_mds_splits=5,
                        bm_splits=5, reps=min(args.reps, 2))
    else:
        P, GRID = 2000, (0, 0.2, 0.4, 0.6, 0.8)
        settings = dict(n=800, p=P, s0=70, T=1000, n_derand=50,
                        n_mds_splits=50, bm_splits=20, reps=args.reps)

    designs, ar, cp = make_designs(P, GRID)
    rep_df = run_study(
        designs, sigma0=1.0, q_star=0.1, gamma=0.03, bm_split=0.8,
        seed=args.seed, n_jobs=args.n_jobs,
        out_csv=os.path.join(args.out_dir, "high_dim_replicates.csv"), **settings)
    report(rep_df, designs, ar, cp, args.out_dir, show=args.show)