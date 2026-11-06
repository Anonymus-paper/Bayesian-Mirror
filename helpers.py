from libraries import np, plt, Patch


# Designs and data generation
def to_correlation(S):
    d = np.sqrt(np.diag(S))
    return S / np.outer(d, d)


def sigma_ar1(p, rho):
    i = np.arange(p)
    return rho ** np.abs(i[:, None] - i[None, :])


def sigma_constant_partial(p, rho):
    Om = np.full((p, p), float(rho)); np.fill_diagonal(Om, 1.0)
    return to_correlation(np.linalg.inv(Om))


def gen_data(n, p, Sigma_chol, s0, rng, sigma0=1.0):
    X = rng.standard_normal((n, p)) @ Sigma_chol
    support = np.sort(rng.choice(p, s0, replace=False))
    b0 = np.zeros(p); b0[support] = rng.standard_normal(s0)
    return X, X @ b0 + sigma0 * rng.standard_normal(n), support


def fdp_power(sel, support, p):
    R = int(sel.sum())
    null = np.setdiff1d(np.arange(p), support)
    return int(sel[null].sum()) / max(R, 1), int(sel[support].sum()) / support.size


def _scale(X):
    Xc = X - X.mean(axis=0)
    s = Xc.std(axis=0, ddof=1); s[s == 0] = 1.0
    return Xc / s


# Thresholds and e-BH
def alphakn_threshold(W, fdr, offset):
    ts = np.sort(np.concatenate(([0.0], np.abs(W))))
    srt = np.sort(W)
    n_pos = srt.size - np.searchsorted(srt, ts, side="left")
    n_neg = np.searchsorted(srt, -ts, side="right")
    ok = (offset + n_neg) <= fdr * np.maximum(n_pos, 1)
    return float(ts[int(np.argmax(ok))]) if ok.any() else np.inf


def stop_early(W, gamma, offset):
    tau = alphakn_threshold(W, gamma, offset)
    sw = W[np.argsort(-np.abs(W), kind="stable")]
    if np.sum(W > 0) >= 1.0 / gamma:
        pos = np.flatnonzero(sw > 0)
        tau1 = float(sw[pos[int(np.ceil(1.0 / gamma)) - 2]])
    else:
        tau1 = 0.0
    return min(tau, tau1)


def ebh(E, alpha):
    p = E.size
    o = np.argsort(-E, kind="stable")
    idx = np.flatnonzero(E[o] >= (p / alpha / np.arange(1, p + 1)))
    return o[: idx.max() + 1] if idx.size else np.array([], dtype=int)


def _mask(rej, p):
    m = np.zeros(p, dtype=bool); m[rej] = True
    return m



# Boxplots 

Q_STAR = 0.1


def _title(des):
    if des.startswith("AR(1) rho="):
        return rf"AR(1), $\rho = {float(des.split('=')[1]):g}$"
    if des.startswith("constant partial "):
        return rf"Constant partial, $\rho = {float(des.split()[-1]):g}$"
    return des


PLOT_ORDER = ["Bayesian Mirror", "Model-X knockoff", "Derand. knockoff",
              "Gaussian Mirror", "Data splitting", "Multiple DS"]
COLORS = dict(zip(PLOT_ORDER, ["#2a78d6", "#eb6834", "#1baf7a",
                               "#eda100", "#e87ba4", "#008300"]))

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8a85", "axes.linewidth": 0.8,
    "xtick.color": "#55554f", "ytick.color": "#55554f",
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def _boxes(ax, d, metric):
    data = [d.loc[d.method == m, metric].to_numpy() for m in PLOT_ORDER]
    bp = ax.boxplot(data, positions=range(len(PLOT_ORDER)), widths=0.62,
                    patch_artist=True, showfliers=True,
                    medianprops=dict(color="#1f1f1d", lw=1.4),
                    whiskerprops=dict(color="#55554f", lw=0.9),
                    capprops=dict(color="#55554f", lw=0.9),
                    boxprops=dict(lw=0.9, edgecolor="#55554f"))
    for patch, fl, m in zip(bp["boxes"], bp["fliers"], PLOT_ORDER):
        patch.set_facecolor(COLORS[m]); patch.set_alpha(0.85)
        fl.set(marker="o", markersize=4.5, markerfacecolor=COLORS[m],
               markeredgecolor="#55554f", markeredgewidth=0.6, zorder=3)
    ax.set_xticks([]); ax.set_xlim(-0.6, len(PLOT_ORDER) - 0.4)
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.grid(True, color="#e4e3de", lw=0.6); ax.set_axisbelow(True)
    if metric == "FDP":
        ax.axhline(Q_STAR, color="#c0392b", ls="--", lw=1.0, zorder=2)
        # Replicates with FDP > q*, as "k/reps" above the 1.0 line.
        ax.set_ylim(-0.03, 1.14)
        for i, x in enumerate(data):
            ax.text(i, 1.085, f"{int(np.sum(x > Q_STAR))}/{x.size}",
                    ha="center", va="center", fontsize=7, color="#c0392b")


def _legend(fig, y):
    handles = [Patch(facecolor=COLORS[m], edgecolor="#55554f", lw=0.8,
                     alpha=0.85, label=m) for m in PLOT_ORDER]
    handles.append(plt.Line2D([], [], color="#c0392b", ls="--", lw=1.0,
                              label=rf"Nominal $q^* = {Q_STAR}$ "
                                    rf"(above: reps with FDP $> q^*$)"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y),
               ncol=len(handles), frameon=False, fontsize=9,
               handlelength=1.4, columnspacing=1.2)


def plot_design(reps, design, fname=None):
    d = reps[reps.design == design]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, metric, lab in zip(axes, ["FDP", "power"], ["FDR", "Power"]):
        _boxes(ax, d, metric); ax.set_ylabel(lab)
    fig.subplots_adjust(top=0.80, wspace=0.22)
    fig.suptitle(_title(design), y=0.99, fontsize=11)
    _legend(fig, 0.84)
    if fname:
        fig.savefig(fname + ".pdf"); fig.savefig(fname + ".png")
    return fig


def plot_all(reps, designs, fname=None):
    fig, axes = plt.subplots(2, len(designs),
                             figsize=(max(3.4 * len(designs), 12.5), 6.2),
                             sharey="row", squeeze=False)
    for j, des in enumerate(designs):
        d = reps[reps.design == des]
        for i, (metric, lab) in enumerate([("FDP", "FDR"), ("power", "Power")]):
            _boxes(axes[i, j], d, metric)
            if j == 0:
                axes[i, j].set_ylabel(lab)
        axes[0, j].set_title(_title(des), fontsize=10.5)
    fig.subplots_adjust(hspace=0.12, wspace=0.08)
    _legend(fig, 0.93)
    if fname:
        fig.savefig(fname + ".pdf"); fig.savefig(fname + ".png")
    return fig


def fdr_exceedances(reps, designs=None):
    d = reps if designs is None else reps[reps.design.isin(designs)]
    tab = (d.assign(exceed=d.FDP > Q_STAR)
            .pivot_table(index="design", columns="method", values="exceed",
                         aggfunc="sum")
            .reindex(columns=PLOT_ORDER).astype(int))
    if designs is not None:
        tab = tab.reindex(designs)
    tab.insert(0, "reps", d.groupby("design").rep.nunique().reindex(tab.index))
    return tab


def save_file_name(design):
    return "boxplot_" + "".join(c if c.isalnum() else "_" for c in design).strip("_")

