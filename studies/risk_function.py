#!/usr/bin/env python3
"""
Risk functions for the slow layer (learn/meta/CONTROL_FROM_SCRATCH_2026-10-12.md
section 2.2-2.3): regressions

    rho_i(speed, sea state, relative heading, hull/sea draw)
      -> the hourly rate of filtered CG-acceleration peaks >= 3 / 5 / 7 g,
         the mean of the highest tenth of those peaks (A1/10, g),
         the hourly rate of bow immersions,

fitted to the frozen plant's runs (studies/_cache/envelope, plus any runs
of studies/risk_data when present), with a one-sided uncertainty, and the
slow layer's read-off: the highest speed whose upper quantile of each
risk stays under the budget b_i / k for a correction factor k.

MODEL. Each target is regressed in log space. The mean is linear in the
standardised inputs (least squares); a Gaussian process with an automatic-
relevance squared-exponential kernel fits the residual, with a known
per-point noise variance (the tail fit's bootstrap spread for extrapolated
rates, Poisson 1/n for counted rates, a 5 % seed spread for A1/10) plus a
fitted jitter. Hyperparameters by maximising the marginal likelihood
(scipy L-BFGS from a few starts). The quantile used by the read-off is
the Gaussian predictive quantile in log space, so "90 %" means the rate is
below the stated value with 90 % probability under the model -- it covers
the randomisation spread and the statistical error of 600 s runs, not a
bias of the plant (that is what k is for).

INPUTS (per run): the mean speed over the statistics window (kn), the
effective Hs (m) and Tp (s) after the draw, cos and sin of the relative
wave heading (180 deg = head seas), the short-wave amplitude factor (1
without a draw) and JONSWAP gamma. Hull draws are deliberately not inputs
here (pass 1 found their effect invisible beside the sea inputs); their
spread is in the residual.

    python -m studies.risk_function fit        (fit, validate, tables, figure)
    python -m studies.risk_function readoff    (tables only, from the saved fit)

Outputs: studies/_cache/risk/{fit.json, tables.md, validation.json,
risk_7g.png}. Read-only use of the run caches.
"""
import glob
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import os as _os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import numpy as np                                      # noqa: E402
from scipy.optimize import minimize                     # noqa: E402

from studies import envelope_stats as S                 # noqa: E402

OUT = os.path.join(ROOT, "studies", "_cache", "risk")
SOURCES = (os.path.join(ROOT, "studies", "_cache", "envelope", "runs"),
           os.path.join(ROOT, "studies", "_cache", "riskdata", "runs"))
KN = 0.514444
TARGETS = ("rate3", "rate5", "rate7", "a110", "immersion")
TARGET_CN = {"rate3": "≥3 g 次/时", "rate5": "≥5 g 次/时",
             "rate7": "≥7 g 次/时", "a110": "重心 A1/10 (g)",
             "immersion": "船头进水 次/时"}
INPUT_NAMES = ("speed_kn", "hs", "tp", "cos_rel", "sin_rel", "short_wave",
               "gamma")
BUDGET = dict(a110=3.0, rate7=1.0, immersion=3.0)
FOLD = (12.5, 14.0)          # the thrust fold: not a holdable speed
RATE_FLOOR = 0.02            # per hour; below this a 600 s run says nothing


# ------------------------------------------------------------------ data
def load_runs():
    rows = []
    for d in SOURCES:
        for pj in sorted(glob.glob(os.path.join(d, "*.json"))):
            with open(pj) as f:
                r = json.load(f)
            if not r.get("stats"):
                continue
            pn = pj[:-5] + ".npz"
            if not os.path.exists(pn):
                continue
            z = np.load(pn)
            r["peaks_cg"] = np.asarray(z["peaks_cg"], float)
            r["source"] = os.path.basename(os.path.dirname(d))
            rows.append(r)
    return rows


def rel_heading_deg(r):
    sp = r["sea_params"]
    th0 = float(sp.get("theta0", math.pi))
    hd = float(r["arg"]["heading_rad"])
    return math.degrees((th0 - hd) % (2 * math.pi))


def features(r):
    sp = r["sea_params"]
    rel = math.radians(rel_heading_deg(r))
    draw = r.get("draw") or {}
    return np.array([r["stats"]["u_mean_kn"], float(sp["hs"]),
                     float(sp["tp"]), math.cos(rel), math.sin(rel),
                     float(draw.get("short_wave", 1.0)),
                     float(sp.get("gamma", 3.3))])


def targets(r, n_boot=60):
    """{target: (log value, log-variance)} for one run."""
    st = r["stats"]
    T = float(r["t_stat"])
    pk = r["peaks_cg"]
    out = {}
    tf = S.tail_fit(pk, T, n_boot=n_boot, k_factors=(1.0,))
    for lev, key in ((3.0, "rate3"), (5.0, "rate5"), (7.0, "rate7")):
        lv = tf["levels"].get(f"k1_{lev:g}g") if tf.get("ok") else None
        if lv is None or not np.isfinite(lv["rate"]) or lv["rate"] <= 0:
            # no fit: fall back to the direct count with a half-event floor
            n = int((pk >= lev).sum())
            rate = (n + 0.5) * 3600.0 / T
            out[key] = (math.log(rate), 1.0 / (n + 0.5))
            continue
        if lv["extrapolated"] and np.isfinite(lv["p5"]) and lv["p5"] > 0 \
                and np.isfinite(lv["p95"]) and lv["p95"] > lv["p5"]:
            sd = (math.log(lv["p95"]) - math.log(lv["p5"])) / (2 * 1.645)
            var = max(sd * sd, 0.05 ** 2)
        else:
            n = max(lv["n_direct"], 1)
            var = 1.0 / n
        out[key] = (math.log(lv["rate"]), var)
    # a 600 s run cannot resolve rates below a few hundredths per hour:
    # the tail fit's extrapolation there is noise (values like 1e-75 per
    # hour occurred), so the log-rate is floored at RATE_FLOOR with a wide
    # uncertainty, and every variance is capped
    for key in ("rate3", "rate5", "rate7"):
        y, v = out[key]
        if y < math.log(RATE_FLOOR):
            y, v = math.log(RATE_FLOOR), 4.0
        out[key] = (y, min(v, 4.0))
    a = st["cg"]["a110"]
    out["a110"] = (math.log(max(a, 1e-3)), 0.05 ** 2)
    # immersions: a plain count; zero in 600 s is written as a tenth of an
    # event (0.6 per hour, well under the budget of 3) with a wide variance
    n = int(st["n_immersion"])
    out["immersion"] = (math.log((n + 0.1) * 3600.0 / T),
                        min(1.0 / (n + 0.1), 4.0))
    return out


def dataset():
    runs = load_runs()
    X = np.array([features(r) for r in runs])
    Y = {k: np.zeros(len(runs)) for k in TARGETS}
    V = {k: np.zeros(len(runs)) for k in TARGETS}
    for i, r in enumerate(runs):
        tg = targets(r)
        for k in TARGETS:
            Y[k][i], V[k][i] = tg[k]
    groups = np.array([r["name"].rsplit("_s", 1)[0] for r in runs])
    names = [r["name"] for r in runs]
    return X, Y, V, groups, names, runs


# ------------------------------------------------------------------- GP
class LinGP:
    """Linear mean + ARD squared-exponential GP on the residual, with a
    known per-point noise variance plus a fitted jitter."""

    def __init__(self, X, y, v, restarts=4, seed=0):
        X = np.asarray(X, float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        A = np.column_stack([np.ones(len(Z)), Z])
        self.beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ self.beta
        self.Z, self.r, self.v = Z, res, np.asarray(v, float)
        d = Z.shape[1]
        rng = np.random.default_rng(seed)
        best = None
        for k in range(restarts):
            x0 = np.concatenate([[math.log(max(res.std(), 1e-3))],
                                 np.log(np.full(d, 1.5))
                                 + (rng.normal(0, 0.5, d) if k else 0.0),
                                 [math.log(0.1 * max(res.std(), 1e-3))]])
            try:
                sol = minimize(self._nlml, x0, method="L-BFGS-B",
                               bounds=[(-6, 1.5)] + [(-2.5, 4)] * d + [(-8, 0.5)])
            except Exception:           # noqa: BLE001
                continue
            if best is None or sol.fun < best.fun:
                best = sol
        self.theta = best.x
        self._factor(self.theta)

    def _kernel(self, Za, Zb, theta):
        sf2 = math.exp(2 * theta[0])
        ell = np.exp(theta[1:-1])
        D = ((Za[:, None, :] - Zb[None, :, :]) / ell) ** 2
        return sf2 * np.exp(-0.5 * D.sum(-1))

    def _factor(self, theta):
        K = self._kernel(self.Z, self.Z, theta)
        K[np.diag_indices_from(K)] += self.v + math.exp(2 * theta[-1])
        L = np.linalg.cholesky(K + 1e-9 * np.eye(len(K)))
        self.L = L
        self.alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.r))
        return L

    def _nlml(self, theta):
        try:
            L = self._factor(theta)
        except np.linalg.LinAlgError:
            return 1e10
        return float(0.5 * self.r @ self.alpha + np.log(np.diag(L)).sum()
                     + 0.5 * len(self.r) * math.log(2 * math.pi))

    def predict(self, Xs):
        Xs = np.atleast_2d(np.asarray(Xs, float))
        Zs = (Xs - self.mu) / self.sd
        A = np.column_stack([np.ones(len(Zs)), Zs])
        Ks = self._kernel(Zs, self.Z, self.theta)
        m = A @ self.beta + Ks @ self.alpha
        W = np.linalg.solve(self.L, Ks.T)
        var = math.exp(2 * self.theta[0]) - (W * W).sum(0) \
            + math.exp(2 * self.theta[-1])
        return m, np.maximum(var, 1e-12)

    def to_dict(self):
        return dict(mu=self.mu.tolist(), sd=self.sd.tolist(),
                    beta=self.beta.tolist(), theta=self.theta.tolist(),
                    lengthscales=np.exp(self.theta[1:-1]).tolist(),
                    sigma_f=math.exp(self.theta[0]),
                    sigma_jitter=math.exp(self.theta[-1]))


# --------------------------------------------------------------- fitting
def fit_all(X, Y, V):
    return {k: LinGP(X, Y[k], V[k]) for k in TARGETS}


def validate(X, Y, V, groups):
    """Leave-one-condition-out (both seeds of a condition held out):
    per target the RMS log error, the median factor error and the
    fraction of held-out points inside the 90 % band."""
    out = {}
    ug = np.unique(groups)
    for k in TARGETS:
        err, inside = [], []
        for g in ug:
            te = groups == g
            gp = LinGP(X[~te], Y[k][~te], V[k][~te], restarts=2)
            m, var = gp.predict(X[te])
            e = Y[k][te] - m
            err.extend(e.tolist())
            sd = np.sqrt(var + V[k][te])
            inside.extend((np.abs(e) <= 1.645 * sd).tolist())
        err = np.asarray(err)
        out[k] = dict(n=int(len(err)), rms_log=float(np.sqrt(np.mean(err**2))),
                      median_factor=float(np.exp(np.median(np.abs(err)))),
                      p90_factor=float(np.exp(np.percentile(np.abs(err), 90))),
                      inside90=float(np.mean(inside)))
    return out


# --------------------------------------------------------------- read-off
def speed_grid():
    v = np.arange(6.0, 32.01, 0.5)
    return v[(v < FOLD[0]) | (v > FOLD[1])]


def feature_row(speed_kn, hs, tp, rel_deg, short_wave=1.0, gamma=3.3):
    rel = math.radians(rel_deg)
    return np.array([speed_kn, hs, tp, math.cos(rel), math.sin(rel),
                     short_wave, gamma])


def readoff(models, hs, tp, rel_deg, k=1.0, q=0.9, short_wave=1.0,
            gamma=3.3):
    """The highest speed on the grid whose q-quantile of every budgeted
    risk (after k) is within budget, per criterion and all together, with
    the predictive standard deviation at that speed (log) as an
    extrapolation warning."""
    from scipy.stats import norm
    zq = norm.ppf(q)
    v = speed_grid()
    Xs = np.array([feature_row(s, hs, tp, rel_deg, short_wave, gamma)
                   for s in v])
    ok = {}
    sdmax = 0.0
    for key, b in BUDGET.items():
        m, var = models[key].predict(Xs)
        up = np.exp(m + zq * np.sqrt(var))
        if key == "a110":
            ok[key] = k * up <= b
        else:
            # rate_k(x) = rate(x / k): a rate of the simulated level b / k
            # is not available from a fixed-level regression, so k acts
            # on the rate through the A1/10 scaling proxy: the 7 g rate at
            # level b/k is approximated by the fitted rate times
            # (k ** gamma7) with gamma7 the local log-slope of the rate in
            # level (3 -> 5 -> 7 g) -- see tail_slope()
            ok[key] = up * tail_slope(models, Xs, k) <= b if key == "rate7" \
                else up <= b
        sdmax = max(sdmax, float(np.sqrt(var).max()))

    def top(mask):
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return None
        # the highest speed such that every lower grid speed also passes
        # (monotone read-off); otherwise the first failure's predecessor
        first_fail = np.flatnonzero(~mask)
        if len(first_fail) == 0:
            return float(v[-1])
        if first_fail[0] == 0:
            return None
        return float(v[first_fail[0] - 1])

    res = {key: top(m) for key, m in ok.items()}
    res["all"] = top(np.logical_and.reduce(list(ok.values())))
    res["sd_log_max"] = sdmax
    return res


def tail_slope(models, Xs, k):
    """Factor by which the >= 7 g rate rises when the simulated
    accelerations are multiplied by k: the rate at level 7/k g, obtained
    from the fitted rates at 5 and 7 g by log-linear interpolation in the
    level (rate ~ exp(-c * level)); 1 when k = 1."""
    if abs(k - 1.0) < 1e-12:
        return np.ones(len(Xs))
    m5, _ = models["rate5"].predict(Xs)
    m7, _ = models["rate7"].predict(Xs)
    c = (m5 - m7) / 2.0                       # log-rate per g, >= 0 expected
    c = np.maximum(c, 0.0)
    return np.exp(c * (7.0 - 7.0 / k))


# ----------------------------------------------------------------- report
def tables(models, val, X, names):
    md = ["# 慢层风险函数：拟合与读数", "",
          f"生成时间 {time.strftime('%Y-%m-%d %H:%M')}；数据 {len(X)} 次运行"
          "（studies/_cache/envelope 与 riskdata）。模型见 "
          "studies/risk_function.py 的说明。", ""]
    md += ["## 1. 留一工况交叉验证（两个种子同时留出）", "",
           "| 目标 | 点数 | 对数均方根误差 | 中位倍数误差 | 90% 倍数误差 | "
           "落在 90% 带内的比例 |", "|---|---|---|---|---|---|"]
    for k in TARGETS:
        v = val[k]
        md.append(f"| {TARGET_CN[k]} | {v['n']} | {v['rms_log']:.2f} | "
                  f"×{v['median_factor']:.2f} | ×{v['p90_factor']:.1f} | "
                  f"{v['inside90']:.2f} |")
    md += ["", "## 2. 各输入的相关长度（标准化单位；越大越不敏感）", "",
           "| 目标 | " + " | ".join(INPUT_NAMES) + " | 信号幅度 | 抖动 |",
           "|---|" + "---|" * (len(INPUT_NAMES) + 2)]
    for k in TARGETS:
        d = models[k].to_dict()
        md.append(f"| {TARGET_CN[k]} | "
                  + " | ".join(f"{x:.1f}" for x in d["lengthscales"])
                  + f" | {d['sigma_f']:.2f} | {d['sigma_jitter']:.2f} |")
    md += ["", "## 3. 慢层读数：预算内的最高航速（节）", "",
           "预算：重心 A1/10 ≤ 3 g；≥7 g 每小时 ≤ 1 次；船头进水每小时 ≤ 3 次。"
           "q 为所用的预测分位数（0.5 = 中位数，0.9 = 一侧可靠），k 为修正系数"
           "（船模加速度乘 k 后再比较）。“—”为网格最低航速都不满足。"
           "最后一列是读数处预测的对数标准差，大于 1 表示已在数据之外外推。", ""]
    md += ["| 海况 Hs/Tp | 相对浪向 | q | k | A1/10 | ≥7 g | 进水 | 三条同时 | "
           "对数标准差 |", "|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for hs, tp in ((0.75, 4.5), (1.0, 5.0), (1.0, 6.0), (1.25, 5.5),
                   (1.5, 6.0)):
        for rel in (180.0, 135.0):
            for q in (0.5, 0.9):
                for k in (1.0, 1.2, 1.5):
                    r = readoff(models, hs, tp, rel, k=k, q=q)
                    rows.append(dict(hs=hs, tp=tp, rel=rel, q=q, k=k, **r))
                    fmt = (lambda x: "—" if x is None else f"{x:g}")
                    md.append(f"| {hs:g} m / {tp:g} s | {rel:g}° | {q:g} | "
                              f"{k:g} | {fmt(r['a110'])} | {fmt(r['rate7'])} "
                              f"| {fmt(r['immersion'])} | {fmt(r['all'])} | "
                              f"{r['sd_log_max']:.2f} |")
    md.append("")
    return "\n".join(md), rows


def figure(models, X, Y, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v = np.arange(8.0, 30.01, 0.25)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (hs, tp, title) in zip(axes, ((1.0, 5.0, "Hs 1.0 m, Tp 5 s"),
                                          (1.5, 6.0, "Hs 1.5 m, Tp 6 s"))):
        for rel, c in ((180.0, "C0"), (135.0, "C1")):
            Xs = np.array([feature_row(s, hs, tp, rel) for s in v])
            m, var = models["rate7"].predict(Xs)
            sd = np.sqrt(var)
            ax.plot(v, np.exp(m), c, label=f"rel {rel:g} deg, median")
            ax.fill_between(v, np.exp(m - 1.645 * sd), np.exp(m + 1.645 * sd),
                            color=c, alpha=0.15, label=f"rel {rel:g} deg, 5-95%")
        sel = (np.abs(X[:, 1] - hs) < 0.2) & (np.abs(X[:, 2] - tp) < 0.8)
        rel_d = np.degrees(np.arctan2(X[sel, 4], X[sel, 3])) % 360
        ax.scatter(X[sel, 0], np.exp(Y["rate7"][sel]), s=14,
                   c=np.where(np.abs(rel_d - 180) < 10, "C0", "C1"),
                   edgecolor="k", linewidth=0.3, label="runs (600 s each)")
        ax.axhline(1.0, color="k", ls="--", lw=0.8)
        ax.axvspan(FOLD[0], FOLD[1], color="0.85")
        ax.set_yscale("log")
        ax.set_xlabel("speed (kn)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("peaks >= 7 g per hour (CG, filtered)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def cmd_fit():
    os.makedirs(OUT, exist_ok=True)
    X, Y, V, groups, names, runs = dataset()
    print(f"{len(X)} runs, {len(np.unique(groups))} conditions", flush=True)
    t0 = time.time()
    models = fit_all(X, Y, V)
    print(f"fit in {time.time() - t0:.0f} s", flush=True)
    t0 = time.time()
    val = validate(X, Y, V, groups)
    print(f"validation in {time.time() - t0:.0f} s", flush=True)
    md, rows = tables(models, val, X, names)
    with open(os.path.join(OUT, "tables.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(OUT, "fit.json"), "w") as f:
        json.dump(dict(inputs=INPUT_NAMES, n_runs=len(X),
                       models={k: models[k].to_dict() for k in TARGETS},
                       validation=val, readoff=rows,
                       data=dict(X=X.tolist(), names=names,
                                 Y={k: Y[k].tolist() for k in TARGETS},
                                 V={k: V[k].tolist() for k in TARGETS})),
                  f, indent=1)
    figure(models, X, Y, os.path.join(OUT, "risk_7g.png"))
    print(md)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if cmd == "fit":
        cmd_fit()
    else:
        raise SystemExit(__doc__)
