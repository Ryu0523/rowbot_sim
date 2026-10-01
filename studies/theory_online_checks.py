#!/usr/bin/env python3
"""
Numerical checks of the formulas in learn/meta/THEORY_ONLINE_FLOW_2026-10-01.md
(sections T1-T7). Each check compares a closed-form expression with a Monte
Carlo simulation of the same idealised model, then prints the formula at
the code's own constants. This checks the algebra, not the method: nothing
here touches the training world, the target or the networks.

    python -m studies.theory_online_checks
"""
import math

import numpy as np
from scipy import stats

RNG = np.random.default_rng(0)
Z90 = stats.norm.ppf(0.95)


def line(s):
    print(s, flush=True)


# ------------------------------------------------------------------ T2
def mixture_regret():
    """Discrete prior over M Gaussian-mean laws: the cumulative log-loss of
    the Bayes mixture equals -log sum_f Pi(f) prod p_f (identity) and its
    regret against the true law is <= log 1/Pi(truth) (bound, any data)."""
    M, n, sig = 50, 300, 1.0
    mus = np.linspace(-1, 1, M)
    pri = RNG.dirichlet(np.ones(M))
    j = int(np.argmin(pri))                       # truth = least likely law
    e = mus[j] + sig * RNG.standard_normal(n)
    logw = np.log(pri)
    seq = 0.0
    for t in range(n):
        lp = stats.norm.logpdf(e[t], mus, sig)
        w = np.exp(logw - logw.max())
        w /= w.sum()
        seq += -np.log(np.sum(w * np.exp(lp)))
        logw = logw + lp
    ll = stats.norm.logpdf(e[:, None], mus[None], sig).sum(0)
    ident = -np.log(np.sum(pri * np.exp(ll - ll.max()))) - ll.max()
    regret = seq + ll[j]
    line(f"T2 mixture identity: sequential {seq:.6f} vs closed form "
         f"{ident:.6f}; regret vs truth {regret:.3f} <= log 1/Pi(truth) "
         f"{-math.log(pri[j]):.3f}: {regret <= -math.log(pri[j]) + 1e-9}")


# ------------------------------------------------------------------ T3
def conjugate_coverage(z, kappa, n):
    """Gaussian law e = mu + xi, xi ~ N(0, 1); prior mu ~ N(0, 1 / kappa)
    (worth kappa observations); truth mu_g = z / sqrt(kappa) (z prior sd
    away). Bias, and exact coverage of the central 90% posterior-predictive
    interval after n observations."""
    s0 = 1.0 / math.sqrt(kappa)
    mu_g = z * s0
    v = 1.0 / (kappa + n)                  # posterior variance
    b = -mu_g * kappa / (kappa + n)        # E m_n - mu_g
    w = n * v * v                          # Var m_n
    c = Z90 * math.sqrt(1.0 + v)
    S = math.sqrt(1.0 + w)
    cov = stats.norm.cdf((c + b) / S) - stats.norm.cdf((-c + b) / S)
    return b, cov


def horizon_coverage(z, kappa, n, H):
    """Same model; coverage of the central 90% predictive interval of the
    SUM of the next H errors (the rollout's integrated error), the law
    held for the H steps: error of the sum ~ N(-H b, H + H^2 w), predictive
    variance H + H^2 v."""
    b, _ = conjugate_coverage(z, kappa, n)
    v = 1.0 / (kappa + n)
    w = n * v * v
    c = Z90 * math.sqrt(H + H * H * v)
    S = math.sqrt(H + H * H * w)
    return stats.norm.cdf((c + H * b) / S) - stats.norm.cdf((-c + H * b) / S)


def check_horizon():
    z, kappa, n, H, R = 3.0, 256.0, 256, 24, 100000
    mu_g = z / math.sqrt(kappa)
    m = (mu_g * n + math.sqrt(n) * RNG.standard_normal(R)) / (kappa + n)
    v = 1 / (kappa + n)
    S = H * mu_g + math.sqrt(H) * RNG.standard_normal(R)
    mc = np.mean(np.abs(S - H * m) <= Z90 * math.sqrt(H + H * H * v))
    line(f"T3 horizon H={H} z={z} kappa={kappa}: formula "
         f"{horizon_coverage(z, kappa, n, H):.4f}, simulation {mc:.4f}")
    ks = (16, 64, 256, 1024, 4096)
    line("T3 table: 90% coverage of the 24-step summed error at n = 256")
    line("      " + "".join(f"k={k:<7d}" for k in ks))
    for zz in (1.645, 2.0, 3.0, 4.0):
        line(f"z={zz:<4} " + "".join(
            f"{horizon_coverage(zz, k, 256, 24):<9.3f}" for k in ks))


def check_conjugate():
    z, kappa, n, R = 3.0, 256.0, 256, 200000
    s0 = 1 / math.sqrt(kappa)
    mu_g = z * s0
    E = mu_g + RNG.standard_normal((R, n))
    m = E.sum(1) / (kappa + n)
    v = 1 / (kappa + n)
    y = mu_g + RNG.standard_normal(R)
    mc = np.mean(np.abs(y - m) <= Z90 * math.sqrt(1 + v))
    b, cov = conjugate_coverage(z, kappa, n)
    line(f"T3 conjugate coverage z={z} kappa={kappa} n={n}: formula "
         f"{cov:.4f}, simulation {mc:.4f}; bias {b:.4f}")
    line("T3 table: 90%-interval coverage at the window end n = L = 256 "
         "(rows: target z prior-sd away; cols: prior worth kappa steps)")
    ks = (16, 64, 256, 1024, 4096)
    line("      " + "".join(f"k={k:<7d}" for k in ks))
    for zz in (1.645, 2.0, 3.0, 4.0):
        line(f"z={zz:<4} " + "".join(
            f"{conjugate_coverage(zz, k, 256)[1]:<9.3f}" for k in ks))


# ------------------------------------------------------------------ T4
def misfit_relation():
    """Using the prior's typical relation (correlation rho_f) on a target
    with rho_g, standardised: MSE ratio to the unconditional variance."""
    rf, rg = 0.24, -0.05                   # DEFECTS M15 pitch numbers
    mse_f = 1 - 2 * rf * rg + rf * rf
    line(f"T4 relation misfit (M15 pitch: prior corr {rf}, target {rg}): "
         f"MSE using the prior relation {mse_f:.4f} x Var, ignoring it "
         f"1.0000, best {1 - rg * rg:.4f}")


# ------------------------------------------------------------------ T5
def ols_coverage(rho):
    """Plug-in Gaussian predictive with sigma^2 from the in-sample
    residuals, p = rho N parameters fitted by least squares: nominal 90%
    coverage (large N, Gaussian design)."""
    return 2 * stats.norm.cdf(Z90 * (1 - rho)) - 1


def check_ols():
    N = 400
    out = []
    for rho in (0.1, 0.25, 0.5, 0.75):
        p = int(rho * N)
        hits, R = 0, 4000
        for _ in range(R):
            X = RNG.standard_normal((N, p))
            beta = RNG.standard_normal(p)
            y = X @ beta + RNG.standard_normal(N)
            bh, *_ = np.linalg.lstsq(X, y, rcond=None)
            s2 = np.sum((y - X @ bh) ** 2) / N
            x = RNG.standard_normal(p)
            yy = x @ beta + RNG.standard_normal()
            hits += abs(yy - x @ bh) <= Z90 * math.sqrt(s2)
        out.append(f"rho {rho}: formula {ols_coverage(rho):.3f} sim "
                   f"{hits / R:.3f}")
    line("T5 plug-in coverage after fitting p = rho N parameters: "
         + "; ".join(out))


def window_prior_sd(tau=0.02):
    for nm, W in (("variant a, |W| = 256", 256), ("w / p, E|W| ~ 178", 178)):
        line(f"T5 effective prior sd tau sqrt(n/|W|), {nm}: " + ", ".join(
            f"n={n}: {tau * math.sqrt(max(n, W) / W):.3f}"
            for n in (256, 1000, 4800, 15000)))


# ------------------------------------------------------------------ T6
def selection_bias():
    """MPPI softmax weights w ~ exp(-(C + eps) / lam), eps ~ N(0, s^2) per
    candidate, equal true costs: weighted mean of eps vs -s^2/lam (K ->
    infinity) and the argmin case -s E[max_K Z]."""
    K, lam, R = 128, 0.6, 20000
    emax = np.mean(np.max(RNG.standard_normal((R, K)), 1))
    rows = []
    for s in (0.1, 0.3, 0.6):
        eps = s * RNG.standard_normal((R, K))
        w = np.exp(-(eps - eps.min(1, keepdims=True)) / lam)
        w /= w.sum(1, keepdims=True)
        rows.append(f"s={s}: sim {np.mean((w * eps).sum(1)):+.4f}, "
                    f"-s^2/lam {-s * s / lam:+.4f}, argmin {-s * emax:+.3f}")
    line(f"T6 selection bias, K={K}, lam={lam} (E max of {K} normals "
         f"{emax:.3f}): " + "; ".join(rows))


# ------------------------------------------------------------------ T7
def trigger():
    """The 90% rule of online_stream.Monitor with the miss share of the
    five velocity channels per step (proxy rows: no head items): fires
    when the 60-step mean > 0.10 + 2 sqrt(0.09 / 60). Probability that the
    window mean is above it, for channel misses independent (300
    Bernoulli) or fully correlated (60 Bernoulli)."""
    thr = 0.10 + 2 * math.sqrt(0.09 / 60)
    k_ind = math.floor(thr * 300) + 1
    k_cor = math.floor(thr * 60) + 1
    rows = []
    for p in (0.10, 0.15, 0.20, 0.25, 0.30):
        pi = stats.binom.sf(k_ind - 1, 300, p)
        pc = stats.binom.sf(k_cor - 1, 60, p)
        rows.append(f"p={p:.2f}: indep {pi:.3f}, corr {pc:.3f}")
    line(f"T7 trigger threshold {thr:.4f}; P(window above): " + "; ".join(rows))
    up = math.ceil(math.log(3) / math.log(1.03))
    dn = math.ceil(math.log(3) / math.log(1.01))
    line(f"T7 inflation 1 -> 3 in {up} steps ({up * 0.24:.1f} s), "
         f"3 -> 1 in {dn} steps ({dn * 0.24:.1f} s) (when the rule acts "
         "every step)")
    mission_trigger()
    tail_rule()
    T, g = 480, 0.05
    for a in (0.1, 0.01):
        line(f"T7 ACI bound one mission (T={T}, gamma={g}, alpha={a}): "
             f"|miss - alpha| <= {(max(a, 1 - a) + g) / (g * T):.3f}")


def mission_trigger(T=480, R=4000):
    """Sliding 90% rule over one mission (miss indicators i.i.d. in time):
    P(fires at least once), firing allowed from n >= 20 (N_MIN)."""
    rows = []
    for p in (0.10, 0.15, 0.20):
        res = []
        for mode in ("indep", "corr"):
            if mode == "indep":
                x = RNG.binomial(5, p, (R, T)) / 5.0
            else:
                x = RNG.binomial(1, p, (R, T)).astype(float)
            c = np.cumsum(x, 1)
            fired = np.zeros(R, bool)
            for t in range(19, T):
                n = min(t + 1, 60)
                tot = c[:, t] - (c[:, t - 60] if t >= 60 else 0.0)
                fired |= tot / n > 0.10 + 2 * math.sqrt(0.09 / n)
            res.append(f"{mode} {fired.mean():.3f}")
        rows.append(f"p={p:.2f}: " + ", ".join(res))
    line(f"T7 P(90% rule fires at least once in a {T}-step mission): "
         + "; ".join(rows))


def tail_rule():
    """Gaussian scale misfit giving 85% coverage of the 90% band: the 99%
    band's miss rate vs the 240-step tail threshold."""
    r = stats.norm.ppf(0.925) / Z90            # sigma_hat / sigma
    m99 = 2 * stats.norm.sf(stats.norm.ppf(0.995) * r)
    thr = 0.01 + 2 * math.sqrt(0.0099 / 240)
    line(f"T7 scale misfit with 85% coverage of the 90% band (sigma_hat = "
         f"{r:.3f} sigma): 99% band miss rate {m99:.4f} vs tail threshold "
         f"{thr:.4f} at n = 240")


def adam_bound():
    b1, b2 = 0.9, 0.999
    r = (1 - b1) / math.sqrt(1 - b2)
    worst = (1 - b1) / math.sqrt((1 - b2) * (1 - b1 * b1 / b2))
    line(f"T5 Adam per-step move: {r:.2f} x lr in Kingma & Ba's sparse case,"
         f" {worst:.2f} x lr worst case (Cauchy-Schwarz, large t); typical "
         f"about lr; LoRA lr 1e-3 -> tau / lr = {0.02 / 1e-3:.0f}")


# ------------------------------------------------------------------ T8
def coef_skill(r, kappa, n):
    """e = beta W + xi, W ~ N(0,1) observed, xi ~ N(0, 1) (sigma = 1);
    prior beta ~ N(0, 1/kappa) (symmetric: worth kappa steps); target
    beta^2 = r. One-step skill (MSE / E e^2) of the posterior-mean
    prediction after n steps (large-n approximation of sum W^2 = n)."""
    return (1.0 + (r * kappa * kappa + n) / (n + kappa) ** 2) / (r + 1.0)


def check_coef():
    r, kappa, n, R = 9.0, 256.0, 256, 20000
    beta = math.sqrt(r)
    W = RNG.standard_normal((R, n))
    e = beta * W + RNG.standard_normal((R, n))
    bh = (W * e).sum(1) / ((W * W).sum(1) + kappa)
    w = RNG.standard_normal(R)
    y = beta * w + RNG.standard_normal(R)
    mc = np.mean((y - bh * w) ** 2) / (r + 1.0)
    line(f"T8 observed-input coefficient, r={r} kappa={kappa} n={n}: "
         f"formula skill {coef_skill(r, kappa, n):.4f}, simulation {mc:.4f}"
         f", floor {1 / (r + 1):.4f}, at n=0 {coef_skill(r, kappa, 0):.3f}")
    rows = []
    for k in (0.1, 16, 256, 1024):
        g = (1 - coef_skill(9.0, k, 256)) / (1 - 1 / 10.0)
        rows.append(f"kappa={k}: {g:.2f}")
    line("T8 share of the available gain used at n = 256 (r = 9): "
         + ", ".join(rows))


def decomposition():
    """Skill = floor + model gap (law of total variance). Floors bounded
    above by DEFECTS M11 (target fine-tuning, held-out); skills: meta5
    one-step on C (run_m15 log)."""
    ch = ("surge", "sway", "yaw", "pitch")
    floor_h = (0.11, 0.12, 0.33, 0.47)        # history only (upper ends)
    floor_w = (0.03, 0.04, 0.10, 0.35)        # + 15-point elevations at t
    sk_a = (0.32, 0.57, 0.78, 1.10)
    sk_w = (0.32, 0.72, 0.81, 0.82)
    line("T8 model gap >= skill - floor bound: a "
         + ", ".join(f"{c} {s - f:.2f}" for c, s, f in zip(ch, sk_a, floor_h))
         + "; w " + ", ".join(f"{c} {s - f:.2f}"
                              for c, s, f in zip(ch, sk_w, floor_w)))


# ------------------------------------------------------------------ T10
def ident_skill(D2, kappa, n, d, B2):
    """e = beta.W + xi, W ~ N(0, I_d) observed, xi ~ N(0, 1); prior
    beta ~ N(beta_bar, I / kappa); |beta_bar - beta_g|^2 = D2,
    |beta_g|^2 = B2. One-step skill of the posterior mean after n steps
    (large n: sum W W^T = n I)."""
    return (1.0 + (kappa * kappa * D2 + n * d) / (kappa + n) ** 2) / (B2 + 1.0)


def check_ident():
    d, n, R, B2 = 6, 256, 4000, 9.0
    bg = np.full(d, math.sqrt(B2 / d))
    bb = bg + np.r_[1.0, np.zeros(d - 1)]          # D2 = 1
    for kappa in (6.0, 64.0):
        err = []
        for _ in range(R):
            W = RNG.standard_normal((n, d))
            e = W @ bg + RNG.standard_normal(n)
            A = W.T @ W + kappa * np.eye(d)
            bh = np.linalg.solve(A, kappa * bb + W.T @ e)
            w = RNG.standard_normal(d)
            err.append((w @ bg + RNG.standard_normal() - w @ bh) ** 2)
        line(f"T10 d={d} D2=1 kappa={kappa}: formula "
             f"{ident_skill(1.0, kappa, n, d, B2):.4f}, simulation "
             f"{np.mean(err) / (B2 + 1):.4f}")
    ks = np.linspace(0.5, 60, 400)
    best = ks[np.argmin([ident_skill(1.0, k, n, d, B2) for k in ks])]
    line(f"T10 best prior strength kappa* = d / D2 = {d}: numeric "
         f"{best:.2f}; excess at kappa* {d / (d / 1.0 + n):.4f} "
         f"(= d D2 / (d + n D2))")
    rows = []
    for dd in (60, 6):
        for D2 in (4.0, 0.25):
            ex = (ident_skill(D2, 64.0, n, dd, B2) - 1 / (B2 + 1))
            rows.append(f"d={dd} D2={D2}: {ex:.3f}")
    line("T10 excess skill at kappa=64, n=256, r=9: " + "; ".join(rows))


def main():
    mixture_regret()
    check_conjugate()
    check_horizon()
    misfit_relation()
    check_ols()
    window_prior_sd()
    adam_bound()
    selection_bias()
    trigger()
    check_coef()
    decomposition()
    check_ident()


if __name__ == "__main__":
    main()
