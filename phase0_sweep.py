#!/usr/bin/env python3
"""
PHASE 0 -- how much wave preview does the controller actually need?

No reinforcement learning here, on purpose. "What is preview information
worth?" is answered by an MPC with PERFECT (oracle) preview swept over the
preview horizon. RL only enters in Phase 2, to tune the weights this study
holds fixed. Building the RL stack first would answer nothing.

The confound this design has to kill: a controller can always cut motion by
simply going slower. So the comparison is made on the PARETO FRONT of
(mean speed) vs (motion severity). Preview has value only if its front
dominates the no-information front, i.e. it is better AT THE SAME SPEED.

Method
  * irregular short-crested seaway, precomputed on an (x, t) grid
  * reduced longitudinal seakeeping model (vessel.py)
  * sampling MPC (MPPI) over commanded speed, parameterised by a few knots so
    the sampler actually explores coordinated slow-down / speed-up plans
  * beyond the preview horizon the controller assumes eta = 0, the statistical
    expectation of a zero-mean surface

Output: four figures written next to this file.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from step0_preview_spec import WaveField, SEA_STATES
import vessel as V

DT = 0.1                 # model timestep (s)
DT_CTRL = 0.5            # control interval (s)
SUB = int(round(DT_CTRL / DT))
H = 24                   # MPPI horizon in control steps -> 12 s
N_KNOT = 7               # control knots across the horizon
K = 384                  # MPPI rollouts
SIGMA = 1.2              # exploration std on knot speeds (m/s)
LAMBDA = 0.5
U_REF = 6.0
W_ACC, W_SLAM, W_DU = 1.0, 12.0, 0.3

QUICK = "--quick" in sys.argv
# Slamming is a RARE event: at ~3/min a 100 s run yields ~5 events, i.e. +-45%
# Poisson noise -- far too coarse to resolve a 20% effect. Long runs are not
# optional here, they are what makes the metric mean anything.
T_SIM = 100.0 if QUICK else 300.0


class WaveGrid:
    """Precomputed eta(x, t) and deta/dt along the track. Turns every wave
    lookup into an interpolation, which is what makes the rollouts affordable.
    Both grids are analytic, not finite-differenced."""

    def __init__(self, wf, x_max=None, t_max=None, dx=1.0, dt=0.2):
        x_max = x_max if x_max is not None else T_SIM * 9.5 + 200.0
        t_max = t_max if t_max is not None else T_SIM + 30.0
        self.xg = np.arange(0.0, x_max + dx, dx)
        self.tg = np.arange(0.0, t_max + dt, dt)
        self.dt = dt
        kx = wf.k * np.cos(wf.th)                            # along-track k
        A = wf.a * np.exp(1j * wf.phi)
        Xm = np.exp(1j * kx[:, None] * self.xg[None, :])     # (nc, nx)
        Tm = np.exp(-1j * wf.w[:, None] * self.tg[None, :])  # (nc, nt)
        self.ETA = np.real((A[:, None] * Xm).T @ Tm)
        Ad = A * (-1j * wf.w)                                # d/dt analytically
        self.ETA_DOT = np.real((Ad[:, None] * Xm).T @ Tm)

    def _line(self, arr, t):
        ft = np.clip(t / self.dt, 0, len(self.tg) - 1.001)
        i0 = int(ft)
        w = ft - i0
        return (1.0 - w) * arr[:, i0] + w * arr[:, i0 + 1]

    def at(self, x, t):
        return np.interp(np.asarray(x).ravel(), self.xg,
                         self._line(self.ETA, t)).reshape(np.shape(x))

    def at_dot(self, x, t):
        return np.interp(np.asarray(x).ravel(), self.xg,
                         self._line(self.ETA_DOT, t)).reshape(np.shape(x))


KNOT_T = np.linspace(0, H - 1, N_KNOT)


def knots_to_seq(knots):
    """(N, N_KNOT) -> (N, H). Smooth plans, and a 7-D search space instead of
    24-D, which is the difference between MPPI working and not working."""
    out = np.empty((knots.shape[0], H))
    grid = np.arange(H)
    for i in range(knots.shape[0]):
        out[i] = np.interp(grid, KNOT_T, knots[i])
    return out


def _advance(hull, grid, s, u_cmd, t, cost=None, w_spd=None):
    """One control interval (SUB model steps). Accumulates cost if asked."""
    acc_last = slam_last = None
    for _ in range(SUB):
        xs = s[:, 0:1] + hull.stations[None, :]
        eta = grid.at(xs, t)
        eta_dot_bow = grid.at_dot(s[:, 0] + hull.stations[-1], t)
        s, a_bow, slam = V.step(hull, s, u_cmd, eta, eta_dot_bow, DT)
        if cost is not None:
            cost += (W_ACC * (a_bow / V.G) ** 2
                     + W_SLAM * slam
                     + w_spd * ((U_REF - s[:, 1]) / U_REF) ** 2) * DT
        acc_last, slam_last = a_bow, slam
        t += DT
    return s, acc_last, slam_last


def _advance_blind(hull, s, u_cmd, cost, w_spd):
    """Same, but the controller cannot see the surface: eta = its expectation."""
    n = s.shape[0]
    zero = np.zeros((n, hull.stations.size))
    zero_b = np.zeros(n)
    for _ in range(SUB):
        s, a_bow, slam = V.step(hull, s, u_cmd, zero, zero_b, DT)
        cost += (W_ACC * (a_bow / V.G) ** 2 + W_SLAM * slam
                 + w_spd * ((U_REF - s[:, 1]) / U_REF) ** 2) * DT
    return s


def rollout_cost(hull, grid, s0, useq, t0, t_prev, w_spd):
    s = np.repeat(s0[None, :], useq.shape[0], axis=0)
    cost = np.zeros(useq.shape[0])
    for h in range(H):
        t = t0 + h * DT_CTRL
        if (h * DT_CTRL) <= t_prev:
            s, _, _ = _advance(hull, grid, s, useq[:, h], t, cost, w_spd)
        else:
            s = _advance_blind(hull, s, useq[:, h], cost, w_spd)
    return cost


def run_mppi(hull, grid, t_prev, w_spd, seed=0, trace=False):
    rng = np.random.default_rng(1000 + seed)
    s = V.init_state(1, U_REF)
    nom = np.full(N_KNOT, U_REF)
    acc, slams, spd, tr = [], 0.0, [], []

    for n in range(int(T_SIM / DT_CTRL)):
        t0 = n * DT_CTRL
        cand_k = np.clip(nom[None, :] + rng.normal(0, SIGMA, (K, N_KNOT)),
                         hull.u_min, hull.u_max)
        useq = knots_to_seq(cand_k)
        c = rollout_cost(hull, grid, s[0], useq, t0, t_prev, w_spd)
        c += W_DU * (np.diff(useq, axis=1) ** 2).sum(axis=1)
        wts = np.exp(-(c - c.min()) / LAMBDA)
        nom = (wts[:, None] * cand_k).sum(0) / wts.sum()

        u_cmd = np.full(1, nom[0])
        t = t0
        for _ in range(SUB):
            xs = s[:, 0:1] + hull.stations[None, :]
            eta = grid.at(xs, t)
            ed = grid.at_dot(s[:, 0] + hull.stations[-1], t)
            s, a_bow, slam = V.step(hull, s, u_cmd, eta, ed, DT)
            acc.append(a_bow[0]); slams += slam[0]; spd.append(s[0, 1])
            if trace:
                tr.append((t, s[0, 0], s[0, 1], a_bow[0], slam[0], eta[0, -1]))
            t += DT
        nom = np.r_[nom[1:], nom[-1]]

    return _metrics(t_prev, w_spd, seed, acc, slams, spd,
                    np.asarray(tr) if trace else None)


def run_constant(hull, grid, u_set, seed=0):
    """No-information baseline: hold a constant commanded speed."""
    s = V.init_state(1, min(u_set, U_REF))
    acc, slams, spd = [], 0.0, []
    u_cmd = np.full(1, u_set)
    t = 0.0
    for _ in range(int(T_SIM / DT)):
        xs = s[:, 0:1] + hull.stations[None, :]
        eta = grid.at(xs, t)
        ed = grid.at_dot(s[:, 0] + hull.stations[-1], t)
        s, a_bow, slam = V.step(hull, s, u_cmd, eta, ed, DT)
        acc.append(a_bow[0]); slams += slam[0]; spd.append(s[0, 1])
        t += DT
    return _metrics(None, None, seed, acc, slams, spd, None)


def _metrics(t_prev, w_spd, seed, acc, slams, spd, tr):
    acc = np.asarray(acc)
    out = dict(t_prev=t_prev, w_spd=w_spd, seed=seed,
               acc_rms=float(np.sqrt((acc ** 2).mean()) / V.G),
               acc_p99=float(np.percentile(np.abs(acc), 99) / V.G),
               slam_per_min=float(slams / (T_SIM / 60.0)),
               u_mean=float(np.mean(spd)))
    if tr is not None:
        out["trace"] = tr
    return out


def main():
    p = SEA_STATES[5]
    hull = V.Hull()
    print(f"SS5 Hs={p['hs']} m Tp={p['tp']} s | L={hull.L} m draft={hull.draft} m "
          f"v_slam={hull.v_slam:.2f} m/s")

    seeds = [0] if QUICK else [0, 1, 2]
    horizons = [0.0, 4.0, 8.0] if QUICK else [0.0, 2.0, 4.0, 8.0, 12.0]
    wspds = [2.0] if QUICK else [1.0, 2.0, 4.0]

    grids = {}
    for sd in seeds:
        wf = WaveField(p["hs"], p["tp"], n_freq=48, n_dir=8, seed=sd)
        grids[sd] = WaveGrid(wf)
        print(f"  grid seed {sd}: realised Hs = {4*grids[sd].ETA.std():.2f} m")

    # ---- no-information baseline front --------------------------------
    print("\nconstant-speed baseline (no wave information):")
    base = []
    for u_set in (2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5):
        rs = [run_constant(hull, grids[sd], u_set, sd) for sd in seeds]
        b = {k: float(np.mean([r[k] for r in rs]))
             for k in ("u_mean", "acc_rms", "acc_p99", "slam_per_min")}
        base.append(b)
        print(f"  u={u_set:4.1f} -> u_mean={b['u_mean']:.2f}  "
              f"acc_rms={b['acc_rms']:.3f}g  p99={b['acc_p99']:.3f}g  "
              f"slam={b['slam_per_min']:.1f}/min")

    # ---- preview-informed MPC -----------------------------------------
    print("\nMPPI with oracle preview:")
    rows, traces = [], {}
    for tp_ in horizons:
        for w in wspds:
            rs = []
            for sd in seeds:
                keep = (sd == seeds[0] and w == 2.0 and tp_ in (0.0, 8.0))
                r = run_mppi(hull, grids[sd], tp_, w, sd, trace=keep)
                if keep:
                    traces[tp_] = r.pop("trace")
                rs.append(r)
            m = {k: float(np.mean([r[k] for r in rs]))
                 for k in ("u_mean", "acc_rms", "acc_p99", "slam_per_min")}
            m.update(t_prev=tp_, w_spd=w)
            rows.append(m)
            print(f"  T_prev={tp_:4.1f}s w_spd={w:3.1f}  u_mean={m['u_mean']:.2f}  "
                  f"acc_rms={m['acc_rms']:.3f}g  p99={m['acc_p99']:.3f}g  "
                  f"slam={m['slam_per_min']:.1f}/min")

    _plot_pareto(base, rows, horizons)
    _plot_knee(rows, wspds, horizons, base)
    _plot_traces(traces)
    _plot_seaway(grids[seeds[0]], traces)
    print("\nfigures: fig1_pareto.png fig2_knee.png fig3_timeseries.png "
          "fig4_seaway.png")


def _cmap(horizons):
    return {h: plt.cm.viridis(i / max(len(horizons) - 1, 1))
            for i, h in enumerate(horizons)}


def _plot_pareto(base, rows, horizons):
    cm = _cmap(horizons)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for i, (key, lab) in enumerate([("acc_p99", "bow accel, 99th pct  (g)"),
                                    ("slam_per_min", "slams per minute")]):
        bu = [b["u_mean"] for b in base]
        ax[i].plot(bu, [b[key] for b in base], "k-o", ms=4, lw=1.6,
                   label="no information (constant speed)")
        for h in horizons:
            pts = [r for r in rows if r["t_prev"] == h]
            ax[i].scatter([r["u_mean"] for r in pts], [r[key] for r in pts],
                          s=70, color=cm[h], edgecolor="k", linewidth=.5,
                          zorder=3, label=f"preview {h:g} s")
        ax[i].set_xlabel("mean speed  (m/s)   $\\rightarrow$ better")
        ax[i].set_ylabel(lab + "   $\\downarrow$ better")
        ax[i].grid(alpha=.3)
    ax[0].legend(fontsize=8)
    fig.suptitle("Phase 0 -- preview is worth something only if its points sit "
                 "BELOW the black curve (better at equal speed)", fontsize=11)
    fig.tight_layout()
    fig.savefig("fig1_pareto.png", dpi=140)


def _plot_knee(rows, wspds, horizons, base):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for i, (key, lab) in enumerate([("acc_p99", "bow accel, 99th pct (g)"),
                                    ("slam_per_min", "slams per minute")]):
        for w in wspds:
            pts = sorted([r for r in rows if r["w_spd"] == w],
                         key=lambda r: r["t_prev"])
            ax[i].plot([r["t_prev"] for r in pts], [r[key] for r in pts],
                       "-o", ms=5, label=f"$w_{{speed}}$={w:g}")
        ax[i].set_xlabel("preview horizon  $T_{prev}$  (s)")
        ax[i].set_ylabel(lab)
        ax[i].grid(alpha=.3)
        ax[i].legend(fontsize=8)
    fig.suptitle("Where is the knee? (each curve holds the speed penalty fixed)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig("fig2_knee.png", dpi=140)


def _plot_traces(traces):
    if not traces:
        return
    fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for tp_, c, lab in [(0.0, "tab:red", "$T_{prev}$=0 s (blind)"),
                        (8.0, "tab:green", "$T_{prev}$=8 s (preview)")]:
        if tp_ not in traces:
            continue
        tr = traces[tp_]
        ax[0].plot(tr[:, 0], tr[:, 5], c=c, lw=.8, alpha=.85, label=lab)
        ax[1].plot(tr[:, 0], tr[:, 2], c=c, lw=1.4, label=lab)
        ax[2].plot(tr[:, 0], tr[:, 3] / V.G, c=c, lw=.7, alpha=.8, label=lab)
        sl = tr[tr[:, 4] > 0]
        if len(sl):
            ax[2].plot(sl[:, 0], sl[:, 3] / V.G, "v", c=c, ms=8, mec="k",
                       mew=.6, ls="none")
    ax[0].set_ylabel("$\\eta$ at bow (m)"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    ax[1].set_ylabel("speed (m/s)"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    ax[2].set_ylabel("bow accel (g)"); ax[2].set_xlabel("time (s)")
    ax[2].grid(alpha=.3); ax[2].legend(fontsize=8)
    ax[2].set_title("markers = slam events", fontsize=9, loc="right")
    fig.suptitle("Blind vs preview-informed speed scheduling (SS5)", fontsize=12)
    fig.tight_layout()
    fig.savefig("fig3_timeseries.png", dpi=140)


def _plot_seaway(grid, traces):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    im = ax.pcolormesh(grid.tg, grid.xg, grid.ETA, cmap="RdBu_r",
                       vmin=-3, vmax=3, shading="auto", rasterized=True)
    plt.colorbar(im, ax=ax, label="$\\eta$ (m)")
    for tp_, c, lab in [(0.0, "k", "blind"), (8.0, "lime", "preview 8 s")]:
        if tp_ in traces:
            tr = traces[tp_]
            ax.plot(tr[:, 0], tr[:, 1], c=c, lw=1.8, label=f"track, {lab}")
    ax.set_xlabel("time (s)"); ax.set_ylabel("along-track position (m)")
    ax.set_ylim(0, 700); ax.legend(fontsize=8, loc="upper left")
    ax.set_title("SS5 seaway in space-time. Crests run down-left toward the "
                 "boat; the boat's track is the rising line.", fontsize=11)
    fig.tight_layout()
    fig.savefig("fig4_seaway.png", dpi=140)


if __name__ == "__main__":
    main()
