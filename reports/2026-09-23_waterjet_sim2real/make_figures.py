#!/usr/bin/env python3
"""
Four figures for the 2026-09-23 slides (English): the waterjet model, its
validation against public data, the first sim-to-real results, and why the
low-fidelity world failed. Drawn from the repository's own results
(studies/_cache/sim2real_wigley10_jet.json, the data tables in
studies/exp_waterjet.py) and short runs of the models.

Run:  .venv/Scripts/python.exe reports/2026-09-23_waterjet_sim2real/make_figures.py
"""
import json
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
os.chdir(REPO)
OUT = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "legend.fontsize": 11.5, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E3E7EB", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "savefig.dpi": 200, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.titlelocation": "left",
    "axes.titleweight": "bold"})
C_TGT = "#1F6F8B"      # full simulation ("the real boat")
C_SRC = "#E07A2F"      # low-fidelity world
C_DATA = "#222222"     # measurements
C_GREY = "#9AA9B8"
C_DARK = "#2B3A4A"
NOTE = dict(fontsize=10.5, color="#666666")
KN = 0.514444


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("   ", name)


def arr(rs, k="score"):
    return np.array([r[k] for r in rs])


# ------------------------------------------------------------------ 1
def fig_waterjet():
    from sim.waterjet import JetPump, Waterjet
    pump = JetPump(np.pi * 0.085 ** 2 / 4, 224e3, eta_pump=0.78)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    u = np.linspace(0, 50, 101)
    for frac, al in ((1.0, 1.0), (0.75, 0.75), (0.5, 0.55), (0.25, 0.35)):
        T = [max(pump.thrust(frac * pump.p_max, x * KN), 0) / 1e3 for x in u]
        a.plot(u, T, color=C_TGT, alpha=al, lw=2.4)
        a.text(1.0, T[2] + 0.25, f"{int(frac * 100)}% power", ha="left",
               va="bottom", fontsize=11.5, color=C_TGT, alpha=max(al, 0.7))
    a.set(xlabel="Boat speed (kn)", ylabel="Forward thrust (kN)",
          xlim=(0, 50), ylim=(0, None))
    a.set_title("(a) Same power, less thrust as the boat speeds up",
                fontsize=13.5)
    u_ref = 20 * KN
    t_max = pump.thrust(pump.p_max, u_ref)
    jet = Waterjet(pump, -2.8, -0.2, -2.3, -0.3, t_max=t_max,
                   t_min=-t_max / 3, tau=0.5, rate_max=1e9,
                   dead_band=t_max * 50 / 12000, delay=0.0, u_ref=u_ref,
                   dt=0.05)
    cmd = np.linspace(0, 1, 201)
    fx, fy = zip(*[jet.forces(c * t_max, np.radians(20), u_ref) for c in cmd])
    b.plot(cmd * 100, np.array(fy) / 1e3, color=C_SRC, lw=2.6,
           label="Side (steering) force")
    b.plot(cmd * 100, np.array(fx) / 1e3, color=C_TGT, lw=2.6,
           label="Forward thrust")
    b.set(xlabel="Throttle (thrust command, %)", ylabel="Force (kN)",
          xlim=(0, 100), ylim=(0, None))
    b.legend(frameon=False, loc="upper left")
    b.annotate("Zero throttle: no thrust and no steering", xy=(0.8, 0.03),
               xytext=(24, 0.25), fontsize=11.5,
               arrowprops=dict(arrowstyle="->", color="#555555"))
    b.annotate("Low throttle: steering force\nexceeds thrust",
               xy=(5, 0.35), xytext=(3, 3.6), fontsize=11.5, color=C_SRC,
               arrowprops=dict(arrowstyle="->", color=C_SRC))
    b.set_title("(b) At 20 kn with the nozzle turned 20°", fontsize=13.5)
    fig.text(0.0, -0.04, "Illustrative VM 18-class pump: 300 hp, 85 mm nozzle, "
             "pump efficiency 0.78 (magnitudes from public data)", **NOTE)
    save(fig, "fig1_waterjet_model.png")


# ------------------------------------------------------------------ 2
def fig_validation():
    from scipy.optimize import least_squares
    from sim.waterjet import JetPump
    from studies.exp_waterjet import (HJ291, HP, KGF, SCARAB, P_RATED,
                                      top_speed, sample)
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.6),
                               gridspec_kw=dict(width_ratios=[1.25, 1]))

    # (a) HamiltonJet curves: fit on 200 hp, predict the rest
    def model(q, hp, kn):
        pump = JetPump(q[0], hp * HP, eta_pump=q[1])
        return np.array([pump.thrust(hp * HP, k * KN) for k in kn]) / KGF
    kn, t = np.array(HJ291[200]).T
    fit = least_squares(lambda q: (model(q, 200, kn) - t) / t, x0=(0.02, 0.7),
                        bounds=([1e-3, 0.3], [0.2, 1.0]))
    cmap = plt.get_cmap("GnBu")
    hps = sorted(HJ291)
    for i, hp in enumerate(hps):
        col = cmap(0.35 + 0.62 * i / (len(hps) - 1))
        kn, t = np.array(HJ291[hp]).T
        ks = np.linspace(kn.min(), 40, 60)
        ms = model(fit.x, hp, ks)
        cal = hp == 200
        a.plot(ks, ms, color=C_SRC if cal else col, lw=3.0 if cal else 2.0)
        a.plot(kn, t, "o", ms=6, mfc="white", mec=C_SRC if cal else C_DATA,
               mew=1.5)
        a.text(40.6, ms[-1], f"{hp} hp" + (" (fit)" if cal else ""),
               va="center", fontsize=10.5, color=C_SRC if cal else C_DARK)
    a.plot([], [], "o", mfc="white", mec=C_DATA, mew=1.5,
           label="Manufacturer chart")
    a.plot([], [], color=cmap(0.8), lw=2, label="Model, predicted")
    a.plot([], [], color=C_SRC, lw=3, label="Model, fitted curve")
    a.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13),
             ncol=3)
    a.set(xlabel="Boat speed (kn)", ylabel="Thrust (kgf)", xlim=(0, 40),
          ylim=(0, 1350))
    a.set_title("(a) Thrust curves: fit 1 of 8, predict 7 (rms error 2.3%)",
                fontsize=13.5)

    # (b) top speed: the near-twin measured vs modelled, and VM 18
    L, beam, beta = SCARAB["L"], SCARAB["beam"], SCARAB["deadrise"]

    def tops(mass, lcg, beam_, keys, cda_rng=None, seed=0):
        rng = np.random.default_rng(seed)
        out = []
        for s in sample(300, rng, keys):
            cda = s["cda_scarab"] if cda_rng is None else rng.uniform(*cda_rng)
            pump = JetPump(np.pi * s["d_nozzle"] ** 2 / 4, P_RATED,
                           eta_pump=s["eta_pump"])
            lcg_ = lcg if lcg is not None else s["lcg_frac"] * L
            v = top_speed(pump, P_RATED, mass, lcg_, s["chine_frac"] * beam_,
                          beta, cda, v_hi=45.0)
            if v:
                out.append(v / KN)
        return np.percentile(out, [10, 50, 90])
    k5 = ("chine_frac", "lcg_frac", "d_nozzle", "eta_pump", "cda_scarab")
    k3 = ("chine_frac", "d_nozzle", "eta_pump")
    rows = [("Scarab 195, measured (2 tests)", None, [42.06, 42.7]),
            ("Scarab 195, model", tops(1490, None, beam, k5), None),
            ("VM 18, model, wet 1206 kg",
             tops(1206, 2.67, 1.67, k3, (0.8, 1.4), 1), None),
            ("VM 18, model, max 1815 kg",
             tops(1815, 2.67, 1.67, k3, (0.8, 1.4), 1), None),
            ("VM 18, brochure", None, None)]
    for i, (nm, pct, pts) in enumerate(rows):
        y = len(rows) - 1 - i
        b.text(36.3, y + 0.22, nm, fontsize=11.5, color="#333333",
               va="bottom")
        if pct is not None:
            b.plot([pct[0], pct[2]], [y, y], color=C_TGT, lw=7, alpha=0.35,
                   solid_capstyle="round")
            b.plot(pct[1], y, "o", color=C_TGT, ms=9)
        elif pts is not None:
            b.plot(pts, [y] * len(pts), "o", ms=8, mfc="white", mec=C_DATA,
                   mew=1.8)
        else:
            b.annotate("", xy=(53.5, y), xytext=(50, y),
                       arrowprops=dict(arrowstyle="-|>", color=C_SRC, lw=2.2))
            b.plot(50, y, "|", color=C_SRC, ms=16, mew=2.5)
            b.text(50.3, y - 0.12, "\"50+ kn\"", color=C_SRC, fontsize=11.5,
                   va="top")
    b.set_yticks([])
    b.set_ylim(-0.6, len(rows) - 0.25)
    b.set(xlabel="Top speed (kn)", xlim=(36, 54))
    b.grid(axis="y", visible=False)
    b.set_title("(b) Top speed: near-twin boat and VM 18", fontsize=13.5)
    fig.text(0.0, -0.2,
             "(a) HamiltonJet unit '291' datasheet (in MacPherson 2000), read "
             "from the chart; the model's two unknowns (nozzle area, pump "
             "efficiency) fitted on 200 hp only.\n(b) Model = waterjet "
             "(Rotax 300 hp) + Savitsky planing resistance + air drag; "
             "unknown hull numbers sampled over stated ranges, band = 80%, "
             "dot = median.\nScarab data: Boating magazine tests. VM 18 "
             "deadrise assumed 20° (as Scarab).", **NOTE)
    save(fig, "fig2_validation.png")


# ------------------------------------------------------------------ 3
METHODS = [
    ("A0 hand", "Hand-tuned weights", 0),
    ("A2 src-DR", "Low-fi + randomization (run 1)", 0),
    ("A2r src-DR", "Low-fi + randomization (run 2)", 0),
    ("B3 model", "Adapt the MPC's internal model", 1),
    ("B1 tgt-CMA", "Target: evolution strategy (CMA-ES)", 1),
    ("B2t tgt-GP", "Target: GP search (run 1)", 1),
    ("B2tr tgt-GP", "Target: GP search (run 2)", 1),
    ("B2 MF-GP", "Target: GP + low-fi prior (run 1)", 2),
    ("B2r MF-GP", "Target: GP + low-fi prior (run 2)", 2),
    ("O oracle", "Large-budget reference", 3),
]
CATS = [("0 target episodes", C_GREY), ("32 target episodes", "#6FA8BF"),
        ("32 target episodes + low-fi prior", C_SRC),
        ("320 target episodes", C_DARK)]


def fig_results(R):
    ev = R["eval"]
    base = arr(ev["A1 src"]["high"])
    rows = []
    for key, lab, cat in METHODS:
        d = -(arr(ev[key]["high"]) - base) / base.mean() * 100
        rows.append((lab, d.mean(), d.std(ddof=1) / np.sqrt(len(d)), cat))
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ys = np.arange(len(rows))[::-1]
    for y, (lab, m, se, cat) in zip(ys, rows):
        ax.barh(y, m, xerr=se, color=CATS[cat][1], height=0.62, capsize=4,
                error_kw=dict(lw=1.2, ecolor="#555555"))
        ax.text(max(m + se, 0) + 1.2, y, f"{m:+.0f}%", va="center",
                fontsize=11.5, color="#333333")
    ax.axvline(0, color="#333333", lw=1.2)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Improvement over tuning in the low-fi world only "
                  "(%, higher is better)")
    ax.set_xlim(-8, 58)
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=col, label=lab) for lab, col in CATS],
              frameon=False, loc="center right")
    fig.text(0.0, -0.03, "Paired evaluation on the same 16 unseen sea "
             "realizations in the full simulation; error bars = standard error. "
             "'Run 2' = the same method with another random seed.", **NOTE)
    save(fig, "fig3_sim2real_results.png")


# ------------------------------------------------------------------ 4
def fig_why(R):
    from scipy.stats import spearmanr
    ev = R["eval"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.4),
                               gridspec_kw=dict(width_ratios=[1, 1.15]))
    names = list(ev)
    x = np.array([arr(ev[n]["low"]).mean() for n in names])
    y = np.array([arr(ev[n]["high"]).mean() for n in names])
    rho = spearmanr(x, y).correlation
    a.plot([0.4, 1.0], [0.4, 1.0], color="#CCCCCC", lw=1.2, ls="--")
    a.scatter(x, y, s=70, color=C_TGT, edgecolor="white", lw=1.2, zorder=3)
    lab = {"A1 src": "tuned in low-fi only", "O oracle": "large-budget ref.",
           "B2r MF-GP": "GP + low-fi prior"}
    for n, xi, yi in zip(names, x, y):
        if n in lab:
            a.annotate(lab[n], (xi, yi), xytext=(-12, 0),
                       textcoords="offset points", va="center", ha="right",
                       fontsize=11)
    a.set(xlim=(0.4, 1.0), ylim=(0.4, 1.0), xlabel="Score in the low-fi world",
          ylabel="Score in the full simulation")
    a.set_aspect("equal")
    a.text(0.42, 0.97, f"13 controllers\nrank correlation {rho:+.2f}",
           va="top", fontsize=12)
    a.text(0.42, 0.43, "lower score = better", fontsize=10.5,
           color="#777777")
    a.set_title("(a) The low-fi world cannot rank controllers",
                fontsize=13.5)

    # paired change from weight 0, per world: the sea realizations are the
    # same at every weight, so this is what the weight itself does
    sw = R["sweep_track"]
    keys = list(sw)
    wt = np.array([float(k) for k in keys])
    for w, col, nm in (("high", C_TGT, "Full simulation"),
                       ("low", C_SRC, "Low-fi world")):
        s0 = arr(sw[keys[0]][w])
        d = [(arr(sw[k][w]) - s0) / s0.mean() * 100 for k in keys]
        b.errorbar(wt, [v.mean() for v in d],
                   yerr=[v.std(ddof=1) / np.sqrt(len(v)) for v in d],
                   color=col, lw=2.4, marker="o", ms=7, capsize=4, label=nm)
        if w == "high":
            for xi, k, v in zip(wt, keys, d):
                u = arr(sw[k]["high"], "u_mean").mean()
                b.annotate(f"{u:.2f} m/s", (xi, v.mean()), xytext=(0, -24),
                           textcoords="offset points", ha="center",
                           fontsize=10.5, color=C_TGT)
    b.axhline(0, color="#999999", lw=0.8)
    b.set(xlabel="Cross-track weight (only this weight changed)",
          ylabel="Change in score vs. weight 0 (%)")
    b.set_ylim(-16, 3)
    b.legend(frameon=False, loc="lower left")
    b.set_title("(b) One weight moves the target, not the low-fi world",
                fontsize=13.5)
    fig.text(0.0, -0.05, "(b) Paired change on the same 16 sea realizations; "
             "labels = mean speed in the full simulation (waves turn the hull, "
             "the MPC predicts growing\ncross-track error and slows down). "
             "The low-fi world has no lateral wave force or added resistance: "
             "its score does not change at all.", **NOTE)
    save(fig, "fig4_why_lowfi_fails.png")


def main():
    R = json.load(open(os.path.join(REPO, "studies", "_cache",
                                    "sim2real_wigley10_jet.json")))
    print("  figures ->", OUT)
    fig_waterjet()
    fig_validation()
    fig_results(R)
    fig_why(R)


if __name__ == "__main__":
    main()
