#!/usr/bin/env python3
"""Figures and report tables of the stability map (studies/stability_map.py
parts figs and report). Reads studies/_cache/stability/*.json."""
import json
import math
import os

import numpy as np

from studies import stability_lin as L

KN = L.KN
G = 9.81
# categorical palette, fixed order (dataviz reference palette, light mode)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
     "#4a3aa7", "#e34948"]
GRID = "#d9d8d3"
INK = "#0b0b0b"
INK2 = "#52514e"
# Savitsky 1964 porpoising-limit chart (Day & Haag 1952 data) for 20 deg
# deadrise, digitised as a cubic in x = sqrt(C_Lbeta / 2) by OpenPlaning
# (Castro-Feliciano, python-openplaning, check_porpoising); valid for
# C_Lbeta 0.0338-0.18 (the chart's range)
SAV20 = (-254.51, 239.65, -23.936, 3.0195)
CL_RANGE = (0.0338, 0.18)
MASS, BEAM = 1400.0, 0.83 * 2.44


def _load(out, name):
    with open(os.path.join(out, name)) as f:
        return json.load(f)


def tau_crit_sav20(cl):
    x = math.sqrt(cl / 2.0)
    a, b, c, d = SAV20
    return a * x ** 3 + b * x ** 2 + c * x + d


def cl_beta(u, rho=1025.0):
    return MASS * G / (0.5 * rho * u * u * BEAM * BEAM) if u > 0 else float("inf")


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=INK2)


def make_all(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = _load(out, "map.json")
    cv = _load(out, "curve.json")
    rows = [r for r in m["rows"] if "error" not in r]
    kn = np.array([r["kn"] for r in rows])

    # ---------------------------------------------------- eigenvalue map
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax = axs[0, 0]
    _style(ax, "Natural frequency of the oscillatory modes", "speed, kn",
           "rad/s")
    for r in rows:
        for md in r["analysis"]["vertical"]["modes"]:
            if md["kind"] == "pair":
                ax.plot(r["kn"], md["wn"], "o", ms=5, color=C[0], mec="#fcfcfb",
                        mew=0.6)
        for md in r["analysis"]["lateral"]["modes"]:
            if md["kind"] == "pair":
                col = C[2] if abs(md["wn"] - 4.448) < 0.3 else C[1]
                ax.plot(r["kn"], md["wn"], "s", ms=5, color=col, mec="#fcfcfb",
                        mew=0.6)
    ax.axhline(3.94, color=C[6], lw=1.2, ls="--")
    ax.text(0.5, 3.94 + 0.15, "BEM heave at rest 3.94", fontsize=7,
            color=C[6])
    ax.plot([], [], "o", color=C[0], label="heave / pitch (coupled)")
    ax.plot([], [], "s", color=C[2], label="roll")
    ax.plot([], [], "s", color=C[1], label="sway-yaw pair")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax = axs[0, 1]
    _style(ax, "Damping ratio of the oscillatory modes", "speed, kn", "zeta")
    for r in rows:
        for md in r["analysis"]["vertical"]["modes"]:
            if md["kind"] == "pair":
                ax.plot(r["kn"], md["zeta"], "o", ms=5, color=C[0],
                        mec="#fcfcfb", mew=0.6)
        for md in r["analysis"]["lateral"]["modes"]:
            if md["kind"] == "pair":
                col = C[2] if abs(md["wn"] - 4.448) < 0.3 else C[1]
                ax.plot(r["kn"], md["zeta"], "s", ms=5, color=col,
                        mec="#fcfcfb", mew=0.6)
    ax.axhline(0.05, color=C[7], lw=1.0, ls="--")
    ax.text(30, 0.06, "zeta 0.05", fontsize=7, color=C[7])
    ax.axhline(0.30, color=C[6], lw=1.2, ls="--")
    ax.text(30, 0.31, "BEM heave at rest 0.30", fontsize=7, color=C[6])
    ax.set_ylim(-0.02, 1.05)
    ax = axs[1, 0]
    _style(ax, "Real part of the non-oscillatory modes (vertical block)",
           "speed, kn", "1/s (symlog)")
    for r in rows:
        for md in r["analysis"]["vertical"]["modes"]:
            if md["kind"] == "real" and abs(md["re"] + 1.681) > 0.01:
                ax.plot(r["kn"], md["re"], "o", ms=5, color=C[0],
                        mec="#fcfcfb", mew=0.6)
    fine = cv["fine"]
    ax.plot([f["kn"] for f in fine], [f["surge_re"] for f in fine], "-",
            lw=1.5, color=C[3], label="slowest real mode, 12-15 kn fine grid")
    ax.axhline(0.0, color=INK2, lw=0.8)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.axvspan(13.0, 13.65, color=C[3], alpha=0.15, lw=0)
    ax.text(13.8, -0.003, "fold 13.0-13.65 kn", fontsize=7, color=C[3])
    ax.plot([], [], "o", color=C[0], label="real modes (surge, overdamped "
                                         "heave/pitch)")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax = axs[1, 1]
    _style(ax, "Real part of the sway-yaw modes (lateral block)", "speed, kn",
           "1/s (symlog)")
    for r in rows:
        for e in r["sway_yaw_eig"]:
            ax.plot(r["kn"], e[0], "s", ms=5, color=C[1], mec="#fcfcfb",
                    mew=0.6)
    ax.axhline(0.0, color=INK2, lw=0.8)
    ax.set_yscale("symlog", linthresh=0.01)
    for r in rows:
        mx = max(e[0] for e in r["sway_yaw_eig"])
        if mx > 0:
            ax.annotate(f"{mx:+.4f}", (r["kn"], mx), fontsize=7, color=C[7],
                        xytext=(3, 4), textcoords="offset points")
    fig.suptitle("scarab195@as1, calm water: open-loop eigenvalues of the "
                 "one-step map (dt 0.01 s) along the equilibrium branch",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "stability_map.png"))
    plt.close(fig)

    # --------------------------------------------------- thrust curve
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    co = cv["coarse"]
    ax = axs[0]
    _style(ax, "Thrust that holds the speed, c = T*/t_max", "speed, kn",
           "throttle fraction")
    ax.plot([r["kn"] for r in co], [r["c"] for r in co], "-", lw=1.8,
            color=C[0])
    db = 26.5236 / 6365.663
    ax.axhline(db, color=C[7], lw=1.0, ls="--")
    ax.text(20, db + 0.01, "command dead band 26.5 N (c 0.0042): below "
            "4.75 kn no constant command holds the speed", fontsize=7,
            color=C[7])
    ax.axvspan(13.0, 13.65, color=C[3], alpha=0.2, lw=0)
    ax.set_ylim(0, 1.0)
    ax = axs[1]
    _style(ax, "The fold, 12-15 kn (fine grid 0.05 kn)", "speed, kn",
           "throttle fraction")
    ax.plot([r["kn"] for r in fine], [r["c"] for r in fine], "-", lw=1.8,
            color=C[0])
    w = cv["fold_fine"]["c_window"]
    if w:
        ax.axhspan(w[0], w[1], color=C[3], alpha=0.35, lw=0)
        ax.text(12.1, w[1] + 0.0003, f"three equilibria for c in "
                f"{w[0]:.5f}-{w[1]:.5f}", fontsize=7, color=C[3])
    ax.set_ylim(0.228, 0.238)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "thrust_curve.png"))
    plt.close(fig)

    # ------------------------------------------------------ porpoising
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    _style(ax, "Running trim against the Savitsky (1964) porpoising limit, "
               "deadrise 20 deg", "speed, kn", "trim, deg (bow-up)")
    ks = np.array([r["kn"] for r in co])
    tr = np.array([r["trim_deg"] for r in co])
    cls = np.array([cl_beta(k * KN) for k in ks])
    lim = np.array([tau_crit_sav20(c) if 0 < c < 5 else np.nan for c in cls])
    inr = (cls >= CL_RANGE[0]) & (cls <= CL_RANGE[1])
    ax.plot(ks, tr, "-", lw=1.8, color=C[0], label="plant running trim")
    ax.plot(ks[inr], lim[inr], "-", lw=1.8, color=C[1],
            label="porpoising limit (chart range C_L 0.034-0.18)")
    ax.plot(ks[~inr & (ks > 10)], lim[~inr & (ks > 10)], ":", lw=1.4,
            color=C[1], label="limit extrapolated beyond the chart")
    ax.set_xlim(8, 45)
    ax.set_ylim(0, 12)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "porpoising.png"))
    plt.close(fig)

    # ----------------------------------------------------- sensitivity
    try:
        sj = _load(out, "sens.json")
    except OSError:
        sj = None
    if sj:
        jobs = [j for j in sj["jobs"] if "error" not in j]
        nom = [j for j in jobs if j["param"] == "nominal"][0]
        params = sj["params"]
        fig, axs = plt.subplots(2, 4, figsize=(13, 6.5), dpi=150)
        fig.patch.set_facecolor("#fcfcfb")
        for ci, kn_ in enumerate(sj["speeds"]):
            def pick(j):
                r = j["rows"][ci]
                v = r["analysis"]["vertical"]["modes"]
                lat = r["sway_yaw_eig"] if "sway_yaw_eig" in r else None
                pairs = [x for x in v if x["kind"] == "pair"]
                zmin = min(x["zeta"] for x in pairs) if pairs else np.nan
                lam = [x for x in r["analysis"]["lateral"]["modes"]
                       if abs(x["re"] + 10.536) > 0.05
                       and not (x["kind"] == "pair" and abs(x["wn"] - 4.448)
                                < 0.4)]
                slow = max(x["re"] for x in lam) if lam else np.nan
                return zmin, slow, r["dc_du"]
            z0, s0, d0 = pick(nom)
            for row, (lab, idx) in enumerate((("smallest zeta, heave/pitch "
                                              "pairs", 0),
                                             ("slowest sway-yaw eigenvalue, "
                                              "1/s", 1))):
                ax = axs[row, ci]
                _style(ax, f"{kn_:g} kn", lab, None)
                for k, prm in enumerate(params):
                    lo = [j for j in jobs if j["param"] == prm
                          and j["which"] == "lo"]
                    hi = [j for j in jobs if j["param"] == prm
                          and j["which"] == "hi"]
                    if not lo or not hi:
                        continue
                    a, b = pick(lo[0])[idx], pick(hi[0])[idx]
                    ax.plot([a, b], [k, k], "-", lw=2, color=C[0])
                    ax.plot(a, k, "o", ms=6, color=C[0], mec="#fcfcfb")
                    ax.plot(b, k, "s", ms=6, color=C[1], mec="#fcfcfb")
                ax.axvline([z0, s0][idx], color=INK2, lw=0.8)
                if idx == 1:
                    ax.axvline(0.0, color=C[7], lw=0.8, ls="--")
                ax.set_yticks(range(len(params)))
                ax.set_yticklabels(params if ci == 0 else [""] * len(params),
                                   fontsize=7)
                ax.invert_yaxis()
        axs[0, 0].plot([], [], "o", color=C[0], label="low end of the range")
        axs[0, 0].plot([], [], "s", color=C[1], label="high end")
        axs[0, 0].legend(fontsize=7, frameon=False, loc="lower right")
        fig.suptitle("Randomisation extremes, one parameter at a time "
                     "(vertical line: nominal)", fontsize=10, color=INK)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "sensitivity.png"))
        plt.close(fig)
    print("figures written to", out)


# ================================================================ report
def _pairs(modes):
    return "; ".join(f"ω {x['wn']:.2f} ζ {x['zeta']:.3f}" for x in modes
                     if x["kind"] == "pair") or "—"


def _reals(modes, skip_thrust=True):
    out = []
    for x in modes:
        if x["kind"] != "real":
            continue
        if skip_thrust and abs(x["re"] + 1.681) < 0.01:
            continue
        out.append(f"{x['re']:+.3f}")
    return ", ".join(out) or "—"


def report(out):
    m = _load(out, "map.json")
    cv = _load(out, "curve.json")
    rows = [r for r in m["rows"] if "error" not in r]
    st = m["states"]
    lines = []
    P = lines.append
    P("# stability map tables (generated by studies/stability_map.py report)")
    P("")
    P("## T1 equilibria (calm water, free surge; GZ twin, dt 0.01)")
    P("| kn | rise z m | trim deg | c = T*/t_max | R N | pressure/friction/air N"
      " | wetted keel m | wet stations | transom h_w m | holdable by a "
      "constant command | roots found | residual (deriv, step) |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        e = r["eq"]
        P(f"| {r['kn']:g} | {e['z']:+.4f} | {e['trim_deg']:.3f} | {e['c']:.5f}"
          f" | {e['R']:.1f} | {e['pressure']:.0f}/{e['friction']:.0f}/"
          f"{e['air']:.0f} | {e['lw']:.2f} | {e['n_wet']} | "
          f"{e.get('hw', float('nan')):.3f} | {'yes' if r['held_by_command'] else 'no (T* below dead band)'}"
          f" | {len(r['census'])} | {r['deriv_resid']:.0e}, {r['step_resid']:.0e} |")
    P("")
    P("## T2 vertical block (z, th, u, w, q, T): continuous eigenvalues from "
      "the one-step map, eps 1e-4, dt 0.01")
    P("| kn | oscillatory pairs | real modes 1/s (thrust lag −1.68 omitted) | "
      "surge: slowest real | flags |")
    P("|---|---|---|---|---|")
    for r in rows:
        v = r["analysis"]["vertical"]["modes"]
        reals = [x for x in v if x["kind"] == "real"
                 and abs(x["re"] + 1.681) > 0.01]
        surge = max(reals, key=lambda x: x["re"]) if reals else None
        fl = []
        for x in v:
            if x["re"] > 1e-6:
                fl.append(f"Re>0 ({x['re']:+.1e})")
            if x["kind"] == "pair" and x["zeta"] < 0.05:
                fl.append(f"ζ<0.05 ({x['zeta']:.3f})")
        P(f"| {r['kn']:g} | {_pairs(v)} | {_reals(v)} | "
          f"{surge['re']:+.4f} (τ {surge['tau']:.0f} s)" if surge else
          f"| {r['kn']:g} | {_pairs(v)} | {_reals(v)} | —"
          )
        lines[-1] += f" | {'; '.join(fl) or '—'} |"
    P("")
    P("## T3 lateral block (phi, v, p, r, delta)")
    P("| kn | roll ω, ζ | sway–yaw eigenvalues 1/s | flags |")
    P("|---|---|---|---|")
    for r in rows:
        lat = r["analysis"]["lateral"]["modes"]
        roll = [x for x in lat if x["kind"] == "pair" and abs(x["wn"] - 4.448) < 0.4]
        sy = r["sway_yaw_eig"]
        fl = [f"Re>0 ({e[0]:+.4f})" for e in sy if e[0] > 1e-6]
        P(f"| {r['kn']:g} | {_pairs(roll)} | "
          + "; ".join(f"{e[0]:+.4f}{e[1]:+.3f}j" if abs(e[1]) > 1e-9 else
                      f"{e[0]:+.4f}" for e in sy)
          + f" | {'; '.join(fl) or '—'} |")
    P("")
    P("## T4 linearity and switches (max over columns of the relative "
      "difference of the one-step Jacobians)")
    P("| kn | 1e-5 vs 1e-4 (state) | 1e-4 vs 1e-3 (state) | switches crossed "
      "within ±1e-3 | within ±1e-4 |")
    P("|---|---|---|---|---|")
    for r in rows:
        l1 = r["linearity"]["1e-5 vs 1e-4"]
        l2 = r["linearity"]["1e-4 vs 1e-3"]
        i1, i2 = int(np.argmax(l1)), int(np.argmax(l2))
        sw3 = {k: v for k, v in r["switches"]["0.001"].items() if v}
        sw4 = {k: v for k, v in r["switches"]["0.0001"].items() if v}

        def fmt(sw):
            if not sw:
                return "none"
            return "; ".join(f"{k}: " + ", ".join(
                f"{kk} {str(vv)[:40]}" for kk, vv in d.items())
                for k, d in sw.items())
        P(f"| {r['kn']:g} | {l1[i1]:.1e} ({st[i1]}) | {l2[i2]:.1e} ({st[i2]})"
          f" | {fmt(sw3)} | {fmt(sw4)} |")
    P("")
    P("## T5 step-size check: eigenvalues at dt 0.005 vs 0.01 (vertical and "
      "lateral blocks, actuator poles excluded)")
    P("| kn | max relative deviation of |λ| | max deviation of Re λ, 1/s |")
    P("|---|---|---|")
    for r in rows:
        dev_abs, dev_re = 0.0, 0.0
        for blk in ("vertical", "lateral"):
            a = [complex(x[0], x[1]) for x in r["analysis"][blk]["lam"]]
            b = [complex(x[0], x[1]) for x in r["analysis_dt2"][blk]["lam"]]
            a = [x for x in a if abs(x + 1.681) > 0.02 and abs(x + 10.536) > 0.5]
            b = [x for x in b if abs(x + 1.674) > 0.02 and abs(x + 10.259) > 0.5]
            for x in a:
                y = min(b, key=lambda y: abs(y - x))
                if abs(x) > 1e-6:
                    dev_abs = max(dev_abs, abs(abs(y) - abs(x)) / abs(x))
                dev_re = max(dev_re, abs(y.real - x.real))
        P(f"| {r['kn']:g} | {dev_abs:.1e} | {dev_re:.1e} |")
    P("")
    P("## T6 decay from the equilibrium (60 s, commands held): vertical "
      "(rise +2 cm, trim +0.5 deg) and lateral (sway 0.2 m/s, yaw rate "
      "0.05 rad/s)")
    P("| kn | rise dev. at 10 / 60 s, m | trim dev. at 10 / 60 s, deg | "
      "peak-to-peak last 10 s: rise m, trim deg | speed drift at 60 s, m/s | "
      "yaw rate at 10 / 60 s, rad/s |")
    P("|---|---|---|---|---|---|")
    for r in rows:
        dv, dl = r["decay_vertical"], r["decay_lateral"]
        P(f"| {r['kn']:g} | {dv['z_at']['10']:+.4f} / {dv['z_at']['60']:+.4f}"
          f" | {dv['trim_at']['10']:+.3f} / {dv['trim_at']['60']:+.3f} | "
          f"{dv['pp_z']:.1e}, {dv['pp_trim']:.1e} | {dv['u_at']['60']:+.3f} | "
          f"{dl['r_at']['10']:+.4f} / {dl['r_at']['60']:+.4f} |")
    P("")
    P("## T7 fold of the thrust–speed curve")
    P(f"fine grid (12–15 kn, 0.05 kn): {cv['fold_fine']['summary']}")
    ex = cv["fold_fine"]["example"]
    if ex:
        P(f"example c {ex['c']:.5f}: equilibria at "
          + ", ".join(f"{k:.3f} kn ({'stable' if s else 'unstable'})"
                      for k, s in zip(ex["roots"], ex["stable"])))
    P(f"coarse grid (0.5–45 kn, 0.25 kn): {cv['fold_coarse']['summary']}")
    co = cv["coarse"]
    lo = [r for r in co if r["T"] >= 26.5236]
    P(f"lowest speed whose holding thrust state is at or above the command "
      f"dead band (26.5 N): {lo[0]['kn']:.2f} kn (T* {lo[0]['T']:.1f} N)")
    dry = [r for r in co if r["hw"] is not None and r["hw"] <= 0]
    P(f"transom face dry (h_w = 0) from {dry[0]['kn']:.2f} kn")
    P("")
    P("| kn (fine) | c | trim deg | rise m | pressure N | friction N | wetted"
      " keel m | wet stations | h_w m | slowest real λ 1/s |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for r in cv["fine"][::4]:
        P(f"| {r['kn']:.2f} | {r['c']:.5f} | {r['trim_deg']:.3f} | {r['z']:+.4f}"
          f" | {r['pressure']:.0f} | {r['friction']:.0f} | {r['lw']:.2f} | "
          f"{r['n_wet']} | {r['hw']:.3f} | {r['surge_re']:+.5f} |")
    P("")
    P("## T9 porpoising limit (Savitsky 1964 chart, 20 deg deadrise, "
      "OpenPlaning cubic) vs the plant's running trim")
    P("| kn | C_Lβ | x = sqrt(C_Lβ/2) | limit trim deg | plant trim deg | "
      "margin deg | chart range |")
    P("|---|---|---|---|---|---|---|")
    for r in rows:
        if r["kn"] < 8:
            continue
        cl = cl_beta(r["u"])
        lim = tau_crit_sav20(cl)
        P(f"| {r['kn']:g} | {cl:.4f} | {math.sqrt(cl / 2):.3f} | {lim:.2f} | "
          f"{r['eq']['trim_deg']:.2f} | {lim - r['eq']['trim_deg']:+.2f} | "
          f"{'in' if CL_RANGE[0] <= cl <= CL_RANGE[1] else 'OUT'} |")
    P("")
    # ---------------------------------------------------- sensitivity
    try:
        sj = _load(out, "sens.json")
    except OSError:
        sj = None
    if sj:
        jobs = [j for j in sj["jobs"] if "error" not in j]
        nom = [j for j in jobs if j["param"] == "nominal"][0]
        for ci, kn_ in enumerate(sj["speeds"]):
            P(f"## T10 sensitivity at {kn_:g} kn (one parameter at a time at "
              f"its range ends; nominal first)")
            P("| parameter | end | value | trim deg | rise m | c | dc/du "
              "(per m/s) | surge λ 1/s | heave/pitch pairs ω, ζ | other real "
              "λ | sway–yaw λ | roll ω, ζ | linearity 1e-4 vs 1e-3 |")
            P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")

            def line(j, label, end):
                r = j["rows"][ci]
                v = r["analysis"]["vertical"]["modes"]
                lat = r["analysis"]["lateral"]["modes"]
                reals = [x for x in v if x["kind"] == "real"
                         and abs(x["re"] + 1.681) > 0.01]
                surge = max(reals, key=lambda x: x["re"]) if reals else None
                others = [x for x in reals if x is not surge]
                roll = [x for x in lat if x["kind"] == "pair"
                        and abs(x["wn"] - 4.448) < 0.4]
                sy = [x for x in lat if abs(x["re"] + 10.536) > 0.05
                      and x not in roll]
                e = r["eq"]
                val = "" if j["value"] is None else f"{j['value']:.4g}"
                P(f"| {label} | {end} | {val} | {e['trim_deg']:.3f} | "
                  f"{e['z']:+.4f} | {e['c']:.5f} | {r['dc_du']:+.5f} | "
                  f"{surge['re']:+.4f} | {_pairs(v)} | "
                  f"{', '.join(f'{x[chr(114)+chr(101)]:+.2f}' for x in others) or '—'}"
                  f" | " + "; ".join(
                      f"{x['re']:+.3f}{x['im']:+.3f}j" if x["im"] else
                      f"{x['re']:+.4f}" for x in sy)
                  + f" | {_pairs(roll)} | {max(r['linearity']):.1e} |")
            line(nom, "nominal", "—")
            for prm in sj["params"]:
                for end in ("lo", "hi"):
                    jj = [j for j in jobs if j["param"] == prm
                          and j["which"] == end]
                    if jj:
                        line(jj[0], prm, end)
            P("")
    # -------------------------------------------------------- reduced
    try:
        rd = _load(out, "reduced.json")
    except OSError:
        rd = None
    if rd:
        p = rd["params"]
        P("## T11 the MPC's reduced model vs the plant's linearisation")
        P(f"reduced model (cached identification, u_ss {p['u_ss'] / KN:.1f} kn):"
          f" heave ω {p['wn_heave']:.2f} ζ {p['z_heave']:.3f}; pitch ω "
          f"{p['wn_pitch']:.2f} ζ {p['z_pitch']:.3f}; tau_u {p['tau_u']:.2f} s"
          f" (m_surge {p['m_surge']:.0f} kg, k_drag {p['k_drag']:.2f}); tau_r "
          f"{p['tau_r']:.2f} s; k_nomoto_f {p['k_nomoto_f']:.3e}, k_jet_side "
          f"{p['k_jet_side']:.3f}; k_lin_sway {p['k_lin_sway']:.0f} N s/m, "
          f"m_sway {p['m_sway']:.0f} kg; z0 {p['z0']:.4f} m, th0 "
          f"{p['th0']:.4f} rad")
        P("| kn | plant diag. heave ω, ζ | plant diag. pitch ω, ζ | plant "
          "coupled modes (label) | plant surge τ s (fixed attitude) | reduced"
          " surge τ s | plant sway–yaw λ | reduced sway/yaw poles | Nomoto "
          "gain r/δ plant / reduced, 1/s |")
        P("|---|---|---|---|---|---|---|---|---|")
        for r in rd["rows"]:
            d = r["plant_diag"]
            vm = r["plant_modes_vertical"]
            lab = "; ".join(
                (f"{x['label']} ω {x['wn']:.2f} ζ {x['zeta']:.2f}" if x["kind"]
                 == "pair" else f"{x['label']} {x['re']:+.3f}")
                for x in vm if x["label"] != "thrust")
            red = r["reduced_sway_yaw_A"]
            P(f"| {r['kn']:g} | {d['wn_heave']:.2f}, {d['z_heave']:.3f} | "
              f"{d['wn_pitch']:.2f}, {d['z_pitch']:.3f} | {lab} | "
              f"{r['plant_surge_tau']:.1f} ({r['plant_surge_tau_fixed_attitude']:.1f})"
              f" | {r['reduced_surge_tau']:.1f} | "
              + "; ".join(f"{e[0]:+.3f}{e[1]:+.3f}j" if abs(e[1]) > 1e-9 else
                          f"{e[0]:+.4f}" for e in r["plant_sway_yaw_eig"])
              + f" | {red[0][0]:+.3f}, {red[1][1]:+.3f} | "
              f"{r['plant_K_nomoto']:+.3f} / {r['reduced_K_nomoto']:+.3f} |")
        P("")
        P("### T11b the plant's own heave/pitch coefficients (continuous "
          "Jacobian A_c of the free plant, 1/s^2 and 1/s)")
        P("| kn | A[w,z] (heave stiffness) | A[w,w] (heave damping) | A[w,th] |"
          " A[w,q] | A[q,z] | A[q,w] | A[q,th] (pitch stiffness) | A[q,q] "
          "(pitch damping) | A[u,z] | A[u,th] | A[w,u] | A[q,u] |")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        ii = {n: k for k, n in enumerate(st)}
        for r in rows:
            A = np.array(r["A"])

            def a(x, y):
                return A[ii[x], ii[y]]
            P(f"| {r['kn']:g} | {a('w', 'z'):+.2f} | {a('w', 'w'):+.3f} | "
              f"{a('w', 'th'):+.2f} | {a('w', 'q'):+.3f} | {a('q', 'z'):+.2f} |"
              f" {a('q', 'w'):+.3f} | {a('q', 'th'):+.2f} | {a('q', 'q'):+.3f}"
              f" | {a('u', 'z'):+.3f} | {a('u', 'th'):+.3f} | "
              f"{a('w', 'u'):+.3f} | {a('q', 'u'):+.3f} |")
        P("")
    # --------------------------------------------------------- closed
    try:
        cl = _load(out, "closed.json")
    except OSError:
        cl = None
    if cl:
        P("## T12 closed loop: throttle P+I hold and heading autopilot, control"
          " step 0.25 s, delay 0.1 s")
        P("| kn | autopilot a, b (rad per rad, rad per rad/s), K used 1/s | "
          "speed loop: slowest mode | gain margin factor (dB) | phase margin "
          "deg | heading loop: slowest mode | gain margin factor (dB) | phase "
          "margin deg | all closed-loop |λ_d| < 1 |")
        P("|---|---|---|---|---|---|---|---|---|")
        held = {r["kn"]: r["held_by_command"] for r in rows}
        for r in cl["rows"]:
            g = r["gains"]
            sp, hd = r["speed"], r["heading"]
            if not held.get(r["kn"], True) or r["kn"] == 0:
                sp = dict(sp, slowest="no thrust authority (T* below the "
                                      "command dead band)")

            def gmf(x):
                f = x["gain_margin_factor"]
                if not np.isfinite(f):
                    return "∞"
                if f <= 0.0:
                    return "0 (unstable at the nominal gain)"
                return f"{f:.2f} ({20 * math.log10(f):.1f})"

            def pm(x):
                v = x["phase_margin_deg"]
                return "—" if v is None else f"{v:.0f}"
            P(f"| {r['kn']:g} | {g['a']:+.3f}, {g['b']:+.3f}, {g['K']:+.3f} | "
              f"{sp['slowest']} | {gmf(sp)} | {pm(sp)} | {hd['slowest']} | "
              f"{gmf(hd)} | {pm(hd)} | {'yes' if r['stable'] else 'NO'} |")
        P("")
        P("### closed-loop check: nonlinear plant with the controllers as "
          "written vs the linear lifted model (60 s)")
        P("| kn | heading: 5 deg error — |ψ| max after 10 s nonlinear / linear"
          " deg, rms difference deg | speed: −1 m/s error — |Δu| max after "
          "10 s nonlinear / linear m/s, rms difference m/s |")
        P("|---|---|---|")
        for r in cl["check"]:
            h, s = r["heading"], r["speed"]
            P(f"| {r['kn']:g} | {math.degrees(h['nonlinear_abs_max_after_10s']):.2f}"
              f" / {math.degrees(h['linear_abs_max_after_10s']):.2f}, "
              f"{math.degrees(h['rms_diff']):.2f} | "
              f"{s['nonlinear_abs_max_after_10s']:.3f} / "
              f"{s['linear_abs_max_after_10s']:.3f}, {s['rms_diff']:.3f} |")
        P("")
    txt = "\n".join(lines)
    with open(os.path.join(out, "report.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"report written: {len(lines)} lines")
