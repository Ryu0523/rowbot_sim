#!/usr/bin/env python3
"""
S3 -- radiation-memory fitting tool for the forward-speed kernel
(learn/meta/HULL_FORCE_AUDIT_REVIEW_2026-10-09.md, "综合判断" -> "下一步" item 3).

At 5 kn the retardation function has a t^-1 oscillating tail at the critical
frequency g/(4U) (0.95 rad/s); the repository's time-domain fit (<= 10 states,
30 s window, A_inf estimated from the tail) gets the damping wrong by 21-41 %
near the following-sea encounter frequencies. This study

  1  diagnoses the two failures (aliasing of the cosine-sum kernel for windows
     longer than pi / d_omega; the A_inf estimate biased by the frequency
     truncation of the table, not mainly by the tail);
  2  sweeps the repaired time-domain path (fine-grid kernel, external A_inf,
     order 10-24, window 30-90 s) and the new frequency-domain path
     (vector fitting of H(iw) = B + i w (A - A_inf), stability floor,
     passivity enforcement, band weighting; one entry at a time and the
     heave-pitch block with common poles);
  3  checks the zero-speed table (B1.12 quality must be reproduced), the
     solver's own values at the encounter frequencies, stability/passivity,
     a time-domain convolution test at 0.94 and 0.955 rad/s;
  4  measures the online cost of the resulting state spaces.

Outputs: studies/_cache/derive/s3_tables.txt, s3_results.json, s3_fits.png,
s3_kernel.png, s3_timecheck.png.

Run: .venv/Scripts/python.exe -m studies.derive_s3_memory_fit [--parts diag,era,vf,rest,td,cost,plots]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
OUT = os.path.join(HERE, "_cache", "derive")
G = 9.81
N_FINE = 8001                    # fine cosine-transform grid: period 2 pi / (7.8/8000) = 6400 s
BAND = (0.8, 1.0)                # following-sea encounter band at 5 kn
ENTRIES = dict(hh=(0, 0), hp=(0, 1), ph=(1, 0), pp=(1, 1))
_LINES = []


def say(s=""):
    print(s, flush=True)
    _LINES.append(s)


# ------------------------------------------------------------------ tables
def load_5kn(label="base_r0.3"):
    d = np.load(os.path.join(OUT, f"v1b_tab_{label}.npz"))
    j = json.load(open(os.path.join(OUT, f"v1b_tab_{label}.json")))
    ix = [1, 2]                                      # heave, pitch of modes (1, 3, 5)
    lin = dict(omega=d["omega"], A=d["A"][:, ix][:, :, ix], B=d["B"][:, ix][:, :, ix],
               A_inf=d["A_inf"][np.ix_(ix, ix)], B_inf=d["B_inf"][np.ix_(ix, ix)],
               C0=d["C0"][np.ix_(ix, ix)], u=float(j["plant.u"]), label=label)
    cases = {}
    for key in ("head_Tp3", "head_Tp5", "follow_Tp3", "follow_Tp5"):
        cases[key] = dict(omega_e=float(j[f"cases.{key}.omega_e"]),
                          A_e=d[f"cases.{key}.A_e"][np.ix_(ix, ix)], B_e=d[f"cases.{key}.B_e"][np.ix_(ix, ix)])
    lin["cases"] = cases
    lin["uniform"] = np.abs(lin["omega"] * 10 - np.round(lin["omega"] * 10)) < 1e-6
    return lin


def tilde(lin, sel=None):
    """The integrable pair: A + C0/w^2 (-> A_inf) and B - B_inf, (nw, 2, 2)."""
    sel = np.ones(len(lin["omega"]), bool) if sel is None else sel
    w = lin["omega"][sel]
    At = lin["A"][sel] + lin["C0"][None] / w[:, None, None] ** 2
    Bt = lin["B"][sel] - lin["B_inf"][None]
    return w, At, Bt


def case_targets(lin):
    out = {}
    for key, c in lin["cases"].items():
        we = c["omega_e"]
        out[key] = dict(omega_e=we, At=c["A_e"] + lin["C0"] / we ** 2, Bt=c["B_e"] - lin["B_inf"])
    return out


def load_rest():
    z = np.load(os.path.join(HERE, "_cache", "anyspeed", "bem_probe_lid_24_64.npz"))
    return dict(omega=z["omega"], A=z["A"], B=z["B"])


# ------------------------------------------------------------------ helpers
def H_at(f, w):
    from hydro.retardation import transfer_function
    return transfer_function(np.atleast_1d(w), f["Ar"], f["Br"], f["Cr"])


def case_err(f, lin, i, j, keys=("follow_Tp5", "follow_Tp3", "head_Tp5", "head_Tp3")):
    """Relative errors of the fit at the solver's exact encounter-frequency values
    (interpolation-free): |B_fit - Bt_e| / |Bt_e| and |A_fit - At_e| / |At_e - A_inf|."""
    tg = case_targets(lin)
    A_inf = lin["A_inf"][i, j]
    out = {}
    for k in keys:
        we = tg[k]["omega_e"]
        H = H_at(f, we)[0]
        Bf, Af = float(np.real(H)), A_inf + float(np.imag(H)) / we
        Bt, At = tg[k]["Bt"][i, j], tg[k]["At"][i, j]
        out[k] = dict(omega_e=we, B_fit=Bf, B_table=float(Bt), A_fit=Af, A_table=float(At),
                      rel_B=float(abs(Bf - Bt) / max(abs(Bt), 1e-30)),
                      rel_A=float(abs(Af - At) / max(abs(At - A_inf), 1e-30)))
    return out


def block_case_err(fb, lin, keys=("follow_Tp5", "follow_Tp3", "head_Tp5", "head_Tp3")):
    from hydro.retardation import block_transfer_function
    tg = case_targets(lin)
    out = {}
    for k in keys:
        we = tg[k]["omega_e"]
        H = block_transfer_function(np.array([we]), fb["Ar"], fb["Br"], fb["Cr"])[0]
        Bf = np.real(H)
        Af = lin["A_inf"] + np.imag(H) / we
        out[k] = {nm: dict(rel_B=float(abs(Bf[i, j] - tg[k]["Bt"][i, j]) / max(abs(tg[k]["Bt"][i, j]), 1e-30)),
                           rel_A=float(abs(Af[i, j] - tg[k]["At"][i, j]) / max(abs(tg[k]["At"][i, j] - lin["A_inf"][i, j]), 1e-30)))
                  for nm, (i, j) in ENTRIES.items()}
    return out


def fmt_pct(x):
    return "   nan" if x is None or not np.isfinite(x) else f"{100 * x:6.1f}"


def pole_near(f, w0=1.0):
    lam = np.linalg.eigvals(f["Ar"])
    lam = lam[np.imag(lam) >= 0]
    k = int(np.argmin(np.abs(np.imag(lam) - w0)))
    return complex(lam[k])


# ================================================================== part 1: diagnosis
def part_diag(lin, res):
    from hydro.retardation import retardation_function, infinite_added_mass, taper
    say("\n== 1. diagnosis: why the 90 s window blew up, why A_inf was biased ==")
    w, At, Bt = tilde(lin, lin["uniform"])
    dw = float(np.mean(np.diff(w)))
    t = np.arange(0.0, 90.0, 0.02)
    K_plain = retardation_function(w, Bt[:, 0, 0], t)
    K_fine = retardation_function(w, Bt[:, 0, 0], t, n_fine=N_FINE, extend_to_zero=True)
    T_alias = 2 * np.pi / dw
    i_al = int(np.argmin(np.abs(t - T_alias)))
    seg = (t > 55) & (t < 70)
    say(f"  heave kernel from the uniform grid (79 points, step {dw:.2f} rad/s), raised-cosine taper 5.2-8 rad/s:")
    say(f"    plain trapezoid: K(0) = {K_plain[0]:.0f} N/m, K({t[i_al]:.2f} s) = {K_plain[i_al]:.0f}, max |K| in 55-70 s = {np.abs(K_plain[seg]).max():.0f}"
        f"  -> the cosine SUM is periodic with 2 pi / dw = {T_alias:.1f} s; everything past {T_alias / 2:.1f} s is a mirror image")
    say(f"    fine grid ({N_FINE} points, PCHIP): K(0) = {K_fine[0]:.0f}, K({t[i_al]:.2f} s) = {K_fine[i_al]:.0f}, max |K| in 55-70 s = {np.abs(K_fine[seg]).max():.0f}")
    say(f"    (fit_coefficient's default window is min(pi / dw, 30) = {min(np.pi / dw, 30):.1f} s, which is why 30 s worked and 60 / 90 s saw the alias)")
    res["alias"] = dict(T_alias=T_alias, K0=float(K_plain[0]), K_alias_plain=float(K_plain[i_al]), K_alias_fine=float(K_fine[i_al]),
                        max_55_70_plain=float(np.abs(K_plain[seg]).max()), max_55_70_fine=float(np.abs(K_fine[seg]).max()))

    # A_inf estimator decomposition (heave-heave)
    say("\n  A_inf estimator (time-domain Ogilvie relation on a finite window, band average):")
    A_solver = lin["A_inf"][0, 0]
    rows = {}
    for name, T, nf, use_taper in (("plain grid, 30 s, taper", 30.0, None, True),
                                   ("fine grid, 30 s, taper", 30.0, N_FINE, True),
                                   ("fine grid, 90 s, taper", 90.0, N_FINE, True),
                                   ("fine grid, 200 s, taper", 200.0, N_FINE, True),
                                   ("fine grid, 200 s, hard cut at 8", 200.0, N_FINE, False)):
        tt = np.arange(0.0, T, 0.02)
        K = retardation_function(w, Bt[:, 0, 0], tt, use_taper=use_taper, n_fine=nf, extend_to_zero=nf is not None)
        est, spread = infinite_added_mass(w, At[:, 0, 0], tt, K)
        rows[name] = dict(A_inf_est=est, spread=spread)
        say(f"    {name:34s}: {est:7.0f} kg (spread {spread:.3f}); solver {A_solver:.0f}; offset {est - A_solver:+.0f}")
    # analytic share of the unknown part above 8 rad/s: power-law extrapolation of B - B_inf fitted on 6-8 rad/s,
    # Kramers-Kronig at the band centre
    m = (w >= 6.0)
    p = -np.polyfit(np.log(w[m]), np.log(Bt[m, 0, 0]), 1)[0]
    B8 = Bt[-1, 0, 0]
    wq = np.geomspace(8.0, 2000.0, 20001)
    Bext = B8 * (8.0 / wq) ** p
    share = {}
    for wc in (2.05, 3.05, 5.05):
        share[wc] = float((2 / np.pi) * np.trapezoid(Bext / (wq ** 2 - wc ** 2), wq))
    tp = taper(w)
    share_taper = {}
    for wc in (2.05, 3.05, 5.05):
        integrand = (1 - tp) * Bt[:, 0, 0] / (w ** 2 - wc ** 2)
        share_taper[wc] = float((2 / np.pi) * np.trapezoid(integrand, w))
    say(f"    B - B_inf at 8 rad/s is {B8:.0f} = {B8 / Bt[:, 0, 0].max():.2f} of its maximum and decays like w^-{p:.2f} on 6-8 rad/s;")
    say(f"    Kramers-Kronig share of the part above 8 rad/s (power-law extrapolation) at w = 2 / 3 / 5: "
        f"{share[2.05]:+.0f} / {share[3.05]:+.0f} / {share[5.05]:+.0f} kg; share of the tapered 5.2-8 band: "
        f"{share_taper[2.05]:+.0f} / {share_taper[3.05]:+.0f} / {share_taper[5.05]:+.0f} kg")
    say("    -> the +1000 kg offset is the frequency truncation of the table (taper + unknown tail above 8 rad/s), the same at every")
    say("       window; the slow t^-1 tail only moves the estimate by the window-to-window differences above (a few hundred kg).")
    res["A_inf_estimator"] = dict(rows=rows, solver=float(A_solver), B8_share=float(B8 / Bt[:, 0, 0].max()), tail_power_B=float(p),
                                  kk_share_above_8=share, kk_share_taper=share_taper)

    # kernel tail from the fine grid (long window)
    tl = np.arange(0.0, 200.0, 0.02)
    Kl = retardation_function(w, Bt[:, 0, 0], tl, n_fine=N_FINE, extend_to_zero=True)
    env = np.abs(Kl)
    idx = [i for i in range(1, len(tl) - 1) if env[i] > env[i - 1] and env[i] > env[i + 1] and 5.0 < tl[i] < 60.0]
    pw = -np.polyfit(np.log(tl[idx]), np.log(env[idx]), 1)[0]
    zc = np.where(np.diff(np.sign(Kl[tl > 10])) != 0)[0]
    w_tail = math.pi / np.mean(np.diff(tl[tl > 10][zc]))
    peak = env.max()
    say(f"\n  heave kernel tail (fine grid, no alias, 0-200 s): peak {peak:.0f}; envelope 5-60 s ~ t^-{pw:.2f}; oscillation {w_tail:.3f} rad/s "
        f"(critical g/(4U) = {G / (4 * lin['u']):.3f}; table spike at 1.0); |K| < 2 % after {tl[np.max(np.where(env > 0.02 * peak)[0])]:.1f} s, "
        f"< 0.5 % after {tl[np.max(np.where(env > 0.005 * peak)[0])]:.1f} s, < 0.2 % after {tl[np.max(np.where(env > 0.002 * peak)[0])]:.1f} s")
    say("    (beyond ~63 s the kernel is that of the interpolant between grid points: the table's 0.1 rad/s step cannot resolve longer memories)")
    res["kernel_tail"] = dict(peak=float(peak), power=float(pw), w_tail=float(w_tail), w_crit=float(G / (4 * lin["u"])))
    return dict(t=t, K_fine=K_fine, K_plain=K_plain)


# ================================================================== part 2: ERA sweep
def part_era(lin, res):
    from hydro.retardation import fit_coefficient
    say("\n== 2. repaired time-domain path (ERA): fine-grid kernel, external A_inf, order x window ==")
    say("   errors: K on the window (impulse response vs tapered table kernel), B over 0.2-5.2 rad/s and A (both / global max),")
    say("   'passiv' = min Re H / max |Re H| on a dense grid (meaningful for hh and pp only; an off-diagonal damping changes sign)")
    say("   band 0.8-1.0: B error / global max and / in-band max; at w_e 0.843 / 0.944: |dB|/|B| and |dA|/|A - A_inf| of the solver's own values")
    w, At, Bt = tilde(lin, lin["uniform"])
    sweep = {}
    t0 = time.time()
    for ainf_mode in ("est", "solver"):
        say(f"\n  A_inf {'estimated from the kernel (original)' if ainf_mode == 'est' else 'from the solver (new)'}:")
        say(f"  {'entry':>5}{'order':>6}{'T s':>5}{'n':>4}{'err_K':>7}{'err_B':>7}{'err_A':>7}{'B band':>8}{'B bandL':>8}{'A band':>8}"
            f"{'B@.843':>8}{'A@.843':>8}{'B@.944':>8}{'A@.944':>8}{'passiv':>8}{'slow p':>8}")
        for nm, (i, j) in ENTRIES.items():
            for order in (10, 16, 20, 24):
                for T in (30.0, 60.0, 90.0):
                    kw = dict(order=order, t_max=T, n_fine=N_FINE, extend_to_zero=True, eval_bands=dict(band=BAND))
                    if ainf_mode == "solver":
                        kw["A_inf"] = lin["A_inf"][i, j]
                    try:
                        f = fit_coefficient(w, At[:, i, j], Bt[:, i, j], **kw)
                        ce = case_err(f, lin, i, j)
                        eb = f["err_bands"]["band"]
                        row = dict(n_states=f["n_states"], err_K=f["err_K"], err_B=f["err_B"], err_A=f["err_A"],
                                   err_B_band=eb["err_B"], err_B_band_local=eb["err_B_local"], err_A_band=eb["err_A"],
                                   passivity=f["passivity"], max_real_eig=f["max_real_eig"], A_inf=f["A_inf"], A_inf_est=f["A_inf_est"],
                                   cases=ce)
                        say(f"  {nm:>5}{order:6d}{T:5.0f}{f['n_states']:4d}{fmt_pct(f['err_K']):>7}{fmt_pct(f['err_B']):>7}{fmt_pct(f['err_A']):>7}"
                            f"{fmt_pct(eb['err_B']):>8}{fmt_pct(eb['err_B_local']):>8}{fmt_pct(eb['err_A']):>8}"
                            f"{fmt_pct(ce['follow_Tp5']['rel_B']):>8}{fmt_pct(ce['follow_Tp5']['rel_A']):>8}"
                            f"{fmt_pct(ce['follow_Tp3']['rel_B']):>8}{fmt_pct(ce['follow_Tp3']['rel_A']):>8}{f['passivity']:8.3f}{f['max_real_eig']:8.3f}")
                    except Exception as e:      # noqa: BLE001
                        row = dict(error=repr(e))
                        say(f"  {nm:>5}{order:6d}{T:5.0f}  failed: {e!r}")
                    sweep[f"{ainf_mode}_{nm}_o{order}_T{T:g}"] = row
    say(f"  ({len(sweep)} fits, {time.time() - t0:.0f} s)")
    res["era_sweep"] = sweep
    res["era_time_s"] = time.time() - t0
    # the realisation used later (best in the band with solver A_inf): choose by band-local B error, heave-heave, pooled over orders at 60 s
    return sweep


# ================================================================== part 3: VF sweep
def part_vf(lin, res):
    from hydro.retardation import fit_coefficient, fit_block_fd
    say("\n== 3. frequency-domain path (vector fitting of B + i w (A - A_inf), solver A_inf, stable, passive) ==")
    say("   no time window in the fit; err_K is reported on 30 / 60 / 90 s through the same taper as the table kernel;")
    say("   'pole@1' is the fitted pole nearest 1 rad/s (decay rate 1/s, frequency rad/s): it is what carries the spike")
    t90 = np.arange(0.0, 90.0, 0.02)
    sweep = {}
    keep = {}
    t0 = time.time()
    for grid in ("uniform79", "all82"):
        sel = lin["uniform"] if grid == "uniform79" else np.ones(len(lin["omega"]), bool)
        w, At, Bt = tilde(lin, sel)
        for bw in (1.0, 4.0):
            say(f"\n  grid {grid}, band weight x{bw:g} on 0.8-1.0 rad/s:")
            say(f"  {'entry':>5}{'order':>6}{'n':>4}{'K30':>6}{'K60':>6}{'K90':>6}{'err_B':>7}{'err_A':>7}{'B band':>8}{'B bandL':>8}{'A band':>8}"
                f"{'B@.843':>8}{'A@.843':>8}{'B@.944':>8}{'A@.944':>8}{'passiv':>8}{'pas.pre':>8}{'it':>4}{'pole@1':>18}")
            for nm, (i, j) in ENTRIES.items():
                for order in (10, 16, 20, 24):
                    try:
                        # passivity (Re H >= 0) is a property of the diagonal entries and of the whole block; an
                        # off-diagonal damping changes sign with frequency and must not be constrained on its own
                        f = fit_coefficient(w, At[:, i, j], Bt[:, i, j], order=order, method="vf", A_inf=lin["A_inf"][i, j],
                                            t_max=90.0, n_fine=N_FINE, extend_to_zero=True, eval_bands=dict(band=BAND), band=BAND if bw != 1.0 else None,
                                            band_weight=bw, passive=(i == j))
                        ce = case_err(f, lin, i, j)
                        eb = f["err_bands"]["band"]
                        dK = np.abs(f["K_ss_tapered"] - f["K"]) / np.abs(f["K"]).max()
                        K30, K60, K90 = (float(dK[t90 < T].max()) for T in (30.0, 60.0, 90.0))
                        p1 = pole_near(f, 1.0)
                        row = dict(n_states=f["n_states"], err_K30=K30, err_K60=K60, err_K90=K90, err_B=f["err_B"], err_A=f["err_A"],
                                   err_B_all=f["err_B_all"], err_A_all=f["err_A_all"],
                                   err_B_band=eb["err_B"], err_B_band_local=eb["err_B_local"], err_A_band=eb["err_A"],
                                   passivity=f["passivity"], passivity_before=f["passive_info"]["min_before"],
                                   passive_enforced=f["passive_info"]["enforced"], max_real_eig=f["max_real_eig"],
                                   iterations=f["vf_iterations"], pole_change=f["vf_pole_change"],
                                   pole_near_1=[p1.real, p1.imag], cases=ce)
                        say(f"  {nm:>5}{order:6d}{f['n_states']:4d}{fmt_pct(K30):>6}{fmt_pct(K60):>6}{fmt_pct(K90):>6}{fmt_pct(f['err_B']):>7}{fmt_pct(f['err_A']):>7}"
                            f"{fmt_pct(eb['err_B']):>8}{fmt_pct(eb['err_B_local']):>8}{fmt_pct(eb['err_A']):>8}"
                            f"{fmt_pct(ce['follow_Tp5']['rel_B']):>8}{fmt_pct(ce['follow_Tp5']['rel_A']):>8}"
                            f"{fmt_pct(ce['follow_Tp3']['rel_B']):>8}{fmt_pct(ce['follow_Tp3']['rel_A']):>8}{f['passivity']:8.3f}"
                            f"{f['passive_info']['min_before']:8.3f}{f['vf_iterations']:4d}  {p1.real:+.3f} {p1.imag:+.3f}j")
                        keep[(grid, bw, nm, order)] = f
                    except Exception as e:      # noqa: BLE001
                        row = dict(error=repr(e))
                        say(f"  {nm:>5}{order:6d}  failed: {e!r}")
                    sweep[f"{grid}_bw{bw:g}_{nm}_o{order}"] = row
    res["vf_sweep"] = sweep

    # A_inf identified as the linear term (consistency of the table with the solver's A_inf)
    say("\n  A_inf identified as the linear term of B + i w A (all 82 points, band weight 4) vs the solver:")
    w, At, Bt = tilde(lin)
    ident = {}
    for nm, (i, j) in ENTRIES.items():
        vals = {}
        for order in (16, 24):
            f = fit_coefficient(w, At[:, i, j], Bt[:, i, j], order=order, method="vf", A_inf=None, t_max=90.0, n_fine=N_FINE, extend_to_zero=True, band=BAND, band_weight=4.0)
            vals[order] = dict(A_inf_fit=f["A_inf"], err_B=f["err_B"], err_A=f["err_A"])
        ident[nm] = vals
        say(f"    {nm}: order 16 -> {vals[16]['A_inf_fit']:8.0f} (err_B {100 * vals[16]['err_B']:.1f} %), order 24 -> {vals[24]['A_inf_fit']:8.0f}; solver {lin['A_inf'][i, j]:8.0f}")
    res["vf_A_inf_identified"] = ident

    # block fit with common poles
    say("\n  heave-pitch block with COMMON poles (all 82 points, band weight 4, passivity of the 2x2 Hermitian part enforced):")
    say(f"  {'order':>6}{'states':>7}{'err_B':>7}{'err_A':>7}  per entry B / A (0.2-5.2):  hh      hp      ph      pp   | band B (local): hh hp ph pp | B@.944 hh pp | passiv pre/post | it")
    blocks = {}
    for order in (10, 16, 20, 24):
        tb = time.time()
        fb = fit_block_fd(w, At, Bt, lin["A_inf"], order, band=BAND, band_weight=4.0, eval_bands=dict(band=BAND))
        ce = block_case_err(fb, lin)
        en = fb["entries"]
        blocks[order] = dict(n_states=fb["n_states"], err_B=fb["err_B"], err_A=fb["err_A"],
                             entries={nm: dict(err_B=en[ij]["err_B"], err_A=en[ij]["err_A"], band=en[ij]["err_bands"].get("band")) for nm, ij in ENTRIES.items()},
                             passivity=fb["passivity"], passivity_before=fb["passive_info"]["min_before"], iterations=fb["vf_iterations"],
                             max_real_eig=fb["max_real_eig"], cases=ce, time_s=time.time() - tb, poles=[[p.real, p.imag] for p in fb["poles"]])
        say(f"  {order:6d}{fb['n_states']:7d}{fmt_pct(fb['err_B']):>7}{fmt_pct(fb['err_A']):>7}  "
            + "  ".join(f"{100 * en[ij]['err_B']:4.1f}/{100 * en[ij]['err_A']:4.1f}" for ij in ENTRIES.values())
            + "  | " + " ".join(f"{100 * en[ij]['err_bands']['band']['err_B_local']:5.1f}" for ij in ENTRIES.values())
            + f" | {100 * ce['follow_Tp3']['hh']['rel_B']:5.1f} {100 * ce['follow_Tp3']['pp']['rel_B']:5.1f}"
            + f" | {fb['passive_info']['min_before']:+.3f}/{fb['passivity']:+.3f} | {fb['vf_iterations']} ({time.time() - tb:.0f} s)")
        keep[("block", order)] = fb
    res["vf_block"] = blocks
    # where the largest errors sit (order 16, all 82 points, band weight 4; scalar and block)
    say("\n  location of the largest errors over 0.2-5.2 rad/s (order 16, 82 points, band x4): entry, w of max |dB| and dB/max|B|, w of max |dA| and dA/max|A-A_inf|")
    ev = (w >= 0.2) & (w <= 5.2)
    loc = {}
    fb16 = keep[("block", 16)]
    for nm, (i, j) in ENTRIES.items():
        f = keep.get(("all82", 4.0, nm, 16))
        for lab, Bf, Af in (("scalar", f["B_fit"], f["A_fit"]), ("block ", fb16["B_fit"][:, i, j], fb16["A_fit"][:, i, j])):
            dB = np.abs(Bf - Bt[:, i, j]) * ev
            dA = np.abs(Af - At[:, i, j]) * ev
            kB, kA = int(np.argmax(dB)), int(np.argmax(dA))
            loc[f"{nm}_{lab.strip()}"] = dict(w_B=float(w[kB]), dB=float(dB[kB] / np.abs(Bt[:, i, j]).max()), w_A=float(w[kA]), dA=float(dA[kA] / np.abs(At[:, i, j] - lin["A_inf"][i, j]).max()),
                                           B_table=float(Bt[kB, i, j]), B_fit=float(Bf[kB]), A_table=float(At[kA, i, j]), A_fit=float(Af[kA]))
            say(f"    {nm} {lab}: dB max at w = {w[kB]:.3f} ({100 * dB[kB] / np.abs(Bt[:, i, j]).max():.1f} %: table {Bt[kB, i, j]:.0f}, fit {Bf[kB]:.0f}); "
                f"dA max at w = {w[kA]:.3f} ({100 * dA[kA] / np.abs(At[:, i, j] - lin['A_inf'][i, j]).max():.1f} %: table {At[kA, i, j]:.0f}, fit {Af[kA]:.0f})")
    res["vf_error_location"] = loc
    res["vf_time_s"] = time.time() - t0
    say(f"  (frequency-domain part: {time.time() - t0:.0f} s)")
    return keep


# ================================================================== part 4: zero speed
def part_rest(res):
    from hydro.retardation import fit_coefficient
    say("\n== 4. zero-speed table (Capytaine, 24 frequencies 0.5-20 rad/s): B1.12 quality through both paths ==")
    bem = load_rest()
    om, A, B = bem["omega"], bem["A"], bem["B"]
    t_ref = np.arange(0.0, 30.0, 0.02)
    out = {}
    say(f"  {'entry':>5}{'path':>32}{'order':>6}{'n':>4}{'err_K':>7}{'err_B':>7}{'err_A':>7}{'A_inf':>8}{'passiv':>8}{'slow p':>8}")
    for key, (i, j) in dict(hh=(2, 2), pp=(4, 4), hp=(2, 4)).items():
        Aij = 0.5 * (A[:, i, j] + A[:, j, i])
        Bij = 0.5 * (B[:, i, j] + B[:, j, i])
        f0 = fit_coefficient(om, Aij, Bij)                                   # the V1 / B1.12 call, untouched
        f0x = fit_coefficient(om, Aij, Bij, eval_bands={})                   # same with diagnostics
        rows = {"era auto (B1.12/V1)": f0x}
        # the frequency-domain fit uses only the range the time-domain path evaluates on (below the taper start,
        # 13 rad/s): the table's upper end (a bad point near 16.6 rad/s) is discarded by the taper there, and a joint
        # fit of w (A - A_inf) is dominated by it otherwise (err_B 12-16 % with all 24 points, see s3_progress_zh.md)
        sel = om <= 0.65 * om.max()
        for order in (6, 10):
            rows[f"vf <=13 rad/s, era A_inf, {order}"] = fit_coefficient(om[sel], Aij[sel], Bij[sel], order=order, method="vf", A_inf=f0["A_inf"],
                                                                        t_max=30.0, n_fine=N_FINE, extend_to_zero=True, passive=(i == j))
        rows["vf <=13, A_inf identified, 10"] = fit_coefficient(om[sel], Aij[sel], Bij[sel], order=10, method="vf", A_inf=None, t_max=30.0,
                                                                n_fine=N_FINE, extend_to_zero=True, passive=(i == j))
        rows["vf all 24 pts, era A_inf, 10"] = fit_coefficient(om, Aij, Bij, order=10, method="vf", A_inf=f0["A_inf"], t_max=30.0, n_fine=N_FINE,
                                                               extend_to_zero=True, passive=(i == j))
        rows["era auto, max_order 16"] = fit_coefficient(om, Aij, Bij, max_order=16)
        for lab, f in rows.items():
            out[f"{key}:{lab}"] = dict(order=int(f["order"]), n_states=int(f["Ar"].shape[0]), err_K=f["err_K"], err_B=f["err_B"], err_A=f["err_A"],
                                       A_inf=f["A_inf"], passivity=f.get("passivity"), max_real_eig=f["max_real_eig"])
            say(f"  {key:>5}{lab:>32}{f['order']:6d}{f['Ar'].shape[0]:4d}{fmt_pct(f['err_K']):>7}{fmt_pct(f['err_B']):>7}{fmt_pct(f['err_A']):>7}{f['A_inf']:8.0f}"
                f"{(f.get('passivity') if f.get('passivity') is not None else float('nan')):8.3f}{f['max_real_eig']:8.3f}")
        assert np.array_equal(f0["Ar"], f0x["Ar"]) and f0["err_B"] == f0x["err_B"], "diagnostic opt-in changed the result"
    say("  (B1.12 reference: heave 0.009 / 0.015 / 0.134, pitch 0.013 / 0.133 / 0.144, heave-pitch 0.087 / 0.135 / 0.195 for err_K / err_B / err_A;")
    say("   the vf err_K is through the taper, the era err_K is the raw impulse response, so the two K columns are not the same measure)")
    res["rest"] = out


# ================================================================== part 5: time-domain check
def rk4_memory(Ar, Br, Cr, nu_fun, t):
    """x' = Ar x + Br nu(t), mu = Cr x, classical RK4 on the grid t (Br, Cr for one input / one output)."""
    n = Ar.shape[0]
    x = np.zeros(n)
    mu = np.empty(len(t))
    mu[0] = 0.0
    for k in range(1, len(t)):
        h = t[k] - t[k - 1]
        t0 = t[k - 1]
        k1 = Ar @ x + Br * nu_fun(t0)
        k2 = Ar @ (x + 0.5 * h * k1) + Br * nu_fun(t0 + 0.5 * h)
        k3 = Ar @ (x + 0.5 * h * k2) + Br * nu_fun(t0 + 0.5 * h)
        k4 = Ar @ (x + h * k3) + Br * nu_fun(t0 + h)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        mu[k] = Cr @ x
    return mu


def fit_cos_sin(t, y, w, t_from):
    m = t >= t_from
    M = np.stack([np.cos(w * t[m]), np.sin(w * t[m])], 1)
    a, b = np.linalg.lstsq(M, y[m], rcond=None)[0]
    return float(a), float(b)


def part_td(lin, res, keep, era_pick):
    from hydro.retardation import retardation_function, fit_coefficient
    from scipy.interpolate import PchipInterpolator
    say("\n== 5. time-domain check: velocity cos(w t), heave force by (a) convolution with the table kernel (90 s window),")
    say("   (b) the fitted state spaces (RK4, dt 0.005 s), (c) the frequency-domain table value. Steady state fitted on 200-260 s. ==")
    say("   mu = a cos(w t) + b sin(w t): a is the damping part (B - B_inf), -b/w is the added-mass part (A + C0/w^2 - A_inf).")
    w_all, At, Bt = tilde(lin)
    i = j = 0
    A_inf = lin["A_inf"][0, 0]
    dt = 0.005
    t = np.arange(0.0, 260.0 + 1e-9, dt)
    tk = np.arange(0.0, 90.0, dt)
    K = retardation_function(w_all, Bt[:, i, j], tk, n_fine=N_FINE, extend_to_zero=True)   # tapered 5.2-8, like all fits
    out = {}
    tg = case_targets(lin)
    for w0 in (0.94, 0.955):
        nu = np.cos(w0 * t)
        # convolution: mu(t) = int_0^t K(tau) nu(t - tau) dtau, trapezoid weights on K
        wts = np.full(len(K), dt)
        wts[0] = wts[-1] = 0.5 * dt
        mu_conv = np.convolve(K * wts, nu)[:len(t)]
        a, b = fit_cos_sin(t, mu_conv, w0, 200.0)
        rowd = {"convolution, 90 s kernel": dict(B=a, A=A_inf - b / w0)}
        for lab, f in era_pick.items():
            mu = rk4_memory(f["Ar"], f["Br"], f["Cr"], lambda s: np.cos(w0 * s), t)
            a, b = fit_cos_sin(t, mu, w0, 200.0)
            rowd[lab] = dict(B=a, A=A_inf - b / w0)
        for lab, f in keep.items():
            mu = rk4_memory(f["Ar"], f["Br"], f["Cr"], lambda s: np.cos(w0 * s), t)
            a, b = fit_cos_sin(t, mu, w0, 200.0)
            rowd[lab] = dict(B=a, A=A_inf - b / w0)
        # references
        Bp = float(PchipInterpolator(w_all, Bt[:, i, j])(w0))
        Ap = float(PchipInterpolator(w_all, At[:, i, j])(w0))
        refs = {"table, PCHIP at w": dict(B=Bp, A=Ap)}
        k = int(np.argmin(np.abs(w_all - w0)))
        refs[f"table, nearest grid point {w_all[k]:.3f}"] = dict(B=float(Bt[k, i, j]), A=float(At[k, i, j]))
        if abs(w0 - 0.94) < 1e-9:
            refs["solver at w_e 0.944 (follow Tp3)"] = dict(B=float(tg["follow_Tp3"]["Bt"][i, j]), A=float(tg["follow_Tp3"]["At"][i, j]))
        ref = refs["table, PCHIP at w"]
        say(f"\n  w = {w0} rad/s, heave-heave; reference (PCHIP of the table at w): B - B_inf = {ref['B']:.0f} N s/m, A + C0/w^2 = {ref['A']:.0f} kg")
        for lab, r in refs.items():
            say(f"    {lab:44s}: B {r['B']:8.0f}  A {r['A']:8.0f}")
        say(f"    {'time-domain model':44s}: {'B':>8}  {'A':>8}  {'dB/B %':>7} {'dA/(A-Ainf) %':>14} {'|H| ratio':>10} {'phase deg':>10}")
        Href = ref["B"] + 1j * w0 * (ref["A"] - A_inf)
        for lab, r in rowd.items():
            Hr = r["B"] + 1j * w0 * (r["A"] - A_inf)
            r.update(rel_B=(r["B"] - ref["B"]) / ref["B"], rel_A=(r["A"] - ref["A"]) / (ref["A"] - A_inf),
                     amp_ratio=abs(Hr) / abs(Href), phase_deg=math.degrees(np.angle(Hr / Href)))
            say(f"    {lab:44s}: {r['B']:8.0f}  {r['A']:8.0f}  {100 * r['rel_B']:7.1f} {100 * r['rel_A']:14.1f} {r['amp_ratio']:10.3f} {r['phase_deg']:10.2f}")
        out[str(w0)] = dict(refs=refs, models=rowd)
    res["time_domain"] = out
    return out


# ================================================================== part 6: online cost
def part_cost(res, states_hp_scalar, states_hp_block):
    say("\n== 6. online cost of the memory state space (this machine, numpy, one RK4 step = 4 derivative evaluations) ==")
    import timeit
    rows = {}
    for n, m in ((40, 2), (64, 2), (96, 2), (48, 2), (32, 2), (144, 6), (288, 6), (432, 6)):
        rng = np.random.default_rng(0)
        Ar = np.zeros((n, n))
        for k in range(0, n - 1, 2):
            Ar[k:k + 2, k:k + 2] = [[-0.1, 1.0], [-1.0, -0.1]]
        Br = rng.normal(size=(n, m))
        Cr = rng.normal(size=(m, n))
        x = np.zeros(n)
        nu = np.ones(m)
        h = 0.005

        def rk4_dense():
            k1 = Ar @ x + Br @ nu
            k2 = Ar @ (x + 0.5 * h * k1) + Br @ nu
            k3 = Ar @ (x + 0.5 * h * k2) + Br @ nu
            k4 = Ar @ (x + h * k3) + Br @ nu
            return Cr @ (x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))

        a = np.diag(Ar)[0::2].copy()
        wv = Ar[0::2, 1::2].diagonal().copy()

        def deriv_block(xx):
            x0, x1 = xx[0::2], xx[1::2]
            d = np.empty_like(xx)
            d[0::2] = a * x0 + wv * x1
            d[1::2] = -wv * x0 + a * x1
            return d + Br @ nu

        def rk4_block():
            k1 = deriv_block(x)
            k2 = deriv_block(x + 0.5 * h * k1)
            k3 = deriv_block(x + 0.5 * h * k2)
            k4 = deriv_block(x + h * k3)
            return Cr @ (x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))

        nrep = 2000
        td = min(timeit.repeat(rk4_dense, number=nrep, repeat=5)) / nrep * 1e3
        tb = min(timeit.repeat(rk4_block, number=nrep, repeat=5)) / nrep * 1e3
        flops_dense = 4 * (n * n + n * m) + m * n
        flops_block = 4 * (2 * n + n * m) + m * n
        rows[f"n{n}_m{m}"] = dict(n_states=n, n_dof=m, ms_dense=td, ms_block=tb, madds_dense=flops_dense, madds_block=flops_block)
        say(f"  {n:4d} states, {m} DOF: dense matrices {td:.4f} ms / RK4 step ({flops_dense} multiply-adds), modal 2x2 blocks {tb:.4f} ms ({flops_block} multiply-adds)")
    say(f"  heave-pitch at 5 kn: separate scalar fits -> {states_hp_scalar} states; common-pole block -> {states_hp_block} states.")
    say("  one speed node is active at a time (the realisation is picked or blended per node), so the state count does not multiply with the")
    say("  number of speed nodes; blending two neighbouring nodes doubles it. 6 DOF with port-starboard symmetry: two 3x3 groups, so")
    say("  3 x (states per group pole set) each, i.e. 6 x order states with common poles (96 at order 16) against 18 x order with scalar fits.")
    res["cost"] = rows


# ================================================================== plots
def part_plots(lin, res, keep, era_pick, diag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hydro.retardation import transfer_function, retardation_function
    w_all, At, Bt = tilde(lin)
    wd = np.linspace(0.2, 8.0, 2000)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for col, (nm, (i, j)) in enumerate((("hh", (0, 0)), ("pp", (1, 1)))):
        ax[0, col].plot(w_all, Bt[:, i, j], "k.", ms=4, label="table")
        ax[1, col].plot(w_all, At[:, i, j], "k.", ms=4, label="table")
        curves = (list(era_pick.items()) if nm == "hh" else []) + [(f"vf order {o}", keep[("all82", 4.0, nm, o)]) for o in (10, 16, 24) if ("all82", 4.0, nm, o) in keep]
        if ("block", 16) in keep:
            curves.append(("vf block order 16 (common poles)", None))
        for lab, f in curves:
            if f is None:
                from hydro.retardation import block_transfer_function
                fb = keep[("block", 16)]
                Hb = block_transfer_function(wd, fb["Ar"], fb["Br"], fb["Cr"])[:, i, j]
                ax[0, col].plot(wd, np.real(Hb), "k--", lw=0.8, label=lab)
                ax[1, col].plot(wd, lin["A_inf"][i, j] + np.imag(Hb) / wd, "k--", lw=0.8)
                continue
            H = transfer_function(wd, f["Ar"], f["Br"], f["Cr"])
            ax[0, col].plot(wd, np.real(H), lw=1, label=lab)
            ax[1, col].plot(wd, lin["A_inf"][i, j] + np.imag(H) / wd, lw=1)
        ax[0, col].set_title(f"{nm}: B - B_inf at 5 kn", fontsize=9)
        ax[1, col].set_title(f"{nm}: A + C0/w^2", fontsize=9)
        for r in range(2):
            ax[r, col].set_xlim(0.2, 3.0)
            ax[r, col].axvspan(0.8, 1.0, color="0.9")
            ax[r, col].set_xlabel("encounter frequency, rad/s")
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "s3_fits.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    t = diag["t"]
    ax[0].plot(t, diag["K_plain"], lw=0.8, label="plain trapezoid (aliases at 62.8 s)")
    ax[0].plot(t, diag["K_fine"], lw=0.8, label="fine grid")
    ax[0].set_ylim(-3000, 3000)
    ax[0].set_title("heave kernel at 5 kn, tapered 5.2-8 rad/s", fontsize=9)
    ax[0].legend(fontsize=7)
    ax[1].semilogy(t, np.abs(diag["K_fine"]) + 1e-3, "k", lw=0.8, label="table (fine grid)")
    for lab, f in era_pick.items():
        ax[1].semilogy(f["t"], np.abs(f["K_ss"]) + 1e-3, lw=0.8, label=lab)
    for o in (16, 24):
        if ("all82", 4.0, "hh", o) in keep:
            f = keep[("all82", 4.0, "hh", o)]
            ax[1].semilogy(f["t"], np.abs(f["K_ss_tapered"]) + 1e-3, lw=0.8, label=f"vf order {o} (through taper)")
    ax[1].set_ylim(1, 1e5)
    ax[1].set_xlabel("t, s")
    ax[1].set_title("|K(t)| heave-heave", fontsize=9)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "s3_kernel.png"), dpi=130)
    plt.close(fig)

    # error vs order
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    orders = [10, 16, 20, 24]
    for nm in ("hh", "pp"):
        for T in (30, 60, 90):
            ys = [res["era_sweep"].get(f"solver_{nm}_o{o}_T{T}", {}).get("err_B_band_local", np.nan) for o in orders]
            ax[0].plot(orders, 100 * np.array(ys), "o-" if nm == "hh" else "s--", label=f"ERA {nm} {T} s")
        ys = [res["vf_sweep"].get(f"all82_bw4_{nm}_o{o}", {}).get("err_B_band_local", np.nan) for o in orders]
        ax[1].plot(orders, 100 * np.array(ys), "o-" if nm == "hh" else "s--", label=f"VF {nm} (82 pts, band x4)")
        ys = [res["vf_block"].get(o, {}).get("entries", {}).get(nm, {}).get("band", {}).get("err_B_local", np.nan) for o in orders]
        ax[1].plot(orders, 100 * np.array(ys), "^:", label=f"VF block {nm}")
    for a in ax:
        a.axhline(5, color="r", lw=0.8)
        a.set_xlabel("order")
        a.set_ylabel("B error in 0.8-1.0 rad/s, % of in-band max")
        a.set_ylim(0, 60)
        a.legend(fontsize=7)
    ax[0].set_title("time-domain path (solver A_inf, fine grid)", fontsize=9)
    ax[1].set_title("frequency-domain path", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "s3_errors.png"), dpi=130)
    plt.close(fig)
    say("  figures: s3_fits.png, s3_kernel.png, s3_errors.png")


# ================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="diag,era,vf,rest,td,cost,plots")
    a = ap.parse_args()
    parts = a.parts.split(",")
    T0 = time.time()
    res_path = os.path.join(OUT, "s3_results.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) and a.parts != "diag,era,vf,rest,td,cost,plots" else {}
    lin = load_5kn()
    say(f"S3 memory fit study -- table {lin['label']}, U = {lin['u']:.3f} m/s, critical frequency g/(4U) = {G / (4 * lin['u']):.3f} rad/s, "
        f"{len(lin['omega'])} frequencies ({int(lin['uniform'].sum())} on the uniform 0.1 grid)")
    say(f"  solver A_inf heave {lin['A_inf'][0, 0]:.0f} kg, pitch {lin['A_inf'][1, 1]:.0f} kg m2, cross {lin['A_inf'][0, 1]:.0f} / {lin['A_inf'][1, 0]:.0f}")

    diag = part_diag(lin, res) if "diag" in parts else None
    if "era" in parts:
        part_era(lin, res)
    keep = part_vf(lin, res) if "vf" in parts else {}
    if "rest" in parts:
        part_rest(res)
    # realisations for the time-domain check and the plots
    from hydro.retardation import fit_coefficient
    w, At, Bt = tilde(lin, lin["uniform"])
    era_pick = {
        "era order 10, 30 s, A_inf est (original)": fit_coefficient(w, At[:, 0, 0], Bt[:, 0, 0], order=10, eval_bands={}),
        "era order 16, 60 s, solver A_inf, fine": fit_coefficient(w, At[:, 0, 0], Bt[:, 0, 0], order=16, t_max=60.0, n_fine=N_FINE, extend_to_zero=True, A_inf=lin["A_inf"][0, 0]),
        "era order 24, 90 s, solver A_inf, fine": fit_coefficient(w, At[:, 0, 0], Bt[:, 0, 0], order=24, t_max=90.0, n_fine=N_FINE, extend_to_zero=True, A_inf=lin["A_inf"][0, 0]),
    }
    vf_pick = {}
    for o in (16, 24):
        if ("all82", 4.0, "hh", o) in keep:
            vf_pick[f"vf order {o}, 82 pts, band x4, passive"] = keep[("all82", 4.0, "hh", o)]
    if "td" in parts:
        part_td(lin, res, vf_pick, era_pick)
    if "cost" in parts:
        n_scalar = sum(res.get("vf_sweep", {}).get(f"all82_bw4_{nm}_o16", {}).get("n_states", 16) for nm in ENTRIES)
        n_block = res.get("vf_block", {}).get(16, res.get("vf_block", {}).get("16", {})).get("n_states", 32)
        part_cost(res, n_scalar, n_block)
    if "plots" in parts and diag is not None and keep:
        part_plots(lin, res, keep, era_pick, diag)
    res["total_time_s"] = time.time() - T0
    say(f"\ntotal {time.time() - T0:.0f} s")

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o
    json.dump(clean(res), open(res_path, "w"), indent=1)
    open(os.path.join(OUT, "s3_tables.txt"), "w", encoding="utf-8").write("\n".join(_LINES) + "\n")


if __name__ == "__main__":
    main()
