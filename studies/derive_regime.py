#!/usr/bin/env python3
"""Regime map of the scarab195 planing USV over its envelope
(learn/meta/HULL_FORCE_DERIVATION_2026-10-09.md, section 2).

Reads the plant's own calm equilibria (ZarnickVessel, ZR configuration) and
Savitsky's (hydro/planing.savitsky) and prints the nondimensional groups the
derivation uses. No plant code is changed. Output:
studies/_cache/derive/regime.json and regime.txt.
"""
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "studies", "_cache", "derive")
os.makedirs(OUT, exist_ok=True)

G, RHO, NU = 9.81, 1025.0, 1.19e-6
KN = 0.514444


def plant():
    from sim import config
    from sim.planing_zarnick import ZarnickVessel, ZOPT_ZR
    from sim.test_vessel import Monochromatic
    h, _ = config.load("scarab195")
    return h, ZarnickVessel(h, Monochromatic(1.0, 0.0), dt=0.02,
                            zopt=dict(ZOPT_ZR))


def calm_rows(h, p):
    from hydro.planing import savitsky
    b, L = h.b, h.L
    vol = h.mass / RHO
    tanb = math.tan(math.radians(h.beta_deg))
    rows = []
    for kn in [0.0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 25, 30, 35, 40]:
        U = kn * KN
        if U > 0:
            r = p.calm_resistance(U)
            z, th = p.running_attitude(U)
        else:
            r = dict(trim_deg=float(-np.degrees(p.Th_rest)), R=0.0)
            z, th = 0.0, 0.0
        s = np.zeros(14)
        s[6], s[2], s[4] = U, z, th
        p.captive_u = U
        _, aux = p.deriv(s, 0.0)
        p.captive_u = None
        d = aux["d"]
        wet = d > 0
        cw = 0.5 * np.pi * np.maximum(d, 0) / tanb
        chine_wet = wet & (cw >= p.yc)
        dx = p.dx
        LK = wet.sum() * dx
        LC = chine_wet.sum() * dx
        lam = 0.5 * (LK + LC) / b
        trim = r["trim_deg"]
        tr = math.radians(max(trim, 1e-3))
        Fn_L = U / math.sqrt(G * L)
        Cv = U / math.sqrt(G * b)
        Fn_vol = U / math.sqrt(G * vol ** (1 / 3))
        # cross-plane Froude number of a chine-dry section in steady planing
        Vn = U * math.tan(tr)
        Fn_c = Vn / math.sqrt(G * 0.5 * b)
        # local, per wetted station: V / sqrt(g c)
        V = aux["V"]
        c = np.minimum(cw, p.yc)
        with np.errstate(divide="ignore", invalid="ignore"):
            fc = np.where(wet & (c > 1e-3), np.abs(V) / np.sqrt(G * c), np.nan)
        Fn_Lw = U / math.sqrt(G * max(LK, 1e-6))
        sv = savitsky(U, h.mass, h.lcg, b, h.beta_deg) if kn >= 8 else None
        row = dict(kn=kn, U=U, Fn_L=Fn_L, Cv=Cv, Fn_vol=Fn_vol,
                   trim_ZR=trim, rise_ZR=float(z), R_ZR=float(r.get("R", 0)),
                   LK=LK, LC=LC, LK_minus_LC=LK - LC, lam_ZR=lam,
                   eps_bLK=b / max(LK, 1e-6),
                   chine_dry_share=(LK - LC) / max(LK, 1e-6),
                   Vn=Vn, Fn_c=Fn_c,
                   Fn_c_station_max=float(np.nanmax(fc)) if np.isfinite(
                       fc).any() else None,
                   Fn_Lw=Fn_Lw, kelvin_over_LK=2 * math.pi * U ** 2 / G
                   / max(LK, 1e-6),
                   Re=U * LK / NU, T_pass=LK / U if U > 0 else None)
        if sv is not None:
            row.update(trim_sav=sv["tau_deg"], lam_sav=sv["lam"],
                       R_sav=sv["R"], sav_valid=bool(sv["valid"]),
                       # Savitsky's hydrostatic/dynamic lift-term ratio
                       sav_static_over_dyn=0.458 * sv["lam"] ** 2 / Cv ** 2,
                       # Savitsky chine-dry length (L_K - L_C) = b tanb/(pi tan tau)
                       sav_LK_minus_LC=b * tanb / (math.pi * math.tan(
                           math.radians(sv["tau_deg"]))),
                       Fn_c_sav=U * math.tan(math.radians(sv["tau_deg"]))
                       / math.sqrt(G * 0.5 * b))
        rows.append(row)
    return rows


def wave_rows(LK_of_kn, b):
    """Encounter groups for Tp 3-10 s at 0-40 kn, head/beam/following."""
    out = []
    for kn in [0, 5, 10, 20, 30, 40]:
        U = kn * KN
        LK = LK_of_kn(kn)
        for Tp in [3.0, 5.0, 8.0, 10.0]:
            w = 2 * math.pi / Tp
            k = w * w / G
            lw = 2 * math.pi / k
            # steepest sea of Hs 1.5 m physically allowed: Hs/lam_p <= 1/15
            hs = min(1.5, lw / 15.0)
            for head, mu in [("head", 0.0), ("beam", 90.0),
                             ("following", 180.0)]:
                we = w + k * U * math.cos(math.radians(mu))
                out.append(dict(kn=kn, Tp=Tp, heading=head, omega=w,
                                omega_e=we, tau_brard=U * w / G,
                                kb=k * b, kLK=k * LK, lam_over_LK=lw / LK,
                                hs_used=hs, steep_kA=k * hs / 2,
                                xplane_freq=we * we * 0.5 * b / G,
                                Te=2 * math.pi / abs(we) if abs(we) > 1e-6
                                else None))
    return out


def impact_rows(b, beta_deg):
    """Section-entry groups. V = relative normal entry velocity, c = half
    beam at the chine; Wagner chine-wetting time t_cw = (2/pi) c tanb / V."""
    tanb = math.tan(math.radians(beta_deg))
    out = []
    for V in [0.5, 1.0, 2.0, 3.0, 5.0]:
        for c in [0.25, 0.5, 1.0]:
            t_cw = (2 / math.pi) * c * tanb / V
            out.append(dict(V=V, c=c, Fn_c=V / math.sqrt(G * c), t_cw=t_cw,
                            steps_per_entry_dt002=t_cw / 0.02))
    return out


def pulse_sampling(dt=0.02):
    """Peak lost by sampling a half-sine pulse of duration T on a grid of
    step dt (worst phase): 1 - cos(pi dt / (2 T))."""
    return [dict(T=T, worst_loss=1 - math.cos(math.pi * dt / (2 * T)),
                 mean_loss=1 - (2 * T / (math.pi * dt))
                 * math.sin(math.pi * dt / (2 * T)))
            for T in [0.03, 0.05, 0.08, 0.12, 0.2]]


def main():
    h, p = plant()
    rows = calm_rows(h, p)
    tab = {r["kn"]: r["LK"] for r in rows}

    def LK_of(kn):
        ks = sorted(tab)
        return float(np.interp(kn, ks, [tab[k] for k in ks]))
    waves = wave_rows(LK_of, h.b)
    imp = impact_rows(h.b, h.beta_deg)
    pul = pulse_sampling()
    res = dict(hull=dict(L=h.L, b=h.b, beta=h.beta_deg, mass=h.mass,
                         lcg=h.lcg, vol=h.mass / RHO, L_over_b=h.L / h.b,
                         T_rest=p.T),
               calm=rows, waves=waves, impact=imp, pulse=pul)
    with open(os.path.join(OUT, "regime.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    lines = [f"hull L {h.L} b {h.b:.3f} L/b {h.L/h.b:.2f} beta {h.beta_deg} "
             f"mass {h.mass} lcg {h.lcg:.3f} T_rest {p.T:.3f}"]
    lines.append("kn  FnL  Cv  trimZR trimSav  LK  LC  LK-LC(sav)  lamZR "
                 "lamSav  b/LK  Fn_c  Fn_c,max  FnLw  2piU2/g/LK  "
                 "stat/dyn(Sav)  Re  Tpass  R_ZR R_sav")
    for r in rows:
        lines.append(
            f"{r['kn']:4.0f} {r['Fn_L']:.2f} {r['Cv']:.2f} {r['trim_ZR']:5.2f} "
            f"{r.get('trim_sav', float('nan')):5.2f} {r['LK']:.2f} "
            f"{r['LC']:.2f} {r['LK_minus_LC']:.2f}"
            f"({r.get('sav_LK_minus_LC', float('nan')):.2f}) "
            f"{r['lam_ZR']:.2f} {r.get('lam_sav', float('nan')):.2f} "
            f"{r['eps_bLK']:.2f} {r['Fn_c']:.3f} "
            f"{(r['Fn_c_station_max'] or float('nan')):.3f} "
            f"{r['Fn_Lw']:.2f} {r['kelvin_over_LK']:.2f} "
            f"{r.get('sav_static_over_dyn', float('nan')):.2f} "
            f"{r['Re']:.2e} {(r['T_pass'] or float('nan')):.3f} "
            f"{r['R_ZR']:.0f} {r.get('R_sav', float('nan')):.0f}")
    lines.append("")
    lines.append("waves: kn Tp heading omega_e tau=U w/g kb kLK lam/LK kA "
                 "we^2(b/2)/g Te")
    for w in waves:
        lines.append(f"{w['kn']:3d} {w['Tp']:4.1f} {w['heading']:9s} "
                     f"{w['omega_e']:6.2f} {w['tau_brard']:5.2f} "
                     f"{w['kb']:.3f} {w['kLK']:.2f} {w['lam_over_LK']:5.1f} "
                     f"{w['steep_kA']:.3f} {w['xplane_freq']:6.2f} "
                     f"{(w['Te'] or float('nan')):5.2f}")
    lines.append("")
    lines.append("impact: V c Fn_c t_cw steps(dt 0.02)")
    for i in imp:
        lines.append(f"{i['V']:.1f} {i['c']:.2f} {i['Fn_c']:.2f} "
                     f"{i['t_cw']:.3f} {i['steps_per_entry_dt002']:.2f}")
    lines.append("")
    lines.append("half-sine pulse sampled at 0.02 s: T worst_loss mean_loss")
    for q in pul:
        lines.append(f"{q['T']:.2f} {q['worst_loss']:.3f} {q['mean_loss']:.3f}")
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "regime.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
