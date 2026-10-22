#!/usr/bin/env python3
"""
Check V2 of learn/meta/HULL_FORCE_AUDIT_REVIEW_2026-10-09.md section 4:
second-order "double-input" forces on the Scarab 195 hull at rest, zero speed,
in the frequency domain, with the zero-speed Rankine solver of
hydro/rankine/doublebody.py (helpers: studies/derive_v2_hull.py).

Conventions. Time factor e^{i w t}; a physical quantity is Re[X e^{i w t}].
Earth frame: x forward, z up, calm water z = 0; moments about the CG, pitch
rotation vector along +y (positive = bow DOWN, unsteady.mode_vectors). Head
wave: eta = Re[A e^{i(w t + k x)}], travelling toward -x (beta = pi).

Second-order amplitudes from a bilinear form q(a, b) of two first-order
inputs a (at w_a) and b (at w_b), each stored with all derivative fields:
    sum frequency w_a + w_b :  F+ = (q(a, b) + q(b, a)) / 2
    difference w_a - w_b    :  F- = (q(a, conj b) + q(conj b, a)) / 2   (mean = Re F- when w_a = w_b)
    one input with itself   :  half of the above (F(2w) = q(a, a)/2, mean = Re q(a, conj a)/2)
so the "double-input" combination F(a, A) - F(a, 0) - F(0, A) + F(0, 0) is a A times the
mixed F+/F- above.

What is computed:
1. First-order fields at w in OMEGAS for unit heave (1 m), unit pitch (1 rad), unit head wave (1 m):
   potentials on the body and the free surface, gradients (second-order structured differences on the
   hull mesh and on the free-surface band grid; vertical derivative from the free-surface conditions,
   phi_zz from Laplace), elevations, first-order forces; radiation/excitation coefficients.
2. Quadratic part of the second-order force on the mean wetted surface (near-field, Pinkster 1980):
   -rho/2 |grad phi|^2, -rho (d . grad) phi_t, the waterline term -rho g/2 zeta_rel^2 n dl / sqrt(1 - n_z^2)
   (flare factor), and alpha x F1 (rotated normal). The second-order rotation terms of pure pitch
   (products of two angles in the displacement) are NOT included: they are motion x motion only.
3. Second-order potential: free-surface forcing (8) of the audit through solve_freq's f_k, f_d, body
   forcing phi2_n = -n.(d.grad)grad phi1 + n.[alpha x (grad phi1 - v_translation)] with the Hessian term
   on the piecewise-flat hull written with tangential derivatives (-d_n lap_S phi + d_t . grad_S phi_n;
   the curvature concentrated on the keel/chine lines is neglected), solved at Omega = w_i + w_j and
   |w_i - w_j| (beach scaled with Omega), pressure -rho i Omega phi_2 on the mean surface.
   The incident x incident free-surface forcing is excluded (its exact solution is phi_2 = 0 in deep water).
4. Tables of the mixed terms relative to the first-order forces, convergence in the free-surface grid,
   compute time and memory.

Run: cd repo && .venv/Scripts/python.exe -m studies.derive_v2_second_order [--quick]
Outputs: studies/_cache/derive/v2_results.json, v2_tables.txt, v2_run.log, v2_mixed.png
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch

from hydro.rankine.unsteady import incident
from studies.derive_v2_hull import (scarab_form, build_zero_speed, mode_normals, body_gradient, fs_metric,
                                    fs_hgrad, incident_fields, waterline_data, waterline_values, mem_report, G0)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_cache", "derive")
OMEGAS = (3.0, 3.9, 4.7)
INPUTS = ("heave", "pitch", "wave")
AMPS_SMALL = dict(heave=0.01, pitch=0.01, wave=0.01)        # m, rad, m
AMPS_OPER = dict(heave=0.10, pitch=0.05, wave=0.30)
NU_REF = 3.9                                                 # the beach nu of the validated first-order case


# ============================================================ first order
def first_order(st, omega, mn, wl, met, beta=np.pi):
    """One LU at omega, three right-hand sides. Returns dict input -> fields (unit amplitude)."""
    s, G, body = st["s"], st["G"], st["body"]
    rho, g = s.rho, s.g
    a = body.area
    nB, nF = s.NB, s.NF
    n3, n5 = mn["n3"], mn["n5"]
    phI_B, dnI_B, _ = incident(body.c, body.n, omega, beta, g)
    phin = np.stack([1j * omega * n3, 1j * omega * n5, -dnI_B], 1)
    t0 = time.time()
    sol = s.solve_freq(omega, phin)
    t_solve = time.time() - t0
    Pf = G["fs"].c
    nu = s.nu
    incF = incident_fields(Pf, omega, beta, g)
    incB = incident_fields(body.c, omega, beta, g)
    Pw = np.stack([wl["x_mid"], wl["y_mid"], np.zeros_like(wl["x_mid"])], 1)
    incW = incident_fields(Pw, omega, beta, g)
    out = {}
    for c, name in enumerate(INPUTS):
        f = dict(name=name, omega=omega)
        phiB = sol["phi_B"][:, c].copy()
        pn = phin[:, c].copy()
        phiF = sol["phi_F"][:, c].copy()
        zeta = -(1j * omega + nu) * phiF / g                       # dynamic condition (zeta_of at U = 0)
        phiz = (1j * omega + nu) * zeta                            # kinematic condition
        fx, fy = fs_hgrad(G, met, phiF)
        fxx, _ = fs_hgrad(G, met, fx)
        _, fyy = fs_hgrad(G, met, fy)
        phizz = -(fxx + fyy)
        zx, zy = fs_hgrad(G, met, zeta)
        if name == "wave":
            gradB = body_gradient(st, phiB, pn) + incB["grad"]
            phiB = phiB + incB["phi"]
            pn = np.zeros(nB, complex)                             # total normal velocity on the fixed body
            k, cb = incF["k"], np.cos(beta)
            zI = incF["zeta"]
            f["inc"] = dict(phiF=incF["phi"], zeta=zI, phix=incF["grad"][:, 0], phiy=incF["grad"][:, 1],
                            phiz=incF["grad"][:, 2], phitz=1j * omega * incF["grad"][:, 2], phizz=incF["phizz"],
                            zx=(-1j * k * cb) * zI, zy=np.zeros(nF, complex), omega=omega)
            phiF = phiF + incF["phi"]
            zeta = zeta + zI
            fx = fx + incF["grad"][:, 0]
            fy = fy + incF["grad"][:, 1]
            phiz = phiz + incF["grad"][:, 2]
            phizz = phizz + incF["phizz"]
            zx = zx + (-1j * k * cb) * zI
            d = np.zeros((nB, 3), complex)
            alpha = np.zeros(3)
            vtr = np.zeros(3, complex)
            dz_wl = np.zeros(len(Pw), complex)
        else:
            gradB = body_gradient(st, phiB, pn)
            if name == "heave":
                d = mn["d3"].astype(complex)
                alpha = np.zeros(3)
                vtr = np.array([0.0, 0.0, 1j * omega])             # translation velocity of the CG
                dz_wl = np.ones(len(Pw), complex)
            else:
                d = mn["d5"].astype(complex)
                alpha = np.array([0.0, 1.0, 0.0])
                vtr = np.zeros(3, complex)
                dz_wl = -(Pw[:, 0] - mn["xg"]) * np.ones(len(Pw), complex)
        # tangential pieces for the second-order body condition
        gt = gradB - body.n * np.einsum("ij,ij->i", gradB, body.n)[:, None]
        lapS = np.zeros(nB, complex)
        for cc in range(3):
            lapS += body_gradient(st, gt[:, cc], np.zeros(nB))[:, cc]
        gS_pn = body_gradient(st, pn, np.zeros(nB))
        # first-order force (hydrodynamic + hydrostatic variation at the displaced point), moment about the CG
        p1 = -rho * (1j * omega * phiB + g * d[:, 2])
        fdens = -p1[:, None] * body.n
        F1 = 2 * np.sum(fdens * a[:, None], 0)
        M1 = 2 * np.sum(np.cross(mn["r"], fdens) * a[:, None], 0)
        # waterline: total first-order elevation at the hull waterline (body potential extrapolated to z = 0)
        zeta_wl = -1j * omega * waterline_values(st, phiB) / g
        f.update(phiB=phiB, pn=pn, gradB=gradB, gradtB=1j * omega * gradB, lapS=lapS, gS_pn=gS_pn,
                 d=d, alpha=alpha, vtr=vtr, F1=F1, M1=M1,
                 phiF=phiF, zeta=zeta, phix=fx, phiy=fy, phiz=phiz, phitz=1j * omega * phiz, phizz=phizz,
                 zx=zx, zy=zy, zeta_wl=zeta_wl, dz_wl=dz_wl, zrel_wl=zeta_wl - dz_wl, t_solve=t_solve)
        if name == "wave":
            # incident-only body quantities (the "strip-like" part of the mixed term: what a strip model
            # with the undisturbed incident wave sees)
            p1i = -rho * 1j * omega * incB["phi"]
            fdi = -p1i[:, None] * body.n
            f["inc_only"] = dict(f, gradB=incB["grad"], gradtB=1j * omega * incB["grad"], zrel_wl=incW["zeta"],
                                 F1=2 * np.sum(fdi * a[:, None], 0), M1=2 * np.sum(np.cross(mn["r"], fdi) * a[:, None], 0))
            f["hf"] = f["inc_only"]
        else:
            # "F_HF" variant of a motion input: hydrostatic pressure only (no radiation potential, no radiated
            # wave at the waterline): the second-order part of the exact hydrostatics of the moving body
            p1h = -rho * g * d[:, 2]
            fdh = -p1h[:, None] * body.n
            f["hf"] = dict(f, gradB=np.zeros((nB, 3), complex), gradtB=np.zeros((nB, 3), complex), zrel_wl=-dz_wl,
                           F1=2 * np.sum(fdh * a[:, None], 0), M1=2 * np.sum(np.cross(mn["r"], fdh) * a[:, None], 0))
        out[name] = f
    return out


def conj_fields(f):
    out = {}
    for k, v in f.items():
        if isinstance(v, dict):
            out[k] = conj_fields(v)
        elif isinstance(v, np.ndarray) and np.iscomplexobj(v):
            out[k] = np.conj(v)
        elif k == "omega":
            out[k] = -v
        else:
            out[k] = v
    return out


# ============================================================ bilinear forms
def fs_forcing(a, b):
    """(f_k, f_d) of the audit's (8): f_k = eta_a phi_b,zz - grad eta_a . grad_H phi_b,
    f_d = -eta_a phi_b,tz - (1/2) grad phi_a . grad phi_b."""
    fk = a["zeta"] * b["phizz"] - (a["zx"] * b["phix"] + a["zy"] * b["phiy"])
    fd = -a["zeta"] * b["phitz"] - 0.5 * (a["phix"] * b["phix"] + a["phiy"] * b["phiy"] + a["phiz"] * b["phiz"])
    return fk, fd


def body_forcing(a, b, body):
    """phi2_n = -n.[(d_a . grad) grad phi_b] + n.[alpha_a x (grad phi_b - v_a)] (module docstring 3)."""
    n = body.n
    dn = np.einsum("ij,ij->i", a["d"], n)
    dt = a["d"] - n * dn[:, None]
    hess = -dn * b["lapS"] + np.einsum("ij,ij->i", dt, b["gS_pn"])
    rot = np.einsum("ij,ij->i", np.cross(np.broadcast_to(a["alpha"], n.shape), b["gradB"] - a["vtr"][None, :]), n)
    return -hess + rot


def quad_force(a, b, st, mn, wl, rho, g):
    """Quadratic second-order (heave force, pitch moment about the CG) as a bilinear form; pieces:
    vel: p = -rho/2 grad phi_a . grad phi_b;  dgrad: p = -rho (d_a . grad) phi_b,t;
    wl: -rho g/2 zr_a zr_b n flare dl;  rot: alpha_a x F1_b, alpha_a x M1_b."""
    body = st["body"]
    n, area, r = body.n, body.area, mn["r"]

    def FM(minus_p):
        fd = minus_p[:, None] * n
        F = 2 * np.sum(fd * area[:, None], 0)
        M = 2 * np.sum(np.cross(r, fd) * area[:, None], 0)
        return np.array([F[2], M[1]])
    out = {}
    out["vel"] = FM(0.5 * rho * np.einsum("ij,ij->i", a["gradB"], b["gradB"]))
    out["dgrad"] = FM(rho * np.einsum("ij,ij->i", a["d"], b["gradtB"]))
    coef = -0.5 * rho * g * a["zrel_wl"] * b["zrel_wl"] * wl["flare"] * wl["dl"]
    fl = coef[:, None] * wl["n"]
    Pw = np.stack([wl["x_mid"], wl["y_mid"], np.zeros_like(wl["x_mid"])], 1) - np.array([mn["xg"], 0.0, mn["zg"]])
    Fw = 2 * np.sum(fl, 0)
    Mw = 2 * np.sum(np.cross(Pw, fl), 0)
    out["wl"] = np.array([Fw[2], Mw[1]])
    Fr = np.cross(a["alpha"], b["F1"])
    Mr = np.cross(a["alpha"], b["M1"])
    out["rot"] = np.array([Fr[2], Mr[1]])
    out["total"] = out["vel"] + out["dgrad"] + out["wl"] + out["rot"]
    return out


def sym(fun, a, b, diff=False, self_pair=False):
    """Second-order amplitude at w_a + w_b (diff=False) or w_a - w_b (diff=True); self_pair halves it."""
    if diff:
        b = conj_fields(b)
    qa, qb = fun(a, b), fun(b, a)
    fac = 0.25 if self_pair else 0.5

    def comb(x, y):
        return fac * (x + y)
    if isinstance(qa, dict):
        return {k: comb(qa[k], qb[k]) for k in qa}
    if isinstance(qa, tuple):
        return tuple(comb(x, y) for x, y in zip(qa, qb))
    return comb(qa, qb)


def _sub(x, y):
    if isinstance(x, dict):
        return {k: x[k] - y[k] for k in x}
    if isinstance(x, tuple):
        return tuple(p - q for p, q in zip(x, y))
    return x - y


def _conj(x):
    if isinstance(x, dict):
        return {k: np.conj(v) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(np.conj(v) for v in x)
    return np.conj(x)


# ============================================================ second-order potential
def potential_force(st, Omega, phin2, fk, fd, mn):
    """Solve at Omega (columns = right-hand sides) with the beach scaled to Omega/NU_REF; return
    (F3, M5) of -rho i Omega phi_2 on the mean wetted surface per column, and the fields."""
    s, body = st["s"], st["body"]
    s.set_operator(sp.csr_matrix((s.NF, s.NF)), st["nu0"] * (Omega / NU_REF))
    t0 = time.time()
    sol = s.solve_freq(Omega, phin2, fd=fd, fk=fk)
    dt = time.time() - t0
    a = body.area
    minus_p2 = s.rho * 1j * Omega * sol["phi_B"]
    F3 = 2 * np.sum(minus_p2 * (body.n[:, 2] * a)[:, None], 0)
    M5 = 2 * np.sum(minus_p2 * (mn["n5"] * a)[:, None], 0)
    return dict(F=np.stack([F3, M5], 1), phi2_B=sol["phi_B"], phi2_F=sol["phi_F"], t_solve=dt)


# ============================================================ driver
POT_GRID_DEFAULT = dict(npl=12, lam_mult=4.0, diff=dict(M=40, up=4.0, lat=4.0, b_up=2.0, b_lat=2.0), NF_cap=8500)


def dedicated_grid(Om, G, hull, omegas, pg):
    """Grid parameters (unsteady.build_grid units) for the second-order solve at Omega. Sum frequencies: dx =
    lambda_2/npl, extent lam_mult lambda_2 beyond the waterline ends and the beam, beach the outer half.
    Difference frequencies: the wide coarse grid pg['diff']. None if the estimated size exceeds the cap."""
    L = hull.L
    lam2 = 2 * np.pi * G0 / Om ** 2
    Lwl = G["xe"][G["i_b"]] - G["xe"][G["i_s"]]
    ymax = float(G["ywl_e"].max())
    if Om >= 0.9 * 2 * min(omegas):
        dx = lam2 / pg["npl"]
        M2 = int(np.ceil(Lwl / dx))
        ext = pg["lam_mult"] * lam2
        kw = dict(M=M2, aspect=1.0, J_mult=2.0, up=ext / L, down=ext / L, lat=(ymax + ext) / L, r_up=1.12, r_dn=1.12,
                  r_lat=1.10, h_max=0.5, b_up=0.5 * ext / L, b_dn=0.5 * ext / L, b_lat=0.5 * ext / L, nu_max=1.5)
        # size estimate: band + geometric sides
        dx2 = Lwl / M2
        nside = int(np.log(1 + ext * 0.12 / dx2) / np.log(1.12)) + 1
        nx = M2 + 2 * nside
        J = int(np.ceil(2.0 * ymax / dx2))
        nlat = int(np.log(1 + ext * 0.10 / dx2) / np.log(1.10)) + 1
        if nx * (J + nlat) > pg["NF_cap"]:
            return None
        return kw
    d = dict(pg["diff"])
    kw = dict(M=d["M"], aspect=1.0, J_mult=2.0, up=d["up"], down=d["up"], lat=d["lat"], r_up=1.12, r_dn=1.12,
              r_lat=1.10, h_max=0.5, b_up=d["b_up"], b_dn=d["b_up"], b_lat=d["b_lat"], nu_max=1.5)
    return kw


def interp_fs(G, G2, f):
    """Interpolate free-surface panel fields (NF, m) of grid G to the centroids of grid G2 (linear in (x, y),
    nearest outside the convex hull)."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    P = G["fs"].c[:, :2]
    P2 = G2["fs"].c[:, :2]
    f = np.asarray(f)
    lin = LinearNDInterpolator(P, f)
    out = lin(P2)
    bad = np.isnan(out)
    if bad.any():
        near = NearestNDInterpolator(P, f)
        o2 = near(P2)
        out[bad] = o2[bad]
    return out


def run_case(M=40, hull_mesh=(40, 8), face_ny=6, grid_extra=None, omegas=OMEGAS, label="base", device=None,
             say=print, with_potential=True, pairs_only=None, pot_grid=POT_GRID_DEFAULT):
    """pairs_only: optional list of (w_motion, w_wave) ordered pairs to restrict the second-order work.
    pot_grid: parameters of the dedicated second-order grids (None: same grid only)."""
    hull, p = scarab_form()
    st = build_zero_speed(hull, hull_mesh=hull_mesh, face_ny=face_ny, grid_kw=dict(M=M, **(grid_extra or {})),
                          device=device)
    s, G, body = st["s"], st["G"], st["body"]
    st["nu0"] = s.nu.copy()
    rho, g = s.rho, s.g
    xg, zg = 0.0, -hull.sink_rest
    mn = mode_normals(body, (xg, zg))
    mn["xg"], mn["zg"] = xg, zg
    wl = waterline_data(st)
    met = fs_metric(G)
    say(f"[{label}] hull mesh {hull_mesh} + transom face {face_ny}x{hull_mesh[1]}: NB {s.NB}; free surface M {M}: "
        f"NF {s.NF}, shape {G['shape']}, dx {G['dx']:.3f} m, x {G['xe'][0]:.1f}..{G['xe'][-1]:.1f} m, "
        f"y to {G['ylines'][-1]:.1f} m; build {st['t_build']:.1f} s; mem {mem_report()}")
    res = dict(label=label, M=M, hull_mesh=list(hull_mesh), NB=int(s.NB), NF=int(s.NF), dx=float(G["dx"]),
               x_extent=[float(G["xe"][0]), float(G["xe"][-1])], y_extent=float(G["ylines"][-1]),
               nu_max=float(st["nu0"].max()), t_build=st["t_build"], first=[], pairs=[], grid_kw=st["grid_kw"])
    FO = {}
    a = body.area
    for w in omegas:
        f = first_order(st, w, mn, wl, met)
        FO[w] = f
        T = {}
        for u, nj in (("3", mn["n3"]), ("5", mn["n5"])):
            for v, name in (("3", "heave"), ("5", "pitch")):
                T[u + v] = -rho * 2 * np.sum(1j * w * f[name]["phiB"] * nj * a)
        X = {u: rho * 2 * np.sum(1j * w * f["wave"]["phiB"] * nj * a)
             for u, nj in (("1", body.n[:, 0]), ("3", mn["n3"]), ("5", mn["n5"]))}
        row = dict(omega=w, k=w * w / g, lam=2 * np.pi * g / w ** 2, t_solve=f["heave"]["t_solve"],
                   A33=-T["33"].real / w ** 2, B33=T["33"].imag / w, A55=-T["55"].real / w ** 2, B55=T["55"].imag / w,
                   A35=-T["35"].real / w ** 2, B35=T["35"].imag / w, A53=-T["53"].real / w ** 2, B53=T["53"].imag / w,
                   T={k: complex(v) for k, v in T.items()}, X={k: complex(v) for k, v in X.items()},
                   zeta_wl_max=dict(heave=float(np.abs(f["heave"]["zeta_wl"]).max()),
                                    pitch=float(np.abs(f["pitch"]["zeta_wl"]).max()),
                                    wave=float(np.abs(f["wave"]["zeta_wl"]).max())),
                   lapS_max=dict(heave=float(np.abs(f["heave"]["lapS"]).max()),
                                 wave=float(np.abs(f["wave"]["lapS"]).max())),
                   phiF_max=dict(heave=float(np.abs(f["heave"]["phiF"]).max()),
                                 wave=float(np.abs(f["wave"]["phiF"]).max())))
        res["first"].append(row)
        say(f"  w {w:.2f} (lambda {row['lam']:.2f} m): A33 {row['A33']:.0f} B33 {row['B33']:.0f} A55 {row['A55']:.0f} "
            f"B55 {row['B55']:.0f} A35 {row['A35']:.0f} |X1| {abs(X['1']):.0f} |X3| {abs(X['3']):.0f} "
            f"|X5| {abs(X['5']):.0f}; solve {row['t_solve']:.2f} s; |zeta_wl| max heave "
            f"{row['zeta_wl_max']['heave']:.3f} wave {row['zeta_wl_max']['wave']:.3f}")
    res["mem_first_order"] = mem_report()
    # ---------------------------------------------------- second-order pairs
    jobs = []
    pairs = []
    qf = lambda x, y: quad_force(x, y, st, mn, wl, rho, g)
    bf = lambda x, y: body_forcing(x, y, body)
    for wi in omegas:
        for wj in omegas:
            if pairs_only is not None and (wi, wj) not in pairs_only:
                continue
            for u in ("heave", "pitch"):
                for diff in (False, True):
                    Om = abs(wi - wj) if diff else wi + wj
                    a_, b_ = FO[wi][u], FO[wj]["wave"]
                    q = sym(qf, a_, b_, diff)
                    q_inc = sym(qf, a_, b_["inc_only"], diff)
                    q_hf = sym(qf, a_["hf"], b_["hf"], diff)
                    flip = diff and wi < wj                       # amplitude at the positive frequency w_j - w_i
                    if flip:
                        q, q_inc, q_hf = _conj(q), _conj(q_inc), _conj(q_hf)
                    pr = dict(kind="mixed", motion=u, w_motion=wi, w_wave=wj, diff=diff, Omega=Om,
                              quad={k: [complex(v[0]), complex(v[1])] for k, v in q.items()},
                              quad_inc_only={k: [complex(v[0]), complex(v[1])] for k, v in q_inc.items()},
                              quad_hf={k: [complex(v[0]), complex(v[1])] for k, v in q_hf.items()})
                    if with_potential and Om > 1e-9:
                        fk, fd = sym(fs_forcing, a_, b_, diff)
                        pn2 = sym(bf, a_, b_, diff)
                        if flip:
                            fk, fd, pn2 = np.conj(fk), np.conj(fd), np.conj(pn2)
                        jobs.append((Om, len(pairs), pn2, fk, fd))
                    pairs.append(pr)
        if pairs_only is not None and (wi, wi) not in pairs_only:
            continue
        for u in ("heave", "pitch", "wave"):
            for diff in (False, True):
                Om = 0.0 if diff else 2 * wi
                a_ = FO[wi][u]
                q = sym(qf, a_, a_, diff, self_pair=True)
                q_hf = sym(qf, a_["hf"], a_["hf"], diff, self_pair=True)
                pr = dict(kind=u + "x" + u, motion=u, w_motion=wi, w_wave=wi, diff=diff, Omega=Om,
                          quad={k: [complex(v[0]), complex(v[1])] for k, v in q.items()},
                          quad_hf={k: [complex(v[0]), complex(v[1])] for k, v in q_hf.items()})
                if with_potential and Om > 1e-9:
                    fk, fd = sym(fs_forcing, a_, a_, diff, self_pair=True)
                    if u == "wave":
                        fk, fd = _sub((fk, fd), sym(fs_forcing, a_["inc"], a_["inc"], diff, self_pair=True))
                    pn2 = sym(bf, a_, a_, diff, self_pair=True)
                    jobs.append((Om, len(pairs), pn2, fk, fd))
                pairs.append(pr)
    t_pot = 0.0
    by_Om = {}
    for Om, idx, pn2, fk, fd in jobs:
        by_Om.setdefault(round(Om, 6), []).append((idx, pn2, fk, fd))
    res["pot_grids"] = []
    for Om, lst in sorted(by_Om.items()):
        pn2 = np.stack([x[1] for x in lst], 1)
        fk = np.stack([x[2] for x in lst], 1)
        fd = np.stack([x[3] for x in lst], 1)
        lam2 = 2 * np.pi * g / Om ** 2
        # (i) on the SAME grid as the first order
        r = potential_force(st, Om, pn2, fk, fd, mn)
        r_fs = potential_force(st, Om, np.zeros_like(pn2), fk, fd, mn)
        t_pot += r["t_solve"] + r_fs["t_solve"]
        for c, (idx, _, _, _) in enumerate(lst):
            pairs[idx]["pot_same"] = [complex(r["F"][c, 0]), complex(r["F"][c, 1])]
            pairs[idx]["pot_same_fs_only"] = [complex(r_fs["F"][c, 0]), complex(r_fs["F"][c, 1])]
            pairs[idx]["forcing"] = dict(fk_max=float(np.abs(fk[:, c]).max()), fd_max=float(np.abs(fd[:, c]).max()),
                                         phin2_max=float(np.abs(pn2[:, c]).max()),
                                         phi2B_max=float(np.abs(r["phi2_B"][:, c]).max()),
                                         phi2F_max=float(np.abs(r["phi2_F"][:, c]).max()),
                                         lam2=lam2, panels_per_lam2_same=lam2 / G["dx"])
        say(f"  Omega {Om:.2f} rad/s: {len(lst)} rhs; same grid: lambda_2 {lam2:.2f} m = {lam2 / G['dx']:.1f} dx, domain "
            f"{(G['xe'][-1] - G['xe'][0]) / lam2:.1f} lambda_2, 2 solves {r['t_solve'] + r_fs['t_solve']:.2f} s")
        # (ii) on a DEDICATED grid for Omega (fine and short for sum frequencies, coarse and wide for difference
        # frequencies); the free-surface forcing is interpolated from the first-order grid, the body forcing is
        # on the same hull mesh
        if pot_grid is not None:
            kw2 = dedicated_grid(Om, G, hull, omegas, pot_grid)
            if kw2 is None:
                say(f"    dedicated grid skipped (estimated size over the memory cap)")
                continue
            st2 = build_zero_speed(hull, hull_mesh=hull_mesh, face_ny=face_ny, grid_kw=kw2, device=device)
            st2["nu0"] = st2["s"].nu.copy()
            G2 = st2["G"]
            fk2 = interp_fs(G, G2, fk)
            fd2 = interp_fs(G, G2, fd)
            r2 = potential_force(st2, Om, pn2, fk2, fd2, mn)
            r2fs = potential_force(st2, Om, np.zeros_like(pn2), fk2, fd2, mn)
            t_pot += r2["t_solve"] + r2fs["t_solve"] + st2["t_build"]
            # diagnostic: the incident x incident free-surface forcing alone (exact answer phi_2 = 0 in deep
            # water; whatever the discrete problem returns is the spurious response of the truncated, damped
            # free surface to a forcing that does not decay) -- an error indicator for the sum frequency
            w_half = [w for w in omegas if abs(2 * w - Om) < 1e-6]
            if w_half:
                inc = FO[w_half[0]]["wave"]["inc"]
                fkI, fdI = sym(fs_forcing, inc, inc, False, self_pair=True)
                rI = potential_force(st2, Om, np.zeros((len(body), 1), complex), interp_fs(G, G2, fkI[:, None]),
                                     interp_fs(G, G2, fdI[:, None]), mn)
                res.setdefault("spurious_II", {})[f"{Om:.2f}"] = dict(F3=complex(rI["F"][0, 0]), M5=complex(rI["F"][0, 1]),
                                                                       fk_max=float(np.abs(fkI).max()), fd_max=float(np.abs(fdI).max()))
                say(f"    incident x incident forcing alone (should give 0): |F3| {abs(rI['F'][0, 0]):.0f} N/m^2, "
                    f"|M5| {abs(rI['F'][0, 1]):.0f} N m/m^2 per A^2")
                t_pot += rI["t_solve"]
            info = dict(Omega=Om, lam2=lam2, M=kw2["M"], NF=int(st2["s"].NF), dx=float(G2["dx"]), panels_per_lam2=lam2 / G2["dx"],
                        x_extent=[float(G2["xe"][0]), float(G2["xe"][-1])], y_extent=float(G2["ylines"][-1]),
                        domain_over_lam2=(G2["xe"][-1] - G2["xe"][0]) / lam2, beach_L=kw2["b_up"], t_build=st2["t_build"],
                        t_solve=r2["t_solve"], mem=mem_report(), grid_kw=kw2)
            res["pot_grids"].append(info)
            for c, (idx, _, _, _) in enumerate(lst):
                pairs[idx]["pot"] = [complex(r2["F"][c, 0]), complex(r2["F"][c, 1])]
                pairs[idx]["pot_fs_only"] = [complex(r2fs["F"][c, 0]), complex(r2fs["F"][c, 1])]
                pairs[idx]["pot_body_only"] = [complex(r2["F"][c, 0] - r2fs["F"][c, 0]), complex(r2["F"][c, 1] - r2fs["F"][c, 1])]
                pairs[idx]["pot_grid"] = dict(NF=info["NF"], dx=info["dx"], panels_per_lam2=info["panels_per_lam2"],
                                              domain_over_lam2=info["domain_over_lam2"])
            say(f"    dedicated grid: M {kw2['M']}, NF {info['NF']}, dx {info['dx']:.3f} m = lambda_2/{info['panels_per_lam2']:.1f}, "
                f"x {info['x_extent'][0]:.1f}..{info['x_extent'][1]:.1f} m ({info['domain_over_lam2']:.1f} lambda_2), "
                f"build {st2['t_build']:.1f} s, solve {r2['t_solve']:.2f} s, GPU peak {info['mem'].get('gpu_peak_MB', 0):.0f} MB; "
                f"|F3 pot| same {abs(r['F'][0, 0]):.0f} -> dedicated {abs(r2['F'][0, 0]):.0f} (first rhs)")
            st2["s"].release()
            del st2, r2, r2fs
            _free()
        else:
            for c, (idx, _, _, _) in enumerate(lst):
                pairs[idx]["pot"] = pairs[idx]["pot_same"]
                pairs[idx]["pot_fs_only"] = pairs[idx]["pot_same_fs_only"]
    s.set_operator(sp.csr_matrix((s.NF, s.NF)), st["nu0"])
    res["pairs"] = pairs
    res["t_potential_solves"] = t_pot
    res["mem"] = mem_report()
    # hydrostatic check of the waterline term: exact buoyancy vs sinkage (second derivative) against the
    # heave x heave mean waterline term (-rho g/2 zr^2 flare n dl with zr = -1 per unit heave)
    hs = []
    for dz in (-0.02, 0.0, 0.02):
        hl = hull.hydrostatic_load(hull.sink_rest - dz, hull.trim_rest, rho=rho, g=g, cg=(0.0, 0.0))
        hs.append((hl["Fz"], hl["My"]))
    hs = np.array(hs)
    d2 = (hs[2] - 2 * hs[1] + hs[0]) / 0.02 ** 2                 # d^2 F / d xi3^2
    wl_geo = -0.5 * rho * g * np.sum(wl["flare"] * wl["dl"] * wl["n"][:, 2])      # = Re q(a, conj a)/2 for zr = -1
    wl_geo_M = -0.5 * rho * g * np.sum(wl["flare"] * wl["dl"] * (
        (wl["x_mid"] - 0 * xg) * 0 + ((0.0 - zg) * wl["n"][:, 0] - (wl["x_mid"] - xg) * wl["n"][:, 2])))
    res["hydrostatic_check"] = dict(d2Fz_dxi3_2=float(d2[0]), d2My_dxi3_2=float(d2[1]), C33=float(-(hs[2][0] - hs[0][0]) / 0.04),
                                    wl_geometric_mean_F3_per_a2=float(wl_geo), wl_geometric_mean_M5_per_a2=float(wl_geo_M),
                                    note="mean second-order hydrostatic heave force per a^2 = d2F/4; compare with Re(wl term) of heave x heave at Omega 0")
    res["_FO"] = FO
    res["_st"] = st
    res["_mn"] = mn
    return res


# ============================================================ reporting
def mixed_table(res):
    first = {r["omega"]: r for r in res["first"]}
    rows = []
    for pr in res["pairs"]:
        if pr["kind"] != "mixed":
            continue
        u, wi, wj = pr["motion"], pr["w_motion"], pr["w_wave"]
        q = np.array(pr["quad"]["total"])
        qi = np.array(pr["quad_inc_only"]["total"])
        pot = np.array(pr.get("pot", [0.0, 0.0]))
        pot_same = np.array(pr.get("pot_same", [0.0, 0.0]))
        tot = q + pot
        fi, fj = first[wi], first[wj]
        T3u = fi["T"]["33"] if u == "heave" else fi["T"]["35"]
        T5u = fi["T"]["53"] if u == "heave" else fi["T"]["55"]
        out = dict(motion=u, wi=wi, wj=wj, Omega=pr["Omega"], diff=pr["diff"],
                   F3_unit=complex(tot[0]), M5_unit=complex(tot[1]), F3_quad_unit=complex(q[0]), F3_pot_unit=complex(pot[0]),
                   F3_pot_same_unit=complex(pot_same[0]), M5_pot_same_unit=complex(pot_same[1]),
                   pot_grid=pr.get("pot_grid"),
                   M5_quad_unit=complex(q[1]), M5_pot_unit=complex(pot[1]), F3_3d_unit=complex(tot[0] - qi[0]),
                   M5_3d_unit=complex(tot[1] - qi[1]),
                   pieces={k: complex(v[0]) for k, v in pr["quad"].items()})
        for key, amps in (("small", AMPS_SMALL), ("oper", AMPS_OPER)):
            a_, A_ = amps[u], amps["wave"]
            ref3 = abs(T3u) * a_ + abs(fj["X"]["3"]) * A_
            ref5 = abs(T5u) * a_ + abs(fj["X"]["5"]) * A_
            out[key] = dict(a=a_, A=A_, F3=abs(tot[0]) * a_ * A_, F3_quad=abs(q[0]) * a_ * A_, F3_pot=abs(pot[0]) * a_ * A_,
                            F3_3d=abs(tot[0] - qi[0]) * a_ * A_, ref_F3=ref3, frac_F3=abs(tot[0]) * a_ * A_ / ref3,
                            frac3d_F3=abs(tot[0] - qi[0]) * a_ * A_ / ref3,
                            M5=abs(tot[1]) * a_ * A_, M5_quad=abs(q[1]) * a_ * A_, M5_pot=abs(pot[1]) * a_ * A_,
                            M5_3d=abs(tot[1] - qi[1]) * a_ * A_, ref_M5=ref5, frac_M5=abs(tot[1]) * a_ * A_ / ref5,
                            frac3d_M5=abs(tot[1] - qi[1]) * a_ * A_ / ref5)
        rows.append(out)
    return rows


def _json(o):
    if isinstance(o, dict):
        return {str(k): _json(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_json(v) for v in o]
    if isinstance(o, complex):
        return [o.real, o.imag]
    if isinstance(o, np.ndarray):
        return _json(o.tolist())
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.complexfloating,)):
        return [float(o.real), float(o.imag)]
    return o


def write_tables(res, rows, conv, say):
    rho_g_b = 1025.0 * G0 * 2.0252
    say("")
    say("=== first order (Rankine, this mesh) vs Capytaine table bem_probe_lid_24_64.txt ===")
    tab = {3.04: (5215, 13628, 11379, 17108), 3.89: (4248, 13205, 8826, 20915), 4.74: (3891, 12306, 7549, 19959)}
    say(f"  {'w':>5} {'A33':>7} {'B33':>7} {'A55':>7} {'B55':>7} | table (nearest w): A33 B33 A55 B55")
    for r in res["first"]:
        wt = min(tab, key=lambda x: abs(x - r["omega"]))
        say(f"  {r['omega']:>5.2f} {r['A33']:>7.0f} {r['B33']:>7.0f} {r['A55']:>7.0f} {r['B55']:>7.0f} | {wt:.2f}: "
            f"{tab[wt][0]} {tab[wt][1]} {tab[wt][2]} {tab[wt][3]}   |X3| {abs(r['X']['3']):.0f} N/m  |X5| {abs(r['X']['5']):.0f} N  "
            f"|X1| {abs(r['X']['1']):.0f} N/m")
    say("")
    say("=== mixed second-order amplitudes per unit a (m or rad) x A (m); heave force F3 [N/m^2], pitch moment M5 [N m/m^2] ===")
    say("  pieces: vel = -rho/2 grad.grad; dgrad = -rho (d.grad) phi_t; wl = waterline (flare); rot = alpha x F1; pot = -rho i Omega phi_2")
    say(f"  {'motion':>6} {'w_mot':>5} {'w_wav':>5} {'Omega':>6} {'|F3|':>9} {'|F3quad|':>9} {'|F3pot|':>9} {'|F3potSm|':>9} {'|F3 3d|':>9} "
        f"{'vel':>8} {'dgrad':>8} {'wl':>8} {'rot':>8} | {'|M5|':>9} {'|M5quad|':>9} {'|M5pot|':>9} {'|M5potSm|':>9} {'|M5 3d|':>9}  F3/(rho g b)")
    for r in rows:
        pc = r["pieces"]
        say(f"  {r['motion']:>6} {r['wi']:>5.1f} {r['wj']:>5.1f} {r['Omega']:>6.2f} {abs(r['F3_unit']):>9.0f} {abs(r['F3_quad_unit']):>9.0f} "
            f"{abs(r['F3_pot_unit']):>9.0f} {abs(r['F3_pot_same_unit']):>9.0f} {abs(r['F3_3d_unit']):>9.0f} {abs(pc['vel']):>8.0f} {abs(pc['dgrad']):>8.0f} "
            f"{abs(pc['wl']):>8.0f} {abs(pc['rot']):>8.0f} | {abs(r['M5_unit']):>9.0f} {abs(r['M5_quad_unit']):>9.0f} "
            f"{abs(r['M5_pot_unit']):>9.0f} {abs(r['M5_pot_same_unit']):>9.0f} {abs(r['M5_3d_unit']):>9.0f}  {abs(r['F3_unit']) / rho_g_b:.3f}")
    for key, amps in (("small", AMPS_SMALL), ("oper", AMPS_OPER)):
        say("")
        say(f"=== mixed term at a = {amps['heave']} m heave / {amps['pitch']} rad pitch, A = {amps['wave']} m: "
            f"fraction of the first-order force |T3u| a + |X3| A (moment: |T5u| a + |X5| A) ===")
        say(f"  {'motion':>6} {'w_mot':>5} {'w_wav':>5} {'Omega':>6} {'F3 mixed':>9} {'ref':>9} {'frac':>7} {'frac3d':>7} | "
            f"{'M5 mixed':>9} {'ref':>9} {'frac':>7} {'frac3d':>7}")
        for r in rows:
            d = r[key]
            say(f"  {r['motion']:>6} {r['wi']:>5.1f} {r['wj']:>5.1f} {r['Omega']:>6.2f} {d['F3']:>9.1f} {d['ref_F3']:>9.1f} "
                f"{d['frac_F3']:>7.3f} {d['frac3d_F3']:>7.3f} | {d['M5']:>9.1f} {d['ref_M5']:>9.1f} {d['frac_M5']:>7.3f} {d['frac3d_M5']:>7.3f}")
    say("")
    say("=== self pairs (one input with itself): |F3|, |M5| per unit amplitude squared; mean (Omega 0) is Re ===")
    for pr in res["pairs"]:
        if pr["kind"] == "mixed":
            continue
        q = pr["quad"]["total"]
        pot = pr.get("pot", [0, 0])
        pots = pr.get("pot_same", [0, 0])
        say(f"  {pr['kind']:>10} w {pr['w_motion']:.1f} Omega {pr['Omega']:.2f}: quad F3 {complex(q[0]):.1f} M5 {complex(q[1]):.1f}; "
            f"pot F3 {complex(pot[0]):.1f} M5 {complex(pot[1]):.1f} (same grid F3 {complex(pots[0]):.1f}); pieces F3: "
            + ", ".join(f"{k} {complex(v[0]):.0f}" for k, v in pr["quad"].items() if k != "total"))
    hc = res["hydrostatic_check"]
    say(f"  hydrostatic check (heave x heave, hydrostatic part of the waterline term): exact d2Fz/dxi3^2 = "
        f"{hc['d2Fz_dxi3_2']:.0f} N/m^2 -> mean per a^2 {hc['d2Fz_dxi3_2'] / 4:.0f} N/m^2 (moment {hc['d2My_dxi3_2'] / 4:.0f}); "
        f"panel waterline integral with flare: {hc['wl_geometric_mean_F3_per_a2']:.0f} N/m^2 (moment "
        f"{hc['wl_geometric_mean_M5_per_a2']:.0f}); C33 exact {hc['C33']:.0f} N/m")
    if res.get("spurious_II"):
        say("  incident x incident free-surface forcing alone on the dedicated grids (exact: 0): "
            + "; ".join(f"Omega {k}: |F3| {abs(complex(*v['F3'])) if isinstance(v['F3'], list) else abs(v['F3']):.0f}, "
                        f"|M5| {abs(complex(*v['M5'])) if isinstance(v['M5'], list) else abs(v['M5']):.0f}"
                        for k, v in res["spurious_II"].items()))
    say("")
    say("=== convergence (mixed heave x wave at 3.9 + 3.9, sum frequency 7.8 rad/s; wave x wave mean at Omega 0) ===")
    say("  first-order grid: NF, dx, x-extent; A33/B33 at 3.9; quad = quadratic part (first-order fields only);")
    say("  pot same = second-order potential on the first-order grid; pot = on the dedicated grid (NF2, dx2 = lambda_2/ppl, domain/lambda_2)")
    say(f"  {'case':>26} {'NF':>5} {'dx':>6} {'x-ext':>12} {'A33':>5} {'B33':>6} {'|F3 quad|':>9} {'|F3potSame|':>11} {'|F3 pot|':>8} "
        f"{'|F3pot fs|':>10} {'NF2':>5} {'dx2':>6} {'ppl':>5} {'dom':>5} {'|M5 quad|':>9} {'|M5 pot|':>8} "
        f"{'meanF3w':>8} {'meanM5w':>8} {'t_build':>7} {'t_solve':>7} {'RSS MB':>7} {'GPU MB':>7}")
    for c in conv:
        say(f"  {c['label']:>26} {c['NF']:>5} {c['dx']:>6.3f} {c['x_extent'][0]:>5.1f}..{c['x_extent'][1]:<5.1f} {c['A33']:>5.0f} {c['B33']:>6.0f} "
            f"{c['F3q']:>9.0f} {c['F3p_same']:>11.0f} {c['F3p']:>8.0f} {c['F3pfs']:>10.0f} {c['NF2']:>5} {c['dx2']:>6.3f} {c['ppl2']:>5.1f} "
            f"{c['dom2']:>5.1f} {c['M5q']:>9.0f} {c['M5p']:>8.0f} {c['meanF3w']:>8.0f} {c['meanM5w']:>8.0f} "
            f"{c['t_build']:>7.1f} {c['t_solve']:>7.2f} {c['peak_rss']:>7.0f} {c['gpu']:>7.0f}")
    if res.get("pot_grids"):
        say("")
        say("=== dedicated second-order grids (base case) ===")
        say(f"  {'Omega':>6} {'lam2 m':>7} {'M':>4} {'NF':>6} {'dx m':>6} {'lam2/dx':>7} {'x-ext m':>14} {'dom/lam2':>8} {'t_build':>7} {'t_solve':>7} {'GPU MB':>7}")
        for gI in res["pot_grids"]:
            say(f"  {gI['Omega']:>6.2f} {gI['lam2']:>7.2f} {gI['M']:>4} {gI['NF']:>6} {gI['dx']:>6.3f} {gI['panels_per_lam2']:>7.1f} "
                f"{gI['x_extent'][0]:>6.1f}..{gI['x_extent'][1]:<6.1f} {gI['domain_over_lam2']:>8.1f} {gI['t_build']:>7.1f} {gI['t_solve']:>7.2f} "
                f"{gI['mem'].get('gpu_peak_MB', 0):>7.0f}")


def conv_entry(res):
    pr = [p for p in res["pairs"] if p["kind"] == "mixed" and p["motion"] == "heave" and p["w_motion"] == 3.9
          and p["w_wave"] == 3.9 and not p["diff"]][0]
    ww = [p for p in res["pairs"] if p["kind"] == "wavexwave" and p["w_motion"] == 3.9]
    wmean = [p for p in ww if p["diff"]][0]
    wsum = [p for p in ww if not p["diff"]][0]
    f0 = [r for r in res["first"] if r["omega"] == 3.9][0]
    pg = pr.get("pot_grid", {})
    pot = pr.get("pot", pr["pot_same"])
    potfs = pr.get("pot_fs_only", pr["pot_same_fs_only"])
    return dict(label=res["label"], NF=res["NF"], dx=res["dx"], x_extent=res["x_extent"],
                F3q=abs(complex(pr["quad"]["total"][0])), F3p=abs(complex(pot[0])), F3pfs=abs(complex(potfs[0])),
                F3p_same=abs(complex(pr["pot_same"][0])), M5p_same=abs(complex(pr["pot_same"][1])),
                M5q=abs(complex(pr["quad"]["total"][1])), M5p=abs(complex(pot[1])),
                NF2=pg.get("NF", 0), dx2=pg.get("dx", 0.0), ppl2=pg.get("panels_per_lam2", 0.0), dom2=pg.get("domain_over_lam2", 0.0),
                meanF3w=complex(wmean["quad"]["total"][0]).real, meanM5w=complex(wmean["quad"]["total"][1]).real,
                F3ww=abs(complex(wsum["quad"]["total"][0]) + complex(wsum.get("pot", wsum["pot_same"])[0])),
                A33=f0["A33"], B33=f0["B33"], t_build=res["t_build"], t_solve=f0["t_solve"],
                peak_rss=res["mem"]["peak_rss_MB"], gpu=res["mem"].get("gpu_peak_MB", 0.0))


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, (key, title) in enumerate((("small", "a = 1 cm, A = 1 cm"), ("oper", "a = 0.1 m (pitch 0.05 rad), A = 0.3 m"))):
        for u, mk in (("heave", "o"), ("pitch", "s")):
            for diff, ls in ((False, "-"), (True, "--")):
                rr = [r for r in rows if r["motion"] == u and r["diff"] == diff]
                x = [r["Omega"] for r in rr]
                y = [100 * r[key]["frac_F3"] for r in rr]
                ax[k].plot(x, y, mk + ls, label=f"{u} x wave, {'difference' if diff else 'sum'} frequency")
        ax[k].axhspan(2, 10, color="0.85", label="original guess 2-10 %")
        ax[k].set_xlabel("output frequency Omega [rad/s]")
        ax[k].set_ylabel("mixed second-order heave force / first-order force [%]")
        ax[k].set_title(title)
        ax[k].set_yscale("log")
        ax[k].grid(True, which="both", alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    quick = "--quick" in sys.argv
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)
    t_all = time.time()
    say(f"V2 second-order double-input check, Scarab 195 at rest, zero speed ({time.strftime('%Y-%m-%d %H:%M')})")
    say(f"omegas {OMEGAS}; amplitudes small {AMPS_SMALL}, operating {AMPS_OPER}; beach of the second-order solve "
        f"scaled by Omega/{NU_REF}")
    res = run_case(M=40, hull_mesh=(40, 8), label="base M40 hull40x8", say=say)
    rows = mixed_table(res)
    t_base = time.time() - t_all
    say(f"base case total {t_base:.0f} s; {res['mem']}")
    conv = [conv_entry(res)]
    del res["_st"]
    _free()
    if not quick:
        for label, kw in (("M60 hull40x8", dict(M=60, hull_mesh=(40, 8))),
                          ("M40 extent2.5 hull40x8", dict(M=40, hull_mesh=(40, 8), grid_extra=dict(up=2.5, lat=4.0, b_up=1.25, b_lat=2.0))),
                          ("M40 hull60x12", dict(M=40, hull_mesh=(60, 12), face_ny=8)),
                          ("M40 pot npl18 lam4", dict(M=40, hull_mesh=(40, 8), pot_grid=dict(POT_GRID_DEFAULT, npl=18))),
                          ("M40 pot npl12 lam6", dict(M=40, hull_mesh=(40, 8), pot_grid=dict(POT_GRID_DEFAULT, lam_mult=6.0))),
                          ("M40 pot npl18 lam6", dict(M=40, hull_mesh=(40, 8), pot_grid=dict(POT_GRID_DEFAULT, npl=18, lam_mult=6.0, NF_cap=11000)))):
            t0 = time.time()
            try:
                r2 = run_case(label=label, say=say, omegas=(3.9,), pairs_only=[(3.9, 3.9)], **kw)
                conv.append(conv_entry(r2))
                r2["t_total"] = time.time() - t0
                res.setdefault("convergence_runs", []).append(_json({k: v for k, v in r2.items() if not k.startswith("_")}))
                del r2["_st"]
                del r2
            except (RuntimeError, MemoryError) as e:
                say(f"  [{label}] FAILED: {type(e).__name__}: {str(e)[:200]}")
            _free()
    write_tables(res, rows, conv, say)
    res["t_total"] = time.time() - t_all
    say(f"\ntotal wall time {res['t_total']:.0f} s")
    with open(os.path.join(OUT, "v2_results_quick.json" if quick else "v2_results.json"), "w", encoding="utf-8") as f:
        json.dump(_json(dict(res=res, mixed=rows, convergence=conv)), f, indent=1)
    with open(os.path.join(OUT, "v2_tables_quick.txt" if quick else "v2_tables.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    try:
        plot(rows, os.path.join(OUT, "v2_mixed.png"))
    except Exception as e:              # noqa: BLE001
        say(f"plot failed: {e}")
    return res, rows


def _free():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
