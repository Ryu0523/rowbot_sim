#!/usr/bin/env python3
"""Check V3 of learn/meta/HULL_FORCE_AUDIT_REVIEW_2026-10-09.md (section 4):
the gravity-free three-dimensional steady correction of HULL_FORCE_DERIVATION
section 3.2, applied to the plant's own ZR+MLM calm equilibria and compared
with Savitsky, against the derivation's predictions (section 7.1):
trim changes -0.81, -0.25, -0.01, +0.12, +0.19, +0.28, +0.30 deg at
12, 16, 20, 25, 30, 35, 40 kn, residuals after correction -0.53, +0.09,
-0.15, -0.27, -0.37, -0.21, -0.08 deg.

Method (nothing is tuned):
  1. ZR+MLM equilibrium (sim/planing_zarnick.ZarnickVessel, ZOPT_ZR with
     mlm=True) at each speed: trim, rise, keel / chine wetted lengths from
     the section immersions (wet: d > 0; chine wet: Wagner half-width
     pi d / (2 tan beta) >= chine half-beam), dynamic lift L_dyn = sum of
     the momentum and crossflow terms (the strip force minus its
     hydrostatic part) and its centre of pressure x_cp,2D.
  2. Wetted planform: spray-root triangle keel -> chine over L_K - L_C, then
     constant beam to the transom. Vortex lattice of studies/derive_slender
     (same Richardson extrapolation) gives k = L_2D+t / L_exact (Jones
     slender-wing over the lattice) and the CP shift dx_cp (exact aft of
     2D+t, metres).
  3. External load on the plant through the opt-in `ext_load` hook of
     ZarnickVessel: dFz = -(1 - 1/k) L_dyn at x_cp,2D plus the moment of
     moving L_dyn / k aft by dx_cp. Re-solve the captive calm equilibrium,
     rebuild the planform, repeat until the trim moves < 0.01 deg.
  4. Report against Savitsky and the predictions.

Run: python -m studies.derive_v3_3d_correction [run|hook]
  hook: proves the ext_load hook inert -- reruns a ZR Fridsma-B trajectory
        and scarab195 calm equilibria and compares with the record made
        before the hook existed (studies/_cache/derive/v3_hook_before.npz)
        and with studies/_cache/sim_redesign/step3.json.
Outputs: studies/_cache/derive/v3_3d_correction.json / .txt
"""
import json
import math
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "studies", "_cache", "derive")
os.makedirs(OUT, exist_ok=True)

from studies.derive_slender import vlm, jones, rich  # noqa: E402

G, KN = 9.81, 0.514444
SPEEDS = (12, 16, 20, 25, 30, 35, 40)
PRED_DTRIM = dict(zip(SPEEDS, (-0.81, -0.25, -0.01, 0.12, 0.19, 0.28, 0.30)))
PRED_RESID = dict(zip(SPEEDS, (-0.53, 0.09, -0.15, -0.27, -0.37, -0.21,
                               -0.08)))


# ------------------------------------------------------------- planform VLM
_VLM_CACHE = {}


def planform_3d(LK_b, LC_b, Ns=(16, 24, 32)):
    """Beam 1. Keel apex at x = 0, spray-root lines to the chines over
    D = LK - LC, chine-wet to the transom at x = LK. Returns k = Lj/L_exact,
    dx_cp = x_cp,exact - x_cp,Jones (aft positive, beams), the Richardson
    orders and the three raw lattice values (convergence record)."""
    key = (round(LK_b, 4), round(LC_b, 4), Ns)
    if key in _VLM_CACHE:
        return _VLM_CACHE[key]
    D = max(LK_b - LC_b, 1e-6)
    s_T = 0.5 * min(1.0, LK_b / D)

    def x_le(y):
        return abs(y) / 0.5 * D

    def semi_of_x(x):
        return np.minimum(0.5 * np.asarray(x) / D, 0.5)

    raw = [vlm(x_le, lambda y: LK_b, s_T, n, n) for n in Ns]
    Li, pL, xi, px = rich(lambda n: vlm(x_le, lambda y: LK_b, s_T, n, n), Ns)
    Lj, xj = jones(semi_of_x, 0.0, LK_b)
    res = dict(k=Lj / Li, dx_cp_b=xi - xj, L_exact=Li, L_jones=Lj,
               xcp_exact_b=xi, xcp_jones_b=xj, order_L=pL, order_x=px,
               raw=[dict(n=n, k=Lj / L, dx_cp_b=xc - xj)
                    for n, (L, xc) in zip(Ns, raw)],
               LK_b=LK_b, LC_b=LC_b, s_T=s_T)
    _VLM_CACHE[key] = res
    return res


# ------------------------------------------------------------- plant side
def make_vessel():
    from sim import config
    from sim.planing_zarnick import ZarnickVessel, ZOPT_ZR
    from sim.test_vessel import Monochromatic
    h, _ = config.load("scarab195")
    v = ZarnickVessel(h, Monochromatic(1.0, 0.0), dt=0.02,
                      zopt=dict(ZOPT_ZR, mlm=True))
    return h, v


def settle(v, U, ext, s0=None, n_steps=4000, dt=0.005):
    """The plant's own captive settling run (running_attitude's loop: Heun,
    dt 0.005 s, surge held), with the external load `ext` and an optional
    warm start s0 (continuation from the previous equilibrium instead of
    from rest). Returns (z, th, diag): diag has the oscillation of z and
    pitch over the last 200 steps and a finiteness flag."""
    from sim.test_vessel import Monochromatic
    w = type(v).__new__(type(v))
    w.__dict__.update(v.__dict__)
    w.sea = Monochromatic(1.0, 0.0)
    w._wave_arrays()
    w.captive_u = float(U)
    w.ext_load = None if ext is None else np.asarray(ext, float)
    s = np.zeros(14) if s0 is None else np.array(s0, float)
    s[6] = U
    tail = []
    for i in range(n_steps):
        k1, _ = w.deriv(s, 0.0)
        s2 = s + dt * k1
        k2, _ = w.deriv(s2, 0.0)
        s = s + 0.5 * dt * (k1 + k2)
        s[6] = U
        if i >= n_steps - 200:
            tail.append((s[2], s[4]))
        if not np.isfinite(s[2]) or not np.isfinite(s[4]):
            return float("nan"), float("nan"), dict(finite=False, step=i)
    tail = np.array(tail)
    return float(s[2]), float(s[4]), dict(
        finite=True, osc_z=float(np.ptp(tail[:, 0])),
        osc_th_deg=float(np.degrees(np.ptp(tail[:, 1]))))


def equilibrium(v, U, ext=None, warm=None):
    """Captive calm equilibrium of `v` at speed U with the external load
    `ext` (6-vector or None); warm = (z, th) to continue from, else the
    plant's own cold start from rest (running_attitude). Returns the
    quantities V3 needs."""
    from sim.planing_vessel import SP
    if warm is None and ext is None:
        v.ext_load = None
        z, th = v.running_attitude(U)
        diag = dict(finite=True, cold=True)
    else:
        s0 = None
        if warm is not None:
            s0 = np.zeros(14)
            s0[2], s0[4] = warm
        z, th, diag = settle(v, U, ext, s0)
        diag["cold"] = warm is None
    if not np.isfinite(z):
        return dict(U=U, kn=U / KN, finite=False, diag=diag,
                    ext=None if ext is None else [float(e) for e in ext])
    v.ext_load = None if ext is None else np.asarray(ext, float)
    s = np.zeros(14)
    s[6], s[2], s[4] = U, z, th
    v.captive_u = U
    _, aux = v.deriv(s, 0.0)
    v.captive_u = None
    v.ext_load = None
    cr = dict(R=aux["fx"], pressure=-aux["Xp"], friction=aux["Df"],
              air=aux["Da"], trim_deg=float(-np.degrees(v.Th_rest + th)))
    d, V = aux["d"], aux["V"]
    x, dx, rho = v.x, v.dx, v.rho
    wet = d > 0
    cw = 0.5 * np.pi * np.maximum(d, 0) / v.tanb
    chine_wet = wet & (cw >= v.yc)
    # the strip terms as deriv writes them (ZR: normal=False, mom='entry',
    # cf='abs'); at equilibrium ett = q = 0 and the keel is straight
    A, c, ma, dma, girth = v._section(d)
    ctr = v._ctr(U)
    ab = v._a_b(U)
    cf = 0.5 * rho * v.c_dc * 2.0 * c * V * np.abs(V) * ctr
    mom = aux["mom"]
    buoy = ab * rho * G * A * ctr
    f_dyn = mom + cf
    L_dyn = f_dyn.sum() * dx
    L_buoy = buoy.sum() * dx
    x_cp = (x * f_dyn).sum() * dx / L_dyn
    x_cp_buoy = (x * buoy).sum() * dx / max(L_buoy, 1e-9)
    ext_z = 0.0 if ext is None else float(ext[2])
    # keel wetted length: d is linear in x in calm water, so the wetted
    # point is where d crosses zero (continuous), station count as well
    LK_sta, LC_sta = wet.sum() * dx, chine_wet.sum() * dx
    LK = _zero_cross(x, d, v) - x[0] + 0.5 * dx
    LC = _zero_cross(x, cw - v.yc, v) - x[0] + 0.5 * dx
    LK, LC = min(LK, v.L), max(min(LC, LK), 0.0)
    trim = cr["trim_deg"]
    tau_mean = float(SP * (v.Th_rest + th))      # uniform bottom slope
    return dict(U=U, kn=U / KN, finite=True, diag=diag, z=float(z),
                th=float(th), trim_deg=trim,
                R_total=float(cr["R"]), R_air=float(cr["air"]),
                R=float(cr["R"] - cr["air"]), R_fric=float(cr["friction"]),
                R_press=float(cr["pressure"]), LK=float(LK), LC=float(LC),
                LK_sta=float(LK_sta), LC_sta=float(LC_sta),
                L_dyn=float(L_dyn), L_buoy=float(L_buoy), x_cp=float(x_cp),
                x_cp_buoy=float(x_cp_buoy),
                L_mom=float(mom.sum() * dx), L_cf=float(cf.sum() * dx),
                balance=float(L_dyn + L_buoy + ext_z - v.m * G),
                tau_mean=tau_mean,
                x_apex=float(_zero_cross(x, d, v)),
                ext=None if ext is None else [float(e) for e in ext])


def _zero_cross(x, f, v):
    """Forward end of the contiguous run of stations with f > 0 that starts
    at the transom (f: immersion d, or Wagner half-width minus chine
    half-beam), by linear interpolation of the first sign change; the bow
    if the run reaches the stem, the transom if it is empty. Contiguous
    from the transom so that the bow taper, where the chine half-beam goes
    to zero, cannot count as 'chine wet'."""
    f = np.asarray(f, float)
    if f[0] <= 0 or not np.isfinite(f[0]):
        return float(x[0] - 0.5 * v.dx)
    neg = np.flatnonzero(~(f > 0))
    if len(neg) == 0:
        return float(x[-1] + 0.5 * v.dx)
    i = neg[0] - 1
    return float(x[i] + (x[i + 1] - x[i]) * f[i] / (f[i] - f[i + 1]))


def correction(eq, b, Ns=(16, 24, 32)):
    """3-D table for the planform of equilibrium `eq`, and the external
    load (6-vector, body frame about the CG) it implies."""
    from sim.planing_vessel import SP
    p3 = planform_3d(eq["LK"] / b, eq["LC"] / b, Ns)
    k, dx_cp = p3["k"], p3["dx_cp_b"] * b
    L, xcp = eq["L_dyn"], eq["x_cp"]
    dFz = -(1.0 - 1.0 / k) * L
    # moment of L/k at (xcp - dx_cp) minus moment of L at xcp
    dM = SP * ((L / k) * (xcp - dx_cp) - L * xcp)
    ext = np.array([0.0, 0.0, dFz, 0.0, dM, 0.0])
    # where the plant's own 2D+t puts the dynamic CP, measured aft from the
    # wetted keel apex, in beams -- against Jones's for the same planform
    xcp_plant_from_apex_b = (eq["x_apex"] - xcp) / b
    return dict(k=k, dx_cp=dx_cp, dFz=dFz, dM=dM, ext=ext.tolist(),
                xcp_plant_from_apex_b=xcp_plant_from_apex_b,
                xcp_jones_from_apex_b=p3["xcp_jones_b"],
                xcp_exact_from_apex_b=p3["xcp_exact_b"], vlm=p3), ext


def savitsky_ref(h, U):
    from hydro.planing import savitsky
    sv = savitsky(U, h.mass, h.lcg, h.b, h.beta_deg)
    tanb = math.tan(math.radians(h.beta_deg))
    D = h.b * tanb / (math.pi * math.tan(math.radians(sv["tau_deg"])))
    LK = min(sv["lam"] * h.b + 0.5 * D, h.L)
    return dict(trim_deg=sv["tau_deg"], R=sv["R"], lam=sv["lam"],
                LK=LK, LC=max(LK - D, 0.0), LK_minus_LC=D,
                valid=bool(sv["valid"]))


def run_speed(kn, tol_deg=0.01, max_iter=12):
    h, v = make_vessel()
    U = kn * KN
    b = h.b
    t0 = time.perf_counter()
    before = equilibrium(v, U)                   # the plant's own (cold)
    hist = [dict(iter=0, eq=before)]
    eq, ext = before, None
    converged, diverged = False, False
    for it in range(1, max_iter + 1):
        corr, ext = correction(eq, b)
        # continuation: re-solve from the previous equilibrium, so that a
        # small load cannot throw a hump-region case into the other basin
        eq_new = equilibrium(v, U, ext, warm=(eq["z"], eq["th"]))
        hist.append(dict(iter=it, corr=corr, eq=eq_new))
        if not eq_new["finite"]:
            diverged = True
            break
        d_trim = eq_new["trim_deg"] - eq["trim_deg"]
        eq = eq_new
        if abs(d_trim) < tol_deg:
            converged = True
            break
    # the cold start (from rest, as the plant does it) with the final load:
    # does the plant have a second equilibrium there?
    cold = equilibrium(v, U, ext, warm=None) if ext is not None else None
    if diverged:
        return dict(kn=kn, U=U, converged=False, diverged=True,
                    n_iter=len(hist) - 1, before=before, hist=hist,
                    cold_final=cold, sav=savitsky_ref(h, U),
                    wall_s=time.perf_counter() - t0)
    # the correction table at the final attitude (what was used last is
    # hist[-1]['corr']; recomputed here for the report of the converged state)
    corr_final, _ = correction(eq, b)
    corr_wide, _ = correction(eq, b, Ns=(24, 32, 40))
    sav = savitsky_ref(h, U)
    after = eq
    # resistance: the plant's captive force (hull pressure drag from the
    # strip force + friction; air excluded) and the same plus the pressure
    # drag share of the external vertical force, dFz * tan(trim)
    R_after_plant = after["R"]
    R_after_ext = after["R"] + after["ext"][2] * after["tau_mean"]
    out = dict(kn=kn, U=U, converged=converged, diverged=False,
               n_iter=len(hist) - 1, wall_s=time.perf_counter() - t0,
               before=before, after=after, sav=sav, cold_final=cold,
               corr_used=hist[-1]["corr"], corr_final=corr_final,
               corr_final_wide=corr_wide,
               R_after_plant=R_after_plant, R_after_ext=R_after_ext,
               d_trim=after["trim_deg"] - before["trim_deg"],
               pred_d_trim=PRED_DTRIM[kn],
               resid_before=before["trim_deg"] - sav["trim_deg"],
               resid_after=after["trim_deg"] - sav["trim_deg"],
               pred_resid=PRED_RESID[kn],
               lam_before=0.5 * (before["LK"] + before["LC"]) / b,
               lam_after=0.5 * (after["LK"] + after["LC"]) / b,
               hist=hist)
    return out


def _job(kn):
    return run_speed(kn)


def report(rows):
    L = []
    L.append("V3: gravity-free 3-D steady correction on the ZR+MLM calm "
             "equilibria of scarab195 (beam %.3f m), iterated to |d trim| "
             "< 0.01 deg" % 2.025)
    L.append("kn | trim before / after / Savitsky | d_trim got / predicted | "
             "resid after got / predicted | R before / after(plant) / "
             "after(+ext drag) / Sav  [N] | LK before/after (Sav) | lambda "
             "before/after (Sav) | k | dx_cp m | iters")
    div = [r for r in rows if r.get("diverged")]
    rows = [r for r in rows if not r.get("diverged")]
    for r in rows:
        bf, af, sv, c = r["before"], r["after"], r["sav"], r["corr_used"]
        L.append(f"{r['kn']:2d} | {bf['trim_deg']:5.2f} / {af['trim_deg']:5.2f}"
                 f" / {sv['trim_deg']:5.2f} | {r['d_trim']:+5.2f} / "
                 f"{r['pred_d_trim']:+5.2f} | {r['resid_after']:+5.2f} / "
                 f"{r['pred_resid']:+5.2f} | {bf['R']:5.0f} / "
                 f"{r['R_after_plant']:5.0f} / {r['R_after_ext']:5.0f} / "
                 f"{sv['R']:5.0f} | {bf['LK']:.2f}/{af['LK']:.2f} "
                 f"({sv['LK']:.2f}) | {r['lam_before']:.2f}/{r['lam_after']:.2f}"
                 f" ({sv['lam']:.2f}) | {c['k']:.3f} | {c['dx_cp']:+.3f} | "
                 f"{r['n_iter']}{'' if r['converged'] else ' (not conv)'}")
    L.append("")
    L.append("details per speed: L_dyn (momentum + crossflow) and hydrostatic"
             " share before -> after; CP of the plant's dynamic lift aft of "
             "the wetted apex vs Jones and exact (beams); LC; VLM "
             "convergence (k at 16/24/32 panels, Richardson order; wide "
             "triple 24/32/40)")
    for r in rows:
        bf, af, c, cw = r["before"], r["after"], r["corr_used"], \
            r["corr_final_wide"]
        raw = " ".join(f"{q['k']:.3f}" for q in c["vlm"]["raw"])
        raww = " ".join(f"{q['k']:.3f}" for q in cw["vlm"]["raw"])
        L.append(f"{r['kn']:2d} kn: L_dyn {bf['L_dyn']:6.0f} -> {af['L_dyn']:6.0f}"
                 f" N (mom {af['L_mom']:.0f}, cf {af['L_cf']:.0f}), hydrostatic "
                 f"{bf['L_buoy']:5.0f} -> {af['L_buoy']:5.0f} N, ext Fz "
                 f"{af['ext'][2]:+6.0f} N, ext M {af['ext'][4]:+7.0f} N m, "
                 f"balance {af['balance']:+.1f} N; LC {bf['LC']:.2f} -> "
                 f"{af['LC']:.2f} m; x_cp,dyn from apex: plant "
                 f"{c['xcp_plant_from_apex_b']:.3f} b, Jones "
                 f"{c['xcp_jones_from_apex_b']:.3f} b, exact "
                 f"{c['xcp_exact_from_apex_b']:.3f} b; k raw [{raw}] order "
                 f"{c['vlm']['order_L']:.2f} -> {c['k']:.3f}; wide [{raww}] "
                 f"order {cw['vlm']['order_L']:.2f} -> {cw['k']:.3f}, dx_cp "
                 f"{c['dx_cp']:+.3f} / {cw['dx_cp']:+.3f} m; wall "
                 f"{r['wall_s']:.0f} s")
    L.append("")
    L.append("iteration history (trim deg [settling oscillation of pitch "
             "over the last 1 s, deg], k, dx_cp m); 'cold' = the plant's "
             "own start from rest with the final load (second basin?)")
    for r in rows + div:
        seq = " -> ".join(
            (f"{hh['eq']['trim_deg']:.3f}" if hh['eq']['finite'] else "NaN")
            + (f"[{hh['eq']['diag'].get('osc_th_deg', 0):.3f}]"
               if hh['eq']['finite'] and 'osc_th_deg' in hh['eq']['diag']
               else "")
            + (f"(k {hh['corr']['k']:.3f}, dx {hh['corr']['dx_cp']:+.3f})"
               if 'corr' in hh else "") for hh in r["hist"])
        cold = r.get("cold_final")
        cs = ("" if cold is None else
              (f"; cold {cold['trim_deg']:.3f}"
               f"[{cold['diag'].get('osc_th_deg', 0):.3f}]"
               if cold["finite"] else "; cold NaN"))
        L.append(f"{r['kn']:2d} kn: {seq}{cs}"
                 + (" DIVERGED" if r.get("diverged") else ""))
    return "\n".join(L)


def hook_check():
    """The ext_load hook must leave the default behaviour bit-identical."""
    from studies.exp_planing import fridsma_hull
    from sim.planing_zarnick import ZarnickVessel, ZOPT_ZR
    from sim.test_vessel import Monochromatic
    from sim import config
    hull, V = fridsma_hull("B", tow="cg")
    k = 2 * np.pi / (3.0 * hull.L)
    sea = Monochromatic(np.sqrt(G * k), 0.0555 * hull.b, theta0=np.pi)
    p = ZarnickVessel(hull, sea, dt=0.004, captive_u=V, zopt=dict(ZOPT_ZR))
    s = np.zeros(14)
    s[6] = V
    s[2], s[4] = p.running_attitude(V)
    rec = []
    for i in range(1000):
        s = p.step(s, i * 0.004, 0.0, 0.0)
        s[6] = V
        rec.append(np.r_[s, p.last_cg_acc, p.last_bow_acc])
    traj = np.array(rec)
    h, _ = config.load("scarab195")
    calm = []
    for mlm in (False, True):
        q = ZarnickVessel(h, Monochromatic(1.0, 0.0), dt=0.02,
                          zopt=dict(ZOPT_ZR, mlm=mlm))
        for kn in (20, 40):
            r = q.calm_resistance(kn * KN)
            calm.append([r["R"], r["trim_deg"], r["rise"],
                         r["wetted_length"]])
    calm = np.array(calm)
    out = {}
    ref = os.path.join(OUT, "v3_hook_before.npz")
    if os.path.exists(ref):
        a = np.load(ref)
        out["traj_identical"] = bool(np.array_equal(a["traj"], traj))
        out["calm_identical"] = bool(np.array_equal(a["calm"], calm))
        out["traj_max_abs_diff"] = float(np.abs(a["traj"] - traj).max())
    rec3 = os.path.join(ROOT, "studies", "_cache", "sim_redesign",
                        "step3.json")
    if os.path.exists(rec3):
        d = json.load(open(rec3))
        sav = {(x["name"], x["kn"]): x for x in d["sav"]
               if isinstance(x, dict)}
        # calm_resistance R minus air, as sav_job recorded it
        from sim.test_vessel import Monochromatic as _M
        chk = []
        for i, (mlm, kn) in enumerate([(False, 20), (False, 40), (True, 20),
                                       (True, 40)]):
            q = ZarnickVessel(h, _M(1.0, 0.0), dt=0.02,
                              zopt=dict(ZOPT_ZR, mlm=mlm))
            r = q.calm_resistance(kn * KN)
            rr = sav[("ZRM" if mlm else "ZR", kn)]
            chk.append([r["R"] - r["air"] - rr["R"],
                        r["trim_deg"] - rr["trim"]])
        out["step3_sav_diff_R_trim"] = chk
    return out


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "run"
    if what == "hook":
        print(json.dumps(hook_check(), indent=1))
        sys.exit(0)
    from studies.parallel_runner import run_parallel
    t0 = time.time()
    rows = run_parallel(_job, list(SPEEDS), gb_per_job=0.3, reserve_gb=1.5,
                        max_procs=7)
    bad = [(kn, repr(r)) for kn, r in zip(SPEEDS, rows)
           if not isinstance(r, dict)]
    rows = [r for r in rows if isinstance(r, dict)]
    txt = report(rows)
    if bad:
        txt += "\nfailed: " + "; ".join(f"{k}: {e}" for k, e in bad)
    txt += f"\nwall {time.time() - t0:.0f} s"
    print(txt)
    open(os.path.join(OUT, "v3_3d_correction.txt"), "w").write(txt + "\n")
    json.dump(dict(rows=rows, failed=bad, wall_s=time.time() - t0),
              open(os.path.join(OUT, "v3_3d_correction.json"), "w"),
              indent=1, default=float)
