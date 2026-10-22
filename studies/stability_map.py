#!/usr/bin/env python3
"""
Stability map of the frozen plant scarab195@as1 versus speed (control-side
analysis; learn/meta/STABILITY_MAP_2026-10-12.md). Read-only use of sim/,
hydro/, app/. Conventions: studies/stability_lin.py.

    python -m studies.stability_map equil    equilibria, census, Jacobians,
                                             decays at the map speeds
    python -m studies.stability_map curve    thrust-speed curve and the fold
    python -m studies.stability_map sens     randomisation extremes
    python -m studies.stability_map reduced  the MPC's reduced model
    python -m studies.stability_map closed   closed loop of the test
                                             controllers
    python -m studies.stability_map figs     figures
    python -m studies.stability_map report   tables (report.txt)
    python -m studies.stability_map all

Outputs: studies/_cache/stability/*.json, *.png, report.txt.
"""
import json
import math
import os
import sys
import time
import warnings

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np   # noqa: E402

from studies import stability_lin as L   # noqa: E402

OUT = os.path.join(ROOT, "studies", "_cache", "stability")
KN = L.KN
SPEEDS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 11.0, 12.0, 13.0,
          13.3, 13.65, 14.0, 16.0, 20.0, 25.0, 30.0, 35.0, 40.0)
SENS_SPEEDS = (5.0, 12.0, 20.0, 30.0)
SENS_PARAMS = ("mass", "lcg", "vcg", "c_dc", "transom_ft_crit", "roll_zeta",
               "cd_lat", "cda", "rho")
EPS = (1e-5, 1e-4, 1e-3)
DT, DT2 = 0.01, 0.005
MAX_PROCS = 8


def _save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1, default=_plain)


def _plain(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (set, frozenset)):
        return sorted(v)
    raise TypeError(type(v))


def _load(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


def _mem_gb():
    from studies.parallel_runner import free_gb
    return free_gb()


# ================================================================== equil
def decay_run(p, s, dz=0.0, dth=0.0, dv=0.0, dr=0.0, du=0.0, T_end=60.0,
              every=0.25):
    """Free run of the plant from the equilibrium s plus a perturbation,
    constant commands (the equilibrium thrust, nozzle 0). Returns the
    decimated deviation series and a settling summary."""
    s0 = np.array(s, float)
    x = s0.copy()
    x[2] += dz
    x[4] += dth
    x[7] += dv
    x[11] += dr
    x[6] += du
    L.prep(p, s0[12], 0.0)
    dt = p.dt
    n = int(round(T_end / dt))
    k = max(int(round(every / dt)), 1)
    rows = []
    t = 0.0
    for i in range(n + 1):
        if i % k == 0:
            rows.append([t, x[2] - s0[2], L.trim_deg(p, x[4])
                         - L.trim_deg(p, s0[4]), x[6] - s0[6], x[7], x[11],
                         x[3]])
        if i == n:
            break
        x = p.step(x, t, s0[12], 0.0, dt)
        t += dt
        if not np.all(np.isfinite(x)):
            break
    R = np.array(rows)
    last = R[R[:, 0] >= R[-1, 0] - 10.0]

    def pp(col):
        return float(last[:, col].max() - last[:, col].min())

    def at(tt, col):
        i = int(np.argmin(np.abs(R[:, 0] - tt)))
        return float(R[i, col])
    return dict(t_end=float(R[-1, 0]), finite=bool(np.all(np.isfinite(x))),
                pp_z=pp(1), pp_trim=pp(2), pp_u=pp(3), pp_v=pp(4), pp_r=pp(5),
                pp_phi=pp(6),
                z_at={tt: at(tt, 1) for tt in (0, 5, 10, 20, 40, 60)},
                trim_at={tt: at(tt, 2) for tt in (0, 5, 10, 20, 40, 60)},
                u_at={tt: at(tt, 3) for tt in (0, 5, 10, 20, 40, 60)},
                r_at={tt: at(tt, 5) for tt in (0, 5, 10, 20, 40, 60)},
                series=R.tolist())


def speed_job(kn):
    warnings.filterwarnings("ignore")
    t0 = time.time()
    p = L.build(dt=DT)
    u = kn * KN
    e = L.equilibrium(p, u)
    s = np.array(e["s"])
    dres, sres = L.free_residual(p, s)
    cen = L.census(p, u) if u > 0 else []
    # the one-step Jacobians at three perturbation sizes
    P, B = {}, {}
    for eps in EPS:
        P[eps], B[eps] = L.step_jacobian(p, s, eps, dt=DT)
    sw = {eps: L.switch_changes(p, s, eps) for eps in (1e-4, 1e-3)}
    A = L.deriv_jacobian(p, s, 1e-4)
    an = L.analyse(P[1e-4], DT)
    anA = dict(vertical=L.modes(np.linalg.eigvals(L.block(A, L.STATES,
                                                            L.VERT))),
               lateral=L.modes(np.linalg.eigvals(L.block(A, L.STATES,
                                                           L.LAT))))
    # the half step
    p2 = L.build(dt=DT2)
    P2, B2 = L.step_jacobian(p2, s, 1e-4, dt=DT2)
    an2 = L.analyse(P2, DT2)
    ind = L.indicators(p, s)
    sub = p.prop_submergence(s[:6], 0.0)
    # decays: vertical (rise +2 cm, bow-up 0.5 deg) and lateral (sway
    # 0.2 m/s, yaw rate 0.05 rad/s), 60 s each, the thrust state held at
    # its equilibrium command (below the dead band it cannot be: then the
    # command is the dead-band value and the run is marked)
    db = p.prop.dead_band
    held = e["T"] >= db or e["T"] == 0.0
    dec_v = decay_run(p, s, dz=0.02, dth=L.th_of_trim(p, 0.5) - L.th_of_trim(
        p, 0.0))
    dec_l = decay_run(p, s, dv=0.2, dr=0.05)
    # the sway-yaw 2 x 2 of the continuous Jacobian, for the course
    # stability discussion (Nomoto-type criterion)
    iv, ir = L.STATES.index("v"), L.STATES.index("r")
    A2 = A[np.ix_([iv, ir], [iv, ir])]
    out = dict(kn=kn, u=u, eq=e, deriv_resid=dres, step_resid=sres,
               census=cen, dead_band=float(db), held_by_command=bool(held),
               submergence=float(sub), prime_band=float(p.prop.prime_band),
               indicators={k: (sorted(v) if isinstance(v, frozenset) else v)
                           for k, v in ind.items()},
               Phi={str(k): v.tolist() for k, v in P.items()},
               B={str(k): v.tolist() for k, v in B.items()},
               Phi_dt2=P2.tolist(), B_dt2=B2.tolist(), A=A.tolist(),
               linearity={"1e-5 vs 1e-4": L.linearity(P[1e-4], P[1e-5]),
                          "1e-4 vs 1e-3": L.linearity(P[1e-4], P[1e-3])},
               switches={str(k): v for k, v in sw.items()},
               analysis=an, analysis_A=anA, analysis_dt2=an2,
               sway_yaw_A=A2.tolist(),
               sway_yaw_eig=[[float(l.real), float(l.imag)]
                             for l in np.linalg.eigvals(A2)],
               decay_vertical=dec_v, decay_lateral=dec_l,
               seconds=time.time() - t0)
    return out


def part_equil():
    from studies.parallel_runner import run_parallel
    t0 = time.time()
    print(f"equil: {len(SPEEDS)} speeds, free RAM {_mem_gb():.1f} GB",
          flush=True)
    res = run_parallel(speed_job, list(SPEEDS), gb_per_job=0.25,
                       reserve_gb=1.5, max_procs=MAX_PROCS)
    rows = []
    for kn, r in zip(SPEEDS, res):
        if isinstance(r, Exception):
            print(f"  {kn} kn FAILED: {r!r}")
            rows.append(dict(kn=kn, error=repr(r)))
        else:
            e = r["eq"]
            v = r["analysis"]["vertical"]["modes"]
            print(f"  {kn:5.2f} kn  z {e['z']:+.4f} trim {e['trim_deg']:6.3f}"
                  f"  c {e['c']:.5f}  resid {r['deriv_resid']:.1e}/"
                  f"{r['step_resid']:.1e}  roots {len(r['census'])}  modes "
                  + " ".join(f"{m['re']:+.3f}{'+' if m['im'] else ''}"
                             f"{m['im']:.3f}j" if m['im'] else
                             f"{m['re']:+.3f}" for m in v)
                  + f"  ({r['seconds']:.0f} s)", flush=True)
            rows.append(r)
    _save("map.json", dict(speeds=list(SPEEDS), dt=DT, dt2=DT2, eps=list(EPS),
                           states=list(L.STATES), rows=rows,
                           wall_seconds=time.time() - t0))
    print(f"equil done in {time.time() - t0:.0f} s")


# ================================================================== curve
def curve_job(arg):
    """Equilibria along a speed grid by continuation (fsolve + Newton from
    the previous root; the first from the captive run), with the one-step
    Jacobian's vertical-block eigenvalues at each."""
    warnings.filterwarnings("ignore")
    kns = np.asarray(arg["kns"], float)
    t0 = time.time()
    p = L.build(dt=DT)
    rows = []
    guess = None
    for kn in kns:
        u = kn * KN
        e = L.equilibrium(p, u, guess=guess)
        guess = (e["z"], e["th"])
        s = np.array(e["s"])
        Phi, _ = L.step_jacobian(p, s, 1e-4, dt=DT)
        Pv = L.block(Phi, L.STATES, L.VERT)
        lam = L.to_continuous(np.linalg.eigvals(Pv), DT)
        # the slowest real eigenvalue of the vertical block (surge)
        re = [l for l in lam if abs(l.imag) < 1e-9]
        surge = max(re, key=lambda l: l.real) if re else None
        rows.append(dict(kn=float(kn), z=e["z"], th=e["th"],
                         trim_deg=e["trim_deg"], R=e["R"], T=e["T"],
                         c=e["c"], resid=e["resid"], lw=e["lw"],
                         hw=e.get("hw"), n_wet=e["n_wet"],
                         n_chine_wet=e["n_chine_wet"],
                         pressure=e["pressure"], friction=e["friction"],
                         air=e["air"],
                         lam_vert=[[float(l.real), float(l.imag)]
                                   for l in lam],
                         surge_re=(float(surge.real) if surge is not None
                                   else None),
                         max_re=float(lam.real.max())))
    return dict(rows=rows, seconds=time.time() - t0)


def part_curve():
    from studies.parallel_runner import run_parallel
    t0 = time.time()
    coarse = np.round(np.arange(0.5, 45.001, 0.25), 4)
    fine = np.round(np.arange(12.0, 15.001, 0.05), 4)
    chunks = [dict(kns=c.tolist()) for c in np.array_split(coarse, 6)]
    chunks += [dict(kns=c.tolist()) for c in np.array_split(fine, 2)]
    print(f"curve: {len(coarse)} coarse + {len(fine)} fine speeds in "
          f"{len(chunks)} jobs", flush=True)
    res = run_parallel(curve_job, chunks, gb_per_job=0.25, reserve_gb=1.5,
                       max_procs=MAX_PROCS)
    rows_c, rows_f = [], []
    for i, r in enumerate(res):
        if isinstance(r, Exception):
            print(f"  job {i} FAILED: {r!r}")
            continue
        (rows_c if i < 6 else rows_f).extend(r["rows"])
    rows_c.sort(key=lambda r: r["kn"])
    rows_f.sort(key=lambda r: r["kn"])
    fold = fold_analysis(rows_f)
    fold_c = fold_analysis(rows_c, c_step=1e-4)
    print(f"  fine fold: {fold['summary']}")
    print(f"  coarse fold: {fold_c['summary']}")
    _save("curve.json", dict(coarse=rows_c, fine=rows_f, fold_fine=fold,
                             fold_coarse=fold_c,
                             wall_seconds=time.time() - t0))
    print(f"curve done in {time.time() - t0:.0f} s")


def fold_analysis(rows, c_step=1e-5):
    """Where the thrust-speed curve T*(u) (thrust state that holds u) is
    not monotone: intervals with dT*/du < 0, the throttle window with
    three equilibria (count of crossings of T* = c t_max over the grid),
    and the intervals where the surge eigenvalue of the linearisation is
    positive."""
    kn = np.array([r["kn"] for r in rows])
    c = np.array([r["c"] for r in rows])
    sr = np.array([r["surge_re"] if r["surge_re"] is not None else np.nan
                   for r in rows])
    dc = np.diff(c)
    falling = [(float(kn[i]), float(kn[i + 1]), float(dc[i]))
               for i in range(len(dc)) if dc[i] < 0]
    pos = [(float(kn[i]), float(sr[i])) for i in range(len(sr))
           if np.isfinite(sr[i]) and sr[i] > 0]
    lo, hi = c.min(), c.max()
    cs = np.arange(math.floor(lo / c_step) * c_step, hi + c_step, c_step)
    multi = []
    for cc in cs:
        f = c - cc
        idx = np.nonzero(np.sign(f[1:]) != np.sign(f[:-1]))[0]
        if len(idx) > 1:
            roots = [float(kn[i] + f[i] / (f[i] - f[i + 1])
                           * (kn[i + 1] - kn[i])) for i in idx]
            stab = [bool(f[i + 1] > f[i]) for i in idx]   # dT*/du > 0 stable
            multi.append(dict(c=float(cc), roots=roots, stable=stab))
    summ = dict(falling_intervals=falling,
                positive_surge=pos,
                c_window=([multi[0]["c"], multi[-1]["c"]] if multi else None),
                n_c_multi=len(multi),
                example=(multi[len(multi) // 2] if multi else None))
    summ["summary"] = (f"dT*/du < 0 on {len(falling)} interval(s)"
                       + (f" {falling[0][0]:.2f}-{falling[-1][1]:.2f} kn"
                          if falling else "")
                       + f"; surge eigenvalue > 0 at {len(pos)} grid "
                       f"speed(s)"
                       + (f" {pos[0][0]:.2f}-{pos[-1][0]:.2f} kn, max "
                          f"{max(x[1] for x in pos):+.2e} 1/s" if pos else "")
                       + (f"; three equilibria for c in "
                          f"{multi[0]['c']:.5f}..{multi[-1]['c']:.5f}"
                          if multi else "; no c with several equilibria"))
    return summ


# =================================================================== sens
def sens_job(arg):
    warnings.filterwarnings("ignore")
    t0 = time.time()
    from hydro import hulls
    from sim import physics
    from app.randomize import hull_for, Randomization
    name, which = arg["param"], arg["which"]
    base = hulls.get("scarab195")
    rz = Randomization()
    if which == "nom":
        hull, applied, value = hulls.get(L.HULL), {}, None
    else:
        prm = getattr(rz, name)
        value = prm.lo if which == "lo" else prm.hi
        hull, applied, _ = hull_for(base, "as1", {name: value},
                                    name=f"scarab195~{name}_{which}")
    p = physics.gz_twin(hull.plant(L.calm(), dt=DT))
    rows = []
    for kn in SENS_SPEEDS:
        u = kn * KN
        e = L.equilibrium(p, u)
        s = np.array(e["s"])
        P4, B4 = L.step_jacobian(p, s, 1e-4, dt=DT)
        P3, _ = L.step_jacobian(p, s, 1e-3, dt=DT)
        an = L.analyse(P4, DT)
        an3 = L.analyse(P3, DT)
        # the slope of the thrust-speed curve across +-0.1 kn: the
        # quasi-static surge stability (dT*/du > 0 stable), independent of
        # any switch the Jacobian's perturbation may straddle
        du = 0.1 * KN
        ep = L.equilibrium(p, u + du, guess=(e["z"], e["th"]))
        em = L.equilibrium(p, u - du, guess=(e["z"], e["th"]))
        dc_du = (ep["c"] - em["c"]) / (2 * du)
        rows.append(dict(kn=kn, eq={k: v for k, v in e.items() if k != "s"},
                         analysis=an, analysis_1e3=an3, dc_du=float(dc_du),
                         linearity=L.linearity(P4, P3),
                         switches=L.switch_changes(p, s, 1e-3),
                         Phi=P4.tolist(), B=B4.tolist()))
    return dict(param=name, which=which, value=value, applied=applied,
                hull_fields=dict(mass=hull.mass, lcg=hull.lcg, vcg=hull.vcg,
                                 cda=hull.cda, rho=hull.rho, c_dc=hull.c_dc,
                                 roll_zeta=hull.roll_zeta,
                                 cd_lat=hull.cd_lat,
                                 transom_closure=getattr(
                                     hull, "transom_closure", None)),
                plant=dict(T=float(p.T), GM=float(p.GM),
                           Mtot=np.diag(p.Mtot).tolist(),
                           t_max=float(p.prop.t_max)),
                rows=rows, seconds=time.time() - t0)


def part_sens():
    from studies.parallel_runner import run_parallel
    t0 = time.time()
    jobs = [dict(param="nominal", which="nom")]
    for prm in SENS_PARAMS:
        jobs += [dict(param=prm, which="lo"), dict(param=prm, which="hi")]
    print(f"sens: {len(jobs)} hulls x {len(SENS_SPEEDS)} speeds, free RAM "
          f"{_mem_gb():.1f} GB", flush=True)
    res = run_parallel(sens_job, jobs, gb_per_job=0.25, reserve_gb=1.5,
                       max_procs=MAX_PROCS)
    out = []
    for j, r in zip(jobs, res):
        if isinstance(r, Exception):
            print(f"  {j} FAILED: {r!r}")
            out.append(dict(**j, error=repr(r)))
        else:
            print(f"  {j['param']:>16} {j['which']}  value "
                  f"{r['value']}  ({r['seconds']:.0f} s)", flush=True)
            out.append(r)
    _save("sens.json", dict(speeds=list(SENS_SPEEDS), params=list(SENS_PARAMS),
                            jobs=out, wall_seconds=time.time() - t0))
    print(f"sens done in {time.time() - t0:.0f} s")


# ================================================================ reduced
def part_reduced():
    from studies import stability_ctrl as C
    t0 = time.time()
    m = _load("map.json")
    out = C.reduced_comparison(m)
    out["wall_seconds"] = time.time() - t0
    _save("reduced.json", out)
    for r in out["rows"]:
        print(f"  {r['kn']:5.2f} kn  plant heave/pitch pairs "
              f"{r['plant_pairs']}  surge tau {r['plant_surge_tau']:.1f} s"
              f" (reduced {r['reduced_surge_tau']:.1f} s)  nomoto K plant "
              f"{r['plant_K_nomoto']:+.3f} reduced {r['reduced_K_nomoto']:+.3f}")
    print(f"reduced done in {time.time() - t0:.0f} s")


# ================================================================= closed
def part_closed():
    from studies import stability_ctrl as C
    t0 = time.time()
    m = _load("map.json")
    red = _load("reduced.json")
    out = C.closed_loop(m, red["params"])
    out["check"] = C.closed_loop_check(red["params"], kns=(5.0, 12.0, 20.0,
                                                            30.0))
    out["wall_seconds"] = time.time() - t0
    _save("closed.json", out)
    for r in out["rows"]:
        print(f"  {r['kn']:5.2f} kn  speed loop: slowest {r['speed']['slowest']}"
              f"  GM {r['speed']['gain_margin_db']}  PM "
              f"{r['speed']['phase_margin_deg']} | heading: slowest "
              f"{r['heading']['slowest']}  GM {r['heading']['gain_margin_db']}"
              f"  PM {r['heading']['phase_margin_deg']}")
    print(f"closed done in {time.time() - t0:.0f} s")


# =================================================================== figs
def part_figs():
    from studies import stability_figs as F
    F.make_all(OUT)


def part_report():
    from studies import stability_figs as F
    F.report(OUT)


PARTS = dict(equil=part_equil, curve=part_curve, sens=part_sens,
             reduced=part_reduced, closed=part_closed, figs=part_figs,
             report=part_report)

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    args = sys.argv[1:] or ["all"]
    if args == ["all"]:
        args = ["equil", "curve", "sens", "reduced", "closed", "figs",
                "report"]
    for a in args:
        PARTS[a]()
