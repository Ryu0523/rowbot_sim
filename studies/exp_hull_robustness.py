#!/usr/bin/env python3
"""
Robustness: the whole plant on hulls it was not built for.

The other experiments check one method against one measurement. This one asks
a plainer question of the whole chain -- Hull -> mesh -> sections -> BEM
database -> NonlinearVessel -> a run in waves -- on five hulls: does it run,
and are its numbers self-consistent (displacement closes, the vessel floats
level in calm water, nothing blows up in a sea)?

  wigley10    the 10 m USV the project was built on, through the Hull path
  cat2        2 m catamaran: two Wigley demihulls 0.6 m apart
  barge20     20 x 8 x 2 m box barge
  kvlcc2_68   KVLCC2 at 1:68 from its IGES, SSPA loading
  kcs_38      KCS at 1:37.89 from its IGES, T2015 case 2.10 loading

Sea: the USV's SS5 (Hs 3.25 m, Tp 9.7 s) Froude-scaled to each hull, head
seas, 30% of the (scaled) thrust limit. The placeholders -- viscous damping,
actuators -- are the USV's, Froude-scaled; NonlinearVessel says so. This
tests that the chain RUNS and CLOSES, not that a barge behaves like a barge.

Superseded by studies/hull_acceptance.py, which runs on the registered hulls
(hydro/hulls.py) and goes on through the control chain. Kept because DEFECTS
F9 quotes its numbers.

Run: python -m studies.exp_hull_robustness
"""
import os
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KV_IGS = os.path.join(HERE, "data", "external", "kvlcc2_geometry", "kvlcc2.igs")
KCS_IGS = os.path.join(HERE, "data", "external", "kcs_geometry",
                       "KCS_hull_Case2-11", "FinalHull_KCS.igs")
CACHE = os.path.join(HERE, "studies", "_cache")
REF_HS, REF_TP = 3.25, 9.7


def hulls():
    import capytaine as cpt
    from hydro.geometry import wigley_mesh
    from hydro.hull import Hull, wigley_10m
    out = [wigley_10m(name="robust_wigley10", k_roll=0.365, k_pitch=0.25,
                      k_yaw=0.26)]
    demi = wigley_mesh(2.0, 0.2, 0.1, 30, 8)
    cat = demi.translated_y(0.3).join_meshes(demi.translated_y(-0.3))
    out.append(Hull(name="robust_cat2", L=2.0, B=0.8, T=0.1,
                    mesh_kind="object", extras={"mesh": cat}, k_roll=0.35,
                    k_pitch=0.25, k_yaw=0.26))
    barge = cpt.mesh_parallelepiped(size=(20.0, 8.0, 2.0), center=(0, 0, -1.0),
                                    resolution=(24, 10, 4),
                                    missing_sides={"top"})
    out.append(Hull(name="robust_barge20", L=20.0, B=8.0, T=2.0,
                    mesh_kind="object", extras={"mesh": barge}, k_roll=0.35,
                    k_pitch=0.25, k_yaw=0.26))
    s = 1.0 / 68.0
    out.append(Hull(name="robust_kvlcc2_68", L=320 * s, B=58 * s, T=20.8 * s,
                    mesh_kind="cad", mesh_path=KV_IGS,
                    cad=dict(scale=s, forward="-x", up="-z", size_max=8.0,
                             symmetric=True),
                    displacement=1000.0 * 312653.0 * s ** 3, rho=1000.0,
                    z_cog=(18.6 - 20.8) * s, k_roll=0.40, k_pitch=0.25,
                    k_yaw=0.25, freeboard=(30.0 - 20.8) * s))
    s = 6.0702 / 230.0
    out.append(Hull(name="robust_kcs_38", L=6.0702, B=0.8498, T=0.2850,
                    mesh_kind="cad", mesh_path=KCS_IGS,
                    cad=dict(scale=1e-3 * s, half=False, size_max=6000.0,
                             min_component=0.1),
                    displacement=998.63 * 0.9571, rho=998.63,
                    z_cog=0.378 - 0.2850, k_roll=0.40, k_pitch=0.25,
                    k_yaw=0.25, freeboard=(19.0 - 10.8) * s))
    return out


def run_one(h, n_omega=40, n_dir=7, t_calm=20.0, t_sea=60.0):
    from sim.forces import Wind
    from sim.test_vessel import Monochromatic
    from sim.vessel import NonlinearVessel
    from sim.wavefield import SeaState
    lam = h.L / 10.0
    r = dict(name=h.name, lam=lam)
    t0 = time.time()
    rows = h.check(verbose=False)
    r["check"] = rows[2][1]                      # displacement vs lines
    omegas = np.linspace(0.1, 14.5, n_omega) / np.sqrt(lam)
    db = h.database(omegas=omegas, directions=np.linspace(0, np.pi, n_dir),
                    out_dir=CACHE, verbose=False)
    r["t_db"] = time.time() - t0
    dt = 0.05 * np.sqrt(lam)
    # calm water, no thrust, released level: it must stay level
    v = NonlinearVessel(db, Monochromatic(1.0, 0.0), hull=h, dt=dt,
                        wind=Wind())
    s = v.initial_state(0.0)
    for i in range(int(t_calm * np.sqrt(lam) / dt)):
        s = v.step(s, i * dt, 0.0, 0.0)
    r["calm_heave_T"] = s[2] / h.T
    r["calm_pitch_deg"] = np.degrees(s[4])
    # a Froude-scaled SS5, head seas
    sea = SeaState(REF_HS * lam, REF_TP * np.sqrt(lam), theta0=np.pi,
                   n_freq=24, n_dir=4, seed=3)
    v = NonlinearVessel(db, sea, hull=h, dt=dt)
    s = v.initial_state(0.5 * v.prop.u_ref)
    thr = 0.3 * v.prop.t_max
    n = int(t_sea * np.sqrt(lam) / dt)
    hist = np.empty((n, 12))
    for i in range(n):
        s = v.step(s, i * dt, thr, 0.0)
        hist[i] = s[:12]
    r["finite"] = bool(np.all(np.isfinite(hist)))
    tail = hist[n // 2:]
    r["u_over_uref"] = float(np.mean(tail[:, 6]) / v.prop.u_ref)
    r["heave_T"] = float(np.max(np.abs(hist[:, 2])) / h.T)
    r["pitch_deg"] = float(np.degrees(np.max(np.abs(hist[:, 4]))))
    r["roll_deg"] = float(np.degrees(np.max(np.abs(hist[:, 3]))))
    r["yaw_deg"] = float(np.degrees(np.max(np.abs(hist[:, 5]))))
    r["slams"] = v.slam_count
    r["t_total"] = time.time() - t0
    return r


def main():
    import warnings
    warnings.filterwarnings("ignore")
    os.makedirs(CACHE, exist_ok=True)
    print("\nROBUSTNESS -- the whole plant on five hulls\n")
    res = []
    for h in hulls():
        print(f"== {h.name}")
        try:
            res.append(run_one(h))
            print(f"   done in {res[-1]['t_total']:.0f} s")
        except Exception as e:            # report and go on: that is the test
            tb = traceback.extract_tb(e.__traceback__)[-1]
            print(f"   FAILED: {type(e).__name__}: {e}  "
                  f"[{os.path.basename(tb.filename)}:{tb.lineno}]")
            res.append(dict(name=h.name, error=f"{type(e).__name__}: {e}"))
    print(f"\n  {'hull':<18}{'lambda':>7}{'calm z/T':>10}{'calm pitch':>11}"
          f"{'finite':>7}{'u/u_ref':>8}{'|z|/T':>7}{'pitch':>7}{'roll':>7}"
          f"{'yaw':>7}{'slams':>6}")
    for r in res:
        if "error" in r:
            print(f"  {r['name']:<18}  FAILED  {r['error'][:90]}")
            continue
        print(f"  {r['name']:<18}{r['lam']:>7.3f}{r['calm_heave_T']:>10.2e}"
              f"{r['calm_pitch_deg']:>10.3f}d{str(r['finite']):>7}"
              f"{r['u_over_uref']:>8.2f}{r['heave_T']:>7.2f}"
              f"{r['pitch_deg']:>6.1f}d{r['roll_deg']:>6.1f}d"
              f"{r['yaw_deg']:>6.1f}d{r['slams']:>6d}")
    return res


if __name__ == "__main__":
    main()
