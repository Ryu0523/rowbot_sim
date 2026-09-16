#!/usr/bin/env python3
"""
Acceptance battery for ANY vessel in hydro/hulls.py. Run it on your own boat
before trusting a single number from its simulation.

    python -m studies.hull_acceptance                 every registered hull
    python -m studies.hull_acceptance our_boat        just yours

None of these checks compares the simulation with the real vessel -- only
trials can do that. They establish that the simulation is at least a faithful
solution of its OWN equations on this hull, and that nothing in the chain is
still quietly about the 10 m USV:

  A  integrity      Hull.check(): the stated displacement closes against the
                    lines; rudders and propellers are under water
  B  calm float     no waves, no thrust: the vessel settles and stays there.
                    Reports the static sinkage and trim (an LCG away from the
                    LCB trims it)
  C  linear limit   small head waves: heave and pitch of the full nonlinear
                    plant against the frequency-domain solution of the SAME
                    BEM database. Exercises the station table against the BEM
                    hydrostatics, the radiation memory, the excitation and the
                    sign conventions on this hull's geometry. Tolerances are
                    the M4 gate's: 5% heave, 8% pitch
  D  heading        turning the vessel and turning the waves by the same angle
                    must give the same heave, roll and pitch (DEFECTS F4)
  E  propulsion     full thrust in calm water reaches the design speed. If it
                    does not, t_max, the resistance or u_design is inconsistent
  F  manoeuvring    steady turn at full rudder against IMO MSC.137(76) -- a
                    SHIP standard, borrowed: tactical diameter <= 5 L -- and no
                    directional divergence once the rudder is centred
  G  mission sea    the USV's SS5 Froude-scaled to this hull (the same
                    relative severity): finite, no capsize; the speed it costs
  H  control chain  the reduced model identified on this hull, one closed-loop
                    MPC episode in scaled SS4, and the RL environment stepping

Each check is isolated: one that crashes is a FAIL with its error, and the
rest still run. Results go to studies/_cache/acceptance_<hull>.json.
"""
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from sim import config                                          # noqa: E402
from sim.forces import Wind                                     # noqa: E402
from sim.test_vessel import Monochromatic, _run                 # noqa: E402
from sim.cummins import rao_frequency_domain, extract_amplitude_phase  # noqa
from sim.wavefield import SeaState                              # noqa: E402

G = 9.81
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache")


def _amp(t, y, w):
    """Amplitude at w over the second half of the record."""
    i = len(t) // 2
    a, _ = extract_amplitude_phase(t[i:], y[i:], w)
    return float(a)


def _tail(y, frac=0.2):
    return float(np.mean(y[-max(int(frac * len(y)), 1):]))


# ------------------------------------------------------------------ checks
def check_integrity(h, c):
    h.check(verbose=True)                       # raises on a failed invariant
    return True, dict(placeholders=[k for k, _ in h.placeholders()])


def check_calm(h, c):
    plant, _ = config.calm_plant(h, db=c["db"])
    t, s = _run(plant, 80.0 * c["rs"])
    q = s[int(0.75 * len(s)):]
    info = dict(sinkage_over_T=float(q[-1, 2] / h.T),
                trim_deg=float(np.degrees(q[-1, 4])),
                heave_ptp_over_T=float(np.ptp(q[:, 2]) / h.T),
                pitch_ptp_deg=float(np.degrees(np.ptp(q[:, 4]))),
                roll_max_deg=float(np.degrees(np.max(np.abs(q[:, 3])))))
    ok = bool(np.all(np.isfinite(s)) and info["heave_ptp_over_T"] < 1e-3
              and np.ptp(q[:, 4]) < 1e-4 and np.max(np.abs(q[:, 3])) < 1e-4)
    return ok, info


def check_linear(h, c):
    db, rs, lam = c["db"], c["rs"], c["lam"]
    head = config.head_index(db)
    rows = []
    # the M4 gate's frequencies for the USV, Froude-scaled
    for w0 in np.array([0.6, 1.0, 1.4, 1.8]) / rs:
        xi, wa = rao_frequency_domain(db, w0, head)
        amp = 0.02 * lam
        v = config.plant_for(db, Monochromatic(wa, amp), h, dt=0.02 * rs,
                             visc=np.zeros(6), wind=Wind())
        t, s = _run(v, 30 * 2 * np.pi / wa)
        rh, rp = _amp(t, s[:, 2], wa) / amp, _amp(t, s[:, 4], wa) / amp
        rows.append(dict(
            omega=float(wa), lam_over_L=float(2 * np.pi * G / wa ** 2 / h.L),
            heave_fd=float(abs(xi[2])), heave_td=rh,
            pitch_fd=float(abs(xi[4])), pitch_td=rp,
            err_heave=float(abs(rh - abs(xi[2])) / max(abs(xi[2]), 1e-9)),
            err_pitch=float(abs(rp - abs(xi[4])) / max(abs(xi[4]), 1e-9))))
    eh = max(r["err_heave"] for r in rows)
    ep = max(r["err_pitch"] for r in rows)
    return bool(eh < 0.05 and ep < 0.08), dict(worst_heave=eh,
                                               worst_pitch=ep, rows=rows)


def check_heading(h, c, delta=np.radians(60.0)):
    db, rs, lam = c["db"], c["rs"], c["lam"]
    w, a = 1.0 / rs, 0.05 * lam
    amps = []
    # the same relative heading two ways: waves turned, or vessel turned
    for theta0, psi0 in ((np.pi + delta, 0.0), (np.pi, -delta)):
        v = config.plant_for(db, Monochromatic(w, a, theta0=theta0), h,
                             wind=Wind(), captive=("surge", "sway", "yaw"))
        s = v.initial_state(0.0)
        s[5] = psi0
        n = int(30 * 2 * np.pi / w / v.dt)
        rec = np.empty((n, 3))
        for i in range(n):
            s = v.step(s, i * v.dt, 0.0, 0.0)
            rec[i] = s[2], s[3], s[4]
        t = np.arange(n) * v.dt
        amps.append([_amp(t, rec[:, j], w) for j in range(3)])
    A, B = np.array(amps)
    rel = np.abs(A - B) / np.maximum(np.maximum(A, B), 1e-12)
    info = dict(waves_turned=dict(heave=A[0], roll_deg=np.degrees(A[1]),
                                  pitch_deg=np.degrees(A[2])),
                vessel_turned=dict(heave=B[0], roll_deg=np.degrees(B[1]),
                                   pitch_deg=np.degrees(B[2])),
                worst_rel=float(rel.max()))
    return bool(rel.max() < 0.02), info


def check_propulsion(h, c):
    plant, _ = config.calm_plant(h, db=c["db"])
    ud = float(c["sc"]["u_design"])
    t, s = _run(plant, 300.0 * c["rs"], thrust=plant.prop.t_max, u0=ud)
    u_full = _tail(s[:, 6], 0.1)
    return bool(np.isfinite(u_full) and u_full >= ud), dict(
        u_full=u_full, u_design=ud, Fn_full=u_full / np.sqrt(G * h.L),
        t_max=float(plant.prop.t_max))


def check_manoeuvring(h, c):
    db, rs = c["db"], c["rs"]
    ud = float(c["sc"]["u_design"])

    def run(segments):
        v, _ = config.calm_plant(h, db=db)
        thr = 7000.0 * v.prop.t_max / 12000.0
        s, t = v.initial_state(ud), 0.0
        for delta, dur in segments:
            for _ in range(int(dur / v.dt)):
                s = v.step(s, t, thr, delta)
                t += v.dt
        return v, s

    v, s = run([(None, 0.0)]) if False else run([])
    v, s = run([(v.rudder.max, 160.0 * rs)])
    u, vv, r = float(s[6]), float(s[7]), float(s[11])
    R = u / abs(r) if abs(r) > 1e-9 else np.inf
    _, s1 = run([(np.radians(10.0), 20.0 * rs)])
    _, s2 = run([(np.radians(10.0), 20.0 * rs), (0.0, 60.0 * rs)])
    r_kick, r_after = abs(float(s1[11])), abs(float(s2[11]))
    td = 2.2 * R / h.L          # tactical diameter ~ 1.1 x 2 R, as sim/test_manoeuvre
    info = dict(rudder_deg=float(np.degrees(v.rudder.max)),
                radius_over_L=float(R / h.L), tactical_diameter_over_L=td,
                drift_deg=float(np.degrees(np.arctan2(vv, max(u, 1e-9)))),
                speed_in_turn=u, r_kick=r_kick, r_after=r_after)
    return bool(td <= 5.0 and r_after < 0.5 * r_kick), info


def check_mission(h, c):
    db, rs, lam = c["db"], c["rs"], c["lam"]
    ud = float(c["sc"]["u_design"])
    plant, _ = config.calm_plant(h, db=db)
    thr = 7000.0 * plant.prop.t_max / 12000.0
    _, sc_ = _run(plant, 200.0 * rs, thrust=thr, u0=0.8 * ud)
    sea = SeaState(3.25 * lam, 9.7 * rs, n_freq=24, n_dir=5, seed=1)
    v = config.plant_for(db, sea, h)
    _, s = _run(v, 200.0 * rs, thrust=thr, u0=0.8 * ud)
    fin = bool(np.all(np.isfinite(s)))
    info = dict(hs=3.25 * lam, tp=9.7 * rs, finite=fin)
    if fin:
        u_c, u_w = _tail(sc_[:, 6]), _tail(s[:, 6])
        info.update(u_calm=u_c, u_waves=u_w,
                    speed_loss_pct=100.0 * (u_c - u_w) / max(u_c, 1e-9),
                    roll_max_deg=float(np.degrees(np.abs(s[:, 3]).max())),
                    pitch_max_deg=float(np.degrees(np.abs(s[:, 4]).max())),
                    heave_max_over_T=float(np.abs(s[:, 2]).max() / h.T),
                    slams=int(v.slam_count))
    # a vessel past 57 deg of roll or pitch has capsized or pitchpoled
    ok = fin and info["roll_max_deg"] < 57.3 and info["pitch_max_deg"] < 57.3
    return bool(ok), info


def check_control(h, c):
    from control.reduced import ReducedModel
    from sim.env import Episode
    from sim.rl_env import USVControlEnv
    db, rs, lam = c["db"], c["rs"], c["lam"]
    plant, _ = config.calm_plant(h, db=db)
    red = ReducedModel.identify(plant)
    p = red.p
    ep = Episode(db, red, hs=1.88 * lam, tp=8.0 * rs, seed=2, n_freq=16,
                 n_dir=4, hull=h, autopilot=True, use_rudder=False,
                 n_samples=64)
    m = ep.run(60.0 * rs)
    env = USVControlEnv(db=db, hull=h, seeds=(0,), hs=1.88 * lam,
                        tp=8.0 * rs, n_freq=16, n_dir=4)
    o, _ = env.reset(seed=0)
    fin, worst = bool(np.all(np.isfinite(o))), float(np.max(np.abs(o)))
    for _ in range(10):
        o, r, *_ = env.step(np.array([0.2, 0.0]))
        fin = fin and bool(np.all(np.isfinite(o)) and np.isfinite(r))
        worst = max(worst, float(np.max(np.abs(o))))
    info = dict(u_ss_over_u_design=p["u_ss"] / p["u_design"],
                tau_u=p["tau_u"], wn_heave=p["wn_heave"],
                wn_pitch=p["wn_pitch"], k_nomoto=p["k_nomoto"],
                tau_r=p["tau_r"], episode_finite=bool(m["finite"]),
                u_mean_over_u_ref=m["u_mean"] / ep.u_ref,
                heading_rms_deg=float(np.degrees(m["heading_rms"])),
                cross_rms_over_L=m["cross_rms"] / h.L,
                acc_p99_g=m["acc_p99"], rl_env_finite=fin,
                rl_obs_max=worst)
    return bool(m["finite"] and fin), info


CHECKS = [("A", "integrity", check_integrity),
          ("B", "calm float", check_calm),
          ("C", "linear limit vs frequency domain", check_linear),
          ("D", "heading invariance", check_heading),
          ("E", "propulsion reaches design speed", check_propulsion),
          ("F", "manoeuvring", check_manoeuvring),
          ("G", "mission sea, Froude-scaled SS5", check_mission),
          ("H", "control chain", check_control)]

SHOW = {"B": ("sinkage_over_T", "trim_deg", "heave_ptp_over_T",
              "pitch_ptp_deg"),
        "C": ("worst_heave", "worst_pitch"),
        "D": ("worst_rel",),
        "E": ("u_full", "u_design", "Fn_full"),
        "F": ("radius_over_L", "tactical_diameter_over_L", "drift_deg",
              "r_kick", "r_after"),
        "G": ("speed_loss_pct", "roll_max_deg", "pitch_max_deg",
              "heave_max_over_T", "slams"),
        "H": ("u_ss_over_u_design", "tau_u", "k_nomoto", "tau_r",
              "u_mean_over_u_ref", "heading_rms_deg", "cross_rms_over_L",
              "rl_obs_max")}


def _fmt(key, info):
    if "error" in info:
        return info["error"][:300]
    return ", ".join(f"{k} {info[k]:.4g}" if isinstance(info[k], float)
                     else f"{k} {info[k]}" for k in SHOW.get(key, ())
                     if k in info)


def run(name):
    t0 = time.time()
    print(f"\n{'=' * 72}\n  {name}\n{'=' * 72}")
    h, db = config.load(name, verbose=True)
    sc = h.scales()
    c = dict(db=db, sc=sc, lam=sc["lam"], rs=float(np.sqrt(sc["lam"])))
    res = {}
    for key, title, fn in CHECKS:
        t1 = time.time()
        try:
            ok, info = fn(h, c)
        except Exception as exc:                     # noqa: BLE001
            ok, info = False, {"error": f"{type(exc).__name__}: {exc}"}
        res[key] = dict(title=title, ok=bool(ok), info=info,
                        seconds=round(time.time() - t1, 1))
        print(f"  [{'PASS' if ok else 'FAIL'}]  {key} {title}: "
              f"{_fmt(key, info)}")
    if "C" in res and "rows" in res["C"]["info"]:
        print(f"\n      linear limit, head seas:  omega   lam/L   heave fd/td"
              f"          pitch fd/td")
        for r in res["C"]["info"]["rows"]:
            print(f"      {r['omega']:>28.3f}{r['lam_over_L']:>8.2f}"
                  f"{r['heave_fd']:>8.3f}/{r['heave_td']:<7.3f}"
                  f"({100*r['err_heave']:.1f}%) {r['pitch_fd']:>8.4f}/"
                  f"{r['pitch_td']:<8.4f}({100*r['err_pitch']:.1f}%)")
    ph = res["A"]["info"].get("placeholders", [])
    print(f"\n  still placeholders ({len(ph)}): {', '.join(ph) or 'none'}")
    print(f"  {name}: {sum(r['ok'] for r in res.values())}/{len(res)} pass, "
          f"{time.time() - t0:.0f} s")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"acceptance_{name}.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    return res


def main(names=None):
    from hydro import hulls
    names = names or [n for n in hulls.REGISTRY if n != "our_boat"]
    table = {n: run(n) for n in names}
    print(f"\n{'=' * 72}\n  summary\n")
    print("  " + f"{'hull':<12}" + "".join(f"{k:>4}" for k, _, _ in CHECKS))
    for n, res in table.items():
        print("  " + f"{n:<12}" + "".join(
            f"{'ok' if res[k]['ok'] else 'X':>4}" for k, _, _ in CHECKS))
    return table


if __name__ == "__main__":
    main(sys.argv[1:])
