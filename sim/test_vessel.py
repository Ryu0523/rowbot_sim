#!/usr/bin/env python3
"""
M4 GATE -- the nonlinear vessel, checked against things that are already known.

  1. calm water        no waves, no thrust: the vessel sits at its equilibrium
                       and stays there. Catches a mis-balanced weight/buoyancy
                       or a stray force.
  2. linear limit      in a small regular wave the heave and pitch RAOs must
                       match the frequency-domain solution that M3 already
                       reproduced. This is the check that validates the whole
                       nonlinear strip integral, and it also exercises the
                       pitch sign convention, which C55 alone could not settle.
  3. propulsion        steady thrust in calm water reaches a steady speed.
  4. added resistance  the SAME thrust in waves reaches a LOWER speed. Without
                       this the sea is free to sail through and any controller
                       that slows down looks worse than it is.
  5. ventilation       thrust falls as the propeller approaches the surface.
  6. robustness        a long SS5 run stays finite and bounded.
"""
import numpy as np

from hydro import bem
from .vessel import NonlinearVessel, LIMITATIONS
from .wavefield import SeaState
from .cummins import rao_frequency_domain, extract_amplitude_phase

DB = "hydro_wigley_10m.npz"
HEAD = 12          # direction index for beta = pi


class Monochromatic(SeaState):
    """Single regular wave, so the model can be compared against an exact RAO."""

    def __init__(self, omega, amplitude, theta0=np.pi, hs=None, tp=None):
        self.a = np.array([amplitude])
        self.w = np.array([omega])
        self.th = np.array([theta0])
        self.k = self.w ** 2 / 9.81
        self.phi = np.array([0.0])
        self.hs = 4 * amplitude / np.sqrt(2) if hs is None else hs
        self.tp = 2 * np.pi / omega if tp is None else tp
        self.theta0 = theta0


def _run(ves, t_end, thrust=0.0, rudder=0.0, u0=0.0, dt=None):
    dt = ves.dt if dt is None else dt
    s = ves.initial_state(u0)
    n = int(t_end / dt)
    out = np.empty((n + 1, len(s)))
    out[0] = s
    for i in range(n):
        s = ves.step(s, i * dt, thrust, rudder, dt)
        out[i + 1] = s
    return np.arange(n + 1) * dt, out


def main():
    db = bem.load(DB)
    results = {}

    # ---- 1. calm water equilibrium ------------------------------------
    calm = Monochromatic(1.0, 0.0)
    ves = NonlinearVessel(db, calm, L=db.L, dt=0.02)
    t, s = _run(ves, 30.0)
    drift = np.max(np.abs(s[len(s) // 2:, 2]))
    pitch_drift = np.max(np.abs(s[len(s) // 2:, 4]))
    print(f"[1] calm water: |heave| <= {drift:.2e} m, "
          f"|pitch| <= {np.degrees(pitch_drift):.2e} deg")
    results["calm"] = drift < 1e-3 and pitch_drift < 1e-4

    # ---- 2. linear limit: RAO vs the exact frequency-domain answer -----
    print(f"\n[2] small-amplitude RAO vs frequency domain (a = 0.02 m)")
    print(f"  {'omega':>7}{'heave fd':>10}{'heave nl':>10}{'err%':>7}"
          f"{'pitch fd':>10}{'pitch nl':>10}{'err%':>7}")
    errs_h, errs_p = [], []
    for w in (0.6, 1.0, 1.4, 1.8):
        amp = 0.02
        sea = Monochromatic(w, amp)
        v = NonlinearVessel(db, sea, L=db.L, dt=0.01,
                            visc=np.zeros(6))       # linear model has none
        xi, wa = rao_frequency_domain(db, w, HEAD)
        t, s = _run(v, 45 * 2 * np.pi / wa, dt=0.01)
        ah, _ = extract_amplitude_phase(t, s[:, 2], wa)
        ap, _ = extract_amplitude_phase(t, s[:, 4], wa)
        rh, rp = ah / amp, ap / amp
        eh = abs(rh - abs(xi[2])) / max(abs(xi[2]), 1e-9)
        ep = abs(rp - abs(xi[4])) / max(abs(xi[4]), 1e-9)
        errs_h.append(eh); errs_p.append(ep)
        print(f"  {wa:>7.2f}{abs(xi[2]):>10.4f}{rh:>10.4f}{100*eh:>7.2f}"
              f"{abs(xi[4]):>10.4f}{rp:>10.4f}{100*ep:>7.2f}")
    results["linear_heave"] = max(errs_h) < 0.05
    results["linear_pitch"] = max(errs_p) < 0.08

    # ---- 3/4. propulsion and added resistance -------------------------
    thrust = 3000.0
    ves_c = NonlinearVessel(db, Monochromatic(1.0, 0.0), L=db.L, dt=0.05)
    t, sc = _run(ves_c, 200.0, thrust=thrust)
    u_calm = float(np.mean(sc[-400:, 6]))

    sea5 = SeaState(3.25, 9.7, n_freq=32, n_dir=6, seed=1)
    ves_w = NonlinearVessel(db, sea5, L=db.L, dt=0.05)
    t, sw = _run(ves_w, 200.0, thrust=thrust)
    u_wave = float(np.mean(sw[-400:, 6]))
    loss = 100 * (u_calm - u_wave) / max(u_calm, 1e-9)
    print(f"\n[3] calm-water steady speed at {thrust:.0f} N: {u_calm:.2f} m/s")
    print(f"[4] same thrust in SS5:            {u_wave:.2f} m/s "
          f"({loss:.1f}% speed loss)")
    results["speed"] = 0.5 < u_calm < 20.0
    results["added_resistance"] = 0.5 < loss < 80.0

    # ---- 5. ventilation ------------------------------------------------
    p = ves_w.prop
    subs = [0.6, 0.33, 0.15, 0.0, -0.1]
    facs = [p.ventilation_factor(x) for x in subs]
    print(f"\n[5] ventilation factor vs submergence: "
          + ", ".join(f"{a:.2f}m->{b:.2f}" for a, b in zip(subs, facs)))
    results["ventilation"] = (facs[0] == 1.0 and facs[-1] == 0.0
                              and all(np.diff(facs) <= 1e-12))

    # ---- 6. robustness -------------------------------------------------
    finite = np.all(np.isfinite(sw))
    bounded = np.max(np.abs(sw[:, 2])) < 10.0 and np.max(np.abs(sw[:, 4])) < 1.0
    print(f"\n[6] SS5 200 s run: finite={finite}, "
          f"max|heave|={np.max(np.abs(sw[:,2])):.2f} m, "
          f"max|pitch|={np.degrees(np.max(np.abs(sw[:,4]))):.1f} deg, "
          f"slams={ves_w.slam_count}")
    results["robust"] = bool(finite and bounded)

    names = {
        "calm": "calm-water equilibrium holds",
        "linear_heave": "heave RAO matches linear theory within 5%",
        "linear_pitch": "pitch RAO matches linear theory within 8%",
        "speed": "steady speed reached under constant thrust",
        "added_resistance": "waves cost speed (added resistance active)",
        "ventilation": "ventilation monotone, 1 when deep, 0 when broached",
        "robust": "SS5 run finite and bounded",
    }
    print("\n  " + "-" * 52)
    for k, v in results.items():
        print(f"  [{'PASS' if v else 'FAIL'}]  {names[k]}")
    print("  " + "-" * 52)
    ok = all(results.values())
    print(f"  M4 NONLINEAR VESSEL GATE: {'PASSED' if ok else 'FAILED'}")
    print("\n  carried limitations:")
    for l in LIMITATIONS:
        print(f"    - {l}")
    return ok


if __name__ == "__main__":
    main()
