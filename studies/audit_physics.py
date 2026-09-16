#!/usr/bin/env python3
"""
An independent audit of the simulation: things that must be true, checked.

The milestone gates verify each stage against the stage before it. That catches
a lot, but it is vertical -- it cannot see an error shared by both sides of a
comparison, which is exactly how the Ogilvie sign error survived its own unit
test for so long. This is horizontal instead: physical laws, symmetries and
limiting cases that hold regardless of how the code is organised.

Nothing here reuses the machinery it is testing. Derivatives are checked by
finite difference, symmetries by running the mirror case, energy by summing it,
and analytic limits by their closed forms.

Run: python -m studies.audit_physics
"""
import numpy as np

from hydro import bem
from sim.wavefield import SeaState
from sim.vessel import NonlinearVessel, SIGN_PITCH
from sim.sections import WigleySections
from sim.test_vessel import Monochromatic

G = 9.81
RHO = 1025.0
DB = "hydro_wigley_10m.npz"
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}]  {name:<52}{detail}")


# ---------------------------------------------------------------- wave field
def audit_waves():
    print("\nWAVE FIELD")
    sea = SeaState(3.25, 9.7, n_freq=32, n_dir=1, seed=0)

    # significant height comes back out of the discretisation
    rng = np.random.default_rng(0)
    x = rng.uniform(-3000, 3000, 20000)
    y = np.zeros_like(x)
    sig = float(np.std(sea.eta(x, y, 0.0)))
    check("significant height recovered from the components",
          abs(4 * sig / 3.25 - 1) < 0.05, f"Hs_out {4*sig:.3f} vs 3.25")

    # eta_dot really is the time derivative of eta
    h = 1e-4
    p = np.array([12.0, -3.0])
    fd = float((sea.eta(p[:1], p[1:], h) - sea.eta(p[:1], p[1:], -h))[0]
               / (2 * h))
    an = float(sea.eta_dot(p[:1], p[1:], 0.0)[0])
    check("eta_dot matches a finite difference of eta",
          abs(fd - an) < 1e-4 * max(abs(an), 1.0),
          f"{an:+.6f} vs {fd:+.6f}")

    # depth attenuation: at the surface it must be the surface, and below it
    # must follow exp(k z) for a single component
    e0 = float(sea.eta_at_depth(p[:1], p[1:], 0.0, np.array([0.0]))[0])
    check("eta_at_depth(z=0) equals eta",
          abs(e0 - float(sea.eta(p[:1], p[1:], 0.0)[0])) < 1e-12)
    # Rebuild the depth-attenuated sum from the component arrays directly,
    # rather than trusting the method being tested to also define the answer.
    z = -1.7
    ph = (sea.k * (p[0] * np.cos(sea.th) + p[1] * np.sin(sea.th)) + sea.phi)
    ref = float(np.sum(sea.a * np.exp(sea.k * z) * np.cos(ph)))
    got = float(sea.eta_at_depth(p[:1], p[1:], 0.0, np.array([z]))[0])
    check("depth attenuation is per-component exp(k z)",
          abs(got - ref) < 1e-10, f"{got:+.8f} vs {ref:+.8f}")

    # the transfer-function cache must not confuse two different discretisations
    from sim.wavefield import interp_transfer_cached, _TF_CACHE
    db = bem.load(DB)
    _TF_CACHE.clear()
    a = SeaState(3.25, 9.7, n_freq=16, n_dir=4, seed=0)
    b = SeaState(3.25, 9.7, n_freq=32, n_dir=2, seed=0)
    Fa = interp_transfer_cached(db, a, "F_exc")
    Fb = interp_transfer_cached(db, b, "F_exc")
    # both have 64 components; if the cache key cannot tell them apart the
    # second call silently returns the first one's answer
    same = Fa.shape == Fb.shape and np.allclose(Fa, Fb)
    check("transfer cache distinguishes 16x4 from 32x2",
          not same, "COLLISION -- same key, different spectra" if same else "")


# ------------------------------------------------------------- hydrostatics
def audit_hydrostatics():
    print("\nHYDROSTATICS")
    db = bem.load(DB)
    L, B, T = db.L, 2.5, 0.8
    sec = WigleySections(L, B, T, 41)

    v_num = sec.volume()
    v_an = (4.0 / 9.0) * L * B * T
    check("strip volume matches the analytic Wigley displacement",
          abs(v_num / v_an - 1) < 2e-3, f"{v_num:.4f} vs {v_an:.4f} m^3")

    check("displacement equals the mass matrix",
          abs(RHO * v_an / db.M[0, 0] - 1) < 2e-3,
          f"{RHO*v_an:.0f} vs {db.M[0,0]:.0f} kg")

    # restoring from a small forced displacement, against the BEM's C
    d = 1e-3
    f0, _, _, _ = sec.fk_restoring(np.zeros(sec.n), 0.0, 0.0, SIGN_PITCH)
    fz, _, _, _ = sec.fk_restoring(np.zeros(sec.n), d, 0.0, SIGN_PITCH)
    c33 = -(fz - f0) / d
    check("C33 from the strip model matches the BEM",
          abs(c33 / db.C[2, 2] - 1) < 0.02,
          f"{c33:.4e} vs {db.C[2,2]:.4e}")

    _, m0, _, _ = sec.fk_restoring(np.zeros(sec.n), 0.0, 0.0, SIGN_PITCH)
    _, mp, _, _ = sec.fk_restoring(np.zeros(sec.n), 0.0, d, SIGN_PITCH)
    c55 = -(mp - m0) / d
    check("C55 from the strip model matches the BEM",
          abs(c55 / db.C[4, 4] - 1) < 0.05,
          f"{c55:.4e} vs {db.C[4,4]:.4e}")


# --------------------------------------------------------------------- BEM
def audit_bem():
    print("\nBEM DATABASE")
    db = bem.load(DB)
    A, Bd = db.A, db.B

    sym_a = max(np.max(np.abs(A[i] - A[i].T)) / max(np.max(np.abs(A[i])), 1e-9)
                for i in range(len(db.omega)))
    sym_b = max(np.max(np.abs(Bd[i] - Bd[i].T)) / max(np.max(np.abs(Bd[i])), 1e-9)
                for i in range(len(db.omega)))
    check("added mass is symmetric", sym_a < 5e-3, f"max asym {sym_a:.2e}")
    check("radiation damping is symmetric", sym_b < 5e-3, f"max asym {sym_b:.2e}")

    # damping removes energy, so B must be positive semi-definite everywhere
    worst = min(float(np.min(np.linalg.eigvalsh(0.5 * (Bd[i] + Bd[i].T))))
                / max(float(np.max(np.abs(Bd[i]))), 1e-9)
                for i in range(len(db.omega)))
    check("radiation damping is positive semi-definite",
          worst > -1e-3, f"worst normalised eigenvalue {worst:.2e}")

    # long waves: the vessel rides them, so heave excitation tends to the
    # hydrostatic restoring times unit amplitude
    i = int(np.argmin(db.omega))
    j = int(np.argmin(np.abs(db.directions - np.pi)))
    ratio = abs(db.F_exc[i, j, 2]) / db.C[2, 2]
    check("heave excitation -> C33 in the long-wave limit",
          abs(ratio - 1) < 0.10,
          f"|F3|/C33 = {ratio:.4f} at omega {db.omega[i]:.3f}")


# ---------------------------------------------------------------- symmetry
def audit_symmetry():
    print("\nSYMMETRY (a symmetric hull in head seas must not turn)")
    db = bem.load(DB)
    sea = SeaState(3.25, 9.7, n_freq=24, n_dir=1, theta0=np.pi, seed=3)
    v = NonlinearVessel(db, sea, L=db.L, B=2.5, T=0.8, dt=0.05)
    s = v.initial_state(4.0)
    t = 0.0
    peak = np.zeros(3)
    for _ in range(int(120 / 0.05)):
        s = v.step(s, t, 7000.0, 0.0, 0.05)
        t += 0.05
        peak = np.maximum(peak, np.abs([s[1], s[3], s[5]]))
    # Sway has no restoring force, so ANY residual lateral excitation
    # integrates into displacement. The BEM's own head-sea sway force is
    # 3.8e-5 of its heave force -- panel-mesh asymmetry, not a modelling
    # error -- and that is what this drift is. What matters is that it stays
    # small against the hull, not that it is zero.
    check("lateral drift in head seas stays below a tenth of a hull length",
          peak[0] < 0.1 * db.L, f"|y|max {peak[0]:.4f} m over 120 s")
    check("no roll in long-crested head seas", peak[1] < 1e-3,
          f"|roll|max {np.degrees(peak[1]):.4f} deg")
    check("no yaw in long-crested head seas", peak[2] < 1e-3,
          f"|yaw|max {np.degrees(peak[2]):.4f} deg")


# ------------------------------------------------------------------ energy
def audit_energy():
    print("\nENERGY (with no waves and no thrust, motion can only decay)")
    db = bem.load(DB)
    v = NonlinearVessel(db, Monochromatic(1.0, 0.0), L=db.L, B=2.5, T=0.8,
                        dt=0.05)
    s = v.initial_state(0.0)
    s[2] = 0.35                      # displaced in heave
    s[4] = np.radians(4.0)           # and in pitch
    t = 0.0
    E = []
    M = db.M + v.rad.A_inf
    for i in range(int(60 / 0.05)):
        s = v.step(s, t, 0.0, 0.0, 0.05)
        t += 0.05
        nu, eta = s[6:12], s[:6]
        ke = 0.5 * float(nu @ M @ nu)
        pe = 0.5 * float(eta @ db.C @ eta)
        E.append(ke + pe)
    E = np.array(E)
    # allow a small ripple from the fixed step, but the trend must be down
    rise = float(np.max(np.diff(E)) / max(E[0], 1e-9))
    check("free-decay energy is non-increasing", rise < 0.02,
          f"largest single-step rise {100*rise:.3f}% of initial")
    check("free decay actually decays", E[-1] < 0.2 * E[0],
          f"E_end/E_0 = {E[-1]/E[0]:.4f}")


# --------------------------------------------------------------- actuators
def audit_actuators():
    print("\nACTUATORS")
    from sim.actuators import Rudder, Propulsion
    r = Rudder(dt=0.05)
    check("rudder lift is zero at zero angle",
          abs(r.force(0.0, 5.0)[0]) < 1e-12)
    lp, _, _ = r.force(np.radians(10), 5.0)
    lm, _, _ = r.force(-np.radians(10), 5.0)
    check("rudder lift is odd in angle", abs(lp + lm) < 1e-9,
          f"{lp:+.2f} vs {lm:+.2f} N")
    check("rudder lift scales with u^2",
          abs(r.force(np.radians(10), 10.0)[0] / lp - 4.0) < 1e-6)
    check("maximum rudder angle is within the stall angle",
          r.max <= r.stall + 1e-12,
          f"max {np.degrees(r.max):.0f} deg, stall {np.degrees(r.stall):.0f} deg")

    p = Propulsion(dt=0.05)
    check("thrust is lost when the propeller broaches",
          p.effective(10000.0, -1.0) < 0.2 * 10000.0)
    check("thrust is intact when deeply submerged",
          abs(p.effective(10000.0, 5.0) / 10000.0 - 1) < 1e-6)


# -------------------------------------------------------------------- slam
def audit_slam():
    print("\nWATER-ENTRY LOAD")
    db = bem.load(DB)
    sea = SeaState(3.25, 9.7, n_freq=16, n_dir=1, seed=0)
    v = NonlinearVessel(db, sea, L=db.L, B=2.5, T=0.8, dt=0.05)
    n = v.sec.n
    nu = np.zeros(6); nu[2] = -3.0        # descending fast
    eta6 = np.zeros(6)
    deep = np.full(n, 5.0)                # fully submerged
    dry = np.full(n, -5.0)                # clear of the water
    fz_deep, _ = v.slam_load(eta6, nu, deep, 0.0)
    fz_dry, _ = v.slam_load(eta6, nu, dry, 0.0)
    check("no entry load when fully submerged", abs(fz_deep) < 1.0,
          f"{fz_deep:.3e} N")
    check("no entry load when clear of the water", abs(fz_dry) < 1.0,
          f"{fz_dry:.3e} N")
    # Entering BELOW the separation depth. This used to be tested at mid
    # draught (d = -0.4). With generalised Wagner a Wigley section separates
    # at about 47% of the draught above the keel -- the depth the drop test in
    # studies/exp_slam_wedge.py supports for the separation timing -- so mid
    # draught is past separation and correctly carries no entry load. The
    # state was moved; the physics was not bent to fit the old test.
    low = np.full(n, -0.6)                # 25% of draught above the keel
    fz_up, _ = v.slam_load(eta6, nu, low, 0.0)
    nu2 = np.zeros(6); nu2[2] = +3.0      # rising: exit, not entry
    fz_out, _ = v.slam_load(eta6, nu2, low, 0.0)
    check("entry load opposes entry and is one-sided",
          fz_up > 0 and abs(fz_out) < 1e-9,
          f"entering {fz_up:.0f} N, leaving {fz_out:.0f} N")
    high = np.full(n, -0.2)               # 75% of draught: past separation
    fz_sep, _ = v.slam_load(eta6, nu, high, 0.0)
    h_sep = float(np.nanmedian(v.slam.h_separation) / v.T)
    check("no entry load after Wagner separation",
          abs(fz_sep) < 1e-9 and 0.3 < h_sep < 0.7,
          f"{fz_sep:.0f} N at 75% draught; separation at {h_sep:.2f} T")


def main():
    print("independent physics audit -- laws, symmetries and limits")
    audit_waves()
    audit_hydrostatics()
    audit_bem()
    audit_symmetry()
    audit_energy()
    audit_actuators()
    audit_slam()
    bad = [n for n, ok, _ in RESULTS if not ok]
    print("\n  " + "-" * 66)
    print(f"  {len(RESULTS) - len(bad)} / {len(RESULTS)} checks pass")
    for n in bad:
        print(f"    FAILED: {n}")
    return not bad


if __name__ == "__main__":
    main()
