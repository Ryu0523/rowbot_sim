#!/usr/bin/env python3
"""
Supplement to studies/derive_v1_linear_consistency.py: the heave decay from
a 1 cm offset analysed properly. A half-cycle log decrement is not a good
description of a heavily damped system with radiation memory, so here the
decay record is fitted by least squares with a damped sinusoid
    z(t) = A exp(-zeta wn t) cos(wd t + phi) + z_inf,   wd = wn sqrt(1 - zeta^2)
over the first 2.5 s, for (a) heave and pitch free and (b) pitch locked
(heave alone, comparable to the uncoupled frequency-domain values wn 3.95
rad/s, zeta 0.296 of ledger B1.2/B1.10). Configurations: D1' (ZR, MLM),
pure BEM LTI, plain strip.

Also: at 5 kn the ZR strip's residual at a = 2 cm jumped (part 3): this
script lists which stations leave the chine-wetted state during that
prescribed heave and the size of the momentum-term jump there.

Output: studies/_cache/derive/v1_decay.txt / .json
"""
import json
import math
import os
import time

import numpy as np
from scipy.optimize import least_squares

import studies.derive_v1_linear_consistency as m

OUT = m.OUT
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


class PitchLocked(m.Model2):
    def rhs(self, t, y):
        y = np.array(y, float)
        y[1] = 0.0
        y[3] = 0.0
        dy = super().rhs(t, y)
        dy[1] = 0.0
        dy[3] = 0.0
        return dy


def fit_decay(t, z, t_fit=2.5):
    k = t <= t_fit
    tt, zz = t[k], z[k]

    def model(p):
        A, zeta, wn, phi, zinf = p
        wd = wn * math.sqrt(max(1 - zeta ** 2, 1e-9))
        return A * np.exp(-zeta * wn * tt) * np.cos(wd * tt + phi) + zinf

    p0 = [zz[0], 0.3, 3.9, 0.0, 0.0]
    r = least_squares(lambda p: model(p) - zz, p0,
                      bounds=([-1, 0.0, 0.5, -np.pi, -0.05], [1, 0.99, 20.0, np.pi, 0.05]))
    A, zeta, wn, phi, zinf = r.x
    rms = float(np.sqrt(np.mean((model(r.x) - zz) ** 2)))
    return dict(zeta=float(zeta), wn=float(wn), wd=float(wn * math.sqrt(1 - zeta ** 2)),
                rms_fit=rms, z_inf=float(zinf))


def main():
    t0 = time.time()
    res = {}
    strips, Js = {}, {}
    for lab, mlm in (("ZR", False), ("MLM", True)):
        v = m.make_vessel(mlm, 0.0)
        s = m.Strip(v, 0.0)
        strips[lab] = s
        Js[lab] = m.jacobian(lambda xi: s.dyn(xi)[0], np.zeros(6), m.STEPS)[0]
    bem = m.load_bem()
    fits = m.fit_memory(bem)
    mem = m.Memory2(fits)
    om_tab, A, B = bem["omega"], bem["A"], bem["B"]
    v = strips["ZR"].v
    Chs = -m.jacobian(lambda p: m.exact_hydrostatics(strips["ZR"], p[0], p[1]),
                      np.zeros(2), [1e-6, 1e-6])[0]
    wn = 3.9
    for _ in range(30):
        wn = math.sqrt(Chs[0, 0] / (v.m + np.interp(wn, om_tab, A[:, 2, 2])))
    zeta_fd = np.interp(wn, om_tab, B[:, 2, 2]) / (2 * wn * (v.m + np.interp(wn, om_tab, A[:, 2, 2])))
    say(f"frequency-domain uncoupled heave (BEM A33(w), B33(w), exact C33): wn {wn:.3f} rad/s, "
        f"zeta {zeta_fd:.3f}")
    # strip-only uncoupled: wn = sqrt(C33 / (m + A33_strip)), no damping except cross-flow
    for lab, s in strips.items():
        A33s = s.added_mass_matrix(0, 0)[0, 0]
        say(f"  plain strip {lab} uncoupled heave: wn {math.sqrt(Chs[0, 0] / (v.m + A33s)):.3f} rad/s "
            f"(added mass {A33s:.0f} kg), no linear damping")
    res["fd"] = dict(wn=wn, zeta=zeta_fd)
    configs = [("D1'", True, True), ("pure BEM LTI", False, True), ("plain strip", True, False)]
    dt = 0.005
    for locked in (False, True):
        say(f"\n== heave decay from +1 cm, {'pitch LOCKED (heave alone)' if locked else 'heave and pitch free'} "
            f"== damped-sinusoid fit over the first 2.5 s")
        say(f"  {'config':22s}{'wn rad/s':>10}{'zeta':>8}{'wd rad/s':>10}{'fit rms cm':>11}  first extrema (cm)")
        for lab, s in strips.items():
            for cname, us, ub in configs:
                if not us and lab != "ZR":
                    continue
                key = cname if not us else f"{cname} {lab}"
                cls = PitchLocked if locked else m.Model2
                mdl = cls(s, mem, Js[lab], us, ub)
                y0 = np.zeros(4 + mdl.nm)
                y0[0] = 0.01
                t, rec = mdl.run(y0, 10.0, dt)
                f = fit_decay(t, rec[:, 0])
                _, _, ext = m.decay_stats(t, rec[:, 0])
                res[f"{key}|{'locked' if locked else 'free'}"] = f
                say(f"  {key:22s}{f['wn']:10.3f}{f['zeta']:8.3f}{f['wd']:10.3f}{100 * f['rms_fit']:11.4f}  "
                    + ", ".join(f"{100 * e[1]:+.3f}" for e in ext[:4]))
    # chine-dry excursion at 5 kn, ZR, prescribed heave 2 cm at 3.24 rad/s
    say("\n== 5 kn, ZR: which stations leave the chine-wetted state during a prescribed 2 cm heave ==")
    u = 5.0 * m.KNOT
    v5 = m.make_vessel(False, u)
    z0, th0 = v5.running_attitude(u)
    s5 = m.Strip(v5, u)
    worst = None
    for a in (0.02, 0.01):
        n_dry_max, mom_max = 0, 0.0
        for z in np.linspace(-a, a, 81):
            xi = np.array([z0 + z, th0, 0, 0, 0, 0])
            d = s5.immersion(xi[0], xi[1])[0]
            cw = 0.5 * np.pi * np.maximum(d, 0) / v5.tanb
            dry = (cw < v5.yc) & (d > 0)
            n_dry_max = max(n_dry_max, int(dry.sum()))
            _, aux = s5.dyn(xi)
            mom_max = max(mom_max, float(np.max(aux["mom"])))
        say(f"  a = {100 * a:.0f} cm: up to {n_dry_max} stations chine-dry during the cycle; max momentum-term "
            f"force {mom_max:.1f} N/m (zero at the running state: every station chine-wetted, d m_a/dd frozen to 0)")
        res[f"chine_dry_5kn_a{a}"] = dict(n_dry_max=n_dry_max, mom_max=mom_max)
    d0 = s5.immersion(z0, th0)[0]
    cw0 = 0.5 * np.pi * d0 / v5.tanb
    margin = (cw0 - v5.yc)
    i = int(np.argmin(margin[d0 > 0]))
    say(f"  smallest margin to chine-dry at the running state: station x = {v5.x[i]:+.3f} m, immersion "
        f"{d0[i]:.3f} m, Wagner half-beam {cw0[i]:.3f} m vs chine {v5.yc[i]:.3f} m -> heave rise of "
        f"{margin[i] * v5.tanb / (0.5 * np.pi):.4f} m makes it chine-dry")
    say(f"\ncompute time {time.time() - t0:.0f} s")
    with open(os.path.join(OUT, "v1_decay.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, "v1_decay.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
