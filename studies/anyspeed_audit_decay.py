#!/usr/bin/env python3
"""
Companion to studies/anyspeed_audit.py: does the linearisation agree with
the plant's own time stepping? Captive calm runs from the fsolve equilibrium
plus a heave offset dz (tiny and finite), plant.step at its own dt; the
heave history is fitted with the linear modes' prediction implicitly by
reporting successive peak ratios.
Output: studies/_cache/anyspeed/audit_decay.txt
"""
import os
import numpy as np
from studies.anyspeed_audit import build, KN, OUT, equilibrium


def run(kn, dz, guess, T=8.0):
    u = kn * KN
    h, p = build(captive_u=u)
    sol, r = equilibrium(p, u, guess, only=guess is not None)
    s = np.zeros(14)
    s[2], s[4], s[6] = sol[0] + dz, sol[1], u
    n = int(T / p.dt)
    zz = np.empty(n)
    for i in range(n):
        s = p.step(s, i * p.dt, 0.0, 0.0)
        s[6] = u
        zz[i] = s[2] - sol[0]
    pk = [i for i in range(1, n - 1) if zz[i] > zz[i - 1] and zz[i] >= zz[i + 1]]
    amps = [zz[i] for i in pk][:4]
    tp = [i * p.dt for i in pk][:4]
    return sol, amps, tp, zz


def main():
    L = []
    for kn, guess in ((4, (0.0, 0.0)), (6, (0.0, 0.0)), (25, None)):
        if guess is None:
            from studies.anyspeed_audit import build as b
            _, pp = b()
            guess = pp.running_attitude(kn * KN)
        for dz in (0.001, 0.05):
            sol, amps, tp, zz = run(kn, dz, guess)
            rat = [amps[i + 1] / amps[i] for i in range(len(amps) - 1)
                   if amps[i] != 0]
            L.append(f"{kn:3d} kn dz {dz:.3f}: first peaks {np.round(amps, 5)} "
                     f"at {np.round(tp, 2)} s; ratios {np.round(rat, 3)}; "
                     f"min z-dev {zz.min():+.4f}")
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(OUT, "audit_decay.txt"), "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
