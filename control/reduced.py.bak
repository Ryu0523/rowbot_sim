#!/usr/bin/env python3
"""
Reduced prediction model for the MPC, IDENTIFIED from the verified plant.

The plant is 110+ states. Rolling that forward for a few hundred MPPI samples
at every control step is not affordable, and it is not necessary: the MPC needs
the right timescales, not the right hydrodynamics. So it carries a small model

    surge   first-order thrust response plus quadratic drag and added resistance
    heave   second-order oscillator driven by the wave surface under the hull
    pitch   second-order oscillator driven by the wave slope along the hull
    yaw     first-order (Nomoto) response to rudder

and every coefficient in it is fitted to the M3/M4 model rather than guessed.
That is the difference between this and the Phase 0 toy: the numbers now come
from the hull, so when the hull changes the controller model follows.

Identification uses the same three experiments a real vessel would:
a thrust step, a free decay, and a regular-wave sweep.
"""
import numpy as np

G = 9.81


class ReducedModel:
    """State: [x, y, u, z, zdot, th, thdot, psi, r].

    y is carried so cross-track error is a real quantity in the cost rather
    than a placeholder; sway is neglected, so ydot = u sin(psi).
    """

    NS = 9

    def __init__(self, p):
        self.p = p

    @classmethod
    def identify(cls, plant, db, verbose=False):
        from sim.test_vessel import Monochromatic, _run
        from sim.vessel import NonlinearVessel, SIGN_PITCH

        L = plant.L

        # --- surge: thrust step in calm water -> gain and time constant ----
        calm = Monochromatic(1.0, 0.0)
        v = NonlinearVessel(db, calm, L=L, dt=0.05)
        t, s = _run(v, 240.0, thrust=8000.0)
        u = s[:, 6]
        u_ss = float(np.mean(u[-200:]))
        # first-order rise: u(t) = u_ss (1 - exp(-t/tau))
        idx = np.argmax(u > 0.632 * u_ss)
        tau_u = float(t[idx]) if idx > 0 else 8.0
        k_drag = 8000.0 / max(u_ss ** 2, 1e-6)      # steady drag balance

        # --- heave / pitch: free decay -> natural frequency and damping ----
        wn, zeta = {}, {}
        for dof, col, vcol in (("heave", 2, 8), ("pitch", 4, 10)):
            s0 = v.initial_state()
            s0[col] = 0.5 if dof == "heave" else 0.08
            n = int(30.0 / 0.05)
            y = np.empty(n + 1)
            st = s0.copy()
            for i in range(n):
                st = v.step(st, i * 0.05, 0.0, 0.0, 0.05)
                y[i + 1] = st[col]
            y[0] = s0[col]
            tt = np.arange(n + 1) * 0.05
            zc = np.where(np.diff(np.sign(y)))[0]
            T = 2 * np.mean(np.diff(tt[zc])) if len(zc) > 2 else 2.0
            wn[dof] = 2 * np.pi / T
            pk = [i for i in range(1, n) if y[i] > y[i - 1] and y[i] > y[i + 1]
                  and y[i] > 0.02 * abs(s0[col])]
            if len(pk) >= 2:
                dlt = np.log(abs(y[pk[0]] / y[pk[1]]))
                zeta[dof] = float(dlt / np.sqrt(4 * np.pi ** 2 + dlt ** 2))
            else:
                zeta[dof] = 0.3

        # --- yaw: rudder step -> Nomoto gain and time constant -------------
        st = v.initial_state(u_ss)
        n = int(60.0 / 0.05)
        r = np.empty(n + 1)
        r[0] = 0.0
        for i in range(n):
            st = v.step(st, i * 0.05, 8000.0, np.radians(20.0), 0.05)
            r[i + 1] = st[11]
        r_ss = float(np.mean(r[-100:]))
        k_nomoto = r_ss / np.radians(20.0)
        idx = np.argmax(np.abs(r) > 0.632 * abs(r_ss)) if abs(r_ss) > 1e-9 else 0
        tau_r = float(idx * 0.05) if idx > 0 else 3.0

        p = dict(L=L, T=plant.T, tau_u=tau_u, k_drag=k_drag,
                 wn_heave=wn["heave"], z_heave=zeta["heave"],
                 wn_pitch=wn["pitch"], z_pitch=zeta["pitch"],
                 k_nomoto=k_nomoto, tau_r=max(tau_r, 0.5),
                 sign_pitch=SIGN_PITCH,
                 v_slam=0.093 * np.sqrt(G * L),
                 draft=plant.T, u_ss=u_ss, u_max=u_ss * 1.35,
                 m_surge=max(2.0 * k_drag * u_ss * tau_u, 1e-6))
        if verbose:
            print("  identified reduced model:")
            for k, val in p.items():
                print(f"    {k:>12} = {val:.4g}")
        return cls(p)

    # ------------------------------------------------------------ dynamics
    def step(self, s, thrust, rudder, eta_stations, x_st, dt):
        """Vectorised over rollouts. s is (N, 8); eta_stations is (N, n_st)."""
        p = self.p
        x, y, u, z, zd, th, thd, psi, r = s.T
        eta_bar = eta_stations.mean(axis=1)
        xc = x_st - x_st.mean()
        slope = (eta_stations * xc).sum(axis=1) / max((xc ** 2).sum(), 1e-12)

        # Linearising m_eff*udot = thrust - k u^2 about u_ss gives a first-order
        # time constant tau = m_eff / (2 k u_ss), so m_eff = 2 k u_ss tau.
        # Using u_max here instead of u_ss made the modelled surge response
        # about 30% too fast.
        drag = p["k_drag"] * u * np.abs(u)
        ud = (thrust - drag) / p["m_surge"]

        wz, zz = p["wn_heave"], p["z_heave"]
        wp, zp = p["wn_pitch"], p["z_pitch"]
        zdd = wz ** 2 * (eta_bar - z) - 2 * zz * wz * zd
        alpha = np.arctan(slope) * p["sign_pitch"]
        thdd = wp ** 2 * (alpha - th) - 2 * zp * wp * thd
        rd = (p["k_nomoto"] * rudder - r) / p["tau_r"]

        ns = np.empty_like(s)
        ns[:, 2] = u + ud * dt
        ns[:, 4] = zd + zdd * dt
        ns[:, 6] = thd + thdd * dt
        ns[:, 8] = r + rd * dt
        ns[:, 7] = psi + ns[:, 8] * dt
        ns[:, 0] = x + ns[:, 2] * np.cos(ns[:, 7]) * dt
        ns[:, 1] = y + ns[:, 2] * np.sin(ns[:, 7]) * dt
        ns[:, 3] = z + ns[:, 4] * dt
        ns[:, 5] = th + ns[:, 6] * dt
        a_bow = zdd + p["sign_pitch"] * (p["L"] / 2) * thdd
        z_bow = z + p["sign_pitch"] * (p["L"] / 2) * th
        rel = z_bow - eta_stations[:, -1]
        return ns, a_bow, rel

    def from_plant_state(self, s_plant):
        """Project the full plant state onto the reduced state."""
        eta, nu = s_plant[:6], s_plant[6:12]
        return np.array([eta[0], eta[1], nu[0], eta[2], nu[2],
                         eta[4], nu[4], eta[5], nu[5]])
