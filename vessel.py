#!/usr/bin/env python3
"""
Reduced-order longitudinal seakeeping model for PREVIEW-VALUE studies.

Deliberately not a validated hull. Its only job is to answer "how much is
preview information worth?", which depends on getting the STRUCTURE right
(encounter frequency, size-effect filtering, resonance, slamming) rather than
the coefficients. Replace with a Cummins model from Capytaine output in Phase 2.

States : [x, u, z, zdot, th, thdot, emerged]
         th > 0 is BOW UP.
Control: commanded speed u_cmd, tracked through a first-order thrust lag.

Wave forcing uses the Froude-Krylov "size effect": the hull responds to wave
elevation AVERAGED over its waterline and to the wave SLOPE fitted over the
same stations. That is what makes hull length matter -- a 10 m boat barely
notices a 15 m wave and simply contours a 120 m one.
"""
import numpy as np

G = 9.81
NS = 7          # state dimension


class Hull:
    """Parameters for a small semi-displacement USV."""

    def __init__(self, L=10.0, draft=0.8, n_stations=5,
                 wz=3.0, zeta_z=0.40,      # heave natural freq / damping
                 wth=2.4, zeta_th=0.35,    # pitch natural freq / damping
                 tau_u=3.0,                # speed command lag (s)
                 u_min=1.0, u_max=9.0):
        self.L, self.draft = L, draft
        self.stations = np.linspace(-L / 2, L / 2, n_stations)   # +ve forward
        self.wz, self.zeta_z = wz, zeta_z
        self.wth, self.zeta_th = wth, zeta_th
        self.tau_u = tau_u
        self.u_min, self.u_max = u_min, u_max
        self.v_slam = 0.093 * np.sqrt(G * L)     # Ochi relative-velocity threshold


def step(hull, s, u_cmd, eta_st, eta_dot_bow, dt):
    """Advance an ENSEMBLE of states one step, vectorised over rollouts.

    s            : (N, 7)
    u_cmd        : (N,) commanded speed
    eta_st       : (N, n_stations) surface elevation at each hull station
    eta_dot_bow  : (N,) vertical water velocity at the bow -- NOT negligible,
                   it is the same order as the slam threshold itself
    returns (next_state, a_bow, slam_event)
    """
    x, u, z, zd, th, thd, emg = s.T
    half = hull.L / 2

    # --- wave forcing via size-effect filtering -------------------------
    eta_bar = eta_st.mean(axis=1)
    xs_c = hull.stations - hull.stations.mean()
    slope = (eta_st * xs_c).sum(axis=1) / (xs_c ** 2).sum()
    alpha_w = np.arctan(slope)                    # effective wave slope, bow-up +ve

    zdd = hull.wz ** 2 * (eta_bar - z) - 2 * hull.zeta_z * hull.wz * zd
    thdd = hull.wth ** 2 * (alpha_w - th) - 2 * hull.zeta_th * hull.wth * thd

    u_cmd = np.clip(u_cmd, hull.u_min, hull.u_max)
    ud = (u_cmd - u) / hull.tau_u

    # --- bow kinematics and slamming (Ochi: emergence + fast re-entry) ---
    a_bow = zdd + half * thdd
    z_bow = z + half * th
    zd_bow = zd + half * thd
    rel = z_bow - eta_st[:, -1]                   # bow above local surface
    rel_dot = zd_bow - eta_dot_bow                # relative vertical velocity
    emerged = (rel > hull.draft)
    # EVENT, not a per-timestep condition: keel re-enters with high closing speed
    slam = ((emg > 0.5) & (~emerged) & (rel_dot < -hull.v_slam)).astype(float)

    # semi-implicit (symplectic) Euler -- stable over long runs
    ns = np.empty_like(s)
    ns[:, 1] = u + ud * dt
    ns[:, 3] = zd + zdd * dt
    ns[:, 5] = thd + thdd * dt
    ns[:, 0] = x + ns[:, 1] * dt
    ns[:, 2] = z + ns[:, 3] * dt
    ns[:, 4] = th + ns[:, 5] * dt
    ns[:, 6] = emerged.astype(float)
    return ns, a_bow, slam


def init_state(n, u0):
    s = np.zeros((n, NS))
    s[:, 1] = u0
    return s
