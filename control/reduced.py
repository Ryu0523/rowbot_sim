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
from scipy.linalg import expm

G = 9.81


def _fit_wave_horizontal(plant, L, B, n_st=5, t_end=180.0, dt=None,
                         seeds=(0, 1, 2)):
    """Regress the plant's unexplained sway/yaw acceleration on the wave slope.

    The reduced model already accounts for rudder lift, sway damping and the
    Coriolis term. What it cannot account for is the sea pushing the hull
    sideways and slewing it, and in short-crested waves that is most of what
    happens to heading. So: run the plant with the rudder locked (no lift, no
    Coriolis worth speaking of), measure v_dot and r_dot directly, and fit

        v_dot_residual = c_sway * g * (transverse slope)
        r_dot          = c_yaw  * g * (twist of that slope along the hull)

    Least squares through the origin, pooled over seeds. Returning zero when
    the fit is degenerate is deliberate: a wrong coefficient here is worse than
    none, because it would make the controller act at the wrong phase.
    """
    from sim.wavefield import SeaState
    g = 9.81
    rs = np.sqrt(L / 10.0)
    dt = plant.dt if dt is None else dt
    x_st = np.linspace(plant.sec.x_stern, plant.sec.x_bow, n_st)
    xc = x_st - x_st.mean()
    den = max((xc ** 2).sum(), 1e-12)
    hb = 0.5 * B
    A_s, b_s, A_y, b_y = [], [], [], []
    for sd in seeds:
        sea = SeaState(3.25, 9.7, n_freq=24, n_dir=5, seed=sd)
        v = plant.with_sea(sea)             # the plant's own vessel
        s = v.initial_state(4.0 * rs)
        t = 0.0
        prev = None
        for i in range(int(t_end * rs / dt)):
            s = v.step(s, t, 7000.0 * plant.prop.t_max / 12000.0, 0.0, dt)
            t += dt
            if i % 4:
                continue
            psi = s[5]
            ch, sh = np.cos(psi), np.sin(psi)
            xs = s[0] + xc[:, None] * ch - np.array([-hb, 0, hb])[None, :] * sh
            ys = s[1] + xc[:, None] * sh + np.array([-hb, 0, hb])[None, :] * ch
            e = sea.eta(xs.ravel(), ys.ravel(), t).reshape(n_st, 3)
            tsl = (e[:, 2] - e[:, 0]) / (2 * hb)
            t_slope = float(tsl.mean())
            t_twist = float((tsl * xc).sum() / den)
            if prev is not None:
                dtt = 4 * dt
                vdot = (s[7] - prev[0]) / dtt
                rdot = (s[11] - prev[1]) / dtt
                # subtract what the reduced model already explains in sway
                known = (-plant.visc[1] * prev[0] * abs(prev[0])
                         - plant.M[0, 0] * prev[2] * prev[1]) / plant.Mtot[1, 1]
                A_s.append(g * t_slope); b_s.append(vdot - known)
                A_y.append(g * t_twist); b_y.append(rdot)
            prev = (float(s[7]), float(s[11]), float(s[6]))
    A_s = np.array(A_s); b_s = np.array(b_s)
    A_y = np.array(A_y); b_y = np.array(b_y)
    c_s = float(A_s @ b_s / max(A_s @ A_s, 1e-12)) if A_s.size else 0.0
    c_y = float(A_y @ b_y / max(A_y @ A_y, 1e-12)) if A_y.size else 0.0
    return c_s, c_y


class ReducedModel:
    """State: [x, y, u, z, zdot, th, thdot, psi, r].

    y is carried so cross-track error is a real quantity in the cost rather
    than a placeholder; sway is neglected, so ydot = u sin(psi).
    """

    NS = 9

    def __init__(self, p):
        self.p = p
        self._phi_cache = {}

    @classmethod
    def identify(cls, plant, db=None, verbose=False):
        """Fit the reduced model to `plant` with three calm-water experiments.

        On the plant's OWN vessel: a calm-water clone of it (with_sea), not a
        vessel built here. This used to build NonlinearVessel(db, calm, L=L),
        which took the Wigley's sections and B = 2.5, T = 0.8 whatever the
        plant was, so the surge, heave, pitch and yaw of the reduced model came
        from one boat and its sway from another.

        Test inputs are fractions of the plant's own limits and the windows
        multiples of its own time scale, sqrt(L / 10 m) -- for the 10 m USV
        exactly the old 8 kN step, 240 s, 0.5 m release and so on. As absolute
        numbers they left a tanker model barely moving in 240 s and released a
        2 m boat 4 draughts above the water.
        """
        from sim.forces import Wind
        from sim.test_vessel import Monochromatic, _run
        from sim.vessel import SIGN_PITCH

        L = plant.L
        rs = np.sqrt(L / 10.0)                       # time scale vs the USV
        dt = plant.dt
        t_max = plant.prop.t_max
        # multiplied before dividing, so the USV gets exactly its old 8000 N
        thr_id = 8000.0 * t_max / 12000.0
        u_des = plant.prop.u_ref

        # A FRESH calm-water vessel for every experiment. A plant carries
        # actuator state that initial_state() does not reset -- the thruster's
        # transport-delay line -- and the decays below used to start with the
        # surge step's commands still in it: a kilonewton-scale push during a
        # "free" decay, and on KVLCC2 the push that started a blow-up.
        calm = Monochromatic(1.0, 0.0)

        def fresh():
            return plant.with_sea(calm, wind=Wind())

        # --- surge: thrust step in calm water -> gain and time constant ----
        v = fresh()
        t, s = _run(v, 240.0 * rs, thrust=thr_id)
        u = s[:, 6]
        u_ss = float(np.mean(u[-200:]))
        # first-order rise: u(t) = u_ss (1 - exp(-t/tau))
        idx = np.argmax(u > 0.632 * u_ss)
        tau_u = float(t[idx]) if idx > 0 else 8.0 * rs
        k_drag = thr_id / max(u_ss ** 2, 1e-6)      # steady drag balance

        # --- heave / pitch: free decay -> natural frequency and damping ----
        wn, zeta = {}, {}
        for dof, col, vcol in (("heave", 2, 8), ("pitch", 4, 10)):
            v = fresh()
            s0 = v.initial_state()
            s0[col] = 0.625 * plant.T if dof == "heave" else 0.08
            n = int(30.0 * rs / dt)
            y = np.empty(n + 1)
            st = s0.copy()
            for i in range(n):
                st = v.step(st, i * dt, 0.0, 0.0, dt)
                y[i + 1] = st[col]
            y[0] = s0[col]
            tt = np.arange(n + 1) * dt
            # From the equilibrium the vessel settles to (a trimmed hull's is
            # not zero), and only while the oscillation is still there. The
            # period used to average EVERY zero crossing of the record, the
            # numerical tail's included, and the Wigley's heave frequency
            # moved 3.27 -> 3.58 -> 3.26 rad/s for changes that did not touch
            # its physics (DEFECTS G5).
            y = y - np.median(y[3 * len(y) // 4:])
            thr = 0.02 * abs(y[0])
            live = np.nonzero(np.abs(y) > thr)[0]
            end = int(live[-1]) if len(live) else n
            zc = [i for i in np.nonzero(np.diff(np.sign(y)))[0] if i < end]
            tc = np.array([tt[i] - y[i] * dt / (y[i + 1] - y[i]) for i in zc])
            pk = [i for i in range(1, n) if y[i] > y[i - 1] and y[i] > y[i + 1]
                  and y[i] > thr]
            if len(pk) >= 2:
                dlt = np.log(abs(y[pk[0]] / y[pk[1]]))
                zeta[dof] = float(dlt / np.sqrt(4 * np.pi ** 2 + dlt ** 2))
            else:
                zeta[dof] = 0.3
            # crossings give the DAMPED period; the model's wn is undamped
            Td = 2.0 * float(np.mean(np.diff(tc))) if len(tc) > 2 else 2.0 * rs
            wn[dof] = 2 * np.pi / Td / np.sqrt(max(1.0 - zeta[dof] ** 2, 0.05))

        # --- yaw: rudder step -> Nomoto gain and time constant -------------
        v = fresh()
        st = v.initial_state(u_ss)
        n = int(60.0 * rs / dt)
        r = np.empty(n + 1)
        r[0] = 0.0
        for i in range(n):
            st = v.step(st, i * dt, thr_id, np.radians(20.0), dt)
            r[i + 1] = st[11]
        r_ss = float(np.mean(r[-100:]))
        k_nomoto = r_ss / np.radians(20.0)
        idx = np.argmax(np.abs(r) > 0.632 * abs(r_ss)) if abs(r_ss) > 1e-9 else 0
        tau_r = float(idx * dt) if idx > 0 else 3.0 * rs

        # Sway coefficients are COPIED, not fitted. Fitting would add error to
        # terms the plant computes in closed form and that we can simply read.
        rud = plant.rudder
        # Mass, lift and Coriolis are read straight off the plant. Damping is
        # NOT: the plant also damps sway through the radiation memory, which is
        # linear and which this model has no way to carry. Copying only the
        # quadratic viscous term left the sway response 57% too large. So the
        # LINEAR part is identified from a steady rudder step, by asking what
        # coefficient closes the plant's own force balance:
        #
        #     lift = k_lin v  +  k_quad v|v|  +  m u r
        #
        # Everything but k_lin is known, so this is one division, not a fit.
        m_sway = float(plant.Mtot[1, 1])
        k_quad = float(plant.visc[1])
        m_cor = float(plant.M[0, 0])
        k_lift = float(0.5 * rud.rho * rud.area * rud.cl_alpha)

        v = fresh()
        st = v.initial_state(u_des)
        d_id = np.radians(15.0)
        tt = 0.0
        for _ in range(int(90.0 * rs / dt)):
            st = v.step(st, tt, 7000.0 * t_max / 12000.0, d_id, dt)
            tt += dt
        u_s, v_s, r_s = float(st[6]), float(st[7]), float(st[11])
        lift_s = k_lift * u_s * abs(u_s) * d_id
        k_lin = (lift_s - k_quad * v_s * abs(v_s) - m_cor * u_s * r_s) / v_s
        k_lin = float(max(k_lin, 0.0))

        p_sway = dict(m_sway=m_sway, k_sway=k_quad, m_coriolis=m_cor,
                      k_lift=k_lift, k_lin_sway=k_lin,
                      rud_stall=float(rud.stall))

        # --- wave-induced sway force and yaw moment ------------------------
        # Fitted, not guessed: run the plant in short-crested seas with the
        # rudder locked, and regress the part of its sway and yaw acceleration
        # that the reduced model does NOT already explain onto g*t_slope and
        # g*t_twist. One least-squares solve each.
        # Fitted and found worthless -- R^2 of 0.004 and 0.000. Pinned to zero
        # rather than left in: a coefficient that explains nothing would still
        # make the controller act, at a phase the fit could not determine, and
        # a wrong phase is worse than silence. `_fit_wave_horizontal` is kept
        # so the result can be reproduced.
        c_ws, c_wy = 0.0, 0.0

        p = dict(L=L, T=plant.T, B=float(plant.B),
                 c_wave_sway=c_ws, c_wave_yaw=c_wy, tau_u=tau_u, k_drag=k_drag,
                 wn_heave=wn["heave"], z_heave=zeta["heave"],
                 wn_pitch=wn["pitch"], z_pitch=zeta["pitch"],
                 k_nomoto=k_nomoto, tau_r=max(tau_r, 0.5 * rs),
                 sign_pitch=SIGN_PITCH,
                 v_slam=0.093 * np.sqrt(G * L),
                 draft=plant.T, u_ss=u_ss, u_max=u_ss * 1.35,
                 m_surge=max(2.0 * k_drag * u_ss * tau_u, 1e-6),
                 # the plant's own limits and extent, for the MPC: it used
                 # 12 kN, 35 deg and +-L/2 about the origin whatever the plant
                 t_max=float(t_max), rud_max=float(plant.rudder.max),
                 u_design=float(u_des), dt_ctrl=0.5 * rs,
                 x_bow=float(plant.sec.x_bow),
                 x_stern=float(plant.sec.x_stern),
                 # the bow station's own keel, as the plant's slam criterion
                 draft_bow=float(-plant.sec.keel[-1]),
                 **p_sway)
        if verbose:
            print("  identified reduced model:")
            for k, val in p.items():
                print(f"    {k:>12} = {val:.4g}")
        return cls(p)

    # ------------------------------------------------------------ dynamics
    def _phi(self, wn, zeta, dt):
        """Exact one-step transition for  e_ddot = -wn^2 e - 2 zeta wn e_dot.

        Heave and pitch are LINEAR second-order systems, so their discrete
        transition over a fixed step is available in closed form and is
        unconditionally stable. That matters here because the control step is
        0.5 s while wn is 3.1 (heave) and 3.6 (pitch) rad/s, so wn*dt is 1.5 to
        1.8 -- far outside where explicit Euler is stable on an oscillator.

        This was a real defect, not a refinement. With forward Euler the
        rollout amplified error by sqrt(1 + (wn dt)^2) ~ 1.8-2.1 PER STEP, and
        `studies.model_horizon` measured the consequence: prediction error grew
        to 10^12 times the signal across the MPC's own 12 s horizon, and the
        correlation with the plant fell below 0.5 after 1.0 s. The controller
        was optimising against noise for 90% of its horizon, which made every
        preview result meaningless -- a controller limit masquerading as a
        statement about physics.

        Cached per (wn, zeta, dt); there are only ever two or three.
        """
        key = (round(float(wn), 9), round(float(zeta), 9), round(float(dt), 9))
        hit = self._phi_cache.get(key)
        if hit is None:
            A = np.array([[0.0, 1.0], [-wn ** 2, -2.0 * zeta * wn]])
            hit = expm(A * dt)
            self._phi_cache[key] = hit
        return hit

    def step(self, s, thrust, rudder, eta_stations, x_st, dt):
        """Vectorised over rollouts. s is (N, 10); eta_stations is (N, n_st)."""
        p = self.p
        x, y, u, z, zd, th, thd, psi, r, v = s.T
        eta = np.asarray(eta_stations, float)
        xc = x_st - x_st.mean()
        den = max((xc ** 2).sum(), 1e-12)
        if eta.ndim == 3:
            # (N, n_st, 3): port / centre / starboard
            eta_c = eta[:, :, 1]
            half_b = 0.5 * p.get("B", 2.5)
            # transverse slope at each station, from the two outer samples
            tsl = (eta[:, :, 2] - eta[:, :, 0]) / (2.0 * half_b)
            t_slope = tsl.mean(axis=1)                   # pushes the hull sideways
            # how that slope changes along the hull: bow and stern pushed
            # opposite ways is exactly a yaw moment
            t_twist = (tsl * xc).sum(axis=1) / den
        else:
            eta_c = eta
            t_slope = np.zeros(eta.shape[0])
            t_twist = np.zeros(eta.shape[0])
        eta_bar = eta_c.mean(axis=1)
        slope = (eta_c * xc).sum(axis=1) / den

        # Linearising m_eff*udot = thrust - k u^2 about u_ss gives a first-order
        # time constant tau = m_eff / (2 k u_ss), so m_eff = 2 k u_ss tau.
        # Using u_max here instead of u_ss made the modelled surge response
        # about 30% too fast.
        drag = p["k_drag"] * u * np.abs(u)
        ud = (thrust - drag) / p["m_surge"]

        wz, zz = p["wn_heave"], p["z_heave"]
        wp, zp = p["wn_pitch"], p["z_pitch"]
        alpha = np.arctan(slope) * p["sign_pitch"]

        # Exact transition, measured from the forcing. Holding eta_bar and the
        # wave slope constant across one control step is a zero-order hold --
        # the standard and consistent choice, and the equilibrium of each
        # subsystem under that hold is exactly z = eta_bar and th = alpha.
        Pz = self._phi(wz, zz, dt)
        Pp = self._phi(wp, zp, dt)
        ez, ep_ = z - eta_bar, th - alpha
        nz = Pz[0, 0] * ez + Pz[0, 1] * zd
        nzd = Pz[1, 0] * ez + Pz[1, 1] * zd
        nth = Pp[0, 0] * ep_ + Pp[0, 1] * thd
        nthd = Pp[1, 0] * ep_ + Pp[1, 1] * thd

        # Sway. Not fitted -- every coefficient is read straight off the plant,
        # because the plant's own sway equation is already simple:
        #
        #     (M + A_inf)_22 vdot = rudder lift  -  k_v v|v|  -  m u r
        #
        # The last term is the Coriolis force that makes a turning vessel slide
        # outward, and it is the one that matters: a rudder deflection does not
        # just rotate the boat, it pushes it sideways and then keeps pushing as
        # the turn develops. With no sway state the controller saw none of this
        # and produced 51 m of cross-track in seas with no lateral excitation
        # at all (DEFECTS.md C1).
        # NOT `alpha` -- that name already holds the pitch target derived from
        # the wave slope a few lines above. Reusing it here silently added the
        # rudder deflection to the pitch angle, and with the rudder centred it
        # erased the wave-slope equilibrium altogether. Caught by transcribing
        # this function into JavaScript for the viewer's drive mode and
        # comparing the two step by step.
        rud_eff = np.clip(rudder, -p["rud_stall"], p["rud_stall"])
        lift = p["k_lift"] * u * np.abs(u) * rud_eff
        vd = (lift - p["k_lin_sway"] * v - p["k_sway"] * v * np.abs(v)
              - p["m_coriolis"] * u * r) / p["m_sway"]
        # REVERTED, and the reason is worth keeping. The obvious way to give
        # this model wave-induced sway and yaw is a gravity component: a hull on
        # a laterally tilted surface slides down it, and if that tilt differs
        # bow to stern the two ends are pushed opposite ways and it yaws. Both
        # terms come out dimensionally clean, and the coefficients were fitted
        # against the plant rather than guessed.
        #
        # It explains nothing. Regressed on 3 x 180 s of short-crested seas:
        #
        #     sway   correlation +0.066   R^2 0.004
        #     yaw    correlation +0.003   R^2 0.000
        #
        # and the feature is the same SIZE as the response (std 0.248 against
        # 0.215), so this is not a scaling failure -- the two signals simply do
        # not line up in time. Wave excitation is not instantaneous local
        # geometry; it is the BEM excitation transfer function, which carries a
        # frequency-dependent PHASE LAG. No real coefficient on an instantaneous
        # geometric feature can represent a phase shift.
        #
        # The honest fix is to carry the wave components and apply the complex
        # excitation per component, as the plant does -- see the note in
        # `studies/model_horizon.py`. Left at zero rather than deleted so the
        # measurement above is not repeated by the next person.
        vd = vd + p["c_wave_sway"] * 9.81 * t_slope

        # First-order channels: surge is slow (tau 6.3 s vs dt 0.5) so explicit
        # stepping is fine; yaw is marginal at tau_r = 0.5 s, so it gets the
        # exact first-order update too.
        ay = float(np.exp(-dt / p["tau_r"]))
        r_rud = p["k_nomoto"] * rudder + (r - p["k_nomoto"] * rudder) * ay
        r_new = r_rud + p["c_wave_yaw"] * 9.81 * t_twist * dt

        ns = np.empty_like(s)
        ns[:, 2] = u + ud * dt
        ns[:, 3] = nz + eta_bar
        ns[:, 4] = nzd
        ns[:, 5] = nth + alpha
        ns[:, 6] = nthd
        ns[:, 8] = r_new
        ns[:, 9] = v + vd * dt
        ns[:, 7] = psi + 0.5 * (r + r_new) * dt
        # Full kinematics, matching the plant: a vessel with sway velocity does
        # not travel along its own centreline. Dropping the sway term here was
        # the other half of the same defect.
        c_, s_ = np.cos(ns[:, 7]), np.sin(ns[:, 7])
        ns[:, 0] = x + (ns[:, 2] * c_ - ns[:, 9] * s_) * dt
        ns[:, 1] = y + (ns[:, 2] * s_ + ns[:, 9] * c_) * dt
        # Bow acceleration as the AVERAGE over the step, not the instantaneous
        # value at its leading edge. The wave forcing is held constant across a
        # control step (zero-order hold), so the instantaneous formula sees the
        # staircase edge and over-predicted the amplitude by 2.1x -- measured
        # against the plant in `studies.model_horizon`. Averaging is also what
        # the cost actually wants, since it integrates over the step.
        x_bow = p.get("x_bow", p["L"] / 2)      # the bow, not +L/2 from the CG
        a_bow = ((nzd - zd) + p["sign_pitch"] * x_bow * (nthd - thd)) / dt
        z_bow = z + p["sign_pitch"] * x_bow * th
        # the bow station on the CENTRELINE, not the whole port/centre/stbd
        # triple -- eta_c already selected it
        rel = z_bow - eta_c[:, -1]
        return ns, a_bow, rel

    def from_plant_state(self, s_plant):
        """Project the full plant state onto the reduced state."""
        eta, nu = s_plant[:6], s_plant[6:12]
        # nu[1] (sway velocity) is appended LAST so every existing index stays
        # valid. It used to be dropped entirely, which is why the controller
        # could not see the drift its own rudder was causing.
        return np.array([eta[0], eta[1], nu[0], eta[2], nu[2],
                         eta[4], nu[4], eta[5], nu[5], nu[1]])
