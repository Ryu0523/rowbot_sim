#!/usr/bin/env python3
"""
M4 -- nonlinear 6-DOF vessel in irregular seas.

Built on the linear Cummins core verified in M3, with the terms that matter for
a 10 m USV at SS4-5 added on top:

  nonlinear Froude-Krylov   buoyancy and incident-wave pressure integrated over
                            the INSTANTANEOUS wetted sections. Hs/draught ~ 4
                            here, so emergence and re-entry are routine, not
                            extreme. Replaces the linear C and linear FK in
                            heave and pitch -- applying both would double count.
  diffraction               kept linear, from the BEM, applied component by
                            component so a broad spectrum is not collapsed onto
                            one frequency.
  added resistance          Gerritsma-Beukelman radiated-energy form. Without it
                            the vessel never loses speed to the sea, which
                            flatters any controller that slows down on purpose.
  viscous damping           quadratic, per dof. Potential flow gives roll almost
                            no damping at all, so roll would otherwise resonate
                            without limit.
  actuators                 lag, delay, rate limits, dead band, ventilation.
  forward speed             enters through the vessel querying the wave field at
                            its own moving position; encounter frequency is a
                            consequence, never a parameter.

KNOWN LIMITATIONS, carried deliberately and listed in `LIMITATIONS`:
  * hydrodynamic coefficients are zero-speed. Real forward-speed effects on
    A(w) and B(w) need strip theory or a forward-speed Green function.
  * roll restoring stays linear; the strip integral is fore-aft only.
  * the wave field is linear, so the surface is Gaussian and extreme crests
    are under-represented.
"""
import numpy as np

from hydro import bem
from .cummins import radiation_memory
from .sections import WigleySections, SlamLoad
from .actuators import Propulsion, Rudder
from .forces import Wind, coriolis_force, point_load
from hydro.symmetry import enforce as enforce_symmetry
from .wavefield import SeaState, interp_transfer_cached, HeadingTransfer

G = 9.81
RHO = 1025.0
SIGN_PITCH = -1.0          # dz = -x*theta, from the Capytaine dof definition

LIMITATIONS = [
    "zero-speed hydrodynamic coefficients",
    "linear roll restoring (strip integral is fore-aft only)",
    "Gaussian (linear) wave field: extreme crests under-represented",
    "diffraction linear and at zero speed",
]

IDX = dict(surge=0, sway=1, heave=2, roll=3, pitch=4, yaw=5)


_REPORTED = set()          # hulls whose placeholder notice has been printed


class _ConstRoll:
    """Roll damping from a decay test: fixed b1, b2."""

    def __init__(self, b1, b2):
        self.b = (float(b1), float(b2))

    def coeffs(self, U=0.0):
        return self.b

USV_L = 10.0      # the hull length every placeholder below was chosen for


def froude_placeholders(L):
    """The 10 m USV's placeholder values, Froude-scaled to a hull of length L.

    Lengths ~ lam, times and speeds ~ sqrt(lam), forces ~ lam^3, quadratic
    damping ~ lam^2 (translations) and lam^5 (rotations), lam = L / 10. At
    L = 10 these are exactly the old constants. For any other L they are NOT
    data about that vessel -- they describe a geometrically similar USV of
    that length -- but they have the right dimensions. They used to stay
    absolute whatever the hull: a 12 kN thruster and 2500 N m s^2 of roll
    damping on a 320 m tanker, and on a 2 m boat 50 m/s^2 of acceleration.
    """
    lam = L / USV_L
    r = np.sqrt(lam)
    return dict(
        lam=lam,
        visc=(np.array([280.0, 4000.0, 3000.0, 2500.0, 9000.0, 6000.0])
              * np.array([lam ** 2] * 3 + [lam ** 5] * 3)),
        prop=dict(t_max=12000.0 * lam ** 3, t_min=-4000.0 * lam ** 3,
                  tau=1.5 * r, rate_max=8000.0 * lam ** 2.5,
                  dead_band=50.0 * lam ** 3, delay=0.15 * r, u_ref=4.5 * r),
        rudder=dict(rate_deg=25.0 / r, tau=0.25 * r, delay=0.10 * r,
                    u_min=0.5 * r),
        x_wind=0.5 * lam, ar_tau=4.0 * r)


class NonlinearVessel:
    """6-DOF vessel; state = [eta(6), nu(6), radiation, thrust, rudder]."""

    def __init__(self, db, sea, L=10.0, B=2.5, T=0.8, n_stations=41,
                 dt=0.05, visc=None, verbose=False, use_slam_load=True,
                 wind=None, symmetric=True, hull=None, prop=None, rudder=None,
                 captive=None):
        # everything but the database and the sea, so with_sea() can build
        # the same vessel in other water (the reduced model's identification
        # runs used to build a DIFFERENT vessel -- Wigley sections, B 2.5)
        self._init_kw = dict(L=L, B=B, T=T, n_stations=n_stations, dt=dt,
                             visc=visc, verbose=verbose,
                             use_slam_load=use_slam_load, wind=wind,
                             symmetric=symmetric, hull=hull, prop=prop,
                             rudder=rudder, captive=captive)
        # `hull` is a hydro.hull.Hull and supersedes the loose dimensions. It
        # exists so this class is not about one shape: it supplies the station
        # table (from a mesh, if the hull came from a file), the appendage
        # geometry, and the symmetry declaration. The keyword arguments stay
        # for the Wigley test hull every existing study constructs directly.
        if hull is not None:
            L, B, T = hull.L, hull.B, hull.T
            n_stations, symmetric = hull.n_stations, hull.symmetric
        self.db, self.sea, self.hull = db, sea, hull
        name = getattr(db, "attrs", {}).get("name")
        if hull is None and name is not None and \
                not str(name).startswith("wigley"):
            raise ValueError(
                f"NonlinearVessel: the database is for {name!r}; without "
                f"hull= this would be a Wigley of its main dimensions, with "
                f"the USV's appendages. Pass hull=hydro.hulls.get({name!r}), "
                f"or build it with sim.config.plant_for.")
        self.L, self.B, self.T = L, B, T
        # the hull's water: a fresh-water tank model is 2.5% lighter per
        # cubic metre, and the weight, buoyancy and every appendage force
        # used to assume sea water regardless
        self.rho = hull.rho if hull is not None else RHO
        self.sec = (WigleySections(L, B, T, n_stations) if hull is None
                    else hull.sections())
        self.rad = radiation_memory(db, verbose=verbose, symmetric=symmetric)
        # Port/starboard symmetry is a geometric fact, so it is imposed rather
        # than left to the panel mesh to approximate. The residuals are tiny
        # against the diagonals (4e-5) but they are multiplied by u^2 in the
        # Coriolis term, and carrying them produced 2.9 m of lateral drift in
        # head seas where the answer is exactly zero. See `hydro/symmetry.py`.
        # `symmetric=False` for a hull that genuinely is not, where those
        # entries are physics instead of mesh noise.
        self.M = enforce_symmetry(db.M) if symmetric else db.M
        # restoring is applied once, linearly, for every dof; the strip model
        # contributes only the nonlinear departure on top of it
        self.C_full = np.array(enforce_symmetry(db.C) if symmetric else db.C,
                               float)
        # Restoring acts on heave, roll and pitch: displacements from the
        # equilibrium waterline. Surge, sway and yaw are POSITIONS in the earth
        # frame and nothing restores them. The BEM's matrix carries a roll-yaw
        # entry, -rho g V (x_B - x_G), whenever the centre of gravity is not
        # over the centre of buoyancy; multiplied by the absolute heading it
        # rolled KCS 2.11 (GM 7 mm) 17 deg for having been turned 60 deg, and
        # would capsize it in a turning circle (DEFECTS G5). Exactly zero on
        # the Wigley, whose LCG is its LCB, so nothing there changes.
        self.C_full[:, [0, 1, 5]] = 0.0
        self.Mtot = self.M + self.rad.A_inf
        self.Minv = np.linalg.inv(self.Mtot)

        # Placeholders the vessel's own data should replace, scaled to its
        # length (exactly the old values for the 10 m USV). The hull's own
        # values win where it has them; the `prop`, `rudder`, `visc` and
        # `wind` arguments win over both.
        ph = froude_placeholders(L)
        self.placeholders = ph
        pp, rp = dict(ph["prop"]), dict(ph["rudder"])
        if hull is not None:
            u_d = hull.scales()["u_design"]
            if hull.t_max is not None:
                t_max = float(hull.t_max)
            else:
                # The USV's thrust margin over its resistance (12 kN against
                # 5.7 kN at 4.5 m/s), at THIS vessel's design speed and on its
                # resistance placeholder. Froude-scaling the 12 kN instead gave
                # a vessel designed for another Froude number an operating
                # range around the wrong speed: the KVLCC2 model, designed for
                # 0.97 m/s, made 4.1 m/s at full thrust and its reduced model
                # was identified at 3.4 m/s (DEFECTS G5). Equal to the Froude
                # value when u_design is the USV's, Froude-scaled.
                r0 = hull.visc[0] if hull.visc is not None else ph["visc"][0]
                t_max = 12000.0 * (r0 * u_d ** 2) / (280.0 * 4.5 ** 2)
            # the rest of the thruster in proportion to its limit, as on the USV
            pp.update(t_max=t_max, t_min=-t_max / 3.0,
                      rate_max=t_max * 8000.0 / 12000.0 / np.sqrt(ph["lam"]),
                      dead_band=t_max * 50.0 / 12000.0)
            pp["u_ref"] = u_d
            if hull.rudder_max_deg is not None:
                rp["max_deg"] = hull.rudder_max_deg
            if hull.rudder_rate_deg is not None:
                rp["rate_deg"] = hull.rudder_rate_deg
            hull._fill_x()
            # one or more units per actuator, sharing one command each
            self.thrusters = [tuple(map(float, p)) for p in
                              (hull.thrusters or
                               ((hull.x_prop, 0.0, hull.z_prop),))]
            self.rudder_points = [tuple(map(float, p)) for p in
                                  (hull.rudders or
                                   ((hull.x_rud, 0.0, hull.z_rud),))]
            self.prop = prop or Propulsion(
                dt=dt, x_prop=self.thrusters[0][0],
                z_prop=self.thrusters[0][2], r_prop=0.5 * hull.d_prop,
                series=hull.prop_series, n_units=len(self.thrusters),
                rho=self.rho, **pp)
            self.rudder = rudder or Rudder(
                dt=dt, area=hull.rudder_area, x_rud=self.rudder_points[0][0],
                z_rud=self.rudder_points[0][2], span=hull.rudder_span,
                rho=self.rho, **rp)
        else:
            self.prop = prop or Propulsion(dt=dt, **pp)
            self.rudder = rudder or Rudder(dt=dt, **rp)
            self.thrusters = [(self.prop.x_prop, 0.0, self.prop.z_prop)]
            self.rudder_points = [(self.rudder.x_rud, 0.0, self.rudder.z_rud)]
        self._ar_tau = ph["ar_tau"]
        # Wind. Defaults to the wind that GENERATES this sea state, because a
        # sea state is a wind: taking the waves and dropping the air was a
        # choice, not a neutral omission. Pass wind=Wind() for a flat calm, or
        # any Wind instance to override speed and direction independently.
        # With the hull's projected areas, Blendermann's measured
        # coefficients (sim/forces.py) replace the box-shaped topside.
        if wind is None and hull is not None and hull.wind_areas:
            from .forces import BlendermannWind
            wa = dict(hull.wind_areas)
            self.wind = BlendermannWind.from_sea_state(
                getattr(sea, "hs", 0.0),
                direction=getattr(sea, "theta0", np.pi),
                L_oa=wa.pop("L_oa", L), **wa)
        else:
            self.wind = (Wind.from_sea_state(getattr(sea, "hs", 0.0),
                                             direction=getattr(sea, "theta0",
                                                               np.pi),
                                             L=L, beam=B,
                                             freeboard=(hull.freeboard if hull
                                                        is not None else 0.55),
                                             x_wind=ph["x_wind"])
                         if wind is None else wind)
        # Moments are taken about the centre of gravity, which is where the
        # BEM's rotation centre is: the hull's own z_cog, or -T/3 for the
        # Wigley test hull (hydro/run_wigley.py builds it that way).
        self.z_cog = hull.z_cog if hull is not None else -T / 3.0
        self.dt = dt
        # The bow's own thresholds for slamming and deck wetness: the keel at
        # the bow station, the freeboard there. The closed loop used the USV's
        # 0.8 m draught and 0.55 m freeboard for every vessel.
        self.draft_bow = float(-self.sec.keel[-1])
        self.freeboard = hull.freeboard if hull is not None else 0.55

        # Quadratic viscous damping. The comment here used to say that potential
        # flow leaves roll essentially undamped, so this term was carrying the
        # whole of roll. That is a rule of thumb about SHIPS and it does not
        # apply to this vessel: roll radiation damping depends strongly on roll
        # PERIOD, and a big ship rolling at 12 s sits where B44 is ~0 while this
        # hull rolls near 2 s, close to the peak of B44. Measured in
        # `studies/damping_check.py`, the placeholder supplies about a fifth of
        # this vessel's roll damping, not all of it -- total zeta 10-13% over
        # 5-20 degrees, of which the BEM gives ~9%.
        # Still placeholders pending decay tests -- see CALIBRATION_NEEDED.
        # Surge is the calm-water resistance and sets the speed envelope:
        # 280 N/(m/s)^2 gives ~10 kN at 6 m/s, the right order for a 9 t
        # semi-displacement 10 m hull. If this is wrong the speed/motion
        # trade-off the whole study rests on either saturates or disappears.
        if visc is None and hull is not None and hull.visc is not None:
            visc = hull.visc
        self.visc = np.array(visc if visc is not None else ph["visc"],
                             float)

        # Fd, Fl: snapshots at heading 0, kept for callers that read them.
        # The forces use F_heading, which follows the vessel's heading: the
        # snapshot alone kept a vessel turned beam-on under head-sea forcing
        # (DEFECTS F4).
        self.Fd = interp_transfer_cached(db, sea, "F_diff")      # (n_comp, 6)
        self.Fl = interp_transfer_cached(db, sea, "F_exc")       # linear, for dofs
        self.F_heading = HeadingTransfer(db, sea, "F_exc")
        self.n_rad = self.rad.n
        self.n_state = 12 + self.n_rad + 2
        self._ar_ema = 0.0
        # the mean forward speed, from which the radiation memory's surge
        # input is measured (see deriv); set at the first step of a run
        self._u_mean = None
        self._tau_umean = 20.0 * np.sqrt(L / USV_L)
        self._rad_cache = {}                # exact memory update, per dt
        self.last_bow_acc = 0.0
        self.slam_count = 0
        self.slam = SlamLoad(self.sec, rho=self.rho)
        self.use_slam_load = use_slam_load
        self.peak_slam_force = 0.0
        self.last_slam_force = 0.0
        self.last_rel_bow = 0.0
        self._w_wave_bow = 0.0
        # Hull lateral force derivative and the yaw damping derivative.
        #
        # These used to be tuned until the turning circle looked plausible, and
        # the turning circle was then cited as evidence the model manoeuvres
        # plausibly. `studies/manoeuvring_identify.py` inverts that: the values
        # are taken from the middle of the PUBLISHED non-dimensional ranges
        # (|Y_v'| 0.010-0.040, |N_r'| 0.002-0.008) intersected with the checks
        # that can be run without a trial, and the turning circle is now an
        # OUTPUT -- a prediction a trial can falsify.
        #
        # Two things that sweep showed and are worth carrying:
        #   * the 35 deg radius spans 1.09-2.19 L across the feasible region,
        #     a factor of 2.0, and that is the honest uncertainty on heading
        #     authority -- heading being this project's only effective channel.
        #     Re-identified twice: once after the rudder lift slope became
        #     computed, and again after the Coriolis frame was corrected
        #     (DEFECTS E11), which moved the centre from Nr' 0.0040 to 0.0060.
        #     The 0.0040 had been identified against a vessel that, with the
        #     wrong frame, rolled at 17 deg/s rms in a calm-water turn.
        #   * Y_v' barely moves the turning circle at all -- the rows of the
        #     sweep are flat. A turning trial cannot identify it. That needs
        #     PMM measurements or a zigzag.
        #
        # The Munk moment is no longer absent: it arrives correctly as part of
        # the added-mass Coriolis matrix (`sim/forces.py`), which is why these
        # had to be redone. Adding it alone once made the vessel directionally
        # unstable; it is stable now because the rudder inflow angle supplies
        # the weathervane damping that was missing at the same time.
        #
        # For any other hull, and unless it states its own: Clarke et al.
        # (1983), the standard regression from main dimensions (hydro/
        # manoeuvring.py). The Wigley USV's hull states the identified pair.
        self.Yv_prime, self.Nr_prime = 0.025, 0.0060
        self.manoeuvring_source = "identified for the Wigley USV"
        if hull is not None:
            yv, nr = hull.Yv_prime, hull.Nr_prime
            if yv is None or nr is None:
                from hydro.manoeuvring import clarke_1983
                cl = clarke_1983(hull.L, hull.B, hull.T, hull.form()["C_B"])
                yv = abs(cl["Yv"]) if yv is None else yv
                nr = abs(cl["Nr"]) if nr is None else nr
                self.manoeuvring_source = "Clarke et al. (1983)"
            else:
                self.manoeuvring_source = "given"
            self.Yv_prime, self.Nr_prime = float(yv), float(nr)
        self._emerged = False
        self.v_slam = 0.093 * np.sqrt(G * L)

        # Degrees of freedom held at their initial velocity, as a towing
        # carriage holds a model: the tank tests this plant is checked
        # against were towed (surge) and often restrained in sway and yaw.
        # Set before the roll model, which depends on what is held.
        self.captive = sorted({IDX[k] if isinstance(k, str) else int(k)
                               for k in (captive or ())})
        # The free degrees of freedom obey their OWN block of the mass
        # matrix: Mtot_ff nu_dot_f = tau_f, the held ones taking whatever
        # force holds them. Solving the full system and then zeroing the held
        # accelerations -- as this did -- let a surge force the carriage
        # absorbs drive heave and pitch through the surge-pitch added mass:
        # an energy source, and a towed box barge diverged in small waves
        # (heave 1e39 m after 180 s, DEFECTS G5).
        self._free = [i for i in range(6) if i not in self.captive]
        self._Minv_free = (np.linalg.inv(self.Mtot[np.ix_(self._free,
                                                          self._free)])
                           if self.captive else None)

        # Roll damping beyond the BEM's wave part. The quadratic visc[3] is
        # the USV's placeholder; a hull can instead take the simplified Ikeda
        # method (checked on KVLCC2's decays, DEFECTS F8, G4) or its own decay
        # test. Either replaces visc[3] -- never adds to it.
        self.roll, self.roll_source = None, "placeholder visc[3]"
        mode = hull.roll_damping if hull is not None else "placeholder"
        if mode != "placeholder":
            self.roll, self.roll_source = self._roll_model(mode)

        if hull is not None and hull.name not in _REPORTED:
            # once per hull: the RL environment builds a plant every episode
            _REPORTED.add(hull.name)
            ph_list = hull.placeholders()
            if ph_list:
                print(f"  NonlinearVessel({hull.name}): {len(ph_list)} inputs "
                      f"are placeholders, not data about this vessel "
                      f"(Hull.check() lists them). Roll damping: "
                      f"{self.roll_source}; manoeuvring: "
                      f"{self.manoeuvring_source}.")

    def _roll_wave_damping(self, w):
        """Radiation damping of the roll MODE at w, as this plant sees it.

        A free-roll decay measures the whole damping of the roll mode. With
        sway and yaw free -- a model floating free in the tank -- the roll
        mode also radiates through them, and its wave damping is the roll
        entry of the impedance with sway and yaw condensed out: on KVLCC2
        about half of B44 alone (DEFECTS F8). The decay-test input used to
        subtract B44 alone, which took out wave damping the plant then did
        not have, and it reproduced the measured decay at 0.85x (DEFECTS G4).
        With sway and yaw held (a towed test), the roll entry alone is right.
        """
        db = self.db
        keep = [j for j in (1, 3, 5) if j == 3 or j not in self.captive]
        A = np.array([[np.interp(w, db.omega, db.A[:, i, j]) for j in keep]
                      for i in keep])
        B = np.array([[np.interp(w, db.omega, db.B[:, i, j]) for j in keep]
                      for i in keep])
        ix = np.ix_(keep, keep)
        Z = -w ** 2 * (self.M[ix] + A) - 1j * w * B + self.C_full[ix]
        r = keep.index(3)
        o = [k for k in range(len(keep)) if k != r]
        z = Z[r, r]
        if o:
            z = z - Z[r, o] @ np.linalg.solve(Z[np.ix_(o, o)], Z[o, r])
        return float(-z.imag / w)          # Capytaine's -i w B convention

    def _roll_model(self, mode):
        """(model with .coeffs(U) -> (b1, b2), description).

        "auto" is the simplified Ikeda method, flagged when the hull is
        outside its range. It used to fall back to the USV's quadratic,
        Froude-scaled, outside the range; on KVLCC2 -- outside only through
        C_M -- that placeholder gave 3.1-3.3x the measured damping at 0 kn
        against Ikeda's 1.6-1.7x, and at 15.5 kn the wrong amplitude
        dependence (DEFECTS G4). A multihull keeps the placeholder: the
        method is for monohull sections."""
        from hydro.ikeda import (RollDamping, natural_roll_frequency,
                                 NU_SEA, NU_FRESH)
        h, db = self.hull, self.db
        I44, C44 = float(self.M[3, 3]), float(self.C_full[3, 3])
        if C44 <= 0.0:
            return None, "placeholder visc[3] (no positive roll restoring)"
        wn = natural_roll_frequency(db.omega, db.A[:, 3, 3], I44, C44)
        if isinstance(mode, dict):
            # A decay test measures the TOTAL damping; the plant already has
            # the wave part from the BEM, so only the rest is added.
            w0 = 2.0 * np.pi / float(mode["T"]) if "T" in mode else wn
            I_eff = C44 / w0 ** 2
            b_wave = self._roll_wave_damping(w0)
            b1 = 2.0 * float(mode["zeta1"]) * w0 * I_eff - b_wave
            b2 = float(mode["k2"]) * I_eff
            held = all(j in self.captive for j in (1, 5))
            return (_ConstRoll(max(b1, 0.0), max(b2, 0.0)),
                    f"decay test (zeta1 {mode['zeta1']:.4g}, k2 "
                    f"{mode['k2']:.4g}, minus the BEM wave part of the roll "
                    f"mode, sway and yaw {'held' if held else 'free'})")
        if mode == "auto" and getattr(h, "multihull", False):
            return None, ("placeholder visc[3] -- the simplified Ikeda method "
                          "is for monohulls")
        f = h.form()
        V = h.displacement / h.rho if h.displacement is not None else f["V"]
        bk = h.bilge_keel or (0.0, 0.0)
        rd = RollDamping(L=h.L, B=h.B, d=h.T, C_B=f["C_B"], C_M=f["C_M"],
                         KG=h.T + h.z_cog, w_roll=wn, V=V, rho=h.rho,
                         nu=NU_FRESH if h.rho < 1010.0 else NU_SEA,
                         b_BK=bk[0], l_BK=bk[1])
        return rd, ("simplified Ikeda" + (
            f", OUTSIDE its range: {'; '.join(rd.out_of_range)}"
            if rd.out_of_range else ""))

    def with_sea(self, sea, **overrides):
        """The same vessel in other water (or with some settings changed)."""
        import copy
        kw = dict(self._init_kw)
        for k in ("prop", "rudder"):            # they carry delay buffers
            if kw.get(k) is not None:
                kw[k] = copy.deepcopy(kw[k])
        kw.update(overrides)
        return NonlinearVessel(self.db, sea, **kw)

    # ------------------------------------------------------------ helpers
    def unpack(self, s):
        return (s[:6], s[6:12], s[12:12 + self.n_rad],
                s[12 + self.n_rad], s[13 + self.n_rad])

    def initial_state(self, u0=0.0):
        s = np.zeros(self.n_state)
        s[6] = u0
        self._u_mean = None      # a new run: the mean speed restarts from u0
        return s

    def _stations(self, eta):
        """Earth-frame positions of the strip stations. The hull's axis
        points along the heading; the stations used to be laid along the
        earth x axis whatever the heading was (DEFECTS F4)."""
        c, s_ = np.cos(eta[5]), np.sin(eta[5])
        return eta[0] + self.sec.x * c, eta[1] + self.sec.x * s_

    def _point(self, eta, x_body, y_body=0.0):
        """Earth-frame position of a body point (x_body, y_body)."""
        c, s_ = np.cos(eta[5]), np.sin(eta[5])
        return (np.atleast_1d(eta[0] + x_body * c - y_body * s_),
                np.atleast_1d(eta[1] + x_body * s_ + y_body * c))

    def _submergences(self, eta, t, points, dz=0.0):
        """Water depth over (z + dz) at each body point (x, y, z): negative
        once that point is out of the water. Heave, pitch AND roll move the
        point -- an off-centre unit rises and falls with roll, which the
        centreline-only version never had to consider."""
        pts = np.asarray(points, float)
        xs, ys = self._point(eta, pts[:, 0], pts[:, 1])
        surface = self.sea.eta(xs, ys, t)
        hull_z = (eta[2] + SIGN_PITCH * pts[:, 0] * eta[4]
                  + pts[:, 1] * eta[3])
        return surface - hull_z - (pts[:, 2] + dz)

    # ------------------------------------------------------------- forces
    def wave_forces(self, eta, t):
        """Linear BEM excitation and restoring, plus the nonlinear strip DELTA.

            tau = tau_BEM_linear  -  C eta  +  (strip_nonlinear - strip_linear)

        Written this way the model is exact in the linear limit -- the bracket
        vanishes -- so it inherits M3's verified accuracy, and strip theory,
        whose error grows as the wavelength approaches the hull length, only
        ever acts on the nonlinear correction rather than on the whole force.
        """
        x, y, z = eta[0], eta[1], eta[2]
        pitch = eta[4]
        xs, ys = self._stations(eta)

        # linear excitation from the BEM, component by component, each at its
        # direction relative to the hull
        phase = self.sea.phases(x, y, t)
        drive = (self.sea.a * np.exp(1j * phase))[:, None]
        tau = (np.real(np.sum(self.F_heading.at(eta[5]) * drive, axis=0))
               - self.C_full @ eta)

        # nonlinear departure, heave and pitch only
        eta0 = self.sea.eta(xs, ys, t)
        d0 = self.sec.immersion(eta0, z, pitch, SIGN_PITCH)
        zc = self.sec.centroid_depth(d0)
        eta_eff = self.sea.eta_at_depth(xs, ys, t, np.minimum(zc, 0.0))
        Fz, Mp, d, _ = self.sec.fk_restoring(eta_eff, z, pitch, SIGN_PITCH,
                                             rho=self.rho)
        Fz -= self.rho * G * self.sec.volume()            # subtract weight
        Fz_l, Mp_l = self.sec.fk_restoring_linear(eta_eff, z, pitch,
                                                  SIGN_PITCH, rho=self.rho)
        tau[2] += Fz - Fz_l
        tau[4] += Mp - Mp_l
        # relative water elevation at the bow: the parent signal for slamming,
        # deck wetness and added resistance alike (see sim/seakeeping.py)
        self.last_rel_bow = float(d[-1])
        return tau, d, None

    def added_resistance(self, nu, eta, t):
        """Gerritsma-Beukelman radiated-energy added resistance.

        R_aw = (k / w_e) * integral b'_33(x) <Vz*(x)^2> dx

        Sectional damping is distributed along the hull in proportion to
        beam^2, the usual approximation when only the integrated B(w) is
        available; the mean square of relative vertical velocity is tracked
        with an exponential average because the formula is a mean, not an
        instantaneous force.
        """
        xs, ys = self._stations(eta)
        w_vessel = nu[2] + SIGN_PITCH * self.sec.x * nu[4]
        w_wave = self.sea.eta_dot(xs, ys, t)
        vz = w_vessel - w_wave
        # the slam detector needs the water's own vertical velocity at the bow
        # and this is the one place it is already computed every step
        self._w_wave_bow = float(w_wave[-1])
        wt = self.sec.y0 ** 2
        wt = wt / max(np.sum(wt) * self.sec.dx, 1e-12)

        wp = 2 * np.pi / max(self.sea.tp, 1e-6)
        k = wp ** 2 / G                    # the WAVE's wave number (below)
        # encounter frequency at the heading relative to the mean wave
        # direction; this was the head-seas formula whatever the heading
        cb = np.cos(getattr(self.sea, "theta0", np.pi) - eta[5])
        we = max(abs(wp - k * max(nu[0], 0.0) * cb), 0.1 * wp)
        i = int(np.argmin(np.abs(self.db.omega - we)))
        b33 = self.db.B[i, 2, 2]

        ms = float(np.sum(wt * vz ** 2) * self.sec.dx)
        self._ar_ema += (ms - self._ar_ema) * min(self.dt / self._ar_tau, 1.0)
        # Gerritsma-Beukelman: the energy the hull radiates per encounter
        # period, (pi / w_e) int b' Vza^2 dx, is the work of R_aw over the
        # distance the wave pattern moves past the hull in that period,
        # lambda / |cos b|. So R_aw = (k |cos b| / 2 w_e) int b' Vza^2 dx
        # = (k |cos b| / w_e) int b' <Vz^2> dx, with k = 2 pi / lambda of the
        # WAVE. This used k = w_e^2 / g, which overstates R_aw by (w_e / w)^2:
        # 1.69 at the USV's 4.5 m/s in Tp 9.7 s head seas (DEFECTS F5).
        return -(k * abs(cb) / max(we, 1e-6)) * b33 * self._ar_ema

    def slam_load(self, eta, nu, d, t):
        """Water-entry force from the changing added mass, per station.

            F' = PILE_UP . (dm_a/dd) . v_entry^2      [N per metre, upward]

        where v_entry = dd/dt is how fast the water surface is climbing the
        section RELATIVE to the hull. It is one-sided: exit is not an impact,
        and the flow separates rather than pulling the section back down.

        Note what the shape of the term does on its own, with no thresholds
        and no event logic. Below the keel b = 0, so the force is zero. At the
        design waterline the Wigley section is at its widest, db/dd = 0, so the
        force is zero again. It is non-zero only in between -- while a section
        is actually entering -- which is the definition of slamming. There is
        no slam "trigger" here; there is a force that happens to be large only
        during entry.
        """
        xs, ys = self._stations(eta)
        w_vessel = nu[2] + SIGN_PITCH * self.sec.x * nu[4]
        w_wave = self.sea.eta_dot(xs, ys, t)
        v_entry = np.maximum(w_wave - w_vessel, 0.0)        # = dd/dt, one-sided
        # generalised Wagner: pile-up and flow separation from each section's
        # own shape (SlamLoad._wagner), rather than a wedge's pi/2 everywhere
        f = (self.slam.wagner_slope(d)
             * v_entry ** 2 * self.sec.dx)                  # N, upward
        return float(np.sum(f)), float(SIGN_PITCH * np.sum(self.sec.x * f))

    def hull_side_force(self, nu):
        """Lift-like lateral force from the hull at a drift angle.

        A hull moving with drift angle beta = v/u behaves like a very
        low-aspect-ratio lifting surface, and generates a side force
        proportional to u*v -- NOT to v|v|. That term was missing entirely: the
        only lateral resistance was the quadratic viscous one, which is
        negligible at the small drift angles a controller actually works at.

        The consequence was not subtle. With 5 degrees of rudder the hull
        reached a 30 degree drift angle and a turning radius of 0.7 ship
        lengths, and the surge Coriolis coupling m*v*r then ate 5.1 kN of a
        7 kN thrust. Every preview study in this project ran with the rudder
        LOCKED, and the stated reason was that the reduced model had no sway
        state -- but the plant was misbehaving too, and locking the channel hid
        it.

        Written as  Y = -0.5 rho L^2 |u| v Yv'  with the non-dimensional
        derivative in the usual range for a slender hull. That coefficient is a
        CALIBRATION PLACEHOLDER of the same standing as the viscous terms: the
        form is right, the number needs a turning-circle trial. It is gated on
        producing a physically plausible turning circle (see
        `sim/test_manoeuvre.py`), which pins it far better than nothing.
        """
        u, v, r = nu[0], nu[1], nu[5]
        Y = -0.5 * self.rho * self.L ** 2 * abs(u) * v * self.Yv_prime
        # Yaw damping from the same mechanism: the hull resists rotation.
        N = -0.5 * self.rho * self.L ** 4 * abs(u) * r * self.Nr_prime
        return Y, N

    def coriolis(self, nu):
        """Coriolis-centripetal coupling from the rotation of the frame --
        which in THIS model is yaw only.

        `kinematics()` rotates the horizontal velocities by heading and
        integrates heave, roll and pitch rates directly (z' = w, theta' = q).
        That is the usual hybrid of a manoeuvring frame in the horizontal plane
        and a seakeeping frame for the oscillatory modes, and it decides which
        Coriolis terms exist: the ones generated by the frame's rotation,
        omega = (0, 0, r), acting on the momentum of the manoeuvring motion
        (u, v, r). For the rigid body that is exactly the two terms this
        function always had, -m v r and m u r. For the added mass it adds
        -A22 v r, A11 u r and the Munk moment (A22 - A11) u v, plus the roll
        and pitch moments the off-diagonal added mass produces from them.

        An earlier revision "completed" this into the full body-fixed matrix,
        with w, p and q in it. That matrix is right for a body-fixed frame and
        wrong for this one: it put -m u q into heave -- about 6 kN rms of force
        the kinematics already account for -- and the MPC's internal model,
        which has no such term, lost the plant's bow acceleration (correlation
        0.9 -> 0.4, studies/model_horizon.py). The skew-symmetry and Euler
        checks in sim/forces.py passed throughout, because they check the
        algebra, not which frame the algebra belongs to. The forward-speed
        coupling of the vertical modes that those terms seemed to supply is
        real, but it belongs to forward-speed hydrodynamic coefficients,
        which LIMITATIONS lists as absent (zero-speed). DEFECTS E11.
        """
        h = np.array(nu, float)
        h[2] = h[3] = h[4] = 0.0            # w, p, q live in the seakeeping frame
        return (coriolis_force(self.M, h)
                + coriolis_force(self.rad.A_inf, h))

    def rudder_submergence(self, eta, t):
        """Water depth over the TOP of the shallowest rudder blade."""
        return float(np.min(self._submergences(eta, t, self.rudder_points,
                                               0.5 * self.rudder.span)))

    # ------------------------------------------------------------ dynamics
    def deriv(self, s, t):
        eta, nu, xr, thr, rud = self.unpack(s)
        tau_w, d, _ = self.wave_forces(eta, t)

        mu = self.rad.force(xr)
        dv = -self.visc * nu * np.abs(nu)
        if self.roll is not None:
            b1, b2 = self.roll.coeffs(max(nu[0], 0.0))
            dv[3] = -(b1 * nu[3] + b2 * nu[3] * abs(nu[3]))

        # Point loads go through r x F about the centre of gravity. The rudder's
        # roll moment used to be written as lift * z_rud: the wrong sign in this
        # z-up frame -- its yaw moment, lift * x_rud, IS r x F, so the two
        # disagreed with each other -- and an arm measured from the waterline
        # instead of the CG. Thrust below the CG also trims the vessel by the
        # stern, and that moment was not there at all. Each thruster and each
        # rudder blade acts at its own position, with its own submergence.
        tau_a = np.zeros(6)
        units = self.prop.effective_units(
            thr, self._submergences(eta, t, self.thrusters), u=nu[0])
        for (x, y, z), T_i in zip(self.thrusters, units):
            tau_a += point_load([T_i, 0.0, 0.0], [x, y, z - self.z_cog])
        subs = self._submergences(eta, t, self.rudder_points,
                                  0.5 * self.rudder.span)
        for (x, y, z), sub in zip(self.rudder_points, subs):
            lift, _, drag = self.rudder.force(
                rud, nu[0], v=nu[1], r=nu[5],
                submergence_factor=self.rudder.ventilation_factor(sub),
                x_rud=x)
            tau_a += point_load([-drag, lift, 0.0], [x, y, z - self.z_cog])
        Yh, Nh = self.hull_side_force(nu)
        tau_a[1] += Yh
        tau_a[5] += Nh
        tau_a += self.wind.force6(nu, eta[5], z_cog=self.z_cog)
        tau_a[0] += self.added_resistance(nu, eta, t)

        if self.use_slam_load:
            fz, my = self.slam_load(eta, nu, d, t)
            tau_a[2] += fz
            tau_a[4] += my
            self.peak_slam_force = max(self.peak_slam_force, abs(fz))
            self.last_slam_force = fz

        rhs = tau_w + tau_a + dv - mu - self.coriolis(nu)
        if self.captive:
            nu_dot = np.zeros(6)
            nu_dot[self._free] = self._Minv_free @ rhs[self._free]
        else:
            nu_dot = self.Minv @ rhs

        ds = np.zeros_like(s)
        ds[:6] = self.kinematics(eta, nu)
        ds[6:12] = nu_dot
        # The radiation memory is a zero-speed model of the fluid's response
        # to OSCILLATORY motion, so its surge input is the surge velocity
        # measured from the slowly varying mean forward speed, not the forward
        # speed itself. The steady advance is not radiation -- its waves are
        # the resistance, carried separately -- and the fitted memory is only
        # constrained at wave frequencies: below them it is free, and fed a
        # vessel's acceleration to 8 m/s it put 6e7 N m, thirty times the
        # restoring moment, into the pitch of a box barge and capsized it
        # (DEFECTS G5). A towed model at constant speed now radiates nothing
        # in surge, which is right; wave-frequency surge passes unchanged.
        ds[12:12 + self.n_rad] = self.rad.deriv(xr, self._nu_rad(nu))
        return ds, d

    def _nu_rad(self, nu):
        """The velocity the radiation memory sees: surge from the mean speed."""
        if self._u_mean is None:
            return nu
        r = np.array(nu, float)
        r[0] -= self._u_mean
        return r

    def _rad_step(self, dt):
        """Exact one-step update of the radiation memory, x' = A x + B nu.

        The memory is linear, so over a step its update is closed-form: with
        the velocity varying linearly across the step (first-order hold),
        x1 = Phi x0 + G1 nu0 + (G2 / dt)(nu1 - nu0), from one matrix
        exponential. Heun's method, which the rest of the state uses, is not
        stable for every pole a fit produces: on the box barge 16 of 80 poles
        -- lightly damped, near 8.4 rad/s -- grew by up to 1.3% per step at
        dt = 0.071 s, and the plant exploded after 120-150 s in calm water and
        in small waves; the 2 m catamaran had 6 such poles. At half the step
        both were stable, which is how the cause was found (DEFECTS G5).
        Exact is stable for any stable fit at any step."""
        key = round(float(dt), 12)
        hit = self._rad_cache.get(key)
        if hit is None:
            from scipy.linalg import expm
            A, B = self.rad.A, self.rad.B
            n, m = A.shape[0], B.shape[1]
            Mx = np.zeros((n + 2 * m, n + 2 * m))
            Mx[:n, :n] = A
            Mx[:n, n:n + m] = B
            Mx[n:n + m, n + m:] = np.eye(m)
            E = expm(Mx * dt)
            hit = (E[:n, :n], E[:n, n:n + m], E[:n, n + m:] / dt)
            self._rad_cache[key] = hit
        return hit

    def prop_submergence(self, eta, t):
        """Water depth over the shallowest propeller centre; negative once it
        broaches."""
        return float(np.min(self._submergences(eta, t, self.thrusters)))

    @staticmethod
    def kinematics(eta, nu):
        """Small-angle kinematics: yaw rotates the horizontal velocities."""
        psi = eta[5]
        c, s_ = np.cos(psi), np.sin(psi)
        out = np.empty(6)
        out[0] = c * nu[0] - s_ * nu[1]
        out[1] = s_ * nu[0] + c * nu[1]
        out[2:6] = nu[2:6]
        return out

    def step(self, s, t, thrust_cmd, rudder_cmd, dt=None):
        """Heun step for the continuous states; actuators advance explicitly
        because they carry rate limits and delays that RK stages would smear."""
        dt = self.dt if dt is None else dt
        eta, nu, xr, thr, rud = self.unpack(s)

        if self._u_mean is None:                # a run starts in steady motion
            self._u_mean = float(s[6])
        # Heun for the rigid body; the radiation memory advanced exactly
        # (_rad_step) -- velocity held for the predictor, linear over the step
        # for the update
        n = self.n_rad
        Phi, G1, G2 = self._rad_step(dt)
        x0 = s[12:12 + n]
        nr0 = self._nu_rad(s[6:12])
        k1, d1 = self.deriv(s, t)
        s2 = s + dt * k1
        s2[12:12 + n] = Phi @ x0 + G1 @ nr0
        k2, _ = self.deriv(s2, t + dt)
        s_new = s + 0.5 * dt * (k1 + k2)
        s_new[12:12 + n] = (Phi @ x0 + G1 @ nr0
                            + G2 @ (self._nu_rad(s_new[6:12]) - nr0))
        self._u_mean += ((1.0 - np.exp(-dt / self._tau_umean))
                         * (float(s_new[6]) - self._u_mean))

        thr_new = self.prop.advance(thr, thrust_cmd, dt)
        rud_new = self.rudder.step(rud, rudder_cmd, dt)
        s_new[12 + self.n_rad] = thr_new
        s_new[13 + self.n_rad] = rud_new

        self._update_slam(d1, nu)
        # k1 already holds nu_dot at the current state, so the bow acceleration
        # is free here. Recomputing it separately doubled the plant cost.
        self.last_bow_acc = float(k1[8] + SIGN_PITCH * self.sec.x_bow * k1[10])
        return s_new

    def _update_slam(self, d, nu):
        """Ochi criterion: bow emerges, then re-enters with high RELATIVE
        velocity. An event, not a per-timestep condition.

        Relative means relative to the WATER. The bow velocity alone is not
        the entry velocity -- the surface is moving too, and in a following
        crest it can be moving the same way, which is the case Ochi's
        threshold is meant to exclude.
        """
        # out of the water below the bow station's OWN keel, which on a raked
        # stem is well above -T (it was -T for every hull: the Wigley's keel)
        bow_out = d[-1] <= self.sec.keel[-1]
        rel_v = nu[2] + SIGN_PITCH * self.sec.x[-1] * nu[4] - self._w_wave_bow
        if self._emerged and not bow_out and rel_v < -self.v_slam:
            self.slam_count += 1
        self._emerged = bow_out

    # ----------------------------------------------------------- outputs
    def bow_acceleration(self, s, t):
        """Vertical acceleration at the bow. Prefer `last_bow_acc` after a
        step -- this recomputes the whole derivative and is only for use
        outside the integration loop.

        Note the commanded thrust and rudder are NOT arguments: the force
        depends on the actuator STATES, which lag the commands. Passing the
        commands here would have implied an authority the vessel does not have.
        """
        ds, _ = self.deriv(s, t)
        return float(ds[8] + SIGN_PITCH * self.sec.x_bow * ds[10])


def load_default(hs=3.25, tp=9.7, seed=0, theta0=np.pi, dt=0.05,
                 db_path="hydro_wigley_10m.npz", n_freq=40, n_dir=8,
                 verbose=False):
    db = bem.load(db_path)
    sea = SeaState(hs, tp, theta0=theta0, n_freq=n_freq, n_dir=n_dir, seed=seed)
    return NonlinearVessel(db, sea, L=db.L, B=float(db.attrs.get("B", 2.5)),
                           T=float(db.attrs.get("T", 0.8)), dt=dt,
                           verbose=verbose), db, sea
