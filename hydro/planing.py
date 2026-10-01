#!/usr/bin/env python3
"""
Steady planing: Savitsky's (1964) method for a prismatic hull.

Not part of the time-domain plant -- the plant is a displacement-hull model
and says so -- but the first piece of one, and what it takes to ask whether a
propulsion model sized for a planing boat makes the speeds such boats make.

Savitsky's equations are regressions on towing-tank data for prismatic
planing surfaces (Savitsky 1964, "Hydrodynamic Design of Planing Hulls",
Marine Technology 1(1)), valid for trim 2-15 deg, mean wetted length/beam
up to 4, speed coefficient C_V 0.6-13 (in practice > ~1.5, i.e. planing):

    C_L0  = tau^1.1 (0.0120 lam^0.5 + 0.0055 lam^2.5 / C_V^2)   tau in deg
    C_Lb  = C_L0 - 0.0065 beta C_L0^0.6                          beta in deg
    l_p / (lam b) = 0.75 - 1 / (5.21 C_V^2 / lam^2 + 2.39)
    Delta = 1/2 rho V^2 b^2 C_Lb

with C_V = V / sqrt(g b), b the chine beam, lam the mean wetted length over
b. This is the short form in which the forces pass through the centre of
gravity, so moment balance is l_p = LCG (from the transom); lam follows from
it, then tau from the lift. Friction on the wetted bottom at the reduced
mean velocity V1 = V (1 - p_d/q)^0.5, ITTC-1957 line plus 0.0004 roughness:

    D_f = C_f 1/2 rho V1^2 lam b^2 / cos(beta)
    R   = Delta tan(tau) + D_f / cos(tau)
"""
import numpy as np
from scipy.optimize import brentq

G = 9.81
RHO = 1025.0
NU = 1.19e-6          # sea water, 15 C


def c_l0(tau_deg, lam, cv):
    return tau_deg ** 1.1 * (0.0120 * lam ** 0.5 + 0.0055 * lam ** 2.5
                             / cv ** 2)


def c_lbeta(cl0, beta_deg):
    return cl0 - 0.0065 * beta_deg * cl0 ** 0.6


def savitsky(V, mass, lcg, b, beta_deg, rho=RHO, nu=NU, dcf=0.0004):
    """Running trim, wetted length and resistance of a prismatic planing
    hull at speed V (m/s). mass in kg, lcg from the transom and chine beam b
    in m. Returns a dict; `valid` says whether Savitsky's ranges hold."""
    delta = mass * G
    cv = V / np.sqrt(G * b)
    # moment balance: centre of pressure over the CG
    def cp_err(lam):
        return lam * (0.75 - 1.0 / (5.21 * cv ** 2 / lam ** 2 + 2.39)) \
            - lcg / b
    lam = brentq(cp_err, 0.05, 20.0)
    clb_need = delta / (0.5 * rho * V ** 2 * b ** 2)
    # C_L0 from C_Lb (monotone in C_L0 over the useful range)
    cl0 = brentq(lambda c: c_lbeta(c, beta_deg) - clb_need, 1e-6, 5.0)
    tau = brentq(lambda t: c_l0(t, lam, cv) - cl0, 0.01, 40.0)
    tr = np.radians(tau)
    # mean bottom velocity: dynamic pressure takes part of the free stream
    pdq = (0.0120 * tau ** 1.1 * lam ** 0.5
           - 0.0065 * beta_deg * (0.0120 * tau ** 1.1 * lam ** 0.5) ** 0.6) \
        / (lam * np.cos(tr))
    v1 = V * np.sqrt(max(1.0 - pdq, 0.05))
    re = v1 * lam * b / nu
    cf = 0.075 / (np.log10(re) - 2.0) ** 2 + dcf
    d_f = cf * 0.5 * rho * v1 ** 2 * lam * b ** 2 / np.cos(np.radians(beta_deg))
    r = delta * np.tan(tr) + d_f / np.cos(tr)
    valid = (2.0 <= tau <= 15.0) and (lam <= 4.0) and (0.6 <= cv <= 13.0)
    return dict(V=V, cv=cv, lam=lam, tau_deg=tau, R=r, D_f=d_f,
                wetted_length=lam * b, ehp_kw=r * V / 1e3, valid=valid,
                planing=cv > 1.5)


class PlaningHull:
    """A prismatic planing hull, for the time-domain planing plant
    (sim/planing_vessel.py): constant deadrise and chine beam over the
    afterbody; over the forward `taper` of the length the chine curves in
    to the stem and the keel rises to it by `stem_rise`. Lengths are
    measured from the TRANSOM, heights from the KEEL at the transom.

    Everything not measured on the vessel is a placeholder, and
    placeholders() lists it -- same contract as hydro.hull.Hull."""

    kind = "planing"
    propulsor = "waterjet"
    mesh_kind = "strips"            # listed by `python -m hydro.hulls`

    FIELDS = dict(
        L="hull length along the keel, transom to stem, m",
        b="chine beam, m", beta_deg="deadrise, deg", mass="kg",
        lcg="CG from the transom, m", vcg="CG above the keel, m",
        depth="keel to sheer at the bow, m", k_pitch="x L", k_roll="x b",
        k_yaw="x L", taper="bow fraction where the chine curves in",
        stem_rise="keel rise at the stem, m", cda="air drag C_D A, m^2",
        u_design="m/s")

    def __init__(self, name, L, b, beta_deg, mass, lcg, vcg, depth,
                 k_pitch=0.25, k_roll=0.35, k_yaw=0.25, taper=0.35,
                 stem_rise=0.0, cda=1.7, u_design=None, jet=None,
                 n_stations=48, k_a=1.0, c_dc=None, a_b=0.5, cd_lat=1.0,
                 roll_zeta=0.10, rho=RHO, given=(), sources=None,
                 transom_correction=False):
        self.name, self.L, self.b, self.beta_deg = name, L, b, beta_deg
        self.B = b
        self.mass, self.lcg, self.vcg, self.depth = mass, lcg, vcg, depth
        self.k_pitch, self.k_roll, self.k_yaw = k_pitch, k_roll, k_yaw
        self.taper, self.stem_rise, self.cda = taper, stem_rise, cda
        self.u_design = 25 * 0.514444 if u_design is None else u_design
        self.jet = dict(jet or {})
        self.n_stations = n_stations
        # The 2D+t coefficients are Zarnick's (1978): added mass k_a 1.0,
        # crossflow drag 1.0 cos(beta), buoyancy 0.5 at planing speed.
        # Against Fridsma's (1969) towed-model tests in regular waves they
        # give heave and pitch within ~20-30% and a stable run at both
        # speeds (studies/exp_planing.py P6). A calibration to Savitsky
        # alone picked k_a 0.6 -- and that model porpoised by itself at
        # V/sqrt(L) 4, where Fridsma's ran steadily.
        self.k_a, self.a_b = k_a, a_b
        self.c_dc = (np.cos(np.radians(beta_deg)) if c_dc is None
                     else c_dc)
        self.cd_lat, self.roll_zeta = cd_lat, roll_zeta
        self.transom_correction = transom_correction
        self.rho = rho
        self._given = set(given)
        self.sources = sources or {}
        self.extras = {}
        self.T = None               # static draft at the transom, set by
        #                             the plant on first build

    def scales(self):
        lam = self.L / 10.0
        return dict(lam=lam, u_design=self.u_design, dt=0.02, dt_ctrl=0.25,
                    mass=self.mass, weight=self.mass * G,
                    t_ref=np.sqrt(self.L / G))

    def plant(self, sea, db=None, dt=None, **kw):
        from sim.planing_vessel import PlaningVessel
        return PlaningVessel(self, sea, dt=self.scales()["dt"] if dt is None
                             else dt, **kw)

    def placeholders(self):
        out = [(k, v) for k, v in self.FIELDS.items() if k not in self._given]
        for k in ("d_nozzle", "p_max", "eta_pump", "nozzle_max_deg",
                  "nozzle_rate_deg", "x_intake", "tau"):
            if k not in self.jet or f"jet.{k}" not in self._given:
                out.append((f"jet.{k}", "placeholder"))
        out += [("k_a, c_dc, a_b", "2D+t coefficients, calibrated to "
                 "Savitsky in calm water"),
                ("lateral model", "slender-body lift + crossflow drag, "
                 "no manoeuvring data"),
                ("roll", "static GM, damping ratio placeholder")]
        return out


def speed_at_power(pump, power, mass, lcg, b, beta_deg, v_lo=None,
                   v_hi=40.0, **kw):
    """Steady speed at which the jet (sim/waterjet.JetPump) at shaft power
    `power` pushes as hard as Savitsky's resistance. None if it cannot reach
    the planing range (thrust < resistance at its low end)."""
    v_lo = 1.6 * np.sqrt(G * b) if v_lo is None else v_lo
    def surplus(V):
        return pump.thrust(power, V) - savitsky(V, mass, lcg, b, beta_deg,
                                                **kw)["R"]
    if surplus(v_lo) <= 0.0:
        return None
    if surplus(v_hi) > 0.0:
        return v_hi
    return brentq(surplus, v_lo, v_hi)
