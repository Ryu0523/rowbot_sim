#!/usr/bin/env python3
"""
Everything about a planing vessel that is not its geometry -- modules M2
(mass properties and the register of empirical closures), M4 (the waterjet
specification) and the plant's absolute constants (ACCEPTANCE G4) of the
modular split (learn/meta/ANYSPEED_PRINCIPLES_2026-10-05.md section 1) --
and ParametricPlaningHull, the hull assembled from them and from a
geometry (hydro/hull_geometry.py).

WHY
The Froude-twin test of the principles study (studies/principles_hull_swap.
py) could only make a 1.5x scaled scarab195 similar by patching module
constants of the frozen plant in-process: the water's viscosity (NU), two
time constants of the jet hard-coded where the plant builds it (rate_max =
t_max / 0.3 s, tau_prime_loss 0.05 s), the plant's step; it could not reach
the air-drag moment arm (0.5 m above the CG) at all. Every one of them is an
absolute number that does not follow the hull. Here each is a named,
dimensioned input with v0's value as its default, and the as plants
(sim/planing_anyspeed.py) read them from the hull:

    PlantNumerics   nu, rho_air, the Reynolds-number floors, the plant /
                    control / captive-run steps, the captive run's iteration
                    count and tolerances, the top speed's tolerance
    JetSpec         every jet input the plant reads (sim/planing_vessel.py
                    :118-138 and JetPump's loss coefficients) plus the
                    numbers v0 hard-codes (tau_prime_loss, t_rate, the
                    astern-thrust ratio t_min_div, the dead band's
                    reference pair)
    MassProps       mass, CG, radii of gyration
    Closures        the empirical coefficients the hull carries (k_a, a_b,
                    c_dc, cd_lat, roll_zeta, C_D A and its point of action
                    z_air), each with its register entry in learn/meta/
                    EMPIRICAL_REGISTER.md

Each class knows the Froude-scaling exponent of every field in powers of
the length scale lam (time lam^0.5, speed lam^0.5, mass lam^3, power
lam^3.5, kinematic viscosity lam^1.5 so that the Reynolds number -- and the
ITTC friction coefficient with it -- is unchanged). froude_twin(hull, lam)
builds the geometrically and dynamically similar vessel; on the as plants
every nondimensional output of the twin must equal the base hull's to
round-off (ACCEPTANCE M3, studies/anyspeed_p0_check.py part 'twin').

DEFAULTS: THE REFERENCE HULL, FROUDE-SCALED (ACCEPTANCE G4)
Every default above is scarab195's value (v0's literal): scarab195 is the
REFERENCE hull, keel length L_REF. A plain hydro.planing.PlaningHull keeps
those values as they are -- v0 builds it with its literals, and an as plant
with every switch off must equal v0 (jet_inputs completes its jet dict with
them). A ParametricPlaningHull that does not set an input gets the
reference value Froude-scaled to its own keel length (lam = L / L_REF, the
class's exponent), never the absolute number: a 13 m hull does not get a
0.1 s jet delay, a 1 cm Reynolds floor or a 10 um captive tolerance meant
for 5.4 m. Exceptions, kept as they are: the water's viscosity and the air's
density (physical constants of the fluid, not of the hull), and the plant
and control steps dt, dt_ctrl (the simulation's and the controller's
cadence, chosen by the harness: ACCEPTANCE E1 is set per 0.02 s step, the
reduced model and the MPC run at dt_ctrl; a hull that needs a finer step
sets it, as froude_twin does). The air-drag point z_air, a property of the
topsides, is derived from the geometry when not given (Closures). The
inputs a hull took from defaults are listed in its manifest
(ParametricPlaningHull.defaulted) and in learn/meta/EMPIRICAL_REGISTER.md.

v0 STAYS WHAT IT IS
The frozen v0 plant reads none of this: it builds its geometry from the
hull's L, b, taper, stem_rise and its jet from the literals in its
constructor. A hull whose inputs v0 cannot honour (a geometry outside v0's
family member, a non-default constant, a jet time constant v0 hard-codes)
is refused by sim/physics for v0 (v0_representable) and runs only as
'NAME@as1'.
"""
import warnings
from dataclasses import dataclass, asdict, fields
from types import SimpleNamespace

import numpy as np

from hydro.planing import PlaningHull, RHO
from hydro import hull_geometry as HG

# the reference hull (module docstring, DEFAULTS): scarab195's keel length,
# m (defined in hydro/hull_geometry). Its values are every default of this
# module; the protocol constants of sim/vessel_profile (turning marks) and
# control/mpc.froude_time_ref are referred to it too.
L_REF = HG.L_REF
# v0's air-drag point above the CG (sim/planing_vessel.py:298), what a plain
# PlaningHull runs on
V0_Z_AIR = 0.5


class _Spec:
    """A dataclass with a Froude exponent per field (EXP) and its defaults
    as v0's values."""

    EXP = {}

    def as_dict(self):
        return asdict(self)

    def froude(self, lam):
        d = self.as_dict()
        for k, e in self.EXP.items():
            if e and d.get(k) is not None:
                d[k] = d[k] * lam ** e
        return type(self)(**d)

    @classmethod
    def names(cls):
        return [f.name for f in fields(cls)]

    @classmethod
    def defaults(cls):
        return {f.name: f.default for f in fields(cls)}

    # fields whose default is NOT Froude-scaled for a parametric hull
    # (module docstring, DEFAULTS)
    KEEP = ()

    @classmethod
    def scaled_defaults(cls, L):
        """The defaults for a hull of keel length L: the reference values
        times (L / L_REF)^exponent, KEEP and None defaults as they are."""
        lam = float(L) / L_REF
        d = cls.defaults()
        for k, e in cls.EXP.items():
            if e and k not in cls.KEEP and d.get(k) is not None:
                d[k] = d[k] * lam ** e
        return d

    @classmethod
    def for_hull(cls, given, L):
        """(instance, names defaulted) from the inputs `given` (dict; unknown
        names raise) for a hull of keel length L, every missing input the
        scaled default."""
        given = dict(given or {})
        bad = sorted(set(given) - set(cls.names()))
        if bad:
            raise KeyError(f"{cls.__name__}: inputs {bad} unknown; known: "
                           f"{cls.names()}")
        d = cls.scaled_defaults(L)
        miss = sorted(k for k in cls.names() if k not in given)
        d.update(given)
        return cls(**d), miss


# ------------------------------------------------------- plant constants
@dataclass
class PlantNumerics(_Spec):
    """The absolute constants of the planing plant, v0's values as the
    defaults (each is a literal in sim/planing_vessel.py; the line is given).
    """
    nu: float = 1.19e-6        # :57  m^2/s, sea water 15 C (ITTC 7.5-02-01-03)
    rho_air: float = 1.225     # :56  kg/m^3, ISA sea level
    u_floor: float = 0.05      # :287 m/s, Reynolds number floor (speed)
    l_floor: float = 0.05      # :288 m, Reynolds number floor (length)
    dt: float = 0.02           # hydro/planing.py:138, plant step, s
    dt_ctrl: float = 0.25      # hydro/planing.py:138, control step, s
    dt_captive: float = 0.005  # :407 s, captive run to the running attitude
    n_captive: int = 4000      # :409 steps, at most
    n_check: int = 200         # :415 steps between convergence checks
    tol_z: float = 1e-5        # :417 m, captive convergence, heave
    tol_th: float = 1e-6       # :418 rad, captive convergence, pitch
    u_tol: float = 0.01        # planing_vessel_gz.py:301 m/s, top speed

    EXP = dict(nu=1.5, rho_air=0, u_floor=0.5, l_floor=1, dt=0.5,
               dt_ctrl=0.5, dt_captive=0.5, n_captive=0, n_check=0, tol_z=1,
               tol_th=0, u_tol=0.5)
    # physical constants and the harness's cadence (module docstring)
    KEEP = ("nu", "rho_air", "dt", "dt_ctrl")


# ----------------------------------------------------------------- jet
@dataclass
class JetSpec(_Spec):
    """Every input the planing plant's waterjet reads, v0's defaults
    (sim/planing_vessel.py:118-138; JetPump's zeta_n, zeta_i, wake from
    sim/waterjet.py:57-58). Positions: x from the transom, z above the keel
    at the transom. Hard-coded in v0 (:129-133) and read from here by the
    as plants: tau_prime_loss=0.05, rate_max=t_max / t_rate (0.3 s),
    t_min=-t_max / t_min_div (3), dead_band=t_max * dead_band_ref /
    dead_band_ref_thrust (50 N per 12 kN: the 10 m USV's dead band scaled
    by thrust, register J8 -- kept as v0's two numbers so the expression
    evaluates exactly as v0's)."""
    d_nozzle: float = 0.085
    p_max: float = 224e3
    eta_pump: float = 0.775
    zeta_n: float = 0.02
    zeta_i: float = 0.25
    wake: float = 0.05
    x_nozzle: float = 0.0
    z_nozzle: float = 0.15
    x_intake: float = 0.8
    z_intake: float = 0.0
    k_rev: float = 0.6
    tau: float = 0.6
    t_rate: float = 0.3
    tau_prime_loss: float = 0.05
    tau_prime_back: float = 0.5
    delay: float = 0.1
    nozzle_max_deg: float = 25.0
    nozzle_rate_deg: float = 40.0
    nozzle_tau: float = 0.1
    nozzle_delay: float = 0.1
    t_min_div: float = 3.0
    dead_band_ref: float = 50.0
    dead_band_ref_thrust: float = 12000.0

    EXP = dict(d_nozzle=1, p_max=3.5, eta_pump=0, zeta_n=0, zeta_i=0,
               wake=0, x_nozzle=1, z_nozzle=1, x_intake=1, z_intake=1,
               k_rev=0, tau=0.5, t_rate=0.5, tau_prime_loss=0.5,
               tau_prime_back=0.5, delay=0.5, nozzle_max_deg=0,
               nozzle_rate_deg=-0.5, nozzle_tau=0.5, nozzle_delay=0.5,
               # ratios (the dead band's pair: only their ratio enters)
               t_min_div=0, dead_band_ref=0, dead_band_ref_thrust=0)

    @classmethod
    def of(cls, jet):
        """From a hull's jet dict; unknown keys raise (a misspelt key would
        otherwise run on a default)."""
        bad = sorted(set(jet) - set(cls.names()))
        if bad:
            raise KeyError(f"jet inputs {bad} unknown; known: {cls.names()}")
        return cls(**{k: v for k, v in jet.items()})


# the keys v0's constructor reads (sim/planing_vessel.py:118-138); the rest
# of JetSpec is honoured by the as plants only
V0_JET_KEYS = ("d_nozzle", "p_max", "eta_pump", "x_nozzle", "z_nozzle",
               "x_intake", "z_intake", "k_rev", "tau", "tau_prime_back",
               "delay", "nozzle_max_deg", "nozzle_rate_deg", "nozzle_tau",
               "nozzle_delay")


# ---------------------------------------------------------- mass, closures
@dataclass
class MassProps(_Spec):
    mass: float
    lcg: float                 # m from the transom
    vcg: float                 # m above the keel at the transom
    k_pitch: float = 0.25      # x L
    k_roll: float = 0.35       # x b
    k_yaw: float = 0.25        # x L

    EXP = dict(mass=3, lcg=1, vcg=1, k_pitch=0, k_roll=0, k_yaw=0)


@dataclass
class Closures(_Spec):
    """The empirical coefficients a planing hull carries; defaults as
    hydro/planing.PlaningHull's. Register ids: learn/meta/
    EMPIRICAL_REGISTER.md."""
    k_a: float = 1.0           # E1  Zarnick 1978 added-mass coefficient
    a_b: float = 0.5           # E2  Zarnick 1978 buoyancy factor (planing)
    c_dc: float = None         # E3  crossflow drag, None: cos(beta)
    cd_lat: float = 1.0        # E6  lateral crossflow drag (placeholder)
    roll_zeta: float = 0.10    # E6  roll damping ratio (no source)
    cda: float = 1.7           # E7  air drag C_D A, m^2 (placeholder)
    transom_correction: bool = False   # E9 Garme 2005 (off)
    # E8  air-drag point above the CG, m; None: from the geometry
    # (air_point_of_form: the topside centroid of the widest section)
    z_air: float = None

    EXP = dict(k_a=0, a_b=0, c_dc=0, cd_lat=0, roll_zeta=0, cda=2, z_air=1)


# --------------------------------------------- the transom flow state (S2a)
# The empirical closure of the as1 switch 'transom' (sim/planing_anyspeed.py,
# 'S2a TRANSOM'; learn/meta/EMPIRICAL_REGISTER.md E14-E16). Not a field of
# PlaningHull: a hull carries it only as the optional attribute
# `transom_closure` (a dict of overrides), so the manifests of the hulls in
# the repo -- scarab195's is pinned (sim/physics.LEGACY) -- are unchanged and
# every hull runs on these registered values unless a study gives it others
# (with_transom_closure: a renamed copy, its own manifest and caches).
#
#   F_T_crit  the transom-draught Froude number U / sqrt(g d_T) at which the
#             transom runs dry (full ventilation). Empirical: Doctors,
#             Macfarlane & Young 2007 (ISP 54(2)), Maki et al. 2005, Saunders
#             1957 -- scatter 2.0-4.5 (register E14). Default: the MIDDLE of
#             that registered scatter, chosen before any run and not
#             compared with any validation data (ACCEPTANCE G3: no single
#             tuned value; studies sweep and randomise over the scatter).
#             inf: the transom never ventilates (a closed body; used by the
#             Froude-Krylov identity checks)
#   mode      'progressive' (primary, pre-registered): the dead-water level
#             at the transom sits a constant base-pressure head
#             U^2 / (g F_T_crit^2) below the local surface, so the wetted
#             height h_w = d_T - U^2 / (g F_T_crit^2) falls continuously and
#             reaches zero at F_T = F_T_crit (register E15: the form is an
#             assumption -- a constant base-pressure coefficient
#             C_pb = -2 / F_T_crit^2, i.e. a base-drag coefficient
#             C_D,base = 2 / F_T_crit^2 on the immersed transom area before
#             clearance -- the threshold is the data). It applies F_T_crit
#             to the INSTANTANEOUS transom draught d_T; the sources define
#             it on the static draught T_T0 (register E14).
#             For the sensitivity study only (refused unless `sensitivity`
#             is set -- ACCEPTANCE G1 forbids a speed-indexed switch outside
#             one):
#             'progressive_static': the same progression with the threshold
#             on the static draught, h_w = d_T (1 - U^2 / (g F_T_crit^2
#             T_T0)) (the sources' variable);
#             'step': fully wet below F_T_crit on the STATIC transom
#             draught, dry above -- no progression; an alternative form on
#             a different variable, NOT a bound of the primary form
#   cd_face   drag coefficient of the transom face when it LEADS (astern,
#             u < 0), on the immersed face area: Hoerner 1965 'Fluid-Dynamic
#             Drag' ch. 3, flat-faced forebody 0.8 (range 0.75-1.2; register
#             E16)
#   sensitivity  True only in sensitivity studies and tests: admits the
#             alternative modes above (with_transom_closure passes it on)
TRANSOM_CLOSURE = dict(F_T_crit=3.25, mode="progressive", cd_face=0.8,
                       sensitivity=False)
TRANSOM_SCATTER = (2.0, 4.5)           # register E14
TRANSOM_MODES = ("progressive", "progressive_static", "step")
TRANSOM_PRIMARY = "progressive"


def transom_closure_of(hull):
    """The transom closure `hull` runs on: TRANSOM_CLOSURE with the hull's
    own `transom_closure` overrides (if it carries any). Unknown names, a
    non-positive threshold, an unknown mode, an alternative mode without
    the sensitivity flag or a negative drag coefficient raise."""
    c = dict(TRANSOM_CLOSURE)
    own = dict(getattr(hull, "transom_closure", None) or {})
    bad = sorted(set(own) - set(c))
    if bad:
        raise KeyError(f"hull {getattr(hull, 'name', hull)!r}: "
                       f"transom_closure {bad} unknown; known: {sorted(c)}")
    c.update(own)
    c["F_T_crit"] = float(c["F_T_crit"])
    c["cd_face"] = float(c["cd_face"])
    if not c["F_T_crit"] > 0.0:
        raise ValueError("transom_closure: F_T_crit must be > 0 (inf: never "
                         "ventilates)")
    if c["mode"] not in TRANSOM_MODES:
        raise ValueError(f"transom_closure: mode one of {TRANSOM_MODES}")
    c["sensitivity"] = bool(c["sensitivity"])
    if c["mode"] != TRANSOM_PRIMARY and not c["sensitivity"]:
        raise ValueError(
            f"transom_closure: mode {c['mode']!r} switches the transom state "
            f"on speed through the static draught; ACCEPTANCE G1 admits it "
            f"only in a sensitivity study or a test: set sensitivity=True")
    if c["cd_face"] < 0.0:
        raise ValueError("transom_closure: cd_face >= 0")
    return SimpleNamespace(**c)


def with_transom_closure(hull, name, **closure):
    """A copy of the v0 planing hull `hull` (a registered hull or any plain /
    parametric one, not yet a variant) carrying `transom_closure` = closure,
    put on as1 under the name `name` (sim/physics.variant: a modified hull
    needs a name of its own, so its caches -- keyed by name and manifest --
    are its own). For sensitivity studies and randomisation over the
    registered scatter (ACCEPTANCE G3); an alternative mode ('step',
    'progressive_static') also needs sensitivity=True in the closure."""
    import copy as _copy
    from sim import physics
    h = _copy.deepcopy(hull)
    h.transom_closure = dict(closure)
    transom_closure_of(h)                       # validate
    return physics.variant(h, "as1", name=name)


def air_point_of_form(form, vcg):
    """The air-drag point above the CG derived from the geometry, m: the
    centroid of the topside of the widest section, wall-sided from the
    chine (or the sheer, if the chine is above it) to the sheer -- midway
    between them, as the GZ sections are drawn (sim/planing_vessel_gz.
    _sections). The hull's topsides only: no superstructure, windscreen or
    crew (placeholder grade, register E8)."""
    st = form.stations()
    j = int(np.argmax(st.yc))
    zc = st.rise[j] + min(st.yc[j] * np.tan(np.radians(st.beta_deg)),
                          st.depth)
    return float(0.5 * (zc + st.rise[j] + st.depth) - vcg)


# ------------------------------------------------------------ accessors
def numerics_of(hull):
    """The plant constants of `hull`: its own `numerics` (a parametric hull
    carries all of them) over v0's values (a plain PlaningHull), plus the
    air-drag point z_air -- the hull's closure (Closures.z_air) or, for a
    plain hull, v0's 0.5 m. Unknown names raise."""
    d = PlantNumerics.defaults()
    own = dict(getattr(hull, "numerics", None) or {})
    bad = sorted(set(own) - set(d))
    if bad:
        raise KeyError(f"hull {getattr(hull, 'name', hull)!r}: numerics "
                       f"{bad} unknown; known: {sorted(d)}")
    d.update(own)
    z = getattr(hull, "z_air", None)
    d["z_air"] = V0_Z_AIR if z is None else float(z)
    return SimpleNamespace(**d)


def jet_inputs(hull):
    """The hull's jet inputs as a complete dict (JetSpec.of(hull.jet).
    as_dict()): every key present, so no literal default of the plant's
    constructor ever applies -- one source of truth. A plain PlaningHull's
    missing keys take JetSpec's defaults, which are v0's literals (so the
    jet is v0's, bit for bit); a parametric hull's jet is complete already
    (its defaults Froude-scaled, module docstring)."""
    return JetSpec.of(hull.jet).as_dict()


def v0_representable(hull):
    """(ok, reasons): can the frozen v0 plant build `hull` as its inputs
    say? Not if its geometry is outside v0's family member, a plant
    constant differs from v0's literal (the steps excepted: v0 takes dt
    from the caller, hull.scales()), or a jet input v0 hard-codes differs
    from v0's value."""
    why = []
    if not HG.v0_geometry(hull):
        why.append("geometry outside v0's family member")
    num = vars(numerics_of(hull))
    for k, v in PlantNumerics.defaults().items():
        if k not in ("dt", "dt_ctrl") and num[k] != v:
            why.append(f"numerics.{k} = {num[k]!r} (v0: {v!r})")
    if num["z_air"] != V0_Z_AIR:
        why.append(f"z_air = {num['z_air']!r} (v0 hard-codes {V0_Z_AIR!r})")
    # a JetSpec key v0 does not read, at a value other than the one v0
    # hard-codes (keys JetSpec does not know are ignored by v0 as before;
    # the as plants refuse them, jet_inputs)
    jd = JetSpec.defaults()
    for k, v in getattr(hull, "jet", {}).items():
        if k in jd and k not in V0_JET_KEYS and v != jd[k]:
            why.append(f"jet.{k} = {v!r} (v0 hard-codes {jd[k]!r})")
    return not why, why


# ------------------------------------------------------ the parametric hull
class ParametricPlaningHull(PlaningHull):
    """A PlaningHull assembled from a geometry (hydro/hull_geometry form),
    MassProps, JetSpec, Closures and PlantNumerics. Its PlaningHull fields
    are filled from them (so everything that reads a PlaningHull reads the
    same numbers), and it carries two fields of its own, `form` and
    `numerics`, which the as plants read and which enter its manifest
    (sim/physics.hull_fields) -- so every cache keyed by the hull's sha1
    separates hulls that differ in any of them."""

    def __init__(self, name, form, mass, jet=None, closures=None,
                 numerics=None, u_design=None, rho=RHO, given=(),
                 sources=None):
        f = form
        # an input given as an instance is explicit throughout; as a dict,
        # the missing names take the reference value Froude-scaled to this
        # hull (module docstring, DEFAULTS) and are listed in `defaulted`
        dflt = {}
        if isinstance(f, dict):
            spec = dict(f)
            f = HG.FORMS[spec.pop("kind", "family")](**spec)
            if "yc_floor" not in spec:
                dflt["form"] = ["yc_floor"]
        if isinstance(mass, MassProps):
            mp = mass
        else:
            mp = MassProps(**mass)
            dflt["mass"] = sorted(k for k in ("k_pitch", "k_roll", "k_yaw")
                                  if k not in mass)
        if isinstance(jet, JetSpec):
            js = jet
        else:
            js, dflt["jet"] = JetSpec.for_hull(jet, f.L)
        if isinstance(closures, Closures):
            cl = closures
        else:
            cl, dflt["closures"] = Closures.for_hull(closures, f.L)
        if isinstance(numerics, PlantNumerics):
            nm = numerics
        else:
            nm, dflt["numerics"] = PlantNumerics.for_hull(numerics, f.L)
        if cl.z_air is None:
            cl.z_air = air_point_of_form(f, mp.vcg)
            dflt.setdefault("closures", [])
            if "z_air" not in dflt["closures"]:
                dflt["closures"] = sorted(dflt["closures"] + ["z_air"])
        # MassProps against the geometry
        if not (mp.mass > 0 and 0.0 < mp.lcg < f.L
                and 0.0 < mp.vcg < f.depth
                and min(mp.k_pitch, mp.k_roll, mp.k_yaw) > 0):
            raise ValueError(f"{name!r}: mass > 0, 0 < lcg < L, 0 < vcg < "
                             f"depth and positive radii of gyration "
                             f"({mp.as_dict()}, L {f.L}, depth {f.depth})")
        out = f.validity()
        if out:
            warnings.warn(f"{name!r} is outside the planing family's range "
                          f"of validity (hydro/hull_geometry.RANGES): "
                          + "; ".join(out))
        super().__init__(
            name, L=f.L, b=f.max_beam(), beta_deg=f.beta_deg, mass=mp.mass,
            lcg=mp.lcg, vcg=mp.vcg, depth=f.depth, k_pitch=mp.k_pitch,
            k_roll=mp.k_roll, k_yaw=mp.k_yaw,
            # v0's family fields; an offsets hull has no taper (consumers
            # read hull_geometry.stations_of, not these fields)
            taper=getattr(f, "taper", 0.0), stem_rise=f.stem_rise,
            cda=cl.cda, u_design=u_design, jet=js.as_dict(),
            n_stations=f.n_stations, k_a=cl.k_a, c_dc=cl.c_dc, a_b=cl.a_b,
            cd_lat=cl.cd_lat, roll_zeta=cl.roll_zeta, rho=rho, given=given,
            sources=sources, transom_correction=cl.transom_correction)
        self.form = dict(kind=f.KIND, **f.as_dict())
        self.numerics = nm.as_dict()
        self.z_air = float(cl.z_air)
        # the inputs taken from defaults (scaled reference values, or z_air
        # from the geometry): part of the manifest
        self.defaulted = {k: v for k, v in sorted(dflt.items()) if v}

    def scales(self):
        d = super().scales()
        d["dt"] = self.numerics["dt"]
        d["dt_ctrl"] = self.numerics["dt_ctrl"]
        return d

    def specs(self):
        """(form, MassProps, JetSpec, Closures, PlantNumerics) of this
        hull."""
        spec = dict(self.form)
        f = HG.FORMS[spec.pop("kind")](**spec)
        return (f, MassProps(self.mass, self.lcg, self.vcg, self.k_pitch,
                             self.k_roll, self.k_yaw),
                JetSpec.of(self.jet),
                Closures(k_a=self.k_a, a_b=self.a_b, c_dc=self.c_dc,
                         cd_lat=self.cd_lat, roll_zeta=self.roll_zeta,
                         cda=self.cda,
                         transom_correction=self.transom_correction,
                         z_air=self.z_air),
                PlantNumerics(**self.numerics))


def parametric_of(hull, name=None):
    """The ParametricPlaningHull with exactly a plain PlaningHull's inputs:
    the family member of its geometry, its mass, jet (every default made
    explicit), closures, v0's plant constants. On the as plants it runs
    bit for bit as `hull` itself (studies/anyspeed_p0_check.py)."""
    f = HG.form_of(hull)
    jet = JetSpec.of(dict(hull.jet))
    num0 = numerics_of(hull)
    cl = Closures(k_a=hull.k_a, a_b=hull.a_b, c_dc=hull.c_dc,
                  cd_lat=hull.cd_lat, roll_zeta=hull.roll_zeta, cda=hull.cda,
                  transom_correction=bool(getattr(hull, "transom_correction",
                                                  True)),
                  z_air=num0.z_air)
    mp = MassProps(hull.mass, hull.lcg, hull.vcg, hull.k_pitch, hull.k_roll,
                   hull.k_yaw)
    num = PlantNumerics(**{k: v for k, v in vars(num0).items()
                           if k != "z_air"})
    if not hasattr(hull, "numerics"):
        sc = hull.scales()
        num.dt, num.dt_ctrl = sc["dt"], sc["dt_ctrl"]
    out = ParametricPlaningHull(
        name or f"{hull.name.split('@')[0]}_param", f, mp, jet=jet,
        closures=cl, numerics=num, u_design=hull.u_design, rho=hull.rho)
    _carry_transom(hull, out)
    return out


def _carry_transom(src, dst):
    """A transom closure override (dimensionless: Froude number, mode, drag
    coefficient) is carried to a hull derived from `src`."""
    tc = getattr(src, "transom_closure", None)
    if tc:
        dst.transom_closure = dict(tc)


def froude_twin(hull, lam, name=None):
    """The vessel similar to `hull` at length scale lam: every input scaled
    by its Froude exponent (geometry, mass properties, jet, C_D A, plant
    constants including the viscosity and every time step; speeds lam^0.5).
    Same water and air density, same gravity."""
    p = parametric_of(hull)
    f, mp, js, cl, nm = p.specs()
    out = ParametricPlaningHull(
        name or f"{hull.name.split('@')[0]}_froude{lam:g}", f.froude(lam),
        mp.froude(lam), jet=js.froude(lam), closures=cl.froude(lam),
        numerics=nm.froude(lam), u_design=hull.u_design * lam ** 0.5,
        rho=hull.rho)
    _carry_transom(hull, out)
    return out
