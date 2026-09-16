#!/usr/bin/env python3
"""
One description of a vessel, so the model is not about one hull.

Everything in this project was reachable only through the Wigley test hull:
`NonlinearVessel` built `WigleySections(L, B, T)` itself, the hydrodynamic
database had the hull's name in the filename, the inertia had a closed form
written for it, and studies hard-coded L = 10, B = 2.5, T = 0.8 in a dozen
places. None of that is wrong for a test hull. All of it is in the way the
moment there is a real vessel to import.

A `Hull` carries what a run needs about the vessel:

    geometry     a panel mesh, from an analytic form, a mesh file or CAD
    mass         displacement, centre of gravity, radii of gyration
    operation    design speed
    appendages   thrusters and rudders: where, how big, how fast
    damping      viscous and roll damping, manoeuvring derivatives, wind areas
    hydro        the BEM database, built from the mesh and cached on disk

WHAT A FIELD LEFT AT None MEANS

A placeholder, not data. Every one has a stand-in -- the 10 m USV's value
Froude-scaled, a published regression, a geometric default -- so a run always
goes, and `placeholders()` lists what is still standing in, so the list of what
the real vessel owes the model is printed, not remembered. `check()` prints it.

WHAT MAKES THIS MORE THAN A STRUCT

  displacement closes      the mesh volume, the mass, and rho must agree.
  symmetry is declared     a symmetric hull's cross-coupling entries are
                           exactly zero and are imposed (`hydro/symmetry.py`).
  gyradii are explicit     the geometric value is the inertia of a hull-shaped
                           block of water: 2.4x too little roll inertia.
  the origin is the CG     the plant takes moments and rotates about x = 0; a
                           file's own origin (KVLCC2's is the aft perpendicular)
                           would put the rotation centre 160 m from the ship.
  the geometry is stated   `mesh_kind` has no default: a Hull(name, L, B, T)
                           used to become a Wigley without a word.

Run: python -m hydro.hull
"""
import hashlib
import os
from dataclasses import dataclass, field

import numpy as np

RHO = 1025.0
G = 9.81
USV_L = 10.0          # the length every placeholder was chosen for

# the fields that are data about the vessel; None = placeholder
_DATA = ("displacement", "z_cog", "x_cog", "k_roll", "k_pitch", "k_yaw",
         "freeboard", "u_design", "x_rud", "z_rud", "rudder_area",
         "rudder_span", "x_prop", "z_prop", "d_prop", "thrusters", "rudders",
         "prop_series", "t_max", "rudder_max_deg", "rudder_rate_deg", "visc",
         "bilge_keel", "Yv_prime", "Nr_prime", "wind_areas")


@dataclass
class Hull:
    """Everything about one vessel that the simulation needs."""

    name: str
    L: float
    B: float
    T: float

    # geometry source -- stated, never defaulted (see the module note)
    mesh_kind: str = None           # "wigley" | "file" | "object" | "cad"
    mesh_path: str = None
    # "cad": keyword arguments for hydro.cad_import.import_hull -- units
    # (scale), axes (forward, up), half, size_max, curvature, min_component.
    # Plain values only (no callables): they go into the cache key.
    cad: dict = None
    n_stations: int = 41
    mesh_res: tuple = (60, 20)

    # mass
    displacement: float = None      # kg; None = rho x mesh volume
    rho: float = RHO                # 1000 for a fresh-water tank model
    z_cog: float = None             # m, up from the waterline; None = -T/3
    # LCG, m. "cad": in the file's frame after `cad` axes and scale; None =
    # the centre of buoyancy (level trim). "file"/"object": relative to the
    # mesh origin; None = the origin is taken to be the CG.
    x_cog: float = None
    k_roll: float = None            # x B; None = the bare hull's geometry
    k_pitch: float = None           # x L
    k_yaw: float = None             # x L
    symmetric: bool = True
    multihull: bool = False         # stated: the roll-damping method depends on it
    freeboard: float = None         # m, at the bow

    # operation
    u_design: float = None          # m/s; None = the USV's 4.5 m/s Froude-scaled

    # appendages. Positions are relative to the CG (after the origin move);
    # None = the 10 m USV's proportions, measured from the STERN of this hull
    x_rud: float = None
    z_rud: float = None
    rudder_area: float = None       # one blade, m^2
    rudder_span: float = None
    x_prop: float = None
    z_prop: float = None
    d_prop: float = None
    # several units sharing ONE command (twin screws, twin outboards, twin
    # rudders): ((x, y, z), ...). None = one on the centreline at x_prop/z_prop
    # (x_rud/z_rud). Differential thrust as a control is not modelled.
    thrusters: tuple = None
    rudders: tuple = None
    prop_series: dict = None        # Wageningen B: dict(Z=, AE_A0=, P_D=)
    # total thrust limit, N. None = the USV's margin over its resistance
    # (2.1x) at this vessel's u_design, on the resistance placeholder
    t_max: float = None
    rudder_max_deg: float = None
    rudder_rate_deg: float = None

    # damping, manoeuvring, wind
    visc: tuple = None              # six quadratic coefficients, SI
    # "auto": simplified Ikeda, flagged when the hull is outside its range
    # (a multihull keeps the placeholder); "ikeda"; "placeholder"; or a decay
    # test's own fit, dict(zeta1=, k2=, T=) as studies/exp_kvlcc2_rolldecay.py
    # fits it -- the plant subtracts the BEM wave part itself
    roll_damping: object = "auto"
    bilge_keel: tuple = None        # (width, length), m
    Yv_prime: float = None          # None = Clarke et al. (1983)
    Nr_prime: float = None
    wind_areas: dict = None         # dict(A_F=, A_L=, s_H=, s_L=, vessel=)

    extras: dict = field(default_factory=dict)

    # ------------------------------------------------------------- defaults
    def __post_init__(self):
        if self.mesh_kind not in ("wigley", "file", "object", "cad"):
            raise ValueError(
                f"{self.name}: mesh_kind must be stated -- 'wigley', 'file', "
                f"'object' or 'cad', not {self.mesh_kind!r}. It used to "
                f"default to 'wigley', and a Hull(name, L, B, T) for a new "
                f"vessel quietly became the old test hull.")
        self._given = {k for k in _DATA if getattr(self, k) is not None}
        if self.z_cog is None:
            self.z_cog = -self.T / 3
        if self.freeboard is None:
            self.freeboard = 0.55 * self.T / 0.8 if self.T else 0.55
        if self.z_rud is None:
            self.z_rud = -0.56 * self.T
        if self.rudder_span is None:
            self.rudder_span = 0.69 * self.T
        if self.rudder_area is None:
            # 1.8% of L*T, the usual first cut. This used to be multiplied by
            # 5.0 -- five times the rule of thumb its own comment quoted.
            self.rudder_area = 0.018 * self.L * self.T
        if self.z_prop is None:
            self.z_prop = -0.69 * self.T
        if self.d_prop is None:
            self.d_prop = 0.55 * self.T
        self._mesh = None
        self._form = None
        self._x_shift = 0.0

    def _fill_x(self):
        """Default rudder and propeller positions, from the hull's STERN.

        They were -0.49 L and -0.46 L from the origin: right for the Wigley,
        whose origin is midship, and 15 m inside the hull of a KVLCC2 whose
        CG is 11 m forward of midship. The stern comes from the mesh, so this
        waits until the mesh exists."""
        if self.x_rud is None or self.x_prop is None:
            stern = float(self.sections().x_stern)
            if self.x_rud is None:
                self.x_rud = stern + 0.01 * self.L
            if self.x_prop is None:
                self.x_prop = stern + 0.04 * self.L

    # -------------------------------------------------------------- geometry
    def mesh(self):
        """The hull mesh, waterline at z = 0, origin at the CG. Built once."""
        if self._mesh is None:
            import capytaine as cpt
            x_ref = 0.0
            if self.mesh_kind == "wigley":
                from hydro.geometry import wigley_mesh
                m = wigley_mesh(self.L, self.B, self.T, *self.mesh_res)
            elif self.mesh_kind == "file":
                m = cpt.load_mesh(self.mesh_path)
            elif self.mesh_kind == "object":
                m = self.extras["mesh"]
            else:
                if self.displacement is None:
                    raise ValueError(f"{self.name}: a CAD hull needs its "
                                     f"displacement -- the waterline is found "
                                     f"from it, not taken from the file")
                m, rep = self._cad_mesh()
                self._cad_report = rep
                x_ref = rep["x_B"]
                d = rep["draft"]
                if abs(d - self.T) > 0.01 * self.T:
                    print(f"  WARNING {self.name}: the lines displace "
                          f"{self.displacement:.4g} kg at a draft of {d:.4g} m,"
                          f" not the stated T = {self.T:.4g} m. One of the "
                          f"displacement, T, the units or the axes is wrong.")
            x0 = self.x_cog if self.x_cog is not None else x_ref
            self._x_shift = -float(x0)
            self._mesh = m.translated_x(-x0) if x0 else m
        return self._mesh

    def _cad_mesh(self):
        """Import the CAD file, or load the result of an earlier import.

        Cached next to the BEM database, keyed on everything the import
        depends on. An import takes a minute of gmsh; an RL run starts one
        plant per worker process, and a CAD vessel would otherwise re-import
        its IGES in every one of them."""
        import capytaine as cpt
        from hydro.cad_import import import_hull
        src = os.path.getsize(self.mesh_path) if os.path.exists(
            self.mesh_path) else None
        key = (self.name, self.mesh_path, src, self.displacement, self.rho,
               tuple(sorted((self.cad or {}).items())))
        h = hashlib.sha1(repr(key).encode()).hexdigest()[:8]
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), f"mesh_{self.name}_{h}.npz")
        keys = ("draft", "x_B", "volume", "KM", "wetted_area", "L_wl",
                "B_wl", "n_panels")
        if os.path.exists(path):
            d = np.load(path)
            half = cpt.Mesh(vertices=d["vertices"], faces=d["faces"])
            m = (cpt.ReflectionSymmetricMesh(half=half, plane="xOz")
                 if bool(d["symmetric"]) else half)
            return m, {k: float(d[k]) for k in keys}
        m, rep = import_hull(self.mesh_path,
                             volume=self.displacement / self.rho,
                             verbose=False, **(self.cad or {}))
        sym = hasattr(m, "half")
        base = m.half if sym else m
        np.savez(path, vertices=np.asarray(base.vertices),
                 faces=np.asarray(base.faces), symmetric=sym,
                 **{k: float(rep[k]) for k in keys})
        return m, rep

    def body(self):
        import capytaine as cpt
        m = self.mesh()
        body = cpt.FloatingBody(
            mesh=m, lid_mesh=m.generate_lid(z=0.0),
            center_of_mass=(0.0, 0.0, self.z_cog))
        body.add_all_rigid_body_dofs()
        return body.immersed_part()

    def sections(self):
        """Station table. Analytic where one exists, from the mesh otherwise."""
        if getattr(self, "_sections", None) is None:
            from sim.sections import WigleySections, MeshSections
            if self.mesh_kind == "wigley":
                self._sections = WigleySections(self.L, self.B, self.T,
                                                self.n_stations)
            else:
                self._sections = MeshSections(self.mesh(), self.L, self.T,
                                              self.n_stations,
                                              freeboard=self.freeboard)
        return self._sections

    def form(self):
        """Volume, centre of buoyancy (relative to the CG), C_B, C_M and
        wetted area, from the mesh itself."""
        if self._form is None:
            m = self.mesh()
            if hasattr(m, "merged"):
                m = m.merged()
            wet = m.immersed_part()
            c, n, a = wet.faces_centers, wet.faces_normals, wet.faces_areas
            V = float(np.sum(c[:, 1] * n[:, 1] * a))
            x_B = float(np.sum(0.5 * c[:, 0] ** 2 * n[:, 0] * a)) / V
            sec = self.sections()
            A0 = sec.area(np.zeros(sec.n))
            self._form = dict(V=V, x_B=x_B,
                              C_B=V / (self.L * self.B * self.T),
                              C_M=float(A0.max()) / (self.B * self.T),
                              wetted=float(a.sum()))
        return self._form

    def scales(self):
        """Characteristic numbers every consumer derives its own from."""
        lam = self.L / USV_L
        m = (self.displacement if self.displacement is not None
             else self.rho * self.form()["V"])
        return dict(lam=lam,
                    u_design=(self.u_design if self.u_design is not None
                              else 4.5 * np.sqrt(lam)),
                    dt=0.05 * np.sqrt(lam), dt_ctrl=0.5 * np.sqrt(lam),
                    mass=m, weight=m * G, t_ref=np.sqrt(self.L / G))

    def plant(self, sea, db=None, dt=None, **kw):
        """The nonlinear plant for this vessel in `sea`."""
        from sim.vessel import NonlinearVessel
        if db is None:
            db = self.database(verbose=False)
        return NonlinearVessel(db, sea, hull=self,
                               dt=self.scales()["dt"] if dt is None else dt,
                               **kw)

    # --------------------------------------------------------- placeholders
    def placeholders(self):
        """(field, what stands in for it) for every input that is not data."""
        g = self._given
        out = []

        def need(k, what):
            if k not in g:
                out.append((k, what))
        need("displacement", "rho x the mesh volume")
        need("z_cog", "-T/3, the Wigley rule")
        if self.mesh_kind == "cad":
            need("x_cog", "the centre of buoyancy (level trim)")
        for k in ("k_roll", "k_pitch", "k_yaw"):
            need(k, "the bare hull's geometry -- no machinery, no payload")
        need("u_design", "the USV's 4.5 m/s, Froude-scaled")
        need("freeboard", "0.55 T / 0.8, the USV's proportion")
        for k in ("x_rud", "z_rud", "rudder_area", "rudder_span", "x_prop",
                  "z_prop", "d_prop"):
            need(k, "the USV's proportions")
        need("prop_series", "a linear K_T(J), 1.6x too steep on KVLCC2")
        need("t_max", "the USV's thrust margin (2.1x its resistance) at "
             "this design speed; lags Froude-scaled")
        need("rudder_max_deg", "35 deg")
        need("rudder_rate_deg", "25 deg/s Froude-scaled")
        need("visc", "the USV's quadratic damping, Froude-scaled")
        if self.roll_damping == "placeholder" or (
                self.roll_damping == "auto" and self.multihull):
            out.append(("roll_damping", "the USV's quadratic, Froude-scaled"))
        elif self.roll_damping in ("auto", "ikeda"):
            out.append(("roll_damping", "simplified Ikeda (KVLCC2, outside "
                        "its range: 1.6x the measured decay at 0 kn, 0.75x "
                        "at 15.5 kn)"))
        need("Yv_prime", "Clarke et al. (1983), merchant-hull regression")
        need("Nr_prime", "Clarke et al. (1983)")
        need("wind_areas", "a box-shaped topside, Froude-scaled lever")
        return out

    # ------------------------------------------------------------------ mass
    def inertia(self):
        """6x6 rigid-body inertia, geometry corrected then gyradii applied."""
        from hydro.inertia import (wigley_inertia, inertia_from_mesh,
                                   with_gyradius)
        if self.mesh_kind == "wigley":
            M, _ = wigley_inertia(self.L, self.B, self.T, self.z_cog)
        else:
            M, _ = inertia_from_mesh(self.body().mesh,
                                     (0.0, 0.0, self.z_cog),
                                     mass=self.displacement)
        if self.displacement is not None:
            M = M * (self.displacement / M[0, 0])
        return with_gyradius(M, self.L, self.B,
                             self.k_roll, self.k_pitch, self.k_yaw)

    # ----------------------------------------------------------------- hydro
    def _grids(self, omegas=None, directions=None):
        # Frequencies scale with sqrt(g/L): the 0.1-14.5 rad/s grid was
        # chosen for a 10 m hull and would put a 300 m ship's whole response
        # in the first two points and cut a 2 m boat's off.
        omegas = (np.linspace(0.1, 14.5, 60) * np.sqrt(USV_L / self.L)
                  if omegas is None else np.asarray(omegas, float))
        # Headings: the vessel turns, so the database needs all of them. This
        # defaulted to head seas alone, and every wave component then got
        # head-sea forces at every heading -- no lateral excitation at all.
        if directions is None:
            directions = (np.linspace(0.0, np.pi, 13) if self.symmetric
                          else np.linspace(0.0, 2 * np.pi, 24, endpoint=False))
        return omegas, np.asarray(directions, float)

    def db_path(self, out_dir=".", omegas=None, directions=None):
        """Cache filename that changes when anything affecting the BEM does
        -- the frequency and heading grids included, which it used to leave
        out, so a cached file answered for grids it was never computed on."""
        omegas, directions = self._grids(omegas, directions)
        key = (self.name, self.L, self.B, self.T, self.z_cog, self.x_cog,
               self.mesh_kind, self.mesh_path, tuple(self.mesh_res),
               tuple(sorted((self.cad or {}).items())),
               self.displacement if self.mesh_kind == "cad" else None,
               self.rho, tuple(np.round(omegas, 9)),
               tuple(np.round(directions, 9)),
               (hashlib.sha1(np.asarray(self.extras["mesh"].vertices).tobytes())
                .hexdigest() if self.mesh_kind == "object" else None))
        h = hashlib.sha1(repr(key).encode()).hexdigest()[:8]
        return os.path.join(out_dir, f"hydro_{self.name}_{h}.npz")

    def database(self, omegas=None, directions=None, out_dir=".",
                 rebuild=False, verbose=True):
        """BEM database for this hull, built once and cached on disk.

        The file holds what this returns -- corrected inertia, L, B and T,
        symmetry imposed. It used to be written before those corrections, so
        loading it by path (as sim/env.py does) gave L = 1 m, Capytaine's
        inertia, and B, T falling back to the 10 m USV's 2.5 and 0.8."""
        from hydro import bem
        omegas, directions = self._grids(omegas, directions)
        path = self.db_path(out_dir, omegas, directions)
        if os.path.exists(path) and not rebuild:
            db = bem.load(path)
            if verbose:
                print(f"  {self.name}: cached hydrodynamics {path}")
            if all(k in db.attrs for k in ("L", "B", "T")):
                return db
        else:
            if verbose:
                print(f"  {self.name}: running the BEM (this is the slow part)")
            ds = bem.compute_database(self.body(), omegas,
                                      directions=directions, rho=self.rho)
            db = bem.to_db(ds)
        # the inertia Capytaine puts in the dataset is not trusted -- see
        # DEFECTS A30 -- so it is replaced with the value this class computes
        db.M = self.inertia()
        db.attrs.update(L=str(self.L), B=str(self.B), T=str(self.T),
                        name=self.name, rho=str(self.rho))
        if self.symmetric:
            from hydro.symmetry import enforce_all
            enforce_all(db)
        bem.save(db, path)
        return db

    # ------------------------------------------------------------- integrity
    def check(self, verbose=True):
        """The invariants an imported hull has to satisfy. Raises if not.
        Also prints what is still a placeholder."""
        self._fill_x()
        f = self.form()
        V = f["V"]
        m_geo = self.rho * V
        m = m_geo if self.displacement is None else self.displacement
        rows = []
        rows.append(("displaced volume", f"{V:.5g} m3", True))
        rows.append(("mass", f"{m:.5g} kg", True))
        rel = abs(m - m_geo) / m_geo
        rows.append(("stated displacement matches the lines",
                     f"{rel:.2%} apart", rel < 0.05))
        rows.append(("form", f"C_B {f['C_B']:.3f}, C_M {f['C_M']:.3f}, "
                     f"wetted {f['wetted']:.4g} m2", True))
        # a CG away from the centre of buoyancy trims the vessel: fine if the
        # vessel really trims, a sign of a wrong x_cog if it does not
        rows.append(("LCB - LCG", f"{f['x_B']:+.4g} m ({f['x_B']/self.L:+.2%} L)"
                     + (" -- the vessel will trim" if abs(f["x_B"]) >
                        0.005 * self.L else ""), True))
        M = self.inertia()
        for i, nm, ref, rl in ((3, "k_roll", self.B, "B"),
                               (4, "k_pitch", self.L, "L"),
                               (5, "k_yaw", self.L, "L")):
            k = np.sqrt(M[i, i] / M[0, 0]) / ref
            given = {3: self.k_roll, 4: self.k_pitch, 5: self.k_yaw}[i]
            rows.append((f"{nm} = {k:.3f} {rl}",
                         "given" if given is not None else
                         "FROM GEOMETRY -- bare hull, no machinery", True))
        # The same criterion the Rudder asserts (sim/actuators.py): each
        # blade's top at least h_full deep, h_full = (0.10 / 0.55) span.
        h_full = (0.10 / 0.55) * self.rudder_span
        for x, y, z in (self.rudders or ((self.x_rud, 0.0, self.z_rud),)):
            top = -(z + 0.5 * self.rudder_span)
            rows.append((f"rudder at x {x:+.3g}: blade top depth",
                         f"{top:.3g} m (needs {h_full:.3g})", top >= h_full))
        for x, y, z in (self.thrusters or ((self.x_prop, 0.0, self.z_prop),)):
            tip = -(z + 0.5 * self.d_prop)
            rows.append((f"propeller at x {x:+.3g}: tip depth",
                         f"{tip:.3g} m", tip > 0.0))
        u = self.scales()["u_design"]
        if verbose:
            print(f"\n  {self.name}: L {self.L:.4g} B {self.B:.4g} "
                  f"T {self.T:.4g}, L/B {self.L/self.B:.1f}, "
                  f"Fn {u/np.sqrt(G*self.L):.2f} at {u:.3g} m/s\n")
            for what, val, ok in rows:
                print(f"    [{'ok' if ok else 'FAIL'}]  {what:<42}{val}")
            ph = self.placeholders()
            if ph:
                print(f"\n    still placeholders ({len(ph)}), not data about "
                      f"this vessel:")
                for k, what in ph:
                    print(f"      {k:<16} {what}")
        bad = [w for w, _, ok in rows if not ok]
        if bad:
            raise ValueError(f"{self.name} fails hull integrity: {bad}")
        return rows


# ------------------------------------------------------------- known vessels
def wigley_10m(**kw):
    """The test hull this project was built on. Unchanged defaults; the
    manoeuvring derivatives are the ones identified for it
    (studies/manoeuvring_identify.py) -- still placeholders in the sense of
    DEFECTS C, but chosen for this hull, so they are stated here."""
    # roll_damping stated: the USV's own placeholder (studies/damping_check),
    # which is what its plant has always run; Ikeda's range excludes this
    # hull on four counts
    d = dict(name="wigley10", L=10.0, B=2.5, T=0.8, mesh_kind="wigley",
             x_cog=0.0, u_design=4.5, roll_damping="placeholder",
             x_rud=-4.9, z_rud=-0.45, rudder_area=0.18, rudder_span=0.55,
             x_prop=-4.6, z_prop=-0.55, d_prop=0.44, freeboard=0.55,
             Yv_prime=0.025, Nr_prime=0.0060)
    d.update(kw)
    return Hull(**d)


def realistic_mass(hull):
    """The same hull with a real vessel's mass distribution rather than the
    inertia of a hull-shaped block of water. Roll inertia goes up by 2.4x."""
    from hydro.inertia import K_ROLL_REAL, K_PITCH_REAL, K_YAW_REAL
    import copy
    h = copy.deepcopy(hull)
    h.name = hull.name + "_real_mass"
    h.k_roll = 0.5 * sum(K_ROLL_REAL)
    h.k_pitch = 0.5 * sum(K_PITCH_REAL)
    h.k_yaw = 0.5 * sum(K_YAW_REAL)
    h._given |= {"k_roll", "k_pitch", "k_yaw"}
    return h


def main():
    import warnings
    warnings.filterwarnings("ignore")
    h = wigley_10m()
    h.check()
    r = realistic_mass(h)
    r.check()
    print("\n  Your own vessel: fill in the template in hydro/hulls.py "
          "(our_boat), then\n    python -m hydro.hulls our_boat\n  "
          "Units and axes of a CAD file are stated, see hydro/cad_import.py.")


if __name__ == "__main__":
    main()
