#!/usr/bin/env python3
"""
The vessels this project knows, by name -- and the template for yours.

Every script that builds a plant takes `--hull NAME` and gets its Hull from
here, so moving the whole chain -- plant, reduced model, MPC, RL environment --
to another vessel is one argument rather than an edit in a dozen files.

    python -m hydro.hulls              list them and what each still lacks
    python -m hydro.hulls NAME         build NAME, check it, print its report

Each entry states where its numbers come from. What an entry leaves out runs
on a placeholder, and Hull.check() lists those.
"""
import os
import sys

import numpy as np

from hydro.hull import Hull, wigley_10m

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KVLCC2_IGS = os.path.join(HERE, "data", "external", "kvlcc2_geometry",
                          "kvlcc2.igs")
KCS_IGS = os.path.join(HERE, "data", "external", "kcs_geometry",
                       "KCS_hull_Case2-11", "FinalHull_KCS.igs")
KNOT = 0.514444


def kvlcc2_68():
    """KVLCC2 at 1:68, the SSPA roll-decay loading (Mendeley, CC BY 4.0) with
    the SIMMAN geometry and propeller. Fresh water. No bilge keels (SIMMAN
    lists none for any KVLCC2 model). The IGES has x aft and z down."""
    s = 1.0 / 68.0
    return Hull(
        name="kvlcc2_68", L=320.0 * s, B=58.0 * s, T=20.8 * s,
        mesh_kind="cad", mesh_path=KVLCC2_IGS,
        cad=dict(scale=s, forward="-x", up="-z", size_max=5.0,
                 symmetric=True),
        displacement=1000.0 * 312653.0 * s ** 3, rho=1000.0,
        z_cog=(18.6 - 20.8) * s,
        x_cog=(160.0 + 11.2672) * s,     # SSPA LCG, 11.27 m forward of midship
        k_roll=0.40, k_yaw=0.25,         # SSPA kxx 23.2 m, kzz 80 m
        freeboard=(30.0 - 20.8) * s,     # depth 30 m (SIMMAN)
        u_design=15.5 * KNOT * np.sqrt(s),
        d_prop=9.86 * s,                 # SIMMAN: D 9.86 m, P/D 0.721, AE/A0 0.431
        prop_series=dict(Z=4, AE_A0=0.431, P_D=0.721),
        rudder_area=136.7 * s ** 2)      # SIMMAN lateral area 136.7 m^2


def _kcs(case):
    """KCS T2015 loading conditions, from the workshop's instruction pages.
    Radii of gyration are not given there and are left as placeholders."""
    c = {"2.10": dict(L=6.0702, B=0.8498, T=0.2850, V=0.9571, KG=0.378,
                      U=2.017, rho=998.63),
         "2.11": dict(L=2.7, B=0.378, T=0.1268, V=0.084, KG=0.168, U=1.34,
                      rho=997.8858)}[case]
    s = c["L"] / 230.0
    return Hull(
        name=f"kcs_{case.replace('.', '_')}", L=c["L"], B=c["B"], T=c["T"],
        mesh_kind="cad", mesh_path=KCS_IGS,
        # full hull in mm; the rudder, boss caps and bulwark are separate
        # connected pieces and are dropped (hydro/cad_import.py)
        cad=dict(scale=1e-3 * s, half=False, size_max=4000.0,
                 min_component=0.1),
        displacement=c["rho"] * c["V"], rho=c["rho"],
        z_cog=c["KG"] - c["T"], u_design=c["U"],
        freeboard=(19.0 - 10.8) * s)     # depth 19.0 m (T2015)


def kcs_2_10():
    return _kcs("2.10")


def kcs_2_11():
    return _kcs("2.11")


def catamaran_2m():
    """A 2 m catamaran of two Wigley demihulls 0.6 m apart -- a test vessel
    for the code paths a multihull takes, not a real design."""
    from hydro.geometry import wigley_mesh
    demi = wigley_mesh(2.0, 0.2, 0.1, 30, 8)
    mesh = demi.translated_y(0.3).join_meshes(demi.translated_y(-0.3))
    return Hull(name="cat2", L=2.0, B=0.8, T=0.1, mesh_kind="object",
                extras={"mesh": mesh}, multihull=True,
                thrusters=((-0.92, 0.3, -0.07), (-0.92, -0.3, -0.07)),
                rudders=((-0.98, 0.3, -0.056), (-0.98, -0.3, -0.056)),
                d_prop=0.05, rudder_area=0.0025, rudder_span=0.069)


def box_barge_20m():
    """A 20 x 8 x 2 m box barge: exact hydrostatics, no bilge, no stability
    of course -- a test vessel."""
    import capytaine as cpt
    mesh = cpt.mesh_parallelepiped(size=(20.0, 8.0, 2.0), center=(0, 0, -1.0),
                                   resolution=(24, 10, 4),
                                   missing_sides={"top"})
    return Hull(name="barge20", L=20.0, B=8.0, T=2.0, mesh_kind="object",
                extras={"mesh": mesh})


def our_boat():
    """TEMPLATE -- your vessel. Copy the call below out of this docstring,
    fill in every value, and return it; until then this raises.

    Anything you leave as None runs on a placeholder and Hull.check() lists
    it, so a partial description still simulates -- it just says what it is
    standing in for.

        return Hull(
            name="our_boat",
            L=..., B=..., T=...,              # waterline length, beam, draught, m
            # geometry: CAD surfaces (IGES/STEP) -- or mesh_kind="file" for
            # a panel mesh Capytaine reads (STL, GDF, ...)
            mesh_kind="cad", mesh_path="data/our_boat/hull.igs",
            cad=dict(scale=1e-3,              # file units -> m (mm: 1e-3)
                     forward="+x", up="+z",   # which file axes point bow / up
                     half=True,               # does the file model one side?
                     size_max=...),           # panel size in FILE units, ~L/60
            displacement=...,                 # kg, the loading you simulate
            z_cog=..., x_cog=...,             # CG: inclining test / weight list
            k_roll=..., k_pitch=..., k_yaw=...,   # radii of gyration, x B, x L, x L
            u_design=...,                     # cruise speed, m/s
            thrusters=((x, y, z), ...),       # every propeller / waterjet
            d_prop=..., t_max=...,            # diameter, total thrust, N
            prop_series=dict(Z=..., AE_A0=..., P_D=...),   # if a B-series-like screw
            rudders=((x, y, z), ...), rudder_area=..., rudder_span=...,
            rudder_max_deg=..., rudder_rate_deg=...,
            # a free-roll decay test, fitted as in
            # studies/exp_kvlcc2_rolldecay.py: the most valuable single test
            roll_damping=dict(zeta1=..., k2=..., T=...),
            visc=(...),                       # six quadratic coefficients
            Yv_prime=..., Nr_prime=...,       # turning trial / PMM
            wind_areas=dict(A_F=..., A_L=..., s_H=..., s_L=...,
                            vessel="speed boat"),
        )
    """
    raise NotImplementedError(
        "hydro/hulls.py: our_boat() is a template -- fill in your vessel's "
        "data (see its docstring) and return the Hull.")


REGISTRY = {
    "wigley10": wigley_10m,
    "kvlcc2_68": kvlcc2_68,
    "kcs_2_10": kcs_2_10,
    "kcs_2_11": kcs_2_11,
    "cat2": catamaran_2m,
    "barge20": box_barge_20m,
    "our_boat": our_boat,
}


def get(name):
    """The Hull registered as `name`."""
    if name not in REGISTRY:
        raise KeyError(f"unknown hull {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]()


def main():
    import warnings
    warnings.filterwarnings("ignore")
    if len(sys.argv) > 1:
        h = get(sys.argv[1])
        h.check()
        return h
    print("\n  registered hulls, and how many inputs each still lacks:\n")
    for name, make in REGISTRY.items():
        try:
            h = make()
            print(f"    {name:<12} L {h.L:<8.4g} {h.mesh_kind:<7} "
                  f"{len(h.placeholders()):>2} placeholders")
        except NotImplementedError:
            print(f"    {name:<12} (template -- fill it in)")
    print("\n  python -m hydro.hulls NAME   builds it and runs Hull.check()")


if __name__ == "__main__":
    main()
