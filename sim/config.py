#!/usr/bin/env python3
"""
One place that turns a vessel NAME into what every script needs.

    hull, db = config.load("kvlcc2_68")
    plant, db = config.calm_plant("kvlcc2_68")      # for identification
    plant = config.plant_for(db, sea, hull)         # in a sea state

Scripts take `--hull NAME` (hydro/hulls.py lists the names) and go through
here, so the plant, the reduced model, the MPC and the RL environment all see
the same vessel. Before this, each script loaded `hydro_wigley_10m.npz` by
path and built its own plant, and a second vessel would have had to be edited
into a dozen files -- with every one missed being a silent mix of two boats.

The Wigley USV keeps its original database file (150 frequencies, 13 wave
headings), so every earlier result reproduces bit for bit; any other hull gets
its database from Hull.database(), cached by content next to it.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY_DB = os.path.join(HERE, "hydro_wigley_10m.npz")
USV_L = 10.0


def hull_of(hull):
    """A Hull from a registered name, or the Hull itself."""
    if isinstance(hull, str):
        from hydro import hulls
        return hulls.get(hull)
    return hull


def load(hull="wigley10", verbose=False):
    """(hull, database) for a registered name or a Hull."""
    from hydro import bem
    h = hull_of(hull)
    if h.name == "wigley10" and os.path.exists(LEGACY_DB):
        db = bem.load(LEGACY_DB)
    else:
        db = h.database(out_dir=HERE, verbose=verbose)
    return h, db


def hull_from_db(db):
    """The registered Hull a database was computed for; None for the legacy
    Wigley file, which predates the registry (its hull-less plant is the same
    as hulls.get("wigley10"), checked to the last bit)."""
    name = db.attrs.get("name")
    if name is None:
        if db.attrs.get("hull") == "wigley":
            return None
        raise ValueError("this database does not say which vessel it was "
                         "computed for; pass hull=")
    try:
        return hull_of(name)
    except KeyError:
        raise ValueError(f"the database is for {name!r}, which is not in "
                         f"hydro/hulls.py; pass the Hull itself as hull=") \
            from None


def resolve(db, hull=None):
    """`hull` (a Hull or a name), else the vessel `db` was computed for.
    Resolve ONCE and pass the result on: a Hull caches its mesh and station
    table, and an RL environment builds a plant every episode."""
    return hull_of(hull) if hull is not None else hull_from_db(db)


def scales_for(db, hull=None):
    """Hull.scales() for the vessel of `db`: u_design, dt, dt_ctrl, lam."""
    h = resolve(db, hull)
    if h is not None:
        return h.scales()
    lam = db.L / USV_L
    return dict(lam=lam, u_design=4.5 * np.sqrt(lam), dt=0.05 * np.sqrt(lam),
                dt_ctrl=0.5 * np.sqrt(lam))


def plant_for(db, sea, hull=None, dt=None, **kw):
    """The plant for `db` in `sea`, through its Hull where it has one."""
    h = resolve(db, hull)
    if h is None:
        from sim.vessel import NonlinearVessel
        return NonlinearVessel(db, sea, L=db.L,
                               B=float(db.attrs.get("B", 2.5)),
                               T=float(db.attrs.get("T", 0.8)),
                               dt=scales_for(db)["dt"] if dt is None else dt,
                               **kw)
    return h.plant(sea, db=db, dt=dt, **kw)


def head_index(db):
    """Index of head seas (beta = pi) among the database's wave directions,
    found by angle. The gates used index 12, which is pi only on the
    13-heading grid from 0 to pi."""
    d = np.asarray(db.directions, float)
    return int(np.argmin(np.abs(np.angle(np.exp(1j * (d - np.pi))))))


def calm_plant(hull="wigley10", db=None, **kw):
    """(plant in calm water and still air, database) -- the plant the
    reduced model is identified on."""
    from sim.forces import Wind
    from sim.test_vessel import Monochromatic
    if db is None:
        hull, db = load(hull)
    return plant_for(db, Monochromatic(1.0, 0.0), hull, wind=Wind(), **kw), db
