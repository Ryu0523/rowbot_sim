#!/usr/bin/env python3
"""
S1 stage 2 (S1b) shared helpers: the opt-in grid variants of unsteady.build_grid for the scarab195 5 kn study and the
stern knuckle abscissa (where the calm waterline reaches the chine half-breadth and its slope jumps from ~7 to 0).

Variants (name -> extra build_grid keywords on top of derive_v1b_build.GRIDS):
  s1   : stage-1 grid (nothing)
  ew   : end_width (stem node at the hull's 2 mm half-breadth)
  kn   : x_nodes = [stern knuckle] only (the Neumann-Kelvin Jacobian fix alone)
  es   : end_width + end_sub (2 sub-columns at the stern, 4 at the stem, quadratic clustering) + stern knuckle node
  esg  : es + band "geom" (continuous row height across the band/lattice junction)
  g    : band "geom" only
"""
import numpy as np


def stern_knuckle_x(hull, sink, trim, span=0.6):
    """Earth x where the waterline half-breadth first reaches the chine value (bisection from the stern end)."""
    xs, xb = hull.waterline_ends(sink, trim)
    yc = float(hull.waterline(np.array([0.5 * (xs + xb)]), sink, trim)[0])      # mid-hull: on the chine? use the max
    xx = np.linspace(xs, xs + span, 2001)
    yy = hull.waterline(xx, sink, trim)
    ytop = yy.max()
    k = int(np.argmax(yy >= ytop - 1e-9))
    lo, hi = xx[max(k - 1, 0)], xx[k]
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if hull.waterline(np.array([mid]), sink, trim)[0] < ytop - 1e-9:
            lo = mid
        else:
            hi = mid
    return float(hi), float(ytop)


def grid_opts(name, hull=None, sink=0.0, trim=0.0):
    name = name or "s1"
    if name == "s1":
        return {}
    if name == "ew":
        return dict(end_width=True)
    if name == "g":
        return dict(band="geom")
    xk = stern_knuckle_x(hull, sink, trim)[0]
    if name == "kn":
        return dict(x_nodes=[xk])
    if name == "es":
        return dict(end_width=True, end_sub=(2, 4), x_nodes=[xk])
    if name == "esg":
        return dict(end_width=True, end_sub=(2, 4), x_nodes=[xk], band="geom")
    if name == "es6g":
        return dict(end_width=True, end_sub=(2, 6), x_nodes=[xk], band="geom")
    raise ValueError(name)


VARIANTS = ("s1", "ew", "kn", "es", "esg", "g")
