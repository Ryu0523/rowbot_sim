"""
Flat quadrilateral panels, symmetry images and mesh generators.

Geometry (Hess & Smith 1964, section 4): a curved surface patch given by four
corner points is replaced by the flat panel through their mean point whose
normal is the cross product of the diagonals; the corners are projected onto
that plane. Triangles are quads with one repeated corner (a zero-length edge
contributes nothing to any edge sum in influence.py). The collocation point is
the area centroid of the flat panel. Normals point INTO the fluid.

Symmetry images: a panel reflected in the plane x_k = 0 has its corners
reflected and their order reversed, so that its normal is the reflected normal
(reflection reverses handedness; reversing the order restores it). A solution
of parity s (+1 even, -1 odd) under that reflection carries the same density
times s on the image panel, which makes port/starboard (and fore/aft, up/down
for the rigid-lid / pressure-release limits) symmetry exact by construction.
"""
import numpy as np


class Panels:
    """Flat panels. verts (N, 4, 3), corners ordered counter-clockwise when
    seen from the side the normal points to (the fluid side)."""

    def __init__(self, verts, tag=None):
        v = np.asarray(verts, float)
        assert v.ndim == 3 and v.shape[1:] == (4, 3)
        d1 = v[:, 2] - v[:, 0]
        d2 = v[:, 3] - v[:, 1]
        nv = np.cross(d1, d2)
        nn = np.linalg.norm(nv, axis=1)
        if np.any(nn <= 0):
            raise ValueError("degenerate panel")
        n = nv / nn[:, None]
        m = v.mean(axis=1)
        h = np.einsum("ijk,ik->ij", v - m[:, None, :], n)
        vp = v - h[:, :, None] * n[:, None, :]          # projected, planar
        # area and area centroid of the flat polygon (fan from corner 0)
        a1 = 0.5 * np.einsum("ij,ij->i", np.cross(vp[:, 1] - vp[:, 0],
                                                  vp[:, 2] - vp[:, 0]), n)
        a2 = 0.5 * np.einsum("ij,ij->i", np.cross(vp[:, 2] - vp[:, 0],
                                                  vp[:, 3] - vp[:, 0]), n)
        area = a1 + a2
        c = ((vp[:, 0] + vp[:, 1] + vp[:, 2]) * a1[:, None]
             + (vp[:, 0] + vp[:, 2] + vp[:, 3]) * a2[:, None]) / (3 * area[:, None])
        self.verts = vp
        self.raw = v
        self.n = n
        self.area = area
        self.c = c
        self.warp = np.abs(h).max(axis=1)
        self.tag = (np.zeros(len(v), int) if tag is None
                    else np.asarray(tag, int).copy())
        e = np.roll(vp, -1, axis=1) - vp                # edge vectors
        self.diam = np.linalg.norm(e, axis=2).max(axis=1)
        # max chord, used for far-field switching decisions
        self.size = np.maximum(np.linalg.norm(d1, axis=1),
                               np.linalg.norm(d2, axis=1))

    def __len__(self):
        return len(self.verts)

    def reflect(self, axis):
        """Image panels in the plane x_axis = 0 (normal reflected)."""
        v = self.raw.copy()
        v[:, :, axis] *= -1.0
        return Panels(v[:, ::-1, :], self.tag)

    def scaled(self, s):
        return Panels(self.raw * s, self.tag)

    def __add__(self, other):
        return Panels(np.concatenate([self.raw, other.raw]),
                      np.concatenate([self.tag, other.tag]))

    def subset(self, mask):
        return Panels(self.raw[mask], self.tag[mask])


def from_grid(P, flip=False, tag=0):
    """(nu+1, nv+1, 3) grid of points -> panels (u, v, normal right-handed)."""
    P = np.asarray(P, float)
    q = np.stack([P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]], axis=2)
    q = q.reshape(-1, 4, 3)
    if flip:
        q = q[:, ::-1]
    # drop fully collapsed quads (two repeated corners)
    keep = []
    for k in range(len(q)):
        u = np.unique(np.round(q[k] / (np.abs(q[k]).max() + 1e-300), 13), axis=0)
        keep.append(len(u) >= 3)
    q = q[np.array(keep)]
    return Panels(q, np.full(len(q), tag))


# --------------------------------------------------------------- generators
def ellipsoid(a, b, c, nt, nphi, t_range=(0.0, np.pi), phi_range=(0.0, 2 * np.pi),
              spacing="uniform"):
    """Ellipsoid x = a cos t, y = b sin t cos p, z = c sin t sin p, outward
    normals (into an exterior fluid). t from the +x pole to the -x pole.
    `spacing="cosine"` clusters t near both ends of t_range."""
    s = np.linspace(0.0, 1.0, nt + 1)
    if spacing == "cosine":
        s = 0.5 * (1 - np.cos(np.pi * s))
    t = t_range[0] + (t_range[1] - t_range[0]) * s
    p = np.linspace(phi_range[0], phi_range[1], nphi + 1)
    T, Pp = np.meshgrid(t, p, indexing="ij")
    X = np.stack([a * np.cos(T), b * np.sin(T) * np.cos(Pp),
                  c * np.sin(T) * np.sin(Pp)], axis=-1)
    pan = from_grid(X)
    # orientation check: normal must point away from the centre
    if np.mean(np.einsum("ij,ij->i", pan.c, pan.n)) < 0:
        pan = from_grid(X, flip=True)
    return pan


def sphere(R, nt, nphi, octant=False):
    """Sphere about the x axis. octant=True: x>=0, y>=0, z>=0 part only
    (use with images in all three planes)."""
    if octant:
        return ellipsoid(R, R, R, nt, nphi, (0.0, np.pi / 2), (0.0, np.pi / 2))
    return ellipsoid(R, R, R, nt, nphi)


def hemisphere_quarter(R, n_polar, n_az):
    """Lower hemisphere (z <= 0) in the quadrant x >= 0, y >= 0; polar angle
    measured from the -z axis; outward normals. Images in x and y (and z for
    the rigid-lid / pressure-release limits) complete it."""
    th = np.linspace(0.0, np.pi / 2, n_polar + 1)        # 0 = bottom pole
    ph = np.linspace(0.0, np.pi / 2, n_az + 1)
    T, Pp = np.meshgrid(th, ph, indexing="ij")
    X = np.stack([R * np.sin(T) * np.cos(Pp), R * np.sin(T) * np.sin(Pp),
                  -R * np.cos(T)], axis=-1)
    pan = from_grid(X)
    if np.mean(np.einsum("ij,ij->i", pan.c, pan.n)) < 0:
        pan = from_grid(X, flip=True)
    return pan


def graded_radii(r0, h0, h_max, r_end, growth=1.1):
    """Radii from r0: spacing starts at h0, grows geometrically by `growth`
    up to h_max, then stays at h_max until r_end (last spacing adjusted)."""
    r = [r0]
    h = h0
    while r[-1] < r_end - 1e-12:
        r.append(min(r[-1] + h, r_end))
        h = min(h * growth, h_max)
    r = np.array(r)
    if len(r) > 2 and (r[-1] - r[-2]) < 0.3 * (r[-2] - r[-3]):
        r = np.delete(r, -2)
    return r


def fs_polar_quarter(radii, n_az, tag=1):
    """Free-surface annulus z = 0 in the quadrant x, y >= 0 between radii[0]
    and radii[-1]; normals -z (into the fluid below)."""
    ph = np.linspace(0.0, np.pi / 2, n_az + 1)
    Rr, Pp = np.meshgrid(radii, ph, indexing="ij")
    X = np.stack([Rr * np.cos(Pp), Rr * np.sin(Pp), np.zeros_like(Rr)], axis=-1)
    pan = from_grid(X, tag=tag)
    if np.mean(pan.n[:, 2]) > 0:
        pan = from_grid(X, flip=True, tag=tag)
    return pan


def box_quarter(L, B, T, nx, ny, nz):
    """Wetted surface of a rectangular barge (length L along x, beam B, draught
    T), quadrant x, y >= 0: half bottom, half side, half end; outward normals."""
    xs = np.linspace(0, L / 2, nx + 1)
    ys = np.linspace(0, B / 2, ny + 1)
    zs = np.linspace(-T, 0, nz + 1)
    parts = []
    X, Y = np.meshgrid(xs, ys, indexing="ij")            # bottom z = -T
    parts.append(np.stack([X, Y, -T + 0 * X], -1))
    X, Z = np.meshgrid(xs, zs, indexing="ij")            # side y = B/2
    parts.append(np.stack([X, B / 2 + 0 * X, Z], -1))
    Y, Z = np.meshgrid(ys, zs, indexing="ij")            # end x = L/2
    parts.append(np.stack([L / 2 + 0 * Y, Y, Z], -1))
    outward = [np.array([0, 0, -1.0]), np.array([0, 1.0, 0]), np.array([1.0, 0, 0])]
    pans = []
    for G, o in zip(parts, outward):
        p = from_grid(G)
        if np.mean(p.n @ o) < 0:
            p = from_grid(G, flip=True)
        pans.append(p)
    return pans[0] + pans[1] + pans[2]


def fs_tensor_quarter(xs, ys, hole, tag=1):
    """Free surface z = 0 on the tensor grid xs x ys (both starting at 0)
    minus the rectangle |x| < hole[0], |y| < hole[1]; normals -z."""
    q = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            xm, ym = 0.5 * (xs[i] + xs[i + 1]), 0.5 * (ys[j] + ys[j + 1])
            if xm < hole[0] and ym < hole[1]:
                continue
            q.append([[xs[i], ys[j], 0], [xs[i], ys[j + 1], 0],
                      [xs[i + 1], ys[j + 1], 0], [xs[i + 1], ys[j], 0]])
    pan = Panels(np.array(q, float), np.full(len(q), tag))
    assert np.all(pan.n[:, 2] < 0)
    return pan


def fs_offset_quarter(wl, offsets, tag=1):
    """Free surface z = 0 outside a convex waterline, quadrant x, y >= 0.
    wl (ns+1, 2): waterline points from the x axis (y = 0) to the y axis
    (x = 0); offsets (nr+1,) from 0. Point (s, k) = wl[s] + offsets[k] u(s),
    with u(s) at angle (pi/2) * arclength(s)/total, so the grid is fitted to
    the waterline near the body and polar-like far away. Normals -z."""
    wl = np.asarray(wl, float)
    ds = np.linalg.norm(np.diff(wl, axis=0), axis=1)
    sarc = np.concatenate([[0.0], np.cumsum(ds)]) / ds.sum()
    th = 0.5 * np.pi * sarc
    u = np.stack([np.cos(th), np.sin(th)], 1)
    Pp = wl[None, :, :] + np.asarray(offsets)[:, None, None] * u[None, :, :]
    X = np.concatenate([Pp, np.zeros(Pp.shape[:2] + (1,))], axis=2)
    pan = from_grid(X, tag=tag)
    if np.mean(pan.n[:, 2]) > 0:
        pan = from_grid(X, flip=True, tag=tag)
    return pan


def mode_normals(pan, origin=(0.0, 0.0, 0.0)):
    """Generalised normals n_j, j = 1..6 (surge, sway, heave, roll, pitch,
    yaw) about `origin`, with n out of the body: (N, 6)."""
    r = pan.c - np.asarray(origin, float)
    return np.concatenate([pan.n, np.cross(r, pan.n)], axis=1)


# parity of the rigid-body modes under reflection in x=0, y=0, z=0
# (needed to choose the image signs). Derived from n_j at the mirror point:
# e.g. pitch n5 = z n_x - x n_z is odd under x -> -x and under z -> -z.
MODE_PARITY = {
    1: (-1, +1, +1), 2: (+1, -1, +1), 3: (+1, +1, -1),
    4: (+1, -1, -1), 5: (-1, +1, -1), 6: (-1, -1, +1),
}
