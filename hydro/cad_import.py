#!/usr/bin/env python3
"""
Hull geometry from CAD files (IGES, STEP) to a BEM-ready panel mesh.

This is the import path for a real vessel. Real hulls arrive as CAD surfaces,
not as the analytic Wigley formula, and the things that usually go wrong on
import are exactly the things this module refuses to assume:

  UNITS      are stated by the caller (scale factor), never inferred.
  AXES       are stated by the caller too: which file axis points forward and
             which points up. KVLCC2's IGES has x pointing AFT (origin at the
             AP) and z pointing DOWN (keel at z = 0, hull at negative z). Read
             as z-up, the flat bottom lands on top -- the first import did
             that and "found" a waterline 7.5% shallower than the published
             draft. Read with x forward, everything matches except the sign of
             the LCB: 3.50% AFT against a published 3.48% forward. Volume,
             draft and GM cannot tell bow from stern; the LCB can, which is
             why it is among the published numbers main() compares.
  WATERLINE  is found from the DISPLACEMENT, by bisection on the vertical
             offset until the immersed volume equals the stated volume. A
             waterline taken as "z = 0 in the file" is a guess; a waterline
             that reproduces the displacement is a measurement.
  CLOSURE    is CHECKED with three divergence forms of the immersed volume,
             V = oint x n_x dS = oint y n_y dS = oint z n_z dS. For a surface
             closed below the waterline all three are exact (a linear integrand
             on a flat triangle), so they agree to rounding. Each form is blind
             to surfaces perpendicular to its own axis: the x and y forms
             cannot see a flat bottom at all, so a missing or DOUBLED bottom
             passes them. The first version checked only those two; KVLCC2's
             IGES has its flat bottom twice and it passed. The z form sees it.
  DUPLICATES are removed before meshing reaches the BEM: a CAD patch lying on
             another patch (the doubled bottom above, a thin strip over the
             keel) gives two coincident panels, and a panel method's influence
             matrix is then singular.
  NORMALS    are made consistent by walking the edges patches share (two
             patches' triangles must run opposite ways along a common edge),
             and only the global sign of each connected set is taken from the
             geometry. A per-patch geometric vote, the first version, worked on
             KVLCC2's 19 patches and failed on KCS's 653: ~50 votes too weak to
             trust, some of them wrong, immersed volume forms 2.3% apart. The
             edge walk found KCS's shell already consistent.
  PIECES     that are not the hull -- a rudder, a propeller boss cap, a
             bulwark -- are separate connected sets of patches, and sets under
             `min_component` of the largest one's area can be dropped. Only
             when the caller asks (default 0): whether a file's hull patches
             are sewn to each other is a property of the file. KCS's are (one
             653-patch shell, rudder and caps apart); KVLCC2's are not, and a
             10% rule dropped its bow and stern pieces and broke the hull.
             The connected sets are always reported.

Half hulls (most CAD files model one side) are mirrored about the centreplane.

Steps: gmsh (OpenCASCADE kernel) reads and sews the surfaces and triangulates
them at a requested element size; the triangles become a Capytaine mesh.

Run: python -m hydro.cad_import   (self-test on boxes, then KVLCC2 against
                                   its published hydrostatics)
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cad_triangles(path, size_max, size_min=None, sew=True, curvature=None):
    """Triangulate every surface in a CAD file. Returns (points, tris, patch).

    `curvature`: elements per full turn of the local curvature radius (gmsh
    MeshSizeFromCurvature), so a 2.4 m bilge is not cut by one 5 m chord.
    Sizes are in the FILE's units."""
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        if sew:
            gmsh.option.setNumber("Geometry.OCCSewFaces", 1)
        gmsh.model.occ.importShapes(path)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", size_max)
        gmsh.option.setNumber("Mesh.MeshSizeMin",
                              size_min if size_min else size_max / 4.0)
        if curvature:
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature)
        gmsh.model.mesh.generate(2)
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        P = xyz.reshape(-1, 3)
        pos = {int(t): i for i, t in enumerate(tags)}
        tris, patch = [], []
        for dim, tag in gmsh.model.getEntities(2):
            etypes, _, enodes = gmsh.model.mesh.getElements(dim, tag)
            for et, en in zip(etypes, enodes):
                if int(et) == 2:                       # 3-node triangles
                    t = np.array([pos[int(k)] for k in en]).reshape(-1, 3)
                    tris.append(t)
                    patch.append(np.full(len(t), tag))
    finally:
        gmsh.finalize()
    return P, np.vstack(tris), np.concatenate(patch)


def _tri_geometry(P, T):
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    n = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(n, axis=1)
    return (a + b + c) / 3.0, n / np.maximum(2 * area[:, None], 1e-30), area


def _on_triangle(p, a, b, c, tol):
    """True where point p lies within `tol` of triangle abc (same row)."""
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n, axis=1)
    n = n / np.maximum(nn, 1e-300)[:, None]
    d = np.einsum("ij,ij->i", p - a, n)
    q = p - d[:, None] * n
    v0, v1, v2 = b - a, c - a, q - a
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
    den = np.where(d00 * d11 - d01 * d01 == 0.0, 1.0, d00 * d11 - d01 * d01)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    e = -1e-6
    return (nn > 0) & (np.abs(d) < tol) & (1 - v - w >= e) & (v >= e) & (w >= e)


def redundant_patches(P, T, patch, tol=None, share=0.9):
    """CAD patches lying on top of other patches, largest patch kept first.

    A patch is redundant when at least `share` of its area lies within `tol`
    of patches already kept. Zero-area patches are redundant too. Returns a
    list of (patch id, area) to drop.
    """
    from scipy.spatial import cKDTree
    ctr, _, area = _tri_geometry(P, T)
    L = float(np.ptp(P[:, 0]))
    tol = 1e-4 * L if tol is None else tol
    ids = sorted(np.unique(patch), key=lambda p: -area[patch == p].sum())
    kept, drop = [], []
    for p in ids:
        sel = patch == p
        A_p = float(area[sel].sum())
        if A_p <= 1e-9 * L * L:
            drop.append((int(p), A_p))
            continue
        if kept:
            ks = np.isin(patch, kept)
            kT = T[ks]
            k = min(8, int(ks.sum()))
            _, nb = cKDTree(ctr[ks]).query(ctr[sel], k=k)
            nb = np.asarray(nb).reshape(int(sel.sum()), -1)
            covered = np.zeros(int(sel.sum()), bool)
            for j in range(nb.shape[1]):
                t = kT[nb[:, j]]
                covered |= _on_triangle(ctr[sel], P[t[:, 0]], P[t[:, 1]],
                                        P[t[:, 2]], tol)
            if area[sel][covered].sum() >= share * A_p:
                drop.append((int(p), A_p))
                continue
        kept.append(p)
    return drop


def _merge_nodes(P, tol):
    """Representative index per point, merging points closer than `tol`."""
    from scipy.spatial import cKDTree
    parent = np.arange(len(P))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in cKDTree(P).query_pairs(tol, output_type="ndarray"):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(len(P))])


def patch_topology(P, T, patch, tol=None):
    """How the CAD patches connect.

    Returns (ids, comp, rel): patch ids, a connected-component label per id,
    and {(p, q): signed length of their common edges} -- positive where the
    two patches' triangles run opposite ways along those edges (consistently
    oriented), negative where they run the same way. Only edges used by
    exactly two triangles count; T-junctions of three surfaces say nothing
    about orientation.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    size = float(np.ptp(P, axis=0).max())
    root = _merge_nodes(P, 1e-7 * size if tol is None else tol)
    Tm = root[T]
    E = np.vstack([Tm[:, [0, 1]], Tm[:, [1, 2]], Tm[:, [2, 0]]])
    Ep = np.concatenate([patch, patch, patch])
    _, inv, cnt = np.unique(np.sort(E, axis=1), axis=0, return_inverse=True,
                            return_counts=True)
    order = np.argsort(inv.ravel(), kind="stable")
    starts = np.r_[0, np.cumsum(cnt)[:-1]]
    two = np.where(cnt == 2)[0]
    i, j = order[starts[two]], order[starts[two] + 1]
    other = Ep[i] != Ep[j]
    i, j = i[other], j[other]
    same = E[i, 0] == E[j, 0]
    ln = np.linalg.norm(P[E[i, 0]] - P[E[i, 1]], axis=1)
    rel = {}
    for p, q, s, l in zip(Ep[i], Ep[j], same, ln):
        k = (int(min(p, q)), int(max(p, q)))
        rel[k] = rel.get(k, 0.0) + (-l if s else l)
    ids = np.unique(patch)
    pos = {int(p): n for n, p in enumerate(ids)}
    r = [pos[a] for a, _ in rel] + [pos[b] for _, b in rel]
    c = [pos[b] for _, b in rel] + [pos[a] for a, _ in rel]
    G = coo_matrix((np.ones(len(r)), (r, c)), shape=(len(ids),) * 2)
    _, comp = connected_components(G, directed=False)
    return ids, comp, rel


def orient_patches(P, T, patch):
    """Flip whole CAD patches so their normals point out of the hull.

    First make the patches agree with each other: a breadth-first walk over
    each connected set flips any patch whose triangles run the same way as a
    neighbour's along their common edge. That is exact topology, no geometry.
    Then ONE geometric question per connected set: summed over all its area,
    do the normals point away from the hull's spine (the centreline segment at
    mid-height)? A single patch -- a transom, a small bulb piece -- can be
    nearly perpendicular to that direction; a whole shell cannot.
    """
    ids, comp, rel = patch_topology(P, T, patch)
    nbr = {}
    for (a, b), s in rel.items():
        nbr.setdefault(a, []).append((b, 1 if s > 0 else -1))
        nbr.setdefault(b, []).append((a, 1 if s > 0 else -1))
    ctr, nrm, area = _tri_geometry(P, T)
    x0, x1 = P[:, 0].min(), P[:, 0].max()
    L = x1 - x0
    z_mid = 0.5 * (P[:, 2].min() + P[:, 2].max())
    # nearest point on a SEGMENT of the centreline, so the out-vector of a
    # transom or a bow patch has its x component
    xs = np.clip(ctr[:, 0], x0 + 0.15 * L, x1 - 0.15 * L)
    out = ctr - np.column_stack([xs, np.zeros(len(ctr)),
                                 np.full(len(ctr), z_mid)])
    out /= np.maximum(np.linalg.norm(out, axis=1)[:, None], 1e-30)
    vote_p = {int(p): float(np.sum((area * np.einsum("ij,ij->i", nrm, out))
                                   [patch == p])) for p in ids}
    area_p = {int(p): float(area[patch == p].sum()) for p in ids}
    flip, conflicts, weak = {}, 0, []
    for c in np.unique(comp):
        members = [int(p) for p in ids[comp == c]]
        start = max(members, key=lambda p: area_p[p])
        flip[start] = 1
        stack = [start]
        while stack:
            p = stack.pop()
            for q, s in nbr.get(p, []):
                if q not in flip:
                    flip[q] = flip[p] * s
                    stack.append(q)
                elif flip[q] != flip[p] * s:
                    conflicts += 1
        v = sum(flip[p] * vote_p[p] for p in members)
        if abs(v) < 0.2 * sum(area_p[p] for p in members):
            weak.append(int(c))
        if v < 0:
            for p in members:
                flip[p] = -flip[p]
    T = T.copy()
    flipped = 0
    for p, f in flip.items():
        if f < 0:
            sel = patch == p
            T[sel] = T[sel][:, ::-1]
            flipped += 1
    if conflicts:
        print(f"    orientation: {conflicts // 2} patch adjacencies the walk "
              f"could not satisfy (a non-orientable junction)")
    if weak:
        print(f"    orientation: connected sets with a weak outward vote: "
              f"{weak}")
    return T, flipped


def mirror_y(P, T):
    """Append the mirror image in y, with reversed winding to stay outward."""
    Q = P.copy()
    Q[:, 1] *= -1.0
    return np.vstack([P, Q]), np.vstack([T, T[:, ::-1] + len(P)])


def immersed_volume(P, T, z_w, check=True):
    """Volume below z = z_w of a closed-below-the-waterline triangle mesh.

    Clips with Capytaine (which handles partially wetted panels). With
    `check`, the n_x and n_y forms must agree to 2%. Only the volume is
    checked here: it is exact on any closed triangle mesh, whereas the
    higher moments of hydro.inertia.volume_moments carry the centroid rule's
    quadrature error and fail, correctly, on meshes too coarse for them (a
    12-triangle box). The bisection below skips the check, because near the
    keel the immersed volume is tiny; import_hull adds the n_z form.
    """
    import capytaine as cpt
    m = cpt.Mesh(vertices=P - [0.0, 0.0, z_w],
                 faces=np.column_stack([T, T[:, 2]]))
    wet = m.immersed_part()
    V = closure_forms(wet)
    if check and abs(V[0] - V[1]) > 0.02 * abs(V[0]):
        raise ValueError(f"immersed_volume: the n_x and n_y forms disagree "
                         f"({V[0]:.6g} vs {V[1]:.6g}) -- orientation or "
                         f"closure error in the mesh")
    return 0.5 * float(V[0] + V[1]), wet


def closure_forms(wet):
    """(V_x, V_y, V_z): the three divergence forms of the immersed volume.
    A missing waterplane lid is harmless to all three (z = 0 and n_x = n_y = 0
    on it); a hole or a doubled surface anywhere else is not."""
    c, n, a = wet.faces_centers, wet.faces_normals, wet.faces_areas
    return np.array([float(np.sum(c[:, k] * n[:, k] * a)) for k in range(3)])


def waterline_for_volume(P, T, volume, tol=1e-6):
    """Height z_w in the file's coordinates at which the hull displaces
    `volume`. Bisection -- displaced volume is monotone in z_w."""
    lo, hi = P[:, 2].min(), P[:, 2].max()
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        v, _ = immersed_volume(P, T, mid, check=False)
        if v < volume:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * (P[:, 2].max() - P[:, 2].min()):
            break
    return 0.5 * (lo + hi)


def waterplane(P, T, z_w):
    """Waterplane of the triangulated hull at z = z_w, exactly.

    The waterline is the polygon where the triangles cross the plane; every
    edge is straight, so Green's theorem gives the area and its moments with
    no quadrature error: int dA = oint x m_x dl, int x dA = oint x^2/2 m_x dl,
    int y^2 dA = oint y^3/3 m_y dl, with m the outward normal of the edge in
    the plane (the in-plane part of the hull normal). The area is also taken
    with the y form; the two must agree when the waterline closes.
    """
    tri = P[T]
    s = tri[:, :, 2] - z_w
    s = np.where(s == 0.0, 1e-12, s)
    hit = (s.min(1) < 0) & (s.max(1) > 0)
    tri, s = tri[hit], s[hit]
    _, nrm, _ = _tri_geometry(P, T[hit])
    q = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        cr = s[:, i] * s[:, j] < 0
        t = s[:, i] / np.where(cr, s[:, i] - s[:, j], 1.0)
        q.append(np.where(cr[:, None],
                          tri[:, i] + t[:, None] * (tri[:, j] - tri[:, i]),
                          np.nan))
    q = np.stack(q, 1)
    order = np.argsort(np.isnan(q[:, :, 0]), axis=1, kind="stable")[:, :2]
    r = np.arange(len(q))
    a, b = q[r, order[:, 0], :2], q[r, order[:, 1], :2]
    d = b - a
    ln = np.hypot(d[:, 0], d[:, 1])
    m = np.column_stack([d[:, 1], -d[:, 0]]) / np.maximum(ln, 1e-300)[:, None]
    m *= np.sign(np.einsum("ij,ij->i", m, nrm[:, :2]))[:, None]
    x0, y0, x1, y1 = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
    A = float(np.sum(ln * m[:, 0] * 0.5 * (x0 + x1)))
    A_y = float(np.sum(ln * m[:, 1] * 0.5 * (y0 + y1)))
    Sx = float(np.sum(ln * m[:, 0] * (x0 * x0 + x0 * x1 + x1 * x1) / 6.0))
    Iyy = float(np.sum(ln * m[:, 1] * (y0 + y1) * (y0 * y0 + y1 * y1) / 12.0))
    Ixx = float(np.sum(ln * m[:, 0] * (x0 + x1) * (x0 * x0 + x1 * x1) / 12.0))
    pts = np.vstack([a, b])
    x_F = Sx / A
    return dict(A=A, A_y=A_y, x_F=x_F, I_T=Iyy, I_L=Ixx - A * x_F ** 2,
                L_wl=float(np.ptp(pts[:, 0])), B_wl=float(np.ptp(pts[:, 1])))


def file_axes(forward="+x", up="+z"):
    """Rotation from file coordinates to vessel axes (x forward, y port,
    z up), from the caller's statement of which file axes point forward and
    up. Port is up x forward, so the result is always a proper rotation and
    keeps every triangle's winding; whether the file's third axis is port or
    starboard does not matter for a hull that is mirrored anyway."""
    def vec(s):
        s = s.strip().lower()
        v = np.zeros(3)
        v["xyz".index(s[-1])] = -1.0 if s.startswith("-") else 1.0
        return v
    f, u = vec(forward), vec(up)
    if abs(f @ u) > 0.0:
        raise ValueError(f"forward {forward!r} and up {up!r} are the same axis")
    return np.vstack([f, np.cross(u, f), u])


def import_hull(path, volume, scale=1.0, size_max=None, half=True,
                forward="+x", up="+z", curvature=None, closure_rtol=5e-3,
                symmetric=False, exclude=None, min_component=0.0,
                verbose=True):
    """CAD file -> (Capytaine mesh with the waterline at z = 0, report dict).

    `scale` multiplies the file's coordinates (e.g. 1e-3 for mm, 1/68 for a
    1:68 model of a full-scale file). `volume` is the displaced volume at the
    SAME scale. `forward`, `up`: which file axes point to the bow and up,
    e.g. forward="-x", up="-z" for KVLCC2's IGES. `size_max` is in the
    file's units. `symmetric` (half hulls only): return a Capytaine
    ReflectionSymmetricMesh, so the BEM solves two half-size systems instead
    of one full one -- about four times faster, same answer.
    `exclude(lo, hi, area) -> bool` drops whole CAD patches by their bounding
    box (vessel axes, after scaling): appendages such as a rudder that the
    hull's published wetted area and a hull-only BEM leave out.
    """
    import capytaine as cpt
    from hydro.inertia import volume_moments
    P, T, patch = cad_triangles(path, size_max, curvature=curvature)
    P = (P @ file_axes(forward, up).T) * scale
    if exclude is not None:
        _, _, area = _tri_geometry(P, T)
        out = []
        for p in np.unique(patch):
            s = patch == p
            q = P[T[s]].reshape(-1, 3)
            if exclude(q.min(0), q.max(0), float(area[s].sum())):
                out.append(p)
        if verbose:
            print(f"    excluded by the caller: {len(out)} patches, "
                  f"{area[np.isin(patch, out)].sum():.4g} (units^2)")
        keep = ~np.isin(patch, out)
        T, patch = T[keep], patch[keep]
    drop = redundant_patches(P, T, patch)
    if drop:
        keep = ~np.isin(patch, [p for p, _ in drop])
        T, patch = T[keep], patch[keep]
    ids, comp, _ = patch_topology(P, T, patch)
    _, _, area = _tri_geometry(P, T)
    carea = {int(c): float(area[np.isin(patch, ids[comp == c])].sum())
             for c in np.unique(comp)}
    biggest = max(carea.values())
    if verbose:
        top = sorted(carea.values(), reverse=True)
        print(f"    {len(carea)} connected sets of patches; largest "
              f"{top[0]/sum(top):.0%} of the area"
              + (f", next {', '.join(f'{a/sum(top):.1%}' for a in top[1:4])}"
                 if len(top) > 1 else ""))
    pieces = [c for c, a in carea.items() if a < min_component * biggest]
    if pieces:
        gone = np.isin(patch, ids[np.isin(comp, pieces)])
        if verbose:
            q = P[T[gone]].reshape(-1, 3)
            print(f"    separate pieces dropped (under {min_component:.0%} of "
                  f"the largest): {len(pieces)} sets, {area[gone].sum():.4g} "
                  f"units^2, within x {q[:,0].min():.4g}..{q[:,0].max():.4g}, "
                  f"z {q[:,2].min():.4g}..{q[:,2].max():.4g}")
        T, patch = T[~gone], patch[~gone]
    T, flipped = orient_patches(P, T, patch)
    if half:
        P, T = mirror_y(P, T)
    z_w = waterline_for_volume(P, T, volume)
    v, wet = immersed_volume(P, T, z_w)
    V3 = closure_forms(wet)
    spread = float(np.ptp(V3) / np.mean(V3))
    if spread > closure_rtol:
        raise ValueError(
            f"import_hull: the immersed surface is not closed -- V from the "
            f"n_x, n_y, n_z forms = {V3[0]:.6g}, {V3[1]:.6g}, {V3[2]:.6g} "
            f"({spread:.2%} apart). Only V_z off: a horizontal surface (flat "
            f"bottom) is missing or doubled, or the file's z points down "
            f"(z_down). Only V_y off: a side surface is missing.")
    keel = P[:, 2].min()
    mom = volume_moments(wet.faces_centers, wet.faces_normals,
                         wet.faces_areas, rtol=0.02)
    wp = waterplane(P, T, z_w)
    rep = dict(n_panels=wet.nb_faces, flipped_patches=flipped, dropped=drop,
               z_waterline=z_w, draft=z_w - keel, volume=v, V_forms=V3,
               closure=spread, wetted_area=float(wet.faces_areas.sum()),
               x_B=mom["x"] / mom["V"], KB=mom["z"] / mom["V"] + (z_w - keel),
               BM_T=wp["I_T"] / v, **{k: wp[k] for k in wp})
    rep["KM"] = rep["KB"] + rep["BM_T"]
    rep["C_B_wl"] = v / (rep["L_wl"] * rep["B_wl"] * rep["draft"])
    mesh = cpt.Mesh(vertices=P - [0.0, 0.0, z_w],
                    faces=np.column_stack([T, T[:, 2]]),
                    name=os.path.basename(path))
    if half and symmetric:
        # mirror_y appended the mirror image after the original, so the first
        # halves of P and T are the file's half hull
        nP, nT = len(P) // 2, len(T) // 2
        mesh = cpt.ReflectionSymmetricMesh(
            half=cpt.Mesh(vertices=P[:nP] - [0.0, 0.0, z_w],
                          faces=np.column_stack([T[:nT], T[:nT, 2]])),
            plane="xOz", name=os.path.basename(path))
    if verbose:
        print(f"  {os.path.basename(path)}: {rep['n_panels']} wetted panels, "
              f"{flipped} CAD patches re-oriented")
        if drop:
            print("    dropped, lying on other patches: " + ", ".join(
                f"patch {p} ({a:.4g} m2)" for p, a in drop))
        print(f"    waterline at z = {z_w:.4f} (file coords x scale), draft "
              f"{rep['draft']:.4f}; closure: V_x, V_y, V_z agree to "
              f"{spread:.1e}")
    return mesh, rep


# --------------------------------------------------------------- self-test
def _box(L=10.0, B=2.0, D=2.0):
    """Closed box of 12 triangles, keel at z = 0, outward winding."""
    P = np.array([[-L / 2, -B / 2, 0], [L / 2, -B / 2, 0], [L / 2, B / 2, 0],
                  [-L / 2, B / 2, 0], [-L / 2, -B / 2, D], [L / 2, -B / 2, D],
                  [L / 2, B / 2, D], [-L / 2, B / 2, D]], float)
    T = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [1, 2, 6],
                  [1, 6, 5], [0, 4, 7], [0, 7, 3], [0, 1, 5], [0, 5, 4],
                  [3, 7, 6], [3, 6, 2]])
    return P, T


def selftest(verbose=True):
    """Boxes with exact answers: the checks must pass a good box and catch a
    doubled bottom, a missing bottom and an upside-down file."""
    L, B, D, Tw = 10.0, 2.0, 2.0, 0.8
    P, T = _box(L, B, D)
    ok = True
    z_w = waterline_for_volume(P, T, L * B * Tw, tol=1e-10)
    _, wet = immersed_volume(P, T, z_w)
    V3 = closure_forms(wet)
    wp = waterplane(P, T, z_w)
    exact = dict(z_w=Tw, V=L * B * Tw, A=L * B, I_T=L * B ** 3 / 12)
    got = dict(z_w=z_w, V=V3[0], A=wp["A"], I_T=wp["I_T"])
    for k in exact:
        good = abs(got[k] / exact[k] - 1) < 1e-6
        ok &= good
        if verbose:
            print(f"    box {k:<4} {got[k]:.8g}  exact {exact[k]:.8g}  "
                  f"{'ok' if good else 'FAIL'}")
    good = np.ptp(V3) / V3.mean() < 1e-9
    ok &= good
    if verbose:
        print(f"    closed box: three forms agree to {np.ptp(V3)/V3.mean():.1e}"
              f"  {'ok' if good else 'FAIL'}")
    # A doubled bottom, meshed differently the second time. (An EXACT
    # duplicate never reaches the checks: Capytaine 3.0's Mesh drops
    # identical faces on construction -- 14 faces in, 12 kept. A surface
    # that is in the file twice and triangulated twice is not identical.)
    P2, T2 = P, np.vstack([T, [[0, 3, 1], [1, 3, 2]]])
    patch = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
    drop = redundant_patches(P2, T2, patch)
    good = [p for p, _ in drop] == [6]
    ok &= good
    _, wet2 = immersed_volume(P2, T2, z_w, check=False)
    s2 = np.ptp(closure_forms(wet2)) / L / B / Tw
    good2 = s2 > 0.5
    ok &= good2
    if verbose:
        print(f"    doubled bottom: dropped {drop}  {'ok' if good else 'FAIL'};"
              f" if kept, forms {s2:.0%} apart  {'ok' if good2 else 'FAIL'}")
    # a face whose CAD patch runs the wrong way: the edge walk must fix it
    patch6 = np.repeat(np.arange(6), 2)
    Tf = T.copy()
    Tf[6:8] = Tf[6:8][:, ::-1]
    To, nfl = orient_patches(P, Tf, patch6)
    _, wo = immersed_volume(P, To, D + 1.0, check=False)
    Vo = closure_forms(wo)
    good = nfl == 1 and np.allclose(Vo, L * B * D, rtol=1e-9)
    ok &= good
    if verbose:
        print(f"    one face flipped: {nfl} patch re-oriented, V forms "
              f"{np.round(Vo, 6).tolist()} (exact {L*B*D:g})  "
              f"{'ok' if good else 'FAIL'}")
    # a missing bottom, and an open-topped hull read upside down
    for name, keep, flip in (("missing bottom", slice(2, None), False),
                             ("upside down", np.r_[0:2, 4:12], True)):
        Pb = P * [1, -1, -1] if flip else P
        Tb = T[keep]
        zb = waterline_for_volume(Pb, Tb, L * B * Tw)
        _, wb = immersed_volume(Pb, Tb, zb, check=False)
        sb = np.ptp(closure_forms(wb)) / L / B / Tw
        good = sb > 0.5
        ok &= good
        if verbose:
            print(f"    {name}: forms {sb:.0%} apart  "
                  f"{'caught' if good else 'MISSED'}")
    return ok


# KVLCC2 published hydrostatics. SIMMAN 2008/2014 "Geometry and conditions"
# page (full scale), and the SSPA 1:68 roll-decay release (KG, GM; Mendeley
# data, data/external/kvlcc2_rolldecay). SIMMAN lists GM 5.71 without KG.
KVLCC2 = dict(Lpp=320.0, L_wl=325.5, B_wl=58.0, draft=20.8, volume=312622.0,
              wetted_area=27194.0, C_B=0.8098, LCB_pct=3.48, KG=18.6, GM=5.73)


def main():
    import warnings
    warnings.filterwarnings("ignore")
    print("\nSELF-TEST on boxes (exact answers)\n")
    ok = selftest()
    print(f"  {'passed' if ok else 'FAILED'}")
    igs = os.path.join(HERE, "data", "external", "kvlcc2_geometry", "kvlcc2.igs")
    print("\nKVLCC2, full scale, against its published hydrostatics\n")
    pub = KVLCC2
    ref = dict(pub, KM=pub["KG"] + pub["GM"])
    keys = ("draft", "L_wl", "B_wl", "wetted_area", "C_B", "LCB_pct", "KM")
    rows = {}
    for size, curv in ((5.0, None), (5.0, 16), (2.5, 16)):
        print(f"  mesh: size_max {size} m, "
              f"{'curvature ' + str(curv) + ' per turn' if curv else 'no curvature refinement'}")
        # File axes: x from -327.9 at the bulb to +5.5 at the transom, so x
        # points AFT from the AP; z points down. In vessel axes the FP is at
        # x = 320 and midship at x = 160.
        _, rep = import_hull(igs, volume=pub["volume"], scale=1.0,
                             size_max=size, curvature=curv, forward="-x",
                             up="-z")
        rows[(size, curv)] = dict(
            draft=rep["draft"], L_wl=rep["L_wl"], B_wl=rep["B_wl"],
            wetted_area=rep["wetted_area"],
            C_B=rep["volume"] / (pub["Lpp"] * pub["B_wl"] * rep["draft"]),
            LCB_pct=100 * (rep["x_B"] - 160.0) / pub["Lpp"], KM=rep["KM"],
            panels=rep["n_panels"])
    print(f"\n  {'':<12}" + "".join(f"{f'{s:g} m' + (' c' + str(c) if c else ''):>13}"
                                   for s, c in rows) + f"{'published':>11}")
    for k in keys + ("panels",):
        line = f"  {k:<12}" + "".join(f"{r[k]:>13.3f}" for r in rows.values())
        if k in ref:
            last = list(rows.values())[-1][k]
            line += f"{ref[k]:>11.3f}   (finest {last/ref[k]-1:+.2%})"
        print(line)
    print("  wetted area: SIMMAN's 27194 m2 is without the rudder; so is the "
          "file.")
    print(f"  KM published = KG {pub['KG']} + GM {pub['GM']} (SSPA loading "
          f"condition)")
    return rows


if __name__ == "__main__":
    main()
