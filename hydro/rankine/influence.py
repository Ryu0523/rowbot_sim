"""
Exact potential integrals of a flat polygonal panel with unit density.

For a field point x and a flat panel P with unit normal n (into the fluid):

    S(x)     = int_P 1/|x - xi| dA                      (source potential x 4 pi)
    D(x)     = int_P d/dn_xi (1/|x - xi|) dA
             = int_P n.(x - xi)/|x - xi|^3 dA           (normal-dipole potential x 4 pi)
    grad S(x)                                          (source velocity x 4 pi)

Closed forms (derived here; the same results as Hess & Smith 1964 and Newman
1986, J. Eng. Math. 20:113, written in a form free of branch problems):

  local height  z = (x - c).n,  projection p = x - z n,
  edge k from a to b, length s, outward in-plane normal nu = t x n,
  d_k = (a - x).nu (signed distance of p from the edge line, > 0 inside),
  L_k = log((r_a + r_b + s)/(r_a + r_b - s)) = int_edge ds / |x - xi|.

  D = signed solid angle of P seen from x (> 0 on the fluid side), evaluated as
      a fan of triangles with the van Oosterom & Strackee (1983) formula
      tan(Omega/2) = R1.(R2 x R3)/(r1 r2 r3 + (R1.R2) r3 + (R1.R3) r2 + (R2.R3) r1).
  S = sum_k d_k L_k - z D.
      Proof: in the panel plane, div_2[(rho_vec/rho^2)(sqrt(rho^2+z^2) - |z|)] =
      1/sqrt(rho^2 + z^2); the divergence theorem turns the area integral into
      edge integrals sum_k d_k int (R - |z|)/rho^2 ds, which split into
      sum_k d_k L_k and -|z| times the solid angle (written edge by edge,
      d Omega = (d_k/rho^2)(1 - |z|/R) ds).  |z| Omega = z D.
  grad S = -sum_k L_k nu_k - D n.
      Proof: grad_x S = -int (x - xi)/R^3 dA; the normal part is -D; the
      in-plane part is -int grad_xi,par (1/R) dA = -sum_k nu_k int_edge ds/R.

Self terms (x = own centroid): D = 0 (principal value), grad S has no normal
part (the +-2 pi jump is added by the caller), S = sum_k d_k L_k (finite).
The formulas are exact, so singular and near-singular integration need no
special quadrature (the discretisation error is the flat-panel geometry,
O(h^2)). Floating-point precision (measured, check C1): ~1e-15 relative near
the panel; far away the edge sums cancel (terms of size h add up to A/R), so
the relative error grows like eps R/h: ~1e-12 at 3e3 panel sizes, <= 1e-10
to 1e5 sizes. Two rearrangements keep it there (both added after review,
which measured eps (R/h)^2, i.e. 1e-7 at 3e4 sizes, with the direct forms):
L = log1p(2 s/(r_a + r_b - s)) and the triple product of the solid angle
written with panel edge vectors (see _block). Just off the panel plane
(|z| << h) the inputs themselves fix the precision: z = (x - c).n has an
absolute rounding error ~eps |x|, so D and grad S near an edge carry a
relative error ~eps |x|/|z| whatever the formula.
Verified against adaptive quadrature and a 50-digit closed form for a
rectangle in studies/rankine_verify.py (check C1).
"""
import numpy as np
import torch

_DEF_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _t(a, dev, dtype=torch.float64):
    return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=dev)


def _panel_tensors(pan, dev):
    v = _t(pan.verts, dev)                           # (Nj, 4, 3)
    n = _t(pan.n, dev)
    c = _t(pan.c, dev)
    e = torch.roll(v, -1, dims=1) - v                # edges a->b
    s = torch.linalg.norm(e, dim=2)                  # (Nj, 4)
    t = e / torch.clamp(s, min=1e-300)[..., None]
    nu = torch.cross(t, n[:, None, :].expand_as(t), dim=2)
    return v, n, c, s, nu


def _block(x, v, n, c, s, nu, want_grad):
    """x (Ni,3); panel tensors for Nj panels -> S, D (Ni,Nj), gradS (Ni,Nj,3)."""
    R = v[None, :, :, :] - x[:, None, None, :]       # (Ni,Nj,4,3) xi - x
    r = torch.linalg.norm(R, dim=3)                  # (Ni,Nj,4)
    z = torch.einsum("ijk,jk->ij", x[:, None, :] - c[None, :, :], n)
    # edge terms; L = log1p(2 s/(r_a + r_b - s)) keeps full relative
    # precision when r_a + r_b >> s (far field), where log(num/den) of a
    # ratio near 1 would lose log10(R/s) digits
    rb = torch.roll(r, -1, dims=2)
    num = r + rb + s[None]
    # den = r_a + r_b - s without cancellation (added after review): with
    # A = a - x, B = b - x, e = b - a, (r_a + r_b)^2 - s^2 = 2 (r_a r_b + A.B)
    # = 2 |A x e|^2 / (r_a r_b - A.B) (since (r_a r_b)^2 - (A.B)^2 = |A x B|^2
    # and A x B = A x e). Used where A.B < 0 (x between the edge ends, where
    # the direct difference cancels to O(delta^2/s) at distance delta);
    # elsewhere r_a + r_b - s >= 0.29 max(r_a, r_b) and the direct form is exact
    Rn = torch.roll(R, -1, dims=2)
    AB = (R * Rn).sum(dim=3)
    e_ab = torch.roll(v, -1, dims=1) - v                                  # (Nj, 4, 3)
    AxE = torch.cross(R, e_ab[None].expand_as(R), dim=3)
    den_near = 2.0 * (AxE * AxE).sum(dim=3) / ((r * rb - AB) * num)
    den = torch.where(AB < 0, den_near, r + rb - s[None])
    del Rn, AxE
    ok = den > 1e-14 * num
    L = torch.where(ok, torch.log1p(2.0 * s[None] / torch.where(ok, den, num)),
                    torch.zeros_like(num))
    d = torch.einsum("ijkl,jkl->ijk", R, nu)         # (a - x).nu
    # solid angle: fan (0,1,2), (0,2,3). The triple product
    # A.(B x C) = A.((B - A) x (C - A)) exactly (A.(A x .) = 0), and
    # (B - A) x (C - A) = e_AB x e_AC depends on the panel geometry only, so
    # tri = (xi_A - x).(e_AB x e_AC) carries no cancellation; the direct form
    # cancels terms of size R^3 down to |z| h^2 (relative error ~ eps R^2/h^2
    # far away and ~ eps h/|z| just off the plane, found by review)
    Om = torch.zeros_like(z)
    for (p, q, w) in ((0, 1, 2), (0, 2, 3)):
        A, B, C = R[:, :, p], R[:, :, q], R[:, :, w]
        ra, rbq, rc = r[:, :, p], r[:, :, q], r[:, :, w]
        eAB_x_eAC = torch.cross(v[:, q] - v[:, p], v[:, w] - v[:, p], dim=1)   # (Nj, 3)
        tri = torch.einsum("ijk,jk->ij", A, eAB_x_eAC)
        dd = (ra * rbq * rc + torch.einsum("ijk,ijk->ij", A, B) * rc
              + torch.einsum("ijk,ijk->ij", A, C) * rbq
              + torch.einsum("ijk,ijk->ij", B, C) * ra)
        Om = Om + 2.0 * torch.atan2(tri, dd)
    D = -Om                                          # > 0 on the +n side
    # exactly in the panel plane the dipole potential is its principal value
    # 0 (inside: the +-2 pi jump belongs to the caller; outside: 0 anyway)
    D = torch.where(z.abs() <= 1e-13 * s.max(dim=1).values[None], torch.zeros_like(D), D)
    S = (d * L).sum(dim=2) - z * D
    G = None
    if want_grad:
        G = -torch.einsum("ijk,jkl->ijl", L, nu) - D[..., None] * n[None, :, :]
    return S, D, G


def influence(xf, pan, want_grad=False, self_idx=None, device=None,
              chunk_bytes=6e8, as_numpy=True, acc=None, alpha=1.0):
    """S, D (Ni, Nj) and optionally grad S (Ni, Nj, 3) at field points xf for
    the panels `pan`. If `self_idx` is given (array: field point i is the
    centroid of panel self_idx[i], or -1), the self terms are set to their
    principal values exactly: D = 0, grad S normal part = 0.
    acc = (S, D[, G]) torch tensors on `device`: the blocks are added into
    them, times alpha, chunk by chunk (no new N^2 storage; used for the
    image sums of large problems); returns acc. Not with self_idx."""
    dev = device or _DEF_DEVICE
    x_all = _t(xf, dev)
    v, n, c, s, nu = _panel_tensors(pan, dev)
    Ni, Nj = len(x_all), len(v)
    per_row = Nj * 4 * 3 * 8 * 16
    step = max(1, int(chunk_bytes // per_row))
    if acc is not None:
        assert self_idx is None
        for i0 in range(0, Ni, step):
            i1 = min(Ni, i0 + step)
            Sb, Db, Gb = _block(x_all[i0:i1], v, n, c, s, nu, want_grad)
            acc[0][i0:i1].add_(Sb, alpha=alpha)
            acc[1][i0:i1].add_(Db, alpha=alpha)
            if want_grad:
                acc[2][i0:i1].add_(Gb, alpha=alpha)
            del Sb, Db, Gb
        return acc
    S = torch.empty((Ni, Nj), dtype=torch.float64, device=dev)
    D = torch.empty_like(S)
    G = torch.empty((Ni, Nj, 3), dtype=torch.float64, device=dev) if want_grad else None
    for i0 in range(0, Ni, step):
        i1 = min(Ni, i0 + step)
        Sb, Db, Gb = _block(x_all[i0:i1], v, n, c, s, nu, want_grad)
        S[i0:i1] = Sb
        D[i0:i1] = Db
        if want_grad:
            G[i0:i1] = Gb
    if self_idx is not None:
        si = np.asarray(self_idx)
        rows = np.nonzero(si >= 0)[0]
        cols = si[rows]
        rr, cc = _t(rows, dev, torch.long), _t(cols, dev, torch.long)
        D[rr, cc] = 0.0
        if want_grad:
            gn = (G[rr, cc] * n[cc]).sum(dim=1, keepdim=True)
            G[rr, cc] = G[rr, cc] - gn * n[cc]
    if as_numpy:
        out = (S.cpu().numpy(), D.cpu().numpy())
        if want_grad:
            out = out + (G.cpu().numpy(),)
        return out
    return (S, D, G) if want_grad else (S, D)


# ------------------------------------------------- reference (verification)
_GL_X, _GL_W = np.polynomial.legendre.leggauss(20)


def _graded_u_rule(u_s):
    """Rule on [0, 1] for integrands with a near-singular scale u_s at u = 0:
    the sinh map u = u_s sinh(t) (Johnston & Elliott 2005, Int. J. Numer.
    Meth. Eng. 62:564) makes them smooth in t; composite 20-point
    Gauss-Legendre on panels of width <= 0.5 in t. u_s = 0: plain composite
    Gauss-Legendre (the in-plane integrands are then regular)."""
    if u_s <= 0:
        T, npan = 1.0, 4
        t = np.linspace(0, T, npan + 1)
        u = (0.5 * (t[1:] - t[:-1])[:, None] * (_GL_X[None] + 1) + t[:-1, None]).ravel()
        w = (0.5 * (t[1:] - t[:-1])[:, None] * _GL_W[None]).ravel()
        return u, w
    T = np.arcsinh(1.0 / u_s)
    npan = max(2, int(np.ceil(T / 0.5)))
    t = np.linspace(0, T, npan + 1)
    tt = (0.5 * (t[1:] - t[:-1])[:, None] * (_GL_X[None] + 1) + t[:-1, None]).ravel()
    wt = (0.5 * (t[1:] - t[:-1])[:, None] * _GL_W[None]).ravel()
    return u_s * np.sinh(tt), wt * u_s * np.cosh(tt)


def quad_reference(x, verts, n, kind="S", tol=1e-13):
    """Independent value of S, D or grad S ("G") of one flat panel at one
    point, by numerical quadrature only: triangles with the projection p of x
    as apex, a Duffy-type map xi = p + u (a - p + v (b - a)),
    dA = u (a - p) x (b - a) . n du dv (signed), which removes the 1/rho
    singularity; adaptive Gauss-Kronrod (scipy quad) in v and composite
    Gauss-Legendre graded towards the near-singular scale |z|/|e(v)| in u.
    Not valid for grad S at points in the panel plane (z = 0); the check uses
    it there only for S and D. Used only by the verification (check C1)."""
    from scipy.integrate import quad
    x = np.asarray(x, float)
    n = np.asarray(n, float)
    c = verts.mean(axis=0)
    size = max(np.linalg.norm(verts[2] - verts[0]), np.linalg.norm(verts[3] - verts[1]))
    if np.linalg.norm(x - c) > 500 * size:
        # far points (changed after the first run of the 1e3-1e4 rows): the
        # apex-triangle fan below cancels (p far outside the panel) and lost
        # ~1e-10; a 24 x 24 Gauss-Legendre rule on the bilinear map of the
        # flat panel has no cancellation and an error ~(h/2R)^48 here
        # (checked against 30-digit mpmath quadrature)
        g, wg = np.polynomial.legendre.leggauss(24)
        s, t = np.meshgrid(0.5 * (g + 1), 0.5 * (g + 1), indexing="ij")
        w = np.outer(wg, wg).ravel() * 0.25
        s, t = s.ravel()[:, None], t.ravel()[:, None]
        xi = ((1 - s) * (1 - t) * verts[0] + s * (1 - t) * verts[1] + s * t * verts[2]
              + (1 - s) * t * verts[3])
        ds = (1 - t) * (verts[1] - verts[0]) + t * (verts[2] - verts[3])
        dt = (1 - s) * (verts[3] - verts[0]) + s * (verts[2] - verts[1])
        J = np.cross(ds, dt) @ n
        d = x[None] - xi
        R = np.sqrt(np.einsum("ij,ij->i", d, d))
        if kind == "S":
            return float(np.sum(w * J / R))
        if kind == "D":
            return float(np.sum(w * J * (d @ n) / R ** 3))
        return -np.sum((w * J / R ** 3)[:, None] * d, axis=0)
    z = float((x - c) @ n)
    if abs(z) < 1e-14 * (np.abs(verts).max() + np.abs(x).max()):
        z = 0.0                                     # below one ulp: in plane
    p = x - z * n
    ncomp = 3 if kind == "G" else 1
    tot = np.zeros(ncomp)
    for k in range(4):
        a, b = verts[k], verts[(k + 1) % 4]
        if np.linalg.norm(b - a) < 1e-15:
            continue
        ea, eb = a - p, b - a
        jac = float(np.cross(ea, eb) @ n)
        if abs(jac) < 1e-300:
            continue

        def inner(vv, comp=0):
            e = ea + vv * eb
            e = e - (e @ n) * n                       # exactly in the plane
            le = np.linalg.norm(e)
            u, w = _graded_u_rule(abs(z) / le if le > 0 else 1.0)
            xi = p[None] + u[:, None] * e[None]
            dvec = x[None] - xi
            R = np.sqrt(np.einsum("ij,ij->i", dvec, dvec))
            if kind == "S":
                g = u / R
            elif kind == "D":
                g = u * (dvec @ n) / R ** 3
            else:
                g = -u * dvec[:, comp] / R ** 3
            return float(np.dot(w, g))
        for comp in range(ncomp):
            val, _ = quad(lambda vv: inner(vv, comp), 0.0, 1.0, epsabs=0.0,
                          epsrel=tol, limit=500)
            tot[comp] += jac * val
    return tot if kind == "G" else float(tot[0])
