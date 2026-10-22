"""
Double-body (steady base flow) linearisation of the forward-speed problems --
Track B step B6 (port design WAVEBEM_PORT_DESIGN.md P-2 "double-body =
Dawson", step R3 first part; stage note RANKINE_STAGE1_2026-10-06.md section
11, derivations 11.2). Nothing is copied from WaveBEM (licence rule L); the
formulation follows Dawson (1977), Nakos & Sclavounos (1990), Kring (1994),
Raven (1996) and Ogilvie & Tuck (1969) as re-derived in the stage note.

1. Base flow. Body frame, stream -U x_hat, fluid z < 0, normals out of the
   body (into the fluid). Phi = -U x + phi_db, the rigid-lid double-body
   potential: Laplace outside the hull and its image in z = 0, dPhi/dn = 0 on
   the hull, Phi_z = 0 on z = 0 (Phi even in z). W = grad Phi (W -> -U x_hat far
   away), tangent to the hull everywhere and horizontal on z = 0; at the
   waterline W is therefore tangent to the waterline curve (W.nu = 0 for the
   in-plane normal nu of the waterline, for ANY hull, flared or wall-sided --
   the property that removes every waterline term of stage B5, stage note 11.2).
   phi_db by the Green formulation of bie.py (images y and z), on the hull
   panels; W on the hull from the structured surface gradient (normal part 0
   by construction); W and Phi_zz at free-surface points from the
   representation 4 pi phi(x) = sum_images [D phi_B - S phi_n] by fourth-order
   finite differences (Phi_zz by the even extension in z). Surface derivatives
   (W, div_S W, (W.grad) n, W.grad phi) use surface_gradient2: three-point
   Lagrange formulas in chord length along the mesh lines (second order on the
   cosine-clustered meshes; the index-space central differences of
   piercing.structured_gradient are first order there and gave a 59 % m-term
   error at the poles of the sphere mesh, not decreasing with refinement --
   exploration B6-E1).

2. Free-surface conditions linearised about (W, z = 0) (stage note 11.2 items
   1-2): with sigma = -phi_z (free-surface source strength), beach nu >= 0,
       kinematic   zeta_t + div_h(W zeta) = phi_z - nu zeta + f_k
       dynamic     phi_t + W.grad phi = -g zeta - nu phi + f_d
   (div_h(W zeta) = W.grad zeta - Phi_zz zeta: the Phi_zz ("free-surface
   m-term") is the vertical stretching of the base flow). Radiation: f = 0.
   Steady wave problem: f_d = (U^2 - |W|^2)/2, f_k = 0 (zeta is then the full
   steady elevation). Diffraction with the NK incident wave phi_I (which
   satisfies the uniform-stream conditions): f_d = -w.grad phi_I, f_k =
   -div_h(w zeta_I), w = W + U x_hat (stage note 11.2 item 5).

3. Discretisation of the advection (stage note 11.2 item 3). On the B5 band
   grid (unsteady.build_grid), W.grad f = u^i d_i f + u^j d_j f with the
   contravariant components u^i = W.grad i, u^j = W.grad j (metric derivatives
   with the same stencils as the field, so linear functions are exact where
   the field stencil reaches no ghost), each index derivative fifth-order
   upwind-biased by the sign of its contravariant velocity (unsteady.py
   docstring 4): A_U (advective; dynamic condition). Kinematic operator
   (production, "cons"): K_U = Wa^-1 A_{-U}^T Wa (Wa = panel areas), the exact
   area-weighted adjoint of the reversed-flow advective operator. In the
   interior K_U zeta = J^-1 [D6 (J u^d zeta) - (1/60) delta6 (|u^d| J zeta)],
   a conservative form of div_h(W zeta) with the upwind dissipation of the
   right sign WHERE THE GRID IS SMOOTH, and it makes the discrete free-surface
   Green sum between +U and -U solutions vanish identically (discrete
   Timman-Newman reciprocity, stage note 11.2 item 4 -- an identity for ANY A,
   so it does not test the operator). B6 review (stage note 12.2 item 3, rows
   DXa, DXu): K is NOT consistent at grid non-smoothness lines (band kinks at
   the waterline ends, band/lattice junction, starts of the stretching): the
   panel area is not the Jacobian of the six-point metric stencils, the
   discrete metric identity fails there and K 1 grows like 1/h, even for an
   exact base flow; the advective A is consistent. Remedy planned: an
   edge-flux A/K pair (stage note 12.7). Variant "nat": K = A_U - diag(Phi_zz)
   (Phi_zz from the representation; consistency check of the pair).
   Hull wall: the contravariant velocity u^j vanishes at the wall (W tangent to
   the waterline), so the wall is a CHARACTERISTIC boundary of the advection
   and needs no data; the field is continued by even reflection at the wall
   (mirror ghosts), whose O(1) derivative error multiplies u^j = O(j + 1/2) ->
   O(dy) locally. No hull traces (the B5 trace closure is an NK device).

4. Body condition and forces (stage note 11.2 items 6-7). Ogilvie-Tuck
   m-terms of the double-body flow, from surface quantities only:
       (m_1, m_2, m_3) = -(n.grad) W = n div_S W + (W.grad_S) n,
       (m_4, m_5, m_6) = r x (m_1, m_2, m_3) - n x W      (r from the reference),
   i.e. m_k = div_S(W n_k) for every mode (derivation: H n with H = Hess Phi
   symmetric; W.n = 0 on S gives the tangential part -(W.grad) n and Laplace
   with Phi_n = 0 the normal part Phi_nn = -div_S W). At a sharp stem the base
   flow is a wedge stagnation flow, m ~ r^(pi/(pi - alpha) - 2) (Wigley I:
   r^-0.92), and at a flared waterline the cross-flow ~ r^(1/8) gives m ~
   r^(-7/8): integrable, but no centroid formula converges their integrals
   (exploration B6-E2). Production (gop "fv", mterms "sbp"): the discrete
   W.grad on the hull is the edge-flux form G (wgrad_operator_fv; zero flux
   through the waterline, keel and stem edges), and m_k = -Wa^-1 G^T Wa n_k,
   which equals G n_k + n_k div W panel by panel (discrete summation by parts)
   and makes Tuck's identity exact for every panel function. Information:
   "pointwise" (the formula at the centroids with surface_gradient2; sphere,
   analytic flow: 2.9 % -> 0.4 % from 24x12 to 64x32) and gop "lagr" (the
   three-point Lagrange G, whose adjoint has spurious boundary fluxes at the
   waterline row, B6-E3). Direct pressure (D-form): p = -rho (phi_t + W.grad
   phi), T_jk = -rho int_S (i w phi_k + W.grad phi_k) n_j, W.grad = G;
   Ogilvie-Tuck form T^OT_jk = -rho int_S phi_k (i w n_j - m_j). With W tangent
   to S and to the waterline, Tuck's theorem holds with NO waterline term: the
   two forms coincide in the continuum, and with the production m-terms also
   discretely (to round-off).
"""
import numpy as np
import scipy.sparse as sp
import torch

from .bie import ImageOperator, neumann_green
from .unsteady import (ForwardSpeedLTI, Coefficients, _index_derivative, _geom_derivative, _odd_mirror_derivative,
                       _ts, build_grid, incident)


# ============================================================ surface calculus
def _lagr3(s0, s1, s2, at):
    """Derivative weights at s_at (= s0, s1 or s2) of the quadratic through (s0, s1, s2)."""
    w0 = ((at - s1) + (at - s2)) / ((s0 - s1) * (s0 - s2))
    w1 = ((at - s0) + (at - s2)) / ((s1 - s0) * (s1 - s2))
    w2 = ((at - s0) + (at - s1)) / ((s2 - s0) * (s2 - s1))
    return w0, w1, w2


def _line_deriv(F, S, axis):
    """d F / d s along one index direction of a structured grid, F (nx, nz, ...) and the chord-length
    parameter S (nx, nz) along that direction: three-point Lagrange formulas in s (second order on
    non-uniform spacing; centred inside, one-sided at both ends)."""
    F = np.moveaxis(np.asarray(F, float), axis, 0)
    S = np.moveaxis(np.asarray(S, float), axis, 0)
    e = (Ellipsis,) + (None,) * (F.ndim - S.ndim)
    D = np.empty_like(F)
    w0, w1, w2 = _lagr3(S[:-2], S[1:-1], S[2:], S[1:-1])
    D[1:-1] = w0[e] * F[:-2] + w1[e] * F[1:-1] + w2[e] * F[2:]
    for (i0, i1, i2, at) in ((0, 1, 2, 0), (-3, -2, -1, -1)):
        a, b, c = _lagr3(S[i0], S[i1], S[i2], S[at])
        D[at] = a[e] * F[i0] + b[e] * F[i1] + c[e] * F[i2]
    return np.moveaxis(D, 0, axis)


CHORD_FLOOR = 1e-9     # relative floor of a chord increment (B6 review, finding B5; stage note 12.2 item 6)


def _chord(C, axis):
    """Cumulative chord length of the centroids C (nx, nz, 3) along an index direction (start 0 per line).
    Degenerate chords (coincident centroids, possible on collapsed sliver panels at a pointed end) are floored at
    CHORD_FLOOR x the line's longest increment, so the three-point Lagrange weights stay finite (guard added after
    the B6 review; it changes no chord of the meshes of the B4-B6 rows, whose smallest increment is ~1e-2 of the
    longest)."""
    d = np.linalg.norm(np.diff(C, axis=axis), axis=-1)
    dmax = np.max(d, axis=axis, keepdims=True)
    d = np.maximum(d, CHORD_FLOOR * np.maximum(dmax, 1e-300))
    pad = [(0, 0)] * d.ndim
    pad[axis] = (1, 0)
    return np.cumsum(np.pad(d, pad), axis=axis)


def _line_matrix(S, axis):
    """Sparse (N x N) matrix of _line_deriv along `axis` for the structured ordering k = i * nz + j."""
    nx, nz = S.shape
    idx = np.arange(nx * nz).reshape(nx, nz)
    Sm = np.moveaxis(S, axis, 0)
    Im = np.moveaxis(idx, axis, 0)
    n = Sm.shape[0]
    rows, cols, vals = [], [], []
    w0, w1, w2 = _lagr3(Sm[:-2], Sm[1:-1], Sm[2:], Sm[1:-1])
    for w, sh in ((w0, -1), (w1, 0), (w2, 1)):
        rows.append(Im[1:-1].ravel())
        cols.append(Im[1 + sh:n - 1 + sh].ravel())
        vals.append(w.ravel())
    for (i0, i1, i2, at) in ((0, 1, 2, 0), (n - 3, n - 2, n - 1, n - 1)):
        a, b, c = _lagr3(Sm[i0], Sm[i1], Sm[i2], Sm[at])
        for w, ii in ((a, i0), (b, i1), (c, i2)):
            rows.append(Im[at].ravel())
            cols.append(Im[ii].ravel())
            vals.append(w.ravel())
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nx * nz,) * 2)


def wgrad_operator(pan, W, shape):
    """Sparse G with (G f)_p = W_p . grad_S f at the panels (surface_gradient2 with phin = 0): G = diag(a) D_s +
    diag(b) D_t, (a, b) the first two components of M_p^-T W_p, M_p = [r_s; r_t; n] (module docstring 4)."""
    nx, nz = shape
    C = pan.c.reshape(nx, nz, 3)
    S0, S1 = _chord(C, 0), _chord(C, 1)
    rs, rt = _line_deriv(C, S0, 0), _line_deriv(C, S1, 1)
    M = np.stack([rs, rt, pan.n.reshape(nx, nz, 3)], axis=-2).reshape(-1, 3, 3)
    ab = np.linalg.solve(np.swapaxes(M, 1, 2), np.asarray(W, float)[..., None])[..., 0]
    Ds, Dt = _line_matrix(S0, 0), _line_matrix(S1, 1)
    return sp.diags(ab[:, 0]) @ Ds + sp.diags(ab[:, 1]) @ Dt


def wgrad_operator_fv(pan, W, shape):
    """Edge-flux (finite-volume) form of W.grad_S on a structured panel mesh (module docstring 4, "fv"):
        (G f)_p = (1/(2 a_p)) sum_e F_e (f_q - f_p),    (div W)_p = (1/a_p) sum_e F_e,
    over the interior edges e = (p, q) of the structured mesh, with the edge flux F_e = W_e . nu_e |e| (W_e the mean
    of the two panels, nu_e the mean of the outward edge normal of p and the inward one of q, i.e. F_e^(q) = -F_e^(p)
    exactly). Mesh-boundary edges carry ZERO flux: the waterline (the base flow is tangent to it), the keel and the
    first/last stations on the centre plane (y-symmetry: W lies along these lines or vanishes) and degenerate polar
    edges. Properties: G 1 = 0, and the discrete summation by parts Wa G + (Wa G)^T = -diag(Wa div W) holds exactly,
    the counterpart of int g W.grad f + int f W.grad g = -int f g div W with no boundary term, so the adjoint
    m-terms -Wa^-1 G^T Wa n_k equal G n_k + n_k div W = div_S(W n_k) panel by panel. Returns (G, divW)."""
    nx, nz = shape
    N = nx * nz
    raw = pan.raw.reshape(nx, nz, 4, 3)
    C = pan.c.reshape(nx, nz, 3)
    nrm = pan.n.reshape(nx, nz, 3)
    Wv = np.asarray(W, float).reshape(nx, nz, 3)
    idx = np.arange(N).reshape(nx, nz)
    rows, cols, vals = [], [], []
    div = np.zeros(N)
    for (di, dj) in ((1, 0), (0, 1)):
        P = raw[:nx - di, :nz - dj]
        Q = raw[di:, dj:]
        # the two shared corners: the two closest corner pairs between the panels
        d = np.linalg.norm(P[:, :, :, None, :] - Q[:, :, None, :, :], axis=-1)       # (.., 4, 4)
        flat = d.reshape(d.shape[:2] + (16,))
        o = np.argsort(flat, axis=-1)[..., :2]
        ia, ib = o // 4, o % 4
        a0 = np.take_along_axis(P, ia[..., 0:1, None].repeat(3, -1), axis=2)[:, :, 0]
        a1 = np.take_along_axis(P, ia[..., 1:2, None].repeat(3, -1), axis=2)[:, :, 0]
        e = a1 - a0
        Le = np.linalg.norm(e, axis=-1)
        cp, cq = C[:nx - di, :nz - dj], C[di:, dj:]
        npn, nqn = nrm[:nx - di, :nz - dj], nrm[di:, dj:]
        nup = np.cross(e, npn)
        nup /= np.maximum(np.linalg.norm(nup, axis=-1), 1e-300)[..., None]
        nup *= np.sign(np.einsum("ijk,ijk->ij", nup, cq - cp))[..., None]
        nuq = np.cross(e, nqn)
        nuq /= np.maximum(np.linalg.norm(nuq, axis=-1), 1e-300)[..., None]
        nuq *= np.sign(np.einsum("ijk,ijk->ij", nuq, cp - cq))[..., None]
        nub = nup - nuq
        nub /= np.maximum(np.linalg.norm(nub, axis=-1), 1e-300)[..., None]
        We = 0.5 * (Wv[:nx - di, :nz - dj] + Wv[di:, dj:])
        F = np.einsum("ijk,ijk->ij", We, nub) * Le                      # outward from p
        ip, iq = idx[:nx - di, :nz - dj].ravel(), idx[di:, dj:].ravel()
        F = F.ravel()
        np.add.at(div, ip, F)
        np.add.at(div, iq, -F)
        # (G f)_p += F (f_q - f_p) / (2 a_p);  (G f)_q += (-F)(f_p - f_q) / (2 a_q)
        rows += [ip, ip, iq, iq]
        cols += [iq, ip, ip, iq]
        vals += [0.5 * F, -0.5 * F, -0.5 * F, 0.5 * F]
    a = pan.area
    Gm = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
    Gm.sum_duplicates()
    return sp.diags(1.0 / a) @ Gm, div / a


def surface_gradient2(pan, phi, phin, shape):
    """Full gradient of panel values on a structured mesh (stations x rows, index i * nz + j) as
    piercing.structured_gradient, but with second-order NON-UNIFORM three-point formulas in chord length
    along each index direction for both the field and the centroid positions (the index-space central
    differences of structured_gradient are only first order where the spacing changes quickly, e.g. at the
    cosine-clustered poles and hull ends -- exploration B6-E1). Solves [r_s; r_t; n] grad = [f_s, f_t, phin]."""
    nx, nz = shape
    C = pan.c.reshape(nx, nz, 3)
    F = np.asarray(phi, float).reshape(nx, nz)
    S0, S1 = _chord(C, 0), _chord(C, 1)
    rs, rt = _line_deriv(C, S0, 0), _line_deriv(C, S1, 1)
    fs, ft = _line_deriv(F, S0, 0), _line_deriv(F, S1, 1)
    Nn = pan.n.reshape(nx, nz, 3)
    M = np.stack([rs, rt, Nn], axis=-2)
    rhs = np.stack([fs, ft, np.asarray(phin, float).reshape(nx, nz)], -1)
    return np.linalg.solve(M, rhs[..., None])[..., 0].reshape(-1, 3)


# ===================================================================== base flow
class DoubleBodyFlow:
    """Rigid-lid double-body flow about the panelled hull (half hull y >= 0,
    structured mesh of shape `shape` = (stations, rows) with panel index
    i * nz + j, as hull.mesh and the sphere mesh of panels.ellipsoid produce).
    lid=False: no z-image (unbounded double body; used for the analytic
    sphere comparison). zpar = -1 (information only, B6 review experiment 3'):
    the pressure-release image (perturbation potential odd in z, phi = 0 on
    z = 0), the Fn -> infinity limit of the steady free-surface condition; the
    base flow then crosses z = 0 at the waterline, so the edge-flux operator
    "fv" (zero waterline flux) is not consistent there -- use gop="lagr" and
    pointwise m-terms with it."""

    def __init__(self, body, U, shape, device=None, lid=True, gop="fv", zpar=1):
        self.body, self.U, self.shape = body, float(U), tuple(shape)
        self.gop = gop
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lid = bool(lid)
        self.zpar = int(zpar)
        self.axes = (1, 2) if lid else (1,)
        self.par = (1, self.zpar) if lid else (1,)
        self.phin = self.U * body.n[:, 0]
        op = ImageOperator(body, axes=self.axes, device=self.dev, parities=[self.par])
        self.phi = neumann_green(op, self.phin, self.par, device=self.dev)
        del op
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        g = surface_gradient2(body, self.phi, self.phin, self.shape)
        self.W = g - np.array([self.U, 0.0, 0.0])[None, :]          # total base velocity on the hull panels
        if gop == "fv":                                               # discrete W.grad_S on the panels (sparse)
            self.G, self.divW = wgrad_operator_fv(body, self.W, self.shape)
        else:
            self.G, self.divW = wgrad_operator(body, self.W, self.shape), None

    # ------------------------------------------------------------ hull
    def surface_div(self, V):
        """div_S V of a tangential vector field V (N, 3) on the hull panels (structured surface gradient of
        each Cartesian component, normal part 0)."""
        z = np.zeros(len(self.body))
        return sum(surface_gradient2(self.body, V[:, c], z, self.shape)[:, c] for c in range(3))

    def normal_derivative_along(self, V):
        """(V.grad_S) n on the hull panels (structured surface gradient of the panel-normal components)."""
        z = np.zeros(len(self.body))
        n = self.body.n
        return np.stack([np.einsum("ij,ij->i", surface_gradient2(self.body, n[:, c], z, self.shape), V)
                         for c in range(3)], 1)

    def mode_normals(self, ref=(0.0, 0.0), modes=(1, 3, 5)):
        """Generalised normals n_k of the symmetric modes (pitch about (x_ref, z_ref), bow down positive)."""
        n, c = self.body.n, self.body.c
        r = c - np.array([ref[0], 0.0, ref[1]])[None, :]
        out = {}
        for k in modes:
            if k == 1:
                out[k] = n[:, 0].copy()
            elif k == 3:
                out[k] = n[:, 2].copy()
            elif k == 5:
                out[k] = r[:, 2] * n[:, 0] - r[:, 0] * n[:, 2]
            else:
                raise ValueError("only the port/starboard symmetric modes 1, 3, 5")
        return out

    def mterms(self, ref=(0.0, 0.0), modes=(1, 3, 5), method="sbp"):
        """Ogilvie-Tuck m-terms of the double-body flow for the symmetric modes (module docstring 4); returns
        dict mode -> (n_k, m_k).
        method "sbp" (production): m_k = -Wa^-1 G^T Wa n_k, the area-weighted adjoint of the SAME discrete W.grad
          operator G that the direct-pressure force uses (summation by parts). Then the discrete Tuck identity
          sum_p a_p (n_j G f + m_j f)_p = 0 holds EXACTLY for every panel function f, so the D-form and the OT form
          coincide discretely as they do in the continuum (no waterline term for the double-body flow); in the
          continuum -G^T n_k -> div_S(W n_k) = n_k div_S W + W.grad n_k, which is the m-term of every mode
          (translations: n div_S W + (W.grad) n; rotations: r x m_trans - n x W). Why: at a sharp stem of
          waterline half-angle alpha the base flow is a wedge stagnation flow, W ~ r^(pi/(pi - alpha) - 1)
          (Wigley I: r^0.08), so m ~ r^-0.92 is barely integrable and a pointwise formula cannot converge its
          integral (exploration B6-E2: int m_1 dS, exactly 0 in the continuum, stayed at 45-63 % of int |m_1|
          from 40 to 135 stations); the adjoint form carries the integral exactly (sum_p a_p m_p = 0 since G 1 = 0).
        method "pointwise": m_trans = n div_S W + (W.grad_S) n, m_rot = r x m_trans - n x W evaluated at the panel
          centroids with surface_gradient2 (second order where the base flow is smooth: sphere 0.4 % at 64 x 32)."""
        nk = self.mode_normals(ref, modes)
        if method == "sbp":
            a = self.body.area
            GT = self.G.T.tocsr()
            out = {k: (nk[k], -(GT @ (a * nk[k])) / a) for k in modes}
            return out
        n, c, W = self.body.n, self.body.c, self.W
        mt = n * self.surface_div(W)[:, None] + self.normal_derivative_along(W)
        r = c - np.array([ref[0], 0.0, ref[1]])[None, :]
        mr = np.cross(r, mt) - np.cross(n, W)
        comp = {1: mt[:, 0], 3: mt[:, 2], 5: mr[:, 1]}
        self.m_trans, self.m_rot = mt, mr
        return {k: (nk[k], comp[k].copy()) for k in modes}

    def wgrad(self, phi, phin=None):
        """W.grad phi on the hull panels = G phi (complex allowed; W is tangential, so the normal derivative does
        not enter: phin is accepted for interface compatibility and ignored)."""
        phi = np.asarray(phi)
        if np.iscomplexobj(phi):
            return self.G @ phi.real + 1j * (self.G @ phi.imag)
        return self.G @ phi

    # ------------------------------------------------------------ field
    def potential(self, P, chunk=8192):
        """phi_db at field points P (body + images; exterior identity of bie.py)."""
        P = np.asarray(P, float)
        out = np.empty(len(P))
        for r0 in range(0, len(P), chunk):
            r1 = min(len(P), r0 + chunk)
            op = ImageOperator(self.body, axes=self.axes, xf=P[r0:r1], device=self.dev, parities=[self.par])
            out[r0:r1] = (op.D(self.par) @ self.phi - op.S(self.par) @ self.phin) / (4 * np.pi)
            del op
        return out

    def distance_to_hull(self, P, chunk=256):
        """Exact distance from points P to the flat hull panels and their y and z images (point-to-polygon:
        |height| if the projection falls inside the panel, else the distance to the nearest edge segment)."""
        P = np.asarray(P, float)
        pans = [self.body.verts]
        for ax in self.axes:
            pans = pans + [v * np.where(np.arange(3) == ax, -1.0, 1.0) for v in pans]
        V = np.concatenate(pans)                                  # (Np, 4, 3); orientation irrelevant here
        c = V.mean(axis=1)
        nrm = np.cross(V[:, 2] - V[:, 0], V[:, 3] - V[:, 1])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-300)[:, None]
        A = V
        E = np.roll(V, -1, axis=1) - V                            # edges a -> b
        EE = np.maximum(np.einsum("pkd,pkd->pk", E, E), 1e-300)
        out = np.empty(len(P))
        for r0 in range(0, len(P), chunk):
            Q = P[r0:r0 + chunk]
            z = np.einsum("qpd,pd->qp", Q[:, None, :] - c[None], nrm)
            proj = Q[:, None, :] - z[..., None] * nrm[None]       # (q, p, 3)
            rel = proj[:, :, None, :] - A[None]                   # (q, p, 4, 3)
            side = np.einsum("pd,qpkd->qpk", nrm, np.cross(E[None], rel))
            inside = np.all(side >= 0, axis=2) | np.all(side <= 0, axis=2)
            t = np.clip(np.einsum("qpkd,pkd->qpk", rel, E) / EE[None], 0.0, 1.0)
            dseg = np.linalg.norm(rel - t[..., None] * E[None], axis=3).min(axis=2)
            dist = np.sqrt(z * z + np.where(inside, 0.0, dseg) ** 2)
            out[r0:r0 + chunk] = dist.min(axis=1)
        return out

    def _fd_step(self, P, h):
        """Finite-difference step per point: h0 = 1e-3 x the hull's x-extent (the B6 default), reduced to a
        quarter of the distance to the hull panels (distance_to_hull) where a point is closer than 4 h0 (so
        every stencil point stays at least half that distance away), floored at 1e-3 h0 (round-off). B6 review, finding B8: the fixed
        h0 with +-2 h0 stencils gave a 56 % error at 0.00125 L from the wall (latent: the production grids keep
        their first free-surface centroids beyond 4 h0, where the step is unchanged)."""
        if h is not None:
            return np.full(len(P), float(h))
        h0 = 1e-3 * float(np.ptp(self.body.c[:, 0]))
        hs = np.full(len(P), h0)
        # cheap lower bound first (centroid distance minus the panel's circumradius, hull and its y/z images); the
        # exact point-to-panel distance only where the bound is below 4 h0
        c = self.body.c
        r = np.linalg.norm(self.body.verts - c[:, None, :], axis=2).max(axis=1)
        cs = [c]
        for ax in self.axes:
            cs = cs + [q * np.where(np.arange(3) == ax, -1.0, 1.0) for q in cs]
        C = np.concatenate(cs)
        R = np.tile(r, len(cs))
        lb = np.empty(len(P))
        for r0 in range(0, len(P), 2048):
            Q = P[r0:r0 + 2048]
            lb[r0:r0 + 2048] = (np.sqrt(np.maximum(np.einsum("qd,qd->q", Q, Q)[:, None] - 2 * Q @ C.T
                                                   + np.einsum("pd,pd->p", C, C)[None], 0.0)) - R[None]).min(axis=1)
        near = lb < 4 * h0
        if np.any(near):
            d = self.distance_to_hull(P[near])
            hs[near] = np.clip(0.25 * d, 1e-3 * h0, h0)
        return hs

    def field_velocity(self, P, h=None):
        """Total base velocity W = (-U, 0, 0) + grad phi_db at points P (x and y components; W_z = 0 on z = 0
        with the lid) by fourth-order central differences of the representation, step per point _fd_step (h0 =
        1e-3 x the hull's x-extent away from the hull; h given: that step everywhere)."""
        P = np.asarray(P, float)
        hs = self._fd_step(P, h)
        out = np.zeros((len(P), 2))
        for a in (0, 1):
            e = np.zeros((len(P), 3))
            e[:, a] = hs
            f = {m: self.potential(P + m * e) for m in (-2, -1, 1, 2)}
            out[:, a] = (-f[2] + 8 * f[1] - 8 * f[-1] + f[-2]) / (12 * hs)
        out[:, 0] -= self.U
        return out

    def field_phizz(self, P, h=None):
        """Phi_zz at points on z = 0 by the even extension in z (fourth order):
        Phi_zz = (16 Phi(-h) - Phi(-2h) - 15 Phi(0)) / (6 h^2), step per point as field_velocity."""
        P = np.asarray(P, float)
        hs = self._fd_step(P, h)
        e = np.zeros((len(P), 3))
        e[:, 2] = hs
        f0, f1, f2 = self.potential(P), self.potential(P - e), self.potential(P - 2 * e)
        return (16 * f1 - f2 - 15 * f0) / (6 * hs * hs)


# ============================================================ free-surface operators
def _adv_matrix(G, Wx, Wy, sgn):
    """Sparse A ~ W.grad on the free-surface panels of the band grid G (ordering k = j nx + i), Wx, Wy (nx, ny)
    the base velocity at the centroids (module docstring 3): A f = u^i D_i f + u^j D_j f, each index derivative
    fifth-order upwind-biased against its contravariant velocity (upwind side = -sign(u)), metric derivatives with
    the same stencils (linear ghosts) so that linear functions are differentiated exactly wherever the field
    stencil reaches no ghost. Ends: inflow ghosts zero at the upstream end (sgn = sign(U): stream toward -x for
    sgn > 0), fully upwind second-order formulas at the outflow end, zero ghosts at the lateral edge, mirror
    (even) ghosts at the centre plane AND at the hull wall (characteristic boundary: u^j = 0 there).
    Returns (A, info)."""
    nx, ny = G["shape"]
    X, Y = G["Xc"], G["Yc"]
    hull_j0 = np.zeros(nx, bool)
    if G["J"] > 0:
        hull_j0[G["i_s"]:G["i_b"]] = True
    s = 1.0 if sgn >= 0 else -1.0
    Xi0, Yi0 = _geom_derivative(X, 0), _geom_derivative(Y, 0)
    Xj0, Yj0 = np.empty_like(X), np.empty_like(Y)
    for i in range(nx):
        mir = not hull_j0[i]
        Xj0[i] = _geom_derivative(X[i], 0, mirror_lo=mir, odd=False)
        Yj0[i] = _geom_derivative(Y[i], 0, mirror_lo=mir, odd=True)
    J0 = Xi0 * Yj0 - Xj0 * Yi0
    side_i = -np.sign((Wx * Yj0 - Wy * Xj0) / J0)
    side_j = -np.sign((-Wx * Yi0 + Wy * Xi0) / J0)
    gi_lo, gi_hi = ("zero", "onesided") if s < 0 else ("onesided", "zero")
    Di_f = []
    Xi, Yi = np.empty_like(X), np.empty_like(Y)
    Xj, Yj = np.empty_like(X), np.empty_like(Y)
    for j in range(ny):
        Dg = _index_derivative(nx, side_i[:, j], "linear", "linear")
        Xi[:, j], Yi[:, j] = Dg @ X[:, j], Dg @ Y[:, j]
        Di_f.append(_index_derivative(nx, side_i[:, j], gi_lo, gi_hi))
    Dj_f = []
    for i in range(nx):
        if hull_j0[i]:
            Dg = _index_derivative(ny, side_j[i], "linear", "linear")
            Xj[i], Yj[i] = Dg @ X[i], Dg @ Y[i]
        else:
            Dg = _index_derivative(ny, side_j[i], "mirror", "linear")
            Xj[i] = Dg @ X[i]
            Yj[i] = _odd_mirror_derivative(Y[i], side_j[i], ny)
        Dj_f.append(_index_derivative(ny, side_j[i], "mirror", "zero"))
    Jac = Xi * Yj - Xj * Yi
    assert np.all(Jac > 0), "grid not right-handed"
    ui = (Wx * Yj - Wy * Xj) / Jac
    uj = (-Wx * Yi + Wy * Xi) / Jac
    rows, cols, vals = [], [], []
    for j in range(ny):
        Di = sp.coo_matrix(Di_f[j])
        rows.append(j * nx + Di.row)
        cols.append(j * nx + Di.col)
        vals.append(Di.data * ui[Di.row, j])
    for i in range(nx):
        Dj = sp.coo_matrix(Dj_f[i])
        rows.append(Dj.row * nx + i)
        cols.append(Dj.col * nx + i)
        vals.append(Dj.data * uj[i, Dj.row])
    N = nx * ny
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
    A.sum_duplicates()
    return A, dict(ui=ui, uj=uj, Jac=Jac, side_i=side_i, side_j=side_j)


def fs_operators(G, Wf, sgn, kinematic="cons", phizz=None, area=None):
    """Advective (dynamic) and kinematic free-surface operators of the double-body linearisation on grid G
    (module docstring 3; "cons" is inconsistent at grid non-smoothness lines, B6 review). Wf (NF, 2): base velocity (x, y) at the free-surface centroids in the panel order of
    G["fs"] (k = j nx + i). kinematic "cons": K = Wa^-1 A_{-U}^T Wa (Wa = panel areas; production); "nat": K =
    A - diag(phizz) (phizz = Phi_zz at the centroids, required). Returns (A, K, info)."""
    nx, ny = G["shape"]
    Wx = np.asarray(Wf[:, 0], float).reshape(ny, nx).T
    Wy = np.asarray(Wf[:, 1], float).reshape(ny, nx).T
    A, info = _adv_matrix(G, Wx, Wy, sgn)
    if kinematic == "cons":
        Ar, _ = _adv_matrix(G, -Wx, -Wy, -sgn)
        a = G["fs"].area if area is None else np.asarray(area, float)
        K = sp.diags(1.0 / a) @ Ar.T.tocsr() @ sp.diags(a)
    elif kinematic == "nat":
        if phizz is None:
            raise ValueError("kinematic='nat' needs phizz")
        K = A - sp.diags(np.asarray(phizz, float))
    else:
        raise ValueError(kinematic)
    return sp.csr_matrix(A), sp.csr_matrix(K), info


# ============================================ edge-flux (finite-volume) free-surface operators (S1, 2026-10-09)
# Opt-in (kinematic="flux" of build_db_solver; default "cons" unchanged). Stage note 12.7, derivation in
# studies/_cache/derive/s1_progress_zh.md. Construction:
#   1. geometric face fluxes F_e = int_e W.nu dl (two-point Gauss along each face of the structured grid; hull-wall
#      and centre-plane faces F = 0), cell divergence (div W)_c = (1/a_c) sum_e F_e: the discrete Gauss theorem holds
#      on ANY grid (K 1 = div W by construction, exact for a linear W);
#   2. conservative central operator K_c f = (1/a_c) sum_e F_e f_e, face value f_e from a six-cell finite-volume
#      reconstruction along the grid line through the face in the chord coordinate (primitive-function
#      construction: exact for polynomials of degree 5 on any 1-D spacing; weights (1, -8, 37, 37, -8, 1)/60 on a
#      uniform line, whose face difference is the sixth-order central derivative = the central part of the B5
#      upwind scheme); variant "lin": two-cell distance weights (the transposed operator is then exact for linear
#      functions on any 1-D spacing);
#   3. exact adjoint pair: S = (Wa K_c - (Wa K_c)^T)/2 (skew), D = diag(Wa div W), Wa V = sum_e (|F_e|/60) D3_e^T D3_e
#      (third difference across each face; symmetric positive semi-definite, even in U):
#          A_U = Wa^-1 (S - D/2) + V,   K_U = Wa^-1 (S + D/2) + V,   so  Wa K_U = (Wa A_{-U})^T  exactly.
#      Consistency: K_c -> div(W f); Wa^-1 K_c^T Wa -> -W.grad f (the dual, exact for linear f with "lin"); hence
#      K_U -> div(W f), A_U -> W.grad f. On a uniform grid with a uniform W both reduce to the B5 operators
#      (sixth-order central + (1/60) delta6 dissipation) to round-off.
FLUX_DISS = 1.0 / 60.0


def fs_nodes(G):
    """Node (cell-corner) coordinates (nx+1, ny+1) of the structured free-surface grid G of unsteady.build_grid:
    Xn[i, j] = xe[i]; Yn[i, j] = ywl_e[i] (1 - j/J) + j dy0 for j <= J (band lines bent around the waterline),
    ylines[j] beyond the band."""
    nx, ny = G["shape"]
    xe, yw, yl, J, dy0 = G["xe"], G["ywl_e"], G["ylines"], G["J"], G["dy0"]
    Xn = np.repeat(np.asarray(xe, float)[:, None], ny + 1, axis=1)
    if G.get("Yn") is not None:                               # opt-in graded band (build_grid band="geom")
        return Xn, np.asarray(G["Yn"], float).copy()
    jj = np.arange(ny + 1)
    Yn = np.repeat(np.asarray(yl, float)[None, :], nx + 1, axis=0)
    if J > 0:
        band = jj <= J
        Yn[:, band] = np.asarray(yw, float)[:, None] * (1.0 - jj[band][None, :] / J) + jj[band][None, :] * dy0
    return Xn, Yn


def _lagr_deriv_rows(nodes, x0):
    """Derivative weights at x0 (nf,) of the Lagrange interpolants through nodes (nf, q): returns (nf, q)."""
    nf, q = nodes.shape
    out = np.zeros((nf, q))
    for m in range(q):
        den = np.ones(nf)
        for r in range(q):
            if r != m:
                den = den * (nodes[:, m] - nodes[:, r])
        num = np.zeros(nf)
        for qq in range(q):
            if qq == m:
                continue
            t = np.ones(nf)
            for r in range(q):
                if r != m and r != qq:
                    t = t * (x0 - nodes[:, r])
            num += t
        out[:, m] = num / den
    return out


def _line_recon(sf, sc, lo, hi, recon="fv6t"):
    """Face-value reconstruction and third-difference stencils along ONE grid line of n cells. sf (n+1,): chord
    coordinates of the faces (cell k spans sf[k]..sf[k+1]); sc (n,): chord coordinates of the cell centroids. End
    modes lo/hi: "mirror" (even reflection: wall, centre plane), "zero" (ghost values 0: inflow, lateral edge),
    "onesided" (outflow: the stencil is shifted inward so that it uses real cells only). recon "fv6": six-cell
    finite-volume reconstruction (primitive-function construction, exact for degree-5 polynomial cell averages on any
    spacing); "lin": two-cell distance weights. Returns (rc (n+1, m) int cell indices, -1 = none, rw (n+1, m) weights,
    d3c (n+1, 4) int, d3w (n+1, 4): the third difference across each face, reduced to a first difference where a
    one-sided end leaves fewer than four cells, empty (all -1) at a boundary face without two real cells)."""
    n = len(sf) - 1
    p = 1 if recon == "lin" else 3
    h = np.diff(sf)
    # extended face/cell coordinates with p ghost cells on each side
    gl = [sf[0] - (sf[m] - sf[0]) for m in range(p, 0, -1)] if lo == "mirror" else [sf[0] - m * h[0] for m in range(p, 0, -1)]
    gh = [sf[-1] + (sf[-1] - sf[n - m]) for m in range(1, p + 1)] if hi == "mirror" else [sf[-1] + m * h[-1] for m in range(1, p + 1)]
    sfe = np.concatenate([gl, sf, gh])
    scl = [sf[0] - (sc[m] - sf[0]) for m in range(p - 1, -1, -1)] if lo == "mirror" else [sf[0] - (m + 0.5) * h[0] for m in range(p - 1, -1, -1)]
    sch = [sf[-1] + (sf[-1] - sc[n - 1 - m]) for m in range(p)] if hi == "mirror" else [sf[-1] + (m + 0.5) * h[-1] for m in range(p)]
    sce = np.concatenate([scl, sc, sch])
    # extended cell index ke -> real cell (or -1)
    cmap = np.full(n + 2 * p, -1, int)
    cmap[p:p + n] = np.arange(n)
    for m in range(p):
        if lo == "mirror":
            cmap[p - 1 - m] = min(m, n - 1)
        if hi == "mirror":
            cmap[p + n + m] = max(n - 1 - m, 0)
    nf = n + 1
    ie = np.arange(nf) + p                                   # extended node index of face i
    start = ie - p                                           # first stencil cell (extended index)
    if lo == "onesided":
        start = np.maximum(start, p)
    if hi == "onesided":
        start = np.minimum(start, p + n - 2 * p)
    start = np.clip(start, 0, n + 2 * p - 2 * p)
    cells = start[:, None] + np.arange(2 * p)[None, :]      # (nf, 2p) extended cell indices
    nodes = start[:, None] + np.arange(2 * p + 1)[None, :]  # (nf, 2p+1) extended node indices
    if recon == "fv6":
        Ld = _lagr_deriv_rows(sfe[nodes], sf)                # L'_m(s_face) over the 7 nodes
        tail = np.cumsum(Ld[:, ::-1], axis=1)[:, ::-1]       # sum_{m >= j} L'_m
        he = np.diff(sfe)                                    # extended cell lengths
        rw = he[cells] * tail[:, 1:]                         # w_k = h_k sum_{m >= k+1} L'_m
        rc = cmap[cells]
    elif recon == "fv6t":
        # two-cell distance interpolation plus the telescoping correction (C_{k+1} - C_k)/ds_e, C_m = -(h_ref(m)^2/6)
        # (1.4 D1_m - 0.4 D2_m) with D1, D2 the centred two-point derivative stencils over m +- 1 and m +- 2 (unit
        # first moment, so C_m f = -(h_ref^2/6) f' for linear f and the correction vanishes where h_ref is constant;
        # the telescoping form adds nothing to the dual sums sum_e w_{e,q} ds_e, so the dual keeps the two-cell
        # property); on a uniform line this is exactly (1, -8, 37, 37, -8, 1)/60. h_ref(m): geometric mean of the cell
        # lengths over m +- 3, so the correction is the FV6 one in uniform and smoothly stretched zones and its
        # linear-exactness defect (h_ref(k+1)^2 - h_ref(k)^2) f'/(6 ds) is a few per cent of the local flux at a jump.
        # Near a one-sided end a missing neighbour reduces D to the available centred / one-sided two-point stencil.
        # Known limit (S1): the two-cell weights give sum_e w_{e,q} ds_e = (h_{q-1} + h_{q+1})/2 instead of h_q, so at
        # a spacing jump the transposed (dual) operator is wrong by O(1) in the two adjacent cells (bounded; the
        # production K is O(1/h) there). No local repair exists in this class: the residual pair (+r, -r) carries a
        # first moment that zero-mass zero-moment corrections cannot supply; the exact repair is the double integral
        # of the residual along the line (nonlocal, weights ~2 in the hull region: tried and rejected).
        ne = n + 2 * p
        he = np.diff(sfe)
        lh = np.log(np.maximum(he, 1e-300))
        href = np.array([np.exp(lh[max(0, m - 3):min(ne, m + 4)].mean()) for m in range(ne)])
        rc = np.full((nf, 2 * p), -1, int)
        rw = np.zeros((nf, 2 * p))

        def dstencil(m, half):                               # unit-moment derivative at extended cell m, reach half
            lo_, hi_ = m - half, m + half
            if lo_ < 0 or hi_ >= ne or cmap[lo_] < 0 or cmap[hi_] < 0:
                lo_, hi_ = m - 1, m + 1
                if lo_ < 0 or cmap[lo_] < 0:
                    lo_ = m
                if hi_ >= ne or cmap[hi_] < 0:
                    hi_ = m
                if lo_ == hi_:
                    return {}
            d = sce[hi_] - sce[lo_]
            return {hi_: 1.0 / d, lo_: -1.0 / d}

        def Cfun(m):
            out = {}
            for coef, half in ((1.4, 1), (-0.4, 2)):
                for kq, v in dstencil(m, half).items():
                    out[kq] = out.get(kq, 0.0) + coef * v
            return {kq: -(href[m] * href[m] / 6.0) * v for kq, v in out.items()}
        for i in range(nf):
            kl, kr = ie[i] - 1, ie[i]
            if kl < 0 or kr >= ne or cmap[kl] < 0 or cmap[kr] < 0:
                continue                                     # boundary face: f_e = 0 (not used)
            ds = sce[kr] - sce[kl]
            wmap = {kl: (sce[kr] - sf[i]) / ds, kr: (sf[i] - sce[kl]) / ds}
            for kq, v in Cfun(kr).items():
                wmap[kq] = wmap.get(kq, 0.0) + v / ds
            for kq, v in Cfun(kl).items():
                wmap[kq] = wmap.get(kq, 0.0) - v / ds
            ks = sorted(wmap)
            for c_, kq in enumerate(ks[:2 * p]):
                rc[i, c_] = cmap[kq]
                rw[i, c_] = wmap[kq]
    else:
        scL, scR = sce[cells[:, 0]], sce[cells[:, 1]]
        wl = (scR - sf) / (scR - scL)
        rw = np.stack([wl, 1.0 - wl], 1)
        rc = cmap[cells]
    # third difference across face i: cells ie-2, ie-1 | ie, ie+1 (extended) with weights (-1, 3, -3, 1)
    d3c = np.full((nf, 4), -1, int)
    d3w = np.zeros((nf, 4))
    for i in range(nf):
        ke = np.array([ie[i] - 2, ie[i] - 1, ie[i], ie[i] + 1])
        ok = (ke >= 0) & (ke < n + 2 * p)
        cc = np.where(ok, cmap[np.clip(ke, 0, n + 2 * p - 1)], -1)
        ww = np.array([-1.0, 3.0, -3.0, 1.0])
        # a one-sided (outflow) end that leaves a needed cell undefined: reduce to the first difference across the face
        miss_lo = lo == "onesided" and ie[i] - 2 < p
        miss_hi = hi == "onesided" and ie[i] + 1 >= p + n
        if miss_lo or miss_hi:
            cc = np.array([-1, cmap[ie[i] - 1] if ie[i] - 1 >= 0 else -1, cmap[ie[i]] if ie[i] < n + 2 * p else -1, -1])
            ww = np.array([0.0, 1.0, -1.0, 0.0])
            if cc[1] < 0 or cc[2] < 0:
                cc[:] = -1
                ww[:] = 0.0
        d3c[i], d3w[i] = cc, ww
    return rc, rw, d3c, d3w


def fs_faces(G, sgn):
    """Faces of the structured grid G for the edge-flux operators. Returns dict with, per face: the two endpoints
    p0, p1 (nF, 3) on z = 0, the cells l (outward normal from l) and r (-1 = boundary), the direction d (0: i-face,
    constant x; 1: j-face along a band/lattice line), the line index and the position along the line, and a mask
    `zero` of the faces with F = 0 (hull wall and centre plane: the base flow is tangent / symmetric). Also the
    chord coordinates of faces and centroids per line (for the reconstruction) and the end modes."""
    nx, ny = G["shape"]
    Xn, Yn = fs_nodes(G)
    Xc, Yc = G["Xc"], G["Yc"]
    idx = np.arange(nx * ny).reshape(ny, nx).T               # panel index k = j nx + i -> idx[i, j]
    recs = []
    # i-faces: node column i (0..nx), row j (0..ny-1): from (i, j) to (i, j+1); l = (i-1, j), r = (i, j)
    I, Jf = np.meshgrid(np.arange(nx + 1), np.arange(ny), indexing="ij")
    p0 = np.stack([Xn[I, Jf], Yn[I, Jf], np.zeros_like(Xn[I, Jf])], -1).reshape(-1, 3)
    p1 = np.stack([Xn[I, Jf + 1], Yn[I, Jf + 1], np.zeros_like(Xn[I, Jf])], -1).reshape(-1, 3)
    l = np.where(I >= 1, idx[np.clip(I - 1, 0, nx - 1), Jf], -1).reshape(-1)
    r = np.where(I <= nx - 1, idx[np.clip(I, 0, nx - 1), Jf], -1).reshape(-1)
    recs.append(dict(p0=p0, p1=p1, l=l, r=r, d=np.zeros(len(l), int), line=Jf.reshape(-1), pos=I.reshape(-1),
                     zero=np.zeros(len(l), bool)))
    # j-faces: node row j (0..ny), column i (0..nx-1): from (i+1, j) to (i, j) so that nu |t| = (t_y, -t_x) points to
    # +y (outward from the lower cell); l = (i, j-1), r = (i, j)
    I, Jf = np.meshgrid(np.arange(nx), np.arange(ny + 1), indexing="ij")
    p0 = np.stack([Xn[I + 1, Jf], Yn[I + 1, Jf], np.zeros_like(Xn[I, Jf])], -1).reshape(-1, 3)
    p1 = np.stack([Xn[I, Jf], Yn[I, Jf], np.zeros_like(Xn[I, Jf])], -1).reshape(-1, 3)
    l = np.where(Jf >= 1, idx[I, np.clip(Jf - 1, 0, ny - 1)], -1).reshape(-1)
    r = np.where(Jf <= ny - 1, idx[I, np.clip(Jf, 0, ny - 1)], -1).reshape(-1)
    recs.append(dict(p0=p0, p1=p1, l=l, r=r, d=np.ones(len(l), int), line=I.reshape(-1), pos=Jf.reshape(-1),
                     zero=(Jf == 0).reshape(-1)))             # j = 0: hull wall (hull columns) or centre plane
    F = {k: np.concatenate([q[k] for q in recs]) for k in recs[0]}
    # chord coordinates along the i-lines (fixed j) and j-lines (fixed i): centroid -> face midpoint -> centroid
    mid = 0.5 * (F["p0"] + F["p1"])
    mi = mid[F["d"] == 0].reshape(nx + 1, ny, 3)              # i-face midpoints [i, j]
    mj = mid[F["d"] == 1].reshape(nx, ny + 1, 3)              # j-face midpoints [i, j]
    C = np.stack([Xc, Yc, np.zeros_like(Xc)], -1)             # centroids [i, j]
    def chords(m, c):                                         # m (n+1, L, 3) faces, c (n, L, 3) cells along axis 0
        a = np.linalg.norm(c - m[:-1], axis=-1)               # centroid k - face k
        b = np.linalg.norm(m[1:] - c, axis=-1)                # face k+1 - centroid k
        h = a + b
        sf = np.concatenate([np.zeros((1,) + h.shape[1:]), np.cumsum(h, axis=0)], 0)
        sc = sf[:-1] + a
        return sf, sc
    sfi, sci = chords(mi, C)                                   # (nx+1, ny), (nx, ny)
    sfj, scj = chords(np.swapaxes(mj, 0, 1), np.swapaxes(C, 0, 1))   # (ny+1, nx), (ny, nx)
    # end closures independent of the flow direction (so that M(-U) = -M(U) exactly and the pair is reciprocal):
    # domain-boundary faces carry the undisturbed far-field value f_e = 0 (they lie inside the beaches), the interior
    # faces next to an end use one-sided (shifted) stencils of real cells, j = 0 is the wall / centre plane (mirror)
    ends_i = ("onesided", "onesided")
    ends_j = ("mirror", "onesided")
    F.update(shape=(nx, ny), sfi=sfi, sci=sci, sfj=sfj, scj=scj, ends_i=ends_i, ends_j=ends_j, mid=mid)
    return F


def fs_operators_flux(G, Wfun, sgn, recon="fv6t", gauss=2, c_diss=FLUX_DISS, area=None):
    """Edge-flux (finite-volume) advective and kinematic free-surface operators on grid G (block comment above):
    Wfun(P) -> (n, 2) base velocity (x, y) at points P (n, 3) on z = 0 (DoubleBodyFlow.field_velocity or a closed
    form). Returns (A, K, info) with info = dict(divW (per panel), F (per face), faces, Wmax (max |W| at the Gauss
    points), kinematic="flux")."""
    nx, ny = G["shape"]
    N = nx * ny
    a = G["fs"].area if area is None else np.asarray(area, float)
    fc = fs_faces(G, sgn)
    nF = len(fc["l"])
    p0, p1 = fc["p0"], fc["p1"]
    t = p1 - p0
    need = ~fc["zero"]
    # face fluxes by Gauss quadrature along the face: int W.nu dl = sum_g w_g (W_g . (t_y, -t_x))  (nu |t| = (t_y, -t_x),
    # outward from l for the orientation of fs_faces: i-faces +x, j-faces toward +y)
    if gauss == 2:
        gp, gw = (0.5 * (1 - 1 / np.sqrt(3.0)), 0.5 * (1 + 1 / np.sqrt(3.0))), (0.5, 0.5)
    elif gauss == 1:
        gp, gw = (0.5,), (1.0,)
    else:
        raise ValueError("gauss must be 1 or 2")
    Fe = np.zeros(nF)
    Wmax, Wmax_at, Wg_all = 0.0, None, []
    for q, w in zip(gp, gw):
        P = p0[need] + q * t[need]
        Wg = np.asarray(Wfun(P), float)
        if len(Wg):
            mg = np.linalg.norm(Wg, axis=1)
            Wg_all.append(mg)
            if mg.max() > Wmax:
                Wmax, Wmax_at = float(mg.max()), P[int(np.argmax(mg))].copy()
        Fe[need] += w * (Wg[:, 0] * t[need][:, 1] - Wg[:, 1] * t[need][:, 0])
    Wg_all = np.concatenate(Wg_all) if Wg_all else np.zeros(0)
    l, r = fc["l"], fc["r"]
    # divergence: a div W = sum_e F_e (outward); the operators use the interior-face part (the boundary faces carry
    # f_e = 0), so that K 1 = div_int W and A 1 = 0 hold exactly in every cell; div_full is information
    inter = (l >= 0) & (r >= 0)
    divWa = np.zeros(N)
    np.add.at(divWa, l[inter], Fe[inter])
    np.add.at(divWa, r[inter], -Fe[inter])
    divfull = np.zeros(N)
    np.add.at(divfull, l[l >= 0], Fe[l >= 0])
    np.add.at(divfull, r[r >= 0], -Fe[r >= 0])
    # reconstruction and third-difference stencils per line
    rows, cols, vals = [], [], []            # M = Wa K_c
    drow, dcol, dval = [], [], []            # D3 (nF x N)
    idx = np.arange(N).reshape(ny, nx).T
    sfi, sci, sfj, scj = fc["sfi"], fc["sci"], fc["sfj"], fc["scj"]
    d_arr, line_arr, pos_arr = fc["d"], fc["line"], fc["pos"]
    face_of = {}
    for e in range(nF):
        face_of[(int(d_arr[e]), int(line_arr[e]), int(pos_arr[e]))] = e
    for d, nlines in ((0, ny), (1, nx)):
        for L_ in range(nlines):
            if d == 0:
                sf, sc, lo, hi = sfi[:, L_], sci[:, L_], fc["ends_i"][0], fc["ends_i"][1]
                cell_of = idx[:, L_]
            else:
                sf, sc, lo, hi = sfj[:, L_], scj[:, L_], fc["ends_j"][0], fc["ends_j"][1]
                cell_of = idx[L_, :]
            rc, rw, d3c, d3w = _line_recon(sf, sc, lo, hi, recon)
            n1 = len(sf)
            es = np.array([face_of[(d, L_, i)] for i in range(n1)])
            for i in range(n1):
                e = es[i]
                if fc["zero"][e] or l[e] < 0 or r[e] < 0:      # wall / centre plane (F = 0) or domain boundary (f_e = 0)
                    continue
                ok = rc[i] >= 0
                cq = cell_of[rc[i][ok]]
                wq = rw[i][ok]
                if l[e] >= 0:
                    rows.append(np.full(len(cq), l[e])); cols.append(cq); vals.append(Fe[e] * wq)
                if r[e] >= 0:
                    rows.append(np.full(len(cq), r[e])); cols.append(cq); vals.append(-Fe[e] * wq)
                ok3 = d3c[i] >= 0
                if np.any(ok3):
                    drow.append(np.full(int(ok3.sum()), e)); dcol.append(cell_of[d3c[i][ok3]]); dval.append(d3w[i][ok3])
    M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
    M.sum_duplicates()
    D3 = sp.csr_matrix((np.concatenate(dval), (np.concatenate(drow), np.concatenate(dcol))), shape=(nF, N))
    D3.sum_duplicates()
    S = 0.5 * (M - M.T)
    WaV = (D3.T @ sp.diags(c_diss * np.abs(Fe)) @ D3).tocsr()
    Dg = sp.diags(divWa)
    ia = sp.diags(1.0 / a)
    A = (ia @ (S - 0.5 * Dg + WaV)).tocsr()
    K = (ia @ (S + 0.5 * Dg + WaV)).tocsr()
    A.sum_duplicates(); K.sum_duplicates()
    info = dict(divW=divWa / a, divW_full=divfull / a, F=Fe, faces=fc, Wmax=Wmax, Wmax_at=Wmax_at, Wgauss=Wg_all,
                kinematic="flux", recon=recon, gauss=gauss, WaV=WaV, S=S, M=M)
    return A, K, info


class NearWallSampler:
    """Opt-in near-hull sampling rule for the base flow on z = 0 (S1 stage 2, item 2b; derivation in
    studies/_cache/derive/s1b_progress_zh.md section 2.2). The panel representation of the double-body potential is
    a layer of piecewise-constant sources and dipoles; its velocity field differs from the exact smooth layer's by a
    zero-mean pattern with the period of the panel spacing a, which decays like exp(-2 pi d/a) with the distance d
    from the surface and has edge (log, 1/d x density jump) singularities at the panel edges. A free-surface sampling
    point closer than kappa a to the hull's top edge (the waterline, on z = 0) is therefore moved along the in-plane
    normal from its foot point on the nearest top-edge segment out to the distance kappa a (a = that segment's
    length); points farther away are unchanged. The sampled field stays a linear, odd function of U (the point is
    purely geometric), so every structural property of the edge-flux pair is untouched: F_e(-U) = -F_e(U), hence
    Wa K_U = (Wa A_{-U})^T exactly, K 1 = div_int W of the SAMPLED field exactly, S skew, Wa V >= 0 (passivity of
    the operator pair). What changes is the consistency error: |W(P') - W(P)| <= |grad W| kappa a = O(a) where the
    base flow is smooth, and where it is singular (a corner of the panelled hull) the sampled magnitude is capped at
    the value one panel size away -- the same regularisation the panel method applies to the hull itself.
    `moved`, `shift_max` and `shift_mean` record what the rule did on the last call."""

    def __init__(self, body, hull_shape, kappa=0.5):
        nx, nz = hull_shape
        raw = body.raw.reshape(nx, nz, 4, 3)[:, -1]            # top-row panels
        nrm = body.n.reshape(nx, nz, 3)[:, -1]                  # their normals (out of the body)
        A, B, N = [], [], []
        for i in range(nx):
            pts = raw[i][np.abs(raw[i][:, 2]) <= 1e-12 * max(1.0, float(np.abs(raw[i]).max()))]
            if len(pts) >= 2:
                u = np.unique(np.round(pts[:, :2], 14), axis=0)
                if len(u) >= 2 and np.linalg.norm(u[-1] - u[0]) > 0:
                    e = u[-1] - u[0]
                    nu = np.array([-e[1], e[0]])                # a normal of the segment in the plane z = 0
                    if np.dot(nu, nrm[i, :2]) < 0:              # oriented like the panel normal: OUT of the hull
                        nu = -nu
                    A.append(u[0])
                    B.append(u[-1])
                    N.append(nu / np.linalg.norm(nu))
        self.A, self.B, self.N = np.asarray(A, float), np.asarray(B, float), np.asarray(N, float)
        self.E = self.B - self.A
        self.EE = np.maximum(np.einsum("ij,ij->i", self.E, self.E), 1e-300)
        self.len = np.sqrt(self.EE)
        self.kappa = float(kappa)
        self.moved = 0
        self.inside = 0
        self.shift_max = 0.0
        self.shift_mean = 0.0

    def __call__(self, P):
        """Points P (n, 3) on z = 0 -> sampling points (n, 3). A point whose SIGNED distance from the nearest
        top-edge segment (positive on the fluid side of that segment's outward normal) is below kappa x the segment
        length is placed on the outward normal through its foot point at exactly that distance (this also lifts
        points that lie inside the hull, where a straight band chord cuts a convex waterline)."""
        P = np.asarray(P, float)
        Q = P.copy()
        if len(self.A) == 0 or self.kappa <= 0:
            return Q
        xy = P[:, :2]
        best_k = np.zeros(len(P), int)
        best_f = np.zeros((len(P), 2))
        for r0 in range(0, len(P), 4096):
            r1 = min(len(P), r0 + 4096)
            X = xy[r0:r1]
            t = np.einsum("qkd,kd->qk", X[:, None, :] - self.A[None], self.E) / self.EE[None]
            t = np.clip(t, 0.0, 1.0)
            F = self.A[None] + t[..., None] * self.E[None]            # foot points (q, k, 2)
            d = np.linalg.norm(X[:, None, :] - F, axis=2)
            k = np.argmin(d, axis=1)
            best_k[r0:r1] = k
            best_f[r0:r1] = F[np.arange(r1 - r0), k]
        a = self.len[best_k]
        nu = self.N[best_k]
        signed = np.einsum("qd,qd->q", xy - best_f, nu)
        dist = np.linalg.norm(xy - best_f, axis=1)
        target = self.kappa * a
        # near = the TRUE distance to the segment (foot point clamped to the segment) below kappa a, or inside the hull
        # (negative normal component). The normal component alone is not a distance: beyond a segment's end a point
        # 30 cm away along the segment direction has a small normal component and would be dragged to the hull.
        near = (dist < target) | (signed < 0)
        self.moved = int(near.sum())
        self.inside = int((signed < 0).sum())
        if self.moved:
            Q[near, :2] = best_f[near] + target[near][:, None] * nu[near]
            sh = np.linalg.norm(Q[near, :2] - xy[near], axis=1)
            self.shift_max, self.shift_mean = float(sh.max()), float(sh.mean())
        else:
            self.shift_max = self.shift_mean = 0.0
        return Q


# ===================================================================== LTI system
class DBForwardSpeedLTI(ForwardSpeedLTI):
    """The forward-speed LTI system of unsteady.ForwardSpeedLTI (same representation, blocks and frequency
    elimination) with the double-body free-surface conditions (module docstring 2): N_phi = diag(nu) + A
    (dynamic), N_zeta = diag(nu) + K (kinematic):
        zeta' = -N_zeta zeta - sigma + f_k,    phi_F' = -N_phi phi_F - g zeta + f_d,
    frequency domain [I - (1/g)(i w + N_zeta)(i w + N_phi) R_s] sigma = (1/g)(i w + N_zeta)(i w + N_phi) R_n phi_n
    + f_k - (1/g)(i w + N_zeta) f_d. No hull traces (the hull wall is characteristic, docstring 3)."""

    def __init__(self, body, fs, A, K, nu, U, g=9.81, rho=1.0, device=None, hull_shape=None, keep_blocks=False):
        self._Kop = sp.csr_matrix(K)
        super().__init__(body, fs, A, nu, U, g, rho, device, hull_shape, keep_blocks, wall=None)

    def set_operator(self, D, nu):
        """D: the advective operator A (kept under the B5 attribute name D)."""
        self.D = sp.csr_matrix(D)
        self.nu = np.asarray(nu, float)
        self.Nsp = sp.csr_matrix(sp.diags(self.nu) + self.D)            # N_phi (dynamic)
        self.Nzsp = sp.csr_matrix(sp.diags(self.nu) + self._Kop)        # N_zeta (kinematic)
        self._N = _ts(self.Nsp, self.dev)
        self._Nz = _ts(self.Nzsp, self.dev)
        NpR = torch.sparse.mm(self._N, self.Rs)
        NzR = torch.sparse.mm(self._Nz, self.Rs)
        self.P1 = 0.5 * (NpR + NzR)
        self.P2 = torch.sparse.mm(self._Nz, NpR)
        del NpR, NzR
        self._luRs = None
        self._wall_on = False

    def _Nzapply(self, v):
        if torch.is_complex(v):
            return torch.complex(torch.sparse.mm(self._Nz, v.real.contiguous()),
                                 torch.sparse.mm(self._Nz, v.imag.contiguous()))
        return torch.sparse.mm(self._Nz, v)

    def solve_freq(self, omega, phin, fd=None, fk=None):
        """phin: (NB, m) body normal velocities; fd, fk: optional free-surface forcing (NF, m) of the dynamic and
        kinematic conditions (module docstring 2). Returns dict(phi_B, phi_F, sigma) (numpy complex)."""
        dev, g = self.dev, self.g
        pn = torch.as_tensor(np.asarray(phin, complex).reshape(self.NB, -1), dtype=torch.complex128, device=dev)
        m = pn.shape[1]

        def rmul(Mt, x):
            return torch.complex(Mt @ x.real.contiguous(), Mt @ x.imag.contiguous())
        v = rmul(self.Rn, pn)
        n1 = self.Napply(v)
        rhs = (-omega * omega * v + 1j * omega * (n1 + self._Nzapply(v)) + self._Nzapply(n1)) / g
        if fd is not None:
            f = torch.as_tensor(np.asarray(fd, complex).reshape(self.NF, m), dtype=torch.complex128, device=dev)
            rhs = rhs - (1j * omega * f + self._Nzapply(f)) / g
        if fk is not None:
            rhs = rhs + torch.as_tensor(np.asarray(fk, complex).reshape(self.NF, m), dtype=torch.complex128, device=dev)
        A = self.system(omega)
        sig = torch.linalg.solve(A, rhs)
        del A
        return self._finish(sig, pn, v)

    def zeta_of(self, omega, phiF, fd=None):
        """Free-surface elevation of a frequency-domain solution: zeta = (f_d - (i w + N_phi) phi_F)/g."""
        phiF = np.asarray(phiF, complex)
        out = -(1j * omega * phiF + self.Nsp @ phiF)
        if fd is not None:
            out = out + np.asarray(fd, complex).reshape(phiF.shape)
        return out / self.g

    def state_matrix(self):
        """Dense M of y' = M y, y = (zeta, phi_F), phi_n = 0: zeta' = -N_zeta zeta - R_s^-1 phi_F,
        phi_F' = -g zeta - N_phi phi_F (spectrum check)."""
        dev, g, NF = self.dev, self.g, self.NF
        Ri = torch.linalg.inv(self.Rs)
        M = torch.zeros((2 * NF, 2 * NF), dtype=torch.float64, device=dev)
        M[:NF, :NF] = -torch.as_tensor(self.Nzsp.toarray(), device=dev)
        M[:NF, NF:] = -Ri
        M[NF:, :NF] = -g * torch.eye(NF, dtype=torch.float64, device=dev)
        M[NF:, NF:] = -torch.as_tensor(self.Nsp.toarray(), device=dev)
        out = M.cpu().numpy()
        del M, Ri
        return out

    def rhs_td(self, zeta, phiF, pn, pnd=None, fd=None, fk=None):
        sig = self.sigma_of(phiF, pn)
        dz = -torch.sparse.mm(self._Nz, zeta) - sig
        dp = -torch.sparse.mm(self._N, phiF) - self.g * zeta
        if fk is not None:
            dz = dz + fk
        if fd is not None:
            dp = dp + fd
        return dz, dp, sig

    def release(self):
        super().release()
        self._Nz = None


# ===================================================================== coefficients
class DBCoefficients(Coefficients):
    """Radiation, excitation and limits on a DBForwardSpeedLTI with the double-body m-terms (module docstring
    4). D-form: p = -rho (phi_t + W.grad phi) with W.grad = the discrete operator G of the base flow; OT form
    with the m-terms of DoubleBodyFlow.mterms(method) ("sbp": the adjoint of G, so the two forms coincide
    discretely; "pointwise": information)."""

    def __init__(self, solver, db, ref=(0.0, 0.0), modes=(1, 3, 5), mterms="sbp"):
        self.s = solver
        self.db = db
        self.modes = tuple(modes)
        self.ref = ref
        self.mterm_method = mterms
        self.mv = db.mterms(ref, self.modes, method=mterms)

    def _coeffs(self, omega, sol, ph, want_phi=False, d_form=True):
        s = self.s
        a = s.body.area
        m = len(self.modes)
        T = np.zeros((m, m), complex)
        Tot = np.zeros((m, m), complex)
        wg = []
        for c in range(m):
            phk = sol["phi_B"][:, c]
            gk = self.db.wgrad(phk)
            wg.append(gk)
            for r, j in enumerate(self.modes):
                nj, mj = self.mv[j]
                T[r, c] = -s.rho * 2 * np.sum((1j * omega * phk + gk) * nj * a)
                Tot[r, c] = -s.rho * 2 * np.sum(phk * (1j * omega * nj - mj) * a)
        out = dict(omega=omega, T=T, T_ot=Tot, A=-(T.real) / omega ** 2 if omega else None,
                   B=T.imag / omega if omega else None)
        if want_phi:
            out.update(sol=sol, phin=ph, wgrad=np.stack(wg, 1))
        return out

    def instantaneous(self):
        """A_inf, B_inf, C_inf of the discrete system (phi_F = 0 as omega -> infinity, phi_B -> i w phi_inf[n_k] +
        phi_inf[m_k]); D-form with G, OT form with the m-terms (stage note 10.2 item 5 with -U d_x -> W.grad)."""
        s = self.s
        a = s.body.area
        m = len(self.modes)
        nmat = np.stack([self.mv[k][0] for k in self.modes], 1)
        mmat = np.stack([self.mv[k][1] for k in self.modes], 1)
        sn = s.solve_inf(nmat)["phi_B"]
        sm = s.solve_inf(mmat)["phi_B"] if s.U != 0 else np.zeros_like(sn)
        gn = np.stack([self.db.wgrad(sn[:, c]) for c in range(m)], 1)
        gm = np.stack([self.db.wgrad(sm[:, c]) for c in range(m)], 1)
        rho = s.rho
        A = np.zeros((m, m))
        B = np.zeros((m, m))
        C = np.zeros((m, m))
        Bot = np.zeros((m, m))
        Cot = np.zeros((m, m))
        for r in range(m):
            nj, mj = nmat[:, r], mmat[:, r]
            for c in range(m):
                A[r, c] = -rho * 2 * np.sum(sn[:, c] * nj * a)
                B[r, c] = -rho * 2 * np.sum((sm[:, c] + gn[:, c]) * nj * a)
                C[r, c] = -rho * 2 * np.sum(gm[:, c] * nj * a)
                Bot[r, c] = -rho * 2 * np.sum((sm[:, c] * nj - sn[:, c] * mj) * a)
                Cot[r, c] = rho * 2 * np.sum(sm[:, c] * mj * a)
        return dict(A=A, B=B, C=C, B_ot=Bot, C_ot=Cot)

    def static_stiffness(self):
        """C0 = T(omega = 0) with the displacement (m-term) body condition: D-form -rho int (W.grad phi_k) n_j,
        OT form rho int phi_k m_j. Returns (C0_D, C0_OT)."""
        s = self.s
        a = s.body.area
        m = len(self.modes)
        if s.U == 0:
            return np.zeros((m, m)), np.zeros((m, m))
        ph = np.stack([self.mv[k][1] for k in self.modes], 1).astype(complex)
        sol = s.solve_freq(0.0, ph)
        C = np.zeros((m, m))
        Cot = np.zeros((m, m))
        for c in range(m):
            phk = sol["phi_B"][:, c].real
            gk = self.db.wgrad(phk)
            for r, j in enumerate(self.modes):
                nj, mj = self.mv[j]
                C[r, c] = -s.rho * 2 * np.sum(gk * nj * a)
                Cot[r, c] = s.rho * 2 * np.sum(phk * mj * a)
        return C, Cot

    def incident_forcing(self, omega0, beta):
        """Free-surface forcing of the diffraction problem with the uniform-stream incident wave phi_I (module
        docstring 2): f_d = -w.grad phi_I, f_k = -div_h(w zeta_I) = -w.grad zeta_I + Phi_zz zeta_I, w = W + U x_hat,
        zeta_I = -(1/g)(i w_e phi_I - U phi_I,x) (= the incident elevation), at the free-surface centroids."""
        s = self.s
        g = s.g
        k = omega0 ** 2 / g
        we = omega0 - s.U * k * np.cos(beta)
        P = s.fs.c
        Wf, phizz = self.fs_base()
        w = Wf + np.array([s.U, 0.0])[None, :]
        phI, _, gxI = incident(P, np.tile([0.0, 0.0, 1.0], (len(P), 1)), omega0, beta, g)
        # y-even part: phi_I = c0 e^{kz - i k x cos b} cos(k y sin b)
        cb, sb = np.cos(beta), np.sin(beta)
        c0 = 1j * g / omega0
        ex = np.exp(k * P[:, 2] - 1j * k * P[:, 0] * cb)
        gyI = -c0 * ex * k * sb * np.sin(k * P[:, 1] * sb)
        zI = -(1j * we * phI - s.U * gxI) / g
        # grad zeta_I: zeta_I is a combination of phi_I-like functions: d/dx -> -i k cos b, d/dy of cos(k y sin b)
        zx = -1j * k * cb * zI
        zy = -(1j * we - s.U * (-1j * k * cb)) / g * gyI
        fd = -(w[:, 0] * gxI + w[:, 1] * gyI)
        fk = -(w[:, 0] * zx + w[:, 1] * zy) + phizz * zI
        return fd, fk, zI, we

    def fs_base(self):
        """Base velocity and Phi_zz at the free-surface centroids (cached on the solver)."""
        s = self.s
        if getattr(s, "_fsbase", None) is None:
            s._fsbase = (self.db.field_velocity(s.fs.c), self.db.field_phizz(s.fs.c))
        return s._fsbase

    def rad_exc(self, omega0, beta, forcing=True):
        """Radiation at the encounter frequency and the excitation with ONE LU (radiation modes + diffraction
        right-hand sides). Excitation: X_j = rho int (i w_e + W.grad)(phi_I + phi_D) n_j (D-form, W.grad = G) and
        X_ot = rho int (phi_I + phi_D)(i w_e n_j - m_j) (equal with SBP m-terms); X_fk = rho int i w0 phi_I n_j
        (classical Froude-Krylov, information). forcing=False: diffraction without the incident-wave/base-flow
        free-surface forcing (information)."""
        s = self.s
        g = s.g
        k = omega0 ** 2 / g
        we = omega0 - s.U * k * np.cos(beta)
        a = s.body.area
        m = len(self.modes)
        ph = np.stack([1j * we * self.mv[kk][0] + self.mv[kk][1] for kk in self.modes], 1)
        phI, dnI, _ = incident(s.body.c, s.body.n, omega0, beta, g)
        NFm = s.NF
        fd = np.zeros((NFm, m + 1), complex)
        fk = np.zeros((NFm, m + 1), complex)
        if forcing and s.U != 0:
            fdi, fki, _, _ = self.incident_forcing(omega0, beta)
            fd[:, m], fk[:, m] = fdi, fki
        sol = s.solve_freq(we, np.concatenate([ph, (-dnI)[:, None]], 1), fd=fd, fk=fk)
        rsol = dict(phi_B=sol["phi_B"][:, :m], phi_F=sol["phi_F"][:, :m], sigma=sol["sigma"][:, :m])
        rad = self._coeffs(we, rsol, ph)
        phD = sol["phi_B"][:, m]
        tot = phI + phD
        gt = self.db.wgrad(tot)
        X, Xot, Xfk, Xd, Xdot = [], [], [], [], []
        for j in self.modes:
            nj, mj = self.mv[j]
            X.append(s.rho * 2 * np.sum((1j * we * tot + gt) * nj * a))
            Xot.append(s.rho * 2 * np.sum(tot * (1j * we * nj - mj) * a))
            Xfk.append(s.rho * 2 * np.sum(1j * omega0 * phI * nj * a))
            Xd.append(s.rho * 2 * np.sum((1j * we * phD + self.db.wgrad(phD)) * nj * a))
            Xdot.append(s.rho * 2 * np.sum(phD * (1j * we * nj - mj) * a))
        exc = dict(omega0=omega0, omega_e=we, k=k, beta=beta, X=np.array(X), X_ot=np.array(Xot), X_fk=np.array(Xfk),
                   X_d=np.array(Xd), X_d_ot=np.array(Xdot), sol=sol, phi_I=phI, dn_I=dnI, fd=fd[:, m], fk=fk[:, m])
        return rad, exc

    def excitation(self, omega0, beta, want_phi=False, forcing=True):
        """Excitation only (same definitions as rad_exc)."""
        s = self.s
        g = s.g
        k = omega0 ** 2 / g
        we = omega0 - s.U * k * np.cos(beta)
        a = s.body.area
        phI, dnI, _ = incident(s.body.c, s.body.n, omega0, beta, g)
        fd = fk = None
        if forcing and s.U != 0:
            fdi, fki, _, _ = self.incident_forcing(omega0, beta)
            fd, fk = fdi[:, None], fki[:, None]
        sol = s.solve_freq(we, (-dnI)[:, None], fd=fd, fk=fk)
        phD = sol["phi_B"][:, 0]
        tot = phI + phD
        gt = self.db.wgrad(tot)
        X, Xot, Xfk, Xd, Xdot = [], [], [], [], []
        for j in self.modes:
            nj, mj = self.mv[j]
            X.append(s.rho * 2 * np.sum((1j * we * tot + gt) * nj * a))
            Xot.append(s.rho * 2 * np.sum(tot * (1j * we * nj - mj) * a))
            Xfk.append(s.rho * 2 * np.sum(1j * omega0 * phI * nj * a))
            Xd.append(s.rho * 2 * np.sum((1j * we * phD + self.db.wgrad(phD)) * nj * a))
            Xdot.append(s.rho * 2 * np.sum(phD * (1j * we * nj - mj) * a))
        out = dict(omega0=omega0, omega_e=we, k=k, beta=beta, X=np.array(X), X_ot=np.array(Xot), X_fk=np.array(Xfk),
                   X_d=np.array(Xd), X_d_ot=np.array(Xdot))
        if want_phi:
            out.update(sol=sol, phi_I=phI, dn_I=dnI, fd=None if fd is None else fd[:, 0],
                       fk=None if fk is None else fk[:, 0])
        return out


def build_db_solver(hull, U, sink, trim, grid_kw, hull_mesh, g=9.81, rho=1.0, ref=(0.0, 0.0), device=None,
                    kinematic="cons", mterms="sbp", sgn=None, gop="fv", flux_kw=None):
    """One double-body LTI system at (U, sink, trim): hull panelled at the attitude (hull.mesh), free-surface grid
    unsteady.build_grid(hull, sink, trim, sign(U), **grid_kw) (U = 0: symmetric extents), base flow
    DoubleBodyFlow, operators fs_operators. ref: moments about the body-frame point (x_ref, z_ref) carried with the
    attitude (as tables.build_solver). kinematic "cons" (production) / "nat" (fs_operators) or "flux": the edge-flux
    pair fs_operators_flux (opt-in, S1 2026-10-09; flux_kw: recon, gauss, c_diss). Returns (solver, DBCoefficients,
    grid, base flow)."""
    sgn = (1.0 if U >= 0 else -1.0) if sgn is None else sgn
    body = hull.mesh(*hull_mesh, sink=sink, trim=trim)
    kw = dict(grid_kw)
    if U == 0 and "down" not in grid_kw:
        kw["down"] = kw.get("up", 1.5)
        kw["b_dn"] = kw.get("b_up", 0.75)
    G = build_grid(hull, sink, trim, sgn, **kw)
    db = DoubleBodyFlow(body, U, hull_mesh, device, gop=gop)
    # opt-in near-hull sampling rule (NearWallSampler; flux_kw["near_wall"] = dict(kappa=0.5, forcing=False)):
    # Gauss points of the edge fluxes (and, with forcing=True, the centroid values that feed the diffraction forcing
    # f_d, f_k and fs_base) closer than kappa x the local waterline panel length to the hull are sampled kappa a away
    fkw = dict(flux_kw or {})
    nw = fkw.pop("near_wall", None)
    mover = None
    nw_stats = None
    if nw:
        mover = NearWallSampler(body, hull_mesh, kappa=nw.get("kappa", 0.5))
    if U != 0:
        Pc = mover(G["fs"].c) if (mover is not None and nw.get("forcing", False)) else G["fs"].c
        Wf = db.field_velocity(Pc)
        phizz = db.field_phizz(Pc)
        if mover is not None and nw.get("forcing", False):
            nw_stats = dict(centroids_moved=mover.moved, centroids_inside=mover.inside, centroid_shift_max=mover.shift_max)
    else:
        Wf = np.zeros((len(G["fs"]), 2))
        phizz = np.zeros(len(G["fs"]))
    if kinematic == "flux":
        if U == 0:
            Wfun = lambda P: np.zeros((len(P), 2))  # noqa: E731
        elif mover is None:
            Wfun = db.field_velocity
        else:
            def Wfun(P):
                Q = mover(P)
                nw_stats_g.append(dict(moved=mover.moved, inside=mover.inside, shift_max=mover.shift_max,
                                       shift_mean=mover.shift_mean, n=len(P)))
                return db.field_velocity(Q)
            nw_stats_g = []
        A, K, info = fs_operators_flux(G, Wfun, sgn, **fkw)
        if mover is not None and U != 0:
            info["near_wall"] = dict(kappa=mover.kappa, gauss=nw_stats_g, centroids=nw_stats)
    else:
        A, K, info = fs_operators(G, Wf, sgn, kinematic, phizz)
    s = DBForwardSpeedLTI(body, G["fs"], A, K, G["nu"] * np.sqrt(g / hull.L), U, g, rho, device, hull_shape=hull_mesh)
    s._fsbase = (Wf, phizz)
    s._opinfo = info
    refE = hull.to_earth(np.array(ref[0]), np.array(0.0), np.array(ref[1]), sink, trim)
    co = DBCoefficients(s, db, ref=(float(refE[0]), float(refE[2])), mterms=mterms)
    return s, co, G, db


# ===================================================================== base-pressure stiffness (B6 review)
def attitude_of_modes(hull, sink, trim, x3=0.0, x5=0.0, ref=(0.0, 0.0)):
    """(sink, trim) of the hull after a heave x3 (up) and a pitch x5 (bow DOWN, about the body point ref =
    (x_r, z_r) carried with the attitude) from (sink, trim): trim' = trim - x5, sink' chosen so that the earth z of
    ref moves by x3 only (ref earth z = z_r cos(theta) + x_r sin(theta) - sink). The earth x of ref moves by
    O(x5 z_r) (a surge shift, immaterial for the rigid-image flow by translation invariance)."""
    xr, zr = ref
    th = trim - x5
    s = sink + zr * (np.cos(th) - np.cos(trim)) + xr * (np.sin(th) - np.sin(trim)) - x3
    return float(s), float(th)


def base_pressure_force(hull, U, sink, trim, hull_mesh, ref=(0.0, 0.0), rho=1.0, device=None, zpar=1, gop="fv"):
    """Generalised force F_j = -2 int_S rho f_0 n_j dS (j = 1, 3, 5 in this order; n_5 about the body point ref
    carried with the attitude; factor 2 for the half hull) of the base pressure rho f_0 = rho (U^2 - |W|^2)/2 of the
    rigid-image double-body flow about the hull panelled at (sink, trim); hydrostatics excluded. zpar = +1 rigid
    lid (production), -1 pressure-release image (experiment 3'). Returns (F (3,), db)."""
    body = hull.mesh(*hull_mesh, sink=sink, trim=trim)
    db = DoubleBodyFlow(body, U, hull_mesh, device, gop=gop, zpar=zpar)
    refE = hull.to_earth(np.array(ref[0]), np.array(0.0), np.array(ref[1]), sink, trim)
    nk = db.mode_normals(ref=(float(refE[0]), float(refE[2])))
    p = 0.5 * rho * (db.U ** 2 - np.einsum("ij,ij->i", db.W, db.W))
    F = np.array([-2.0 * np.sum(p * body.area * nk[k]) for k in (1, 3, 5)])
    return F, db


def base_pressure_stiffness(hull, U, sink, trim, hull_mesh, ref=(0.0, 0.0), rho=1.0, delta=1e-3, device=None,
                            zpar=1, gop="fv", mterms="sbp"):
    """Frozen-base-pressure displacement stiffness C_s of the double-body linearisation (B6 review, finding B3;
    stage note 12.2 item 4). The static force of a displaced hull in the double-body model has three parts: the
    hydrostatics (C_hs, plant side), the change of the base flow and of the waves with the displacement (the
    radiation problem's omega = 0 response to the m-term body condition: C0), and the motion of the hull surface
    through the FROZEN base-pressure field rho f_0 (pressure gradient along the displacement, rotation of the
    normal, change of the wetted strip at the waterline): C_s. In the uniform-stream linearisation f_0 = 0 and C_s
    = 0; here f_0 != 0. Computed as C_s = K_db - C0_lid: K_db = -dF_db/dq, the total attitude derivative of the
    rigid-image base-pressure force (base_pressure_force, hull re-panelled and flow re-solved at q +- delta e_k;
    central differences, delta in hull.L units for heave and rad for pitch; the surge column is zero by
    translation invariance), C0_lid = the omega = 0 rigid-image response to the m-terms (no free surface), D-form
    -rho int (W.grad phi_k) n_j -- the part of K_db that comes from the change of the base flow, so the difference
    is the frozen-field part. Modes (1, 3, 5): q = (x1, x3 heave up, x5 pitch bow down about ref). Returns
    dict(C_s, K_db, C0_lid, F0) (3 x 3 matrices, rows j, columns k)."""
    from .bie import neumann_green
    F0, db0 = base_pressure_force(hull, U, sink, trim, hull_mesh, ref, rho, device, zpar, gop)
    body = db0.body
    refE = hull.to_earth(np.array(ref[0]), np.array(0.0), np.array(ref[1]), sink, trim)
    mv = db0.mterms(ref=(float(refE[0]), float(refE[2])), modes=(1, 3, 5), method=mterms)
    op = ImageOperator(body, axes=(1, 2), device=db0.dev, parities=[(1, db0.zpar)])
    a = body.area
    C0l = np.zeros((3, 3))
    for c, k in enumerate((1, 3, 5)):
        phk = neumann_green(op, mv[k][1], (1, db0.zpar), device=db0.dev)
        gk = db0.wgrad(phk)
        for r, j in enumerate((1, 3, 5)):
            C0l[r, c] = -rho * 2 * np.sum(gk * mv[j][0] * a)
    del op
    K = np.zeros((3, 3))
    for c, (d3, d5) in ((1, (1.0, 0.0)), (2, (0.0, 1.0))):
        Fp = base_pressure_force(hull, U, *attitude_of_modes(hull, sink, trim, delta * d3, delta * d5, ref), hull_mesh,
                                 ref, rho, device, zpar, gop)[0]
        Fm = base_pressure_force(hull, U, *attitude_of_modes(hull, sink, trim, -delta * d3, -delta * d5, ref),
                                 hull_mesh, ref, rho, device, zpar, gop)[0]
        K[:, c] = -(Fp - Fm) / (2 * delta)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dict(C_s=K - C0l, K_db=K, C0_lid=C0l, F0=F0)


def frozen_pressure_stiffness(db, refE, rho=1.0):
    """C_s from the frozen-field formula (independent of the re-panelling of base_pressure_stiffness; B6 review
    item 4, stage note 12.2 item 4). For a rigid displacement xi_k of the hull (xi_1 = e_x, xi_3 = e_z, xi_5 = e_y x
    (r - r_ref), i.e. pitch bow down about ref) in the FROZEN pressure field p = rho f_0, F_j = -int_S(q) p n_j dS
    changes by
        dF_j = -int_S (xi_k . grad p) n_j dS - int_S p dn_j dS + oint_C p n_j xi_z / |n_h| dl,
    (surface moved through the field; normal rotated, dn = e_y x n for pitch, n_5 about the carried ref is
    rotation-invariant; the wetted strip of width xi_z/|n_h| leaves the water at the waterline C, |n_h| = the
    horizontal part of the hull normal there), so C_s,jk = -dF_j/dq_k. On S the base flow is tangent, so grad f_0 =
    -(W.grad)W = -(W.grad_S)W is a surface derivative (the discrete W.grad G of the base flow applied to W). The
    same formula with p = -rho g z gives the hydrostatic stiffness (check: C33 = rho g A_w). Waterline values by
    linear extrapolation from the top two rows; dl from the top edges of the top-row panels. db: DoubleBodyFlow;
    refE: (x, z) of the reference point in the earth frame. Returns C_s (3 x 3, modes 1, 3, 5)."""
    from .unsteady import hull_waterline_values
    body, W, U = db.body, db.W, db.U
    nx, nz = db.shape
    xr, zr = refE
    WW = np.stack([db.wgrad(W[:, c]) for c in range(3)], 1)                # (W.grad_S) W
    f0 = 0.5 * (U * U - np.einsum("ij,ij->i", W, W))
    n, c, a = body.n, body.c, body.area
    rx, rz = c[:, 0] - xr, c[:, 2] - zr
    nj = {1: n[:, 0], 3: n[:, 2], 5: rz * n[:, 0] - rx * n[:, 2]}
    xi = {1: np.tile([1.0, 0.0, 0.0], (len(c), 1)), 3: np.tile([0.0, 0.0, 1.0], (len(c), 1)),
          5: np.stack([rz, np.zeros_like(rz), -rx], 1)}
    dn = {(1, 5): n[:, 2], (3, 5): -n[:, 0]}
    C = np.zeros((3, 3))
    for r_, j in enumerate((1, 3, 5)):
        for c_, k in enumerate((1, 3, 5)):
            C[r_, c_] = -rho * np.sum(np.einsum("ij,ij->i", xi[k], WW) * nj[j] * a) * 2
            if (j, k) in dn:
                C[r_, c_] += rho * np.sum(f0 * dn[(j, k)] * a) * 2
    # waterline strip
    xw, f0w = hull_waterline_values(body, f0, db.shape)
    nw = np.stack([hull_waterline_values(body, n[:, q], db.shape)[1] for q in range(3)], 1)
    nw /= np.linalg.norm(nw, axis=1)[:, None]
    raw = body.raw.reshape(nx, nz, 4, 3)[:, -1]                            # top-row panels
    dl = np.zeros(nx)
    xm = np.zeros(nx)
    for i in range(nx):
        top = raw[i][np.abs(raw[i][:, 2]) <= 1e-12 * max(1.0, np.abs(raw[i]).max())]
        if len(top) >= 2:
            u = np.unique(np.round(top, 14), axis=0)
            if len(u) >= 2:
                dl[i] = np.linalg.norm(u[0] - u[-1])
                xm[i] = 0.5 * (u[0][0] + u[-1][0])
    nh = np.maximum(np.hypot(nw[:, 0], nw[:, 1]), 1e-12)
    njw = {1: nw[:, 0], 3: nw[:, 2], 5: (0.0 - zr) * nw[:, 0] - (xm - xr) * nw[:, 2]}
    xiz = {1: np.zeros(nx), 3: np.ones(nx), 5: -(xm - xr)}
    for r_, j in enumerate((1, 3, 5)):
        for c_, k in enumerate((1, 3, 5)):
            C[r_, c_] -= rho * np.sum(f0w * njw[j] * xiz[k] / nh * dl) * 2
    return C


def static_stability(Mm, C):
    """Static stability of M x'' + C x = 0 with a NON-symmetric stiffness C (B6 review, finding B4; the minimum
    eigenvalue of the symmetric part, used by the B6 CD row, is sufficient, not necessary): eigenvalues lambda of
    M^-1 C (motions e^(s t) with s^2 = -lambda). Statically stable = every eigenvalue has a positive real part (no
    divergence; for 2 x 2: det C > 0 and trace(M^-1 C) > 0). A real negative eigenvalue is a divergence; a complex
    pair with positive real part is a circulatory (flutter-type) coupling of the UNDAMPED system, which the damped
    trajectory of the plant law decides (reported as `circulatory`). Returns dict(eig, eig_C, det, sym_min, stable,
    circulatory)."""
    lam = np.linalg.eigvals(np.linalg.solve(np.asarray(Mm, float), np.asarray(C, float)))
    lc = np.linalg.eigvals(np.asarray(C, float))
    circ = bool(np.any(np.abs(lam.imag) > 1e-12 * np.abs(lam).max()))
    return dict(eig=lam, eig_C=lc, det=float(np.linalg.det(C)), sym_min=float(np.linalg.eigvalsh(0.5 * (C + C.T)).min()),
                stable=bool(np.all(lam.real > 0)), circulatory=circ)


# ===================================================================== time domain
class DBTimeDomain:
    """Explicit RK4 integration of the double-body LTI system with prescribed rigid-body motion (as
    unsteady.TimeDomain). Forces: D-form rho int (phi_t + W.grad phi) n_j (W.grad = G), OT form rho int (phi_t n_j
    - m_j phi). Pseudo-energy of the double-body conditions (stage note 11.2 item 8):
        H = E_kin + E_pot + M_W,  E_kin = (rho/2)[int_F phi phi_z - int_S phi phi_n],  E_pot = (rho g/2) int_F zeta^2,
        M_W = rho int_F zeta (W.grad phi),
        dH/dt = P_body + P_beach (continuum, exactly; NO waterline power: W.nu = 0 at the waterline),
        P_body = -rho int_S phi_t phi_n,
        P_beach = rho int nu phi sigma - rho g int nu zeta^2 - rho int [nu zeta (W.grad phi) + zeta W.grad(nu phi)];
    the discrete residual is the upwind dissipation (and the BIE/RK4 errors)."""

    def __init__(self, co):
        self.co, self.s = co, co.s
        s = self.s
        s.dtn_factor()
        dev = s.dev
        self.n = torch.as_tensor(np.stack([co.mv[k][0] for k in co.modes], 1), device=dev)
        self.m = torch.as_tensor(np.stack([co.mv[k][1] for k in co.modes], 1), device=dev)
        self.aF = torch.as_tensor(s.fs.area, device=dev)
        self.aB = torch.as_tensor(s.body.area, device=dev)
        self.nuF = torch.as_tensor(s.nu, device=dev)
        self._A = _ts(s.D, dev)

    def run(self, motion, dt, nsteps, zeta0=None, phi0=None, energy=True, every=1):
        """motion(t) -> (x, xd, xdd) over the modes (or None). Returns dict of time series."""
        s, dev = self.s, self.s.dev
        NF = s.NF
        zeta = torch.zeros((NF, 1), dtype=torch.float64, device=dev) if zeta0 is None else \
            torch.as_tensor(np.asarray(zeta0, float).reshape(NF, 1), device=dev)
        phiF = torch.zeros_like(zeta) if phi0 is None else torch.as_tensor(np.asarray(phi0, float).reshape(NF, 1),
                                                                           device=dev)
        mm = len(self.co.modes)

        def pn_of(t):
            if motion is None:
                return torch.zeros((s.NB, 1), dtype=torch.float64, device=dev)
            x, xd, _ = (torch.as_tensor(np.asarray(v, float), device=dev) for v in motion(t))
            return (self.n @ xd + self.m @ x)[:, None]

        def pnd_of(t):
            if motion is None:
                return torch.zeros((s.NB, 1), dtype=torch.float64, device=dev)
            _, xd, xdd = (torch.as_tensor(np.asarray(v, float), device=dev) for v in motion(t))
            return (self.n @ xdd + self.m @ xd)[:, None]
        out = {k: [] for k in ("t", "F_D", "F_OT", "H", "Ek", "Ep", "MW", "P_body", "P_beach", "x")}

        def record(t, zeta, phiF):
            pn, pnd = pn_of(t), pnd_of(t)
            dz, dp, sig = s.rhs_td(zeta, phiF, pn)
            sigd = s.sigma_of(dp, pnd)
            phB = s.phiB_of(sig, pn)
            phBd = s.phiB_of(sigd, pnd)
            phB_n, phBd_n = phB[:, 0].cpu().numpy(), phBd[:, 0].cpu().numpy()
            gB = self.co.db.wgrad(phB_n)
            a = s.body.area
            FD, FO = [], []
            for kk, j in enumerate(self.co.modes):
                nj, mj = self.co.mv[j]
                FD.append(s.rho * 2 * np.sum((phBd_n + gB) * nj * a))
                FO.append(s.rho * 2 * np.sum((phBd_n * nj - mj * phB_n) * a))
            out["t"].append(t)
            out["F_D"].append(FD)
            out["F_OT"].append(FO)
            out["x"].append(np.asarray(motion(t)[0], float) if motion is not None else np.zeros(mm))
            if energy:
                rho, g = s.rho, s.g
                Ek = 0.5 * rho * 2 * (-(self.aF * phiF[:, 0] * sig[:, 0]).sum() - (self.aB * phB[:, 0] * pn[:, 0]).sum())
                Ep = 0.5 * rho * g * 2 * (self.aF * zeta[:, 0] ** 2).sum()
                Aphi = torch.sparse.mm(self._A, phiF)[:, 0]
                MW = rho * 2 * (self.aF * zeta[:, 0] * Aphi).sum()
                Pb = -rho * 2 * (self.aB * phBd[:, 0] * pn[:, 0]).sum()
                Anphi = torch.sparse.mm(self._A, (self.nuF * phiF[:, 0])[:, None])[:, 0]
                Pbe = rho * 2 * ((self.aF * self.nuF * phiF[:, 0] * sig[:, 0]).sum()
                                 - g * (self.aF * self.nuF * zeta[:, 0] ** 2).sum()
                                 - (self.aF * zeta[:, 0] * (self.nuF * Aphi + Anphi)).sum())
                out["Ek"].append(float(Ek))
                out["Ep"].append(float(Ep))
                out["MW"].append(float(MW))
                out["H"].append(float(Ek + Ep + MW))
                out["P_body"].append(float(Pb))
                out["P_beach"].append(float(Pbe))
        t = 0.0
        record(t, zeta, phiF)
        for it in range(nsteps):
            p1, p2, p4 = pn_of(t), pn_of(t + 0.5 * dt), pn_of(t + dt)
            k1z, k1p, _ = s.rhs_td(zeta, phiF, p1)
            k2z, k2p, _ = s.rhs_td(zeta + 0.5 * dt * k1z, phiF + 0.5 * dt * k1p, p2)
            k3z, k3p, _ = s.rhs_td(zeta + 0.5 * dt * k2z, phiF + 0.5 * dt * k2p, p2)
            k4z, k4p, _ = s.rhs_td(zeta + dt * k3z, phiF + dt * k3p, p4)
            zeta = zeta + (dt / 6) * (k1z + 2 * k2z + 2 * k3z + k4z)
            phiF = phiF + (dt / 6) * (k1p + 2 * k2p + 2 * k3p + k4p)
            t += dt
            if (it + 1) % every == 0:
                record(t, zeta, phiF)
            if not torch.isfinite(zeta).all():
                raise FloatingPointError(f"time integration diverged at t = {t:g}")
        res = {k: np.array(v) for k, v in out.items() if len(v)}
        res["zeta_end"] = zeta[:, 0].cpu().numpy()
        res["phi_end"] = phiF[:, 0].cpu().numpy()
        return res


# ===================================================================== steady problem
def steady_db(s, co, hull, sink=0.0, trim=0.0, cg=(0.0, 0.0)):
    """Steady wave problem of the double-body linearisation on the system s (stage note 11.2 item 9): the omega =
    0 response of the same LTI system to the dynamic forcing f_d = (U^2 - |W|^2)/2 (phi_n = 0 on the hull: the
    base flow satisfies the hull condition), i.e.
        g phi_z + div_h(W (W.grad phi)) = div_h(W (U^2 - |W|^2)/2)   on z = 0,
    zeta = (f_d - W.grad phi)/g the FULL steady elevation. Hull pressure (Bernoulli about the base flow):
        p = rho [(U^2 - |W|^2)/2 - W.grad phi - |grad phi|^2/2]   (hydrostatics separate),
    the base-flow part p_db = rho (U^2 - |W|^2)/2 carries no x-force in the continuum (d'Alembert for the double body);
    its discrete x-force is reported (dalembert) and removed from the resistance:
        R_p = -int [p - p_db] n_x dS (linear part only: R_lin; with the quadratic term: R_quad).
    Fz, My (about the CG, body (x_G, z_G) carried with the attitude) include p_db (dominant at low Fn) and the
    linear wave part; the quadratic term is reported separately. Waterline elevation from the hull side,
    zeta_wl = (f_d - W.grad phi)/g extrapolated to z = 0. Returns dict."""
    from .unsteady import hull_waterline_values
    U, g, rho = s.U, s.g, s.rho
    pan = s.body
    Wf, phizz = co.fs_base()
    fd = 0.5 * (U * U - np.einsum("ij,ij->i", Wf, Wf))
    sol = s.solve_freq(0.0, np.zeros((s.NB, 1)), fd=fd[:, None])
    phB = np.real(sol["phi_B"][:, 0])
    phF = np.real(sol["phi_F"][:, 0])
    zeta = np.real(s.zeta_of(0.0, phF, fd))
    WB = co.db.W
    gph = surface_gradient2(pan, phB, np.zeros(len(pan)), co.db.shape)
    pdb = 0.5 * rho * (U * U - np.einsum("ij,ij->i", WB, WB))
    plin = -rho * np.einsum("ij,ij->i", WB, gph)
    pq = -0.5 * rho * np.einsum("ij,ij->i", gph, gph)
    cgE = hull.to_earth(np.array(cg[0]), np.array(0.0), np.array(cg[1]), sink, trim)
    r = pan.c - cgE

    def FM(p):
        F = -2 * np.sum((p * pan.area)[:, None] * pan.n, axis=0)
        My = -2 * np.sum(p * pan.area * (r[:, 2] * pan.n[:, 0] - r[:, 0] * pan.n[:, 2]))
        return F, My
    Fdb, Mdb = FM(pdb)
    Fl, Ml = FM(plin)
    Fq, Mq = FM(pq)
    xa, gx = hull_waterline_values(pan, np.einsum("ij,ij->i", WB, gph), co.db.shape)
    _, fdw = hull_waterline_values(pan, pdb / rho, co.db.shape)
    zwl = (fdw - gx) / g
    out = dict(phi_B=phB, phi_F=phF, zeta=zeta, fd=fd, Fz=float(Fdb[2] + Fl[2]), My=float(Mdb + Ml),
               Fz_db=float(Fdb[2]), My_db=float(Mdb), Fz_q=float(Fq[2]), My_q=float(Mq), R_lin=float(-Fl[0]),
               R_quad=float(-(Fl[0] + Fq[0])), dalembert=float(-Fdb[0]), x_wl=xa, zeta_wl=zwl)
    return out


STEADY_GRID = dict(up=1.0, down=2.5, lat=1.5, r_up=1.10, r_dn=1.04, r_lat=1.06, b_up=0.5, b_dn=1.5, b_lat=0.75,
                   u_dn=1.2, u_lat=0.4, nu_max=1.5)


def steady_grid(hull, U, g=9.81, cells=30, m_min=40, m_max=90, scale=True, **over):
    """Free-surface grid settings for the steady problem: compact domain (STEADY_GRID, lengths in hull.L) and the
    number of columns over the waterline M chosen so that the transverse wave lambda0 = 2 pi U^2/g spans >= `cells`
    cells (as B4's n_per_lam), clipped to [m_min, m_max] (dense-LU memory).
    scale=True (since the B6 review, finding B6; stage note 12.2 item 7; numerical device registered in
    EMPIRICAL_REGISTER_RANKINE.md): the domain grows with the steady wavelength l = lambda0/L where STEADY_GRID is
    shorter: every beach at least one wavelength (b_up, b_lat >= l; b_dn >= max(1.5, l)), at least 0.5 L of
    undamped surface upstream and 0.75 L laterally inside the beaches, and the downstream undamped zone long enough
    for the wave cut 1 lambda0 behind the stern plus half a wavelength (down >= 1.5 l + b_dn):
        b_up = max(0.5, l), b_dn = max(1.5, l), b_lat = max(0.75, l),
        up = max(1.0, 0.5 + b_up), down = max(2.5, 1.5 l + b_dn), lat = max(1.5, 0.75 + b_lat).
    The B6 domain (unscaled) placed the upstream beach inside one wavelength above Fn ~0.3 and the Fn 0.408 cut
    inside the downstream beach; on a deep spheroid at Fn 0.5 it gave +13 % against the exact Kelvin solve, the
    scaled domain +1 % (exploration B6R-E4). For Fn <= 0.28 (l <= 0.5) the rule returns STEADY_GRID unchanged.
    Returns the grid keyword dict."""
    lam = 2 * np.pi * U * U / g
    xs, xb = hull.waterline_ends()
    M = int(np.clip(np.ceil((xb - xs) / (lam / cells)), m_min, m_max))
    kw = dict(STEADY_GRID)
    if scale:
        l = lam / hull.L
        b_up, b_dn, b_lat = max(0.5, l), max(1.5, l), max(0.75, l)
        kw.update(b_up=b_up, b_dn=b_dn, b_lat=b_lat, up=max(1.0, 0.5 + b_up), down=max(2.5, 1.5 * l + b_dn),
                  lat=max(1.5, 0.75 + b_lat))
    kw.update(M=M, **over)
    return kw


def steady_cut(s, G, zeta, hull, U, g, rho=1.0, d=1.0, sink=0.0, trim=0.0):
    """Wave resistance from the transverse cut d wavelengths behind the stern (piercing.cut_from_samples, B4 check
    H7 calibration), zeta_x by the B5 d/dx of the grid (wall rule: the cut column is far from the hull)."""
    from .unsteady import dx_operator
    from .piercing import cut_from_samples
    D, _ = dx_operator(G, 1.0 if U >= 0 else -1.0, wall="rule")
    zx = D @ zeta
    nx, ny = G["shape"]
    lam = 2 * np.pi * U * U / g
    ic = int(np.argmin(np.abs(G["Xc"][:, 0] - (hull.waterline_ends(sink, trim)[0] - d * lam))))
    ks = np.arange(ny) * nx + ic
    R, info = cut_from_samples(G["fs"].c[ks, 1], zeta[ks], zx[ks], G["ylines"][-1], U, U * U / g, rho,
                               0.5 * np.pi / G["dy0"])
    return R, dict(info, x_cut=float(G["Xc"][ic, 0]))


def db_free_equilibrium(hull, U, grid_kw, hull_mesh, g=9.81, rho=1.0, cg=None, tol=1e-6, max_iter=10, verbose=False,
                        x0=(0.0, 0.0), device=None):
    """Free sinkage and trim of the steady double-body problem (design P-7, as piercing.free_equilibrium): Fz_total
    = W, My_total = 0 about the CG (default (LCB at rest, 0)), total = exact hydrostatics (hull.hydrostatic_load) +
    the dynamic force of steady_db (base-flow pressure + linear wave part) at the attitude; Newton with the exact
    hydrostatic Jacobian as the initial matrix, good-Broyden updates; every residual is a full solve at the new
    attitude (hull re-panelled, grid, base flow and operators rebuilt). Returns dict(sink, trim, out, history,
    converged)."""
    hs0 = hull.hydrostatics(0.0, 0.0)
    W = rho * g * hs0["V"]
    if cg is None:
        cg = (float(hs0["centroid_body"][0]), 0.0)
    L = hull.L

    def resid(x):
        s, co, G, db = build_db_solver(hull, U, x[0], x[1], grid_kw, hull_mesh, g=g, rho=rho, device=device)
        out = steady_db(s, co, hull, x[0], x[1], cg)
        hl = hull.hydrostatic_load(x[0], x[1], rho, g, cg)
        r = np.array([(hl["Fz"] + out["Fz"] - W) / W, (hl["My"] + out["My"]) / (W * L)])
        return r, (s, co, G, db, out)

    def hs_res(x):
        hl = hull.hydrostatic_load(x[0], x[1], rho, g, cg)
        return np.array([(hl["Fz"] - W) / W, hl["My"] / (W * L)])
    x = np.array(x0, float)
    eps = np.array([1e-4 * getattr(hull, "T", 0.1 * L), 1e-4])
    Jm = np.zeros((2, 2))
    for c in range(2):
        e = np.zeros(2)
        e[c] = eps[c]
        Jm[:, c] = (hs_res(x + e) - hs_res(x - e)) / (2 * eps[c])
    r, pack = resid(x)
    hist = [dict(sink=float(x[0]), trim=float(x[1]), res=r.tolist())]
    if verbose:
        print(f"    db eq it 0: s {x[0]:.6g} th {x[1]:.6g} res {r}", flush=True)
    for it in range(1, max_iter + 1):
        if np.all(np.abs(r) <= tol):
            break
        dxv = -np.linalg.solve(Jm, r)
        pack[0].release()
        xn = x + dxv
        rn, packn = resid(xn)
        Jm = Jm + np.outer((rn - r) - Jm @ dxv, dxv) / (dxv @ dxv)
        x, r, pack = xn, rn, packn
        hist.append(dict(sink=float(x[0]), trim=float(x[1]), res=r.tolist()))
        if verbose:
            print(f"    db eq it {it}: s {x[0]:.6g} th {x[1]:.6g} res {r}", flush=True)
    return dict(sink=float(x[0]), trim=float(x[1]), pack=pack, history=hist, converged=bool(np.all(np.abs(r) <= tol)),
                cg=cg)
