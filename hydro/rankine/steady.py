"""
Steady forward-speed problem with the free surface linearised about the
uniform stream (Neumann-Kelvin; design section 8.3 P-2/P-3, step S1).

Since stage B3 (decision D1) the production free-surface operator is the
shifted Numerov operator of shifted.py: the upstream-difference rows of this
module are linear multistep methods of order > 2 and therefore (Dahlquist
1978) unstable on the short diverging waves of every practical grid
(upwind5 grows 4.5 % per wavelength on transverse waves at 30 panels per
wavelength; studies/rankine_d1.py). SteadyLinearFS is kept unchanged so the
stage-1 numbers stay reproducible; the helpers below (fs_rect, double-body
flow, base-flow derivatives, surface gradient, d'Alembert-corrected near
field) are shared with shifted.py.

Frame fixed to the body, stream -U x_hat (body moving at +U), fluid z < 0,
total potential -U x + phi. Linearising the exact steady conditions (the
WaveBEM equations W-18/W-19 with d/dt = 0, checked term by term in
learn/meta/WAVEBEM_PORT_DESIGN.md section 3.1) about the uniform stream:

  kinematic  -U zeta_x = phi_z,  dynamic  g zeta - U phi_x = 0
  =>  U^2 phi_xx + g phi_z = 0  on z = 0,     zeta = (U/g) phi_x           (K)

On the free-surface panels (normal -z): dphi/dn = -phi_z = (U^2/g) phi_xx.
phi_xx is taken from the panel values along each streamwise grid row with
an upstream finite-difference operator of Dawson's (1977) type (dxx_upwind:
Dawson's four-point form, or the five-point form exact for quartics, the
default), which supplies the radiation condition (no waves upstream)
through its upstream bias. Upstream of the first column the virtual points
of the stencil are continued linearly, phi_{-m} = (1 + m) phi_0 - m phi_1
(changed after review, see dxx_upwind: the earlier zero-gradient
continuation put a line of spurious free-surface sources along the upstream
edge whenever phi_x != 0 there, e.g. for the double-body part of phi).
With identity (E) (bie.py) on body + free surface:

  [2 pi I - D + S_F (U^2/g) Dxx] phi = -S_B (dphi/dn)_B - S_F f_s           (S)

where f_s is the free-surface forcing of a prescribed singularity phi_s
(phi = phi_s + psi; f_s = (U^2/g) phi_s,xx + phi_s,z), used by the check S1,
or of the double-body flow in the split form (base_flow_forcing).
Body boundary condition dphi/dn = U n_x (n out of the body).

Pressure on the body from the full Bernoulli equation for the perturbation,
p = rho (U phi_x - |grad phi|^2 / 2) - rho g z, with grad phi = tangential
gradient (least squares on neighbouring panels, images included) + U n_x n.
Wave resistance R = int p n_x dS (force -int p n dS, resistance = -F_x).

Near-field resistance and d'Alembert (added after review). The x-force of
the double-body (rigid-lid) flow phi_db is exactly zero (d'Alembert: body +
image in an unbounded steady stream; the image carries the same x-force by
symmetry, and the pair carries none). On a panel mesh the pressure integral
of phi_db is not zero but a discretisation error which, on a fore-aft
asymmetric mesh, is of the order of the wave resistance of a deep body at low
Fn (review: +6 %, +0.7 %, -2.8 % of R_Havelock at Fn 0.4, f/b = 4). Since
int p(phi_db) n_x dS = 0 exactly,
    R = -int [p(phi) - p(phi_db)] n_x dS
holds exactly in the continuum; evaluated on the same mesh with the same
gradient operator, the discrete error of the double-body part cancels and
what remains is the error of the smaller free-surface-induced part,
p(phi) - p(phi_db) = rho [U psi_x - grad phi_db . grad psi - |grad psi|^2/2],
psi = phi - phi_db. near_field_resistance returns the raw and the corrected
value; the far-field (Kochin) value of farfield.py is the independent check.
"""
import numpy as np
import scipy.sparse as sp
import torch

from .bie import ImageOperator, neumann_green
from .panels import Panels


def fs_rect(x_up, x_down, y_max, nx, ny, tag=1):
    """Uniform free-surface grid on [x_down, x_up] x [0, y_max], z = 0.
    Panel k = j * nx + i; column i = 0 is the most upstream (largest x).
    Normals -z. Returns (Panels, dict(nx, ny, dx, dy))."""
    xs = np.linspace(x_up, x_down, nx + 1)
    ys = np.linspace(0.0, y_max, ny + 1)
    q = np.zeros((ny, nx, 4, 3))
    for j in range(ny):
        for i in range(nx):
            # corners ordered so that the normal is -z
            q[j, i] = [[xs[i], ys[j], 0], [xs[i + 1], ys[j], 0],
                       [xs[i + 1], ys[j + 1], 0], [xs[i], ys[j + 1], 0]]
    pan = Panels(q.reshape(-1, 4, 3), np.full(nx * ny, tag))
    if np.mean(pan.n[:, 2]) > 0:
        pan = Panels(q.reshape(-1, 4, 3)[:, ::-1], np.full(nx * ny, tag))
    return pan, dict(nx=nx, ny=ny, dx=(x_up - x_down) / nx, dy=y_max / ny)


def upwind_weights(scheme):
    """Weights w_m of phi_xx ~ sum_m w_m phi(x + m dx) / dx^2 (m = 0, 1, ...
    upstream). "dawson4": Dawson (1977). "upwindP" (P = 5, 6, 7): the unique
    P-point upstream formula exact for polynomials of degree P - 1
    (moment conditions sum_m w_m m^n = 2 delta_n2, n < P)."""
    if scheme == "dawson4":
        return np.array([2.0, -5.0, 4.0, -1.0])
    P = int(scheme.replace("upwind", ""))
    M = np.array([[float(m) ** n for m in range(P)] for n in range(P)])
    b = np.zeros(P)
    b[2] = 2.0
    return np.linalg.solve(M, b)


def dxx_upwind(nx, ny, dx, scheme="upwind5", upstream="linear"):
    """Sparse upstream second-difference operator on the fs_rect ordering.
    Symbol: D(alpha) dx^2 = sum_m w_m e^{i m theta}, theta = alpha dx; with the
    2-D free-surface condition k D(alpha) + |alpha| = 0 a relative symbol
    error -alpha^2 (1 + e) moves the wave number to alpha ~ k0 (1 - e), so
    Re e is a wavelength error and Im e > 0 a downstream DAMPING, Im e < 0
    a downstream GROWTH of the waves (exp(Im alpha |x|) behind the body).
    dawson4: (2, -5, 4, -1)/dx^2 on (i, i-1, i-2, i-3), exact for cubics
             (Dawson 1977); e = (11/12) theta^2 + i theta^3 + ...: 3.7 %
             wavelength error and 4 % damping per wavelength at 30 panels
             per wavelength (check S1b: R 6-13 % off).
    upwind5: (35/12, -26/3, 19/2, -14/3, 11/12)/dx^2 on (i .. i-4), exact for
             quartics (default); e = -(5/6) i theta^3 + 1.32 theta^4 + ...:
             0.26 % wavelength error but a GROWTH of (5/6) k0 theta^3 per unit
             length, +4.5 % amplitude per wavelength at 30 panels per
             wavelength, +1.4 % at 45 (found after review by check S3b,
             which measures +5 to +6 % per wavelength).
    upwind6, upwind7: the 6- and 7-point formulas exact for degree 5 and 6
             (1-D symbol: +0.4 % and -0.15 % per wavelength at 30); tested
             after review and NOT used: in the coupled panel system upwind6
             shows a 39 % error band near the centre line and upwind7 is
             unstable at 20 panels per wavelength (wake amplitude x100), so
             the 1-D symbol does not decide stability (decision D1).
    Upstream virtual points (columns i - m < 0):
      "linear" (default since the 2026-10-06 review): phi_{-m} = (1 + m) phi_0
          - m phi_1, so the stencil stays exact for phi linear in x. The
          extrapolated value errs by m (m + 1)/2 dx^2 phi_xx + O(dx^3), so
          the boundary rows carry an error of the size of phi_xx itself
          there (columns 0 and 1 reduce to phi_xx = 0, a rigid lid), which
          vanishes as the upstream extent grows (the non-wave field decays
          like a dipole, phi_xx ~ r^-4).
      "zero_gradient" (before the review): phi_{-m} = phi_0. The row of
          column 1 becomes 35 (phi_1 - phi_0)/(12 dx^2) ~ 2.9 |phi_x|/dx: an
          error O(phi_x/dx) that does not vanish as dx -> 0 -- a line of
          spurious free-surface sources (net ~ k phi_x per unit span) along
          the upstream edge. It moved the S2 pressure resistance by 16 %
          with the upstream extent. Kept only to reproduce earlier numbers.
      "zero": phi = 0 upstream (stiff boundary forcing, erratic; kept for
          the record)."""
    w = upwind_weights(scheme)
    rows, cols, vals = [], [], []
    for j in range(ny):
        for i in range(nx):
            k = j * nx + i
            for m, wm in enumerate(w):
                ii = i - m
                if ii >= 0:
                    rows.append(k)
                    cols.append(j * nx + ii)
                    vals.append(wm / dx ** 2)
                elif upstream == "linear":
                    mm = -ii
                    rows += [k, k]
                    cols += [j * nx, j * nx + 1]
                    vals += [wm * (1 + mm) / dx ** 2, -wm * mm / dx ** 2]
                elif upstream == "zero_gradient":
                    rows.append(k)
                    cols.append(j * nx)
                    vals.append(wm / dx ** 2)
                elif upstream != "zero":
                    raise ValueError(upstream)
    return sp.csr_matrix((vals, (rows, cols)), shape=(nx * ny, nx * ny))


def dx_central(nx, ny, dx):
    """Central first difference along rows (one-sided at the ends)."""
    rows, cols, vals = [], [], []
    for j in range(ny):
        for i in range(nx):
            k = j * nx + i
            if 0 < i < nx - 1:
                pairs = [(i - 1, 1.0), (i + 1, -1.0)]     # x decreases with i
                scale = 2 * dx
            elif i == 0:
                pairs = [(0, 1.0), (1, -1.0)]
                scale = dx
            else:
                pairs = [(nx - 2, 1.0), (nx - 1, -1.0)]
                scale = dx
            for ii, c in pairs:
                rows.append(k)
                cols.append(j * nx + ii)
                vals.append(c / scale)
    return sp.csr_matrix((vals, (rows, cols)), shape=(nx * ny, nx * ny))


def _torch_sparse(A, dev):
    A = A.tocoo()
    idx = torch.as_tensor(np.vstack([A.row, A.col]), dtype=torch.long, device=dev)
    val = torch.as_tensor(A.data, dtype=torch.float64, device=dev)
    return torch.sparse_coo_tensor(idx, val, A.shape).coalesce()


class SteadyLinearFS:
    """Body (Panels or None) + free-surface grid, port/starboard image.

    base = "rigid_lid" (default): the even image in z = 0 is added to every
    panel, i.e. the Green function is the rigid-lid (Neumann) one,
    1/r + 1/r'. Then the dipoles of the free-surface panels cancel with
    their images, the free-surface panels carry only the sources
    dphi/dn = (U^2/g) phi_xx, which vanish as U -> 0 (the double-body flow
    is exact through the images, design P-13), and truncating the free
    surface no longer truncates the wide non-wave image flow (check S1/S2:
    without it, a deep body showed a spurious resistance of ~5 % that did
    not converge with resolution). A collocation point on z = 0 is then an
    interior point of the doubled domain, so its coefficient is 4 pi.
    base = "none": plain identity (E) on body + free surface.

    The influence matrices stay on `device` (torch) and the system matrix is
    assembled there: host RAM stays at a few hundred MB for N ~ 1e4; device
    memory is about 4 N^2 x 8 bytes during a solve (S, D, system matrix, LU)."""

    def __init__(self, body, fs, grid, device=None, scheme="upwind5",
                 upstream="linear", base="rigid_lid", entry=None, beach=None):
        """entry: length of a smooth upstream entry zone (None: none); the
        free-surface sources are multiplied by w(x), rising from 0 at the
        upstream edge to 1 at entry length with the C2 ramp 6s^5 - 15s^4 +
        10s^3, so the truncation of the free surface upstream does not act
        as a line wavemaker across the whole width.
        beach: dict(length=L_b, decay=a) for a downstream damping zone over
        the last L_b of the grid: Rayleigh damping in the free-surface
        condition, U^2 (phi_xx - nu(x) phi_x) + g phi_z = 0, nu = nu_max s^2
        (s from 0 at the zone start to 1 at the end, nu_max = 3 a / L_b), so
        transverse waves decay by e^{-a} (amplitude) across the zone
        (dispersion: alpha = k0 + i nu for nu << k0; oblique waves decay at
        nu/(2 - cos^2 theta)). The steady counterpart of WaveBEM's quadratic
        damping zone (free_surface.cc, wave_damping_pressure); removes the
        abrupt end of the wake, whose oscillating truncation error decays
        only slowly with the downstream extent."""
        self.body, self.fs, self.grid = body, fs, grid
        self.NB = 0 if body is None else len(body)
        self.pan = fs if body is None else body + fs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.base = base
        if base == "rigid_lid":
            self.axes, self.par = (1, 2), (1, 1)
        else:
            self.axes, self.par = (1,), (1,)
        self.op = ImageOperator(self.pan, axes=self.axes, device=self.device,
                                parities=[self.par], on_device=True)
        Dxx = dxx_upwind(grid["nx"], grid["ny"], grid["dx"], scheme, upstream)
        self.Dx = dx_central(grid["nx"], grid["ny"], grid["dx"])
        x = fs.c[:, 0]
        x_up, x_dn = x.max() + 0.5 * grid["dx"], x.min() - 0.5 * grid["dx"]
        self.nu = np.zeros(len(fs))
        if beach is not None:
            Lb = beach["length"]
            s = np.clip((x_dn + Lb - x) / Lb, 0.0, 1.0)
            self.nu = 3.0 * beach.get("decay", 6.0) / Lb * s * s
            Dxx = Dxx - sp.diags(self.nu) @ self.Dx
        self.w = np.ones(len(fs))
        if entry:
            s = np.clip((x_up - x) / entry, 0.0, 1.0)
            self.w = s ** 3 * (10 - 15 * s + 6 * s * s)
            Dxx = sp.diags(self.w) @ Dxx
        self.Dxx = sp.csr_matrix(Dxx)
        self._DxxT = _torch_sparse(self.Dxx.T, self.device)

    def forcing(self, U, g, phis_xx, phis_x=None):
        """dphi/dn on the free surface of a prescribed field phi_s (for the
        rigid-lid base pass the derivatives of phi_s + its z-image, i.e. 2 x
        those of phi_s at z = 0; phi_s,z = 0 there), consistent with the
        entry ramp and the beach: (U^2/g) w (phi_s,xx - nu phi_s,x)."""
        f = np.asarray(phis_xx, float)
        if phis_x is not None:
            f = f - self.nu * np.asarray(phis_x, float)
        return (U * U / g) * self.w * f

    def _system(self, k, consume=False):
        """2 pi I - D (+ 2 pi I on the FS rows for the rigid-lid base) + k S_F L,
        L = diag(w) (Dxx - diag(nu) Dx) the free-surface operator (self.Dxx).
        consume=True builds it in the storage of D (one solve only)."""
        S, D = self.op.S(self.par), self.op.D(self.par)
        NB, N = self.NB, S.shape[0]
        M = D.neg_() if consume else -D
        M.diagonal().add_(2 * np.pi)
        if self.base == "rigid_lid":
            M.diagonal()[NB:].add_(2 * np.pi)
        if k != 0.0:
            SF = S[:, NB:]
            step = max(1, int(2.5e8 // (8 * (N - NB))))
            for r0 in range(0, N, step):
                r1 = min(N, r0 + step)
                # (S_F Dxx)[rows] = (Dxx^T S_F[rows]^T)^T
                blk = torch.sparse.mm(self._DxxT, SF[r0:r1].T.contiguous())
                M[r0:r1, NB:].add_(blk.T, alpha=k)
                del blk
        return M

    def solve(self, U, g, phin_body=None, fs_forcing=None, consume=False):
        """Solve (S). fs_forcing: dphi/dn on the free surface due to a
        prescribed singularity or a base flow (use self.forcing). Returns
        phi on all panels and dphi/dn on the free surface (numpy).
        consume=True: single solve; the matrix is built in place of D and S
        is freed before the LU (device memory 2 N^2 x 8 bytes instead of 4,
        N ~ 2.5e4 on a 16 GB GPU); the solver cannot solve again."""
        NB = self.NB
        S = self.op.S(self.par)
        k = U * U / g
        M = self._system(k, consume)
        rhs = torch.zeros(S.shape[0], dtype=torch.float64, device=self.device)
        if NB:
            rhs -= S[:, :NB] @ torch.as_tensor(np.asarray(phin_body, float),
                                               device=self.device)
        if fs_forcing is not None:
            rhs -= S[:, NB:] @ torch.as_tensor(np.asarray(fs_forcing, float),
                                               device=self.device)
        del S
        if consume:
            self.op = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        phi = torch.linalg.solve(M, rhs)
        del M
        phi = phi.cpu().numpy()
        phin_F = k * (self.Dxx @ phi[NB:])
        if fs_forcing is not None:
            phin_F = phin_F + fs_forcing
        return phi, phin_F

    def elevation(self, U, g, phi_fs, phis_x=None):
        """zeta = (U/g) phi_x on the free-surface panels (central differences)."""
        px = self.Dx @ phi_fs
        if phis_x is not None:
            px = px + phis_x
        return U / g * px

    def release(self):
        """Free the device matrices."""
        self.op = None
        self._DxxT = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ----------------------------------------------- prescribed point sources
def point_source_fields(Q, X, P):
    """(phi, grad phi, phi_xx) at points P (M,3) of sources Q_k at X_k
    (phi = -Q/(4 pi r)); the y-image is NOT added (list the sources in full)."""
    d = P[:, None, :] - X[None, :, :]                 # (M, K, 3)
    r = np.linalg.norm(d, axis=2)
    c = -np.asarray(Q)[None, :] / (4 * np.pi)
    phi = (c / r).sum(1)
    grad = (-c[..., None] * d / r[..., None] ** 3).sum(1)
    xx = (c * (3 * d[..., 0] ** 2 / r ** 5 - 1 / r ** 3)).sum(1)
    return phi, grad, xx


def lagally_resistance(solver, phin_F, Q, X, rho=1.0):
    """R = rho sum_k Q_k u_x(X_k) for prescribed sources in a body-free
    problem, u = velocity of everything except the sources themselves:
    the free-surface sources (sigma = 2 dpsi/dn: half-space Neumann
    representation of the free-surface response) plus, with
    base="rigid_lid", the rigid-lid images of the sources."""
    from .influence import influence
    sig = 2.0 * phin_F
    _, _, G1 = influence(X, solver.fs, want_grad=True, device=solver.device)
    _, _, G2 = influence(X, solver.fs.reflect(1), want_grad=True, device=solver.device)
    grad = -np.einsum("kjc,j->kc", G1 + G2, sig) / (4 * np.pi)
    if solver.base == "rigid_lid":
        Xi = np.asarray(X, float).copy()
        Xi[:, 2] *= -1
        _, gi, _ = point_source_fields(Q, Xi, np.asarray(X, float))
        grad = grad + gi
    return rho * float(np.sum(np.asarray(Q) * grad[:, 0]))


def prescribed_forcing(Q, X, P, U, g, base="rigid_lid"):
    """Free-surface forcing f_s = (U^2/g) phi_s,xx + phi_s,z at points P on
    z = 0 for point sources (with their rigid-lid images when base is
    rigid_lid: then phi_s,z = 0 and phi_s,xx doubles)."""
    _, grad, xx = point_source_fields(Q, X, P)
    if base == "rigid_lid":
        return 2.0 * (U * U / g) * xx
    return (U * U / g) * xx + grad[:, 2]


# ------------------------------------------------ double-body (base) flow
def double_body(body, U, device=None):
    """Rigid-lid double-body potential on the body panels (port/starboard and
    z = 0 images), body condition dphi/dn = U n_x."""
    op = ImageOperator(body, axes=(1, 2), device=device, parities=[(1, 1)])
    phi = neumann_green(op, U * body.n[:, 0], (1, 1), device=device)
    del op
    return phi


def rigid_lid_field(body, phi_b, phin_b, P, device=None):
    """phi at fluid points P from body data with the y and rigid-lid images:
    4 pi phi(x) = sum_images [D phi - S dphi/dn] (identity of bie.py at an
    exterior point; no jump term)."""
    op = ImageOperator(body, axes=(1, 2), xf=P, device=device, parities=[(1, 1)])
    v = (op.D((1, 1)) @ phi_b - op.S((1, 1)) @ phin_b) / (4 * np.pi)
    del op
    return v


def base_flow_derivatives(body, phi_db, phin_b, P, h=None, device=None):
    """phi_db,x and phi_db,xx at points P on z = 0 of the double-body flow
    (body + images), for the split phi = phi_db + psi (psi: body condition 0,
    free surface dpsi/dn = (U^2/g) w (Dxx psi - nu Dx psi + phi_db,xx -
    nu phi_db,x), i.e. fs_forcing = solver.forcing(U, g, phi_db,xx,
    phi_db,x)). Fourth-order central differences of the exact representation
      f'  = (-f(x+2h) + 8 f(x+h) - 8 f(x-h) + f(x-2h)) / (12 h)
      f'' = (-f(x+2h) + 16 f(x+h) - 30 f(x) + 16 f(x-h) - f(x-2h)) / (12 h^2)
    (truncation O(h^4); default h = 0.1 x the depth of the body's top,
    ~1e-6 relative). Independent of the upwind operator, so the split serves
    as the reference for the base-flow part of the solution (check S4b)."""
    P = np.asarray(P, float)
    if h is None:
        h = 0.1 * max(1e-3, -float(body.verts[:, :, 2].max()))
    f = {m: rigid_lid_field(body, phi_db, phin_b, P + [m * h, 0.0, 0.0], device)
         for m in (-2, -1, 0, 1, 2)}
    fx = (-f[2] + 8 * f[1] - 8 * f[-1] + f[-2]) / (12 * h)
    fxx = (-f[2] + 16 * f[1] - 30 * f[0] + 16 * f[-1] - f[-2]) / (12 * h * h)
    return fx, fxx


# ----------------------------------------------------- body surface gradient
def _neighbours(C, pts, k):
    """Indices of the k nearest points of C to each of pts; distances equal
    to 1e-9 relative are treated as ties and broken by index, so the choice
    does not depend on round-off (a Froude-scaled or translated mesh gets
    identical neighbour sets: check S5)."""
    from scipy.spatial import cKDTree
    kq = min(len(C), k + 9)
    dist, idx = cKDTree(C).query(pts, k=kq)
    out = np.empty((len(pts), k), dtype=int)
    for i in range(len(pts)):
        o = np.argsort(dist[i], kind="stable")
        d, ix = dist[i][o], idx[i][o]
        grp = np.concatenate([[0], np.cumsum(np.diff(d) > 1e-9 * np.maximum(d[1:], 1e-300))])
        out[i] = ix[np.lexsort((ix, grp))][:k]
    return out


def surface_gradient(pan, phi, axes=(1,), parity=(1,), k=12):
    """Tangential gradient of panel values by weighted least squares of a
    quadratic in the local tangent plane over the k nearest centroids
    (image panels included with their parity). Returns (N, 3)."""
    import itertools
    C, V = [pan.c], [phi]
    for cmb in itertools.product((0, 1), repeat=len(axes)):
        if not any(cmb):
            continue
        cc = pan.c.copy()
        sg = 1.0
        for ax, use, pr in zip(axes, cmb, parity):
            if use:
                cc[:, ax] *= -1
                sg *= pr
        C.append(cc)
        V.append(sg * phi)
    C = np.concatenate(C)
    V = np.concatenate(V)
    idx = _neighbours(C, pan.c, k + 1)
    out = np.zeros((len(pan), 3))
    for i in range(len(pan)):
        n = pan.n[i]
        t1 = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        nb = idx[i][1:]
        d = C[nb] - pan.c[i]
        u, v = d @ t1, d @ t2
        h2 = np.maximum(u * u + v * v, 1e-300)
        wgt = 1.0 / h2
        hs = np.sqrt(np.median(h2))                    # local length scale
        A = np.stack([u / hs, v / hs, 0.5 * (u / hs) ** 2, u * v / hs ** 2,
                      0.5 * (v / hs) ** 2], 1)
        b = V[nb] - phi[i]
        sw = np.sqrt(wgt * hs * hs)
        # truncated SVD: near the tips of a body the neighbours form a ring
        # and the quadratic fit is nearly singular (found by check S2)
        coef, *_ = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=1e-6)
        coef = coef / np.array([hs, hs, hs * hs, hs * hs, hs * hs])
        out[i] = coef[0] * t1 + coef[1] * t2
    return out


def body_force(pan, phi, U, rho=1.0, copies=2, axes=(1,), parity=(1,)):
    """Hydrodynamic force (x, z) and pressure from full Bernoulli of the
    perturbation; n out of the body; copies = symmetric copies of `pan`."""
    gs = surface_gradient(pan, phi, axes, parity)
    grad = gs + (U * pan.n[:, 0])[:, None] * pan.n
    p = rho * (U * grad[:, 0] - 0.5 * np.einsum("ij,ij->i", grad, grad))
    F = -copies * np.sum((p * pan.area)[:, None] * pan.n, axis=0)
    return F, p, grad


def near_field_resistance(pan, phi, U, phi_db=None, rho=1.0, copies=2):
    """Pressure-integral wave resistance: raw -F_x and, with phi_db given,
    the d'Alembert-corrected -(F_x - F_x,db) (module docstring), with the
    discrete double-body x-force reported as `dalembert` (exactly 0 in the
    continuum). Also the vertical forces."""
    F, _, _ = body_force(pan, phi, U, rho, copies)
    out = dict(raw=float(-F[0]), Fz=float(F[2]))
    if phi_db is not None:
        Fdb, _, _ = body_force(pan, phi_db, U, rho, copies)
        out.update(dalembert=float(-Fdb[0]), corrected=float(-(F[0] - Fdb[0])),
                   Fz_db=float(Fdb[2]))
    return out
