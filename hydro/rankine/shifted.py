"""
Steady linear free surface with the shifted Numerov operator (decision D1 of
learn/meta/WAVEBEM_PORT_DESIGN.md, stage B3; analysis and results in
learn/meta/RANKINE_STAGE1_2026-10-06.md section 8). Dense and FFT-accelerated
(GMRES on the GPU) solves of the same discrete system.

Why a new operator (summary of the stage note, section 8.2). The steady
condition U^2 phi_xx + g phi_z = 0 is solved as a marching problem in x: the
radiation condition (no waves upstream) comes from the free-surface rows
using only points at or upstream of their own index. Dawson-type rows
sum_m w_m phi_{r-m} = (U^2/g)^-1 sigma_r are linear multistep methods for
y'' = f with f taken at the newest point. For the lateral Fourier mode beta
the coupling with the Rankine single layer acts like y'' = -omega^2 y with
omega^2 h^2 = 2 pi dx^2 / (k S(alpha, beta)) up to ~1 on the grid. Dahlquist
(1978, BIT 18:133) showed that an unconditionally stable multistep method for
y'' = f has order <= 2: the five-point upstream operator (order 3) grows for
every omega h, the 6- and 7-point ones are stable only for omega^2 h^2 <
1e-4 and < 0.35 (studies/rankine_d1.py), i.e. they blow up on the short
diverging waves of any practical grid, and the stable four-point operator
(order 2) damps 4 %/wavelength with a 3.7 % wavelength error.

The shift (Jensen, Mi & Soding 1986; Raven 1996) separates the radiation
condition from the interior accuracy: the free-surface condition of row r is
imposed at the CENTRE c = r - 1 (one panel upstream), with symmetric
(central) formulas there. Every factor e^{i alpha dx} of the shift cancels
between the two sides of the condition, so the interior symbol is real: no
numerical damping or growth at any wavenumber, and the waves keep their
amplitude. The count of rows (one row upstream of the first panel column,
none at the last one) puts both conditions of the second-order marching
problem at the upstream end, which is the radiation condition. Rows
(per grid row; column 0 most upstream, x decreasing with the index; centre
c = r - 1):

  (1/12)(sigma_r + 10 sigma_{r-1} + sigma_{r-2})
        = k w_c [ (phi_{c-1} - 2 phi_c + phi_{c+1})/dx^2
                  - nu_c (phi_{c-1} - phi_{c+1})/(2 dx) ]                (N)

  k = U^2/g, sigma = dphi/dn = -phi_z the free-surface source density,
  phi the TOTAL potential on z = 0 (perturbation phi_F plus any prescribed
  external part), w the entry ramp, nu the downstream beach (steady.py).
  (N) is Numerov's formula for k phi_xx = sigma (+ the Rayleigh term),
  error (dx^4/240) phi^(6), interval of periodicity omega^2 h^2 < 6
  (Lambert & Watson 1976); the grid needs < ~1 (rankine_d1.py), so every
  resolved wave is neutral. sigma_{-1} = sigma_{-2} = 0 (no sources upstream
  of the grid); phi_{-1} at c = 0 by linear extrapolation (w ~ 1e-4 there).

Entry-zone blend. Where the ramp w is small the coupling is weak (effective
omega^2 h^2 ~ 1/w), outside Numerov's interval (the mass polynomial
1 + 10/q + 1/q^2 has a root q = -9.9: growth x10 per cell; an entry zone
twice as long gave R/R_H = -7.3 in the exploration run B3-E5). The mass
weights are therefore blended, a = (1 - l)(1, 10, 1)/12 + l (1, 0, 0),
l = 1 - w: for l >= 0.4 the row is unconditionally stable (the
characteristic polynomial (1 + z a0) q^2 + (z a1 - 2) q + (1 + z a2) has
p(-1) = 4 + z (5 l - 2)/3 > 0, p(1) = z > 0 and root product
(1 + z a2)/(1 + z a0) < 1), and for l < 0.4 (w > 0.6) the rows' z/w <
1.35/0.6 < 2.3 (registered grids) is inside the interval 12/(2 - 5 l) >= 6.
In the interior l = 0 (pure Numerov). In the entry zone the field is
non-wave, so the first-order damping of the l -> 1 rows does no harm.
Resolution guard (blend_weights): on grids that do not resolve the
wavelength (grid z_max > 4.8; z_max is proportional to 1/n, z_max n = 16.2
with the box correction, so fewer than about 3.4 panels per wavelength)
the interior blend is raised to keep every row stable (cap 0.4); inactive on
the registered grids. It makes the discrete operator speed-dependent on a
fixed grid (not a force-formula switch, G1 untouched), so since the B3
review every solve reports info["guard_l"] (largest interior blend weight)
and warns when it is > 0 unless allow_guard=True (only check S4).
Production use: choose the free-surface grid per lambda0 (dx = lambda0/n,
n >= 20), never one grid for a speed sweep, so the guard stays inactive.

Box correction of the panel density. Constant-strength panels at centroid
collocation filter the source density by the panel box: lattice symbol
(2 pi/kappa) sinc(alpha dx/2) sinc(beta dy/2) + aliases (O(theta^3)),
i.e. a relative error -kappa^2 dx^2/24 that dominates the dispersion error
of (N) (0.18 % of the transverse wavenumber, 3 % at kappa dx = 0.84, at 30
panels per wavelength). The panels therefore carry
sigma' = sigma - (1/24)(dx^2 d_xx + dy^2 d_yy) sigma (second differences;
even mirror at y = 0, no correction in the first/last column and the last
row), which cancels the sinc to O(kappa^4 dx^4): coupled wavenumber error
-0.012 % (transverse) to -0.063 % (55 deg) at 30 panels per wavelength
(rankine_d1.py). sigma' is the density of every free-surface panel in
every influence (body rows, free-surface potential, Lagally forces).

Formulation (unknowns phi_B on the body panels, sigma on the free-surface
panels; rigid-lid and port/starboard images as in steady.py):
  body rows   (2 pi I - D_BB) phi_B + S_BF sigma' = -S_BB dphi/dn_B
  FS rows     A sigma - k Bk (phi_F + phi_ext) = 0,
              4 pi phi_F = D_FB phi_B - S_FB dphi/dn_B - S_FF sigma'
(the dipoles of the free-surface panels cancel with their rigid-lid images,
so phi on z = 0 enters only through the representation). S_FF on the
uniform grid is translation invariant: S_FF sigma' is a 2-D convolution of
the y-even extension of sigma' with K(m, n) = 2 S_panel(m dx, n dy) (2 for
the rigid-lid image), done by FFT (FSLattice); the body blocks are dense.
"""
import warnings

import numpy as np
import scipy.sparse as sp
import torch

from .bie import ImageOperator
from .influence import influence
from .panels import Panels
from .steady import point_source_fields
from .krylov import gmres


def _torch_sparse(A, dev):
    A = sp.coo_matrix(A)
    idx = torch.as_tensor(np.vstack([A.row, A.col]), dtype=torch.long, device=dev)
    val = torch.as_tensor(A.data, dtype=torch.float64, device=dev)
    return torch.sparse_coo_tensor(idx, val, A.shape).coalesce()


def ramp_and_beach(fs, grid, entry=None, beach=None):
    """Entry ramp w and beach damping nu per free-surface panel, the same
    definitions as steady.SteadyLinearFS (C2 ramp over `entry` from the
    upstream edge; nu = nu_max s^2 over the last beach['length'],
    nu_max = 3 decay / length)."""
    x = fs.c[:, 0]
    dx = grid["dx"]
    x_up, x_dn = x.max() + 0.5 * dx, x.min() - 0.5 * dx
    nu = np.zeros(len(fs))
    if beach is not None:
        Lb = beach["length"]
        s = np.clip((x_dn + Lb - x) / Lb, 0.0, 1.0)
        nu = 3.0 * beach.get("decay", 6.0) / Lb * s * s
    w = np.ones(len(fs))
    if entry:
        s = np.clip((x_up - x) / entry, 0.0, 1.0)
        w = s ** 3 * (10 - 15 * s + 6 * s * s)
    return w, nu


def lattice_zmax(dx, dy, k, box=True, P=60, ng=7):
    """Largest z = 2 pi dx^2 / (k S C) of the grid (module docstring; S the
    lattice symbol of the panel single layer, C the box-correction symbol),
    sampled on the outer part of the Brillouin zone where S C is smallest.
    Numpy lattice sum, |p|, |q| <= P (alternating tails, converged to <1e-4
    relative, studies/rankine_d1.py)."""
    p = np.arange(-P, P + 1)
    zm = 0.0
    for fa in np.linspace(0.3, 1.0, ng):
        for fb in np.linspace(0.0, 1.0, ng):
            al, be = fa * np.pi / dx, fb * np.pi / dy
            ap = al + 2 * np.pi * p / dx
            bq = be + 2 * np.pi * p / dy
            sa = np.sinc(ap * dx / (2 * np.pi))
            sb = np.sinc(bq * dy / (2 * np.pi))
            S = np.sum(2 * np.pi / np.hypot(ap[:, None], bq[None, :]) * sa[:, None] * sb[None, :])
            C = 1 + (4 * np.sin(al * dx / 2) ** 2 + 4 * np.sin(be * dy / 2) ** 2) / 24 if box else 1.0
            zm = max(zm, 2 * np.pi * dx * dx / (k * S * C))
    return float(zm)


def blend_weights(w_cols, zgrid=None, blend=True, safety=1.25):
    """Blend l per centre column c (module docstring): l = 1 - w (entry
    zone); with zgrid (the grid's largest z) also l >= (2 - 12 w/(safety
    zgrid))/5, capped at 0.4 (unconditionally stable rows): a resolution
    guard that is inactive whenever zgrid < 4.8 (about 3.4 panels per
    wavelength or more; corrected after the B3 review, the first text said
    13) and keeps under-resolved grids (e.g. very low Fn on a coarse grid,
    check S4) stable at the price of damping waves they cannot resolve
    anyway."""
    w_cols = np.asarray(w_cols, float)
    lam = 1.0 - w_cols if blend else np.zeros_like(w_cols)
    if zgrid is not None:
        lam = np.maximum(lam, np.clip((2.0 - 12.0 * w_cols / (safety * zgrid)) / 5.0, 0.0, 0.4))
    return lam


def mass_operator(nx, ny, lam, mass="numerov"):
    """A of (N): row r of every grid row, a = (1 - l)(1, 10, 1)/12 +
    l (1, 0, 0) on sigma_r, sigma_{r-1}, sigma_{r-2}, l = lam[r - 1]
    (l = 1 for the row of the virtual centre r = 0: sigma_0 = 0)."""
    am = np.array([1.0, 10.0, 1.0]) / 12 if mass == "numerov" else np.array([0.0, 1.0, 0.0])
    lr = np.concatenate([[1.0], np.asarray(lam, float)[:nx - 1]])           # per row r
    Ax = (sp.diags((1 - lr) * am[0] + lr, 0, shape=(nx, nx))
          + sp.diags(((1 - lr) * am[1])[1:], -1, shape=(nx, nx))
          + sp.diags(((1 - lr) * am[2])[2:], -2, shape=(nx, nx)))
    return sp.csr_matrix(sp.kron(sp.identity(ny), Ax))


def shifted_operators(nx, ny, dx, dy, w, nu, box=True):
    """Bk (second difference + Rayleigh term, times w, rows shifted to the
    centre c = r - 1; B = k Bk acts on phi) and C (box correction) of (N)
    on the fs_rect ordering k = j nx + i. w, nu per panel; they depend on x
    only (checked)."""
    w = np.asarray(w, float).reshape(ny, nx)
    nu = np.asarray(nu, float).reshape(ny, nx)
    assert np.allclose(w, w[:1]) and np.allclose(nu, nu[:1]), "ramp/beach must depend on x only"
    wc, nc = w[0], nu[0]
    rows, cols, vals = [], [], []
    for c in range(0, nx - 1):
        r = c + 1
        if c == 0:
            # phi_{-1} = 2 phi_0 - phi_1: second difference 0, first (phi_0 - phi_1)/dx
            if nc[0] != 0.0:
                rows += [r, r]
                cols += [0, 1]
                vals += [-wc[0] * nc[0] / dx, wc[0] * nc[0] / dx]
            continue
        for cc, v in ((c - 1, 1 / dx ** 2 - nc[c] / (2 * dx)), (c, -2 / dx ** 2),
                      (c + 1, 1 / dx ** 2 + nc[c] / (2 * dx))):
            rows.append(r)
            cols.append(cc)
            vals.append(wc[c] * v)
    Bx = sp.csr_matrix((vals, (rows, cols)), shape=(nx, nx))
    Bk = sp.csr_matrix(sp.kron(sp.identity(ny), Bx))
    N = nx * ny
    if not box:
        return Bk, sp.identity(N, format="csr")
    d2x = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(nx, nx)).tolil()
    d2x[0, :] = 0.0
    d2x[nx - 1, :] = 0.0                                          # no correction in the end columns
    d2y = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(ny, ny)).tolil()
    d2y[0, 0] = 1.0                                               # even mirror at y = 0
    d2y[ny - 1, :] = 0.0                                          # none in the last (outer) row
    C = (sp.identity(N) + sp.kron(sp.identity(ny), d2x.tocsr()) / 24
         + sp.kron(d2y.tocsr(), sp.identity(nx)) / 24)
    return Bk, sp.csr_matrix(C)


class FSLattice:
    """Single layer of the uniform free-surface grid with its rigid-lid (z)
    and port/starboard (y) images: (S_FF s)_i = sum_j [S_ij(panel) +
    S_ij(z image) + y images] s_j, with all panels the same rectangle.
    Kernel K(m, n) = 2 S_ref(m dx, n dy) (exact panel integral of
    influence.py at the lattice offsets; 2 = panel + coincident z image).
    apply(): FFT convolution of the y-even extension (circular size
    2 nx x 4 ny, which is wrap-free for every needed offset); dense():
    the same matrix entry by entry (verification, small N)."""

    def __init__(self, grid, device=None):
        self.nx, self.ny = grid["nx"], grid["ny"]
        self.dx, self.dy = grid["dx"], grid["dy"]
        self.N = self.nx * self.ny
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dx, dy, nx, ny = self.dx, self.dy, self.nx, self.ny
        ref = Panels(np.array([[[dx / 2, -dy / 2, 0.0], [-dx / 2, -dy / 2, 0.0],
                                [-dx / 2, dy / 2, 0.0], [dx / 2, dy / 2, 0.0]]]))
        m = np.arange(nx)
        n = np.arange(2 * ny)
        Mg, Ng = np.meshgrid(m * dx, n * dy, indexing="ij")
        P = np.stack([Mg.ravel(), Ng.ravel(), np.zeros(Mg.size)], 1)
        S, _ = influence(P, ref, device=self.device, as_numpy=False)
        self.K = (2.0 * S[:, 0]).reshape(nx, 2 * ny)            # K(|m|, |n|)
        del S
        self.Px, self.Py = 2 * nx, 4 * ny
        Kf = torch.zeros((self.Px, self.Py), dtype=torch.float64, device=self.device)
        im = torch.arange(-(nx - 1), nx, device=self.device)
        jn = torch.arange(-(2 * ny - 1), 2 * ny, device=self.device)
        Kf[(im % self.Px)[:, None], (jn % self.Py)[None, :]] = self.K[im.abs()[:, None], jn.abs()[None, :]]
        self.Khat = torch.fft.rfft2(Kf)
        del Kf

    def apply(self, s):
        """S_FF s for s (N,) torch on the device (fs_rect ordering)."""
        nx, ny = self.nx, self.ny
        a = s.reshape(ny, nx).T                                   # [i, j]
        ext = torch.cat([torch.flip(a, dims=[1]), a], dim=1)      # J = ny + j and ny - 1 - j
        Y = torch.fft.irfft2(torch.fft.rfft2(ext, s=(self.Px, self.Py)) * self.Khat, s=(self.Px, self.Py))
        return Y[:nx, ny:2 * ny].T.reshape(-1).contiguous()

    def dense(self, chunk=1024):
        """Dense S_FF from the kernel, entry by entry (torch; verification)."""
        nx, N, dev = self.nx, self.N, self.device
        out = torch.empty((N, N), dtype=torch.float64, device=dev)
        cols = torch.arange(N, device=dev)
        ic, jc = cols % nx, cols // nx
        for r0 in range(0, N, chunk):
            rows = torch.arange(r0, min(N, r0 + chunk), device=dev)
            ir, jr = rows % nx, rows // nx
            di = (ir[:, None] - ic[None, :]).abs()
            out[r0:r0 + len(rows)] = (self.K[di, (jr[:, None] - jc[None, :]).abs()]
                                      + self.K[di, jr[:, None] + jc[None, :] + 1])
            del di
        return out

    def near(self, p):
        """Sparse near part of S_FF: entries with |di| <= p and |dj| <= p
        (direct) or j + j' + 1 <= p (y image), scipy CSR."""
        nx, ny = self.nx, self.ny
        Kc = self.K.cpu().numpy()
        rows, cols, vals = [], [], []
        for j in range(ny):
            for i in range(nx):
                r = j * nx + i
                for jj in range(max(0, j - p), min(ny, j + p + 1)):
                    for ii in range(max(0, i - p), min(nx, i + p + 1)):
                        v = Kc[abs(i - ii), abs(j - jj)]
                        if j + jj + 1 <= p:
                            v += Kc[abs(i - ii), j + jj + 1]
                        rows.append(r)
                        cols.append(jj * nx + ii)
                        vals.append(v)
        return sp.csr_matrix((vals, (rows, cols)), shape=(self.N, self.N))


class SteadyShiftedFS:
    """Body (Panels or None) + uniform free-surface grid (steady.fs_rect),
    shifted Numerov free-surface operator (module docstring). method
    "dense": assemble the full matrix and LU (verification, N <~ 2e4);
    "fft": GMRES with FFT products for S_FF and dense body blocks, block
    preconditioner (free surface: exact in x, circulant in y, one LU per
    lateral Fourier mode on the device; body: dense LU), see
    preconditioner()."""

    def __init__(self, body, fs, grid, device=None, entry=None, beach=None, box=True,
                 blend=True, mass="numerov", method="fft", near_p=2, keep_SFB=None, guard=True,
                 allow_guard=False):
        """allow_guard: the resolution guard may be active without a warning
        (only rows registered to run with it, check S4; B3 review)."""
        self.body, self.fs, self.grid = body, fs, grid
        self.allow_guard = allow_guard
        self.NB = 0 if body is None else len(body)
        self.NF = len(fs)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.method = method
        self.box = box
        self.w, self.nu = ramp_and_beach(fs, grid, entry, beach)
        self.Bk, self.C = shifted_operators(grid["nx"], grid["ny"], grid["dx"], grid["dy"],
                                            self.w, self.nu, box)
        dev = self.device
        self._Bk, self._C = (_torch_sparse(M, dev) for M in (self.Bk, self.C))
        self.mass, self.blend, self.guard = mass, blend, guard
        self._k = None
        self.lat = FSLattice(grid, dev)
        self.near_p = near_p
        self.S_FB = None
        if body is not None:
            ob = ImageOperator(body, axes=(1, 2), device=dev, parities=[(1, 1)], on_device=True)
            self.S_BB, self.D_BB = ob.S((1, 1)), ob.D((1, 1))
            del ob
            self._body_blocks(keep_SFB)

    def _body_blocks(self, keep_SFB=None, chunk=16384):
        """D_FB (body panels with images at the free-surface centroids,
        stored), S_FB (stored only if small or keep_SFB; otherwise S_FB n_x
        is kept and other right-hand sides recompute it by chunks) and S_BF
        (free-surface panels with images at the body centroids). Built in
        chunks of free-surface points/panels, so the device memory is the
        stored blocks plus one chunk (N_F N_B x 8 bytes per stored block)."""
        dev, body, fs = self.device, self.body, self.fs
        NB, NF = self.NB, self.NF
        if keep_SFB is None:
            keep_SFB = NF * NB * 8 < 1.5e9
        self.D_FB = torch.empty((NF, NB), dtype=torch.float64, device=dev)
        if keep_SFB:
            self.S_FB = torch.empty((NF, NB), dtype=torch.float64, device=dev)
        nx_t = torch.as_tensor(body.n[:, 0], device=dev)
        self._SFB_nx = torch.empty(NF, dtype=torch.float64, device=dev)
        for r0 in range(0, NF, chunk):
            r1 = min(NF, r0 + chunk)
            op = ImageOperator(body, axes=(1, 2), xf=fs.c[r0:r1], device=dev, parities=[(1, 1)], on_device=True)
            S, D = op.S((1, 1)), op.D((1, 1))
            self.D_FB[r0:r1] = D
            self._SFB_nx[r0:r1] = S @ nx_t
            if keep_SFB:
                self.S_FB[r0:r1] = S
            del op, S, D
        # free-surface panels at the body points: the rigid-lid image of a
        # panel on z = 0 is the same panel (normal reversed), so S doubles
        # and D cancels; plus the port/starboard image
        self.S_BF = torch.empty((NB, NF), dtype=torch.float64, device=dev)
        for c0 in range(0, NF, chunk):
            c1 = min(NF, c0 + chunk)
            sub = fs.subset(np.arange(c0, c1))
            S1, _ = influence(body.c, sub, device=dev, as_numpy=False)
            S2, _ = influence(body.c, sub.reflect(1), device=dev, as_numpy=False)
            self.S_BF[:, c0:c1] = 2.0 * (S1 + S2)
            del S1, S2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _SFB_apply(self, pn):
        """S_FB pn (torch)."""
        if self.S_FB is not None:
            return self.S_FB @ pn
        nx_t = torch.as_tensor(self.body.n[:, 0], device=self.device)
        c = float(pn @ nx_t / (nx_t @ nx_t))
        if float(torch.linalg.norm(pn - c * nx_t)) <= 1e-14 * max(1.0, float(torch.linalg.norm(pn))):
            return c * self._SFB_nx
        out = torch.empty(self.NF, dtype=torch.float64, device=self.device)
        for r0 in range(0, self.NF, 16384):
            r1 = min(self.NF, r0 + 16384)
            op = ImageOperator(self.body, axes=(1, 2), xf=self.fs.c[r0:r1], device=self.device,
                               parities=[(1, 1)], on_device=True)
            out[r0:r1] = op.S((1, 1)) @ pn
            del op
        return out

    # ------------------------------------------------------------ pieces
    def set_speed(self, k):
        """Mass operator A for k = U^2/g (the resolution guard of
        blend_weights depends on the grid's z, which scales like 1/k)."""
        if self._k == k:
            return
        nx, ny = self.grid["nx"], self.grid["ny"]
        self.zgrid = lattice_zmax(self.grid["dx"], self.grid["dy"], k, self.box)
        self.lam = blend_weights(self.w[:nx], self.zgrid if self.guard else None, self.blend)
        # state of the resolution guard (B3 review): largest blend weight in
        # the interior columns (w = 1, outside the entry zone), where the
        # production rows are pure Numerov (l = 0) unless the guard is active
        wc = np.asarray(self.w[:nx], float)
        interior = wc >= 1.0 - 1e-12
        self.guard_l = float(self.lam[interior].max()) if interior.any() else 0.0
        self.A = mass_operator(nx, ny, self.lam, self.mass)
        self._A = _torch_sparse(self.A, self.device)
        self._k = k

    def _sp(self, M, v):
        return torch.sparse.mm(M, v[:, None])[:, 0]

    def matvec(self, x, k):
        self.set_speed(k)
        NB = self.NB
        phB, sig = x[:NB], x[NB:]
        sp_ = self._sp(self._C, sig)
        conv = self.lat.apply(sp_)
        phiF4 = -conv                                   # 4 pi phi_F (body part added below)
        out = torch.empty_like(x)
        if NB:
            phiF4 = phiF4 + self.D_FB @ phB
            out[:NB] = 2 * np.pi * phB - self.D_BB @ phB + self.S_BF @ sp_
        out[NB:] = self._sp(self._A, sig) - (k / (4 * np.pi)) * self._sp(self._Bk, phiF4)
        return out

    def rhs(self, k, phin_body=None, phi_ext=None):
        dev = self.device
        NB = self.NB
        b = torch.zeros(NB + self.NF, dtype=torch.float64, device=dev)
        ext4 = torch.zeros(self.NF, dtype=torch.float64, device=dev)
        if phi_ext is not None:
            ext4 += 4 * np.pi * torch.as_tensor(np.asarray(phi_ext, float), device=dev)
        if NB:
            pn = torch.as_tensor(np.asarray(phin_body, float), device=dev)
            b[:NB] = -(self.S_BB @ pn)
            self._SFBpn = self._SFB_apply(pn)
            ext4 = ext4 - self._SFBpn
        b[NB:] = (k / (4 * np.pi)) * self._sp(self._Bk, ext4)
        return b

    def dense_matrix(self, k):
        """Full system matrix (torch, (NB+NF)^2) by explicit assembly:
        S_FF from FSLattice.dense (entry by entry), sparse products by
        rows. For verification at overlap sizes."""
        NB, NF, dev = self.NB, self.NF, self.device
        self.set_speed(k)
        M = torch.zeros((NB + NF, NB + NF), dtype=torch.float64, device=dev)
        SFF = self.lat.dense()
        SC = torch.sparse.mm(self._C.t().coalesce(), SFF.T.contiguous()).T     # S_FF C (S_FF symmetric not assumed)
        del SFF
        M[NB:, NB:] = (k / (4 * np.pi)) * torch.sparse.mm(self._Bk, SC)
        M[NB:, NB:] += self._A.to_dense()
        if NB:
            M[:NB, :NB] = -self.D_BB
            M[:NB, :NB].diagonal().add_(2 * np.pi)
            M[:NB, NB:] = torch.sparse.mm(self._C.t().coalesce(), self.S_BF.T.contiguous()).T
            M[NB:, :NB] = -(k / (4 * np.pi)) * torch.sparse.mm(self._Bk, self.D_FB)
        del SC
        return M

    def preconditioner(self, k, kind="ycirc"):
        """Block preconditioner (body block 2 pi I - D_BB by dense LU on the
        device). Free-surface block:
        kind "ycirc" (default): the exact x-structure with a circulant
          approximation in y. A, Bk depend on x only (ramp and beach are
          functions of x), so on the y-even extension (period 2 ny) the
          operator separates into one nx x nx matrix per lateral Fourier mode
          q, O_q = A_x + (k/4pi) Bk_x T_q (C_x + c_q I), T_q[i, i'] the y-DFT
          of K(|i - i'|, n) (offsets wrapped to the shorter distance),
          c_q = (2 - 2 cos(pi q/ny))/24; all O_q are LU-factored at once on
          the device (ny + 1 distinct modes). Exact except for the lateral
          truncation and the box correction's edge rows, which GMRES absorbs.
        kind "near": A + (k/4pi) Bk S_near C with the near-field part of
          S_FF (|di|, |dj| <= near_p), sparse LU on the CPU (SuperLU; fill
          grows like N^1.5, kept for the record)."""
        dev = self.device
        NB = self.NB
        self.set_speed(k)
        luB = None
        if NB:
            MB = -self.D_BB.clone()
            MB.diagonal().add_(2 * np.pi)
            luB = torch.linalg.lu_factor(MB)
            del MB
        if kind == "near":
            import scipy.sparse.linalg as spla
            On = (self.A + (k / (4 * np.pi)) * (self.Bk @ self.lat.near(self.near_p) @ self.C)).tocsc()
            lu = spla.splu(On, permc_spec="COLAMD")

            def fs_apply(v):
                return torch.as_tensor(lu.solve(v.cpu().numpy()), device=dev)
        else:
            nx, ny = self.grid["nx"], self.grid["ny"]
            lat = self.lat
            n2 = 2 * ny
            nn = torch.arange(n2, device=dev)
            Kt = lat.K[:, torch.minimum(nn, n2 - nn)]                              # (nx, 2ny), wrapped
            Kq = torch.fft.fft(Kt, dim=1).real[:, :ny + 1]                         # (nx, ny+1)
            ii = torch.arange(nx, device=dev)
            T = Kq[(ii[:, None] - ii[None, :]).abs()].permute(2, 0, 1).contiguous()  # (ny+1, nx, nx)
            del Kq, Kt
            Ax = torch.as_tensor(self.A[:nx, :nx].toarray(), device=dev)
            Bx = torch.as_tensor(self.Bk[:nx, :nx].toarray(), device=dev)
            Cx = np.eye(nx)
            cq = torch.zeros(ny + 1, dtype=torch.float64, device=dev)
            if self.box:
                for i in range(1, nx - 1):
                    Cx[i, i - 1:i + 2] += np.array([-1.0, 2.0, -1.0]) / 24
                cq = (2 - 2 * torch.cos(np.pi * torch.arange(ny + 1, device=dev, dtype=torch.float64) / ny)) / 24
            Cxq = (torch.as_tensor(Cx, device=dev)[None]
                   + cq[:, None, None] * torch.eye(nx, device=dev, dtype=torch.float64)[None])
            O = T @ Cxq
            del T, Cxq
            O = (k / (4 * np.pi)) * (Bx[None] @ O) + Ax[None]
            LU, piv = torch.linalg.lu_factor(O)
            del O
            d = torch.arange(ny + 1, device=dev)
            dm = (n2 - d) % n2                                                      # partner mode 2ny - d

            def fs_apply(v):
                a = v.reshape(ny, nx)
                ext = torch.cat([torch.flip(a, dims=[0]), a], dim=0)               # (2ny, nx), J = ny + j
                R = torch.fft.fft(ext, dim=0)                                       # (2ny, nx) complex
                rhs = torch.stack([R[d].real, R[d].imag, R[dm].real, R[dm].imag], dim=2)   # (ny+1, nx, 4)
                X = torch.linalg.lu_solve(LU, piv, rhs)
                Xc = torch.empty_like(R)
                Xc[dm] = torch.complex(X[..., 2], X[..., 3])
                Xc[d] = torch.complex(X[..., 0], X[..., 1])
                out = torch.fft.ifft(Xc, dim=0).real[ny:]
                return out.reshape(-1).contiguous()

        def apply(v):
            out = torch.empty_like(v)
            if NB:
                out[:NB] = torch.linalg.lu_solve(*luB, v[:NB, None])[:, 0]
            out[NB:] = fs_apply(v[NB:])
            return out
        return apply

    def diagonal(self, k):
        """Exact diagonal of the system matrix (the stencils of A, Bk, C are
        short, so S_near with p = 3 contains every entry that reaches it)."""
        self.set_speed(k)
        d = (self.A + (k / (4 * np.pi)) * (self.Bk @ self.lat.near(3) @ self.C)).diagonal()
        out = torch.as_tensor(d, device=self.device)
        if self.NB:
            out = torch.cat([2 * np.pi - self.D_BB.diagonal(), out])
        return out

    # ------------------------------------------------------------- solve
    def solve(self, U, g, phin_body=None, phi_ext=None, method=None, tol=1e-12, restart=300,
              maxiter=6000, precond="ycirc", verbose=False, fs_forcing=None, floor=None, strict=True):
        """Returns dict(phi_B, sigma, sigma_eff (= C sigma, the panel
        densities), phi_F (free-surface potential without phi_ext), info).
        phi_ext: prescribed potential (point values at the free-surface
        centroids, rigid-lid image included) whose total with phi_F
        satisfies (N). fs_forcing: an explicit addition to the right-hand
        side of the free-surface rows (check S4b).

        Acceptance (B3 review): the true relative residual |b - A x|/|b| is
        reported for every solve (info["relres"], also for method "dense")
        and must be <= floor (default 100 tol = 1e-10, registered in the
        stage note section 8.8); with strict=True (default) a larger
        residual raises RuntimeError, so an unconverged solve can never be
        used silently (a verification row then turns into FAIL).
        info["guard_l"] is the largest blend weight in the interior columns:
        0 for the production rows; > 0 means the resolution guard replaced
        them by the damped first-order blend (grid too coarse for lambda0),
        which warns unless the solver was built with allow_guard=True."""
        method = method or self.method
        floor = 100.0 * tol if floor is None else float(floor)
        k = U * U / g
        b = self.rhs(k, phin_body, phi_ext)
        if fs_forcing is not None:
            b[self.NB:] += torch.as_tensor(np.asarray(fs_forcing, float), device=self.device)
        info = {}
        if method == "dense":
            M = self.dense_matrix(k)
            x = torch.linalg.solve(M, b)
            rel = float(torch.linalg.norm(b - M @ x) / torch.linalg.norm(b))
            del M
            info = dict(method="dense", relres=rel, iters=0, converged=rel <= tol, stagnated=False,
                        accepted=rel <= floor, floor=floor)
        else:
            P = None
            if precond in ("near", "ycirc"):
                P = self.preconditioner(k, precond)
            elif precond == "jacobi":
                d = self.diagonal(k)
                P = (lambda v: v / d)
            x, info = gmres(lambda v: self.matvec(v, k), b, tol=tol, restart=restart, maxiter=maxiter,
                            precond=P, verbose=verbose, floor=floor)
            info["method"] = "fft"
            info.pop("history", None)
        info["guard_l"] = self.guard_l
        info["zgrid"] = self.zgrid
        if self.guard_l > 0 and not self.allow_guard:
            warnings.warn(f"resolution guard active: interior blend l = {self.guard_l:.3f} (grid z_max "
                          f"{self.zgrid:.2f} > 4.8 at U = {U:g}); the rows are the damped first-order blend, "
                          "not the production Numerov rows -- choose the grid per lambda0", RuntimeWarning)
        if strict and not info["accepted"]:
            raise RuntimeError(f"steady free-surface solve not accepted: true relative residual "
                               f"{info['relres']:.2e} > floor {floor:.0e} ({info.get('iters')} iterations, "
                               f"method {info['method']})")
        NB = self.NB
        sig = x[NB:]
        sp_ = self._sp(self._C, sig)
        phiF4 = -self.lat.apply(sp_)
        if NB:
            phiF4 = phiF4 + self.D_FB @ x[:NB] - self._SFBpn
        return dict(phi_B=x[:NB].cpu().numpy(), sigma=sig.cpu().numpy(), sigma_eff=sp_.cpu().numpy(),
                    phi_F=(phiF4 / (4 * np.pi)).cpu().numpy(), x=x, info=info)

    def residual(self, x, U, g, phin_body=None, phi_ext=None):
        k = U * U / g
        b = self.rhs(k, phin_body, phi_ext)
        return float(torch.linalg.norm(self.matvec(x, k) - b) / torch.linalg.norm(b))

    def release(self):
        for a in ("S_BB", "D_BB", "S_FB", "D_FB", "S_BF", "_SFB_nx", "lat", "_A", "_Bk", "_C"):
            if hasattr(self, a):
                setattr(self, a, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ------------------------------------------------------------ evaluation
def phi_x4(phi, nx, ny, dx):
    """d phi / dx on the fs_rect grid (x decreasing with the column index):
    fourth-order central differences in the interior, second order in the
    columns next to the ends, one-sided at the ends."""
    P = np.asarray(phi, float).reshape(ny, nx)
    px = np.zeros_like(P)
    px[:, 2:-2] = (-P[:, :-4] + 8 * P[:, 1:-3] - 8 * P[:, 3:-1] + P[:, 4:]) / (12 * dx)
    px[:, 1] = (P[:, 0] - P[:, 2]) / (2 * dx)
    px[:, -2] = (P[:, -3] - P[:, -1]) / (2 * dx)
    px[:, 0] = (P[:, 0] - P[:, 1]) / dx
    px[:, -1] = (P[:, -2] - P[:, -1]) / dx
    return px.reshape(-1)


def lagally_resistance(fs, sigma_eff, Q, X, rho=1.0, device=None):
    """R = rho sum_k Q_k u_x(X_k) for prescribed sources (body-free case):
    u = velocity of the free-surface panels (densities sigma_eff, doubled by
    the rigid-lid image, with port/starboard images) plus the rigid-lid
    images of the sources themselves."""
    _, _, G1 = influence(X, fs, want_grad=True, device=device)
    _, _, G2 = influence(X, fs.reflect(1), want_grad=True, device=device)
    grad = -np.einsum("kjc,j->kc", G1 + G2, 2.0 * np.asarray(sigma_eff)) / (4 * np.pi)
    Xi = np.asarray(X, float).copy()
    Xi[:, 2] *= -1
    _, gi, _ = point_source_fields(Q, Xi, np.asarray(X, float))
    return rho * float(np.sum(np.asarray(Q) * (grad + gi)[:, 0]))
