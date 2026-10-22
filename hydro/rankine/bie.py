"""
Boundary-integral operators with symmetry images, and the unbounded / image
radiation solves used by stage 1 (design section 8.4, step R1).

Green's third identity with G = -1/(4 pi r) and normals INTO the fluid gives
for a collocation point x_i on a smooth part of the boundary S (derivation in
the stage-1 note learn/meta/RANKINE_STAGE1_2026-10-06.md, section 2):

    2 pi phi_i - sum_j D_ij phi_j + sum_j S_ij (dphi/dn)_j = 0          (E)

with S_ij, D_ij the panel integrals of influence.py (self term D_ii = 0, the
2 pi is the jump). For a function harmonic INSIDE a closed surface (normals
out of that interior), the same identity reads
    2 pi phi_i + sum_j D_ij phi_j - sum_j S_ij (dphi/dn)_j = 0.          (I)

Source formulation (used for the forward-speed problem): phi(x) =
-(1/4 pi) sum_j sigma_j S_j(x), grad phi = -(1/4 pi) sum_j sigma_j grad S_j,
and on the fluid side of panel i dphi/dn = sigma_i / 2 + principal value.

Images: `syms` is a tuple of reflection axes (0 = x, 1 = y, 2 = z) and the
problem's parity under each (+1 even, -1 odd). The operators are summed over
all 2^k reflections with the product of the parities.
"""
import itertools
import numpy as np
import torch

from .influence import influence


class ImageOperator:
    """Influence matrices of `pan` (and its images) at its own centroids or
    at field points xf. With `parities` given (list of parity tuples), the
    images are accumulated per parity as they are computed (memory 2-5 N^2
    per parity); otherwise every image block is kept and any parity can be
    combined later. Access with .S(par), .D(par), .G(par)."""

    def __init__(self, pan, axes=(), xf=None, want_grad=False, device=None,
                 parities=None, on_device=False):
        """on_device=True keeps the accumulated matrices as torch tensors on
        `device` (no host copy; used by the steady solver to keep host RAM
        at a few hundred MB for N ~ 1e4). Default: numpy arrays."""
        self.pan = pan
        self.axes = tuple(axes)
        own = xf is None
        x = pan.c if own else np.asarray(xf, float)
        self.combos = list(itertools.product((0, 1), repeat=len(self.axes)))
        self.keep = parities is None
        self.mats = {}
        self.acc = {}
        for cmb in self.combos:
            p = pan
            for ax, use in zip(self.axes, cmb):
                if use:
                    p = p.reflect(ax)
            identity = not any(cmb)
            si = np.arange(len(pan)) if (own and identity) else None
            if on_device and parities is not None and len(parities) == 1 and self.acc:
                # one parity: add this image straight into the accumulators
                par = tuple(parities[0])
                influence(x, p, want_grad=want_grad, device=device,
                          acc=self.acc[par], alpha=self._sign(par, cmb))
                continue
            m = influence(x, p, want_grad=want_grad, self_idx=si, device=device,
                          as_numpy=not on_device)
            if self.keep:
                self.mats[cmb] = m
                continue
            for ip, par in enumerate(parities):
                par = tuple(par)
                sgn = self._sign(par, cmb)
                if on_device:
                    # in place on the GPU: no N^2 temporaries
                    if par not in self.acc:
                        last = ip == len(parities) - 1
                        self.acc[par] = [(a if last else a.clone()).mul_(sgn) for a in m]
                    else:
                        for a_acc, a in zip(self.acc[par], m):
                            a_acc.add_(a, alpha=sgn)
                elif par not in self.acc:
                    self.acc[par] = [sgn * a for a in m]
                else:
                    for a_acc, a in zip(self.acc[par], m):
                        a_acc += sgn * a
            del m

    @staticmethod
    def _sign(par, cmb):
        sgn = 1.0
        for pr, use in zip(par, cmb):
            if use:
                sgn *= pr
        return sgn

    def _sum(self, k, parity):
        par = tuple(parity) if parity is not None else (1,) * len(self.axes)
        if not self.keep:
            return self.acc[par][k]
        out = None
        for cmb in self.combos:
            m = self.mats[cmb][k]
            sgn = self._sign(par, cmb)
            out = sgn * m if out is None else out + sgn * m
        return out

    def S(self, parity=None):
        return self._sum(0, parity)

    def D(self, parity=None):
        return self._sum(1, parity)

    def G(self, parity=None):
        return self._sum(2, parity)


def lu_solve(A, B, device=None):
    """Dense solve A X = B (real or complex), torch LU on `device`."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cplx = np.iscomplexobj(A) or np.iscomplexobj(B)
    dt = torch.complex128 if cplx else torch.float64
    At = torch.as_tensor(A, dtype=dt, device=dev)
    Bt = torch.as_tensor(B, dtype=dt, device=dev)
    X = torch.linalg.solve(At, Bt)
    return X.cpu().numpy()


def neumann_green(op, phin, parity=None, device=None):
    """Exterior Neumann problem by (E): returns phi on the panels."""
    S, D = op.S(parity), op.D(parity)
    A = 2 * np.pi * np.eye(len(S)) - D
    return lu_solve(A, -S @ phin, device)


def neumann_source(op, phin, parity=None, device=None):
    """Exterior Neumann problem by the source formulation; returns
    (sigma, phi on panels). Requires op built with want_grad=True."""
    G = op.G(parity)
    n = op.pan.n
    K = -np.einsum("ijk,ik->ij", G, n) / (4 * np.pi)
    K[np.diag_indices_from(K)] += 0.5
    sig = lu_solve(K, phin, device)
    phi = -(op.S(parity) @ sig) / (4 * np.pi)
    return sig, phi


def added_mass(phi_k, nj, area, rho=1.0):
    """A_jk = -rho int phi_k n_j dS (n out of the body), over the panels in
    the solve; multiply by the number of symmetric copies for the full body."""
    return -rho * float(np.sum(phi_k * nj * area))
