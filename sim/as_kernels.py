"""
Row-stable kernels of the any-speed plant (step P1c; ACCEPTANCE E1, N2):
the incident wave's phases along the hull and the exact linear
Froude-Krylov pressure of switch fk_exact (sim/planing_anyspeed.py, module
docstring 'P1b fk_exact'), for N boats at once.

ROW-STABLE means: every per-section (per-row) number depends on that row's
inputs only -- never on how many sections are wet, on which other boats
share the call, or on where a row sits in the arrays. One boat
(sim/planing_anyspeed._StripAS._waves_b and _fk_b call these with N = 1)
and N boats in one call (sim/planing_batch.py) therefore give the same
bits. What this rules out, and what is used instead:

  - BLAS matrix-vector and matrix-matrix products over a variable set of
    rows (OpenBLAS blocks rows in fours; a row's dot product depends on its
    position in the block -- measured on this machine): every per-row
    reduction over the wave components is an einsum over the components
    (numpy's sum-of-products loop runs each output element's sum on its
    own; checked row by row against single rows, 1 to 4000 rows); the
    products over a FIXED set of rows (the n stations of one boat) are
    per-boat stacked products of the same shape;
  - sums over a variable subset of stations: the per-row values are
    scattered into the (N, n) station array (zero where dry) and summed
    over the n stations (numpy's pairwise sum of a row of fixed length).

The physics is P1b's (the same section integrals, strip moments and
long-wave subtraction; the derivation is in sim/planing_anyspeed._fk_b and
hydro/section2d.section_fk); the floating-point association differs from
P1b's implementation at the round-off level (P1c re-pins the goldens of
the switch sets that contain fk_exact: learn/meta/
ANYSPEED_PLAN_2026-10-05.md, P1c).

The phases along the hull are P1b's running product (equally spaced
stations: e^{i psi_j} = e^{i psi_0} (e^{i kappa dx})^j), boat by boat the
same bits as before P1c; so are v0's wave quantities at the stations.
"""
import numpy as np

from hydro.section2d import GRID_TOL

# number of Taylor terms of the strip integrals (sim/planing_anyspeed.
# _StripAS.FK_TERMS; register P16)
FK_TERMS = 16


# ============================================================== the phases
def phases(kx, ky, w, wphi, X0, Y0, psi, t, n, dx, dxT):
    """The components' phases at the n equally spaced stations and at the
    transom, for N boats: kx, ky, w, wphi (N, M); X0, Y0 the earth position
    of station 0, psi the heading, t the time (N,); dx the station spacing,
    dxT the transom's body x less station 0's. Returns (kap, lam, Z, CT,
    ST): the body-frame wavenumbers along and across the hull (N, M), Z =
    e^{i psi_j} (N, n, M) complex -- P1b's running product along the hull
    (_waves_b), boat by boat the same bits --, cos and sin of the
    transom's phase (N, M)."""
    cps = np.cos(psi)[:, None]
    sps = np.sin(psi)[:, None]
    kap = kx * cps + ky * sps
    lam = -kx * sps + ky * cps
    ph0 = X0[:, None] * kx + Y0[:, None] * ky - (w * t[:, None] - wphi)
    N, M = kx.shape
    Z = np.empty((N, n, M), complex)
    Z[:, 0] = np.exp(1j * ph0)
    Z[:, 1:] = np.exp(1j * (kap * dx))[:, None, :]
    Z = np.cumprod(Z, axis=1)
    phT = ph0 + dxT * kap
    return kap, lam, Z, np.cos(phT), np.sin(phT)


def station_waves(Z, cc, sc):
    """v0's five wave quantities at the stations from the phases (P1b's
    _waves_b): (cos @ [a, -a w^2], sin @ [a w, -a kx, -a ky]), one stacked
    product per boat (cc (N, M, 2), sc (N, M, 3), or (1, ...) shared);
    returns e, et, ett, ex, ey (N, n) each."""
    o1 = np.matmul(Z.real, cc)
    o2 = np.matmul(Z.imag, sc)
    return o1[..., 0], o2[..., 0], o1[..., 1], o2[..., 1], o2[..., 2]


# ================================================================ sections
def vsec_q(d, yc, tb):
    """sim/planing_anyspeed._vsec_qs on arrays, operation for operation
    (the same floats: the transom face's (q, s2, A, m1) of each boat)."""
    dp = np.where(d > 0.0, d, 0.0)
    dch = yc * tb
    b = np.where(dp < dch, dp, dch)
    e = np.where(dp > dch, dp - dch, 0.0)
    top = dch + e
    A = b * b / tb + 2.0 * yc * e
    m1 = 2.0 * b ** 3 / (3.0 * tb) + yc * (top * top - dch * dch)
    m2 = b ** 4 / (2.0 * tb) + 2.0 / 3.0 * yc * (top ** 3 - dch ** 3)
    return dp * A - m1, dp * m1 - m2, A, m1


def vsec_qs1(d, yc, tanb):
    """sim/planing_anyspeed._vsec_qs (one section, floats): the same
    operations as vsec_q, so one boat and N boats give the same bits."""
    dp = d if d > 0.0 else 0.0
    dch = yc * tanb
    b = dp if dp < dch else dch
    e = dp - dch if dp > dch else 0.0
    top = dch + e
    A = b * b / tanb + 2.0 * yc * e
    m1 = 2.0 * b ** 3 / (3.0 * tanb) + yc * (top * top - dch * dch)
    m2 = b ** 4 / (2.0 * tanb) + 2.0 / 3.0 * yc * (top ** 3 - dch ** 3)
    return dp * A - m1, dp * m1 - m2, A, m1


def vsec_m1(d, yc, tb):
    """m1 of sim/planing_anyspeed._vsec (the first moment of the section
    area about the keel), its operations."""
    dp = np.maximum(d, 0.0)
    dch = yc * tb
    b = np.minimum(dp, dch)
    e = np.maximum(dp - dch, 0.0)
    top = dch + e
    return 2.0 * b ** 3 / (3.0 * tb) + yc * (top * top - dch * dch)


def strip_moments(mu, dx, terms=FK_TERMS):
    """The strips' chine moments arranged for the strip factors: muE (3n, 9)
    and muO (3n, 8), row off * n + j holding mu[j, off + t] / dx for the
    even (muE) and odd (muO) t of the Taylor series (off = 0, 1, 2: the
    moments m = 0, 1, 2 against xi^t)."""
    n = mu.shape[0]
    ev = [mu[:, off:off + terms + 1:2] for off in (0, 1, 2)]
    od = [mu[:, off + 1:off + terms + 1:2] for off in (0, 1, 2)]
    muE = np.concatenate(ev, axis=0) / dx
    muO = np.concatenate(od, axis=0) / dx
    assert muE.shape == (3 * n, terms // 2 + 1)
    return np.ascontiguousarray(muE), np.ascontiguousarray(muO)


# the factors s_t / t of the running product of _series: s_t = +1 for odd
# t, -1 for even t (the sign pattern 1, 1, -1, -1, ... of i^t / i^(t mod 2))
_SF = np.array([(1.0 if t % 2 else -1.0) / t for t in range(1, FK_TERMS + 1)])


def _series(kap):
    """(ev, od) (N, 9, M), (N, 8, M): the real coefficients of the Taylor
    series of e^{i kappa xi} -- (i kappa)^t / t! without its i^(t mod 2),
    sign included -- even and odd t (P1b's _fk_strip), as one running
    product of the factors s_t kappa / t."""
    N, M = kap.shape
    Q = np.empty((N, FK_TERMS + 1, M))
    Q[:, 0] = 1.0
    np.multiply(kap[:, None, :], _SF[None, :, None], out=Q[:, 1:])
    P = np.cumprod(Q, axis=1)
    return P[:, 0::2], P[:, 1::2]


def _fk_sections(rd, ryc, tb, k, lam, ku, kinv, cvm):
    """The section integrals (E, E_k, E_lam, W, Wy, c_V) of rows with keel
    depths rd > 0 and chine half-beams ryc (R,) -- hydro/section2d.
    section_fk_grid's expressions, row by row. k, lam: (M,) shared by every
    row or (R, M) per row; ku, kinv: the distinct wavenumbers and the map
    to the components, each shared ((nu,), (M,)) or per row ((R, nu), (R,
    M)); cvm
    the largest c_V of each row's boat (scalar or (R, 1)): the columns with
    |lam| c_V,max < 1e-3 take S0 and T from their series (section_fk_grid
    decides per boat). The V part's Taylor series (|alpha| h1 < GRID_TOL)
    is evaluated on the rows that have such an element; the wall part on
    the rows with a wall, or on all rows when most have one (a row without
    wall gets exactly nothing from it: P = 1, g = g' = 0)."""
    R = rd.size
    dch = ryc * tb
    h1 = np.minimum(rd, dch)
    Dl = rd - h1
    cV = h1 / tb
    q = lam / tb
    D = k * k + q * q
    iD = 1.0 / D
    zc = np.abs(lam) * cvm < 1e-3
    ilam = 1.0 / np.where(zc, 1.0, lam)
    ilam[zc] = 0.0
    x = cV[:, None] * lam
    xinv = (1.0 / cV)[:, None] * ilam
    cx = np.cos(x)
    S0 = np.sin(x)
    S0 *= xinv
    T = S0 - cx
    T *= xinv
    if zc.any():
        # (the same expressions, element by element; one boat's columns by
        # index, several boats' elements by mask)
        sel = (slice(None), np.nonzero(zc)[0]) if zc.ndim == 1 else zc
        xz = x[sel]
        x2 = xz * xz
        S0[sel] = 1.0 - x2 / 6.0 + x2 * x2 / 120.0
        T[sel] = xz * (1.0 / 3.0 - x2 / 30.0 + x2 * x2 / 840.0)
    # e^{-k d} and e^{-k Delta}, per distinct wavenumber, one exponential
    dd = np.concatenate([-rd, -Dl])[:, None]
    ku2 = ku if ku.ndim == 1 else np.concatenate([ku, ku])
    if kinv.ndim == 1:
        KP = np.exp(dd * ku2)[:, kinv]
    else:
        KP = np.take_along_axis(np.exp(dd * ku2),
                                np.concatenate([kinv, kinv]), 1)
    K, P = KP[:R], KP[R:]
    # (in place where a temporary is not needed again: the same operations
    # in the same order)
    h1c = h1[:, None]
    kh1 = h1c * k
    A1 = kh1 * S0
    A1 -= cx
    N = P * A1
    N += K
    Nk = h1c * S0
    Nk -= Dl[:, None] * A1
    Nk *= P
    Nk -= rd[:, None] * K
    Nq = x * S0
    Nq -= kh1 * T
    Nq *= P * h1c
    c2 = 2.0 / tb
    NiD = N * iD
    E = c2 * NiD
    Nk -= (2.0 * k) * NiD
    Nk *= c2
    Nk *= iD
    Ek = Nk
    Nq -= (2.0 * q) * NiD
    Nq *= c2 / tb
    Nq *= iD
    El = Nq
    # the V part's series where |alpha| h1 < GRID_TOL: the rows that have
    # such an element (h1^2 min D below it), then element by element
    Dmin = D.min(axis=-1)
    rs = np.nonzero(h1 * h1 * Dmin < GRID_TOL ** 2)[0]
    if rs.size:
        Ds = D[rs] if D.ndim > 1 else D
        h1s = h1[rs][:, None]
        small = (h1[rs] * h1[rs])[:, None] * Ds < GRID_TOL ** 2
        kap_ = kh1[rs]
        xs_ = x[rs]
        base = 0.5 - kap_ / 6.0 + (kap_ * kap_ - xs_ * xs_) / 24.0
        pre = c2 * P[rs] * h1s * h1s
        Dls = Dl[rs][:, None]
        E[rs] = np.where(small, pre * base, E[rs])
        Ek[rs] = np.where(small, pre * (-Dls * base + h1s * (
            -1.0 / 6.0 + kap_ / 12.0)), Ek[rs])
        El[rs] = np.where(small, (pre / tb) * h1s * (-xs_ / 12.0), El[rs])
    rw = np.nonzero(Dl > 0.0)[0]
    if 2 * rw.size > R:
        # the wall part on every row: g = (1 - P)/k, g' = (Delta P - g)/k
        ik = 1.0 / k
        g = 1.0 - P
        g *= ik
        gk = Dl[:, None] * P
        gk -= g
        gk *= ik
        ys = (2.0 * ryc)[:, None] * S0
        E += ys * g
        Ek += ys * gk
        t2 = (2.0 * ryc * ryc)[:, None] * T
        t2 *= g
        El -= t2
    elif rw.size:
        ik = 1.0 / (k[rw] if k.ndim > 1 else k)
        Pw = P[rw]
        g = (1.0 - Pw) * ik
        gk = (Dl[rw][:, None] * Pw - g) * ik
        ys = (2.0 * ryc[rw])[:, None] * S0[rw]
        E[rw] += ys * g
        Ek[rw] += ys * gk
        El[rw] -= (2.0 * ryc[rw] * ryc[rw])[:, None] * T[rw] * g
    W = (2.0 * cV)[:, None] * S0
    Wy = (2.0 * cV * cV)[:, None] * T
    return E, Ek, El, W, Wy, cV


# ===================================================== the exact FK force
class FKGeo:
    """What fk_rows reads of a plant (sim/planing_anyspeed._StripAS
    ._fk_geo): stations yc, zk, x (n,), tan(deadrise) tb, spacing dx, rho g,
    the pitch sign SP, the transom chine half-beam ycT, the arranged strip
    moments muE, muO (strip_moments)."""

    def __init__(self, yc, zk, x, tb, dx, rg, SP, ycT, mu):
        self.yc = np.asarray(yc, float)
        self.zk = np.asarray(zk, float)
        self.x = np.asarray(x, float)
        self.n = self.yc.size
        self.tb, self.dx, self.rg, self.SP = float(tb), float(dx), \
            float(rg), SP
        self.ycT = float(ycT)
        self.muE, self.muO = strip_moments(np.asarray(mu, float), self.dx)


class FKWaves:
    """The wave inputs of fk_rows for N boats: per sea a, k (N, M), the
    distinct wavenumbers ku (N, nu) and kinv (N, M) (k = ku[kinv]); per
    derivative the body wavenumbers kap, lam (N, M), the station phases Z
    (N, n, M), the transom's CT, ST (N, M) and the elevation e (N, n)."""
    __slots__ = ("a", "k", "ku", "kinv", "kap", "lam", "Z", "CT", "ST", "e")

    def __init__(self, **kw):
        for k_, v in kw.items():
            setattr(self, k_, v)


class FKOut:
    __slots__ = ("fz", "fy", "x", "m", "my_sig", "n_sig", "K")

    def __init__(self, **kw):
        for k_, v in kw.items():
            setattr(self, k_, v)


def fk_rows(g, d, A, Th, ab, dT, hw, head, z0T, wv):
    """The exact linear incident pressure beyond the plant's long-wave field
    for N boats (sim/planing_anyspeed._fk_b's outputs and derivation): d, A
    (N, n) keel depth below the local surface and section area at the
    stations; Th, ab (N,) the pitch and the buoyancy factor a_b; dT, hw,
    head, z0T (N,) the transom's keel depth, wetted face height, dead-water
    head and keel height (_transom_kin); wv an FKWaves. Returns FKOut: fz
    (x a_b), fy (N, n) per unit length; x, m (x a_b), my_sig, n_sig, K (N,).

    Rows: the wet stations of every boat (in station order), then the
    transom sections and the wetted faces of the boats that have them; one
    evaluation of the section integrals. H = Q / yc and its slope along x,
    H' = (H_right - H_left) / (dx nrm) over the wet neighbours of the same
    boat (central where both are wet, one-sided where one is, 0 when
    isolated: P1b's _fk_h); the slope's 1 / (dx nrm) is carried by the
    strip factors it multiplies."""
    N, n = d.shape
    one = N == 1
    tb, dx, rg, SP = g.tb, g.dx, g.rg, g.SP
    k, lam, kap, a = wv.k, wv.lam, wv.kap, wv.a
    ib, jb = np.nonzero(d > 0.0)
    R = ib.size
    hasT = dT > 0.0
    hasF = hasT & (hw > 0.0) & np.isfinite(head)
    iT = np.nonzero(hasT)[0]
    iF = np.nonzero(hasF)[0]
    nT, nF = iT.size, iF.size
    fz = np.zeros((N, n))
    fy = np.zeros((N, n))
    if R + nT == 0:
        z = np.zeros(N)
        return FKOut(fz=fz, fy=fy, x=ab * (z - z + z), m=ab * (z - z + z),
                     my_sig=SP * ab * z * dx, n_sig=z.copy(), K=z.copy())
    rb = np.concatenate([ib, iT, iF])
    rd = np.concatenate([d[ib, jb], dT[iT], hw[iF]])
    ryc = np.concatenate([g.yc[jb], np.full(nT + nF, g.ycT)])
    # the largest c_V of each boat's rows (section_fk_grid's column switch)
    cv_rows = np.minimum(rd, ryc * tb) / tb
    if one:
        cvm = float(cv_rows.max(initial=0.0))
        kr, lr, kur, kir = k[0], lam[0], wv.ku[0], wv.kinv[0]
    else:
        CV = np.zeros((N, n + 2))
        CV[ib, jb] = cv_rows[:R]
        CV[iT, n] = cv_rows[R:R + nT]
        CV[iF, n + 1] = cv_rows[R + nT:]
        cvm = CV.max(axis=1)[rb][:, None]
        lr = lam[rb]
        # the seas' wavenumbers: shared by every row when the boats' seas
        # have the same ones (the same values either way, element by
        # element), else per row
        if (k == k[0]).all() and (wv.ku == wv.ku[0]).all():
            kr, kur = k[0], wv.ku[0]
        else:
            kr, kur = k[rb], wv.ku[rb]
        kir = wv.kinv[0] if (wv.kinv == wv.kinv[0]).all() else wv.kinv[rb]
    E, Ek, El, W, Wy, cV = _fk_sections(rd, ryc, tb, kr, lr, kur, kir,
                                        cvm)
    # ------------------------------------------------- the station rows
    S_ = np.zeros((5, N, n))         # K, xs, ms, mys, nsg per station
    if R:
        M = E.shape[1]
        ks = kr if kr.ndim == 1 else kr[:R]
        ls = lr if lr.ndim == 1 else lr[:R]
        Es, Eks, Els = E[:R], Ek[:R], El[:R]
        di = rd[:R]
        z0 = g.zk[jb] + SP * g.x[jb] * Th[ib]
        zw = z0 + di
        Ezw = zw[:, None] * Es
        Ezw += Eks
        # H = Q / yc, planes [G1, E_lam, GK, E, E z_w]
        H = np.empty((5, R, M))
        iyc = (1.0 / g.yc[jb])[:, None]
        np.multiply(ks, Es, out=H[0])                           # G1
        np.subtract(W[:R], H[0], out=H[0])
        H[0] *= iyc
        np.multiply(Els, iyc, out=H[1])                         # E_lam
        np.multiply(-ks, Els, out=H[2])                         # GK
        H[2] -= ls * Ezw
        H[2] -= Wy[:R]
        H[2] *= iyc
        np.multiply(Es, iyc, out=H[3])
        np.multiply(Ezw, iyc, out=H[4])
        # the slope's differences: neighbours of the same boat at adjacent
        # stations (Gd = H_right - H_left; its 1 / (dx nrm) in Ac, Bc)
        cont = (jb[1:] == jb[:-1] + 1) & (ib[1:] == ib[:-1])
        right = np.concatenate([cont, [False]])
        left = np.concatenate([[False], cont])
        cs = 1.0 / (dx * np.maximum(right.astype(float) + left, 1.0))
        Gd = np.empty_like(H)
        Gd[:, 1:-1] = H[:, 2:] - H[:, :-2]
        ri = np.arange(R)
        ip = ri + right
        im = ri - left
        fix = np.nonzero((ip != ri + 1) | (im != ri - 1))[0]
        if fix.size:
            Gd[:, fix] = H[:, ip[fix]] - H[:, im[fix]]
        # the strip factors: Re and Im of e^{i psi_j} Q_m / dx, Q_m the strip
        # moments against the exact phase (Taylor series on the chine
        # moments), per boat over its n stations (one shape), at the wet
        # rows
        ev, od = _series(kap)
        qr = np.matmul(g.muE, ev).reshape(N * 3 * n, M)
        qi = np.matmul(g.muO, od).reshape(N * 3 * n, M)
        sel = (ib * (3 * n) + jb)[None, :] + (np.arange(3) * n)[:, None]
        qr = qr[sel]                                            # (3, R, M)
        qi = qi[sel]
        Cw = wv.Z.real[ib, jb]
        Sw = wv.Z.imag[ib, jb]
        Af = Cw * qr                                            # A0, A1, A2
        Af -= Sw * qi
        Bf = Sw * qr                                            # B0, B1, B2
        Bf += Cw * qi
        Ac = Af[1:] * cs[None, :, None]             # A1, A2 / (dx nrm)
        Bc = Bf[1:] * cs[None, :, None]             # B1, B2 / (dx nrm)
        # the integrands of the per-row sums over the components, with
        # their weights: [fz (a), yaw x-facing (a kap), roll (a), lateral
        # (a lam), lever (a kap), pitch sigma (a), yaw sigma (a lam),
        # lateral again (a kap), long-wave (a kap)]
        Y = np.empty((9, R, M))
        np.multiply(H[0:2], Af[0], out=Y[0:2])
        Y[0:2] += Gd[0:2] * Ac[0]
        np.multiply(H[2:5], Bf[0], out=Y[2:5])
        Y[2:5] += Gd[2:5] * Bc[0]
        np.multiply(H[0], Af[1], out=Y[5])
        Y[5] += Gd[0] * Ac[1]
        np.multiply(H[3], Bf[1], out=Y[6])
        Y[6] += Gd[3] * Bc[1]
        Y[7] = Y[3]
        Y[8] = Sw
        # the per-row sums (einsum: each row's sum over the components
        # alone, whatever the other rows -- row-stable)
        if one:
            a_, ak, al = a[0], a[0] * kap[0], a[0] * lam[0]
            Wp = np.stack([a_, ak, a_, al, ak, a_, al, ak, ak])
            Dt = np.einsum("qrm,qm->qr", Y, Wp)
        else:
            Wb = np.stack([a, a * kap, a, a * lam, a * kap, a, a * lam,
                           a * kap, a * kap], 1)                # (N, 9, M)
            Dt = np.empty((9, R))
            st_ = np.searchsorted(ib, np.arange(N + 1))
            for b_ in np.nonzero(st_[1:] > st_[:-1])[0]:
                r0, r1 = st_[b_], st_[b_ + 1]
                Dt[:, r0:r1] = np.einsum("qrm,qm->qr", Y[:, r0:r1], Wb[b_])
        Ai = A[ib, jb]
        m1 = vsec_m1(di, g.yc[jb], tb)
        fz[ib, jb] = ab[ib] * (rg * (Dt[0] - 2.0 * cV[:R] * wv.e[ib, jb]))
        fy[ib, jb] = rg * Dt[3]
        Sk = Dt[8]
        S_[:, ib, jb] = np.stack([rg * Dt[2], rg * (Dt[7] - Sk * Ai),
                                  rg * (Dt[4] - Sk * (z0 * Ai + m1)),
                                  rg * Dt[5], rg * (Dt[6] + Dt[1])])
    sums = S_.sum(axis=2)
    K, xsum, msum = sums[0] * dx, sums[1] * dx, sums[2] * dx
    my_sig = SP * ab * sums[3] * dx
    n_sig = sums[4] * dx
    # ------------------------------------------------- the transom rows
    QT, RT, Qf, Rf, NT = (np.zeros(N) for _ in range(5))
    if nT:
        rT = slice(R, R + nT)
        cT = wv.CT * a
        sT = wv.ST * a
        if one:
            AT, m1T = vsec_qs1(float(dT[0]), g.ycT, tb)[2:]
            AT, m1T = np.array([AT]), np.array([m1T])
        else:
            AT, m1T = vsec_q(dT[iT], g.ycT, tb)[2:]
        dTi = dT[iT]
        z0i = z0T[iT]
        zwT = z0i + dTi
        ET, EkT = E[rT], Ek[rT]
        YT = np.empty((3, nT, ET.shape[1]))
        YT[0] = ET - AT[:, None]
        YT[1] = EkT + zwT[:, None] * ET - (z0i * AT + m1T)[:, None]
        YT[2] = El[rT]
        WT = np.stack([cT[iT], cT[iT], sT[iT]])                # (3, nT, M)
        DT = np.einsum("qrm,qrm->qr", YT, WT)
        QT[iT] = rg * DT[0]
        RT[iT] = rg * DT[1]
        NT[iT] = rg * DT[2]
        if nF:
            rF = slice(R + nT, R + nT + nF)
            hd = head[iF]
            hwi = hw[iF]
            if one:
                dec = np.exp(-k[0] * hd[0])[None]
                Aw, m1w = vsec_qs1(float(hwi[0]), g.ycT, tb)[2:]
                Aw, m1w = np.array([Aw]), np.array([m1w])
            else:
                dec = np.exp(-k[iF] * hd[:, None])
                Aw, m1w = vsec_q(hwi, g.ycT, tb)[2:]
            Ef, Efk = E[rF], Ek[rF]
            zwF = (z0T + dT)[iF]
            z0F = z0T[iF]
            YF = np.empty((3, nF, Ef.shape[1]))
            YF[0] = dec * Ef - Aw[:, None]
            YF[1] = (dec * (Efk - hd[:, None] * Ef) + zwF[:, None] * dec
                     * Ef - (z0F * Aw + m1w)[:, None])
            YF[2] = dec * El[rF]
            WF = np.stack([cT[iF], cT[iF], sT[iF]])
            DF = np.einsum("qrm,qrm->qr", YF, WF)
            Qf[iF] = rg * DF[0]
            Rf[iF] = rg * DF[1]
            NT[iF] -= rg * DF[2]
        n_sig = n_sig + NT
    return FKOut(fz=fz, fy=fy, x=ab * (xsum - QT + Qf),
                 m=ab * (msum - RT + Rf), my_sig=my_sig, n_sig=n_sig, K=K)
