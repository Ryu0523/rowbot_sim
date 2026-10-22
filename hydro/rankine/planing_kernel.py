"""
Linear steady free-surface response to a pressure distribution moving at
speed U (deep water, Kelvin free-surface condition, radiation condition),
tabulated on a lattice for the steady planing solver (stage B7, the planing
feasibility gate; derivation in learn/meta/RANKINE_STAGE1_2026-10-06.md,
section 13.2).

Nondimensional variables (one parameter only, the beam Froude number):
    lengths / b (beam), pressure / (rho U^2), elevation / b,
    C_V = U / sqrt(g b),  k0 = 1 / C_V^2 (transverse wavenumber * b).
Frame fixed to the hull, stream U in +x (downstream = +x), fluid z < 0.
Linear conditions on z = 0 (perturbation potential phi, U = 1):
    kinematic   zeta_x = phi_z
    dynamic     phi_x + zeta / C_V^2 + p = 0
    Laplace, decaying in z < 0:  phihat_z = |k| phihat.
With f(x) = (1/4 pi^2) int fhat(k) e^{i k.x} dk this gives
    zetahat = M(k) phat,
    M(k)    = -|k| / ( |k|/C_V^2 - (k_x - i eps)^2 ),                     (M)
where eps = b / l_mu > 0 is Rayleigh's fictitious friction: the poles of the
k_x integral lie in the upper half plane, so free waves appear only for
x > 0 (downstream; radiation condition) and decay like e^{-x / l_mu}. The
physical kernel is the limit eps -> 0; it is reached by Richardson
extrapolation in eps (first order, the kernel is analytic in eps) -- see
`richardson`. Limits of (M): C_V -> 0 gives M -> -C_V^2 (hydrostatic,
zeta = -p/(rho g)); C_V -> inf gives M -> |k|/k_x^2 (the lower half of the
thin-wing problem, the high-speed planing limit of Tulin 1957, Wang &
Rispin 1971).

Basis elements (pressure in a strip, piecewise linear in x on a lattice of
spacing h, constant in y over the strip [a, b] and over its mirror image
[-b, -a]: the flow is symmetric about the keel plane):
    'hat'   1 - |x|/h on [-h, h]                 (interior nodes)
    'half'  1 - x/h on [0, h], 0 for x < 0       (spray-root node: the
                                                 pressure jumps there)
    'ext'   1 on [-h, 0]                         (continuous spray-root
                                                 position, section 13.3)
Fourier transforms S(k) = int s(x) e^{-i k x} dx:
    hat   h sinc^2(k h / 2)
    half  1/(i k) + (1 - e^{-i k h}) / (h k^2)
    ext   (e^{i k h} - 1)/(i k)

Numerical method (section 13.3):
  * x: periodic box of length L_x = box_mult * l_mu with N = L_x/h points;
    the lattice samples of the inverse transform are exact for the periodic
    sum when the spectrum is folded over the images k + 2 pi l / h (|l| <=
    n_fold; the folded sum converges like l^-3 for 'hat' and, for the jump
    elements, like l^-3 after symmetric pairing). Images of the box are
    e^{-box_mult} smaller (Rayleigh decay).
  * y: exact continuous transform. For target y_t and source strip [a, b]
    (+ mirror) the weight is
        W(k_y) = (1/pi) [sin(k_y (y_t - a)) - sin(k_y (y_t - b))
                         + sin(k_y (y_t + b)) - sin(k_y (y_t + a))] / k_y,
    the large-k_y limit of the x-transform, I_inf(m) = -C_V^2 s(x_m)
    (local hydrostatic response; s(x_m) = 1/2 at a jump), is subtracted and
    integrated exactly (giving -C_V^2 s(x_m) [y_t in the strip]), and the
    remainder R = I - I_inf is integrated with Filon weights (R piecewise
    linear in k_y between nodes, the oscillating weight exact through the
    sine integral Si).
Nothing here is copied from any code; the transfer function is the textbook
one (Wehausen & Laitone 1960, section 13; Doctors 1974 uses the same
pressure-patch kernel in physical space).
"""
import math

import numpy as np
import torch
from scipy.special import sici

ELEMENTS = ("hat", "half", "ext")


def transfer(kx, ky, Cv, eps):
    """M(k) of (M); kx complex-capable torch tensor (real values), ky real.
    Cv = inf gives the gravity-free kernel."""
    kap = torch.sqrt(kx * kx + ky * ky)
    q = (kx - 1j * eps) ** 2
    if math.isinf(Cv):
        return kap / q
    return -kap / (kap / (Cv * Cv) - q)


def shape(name, k, h):
    """Fourier transform of the basis element `name` (lattice spacing h)."""
    u = k * h
    small = torch.abs(u) < 1e-4
    us = torch.where(small, torch.ones_like(u), u)
    if name == "hat":
        s = torch.where(small, 1.0 - u * u / 12.0, (torch.sin(us / 2) / (us / 2)) ** 2)
        return (h * s).to(torch.complex128)
    if name == "half":
        # 1/(ik) + (1 - e^{-iu})/(h k^2) = h [1/(iu) + (1 - e^{-iu})/u^2]; series h(1/2 - iu/6 - u^2/24)
        ser = h * (0.5 - 1j * u / 6.0 - u * u / 24.0)
        ex = h * (1.0 / (1j * us) + (1.0 - torch.exp(-1j * us)) / (us * us))
        return torch.where(small, ser, ex)
    if name == "ext":
        ser = h * (1.0 + 1j * u / 2.0 - u * u / 6.0)
        ex = h * (torch.exp(1j * us) - 1.0) / (1j * us)
        return torch.where(small, ser, ex)
    raise ValueError(name)


def lattice_values(name, m):
    """Pointwise value of the element at lattice offsets m (1/2 at a jump):
    the local hydrostatic response is -C_V^2 times this."""
    m = np.asarray(m)
    if name == "hat":
        return (m == 0).astype(float)
    if name == "half":
        return 0.5 * (m == 0)
    if name == "ext":
        return 0.5 * ((m == 0) | (m == -1))
    raise ValueError(name)


def ky_nodes(Cv, du=0.02, k1=3.0, ratio=1.005, kmax=2e5, fine=16):
    """k_y quadrature nodes (units 1/b): uniform with step min(du, k0/fine)
    on [0, k1], then geometric with `ratio` up to kmax."""
    k0 = 0.0 if math.isinf(Cv) else 1.0 / (Cv * Cv)
    step = du if k0 == 0.0 else min(du, k0 / fine)
    n1 = int(math.ceil(k1 / step))
    a = np.linspace(0.0, n1 * step, n1 + 1)
    g = [a[-1]]
    while g[-1] < kmax:
        g.append(g[-1] * ratio)
    return np.concatenate([a, np.array(g[1:])])


def x_transform(Cv, h, kys, m_max, l_mu, box_mult=8.0, n_fold=8, elements=ELEMENTS,
                device=None, chunk=None):
    """I_e(m, k_y) = (1/2 pi) int M(k_x, k_y) S_e(k_x) e^{i k_x m h} dk_x for
    |m| <= m_max at every k_y in `kys`, for each element e. Periodic box of
    length box_mult * l_mu; eps = 1/l_mu. Returns dict e -> real array
    (n_ky, 2 m_max + 1), offsets m = -m_max..m_max, and the box size N."""
    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eps = 1.0 / l_mu
    N = int(math.ceil(box_mult * l_mu / h))
    N += N % 2
    N = max(N, 4 * (2 * m_max + 1))
    kx = 2 * math.pi * torch.fft.fftfreq(N, d=h, dtype=torch.float64, device=dev)
    if chunk is None:
        chunk = max(1, int(2.5e7 // N))
    idx = torch.cat([torch.arange(0, m_max + 1, device=dev), torch.arange(N - m_max, N, device=dev)])
    out = {e: np.empty((len(kys), 2 * m_max + 1)) for e in elements}
    kys_t = torch.as_tensor(np.asarray(kys, float), dtype=torch.float64, device=dev)
    for c0 in range(0, len(kys), chunk):
        ky = kys_t[c0:c0 + chunk, None]
        acc = {e: torch.zeros((ky.shape[0], N), dtype=torch.complex128, device=dev) for e in elements}
        for l in range(-n_fold, n_fold + 1):
            kl = kx[None, :] + 2 * math.pi * l / h
            Mk = transfer(kl.to(torch.complex128), ky.to(torch.complex128), Cv, eps)
            for e in elements:
                acc[e] += Mk * shape(e, kl, h)
        for e in elements:
            I = torch.fft.ifft(acc[e], dim=1) / h
            Ir = I[:, idx].real.cpu().numpy()
            out[e][c0:c0 + ky.shape[0]] = Ir[:, np.argsort(np.r_[np.arange(0, m_max + 1), np.arange(-m_max, 0)])]
    return out, N


def filon_weights(kys, d):
    """Weights w_i with sum_i w_i R(k_i) = int_0^{k_n} R(k) sin(k d)/k dk for R
    piecewise linear between the nodes (exact in the oscillating factor)."""
    k = np.asarray(kys, float)
    w = np.zeros(len(k))
    if d == 0.0:
        return w
    Si = sici(k * d)[0]          # odd in d: Si(-x) = -Si(x)
    C = np.cos(k * d)
    dk = np.diff(k)
    A = Si[1:] - Si[:-1]                     # int sin(kd)/k
    B = (C[:-1] - C[1:]) / d                 # int sin(kd)
    # R = R_i (k_{i+1} - k)/dk + R_{i+1} (k - k_i)/dk
    w[:-1] += (k[1:] * A - B) / dk
    w[1:] += (B - k[:-1] * A) / dk
    return w


class KernelTables:
    """Tables K_e[t, s, m]: elevation (/b) at the centre y_t of half-strip t,
    lattice offset m = (x_target - x_source)/h, due to the element e of unit
    pressure (/rho U^2) in half-strip s and its mirror (y -> -y).
    `y_edges`: increasing edges of the half-beam strips, 0 ... 0.5 (units b)."""

    def __init__(self, Cv, h, y_edges, m_max, l_mu=None, c_mu=0.5, l_mu_bounds=(20.0, 400.0),
                 box_mult=8.0, n_fold=8, ky_kw=None, device=None, elements=ELEMENTS):
        self.Cv, self.h, self.m_max = float(Cv), float(h), int(m_max)
        self.y_edges = np.asarray(y_edges, float)
        self.yc = 0.5 * (self.y_edges[1:] + self.y_edges[:-1])
        if l_mu is None:
            lam0 = 2 * math.pi * Cv * Cv
            l_mu = float(np.clip(c_mu * lam0, *l_mu_bounds))
        self.l_mu, self.box_mult, self.n_fold = float(l_mu), box_mult, n_fold
        self.kys = ky_nodes(Cv, **(ky_kw or {}))
        I, self.N = x_transform(Cv, h, self.kys, self.m_max, self.l_mu, box_mult, n_fold, elements, device)
        m = np.arange(-self.m_max, self.m_max + 1)
        self.m = m
        ns = len(self.yc)
        self.K = {}
        hyd = -self.Cv ** 2
        # Filon weights per (t, s): four signed distances
        W = np.zeros((ns, ns, len(self.kys)))
        inside = np.zeros((ns, ns))
        for t in range(ns):
            yt = self.yc[t]
            for s in range(ns):
                a, b = self.y_edges[s], self.y_edges[s + 1]
                for d, sg in ((yt - a, 1), (yt - b, -1), (yt + b, 1), (yt + a, -1)):
                    W[t, s] += sg * filon_weights(self.kys, d)
                inside[t, s] = 1.0 if (a < yt < b) else 0.0
        W /= math.pi
        self.W = W
        for e in elements:
            Iinf = hyd * lattice_values(e, m)                      # (2m+1,)
            R = I[e] - Iinf[None, :]                               # (n_ky, 2m+1)
            K = np.einsum("tsk,km->tsm", W, R) + inside[:, :, None] * Iinf[None, None, :]
            self.K[e] = K
        self.I = I

    def table(self, e):
        return self.K[e]


def shape_vlm(k, h, a_frac=0.25):
    """Unit-load narrow hat of half-width a = a_frac h (a smeared vortex-
    lattice load): S = sinc^2(k a / 2)."""
    a = a_frac * h
    u = k * a / 2
    small = torch.abs(u) < 1e-6
    us = torch.where(small, torch.ones_like(u), u)
    return torch.where(small, 1.0 - u * u / 3.0, (torch.sin(us) / us) ** 2).to(torch.complex128)


def x_transform_specs(Cv, h, kys, m_max, l_mu, specs, box_mult=8.0, n_fold=24, a_frac=0.25,
                      device=None, chunk=None):
    """For each spec (name, shift, deriv): I(m, k_y) = (1/2 pi) int (i k_x)^deriv
    M(k_x, k_y) S_vlm(k_x) e^{i k_x (m + shift) h} dk_x, |m| <= m_max: the
    response (elevation for deriv 0, slope d/dx for deriv 1) at x = (m +
    shift) h downstream of a unit smeared load. Same periodic box and folding
    as x_transform; the image phases e^{i 2 pi l shift} are included."""
    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eps = 1.0 / l_mu
    N = int(math.ceil(box_mult * l_mu / h))
    N += N % 2
    N = max(N, 4 * (2 * m_max + 1))
    kx = 2 * math.pi * torch.fft.fftfreq(N, d=h, dtype=torch.float64, device=dev)
    if chunk is None:
        chunk = max(1, int(1.5e7 // N))
    idx = torch.cat([torch.arange(0, m_max + 1, device=dev), torch.arange(N - m_max, N, device=dev)])
    perm = np.argsort(np.r_[np.arange(0, m_max + 1), np.arange(-m_max, 0)])
    out = {sp[0]: np.empty((len(kys), 2 * m_max + 1)) for sp in specs}
    kys_t = torch.as_tensor(np.asarray(kys, float), dtype=torch.float64, device=dev)
    for c0 in range(0, len(kys), chunk):
        ky = kys_t[c0:c0 + chunk, None]
        acc = {sp[0]: torch.zeros((ky.shape[0], N), dtype=torch.complex128, device=dev) for sp in specs}
        for l in range(-n_fold, n_fold + 1):
            kl = kx[None, :] + 2 * math.pi * l / h
            Mk = transfer(kl.to(torch.complex128), ky.to(torch.complex128), Cv, eps)
            shp = {}
            for sp in specs:
                el = sp[3] if len(sp) > 3 else "vlm"
                if el not in shp:
                    shp[el] = shape_vlm(kl, h, a_frac) if el == "vlm" else shape(el, kl, h)
            for sp in specs:
                name, shift, deriv = sp[:3]
                el = sp[3] if len(sp) > 3 else "vlm"
                f = Mk * shp[el] * torch.exp(1j * kl * (shift * h))
                if deriv:
                    f = f * (1j * kl)
                acc[name] += f
        for sp in specs:
            name = sp[0]
            I = torch.fft.ifft(acc[name], dim=1) / h
            out[name][c0:c0 + ky.shape[0]] = I[:, idx].real.cpu().numpy()[:, perm]
    return out, N


VLM_SPECS = (("sx", 0.5, 1), ("zx", 0.75, 0))
# hat basis (piecewise-linear pressure), slope at cell midpoints, level at nodes
HAT_SPECS = (("s_hat", 0.5, 1, "hat"), ("s_half", 0.5, 1, "half"), ("s_ext", 0.5, 1, "ext"),
             ("z_hat", 0.0, 0, "hat"), ("z_half", 0.0, 0, "half"), ("z_ext", 0.0, 0, "ext"))


class VLMTables:
    """Tables K[name][t, s, m] for the vortex-lattice-type discretisation
    (section 13.3): 'sx' slope at the 3/4 point of cell i due to the unit
    load of cell j (m = j - i, offset (m + 1/2) h); 'zx' elevation at the
    spray-root node n due to the unit load of cell j (m = j - n, offset
    (m + 3/4) h). Loads: smeared over a hat of half-width a_frac h centred
    at the cell 1/4 point; strip s plus its mirror. No hydrostatic
    subtraction is needed: every target lies outside the load supports."""

    def __init__(self, Cv, h, y_edges, m_max, l_mu=None, c_mu=0.5, l_mu_bounds=(20.0, 400.0),
                 box_mult=8.0, n_fold=24, a_frac=0.25, ky_kw=None, device=None, y_targets=None,
                 specs=VLM_SPECS):
        self.Cv, self.h, self.m_max = float(Cv), float(h), int(m_max)
        self.y_edges = np.asarray(y_edges, float)
        self.yc = 0.5 * (self.y_edges[1:] + self.y_edges[:-1])
        self.yt = self.yc if y_targets is None else np.asarray(y_targets, float)
        if l_mu is None:
            lam0 = 2 * math.pi * Cv * Cv
            l_mu = float(np.clip(c_mu * lam0, *l_mu_bounds))
        self.l_mu, self.box_mult, self.n_fold, self.a_frac = float(l_mu), box_mult, n_fold, a_frac
        self.kys = ky_nodes(Cv, **(ky_kw or {}))
        I, self.N = x_transform_specs(Cv, h, self.kys, self.m_max, self.l_mu, specs, box_mult, n_fold,
                                      a_frac, device)
        self.specs = specs
        ns, nt = len(self.yc), len(self.yt)
        W = np.zeros((nt, ns, len(self.kys)))
        for t in range(nt):
            yt = self.yt[t]
            for s in range(ns):
                a, b = self.y_edges[s], self.y_edges[s + 1]
                for d, sg in ((yt - a, 1), (yt - b, -1), (yt + b, 1), (yt + a, -1)):
                    W[t, s] += sg * filon_weights(self.kys, d)
        W /= math.pi
        self.W = W
        inside = np.array([[1.0 if (self.y_edges[s] < self.yt[t] < self.y_edges[s + 1]) else 0.0
                            for s in range(ns)] for t in range(nt)])
        m = np.arange(-self.m_max, self.m_max + 1)
        self.K = {}
        for sp in specs:
            name, shift, deriv = sp[:3]
            el = sp[3] if len(sp) > 3 else "vlm"
            # local hydrostatic limit (elevation rows at lattice nodes only); slope targets lie off jumps
            if deriv == 0 and shift == 0.0 and el != "vlm":
                Iinf = -self.Cv ** 2 * lattice_values(el, m)
            elif deriv == 1 and shift == 0.5 and el in ("hat", "half"):
                # slope of the local hydrostatic response at cell midpoints
                dv = np.zeros(len(m))
                if el == "hat":
                    dv[m == -1] = 1.0 / h
                    dv[m == 0] = -1.0 / h
                else:
                    dv[m == 0] = -1.0 / h
                Iinf = -self.Cv ** 2 * dv
            else:
                Iinf = np.zeros(len(m))
            R = I[name] - Iinf[None, :]
            self.K[name] = np.einsum("tsk,km->tsm", W, R) + inside[:, :, None] * Iinf[None, None, :]


def richardson(t1, t2):
    """First-order Richardson extrapolation eps -> 0 from tables with l_mu
    and 2 l_mu (t2 has the larger l_mu, i.e. half the damping)."""
    assert abs(t2.l_mu / t1.l_mu - 2.0) < 1e-9
    out = {}
    for e in t1.K:
        out[e] = 2.0 * t2.K[e] - t1.K[e]
    return out


# ---------------------------------------------------------------- lateral hats
def y_elements(nodes, chine="jump"):
    """Lateral basis on the half-beam nodes 0 = y_0 < ... < y_n = 1/2 (units
    b), mirror-symmetric about the keel: keel hat on [-y_1, y_1] (its own
    mirror), interior hats on [y_{j-1}, y_{j+1}] plus mirror, chine half-hat
    rising from y_{n-1} to 1 at the chine y_n, zero outside (pressure jump at
    the chine) plus mirror. Returns per element the corner list
    [(y_c, value jump J_c, slope jump S_c)], the value at every node (inside
    limit at the chine) and the half-beam integral weight."""
    y = np.asarray(nodes, float)
    n = len(y) - 1
    els, w = [], []
    for j in range(n + 1 if chine == "jump" else n):
        c = []
        if j == 0:
            a = y[1]
            c = [(-a, 0.0, 1.0 / a), (0.0, 0.0, -2.0 / a), (a, 0.0, 1.0 / a)]
            w.append(0.5 * a)
        elif j < n:
            l, r = y[j] - y[j - 1], y[j + 1] - y[j]
            half = [(y[j - 1], 0.0, 1.0 / l), (y[j], 0.0, -1.0 / l - 1.0 / r), (y[j + 1], 0.0, 1.0 / r)]
            c = half + [(-yy, J, S) for (yy, J, S) in half]
            w.append(0.5 * (l + r))
        else:
            l = y[n] - y[n - 1]
            half = [(y[n - 1], 0.0, 1.0 / l), (y[n], -1.0, -1.0 / l)]
            # mirror: g(-y) -> jumps at -y_n (value +1 going right), slopes mirrored
            mir = [(-y[n - 1], 0.0, 1.0 / l), (-y[n], 1.0, -1.0 / l)]
            c = half + mir
            w.append(0.5 * l)
        els.append(c)
    ne = len(els)
    val = np.zeros((ne, ne))                # val[t, s] = element s at node t (inside limit)
    for s in range(ne):
        val[s, s] = 1.0
    return els, val, np.array(w)


def _w_direct(k, yt, corners_pieces):
    """W(k) = (1/pi) int g(y') cos(k (y_t - y')) dy' by Gauss quadrature over
    the linear pieces (used where k |d| is small)."""
    gx, gw = np.polynomial.legendre.leggauss(6)
    out = np.zeros_like(k)
    for (ya, yb, ga, gb) in corners_pieces:
        yy = 0.5 * (ya + yb) + 0.5 * (yb - ya) * gx
        gg = ga + (gb - ga) * (yy - ya) / (yb - ya)
        out += 0.5 * (yb - ya) * np.sum(gw[None, :] * gg[None, :] * np.cos(k[:, None] * (yt - yy[None, :])), axis=1)
    return out / math.pi


def _pieces_of(nodes, s):
    y = np.asarray(nodes, float)
    n = len(y) - 1
    if s == 0:
        return [(-y[1], 0.0, 0.0, 1.0), (0.0, y[1], 1.0, 0.0)]
    if s < n:
        p = [(y[s - 1], y[s], 0.0, 1.0), (y[s], y[s + 1], 1.0, 0.0)]
    else:
        p = [(y[n - 1], y[n], 0.0, 1.0)]
    return p + [(-b, -a, gb, ga) for (a, b, ga, gb) in p]


def filon_hat_weights(kys, yt, corners, pieces, k_switch=2.0):
    """Weights v_i with sum_i v_i R(k_i) = int_0^{k_n} W(k) R(k) dk, R
    piecewise linear, W(k) = (1/pi) [sum J_c sin(k d_c)/k - sum S_c cos(k
    d_c)/k^2], d_c = y_t - y_c (exact for the oscillating factor on the
    intervals with k max|d| >= k_switch; 6-point Gauss with W evaluated
    directly below)."""
    k = np.asarray(kys, float)
    nk = len(k)
    v = np.zeros(nk)
    dmax = max(abs(yt - c[0]) for c in corners) + 1e-12
    kl, kr = k[:-1], k[1:]
    dk = kr - kl
    analytic = kl * dmax >= k_switch
    # analytic part
    if np.any(analytic):
        ia = np.nonzero(analytic)[0]
        a, b = kl[ia], kr[ia]
        A0 = np.zeros(len(ia))     # int W           (coefficient of alpha)
        A1 = np.zeros(len(ia))     # int W k         (coefficient of beta)
        for (yc, J, S) in corners:
            d = yt - yc
            if J != 0.0 and d != 0.0:
                Sia, Sib = sici(a * d)[0], sici(b * d)[0]
                A0 += J * (Sib - Sia)
                A1 += J * (np.cos(a * d) - np.cos(b * d)) / d
            if S != 0.0:
                if d != 0.0:
                    sa, ca_ = sici(a * abs(d))
                    sb, cb_ = sici(b * abs(d))
                    Fa = -np.cos(a * d) / a - d * sici(a * d)[0]
                    Fb = -np.cos(b * d) / b - d * sici(b * d)[0]
                    A0 += -S * (Fb - Fa)
                    A1 += -S * (cb_ - ca_)
                else:
                    A0 += -S * (-1.0 / b + 1.0 / a)
                    A1 += -S * (np.log(b) - np.log(a))
        A0 /= math.pi
        A1 /= math.pi
        # R = R_i (b - k)/dk + R_{i+1} (k - a)/dk
        v[ia] += (b * A0 - A1) / dk[ia]
        v[ia + 1] += (A1 - a * A0) / dk[ia]
    # direct part
    idd = np.nonzero(~analytic)[0]
    if len(idd):
        gx, gw = np.polynomial.legendre.leggauss(6)
        kk = kl[idd][:, None] + 0.5 * (1 + gx)[None, :] * dk[idd][:, None]
        Wk = _w_direct(kk.ravel(), yt, pieces).reshape(kk.shape)
        wq = 0.5 * dk[idd][:, None] * gw[None, :] * Wk
        np.add.at(v, idd, np.sum(wq * (kr[idd][:, None] - kk) / dk[idd][:, None], axis=1))
        np.add.at(v, idd + 1, np.sum(wq * (kk - kl[idd][:, None]) / dk[idd][:, None], axis=1))
    return v


_W_CACHE = {}


def lateral_targets(nodes, chine_frac=0.25):
    """Collocation lines: the lateral nodes, except the chine line, which is
    moved inboard by chine_frac of the last interval (off the pressure jump
    at the chine, where the local response is the jump average)."""
    y = np.asarray(nodes, float).copy()
    y[-1] = y[-1] - chine_frac * (y[-1] - y[-2])
    return y


def element_values(nodes, targets, chine="jump"):
    """val[t, s] = lateral element s at target t."""
    y = np.asarray(nodes, float)
    n = len(y) - 1
    ne = n + 1 if chine == "jump" else n
    val = np.zeros((len(targets), ne))
    for t, yt in enumerate(targets):
        yt = abs(yt)
        for s in range(ne):
            for (a, b, ga, gb) in _pieces_of(nodes, s):
                if a <= yt <= b and b > a:
                    val[t, s] = ga + (gb - ga) * (yt - a) / (b - a)
                    break
    return val


def hat_weight_matrix(kys, nodes, targets=None, chine="jump"):
    """W[t, s, k] for the lateral hat basis (cached per node set, targets and k grid)."""
    targets = lateral_targets(nodes) if targets is None else np.asarray(targets, float)
    key = (len(kys), float(kys[-1]), float(np.sum(kys)), tuple(np.round(nodes, 12)), tuple(np.round(targets, 12)),
           chine)
    if key in _W_CACHE:
        return _W_CACHE[key]
    els, _, wy = y_elements(nodes, chine)
    val = element_values(nodes, targets, chine)
    nn = len(els)
    W = np.zeros((len(targets), nn, len(kys)))
    for t in range(len(targets)):
        for s in range(nn):
            W[t, s] = filon_hat_weights(kys, targets[t], els[s], _pieces_of(nodes, s))
    _W_CACHE[key] = (W, val, wy)
    return W, val, wy


class YHatTables:
    """Tables K[name][t, s, m] with the lateral hat basis of `y_elements`
    (targets at the lateral nodes, inside limit at the chine), x specs as
    in HAT_SPECS. The local hydrostatic limit is subtracted for the
    elevation and slope rows and added back with the element value at the
    target node."""

    def __init__(self, Cv, h, nodes, m_max, l_mu=None, c_mu=0.5, l_mu_bounds=(20.0, 400.0),
                 box_mult=8.0, n_fold=8, ky_kw=None, device=None, specs=None, chine="jump"):
        specs = HAT_SPECS if specs is None else specs
        self.chine = chine
        self.Cv, self.h, self.m_max = float(Cv), float(h), int(m_max)
        self.nodes = np.asarray(nodes, float)
        if l_mu is None:
            lam0 = 2 * math.pi * Cv * Cv
            l_mu = float(np.clip(c_mu * lam0, *l_mu_bounds))
        self.l_mu = float(l_mu)
        self.kys = ky_nodes(Cv, **(ky_kw or {}))
        I, self.N = x_transform_specs(Cv, h, self.kys, self.m_max, self.l_mu, specs, box_mult, n_fold, 0.25, device)
        self.targets = lateral_targets(self.nodes) if chine == "jump" else np.asarray(self.nodes[:-1], float)
        W, val, self.wy = hat_weight_matrix(self.kys, self.nodes, self.targets, chine)
        self.W = W
        m = np.arange(-self.m_max, self.m_max + 1)
        self.K = {}
        for sp in specs:
            name, shift, deriv = sp[:3]
            el = sp[3] if len(sp) > 3 else "vlm"
            if deriv == 0 and shift == 0.0 and el != "vlm":
                xinf = -self.Cv ** 2 * lattice_values(el, m)
            elif deriv == 1 and shift == 0.5 and el in ("hat", "half"):
                dv = np.zeros(len(m))
                if el == "hat":
                    dv[m == -1] = 1.0 / h
                    dv[m == 0] = -1.0 / h
                else:
                    dv[m == 0] = -1.0 / h
                xinf = -self.Cv ** 2 * dv
            else:
                xinf = np.zeros(len(m))
            R = I[name] - xinf[None, :]
            # the Fourier integral of the hydrostatic part gives the jump average at the chine node;
            # the inside limit is used (val), i.e. the subtracted part is added back exactly
            self.K[name] = np.einsum("tsk,km->tsm", W, R) + val[:, :, None] * xinf[None, None, :]
