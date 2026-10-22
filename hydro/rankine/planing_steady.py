"""
Steady planing of a prismatic hard-chine hull (constant deadrise beta, beam b,
transom) in the LINEAR theory with gravity: potential flow, Kelvin
free-surface condition linearised about z = 0, hull boundary condition
transferred to z = 0 (thin planing-surface theory), wetted area determined by
the solution (spray-root condition), Kutta condition at the transom (stage
B7, the planing feasibility gate; derivation, assumptions and the registered
checks in learn/meta/RANKINE_STAGE1_2026-10-06.md, section 13).

Formulation (section 13.2). Hull-fixed frame, stream U in +x, transom at
x = 0, keel depth d below the undisturbed surface at the transom, trim tau,
deadrise beta. Bottom height (nondimensional, units b):
    h(x, y) = -d + (-x) tan(tau) + |y| tan(beta)        (x <= 0, |y| <= 1/2)
Unknown: the pressure p (/rho U^2) on the wetted planform W, zero outside.
The free-surface elevation on z = 0 is zeta = G[p] (exact linear response,
planing_kernel). Conditions:
  (B)  zeta = h on W: the slope form zeta_x = h_x = -tan(tau) (the
       linearised kinematic body condition) at every collocation point, plus
       the level zeta = h at ONE point of W per strip (the spray-root node);
       with zeta continuous this makes the free surface meet the bottom at
       the upstream boundary of W (Wagner's wetting condition, linearised).
  (K)  Kutta at the transom (bounded pressure, p -> 0, smooth detachment):
       implicit in the 1/4-3/4 placement of the vortex-lattice-type
       discretisation (exact for the 2-D flat plate, James 1972).
  (S)  spray root x_LE(y): unknown, one per strip, balanced by the level row.
  (C)  chines: W is bounded by |y| = 1/2 aft of the chine-wetting point; they
       are dry forward of it because each strip starts at its own spray root.
       No chine condition is imposed (none can be in this linear theory,
       section 13.2 item C); monitored by `chine_ratio`.
Discretisation (section 13.3): half-beam strips (pressure constant across a
strip, mirror-symmetric about the keel); cells c = 0, 1, ... of length h
from the transom upstream ([-(c+1) h, -c h]); the load of cell c (per unit
span) acts at its 1/4 point -(c + 3/4) h (smeared over a narrow hat); the
slope condition is collocated at the 3/4 point -(c + 1/4) h of cells
0..N-1; cell N is the partially wetted spray-root cell, the level row is at
node -N h. The wetted fraction f of cell N follows from the load ratio
rho = P_N / P_{N-1} under the inverse-square-root spray-root behaviour,
rho = sqrt(f) / (sqrt(1 + f) - sqrt(f)); N is moved when f leaves [0, 1).
For a fixed set {N} the system is linear in (d, tan tau, tan beta).
"""
import math

import numpy as np
from scipy.optimize import brentq

from .planing_kernel import VLMTables

RHO1 = 1.0 / (math.sqrt(2.0) - 1.0)        # rho(f = 1)


def _dense_solve(A, b):
    """LU solve; GPU (float64) for large systems."""
    if A.shape[0] > 1200:
        try:
            import torch
            if torch.cuda.is_available():
                At = torch.as_tensor(A, dtype=torch.float64, device="cuda")
                bt = torch.as_tensor(b, dtype=torch.float64, device="cuda")
                return torch.linalg.solve(At, bt).cpu().numpy()
        except Exception:
            pass
    return np.linalg.solve(A, b)


def strip_edges(ns, cluster="cos"):
    i = np.arange(ns + 1)
    if cluster == "cos":
        return 0.25 * (1.0 - np.cos(np.pi * i / ns))
    return 0.5 * i / ns


def f_of_rho(rho):
    """Inverse of rho(f) = sqrt(f)/(sqrt(1+f) - sqrt(f)) on [0, 1]; values
    outside give f < 0 (rho < 0) or f >= 1 (rho >= RHO1)."""
    if not np.isfinite(rho) or rho < 0.0:
        return -1.0
    if rho >= RHO1:
        return 1.0 + (rho - RHO1)
    # sqrt(f) = rho (sqrt(1+f) - sqrt(f)) -> s (1 + rho) = rho sqrt(1 + s^2), s = sqrt(f)
    s = rho / math.sqrt((1.0 + rho) ** 2 - rho * rho)
    return s * s


class PrismaticPlaning:
    """Linear steady planing solver for one beam Froude number C_V and one
    deadrise (vortex-lattice-type discretisation of section 13.3). Kernel
    tables built once; Rayleigh damping removed by Richardson extrapolation
    (l_mu and 2 l_mu)."""

    def __init__(self, Cv, beta_deg, h=1.0 / 32, ns=8, cluster="cos", lam_max=6.0, c_mu=0.5,
                 l_mu=None, richardson_mu=True, ky_kw=None, device=None, a_frac=0.25, n_fold=24,
                 box_mult=8.0):
        self.Cv, self.beta_deg = float(Cv), float(beta_deg)
        self.tb = math.tan(math.radians(beta_deg))
        self.h, self.ns = float(h), int(ns)
        self.edges = strip_edges(ns, cluster)
        self.yc = 0.5 * (self.edges[1:] + self.edges[:-1])
        self.w = np.diff(self.edges)
        self.m_max = int(math.ceil(lam_max / h)) + 4
        kw = dict(c_mu=c_mu, ky_kw=ky_kw, device=device, a_frac=a_frac, n_fold=n_fold, box_mult=box_mult)
        T1 = VLMTables(Cv, h, self.edges, self.m_max, l_mu=l_mu, **kw)
        self.l_mu = T1.l_mu
        if richardson_mu:
            T2 = VLMTables(Cv, h, self.edges, self.m_max, l_mu=2.0 * T1.l_mu, **kw)
            self.K = {k: 2.0 * T2.K[k] - T1.K[k] for k in T1.K}
            self.K_damped = T1.K
        else:
            self.K = T1.K
            self.K_damped = T1.K

    # ------------------------------------------------------------------ assembly
    def assemble(self, n, mode="spray"):
        """Rows: slope at cells 0..N_t-1 of every strip, then (spray mode) one
        level row per strip at node N_t. Columns: loads of cells 0..N_s
        (spray) or 0..N_s-1 (planform)."""
        n = np.asarray(n, int)
        extra = 1 if mode == "spray" else 0
        cs = np.concatenate([np.full(n[s] + extra, s) for s in range(self.ns)])
        cj = np.concatenate([np.arange(n[s] + extra) for s in range(self.ns)])
        rt = np.concatenate([np.full(n[t], t) for t in range(self.ns)])
        ri = np.concatenate([np.arange(n[t]) for t in range(self.ns)])
        mm = self.m_max
        m = cj[None, :] - ri[:, None] + mm
        if m.min() < 0 or m.max() > 2 * mm:
            raise ValueError("strip longer than lam_max: increase lam_max")
        A = self.K["sx"][rt[:, None], cs[None, :], m]
        R = np.zeros((len(rt), 3))
        R[:, 1] = -1.0                                   # slope rows: zeta_x = -tan(tau)
        if mode == "spray":
            lt = np.arange(self.ns)
            ml = cj[None, :] - n[lt][:, None] + mm
            Al = self.K["zx"][lt[:, None], cs[None, :], ml]
            A = np.vstack([A, Al])
            Rl = np.stack([-np.ones(self.ns), n * self.h, self.yc], axis=1)
            R = np.vstack([R, Rl])
        return A, R, (cs, cj)

    # ------------------------------------------------------------------ outputs
    def _pieces(self, X, lay, n):
        cs, cj = lay
        L = np.zeros((self.ns, 3))
        Mo = np.zeros((self.ns, 3))
        PN = np.zeros((self.ns, 3))
        PN1 = np.zeros((self.ns, 3))
        lever = (cj + 0.75) * self.h
        for k in range(len(cs)):
            s = cs[k]
            L[s] += X[k]
            Mo[s] += lever[k] * X[k]
            if cj[k] == n[s]:
                PN[s] = X[k]
            if cj[k] == n[s] - 1:
                PN1[s] = X[k]
        return L, Mo, PN, PN1

    def _state(self, n, coef, pieces, mode="spray"):
        L, Mo, PN, PN1 = pieces
        v = np.asarray(coef, float)
        Ls, Ms, pn, pn1 = L @ v, Mo @ v, PN @ v, PN1 @ v
        if mode == "spray":
            f = np.array([f_of_rho(pn[t] / pn1[t]) if pn1[t] > 0 else -1.0 for t in range(self.ns)])
        else:
            f = np.zeros(self.ns)
        ell = (np.asarray(n) + np.clip(f, 0.0, 1.0)) * self.h
        Lt = 2.0 * np.sum(self.w * Ls)           # lift / (rho U^2 b^2), both sides
        Mt = 2.0 * np.sum(self.w * Ms)           # moment about the transom / (rho U^2 b^3)
        y0, y1 = self.yc[0], self.yc[1]
        LK = ell[0] - y0 * (ell[1] - ell[0]) / (y1 - y0)
        ya, yb = self.yc[-2], self.yc[-1]
        LC = ell[-1] + (0.5 - yb) * (ell[-1] - ell[-2]) / (yb - ya)
        return dict(f=f, ell=ell, lift=Lt, CL=2.0 * Lt, moment=Mt, lp=Mt / Lt if Lt != 0 else np.nan,
                    LK=LK, LC=LC, lam=0.5 * (LK + LC), area=2.0 * np.sum(self.w * ell),
                    strip_load=Ls, P_N=pn, P_N1=pn1)

    def _solve_nodes(self, n, mode):
        A, R, lay = self.assemble(n, mode)
        X = np.linalg.solve(A, R)
        return X, lay

    @staticmethod
    def _ok(f):
        return bool(np.all((f >= 0.0) & (f < 1.0)))

    def _update(self, n, f):
        n2 = np.array(n, int).copy()
        for t in range(self.ns):
            if f[t] >= 1.0:
                n2[t] = n[t] + 1
            elif f[t] < 0.0:
                n2[t] = max(2, n[t] - 1)
        return n2

    # ------------------------------------------------------------------ drivers
    def initial_nodes(self, tau, d, pile=1.3):
        tt = math.tan(tau)
        ell = pile * np.maximum(d - self.yc * self.tb, 0.05) / tt
        return np.maximum(3, np.round(ell / self.h).astype(int))

    def solve(self, tau, d, n0=None, max_iter=400):
        """Given trim tau (rad) and transom keel depth d (units b)."""
        n = self.initial_nodes(tau, d) if n0 is None else np.array(n0, int)
        coef = (d, math.tan(tau), self.tb)
        seen = set()
        for it in range(max_iter):
            X, lay = self._solve_nodes(n, "spray")
            st = self._state(n, coef, self._pieces(X, lay, n))
            if self._ok(st["f"]):
                st.update(n=n, iters=it + 1, converged=True, X=X, lay=lay, tau=tau, d=d)
                return st
            key = tuple(n)
            if key in seen:
                break
            seen.add(key)
            n = self._update(n, st["f"])
        st.update(n=n, iters=it + 1, converged=False, X=X, lay=lay, tau=tau, d=d)
        return st

    def solve_planform(self, tau, ell):
        """Given planform (wetted length per strip, units b; rounded to whole
        cells): the vortex-lattice lifting-surface problem with gravity (used
        for the high-speed thin-wing check)."""
        n = np.maximum(1, np.round(np.asarray(ell, float) / self.h).astype(int))
        X, lay = self._solve_nodes(n, "planform")
        st = self._state(n, (0.0, math.tan(tau), self.tb), self._pieces(X, lay, n), mode="planform")
        st.update(n=n, converged=True, X=X, lay=lay, tau=tau, d=0.0)
        return st

    def _lam_fixed(self, n, pieces, tt, d):
        L, Mo, PN, PN1 = pieces
        v = np.array([d, tt, self.tb])
        pn, pn1 = PN @ v, PN1 @ v
        f = np.array([f_of_rho(pn[t] / pn1[t]) if pn1[t] > 0 else -1.0 for t in range(self.ns)])
        ell = (np.asarray(n) + f) * self.h          # unclipped (continuous in d within the node set)
        y0, y1 = self.yc[0], self.yc[1]
        LK = ell[0] - y0 * (ell[1] - ell[0]) / (y1 - y0)
        ya, yb = self.yc[-2], self.yc[-1]
        LC = ell[-1] + (0.5 - yb) * (ell[-1] - ell[-2]) / (yb - ya)
        return 0.5 * (LK + LC)

    def solve_lambda(self, tau, lam, n0=None, max_iter=400):
        """Transom depth d such that the mean wetted length (L_K + L_C)/(2 b)
        equals lam (trim tau given): node sets updated from the d of the
        previous set (within a node set lam(d) is continuous)."""
        tt = math.tan(tau)
        n = (np.maximum(3, np.round(np.full(self.ns, lam) / self.h).astype(int)) if n0 is None
             else np.array(n0, int))
        seen = set()
        d = None
        st = None
        for it in range(max_iter):
            X, lay = self._solve_nodes(n, "spray")
            pieces = self._pieces(X, lay, n)
            g = lambda dd: self._lam_fixed(n, pieces, tt, dd) - lam
            grid = np.linspace(-0.2, 1.5, 341)
            vals = np.array([g(x) for x in grid])
            ok = np.isfinite(vals)
            br = np.nonzero(ok[:-1] & ok[1:] & (np.sign(vals[:-1]) != np.sign(vals[1:])))[0]
            if len(br) == 0:
                d = grid[np.nanargmin(np.abs(vals))] if np.any(ok) else (d or 0.1)
            else:
                cands = [brentq(g, grid[k], grid[k + 1], xtol=1e-13) for k in br]
                d = cands[0] if d is None else min(cands, key=lambda c: abs(c - d))
            st = self._state(n, (d, tt, self.tb), pieces)
            if self._ok(st["f"]):
                st.update(n=n, iters=it + 1, converged=True, tau=tau, d=d, X=X, lay=lay)
                return st
            key = tuple(n)
            if key in seen:
                break
            seen.add(key)
            n = self._update(n, st["f"])
        st.update(n=n, iters=it + 1, converged=False, tau=tau, d=d, X=X, lay=lay)
        return st

    def equilibrium(self, lift_target, lp_target, tau0=math.radians(4.0), d0=0.25, max_iter=400):
        """Free trim and sinkage: lift / (rho U^2 b^2) = lift_target and centre
        of pressure (from the transom, units b) = lp_target (moment balance
        of the pressure force about the centre of gravity at lp_target)."""
        n = self.initial_nodes(tau0, d0)
        seen = set()
        st = None
        for it in range(max_iter):
            X, lay = self._solve_nodes(n, "spray")
            L, Mo, PN, PN1 = self._pieces(X, lay, n)
            Lv = 2.0 * (self.w @ L)
            Mv = 2.0 * (self.w @ Mo)
            A2 = np.array([[Lv[0], Lv[1]], [Mv[0] - lp_target * Lv[0], Mv[1] - lp_target * Lv[1]]])
            b2 = np.array([lift_target - Lv[2] * self.tb, -(Mv[2] - lp_target * Lv[2]) * self.tb])
            d, T = np.linalg.solve(A2, b2)
            tau = math.atan(T) if T > 0 else float("nan")
            st = self._state(n, (d, T, self.tb), (L, Mo, PN, PN1))
            if T > 0 and self._ok(st["f"]):
                st.update(n=n, iters=it + 1, converged=True, tau=tau, d=d, X=X, lay=lay)
                return st
            key = tuple(n)
            if key in seen or T <= 0:
                break
            seen.add(key)
            n = self._update(n, st["f"])
        st.update(n=n, iters=it + 1, converged=False, tau=tau, d=d, X=X, lay=lay)
        return st

    # ------------------------------------------------------------------ diagnostics
    def loads(self, st):
        """Cell loads per strip (list of arrays from the transom, c = 0..N)."""
        cs, cj = st["lay"]
        v = np.array([st["d"], math.tan(st["tau"]), self.tb])
        x = st["X"] @ v
        return [x[cs == s] for s in range(self.ns)]

    def chine_ratio(self, st):
        """Strip-mean pressure extrapolated linearly to the chine, divided by
        the mean pressure over W (zero for smooth detachment)."""
        pm = st["strip_load"] / np.maximum(st["ell"], 1e-12)
        ya, yb = self.yc[-2], self.yc[-1]
        pc = pm[-1] + (0.5 - yb) * (pm[-1] - pm[-2]) / (yb - ya)
        mean = st["lift"] / max(st["area"], 1e-12)
        return pc / mean


def savitsky(tau_deg, lam, Cv, beta_deg):
    """Savitsky (1964) empirical planing equations: lift coefficient C_L,beta
    (Delta / (0.5 rho V^2 b^2)), centre of pressure from the transom / (lam
    b), and the keel-chine wetted-length difference / b (Wagner-type)."""
    CL0 = tau_deg ** 1.1 * (0.0120 * lam ** 0.5 + 0.0055 * lam ** 2.5 / Cv ** 2)
    CLb = CL0 - 0.0065 * beta_deg * CL0 ** 0.6
    cp = 0.75 - 1.0 / (5.21 * Cv ** 2 / lam ** 2 + 2.39)
    dK = math.tan(math.radians(beta_deg)) / (math.pi * math.tan(math.radians(tau_deg)))
    return dict(CL0=CL0, CL=CLb, cp=cp, LKmLC=dK)


class PrismaticPlaningHat(PrismaticPlaning):
    """Same formulation, piecewise-linear pressure (scheme H of section 13.3):
    nodes i = 0 (transom, p = 0: Kutta) .. N (spray-root node, half-hat: the
    pressure jumps there), plus an 'ext' box of length h upstream of node N
    whose coefficient c gives the wetted fraction f = c / p_N of the next
    cell. Rows: slope zeta_x = -tan(tau) at the midpoints of cells 0..N-1
    and the level zeta = h at node N. The piecewise-linear pressure makes the
    local (hydrostatic) part of the response visible at every row."""

    def __init__(self, Cv, beta_deg, h=1.0 / 32, ns=8, cluster="cos", lam_max=6.0, c_mu=0.5,
                 l_mu=None, richardson_mu=True, ky_kw=None, device=None, n_fold=8, box_mult=8.0,
                 lateral="box"):
        from .planing_kernel import VLMTables, HAT_SPECS, YHatTables
        if ky_kw is None:
            # k_y range: well beyond the hydrostatic transition of the shortest lattice mode,
            # k_y ~ C_V^2 (pi/h)^2 (section 13.3; exploration B7-E4)
            ky_kw = dict(kmax=max(2e6, 20.0 * Cv * Cv * (math.pi / h) ** 2))
        self.ky_kw = ky_kw
        self.Cv, self.beta_deg = float(Cv), float(beta_deg)
        self.tb = math.tan(math.radians(beta_deg))
        self.h, self.ns = float(h), int(ns)
        self.edges = strip_edges(ns, cluster)
        self.yc = 0.5 * (self.edges[1:] + self.edges[:-1])
        self.w = np.diff(self.edges)
        self.m_max = int(math.ceil(lam_max / h)) + 4
        self.lateral = lateral
        kw = dict(c_mu=c_mu, ky_kw=ky_kw, device=device, n_fold=n_fold, box_mult=box_mult, specs=HAT_SPECS)
        if lateral in ("hat", "hat0"):
            # ns = number of lateral intervals; nodes keel .. chine (section 13.3, scheme H-Y);
            # "hat0": no chine element, p = 0 at the chine (scheme H-Y0)
            self.nodes = strip_edges(ns, cluster)
            self.ns = ns + 1 if lateral == "hat" else ns
            self.yc = self.nodes
            Tab = YHatTables
            arg = self.nodes
            kw["chine"] = "jump" if lateral == "hat" else "zero"
            if lateral == "hat0":
                self.chine_extrap = False
        else:
            Tab = VLMTables
            arg = self.edges
        T1 = Tab(Cv, h, arg, self.m_max, l_mu=l_mu, **kw)
        if lateral in ("hat", "hat0"):
            self.w = T1.wy
            self.yc = T1.targets
        self.l_mu = T1.l_mu
        if richardson_mu:
            T2 = Tab(Cv, h, arg, self.m_max, l_mu=2.0 * T1.l_mu, **kw)
            self.K = {k: 2.0 * T2.K[k] - T1.K[k] for k in T1.K}
            self.K_damped = T1.K
        else:
            self.K = T1.K
            self.K_damped = T1.K

    def assemble(self, n, mode="spray"):
        n = np.asarray(n, int)
        # columns: (s, j, e) e: 0 hat (j=1..N-1), 1 half (j=N), 2 ext (j=N)
        cs, cj, ce = [], [], []
        for s in range(self.ns):
            for j in range(1, n[s] + 1):
                cs.append(s); cj.append(j); ce.append(0 if j < n[s] else 1)
            if mode == "spray":
                cs.append(s); cj.append(n[s]); ce.append(2)
        cs, cj, ce = np.array(cs), np.array(cj), np.array(ce)
        rt = np.concatenate([np.full(n[t], t) for t in range(self.ns)])
        ri = np.concatenate([np.arange(n[t]) for t in range(self.ns)])
        mm = self.m_max
        A = np.zeros((len(rt), len(cs)))
        names = ("s_hat", "s_half", "s_ext")
        for code in range(3):
            cm = np.nonzero(ce == code)[0]
            if len(cm) == 0:
                continue
            m = cj[cm][None, :] - ri[:, None] - 1 + mm
            if m.min() < 0 or m.max() > 2 * mm:
                raise ValueError("strip longer than lam_max: increase lam_max")
            A[:, cm] = self.K[names[code]][rt[:, None], cs[cm][None, :], m]
        R = np.zeros((len(rt), 3))
        R[:, 1] = -1.0
        if mode == "spray":
            lt = np.arange(self.ns)
            Al = np.zeros((self.ns, len(cs)))
            zn = ("z_hat", "z_half", "z_ext")
            for code in range(3):
                cm = np.nonzero(ce == code)[0]
                ml = cj[cm][None, :] - n[lt][:, None] + mm
                Al[:, cm] = self.K[zn[code]][lt[:, None], cs[cm][None, :], ml]
            A = np.vstack([A, Al])
            R = np.vstack([R, np.stack([-np.ones(self.ns), n * self.h, self.yc], axis=1)])
        else:
            # planform: unknowns p_1..p_N per strip, rows: N slopes -> add per-strip level offset? no:
            # drop the half-hat freedom is not needed; square already (N unknowns, N rows)
            pass
        return A, R, (cs, cj, ce)

    def _pieces(self, X, lay, n):
        cs, cj, ce = lay
        h = self.h
        w = np.where(ce == 0, h, np.where(ce == 1, 0.5 * h, h))
        lever = np.where(ce == 0, cj * h, np.where(ce == 1, cj * h - h / 3.0, (cj + 0.5) * h))
        L = np.zeros((self.ns, 3)); Mo = np.zeros((self.ns, 3))
        PN = np.zeros((self.ns, 3)); C = np.zeros((self.ns, 3))
        for k in range(len(cs)):
            s = cs[k]
            L[s] += w[k] * X[k]
            Mo[s] += w[k] * lever[k] * X[k]
            if ce[k] == 1:
                PN[s] = X[k]
            if ce[k] == 2:
                C[s] = X[k]
        return L, Mo, PN, C

    def _fvals(self, pn, c):
        return np.array([c[t] / pn[t] if pn[t] > 0 else -1.0 for t in range(self.ns)])

    def _state(self, n, coef, pieces, mode="spray"):
        L, Mo, PN, C = pieces
        v = np.asarray(coef, float)
        Ls, Ms, pn, c = L @ v, Mo @ v, PN @ v, C @ v
        f = self._fvals(pn, c) if mode == "spray" else np.zeros(self.ns)
        ell = (np.asarray(n) + np.clip(f, 0.0, 1.0)) * self.h
        Lt = 2.0 * np.sum(self.w * Ls)
        Mt = 2.0 * np.sum(self.w * Ms)
        y0, y1 = self.yc[0], self.yc[1]
        LK = ell[0] - y0 * (ell[1] - ell[0]) / (y1 - y0)
        ya, yb = self.yc[-2], self.yc[-1]
        LC = ell[-1] + (0.5 - yb) * (ell[-1] - ell[-2]) / (yb - ya)
        return dict(f=f, ell=ell, lift=Lt, CL=2.0 * Lt, moment=Mt, lp=Mt / Lt if Lt != 0 else np.nan,
                    LK=LK, LC=LC, lam=0.5 * (LK + LC), area=2.0 * np.sum(self.w * ell),
                    strip_load=Ls, P_N=pn, c_ext=c)

    def _lam_fixed(self, n, pieces, tt, d):
        L, Mo, PN, C = pieces
        v = np.array([d, tt, self.tb])
        f = self._fvals(PN @ v, C @ v)
        ell = (np.asarray(n) + f) * self.h
        y0, y1 = self.yc[0], self.yc[1]
        LK = ell[0] - y0 * (ell[1] - ell[0]) / (y1 - y0)
        ya, yb = self.yc[-2], self.yc[-1]
        LC = ell[-1] + (0.5 - yb) * (ell[-1] - ell[-2]) / (yb - ya)
        return 0.5 * (LK + LC)

    def loads(self, st):
        """Nodal pressures per strip (arrays from the transom, i = 0..N)."""
        cs, cj, ce = st["lay"]
        v = np.array([st["d"], math.tan(st["tau"]), self.tb])
        x = st["X"] @ v
        out = []
        for s in range(self.ns):
            p = np.zeros(int(st["n"][s]) + 1)
            for k in np.nonzero((cs == s) & (ce < 2))[0]:
                p[cj[k]] = x[k]
            out.append(p * self.h)
        return out


class PlaningSprayRoot(PrismaticPlaningHat):
    """Scheme H with the spray-root iteration of section 13.3 (production
    solver of the gate):
      * for a given wetted planform (spray-root node N_t per strip) the
        lifting-surface problem with gravity is solved: slope rows at the cell
        midpoints, Kutta p = 0 at the transom node (basis); its pressure is
        proportional to tan(tau) and independent of d and beta;
      * the spray-root (level) residual r_t = zeta(-N_t h) - h(-N_t h) is
        linear in (d, tan tau, tan beta);
      * the continuous spray-root position N_t + f_t is found from the
        linearisation r(N + f) = r(N) + J f, J_ts = r_t(N + e_s) - r_t(N)
        (one planform solve per strip), N moved until every f_t is in [0, 1);
        all outputs are interpolated the same way (first order in h).
    """

    def planform(self, n):
        """Planform solve for tan(tau) = 1: loads, moments (about the
        transom), elevation at the spray-root nodes."""
        n = np.asarray(n, int)
        A, R, lay = self.assemble(n, mode="planform")
        x = _dense_solve(A, -np.ones(A.shape[0]))              # slope rows: zeta_x = -1 * tan(tau)
        cs, cj, ce = lay
        h = self.h
        w = np.where(ce == 0, h, 0.5 * h)
        lever = np.where(ce == 0, cj * h, cj * h - h / 3.0)
        Ls = np.bincount(cs, weights=w * x, minlength=self.ns)
        Ms = np.bincount(cs, weights=w * lever * x, minlength=self.ns)
        mm = self.m_max
        z = np.zeros(self.ns)
        for code, nm in ((0, "z_hat"), (1, "z_half")):
            cm = np.nonzero(ce == code)[0]
            lt = np.arange(self.ns)
            ml = cj[cm][None, :] - n[lt][:, None] + mm
            z += self.K[nm][lt[:, None], cs[cm][None, :], ml] @ x[cm]
        return dict(x=x, lay=lay, Ls=Ls, Ms=Ms, z=z, n=n)

    def _residual(self, P, d, T):
        n = P["n"]
        hN = -d + n * self.h * T + self.yc * self.tb
        return T * P["z"] - hN

    def _bundle(self, n):
        base = self.planform(n)
        nb = [self.planform(n + np.eye(self.ns, dtype=int)[s]) for s in range(self.ns)]
        return base, nb

    def _interp(self, base, nb, f, key):
        q = base[key].copy() if np.ndim(base[key]) else base[key]
        for s in range(self.ns):
            q = q + f[s] * (nb[s][key] - base[key])
        return q

    def _outputs(self, base, nb, f, d, T):
        Ls = self._interp(base, nb, f, "Ls") * T
        Ms = self._interp(base, nb, f, "Ms") * T
        ell = (base["n"] + f) * self.h
        Lt = 2.0 * np.sum(self.w * Ls)
        Mt = 2.0 * np.sum(self.w * Ms)
        y0, y1 = self.yc[0], self.yc[1]
        LK = ell[0] - y0 * (ell[1] - ell[0]) / (y1 - y0)
        ya, yb = self.yc[-2], self.yc[-1]
        LC = ell[-1] + (0.5 - yb) * (ell[-1] - ell[-2]) / (yb - ya)
        return dict(f=f, ell=ell, lift=Lt, CL=2.0 * Lt, moment=Mt, lp=Mt / Lt if Lt != 0 else np.nan,
                    LK=LK, LC=LC, lam=0.5 * (LK + LC), area=2.0 * np.sum(self.w * ell), strip_load=Ls,
                    d=d, tau=math.atan(T), n=base["n"].copy())

    def _iterate(self, n, step, max_iter=200):
        """Generic node-set iteration: `step(base, nb)` returns (f, d, T) for
        the current node set (linearised conditions)."""
        n = np.array(n, int)
        seen = {}
        for it in range(max_iter):
            if np.any(n < 2) or np.any(n > 2 * self.m_max // 2 - 4):
                n = np.clip(n, 2, self.m_max - 5)
            base, nb = self._bundle(n)
            f, d, T = step(base, nb)
            viol = float(np.max(np.maximum(-f, f - 1.0)))
            if viol < 0.0 or (seen.get(tuple(n), 0) >= 1 and viol <= self.f_slack):
                # accepted; a slack of f_slack cells is allowed when the iteration returns to a node set
                # (the spray root sits within f_slack h of a lattice node; first-order interpolation)
                out = self._outputs(base, nb, np.clip(f, 0.0, 1.0), d, T)
                out.update(converged=True, iters=it + 1, f_violation=max(viol, 0.0))
                out["_bundle"] = (base, nb)
                return out
            key = tuple(n)
            seen[key] = seen.get(key, 0) + 1
            fin = np.where(np.isfinite(f), f, -1.0)
            shift = np.floor(fin).astype(int)
            shift = np.clip(shift, -self.max_shift, self.max_shift)
            if seen[key] > 1:          # cycling: move one strip at a time, the worst first
                k = int(np.argmax(np.abs(f - 0.5)))
                shift = np.zeros(self.ns, int)
                shift[k] = int(np.sign(np.floor(f[k]))) or 1
                if seen[key] > 3:
                    break
            n = n + shift
        out = self._outputs(base, nb, np.clip(f, 0.0, 1.0), d, T)
        out.update(converged=False, iters=it + 1, f_raw=f)
        return out

    max_shift = 2
    f_slack = 0.15

    def wagner_nodes(self, tau, lam):
        """Initial spray-root nodes: straight line with the slender-body (Wagner,
        Tulin) keel-chine difference tan(beta)/(pi tan(tau)) about the mean
        wetted length lam."""
        dKC = self.tb / (math.pi * math.tan(tau))
        ell = lam + dKC * (0.5 - 2.0 * np.asarray(self.yc)) / 1.0
        ell = np.maximum(ell, 3 * self.h)
        return np.maximum(3, np.round(ell / self.h).astype(int))

    chine_extrap = True

    def _chine_row(self, r0, J, n):
        """Replace the chine line's level condition by linear extrapolation of the
        spray-root line from the two neighbouring lines (section 13.3: the chine
        line lies on the pressure jump of item C, where the level residual is
        not a usable condition; exploration B7-E10)."""
        if not self.chine_extrap or self.ns < 4:
            return r0, J
        y = np.asarray(self.yc)
        a, b, c = y[-3], y[-2], y[-1]
        wa = (c - b) / (b - a)
        r0 = r0.copy()
        J = J.copy()
        # (N+f)_c - (N+f)_b - wa ((N+f)_b - (N+f)_a) = 0
        r0[-1] = n[-1] - n[-2] - wa * (n[-2] - n[-3])
        J[-1, :] = 0.0
        J[-1, -1] = 1.0
        J[-1, -2] = -(1.0 + wa)
        J[-1, -3] = wa
        return r0, J

    def solve(self, tau, d, n0=None, max_iter=200):
        T = math.tan(tau)
        if n0 is None:
            lam_est = max(1.3 * d / T - 0.5 * self.tb / (math.pi * T), 0.3)
            n0 = self.wagner_nodes(tau, lam_est)
        n = np.array(n0, int)

        def step(base, nb):
            r0 = self._residual(base, d, T)
            J = np.stack([self._residual(nb[s], d, T) - r0 for s in range(self.ns)], axis=1)
            r0, J = self._chine_row(r0, J, base["n"])
            f = np.linalg.solve(J, -r0)
            return f, d, T
        return self._iterate(n, step, max_iter)

    def solve_lambda(self, tau, lam, n0=None, max_iter=200):
        T = math.tan(tau)
        n = self.wagner_nodes(tau, lam) if n0 is None else np.array(n0, int)
        y0, y1 = self.yc[0], self.yc[1]
        ya, yb = self.yc[-2], self.yc[-1]
        # lam = 0.5 (LK + LC) as a linear functional of ell
        g = np.zeros(self.ns)
        g[0] += 0.5 * (1 + y0 / (y1 - y0))
        g[1] += 0.5 * (-y0 / (y1 - y0))
        g[-1] += 0.5 * (1 + (0.5 - yb) / (yb - ya))
        g[-2] += 0.5 * (-(0.5 - yb) / (yb - ya))

        def step(base, nb):
            # unknowns (f_1..f_ns, d): r(N) + J f + d * 1 = 0 (r linear in d with slope +1), g.(N+f) h = lam
            r0 = self._residual(base, 0.0, T)
            J = np.stack([self._residual(nb[s], 0.0, T) - r0 for s in range(self.ns)], axis=1)
            r0, J = self._chine_row(r0, J, base["n"])
            A = np.zeros((self.ns + 1, self.ns + 1))
            A[: self.ns, : self.ns] = J
            A[: self.ns, self.ns] = 1.0
            if self.chine_extrap and self.ns >= 4:
                A[self.ns - 1, self.ns] = 0.0           # the extrapolation row does not involve d
            A[self.ns, : self.ns] = g * self.h
            b = np.concatenate([-r0, [lam - self.h * (g @ base["n"])]])
            sol = np.linalg.solve(A, b)
            return sol[: self.ns], sol[self.ns], T
        return self._iterate(n, step, max_iter)

    def equilibrium(self, lift_target, lp_target, tau0=math.radians(4.0), lam0=2.5, tol=1e-7, max_iter=40):
        """Free trim and sinkage at given lift (/(rho U^2 b^2)) and centre of
        gravity lp_target (from the transom, units b): secant iteration in
        (tau, lam) on top of solve_lambda (lift and moment balance)."""
        from scipy.optimize import fsolve

        cache = {}

        def F(v):
            tau, lam = v
            st = self.solve_lambda(tau, lam)
            cache["st"] = st
            return [st["lift"] / lift_target - 1.0, st["lp"] - lp_target]
        sol, info, ier, msg = fsolve(F, [tau0, lam0], full_output=True, xtol=tol)
        st = self.solve_lambda(sol[0], sol[1])
        st.update(eq_ok=(ier == 1), eq_msg=msg, eq_nfev=info["nfev"])
        return st

    def chine_ratio(self, st):
        pm = st["strip_load"] / np.maximum(st["ell"], 1e-12)
        ya, yb = self.yc[-2], self.yc[-1]
        pc = pm[-1] + (0.5 - yb) * (pm[-1] - pm[-2]) / (yb - ya)
        mean = st["lift"] / max(st["area"], 1e-12)
        return pc / mean


class PlaningSprayRootFast(PlaningSprayRoot):
    """PlaningSprayRoot with the neighbour planforms N + e_s obtained from the
    base factorisation (one LU per node set): the added cell changes the
    spray-root column of line s (half-hat -> hat, a rank-one update), adds the
    new half-hat column and the slope row of the new cell (a border); the
    Sherman-Morrison formula and the Schur complement give each neighbour in
    O(m^2) (identical to a fresh solve up to round-off; checked in
    exploration B7-E8)."""

    def _cols_vals(self, name_by_code, ce):
        return [name_by_code[c] for c in ce]

    def _bundle(self, n):
        n = np.asarray(n, int)
        A, R, lay = self.assemble(n, mode="planform")
        cs, cj, ce = lay
        m = A.shape[0]
        mm = self.m_max
        rt = np.concatenate([np.full(n[t], t) for t in range(self.ns)])
        ri = np.concatenate([np.arange(n[t]) for t in range(self.ns)])
        b = -np.ones(m)
        h = self.h
        csl = [np.nonzero((cs == s) & (ce == 1))[0][0] for s in range(self.ns)]
        Wc = np.zeros((m, self.ns))
        Uc = np.zeros((m, self.ns))
        for s in range(self.ns):
            Ns = n[s]
            Wc[:, s] = self.K["s_hat"][rt, s, Ns - ri - 1 + mm] - A[:, csl[s]]
            Uc[:, s] = self.K["s_half"][rt, s, (Ns + 1) - ri - 1 + mm]
        Y = _dense_solve(A, np.concatenate([b[:, None], Wc, Uc], axis=1))
        y0, Yw, Yu = Y[:, 0], Y[:, 1:1 + self.ns], Y[:, 1 + self.ns:]
        names = np.array(["s_hat", "s_half"])
        znames = np.array(["z_hat", "z_half"])
        wbase = np.where(ce == 0, h, 0.5 * h)
        lbase = np.where(ce == 0, cj * h, cj * h - h / 3.0)

        def zrow(t, node, ce_mod):
            r = np.zeros(m)
            for code in (0, 1):
                cm = np.nonzero(ce_mod == code)[0]
                r[cm] = self.K[znames[code]][t, cs[cm], cj[cm] - node + mm]
            return r

        Z = np.stack([zrow(t, n[t], ce) for t in range(self.ns)])
        base = dict(x=y0, lay=lay, n=n,
                    Ls=np.bincount(cs, weights=wbase * y0, minlength=self.ns),
                    Ms=np.bincount(cs, weights=wbase * lbase * y0, minlength=self.ns),
                    z=Z @ y0)
        nbs = []
        for s in range(self.ns):
            Ns = n[s]
            c = csl[s]
            yw, yu = Yw[:, s], Yu[:, s]
            den = 1.0 + yw[c]
            p1 = y0 - yw * (y0[c] / den)
            p2 = yu - yw * (yu[c] / den)
            ce_mod = ce.copy()
            ce_mod[c] = 0
            # new slope row: cell i = Ns of line s
            r = np.zeros(m)
            for code in (0, 1):
                cm = np.nonzero(ce_mod == code)[0]
                r[cm] = self.K[names[code]][s, cs[cm], cj[cm] - Ns - 1 + mm]
            delta = self.K["s_half"][s, s, (Ns + 1) - Ns - 1 + mm]
            xi = (-1.0 - r @ p1) / (delta - r @ p2)
            x = p1 - p2 * xi
            w = wbase.copy()
            lev = lbase.copy()
            w[c] = h
            lev[c] = cj[c] * h
            Ls = np.bincount(cs, weights=w * x, minlength=self.ns)
            Ms = np.bincount(cs, weights=w * lev * x, minlength=self.ns)
            Ls[s] += 0.5 * h * xi
            Ms[s] += 0.5 * h * ((Ns + 1) * h - h / 3.0) * xi
            z = np.zeros(self.ns)
            for t in range(self.ns):
                node = n[t] + (1 if t == s else 0)
                zr = zrow(t, node, ce_mod) if t == s else Z[t].copy()
                if t != s:
                    zr[c] = self.K["z_hat"][t, s, Ns - node + mm]
                z[t] = zr @ x + self.K["z_half"][t, s, (Ns + 1) - node + mm] * xi
            nn = n.copy()
            nn[s] += 1
            nbs.append(dict(Ls=Ls, Ms=Ms, z=z, n=nn))
        return base, nbs


class PlaningSmoothLE(PlaningSprayRootFast):
    """Production solver of the gate (section 13.3): as PlaningSprayRootFast, but
    the spray-root line is represented by le_modes Legendre polynomials in
    s = 4 |y| - 1 (|y| <= 1/2; a function of |y|, so the keel apex of a V-hull is
    represented) and the line-wise level conditions are satisfied in the least-
    squares sense (Gauss-Newton on the linearisation r + J f). Reason
    (exploration B7-E9): with per-line positions the system is ill-conditioned
    for lateral oscillations of the spray-root line when neighbouring lines
    have nearly the same wetted length (flat and low-deadrise plates); a
    smooth spray-root line removes those modes. The residual is reported as
    an equivalent spray-root shift in cells (`le_res_cells`)."""

    le_modes = 6

    def _basis(self):
        from numpy.polynomial import legendre
        s = 4.0 * np.abs(np.asarray(self.yc)) - 1.0
        K = min(self.le_modes, self.ns)
        return np.stack([legendre.legval(s, np.eye(K)[k]) for k in range(K)], axis=1)

    def _ls_step(self, base, nb, T, d_fixed=None, lam=None, g=None):
        n = base["n"]
        r0 = self._residual(base, 0.0 if d_fixed is None else d_fixed, T)
        J = np.stack([self._residual(nb[s], 0.0 if d_fixed is None else d_fixed, T) - r0 for s in range(self.ns)],
                     axis=1)
        B = self._basis() / self.h                    # f = B a - n  (a: ell coefficients in units b)
        K = B.shape[1]
        if d_fixed is None:
            A = np.zeros((self.ns + 1, K + 1))
            A[: self.ns, :K] = J @ B
            A[: self.ns, K] = 1.0
            rhs = np.concatenate([-r0 + J @ n, [lam]])
            A[self.ns, :K] = (g @ B) * self.h
            wgt = np.ones(self.ns + 1)
            wgt[self.ns] = 1e4 / max(np.abs(J).max(), 1e-12) ** -1 if False else 1e4
            sol = np.linalg.lstsq(A * wgt[:, None], rhs * wgt, rcond=None)[0]
            a, d = sol[:K], sol[K]
        else:
            A = J @ B
            rhs = -r0 + J @ n
            a = np.linalg.lstsq(A, rhs, rcond=None)[0]
            d = d_fixed
        f = B @ a - n
        res = r0 + J @ f + (d if d_fixed is None else 0.0)
        self._last_res = np.abs(res / np.diag(J)).max()
        return f, d, T

    def _iterate(self, n, step, max_iter=200):
        n = np.array(n, int)
        seen = {}
        for it in range(max_iter):
            n = np.clip(n, 2, self.m_max - 5)
            base, nb = self._bundle(n)
            f, d, T = step(base, nb)
            if np.all((f >= 0.0) & (f < 1.0)):
                out = self._outputs(base, nb, f, d, T)
                out.update(converged=bool(self._last_res <= 1.0), iters=it + 1, le_res_cells=float(self._last_res))
                return out
            key = tuple(n)
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > 2:
                break
            shift = np.clip(np.floor(f).astype(int), -self.max_shift, self.max_shift)
            n = n + shift
        out = self._outputs(base, nb, np.clip(f, 0.0, 1.0), d, T)
        out.update(converged=False, iters=it + 1, f_raw=f, le_res_cells=float(self._last_res))
        return out

    def solve(self, tau, d, n0=None, max_iter=200):
        T = math.tan(tau)
        if n0 is None:
            lam_est = max(1.3 * d / T - 0.5 * self.tb / (math.pi * T), 0.3)
            n0 = self.wagner_nodes(tau, lam_est)
        return self._iterate(n0, lambda base, nb: self._ls_step(base, nb, T, d_fixed=d), max_iter)

    def solve_lambda(self, tau, lam, n0=None, max_iter=200):
        T = math.tan(tau)
        n = self.wagner_nodes(tau, lam) if n0 is None else np.array(n0, int)
        y0, y1 = self.yc[0], self.yc[1]
        ya, yb = self.yc[-2], self.yc[-1]
        g = np.zeros(self.ns)
        g[0] += 0.5 * (1 + y0 / (y1 - y0))
        g[1] += 0.5 * (-y0 / (y1 - y0))
        g[-1] += 0.5 * (1 + (0.5 - yb) / (yb - ya))
        g[-2] += 0.5 * (-(0.5 - yb) / (yb - ya))
        return self._iterate(n, lambda base, nb: self._ls_step(base, nb, T, lam=lam, g=g), max_iter)
