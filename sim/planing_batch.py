"""
Batch mode of the any-speed plant (step P1c; ACCEPTANCE E1: "扫描类计算提供批量
模式"): N boats of one as hull and one switch set, each in its own sea and
under its own commands, stepped in one vectorised call -- the same bits as
N separate plants. Not a fast path for one boat: at N = 1 it is slower than
the plant's own step, and the single plant never goes through it.

    from sim.planing_batch import BatchAS
    plants = [PlaningVesselAS(hull, sea_i, dt=0.02, ...) for sea_i in seas]
    B = BatchAS(plants)
    S = np.stack([p.initial_state(u0) for p in plants])   # (N, 14)
    for i in range(n):
        S = B.step(S, t, thrust_cmds, nozzle_cmds)         # (N,) each
        t += B.dt
    plants[i].last_cg_acc, ...   each plant's measurements, as its own step

WHAT IT IS. The derivative is sim/planing_anyspeed._StripAS.deriv for the
switch sets it accepts (SETS: P1a's, P1a + fk_exact; either with the
implementation switch fast_path), written for arrays of boats line by line:
the same operations in the same order on the same operands, so that every
elementwise result is the plant's bit for bit (IEEE arithmetic does not
depend on the array a number sits in). The reductions are the delicate
part, and each one is matched to what the plant does:

  - sums over the n stations: numpy's pairwise sum of each boat's row
    (the plant sums a 1-D array of the same n numbers in the same order);
  - the mean keel height of the wet stations (the friction's height), a
    sum over a SUBSET in the plant: numpy's own np.mean of that subset,
    boat by boat (no cache: P1c review);
  - products over the wave components: per-boat stacked products of the
    plant's shapes (numpy's matmul calls BLAS once per stacked matrix: the
    same call as the plant's); the exact Froude-Krylov pressure goes
    through sim/as_kernels.fk_rows, row-stable by construction, which the
    plant calls with one boat;
  - scalar Python arithmetic of the plant (the jet, the transom face, the
    restoring arm table): the same IEEE operations on arrays (numpy's sin,
    cos, exp, log10, sqrt, remainder, interp and power with exponents 3, 4
    and 1/3 give the same bits on scalars and on arrays: checked on this
    machine) or, for a few boats, the plant's own scalar code. The one
    exception found: x ** 2 of a numpy scalar is C's pow, of an array
    np.square (x * x), and the two differ in the last bit for some x (the
    friction line's square: done per boat on scalars, as the plant).

The plants own their actuators (the pump's prime, the delay lines, the
lags) and their measurements: the batch step advances them with the
plants' own methods, boat by boat (scalar work, a few per cent of the
step), and writes each plant's measurements as its step would (record=True
plants get their log rows). The step is the plant's (_StripAS._step_verlet
through PlaningVesselGZ.step and _ASMixin.step): the prime from the intake
submergence, the first derivative, v0's substep rule on the lateral rate
(boats needing substeps grouped by their count), the Verlet stage, the
actuators, the measurements; a GZ boat that has capsized holds its state.
Boats are processed in chunks of CHUNK (the arrays then stay in cache;
row-stable, so the chunking changes no bit).

Not implemented (refused): the wagner_entry, orbital and lateral_phi0
switches, captive plants, the transom closure's sensitivity modes; and,
until a batch check covers them (P1c review), stations that are not equally
spaced (the plant's _waves_b then takes cos and sin of every phase instead
of the running product sim/as_kernels.phases uses) and a chine with
interior knots inside the strips (an offsets-table hull: _taper().knots;
both registered as1 hulls have none).

CHECK. studies/anyspeed_p1c_check.py part batch: N plants against the
batch over full runs in different seas (calm included), every state and
measurement equal; part fastpath: the golden runs with fast_path equal to
the reference path.
"""
import numpy as np

from . import as_kernels as _ak
from . import planing_anyspeed as PA
from . import planing_vessel as _pv
from .planing_vessel import G, SP
from .planing_vessel_gz import PlaningVesselGZ, PHI_TAB

# the switch sets the batch implements (the physics; 'fast_path' is an
# implementation switch and may be added to either)
SETS = (PA.AS1_P1A, PA.AS1_P1A + ("fk_exact",))
# boats per chunk of the vectorised work; at most this many boats take the
# plants' own scalar code for the jet and the transom face
CHUNK = 8
SCALAR_MAX = 4


class BatchAS:
    """N plants (PlaningVesselAS or PlaningVesselASLin) of one hull, one
    switch set in SETS and one step, each with its own sea, stepped together
    (module docstring)."""

    def __init__(self, plants, chunk=CHUNK):
        plants = list(plants)
        if not plants:
            raise ValueError("no plants")
        p0 = plants[0]
        sw = tuple(s for s in p0.switches if s != "fast_path")
        if sw not in SETS:
            raise NotImplementedError(
                f"BatchAS implements the switch sets {SETS} (+ fast_path); "
                f"got {p0.switches}")
        cls = type(p0)
        self.gz = isinstance(p0, PlaningVesselGZ)
        for p in plants:
            if type(p) is not cls or p.switches != p0.switches:
                raise ValueError("the plants differ in class or switches")
            if p.hull is not p0.hull:
                raise ValueError("the plants must share one hull object")
            if float(p.dt) != float(p0.dt):
                raise ValueError("the plants differ in step")
            if p.captive_u is not None:
                raise NotImplementedError("captive plants")
            if getattr(p.hull, "transom_correction", True):
                raise NotImplementedError("transom_correction")
            if p.prop.delay.n != p0.prop.delay.n:
                raise ValueError("the plants' delay lines differ")
        tg = p0._transom_geom()
        if tg.cl.mode != "progressive":
            raise NotImplementedError(f"transom closure mode {tg.cl.mode}")
        # geometries no batch check covers (module docstring): stations not
        # equally spaced (_StripAS._waves_b's test) and interior chine knots
        if not bool(np.all(np.abs(np.diff(p0.x) - p0.dx)
                           <= 1e-12 * p0.dx)):
            raise NotImplementedError("stations not equally spaced")
        if p0._taper().knots is not None:
            raise NotImplementedError("a chine with interior knots in the "
                                      "strips (offsets table)")
        self.plants = plants
        self.N = len(plants)
        self.chunk = max(int(chunk), 1)
        self.fk = "fk_exact" in sw
        self.dt = float(p0.dt)
        self._geometry(p0)
        self._seas(plants)
        self._jetc(plants)

    # ------------------------------------------------------------ set up
    def _geometry(self, p):
        h = p.hull
        tg = p._transom_geom()
        self.H = H = dict(
            x=p.x, dx=p.dx, n=p.x.size, yc=p.yc, zk=p.zk, kp=p.kp, kpp=p.kpp,
            tanb=p.tanb, cosb=p.cosb, k_a=p.k_a, rho=p.rho, c_dc=p.c_dc,
            cd_lat=p.cd_lat, m=p.m, Iy=p.Iy, Ix=p.Ix, GM=p.GM, Mtot=p.Mtot,
            Z_rest=p.Z_rest, Th_rest=p.Th_rest, a_b=h.a_b, b=h.b,
            cda=h.cda, roll_zeta=h.roll_zeta, num=p.num, tg=tg,
            xn=p.thrusters[0][0], zn=p.thrusters[0][2], keel=p.sec.keel,
            v_slam=p.v_slam, intake=p.intakes[0])
        H["wn_r"] = np.sqrt(max(p.m * G * p.GM, 1e-6) / p.Ix)
        H["SPx"] = SP * p.x
        H["dxT"] = tg.xT - p.x[0]
        if self.gz:
            H["r_tab"] = p.r_tab
            H["phi_vanish"] = p.phi_vanish
        self._taper = p._taper()
        if self.fk:
            self._fkg = p._fk_geo()

    def _seas(self, plants):
        N = self.N
        for p in plants:
            if not hasattr(p, "_wa"):
                p._wave_arrays()
        self.wavy = np.array([p._wa is not None for p in plants])
        iw = np.nonzero(self.wavy)[0]
        self.M = 0
        self.W = None
        if not iw.size:
            return
        Ms = {plants[i]._wa.shape[1] for i in iw}
        if len(Ms) != 1:
            raise ValueError(f"the seas differ in their number of "
                             f"components {sorted(Ms)}")
        M = self.M = Ms.pop()
        W = dict(kx=np.zeros((N, M)), ky=np.zeros((N, M)),
                 ww=np.zeros((N, M)), wphi=np.zeros((N, M)),
                 wa=np.zeros((N, 5, M)))
        for i in iw:
            p = plants[i]
            W["kx"][i], W["ky"][i] = p._wkx, p._wky
            W["ww"][i], W["wphi"][i] = p._ww, p._wphi
            W["wa"][i] = p._wa
        W["wa0"] = np.ascontiguousarray(W["wa"][:, 0])[:, :, None]
        if self.fk:
            sts = {i: plants[i]._wave_setup_b() for i in iw}
            nu = max(st.ku.size for st in sts.values())
            W.update(a=np.zeros((N, M)), k=np.ones((N, M)),
                     ku=np.ones((N, nu)), kinv=np.zeros((N, M), int),
                     cc=np.zeros((N, M, 2)), sc=np.zeros((N, M, 3)))
            for i, st in sts.items():
                W["a"][i], W["k"][i] = st.a, st.k
                W["ku"][i, :st.ku.size] = st.ku
                W["kinv"][i] = st.kinv
                W["cc"][i], W["sc"][i] = st.cc, st.sc
        self.W = W

    def _jetc(self, plants):
        """The jet's constants (the vectorised Waterjet.forces); the plants'
        jets must agree."""
        p0 = plants[0]
        pr, pump = p0.prop, p0.prop.pump

        def key(q):
            return (q.pump.area, q.pump.p_max, q.pump.eta, q.pump.zeta_n,
                    q.pump.zeta_i, q.pump.wake, q.pump.rho, q.k_rev,
                    q.dead_band, q.u_ref)
        for p in plants:
            if key(p.prop) != key(pr):
                raise ValueError("the plants' jets differ")
        self.J = dict(area=pump.area, eta=pump.eta, zeta_n=pump.zeta_n,
                      zeta_i=pump.zeta_i, wake=pump.wake, rho=pump.rho,
                      k_rev=pr.k_rev, dead_band=pr.dead_band, u_ref=pr.u_ref)

    # ------------------------------------------------------------ the jet
    def _jet(self, thr, rud, u, bi):
        """Waterjet.forces for the boats bi: the plants' own (a few boats) or
        the same operations on arrays (flow, power_for_thrust,
        jet_velocity's Newton iteration with its per-boat exits)."""
        m = bi.size
        if m <= SCALAR_MAX:
            fx, fy = np.empty(m), np.empty(m)
            for j in range(m):
                fx[j], fy[j] = self.plants[bi[j]].prop.forces(
                    thr[j], rud[j], u[j])
            return fx, fy
        J = self.J
        prime = np.array([self.plants[i].prop.prime for i in bi], float)
        rho, area = J["rho"], J["area"]
        vin_u = (1.0 - J["wake"]) * np.maximum(u, 0.0)
        t = np.abs(thr)
        zero = t <= 0.0
        db = max(J["dead_band"], 1e-9)
        below = t < db
        frac = np.where(below, t / db, 1.0)
        t = np.where(below, db, t)
        vr = (1.0 - J["wake"]) * max(float(J["u_ref"]), 0.0)
        vjp = 0.5 * vr + np.sqrt(0.25 * vr * vr
                                 + np.maximum(t, 0.0) / (rho * area))
        pw = (0.5 * rho * area * vjp * ((1.0 + J["zeta_n"]) * vjp * vjp
                                        - (1.0 - J["zeta_i"]) * vr * vr)
              / J["eta"])
        a = 1.0 + J["zeta_n"]
        b = (1.0 - J["zeta_i"]) * vin_u * vin_u
        c = 2.0 * J["eta"] * np.maximum(pw, 0.0) / (rho * area)
        v = (c / a) ** (1.0 / 3.0) + np.sqrt(b / a)
        act = np.ones(m, bool)
        for _ in range(8):
            f = a * v ** 3 - b * v - c
            fp = 3.0 * a * v * v - b
            act = act & ~(fp <= 0.0)
            if not act.any():
                break
            dv = np.where(act, f / np.where(act, fp, 1.0), 0.0)
            v = np.where(act, v - dv, v)
            act = act & ~(np.abs(dv) < 1e-10 * np.maximum(v, 1.0))
            if not act.any():
                break
        vj = np.where(zero, 0.0, v)
        m_dot = np.where(zero, 0.0, frac * prime * rho * area * v)
        cd, sd = np.cos(rud), np.sin(rud)
        fwd = thr >= 0.0
        kr = J["k_rev"]
        fx = np.where(fwd, m_dot * (vj * cd - vin_u),
                      -m_dot * (kr * vj * cd + vin_u))
        fy = np.where(fwd, m_dot * vj * sd, -kr * m_dot * vj * sd)
        return fx, fy

    # ------------------------------------------------------ the derivative
    def _substrip(self, d, slope):
        """_StripAS._substrip for (m, n) arrays: (on (m, n), ycap, iunc (m,
        n)) -- the same per-strip arithmetic."""
        tg = self._taper
        g = tg.g
        kc = tg.kc
        idx = tg.idx
        di, mi = d[:, idx], slope[:, idx]
        h2 = tg.h2
        ca = kc * np.maximum(di - mi * h2, 0.0) >= tg.yca
        cb = kc * np.maximum(di + mi * h2, 0.0) >= tg.ycb
        on = ca | cb
        # (no interior chine knots: refused in __init__)
        if not on.any():
            return on, idx, None, None
        gen = on & ~(tg.flat & ca & cb)
        ycap = np.zeros(di.shape)
        iunc = np.zeros(di.shape)
        if not gen.any():
            return on, idx, ycap, iunc
        bi, si = np.nonzero(gen)
        dg, mg = di[bi, si], mi[bi, si]
        xa, xb, xc = tg.xa[si], tg.xb[si], tg.xc[si]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            roots = g.roots(xc, dg, mg, kc)
        P = np.stack([xa, xb] + list(roots), axis=1)
        P = np.where(np.isfinite(P), P, xb[:, None])
        P = np.clip(P, xa[:, None], xb[:, None])
        P.sort(axis=1)
        lo, hi = P[:, :-1], P[:, 1:]
        ln = hi - lo
        dc, mc = dg[:, None], mg[:, None]
        xcc = xc[:, None]
        mid = 0.5 * (lo + hi)
        dmid = dc + mc * (mid - xcc)
        k = lo.shape[1]
        ych = g.chine(np.concatenate([mid, lo, hi], axis=1))
        cap = (ln > 0.0) & (dmid > 0.0) & (kc * dmid >= ych[:, :k])
        ylo, yhi = ych[:, k:2 * k], ych[:, 2 * k:]
        yc_ = np.where(cap, ylo * ylo - yhi * yhi, 0.0).sum(axis=1)
        dlo, dhi = dc + mc * (lo - xcc), dc + mc * (hi - xcc)
        plo, phi_ = np.maximum(dlo, 0.0), np.maximum(dhi, 0.0)
        mixed = (dlo > 0.0) != (dhi > 0.0)
        part = np.where(mixed, 0.5 * np.maximum(plo, phi_) ** 2 * ln
                        / np.where(mixed, np.abs(dhi - dlo), 1.0),
                        0.5 * (plo + phi_) * ln)
        iu = np.where(cap, 0.0, part).sum(axis=1)
        ycap[bi, si] = yc_
        iunc[bi, si] = iu
        return on, idx, ycap, iunc

    def deriv(self, S, t, bi=None):
        """The plants' derivatives at states S (m, 14) and times t (m,) for
        the boats bi (indices into the batch; all by default): (ds (m, 14),
        aux dict of (m, ...) arrays -- d, V, mom, wd, qd, mom_full, fx,
        fy)."""
        S = np.asarray(S, float)
        m = S.shape[0]
        bi = np.arange(m) if bi is None else np.asarray(bi)
        t = np.broadcast_to(np.asarray(t, float), (m,))
        H = self.H
        x = H["x"]
        n = H["n"]
        dx, rho = H["dx"], H["rho"]
        x0, y0, z, phi, th, psi = S[:, 0], S[:, 1], S[:, 2], S[:, 3], \
            S[:, 4], S[:, 5]
        u, v, w, p, q, r = S[:, 6], S[:, 7], S[:, 8], S[:, 9], S[:, 10], \
            S[:, 11]
        cps, sps = np.cos(psi), np.sin(psi)
        cpc, spc = cps[:, None], sps[:, None]
        X = x0[:, None] + x * cpc
        Y = y0[:, None] + x * spc
        wv = self.wavy[bi]
        allw = bool(wv.all())
        iw = np.arange(m) if allw else np.nonzero(wv)[0]
        bw = bi[iw]
        W = self.W
        Z = kap = lam = CT = ST = None
        if not iw.size:
            e = np.zeros((m, n))
            et = ett = ex = ey = e
        else:
            if self.fk:
                # _waves_b: the running product of the phases, v0's five
                # quantities (sim/as_kernels)
                kap, lam, Z, CT, ST = _ak.phases(
                    W["kx"][bw], W["ky"][bw], W["ww"][bw], W["wphi"][bw],
                    X[iw, 0], Y[iw, 0], psi[iw], t[iw], n, dx, H["dxT"])
                o = _ak.station_waves(Z, W["cc"][bw], W["sc"][bw])
            else:
                # v0's _waves: cos and sin of every phase, five products
                ph = (X[iw][:, :, None] * W["kx"][bw][:, None, :]
                      + Y[iw][:, :, None] * W["ky"][bw][:, None, :]
                      - W["ww"][bw][:, None, :] * t[iw][:, None, None]
                      + W["wphi"][bw][:, None, :])
                c_, s_ = np.cos(ph), np.sin(ph)
                A5 = W["wa"][bw][:, :, :, None]
                o = (np.matmul(c_, A5[:, 0])[..., 0],
                     np.matmul(s_, A5[:, 1])[..., 0],
                     np.matmul(c_, A5[:, 2])[..., 0],
                     np.matmul(s_, A5[:, 3])[..., 0],
                     np.matmul(s_, A5[:, 4])[..., 0])
            if allw:
                e, et, ett, ex, ey = o
            else:
                e = np.zeros((m, n))
                et, ett, ex, ey = e.copy(), e.copy(), e.copy(), e.copy()
                e[iw], et[iw], ett[iw], ex[iw], ey[iw] = o
        s_lon = ex * cpc + ey * spc
        s_tr = -ex * spc + ey * cpc
        Th = H["Th_rest"] + th
        SPx = H["SPx"]
        d = e - ((H["Z_rest"] + z)[:, None] + H["zk"] + SPx * Th[:, None])
        tau = H["kp"] + (SP * Th)[:, None]
        V = et - (w[:, None] + SPx * q[:, None]) + u[:, None] * tau
        # v0's _section
        tb, yc = H["tanb"], H["yc"]
        dp = np.maximum(d, 0.0)
        dch = yc * tb
        below = np.minimum(dp, dch)
        ex_ = np.maximum(dp - dch, 0.0)
        A = below ** 2 / tb + 2.0 * yc * ex_
        cw = 0.5 * np.pi * dp / tb
        c = np.minimum(cw, yc)
        ka = H["k_a"]
        ma = ka * 0.5 * np.pi * rho * c ** 2
        dma = np.where((cw < yc) & (dp > 0),
                       ka * np.pi * rho * c * 0.5 * np.pi / tb, 0.0)
        girth = 2.0 * c / H["cosb"] + 2.0 * ex_
        wet = d > 0.0
        cv = np.abs(u) / np.sqrt(G * H["b"])
        ab = 1.0 - (1.0 - H["a_b"]) * np.clip((cv - 0.5), 0, 1)
        # the transom's depth and flow state (_transom_kin)
        tg = H["tg"]
        eT = np.zeros(m)
        if iw.size:
            XT = x0[iw] + tg.xT * cps[iw]
            YT = y0[iw] + tg.xT * sps[iw]
            phT_ = (XT[:, None] * W["kx"][bw] + YT[:, None] * W["ky"][bw]
                    - W["ww"][bw] * t[iw][:, None] + W["wphi"][bw])
            eT[iw] = np.matmul(np.cos(phT_)[:, None, :], W["wa0"][bw])[
                :, 0, 0]
        z0T = tg.zkT + SP * tg.xT * Th
        dT = eT - (H["Z_rest"] + z) - z0T
        up = np.maximum(u, 0.0)
        cl = tg.cl
        head = up * up / (G * cl.F_T_crit * cl.F_T_crit)
        hw = np.maximum(dT - head, 0.0)
        mom = np.where(V > 0.0, dma * V * V, 0.0)
        flux = self._flux(d, V, u, mom, s_lon - tau, dT)
        cf = 0.5 * rho * H["c_dc"] * 2.0 * c * V * np.abs(V)
        buoy = (ab * rho * G)[:, None] * A
        fk = None
        if self.fk and iw.size:
            fk = _ak.fk_rows(
                self._fkg, d[iw], A[iw], Th[iw], ab[iw], dT[iw], hw[iw],
                head[iw], z0T[iw],
                _ak.FKWaves(a=W["a"][bw], k=W["k"][bw], ku=W["ku"][bw],
                            kinv=W["kinv"][bw], kap=kap, lam=lam, Z=Z, CT=CT,
                            ST=ST, e=e[iw]))
            if allw:
                buoy = buoy + fk.fz
            else:
                buoy[iw] = buoy[iw] + fk.fz
        kpp = H["kpp"]
        dyn = ma * (ett + (2.0 * SP * u * q)[:, None]
                    - (u * u)[:, None] * kpp) + flux + cf
        rest = dyn + buoy
        # everything that is not the strip force (tau_o, component by
        # component, in the plant's order of accumulation)
        num = H["num"]
        nwet = wet.sum(axis=1)
        lw = nwet * dx
        Sw = (girth * wet).sum(axis=1) * dx
        uu = np.maximum(np.abs(u), num.u_floor)
        re = uu * np.maximum(lw, num.l_floor) / num.nu
        # the plant squares a numpy SCALAR here: its ** is C's pow, which is
        # not always x * x to the last bit (an array's ** 2 is np.square);
        # the same scalar operation per boat (found by the batch check)
        lg = np.log10(re) - 2.0
        cfr = 0.075 / np.array([v ** 2 for v in lg]) + 0.0004
        Df = 0.5 * rho * u * np.abs(u) * cfr * Sw
        pos = Sw > 0
        xw = np.where(pos, (x * girth * wet).sum(axis=1) * dx
                      / np.where(pos, Sw, 1.0), 0.0)
        zw = self._zwet(wet, pos & (nwet > 0))
        # _load(-Df, 0, 0, xw, 0, zw), then _load(-Da, 0, 0, 0, 0, z_air)
        mDf = -Df
        Da = 0.5 * num.rho_air * H["cda"] * u * np.abs(u)
        mDa = -Da
        za = num.z_air
        t0_ = (0.0 + mDf) + mDa
        t1_ = np.zeros(m)
        t2_ = np.zeros(m)
        t3_ = (0.0 + (0.0 * 0.0 - zw * 0.0)) + (0.0 * 0.0 - za * 0.0)
        t4_ = (0.0 + (zw * mDf - xw * 0.0)) + (za * mDa - 0.0 * 0.0)
        t5_ = (0.0 + (xw * 0.0 - 0.0 * mDf)) + (0.0 * 0.0 - 0.0 * mDa)
        # S2a: the longitudinal hydrostatic + FK force of the hull and face
        Xh, Mh = self._x_hydro(d, A, s_lon, ab, u, Th, dT, hw, z0T, iw, bi)
        if fk is not None:
            Xh[iw] = Xh[iw] + fk.x
            Mh[iw] = Mh[iw] + fk.m
        t4_ = t4_ + Mh
        Xn = -(ma * V * ((SP * q)[:, None] - u[:, None] * kpp)).sum(
            axis=1) * dx
        Ysec = -rho * G * ab[:, None] * A * s_tr
        if fk is not None:
            Ysec[iw] = fk.fy
            t4_[iw] = t4_[iw] + fk.my_sig
            t5_[iw] = t5_[iw] + fk.n_sig
        t1_ = t1_ + Ysec.sum(axis=1) * dx
        t5_ = t5_ + (x * Ysec).sum(axis=1) * dx
        # v0's lateral branch
        m22 = 0.5 * np.pi * rho * dp ** 2
        vl = v[:, None] + x * r[:, None]
        au = np.abs(u)
        Ylift = -au * m22[:, 0] * vl[:, 0]
        t1_ = t1_ + Ylift
        t5_ = t5_ + (x[0] * Ylift - au * (m22 * vl).sum(axis=1) * dx)
        Ycf = -0.5 * rho * H["cd_lat"] * dp * vl * np.abs(vl)
        t1_ = t1_ + Ycf.sum(axis=1) * dx
        t5_ = t5_ + (x * Ycf).sum(axis=1) * dx
        # roll
        wbar = (A * s_tr).sum(axis=1) / np.maximum(A.sum(axis=1), 1e-12)
        mass = H["m"]
        GM = H["GM"]
        damp = 2.0 * H["roll_zeta"] * H["wn_r"] * H["Ix"] * p
        roll = -mass * G * GM * (phi - wbar) - damp
        if fk is not None:
            # _roll_b: the plant's restoring + the exact wave roll moment
            # beyond its long-wave part
            Wt = mass * G
            rel = phi[iw] - wbar[iw]
            Kr = -Wt * self._gz(rel) if self.gz else -Wt * GM * rel
            roll[iw] = Kr + fk.K - Wt * GM * wbar[iw] - damp[iw]
        t3_ = t3_ + roll
        # the jet
        thr, rud = S[:, 12], S[:, 13]
        fx, fy = self._jet(thr, rud, u, bi)
        t0_ = t0_ + mass * v * r
        t1_ = t1_ - mass * u * r
        # heave and pitch
        A33 = ma.sum(axis=1) * dx
        A35 = (ma * SP * x).sum(axis=1) * dx
        A55 = (ma * x * x).sum(axis=1) * dx
        Fz = rest.sum(axis=1) * dx - mass * G
        My = (SPx * rest).sum(axis=1) * dx
        xn, zn = H["xn"], H["zn"]
        t0_ = t0_ + fx
        t1_ = t1_ + fy
        t2_ = t2_ + 0.0
        t3_ = t3_ + (0.0 * 0.0 - zn * fy)
        t4_ = t4_ + (zn * fx - xn * 0.0)
        t5_ = t5_ + (xn * fy - 0.0 * fx)
        m11, m12, m22_ = mass + A33, A35, H["Iy"] + A55
        r1, r2 = Fz + t2_, My + t4_
        det = m11 * m22_ - m12 * m12
        wd = (r1 * m22_ - m12 * r2) / det
        qd = (m11 * r2 - m12 * r1) / det
        f = rest - ma * (wd[:, None] + SPx * qd[:, None])
        Xp = -((f - buoy) * tau).sum(axis=1) * dx + Xh + Xn
        Mt = H["Mtot"]
        ds = np.zeros((m, 14))
        ds[:, 0] = u * cps - v * sps
        ds[:, 1] = u * sps + v * cps
        ds[:, 2:6] = S[:, 8:12]
        ds[:, 6] = (t0_ + Xp) / Mt[0, 0]
        ds[:, 7] = t1_ / Mt[1, 1]
        ds[:, 8] = wd
        ds[:, 9] = t3_ / Mt[3, 3]
        ds[:, 10] = qd
        ds[:, 11] = t5_ / Mt[5, 5]
        if self.gz:
            # PlaningVesselGZ.deriv's swap of the roll row, where the plant
            # does not compute the whole roll moment itself (P1a; calm
            # water with fk_exact)
            if fk is None:
                rel = phi - wbar
                ds[:, 9] = ds[:, 9] + mass * G * (GM * rel - self._gz(rel)) \
                    / Mt[3, 3]
            elif not allw:
                sw_ = ~wv
                rel = phi[sw_] - wbar[sw_]
                ds[sw_, 9] = ds[sw_, 9] + mass * G * (
                    GM * rel - self._gz(rel)) / Mt[3, 3]
        return ds, dict(d=d, V=V, mom=mom, wd=wd, qd=qd, mom_full=flux,
                        fx=fx, fy=fy)

    def _zwet(self, wet, need):
        """np.mean(zk[wet]) of each boat that needs it (else zk[0]): numpy's
        own mean of the subset (its pairwise sum), boat by boat -- the
        plant's expression. (P1c had a cache keyed by the wet pattern; it
        grew without bound in long rough-sea scans and is gone: the same
        bits.)"""
        zk = self.H["zk"]
        zw = np.full(wet.shape[0], zk[0])
        for j in np.nonzero(need)[0]:
            zw[j] = float(np.mean(zk[wet[j]]))
        return zw

    def _gz(self, rel):
        """PlaningVesselGZ.gz on an array."""
        a = (rel + np.pi) % (2.0 * np.pi) - np.pi
        return a * np.interp(np.abs(a), PHI_TAB, self.H["r_tab"])

    def _flux(self, d, V, u, mom, slope, dT):
        """_StripAS._flux_dDt (and _flux_astern) for arrays."""
        H = self.H
        on, idx, ycap, iunc = self._substrip(d, slope)
        astern = u < 0.0
        if ycap is None and not astern.any():
            return mom.copy()
        flux = mom.copy()
        km = H["k_a"] * 0.5 * np.pi * H["rho"]
        kc = 0.5 * np.pi / H["tanb"]
        dx = H["dx"]
        if ycap is None:
            ycap = iunc = np.zeros(d.shape)
        else:
            Vj = V[:, idx]
            uu = np.where(astern, u, np.maximum(u, 0.0))
            Dma = (Vj * (2.0 * km * kc * kc) * iunc
                   + uu[:, None] * km * ycap) / dx
            fl = np.where((Dma > 0.0) & (Vj > 0.0), Vj * Dma, 0.0)
            flux[:, idx] = np.where(on, fl, flux[:, idx])
        if astern.any():
            ia = np.nonzero(astern)[0]
            tg = H["tg"]
            V0 = V[ia, 0]
            ua = u[ia]
            # station 0's strip: located (on) or v0's centre value
            j0 = np.nonzero(idx == 0)[0]
            if j0.size:
                j0 = int(j0[0])
                hit = on[ia, j0]
                D_loc = (V0 * (2.0 * km * kc * kc) * iunc[ia, j0]
                         + ua * km * ycap[ia, j0]) / dx
            else:
                hit = np.zeros(ia.size, bool)
                D_loc = np.zeros(ia.size)
            d0 = np.maximum(d[ia, 0], 0.0)
            cwa = kc * d0
            D_v0 = np.where((cwa < H["yc"][0]) & (d0 > 0.0),
                            H["k_a"] * np.pi * H["rho"] * cwa * kc * V0, 0.0)
            D0 = np.where(hit, D_loc, D_v0)
            cT = np.minimum(kc * np.maximum(dT[ia], 0.0), tg.ycT)
            D0 = D0 - ua * km * cT * cT / dx
            flux[ia, 0] = np.where((D0 > 0.0) & (V0 > 0.0), V0 * D0, 0.0)
        return flux

    def _x_hydro(self, d, A, s_lon, ab, u, Th, dT, hw, z0T, iw, bi):
        """_StripAS._x_hydro for arrays: (X, M)."""
        H = self.H
        tg = H["tg"]
        rg = H["rho"] * G
        tb = H["tanb"]
        m = d.shape[0]
        sx = np.zeros(m)
        sm = np.zeros(m)
        if iw.size:
            m1 = _ak.vsec_m1(d[iw], H["yc"], tb)
            z0 = H["zk"] + H["SPx"] * Th[iw][:, None]
            Ai = A[iw]
            sl = s_lon[iw]
            sx[iw] = (Ai * sl).sum(axis=1) * H["dx"]
            sm[iw] = (sl * (z0 * Ai + m1)).sum(axis=1) * H["dx"]
        if m <= SCALAR_MAX:
            V4 = np.array([_ak.vsec_qs1(float(dT[j]), tg.ycT, tb)
                           for j in range(m)]).T
            qT, sT, AT, m1T = V4
            Vw = np.array([_ak.vsec_qs1(float(hw[j]), tg.ycT, tb)[:2]
                           for j in range(m)]).T
            qw, sw_ = Vw
        else:
            qT, sT, AT, m1T = _ak.vsec_q(dT, tg.ycT, tb)
            qw, sw_ = _ak.vsec_q(hw, tg.ycT, tb)[:2]
        QT, RT = rg * qT, rg * (z0T * qT + sT)
        Qw, Rw = rg * qw, rg * (z0T * qw + sw_)
        X = -ab * (rg * sx + QT) + ab * Qw
        M = -ab * (rg * sm + RT) + ab * Rw
        face = (u < 0.0) & (AT > 0.0)
        if face.any():
            Dface = np.where(face, 0.5 * H["rho"] * tg.cl.cd_face * AT * u
                             * u, 0.0)
            zA = np.where(face, z0T + m1T / np.where(face, AT, 1.0), 0.0)
            X = X + Dface
            M = M + zA * Dface
        else:
            X = X + 0.0
            M = M + 0.0 * 0.0
        return X, M

    def _lateral_rate(self, S, d):
        """v0's PlaningVessel._lateral_rate for arrays."""
        H = self.H
        u, v, r = S[:, 6], S[:, 7], S[:, 11]
        x, dx, rho = H["x"], H["dx"], H["rho"]
        dp = np.maximum(d, 0.0)
        m22 = 0.5 * np.pi * rho * dp ** 2
        au = np.abs(u)
        k = rho * H["cd_lat"] * dp * np.abs(v[:, None] + x * r[:, None])
        x0, mT = x[0], au * m22[:, 0]
        kx = k * x
        Yv = -mT - k.sum(axis=1) * dx
        Yr = -mT * x0 - kx.sum(axis=1) * dx - H["m"] * u
        Nv = -mT * x0 - au * m22.sum(axis=1) * dx - kx.sum(axis=1) * dx
        Nr = (-mT * x0 * x0 - au * (m22 * x).sum(axis=1) * dx
              - (kx * x).sum(axis=1) * dx)
        Mt = H["Mtot"]
        a, b = Yv / Mt[1, 1], Yr / Mt[1, 1]
        c, e = Nv / Mt[5, 5], Nr / Mt[5, 5]
        half, det = 0.5 * (a + e), a * e - b * c
        disc = half * half - det
        with np.errstate(invalid="ignore"):
            return np.where(disc >= 0.0,
                            np.abs(half) + np.sqrt(np.maximum(disc, 0.0)),
                            np.sqrt(np.where(disc >= 0.0, 0.0, det)))

    # ---------------------------------------------------------------- step
    def _submergence(self, S, t, bi):
        """PlaningVessel.prop_submergence for arrays."""
        H = self.H
        xi, yi, zi = H["intake"]
        c, s_ = np.cos(S[:, 5]), np.sin(S[:, 5])
        X = S[:, 0] + xi * c - yi * s_
        Y = S[:, 1] + xi * s_ + yi * c
        surf = np.zeros(S.shape[0])
        iw = np.nonzero(self.wavy[bi])[0]
        if iw.size:
            W = self.W
            b = bi[iw]
            ph = (X[iw][:, None] * W["kx"][b] + Y[iw][:, None] * W["ky"][b]
                  - W["ww"][b] * t[iw][:, None] + W["wphi"][b])
            surf[iw] = np.matmul(np.cos(ph)[:, None, :], W["wa0"][b])[
                :, 0, 0]
        hull_z = (H["Z_rest"] + S[:, 2] + zi + SP * xi * (H["Th_rest"]
                                                         + S[:, 4])
                  + yi * S[:, 3])
        return surf - hull_z

    @staticmethod
    def _kin(S):
        psi = S[:, 5]
        u, v = S[:, 6], S[:, 7]
        cps, sps = np.cos(psi), np.sin(psi)
        out = np.empty((S.shape[0], 6))
        out[:, 0] = u * cps - v * sps
        out[:, 1] = u * sps + v * cps
        out[:, 2:] = S[:, 8:12]
        return out

    def _sv(self, S, k1, t1, h, bi):
        sE = S + h * k1
        pos = S[:, :6] + 0.5 * h * (k1[:, :6] + self._kin(sE))
        s2 = sE.copy()
        s2[:, :6] = pos
        k2, aux2 = self.deriv(s2, t1, bi)
        s_new = S + 0.5 * h * (k1 + k2)
        s_new[:, :6] = pos
        return s_new, aux2

    def step(self, S, t, thrust_cmd, rudder_cmd, dt=None):
        """One step of every boat: S (N, 14) -> (N, 14); t scalar or (N,);
        the commands scalar or (N,). Each plant's step, its actuators and
        measurements included (module docstring)."""
        S = np.array(S, float)
        N = self.N
        if S.shape != (N, 14):
            raise ValueError(f"states {S.shape}, expected ({N}, 14)")
        PA.check_v0_globals()
        dt = self.dt if dt is None else float(dt)
        t = np.broadcast_to(np.asarray(t, float), (N,))
        thr_c = np.broadcast_to(np.asarray(thrust_cmd, float), (N,))
        rud_c = np.broadcast_to(np.asarray(rudder_cmd, float), (N,))
        out = S.copy()
        live = []
        for i, p in enumerate(self.plants):
            if self.gz and p.capsized:
                # PlaningVesselGZ.step: the run is over
                p.last_cg_acc = p.last_bow_acc = 0.0
                p.last_n_sub = 0
                out[i] = p._capsize_state
            else:
                live.append(i)
        for c0 in range(0, len(live), self.chunk):
            idx = np.array(live[c0:c0 + self.chunk])
            out[idx] = self._step_chunk(S[idx], t[idx], thr_c[idx],
                                        rud_c[idx], dt, idx)
        return out

    def _step_chunk(self, S, t, thr_c, rud_c, dt, bi):
        H = self.H
        P = [self.plants[i] for i in bi]
        sub = self._submergence(S, t, bi)
        for j, p in enumerate(P):
            s_j = float(sub[j])
            if self.gz:
                p.last_submergence = s_j
            p.prop.update_prime(s_j, dt)
        k1, aux = self.deriv(S, t, bi)
        lam = self._lateral_rate(S, aux["d"]) * dt
        nsub = np.ones(bi.size, int)
        stiff = lam > _pv.STIFF_STEP
        if stiff.any():
            nsub[stiff] = np.minimum(np.ceil(lam[stiff] / _pv.STIFF_SUB)
                                     .astype(int), _pv.N_SUB_MAX)
        if (nsub == 1).all():
            new, a2 = self._sv(S, k1, t + dt, dt, bi)
            fxl, fyl = a2["fx"], a2["fy"]
        else:
            new = np.empty_like(S)
            fxl, fyl = np.empty(bi.size), np.empty(bi.size)
            for nn in np.unique(nsub):
                g = np.nonzero(nsub == nn)[0]
                h = dt / nn
                s_new, k = S[g], k1[g]
                for i in range(nn):
                    if i:
                        k, _ = self.deriv(s_new, t[g] + i * h, bi[g])
                    s_new, a2 = self._sv(s_new, k, t[g] + (i + 1) * h, h,
                                         bi[g])
                new[g] = s_new
                fxl[g], fyl[g] = a2["fx"], a2["fy"]
        wd, qd = aux["wd"], aux["qd"]
        bow = wd + SP * H["x"][-1] * qd
        slam = aux["mom"].sum(axis=1) * H["dx"]
        full = aux["mom_full"].sum(axis=1) * H["dx"]
        d_b, V_b = aux["d"][:, -1], aux["V"][:, -1]
        rel_bow = d_b + float(H["keel"][-1])
        for j, p in enumerate(P):
            new[j, 12] = p.prop.advance(S[j, 12], thr_c[j], dt)
            new[j, 13] = p.rudder.step(S[j, 13], rud_c[j], dt)
            p.last_n_sub = int(nsub[j])
            p._jet_f = (fxl[j], fyl[j])
            p.last_jet_force = p._jet_f
            p.last_cg_acc = float(wd[j])
            p.last_bow_acc = float(bow[j])
            sl = float(slam[j])
            p.last_slam_force = sl
            p.peak_slam_force = max(p.peak_slam_force, sl)
            p.last_rel_bow = float(rel_bow[j])
            bow_out = bool(d_b[j] <= 0.0)
            if p._emerged and not bow_out and V_b[j] > p.v_slam:
                p.slam_count += 1
            p._emerged = bow_out
            if self.gz:
                # PlaningVesselGZ.step: finiteness of the whole new state,
                # actuators included (after they are written; P1c review)
                okj = bool(np.all(np.isfinite(new[j])))
                if okj and abs(new[j, 3]) >= p.phi_vanish:
                    p.capsized = True
                    p.capsize_t = float(t[j] + dt)
                    p._capsize_state = new[j].copy()
                if p.record:
                    p._log.append((float(t[j]), np.array(S[j], float),
                                   p.last_cg_acc, p.last_bow_acc,
                                   p.last_rel_bow, p.last_n_sub,
                                   p.last_submergence, float(thr_c[j]),
                                   float(rud_c[j]), okj, p.capsized))
            f_ = float(full[j])
            p.last_slam_force_full = f_
            p.peak_slam_force_full = max(p.peak_slam_force_full, f_)
        return new
