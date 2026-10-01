#!/usr/bin/env python3
"""
The D10a error family 'cat' (learn/meta/PRIOR_D10_DRAFT.md, the force
catalogue; PRIOR_DERIVATION.md D10): one draw = one imagined true boat =
the low-fidelity boat plus
  layer 1  the D8.12 rigid-body family (cat_base.OperatorRBL1: the
           stability rule replaced by divergence-only rejection) with the
           overlap switches of draft section 1 (T8 off, T4 speed
           polynomial constant / tanh stiffness off, T5 eta_T / eta_S off,
           T7 quadratic off, T3 kappa off per W7 part, T3 surge gated below
           16 kn when W11 is on, T2c's rigid m v r off);
  layer 2  the catalogue items learn/meta/cat_*.py (W1-W12, H1-H3, A1-A3,
           B1-B3, P1-P14, E1-E4) switched on with their probabilities
           (15 % sparse episodes with 1-3 items);
  layer 3  the D9 general residual (cat_base.OperatorGenL3) at the declared
           shares (small 0.75 / fail 0.20 / none 0.05) plus the D9.1 rigid-
           body part: its M_t maps every catalogue force, its exact
           Coriolis difference is added once;
all evaluated at every 0.04 s substep in closed loop (cat_base.CatDraw
.substep, subclassed here as CatDrawD10). This module adds what a pipeline
needs, with the interface of operators_gen:
  - OperatorCat: a draw (seed -> CatDrawD10) with the ACCEPTANCE TEST of
    operators_gen (a perturbed calm-water row and a standard-sea row, 60 s,
    the data autopilot, ideal actuators, no noise; divergence ONLY: non-
    finite, u > 1.5 u_max, |theta| > 45 deg). A rejected draw is replaced
    by the next attempt stream of the same seed (CatDraw.next_attempt),
    every rejection logged with the items that were on (per-item rejection
    rates); after N_REJECT rejections the draw keeps layer 1 and the
    structural contact gate W4 only (forces_off). new_state / step /
    cat_accel / snapshot / slow_state / set_slow / style as the other
    families;
  - simulate_cat: the batched closed-loop simulator (the simulate_gen
    pattern) with the catalogue's IMPULSE channel: velocity jumps (W3 slams,
    E3 debris) are added after the clipped kick, not clipped per substep;
    per control step their sum is bounded by the clip limit x 0.24 s (the
    control-step average, draft section 6 item 3; a causal budget that is
    logged when it binds) and the control-step average of push + impulse
    is checked against CLIP x A_REF (recorded, never a rejection). The
    impulse enters the vertical acceleration of the safety quantities
    APK / AMIN;
  - CatPlant (an RBPlant that also keeps each substep's start time and
    commands) + CatInjector: the Mission path, the same arithmetic as
    simulate_cat (studies/test_operators_cat.py checks equality);
  - the extra observed signals of draft section 11 (OBS_EXTRA: roll, roll
    rate, rpm, relative wind speed / angle, speed through water u / v,
    GNSS quality), computed every substep from their hidden states and
    returned per control step (value at the step's first substep) for
    STORAGE only: D10a keeps the meta5 network inputs, the observation
    model O1-O5 is OFF.
The hull geometry (keel at the 5 stations, the jet intake, v_slam) is the
task plant's (cat_geometry, sim.lofi.plant_for on the task hull), the
same in every process.
"""
import copy

import numpy as np

from learn.meta import cat_base as CB
from learn.meta import operators_gen as G
from learn.meta import operators_rb as R
from learn.meta.operators import A_REF, CLIP, clip_push

CHANNELS = R.CHANNELS
RED_VEL = R.RED_VEL
VEL_IDX = R.VEL_IDX
KN = R.KN
OBS_EXTRA = CB.OBS_EXTRA
N_REJECT = 30
TH_MAX = G.TH_MAX
set_slow = R.set_slow              # family-neutral slow-state restore
LIM = CLIP * A_REF

_GEOM = {}


def cat_geometry():
    """The task hull's keel height at the 5 stations, jet intake point and
    slam threshold v_slam (sim.lofi.plant_for on learn.repro.task.ctx()'s
    hull; computed once per process, deterministic). The intake keeps the
    plant's x, y but sits at the hull's keel height at x (the opening is
    flush with the bottom): the plant's intake point is the propulsor
    centre, 0.19 m below the keel there, which read as submerged while the
    bottom at the intake was already dry (review 2026-09-30; D10.1)."""
    if "g" not in _GEOM:
        from learn.repro.task import ctx
        from sim import lofi
        c = ctx()
        pl = lofi.plant_for(c["db"], None, c["h"], dt=0.04)
        p = pl.p
        x_st = np.linspace(p["x_stern"], p["x_bow"], 5)
        sx = np.asarray(pl.sec.x, float)
        o = np.argsort(sx)
        keel = np.interp(x_st, sx[o], np.asarray(pl.sec.keel, float)[o])
        intake = None
        if getattr(pl, "intakes", None):
            intake = np.array(pl.intakes[0], float)
            intake[2] = float(np.interp(intake[0], sx[o],
                                        np.asarray(pl.sec.keel, float)[o]))
        _GEOM["g"] = dict(keel=keel, intake=intake,
                          v_slam=float(getattr(pl, "v_slam", np.nan)))
    return dict(_GEOM["g"])


class _KeepProp(dict):
    """A copy of the shared-quantity dict that keeps the propulsion chain's
    dict when CatDraw._prop deletes it (diagnostics)."""

    def __delitem__(self, key):
        if key == "prop":
            dict.__setitem__(self, "_prop_last", self["prop"])
        dict.__delitem__(self, key)


class CatDrawD10(CB.CatDraw):
    """cat_base.CatDraw with: the layer-1 state initialised from the first
    inputs (init_s, as the other families), the replacement draw of the
    same class, and the propulsion chain's summary kept in
    q['prop_diag'] (suction p = P1 x P13, P1's own suction p1, P13's own
    air factor p13 and P1's latch lost (NaN / False when the item is off),
    fault factor rho, thrust before / after the cavitation limit, nozzle
    before / after P7 / P11, command changed by P12) for previews."""

    def new_state(self, B=1, hid_seed=0, rng=None, init_s=None):
        st = dict(k=0, items={}, emerged=np.zeros(B, bool), act=None)
        for c, prm in self.on.items():
            r = self._rng(CB.SID_ITEM_STATE + CB.CODES.index(c), hid_seed)
            st["items"][c] = CB.REGISTRY[c].init_state(prm, B, rng=r)
        st["l1"] = None if self.op_rb is None else self.op_rb.new_state(
            B, rng=rng, init_s=init_s)
        st["l3"] = self.op_gen.new_state(B, hid_seed=hid_seed,
                                         forces=bool(self.op_gen.forces))
        return st

    def next_attempt(self, reason=""):
        nxt = type(self)(self.seed, attempt=self.attempt + 1, **self._kw)
        nxt.reject_log = self.reject_log + [dict(
            attempt=self.attempt, reason=reason, tier=self.tier,
            sparse=self.sparse, items=sorted(self.on))]
        return nxt

    def _prop(self, st, q, dt):
        qk = _KeepProp(q)
        out = super()._prop(st, qk, dt)
        pc = dict.pop(qk, "_prop_last", None)
        q.update(qk)
        if pc is not None:
            B = q["B"]
            f = lambda v: np.broadcast_to(np.asarray(      # noqa: E731
                v if v is not None else np.nan, float), (B,)).copy()
            q["prop_diag"] = dict(
                p=f(pc["p"]), p1=f(pc.get("p1")), p13=f(pc.get("p13")),
                lost=np.broadcast_to(np.asarray(pc.get("lost", False),
                                                bool), (B,)).copy(),
                rho=f(pc["rho"]), T_pre=f(pc["T_pre"]),
                T_eff=f(pc["T_eff"]), noz_act=f(pc["noz_act"]),
                noz_eff=f(pc.get("noz_eff", pc["noz_act"])),
                cmd_changed=bool(pc["cmd_changed"]))
        return out


# ------------------------------------------------------------ the operator
class OperatorCat:
    """One draw of the D10a family (module docstring).

    OperatorCat(seed, lib, dt, L, relay, null, p, act, accept, geom, **kw):
    lib the M15 library (layer 1's T7 loads and old T9 noise; None = no
    layer 1); act the episode's M10 actuator dict (lofi.ACT_KEYS scalars;
    used only when a P12 rule changes the command: the true boat's
    actuator copy); geom None = cat_geometry(), False = cat_base defaults,
    or a dict(keel, intake, v_slam); kw passes to CatDraw (items,
    force_items, sparse, tier, obs_on, has_waves, layer1). relay / null are
    accepted for the common signature and not used (T8 is off, so the old
    rule and its relays are 0). accept=True runs the acceptance test now
    (accept_cats for many draws at once)."""

    def __init__(self, seed, lib=None, dt=0.24, L=5.4, relay=False,
                 null=None, p=None, act=None, accept=True, geom=None, **kw):
        self.seed, self.dt = int(seed), float(dt)
        self.relay = bool(relay)
        self.p = dict(R.default_params() if p is None else p)
        g = cat_geometry() if geom is None else (geom or {})
        gk = {k: g[k] for k in ("keel", "intake", "v_slam") if k in g}
        self._kw = dict(lib=lib, p=self.p, dtc=self.dt, act=act, **gk, **kw)
        self.cat = CatDrawD10(self.seed, attempt=0, **self._kw)
        self.n_reject, self.forces_off, self.accepted = 0, False, False
        self.test_clip = np.nan
        if accept:
            accept_cats([self])

    # ---------------------------------------------------------- access
    @property
    def ctx(self):
        return self.cat.ctx

    @property
    def on(self):
        return self.cat.on

    @property
    def reject_log(self):
        return self.cat.reject_log

    def set_act(self, act):
        """The episode's M10 actuator (the true-actuator copy P12 needs)."""
        self.cat.act = None if act is None else dict(act)
        self.cat._kw["act"] = self.cat.act
        self._kw["act"] = self.cat.act

    def forces_off_now(self):
        """After N_REJECT rejections: layer 1 and the structural contact
        gate W4 only (no other item, layer 3 'none'); flagged."""
        kw = dict(self._kw, items=("W4",), tier="none", sparse=False,
                  force_items=None)
        log = self.cat.reject_log
        self.cat = CatDrawD10(self.seed, attempt=N_REJECT, **kw)
        self.cat.reject_log = log
        self.forces_off = True

    # ----------------------------------------------------------- state
    def new_state(self, B=1, rng=None, init_s=None, hid_seed=0):
        """Fresh state for B rows (items' own process streams, layer 1's
        with rng pre-rolling its old noise, layer 3's, the slam detector,
        the true-actuator copy)."""
        return self.cat.new_state(B, hid_seed=hid_seed, rng=rng,
                                  init_s=init_s)

    @staticmethod
    def snapshot(st):
        """A deep copy of the full state (items' hidden states and their
        generators, layer 1 incl. the old noise part, layer 3's filters,
        the slam detector, the true-actuator copy)."""
        return copy.deepcopy(st)

    def slow_state(self, st):
        """Layer 1's old noise part's slow state (operators.Operator flat),
        or the clock alone without layer 1."""
        if self.cat.op_rb is None:
            return dict(k=st["k"])
        return self.cat.op_rb.slow_state(st["l1"])

    def set_slow(self, st, slow):
        if self.cat.op_rb is None:
            st["k"] = slow["k"]
            return
        self.cat.op_rb.set_slow(st["l1"], slow)
        st["k"] = st["l1"]["k"]

    def bind_sea(self, st, rowseas):
        """A1's wind direction relative to the waves needs the rows' mean
        wave direction before the first substep (cat_air_env
        .bind_wave_dir); a no-op when A1 is off or already bound."""
        s = st["items"].get("A1")
        if s is not None and s.get("wave_dir") is None \
                and s.get("psi0") is None:
            from learn.meta.cat_air_env import bind_wave_dir
            bind_wave_dir(s, rowseas)

    def run(self, *args, **kwargs):
        raise NotImplementedError(
            "OperatorCat.run: the catalogue is state feedback at every "
            "substep; use simulate_cat (closed loop) or the Mission path "
            "(CatPlant + CatInjector)")

    def run_residual(self, S_raw, slow=None, noise=False, rng=None, hold=0):
        """The control-step part alone (the old noise; the rule is 0) on
        recorded inputs (B, T, 28): rule (B, T, 5) [, noise]."""
        S_raw = np.asarray(S_raw, float)
        B, T, _ = S_raw.shape
        st = self.new_state(B, init_s=S_raw[:, 0])
        if slow is not None:
            self.set_slow(st, slow)
        Rr, N = np.zeros((B, T, 5)), np.zeros((B, T, 5))
        for k in range(T):
            Rr[:, k], N[:, k] = self.step(st, S_raw[:, k], noise=noise,
                                          rng=rng, freeze=slow is not None)
        return (Rr, N) if noise else Rr

    # ------------------------------------------------------ the parts
    def step(self, st, s_raw, noise=True, rng=None, freeze=False,
             hold_relays=False):
        """The control-step part (held over the substeps): layer 1's old
        noise part (T9 x w_noise); the rule is 0 (T8 off)."""
        return self.cat.step(st, s_raw, noise=noise, rng=rng, freeze=freeze)

    def cat_accel(self, st, sr, thr, noz, nu0, sea, dt, cmd=None, t=None,
                  freeze=False, parts=False):
        """THE substep (CatDraw.substep): sr (B, 10) the start state, thr /
        noz the applied actuators, nu0 (B, 5) the low-fidelity velocity
        increment / dt over the substep, sea a CatSea.sample dict at the
        substep start, cmd (B, 2) the physical commands. Returns dict(acc,
        imp, obs, slam, q[, PARTS])."""
        return self.cat.substep(st, sr, thr, noz, sea, nu0=nu0, cmd=cmd,
                                dt=dt, t=t, freeze=freeze, parts=parts)

    # ----------------------------------------------------------- style
    @property
    def style(self):
        s = self.cat.style()
        s["item_params"] = {c: {k: v for k, v in prm.items()
                                if isinstance(v, (float, int, bool))}
                            for c, prm in s["item_params"].items()}
        s.update(op_family="cat", prior_family="cat",
                 n_reject=int(self.n_reject), forces_off=bool(
                     self.forces_off), test_clip=float(self.test_clip),
                 reject_reasons=[r["reason"] for r in self.reject_log],
                 reject_items=[list(r["items"]) for r in self.reject_log],
                 # the evaluation's keys: never null (the Coriolis
                 # difference and the gate are always on); the structural
                 # contact gate W4 reads the keel immersion, i.e. the waves
                 # met in the same substep
                 null=False, no_forces=bool(self.forces_off),
                 reads_waves=True, wave_same_step=True)
        return s


# ------------------------------------------------------- acceptance test
def _accept_world(env, n):
    """operators_gen's test world (start states; calm rows 0, 2, ... with
    their start perturbation, standard-sea rows 1, 3, ...) as one RowSeas
    whose calm rows have zero amplitude."""
    xs, _ = G._test_world(env, n)
    sea = R.sea_state(G.TEST_SEA, G.TEST_SEA_SEED)
    seas = R.RowSeas([sea] * (2 * n), env["x_st"], env["y_off"])
    seas.a[0::2] = 0.0
    seas.aw[0::2] = 0.0
    return xs, seas


def acceptance_test(ops, env=None):
    """operators_gen.acceptance_test for OperatorCat draws: [(ok, reason)];
    also sets op.test_clip (the rule-clip share of substeps from TEST_SKIP
    on, both rows; recorded, never a reason to reject)."""
    env = G._test_env() if env is None else env
    n = len(ops)
    xs, seas = _accept_world(env, n)
    groups = [dict(op=o, st=o.new_state(2, hid_seed=G.TEST_HID),
                   rows=np.array([2 * i, 2 * i + 1]), rng=None)
              for i, o in enumerate(ops)]
    out = simulate_cat(env, xs, G._PairPilot(env, n), np.zeros(2 * n), seas,
                       groups, G.TEST_STEPS, noise=False, obs=False)
    res = []
    n_sub = 2 * (G.TEST_STEPS - G.TEST_SKIP) * env["sub"]
    for i, o in enumerate(ops):
        c, s = 2 * i, 2 * i + 1
        o.test_clip = float(out["clip_rule"][[c, s], G.TEST_SKIP:].sum()
                            / n_sub)
        res.append((False, "diverged") if (out["div"][[c, s]] >= 0).any()
                   else (True, ""))
    return res


def accept_cats(ops, env=None, max_reject=N_REJECT):
    """Run the acceptance test on every not-yet-accepted draw at once and
    replace the rejected ones by their next attempt until all pass (a
    draw's outcome depends only on its own seed, never on the batch)."""
    pending = [o for o in ops if not o.accepted]
    while pending:
        res = acceptance_test(pending, env)
        nxt = []
        for o, (ok, why) in zip(pending, res):
            if ok:
                o.accepted = True
                continue
            o.n_reject += 1
            if o.n_reject >= max_reject:
                o.cat.reject_log.append(dict(
                    attempt=o.cat.attempt, reason=why, tier=o.cat.tier,
                    sparse=o.cat.sparse, items=sorted(o.cat.on)))
                o.forces_off_now()
                o.accepted = True
            else:
                o.cat = o.cat.next_attempt(why)
                nxt.append(o)
        pending = nxt
    return ops


def build_ops(seeds, lib=None, env=None, **kw):
    """OperatorCat draws for many seeds, the acceptance test batched."""
    ops = [OperatorCat(s, lib, accept=False, **kw) for s in seeds]
    accept_cats(ops, env)
    return ops


# ---------------------------------------------------------- the sea rows
class _SeaRows:
    """The components of some rows of an operators_rb.RowSeas (what
    cat_base.CatSea and bind_wave_dir read)."""

    def __init__(self, rs, rows):
        rows = np.asarray(rows)
        for k in ("a", "k", "c", "s", "w", "phi", "aw"):
            setattr(self, k, np.asarray(getattr(rs, k))[rows])


# --------------------------------------------------------- batched world
def simulate_cat(env, xs14, U, t0, seas, groups, n_steps, noise=True,
                 freeze=False, waves="mid", act=None, parts=False,
                 substeps=False, fb=None, hook=None, obs=True):
    """Closed-loop batched rollout with the D10a family (the simulate_gen
    pattern). Per substep: actuators, the surface at the stations at the
    substep start, the low-fidelity step, nu0 = its velocity increment /
    dt, the catalogue substep from the START state (its own CatSea sample
    of the rows' seas at the start time, the physical commands), clip_push
    with the held control-step part, the velocity kick, then the impulse
    channel (bounded per control step by LIM x 0.24 s; clip_imp counts the
    steps where that bound acts). Rows in no group get no push.

    xs14 (B, 14), U (B, n, 2) or callable U(step, xs14_now), t0 (B,), seas
    an operators_rb.RowSeas, groups [dict(op, st, rows, rng)], act None
    (ideal) or stacked lofi.draw_act rows (the M10 chain; a one-row
    group's operator gets its row's actuator for P12). hook(k, kk, j, g,
    out) is called after every group's substep (previews). Divergence as
    simulate_gen (frozen, flagged, never redrawn).

    Returns dict(XS, U, P (B, n, 5) mean clipped push, IMP (B, n, 5) the
    impulse channel's control-step mean acceleration, clip (B, n, 5),
    clip_rule (B, n), clip_imp (B, n) impulse bound acted, clip_avg (B, n,
    5) |P + IMP| > LIM (the control-step-average check), hpk / hpk_clip
    (B, n) substeps with |heave rule| > LIM / 2 and of those clipped, RULE,
    NOISE, APK / AMIN / HMIN (impulses included in the vertical
    acceleration), E15, SLAM (B, n) low-fidelity-rule slams (bow station),
    OBS (B, n, 8) the extra observed signals at each step's first substep,
    div); parts=True adds PARTS [per group {part: (B_g, n, 5) control-step
    mean; ':imp' parts as accelerations}]; substeps=True adds SUB, PS, AZ,
    HS as simulate_gen."""
    red, dt, sub = env["red"], env["dt"], env["sub"]
    dtc = dt * sub
    cap = LIM * dtc
    p = red.p
    fb = G.freeboard(p) if fb is None else float(fb)
    x_b, s_p = float(env["x_st"][-1]), float(p["sign_pitch"])
    u_max = float(p["u_max"])
    B = len(xs14)
    rv = list(RED_VEL)
    sr = R.to_reduced(xs14)
    thr = np.asarray(xs14, float)[:, 12].copy()
    noz = np.asarray(xs14, float)[:, 13].copy()
    t0 = np.broadcast_to(np.asarray(t0, float), (B,)).copy()
    XS = np.zeros((B, n_steps + 1, 14))
    UU = np.zeros((B, n_steps, 2))
    P, CL = np.zeros((B, n_steps, 5)), np.zeros((B, n_steps, 5))
    IMP, CLA = np.zeros((B, n_steps, 5)), np.zeros((B, n_steps, 5), bool)
    CLR, CLI = np.zeros((B, n_steps)), np.zeros((B, n_steps))
    HPK, HPC = np.zeros((B, n_steps)), np.zeros((B, n_steps))
    RU, NZ = np.zeros((B, n_steps, 5)), np.zeros((B, n_steps, 5))
    E15 = np.zeros((B, n_steps, 15))
    SLAM = np.zeros((B, n_steps))
    OBS = np.zeros((B, n_steps, len(OBS_EXTRA)))
    AZ = np.zeros((B, n_steps * sub))
    HS = np.zeros((B, n_steps * sub + 1))
    div = np.full(B, -1)
    alive = np.ones(B, bool)
    PT = [{} for _ in groups] if parts else None
    if substeps:
        SUB = np.zeros((B, n_steps, sub, 10))
        PS = np.zeros((B, n_steps, sub, 5))
    XS[:, 0] = R.to14(sr, thr, noz)
    for g in groups:
        rows = np.asarray(g["rows"])
        g["_cs"] = CB.CatSea(_SeaRows(seas, rows), g["op"].ctx)
        if act is not None and len(rows) == 1 and hasattr(g["op"],
                                                          "set_act"):
            from sim import lofi
            g["op"].set_act({k_: float(np.broadcast_to(np.asarray(
                act[k_], float), (B,))[rows[0]]) for k_ in lofi.ACT_KEYS})
        g["op"].bind_sea(g["st"], g["_cs"].rs)
    if act is not None:
        from sim import lofi
        A = {k_: np.broadcast_to(np.asarray(act[k_], float), (B,)).copy()
             for k_ in lofi.ACT_KEYS}
        if not (A["act_family"] > 0.5).all():
            raise ValueError("simulate_cat: act rows must be M10 draws")
        q = lofi.act_arrays(A, env["t_max"], env["rud_max"])
        dly = q["delay"]
        pos0 = np.stack([thr, noz], 1)
        vel = np.zeros((B, 2))
        CMD = np.zeros((B, n_steps, 2))
        rows_, cols_ = np.arange(B)[:, None], np.arange(2)[None, :]
    for k in range(n_steps):
        t = t0 + k * dtc
        xs_now = R.to14(sr, thr, noz)
        Uk = np.asarray(U(k, xs_now) if callable(U) else U[:, k], float)
        UU[:, k] = Uk
        if waves == "mid":
            w15 = R.mid_waves(seas, sr, t, 0.5 * dtc)
        else:
            w15 = seas.stations(sr[:, 0], sr[:, 1], sr[:, 7], t)[0].reshape(
                B, 15)
        s28 = R.s28_of(env, xs_now, w15, Uk)
        held_r, held_n = np.zeros((B, 5)), np.zeros((B, 5))
        for g in groups:
            rows = g["rows"]
            rr, nn = g["op"].step(g["st"], s28[rows], noise=noise,
                                  rng=g.get("rng"), freeze=freeze)
            if not noise:
                nn = np.zeros_like(rr)
            held_r[rows], held_n[rows] = clip_push(rr, nn)
        RU[:, k], NZ[:, k] = held_r, held_n
        cmd_t, cmd_r = Uk[:, 0] * env["t_max"], Uk[:, 1] * env["rud_max"]
        if act is not None:
            CMD[:, k, 0], CMD[:, k, 1] = cmd_t, cmd_r
        cmd2 = np.stack([cmd_t, cmd_r], 1)
        sr_k, thr_k, noz_k = sr.copy(), thr.copy(), noz.copy()
        isum = np.zeros((B, 5))
        for kk in range(sub):
            tt = t + kk * dt
            if act is None:
                thr_app, noz_app = cmd_t, cmd_r
            else:
                gi = k * sub + kk - dly
                ci = np.maximum(gi, 0) // sub
                c_d = np.where(gi >= 0, CMD[rows_, ci, cols_], pos0)
                pos, vel, app = lofi.act_step(q, np.stack([thr, noz], 1),
                                              vel, c_d, dt)
                thr_app, noz_app = app[:, 0], app[:, 1]
                thr, noz = pos[:, 0], pos[:, 1]
            eta, _ = seas.stations(sr[:, 0], sr[:, 1], sr[:, 7], tt)
            if kk == 0:
                E15[:, k] = eta.reshape(B, 15)
            HS[:, k * sub + kk] = fb + sr[:, 3] + s_p * x_b * sr[:, 5] \
                - eta[:, 4, :].max(1)
            thr_app = np.broadcast_to(np.asarray(thr_app, float), (B,))
            noz_app = np.broadcast_to(np.asarray(noz_app, float), (B,))
            ns = red.step(sr, thr_app, noz_app, eta, env["x_st"], dt)[0]
            nu0 = (ns[:, rv] - sr[:, rv]) / dt
            push, impv = np.zeros((B, 5)), np.zeros((B, 5))
            for j, g in enumerate(groups):
                rows = g["rows"]
                sea_s = g["_cs"].sample(sr[rows], tt[rows])
                o = g["op"].cat_accel(g["st"], sr[rows], thr_app[rows],
                                      noz_app[rows], nu0[rows], sea_s, dt,
                                      cmd=cmd2[rows], t=tt[rows],
                                      freeze=freeze, parts=parts)
                a = o["acc"]
                if parts:
                    for key, v in o["PARTS"].items():
                        acc_ = PT[j].setdefault(key, np.zeros(
                            (len(rows), n_steps, 5)))
                        acc_[:, k] += (v / dt if key.endswith(":imp")
                                       else v) / sub
                rc, nc = clip_push(a + held_r[rows], held_n[rows])
                push[rows] = rc + nc
                CL[rows, k] += (np.abs(a + held_r[rows] + held_n[rows])
                                > LIM)
                CLR[rows, k] += (np.abs(a + held_r[rows]) > LIM).any(-1)
                hv = np.abs(a[:, 3] + held_r[rows, 3])
                HPK[rows, k] += hv > 0.5 * LIM[3]
                HPC[rows, k] += hv > LIM[3]
                im = np.asarray(o["imp"], float)
                capd = np.clip(isum[rows] + im, -cap, cap)
                ima = capd - isum[rows]
                CLI[rows, k] += (np.abs(ima - im) > 1e-12).any(-1)
                isum[rows] = capd
                impv[rows] = ima
                SLAM[rows, k] += np.asarray(o["slam"], float)
                if obs and kk == 0:
                    OBS[rows, k] = np.stack([np.broadcast_to(np.asarray(
                        o["obs"][nm], float), (len(rows),))
                        for nm in OBS_EXTRA], 1)
                if hook is not None:
                    hook(k, kk, j, g, o)
            push[~alive] = 0.0
            impv[~alive] = 0.0
            zd_old = sr[:, 4].copy()
            sr = ns
            sr[:, rv] += push * dt
            sr[:, rv] += impv
            AZ[:, k * sub + kk] = (sr[:, 4] - zd_old) / dt
            P[:, k] += push / sub
            if substeps:
                SUB[:, k, kk], PS[:, k, kk] = sr, push
        IMP[:, k] = isum / dtc
        CLA[:, k] = np.abs(P[:, k] + IMP[:, k]) > LIM
        if act is None:
            thr, noz = np.array(cmd_t, float), np.array(cmd_r, float)
        with np.errstate(invalid="ignore"):
            bad = alive & (~np.isfinite(sr).all(1)
                           | (sr[:, 2] > 1.5 * u_max)
                           | (np.abs(sr[:, 5]) > TH_MAX))
        div[bad] = k
        alive &= ~bad
        dead = ~alive
        if dead.any():
            sr[dead], thr[dead], noz[dead] = sr_k[dead], thr_k[dead], \
                noz_k[dead]
            AZ[dead, k * sub:(k + 1) * sub] = np.nan
        XS[:, k + 1] = R.to14(sr, thr, noz)
    eta, _ = seas.stations(sr[:, 0], sr[:, 1], sr[:, 7], t0 + n_steps * dtc)
    HS[:, -1] = fb + sr[:, 3] + s_p * x_b * sr[:, 5] - eta[:, 4, :].max(1)
    for b in np.where(div >= 0)[0]:
        HS[b, div[b] * sub:] = np.nan
    APK, AMIN, HMIN = G.safety_per_step(AZ, HS, sub)
    for g in groups:
        g.pop("_cs", None)
    out = dict(XS=XS, U=UU, P=P, IMP=IMP, clip=CL, clip_rule=CLR,
               clip_imp=CLI, clip_avg=CLA, hpk=HPK, hpk_clip=HPC, RULE=RU,
               NOISE=NZ, APK=APK, AMIN=AMIN, HMIN=HMIN, E15=E15, SLAM=SLAM,
               OBS=OBS, div=div)
    if parts:
        out["PARTS"] = PT
    if substeps:
        out.update(SUB=SUB, PS=PS, AZ=AZ, HS=HS)
    return out


# ------------------------------------------------------------------ plant
_CATPLANT = []


def _catplant_class():
    if not R._RBPLANT:
        R._RBPLANT.append(R._rbplant_class())
    Base = R._RBPLANT[0]

    class CatPlant(Base):
        """operators_rb.RBPlant that also keeps each substep's start time
        and commands (what the catalogue substep needs besides the start
        state, applied actuators and elevations). Bit-identical steps."""

        def __init__(self, params, geom, sea, dt):
            super().__init__(params, geom, sea, dt)
            self.last_t, self.last_cmd = None, None

        def step(self, s, t, thrust_cmd, rudder_cmd, dt=None):
            self.last_t = float(t)
            self.last_cmd = (float(thrust_cmd), float(rudder_cmd))
            return super().step(s, t, thrust_cmd, rudder_cmd, dt)

        def with_sea(self, sea, params=None, **_ignored):
            p = dict(self.p)
            if params:
                p.update(params)
            return CatPlant(p, self.geom, sea, self.dt)

    return CatPlant


def CatPlant(params, geom, sea, dt):   # noqa: N802  (a class factory)
    if not _CATPLANT:
        _CATPLANT.append(_catplant_class())
    return _CATPLANT[0](params, geom, sea, dt)


def cat_plant(plant):
    """A CatPlant copy of a sim.lofi.ReducedPlant (same parameters,
    geometry, sea and step; an M10 plant's actuator state starts empty)."""
    return CatPlant(plant.p, plant.geom, plant.sea, plant.dt)


class CatInjector:
    """Mission's residual hook for the D10a family (the GenInjector
    interface). rule / noise are the held control-step part (setting rule
    starts a new control step: the impulse budget restarts); at every
    substep, after the plant's own step, it reads the start state, applied
    actuators and elevations off the CatPlant, the start time and commands,
    samples the catalogue's sea, runs the substep, clips as clip_push does,
    kicks the velocities, then adds the impulse channel (bounded per
    control step as simulate_cat). Logs per substep the bow height at the
    substep start (h) and the vertical acceleration incl. the impulse
    (az); keeps the last push, the last impulse as an acceleration, the
    extra observed signals and the slam flag."""

    def __init__(self, op, st, freeze=False, sub=None):
        # the impulse budget per control step is LIM x the operator's
        # control step op.dt (= simulate_cat's LIM x dtc); sub, when given,
        # must agree with it (checked at the first substep)
        self.op, self.st, self.freeze = op, st, bool(freeze)
        self.sub = None if sub is None else int(sub)
        self._rule, self.noise = np.zeros(5), np.zeros(5)
        self.last, self.last_imp = np.zeros(5), np.zeros(5)
        self._acc, self._n = np.zeros(5), 0
        self._isum = np.zeros(5)
        self.az, self.h = [], []
        self.obs, self.slam = None, False
        self._cs = None

    @property
    def rule(self):
        return self._rule

    @rule.setter
    def rule(self, value):
        self._rule = np.asarray(value, float).copy()
        self._isum = np.zeros(5)

    @property
    def r(self):
        return self._acc / self._n if self._n else self._rule + self.noise

    @r.setter
    def r(self, value):
        self.rule = value
        self.noise = np.zeros(5)
        self._acc, self._n = np.zeros(5), 0

    def __call__(self, s, dt, plant=None):
        sr, thr, noz, eta = plant.model.last
        sr = np.atleast_2d(sr)
        if self._cs is None:
            rs = R.RowSeas([plant.sea], plant.x_st, plant.y_off)
            self._cs = CB.CatSea(rs, self.op.ctx)
            self.op.bind_sea(self.st, rs)
        post = R.to_reduced(np.asarray(s, float)[None])
        nu0 = (post[:, list(RED_VEL)] - sr[:, list(RED_VEL)]) / dt
        t = plant.last_t
        sea = self._cs.sample(sr, np.array([t]))
        o = self.op.cat_accel(self.st, sr, np.array([thr]), np.array([noz]),
                              nu0, sea, dt, cmd=np.array([plant.last_cmd]),
                              t=np.array([t]), freeze=self.freeze)
        rc, nc = clip_push(o["acc"][0] + self._rule, self.noise)
        push = rc + nc
        if self.sub is not None:
            if abs(dt * self.sub - self.op.dt) > 1e-9 * self.op.dt:
                raise ValueError(f"CatInjector: dt x sub = {dt * self.sub} "
                                 f"!= the operator's control step "
                                 f"{self.op.dt}")
            self.sub = None
        cap = LIM * self.op.dt
        capd = np.clip(self._isum + np.asarray(o["imp"][0], float), -cap,
                       cap)
        imp = capd - self._isum
        self._isum = capd
        s = s.copy()
        for c, i in enumerate(VEL_IDX):
            s[i] += push[c] * dt
        for c, i in enumerate(VEL_IDX):
            s[i] += imp[c]
        plant.last_cg_acc = float(plant.last_cg_acc + push[3] + imp[3] / dt)
        self.h.append(float(CB.bow_height(self.op.ctx, sr, eta)[0]))
        self.az.append(plant.last_cg_acc)
        self.last, self.last_imp = push, imp / dt
        self.obs = {nm: float(np.asarray(o["obs"][nm], float).reshape(-1)[0])
                    for nm in OBS_EXTRA}
        self.slam = bool(np.asarray(o["slam"]).reshape(-1)[0])
        self._acc, self._n = self._acc + push, self._n + 1
        return s
