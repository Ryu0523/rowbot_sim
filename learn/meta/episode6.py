#!/usr/bin/env python3
"""
Episodes of the rigid-body error families (M16 pipeline, cache meta6;
learn/meta/PRIOR_DERIVATION.md D8.12 and D9): everything the meta6 data
chunks are made from, so studies/meta_step6.code6() (the stamp of every
chunk and pack) hashes this module, the family modules and
meta_step2.SOURCES. The branches (rollouts from snapshots) live in
learn/meta/data6.py, outside that stamp.

family (the job key 'family'):
- 'gen' the general family: a rigid body acted on by ARBITRARY causal
        forces (operators_gen.OperatorGen, D9.8). The main line.
- 'rb'  the D8 rigid-body family with hydrodynamic-form terms
        (operators_rb.OperatorRB, D8.12). The comparison arm.

What an episode is. episode5.episode (op_family 'm15') with the family's
operator in the loop instead of the M15 operator, and nothing else changed:
the same random stream in the same order (sea, the M10 actuator draw from
its own stream [seed, 11], scenario, dithers), so an episode of seed s has
the sea, actuators and scenario of the meta5 episode of seed s; the same
Mission, twin and records. The differences:
  - the plant is the family's plant wrapper (operators_rb.rb_plant, an
    RBPlant copy of the Mission's ReducedPlant: bit-identical steps, it
    only keeps each substep's start state, applied actuators and station
    elevations) and the Mission's residual hook is the family's injector
    (GenInjector / RBInjector) inside SafetyHook. The family's substep
    part (the forces / rigid-body terms) is state feedback at every 0.04 s
    substep; its control-step part (the old noise, and for 'rb' the scaled
    old rule) is computed once per step from the 28 M15 inputs (the
    elevations met at mid-step, W_MID, in columns 11:26) and held;
  - a divergence (non-finite, u > 1.5 u_max, |theta| > 45 deg at the end of
    a control step, the envelope of operators_gen.simulate_gen) truncates
    the episode before that step and records its index in 'div' (-1 if
    none); the draw is never redrawn;
  - extra per-step arrays: the safety quantities of D9.6 as simulate_gen
    defines them (APK / AMIN = max / min over the step's six substeps of
    the vertical acceleration after the injection, a_z = plant.last_cg_acc
    after the hook; HMIN = min over the substep ENDS j = 1..6 of the bow
    height F_b + z + s_p x_b theta - the highest bow-station elevation) and
    PUSH (5), the mean push applied over the step (diagnostic);
  - snapshots: at the steps job['snaps'] (the branch moments) the full
    state a continuation needs: the operator state (op.snapshot), its
    random stream, the plant's M10 actuator state (delay line, servo
    velocities), the plant state and the time. data6 continues the episode
    from them (closed loop); these families cannot be relabelled on
    recorded inputs (OperatorGen.run / OperatorRB.run raise).

The recorded arrays are episode5's (S, XS, U, E, RR, RN, ACG, D, AV,
W_MID; S holds the elevations at t, W_MID the elevations met at mid-step,
which is also what model_preview uses as the preview) in the same formats,
so model_preview's training and studies/eval_preview_m15.py read the packs
unchanged. RR / RN are the HELD control-step parts (clipped), as in meta5;
the substep part is in E (and PUSH). The five spatial patterns of M15 are
readings inside the operators (T7 of 'rb'; the old noise envelope), not
arrays of their own, in meta5 as here.

Meta conventions (pack6): op_family = 'm15' (the M15 wave-input FORMAT:
W_MID in the pack; model_preview.DataP requires it), prior_family =
'gen' | 'rb' (the error family), the operator seed renamed op_seed_gen /
op_seed_rb, so every old path that rebuilds an operator from a meta
(data3._replay_operator, relabel._one, data2._targets_one read op_seed;
data5.make_operator reads op_seed_m15) fails with a KeyError instead of
silently rebuilding a v0 or M15 operator. The style gets the keys the
evaluation reads (null, reads_waves, wave_same_step; style6).

No torch here: the episode and branch workers stay numpy-only.
"""
import copy

import numpy as np

from learn.meta import data2
from learn.meta.data2 import (CK0, KN, T_EP, _stations, clip_push,
                              draw_actuators, inputs_of, low_dt, sea_for)
from learn.meta.episode5 import mid_stations_twin
from learn.repro.task import Mission, ctx

FAMILIES6 = ("gen", "rb")
SAFETY = ("APK", "AMIN", "HMIN")
EXTRA6 = SAFETY + ("PUSH",)
DIV_U, DIV_TH = 1.5, np.pi / 4      # the envelope of simulate_gen
N_IMM = slice(13, 28)                # OperatorGen input columns: immersion


# ------------------------------------------------------------ family
def make_op(family, op_seed, lib, dtc, L, relay=False):
    """The operator of a family for a seed (deterministic: 'gen' runs its
    acceptance test, 'rb' its stability redraws, both from the seed)."""
    if family == "gen":
        from learn.meta.operators_gen import OperatorGen
        return OperatorGen(op_seed, lib, dt=dtc, L=L, relay=relay)
    if family == "rb":
        from learn.meta.operators_rb import OperatorRB
        return OperatorRB(op_seed, lib, dt=dtc, L=L, relay=relay)
    raise ValueError(f"family {family!r}: one of {FAMILIES6}")


def wrap_plant(plant):
    """The family's plant wrapper (both families: operators_rb.rb_plant; an
    M10 plant's actuator state starts empty, as a fresh plant's)."""
    from learn.meta import operators_rb as R
    return R.rb_plant(plant)


def make_injector(family, op, st):
    if family == "gen":
        from learn.meta.operators_gen import GenInjector
        return GenInjector(op, st)
    from learn.meta.operators_rb import RBInjector
    return RBInjector(op, st)


def freeboard(p):
    from learn.meta.operators_gen import freeboard as fb
    return fb(p)


class SafetyHook:
    """Mission's residual hook: the family's injector (set .inj once the
    operator state exists), plus per substep the bow height at the
    substep START (h), the vertical acceleration after the injection (az)
    and the applied push (push)."""

    def __init__(self, p, x_st, inj=None):
        self.inj = inj
        self.fb = freeboard(p)
        self.x_b, self.s_p = float(x_st[-1]), float(p["sign_pitch"])
        self.h, self.az, self.push = [], [], []

    def set_held(self, rule, noise):
        """The held control-step part (already clip_push'ed), exactly as
        operators_gen.simulate_gen / operators_rb.simulate_rb hold it."""
        self.inj.rule = np.asarray(rule, float).copy()
        self.inj.noise = np.asarray(noise, float).copy()

    def bow(self, sr, eta):
        e = np.asarray(eta, float).reshape(5, 3)
        sr = np.asarray(sr, float).reshape(-1)
        return float(self.fb + sr[3] + self.s_p * self.x_b * sr[5]
                     - e[4].max())

    def __call__(self, s, dt, plant=None):
        sr, _, _, eta = plant.model.last
        self.h.append(self.bow(sr, eta))
        s = self.inj(s, dt, plant)
        self.az.append(float(plant.last_cg_acc))
        self.push.append(np.asarray(self.inj.last, float).copy())
        return s


def op_inputs6(s0, t, surf, cst, h_mid, cmd):
    """(s26, w_mid, s28) of one step: episode5's m15 inputs (the measured 9,
    heave / pitch about the running attitude, the elevations at t at the
    stations; the operator's 28 with W_MID in 11:26 and the commands).
    surf: anything with _surface / x_st / y_off (the twin, a plant)."""
    x9 = inputs_of(s0, cst["u_ref"], cst["L"], cst["t_max"], cst["rud_max"])
    zr = ((s0[2] - cst["z0"]) / 0.2, (s0[4] - cst["th0"]) / 0.05)
    w15 = _stations(surf, s0, t)
    w_mid = mid_stations_twin(surf, s0, t, h_mid)
    s26 = np.concatenate([x9, zr, w15])
    s28 = np.concatenate([x9, zr, w_mid, np.asarray(cmd, float)])
    return s26, w_mid, s28


def diverged(s, u_max):
    return (not np.all(np.isfinite(s))) or s[6] > DIV_U * u_max \
        or abs(s[4]) > DIV_TH


# ------------------------------------------------------------- style
def style6(op, family, dtc):
    """The family's style plus the keys the evaluation reads:
    null (the error is exactly zero apart from the noise: never for 'gen',
    whose rigid-body Coriolis difference is always on, D9.1 (d); 'rb' with
    the rigid-body part off and a null residual), no_forces ('gen' only: a
    draw without arbitrary forces, P_NONE or forces_off; its error is the
    Coriolis difference plus noise), reads_waves (a DIRECT reading of the
    surface: 'gen' a filter whose source combination has weight on the
    immersion inputs; 'rb' the scaled M15 rule reading waves, or T7 loads),
    wave_same_step (the waves met during a step act within that step:
    'gen' such a filter delayed by less than a control step; 'rb' the M15
    rule's same-step path or any T7 load, which reads the surface at every
    substep). Indirect readings (the low-fidelity boat's own wave response
    through nu0 or the state) are not counted."""
    st = dict(op.style)
    if family == "gen":
        reads, same = False, False
        for f in op.forces:
            wv = np.abs(np.asarray(f["Wc"])[N_IMM]).sum(0) > 0
            for g in f["filt"]["list"]:
                if wv[g["src"]]:            # a filter that is actually used
                    reads = True
                    if g["delay"] < dtc:
                        same = True
        st.update(null=False, no_forces=len(op.forces) == 0,
                  reads_waves=bool(reads), wave_same_step=bool(same),
                  prior_family="gen")
    else:
        t7 = len(op.modes) > 0
        st.update(res_null=bool(st.get("null", False)),
                  res_reads_waves=bool(st.get("reads_waves", False)),
                  res_wave_same_step=bool(st.get("wave_same_step", False)))
        st.update(null=bool(st["res_null"] and op.rb_off),
                  reads_waves=bool(st["res_reads_waves"] or t7),
                  wave_same_step=bool((st["res_wave_same_step"]
                                       and not st["res_null"]) or t7),
                  prior_family="rb")
    return st


# ------------------------------------------------------------ episode
def episode(job):
    """episode5.episode (op_family 'm15') with the family job['family'] in
    the loop (module docstring). Job keys as episode5 (seed, relay, T,
    act_family, act_params, informative, raw64, lib) plus family and snaps
    (steps at which to keep a snapshot). Source world only."""
    from sim import lofi
    fam = job.get("family")
    if fam not in FAMILIES6:
        raise ValueError(f"family {fam!r}: one of {FAMILIES6}")
    world = job.get("world", "low")
    if world != "low" or job.get("lib") is None:
        raise ValueError("episode6: source episodes with a library only")
    c = ctx()
    rng = np.random.default_rng(job["seed"])
    T = job.get("T", T_EP)
    raw64 = bool(job.get("raw64", False))
    snap_at = {int(k) for k in job.get("snaps", ())}
    sea = sea_for(rng, target=False)
    act, act_fam = draw_actuators(job, rng, world)
    m = Mission(world, int(job["seed"]) + 50000, 0, t_end=T, track=0.0,
                sea=sea, plant_params=act)
    new_act = act is not None and act.get("act_family", 0.0) > 0.5
    if new_act and abs(m.dt - low_dt()) > 1e-12:
        raise RuntimeError(f"Mission dt {m.dt} != the dt the delays were "
                           f"drawn for ({low_dt()})")
    dtc = m.sub * m.dt
    op_seed = int(job["seed"]) * 7 + 1
    op = make_op(fam, op_seed, job["lib"], dtc, m.L,
                 relay=job.get("relay", False))
    plant = wrap_plant(m.plant)
    m.plant = m.ep.plant = plant
    p = m.red.p
    hook = SafetyHook(p, plant.x_st)
    m.residual = hook
    rng_op = np.random.default_rng([op_seed, 1])
    st = None                  # built at the first step, from its inputs
    twin = lofi.plant_for(c["db"], m.sea, c["h"], dt=m.dt, params=act)
    twin_sub = int(round(m.sub * m.dt / twin.dt))
    h_mid = 0.5 * dtc
    cst = dict(u_ref=m.u_ref, L=m.L, t_max=m.t_max, rud_max=m.rud_max,
               z0=p.get("z0", 0.0), th0=p.get("th0", 0.0))
    u_max = float(p["u_max"])
    informative = job.get("informative")
    if informative is None:
        informative = bool(rng.random() > 0.25)
    u_tgt = rng.uniform(16, 30) * KN
    psi_tgt = rng.uniform(-np.pi, np.pi) if informative else 0.0
    t_u = t_psi = 0.0
    integ = 0.0
    dth = dict(on=informative and rng.random() < 0.7,
               tau=np.exp(rng.uniform(np.log(0.5), np.log(5.0))),
               amp=rng.uniform(0, 0.15), x=0.0)
    dnz = dict(on=informative and rng.random() < 0.7,
               tau=np.exp(rng.uniform(np.log(0.5), np.log(5.0))),
               amp=rng.uniform(0, 0.15), x=0.0)
    kp = p["m_surge"] / 3.0
    keys = ("S", "XS", "U", "E", "RR", "RN", "ACG", "D", "AV", "W_MID",
            "PUSH")
    rec = {k: [] for k in keys + (("R", "CMD") if raw64 else ())}
    ck, slow, snaps = [], [], {}
    div = -1
    while not m.done():
        t = m.t
        if informative and t >= t_u:
            u_tgt, t_u = rng.uniform(16, 30) * KN, t + rng.uniform(10, 40)
        if informative and t >= t_psi:
            psi_tgt = rng.uniform(-np.pi, np.pi)
            t_psi = t + rng.uniform(20, 60)
        m.ep.heading_ref = psi_tgt
        s0 = m.s.copy()
        e_u = u_tgt - s0[6]
        integ = float(np.clip(integ + e_u * m.dt_ctrl, -20, 20))
        for d in (dth, dnz):
            if d["on"]:
                d["x"] += (-d["x"] / d["tau"]) * m.dt_ctrl + d["amp"] \
                    * np.sqrt(2 * m.dt_ctrl / d["tau"]) * rng.normal()
        thrust = float(np.clip(p["k_drag"] * u_tgt ** 2 + kp * e_u
                               + 0.1 * kp * integ + dth["x"] * m.t_max,
                               0.0, m.t_max))
        rudder = float(np.clip(m.ep._steer(s0, thrust)
                               + dnz["x"] * m.rud_max, -m.rud_max,
                               m.rud_max))
        cmd = [thrust / m.t_max, rudder / m.rud_max]
        s26, w_mid, s28 = op_inputs6(s0, t, twin, cst, h_mid, cmd)
        k = len(rec["S"])
        if st is None:
            st = op.new_state(1, rng=rng_op, init_s=s28[None])
            hook.inj = make_injector(fam, op, st)
        if k >= CK0 and k % CK0 == 0:
            ck.append(k)
            slow.append(op.slow_state(st))
        if k in snap_at:
            snaps[k] = dict(k=k, t=float(t), xs=s0.copy(),
                            st=op.snapshot(st), rng=copy.deepcopy(rng_op),
                            act=plant.act_state())
        rr, rn = op.step(st, s28[None], noise=True, rng=rng_op)
        rr, rn = clip_push(rr[0], rn[0])
        hook.set_held(rr, rn)
        av = m.plant.act_vel() if hasattr(m.plant, "act_vel") \
            else np.zeros(2)
        if new_act:
            # the twin starts the step with the plant's own actuator state
            twin.set_act_state(m.plant.act_state())
        stw = s0.copy()
        for kk in range(twin_sub):
            stw = twin.step(stw, t + kk * twin.dt, thrust, rudder, twin.dt)
        n0 = len(hook.push)
        m.advance(thrust, rudder)
        if not m.finite or diverged(m.s, u_max):
            div = k
            break
        y = np.array([(m.s[i] - stw[i]) / dtc for i in data2.VEL_IDX])
        rec["S"].append(s26)
        rec["XS"].append(s0)
        rec["U"].append(cmd)
        rec["E"].append(y)
        rec["RR"].append(rr)
        rec["RN"].append(rn)
        rec["ACG"].append(m.last_acg)
        rec["D"].append((dth["x"], dnz["x"]))
        rec["AV"].append(av)
        rec["W_MID"].append(w_mid)
        rec["PUSH"].append(np.mean(hook.push[n0:], 0))
        if raw64:
            rec["R"].append(np.mean(hook.push[n0:], 0))
            rec["CMD"].append((thrust, rudder))
    ft = np.float64 if raw64 else np.float32
    n, sub = len(rec["S"]), m.sub
    out = {k_: np.asarray(v, ft) for k_, v in rec.items()}
    if n == 0:
        for k_, w in (("S", 26), ("XS", 14), ("U", 2), ("E", 5), ("RR", 5),
                      ("RN", 5), ("D", 2), ("AV", 2), ("W_MID", 15),
                      ("PUSH", 5), ("R", 5), ("CMD", 2)):
            if k_ in out:
                out[k_] = np.zeros((0, w), ft)
    # safety quantities (simulate_gen's definitions)
    hs = list(hook.h)
    if len(hs) < n * sub + 1:
        # a normal end: the bow height at the start of the next step
        eta = _stations(twin, m.s, m.t)
        sr = data2.plant_to_reduced(m.s[None])[0]
        hs.append(hook.bow(sr, eta))
    az = np.asarray(hook.az[:n * sub], float).reshape(n, sub)
    hh = np.asarray(hs, float)
    out["APK"] = az.max(1).astype(ft) if n else np.zeros(0, ft)
    out["AMIN"] = az.min(1).astype(ft) if n else np.zeros(0, ft)
    out["HMIN"] = np.array([hh[q * sub + 1:(q + 1) * sub + 1].min()
                            for q in range(n)], ft)
    out.update(finite=bool(m.finite), world=world, seed=int(job["seed"]),
               act=act, act_family=act_fam, op_seed=op_seed,
               relay=bool(job.get("relay", False)),
               informative=bool(informative), sea=sea,
               ck=np.asarray(ck, np.int32), slow=slow,
               style=style6(op, fam, dtc), family=fam, div=int(div),
               snaps=snaps)
    return out


_LIB6 = {}


def run_job(job):
    """A worker's episode (studies/meta_step6.py hands this to
    meta_step2.run_jobs): job['lib_path'] is loaded once per process; an
    exception comes back as dict(error, seed) so the chunk goes on."""
    j = dict(job)
    path = j.pop("lib_path", None)
    if j.pop("use_lib", False):
        if _LIB6.get("path") != path:
            d = np.load(path)
            _LIB6.update(path=path, lib=dict(mu=d["mu"], sd=d["sd"],
                                             S=d["S"]))
        j["lib"] = dict(_LIB6["lib"])
    try:
        return episode(j)
    except Exception as ex:                       # keep the chunk going
        import traceback
        return dict(error=repr(ex) + " | " + traceback.format_exc()[-600:],
                    seed=job["seed"])


# ------------------------------------------------------------ packing
def pack6(eps, T_max=None):
    """data2.pack plus W_MID (n, T, 15), APK / AMIN / HMIN (n, T) and PUSH
    (n, T, 5), float32, zero beyond len; one family per pack (raises
    otherwise). Meta: op_family 'm15', prior_family, op_seed renamed
    op_seed_<family>, div (module docstring). Returns (arrays, meta,
    snaps) with snaps {packed episode index: {step: snapshot}} for the
    episodes that have any."""
    kept = [e for e in eps if len(e["E"]) > data2.H + 10]
    fams = {e.get("family") for e in kept}
    if len(fams) > 1 or not fams <= set(FAMILIES6):
        raise ValueError(f"pack6: families {sorted(map(str, fams))}; one "
                         "rigid-body family per pack")
    out, meta = data2.pack(eps, T_max)
    n, T = out["E"].shape[:2]
    for k, w in (("W_MID", 15), ("APK", None), ("AMIN", None),
                 ("HMIN", None), ("PUSH", 5)):
        a = np.zeros((n, T) if w is None else (n, T, w), np.float32)
        for i, e in enumerate(kept):
            L = min(len(e[k]), T)
            a[i, :L] = e[k][:L]
        out[k] = a
    snaps = {}
    for i, (m_, e) in enumerate(zip(meta, kept)):
        fam = e["family"]
        m_[f"op_seed_{fam}"] = m_.pop("op_seed")
        m_.update(op_family="m15", prior_family=fam, div=int(e["div"]))
        if e.get("snaps"):
            snaps[i] = e["snaps"]
    return out, meta, snaps
