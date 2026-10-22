#!/usr/bin/env python3
"""
Does the chain  hull -> plant -> reduced model -> limits -> controllers
carry a hull swap, and which parts are secretly tied to scarab195?

A TEST WITH A KNOWN ANSWER: THE FROUDE TWIN
Scale every length of scarab195 by lam, mass by lam^3, times by lam^0.5,
speeds by lam^0.5, forces by lam^3, power by lam^3.5, and the kinematic
viscosity by lam^1.5 (so the Reynolds number, and with it the ITTC-1957
friction coefficient, is unchanged). Every term of the v0 plant is then
Froude-similar (gravity, inertia, added mass, crossflow, buoyancy, jet
momentum) -- dimensional analysis says the twin's motion, expressed in
L, m, sqrt(L/g), must be the base hull's, exactly. So every nondimensional
output of the chain (running trim, R/(m g), the identified reduced-model
coefficients times their lam power, the turning table r sqrt(L/g) against
u / sqrt(g L), the closed-loop mission metrics, which are all priced in
g, L and u_ref) must agree between base and twin up to round-off and the
captive-run tolerances. Anything that does not is an ABSOLUTE constant
somewhere in the chain -- precisely what makes a hull swap unsafe.

What cannot be scaled through the hull's inputs is scaled in this process
only (no file is edited): sim/planing_vessel.NU (read at every derivative)
and the two time constants hard-coded where the plant builds its jet
(sim/planing_vessel.py:128-133, rate_max = t_max / 0.3 s and
tau_prime_loss = 0.05 s), through a Waterjet subclass put in the module's
namespace. One absolute constant is left on purpose because it cannot be
reached without an edit: the air-drag moment arm, 0.5 m above the CG
(sim/planing_vessel.py:297-298). Config 'noair' (cda = 0) removes it, so
the similarity test is exact there; config 'air' shows its effect.

Nothing is written outside studies/_cache/principles/hull_swap/: the lofi
cache path and final_eval's log directory are redirected in-process.

    python -m studies.principles_hull_swap --config base_air  [--stages ...]
    python -m studies.principles_hull_swap --config twin_air  --lam 1.5
    python -m studies.principles_hull_swap --config base_noair
    python -m studies.principles_hull_swap --config twin_noair --lam 1.5
    python -m studies.principles_hull_swap --compare --lam 1.5

Each config runs in its own process (module-level caches and the NU patch
must not cross configs).
"""
import argparse
import json
import math
import os
import sys
import time
import traceback

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np   # noqa: E402

OUT = os.path.join(HERE, "_cache", "principles", "hull_swap")
G = 9.81
KN = 0.514444
# base-hull speeds of the steady curves (m/s); the twin runs at u sqrt(lam)
U_STEADY = (6.0, 8.0, 10.0, 12.8611, 15.0)
THROTTLES = (0.3, 0.45, 0.6, 0.75, 0.9, 1.0)     # studies/final_eval.py:252

# lam exponent of each identified reduced-model coefficient
# (control/reduced.py:316-333): ratio twin / base must be lam ** e
EXP = dict(L=1, T=1, B=1, draft=1, x_bow=1, x_stern=1, draft_bow=1, z0=1,
           th0=0, tau_u=0.5, tau_r=0.5, dt_ctrl=0.5, k_drag=2,
           wn_heave=-0.5, wn_pitch=-0.5, z_heave=0, z_pitch=0,
           k_nomoto=-0.5, v_slam=0.5, u_ss=0.5, u_max=0.5, u_design=0.5,
           m_surge=3, m_sway=3, m_coriolis=3, t_max=3, rud_max=0,
           rud_stall=0, k_sway=2, k_lin_sway=2.5, k_lift=2,
           c_wave_sway=0, c_wave_yaw=0, k_jet_side=0, k_nomoto_f=-3.5,
           steer_jet=0, planing=0, sign_pitch=0)


def log(msg):
    line = time.strftime("%H:%M:%S ") + str(msg)
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "run.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ------------------------------------------------------------ the hulls
def make_hull(config, lam):
    """(hull, time scale s = sqrt(lam)) for a config name."""
    from hydro import hulls
    from hydro.planing import PlaningHull
    base = hulls.get("scarab195")
    twin = config.startswith("twin")
    air = config.endswith("_air")
    if config == "base_air":
        return base, 1.0                      # exactly what the project runs
    if config == "hullB":
        # A second, NOT similar parametric hull (L/b 3.3, deadrise 16 deg,
        # LCG 0.36 L; inside Savitsky's ranges at 8-15 m/s). Its jet gives a
        # Savitsky top speed of ~46 kn without air drag (as scarab195's ~44
        # kn). Same placeholder bow (taper 0.35, no stem rise).
        return PlaningHull(
            name="probe_hullB", L=6.6, b=2.0, beta_deg=16.0, mass=1900.0,
            lcg=0.36 * 6.6, vcg=0.60, depth=1.2, cda=2.0,
            u_design=25 * KN,
            jet=dict(d_nozzle=0.10, p_max=260e3, eta_pump=0.775,
                     nozzle_max_deg=25.0, nozzle_rate_deg=40.0,
                     x_intake=0.9, z_nozzle=0.15, tau=0.6)), 1.0
    lam = lam if twin else 1.0
    s = math.sqrt(lam)

    class ScaledHull(PlaningHull):
        """PlaningHull whose plant and control steps scale with sqrt(lam)
        (PlaningHull.scales() fixes 0.02 s and 0.25 s,
        hydro/planing.py:136-140)."""
        def scales(self):
            d = super().scales()
            d["dt"] = d["dt"] * s
            d["dt_ctrl"] = d["dt_ctrl"] * s
            return d

    j = dict(base.jet)
    # every jet input the plant reads (sim/planing_vessel.py:118-138), the
    # defaults made explicit, then scaled
    jet = dict(d_nozzle=j["d_nozzle"] * lam, p_max=j["p_max"] * lam ** 3.5,
               eta_pump=j["eta_pump"], nozzle_max_deg=j["nozzle_max_deg"],
               nozzle_rate_deg=j["nozzle_rate_deg"] / s,
               x_intake=j["x_intake"] * lam, z_nozzle=j["z_nozzle"] * lam,
               x_nozzle=j.get("x_nozzle", 0.0) * lam,
               z_intake=j.get("z_intake", 0.0) * lam,
               tau=j["tau"] * s, delay=j.get("delay", 0.1) * s,
               nozzle_tau=j.get("nozzle_tau", 0.1) * s,
               nozzle_delay=j.get("nozzle_delay", 0.1) * s,
               tau_prime_back=j.get("tau_prime_back", 0.5) * s,
               k_rev=j.get("k_rev", 0.6))
    name = (f"scarab195_froude{lam:g}" if twin else "scarab195_lam1") + \
        ("" if air else "_noair")
    h = ScaledHull(
        name=name, L=base.L * lam, b=base.b * lam, beta_deg=base.beta_deg,
        mass=base.mass * lam ** 3, lcg=base.lcg * lam, vcg=base.vcg * lam,
        depth=base.depth * lam, k_pitch=base.k_pitch, k_roll=base.k_roll,
        k_yaw=base.k_yaw, taper=base.taper, stem_rise=base.stem_rise * lam,
        cda=base.cda * lam ** 2 if air else 0.0,
        u_design=base.u_design * s, jet=jet, n_stations=base.n_stations,
        k_a=base.k_a, c_dc=base.c_dc, a_b=base.a_b, cd_lat=base.cd_lat,
        roll_zeta=base.roll_zeta, rho=base.rho,
        transom_correction=base.transom_correction)
    return h, s


def install_scaling(lam, s):
    """The two things a hull cannot carry, scaled in this process only."""
    import sim.planing_vessel as PV
    PV.NU = PV.NU * lam ** 1.5
    Base = PV.Waterjet

    class ScaledWaterjet(Base):
        def __init__(self, *a, **kw):
            kw["rate_max"] = kw["rate_max"] / s
            kw["tau_prime_loss"] = kw["tau_prime_loss"] * s
            super().__init__(*a, **kw)
    PV.Waterjet = ScaledWaterjet


def redirect_caches():
    """lofi's cache and final_eval's log go to OUT, never to the project's
    studies/_cache/lofi_*.json or studies/_cache/final_eval/."""
    from sim import lofi
    lofi._cache_path = lambda name: os.path.join(OUT, f"lofi_{name}.json")
    from studies import final_eval as FE
    FE.log = log
    FE.out_dir = lambda: OUT


# ------------------------------------------------------------ stages
def stage_build(h, s):
    from sim import physics
    from sim.test_vessel import Monochromatic
    calm = Monochromatic(1.0, 0.0)
    pl = h.plant(calm)
    gz = physics.gz_plant_for(None, calm, h)
    return dict(name=h.name, L=h.L, mass=h.mass, T=pl.T, GM=pl.GM,
                gz_max=gz._gz_info["gz_max"],
                phi_vanish_deg=math.degrees(gz.phi_vanish),
                freeboard=pl.freeboard, draft_bow=pl.draft_bow,
                t_max=float(pl.prop.t_max), dt=pl.dt,
                top_speed=float(gz.top_speed()),
                physics_sha1=physics.stamp(pl)["physics_sha1"])


def stage_steady(h, s):
    from sim.test_vessel import Monochromatic
    pl = h.plant(Monochromatic(1.0, 0.0))
    out = []
    from hydro.planing import savitsky
    for ub in U_STEADY:
        u = ub * s
        z, th = pl.running_attitude(u)
        cr = pl.calm_resistance(u)
        sv = savitsky(u, h.mass, h.lcg, h.b, h.beta_deg)
        out.append(dict(u=u, z=z, th=th, trim_deg=cr["trim_deg"],
                        R=cr["R"], pressure=cr["pressure"],
                        friction=cr["friction"], air=cr["air"],
                        lw=cr["wetted_length"],
                        savitsky_trim_deg=float(sv["tau_deg"]),
                        savitsky_R=float(sv["R"]),
                        savitsky_valid=bool(sv["valid"])))
    return out


def stage_ident(h, s):
    from control.reduced import ReducedModel
    from sim import config
    t0 = time.time()
    plant, _ = config.calm_plant(h, db=None)
    p = ReducedModel.identify(plant).p
    return dict(p={k: float(v) for k, v in p.items()},
                seconds=time.time() - t0)


def stage_turn(h, s):
    """studies/final_eval.phase_turn with the times scaled by s, on the GZ
    plant; for the base hull it is phase_turn exactly."""
    from sim import physics
    from sim.test_vessel import Monochromatic
    plant = physics.gz_plant_for(None, Monochromatic(1.0, 0.0), h)
    dt = plant.dt
    t_max, rud = float(plant.prop.t_max), float(plant.rudder.max)
    U, R, Fs = [], [], []
    for f in THROTTLES:
        st = plant.initial_state(0.8 * float(plant.prop.u_ref))
        t, us, rs = 0.0, [], []
        for i in range(int(round(45.0 * s / dt))):
            r_cmd = rud if t >= 20.0 * s else 0.0
            st = plant.step(st, t, f * t_max, r_cmd, dt)
            t += dt
            if t >= 37.0 * s:
                us.append(st[6])
                rs.append(abs(st[11]))
        if not np.all(np.isfinite(st)):
            continue
        U.append(float(np.mean(us)))
        R.append(float(np.mean(rs)))
        Fs.append(f)
    return dict(u=U, r=R, throttle=Fs, plant="gz", dt=dt)


def stage_missions(h, s, lam, rows, T, seeds, legs, turn,
                   similar_model=False):
    from learn.repro import task
    from studies import final_eval as FE
    task.HULL = h                      # config.load accepts a Hull object
    task.HS, task.TP = task.HS * lam, task.TP * s
    FE.OPT.update(plant="gz", device="cpu")
    if similar_model:
        # the controllers' model made EXACTLY similar to the base hull's:
        # every identified coefficient = base value x lam ** EXP (removes
        # the one identification difference the twin test found, k_lin_sway
        # from the endpoint of an oscillating turn, control/reduced.py:270)
        pb = json.load(open(os.path.join(OUT, "base_noair.json")))["ident"]["p"]
        c = task.ctx()
        for k in list(c["p"]):
            if k in EXP and k in pb:
                c["p"][k] = pb[k] * lam ** EXP[k]
        log(f"  controller model forced similar to base ({len(pb)} keys)")
    out = []
    for row in rows:
        spec, why = FE.resolve_row(row)
        if spec is None:
            out.append(dict(row=row, error=why))
            continue
        for seed in seeds:
            for leg in legs:
                t0 = time.time()
                try:
                    r = FE.run_mission(spec, seed, leg, T * s, turn, {})
                    r.pop("events_raw", None)
                    r.pop("_carry_state", None)
                    r["wall_s"] = time.time() - t0
                    out.append(r)
                    log(f"  mission {row} s{seed} l{leg}: score "
                        f"{r['score']:.4f} u_along {r['u_along']:.3f} "
                        f"({r['wall_s']:.0f} s)")
                except Exception as ex:   # a failing stage is a finding
                    out.append(dict(row=row, seed=seed, leg=leg,
                                    error=repr(ex),
                                    tb=traceback.format_exc()[-2000:]))
                    log(f"  mission {row} s{seed} l{leg} FAILED: {ex!r}")
    return out


def stage_traj(h, s, lam, seed, leg, T, similar_model, fixcost):
    """One hand mission, every control step's state and command saved, for
    the divergence test (compare_traj). similar_model: the twin's identified
    model forced to base_noair's, Froude-scaled. fixcost: the hand MPPI's
    objective made time-scale free -- control/mpc.py multiplies the running
    cost by dt (seconds, :161-166) but not the jump terms (:167-168) or the
    softmax temperature lam (:182), so the twin's objective is NOT s x the
    base's; scaling lam and w_dthrust, w_drudder by s makes it so."""
    from learn.repro import task
    from studies import final_eval as FE
    from studies.sim2real_jet import theta, z_hand
    task.HULL = h
    task.HS, task.TP = task.HS * lam, task.TP * s
    FE.OPT.update(plant="gz", device="cpu")
    if similar_model:
        pb = json.load(open(os.path.join(OUT, "base_noair.json")))["ident"]["p"]
        c = task.ctx()
        for k in list(c["p"]):
            if k in EXP and k in pb:
                c["p"][k] = pb[k] * lam ** EXP[k]
    w, floor = theta(z_hand())
    m = FE.make_mission(seed, leg, T * s, weights=w, floor=floor)
    if fixcost:
        m.ep.ctrl.lam *= s
        m.ep.ctrl.w = m.ep.ctrl.w.copy()
        m.ep.ctrl.w[5:7] *= s
    S, CMD = [m.s.copy()], []
    while not m.done():
        thrust, rud = m.mpc_command()
        CMD.append((thrust / m.t_max, rud / m.rud_max))
        m.advance(thrust, rud)
        S.append(m.s.copy())
    return dict(S=np.array(S).tolist(), CMD=np.array(CMD).tolist(),
                L=float(m.L), dtc=float(m.dt * m.sub))


def compare_traj(a):
    """Max over the nondimensional state (x/L, y/L, z/L, roll, pitch, yaw,
    u, v / sqrt(gL), r sqrt(L/g)) of |config - base_noair| per control
    step, for every traj file given in --traj-files."""
    def nd(r):
        S, L = np.asarray(r["S"]), r["L"]
        U = math.sqrt(G * L)
        return np.column_stack([S[:, 0] / L, S[:, 1] / L, S[:, 2] / L,
                                S[:, 3], S[:, 4], S[:, 5], S[:, 6] / U,
                                S[:, 7] / U, S[:, 11] * math.sqrt(L / G)])
    base = json.load(open(os.path.join(OUT, "base_noair_traj.json")))["traj"]
    B, CB = nd(base), np.asarray(base["CMD"])
    out = {}
    for fn in a.traj_files.split(","):
        r = json.load(open(os.path.join(OUT, fn)))["traj"]
        X, CX = nd(r), np.asarray(r["CMD"])
        n = min(len(X), len(B))
        dS = np.abs(X[:n] - B[:n]).max(1)
        dC = np.abs(CX[:n - 1] - CB[:n - 1]).max(1)
        steps = [k for k in (0, 1, 2, 5, 10, 20, 40, 60, 80, 100, 120, 150)
                 if k < n - 1]
        out[fn] = dict(state=[float(dS[k]) for k in steps],
                       cmd=[float(dC[k]) for k in steps], steps=steps,
                       first_above_1e3=int(np.argmax(dS > 1e-3))
                       if (dS > 1e-3).any() else None)
        print(fn, json.dumps(out[fn]))
    with open(os.path.join(OUT, "compare_traj.json"), "w") as f:
        json.dump(out, f, indent=1)


def seed_stats(a):
    """Paired mission statistics, base_noair against a twin and against the
    lam = 1 + 1e-6 chaos control, in Froude units. --twin-files: the twin's
    JSONs (default: the raw lam 1.5 runs)."""
    def load(names):
        out = {}
        for n in names:
            pth = os.path.join(OUT, n)
            if os.path.exists(pth):
                c = json.load(open(pth))
                for m in c.get("missions", []):
                    if "error" not in m:
                        out[(m["row"], m["seed"], m["leg"])] = (m, c["lam"])
        return out
    base = load(("base_noair.json", "base_noair_seeds.json"))
    twin = load(a.twin_files.split(","))
    chaos = load(("twin_noair_chaos.json", "twin_noair_seedschaos.json"))
    L0 = 5.4

    def nd(rec, key):
        m, lam = rec
        L = L0 * lam
        if key == "u_along_nd":
            return m["u_along"] / math.sqrt(G * L)
        if key == "cross_over_L":
            return m["cross_rms"] / L
        return m[key]
    res = {}
    for row in ("hand", "m0"):
        ks = sorted(k for k in base if k[0] == row and k in twin
                    and k in chaos)
        n = len(ks)
        res[row] = dict(n=n)
        print(f"{row}: n = {n} paired missions (twin {a.twin_files})")
        for key in ("score", "c_speed", "c_impact", "c_track", "u_along_nd",
                    "cross_over_L", "acc_cg_p99"):
            b = np.array([nd(base[k], key) for k in ks])
            t = np.array([nd(twin[k], key) for k in ks])
            c = np.array([nd(chaos[k], key) for k in ks])
            dt_, dc = t - b, c - b
            r = dict(base=b.mean(), twin=t.mean(), chaos=c.mean(),
                     rms_tb=float(np.sqrt(np.mean(dt_ ** 2))),
                     rms_cb=float(np.sqrt(np.mean(dc ** 2))),
                     mean_tb=dt_.mean(), se_tb=dt_.std(ddof=1) / math.sqrt(n),
                     mean_cb=dc.mean(), se_cb=dc.std(ddof=1) / math.sqrt(n))
            res[row][key] = {k: float(v) for k, v in r.items()}
            print(f"  {key:<13} " + " ".join(f"{k} {v:.4f}"
                                             for k, v in r.items()))
    with open(os.path.join(OUT, f"seed_stats{a.tag}.json"), "w") as f:
        json.dump(res, f, indent=1)


def run_config(a):
    os.makedirs(OUT, exist_ok=True)
    h, s = make_hull(a.config, a.lam)
    lam = s * s
    if a.config.startswith("twin"):
        install_scaling(lam, s)
    redirect_caches()
    path = os.path.join(OUT, f"{a.config}{a.tag}.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    res.update(config=a.config, lam=lam, s=s, hull=h.name)
    for st in a.stages.split(","):
        t0 = time.time()
        try:
            if st == "build":
                res["build"] = stage_build(h, s)
            elif st == "steady":
                res["steady"] = stage_steady(h, s)
            elif st == "ident":
                res["ident"] = stage_ident(h, s)
            elif st == "turn":
                res["turn"] = stage_turn(h, s)
            elif st == "traj":
                res["traj"] = stage_traj(h, s, lam, int(a.seeds.split(",")[0]),
                                         int(a.legs.split(",")[0]), a.T,
                                         a.similar_model, a.fixcost)
            elif st == "missions":
                turn = res.get("turn") or stage_turn(h, s)
                res["missions"] = stage_missions(
                    h, s, lam, a.rows.split(","), a.T,
                    [int(x) for x in a.seeds.split(",")],
                    [int(x) for x in a.legs.split(",")], turn,
                    similar_model=a.similar_model)
            else:
                raise SystemExit(f"unknown stage {st!r}")
            res.setdefault("errors", {}).pop(st, None)
        except Exception as ex:
            res.setdefault("errors", {})[st] = traceback.format_exc()[-3000:]
            log(f"[{a.config}] stage {st} FAILED: {ex!r}")
        log(f"[{a.config}] {st} done ({time.time() - t0:.1f} s)")
        with open(path + ".tmp", "w") as f:
            json.dump(res, f, indent=1, default=float)
        os.replace(path + ".tmp", path)


# ------------------------------------------------------------ compare
def _rel(a, b):
    return float(a / b - 1.0) if b != 0 else (0.0 if a == 0 else math.inf)


def compare_pair(base, twin, lam):
    s = math.sqrt(lam)
    Lb, Lt = base["build"]["L"], twin["build"]["L"]
    mb, mt = base["build"]["mass"], twin["build"]["mass"]
    out = {}
    if "build" in base and "build" in twin:
        b, t = base["build"], twin["build"]
        out["build"] = {k: _rel(t[k], b[k] * lam ** e) for k, e in (
            ("T", 1), ("GM", 1), ("gz_max", 1), ("freeboard", 1),
            ("draft_bow", 1), ("t_max", 3), ("top_speed", 0.5))}
        out["build"]["phi_vanish_deg_diff"] = t["phi_vanish_deg"] - \
            b["phi_vanish_deg"]
    if "steady" in base and "steady" in twin:
        rows = []
        for b, t in zip(base["steady"], twin["steady"]):
            rows.append(dict(
                Fn=b["u"] / math.sqrt(G * Lb),
                trim_deg_base=b["trim_deg"], trim_deg_diff=t["trim_deg"]
                - b["trim_deg"],
                rise_over_L_rel=_rel(t["z"] / Lt, b["z"] / Lb),
                R_over_W_base=b["R"] / (mb * G),
                R_over_W_rel=_rel(t["R"] / (mt * G), b["R"] / (mb * G)),
                friction_rel=_rel(t["friction"] / mt, b["friction"] / mb),
                pressure_rel=_rel(t["pressure"] / mt, b["pressure"] / mb),
                lw_over_L_rel=_rel(t["lw"] / Lt, b["lw"] / Lb)))
        out["steady"] = rows
    if "ident" in base and "ident" in twin:
        pb, pt = base["ident"]["p"], twin["ident"]["p"]
        out["ident"] = {k: _rel(pt[k], pb[k] * lam ** EXP[k])
                        for k in pb if k in EXP}
        out["ident_unmapped"] = sorted(k for k in pb if k not in EXP)
    if "turn" in base and "turn" in twin:
        tb, tt = base["turn"], twin["turn"]
        out["turn"] = [dict(throttle=f, Fn=ub / math.sqrt(G * Lb),
                            r_nd_base=rb * math.sqrt(Lb / G),
                            r_nd_rel=_rel(rt * math.sqrt(Lt / G),
                                          rb * math.sqrt(Lb / G)),
                            u_rel=_rel(ut, ub * s))
                       for f, ub, rb, ut, rt in zip(tb["throttle"], tb["u"],
                                                    tb["r"], tt["u"],
                                                    tt["r"])]
    if "missions" in base and "missions" in twin:
        mrows = []
        for b in base["missions"]:
            t = next((x for x in twin["missions"] if x.get("row") == b.get(
                "row") and x.get("seed") == b.get("seed")
                and x.get("leg") == b.get("leg")), None)
            if t is None or "error" in b or "error" in t:
                mrows.append(dict(row=b.get("row"), seed=b.get("seed"),
                                  leg=b.get("leg"),
                                  error=(b.get("error"), (t or {}).get(
                                      "error"))))
                continue
            r = dict(row=b["row"], seed=b["seed"], leg=b["leg"])
            for k in ("score", "c_speed", "c_impact", "c_track",
                      "acc_cg_p99", "acc_bow_p99", "heading_rms",
                      "apk_mean_g", "safe_steps"):
                r[k] = (b.get(k), t.get(k))
            r["u_along_nd"] = (b["u_along"] / math.sqrt(G * Lb),
                               t["u_along"] / math.sqrt(G * Lt))
            r["cross_rms_over_L"] = (b["cross_rms"] / Lb,
                                     t["cross_rms"] / Lt)
            r["n_steps"] = (b["n_steps"], t["n_steps"])
            r["wall_s"] = (b.get("wall_s"), t.get("wall_s"))
            mrows.append(r)
        out["missions"] = mrows
    return out


def compare(a):
    lam = a.lam
    res = {}
    for kind in ("air", "noair"):
        pb = os.path.join(OUT, f"base_{kind}.json")
        pt = os.path.join(OUT, f"twin_{kind}{a.tag}.json")
        if os.path.exists(pb) and os.path.exists(pt):
            res[kind] = compare_pair(json.load(open(pb)), json.load(open(pt)),
                                     lam)
    # the base turn table against the project's cached one (harness check)
    pb = os.path.join(OUT, "base_air.json")
    cached = os.path.join(HERE, "_cache", "final_eval", "turn_gz.json")
    if os.path.exists(pb) and os.path.exists(cached):
        mine = json.load(open(pb)).get("turn")
        theirs = json.load(open(cached))
        if mine:
            res["turn_vs_final_eval_cache"] = dict(
                max_abs_diff_u=float(np.max(np.abs(np.subtract(
                    mine["u"], theirs["u"])))),
                max_abs_diff_r=float(np.max(np.abs(np.subtract(
                    mine["r"], theirs["r"])))))
    path = os.path.join(OUT, f"compare{a.tag}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=("base_air", "twin_air", "base_noair",
                                         "twin_noair", "hullB"))
    ap.add_argument("--lam", type=float, default=1.5)
    ap.add_argument("--stages", default="build,steady,ident,turn")
    ap.add_argument("--rows", default="hand,m0")
    ap.add_argument("--T", type=float, default=48.0)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--legs", default="0")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--fixcost", action="store_true",
                    help="traj: time-scale-free hand MPPI objective")
    ap.add_argument("--compare-traj", action="store_true")
    ap.add_argument("--seed-stats", action="store_true")
    ap.add_argument("--twin-files",
                    default="twin_noair.json,twin_noair_seeds15.json")
    ap.add_argument("--traj-files", default="")
    ap.add_argument("--similar-model", action="store_true",
                    help="missions: force the controllers' identified model "
                    "to base_noair's, Froude-scaled")
    ap.add_argument("--tag", default="", help="suffix of the twin's output "
                    "file (e.g. _lam1e-6 for the chaos control run)")
    a = ap.parse_args()
    if a.seed_stats:
        seed_stats(a)
    elif a.compare_traj:
        compare_traj(a)
    elif a.compare:
        compare(a)
    else:
        if a.config is None:
            raise SystemExit("--config or --compare")
        run_config(a)


if __name__ == "__main__":
    main()
