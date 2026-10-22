#!/usr/bin/env python3
"""
How much can per-wave throttle modulation do on the high-fidelity planing
plant? The physics behind "accelerate while climbing a wave, ease off at the
crest" for scarab195 in the final-evaluation sea.

Phases (results in studies/_cache/wave_phase/, one JSON / pickle each):

  config   the propulsion chain as the plant actually builds it (delay, lag,
           rate limit, dead band, prime law, intake / nozzle position), the
           calm-water resistance, the thrust command that holds each speed,
           the jet's net thrust against speed, the surge time constant
  calm     calm water, steady 20 and 26 kn: a thrust COMMAND modulated
           around the holding command (sine +-30 %, and a square wave with
           the largest symmetric swing the thrust range allows) at periods
           1.5-20 s, held per control step (0.24 s) as any controller in
           studies/final_eval.py does; amplitude and phase of the command,
           thrust state, jet force, surge speed, pitch, heave, CG and bow
           vertical acceleration at the forcing frequency (least squares on
           the last whole periods)
  sea      the evaluation sea (studies/final_eval.make_mission: gz plant,
           Hs 1 m, Tp 5 s, leg 0 head seas, leg 1 45 deg off the bow),
           constant thrust commands, the heading autopilot on the nozzle
           (m.ep._steer, as rows hand / m0 with --steer auto): mean speed,
           wave-induced spread of the same quantities, the encounter
           frequency, the CG peaks of studies/safety_events (k = 2), bow
           wetting, intake emergence and the pump's prime
  lock     the same missions under an ORACLE wave-phase-locked bang-bang
           throttle: the phase of the encountered elevation at the CG
           (read from the plant's own sea), thrust high on one half of the
           encounter cycle and low on the other, the switching phase swept
           over a full cycle -- 'accelerate on the face, ease at the crest'
           and every other timing; compared with the constant-thrust runs
           of the same seed and leg at the same mean speed
  peaks    what the >= 7 g peaks (k = 2) of the sea runs are: hull airborne
           before, bow out before, the slamming (Wagner momentum) force and
           where along the hull it acts, the wave phase at the CG
  report   the tables

    .venv/Scripts/python.exe -m studies.wave_phase_physics --phase config,calm
    .venv/Scripts/python.exe -m studies.wave_phase_physics --phase sea \
        --seeds 0-3
    .venv/Scripts/python.exe -m studies.wave_phase_physics --phase lock \
        --seeds 0-3
    .venv/Scripts/python.exe -m studies.wave_phase_physics --phase report
"""
import argparse
import json
import math
import os
import pickle
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np   # noqa: E402

OUT = os.path.join(HERE, "_cache", "wave_phase")
KN = 0.514444
G = 9.81
DT, SUB = 0.02, 12            # plant step, plant steps per control step
DTC = DT * SUB                # 0.24 s
T_SKIP = 15.0                 # s of each sea run left out (start transient)


def log(msg):
    os.makedirs(OUT, exist_ok=True)
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(os.path.join(OUT, "run.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save(obj, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    if name.endswith(".json"):
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=1, default=float)
    else:
        with open(p + ".tmp", "wb") as f:
            pickle.dump(obj, f)
    os.replace(p + ".tmp", p)


def load(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return None
    if name.endswith(".json"):
        return json.load(open(p, encoding="utf-8"))
    return pickle.load(open(p, "rb"))


def free_gb():
    from learn.repro.task import avail_gb
    return avail_gb()


# ------------------------------------------------------------ plants
def calm_plant():
    """The gz plant of the evaluation (sim/planing_vessel_gz.plant_for, the
    path studies/final_eval.plant_factory('gz') takes) in calm water."""
    from learn.repro import task
    from sim import planing_vessel_gz as gz
    from sim.test_vessel import Monochromatic
    c = task.ctx()
    return gz.plant_for(c["db"], Monochromatic(1.0, 0.0), c["h"], dt=DT)


# ------------------------------------------------------------ config
def phase_config():
    pl = calm_plant()
    p = pl.prop
    xn, _, zn = pl.thrusters[0]
    xi, _, zi = pl.intakes[0]
    out = dict(
        t_max_N=p.t_max, t_min_N=p.t_min, tau_s=p.tau,
        rate_max_N_per_s=p.rate_max, rate_max_tmax_per_s=p.rate_max / p.t_max,
        dead_band_N=p.dead_band, delay_steps=p.delay.n,
        delay_s=p.delay.n * DT, plant_dt=pl.dt, u_ref=p.u_ref,
        prime_band_m=p.prime_band, tau_prime_loss_s=p.tau_prime_loss,
        tau_prime_back_s=p.tau_prime_back,
        nozzle_x_from_cg=xn, nozzle_z_from_cg=zn,
        intake_x_from_cg=xi, intake_z_from_cg=zi,
        nozzle_area_m2=p.pump.area, p_max_W=p.pump.p_max,
        mass=pl.m, surge_mass=pl.Mtot[0, 0], Iy=pl.Iy, L=pl.L, B=pl.B,
        th_rest_deg=math.degrees(pl.Th_rest), z_rest=pl.Z_rest,
        freeboard_bow=pl.freeboard, draft_bow=pl.draft_bow,
        x_bow=float(pl.x[-1]), x_stern=float(pl.x[0]),
        v_slam=pl.v_slam)
    rows = []
    for kn in (12, 15, 18, 20, 22, 24, 26, 28, 30, 32):
        u = kn * KN
        cr = pl.calm_resistance(u)
        s = pl.initial_state(u)
        T0 = float(s[12])
        # +-1 kn: the captive runs converge to a tolerance, and a +-0.1
        # m/s difference of two of them is mostly that tolerance
        du = 1.0 * KN
        dR = (pl.calm_resistance(u + du)["R"]
              - pl.calm_resistance(u - du)["R"]) / (2 * du)
        dF = (p.forces(T0, 0.0, u + du)[0] - p.forces(T0, 0.0, u - du)[0]) \
            / (2 * du)
        dFdT = (p.forces(1.01 * T0, 0.0, u)[0]
                - p.forces(0.99 * T0, 0.0, u)[0]) / (0.02 * T0)
        rows.append(dict(
            kn=kn, R_N=cr["R"], trim_deg=cr["trim_deg"], rise_m=cr["rise"],
            wetted_length_m=cr["wetted_length"], cmd_hold_N=T0,
            cmd_hold_frac=T0 / p.t_max,
            net_at_cmd_N=p.forces(T0, 0.0, u)[0],
            net_full_power_N=p.forces(p.t_max, 0.0, u)[0],
            dR_du=dR, dF_du_at_cmd=dF, dF_dcmd=dFdT,
            T_surge_s=pl.Mtot[0, 0] / max(dR - dF, 1e-9)))
    out["speeds"] = rows
    out["top_speed_kn"] = pl.top_speed() / KN
    save(out, "config.json")
    log(f"config: t_max {p.t_max:.0f} N, tau {p.tau} s, delay "
        f"{p.delay.n} x {DT} s, rate {p.rate_max:.0f} N/s, prime band "
        f"{p.prime_band:.4f} m, loss/back {p.tau_prime_loss}/"
        f"{p.tau_prime_back} s, top {out['top_speed_kn']:.1f} kn")
    for r in rows:
        log(f"  {r['kn']:>3} kn  R {r['R_N']:7.0f} N  trim "
            f"{r['trim_deg']:5.2f} deg  rise {r['rise_m']:.3f} m  cmd "
            f"{r['cmd_hold_N']:6.0f} N ({r['cmd_hold_frac']:.2f})  "
            f"dR/du {r['dR_du']:6.0f}  dF/du {r['dF_du_at_cmd']:6.0f}  "
            f"dF/dcmd {r['dF_dcmd']:.2f}  T_surge {r['T_surge_s']:.1f} s")
    return out


# ------------------------------------------------------------ fits
def harmonic(t, y, w):
    """Least squares y = c0 + c1 (t - tm) + a cos wt + b sin wt; returns
    (amplitude, phase rad: y ~ A cos(wt + ph)) of the fundamental."""
    tm = t.mean()
    X = np.column_stack([np.ones_like(t), t - tm, np.cos(w * t),
                         np.sin(w * t)])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = c[2], c[3]
    return float(np.hypot(a, b)), float(np.arctan2(-b, a))


QUANT = ("cmd", "thr", "fx", "u", "pitch_deg", "heave", "w", "a_cg_g",
         "a_bow_g", "rel_bow", "prime")


def run_calm(U_kn, P, shape, amp=0.3):
    """One calm-water run: steady at U_kn, then the command modulated with
    period P from t = 0 (shape 'sine': T0 (1 + amp sin); 'square': T0 + D
    sign(sin), D the largest symmetric swing within [0, t_max])."""
    pl = calm_plant()
    u = U_kn * KN
    s = pl.initial_state(u)
    T0 = float(s[12])
    tmax = pl.prop.t_max
    D = min(T0, tmax - T0)
    settle = max(20.0, 3.0 * P)
    win = P * math.ceil(max(25.0, 5.0 * P) / P)
    n_ctrl = int(math.ceil((settle + win) / DTC))
    w = 2 * np.pi / P
    rec = {q: [] for q in QUANT}
    rec["t"] = []
    t = 0.0
    for k in range(n_ctrl):
        tc = k * DTC
        x = math.sin(w * tc)
        if shape == "sine":
            cmd = T0 * (1.0 + amp * x)
        elif shape == "square":
            cmd = T0 + D * (1.0 if x >= 0.0 else -1.0)
        else:
            cmd = T0
        for _ in range(SUB):
            s0 = s
            s = pl.step(s, t, cmd, 0.0, DT)
            rec["t"].append(t)
            rec["cmd"].append(cmd)
            rec["thr"].append(s0[12])
            rec["fx"].append(pl.last_jet_force[0])
            rec["u"].append(s0[6])
            rec["pitch_deg"].append(math.degrees(s0[4]))
            rec["heave"].append(s0[2])
            rec["w"].append(s0[8])
            rec["a_cg_g"].append(pl.last_cg_acc / G)
            rec["a_bow_g"].append(pl.last_bow_acc / G)
            rec["rel_bow"].append(pl.last_rel_bow)
            rec["prime"].append(pl.prop.prime)
            t += DT
        if not np.all(np.isfinite(s)):
            break
    a = {q: np.asarray(v, float) for q, v in rec.items()}
    sel = (a["t"] >= settle) & (a["t"] < settle + win - 1e-9)
    tt = a["t"][sel]
    res = dict(U_kn=U_kn, P=P, shape=shape, amp=amp, T0=T0, D=D,
               settle=settle, win=win, finite=bool(np.all(np.isfinite(s))))
    for q in QUANT:
        y = a[q][sel]
        A, ph = harmonic(tt, y, w)
        res[q] = dict(amp=A, phase_deg=math.degrees(ph),
                      mean=float(y.mean()),
                      p2p=float(y.max() - y.min()),
                      std=float(y.std()))
    # phase relative to the command's fundamental (negative: lags it)
    pc = res["cmd"]["phase_deg"]
    for q in QUANT:
        d = res[q]["phase_deg"] - pc
        res[q]["phase_rel_cmd_deg"] = float((d + 180.0) % 360.0 - 180.0)
    return res


def phase_calm(speeds=(20.0, 26.0), periods=(1.5, 2.0, 2.5, 3.0, 5.0, 10.0,
                                              20.0)):
    out = load("calm.json") or {}
    for U in speeds:
        # an unmodulated reference: how steady 'steady' is
        key = f"{U:g}_const"
        if key not in out:
            out[key] = run_calm(U, 5.0, "const")
            save(out, "calm.json")
        for shape in ("sine", "square"):
            for P in periods:
                key = f"{U:g}_{shape}_{P:g}"
                if key in out:
                    continue
                t0 = time.time()
                r = run_calm(U, P, shape)
                out[key] = r
                save(out, "calm.json")
                log(f"calm {key}: cmd {r['cmd']['amp']:.0f} N  thr "
                    f"{r['thr']['amp']:.0f}  fx {r['fx']['amp']:.0f}  u "
                    f"{r['u']['amp']:.3f} m/s  pitch "
                    f"{r['pitch_deg']['amp']:.3f} deg  heave "
                    f"{r['heave']['amp'] * 100:.2f} cm  a_cg "
                    f"{r['a_cg_g']['amp']:.4f} g (p2p "
                    f"{r['a_cg_g']['p2p']:.4f})  [{time.time() - t0:.0f} s]")
    return out


# ------------------------------------------------------------ sea runs
class Hook:
    """Mission residual hook (state unchanged): per plant step what the
    plant measured from the step's START state (its last_* fields), the
    pump's prime used in the step, the jet force, the slamming force."""

    def __init__(self):
        self.fx, self.prime, self.slam, self.acg, self.abow = \
            [], [], [], [], []

    def __call__(self, s, dt, plant):
        self.fx.append(float(plant.last_jet_force[0]))
        self.prime.append(float(plant.prop.prime))
        self.slam.append(float(plant.last_slam_force))
        self.acg.append(float(plant.last_cg_acc))
        self.abow.append(float(plant.last_bow_acc))
        return s


def enc_omega(sea, U, psi):
    """Mean (zero-crossing) encounter frequency of the sea's components at
    speed U on heading psi: sqrt(sum a^2 we^2 / sum a^2)."""
    a = np.asarray(sea.a, float).ravel()
    w = np.asarray(sea.w, float).ravel()
    k = np.asarray(sea.k, float).ravel()
    th = np.asarray(sea.th, float).ravel()
    we = w - k * U * np.cos(th - psi)
    return float(np.sqrt(np.sum(a * a * we * we) / np.sum(a * a)))


def cg_wave(pl, s, t):
    """(elevation, its rate along the boat's path) at the CG."""
    e, et, _, ex, ey = pl._waves(np.array([s[0]]), np.array([s[1]]), t)
    c, sn = math.cos(s[5]), math.sin(s[5])
    xd = s[6] * c - s[7] * sn
    yd = s[6] * sn + s[7] * c
    return float(e[0]), float(et[0] + xd * ex[0] + yd * ey[0])


def const_policy(frac):
    def pol(m, k, st):
        return frac * m.t_max
    pol.desc = dict(kind="const", frac=frac)
    return pol


def lock_policy(delta_deg, hi, lo):
    """Oracle: phase phi of the encountered elevation at the CG, eta = A
    cos(phi), eta' = -A we sin(phi); thrust hi while sin(phi + delta) < 0
    (delta 0: hi exactly while the water under the CG rises -- climbing
    the face -- and lo from the crest to the trough), else lo."""
    dl = math.radians(delta_deg)

    def pol(m, k, st):
        e, ed = cg_wave(m.plant, m.s, m.t)
        we = enc_omega(m.sea, float(m.s[6]), float(m.s[5]))
        ph = math.atan2(-ed / we, e)
        st.setdefault("phase", []).append(ph)
        return (hi if math.sin(ph + dl) < 0.0 else lo) * m.t_max
    pol.desc = dict(kind="lock", delta_deg=delta_deg, hi=hi, lo=lo)
    return pol


def run_sea(seed, leg, pol, T=120.0):
    """One evaluation mission (final_eval.make_mission, gz plant, record)
    with the thrust from pol and the heading autopilot on the nozzle."""
    from studies import final_eval as FE
    from studies import safety_events as SE
    from learn.repro import task
    m = FE.make_mission(seed, leg, T)
    pl = m.plant
    hook = Hook()
    m.residual = hook
    st = {}
    cmds = []
    while not m.done():
        thr = float(pol(m, m.k, st))
        rud = m.ep._steer(m.s, thr)
        cmds.append(thr)
        m.advance(thr, rud)
        if not m.finite or getattr(pl, "capsized", False):
            break
    met = m.metrics()
    tr = pl.trajectory()
    ev = SE.safety_events(tr, k=2.0, k_report=(), track=m.phi)
    t = tr["t"]
    s = tr["s"]
    n = len(t)
    fx = np.asarray(hook.fx[:n])
    prime = np.asarray(hook.prime[:n])
    slam = np.asarray(hook.slam[:n])
    # per row: CG elevation and its rate along the path, wet stations
    eta = np.zeros(n)
    etad = np.zeros(n)
    nwet = np.zeros(n, int)
    dbow = np.zeros(n)
    x = pl.x
    for i in range(n):
        si = s[i]
        eta[i], etad[i] = cg_wave(pl, si, t[i])
        c, sn = math.cos(si[5]), math.sin(si[5])
        e = pl._waves(si[0] + x * c, si[1] + x * sn, t[i])[0]
        d = e - (pl.Z_rest + si[2] + pl.zk + (-1.0) * x
                 * (pl.Th_rest + si[4]))
        nwet[i] = int((d > 0).sum())
        dbow[i] = d[-1]
    sel = t >= T_SKIP
    phi = m.phi
    dpsi = s[:, 5] - phi
    ua = s[:, 6] * np.cos(dpsi) - s[:, 7] * np.sin(dpsi)
    U = float(ua[sel].mean())
    we_nom = enc_omega(m.sea, float(s[sel, 6].mean()), phi)
    # measured encounter frequency of the CG elevation: sqrt(m2/m0)
    we_meas = float(np.std(etad[sel]) / np.std(eta[sel]))
    idx, pk = SE.acc_peaks(tr["a_cg"], DT, 2.0)
    keep = t[idx] >= T_SKIP
    idx, pk = idx[keep], pk[keep]
    hours = float(sel.sum() * DT / 3600.0)

    def spread(y):
        y = np.asarray(y, float)[sel]
        return dict(mean=float(y.mean()), std=float(y.std()),
                    amp_eq=float(np.sqrt(2.0) * y.std()))

    def band(y, lo=0.6, hi=1.6):
        """std of y in the band [lo, hi] x we_meas (FFT, detrended)."""
        y = np.asarray(y, float)[sel]
        y = y - y.mean()
        F = np.fft.rfft(y)
        f = np.fft.rfftfreq(y.size, DT) * 2 * np.pi
        msk = (f >= lo * we_meas) & (f <= hi * we_meas)
        return float(np.sqrt(2.0 * np.sum(np.abs(F[msk]) ** 2)) / y.size)

    def pk_freq(y):
        y = np.asarray(y, float)[sel]
        y = y - y.mean()
        F = np.abs(np.fft.rfft(y * np.hanning(y.size))) ** 2
        f = np.fft.rfftfreq(y.size, DT) * 2 * np.pi
        Fs = np.convolve(F, np.ones(9) / 9.0, "same")
        msk = f > 0.5
        return float(f[msk][np.argmax(Fs[msk])])

    hb = SE.bow_height(tr)
    wet_runs = SE._runs((hb < 0.0) & sel)
    sub = np.asarray(tr["submergence"])
    em_runs = SE._runs((sub < 0.0) & sel)
    air_runs = SE._runs((nwet == 0) & sel)
    summ = dict(
        seed=seed, leg=leg, policy=pol.desc, finite=bool(m.finite),
        capsized=bool(getattr(pl, "capsized", False)),
        kn=U / KN, kn_whole=met["u_along"] / KN if met.get("finite")
        else float("nan"),
        cmd_mean_frac=float(np.mean(cmds) / m.t_max),
        thr_mean_frac=float(s[sel, 12].mean() / m.t_max),
        fx_mean=float(fx[sel].mean()),
        we_nom=we_nom, we_meas=we_meas, Te_meas=2 * np.pi / we_meas,
        we_pitch_peak=pk_freq(s[:, 4]),
        u=spread(s[:, 6]), pitch_deg=spread(np.degrees(s[:, 4])),
        heave=spread(s[:, 2]), w=spread(s[:, 8]),
        a_cg_g=spread(tr["a_cg"] / G), a_bow_g=spread(tr["a_bow"] / G),
        fx=spread(fx), eta_cg=spread(eta),
        band=dict(u=band(s[:, 6]), pitch_deg=band(np.degrees(s[:, 4])),
                  heave=band(s[:, 2]), a_cg_g=band(tr["a_cg"] / G),
                  fx=band(fx), eta_cg=band(eta)),
        hours=hours,
        n_peaks=int(pk.size), A110=SE.a_1_10(pk),
        n_ge3=int((pk >= 3).sum()), n_ge5=int((pk >= 5).sum()),
        n_ge7=int((pk >= 7).sum()), n_ge10=int((pk >= 10).sum()),
        peak_max=float(pk.max(initial=0.0)),
        ge7_per_h=float((pk >= 7).sum() / hours),
        bow_wet=len(wet_runs), bow_wet_per_h=len(wet_runs) / hours,
        intake_out=len(em_runs), intake_out_per_h=len(em_runs) / hours,
        intake_out_share=float((sub[sel] < 0).mean()),
        prime_mean=float(prime[sel].mean()),
        prime_lt_half_share=float((prime[sel] < 0.5).mean()),
        airborne=len(air_runs), airborne_per_h=len(air_runs) / hours,
        airborne_share=float((nwet[sel] == 0).mean()),
        bow_out_share=float((dbow[sel] <= 0).mean()),
        events_acc=ev["acc"][2.0])
    # ---- every >= 7 g peak (k = 2), and a sample of 3-7 g ones: what it is
    raw = tr["a_cg"]
    pk_info = []
    for j, (i, v) in enumerate(zip(idx, pk)):
        if v < 3.0:
            continue
        i0 = max(i - 6, 0)
        ir = i0 + int(np.argmax(raw[i0:i + 1]))      # the raw maximum
        back = int(round(1.5 / DT))
        lo_ = max(ir - back, 0)
        win_nwet = nwet[lo_:ir + 1]
        win_dbow = dbow[lo_:ir + 1]
        dry = np.nonzero(win_nwet == 0)[0]
        bdry = np.nonzero(win_dbow <= 0)[0]
        # the plant's derivative at the recorded start state of the row
        pl.prop.prime = float(prime[ir])
        _, aux = pl.deriv(s[ir], t[ir])
        mom = aux["mom"] * pl.dx
        msum = float(mom.sum())
        xm = float((mom * x).sum() / msum) if msum > 0 else float("nan")
        ent = aux["V"] > 0
        wet_ = aux["d"] > 0
        e0, ed0 = eta[ir], etad[ir]
        ph = math.atan2(-ed0 / we_meas, e0)
        pk_info.append(dict(
            t=float(t[ir]), g_k2=float(v), a_cg_raw_g=float(raw[ir] / G),
            airborne_before=bool(dry.size > 0),
            t_since_airborne=(float((ir - lo_ - dry[-1]) * DT)
                              if dry.size else None),
            airborne_dur=float((win_nwet == 0).sum() * DT),
            bow_out_before=bool(bdry.size > 0),
            t_since_bow_out=(float((ir - lo_ - bdry[-1]) * DT)
                             if bdry.size else None),
            nwet=int(nwet[ir]), slam_over_W=msum / (pl.m * G),
            x_slam=xm, x_bow=float(x[-1]),
            V_bow=float(aux["V"][-1]), V_max=float(aux["V"][wet_].max())
            if wet_.any() else float("nan"),
            n_entering=int((ent & wet_).sum()),
            pitch_deg=float(math.degrees(s[ir, 4] + pl.Th_rest)),
            q_deg_s=float(math.degrees(s[ir, 10])),
            w=float(s[ir, 8]), u=float(s[ir, 6]),
            eta_cg=float(e0), etad_cg=float(ed0),
            wave_phase_deg=math.degrees(ph),
            prime=float(prime[ir]), fx=float(fx[ir])))
    summ["peaks"] = pk_info
    series = None
    if pol.desc["kind"] == "const" or pol.desc.get("keep"):
        f32 = (lambda y: np.asarray(y, np.float32))
        series = dict(t=f32(t), u=f32(s[:, 6]), pitch=f32(s[:, 4]),
                      heave=f32(s[:, 2]), w=f32(s[:, 8]), q=f32(s[:, 10]),
                      thr=f32(s[:, 12]), a_cg=f32(raw), fx=f32(fx),
                      prime=f32(prime), slam=f32(slam), eta=f32(eta),
                      etad=f32(etad), nwet=nwet.astype(np.int8),
                      dbow=f32(dbow), sub=f32(sub), x0=f32(s[:, 0]),
                      y0=f32(s[:, 1]), psi=f32(s[:, 5]))
    return summ, series


def parse_range(s):
    out = []
    for part in str(s).split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _sea_loop(name, policies, seeds, legs, keep_series=False):
    res = load(name) or {}
    ser_name = name.replace(".pkl", "_series.pkl")
    ser = (load(ser_name) or {}) if keep_series else None
    for pkey, pol in policies:
        for leg in legs:
            for seed in seeds:
                key = f"{pkey}|s{seed}|l{leg}"
                if key in res:
                    continue
                g = free_gb()
                if g < 1.5:
                    log(f"  only {g:.1f} GB free: waiting")
                    while free_gb() < 1.5:
                        time.sleep(30)
                t0 = time.time()
                r, srs = run_sea(seed, leg, pol)
                res[key] = r
                save(res, name)
                if keep_series and srs is not None:
                    ser[key] = srs
                    save(ser, ser_name)
                log(f"{name} {key}: {r['kn']:.1f} kn  pitch sd "
                    f"{r['pitch_deg']['std']:.2f} deg  heave sd "
                    f"{r['heave']['std'] * 100:.1f} cm  a_cg sd "
                    f"{r['a_cg_g']['std']:.2f} g  A1/10 {r['A110']:.2f} g  "
                    f">=7g {r['n_ge7']}  Te {r['Te_meas']:.2f} s  prime "
                    f"{r['prime_mean']:.2f}  air {r['airborne_share']:.2f}"
                    f"  [{time.time() - t0:.0f} s]")
    return res


FRACS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
DELTAS = (0, 45, 90, 135, 180, 225, 270, 315)


def phase_sea(seeds, legs):
    pols = [(f"const{f:.1f}", const_policy(f)) for f in FRACS]
    return _sea_loop("sea.pkl", pols, seeds, legs, keep_series=True)


def phase_lock(seeds, legs, pairs=((1.0, 0.4),)):
    pols = []
    for hi, lo in pairs:
        for dl in DELTAS:
            pols.append((f"lock{hi:.1f}_{lo:.1f}_{dl}",
                         lock_policy(dl, hi, lo)))
    return _sea_loop("lock.pkl", pols, seeds, legs)


# ------------------------------------------------------------ spectra
def phase_spectra():
    """From the constant-thrust series: Welch spectra (20.48 s segments) of
    pitch, heave, CG acceleration and surge speed; the pitch spectrum's
    peak frequency (the dominant encounter frequency of the motion) and
    the std of each quantity in [0.5, 1.5] x that frequency, as an
    equivalent amplitude sqrt(2) std; next to it the calm-water response
    to the largest symmetric square-wave throttle swing at the nearest
    measured forcing period (calm.json, 20 kn for runs below 23 kn, else
    26 kn). And, over all >= 3 g peaks, the correlation of the peak with
    the surge speed at the peak, within each run (both de-meaned per run)
    and across runs (run means)."""
    from scipy.signal import welch
    ser = load("sea_series.pkl")
    sea = load("sea.pkl")
    calm = load("calm.json")
    lines = ["\n== spectra of the constant-thrust runs (t >= "
             f"{T_SKIP:g} s): pitch-peak period Te, band [0.5, 1.5] x "
             "1/Te, equivalent amplitude sqrt(2) x band std  |  calm "
             "square-wave throttle response at the nearest period"]
    rows = {}
    for key, s in ser.items():
        r = sea[key]
        sel = s["t"] >= T_SKIP
        sig = dict(pitch_deg=np.degrees(s["pitch"]), heave=s["heave"],
                   a_cg_g=s["a_cg"] / G, u=s["u"], w=s["w"])
        sp = {}
        for q, y in sig.items():
            y = np.asarray(y, float)[sel]
            f, P = welch(y - y.mean(), fs=1.0 / DT, nperseg=1024)
            sp[q] = (f, P)
        f, Pp = sp["pitch_deg"]
        msk = f > 0.15
        fp = float(f[msk][np.argmax(Pp[msk])])
        df = f[1] - f[0]
        band = (f >= 0.5 * fp) & (f <= 1.5 * fp)
        amp = {q: float(np.sqrt(2.0 * np.sum(P[band]) * df))
               for q, (f_, P) in sp.items()}
        rows[key] = dict(leg=r["leg"], frac=r["policy"]["frac"], kn=r["kn"],
                         Te=1.0 / fp, amp=amp)
    res = dict(rows=rows)
    for leg in (0, 1):
        fr = sorted({v["frac"] for v in rows.values() if v["leg"] == leg})
        for f_ in fr:
            rs = [v for v in rows.values()
                  if v["leg"] == leg and v["frac"] == f_]
            kn = np.mean([v["kn"] for v in rs])
            Te = np.mean([v["Te"] for v in rs])
            A = {q: np.mean([v["amp"][q] for v in rs]) for q in rs[0]["amp"]}
            U = 20 if kn < 23 else 26
            Ps = sorted({c["P"] for c in calm.values()
                         if c["shape"] == "square" and c["U_kn"] == U})
            Pn = min(Ps, key=lambda p: abs(p - Te))
            c = calm[f"{U:g}_square_{Pn:g}"]
            lines.append(
                f"  leg {leg} thrust {f_:.1f} {kn:5.1f} kn  Te {Te:.2f} s  "
                f"waves: pitch {A['pitch_deg']:.2f} deg, heave "
                f"{A['heave'] * 100:.1f} cm, a_cg {A['a_cg_g']:.3f} g, u "
                f"{A['u']:.3f} m/s  |  throttle ({U} kn, P {Pn:g} s, cmd "
                f"{c['cmd']['amp']:.0f} N): pitch "
                f"{c['pitch_deg']['amp']:.3f}, heave "
                f"{c['heave']['amp'] * 100:.2f} cm, a_cg "
                f"{c['a_cg_g']['amp']:.4f} g, u {c['u']['amp']:.3f}  |  "
                f"ratio pitch {c['pitch_deg']['amp'] / A['pitch_deg']:.3f}"
                f" heave {c['heave']['amp'] / A['heave']:.3f} a_cg "
                f"{c['a_cg_g']['amp'] / A['a_cg_g']:.3f} u "
                f"{c['u']['amp'] / A['u']:.2f}")
    # peaks vs the speed at the peak
    g, u, run = [], [], []
    for key, r in sea.items():
        for p in r["peaks"]:
            g.append(p["g_k2"])
            u.append(p["u"])
            run.append(key)
    g, u, run = np.array(g), np.array(u), np.array(run)
    gd, ud = g.copy(), u.copy()
    for k in set(run):
        m = run == k
        gd[m] -= g[m].mean()
        ud[m] -= u[m].mean()
    within = float(np.corrcoef(gd, ud)[0, 1])
    slope_w = float(np.sum(gd * ud) / np.sum(ud * ud))
    keys = sorted(set(run))
    gm = np.array([g[run == k].mean() for k in keys])
    um = np.array([u[run == k].mean() for k in keys])
    across = float(np.corrcoef(gm, um)[0, 1])
    res["peaks_u"] = dict(n=int(g.size), within_r=within,
                          within_slope_g_per_ms=slope_w,
                          sd_u_within=float(ud.std()), across_r=across)
    lines.append(
        f"  >= 3 g peaks: n {g.size}; speed at the peak, within-run sd "
        f"{ud.std():.2f} m/s; correlation peak vs speed within runs "
        f"{within:+.2f} (slope {slope_w:+.2f} g per m/s), across run "
        f"means {across:+.2f}")
    # the take-off before each >= 3 g peak that followed a flight: the
    # last row with some station wet before the airborne spell
    to = {q: [] for q in ("g", "run", "u", "w", "q", "pitch", "air",
                          "fx", "prime")}
    for key, r in sea.items():
        s = ser.get(key)
        if s is None:
            continue
        nw = np.asarray(s["nwet"])
        for p in r["peaks"]:
            i = int(round(p["t"] / DT))
            j = i
            while j > 0 and nw[j] > 0 and i - j < int(1.5 / DT):
                j -= 1
            if nw[j] > 0:
                continue                  # no flight in the last 1.5 s
            ja = j
            while ja > 0 and nw[ja] == 0:
                ja -= 1
            to["g"].append(p["g_k2"])
            to["run"].append(key)
            to["u"].append(float(s["u"][ja]))
            to["w"].append(float(s["w"][ja]))
            to["q"].append(float(np.degrees(s["q"][ja])))
            to["pitch"].append(float(np.degrees(s["pitch"][ja])))
            to["air"].append((j - ja) * DT)
            to["fx"].append(float(s["fx"][ja]))
            to["prime"].append(float(s["prime"][ja]))
    A = {q: np.asarray(v) for q, v in to.items()}
    if A["g"].size > 10:
        dm = {}
        for q in ("g", "u", "w", "q", "pitch", "air"):
            x = A[q].astype(float).copy()
            for k in set(A["run"]):
                m = A["run"] == k
                x[m] -= x[m].mean()
            dm[q] = x
        cors = {q: float(np.corrcoef(dm["g"], dm[q])[0, 1])
                for q in ("u", "w", "q", "pitch", "air")}
        res["takeoff"] = dict(n=int(A["g"].size), within_r=cors,
                              sd_within={q: float(dm[q].std())
                                         for q in dm},
                              prime_median=float(np.median(A["prime"])),
                              air_median=float(np.median(A["air"])))
        lines.append(
            f"  take-off before the >= 3 g peaks that followed a flight: n "
            f"{A['g'].size}; within-run correlation of the peak with the "
            f"take-off value: " + ", ".join(
                f"{q} {c:+.2f} (sd {dm[q].std():.2f})"
                for q, c in cors.items())
            + f"; flight median {np.median(A['air']):.2f} s; prime at "
            f"take-off median {np.median(A['prime']):.2f}")
    save(res, "spectra.json")
    return lines


# ------------------------------------------------------------ report
def _interp_const(sea, seed, leg, kn, field, exclude=None):
    """The constant-thrust runs of this seed and leg, metric `field`
    against mean speed, read at speed kn (linear between the two nearest
    measured thrust levels; None outside their range). exclude: a thrust
    level left out (the null reference: a constant run against its
    neighbours)."""
    pts = []
    for f in FRACS:
        if exclude is not None and abs(f - exclude) < 1e-9:
            continue
        r = sea.get(f"const{f:.1f}|s{seed}|l{leg}")
        if r is not None and r["finite"]:
            pts.append((r["kn"], field(r)))
    if len(pts) < 2:
        return None
    pts.sort()
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    if kn < xs[0] or kn > xs[-1]:
        return None
    return float(np.interp(kn, xs, ys))


def phase_report():
    lines = []
    cfg = load("config.json")
    if cfg:
        lines.append("== propulsion chain and calm water (config.json)")
        for k in ("t_max_N", "t_min_N", "tau_s", "rate_max_N_per_s",
                  "dead_band_N", "delay_s", "prime_band_m",
                  "tau_prime_loss_s", "tau_prime_back_s",
                  "intake_x_from_cg", "intake_z_from_cg",
                  "nozzle_x_from_cg", "nozzle_z_from_cg", "surge_mass",
                  "Iy", "top_speed_kn"):
            lines.append(f"  {k:<22} {cfg[k]:.4g}")
        for r in cfg["speeds"]:
            lines.append(
                f"  {r['kn']:>3} kn  R {r['R_N']:6.0f} N  trim "
                f"{r['trim_deg']:5.2f}  rise {r['rise_m']:.3f}  hold cmd "
                f"{r['cmd_hold_frac']:.3f} t_max  full-power net "
                f"{r['net_full_power_N']:6.0f} N  dR/du {r['dR_du']:5.0f}"
                f"  dF/du {r['dF_du_at_cmd']:6.0f}  T_surge "
                f"{r['T_surge_s']:.1f} s")
    calm = load("calm.json")
    if calm:
        lines.append("\n== calm water: response to the thrust command "
                     "(amplitude of the fundamental; phase vs command)")
        hdr = ("  U   shape   P    cmd N   thr N   fx N   u m/s  "
               "pitch deg  heave cm  a_cg g  a_bow g  (p2p a_cg g)  "
               "ph(u) ph(pitch)")
        lines.append(hdr)
        for key, r in calm.items():
            if r["shape"] == "const":
                lines.append(
                    f"  {r['U_kn']:g} const: u p2p {r['u']['p2p']:.4f} m/s,"
                    f" pitch p2p {r['pitch_deg']['p2p']:.4f} deg, a_cg p2p"
                    f" {r['a_cg_g']['p2p']:.5f} g")
                continue
            lines.append(
                f"  {r['U_kn']:g} {r['shape']:<6} {r['P']:>4g} "
                f"{r['cmd']['amp']:7.0f} {r['thr']['amp']:7.0f} "
                f"{r['fx']['amp']:6.0f} {r['u']['amp']:6.3f} "
                f"{r['pitch_deg']['amp']:9.4f} "
                f"{r['heave']['amp'] * 100:8.3f} "
                f"{r['a_cg_g']['amp']:7.4f} {r['a_bow_g']['amp']:7.4f}  "
                f"({r['a_cg_g']['p2p']:.4f})  "
                f"{r['u']['phase_rel_cmd_deg']:6.0f} "
                f"{r['pitch_deg']['phase_rel_cmd_deg']:6.0f}")
    sea = load("sea.pkl")
    if sea:
        lines.append("\n== evaluation sea, constant thrust (t >= "
                     f"{T_SKIP:g} s; per leg, mean over seeds)")
        for leg in (0, 1):
            for f in FRACS:
                rs = [r for k, r in sea.items()
                      if r["leg"] == leg and r["policy"]["frac"] == f]
                if not rs:
                    continue
                mean = (lambda fn: float(np.mean([fn(r) for r in rs])))
                lines.append(
                    f"  leg {leg} thrust {f:.1f}: n {len(rs)}  "
                    f"{mean(lambda r: r['kn']):5.1f} kn  Te "
                    f"{mean(lambda r: r['Te_meas']):.2f} s (nominal "
                    f"{mean(lambda r: 2 * np.pi / r['we_nom']):.2f})  "
                    f"sd: u {mean(lambda r: r['u']['std']):.3f} m/s, pitch "
                    f"{mean(lambda r: r['pitch_deg']['std']):.2f} deg, heave "
                    f"{mean(lambda r: r['heave']['std']) * 100:.1f} cm, a_cg "
                    f"{mean(lambda r: r['a_cg_g']['std']):.3f} g, fx "
                    f"{mean(lambda r: r['fx']['std']):.0f} N | band sd: u "
                    f"{mean(lambda r: r['band']['u']):.3f}, pitch "
                    f"{mean(lambda r: r['band']['pitch_deg']):.2f}, heave "
                    f"{mean(lambda r: r['band']['heave']) * 100:.1f} | A1/10 "
                    f"{mean(lambda r: r['A110']):.2f} g  >=7g/h "
                    f"{mean(lambda r: r['ge7_per_h']):.0f}  bow wet/h "
                    f"{mean(lambda r: r['bow_wet_per_h']):.0f}  intake out "
                    f"{mean(lambda r: r['intake_out_share']):.2f} of time, "
                    f"prime<0.5 {mean(lambda r: r['prime_lt_half_share']):.2f}"
                    f", airborne {mean(lambda r: r['airborne_share']):.3f}")
        # the >= 7 g peaks
        allp = [(k, p) for k, r in sea.items() for p in r["peaks"]
                if p["g_k2"] >= 7.0]
        if allp:
            P_ = [p for _, p in allp]
            arr = (lambda fn: np.array([fn(p) for p in P_], float))
            lines.append(
                f"\n== the >= 7 g peaks (k 2) of the constant-thrust runs: "
                f"{len(P_)} peaks")
            lines.append(
                f"  hull fully airborne within 1.5 s before: "
                f"{np.mean(arr(lambda p: p['airborne_before'])):.2f}; bow "
                f"station out within 1.5 s: "
                f"{np.mean(arr(lambda p: p['bow_out_before'])):.2f}")
            ta = arr(lambda p: p["t_since_airborne"]
                     if p["t_since_airborne"] is not None else np.nan)
            tb = arr(lambda p: p["t_since_bow_out"]
                     if p["t_since_bow_out"] is not None else np.nan)
            lines.append(
                f"  time since last fully airborne (median, of those): "
                f"{np.nanmedian(ta):.2f} s; since bow out: "
                f"{np.nanmedian(tb):.2f} s; airborne duration median "
                f"{np.median(arr(lambda p: p['airborne_dur'])):.2f} s")
            lines.append(
                f"  slamming (Wagner momentum) force / weight: median "
                f"{np.median(arr(lambda p: p['slam_over_W'])):.2f}; its "
                f"centre x from CG median "
                f"{np.nanmedian(arr(lambda p: p['x_slam'])):.2f} m (bow at "
                f"{P_[0]['x_bow']:.2f}); pitch at peak median "
                f"{np.median(arr(lambda p: p['pitch_deg'])):.1f} deg; w "
                f"median {np.median(arr(lambda p: p['w'])):.2f} m/s; "
                f"V_max median {np.nanmedian(arr(lambda p: p['V_max'])):.2f}"
                f" m/s; prime median "
                f"{np.median(arr(lambda p: p['prime'])):.2f}")
            ph = arr(lambda p: p["wave_phase_deg"])
            hist, edges = np.histogram(ph, bins=8, range=(-180, 180))
            lines.append("  wave phase at the CG at the peak (0 crest, "
                         "+-180 trough, -90 rising face, +90 falling back):")
            lines.append("   " + "  ".join(
                f"[{edges[i]:.0f},{edges[i + 1]:.0f}) {hist[i]}"
                for i in range(8)))
    if sea and calm and os.path.exists(os.path.join(OUT,
                                                     "sea_series.pkl")):
        lines += phase_spectra()
    lock = load("lock.pkl")
    if lock and sea:
        lines.append("\n== oracle phase-locked bang-bang vs constant thrust "
                     "at the same mean speed (same seed and leg; constant "
                     "runs linearly interpolated in speed); diff = lock - "
                     "const, mean over seeds")
        fields = dict(
            A110=lambda r: r["A110"], ge7=lambda r: r["n_ge7"],
            ge5=lambda r: r["n_ge5"], pitch_sd=lambda r: r["pitch_deg"]["std"],
            acg_sd=lambda r: r["a_cg_g"]["std"],
            bow_wet=lambda r: r["bow_wet"], air=lambda r: r["airborne_share"])
        # the null reference: each interior constant level against the
        # interpolation of its two neighbours -- what 'diff' looks like
        # for a policy with no per-wave timing at all
        for leg in (0, 1):
            nd = {f: [] for f in fields}
            for f_ in FRACS[1:-1]:
                for k, r in sea.items():
                    if r["leg"] != leg or r["policy"]["frac"] != f_:
                        continue
                    for f, fn in fields.items():
                        c = _interp_const(sea, r["seed"], leg, r["kn"], fn,
                                          exclude=f_)
                        if c is not None:
                            nd[f].append(fn(r) - c)
            lines.append(f"  leg {leg} NULL (const vs neighbours), sd of "
                         f"one paired diff: " + "  ".join(
                             f"{f} {np.mean(v):+.3f} sd {np.std(v):.3f}"
                             f"(n{len(v)})" for f, v in nd.items() if v))
        keys = sorted({k.split("|")[0] for k in lock})
        pooled = {}
        for leg in (0, 1):
            for pk_ in keys:
                rs = [r for k, r in lock.items()
                      if k.startswith(pk_ + "|") and r["leg"] == leg]
                if not rs:
                    continue
                diffs = {f: [] for f in fields}
                for r in rs:
                    for f, fn in fields.items():
                        c = _interp_const(sea, r["seed"], leg, r["kn"], fn)
                        if c is not None:
                            diffs[f].append(fn(r) - c)
                kn = np.mean([r["kn"] for r in rs])
                pooled.setdefault(pk_.rsplit("_", 1)[0], []).append(
                    np.array(diffs["A110"]))
                parts = []
                for f in fields:
                    d = np.array(diffs[f])
                    if d.size:
                        se = d.std(ddof=1) / np.sqrt(d.size) if d.size > 1 \
                            else float("nan")
                        parts.append(f"{f} {d.mean():+.3f}+-{se:.3f}"
                                     f"(n{d.size})")
                lines.append(f"  leg {leg} {pk_:<18} {kn:5.1f} kn  "
                             + "  ".join(parts))
        for pr, ds in pooled.items():
            d = np.concatenate(ds)
            lines.append(f"  {pr}: A1/10 diff over every phase, leg and "
                         f"seed: mean {d.mean():+.3f} sd {d.std():.3f} "
                         f"(n {d.size}); best phase-leg cell mean "
                         f"{min(x.mean() for x in ds if x.size):+.3f}, "
                         f"worst {max(x.mean() for x in ds if x.size):+.3f}")
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="config,calm")
    ap.add_argument("--seeds", default="0-3")
    ap.add_argument("--legs", default="0,1")
    ap.add_argument("--pairs", default="1.0:0.4",
                    help="lock: hi:lo thrust fractions, comma separated")
    a = ap.parse_args()
    seeds, legs = parse_range(a.seeds), parse_range(a.legs)
    log(f"start {a.phase} (free {free_gb():.1f} GB)")
    for ph in a.phase.split(","):
        if ph == "config":
            phase_config()
        elif ph == "calm":
            phase_calm()
        elif ph == "sea":
            phase_sea(seeds, legs)
        elif ph == "lock":
            pairs = [tuple(float(x) for x in p.split(":"))
                     for p in a.pairs.split(",")]
            phase_lock(seeds, legs, pairs)
        elif ph == "report":
            phase_report()
        else:
            raise SystemExit(f"unknown phase {ph}")


if __name__ == "__main__":
    main()
