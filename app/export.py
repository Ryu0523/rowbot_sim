"""
Exports of a Record for the two viewers.

viewer_json(rec, path)    viewer/demo.html (and viewer/proto_web.html) data:
                          the format of studies/visual_export_web.py --
                          meta, sea (rebased phases), hull (the plant's own
                          stations), frames at every plant step -- plus
                          the track, the policy and the commands
blender_scene(rec, path)  viewer/proto_blender/render.py input: the format
                          of studies/visual_export.py -- frames at the video
                          rate, the past track for the wake, plant steps
                          with keel immersion d and immersion rate V
blender_command(scene, out, clip=(a, b), mp4=...)  the render.py command
                          line: a still, or frames a..b as PNGs plus an MP4
                          (render.py encodes it with Blender's own ffmpeg);
                          find_blender() locates blender.exe (BLENDER, PATH,
                          the Blender Foundation folder)

Both exports draw the hull from the CURRENT code for the record's hull key;
when the record's hull sha1 or physics stamp differ from it
(record.code_changes) the viewer meta lists the differences in
"code_changes" and the command line prints them.

Both take the exact sea of the run (the record's components) and the
hull's geometry from its own plant class (sim/physics.gz_class), so the
pictures show the waves and the attitude the plant integrated.

REBASING (viewer_json, exact): positions are written relative to the start
(X0, Y0) and every component phase is shifted to keep the sum identical,
phi' = phi + k (X0 cos th + Y0 sin th) (times start at 0 already), so the
page's float32 shaders never see large numbers.
"""
import glob
import json
import math
import os
import shutil
from types import SimpleNamespace

import numpy as np

from . import hulls
from .scenario import Scenario

G = 9.81
KN = 0.514444


def sea_of(header):
    sd = header["sea"]
    return SimpleNamespace(**{k: np.asarray(sd[k], float)
                              for k in ("a", "k", "th", "w", "phi")})


def sea_eval(sea, X, Y, t):
    X, Y, t = (np.asarray(v, float) for v in (X, Y, t))
    ph = (sea.k * (X[..., None] * np.cos(sea.th) + Y[..., None]
                   * np.sin(sea.th)) - sea.w * t[..., None] + sea.phi)
    return (sea.a * np.cos(ph)).sum(-1)


def geometry_plant(header):
    """The hull's GZ-class plant on the record's sea (geometry only)."""
    from sim import physics
    sc = Scenario(header["scenario"])
    h = hulls.resolve(sc.hull_spec)
    return h, physics.gz_class(h)(h, sea_of(header), dt=header["dt"])


# the drawn chine and keel: linear between the exported points, within
# LINE_TOL of hydro/hull_geometry everywhere (line_points); the water
# shader's table: at most N_TAB points, within the smallest tolerance that
# allows (line_table)
LINE_TOL = 2.5e-4
N_BASE = 101
N_TAB = 64
# the viewer JSON's format: 2 = hull lines at line_points + the shader table
# x_tab / yc_tab + meta.server_code_stale (the server's viewer cache files
# are named by it)
VIEWER_FORMAT = 2


def _lines(g):
    return lambda x: np.stack([g.chine(x), g.keel(x)[0]])


def line_points(g, tol=LINE_TOL, n_base=N_BASE, n_sub=16):
    """x (from the transom, m) at which the chine and the keel of the
    geometry `g` (hydro/hull_geometry Stations) are sampled so that the
    polyline through the samples is within `tol` of both everywhere:
    n_base uniform points (and the offsets' knots), then every interval
    whose chord leaves the curves by more than `tol` at any of n_sub - 1
    interior points is halved, until none does (or an interval is 1e-9 L
    wide). General for any form -- the elliptic closure at the stem (the
    chine goes like the square root of the distance to the stem: a uniform
    201-point table was 43-46 mm off there), a transom run's kink, an
    offsets table's knots -- and deterministic: the same geometry gives the
    same points to the bit."""
    L = float(g.L)
    f = _lines(g)
    x = np.linspace(0.0, L, int(n_base))
    kn = np.asarray(getattr(g, "knots", ()), float)
    if kn.size:
        x = np.union1d(x, kn[(kn > 0.0) & (kn < L)])
    t = np.arange(1, n_sub) / n_sub
    for _ in range(80):
        a, b = x[:-1], x[1:]
        xs = a[:, None] + (b - a)[:, None] * t[None]
        F = f(xs.ravel()).reshape(2, *xs.shape)
        Fa, Fb = f(a), f(b)
        lin = Fa[..., None] + (Fb - Fa)[..., None] * t
        err = np.abs(F - lin).max(axis=(0, 2))
        bad = (err > tol) & ((b - a) > 1e-9 * L)
        if not bad.any():
            break
        x = np.union1d(x, 0.5 * (a[bad] + b[bad]))
    return x


def line_table(g, n_max=N_TAB):
    """(x, tol): at most n_max points for the water shader's chine table,
    line_points from a 9-point base with the smallest tolerance (from
    LINE_TOL up by factors of 1.25) that needs no more."""
    tol = LINE_TOL
    while True:
        x = line_points(g, tol, n_base=9)
        if len(x) <= n_max:
            return x, tol
        tol *= 1.25


def hull_dict(h, p):
    """The hull as the viewers draw it: the plant's station arrays (x, yc,
    zk at the strip centres, the GZ sections' chine and deck heights) and
    the chine and keel of hydro/hull_geometry -- the module the plants
    read -- at the points line_points chooses (x_fine, yc_fine, zk_fine:
    linear between them, within LINE_TOL of the module everywhere, the
    stem included), plus the water shader's chine table (x_tab, yc_tab, at
    most N_TAB points). The viewers draw from these and never rebuild the
    lines from the hull fields taper / stem_rise, which describe v0's
    family member only (wrong for a transom run, a raked keel or an
    offsets hull: planing_b). cache_tag is sim/physics.cache_tag(h) (None:
    the legacy scarab195 on v0), the key the viewer caches are kept
    under."""
    from hydro.hull_geometry import stations_of
    from sim import physics
    py, pz, zw = p._sections()
    g = stations_of(h)
    if not (np.array_equal(g.yc, p.yc)
            and np.array_equal(-h.vcg + g.rise, p.zk)):
        raise ValueError(f"{h.name}: hydro/hull_geometry's stations differ "
                         f"from the plant's -- the drawing would not be the "
                         f"plant's hull")
    xf = line_points(g)
    rf = g.keel(xf)[0]
    xt, tol_t = line_table(g)
    return dict(
        name=h.name, L=h.L, b=h.b, beta_deg=float(g.beta_deg), lcg=h.lcg,
        vcg=h.vcg, depth=float(g.depth), mass=h.mass,
        Z_rest=float(p.Z_rest), Th_rest=float(p.Th_rest),
        T_rest=float(p.T), x=p.x.tolist(), yc=p.yc.tolist(),
        zk=p.zk.tolist(), z_chine=pz[:, 1].tolist(),
        z_deck=pz[:, 2].tolist(), z_waterline_rest=zw.tolist(),
        x_fine=(xf - h.lcg).tolist(), yc_fine=g.chine(xf).tolist(),
        zk_fine=(-h.vcg + rf).tolist(), line_tol=LINE_TOL,
        x_tab=(xt - h.lcg).tolist(), yc_tab=g.chine(xt).tolist(),
        tab_tol=float(tol_t),
        nozzle=[float(v) for v in p.thrusters[0]],
        intake=[float(v) for v in p.intakes[0]],
        x_transom=float(-h.lcg), x_stem=float(h.L - h.lcg),
        geometry=str(getattr(g, "source", "hydro/hull_geometry")),
        cache_tag=physics.cache_tag(h),
        note=("body frame from the CG: x forward, y left, z up; stations "
              "transom to stem; section = keel (0, zk), chine (+-yc, zk + "
              "min(yc tan(beta), depth)), deck edge (+-yc, zk + depth) -- "
              "the plant's prismatic hull; x_fine / yc_fine / zk_fine: the "
              "chine and keel from hydro/hull_geometry at points chosen so "
              "that the polyline is within line_tol (m) of them everywhere, "
              "denser where they bend (the stem); x_tab / yc_tab: the same "
              "for the chine with at most 64 points, within tab_tol (draw "
              "from these, never from taper / stem_rise)"))


def immersion(p, sea, S, T):
    """Keel immersion at every station (n, n_st) for states S at times T."""
    from sim.planing_vessel import SP
    X = S[:, :1] + p.x[None] * np.cos(S[:, 5:6])
    Y = S[:, 1:2] + p.x[None] * np.sin(S[:, 5:6])
    out = np.empty_like(X)
    for i0 in range(0, len(S), 500):
        sl = slice(i0, i0 + 500)
        out[sl] = sea_eval(sea, X[sl], Y[sl],
                           np.repeat(T[sl, None], X.shape[1], 1))
    keel = (p.Z_rest + S[:, 2:3] + p.zk[None]
            + SP * p.x[None] * (p.Th_rest + S[:, 4:5]))
    return out - keel


def frames_of(rec, p, sea, S, T, a_cg, a_bow, thr_cmd, noz_cmd, X0, Y0):
    imm = immersion(p, sea, S, T)
    fr = dict(
        t=np.round(T, 4), x=np.round(S[:, 0] - X0, 4),
        y=np.round(S[:, 1] - Y0, 4), z=np.round(p.Z_rest + S[:, 2], 5),
        roll=np.round(S[:, 3], 6), pitch=np.round(p.Th_rest + S[:, 4], 6),
        yaw=np.round(S[:, 5], 6), u=np.round(S[:, 6], 4),
        v=np.round(S[:, 7], 4), w=np.round(S[:, 8], 4),
        a_cg=np.round(a_cg, 3), a_bow=np.round(a_bow, 3),
        thrust=np.round(S[:, 12], 1), nozzle=np.round(S[:, 13], 5),
        thr_cmd=np.round(thr_cmd, 1), noz_cmd=np.round(noz_cmd, 5),
        wet=np.round((imm > 0).mean(1), 4))
    return {k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).tolist()
            for k, v in fr.items()}


def rebased_sea(sea, X0, Y0):
    phi2 = np.mod(sea.phi + sea.k * (X0 * np.cos(sea.th) + Y0
                                     * np.sin(sea.th)), 2 * np.pi)
    return dict(a=sea.a.tolist(), k=sea.k.tolist(), th=sea.th.tolist(),
                w=sea.w.tolist(), phi=phi2.tolist())


def policy_label(header):
    """The policy's label in a record header ('hand (heading frame)', ...),
    for records written before labels from the spec."""
    pol = header.get("policy") or {}
    if pol.get("label"):
        return pol["label"]
    spec = pol.get("spec")
    if isinstance(spec, (dict, str)):
        from .policies import label
        lb = label(spec)
        if lb:
            return lb
    return pol.get("name") or "policy"


def state_source(spec, summary=None):
    """What a policy's loop runs on: 'truth' (the plant state: scripted and
    hand by default, m0, learned rows), 'sensors' (state_source
    'sensors'), 'observation' (callables and external clients read the
    observation, i.e. the sensors)."""
    summary = summary or {}
    if summary.get("state_source"):
        return summary["state_source"]
    if isinstance(spec, dict):
        t = spec.get("type", "scripted")
        if t in ("callable", "external"):
            return "observation"
        return spec.get("state_source", "truth")
    if isinstance(spec, str):
        return "observation" if (":" in spec and not spec.startswith(
            "learned:")) else "truth"
    return "observation" if spec is not None else "truth"


def hull_tag(h):
    """The key a viewer cache of `h` is kept under: sim/physics.cache_tag,
    'legacy:NAME' for the legacy hull (whose tag is None)."""
    from sim import physics
    tag = physics.cache_tag(h)
    return tag if tag is not None else f"legacy:{h.name}"


def viewer_meta(rec, h, X0, Y0, run=None):
    from sim import physics
    from . import codeguard
    from .record import code_changes
    hd = rec.header
    sc = Scenario(hd["scenario"])
    tr = dict(hd["track"])
    tr["points"] = [[x - X0, y - Y0] for x, y in tr["points"]]
    return dict(
        run=run or sc.name, scenario=sc.name, hull_key=hd["hull_key"],
        hull_tag=hull_tag(h),
        physics=hd["physics"]["physics"], plant=hd["plant_class"],
        policy=policy_label(hd), code_changes=code_changes(rec),
        # the process that wrote this (the server, for its viewer data)
        # runs physics code older than the disk: restart it (codeguard)
        server_code_stale=codeguard.notes([physics.version(h)]),
        state_source=state_source((hd.get("policy") or {}).get("spec"),
                                  (hd.get("policy") or {}).get("summary")),
        warnings=hd.get("scenario_warnings") or [],
        error=(hd.get("error") or {}).get("message"),
        hs=float(hd["sea"]["hs"]), tp=float(hd["sea"]["tp"]),
        theta0=float(hd["sea"].get("theta0", math.pi)),
        dt=float(hd["dt"]), t0=0.0, t1=float(hd["t_end"]), t_first=0.0,
        origin=[X0, Y0], g=G, reference_position=hd["reference_position"],
        track=tr, reason=hd.get("reason"), seed=hd["seed"],
        t_max=float(hd["t_max"]), nozzle_max=float(hd["rud_max"]),
        fingerprint=rec.fingerprint(),
        conventions=(
            "world X,Y horizontal (ENU: X east, Y north), Z up, "
            "right-handed; positions minus origin; eta = sum a cos(k(X cos "
            "th + Y sin th) - w t + phi) with the rebased phi; z = CG height "
            "above calm water; pitch = total trim, > 0 bow down; R = "
            "Rz(yaw) Ry(pitch) Rx(roll); a_cg, a_bow = vertical "
            "acceleration (kinematic, m/s^2)"),
        stills=[])


def viewer_json(rec, path=None, run=None):
    """The viewer data of a whole record (dict; written to `path` too)."""
    h, p = geometry_plant(rec.header)
    sea = sea_of(rec.header)
    T, S = rec.states_with_start()
    X0, Y0 = float(rec.s0[0]), float(rec.s0[1])
    P = rec.plant
    a_cg = np.r_[0.0, P["a_cg"]]
    a_bow = np.r_[0.0, P["a_bow"]]
    thr = np.r_[rec.s0[12], P["thrust"]]
    noz = np.r_[rec.s0[13], P["nozzle"]]
    out = dict(meta=viewer_meta(rec, h, X0, Y0, run),
               sea=rebased_sea(sea, X0, Y0), hull=hull_dict(h, p),
               frames=frames_of(rec, p, sea, S, T, a_cg, a_bow, thr, noz,
                                X0, Y0))
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, separators=(",", ":"))
    return out


def _rot_xyz(roll, pitch, yaw):
    from .geo import rot_zyx
    return rot_zyx(roll, pitch, yaw)


def blender_scene(rec, path, t0=None, t1=None, fps=30.0, stills=None,
                  track_s=30.0):
    """viewer/proto_blender/render.py's scene.json for [t0, t1] of a
    record (studies/visual_export.py's format; absolute positions)."""
    from sim.planing_vessel import SP
    T, S = rec.states_with_start()
    t_end = float(T[-1])
    t0, t1 = clip_window(t_end, t0, t1)
    if fps is None or not (float(fps) > 0) or not math.isfinite(float(fps)):
        raise ValueError(f"fps: > 0, not {fps!r}")
    for ts in (stills or []):
        if not t0 <= float(ts) <= t1:
            raise ValueError(f"still at {ts} s is outside the window "
                             f"[{t0:g}, {t1:g}] s")
    h, p = geometry_plant(rec.header)
    sea = sea_of(rec.header)
    A = np.r_[0.0, rec.plant["a_cg"]]
    stills = list(stills or [0.5 * (t0 + t1)])
    tf = np.arange(t0, t1 + 0.5 / fps, 1.0 / fps)
    tf = tf[tf <= t1 + 1e-9]

    def state_at(t):
        i = int(np.clip(np.searchsorted(T, t) - 1, 0, len(T) - 2))
        f = (t - T[i]) / (T[i + 1] - T[i])
        s = (1 - f) * S[i] + f * S[i + 1]
        dy = (S[i + 1, 5] - S[i, 5] + np.pi) % (2 * np.pi) - np.pi
        s[5] = S[i, 5] + f * dy
        return s, (1 - f) * A[i] + f * A[i + 1]

    def imm(s, t):
        X = s[0] + p.x * np.cos(s[5])
        Y = s[1] + p.x * np.sin(s[5])
        e = sea_eval(sea, X, Y, np.full(len(X), t))
        keel = p.Z_rest + s[2] + p.zk + SP * p.x * (p.Th_rest + s[4])
        return e - keel, keel

    frames, err = [], 0.0
    for t in tf:
        s, acg = state_at(t)
        dd, keel = imm(s, t)
        pose = dict(t=float(t), x=float(s[0]), y=float(s[1]),
                    z=float(p.Z_rest + s[2]), roll=float(s[3]),
                    pitch=float(p.Th_rest + s[4]), yaw=float(s[5]),
                    u=float(s[6]), v=float(s[7]), a_cg=float(acg),
                    thrust=float(s[12]), nozzle=float(s[13]),
                    wet=float((dd > 0).mean()), d=np.round(dd, 4).tolist())
        R = _rot_xyz(pose["roll"], pose["pitch"], pose["yaw"])
        P3 = np.stack([p.x, np.zeros_like(p.x), p.zk])
        err = max(err, float(np.abs(pose["z"] + (R @ P3)[2] - keel).max()))
        frames.append(pose)
    track = []
    for t in np.arange(max(T[0], t0 - track_s), t1 + 1e-9, 0.05):
        s, _ = state_at(t)
        dd, _ = imm(s, t)
        wet = dd > 0
        track.append([round(float(t), 3), float(s[0]), float(s[1]),
                      float(s[5]), float(s[6]), float(wet.mean()),
                      float(max(dd[0], 0.0)),
                      float(dd[wet].mean()) if wet.any() else 0.0])
    steps = []
    for k in np.nonzero((T >= t0 - 1.0) & (T <= t1 + 1e-9))[0]:
        try:
            _, aux = p.deriv(S[k].copy(), T[k])
            d, V = np.round(aux["d"], 4).tolist(), np.round(aux["V"],
                                                             3).tolist()
        except Exception:                            # noqa: BLE001
            d, V = imm(S[k], T[k])[0].round(4).tolist(), None
        steps.append(dict(t=round(float(T[k]), 4), x=float(S[k, 0]),
                          y=float(S[k, 1]), z=float(p.Z_rest + S[k, 2]),
                          roll=float(S[k, 3]),
                          pitch=float(p.Th_rest + S[k, 4]),
                          yaw=float(S[k, 5]), u=float(S[k, 6]), d=d, V=V))
    ev = []
    for ts in stills:
        k = int(np.argmin(np.abs(tf - ts)))
        fr = frames[k]
        ev.append(dict(t_req=ts, frame=k, t=fr["t"], z=fr["z"],
                       pitch_deg=math.degrees(fr["pitch"]),
                       roll_deg=math.degrees(fr["roll"]), kn=fr["u"] / KN,
                       a_cg=fr["a_cg"], wet=fr["wet"], d_min=min(fr["d"]),
                       d_max=max(fr["d"])))
    hd = rec.header
    from .record import code_changes
    out = dict(
        meta=dict(source=f"app record {rec.fingerprint()[:16]}",
                  code_changes=code_changes(rec),
                  run=Scenario(hd["scenario"]).name, row=policy_label(hd),
                  seed=hd["seed"], leg=0, hull_key=hd["hull_key"],
                  plant=f"{hd['plant_class']} ({hd['hull_key']})",
                  hs=float(hd["sea"]["hs"]), tp=float(hd["sea"]["tp"]),
                  fps=fps, t0=t0, t1=t1,
                  conventions=("world X,Y horizontal, Z up, right-handed; "
                               "eta = sum a cos(k(X cos th + Y sin th) - w t "
                               "+ phi); pose z = CG height above calm "
                               "water; Blender rotation_euler (roll, pitch, "
                               "yaw) mode XYZ; pitch > 0 = bow down"),
                  check_keel_rigid_vs_plant_m=err, stills=ev,
                  track_cols=["t", "x", "y", "yaw", "u", "wet_share",
                              "transom_depth", "mean_wet_depth"]),
        sea={k: getattr(sea, k).tolist() for k in ("a", "k", "th", "w",
                                                   "phi")},
        hull=hull_dict(h, p), frames=frames, track=track, steps=steps)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)
    return out


def clip_window(t_end, t0=None, t1=None):
    """(t0, t1) of an export window: defaults the last 4 s; must satisfy
    0 <= t0 < t1 <= t_end (ValueError with the record's span otherwise)."""
    t1 = float(t_end) if t1 is None else float(t1)
    t0 = max(0.0, t1 - 4.0) if t0 is None else float(t0)
    if not (math.isfinite(t0) and math.isfinite(t1)) or not (
            0.0 <= t0 < t1 <= float(t_end) + 1e-9):
        raise ValueError(f"window [{t0:g}, {t1:g}] s: need 0 <= t0 < t1 <= "
                         f"{float(t_end):g} s (the record's end)")
    return t0, t1


def find_blender():
    """blender(.exe): $BLENDER, then PATH, then the newest
    'Blender Foundation/Blender X.Y' under Program Files; None if absent."""
    env = os.environ.get("BLENDER")
    if env and os.path.isfile(env):
        return env
    w = shutil.which("blender")
    if w:
        return w
    cands = []
    for base in (os.environ.get("ProgramFiles", r"C:\Program Files"),
                 os.environ.get("ProgramFiles(x86)", "")):
        if base:
            cands += glob.glob(os.path.join(base, "Blender Foundation",
                                            "Blender*", "blender.exe"))

    def ver(p):
        v = os.path.basename(os.path.dirname(p)).replace("Blender", "")
        try:
            return tuple(int(x) for x in v.strip().split("."))
        except ValueError:
            return ()
    cands.sort(key=ver)
    return cands[-1] if cands else None


def blender_command(scene_path, out, frame=None, engine="EEVEE",
                    res="960x540", samples=16, clip=None, mp4=None,
                    blender=None):
    """The command line that renders `scene_path` with
    viewer/proto_blender/render.py: a still to the PNG `out` (frame index
    `frame`), or with clip=(a, b) the frames a..b (indices into the scene's
    frames, inclusive) as PNGs into the folder `out` plus the MP4 `mp4`
    (render.py encodes it with Blender's own ffmpeg). Nothing here runs
    it; blender defaults to find_blender() (or 'blender')."""
    from . import ROOT
    render = os.path.join(ROOT, "viewer", "proto_blender", "render.py")
    exe = blender or find_blender() or "blender"
    cmd = [exe, "-b", "--factory-startup", "--python", render, "--",
           "--scene", os.path.abspath(scene_path), "--out",
           os.path.abspath(out), "--engine", engine, "--res", res,
           "--samples", str(samples)]
    if clip is not None:
        a, b = (int(v) for v in clip)
        if not 0 <= a <= b:
            raise ValueError(f"clip ({a}, {b}): frame indices 0 <= a <= b")
        cmd += ["--clip", f"{a}:{b}", "--mp4",
                os.path.abspath(mp4 or os.path.join(out, "clip.mp4"))]
    elif frame is not None:
        cmd += ["--frame", str(int(frame))]
    return cmd
