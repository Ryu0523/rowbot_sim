#!/usr/bin/env python3
"""
Bow-camera observations of the sea ahead, computed after the fact from a
recorded run of the frozen plant (studies/preview_value_run.py saves the CG
pose every 0.05 s and the sea's component arrays, so the surface the vessel
met can be rebuilt exactly without the plant).

GEOMETRY (the information ceiling of any optical sensor at this height).
A camera on the stem, `h_mast` metres above the deck, horizon-stabilised
(yaw follows the vessel; roll and pitch removed, as an ideal IMU would).
Rays on a grid of azimuth (+-30 deg about the bow) and depression below
the horizon (0.2-15 deg, geometric spacing: 0.2 deg reaches about 500 m at
a 1.9 m eye height, 15 deg about 7 m). Each ray is marched out to `s_max`
metres and stopped at its first intersection with the surface; crests hide
what lies behind them, so the far field is seen only in patches. Per ray:
the range to the hit (s_max when the ray reaches the horizon unobstructed),
the surface elevation at the hit and the surface normal there.

APPEARANCE (for the degradation study, studies/bow_camera_train.py). A
grey image from the same hits: Fresnel reflection of a sky whose radiance
rises toward the horizon, a sun glint lobe, a dark water body, and foam
where the local slope is steep. A sketch of a sea picture, not a renderer;
the degradations (glare, spray, blur, horizon jitter) are applied at
training time.

TARGETS AND ONBOARD INPUTS per frame (every `every` seconds):
  fut_eta   the elevation the bow actually met at tau = 0.25 .. 10 s
  past_eta  the elevation at the bow over the past 20 s (0.25 s steps)
  motion    heave, pitch, speed over the past 10 s (0.25 s steps)
  imp_max   the largest filtered CG acceleration (g) in the next 2.5, 5,
            10 s
  h_next    the largest zero-crossing wave height met in the next 8 s,
            over Hs (the preview_h rule's metric)

    python -m studies.bow_camera <run name or glob> [--every 0.5]

Writes studies/_cache/camera/<run>.npz (float16 images). Read-only use of
the preview cache.
"""
import glob
import json
import math
import os
import sys
import time

import os as _os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from studies.preview_value_run import next_wave_height   # noqa: E402

RUNS = os.path.join(ROOT, "studies", "_cache", "preview", "runs")
OUT = os.path.join(ROOT, "studies", "_cache", "camera")
KN = 0.514444
G = 9.81
POSE_DT = 0.05            # preview_value_run.POSE_DECIM x dt 0.01
STEP = 0.25               # the sampling of past/future series
N_PAST, N_FUT, N_MOT = 80, 40, 40
IMP_WINDOWS = (2.5, 5.0, 10.0)


class Surface:
    """The linear surface of the recorded sea components."""

    def __init__(self, a, w, th, phi):
        self.a, self.w, self.phi = (np.asarray(v, float) for v in (a, w, phi))
        self.k = self.w ** 2 / G
        self.cth, self.sth = np.cos(th), np.sin(th)
        self.kx, self.ky = self.k * self.cth, self.k * self.sth

    def eta(self, x, y, t):
        """Elevation at points x, y (arrays of one shape) at time(s) t (a
        scalar or an array of the same shape)."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        t = np.asarray(t, float)
        shp = x.shape
        ph = (x.reshape(-1, 1) * self.kx + y.reshape(-1, 1) * self.ky
              - np.broadcast_to(t, shp).reshape(-1, 1) * self.w + self.phi)
        return (self.a * np.cos(ph)).sum(-1).reshape(shp)

    def grad(self, x, y, t):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        shp = x.shape
        ph = (x.reshape(-1, 1) * self.kx + y.reshape(-1, 1) * self.ky
              - float(t) * self.w + self.phi)
        s = -self.a * np.sin(ph)
        return ((s * self.kx).sum(-1).reshape(shp),
                (s * self.ky).sum(-1).reshape(shp))


def rot_zyx(phi, theta, psi):
    """Body -> world rotation for roll phi, pitch theta, yaw psi (ENU,
    body x forward, y left, z up)."""
    cf, sf = math.cos(phi), math.sin(phi)
    ct, st = math.cos(theta), math.sin(theta)
    cp, sp = math.cos(psi), math.sin(psi)
    return np.array([
        [cp * ct, cp * st * sf - sp * cf, cp * st * cf + sp * sf],
        [sp * ct, sp * st * sf + cp * cf, sp * st * cf - cp * sf],
        [-st, ct * sf, ct * cf]])


class Camera:
    def __init__(self, n_az=32, az_max_deg=30.0, n_dep=24,
                 dep_deg=(0.2, 15.0), s_max=400.0, n_s=160, grid_dx=2.0,
                 sun_az_deg=35.0, sun_el_deg=25.0):
        self.az = np.radians(np.linspace(-az_max_deg, az_max_deg, n_az))
        self.dep = np.radians(np.geomspace(dep_deg[0], dep_deg[1], n_dep))
        self.s = np.geomspace(2.0, s_max, n_s)
        self.s_max, self.dx = float(s_max), float(grid_dx)
        # ray directions in the yaw-aligned frame (forward, left, up)
        D, A = np.meshgrid(self.dep, self.az, indexing="ij")
        self.dir_f = np.cos(D) * np.cos(A)
        self.dir_l = np.cos(D) * np.sin(A)
        self.dir_u = -np.sin(D)
        self.lat_max = s_max * math.sin(math.radians(az_max_deg)) + 4.0
        self.f_grid = np.arange(0.0, s_max + grid_dx, grid_dx)
        self.l_grid = np.arange(-self.lat_max, self.lat_max + grid_dx, grid_dx)
        se, sa = math.radians(sun_el_deg), math.radians(sun_az_deg)
        self.sun = np.array([math.cos(se) * math.cos(sa),
                             math.cos(se) * math.sin(sa), math.sin(se)])

    def frame(self, surf, c, psi, t):
        """Images for a camera at world position c (3,), yaw psi, time t:
        (log range, elevation at hit, grey, hit mask)."""
        cp, sp = math.cos(psi), math.sin(psi)
        F, Lt = np.meshgrid(self.f_grid, self.l_grid, indexing="ij")
        X = c[0] + F * cp - Lt * sp
        Y = c[1] + F * sp + Lt * cp
        E = surf.eta(X, Y, t)                       # (nf, nl)
        # ray samples -> grid coordinates (bilinear)
        s = self.s[None, None, :]
        f = self.dir_f[..., None] * s
        lat = self.dir_l[..., None] * s
        z = c[2] + self.dir_u[..., None] * s
        fi = np.clip(f / self.dx, 0, len(self.f_grid) - 1.001)
        li = np.clip((lat + self.lat_max) / self.dx, 0, len(self.l_grid)
                     - 1.001)
        i0, j0 = fi.astype(int), li.astype(int)
        tf, tl = fi - i0, li - j0
        eta_s = ((1 - tf) * (1 - tl) * E[i0, j0] + tf * (1 - tl) * E[i0 + 1, j0]
                 + (1 - tf) * tl * E[i0, j0 + 1] + tf * tl * E[i0 + 1, j0 + 1])
        below = z <= eta_s
        hit = below.any(-1)
        idx = np.where(hit, below.argmax(-1), len(self.s) - 1)
        # refine between the last sample above and the first below
        i1 = idx
        i0s = np.maximum(idx - 1, 0)
        za = np.take_along_axis(z - eta_s, i0s[..., None], -1)[..., 0]
        zb = np.take_along_axis(z - eta_s, i1[..., None], -1)[..., 0]
        w = np.where(np.abs(za - zb) > 1e-9, za / (za - zb), 1.0)
        w = np.clip(w, 0.0, 1.0)
        rng = np.where(hit, self.s[i0s] + w * (self.s[i1] - self.s[i0s]),
                       self.s_max)
        rng = np.where(idx == 0, self.s[0], rng)
        # hit points in the world
        hf = self.dir_f * rng
        hl = self.dir_l * rng
        hx = c[0] + hf * cp - hl * sp
        hy = c[1] + hf * sp + hl * cp
        hz = c[2] + self.dir_u * rng
        eta_hit = surf.eta(hx, hy, t)
        gx, gy = surf.grad(hx, hy, t)
        grey = self.shade(gx, gy, cp, sp, hit)
        return (np.log(rng).astype(np.float32), eta_hit.astype(np.float32),
                grey.astype(np.float32), hit)

    def shade(self, gx, gy, cp, sp, hit):
        """Grey level from the surface normal at each hit (module
        docstring)."""
        # view direction in the world
        dw = np.stack([self.dir_f * cp - self.dir_l * sp,
                       self.dir_f * sp + self.dir_l * cp, self.dir_u], -1)
        n = np.stack([-gx, -gy, np.ones_like(gx)], -1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        cos_i = np.clip(-(dw * n).sum(-1), 0.0, 1.0)
        r = dw + 2.0 * cos_i[..., None] * n              # reflected ray
        el = np.arcsin(np.clip(r[..., 2], -1.0, 1.0))
        sky = np.where(el >= 0, 0.35 + 0.45 * np.exp(-el / 0.25), 0.30)
        R0 = 0.02
        fres = R0 + (1.0 - R0) * (1.0 - cos_i) ** 5
        sun_w = np.array([self.sun[0] * cp - self.sun[1] * sp,
                          self.sun[0] * sp + self.sun[1] * cp, self.sun[2]])
        cos_rs = np.clip((r * sun_w).sum(-1), -1.0, 1.0)
        ang = np.arccos(cos_rs)
        glint = 3.0 * np.exp(-(ang / math.radians(4.0)) ** 2)
        slope = np.hypot(gx, gy)
        foam = np.clip((slope - 0.25) / 0.15, 0.0, 1.0) * 0.6
        water = 0.08
        img = fres * (sky + glint) + (1.0 - fres) * water + foam
        img = np.where(hit, img, 0.80)                  # horizon sky
        return np.clip(img, 0.0, 1.0)


def load_run(name):
    with open(os.path.join(RUNS, name + ".json")) as f:
        meta = json.load(f)
    z = np.load(os.path.join(RUNS, name + ".npz"))
    return meta, z


def process(name, every=0.5, h_mast=1.0, cam=None, verbose=True):
    cam = cam or Camera()
    meta, z = load_run(name)
    surf = Surface(z["sea_a"], z["sea_w"], z["sea_th"], z["sea_phi"])
    pose = np.asarray(z["pose"], float)                # (n, 10) at 0.05 s
    dt = float(meta["dt"])
    t_pose = dt * (1 + 5 * np.arange(len(pose)))
    x_st = float(meta["x_station"])
    Z_rest, Th_rest = float(meta["Z_rest"]), float(meta["Th_rest"])
    fb0 = float(meta["freeboard_static"])
    hs = float(meta["hs_eff"])
    z_cam_body = fb0 - Z_rest + h_mast
    # the elevation at the bow along the whole run (what the bow met)
    xb = pose[:, 0] + x_st * np.cos(pose[:, 5])
    yb = pose[:, 1] + x_st * np.sin(pose[:, 5])
    eta_bow = surf.eta(xb, yb, t_pose)
    acc = np.asarray(z["acc_f"], float)[:, 0]          # CG, g, at dt
    t_acc = dt * np.arange(1, len(acc) + 1)
    k_step = int(round(STEP / POSE_DT))                # 5 pose rows
    n_every = int(round(every / POSE_DT))
    i_first = N_PAST * k_step
    i_last = len(pose) - N_FUT * k_step - 1
    frames = list(range(i_first, i_last, n_every))
    n_f = len(frames)
    nd, na = len(cam.dep), len(cam.az)
    rng_img = np.empty((n_f, nd, na), np.float16)
    eta_img = np.empty((n_f, nd, na), np.float16)
    grey_img = np.empty((n_f, nd, na), np.float16)
    hit_frac = np.empty(n_f, np.float32)
    past = np.empty((n_f, N_PAST), np.float32)
    fut = np.empty((n_f, N_FUT), np.float32)
    motion = np.empty((n_f, 3, N_MOT), np.float32)
    imp = np.empty((n_f, len(IMP_WINDOWS)), np.float32)
    h_next = np.empty(n_f, np.float32)
    t_f = np.empty(n_f, np.float32)
    speed = np.empty(n_f, np.float32)
    cam_z = np.empty(n_f, np.float32)
    t0 = time.time()
    for q, i in enumerate(frames):
        p = pose[i]
        t = t_pose[i]
        R = rot_zyx(p[3], Th_rest + p[4], p[5])
        c = np.array([p[0], p[1], Z_rest + p[2]]) + R @ np.array(
            [x_st, 0.0, z_cam_body])
        lr, eh, gr, hit = cam.frame(surf, c, p[5], t)
        rng_img[q], eta_img[q], grey_img[q] = lr, eh, gr
        hit_frac[q] = hit.mean()
        past[q] = eta_bow[i - (N_PAST - 1) * k_step:i + 1:k_step]
        fut[q] = eta_bow[i + k_step:i + (N_FUT + 1) * k_step:k_step]
        sl = slice(i - (N_MOT - 1) * k_step, i + 1, k_step)
        motion[q, 0] = pose[sl, 2]
        motion[q, 1] = pose[sl, 4]
        motion[q, 2] = pose[sl, 6]
        ia = int(round(t / dt))
        for j, w in enumerate(IMP_WINDOWS):
            imp[q, j] = acc[ia:ia + int(round(w / dt))].max()
        h, _ = next_wave_height(eta_bow[i:i + int(round(8.0 / POSE_DT)) + 1
                                        :k_step], STEP)
        h_next[q] = h / hs
        t_f[q], speed[q], cam_z[q] = t, p[6], c[2]
        if verbose and (q % 200 == 0):
            print(f"  {name}: frame {q}/{n_f}  {time.time() - t0:.0f} s",
                  flush=True)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(
        os.path.join(OUT, name + ".npz"), t=t_f, speed=speed, cam_z=cam_z,
        rng_img=rng_img, eta_img=eta_img, grey_img=grey_img,
        hit_frac=hit_frac, past_eta=past, fut_eta=fut, motion=motion,
        imp_max=imp, h_next=h_next, hs=hs, dep_deg=np.degrees(cam.dep),
        az_deg=np.degrees(cam.az), s_max=cam.s_max, h_mast=h_mast,
        meta=json.dumps(dict(name=name, every=every, kind=meta["arg"]["kind"],
                             v_hi_kn=meta["arg"]["v_hi_kn"],
                             seed=meta["seed"], sea_seed=meta["sea_seed"],
                             heading_rad=meta["arg"]["heading_rad"],
                             sea_params=meta["sea_params"])))
    if verbose:
        print(f"{name}: {n_f} frames in {time.time() - t0:.0f} s; hit "
              f"fraction {hit_frac.mean():.2f}; camera height "
              f"{cam_z.mean():.2f} m", flush=True)
    return n_f


def _process_quiet(name, every):
    try:
        return process(name, every=every, verbose=False)
    except Exception as e:          # noqa: BLE001
        print(f"FAILED {name}: {e!r}", flush=True)
        return None


if __name__ == "__main__":
    args = sys.argv[1:]
    every, procs = 0.5, 1
    if "--every" in args:
        i = args.index("--every")
        every = float(args[i + 1])
        del args[i:i + 2]
    if "--procs" in args:
        i = args.index("--procs")
        procs = int(args[i + 1])
        del args[i:i + 2]
    pats = args or ["*"]
    names = []
    for pat in pats:
        names += [os.path.basename(p)[:-5] for p in
                  sorted(glob.glob(os.path.join(RUNS, pat + ".json")))]
    todo = [nm for nm in names
            if not os.path.exists(os.path.join(OUT, nm + ".npz"))]
    print(f"{len(todo)} runs to process, every {every} s, {procs} processes",
          flush=True)
    if procs <= 1:
        for nm in todo:
            process(nm, every=every)
    else:
        import multiprocessing as mp
        from functools import partial
        with mp.get_context("spawn").Pool(procs) as pool:
            pool.map(partial(_process_quiet, every=every), todo, chunksize=1)
