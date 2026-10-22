#!/usr/bin/env python3
"""
Helpers for check V2 (learn/meta/HULL_FORCE_AUDIT_REVIEW_2026-10-09.md section 4):
the Scarab 195 planing hull as a hydro.rankine HullForm at the plant's rest
attitude, a ZERO-SPEED Rankine free-surface solver on it
(hydro/rankine/doublebody.DBForwardSpeedLTI with zero advection operators, i.e.
the kinematic/dynamic pair of radiation.py with the beach nu), and the field
utilities the second-order forcing needs (surface gradients on the structured
hull mesh, horizontal gradients on the structured free-surface band grid).

Nothing under hydro/ or sim/ is modified: this file only composes what those
modules export.

Frames (hydro/rankine/hull.py): body frame origin at the CG (x_b forward, z_b
up); earth frame z = 0 calm water, fluid z < 0; sinkage s > 0 moves the origin
DOWN, trim theta > 0 is bow UP. The planing plant (sim/planing_vessel.py)
gives the CG height Z_rest above the water and the pitch state Th_rest with
SIGN_PITCH = -1 (bow up < 0), so sink = -Z_rest, trim = -Th_rest.
"""
import time

import numpy as np
import scipy.sparse as sp
import torch

from hydro.rankine.hull import HullForm
from hydro.rankine.unsteady import build_grid, incident
from hydro.rankine.doublebody import DBForwardSpeedLTI, surface_gradient2, _line_deriv, _chord

G0 = 9.81


# ------------------------------------------------------------------ geometry
def scarab_form():
    """The plant's own prismatic hull (sim/planing_vessel.py:91-102, the v0
    geometry of hydro/hull_geometry.py) as a HullForm: constant deadrise V
    bottom from the straight keel (stem_rise 0) to the chine yc(x) (full beam
    aft, quarter-ellipse taper over the bow 35 % of L, 1e-4 m floor at the
    stem), vertical side from the chine to the deck, flat transom at the aft
    end. Returns (hull, plant) with the plant for its rest attitude."""
    from hydro.hulls import scarab_195
    from sim.planing_vessel import PlaningVessel
    from sim.test_vessel import Monochromatic
    h = scarab_195()
    p = PlaningVessel(h, Monochromatic(1.0, 0.0), dt=0.01)
    L, b, lcg, vcg = h.L, h.b, h.lcg, h.vcg
    tb = np.tan(np.radians(h.beta_deg))
    x0 = (1.0 - h.taper) * L
    lt = max(L - x0, 1e-9)
    eps = 1e-4

    def yc(xb):
        x = np.asarray(xb, float) + lcg                      # from the transom
        s = np.clip((x - x0) / lt, 0.0, 1.0)
        return 0.5 * b * np.sqrt(np.maximum(1.0 - s ** 2, 0.0)) + eps

    def D(xb):
        return np.full_like(np.asarray(xb, float), vcg)       # keel z_b = -vcg (straight keel)

    def hh(xb, zb):
        xb = np.asarray(xb, float)
        zb = np.asarray(zb, float)
        return np.clip((zb + vcg) / tb, 0.0, yc(xb))
    hull = HullForm(hh, D, -lcg, L - lcg, "scarab195_rest", transom=True, above="form")
    hull.B, hull.T, hull.beta = b, p.T, np.radians(h.beta_deg)
    hull.plant = p
    hull.rho = h.rho
    hull.sink_rest, hull.trim_rest = -float(p.Z_rest), -float(p.Th_rest)
    hull.yc = yc
    return hull, p


# ------------------------------------------------------------------ solver
DEFAULT_GRID_V2 = dict(M=40, aspect=1.0, J_mult=2.0, up=1.5, lat=3.0, r_up=1.12, r_lat=1.10, h_max=0.5,
                       b_up=0.75, b_lat=1.5, nu_max=1.5)


def build_zero_speed(hull, hull_mesh=(40, 8), face_ny=6, grid_kw=None, g=G0, rho=None, device=None):
    """Zero-speed Rankine solver on the Scarab at its rest attitude.
    Body: HullForm.mesh(nx, nz) (cosine-clustered stations and rows, tag 0) +
    the wetted transom face (face_ny x nz, tag 2). Free surface: the band grid
    of unsteady.build_grid with symmetric extents (U = 0). Operators: A = K = 0
    (no base flow), so the conditions are zeta' = -nu zeta - sigma + f_k,
    phi' = -nu phi - g zeta + f_d with sigma = -phi_z. Returns dict."""
    kw = dict(DEFAULT_GRID_V2, **(grid_kw or {}))
    kw.setdefault("down", kw["up"])
    kw.setdefault("b_dn", kw["b_up"])
    kw.setdefault("r_dn", kw["r_up"])
    sink, trim = hull.sink_rest, hull.trim_rest
    nx, nz = hull_mesh
    t0 = time.time()
    hull_pan = hull.mesh(nx, nz, sink=sink, trim=trim)
    face = hull.face_mesh(face_ny, nz, sink=sink, trim=trim)
    body = hull_pan + face
    G = build_grid(hull, sink, trim, 1.0, **kw)
    NF = len(G["fs"])
    Z = sp.csr_matrix((NF, NF))
    rho = hull.rho if rho is None else rho
    s = DBForwardSpeedLTI(body, G["fs"], Z, Z, G["nu"] * np.sqrt(g / hull.L), 0.0, g, rho, device,
                          hull_shape=(nx, nz))
    t_build = time.time() - t0
    return dict(s=s, G=G, body=body, n_hull=len(hull_pan), hull_shape=(nx, nz), face_shape=(face_ny, nz),
                sink=sink, trim=trim, t_build=t_build, grid_kw=kw)


def mode_normals(body, ref_earth):
    """Generalised normals about the earth point ref_earth = (x_r, z_r) (the
    CG): heave n3 = n_z; pitch n5 = (z - z_r) n_x - (x - x_r) n_z, the
    rotation vector along +y, i.e. positive pitch = bow DOWN (unsteady.
    mode_vectors convention). Also the displacement field d(r) per unit
    amplitude: heave (0, 0, 1); pitch alpha y_hat x r = (z - z_r, 0, -(x - x_r))."""
    n, c = body.n, body.c
    xr, zr = ref_earth
    r = np.stack([c[:, 0] - xr, c[:, 1], c[:, 2] - zr], 1)
    n3 = n[:, 2].copy()
    n5 = r[:, 2] * n[:, 0] - r[:, 0] * n[:, 2]
    d3 = np.tile([0.0, 0.0, 1.0], (len(body), 1))
    d5 = np.stack([r[:, 2], np.zeros(len(body)), -r[:, 0]], 1)
    return dict(n3=n3, n5=n5, d3=d3, d5=d5, r=r)


# ------------------------------------------------------------------ fields
def body_gradient(setup, phi, phin):
    """Full gradient of a (complex) panel field on the body: structured
    second-order formulas (surface_gradient2) on the hull block and on the
    transom-face block separately, normal part from phin."""
    body, nh = setup["body"], setup["n_hull"]
    phi = np.asarray(phi)
    phin = np.asarray(phin)
    out = np.zeros((len(body), 3), complex)
    hull_pan = body.subset(np.arange(len(body)) < nh)
    face_pan = body.subset(np.arange(len(body)) >= nh)
    for pan, sl, shape in ((hull_pan, slice(0, nh), setup["hull_shape"]), (face_pan, slice(nh, None), setup["face_shape"])):
        gr = surface_gradient2(pan, phi[sl].real, phin[sl].real, shape)
        gi = surface_gradient2(pan, phi[sl].imag, phin[sl].imag, shape)
        out[sl] = gr + 1j * gi
    return out


def fs_metric(G):
    """Chord parameters and 2x2 inverse metric of the band grid (centroids Xc,
    Yc of shape (nx, ny)): returns dict(S0, S1, Xi, Yi, Xj, Yj, det)."""
    X, Y = G["Xc"], G["Yc"]
    C = np.stack([X, Y, np.zeros_like(X)], -1)
    S0, S1 = _chord(C, 0), _chord(C, 1)
    Xi, Yi = _line_deriv(X, S0, 0), _line_deriv(Y, S0, 0)
    Xj, Yj = _line_deriv(X, S1, 1), _line_deriv(Y, S1, 1)
    det = Xi * Yj - Xj * Yi
    return dict(S0=S0, S1=S1, Xi=Xi, Yi=Yi, Xj=Xj, Yj=Yj, det=det)


def fs_hgrad(G, met, f):
    """Horizontal gradient (f_x, f_y) of a complex panel field on the free
    surface (panel order k = j nx + i): three-point Lagrange formulas in chord
    length along the two index lines (second order), then the chain rule
    through the 2x2 Jacobian. Even symmetry in y at the centre plane is not
    enforced (one-sided formula at j = 0). Returns arrays in panel order."""
    nx, ny = G["shape"]
    F = np.asarray(f).reshape(ny, nx).T
    out = []
    for part in (F.real, F.imag):
        fi = _line_deriv(part, met["S0"], 0)
        fj = _line_deriv(part, met["S1"], 1)
        fx = (fi * met["Yj"] - fj * met["Yi"]) / met["det"]
        fy = (-fi * met["Xj"] + fj * met["Xi"]) / met["det"]
        out.append((fx, fy))
    fx = out[0][0] + 1j * out[1][0]
    fy = out[0][1] + 1j * out[1][1]
    return fx.T.reshape(-1), fy.T.reshape(-1)


def incident_fields(P, omega, beta, g=G0, amp=1.0):
    """Deep-water incident wave phi_I = (i g A/omega) e^{kz} e^{-ik(x cos b + y sin b)}
    (y-even part, as unsteady.incident): phi, grad (3), zeta = -i omega phi/g at z = 0,
    phi_zz = k^2 phi. For beta = pi (head wave travelling toward -x)."""
    k = omega * omega / g
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    cb, sb = np.cos(beta), np.sin(beta)
    c0 = 1j * g * amp / omega
    ex = np.exp(k * z - 1j * k * x * cb)
    phi = c0 * ex * np.cos(k * y * sb)
    gx = -1j * k * cb * phi
    gy = -c0 * ex * k * sb * np.sin(k * y * sb)
    gz = k * phi
    return dict(phi=phi, grad=np.stack([gx, gy, gz], 1), zeta=-1j * omega * phi / g, phizz=k * k * phi, k=k)


def waterline_data(setup):
    """Waterline (z = 0) edge of every top-row hull panel and of the top row
    of the transom face: returns dict(idx (panel indices), dl (edge length),
    n (panel normal), flare = 1/sqrt(1 - n_z^2), x_mid, y_mid) -- the mean
    waterline curve on the half hull."""
    body, nh = setup["body"], setup["n_hull"]
    nx, nz = setup["hull_shape"]
    fy, _ = setup["face_shape"]
    idx = np.concatenate([np.arange(nx) * nz + (nz - 1), nh + np.arange(fy) * nz + (nz - 1)])
    raw = body.raw[idx]
    z = raw[:, :, 2]
    dl = np.zeros(len(idx))
    xm = np.zeros(len(idx))
    ym = np.zeros(len(idx))
    for q in range(len(idx)):
        top = np.where(np.abs(z[q]) < 1e-9)[0]
        if len(top) >= 2:
            a, b = raw[q, top[0]], raw[q, top[-1]]
            dl[q] = np.linalg.norm(a - b)
            xm[q], ym[q] = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
    n = body.n[idx]
    flare = 1.0 / np.sqrt(np.maximum(1.0 - n[:, 2] ** 2, 1e-12))
    return dict(idx=idx, dl=dl, n=n, flare=flare, x_mid=xm, y_mid=ym)


def waterline_values(setup, f):
    """Linear extrapolation of a panel field to z = 0 along each station of the hull block and of the face
    block (unsteady.hull_waterline_values logic), in the order of waterline_data(setup)['idx']."""
    body, nh = setup["body"], setup["n_hull"]
    nx, nz = setup["hull_shape"]
    fy, _ = setup["face_shape"]
    f = np.asarray(f)
    out = []
    for off, ns in ((0, nx), (nh, fy)):
        C = body.c[off:off + ns * nz].reshape(ns, nz, 3)
        F = f[off:off + ns * nz].reshape(ns, nz)
        z1, z2 = C[:, -1, 2], C[:, -2, 2]
        t = (0.0 - z1) / (z1 - z2)
        out.append(F[:, -1] + t * (F[:, -1] - F[:, -2]))
    return np.concatenate(out)


def mem_report():
    import psutil
    pr = psutil.Process()
    mi = pr.memory_info()
    out = dict(rss_MB=mi.rss / 1e6, peak_rss_MB=getattr(mi, "peak_wset", mi.rss) / 1e6)
    if torch.cuda.is_available():
        out["gpu_alloc_MB"] = torch.cuda.memory_allocated() / 1e6
        out["gpu_peak_MB"] = torch.cuda.max_memory_allocated() / 1e6
    return out
