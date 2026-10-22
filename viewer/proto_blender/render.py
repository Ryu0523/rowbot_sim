"""
Blender prototype: our boat (the plant's scarab195 stations) on our exact
wave field at a recorded moment of a closed-loop run.

Run in background mode only:
    blender.exe -b --factory-startup --python render.py -- [options]

    --scene  scene.json (studies/visual_export.py writes it)
    --frame  N         frame index into scene.json "frames" (default: the
                       first still listed in meta.stills)
    --engine CYCLES | EEVEE
    --out    PNG path (still) or folder (clip)
    --res    1280x720
    --samples 128      Cycles samples (adaptive) / EEVEE TAA samples
    --clip A:B         render frames A..B (inclusive) as PNGs + an MP4
    --water  opaque | volume   (volume: Cycles only, transmission + absorbing
                       / scattering slab; opaque: Fresnel-specular surface
                       over a dark upwelling colour, works in both engines)
    --cam_dist 18.4 --cam_az -0.62 --cam_el 0.34 --fov 46  (the viewer's
                       chase view scaled from its 10 m hull to 5.4 m)
    --sun_el 24 --sun_rel 150  (sun elevation; sun azimuth relative to the
                       camera's horizontal look direction, deg CCW)
    --exposure -2.3    --blur 0.5 (motion-blur shutter in frames, 0 = off)
    --blend path       also save the .blend

What is exact and what is not (also printed at the end):
  exact    wave elevation: the recorded 120-component linear sum, evaluated
           per vertex at the frame's time (the same formula as
           learn/meta/data5.sea_eval); the boat's pose from the plant state
           (CG height Z_rest + z, pitch Th_rest + pitch, roll, yaw), the
           hull's wetted shape (keel, 20 deg deadrise, chine plan) from the
           plant's own station formula.
  filtered far away: components shorter than ~3-6 local grid spacings are
           faded out (anti-aliasing), their slope variance goes into a
           roughness that grows with distance.
  artistic foam (whitecaps from the Longuet-Higgins downward-acceleration
           criterion with a tuned threshold and a fade over ~2 s; wake from
           the recorded track gated by the recorded wetted length; contact
           foam from the recorded station immersion), fine ripples (bump
           noise), deck hardware (console, mast, rails), colours.
  missing  the boat's own waves (Kelvin wake, bow wave), spray and spray
           sheets, water thrown up at the landing, wet hull, sea spray haze.
"""
import json
import math
import os
import sys
import time

import bmesh
import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

G = 9.81
HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ args
def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    o = dict(scene=os.path.join(HERE, "scene.json"), frame=None,
             engine="CYCLES", out=None, res="1280x720", samples=128,
             clip=None, water="opaque", cam_dist=18.4, cam_az=-0.62,
             cam_el=0.34, fov=46.0, sun_el=24.0, sun_rel=150.0,
             exposure=-2.7, blur=0.2, blend=None, spray=1, encode=None,
             mp4=os.path.join(HERE, "clip.mp4"), resume=0, ee_rt=1, grid="normal",
             look="AgX - Medium High Contrast", stats=1)
    i = 0
    while i < len(argv):
        k = argv[i].lstrip("-")
        v = argv[i + 1]
        if k not in o:
            raise SystemExit(f"unknown option --{k}")
        d = o[k]
        o[k] = (type(d)(v) if isinstance(d, (int, float)) and d is not None
                and not isinstance(d, bool) else v)
        i += 2
    if o["frame"] is not None:
        o["frame"] = int(o["frame"])
    # Blender resolves a bare relative output path against the drive root,
    # not the working directory: make every path absolute here
    for k in ("scene", "out", "blend", "encode", "mp4"):
        if o[k]:
            o[k] = os.path.abspath(o[k])
    return o


def log(*a):
    print("[proto]", *a, flush=True)


# ------------------------------------------------------------------ waves
class Sea:
    def __init__(self, sd):
        self.a = np.asarray(sd["a"])
        self.k = np.asarray(sd["k"])
        self.th = np.asarray(sd["th"])
        self.w = np.asarray(sd["w"])
        self.phi = np.asarray(sd["phi"])
        self.kx, self.ky = self.k * np.cos(self.th), self.k * np.sin(self.th)
        self.lam = 2 * np.pi / self.k

    def lod(self, h):
        """Per-vertex, per-component weight: 1 when the wavelength is >= 6
        local grid spacings, 0 when <= 3 (smoothstep in between)."""
        if h is None:
            return None
        r = self.lam[None, :] / np.maximum(h[:, None], 1e-6)
        s = np.clip((r - 3.0) / 3.0, 0.0, 1.0)
        return s * s * (3 - 2 * s)

    def fields(self, X, Y, t, h=None, want=("eta",), chunk=24000):
        """eta and/or eta_tt (surface vertical acceleration) at world X, Y.
        Phases in float64 for the large X*k products, cosines in float32."""
        out = {k: np.zeros(X.shape[0]) for k in want}
        for i in range(0, X.shape[0], chunk):
            sl = slice(i, i + chunk)
            if h is not None:
                act = self.lam > 3.0 * float(h[sl].min())
                if not act.any():
                    continue
                f = self.lod(h[sl])[:, act]
            else:
                act = np.ones(self.k.size, bool)
                f = 1.0
            ph = (X[sl, None] * self.kx[act] + Y[sl, None] * self.ky[act]
                  - (self.w[act] * t - self.phi[act]))
            c = np.cos(ph.astype(np.float32))
            if "eta" in want:
                out["eta"][sl] = (c * (self.a[act] * f)).sum(1)
            if "ett" in want:
                wa = self.a[act] * self.w[act] ** 2
                out["ett"][sl] = -((c * (wa * f)).sum(1))
        return out


# ------------------------------------------------------------------ scene
def clear_scene():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras,
                 bpy.data.lights, bpy.data.curves, bpy.data.worlds):
        for d in list(coll):
            coll.remove(d)


def new_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    pb = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    return m, nt, pb


def principled(name, color, rough, metal=0.0, coat=0.0, ior=1.5,
               transmission=0.0):
    m, nt, pb = new_mat(name)
    pb.inputs["Base Color"].default_value = (*color, 1.0)
    pb.inputs["Roughness"].default_value = rough
    pb.inputs["Metallic"].default_value = metal
    pb.inputs["Coat Weight"].default_value = coat
    pb.inputs["IOR"].default_value = ior
    pb.inputs["Transmission Weight"].default_value = transmission
    return m


def sock(coll, name, typ):
    return next(x for x in coll if x.name == name and x.type == typ)


def hull_materials():
    """0 topsides (orange paint, dark sheer stripe), 1 bottom (antifouling),
    2 deck (anti-slip: noise bump)."""
    top, nt, pb = new_mat("topsides")
    pb.inputs["Base Color"].default_value = (0.80, 0.81, 0.82, 1)
    pb.inputs["Roughness"].default_value = 0.22
    pb.inputs["Coat Weight"].default_value = 0.6
    pb.inputs["Coat Roughness"].default_value = 0.05
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    # navy stripe 0.40..0.47 m above the CG (deck at +0.55)
    a = nt.nodes.new("ShaderNodeMath"); a.operation = "GREATER_THAN"
    a.inputs[1].default_value = 0.40
    b = nt.nodes.new("ShaderNodeMath"); b.operation = "LESS_THAN"
    b.inputs[1].default_value = 0.47
    nt.links.new(sep.outputs[2], a.inputs[0])
    nt.links.new(sep.outputs[2], b.inputs[0])
    mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"
    nt.links.new(a.outputs[0], mul.inputs[0])
    nt.links.new(b.outputs[0], mul.inputs[1])
    mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = "RGBA"
    sock(mix.inputs, "A", "RGBA").default_value = (0.80, 0.81, 0.82, 1)
    sock(mix.inputs, "B", "RGBA").default_value = (0.025, 0.028, 0.032, 1)
    nt.links.new(mul.outputs[0], sock(mix.inputs, "Factor", "VALUE"))
    a2 = nt.nodes.new("ShaderNodeMath"); a2.operation = "GREATER_THAN"
    a2.inputs[1].default_value = -0.20
    b2 = nt.nodes.new("ShaderNodeMath"); b2.operation = "LESS_THAN"
    b2.inputs[1].default_value = -0.13
    nt.links.new(sep.outputs[2], a2.inputs[0])
    nt.links.new(sep.outputs[2], b2.inputs[0])
    mul2 = nt.nodes.new("ShaderNodeMath"); mul2.operation = "MULTIPLY"
    nt.links.new(a2.outputs[0], mul2.inputs[0])
    nt.links.new(b2.outputs[0], mul2.inputs[1])
    mix2 = nt.nodes.new("ShaderNodeMix"); mix2.data_type = "RGBA"
    nt.links.new(sock(mix.outputs, "Result", "RGBA"),
                 sock(mix2.inputs, "A", "RGBA"))
    sock(mix2.inputs, "B", "RGBA").default_value = (0.75, 0.10, 0.005, 1)
    nt.links.new(mul2.outputs[0], sock(mix2.inputs, "Factor", "VALUE"))
    nt.links.new(sock(mix2.outputs, "Result", "RGBA"), pb.inputs["Base Color"])

    bot = principled("bottom", (0.035, 0.035, 0.04), 0.45, coat=0.2)

    deck, nt, pb = new_mat("deck")
    pb.inputs["Base Color"].default_value = (0.62, 0.62, 0.60, 1)
    pb.inputs["Roughness"].default_value = 0.75
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 900.0
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.25
    nt.links.new(nz.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], pb.inputs["Normal"])
    return [top, bot, deck]


def hull_lines(H):
    """(x0, chine(xi), keel(xi), knots) for xi measured from the transom:
    the exported x_fine / yc_fine / zk_fine (the chine and keel of
    hydro/hull_geometry, the module the plant reads, at points dense where
    they bend; app/export.hull_dict, studies/visual_export.py), linear
    between them; an older scene without them: the plant's strip-centre
    arrays x / yc / zk closed at the transom and the stem (chine 0, keel
    held), as viewer/demo.html does -- never the v0 family formula of
    taper / stem_rise, which is right for v0's family member only (0.10 m
    off at the chine of planing_b). knots: the points the lines are linear
    between (build_hull puts a ring at each). x0 is where the bow starts
    (the last point at full beam): the bow is sampled more finely."""
    L, lcg = H["L"], H["lcg"]
    if "x_fine" in H:
        xf = np.asarray(H["x_fine"], float) + lcg
        yf, zf = (np.asarray(H[k], float) for k in ("yc_fine", "zk_fine"))
    elif all(k in H for k in ("x", "yc", "zk")):
        xt = float(H.get("x_transom", -lcg)) + lcg
        xs = float(H.get("x_stem", L - lcg)) + lcg
        xf = np.r_[xt, np.asarray(H["x"], float) + lcg, xs]
        yc, zk = np.asarray(H["yc"], float), np.asarray(H["zk"], float)
        yf = np.r_[yc[0], yc, 0.0]
        zf = np.r_[zk[0], zk, zk[-1]]
        print("render.py: the scene has no x_fine / yc_fine / zk_fine (an "
              "older export): the hull is drawn from the plant's station "
              "arrays; re-export for the exact chine and keel", flush=True)
    else:
        raise SystemExit("render.py: the scene's hull has neither x_fine / "
                         "yc_fine / zk_fine nor the station arrays x / yc / "
                         "zk: re-export it (app/export.blender_scene or "
                         "studies/visual_export.py)")
    full = np.nonzero(yf >= 0.999 * yf.max())[0]
    x0 = float(min(xf[full[-1]], L - 0.05))
    return (x0, lambda xi: np.interp(xi, xf, yf),
            lambda xi: np.interp(xi, xf, zf), xf)


def build_hull(H, n_par=50, n_bow=110):
    """Closed hull mesh in the body frame (CG origin, x fwd, y left, z up),
    from the exported hull lines (hull_lines), resampled finely (the plant
    uses 48 station centres). Rings: keel -> +y chine -> +y deck edge ->
    deck -> -y deck edge -> -y chine -> keel."""
    L, lcg = H["L"], H["lcg"]
    depth = H["depth"]
    tanb = math.tan(math.radians(H["beta_deg"]))
    x0, chine, keel, knots = hull_lines(H)
    xi = np.concatenate([np.linspace(0, x0, n_par, endpoint=False),
                         x0 + (L - x0) * np.sin(np.linspace(0, np.pi / 2,
                                                            n_bow))])
    # a ring at every exported point as well: the mesh's chine and keel are
    # then the exported polyline itself (within its line_tol of the module)
    xi = np.union1d(xi, knots[(knots >= 0.0) & (knots <= L)])
    xi = xi[np.r_[True, np.diff(xi) > 1e-9]]
    yc = chine(xi)
    zk = keel(xi)
    zc = zk + np.minimum(yc * tanb, depth)
    zd = zk + depth
    nb, ns, nd = 5, 4, 8
    rings, mats = [], []
    for i in range(len(xi)):
        pts = [(0.0, zk[i])]
        for t in np.linspace(0, 1, nb + 1)[1:]:             # bottom +y
            pts.append((t * yc[i], zk[i] + t * (zc[i] - zk[i])))
        for t in np.linspace(0, 1, ns + 1)[1:]:             # side +y
            pts.append((yc[i], zc[i] + t * (zd[i] - zc[i])))
        for t in np.linspace(0, 1, nd + 1)[1:]:             # deck
            pts.append((yc[i] * (1 - 2 * t), zd[i]))
        for t in np.linspace(0, 1, ns + 1)[1:]:             # side -y
            pts.append((-yc[i], zd[i] - t * (zd[i] - zc[i])))
        for t in np.linspace(0, 1, nb + 1)[1:-1]:           # bottom -y
            pts.append((-yc[i] * (1 - t), zc[i] - t * (zc[i] - zk[i])))
        rings.append([(xi[i] - lcg, y, z) for y, z in pts])
    seg = ([1] * nb + [0] * ns + [2] * nd + [0] * ns + [1] * nb)
    bm = bmesh.new()
    V = [[bm.verts.new(p) for p in r] for r in rings]
    m = len(rings[0])
    for i in range(len(rings) - 1):
        for j in range(m):
            jn = (j + 1) % m
            try:
                f = bm.faces.new((V[i][j], V[i + 1][j], V[i + 1][jn],
                                  V[i][jn]))
                f.material_index = seg[j]
            except ValueError:
                pass
    f = bm.faces.new(V[0][::-1])            # transom
    f.material_index = 0
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=1e-6)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new("hull")
    bm.to_mesh(me)
    bm.free()
    for mat in hull_materials():
        me.materials.append(mat)
    ob = bpy.data.objects.new("hull", me)
    bpy.context.scene.collection.objects.link(ob)
    ob.rotation_mode = "XYZ"
    return ob, dict(x_tr=-lcg, x_bow=L - lcg, z_deck=float(zd[0]),
                    yc_max=float(np.max(yc)), zk=float(zk[0]))


def add_box(name, parent, mat, cx, cy, cz, sx, sy, sz):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co = Vector((cx + v.co.x * sx, cy + v.co.y * sy, cz + v.co.z * sz))
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.parent = parent
    return ob


def add_tube(name, parent, mat, pts, radius):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = 3
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for p, q in zip(sp.points, pts):
        p.co = (*q, 1.0)
    cu.materials.append(mat)
    ob = bpy.data.objects.new(name, cu)
    bpy.context.scene.collection.objects.link(ob)
    ob.parent = parent
    return ob


def add_cyl(name, parent, mat, p0, p1, r, seg=24):
    p0, p1 = Vector(p0), Vector(p1)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=seg, radius1=r,
                          radius2=r, depth=(p1 - p0).length)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.parent = parent
    ob.location = (p0 + p1) / 2
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = (p1 - p0).to_track_quat("Z", "Y")
    return ob


def add_sphere(name, parent, mat, c, r, sz=1.0):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=r)
    for v in bm.verts:
        v.co.z *= sz
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.parent = parent
    ob.location = c
    return ob


def build_fittings(hull, g, H):
    """Cosmetic deck hardware for scale (not in the plant): centre console
    with windshield, leaning post, stern hatch, sensor mast with radome and
    antennas, bow rail, rub rail, jet nozzle."""
    white = principled("gelcoat", (0.50, 0.52, 0.55), 0.3, coat=0.5)
    dark = principled("rubber", (0.02, 0.02, 0.022), 0.6)
    steel = principled("steel", (0.85, 0.86, 0.88), 0.18, metal=1.0)
    glass = principled("glass", (0.6, 0.7, 0.75), 0.02, ior=1.5,
                       transmission=1.0)
    nozzle = principled("nozzle", (0.25, 0.25, 0.27), 0.35, metal=1.0)
    zd = g["z_deck"]
    b2 = g["yc_max"]
    add_box("console", hull, white, 0.15, 0.0, zd + 0.45, 0.75, 0.75, 0.9)
    add_box("console_top", hull, dark, 0.05, 0.0, zd + 0.905, 0.5, 0.6, 0.01)
    ws = add_box("windshield", hull, glass, 0.0, 0.0, 0.0, 0.02, 0.78, 0.38)
    ws.location = (0.50, 0.0, zd + 1.05)
    ws.rotation_euler = (0, math.radians(-30), 0)
    add_box("leaning_post", hull, white, -0.6, 0.0, zd + 0.4, 0.45, 0.85,
            0.8)
    add_box("stern_hatch", hull, white, g["x_tr"] + 0.38, 0.0, zd + 0.17,
            0.72, 2 * b2 - 0.2, 0.34)
    add_cyl("mast", hull, steel, (0.0, 0, zd + 0.9), (-0.05, 0, zd + 2.0),
            0.035)
    add_sphere("radome", hull, white, (-0.05, 0, zd + 2.12), 0.28, 0.45)
    add_cyl("ant1", hull, white, (0.0, 0.25, zd + 0.9), (0.0, 0.25,
                                                          zd + 2.3), 0.012)
    add_cyl("ant2", hull, white, (0.0, -0.25, zd + 0.9), (0.0, -0.25,
                                                           zd + 1.9), 0.012)
    # bow rail along the deck edge, 0.45 m high, from x 1.2 m to the stem
    lcg = H["lcg"]
    _, chine, _, _ = hull_lines(H)

    def half(xb):
        return float(chine(xb + lcg))
    xs = np.linspace(1.0, g["x_bow"] - 0.12, 24)
    rail = ([(x, half(x) - 0.06, zd + 0.45) for x in xs]
            + [(x, -(half(x) - 0.06), zd + 0.45) for x in xs[::-1]])
    add_tube("bow_rail", hull, steel, rail, 0.016)
    for x in np.linspace(1.0, g["x_bow"] - 0.5, 4):
        for sgn in (1, -1):
            y = sgn * (half(x) - 0.06)
            add_cyl("stanchion", hull, steel, (x, y, zd), (x, y, zd + 0.45),
                    0.013, seg=12)
    # rub rail around the sheer
    xr = np.linspace(g["x_tr"], g["x_bow"], 80)
    rub = ([(x, half(x) + 0.01, zd - 0.02) for x in xr]
           + [(x, -(half(x) + 0.01), zd - 0.02) for x in xr[::-1]])
    rub.append(rub[0])
    add_tube("rub_rail", hull, dark, rub, 0.035)
    # waterjet nozzle at the transom
    zn = -H["vcg"] + 0.15
    add_cyl("jet_nozzle", hull, nozzle, (g["x_tr"] + 0.02, 0, zn),
            (g["x_tr"] - 0.32, 0, zn), 0.095)


# ------------------------------------------------------------------ water
class PolarGrid:
    """Camera-centred polar grid: dense angular sector around the view
    direction, sparse elsewhere; geometric radial spacing so that the
    screen-space density is roughly even. Local spacing h per vertex drives
    the wave LOD filter."""

    def __init__(self, quality="normal"):
        if quality == "draft":
            nd, ns, q = 260, 40, 1.012
        elif quality == "fine":
            nd, ns, q = 560, 64, 1.006
        else:
            nd, ns, q = 420, 56, 1.008
        half = math.radians(64.0)
        dense = np.linspace(-half, half, nd, endpoint=False)
        sparse = np.linspace(half, 2 * np.pi - half, ns, endpoint=False)
        self.ang = np.concatenate([dense, sparse])
        dang = np.diff(np.concatenate([self.ang, [self.ang[0] + 2 * np.pi]]))
        dang = np.maximum(dang, np.roll(dang, 1))
        r0, r1 = 0.6, 25000.0
        nr = int(math.ceil(math.log(r1 / r0) / math.log(q))) + 1
        self.r = r0 * q ** np.arange(nr)
        self.nr, self.na = nr, len(self.ang)
        R, A = np.meshgrid(self.r, self.ang, indexing="ij")
        DA = np.broadcast_to(dang[None, :], R.shape)
        self.lx = (R * np.cos(A)).ravel()
        self.ly = (R * np.sin(A)).ravel()
        self.rad = R.ravel()
        self.h = np.maximum(R * DA, R * (q - 1)).ravel()
        # quads, angular index wraps
        i = np.arange(nr - 1)[:, None]
        j = np.arange(self.na)[None, :]
        jn = (j + 1) % self.na
        v00 = i * self.na + j
        v10 = (i + 1) * self.na + j
        v11 = (i + 1) * self.na + jn
        v01 = i * self.na + jn
        # counter-clockwise seen from above: normals point up (+Z)
        self.quads = np.stack([v00, v10, v11, v01], -1).reshape(-1, 4)
        log(f"water grid: {nr} rings x {self.na} angles = {nr * self.na} "
            f"vertices, {len(self.quads)} quads, ring ratio {q}")

    def world(self, cx, cy, az):
        c, s = math.cos(az), math.sin(az)
        return cx + c * self.lx - s * self.ly, cy + s * self.lx + c * self.ly


def make_water_mesh(grid):
    me = bpy.data.meshes.new("water")
    n = grid.nr * grid.na
    me.vertices.add(n)
    me.loops.add(grid.quads.size)
    me.polygons.add(len(grid.quads))
    me.loops.foreach_set("vertex_index", grid.quads.ravel().astype(np.int32))
    me.polygons.foreach_set("loop_start",
                            np.arange(0, grid.quads.size, 4, dtype=np.int32))
    co = np.zeros((n, 3), np.float32)
    me.vertices.foreach_set("co", co.ravel())
    me.update(calc_edges=True)
    me.polygons.foreach_set("use_smooth", np.ones(len(grid.quads), bool))
    for nm in ("foam", "cap", "wake"):
        me.attributes.new(nm, "FLOAT", "POINT")
    ob = bpy.data.objects.new("water", me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def foam_fields(sea, X, Y, grid, t, fr, track, H, r_foam=600.0):
    """(cap, wake, contact) foam masks in 0..1 per vertex. X, Y world
    coordinates (with the scene offset added back)."""
    n = X.shape[0]
    cap = np.zeros(n)
    wake = np.zeros(n)
    na = grid.na
    nrf = int(np.searchsorted(grid.r, r_foam))
    sr, sa = np.arange(0, nrf, 2), np.arange(0, na, 2)
    idx = (sr[:, None] * na + sa[None, :]).ravel()
    h = grid.h
    capsub = np.zeros(idx.size)
    # whitecaps: breaking crests are where the surface accelerates downward
    # hardest (Longuet-Higgins: a limiting crest reaches -g/2). The recorded
    # linear sea never gets there (std of eta_tt 0.075 g), so the threshold
    # is set by coverage instead: -eta_tt > ~2.8 std marks ~0.3-0.5 % of the
    # surface, Monahan & O'Muircheartaigh's (1980) whitecap cover at the
    # ~7 m/s wind of a fully developed Hs 1 m sea. Foam then fades over TAU
    # s where it was made (the crest moves on, the foam stays).
    sd = math.sqrt(float(np.sum(sea.a ** 2 * sea.w ** 4)) / 2.0)
    A0, A1, TAU = 2.6 * sd, 3.3 * sd, 1.8
    for lag in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        f = sea.fields(X[idx], Y[idx], t - lag, h[idx], want=("ett",))["ett"]
        crit = smoothstep(A0, A1, -f) * math.exp(-lag / TAU)
        capsub = np.maximum(capsub, crit)
    C = capsub.reshape(sr.size, sa.size)
    C = np.repeat(np.repeat(C, 2, 0), 2, 1)[:nrf, :na]
    cap[:nrf * na] = C.ravel()
    # wake: per vertex the track sample whose cross-line passes through it
    tr = np.asarray(track)
    tr = tr[tr[:, 0] <= t + 1e-6]
    T, TX, TY, TPSI, TU, TW = (tr[:, i] for i in range(6))
    hx, hy = np.cos(TPSI), np.sin(TPSI)
    b = H["b"]
    x_tr = -H["lcg"]
    # the transom position of each sample (the wake starts there)
    PX, PY = TX + x_tr * hx, TY + x_tr * hy
    xmin, xmax = PX.min() - 40, PX.max() + 40
    ymin, ymax = PY.min() - 40, PY.max() + 40
    sel = np.nonzero((X > xmin) & (X < xmax) & (Y > ymin) & (Y < ymax))[0]
    if sel.size:
        ii = np.empty(sel.size, np.int64)
        P = np.stack([PX, PY], 1).astype(np.float32)
        Q = np.stack([X[sel], Y[sel]], 1).astype(np.float32)
        for c0 in range(0, sel.size, 4000):
            dq = Q[c0:c0 + 4000, None, :] - P[None, :, :]
            ii[c0:c0 + 4000] = np.argmin((dq * dq).sum(-1), 1)
        dx, dy = X[sel] - PX[ii], Y[sel] - PY[ii]
        along = dx * hx[ii] + dy * hy[ii]
        lat = -dx * hy[ii] + dy * hx[ii]
        age = t - T[ii] - along / np.maximum(TU[ii], 1.0)   # ~ time behind
        age = np.maximum(age, 0.0)
        behind = (along < 0.4)                               # behind transom
        src = np.clip(TW[ii] * 1.6, 0, 1)                    # wetted share
        # centreline wash: jet + hull turbulence, widening; solid white only
        # while young, then a long-lived faint (lacy, aerated) trail
        wc = 0.45 * b + 0.22 * age + 0.03 * age ** 1.5
        prof = np.exp(-(lat / wc) ** 2)
        wash = prof * np.maximum(np.exp(-age / 1.6),
                                 0.62 * np.exp(-age / 22.0))
        # two diverging edge lines (~8 deg, the wake-angle of a fast hull)
        u = TU[ii]
        off_l = 0.5 * b + u * age * math.tan(math.radians(8.0)) * 0.9
        edge = (np.exp(-((np.abs(lat) - off_l) / (0.25 + 0.1 * age)) ** 2)
                * np.exp(-age / 3.0))
        wake[sel] = np.clip(np.maximum(wash, 0.42 * edge) * src * behind, 0,
                            1)
    # contact foam where the hull is in the water now (recorded immersion)
    d = np.asarray(fr["d"])
    xs = np.asarray(H["x"])
    yc = np.asarray(H["yc"])
    tanb = math.tan(math.radians(H["beta_deg"]))
    c, s = math.cos(fr["yaw"]), math.sin(fr["yaw"])
    dx, dy = X - fr["x"], Y - fr["y"]
    xb = dx * c + dy * s
    yb = -dx * s + dy * c
    box = np.nonzero((xb > xs[0] - 0.3) & (xb < xs[-1] + 1.0)
                     & (np.abs(yb) < H["b"] / 2 + 2.5))[0]
    contact = np.zeros(n)
    if box.size and (d > 0).any():
        dxs = xs[1] - xs[0]
        k = np.clip(np.round((xb[box] - xs[0]) / dxs).astype(int), 0,
                    len(xs) - 1)
        dk = np.maximum(d[k], 0.0)
        cw = np.minimum(0.5 * np.pi * dk / tanb, yc[k])   # Wagner half-beam
        edge = np.maximum(np.abs(yb[box]) - np.maximum(cw, yc[k]), 0.0)
        strength = np.clip(dk / 0.12, 0, 1) * np.clip(fr["u"] / 8.0, 0, 1)
        contact[box] = strength * np.exp(-edge / (0.35 + 0.6 * dk))
    return cap, wake, contact


def water_material(kind, sea_dir=math.pi):
    """Water surface. Two scales of slope on top of the exact geometry:
    the recorded spectrum stops at 4.3 m wavelength, a real Hs 1 m wind sea
    (Pierson-Moskowitz wind ~7 m/s) has a total mean-square slope of ~0.04
    (Cox & Munk 1954) of which the recorded components give 0.006. The rest
    is added as (a) a bump of anisotropic noise near the camera and (b)
    microfacet roughness, which takes over with distance as the bump is
    faded out (GGX alpha ~ sqrt(2) x rms slope -> Blender roughness ~0.5)."""
    m, nt, pb = new_mat("water")
    N, L = nt.nodes, nt.links
    out = next(n for n in N if n.type == "OUTPUT_MATERIAL")
    pb.inputs["IOR"].default_value = 1.333
    pb.inputs["Specular IOR Level"].default_value = 0.5

    def math_node(op, a, b=None, clamp=False):
        n = N.new("ShaderNodeMath")
        n.operation = op
        n.use_clamp = clamp
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = v
            else:
                L.new(v, n.inputs[i])
        return n.outputs[0]

    def map_range(v, a0, a1, b0, b1):
        n = N.new("ShaderNodeMapRange")
        n.inputs["From Min"].default_value = a0
        n.inputs["From Max"].default_value = a1
        n.inputs["To Min"].default_value = b0
        n.inputs["To Max"].default_value = b1
        L.new(v, n.inputs["Value"])
        return n.outputs["Result"]

    cam = N.new("ShaderNodeCameraData")
    dist = cam.outputs["View Distance"]
    tc = N.new("ShaderNodeTexCoord")
    # coordinates rotated so that local x runs along the wave direction and
    # stretched across it (short wind waves have long-ish crests)
    mp = N.new("ShaderNodeMapping")
    mp.inputs["Rotation"].default_value = (0, 0, -sea_dir)
    mp.inputs["Scale"].default_value = (1.0, 0.45, 1.0)
    L.new(tc.outputs["Object"], mp.inputs["Vector"])
    n1 = N.new("ShaderNodeTexNoise")
    n1.inputs["Scale"].default_value = 0.55
    n1.inputs["Detail"].default_value = 4.5
    n1.inputs["Roughness"].default_value = 0.58
    n1.inputs["Lacunarity"].default_value = 2.1
    L.new(mp.outputs["Vector"], n1.inputs["Vector"])
    n3 = N.new("ShaderNodeTexNoise")
    n3.inputs["Scale"].default_value = 3.0
    n3.inputs["Detail"].default_value = 1.5
    n3.inputs["Roughness"].default_value = 0.5
    L.new(mp.outputs["Vector"], n3.inputs["Vector"])
    height = math_node("ADD", n1.outputs["Fac"],
                       math_node("MULTIPLY", n3.outputs["Fac"], 0.25))
    bump = N.new("ShaderNodeBump")
    bump.inputs["Distance"].default_value = 0.30
    L.new(map_range(dist, 6.0, 220.0, 1.0, 0.0), bump.inputs["Strength"])
    L.new(height, bump.inputs["Height"])
    # roughness: capillary scale near, all unresolved slopes far
    rough = map_range(dist, 10.0, 600.0, 0.10, 0.42)
    # wake: aerated (bubbly) water under the foam is a lighter turquoise
    att = {}
    for nm in ("cap", "wake", "foam"):
        a = N.new("ShaderNodeAttribute")
        a.attribute_name = nm
        att[nm] = a.outputs["Fac"]
    aer = math_node("MULTIPLY", math_node("MAXIMUM", att["wake"],
                                          att["foam"]), 1.6, clamp=True)
    if kind == "volume":
        pb.inputs["Base Color"].default_value = (1, 1, 1, 1)
        pb.inputs["Transmission Weight"].default_value = 1.0
    else:
        mix = N.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        # deep-water upwelling: water-leaving reflectance of a few % in the
        # blue-green; the rest of the look is the Fresnel reflection
        sock(mix.inputs, "A", "RGBA").default_value = (0.003, 0.018, 0.026, 1)
        sock(mix.inputs, "B", "RGBA").default_value = (0.03, 0.17, 0.17, 1)
        L.new(aer, sock(mix.inputs, "Factor", "VALUE"))
        L.new(sock(mix.outputs, "Result", "RGBA"), pb.inputs["Base Color"])
    # foam: vertex masks broken up by procedural noise and a lace pattern
    n2 = N.new("ShaderNodeTexNoise")
    n2.inputs["Scale"].default_value = 0.45
    n2.inputs["Detail"].default_value = 8.0
    n2.inputs["Roughness"].default_value = 0.65
    L.new(tc.outputs["Object"], n2.inputs["Vector"])
    vor = N.new("ShaderNodeTexVoronoi")
    vor.feature = "DISTANCE_TO_EDGE"
    vor.inputs["Scale"].default_value = 1.6
    L.new(tc.outputs["Object"], vor.inputs["Vector"])
    vor2 = N.new("ShaderNodeTexVoronoi")
    vor2.feature = "DISTANCE_TO_EDGE"
    vor2.inputs["Scale"].default_value = 6.0
    L.new(tc.outputs["Object"], vor2.inputs["Vector"])
    lace1 = math_node("SUBTRACT", 1.0, math_node("MULTIPLY",
                                                 vor.outputs["Distance"], 4.0),
                      clamp=True)
    lace2 = math_node("SUBTRACT", 1.0, math_node("MULTIPLY",
                                                 vor2.outputs["Distance"],
                                                 3.0), clamp=True)
    lace = math_node("MAXIMUM", lace1, math_node("MULTIPLY", lace2, 0.8))
    # streak noise: foam lines drawn out along the wave/wind direction
    mps = N.new("ShaderNodeMapping")
    mps.inputs["Rotation"].default_value = (0, 0, -sea_dir)
    mps.inputs["Scale"].default_value = (0.25, 1.0, 1.0)
    L.new(tc.outputs["Object"], mps.inputs["Vector"])
    ns = N.new("ShaderNodeTexNoise")
    ns.inputs["Scale"].default_value = 1.4
    ns.inputs["Detail"].default_value = 8.0
    ns.inputs["Roughness"].default_value = 0.7
    L.new(mps.outputs["Vector"], ns.inputs["Vector"])
    # whitecaps: a broken, streaky patch, solid only at the very crest
    cb = math_node("MULTIPLY", att["cap"], math_node(
        "ADD", 0.25, math_node("MULTIPLY", ns.outputs["Fac"], 1.5)))
    cap_solid = map_range(cb, 0.85, 1.15, 0.0, 1.0)
    cap_lacy = math_node("MULTIPLY", map_range(cb, 0.35, 0.85, 0.0, 1.0),
                         lace)
    cap_m = math_node("MAXIMUM", cap_solid, cap_lacy)
    # wake and contact foam: solid white wash where the mask is high,
    # lace and streaks where it fades
    wk = math_node("MAXIMUM", att["wake"], att["foam"])
    nb = math_node("MULTIPLY", math_node("SUBTRACT", n2.outputs["Fac"], 0.5),
                   1.2)
    wb = math_node("ADD", wk, nb)
    wake_solid = map_range(wb, 0.8, 1.1, 0.0, 1.0)
    wake_lacy = math_node("MULTIPLY", map_range(wb, 0.12, 0.75, 0.0, 1.0),
                          math_node("MAXIMUM", lace, math_node(
                              "MULTIPLY", ns.outputs["Fac"], 0.6)))
    wake_m = math_node("MAXIMUM", wake_solid, wake_lacy)
    mask = math_node("MAXIMUM", cap_m, wake_m, clamp=True)
    mask = math_node("MULTIPLY", mask, 0.95)
    foam = N.new("ShaderNodeBsdfPrincipled")
    fmix = N.new("ShaderNodeMix")
    fmix.data_type = "RGBA"
    sock(fmix.inputs, "A", "RGBA").default_value = (0.50, 0.58, 0.60, 1)
    sock(fmix.inputs, "B", "RGBA").default_value = (0.86, 0.89, 0.90, 1)
    L.new(map_range(math_node("ADD", lace, math_node(
        "MULTIPLY", n2.outputs["Fac"], 0.8)), 0.4, 1.3, 0.0, 1.0),
        sock(fmix.inputs, "Factor", "VALUE"))
    L.new(sock(fmix.outputs, "Result", "RGBA"), foam.inputs["Base Color"])
    foam.inputs["Roughness"].default_value = 0.9
    fb = N.new("ShaderNodeBump")
    fb.inputs["Distance"].default_value = 0.05
    fb.inputs["Strength"].default_value = 0.6
    L.new(math_node("ADD", lace, n3.outputs["Fac"]), fb.inputs["Height"])
    L.new(fb.outputs["Normal"], foam.inputs["Normal"])
    L.new(rough, pb.inputs["Roughness"])
    L.new(bump.outputs["Normal"], pb.inputs["Normal"])
    mixs = N.new("ShaderNodeMixShader")
    L.new(mask, mixs.inputs["Fac"])
    L.new(pb.outputs["BSDF"], mixs.inputs[1])
    L.new(foam.outputs["BSDF"], mixs.inputs[2])
    L.new(mixs.outputs["Shader"], out.inputs["Surface"])
    if kind == "volume":
        vol = N.new("ShaderNodeVolumePrincipled")
        # open-ocean water: weak scattering (albedo of the scattered light
        # blue-green), absorption strongest in the red; the slab bottom is
        # black, so deep water ends up dark navy as observed
        vol.inputs["Color"].default_value = (0.02, 0.16, 0.25, 1)
        vol.inputs["Density"].default_value = 0.025
        vol.inputs["Absorption Color"].default_value = (0.45, 0.85, 0.93, 1)
        L.new(vol.outputs["Volume"], out.inputs["Volume"])
    return m


def spray_points(t_now, steps, H, sea, off, seed=7, k_emit=26000.0):
    """Spray droplets alive at t_now, (positions (n, 3) in scene
    coordinates, radii). Physically motivated, not simulated:
    2D+t / Wagner: in an earth-fixed transverse plane a V section entering
    the water at rate V pushes its spray root out at dc/dt = (pi/2) V /
    tan(beta); the jet leaves the hull at the chine, so every plant step
    and every wetted station with V > 0 emits droplets at the chine, in the
    earth-fixed plane (no forward velocity), with speed ~1.4 dc/dt capped at
    22 m/s, launched outward and up, then ballistic with a linear air drag
    (time constant 0.25-1 s); droplets that fall back below the exact
    surface are dropped. Emission count ~ V dt dx (tuned constant k_emit),
    droplet radii 4-16 mm (inflated: real spray drops are ~0.1-2 mm, a
    pixel here is ~2 cm at the boat) so a sheet reads at 1280x720."""
    rng = np.random.default_rng(seed)
    tanb = math.tan(math.radians(H["beta_deg"]))
    xs, yc, zk = (np.asarray(H[k]) for k in ("x", "yc", "zk"))
    zc = zk + yc * tanb
    dx = xs[1] - xs[0]
    P, R = [], []
    dt = 0.02
    for st in steps:
        age0 = t_now - st["t"]
        if age0 < 0 or age0 > 1.2:
            continue
        d, V = np.asarray(st["d"]), np.asarray(st["V"])
        on = (d > 0) & (V > 0)
        if not on.any():
            continue
        dcdt = 0.5 * np.pi * V / tanb
        lam = k_emit * V * dt * dx
        cnt = rng.poisson(np.where(on, lam, 0.0))
        cr, sr = math.cos(st["roll"]), math.sin(st["roll"])
        cp, sp = math.cos(st["pitch"]), math.sin(st["pitch"])
        cy, sy = math.cos(st["yaw"]), math.sin(st["yaw"])
        Rm = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array(
            [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @ np.array(
            [[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        for side in (1.0, -1.0):
            ii = np.repeat(np.arange(len(xs)), cnt)
            m = ii.size
            if m == 0:
                continue
            body = np.stack([xs[ii] + rng.uniform(-dx / 2, dx / 2, m),
                             side * yc[ii], zc[ii]], 0)
            p0 = (Rm @ body).T + np.array([st["x"] - off[0],
                                           st["y"] - off[1], st["z"]])
            lat = Rm @ np.array([0.0, side, 0.0])
            lat[2] = 0.0
            lat /= max(np.linalg.norm(lat), 1e-9)
            vj = np.minimum(1.4 * dcdt[ii], 22.0) * rng.uniform(0.35, 1.0, m)
            gam = np.radians(H["beta_deg"] + rng.uniform(-8.0, 40.0, m))
            fwd = np.array([math.cos(st["yaw"]), math.sin(st["yaw"]), 0.0])
            v0 = (lat[None, :] * (vj * np.cos(gam))[:, None]
                  + np.array([0, 0, 1.0])[None, :] * (vj * np.sin(gam))[:, None]
                  + fwd[None, :] * rng.normal(0.0, 0.8, m)[:, None])
            tau = rng.uniform(0.25, 1.0, m)
            age = age0 + rng.uniform(0, dt, m)
            e = np.exp(-age / tau)
            gv = np.array([0, 0, -G])
            pos = (p0 + (tau * (1 - e))[:, None] * (v0 - tau[:, None] * gv)
                   + (tau * age)[:, None] * gv)
            P.append(pos)
            R.append(0.004 + 0.012 * rng.beta(1.2, 3.0, m))
    if not P:
        return np.zeros((0, 3)), np.zeros(0)
    P, R = np.concatenate(P), np.concatenate(R)
    eta = sea.fields(P[:, 0] + off[0], P[:, 1] + off[1], t_now,
                     want=("eta",))["eta"]
    keep = P[:, 2] > eta - 0.02
    return P[keep], R[keep]


# ------------------------------------------------------------------ world
def setup_world(o, look_az):
    sc = bpy.context.scene
    w = bpy.data.worlds.new("sky")
    sc.world = w
    w.use_nodes = True
    nt = w.node_tree
    bg = next(n for n in nt.nodes if n.type == "BACKGROUND")
    sky = nt.nodes.new("ShaderNodeTexSky")
    try:
        sky.sky_type = "MULTIPLE_SCATTERING"
    except TypeError:
        sky.sky_type = "NISHITA"
    sun_az = look_az + math.radians(o["sun_rel"])
    sky.sun_elevation = math.radians(o["sun_el"])
    # calibrated on this Blender: rotation 0 = sun at +Y, azimuth (CCW from
    # +X) = 90 deg - rotation
    sky.sun_rotation = (math.pi / 2 - sun_az) % (2 * math.pi)
    sky.sun_disc = True
    try:
        sky.air_density = 1.0
        sky.aerosol_density = 1.6       # a bit of marine haze
        sky.ozone_density = 1.0
    except Exception:
        pass
    nt.links.new(sky.outputs[0], bg.inputs[0])
    bg.inputs["Strength"].default_value = 1.0
    sd = (math.cos(math.radians(o["sun_el"])) * math.cos(sun_az),
          math.cos(math.radians(o["sun_el"])) * math.sin(sun_az),
          math.sin(math.radians(o["sun_el"])))
    log("sun direction (towards the sun) %.2f %.2f %.2f, sky %s" % (
        *sd, sky.sky_type))
    return sd


# ------------------------------------------------------------------ setup
def setup_render(o):
    sc = bpy.context.scene
    rx, ry = (int(v) for v in o["res"].split("x"))
    sc.render.resolution_x, sc.render.resolution_y = rx, ry
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    try:
        sc.view_settings.view_transform = "AgX"
        sc.view_settings.look = o["look"]
    except TypeError as e:
        log("colour management:", e)
    sc.view_settings.exposure = o["exposure"]
    sc.render.use_motion_blur = o["blur"] > 0
    sc.render.motion_blur_shutter = max(o["blur"], 0.01)
    if o["engine"] == "CYCLES":
        sc.render.engine = "CYCLES"
        prefs = bpy.context.preferences.addons["cycles"].preferences
        dev = None
        for t in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = t
                prefs.get_devices()
                ok = [d for d in prefs.devices if d.type == t]
                if ok:
                    for d in prefs.devices:
                        d.use = d.type == t
                    dev = t
                    break
            except TypeError:
                continue
        sc.cycles.device = "GPU" if dev else "CPU"
        log("Cycles device", dev or "CPU")
        sc.cycles.samples = o["samples"]
        sc.cycles.use_adaptive_sampling = True
        sc.cycles.adaptive_threshold = 0.015
        sc.cycles.use_denoising = True
        try:
            sc.cycles.denoiser = "OPTIX" if dev == "OPTIX" else \
                "OPENIMAGEDENOISE"
        except TypeError:
            pass
        sc.cycles.max_bounces = 8
        sc.cycles.glossy_bounces = 4
        sc.cycles.transmission_bounces = 8
        sc.cycles.volume_bounces = 2
        sc.cycles.diffuse_bounces = 3
        sc.cycles.sample_clamp_indirect = 8.0
        sc.cycles.caustics_reflective = False
        sc.cycles.caustics_refractive = False
        sc.cycles.use_animated_seed = True
    else:
        sc.render.engine = "BLENDER_EEVEE"
        ee = sc.eevee
        ee.taa_render_samples = o["samples"]
        for k, v in (("use_raytracing", bool(o["ee_rt"])),
                     ("use_shadows", True),
                     ("shadow_ray_count", 2), ("shadow_step_count", 8)):
            try:
                setattr(ee, k, v)
            except Exception as e:
                log("eevee", k, e)
        try:
            ee.ray_tracing_options.resolution_scale = "1"
        except Exception:
            pass
        w = sc.world
        for k, v in (("sun_threshold", 5.0), ("sun_angle", math.radians(0.6)),
                     ("use_sun_shadow", True)):
            try:
                setattr(w, k, v)
            except Exception as e:
                log("world", k, e)


def smooth_series(v, n):
    if n <= 1:
        return np.asarray(v)
    k = np.ones(n) / n
    pad = np.pad(np.asarray(v), (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(pad, k, mode="valid")


def chase_camera(o, frames, off):
    """The viewer's chase view: a world-fixed azimuth/elevation offset at a
    distance, looking at the CG + 0.32 m; the target height is the CG height
    smoothed over ~1 s so the jump shows against the frame."""
    fps = 30
    zs = smooth_series([f["z"] for f in frames], fps)
    a, p, d = o["cam_az"], o["cam_el"], o["cam_dist"]
    offv = Vector((math.cos(a) * math.cos(p), math.sin(a) * math.cos(p),
                   math.sin(p))) * d
    poses = []
    for f, z in zip(frames, zs):
        tgt = Vector((f["x"] - off[0], f["y"] - off[1], z + 0.32))
        pos = tgt + offv
        q = (tgt - pos).to_track_quat("-Z", "Y")
        poses.append((pos, q))
    return poses, a + math.pi


def encode_mp4(frames_dir, mp4, fps):
    """PNG sequence -> H.264 MP4 with Blender's own FFmpeg (sequencer)."""
    files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    if not files:
        log("encode: no frames in", frames_dir)
        return
    sc = bpy.data.scenes.new("encode")
    bpy.context.window_manager.windows[0].scene = sc         if bpy.context.window_manager.windows else None
    im = bpy.data.images.load(os.path.join(frames_dir, files[0]))
    sc.render.resolution_x, sc.render.resolution_y = im.size
    sc.render.resolution_percentage = 100
    bpy.data.images.remove(im)
    sc.render.fps = int(fps)
    ed = sc.sequence_editor_create()
    coll = ed.strips if hasattr(ed, "strips") else ed.sequences
    st = coll.new_image("clip", os.path.join(frames_dir, files[0]), 1, 1)
    for f in files[1:]:
        st.elements.append(f)
    sc.frame_start, sc.frame_end = 1, len(files)
    ims = sc.render.image_settings
    if hasattr(ims, "media_type"):
        try:
            ims.media_type = "VIDEO"
        except Exception as e:
            log("media_type", e)
    ims.file_format = "FFMPEG"
    ff = sc.render.ffmpeg
    ff.format = "MPEG4"
    ff.codec = "H264"
    try:
        ff.constant_rate_factor = "HIGH"
    except Exception:
        pass
    sc.view_settings.view_transform = "Standard"
    sc.render.filepath = mp4
    sc.render.use_sequencer = True
    t0 = time.time()
    bpy.ops.render.render(animation=True, scene=sc.name)
    log("encoded %d frames -> %s (%.1f s)" % (len(files), mp4,
                                               time.time() - t0))


def main():
    o = parse()
    if o["encode"]:
        encode_mp4(o["encode"], o["mp4"], 30)
        return
    t_all = time.time()
    S = json.load(open(o["scene"]))
    frames, H, track = S["frames"], S["hull"], S["track"]
    sea = Sea(S["sea"])
    meta = S["meta"]
    if o["frame"] is None:
        o["frame"] = meta["stills"][0]["frame"]
    # scene offset: keep Blender coordinates small (float32 meshes)
    off = (round(frames[0]["x"]), round(frames[0]["y"]))
    log(f"run {meta['run']} ({meta['source']}), offset {off}")
    clear_scene()
    sc = bpy.context.scene
    sc.render.fps = int(meta["fps"])
    hull, g = build_hull(H)
    build_fittings(hull, g, H)
    cam_poses, look_az = chase_camera(o, frames, off)
    cam_d = bpy.data.cameras.new("chase")
    cam_d.sensor_fit = "VERTICAL"
    cam_d.angle_y = math.radians(o["fov"])
    cam_d.clip_start, cam_d.clip_end = 0.3, 40000.0
    cam = bpy.data.objects.new("chase", cam_d)
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.rotation_mode = "QUATERNION"
    hull.rotation_mode = "XYZ"
    # keyframes for every frame (Blender frame = index + 1): motion blur
    for i, f in enumerate(frames):
        hull.location = (f["x"] - off[0], f["y"] - off[1], f["z"])
        hull.rotation_euler = (f["roll"], f["pitch"], f["yaw"])
        hull.keyframe_insert("location", frame=i + 1)
        hull.keyframe_insert("rotation_euler", frame=i + 1)
        cam.location, cam.rotation_quaternion = cam_poses[i]
        cam.keyframe_insert("location", frame=i + 1)
        cam.keyframe_insert("rotation_quaternion", frame=i + 1)
    for ob in (hull, cam):
        if ob.animation_data and ob.animation_data.action:
            try:
                for fc in ob.animation_data.action.fcurves:
                    for kp in fc.keyframe_points:
                        kp.interpolation = "LINEAR"
            except AttributeError:
                pass   # layered actions (5.x): Bezier through 30 fps keys
    setup_world(o, look_az)
    grid = PolarGrid(o["grid"])
    water = make_water_mesh(grid)
    a2 = np.asarray(S["sea"]["a"]) ** 2
    th = np.asarray(S["sea"]["th"])
    sea_dir = math.atan2((a2 * np.sin(th)).sum(), (a2 * np.cos(th)).sum())
    wmat = water_material(o["water"] if o["engine"] == "CYCLES"
                          else "opaque", sea_dir)
    water.data.materials.append(wmat)
    if o["water"] == "volume" and o["engine"] == "CYCLES":
        sol = water.modifiers.new("slab", "SOLIDIFY")
        sol.thickness = 12.0
        sol.offset = -1.0
        sol.use_even_offset = False
        black = principled("seabed", (0.0, 0.0, 0.0), 1.0)
        water.data.materials.append(black)
        sol.material_offset_rim = 1
        sol.material_offset = 1     # the slab's bottom shell is black
    setup_render(o)
    pcd = bpy.data.pointclouds.new("spray")
    pcd.attributes.new("radius", "FLOAT", "POINT")
    # EEVEE refracts tiny transmissive points to near invisibility: there
    # the droplets are plain white-ish dielectric spheres
    spm = principled("spray", (0.90, 0.93, 0.95), 0.2, ior=1.333,
                     transmission=0.55 if o["engine"] == "CYCLES" else 0.0)
    pcd.materials.append(spm)
    spray = bpy.data.objects.new("spray", pcd)
    sc.collection.objects.link(spray)
    steps = S.get("steps", [])

    def update(i):
        f = frames[i]
        sc.frame_set(i + 1)
        pos = cam_poses[i][0]
        t0 = time.time()
        Xw, Yw = grid.world(pos.x + off[0], pos.y + off[1], look_az)
        F = sea.fields(Xw, Yw, f["t"], grid.h, want=("eta",))
        co = np.stack([Xw - off[0], Yw - off[1], F["eta"]], -1)
        me = water.data
        me.vertices.foreach_set("co", co.astype(np.float32).ravel())
        t1 = time.time()
        cap, wake, contact = foam_fields(sea, Xw, Yw, grid, f["t"], f,
                                         track, H)
        log("  surface %.1f s, foam %.1f s" % (t1 - t0, time.time() - t1))
        me.attributes["cap"].data.foreach_set("value", cap.astype(np.float32))
        me.attributes["wake"].data.foreach_set("value",
                                               wake.astype(np.float32))
        me.attributes["foam"].data.foreach_set("value",
                                               contact.astype(np.float32))
        me.update()
        if o["spray"] and steps:
            ts = time.time()
            P, R = spray_points(f["t"], steps, H, sea, off)
            pcd.resize(max(len(P), 1))
            if len(P) == 0:
                P, R = np.array([[0.0, 0.0, -50.0]]), np.array([0.001])
            pcd.attributes["position"].data.foreach_set(
                "vector", P.astype(np.float32).ravel())
            pcd.attributes["radius"].data.foreach_set(
                "value", R.astype(np.float32))
            pcd.update_tag()
            log("  spray: %d droplets (%.1f s)" % (len(P), time.time() - ts))
        log("frame %d t %.3f: water %.1f s; whitecap share (r<300 m) %.3f, "
            "wake verts %d, contact verts %d" % (
                i, f["t"], time.time() - t0,
                float((cap[grid.rad < 300] > 0.5).mean()),
                int((wake > 0.3).sum()), int((contact > 0.3).sum())))
        return Xw, Yw

    def check_clearance(i):
        """Hull bottom vertices in world space (Blender's own transform)
        against the exact wave sum there: verifies pose, sign and frame."""
        f = frames[i]
        dg = bpy.context.evaluated_depsgraph_get()
        he = hull.evaluated_get(dg)
        M = np.array(he.matrix_world)
        V = np.array([v.co[:] for v in he.data.vertices])
        Wv = (M[:3, :3] @ V.T).T + M[:3, 3]
        eta = sea.fields(Wv[:, 0] + off[0], Wv[:, 1] + off[1], f["t"],
                         want=("eta",))["eta"]
        cl = Wv[:, 2] - eta
        log("clearance check t %.3f: hull vertices below the exact surface "
            "%d of %d, min clearance %.3f m, keel immersion (plant) %.3f.."
            "%.3f m, pitch %.1f deg (bow up < 0), CG z %.2f m" % (
                f["t"], int((cl < 0).sum()), len(cl), cl.min(),
                min(f["d"]), max(f["d"]), math.degrees(f["pitch"]), f["z"]))

    def stats(path):
        if not o["stats"]:
            return
        im = bpy.data.images.load(path)
        px = np.array(im.pixels[:]).reshape(im.size[1], im.size[0], -1)
        rgb = px[..., :3]
        lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
        log("image %s: mean %.3f, clipped %.2f %%, black %.2f %%, top-row "
            "mean %.3f" % (os.path.basename(path), lum.mean(),
                           100 * (rgb.max(-1) > 0.995).mean(),
                           100 * (lum < 0.01).mean(), lum[-10:].mean()))
        bpy.data.images.remove(im)

    if o["clip"]:
        a, b = (int(v) for v in o["clip"].split(":"))
        outdir = o["out"] or os.path.join(HERE, "clip_frames")
        os.makedirs(outdir, exist_ok=True)
        times = []
        tw = []
        for i in range(a, b + 1):
            fp = os.path.join(outdir, "f%04d.png" % i)
            if o["resume"] and os.path.exists(fp):
                continue
            t0 = time.time()
            update(i)
            tw.append(time.time() - t0)
            sc.render.filepath = fp
            t0 = time.time()
            bpy.ops.render.render(write_still=True)
            times.append(time.time() - t0)
            log("rendered frame %d in %.1f s (+ %.1f s water/foam/spray)" % (
                i, times[-1], tw[-1]))
        if times:
            log("CLIP %s %s %d spp: %d frames, render %.1f s/frame, scene "
                "update %.1f s/frame, total %.0f s" % (
                    o["engine"], o["res"], o["samples"], len(times),
                    np.mean(times), np.mean(tw), time.time() - t_all))
        encode_mp4(outdir, o["mp4"], meta["fps"])
    else:
        i = o["frame"]
        update(i)
        check_clearance(i)
        if o["blend"]:
            bpy.ops.wm.save_as_mainfile(filepath=o["blend"])
        out = o["out"] or os.path.join(
            HERE, "still_%s.png" % o["engine"].lower())
        sc.render.filepath = out
        t0 = time.time()
        bpy.ops.render.render(write_still=True)
        dt = time.time() - t0
        log("RENDER %s %s %s %d spp: %.1f s (total with setup %.1f s) -> %s"
            % (o["engine"], o["res"], o["water"], o["samples"], dt,
               time.time() - t_all, out))
        stats(out)


main()
