#!/usr/bin/env python3
"""
The vessel's profile -- module M6 of the modular split
(learn/meta/ANYSPEED_PRINCIPLES_2026-10-05.md section 1; ACCEPTANCE N3):
what the controllers and the evaluation need to know about a built plant,
computed from the plant and cached under the plant's physics sha1.

    prof = VesselProfile("planing_b@as1")
    prof.limits()          thrust and nozzle range, top speed, draught, GM,
                           the GZ curve's angles, steps
    prof.steady()          running attitude and calm resistance at Froude
                           numbers FN_STEADY, with the captive run's residual
    prof.reduced_model()   control/reduced.ReducedModel.identify on the calm
                           plant (what sim/lofi.template identifies), with
                           lofi's fingerprint of that plant
    prof.turn_table()      steady full-nozzle turn rates, calm water, on the
                           GZ twin, in a Froude-exact protocol (below)

WHERE IT IS CACHED, AND WHY THERE
studies/_cache/profiles/<hull key>/<physics_sha1[:16]>/<part>.json, the sha1
being sim/physics.key_sha1(hull): the code hash of the hull's physics
version, the hull's manifest (every input, geometry and plant constants
included) and the preset's switches. A part is read back only if its stored
meta equals the current one: physics_sha1, code sha1, hull sha1, switches
AND the protocol (PROTOCOL_CODE's bytes -- this file, the identification
control/reduced.py and sim/lofi.py --, lofi.IDENT_VERSION and the protocol
constants FN_STEADY, THROTTLES, TURN_T, L_REF); anything else is recomputed.
A part's call arguments are in its name: steady() at a Froude-number list
other than FN_STEADY is stored as steady_<sha1 of the list>. Nothing here
writes the project's legacy caches (studies/_cache/lofi_<name>.json,
studies/_cache/final_eval/turn_gz.json): those are keyed by name alone and
v0 depends on them bit for bit.

THE TURNING PROTOCOL, FROUDE-EXACT
studies/final_eval.phase_turn runs 20 s straight, turns until 45 s and
averages over t >= 37 s, comparing an accumulated time with those marks; on
a Froude twin the accumulated float time crosses a mark one step earlier or
later than on the base hull (studies/_cache/anyspeed/p0: base turns at step
1001, the 1.5x twin at 1000). Here the marks are Froude-scaled from the
reference hull (L_REF, hydro/vessel_spec: 5.4 m, scarab195's keel -- a
protocol choice: t x sqrt(L / L_REF)) and
turned into integer step counts round(t / dt) once, so a similar hull runs
the same steps. The start speed is 0.8 u_design, as there. For scarab195
the numbers therefore differ from turn_gz.json by the one-step offset of
the turn start; the table of record for v0 stays final_eval's.
"""
import hashlib
import json
import math
import os
import time

import numpy as np

from hydro.vessel_spec import L_REF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "studies", "_cache", "profiles")
G = 9.81
# the code whose bytes are the profile's protocol (module docstring)
PROTOCOL_CODE = ("sim/vessel_profile.py", "control/reduced.py",
                 "sim/lofi.py")
TURN_T = dict(straight=20.0, record=37.0, end=45.0)
THROTTLES = (0.3, 0.45, 0.6, 0.75, 0.9, 1.0)
# length Froude numbers of the steady curve (scarab195: 2.9 to 15 m/s)
FN_STEADY = (0.4, 0.8, 1.1, 1.4, 1.767, 2.061)


def _calm():
    from sim.test_vessel import Monochromatic
    return Monochromatic(1.0, 0.0)


def protocol():
    """What the parts' numbers depend on besides the plant: the sha1 of
    PROTOCOL_CODE's bytes, the identification's version and the protocol
    constants."""
    from sim import lofi
    h = hashlib.sha1()
    for f in PROTOCOL_CODE:
        h.update(f.encode())
        with open(os.path.join(ROOT, f), "rb") as fh:
            h.update(fh.read())
    return dict(code_sha1=h.hexdigest(), ident_version=lofi.IDENT_VERSION,
                fn_steady=list(FN_STEADY), throttles=list(THROTTLES),
                turn_t=dict(TURN_T), l_ref=L_REF)


def _safe(name):
    return "".join(c if c.isalnum() or c in "-_.@" else "_" for c in name)


def _plain(v):
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


class VesselProfile:
    """The profile of one hull (a key such as 'planing_b@as1' or a Hull);
    every part computed once per physics sha1 (module docstring)."""

    def __init__(self, hull, root=None):
        from sim import config, physics
        self.hull = config.hull_of(hull)
        if getattr(self.hull, "kind", None) != "planing":
            raise ValueError("the profile is the planing plant's")
        self.version = physics.version(self.hull)
        self.switches = list(physics.preset(self.version))
        self.sha1 = physics.key_sha1(self.hull)
        self.dir = os.path.join(root or CACHE, _safe(self.hull.name),
                                self.sha1[:16])
        self._plants = None

    # ---------------------------------------------------------- caching
    def meta(self):
        from sim import physics
        return dict(hull=self.hull.name, physics=self.version,
                    switches=self.switches, physics_sha1=self.sha1,
                    code_sha1=physics.code_sha1(self.version),
                    hull_sha1=physics.hull_sha1(self.hull),
                    protocol=protocol())

    def path(self, part):
        return os.path.join(self.dir, f"{part}.json")

    def _cached(self, part, fn, force=False):
        path = self.path(part)
        meta = self.meta()
        if os.path.exists(path) and not force:
            with open(path) as f:
                rec = json.load(f)
            if rec.get("meta") == json.loads(json.dumps(meta)):
                return rec["data"]
        t0 = time.time()
        data = _plain(fn())
        rec = dict(meta=meta, data=data, seconds=time.time() - t0,
                   written=time.strftime("%Y-%m-%d %H:%M:%S"))
        os.makedirs(self.dir, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(rec, f, indent=1)
        os.replace(tmp, path)
        return data

    def plants(self):
        """(the calm plant hull.plant builds, its GZ twin)."""
        if self._plants is None:
            from sim import physics
            p = self.hull.plant(_calm())
            self._plants = (p, physics.gz_twin(p))
        return self._plants

    # ------------------------------------------------------------ parts
    def limits(self, force=False):
        def fn():
            from hydro.hull_geometry import form_of
            p, gz = self.plants()
            sc = self.hull.scales()
            rep = gz.report()
            return dict(L=float(p.L), B=float(p.B), mass=float(p.m),
                        T=float(p.T), GM=float(p.GM),
                        freeboard=float(p.freeboard),
                        draft_bow=float(p.draft_bow),
                        t_max=float(p.prop.t_max), t_min=float(p.prop.t_min),
                        rud_max=float(p.rudder.max),
                        u_design=float(self.hull.u_design),
                        top_speed=float(gz.top_speed()),
                        gz_max=rep["gz_max"],
                        phi_vanish_deg=rep["phi_vanish_deg"],
                        phi_fail_deg=rep["phi_fail_deg"],
                        T_roll=rep["T_roll"], dt=float(sc["dt"]),
                        dt_ctrl=float(sc["dt_ctrl"]),
                        outside_validity=form_of(self.hull).validity())
        return self._cached("limits", fn, force)

    def steady(self, fn_list=FN_STEADY, force=False):
        """Running attitude and calm resistance at length Froude numbers
        fn_list; 'res_w', 'res_q' are the captive equations' residuals at
        the attitude found (w' / g and q' L / g: zero at an equilibrium; a
        captive run that ends on an oscillation says so here). Cached as
        'steady' for FN_STEADY, else under a name carrying the list."""
        fn_list = tuple(float(f) for f in fn_list)
        part = "steady" if fn_list == tuple(FN_STEADY) else (
            "steady_" + hashlib.sha1(json.dumps(list(fn_list)).encode())
            .hexdigest()[:12])

        def fn():
            p, _ = self.plants()
            L = float(p.L)
            rows = []
            for Fn in fn_list:
                u = Fn * math.sqrt(G * L)
                z, th = p.running_attitude(u)
                cr = p.calm_resistance(u)
                ds = self._captive_deriv(p, u, z, th)
                rows.append(dict(Fn=Fn, u=u, z=z, th=th,
                                 trim_deg=cr["trim_deg"], R=cr["R"],
                                 pressure=cr["pressure"],
                                 friction=cr["friction"], air=cr["air"],
                                 lw=cr["wetted_length"],
                                 res_w=float(abs(ds[8]) / G),
                                 res_q=float(abs(ds[10]) * L / G)))
            return rows
        return self._cached(part, fn, force)

    def _captive_deriv(self, p, u, z, th):
        """The captive equations' derivative at (u, z, th): the as plant's
        public captive_deriv; for a v0 plant (which has none) the same calm
        captive copy v0's running_attitude integrates."""
        if hasattr(p, "captive_deriv"):
            return p.captive_deriv(u, z, th)[0]
        from sim.planing_vessel import PlaningVessel
        v = PlaningVessel.__new__(PlaningVessel)
        v.__dict__.update(p.__dict__)
        v.sea = _calm()
        v._wave_arrays()
        v.captive_u = float(u)
        s = np.zeros(14)
        s[6], s[2], s[4] = u, z, th
        return v.deriv(s, 0.0)[0]

    def reduced_model(self, force=False):
        """ReducedModel.identify on config.calm_plant (sim/lofi.template's
        plant) and lofi's fingerprint of it -- never written to
        studies/_cache/lofi_<name>.json."""
        def fn():
            from control.reduced import ReducedModel
            from sim import config, lofi
            plant, _ = config.calm_plant(self.hull, db=None)
            fp = lofi._fingerprint(plant)
            t0 = time.time()
            p = ReducedModel.identify(plant).p
            return dict(p={k: float(v) for k, v in p.items()},
                        fingerprint=fp, ident_version=lofi.IDENT_VERSION,
                        seconds=time.time() - t0)
        return self._cached("reduced_model", fn, force)

    def turn_table(self, force=False):
        """Steady yaw rate with the nozzle hard over, calm water, the GZ
        twin, throttles THROTTLES; the Froude-exact protocol (module
        docstring). 'r_std' is the standard deviation of |r| over the
        averaging window (an oscillating turn shows there)."""
        def fn():
            _, plant = self.plants()
            dt = float(plant.dt)
            ts = math.sqrt(float(plant.L) / L_REF)
            n_turn = int(round(TURN_T["straight"] * ts / dt))
            n_rec = int(round(TURN_T["record"] * ts / dt))
            n_end = int(round(TURN_T["end"] * ts / dt))
            t_max, rud = float(plant.prop.t_max), float(plant.rudder.max)
            U, R, RS, Fs = [], [], [], []
            for f in THROTTLES:
                s = plant.initial_state(0.8 * float(self.hull.u_design))
                us, rs = [], []
                for i in range(n_end):
                    r_cmd = rud if i >= n_turn else 0.0
                    s = plant.step(s, i * dt, f * t_max, r_cmd, dt)
                    if i + 1 >= n_rec:
                        us.append(s[6])
                        rs.append(abs(s[11]))
                if not np.all(np.isfinite(s)):
                    continue
                U.append(float(np.mean(us)))
                R.append(float(np.mean(rs)))
                RS.append(float(np.std(rs)))
                Fs.append(f)
            return dict(u=U, r=R, r_std=RS, throttle=Fs, plant="gz", dt=dt,
                        steps=dict(turn=n_turn, record=n_rec, end=n_end),
                        note="steady full-nozzle turn in calm water, "
                             "Froude-exact protocol (sim/vessel_profile)")
        return self._cached("turn_table", fn, force)

    def build(self, parts=("limits", "steady", "reduced_model",
                           "turn_table"), force=False):
        return {k: getattr(self, k)(force=force) for k in parts}
