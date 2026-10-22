"""
One record per run, and replay.

A Record holds
  header   JSON: scenario (as given) and its sha256, hull key and hull
           sha1, physics stamp (sim/physics.stamp: version, switches, code
           sha1), plant class, policy spec and summary, seeds, the sea's
           components (a, k, th, w, phi: the exact surface), time steps,
           actuator limits, rest attitude, sensors' configuration, end
           reason, git commit, versions, wall times
  s0       the state at reset
  plant    per plant step: t, s (14), a_cg, a_bow (m/s^2), n_sub, the
           applied thrust (N) and nozzle (rad)
  ctrl     per control step: t, the action as given (2 numbers), applied
           thrust, nozzle, heading reference (nan in direct mode), reward,
           cost parts (speed, impact, track), leg, safe-mode flag, wall time
  sensors  per sensor: t_meas, t_avail, valid, values

Saved as one compressed .npz (`header` is a JSON string inside it).
fingerprint() is a sha256 over every array except wall times, in a fixed
order with dtype and shape, AND over the header's identity (IDENTITY_KEYS:
scenario sha256, physics stamp, hull sha1, seeds, the sea's components),
canonical JSON: equal fingerprints = bit-identical runs of the same code on
the same inputs.

replay(record) rebuilds the env from the header's scenario and seed and
feeds the APPLIED commands (after policy, autopilot and safe mode) open
loop: the plant sees exactly the same numbers, so the trajectory and every
sensor sample must come back bit for bit (ACCEPTANCE P4).

FROM ANY ROW (format 2). Every plant row also carries the plant's state
outside s (app/plantstate.py): plant['aux'] (v0's carried actuator state:
delay lines, the jet's prime, the slam detector, capsize; columns in
header['state_cols']) and, when the plant has one, plant['hidden'] (the
physics' hidden state, plant.hidden_state(), e.g. switch wagner_entry's
flow-state field; exact JSON bytes). Every control row carries the env's
controller state (ctrl['ap']: autopilot integrator and reference;
ctrl['safe_state']). replay(record, start=k) resumes the env after control
step k (UsvEnv.resume) and feeds the commands from there; plant_at(record,
j) gives a fresh plant at plant row j (any row, also inside a control
step). A format-1 record replays from the start only.

STALE RECORDS. code_changes(record) compares the record's physics stamp
and hull sha1 with what the current code builds for the same hull key, and
says what changed ("physics code changed since recording (old -> new)");
replay, export and the command line report it before any comparison.
"""
import hashlib
import json
import os

import numpy as np

SKIP_FP = {("ctrl", "wall")}


class RecordMismatch(ValueError):
    """A record cannot be resumed or replayed exactly here: another
    scenario, physics code, test switches, or a format without the rows."""
# the header fields a fingerprint covers (the record's identity)
IDENTITY_KEYS = ("scenario_sha256", "physics", "hull_sha1", "seed",
                 "sea_seed", "sensor_seed")
SEA_KEYS = ("a", "k", "th", "w", "phi")


def identity(header):
    """Canonical JSON of the header's identity fields and sea components."""
    d = {k: header.get(k) for k in IDENTITY_KEYS}
    sea = header.get("sea") or {}
    d["sea"] = {k: sea.get(k) for k in SEA_KEYS}
    return json.dumps(d, sort_keys=True, separators=(",", ":"),
                      default=_json_default)


class Record:
    def __init__(self, header, s0, plant, ctrl, sensors):
        self.header = header
        self.s0 = np.asarray(s0, float)
        self.plant = plant
        self.ctrl = ctrl
        self.sensors = sensors

    # ------------------------------------------------------------ access
    @property
    def t(self):
        return self.plant["t"]

    @property
    def s(self):
        return self.plant["s"]

    def states_with_start(self):
        """(t, s) including the reset state at t = 0."""
        return (np.r_[0.0, self.plant["t"]],
                np.vstack([self.s0[None], self.plant["s"]]))

    def summary(self):
        h, c = self.header, self.ctrl
        s = self.plant["s"]
        fin = np.all(np.isfinite(s), 1) if len(s) else np.zeros(0, bool)
        out = dict(hull=h["hull_key"], physics=h["physics"]["physics"],
                   plant=h["plant_class"],
                   policy=(h.get("policy") or {}).get("spec"),
                   t_end=float(h["t_end"]), steps=int(len(c["t"])),
                   reason=h["reason"],
                   score=float(-np.mean(c["reward"])) if len(c["reward"])
                   else float("nan"),
                   u_mean=float(np.mean(s[fin, 6])) if fin.any()
                   else float("nan"),
                   legs_done=int(c["leg"][-1]) if len(c["leg"]) else 0,
                   a_cg_p99_g=float(np.percentile(np.abs(
                       self.plant["a_cg"][fin]) / 9.81, 99)) if fin.any()
                   else float("nan"),
                   fingerprint=self.fingerprint()[:16])
        return out

    # ------------------------------------------------------------ hashing
    def _arrays(self):
        yield ("s0",), self.s0
        for grp, d in (("plant", self.plant), ("ctrl", self.ctrl)):
            for k in sorted(d):
                if (grp, k) not in SKIP_FP:
                    yield (grp, k), np.asarray(d[k])
        for name in sorted(self.sensors):
            for k in sorted(self.sensors[name]):
                yield ("sensors", name, k), np.asarray(self.sensors[name][k])

    PHYSICAL = ("s0", "plant", "sensors", "ctrl/thrust", "ctrl/nozzle",
                "ctrl/reward", "ctrl/parts", "ctrl/leg", "ctrl/t")

    def fingerprint(self, physical=False):
        """sha256 over the header's identity (scenario, physics stamp, hull,
        seeds, sea) and the arrays; physical=True only over the identity and
        what the plant and the sensors produced and the commands it was
        given (what a command replay must reproduce: the policy's own
        action record and the safe-mode flag are left out)."""
        h = hashlib.sha256()
        h.update(b"identity")
        h.update(identity(self.header).encode())
        for key, a in self._arrays():
            if physical:
                k = "/".join(key)
                if not any(k == p or k.startswith(p + "/")
                           for p in self.PHYSICAL):
                    continue
            a = np.ascontiguousarray(a)
            h.update("/".join(key).encode())
            h.update(str(a.dtype).encode())
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
        return h.hexdigest()

    def _is_physical(self, key):
        k = "/".join(key)
        return any(k == p or k.startswith(p + "/") for p in self.PHYSICAL)

    def diff(self, other, physical=False):
        """[(key, max |difference| or 'shape')] of arrays that are not
        bit-identical (physical=True: only those a command replay must
        reproduce); NaN where one side is NaN."""
        a = dict(self._arrays())
        b = dict(other._arrays())
        out = []
        for k in sorted(set(a) | set(b)):
            if physical and not self._is_physical(k):
                continue
            if k not in a or k not in b:
                out.append(("/".join(k), "missing"))
                continue
            x, y = np.asarray(a[k]), np.asarray(b[k])
            if x.shape != y.shape:
                out.append(("/".join(k), f"shape {x.shape} vs {y.shape}"))
            elif x.tobytes() != y.tobytes():
                with np.errstate(invalid="ignore"):
                    dd = np.abs(x.astype(float) - y.astype(float))
                    d = float(np.max(dd[np.isfinite(dd)])) if np.isfinite(
                        dd).any() else float("nan")
                out.append(("/".join(k), d))
        return out

    # ------------------------------------------------------------ files
    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        arrs = {"s0": self.s0}
        for grp, d in (("plant", self.plant), ("ctrl", self.ctrl)):
            for k, v in d.items():
                arrs[f"{grp}__{k}"] = np.asarray(v)
        for name, d in self.sensors.items():
            for k, v in d.items():
                arrs[f"sensors__{name}__{k}"] = np.asarray(v)
        hdr = dict(self.header)
        hdr["fingerprint"] = self.fingerprint()
        arrs["header"] = np.array(json.dumps(hdr, default=_json_default))
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, **arrs)
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        header = json.loads(str(z["header"]))
        plant, ctrl, sens = {}, {}, {}
        for k in z.files:
            if k in ("header", "s0"):
                continue
            parts = k.split("__")
            if parts[0] == "plant":
                plant[parts[1]] = z[k]
            elif parts[0] == "ctrl":
                ctrl[parts[1]] = z[k]
            elif parts[0] == "sensors":
                sens.setdefault(parts[1], {})[parts[2]] = z[k]
        return cls(header, z["s0"], plant, ctrl, sens)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    return repr(o)


def code_changes(rec):
    """[str] of what differs between the code that made `rec` and the
    current code for the same hull key (empty: nothing)."""
    from sim import physics
    from . import hulls
    from .scenario import Scenario
    h = rec.header
    out = []
    try:
        hull = hulls.resolve(Scenario(h["scenario"]).hull_spec)
    except Exception as ex:                          # noqa: BLE001
        return [f"the hull of the record no longer builds: "
                f"{type(ex).__name__}: {ex}"]
    sha_now = physics.hull_sha1(hull)
    if h.get("hull_sha1") and h["hull_sha1"] != sha_now:
        out.append(f"hull inputs changed since recording ({h['hull_sha1'][:12]}"
                   f" -> {sha_now[:12]})")
    # a randomised episode's plant ran a modified hull (another manifest,
    # so another stamp): its own stamp is header['physics'] (what ran), and
    # the comparison with the current code is made on the nominal key's
    # stamp beside it; an episode on the nominal hull compares its own
    st = h.get("physics") or {}
    if h.get("physics_nominal") and h.get("episode_hull_sha1") and \
            h["episode_hull_sha1"] != h.get("hull_sha1"):
        st = h["physics_nominal"]
    ver = st.get("physics", "v0")
    if st.get("physics_sha1"):
        try:
            now = physics.stamp_for(hull, ver, st.get("switches", ()))
            if now["physics_sha1"] != st["physics_sha1"]:
                out.append(f"physics code changed since recording "
                           f"({ver}: {st['physics_sha1'][:12]} -> "
                           f"{now['physics_sha1'][:12]})")
        except Exception as ex:                      # noqa: BLE001
            out.append(f"physics version {ver!r} no longer stamps: "
                       f"{type(ex).__name__}: {ex}")
        if h.get("test_switches") is None:
            # a test-switch record runs its own switches by design
            try:
                pre = list(physics.preset(ver))
                if pre != list(st.get("switches", ())):
                    out.append(f"{ver}'s preset switches changed since "
                               f"recording ({list(st.get('switches', ()))} "
                               f"-> {pre})")
            except Exception:                        # noqa: BLE001
                pass
    tag = h.get("cache_tag", "unrecorded")
    if tag != "unrecorded":
        now = physics.cache_tag(hull)
        if now != tag:
            out.append(f"the hull's cache tag changed since recording "
                       f"({tag} -> {now})")
    return out


def env_of(rec, record=True):
    """A fresh env for `rec`: its scenario as recorded (its sha256 is part
    of the identity), direct actions, safe mode off (the applied commands
    already include it and the autopilot), its test switches."""
    from .env import UsvEnv
    h = rec.header
    # the record's own steps (a record made before the frozen steps became
    # the as versions' default keeps replaying at its 0.02 s / 12)
    env = UsvEnv(h["scenario"], action_mode="direct", record=record,
                 test_switches=h.get("test_switches"), dt=h.get("dt"),
                 dt_ctrl=h.get("dt_ctrl"))
    env.scenario.usv["safe_mode"] = False
    return env


def replay(rec, record=True, start=0):
    """Feed rec's applied commands open loop into a fresh env (same
    scenario, seed, hull key), from control step `start` on (the env
    resumed there: UsvEnv.resume); returns the new Record of the whole
    episode."""
    env = env_of(rec, record)
    obs, info = env.resume(rec, start)
    thr, noz = rec.ctrl["thrust"], rec.ctrl["nozzle"]
    for k in range(start, len(thr)):
        obs, r, term, trunc, info = env.step(
            {"thrust": float(thr[k]), "nozzle_rad": float(noz[k])})
        if term or trunc:
            break
    out = env.record(policy=dict(spec={"type": "replay"},
                                 source_fingerprint=rec.fingerprint(),
                                 start=int(start)))
    return out


def plant_at(rec, row, env=None):
    """(plant, s, t): a fresh plant of `rec`'s episode at plant row `row`
    (-1: the reset state), any row, inside a control step too. Stepping it
    with rec.plant['thrust'][j], ['nozzle'][j] at t, t + dt, ... for
    j = row + 1, ... gives rec.plant['s'][j] bit for bit."""
    from . import plantstate
    h, P = rec.header, rec.plant
    if env is None:
        env = env_of(rec, record=False)
    env.resume(rec, 0, strict=True)       # identity checks, the reset plant
    if row == -1:
        return env.plant, np.array(rec.s0, float), 0.0
    if not 0 <= row < len(P["t"]):
        raise ValueError(f"row {row}: the record has plant rows "
                         f"-1..{len(P['t']) - 1}")
    if "aux" not in P:
        raise RecordMismatch(f"the record ({h.get('format')}) carries no "
                             f"plant state rows")
    hid = plantstate.loads_rows(P["hidden"]) if "hidden" in P else None
    s = np.array(P["s"][row], float)
    plantstate.restore(env.plant, s, None if hid is None else hid[row],
                       P["aux"][row])
    return env.plant, s, float(P["t"][row])
