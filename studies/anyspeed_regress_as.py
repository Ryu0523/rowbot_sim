#!/usr/bin/env python3
"""
Versioning regression of the any-speed plant (design step S0,
learn/meta/anyspeed_design_en.md section 4 and the validation review's
versioning fixes): v0 is frozen and stays the default, the new physics is
chosen by the hull key and nothing else, and the any-speed plants with every
switch off ARE v0 -- equal, not close: PlaningVesselASLin == PlaningVessel
(what hull.plant builds) and PlaningVesselAS == PlaningVesselGZ (the GZ
twin). From step S1 on, the as1 preset has switches on (S1: dDt_full), so it
differs from v0 by design and is guarded by its own 'on' golden instead.

Tiers (each optional; read-only but for the report in
studies/_cache/anyspeed/):
  freeze    sha1 of the four v0 physics files (raw bytes as checked out --
            CRLF here, as the meta code hashes read them -- the design's
            08ac5e0c / 1597caef / 9b2d9bfd / d5668c59), of hydro/planing.py
            (the hull-keyed dispatch, physics.build), of the reference data
            (lofi_scarab195.json, golden_v0.npz, turn_gz.json, meta7's
            results.pkl, the onestep streams) and of the as1 'on' golden,
            and of the scarab195 hull manifest (sim/physics.hull_fields; S0
            added no instance field, so it is the pre-S0 value); the as1
            hull equals it but for name and physics. COPY: the methods
            sim/planing_anyspeed.py copies from v0 (_StripAS.deriv,
            running_attitude, calm_resistance, top_speed, with_sea), each
            projected on its off branches (every 'if self.sw.X: ... else:
            <v0>' replaced by its else) and with an allow-list of known
            lines (the copy's class, the switch-keyed caches, the guard of
            v0's module globals, the S1 flux alias), must equal v0's source
            AST exactly, and the copy may read no v0 module global that
            check_v0_globals does not guard -- so a slip in an off branch
            that no golden run executes still fails. Since P0 the
            constructor is copied too (_StripAS.__init__: the geometry from
            hydro/hull_geometry, a BLOCK standing for v0's geometry lines)
            and the plant's absolute constants are parameters (self.num.*,
            the jet's time constants): an EXPRESSION map sends each back to
            v0's literal, each used the expected number of times, and the
            parameters' values for scarab195 must be those literals
  select    the single selector. v0 key -> PlaningVessel (hull.plant,
            config.plant_for) and PlaningVesselGZ (physics.gz_twin,
            gz_plant_for); '@as1' key -> PlaningVesselASLin (hull.plant,
            config.plant_for, sim/env.Episode, sim/rl_env) and
            PlaningVesselAS (gz_twin, gz_plant_for) on its preset, test
            switches carried by gz_twin and with_sea. Refusals: a v0 hull
            never yields an as plant; a name and a physics attribute that
            disagree; '@v0', unknown versions and vessels, non-planing
            hulls, unimplemented and unknown switches, every 'needs' of a switch
            (each switch without each of its needs), the transom closure's
            alternative modes without the sensitivity flag, the old switches=
            keyword, test_switches= for a v0 hull, the Garme transom
            correction with dDt_full; physics.verify refuses test switches
            unless allowed; a patched v0 module global (PV.NU) stops the as
            plants; variant() refuses a modified hull under a registered
            name. USV_HULL disagreeing with the hull in hand raises. Spawned
            workers get the key through USV_HULL and build the same plant as
            the parent (the as1 preset), to the last bit; the parent's
            all-off plants equal v0 exactly, and the as1 preset differs from
            v0 once a switch is on. The frozen from_plant applied to an as1
            plant builds v0 (it cannot see the hull) and physics.verify
            catches it. An as1 hull's reduced-model cache is its own file;
            the stamp is taken at build with the code hash of the import.
            PENDING (listed, not counted): the frozen helpers' call sites
            in the code reachable from studies/final_eval.py and
            studies/safety_events.py, and learn/repro/task.py's hard-coded
            HULL (Mission cannot select as1) -- to switch before S9
  golden    studies/_cache/anyspeed/golden_v0.npz (anyspeed_regress.
            golden_runs) reproduced exactly by: v0; PlaningVesselAS all off
            (preset 'v0'); v0 again, after the as plants (no cache leaks
            between the classes). The as1 preset through the selector:
            identical to golden_v0 while the preset is empty (S0), else
            different from it and equal, exactly, to its own 'on' golden
            golden_as_<switches>.npz (the same runs plus
            last_slam_force_full), written once with --golden-on write;
            the same switches given explicitly == the selector's run;
            on the task-sea and calm runs, re-evaluated state by state: the
            new flux equals v0's at every station whose taper strip is not
            chine-wetted, is never negative (entry only), and the recorded
            last_slam_force_full is the flux at the start of each step.
            Since P1a: the preset's runs equal the P1a candidate reference
            the review checked (P1A_REF), array for array; and the switch
            sets published before (GOLDEN_SETS: the preset before P1a, and
            it plus verlet) re-run with test_switches= against their own
            goldens, so every switch combination released keeps an 'on'
            test (N2) after the preset moves on
  lofi      sim/lofi.py's fingerprint of the calm plant: v0's PlaningVessel
            == the stored 36.50273627118908, and so does PlaningVesselASLin
            all off, exactly (the class an as1 hull's reduced model is
            identified on); PlaningVesselAS all off == PlaningVesselGZ v0
            (the GZ value differs from the stored one: the jet's side force
            heels the boat, and GZ and the linear roll part there)
  missions  final_eval rows as meta7 ran them (anyspeed_regress.
            check_missions), the GZ twin replaced by PlaningVesselAS all off
            for an '@as1' hull; every field equal to meta7's results.pkl
Always: studies/_cache/lofi_scarab195.json unchanged (sha1 before, after,
and the frozen value).

    .venv/Scripts/python.exe -m studies.anyspeed_regress_as --all
    .venv/Scripts/python.exe -m studies.anyspeed_regress_as --select \\
        --missions hand:s0_l0,m0:s0_l1
"""
import argparse
import hashlib
import importlib
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from studies import anyspeed_regress as AR   # noqa: E402  (threads, path)
import numpy as np                           # noqa: E402

AS_KEY = "scarab195@as1"
FROZEN = {
    "sim/planing_vessel.py": "08ac5e0ce4207f69d9c13fb30e30fbbd8faefe52",
    "sim/planing_vessel_gz.py": "1597caeffa68f36203aabbc25d4b4bc2a7bc4025",
    "sim/waterjet.py": "9b2d9bfd489c1b6722d9dcf3c748a059faaf7c4c",
    "sim/actuators.py": "d5668c599cf07461d4d77e2af6f329b41b8551ff",
    # the hull-keyed dispatch in PlaningHull.plant() (physics.build, which
    # refuses test_switches= for a v0 hull)
    "hydro/planing.py": "dcc33d429c7b2e79c3a1a51c2decc5e0cfb0d0b7",
    # the reference data every v0 check reads
    "studies/_cache/lofi_scarab195.json":
        "2445f83d36d54aa730e3df1ead3336ae6a27cbad",
    "studies/_cache/anyspeed/golden_v0.npz":
        "e929ca43a3176c42b6c3ef404b89a9323e82eca0",
    "studies/_cache/final_eval/turn_gz.json":
        "44b5bb9eebaed2318623b3eec2fd10f5ad82e5dc",
    "studies/_cache/final_eval/meta7/results.pkl":
        "1a092e523d3295e6387037f77569b373ae105aa4",
    "studies/_cache/onestep_gz/streams_T120.pkl":
        "8458a5e40e6b6f9c19b23f694bbfe3049c3dfa41",
}
# the as1 preset's own golden (rewritten when the preset's physics changes;
# checked by the freeze tier unless --golden-on write)
# (S1: golden_as_dDt_full.npz, c44d9d31...; P0 added chine_exact_all to the
# preset, after S1's standing checks passed on it: studies/_cache/anyspeed/
# p0/s1_rerun/, golden_as_dDt_full+chine_exact_all.npz 0b1a30f2...; the P0
# review fixes added kpp_strip, which is exactly zero on scarab195 (no keel
# rise): the file for the new preset name was written and is byte-identical
# to the previous one, the same sha1 -- the review's other fixes (exact-speed
# steady cache, the complete jet dict, _substrip's shortcut, the scaled GZ
# bracket) changed no bit of scarab195@as1 either; P1a added verlet, the
# integrator, after its registered checks passed (studies/
# anyspeed_p1a_check.py, studies/_cache/anyspeed/p1a/):
# golden_as_dDt_full+chine_exact_all+kpp_strip+verlet.npz 421240cd...; after
# the P1a review the orchestrator adopted transom and normal_rate as a
# recorded exception (sim/planing_anyspeed.PRESETS): the preset's golden is
# the candidate's, and it must reproduce, array for array, the candidate
# reference the review checked (P1A_REF); P1c adopted fk_exact (E1's
# revision) with the implementation switch fast_path: the preset's golden
# is written by studies/anyspeed_p1c_check.py part goldens (two processes
# identical) and the P1a preset's moves to GOLDEN_SETS)
GOLDEN_ON = {
    "studies/_cache/anyspeed/golden_as_dDt_full+chine_exact_all+kpp_strip"
    "+transom+normal_rate+verlet+fk_exact+fast_path.npz":
        "34b658f90ec47b9f3df5dd0ae799984486ca5208"}
# goldens of switch sets that are not (or no longer) the preset, each
# re-run by PlaningVesselAS with test_switches= that set against its file:
# the 'on' tests (N2) of every switch combination published so far stay
# after the preset moves on -- verlet without transom (the preset between
# verlet's and transom's adoption) and the preset before P1a
# the P1a preset (the base of the P1b per-switch goldens)
P1A_SET = ("dDt_full", "chine_exact_all", "kpp_strip", "transom",
           "normal_rate", "verlet")
# the P1c preset (fk_exact and fast_path on P1a's), the base of S5a's sets
AS1_P1C_SET = P1A_SET + ("fk_exact", "fast_path")
GOLDEN_SETS = {
    # P1b, after its review: one golden per switch on the P1a preset (the
    # 'on' tests, N2), written by studies/anyspeed_p1b_fix.py part goldens
    # (two processes identical, the calm runs' largest heel <= 15 deg).
    # The four-switch candidate's golden (golden_as_...+lateral_phi0.npz,
    # 93192dee...) is no longer checked: lateral_phi0 capsized four of its
    # ten runs (R4), a chaotic trajectory that pinned nothing per switch;
    # the file stays on disk. lateral_phi0 is pinned with the nozzle held
    # at zero (GOLDEN_OPTS) until T8 / S5.
    # P1c (track P): the P1a preset, no longer the preset (its file is the
    # former GOLDEN_ON, = P1A_REF); fk_exact's integrals moved to the
    # row-stable kernel sim/as_kernels.fk_rows (round-off: within 2e-12 of
    # P1b's files, studies/_cache/anyspeed/p1c/repin.txt), so the sets with
    # fk_exact are pinned by new files (P1b's, 305eb430... and 6eaed231...,
    # stay on disk, unchecked)
    P1A_SET: (
        "studies/_cache/anyspeed/golden_as_" + "+".join(P1A_SET) + ".npz",
        "6bd250f4d3091ee7de28b116d3dec79776c6424a"),
    P1A_SET + ("fk_exact",): (
        "studies/_cache/anyspeed/golden_as_" + "+".join(P1A_SET)
        + "+fk_exact_p1c.npz",
        "34b658f90ec47b9f3df5dd0ae799984486ca5208"),
    # S5a review (track S): orbital's lateral diffraction takes the orbital
    # velocity's rate at fixed body x (the encounter frequency, plan 'S5a：
    # 复查后的修正' issue C), so its task-sea runs moved: re-pinned by
    # studies/anyspeed_s5a_check.py part goldens (two processes identical,
    # stamped record s5a/r2/goldens_s5a.txt); the P1c file (_p1c.npz,
    # 172dfc48...) stays on disk, unchecked
    P1A_SET + ("fk_exact", "orbital"): (
        "studies/_cache/anyspeed/golden_as_" + "+".join(P1A_SET)
        + "+fk_exact+orbital_s5a2.npz",
        "b2fd0b4b86323cd45bd88ffa5718472c6f9c74f0"),
    # P1c (track W): the field's transport moved to Heun's rule (second
    # order), then (round 2 of the P1c review) to ES_NSUB sub-steps on
    # interpolated kinematics, so wagner_entry's runs changed; its golden is
    # written by studies/anyspeed_p1c_w.py part goldenw (two processes
    # identical). The P1b one (+wagner_entry.npz, ccf3069e...) and round
    # 1's (+wagner_entry_p1c.npz, 3b9f0408...) stay on disk, unchecked.
    # P1c round 3 (track W, plan '### wagner 低速'): an X plane's added
    # mass no longer grows along its path; the golden is written by
    # studies/anyspeed_p1c_w3.py part goldenw3; round 2's (_p1c2.npz,
    # 743b7416...) stays on disk, unchecked.
    P1A_SET + ("wagner_entry",): (
        "studies/_cache/anyspeed/golden_as_" + "+".join(P1A_SET)
        + "+wagner_entry_p1c3.npz",
        "fd9a564ce9a9540b24b4243d19eac8ae4a899b94"),
    # S5a (track S): the 2D table's wedge branch made continuous at the
    # low chine (hydro/section2d.Table._shallow), so lateral_phi0's heeled
    # (task-sea) runs moved: re-pinned by studies/anyspeed_s5a_goldens.py
    # (two processes identical); the P1b file (+lateral_phi0_nozzle0.npz,
    # 4ffbf86c...) stays on disk, unchecked. S5a review: the table's even
    # entries go in phi^2 inside the first heel cell and the orbital rate
    # is body-fixed, so it moved again: re-pinned (_n0_s5a2.npz; with the
    # calm nozzle at 5 deg its calm runs heeled past 15 deg, so the nozzle
    # stays at zero there: its calm runs pin nothing lateral_phi0 changes,
    # recorded); the first round's _nozzle0_s5a.npz (0ba2d5f7...) stays on
    # disk, unchecked
    P1A_SET + ("lateral_phi0",): (
        "studies/_cache/anyspeed/golden_as_" + "+".join(P1A_SET)
        + "+lateral_phi0_n0_s5a2.npz",
        "22352995848fc7e3b30ffba9c554cf863576033f"),
    # S5a (track S): heel_hydro, and the cross plane, on the P1c preset.
    # S5a review: the physics changed (heel_hydro: P1c's a_b convention,
    # the roll axis, the transom's yaw; lateral_cross: one kinetic energy,
    # body-fixed wave rates, the P_x pair; the table at zero heel), and the
    # calm runs now hold the nozzle at 5 deg (the first round's nozzle-at-
    # zero calm runs stayed upright and pinned nothing these switches
    # change; the task runs keep it at zero): re-pinned by
    # studies/anyspeed_s5a_check.py part goldens (two processes identical,
    # every set differing from P1c on calm and task runs; stamped record
    # s5a/r2/goldens_s5a.txt). The first round's _nozzle0.npz files
    # (8420bf71..., 0693c9a1...) stay on disk, unchecked; its two failed
    # moving-nozzle files moved to s5a/h12_fail_hh.npz, h12_fail_hx.npz
    AS1_P1C_SET + ("heel_hydro",): (
        "studies/_cache/anyspeed/golden_as_" + "+".join(AS1_P1C_SET)
        + "+heel_hydro_n0c5_s5a2.npz",
        "cc8a6c97cbca4d63c7d606d7ccc1d389be01e242"),
    AS1_P1C_SET + ("heel_hydro", "lateral_phi0", "lateral_cross"): (
        "studies/_cache/anyspeed/golden_as_" + "+".join(AS1_P1C_SET)
        + "+heel_hydro+lateral_phi0+lateral_cross_n0c5_s5a2.npz",
        "96c06350e864e440c7ef81407410444a75da6a78"),
    ("dDt_full", "chine_exact_all", "kpp_strip", "verlet"): (
        "studies/_cache/anyspeed/"
        "golden_as_dDt_full+chine_exact_all+kpp_strip+verlet.npz",
        "421240cd0c5d9af9985897b25fad9a05a7ea5f03"),
    ("dDt_full", "chine_exact_all", "kpp_strip"): (
        "studies/_cache/anyspeed/"
        "golden_as_dDt_full+chine_exact_all+kpp_strip.npz",
        "0b1a30f2b273796b4170db26110be2b74a146f78"),
}
# options of a golden set's runs (anyspeed_regress.golden_runs keywords)
GOLDEN_OPTS = {P1A_SET + ("lateral_phi0",): dict(nozzle=False),
               AS1_P1C_SET + ("heel_hydro",): dict(nozzle=False,
                                                  calm_nozzle_deg=5.0),
               AS1_P1C_SET + ("heel_hydro", "lateral_phi0",
                              "lateral_cross"): dict(nozzle=False,
                                                     calm_nozzle_deg=5.0)}
# the largest heel of a golden set's calm runs (P1b review: a golden must
# pin a physical trajectory; R4)
GOLDEN_HEEL_MAX_DEG = 15.0
# the P1a candidate's reference (studies/anyspeed_p1a_check.py part golden;
# what the P1a review reproduced bit for bit): the preset's golden must
# hold the same arrays
P1A_REF = ("studies/_cache/anyspeed/p1a/golden_as_dDt_full+chine_exact_all"
           "+kpp_strip+transom+normal_rate+verlet.npz",
           "6bd250f4d3091ee7de28b116d3dec79776c6424a")
# the legacy hull manifests, pinned in one place (sim/physics.LEGACY: the
# hulls whose caches keep their legacy names); the freeze tier checks the
# registered hull against them
from sim.physics import LEGACY as HULLS      # noqa: E402
ON_EXTRA = ("last_slam_force_full",)
MISSIONS = "hand:s0_l0,m0:s0_l1"
N_PROBE = 100


def sha1_file(path):
    with open(path if os.path.isabs(path) else os.path.join(ROOT, path),
              "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


class Tally:
    """Named pass / fail checks, printed as they come."""

    def __init__(self):
        self.rows = []

    def __call__(self, tier, name, ok, detail=""):
        ok = bool(ok)
        self.rows.append(dict(tier=tier, name=name, ok=ok,
                              detail=str(detail)))
        print(f"  [{'ok' if ok else 'FAIL'}] {tier}: {name}"
              + (f"  -- {detail}" if detail != "" else ""), flush=True)
        return ok

    def raises(self, tier, name, exc, fn):
        try:
            fn()
        except exc as e:
            return self(tier, name, True,
                        f"{type(e).__name__}: {str(e)[:110]}")
        except Exception as e:      # noqa: BLE001  (the wrong refusal)
            return self(tier, name, False,
                        f"raised {type(e).__name__} instead: {e}")
        return self(tier, name, False, "did not raise")


def _calm():
    from sim.test_vessel import Monochromatic
    return Monochromatic(1.0, 0.0)


# ---------------------------------------------------------------- freeze
def tier_freeze(T, report, on_mode="check"):
    from hydro import hulls
    from sim import physics
    for f, want in FROZEN.items():
        got = sha1_file(f)
        T("freeze", f"sha1 {f}", got == want, got)
    if on_mode != "write":
        for f, want in GOLDEN_ON.items():
            got = sha1_file(f) if os.path.exists(os.path.join(ROOT, f)) \
                else None
            T("freeze", f"sha1 {f} (the as1 'on' golden)", got == want, got)
    for sw, (f, want) in list(GOLDEN_SETS.items()) + [("P1a ref", P1A_REF)]:
        got = sha1_file(f) if os.path.exists(os.path.join(ROOT, f)) else None
        T("freeze", f"sha1 {f} (golden of {sw})", got == want, got)
    for name, want in HULLS.items():
        got = physics.hull_sha1(hulls.get(name))
        T("freeze", f"hull manifest {name}", got == want, got)
    f0 = physics.hull_fields(hulls.get("scarab195"))
    f1 = physics.hull_fields(hulls.get(AS_KEY))
    keys = set(f0) | set(f1)
    differ = sorted(k for k in keys if k not in f0 or k not in f1
                    or f0[k] != f1[k])
    T("freeze", f"{AS_KEY} = scarab195 but for name and physics",
      differ == ["name", "physics"] and f1["name"] == AS_KEY
      and f1["physics"] == "as1", differ)
    report["code_sha1"] = {v: physics.code_sha1(v) for v in physics.VERSIONS}
    copy_check(T, report)


# ------------------------------------------------------------ copy check
# allowed differences of the copies from v0, per method: statements dropped
# from the copy, statements of the copy and the v0 statement each stands
# for, and variable aliases (the copy's name -> v0's) whose plain alias
# assignment is dropped and whose uses are renamed
_GUARD = "check_v0_globals()"
_KEY_V0 = ("key = (self.hull.name, id(self.hull), round(float(u), 3), "
           "getattr(self.hull, 'transom_correction', True), self.k_a, "
           "self.hull.a_b, self.c_dc)")
_KEY_AS = ("key = (self._stamp['physics_sha1'], float(u), "
           "getattr(self.hull, 'transom_correction', True), self.k_a, "
           "self.hull.a_b, self.c_dc)")
# the plant constants (hydro/vessel_spec.PlantNumerics) and the literal of
# v0 each stands for, with the number of uses expected in the copy
_NUM_DERIV = {"self.num.u_floor": ("0.05", 1), "self.num.l_floor": ("0.05", 1),
              "self.num.nu": ("NU", 1), "self.num.rho_air": ("RHO_AIR", 1),
              "self.num.z_air": ("0.5", 1)}
_NUM_CAPTIVE = {"self.num.dt_captive": ("0.005", 1),
                "self.num.n_captive": ("4000", 1),
                "self.num.n_check - 1": ("199", 1),
                "self.num.n_check": ("200", 1),
                "self.num.tol_z": ("1e-05", 1),
                "self.num.tol_th": ("1e-06", 1)}
# v0's geometry lines (sim/planing_vessel.py:91-102) and the copy's lines
# from the geometry module that stand for them (array-equal: the 'geom'
# check of tier select and studies/anyspeed_p0_check.py part geom)
_GEOM_V0 = (
    "n = h.n_stations",
    "edges = np.linspace(0.0, h.L, n + 1)",
    "xi = 0.5 * (edges[:-1] + edges[1:])",
    "self.dx = h.L / n",
    "self.x = xi - h.lcg",
    "x0 = (1.0 - h.taper) * h.L",
    "s = np.clip((xi - x0) / max(h.L - x0, 1e-09), 0.0, 1.0)",
    "self.yc = 0.5 * h.b * np.sqrt(np.maximum(1.0 - s ** 2, 0.0)) + 0.0001",
    "self.zk = -h.vcg + h.stem_rise * s ** 2",
    "self.kp = 2.0 * h.stem_rise * s / max(h.L - x0, 1e-09)",
    "self.kpp = np.where(s > 0, 2.0 * h.stem_rise / max(h.L - x0, 1e-09) "
    "** 2, 0.0)")
_GEOM_AS = (
    "g = self._geom = _hg.stations_of(h)",
    "n = g.n",
    "self.dx = g.dx",
    "self.x = g.xi - h.lcg",
    "self.yc = g.yc",
    "self.zk = -h.vcg + g.rise",
    "self.kp = g.kp",
    "self.kpp = g.kpp")
_PUMP_V0 = ("pump = JetPump(np.pi * j.get('d_nozzle', 0.085) ** 2 / 4, "
            "j.get('p_max', 224000.0), eta_pump=j.get('eta_pump', 0.775), "
            "rho=self.rho)")
_PUMP_AS = ("pump = JetPump(np.pi * j.get('d_nozzle', 0.085) ** 2 / 4, "
            "j.get('p_max', 224000.0), eta_pump=j.get('eta_pump', 0.775), "
            "zeta_n=j.get('zeta_n', 0.02), zeta_i=j.get('zeta_i', 0.25), "
            "wake=j.get('wake', 0.05), rho=self.rho)")
_JET_EXPRS = {"j.get('tau_prime_loss', 0.05)": ("0.05", 1),
              "j.get('t_rate', 0.3)": ("0.3", 1),
              "j.get('t_min_div', 3.0)": ("3", 1),
              "j.get('dead_band_ref', 50.0)": ("50", 1),
              "j.get('dead_band_ref_thrust', 12000.0)": ("12000", 1)}
COPIES = (
    # (copy, v0 original, drop, substitute, aliases, switch blocks,
    #  expressions {copy: (v0, uses)}, blocks ((copy lines, v0 lines),))
    # (deriv: 7 switch blocks through P1a, 26 with P1b's fk_exact, orbital,
    # wagner_entry and lateral_phi0, 27 after the P1b review: wagner_entry's
    # hx in aux; 28 with P1c's fast_path (its block exports A and wbar to
    # aux); 41 with S5a's heel_hydro (8 blocks: the heel state, the keel
    # depth, the transom section, the vertical force, the longitudinal
    # force, the lateral force, the roll moment, aux) and lateral_cross (5:
    # its lateral flow, the 5x5 solve, its surge tilt, the lateral solve
    # skip, aux); blocks 'if self.sw.X and <cond>:' count too)
    ("sim.planing_anyspeed:_StripAS.deriv",
     "sim.planing_vessel:PlaningVessel.deriv",
     ("aux['mom_full'] = flux",), {}, {"flux": "mom"}, 41, _NUM_DERIV, ()),
    ("sim.planing_anyspeed:_StripAS.__init__",
     "sim.planing_vessel:PlaningVessel.__init__",
     ("self.num = _vs.numerics_of(h)",),
     {"j = _vs.jet_inputs(h)": "j = h.jet", _PUMP_AS: _PUMP_V0}, {}, 1,
     _JET_EXPRS, ((_GEOM_AS, _GEOM_V0),)),
    ("sim.planing_anyspeed:_ASMixin.running_attitude",
     "sim.planing_vessel:PlaningVessel.running_attitude",
     (_GUARD,), {"v = _StripAS.__new__(_StripAS)":
                 "v = PlaningVessel.__new__(PlaningVessel)",
                 _KEY_AS: _KEY_V0}, {}, 0, _NUM_CAPTIVE, ()),
    ("sim.planing_anyspeed:_ASMixin.calm_resistance",
     "sim.planing_vessel:PlaningVessel.calm_resistance",
     (_GUARD,), {"v = _StripAS.__new__(_StripAS)":
                 "v = PlaningVessel.__new__(PlaningVessel)"}, {}, 0, {}, ()),
    ("sim.planing_anyspeed:_ASMixin.top_speed",
     "sim.planing_vessel_gz:PlaningVesselGZ.top_speed",
     (), {"key = (self._stamp['physics_sha1'], float(self.prop.t_max))":
          "key = (self.hull.name, id(self.hull), float(self.prop.t_max))"},
     {}, 0, {"self.num.u_tol": ("0.01", 1)}, ()),
    ("sim.planing_anyspeed:_ASMixin.with_sea",
     "sim.planing_vessel_gz:PlaningVesselGZ.with_sea",
     ("kw = self._rebuild_kw(**kw)",), {}, {}, 0, {}, ()),
)
# every parameter of the expression maps, its value for scarab195 (v0's)
_PARAM_LITERALS = {
    **{k: v for k, (v, _) in _NUM_DERIV.items()},
    **{k: v for k, (v, _) in _NUM_CAPTIVE.items()},
    "self.num.u_tol": "0.01"}


def _fn_source(ref):
    import inspect
    import textwrap
    mod, qual = ref.split(":")
    obj = importlib.import_module(mod)
    for part in qual.split("."):
        obj = getattr(obj, part)
    return textwrap.dedent(inspect.getsource(obj))


def _fn_node(ref, src=None):
    """The FunctionDef of 'module:Class.method' (or of the source text
    src), docstring removed."""
    import ast
    node = ast.parse(_fn_source(ref) if src is None else src).body[0]
    b = node.body
    if b and isinstance(b[0], ast.Expr) and isinstance(
            getattr(b[0], "value", None), ast.Constant) and isinstance(
            b[0].value.value, str):
        node.body = b[1:]
    node.decorator_list = []
    return node


def _project(node, drop, subst, alias, exprs=None, blocks=()):
    """The copy's FunctionDef on its off branches, the allowed blocks and
    lines dropped / substituted, the parameter expressions mapped to v0's
    literals and the aliases inlined: (node, number of switch blocks,
    statements, blocks and expressions dropped or substituted)."""
    import ast
    import copy as _copy
    norm = lambda src: ast.unparse(ast.parse(src))     # noqa: E731
    norm_e = lambda src: ast.unparse(                    # noqa: E731
        ast.parse(src, mode="eval").body)
    drop = {norm(x) for x in drop}
    subst = {norm(k): ast.parse(v).body[0] for k, v in subst.items()}
    exprs = {norm_e(k): ast.parse(v, mode="eval").body
             for k, (v, _) in (exprs or {}).items()}
    blocks = [([norm(x) for x in cb], [ast.parse(x).body[0] for x in vb])
              for cb, vb in blocks]
    used = []

    class Off(ast.NodeTransformer):
        n = 0

        @staticmethod
        def is_sw(t):
            return (isinstance(t, ast.Attribute)
                    and isinstance(t.value, ast.Attribute)
                    and isinstance(t.value.value, ast.Name)
                    and t.value.value.id == "self" and t.value.attr == "sw")

        def visit_If(self, nd):
            self.generic_visit(nd)
            t = nd.test
            # 'if self.sw.X:' and, since the P1b review, 'if self.sw.X and
            # <cond>:' (the block needs switch X; P1b's wave-dependent
            # blocks): with X off the condition is false, the else runs
            if self.is_sw(t) or (isinstance(t, ast.BoolOp)
                                 and isinstance(t.op, ast.And)
                                 and self.is_sw(t.values[0])):
                Off.n += 1
                return nd.orelse
            return nd

    class Lines(ast.NodeTransformer):
        def generic_visit(self, nd):
            for field, val in ast.iter_fields(nd):
                if isinstance(val, list) and val and isinstance(val[0],
                                                                ast.stmt):
                    new = []
                    for st in val:
                        src = ast.unparse(st)
                        if src in drop:
                            used.append(("drop", src))
                            continue
                        if src in subst:
                            used.append(("subst", src))
                            st = subst[src]
                        elif (isinstance(st, ast.Assign)
                              and len(st.targets) == 1
                              and isinstance(st.targets[0], ast.Name)
                              and st.targets[0].id in alias
                              and isinstance(st.value, ast.Name)
                              and st.value.id == alias[st.targets[0].id]):
                            used.append(("alias", src))
                            continue
                        new.append(self.visit(st))
                    setattr(nd, field, new)
                elif isinstance(val, ast.AST):
                    self.visit(val)
                elif isinstance(val, list):
                    for v in val:
                        if isinstance(v, ast.AST):
                            self.visit(v)
            return nd

    class Rename(ast.NodeTransformer):
        def visit_Name(self, nd):
            if nd.id in alias:
                nd.id = alias[nd.id]
            return nd

    class Blocks(ast.NodeTransformer):
        def generic_visit(self, nd):
            for field, val in ast.iter_fields(nd):
                if isinstance(val, list) and val and isinstance(val[0],
                                                                ast.stmt):
                    srcs = [ast.unparse(st) for st in val]
                    for cb, vb in blocks:
                        k = len(cb)
                        for i in range(len(srcs) - k + 1):
                            if srcs[i:i + k] == cb:
                                used.append(("block", cb[0]))
                                val = val[:i] + [_copy.deepcopy(s)
                                                 for s in vb] + val[i + k:]
                                srcs = [ast.unparse(st) for st in val]
                                break
                    setattr(nd, field, [self.visit(st) for st in val])
                elif isinstance(val, ast.AST):
                    self.visit(val)
                elif isinstance(val, list):
                    for v in val:
                        if isinstance(v, ast.AST):
                            self.visit(v)
            return nd

    class Exprs(ast.NodeTransformer):
        def visit(self, nd):
            if isinstance(nd, ast.expr):
                src = ast.unparse(nd)
                if src in exprs:
                    used.append(("expr", src))
                    return _copy.deepcopy(exprs[src])
            return super().visit(nd)
    off = Off()
    node = off.visit(node)
    node = Blocks().visit(node)
    node = Lines().visit(node)
    node = Exprs().visit(node)
    node = Rename().visit(node)
    return node, Off.n, used


def _free_globals(node):
    """Names a function reads that it neither assigns nor takes as
    arguments, nor are builtins: its module globals."""
    import ast
    import builtins
    local = {a.arg for a in node.args.args + node.args.kwonlyargs}
    for a in (node.args.vararg, node.args.kwarg):
        if a is not None:
            local.add(a.arg)
    for nd in ast.walk(node):
        if isinstance(nd, ast.Name) and isinstance(nd.ctx, ast.Store):
            local.add(nd.id)
        elif isinstance(nd, ast.Lambda):
            local |= {a.arg for a in nd.args.args + nd.args.kwonlyargs}
        elif isinstance(nd, (ast.Import, ast.ImportFrom)):
            for al in nd.names:
                local.add((al.asname or al.name).split(".")[0])
    return sorted({nd.id for nd in ast.walk(node) if isinstance(nd, ast.Name)
                   and isinstance(nd.ctx, ast.Load) and nd.id not in local
                   and not hasattr(builtins, nd.id)})


def copy_check(T, report):
    import ast
    import difflib
    from sim import planing_anyspeed as PA
    rows = []
    for ref, orig, drop, subst, alias, n_sw, exprs, blocks in COPIES:
        cp, n, used = _project(_fn_node(ref), drop, subst, alias, exprs,
                               blocks)
        v0 = _fn_node(orig)
        cp.name = v0.name
        same = ast.dump(cp.args) == ast.dump(v0.args) and \
            ast.dump(ast.Module(cp.body, [])) == ast.dump(
                ast.Module(v0.body, []))
        want = (len(drop) + len(subst) + len(alias) + len(blocks)
                + sum(k for _, k in exprs.values()))
        diff = [] if same else list(difflib.unified_diff(
            ast.unparse(v0).splitlines(), ast.unparse(cp).splitlines(),
            "v0", "copy (off branches)", lineterm="", n=1))[:30]
        rows.append(dict(copy=ref, v0=orig, identical=same, switch_blocks=n,
                         allowed_used=used, diff=diff))
        T("freeze", f"copy {ref.split(':')[1]} on its off branches == "
          f"{orig.split(':')[1]} (source AST; {n} switch block(s), "
          f"{len(used)} allowed line(s))",
          same and n == n_sw and len(used) == want,
          ("identical" if same else " | ".join(diff[3:12]))
          + (f"; switch blocks {n}, expected {n_sw}" if n != n_sw else "")
          + (f"; allowed lines used {len(used)} of {want}"
             if len(used) != want else ""))
    # the check can fail: a mutation on an off path of the copied
    # derivative (cross-flow drag 0.5 -> 0.4) must be caught, one inside a
    # switch's on branch (the S1 call) must not; in the copied constructor
    # a mutated geometry block line and a parameter put where v0 has
    # another literal must be caught
    res = {}
    for k, lab, a, b in (
            (0, "off path", "cf = 0.5 * rho", "cf = 0.4 * rho"),
            (0, "on branch", "flux = self._flux_dDt(d, V, u, mom, "
             "s_lon - tau)", "flux = self._flux_dDt(d, V, u, mom, "
             "s_lon + tau)"),
            # the else of a block 'if self.sw.X and <cond>:' is an off path,
            # its body an on branch (P1b review)
            (0, "off path under 'and'", "e, et, ett, ex, ey = self._waves(X,"
             " Y, t)", "e, et, ett, ex, ey = self._waves(Y, X, t)"),
            (0, "on branch under 'and'", "Ydf = a22d * aoyb",
             "Ydf = 2.0 * a22d * aoyb"),
            (1, "geometry block", "self.zk = -h.vcg + g.rise",
             "self.zk = -h.vcg + g.rise * 1.0"),
            (0, "parameter moved", "cf = 0.5 * rho",
             "cf = self.num.z_air * rho")):
        ref, orig, drop, subst, alias, n_sw, exprs, blocks = COPIES[k]
        src = _fn_source(ref)
        v0 = _fn_node(orig)
        assert src.count(a) == 1, a
        cp, _, used = _project(_fn_node(ref, src.replace(a, b)), drop, subst,
                               alias, exprs, blocks)
        cp.name = v0.name
        want = (len(drop) + len(subst) + len(alias) + len(blocks)
                + sum(c for _, c in exprs.values()))
        res[lab] = ast.dump(ast.Module(cp.body, [])) == ast.dump(
            ast.Module(v0.body, [])) and len(used) == want
    T("freeze", "copy check self-test: an off-path mutation (also under a "
      "'self.sw.X and' block), a mutated geometry-block line and a parameter"
      " moved onto another literal are caught, an on-branch change is not",
      res == {"off path": False, "on branch": True,
              "off path under 'and'": False, "on branch under 'and'": True,
              "geometry block": False, "parameter moved": False}, res)
    fg = _free_globals(_fn_node("sim.planing_anyspeed:_StripAS.deriv"))
    unguarded = sorted(set(fg) - set(PA.V0_GLOBALS) - {"np"})
    T("freeze", "the copied derivative reads no v0 module global that "
      "check_v0_globals does not guard", not unguarded,
      f"globals {fg}; unguarded {unguarded}")
    fi = _free_globals(_fn_node("sim.planing_anyspeed:_StripAS.__init__"))
    ung_i = sorted(set(fi) - set(PA.V0_GLOBALS)
                   - {"np", "SimpleNamespace", "_vs", "_hg"})
    T("freeze", "the copied constructor reads no v0 module global that "
      "check_v0_globals does not guard", not ung_i,
      f"globals {fi}; unguarded {ung_i}")
    # the parameters stand for v0's literals: their values for scarab195
    from hydro import hulls
    from hydro import vessel_spec as VS
    import sim.planing_vessel as PV
    num = VS.numerics_of(hulls.get("scarab195"))
    bad = {}
    for expr, lit in _PARAM_LITERALS.items():
        if expr.endswith(" - 1"):
            continue
        val = getattr(num, expr.split(".")[-1])
        want = getattr(PV, lit) if lit.isidentifier() else ast.literal_eval(
            lit)
        if val != want or type(val) is not type(want):
            bad[expr] = (val, want)
    jd = VS.JetSpec.defaults()
    for expr, (lit, _) in _JET_EXPRS.items():
        key = ast.literal_eval(expr.split("(", 1)[1].split(",")[0])
        if jd[key] != ast.literal_eval(lit):
            bad[expr] = (jd[key], lit)
    # every j.get(key, default) of the copied constructor: JetSpec's default
    # is the copy's (= v0's) literal
    for nd in ast.walk(_fn_node("sim.planing_anyspeed:_StripAS.__init__")):
        if (isinstance(nd, ast.Call) and isinstance(nd.func, ast.Attribute)
                and nd.func.attr == "get" and isinstance(nd.func.value,
                                                         ast.Name)
                and nd.func.value.id == "j" and len(nd.args) == 2):
            key, dflt = nd.args[0].value, ast.literal_eval(nd.args[1])
            if jd.get(key) != dflt:
                bad[f"j.get({key!r})"] = (jd.get(key), dflt)
    T("freeze", "the plant constants and jet defaults (hydro/vessel_spec) "
      "for scarab195 are v0's literals the copy check maps them to",
      not bad, bad or f"{len(_PARAM_LITERALS)} constants, "
      f"{len(jd)} jet defaults")
    report["copy_check"] = dict(rows=rows, deriv_globals=fg,
                                init_globals=fi)

# ---------------------------------------------------------------- select
def _short_run(p, n):
    """n plant steps from 5 m/s with moving thrust and nozzle: the states."""
    s = p.initial_state(5.0)
    tm, rm = p.prop.t_max, p.rudder.max
    out = [s.copy()]
    for i in range(n):
        s = p.step(s, i * p.dt, tm * (0.5 + 0.4 * np.sin(0.07 * i)),
                   rm * 0.7 * np.sin(0.05 * i), p.dt)
        out.append(s.copy())
    return np.asarray(out)


def _probe(n):
    """In a spawned worker: the plant of the hull key handed down through
    USV_HULL (default the plain scarab195; hull.plant builds the linear-roll
    class), whether the other key is refused, and a short run of the
    plant."""
    warnings.filterwarnings("ignore")
    from hydro import hulls
    from sim import physics
    key = physics.hull_key("scarab195")
    p = hulls.get(key).plant(_calm())
    other = AS_KEY if key == "scarab195" else "scarab195"
    try:
        hulls.get(other).plant(_calm())
        refused = False
    except RuntimeError:
        refused = True
    return dict(env=os.environ.get(physics.ENV), key=key,
                cls=type(p).__name__, physics=physics.of(p),
                switches=list(getattr(p, "switches", ())),
                refused_other=refused, run=_short_run(p, n))


def _spawn(key, n):
    """_probe in a fresh spawned process that inherits USV_HULL = key."""
    from sim import physics
    prev = physics.use(key)
    try:
        fn = importlib.import_module("studies.anyspeed_regress_as")._probe
        with mp.get_context("spawn").Pool(1) as pool:
            return pool.apply(fn, (n,))
    finally:
        physics.use(prev)


def _import_closure(entries):
    """Repository files reachable by imports (module level or inside
    functions) from the entry files: {relative path: source}."""
    import ast
    seen, todo = {}, list(entries)
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            continue
        src = open(path, encoding="utf-8", errors="replace").read()
        seen[rel] = src
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        pkg = os.path.dirname(rel).replace(os.sep, "/").split("/")
        for nd in ast.walk(tree):
            names = []
            if isinstance(nd, ast.Import):
                names = [a.name for a in nd.names]
            elif isinstance(nd, ast.ImportFrom):
                base = (nd.module or "").split(".") if nd.module else []
                if nd.level:
                    up = pkg[:len(pkg) - nd.level + 1] if nd.level > 1 \
                        else pkg
                    base = [p for p in up if p] + base
                names = [".".join(base)] + [".".join(base + [a.name])
                                            for a in nd.names]
            for m in names:
                parts = [p for p in m.split(".") if p]
                for cand in ("/".join(parts) + ".py",
                             "/".join(parts) + "/__init__.py"):
                    if os.path.exists(os.path.join(ROOT, cand)):
                        todo.append(cand)
    return seen


PENDING_SKIP = ("sim/planing_vessel.py", "sim/planing_vessel_gz.py",
                "sim/physics.py", "sim/planing_anyspeed.py")
PENDING_CALLS = ("from_plant", "PlaningVesselGZ", "PlaningVessel")


def _pending_in(src):
    """(line, what) of the calls in src that build v0 whatever the hull
    says: from_plant(...), PlaningVesselGZ(...), PlaningVessel(...) (by name
    or as an attribute), plant_for(...) of sim/planing_vessel_gz (imported
    by name or through a module alias), and the import of that plant_for.
    Found on the AST, so docstrings and comments do not count."""
    import ast
    tree = ast.parse(src)
    gz_mods, gz_pf = set(), set()
    hits = []
    for nd in ast.walk(tree):
        if isinstance(nd, ast.Import):
            for a in nd.names:
                if a.name.endswith("planing_vessel_gz"):
                    gz_mods.add(a.asname or a.name)
        elif isinstance(nd, ast.ImportFrom):
            mod = nd.module or ""
            for a in nd.names:
                if a.name == "planing_vessel_gz":
                    gz_mods.add(a.asname or a.name)
                if mod.endswith("planing_vessel_gz") and a.name == "plant_for":
                    gz_pf.add(a.asname or a.name)
                    hits.append((nd.lineno, "import plant_for of "
                                 "sim/planing_vessel_gz"))
    for nd in ast.walk(tree):
        if not isinstance(nd, ast.Call):
            continue
        f = nd.func
        if isinstance(f, ast.Name) and (f.id in PENDING_CALLS
                                        or f.id in gz_pf):
            hits.append((nd.lineno, f.id + "(...)"))
        elif isinstance(f, ast.Attribute):
            if f.attr in PENDING_CALLS or (
                    f.attr == "plant_for" and isinstance(f.value, ast.Name)
                    and f.value.id in gz_mods):
                hits.append((nd.lineno, ast.unparse(f) + "(...)"))
    return sorted(set(hits))


def pending_repo():
    """The same scan over every Python file of the repository (not the
    virtual environment, caches, the frozen v0 files, sim/physics.py and
    sim/planing_anyspeed.py): direct v0 constructor and frozen-helper calls
    build v0's geometry from the hull fields whatever the hull is -- given a
    hull v0 cannot honour (a parametric hull with a transom run, an offsets
    table) they silently build another boat. Listed, not counted: most are
    deliberate v0 references in studies; new studies build through
    physics.build / hull.plant, and physics.verify refuses a v0 plant built
    for a hull v0 cannot honour. Returns (hits, files scanned)."""
    skip_dirs = {".venv", "_cache", ".git", "__pycache__", "node_modules",
                 ".claude"}
    hits, n = [], 0
    for dp, dns, fns in os.walk(ROOT):
        dns[:] = [d for d in dns if d not in skip_dirs]
        for fn in fns:
            if not fn.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(dp, fn), ROOT).replace(
                os.sep, "/")
            if rel in PENDING_SKIP:
                continue
            n += 1
            try:
                src = open(os.path.join(ROOT, rel), encoding="utf-8",
                           errors="replace").read()
                found = _pending_in(src)
            except (SyntaxError, ValueError):
                continue
            for ln, what in found:
                hits.append(f"{rel}:{ln}: {what}")
    return sorted(hits), n


def pending_callers():
    """Call sites of the frozen helpers and direct v0 constructors in the
    non-frozen code reachable from studies/final_eval.py and
    studies/safety_events.py: they build v0 whatever the hull says and must
    switch to physics.gz_twin / gz_plant_for (and records to
    physics.verify) before any as1 closed-loop run (S9)."""
    files = _import_closure(["studies/final_eval.py",
                             "studies/safety_events.py"])
    hits = []
    for rel, src in sorted(files.items()):
        if rel in PENDING_SKIP:
            continue
        try:
            found = _pending_in(src)
        except SyntaxError:
            continue
        lines = src.splitlines()
        for ln, what in found:
            hits.append(f"{rel}:{ln}: {what}  | {lines[ln - 1].strip()[:70]}")
    return hits, sorted(files)


def tier_select(T, report):
    from hydro import hulls
    from sim import config, lofi, physics
    from sim import planing_vessel as PV
    from sim.planing_anyspeed import (PRESETS, PlaningVesselAS,
                                      PlaningVesselASLin, TEST_KW)
    from sim.planing_vessel import PlaningVessel
    from sim.planing_vessel_gz import PlaningVesselGZ, from_plant
    calm = _calm()
    tier = "select"
    # ---- v0: the plain key, exactly as before
    h0 = hulls.get("scarab195")
    T(tier, "v0 key: name scarab195, version v0, no new instance field",
      h0.name == "scarab195" and physics.version(h0) == "v0"
      and "physics" not in vars(h0))
    p0 = h0.plant(calm)
    T(tier, "v0 key: hull.plant -> PlaningVessel", type(p0) is PlaningVessel,
      type(p0).__name__)
    # the as plant's geometry from hydro/hull_geometry against v0's own
    # lines: array for array, and everything built from it
    pg = PlaningVesselASLin(hulls.get(AS_KEY), calm, dt=p0.dt,
                            test_switches="v0")
    keys = ("x", "dx", "yc", "zk", "kp", "kpp", "Z_rest", "Th_rest", "T",
            "GM", "M", "Mtot", "visc", "draft_bow", "freeboard", "v_slam")
    neq = [k for k in keys if not np.array_equal(getattr(pg, k),
                                                 getattr(p0, k))]
    jet = [k for k in ("t_max", "t_min", "tau", "rate_max", "dead_band",
                       "tau_prime_loss", "tau_prime_back", "prime_band",
                       "k_rev", "u_ref")
           if getattr(pg.prop, k) != getattr(p0.prop, k)]
    T(tier, "geometry (hydro/hull_geometry) and jet of the as plant == v0's "
      "own lines, exactly", not neq and not jet
      and pg.thrusters == p0.thrusters and pg.intakes == p0.intakes,
      f"differ: {neq + jet}" if neq or jet else f"{len(keys)} arrays, "
      f"10 jet constants")
    q0 = config.plant_for(None, calm, "scarab195")
    T(tier, "v0 key: config.plant_for -> PlaningVessel",
      type(q0) is PlaningVessel, type(q0).__name__)
    g0 = physics.gz_twin(p0, record=True)
    T(tier, "v0 key: physics.gz_twin -> PlaningVesselGZ (= from_plant)",
      type(g0) is PlaningVesselGZ and g0.record, type(g0).__name__)
    z0 = physics.gz_plant_for(None, calm, h0)
    T(tier, "v0 key: physics.gz_plant_for -> PlaningVesselGZ",
      type(z0) is PlaningVesselGZ, type(z0).__name__)
    T(tier, "v0 key: physics.of / verify -> v0",
      physics.of(g0) == "v0" and physics.verify(g0, h0) == "v0")
    T.raises(tier, f"v0 key: hull.plant({TEST_KW}=...) refused (a v0 plant "
             "would swallow it)", TypeError,
             lambda: h0.plant(calm, test_switches="as1"))
    T.raises(tier, f"v0 key: physics.gz_plant_for({TEST_KW}=...) refused",
             TypeError, lambda: physics.gz_plant_for(
                 None, calm, h0, test_switches="as1"))
    T.raises(tier, "v0 key: hull.plant(switches=...) refused", TypeError,
             lambda: h0.plant(calm, switches="as1"))
    # ---- as1: the '@as1' key
    h1 = hulls.get(AS_KEY)
    T(tier, f"as1 key: name {AS_KEY}, version as1",
      h1.name == AS_KEY and physics.version(h1) == "as1")
    p1 = h1.plant(calm)
    T(tier, "as1 key: hull.plant -> PlaningVesselASLin (linear roll, as v0's "
      "PlaningVessel) on the as1 preset",
      type(p1) is PlaningVesselASLin and p1.physics == "as1"
      and p1.switches == PRESETS["as1"],
      f"{type(p1).__name__}, switches {p1.switches}")
    q1 = config.plant_for(None, calm, AS_KEY)
    T(tier, "as1 key: config.plant_for(name) -> PlaningVesselASLin",
      type(q1) is PlaningVesselASLin, type(q1).__name__)
    hl, db = config.load(AS_KEY)
    T(tier, "as1 key: config.load(name) -> the as1 hull, no database",
      hl.name == AS_KEY and db is None)
    w1 = p1.with_sea(calm)
    T(tier, "as1: PlaningVesselASLin.with_sea keeps its class (v0's "
      "hard-codes PlaningVessel)", type(w1) is PlaningVesselASLin
      and w1.switches == p1.switches, type(w1).__name__)
    g1 = physics.gz_twin(p1, record=True)
    T(tier, "as1: physics.gz_twin(PlaningVesselASLin) -> PlaningVesselAS, "
      "switches kept", type(g1) is PlaningVesselAS and g1.record
      and g1.switches == p1.switches, type(g1).__name__)
    T(tier, "as1: physics.gz_twin(PlaningVesselAS) -> PlaningVesselAS",
      type(physics.gz_twin(g1)) is PlaningVesselAS)
    dt1 = h1.scales()["dt"]
    off = PlaningVesselAS(h1, calm, dt=dt1, test_switches="v0")
    offl = PlaningVesselASLin(h1, calm, dt=dt1, test_switches="v0")
    w = off.with_sea(calm)
    t = physics.gz_twin(off)
    tl = physics.gz_twin(offl)
    T(tier, "as1: all-off test switches carried by with_sea and gz_twin "
      "(both classes)",
      type(w) is PlaningVesselAS and w.switches == ()
      and type(t) is PlaningVesselAS and t.switches == ()
      and type(tl) is PlaningVesselAS and tl.switches == ()
      and type(offl.with_sea(calm)) is PlaningVesselASLin
      and offl.with_sea(calm).switches == ())
    T(tier, f"as1: the preset's plant carries no {TEST_KW} in _init_kw",
      TEST_KW not in p1._init_kw and TEST_KW not in g1._init_kw
      and off._init_kw.get(TEST_KW) == ())
    z1 = physics.gz_plant_for(None, calm, h1)
    T(tier, "as1: physics.gz_plant_for -> PlaningVesselAS",
      type(z1) is PlaningVesselAS, type(z1).__name__)
    T(tier, "as1: physics.of / verify -> as1 (both classes)",
      physics.of(g1) == "as1" and physics.verify(g1, h1) == "as1"
      and physics.verify(p1, h1) == "as1")
    T.raises(tier, "physics.verify refuses an as plant on test switches",
             RuntimeError, lambda: physics.verify(off))
    T(tier, "... unless allow_test_switches=True",
      physics.verify(off, allow_test_switches=True) == "as1")
    T.raises(tier, "the old switches= keyword is refused", TypeError,
             lambda: PlaningVesselAS(h1, calm, switches="v0"))
    T.raises(tier, "hull.plant(switches=...) on an as1 hull refused",
             TypeError, lambda: h1.plant(calm, switches="v0"))
    st = physics.stamp(g1)
    T(tier, "as1: stamp carries version, switches, hull key, sha1",
      st["physics"] == "as1" and st["hull"] == AS_KEY
      and len(st["physics_sha1"]) == 40, st)
    T(tier, "as1: the stamp is taken at build (plant._stamp) with the code "
      "hash of the import (memoised; the code on disk unchanged since)",
      st == g1._stamp and st == physics.stamp(g1, fresh=True)
      and set(physics._CODE) == set(physics.VERSIONS)
      and not physics.code_changed("as1") and not physics.code_changed("v0"),
      {v: physics.code_sha1(v)[:12] for v in physics.VERSIONS})
    report["stamp_as1"] = st
    # sim/env.Episode and sim/rl_env build through config.plant_for
    from types import SimpleNamespace
    from sim.env import Episode
    stub = SimpleNamespace(p=dict(L=h1.L, B=h1.b, dt_ctrl=0.25,
                                  u_design=h1.u_design, t_max=1e4,
                                  rud_max=0.4))
    ep = Episode(None, stub, hs=1.0, tp=5.0, n_freq=4, n_dir=1,
                 hull=AS_KEY)
    T(tier, "as1: sim/env.Episode(hull='scarab195@as1') -> "
      "PlaningVesselASLin", type(ep.plant) is PlaningVesselASLin
      and ep.plant.switches == PRESETS["as1"], type(ep.plant).__name__)
    ep0 = Episode(None, stub, hs=1.0, tp=5.0, n_freq=4, n_dir=1,
                  hull="scarab195")
    T(tier, "v0: sim/env.Episode(hull='scarab195') -> PlaningVessel",
      type(ep0.plant) is PlaningVessel, type(ep0.plant).__name__)
    from sim.rl_env import USVControlEnv
    env = USVControlEnv(hull=AS_KEY, hs=1.0, tp=5.0, n_freq=4, n_dir=1,
                        seeds=(0,))
    env.reset(seed=0)
    _, rw, *_ = env.step(np.array([0.2, 0.0]))
    T(tier, "as1: sim/rl_env.USVControlEnv(hull='scarab195@as1') -> "
      "PlaningVesselASLin, steps", type(env._plant) is PlaningVesselASLin
      and np.isfinite(rw), f"{type(env._plant).__name__}, reward {rw:.3f}")
    pa, pb = lofi._cache_path(h0.name), lofi._cache_path(h1.name)
    T(tier, "lofi: the as1 hull's reduced-model cache is its own file",
      os.path.normcase(pa) == os.path.normcase(AR.LOFI) and pa != pb
      and os.path.basename(pb) == "lofi_scarab195@as1.json",
      os.path.basename(pb))
    # ---- refusals
    T.raises(tier, "a v0 hull never yields PlaningVesselAS", ValueError,
             lambda: PlaningVesselAS(hulls.get("scarab195"), calm))
    T.raises(tier, "a v0 hull never yields PlaningVesselASLin", ValueError,
             lambda: PlaningVesselASLin(hulls.get("scarab195"), calm))

    def by_hand():
        h = hulls.get("scarab195")
        h.physics = "as1"                 # without the key: refused
        return h.plant(calm)
    T.raises(tier, "a v0 name with physics='as1' set by hand", ValueError,
             by_hand)
    T.raises(tier, "'scarab195@v0' refused (v0 is the plain name)",
             KeyError, lambda: hulls.get("scarab195@v0"))
    T.raises(tier, "unknown version refused", KeyError,
             lambda: hulls.get("scarab195@as9"))
    T.raises(tier, "unknown vessel refused", KeyError,
             lambda: hulls.get("nope@as1"))
    T.raises(tier, "non-planing hull refused", ValueError,
             lambda: hulls.get("wigley10@as1"))
    T.raises(tier, "variant of a variant refused", ValueError,
             lambda: physics.variant(h1, "as1"))
    T.raises(tier, "unimplemented switch refused", NotImplementedError,
             lambda: PlaningVesselAS(h1, calm,
                                     test_switches=("radiation_3d",)))
    T.raises(tier, "unknown switch refused (P1b's orbital replaced the "
             "planned wave_kinematics)", KeyError,
             lambda: PlaningVesselAS(h1, calm,
                                     test_switches=("wave_kinematics",)))

    def garme():
        h = physics.variant(hulls.get("scarab195"), "as1")
        h.transom_correction = True
        return PlaningVesselAS(h, calm, test_switches=("dDt_full",))
    T.raises(tier, "Garme transom correction with dDt_full refused",
             ValueError, garme)
    T.raises(tier, "unknown switch refused", KeyError,
             lambda: PlaningVesselAS(h1, calm, test_switches=("bogus",)))
    # every 'needs' of SWITCHES enforced (sim/planing_anyspeed.resolve)
    from sim.planing_anyspeed import SWITCHES
    for name, spec in SWITCHES.items():
        for need in spec.get("needs", ()):
            # the switch with its other needs met and this one missing
            on_ = (name,) + tuple(n for n in spec["needs"] if n != need)
            try:
                PlaningVesselAS(h1, calm, test_switches=on_)
                ok, msg = False, "did not raise"
            except ValueError as e:
                msg = str(e)
                ok = msg.startswith(f"switch {name} needs") and need in msg
            except Exception as e:      # noqa: BLE001
                ok, msg = False, f"raised {type(e).__name__}: {e}"
            T(tier, f"switch {name} without {need} refused (test_switches="
              f"{on_})", ok, msg[:110])
    # the transom closure's alternative modes only in a sensitivity study
    from hydro import vessel_spec as VS
    for mode in VS.TRANSOM_MODES:
        if mode == VS.TRANSOM_PRIMARY:
            continue
        T.raises(tier, f"transom closure mode {mode!r} without the "
                 "sensitivity flag refused", ValueError,
                 lambda mode=mode: VS.with_transom_closure(
                     hulls.get("scarab195"), f"scarab195_{mode}", mode=mode))
        hs = VS.with_transom_closure(hulls.get("scarab195"),
                                     f"scarab195_{mode}_s", mode=mode,
                                     sensitivity=True)
        T(tier, f"... and admitted with sensitivity=True (mode {mode})",
          type(PlaningVesselAS(hs, calm)) is PlaningVesselAS)
    # a hull modified under a registered name would share its caches

    def modified():
        import copy
        h = copy.deepcopy(hulls.get("scarab195"))
        h.n_stations = 96
        return h
    T.raises(tier, "variant() refuses a modified hull under its registered "
             "name", ValueError, lambda: physics.variant(modified(), "as1"))
    hn = physics.variant(modified(), "as1", name="scarab195_n96")
    T(tier, "... and takes it under a distinct name",
      hn.name == "scarab195_n96@as1" and physics.version(hn) == "as1",
      hn.name)
    T.raises(tier, "variant(name=<a registered name>) refused", ValueError,
             lambda: physics.variant(modified(), "as1", name="scarab195"))
    from hydro import vessel_spec as VS
    tw = VS.froude_twin(hulls.get("scarab195"), 1.5)
    T.raises(tier, "a hull v0 cannot honour (a Froude twin: scaled plant "
             "constants) is refused by the v0 dispatch", ValueError,
             lambda: tw.plant(calm))
    T(tier, "... and builds as NAME@as1",
      type(physics.variant(tw, "as1").plant(calm)) is PlaningVesselASLin)
    # a patched v0 module global (studies/principles_hull_swap sets PV.NU)
    # reaches v0's derivative and not the copy's: the as plants refuse
    nu0 = PV.NU
    run_p = PlaningVesselAS(h1, calm, dt=dt1)
    try:
        PV.NU = PV.NU * 2.0
        T.raises(tier, "PV.NU patched: building an as plant raises",
                 RuntimeError, lambda: PlaningVesselASLin(h1, calm, dt=dt1))
        T.raises(tier, "PV.NU patched: stepping a built as plant raises",
                 RuntimeError, lambda: run_p.step(
                     np.r_[np.zeros(6), 5.0, np.zeros(7)], 0.0, 0.0, 0.0))
        T.raises(tier, "PV.NU patched: running_attitude raises",
                 RuntimeError, lambda: run_p.running_attitude(3.21))
    finally:
        PV.NU = nu0
    T(tier, "PV.NU restored (the same object): as plants build again",
      PV.NU is nu0 and type(PlaningVesselASLin(h1, calm, dt=dt1))
      is PlaningVesselASLin)
    fz = from_plant(p1)
    T(tier, "frozen from_plant on an as1 plant builds v0 PlaningVesselGZ "
      "(it cannot see the hull)", type(fz) is PlaningVesselGZ,
      type(fz).__name__)
    T.raises(tier, "... and physics.verify catches it", RuntimeError,
             lambda: physics.verify(fz))
    # ---- the environment variable carries a key and must agree
    prev = physics.use(AS_KEY)
    try:
        T.raises(tier, f"USV_HULL={AS_KEY}: building plain scarab195 "
                 "raises", RuntimeError,
                 lambda: hulls.get("scarab195").plant(calm))
        T.raises(tier, "... through config.plant_for too", RuntimeError,
                 lambda: config.plant_for(None, calm, "scarab195"))
        T.raises(tier, "... and through physics.gz_twin of a v0 plant",
                 RuntimeError, lambda: physics.gz_twin(p0))
        pk = hulls.get(physics.hull_key("scarab195")).plant(calm)
        T(tier, f"USV_HULL={AS_KEY}: hull_key -> the as1 plant builds",
          type(pk) is PlaningVesselASLin, type(pk).__name__)
        physics.use("scarab195")
        T.raises(tier, "USV_HULL=scarab195: building the as1 hull raises",
                 RuntimeError, lambda: hulls.get(AS_KEY).plant(calm))
        T.raises(tier, "physics.use refuses an invalid key", KeyError,
                 lambda: physics.use("scarab195@v0"))
    finally:
        physics.use(prev)
    T(tier, "USV_HULL restored", os.environ.get(physics.ENV) == prev)
    # ---- spawned workers, the key handed down through USV_HULL; the
    # parent's references at the hull's own time step, as hull.plant builds
    t0 = time.time()
    dt = h1.scales()["dt"]
    ref1 = _short_run(PlaningVesselASLin(h1, calm, dt=dt), N_PROBE)
    refl = _short_run(PlaningVesselASLin(h1, calm, dt=dt,
                                         test_switches="v0"), N_PROBE)
    refo = _short_run(PlaningVesselAS(h1, calm, dt=dt, test_switches="v0"),
                      N_PROBE)
    refg = _short_run(PlaningVesselGZ(h0, calm, dt=dt), N_PROBE)
    ref0 = _short_run(p0, N_PROBE)
    r1 = _spawn(AS_KEY, N_PROBE)
    T(tier, f"spawned worker, USV_HULL={AS_KEY}: hull.plant -> "
      "PlaningVesselASLin on as1, plain scarab195 refused",
      r1["env"] == AS_KEY and r1["key"] == AS_KEY
      and r1["cls"] == "PlaningVesselASLin" and r1["physics"] == "as1"
      and tuple(r1["switches"]) == PRESETS["as1"] and r1["refused_other"],
      {k: v for k, v in r1.items() if k != "run"})
    T(tier, f"... its {N_PROBE}-step run == the parent's PlaningVesselASLin "
      "on the as1 preset, exactly", np.array_equal(r1["run"], ref1),
      f"max |diff| {np.abs(r1['run'] - ref1).max():.3e}")
    T(tier, f"parent: all-off PlaningVesselASLin {N_PROBE}-step run == v0 "
      "PlaningVessel, exactly", np.array_equal(refl, ref0),
      f"max |diff| {np.abs(refl - ref0).max():.3e}")
    T(tier, f"parent: all-off PlaningVesselAS {N_PROBE}-step run == v0 "
      "PlaningVesselGZ, exactly", np.array_equal(refo, refg),
      f"max |diff| {np.abs(refo - refg).max():.3e}")
    dif = float(np.abs(ref1 - ref0).max())
    T(tier, f"parent: as1 preset {PRESETS['as1']} "
      + ("differs from v0 (a switch is on)" if PRESETS["as1"] else
         "== v0 (no switch on yet)"),
      (dif > 0.0) == bool(PRESETS["as1"]), f"max |diff| {dif:.3e}")
    r0 = _spawn(None, N_PROBE)
    T(tier, "spawned worker, USV_HULL unset: default key -> PlaningVessel",
      r0["env"] is None and r0["key"] == "scarab195"
      and r0["cls"] == "PlaningVessel" and r0["physics"] == "v0"
      and not r0["refused_other"],
      {k: v for k, v in r0.items() if k != "run"})
    T(tier, f"... its {N_PROBE}-step run == the parent's v0, exactly",
      np.array_equal(r0["run"], ref0),
      f"max |diff| {np.abs(r0['run'] - ref0).max():.3e}; "
      f"{time.time() - t0:.0f} s")
    # ---- pending: what still builds v0 whatever the hull says (S9)
    hits, files = pending_callers()
    report["pending"] = dict(
        frozen_helper_calls=hits, files_scanned=len(files),
        mission="learn/repro/task.py:55 hard-codes HULL = 'scarab195' (a "
                "hashed file): Mission, and so final_eval and safety_events, "
                "cannot select as1 -- to decide at S9")
    print(f"  [pending] {len(hits)} call(s) of the frozen helpers / direct "
          f"v0 constructors in the {len(files)} files reachable from "
          f"final_eval and safety_events (switch to physics.gz_twin / "
          f"gz_plant_for and physics.verify before S9):", flush=True)
    for h in hits:
        print(f"      {h}", flush=True)
    print(f"  [pending] {report['pending']['mission']}", flush=True)
    rh, rn = pending_repo()
    by_file = {}
    for h_ in rh:
        f_ = h_.split(":", 1)[0]
        by_file[f_] = by_file.get(f_, 0) + 1
    report["pending"]["repo_direct_v0_calls"] = rh
    report["pending"]["repo_files_scanned"] = rn
    print(f"  [pending] repository-wide: {len(rh)} direct v0 constructor / "
          f"frozen-helper call(s) in {len(by_file)} of {rn} files (listed in "
          f"the report; physics.verify refuses a v0 plant built for a hull "
          f"v0 cannot honour)", flush=True)
    # ---- the v0 route is the frozen helper itself, actuators included;
    # an as plant carries given actuators (the one all-off != v0 case)
    import copy as _copy
    from sim.waterjet import Waterjet as _WJ
    ref = h0.plant(calm)
    jet = _WJ(_copy.deepcopy(ref.prop.pump), ref.prop.x_nozzle,
              ref.prop.z_nozzle, ref.prop.x_intake, ref.prop.z_intake,
              t_max=ref.prop.t_max, t_min=ref.prop.t_min, tau=0.9,
              rate_max=ref.prop.rate_max, dead_band=ref.prop.dead_band,
              delay=0.2, u_ref=ref.prop.u_ref, dt=ref.dt)
    pv = h0.plant(calm, prop=_copy.deepcopy(jet))
    gv, fv = physics.gz_twin(pv), from_plant(pv)
    pa = h1.plant(calm, prop=_copy.deepcopy(jet))
    ga = physics.gz_twin(pa)
    T(tier, "v0 plant built with prop=: physics.gz_twin IS the frozen "
      "from_plant (drops it, as v0 always did); an as plant hands it on "
      "(sim/planing_anyspeed 'ACTUATORS PASSED IN ARE KEPT')",
      pv.prop.tau == 0.9 and gv.prop.tau == fv.prop.tau == 0.6
      and not hasattr(pv, "_ext_v0") and ga.prop.tau == 0.9
      and ga.prop is not pa.prop,
      f"v0: built {pv.prop.tau}, gz_twin {gv.prop.tau}, from_plant "
      f"{fv.prop.tau}; as1: gz_twin {ga.prop.tau}")
    # ---- physics.verify refuses a v0 plant built directly for a hull the
    # frozen plant cannot honour (it builds v0's geometry from the fields)
    hb = hulls.get("planing_b")
    pb = PlaningVessel(hb, calm, dt=0.02)
    T.raises(tier, "physics.verify refuses a v0 PlaningVessel built directly "
             "for planing_b (geometry outside v0's family member)",
             RuntimeError, lambda: physics.verify(pb))


# ---------------------------------------------------------------- golden
def tier_golden(T, report, on_mode="check"):
    from hydro import hulls
    from sim import physics
    from sim.planing_anyspeed import P1A_CANDIDATE, PRESETS, PlaningVesselAS
    ref = np.load(AR.GOLDEN)
    R = {k: ref[k] for k in ref.files}
    report["golden"] = []

    def check(name, twin=None, ref=R, extra=(), expect=True, opts=None):
        """golden_runs through twin (keywords opts), compared with ref:
        expect True -> identical, False -> different. Returns (plants
        built, runs)."""
        built = []

        def tw(plant):
            p = twin(plant)
            built.append(p)
            return p
        t0 = time.time()
        g = AR.golden_runs(None if twin is None else tw, extra=extra,
                           **(opts or {}))
        d = AR.diff(g, ref)
        same = not d and set(g) == set(ref)
        report["golden"].append(dict(name=name, identical=same,
                                     expected_identical=expect,
                                     diffs=d[:20],
                                     seconds=time.time() - t0))
        T("golden", name, same == expect, f"{len(g)} arrays, {len(d)} "
          f"differ, {time.time() - t0:.1f} s {d[:3]}")
        return built, g

    check("v0 PlaningVesselGZ (first) == golden_v0")
    h1 = hulls.get(AS_KEY)
    b, _ = check("PlaningVesselAS all off (preset 'v0') == golden_v0",
                 lambda p: PlaningVesselAS(h1, p.sea, dt=p.dt, record=True,
                                           test_switches="v0"))
    T("golden", "... it ran PlaningVesselAS with no switch on",
      len(b) == 1 and type(b[0]) is PlaningVesselAS and b[0].switches == ())
    on = PRESETS["as1"]
    selector = (lambda p: physics.gz_twin(hulls.get(AS_KEY).plant(
        p.sea, dt=p.dt), record=True))
    b, _ = check(f"PlaningVesselAS by the selector ({AS_KEY}, as1 preset "
                 f"{on}) " + ("!= golden_v0 (switches on)" if on else
                              "== golden_v0 (no switch yet)"),
                 selector, expect=not on)
    T("golden", "... it ran PlaningVesselAS on the as1 preset",
      len(b) == 1 and type(b[0]) is PlaningVesselAS
      and b[0].physics == "as1" and b[0].switches == on,
      b[0].switches if b else None)
    if on:
        path = os.path.join(AR.OUT, f"golden_as_{'+'.join(on)}.npz")
        g_on = AR.golden_runs(selector, extra=ON_EXTRA)
        if on_mode == "write":
            if os.path.exists(path):
                raise SystemExit(f"{path} exists; delete it first")
            np.savez(path, **g_on)
            T("golden", f"'on' golden written: {os.path.basename(path)}",
              True, f"{len(g_on)} arrays")
        if not os.path.exists(path):
            T("golden", f"'on' golden {os.path.basename(path)} exists", False,
              "write it with --golden-on write")
        else:
            ref_on = np.load(path)
            R_on = {k: ref_on[k] for k in ref_on.files}
            d = AR.diff(g_on, R_on)
            T("golden", f"as1 preset by the selector == its 'on' golden "
              f"{os.path.basename(path)} (with {', '.join(ON_EXTRA)})",
              not d and set(g_on) == set(R_on), f"{len(d)} differ {d[:3]}")
            report["golden_on"] = dict(path=path, sha1=sha1_file(path),
                                       identical=not d, diffs=d[:20])
            check(f"PlaningVesselAS test_switches={on} given explicitly "
                  f"== its 'on' golden",
                  lambda p: PlaningVesselAS(h1, p.sea, dt=p.dt, record=True,
                                            test_switches=on),
                  ref=R_on, extra=ON_EXTRA)
        ref_p = os.path.join(ROOT, P1A_REF[0])
        if tuple(on) == tuple(P1A_CANDIDATE) and os.path.exists(ref_p):
            with np.load(ref_p) as z:
                R_p = {k: z[k] for k in z.files}
            d = AR.diff(g_on, R_p)
            T("golden", "the as1 preset's runs == the P1a candidate "
              f"reference the review checked ({P1A_REF[0]}), array for "
              "array", not d and set(g_on) == set(R_p),
              f"{len(d)} differ {d[:3]}")
        flux_check(T, report, g_on)
    # the switch sets published before (N2: their 'on' tests stay)
    for sw, (f, _) in GOLDEN_SETS.items():
        path = os.path.join(ROOT, f)
        if not os.path.exists(path):
            T("golden", f"golden of {sw} exists", False, f)
            continue
        with np.load(path) as z:
            R_s = {k: z[k] for k in z.files}
        _, g_s = check(f"PlaningVesselAS test_switches={sw} == its golden "
                       f"{os.path.basename(f)}",
                       lambda p, sw=sw: PlaningVesselAS(h1, p.sea, dt=p.dt,
                                                        record=True,
                                                        test_switches=sw),
                       ref=R_s, extra=ON_EXTRA,
                       opts=GOLDEN_OPTS.get(sw, {}))
        heel = max(float(np.degrees(np.abs(np.asarray(v)[:, 3]).max()))
                   for k, v in g_s.items()
                   if k.startswith("calm_") and k.endswith("_s"))
        T("golden", f"... the largest heel of its calm runs <= "
          f"{GOLDEN_HEEL_MAX_DEG} deg", heel <= GOLDEN_HEEL_MAX_DEG,
          f"{heel:.1f} deg")
    check("v0 PlaningVesselGZ (again, after the as plants) == golden_v0")


def flux_check(T, report, g_on):
    """The S1 flux on the as1 golden runs, re-evaluated at every recorded
    state (the derivative at the start of each step, as the plant measures
    it): equal to v0's flux at every station whose taper strip is not
    chine-wetted (bit for bit), never negative (entry only: the as-built
    gate let it go negative where a chine-wetted section rose out), and the
    recorded last_slam_force_full is its sum; last_slam_force_full ==
    last_slam_force at every step with no chine-wetted taper strip."""
    from hydro import hulls
    from learn.repro import task
    from sim import physics
    from sim.test_vessel import Monochromatic
    m = task.Mission("high", 0, 0, t_end=1.0, track=task.LEGS[0])
    base = physics.gz_twin(hulls.get(AS_KEY).plant(m.plant.sea,
                                                   dt=m.plant.dt))
    plants = dict(task=base, calm=base.with_sea(Monochromatic(1.0, 0.0)))
    n_st = n_cap_st = n_steps_cap = 0
    bad_eq = bad_neg = bad_sum = bad_slam = 0
    min_flux = np.inf
    for k in sorted(g_on):
        if not k.endswith("_s"):
            continue
        sea_name = k.split("_")[0]
        p = plants[sea_name]
        S, M = g_on[k], g_on[k[:-2] + "_m"]
        t = 0.0                    # accumulated as golden_runs does
        for i in range(len(M)):
            p._taper_cap = None
            _, aux = p.deriv(S[i], t)
            t += p.dt
            cap = p._taper_cap
            free = np.ones(p.x.size, bool)
            free[cap] = False
            mf, mo = aux["mom_full"], aux["mom"]
            n_st += 1
            n_cap_st += int(cap.size)
            n_steps_cap += int(cap.size > 0)
            bad_eq += int(not np.array_equal(mf[free], mo[free]))
            bad_neg += int((mf < 0.0).any())
            min_flux = min(min_flux, float(mf.min()))
            bad_sum += int(float(mf.sum() * p.dx) != M[i, 7])
            if cap.size == 0:
                bad_slam += int(M[i, 7] != M[i, 3])
    report["golden_on"]["flux_check"] = dict(
        states=n_st, states_with_capped_taper=n_steps_cap,
        capped_taper_strips=n_cap_st, min_flux=min_flux)
    T("golden", "S1 flux == v0's at every station whose taper strip is not "
      "chine-wetted, at every recorded state", bad_eq == 0,
      f"{bad_eq} of {n_st} states differ; {n_steps_cap} states have a "
      f"chine-wetted taper strip ({n_cap_st} strip-states)")
    T("golden", "S1 flux never negative (entry only)", bad_neg == 0,
      f"{bad_neg} states with a negative station flux; min {min_flux:.3e}")
    T("golden", "recorded last_slam_force_full == the flux at the start of "
      "each step; == last_slam_force where no taper strip is chine-wetted",
      bad_sum == 0 and bad_slam == 0,
      f"sum mismatches {bad_sum}, slam mismatches {bad_slam}")


# ------------------------------------------------------------------ lofi
def tier_lofi(T, report):
    from hydro import hulls
    from sim import config, lofi
    from sim.planing_anyspeed import PlaningVesselAS, PlaningVesselASLin
    from sim.planing_vessel_gz import PlaningVesselGZ
    calm = _calm()
    fv = lofi._fingerprint(PlaningVesselGZ(hulls.get("scarab195"), calm,
                                           dt=0.02))
    fa = lofi._fingerprint(PlaningVesselAS(hulls.get(AS_KEY), calm,
                                           dt=0.02, test_switches="v0"))
    fl = lofi._fingerprint(PlaningVesselASLin(hulls.get(AS_KEY), calm,
                                              dt=0.02, test_switches="v0"))
    p_as1 = config.calm_plant(AS_KEY)[0]
    f1 = lofi._fingerprint(p_as1)
    fp = lofi._fingerprint(config.calm_plant("scarab195")[0])
    stored = json.load(open(AR.LOFI))["fingerprint"]
    report["lofi"] = dict(gz_v0=fv, as_off=fa, as_lin_off=fl, as1=f1, v0=fp,
                          stored=stored)
    T("lofi", "fingerprint: v0 PlaningVessel == stored", fp == stored,
      f"{fp!r} vs {stored!r}")
    T("lofi", "fingerprint: PlaningVesselASLin all off == stored, exactly "
      "(what an as1 reduced model is identified on)", fl == stored,
      f"{fl!r} vs {stored!r}")
    T("lofi", "fingerprint: PlaningVesselAS all off == PlaningVesselGZ v0",
      fa == fv, f"{fa!r} vs {fv!r}")
    T("lofi", "config.calm_plant('scarab195@as1') is PlaningVesselASLin on "
      "the as1 preset; its fingerprint differs from v0's (a switch is on)",
      type(p_as1) is PlaningVesselASLin and f1 != stored,
      f"{type(p_as1).__name__}: {f1!r}")


# -------------------------------------------------------------- missions
def tier_missions(T, report, pairs):
    from hydro import hulls
    import sim.planing_vessel_gz as GZ
    from sim.planing_anyspeed import PlaningVesselAS
    R = pickle.load(open(AR.META7, "rb"))
    FE = AR.setup(R["meta"])
    timer = AR.StepTimer()
    timer.wrap(GZ.PlaningVesselGZ)      # PlaningVesselAS steps through it
    h1 = hulls.get(AS_KEY)
    built = []

    def as_from_plant(plant, **kw):
        # sim/planing_vessel_gz.from_plant, building the all-off as plant
        # for the as1 hull (final_eval imports from_plant at call time)
        opts = dict(plant._init_kw)
        opts.update(kw)
        opts.pop("wind", None)
        if plant.captive_u is not None:
            opts.setdefault("captive_u", plant.captive_u)
        p = PlaningVesselAS(h1, plant.sea, test_switches="v0", **opts)
        built.append(p)
        return p
    sub = dict(missions=[])
    orig = GZ.from_plant
    GZ.from_plant = as_from_plant
    try:
        for row, key in pairs:
            n0 = len(built)
            AR.check_missions(FE, [row], [key], R, timer, sub)
            rec = sub["missions"][-1]
            mine = built[n0:]
            ok_cls = bool(mine) and all(type(p) is PlaningVesselAS
                                        and p.switches == () for p in mine)
            T("missions", f"{row} {key}: PlaningVesselAS all off == meta7",
              rec["identical"] and ok_cls,
              f"{rec['n_diff']} differ of {rec['n_compared']} fields, kn "
              f"{rec['kn']!r} vs {rec['kn_ref']!r}, as plants built "
              f"{len(mine)}, wall {rec['wall_s']:.0f} s")
    finally:
        GZ.from_plant = orig
    report["missions"] = sub["missions"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help=f"every tier, missions {MISSIONS}")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--golden", action="store_true")
    ap.add_argument("--lofi", action="store_true")
    ap.add_argument("--missions", default="",
                    help="row:key pairs, e.g. " + MISSIONS)
    ap.add_argument("--golden-on", choices=("check", "write"),
                    default="check", help="the as1 preset's own golden "
                    "(golden_as_<switches>.npz): check it, or write it once")
    ap.add_argument("--report", default="regress_as.json")
    a = ap.parse_args()
    if a.all:
        a.freeze = a.select = a.golden = a.lofi = True
        a.missions = a.missions or MISSIONS
    warnings.filterwarnings("ignore")
    from sim import physics
    if os.environ.get(physics.ENV):
        raise SystemExit(f"{physics.ENV} is set in this shell "
                         f"({os.environ[physics.ENV]!r}); the checks need it "
                         f"unset")
    T, report = Tally(), {}
    lofi0 = sha1_file(AR.LOFI)
    t0 = time.time()
    if a.freeze:
        tier_freeze(T, report, a.golden_on)
    if a.select:
        tier_select(T, report)
    if a.golden:
        tier_golden(T, report, a.golden_on)
    if a.lofi:
        tier_lofi(T, report)
    if a.missions:
        pairs = [tuple(p.split(":")) for p in a.missions.split(",") if p]
        tier_missions(T, report, pairs)
    T("always", "studies/_cache/lofi_scarab195.json unchanged (before == "
      "after == frozen)", sha1_file(AR.LOFI) == lofi0 == FROZEN[
          "studies/_cache/lofi_scarab195.json"], lofi0)
    T("always", "no lofi_scarab195@as1.json was created",
      not os.path.exists(os.path.join(ROOT, "studies", "_cache",
                                      "lofi_scarab195@as1.json")))
    n_bad = sum(not r["ok"] for r in T.rows)
    report.update(checks=T.rows, n_checks=len(T.rows), n_failed=n_bad,
                  seconds=time.time() - t0)
    with open(os.path.join(AR._out(), a.report), "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(f"\n{len(T.rows)} checks, {n_bad} failed, "
          f"{time.time() - t0:.0f} s -> {a.report}", flush=True)
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
