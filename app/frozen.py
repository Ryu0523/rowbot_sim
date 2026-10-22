"""
The frozen final simulator configuration (learn/meta/SIM_FINAL_2026-10-09.md
sections 4.1 and 6), as constants the product layer defaults to.

    MODEL    'scarab195@as1' -- the any-speed plant, preset as1
             (sim/planing_anyspeed.PRESETS['as1']); built through
             hydro.hulls.get(MODEL).plant(sea, dt=DT) and, where capsize is
             modelled (the default), its large-angle twin through
             sim.physics.gz_twin (never the frozen sim/planing_vessel_gz
             helpers, which build v0 whatever the hull says)
    PLANT    'gz' (PlaningVesselAS, the large-angle roll twin)
    DT       0.01 s plant step; DT_CTRL 0.25 s control step = 25 sub-steps
             (mean-type statistics after the 20 Hz filter are within 2 % of
             the finest step from 0.01 s on; A1/10 and extremes need
             DT_EXTREMES = 0.005 s -- set usv.dt in the scenario)
    FILTER   2nd-order Butterworth, 20 Hz full scale, causal online, zero
             phase offline (app/accel.py)
    OUTPUTS  CG and bow (0.1 L aft of the stem) vertical acceleration in g,
             filtered, in the observation (key 'accel') and the record
             (plant['acc_obs']); raw m/s^2 kept in the record (a_cg, a_bow)
    SEEDS    evaluation seeds learn/repro/task.EVAL_SEEDS (sea seed = episode
             seed); training draws never return one of them
    STAMP    every record carries sim.physics.stamp(plant) (header
             'physics'), the nominal key's stamp beside it
             ('physics_nominal') and the randomisation draw ('randomization')

A v0 hull (a plain name) keeps its own steps (Hull.scales(): 0.02 s, 12
sub-steps), so the frozen v0 results reproduce bit for bit; the frozen
configuration's steps apply to the as versions.
"""
import math

MODEL = "scarab195@as1"
VERSION = "as1"
PLANT = "gz"
DT = 0.01
DT_CTRL = 0.25
DT_EXTREMES = 0.005
FILTER_HZ = 20.0
FILTER_ORDER = 2
BOW_GAUGE = 0.1
DROP = 0.1
SHORT_WAVE_RATIO = 1.5        # 'short' components: wavelength < 1.5 L
# the evaluation sea (SIM_FINAL section 6; studies/final_eval.make_mission)
SEA = dict(hs=1.0, tp=5.0, gamma=3.3, n_freq=24, n_dir=5)


def default_steps(hull):
    """(dt, dt_ctrl) for a hull: the frozen steps for an as version, the
    hull's own scales for v0."""
    from sim import physics
    sc = hull.scales()
    if physics.version(hull) == physics.DEFAULT:
        return float(sc["dt"]), float(sc["dt_ctrl"])
    return DT, DT_CTRL


def is_frozen(hull_key, plant_kind, dt, dt_ctrl):
    """True when (hull key, plant kind, steps) are the frozen final
    configuration's."""
    return (hull_key == MODEL and plant_kind == PLANT
            and abs(float(dt) - DT) < 1e-12
            and abs(float(dt_ctrl) - DT_CTRL) < 1e-12)


def scenario(hs=SEA["hs"], tp=SEA["tp"], theta0=math.pi, heading=0.0,
             duration=60.0, seed=0, speed=None, **usv):
    """A straight-track scenario on the frozen configuration (head seas of
    the evaluation sea state by default; `speed` m/s initial, None = 0.8
    u_design)."""
    from .scenario import Scenario
    agent = dict(model=MODEL, plant=PLANT, heading_rad=heading)
    if speed is not None:
        agent["speed"] = float(speed)
    doc = dict(
        name=f"frozen_{MODEL}",
        environment=dict(ocean=dict(wave_height=hs, wave_period=tp,
                                    theta0=theta0, gamma=SEA["gamma"],
                                    n_freq=SEA["n_freq"],
                                    n_dir=SEA["n_dir"])),
        agents={"usv0": agent},
        usv=dict(duration=duration, seed=seed, **usv))
    return Scenario(doc)
