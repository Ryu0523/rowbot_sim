# usv-rl-mpc

Time-domain seakeeping simulator and predictive controller for a 10 m unmanned
surface vessel in Sea State 4–5, with a real-time WebGL viewer.

The fluid is solved **once, offline** with a boundary-element method; the result
is converted to a state-space system so the online plant is a 110-state ODE that
runs at **29× real time** while still carrying genuine hydrodynamic memory
(2.2–2.9 s). Everything downstream — controller, studies, viewer — runs on that.

```
hull mesh ──▶ BEM (150 ω) ──▶ memory kernel ──▶ state space ──▶ 110-state plant ──▶ MPC
   1056 panels   hours, once      2.2–2.9 s        ERA          29× real time      192 rollouts
```

---

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
```

Windows PowerShell uses `.venv\Scripts\Activate.ps1`; Linux/macOS use
`.venv/bin/activate`. Python 3.12+ (developed on 3.14.7).

`hydro_wigley_10m.npz` and `freesurface_wigley.npz` are committed, so **nothing
below needs Capytaine or a BEM re-run** unless you change the hull.

---

## Run the real-time viewer

Three commands. The middle one is the only slow part.

```bash
python -m studies.export_viewer
```

Simulates 4 runs × 140 s, bakes the Kelvin wake, packs the disturbance table,
and writes `viewer_data.json` (~1.3 MB). **~2 minutes.**

```bash
python -m viewer.build
```

Inlines the JSON into `viewer/seaway.html` → `viewer/seaway_full.html`.
Instant.

```bash
python -m http.server 8731 --directory viewer
```

Then open <http://127.0.0.1:8731/seaway_full.html>.

`seaway_full.html` is fully self-contained — no fetches, no assets — so you can
also just double-click it, or email it to someone. The local server is only
needed if your browser is strict about `file://`.

### Drive it yourself

The viewer has a **Take the helm** button. `W`/`S` throttle, `A`/`D` rudder,
space centres the rudder, `R` restarts. The panel shows speed, heading, angle
relative to the waves, bow acceleration, velocity made good, and a ride cost
(bow acceleration squared per metre of progress). Head seas is the reference —
try 45 degrees off and watch the cost.

Out of the box this integrates the **10-state reduced model**, the one the MPC
plans with, because a browser cannot run the 110-state solve. For the real
thing:

```bash
python -m viewer.helm
```

Then open <http://127.0.0.1:8770/seaway_full.html>. The badge changes to **LIVE
PLANT · 110 states**. The plant now runs in Python at wall-clock pace (measured
1.001x real time) and the browser is reduced to rendering and input.

That matters for more than fidelity: it is **exactly the environment
`sim/rl_env.py` trains on**, so a human lap and a learned policy are directly
comparable. It also returns three things the reduced model structurally cannot:
roll, slam force, and the speed you lose to waves.

The page falls back to the reduced model automatically when no server answers,
which is why the published artifact still works standalone.

### What you are looking at

The browser is **not playing a recording of the water.** It re-evaluates the same
64-component wave spectrum the physics used, in a vertex shader, 60 times a
second. Scrub the timeline and the surface is recomputed at that instant, so the
sea and the hull pose are always consistent. Only the vessel *trajectory* is
recorded, because a 110-state nonlinear solve is not something a browser can
redo.

| Control | Effect |
|---|---|
| **Wave realisation** `#0–#3` | Same spectrum, different random phases — a different sea, and a different run |
| **Camera** | Chase / Beam / Bird. Drag to orbit, scroll to zoom |
| **Preview 8 s** | Highlights the water the controller can see ahead |
| **Wireframe** | Hull mesh instead of solid |
| **Kelvin wake** `off / ×1 / ×3` | The steady bow wave and V-wake. At ×1 it is honest but subtle against 2.2 m waves; ×3 makes the divergent crests obvious |
| **Diffraction & radiation** `off / ×1 / ×25` | The unsteady disturbance. At ×1 it is invisible — peak 0.042 m — which is the point; ×25 shows the ring pattern |
| **Timeline** | Orange ticks mark slam events. Space bar pauses |

Requires WebGL with at least one vertex texture unit (any GPU since ~2012). If
something fails it says so on screen rather than showing a black page.

### Offline figures instead

```bash
python -m studies.visualise
```

Matplotlib 3-D scenes plus an incident / disturbance / total decomposition →
`fig12_scene.png`, `fig13_decomposition.png`. Useful for documents; no browser.

---

## Verification gates

Every one of these prints `PASSED`/`FAILED` and exits noisily if the physics
drifts. Run them after any change to `hydro/` or `sim/`. Times are measured,
sequentially, on one laptop.

```bash
python -m hydro.verify_sphere        # 11 s   BEM vs analytic sphere + reciprocity
python -m hydro.verify_retardation   #  4 s   memory kernel + state-space fit (28 checks)
python -m hydro.inertia              # 18 s   rigid-body inertia vs EXACT hemisphere,
                                     #        box and closed-form Wigley (DEFECTS A30)
python -m hydro.symmetry             #  1 s   what the mesh gets wrong that geometry says is 0
python -m hydro.hull                 #  5 s   hull integrity: displacement, gyradii, rudder
python -m hydro.hulls                #  1 s   the registered vessels, what each still lacks
python -m hydro.manoeuvring          # <1 s   Clarke (1983) against the MSS reference code
python -m hydro.cad_import           # minutes CAD import: exact box checks, then KVLCC2's
                                     #        IGES against its published hydrostatics
python -m hydro.ikeda                # ~1 s   simplified Ikeda: 208 coefficients vs the ITTC
                                     #        tables, exact identities, Kawahara Fig. 1
python -m sim.forces                 # <1 s   Coriolis is skew-symmetric and reduces to Euler
python -m sim.verify_rao             # 22 s   time domain vs exact frequency domain, both in
                                     #        Capytaine's e^-iwt convention (DEFECTS F3)
python -m sim.test_sections          #  1 s   exact panel integration vs closed form
python -m sim.test_vessel            # 61 s   nonlinear plant, 7 checks
python -m sim.test_manoeuvre         # 14 s   turning circle, drift, course stability
python -m studies.audit_physics      #  5 s   28 independent checks of the force model
python -m hydro.kelvin               # 97 s   steady wake geometry, 6 checks
python -m studies.model_horizon      #  4 m   can the MPC's internal model carry preview
python -m studies.hull_acceptance NAME  # min  ANY vessel: 8 checks through the whole
                                     #        chain, plant to RL environment (DEFECTS G)
```

**What these gates are, and are not.** Every one of them checks the code
against mathematics or against itself. Comparisons with physical experiments
are separate -- the `studies/exp_*` scripts below, each on the hull its data
belongs to. The gates that caught real bugs did it by bringing in an EXTERNAL
truth (`hydro.inertia` found the mass matrix 17-23% low after 27 internal
audits had passed), and one gate was wrong in a way that let it pass: M3
compared the time domain with a frequency-domain reference in the other time
convention, and chose its forcing sign to match (DEFECTS F3; invisible on the
fore-aft symmetric Wigley, 29% in heave on KCS). `test_manoeuvre` in particular
is a plausibility bound, not a validation: it accepts a 42% change in the very
turning radius it gates (DEFECTS E2, E3). See `studies.external_methods` for
which published methods have been applied and which datasets are still needed.

Reference numbers live in `PLAN.md` (they predate the physics additions of
DEFECTS section E -- turning radius, roll period and bow acceleration have all
moved; `DEFECTS.md` E10 lists what is stale). Every historical failure and its
cause is in `DEFECTS.md`.

---

## Studies

All print a table; most write a `.json`. Times are approximate.

```bash
# what is missing, and how well each claim is known
python -m studies.missing_terms           #  ~6 m  missing physics, by evidence tier
python -m studies.external_methods        #  <1 s  published methods applied to this hull
python -m studies.manoeuvring_identify    #  ~5 m  Yv'/Nr' from published ranges
python -m studies.damping_check           #   1 s  placeholders as fractions of critical
python -m studies.rao_relevance           #   1 s  does tank data cover our regime

# against measurements, each on the hull the data belongs to (DEFECTS F8)
python -m studies.exp_kvlcc2_rolldecay        #  ~1 m  SSPA KVLCC2 roll decays, fitted
python -m studies.exp_kvlcc2_roll_prediction  #  ~2 m  ...predicted from lines and loading
python -m studies.exp_kvlcc2_plant_decay      #  ~2 m  ...and the time-domain plant against them
python -m studies.exp_kcs_t2015_seakeeping    #  ~8 m  KCS heave/pitch/roll vs T2015 (BEM cached)
python -m studies.exp_kcs_stf_speed           #  ~2 m  KCS: forward-speed (STF) terms vs T2015
python -m studies.exp_slam_wedge              #  <1 s  Wagner vs a 30 deg wedge drop test
python -m studies.exp_propeller               #  <1 s  B-series vs KVLCC2 open-water data
python -m studies.exp_hull_robustness         #   long the plant on five hulls (F9; superseded
                                              #        by studies.hull_acceptance)

# the preview / control questions
python -m studies.preview_is_used         #  20 s  proof the controller reads the preview
python -m studies.slam_effect             #  37 s  does the impact load change anything
python -m studies.placeholder_sensitivity #   6 m  which uncalibrated parameter actually matters
python -m studies.sea_state_value         # ~15 m  is knowing Hs/Tp worth anything
python -m studies.preview_metrics         # ~25 m  preview vs 9 criteria, 10 seeds
python -m studies.preview_timescale       # ~25 m  wave-by-wave vs wave-group preview
python -m studies.preview_sweep           # ~30 m  the M7 sweep, 12 seeds
python -m learn.tune                      #  1-2 h CMA-ES weight tuning
```

Headline results are summarised in `PLAN.md`. The short version: preview
improves **0 of 27** metric × strategy combinations beyond noise, because the
surge time constant is several times longer than a quarter of the encounter
period -- the constraint is actuator bandwidth, not sensing range. (With the
propeller advance-ratio curve now in, that margin shrinks from 3.4x to about
2.2x; the conclusion stands, its size does not.)

---

## Importing your own vessel

Nothing in the chain is specific to the Wigley test hull any more: not the
plant, and (DEFECTS G) not the reduced model, the MPC, the environments or the
training scripts either. A vessel is a `hydro.hull.Hull`, registered by name
in `hydro/hulls.py`, and every script takes `--hull NAME`:

1. **Fill in the template.** `our_boat()` in `hydro/hulls.py` lists every
   field and where its value comes from (CAD file, loading condition, radii of
   gyration, appendages, a roll-decay test, a turning trial). A field left
   `None` runs on a placeholder and is reported as one.
2. **Check it.** `python -m hydro.hulls our_boat` builds the mesh and runs
   `Hull.check()`: displacement against the lines, gyradii given or taken
   from geometry, rudder and propeller depths, and the placeholder list.
3. **Run the acceptance battery.** `python -m studies.hull_acceptance our_boat`
   -- calm float, the plant against the frequency domain in the linear limit,
   heading invariance, design speed reachable, turning, the mission sea, and
   the whole control chain. The first run computes the BEM database (tens of
   minutes for a CAD hull). It shows the simulation is consistent ON your
   hull; only trials show it matches your boat.
4. **Use it.** `python -m learn.tune --hull our_boat`,
   `python -m learn.ppo_train --hull our_boat`,
   `python -m viewer.helm --hull our_boat`. In code:
   `hull, db = sim.config.load("our_boat")`, then
   `plant = sim.config.plant_for(db, sea, hull)`.

What follows the vessel, exactly as the 10 m USV's constant for the USV and
Froude-scaled (lengths x lam, times x sqrt(lam), lam = L / 10 m) otherwise:
time step and control step, design speed, thrust limit (from the Hull, else
the USV's 2.1x margin over its resistance at THIS vessel's design speed),
rudder limits, MPC stations (stern to bow), autopilot bandwidth, slamming
and deck-wetness thresholds (the bow's own keel depth and freeboard), the
cross-track unit (hull lengths), identification inputs, episode lengths and
preview horizon, and the RL observation. Sea states stay absolute: a sea state
is the mission, not the vessel.

```python
from hydro.hull import Hull
h = Hull(name="ourboat", L=10.0, B=2.6, T=0.7,
         mesh_kind="file", mesh_path="ourboat.stl",   # stated: never defaulted
         displacement=8500.0, z_cog=0.15,              # kg; m above the waterline
         k_roll=0.36, k_pitch=0.25, k_yaw=0.26,        # radii of gyration, x B / x L
         x_rud=-4.8, z_rud=-0.40, rudder_area=0.20, rudder_span=0.50,
         x_prop=-4.5, z_prop=-0.50, d_prop=0.45)
h.check()               # raises if the displacement and the lines disagree, etc.
db = h.database()       # runs the BEM once (slow), caches it by parameter hash
plant = h.plant(sea, db=db)
```

What `check()` enforces, because each one has already bitten this project:
the stated displacement must match the mesh volume; each gyradius is reported
as GIVEN or FROM GEOMETRY (the geometric value is a hull-shaped block of water,
2.4x too little roll inertia for a real boat); the rudder blade must be deep
enough for the Rudder's own ventilation criterion. `database()` computes 13
wave headings (the vessel's heading enters the excitation, DEFECTS F4),
replaces Capytaine's inertia with the corrected one (DEFECTS A30), imposes
port/starboard symmetry unless `symmetric=False`, and stores L, B, T with it.

From CAD surfaces (IGES/STEP), with units, axes and pieces STATED rather than
guessed (`hydro/cad_import.py`; KVLCC2's IGES -- x aft, z down, flat bottom
twice -- is the worked example, DEFECTS F1-F2):

```python
h = Hull(name="ourboat", L=10.0, B=2.6, T=0.7, mesh_kind="cad",
         mesh_path="ourboat.igs", displacement=8500.0, z_cog=0.15,
         cad=dict(scale=1e-3,               # file in mm
                  forward="+x", up="+z",    # which file axes point to bow and up
                  half=True,                # the file models one side
                  size_max=150.0),          # panel size, in FILE units
         k_roll=0.36, k_pitch=0.25, k_yaw=0.26)
```

The waterline is found from the displacement, not taken as z = 0 of the file;
the origin is moved to the centre of buoyancy. The import checks closure with
three volume forms, drops patches lying on other patches, orients patches along
their shared edges, and lists the connected pieces (a rudder or boss cap can be
dropped with `min_component=`).

The plant Froude-scales the 10 m USV's placeholders -- viscous damping,
thruster and rudder dynamics, wind lever -- to the hull's length and says so
once. Where a published method covers the hull it is used instead: Clarke et
al. (1983) for Y_v', N_r' (`hydro/manoeuvring.py`), the simplified Ikeda
method for roll damping when the hull is inside its range
(`roll_damping="auto"`), the Wageningen B-series when `prop_series` is given.

What a real vessel still needs from the real vessel, because no simulation
supplies it, in order of value: a free-roll decay test (enter it as
`roll_damping=dict(zeta1=, k2=, T=)`; the plant subtracts the BEM's wave part
itself), the radii of gyration, the hull manoeuvring
derivatives (turning trial or PMM), the propeller open-water curve and the
thrust limit, the remaining viscous damping, and wind areas.
`python -m hydro.hulls NAME` prints what is still standing in.

---

## Rebuilding the hydrodynamics

Only needed if you change the hull. Requires `capytaine`. For a new vessel,
use `Hull.database()` above instead -- it does all of this.

```bash
python -m hydro.run_wigley          # hours  -> hydro_wigley_10m.npz
python -m hydro.freesurface         # ~1 h   -> freesurface_wigley.npz
```

Capytaine's inertia matrix is wrong for this hull by 17-23% (DEFECTS A30: it
averages three divergence-theorem forms and one is ill-conditioned on a
wall-sided hull). `run_wigley` now replaces it with the verified closed form
before anything is written, so a rebuild cannot silently put the bug back.
`python -m hydro.fix_inertia_db --apply` does the same to a database built
before that change, and keeps a backup.

---

## Code structure

```
hydro/                  offline: geometry, fluid solution, frequency → time
  geometry.py           Wigley hull mesh; analytic volume/waterplane/restoring
                        used as ground truth for the mesh
  bem.py                Capytaine driver + HydroDB storage (.npz).
                        Splits excitation into Froude–Krylov and diffraction
  hull.py               Hull: one description of a vessel -- mesh, mass,
                        appendages, cached BEM. The import path for real boats
  inertia.py            rigid-body inertia that does not trust Capytaine's,
                        checked against three exact solutions; gyradius knob
  symmetry.py           zeroes the vertical/lateral coupling a symmetric hull
                        cannot have, and reports how much mesh noise it removed
  fix_inertia_db.py     rewrites a stored database's inertia (A30)
  retardation.py        Ogilvie transform → memory kernel K(t); ERA state-space
                        realisation; mode pruning
  run_wigley.py         builds hydro_wigley_10m.npz (150 frequencies)
  freesurface.py        disturbance transfer functions on a 61×41 field grid
  kelvin.py             steady Kelvin wave system (Michell thin-ship theory),
                        and the wake texture the viewer samples
  verify_*.py           M1/M2 gates

sim/                    the plant
  wavefield.py          JONSWAP spectrum, cos^2s spreading, SeaState
  sections.py           41 stations. Closed-form Wigley + exact panel
                        integration for arbitrary meshes. SlamLoad table
  cummins.py            linear 6-DOF time-domain core (radiation memory)
  vessel.py             NonlinearVessel — blended nonlinear Froude–Krylov,
                        added resistance, water-entry load, viscous damping,
                        Coriolis from the yaw rotation of rigid-body and
                        added-mass momentum (incl. the Munk moment), wind.
                        Takes a Hull, or the Wigley defaults
  forces.py             Coriolis-centripetal matrix (skew-symmetric by
                        construction) and apparent-wind loads. Which terms a
                        model may use depends on its FRAME -- see DEFECTS E11
  actuators.py          propulsion: lag, rate limit, delay, ventilation,
                        advance-ratio curve K_T(J). Rudder: inflow angle from
                        drift and yaw rate, ventilation, lift slope computed
                        from the blade (Whicker & Fehlner)
  env.py                Episode, autopilot, operator objective, Gym wrapper
  seakeeping.py         the criteria: relative motion, Ochi rates, envelope,
                        fatigue. Read this before adding a metric
  rl_env.py             direct-control RL environment: action is (thrust,
                        rudder), 39-dim observation incl. wave preview, dense
                        bounded reward. No gymnasium required
  test_*.py, verify_*.py  M3/M4 gates

control/
  reduced.py            10-state model identified FROM the plant (sway
                        included), exact expm transitions, for MPC rollouts
  mpc.py                MPPI sampling controller + PreviewProvider

learn/
  tune.py               CMA-ES over 7 bounded weights
  ppo_train.py          PPO on the full plant, disjoint train/eval seas
  ppo_bc.py, bc_only.py behaviour cloning of the MPC, then PPO fine-tuning.
                        Demonstrations are cached and fingerprinted by what
                        the plant DOES, so a physics change invalidates them

studies/                each answers one question and prints/writes its answer
viewer/
  seaway.html           the WebGL viewer (three.js r128, single file).
                        Drive mode integrates the reduced model locally, or
                        defers to viewer/helm.py when it is running
  build.py              inlines viewer_data.json → seaway_full.html
  helm.py               serves the page AND runs the real plant in real time,
                        so a person can steer the same environment the RL
                        policy trains on. Standard library only

step0_preview_spec.py   JONSWAP spectrum and directional spreading.
                        Still imported by sim/wavefield.py, so it must stay at
                        the repository ROOT — which is also why every command
                        above is `python -m ...` run from the root
```

`vessel.py`, `phase0_sweep.py`, `verify_model.py` and `explain_cummins.py` at the
root are Phase-0 artifacts from the toy model that preceded this one. Nothing
imports them; they are kept because `PLAN.md` cites their numbers.

### Data files

| File | Produced by | Consumed by |
|---|---|---|
| `hydro_wigley_10m.npz` | `hydro.run_wigley` (inertia corrected at build) | everything |
| `hydro_wigley_10m.npz.pre_inertia_fix` | `hydro.fix_inertia_db` | backup only |
| `hydro_<name>_<hash>.npz` | `Hull.database()` | a `Hull`'s vessel |
| `freesurface_wigley.npz` | `hydro.freesurface` | `studies.visualise`, `studies.export_viewer` |
| `bc_demos.npz` | `learn.bc_only` | `learn.ppo_bc`; re-collected automatically when stale |
| `viewer_data.json` | `studies.export_viewer` | `viewer.build` |
| `viewer/seaway_full.html` | `viewer.build` | your browser |
| `tuned_weights.json` | `learn.tune` | optional MPC weights |

The PPO / BC checkpoints (`ppo_*.zip`, `bc_clone.zip` and their
`_vecnorm.pkl`) were trained before the section-E physics and need retraining.

### Documents

- **`PLAN.md`** — architecture, milestones, acceptance gates, results
- **`METHODS.md`** — per-module: which theory computes which quantity, and how it was verified
- **`DEFECTS.md`** — A1–A30 fixed defects with symptom and cause; D, what the model
  leaves out, sorted by how well each claim is known; E, the physics added since
  and what it overturned. Read A4, A22, A30 and E4 before trusting a new
  measurement: a large share of the bugs here were in the *ruler*, not the model

---

## Notes

- **Run everything from the repository root** with `python -m package.module`.
  `sim/wavefield.py` imports `step0_preview_spec` from the root, so `cd`-ing into
  a subdirectory breaks the import.
- `gymnasium` is optional. `sim/env.py` guards the `USVEnv` wrapper behind
  `HAVE_GYM`; every gate and study runs without it.
- Studies are CPU-bound, single-threaded, and independent per seed — run several
  in separate shells if you have cores to spare.
- Figures are written to the repository root as `figN_*.png`.
