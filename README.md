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
drifts. Run them after any change to `hydro/` or `sim/`.

```bash
python -m hydro.verify_sphere        # 11 s   BEM vs analytic sphere + reciprocity
python -m hydro.verify_retardation   #  4 s   memory kernel + state-space fit (28 checks)
python -m sim.verify_rao             # 22 s   time domain vs exact frequency domain
python -m sim.test_sections          #  1 s   exact panel integration vs closed form
python -m sim.test_vessel            # 61 s   nonlinear plant, 7 checks
python -m hydro.kelvin               # 97 s   steady wake geometry, 6 checks
```

All six currently pass. Reference numbers live in `PLAN.md`; every historical
failure and its cause is in `DEFECTS.md`.

---

## Studies

All write a `.json` and print a table with a standard error on every number.
Times are wall-clock on one laptop core and approximate.

```bash
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
surge time constant (6.3 s) is 3.4× longer than the encounter period (7.5 s) —
the constraint is actuator bandwidth, not sensing range.

---

## Rebuilding the hydrodynamics

Only needed if you change the hull. Requires `capytaine`.

```bash
python -m hydro.run_wigley       # hours  -> hydro_wigley_10m.npz
python -m hydro.freesurface      # ~1 h   -> freesurface_wigley.npz
```

---

## Code structure

```
hydro/                  offline: geometry, fluid solution, frequency → time
  geometry.py           Wigley hull mesh; analytic volume/waterplane/restoring
                        used as ground truth for the mesh
  bem.py                Capytaine driver + HydroDB storage (.npz).
                        Splits excitation into Froude–Krylov and diffraction
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
                        added resistance, water-entry load, viscous damping
  actuators.py          propulsion (lag, rate limit, delay, ventilation), rudder
  env.py                Episode, autopilot, operator objective, Gym wrapper
  seakeeping.py         the criteria: relative motion, Ochi rates, envelope,
                        fatigue. Read this before adding a metric
  test_*.py, verify_*.py  M3/M4 gates

control/
  reduced.py            9-state model identified FROM the plant, for MPC rollouts
  mpc.py                MPPI sampling controller + PreviewProvider

learn/
  tune.py               CMA-ES over 7 bounded weights

studies/                each answers one question and writes a .json
viewer/
  seaway.html           the WebGL viewer (three.js r128, single file)
  build.py              inlines viewer_data.json → seaway_full.html

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
| `hydro_wigley_10m.npz` | `hydro.run_wigley` | everything |
| `freesurface_wigley.npz` | `hydro.freesurface` | `studies.visualise`, `studies.export_viewer` |
| `viewer_data.json` | `studies.export_viewer` | `viewer.build` |
| `viewer/seaway_full.html` | `viewer.build` | your browser |
| `tuned_weights.json` | `learn.tune` | optional MPC weights |

### Documents

- **`PLAN.md`** — architecture, milestones, acceptance gates, results
- **`METHODS.md`** — per-module: which theory computes which quantity, and how it was verified
- **`DEFECTS.md`** — all 23 defects found, with symptom and cause. Read A4, A18, A19 and A22 before trusting any new measurement; roughly a third of the bugs in this project were in the *ruler*, not the model

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
