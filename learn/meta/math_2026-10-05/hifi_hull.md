# High-fidelity plant "gz": complete equations of motion (from the code)

Repository root `C:/Users/Administrator/Documents/usv-rl-mpc`. All citations are `file:line` in that tree.
Every equation below was read from the implementation. Where a docstring disagrees with the code, the code is followed and the disagreement is listed in section 14.
Numbers marked "(probe)" were computed by running the actual classes (scripts `probe_hifi.py`, `probe_hifi2.py` in this folder, each under 10 s).

---------------------------------------------------------------------------------------------------

## 0. Which files define the plant, and which do not

Construction path used by the recorded target data (`studies/onestep_gz.py:136-169` `record()`):

1. `studies/final_eval.py:208-223` `make_mission` builds `learn/repro/task.Mission("high", seed, leg, t_end=T, track=LEGS[leg])`, then replaces its plant by `sim/planing_vessel_gz.from_plant(m.plant, record=True)` (`planing_vessel_gz.py:359-368`) and re-initialises the state with `pl.initial_state(0.8*u_ref)`, `s[5] = phi_track` (`final_eval.py:221-222`).
2. `learn/repro/task.py:230-284` `Mission` builds `sim/env.Episode` (`sim/env.py:54-80`), which builds the sea `sim/wavefield.SeaState(...)` (`env.py:68-69`) and the plant `sim/config.plant_for(db, sea, hull, dt, fidelity="high")` (`config.py:88-110`) -> `hydro/planing.PlaningHull.plant` (`hydro/planing.py:138-141`) -> `sim/planing_vessel.PlaningVessel`.
3. Hull: `hydro/hulls.py:131-156` `scarab_195()` -> `PlaningHull(...)` with the defaults of `hydro/planing.py:99-130`.

Files the task listed that the planing plant does NOT use for its dynamics (checked by the import lists and call sites):

| file | status for this plant |
|---|---|
| `sim/sections.py` (strip theory of the displacement plant) | not imported by `planing_vessel.py`/`planing_vessel_gz.py`; imported only by `sim/vessel.py:38` (displacement plant). |
| `sim/cummins.py` | imported only by `sim/vessel.py:37`. No radiation memory in the planing plant. |
| `sim/seakeeping.py` | not called by the plant (used by `Episode.run` summaries, which `Mission` does not call). |
| `sim/forces.py` | `planing_vessel.py:51` imports `point_load` with `# noqa: F401`; it is never called. The plant uses its own `_load` (`planing_vessel.py:67-71`). |
| `sim/vessel.py` | only the constant `SIGN_PITCH = -1.0` (`vessel.py:46`, imported at `planing_vessel.py:52`). |
| `sim/wavefield.py` | only as the class of the sea object; the plant reads the attributes `a, w, k, th, phi` (`planing_vessel.py:210-219`). `eta_at_depth`, `eta_dot`, `HeadingTransfer`, `interp_transfer` are NOT called. The attributes are created by the base class `step0_preview_spec.WaveField.__init__` (`step0_preview_spec.py:59-92`). |

The dynamics are therefore entirely in `sim/planing_vessel.py:250-369` (`deriv`), modified in one row by `sim/planing_vessel_gz.py:223-239`, integrated by `planing_vessel.py:522-569` (`step`) wrapped by `planing_vessel_gz.py:254-273`.

---------------------------------------------------------------------------------------------------

## 1. Frames and sign conventions

* Earth frame (X, Y, Z): Z upward, still-water surface at Z = 0. Horizontal position kinematics `ds[0] = u cos psi - v sin psi`, `ds[1] = u sin psi + v cos psi` (`planing_vessel.py:363-364`): psi is measured from +X toward +Y.
* Body frame (x, y, z): origin at the centre of gravity G, x forward, z upward (the keel is at body `z_k = -vcg < 0`, `planing_vessel.py:99`), y completing a right-handed set, so +y points to port. (The comments at `planing_vessel_gz.py:125-127,158` call +y "starboard"; that label is inconsistent with a right-handed z-up frame. It does not change any equation.)
* Pitch sign: `SP = SIGN_PITCH = -1` (`vessel.py:46`, `planing_vessel.py:58`). The height of a hull point at body x changes by `SP * x * theta = -x theta`: positive theta = bow DOWN; bow-up trim is negative (`planing_vessel.py:28-29,42-44`).
* Roll sign: positive phi lifts the +y side (`planing_vessel_gz.py:140`, rotation at `planing_vessel_gz.py:146`).
* Moments of a point load: `_load(fx, fy, fz, x, y, z) = [fx, fy, fz, y fz - z fy, z fx - x fz, x fy - y fx]` (`planing_vessel.py:67-71`), i.e. r x F in the body frame. A pitch moment > 0 drives theta > 0 (bow down), consistent with `My = sum SP x f` (`planing_vessel.py:342`).
* Small-angle formulation (exact statement): the code uses p = d(phi)/dt, q = d(theta)/dt, r = d(psi)/dt and w = dz/dt (`planing_vessel.py:365`), with no Euler-angle transformation; horizontal kinematics use psi only; heave w is the EARTH-vertical velocity of G, not a body-axis velocity. All strip forces are taken vertical, the pressure drag is their horizontal projection (`planing_vessel.py:355`), and gravity appears only in the heave row (`planing_vessel.py:341`). Hence u, v act as horizontal velocities along / across the heading.

---------------------------------------------------------------------------------------------------

## 2. Fixed parameters (scarab195, deterministic; no parameter is randomised in the high world)

`Mission(world="high")` passes `plant_params=None`; `config.plant_for` raises if params are given for `fidelity="high"` (`config.py:99-101`). The only random quantity in the plant is the wave phase vector (section 4).

| symbol | code | value | unit / meaning |
|---|---|---|---|
| L | `h.L` | 5.4 | m, keel length transom to stem (`hulls.py:146`) |
| b | `h.b` | 0.83*2.44 = 2.0252 | m, chine beam |
| beta | `h.beta_deg` | 20 deg | deadrise; tan(beta) = 0.36397, cos(beta) = 0.93969 |
| m | `h.mass` | 1400 | kg |
| lcg | `h.lcg` | 0.38*5.74 = 2.1812 | m, G from transom |
| vcg | `h.vcg` | 0.55 | m, G above keel |
| D_h | `h.depth` | 1.1 | m, keel to deck (used only by the GZ curve) |
| k_pitch, k_roll, k_yaw | defaults | 0.25, 0.35, 0.25 | radii of gyration / (L, b, L) (`planing.py:100`) |
| taper | default | 0.35 | bow fraction where chine curves in |
| stem_rise | default | 0.0 | m -> flat keel line |
| C_DA | `h.cda` | 1.7 | m^2, air drag area |
| u_d | `h.u_design` | 25 kn = 12.8611 | m/s |
| N | `h.n_stations` | 48 | strips |
| k_a | default | 1.0 | added-mass factor (`planing.py:102,120`) |
| C_dc | default | cos(beta) = 0.93969 | cross-flow (vertical) drag coefficient (`planing.py:121-122`) |
| a_b,inf | default | 0.5 | buoyancy factor at planing speed |
| C_D,lat | default | 1.0 | lateral cross-flow drag coefficient |
| zeta | default `roll_zeta` | 0.10 | roll damping "ratio" parameter |
| rho | default | 1025 | kg/m^3 |
| rho_air, nu, g | `planing_vessel.py:55-57` | 1.225, 1.19e-6, 9.81 | |
| transom correction | default | False (`planing.py:104`) | so chi_i(u) = 1 for all i (`planing_vessel.py:237-239`, probe) |

Derived constants (code line; value from probe):

* Rigid-body inertia `M = diag(m, m, m, Ix, Iy, Iz)` (`planing_vessel.py:104-109`): Ix = m (0.35 b)^2 = 703.40 kg m^2; Iy = Iz = m (0.25 L)^2 = 2551.5 kg m^2. No products of inertia.
* Rest attitude (Z_rest, Theta_rest) from `_static` (`planing_vessel.py:177-190`): solves `sum rho g A_i(d_i) dx = m g` and `sum SP x_i rho g A_i dx = 0` with `d_i = -(Z + z_k,i + SP x_i Theta)`. Z_rest = 0.23335 m (absolute height of G), Theta_rest = -0.031994 rad (-1.833 deg, bow up). Transom draft T = 0.3846 m (`planing_vessel.py:112-113`).
* GM = KB + BM - vcg from the V sections at rest (`planing_vessel.py:192-202`): `KB = sum (2/3) d0+ A dx / vol`, `BM = sum (2/3) half^3 dx / vol` with `half = min(d0+/tan beta, y_c)`. GM = 1.21583 m.
* Lateral/roll/surge total inertias `Mtot`, fixed at construction (`planing_vessel.py:371-383`), using the immersion at the running attitude at u_d (z = 0.15366 m, theta = -0.009229 rad, probe):
  * M11 = 1.05 m = 1470 kg (`:379`)
  * M22 = m + sum_i m22_i^des dx = 1610.20 kg (`:380`), m22_i^des = (pi/2) rho (d_i^des+)^2
  * M44 = 1.2 Ix = 844.08 kg m^2 (`:381`)
  * M66 = Iz + sum_i x_i^2 m22_i^des dx = 2932.49 kg m^2 (`:382`)
  * Mtot[2,2] = m and Mtot[4,4] = Iy are NOT used by `deriv` (heave/pitch use the instantaneous added mass, section 6).
  * `visc[1]` (`:383`) is computed but never used by `deriv`.
* Jet nozzle point (x_n, 0, z_n) = (-lcg + 0, 0, -vcg + 0.15) = (-2.1812, 0, -0.40) m (`planing_vessel.py:119-120,139`). Intake point (x_in, 0, z_in) = (-1.3812, 0, -0.55) m (`:121-122,141`).
* Roll frequency used in the damping term: omega_n = sqrt(m g GM / Ix) = 4.872 rad/s (`planing_vessel.py:323`). Note Ix, not M44.

### 2.1 Station geometry (`planing_vessel.py:91-102`)

For i = 1..N: xi_i = (i - 1/2) L/N (from transom), dx = L/N = 0.1125 m, body x_i = xi_i - lcg in [-2.12495, 3.16255] m.
s_i = clip((xi_i - (1 - taper) L) / (taper L), 0, 1).

* chine half-beam y_c,i = (b/2) sqrt(max(1 - s_i^2, 0)) + 1e-4 (1.0127 m aft, 0.2453 m at the foremost station)
* keel height (body) z_k,i = -vcg + stem_rise s_i^2 = -0.55 m for all i (stem_rise = 0)
* keel slope k'_i = 2 stem_rise s_i/(taper L) = 0; curvature k''_i = 0 (both zero for scarab195)
* chine height above keel h_c,i = y_c,i tan(beta)

### 2.2 Section functions of the immersion d (`planing_vessel.py:161-175`)

With d+ = max(d, 0):

* A_i(d) = min(d+, h_c,i)^2 / tan(beta) + 2 y_c,i max(d+ - h_c,i, 0)   (immersed area of V bottom + vertical sides)
* c_w(d) = (pi/2) d+ / tan(beta)   (Wagner wetted half-beam);  c_i(d) = min(c_w, y_c,i)
* m_a,i(d) = k_a (pi/2) rho c_i^2        (2-D added mass per unit length)
* m'_a,i(d) = k_a pi rho c_i (pi/(2 tan beta)) if (c_w < y_c,i and d > 0), else 0   (= d m_a/d d before chine wetting)
* girth_i(d) = 2 c_i / cos(beta) + 2 max(d+ - h_c,i, 0)

In `deriv`, m_a and m'_a are multiplied by chi_i(u) (`:276`), which is 1 here.

---------------------------------------------------------------------------------------------------

## 3. State, inputs, hidden states, noise

### 3.1 State vector s (14 components, `planing_vessel.py:41-44,251-254`)

| idx | symbol | meaning | frame / reference |
|---|---|---|---|
| 0 | x0 | X of G | earth |
| 1 | y0 | Y of G | earth |
| 2 | z | height of G above its rest height Z_rest | earth vertical, up > 0 |
| 3 | phi | roll angle | positive lifts +y side |
| 4 | theta | pitch from rest trim; total trim Theta = Theta_rest + theta (`:261`) | positive bow down |
| 5 | psi | heading | from +X toward +Y |
| 6 | u | surge velocity | body x (horizontal, section 1) |
| 7 | v | sway velocity | body y |
| 8 | w | dz/dt | earth vertical (`:365`) |
| 9 | p | d phi/dt | |
| 10 | q | d theta/dt | |
| 11 | r | d psi/dt | |
| 12 | T_s | thrust state of the jet (N "at design speed") | actuator state |
| 13 | delta | nozzle angle (rad) | actuator state |

The learned predictor's five channels (`learn/meta/operators_rb.py:92` RED_VEL = (2,9,8,4,6) on the reduced state `[x, y, u, z, w, theta, q, psi, r, v]` of `control/reduced.py:519-526`) are, in plant indices: u = s[6], v = s[7], r = s[11], heave rate w = s[8], pitch rate q = s[10].

### 3.2 Inputs

* Commands (thrust_cmd, rudder_cmd) passed to `step` (`planing_vessel.py:522`), held constant over a control step of 12 plant steps (`task.py:318-325`).
* Waves: the deterministic function eta(X, Y, t) of section 4 (explicit time dependence of the right-hand side is ONLY through the waves).

### 3.3 Hidden states outside s

* Pump prime P in [0, 1]: `self.prop.prime` (`waterjet.py:150`), advanced once per plant step before the derivative (`planing_vessel.py:524`) by `waterjet.py:208-217`: target = clip(h_in / band, 0, 1) with band = 0.0425 m (section 15), first-order lag with tau 0.05 s (falling) or `tau_prime_back` = 0.5 s (rising, `planing_vessel.py:130`); h_in = intake submergence (`planing_vessel.py:486-495`): h_in = eta(X_in, Y_in, t) - (Z_rest + z + z_in + SP x_in Theta + y_in phi), y_in = 0. P multiplies the jet mass flow (`waterjet.py:172`), hence both f_x and f_y. Changing P changes all six acceleration rows (probe 2).
* Actuator transport-delay buffers inside `prop.advance` (`actuators.py:186-196`, `self.delay`) and `Nozzle.step`. Details belong to the propulsion reader.
* Measurement-only attributes (`_emerged`, `slam_count`, `last_*`) do not enter `deriv`.

### 3.4 Noise

None inside the plant: `Mission(disturb=None)` by default, `final_eval.make_mission` passes no disturbance, the `Recorder` hook returns the state unchanged (`final_eval.py:343-374`). Given (seed, leg, command sequence) the plant is deterministic.

---------------------------------------------------------------------------------------------------

## 4. Wave field (spectrum, directions, seeds, kinematics)

Sea used by the missions: `SeaState(hs=1.0, tp=5.0, theta0=pi, n_freq=24, n_dir=5, seed=seed)` with defaults gamma = 3.3, spread_s = 8, band = (0.5, 3.0), jitter = False (`task.py:56,59,68,243-261`, `env.py:68-69`, `step0_preview_spec.py:59-92`). `sea_kw` is None for these missions.

* Peak frequency omega_p = 2 pi / Tp = 1.25664 rad/s. Frequencies omega_f = linspace(0.5 omega_p, 3 omega_p, 24) = 0.6283 ... 3.7699 rad/s, d omega = 2.5 omega_p / 23 = 0.13659 rad/s (`step0_preview_spec.py:68,73`).
* Directions of TRAVEL theta_l = linspace(theta0 - pi/2, theta0 + pi/2, 5) = {pi/2, 3pi/4, pi, 5pi/4, 3pi/2}, d theta = pi/4 (`:81-82`). theta0 = pi: waves travel toward -X. Leg 0 holds psi = 0 (head seas), leg 1 holds psi = pi/4 (`task.py:69`).
* JONSWAP (`step0_preview_spec.py:30-39`): S(omega) = C omega^-5 exp(-1.25 (omega_p/omega)^4) gamma^r, r = exp(-(omega - omega_p)^2 / (2 sigma^2 omega_p^2)), sigma = 0.07 (omega <= omega_p) or 0.09; C chosen so the trapezoid integral over the 24-point grid equals (Hs/4)^2.
* Spreading (`:42-46`): D(theta) = cos^16((theta - theta0)/2) / trapz(cos^16, over the 5 directions).
* Amplitudes a_j = sqrt(2 S(omega_f) D(theta_l) d omega d theta), j = (f, l), M = 120 components (`:83-89`). Sum a_j^2/2 = 0.062702 vs (Hs/4)^2 = 0.0625 (probe; trapezoid normalisation vs rectangle sum). Energy share per direction: 0.25 %, 17.9 %, 63.6 %, 17.9 %, 0.25 % (probe).
* Deep-water dispersion k_j = omega_j^2 / g (`:90`): k in [0.0402, 1.4487] 1/m, wavelengths 156 m down to 4.34 m (hull L = 5.4 m).
* Phases phi0_j ~ i.i.d. Uniform(0, 2 pi) from `numpy.random.default_rng(seed)` (`:86,91`). seed = the mission seed (0..7; onestep_gz default train 0-3, held out 4-7, `onestep_gz.py:10-11`); both legs of one seed share the same sea. These 120 phases are the plant's only random parameters.
* Because jitter = False the frequencies form an evenly spaced line spectrum: at a fixed earth point the surface repeats every 2 pi / d omega = 46.0 s (cf. `step0_preview_spec.py:99-103`); missions last 120 s (`onestep_gz.py:857`).

Surface quantities, all evaluated at the same phase (`planing_vessel.py:205-231`), with Phi_j(X, Y, t) = k_j (X cos theta_j + Y sin theta_j) - omega_j t + phi0_j:

* eta = sum_j a_j cos Phi_j
* eta_t = sum_j a_j omega_j sin Phi_j = d eta/dt at a FIXED earth point
* eta_tt = - sum_j a_j omega_j^2 cos Phi_j
* eta_X = - sum_j a_j k_j cos(theta_j) sin Phi_j,  eta_Y = - sum_j a_j k_j sin(theta_j) sin Phi_j

Where they are sampled: only on the hull centreline at the N station points X_i = x0 + x_i cos psi, Y_i = y0 + x_i sin psi (`:257-258`) (pitch and roll do not move the sampling points), plus once at the intake for the prime.
Slopes in the body axes: sigma_lon,i = eta_X cos psi + eta_Y sin psi, sigma_tr,i = -eta_X sin psi + eta_Y cos psi (`:259-260`).

What is NOT in the wave model of this plant: no pressure/velocity depth attenuation e^{kz} (`SeaState.eta_at_depth` unused), no horizontal orbital velocity anywhere (u, v enter the forces as hull velocities through still water), no diffraction transfer functions, no wave variation across the beam except through the slope sigma_tr, no second-order waves.

---------------------------------------------------------------------------------------------------

## 5. Immersion and relative vertical velocity per strip (`planing_vessel.py:261-264`)

Theta = Theta_rest + theta. With SP = -1:

* immersion (keel depth below the local surface):
  d_i = eta_i - (Z_rest + z + z_k,i + SP x_i Theta) = eta_i - Z_rest - z + 0.55 + x_i Theta
* local bottom slope: tau_i = k'_i + SP Theta = -Theta  (scarab195)
* relative vertical velocity in the earth-fixed transverse plane (the 2D+t "dd/dt"):
  V_i = eta_t,i - (w + SP x_i q) + u tau_i = eta_t,i - w + x_i q - u Theta

The u tau_i term is the planing (keel descending through a fixed plane) term. Roll phi, sway v and yaw rate r do not enter d_i or V_i.

---------------------------------------------------------------------------------------------------

## 6. Heave and pitch (vertical plane, 2D+t strip theory)

Per strip, the "known" part (`planing_vessel.py:268-281`):

rest_i = m_a,i [eta_tt,i + 2 SP u q - u^2 k''_i] + 1[V_i > 0] m'_a,i V_i^2 + rho C_dc c_i V_i |V_i| chi_i + a_b(u) rho g A_i(d_i) chi_i

with SP = -1 (so 2 SP u q = -2 u q), k''_i = 0, chi_i = 1, and the buoyancy factor (`:246-248`)

a_b(u) = 1 - (1 - a_b,inf) clip(C_V - 0.5, 0, 1),  C_V = |u| / sqrt(g b)  (a_b = 1 for C_V <= 0.5, 0.5 for C_V >= 1.5; C_V = 1 at u = 4.457 m/s).

Full vertical strip force (`:354`):

f_i = rest_i - m_a,i (wdot + SP x_i qdot) = rest_i - m_a,i (wdot - x_i qdot)

The term 1[V>0] m'_a V^2 is the momentum (impact / planing-lift) term, active only while the section enters (c_w < y_c); it is the plant's "slamming" force, and its sum is logged as `last_slam_force` (`:558`) but has no separate slamming model. rho C_dc c V|V| is the vertical cross-flow drag. a_b rho g A(d) is the hydrostatic force on the immersed section measured from the instantaneous LOCAL wave surface, i.e. the Froude-Krylov heave/pitch force (undisturbed hydrostatic pressure, no depth attenuation), reduced by a_b at speed. m_a eta_tt is the wave-acceleration (relative added-mass) term.

Generalised added mass (instantaneous, recomputed every derivative call, `:338-340`):

A33 = sum m_a,i dx,  A35 = sum SP x_i m_a,i dx,  A55 = sum x_i^2 m_a,i dx

Equations (`:341-353`), solved by Cramer's rule:

[ m + A33     A35     ] [wdot]   [ sum_i rest_i dx - m g + F_z,o            ]
[ A35      Iy + A55   ] [qdot] = [ sum_i SP x_i rest_i dx + M_y,o           ]

with the non-strip generalised forces (`tau_o`, section 7-9):

* F_z,o = tau_o[2] = 0 (no term writes into tau_o[2]: friction, air, jet all have f_z = 0)
* M_y,o = tau_o[4] = -zbar_k,wet D_f - 0.5 D_a + z_n f_x, where zbar_k,wet = mean of z_k over wet strips = -0.55 m (`:293`), so M_y,o = 0.55 D_f - 0.5 D_a - 0.40 f_x.

Exact content of d(m_a V)/dt implied by the code: for the earth-fixed plane (x moving at dx/dt = -u), dV/dt = eta_tt - wdot - SP x qdot + 2 SP u q - u^2 k'' + udot tau. The code keeps every term except udot tau: the surge acceleration does NOT enter the vertical strip force (`:280,354`). Whether that is intended is UNKNOWN (no comment in the code).

No Coriolis/centripetal coupling of the body-axis type (m u q, m v p, ...) appears in heave or pitch; heave is earth-vertical.

---------------------------------------------------------------------------------------------------

## 7. Surge (`planing_vessel.py:284-302,327-336,355-358`)

M11 udot = f_x - D_f - D_a + X_w + m v r + X_p,   M11 = 1.05 m

* Friction (ITTC-57 + 0.0004): D_f = (1/2) rho u|u| C_F S_w, C_F = 0.075/(log10 Re - 2)^2 + 0.0004, Re = max(|u|, 0.05) max(l_w, 0.05) / nu, l_w = dx * #{i : d_i > 0}, S_w = sum_i girth_i(d_i) 1[d_i > 0] dx. Applied at (x_w, 0, zbar_k,wet) -> pitch moment -zbar D_f, no roll or yaw moment (`:296`).
* Air drag: D_a = (1/2) rho_air C_DA u|u| applied at (0, 0, 0.5) -> pitch moment -0.5 D_a (`:297-298`).
* Froude-Krylov surge force of the longitudinal slope: X_w = - rho g a_b(u) sum_i A_i(d_i) sigma_lon,i dx (`:300`).
* Pressure drag (horizontal projection of the vertical strip force on the local slope): X_p = - sum_i f_i tau_i dx (`:355`). Because f_i contains wdot and qdot, surge depends on the heave/pitch accelerations of the same call (solved first).
* Rigid-body Coriolis: + m v r (`:334`), divided by M11 = 1.05 m (not by m).
* Jet: + f_x (`:347-348`).
* No gravity component along x (horizontal formulation), no added-mass Munk terms (e.g. (A22 - A11) v r).

---------------------------------------------------------------------------------------------------

## 8. Sway and yaw (slender body + cross-flow, `planing_vessel.py:300-320,334-335,359-361`)

Per strip: m22_i = (pi/2) rho (d_i+)^2 (instantaneous immersion), v_l,i = v + x_i r.

Sway:
M22 vdot = f_y + Y_w + Y_L + sum_i Y_cf,i dx - m u r

Yaw:
M66 rdot = x_n f_y + N_w + x_1 Y_L - |u| sum_i m22_i v_l,i dx + sum_i x_i Y_cf,i dx

* Froude-Krylov of the transverse slope: Y_sec,i = - rho g a_b(u) A_i sigma_tr,i; Y_w = sum Y_sec,i dx; N_w = sum x_i Y_sec,i dx (`:301-304`).
* Munk / slender-body lift at the aftmost STATION (i = 1, x_1 = -2.12495 m, not the transom x = -2.1812 m): Y_L = -|u| m22_1 v_l,1 (`:315-317`), plus the Munk moment -|u| sum m22_i v_l,i dx.
* Lateral cross-flow drag: Y_cf,i = -(1/2) rho C_D,lat d_i+ v_l,i |v_l,i| (`:318-320`).
* Rigid-body Coriolis -m u r (`:335`) divided by M22 (not m).
* Jet: f_y at (x_n, 0, z_n) -> yaw moment x_n f_y (`:347-348`).
* M22, M66 are constants from the design attitude, while the forces use the instantaneous d (this mismatch is why the sub-stepping of section 11 exists, `:526-534`). No sway-yaw added-mass coupling (Mtot is diagonal). No lateral wave diffraction / orbital-velocity term.

---------------------------------------------------------------------------------------------------

## 9. Roll

Parent plant (`planing_vessel.py:322-325`), plus the jet roll moment added at `:348`:

M44 pdot = -m g GM (phi - wbar) - 2 zeta omega_n Ix p - z_n f_y

* wbar = sum_i A_i(d_i) sigma_tr,i / sum_i A_i(d_i)  (area-weighted transverse wave slope under the hull, `:322`)
* -z_n f_y = +0.40 f_y: the jet side force acts 0.40 m below G (from `_load`, `:67-71,347`). Verified: a 10 deg nozzle (f_y = 568 N) changes pdot by 0.26924 rad/s^2 = -z_n f_y / M44 exactly (probe).
* Effective linear damping ratio of the parent equation: zeta_eff = zeta / sqrt(1.2) = 0.0913, because omega_n uses Ix while the inertia is 1.2 Ix. Undamped roll period with M44: 1.4127 s (`planing_vessel_gz.py:207,215`, probe).

GZ plant (`planing_vessel_gz.py:223-239`): the derivative adds m g (GM rel - GZ(rel)) / M44 to pdot with rel = phi - wbar (wbar recomputed exactly as the parent does), so

M44 pdot = -m g GZ(phi - wbar) - 2 zeta omega_n Ix p - z_n f_y

GZ(phi) (`planing_vessel_gz.py:124-202`), computed once per hull:
* each station is closed into a polygon in body (y, z): keel (0, z_k), chine (+-y_c, z_k + min(y_c tan beta, D_h)), deck edge (+-y_c, z_k + D_h) (`:124-135`);
* heeled by phi about body x at the REST trim and rest heave (free trim and free heave not solved), waterline raised by one sinkage delta(phi) found by bisection so that the immersed volume equals the rest volume v0 = 1.36585 m^3 (= m/rho) (`:137-160`);
* GZ_geom(phi) = -y_B = -(sum_i int Y dA) / (sum_i A_i) via boundary integrals (`:80-97,156-157`);
* GZ(phi) = GZ_geom(phi) + (GM - GM_geom) sin(phi), GM_geom = 1.21565 m, shift 1.85e-4 m (`:169-174`), so dGZ/dphi(0) = GM exactly;
* tabulated every 0.25 deg on [0, pi] as GZ/phi and linearly interpolated, extended as an odd 2pi-periodic function (`:189-202`);
* key values (probe): GZ_max = 0.4176 m at 51.25 deg; deck-edge immersion 46.04 deg; vanishing angle 103.34 deg; phi_fail = min(40 deg, ...) = 40 deg (used only for reporting/safety, not in the dynamics).
* GZ depends on phi only (no dependence on speed, heave, pitch, waves other than through the wbar shift).

Capsize (`planing_vessel_gz.py:254-266`): if |phi| >= 103.34 deg at the end of a step, the state is frozen from then on.

---------------------------------------------------------------------------------------------------

## 10. Where thrust and nozzle enter (propulsion internals deferred)

* `(f_x, f_y) = self.prop.forces(T_s, delta, u)` (`planing_vessel.py:327-330`), with T_s = s[12], delta = s[13] and u = s[6] at the derivative's argument, and the current prime P. For T_s >= 0: f_x = mdot (v_j cos delta - v_in), f_y = mdot v_j sin delta, mdot proportional to P (`waterjet.py:172,175-188`).
* Applied as a point load at (x_n, 0, z_n) = (-2.1812, 0, -0.40) m with f_z = 0 (`:347`): generalised force [f_x, f_y, 0, -z_n f_y, z_n f_x, x_n f_y]. So the jet enters the surge row (f_x), sway (f_y), ROLL (-z_n f_y = 0.40 f_y), pitch (z_n f_x = -0.40 f_x; through the 2x2 system also wdot) and yaw (x_n f_y). The thrust is not tilted by the trim (no vertical thrust component).
* Inside `deriv`, ds[12] = ds[13] = 0 (`:362`): T_s and delta are constant through the Heun stages; after the step s[12] <- prop.advance(s[12], thrust_cmd, dt) and s[13] <- rudder.step(s[13], rudder_cmd, dt) (`:552-553`), from the START-of-step values.

---------------------------------------------------------------------------------------------------

## 11. Kinematics, integrator, time steps

Kinematics (`planing_vessel.py:363-365`): xdot0 = u cos psi - v sin psi, ydot0 = u sin psi + v cos psi, zdot = w, phidot = p, thetadot = q, psidot = r.

Integrator (`planing_vessel.py:522-553`): explicit Heun (trapezoidal RK2) with plant step dt = 0.02 s (`hydro/planing.py:134`; `env.py:62-64`; `from_plant` keeps dt, `planing_vessel_gz.py:363`):

1. P <- update_prime(h_in(s, t), dt)  (once, before any stage)
2. k1 = f(s, t); if lambda dt <= 1.8 (lambda = largest |eigenvalue| of the linearised (v, r) equations, `:497-520`): k2 = f(s + dt k1, t + dt), s_new = s + dt (k1 + k2)/2.
   Otherwise n = min(ceil(lambda dt / 0.5), 64) Heun sub-steps of dt/n (`:535-550`). At the initial state lambda dt = 0.10 (probe), so sub-stepping occurs only when the hull is deeply immersed.
3. s_new[12], s_new[13] from the actuator maps (section 10).

Control step: `Episode.sub = max(int(round(dt_ctrl/dt)), 1)` with dt_ctrl = 0.25 s (`env.py:80`); 0.25/0.02 evaluates to 12.5 and Python's round gives 12 (probe). So one control step = 12 plant steps = 0.24 s of plant time, not 0.25 s (the 0.04 s x 6 grid in `final_eval.py:377-383` is consistent with 0.24 s). The one-step error the network predicts is therefore over 0.24 s.

---------------------------------------------------------------------------------------------------

## 12. Complete right-hand side in one place

Let e_i = (eta, eta_t, eta_tt, eta_X, eta_Y) at (x0 + x_i cos psi, y0 + x_i sin psi, t). Then, for scarab195 (chi = 1, k' = k'' = 0, SP = -1):

```
d_i      = eta_i - Z_rest - z + 0.55 + x_i Theta,        Theta = Theta_rest + theta
V_i      = eta_t,i - w + x_i q - u Theta
rest_i   = m_a(d_i)[eta_tt,i - 2 u q] + 1[V_i>0] m_a'(d_i) V_i^2 + rho C_dc c(d_i) V_i|V_i| + a_b(u) rho g A(d_i)
F        = sum rest_i dx - m g
My       = -sum x_i rest_i dx + 0.55 D_f - 0.5 D_a - 0.40 f_x
[m+A33, A35; A35, Iy+A55] [wdot; qdot] = [F; My]       (A33 = sum m_a dx, A35 = -sum x m_a dx, A55 = sum x^2 m_a dx)
f_i      = rest_i - m_a(d_i)(wdot - x_i qdot)
1.05 m udot = f_x - D_f(u, d) - D_a(u) - rho g a_b sum A_i sigma_lon,i dx + m v r + Theta sum f_i dx      (X_p = -sum f_i tau_i dx, tau_i = -Theta)
M22 vdot = f_y - rho g a_b sum A_i sigma_tr,i dx - |u| m22(d_1)(v + x_1 r) - sum (rho/2) C_lat d_i+ v_l,i|v_l,i| dx - m u r
M66 rdot = x_n f_y - rho g a_b sum x_i A_i sigma_tr,i dx - |u| x_1 m22(d_1)(v + x_1 r) - |u| sum m22(d_i)(v + x_i r) dx
           - sum x_i (rho/2) C_lat d_i+ v_l,i|v_l,i| dx
M44 pdot = -m g GZ(phi - wbar) - 2 zeta omega_n Ix p + 0.40 f_y,      wbar = sum A_i sigma_tr,i / sum A_i
(f_x, f_y) = jet(T_s, delta, u, P)
```

---------------------------------------------------------------------------------------------------

## 13. Coupling map (which state enters which acceleration row)

Structural dependence, verified by finite differences of `PlaningVesselGZ.deriv` at a generic state (seed 3, t = 7.3 s, v, r, phi, p, w, q, delta all non-zero; probe 2). X = the row depends on the column.

| row \ col | x0 | y0 | z | phi | theta | psi | u | v | w | p | q | r | T_s | delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| udot | X | X | X | . | X | X | X | X | X | . | X | X | X | X |
| vdot | X | X | X | . | X | X | X | X | . | . | . | X | X | X |
| wdot | X | X | X | . | X | X | X | . | X | . | X | . | X | X |
| pdot | X | X | X | X | X | X | X | . | . | X | . | . | X | X |
| qdot | X | X | X | . | X | X | X | . | X | . | X | . | X | X |
| rdot | X | X | X | . | X | X | X | X | . | . | . | X | X | X |

Reading:
* phi and p enter ONLY the roll row (confirmed; `planing_vessel_gz.py:17-18` is correct on this point). Roll is a one-way output.
* Roll is driven by: wbar (waves; its weights A_i depend on z, theta, x0, y0, psi, t), and the jet side force (T_s, delta, u, P). It is NOT driven only by the wave slope.
* Vertical block (w, q) and surge (u) form one coupled block; it depends on v, r only through the surge Coriolis term m v r and through psi, x0, y0 (where the waves are sampled). wdot, qdot have no direct dependence on v or r.
* Lateral block (v, r) depends on the vertical motion through the immersion d_i (z, theta, waves) in m22, cross-flow and A, and on u (Munk |u|, Coriolis); it does not depend on w or q directly.
* x0, y0, psi enter every acceleration row only through the wave field (in calm water those columns are zero).
* P (prime) multiplies f_x and f_y and changes all six rows (probe 2).

---------------------------------------------------------------------------------------------------

## 14. Docstring vs code, and absent physics (exact statements)

Docstring statements the code contradicts:
1. `planing_vessel_gz.py:61-63` "nothing but the transverse wave slope drives [roll]" and `planing_vessel.py:39` "roll: static GM about the wave slope, linear damping": the code also adds the jet roll moment -z_n f_y = 0.40 f_y (`planing_vessel.py:347-348,360`). Verified numerically.
2. `planing_vessel.py:36-37` "Y = -U m22(transom) v_T ... acting at the transom": the code uses the aftmost station centre x_1 = -lcg + dx/2 (`:315,317`).
3. `planing_vessel_gz.py:125-127,158` call +y "starboard"; in the right-handed z-up body frame used by the equations +y is port (label only).
4. `task.py:209` "one control step (0.25 s)": the plant advances 12 x 0.02 = 0.24 s per control step (`env.py:80`).
5. `planing_vessel.py:18-21` gives d/dt(m_a V) as the strip force; the code's dV/dt omits udot tau (section 6).

Physics absent from the high-fidelity plant (each checked in `deriv`):
* roll -> anything (no heel-induced lift asymmetry, no heel effect on immersion, sway or yaw);
* body-axis Coriolis/centripetal terms other than +m v r (surge) and -m u r (sway); no added-mass Coriolis (Munk) terms in surge; no p, q coupling terms;
* gravity component along the hull; thrust tilt with trim; vertical thrust component;
* horizontal orbital velocities; depth attenuation of wave pressure; lateral wave diffraction; wave variation across the beam except the slope;
* udot tau in the vertical strip force;
* lateral added mass as a function of the instantaneous immersion (forces use instantaneous d, inertias use the design attitude);
* any radiation memory (no Cummins states), any separate slamming model beyond the momentum term m'_a V^2 for V > 0;
* process noise (none).

## 15. UNKNOWN

* Whether the omission of m_a udot tau (section 6) is deliberate: UNKNOWN, the code has no comment on it.
* Internals of `prop.advance` (delay length in plant steps, dead band, rate limit), `Nozzle.step` and `JetPump.v_in/jet_velocity`: not derived here (propulsion reader); only their entry points are stated (section 10).
* (resolved) `prime_band`: the planing plant passes none (`planing_vessel.py:128-133`), so `waterjet.py:148` uses 0.5 * d_nozzle with d_nozzle = sqrt(4 area / pi) = 0.085 m (`waterjet.py:63-65`, area from `planing_vessel.py:123`): band = 0.0425 m of water over the intake for full prime.
