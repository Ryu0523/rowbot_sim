# The low-fidelity reduced model and the exact one-step error target

Scope: the reduced model `ReducedModel` (control/reduced.py) as used (a) as the
reference "model0" that defines the error, (b) as the source plant
`ReducedPlant` (sim/lofi.py) and its batched replicas; the identification of
its coefficients; the exact definition of the error `e0` the network predicts,
its normalisation, and how a high-fidelity (gz) stream becomes training /
evaluation tokens. Every equation cites file:line of the code that implements
it. Values marked (checked) were printed by a short script on this machine
(scratchpad `wf_math/check_lofi*.py`, read-only, < 1 min each).

Hull: `scarab195` (learn/repro/task.py:55). All numbers below are for this hull.

---------------------------------------------------------------------------

## 0. Time bases and index conventions

| symbol | meaning | value | source |
|---|---|---|---|
| Δ | control step actually covered by one recorded step, s | 0.24 (both worlds) | env.py:80 `sub = round(dt_ctrl/dt)`; checked: low dt 0.04 x sub 6, high dt 0.02 x sub 12 |
| dt_ctrl | Mission's nominal control period, s | 0.25 | env.py:65; used by autopilot integral (env.py:152) and data dithers, NOT by Δ |
| h | substep of the reduced model in model0 and in the source plant, s | 0.04 | relabel.py:239-241 (`env["dt"] = m.dt` of `Mission("low")`), lofi.default_dt = 2 x hull dt (lofi.py:344-349) |
| n_sub | substeps of model0 per control step | 6 | relabel.py:239 `sub=m.sub` |
| dt_hi | high-fidelity plant step, s | 0.02, 12 per control step | env.py:64 (`sc["dt"]`), checked |
| k | control-step index, t_k = k Δ (stream times step by exactly 0.24, checked) | | onestep_gz.py:147-158 |

`env` everywhere below is `relabel._env()` (relabel.py:232-242): the reduced
model `red = Mission("low",1,0).red = ReducedModel(p)` with `p = task.ctx()["p"]`
(identified parameters + EXTRA neutral values, task.py:116-126, lofi.py:632),
dt = 0.04, sub = 6, t_max, rud_max, u_ref, L, and the MPC stations x_st, y_off.
mpc_learned.env() returns exactly this object for every e0 computation, also on
the live gz Mission (mpc_learned.py:80-84, 46-48), and check_env asserts that
dt·sub, t_max and rud_max agree with the live Mission (mpc_learned.py:87-93).

---------------------------------------------------------------------------

## 1. State vectors

### 1.1 High-fidelity plant state s^hi (14 numbers)

planing_vessel.py:41-44, deriv unpacking planing_vessel.py:251-254, kinematics
planing_vessel.py:362-366.

| idx | symbol | meaning | frame / unit |
|---|---|---|---|
| 0 | x | CG position | earth-fixed x, m |
| 1 | y | CG position | earth-fixed y, m |
| 2 | z | heave displacement of CG from the floating-at-rest height, positive up (keel immersion d = η − (Z_rest + z + z_k + SP·x·Θ), planing_vessel.py:262) | m |
| 3 | φ | roll angle | rad |
| 4 | θ | pitch from rest trim; SP = SIGN_PITCH = −1, so dz = SP·x·θ and bow-up trim is negative (planing_vessel.py:28-29, 58) | rad |
| 5 | ψ | heading (yaw angle), not wrapped by the plant (ds[5] = r) | rad, earth-fixed |
| 6 | u | surge velocity | body x (rotated by ψ only), m/s |
| 7 | v | sway velocity | body y, m/s |
| 8 | w | heave velocity; the plant integrates ż = w exactly (ds[2] = w) | m/s |
| 9 | p | roll rate (ds[3] = p) | rad/s |
| 10 | q | pitch rate; θ̇ = q exactly (ds[4] = q) | rad/s |
| 11 | r | yaw rate; ψ̇ = r exactly (ds[5] = r) | rad/s |
| 12 | T_s | thrust STATE of the jet actuator ("N at the design speed"; after delay, dead band, rate limit, lag) | N |
| 13 | δ_s | nozzle angle STATE | rad |

Kinematics: ẋ = u cos ψ − v sin ψ, ẏ = u sin ψ + v cos ψ (planing_vessel.py:363-364).
There are no Euler-angle couplings: (ż, φ̇, θ̇, ψ̇) = (w, p, q, r) (planing_vessel.py:365).

Hidden plant states OUTSIDE the 14-vector (object attributes):
- thrust delay line `prop.delay.buf`: 5 entries (0.1 s / 0.02 s; checked `delay steps 5`), actuators.py:29-42, planing_vessel.py:128-133;
- nozzle delay line `rudder.delay.buf`: 5 entries (checked), waterjet.py:234, 237-240;
- pump prime P ∈ [0,1] (`prop.prime`), waterjet.py:150, 208-217, advanced once per step from intake submergence (planing_vessel.py:524);
- GZ plant only: `capsized` flag and frozen state (planing_vessel_gz.py:246-266).

### 1.2 Reduced state s^red (10 numbers)

Defined by the projection P: s^red = P(s^hi) — `ReducedModel.from_plant_state`
(reduced.py:519-526), identically `data2.plant_to_reduced` (data2.py:361-365)
and `operators_rb.to_reduced` (operators_rb.py:1399-1402):

| red idx | symbol | = s^hi index | unit |
|---|---|---|---|
| 0 | x | 0 | m |
| 1 | y | 1 | m |
| 2 | u | 6 | m/s |
| 3 | z | 2 | m |
| 4 | ż (= w) | 8 | m/s |
| 5 | θ | 4 | rad |
| 6 | θ̇ (= q) | 10 | rad/s |
| 7 | ψ | 5 | rad |
| 8 | r | 11 | rad/s |
| 9 | v | 7 | m/s |

The inverse used by the source simulators, `to14` (operators_rb.py:1390-1396,
relabel.py:167-172), writes these 10 back and puts the actuator positions in
s[12], s[13]; it leaves φ = s[3] = 0 and p = s[9] = 0. The source plant
`ReducedPlant` keeps the 14-number layout "with roll and roll rate held at
zero" (lofi.py:353-359, 535-540).

`RED_VEL = (2, 9, 8, 4, 6)` (operators_rb.py:92, data2.py:368): the reduced
indices of (u, v, r, ż, θ̇) — the five velocity channels, in that order.
`VEL_IDX = (6, 7, 11, 8, 10)` is the same five in the 14-vector
(operators_rb.py:93, residuals.py:49).

### 1.3 High-fidelity components NOT in s^red

s^hi[3] = φ, s^hi[9] = p, s^hi[12] = T_s, s^hi[13] = δ_s, plus the hidden
attributes of 1.1 (two delay lines, prime, capsize flag). Their role in the
error law is worked out in section 8.

---------------------------------------------------------------------------

## 2. Parameters of the reduced model

### 2.1 Identified (fixed) parameters, cache studies/_cache/lofi_scarab195.json

Produced by `ReducedModel.identify(plant)` (reduced.py:103-335) on
`config.calm_plant(h, db)` (lofi.py:604, config.py:121-128), i.e. the hull's
own high-fidelity plant class `PlaningVessel` (NOT the GZ subclass; the GZ
swap changes only the roll restoring, planing_vessel_gz.py:223-239) at its
own dt = 0.02 s in calm water (`Monochromatic(1.0, 0.0)` = amplitude 0,
reduced.py:135, test_vessel.py:31-43) and still air. Cached while the calm
plant's fingerprint is unchanged (lofi.py:581-594, 614-631; IDENT_VERSION 1).
Notation rs = sqrt(L/10 m) = 0.7348; thr_id = 8000·t_max/12000 = 4243.8 N
(reduced.py:127); u_des = 12.8611 m/s = 25 kn.

| symbol (code key) | value | unit | how it was obtained (file:line) |
|---|---|---|---|
| t_max | 6365.66 | N | plant.prop.t_max (reduced.py:125, 323) = pump thrust at p_max, u_design (planing_vessel.py:127) |
| rud_max (= rud_stall) | 0.43633 (25°) | rad | plant.rudder.max (reduced.py:323); stall = max for a nozzle (waterjet.py:229) |
| u_design | 12.8611 | m/s | plant.prop.u_ref (reduced.py:128) |
| u_ss | 17.4185 | m/s | surge step: from u0 = 0.8 u_des with constant command thr_id for 240 rs = 176 s; mean of last 200 samples (reduced.py:151-155) |
| tau_u | 4.04 | s | time at which (u − u0)/(u_ss − u0) first exceeds 0.632 (reduced.py:157, 164) |
| k_drag | 13.9872 | N s²/m² | thr_id / u_ss² (steady balance in COMMAND units, reduced.py:165) |
| m_surge | 1968.58 | kg | 2·k_drag·u_ss·tau_u (linearisation of m ü = T − k u², reduced.py:321; comment reduced.py:390-393) |
| wn_heave, z_heave | 5.83486, 0.372324 | rad/s, – | free decay at speed u_ss, command thr_id held, heave released +0.5 T from the running attitude at u_ss, 30 rs = 22 s; AR(2) least-squares fit of the decay, poles → (ω_n, ζ) (reduced.py:169-225; planing branch 209-225) |
| wn_pitch, z_pitch | 3.71347, 0.155060 | rad/s, – | same with pitch released +0.02 rad (reduced.py:173) |
| k_jet_side | 1.61397 | – (side N per command N per sin δ) | steady turn from u_des, command 7000 t_max/12000 = 3713.3 N, nozzle 15°, 90 rs = 66 s: plant's last_jet_force[1] / (3713.3·sin 15°) (reduced.py:263-280) |
| k_nomoto_f | −1.69883e-4 | rad/s per N | yaw step from u_ss, command thr_id, nozzle 20°, 60 rs = 44 s: r_ss / (k_jet_side·thr_id·sin 20°), r_ss = mean of last 100 samples (reduced.py:229-238, 293-296) |
| k_nomoto | −1.14010 | rad/s per rad | r_ss / 20° (unused for the jet; reduced.py:239, 484-487) |
| tau_r | 0.42 | s | first time |r| > 0.632 |r_ss| in the yaw step; stored as max(tau_r, 0.5 rs = 0.367) (reduced.py:240-241, 317) |
| m_sway | 1610.20 | kg | COPIED: plant.Mtot[1,1] = m + Σ m22 dx at the design attitude (reduced.py:257; planing_vessel.py:378-380) |
| k_sway | 391.866 | kg/m | COPIED: plant.visc[1] (reduced.py:258; planing_vessel.py:383) |
| m_coriolis | 1400 | kg | COPIED: plant.M[0,0] = m (reduced.py:259) |
| k_lift | 0 | – | nozzle has no cl_alpha (reduced.py:260-261); unused for the jet |
| k_lin_sway | 1679.94 | kg/s | closes the steady-turn sway balance: (lift_s − k_sway v_s|v_s| − m_cor u_s r_s)/v_s, floored at 0 (reduced.py:283-284) |
| z0, th0 | 0.153663, −0.00922914 | m, rad | plant.running_attitude(u_design) (reduced.py:297-299); captive calm run (planing_vessel.py:386-422) |
| c_wave_sway, c_wave_yaw | 0, 0 | – | pinned to zero (reduced.py:301-311) |
| sign_pitch | −1 | – | sim/vessel.SIGN_PITCH (reduced.py:318) |
| x_bow, x_stern | 3.16255, −2.12495 | m | plant.sec (reduced.py:326-327) |
| B, L, T | 2.0252, 5.4, 0.38463 | m | reduced.py:313 |
| steer_jet, planing | 1, 1 | flags | reduced.py:294, 299 |

Operating points of the identification (all calm water, rudder/nozzle otherwise 0):
surge drag and inertia from ONE step ending at u_ss = 17.42 m/s (33.9 kn, above
the task range 16-30 kn, operators_rb.py:98); heave/pitch frequency and damping
at u_ss; yaw at u_ss; sway at u_des; z0, th0 at u_des.

### 2.2 EXTRA parameters (neutral in model0)

lofi.py:146-149, added to the template (lofi.py:632): k_wave_heave = 1,
k_wave_pitch = 1, tau_thrust = 0, rud_rate = 0, act_family = 0 and the M10
chain neutral (ACT_NEUTRAL, lofi.py:137-138). Checked: ctx p has exactly
these values. With η = 0 (model0) the two k_wave gains multiply zero
(reduced.py:406-407) and have no effect.

### 2.3 Random parameters (source world only, not in model0)

- M10 actuator chain per channel c ∈ {thr, noz}: keys `{c}_{delay,dz,gain,off,db,tau,ratio,zeta,rate}` drawn by `lofi.draw_act` (lofi.py:197-224) from ACT_FAMILY (lofi.py:158-186): w.p. 0.2 the channel is ideal; else delay ~ U[0,0.30]·rs s w.p. 0.5 (rounded to whole 0.04-s steps); tau_thr ~ LogU[0.05,2]·rs, tau_noz ~ LogU[0.02,0.5]·rs w.p. 0.7; given tau: ratio ~ LogU[0.5,2] w.p. 0.3, zeta ~ LogU[0.3,1.5] w.p. 0.3; rate_thr: full range in LogU[0.2,3]·rs s, rate_noz ~ LogU[0.25,2]/rs rad/s w.p. 0.7; dz, db ~ U[0,0.03]·range w.p. 0.3; gain ~ LogU[0.9,1.1] w.p. 0.3 with off ~ U[−0.03,0.03]·range. Training episodes use these (episode8.py:153-160: act_family must be 'm10' or 'at'; Mission plant_params = act).
- lofi.PRIOR (lofi.py:77-107) multipliers of the identified coefficients exist, but the meta7/meta8 episodes pass only `act` as plant_params (episode8.py:158-160), so the source plant's hydrodynamic coefficients are the identified nominal ones; deviations come from the catalogue operators (other derivation).

---------------------------------------------------------------------------

## 3. One substep of the reduced model, f_h

`ReducedModel.step(s, T, δ, η_st, x_st, h)` (reduced.py:366-517), vectorised
over rows. Inputs over the substep (held constant, zero-order hold):
T = applied thrust, N; δ = applied nozzle angle, rad; η_st = surface
elevations, m, at the 5 x 3 stations (shape (N, 5, 3), port / centre /
starboard), evaluated at the substep's START pose and START time by the caller.

Station geometry (lofi.py:380-381, operators_rb.py:144-145): x_st,i =
linspace(x_stern, x_bow, 5) = (−2.12495, −0.80308, 0.51880, 1.84067, 3.16255) m;
y_off = (−B/2, 0, +B/2) = (−1.0126, 0, 1.0126) m (checked). x̃_i = x_st,i − mean(x_st);
D = Σ x̃_i² (reduced.py:371-372).

Wave features (reduced.py:373-388), η_c,i = η_st[i, centre]:
- η̄ = (1/5) Σ_i η_c,i (mean centreline elevation)
- σ = Σ_i x̃_i η_c,i / D (least-squares longitudinal slope, m/m)
- τ_i = (η_st[i, stbd] − η_st[i, port]) / B; t_slope = mean_i τ_i; t_twist = Σ_i τ_i x̃_i / D (used only with c_wave_* = 0).

Equilibria (reduced.py:399-412):
- z̄ = k_wh · η̄ + z0
- ᾱ = k_wp · s_p · arctan(σ) + th0, s_p = sign_pitch = −1.

Substep update, all right-hand sides at the substep START values (u, v, r, …):

1. Surge, explicit Euler (reduced.py:394-395, 491):
   u⁺ = u + h·(T − k_drag·u|u|)/m_surge.
2. Heave, exact transition of z̈ = −ω_z²(z − z̄) − 2ζ_z ω_z ż under ZOH of z̄
   (reduced.py:338-364, 418-422, 492-493):
   [z⁺ − z̄, ż⁺]ᵀ = Φ(ω_z, ζ_z, h)·[z − z̄, ż]ᵀ, Φ(ω, ζ, h) = expm([[0, 1], [−ω², −2ζω]]·h).
3. Pitch, same with (ω_θ, ζ_θ, ᾱ) (reduced.py:419, 423-424, 494-495):
   [θ⁺ − ᾱ, θ̇⁺]ᵀ = Φ(ω_θ, ζ_θ, h)·[θ − ᾱ, θ̇]ᵀ.
4. Jet side force (reduced.py:443-450): δ_e = clip(δ, ±rud_stall);
   F_y = k_jet_side · max(T, 0) · sin δ_e  [N].
5. Sway, explicit Euler (reduced.py:452-453, 477, 497):
   v⁺ = v + h·[(F_y − k_lin_sway·v − k_sway·v|v| − m_coriolis·u·r)/m_sway + c_wave_sway·g·t_slope].
6. Yaw, exact first-order transition towards r_tgt = k_nomoto_f·F_y held over the substep
   (reduced.py:482-488, 496):
   r⁺ = r_tgt + (r − r_tgt)·exp(−h/τ_r) + c_wave_yaw·g·t_twist·h.
7. Heading, trapezoid (reduced.py:498): ψ⁺ = ψ + h·(r + r⁺)/2.
8. Position, with the NEW u, v, ψ (reduced.py:502-504):
   x⁺ = x + h·(u⁺ cos ψ⁺ − v⁺ sin ψ⁺), y⁺ = y + h·(u⁺ sin ψ⁺ + v⁺ cos ψ⁺).

Continuous-time content (what the substep discretises): surge first order
with quadratic drag; heave and pitch two independent LINEAR second-order
oscillators about a wave-dependent equilibrium; sway first order with linear +
quadratic damping and the Coriolis term m u r; yaw first-order Nomoto driven
by the jet side force. No roll; no heave-pitch coupling; no speed dependence of
ω, ζ, z0, th0; no wave forcing of surge, sway or yaw (c_wave_* = 0); no
added resistance (despite reduced.py:9, see section 10).

Couplings that do exist: sway ← u·r (Coriolis), sway and yaw ← T through F_y;
x, y ← u, v, ψ. Heave/pitch depend on the other states only through the
station positions at which η is sampled (none in model0, where η = 0).

---------------------------------------------------------------------------

## 4. The three places the reduced model runs

### 4.1 model0 — the reference that DEFINES the error

`data3.model0_step(env, s^red, U)` (data3.py:66-75):

M(s^red, U) = f_h ∘ … ∘ f_h (n_sub = 6 times), with
T = U_0·t_max, δ = U_1·rud_max held for all 6 substeps (ideal actuators),
η_st ≡ 0 at every substep (a (N, 5, 3) zero array).

U = (U_0, U_1) are the commanded thrust and nozzle as FRACTIONS of t_max and
rud_max (onestep_gz.py:156, data2.py:269, episode8.py:227).

Consequences of η ≡ 0 (exact): η̄ = σ = t_slope = t_twist = 0, so z̄ = z0,
ᾱ = th0 for every speed; heave and pitch relax towards the design-speed
running attitude; u, v, r evolve as in calm water; the actuator positions
s^hi[12:14] are NOT read (model0 uses the commands). x_st does not matter for
model0 (it only enters through slopes of η).

model0 is the same function in torch (Model0T, model3.py:462ff; tested equal
to < 1e-9, studies/test_m9.py:84-96) and is what the learned MPC rolls out
(mpc_constrained.py:252, 276).

### 4.2 The source plant (training data of the catalogue family)

`Mission("low")` builds `ReducedPlant` at h = 0.04 s, n_sub = 6 (env.py:62-80,
checked); episode8 wraps it (episode8.py:170-174) and adds the catalogue's
pushes through the residual hook after each substep (task.py:332-333). The
batched replica `operators_cat.simulate_cat` (operators_cat.py:397-604) runs,
per substep j of control step k (t = t_k + j h):

1. actuators: ideal (applied = command) or the M10 chain
   `lofi.act_step` (lofi.py:273-341; delay line lofi.py:476-493 / operators_cat.py:511-517),
   giving applied T_app, δ_app (time averages over the substep) and new positions;
2. η_st at the 5 x 3 stations of the substep-START pose at the substep-START time
   (ZOH, lofi.py:514-525; operators_cat.py:518);
3. s^red ← f_h(s^red, T_app, δ_app, η_st) (lofi.py:527-529; operators_cat.py:525);
4. velocity kicks: s^red[RED_VEL] += push·h (+ impulse channel)
   (operators_cat.py:567-568; for Mission: CatInjector, operators_cat.py:697ff).

Old-style actuator block (act_family 0): first-order thrust lag with exact
step average, nozzle rate limit with mid-point average (lofi.py:495-512).

Surface: η(X, Y, t) = Σ_c a_c cos(k_c(X cos θ_c + Y sin θ_c) − ω_c t + φ_c)
(lofi.py:450-457; relabel.py:62-73), deep-water k_c = ω_c²/g
(step0_preview_spec.py:90, 139). Station earth positions:
X = x + x_st cos ψ − y_off sin ψ, Y = y + x_st sin ψ + y_off cos ψ (lofi.py:517-519).

### 4.3 The MPC's own model (for completeness)

control/mpc.py's MPPI (the hand controller that generated the gz streams) does
NOT roll out model0: one 0.25-s step, the measured wave at h = 0, a thrust
floor (data3.py:12-15). Only model0 defines e0.

---------------------------------------------------------------------------

## 5. Actuator model inside the reduced model, and the target's for comparison

- model0: none. T and δ are the commands, applied from the first substep.
- source plant: M10 chain or old block (4.2). The SAME parameters are given to
  the plant and to the nominal twin (data2.py:37-45); the twin only produces the
  diagnostic E (section 7.3), not e0.
- target (gz) thrust (actuators.py:186-196, planing_vessel.py:128-133, checked):
  per 0.02-s step, c_d = command delayed by 5 steps; zero if |c_d| < 26.52 N
  (dead band); clip to [t_min, t_max] = [−2121.9, 6365.7] N;
  T_s⁺ = T_s + dt_hi·clip((c_d − T_s)/0.6 s, ±21218.9 N/s). The force on the hull
  uses T_s at the START of the 0.02-s step (ds[12] = 0 inside Heun,
  planing_vessel.py:362, 552; task.py:326-329) and passes through the jet:
  (f_x, f_y) = Waterjet.forces(T_s, δ_s, u) (waterjet.py:175-188), which depends
  on u and on the prime P (waterjet.py:172).
- target nozzle (waterjet.py:237-240, checked): c_d = command delayed 5 steps,
  clipped to ±0.43633 rad; δ_s⁺ = δ_s + dt_hi·clip((c_d − δ_s)/0.1 s, ±0.69813 rad/s).
- Delay-line content at t_k: commands are held constant over the 12 substeps
  of a control step (task.py:323-325), and 5 < 12, so for k ≥ 1 both delay
  lines contain only u_{k−1} (in physical units); at k = 0 `Delay` fills its
  buffer with the first command (actuators.py:39-40), i.e. u_0. During step k
  the actuator target is u_{k−1} for substeps 0-4 and u_k for substeps 5-11.

---------------------------------------------------------------------------

## 6. How waves enter

- model0: they do not (η ≡ 0).
- source plant: only through z̄ = η̄ + z0 and ᾱ = s_p arctan(σ) + th0 (with
  k_wave = 1), i.e. through the centreline elevations at the 5 MPC stations,
  sampled at each 0.04-s substep start (ZOH). Surge, sway and yaw get no wave
  term from the reduced model (c_wave_* = 0, reduced.py:311, 477, 488).
- target: through the strip sums over the plant's own n_stations centreline
  stations (planing_vessel.py:257-262): η, η_t, η_tt, ∂η/∂X, ∂η/∂Y at
  X = x + x_i cos ψ, Y = y + x_i sin ψ, re-evaluated at both Heun stages of every
  0.02-s step (planing_vessel.py:540-542), entering surge (Froude-Krylov X_w and
  pressure drag), sway/yaw (Froude-Krylov Y), heave/pitch (strip force), roll.

---------------------------------------------------------------------------

## 7. The error target

### 7.1 Exact definition

`data3.e0_from(env, xs_k, xs_{k+1}, U_k)` (data3.py:78-88):

    a = P(s^hi_k),  b = P(s^hi_{k+1}),  m = M(a, U_k)       (P: section 1.2, M: section 4.1)
    e0_k[0:8]  = (b[E7] − m[E7]) / Δ
    e0_k[8:10] = (s^hi_{k+1}[12:14] / (t_max, rud_max) − U_k) / Δ

with E7 = (2, 9, 8, 4, 6, 3, 5, 7) (data3.py:59) and Δ = env.dt·env.sub = 0.24 s
(data3.py:84). Channels (data3.py:60-61):

| ch | name | quantity | unit | e_sd (train) |
|---|---|---|---|---|
| 0 | surge | (u_{k+1} − M_u)/Δ | m/s² | 1.59273 |
| 1 | sway | (v_{k+1} − M_v)/Δ | m/s² | 1.43792 |
| 2 | yaw | (r_{k+1} − M_r)/Δ | rad/s² | 0.53958 |
| 3 | heave | (ż_{k+1} − M_ż)/Δ | m/s² | 4.49307 |
| 4 | pitch | (θ̇_{k+1} − M_θ̇)/Δ | rad/s² | 1.33612 |
| 5 | heave_pos | (z_{k+1} − M_z)/Δ | m/s | 0.69979 |
| 6 | pitch_pos | (θ_{k+1} − M_θ)/Δ | rad/s | 0.16066 |
| 7 | heading | (ψ_{k+1} − M_ψ)/Δ, no angle wrap | rad/s | 0.06045 |
| 8 | thrust_act | (T_s,k+1/t_max − U_k,0)/Δ | 1/s | 0.22140 |
| 9 | nozzle_act | (δ_s,k+1/rud_max − U_k,1)/Δ | 1/s | 0.32502 |

"Which next velocity": the high-fidelity (or source-plant) state at t_{k+1},
projected by P. "Which prediction": model0 from the SAME time-k state projected
by P, with the commands U_k applied during step k, no waves, ideal actuators,
6 substeps of 0.04 s. Since a and b share the time-k values,

    e0_k,c = (1/Δ) ∫_{t_k}^{t_k+Δ} ṡ_c^true dt − (M_c(a, U_k) − a_c)/Δ,  c ∈ E7,

i.e. (mean true rate of change of channel c over the step) − (mean model0 rate
of change). For the 5 velocity channels this is a difference of step-mean
accelerations (m/s² or rad/s²).

The network's head outputs all 10 channels (model3.py:46 C7 = 10); the one-step
yardstick reports the first five (onestep_gz.py:118, 36-41) and the MPC cost
reads channels 0 and 3 (mpc_learned.py:76).

### 7.2 Normalisation and the transformed target

- e_sd = per-channel standard deviation of e0 over the valid steps of the
  TRAINING split ("train", meta7 source episodes, D10 catalogue family) + 1e-6
  (model3.py:98-101; meta_step7.py:408-433; the w/p variants reuse model3.pt's
  stats, meta_step7.py:430-433). Values (checked, model_p.pt and model3.pt
  identical): see table above.
- Network target in normalised units: E_k = float32(e0_k) / e_sd (mpc_learned.py:541-542, 584-591; model3.py:108).
- The flow head models y = asinh(E_k/κ), κ = 0.5 (model3.py:50, 238); samples are mapped back by κ sinh(y) (model3.py:178, 197).
- In a rollout, the next reduced state is m + Δ·(e·e_sd)[0:8] placed at E7, and the next actuator positions clip(U + Δ·(e·e_sd)[8:10], [0,−1], [1,1]) (mpc_constrained.py:246-278), i.e. e0's definition inverted; x, y get no correction (not in E7).
- Previous error enters the next token as asinh(E_{k−1}/κ) (model3.py:118-121); input state features are inputs11 (section 7.4) normalised by lib mu/sd (model3.py:100-104; values printed in check output).

### 7.3 Quantities with similar names that are NOT the target

- data2 / episode8 `E` (data2.py:266, episode8.py:257): (plant − nominal twin)/Δ on the 5 velocities, where the twin is the reduced plant WITH the same waves and the same actuator parameters and actuator hidden state; it isolates the operator's push. Diagnostic only; data2.py:52-54 still calls it "the target quantity" (outdated, section 10).
- relabel `e` (relabel.py:76-85, 194-195): push_response_red, the effect of a push on model0 with no waves and commanded actuators; "Nothing downstream of step 3 uses this e" (relabel.py:126-131). relabel's role for the current nets is only `_env()` (the env of model0) and `simulate` (closed-loop branch truths whose E0 is again e0_from, data3.py:193-221).

### 7.4 From a gz stream to tokens

1. Record (onestep_gz.py:136-169): `FE.make_mission` = Mission("high") with its plant replaced by the GZ twin (final_eval.py:208-223); per control step: thrust from the hand MPC, nozzle from the heading autopilot (task.py:295-300, env.py:93-154), SafeMode may modify both (onestep_gz.py:150); `m.advance` runs 12 plant steps of 0.02 s (task.py:318-356); the Recorder hook returns the state unchanged (final_eval.py:343-373). Stored: xs (n+1, 14) at t_k, t (n+1), u (n, 2) = APPLIED command fractions, apk, hmin, the sea components (a, k, th, w, phi), consts. A capsized step ends the recording and is not stored (onestep_gz.py:153-154).
2. `full_live` (onestep_gz.py:217-221) → `LiveDataW.push` (mpc_constrained.py:201-208) → `LiveData.push` (mpc_learned.py:579-596): E0[k] = e0_from(env, xs_k, xs_{k+1}, u_k) (float64), cast to float32, E[k] = E0[k]/e_sd; X[k+1] = normalised inputs11(xs_{k+1}); U[k] = (u_k − u_mu)/u_sd; valid[k] = True. Waves: W15 = true elevations at the 15 MPC stations of the MEASURED pose at t_{k+1}, WMID15 = true elevations at the stations of the pose dead-reckoned h_m = 0.12 s ahead (x + (u cos ψ − v sin ψ)h_m, y + (u sin ψ + v cos ψ)h_m, ψ + r h_m) at t + 0.12 s (mpc_constrained.py:179-195; episode5.py:44-67).
3. inputs11(s^hi) (mpc_learned.py:524-530, residuals.py:52-56) = [u/u_ref − 1, v, r·L/u_ref, cos ψ, sin ψ, T_s/t_max, δ_s/rud_max, ż, θ̇, (z − z0)/0.2, (θ − th0)/0.05]. NOT included: x, y, φ, p.
4. Variant p tokens add 34 wave columns (model_preview.py:8-17): 15 measured elevations (sensor noise sd 0.01 library units at evaluation, onestep_gz.py:246, 311), 15 preview elevations of W_MID with forecast error level λ = FE.OPT fc_lam = 0.1 (final_eval.py:106), flags, steps since measurement, error level.

---------------------------------------------------------------------------

## 8. The error law as a function: what it depends on, what is observed

### 8.1 Exact dependence

Let Π_k denote the true plant's state trajectory on [t_k, t_k + Δ]. For a velocity
channel c, e0_k,c = Ā_c − Ā_c^M with Ā_c = (1/Δ)∫ a_c^hi dt over Π_k and
Ā_c^M = (M_c(P s^hi_k, U_k) − (P s^hi_k)_c)/Δ.

- Ā_c^M is a known, deterministic function of (P s^hi_k, U_k) only (model0).
- Ā_c is determined by: the hi-fi state at t_k, the hidden actuator/prime states,
  the commands, and the wave field over the hull during the step:
  Ā_c = G_c(s^hi_k, buf_thr,k, buf_noz,k, P_k, U_k, {η(·,·,t)}_{t ∈ [t_k, t_k+Δ]}).

### 8.2 Roll does not feed back (code fact)

In `PlaningVessel.deriv` (planing_vessel.py:250-369) φ appears only in the roll
restoring moment (line 324) and p only in roll damping (line 325); both enter
only τ_o[3] → ṗ (line 360). The immersion d (line 262), the sway/yaw slender-body
and crossflow forces (lines 313-320), the heave/pitch strip force (lines 280-354)
and the jet (lines 327-348) contain no φ or p; Mtot is diagonal
(planing_vessel.py:108, 378-382), so no inertial coupling. The intake lies on the
centreline (y_i = 0, planing_vessel.py:141), so prime does not see roll
(planing_vessel.py:492-494). The GZ subclass changes only ds[9]
(planing_vessel_gz.py:237-238). Hence, as long as the GZ plant has not capsized,
G_c for c ∈ {u, v, r, ż, θ̇} does NOT depend on φ or p. Roll is a pure output in
this target; it is not an unobserved input of the five-channel error law
(only of the capsize stop, which ends a stream). Magnitude in the 16 recorded
missions (checked): std φ = 0.0755 rad, max |φ| = 0.305 rad, std p = 0.291 rad/s.

### 8.3 Classification of every argument of G_c

| argument | in s^red? | used by model0? | in the network's input? | status for the error law |
|---|---|---|---|---|
| u, v, r, z, ż, θ, θ̇ | yes | yes | yes (inputs11) | observed |
| ψ | yes | yes (kinematics only) | yes (cos ψ, sin ψ) | observed |
| x, y | yes | yes (kinematics only) | NO | enter G_c only through the wave phase k_c(X cos θ_c + Y sin θ_c); unobserved, replaced by the wave columns |
| φ, p | no | no | no | no effect on G_c (8.2) |
| T_s, δ_s (actuator states) | no | no (commands used) | yes (inputs11[5:7]) | observed |
| thrust / nozzle delay lines | no | no | indirectly: = u_{k−1}, the previous token's command (k ≥ 1); = u_0 at k = 0 (section 5) | observed through history |
| pump prime P_k | no | no | no | hidden state; P_k < 1 only after the intake rose to less than 0.0425 m of water (prime_band, checked); inferable only from history |
| U_k (commands during step k) | – | yes | yes | observed |
| wave field over the whole hull during [t_k, t_k+Δ]: η, η_t, η_tt, ∂η/∂X, ∂η/∂Y at the plant's n_stations stations, 24 Heun stage evaluations | no | no (η ≡ 0) | partially: 15 elevations at t_k at the MPC's 5 x 3 stations (+ sensor noise) and 15 forecast elevations at the dead-reckoned mid-step pose (variant p) | partially observed |
| Heun sub-substep count n (planing_vessel.py:535-551) | – | – | – | deterministic function of the state; not an input |

Not present in the target: sensor noise on xs (the streams store exact plant
states), wind (task Mission has none), currents.

### 8.4 What the source world's e0 contains (for comparison with 8.3)

In the source world, s^hi is the reduced plant's 14-vector (φ = p = 0), so
e0^src_k = (1/Δ)[s^red_{k+1} − M(s^red_k, U_k)] with s^red_{k+1} from 4.2, which
contains (i) the reduced model's own heave/pitch wave response (z̄ = η̄ + z0,
ᾱ = s_p arctan σ + th0 instead of z0, th0), (ii) the M10 actuator effect
(applied ≠ command), (iii) the catalogue pushes and their propagation. Heave and
pitch of the reduced model are linear with constant coefficients, so for those
two channels (i) and the push contribution add exactly for given station
positions; surge/sway/yaw contain the nonlinear terms u|u|, v|v|, u·r and
max(T,0)·sin δ, so the parts do not add exactly there.

---------------------------------------------------------------------------

## 9. Numbers checked on this machine

### 9.1 Steady straight run in calm water (no waves, actuators settled)

Target thrust state T_eq solving Waterjet.forces(T_eq, 0, u)[0] = calm resistance
R(u) (planing_vessel.py:424-439), state at the target's running attitude at u,
command = T_eq/t_max; e0 = e0_from(s, s, U) (true next = same state):

| speed | R(u) N | T_eq N | k_drag u² N | e0 (u, v, r, ż, θ̇) | e0 (z, θ) | target trim θ(u) vs th0 |
|---|---|---|---|---|---|---|
| 16 kn | 1868.5 | 1492.5 | 947.6 | (−0.2736, 0, 0, −0.369, −0.441) | (−0.063, −0.059) | −0.0512 vs −0.0092 |
| 20 kn | 1923.5 | 1706.9 | 1480.7 | (−0.1132, 0, 0, −0.074, −0.210) | (−0.013, −0.028) | −0.0292 |
| 25 kn | 2311.2 | 2311.2 | 2313.6 | (0.0012, 0, 0, 0, 0) | (0, 0) | −0.0092 |
| 30 kn | 2981.2 | 3266.0 | 3331.6 | (0.0326, 0, 0, −0.039, 0.120) | (−0.007, 0.016) | +0.0022 |

So e0 has a static, speed-dependent part even in calm water: model0's surge
balance T − k_drag u² is calibrated at one point (u_ss) in command units, and its
heave/pitch equilibria are the 25-kn attitude at every speed. Near-zero error at
25 kn is a numerical observation (2.4 N); the code does not enforce it
(k_drag is set at u_ss, reduced.py:165).

### 9.2 All 16 recorded gz missions (streams_T120, 7680 steps)

e0 std (physical units): surge 0.924, sway 0.714, yaw 0.311, heave 4.620,
pitch 0.531, heave_pos 0.780, pitch_pos 0.0824, heading 0.0434,
thrust_act 0.268, nozzle_act 0.515. Ratio e0 std / e_sd(train): surge 0.580,
sway 0.497, yaw 0.576, heave 1.028, pitch 0.398, heave_pos 1.115,
pitch_pos 0.513, heading 0.718, thrust_act 1.208, nozzle_act 1.584. Means / e_sd: surge −0.280, heave −0.136, pitch −0.108,
heave_pos −0.149, pitch_pos −0.122, others |·| < 0.011. Thrust command
fractions in [4.6e-7, 0.897].

### 9.3 One mission (seed 0, leg 0): how much of target e0 is the reduced model's own wave response

Running the reduced model with the TRUE sea at its stations (ZOH per 0.04-s
substep), ideal actuators, from each recorded state, gives a "lofi + waves"
prediction. Std of the 5 velocity errors:

| | surge | sway | yaw | heave | pitch |
|---|---|---|---|---|---|
| e0 (vs model0, the target) | 1.002 | 0.734 | 0.304 | 5.164 | 0.618 |
| vs lofi + true waves | 1.002 | 0.734 | 0.304 | 2.729 | 0.664 |
| lofi wave part alone ((lofi+waves − model0)/Δ) | 0 | 0 | 0 | 3.693 | 0.515 |
| corr(e0, lofi wave part) | – | – | – | 0.861 | 0.323 |

Single mission only; not a population statement.

---------------------------------------------------------------------------

## 10. Where docstrings disagree with the code (code followed)

1. reduced.py:90-94 (class docstring): "State: [x, y, u, z, zdot, th, thdot, psi, r] … sway is neglected, so ydot = u sin(psi)". Code: 10 states with v at index 9 and full kinematics (reduced.py:369, 452-453, 502-504, 519-526).
2. reduced.py:9: "surge first-order thrust response plus quadratic drag and added resistance". Code: no added-resistance term (reduced.py:394-395).
3. lofi.py:19-20: "integrated at a finer step than the MPC's 0.5 s (dt = 0.1 s for the USV…)". For this hull the source plant runs at 0.04 s (Episode passes lofi.default_dt, env.py:64; checked) and the control step is 0.24 s; lofi.plant_for's own default (DT_USV·sqrt(L/10) = 0.0735 s, lofi.py:656-657) is not the one used by Mission or relabel._env.
4. reduced.py:343-345 (_phi) and 479-481: "control step 0.5 s, wn 3.1 / 3.6, tau 6.3 s, tau_r 0.5" — USV numbers; here h = 0.04 s, ω_z = 5.83, ω_θ = 3.71, tau_u = 4.04, tau_r = 0.42.
5. model3.py:11: "7 channels"; code C7 = 10 (model3.py:46).
6. data2.py:52-54 calls E "the target quantity"; the current target is e0 (model3.py:86, data3.py).
7. lofi.py:13-17 describes rudder lift and rudder Nomoto; for this hull the jet branch runs (steer_jet = 1): F_y = k_jet_side·max(T,0)·sin δ, r_tgt = k_nomoto_f·F_y (reduced.py:444-450, 483-485).
8. env.py:100: "K = −1.50 and T = 3.35" (USV); here K = k_nomoto_f·k_jet_side·T_now, T = 0.42 s.

---------------------------------------------------------------------------

## 11. UNKNOWN

- Whether the near-equality k_drag·u_des² ≈ R(u_des) (2313.6 vs 2311.2 N) is designed: no code enforces it; reason unknown.
- The exact distribution of the meta7 "train" split (which episodes, catalogue items on) behind e_sd: not derived here (catalogue derivation is another task); only the resulting e_sd values were read.
- How often the pump prime drops below 1 in the gz streams: the streams do not store prime (onestep_gz.py:165-169; the GZ plant's own log has submergence but record() does not save it); not determined.
- control/mpc.py's internal rollout details (the data-generating controller's model): only the data3.py:12-15 statement was used; mpc.py not read.
