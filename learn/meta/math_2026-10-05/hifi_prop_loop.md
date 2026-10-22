# High-fidelity (gz) target: propulsion, actuators, control loop and recording protocol

Scope: the waterjet thrust channel, the steering nozzle, the pump prime, the jet
forces and moments as they enter the gz plant's equations of motion; the
"hand" controller that drove the 16 recorded missions; the sea of those
missions; mission length and speeds; and the exact content and rate of the
recorded streams (`studies/_cache/onestep_gz/streams_T120.pkl`).

All paths are relative to `C:/Users/Administrator/Documents/usv-rl-mpc`.
Every equation cites the implementing line(s). Where a docstring and the code
disagree, the code is followed and the disagreement is listed in section 12.

## 0. Verification done for this document (all on this machine, 2026-10-04)

| check | result |
|---|---|
| `sim/lofi.py` identified-parameter cache `studies/_cache/lofi_scarab195.json` is the one in use | fingerprint of today's calm plant = 36.50273627118908 = cached value (`lofi.template`, sim/lofi.py:597-634), so the numbers quoted below from that file are the ones every run loads |
| closed-loop rerun of the hand row with today's code, mission seed 0 leg 0, all 480 control steps (script `wf_math/closed_loop.py`, the loop of `onestep_gz.record` without the recorder) | applied commands and all 481 stored states reproduced **bit-exactly** (max abs difference 0.0) |
| the same for seed 7 leg 1, first 120 steps | bit-exact |
| open-loop replay of the stored commands through a fresh gz plant (`wf_math/replay.py`) for missions (0,0),(0,1),(2,0),(3,1),(4,0),(7,1) | (7,1) exact; (3,1) max state difference 1.2e-11; (0,0) 1.3e-6; (0,1) 1.1e-7; (2,0),(4,0) deviate by > 1e-3 after control steps 389 / 359 (a 1e-12 difference from the stored command fraction `thrust/t_max` times `t_max` grows in the open loop; no heading feedback in a replay). Used only for the hidden-state statistics of section 5.4 |

Conclusion of the checks: the code quoted below is the code that produced the
stored streams (recorded 2026-10-02 19:12; `git log` shows no commit touching
`sim/`, `control/`, `learn/repro/task.py`, `studies/sim2real_jet.py` after
2026-10-01; `studies/onestep_gz.py` is untracked, but the closed-loop rerun
shows its `record()` loop law equals the stored data).

## 1. Symbols (defined once)

Frames. Earth frame: horizontal axes (X, Y), calm free surface at height 0.
Body axes about the centre of gravity (CG): x forward, y lateral, z up (the
planing plant's kinematics, sim/planing_vessel.py:363-365, use dX/dt = u cos psi - v sin psi,
dY/dt = u sin psi + v cos psi and dz/dt = w, d(phi)/dt = p, d(theta)/dt = q, d(psi)/dt = r directly; no Euler-rate
transformation).

Plant state vector `s` (14 entries, sim/planing_vessel.py:41-44, 463-464):

| index | symbol | meaning | unit |
|---|---|---|---|
| 0,1 | X, Y | CG position, earth frame | m |
| 2 | z | heave, measured from the rest floating position (z > 0 = risen) | m |
| 3 | phi | roll | rad |
| 4 | theta | pitch, from the rest trim; SIGN_PITCH SP = -1 (sim/vessel.py:46): bow-up = negative theta | rad |
| 5 | psi | heading | rad |
| 6 | u | surge velocity, body | m/s |
| 7 | v | sway velocity, body | m/s |
| 8 | w | heave rate = dz/dt | m/s |
| 9 | p | roll rate | rad/s |
| 10 | q | pitch rate = d(theta)/dt | rad/s |
| 11 | r | yaw rate = d(psi)/dt | rad/s |
| 12 | T | thrust state of the waterjet, "net thrust it would deliver at the design speed" | N |
| 13 | delta | nozzle angle | rad |

The five learned channels (u, v, r, heave rate, pitch rate) are `s[6], s[7], s[11], s[8], s[10]`;
in the reduced state `sr = ReducedModel.from_plant_state(s) = [X, Y, u, z, w, theta, q, psi, r, v]`
(control/reduced.py:519-526) they are indices (2, 9, 8, 4, 6) = operators_rb.RED_VEL.

Time indices. Plant step n, length Delta_t = 0.02 s. Control step k, consisting of
N_sub = 12 plant steps, length Delta_t_c = 12 x 0.02 = 0.24 s. Plant step n = 12 k + j, j = 0..11.

Inputs (commands), constant over one control step k (zero-order hold):
* U_k^T in [0, 1]: thrust command fraction; c_T,k = U_k^T * T_max (N).
* U_k^delta in [-1, 1]: nozzle command fraction; c_delta,k = U_k^delta * delta_max (rad).

Hidden actuator states not in `s`: the two transport-delay lines (5 plant steps each) and the
pump prime pi in [0, 1] (section 5).

Fixed parameters of the scarab195 hull (hydro/hulls.py:145-156, hydro/planing.py:99-136) used here:
L = 5.4 m, chine beam b = B = 2.0252 m, mass m = 1400 kg, lcg = 0.38 x 5.74 = 2.1812 m from the transom,
vcg = 0.55 m above the keel, depth 1.1 m, u_design = U_ref = 25 kn = 12.8611 m/s,
rho = 1025 kg/m^3, plant dt = 0.02 s, nominal control step dt_ctrl = 0.25 s (planing.py:134).

There are **no random parameters** in the gz target: every plant and actuator coefficient is a
fixed number (no draw from any prior). The only random quantities per mission are the wave phases
and the MPPI sampling noise, both seeded by the mission seed (sections 8, 9).

## 2. Time grid and loop order

* Plant step Delta_t = 0.02 s: `Hull.scales()["dt"]` (hydro/planing.py:134) via `Episode` with
  fidelity "high" (sim/env.py:62-64) and copied into the gz twin by `from_plant`
  (sim/planing_vessel_gz.py:359-368; `_init_kw` dt, planing_vessel_gz.py:118).
* Control step: `Episode.sub = max(int(round(dt_ctrl / dt)), 1) = round(0.25/0.02) = round(12.5) = 12`
  (sim/env.py:80; Python rounds half to even), so the plant advances 12 x 0.02 = 0.24 s per
  control step (task.Mission.advance, learn/repro/task.py:318-356).
* Mission length: `n_ctrl = int(round(t_end / dt_ctrl)) = round(120 / 0.25) = 480` control steps
  (task.py:282), i.e. 480 x 0.24 = 115.2 s of plant time (stored `t[-1] = 115.2` in every stream).
  Note: `dt_ctrl` = 0.25 s sets only the count, not the step actually taken.

Loop of one recorded mission (`studies/onestep_gz.py:136-169`, the hand row of
`studies/final_eval.py:455-457, 521-565`), for k = 0..479:

1. `thrust, rud = m.mpc_command()` (task.py:295-300): MPPI thrust from the reduced state of the
   current plant state; autopilot nozzle from the current plant state (section 8).
2. `thrust, rud, on = safe(m.s[3], thrust, rud)`: measured-roll safe mode (section 8.3).
3. `m.advance(thrust, rud)`: 12 plant steps with the same (thrust, rud) (task.py:323-325).
4. Store `u_k = (thrust/T_max, rud/delta_max)`, `xs_{k+1} = s`, `t_{k+1}`, `apk_k`, `hmin_k`.

The command at step k is computed from the state at t_k with no computation delay; the actuator
transport delays (section 3, 4) then act inside the plant.

Inside one plant step (`PlaningVessel.step`, sim/planing_vessel.py:522-569; the gz override
planing_vessel_gz.py:254-273 only adds the capsize hold and logging):

1. prime update from the intake submergence at the step-start state and time t_n
   (planing_vessel.py:524; section 5);
2. rigid-body integration by Heun (explicit trapezoid) of `deriv` (planing_vessel.py:539-550);
   substeps when `_lateral_rate * dt > 1.8` (planing_vessel.py:535-538; never triggered in the
   six replayed missions: `last_n_sub = 1` at every plant step). `deriv` returns
   ds[12] = ds[13] = 0 (planing_vessel.py:362-366), so both Heun stages use the step-start
   thrust state T(n) and nozzle angle delta(n), and the prime value just updated;
3. actuator update overwrites `s_new[12] = prop.advance(s[12], thrust_cmd, dt)`,
   `s_new[13] = rudder.step(s[13], rudder_cmd, dt)` (planing_vessel.py:552-553): explicit Euler,
   using the step-start actuator state.

So within a plant step the jet acts with a zero-order hold of (T(n), delta(n), pi_n^+), and the
surge speed u at each Heun stage.

## 3. Thrust channel (commands -> thrust state)

Objects: `Waterjet(Propulsion)` built inside `PlaningVessel.__init__`
(sim/planing_vessel.py:118-133), NOT by `sim/waterjet.build()` (waterjet.py:246-280 is unused by
the planing plant; its defaults 27 deg, tau 0.15 r, sizing by ratio 0.35 do not apply).

Parameters (values printed from the live object, `wf_math/check_plant.py`):

| symbol | code | value | source line |
|---|---|---|---|
| T_max | `t_max = pump.thrust(p_max, u_design)` | 6365.66 N | planing_vessel.py:126-127 |
| T_min | `-t_max/3` | -2121.89 N | planing_vessel.py:131 |
| tau_T | `j.get("tau", 0.6)`, hull gives 0.6 | 0.6 s | planing_vessel.py:131; hulls.py:151 |
| Tdot_max | `t_max/0.3` | 21218.9 N/s | planing_vessel.py:132 |
| T_db | `t_max*50/12000` | 26.524 N | planing_vessel.py:132 |
| n_d,T | `Delay(0.1, 0.02).n = round(0.1/0.02)` | 5 plant steps = 0.10 s | planing_vessel.py:133; actuators.py:32-34 |

Delay line (actuators.py:29-42): on its first call the buffer is filled with n copies of the first
value; afterwards each call appends the new value and returns the oldest. Hence the released
command is

    chat_T(n) = c_T(n - 5)  for n >= 5,     chat_T(n) = c_T(0) for n < 5,

with c_T(n) = c_T,k for n in control step k. A fresh plant (hence fresh, empty delay lines) is built
for every mission (`make_mission` -> `from_plant`, final_eval.py:208-223), and `initial_state`
does not touch the delay lines.

Update (`Propulsion.advance`, actuators.py:186-196), in code order delay -> dead band -> saturation
-> rate-limited first-order lag:

    c' = chat_T(n) * 1[|chat_T(n)| >= T_db]
    c'' = clip(c', T_min, T_max)
    T(n+1) = T(n) + Delta_t * clip((c'' - T(n)) / tau_T, -Tdot_max, Tdot_max).

Facts that follow exactly from these numbers and the recorded command range:
* Commands satisfy 0 <= c_T <= T_max (MPPI thrust fraction in [floor, 1] = [0, 1],
  control/mpc.py:175, 151; safe mode only halves), so saturation never acts and the reverse
  branch of the jet is never used.
* For c'', T in [0, T_max]: |c'' - T|/tau_T <= T_max/0.6 = 0.5 Tdot_max, so **the rate limit never
  acts**. The thrust channel is exactly delay + dead band + linear first-order lag.
* With alpha = Delta_t/tau_T = 1/30, over one control step (5 plant steps release c_{k-1}, then 7
  release c_k; for k = 0 both are c_0):

      T_{k+1} = (1-alpha)^12 T_k + [(1-alpha)^7 - (1-alpha)^12] D(c_T,k-1) + [1 - (1-alpha)^7] D(c_T,k)
              = 0.665765 T_k + 0.122981 D(c_T,k-1) + 0.211254 D(c_T,k),
      D(c) = c * 1[c >= T_db].

  T_k = xs[k][12] is stored, so this map uses only stored quantities.
* The dead band acts when 0 < U^T < 26.524/6365.66 = 0.004167; the released command was inside
  (0, T_db) at 0-0.42 % of plant steps in the replays. Recorded thrust state range over the 16
  missions: 164-5110 N.

## 4. Nozzle channel (commands -> nozzle angle)

Object: `Nozzle` (sim/waterjet.py:220-240), built at planing_vessel.py:134-138.

| symbol | code | value |
|---|---|---|
| delta_max | `nozzle_max_deg` 25 (hulls.py:150) | 0.436332 rad = 25 deg |
| deltadot_max | `nozzle_rate_deg` 40 (hulls.py:150) | 0.698132 rad/s |
| tau_delta | `j.get("nozzle_tau", 0.1)`, not in hull dict | 0.1 s |
| n_d,delta | `j.get("nozzle_delay", 0.1)` -> 5 plant steps | 0.10 s |

Update (waterjet.py:237-240), no dead band:

    chat_delta(n) = c_delta(n - 5)   (same delay-line rule as section 3)
    delta(n+1) = delta(n) + Delta_t * clip((clip(chat_delta(n), -delta_max, delta_max) - delta(n)) / tau_delta,
                                          -deltadot_max, deltadot_max).

* The rate limit acts when |c - delta| > deltadot_max * tau_delta = 0.0698 rad = 4 deg. In the
  replays it acted at 6.0-9.7 % of plant steps; the released command sat at a stop (+-25 deg) at
  4.8-8.1 %; the stored nozzle command fraction is +-1 at 3.1-8.5 % of control steps (16 missions).
* Without the rate limit (beta = Delta_t/tau_delta = 0.2), over one control step:

      delta_{k+1} = 0.8^12 delta_k + (0.8^7 - 0.8^12) c_delta,k-1 + (1 - 0.8^7) c_delta,k
                  = 0.068719 delta_k + 0.140996 c_delta,k-1 + 0.790285 c_delta,k ;

  with the rate limit active the map is piecewise (slew at 0.698 rad/s). delta_k = xs[k][13] is
  stored. Full-travel slew 50 deg / 40 deg/s = 1.25 s = 5.2 control steps.

## 5. Pump prime (hidden state)

### 5.1 Intake submergence (sim/planing_vessel.py:486-495)
Intake point in body coordinates about the CG: x_in = -lcg + 0.8 = -1.3812 m, y_in = 0,
z_in = -vcg + 0 = -0.55 m (planing_vessel.py:121-122; hull `x_intake` 0.8, hulls.py:150).
With the incident elevation eta(X, Y, t) of section 9:

    X_in = X + x_in cos psi,  Y_in = Y + x_in sin psi
    h_in(n) = eta(X_in, Y_in, t_n) - [Z_rest + z + z_in + SP x_in (Theta_rest + theta)]      (y_in phi term = 0)

Z_rest = 0.233355 m, Theta_rest = -0.0319937 rad (the rest attitude, planing_vessel.py:111, 177-190).
At the initial state (z = 0.1487 m, theta = -0.02921 rad) the bracket is -0.2526 m: the intake is
0.25 m under calm water. The submergence uses the step-start state and time.

### 5.2 Prime law (sim/waterjet.py:208-217)
prime band b_pi = 0.5 d_nozzle = 0.0425 m (waterjet.py:148-149; `prime_band` not passed),
tau_loss = 0.05 s, tau_back = `j.get("tau_prime_back", 0.5)` = 0.5 s (planing_vessel.py:130):

    pi*(n) = clip(h_in(n) / b_pi, 0, 1)
    tau_pi = tau_loss if pi*(n) < pi(n) else tau_back
    pi(n+1) = pi(n) + (pi*(n) - pi(n)) * (1 - exp(-Delta_t / tau_pi)),

per plant step the gain is 0.32968 (loss) or 0.03921 (recovery); recovery from 0 to 0.999 takes
0.5 ln 1000 = 3.45 s = 14.4 control steps. pi = 1 at `_reset` (planing_vessel.py:453-454), i.e. at
`initial_state`. The value pi(n+1) updated at the start of plant step n is the one used by the
forces during step n (planing_vessel.py:524 precedes 525).

### 5.3 What is stored about the prime
Nothing: pi is not in `s`, not in the stream. It is a function of h_in at the 0.02-s plant steps,
which needs X, Y, z, theta, psi at every plant step (the stream stores them only every 12th step) and
the sea (stored).

### 5.4 Measured size (open-loop replays, plant-step level, 5760 steps each)

| mission | replay vs record | mean pi | P(pi < 0.999) | P(pi < 0.9) | P(pi < 0.5) | P(h_in < 0) |
|---|---|---|---|---|---|---|
| (0,0) | <= 1.3e-6 | 0.758 | 0.770 | 0.467 | 0.203 | 0.103 |
| (0,1) | <= 1.1e-7 | 0.864 | 0.659 | 0.290 | 0.097 | 0.045 |
| (3,1) | <= 1.2e-11 | 0.849 | 0.637 | 0.312 | 0.118 | 0.053 |
| (7,1) | exact | 0.865 | 0.572 | 0.256 | 0.111 | 0.055 |
| (4,0) | deviates after k = 359 | 0.787 | 0.763 | 0.412 | 0.172 | 0.092 |
| (2,0) | deviates after k = 389 | 0.792 | 0.721 | 0.407 | 0.172 | 0.085 |

min pi = 0.000 in all six. The prime multiplies both the jet's surge and side forces (section 6).

## 6. Jet forces and moments (state -> forces)

### 6.1 Pump (sim/waterjet.py:54-106)
JetPump(area A, power P_max, eta_p, zeta_n, zeta_i, w_in, rho) built at planing_vessel.py:123-125:
d_nozzle = 0.085 m, A = pi 0.085^2/4 = 0.0056745 m^2, P_max = 300 x 745.7 = 223 710 W,
eta_p = 0.775, zeta_n = 0.02, zeta_i = 0.25, w_in = 0.05 (defaults, waterjet.py:57-58).

* Inlet velocity: V_in(u) = (1 - w_in) max(u, 0) (waterjet.py:67-68).
* Power that gives net thrust T' >= 0 at the design speed (waterjet.py:98-106):

      V_in,r = (1 - w_in) U_ref = 12.2180 m/s
      V_r(T') = V_in,r/2 + sqrt(V_in,r^2/4 + T'/(rho A))
      P(T') = rho A V_r ((1+zeta_n) V_r^2 - (1-zeta_i) V_in,r^2) / (2 eta_p)

* Jet velocity at shaft power P and speed u: the largest root of
  (1+zeta_n) V^3 - (1-zeta_i) V_in(u)^2 V - 2 eta_p P / (rho A) = 0, by up to 8 Newton steps from
  V0 = (c/a)^(1/3) + sqrt(b/a) (waterjet.py:70-90). Substituting P(T'), rho A and eta_p cancel:

      (1+zeta_n) V_j^3 - (1-zeta_i)(1-w_in)^2 u_+^2 V_j = (1+zeta_n) V_r^3 - (1-zeta_i)(1-w_in)^2 U_ref^2 V_r ,
      i.e.  1.02 V_j^3 - 0.676875 u_+^2 V_j = 1.02 V_r(T')^3 - 0.676875 U_ref^2 V_r(T').

  So V_j = V_r(T') exactly when u = U_ref, and the same hydraulic power is delivered at any u.

### 6.2 Flow and forces (sim/waterjet.py:153-188), for thrust state T, nozzle delta, speed u

    if T = 0:          m_dot = 0
    else:              f(T) = min(|T| / T_db, 1),  T' = max(|T|, T_db)
                       V_j = V_j(P(T'), u)
                       m_dot = f(T) * pi * rho * A * V_j
    T >= 0:  F_x = m_dot (V_j cos delta - V_in(u)),   F_y = m_dot V_j sin delta
    T <  0:  F_x = -m_dot (k_rev V_j cos delta + V_in(u)),  F_y = -k_rev m_dot V_j sin delta,  k_rev = 0.6
             (never reached in the recordings: T >= 0 always, section 3)

Consequences (exact): at u = U_ref, delta = 0, pi = 1, T >= T_db: F_x = T. Steering costs surge thrust
through cos delta (cos 25 deg = 0.906); the side force depends on (T, u) through V_j and is
proportional to pi.

Numbers (pi = 1; check_plant.py):

| u (m/s) | T/T_max | V_j (m/s) | m_dot (kg/s) | F_x(delta=0) N | F_x(25 deg) N | F_y(25 deg) N | F_y/(T sin 25 deg) |
|---|---|---|---|---|---|---|---|
| 0 | 0.15 | 18.31 | 106.5 | 1951 | 1768 | 824 | 2.04 |
| 10 | 0.15 | 19.52 | 113.5 | 1137 | 930 | 937 | 2.32 |
| 12.86 | 0.15 | 20.30 | 118.1 | 955 | 730 | 1013 | 2.51 |
| 10 | 0.5 | 29.79 | 173.3 | 3515 | 3032 | 2181 | 1.62 |
| 12.86 | 0.5 | 30.29 | 176.2 | 3183 | 2683 | 2255 | 1.68 |
| 12.86 | 1.0 | 39.75 | 231.2 | 6366 | 5505 | 3884 | 1.44 |
| 17 | 1.0 | 40.45 | 235.3 | 5719 | 4827 | 4023 | 1.50 |

(The reduced model uses the constant k_jet_side = 1.61397 for the last column, section 8.1.)

### 6.3 Where the jet enters the equations of motion
Application point (x_n, 0, z_n) = (-lcg + 0, 0, -vcg + 0.15) = (-2.1812, 0, -0.40) m
(planing_vessel.py:119-120; hull `z_nozzle` 0.15, `x_nozzle` absent -> 0). Generalised load
`_load(F_x, F_y, 0, x_n, 0, z_n)` = [F_x, F_y, 0, -z_n F_y, z_n F_x, x_n F_y]
(planing_vessel.py:67-71, 347-348):

    surge force   F_x
    sway force    F_y
    roll moment   K_jet = -z_n F_y = +0.40 F_y
    pitch moment  M_jet = z_n F_x = -0.40 F_x      (negative = bow up under SP = -1)
    yaw moment    N_jet = x_n F_y = -2.1812 F_y

Rows (planing_vessel.py:356-361, Mtot from `_design_lateral` planing_vessel.py:371-383, values
printed): u_dot = (tau_0 + X_p)/1470 kg, v_dot = tau_1/1610.20 kg, p_dot = tau_3/844.08 kg m^2
(+ the gz righting-arm swap, planing_vessel_gz.py:223-239), r_dot = tau_5/2932.49 kg m^2; the pitch
moment enters the coupled heave-pitch solve with instantaneous added mass
(planing_vessel.py:349-353). Hence the jet's direct accelerations are F_x/1470, F_y/1610.2,
0.40 F_y/844.08 and -2.1812 F_y/2932.49 = -7.438e-4 F_y. (The hull, wave and Coriolis terms of the
same rows, tau_0 += m v r, tau_1 -= m u r at planing_vessel.py:334-335, are outside this document.)
Sign: delta > 0 gives F_y > 0 at the stern and N_jet < 0 (turns towards negative r).

### 6.4 Unused propulsion code
`sim/propeller.py` (Wageningen B) is not used: `Propulsion.series` is None (no `series` argument,
actuators.py:76-81). `Propulsion.speed_correction`, `effective`, `effective_units` and the propeller
ventilation curve (actuators.py:95-184) are never called by the planing plant (grep of
planing_vessel*.py); `Nozzle.ventilation_factor` returns 1 and is not called.

## 7. Initial condition and speeds

`initial_state(0.8 U_ref)` (planing_vessel.py:466-484), called in `make_mission`
(final_eval.py:221-222): u0 = 10.2889 m/s (20 kn); (z, theta) = calm running attitude at u0 by a
captive run (planing_vessel.py:386-422): z = 0.14870 m, theta = -0.029210 rad; thrust state T0 by
bisection so that F_x(T0, 0, u0) equals the calm resistance: T0 = 1706.86 N; delta0 = 0;
X = Y = 0; psi0 = phi_track (final_eval.py:222); v = w = p = q = r = 0, roll 0; prime 1.

Speed target of the controller: U_ref = 12.8611 m/s (25 kn). Recorded (16 missions): mean surge
10.55-11.46 m/s; mean speed made good along the track 20.48-22.24 kn (stored `kn`); surge range
3.98-14.49 m/s; mean thrust fraction 0.394-0.445; max |psi - phi_track| 13.5-26.7 deg; max |roll|
9.1-17.5 deg; final cross-track -16.0 to +13.2 m; no safe-mode step, no capsize, no non-finite step
(all n = 480).

## 8. The hand controller (what drove the recordings)

"Hand" = `theta(z_hand())` (studies/sim2real_jet.py:97-109): the MPPI cost weights equal to
`DEFAULT_WEIGHTS` and thrust floor 0 (z[7] = -1 -> floor = 0.5 (z7+1) 0.5 = 0). It is **not** a
fixed-gain law for thrust: thrust comes from a sampling MPC (8.1); the nozzle from a classical
heading autopilot (8.2). Only `theta`, `z_hand` are taken from sim2real_jet.py; its own task
(wigley10_jet, SS5) is not used.

### 8.1 Thrust: MPPI (control/mpc.py:74-205), built by Episode (sim/env.py:72-77) from Mission
(task.py:255-261)

Settings: K = 128 candidates (`EP_KW`, task.py:59), horizon H = 24 steps of Delta_t_m = 0.25 s
(Episode passes dt_ctrl = 0.25, env.py:65, 75; the plant step is 0.24 s), 7 knots at
knot_t = linspace(0, 23, 7) (mpc.py:104), sigma = (0.25, 0.25), temperature lambda = 0.6,
`use_rudder=False` (nozzle noise and nozzle plan forced to 0, mpc.py:172-173, 181-182),
`thrust_floor` = 0, RNG `default_rng(seed)` with the mission seed (mpc.py:103, env.py:76).
Initial plan: thrust knots 0.5, nozzle 0 (mpc.py:105-106).

Signals read at control step k: the reduced state of the current plant state in the track frame
`track_state()` (task.py:302-310: X, Y rotated by phi_track, psi - phi_track; u, v, z, w, theta,
q, r unchanged), the time t_k, and the true incident elevation at 15 points (5 stations
x_st = linspace(x_stern, x_bow, 5) = [-2.12495, -0.803075, 0.5188, 1.840675, 3.16255] m times
y_off = [-1.0126, 0, 1.0126] m, rotated by each candidate's predicted heading) at time t_k for the
first horizon step only: `PreviewProvider` with t_preview = 0, noise 0 returns zeros for
t_ahead > 0 (mpc.py:64-65; Mission does not pass t_preview, env.py:54). For leg 1 the points are
mapped back to the earth frame by `RotPreview` (task.py:193-204, 262-264).

Per call (mpc.py:169-186):

    noise_i ~ N(0, 0.25^2) on the 7 thrust knots (nozzle noise 0)
    cand_i = clip(nominal + noise_i, 0, 1) (thrust), 0 (nozzle)
    seq_i = linear interpolation of cand_i onto h = 0..23
    J_i = rollout cost (below)
    weights_i = exp(-(J_i - min J) / 0.6);  nominal = sum_i weights_i cand_i / sum_i weights_i
    command U_k^T = nominal[thrust, knot 0];  plan shifted by one step (mpc.py:188-202)
    thrust c_T,k = U_k^T T_max  (to_actuator, mpc.py:204-205)

Rollout model: `ReducedModel(p).step` (control/reduced.py:366-517) with the identified
parameters p of `studies/_cache/lofi_scarab195.json` plus `lofi.EXTRA` at neutral values
(lofi.py:146-150, 632), one step of 0.25 s per horizon step, nozzle 0, waves only at h = 0.
With the nozzle at 0 the jet side force of the model is k_jet_side max(c_T,0) sin 0 = 0
(reduced.py:448-449), so in every candidate r_{h+1} = r_h exp(-0.25/0.42) (reduced.py:482-485):
the plan cannot change the heading. Surge: u_{h+1} = u_h + 0.25 (c_T - 13.98718 u_h|u_h|)/1968.578
(reduced.py:394-395, 491). Heave and pitch: exact second-order transitions about
eta_bar + z0 and alpha + th0 (reduced.py:397-425) with omega_z = 5.83486, zeta_z = 0.372324,
omega_theta = 3.71347, zeta_theta = 0.155060, z0 = 0.153663 m, th0 = -0.00922914 rad.
Sway: v_{h+1} = v_h + 0.25 (-1679.94 v - 391.866 v|v| - 1400 u r)/1610.20 (reduced.py:452-453).

Cost per candidate (mpc.py:137-166), states after the step, weights
w = (w_acc, w_slam, w_speed, w_track, w_heading, w_dthrust, w_drudder) = (1, 12, 2, 1, 4, 0.3, 0.3)
(mpc.py:30-35):

    J = sum_{h=0}^{23} 0.25 [ 1.0 (a_bow,h / 9.81)^2 + 12 slam_h + 2 ((U_ref - u_{h+1}) / U_ref)^2
                              + 1.0 (y_{h+1} / 5.4)^2 + 4 (psi_{h+1} - 0)^2 ]
        + 0.3 sum_{h=0}^{22} (seq^T_{h+1} - seq^T_h)^2          (nozzle term identically 0)

    a_bow,h = [(w_{h+1} - w_h) + SP x_bow (q_{h+1} - q_h)] / 0.25,   x_bow = 3.16255 m
    rel_h = z_h + SP x_bow theta_h - eta_centre,bow(h)                   (step-start pose)
    out_h = rel_h > draft_bow = 0.215464 m
    slam_h = out_{h-1} and not out_h and (w_{h+1} < -v_slam), v_slam = 0.676884 m/s, out_{-1} = False

Targets: speed U_ref = 12.8611 m/s; cross-track y = 0; heading psi - phi_track = 0. Because the
heading cannot be changed by the plan, the heading term is common to all candidates and does not
affect the weights; the cross-track term depends on the thrust plan through
y_{h+1} = y_h + 0.25 (u_{h+1} sin psi_{h+1} + v_{h+1} cos psi_{h+1}) whenever psi != 0.

### 8.2 Nozzle: heading autopilot `Episode._steer` (sim/env.py:93-154), called by
`Mission.mpc_command` with the current plant state and the MPPI thrust command (task.py:300)

Signals read: psi = s[5], r = s[11] (plant, measured, no noise), the track angle phi_track
(heading_ref, task.py:261), the MPPI thrust command c_T,k (N, before safe mode), and its own
integrator I_psi.

    r_s = sqrt(L/10) = 0.734847
    omega_n = 0.30 (U_ref / L) / 0.45 = 1.587790 rad/s           (planing branch, env.py:127-128)
    zeta = 1.0;  T_N = max(tau_r, 0.5 r_s) = max(0.42, 0.367) = 0.42 s   (env.py:141)
    t_now = max(c_T,k, 0.15 T_max)                                 (env.py:136-137)
    K(t_now) = k_nomoto_f k_jet_side t_now = (-1.698831e-4)(1.613969) t_now = -2.741847e-4 t_now  (rad/s per rad)
    a = omega_n^2 T_N / K = -3862.0 / t_now                         (env.py:142)
    b = max(2 zeta omega_n T_N - 1, 0) / K = 0.333744 / K = -1217.2 / t_now   (env.py:148, planing)
    e_psi = wrap(psi - phi_track) in (-pi, pi]
    raw = -(a e_psi + b r + (0.12 / r_s) a I_psi),  0.12/r_s = 0.163299
    if |raw| < delta_max: I_psi <- clip(I_psi + e_psi * 0.25, -6 r_s, 6 r_s) = clip(., -4.40908, 4.40908)
    c_delta,k = clip(raw, -delta_max, delta_max), delta_max = 0.436332 rad

(raw uses I_psi before the update; the integrator advances 0.25 s per call while 0.24 s pass.)
Gain values: at t_now = 0.15 / 0.5 / 1.0 T_max, (a, b) = (-4.0444, -1.2748) / (-1.2133, -0.38243) /
(-0.60666, -0.19122). Since a, b < 0: c_delta = |a| e_psi + |b| r + 0.1633 |a| I_psi, a negative
feedback with the sign of section 6.3. The integrator I_psi starts at 0 (env.py:91) per mission.

### 8.3 Safe mode (learn/meta/mpc_constrained.py:620-642), dt = 0.24 s (onestep_gz.py:146)
If |phi| > 20 deg: thrust halved, nozzle 0, until |phi| < 10 deg for 2 s. In all 16 stored
missions `n_safe = 0`, so the stored commands equal the MPPI thrust and autopilot nozzle.

### 8.4 Determinism
Given (seed, leg) the whole loop is deterministic (fixed plant, seeded sea, seeded MPPI RNG);
confirmed by the bit-exact rerun (section 0).

## 9. Sea state of the 16 missions

Built by `Episode` (sim/env.py:68-69) from `Mission(..., track=LEGS[leg])`
(final_eval.py:209-210, task.py:243-261) with sea = None:
`SeaState(hs=1.0, tp=5.0, theta0=pi, n_freq=24, n_dir=5, seed=seed)`; gamma = 3.3, spreading
s = 8, band (0.5, 3.0) omega_p, jitter False (defaults, step0_preview_spec.py:59-61).

* Elevation (step0_preview_spec.py:143-148; the plant's own sum planing_vessel.py:205-231 gives
  eta, eta_t, eta_tt, eta_X, eta_Y from the same components):

      eta(X, Y, t) = sum_{i=1}^{120} a_i cos(k_i (X cos th_i + Y sin th_i) - omega_i t + phi_i),  k_i = omega_i^2 / g

  (deep water, g = 9.81). th is the direction the waves travel TOWARD (step0_preview_spec.py:55-56).
* Frequencies: omega_p = 2 pi / 5 = 1.256637 rad/s; 24 equally spaced omega in
  [0.628319, 3.769911] rad/s, Delta_omega = 0.136591 rad/s.
* Directions: 5 equally spaced th in [pi/2, 3pi/2]: {pi/2, 3pi/4, pi, 5pi/4, 3pi/2}, Delta_th = pi/4.
* Amplitudes (step0_preview_spec.py:81-85): a_ij = sqrt(2 S(omega_i) D(th_j) Delta_omega Delta_th) with
  S = JONSWAP (sigma 0.07 / 0.09, gamma 3.3) rescaled so that its trapezoid integral over the 24
  frequencies is (Hs/4)^2 (step0_preview_spec.py:30-39), and D = cos^{16}((th - pi)/2) rescaled to a
  trapezoid integral of 1 over the 5 directions (lines 42-46). Realised 4 sqrt(sum a^2/2) = 1.0016 m.
* Phases phi_i ~ U[0, 2 pi) i.i.d. from `np.random.default_rng(seed)` (line 86, 91).

What differs between missions: only the seed (0..7: wave phases, and the MPPI noise stream) and the
leg (0: track phi_track = 0, the boat heads +X into waves travelling toward -X = head seas;
1: phi_track = pi/4, mean wave direction 45 deg off the bow on the negative-body-y side; task.py:66-69).
Spectrum, Hs, Tp, spreading, mean direction theta0 = pi and the component grid are the same in all
16. For a given seed, legs 0 and 1 have identical a, k, th, omega, phi (verified on the stored
`sea` dicts). The line spectrum repeats at a fixed point every 2 pi / Delta_omega = 46.0 s
(not at the moving hull). Encounter frequency of a component met head-on at U = 11 m/s:
omega_e = omega + omega^2 U / g, from 1.07 rad/s (omega = 0.628) to 19.7 rad/s (omega = 3.77),
above the Nyquist frequency pi / 0.24 = 13.1 rad/s of the control-step sampling.

## 10. The streams (what is stored, at what rate)

File `studies/_cache/onestep_gz/streams_T120.pkl` (`phase_record`, onestep_gz.py:176-190),
dict keyed by (seed, leg) for seeds 0..7 x legs 0, 1 (`--train 0-3 --held 4-7`, onestep_gz.py:855-856,
913-916). Each value (`record`, onestep_gz.py:165-169):

| key | shape | content | rate / time |
|---|---|---|---|
| `xs` | (481, 14) float64 | full plant state `s` (section 1, all 14 entries incl. thrust state T and nozzle angle delta) at t_k, k = 0..480 | every control step, 0.24 s |
| `t` | (481,) | t_k = sum of 0.02 increments (= 0.24 k up to rounding) | same |
| `u` | (480, 2) | applied commands held over [t_k, t_k+1): (c_T,k / T_max, c_delta,k / delta_max), after safe mode | per control step |
| `apk` | (480,) | APK_k = max_{j=1..6} (w(2j) - w(2j-2)) / 0.04, w(i) = heave rate s[8] after plant step i of control step k, w(0) = s0[8] (final_eval.py:377-388) - signed maximum, m/s^2 (stored range -9.8 to 99.3; -9.8 = free fall) | per control step, from 0.04-s differences |
| `hmin` | (480,) | HMIN_k = min_{j=1..6} [F_b + z + SP x_b theta - max_{3 bow points} eta] at the states after plant steps 2j, F_b = plant.freeboard = 0.884536 m, x_b = 3.16255 m, bow points (x_b, y_off) (final_eval.py:343-367, 385-387), m | per control step |
| `sea` | dict of 5 arrays (120,) | a, k, th, w (=omega), phi of the mission's SeaState | constant |
| `consts` | dict | u_ref 12.8611, L 5.4, t_max 6365.663, rud_max 0.436332, z0 0.153663, th0 -0.00922914 (mpc_learned.py:515-521) | constant |
| `n`, `n_ctrl` | int | 480, 480 | |
| `n_safe` | int | 0 in all 16 | |
| `kn` | float | mean speed made good along the track / 0.514444 (task.py:312-316, 372-382) | |
| `seed`, `leg` | int | | |

Not stored: the 0.02-s plant-step states (the Recorder and the gz plant log keep them in memory
only), the prime pi (section 5.3), the delay-line contents, the MPPI plan and RNG state, the
autopilot integrator I_psi, the plant's CG / bow accelerations, the substep counts.

Delay-line contents at a control-step boundary are recoverable: since 5 < 12, both lines hold five
copies of the previous command u_{k-1} at t_k (u_0 at k = 0).

How the stream is consumed (`full_live`, onestep_gz.py:204-221 -> `LiveDataW`,
mpc_constrained.py:138-208 -> `LiveData`, mpc_learned.py:545-596): per step the 11 state inputs
`inputs11(xs_k)`, the command u_k, the error target
e_k = e0_from(env, xs_k, xs_{k+1}, u_k) (learn/meta/data3.py:78-88: (reduced next state minus
model0's next state)/0.24 s for the channels (u, v, r, w, q, z, theta, psi), plus
(actuator position after the step - command)/0.24 s for thrust and nozzle; model0 = the reduced
model, no waves, ideal actuators, 6 substeps of 0.04 s, data3.py:66-75, env from relabel._env,
relabel.py:232-242: dt 0.04, sub 6, same x_st, y_off), the elevations W15 at the 15 stations at t_k
and WMID15 at the dead-reckoned mid-step pose at t_k + 0.12 s, recomputed from `sea`
(mpc_constrained.py:179-195), and Y_k = (APK_k, HMIN_k).

## 11. Facts relevant to an "is the target in the family" comparison (stated, not concluded)

These are exact properties of the target's propulsion and loop as coded; whether the prior
catalogue D10 (learn/meta/operators_cat.py, cat_*.py, operators_rb.py, operators_gen.py) contains
terms of each form was NOT checked here (UNKNOWN for this document).

1. Thrust actuator: affine control-step map of (T_k, D(c_T,k-1), D(c_T,k)) with fixed coefficients
   (section 3); rate limit and saturation inactive; dead band at 0.004167 of range.
2. Nozzle actuator: delay 5 plant steps (0.10 s), lag 0.1 s, slew limit 40 deg/s active at 6-10 % of
   plant steps, stops at +-25 deg; control-step map piecewise in (delta_k, c_delta,k-1, c_delta,k).
3. Within a control step the delayed command switches after 5 of 12 plant steps, so the force
   history inside step k depends on both u_{k-1} and u_k, not on u_k alone.
4. Jet surge force F_x = pi f(T) rho A V_j(T, u) (V_j cos delta - 0.95 u) and side force
   F_y = pi f(T) rho A V_j(T, u)^2 sin delta, V_j from the power-balance cubic (section 6.1). The
   reduced model (MPPI and model0) uses instead: surge force = command c_T (no delay, no lag, no u or
   delta dependence; reduced.py:394-395), side force = 1.61397 max(c_T, 0) sin(delta) (reduced.py:448-449).
   The plant ratio F_y/(T sin delta) ranges 1.44-2.51 over the table of section 6.2 at pi = 1.
5. The pump prime pi is a hidden state with asymmetric first-order memory (0.05 s loss, 0.5 s
   recovery) driven by the intake submergence at 0.02-s resolution; it scales both F_x and F_y;
   in the replays its mean is 0.76-0.87 and it is below 0.5 at 10-20 % of plant steps.
6. The jet side force also produces a roll moment +0.40 F_y (section 6.3).
7. The controller's thrust is a deterministic but non-smooth function (sampling MPC with a seeded
   noise stream and warm-started plan) of the full state history and the true current wave field;
   the nozzle is a PI-D-type function of (psi, r, I_psi, c_T). The recorded inputs are therefore
   closed-loop (correlated with the state and the waves).

## 12. Docstring / code disagreements (code followed)

1. `Propulsion` docstring (actuators.py:48-49): "dead band, saturation, transport delay, rate limit
   and first-order lag, in that order"; code order (actuators.py:190-196): delay, dead band,
   saturation, then a lag whose derivative is rate-clipped. Delay commutes with the memoryless dead
   band / saturation, so the output is the same.
2. `planing_vessel_gz.py:62-64`: roll "nothing but the transverse wave slope drives it"; code: the
   jet's roll moment -z_n F_y = +0.40 F_y is added to the roll row (planing_vessel.py:347-348, 360).
3. `Mission` docstring (task.py:209): "stepped one control step (0.25 s)"; actual 0.24 s
   (env.py:80, sub = 12); the autopilot integrator and the MPPI rollouts also use 0.25 s.
4. `task.WAVES_FROM = pi` (task.py:68): in the wave field's convention theta0 is the direction of
   travel (step0_preview_spec.py:55-56); waves travel toward -X, i.e. come from +X; leg 0 (heading 0)
   is head seas, consistent with the task text.
5. `sim/waterjet.build()` defaults (27 deg nozzle, tau 0.15 r_s, delays 0.10 r_s, jet sized at ratio
   0.35) are not the target's; the planing plant builds its own Waterjet / Nozzle
   (planing_vessel.py:118-138).
6. `lofi.act_target_like` (lofi.py:241-256) describes the target's delays as 3 steps of 0.04 s =
   0.12 s for the source world; the target itself has exactly 0.10 s (5 x 0.02 s) on both channels
   (as lofi.py:230-238 states).

## 13. UNKNOWN

* Whether the D10 family contains terms of the forms in section 11: not examined here (outside this
  task's files).
* `lofi.ACT_FAMILY` / `act_step` (lofi.py:158-342) semantics of delay, lag and rate in the source
  world were not read in full; no statement about their equality with section 3/4 is made.
* The prime statistics of section 5.4 cover 6 of 16 missions (two of them only up to control
  steps 359 / 389 exactly); the other 10 were not replayed (time budget).
* The exact moment the 16 streams were recorded relative to the untracked edits of
  studies/onestep_gz.py is not recorded in git; the bit-exact closed-loop rerun of mission (0,0)
  (all steps) and (7,1) (120 steps) is the evidence that the current code produced them.
