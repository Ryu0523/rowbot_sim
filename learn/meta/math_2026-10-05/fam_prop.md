# The D10 catalogue's propulsion chain (P1-P14) against the high-fidelity waterjet

Scope: `learn/meta/cat_prop.py` (P1-P14), the chain driver `learn/meta/cat_base.py`
(`CatDraw._prop`, `PROP_ORDER`, overlap switches), the layer-1 jet term T5 it switches
(`learn/meta/operators_rb.py`), the low-fidelity jet law (`control/reduced.py`), and the
high-fidelity jet (`sim/waterjet.py`, `sim/actuators.py`, used by `sim/planing_vessel.py`,
unchanged by `sim/planing_vessel_gz.py`). Every equation cites file:line of the code; where
a docstring and the code differ, the code is followed and the difference is listed in
section 7. Numbers in section 6 come from the two scripts named in section 8 (each ran
< 1 min, one process).

Hull: `scarab195` (`hydro/hulls.py:146-151`). Its reduced (low-fidelity) parameters are the
identified cache `studies/_cache/lofi_scarab195.json` (`operators_rb.default_params`,
`operators_rb.py:126-132`).

---------------------------------------------------------------------------------------

## 1. Symbols

Frames. Body axes from the centre of gravity (CG): x forward, y to port, z up. Pitch angle
theta follows the project's sign `s_p = sign_pitch = -1`: the height of a hull point at x is
`z + s_p x theta`, so bow-up trim is theta < 0 (`sim/planing_vessel.py:28-29`,
`cat_base.py:301-305`). Reduced state `sr` = (x, y, u, z, zdot, theta, thetadot, psi, r, v)
(`control/reduced.py:369`). The 5 output channels are (u, v, r, zdot, thetadot) =
`RED_VEL = (2, 9, 8, 4, 6)` (`operators_rb.py:92`).

### 1.1 High-fidelity jet (symbols with superscript hf)

| symbol | meaning | unit | kind | value (Scarab) / source |
|---|---|---|---|---|
| c | thrust STATE of the drive, plant state s[12]; means "net thrust at u_ref" | N | actuator state | `planing_vessel.py:252` |
| c_cmd | thrust command | N | input | |
| delta | nozzle angle STATE, s[13] | rad | actuator state | |
| delta_cmd | nozzle command | rad | input | |
| u | body surge speed s[6] (no current in this plant) | m/s | state | `planing_vessel.py:254,328` |
| h | water depth over the intake (negative when out) | m | derived from state + waves | `planing_vessel.py:486-495` |
| pi | pump prime (fraction of flow) | - | hidden state | `waterjet.py:150,208-217` |
| rho | water density 1025 | kg/m^3 | fixed | `planing_vessel.py:85` (hull rho) |
| A | nozzle area pi 0.085^2/4 = 5.6745e-3 | m^2 | fixed | `planing_vessel.py:123`, `hulls.py:149` |
| P_max | shaft power 300 x 745.7 = 223 710 | W | fixed | `hulls.py:149` |
| eta_p | pump efficiency 0.775 | - | fixed | `hulls.py:149` |
| zeta_n, zeta_i | nozzle / inlet loss 0.02 / 0.25 | - | fixed (JetPump defaults) | `waterjet.py:57` |
| w | inlet wake fraction 0.05 | - | fixed | `waterjet.py:57` |
| u_ref | design speed 25 kn = 12.8611 | m/s | fixed | `hulls.py:148` |
| t_max | P_max-thrust at u_ref = 6365.66 | N | fixed | `planing_vessel.py:127` |
| c_db | dead band t_max 50/12000 = 26.52 | N | fixed | `planing_vessel.py:132` |
| tau_c, R_c, d_c | thrust lag 0.6 s, rate t_max/0.3 = 21 219 N/s, delay 0.1 s | | fixed | `planing_vessel.py:131-133`, `hulls.py:151` |
| tau_d, R_d, d_d, delta_max | nozzle lag 0.1 s, rate 40 deg/s, delay 0.1 s, travel 25 deg | | fixed | `planing_vessel.py:134-138` |
| h_b | prime band 0.5 x 0.085 = 0.0425 | m | fixed | `waterjet.py:148-149` |
| tau_pi_down, tau_pi_up | prime loss 0.05 s, recovery 0.5 s | s | fixed | `planing_vessel.py:130` |
| (x_n, z_n) | nozzle point (-2.1812, -0.4000) | m | fixed | `planing_vessel.py:119-120` |
| (x_i, z_i) | intake point (-1.3812, -0.5500) (= keel) | m | fixed | `planing_vessel.py:121-122` |
| dt_p | plant step 0.02 | s | fixed | `hydro/planing.py:134` |
| M_hf | rigid + added inertia diag: surge 1470, sway 1610.2, yaw 2932.5 kg(m^2) | | fixed | `planing_vessel.py:378-382` (printed) |

### 1.2 Catalogue chain (symbols without superscript)

| symbol | meaning | unit | kind | source |
|---|---|---|---|---|
| c_c, delta_c | physical commands (q['cmd']) | N, rad | input | `cat_base.py:1120-1122` |
| c_a, delta_a | low-fidelity boat's APPLIED actuator values over the substep (after the episode's M10 chain), q['thr'], q['noz'] | N, rad | input | `cat_base.py:1123-1124`, `operators_cat.py:509-525` |
| c_n | clip(c_a / t_max, 0, 1) (`_cn`) | - | derived | `cat_prop.py:86-88` |
| delta_e | effective nozzle angle after P7, P11 (`noz_eff`) | rad | derived | `cat_base.py:1134` |
| u_r | water-relative surge u - nu_c,x (current from E1) | m/s | derived | `cat_base.py:570-572` |
| s | u_r / u_id | - | derived | `cat_prop.py:370` |
| h_in | intake immersion eta_in - (z + s_p x_i theta + y_i phi) - z_i | m | derived | `cat_base.py:566-568` |
| trim_rel | s_p (theta - th0), bow up positive | rad | derived | `cat_base.py:562` |
| t_max | 6365.66 (lofi `t_max`) | N | fixed | `cat_base.py:375` |
| u_id = u_des | 12.8611 (lofi `u_design`) | m/s | fixed | `cat_base.py:372` |
| delta_max = rud_max = delta_stall | 0.43633 | rad | fixed | `cat_base.py:375-376` |
| k_js | lofi `k_jet_side` = 1.6140 | - | fixed | `cat_base.py:377` |
| h_run | calm-running intake immersion, 0.2199 m | m | fixed | `cat_base.py:389-390` (value printed, section 8) |
| (x_j, z_j), k_c | jet point and axial-loss share, from layer-1 T5 if drawn | m, - | random (layer 1) | `cat_base.py:398,927-930` |
| M_t | layer-3 generalised inertia (5x5) | | random (layer 3) | `operators_gen.py:283-315` |
| u_lo | 16 kn = 8.231 m/s | m/s | fixed | `cat_base.py:374`, `operators_rb.py:98` |
| dt | substep 0.04 | s | fixed | `cat_base.py:76` |

LogU[a, b] = log-uniform, U[a, b] = uniform (`learn/meta/operators._logu`).

---------------------------------------------------------------------------------------

## 2. The high-fidelity jet, written out

### 2.1 Actuators (`sim/actuators.py`, `sim/waterjet.py`)
Called once per plant step AFTER the derivative (`planing_vessel.py:552-553`); the forces of
step k use the start-of-step states c_k, delta_k (the derivative reads s[12], s[13],
`planing_vessel.py:252`; ds[12] = ds[13] = 0, `planing_vessel.py:362-366`).

Thrust (`actuators.py:186-196`):
```
c_d    = c_cmd delayed by n = round(d_c / dt_p) = 5 steps        (Delay, actuators.py:29-42)
c_d    = 0 if |c_d| < c_db
c_d    = clip(c_d, -t_max/3, t_max)
c_{k+1} = c_k + dt_p * clip((c_d - c_k)/tau_c, -R_c, R_c)       (explicit Euler)
```
Nozzle (`waterjet.py:237-240`):
```
delta_d     = clip(delta_cmd delayed 5 steps, -delta_max, delta_max)
delta_{k+1} = delta_k + dt_p * clip((delta_d - delta_k)/tau_d, -R_d, R_d)
```
### 2.2 Pump: jet velocity as a function of (c, u) (`waterjet.py:67-106,153-173`)
```
V_in(u)  = (1 - w) max(u, 0)                                         waterjet.py:67-68
c_bar    = max(c, c_db),  kappa_db(c) = min(c / c_db, 1)   (c > 0)    waterjet.py:166-169
V_r(c)   = V_in(u_ref)/2 + sqrt(V_in(u_ref)^2/4 + c_bar/(rho A))      waterjet.py:100-103
P(c)     = 0.5 rho A V_r ((1+zeta_n) V_r^2 - (1-zeta_i) V_in(u_ref)^2) / eta_p   waterjet.py:104-106
V_j(c,u) = largest root of (1+zeta_n) V^3 - (1-zeta_i) V_in(u)^2 V = 2 eta_p P(c)/(rho A)   waterjet.py:70-90
```
Substituting P(c): `Phi(V_j; V_in(u)) = Phi(V_r(c); V_in(u_ref))`, with
`Phi(V; V_i) = (1+zeta_n) V^3 - (1-zeta_i) V_i^2 V`. eta_p cancels (it only sets t_max through
P_max). By construction T_net(c, u_ref) = c for c >= c_db (checked numerically: difference 0
to 1e-12 t_max, section 8).

### 2.3 Forces and moments (`waterjet.py:175-188`, `planing_vessel.py:67-71,326-348`)
For c >= 0 (all recorded rows have c >= 0, section 6.0):
```
m_dot = kappa_db(c) * pi * rho A V_j                       waterjet.py:172
F_x^hf = m_dot (V_j cos delta - V_in)                      waterjet.py:185-186
F_y^hf = m_dot V_j sin delta
```
Define the (prime-free) net thrust T_net = kappa_db rho A V_j (V_j - V_in), gross momentum flux
G = kappa_db rho A V_j^2 and inlet ratio mu = V_in / V_j. Then exactly
```
F_x^hf = pi [ T_net - G (1 - cos delta) ]
F_y^hf = pi G sin delta,      G = T_net / (1 - mu) = T_net + kappa_db rho A V_j V_in
```
applied at (x_n, 0, z_n) through `_load(fx, fy, 0, x_n, 0, z_n)` (`planing_vessel.py:347`):
surge F_x, sway F_y, heave 0, roll -z_n F_y, pitch (plant convention) z_n F_x, yaw x_n F_y.
Surge, sway, yaw are divided by the diagonal M_hf (`planing_vessel.py:357-361`): 1470 kg,
1610.2 kg, 2932.5 kg m^2. The jet does not read v, r or the wave orbital velocity.
For c < 0 a bucket law with k_rev = 0.6 applies (`waterjet.py:187-188`); not reached.

### 2.4 Prime (`waterjet.py:208-217`, called at the START of each step, `planing_vessel.py:524`)
```
pi* = clip(h / h_b, 0, 1)
pi_{k+1} = pi_k + (pi* - pi_k)(1 - exp(-dt_p / tau)),  tau = tau_pi_down if pi* < pi_k else tau_pi_up
```
h is the intake depth (`planing_vessel.py:486-495`): surface at the intake minus
(Z_rest + z + z_i + s_p x_i (Th_rest + theta) + y_i phi), y_i = 0. pi multiplies BOTH F_x
(including its ram-drag part) and F_y linearly. No latch, no dwell, no engine-speed state.
`planing_vessel_gz.py` changes only the roll row (`planing_vessel_gz.py:223-239`) and records
h (`:241-244`); the jet is the parent's.

---------------------------------------------------------------------------------------

## 3. The low-fidelity jet law and the error target

Low-fidelity (MPC) model, applied actuators c_a, delta_a (`control/reduced.py`):
```
surge:  udot = (c_a - k_drag u|u|) / m_surge                               reduced.py:394-395
sway:   lift = k_js max(c_a, 0) sin(clip(delta_a, +-delta_stall))          reduced.py:443-449
        vdot = (lift - k_lin_sway v - k_sway v|v| - m_coriolis u r) / m_sway  reduced.py:452-453
yaw:    r <- r_t + (r - r_t) exp(-dt/tau_r),  r_t = k_nomoto_f lift        reduced.py:482-485
```
No axial loss cos delta, no speed dependence of thrust, no pitch moment from the jet.
Identified values (`lofi_scarab195.json`): m_surge 1968.58, k_drag 13.987, k_js 1.6140 (from a
steady turn at c = 7000 t_max/12000, delta = 15 deg, `reduced.py:271-280`), k_nomoto_f
-1.6988e-4, tau_r 0.42.

Error target. The network's one-step target is
`e0 = (plant state at k+1 - model0_step(plant state at k, command U)) / dtc`
(`learn/meta/data3.py:78-88`), where `model0_step` runs the reduced model with the COMMAND,
ideal actuators and NO waves (`data3.py:66-75`); the live path uses the same function
(`mpc_learned.py:579-591`). The network's actuator tokens are the plant's actuator STATES
xs[12:14] / amax (`mpc_constrained.py:240`). Hence the high-fidelity actuator dynamics
(section 2.1) and the jet force law (2.2-2.4) both enter e0. In the catalogue world the first
is carried by the episode's M10 actuator on the low-fidelity plant (`operators_cat.py:509-525`;
drawn by `lofi.draw_act`, `learn/meta/data2.py:155-157`), the second by the propulsion chain,
which reads the applied value c_a (section 4).

---------------------------------------------------------------------------------------

## 4. The chain framework (`cat_base.CatDraw._prop`, `cat_base.py:1110-1189`)

### 4.1 Order and chain dictionary
`PROP_ORDER = (P12, M10, P7, P11, P3, P10, P1, P13, P2, P9, P4, P5, P8, P14, P6)`
(`cat_base.py:134-137`). Initial chain dict pc (`cat_base.py:1122-1127`):
c_thr = c_c, c_noz = delta_c, thr_act = c_a, noz_act = delta_a, T = None, rho = 1, p = 1,
nu_fac = 1, pulse = 0, F_z = 0, dF = (0, 0). Steps:

1. P12 may rewrite c_thr; then the M10 slot: if the command changed, thr_act/noz_act are
   recomputed by a copy of the episode's actuator (`_true_actuator`, `cat_base.py:1167-1189`;
   ideal copy when act is None); in every case `noz_eff = noz_act` (`cat_base.py:1131-1135`).
2. P7, P11 rewrite noz_eff = delta_e.
3. Thrust factors P3 (writes T), P10 (rho), P1 (p, load), P13 (p, load), P2 (nu_fac), P9 (pulse).
4. At the P4 slot (whether or not P4 is on): `T_pre = T x rho x p x nu_fac x (1 + pulse)`
   with T = thr_act if P3 is off (`cat_base.py:1136-1137,1161-1165`). P4 (if on) writes T_eff.
5. At the P5 slot: T_eff = T_pre if unset (`cat_base.py:1138-1142`). P5 (if on) writes F_y;
   if P5 is off, the low-fidelity law `F_y = k_js max(T_eff, 0) sin(clip(delta_e, +-delta_stall))`
   (`cat_base.py:1147-1149`).
6. P8 (multiplies F_x, F_y), P14 (adds dF), P6 (adds F_z and dF_x).
7. Final force and mapping (`cat_base.py:1150-1159`):
```
ax(d)  = 1 + k_c (cos d - 1)
F_x    = T_eff ax(delta_e)              (or P8's value)
F_x0   = c_a ax(delta_a),   F_y0 = k_js max(c_a,0) sin(clip(delta_a, +-delta_stall))
DeltaF = (F_x - F_x0 + dF_x,  F_y - F_y0 + dF_y,  F_z)
Q      = lever_Q((x_j, 0, z_j), DeltaF, theta)
       = (DF_x, DF_y, x_j DF_y, F_z + s_p theta DF_x, s_p (x_j F_z - z_j DF_x))   cat_base.py:298-311
a_prop = M_t^-1 Q                                                               cat_base.py:314-316
```
a_prop is NOT soft-saturated (the chain bypasses `item_accel`, `cat_base.py:1083-1088` vs
`:724-740`); it is clipped later together with everything else (`clip_push`,
`operators_cat.py:542`).

### 4.2 Multiplicative structure (all items on, before P4)
```
T_pre = T_P3(c_n, u_r) * rho_P10(t) * p_P1(h) * p_P13(slams) * nu_fac_P2 * (1 + pulse_P9)
T_eff = S_P4(T_pre; u_r, rho)                         (identity if P4 off)
F_x   = T_eff (1 + k_c (cos delta_e - 1)) * f_P8
F_y   = [k_js T_eff^+ psi(u_r) (p rho)^(kappa-1) + k_i t_max iota(c_n) (p rho)^kappa] h(delta_e) * f_P8
```
with p = p_P1 p_P13 and kappa = (kappa_p, kappa_rho) applied per factor (section 5, P5).
So with P5 on, F_y scales as p^kappa_p rho^kappa_rho (the T_eff factor cancels one power);
P9's pulse and P4's saturation reach F_y only through T_eff.

### 4.3 Which layer-1 terms are replaced, and what stays
Layer-1 T5 (drawn w.p. P_T5 = 0.7 if layer 1 is not off, P_RB_OFF = 0.05;
`operators_rb.py:107,110,432-445`):
```
F_x^T5 = exp(g(xi; c_eT)) c_a (1 + k_c (cos delta_a - 1))          operators_rb.py:836-837
F_y^T5 = exp(g(xi; c_eS)) lift0 (1 + c3 delta_a^2),  lift0 = k_js c_a^+ sin delta_a   :835,838
a_u = (F_x^T5 - c_a)/m_u,  a_v = (F_y^T5 - lift0)/m_v                                  :839-840
a_r = x_j F_y^T5 / I_r - k_nf_tau lift0 - (f_r - 1) r / tau_r                          :841-842
a_q = z_j (F_x^T5 - k_drag u|u|) / I_q                                                 :843
g(xi; c) = c0 + c1 xi + c2 (1.5 xi^2 - 0.5), xi = clip((u_r - u_mid)/u_half, -1, 1)    :159-160,784
```
Priors: c_eT0 ~ U[log 0.8, log 1.2], c_eS0 ~ U[log 0.7, log 1.3], each speed-constant w.p. 0.4
else c1 ~ N(0, (0.1 rho)^2), c2 ~ N(0, (0.1 rho^2)^2), rho ~ U[0.3, 0.8] (`_smooth`,
`operators_rb.py:148-157,435-438`); c3 = U[-1, 1]/rud_max^2 w.p. 0.3 (`:439-440`);
x_j = x_stern + U[-0.1, 0.1] L, z_j ~ U[-1.5 T, 0] (T = 0.3846 m), k_c ~ U[0, 1]
(`:441-445`, `KC` `:116`). Off-values: k_c = 0, x_j = x_stern = -2.1250, z_j = 0 (`:366-368`).

Switches (`cat_base.py:115-123`, applied `:771-800`):
- P3 on -> `T5_etaT_off`: c_eT := 0, i.e. eta_T(u) = exp(g) := 1 (`cat_base.py:786-787`).
- P5 on -> `T5_etaS_off`: c_eS := 0 and c3 := 0 (`cat_base.py:788-790`).

What stays of T5 when both are switched: a_u = c_a k_c (cos delta_a - 1)/m_u, a_v = 0, the
structural yaw part a_r = x_j lift0/I_r - k_nf_tau lift0 - (f_r - 1) r/tau_r (always on), and
the pitch part a_q = z_j (c_a(1 + k_c(cos delta_a - 1)) - k_drag u|u|)/I_q. The chain takes
x_j, z_j, k_c from T5 (`cat_base.py:927-930`) and subtracts F_x0 = c_a ax(delta_a), so
T5 surge + chain surge = (F_x - c_a)/m_u when M_t's surge row equals m_u (it does on the
diagonal: M_t[0,0] = m_u = 1968.58 for every draw, `operators_gen.py:289,303,308`, sampled
in section 8; the off-diagonal m z_g couples surge with pitch).
The layer-1 T5 inertias (I_r = m_v rho_r^2, rho_r ~ LogU[0.22, 0.37] L, `operators_rb.py:383-384`;
I_q `:387-388`) are NOT the M_t used for the chain's DeltaF (section 6.6).

When P3 (or P5) is not on - possible only in sparse episodes - eta_T (eta_S, c3) stay on and the
chain uses T = c_a (or the low-fidelity side-force law).

### 4.4 On-probabilities as coded
`P_ON` (`cat_base.py:92-105`), dependencies `DEPENDS = {P2: P1, P13: P1}` (`:106`, enforced
`:962-963`), sparse episodes w.p. 0.15 with 1-3 items uniformly from all non-structural codes
(38 codes; a chosen dependent brings its parent) (`:145,944-951,964-966`). Non-sparse values:

| item | P_ON coded | effective (non-sparse) |
|---|---|---|
| P1 | 0.7 | 0.7 |
| P2 | 1.0 | 0.7 (only with P1) |
| P3 | 1.0 | 1.0 |
| P4 | 0.3 | 0.3 |
| P5 | 1.0 | 1.0 |
| P6 | 1 - 0.5 x 0.4 = 0.8 | 0.8 (parts a 0.5, b 0.6, conditional on >=1) |
| P7 | 1 - 0.7 x 0.7 x 0.8 = 0.608 | 0.608 |
| P8 | 0.02 | 0.02 |
| P9 | 1.0 | 1.0 |
| P10 | 0.075 | 0.075 |
| P11 | 0.03 | 0.03 |
| P12 | 0.8 (1 - 0.97 x 0.8 x 0.95 x 0.98) = 0.2220 | 0.2220 |
| P13 | 0.3 | 0.21 (only with P1) |
| P14 | 0.5 | 0.5 |

Overall P(P3 on) = 0.85 + 0.15 x (2/38) = 0.858 (mean sparse count 2 over 38 codes); the same
for P5. The data generator builds `OperatorCat(op_seed, lib, dt, L, relay, act)` with default
items / sparse / tier (`learn/meta/episode7.py:80`), so these probabilities apply.

---------------------------------------------------------------------------------------

## 5. The fourteen items

Notation: "ch" = channels reached through lever_Q (u, v, r, zdot, thetadot); every item acts only
through DeltaF at (x_j, 0, z_j), so a thrust change reaches u, zdot (via s_p theta DF_x) and
thetadot (via -s_p z_j DF_x); a side-force change reaches v, r; F_z reaches zdot, thetadot.
Exact exponential = `exp_update` (`cat_base.py:180-184`); asymmetric = `exp_update_asym`
(`:187-191`, tau_up when the target is above the state).

**P12 - factory controller rules** (`cat_prop.py:732-795`). Inputs: c_c/t_max, delta_c, u (over
ground, sr[:,2], `:781`), episode clock. Rules, each on w.p. (assist 0.2, slim 0.05, limp 0.03,
heat 0.02) conditional on >= 1 (`_parts`, `:99-105,750-756`):
assist: if c < c_th and |delta_c| > d_th for t_d, c := max(c, c_a) held t_h (`:773-779`);
slim: c := c clip((u_cap - u)/du, 0, 1) (`:780-782`); limp: c := min(c, c_cap) for t >= t_f
(`:783-784`); heat: T_h' = (clip(c,0,1) - T_h)/tau_h (exact), c := min(c, 1 - k_h clip((T_h -
T_on)/(1 - T_on), 0, 1)) (`:785-790`). Changed command -> true actuator copy (`:791-794`).
Priors (`:757-767`): c_th U[0.03, 0.2], d_th U[0.5, 0.95] rud_max, t_d U[0, 0.5] s, c_a(rule)
U[0.1, 0.3], t_h U[0.5, 3] s, u_cap U[0.6, 1] u_max, du U[0.5, 2] m/s, c_cap U[0.3, 0.7],
t_f U[0, 90] s, tau_h LogU[60, 600] s, T_on U[0.5, 0.9], k_h U[0.3, 0.7], T_h(0) U[0.2, 1].
Ch: all (through the changed actuator).

**M10 slot** - not an item: the true-actuator copy (`cat_base.py:1167-1189`, `lofi.act_step`).

**P7 - steering load effects** (`cat_prop.py:518-561`). g = c_n (`:544`). Parts (rate 0.3,
compliance 0.3, backlash 0.2, conditional on >= 1, `:533`): extra rate limit
|Delta delta| <= rate0/(1 + lam_r g) dt (`:546-550`); backlash half-width 0.5 db0 lam_b g
(`:551-557`); compliance delta := delta - k_c7 g sin delta (`:558-559`). Priors (`:534-538`):
lam_r, lam_b U[0, 1], rate0 LogU[0.25, 2]/rs rad/s (rs = sqrt(L/10) = 0.7348), k_c7 U[0, 0.1]
rud_max, db0 U[0, 0.03] rud_max. Ch: v, r (and u via cos in ax, P5's h).

**P11 - nozzle faults** (`cat_prop.py:686-728`), from t_f ~ U[0, 90] s, one mode: stuck (0.4),
hard over side x rud_max (0.1), reduced authority rho_d delta + d_off (0.3), random-walk drift
(0.2); result clipped to +-rud_max (`:710-727`). Priors (`:697-705`): rho_d U[0.3, 0.9],
d_off U[-0.05, 0.05] rud_max, drift LogU[0.1, 2] deg/sqrt(min). Ch: v, r, u.

**P3 - thrust-speed-throttle surface** (always on; replaces eta_T) (`cat_prop.py:312-382`):
```
T = t_max [ g(c_n) min(phi(s), eta_P P_eng / (t_max max(u_r, 1))) - k_idle s^s_i (1 - g(c_n)/g_0)_+^q_i ]   :368-377
phi(s) = phi_id exp(-k1 (s - 1) - k2 max(s - 1, 0)^2),  s = max(u_r, 0)/u_id
g(c) = sum_k b_k C(n,k) c^k (1-c)^(n-k)   (Bernstein; b_0 = 0, b_n = 1)                 :360-366
```
Priors (`:340-355`): w.p. 0.6 n ~ {3, 4, 5} with increments Dirichlet(2,...,2), else g(c) = c;
phi_id U[0.9, 1.1]; k1 U[0, 0.6]; k2 U[0, 0.3]; k_idle U[0, 0.4]; s_i U[1, 2]; q_i U[1, 3];
g_0 U[0.05, 0.1]; P_eng U[150, 250] kW; eta_P U[0.55, 0.70]; eta_P P_eng floored at
phi_id t_max u_id. Property (checked in `:1005-1051`): T(c, u_id) = t_max phi_id g(c) where
g(c) >= g_0. Structure: for g >= g_0 the surface is SEPARABLE, T = t_max g(c) Phi(u). Ch: u
(+ zdot, thetadot via the lever).

**P10 - partial thrust loss** (`cat_prop.py:638-682`): rho = 1 before t_f, then rho_f;
modes persistent 0.6 / recovering to 1 (exact, tau_r) 0.2 / intermittent two-state Markov
(dwell rates 1/d_on, 1/d_off) 0.2 (`:664-682`). Priors (`:651-659`): rho_f U[0.3, 0.95],
tau_r LogU[2, 60] s, d_on, d_off LogU[1, 30] s, t_f U[0, 90] s, kappa_rho U[0.8, 1.5].
Writes rho (thrust x rho, P4 limit x rho, P5 x rho^kappa_rho, P9 sigma up).

**P1 - intake suction** (`cat_prop.py:169-240`):
```
h  = h_in - s_p dx theta + slope dx - dz                                   :208-211
S(h) = clip((sig((h_on - h)/w) - s0)/(1 - s0), 0, 1),  s0 = sig((h_on - h_run_p)/w)   :213-219
beta = 1 - A S(h)
lost := (lost or h < h_on) and not reimm;  reimm = lost, h > h_on + dh for >= t_dwell   :222-228
target = min(beta, p) if lost else beta
p <- exp_update_asym(p, target, tau_rec (rising), tau_loss (falling), dt)                :229-232
```
Writes p *= p1, load *= p1^m (`:233-235`). Priors (`:189-203`): dx U[-0.05, 0.05] L,
dz U[-0.3, 0.3] h_run, h_run_p = h_run - s_p dx th0 - dz, h_on U[-0.3, 0.6] h_run,
w LogU[0.05, 1] h_run, A U[0.6, 1], dh U[0, 0.5] h_run capped to <= 0.9 h_run_p - h_on,
t_dwell U[0, 0.5] s, tau_loss LogU[0.02, 0.3] s, tau_rec LogU[0.2, 5] s, kappa_p U[0.8, 1.5],
m U[0.8, 0.85]. Ch: u, v, r (through thrust and side force).

**P13 - slam-entrained air** (`cat_prop.py:799-848`), only with P1: each low-fidelity slam
(entry speed v_k) adds alpha_k(t) = A_k g((t - t_k - D_k)/w_p), g(x) = x e^(1-x) (x > 0),
A_k = cA (v_k/(3 v_slam))^gam, D_k = L_p / max(u_r, 1) (`:828-839`); p13 follows
1 - tanh(sum alpha) through P1's asymmetric lag (`:840-842`); p *= p13, load *= p13^m (`:845-846`).
Priors (`:815-819`): L_p U[0.5, 1] (x_bow - x_i), w_p LogU[0.05, 0.5] s, cA U[0, 0.5],
gam U[0.5, 1.5]; at most 8 pending pulses.

**P2 - engine-pump speed / rev limiter** (`cat_prop.py:244-308`), only with P1.
nu_ss(c) = sqrt(c_idle + (1 - c_idle) c_n) (`:274-276`). Full variant (w.p. 0.5):
tau_n nu' = nu_ss^2 (1 - beta_e (nu/nu_ss - 1)) - load nu^2, equilibrium
nu_inf = 2 C0 / (b + sqrt(b^2 + 4 load C0)), b = beta_e nu_ss, C0 = (1 + beta_e) nu_ss^2, capped
at r_lim (`:278-283`); advanced exactly towards nu_inf with tau_eff = tau_n (beta_e + 2)/
(beta_e nu_ss + 2 load nu_inf), clipped at r_lim (`:298-305`); factor (nu/nu_ss)^2 from the
state BEFORE the update (`:300-301`). Simple variant: at reimm x := r_lim^2 - 1, factor (1 + x),
x <- x e^(-dt/tau_n) (`:290-296`). Priors (`:264-269`): tau_n LogU[0.1, 1.5] s, r_lim
U[1.02, 1.25], beta_e LogU[2, 10], c_idle U[0.0225, 0.09]. At load = 1 and constant c_n,
nu = nu_ss and the factor is 1. Observed 'rpm' = nu (stored only).

**P9 - thrust pulsation** (`cat_prop.py:597-634`), always on:
```
sigma = s0 + s_v (1 - p) + s_r (1 - rho) + s_w |u_orb(x_i)| / max(u_r, u_lo/2)     :621-625
pulse = 0.9 tanh(sigma eps / 0.9),  eps = unit OU, bandwidth om (exact step)       :627-632
```
(eps of the current substep is the state before the OU step; u_orb is the centre-line body-x
orbital velocity interpolated to x_i, `:91-96,622`). Priors (`:612-616`): s0 LogU[0.003, 0.03],
s_v U[0, 0.3], s_c U[0, 0.5], s_w U[0, 3], s_r U[0, 0.3], om LogU[2, 60] rad/s.

**P4 - cavitation saturation** (`cat_prop.py:386-424`), last thrust factor:
T_lim = T_lim0 rho (1 + eps (u_r/u_a)^2), u_a = sqrt(2 (p_atm - p_v)/rho_w) = 13.9 m/s
(`:78,403-406`); x = T_pre^+/T_lim; P9's cavitation noise T_pre *= 1 + 0.9 tanh(s_c cav eps/0.9),
cav = 1 - (1 + x^m)^(-1/m) (`:416-420`); T_eff = T_lim x (1 + x^m)^(-1/m) for T_pre > 0
(`:421-422`). Priors (`:399-401`): T_lim0 LogU[0.6, 2] t_max, m U[2, 8], eps U[0.5, 0.9].

**P5 - nozzle side force** (always on; replaces eta_S and c3) (`cat_prop.py:428-475`):
```
F_y = [ k_js T_eff^+ psi(u_r) p^(kp-1) rho^(kr-1) + k_i t_max exp(-c_n/c_i) p^kp rho^kr ] h(delta_e)   :462-474
psi = clip(exp(k_s (s - 1)), 0.5, 2.5)                                                  :452-455
h(d) = sin(e_d d_s tanh(d/d_s)) (1 + a sgn d)                                           :457-460
```
kp, kr = kappa of P1 / P10 (default 1). Priors (`:445-450`): k_s U[0, 1], k_i U[0, 0.15],
c_i U[0.05, 0.3], e_d U[0.75, 1], d_s LogU[0.6, 3] rud_max, a U[-0.1, 0.1]. Since k_s >= 0,
psi <= 1 below u_id. Ch: v, r.

**P8 - stuck reversing bucket** (`cat_prop.py:565-593`): from t_f, F_x and F_y x
f = 1 - (1 + k_rev) (b - b0)_+/(1 - b0) (`:580-592`). Priors (`:575-578`): k_rev U[0.4, 0.8],
b0 U[0, 0.15], b U[0.05, 0.3], t_f U[0, 90] s.

**P14 - general residual** (`cat_prop.py:852-883`): dF = t_max eps c_n tanh(W phi(X)),
phi(X) = sqrt(2/24) cos(Omega X + b), X = (c_n, u_r/u_id, delta_e/rud_max, h_in/h_run, nu)
(nu = P2's or sqrt(c_n)) (`:871-879`); |dF_x|, |dF_y| <= eps t_max c_n. Priors (`:864-869`):
eps LogU[0.005, 0.1], ell LogU[0.3, 3], Omega = N(0,1)/ell (24 x 5), b U[0, 2 pi], W N(0,1)
(2 x 24). Ch: u, v, r (+ lever).

**P6 - jet-hull interaction / nozzle pitch bias** (`cat_prop.py:479-514`): parts (a) 0.5, (b) 0.6
conditional on >= 1 (`:498`):
F_z = a [k_z (T_eff - T_id) + k_th T_eff trim_rel] + b T_eff sin d_tp,
dF_x = -a t_d T_eff + b T_eff (cos d_tp - 1), T_id = k_drag u_id^2 = 2313.6 N (`:503-514`).
Priors (`:500-501`): k_z U[-0.08, 0.12], k_th U[-1, 1], t_d U[-0.05, 0.1], d_tp U[-6, 6] deg.
Ch: zdot, thetadot (F_z at the jet point), u.

---------------------------------------------------------------------------------------

## 6. Comparison with the high-fidelity jet

### 6.0 Operating region used for the numbers
The 16 recorded high-fidelity missions `studies/_cache/onestep_gz/streams_T120.pkl` (7680
control steps, states at step start): c/t_max quantiles 1/50/99 % = 0.055 / 0.431 / 0.684;
u = 7.00 / 11.30 / 13.29 m/s (s = 0.54 / 0.88 / 1.03); delta at +-rud_max in 4.3 % of steps;
c >= 0 in all rows; c/t_max < 0.05 in 0.8 %. The prime pi is not recorded in these streams,
so every force comparison below takes pi = 1 (UNKNOWN: how often pi < 1 in these missions).

### 6.1 Thrust surface T(c, u) at delta = 0 (P3 vs T_net)
- **Exact at u = u_ref**: T_net(c, u_ref) = c (section 2.2) equals P3 with g(c) = c (drawn
  w.p. 0.4), phi_id = 1 (inside U[0.9, 1.1]); u_id = u_ref = 12.8611 in both.
- **Not exact elsewhere, because P3 is separable and T_net is not.** Ratio T_net(c, u)/c:

| u (m/s) | c=0.05 | 0.1 | 0.3 | 0.5 | 0.7 | 1.0 (x t_max) |
|---|---|---|---|---|---|---|
| 0 | 3.043 | 2.322 | 1.709 | 1.540 | 1.452 | 1.376 |
| 8 | 1.637 | 1.419 | 1.234 | 1.182 | 1.154 | 1.130 |
| 12.86 | 1 | 1 | 1 | 1 | 1 | 1 |
| 16 | 0.620 | 0.755 | 0.863 | 0.893 | 0.908 | 0.922 |
| 23.5 | -0.291 | 0.197 | 0.566 | 0.661 | 0.709 | 0.752 |

  P3 (g >= g_0) has one factor Phi(u) for all c. Best separable (rank-1) approximation of
  T_net on c in [0.05, 0.75], u in [7, 13.5]: residual rms 37.9 N (second singular value
  1.2 % of the first); on c in [0.05, 1], u in [0, 23.5]: 158 N.
- Best P3 inside its prior box, least squares on the recorded rows (g = c): phi_id 0.9997,
  k1 0.4685, k2 0.280, k_idle 0 (others inactive); residual rms 44.2 N, max 284.5 N
  (rms T_net 2932 N; relative rms 4.8 %). Freeing k1 to [-3, 3] does not change it, so the
  residual is the separable structure, not the prior bounds.
- Ram drag: T_net < 0 only for u > u_ref and small c (c <= ~0.06 t_max at 23.5 m/s). P3's
  ram term -k_idle s^s_i (1 - g/g_0)^q_i is negative at every s > 0 when k_idle > 0, while
  T_net > 0 at all u < u_ref; so P3 matches with k_idle = 0 below u_ref and cannot produce the
  negative thrust above u_ref at the same time. Outside the recorded region (u > u_ref with
  c < 0.06 t_max: < 1 % of rows).
- Dead band: for 0 < c < c_db both laws are linear in c (P3: g(c) Phi(u) with g(c) = c).

### 6.2 Axial (cos delta) loss
F_x^hf - F_x^hf(delta = 0) = -pi G (1 - cos delta). Chain: -k_c T_eff (1 - cos delta) with
k_c ~ U[0, 1] (or 0 without T5). Required k_c = G/T_net = 1/(1 - mu): recorded quantiles
1/50/99 % = 1.33 / 1.63 / 2.26, i.e. **outside the prior support [0, 1]; not representable**.
With k_c = 1 the shortfall (G - T_net)(1 - cos delta) = m_dot V_in (1 - cos delta) is 8.6 N
mean, 103 N at 99 %, 136 N max (0.052 m/s^2 surge at 99 % with m_u = 1968.6 kg); the whole
high-fidelity axial loss is 19.8 N mean, 236 N at 99 %.
Combined surge-force check (pi = 1): the true jet-force error F_x^hf - c_a has rms 208.5 N; after
the best P3 the residual is 46.6 N rms with k_c = 0 and 36.1 N rms (max 288 N) with k_c = 1.

### 6.3 Side force (P5 vs pi G sin delta)
- Angle shape: high-fidelity sin delta exactly (no clip, no asymmetry). P5's h(delta) equals
  sin delta only in the limit d_s -> infinity; inside the prior (e_d = 1, a = 0, d_s = 3 rud_max,
  the largest) the maximum relative error over |delta| <= rud_max is 3.3 % (at rud_max);
  at d_s = rud_max it is 22.8 %. **Approximate**.
- Gain: G(c, u) = T_net/(1 - mu) depends on c at fixed u:

| u (m/s) | c=0.05 | 0.1 | 0.3 | 0.5 | 0.7 | 1.0 |
|---|---|---|---|---|---|---|
| G/T_net at 8 | 2.19 | 1.82 | 1.46 | 1.35 | 1.29 | 1.24 |
| G/T_net at 12.86 | 4.51 | 3.03 | 1.94 | 1.68 | 1.55 | 1.44 |
| G/c at 8 (lofi uses 1.614) | 3.58 | 2.59 | 1.80 | 1.59 | 1.49 | 1.40 |
| G/c at 12.86 | 4.51 | 3.03 | 1.94 | 1.68 | 1.55 | 1.44 |

  P5's jet term needs k_js psi(u) = G/T_net, a function of c and u; psi depends on u only, and
  psi <= 1 below u_id (k_s >= 0) while the required psi exceeds 1 at low c. The c-dependence
  can only be carried by the idle term k_i t_max exp(-c_n/c_i), which is speed-independent;
  the high-fidelity excess G - T_net = kappa_db rho A V_j V_in grows with u (and with c).
  At c -> 0 the high-fidelity side force goes to 0 (m_dot ~ c/c_db below c_db; G(c_db) =
  597 N at 8 m/s, 921 N at 12.86 m/s) while P5's idle term is largest (k_i t_max <= 955 N).
- Best P5 inside the prior on the recorded rows (with T_eff = T_net): k_s 0.574, k_i 0.15 and
  c_i 0.3 (both at their upper bounds); residual rms 120.5 N per unit sin delta (G rms 4645 N),
  max 573 N; with the bounds removed 92.6 N. In force: the true side-force error F_y^hf - F_y^lo
  has rms 70.8 N; after the best P3 + P5 (e_d = 1, d_s = 3 rud_max, a = 0) the residual is
  22.9 N rms, max 142 N (F_y^hf rms 341 N). **Approximate**.
- Loss scaling: F_y^hf proportional to pi (exponent 1); P5 gives p^kappa_p with kappa_p ~
  U[0.8, 1.5] (1 inside). **Exact form**.

### 6.4 Suction / prime (P1, with the forced P2)
- Input: catalogue h_in equals the high-fidelity h at the same state in calm water: the
  catalogue intake is (x_i, 0, keel_rest(x_i)) = (-1.3812, 0, -0.3608) (`operators_cat.py:76-101`),
  the plant's intake in its rest frame is Z_rest + z_i + s_p x_i Th_rest = -0.3608 (printed);
  both give 0.2199 m at the running attitude for u_ref. Roll does not enter either (y_i = 0).
- Static map: high-fidelity pi* = clip(h/h_b, 0, 1) with h_b/h_run = 0.193; P1 beta(h) = 1 -
  A S(h) (logistic). A = 1 (edge of U[0.6, 1]) gives the zero floor. Best logistic (minimax over
  h) has max |beta - pi*| = 0.056 at h_on = 0.097 h_run, w = 0.034 h_run, but w >= 0.05 h_run in
  the prior; at w = 0.05 h_run the best is 0.126. dx = dz = 0 inside the prior.
- Dynamics: both are asymmetric exponential lags of the same form (tau_loss = 0.05 s inside
  LogU[0.02, 0.3]; tau_rec = 0.5 s inside LogU[0.2, 5]). P1 adds a latch the plant lacks: while
  h < h_on, p cannot rise (target min(beta, p)), whereas the plant's prime rises towards h/h_b
  for 0 < h < h_on. With dh = 0 and t_dwell = 0 (both inside the prior) the latch clears at the
  first substep with h > h_on. **Approximate**.
- P2 is drawn whenever P1 is (`DEPENDS`, P_ON 1.0). The plant has no engine-speed state:
  pi multiplies flow and power is P(c) regardless. P2 cannot be neutral: simple variant gives a
  thrust overshoot factor >= r_lim^2 = 1.0404 decaying with tau_n >= 0.1 s after every latch
  clearing; full variant adds a lag tau_eff on every change of c_n and overspeed nu_inf > nu_ss
  when load = p^m < 1. **Not representable exactly when P1 is on; near only with the simple
  variant, r_lim -> 1.02, tau_n -> 0.1 s.**
- P13 must be off (or cA = 0): the plant has no slam-air effect.

### 6.5 Actuators (episode M10 chain vs `Propulsion.advance` / `Nozzle.step`)
Same stage order in both: delay -> dead zone -> (gain, offset) -> clip -> (backlash) ->
rate-limited first order (`lofi.py:293-316` vs `actuators.py:190-196`, `waterjet.py:237-240`).
Differences: (i) the plant integrates with explicit Euler at 0.02 s and applies the
start-of-step value; M10 uses the exact solution and applies the step average at 0.04 s
(per 0.04 s decay factor 0.93444 vs 0.93551 at tau = 0.6 s); (ii) the plant delay is
0.1 s; M10 delays are whole 0.04 s steps (round(U[0, 0.3] rs / 0.04), `lofi.py:206-207`), so
0.08 or 0.12 s, never 0.10 s; (iii) M10 clips thrust to [0, t_max] (`lofi.py:267-268`), the
plant to [-t_max/3, t_max] (no effect, c >= 0). Values inside the M10 ranges
(`lofi.py:159-186`): thrust tau 0.6 s (LogU[0.05, 2] rs = [0.037, 1.47] s), rate t_max/0.3 s
(t_max/[0.147, 2.20] s), dead band 26.5 N = 0.0042 t_max (U[0, 0.03] t_max); nozzle tau 0.1 s
([0.015, 0.367] s), 0.698 rad/s ([0.34, 2.72] rad/s). **Approximate (delay +-0.02 s).**
P7 must be off or neutral (the plant's nozzle does not depend on thrust): neutral is reachable
(lam_r = lam_b = 0, k_c7 = 0 leaves only rate0 >= 0.34 rad/s, which binds at < 0.698 rad/s - so
**not neutral when its rate part is on** unless rate0 > 0.698 rad/s, inside LogU[0.34, 2.72]).

### 6.6 Moments, lever and inertia (how DeltaF reaches the 5 channels)
- Yaw arm: plant x_n = -2.1812 lies inside x_j in [x_stern - 0.54, x_stern + 0.54] =
  [-2.665, -1.585] (T5 on); with T5 off x_j = x_stern = -2.125. **Representable**.
- Pitch arm: plant z_n = -0.40 m inside z_j ~ U[-0.577, 0] (T5 on); with T5 off (probability
  1 - 0.95 x 0.7 = 0.335) z_j = 0 and the thrust error has no pitch moment. **Representable
  only with T5 on.**
- Heave: lever_Q adds s_p theta DF_x to the heave row (`cat_base.py:310`); the plant's jet adds
  no vertical force and no theta F_x term (`planing_vessel.py:347,350`). **A small extra term
  the family always has** (|theta| about 0.01-0.05 rad, so <= 5 % of DF_x).
- Roll moment -z_n F_y: no roll output channel; not compared.
- Inertia (cross-layer, affects every P item): surge M_t[0,0] = m_u = 1968.58 kg for every draw
  vs plant 1470 kg; yaw M_t[2,2] = max(I_r0 LogU[0.8, 1.25], 1.05 I_zz) with I_r0 = 5253.5
  (40 draws: 4378-6381) vs plant 2932.5 kg m^2. Yaw acceleration per newton of side force at
  the nozzle: plant 7.44e-4; low-fidelity k_nomoto_f/tau_r = 4.04e-4; chain (diagonal of M_t,
  x_j and I_r at their extremes) 2.41e-4 to 6.34e-4 -> **the plant's value lies outside the
  chain's range**. The layer-1 T5 structural yaw part uses its own I_r = m_v rho_r^2 in
  [2275, 6416] (contains 2932.5) - so the low-fidelity-law share and the error share of the same
  side force are divided by different yaw inertias. Off-diagonal M_t couplings (m x_g, m z_g)
  were not included in these bounds.

### 6.7 Items the plant needs off, and whether they can be off or neutral
| item | plant counterpart | can it vanish? |
|---|---|---|
| P2 | none | no when P1 on (6.4) |
| P4 | none | off w.p. 0.7; on: T_lim0 = 2 t_max, m = 8 gives T_eff/T_pre >= (1 + 0.5^8)^(-1/8) = 0.99951 for T_pre <= t_max |
| P6 | none (F_z = 0, no bias) | off w.p. 0.2; on: k_z = k_th = t_d = 0, d_tp = 0 inside the priors -> exactly 0 |
| P7 | none | off w.p. 0.392; on: see 6.5 |
| P8 | bucket only for c < 0 | off w.p. 0.98; on: b <= b0 possible -> f = 1 |
| P9 | none (deterministic) | **always on, s0 >= 0.003**: multiplicative coloured noise of std >= 0.3 % of T_eff (median s0 0.0095); s_w = s_v = s_r = s_c = 0 allowed |
| P10, P11 | none | off w.p. 0.925, 0.97; on: not neutral after t_f (rho_f <= 0.95; rho_d <= 0.9 or stuck) |
| P12 | none | off w.p. 0.778; limp rule binds above c_cap <= 0.7 |
| P13 | none | off w.p. 0.7 given P1; cA = 0 allowed |
| P14 | none | off w.p. 0.5; on: |dF| up to eps t_max c_n, eps >= 0.005, not 0 |

Probability (non-sparse) that P4, P6, P7, P8, P10, P11, P12, P14 are all off:
0.7 x 0.2 x 0.392 x 0.98 x 0.925 x 0.97 x 0.778 x 0.5 = 0.0188 (x 0.7 more if P1 is on and
P13 must be off).

### 6.8 Summary
| plant relation | chain element | verdict |
|---|---|---|
| T_net(c, u_ref) = c | P3 g = c, phi_id = 1 | exact |
| T_net(c, u) off u_ref (non-separable) | P3 separable g(c) Phi(u) | approximate: 38-44 N rms in the recorded region, 158 N over the full envelope |
| negative T_net at u > u_ref, small c | P3 ram term | not together with T_net > 0 below u_ref |
| axial loss G (1 - cos delta), G/T_net = 1.33-2.26 | k_c T_eff (1 - cos delta), k_c <= 1 | not (prior too narrow); <= 136 N short |
| side force pi G sin delta | P5 [k_js psi(u) T + k_i t_max iota(c)] h(delta) | approximate: 23 N rms force residual (prior bounds active) |
| sin delta | h(delta) | approximate (>= 3.3 % at rud_max) |
| prime pi: ramp + asymmetric lag, linear in F_x, F_y | P1 logistic + latch, kappa = 1 | approximate (max |beta - pi*| >= 0.126 inside the prior) |
| no engine-speed dynamics | P2 forced with P1 | not exact when P1 on |
| deterministic | P9 always on | not exact (noise floor 0.3 %) |
| actuator chain | episode M10 | approximate (delay quantised 0.08/0.12 vs 0.10 s) |
| moments x_n F_y, z_n F_x | lever_Q at (x_j, 0, z_j) | arms representable (z_n only with T5 on) |
| per-newton yaw response x_n / 2932.5 | x_j / M_t[2,2] | not (M_t yaw inertia 4378-6381 vs 2932.5) |
| surge inertia 1470 kg | M_t[0,0] = 1968.58 kg fixed | not (cross-layer) |
| heave from the jet = 0 | lever_Q adds s_p theta DF_x | small extra term always present |

---------------------------------------------------------------------------------------

## 7. Docstring vs code, and UNKNOWNs

1. `cat_prop.py:15` and `cat_base.py:1116-1117` say the error is the jet force "minus the
   low-fidelity boat's own jet force". The code subtracts F_x0 = c_a (1 + k_c (cos delta_a - 1))
   (`cat_base.py:1152`), which is not the low-fidelity surge force (c_a, `reduced.py:395`) when
   k_c > 0; the difference is supplied by layer-1 T5's surge part (`operators_rb.py:839`), so
   the sum is consistent only through T5.
2. P5's docstring formula multiplies the jet term by `(p rho)^(kappa - 1)` with one kappa; the
   code uses separate exponents p^(kappa_p - 1) rho^(kappa_rho - 1) (`cat_prop.py:466-473`).
3. P1's docstring writes S with `( . )_+`; the code clips to [0, 1] (`cat_prop.py:217-218`)
   (same values for s0 < 1).
4. The module docstring lists the chain as "T_pre = T_real(P3) x rho(P10) x p(P1, P13) ...";
   the code computes T_pre once at the P4 slot whether or not P4 is on (`cat_base.py:1136-1137`)
   - consistent; noted because T_pre is NOT recomputed after P5 (P14 and P6 do not change it).
5. P12's speed limit reads u over ground (sr[:,2], `cat_prop.py:781`), not u_r as P3/P5 do.
6. P13 keeps its own clock inside `chain` (`cat_prop.py:847`), not through `clock = True`.

UNKNOWN:
- How often the prime pi < 1 in the recorded high-fidelity missions (pi and h are not in
  `streams_T120.pkl`; `PlaningVesselGZ` records h only with record=True, `planing_vessel_gz.py:267-272`).
  All force numbers in 6.1-6.3 assume pi = 1.
- The scale of these force residuals relative to the network's other error sources (needs the
  per-channel target statistics; not computed here).
- Whether the M_t off-diagonal terms move the yaw bound of 6.6 by more than a few percent (not
  computed).

---------------------------------------------------------------------------------------

## 8. Reproduction
- `scratchpad/wf_math/prop_check.py`: builds the Scarab `JetPump`/`Waterjet` with the plant's
  parameters (no plant run), tabulates T_net, G, mu, fits P3 and P5 inside their prior boxes on
  the 7680 recorded rows, the rank-1 bound, h(delta) error, required k_c, and prints the
  catalogue intake and h_run (`operators_cat.cat_geometry`).
- `scratchpad/wf_math/prop_check2.py`: force residuals of P3 + k_c and P3 + P5, yaw per-newton
  values, P1 logistic-vs-ramp minimax.
- Inline checks: plant inertias, intake height, calm submergence (`PlaningVesselGZ` built for
  `scarab195` with a calm `Monochromatic(1.0, 0.0)` sea); 40 draws of `OperatorGenL3` for the
  M_t diagonal.
