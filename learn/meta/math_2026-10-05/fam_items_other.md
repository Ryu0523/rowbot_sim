# D10a catalogue family: hidden (H1-H3), boat (B1-B3), air (A1-A3) and environment (E1-E4) items, the contact gate W4, the observation model O1-O5, and the per-mission redraws

Source code read (all citations are `file:line` in the repository
`C:/Users/Administrator/Documents/usv-rl-mpc`):

- `learn/meta/cat_base.py`, `cat_hidden_boat.py`, `cat_air_env.py`, `cat_water_a.py` (W4 only),
  `cat_obs.py`, `operators_cat.py`: last commit touching them `af7fb6d`; no working-tree change
  (git status 2026-10-04).
- `learn/meta/episode8.py`: untracked working-tree file, read 2026-10-04.
- Context read: `learn/meta/episode6.py`, `episode7.py`, `operators.py`, `operators_gen.py`,
  `operators_rb.py`, `control/reduced.py`, `learn/meta/data2.py`, `studies/meta_step8.py`.
- Numerical values of the low-fidelity parameters were printed by a 1-second script
  (`R.default_params()`, `operators_cat.cat_geometry()`, `cat_base.CatCtx`), 2026-10-04.
  Realised on-frequencies were counted from `studies/_cache/meta8/held8_meta.pkl` (100 held-out boats).

Where a docstring and the code disagree, the code is followed and the disagreement is listed in
section 9.

---------------------------------------------------------------------------------------------------

## 0. Conventions, frames and symbols

### 0.1 Frames and the reduced state

- Body / level frame of every item: x forward, y to PORT, z up. Source: `lever_Q` docstring
  `cat_base.py:299-305`; lateral sample offsets `y_off = (-B/2, 0, +B/2)` labelled "stbd..port"
  `cat_base.py:351`.
- Earth frame: horizontal (X, Y), elevation up. A body point (x_p, y_p) is at
  X = x + x_p cos psi - y_p sin psi, Y = y + x_p sin psi + y_p cos psi (`cat_base.py:454-455`), so
  heading psi grows counter-clockwise seen from above; yaw rate r = d psi/dt (`control/reduced.py:498`),
  r > 0 = turning to port.
- Reduced state of one row, s_r in R^10 (`cat_base.py:531`, `cat_obs.py:17-21`):
  s_r = (x, y, u, z, zd, th, thd, psi, r, v).
  - x, y: earth position (m). psi: heading (rad).
  - u, v: surge and sway velocity OVER GROUND in level-frame body axes (m/s).
  - z: heave of the reference point, up positive (m); zd = dz/dt.
  - th: pitch angle in the low-fidelity convention; s_p := `sign_pitch` = -1 (default parameters), so the
    BOW-UP angle is s_p th = -th. The height of hull point (x_p, y_p) is
    z + s_p x_p th + y_p phi (`cat_base.py:537-538`), phi = the hidden roll of H1 (0 when H1 is off).
  - r: yaw rate (rad/s).
- Error channels c in (u, v, r, w, q) = reduced indices `RED_VEL = (2, 9, 8, 4, 6)`
  (`operators_rb.py:92`), i.e. (du/dt, dv/dt, dr/dt, d(zd)/dt, d(thd)/dt). Channel q is the rate of th
  (lofi convention), NOT the bow-up rate.
- Stations: 5 longitudinal stations x_i, i = 1..5, linspace(x_stern, x_bow) (`cat_base.py:350`),
  3 lateral samples j in {stbd (y = -B/2), centre (0), port (+B/2)} (`cat_base.py:351`).

### 0.2 Distributions

U[a,b] uniform; LogU[a,b] = exp(U[ln a, ln b]) (`cat_hidden_boat.py:62-63`, `cat_air_env.py:106-107`);
N(mu, s^2) normal; Bern(p) Bernoulli; t_nu Student t with nu degrees of freedom;
sigma(x) = 1/(1+e^-x); "w.p." = with probability.

### 0.3 Low-fidelity constants (fixed parameters, same for every draw unless B1 rescales them)

From `R.default_params()` (the operator gets no `p`, so `OperatorCat` uses `R.default_params()`,
`operators_cat.py:180`; `episode7.py:74-80`), task hull geometry from `cat_geometry()` (`operators_cat.py:76-101`):

| symbol | code | value | unit |
|---|---|---|---|
| L, B, T | `L, B, T` | 5.4, 2.0252, 0.38463 | m |
| m (rigid mass used by the catalogue) | `m_coriolis` -> `ctx.m` (`cat_base.py:368`) | 1400 | kg |
| W = m g | `ctx.W` (`cat_base.py:369`) | 13734 | N |
| m_u, m_v (lofi surge / sway mass incl. added mass) | `m_surge, m_sway` | 1968.58, 1610.20 | kg |
| k_d (lofi quadratic surge drag, drag = k_d u abs(u), `control/reduced.py:394`) | `k_drag` | 13.987 | N s^2/m^2 |
| k_js (jet side-force gain) | `k_jet_side` | 1.6140 | - |
| T_max, delta_max = delta_stall | `t_max, rud_max, rud_stall` | 6365.7 N, 0.43633 rad | |
| u_id (identification speed) | `u_design` -> `ctx.u_id` | 12.861 (25 kn) | m/s |
| u_max | `u_max` | 23.515 | m/s |
| u_lo, u_hi (task speed range) | `R.U_RANGE` (`operators_rb.py:98`) | 8.2311, 15.433 (16, 30 kn) | m/s |
| omega_h, zeta_h | `wn_heave, z_heave` | 5.8349 rad/s, 0.37232 | |
| omega_p, zeta_p | `wn_pitch, z_pitch` | 3.7135 rad/s, 0.15506 | |
| z0, th0 (lofi running attitude) | `z0, th0` | 0.15366 m, -0.0092291 rad | |
| tau_r (lofi yaw time constant) | `tau_r` | 0.42 | s |
| k_lin_sway, k_sway | | 1679.94 N s/m, 391.87 N s^2/m^2 | |
| steer_jet | | 1 (waterjet) | |
| c_wave_sway, c_wave_yaw | | 0, 0 | |
| k_wh, k_wp | `k_wave_heave/pitch` absent -> 1.0 (`cat_base.py:381-382`) | 1, 1 | |
| x_i | `ctx.x_st` | (-2.125, -0.803, 0.519, 1.841, 3.163) | m |
| k_i (keel height, rest frame) | `ctx.keel5` | (-0.385, -0.342, -0.300, -0.258, -0.215) | m |
| h_run,i (calm-running keel immersion) | `ctx.hc_run` (`cat_base.py:386-387`) | (0.2506, 0.1961, 0.1416, 0.0871, 0.0326) | m |
| F_b (bow freeboard) | `ctx.F_b` = depth - draft_bow (`operators_gen.py:158-162`) | 0.88454 | m |
| a_ref | `ctx.a_ref` (`operators_gen.py:165-174`) | (3.234, 2.697, 1.756, 13.095, 1.964) | m/s^2 or rad/s^2 |
| A_REF, CLIP | `operators.py:63,67` | (4, 4, 2, 12, 3), 3 | |

The low-fidelity boat (`control/reduced.py:394-498`), needed because several items shift its own terms:
m_u du/dt = thr - k_d u abs(u); z'' = -omega_h^2 (z - zbar) - 2 zeta_h omega_h zd with
zbar = k_wh mean_i eta_{i,c} + z0; th'' = -omega_p^2 (th - alpha) - 2 zeta_p omega_p thd with
alpha = k_wp s_p atan(slope) + th0 (exact 2x2 transition, `reduced.py:397-424`);
m_v dv/dt = k_js max(thr,0) sin(clip(noz, +-delta_stall)) - k_lin_sway v - k_sway v abs(v) - m u r
(`reduced.py:443-453`); r' = (k_nomoto_f * lift - r)/tau_r (exact first-order, `reduced.py:482-488`).

### 0.4 Per-draw hidden constants and per-substep shared quantities

Per draw (`CatDraw.__init__`, `cat_base.py:891-932`):
- tau0 ~ U[2 deg, 6 deg]: hidden TRUE running trim (bow-up), `cat_base.py:147,905`.
- M_t (5x5, channels u, v, r, w, q): the shared inertia; set by layer 3 (`cat_base.py:910-912`) and,
  when B1 is on, replaced by B1's M_new (`cat_hidden_boat.py:544-547`). M_t^-1 =: `ctx.Minv`.

Per 0.04 s substep, the dict q (`substep_quantities` `cat_base.py:495-525`, `derive_quantities`
`cat_base.py:528-578`), evaluated at the START state of the substep:

| symbol | definition | cite |
|---|---|---|
| thr, noz | the low-fidelity boat's APPLIED thrust (N) and nozzle angle (rad) | `cat_base.py:509-510` |
| eta_{i,j}, etad_{i,j} | station elevation / rate along the moving station = lofi sea (`CatSea.sample`) + E4 packet (`eta_add`) | `cat_base.py:428-476, 533-534` |
| d_{i,j} | eta_{i,j} - (z + s_p x_i th + y_j phi) (station immersion) | `cat_base.py:537,542` |
| dd_{i,j} | etad_{i,j} - (zd + s_p x_i thd + y_j phid) (entry velocity, > 0 into water) | `cat_base.py:539,543` |
| h_{i,j}, hc_i | h = d - k_i (keel immersion), hc_i = h_{i,centre} | `cat_base.py:544-545` |
| slope | sum_i eta_{i,c} xt_i / sum_i xt_i^2, xt_i = x_i - mean(x) | `cat_base.py:547` |
| t_slope | mean_i (eta_{i,port} - eta_{i,stbd}) / B (transverse water slope) | `cat_base.py:549` |
| z_ref, th_ref | k_wh mean_i eta_{i,c} + z0 ;  k_wp s_p atan(slope) + th0 | `cat_base.py:553-554` |
| e_z, e_th | z - z_ref ; th - th_ref | `cat_base.py:557` |
| w_r, q_r | zd - zd_ref ; thd - thd_ref | `cat_base.py:558` |
| theta_rel (`trim_rel`) | s_p (th - th0): bow-up pitch relative to the lofi running attitude | `cat_base.py:562` |
| (u_c, v_c) (`nu_c`) | body-frame current, 0 unless E1 is on | `cat_base.py:519`, `cat_air_env.py:489` |
| u_r, v_r, U_r, U_g | u - u_c, v - v_c, sqrt(u_r^2+v_r^2), sqrt(u^2+v^2) | `cat_base.py:570-572` |
| nu0 (B x 5) | the lofi boat's own acceleration over the substep: (u, v, r) = (lofi next - start)/dt; (w, q) replaced by a_lofi | `cat_base.py:521-524` |
| a_lofi = (a_lofi,z, a_lofi,th) | (-omega_h^2 (z - z_ref') - 2 zeta_h omega_h zd, -omega_p^2 (th - th_ref') - 2 zeta_p omega_p thd), computed from the LOFI sea eta0 only (no E4 packet) | `cat_base.py:480-492, 520` |
| H_b (bow height) | F_b + z + s_p x_5 th - max_j eta_{5,j} | `cat_base.py:600-606` |

---------------------------------------------------------------------------------------------------

## 1. Framework: how an item's output becomes the recorded error

### 1.1 Which items are on in a draw

`CatDraw._draw_items` (`cat_base.py:938-973`):

1. sparse ~ Bern(0.15) (`cat_base.py:145, 906`). If sparse: n ~ U{1,2,3} items chosen without replacement
   from the pool of the 38 registered non-structural codes (all 39 codes of `CODES` are registered;
   `STRUCTURAL = ("W4",)`), a chosen dependent brings its parent (`DEPENDS = {A2: A1, P2: P1, P13: P1}`,
   `cat_base.py:106, 944-951`). Chosen items are drawn with `draw_on` (no on-probability, `:964-966`);
   unchosen non-structural items are off.
2. Not sparse: item c is on w.p. P_ON[c] (`CatItem.draw`, `cat_base.py:659-663`), with
   P_ON (`cat_base.py:92-105`): H1 0.7, H2 0.4, H3 0.15, A1 0.7, A2 1.0 (only if A1 is on, `:962`),
   A3 0.6, B1 0.8, B2 1 - 0.95*0.97*0.97 = 0.10614, B3 0.3, E1 0.6, E2 0.1, E3 0.3, E4 0.3, W4 1.0.
3. W4 is drawn in both modes and P_ON = 1, so W4 is always on (`cat_base.py:964-970`).
4. B1 is drawn FIRST (`cat_base.py:953`) and every on item's `adjust_ctx` runs right after its draw
   (`:971-972`); B1's `adjust_ctx` rescales m, W, vol and sets M_t (section 3.1), so every later draw
   that reads `ctx.m`, `ctx.W`, `ctx.Mt` uses the loaded boat.

Marginal prior on-probability per draw attempt, P = 0.85 P_ON + 0.15 P_sparse, P_sparse = E[n]/38 = 2/38
for a single code, and for A1 P_sparse = 1 - mean_n C(36,n)/C(38,n) = 0.10337 (A1 chosen or A2 chosen):

| item | prior marginal | realised in 100 held-out meta8 boats |
|---|---|---|
| H1 | 0.603 | 0.61 |
| H2 | 0.348 | 0.32 |
| H3 | 0.135 | 0.20 |
| A1 | 0.611 | 0.63 |
| A2 | 0.603 | 0.63 |
| A3 | 0.518 | 0.50 |
| B1 | 0.688 | 0.77 |
| B2 | 0.098 | 0.08 |
| B3 | 0.263 | 0.23 |
| E1 | 0.518 | 0.54 |
| E2 | 0.093 | 0.09 |
| E3 | 0.263 | 0.25 |
| E4 | 0.263 | 0.30 |
| W4 | 1 | 1.00 |

Acceptance (`operators_cat.py:329-375`): divergence-only closed-loop test (60 s, one calm and one
standard-sea row, mission-0 environment); a rejected draw is replaced by the next attempt stream; after
30 rejections only layer 1 + W4 remain (`forces_off_now`, `operators_cat.py:209-217`). In the 100
held-out boats: mean 0.04 rejections, 0 forces_off, 13 sparse.

### 1.2 Order inside one substep

`CatDraw.substep` (`cat_base.py:1017-1108`), batched over rows:

1. shared quantities (section 0.4);
2. stage 'env' items in `CODES` order: A1, A2, E1, E4 (`:1054-1057`; `self.on` is ordered by `CODES`,
   `:973`); re-derive q (`:1058-1059`);
3. stage 'hidden': H1; re-derive q (so the roll phi changes d, dd, h at the port / stbd samples);
4. slam detection (`:1060`); layer 1 at the water-relative velocities (`:1063-1078`), its vertical
   hydrodynamic parts a_L1 = sum over T2c, T3, T4, T7, Teven of their (w, q) rows (`:131, 1074`);
5. stage 'force' in `CODES` order: (W items), H2, H3, A3, B1, B2, B3, E2, E3 (`:1080-1082`);
   then the propulsion chain (`:1083-1088`); then stage 'gate': W4 (`:1089-1091`);
6. layer 3 and the Coriolis difference (`:1093-1102`).

### 1.3 Mapping of an item's output

`item_accel` (`cat_base.py:724-740`): an item returns a generalised force Q (B x 5) and / or an
acceleration `acc` (B x 5) and / or an impulse `imp` (B x 5 velocity jumps). Its acceleration
contribution is

  a_item = S( M_t^-1 Q + acc ),   S(y)_c = 3 n_c tanh( y_c / (3 n_c) )   (`cat_base.py:161-164, 737-738`)

with n_c = `nominal()` = a_ref for every item of this document (no override), except W4 whose
`saturate = False` (`cat_water_a.py:289`). Impulses are not saturated.

Generalised force of body forces F = (F_x, F_y, F_z) at hull point (x_p, y_p, z_p)
(`lever_Q`, `cat_base.py:298-311`):

  Q = ( F_x,  F_y,  x_p F_y - y_p F_x,  F_z + s_p th F_x,  s_p (x_p F_z - z_p F_x) )   (G1)

(the heave row is the earth-vertical force, the pitch row in the lofi th convention).

Items with `gated = True` add the (w, q) part of their SATURATED a_item to
a_cat = (a_cat,z, a_cat,th) (`cat_base.py:1046-1047`), which W4 reads. Gated items among all codes:
W1, W2, W5, W8, W11, H1, B1, E2, E4 (`cat_w1.py:55`, `cat_water_a.py:110,353`, `cat_water_b.py:241,382`,
`cat_hidden_boat.py:117,521`, `cat_air_env.py:527,691`).

### 1.4 From the substep acceleration to the recorded label

- Substep total: a_sub = sum_items a_item + a_L1 + a_prop + a_W4 + a_L3 + a_cor (`cat_base.py:1078-1102`).
- Applied push per substep: p = clip(a_sub + rule + noise, +-CLIP*A_REF) with CLIP*A_REF =
  (12, 12, 6, 36, 9) (`operators.py:76-82`, `CatInjector.__call__` `operators_cat.py:711-712`);
  rule = 0 (T8 off), noise = layer 1's old T9 noise held over the control step.
- Velocity update after the lofi substep: s_vel += p dt; then the impulse channel
  s_vel += imp_clipped with sum over the control step clipped to +-CLIP*A_REF*0.24 s
  (`operators_cat.py:719-728`).
- One control step = 6 substeps of 0.04 s (dtc = 0.24 s). Label recorded per control step k
  (`episode8.py:249-257`): y_k = (x^plant_{k+1} - x^twin_{k+1}) / 0.24 s in the 5 velocity channels,
  where x^twin is the lofi plant stepped from the same x_k with the same commands. So an item enters the
  label through 6 substeps of injection integrated through the lofi dynamics, not as a_item itself.
- Network inputs per step: s26 = (9 measured state features, 2 heave / pitch offsets, 15 station
  elevations) from the TRUE plant state (`episode6.py:151-162`) and the commands (`episode8.py:260`).
  None of the hidden states below (phi, slosh, bilge water, wind, current, depth, packet, loading) is an
  input. The section-11 signals (roll, wind, speed through water, ...) are STORED in `OBS`
  (`episode8.py:271`), not fed to the network (episode7 docstring `episode7.py:33-40`; s26 does not use
  them). How long a history window the network gets is set by the training code: UNKNOWN here (not
  read for this document).

---------------------------------------------------------------------------------------------------

## 2. Hidden degrees of freedom

### 2.1 H1 hidden roll (`cat_hidden_boat.py:102-275`)

Stage 'hidden', `gated = True` (`:116-117`), saturated with n = a_ref. On w.p. 0.7 (prior marginal 0.603).

**Hidden state** (created): phi (rad, > 0 raises the PORT side), phid = d phi/dt (rad/s),
n (unit-variance Ornstein-Uhlenbeck noise), init flag (`:170-173`). Process stream per mission.

**Parameters and coded priors** (`draw_on`, `:120-168`):

| symbol | key | prior | unit |
|---|---|---|---|
| omega0 | `w0` | LogU[1.5, 5.5] * sqrt(2.03 / B) (= [1.5018, 5.5066] for B = 2.0252) | rad/s |
| k_x | `kx` | U[0.35, 0.45] B | m |
| I_x | `I_x` | m k_x^2 (m = ctx.m after B1) | kg m^2 |
| GM | `GM` | omega0^2 k_x^2 / g | m |
| persist | `persist` | Bern(0.15/0.7 = 0.2143) | |
| U_c (persist only) | `U_c` | U[u_lo, u_max] | m/s |
| kappa | `kap` | persist: (u_max/U_c)^2 in [1, 8.16]; else U[0, 0.5] | - |
| phi_l (persist only) | `phi_l` | U[2, 8] deg | rad |
| c3 | `c3` | persist: omega0^2 max(kappa - 1, 0.25) / phi_l^2; else 0 | 1/(rad^2 s^2) |
| zeta | `zeta` | LogU[0.05, 0.4] | - |
| phi_b | `phi_b` | N(0, (2 deg)^2) + phi_load | rad |
| phi_load | `phi_load` | B1 on: -10 deg * tanh( dy_g / GM / (10 deg) ); else 0 | rad |
| c_turn | `turn` | LogU[0.05, 0.5], sign - w.p. 0.8 (heel into the turn), + w.p. 0.2 | - |
| c_w | `c_w` | U[0, 1] | - |
| c_d | `c_d` | U[0, 0.5] | - |
| c_K | `c_K` | U[0.5, 1.5] | - |
| phi_Q | `phi_Q` | U[0, 3] deg x (+1 or -1 w.p. 0.5) | rad |
| a1, a3 | | a1 ~ U[-0.3, 1.0], a3 = U[-1, 1] a1 | - |
| l1, l3 | | l1 ~ U[-0.15, 0.15] L, l3 = U[-1, 1] l1 | m |
| n_p | | U[-0.05, 0.05] | - |
| c2 | | U[0, 1] | - |
| c_x | `cx` | U[0, 2] | - |
| x_tf | | U[-0.15, 0.15] L | m |
| e_phi | | U[0, 20] | 1/rad^2 |
| cw (chine walk) | `cw` | Bern(0.2/0.7 = 0.2857) | |
| gamma, U_cw, w_U, phi_sat (cw only; else 0, 0, 1, 1) | `gam, U_cw, w_U, phi_sat` | U[0, 0.3]; U[25, 50] kn; U[1, 5] kn; LogU[2, 10] deg | |
| phi_n | | LogU[0.05, 0.5] deg | rad |
| tau_n | | LogU[0.3, 3] | s |

**Dynamics** (exact statement of `step`, `:192-249`). With U = U_r (water-relative speed):

  k_lin(U) = omega0^2 (1 - kappa (U/u_max)^2)                                          (`:176-177`)
  Fy0 = k_js max(thr, 0) sin(clip(noz, +-delta_stall))   (the lofi jet side force)         (`:183-184`)
  phi_eq = phi_b + c_turn u_r r / g + c_w t_slope + c_d Fy0 / W + phi_Q thr / T_max        (`:185-187`)
  F = omega0^2 phi_eq + c_K K_roll / I_x + omega0^2 phi_n n                                  (`:189-190`)
  K_roll = roll moment put in q by A1 / A2 (N m, > 0 raises port; 0 if absent)               (`:188`)
  omega_d = max( sqrt(abs(k_lin)), 0.3 omega0 )                                                (`:213-214`)
  s_g = sigma((U - U_cw)/w_U) if cw else 0                                                   (`:215-217`)
  zeta_eff(phi) = zeta - gamma s_g (1 - phi^2/phi_sat^2)                                      (`:224-225`)

The continuous model is

  phi'' = F - k_lin phi - c3 phi^3 - 2 zeta_eff(phi) omega_d phi',   abs(phi) <= 0.5 rad (inelastic stop)

and the code integrates it over the 0.04 s substep with n_in = clip(ceil(rate*dt/0.3), 1, 8) inner steps,
rate = max over rows of max(sqrt(abs(k_t)), 0.5 abs(c)) at the substep start (`:228-231`). In each inner
step the cubic is linearised at the current phi_1 (Newton form):
k_t = k_lin + 3 c3 phi_1^2, F_t = F + 2 c3 phi_1^3, c = 2 zeta_eff(phi_1) omega_d (`:219-226`), and
the step is the exact 2x2 transition of e'' = -k_t e - c e' about x_eq = F_t/k_t (`osc_step`,
`cat_base.py:264-273`, `osc_phi` `cat_base.py:230-261`), or a velocity kick F_t dt when
abs(k_t) <= 1e-6 omega0^2 (`:235-238`). After each inner step abs(phi) is clipped at 0.5 rad and an
outward phid is set to 0 (`:239-241`). Over the substep F, k_lin, omega_d, s_g are held at their start
values. Noise: n <- e^{-dt/tau_n} n + sqrt(1 - e^{-2dt/tau_n}) N(0,1) (`cat_base.py:204-208`, `:242`).

Initialisation at the first substep (`:200-211`): stable rows (k_lin > 1e-6 omega0^2) start at
phi = F / k_lin (the cubic is ignored here), bistable rows at +-sqrt(-k_lin/c3) with a random side;
phid = 0.

Consequences of the priors: non-persist draws have k_lin >= 0.5 omega0^2 at U <= u_max and, in the task
range U <= u_hi, k_lin >= (1 - 0.5 * 0.4307) omega0^2 = 0.785 omega0^2 (static-heel amplification
1/(1 - kappa (U/u_max)^2) <= 2, checked at `:698-711`). Persist draws have k_lin < 0 for U > U_c: two
stable heels +-sqrt(-k_lin/c3). Chine walk: zeta_eff < 0 near phi = 0 when gamma s_g > zeta, which gives
a limit cycle bounded by phi_sat (check `:785-801`).

**Outputs** at the substep START values phi0, phid0 (`:244-249`):

  X_phi  = - c_x k_d u_r abs(u_r) phi0^2                                                     (`:256`)
  Fz_phi = - c2 W phi0^2                                                                      (`:257`)
  Q_H1 = (G1) at (x_tf, 0, 0) with F = (X_phi, 0, Fz_phi)
         + (0, -W (a1 phi0 + a3 phi0 abs(phi0)), W (l1 phi0 + l3 phi0 abs(phi0)) + n_p L W phid0 / omega0, 0, 0)   (`:258-263`)
       = ( X_phi, Y_phi, N_phi, Fz_phi + s_p th X_phi, s_p x_tf Fz_phi )
  acc_H1 = (0, A_v, A_r, 0, 0),  e = e_phi phi0^2,                                              (`:266-275`)
     A_v = LD( e (k_lin_sway + k_sway abs(v_r)) / m_v, v_r ),   A_r = LD( e / tau_r, r )
  LD(c, x) = -x (1 - e^{-c dt}) / dt for c >= 0 (`lin_damp_acc`, `cat_base.py:194-201`)

Parity: sway and yaw parts odd in phi, surge / heave / pitch parts even (check `:823-836`).
Channels: u, v, r, w, q (all five; heave and pitch through c2 and the lever x_tf).

**Interfaces.** Writes q['phi'] = phi0, q['phid'] = phid0, q['phidd'] = (phid1 - phid0)/dt and
q['H1'] = {state, I_x} (`:244-246`). Through the re-derive after the 'hidden' stage, phi changes the
immersion of the port / stbd samples by -+(B/2) phi0 and their entry velocity by -+(B/2) phid0
(`cat_base.py:537-543`); the centre-line samples (hc) and the intake (its y = 0) are unchanged. Reads
K_roll from A1 / A2, ctx.load from B1. Receives roll moments from H2 and B2 via
`roll_kick(q, K, dt)`: phid += K dt / I_x (`:66-74`), felt from the next substep (no c_K factor).
Stores obs roll = phi0, roll_rate = phid0 (not a network input).

**Function class.** For non-persist, non-cw draws at fixed U: phi is the output of a linear,
time-invariant 2nd-order low-pass (natural frequency sqrt(k_lin), damping 2 zeta sqrt(k_lin)) driven by
a linear combination of (u_r r, t_slope, Fy0(thr, noz), thr, K_roll, coloured noise) plus a constant;
the error is a polynomial of degree <= 2 in (phi, phid) times known factors (W, u_r abs(u_r), v_r, r).
U enters k_lin (a speed-scheduled filter). Persist and cw draws make the filter nonlinear (bistable /
limit cycle). Decay times 1/(zeta omega_d) of non-persist, non-cw draws in the task speed range span
1/(0.4 * 5.507) = 0.45 s to 1/(0.05 * sqrt(0.785) * 1.502) = 15 s, so the error depends on the input
history over up to tens of control steps, not on the current input alone.

### 2.2 H2 fuel sloshing (`cat_hidden_boat.py:278-375`)

Stage 'force', not gated, `calm_zero = True` (`:298-299`). On w.p. 0.4 (marginal 0.348).

**Parameters** (`draw_on`, `:301-324`): number of tanks n_t = 1 w.p. 0.6 else 2; total fuel share
f ~ LogU[0.02, 0.15]; tank weights w ~ Dirichlet(1,1) (n_t = 2) or 1; lev = 1 + eps_m (B1's eps, or 1).
Per tank i:

| symbol | key | prior / formula | unit |
|---|---|---|---|
| a_x,i, a_y,i (tank lengths along x, y) | `ax, ay` | U[0.4, 1.5], U[0.3, 1.2] | m |
| h_i (liquid depth) | `h` | clip(U[0.05, 0.8] lev, 0.03, 0.9) a_x,i | m |
| m_liq,i | `m_liq` | f w_i m | kg |
| omega_d,i^2, m_1d,i (d in x, y) | `w2x, w2y, m1x, m1y` | Dodge first mode: omega^2 = (pi g / a_d) tanh(pi h / a_d); m_1 = m_liq 8 a_d tanh(pi h / a_d) / (pi^3 h) (`:279-284`) | rad^2/s^2, kg |
| zeta_x,i, zeta_y,i | `zx, zy` | LogU[0.005, 0.15] each | - |
| x_t,i | `x_t` | U[-0.3, 0.2] L | m |
| z_t,i | `z_t` | U[-0.2, 0.3] | m |

**Hidden state**: s_x,i, sd_x,i, s_y,i, sd_y,i (slosh-mass displacement relative to the tank, m;
start 0) (`:326-329`).

**Dynamics** (`:342-361`), per tank and direction:

  s'' + 2 zeta omega s' + omega^2 s = -a_t,    abs(s) <= a_d / 4 (inelastic wall, `:336-339`)
  a_t,x = (nu0_u - r v) + g sin(theta_rel) - z_t s_p nu0_q                                   (`:352`)
  a_t,y = (nu0_v + u r) + x_t nu0_r + g sin(phi) - z_t phidd                                 (`:353-354`)

theta_rel = pitch relative to the lofi running attitude (the hidden trim tau0 is NOT included);
phi, phidd from H1 (0 if H1 off). Integrated by the exact 2x2 transition with x_eq = -a_t/omega^2 held
over the substep (`:332-335`).

**Outputs** (`:362-375`), sdd = (sd1 - sd0)/dt, sbar = (s0 + s1)/2:

  F_x,i = -m_1x,i sdd_x,i,  F_y,i = -m_1y,i sdd_y,i,   Q_H2 = sum_i (G1) at (x_t,i, 0, z_t,i) with (F_x,i, F_y,i, 0)
  Q_H2,q += s_p sum_i ( -m_1x,i g sbar_x,i )      (moment of the moved weight)
  roll moment to H1: K = sum_i ( -m_1y,i g sbar_y,i + m_1y,i z_t,i sdd_y,i )  -> roll_kick

Channels u, v, r (x_t F_y), w (s_p th F_x only), q. Calm steady straight running: a_t = 0, s = 0, output 0.

**Function class**: a bank of linear 2nd-order filters (frequencies from the Dodge formula) driven by
the lofi accelerations nu0, the centripetal terms r v, u r, gravity along the relative pitch / hidden
roll; output = mass x filter acceleration (high-pass) plus mass x filter displacement (low-pass), with a
saturating wall nonlinearity.

### 2.3 H3 bilge / green water (`cat_hidden_boat.py:378-441`)

Stage 'force', not gated, not calm-zero. On w.p. 0.15 (marginal 0.135).

**Parameters** (`:394-406`): x_aft = x_1 + 0.03 L; x_fwd = x_1 + U[0.3, 0.7] L;
area = (x_fwd - x_aft) B / 2; m_0 ~ LogU[0.002, 0.05] m (kg); lambda_g ~ LogU[0.2, 2] 1/s;
tau_d ~ LogU[5, 300] s; leak depth rate LogU[0.001, 0.05] m/h -> leak_kg = rho_w area leak (kg/s);
c_f ~ LogU[0.5, 5] (1/m); m_cap = 0.1 m.

**Hidden state**: water mass m_b (start m_0), position x_b (start x_aft), velocity xd_b (start 0) (`:408-411`).

**Dynamics** (`:413-435`):

  green water: Poisson events at rate lambda_g * [H_b < 0]; each adds LogU[5, 100] kg      (`:418-420`)
  m_b <- min( x_inf + (m_b - x_inf) e^{-dt/tau_d} + adds, m_cap ),  x_inf = leak_kg tau_d   (`:423-425`)
  theta_abs = tau0 + theta_rel (absolute running trim, bow-up)                              (`:426`)
  z_b = keel(x_b) + 0.05 m;  (a_x,pt, a_z,pt) at (x_b, 0, z_b) from nu0 (`_point_acc` `:81-89`):
     a_x,pt = nu0_u - r v - z_b s_p nu0_q,   a_z,pt = nu0_w + s_p x_b nu0_q
  xd_b' = -a_x,pt - g sin(theta_abs) - c_f xd_b abs(xd_b)   (explicit gravity / inertia kick, then
     linearised implicit quadratic friction `implicit_quad_drag`, `cat_base.py:276-283`), x_b in
     [x_aft, x_fwd] with inelastic stops                                                    (`:429-434`)

**Outputs** (`:436-441`), with m_b0 = start-of-substep mass and xdd = (xd1 - xd0)/dt:

  F_x = -m_b0 (a_x,pt + xdd) - m_b0 g sin(theta_abs),   F_z = -m_b0 (g + a_z,pt),
  Q_H3 = (G1) at (x_b0, 0, z_b) with (F_x, 0, F_z)

Channels u, w, q (r and v zero). Not zero in calm water: a static weight -m_b g and surge
-m_b g sin(tau0) (check `:895-928`), i.e. a slowly varying offset in heave / pitch / surge.

---------------------------------------------------------------------------------------------------

## 3. The boat's own changes

### 3.1 B1 loading, merged with the layer-3 inertia (`cat_hidden_boat.py:444-584`)

Stage 'force', `gated = True` (`:520-521`), saturated with n = a_ref (the Coriolis part below is inside
the saturation). On w.p. 0.8 (marginal 0.688). Drawn before every other item (`cat_base.py:953`).
No hidden state.

**Parameters** (`draw_on`, `:523-531`): eps ~ U[-0.1, 0.25] (relative mass change);
dx_g ~ U[-0.03, 0.03] L; dy_g ~ U[-0.03, 0.03] B (m); a_A ~ U[0, 1]; a_R ~ U[0.5, 1.5];
b_x ~ U[0.5, 1.5] deg / 0.03 (rad per unit dx_g/L); b_m, d_m, e_m ~ U[0, 1] each.
Derived (`build`, `:534-542`): dth0 = -b_x dx_g / L + b_m eps tau0 (bow-up trim change, CG aft -> bow up),
dz0 = -d_m eps T.

**Inertia** (`b1_inertia`, `:471-489`). Base = the layer-3 rigid body (m, x_g, y_g, z_g, I_xz, I_zz, I_yy)
and M_base = ctx.Mt (layer 3; drawn in `operators_gen.py:283-315`). With
M_RB(m, x_g, y_g, z_g, I_zz, I_yy) = diag(m, m, I_zz, m, I_yy) plus symmetric (u,r) = -m y_g,
(u,q) = m z_g, (v,r) = m x_g, (w,q) = -m x_g (`:445-452`):

  M_A = M_base - M_RB(base);  r_zz^2 = I_zz/m - x_g^2 - y_g^2;  r_yy^2 = I_yy/m - x_g^2 - z_g^2
  m1 = m (1+eps), x_g1 = x_g + dx_g, y_g1 = y_g + dy_g, z_g unchanged
  I_zz1 = m1 (r_zz^2 + x_g1^2 + y_g1^2),  I_yy1 = m1 (r_yy^2 + x_g1^2 + z_g^2)
  I_xz1 = (I_xz - m x_g z_g)(1+eps) + m1 x_g1 z_g
  M_new = M_RB(m1, x_g1, y_g1, z_g, I_zz1, I_yy1) + M_A (1 + a_A eps)

`adjust_ctx` (`:544-554`): ctx.Mt <- M_new (shared by every item's M_t^-1 and by W4), ctx.m, ctx.W,
ctx.vol multiplied by (1+eps), ctx.load = {eps_m = eps, dxg, dyg, dxg/L, dyg/B}.

**Output** (`step`, `:556-584`). Heave / pitch restoring of the loaded boat, f = 1 - e_m eps:

  hp_z  = -( (omega_h f)^2 - omega_h^2 ) e_z  + (omega_h f)^2 dz0       + LD( 2 zeta_h omega_h (f - 1), zd )
  hp_th = -( (omega_p f)^2 - omega_p^2 ) e_th + (omega_p f)^2 s_p dth0  + LD( 2 zeta_p omega_p (f - 1), thd )

(LD as in H1; for c < 0, LD = -c x, explicit; `cat_base.py:194-201`). With
dM' = M_new - M_base with its (w,w) and (q,q) entries set to 0, nut = nu0 + (0,0,0,hp_z,hp_th):

  a_B1 = M_new^-1 [ -dM' nut + (-a_R eps k_d u_r abs(u_r), 0, 0, 0, 0) ] + (0,0,0,hp_z,hp_th)
         + c(s_r; rb_new, M_new) - c(s_r; rb_base, M_base)

with the rigid-body Coriolis difference of `operators_gen.py:833-851`,
c(s_r; rb, M) = -M^-1 C, w = zd + u th, q = thd, c1 = q w - r v,
C = ( m(c1 - (q^2+r^2) x_g), m(r u + r q z_g - r^2 y_g) - m_0 u r, I_xz q r + m(x_g r u - y_g c1),
m(-q^2 z_g + q r y_g), -I_xz r^2 + m(z_g c1 + x_g q u) ), m_0 = m_coriolis = 1400 kg
(`rigid_coriolis`, `:495-500`). Layer 3 already adds c(s_r; rb_base, M_base) (`cat_base.py:1101`), so
with B1 on the total Coriolis is c(s_r; rb_new, M_new).

Effect of dy_g: H1's phi_b through ctx.load, M_t couplings (u,r), and the y_g terms of C. Effect of
dx_g: M_t couplings (v,r), (w,q), I_zz, I_yy, the trim change dth0, the x_g terms of C.
Channels: all five. Calm steady running at the lofi attitude: hp = w1^2 * offsets != 0 when eps or
dx_g != 0 (a static attitude change), i.e. B1 is not calm-zero.

**Function class**: an affine function of (nu0, e_z, e_th, zd, thd, u_r abs(u_r)) plus a quadratic form
in the velocities (Coriolis), constant coefficients per boat. No memory beyond what nu0 carries.

### 3.2 B2 discrete load events (`cat_hidden_boat.py:587-672`)

Stage 'force', not gated. Item-level P_ON = 1 - 0.95*0.97*0.97 = 0.10614 (marginal 0.098). Given the
item is on, the three sub-parts are drawn Bern(0.05), Bern(0.03), Bern(0.03) and redrawn until at least
one is on (`:600-608`).

**Hidden state** (`:628-631`): CG offsets dx, dy (m, start 0); mass-step flag (start False); tow on/off
(start Bern(0.5)). All restart each mission (new state per mission, section 8).

(a) CG jumps (`:610-614, 639-648`): Poisson rate LogU[1/3600, 1/60] 1/s; increments
s_x t_df, s_y t_df with s_x ~ LogU[0.005, 0.03] L, s_y ~ LogU[0.01, 0.05] B, df ~ U[2, 5];
dx clipped to +-0.05 L, dy to +-0.1 B. Output Q_q += s_p dx (-W); roll moment -W dy -> roll_kick (H1).

(b) one point-mass step (`:615-619, 649-657`): rate LogU[1/300, 1/20] 1/s, at most once per mission;
dm ~ U[-0.1, 0.1] m, at (x_m, 0, z_m), x_m ~ U[-0.3, 0.3] L, z_m ~ U[0, 0.5] m. After the event:
F = ( -dm (a_x,pt + g sin(theta_abs)), 0, -dm (g + a_z,pt) ) at (x_m, 0, z_m) through (G1);
theta_abs = tau0 + theta_rel; (a_x,pt, a_z,pt) from nu0 as in H3. M_t is not changed.

(c) towed drag (`:620-625, 658-671`): c_t = LogU[0.02, 0.3] k_d; two-state Markov switch with mean
dwell on / off LogU[10, 300] s each (flip w.p. 1 - e^{-dt/dwell}); x_tow = x_1 (transom);
psi_off ~ U[-10, 10] deg. V_x = u_r, V_y = v_r + r x_tow, V = sqrt(V_x^2 + V_y^2); unit vector
e = R(psi_off) (V_x, V_y)/V; g_e = (G1) of unit force e at (x_tow, 0, 0);
m_pt = 1 / (g_e^T M_t^-1 g_e); Q += [on] m_pt * ( -c_t V^2 / (m_pt + 2 dt c_t V) ) g_e.

### 3.3 B3 fouling (`cat_hidden_boat.py:675-694`)

Stage 'force'. On w.p. 0.3 (marginal 0.263). phi_F ~ U[0.3, 0.6], D_f ~ LogU[0.05, 0.8];
c = phi_F k_d D_f; m_pt = 1 / (M_t^-1)_{uu};
Q_u = m_pt * ( -c u_r abs(u_r) / (m_pt + 2 dt c abs(u_r)) ) (linearised implicit quadratic drag).
Only channel u through Q, other channels through the off-diagonal entries of M_t^-1.
Function class: a fixed multiple of the lofi drag shape u_r abs(u_r).

---------------------------------------------------------------------------------------------------

## 4. Air

### 4.1 A1 mean wind load (`cat_air_env.py:144-255`)

Stage 'env'. On w.p. 0.7 (marginal 0.611).

**Parameters** (`draw_on`, `:196-220`): z_w ~ U[1, 2] m (boat height), z_r0 ~ LogU[1e-4, 1e-2] m
(roughness), zr = ln(z_w/z_r0)/ln(10/z_r0) (log profile ratio, `:203`); f_air ~ LogU[0.03, 0.3]
(still-air drag share of the lofi drag); C_A = 2 f_air k_d / rho_a (= C_X(0) A_F, `:200`);
C_Dl ~ U[0.45, 0.95]; A_F = C_A / C_Dl; A_L = A_F U[1.5, 3.5]; U10 ~ U[0, 15] m/s;
wave_rel ~ Bern(0.7); dpsi ~ U[-30, 30] deg; psi_u ~ U[0, 2 pi); tau_d ~ LogU[300, 3000] s;
z_a ~ U[0.5, 1.5] m (heel lever). Shape: w.p. 0.5 Blendermann with C_Dt ~ U[0.7, 1.0],
delta ~ U[0.1, 0.8], stern ~ U[0.9, 1.2], s_L ~ U[-0.15, 0.15]; else Fourier with
a = C_Dl (1, U[-.3,.3], U[-.3,.3]) / sum, b = b1 (1, U[-.3,.3], U[-.3,.3]), b1 ~ U[0.7, 1],
c = b1 (U[-.15,.15], U[-.1,.3], U[-.05,.05]), asym = U[-1,1]^3 w.p. 0.2 else 0.

**Hidden state** (`:222-225`): x in R^2 (unit-variance OU, start N(0, I)), psi0 (set at the first step).

**Mean wind** (`mean_wind`, `:227-237`): psi0 = wave_dir + dpsi if wave_rel and the rows' mean wave
direction is bound (energy-weighted mean propagation direction, `bind_wave_dir` `:184-189`, bound by
`CatInjector` at its first call, `operators_cat.py:701-703`), else psi_u (+ dpsi if wave_rel);
U_w = U10 zr max(1 + 0.1 x_1, 0), psi_w = psi0 + 10 deg x_2;
V = U_w (cos psi_w, sin psi_w) (earth velocity of the air). x advances by the exact OU step with
tau_d AFTER V is used (`:241-242`).

**Load** (`wind_load` `:172-181`, `wind_coeffs` `:145-169`): air velocity relative to the boat in body
axes (w_x, w_y) = R(psi)^T V - (u, v) (ground velocity, not water-relative); q_a = rho_a (w_x^2+w_y^2)/2;
gamma = atan2(-w_y, -w_x) (0 = air from ahead, > 0 from port).

  F_x = q_a A_F C_X(gamma),  F_y = -q_a A_L C_Y(gamma),  N = -q_a A_L L C_N(gamma)

Blendermann: C_DlAF = C_Dl (abs(gamma) <= pi/2) or C_Dl * stern; C_Dl' = C_DlAF A_F / A_L;
den = 1 - delta/2 (1 - C_Dl'/C_Dt) sin^2(2 abs(gamma)); C_X = -C_DlAF cos(abs(gamma))/den;
C_Y = sign(gamma) C_Dt sin(abs(gamma))/den; C_N = (s_L - 0.18 (abs(gamma) - pi/2)) C_Y.
Fourier: C_X = -sum_k a_k cos(k gamma) + 0.05 a_1 asym_1 sin gamma;
C_Y = sum_k b_k sin(k gamma) + 0.05 b_1 asym_2 cos gamma;
C_N = sum_k c_k sin(k gamma) + 0.005 b_1 asym_3 cos gamma (k = 1..3).

**Output** (`:243-255`): Q_A1 = (F_x(V) - F_x(0), F_y(V) - F_y(0), N(V) - N(0), 0, 0) ("with wind minus
without wind": the still-air drag of the boat is in k_d). K_roll -= z_a F_y(V) (written to q for H1).
Writes q['wind'] = {U10, zr, psi0, Vmean = V, Vtot = V, w_g = 0, z_a, load(.)}; obs relative wind.
Channels u, v, r.

**Function class**: quadratic in the relative air velocity times a smooth periodic function of the
relative angle; depends on psi (heading) and on a hidden, slowly drifting earth vector V (within a 90 s
mission nearly constant because tau_d >= 300 s).

### 4.2 A2 gusts and squalls (`cat_air_env.py:258-338`)

Stage 'env', only when A1 is on (`DEPENDS`), then P_ON = 1 (marginal 0.603).

**Parameters** (`:263-269`): I_u ~ U[0.08, 0.2]; r_v ~ U[0.75, 0.8]; r_w = 0.5; L_u ~ LogU[3, 200] m;
dryden ~ Bern(0.3); squall rate lambda_s ~ LogU[1/7200, 1/900] 1/s; tau_s ~ LogU[5, 60] s.
OU-sum constants (von Karman fit): beta = (1, 0.2, 0.04, 0.008), weights
w = (0.48087, 0.14056, 0.01684, 0.01621)/sum (`:92-94`); Dryden: one pole, beta = 1, w = 1.

**Hidden state** (`:271-278`): x in R^{3 x n} (OU components, n = 4 or 1), squall flag / remaining time /
targets (dU*, dpsi*), lagged (dU, dpsi).

**Dynamics** (`:286-329`):
- squall onset Poisson(lambda_s) when none active: dU* = U[0.3, 1] U10 zr, dpsi* = U[-90, 90] deg,
  duration U[600, 3600] s; afterwards targets 0; dU, dpsi follow exact first-order lags tau_s.
- spd = abs(Vmean) + dU, ang = arg(Vmean) + dpsi, e1 = (cos ang, sin ang), e2 = (-sin ang, cos ang),
  Vm2 = spd e1.
- omega_c = max(V_rel, 0.5)/L_u with V_rel = air speed relative to the boat for Vm2;
  each OU component with tau_i = beta_i / omega_c; g_k = sum_i sqrt(w_i) x_{k,i}.
- gust: g_u = I_u spd g_1, g_v = r_v I_u spd g_2, g_w = r_w I_u spd g_3;
  Vtot = Vm2 + g_u e1 + g_v e2.

**Output**: Q_A2 = load(Vtot) - load(Vmean) (A1's load function; includes the squall),
K_roll -= z_a Q_A2,v; q['wind'] Vtot, w_g = g_w (vertical gust, read by A3). Channels u, v, r.

### 4.3 A3 aerodynamic lift and pitch moment (`cat_air_env.py:341-413`)

Stage 'force', not gated, not calm-zero (draft section 10 item 18: not subtracted at the calibration
state). On w.p. 0.6 (marginal 0.518).

**Parameters** (`draw_on`, `:359-389`): kappa_th ~ LogU[0.01, 0.3] (aero / lofi pitch stiffness at u_max),
lambda_L ~ LogU[0.003, 0.08] (lift / W at u_max), r_v ~ U[0, 3] (Polhamus vortex / potential ratio),
x_1 ~ U[0, 0.15] L, a_0 ~ U[-0.03, 0.1] rad, c_g ~ U[0, 1], l_g ~ U[0.2, 1] m. Derived at the
reference state (calm, no wind, U = u_max, h_b,ref = max(F_b + z0 + s_p x_5 th0, 0) = 1.067 m,
q_ref = rho_a u_max^2 / 2; `ref_state` `:354-357`), alpha_ref = a_0 + tau0, G_ref = 1 + c_g e^{-h_b,ref/l_g}:

  S K_p = lambda_L W / ( q_ref C(alpha_ref) G_ref ),   A = q_ref S K_p G_ref
  x_0 = ( kappa_th omega_p^2 I_q / A + x_1 cos(alpha_ref) C(alpha_ref) ) / C'(alpha_ref) + x_1 sin(alpha_ref)

I_q = (M_t)_{qq} at draw time (after B1). Accepted when x_0 in [0.05, 0.35] L; up to 200 redraws,
then the FIRST draw is kept with x_0 clipped to the interval (`:384-389`).

**Output** (`forces` `:391-406`, `step` `:408-413`): w_x = body-x component of Vtot (0 without A1);
V_x = max(u - w_x, 0); q_x = rho_a V_x^2 / 2;
alpha = clip( a_0 + tau0 + theta_rel + (w_g - (zd + s_p x_0 thd)) / max(V_x, 1), +-pi/4 );
G = 1 + c_g e^{-max(H_b,0)/l_g}; C(alpha) = sin a cos^2 a + r_v cos a sin a abs(sin a) (`:342-345`);

  F_z = q_x S K_p C(alpha) G,  F_x = -F_z tan(alpha),  x_ac = x_0 - x_1 sin(alpha),
  Q_A3 = (G1) at (x_ac, 0, 0) with (F_x, 0, F_z)

Channels u, w, q. At calm steady running the output is q_x-dependent and non-zero: a speed-dependent
static lift / moment offset, with a destabilising (bow-up) pitch stiffness (check `:884-930`).

---------------------------------------------------------------------------------------------------

## 5. Environment

### 5.1 E1 current (`cat_air_env.py:416-509`)

Stage 'env'. On w.p. 0.6 (marginal 0.518).

**Parameters** (`:421-429`): wind_mode ~ Bern(0.5); k_w ~ U[0.015, 0.03]; dpsi ~ U[-30, 30] deg;
U10_fb ~ U[0, 15], psi_fb ~ U[0, 2 pi) (used when wind_mode and A1 off); V_tid ~ LogU[0.1, 2] m/s,
psi_t ~ U[0, 2 pi); tau_d ~ LogU[300, 3000] s; front spacing s_f ~ LogU[200, 5000] m; front width
w_f ~ LogU[5, 50] m.

**Hidden state** (`:431-435`): Vbar (earth mean current, set at the first step), x in R^2 (OU),
tgt, cur in R^2 (front offset target / value), V (current value).

**Dynamics** (`:437-484`):
- Vbar = V_tid (cos psi_t, sin psi_t) (tidal), or k_w U10 (cos(psi0_wind + dpsi), sin(.)) with A1's U10,
  psi0 (wind mode; fallback U10_fb, psi_fb + dpsi).
- V = Vbar + 0.2 abs(Vbar) x + cur (`:452-454`); x: exact OU step in time, tau_d.
- fronts: Poisson in distance ds = U_g dt at rate 1/s_f; jump J = abs(Vbar) U[0,1] (cos ph, sin ph),
  tgt <- (tgt + J) min(1, abs(Vbar)/abs(tgt + J)); cur follows tgt by an exact first-order lag with time
  constant w_f / max(U_g, 1e-3).

**Output** (`:485-509`), with (u_c, v_c) = R(psi)^T V (start value), (du_x, du_y) = R(psi)^T (V_new - V)/dt,
du_c = r v_c + du_x, dv_c = -r u_c + du_y, chi = psi + atan2(v, u) (course),
omega = cos(chi) d cur_y/ds - sin(chi) d cur_x/ds (front vorticity along the track):

  q['nu_c'] += (u_c, v_c)    (every later term reads u_r = u - u_c, v_r = v - v_c)
  acc_u = [ -k_d (u_r abs(u_r) - u abs(u)) + max(m_u - m, 0) du_c ] / m_u
  acc_v = [ lift_shift - k_lin_sway (v_r - v) - k_sway (v_r abs(v_r) - v abs(v)) + max(m_v - m, 0) dv_c ] / m_v
  acc_r = 0.5 omega / tau_r

lift_shift = k_lift (u_r abs(u_r) - u abs(u)) clip(noz) for a rudder boat; 0 here (steer_jet = 1).
m = ctx.m after B1. Channels u, v, r (directly) plus all channels through layer 1 / items reading u_r.
Obs stw_u = u_r, stw_v = v_r. The current changes forces only; position kinematics use u, v (ground).

**Function class**: the lofi drag / damping shapes evaluated at a shifted velocity (a hidden,
nearly constant 2-vector rotated by psi), plus small added-mass and vorticity terms at fronts.

### 5.2 E2 shallow water and banks (`cat_air_env.py:512-584`)

Stage 'force', `gated = True`. On w.p. 0.1 (marginal 0.093).

**Parameters** (`:529-542`): h_bar ~ LogU[max(1.5, 4T), 60] m (= LogU[1.539, 60]); s_h ~ U[0.1, 0.5];
l_h ~ LogU[100, 2000] m; a_h ~ U[-0.2, 0.3]; w ~ U[0.1, 0.4]; dz_s ~ U[-0.3, 0.3] T;
dth_s ~ U[-0.35, 0.15] tau0; dz_T ~ U[0, 0.3] T; dth_T ~ U[-0.35, 0.1] tau0; bank ~ Bern(E2_BANK_P = 0),
i.e. banks always off (`:99`); side +-1, d_bar ~ LogU[3, 50] m, s_d ~ U[0.2, 0.6], l_d ~ LogU[50, 1000] m,
c_y ~ LogU[0.002, 0.05], c_n = U[0.2, 1] c_y, l_b = U[1, 4] B (bank terms, inactive).

**Hidden state**: x_h, x_d (unit-variance OU in DISTANCE, length scales l_h, l_d), start N(0,1);
stepped with ds = U_g dt after the output (`:577-584`).

**Output** (`fields` `:550-554`, `terms` `:556-575`): h = max(h_bar e^{s_h x_h}, 2T);
F_h = abs(u_r)/sqrt(g h); x = (F_h - 1)/w; f = (2T/h)^2; S(x) = smoothstep (0 for x <= -1, 1 for
x >= 1, C^1 cubic between, `:513-516`); Bump(x) = max(1 - x^2, 0)^2 (`:519-521`):

  acc_u = -a_h Bump(x) k_d u_r abs(u_r) / m_u
  acc_w = omega_h^2 ( dz_s S(x) + dz_T f )
  acc_q = omega_p^2 s_p ( dth_s S(x) + dth_T f )
  (bank off: Q = 0)

Channels u, w, q. S = 1 and Bump = 0 exactly when x >= 1, i.e. h <= U_r^2 / (g (1 + w)^2): at
U_r = 8.231 m/s (16 kn) h <= 3.5 to 5.7 m, at U_r = 15.433 m/s (30 kn) h <= 12.4 to 20.1 m (w in
[0.1, 0.4]). There the item is a heave / pitch offset omega^2 (dz_s + dz_T f) that varies only through
the slowly varying h; where x <= -1 (h >= U_r^2 / (g (1 - w)^2)) only the dz_T f, dth_T f terms
remain, with f = (2T/h)^2 in (0, 1] (h >= 2T by the clip).

### 5.3 E3 debris impacts (`cat_air_env.py:587-637`)

Stage 'force', IMPULSE channel. On w.p. 0.3 (marginal 0.263).

**Parameters** (`:592-593`): rate ~ LogU[1e-4, 3e-3] 1/s; p_dmg = 0.3.
**Hidden state**: remaining impulse rem (B x 5), remaining duration rem_t, count.

**Per event** (`event`, `:601-614`): du = min(LogU[0.05, 1.5], 0.5 max(U_r, 0)) m/s; duration
d ~ LogU[0.02, 0.2] s; point x ~ U[0, x_5], y = clip(N(0, (0.25B)^2), +-B/2), z = keel(x);
direction e = normalise(-1, N(0, 0.2^2), U[0, 1]) (aft, small lateral, up); J = m du e;
dnu = M_t^-1 (G1)(J at (x, y, z)). Delivered linearly over d: each substep imp = rem min(dt, rem_t)/rem_t
(`:631-636`). Not saturated; the per-control-step sum is clipped at CLIP*A_REF*0.24 s (section 1.4).
q['debris_hit'] set w.p. 0.3 per event (`:627-630`) - read by nothing (section 9).
Expected events per 90 s mission: 0.009 to 0.27.

### 5.4 E4 wave packets (`cat_air_env.py:640-763`)

Stage 'env', `gated = True`. On w.p. 0.3 (marginal 0.263).

**Parameters** (`:693-696`): rate ~ LogU[1/1800, 1/120] 1/s; own ~ Bern(0.5) (own-wake trigger);
p_trig ~ U[0.2, 0.8].

**Hidden state** (`:698-704`): one packet per row (A, omega, k, beta, phase, t_k, sigma), item clock t
(starts 0 each mission), 61-sample heading / course ring at 1 s, armed flag.

**Packet births** (`:719-750`): Poisson(rate) when no packet active; with own: if the summed absolute
heading change over the last 60 s exceeds 150 deg and the row is armed, fire w.p. p_trig, direction =
course over ground 30-60 s ago. New packet (`new_packets`, `:707-717`): A = min(LogU[0.05, 0.6], lambda/14);
T_p ~ U[1.5, 5] s, lambda = g T_p^2/(2 pi), omega = 2 pi/T_p, k = omega^2/g; sigma = n T_p/4,
n ~ U{3..10}; phase ~ U[0, 2 pi); t_k = t + 3 sigma; beta ~ U[0, 2 pi) (or the wake direction).

**Packet field** (`packet_sea`, `:641-665`), tau = (t - t_k)/sigma, E = (e^{-tau^2/2} - e^{-4.5})/(1 - e^{-4.5})
for abs(tau) < 3 else 0, xi = X cos beta + Y sin beta (earth coordinate of each station):

  eta_p = A E cos(k xi - omega (t - t_k) + phase),  etad_p = A (E' cos(.) + E sin(.) (omega - k dxi/dt)),
  u_orb,p = A omega E cos(.) along beta (body components with beta - psi)

added to q['eta_add'], q['etad_add'], q['eta_in_add'], q['uorb'], q['vorb'] (`:752-758`), so every later
term (layer 1, W items, H1 via t_slope, H3 via H_b, W4 via hc) sees the packet; a_lofi and layer 3 read
the lofi sea eta0 only.

**Output** (`lofi_wave_response`, `:668-685`): the lofi boat's own response to the packet, which the
lofi itself never sees:

  acc_w = omega_h^2 k_wh mean_i eta_p,{i,c}
  acc_q = omega_p^2 k_wp s_p ( atan(slope0 + slope_p) - atan(slope0) )
  acc_v = c_wave_sway g mean(tsl_p) = 0,  acc_r = c_wave_yaw g (...) = 0   (both coefficients 0)

Expected packets per 90 s mission: 0.05 to 0.75 (plus own-wake ones).

---------------------------------------------------------------------------------------------------

## 6. W4 contact gate (`cat_water_a.py:274-323`)

Stage 'gate' (last item stage, after the propulsion chain), STRUCTURAL: always on, also in sparse draws
and in forces_off draws (`cat_base.py:108`, `operators_cat.py:212`). `saturate = False`,
`calm_zero = True` (`cat_water_a.py:289`). No hidden state.

**Parameters** (`:291-294`): h_off ~ U[-0.05, 0.05] m; w_d ~ U[0.02, 0.1] m; r_exit ~ U[0, 0.3]
(published as ctx.w4_r_exit for W1 / W7, `:296-297`; not used inside W4).

**Formula** (`step`, `:305-323`). Vertical hydrodynamic acceleration sum (B x 2):

  A = (g, 0) + a_lofi + a_L1 + a_cat
  a_lofi  = lofi heave / pitch acceleration (`cat_base.py:480-492`),
  a_L1    = layer-1 parts T2c, T3, T4, T7, Teven, rows (w, q) (`cat_base.py:131, 1074`),
  a_cat   = saturated (w, q) rows of the gated items run so far (W1, W2, W5, W8, W11, H1, B1, E2, E4).

Generalised heave force / pitch moment producing A: Q_hp = M_hp A with M_hp = ((M_t^-1)_{[w,q],[w,q]})^-1
(`_hp_mass`, `cat_water_a.py:64-67`). Weights w_i = max(h_run,i, 0) (all five positive here);
(alpha, beta) solves [[sum w, sum w x],[sum w x, sum w x^2]] (alpha, beta) = (Q_hp,z, s_p Q_hp,th);
f_i = w_i (alpha + beta x_i) (so sum f_i = Q_hp,z and s_p sum f_i x_i = Q_hp,th). Gate:

  chi_i = min( 1, sigma((hc_i - h_off)/w_d) / sigma((h_run,i - h_off)/w_d) )         (`:300-303`)
  Q_W4 = (0, 0, 0, sum_i (chi_i - 1) f_i, s_p sum_i (chi_i - 1) f_i x_i),   a_W4 = M_t^-1 Q_W4

chi_i = 1 whenever station i is at or below its calm-running immersion, so W4 only removes vertical
force when stations rise above it; fully airborne (all chi = 0): a_W4,(w,q) = -A (heave error
-g - a_lofi,z - a_L1,z - a_cat,z). Through the off-diagonal entries of M_t^-1 (B1 / layer-3 CG offsets)
a_W4 also has u, v, r components. Function class: a product of a per-station sigmoid of the centre-line
keel immersion and a linear functional of the other vertical terms.

---------------------------------------------------------------------------------------------------

## 7. Observation model O1-O5 (`learn/meta/cat_obs.py`)

**Is it on in D10a? No.**
- `OBS_ON_D10A = False` (`cat_base.py:157`); `ObsModel(on=None)` uses it (`cat_obs.py:823`) and then
  `observe` returns the truth sampled at the control steps (`ideal_obs`, `cat_obs.py:616-624, 840-842`).
- `CatDraw(obs_on=False)` default (`cat_base.py:893`); `OperatorCat` is built without `obs_on`
  (`episode7.py:80`, `operators_cat.py:183`).
- No pipeline imports `cat_obs`, `ObsModel` or `run_obs`: a repository-wide search finds them only in
  `cat_base.py`, `cat_obs.py`, `studies/test_cat_items.py`. The meta7 / meta8 data never pass through it.
- The `G12_off` switch that `obs_on=True` would add (`cat_base.py:917-918`) is not handled by
  `apply_l1_switches` (`cat_base.py:771-800`), so it would have no effect on layer 1 either.

What it would do if switched on (applied to a recorded trajectory, never to the dynamics,
`cat_obs.py:7-15`), stage order (`:75-77`), all items on w.p. 1 when the model is on (`:74`):
- O5 mount (`:253-270`, priors `:801-810`): antenna r_a = (U[-0.3,0.3] L, U[-0.2,0.2] B, U[0.3,2] m),
  IMU r_i = (U[-0.3,0.3] L, U[-0.2,0.2] B, U[-0.3,0.8] m), calibration error N(0, s^2), s ~ U[0.05,0.2] m,
  lever-arm compensation w.p. 0.7; point kinematics V_p = V + R (w x r_p),
  a_p = a + R_l (w' x r_p + w x (w x r_p)), specific force f_b = R_l^T (a_p + g e_z).
- O2 IMU (`:299-322`, priors `:707-729`): w_m = (1 + s_g) R_mis w + b_g(t) + G_s f + n_g;
  f_m = clip((1 + s_a) R_mis f + b_a(t) + n_a + a_vib, +-a_FS); bias random walks; vibration at engine
  orders through an anti-alias filter plus hull ringing.
- O3 attitude / heading / heave (`:325-346, 441-457, 481-501`, priors `:746-758`): tilt error
  kappa_a LPF_tau_a(a/g) + bias + noise; magnetic or dual-antenna heading; heave from GNSS or a
  2nd-order high-passed INS height.
- O1 GNSS (`:349-438`, priors `:675-684`): 5/10/20 Hz fixes, latency tau_g ~ U[0.02, 0.25] s,
  Gilbert-Elliott dropouts, RTK jumps, hold or INS bridge.
- O5 convert (`:460-478`), O4 output (`:504-578`, priors `:776-785`): per signal-group low-pass
  LogU[1, 20] Hz and delay U[0, 0.15] s, jitter, command offset tau_c ~ U[-0.05, 0.1] s, plain-sensor
  scale / noise; `e0_observed` (`:855-867`) would recompute the labels on the observed record.

---------------------------------------------------------------------------------------------------

## 8. Per-mission redraws (`learn/meta/episode8.py`)

A boat ("law") is one catalogue draw `OperatorCat(op_seed = 7 law + 1)` (`episode8.py:166-169`): which
items are on, every item parameter, layer 1 and layer 3, accepted with mission 0's environment. A
mission mi = 0..3 (meta_step8 default `--missions 4`, `studies/meta_step8.py:356`) of T = 90 s
(`T_EP`, `data2.py:87`; meta_step8 jobs set no T, `studies/meta_step8.py:217-221`) changes:

1. Sea, scenario, dithers: stream [law, mi, 8, 0x5EA] (mission 0: stream law) (`episode8.py:145-149`).
2. Mission seed (wave phases) law + 50000 + 10^6 mi (`:159`); operator noise stream [op_seed, 1, mi, 0x0B5] (`:176-177`).
3. `hid_seed` = 1000 + mi (0 for mission 0) (`:178`): every item's process stream
   [seed, 71, attempt, 500 + code index, hid_seed] and its fresh hidden-state start
   (`operators_cat.py:123-132`, `cat_base.py:991-993`). Hence ALL hidden states of section 2-6 restart per
   mission: H1 (phi from its static heel, noise), H2 (s = 0), H3 (m_b = m_0, x_b = x_aft), B2 (dx = dy = 0,
   mass step not yet happened, tow state Bern(0.5)), A1 OU x, A2 filters and squalls, E1 (Vbar re-derived,
   x, fronts), E2 (x_h, x_d), E3 (nothing pending), E4 (clock 0, no packet).
4. `mission_env` (`:79-101`), called BEFORE the operator state is built (`:179` vs `:230-233`): for each
   ON environment item c in ENV_KEYS (`:68-72`), a fresh `draw_on` with stream
   [op_seed, 0xE7, attempt, 900 + code index, mi] and the listed keys copied into the boat's parameters:

   | item | redrawn per mission | kept from the boat draw |
   |---|---|---|
   | A1 | U10, psi_u, wave_rel, dpsi, tau_d | zr (z_w, z_r0), f_air, C_Dl, A_F, A_L, z_a, shape (blend, C_Dt, delta, stern, s_L / a, b, c, asym) |
   | A2 | all keys (I_u, r_v, r_w, L_u, dryden, lambda_s, tau_s) | - |
   | E1 | all keys | - |
   | E2 | h_bar, s_h, l_h, d_bar, s_d, l_d, side, bank | a_h, w, dz_s, dth_s, dz_T, dth_T, c_y, c_n, l_b |
   | E3 | rate | p_dmg (constant 0.3) |
   | E4 | rate, own, p_trig (= all keys) | - |
   | layer 3 | psi_w (earth-fixed direction) | everything else |

   Not redrawn: whether an item is on (a boat drawn without wind never meets wind, `:26-28`); A3, H1-H3,
   B1-B3, W4 parameters (boat properties). A later mission whose environment makes the boat diverge is
   truncated, not redrawn (`:29-31`).
5. `mission_state` (`:104-125`): layer 1's slow processes restart on stream [res.seed, 2, mi, 0x51];
   items with attribute `clock` (fault items P8, P10, P11, P12 per the docstring `:11-14`) get t = mi T.
   None of the items of this document has `clock`.

Mission-scale event counts implied by the priors (T = 90 s): A2 squalls 0.0125-0.1; B2 CG jumps 0.025-1.5;
B2 mass step P(>= 1) = 1 - e^{-90 rate} in [0.26, 0.99]; E3 0.009-0.27; E4 0.05-0.75; E1 fronts
(at about 12 m/s, 1080 m per mission) 0.2-5.4; A1 / E1 drifts (tau_d >= 300 s) nearly constant within a mission.

---------------------------------------------------------------------------------------------------

## 9. Docstring-versus-code disagreements, dead interfaces, UNKNOWNs

1. B1 "dx_g -> W2's margin" (`cat_hidden_boat.py:518`) and the module docstring "ctx.load read by ... W2
   for its dolphin margin m_0" (`:38-40`): no W item reads ctx.load (search of `cat_water_a.py`,
   `cat_water_b.py`, `cat_w1.py`, `cat_prop.py` for `load`, `eps_m`, `dxg`: only P-chain pump "load", an
   unrelated variable). dx_g acts only through M_t, the Coriolis difference and dth0.
2. E3 "w.p. 0.3 an event flags q['debris_hit'] for P10" (`cat_air_env.py:50,73`): P10
   (`cat_prop.py:639-...`) never reads `debris_hit`; the flag is a dead output.
3. W5(b) publishes q['W5b'] = {ds, k_roll} "for H1" (`cat_water_a.py:349-350, 453`): H1 never reads it,
   and H1 (stage 'hidden') runs before W5 (stage 'force') in the same substep.
4. H2 docstring "g sin(theta_b)": the code uses theta_rel (pitch relative to the lofi running attitude),
   while H3 and B2(b) use tau0 + theta_rel. Consistent with H2 being calm-zero; noted because the
   symbols differ.
5. H1 initialisation sets stable rows to F / k_lin, ignoring c3 phi^3 (`:208`); for persist draws below
   U_c this is not the exact static heel.
6. A3: after 200 failed tries the FIRST draw is kept with x_0 clipped (`:386-389`), so lambda_L / kappa_th
   no longer hold jointly for that draw (the docstring says "redrawn").
7. `G12_off` (observation-model switch) is added to `switches` but no code acts on it (section 7).
8. The B1 Coriolis difference is inside B1's soft saturation (3 a_ref per channel), so for large rates the
   total Coriolis is not exactly c(s_r; rb_new, M_new).
9. The `CatItem` docstring lists "W7 vertical" as gated (`cat_base.py:645-646`), the code marks W7 not
   gated (`cat_water_b.py:182`) - outside this document's items, recorded for the W-item author.
10. UNKNOWN: the network's history window length (training code not read here).
11. UNKNOWN: how the acceptance test changes the joint parameter distribution beyond the item
    frequencies (only the on-frequencies were measured: within sampling error of the prior marginals,
    mean 0.04 rejections in 100 boats).

---------------------------------------------------------------------------------------------------

## 10. Summary table

| item | stage / flags | P_ON (marginal) | hidden state created | reads (beyond s_r, nu0) | channels of the output |
|---|---|---|---|---|---|
| H1 | hidden, gated | 0.7 (0.603) | phi, phid, noise n | U_r, u_r r, t_slope, thr, noz, K_roll (A1/A2), ctx.load (B1), kicks from H2/B2 | u, v, r, w, q; also changes port / stbd immersion |
| H2 | force, calm-zero | 0.4 (0.348) | s, sd per tank x {x, y} | theta_rel, phi, phidd (H1), eps_m (B1) | u, v, r, (w), q; roll moment to H1 |
| H3 | force | 0.15 (0.135) | m_b, x_b, xd_b | H_b (waves), tau0, theta_rel | u, w, q (static weight) |
| B1 | force, gated, drawn first | 0.8 (0.688) | none | e_z, e_th, zd, thd, u_r; sets M_t, m, W, vol, load | u, v, r, w, q |
| B2 | force | 0.106 (0.098) | dx, dy, mass flag, tow flag | u_r, v_r, tau0, theta_rel | u, v, r, w, q; roll moment to H1 |
| B3 | force | 0.3 (0.263) | none | u_r | u (+ M_t coupling) |
| A1 | env | 0.7 (0.611) | OU x (2), psi0 | ground (u, v), psi, wave direction | u, v, r; K_roll to H1 |
| A2 | env, needs A1 | 1.0 given A1 (0.603) | OU gusts, squall | A1's wind | u, v, r; K_roll; w_g to A3 |
| A3 | force | 0.6 (0.518) | none | wind (A1/A2), theta_rel, tau0, zd, thd, H_b | u, w, q (not calm-zero) |
| E1 | env | 0.6 (0.518) | Vbar, OU x, front cur/tgt | A1's U10, psi0 (wind mode) | u, v, r; sets nu_c for all later terms |
| E2 | force, gated | 0.1 (0.093) | depth / bank OU in distance | u_r, U_g | u, w, q (bank off) |
| E3 | force, impulse | 0.3 (0.263) | pending impulse | U_r | impulse in all channels |
| E4 | env, gated | 0.3 (0.263) | packet, clock, heading ring | position, heading, course | w, q (lofi response); packet added to the sea of all later terms |
| W4 | gate, structural, unsaturated | 1.0 (1.0) | none | hc, a_lofi, a_L1, a_cat, M_t | w, q (+ coupling via M_t^-1) |
| O1-O5 | not in the dynamics | OFF in D10a, unused by any pipeline | - | - | - |
