# Catalogue family D10, WATER items W1-W12: exact mathematical form (from code)

Scope: `learn/meta/cat_w1.py` (W1), `learn/meta/cat_water_a.py` (W2-W6), `learn/meta/cat_water_b.py` (W7-W12), and the shared machinery they read in `learn/meta/cat_base.py` (CatCtx, CatSea, substep_quantities, derive_quantities, item_accel, CatDraw.substep). Where a docstring and the implementation disagree, the implementation is followed and the disagreement is listed in section 6. All numeric values of fixed parameters are read from `studies/_cache/lofi_scarab195.json` through `operators_rb.default_params()` (operators_rb.py:126-132) by a short check script (values printed, not guessed). Items outside W (H1, layer 1, layer 3, E items) are only referenced where they decide the answer to the lateral questions of section 5; they are not derived here.

---------------------------------------------------------------------------

## 0. Symbols and conventions (each defined once)

### 0.1 Frames, signs
- Body axes: x forward, y to port, z up (cat_base.py:298-311 docstring of lever_Q; cat_water_b.py:7-11). Yaw rate r > 0 turns the bow to port.
- Lateral index j of a station point: j = 0 starboard (y = -B/2), j = 1 centre line (y = 0), j = 2 port (y = +B/2) (cat_base.py:351 `y_off = [-0.5 B, 0, 0.5 B]`). Station index i = 0 stern ... 4 bow.
- Pitch: theta is the low-fidelity pitch coordinate. The height of hull point (x, y) is `z + s_p x theta + y phi` (cat_base.py:537-538), with s_p = `sign_pitch` = -1 (value read). So theta > 0 lowers the bow (bow-down); bow-up trim relative to the running attitude is `trim_rel = s_p (theta - theta_0)` (cat_base.py:562).
- Roll phi: not a low-fidelity state. phi = 0 unless the hidden item H1 is on (H1 writes q['phi'], cat_hidden_boat.py:244; re-derivation cat_base.py:1054-1059). phi > 0 raises the port side.

### 0.2 State vector (reduced, the low-fidelity boat's)
`s = (x, y, u, z, zdot, theta, thetadot, psi, r, v)` (cat_base.py:531; operators_rb.to_reduced 1399-1402).
- x, y [m]: earth position; psi [rad]: heading.
- u, v [m/s]: body-frame surge, sway velocity over ground.
- z [m]: heave (up); zdot [m/s].
- theta [rad], thetadot [rad/s]: low-fidelity pitch convention above.
- r [rad/s]: yaw rate.
Velocity channels (network target channels) `nu = (u, v, r, zdot, thetadot)` = state indices (2, 9, 8, 4, 6) = `RED_VEL` (operators_rb.py:92). Channel names used below: surge (0), sway (1), yaw (2), heave (3), pitch (4).

### 0.3 Inputs
- thr [N]: applied thrust; noz [rad]: applied nozzle angle (after the actuator); cmd: physical commands (thrust N, nozzle rad) (cat_base.py:495-511).
- Waves: per row a set of deep-water components n with amplitude a_n [m], wavenumber k_n [1/m], frequency omega_n [rad/s], direction theta_n (stored as c_n = cos theta_n, s_n = sin theta_n), phase phi_n (operators_rb.RowSeas, operators_rb.py:1325-1339). Their distribution is not part of the W items.

### 0.4 Fixed parameters (low-fidelity boat; values read from the cache)
| symbol | code | value | unit |
|---|---|---|---|
| L | p["L"] | 5.4 | m |
| T | p["T"] | 0.38463 | m (stern keel depth below the reference at rest) |
| B = b | p["B"] (ctx.B, ctx.b, cat_base.py:345-346) | 2.0252 | m |
| x_i | linspace(x_stern, x_bow, 5) (cat_base.py:350) | (-2.12495, -0.80308, 0.51880, 1.84068, 3.16255) | m |
| y_j | (cat_base.py:351) | (-1.0126, 0, 1.0126) | m |
| k_i | keel height at station i (cat_base.py:357-360; operators_cat.cat_geometry 76-101 gives the same values) | (-0.38463, -0.34234, -0.30005, -0.25776, -0.21546) | m |
| Dx_i | trapezoid station length (cat_water_a.py:58-61, cat_water_b.py:136-141) | (0.66094, 1.32188, 1.32188, 1.32188, 0.66094) | m |
| z_0, theta_0 | running attitude | 0.15366, -0.0092291 | m, rad |
| m | p["m_coriolis"] | 1400 | kg |
| W = m g | ctx.W (cat_base.py:369) | 13734 | N |
| Vol = m / rho_w | ctx.vol (cat_base.py:370) | 1.36585 | m^3 |
| m_u, m_v | p["m_surge"], p["m_sway"] | 1968.58, 1610.20 | kg |
| k_drag | p["k_drag"] | 13.9872 | N s^2/m^2 |
| k_lin_sway, k_sway | | 1679.94 N s/m, 391.866 N s^2/m^2 | |
| k_js | p["k_jet_side"] | 1.61397 | - |
| k_nomoto_f, tau_r | | -1.69883e-4 rad/(s N), 0.42 s | |
| w_h, z_h | p["wn_heave"], p["z_heave"] | 5.83486 rad/s, 0.372324 | |
| w_p, z_p | p["wn_pitch"], p["z_pitch"] | 3.71347 rad/s, 0.155060 | |
| k_wh, k_wp | p.get("k_wave_heave"/"k_wave_pitch", 1.0) (cat_base.py:381-382) | 1.0, 1.0 (keys absent) | - |
| u_id | p["u_design"] (cat_base.py:372) | 12.8611 | m/s (25 kn) |
| u_lo, u_hi | U_RANGE (operators_rb.py:98) | 8.2311, 15.4333 | m/s (16, 30 kn) |
| u_max | p["u_max"] | 23.515 | m/s |
| c_wave_sway, c_wave_yaw | low-fidelity wave sway / yaw gains | 0.0, 0.0 | - |
| a_ref | a_ref_of(p) (operators_gen.py:165-174) | (3.2336, 2.6965, 1.7563, 13.095, 1.9644) | m/s^2, m/s^2, rad/s^2, m/s^2, rad/s^2 |
| A_REF, CLIP | operators.py:63, 67 | (4, 4, 2, 12, 3), 3 | |
| rho_w, g | cat_base.py:71-72 | 1025 kg/m^3, 9.81 m/s^2 | |
| rs | sqrt(L / 10 m) (cat_base.py:347) | 0.73485 | - |
| hc_run,i | calm keel immersion at (z_0, theta_0) (cat_base.py:386-387) | (0.25058, 0.19609, 0.14160, 0.08710, 0.03261) | m |
| hbar_run | mean_i max(hc_run,i, 0) (cat_base.py:388) | 0.14160 | m |

### 0.5 Per-draw random quantities shared by all items
- tau_0 ~ U[2 deg, 6 deg], the hidden true running trim (cat_base.py:147, 905); tau_run = softplus_floor(tau_0, 0.5 deg, 0.5 deg) (cat_base.py:400, 407-409).
- M_t (5 x 5), the shared generalised inertia drawn by layer 3 (operators_gen.py:283-315; set at cat_base.py:910-912): diagonal (m_u, m_v, I_r, m_w, I_q) with off-diagonals M_t[0,2] = -m' y_g, M_t[0,4] = m' z_g, M_t[1,2] = m' x_g, M_t[3,4] = -m' x_g (operators_gen.py:303-309), m' ~ U[0.9 m, 0.97 min(m_u, m_v)], x_g ~ U[-0.05, 0.05] L, y_g ~ U[-0.03, 0.03] B, z_g ~ U[-0.3, 0.3] T (operators_gen.py:289-292). The coupling graph sway-yaw-surge-pitch-heave is connected, so M_t^{-1} is in general a full matrix (three draws checked: the sway column of M_t^{-1} is about 6.3e-4 (sway), -2.6e-5 to -4.8e-5 (yaw), 3e-7 to 1.3e-6 (surge), below 1e-7 (heave, pitch) per newton).

### 0.6 Helper functions (exact definitions)
- logistic sigma(x) = 1 / (1 + e^{-x}) (cat_water_a.py:50-51 clips x to [-60, 60]; cat_water_b.py:132-133 writes it as (1 + tanh(x/2)) / 2, identical).
- softplus(x) = log(1 + e^x) (cat_water_a.py:54-55).
- softplus_floor(x, lo, s) = lo + s softplus((x - lo)/s) (cat_base.py:331-333).
- soft_sat(y, n) = 3 n tanh(y / (3 n)) per channel (cat_base.py:161-164, SAT = 3 at cat_base.py:78).
- exp_update(x, x_inf, tau, dt) = x_inf + (x - x_inf) e^{-dt/tau} (cat_base.py:180-184); exp_update_asym uses tau_up when x_inf > x, tau_down otherwise (cat_base.py:187-191).
- lin_damp_acc(c, v, dt) = -v (1 - e^{-c dt}) / dt if c >= 0, else -c v (cat_base.py:194-201).
- hyst_switch: discrete h <- 1 if x > up, 0 if x < down, else unchanged; smoothed s <- exp_update(s, h, tau) (cat_base.py:216-224).
- implicit_quad_drag(v, c, m_e, dt) = -c |v| v / (m_e + 2 dt c |v|), c clipped >= 0 (cat_base.py:276-283).
- lever_Q of a body force F = (F_x, F_y, F_z) at point (x_p, y_p, z_p): Q = (F_x, F_y, x_p F_y - y_p F_x, F_z + s_p theta F_x, s_p (x_p F_z - z_p F_x)) (cat_base.py:298-311). Q is a generalised force in the five channels (surge N, sway N, yaw N m, heave N earth-vertical, pitch N m in the theta convention).
- unit_vertical(x_p) = (0, 0, 0, 1, s_p x_p) (cat_base.py:319-322); point_mass(g) = 1 / (g^T M_t^{-1} g) (cat_base.py:325-328).
- M_vv = inv(M_t^{-1}[3:5, 3:5]) (cat_water_a.py:64-67): the heave-pitch generalised force that produces a requested heave-pitch acceleration.
- LogU[a, b] = exp(U[log a, log b]) (operators.py:85-86).

---------------------------------------------------------------------------

## 1. How an item's output reaches the boat (and the network target)

Per substep (dt = 0.04 s, cat_base.py:76), in CatDraw.substep (cat_base.py:1017-1108):
1. q = substep_quantities(START state of the substep, applied thr, noz, sea sampled at the substep start) (cat_base.py:1030-1031).
2. Stage 'env' items (E1 current, E4 packets, A1/A2) and stage 'hidden' items (H1 roll) run, then derive_quantities is re-run (cat_base.py:1053-1059).
3. Layer 1 at the water-relative velocities (cat_base.py:1062-1078).
4. Stage 'force' items in CODES order (W1-W3, W5-W12 among them) (cat_base.py:1080-1082), then the propulsion chain, then stage 'gate' (W4) (cat_base.py:1089-1091).
5. Layer 3 + Coriolis difference (cat_base.py:1092-1102).

For one item (item_accel, cat_base.py:724-740): `a = M_t^{-1} Q + acc`; if `saturate`: `a_c <- soft_sat(a_c, n_c)` per channel with n = item.nominal(); impulses `imp` are returned separately and never soft-saturated.
- nominal for W1, W2, W5: half_clip_nominal = min(a_ref, CLIP A_REF / (2 SAT)) = (2, 2, 1, 6, 1.5) (cat_base.py:167-177) -> per-item caps 3 n = (6, 6, 3, 18, 4.5).
- nominal for W6-W12 (default, cat_base.py:678-680): a_ref -> caps 3 a_ref = (9.70, 8.09, 5.27, 39.28, 5.89).
- W4: saturate = False (cat_water_a.py:289). W3: impulse only.
- Items with `gated = True` (W1, W2, W5, W8, W11 here; also H1) add their saturated heave/pitch accelerations to q['a_cat_vert'] (cat_base.py:1046-1047), which W4 reads.

Application (simulate_cat, operators_cat.py:525-568): the low-fidelity step runs first (`ns = red.step(...)`), then the summed catalogue rule plus the held control-step part is clipped jointly per channel at +-CLIP A_REF = +-(12, 12, 6, 36, 9) (clip_push, operators.py:76-82; operators_cat.py:542) and added as a velocity kick `nu += push dt` (operators_cat.py:567); impulses are added after that, capped cumulatively over a control step at +-LIM dtc = +-(2.88, 2.88, 1.44, 8.64, 2.16) (m/s, m/s, rad/s, m/s, rad/s) for dtc = 0.24 s (operators_cat.py:429, 550-555, 568). The same in the Mission path (CatInjector, operators_cat.py:711-728).
Consequence: every W force is evaluated at the START state of the substep and integrated by one explicit kick, except where an item builds an implicit / exact-exponential form itself (W1 and W2 damping parts, W6, W7, W10, and the W3 impulse).

On-probabilities. Non-sparse episode: item on with P_ON (cat_base.py:92-105), then draw_on. Sparse episode (probability 0.15, cat_base.py:145, 906): n ~ U{1, 2, 3} codes chosen without replacement from the 38 registered non-structural codes (39 registered, W4 structural; checked), each chosen code is on (cat_base.py:944-966). W3 and W8 are skipped entirely when has_waves is False (cat_base.py:107, 960). W4 is always on (cat_base.py:108). Marginal on-rate with waves = 0.85 P_ON + 0.15 x 2/38.

| item | P_ON (cat_base.py) | marginal with waves |
|---|---|---|
| W1 | 0.7 | 0.603 |
| W2 | 0.3 | 0.263 |
| W3 | 0.8 (waves only) | 0.688 |
| W4 | 1.0 (structural) | 1.0 |
| W5 | 1 - 0.4 x 0.5 = 0.8 | 0.688 |
| W6 | 0.6 | 0.518 |
| W7 | 1 - 0.3^2 = 0.91 | 0.781 |
| W8 | 0.8 (waves only) | 0.688 |
| W9 | 0.7 | 0.603 |
| W10 | 0.4 | 0.348 |
| W11 | 0.6 | 0.518 |
| W12 | 0.3 | 0.263 |

---------------------------------------------------------------------------

## 2. Shared per-substep quantities (cat_base.py:428-578)

Inputs: START state s, sea components. For each of the 15 station points (i, j) plus the intake point:
- Earth position X_ij = x + x_i cos psi - y_j sin psi, Y_ij = y + x_i sin psi + y_j cos psi (cat_base.py:454-455).
- Earth point velocity Xdot_ij = u cos psi - v sin psi - r (x_i sin psi + y_j cos psi), Ydot_ij = u sin psi + v cos psi + r (x_i cos psi - y_j sin psi) (cat_base.py:456-459).
- Phase ph_n = k_n (X c_n + Y s_n) - omega_n t + phi_n (cat_base.py:463-464).
- Elevation eta0_ij = sum_n a_n cos ph_n (cat_base.py:467).
- Elevation rate along the MOVING point etad0_ij = sum_n a_n [omega_n - k_n (Xdot c_n + Ydot s_n)] sin ph_n (cat_base.py:468-469).
- Horizontal orbital velocity at the surface, earth frame: (sum_n a_n omega_n cos ph_n c_n, sum_n a_n omega_n cos ph_n s_n) (cat_base.py:470-471), rotated to body axes: u_orb,ij = U_e cos psi + V_e sin psi, v_orb,ij = -U_e sin psi + V_e cos psi (cat_base.py:472-473). Deep-water linear form evaluated at the mean free surface (no depth decay, no wave-steepness correction).
- Calm water: all zero (cat_base.py:446-448).

Environment items may add eta_add, etad_add (E4, cat_air_env.py:754-755), u_orb / v_orb increments (cat_air_env.py:757-758), a body-frame current nu_c (E1, cat_air_env.py:489).

derive_quantities (cat_base.py:528-578), with eta = eta0 + eta_add, etad = etad0 + etad_add:
- hull height hull_ij = z + s_p x_i theta + y_j phi; its rate whull_ij = zdot + s_p x_i thetadot + y_j phidot (537-540).
- station immersion d_ij = eta_ij - hull_ij (542); entry velocity dd_ij = etad_ij - whull_ij (> 0 = point moving into the water) (543).
- keel immersion h_ij = d_ij - k_i (544); centre-line hc_i = h_i1, ddc_i = dd_i1 (545).
- longitudinal slope (least squares over the centre line) slope = sum_i eta_i1 xc_i / sum_i xc_i^2, xc_i = x_i - mean(x) (547-548; ctx.xc, ctx.den at cat_base.py:352-353).
- transverse slope t_slope = mean_i (eta_i2 - eta_i0) / B (549).
- low-fidelity water reference: z_ref = k_wh mean_i eta_i1 + z_0; theta_ref = k_wp s_p arctan(slope) + theta_0; zd_ref = k_wh mean_i etad_i1; thd_ref = k_wp s_p (sum_i etad_i1 xc_i / den) / (1 + slope^2) (552-556).
- e_z = z - z_ref, e_th = theta - theta_ref, w_r = zdot - zd_ref, q_r = thetadot - thd_ref (557-558).
- effective trim tau_e = softplus_floor(tau_0 + s_p (theta - theta_0) - arctan(slope), 0.5 deg, 0.5 deg) (562-564).
- intake immersion h_in (566-568; not read by W items).
- water-relative velocities u_r = u - nu_c,x, v_r = v - nu_c,y, U_r = sqrt(u_r^2 + v_r^2), U_g = sqrt(u^2 + v^2) (570-572). r is not corrected.
- qdyn = rho_w U_r^2 / 2, qb2 = qdyn b^2, Cv = U_r / sqrt(g b), Fn = U_r / sqrt(g L), Fn_vol = U_r / sqrt(g Vol^{1/3}) (573-577). Values: Cv(u_lo, u_id, u_hi) = (1.85, 2.89, 3.46); Fn_vol(u_lo, u_id, u_hi, u_max) = (2.49, 3.90, 4.68, 7.13).
- low-fidelity analytic vertical accelerations a_lofi_vert = (-w_h^2 (z - z_ref) - 2 z_h w_h zdot, -w_p^2 (theta - theta_ref) - 2 z_p w_p thetadot) (cat_base.py:480-492, 520-524).

Note on speed dependence of the immersions: the low-fidelity boat's heave / pitch equilibrium is z_ref, theta_ref (speed independent; control/reduced.py:397-424). The keel immersions h_ij therefore change with speed only through items that shift the equilibrium: W1 (trim shift), W5(b) (trim, stiffness), W11 (trim and sinkage bump), layer-1 T4 (running-attitude shift), and through waves / roll.

---------------------------------------------------------------------------

## 3. The items

Notation in each item: P = parameters with their coded prior; Q = generalised force (goes through M_t^{-1}); acc = acceleration written directly into a channel (bypasses M_t^{-1}). "Channels" lists direct entries; Q entries are spread to the other channels through M_t^{-1} (section 0.5).

### W1 planing-lift nonlinearity and speed extrapolation (cat_w1.py:52-122)
- stage force, gated, calm_zero; P_ON 0.7; no sub-parts; no hidden state.
- P (cat_w1.py:57-61): p_d ~ U[0.3, 0.8], p_h ~ U[1.5, 3.0], q ~ U[0.8, 1.3], c_h ~ U[0.46, 4.1], l_x ~ U[0, 0.15] L, p_tau ~ U[0.5, 1.8], p_K ~ U[0, 1.8], p_B ~ U[0, 1.0].
- Quantities read: hc_i, U_r, Cv, tau_e, e_z, w_r.
- eps = mean_i max(hc_i, 0) / hbar_run - 1 (74-76).
- s = max(U_r / u_id, u_lo / u_id) (79-81) (s >= 0.640, no upper cap).
- tau_eq = softplus_floor(tau_0 s^{-p_tau}, 0.5 deg, 0.5 deg) (84-90).
- r_h = c_h / max(Cv^2, 1e-6), w_hs = r_h / (1 + r_h) (96-97) (at u_id w_hs in [0.052, 0.330]).
- Phi = (tau_e / tau_eq)^q [ (1 - w_hs) (1 + eps)_+^{p_d} + w_hs (1 + eps)_+^{p_h} ] (99-101); Phi_eps = (1 - w_hs) p_d + w_hs p_h; Phi_tau = q / tau_eq (102-103).
- Q_surge = -W (Phi - 1) tan tau_e (112).
- Q_heave = W [Phi - 1 - Phi_eps eps - Phi_tau (tau_e - tau_eq)] (113): zero value and zero gradient at (eps = 0, tau_e = tau_eq).
- Q_pitch = s_p W Phi l_x eps / (1 + |eps|) (114).
- acc_pitch = w_p^2 s_p tau_0 (s^{-p_tau} - 1) (117-118): shifts the pitch equilibrium by s_p tau_0 (s^{-p_tau} - 1).
- acc_heave = -w_h^2 (s^{p_K} - 1) e_z + lin_damp_acc(2 z_h w_h (s^{p_B} - 1), w_r, dt) (119-121) (for s < 1 the damping coefficient is negative: explicit "less damping").
- Channels: surge, heave, pitch. Not sway, not yaw.
- Overlaps (cat_base.py:116-118): layer-1 T4 speed polynomial constant, T4 bounded stiffness off, T4 heave-pitch coupling off.

### W2 porpoising (cat_water_a.py:89-170)
- stage force, gated, calm_zero; P_ON 0.3; sub-part: flutter w.p. 0.3 else van der Pol; no hidden state.
- P (117-127): m0 ~ U[-3, 1.5] deg, m1 ~ U[-1, 1] deg, w_m ~ U[0.2, 0.8] deg, rho_n ~ LogU[0.3, 2.5], A_th ~ LogU[0.5, 4] deg, A_z ~ LogU[0.02, 0.15] m, x_m ~ U[-0.3, 0.3] L, flutter ~ Bernoulli(0.3), rho_f ~ LogU[0.3, 2.5], l_f ~ LogU[0.1, 1] L, sgn = sign(U[-1, 1]).
- Derived (128-141): mode phi = (-s_p x_m, 1); c = M_vv phi / (phi^T M_vv phi); c_s = c^T diag(2 z_h w_h, 2 z_p w_p) phi; c_neg = rho_n c_s; s_sat = min(A_th, A_z / |x_m|) / 2; kappa*^2 = smallest kappa^2 with an unstable root of (l^2 + 2 z_h w_h l + w_h^2)(l^2 + 2 z_p w_p l + w_p^2) + kappa^2 (bisection, 71-86); kap = sqrt(rho_f kappa*^2); k35 = sgn kap l_f; k53 = -sgn kap / l_f.
- Gate gamma = sigma((m0 + m1 (U_r - u_id)/u_id) / w_m) (145-147).
- Flutter (154-159): f = gamma / (1 + (e_z/A_z)^2 + (e_th/A_th)^2); acc_heave = -f k35 e_th; acc_pitch = -f k53 e_z.
- van der Pol (160-170): s_m = c . (e_z, e_th), sdot_m = c . (w_r, q_r); c_eff = -gamma c_neg (1 - s_m^2 / s_sat^2); a_s = lin_damp_acc(c_eff, sdot_m, dt); Q[heave, pitch] = a_s M_vv phi. Heave-pitch acceleration = a_s phi exactly; surge/sway/yaw get M_t^{-1}[0:3, 3:5] M_vv phi a_s (nonzero only through the M_t couplings).
- Channels: heave, pitch.

### W3 slamming impulses (cat_water_a.py:174-271)
- stage force, not gated, calm_zero; P_ON 0.8, only when has_waves; no sub-parts; hidden state m_old (previous substep's half-section masses; the first substep returns nothing, 265-268).
- P (200-207): (beta_stern, beta_bow) = sort of two U[12, 28] deg draws; beta_i linear in x_i; k_w ~ U[1, pi/2]; C_m ~ U[0.2, 0.47]; chine depth d_c,i = tan(beta_i) / (2 k_w) b (range 0.137-0.538 m); p_h ~ U[1, 2.5]; r_exit ~ U[-0.2, 0.2]. adjust_ctx puts d_c on ctx.chine_dc for W5 (209-210).
- Half-section added mass at the starboard and port points (j = 0, 2): m_ij = (1/2) rho_w C_m b^2 Dx_i clip(h_ij / d_c,i, 0, 1)^{p_h} (216-220).
- Impulses, stations from bow (i = 4) to stern (i = 0) (222-259): Qhat_i = lever_Q((x_i, 0, k_i), (-tan tau_e, 0, 1)) = (-tan tau_e, 0, 0, 1 - s_p theta tan tau_e, s_p (x_i + k_i tan tau_e)); g_i = unit_vertical(x_i); m_eff,i = 1 / (g_i^T M_t^{-1} Qhat_i); V_ij = dd_ij - g_i^T Dnu (Dnu the velocity jumps of the stations already processed); Dm_ij = m_new,ij - m_old,ij. For mask E = {Dm > 0, V > 0} (scale 1) and mask X = {Dm < 0, V < 0} (scale r_exit): w = sum_{j in mask} Dm_ij V_ij / (m_eff,i + sum_{j in mask} m_new,ij); dw_i += scale w; J_ij += scale (m_new,ij (V_ij - w) - m_old,ij V_ij). J_i = m_eff,i dw_i. Lateral force F_y,i = (max(J_i0, 0) - max(J_i2, 0)) tan(beta_i). Q_i = J_i Qhat_i + lever_Q((x_i, 0, k_i), (0, F_y,i, 0)) = J_i Qhat_i + (0, F_y,i, x_i F_y,i, 0, 0); Dnu += M_t^{-1} Q_i.
- Output: imp = Dnu (impulse channel, velocity jumps, capped per control step, section 1).
- Channels: surge, heave, pitch (vertical impulse J and its -J tan tau_e surge part); sway, yaw only from the port-starboard difference of the half-section impulses.

### W4 contact gate (cat_water_a.py:275-323)
- stage gate, structural (P_ON 1.0, also in sparse episodes), not saturated, calm_zero.
- P (291-294): h_off ~ U[-0.05, 0.05] m, w_d ~ U[0.02, 0.1] m, r_exit ~ U[0, 0.3] (r_exit is written to ctx.w4_r_exit, 296-297, and read by nothing; section 6).
- chi_i = min(1, sigma((hc_i - h_off)/w_d) / sigma((hc_run,i - h_off)/w_d)) (300-303).
- A = a_lofi_vert + a_L1_vert + a_cat_vert + (g, 0) (309-311); Q_hp = M_vv A (312); w_i = max(hc_run,i, 0); (alpha, beta) solves [[sum w, sum w x], [sum w x, sum w x^2]] (alpha, beta) = (Q_hp,heave, s_p Q_hp,pitch) (313-317); f_i = w_i (alpha + beta x_i) (318).
- Q_heave = sum_i (chi_i - 1) f_i; Q_pitch = s_p sum_i (chi_i - 1) f_i x_i (319-322).
- Channels: heave, pitch.

### W5 chine / spray-rail wetting (cat_water_a.py:327-454)
- stage force, gated, calm_zero only with (b) off (calm_params, 388-391); P_ON 0.8. Sub-parts given on (327, 362-367): (a) only 0.375, (b) only 0.25, both 0.375 -> marginal non-sparse P(a) = 0.6, P(b) = 0.5. Hidden state for (b): discrete h, smoothed s.
- P (368-382): x^_i = (x_i - x_stern)/(x_bow - x_stern); d_c,i = d_c0 (1 + (f_bow - 1) x^_i), d_c0 ~ U[0.3, 1.5] T (0.115-0.577 m), f_bow ~ U[1, 2] -- replaced by W3's d_c,i when W3 is on (371-373; W3 is drawn first in CODES order, cat_base.py:953-972); n_rail ~ U{0, 1, 2}; r ~ U[0.3, 1.5]; w_frac ~ U[0.05, 0.3]; rail_frac = sort(two U[0.2, 0.9]); r_s = two U[0.7, 1.2]; n_r, n_v ~ U[-1, 1]; U_c ~ U[12, 30] kn; hyst ~ U[0, 4] kn; tau_s ~ LogU[1, 10] s; dR ~ U[-0.08, 0.08]; dth ~ U[-0.5, 0.5] deg; k_m ~ LogU[0.8, 1.25]; k_roll ~ LogU[0.7, 1.4] (k_roll has no effect, section 6).
- (a) station stiffness. Kink fractions fr = (rail_frac[:n_rail], 1), plateaus P = cumprod(1, r_s[:n_rail], r), steps D = diff(P) (397-404). For each station, width w_i = w_frac d_c,i; kappa(h) = 1 + sum_k D_k sigma((h - fr_k d_c,i)/w_i); F(h) = h + sum_k D_k w_i [softplus((h - fr_k d_c,i)/w_i) - softplus(-fr_k d_c,i / w_i)] (407-418). k1_i = w_h^2 / (M_t^{-1})_33 / 5 / kappa(hc_run,i) (428). e_i = k1_i [F(hc_i) - F(hc_run,i) - kappa(hc_run,i)(hc_i - hc_run,i)] (431). Q_heave = sum_i e_i; Q_pitch = s_p sum_i e_i x_i (432-433).
- (a) side walls (434-439): Dh = sum_i [ (h_i2 - d_c,i)_+ - (h_i0 - d_c,i)_+ ] (port minus starboard keel immersion above the chine); fac = (Dh / 0.1 m) (U_r / u_id)^2; acc_sway = n_v 0.3 a_ref,sway fac; acc_yaw = n_r 0.3 a_ref,yaw fac. Coefficients for |n| = 1: 8.09 m/s^2 and 5.27 rad/s^2 per metre of Dh at U_r = u_id. No station lever arm: the yaw part reads the same sum Dh as the sway part.
- (b) speed switch (440-453): h flips to 1 above U_c + hyst/2 and to 0 below U_c - hyst/2 on U_r, s follows with lag tau_s; ds = s - [u_id > U_c]; Q_surge += -ds dR k_drag u_r |u_r|; acc_pitch += w_p^2 s_p dth ds; acc_heave += -w_h^2 (k_m - 1) ds e_z.
- Channels: (a) heave, pitch (Q) and sway, yaw (acc); (b) surge (Q), heave, pitch (acc).
- Overlap: T4 bounded stiffness off when (a) is on (385-386).

### W6 whisker-spray drag (cat_water_a.py:467-508)
- stage force, not gated, not calm_zero (a real calm-water drag); P_ON 0.6; no hidden state.
- P (482-491): f_s ~ U[0, 0.15], p_s ~ U[0, 1.5], f_b ~ U[0, 1], z_s ~ U[-1, 0.5] T, h_b0 = hc_run,4 + U[0, 0.3] T; kap_s = f_s k_drag / ((1/2) rho_w C_f(u_id) b^2); kap_b = f_b (1/2)(0.07 W) / ((1/2) rho_w u_id^2 b 0.1 m).
- C_f(U) = 0.075 / (log10(max(|U| L / nu, 1e5)) - 2)^2, nu = 1.19e-6 m^2/s (462-464).
- c_main = (1/2) rho_w C_f(U_r) b^2 kap_s (tau_run / tau_e)^{p_s}; c_bow = kap_b (1/2) rho_w b max(hc_4 - h_b0, 0) (493-498).
- a_x = implicit_quad_drag(u_r, c_main + c_bow, 1/(M_t^{-1})_00, dt); F = ((1/(M_t^{-1})_00) a_x, 0, 0) at (0, 0, z_s) -> Q = (F_x, 0, 0, s_p theta F_x, -s_p z_s F_x) (500-507).
- Channels: surge, heave (through s_p theta F_x), pitch.

### W7 viscous crossflow drag, vertical and lateral (cat_water_b.py:179-230)
- stage force, not gated (gates itself), calm_zero; P_ON 0.91; sub-parts vert, lat each Bernoulli(0.7) redrawn until at least one is on (170-175): given on, vert only 0.231, lat only 0.231, both 0.538; marginal non-sparse P(vert) = P(lat) = 0.7. No hidden state.
- P (184-189): C_Dc ~ U[0.5, 1.2], h_off ~ U[-0.05, 0.05] m, w_d ~ U[0.02, 0.1] m, r_exit ~ U[0, 0.3], C_Dl ~ U[0.3, 1.5], c_orb ~ U[0, 1].
- Vertical (205-213), at all 15 points: chi_ij = sigma((h_ij - h_off)/w_d) (plain logistic, not normalised to the running value); c_ij = rho_w (b/3) Dx_i C_Dc chi_ij (r_exit if dd_ij < 0 else 1); F_v,i = sum_j c_ij |dd_ij| dd_ij (upward when the point enters); generalised force sum_i F_v,i g_v,i with g_v,i = (0, 0, 0, 1, s_p x_i); D_v = sum_i (sum_j c_ij |dd_ij|) g_v,i g_v,i^T.
- Lateral (214-222), per station on the centre line: w_i = v_r + x_i r - c_orb v_orb,i1 (lateral velocity of the hull section relative to water moving at c_orb times the surface orbital velocity); c_i = (1/2) rho_w C_Dl max(hc_i, 0) Dx_i; F_y,i = -c_i |w_i| w_i; generalised force sum_i F_y,i g_l,i with g_l,i = (0, 1, x_i, 0, 0); D_l = sum_i c_i |w_i| g_l,i g_l,i^T.
- Update (225-230): acc = (M_t + 2 dt (D_v + D_l))^{-1} (F_v + F_l): the vector form of implicit_quad_drag; dissipative, never reverses the relative velocity (check w7_contracts_energy, 524-552); then soft-saturated at 3 a_ref per channel.
- Channels: vertical part heave, pitch; lateral part sway, yaw; the solve couples all channels through M_t.
- Overlaps (191-197): with vert, T3's heave/pitch quadratic kappa off; with lat, T3's sway/yaw quadratic kappa off (operators_rb.py:806-808 are the switched terms: -kap_0 |v| v, -kap_1 |r| r).
- Size at calm running (h = hc_run): c_i / C_Dl = (84.9, 132.8, 95.9, 59.0, 11.0) N s^2/m^2, sum 383.7; stations 0-1 (aft of x = -0.8 m) carry 57 % of it. Sum over C_Dl range: 115-576 N s^2/m^2 (the low-fidelity quadratic sway coefficient k_sway is 391.9 N s^2/m^2).

### W8 second-order wave forces (cat_water_b.py:238-287)
- stage force, gated, calm_zero; P_ON 0.8, only when has_waves; hidden states dbar (5 x 3 first-order mean of d) and L (5 low-passed products).
- P (243-257): c1 ~ U[0.1, 0.6], c2 ~ U[1, 3], p_u ~ U[-1, 1], tau2 ~ LogU[2, 30] s, tau_m ~ LogU[5, 20] s, Q_x = 16 A A^T / tr(A A^T) with A_kl ~ N(0, 1) (5 x 5), Q_y, Q_n = N(0,1)^{5x5} / Frobenius norm (not symmetric), Q_z, Q_t = (N + N^T) / Frobenius norm, k_y, k_n ~ U[0, 1], k_z, k_t ~ U[-1, 1].
- zeta_ij = d_ij - dbar_ij, then dbar <- exp_update(dbar, d, tau_m) (265-268); zeta_s,i = mean_j zeta_ij; zeta_a,i = (zeta_i2 - zeta_i0)/2 (269-270).
- f = (zeta_s^T Q_x zeta_s, zeta_s^T Q_y zeta_a, zeta_s^T Q_n zeta_a, zeta_s^T Q_z zeta_s, zeta_s^T Q_t zeta_s); L <- exp_update(L, f, tau2) (271-274).
- s_w = sqrt(max(L_0, 0)) / b; s_u = max(U_r/u_id, u_lo/u_id) (275-276).
- Q_surge = -tanh(u_r / 0.5 m/s) W c1 s_w / (1 + c2 s_w) s_u^{p_u} (279-280).
- acc_sway = k_y 0.3 L_1 / 0.25^2; acc_yaw = k_n 0.1 L_2 / 0.25^2; acc_heave = k_z 0.03 a_ref,heave L_3 / 0.25^2; acc_pitch = k_t 0.03 a_ref,pitch L_4 / 0.25^2 (282-286).
- Channels: all five. Every term is quadratic in the immersion fluctuation zeta (which contains hull motion, not only waves).
- Overlap: T7 one-direction quadratic wave term off (cat_base.py:121).

### W9 Froude-Krylov surge force in following seas (cat_water_b.py:296-325)
- stage force, not gated, calm_zero; P_ON 0.7; sub-parts filt ~ Bernoulli(0.5), orb ~ Bernoulli(0.5); hidden state y (filter).
- P (303-306): mu_x ~ U[0.4, 1.3], tau_p ~ LogU[0.07, 1.4] rs s, ratio = tau_z/tau_p ~ LogU[1/3, 3], mu_o ~ U[0, 1].
- wet_i = [hc_i > 0]; sl_i = d eta_i1 / dx by numpy.gradient over x_i (central inside, one-sided at the ends); sbar = wet-mean of sl (0 when no station is wet) (313-315; 291-293).
- filt: y <- exp_update(y, sbar, tau_p); s = ratio sbar + (1 - ratio) y (316-318) (the lead/lag (1 + tau_z p)/(1 + tau_p p), unit static gain).
- Q_surge = -mu_x m g s (321); orb: Q_surge += 2 k_drag |u_r| mu_o ubar_orb, ubar_orb = wet-mean of u_orb,i1 (322-324).
- Channels: surge only. Reads only the LONGITUDINAL slope.

### W10 bow lateral force when the bow digs in (cat_water_b.py:329-375)
- stage force, not gated, calm_zero; P_ON 0.4; sub-part mod ~ Bernoulli(0.5); no hidden state.
- P (334-339): x_b ~ U[0.25, 0.5] L (1.35-2.70 m forward of the reference point), q_b ~ U[1, 2], d_0 ~ U[0, 0.5] T (0-0.192 m), kap_b ~ LogU[0.1, 2], eps_r ~ U[-1, 1].
- h_bow = (hc_3 + hc_4)/2; g_b = ((h_bow - d_0)_+ / T)^{q_b}; c = kap_b k_lin_sway (U_r / 10 m/s) g_b [N s/m] (345-351).
- g = (0, 1, x_b, 0, 0); m_g = 1 / (g^T M_t^{-1} g); w = v_r + x_b r; F = m_g w (e^{-c dt / m_g} - 1) / dt (-> -c w for c dt << m_g); Q = F g (355-360): sway F, yaw x_b F.
- mod (362-374): m_r = eps_r tanh(slope / 0.05); with steer_jet = 1: r_t = k_nomoto_f k_js max(thr, 0) sin(clip(noz, +-rud_stall)); acc_yaw = m_r r_t / tau_r + lin_damp_acc(m_r / tau_r, r, dt) (= (m_r / tau_r)(r_t - r) to first order in dt).
- Channels: sway, yaw.
- Size at calm running: h_bow = 0.0599 m; with d_0 = 0, q_b = 1, kap_b = 2: c = 672 N s/m at U_r = u_id (k_lin_sway = 1680 N s/m for comparison); c decreases with d_0 and q_b, vanishes when h_bow <= d_0.

### W11 transition resistance and trim hump (cat_water_b.py:379-422)
- stage force, gated, not calm_zero; P_ON 0.6; sub-parts asym ~ Bernoulli(0.5), hyst ~ Bernoulli(0.3); hidden state (h, s) when hyst.
- P (384-394): F_h ~ U[0.9, 2], w_b ~ U[0.2, 0.6] (code name w_h, not the heave frequency), r_h ~ U[0, 0.15], th_h ~ U[0, 5] deg, z_hb ~ U[0, 0.3] T (code z_h), f_lo, f_hi ~ U[0.6, 1.6] if asym else 1, dF ~ U[0, 0.3] if hyst else 0, tau_s ~ LogU[1, 10] s.
- centre c_F = F_h, or with hyst c_F = F_h + (dF/2)(1 - 2 s) (s the smoothed switch on Fn_vol with thresholds F_h +- dF/2); x_F = Fn_vol - c_F; width w = w_b f_lo (x_F < 0) or w_b f_hi; Bm = exp(-x_F^2 / (2 w^2)) (399-411).
- Q_surge = -W r_h Bm tanh(u_r / 0.5 m/s); acc_heave = -w_h^2 z_hb Bm; acc_pitch = w_p^2 s_p th_h Bm (413-422).
- Channels: surge, heave, pitch. The peak sits at Fn_vol in [0.9, 2] while the task range is Fn_vol in [2.49, 4.68].

### W12 stepped hull (cat_water_b.py:426-494)
- stage force, not gated, not calm_zero; P_ON 0.3; hidden states: speed switch (h, s) and ventilation v_j per side.
- P (431-446): f_s ~ U[0.25, 0.5], x_s = x_stern + f_s (x_bow - x_stern) (-0.803 to 0.519 m); x_c = x_s - U[0, 0.15] L; z_k = keel at x_s; h_run = calm keel immersion at x_s (linear interpolation); U_v ~ U[10, 25] kn, h_U ~ U[0, 4] kn, tau_s ~ LogU[0.3, 3] s, dh_v ~ U[0.2, 2] h_run, w_v ~ U[0.1, 0.4] h_run, k_beta ~ U[-2, 2], tau_loss ~ LogU[0.02, 0.3] s, tau_rec ~ LogU[0.1, 2] s, l_c ~ U[-0.15, 0.1], r_c ~ U[0, 0.05], c_y ~ U[-0.25, 0.25], l_v ~ U[0, 0.1], a_h ~ U[0.5, 2], a_tau ~ U[0, 1.5].
- h_s,j = keel immersion at x_s (linear interpolation, j = 0, 1, 2) (455); s_U = hyst switch on U_r (U_v +- h_U/2, lag tau_s) (457-461); beta_s = (v_r + x_s r) / max(U_r, 2 m/s) (462); h_eff,j = h_s,j + k_beta sg_j (B/2) beta_s, sg = -1 (starboard), +1 (port) (463-465); v_inf,j = s_U (1 - smoothstep((h_eff,j - h_run - dh_v)/w_v)), smoothstep(x) = 3x^2 - 2x^3 on [0, 1] (466-467); v_j <- exp_update_asym(v_j, v_inf,j, tau_rec rising, tau_loss falling) (470-471).
- Collapse per side (481-487): c_j = clip(h_s,j / 0.05 m, 0, 1), wet_j = (1 - v_j) c_j; F = (W/2) wet_j (-r_c, -sg_j c_y, l_c) at (x_c, sg_j B/4, z_k) -> Q via lever_Q: yaw = x_c F_y - sg_j (B/4) F_x.
- Ventilated-region lift loss (488-493): c_c = clip(h_s,1 / 0.05, 0, 1); G = (max(h_s,1, 0)/h_run)^{a_h} (tau_e / tau_run)^{a_tau}; F_z = -W l_v (G - 1) mean_j(v_j) c_c; F = (-F_z tan tau_e, 0, F_z) at (x_c, 0, z_k).
- Channels: surge, sway, yaw, heave, pitch (collapse); surge, heave, pitch (lift loss). Net sway force is proportional to c_y (wet_0 - wet_2) (W/2): zero for a symmetric collapse; at most 0.125 W = 1717 N per side.

---------------------------------------------------------------------------

## 4. Channel map (direct entries; Q entries also reach the other channels through M_t^{-1})

| item | surge | sway | yaw | heave | pitch | reads waves through |
|---|---|---|---|---|---|---|
| W1 | Q | - | - | Q + acc | Q + acc | hc_i, tau_e, z_ref |
| W2 | - | - | - | Q or acc | Q or acc | e_z, e_th, w_r, q_r |
| W3 | imp | imp (asym.) | imp (asym.) | imp | imp | h_ij, dd_ij (port/stbd points) |
| W4 | - | - | - | Q | Q | hc_i |
| W5 (a) | - | acc (side walls) | acc (side walls) | Q | Q | hc_i, h_i0, h_i2 |
| W5 (b) | Q | - | - | acc | acc | (U_r only) |
| W6 | Q | - | - | Q (s_p theta F_x) | Q | hc_4, tau_e |
| W7 vert | - | - | - | acc (implicit) | acc (implicit) | h_ij, dd_ij |
| W7 lat | - | acc (implicit) | acc (implicit) | - | - | hc_i, v_orb,i1 |
| W8 | Q | acc | acc | acc | acc | d_ij fluctuation, quadratic |
| W9 | Q | - | - | - | - | longitudinal slope, u_orb |
| W10 | - | Q | Q (+ acc mod) | - | - | hc_3, hc_4, slope |
| W11 | Q | - | - | acc | acc | (Fn_vol only) |
| W12 | Q | Q | Q | Q | Q | h at x_s (3 points), tau_e |

---------------------------------------------------------------------------

## 5. The three lateral questions

Reference: the low-fidelity lateral law the items add to (control/reduced.py:443-453, 477, 482-488):
m_v vdot = lift - k_lin_sway v - k_sway v |v| - m u r + c_wave_sway g t_slope, with c_wave_sway = 0;
r follows r_t = k_nomoto_f lift with lag tau_r, plus c_wave_yaw g t_twist dt, c_wave_yaw = 0;
lift = k_js max(thr, 0) sin(clip(noz)). Its sway damping does not depend on speed or immersion, and it has no lateral wave forcing.

### 5.1 Aft (stern) lateral damping that varies with speed and draft
Target form for comparison: Y_aft = -c(U, h_aft) (v + x_aft r), N_aft = x_aft Y_aft, x_aft < 0, c linear in U (lifting-surface sway damping) and increasing with the aft immersion h_aft.
- No W item has this form.
- W7 lateral (cat_water_b.py:214-222) is the only W item whose lateral damping depends on the local draft: c_i proportional to max(hc_i, 0) Dx_i, applied at every station, so stern-weighted by the calm immersion distribution (57 % on stations 0-1). Its form is quadratic, F_y,i = -c_i |w_i| w_i: zero slope at w_i = 0 (no linear damping for small sideslip; equivalent linear rate 2 c_i |w_0| about a sideslip w_0), and no explicit speed factor. Speed acts only through hc_i (the attitude shifts of W1 / W5(b) / W11 / layer-1 T4, section 2 note). Bounds: C_Dl in [0.3, 1.5]; at calm running sum_i c_i in [115, 576] N s^2/m^2, stern station 0 in [25, 127], station 1 in [40, 199]. Item cap 8.09 m/s^2 sway, 5.27 rad/s^2 yaw. Marginal on-rate of the lateral part: 0.7 (non-sparse).
- W10 (cat_water_b.py:345-360) has the linear-in-velocity, speed-proportional, immersion-dependent form, Y = -kap_b k_lin_sway (U_r / 10 m/s) ((h_bow - d_0)_+/T)^{q_b} (v_r + x_b r), but only at a BOW point x_b in [0.25 L, 0.5 L] = [1.35, 2.70] m and only reading the bow immersion (stations 3-4). It cannot represent an aft damper (x_b > 0 always). Bounds: kap_b in LogU[0.1, 2], q_b in [1, 2], d_0 in [0, 0.5] T; c <= 672 N s/m at u_id in calm running (calm bow immersion 0.060 m). On-rate 0.4.
- W12 collapse (cat_water_b.py:481-487) can act aft of the centre (x_c in [-1.61, 0.52] m) and depends on sideslip through h_eff,j, but it is a switched force (smoothstep of beta_s, first-order lags), zero when the step is ventilated (calm running above U_v), not a damping law; on-rate 0.3.
- Outside the W items (pointers only, not derived here): layer-1 T3 point damper F = -c(xi) (v + x_pd r), x_pd ~ U[x_stern, x_bow], c(xi) = exp(polynomial in xi), xi = clip((u_r - u_mid)/u_half, -1, 1) (operators_rb.py:403-407, 784, 801-805): speed-dependent, immersion-independent, random point (aft possible). Layer-1 Teven lateral term: sway-yaw damping matrix modulated by tanh(e_z / (0.5 T)), tanh(e_th / (T/L)), tanh(w_r / ...), tanh((u - u_mid)/u_half); the modulation matrix X = sum_k n_k beta_k G_k has spectral norm <= sum_k beta_k, sum_k beta_k ~ U[0.1, 0.5], applied in the damping-normalised coordinates of the current linear sway / yaw damping (operators_rb.py:472-482, 884-899). H1 damping multiplier (1 + e_phi phi^2) on the low-fidelity sway / yaw damping (cat_hidden_boat.py:266-275).

### 5.2 Lateral cross-flow
- W7 lateral is the crossflow-drag item: F_y,i = -(1/2) rho_w C_Dl max(hc_i, 0) Dx_i |w_i| w_i, w_i = v_r + x_i r - c_orb v_orb,i1, per station, yaw = sum_i x_i F_y,i, implicit update. Local "draft" is the centre-line keel immersion (no chine / deadrise geometry, no separate side-wall height, no roll). One coefficient C_Dl for all stations. The orbital term c_orb v_orb uses the surface orbital velocity at the centre point, c_orb in [0, 1].
- When W7 lateral is on, layer-1 T3's quadratic sway / yaw terms (operators_rb.py:807-808) are switched off (cat_base.py:122, 795-796).
- No other W item has a crossflow form. W10 is linear in w (lift type), at the bow.

### 5.3 First-order lateral wave-slope forces
- No W item reads the transverse slope t_slope (grep: the only reader in cat_*.py is H1, cat_hidden_boat.py:186).
- W5(a) side walls (cat_water_a.py:434-439) are the only W term that is LINEAR in the port-starboard immersion difference, and only piecewise: Dh = sum_i [(h_i2 - d_c,i)_+ - (h_i0 - d_c,i)_+]. Where both sides of station i are above the chine depth, that station contributes h_i2 - h_i0 = (eta_i2 - eta_i0) - B phi = B (tsl_i - phi), tsl_i the station's transverse slope; where one side is above, a one-sided ReLU term; where neither, nothing. acc_sway = n_v 0.3 a_ref,sway (Dh / 0.1 m)(U_r/u_id)^2 = 8.09 n_v Dh (U_r/u_id)^2 m/s^2 (Dh in m), i.e. 16.4 n_v tsl_i (U_r/u_id)^2 m/s^2 per fully wetted station; acc_yaw = 5.27 n_r Dh (U_r/u_id)^2 rad/s^2, same Dh (no station lever arm). Signs n_v, n_r ~ U[-1, 1] independent. Active only for stations whose keel immersion exceeds d_c,i: at calm running the stern station (hc_run,0 = 0.251 m) is above the chine when d_c0 < 0.251 m (probability 0.29 with W5's own d_c0; with W3's d_c in [0.137, 0.538] m, by W3's draw); the bow stations essentially never (hc_run,4 = 0.033 m < 0.115 m). The speed factor is (U_r/u_id)^2, not tied to a hydrostatic g t_slope. Capped at 6 m/s^2 sway and 3 rad/s^2 yaw (half-clip nominal). On-rate of (a): 0.6.
- W3 lateral impulse F_y,i = (J_i0^+ - J_i2^+) tan beta_i (cat_water_a.py:252-253): needs a growing added mass (entry, Dm > 0 with V > 0) on one side more than the other: an impact term, product of immersion growth and entry velocity, not a slope force.
- W7 lateral orbital forcing (cat_water_b.py:217-218): with the hull not moving sideways, F = c_i |c_orb v_orb| c_orb v_orb, quadratic in the orbital velocity (second order in wave amplitude); the first-order part exists only through the cross term 2 c_i |v_r + x_i r| c_orb v_orb, i.e. proportional to the hull's own lateral velocity (zero in straight running).
- W8 sway / yaw (cat_water_b.py:271-286): zeta_s^T Q zeta_a, second order, low-passed with tau2 in [2, 30] s (drift type).
- Outside the W items (pointer): H1 gives Y = -W (a1 phi + a3 phi |phi|), N = W (l1 phi + l3 phi |phi|) + ... (cat_hidden_boat.py:252-263), with phi driven towards phi_eq containing c_w t_slope, c_w ~ U[0, 1], a1 ~ U[-0.3, 1] (cat_hidden_boat.py:148, 152, 185-187): a lateral force linear in t_slope filtered by the roll oscillator (w0 ~ LogU[1.5, 5.5] sqrt(B_REF/B) rad/s). Layer-1 T7 random environmental loads (operators_rb.py:63-71) read wave patterns along random generalised directions and may contain lateral first-order components: UNKNOWN here (not derived in this task).

---------------------------------------------------------------------------

## 6. Docstring / code disagreements and UNKNOWNs

1. W4 publishes `ctx.w4_r_exit` "for the velocity parts of W1 / W7" (cat_water_a.py:36-37, 296-297). No module reads it (grep over learn/meta/*.py). W7 uses its own r_exit; W1 has no velocity part with r_exit. W4's r_exit has no effect.
2. W5 docstring: "H1's roll restoring multiplier k_roll and ds are published in q['W5b'] for H1" (cat_water_a.py:349-350). q['W5b'] is written (453) and never read; also H1 runs in stage 'hidden' before stage 'force' (cat_base.py:1053-1082), so it could not read the same substep's value. k_roll has no effect.
3. W5 docstring: k1_i makes the stiffness equal "w_h^2 m_heave / 5"; code uses 1 / (M_t^{-1})_33 as the heave mass (cat_water_a.py:428), i.e. the draw's M_t, not a low-fidelity mass.
4. W7 docstring: chi is "W4's form and ranges" (cat_water_b.py:18-19). Code: plain logistic sigma((h - h_off)/w_d) without W4's normalisation to 1 at the running immersion (cat_water_b.py:208); ranges equal.
5. D10 draft table (PRIOR_D10_DRAFT.md:52) lists t_slope as used by W9, W10, H1. Code: W9 uses the local longitudinal gradient (cat_water_b.py:314), W10 the longitudinal least-squares slope (366); only H1 uses t_slope.
6. control/reduced.py:374 comments the lateral order as "port / centre / starboard"; the station offsets used everywhere in the catalogue are (-B/2, 0, +B/2) with y to port (cat_base.py:351), i.e. index 0 = starboard. The formula (eta_2 - eta_0)/B is the same in both files; only the comment label differs.
7. W11 uses the code name w_h for the bump width; the heave natural frequency is ctx.wh. acc_heave uses ctx.wh (natural frequency) and prm['z_h'] (cat_water_b.py:420).
8. UNKNOWN: the sea-state distribution used when the D10 pretraining data are generated (which seas, which speeds) is not part of these files; ranges quoted as "calm running" values use h = hc_run.
9. UNKNOWN: whether the E1 current / E4 packet items are on in the target recordings' counterpart draws; they change u_r, v_r, eta, u_orb, v_orb read by the W items (cat_air_env.py:489, 754-758); not derived here.
10. Every W item is deterministic given state, parameters and its hidden states (no random number is drawn in any `step`).
