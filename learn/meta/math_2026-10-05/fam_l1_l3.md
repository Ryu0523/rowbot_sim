# The D10 catalogue family: layer 1, layer 3 and the top-level composition, derived from the code

Working tree: `C:/Users/Administrator/Documents/usv-rl-mpc`, branch `usv-priors-mpc-2026-10-01`, read on 2026-10-04.
Every equation cites `file:line` of that tree. Files: `learn/meta/operators_cat.py`, `learn/meta/cat_base.py`,
`learn/meta/operators_rb.py`, `learn/meta/operators_gen.py`, `learn/meta/operators.py`, `learn/meta/ops_m15.py`,
`control/reduced.py`, `learn/meta/episode7.py`, `learn/meta/cat_water_a.py` (W4 only), `learn/meta/cat_hidden_boat.py` (B1 only).
Layer-2 catalogue items (`cat_*.py`) are NOT derived here, except the structural contact gate W4 and the final
mapping of the propulsion chain, which the composition needs.

## 0. Corrections to the task statement and docstring/code disagreements found

| # | Statement | What the code does |
|---|---|---|
| C1 | "operators_rb.py (OperatorRBL1 ...)" | `OperatorRBL1` is defined in `cat_base.py:760-768` as a subclass of `operators_rb.OperatorRB`; its only change is `stable()` returning `True`. `apply_l1_switches` and `l1_post` are also in `cat_base.py:771-814`. |
| C2 | operators_rb docstring: "the rigid-body surge term m v r ... (always on, structural)" (`operators_rb.py:52-53`) | In D10 that term is computed (`operators_rb.py:863`) and then subtracted again (`cat_base.py:809-810`, switch `T2c_rigid_off` is in `ALWAYS_SWITCHES`, `cat_base.py:112-114`). Net: zero in every D10 draw. |
| C3 | "layer 3 M_t is the shared inertia of every lever arm" (`cat_base.py:909`, `operators_cat.py:17`) | Layer 1 never uses M_t: it divides by its OWN diagonal masses `(m_u, m_v, I_r, m_w, I_q)` drawn in `operators_rb.py:383-388` (`_gen`, `operators_rb.py:746-749`). When item B1 is on, `B1.adjust_ctx` replaces the context's M_t by a loaded-boat matrix (`cat_hidden_boat.py:544-547`); layer-2 items then use that matrix, while layer 3's own forces and Coriolis term keep the unloaded `op_gen.Minv` (`operators_gen.py:809, 851, 905`). So up to three different inertias are in use in one draw. |
| C4 | `CatDraw.substep` passes `nu0` to items and to layer 3 | Items read `q['nu0']`, whose heave and pitch entries are REPLACED by the analytic low-fidelity accelerations at the substep start (`cat_base.py:520-524`); layer 3 receives the caller's numerical increment `nu0` unchanged (`cat_base.py:1095`). |
| C5 | Heave row of a point force | Layer 3 and layer 2 use the earth-vertical heave row `Fz + s_p θ Fx` (`operators_gen.py:815`, `cat_base.py:310`); layer-1 T7 point loads use `Fz` without the `θ Fx` term (`operators_rb.py:758-759`). The pitch rows agree only because `s_p = -1` for this hull (`operators_rb.py:759` has no `s_p`; `cat_base.py:311` has it). |
| C6 | Soft saturation "of every item" | Only items run through `item_accel` (`cat_base.py:724-740`) are soft-saturated. The propulsion chain is added unsaturated (`cat_base.py:1083-1085`, `1159`); W4 has `saturate=False` (`cat_water_a.py:289`); layer 1 has no per-term saturation; layer 3 saturates each readout output (`operators_gen.py:765`). |

## 1. Symbols (defined once)

### 1.1 State
`s = (x, y, u, z, ż, θ, θ̇, ψ, r, v)`: the 10-number reduced state (`cat_base.py:531`, `control/reduced.py:369`).
- `x, y` [m]: earth-fixed horizontal position of the reference point; `ψ` [rad]: heading.
- `u, v` [m/s]: surge and sway velocity over ground, body axes; `r` [rad/s]: yaw rate.
- `z` [m]: earth-vertical position of the reference point, up positive; `ż` its rate.
- `θ` [rad]: pitch in the low-fidelity convention: a hull point at body position x has height `z + s_p x θ`, `s_p = sign_pitch = -1` (`cat_base.py:537-538`), so `θ > 0` is bow down; `θ̇` its rate.
- Channel vector `ν = (u, v, r, ż, θ̇)` = `s[(2, 9, 8, 4, 6)]` (`operators_rb.py:92`); in the 14-number plant state these are indices `(6, 7, 11, 8, 10)` (`operators_rb.py:93`).

### 1.2 Inputs
- `T_a` [N], `δ_a` [rad]: the APPLIED thrust and nozzle angle over a substep, the values the low-fidelity step receives (`operators_cat.py:508-517, 525`). With ideal actuators they equal the commands.
- `T_c = U_0 t_max`, `δ_c = U_1 rud_max`: the commands (`operators_cat.py:500, 503`); `U ∈ [0,1]×[-1,1]` command fractions.

### 1.3 Waves (exogenous inputs)
- Stations `x_i = linspace(x_stern, x_bow, 5)`, i = 0 stern ... 4 bow; lateral offsets `y_j = (-B/2, 0, +B/2)`, j = 0 starboard (`cat_base.py:350-351`). Values: `x_i = (-2.125, -0.803, 0.519, 1.841, 3.163)` m.
- Sea of a row: `η(X, Y, t) = Σ_n a_n cos φ_n`, `φ_n = k_n (X cos β_n + Y sin β_n) - ω_n t + ϕ_n` (`cat_base.py:463-467`).
- `η_ij` [m]: elevation at the earth position of station (i, j) at the substep START; `η̇_ij` [m/s]: its rate along the MOVING station, `Σ_n a_n (ω_n - k_n (Ẋ cos β_n + Ẏ sin β_n)) sin φ_n` (`cat_base.py:468-469`); `u_orb, v_orb`: body-axis horizontal orbital velocity `Σ a_n ω_n cos φ_n` along each component direction (`cat_base.py:470-473`); `η_in`: elevation at the intake point (`cat_base.py:476`).
- Centre-line `η_c,i = η_i1`; `x̃_i = x_i - mean(x)`, `D = Σ x̃_i²` (`cat_base.py:352-353`); wave slope `σ = Σ η_c,i x̃_i / D`, its rate `σ̇ = Σ η̇_c,i x̃_i / D` (`cat_base.py:547, 552`).
- Low-fidelity water reference (`cat_base.py:553-558`, identical to `operators_rb.py:728-744`):
  `z_ref = k_wh mean_i η_c,i + z0`, `θ_ref = k_wp s_p arctan σ + θ0`, `ż_ref = k_wh mean_i η̇_c,i`, `θ̇_ref = k_wp s_p σ̇ / (1 + σ²)`;
  `e_z = z - z_ref`, `e_θ = θ - θ_ref`, `w_r = ż - ż_ref`, `q_r = θ̇ - θ̇_ref`. `k_wh = k_wp = 1` (keys absent in the parameter file, default 1.0).

### 1.4 Fixed parameters (low-fidelity model, `R.default_params()` = `studies/_cache/lofi_scarab195.json`, `operators_rb.py:126-132`; values printed by a check)
`L = 5.4`, `T = 0.3846` (draught), `B = 2.0252`, `x_stern = -2.125`, `x_bow = 3.163`, `s_p = -1`, `z0 = 0.1537`, `θ0 = -0.00923`,
`m = m_coriolis = 1400`, `m_u = m_surge = 1968.6`, `m_v = m_sway = 1610.2`, `k_drag = 13.99`, `k_lin_sway = 1679.9`, `k_sway = 391.9`,
`τ_r = 0.42`, `k_nf = k_nomoto_f = -1.699e-4`, `k_js = k_jet_side = 1.614`, `t_max = 6365.7`, `rud_max = rud_stall = 0.4363`,
`ω_h = wn_heave = 5.835`, `ζ_h = 0.3723`, `ω_p = wn_pitch = 3.713`, `ζ_p = 0.1551`, `u_des = 12.861`, `u_max = 23.515`, `v_slam = 0.6769`,
`c_wave_sway = c_wave_yaw = 0`, `steer_jet = 1` (SI units).
Hull scale `rs = sqrt(L / 10 m) = 0.7348` (`operators_rb.py:285`). Task speed range `U_RANGE = (16, 30) kn = (8.231, 15.433)` m/s, `u_mid = 11.832`, `u_half = 3.601` (`operators_rb.py:98, 295-297`).

### 1.5 The low-fidelity step the family is added to (`control/reduced.py:366-517`)
- surge `u̇ = (T_a - k_drag u|u|)/m_u`, explicit Euler (`reduced.py:394-395, 491`);
- heave `z̈ = -ω_h²(z - z_ref) - 2ζ_h ω_h ż`, pitch `θ̈ = -ω_p²(θ - θ_ref) - 2ζ_p ω_p θ̇`, each by its exact 2 × 2 transition with the station elevations held over the substep (`reduced.py:397-424`);
- sway `v̇ = (λ - k_lin_sway v - k_sway v|v| - m u r)/m_v`, `λ = k_js max(T_a, 0) sin(clip(δ_a, ±rud_stall))` (`reduced.py:443-453`);
- yaw: exact update of `ṙ = (k_nf λ - r)/τ_r` (`reduced.py:482-488`).
No wave forcing in surge, sway or yaw (`c_wave_* = 0`, `reduced.py:477, 488`).

### 1.6 Reference scales
- `A_REF = (4, 4, 2, 12, 3)` (`operators.py:63`), `CLIP = 3` (`operators.py:67`), clip limit `LIM = CLIP · A_REF = (12, 12, 6, 36, 9)` in (m/s², m/s², rad/s², m/s², rad/s²) (`operators_cat.py:71`).
- `a_ref = a_ref_of(p) = (t_max/m_u, S/m_v, S|k_nf|/τ_r, ω_h² T, ω_p² T/(L/2))`, `S = k_js t_max sin(rud_max)` (`operators_gen.py:165-174`) `= (3.234, 2.697, 1.756, 13.09, 1.964)`.
- `LogU[a, b]` means `exp(U[ln a, ln b])` (`operators.py:85-86`); `U[a, b]` uniform; `N(μ, s²)` normal.

## 2. Top-level composition

### 2.1 What one draw is, and the order of the draw
A draw is `CatDrawD10(seed, attempt)` (`operators_cat.py:114`, `cat_base.py:873-932`). Every random stream is
`default_rng([seed, 71, attempt, id, ...])` (`cat_base.py:77, 934-936`). In training, `seed = op_seed = 7 · episode_seed + 1` (`episode7.py:144`).
Order (`cat_base.py:904-931`), meta stream id 1:
1. `τ0 ~ U[2°, 6°]` hidden true running trim (`cat_base.py:905, 147`; used by layer-2 items only).
2. `sparse ~ Bernoulli(0.15)` (`cat_base.py:906-907, 145`).
3. layer-3 tier: small 0.75, fail 0.20, none 0.05 (`cat_base.py:908, 141, 817-818`; `operators_gen.py:139-145`).
4. Layer 3 (`OperatorGenL3`, stream id 7) is built FIRST; its M_t is copied into the context (`cat_base.py:910-912`).
5. Catalogue items (stream id `100 + index`) and the overlap switches they imply (`cat_base.py:913-918`).
6. Layer 1 (`OperatorRBL1`) with seed `s1 = SeedSequence([seed, 71, attempt, 3]).generate_state(1)[0]`, switches applied (`cat_base.py:920-926`).
7. If layer 1 drew T5 (and is not off): context jet point `(x_j, z_j)` and axial-loss share `k_c` taken from layer 1 (`cat_base.py:927-930`).

Acceptance (`operators_cat.py:329-375`): two rows (a calm row started perturbed: `z += T/4`, `θ += s_p T/(4 x_b)`, `ψ = 0.3`, `r = 0.05`; a standard-sea row Hs 1 m, Tp 5.5 s), 23 kn start, 250 control steps = 60 s, the data autopilot, ideal actuators, `noise=False` (`operators_gen.py:121-128, 1297-1312`). Only criterion: divergence = non-finite, or `u > 1.5 u_max`, or `|θ| > π/4` at any control-step end (`operators_cat.py:577-582`). A rejected draw is replaced by `attempt + 1`, which changes EVERY stream (layer 1, layer 3, items). After 30 rejections: `forces_off_now` = items `(W4,)` only, tier `none`, layer 1 kept (`operators_cat.py:209-217, 365-370`). So the trained-on distribution is the prior conditioned on passing this test. Measured on `studies/_cache/meta7/train_meta.pkl` (9860 kept episodes): 467 rejections = 4.5 % of attempts, 0 draws with forces off; the highest per-item rejection share was 5.5 % (P11), i.e. no item dominates the rejections.

### 2.2 The control-step part (held over 6 substeps)
Control step `Δ = 0.24 s = 6 h`, substep `h = 0.04 s` (`operators_cat.py:427-428`). Once per control step `k`,
`(r_k, n_k) = CatDraw.step(s28_k)` (`cat_base.py:1002-1014` -> `operators_rb.py:1057-1130`):
- `r_k ≡ 0`: the T8 rule is off (`T8_off` always, `cat_base.py:112-114, 775-776`), so `ym = 0 · ym`, relays and events are multiplied by `on8 = 0` (`operators_rb.py:1074-1075, 1088-1093, 1120`), and `A · e^{ga} · 2.5 tanh(0) = 0`.
- `n_k` = the T9 noise (section 5.11).
- Held values: `r̄_k = clip(0, ±LIM) = 0`, `n̄_k = clip(n_k, ±LIM)` (`operators_cat.py:494-499`, `operators.py:76-82`).
- The input `s28_k` (`operators_rb.py:1405-1417`): `[u/u_ref - 1, v, r L/u_ref, cos ψ, sin ψ, T/t_max, δ/rud_max, ż, θ̇, (z - z0)/0.2, (θ - θ0)/0.05, 15 elevations at the stations at the step midpoint (pose dead-reckoned, operators_rb.py:1420-1428), 2 command fractions]`.

### 2.3 The substep update (exact arithmetic of `simulate_cat`; `CatInjector` is the same, `operators_cat.py:697-737`)
At substep j of control step k, start state `s_j`, time `t_j = t_k + j h` (`operators_cat.py:506-507`):
1. Low-fidelity step: `s^lo = F_lo(s_j, T_a, δ_a, η(s_j, t_j), h)` (`operators_cat.py:525`).
2. `ν0 = (ν(s^lo) - ν(s_j)) / h` (`operators_cat.py:526`).
3. Family substep: `(a_j, ι_j) = CatDraw.substep(s_j, T_a, δ_a, sea(s_j, t_j), ν0, (T_c, δ_c))` (`operators_cat.py:530-535`): `a_j` ∈ R^5 acceleration, `ι_j` ∈ R^5 velocity jumps (impulse channel).
4. Clipped push `p_j = clip(a_j + r̄_k + n̄_k, -LIM, +LIM)` per channel (`operators_cat.py:542-543`; the two outputs of `clip_push(a + r̄, n̄)` always sum to `clip(a + r̄ + n̄)`, `operators.py:80-82`).
5. Impulse budget: `I_{-1} = 0` at each control step (`operators_cat.py:505`); `I_j = clip(I_{j-1} + ι_j, ±LIM Δ)`, applied `ι'_j = I_j - I_{j-1}` (`operators_cat.py:550-555`, cap `LIM Δ = (2.88, 2.88, 1.44, 8.64, 2.16)`, `operators_cat.py:429`). Impulses are not soft-saturated and not clipped per substep.
6. `s_{j+1} = s^lo`, then `ν(s_{j+1}) += h p_j + ι'_j` (`operators_cat.py:565-568`).

### 2.4 The family acceleration of one substep (`cat_base.py:1029-1108`)
```
a_j = a_L1                                   (layer 1, section 5)
    + Σ_{c on, stage env/hidden/force/gate} sat_c(a_c)   (layer 2 items; W4 unsaturated)
    + a_prop                                 (propulsion chain, unsaturated)
    + a_L3 + a_cor                           (layer 3 forces + Coriolis difference, section 6)
ι_j = Σ_c ι_c                                (items W3 slam, E3 debris)
```
Verified numerically: with `parts=True`, the sum of all parts equals the applied mean push to `2.7e-15` on an unclipped row (check 3, section 10).

### 2.5 What the network is trained on
`E_k = (ν_true(t_k + Δ) - ν_twin(t_k + Δ)) / Δ` (`episode7.py:229`), where the twin is the low-fidelity plant started from the same state `s0` with the same commands and actuator state and no injection (`episode7.py:155, 219-223`). T9 noise is ON in training episodes (`episode7.py:214`, `noise=True`), OFF in the acceptance test (`operators_cat.py:339-340`).
Relation of `E_k` to the pushes: `E_k` is the 0.24-s velocity difference of two low-fidelity flows, one of which receives `h p_j + ι'_j` after each substep. To first order in the kicks, `E_k = Δ^{-1} Σ_{j=0}^{5} Φ_j (h p_j + ι'_j)` with `Φ_j = ∂ν(t_k + Δ)/∂ν(t_j^+)` of the low-fidelity flow (a linearisation, not an identity: the heave/pitch transitions are affine in the state for fixed station elevations, `reduced.py:418-424`, but the elevations depend on `x, y, ψ`, which the kicks change; surge drag `u|u|`, sway `v|v|` and `m u r` are nonlinear).

## 3. Shared quantities of a substep (`cat_base.py:495-597`)
From the START state `s_j`, `T_a`, `δ_a`, the sea sample, `ν0`, the commands:
- `a_lo,v = (-ω_h²(z - z_ref) - 2ζ_h ω_h ż, -ω_p²(θ - θ_ref) - 2ζ_p ω_p θ̇)` analytic (`cat_base.py:480-492`); `q['nu0'] = (ν0_u, ν0_v, ν0_r, a_lo,v)` (`cat_base.py:520-524`).
- `derive_quantities` (`cat_base.py:528-578`), with `η = η0 + η_add`, `η̇ = η̇0 + η̇_add` (env items add), roll `φ, φ̇` (hidden item H1, else 0):
  hull height `z + s_p x_i θ + y_j φ`; station immersion `d_ij = η_ij - hull_ij`; entry velocity `ḋ_ij = η̇_ij - (ż + s_p x_i θ̇ + y_j φ̇)`; keel immersion `h_ij = d_ij - k_i` with keel heights `k_i` (task hull: `(-0.385, -0.342, -0.300, -0.258, -0.216)` m, `operators_cat.py:76-101`); centre line `h_c,i = h_i1`, `ḋ_c,i = ḋ_i1` (`cat_base.py:536-545`);
  water-relative velocities `u_r = u - ν_c,u`, `v_r = v - ν_c,v` (`ν_c` body-frame current from item E1, else 0), `U_r = hypot(u_r, v_r)`, `q_dyn = ½ ρ_w U_r²` (`cat_base.py:570-577`);
  trim `τ_e = softplus_floor(τ0 + s_p(θ - θ0) - arctan σ, 0.5°, 0.5°)` (`cat_base.py:562-564`); intake immersion `h_in` (`cat_base.py:566-568`).
- Stage order: env items, re-derive; hidden items, re-derive (`cat_base.py:1054-1059`).
- `slam_events` (`cat_base.py:589-597`, after the re-derivation, `cat_base.py:1060`): `out = (h_c,4 ≤ 0)`; `slam = emerged_prev ∧ ¬out ∧ (ḋ_c,4 > v_slam)`; `emerged ← out`. Reported only; not added to `a`.
- Calm-running keel immersion `h_run,i = -(z0 + s_p x_i θ0) - k_i = (0.251, 0.196, 0.142, 0.087, 0.033)` m (`cat_base.py:386-387`, values from check 4).

## 4. Layer order inside `CatDraw.substep`, item mapping, the gate

### 4.1 Order (`cat_base.py:1053-1102`)
`env` items → re-derive → `hidden` items → re-derive → slam detector → **layer 1** at `(u_r, v_r)` (`sr_hydro`, `cat_base.py:581-586, 1065-1078`) → `force` items → propulsion chain in `PROP_ORDER` (`cat_base.py:134-137, 1083-1088`) → `gate` item W4 (`cat_base.py:1089-1091`) → **layer 3** forces + Coriolis (`cat_base.py:1093-1102`).

### 4.2 Item mapping and soft saturation (`item_accel`, `cat_base.py:724-740`)
An item returns acceleration `a^o` and/or generalised force `Q` and/or jumps `ι`. `a_c = M_ctx^{-1} Q + a^o` (`to_acc`, `cat_base.py:314-316`; `M_ctx` is layer 3's M_t, or B1's loaded matrix when B1 is on). If `saturate`: `sat(a)_c = 3 n_c tanh(a_c / (3 n_c))` (`cat_base.py:161-164`), `n` = `item.nominal` = `a_ref` by default (`cat_base.py:678-680`); W1, W2, W5 use `min(a_ref, LIM/6) = (2, 2, 1, 6, 1.5)` (`cat_base.py:167-177`). Gated items (`gated=True`) add their SATURATED heave/pitch accelerations to `a_cat,v` (`cat_base.py:1046-1047`).

### 4.3 The contact gate W4 (structural, always on; `cat_water_a.py:276-323`)
Parameters `h_off ~ U[-0.05, 0.05]` m, `w_d ~ U[0.02, 0.1]` m, `r_exit ~ U[0, 0.3]` (`cat_water_a.py:291-294`; `r_exit` is published for W1/W7).
- `χ_i = min(1, S((h_c,i - h_off)/w_d) / S((h_run,i - h_off)/w_d))`, `S` logistic (`cat_water_a.py:300-303`).
- Vertical hydrodynamic acceleration `A = a_lo,v + a_L1,v + a_cat,v + (g, 0)` (`cat_water_a.py:309-311`), with `a_L1,v` = heave/pitch columns of layer-1 parts T2c, T3, T4, T7, Teven (`cat_base.py:131, 1074-1075`; NOT T5, NOT T9).
- `Q_hp = M_hp A`, `M_hp = ((M_ctx^{-1})_{[3:5,3:5]})^{-1}` (`cat_water_a.py:64-67, 312`).
- Spread over stations with `w_i = max(h_run,i, 0)`: `f_i = w_i (α + β x_i)`, `(α, β)` solving `[[Σw, Σw x],[Σw x, Σw x²]] (α, β)ᵀ = (Q_hp,1, s_p Q_hp,2)ᵀ` (`cat_water_a.py:313-318`).
- `Q_heave = Σ_i (χ_i - 1) f_i`, `Q_pitch = s_p Σ_i (χ_i - 1) f_i x_i`, other rows 0; `a_W4 = M_ctx^{-1} Q` (full 5 × 5, so CG couplings of M_t carry it into surge, sway and yaw) (`cat_water_a.py:319-323`). Not saturated. The propulsion chain's vertical force (P6) is not in `a_cat,v` and is not gated.

### 4.4 Propulsion chain output (layer 2, final mapping only; `cat_base.py:1110-1159`)
`F = (F_x - F_x0 + dF_x, F_y - F_y0 + dF_y, F_z)`, `F_x0 = T_a (1 + k_c (cos δ_a - 1))`, `F_y0 = k_js max(T_a,0) sin(clip(δ_a, ±rud_stall))`; `Q = lever_Q((x_j, 0, z_j), F, θ)` with `lever_Q = (F_x, F_y, x F_y - y F_x, F_z + s_p θ F_x, s_p(x F_z - z F_x))` (`cat_base.py:298-311`); `a_prop = M_ctx^{-1} Q`. The nominal `F_x0` already contains layer-1's `k_c`, so the axial-loss part stays in layer 1.

## 5. Layer 1 (`OperatorRBL1` = `operators_rb.OperatorRB` with `stable() ≡ True`)

### 5.1 Draw procedure and streams
`OperatorRB.__init__(s1, lib, dt=0.24, L, p)` (`operators_rb.py:251-279`):
- `res = OperatorM15(s1, lib, relay=False, null=None)`: the old v0/M15 operator, used for T9 only (`operators_rb.py:256-257`); its streams `default_rng(s1)`, `[s1, 13]`, `[s1, 14]` (`operators.py:200`, `ops_m15.py:143-144`), slow stream `[s1, 2]` (`operators.py:495`).
- `rng = default_rng([s1, 31])` (`operators_rb.py:258`): `w_res ~ LogU[0.1, 1]` (unused: T8 off), `w_noise ~ LogU[0.1, 1]`, then `rb_off ~ Bernoulli(0.05)` (`operators_rb.py:107, 259-263`).
- If not `rb_off`: exactly ONE call of `_draw(rng)` (the loop at `operators_rb.py:265-272` breaks at once because `stable()` is `True`, `cat_base.py:767-768`); then `_draw_env(rng, lib)` (T7) (`operators_rb.py:275`). If `rb_off`: neutral values (`_draw_off`, `operators_rb.py:346-374`) and `rb_accel ≡ 0` (`operators_rb.py:776-777`).
- `enabled = {T3, T4, T5, T2c, Teven, T7, T8, T9}` minus `T8` (`operators_rb.py:277`, `cat_base.py:775-776`).

### 5.2 Layer-1's own inertias (not M_t) (`operators_rb.py:383-392`)
`I_r = m_v ρ_r²`, `ρ_r = LogU[0.22, 0.37] L`; `m_w = m (1 + a33)`, `a33 = LogU[0.3, 2.5]`; `I_q = m_w ρ_q²`, `ρ_q = U[0.18, 0.27] L`; `A11 = m U[0.02, 0.1]`; `A22 = max(m_v - m, 0.05 m) = 210.2` kg; wetted length `l_w = U[0.4, 1.0] (x_bow - x_stern)`.
Masses vector `M1 = (m_u, m_v, I_r, m_w, I_q)`; generalised force → acceleration by `Q / M1` elementwise (`operators_rb.py:746-749`).
Yaw scale `f_r = (x_n0/I_r)/(k_nf/τ_r) = I_r0/I_r`, `x_n0 = x_stern`, `I_r0 = x_n0 τ_r / k_nf = 5253.5` kg m² (`operators_rb.py:294, 311, 395`). Since `ln ρ_r` is uniform, `f_r ~ LogU[0.817, 2.31]`, `P(f_r > 1) = 0.81`.

### 5.3 Smooth functions of speed (`operators_rb.py:148-160, 615-617`)
`ξ = clip((u_r - u_mid)/u_half, -1, 1)` (`operators_rb.py:784`); `g_c(ξ) = c0 + c1 ξ + c2 (3ξ² - 1)/2`.
`_smooth(c0, σ_s)`: w.p. 0.4 `(c0, 0, 0)`; else `ρ ~ U[0.3, 0.8]`, `c1 ~ N(0, (σ_s ρ)²)`, `c2 ~ N(0, (σ_s ρ²)²)`.
A coefficient triple that is not drawn stays `(0,0,0)`, so `exp(g) = 1` (neutral).

All layer-1 terms read `s_h` = `s` with `(u, v)` replaced by `(u_r, v_r)` (`cat_base.py:1066`); below `u, v` mean `u_r, v_r`. `drag0 = k_drag u|u|` (`operators_rb.py:790`).

### 5.4 T3 dissipation (on w.p. `P_T3 = 0.85` given not rb_off; `operators_rb.py:397-410, 792-810`)
Random parameters: `c_R = _smooth(U[ln 0.8, ln 1.25], 0.15)`; `c_S,k = _smooth(U[ln 0.6, ln 2.4], 0.2)`, k = 0..3; point damper w.p. 0.5: `x_pd ~ U[x_stern, x_bow]`, `c_pd = _smooth(ln(LogU[0.05, 1] m_v D0_v), 0.2)`; quadratic w.p. 0.3: `κ_k = LogU[0.05, 1] D0_{k+1} / v_typ,k · 1[U < 0.6]`, k = 0..3.
`D0 = (2 k_drag u_mid/m_u, (k_lin_sway + 0.08 k_sway u_mid)/m_v, 1/τ_r, 2ζ_h ω_h, 2ζ_p ω_p) = (0.168, 1.274, 2.381, 4.345, 1.152)` (`operators_rb.py:300-305`); `v_typ = (0.08 u_mid, 0.08 u_mid/(0.35 L), 0.05 sqrt(gL), 0.05 sqrt(gL)/(0.25 L)) = (0.947, 0.501, 0.364, 0.270)` (`operators_rb.py:308-310`).
```
a_u = -(e^{g_R} - 1) k_drag u|u| / m_u
a_v = -(e^{g_S0} - 1)(k_lin_sway v + k_sway v|v|)/m_v + F_pd/m_v - κ0 |v| v
a_r = -(e^{g_S1} - 1) f_r r / τ_r + x_pd F_pd / I_r - κ1 |r| r
a_ż = -(e^{g_S2} - 1) D0_ż w_r - κ2 |w_r| w_r
a_θ̇ = -(e^{g_S3} - 1) D0_θ̇ q_r - κ3 |q_r| q_r
F_pd = -e^{g_pd}(v + x_pd r)   (0 when no point damper)
```
Switches: `T3_kap_lat_off` (W7 lateral part on) sets `κ0 = κ1 = 0`; `T3_kap_vert_off` (W7 vertical part on) sets `κ2 = κ3 = 0` (`cat_base.py:795-798`); `T3_surge_low_gate` (W11 on) multiplies `a_u` by `1/(1 + exp(-(u_r - 8.231)/0.5144))` (`cat_base.py:811-813, 128`).

### 5.5 T4 restoring (w.p. `P_T4 = 0.7`; `operators_rb.py:412-431, 812-828`)
Random parameters (each sub-part independently): `c_Kz` w.p. 0.5 `= _smooth(U[ln 0.64, ln 1.56], 0.1)`; `c_Kq` same w.p. 0.5; `c_ks` w.p. 0.5 `= _smooth(U[-0.3, 0.3], 0.1)`; `c_ka` w.p. 0.3 `= _smooth(U[-0.15, 0.15], 0.05)`; offsets w.p. 0.6: `c_dz = (U[-0.2, 0.2] T, N(0, (0.13 T)²), 0)`, `c_dq = (U[-0.3, 0.3] T/L, N(0, (0.2 T/L)²), 0)`; nonlinear w.p. 0.3: `g_z, g_q ~ U[-0.5, 0.5]`, `l_z ~ U[0.5, 2] T`, `l_q ~ U[0.5, 2] T/L`.
```
K_zz = ω_h² e^{g_Kz},  K_qq = ω_p² e^{g_Kq}
k_zq = (g_ks + g_ka) ω_h ω_p ρ_q,  k_qz = (g_ks - g_ka) ω_h ω_p / ρ_q
f_z = exp(g_z tanh(e_z/l_z)),  f_q = exp(g_q tanh(e_θ/l_q))   (1 without the nonlinear part)
a_ż = -[(K_zz f_z - ω_h²) e_z + k_zq e_θ] + K_zz g_dz + k_zq g_dq
a_θ̇ = -[(K_qq f_q - ω_p²) e_θ + k_qz e_z] + K_qq g_dq + k_qz g_dz
```
(`g_dz = c_dz,0 + c_dz,1 ξ`.) Switches (`cat_base.py:777-785`): `T4_speed_const` (W1 on): `c_Kz, c_Kq, c_dz, c_dq` keep only `c0`; `T4_nl_off` (W1 or W5(a) on): `f_z = f_q = 1`; `T4_coupling_off` (W1 on): `c_ks = c_ka = 0`.

### 5.6 T5 jet effectiveness (`operators_rb.py:433-445, 833-843`)
Always evaluated (structural Nomoto part); the extra parameters only w.p. `P_T5 = 0.7`: `c_eT = _smooth(U[ln 0.8, ln 1.2], 0.1)`, `c_eS = _smooth(U[ln 0.7, ln 1.3], 0.1)`, w.p. 0.3 `c3 = U[-1, 1]/rud_max²`, `x_j = x_stern + U[-0.1, 0.1] L`, `z_j ~ U[-1.5 T, 0]`, `k_c ~ U[0, 1]`; otherwise `c_eT = c_eS = 0`, `c3 = 0`, `x_j = x_stern`, `z_j = 0`, `k_c = 0`.
```
λ0 = k_js max(T_a, 0) sin δ_a
F_x = e^{g_eT} T_a (1 + k_c (cos δ_a - 1)),  F_y = e^{g_eS} λ0 (1 + c3 δ_a²)
a_u = (F_x - T_a)/m_u,  a_v = (F_y - λ0)/m_v
a_r = x_j F_y / I_r - (k_nf/τ_r) λ0 - (f_r - 1) r/τ_r
a_θ̇ = z_j (F_x - k_drag u|u|)/I_q,  a_ż = 0
```
With no T5 parameters this is exactly `a_r = (f_r - 1)(k_nf λ0 - r)/τ_r`, i.e. `(f_r - 1)` times the low-fidelity yaw acceleration (derived from `f_r k_nf/τ_r = x_n0/I_r`). Switches (`cat_base.py:786-790`): `T5_etaT_off` (P3 on) `c_eT = 0`; `T5_etaS_off` (P5 on) `c_eS = 0`, `c3 = 0`. In non-sparse draws P3 and P5 are always on (`P_ON = 1`, `cat_base.py:100`), so there T5 keeps only `k_c, x_j, z_j` and the `f_r` rescale.

### 5.7 T2c speed-proportional couplings (`operators_rb.py:450-459, 854-872`)
On w.p. `P_T2C = 0.8`; each of `λ` (w.p. 0.7: `_smooth(U[-1, 1.5], 0.375)`), `M` (w.p. 0.7: `_smooth(U[-0.3, 0.3], 0.09)`), `z_q` (w.p. 0.7: `c0 = ±LogU[0.05, 0.5] sqrt(D0_ż D0_θ̇)` random sign, `_smooth(c0, 0.3|c0|)`) drawn independently; not drawn = 0.
```
v_c = 0.1 max(|u|, 1),  s(v) = v_c tanh(|v|/v_c)/|v|  (1 at v = 0),  λ_s = g_λ(ξ) s(v)
a_u = (m + λ_s A22) v r / m_u   - m v r / m_u  [removed by T2c_rigid_off, cat_base.py:809-810]
a_v = -λ_s A11 u r / m_v
a_r = -λ_s (A22 - A11) u v / I_r
a_ż = g_zq(ξ) (u/u_mid) ρ_q q_r
a_θ̇ = -g_zq(ξ) (u/u_mid) m_w ρ_q w_r / I_q - g_M(ξ) ((u² - u_des²)/u_des²) ω_p² e_θ
```
Net surge in D10: `λ_s A22 v r / m_u`. The lateral part is workless with the low-fidelity `-m u r` for any `λ` (`operators_rb.py:846-853`).

### 5.8 Teven couplings even in the lateral quantities (`operators_rb.py:461-482, 874-901`)
On w.p. `P_TEVEN = 0.5`.
- Longitudinal (each of surge, heave, pitch independently w.p. 0.5): `q = N(0,1)^4 · 1[U < 0.6]` (at least one nonzero), normalised; `W = LogU[0.01, 0.15] A_REF[ch]`; `shape = _smooth(0, 0.2)`.
  `n_v = v/(0.1 u_mid)`, `n_r = r L/(0.1 u_mid)`, `F = φ(n_v², n_r², n_v n_r, (δ_a/rud_max)²)`, `φ(x) = x/(1 + |x|/4)` elementwise; `a_ch += W e^{g_shape(ξ)} F·q`.
- Lateral damping modulation (w.p. 0.5): `on ~ Bernoulli(0.5)^4` (≥ 1 on), `G_k = N(0,1)^{2×2}/‖·‖_2`, `β = Dirichlet(1^4) ⊙ on`, renormalised to sum `U[0.1, 0.5]`.
  `n = (tanh(e_z/(0.5T)), tanh(e_θ/(T/L)), tanh(w_r/(0.05 sqrt(gL))), tanh((u - u_mid)/u_half))`, `X = Σ_k n_k β_k G_k`, `Dh = sqrt(e^{g_S0} k_lin_sway/m_v, e^{g_S1} f_r/τ_r)`, `Mh = sqrt(m_v, I_r)`;
  `Q_t = -(Dh X Dh)(Mh ⊙ (v, r))`, `a_v += Q_t,0/Mh_0`, `a_r += Q_t,1/Mh_1`.

### 5.9 T7 wave loads (`operators_rb.py:484-564, 903-926, 931-961`)
On iff not rb_off and `U ≥ 0.15` (`operators_rb.py:487-488`); number `n_L = min(1 + Poisson(1.2), 5)` (`operators_rb.py:489`).
Input: standardised elevations `z15 = (η - μ_w)/σ_w` (15 stations, absolute elevation incl. env additions; `μ_w, σ_w` = library mean/std of the 15 wave columns, `operators_rb.py:317, 905`).
Per load: kind point 0.45 / gen 0.20 / local 0.35; filter order 1 w.p. 0.4 else 2; `ω_L = LogU[0.7, 11]/rs` ∈ [0.953, 14.97] rad/s; `ζ_L = LogU[0.2, 1.5]`; `A = LogU[0.02, 1] a_env`; `k_h = LogU[0.05, 0.5]` (`operators_rb.py:492-500`).
`a_env = ω_h² sd(mean η_c) = 7.774` m/s², `a_env,p = ω_p² sd(arctan σ) = 0.535` rad/s² over the library (`operators_rb.py:321-332`).
Filter (`operators_rb.py:168-225`): `H(s) = (d0 + d1 s/ω_L)/(1 + s/ω_L)` or `/(1 + 2ζ_L s/ω_L + s²/ω_L²)`; two outputs of one state: `y_f = H u_in` (surge, sway, yaw path) and `y_h = (d1 s/ω_L)/den · u_in` (heave, pitch path; zero gain at zero frequency); numerators scaled so `max_{ω∈[0.3,13]} |H| = 1`; exact zero-order-hold step at `h`, state from zero at episode start.
- pattern kinds (point, gen): `u_in = z15·q - q_μ`, `q` = a random combination of the 5 spatial patterns (mean, longitudinal gradient, transverse difference, longitudinal curvature, twist; the 4 non-mean orthogonalised against the mean on the library; each pattern on w.p. 0.6, N(0,1) coefficients, unit library std) (`ops_m15.py:77-122`, `operators_rb.py:537-540`); `d0, d1 ~ N(0,1)`.
  - point: `pt = (U[x_stern, x_stern + l_w], U[-B/2, B/2], U[-1.5T, 0])`, `d` unit 3-vector; `s_v = min(1, a_env,p I_q / (a_env m_w |x_pt|))`, `s_p' = min(1, a_env,p I_q / (a_env m_v |z k_h d_x - x s_v d_z|))` (`operators_rb.py:545-555, 566-582`).
    `F = m_v A y_f (k_h d_x, k_h d_y, s_v d_z)`, `F^h = m_v A s_p' y_h (k_h d_x, k_h d_y, s_v d_z)`; w.p. 0.3 a quadratic horizontal part `F_{x,y} += m_v A k_h β_q (y_f² - ȳ²) h_q` (`β_q ~ U[-1,1]`, `h_q` unit 2-vector, `ȳ²` library mean square) (`operators_rb.py:557-560, 954-960`).
    `a = (F_x, F_y, x F_y - y F_x, F^h_z, z F^h_x - x F^h_z) / M1` (`operators_rb.py:751-760`).
  - gen: `g` unit 5-vector; `a = A g ⊙ sqrt(m_v/M1) ⊙ (k_h y_f, k_h y_f, k_h y_f, y_h, y_h ρ_q a_env,p/a_env)` (`operators_rb.py:948-953`).
- local: each of the 15 points reads its own elevation with the mean reading regressed out, `u_in,p = (z15_p - μ_p - m β_p)/s_p`, `m = z15·q0 - q0_μ` (`operators_rb.py:333-343, 912-915`); shared poles, numerators `d0_p = a0 + a1 x̂_p`, `d1_p = b0 + b1 x̂_p`, `x̂ = x/(L/2)`; force density `f_p = w_p (f_x, b_y0 + b_y1 x̂_p, b_z0 + b_z1 x̂_p)`, `f_x ~ N(0, 0.3²)` common, `w_p = 1/(1 + exp((x_p - x_stern - l_w)/(0.05L)))`, normalised `Σ_p ‖f_p‖ = 1`; point height `Z = -T/2`; caps `s_v`, `s_p'` from sums of absolute contributions (`operators_rb.py:501-535`).
  `Q = (Σ m_v A k_h f_x y_f,p, Σ m_v A k_h f_y,p y_f,p, Σ(x_p F_y,p - y_p F_x,p), Σ m_v A s_v s_p' f_z,p y_h,p, Σ(Z F^h_x,p - x_p F^h_z,p))`, `a = Q / M1` (`operators_rb.py:937-947`).
Switch `T7_quad_off` (W8 on) removes the quadratic part (`cat_base.py:791-794`).

### 5.10 Switch application (`cat_base.py:771-814`)
`switches = ALWAYS ∪ ⋃_{items on} overlaps(item)`; ALWAYS = `{stab_divergence_only, T8_off, T2c_rigid_off}` (`cat_base.py:112-114, 914-916`). Parameter switches act on the drawn coefficients before `_pack` (`cat_base.py:771-800`); `l1_post` acts on the computed parts (`cat_base.py:803-814`). Layer-1 total: `a_L1 = a_T3 + a_T4 + a_T5 + a_T2c + a_Teven + a_T7` (`cat_base.py:1070`).

### 5.11 T9 old noise (control-step part; `operators_rb.py:1122-1125`, `operators.py:296-337, 622-673`)
`n_k = w_noise · { A ⊙ e^{ga_k} ⊙ sqrt(π_n) ⊙ σ_k (sqrt(u_c) col_k + sqrt(1 - u_c) ε_k) + b_k }`
- `A = A_REF ⊙ LogU[0.03, 1]^5`, times 0 w.p. 0.05 (`null`) (`operators.py:206-207`); `(π_m,c, π_n,c) ~ Dirichlet(10(p_m, p_n) + 10^{-3})`, `(p_m, p_n) ~ Dirichlet(2, 1.5)` (`operators.py:208-211`).
- `col_k`: per channel, Student-t innovations (`ν_t ~ LogU[3, 50]`, correlated across channels by a random correlation matrix, shared or per-channel mixing, scaled to unit variance) through 1-2 sections (resonant w.p. 0.6: `ω ~ LogU[0.3, 12.5]`, `ζ ~ LogU[0.05, 0.7]`; first-order w.p. 0.6 or if none: `τ ~ LogU[0.1, 30]`), normalised to a unit-energy impulse response; `ε_k ~ N(0, I_5)`; `u_c ~ U[0,1]` (`operators.py:297-320, 622-634`).
- `σ_k`: w.p. 0.5 `exp(g_e (x_e - μ_e)/s_e)/Z_e`, `x_e` a low-pass (τ ~ LogU[1, 10]) of `|z_k·p_e|`, `g_e ~ U[-0.7, 0.7]` (`operators.py:322-325, 639-643`); else 1.
- `b_k`: bursts w.p. 0.3: Poisson hits of rate `min(λ0 exp(κ z̃ - κ²/2), 1/(4Δ))` (`λ0 ~ LogU[0.02, 0.5]`, `κ` 0 or U[0,2]), sizes `t_3` clipped ±20, ringing response, direction `dvec`, height `H ~ LogU[0.3, 3]` times `A` (`operators.py:326-337, 648-658`).
- `ga_k`: w.p. 0.3 an OU slow gain (τ ~ LogU[20, 200] s, sd ~ U[0.1, 0.5]), else 0 (`operators.py:341-342, 661-669`).
- `z_k = (s28_ext - μ)/σ`: standardised control-step inputs on the M15 library (`operators_rb.py:1063`, `ops_m15.py:150-158`). The T9 noise is the unpredictable part of the target.

### 5.12 The M15 library `lib.npz` (`studies/_cache/meta7/lib.npz`: `mu (28)`, `sd (28)`, `S (16, 250, 28)` normalised control-step histories)
Used by layer 1 for: the T9 operator's normalisation and all its thresholds (`operators.py:676-745`, `ops_m15.py:185-262`); T7's `μ_w, σ_w`, the 5 pattern projections `Q`, `sd_η = 0.228` m, `sd_α = 0.0388` rad, `a_env`, `a_env,p` and the local-reading regressions (`operators_rb.py:313-343`); `ȳ²` of quadratic loads (`operators_rb.py:584-598`). Layer 3 does NOT use this library (section 6.4).

## 6. Layer 3 (`OperatorGenL3`, `cat_base.py:821-869`; stream `[seed, 71, attempt, 7, 1]`)

### 6.1 Draw order (`cat_base.py:837-869`)
M_t → `ψ_w ~ U[0, 2π]` → `m_H = min(Poisson(1.5), 4)` hidden processes → tier-dependent force sizes → forces. No own acceptance test, no T9 (`lib=None`, so `step` returns zeros, `operators_gen.py:671-673`).

### 6.2 Inertia M_t (`operators_gen.py:283-315`)
`m ~ U[0.9 m_coriolis, 0.97 min(m_u, m_v)] = U[1260, 1561.9]`; `x_g ~ U[-0.05, 0.05] L`; `y_g ~ U[-0.03, 0.03] B`; `z_g ~ U[-0.3, 0.3] T`; `r_yy, r_zz ~ U[0.22, 0.28] L`; `r_xx ~ U[0.35, 0.45] B`; `δ_I ~ U[-5°, 5°]`.
`I_yy = m(r_yy² + x_g² + z_g²)`, `I_zz = m(r_zz² + x_g² + y_g²)`, `I_xz = ½ m (r_zz² - r_xx²) sin 2δ_I + m x_g z_g`;
`I_r = max(I_r0 LogU[0.8, 1.25], 1.05 I_zz)`, `m_w = m(1 + LogU[0.3, 2.5])`, `I_q = I_yy(1 + LogU[0.3, 2.5])`.
In channel order (u, v, r, ż, θ̇):
```
M_t = diag(m_u, m_v, I_r, m_w, I_q) + offdiag:  M_ur = -m y_g,  M_uθ̇ = m z_g,  M_vr = m x_g,  M_żθ̇ = -m x_g   (symmetric)
```
(the rigid part `diag(m, m, I_zz, m, I_yy)` plus the added diagonal `M_A = diag(m_u - m, m_v - m, I_r - I_zz, m_w - m, I_q - I_yy)` make the diagonal equal to the low-fidelity surge/sway masses and the drawn yaw/heave/pitch inertias; `operators_gen.py:303-311`). `I_xz` enters only the Coriolis term. Measured over 40 draws: `I_r ∈ [4424, 6426]`, `m_w ∈ [1743, 4615]`, `I_q ∈ [3037, 9439]`.

### 6.3 Coriolis difference (always on, every tier; `operators_gen.py:833-851`)
`w = ż + u θ`, `q = θ̇`, `c1 = q w - r v`:
```
C = ( m(c1 - (q² + r²) x_g),
      m(r u + r q z_g - r² y_g) - m_coriolis u r,
      I_xz q r + m(x_g r u - y_g c1),
      m(-q² z_g + q r y_g),
      -I_xz r² + m(z_g c1 + x_g q u) )
a_cor = -M_t^{-1} C
```
It reads `u, v` OVER GROUND (`q['sr']`, `cat_base.py:1094, 1101`), not `u_r, v_r`.

### 6.4 Inputs `X ∈ R^{30 + m_H}` (`operators_gen.py:686-723`)
- Columns 0-27, standardised by mean/std of a LOW-FIDELITY-ONLY library (12 rows × 60 s, the data autopilot, training seas `Hs ~ U[0.5,1.5]`, `Tp ~ U[4,7]`, ideal actuators, statistics from 10 s on; `operators_gen.py:984-1018`):
  `(u, v, r, ż, θ̇, θ, T_a, δ_a, ν0 (5), d_i^0 = η0_ij - z - s_p x_i θ for the 15 stations)`. Here `η0` is the sea sample WITHOUT env-item additions (`cat_base.py:1096`), `ν0` the numerical low-fidelity increment (C4), `T_a, δ_a` the low-fidelity applied actuators, `u, v` over ground. Library means/stds: `u 12.37/1.858`, `v 0.046/0.790`, `r -0.0043/0.0871`, `ż 0.0018/0.701`, `θ̇ 0.0009/0.235`, `θ -0.0094/0.0716`, `T_a 2122/890`, `δ_a 0.0134/0.166`, `ν0 = (-0.033/0.316, -0.013/0.363, 0.0015/0.0616, -0.014/2.784, 0.0025/0.916)`, immersions mean ≈ -0.13 to -0.19 m, std 0.13-0.29 m (check 2).
- Columns 28 .. 27+m_H: hidden processes `H_l(t)` (unit stationary std, not standardised): OU (0.4, `τ ~ LogU[0.3, 300]` s), drift (0.2, OU with `τ ~ LogU[60, 3000]` s), jump (0.2: Poisson rate `LogU[1/300, 1/10]` /s, new value `t_df` clipped ±10, `df ~ U[1.5, 5]`), switch (0.2: telegraph ±1, dwell `LogU[2, 120]` s) (`operators_gen.py:112-115, 317-326, 853-868`). Exogenous: they do not depend on the state.
- Last two columns: `cos(ψ - ψ_w)`, `sin(ψ - ψ_w)` (`operators_gen.py:721-722`).
- Input groups for selection: state = cols 0-5, act = 6-7, nu0 = 8-12, imm = 13-27, hid (if `m_H ≥ 1`), dir (`operators_gen.py:845-849` via `cat_base.py:845-849`).

### 6.5 Tiers and sizes (`cat_base.py:142-143, 851-862`)
- small (0.75): `N = min(Poisson(1), 3)`, `κ_i ~ LogU[0.005, 0.1]` i.i.d.
- fail (0.20): `N = min(1 + Poisson(2), 8)`, `κ0 ~ LogU[0.01, 1]`, `κ_i = κ0 sqrt(w_i)`, `w ~ Dirichlet(1_N)`.
- none (0.05): `N = 0`.
`P(N = 0) = 0.05 + 0.75 e^{-1} = 0.326` (training pack: 3243/9860 = 0.329).

### 6.6 One force i: the function class (`operators_gen.py:516-571, 767-831`)
**(a) Sparse projections.** Each group is selected w.p. 0.5 (at least one) (`operators_gen.py:518-521`); `n_p = min(1 + Poisson(2), 6)`; column j of `W_c ∈ R^{n_in × n_p}`: 1-3 picks (uniform), each pick = a uniformly chosen selected group, a uniformly chosen column in it, `+N(0,1)`; unit norm (`operators_gen.py:522-528`). `c(t) = W_cᵀ X(t)`.
**(b) Filter bank** (`operators_gen.py:331-400`): `K = min(1 + Poisson(3), 8)` filters; each has a uniformly chosen source `c_src`, a pure delay w.p. 0.2 (`clip(LogU[0.04, 1] rs, 0.04, 1)` s → `d ∈ {1..18}` substeps), and a kind:
- direct (0.2): `y = c`;
- low (0.25): `1/(τ s + 1)`, `τ = clip(LogU[0.04, 30] rs, 0.04, 30)` ∈ [0.04, 22.0] s;
- lead (0.15): `(τ2/τ1)(τ1 s + 1)/(τ2 s + 1)`, `τ2` ∈ [0.04, 2.20] s, `τ1 = τ2 LogU[1, 10]` (high-frequency gain 1, zero-frequency gain τ2/τ1);
- res (0.30): `(1/P) ω²/(s² + 2ζωs + ω²)`, `ω = min(LogU[0.3, 20]/rs, 20)` ∈ [0.408, 20] rad/s, `ζ ~ LogU[0.05, 2]`, `P = 1/(2ζ sqrt(1 - ζ²))` if `ζ < 0.7071` else 1 (zero-free numerator);
- relay (0.10): `x_r` = first-order low-pass of `c_src` (τ ∈ [0.04, 2.20] s); `y ∈ {-1, +1}`: `+1` if `x_r > θ_hi`, `-1` if `x_r < θ_lo`, else unchanged; `θ_hi = μ + h_u σ`, `θ_lo = θ_hi - w_u σ`, `h_u ~ U[-1.5, 1.5]`, `w_u ~ U[0.1, 1.5]`, `(μ, σ)` the library mean/std of `x_r` (`operators_gen.py:440-443, 791-800`).
Linear outputs are standardised by their own library mean/std (`operators_gen.py:428-432, 787-788`). Exact ZOH discretisation at `h`; at the first substep the filter states are set to the steady state of the first input and the delay line is filled with it (`operators_gen.py:774-786`). The library itself is the low-fidelity-only library above.
**(c) Readout** `o = R(z)`, `z ∈ R^K` the filter outputs (`operators_gen.py:453-501, 746-765`):
```
h1 = σ_a(g (W1ᵀ z - b1))                        W1 ~ N(0, 1/K)^{K×H1}, H1 = round(LogU[8, 64])
[w.p. 0.3: h2 = σ_a(g (W2ᵀ std(h1) - b2)), W2 ~ N(0, 1/H1)^{H1×H2}]
net = hᵀ V,  V ~ N(0, 1/H)^{H×n_out}
[w.p. 0.3: net = sqrt(1-ρ) net/s0 + sqrt(ρ) v_p z_a z_b,  ρ ~ U[0,1], v_p ~ N(0,1)^{n_out}, a, b uniform in 1..K]
net = net / rms
y = sqrt(β) W_lᵀ z + sqrt(1-β) net,  β ~ U[0,1],  W_l ~ N(0,1)^{K×n_out} column-normalised
[w.p. 0.5: y = sqrt(1-γ) y + sqrt(γ) sgn,  γ ~ U[0, 0.5], sgn ∈ {±1}^{n_out}]
o = 3 tanh(y/3)          (each output in (-3, 3))
```
`σ_a` is ONE of {tanh, softplus, relu, abs} per force (uniform); `g ~ LogU[0.3, 3]`; each hidden unit's bias `b_u = W_uᵀ z^{(u)}` with `z^{(u)}` a sample of the force's own library filter outputs (each unit's hyperplane passes through a library point); `std(h1)`, `s0`, `rms` are fixed constants computed on 1024 library samples. All weights are RANDOM and fixed per draw; nothing is fitted.
**(d) Output map** (kind point 0.5 / dist 0.2 / gen 0.3):
- point: `pt = (U[x_stern - 0.05L, x_bow + 0.05L], U[-B/2, B/2], U[-0.742, 0.55])` (`z_box = (-vcg - T/2, D - vcg)`, `vcg = 0.55`, `D = 1.1`; `operators_gen.py:118-119, 232-235`); direction: body (0.6, fixed unit `d`, `F = A o d`, `n_out = 1`), earth (0.2, `F = A o (cos(ψ_d - ψ), sin(ψ_d - ψ), 0)`, `ψ_d ~ U[0, 2π]`), readout (0.2, `F = A o`, `n_out = 3`); `Q = (F_x, F_y, x F_y - y F_x, F_z - θ F_x, z F_x - x F_z)`, `a = M_t^{-1} Q` (`operators_gen.py:532-546, 811-828`).
- dist: `n_b ∈ {2, 3}` Legendre shapes `P_0, P_1, P_2` of `x̂ ∈ [-1, 1]` over `x_box`, fixed `(y_d, z_d)` and unit `d`; `Q_k = mean_x P_k(x̂) J(x, y_d, z_d) d`; `Q = A Σ_k o_k Q_k` with heave `-θ A Σ_k o_k mean(P_k) d_x`; `a = M_t^{-1} Q` (`operators_gen.py:547-561, 829-831`).
- gen: `k ~ U{1..5}` outputs; `G = N(0,1)^{5×k} ⊙ LogU[0.05, 1]^5` (per row), divided by its maximum row L1 norm; `a = κ_i a_ref ⊙ (G o)` directly, not through M_t (`operators_gen.py:562-567, 889-891`).
**(e) Amplitude cap.** For point/dist: `A = κ_i min_c a_ref,c / (Σ_o |(M_t^{-1} Q_o)_c| + (π/4) Σ|F_x per unit| |(M_t^{-1})_{c,3}|)` (`operators_gen.py:509-514, 568`). Hence for every force and every channel c, `|a_c| < 3 κ_i a_ref,c` whenever `|θ| ≤ π/4` (gen: `|a_c| ≤ 3 κ_i a_ref,c × row-L1_c / max row-L1`).
**Total layer 3:** `a_L3(t) = Σ_{i=1}^{N} B_i(θ(t), ψ(t)) R_i(z_i(t))`, `z_i(t) = Filters_i[W_c,iᵀ X(·)](t)`, with `B_i` a fixed 5 × n_out matrix (θ enters only through the `-θ F_x` heave term, ψ only for earth-fixed directions). Bound: `|a_L3,c| < 3 a_ref,c Σ_i κ_i` (small tier ≤ 0.9 a_ref,c; fail tier ≤ 3 κ0 sqrt(N) a_ref,c ≤ 8.5 a_ref,c).

What a single force can and cannot express, from (a)-(e):
- It is a causal operator on at most `n_p ≤ 6` projections, each a combination of at most 3 input columns; total at most `6 × 3 = 18` distinct input columns per force.
- Linear dynamics per projection: sums (via `W_l` and the network) of ≤ 8 filters, each first order, lead, or a zero-free second-order resonance; pure delays ≤ 0.735 s; the longest linear memory is a low-pass with τ ≤ 22 s; relays hold their state indefinitely; hidden processes give exogenous memory up to 3000 s.
- Static nonlinearity: one random network of width ≤ 64 (≤ 2 layers), one activation type, one optional product of two filter outputs, then `3 tanh(·/3)`.
- The family as a set: the readout weights `W1, W2, V, W_l` have Gaussian (full) support and the filter parameters have intervals as support, so the prior puts positive density on every member of the class above with parameters inside those intervals; it puts ZERO density on any law that needs an input outside `X` (section 8), a pole outside the listed intervals, a complex pole pair with a numerator zero inside one filter, a delay > 0.735 s (except through low-passes), more than 8 forces, or a per-channel size above `3 κ a_ref`.

## 7. Probability that each part is on (prior; training-pack frequency in brackets where available)

Draw-level: `sparse` 0.15 [0.140]; tier small/fail/none 0.75/0.20/0.05 [0.759/0.193/0.048].
Item probability in a sparse draw: `n ~ U{1,2,3}` items from the 38 non-structural codes, so a non-dependent item has `E[n]/38 = 0.0526`.

| Part | Probability on | Source |
|---|---|---|
| Layer 1 present | 1 when `lib` is given (training) | `cat_base.py:920` |
| Layer 1 all off (`rb_off`) | 0.05 [0.049] | `operators_rb.py:107, 262` |
| T3 | 0.95 × 0.85 = 0.808 [0.812]; point damper 0.404; quadratic 0.242 (each channel 0.6) before switches | `operators_rb.py:397-410` |
| T4 | 0.665 [0.661]; `c_Kz`, `c_Kq`, `c_ks` 0.333 each; `c_ka` 0.200; offsets 0.399; nonlinear 0.200 (before switches) | `operators_rb.py:412-431` |
| T5 `f_r` yaw rescale | 0.95 (always unless rb_off) | `operators_rb.py:395, 833-842` |
| T5 extra parameters | 0.665 [0.670] | `operators_rb.py:433` |
| T2c (net, rigid part removed) | 0.76; each of `λ`, `M`, `z_q` 0.532 | `operators_rb.py:450-459` |
| Teven | 0.475; each longitudinal channel 0.2375; lateral part 0.2375 | `operators_rb.py:461-482` |
| T7 | 0.808 [0.806]; `n_L = min(1 + Poisson(1.2), 5)`; kinds 0.45/0.20/0.35 [measured 0.52/0.19/0.29 on 232 loads] | `operators_rb.py:487-493` |
| T9 noise | 0.95 (`null` 0.05), only when `noise=True` | `operators.py:206-207` |
| `T4_speed_const`, `T4_coupling_off` (W1) | 0.85 × 0.7 + 0.15 × 0.0526 = 0.603 [W1 0.595] | `cat_base.py:93, 116-118` |
| `T4_nl_off` (W1 or W5(a)) | 0.85 × 0.88 + 0.15 × 0.0907 = 0.762 | `cat_water_a.py:385-386` |
| `T5_etaT_off` (P3), `T5_etaS_off` (P5) | 0.858 each [0.867, 0.868] | `cat_base.py:100, 120` |
| `T7_quad_off` (W8) | 0.688 [0.693] | `cat_base.py:96, 121` |
| `T3_kap_lat_off`, `T3_kap_vert_off` (W7 parts) | 0.601 each | `cat_water_b.py:170-197` |
| `T3_surge_low_gate` (W11) | 0.518 [0.517] | `cat_base.py:96, 123` |
| Coriolis difference, M_t | 1 | `cat_base.py:1101`, `operators_gen.py:283` |
| Layer-3 forces present | 0.674 [0.671] | `cat_base.py:851-862` |
| Layer-3 force kind | point 0.5 (body 0.6 / earth 0.2 / readout 0.2), dist 0.2, gen 0.3 | `operators_gen.py:88-89` |
| Layer-3 hidden processes ≥ 1 | 1 - e^{-1.5} = 0.777 | `operators_gen.py:112`, `cat_base.py:843` |
| W4 gate | 1 (structural) | `cat_base.py:93, 108` |
| Impulse channel nonzero | W3 0.85 × 0.8 + sparse share; E3 0.85 × 0.3 + sparse share [W3 0.709, E3 0.265] | `cat_base.py:93, 105` |

Effective active probabilities after switches, examples: `c_eT` active = 0.665 × (1 - 0.858) = 0.094; T4 nonlinear active ≈ 0.200 × (1 - 0.762) = 0.048 (treating the T4 draw and the items as independent, which they are: separate streams).

## 8. Facts from the code that matter for "is the high-fidelity boat inside the family"
1. **Inputs read.** Layer 1: `u_r, v_r, r, z, ż, θ, θ̇` (through `e_z, e_θ, w_r, q_r`), `T_a, δ_a`, the 15 elevations and their moving-station rates (references, T7). Layer 3: `u, v, r, ż, θ̇, θ`, `T_a, δ_a`, `ν0`, the 15 immersions `η0 - z - s_p x θ`, hidden exogenous processes, `cos/sin(ψ - ψ_w)`. Neither layer reads: `x, y`, wave orbital velocities, roll, the commands (only applied actuators), actuator internal states, keel-referenced immersion, `η̇` (layer 3), water-relative velocity (layer 3), relative wind. Layer-2 items read some of these (orbital velocity, roll via H1, wind via A1/A2, current via E1) but only through their own catalogue forms.
2. **Evaluation timing.** Everything is evaluated from the substep START state and held for 0.04 s (`operators_cat.py:528-535`); T9 is held for 0.24 s (`operators_cat.py:491-499`).
3. **Size bounds.** The total push is clipped at `LIM = (12, 12, 6, 36, 9)` per substep; impulses at `LIM × 0.24 s` per control step; each layer-3 force below `3 κ a_ref`; each saturated item below `3 × nominal`. Layer 1 has no own bound except the clip.
4. **Inertias are not shared** (C3): layer 1 uses `(m_u, m_v, I_r, m_w, I_q)` with its own draws of `ρ_r, a33, ρ_q`; layer 3 and the items use M_t (or B1's matrix).
5. **The structural yaw rescale** `f_r ~ LogU[0.817, 2.31]` is present in 95 % of draws and multiplies the low-fidelity yaw response (section 5.6), with `f_r > 1` in 81 % of draws.
6. **The trained-on distribution is conditioned** on passing a 60-s divergence test in calm water and one standard sea (4.5 % of attempts rejected), and episodes that diverge are truncated (4.8 % of training episodes, `train_meta.pkl` field `div`).
7. **Layer-1 forms are fixed parametric shapes** with few random coefficients (T3: ≤ 3 × 5 speed coefficients + 4 point-damper numbers + 4 quadratic gains; T4: ≤ 6 speed triples + 4 nonlinear numbers; T5: ≤ 2 triples + 4 numbers; T2c: 3 triples; Teven: ≤ 3 longitudinal blocks + lateral 4 matrices; T7: ≤ 5 loads, each a pattern or local reading through one order-≤2 filter, one point/direction/density). Only layer 3 (and T9) are generic.

## 9. UNKNOWN
- The exact shape of the conditioned (accepted) prior: only rejection counts are logged; the accepted distribution was not computed. Reason: needs re-running the acceptance test over many draws (> 2 min).
- Whether the Mission's low-fidelity world plant (`Mission(world='low', ...)`, `episode7.py:137`) uses exactly `R.default_params()` (which the family uses for its constants) or the actuator-modified `act` parameters in other places: the family is built with `p=None` → `default_params()` (`operators_cat.py:180`), the plant with `plant_params=act`. Not checked beyond that.
- Layer-2 item formulas (W1-W12 except W4, H, A, B, P, E): out of scope of this file.
- The library-dependent constants of T9 (thresholds, envelope statistics) per draw: computed at construction; not enumerated.

## 10. Checks run (scripts in this folder; each < 30 s, one process at a time)
1. `chk_l1.py`: 39 items registered; `a_env = 7.774`, `a_env,p = 0.535`, `sd_η = 0.228`, `sd_α = 0.0388`, `D0`, `v_typ`, `A22 = 210.2`; over 120 layer-1 draws: rb_off 0.017, T3 0.875, T4 0.667, T5 0.658, T2c 0.792, Teven 0.483, T7 0.85, null 0.05.
2. `chk_l3.py`: layer-3 library (12 × 1500 substeps) means/stds; example forces of each tier; `I_r0 = 5253.5`; an example M_t; M_t diagonal ranges over 40 draws.
3. `chk_comp.py`: two full draws (accept=False, default keel), 25 control steps: part keys; on the unclipped draw `max |Σ parts - P| = 2.7e-15`.
4. Task-hull geometry: keel `(-0.385, -0.342, -0.300, -0.258, -0.216)` m, intake `(-1.381, 0, -0.361)` m, `h_run = (0.251, 0.196, 0.142, 0.087, 0.033)` m, `F_b = 0.885` m.
5. `chk_meta.py`, `chk_rej.py` on `studies/_cache/meta7/train_meta.pkl` (9860 episodes): tier, sparse, layer-1 and item frequencies, rejection shares, truncated-episode share.
