> **Correction added 2026-10-05 (after this document was written).** Pooled over all 16 recorded missions (`studies/gz_terms.py`, log `studies/_cache/gz_terms.log`), the yaw error is NOT dominated by the stern slender-body lift. Its moment `x_1 Y_lift / M66` has share −1.04 of var(e0_yaw); the distributed Munk moment `−|u| Σ m22(d_i)(v + x_i r) dx / M66` has share +1.68; both have std ≈ 1.1 rad/s² (3.6 × the yaw error std 0.31) and largely cancel. In sway the stern lift is the main term (share 0.89 alone; hull lateral force with cross-flow and the model0 damping 0.75). Read the D.2 rows on the stern Munk lift as applying to sway; for yaw the relevant gz term is the distributed Munk moment, whose family counterpart is T2c's u·v moment (no immersion dependence) or layer 3. The main summary is `learn/meta/GZ_VS_FAMILY_2026-10-05.md`.

# gz plant, reduced model and catalogue family D10: the exact one-step error law and its place in the family (final, corrected)

Repository root `C:/Users/Administrator/Documents/usv-rl-mpc`, branch `usv-priors-mpc-2026-10-01`, read 2026-10-05.
This is `comparison.md` with every verified correction from `verify_gz_side.md` (VG n) and `verify_family_side.md` (VF n)
applied. Where a verifier and `comparison.md` disagreed I read the code again. Section 0.3 lists each correction, my check
and the cases where I disagree with a verifier. Every equation cites `file:line`. Where a docstring and the code disagree,
the code is followed (section 0.2).

Confidence tags used on every classification:
- **[H]**: read in the code and checked numerically, or confirmed independently by two passes.
- **[M]**: read in the code, with a hand calculation.
- **[L]**: inference that has not been checked.

## 0. Checks, code/docstring disagreements, corrections applied

### 0.1 Numerical checks (scripts in this folder; one process each, under 70 s, read-only)

| script | what it does | used in |
|---|---|---|
| `cmp_decomp.py 0_0`, `7_1` | Open-loop replay of two recorded missions (`studies/_cache/onestep_gz/streams_T120.pkl`, rebuilt by `studies.final_eval.make_mission`) through a fresh gz plant. At every 0.02-s step and both Heun stages it evaluates `deriv` in four variants (waves with actual P; waves with P = 1; no waves with P = 1; no waves with T_s = 0), plus the reduced right-hand side and model0. Saves `decomp_0_0.pkl`, `decomp_7_1.pkl` | B, C.4 |
| `cmp_lin.py`, `vgz_check.py`, `vgz_check2.py` | Central-difference Jacobians of the gz velocity rows at the calm running state at 16/20/25/30 kn; added masses; the closed form of X_p on 20 random wavy states | A.5, D |
| `vgz_check3.py` | Captive 30-s run at 30 kn (Heun, 0.005 s) | A.5 |
| `vf_check.py` | W2 critical product; W1 slope; yaw joint constraint | D |
| `final_check.py` (new) | Kick-vs-continuous ratios; means of the combined heave/pitch defect; W2 κ*²; W9 minimum with B1; **free-surge** calm runs at 30 and 16 kn with the thrust state held at T_eq | A.5, B.3, D |
| `final_sigma.py` (new) | Distance, in standard deviations, of three gz values beyond the bounded part of the untruncated Gaussian speed coefficients | D, E(3) |

The identity of B.1 holds on both missions: `max |<G>_Heun − (ν_{k+1} − ν_k)/Δ| = 0` in all five channels (`last_n_sub = 1` at every plant step). The residual `max |e0 − Σ(named terms) − Π|` is 2e-14 on (7,1). On (0,0) it is 3.1e-6, because e0 is computed from the stored states while the terms come from a replay that drifts 1.3e-6 from them (cmp_decomp.py:106, 129) (VG16).

### 0.2 Corrections to the derivation files, and docstring/code disagreements

1. fam_l1_l3 §2.5 says the network is trained on `E = (ν_true − ν_twin)/Δ` (episode7.py:229). In fact training reads `<split>_e0.npz` (learn/meta/model3.py:86, 108-109). That file is made by `data3.e0_split` → `e0_from` against model0 (data3.py:78-107). `E` is stored but never read.
2. The claim "no family term has a first-order transverse-slope side force" (inventory D1 and §4 item 8, fam_items_water §5.3) is wrong. The layer-1 T7 pattern loads draw from five patterns, and two of them are the transverse elevation difference and the twist (ops_m15.py:77-87, `draw_pattern` 112-122, used at operators_rb.py:537).
3. hifi_hull §7 writes `X_p = −Σ f_i τ_i dx` (planing_vessel.py:354-355). For scarab195 this is exactly `X_p = m (g + ẇ)(Θ_rest + θ)`: `τ_i = −Θ` (kp = 0, :263), and nothing writes `tau_o[2]`, so `Σ f_i dx = m ẇ + m g` (:341, 349-354). Checked to 1.8e-12 on 20 random wavy states (vgz_check.py).
4. The momentum, cross-flow and buoyancy terms are at planing_vessel.py:277-281.
5. The docstring of `Propulsion.advance` (sim/actuators.py:186-189) says it is called "exactly once per control step". The code calls it once per 0.02-s plant step (planing_vessel.py:552), so the 5-slot delay is 0.10 s.
6. The comment at reduced.py:429 gives the plant's sway law as `(M+A)v̇ = lift − k_v v|v| − m u r`. That is not the planing plant's law. `k_sway` is `plant.visc[1]` (reduced.py:258), which the planing `deriv` never uses.
7. (new) The docstring of `OperatorRB.stable` describes a damping-ratio and lateral-stability reject rule (operators_rb.py:706-725). In D10, layer 1 is `OperatorRBL1`, whose `stable()` returns True (cat_base.py:760-768). The only conditioning of the D10 prior is the closed-loop divergence test (operators_cat.py:329-376).

### 0.3 Corrections applied in this version (source, my check)

gz side (sections A and B):

| # | correction | code | my check |
|---|---|---|---|
| VG1 | The A.5 "static calm" table is the defect `d = G − L = −L` at a gz equilibrium, not the reduced acceleration. lofi_error §9.1 lists e0. For w and q, `d = e0 / (0.440, 0.763)` | reduced.py:394-424 | agree [H] |
| VG2 | 30 kn is not a gz equilibrium: `G = (−0.014, 0, 0, −0.210, +0.085)` at the cached attitude. The 30-kn defect is (+0.019, −0.298, +0.243). The 30-kn Jacobian and eigenvalues are taken about a point on an oscillation | planing_vessel.py:386-422 | agree. **New**: with free surge and the thrust state held at T_eq the oscillation persists. Over the last 8 s of 25 s: z ∈ [0.1394, 0.1580] m, θ ∈ [−0.0028, 0.0027] rad, u ∈ [15.22, 15.27] m/s. At 16 kn the state stays at G = 0 (final_check.py) [H]. Cause not analysed: UNKNOWN |
| VG3 | Only surge, the heave/pitch decays and the yaw step were identified at u_ss = 17.4 m/s. z0 and θ0 are at u_design = 25 kn. m_v = Mtot[1,1] and k_sw = visc[1] are at the 25-kn attitude. k_lin and k_js come from a 90-s turn started at 25 kn (7000/12000 t_max, 15°) | reduced.py:149-165, 172, 228-241, 263-299; planing_vessel.py:371-383 | agree [H] |
| VG4 | Model0's yaw `ṙ = (k_nf k_js T⁺ sin δ̃ − r)/τ_r` is linear in r and has no `u r` term, so Π_r is exactly linear in the r deviation | reduced.py:482-488 | agree [H] |
| VG5 | Sway constant-defect factor `(1 − e^{−aΔ})/(aΔ) = 0.885`, with `a = k_lin/m_v = 1.0433 /s` | — | agree [H] |
| VG6 | The "(1 − M11/m_u) u̇^gz" piece uses the full gz acceleration (cmp_decomp.py:123). It is a split of the whole surge defect, not a part of U-h (section D.0) | — | agree [H] |
| VG7 | Mean of `H-h + H-w − W^lo` on (0,0) is −1.018 (−0.99 is the mean of R) | — | agree; recomputed [H] |
| VG8 | Actuator decays swapped. Exact first order over 0.04 s gives 0.93551; gz Euler over 2 × 0.02 s gives 0.93444 | actuators.py:194-196 | agree [H] |
| VG9 | Kick vs continuous, for a constant push over Δ: 1.204 (heave), 1.059 (pitch), 1.048 (yaw), not "9 %" | data2.py:106-120; task.py:332-333 | agree (recomputed). Refinement: per-substep heave ratios are 1.10-1.55 for substeps 2-6. For substep 1 the continuous propagation integral is ≈ 0 (Φ_22 changes sign inside it), so the ratio is undefined there [H] |
| VG10 | Source-world model0 feedback: Euler (h·L) only in surge and sway. w, q and r use exact transitions, so `Π^src` integrates L along model0's own sub-paths | reduced.py:418-424, 482-488, 491, 497 | agree [H] |
| VG11 | The jet and prime moment `z_n f_x` reach surge through ẇ in X_p. So `U-p = (P−1)[1 + mΘ z_n (M_hp⁻¹)_{12}] f_x/M11`. U-j and H-j also read z and θ. H-w also contains `0.55 ΔD_f`. Use d_i⁰ in R-h and D_f⁰ in H-h | planing_vessel.py:285-298, 347-355 | agree [H] |
| VG12 | M10 is exact only for first-order channels. Second order (`zeta`, w.p. 0.3 given a lag) uses 4 trapezoid substeps. The asymmetric `ratio` is omitted. M10 thrust travel is [0, t_max] | lofi.py:158-187, 262-270, 316-341 | agree [H] |
| VG13 | The thrust map is piecewise affine (commands below 26.52 N → 0). The rate limit never binds | actuators.py:190-196 | agree [H] |
| VG14 | Five-station sampling resolves each component only in the no-aliasing sense. 30 numbers do not determine 120 components. Earlier tokens (256-token context) also carry the field | task.py:59; step0_preview_spec.py:42-46 | agree [M] |
| VG15 | Jet gain ranges restated with percentiles (B.3, D.2) | waterjet.py:163-172 | agree [H] (VG's numbers) |
| VG16 | B.1 residuals are 2e-14 and 3.1e-6, not 0.0 | — | agree [H] |
| VG17 | Small citation and symbol fixes (zero jet flow at waterjet.py:163-165; reverse branch :187-188; ε and χ undefined; roll also at planing_vessel_gz.py:236-238) | — | applied |

Family side (sections C, D, E):

| # | correction | code | my check |
|---|---|---|---|
| VF1 | The stern Munk lift has an exact-form counterpart: W10, `−κ_b k_lin (U_r/10)((h_bow − d_0)⁺/T)^{q_b} (v_r + x_b r)`, at the wrong place (x_b ≥ 1.35 m, bow stations 3-4). Class EF-OOR | cat_water_b.py:126, 334-359 | agree [H] |
| VF2a | The W1 moment gives `∂q̇/∂z = W l_x/(hbar_run I_q,t)` ∈ [0, 34] /s², with gz's sign | cat_w1.py:74-76, 114 | agree, with three additions: (i) the slope does not depend on speed, while gz varies ×3.3; (ii) W1 on switches T4's coupling off (cat_base.py:115-118, 783-785), so ∂ẇ/∂θ then has no layer-1 source except W2; (iii) the W1 moment adds `∂q̇/∂θ = −mean(x_st) × slope = −0.519 × slope` (the wrong way for gz) [H] |
| VF2b | W2 flutter: gz's sign structure, `|k35 k53| ≤ 2.5 × 158.19 = 395.5` against gz 743-1302. EF-OOR ×1.9-3.3 | cat_water_a.py:117-142, 154-159 | agree; κ*² recomputed [H] |
| VF2c | The W2 gate "0.9994 over the task range" | cat_water_a.py:119-120, 145-147 | **partly disagree**. That value is one corner of the draw (m0 = 1.5°, w_m = 0.2°, m1 = 0). The draws are m0 ∈ U[−3°, 1.5°], w_m ∈ U[0.2°, 0.8°], m1 ∈ U[−1°, 1°], so the gate ranges over (0, 1) by draw, and some draws keep it ≥ 0.997 over 16-30 kn [H] |
| VF3 | `_smooth`'s c1 and c2 are untruncated Gaussians, so g(ξ) has unbounded support. Many EF-OOR labels become "in support, negligible density" | operators_rb.py:148-160, 615-617 | agree, with three corrections: (i) with W1 on, T4's stiffness polynomials are speed-constant (c1 = c2 = 0, cat_base.py:776-780) and its coupling is zero, so there are no T4 tails in that branch; (ii) **disagree** with "a negative pitch stiffness is out of support even with the tails". B1's frequency change (factor f² ∈ [0.5625, 1.21], cat_hidden_boat.py:563-570) and T2c's Munk stiffness (operators_rb.py:870-872) subtract from T4's `e^{g_Kq} > 0`. The total can be negative through the tails when W1 is off; it is hard-bounded below at `0.2025 ω_θ²` when W1 is on, at 25 kn (D.3); (iii) D10 has no layer-1 stability rule (0.2 item 7) [H] |
| VF4 | Coriolis pair: Teven's `n_v n_r = 3.857 v r` feature supplies the missing 0.241 v r. APPROXIMABLE | operators_rb.py:461-471, 874-883 | agree [H] |
| VF5 | The bound "thrust gain ≤ 1.333" is wrong. P3's φ > φ_id below u_id (up to 1.365 at 16 kn); the Bernstein g(c) can exceed c; P14 ≤ 0.1 t_max c. Class UNKNOWN | cat_prop.py:349-377, 871-879 | agree [H] |
| VF6 | Yaw damping and the lofi jet share are not both inside together: f_r ∈ [1.466, 2.31] forces `e^{g_S1}` ≤ 0.40-0.50 | operators_rb.py:383-395, 798, 841-842 | agree, but **disagree** with "only through the layer-3 Coriolis term". Teven's lateral term scales yaw damping by `1 + X_rr`, with `|X_rr| ≤ 0.5 max_k |n_k|` (operators_rb.py:884-901). That reaches the need at 16 kn but not at 20-25 kn (D.2) [M] |
| VF7 | W7 vertical uses width b (three points × b/3). Required `C_Dc = 0.9397 c/b ≤ 0.47` < 0.5; r_exit ≤ 0.3 against gz's full drag at V < 0 | cat_water_b.py:205-213; planing_vessel.py:170, 278 | agree [H] |
| VF8 | Jet pitch moment: T5 has no heave row. The heave projection (via M_t off-diagonal, |m x_g| ≤ 422 kg m against gz A35 = 1460-3450) is EF-OOR | operators_rb.py:843; operators_gen.py:289-307 | agree [H] |
| VF9 | W1 trim drag needs q = 1.339 > 1.3 (inside only with B1). The slope term is `W q arctan σ`. W1 also has a linear ∂u̇/∂z | cat_w1.py:59, 112; cat_hidden_boat.py:548-551 | agree, but my B1 threshold is ε ≥ 0.088-0.163, depending on the base mass and with a_A = 0 (D.1), not 0.111. **New**: W1's slope term has the opposite sign to gz's Froude-Krylov surge term and conflicts with W9 (D.1) [M] |
| VF10 | W7 vertical and the layer-3 Coriolis term add pitch/heave damping. B1 frequency × T4 puts the 30-kn stiffness factors inside | cat_hidden_boat.py:563-570 | agree. W7 vertical has zero slope at calm running (dd = 0), so its damping depends on amplitude [M] |
| VF11 | W9 comparison in accelerations: ≥ 2.79 m/s² per unit slope against gz 1.23-1.89. The slope is the wet mean of `np.gradient`. T7 can cancel the excess | cat_water_b.py:311-321 | agree; with B1 the minimum is 2.68 (final_check.py) [H] |
| VF12 | W4 and the W3/E3 impulses are unsaturated. W1, W2, W5 use the half-clip nominal. Layer-3 forces keep the unloaded Minv | cat_base.py:167-177; cat_water_a.py:289 | agree [H] |
| VF13 | D.5 mixed raw and sparse-adjusted probabilities | cat_base.py:92-108, 145, 944-951 | agree; all values are now effective, `0.85 p + 0.15·(2/38)` [H] |
| VF14 | Target definition verified. XS is `s0` at episode7.py:188/231 and episode8.py:214/259. The target has 10 channels | data3.py:15-24, 78-107 | agree [H] |
| VF15 | T4 `c_ka` sd 0.05; `d1 = d0` static only for first-order filters; P14 reads h_in; axial-loss shortfall approximable by P14 and Teven | operators_rb.py:421, 494 | agree [H] |

Found here (new, not in either verifier):
- N1. The overlap switches change which routes exist: W1 → T4 speed-constant, no T4 non-linearity, no T4 coupling; P3 (P_ON 1.0) → T5's thrust efficiency off; P5 (1.0) → T5's side efficiency and c3 off; always: T2c's rigid `m v r` off (cat_base.py:112-123, 771-814). [H]
- N2. Every family **generalised** yaw moment (items through `to_acc`, the propulsion chain, layer 3) is divided by `M_t[2,2] ≥ 4203 kg m² = 1.433 × M66` (operators_gen.py:300). Layer-1 terms use I_r ∈ [2272, 6428] (operators_rb.py:383-384), which contains M66. [H]
- N3. The gz thrust actuator, sampled at the control instants, **is** an exact first-order lag with τ' = 0.590 s and a 0.100-s delay (section A.3). [M]
- N4. Layer 3's small tier can reach the heave/pitch coupling size with N ≥ 2 forces near the κ cap. The fail tier is not the only route (E(3)). [M]
- N5. Gaussian-tail distances, from final_sigma.py with ρ = 0.8, one speed at a time:

  | quantity | 16 kn | 20 kn | 25 kn |
  |---|---|---|---|
  | T4 coupling `k_s + k_a` | 7.7σ | 35σ | 72σ |
  | pitch damping factor | 3.9σ | 14.5σ | 21σ |
  | yaw S1 factor | 0.9σ | 4.8σ | 6.1σ |

  [H for the numbers, M for their use]

## 1. Symbols (each defined once)

Time. `k` is the control step and `t_k = kΔ`. `Δ = 0.24 s` (env.py:80; 12 × 0.02 s in gz, 6 × 0.04 s in the low world). `dt = 0.02 s` is the gz plant step, with plant step `n = 12k + j`, `j = 0..11`. `h = 0.04 s` is the substep of model0 and of the source plant.

gz state `s ∈ R^14` (planing_vessel.py:251-254): `[X, Y, z, φ, θ, ψ, u, v, w, p, q, r, T_s, δ_s]`.
- `X, Y`: earth position of the centre of gravity (m).
- `z`: heave of the centre of gravity above the rest height `Z_rest = 0.23335 m` (m, up positive).
- `φ`: roll (rad).
- `θ`: pitch from the rest trim `Θ_rest = −0.031994 rad`. With `SP = s_p = −1`, θ > 0 is bow down. Total trim is `Θ = Θ_rest + θ`.
- `ψ`: heading (rad).
- `u, v`: surge and sway velocity in heading axes (m/s).
- `w = ż` (m/s).
- `p = φ̇`, `q = θ̇`, `r = ψ̇` (rad/s).
- `T_s`: thrust state (N).
- `δ_s`: nozzle-angle state (rad).

Hidden gz states: the pump prime `P ∈ [0, 1]` (waterjet.py:208-217) and two 5-slot delay lines (actuators.py:29-42).

Reduced state `s^r = 𝒫 s = s[0,1,6,2,8,4,10,5,11,7] = [X, Y, u, z, w, θ, q, ψ, r, v]` (control/reduced.py:519-526). Velocity channels `ν = (u, v, r, w, q) = s[6,7,11,8,10]` (operators_rb.py:93).

Inputs. `U_k = (U^T_k, U^δ_k)` are command fractions held over step k. `T_c = U^T t_max`, `δ_c = U^δ rud_max`, with `t_max = 6365.66 N` and `rud_max = 0.436332 rad`. The sea is a fixed sum of 120 components, `η(X, Y, t) = Σ_c a_c cos(k_c(X cos θ_c + Y sin θ_c) − ω_c t + φ_c)` (step0_preview_spec.py:83-91; planing_vessel.py:205-231). gz reads `η, η_t, η_tt, η_X, η_Y` at the 48 centreline stations `x_i ∈ [−2.12495, 3.16255] m`, `dx = 0.1125 m` (:257-260). The reduced model and the family read the 5 × 3 stations `x_st = (−2.12495, −0.80308, 0.51880, 1.84067, 3.16255) m`, `y_off = (−1.0126, 0, 1.0126) m`. Their centreline mean is `η̄`, and `σ = Σ x̃_i η_i / Σ x̃_i²` is the least-squares longitudinal slope (reduced.py:371-388; cat_base.py:546-548).

Fixed gz parameters (hifi_hull §2): `m = 1400 kg`; `M11 = 1470`, `M22 = 1610.20 kg`, `M66 = 2932.49 kg m²` (planing_vessel.py:378-382); `Iy = 2551.5 kg m²`; nozzle point `(x_n, z_n) = (−2.1812, −0.40) m`; `x_1 = −2.12495 m`; `ρ = 1025 kg/m³`; `C_lat = 1.0`; `C_dc = cos 20° = 0.93969`; `a_b = 0.5` for `u ≥ 6.69 m/s` (:246-248). The local bottom-slope terms `kp`, `kpp` are 0 (stem rise 0) and the transom taper `ctr = 1` (`transom_correction` False, :237-239). Prime band `b_π = 0.0425 m` (waterjet.py:148-149), `τ_loss = 0.05 s`, `τ_back = 0.5 s` (planing_vessel.py:130; hifi_prop_loop §5).

Identified reduced parameters (fixed; lofi_error §2.1): `m_u = 1968.58 kg`, `k_D = 13.9872 N s²/m²`; `m_v = 1610.20 kg`, `k_lin = 1679.94 N s/m`, `k_sw = 391.866 N s²/m²`, `m_c = 1400 kg`; `τ_r = 0.42 s`, `k_nf = −1.69883e-4 rad/(s N)`, `k_js = 1.61397`; `ω_z = 5.83486 rad/s`, `ζ_z = 0.372324`, `ω_θ = 3.71347 rad/s`, `ζ_θ = 0.155060`; `z0 = 0.153663 m`, `θ0 = −0.00922914 rad`; draft `T = 0.384630 m`; `L = 5.4 m`; `B = 2.0252 m`; `u_design = u_id = 12.8611 m/s`.

Section maps (planing_vessel.py:161-175, 262-264, 313):
- immersion `d_i = η_i − Z_rest − z + 0.55 + x_i Θ`;
- relative vertical velocity `V_i = η_t,i − w + x_i q − uΘ`;
- wetted half-beam `c(d) = min(π d⁺/(2 tan β), y_c)`;
- section area `A(d)`;
- added mass `m_a(d) = (π/2)ρ c²` and its derivative `m'_a(d)`;
- lateral added mass `m22(d) = (π/2)ρ (d⁺)²`.

Model0 `M(s^r, U)`: 6 substeps `f_h` of `ReducedModel.step`, with η ≡ 0 and the commands applied as forces (data3.py:66-75).

Step means. `<g>_k := (1/24) Σ_{j=0}^{11} [g(s_n, t_n) + g(s_n + dt k1_n, t_n + dt)]`, where `k1_n` is the first Heun stage. The actuator states and P are held over a plant step (planing_vessel.py:524, 552-553).

Family random parameters. These are the draw `θ` (CatDrawD10, operators_cat.py:114; CatDraw, cat_base.py:873-932), the M10 actuator draw `α` (lofi.draw_act, lofi.py:198-224) and the sea `𝒮`. Their coded priors are cited row by row in section D. Hidden family states are the item states, the T7 filter states, the layer-3 exogenous processes and the M10 states. Noise is T9's coloured process `n_k`, held over a control step.

---------------------------------------------------------------------------------------------------

## (A) Equations side by side for the five velocity channels

### A.1 Mapping gz → reduced

`𝒫` is a pure coordinate selection. The two models share the frames, `SP = s_p = −1`, the zero of z and the zero of θ. `z0` and `θ0` are gz's calm running attitude at 25 kn (reduced.py:297-299; cmp_lin.py gives 0.1537 m and −0.0092 rad). `𝒫` drops `φ, p, T_s, δ_s`, P and the delay lines.

### A.2 Right-hand sides (gz exact for scarab195: kp = kpp = 0, ctr = 1, SP = −1)

| channel | gz (sim/planing_vessel.py) | reduced (control/reduced.py, step at :366) |
|---|---|---|
| u | `M11 u̇ = f_x − D_f − D_a + X_w + m v r + m(g + ẇ)(Θ_rest + θ)`. Here `D_f = ½ρ u|u| C_F(Re) S_w` with `S_w` and `l_w` from the wet stations (:285-290); `D_a` (:297); `X_w = −ρ g a_b Σ A(d_i) σ_lon,i dx` (:300); Coriolis (:334); X_p closed form (0.2 item 3) | `m_u u̇ = T − k_D u|u|` (:394-395; Euler :491) |
| v | `M22 v̇ = f_y + Y_w − |u| m22(d_1)(v + x_1 r) − Σ_i ½ρ C_lat d_i⁺ v_l,i|v_l,i| dx − m u r`, with `v_l,i = v + x_i r` (:301-303, 313-320, 335, 359) | `m_v v̇ = k_js T⁺ sin δ̃ − k_lin v − k_sw v|v| − m_c u r`, with `δ̃ = clip(δ, ±rud_max)` (:443-453; Euler :497); `c_wave_sway = 0` |
| r | `M66 ṙ = x_n f_y + N_w − |u| x_1 m22(d_1)(v + x_1 r) − |u| Σ m22(d_i)(v + x_i r) dx − Σ x_i ½ρ C_lat d_i⁺ v_l,i|v_l,i| dx` (:303-304, 316-320, 347, 361) | `ṙ = (k_nf k_js T⁺ sin δ̃ − r)/τ_r`, exact first-order transition (:482-488); linear in r |
| w, q | `[m + A33, A35; A35, Iy + A55][ẇ; q̇] = [Σ rest_i dx − m g ; −Σ x_i rest_i dx + 0.55 D_f − 0.5 D_a + z_n f_x]`. Here `rest_i = m_a(d_i)(η_tt,i − 2uq) + 1[V_i > 0] m'_a V_i² + ρ C_dc c(d_i) V_i|V_i| + a_b ρ g A(d_i)`, and `A33 = Σ m_a dx`, `A35 = −Σ x_i m_a dx`, `A55 = Σ x_i² m_a dx` at the instantaneous `d_i` (:277-281, 296-298, 338-353) | two independent linear oscillators, each by its exact 2 × 2 transition (:397-424): `ẇ = −ω_z²(z − z̄) − 2ζ_zω_z w`, `q̇ = −ω_θ²(θ − ᾱ) − 2ζ_θω_θ q`. Model0: `z̄ = z0`, `ᾱ = θ0`. Source plant: `z̄ = η̄ + z0`, `ᾱ = s_p arctan σ + θ0` |

Jet (sim/waterjet.py:153-188), valid for `T_s ≥ 0`; the reverse branch :187-188 is unused because commands are ≥ 0.
- `f_x = P [T_net − G (1 − cos δ_s)]` and `f_y = P G sin δ_s`.
- `T_net = κ ρ A V_j (V_j − 0.95 u⁺)`, `G = κ ρ A V_j²`, `κ = min(T_s/26.52 N, 1)`. `V_j` comes from the power-balance cubic, evaluated at `max(T_s, 26.52 N)` (:163-172).
- f_x and f_y are exactly linear in P and do not read v, r or the waves.

Roll (gz only): `M44 ṗ = −m g GM(φ − w̄) − 2ζω_n Ix p + 0.40 f_y` (planing_vessel.py:322-325; planing_vessel_gz.py:223-239). φ and p enter no other row of the five channels. They also appear at planing_vessel_gz.py:236-238 and planing_vessel.py:492-494 (`y_i φ`, with y_i = 0), which does not change this.

### A.3 Actuators

| | gz (per 0.02 s) | model0 | source plant (family baseline M10) |
|---|---|---|---|
| thrust | 5-slot delay → dead band 26.52 N (below → 0) → clip [−t_max/3, t_max] → `T_s ← T_s + dt clip((c − T_s)/0.6, ±21219 N/s)`, explicit Euler. The force uses the step-start T_s (actuators.py:186-196; planing_vessel.py:552). The rate limit never binds | `T = T_c` | w.p. 0.2 ideal. Otherwise: delay w.p. 0.5, `round(U[0, 0.30] s × rs / 0.04)` whole substeps; lag τ ∈ LogU[0.05, 2]·rs w.p. 0.7, optional asymmetric ratio and second-order ζ; rate limit; dead zone; backlash; gain and offset. Travel [0, t_max]. `rs = √(L/10) = 0.7348` (lofi.py:158-224, 262-270, 316-341). The applied value is the substep average |
| nozzle | 5-slot delay → clip ±25° → `δ_s ← δ_s + dt clip((c − δ_s)/0.1, ±0.698 rad/s)` (waterjet.py:237-240) | `δ = δ_c` | same family; τ_noz ∈ LogU[0.02, 0.5]·rs, slew ∈ [0.34, 2.72] rad/s |

Delay content at t_k: 5 plant steps of `U_{k−1}`, then 7 plant steps of `U_k`.

(new, N3) Closed form at the control instants. Assumptions: command ≥ 26.52 N; the rate limit idle (it is, VG13). Then, with `a = 1 − 0.02/0.6 = 0.96667` (actuators.py:194-196):

`T_s,k+1 = a^{12} T_s,k + a^7 (1 − a^5) T_c,k−1 + (1 − a^7) T_c,k = 0.6658 T_s,k + 0.1230 T_c,k−1 + 0.2113 T_c,k`.

Because `a = e^{−dt/τ'}` with `τ' = −dt/ln a = 0.590 s`, this is exactly a continuous first-order lag with time constant τ' = 0.590 s and delay 5 dt = 0.100 s.
- M10's lag range [0.037, 1.47] s contains 0.590 s.
- Its delay grid {0.08, 0.12} s does not contain 0.100 s. The nearest members put 0.2376 or 0.1840 on `T_c,k`, against gz's 0.2113 [M].

The nozzle is the same Euler form, τ' = 0.0896 s, while the slew limit is idle (a step ≤ 0.07 rad).

### A.4 Integrators

- gz: Heun, dt = 0.02 s. The prime is updated once before the stages (planing_vessel.py:522-553). Sub-stepping occurs when `λ dt > 1.8`; it never happened in the replays.
- Model0: 6 × 0.04 s. Surge and sway use explicit Euler; heave, pitch and yaw use exact transitions; heading uses the trapezoid rule; position uses the new u, v, ψ (reduced.py:490-504).
- Source plant: the same `f_h`, plus waves sampled at the substep start (lofi.py:514-529). The family push is added as a velocity kick after each substep (task.py:332-333; operators_cat.py:565-568).

### A.5 Linearised rows at the gz calm running state (cmp_lin.py, vgz_check.py, vgz_check2.py)

Units are 1/s or 1/s²; the thrust column is per kN. **The 30-kn column is a linearisation about a point on a persistent oscillation, not an equilibrium** (VG2, final_check.py). It is kept for reference only.

| entry | 16 kn | 20 kn | 25 kn | 30 kn (not equilibrium) | reduced |
|---|---|---|---|---|---|
| ∂u̇/∂u | −0.225 | −0.227 | −0.254 | −0.297 | −2k_D u/m_u = −0.117 … −0.219 |
| ∂u̇/∂T_s (per kN) | 0.761 | 0.721 | 0.679 | 0.649 | 0.508 |
| ∂u̇/∂z, ∂u̇/∂θ | 4.14, 11.26 | 5.61, 9.78 | 8.03, 8.07 | 11.06, 7.87 | 0, 0 |
| ∂v̇/∂v | −1.093 | −0.914 | −0.808 | −0.809 | −1.043 (v = 0) |
| ∂ṙ/∂v | 0.365 | 0.224 | 0.020 | −0.187 | 0 |
| ∂ṙ/∂r | −1.734 | −1.465 | −1.399 | −1.597 | −2.381 |
| ∂ṙ/∂δ | −2.143 | −2.434 | −3.144 | −4.192 | −0.974 … −2.132 |
| ∂ẇ/∂z, ∂ẇ/∂w | −29.9, −8.2 | −34.8, −9.5 | −27.1, −10.0 | −21.0, −9.1 | −34.05, −4.35 |
| ∂ẇ/∂θ, ∂ẇ/∂q | −42.0, −7.2 | −60.1, −9.3 | −99.4, −12.9 | −125.9, −19.7 | 0, 0 |
| ∂q̇/∂z, ∂q̇/∂w | 17.7, 4.1 | 20.7, 4.1 | 13.1, 3.0 | 6.3, 1.3 | 0, 0 |
| ∂q̇/∂θ, ∂q̇/∂q | +7.2, −6.2 | +3.1, −8.1 | +2.6, −11.3 | −6.2, −11.3 | −13.79, −1.15 |
| ∂ẇ/∂u, ∂q̇/∂u | 0.67, −0.33 | 0.57, −0.23 | 0.40, −0.11 | 0.27, −0.02 | 0, 0 |

Heave-pitch eigenvalues (z, w, θ, q):
- 16 kn: oscillatory 6.39 rad/s with ζ 0.45; real −6.73 and −1.93.
- 20 kn: 7.07 rad/s with ζ 0.49; real −7.71 and −2.96.
- 25 kn: 5.94 rad/s with ζ 0.67; real −9.84 and −3.54.
- The reduced model has 5.83 rad/s with ζ 0.37 (heave) and 3.71 rad/s with ζ 0.155 (pitch), uncoupled.

So **at 16-25 kn** gz has no lightly damped pitch mode and its heave and pitch are strongly coupled. At 30 kn there is no equilibrium. The reduced identification points are listed in VG3.

Running-state numbers (16 / 25 / 30 kn):
- `|u| m22(d_1)` = 1759 / 1300 / 1302 N s/m, against `k_lin = 1680`.
- `Σ ½ρC_lat d_i dx` = 420 / 392 / 416 N s²/m², against `k_sw = 391.9`.
- A33 = 4221 / 3857 / 3992 kg.
- A55 = 6102 / 6855 / 7753 kg m².
- A35 = 1463-3446 kg m.
- `d_1` = 0.364 m at 16 kn and 0.229 m at 30 kn.

Static calm defect `d = G − L` at the gz calm equilibrium (G = 0 there at 16-25 kn), command = equilibrium thrust state. It equals −L (VG1):

| speed | d_u (m/s²) | d_w (m/s²) | d_q (rad/s²) |
|---|---|---|---|
| 16 kn | −0.277 | −0.838 | −0.578 |
| 20 kn | −0.115 | −0.169 | −0.276 |
| 25 kn | 0.001 | 0 | 0 |
| 30 kn (not equilibrium) | +0.019 | −0.298 | +0.243 |

lofi_error §9.1 lists the propagated e0 instead, e.g. (−0.369, −0.441) in (w, q) at 16 kn: `d = e0/(0.440, 0.763)`. Its 30-kn row assumes steadiness.

---------------------------------------------------------------------------------------------------

## (B) The exact error law

### B.1 Identity

Target (data3.py:78-88), channel c ∈ {u, v, r, w, q}: `e0_{k,c} = [(𝒫 s_{k+1})_c − M_c(𝒫 s_k, U_k)]/Δ`.

Symbols. `G(s, P)` is the gz velocity right-hand side (rows 6, 7, 11, 8, 10 of `deriv`, planing_vessel.py:250-369), with the sea through `t`. `L(s, U) = L_calm(s) + L_act(T_c, δ_c)` is model0's continuous right-hand side:

```
L_calm  = ( −k_D u|u|/m_u, (−k_lin v − k_sw v|v| − m_c u r)/m_v, −r/τ_r, −ω_z²(z − z0) − 2ζ_zω_z w, −ω_θ²(θ − θ0) − 2ζ_θω_θ q )
L_act(T, δ) = ( T/m_u, k_js T⁺ sin δ̃/m_v, k_nf k_js T⁺ sin δ̃/τ_r, 0, 0 )
```

With no gz sub-stepping, `(ν_{k+1} − ν_k)/Δ = <G>_k` exactly, and

```
e0_k = <d>_k + Π_k,   d := G − L(·, U_k) along the gz stage points,
Π_k := <L(·, U_k)>_k − [M(𝒫 s_k, U_k) − 𝒫 s_k]_ν / Δ
```

Residual 2e-14 / 3.1e-6 (0.1). `Π` is model0's feedback on its own deviation, and has the same functional form in both worlds.

### B.2 The propagation term Π

For heave, pitch and yaw, model0 is linear time-invariant at η ≡ 0. For yaw this holds exactly (VG4). A defect `d_c(τ)` on [0, Δ] reaches the velocity channel as `e0_c = (1/Δ)∫Φ_22(Δ − τ) d_c(τ) dτ`, with `Φ(t) = expm([[0,1],[−ω², −2ζω]] t)`. This holds up to gz's position update, which differs from the trapezoid of w by `dt²(k1 − k2)/4`.

| channel | constant defect | defect at the step start | defect at the step end |
|---|---|---|---|
| heave | Φ_12(Δ)/Δ = 0.440 | Φ_22(Δ) = −0.070 | 1 |
| pitch | 0.763 | 0.449 | 1 |
| yaw | τ_r(1 − e^{−Δ/τ_r})/Δ = 0.762 | 0.565 | 1 |
| sway | (1 − e^{−aΔ})/(aΔ) = 0.885, a = 1.0433 /s (v = 0) | | |
| surge | ≈ 0.98 (2k_D u/m_u ≈ 0.16 /s) | | |

- Π is nonlinear in the deviation only in surge (`u|u|`) and sway (`v|v|`, `m_c u r`).
- Measured rms of Π, missions (7,1) / (0,0): heave 5.57 / 6.54 m/s²; sway 0.38 / 0.34 m/s²; yaw 0.12 / 0.11 rad/s²; pitch 0.20 / 0.25 rad/s²; surge 0.02 m/s².
- `corr(e0, Π)` = −0.94 … −0.97 in u, r and w [H].

### B.3 Decomposition of the defect into named terms (each term = gz minus reduced)

Variants, evaluated at the same stage point with the same P:
- `G_calm0`: deriv without waves and with T_s = 0, so f_x = f_y = 0 (waterjet.py:163-165).
- `G_jet` = (no waves, P = 1, actual T_s and δ_s) − G_calm0.
- `G_wave` = (waves, P = 1) − (no waves, P = 1).
- `G_prime` = (waves, actual P) − (waves, P = 1).
- `W^lo = (0, 0, 0, ω_z² η̄, ω_θ² s_p arctan σ)` is the reduced wave response, i.e. the family baseline.

Exactly:

```
d = D_hull + D_jet + D_act + D_wave + D_prime
D_hull = G_calm0 − L_calm;   D_jet = G_jet − L_act(T_s, δ_s);   D_act = L_act(T_s, δ_s) − L_act(T_c, δ_c)
D_wave = G_wave;   D_prime = G_prime
```

What the token holds (net_io §4; residuals.py:52-56, mpc_learned.py:524-530):
- normalised u, v, r, w, q, z, θ; cos ψ, sin ψ;
- T_s and δ_s at t_k; U_k; U_{k−1}; e0_{k−1};
- 15 measured elevations at t_k, and 15 preview elevations at `t_k + 0.12 s` (dead-reckoned pose, forecast error λ = 0.1);
- all earlier tokens of the 256-token context.

It holds no X, Y (except through elevations), no P, φ or p, no 48-station η/η_t/η_tt/η_X/η_Y, and nothing at sub-step resolution.

Magnitudes are `<·>_k` rms (and mean) per control step on two missions: (7,1) leg 1, 45° off the bow; (0,0) leg 0, head seas.

**Cross-cutting surge inertia (VG6).** Write `G_u = F^gz_u/M11` and `L_u = F^lo_u/m_u`, where F are surge generalised forces. Then exactly `d_u = (1 − M11/m_u) G_u + (F^gz_u − F^lo_u)/m_u = 0.2533 G_u + (F^gz_u − F^lo_u)/m_u`. The first piece has rms 0.25 / 0.29 m/s² (cmp_decomp.py:123). Inside D_hull alone the piece is `0.2533 G_calm0,u`.

#### Surge u (e0 rms 0.92 / 1.16, mean −0.38 / −0.59 m/s²)

| ID | term (gz − reduced) | variables | tokens | rms (mean) |
|---|---|---|---|---|
| U-h | `[−D_f⁰ − D_a + m v r + m(g + ẇ⁰)(Θ_rest + θ)]/M11 + k_D u|u|/m_u`, where superscript 0 means calm and no jet. Sub-terms: (ii) Coriolis `m v r/M11 = 0.952 v r` (rms 0.14 / 0.11); (iii) trim drag `m g θ/M11 = 9.34 θ` and the heave-acceleration drag `m ẇ⁰ Θ/M11`; (iv) resistance shape `R(u)` against `k_D u²` | u, v, r, z, θ, w, q | yes | 0.98 / 1.02 (−0.49 / −0.48) |
| U-j | `[1 + mΘ z_n (M_hp⁻¹(d⁰))_{12}] f_x(T_s, δ_s, u, 1)/M11 − T_s/m_u`. T_net(T, u) is non-separable. Axial loss `G(1 − cos δ)`, with G/T_net at the 1-99 % percentiles 1.33-2.41 (7,1) and 1.27-2.62 (0,0) | T_s, δ_s, u, z, θ | yes | 0.57 / 0.66 (+0.54 / +0.62) |
| U-a | `(T_s − T_c)/m_u`. Piecewise affine in (T_s,k, U_{k−1}, U_k) (A.3) | visible | yes | 0.29 / 0.34 |
| U-p | `(P − 1)[1 + mΘ z_n (M_hp⁻¹(d))_{12}] f_x(T_s, δ_s, u, 1)/M11`. Exactly linear in P; the ẇ path is −0.13 % at 25 kn | P (hidden) | P no | 0.52 / 0.86 (−0.24 / −0.50) |
| U-w | `X_w/M11` (rms 0.24 / 0.41), plus the wave change of D_f (wetted surface), plus the wave-driven `m ẇ Θ/M11` | 48-station η, η_X, η_Y, η_t, η_tt; z, θ, u | partly | 0.72 / 0.86 (−0.20 / −0.24) |

`U-h + U-w` rms 1.09 / 1.28. In calm steady running at a true equilibrium, the static parts of U-h and U-j sum to `(k_D u² − T_eq)/m_u`. That is −0.277 at 16 kn and +0.001 at 25 kn, so they cancel only near 25 kn (VG2).

#### Sway v (e0 rms 0.78 / 0.74 m/s²)

| ID | term | variables | tokens | rms |
|---|---|---|---|---|
| V-h | `[−|u| m22(d_1⁰)(v + x_1 r) − Σ ½ρC_lat d_i⁰⁺ v_l|v_l| dx]/M22 + (k_lin v + k_sw v|v|)/m_v`. The Coriolis terms cancel exactly, because `m/M22 = m_c/m_v` | u, v, r, z, θ | yes | 1.80 / 1.53 |
| V-j | `f_y/M22 − k_js T_s sin δ_s/m_v` | T_s, δ_s, u | yes | 0.05 / 0.04 |
| V-a | `k_js(T_s sin δ_s − T_c sin δ_c)/m_v` | visible | yes | 0.13 / 0.14 |
| V-p | `(P − 1) f_y/M22` | P | no | 0.05 / 0.04 |
| V-w | `Y_w/M22` (rms 0.24 / 0.13), plus the wave modulation of `m22(d_1)` and `d_i⁺` | 48-station; v, r, u | partly | 1.40 / 1.08 |

`V-h + V-w` rms 0.88 / 0.83; the two halves are strongly anti-correlated.

#### Yaw r (e0 rms 0.36 / 0.30 rad/s²)

| ID | term | variables | tokens | rms |
|---|---|---|---|---|
| R-h | `[−|u| x_1 m22(d_1⁰)(v + x_1 r) − |u| Σ m22(d_i⁰)(v + x_i r) dx − Σ x_i ½ρC_lat d_i⁰⁺ v_l|v_l| dx]/M66 + r/τ_r` | u, v, r, z, θ | yes | 0.82 / 0.79 |
| R-j | `x_n f_y/M66 − k_nf k_js T_s sin δ_s/τ_r`. Yaw per newton of side force: gz −7.438e-4, reduced −4.045e-4 | T_s, δ_s, u | yes | 0.16 / 0.13 |
| R-a | `k_nf k_js(T_s sin δ_s − T_c sin δ_c)/τ_r` | visible | yes | 0.09 / 0.09 |
| R-p | `x_n (P − 1) f_y/M66` | P | no | 0.05 / 0.05 |
| R-w | `N_w/M66` (0.08 / 0.07), plus the Munk and cross-flow modulation | 48-station | partly | 0.76 / 0.72 |

`R-h + R-w` rms 0.40 / 0.36. G/T_s at the 1-99 % percentiles: 1.54-4.06 (7,1), 1.49-4.99 (0,0); at 5-95 %: 1.58-2.44 and 1.51-2.50. G/T_s reaches 23-44 at T_s = 26.5 N (VG15).

#### Heave w (e0 rms 4.37 / 5.18 m/s²) and pitch q (e0 rms 0.51 / 0.65 rad/s²)

| ID | term | variables | tokens | rms w ; q |
|---|---|---|---|---|
| H-h | `M_hp(d⁰)⁻¹[Σ rest_i⁰ dx − m g ; −Σ x_i rest_i⁰ dx + 0.55 D_f⁰ − 0.5 D_a] − L_calm,hp`. Ingredients: V bottom then chine walls at d = 0.3686 m; emergence d⁺; planing momentum `1[V>0] m'_a V²` with `V = −w + xq − uΘ`; cross-flow `ρC_dc c(d)V|V|`; `−2 m_a u q`; instantaneous A33, A35, A55 | z, θ, w, q, u | yes | 8.72 / 10.75 ; 1.50 / 1.91 |
| H-j | `M_hp(d⁰)⁻¹(0, z_n f_x)`, the pitch moment −0.40 f_x | T_s, δ_s, u, z, θ | yes | 0.07 / 0.08 ; 0.24 / 0.28 (mean −0.18 / −0.21) |
| H-p | `M_hp(d)⁻¹(0, z_n (P − 1) f_x)` | P | no | 0.01 / 0.02 ; 0.09 / 0.15 |
| H-w | Froude-Krylov `a_b ρ g [A(d_i) − A(d_i⁰)]`; `m_a η_tt`; `η_t` inside `V_i`; the change of M_hp with d_i; `0.55 ΔD_f` (wetted surface) | 48-station η, η_t, η_tt; z, θ, w, q, u | partly | 4.58 / 5.63 ; 1.52 / 1.96 |
| (baseline) | W^lo: `ω_z² η̄`, `ω_θ² s_p arctan σ` | 5 elevations | yes | 7.71 / 8.27 ; 0.51 / 0.67 |

Combined, recomputed in final_check.py:
- `H-h + H-w`: rms 9.77 / 11.52 (heave), 0.67 / 0.82 (pitch).
- `H-h + H-w − W^lo`: rms 4.72 / 6.07 (heave), 0.65 / 0.86 (pitch). Means −1.52 / −1.02 (heave) and −0.06 / −0.12 (pitch).

The reduced wave response removes about half the heave hydrodynamic defect and none of the pitch defect.

#### States outside the reduced state

- **Roll** contributes exactly 0 to the five channels while upright (planing_vessel.py:324-325; the intake is on the centreline, :141).
- **Prime P** (U-p, V-p, R-p, H-p): `min P < 0.999` on 0.60 / 0.79 of steps; `< 0.5` on 0.15 / 0.27. On steps with P < 0.5, U-p rms is 1.27 / 1.57 m/s². P follows `P += (clip(h_in/band, 0, 1) − P)(1 − e^{−dt/τ})`, with τ = 0.05 s falling and 0.5 s rising (waterjet.py:208-217). `h_in` is the intake submergence at x = −1.381 m, between stations 0 and 1. The tokens hold neighbouring elevations at t_k and a forecast at t_k + 0.12 s, not h_in.
- **Actuator states**: deterministic functions of `(T_s,k, δ_s,k, U_{k−1}, U_k)`. Thrust is piecewise affine (dead band). The nozzle is piecewise (slew). So U-a, V-a and R-a are functions of visible quantities.

#### Wave-field terms

All of D_wave is error relative to model0. gz integrates η, η_t, η_tt, η_X, η_Y over 48 points at both Heun stages, weighted by the instantaneous section functions. Five-station sampling avoids aliasing: `λ_min = 4.34 m > 2 × 1.32 m`. But 30 numbers per token do not determine 120 components, and fixed-point time derivatives are not given. The field enters through the elevations of the current and earlier tokens, the states and commands of the context, and the error history (VG14).

#### Integration and timing differences

1. Inside gz: none for the velocity channels.
2. Model0's own discretisation is inside Π, which is common to both worlds.
3. `I_Π` (between worlds). gz evaluates `<L>` at 24 Heun stages. The source world evaluates model0's increment per substep: Euler `h L(x_j)` in surge and sway, the exact transition `(Φ(h) − I) x_j` in w, q and r (VG10).
4. `I_ZOH`. Family pushes are evaluated at the substep-start state, held 0.04 s, and added as a kick at the substep end. For a constant push over Δ, kicks against continuous force give 1.204 (heave), 1.059 (pitch), 1.048 (yaw). Per-substep heave ratios are 1.10-1.55 for substeps 2-6 and undefined for substep 1 (VG9). Waves are sampled about 0.02 s late on average, a phase lag of `ω_e × 0.02 s` (≈ 0.06 rad at ω_e ≈ 3 rad/s).

---------------------------------------------------------------------------------------------------

## (C) The catalogue family as a function class, and its training target

### C.1 The source world

One draw θ (CatDrawD10 → CatDraw, cat_base.py:873-932), one M10 draw α (lofi.draw_act) and one sea 𝒮 define the source plant. Per substep j of step k (operators_cat.py:506-568; task.py:323-333):

```
(T_app,j, δ_app,j, pos_j) = M10_α(command history)                     (lofi.py:474-493)
s^r_{j+1} = f_h(s^r_j, T_app,j, δ_app,j, η_𝒮(5×3 stations of s^r_j, t_j))
ν(s^r_{j+1}) += h · clip(a_θ,j + n̄_k, ±LIM) + ι'_j                     (LIM = (12, 12, 6, 36, 9), operators_cat.py:71)
a_θ,j = items(env, hidden) + L1(T2c,T3,T4,T5,Teven,T7) + items(force) + prop chain + W4 gate + L3 + Coriolis   (cat_base.py:1017-1108)
```

Every term reads the substep-start state, the applied actuators, the 5 × 3 sea sample (with its moving-station rates and orbital velocities), ν0, the commands and the hidden states. `n̄_k` is T9's held noise. `ι'` are the W3/E3 impulses: unsaturated, and bounded per control step by LIM × 0.24 s.

### C.2 The parameterised class

`F = {f_θ}`, where `f_θ` maps the substep inputs to the push. Structure:
- **Layer 1** (`OperatorRBL1`, cat_base.py:760-800). T2c, T3, T4, T5, Teven and T7 with the switches N1. It has **no stability rule** (0.2 item 7). Each layer-1 term divides by its own layer-1 masses `(m_u, m_v, I_r, m_w, I_q)` (operators_rb.py:746-749), with I_r = m_v ρ_r² ∈ [2272, 6428] kg m².
- **Items** W1-W12, H1-H3, B1-B3, A1-A3, E1-E4. Accelerations are soft-saturated at 3 × nominal (cat_base.py:161-164, 724-740). Exceptions: W4 is unsaturated; W1, W2 and W5 use the half-clip nominal (caps 6, 6, 3, 18, 4.5; cat_base.py:167-177). Generalised forces Q pass through `M_t⁻¹`, or `M_new⁻¹` when B1 is on.
- **Propulsion chain** P1-P14 (cat_base.py:1110-1159). Its error force acts at (x_j, 0, z_j) through M_t; it is not saturated.
- **Layer 3** (`OperatorGenL3`, cat_base.py:821-869). Tiers: small 0.75 (N ~ Poisson(1) ≤ 3, κ ∈ LogU[0.005, 0.1]); fail 0.20 (N = 1 + Poisson(2) ≤ 8, κ0 ∈ LogU[0.01, 1]); none 0.05. Each force is capped at 3κ a_ref, with `a_ref = (3.23, 2.70, 1.76, 13.1, 1.96)` (operators_gen.py:165-174). Its forces use the unloaded `M_t⁻¹`.
- **Rigid Coriolis difference** with M_t (operators_gen.py:833-851). B1 adds its loaded-minus-base difference.
- **M_t** (operators_gen.py:283-315):
  - `M_t[0,0] = m_u`, `M_t[1,1] = m_v` (= M22 exactly);
  - `M_t[2,2] = I_r,t ≥ 0.8 × 5253.6 = 4203 kg m²` (= 1.433 M66);
  - heave `m_w = m(1 + LogU[0.3, 2.5])`, pitch `I_q,t ∈ [2311, 12968]`;
  - off-diagonals `m z_g`, `−m y_g`, `m x_g`, `−m x_g`, with `m ∈ U[1260, 1561.9]`, `|x_g| ≤ 0.27 m`.
- **T9** coloured noise, not determined by the inputs.
- **Speed coefficients.** Every layer-1 smooth-in-speed coefficient is `g(ξ) = c0 + c1 ξ + c2(1.5ξ² − 0.5)`, with `ξ = clip((u − u_mid)/u_half, ±1)`, `u_mid = 23 kn`, `u_half = 7 kn`. c0 is uniform and hard-bounded. w.p. 0.6, `c1 ~ N(0, (σρ)²)` and `c2 ~ N(0, (σρ²)²)` with ρ ~ U[0.3, 0.8], **untruncated** (operators_rb.py:148-160, 615-617, 784).
- **Conditioning.** The trained-on distribution is the prior conditioned on the closed-loop divergence test; 4.5 % of attempts are rejected (fam_l1_l3 §2.1).

### C.3 The training target (VERIFIED, VF14)

Both worlds use `e0_split → e0_from` with `relabel._env()` (data3.py:78-107; model3.py:86-109; relabel.py:232-240). `XS_k` is `s0 = m.s.copy()` at the step start (episode7.py:188, 231; episode8.py:214, 259). In the source world, `s[12:14]` are the M10 positions at the end of the last substep.

The target has **10 channels** (data3.py:15-24, 85-88):
- the 5 velocities;
- z, θ and ψ (position channels);
- `(act_{k+1} − U_k)/Δ` for thrust and nozzle (actuator channels).

Packing is float32 (< 1e-5 σ_e).

Source identity (exact), with `Π^src` as in B.3 item 3:

```
e0^src_k = Π^src_k + (1/Δ) Σ_j { [f_h(s_j, T_app, δ_app, η_j) − f_h(s_j, T_app, δ_app, 0)]_ν    (W^lo; heave −Φ^z_21(h) η̄_j = 1.239 η̄_j per substep)
                              + [f_h(s_j, T_app, δ_app, 0) − f_h(s_j, T_c, δ_c, 0)]_ν          (A^M10)
                              + h p_j + ι'_j }                                                (family push K_θ)
```

### C.4 Matching condition

gz's law lies in F exactly when some `(θ, α, hidden initial states)` makes the source trajectories equal gz's at the control instants (all 10 channels) along every gz input history. Since 𝒫 and M are common, this is equivalent, up to `I_Π` and `I_ZOH`, to

`K_θ(t) = R(t) := D_hull + D_jet + (D_act − A^M10_α) + (D_wave − W^lo) + D_prime`, together with M10 positions = (T_s, δ_s) at the control instants.

Measured R, assuming `D_act − A^M10 = 0` (final_check.py; units m/s², m/s², rad/s², m/s², rad/s²):

| mission | u | v | r | w | q |
|---|---|---|---|---|---|
| rms (7,1) | 0.92 | 0.87 | 0.42 | 4.73 | 0.66 |
| rms (0,0) | 1.18 | 0.82 | 0.38 | 6.07 | 0.87 |
| mean (7,1) | −0.39 | 0.01 | 0.01 | −1.50 | −0.21 |
| mean (0,0) | −0.60 | −0.02 | −0.00 | −0.99 | −0.27 |

These are defect level, before propagation (B.2).
- Position channels add no new term: e0_z is `(1/Δ)∫Φ_12(Δ − τ) d_w dτ` plus the Π analogue, and similarly for θ and ψ (heading through r). Their classification follows the velocity rows [M].
- Actuator channels are classified in D.6.

---------------------------------------------------------------------------------------------------

## (D) Term-by-term classification against F

Classes:
- **EXACT**: same form, with the required parameters inside the hard bounds.
- **EXACT-TAIL**: same form, but the value is reachable only through the untruncated Gaussian c1, c2 (C.2). It is in the support with negligible density; kσ is given where computed.
- **EF-OOR**: same form, but a required parameter lies outside a hard bound (a uniform/log-uniform range, a `max()`, the delay grid, or a sign constraint).
- **APPROXIMABLE**: no same-form term, or a different form; reachable through other terms, bounded modulation or layer 3.
- **NR**: no term has the inputs.
- **UNKNOWN**: open; the item in section G that settles it is named.

"Layer 3 possible" means three things:
- the inputs are among layer 3's columns: u, v, r, w, q, θ, T_a, δ_a, ν0, the 15 immersions `η0 − z − s_p x θ`, hidden processes and heading relative to ψ_w (operators_gen.py:686-723);
- the size is under the caps: per force 0.97, 0.81, 0.53, 3.9, 0.59 (small tier), up to 8.5 a_ref in total (fail tier);
- layer 3 never reads P, X, Y, fixed-point η_t or η_tt, or the commands.

### D.0 Cross-cutting

| term | family counterpart | class | evidence | conf |
|---|---|---|---|---|
| surge inertia: every surge force must equal `(M_t[0,0]/M11) × gz force = 1.339 ×` | P3 `t_max g(c) min(φ, cap)`, with `φ = φ_id e^{−k1(s−1) − k2 max(s−1,0)²}` > φ_id below u_id: up to 1.365 at 16 kn, 1.52 with B1 (cat_prop.py:349-377). Bernstein g(c) can exceed c (:361-366). P14 ≤ 0.1 t_max c (:871-879). B1 changes M_t[0,0] (cat_hidden_boat.py:471-489). T3 drag factor | UNKNOWN → G7 | The 44-N fit (fam_prop §6.1) targeted gain 1, which is inconsistent with this requirement | [M] |
| yaw inertia of generalised forces (N2) | `M_t[2,2] ≥ 4203 = 1.433 M66` for items, the prop chain and layer 3; layer-1 I_r ∈ [2272, 6428] ∋ M66 | EF-OOR (hard bound) for every generalised yaw moment | operators_gen.py:300 | [H] |

### D.1 Surge

| term | family counterpart (prior) | class | evidence / numbers | conf |
|---|---|---|---|---|
| U-h(ii) Coriolis `0.952 v r` | Layer-3 Coriolis `(m/m_u) v r` (operators_gen.py:844-851). m = 1400 makes sway exact and gives 0.711. Teven surge feature `n_v n_r = v r L/(0.1 u_mid)² = 3.857 v r` through `x/(1 + |x|/4)` with weight `W ∈ [0.04, 0.6]` m/s² (operators_rb.py:461-471, 874-883): `W q_3 = 0.0625` supplies 0.241 v r | APPROXIMABLE (exact to first order) | The φ saturation costs 6.5 % of a 0.02 m/s² term at v = 0.8, r = 0.09. Without Teven: λ = 474.1/(210.2 − A11) ∈ [2.60, 6.75] > 1.5, i.e. EF-OOR (c0 hard), tails only. Teven on for channel 0 w.p. 0.95·0.5·0.5 | [H] |
| U-h(iii) trim drag `m(g + ẇ)(Θ_rest + θ)/M11` | W1 `X = −W(Φ − 1) tan τ_e` (cat_w1.py:112), with `τ_e = τ0 + s_p(θ − θ0) − arctan σ` floored (cat_base.py:562-564). Linearised: `X ≈ W q (θ − θ0) + W q arctan σ − W q (τ0 − τ_eq)`, so the θ coefficient is `W q/m_u = 6.98 q` against `m g/M11 = 9.34` (the A.5 total ∂u̇/∂θ is 7.87-11.26) | EF-OOR without B1 (q = 1.339 > 1.3). EXACT for the θ coefficient at one speed with B1 at ε ≥ 0.088 (base mass 1260) to 0.163 (1561.9), a_A = 0 (cat_hidden_boat.py:471-489, 544-551). APPROXIMABLE overall: there is no `ẇ Θ` product, and the constant coefficient cannot follow gz's speed variation | Hand calculation from b1_inertia; P(W1) = 0.603, P(B1) = 0.688 | [M] |
| ∂u̇/∂z (gz 4.1-8.0 at 16-25 kn) | W1 `−W Pe τ_eq ∂eps/∂z = W Pe τ_eq/(hbar_run m_u)`, with Pe ∈ [0.3, 3] (cat_w1.py:100-102) and hbar_run = 0.1418 m (cat_base.py:388); range ≈ 0.5-26 | EXACT linear form, inside (one speed) | VF9 | [M] |
| U-h(iv) calm curve `R(u)` vs `k_D u²` (`T_eq/k_D u²` = 1.575, 1.153, 0.999 at 16/20/25 kn) | T3 `e^{g_R(ξ)}`, c0 ∈ [ln 0.8, ln 1.25] (operators_rb.py:399-400, 795); W11 hump tail; B3; W6 | At gain 1: APPROXIMABLE, with c1 ≈ −0.24 (≈ 2σ) and c2 ≈ 0.11. At gain 1.339 the drag factor must be 2.64 / 1.74 / 1.34: EXACT-TAIL for T3 (c0 hard ≤ 0.223), APPROXIMABLE with B3 and W11 | Static d −0.28 … 0 m/s² (A.5) | [M] |
| U-j non-separable `T_net(T, u)` | P3 separable `t_max g(c) φ(u)` | APPROXIMABLE: 44 N rms (gain 1). Gain-1.339 target UNKNOWN → G7 | fam_prop §6.1 | [M] |
| U-j axial loss `G(1 − cos δ)` | `k_c T_eff(1 − cos δ)` with `k_c ∈ U[0, 1]` (operators_rb.py:116, 445; cat_base.py:1150-1152) | EF-OOR for k_c (needs G/T_net = 1.27-2.62). APPROXIMABLE through P14 (reads c and δ) and the Teven `(δ/rud_max)²` feature | Shortfall ≤ 136 N, ≤ 0.07 m/s² | [H] |
| U-a actuator | M10 baseline. gz = first-order lag, τ' = 0.590 s, delay 0.100 s (A.3) | EF-OOR (delay grid 0.08/0.12 s, hard). Not part of F | lofi.py:198-224 | [H] |
| U-p prime | P1 logistic suction + latch + asymmetric lag (cat_prop.py:169-240); P2 forced with P1; P13; P14 reads h_in/h_run (≤ 637 c N) | APPROXIMABLE (form differs from gz's ramp target + exact asymmetric lag). Best in-prior `max|β − π*| ≥ 0.126`. Layer 3 cannot help | fam_prop §6.4 | [M] |
| U-w Froude-Krylov surge (gz 1.89 / 1.23 / 1.25 m/s² per unit slope at 16 / 25 / 30 kn) | W9 `−μ_x m g s̄`, μ_x ∈ U[0.4, 1.3], s̄ = wet mean of `np.gradient` (cat_water_b.py:303-321): ≥ 2.79 m/s² per unit slope (2.68 with B1). W1's slope term `+W q arctan σ` has the **opposite sign**. T7 gradient-pattern loads | W9 alone: EF-OOR (×1.42-2.27). W1 + W9: the net coefficient `m g (μ_x − q)/m_u` must be 1.23-1.89, i.e. `μ_x − q = 0.176-0.271`; with the trim-drag value q = 1.339 this needs μ_x ≥ 1.52 > 1.3, a joint EF-OOR. With T7 (fixed linear filtered pattern, no `A(d_i)` weighting): APPROXIMABLE | final_check.py; FK rms 0.24 / 0.41 | [M] |
| U-w mean (added resistance −0.20 / −0.24) | W8 (cat_water_b.py:262-287) | APPROXIMABLE | | [M] |

### D.2 Sway and yaw

| term | family counterpart | class | evidence / numbers | conf |
|---|---|---|---|---|
| Stern Munk lift `−|u|(π/2)ρ d_1²(v + x_1 r)` | **W10**: `c = κ_b k_lin (U_r/10)((h_bow − d_0)⁺/T)^{q_b}` on `v_r + x_b r`, yaw moment `x_b Y` (cat_water_b.py:126, 334-359). q_b = 2 is the edge of U[1, 2]; d_0 = 0 is the edge of U[0, 0.5T]; κ_b = 1.418 ∈ LogU[0.1, 2]. **But** x_b ∈ U[0.25, 0.5]L = [1.35, 2.70] m and h_bow is the mean of stations 3-4. T3 point damper `−e^{g_pd}(v + x_pd r)`, x_pd ∈ U[x_stern, x_bow] (x_1 at the edge), coefficient 103-2051 N s/m with speed polynomial and no immersion dependence (operators_rb.py:403-407, 801-805). Teven lateral (≤ 0.5). Layer 3 | W10: EF-OOR (location, station). Linear speed part: possible through the T3 point damper (G3). Immersion² modulation `(d_1/d_1,run)² − 1 ∈ [−1, ≈3]`: beyond Teven; layer 3 only, so APPROXIMABLE | The lofi `k_lin v` at x = 0 can be reduced only to 0.6 k_lin (bounded; tails beyond) (VF1). P(W10) = 0.348 | [H] |
| Linear lateral block (∂v̇/∂v, ∂v̇/∂r, ∂ṙ/∂v, ∂ṙ/∂r at 16-25 kn, A.5) | lofi + T3 (S0, S1, pd) + T5 (f_r, x_j/I_r) + T2c λ (u·v Munk, `−λ(A22 − A11)uv/I_r`: gz needs λ ≈ −0.55 at 16 kn for I_r = 2272 and A11 = 28; ≈ 0 at 25 kn) + Teven lateral + layer-3 Coriolis + W10 | UNKNOWN → G3 (joint feasibility) | | — |
| V crossflow `−Σ ½ρC_lat d_i⁺ v_l|v_l| dx` | W7 lateral `−½ρ C_Dl max(hc_i, 0) Dx_i |w_i| w_i`, C_Dl ∈ U[0.3, 1.5] ∋ 1, c_orb ∈ U[0, 1] ∋ 0, hc_i = gz d_i at the same x (cat_water_b.py:187-188, 214-222) | Sway part: EXACT form (5 trapezoid stations vs 48), but the lofi `−k_sw v|v|` stays ≥ 0.6 k_sw (T3, bounded; EXACT-TAIL to remove it). Yaw part: through M_t[2,2] ≥ 1.433 M66, EF-OOR (N2) | Quadratic coefficients 392-420 (gz) vs 392 (lofi) | [H] |
| Yaw damping alone | `−e^{g_S1} f_r r/τ_r` (lofi + T3 + T5; operators_rb.py:395, 798, 841-842): range [0.49, 5.55] × 2.381 contains gz's 0.588-0.728 × 2.381 | EXACT (alone) | | [H] |
| Yaw damping **jointly** with the lofi jet share | `x_j/I_r = −7.438e-4` with x_j ∈ [−2.665, −1.585] m forces I_r ∈ [2272, 3583] and f_r = 5253.6/I_r ∈ [1.466, 2.31]. Then `e^{g_S1}(1 + X_rr) ≤ 0.497 / 0.420 / 0.401` at 16 / 20 / 25 kn. Teven `1 + X_rr ≥ 1 − 0.5 max_k|n_k|` = 0.619 / 0.798 / 0.861 at the gz running states (operators_rb.py:884-901). Layer-3 Coriolis `−m x_g u r/I_r,t` | Bounded layer 1: inside at 16 kn only (0.6 × 0.619 = 0.371 ≤ 0.497); EF-OOR at 20 and 25 kn (0.479 > 0.420, 0.517 > 0.401), and one Teven coefficient cannot do both. EXACT-TAIL: 0.9σ / 4.8σ / 6.1σ. Inside the prior with layer-3 x_g ≈ −0.16 m (at 25 kn), at the cost of lower pitch damping (`+m|x_g|u q/I_q,t`). H1's roll path (u r → φ → yaw) is UNKNOWN | final_sigma.py; vf_check.py | [M] |
| R Munk moment, sway→yaw u·v (gz `∂ṙ/∂v/u` = +0.044 / +0.022 / +0.002 per (m/s)² at 16/20/25 kn) | T2c `−λ_s(A22 − A11) u v/I_r`, λ = g_λ(ξ) with c0 ∈ U[−1, 1.5] (operators_rb.py:453-456, 865) | EXACT form, inside at one speed. Since Teven carries the surge `v r` and layer-3 m can absorb the sway `−λ A11 u r` (m = 1400 − λ A11 ∈ [1260, 1562]), λ is not forced by D.1. Speed variation needs c1 (Gaussian). Joint fit with the stern lift → G3 | | [M] |
| V/R jet law (G/T_s 1.5-4.1 at 1-99 %, against k_js = 1.614; sin δ) | P5 `[k_js T ψ(u) + k_i t_max e^{−c/c_i}] h(δ)` with `h = sin(e_d d_s tanh(δ/d_s))(1 + a sgn δ)` (cat_prop.py:445-475) | APPROXIMABLE: 23 N rms with bounds active | fam_prop §6.3 | [M] |
| R jet yaw per newton (−7.438e-4) | Lofi share `x_j/I_r` (T5, layer 1). Chain error share `x_j/M_t[2,2]`, ≥ −2.665/4203 = −6.34e-4 | Lofi share: EXACT (inside). Error share: EF-OOR (N2) | Shortfall ≈ 0.02 rad/s² rms (ΔF_y rms 71 N) | [H] |
| V/R actuator | M10 baseline. Nozzle gz = Euler lag τ' = 0.0896 s, slew 0.698 rad/s (inside [0.34, 2.72]), delay 0.10 s | EF-OOR (delay grid) | | [H] |
| V/R prime | P1 (`p^{κ_p}`, κ_p ∈ [0.8, 1.5] ∋ 1) | APPROXIMABLE | 0.05 | [M] |
| V/R first-order FK `−ρ g a_b Σ A(d_i) σ_tr,i dx` and moment | T7 point/gen/local loads on the transverse and twist patterns. Filter `H = (d0 + d1 s/ω)/den`; static when d1 = d0 and the filter is first-order (w.p. 0.4). T7 loads use the layer-1 masses (operators_rb.py:931-961; ops_m15.py:77-122). H1 roll path | APPROXIMABLE. Fixed coefficient (no `A(d_i)` weighting); finite difference over B (attenuation down to 0.68 at k = 1.45 /m); 5 vs 48 stations | Y_w 0.24 / 0.13; N_w 0.08 / 0.07 | [M] |
| V/R wave × lateral-velocity modulation | W7 lateral (immersion × quadratic; yaw via M_t); Teven (bounded); layer 3 | APPROXIMABLE | Part of V-w 1.40 / 1.08 | [M] |

### D.3 Heave and pitch

Branch note. With W1 on (effective P = 0.603), T4's coupling is zero and its stiffness and offsets are speed-constant (N1).

| term | family counterpart | class | evidence / numbers | conf |
|---|---|---|---|---|
| ∂ẇ/∂θ (gz −42 / −60 / −99 at 16/20/25 kn) | W1 off: T4 `k_zq = (k_s + k_a) ω_z ω_θ ρ_q`, bounded \|k_zq\| ≤ 14.2 (operators_rb.py:412-431, 818, 825). W2 flutter `a_z = −f k35 e_θ` with `k35 k53 = −ρ_f κ*²`, ρ_f ≤ 2.5, κ*² = 158.19 (cat_water_a.py:117-142, 154-159). W1 on: only W2 and layer 3 | EF-OOR in layer 1 and the catalogue (T4 ×3-9; W2 product ×1.9-3.3, with l_f = 1.5-2.8 m inside). T4 EXACT-TAIL: 7.7σ / 35σ / 72σ. APPROXIMABLE through layer 3 | final_check.py; final_sigma.py | [H] |
| ∂q̇/∂z (gz 17.7 / 20.7 / 13.1) | W1 on: moment `Q_θ = s_p W Φ l_x eps/(1 + |eps|)` gives slope `W l_x/(hbar_run I_q,t)` ∈ [0, 34] (cat_w1.py:114), not speed-dependent, and adds `∂q̇/∂θ = −0.519 ×` slope. W1 off: T4 `k_qz` ≤ 10.0; W2 k53 | W1: EXACT linear form, inside at one speed. Across 16-25 kn: APPROXIMABLE (gz varies ×1.6 there, ×3.3 to 30 kn). T4: EF-OOR ×1.3-2.1 (tails) | VF2a | [H] |
| ∂ẇ/∂q, ∂q̇/∂w (gz −7.2…−12.9 and 4.1…3.0) | T2c antisymmetric `c_zq(u/u_mid)ρ_q`, \|c0\| ≤ 1.118 (operators_rb.py:457-459, 869-870): \|∂ẇ/∂q\| ≤ 2.1. The sign structure matches gz (product < 0). The ratio `|∂ẇ/∂q|/|∂q̇/∂w| = ρ_q²` needs 1.32-2.07 m against ρ_q ∈ [0.97, 1.46]. W7 vertical (zero slope at calm running). M_t off-diagonal | EF-OOR ×6.3-7.3 at 16-25 kn (T2c maximum 1.13 / 1.42 / 1.77; c0 hard; tails). APPROXIMABLE through layer 3 | | [M] |
| Pitch damping (gz −6.2 / −8.1 / −11.3 vs −1.15) | T3 `e^{g_S3} ∈ [0.6, 2.4]` (bounded) gives ≤ 2.76 (operators_rb.py:401-402, 800). Layer-3 Coriolis `−m x_g u q/I_q,t`, \|·\| ≤ 2.8 /s at 30 kn with I_q,t = 2311, sign from x_g. W7 vertical and T3 κ (quadratic, amplitude-dependent). B1 changes it by ×[0.75, 1.1] | Linear at calm: EF-OOR ×2.2-4.1 (bounded); EXACT-TAIL 3.9σ / 14.5σ / 21σ. With layer-3 x_g > 0 partly closes, which conflicts with the yaw requirement x_g < 0. APPROXIMABLE (amplitude-dependent W7) | final_sigma.py | [M] |
| Heave damping (gz −8.2…−10.0 vs −4.35) | T3 `e^{g_S2}` ≤ 2.4; W1 `s^{p_B}` | EXACT, inside (factor 1.89-2.30) | | [H] |
| Heave stiffness (factors 0.878 / 1.022 / 0.796 at 16/20/25) | T4 `e^{g_Kz}` ∈ [0.64, 1.56]; B1 `(1 − e_m ε)²` (total ≥ 0.2025); W1 `s^{p_K}` | EXACT, inside | 30 kn is a non-equilibrium point (VG2) | [H] |
| Pitch stiffness ∂q̇/∂θ (gz +7.2 / +3.1 / +2.6 → factor −0.52 / −0.22 / −0.19) | Total factor `e^{g_Kq} + (f_B1² − 1) − c_M(u² − u_des²)/u_des²`, with `f_B1 = 1 − e_m ε ∈ [0.75, 1.1]` (cat_hidden_boat.py:563-570) and `c_M` from T2c (operators_rb.py:453-456, 870-872; exactly 0 at u_des = 25 kn). The W1 moment adds stiffness | W1 on (T4 speed-constant): the hard minimum is 0.2025 at 25 kn, so a negative value is **outside the support** (EF-OOR). At 16/20 kn the bounded minimum is 0.025 / 0.094; negative only through T2c tails. W1 off: negative through T4 tails (+ B1): EXACT-TAIL, negligible density. W5(a) and H1 not checked: UNKNOWN. Layer 3 (θ column): APPROXIMABLE | VF3 corrected here | [M] |
| Static running-attitude shift (gz `θ_run − θ0` = −0.042 / −0.020 / 0; `z_run − z0` = −0.025 / −0.005 / 0) | W1 trim shift `w_p² s_p τ0 (s^{−p_τ} − 1)` (cat_w1.py:117-118): p_τ ≈ 1.4 ∈ [0.5, 1.8] and τ0 ≈ 0.048 ∈ [2°, 6°] give −0.042 / −0.018 / 0. T4 offsets (speed-constant when W1 on); B1 dz0, dth0 | APPROXIMABLE, close in-prior fit | Static pitch defect −0.58 rad/s² at 16 kn | [M] |
| Chine kink in `A(d)` | W5(a) station stiffness kinks (cat_water_a.py:397-433) | APPROXIMABLE | | [L] |
| Emergence d⁺ | W4 gate χ_i (structural, unsaturated; cat_water_a.py:275-300) | APPROXIMABLE (not a per-station force) | | [M] |
| Momentum term, wave-entry part | W3 slam impulse: C_m = π/8 ∈ [0.2, 0.47], p_h = 2, d_c = 0.2346 m at k_w = π/2 (endpoint), β = 20° both ends (cat_water_a.py:200-220) | EXACT form at the support edge. Impulse budget, m_eff coupling and the I_ZOH factor (1.2 in heave) differ, so APPROXIMABLE in practice | | [M] |
| Momentum term, steady planing part (V contains −uΘ) | None: the family's `dd = η̇ − (ż + s_p x θ̇)` has no uτ term (cat_base.py:539-543) | APPROXIMABLE only through T3/W1 speed-scheduled damping and layer 3 | | [H] |
| Vertical cross-flow `ρ C_dc c(d) V|V|` | W7 vertical `ρ(b/3) dx C_Dc χ |dd| dd` summed over 3 points, width b = 2.0252 (cat_water_b.py:205-213; cat_base.py:346) | APPROXIMABLE (form differs: constant width b vs 2c(d); dd vs V). The nearest-form parameters are out of range: C_Dc needs ≤ 0.47 < 0.5, r_exit needs 1 > 0.3 | VF7 | [H] |
| Jet pitch moment `z_n f_x` | T5 `z_j(F_x − k_D u|u|)/I_q` with z_j ∈ U[−0.577, 0] ∋ −0.40 (operators_rb.py:442, 843). No heave row. Chain lever `z_j ΔF_x` through M_t (cat_base.py:1157) | Pitch row: EXACT. Heave projection (H-j w 0.07-0.08): EF-OOR (\|m x_g\| ≤ 422 vs A35 1460-3450). The drag moment sits at z_j instead of −0.55/+0.5 m (static, absorbed by offsets). T5 on w.p. 0.665 | VF8 | [H] |
| Wave forcing `H-w − W^lo` | Baseline W^lo; T4 via e_z; T7 heave path (zero DC gain), local per station; W1, W5, W7, W8; layer 3 | APPROXIMABLE. Fixed-point η_t and η_tt are not inputs; per-station weighting only through T7-local and layer 3 | rms 4.72 / 6.07 (w), 0.65 / 0.86 (q) | [M] |
| Prime in pitch | P1 through z_j | APPROXIMABLE | 0.09-0.15 | [M] |

### D.4 Terms with no reduced counterpart that are zero in gz

Roll in the five channels is 0 in gz. The family's H1 (effective P 0.603) puts roll into all five channels (cat_hidden_boat.py:244-275), and all its output coefficients include 0, so H1 can be made neutral [M]. Its lateral path competes with T7's transverse load for the same input (G13).

### D.5 Family terms with no counterpart in gz

All probabilities are effective: `0.85 p + 0.15·(2/38) = 0.85 p + 0.0079` (cat_base.py:92-108, 145, 944-951), with dependents approximate.

| term | effective P(on) | can it be neutral? | size on gz trajectories | conf |
|---|---|---|---|---|
| Layer-3 surge Coriolis `−(m/m_u) q(ż + uθ)` | 1 | no (m ≥ 1260) | 0.17-0.21 m/s² rms | [H] |
| Lever heave `s_p θ ΔF_x` (cat_base.py:310) | 1 when the chain is on | only with ΔF_x = 0 | ≤ 5 % of ΔF_x/M | [M] |
| T9 coloured noise | 0.95 | null w.p. 0.05 | unpredictable part of every target | [M] |
| P9 pulsation (s0 ≥ 0.003) | 0.858 (off in sparse episodes unless chosen) | no when on | ≥ 0.3 % of T_eff | [M] |
| P2 engine speed (forced with P1) | ≈ P(P1) ≈ 0.60-0.62 | no when on | overshoot ≥ 1.04 after latch clearing | [M] |
| P10, P11, P12 | 0.072, 0.033, 0.197 | no after onset | | [M] |
| P4, P6, P7, P8, P13, P14 | 0.263, 0.688, 0.525, 0.025, ≈ 0.19, 0.433 | yes (P7 needs rate0 > 0.698 rad/s; P14 ε ≥ 0.005) | | [M] |
| H2, H3, B2 | 0.348, 0.135, 0.098 | small amplitudes inside | | [M] |
| A1-A3 | 0.603, ≈ 0.60, 0.518 | U10 → 0 possible; A3 λ_L ≥ 0.003 W | | [M] |
| E1-E4 | 0.518, 0.093, 0.263, 0.263 | E1 wind mode; E2 zeros possible; E3, E4 rare | | [M] |
| W2, W10, W11, W12 | 0.263, 0.348, 0.518, 0.263 | yes | | [M] |
| W9 | 0.603 | no alone (μ_x ≥ 0.4); a W1 slope term or a T7 load can cancel the excess | | [M] |
| Layer-3 hidden processes | 0.777 | yes (weight 0) | | [M] |

### D.6 Actuator channels (e0 channels 8-9)

| channel | gz | family | class | conf |
|---|---|---|---|---|
| thrust `(T_s,k+1/t_max − U^T_k)/Δ` | First-order lag, τ' = 0.590 s, delay 0.100 s, dead band 26.52 N (A.3) | M10: τ inside, delay ∈ {0, 0.04, …} s (rounded `U[0, 0.3]·rs`) | EF-OOR (delay grid). The best member's weight on T_c,k is off by ±0.026 of `|T_c,k − T_c,k−1|`, i.e. 0.11 `|ΔU^T|` /s in e0 units | [M] |
| nozzle | Euler lag τ' = 0.0896 s, slew 0.698 rad/s, delay 0.10 s | M10 | EF-OOR (delay grid) | [M] |

---------------------------------------------------------------------------------------------------

## (E) Is gz in the family?

### (1) Support: does some θ reproduce gz's error law exactly? **No** [H]

Each reason below is a code fact that no parameter choice removes.
1. **Non-neutral family terms.**
   - The layer-3 surge Coriolis `−(m/m_u) q(ż + uθ)`, m ≥ 1260, is always present (operators_gen.py:844-851) and no other term has its form.
   - T9 noise is present w.p. 0.95.
2. **Hard bounds on exact-form terms** (definition EF-OOR):
   - M_t yaw inertia ≥ 1.433 M66 for every generalised yaw moment (N2);
   - k_c ≤ 1;
   - ρ_f ≤ 2.5;
   - W7 C_Dc ≥ 0.5 and r_exit ≤ 0.3;
   - W10 location;
   - the M10 delay grid (baseline);
   - in the W1-on branch, the pitch-stiffness floor 0.2025 ω_θ² at 25 kn.
3. **Forms that differ:**
   - the prime law (P1 logistic + latch + forced P2, against a ramp target with an exact asymmetric lag);
   - separable P3 against `T_net(T, u)`;
   - P5's `h(δ)` against `sin δ`;
   - the W4 gate against per-station emergence;
   - no steady planing momentum (no uτ in dd);
   - 48-station strip integrals against 5 × 3 samples.
4. Layer 3 is a random-feature class and contains no exact strip-theory integral.

Many labels that `comparison.md` called EF-OOR are EXACT-TAIL instead, because the Gaussian speed coefficients have unbounded support (VF3, N5). This does not change the answer, since items 1-3 hold regardless.

### (2) Closure: how small is the best θ's error? **UNKNOWN** (no constrained fit has been run)

Known facts:
1. Required push R at defect level (C.4), rms: u 0.92-1.18; v 0.82-0.87; r 0.38-0.42; w 4.73-6.07; q 0.66-0.87. Relative to e0 rms this is 1.0-1.3 in every channel.
2. Exact or inside-prior coverage exists for:
   - heave damping and stiffness;
   - yaw damping alone, and the lofi jet yaw share;
   - the jet pitch lever (pitch row);
   - ∂q̇/∂z through W1 at one speed;
   - ∂u̇/∂z through W1;
   - the surge Coriolis through Teven;
   - the W7 crossflow form (sway part);
   - the W3 entry momentum form (edge);
   - the static trim shift through W1;
   - the trim drag θ coefficient, with B1.
3. These need layer 3 or bounded modulation:
   - ∂ẇ/∂θ (×3-9 beyond layer 1);
   - pitch damping (×2.2-4.1);
   - negative pitch stiffness;
   - the stern lift × immersion²;
   - the wave forcing `H-w − W^lo` (4.7-6.1 m/s² heave);
   - the prime (P1 only);
   - the surge gain (UNKNOWN).
4. Layer 3's envelope is large enough. The small tier, with N ≥ 2 forces near the κ cap, reaches the heave/pitch coupling size: 2-3 × 3.9 m/s² against ≈ 4.3-5.0 m/s² needed for the ∂ẇ/∂θ shortfall × θ sd 0.05 rad at 25 kn. The fail tier goes up to 8.5 a_ref. Each force reads ≤ 18 columns through ≤ 8 filters with a random readout and one optional product.
5. Joint conflicts (one parameter needed at two values):
   - layer-3 x_g: < 0 for yaw damping, > 0 for pitch damping;
   - W1 q: 1.339 for trim drag, ≤ 1.13 for the slope term with W9 ≤ 1.3;
   - W1 on (∂q̇/∂z) against W1 off (T4 coupling route for ∂ẇ/∂θ).

### (3) Prior mass near the best θ: **UNKNOWN** (no metric computed)

Code facts:
1. Effective probabilities (D.5): P(W1 off) = 0.397. P(T4 on) = 0.665, with k_s drawn w.p. 0.5 and k_a w.p. 0.3. P(B1) = 0.688. P(T5) = 0.665. P(Teven channel-0 term) ≈ 0.24. Layer-3 tiers 0.75 / 0.20 / 0.05. P(small tier N ≥ 2) = 0.264. P(κ ≥ 0.07) = 0.12 per force.
2. Single-speed tail distances (N5): T4 coupling 7.7σ-72σ; pitch damping 3.9σ-21σ; yaw S1 0.9σ-6.1σ. The joint over speeds is smaller still.
3. Values at support edges: W3 k_w = π/2 and both β = 20°; W10 q_b = 2 and d_0 = 0; T3 x_pd = x_stern; W1 q = 1.3 (without B1).
4. P(every no-counterpart propulsion item off or neutral) = 0.0188 (fam_prop §6.7). Parameter neutralisation (D.5) is more relevant than on/off products.
5. fam_l1_l3 §8 and inventory 11.12 found no training law in the "six gz-like items strong" corner. That corner is defined by item strengths, not by this error law.

### Train/test differences (inventory.md): law vs input distribution

The plant law is the map `(s_k, P, delay lines, U_k, wave field on [t_k, t_k + Δ]) → e0_k` (10 channels).

**Change the law itself.**
- D1 (mechanism): all of section B.
- D10 (actuator model): gz's fixed jet with a 0.100-s delay; the family's M10 is random per episode and on a 0.04-s grid.
- D11 (family-only roll, wind, current, shallow water, debris, packets): present in training laws, zero in gz.
- D17 (APK): safety head only.

**Change only the input distribution.**
- D2 driver, D3 speed and thrust range, D4 turn pattern.
- D5 sea parameters, D6 line spectrum vs jitter, D7 fixed vs uniform wave direction, D8 one law for all gz missions.
- D9 window, D12 wave measurement and preview, D13 history noise, D14 normalisation.

**Interaction.** The network learns `p(e0 | tokens)`. Part of the law reads variables outside the tokens: P, fixed-point η_t and η_tt, 48-station integrals and intra-step evolution. So D2, D3 and D5-D7 change the *observable* law although the plant law is unchanged. In addition, A1-A3, E1-E4 and H1 change which hidden causes the family assigns to a token pattern.

D3: gz missions occupy 16-25 kn. There gz has stable equilibria; at 30 kn it oscillates (VG2), so D3 selects a sub-region of the law.

---------------------------------------------------------------------------------------------------

## (F) Open classifications and the computation (section G) that settles each

| open classification (section) | settled by |
|---|---|
| surge inertia / gain 1.339 (D.0), U-j surface (D.1) | G7 |
| linear lateral block, the λ schedule, stern-lift linear part (D.2) | G3 |
| stern lift × immersion², crossflow yaw, wave modulation, FK (D.1, D.2) | G4, G5 |
| heave-pitch couplings, pitch damping and stiffness, wave forcing (D.3) | G2, G5 |
| prime (D.1-D.3) | G6 |
| actuator channels, U-a / V-a / R-a (D.1, D.2, D.6) | G9 |
| static curve and attitude (D.1, D.3) | G8 |
| timing factors I_ZOH, I_Π (B.3) | G10 |
| closure E(2) | G2-G5 then G10 |
| prior mass E(3) | G11, G12 |
| pooled statistics (B.3, C.4 are two missions) | G1 |
| H1 vs T7 attribution (D.4) | G13 |
| observable-law dependence (E) | G14 |
| position channels (C.4) | G15 |

---------------------------------------------------------------------------------------------------

## (G) Open items needing numerical computation, ranked

The three senses:
- **S1 Support**: exact membership. Already **No** from code facts, E(1), [H]; no computation below can reverse it.
- **S2 Closure**: the smallest error-law distance over the coded support.
- **S3 Prior mass**: how much prior probability lies near the best θ.

Ranks are 1 = most decisive; "—" = does not bear on that sense. Every computation must run one mission or one chunk per process, check free RAM first (≈ 16 GB machine, often 2-10 GB free), and not overlap the running data job.

| item | S1 | S2 | S3 | cost |
|---|---|---|---|---|
| G1 stage-resolution defect arrays, 16 missions | prerequisite | prerequisite | prerequisite | 16 × ≈ 65 s, ≤ 0.4 GB |
| G2 heave-pitch catalogue + layer-1 closure | — | 1 | 3 | hours (optimiser), ≤ 1 GB |
| G5 layer-3 closure of the remaining residual | — | 2 | 5 | hours, ≤ 2 GB |
| G4 lateral and surge non-linear closure | — | 3 | 6 | hours |
| G3 lateral linear block (closed-form NLP) | quantifies the N2 hard bound | 4 | 4 | seconds |
| G6 prime: P1 / P2 / P13 / P14 against gz P | — | 5 | 7 | minutes |
| G7 surge gain and thrust surface (P3, B1, P14) | settles the D.0 UNKNOWN | 6 | 8 | minutes |
| G10 timing factors per term | — | 7 | — | minutes |
| G9 actuator channels on recorded commands | quantifies the delay-grid bound | 8 | — | seconds |
| G8 static calm curve and attitude, 14-32 kn | — | 9 | 9 | minutes |
| G11 prior-mass distribution D(θ) | — | — | 1 | 10 chunks × ≈ 50 draws |
| G12 analytic branch and tail probabilities | — | — | 2 | seconds |
| G13 H1 vs T7 attribution | — | — | 10 | needs a trained network |
| G14 observable-law dependence (D6, D7) | — | — | — (network-level question) | 4 re-recorded missions |
| G15 position channels | — | 10 | — | seconds (from G1) |

Why this order for S2: the largest required push relative to e0 is in heave and pitch (C.4: R/e0 = 1.08 and 1.29-1.34), and layer 3 is the catch-all whose success decides "approximable how well" for every APPROXIMABLE row. The prime is the largest term the network cannot observe (U-p 0.52-0.86 m/s²).

**G1. Defect arrays at stage resolution (prerequisite).**
- Inputs: `studies/_cache/onestep_gz/streams_T120.pkl` (16 missions, keys (seed, leg); XS, U). Each mission is rebuilt with `studies.final_eval.make_mission(seed, leg, 120.0, weights=wts, floor=floor)`, where `wts, floor = studies.sim2real_jet.theta(z_hand())`.
- Entry: scratchpad `wf_math/cmp_decomp.py` (`variants`, `L_calm`, `L_act`, `W_lo`, `FK`; `PlaningVesselGZ.step`, planing_vessel.py:522-553). Extend it to store, per plant step and Heun stage:
  - the stage state and time, P and `last_n_sub`;
  - the four variant derivatives;
  - `h_in = pl.prop_submergence(s[:6], t)` (as used at planing_vessel.py:524);
  - the 5 × 3 station elevations and moving-station rates;
  - `R(t)` per B.3.
- Output: `decomp_<seed>_<leg>.pkl` (step means as now, plus stage arrays) and the pooled B.3 and C.4 tables over 16 missions. Also check `max |<G> − Δν/Δ|` and that the identity residual stays ≤ 1e-5.

**G2. Heave-pitch closure (catalogue + layer 1), separately in the W1-on and W1-off branches.**
- Inputs: G1 arrays, gz commands and gz sea components (`pl.sea` a, ω, k, θ, φ). Write an adapter to an `operators_rb.RowSeas`; the adapter is not in the repository.
- Free parameters, in their coded boxes:
  - T3: c(S2), c(S3), κ, T4 `cKz`, `cKq`, `cks`, `cka`, `cdz`, `cdq`;
  - T2c: `c2['zq']`, `c2['M']`;
  - layer-1 m_w and ρ_q;
  - W1 (`p_d, p_h, q, c_h, l_x, p_tau, p_K, p_B`), W2 (flutter and vdP), W5, W7 vertical;
  - B1 via `B1.build` (cat_hidden_boat.py:534);
  - T7 heave and pitch loads;
  - layer-3 M_t (m, x_g, z_g, I_q,t, m_w).
- Entry points:
  - build a `CatDrawD10` (operators_cat.py:114) with `force_items`, then overwrite `draw.on[code]` and `draw.op_rb` attributes, then call `apply_l1_switches` + `_pack` (cat_base.py:771-800);
  - per stage point: `data3.plant_to_reduced`, `CatSea.sample` (cat_base.py:441), then `CatDraw.substep(st, sr, thr=T_s, noz=δ_s, sea, nu0, cmd=(T_c, δ_c), parts=True)` (cat_base.py:1017-1108).
- Objective A (defect level): `Σ_k |<R_hp>_k − <K_θ,hp>_k|²`.
- Objective B (e0 level, teacher-forced): one control step from each gz state with `operators_cat.simulate_cat(env=relabel._env(), xs14=XS_k, U=U_k, t0=t_k, seas=RowSeas, groups=[dict(op=OperatorCat, st, rows, rng)], n_steps=1, noise=False, act=G9 row)` (operators_cat.py:397). Hidden states are carried along the gz trajectory. Then `e0^θ − e0^gz = (XS^θ_{k+1} − XS^gz_{k+1})` in channels / Δ, through `data3.e0_from` (data3.py:78).
- Optimiser: box-constrained multi-start, e.g. `scipy.optimize.minimize(method="L-BFGS-B")` or CMA-ES for the gated items.
- Bounds, three ways: (i) as coded; (ii) Gaussian c1, c2 allowed to 3σ; (iii) every box ×3.
- Output, per branch and bound set: rms residual in w and q (and z, θ) at defect and e0 level; the active bounds; the fitted θ*.

**G3. Lateral linear block (closed-form feasibility).**
- Inputs: gz Jacobian rows (v̇, ṙ) with respect to (v, r, δ_s, T_s) at the calm equilibria of 16, 20, 25 kn (`vgz_check2.py` / `cmp_lin.py`, prime 1).
- Family open-loop linear matrix, re-implemented from:
  - operators_rb.py:665-686 (lofi + T3 S0/S1/pd + T5 f_r and x_j/I_r + T2c λ + Teven lateral with `n_k` at the gz running state), dropping the autopilot term `outer(bcol, Kc)`;
  - the layer-3 Coriolis linearisation (operators_gen.py:833-851) with M_t;
  - W10 `c(v + x_b r)` (cat_water_b.py:345-359);
  - P5's slope.
- Free parameters: e^{g_S0}(ξ), e^{g_S1}(ξ), x_pd, c_pd(ξ), λ(ξ), β, G, ρ_r, x_j, m, x_g, I_r,t, W10 (x_b, κ_b, q_b, d_0).
- Objective: weighted `Σ_u ||J^fam(θ, u) − J^gz(u)||²` in physical units.
- Output: residual per entry and speed; which hard bounds bind (expected: M_t[2,2], f_r range, W10 x_b); whether tails or layer-3 x_g are needed.

**G4. Lateral and surge non-linear closure.** Same machinery as G2, channels u, v, r.
- Items: W7 lateral, W10, W8, W9, W1 (surge parts), T7 (transverse, twist and gradient patterns), Teven, T2c, T3 (R, S0, S1, pd), H1 (neutral and on), P3/P5/P14 jet.
- Output: per-channel residual at defect and e0 level, the active bounds, and the W1-q / W9-μ_x conflict value.

**G5. Layer-3 closure of the remaining residual.**
- Inputs: `r(t) = R(t) − K_θ*(t)` from G2-G4 at the stage points.
- Model: `OperatorGenL3` (cat_base.py:821-869; `_draw_force` operators_gen.py:516; `gen_accel` :870/975). Keep its input columns (operators_gen.py:686-723), filter types and per-force caps 3κ a_ref; make the projection, filter poles and readout weights trainable (torch or scipy).
- Sweep: N ∈ {1, 2, 3} with κ ≤ 0.1 (small tier); N ≤ 8 with κ0 ≤ 1 (fail tier).
- Output: residual rms per channel against N and tier, at defect and e0 level (teacher-forced as in G2). This is the "APPROXIMABLE: how well" number.

**G6. Prime.**
- Inputs: from G1, per-plant-step h_in and P. gz law: waterjet.py:208-217 (prime band, τ_loss 0.05 s, τ_back 0.5 s).
- Entry: P1 (cat_prop.py:169-240), driven by `q["h_in"]` (cat_base.py:566-568) at the 0.04-s substep starts; variants with P2 (simple and full) and P13; P14 (cat_prop.py:871-879).
- Fit inside the P1 box: `min Σ (p_P1 − P)²`.
- Output: rms and max `|p − P|`; the U-p, V-p, R-p, H-p residual `(p − P) × (terms of B.3)` propagated to e0 by teacher forcing.

**G7. Surge gain and thrust surface.**
- Inputs: visited rows (c = U^T, T_s, δ_s, u) from G1.
- Target: `F*(row) = (M_t[0,0]/M11) f_x^gz(T_s, δ_s, u, P = 1)` (waterjet.py:153-188), with M_t[0,0] = m_u, or M_new[0,0] with B1.
- Family: `P3.thrust(prm, ctx, c, u)` (cat_prop.py:368-377) × the B1 rescale, plus P14 (:871-879), plus the `k_c` axial loss (cat_base.py:1150-1152), as a function of the M10 applied command.
- Output: residual rms and max (N), the active bounds (φ_id, k1, ε, P14 ε), and whether `U-h(i)` is EXACT, EF-OOR or APPROXIMABLE.

**G8. Static calm curve and attitude, 14-32 kn at 0.5-kn spacing.**
- Inputs: `running_attitude(u)` (planing_vessel.py:386), `calm_resistance(u)` (:424), T_eq by `brentq` on `prop.forces` (as vgz_check.py), and the free-surge run of final_check.py to flag non-equilibria. Output: the speed band without an equilibrium.
- Family static e0 in calm water: T3 g_R, P3, W11, B3, W1 (τ_eq, s^{p_K}), B1 (dz0, dth0), T4 offsets.
- Output: static e0 residual per speed, and the needed c's in σ units.

**G9. Actuator channels.**
- Inputs: recorded `U_k` and `XS[:, 12:14]` from streams_T120.pkl.
- Thrust: the closed form of A.3 against the M10 exact first-order lag (lofi.py:262-341) at delays {0.08, 0.12} s with the best τ'.
- Nozzle: Euler with slew (waterjet.py:237-240) against M10 `rate_noz`, `tau_noz` and the delay grid.
- Output: rms mismatch of e0 channels 8-9, and the `D_act − A^M10` residual in u, v, r for the best member.

**G10. Timing factors.** On G1 stage arrays, compute each B.3 term's effective kick-vs-continuous gain from its within-step time profile: `h Σ_j Φ(Δ − t_{j+1}) R(t_j) / ∫Φ(Δ − τ) R(τ) dτ`, per term and channel. Output: the coefficient rescaling a physically exact family term needs (constant-push values 1.204 / 1.059 / 1.048).

**G11. Prior mass.**
- Inputs: N ≈ 500 accepted draws: `operators_cat.build_ops` + `accept_cats` (operators_cat.py:352-385), in chunks of 50 per process.
- For each draw: teacher-forced one-step source predictions from the gz states of 2-4 missions (as G2), and `D(θ) = Σ_k ||e0^θ_k − e0^gz_k||²` normalised by the training σ_e (net_io §6).
- Output: the distribution of D(θ), the rank of D(θ*) from G2-G5 within it, and the mass within (1 + ε) D(θ*).

**G12. Analytic branch and tail probabilities.**
- Inputs: code constants (cat_base.py:92-145; operators_rb.py:107-116, 148-160, 376-482; operators_gen.py:283-315).
- Compute: the prior probability of each required corner (e.g. W1 off ∧ T4 on ∧ cks, cka drawn ∧ tail mass at the needed k_s + k_a over 16-25 kn jointly), and the product over independent draws. Extend final_sigma.py to joint-speed tail masses.
- Output: an upper bound on the prior mass of exact-form matches, per term and in total.

**G13. Attribution check (D.4).** On gz windows, compare the trained network's posterior samples with the H1-on and H1-off laws fitted in G4: the share of lateral e0 explained by `c_w t_slope`-driven roll terms against T7 transverse loads.

**G14. Observable-law dependence.** Re-record 4 gz missions with a jittered sea (`SeaState(..., jitter=True)`) and a random wave direction, keeping the hand controller. Recompute B.3 and the network's skill. This isolates D6 and D7; it does not bear on plant-law membership.

**G15. Position channels.** From G1, propagate R to the z, θ, ψ channels (`Φ_12` weighting; heading through r). Confirm that the fits of G2-G5 cover them with no new term.

## UNKNOWN (with reason)

- Closure (E2) and prior mass (E3): not computed (G2-G5, G11, G12).
- B.3 and C.4 are per-mission statistics on 2 of 16 missions (G1).
- The cause of gz's 30-kn oscillation (captive and free surge both oscillate; not analysed).
- The surge-gain class (G7); the lateral linear block (G3); W5(a) and H1 contributions to the linear pitch stiffness and yaw damping (not read).
- Whether the omission of `m_a u̇ τ` in gz's vertical strip force is deliberate (no comment in the code). It does not change any class, because the term is absent from gz.
- The accepted prior's parameter distribution beyond item frequencies (fam_l1_l3 §9).
- The adapter from the gz SeaState to an `operators_rb.RowSeas` needed by G2-G5 and G11 is not in the repository.
