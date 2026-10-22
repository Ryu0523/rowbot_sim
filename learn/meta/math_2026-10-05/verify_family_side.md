# Verification of comparison.md sections C, D, E (family side) against the code

Code read: operators_rb.py (_smooth 148-160, draws 376-480, rb_accel 776-961, T7 484-600), ops_m15.py 77-122, cat_base.py (95-180, 290-420, 495-597, 1015-1159), cat_w1.py, cat_water_a.py (W2-W6), cat_water_b.py (W7-W12), cat_hidden_boat.py (H1, B1), cat_prop.py (P3, P5, P14), operators_gen.py 283-316, 686-725, 833-852, operators_cat.py 486-570, 690-737, data3.py 1-107, model3.py 79-115, episode7.py 125-240, episode8.py, reduced.py 366-505. Check script: wf_math/vf_check.py (one process, < 5 s).

## Errors (most severe first), with corrections

1. **D.2 stern Munk row, "No term of exactly this form exists": wrong.** W10 has exactly the form `Y = -kap_b k_lin (U_r/10) ((h_bow - d_0)+/T)^q_b (v_r + x_b r)`, with yaw moment `x_b Y` (cat_water_b.py:334-361). Setting q_b = 2 ∈ U[1,2] and d_0 = 0 ∈ U[0, 0.5T] gives `|u| d^2 (v + x r)`. The gz coefficient (π/2)ρ needs kap_b = 1.418 ∈ LogU[0.1, 2]. What is out of range is the place: x_b ∈ U[0.25, 0.5]L = [1.35, 2.70] m, and h_bow is the mean of stations 3-4 (:344), whereas gz uses x_1 = -2.125 m and the stern immersion. Correct class: **EF-OOR (location and station)**. A second point the row misses: layer 1 cannot move the lofi `k_lin v` (at x = 0) to the arm x_1, because T3's factor applies to it with `e^{g_S0} ≥ 0.6` in the bounded part (operators_rb.py:796-797).
2. **D.3 heave-pitch coupling, "APPROXIMABLE only through layer 3", and E(3).4: wrong.**
   (a) The W1 moment `Q_θ = s_p W Φ l_x eps/(1+|eps|)` (cat_w1.py:26-30, 114) has the linear slope `∂q̇/∂z = W l_x/(hbar_run I_q)`, with hbar_run = 0.1418 (cat_base.py:394). It runs from 0 to 25.8 /s² over the 40-draw I_q range [3037, 9439], or to about 34 over the full M_t range. The sign matches gz (z up gives bow-down acceleration). So gz's 6.3-20.7 is **inside the prior** (W1 on, P = 0.603). Side effect: the same term adds pitch stiffness through `∂eps/∂θ = mean(x_i)/hbar_run = 3.66`, which goes the wrong way for gz.
   (b) The W2 flutter part, `a_z = -f k35 e_θ`, `a_θ = -f k53 e_z` with k35·k53 < 0 (cat_water_a.py:107-108, 135-138, 152-157), has gz's sign structure (k35 > 0, k53 < 0). Its product |k35 k53| = ρ_f κ*² is at most 2.5 × 158.2 = 395. gz needs 743 / 1244 / 1302 / 793 at 16 / 20 / 25 / 30 kn (ρ_f = 4.7-8.2), with l_f = 1.5-4.5 m inside [0.54, 5.4] m. Class: **EF-OOR, product short by a factor 1.9-3.3**. The form is linear only for |e_θ| << A_θ ≤ 4° and |e_z| << A_z ≤ 0.15 m.
   (c) "Only inside its porpoising gate" is misleading. The gate `σ((m0 + m1(U-U_id)/U_id)/w_m)` with m0 = 1.5° and w_m = 0.2° is 0.9994 over the whole task range (:118-119, 145-147).
3. **Gaussian support makes many EF-OOR labels wrong as stated.** In `_smooth`, c1 and c2 are drawn N(0, (σρ)²) and N(0, (σρ²)²) with no truncation, and neither `_g` nor `_gv` clips them (operators_rb.py:148-160, 615-617). So g(ξ) has unbounded support at every ξ, including ξ = 0, where g = c0 - c2/2. Under the document's own definition ("parameter outside the support"), the following are inside the support, at roughly 6-20σ: T4 coupling (×3-9), pitch damping (×2.2-4.1), T2c λ = 2.6, T3 c_R0 ≈ 0.45, the heave stiffness at 30 kn, and e^{g_S0} → 0. They should be relabelled "in support, negligible density (kσ)". Bounds that really are hard: k_c ∈ U[0,1] (:116), μ_x ∈ U[0.4,1.3], layer-3 m ∈ U[1260, 1561.9], M_t yaw inertia ≥ 4203 (operators_gen.py:288-300), the M10 0.04-s delay grid, ρ_f ≤ 2.5, and K = ω² e^g > 0. Because of the last one, a negative pitch-stiffness factor (gz at 16-25 kn) is outside the support even with the Gaussian tails.
4. **D.1 U-h(ii), Coriolis pair "EF-OOR as a pair": wrong.** The Teven surge feature `n_v n_r = v r L/(0.1 u_mid)² = 3.857 v r` passes through φ(x) = x/(1+|x|/4) with weight W ∈ [0.04, 0.6] m/s² (operators_rb.py:462-471, 876-883). With layer-3 m = 1400 (sway exactly equal to the lofi law) and λ = 0, the surge needs 0.952 - 0.711 = 0.241 v r, i.e. W q_3 = 0.0625, which is inside. φ costs about 7 % of a term of about 0.02 m/s² at v = 0.8, r = 0.09. Class: **APPROXIMABLE, exact to first order in v r**. (Without Teven, the pair needs λ = 474.1/(210.2 - A11) ∈ [2.60, 6.74].)
5. **D.1 U-h(i), "thrust gain ≤ (1.1 + 0.1)·1.111 = 1.333 < 1.339": the bound is wrong.**
   - P3 thrust is `t_max g(c) φ(u)` with `φ = φ_id exp(-k1(s-1) - k2 max(s-1,0)²)` (cat_prop.py:364-373). For s < 1, which covers all of 16-25 kn, φ > φ_id: up to 1.1 e^{0.6×0.36} = 1.365 at 16 kn, or 1.52 with B1.
   - The Bernstein g(c) (:356-362) can exceed c for c < 1.
   - P14 is bounded by ε t_max c_n, i.e. 0.1 T_c, not 0.1 T_a (cat_prop.py:855-858, 878-879).
   - So the 1.333 bound holds only at c = 1 with u ≥ u_id.
   - The U-j fit (44 N, fam_prop §6.1) targets f_x at gain 1. That is inconsistent with U-h(i), whose framework needs the target `(m_u/M11) f_x`.
   - Class: **UNKNOWN** until P3 (+B1, P14) is fitted to 1.339 f_x on the visited rows.
6. **Yaw: "yaw damping inside" and "lofi jet share inside" are each true but not true together** (E(2).2 lists them as independent). The lofi + T3 + T5 yaw damping is `-e^{g_S1} f_r r/τ_r` with `f_r = I_r0/I_r` (operators_rb.py:395, 798, 840-841). Matching x_j/I_r = -7.438e-4 with x_j ∈ [-2.665, -1.585] forces f_r ∈ [1.466, 2.31]. Then gz's -1.399 /s at 25 kn needs e^{g_S1} ∈ [0.24, 0.40], below the bounded 0.6. Inside the prior only through the layer-3 Coriolis term -m x_g u r / I_r,t with x_g ≈ -0.16 m ∈ [-0.27, 0.27] (operators_gen.py:847). That same x_g lowers pitch damping by +m x_g u q / I_q,t.
7. **D.3 vertical cross-flow, the reason given for "APPROXIMABLE" is incomplete; two parameters are out of range.** W7 sums ρ(b/3) dx C_Dc |V|V over three points, so its width is b (cat_water_b.py:204-213). gz uses ρ C_dc c(d) with c ≤ y_c ≤ b/2 (hifi_hull §2.2). So the required C_Dc = 0.9397 c/b ≤ 0.47, below U[0.5, 1.2]. Exiting points are also multiplied by r_exit ∈ U[0, 0.3], while gz applies the full drag at V < 0. Class: **EF-OOR (C_Dc, r_exit) plus a form difference** (no uΘ in dd).
8. **D.3 jet pitch moment, "EXACT for the thrust part": not exact.**
   - T5 sets a_ż = 0 (operators_rb.py:843).
   - Through the M_t off-diagonal, only |m x_g| ≤ 422 kg m is available, against gz A35 = 1460-3450, so the heave projection (H-j w 0.07-0.08) is EF-OOR.
   - The lofi share uses layer-1 I_q and the chain error uses M_t, two different inertias. z_j is drawn only w.p. 0.665.
   - Class: EXACT for the pitch row only.
9. **D.1 U-h(iii) (W1): three corrections.**
   - Without B1, W1 needs q = (m g/M11)·m_u/W = 1.339, outside U[0.8, 1.3] (cat_w1.py:59). It is inside only with B1 ε ≥ 0.111, because B1 scales ctx.W (cat_hidden_boat.py:549-552).
   - The slope term is W q arctan σ, not W arctan σ.
   - Omission: W1 also has `-W Pe τ_eq eps` (cat_w1.py:112). This is an in-prior linear counterpart of gz's ∂u̇/∂z = 4.1-11.1 /s² (A.5), coefficient range 0.5-26, and section D never classifies ∂u̇/∂z.
10. **D.3 damping and stiffness counterparts are missing.**
    - Pitch and heave damping, and the ∂ẇ/∂q coupling, also come from W7 vertical, a quadratic term distributed over stations (Σdx x² = 15.3 m³; at |V| = 1 m/s it is worth about 4-12 /s in pitch), and from the layer-3 Coriolis term -m x_g u q/I_q,t (up to 1.8 /s). So "×2.2-4.1 beyond" overstates the shortfall.
    - B1's frequency change (1 - e_m ε)² ≥ 0.5625 (cat_hidden_boat.py:566-576), together with T4's 0.64, puts the 30-kn pitch stiffness factor (0.45) and the heave stiffness factor (0.617) inside the prior.
11. **W9 (D.1 U-w, D.5, E(1)).**
    - The comparison should be in accelerations: μ_x m g (1+ε)/M_ctx,00 ≥ 2.79 m/s² per unit slope against gz 1.23-1.89, i.e. a factor 1.5-2.3, not 1.8-3.
    - W9's slope is the wet-station mean of np.gradient (cat_water_b.py:311-322), not σ.
    - A static T7 gradient-pattern load can cancel W9's excess, so "W9 coefficient" is not a support failure of the family as a whole.
12. **C.2, "each item soft-saturated at 3 × nominal".**
    - W4 is unsaturated (cat_water_a.py:289), and the W3/E3 impulses are unsaturated.
    - W1, W2, W5 use the half-clip nominal (cat_base.py:167-177).
    - Layer-3 forces and its Coriolis term keep the unloaded Minv even when B1 replaces M_ctx.
13. **D.5 mixes two kinds of probability.** The P-item column (0.3, 0.8, 0.608, …) gives raw P_ON, while the W/H/B/A/E entries are sparse-adjusted (0.85 p + 0.0079). For example, P4's effective probability is 0.263.
14. **C.3: the target definition is VERIFIED.** Both worlds use e0_split → e0_from with relabel._env() (data3.py:78-107, model3.py:86-109, relabel.py:232-240). XS is s0 at step start, at episode7.py:188/231 (not 207/230) and episode8.py:214/259. The target has 10 channels: 5 velocities, z, θ, ψ, and 2 actuators (data3.py:19-24, 85-88). Sections B and D analyse only the 5 velocity channels, so the position and actuator channels are unclassified.
15. **Minor.**
    - T4 c_ka smooth sd is 0.05, not 0.1 (operators_rb.py:421).
    - "d1 = d0 makes T7 static" holds only for first-order filters, drawn w.p. 0.4 (:494).
    - U-p row: P14 reads h_in/h_run (cat_prop.py:874-875), a bounded counterpart of the prime term (≤ 637 c N).
    - Axial-loss shortfall (≤ 136 N): approximable by P14 (which reads c and δ) and by the Teven (δ/rud_max)² surge feature.

## Checked and correct
- LIM = (12, 12, 6, 36, 9), and the layer-3 caps 0.9 a_ref (small tier) and 8.5 a_ref (fail tier).
- m ∈ [1260, 1561.9], M_t I_r ≥ 4203, k_c ∈ U[0,1].
- W7 lateral form, with C_Dl = 1 inside. The keel line k_i = Z_rest - 0.55 - x_i Θ_rest reproduces the family keel5, so hc_i equals gz d_i.
- W3: C_m = π/8, and d_c = 0.2346 m at k_w = π/2.
- T7 transverse and twist patterns, and correction 0.2 item 2.
- W1 trim-shift numbers.
- The layer-3 surge Coriolis term q(ż + uθ) cannot be made neutral.
- H1 can be made neutral.
- The source-world composition (C.1) matches the CatInjector path (operators_cat.py:697-737).
- W^lo heave 1.239 η̄ per substep (Φ21(h) = -1.2389).
