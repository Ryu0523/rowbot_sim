# verify_gz_side: comparison.md sections A and B checked against the code

Method: re-read sim/planing_vessel.py, planing_vessel_gz.py, waterjet.py, actuators.py, env.py, control/reduced.py, sim/lofi.py, data3.py, relabel.py, data2.py, task.py, onestep_gz.py, cmp_decomp.py. Read-only checks in this folder: vgz_check.py, vgz_check2.py, vgz_check3.py (one process each, < 60 s); B.2/B.3 statistics recomputed from decomp_7_1.pkl and decomp_0_0.pkl; jet ratios from streams_T120.pkl.

## Errors (claim -> correct statement, code)

1. A.5 static calm defect table, label. The text says the table is "the reduced acceleration with command = equilibrium thrust". The numbers are minus that, i.e. the defect d = G − L with G = 0. At 16 kn the reduced acceleration is u̇ = (T_eq − k_D u²)/m_u = (1492.5 − 947.6)/1968.58 = +0.277, ẇ = −ω_z²(z_run − z0) = +0.838, q̇ = −ω_θ²(θ_run − θ0) = +0.578 (reduced.py:394-395, 397-424; vgz_check.py). The table's signs fit d, not L. lofi_error §9.1 lists e0 (propagated: −0.369, −0.441 in w, q), so the table does not reproduce it. It reproduces d = e0 / (0.440, 0.763).
2. A.5, the 30-kn column. The "calm running state" at 30 kn is not a gz equilibrium. At (z_run, θ_run, T_eq) gz gives G = (u̇, v̇, ṙ, ẇ, q̇) = (−0.014, 0, 0, −0.210, +0.085) (vgz_check2.py). The captive calm run (planing_vessel.py:386-422) never settles: over the last 10 s of a 30-s run, z stays in [0.140, 0.157] m and θ in [−0.0021, +0.0029] rad (vgz_check3.py). So the 30-kn defect is d = G − L = (+0.019, −0.298, +0.243), not (0.033, −0.088, 0.158). The 30-kn Jacobian and eigenvalues (4.69 rad/s, ζ 0.73) are linearisations about a point on a limit cycle. "At task speeds gz has no lightly damped pitch oscillation" does not hold at 30 kn (16-25 kn: confirmed). The sentence "static parts of U-h and U-j cancel" is also inexact: their sum is (k_D u² − T_eq)/m_u, which is −0.277 at 16 kn (not zero), and that holds only at a true equilibrium.
3. A.5 end: "the reduced coefficients were identified at u_ss = 17.4 m/s". This is true only for surge (k_D = thr_id/u_ss², m_u = 2k_D u_ss τ_u, reduced.py:150-165, 321), the heave/pitch decays (:172) and the yaw step (τ_r, k_nf, :231). The rest were identified elsewhere: z0 and θ0 at u_design = 25 kn (:297-299); m_v = Mtot[1,1] and k_sw = visc[1] at the 25-kn design attitude (planing_vessel.py:374-383); k_lin and k_js from a 90-s turn started at 25 kn with 7000/12000 t_max and a 15° nozzle (reduced.py:263-284).
4. B.2: "for surge, sway and yaw the nonlinear terms u|u|, v|v| and u r make Π only approximately linear". Model0's yaw is ṙ = (k_nf k_js T_c⁺ sin δ̃_c − r)/τ_r, which is linear in r and has no u r term (reduced.py:482-488). So Π_r is exactly linear in the r deviation. The term `u r` sits in sway (`m_c u r`, :452-453), and `u|u|` in surge.
5. B.2 sway factor "≈ 0.875". That is the first-order value 1 − aΔ/2. The exact constant-defect factor is (1 − e^{−aΔ})/(aΔ) = 0.885, with a = k_lin/m_v = 1.0433 /s (v = 0; model0 sway is Euler, so even this is approximate).
6. B.3 U-h sub-term (i). The term "(1 − M11/m_u) u̇^gz, rms 0.25 / 0.29" uses the full gz acceleration (cmp_decomp.py:123 uses Gf[0], with waves, jet and actual P). So it is a split of the whole surge defect d_u, not a part of U-h = D_hull,u. Inside D_hull alone the inertia piece is 0.2533 · G_calm0,u (T_s = 0, no waves).
7. B.3 combined heave. The mean of `H-h + H-w − W^lo` on (0,0) is −1.02, not −0.99 (−0.99 is the mean of R, the C.4 quantity). The other combined numbers are confirmed.
8. B.3 actuator decay numbers are swapped. Exact first order over 0.04 s (M10 style) gives e^{−0.04/0.6} = 0.93551. gz's explicit Euler over 2 × 0.02 s (actuators.py:194-196) gives (1 − 0.02/0.6)² = 0.93444.
9. B.3 I_ZOH, "about 9 % stronger in heave". The ratio 1/[(1/h)∫₀ʰΦ_22] = 1/0.910 applies only to the last substep. A kick after substep j reaches the step end through Φ_22(Δ − t_{j+1}), and the continuous force reaches it through ∫Φ_22(Δ − τ)dτ. For a constant push over the control step, kicks/continuous = h Σ_{i=0}^{5} Φ_22(ih) / Φ_12(Δ) = 1.204 (heave), 1.059 (pitch) and 1.048 (yaw); the per-substep heave ratios run from 1.10 to 1.55. Source of the kick: ZOHInjector (data2.py:106-120) after each plant substep (task.py:332-333).
10. B.3 integration item 3, "the source world [evaluates <L>] at 6 substep starts, a quadrature of the same ∫L dt". That is true only for surge and sway (Euler, reduced.py:491, 497). For w, q and r, f_h uses exact transitions (:418-424, 482-488). The source term is therefore Σ_j [(Φ(h) − I) x_j]_vel / Δ, which integrates L along model0's own sub-path from each x_j, not h·L(x_j). This matches C.3's own g0(s_j) = [f_h(s_j, ·) − s_j]_ν.
11. Missing variables or paths in the B.3 forms. The jet's and the prime's moment z_n f_x enters surge through ẇ in X_p = m(g + ẇ)Θ (planing_vessel.py:349-355). Exact forms:
    - U-p = (P − 1)[1 + mΘ z_n (M_hp⁻¹(d))_{12}] f_x / M11. This is still exactly linear in P, but it misses the ẇ path (−0.13 % at 25 kn).
    - U-j and H-j also read z and θ, through Θ and M_hp(d⁰).
    - H-w omits the wave change of the friction pitch moment 0.55 ΔD_f (wetted surface; :285-296).
    - R-h writes d_i where it means d_i⁰, and H-h writes D_f where it means D_f⁰.
12. A.3 M10 "exact solution". Exact only for first-order channels. Second-order channels (`zeta` drawn w.p. 0.3 given a lag) use N_SERVO = 4 trapezoidal substeps (lofi.py:187, 316-341). The list also omits the asymmetric `ratio` and `zeta`, and M10 thrust travel is [0, t_max] (lofi.py:267-268), not gz's [−t_max/3, t_max].
13. B.3 U-a and "states outside", "thrust: affine". The map is piecewise affine: a delayed command below 26.52 N is set to 0 (actuators.py:191-192), and recorded thrust fractions go down to 4.6e-7 (lofi_error §9.2). The rate limit ±21219 N/s never binds, because |c − T_s| ≤ 4/3 t_max < 0.6 × 21219.
14. B.3 wave-field paragraph. "5-station sampling resolves every component" holds only in the no-aliasing sense: λ_min = 2πg/(3ω_p)² = 4.34 m > 2 × 1.32 m. Thirty numbers per token do not determine 120 components (24 freq × 5 dir, task.py:59; nonzero spreading at ±90°, step0_preview_spec.py:42-46). "The field enters only through these 30 numbers plus the error history" omits the elevations, states and commands of every earlier token in the 256-token context.
15. Jet gain ranges ("G/T_net = 1.33 … 2.26", "side-force gain 1.44-2.51"). The statistic is not stated. Recomputed on the replayed states (κ and the dead-band floor of V_j, waterjet.py:163-172):
    - G/T_s: 1-99 % 1.54-4.06 (7,1), 1.49-4.99 (0,0); 5-95 % 1.58-2.44, 1.51-2.50.
    - G/T_net: 1-99 % 1.33-2.41, 1.27-2.62.
    - G/T_s → 23-44 at T_s = 26.5 N.
    - D.2 gives 1.4-4.5 for the same quantity.
16. B.1 "verified to 0.0, exact". The residual max |e0 − Σ terms − Π| is 2e-14 on (7,1) and 3.1e-6 on (0,0). On (0,0) the terms come from a replay that drifts 1.3e-6 from the stored states, while e0 uses the stored states (cmp_decomp.py:106, 129). "0.0" is print precision.
17. Small items:
    - Zero jet flow at T_s = 0 is waterjet.py:163-165 (not 166-172).
    - The A.2 jet formula covers T_s ≥ 0 only. The reverse branch is :187-188, unused since commands are ≥ 0, and V_j is evaluated at max(T_s, 26.52 N) (:168-171).
    - Symbols ε (in G(s, P, ε)) and χ (A.2) are never defined.
    - Roll also appears at planing_vessel_gz.py:236-238 (rel = φ − w̄) and planing_vessel.py:492-494 (y_i φ with y_i = 0). The conclusion "0 in the five channels" stands.
    - h_in "every 0.12 s": the second sample is a forecast (λ = 0.1) at a dead-reckoned pose.

## Confirmed against the code (no change needed)

- A.1 mapping and indices: reduced.py:519-526, operators_rb.py:92-93.
- Every gz row in A.2: planing_vessel.py:258-366 with stem_rise = 0 → k' = k'' = 0, transom_correction False → ctr = 1, k_a = 1, c_dc = cos 20°, vcg 0.55, z_n −0.40 (hydro/hulls.py:146-151, hydro/planing.py:101-121). The closed form X_p = m(g + ẇ)Θ holds to 1.8e-12 on 20 random wavy states.
- Roll row (planing_vessel_gz.py:223-239).
- Reduced rows (reduced.py:394-498) and every parameter value: lofi_scarab195.json, rud_stall = rud_max.
- Actuator chains and delay content (actuators.py:29-42, 186-196; waterjet.py:208-240; planing_vessel.py:119-138, 552-553).
- Δ = 12 × 0.02 = 6 × 0.04 (env.py:80); Mtot, Iy, Z_rest, Θ_rest, t_max, the dead band, the intake point, the prime band and the time constants.
- A.5: every Jacobian entry at 16/25/30 kn and every eigenvalue at 16/20/25 kn (30 kn see item 2); |u|m22(d_1), Σ½ρC_lat d dx, A33/A55, d_1; A35 = 1463-3446 kg m.
- B.1 algebra; B.2 heave 0.440/−0.070, pitch 0.763/0.449, yaw 0.762/0.565, surge 0.98, and the position-update caveat.
- Π rms and corr(e0, Π) = −0.94 … −0.97 in u, r, w.
- Every B.3 rms and mean except item 7; FK, Coriolis and inertia rms; the prime shares 0.60/0.79 and 0.15/0.27; U-p rms 1.27/1.57 on steps with P < 0.5.
- ZOH factors 0.910/0.974/0.954; the token content: residuals.py:52-56 + mpc_learned.py:524-530, so X, Y, P, φ, p are not in the token.

## UNKNOWN

- Whether the 30-kn limit cycle persists with free surge (only the captive run was checked; that check is cheap).
- Which percentile defines the jet-gain ranges in B.3 and D.2. The text does not say.
