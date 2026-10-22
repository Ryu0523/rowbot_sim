# Empirical register, Track B (Rankine solver, `hydro/rankine/`) -- entries to merge into EMPIRICAL_REGISTER.md

Acceptance rule G3 (`learn/meta/ACCEPTANCE_2026-10-06.md`): every empirical input is registered with source, data
range, whether our vessel is inside it, and its uncertainty. This file collects the Track B entries; the orchestrator
merges them into `learn/meta/EMPIRICAL_REGISTER.md` (owned by the physics track).

## Stage B3 (2026-10-06, updated after the B3 review): no empirical inputs

The steady free-surface operator chosen in B3 (`hydro/rankine/shifted.py`, decision D1) contains no empirical input:
every coefficient is derived (Numerov weights, box correction 1/24, the stability bounds 6 and 0.4 of the blended rows),
and the physical constants g and rho are parameters (G4). The remaining choices are numerical devices, not physical
inputs, and each has a registered sensitivity check instead of a G3 entry:

| Device | Value | Check (registered tolerance) | Where |
|---|---|---|---|
| Entry ramp length | 1 lambda0 (C2 ramp of the free-surface condition) | S6 entry x2: change < 1 % | stage note 1.1, 8.1 |
| Downstream beach | length 2 lambda0, amplitude decay e^-6 | S6 beach x2: change < 1 % | stage note 1.1, 8.1 |
| Domain extents | upstream 1.5 max(L, 3f) + entry, downstream max(2.5 lambda0, 3f) + beach, lateral max(0.8 L + 0.5 lambda0, 3f) | S6 each extent x1.5: change < 1 % | stage note 1.1 |
| Resolution guard | blend weight l >= (2 - 12 w/(1.25 z_max))/5, cap 0.4; inactive at >= ~3.4 panels per wavelength (z_max n = 16.2; corrected after the B3 review, first entry said ~13); state reported per solve (`info["guard_l"]`), warning outside S4; production grids chosen per lambda0 (n >= 20) so it stays inactive | D1c (row stability), S4 runs it active, S4b production members require l = 0 | shifted.blend_weights, stage note 8.2 |
| GMRES tolerance and acceptance | true relative residual 1e-12; stall rule (two restarts without halving the true residual) stops only at or below the registered floor 1e-10 = 100 tol, otherwise iterate to maxiter and report converged = False; `SteadyShiftedFS.solve` raises above the floor (measured round-off floor of the FFT products 1e-12 ... 2.6e-12 above N ~ 5e4) | F3: FFT-GMRES = dense LU to 1e-8; residual printed in every stage-3 row | stage note 8.1, 8.3, 8.8 |
| Kelvin-function quadrature (check S2K only) | 64 Gauss-Legendre nodes in theta; E1 by series / continued fraction / asymptotic series | S2K-G: 64 vs 128 nodes 1e-8; E1 vs mpmath 1e-12 | `hydro/rankine/kelvin.py`, stage note 8.8 |

## Stage B4 (2026-10-06): transom wet branch (and assumptions of the hull case)

The steady surface-piercing solver (`hydro/rankine/piercing.py`) is derived except for the WET transom branch, which
is WaveBEM's empirical model (port design W-33/W-34, P-10) re-implemented as a boundary condition. Its inputs are
registered here (G3); they are used only as constants of that branch, never to weight forces between branches (G1):
both branches are tabulated wherever both exist.

| # | Input | Value used | Source | Data range | Our cases inside? | Uncertainty / scatter used |
|---|---|---|---|---|---|---|
| B4-1 | Transom dry fraction eta (0 wet, 1 dry) [**open G3 item** (B4 review): the scatter used is an assumed +-20 % shift in Fn_T, not the measured scatter G3 requires; the eta scatter and l_h = 3 ... 8 are run in the next transom step] | eta = min(1, 0.05 Fn_T^2.834 (B_T/T_T)^0.1352 Re_T^0.01338), Fn_T = U/sqrt(g T_T), Re_T = sqrt(g T_T^3)/nu, nu = 1.307e-6 m^2/s, at model scale (L = 2.5 m) | WaveBEM `free_surface.cc` (comment "Doctors regression formulas"), most likely Doctors, Macfarlane & Young (2007), Int. Shipbuilding Progress 54:145-163 (five simplified transom models, local transom wave measurements). **Original not obtained** (paywalled; WebFetch 403): coefficients as quoted by WaveBEM, not verified against the paper | not available (the paper's model range could not be checked); the published abstract states the Reynolds number plays a minor role | unknown; Wigley-transom test hull B_T/T_T = 2.3, Fn_T = 2.8 (Fn 0.35), 3.6 (Fn 0.45), Re_T ~ 1.9e4 | measured scatter not available: registered substitute = a +-20 % shift of the ventilation curve in Fn_T (eta evaluated at 0.8 Fn_T and 1.2 Fn_T), plus the two pure branches eta = 0 and eta = 1 which bracket every value (check T4). To be replaced by the paper's scatter if the orchestrator obtains it |
| B4-2 | Hollow length behind the transom | l_h = 5 transom draughts (WaveBEM's constant; its Doctors-type hollow-length regression is commented out in WaveBEM) | WaveBEM `free_surface.cc` (constant `lh = 5`) | -- | -- | not varied in B4 or in the B4 review re-run (open G3 item: sensitivity l_h = 3 ... 8 in the next transom step) |
| B4-3 | Hollow shape | parabola from the transom edge depth with the buttock slope to z = 0 at l_h abs(z_T) | WaveBEM (W-34) | -- | -- | part of the model, not varied |

Assumptions of the B4 hull case (not empirical inputs, listed for completeness): Wigley sides vertical above the design
waterline (published offsets end at z = 0; effect of second order in sinkage/trim); CG at the design waterline above the
LCB and tow force through the CG (WaveBEM W-38; the free-model data do not give the model's KG); the transom face closure
of the dry branch (phi_x = g z_T/U on the face, stage note 9.2) is a modelling choice of the linear transfer, not an
empirical input.

The steady operator itself (band rows, hull-wall virtual point, wave-cut functional) contains no empirical input; its
numerical devices and their checks:

| Device | Value | Check | Where |
|---|---|---|---|
| Band width | J = ceil(2 y_max/dy) lines (line spacing compressed by at most 1/2) | H11 grid series; B4-E3/E7 | piercing.build_grid |
| Hull-wall virtual point | first-order Taylor ghost from the wall conditions at z = 0 on the hull side y = H(x, z): phi_y = -H_x (U - phi_x) - H_z sigma and, for H_z = 0, sigma_y = H_x sigma_x + H_xz (U - phi_x) + H_zz sigma (separate sigma and phi ghost lists). [Corrected in the B4 review: the B4 entry said d sigma/dn = 0 at a vertical side, which is wrong wherever the sections curve below the waterline (Wigley H_zz = -25.6/L at midship); the flare terms of the sigma condition (H_z (2 phi_yy + phi_xx) - H_x H_z phi_xy) are still omitted] | H4 split, H6 near/far, H11; H13 (flared hull, before any non-Wigley use) | piercing._Columns, stage note 9.7.4 |
| Wave-cut functional | cut 1 lambda0 behind the stern; lateral wavenumbers <= 0.5 pi/dy | H7 (exact Kelvin field, 1 %) | piercing.cut_from_samples |

## Earlier Track B stages

Stages 1-2 (BIE core, zero-speed free surface) and the stage-3 review used no empirical inputs either (references:
Lamb, Hulme 1982 table, Havelock 1931 closed forms, Capytaine as an independent solver). The empirical inputs foreseen
by the port design (`learn/meta/WAVEBEM_PORT_DESIGN.md` section 8.8: Doctors' transom-ventilation regression,
transom hollow length, ITTC-1957 friction if total resistance is reported): the first two are registered in the B4
section above; ITTC-1957 is still unused (B4 compares wave resistance only).

## B4 review (2026-10-06): status of the transom entries

The transom branches are implemented but **not verified** (stage note 9.7.3/9.8): the momentum identity of the linear
model with the face closure (check T3) does not close at the grids used, and the double-body d'Alembert defect of the
transom meshes (check T8) is of the size of the resistance. Until that is resolved the empirical inputs B4-1 to B4-3 are
exercised (wet free equilibrium at eta = 0 and at the lower scatter bound eta = 0.628, check T5) but their effect on
forces cannot be separated from the discretisation error. No new empirical input was added in the review.

## Stage B5 (2026-10-06): forward-speed radiation and diffraction (`hydro/rankine/unsteady.py`, `tables.py`)

No empirical input. The linear Neumann-Kelvin radiation/diffraction model, the m-terms, the force definitions, the
Kramers-Kronig and Cummins relations and the energy identity are derived (stage note section 10.2); g and rho are
parameters (G4). Data-side assumption (validation only, not a model input): the KG of Journee's 3 m Wigley I model is
not given in the available sources; KG = 0.170 m is assumed for the RAO rows (V4) and its effect is reported for
+-0.05 m. The numerical devices and their registered checks:

| Device | Value | Check (registered tolerance) | Where |
|---|---|---|---|
| Beaches on all sides | nu = nu_max max(s_up^2, s_dn^2, s_lat^2), nu_max = 1.5 sqrt(g/L), zones 0.75 / 2.0 / 1.5 L (upstream / downstream / lateral) | DOM: nu_max x2 and x0.5, zone lengths x1.5, extents x1.5: each change < 2 % | build_grid, stage note 10.1 |
| Upwind-biased d/dx | fifth order (6 points) in the interior, third order next to the ends, inflow ghosts zero, fully upwind at the outflow end | SP (every mode decays at forward speed), GS (grid series < 2 %), TD/CU, ST0 near/cut (information) | dx_operator, stage note 10.2 item 2 |
| Wall rule [superseded in the B5 review by the trace closure: first-order inconsistent, error w' f_y on the wall line] | cross term of d/dx at fixed y dropped at the hull-adjacent band point where its upwind side lies behind the hull wall (afterbody), central at the next point | SP (B5-E4: the downwind alternative is unstable), GS | _index_derivative "wall" (kept as dx_operator(wall="rule")) |
| Grid zones | uniform cells 0.5 L downstream and 0.3 L laterally, stretching 1.12 / 1.06 / 1.08 per cell | GS, DOM | GRID_BASE |
| Validity band of a table [superseded in the B5 review: the zero-speed wavelength ignores the forward-speed shortening of the k2 waves; see the B5 review section] | omega_hi: zero-speed wavelength >= 8 hull-region cells; omega_lo: lateral beach >= 1 zero-speed wavelength (values outside are solutions of the same causal system but not resolution-verified) | -- (reported per table) | tables.validity_band |

## B5 review (2026-10-07): numerical devices changed or added (still no empirical input)

The review fixes add no empirical input. The reference model of the submerged-sphere check (m-terms of the double-body
flow past a sphere, stage note 10.8.4) is a model of the REFERENCE solution (Wu & Eatock Taylor 1988), derived
analytically, not an input of the solver. Numerical devices (stage note 10.8; each with its check):

| Device | Value | Derivation / check (registered tolerance) | Where |
|---|---|---|---|
| Hull-wall closure of d/dx (replaces the B5 wall rule) | wall node on the waterline (band index -1/2) in the columns where the fixed-y characteristics of the uniform stream leave the hull; Lagrange weights on {-1/2, 0, 1} and {-1/2, 0, 1, 2}; traces phi_w (hull solution extrapolated to z = 0) and zeta_w = -(1/g)(phi_t,w - U phi_x,w) with phi_x,w = (d phi_w/dx - w' Q phi_n)/(1 + w'^2) | exact body-condition relation on a wall-sided waterline (stage note 10.8.2); DXM (manufactured d/dx: order >= 1 on the wall line), SP (every mode decays), TN (b), GS/GSo, CU, KK. Not consistent at flared waterlines (term w' H_z phi_z omitted; the uniform-stream corner is singular there, 10.8.1) | unsteady.dx_operator(wall="trace"), wall_trace_data, ForwardSpeedLTI.set_operator |
| Validity band of a table (replaces the B5 zero-speed rule; review minor 8) | per node: omega_hi where the shortest along-track wave max(k0, k2, k4 [tau < 1/4]) spans 8 hull-region cells; omega_lo where every beach is at least one wavelength of the waves reaching it (lateral: k0, transverse waves; downstream: k1; upstream: k3 for tau < 1/4; amendment 1 of the review, stage note 10.8.11) | forward-speed dispersion along the track (omega -+ U k)^2 = g k (stage note 10.8.6); values outside are solutions of the same causal system but not resolution-verified; reported per node (PV, CD notes) | tables.validity_band, along_track_wavenumbers |
| Kernel window of the plant kernels | K from B_inf + W (B - B_inf), W = cos^2 from omega_t to 2 omega_t, omega_t where the zero-speed wavelength spans 4 hull-region cells; arrays A, B unmodified; KK distortion stored per node (taper_dA) | above omega_t even the zero-speed waves are unrepresentable on the grid; exploration B5R-E8: windows at the dispersion-aware omega_hi or at 4 cells of the k2 wave removed resolved damping, distorted A by 6-20 % and destabilised the plant law at one node, this window does not (distortion <= 1.4 %); check CD (free decay bounded at every node, both forms) | tables.kernel_window_start, memory_kernel(taper=...), node_slice |
| Plant force law | F = -A_inf a - B_inf v - int K v ds - C0 (q - qbar) + F_steady(U, qbar), one force form | the radiation solver's LTI system about the slice attitude (stage note 10.8.5); checks CU (Cummins through the law, both forms), C0B (C0 vs the steady-force slope, 5 % of the hydrostatic stiffness), CD | tables.RadiationForce(stiffness=True), free_decay |

## Stage B6 (2026-10-07): double-body linearisation (`hydro/rankine/doublebody.py`)

No empirical input. The base flow (rigid-lid double body), the linearised free-surface conditions, the m-terms, the
force forms, the Timman-Newman / Haskind / pseudo-energy identities and the steady wave problem are derived (stage note
11.2); g and rho are parameters (G4). Numerical devices and their checks:

| Device | Value | Derivation / check (registered tolerance) | Where |
|---|---|---|---|
| Kinematic operator | K_U = Wa^-1 A_{-U}^T Wa (area-weighted adjoint of the reversed-flow advective operator) | conservative form of div_h(W zeta) in the interior with upwind dissipation of the right sign; exact discrete reciprocity (FSR 1e-10); DXD (orders), SP, GS, TN | fs_operators, stage note 11.2 item 3 |
| Hull-wall closure of the advection | mirror (even) ghosts at the wall for both directions of the band grid | the wall is a characteristic boundary (u^j = 0 there); DXD measures the local consistency of A and K on the first lines | _adv_matrix, 11.2 item 3 |
| Surface derivatives of the base flow | three-point Lagrange formulas in chord length along the hull mesh lines (surface_gradient2) | MT (sphere, analytic flow, 1 %, order >= 1.5) | surface_gradient2, 11.2 item 6 |
| Discrete W.grad on the hull and m-terms | edge-flux form with zero flux through the waterline, keel and stem edges; m_k = -Wa^-1 G^T Wa n_k | SBP (Tuck exact, 1e-12), TK (D = OT, 1e-10), WE, GSp | wgrad_operator_fv, DoubleBodyFlow.mterms |
| Steady free-surface grid [**superseded by the B6 review (2026-10-07): extents scaled with the steady wavelength**, see below] | compact domain (up 1.0 L, down 2.5 L, lateral 1.5 L, beaches 0.5 / 1.5 / 0.75 L), M from >= 30 cells per transverse wavelength, 40 <= M <= 90 | as B4's n_per_lam = 30; steady validation rows SF (Fn 0.2 not run: dense-LU memory) | steady_grid |
| Kramers-Kronig entry skip | entries below 1e-3 x the band maximum of abs(A33) (B5: maximum over all frequencies) | needed because C0_33 != 0 in the double-body model (A33 ~ -C0_33/omega^2 at the low end) | studies/rankine_db_kk.py |

## B6 review (2026-10-07): numerical devices changed or added (still no empirical input)

No empirical input. The frozen base-pressure stiffness C_s, the static-stability criterion and the attitude handling of
pointed ends are derived (stage note 12.2 items 4-6). Numerical devices (each with its check):

| Device | Value | Derivation / check (registered tolerance) | Where |
|---|---|---|---|
| Steady free-surface domain (replaces the fixed B6 extents) | l = lambda0/L = 2 pi Fn^2: beaches b_up = max(0.5, l), b_dn = max(1.5, l), b_lat = max(0.75, l) L; extents up = max(1.0, 0.5 + b_up), down = max(2.5, 1.5 l + b_dn), lat = max(1.5, 0.75 + b_lat) L; unchanged for Fn <= 0.28 | every beach at least one steady wavelength, undamped zones of 0.5 L upstream and 0.75 L laterally, the wave cut 1 lambda0 behind the stern plus half a wavelength before the downstream beach; SK (deep spheroid vs the exact Kelvin solve, 3 %; B6 domain +12.8 %, scaled +1.0 %), SKd / SFd (extents x 1.5, information) | doublebody.steady_grid (stage note 12.2 item 7) |
| Finite-difference step of the base flow at free-surface points | h = min(1e-3 L, d/4), d = exact distance to the hull panels and their images, floor 1e-6 L; unchanged where d >= 4e-3 L | the fourth-order stencil of half-width 2 h stays at least d/2 from the panels (B6: 56 % error at d = 0.00125 L with the fixed h); affects only the M 80 grids and the steady Fn 0.25 grid (stage note 12.3, B6R-E8); DXb (information) | DoubleBodyFlow._fd_step |
| Chord floor of the surface calculus | chord increments >= 1e-9 x the longest increment of the line | guard for coincident centroids of collapsed panels; changes no registered mesh | doublebody._chord (CHORD_FLOOR) |
| Wetted range of the hull mesh at an attitude | stations over [x_lo, x_hi], the keel-emergence points zeta_w + D = 0 (bisection), the B4-B6 stations when both ends are wet | geometric (stage note 12.2 item 6); AT1 (no dry station, no inverted panel), AT2-AT5 | HullForm.wet_range, mesh, waterline, waterline_ends |
| Plant force law (since the B6 review) | F = -A_inf a - B_inf v - int K v ds - (C0 + C_s)(q - qbar) + F_steady(U, qbar); C_s = the frozen base-pressure stiffness (zero for the uniform-stream tables) | derived (stage note 12.2 item 4); CS (C_s two ways, 2 %), C0B (vs the steady slope, 5 % of the dynamic stiffness: expected FAIL, the two are different linear models), CD/CDw/CDs | tables.RadiationForce, free_decay, doublebody.frozen_pressure_stiffness |
| Hull above the design waterline | prismatic_v: the form itself (V to the chine, vertical above); Wigley models: vertical sides (assumption of B4, unchanged) | geometric: the prism's vertical extension from z_b = 0 created a knuckle at the waterline for any sinkage; CS (flared Wigley with its own form above the waterline) | HullForm(above=...) |


## Stage B7 (2026-10-07): planing feasibility gate (`hydro/rankine/planing_kernel.py`, `planing_steady.py`, `vlm_ref.py`)

**No empirical input in the formulation.** The linear steady planing theory with gravity (stage note 13.2) is derived:
the transfer function of the free surface to an applied pressure (Kelvin condition, radiation condition), the body
condition zeta = h on the wetted planform, the linearised spray-root (Wagner wetting) condition and the transom Kutta
condition. g and rho are parameters (G4, through C_V = U/sqrt(g b)). The chine treatment (pressure zero at the chine
edge in the production discretisation) is a discretisation of the same linear problem (its continuous limit is the
pressure jump of item C), not a model input. **Savitsky's (1964) equations are used only as the validation reference,
never inside the model** (V2). Nothing was tuned: the tolerances of 13.1 were written before the production scheme was
settled and no constant was changed after a result was seen. What the gate found missing (13.6) -- the deadrise
reduction of lift (nonlinear wedge entry) and chine-flow separation -- would, in the established planing methods, come
either from nonlinear theory or from empirical corrections (Savitsky's -0.0065 beta C_L0^0.6, Shuford's cross-flow drag
coefficient, Sun & Faltinsen's 0.5 b transom correction); none of them is used here.

Numerical devices and their registered checks (stage note 13.1, 13.3):

| Device | Value | Check (registered tolerance) | Where |
|---|---|---|---|
| Rayleigh damping and its removal | eps = b/l_mu, l_mu = clip(0.5 lambda_0, 20 b, 400 b) (50 b for the high-speed checks); first-order Richardson K = 2 K(2 l_mu) - K(l_mu) | PK4 Richardson pair (2 l_mu, 4 l_mu): C_L change 0.5 %; PK4i (information: without Richardson) | planing_kernel (KernelTables/YHatTables), planing_steady |
| Periodic x box and spectral folding | box 8 l_mu (images e^-8), 8 folds of k_x + 2 pi l / h | PK1 (brute-force 2-D FFT, 5e-3), PK4 box 12 l_mu, 16 folds (0.5 %) | x_transform_specs |
| k_y quadrature | uniform step min(0.02, k0/16) to 3/b, geometric ratio 1.005 to k_max = max(2e6, 20 C_V^2 (pi/h)^2)/b; Filon weights (Si, Ci) with a 6-point Gauss rule where k max abs(d) < 2 | PK2 (Filon vs accurate quadrature, 1e-6), PK4 k_max x 10 and ratio 1.0025 (0.5 %), exploration B7-E2/E4 | ky_nodes, filon_hat_weights, hat_weight_matrix |
| Pressure basis | x: hats on h = b/32, half-hat at the spray-root node, p = 0 at the transom node (Kutta); y: hats on 16 uniform half-beam intervals, pressure zero at the chine edge (scheme H-Y0) | PC1 grid series (2 % between the two finest), PC1u (U_num 5 %), PL1 (3 %), PL2 (10 %) | planing_steady.PrismaticPlaningHat / PlaningSprayRootFast |
| Collocation | slope condition at cell midpoints; level (spray-root) condition at the spray-root node of each lateral line | as above; exploration B7-E3 (nodal elevation collocation unstable) | assemble, _residual |
| Continuous spray-root position | first-order interpolation between lattice node sets, J from neighbour planforms (rank-one + border update), at most 2 cells per step, slack 0.15 cell | convergence flag per condition (validation rows count non-converged conditions as failures) | PlaningSprayRoot._iterate, PlaningSprayRootFast._bundle |
| Initial spray-root line | slender-body (Wagner-Tulin) keel-chine difference tan(beta)/(pi tan(tau)) | none needed (only a starting guess; the converged state satisfies the level conditions) | wagner_nodes |

## Stage B8 (2026-10-07/08): route 1 of the planing gate (`hydro/rankine/planing_wedge2d.py`, `planing_nl.py`)

The composite (stage note 14.2, eq. 14.1), the matching (14.2), the chine-dry section (modified Logvinovich model,
derived, eq. 14.3) and the steady free-streamline force Cs_inf (exact, Gurevich) contain no empirical input. One
element is a fit, registered here (G3); it is used as a model of the 2-D section, never as a speed-indexed weight (G1):

| # | Input | Value used | Source | Data range | Our cases inside? | Uncertainty / scatter used |
|---|---|---|---|---|---|---|
| E-B8-1 | Decay of the chine-wetted (transition-stage) section force of a wedge entering at constant speed, from its value at chine wetting to the steady supercavitating force | F = rho V^2 l [Cs_inf + C*(h*) (Cs_cw - Cs_inf)], h* = ((h - h_cw)/h_cw) cot(beta)^1.3075, C* = (1.539 h* + 2.618)/(h*^2 + 8.081 h* + 2.169) (1 for h* <= 0.067); anchor Cs_cw = MLM force at chine wetting (14.2) | Wen, Liu, Del Buono, Qu & Iafrati (2022) J. Fluids Struct., arXiv 2201.00959, eqs 16-22: a fit to FVM-VOF solutions (checked against the BEM-FEM of Iafrati & Battistin) of the gravity-free, inviscid nonlinear entry problem; transcription `data/external/wedge_entry/wen2022_transition_constants.csv` | deadrise 15-45 deg for the constants, compared with CFD 5-70 deg; h* in [0, 7] | deadrise 10, 20, 30 deg (10 deg outside the constants' range, inside the compared range); h/h0 at the transom 0.64-3 at the registered conditions (h* up to about 7-20 at 10 deg: beyond the fitted range there) | not a measurement scatter: the fit's deviation from its own CFD is a few per cent in the paper's figures (16-18); not varied in B8 (the gate's outcome does not hinge on it, stage note 14.5); sensitivity rows: anchor at the similarity value (anchor_wen), separated force removed (mlm_only) |

Rejected (not used): Logvinovich's zero-pressure separation condition in MLM form (Tassin, Korobkin & Cooker 2014) --
derived, but its transition-stage force is 0.21-0.30 of the CFD value (stage note 14.2, rows W2R). Not used either:
Sun & Faltinsen's 0.5 b transom correction, Garme's transom reduction function, Shuford's cross-flow drag coefficient,
Savitsky's deadrise term (validation reference only).
