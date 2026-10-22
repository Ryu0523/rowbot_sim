# WaveBEM port design (Track B, step B1) -- 2026-10-06

Scope: read the WaveBEM source (`C:\Users\Administrator\Documents\ext_sims\WaveBEM`, SISSA mathLab,
LGPL-2.1, ~42k lines C++ on deal.II 8.4 + OpenCASCADE + SUNDIALS IDA), record exactly what it
computes and how, and plan a Python re-implementation in a new package `hydro/rankine/` that gives
the any-speed plant (a) steady sinkage / trim / resistance maps and (b) forward-speed radiation and
diffraction (Cummins kernels) as computed tables. Acceptance rules: `learn/meta/ACCEPTANCE_2026-10-06.md`
(cited below as G1, M3, ... ). Licence rule L: algorithms are re-implemented from this description and
from the papers; no source is copied.

All file:line references are into `ext_sims/WaveBEM/` unless written otherwise. "W-" prefixes a
WaveBEM fact read from the code; "P-" prefixes a port decision.

---

## 中文摘要（给用户）

- **WaveBEM 是什么：** 一个三维、时域、带航速、自由面完全非线性的势流程序。它把船固定在"拖曳水池"里，让水以航速流过，一步步积分到稳态，算出兴波阻力、下沉和纵倾。
- **它能给我们的：** 一整套已经在排水型船上跑通的做法：边界积分方程的离散、奇异积分、自由面的运动方程（网格只上下动）、船体压力和受力、下沉和纵倾的耦合、船尾的干湿处理、数值消波区，以及随源码附带的 Wigley、DTMB 5415、NPL 船型试验数据和它自己的计算结果，可以拿来对照。
- **它不能给我们的：** 没有滑行（没有折角线处的流动分离、没有飞溅）；船尾"湿"的情况并不是真的去解湿船尾，而是用一个经验公式（Doctors 的回归式）估计船尾湿的程度，再在船尾后的水面加一块压力；横摇、艏摇的力矩没有接进运动方程；侧壁会反射波浪；每个算例要几小时。
- **搬运方案：** 不照搬它的时域非线性主程序（太慢），而是：
  1. 先写一个通用的边界积分核心，用显卡直接求解，代替它的快速多极算法；
  2. 稳态：先做线性化自由面（绕平静水面或绕双体流），再做完全非线性的迭代解，下沉纵倾自由，船尾干湿作为边界条件；
  3. 有航速的辐射和绕射：在稳态流动上线性化，直接得到记忆函数和频域系数，自动满足因果关系；
  4. 最后才把它的时域非线性方法作为"参考解"，用来测量"按当前姿态冻结记忆函数"这一近似的误差。
- **每一步都先过数学检查：** 球和椭球的精确解、半球的已知解、潜体兴波阻力、开尔文波系角度、零速对零速面元法、无重力极限、Haskind 和 Timman–Newman 关系、弗劳德相似和左右对称。
- **卡在哪里：** 计算本身很快（实测显卡上八千个面元一次分解约一秒）；真正慢的是验证和调试，尤其是非线性迭代的稳健性、高速时的波长很长要很大的计算域、以及滑行（折角线分离和飞溅）这一段，后者是研究性质的，能不能做成要到那一步才知道。

---

## 1. Source map (what is where)

| File | Lines | Role |
|---|---|---|
| `source/free_surface.cc` | 10670 | The whole physics: DAE residual and Jacobian (Sacado AD), free-surface (FS) conditions, hull BC, pressure, forces, rigid body, restarts, remeshing, output |
| `source/numerical_towing_tank.cc` | 4040 | Tank geometry, block-structured initial mesh around the hull, node flags, double nodes, CAD projections, mesh smoothing |
| `source/bem_problem.cc` | 1297 | Collocation BEM: assembly, alpha (solid angle), mixed Dirichlet/Neumann solve, edge constraints, surface gradients |
| `source/bem_fma.cc` + `multipole_expansion.cc`, `local_expansion*.cc`, `ass_leg_function.cc` (11290, generated), `octree_block.cc` | ~13.5k | Fast multipole (FMA) alternative to the direct solver |
| `source/boat_model.cc` | 1184 | IGES hull, mirroring, hydrostatics on the CAD surface, keel/transom curves, rigid placement |
| `source/dae_time_integrator.cc` | 417 | SUNDIALS IDA wrapper (variable-order BDF), restarts |
| `source/newton_solver.cc`, `restart_nonlinear_problem_*.cc` | ~640 | Newton solver for consistent restarts |
| `source/occ_*.cc`, `surface_smoothing.cc` | ~3.4k | OpenCASCADE projections, line and surface smoothing |
| `include/laplace_kernel.h` | 65 | G and grad G |
| `prm-files/default.prm` | | Reference parameter set (DTMB 5415 model "goteborg", Fr 0.28) |
| `experimental-results/` | | Wigley, DTMB 5415, NPL, KCS/KVLCC2 data + WaveBEM outputs (Section 6) |
| `tests/` | | Only a template (`tests/template.cc` prints 0): **no automated tests** |
| `scripts/tokyo-tests/` | | PBS batch for the Wigley Fr sweep (older prm format) |

Driver: `source/driver.cc:45-80` -- read domain, refine, `bem_problem.reinit`, `free_surface.initial_conditions`,
`prepare_restart`, then `dae_time_int.start_ode`. The computational domain object is a `NumericalTowingTank`
(`include/driver.h:46`).

---

## 2. Mathematical formulation as implemented

### 2.1 Frame, unknowns, state vector

- W-1 Frame: tank-fixed, hull nominally at rest, stream ("wind") `V_inf(t)` along +x ("aero-like": bow at -x,
  `boat_model.cc:305-311` comment). Total potential = `V_inf . x + phi`; `phi` is the perturbation.
  The speed ramp is in the prm (`Wind function 3d`, e.g. `t/1.5` ramp in `scripts/tokyo-tests/base.prm`).
- W-2 DAE state `y` (`free_surface.cc:281-285`): node positions (3 per node, as displacements from reference
  points, `map_points`), `phi` per node, `dphi/dn` per node, then 13 rigid-body unknowns: hull velocity (3),
  displacement (3), angular velocity (3), quaternion (4).
- W-3 Which components are differential (`free_surface.cc:8222-8330`): on interior FS nodes `z` and `phi`
  are differential; FS `x, y` are algebraic (mesh smoothing); on FS edge nodes `x, y` are algebraic;
  transom-edge and near-inflow FS nodes are fully algebraic; on all non-FS nodes `phi` or `dphi/dn` is
  algebraic (BEM), and the rigid-body unknowns are differential.

### 2.2 Boundary integral equation and Dirichlet/Neumann split

- W-4 Kernel (`include/laplace_kernel.h:45-62`): `G = 1/(4 pi r)`, `D = grad_y G = -(y - x_i)/(4 pi r^3)`.
- W-5 Collocated BIE at every node `x_i` (`bem_problem.cc:107-590`, `620-670`):

  `alpha_i phi_i + sum_j N_ij phi_j = sum_j D_ij (dphi/dn)_j`, with
  `N_ij = int D(y - x_i) . n(y) N_j(y) dS`, `D_ij = int G N_j dS` (deal.II naming: `neumann_matrix`,
  `dirichlet_matrix`).
- W-6 `alpha_i` (solid-angle term) by the rigid-mode identity, `alpha = -N . 1` (`bem_problem.cc:594-618`).
  This is exact only because the domain is a **closed** box (FS + hull + side walls + bottom + inflow +
  outflow, Section 3.5).
- W-7 Split (`numerical_towing_tank.cc:1799-1807`): FS nodes (`water`) and optional `pressure` nodes are
  Dirichlet (`phi` given, `dphi/dn` unknown); everything else (hull, walls, bottom, inflow, outflow) is
  Neumann (`dphi/dn` given, `phi` unknown). `vmult` builds the operator on the unknowns
  (`bem_problem.cc:620-670`), `compute_rhs` the known part (`673-811`). A pure-Neumann fix subtracts the
  vector norm (`bem_problem.cc:665-668`, a hack).
- W-8 Boundary data: hull `dphi/dn = n . (v_node - V_inf)`, imposed in weak (L2-projected) form with the
  element mass matrix (`free_surface.cc:6930-6967`); inflow `dphi/dn` = prescribed function (incoming waves,
  `6969-6997`); walls/bottom/outflow `dphi/dn = 0` (`6999-7025`).
- W-9 Edges and corners use **double (triple) nodes** -- coincident nodes, one per face
  (`double_nodes_set`). Constraints (`bem_problem.cc:990-1105`):
  - Dirichlet-Dirichlet on a flat edge: equal `dphi/dn` (`1048-1052`);
  - Dirichlet-Dirichlet on a sharp edge: both normal derivatives recovered from the two surface gradients,
    `dphi/dn_i = [grad_s phi_j . n_i + (grad_s phi_i . n_j)(n_i . n_j)] / (1 - (n_i . n_j)^2)` (`1056-1075`);
  - Dirichlet-Neumann (the waterline: FS node / hull node): equal `phi` (`1077-1081`);
  - Neumann-Neumann (keel, chine-like edges, wall corners): equal `phi` (`1090-1098`);
  - plus deal.II hanging-node constraints from adaptive refinement (`1005`).
- W-10 Surface gradients of `phi` at nodes: L2 projection of the element gradient onto a vector Q1 space,
  direct sparse solve (`bem_problem.cc:1153-1293`). The full nodal gradient is `grad_s phi + n dphi/dn`.

### 2.3 Elements, quadrature, singular and near-singular integration

- W-11 Elements: bilinear quadrilaterals (`FE_Q<2,3>(1)`), isoparametric through `MappingQEulerian`
  of degree 1 on the displacement field (`computational_domain.cc:57-59`, `numerical_towing_tank.cc:142-143`).
  Collocation at the element vertices (support points).
- W-12 Regular integrals: tensor Gauss rule, order from the prm (`Quadrature order = 4`, i.e. 4x4,
  `computational_domain.cc:125-129`; `bem_problem.cc:133-137, 243-259`).
- W-13 Singular integrals (cell contains the collocation node, or any of its double nodes,
  `bem_problem.cc:223-230`): deal.II `QGaussOneOverR` built once per vertex
  (`bem_problem.cc:116-120, 475-525`) -- a Duffy / Lachat-Watson type collapse of the square into
  triangles meeting at the singular vertex, exact for `f/R` with polynomial `f`; order 5 in `default.prm`.
  It handles both the `1/R` single layer and the `O(1/R)` double layer of a curved bilinear element.
- W-14 **Near-singular integrals: no special treatment.** Neighbouring cells use the regular 4x4 rule;
  accuracy relies on the mesh-anisotropy limit (`Max aspect ratio = 1.5`, `remove_mesh_anisotropy`,
  `numerical_towing_tank.cc:3610`).

### 2.4 Linear solver and fast multipole

- W-15 Direct mode (`Solution method = Direct`, the default): dense O(N^2) assembly of `N` and `D` at
  every new time value (`bem_problem.cc:969-987`), GMRES(100) (`826-827`) preconditioned by a banded
  sub-matrix (band 100) factorised with UMFPACK (`1107-1149`).
- W-16 FMA mode: octree (`Number of Octree Levels = 10`, one element per block,
  `computational_domain.cc:17, 312-900`), near field integrated directly (singular rule as above,
  `bem_fma.cc:56-660`), far field by spherical-harmonic multipole/local expansions of order 4
  (`bem_fma.cc:663-1197`); preconditioner = sparse near-field matrix, UMFPACK (`bem_fma.cc:311-320, 525-646`).
- W-17 Problem size in practice: `Maximum number of dofs = 4750` (`default.prm`), 5000 in the Wigley runs.

---

## 3. Free-surface treatment

### 3.1 Semi-Lagrangian (ALE) kinematic and dynamic conditions -- checked term by term

Notation: `u = V_inf + grad phi` (fluid velocity), `w` = node velocity, FS as `z = eta(x, y, t)`,
`grad_h` horizontal gradient, `Psi(x,y,t) = phi(x, y, eta, t)` the trace of `phi` on the FS.

- W-18 Kinematic (`free_surface.cc:6841-6842`):
  `d eta/dt |_node = phi_z + grad_h eta . (w - u)`.
  Check: for `z = eta`, `eta_t + u_h . grad_h eta = phi_z`, and a node moving with `w_h` sees
  `d eta/dt = eta_t + w_h . grad_h eta`; equal. (V_inf horizontal.)
- W-19 Dynamic (`free_surface.cc:6843-6845`):
  `d phi/dt |_node = |grad phi|^2/2 - g z + grad_h Psi . (w - u) - p_extra/rho`,
  where `grad_h Psi` is coded as the "corrected" surface gradient
  `(grad_s phi)_h - (grad_s phi)_z n_h / n_z` (`6683-6685`) and `p_extra` collects the breaking-wave pressure,
  the transom pressure patch and the beach (below).
  Check: Bernoulli for the perturbation with atmospheric pressure zero,
  `phi_t + V_inf . grad phi + |grad phi|^2/2 + g z + p/rho = 0`. With `Psi_t = phi_t + phi_z eta_t` and the
  kinematic condition, `dPsi/dt|_node = Psi_t + w_h . grad_h Psi` expands to
  `-V_inf . grad phi - |grad phi|^2/2 - g eta + phi_z eta_t + w_h . grad_h Psi - p/rho`; expanding the code's
  `grad_h Psi . (w - u)` with `grad_h Psi = grad_h phi + phi_z grad_h eta` and `u_h . grad_h eta = phi_z - eta_t`
  gives the same expression. **The code is exact Bernoulli + exact kinematics on a moving node.**
- W-20 Node motion: interior FS nodes move vertically with `z` from W-18 and horizontally by an
  algebraic Laplace-Beltrami smoothing of the `x, y` displacements, blended in after each remesh with
  `sin(pi (t - t_remesh)/T_remesh)` (`free_surface.cc:5392-5413, 7076-7077`). Option `lagrangian` is parsed
  (`258`) but not used anywhere: the scheme is always semi-Lagrangian/ALE.
- W-21 Weak form with SUPG (Brooks-Hughes) test functions on every FS element (`6850-6884`):
  `N_i + tau u_hat . grad_s N_i`, `tau = h_cell/sqrt(2)`, `u_hat = u/|u|` (`6855-6856`), applied consistently
  to the mass term and the right-hand side (`7073-7074`). Tiny `|u|`, `|grad eta|` are regularised by cubic
  polynomials below 1e-3 (`6696-6701`).

### 3.2 Time integration (DAE)

- W-22 Residual `F(t, y, y') = 0` assembled element by element with Sacado forward AD for the local
  Jacobian `dF/dy + alpha dF/dy'` (`free_surface.cc:5063-8072`; the BEM rows are `bem_phi - phi` or
  `bem_dphi_dn - dphi_dn`, `7974-8045`).
- W-23 SUNDIALS IDA, variable order up to 5 (`dae_time_integrator.cc:386`), SPGMR Krylov with
  user J.v (`jacobian`, `free_surface.cc:4917-5053`, which performs a full BEM solve for the dense block) and
  a preconditioner made of the sparse local Jacobian with the BEM rows replaced by identity
  (`setup_jacobian_prec`, `8150-8220`, UMFPACK `8104-8140`); algebraic variables excluded from the error test
  (`dae_time_integrator.cc:365`), max 10 nonlinear iterations (`370`), absolute tolerances from local cell
  diameters (`free_surface.cc:4750-4794`), relative tolerance from the prm (1e-2 in `default.prm`,
  1e-4 in the Wigley runs).
- W-24 "Keep BEM in sync with geometry = false" (default): BEM matrices are re-assembled once per new time
  value, frozen inside the Newton iterations (`free_surface.cc:6099-6109`); J.v never contains the
  derivative of the BEM operator with respect to node positions (inexact Newton).
- W-25 Restarts after every remesh: solution transfer (`SolutionTransfer`, `free_surface.cc:1245`),
  then a reduced Newton problem for consistent algebraic values (`prepare_restart`, `3331-4750`), then
  `IDACalcIC` (`dae_time_integrator.cc:396-401`).
- W-26 Initial condition: the BEM is solved with FS **and** hull as Neumann (FS `dphi/dn = -n . V_inf = 0`
  on the flat FS, i.e. a rigid lid) and far walls as `phi = 0` Dirichlet (`free_surface.cc:639-691`): the
  run starts from the double-body flow.

### 3.3 Far field, inflow, radiation, beach

- W-27 Closed tank 12 L x 4 L x 2 L (both sides of the hull, no symmetry plane;
  `numerical_towing_tank.cc:724-727, 1271-1274`). Lateral walls and bottom are reflecting (`dphi/dn = 0`);
  FS nodes at the inflow edge are clamped (`free_surface.cc:5332-5336`).
- W-28 Beach: only downstream, `x > 2 L`: an extra dynamic-condition pressure term
  `-((x - 2L)/(4L))^2 dphi/dn` (`free_surface.cc:6835-6845`). No damping upstream or laterally, and none in
  the kinematic condition. No explicit radiation condition: causality of the time march does it.
- W-29 Breaking: an ad hoc dissipative FS pressure switched on where
  `sqrt(eta^2 + |grad eta|^2 Fn^4) > 0.1725 Fn^2` (`free_surface.cc:6710-6753`), with constants 0.1725,
  0.117, 0.2, 1/12 and no cited source -- **empirical** (would need G3 registration).

### 3.4 Mesh adaptivity

- W-30 Every `Remeshing period`: Kelly error estimator on the FS elevation (positions-dot before t=0),
  normalised separately on hull and elsewhere, fixed-number refine/coarsen up to the dof cap, solution transfer,
  refinement floor `adaptive_ref_limit` x initial minimum diameter (`free_surface.cc:1100-1460`, refinement executed at `1450`);
  hull mesh re-projected/smoothed onto the CAD surface (`numerical_towing_tank.cc:38-79, 2572-3360`).

---

## 4. Transom, bow, stern, waterline, spray

- W-31 **Geometry is always "dry transom".** The hull mesh ends at the transom edge (boundary ids 32/37 on
  the hull side, 40/41 on the water side); the transom face is never part of the BEM boundary. FS nodes on
  the edge carry the flag `transom_on_water` (`numerical_towing_tank.cc:2230-2260`).
- W-32 Edge treatment: transom-edge FS nodes move rigidly with the hull (positions,
  `free_surface.cc:5372-5389`), are made **Neumann** in the BEM with `dphi/dn` = mean of their hull twins
  (`6043-6065`), and their `phi` comes from the BEM (algebraic, `8285-8290`). Their elevation is therefore
  pinned to the transom bottom; **no dynamic condition (p = 0) and no smooth-detachment (Kutta) condition
  is imposed at the edge**; the FS downstream evolves freely.
- W-33 Wet versus dry is decided by an empirical regression, quoted in the code as "Doctors regression
  formulas" (`free_surface.cc:6755-6756`):
  `eta_dry = min(1, 0.05 Fn_T^2.834 (B_T/T_T)^0.1352 Re_T^0.01338)`, `Fn_T = |V_inf|/sqrt(g T_T)`,
  `Re_T = sqrt(g T_T^3)/nu`, `nu = 1.307e-6` (`free_surface.cc:6538-6584`, also `456-469`, `3827-3846`,
  `9986-9996`), evaluated on the instantaneous transom draught/beam. Most likely Doctors, Macfarlane &
  Young (2007), Int. Shipbuilding Progress 54:145-163 (to be confirmed when registered, G3). The transom is
  "dry" for `eta_dry = 1`.
- W-34 The wet part is modelled, not solved:
  - FS pressure patch behind the transom, `p = (1 - eta_dry) rho g |z_init|`, added to the dynamic
    condition (`free_surface.cc:6809, 6843-6845`), where `z_init` is a parabolic "hollow" starting at the
    transom bottom with the hull's buttock slope and reaching `z = 0` after `lh = 5` transom draughts
    (`466-575`, refreshed at restarts `3823-3960`). `lh = 5` is a constant; the Doctors hollow-length
    regression is present but commented out (`free_surface.cc:466-467`).
  - A hydrostatic force on a virtual transom face of height `(1 - eta_dry) T_T` is **subtracted in
    post-processing only** (`free_surface.cc:9979-10106`, total force line `~10140`): it is not in the
    rigid-body residual (`7628-7720`), so sinkage and trim do not feel it.
  - Physical criterion: none beyond the regression. (Potential theory admits both states, see
    `ANYSPEED_PRINCIPLES` section 6.)
- W-35 Waterline (contact line): the FS edge node on the hull is the master. Its `z` follows the FS
  kinematic condition; its horizontal coordinate (y if the hull normal is mostly lateral, x otherwise) is
  the projection onto the hull surface at that height (`free_surface.cc:5415-5713`); the hull twin nodes are
  hooked to it (`5907-6002`). Interior hull nodes move rigidly between remeshes and are re-smoothed on the
  CAD surface at remeshes. This is a **body-exact wetted surface** (below the instantaneous waterline).
- W-36 Bow and stern points slide along the stem/keel curve (or the transom side curves) by intersecting a
  horizontal plane at the node height with the curve (`free_surface.cc:5715-5905`).
- W-37 Spray: none. The FS is single-valued `z = eta(x, y)`; no overturning, no jets, no chine separation.
  Steep crests are damped by W-29.

---

## 5. Body motion and forces

- W-38 Mass and CG: mass = hydrostatic buoyancy of the CAD hull in its file position (or after a secant
  sinkage search for a given displacement, `boat_model.cc:266-292, 1154-1178`); CG placed at the
  longitudinal centre of buoyancy **at z = 0 (the waterplane)** ("we POSTULATE", `boat_model.cc:294-299`).
  Hydrostatics integrate `rho g max(z0 - z, 0) n` over each CAD face with a 300x300 Gauss rule
  (`boat_model.cc:594-833`), i.e. a kinked integrand, not exact.
- W-39 Hull pressure, full Bernoulli (`free_surface.cc:6935-6948`):
  `p = -rho [ phi_t + g z + grad phi . (grad phi/2 + V_inf) ]`, with the fixed-point time derivative
  `phi_t = phi_dot_node - w . grad phi` (`6935`). On the hull `phi` is an *algebraic* DAE unknown, so
  `phi_dot_node` is IDA's BDF derivative of the algebraic history -- not error-controlled (W-23) -- rather
  than the solution of an acceleration-potential problem.
- W-40 Force and moment about the moving CG integrated with the AD types, so their Jacobian enters the DAE
  (`free_surface.cc:6942-6948, 7290-7470`).
- W-41 Rigid body (`free_surface.cc:7628-7860`): `m v_dot = F + m g` (each translation can instead be
  imposed by a function; `default.prm` imposes x and y, leaves z free); rotation by quaternion with
  `I_abs omega_dot + omega x I_abs omega = M`, **but only the pitch moment is applied**
  (`7807`; the roll and yaw moment rows of the force Jacobian are commented out, `7377-7400` and `7424-7448`). Two coding errors that are
  harmless in pure pitch: `rhs_quat_vect` uses `WMatRow1` for the z component (`7774`) and the gyroscopic
  term uses `sservMatRow3` for the y component (`7787`). So in practice: **free sinkage and trim**, captive
  otherwise.
- W-42 Reported total resistance (`free_surface.cc:10119-10146`): hull pressure force - transom patch force
  + ITTC-1957 friction `0.075/(log10 Re - 2)^2` on the wetted surface (post-processing) + breaking-pressure
  force.
- W-43 Hard-coded constants: `g = 9.81`, `rho = 1025.1`, `nu = 1.307e-6`, a wave test case
  (`k = 0.62994`, `omega = 2.4835`, `h = 5.5`) inside the residual (`free_surface.cc:6214, 6239, 6253-6283`)
  and a different amplitude in the post-processing check (`10148-10180`).

---

## 6. Validation material shipped with WaveBEM

Files under `experimental-results/` (no automated tests; `tests/` is a template). Numbers below were
recomputed in this session from the shipped logs with the formula of the shipped Matlab scripts.

| Case | Data | What it shows |
|---|---|---|
| Wigley, L = 2.5 m, Fr 0.250-0.408 (`wigley/`) | Experiments: Tokyo cooperative tests (Kajitani et al. 1983), waterline profiles `tokyoFr*.csv`, wave-resistance coefficient from SRI, UT1-3 (`wigleyDrag*.csv`). WaveBEM: `test_wigley_Fr*_force.txt`, `*_water_line.txt` | Recomputed WaveBEM wave-resistance coefficient (mean of last 31 samples of column 2 / (0.5 rho V^2 S_w), `wigley.m`): Fr 0.250: 1.01e-3; 0.267: 0.99e-3; 0.289: 1.52e-3; 0.316: 1.70e-3; 0.354: 1.58e-3; 0.408: 2.43e-3. Experiments at the same Fr are about 0.83-1.0e-3, 1.25-1.36e-3, 1.42-1.47e-3, 1.33-1.42e-3, 1.53-1.63e-3, 1.99-2.29e-3: WaveBEM is within about 0-25 % and high at the hump. The logs are 20-column files from an older code version with an older prm (no hull-motion keys) and `Fz` falling from 430 to 375 N: **captive (fixed) runs**, so no sinkage/trim check |
| DTMB 5415 (INSEAN 2340), Fr 0.28 (`dtmb-5415/`) | Olivieri et al. (2001) wave cuts at y/L = 0.082, 0.172, 0.259, 0.347; Gothenburg 2010 waterline data; WaveBEM wave-cut exports `waveElevationY*.csv`, waterline file | Wave pattern of a transom-stern combatant at the design speed (dry transom), sinkage/trim free per `default.prm` |
| NPL round-bilge series parent (Bailey 1976; Marwood & Bailey 1969), full scale L = 30.5 m, Fr 0.2-0.4 (`npl/`) | Curve `npl_exp_drag.csv` (coefficient vs speed-length ratio), WaveBEM force logs, `npl_forces.m` | Recomputed with the script's formula (total = pressure - transom + friction + breaking, mean over 60-90 s): error vs the curve +1.1 %, +0.8 %, -1.6 %, -4.2 %, -24.1 % at Fr 0.20, 0.25, 0.30, 0.35, 0.40. But the transom correction is larger than the total at low Fr (6.7 kN against 4.4 kN at Fr 0.2), the file names ("damped_less") indicate retuned damping, and whether the curve is C_T or C_R must be resolved from Bailey (1976) before any use (V3). **Not admissible as validation of the transom physics as is** |
| KCS, KVLCC2 (`kriso/`) | Waterline profiles | Low-Fr displacement ships |

---

## 7. Known limits and defects (relevant to an any-speed planing USV)

1. **Speed range validated:** displacement hulls, Fr_L about 0.2-0.45. No planing, no semi-planing case.
2. **No chine separation, no spray** (W-37). Hull mesh topology is fixed (block-structured from keypoints,
   `numerical_towing_tank.cc:582-1790`), so large changes of wetted length (planing) are out of reach.
3. **Transom:** the wet state is an empirical patch, not a solution (W-33, W-34); its force does not enter
   the equilibrium; `lh = 5` is a fixed constant; no p = 0 or Kutta condition at the edge (W-32).
4. **Motions:** only sinkage and trim are free (W-41); roll, yaw, sway, surge not usable; the quaternion code
   has latent errors outside pure pitch.
5. **Radiation/diffraction:** not a frequency-domain or linear tool; low-speed unsteady problems
   (`tau = U omega/g < 1/4`, waves upstream) are not supported by the downstream-only beach and the
   reflecting side walls (W-27, W-28): it is a towing tank, not open water.
6. **`phi_t` on the hull** from BDF differences of an algebraic unknown (W-39): unverified accuracy.
7. **Near-singular quadrature** uncontrolled (W-14).
8. **CG at the waterplane** (W-38) changes the trim moment of the resistance and the pitch inertia coupling.
9. **Empirical pieces:** breaking pressure (W-29), Doctors regression (W-33), ITTC-57 (W-42).
10. **Cost:** IDA with tolerances 1e-2 (default) and remeshing; the shipped Wigley logs cover 18-50 s of
    model time; NPL 120 s. Hours per case on a workstation (the authors' runs used a PBS cluster).
11. **Code maturity:** hard-coded constants and test cases inside the physics (W-43), dead code
    (`compute_DXDt_and_DphiDt` `8892-9270` is never called), debug output on every residual.

**Consequence for the port:** WaveBEM's value for us is (i) a working and checked formulation of the
nonlinear FS on moving nodes (W-18, W-19 verified above), (ii) the body-exact waterline and transom-edge
mechanics (W-32, W-35, W-36), (iii) the force/rigid-body coupling (W-39 to W-41), and (iv) the shipped data
and results. Its time-domain DAE machinery is far too slow to produce tables over 30+ speeds and
attitudes, and its transom-wet and breaking models are empirical patches that must be registered (G3),
not adopted silently.

---

## 8. Port plan

### 8.1 What each deliverable needs

**(a) Steady sinkage, trim, resistance at every Froude number, transom wet or dry**

Needed: Laplace BIE on hull + FS (W-4 to W-13); steady FS conditions = W-18/W-19 with `d/dt = 0`
(linearised first); body-exact wetted surface with a moving contact line (W-35, W-36); dry-transom edge
condition (improving W-32 with the p = 0 edge condition, P-9); wet-transom model as a registered boundary
condition (W-33/W-34, fixed so that the face force enters the equilibrium); free sinkage and trim (W-41
reduced to two equations solved by Newton); pressure integration (W-39 with `phi_t = 0`); exact
hydrostatics. Not needed: FMA, IDA, adaptivity, breaking pressure, OCC.

**(b) Forward-speed radiation and diffraction (Cummins kernels)**

Needed: the same BIE core; FS conditions linearised about the steady basis flow from (a) (uniform stream
first); body condition with the m-terms (Ogilvie & Tuck 1969); a beach on **all** sides (W-28
generalised); an exact-in-time or stable implicit integrator; impulse responses giving
`A_inf, B_inf, C_inf, K(t)` per mode pair and diffraction transfer functions per (U, heading, omega_e).
WaveBEM itself has no linear radiation solver; only its building blocks carry over.

### 8.2 Mapping WaveBEM pieces to `hydro/rankine/`

| WaveBEM piece (ref) | Port module | Decision |
|---|---|---|
| G, grad G (W-4) + images | `kernels.py` | Port; add y-symmetry images (exact port/starboard symmetry, half the unknowns) and z-images (rigid lid `phi_z = 0` for the Fn -> 0 limit; `phi = 0` for the g -> 0 limit) |
| Q1 isoparametric elements, node collocation (W-11) | `panels.py` | **Simplify for production:** constant-strength panels (flat or bilinear geometry), centroid collocation, so `alpha = 1/2` exactly and no double nodes are needed at the waterline, chine, keel and transom. Keep a Q1 node-collocation variant for the nonlinear time-domain reference (8.3 C) |
| Gauss + `QGaussOneOverR` (W-12, W-13) | `quadrature.py` | Port the idea: Duffy collapse at a vertex for curved panels; for flat panels use the closed forms of Newman (1986) / Hess & Smith (1964); **add** near-singular control (adaptive subdivision or Telles 1987) driven by distance/size, which WaveBEM lacks (W-14) |
| Dense assembly + GMRES/FMA (W-15, W-16) | `solve.py` | Replace by dense assembly on the GPU (torch, batched) and LU (fp64, or fp32 LU + fp64 iterative refinement), re-used across right-hand sides and time steps |
| Double-node edge constraints (W-9) | Q1 variant only | Port only for the reference solver |
| Surface gradient by L2 projection (W-10) | `fields.py` | Port (also for constant panels via least-squares on neighbours); used for pressure |
| Semi-Lagrangian FS equations (W-18, W-19) | `freesurface.py` | Port the equations (checked exact above); steady form by setting time derivatives to zero; linearised forms derived from them |
| SUPG (W-21) | Q1 variant | Port for the reference solver; production FS operators are upwind finite differences on a structured FS grid (Dawson 1977; Raven 1996) or raised-panel collocation shift (Jensen et al. 1986) |
| Contact line and bow/stern sliding (W-35, W-36) | `waterline.py` | Port the logic; hull curves from our offsets (`hydro/hull.py`, `hydro/cad_import.py`, read-only use), no OCC |
| Transom edge (W-31, W-32) | `transom.py` | Port the dry geometry; add the edge conditions of P-9 |
| Doctors wet-transom patch (W-33, W-34) | `transom.py` | Port as a **registered empirical boundary condition** (G3), with its full scatter range; feed the face force into the equilibrium; hollow length as a registered input, not `lh = 5` |
| Bernoulli pressure, `phi_t` by BDF (W-39) | `forces.py` | Port Bernoulli; **replace** BDF `phi_t` by the acceleration-potential BVP (same matrix, extra right-hand side; Tanizawa 1995) -- cheap with LU re-use and checkable |
| Rigid body (W-38, W-41) | `equilibrium.py` | Steady: 2x2 Newton on heave force and pitch moment with the exact hydrostatic stiffness; CG from the hull data (G4), not at the waterplane |
| Beach (W-28) | `freesurface.py` | Port the form, applied in both FS conditions and on all sides; length and strength scaled by the local wavelength, with a sensitivity check |
| Breaking pressure (W-29) | -- | **Do not port.** Detect breaking (steepness/curvature) and flag the table point instead |
| IDA DAE (W-22 to W-25) | `nonlinear_td.py` (phase C) | Own implicit BDF2 with Newton-Krylov (scipy) only for the reference solver |
| Kelly adaptivity, OCC smoothing (W-30) | -- | Drop; structured graded meshes plus mesh-convergence studies (M5) |
| CAD hydrostatics (W-38) | `hydrostatics.py` | Replace by exact polygon clipping at the waterplane + divergence theorem (machine-precision equilibrium, M3) |
| Validation data and WaveBEM results (Section 6) | `studies/rankine_*.py` | Use as references (Section 8.4) |

### 8.3 Formulation (port decisions)

Frame for the port: hull-fixed, x forward, y to port, z up; stream `-U x_hat` (U < 0 = astern); the
mapping to WaveBEM's frame is `x_W = -x`.

- **P-1 One BIE core** (`kernels.py`, `panels.py`, `quadrature.py`, `solve.py`): Rankine sources (and dipoles)
  on hull + truncated FS; deep water (finite depth by a bottom image); half-domain with y-images, symmetric
  and antisymmetric parts solved separately, so port/starboard symmetry is exact by construction (M3).
- **P-2 Linear FS operator as a linear time-invariant (LTI) system.** About a basis flow `W` on a mean
  surface (uniform stream = Neumann-Kelvin; double-body = Dawson; or the nonlinear steady flow of P-6),
  the FS unknowns `y = [zeta, phi_FS]` obey `y' = M y + b u(t)`, outputs `F = C y + D u + E u'`, where the
  dense block of `M` is the Dirichlet-to-Neumann map `phi_FS -> phi_z` built once from the BIE (one LU,
  N_FS right-hand sides). Linearising W-18/W-19 about the basis flow gives the convective terms
  `-(W . grad) zeta`, `-(W . grad) phi` and the m-type source terms (Nakos & Sclavounos 1990;
  Kring 1994). Beach terms `-nu(x, y) zeta`, `-nu(x, y) phi` on every side.
- **P-3 Steady linear solution = one linear solve** `M y = -f`. Justification: with beaches on all sides
  every discrete mode is damped (checked by computing the spectrum of `M`), so `M` is nonsingular and the
  limit of the time march (the causal answer) is the unique solution of the steady equations. Checked
  numerically against an explicit time march (P-13 test S3).
- **P-4 Radiation and diffraction from the same `M`** (deliverable b): impulse responses by time stepping
  with LU re-use (or by eigen-decomposition of `M`), frequency responses `(i omega I - M)^-1 b`. Causality,
  hence Kramers-Kronig, holds by construction; `A_inf` is the `phi = 0`-on-FS problem (instantaneous
  response). The Cummins form with forward speed
  `F = -A_inf x'' - B_inf x' - C_inf x - int_0^t K(t - s) x'(s) ds`
  (Cummins 1962; Ogilvie 1964; Kring 1994) is tabulated per speed (and per trim/sinkage, and per transom
  state where both exist).
- **P-5 Reduced state-space for the plant**, computed from `(M, b, C)` by Krylov / balanced truncation
  with a certified error bound (Hankel singular values), not by fitting `K(t)` -- matches the user's
  "computed, general, not fitted" rule and the E1 budget (about 60-100 states).
- **P-6 Nonlinear steady solution** (deliverable a, body-exact): fixed-point/Newton on the FS shape,
  wetted surface and attitude. Each iteration linearises W-18/W-19 (with `d/dt = 0`) about the current
  iterate and solves P-3 (Jensen, Mi & Söding 1986; Raven 1996). The hull is re-panelled below the current
  contact line (W-35 logic) each iteration. Raised (desingularised) FS sources with collocation-point shift
  are the fallback if the upwind-difference operator proves too dispersive.
- **P-7 Free sinkage and trim:** outer 2x2 Newton with the exact hydrostatic stiffness plus the
  hydrodynamic force derivative from the previous iterate; tolerance 1e-6 of weight.
- **P-8 Captive grid:** besides the free equilibrium at each speed, the steady hydrodynamic force/moment on
  a small (sinkage, trim) grid per speed, because the plant needs steady forces off equilibrium; at run
  time hydrostatics and Froude-Krylov are integrated exactly on the instantaneous wetted surface
  (ANYSPEED_PRINCIPLES section 5) and only the hydrodynamic remainder comes from the table.
- **P-9 Dry-transom edge (G2, derived condition):** FS attached to the transom edge,
  `zeta = z_T(y)` there, and atmospheric pressure at the edge from Bernoulli,
  `|grad Phi|^2 = U^2 - 2 g z_T`, which is what a free surface leaving the edge must satisfy (Raven 1996,
  transom treatment); WaveBEM imposes only the first (W-32). Whether a steady dry solution exists is part
  of the answer: below a transom Froude number the dry free surface steepens to breaking
  (Vanden-Broeck 1980 for the 2D stern wave), so non-convergence or the breaking-steepness criterion
  marks the lower end of the dry branch.
- **P-10 Wet / partly ventilated transom (G2 + G3):** the WaveBEM model (dry geometry + dead-water pressure
  patch + face force for the wetted height `(1 - eta_dry) T_T`, W-34) is the only one available in potential
  theory; it is adopted as a **boundary condition** whose single empirical input, `eta_dry`, is
  registered with its source, data range and scatter, and randomised over that scatter (G3). Both branches
  are tabulated wherever both exist; the plant picks the branch from its transom state variable, never by
  weighting forces.
- **P-11 Astern (U < 0):** same solver, flow reversed; transom face meshed and wet (it leads); beach on the
  other end; no edge condition at the transom.
- **P-12 Planing extension (research, deliverable a at high speed):** when the computed contact line reaches
  the chine, it is pinned there and the FS leaves the chine edge with the P-9 edge conditions -- a real
  boundary-condition change decided by the solution (G2). Spray ahead of the spray root is truncated
  (Savander, Scorpio & Taylor 2002; Lai & Troesch 1995 -- citations to be checked before use). This is not
  in WaveBEM and is the riskiest item (decision point D2 in 8.8).
- **P-13 Low-Fn end:** the steady BVP's own limit Fn -> 0 is the rigid-lid (double-body) problem
  (`Fn^2 phi_xx + phi_z = 0` -> `phi_z = 0`), solved exactly by z-images; the resolved lowest Fn is set by
  about 12 panels per transverse wavelength `2 pi Fn^2 L` (about 3 kn for L = 5.4 m with N ~ 8000). Table
  rows between Fn = 0 and the lowest resolved Fn interpolate solutions of the same equations (G1), with the
  interpolation error reported. High-Fn end: wavelengths reach about 50 L at 40 kn, so the FS mesh is
  graded with distance and the beach is sized by the local wavelength; beach-sensitivity check required.
- **P-14 Reference solver (phase C):** the WaveBEM formulation itself (Q1 nodes, SUPG, semi-Lagrangian,
  implicit BDF, `phi_t` BVP, free sinkage/trim), for a few points, to (i) cross-check P-6 and (ii) measure
  the error of "kernel frozen at the current attitude" (ANYSPEED_PRINCIPLES section 5, step 3).

### 8.4 Order of implementation and checks

Every threshold below is a **proposal registered before the run** (V1); thresholds quoted from the
acceptance file are binding. Quick tier (T1) = runs at coarse meshes, <= 10 min total; full tier = mesh
series and sweeps, <= 1 h (GPU serialised).

**Step R1 -- BIE core** (`kernels`, `panels`, `quadrature`, `solve`; checks `studies/rankine_core_check.py`)

| Check | Reference | Threshold |
|---|---|---|
| Panel integrals vs high-order adaptive quadrature, near and far (M2) | own | rel. 1e-8 |
| Green identity on a closed sphere for `phi = 1, x, x^2 - y^2` (M3) | exact | residual -> 0 at measured order (M5) |
| Translating sphere, unbounded: potential and added mass `rho V/2` | exact | < 0.5 % at finest mesh, observed order >= 1.8 |
| Prolate spheroid (a/b = 5): surge, heave, pitch added mass | Lamb, closed form | < 1 % |
| Hemisphere heave, `phi = 0` on z = 0 (g -> 0 limit): `A33 = rho V/2` | exact (translating-sphere image), as in `hydro/verify_sphere.py` | < 1 % |
| Hemisphere heave, rigid lid (Fn -> 0, omega -> 0): `A33 = 0.8310 rho V` | Hulme 1982 | < 1 % |
| Port/starboard symmetry (M3) | by construction | exact (bitwise for mirrored inputs) |
| Froude/geometric similarity, scale 1.5 (M3) | exact | 1e-4 (expected ~1e-12) |

**Step S1 -- steady linear FS, captive** (`freesurface`, `steady`; `studies/rankine_steady_check.py`)

| Check | Reference | Threshold |
|---|---|---|
| Submerged prolate spheroid, Neumann-Kelvin wave resistance vs Fn | Farell 1973; Havelock 1931; Doctors & Beck 1987; independent Kelvin-source quadrature (Newman 1987 / Telste & Noblesse 1986) | < 5 % where >= 12 panels per wavelength |
| Kelvin pattern: half-angle and transverse wavelength from the computed `zeta` | 19.47 deg; `2 pi U^2/g` | +-1 deg; +-2 % |
| No upstream waves (radiation condition) | theory | upstream `zeta` < 1 % of max |
| Fn -> 0 tends to the double-body solution (M4) | z-image solve | forces converge to it as Fn decreases |
| Thin-ship limit B/L -> 0 tends to Michell | `hydro/kelvin.py` integrator generalised | difference -> 0 with B/L (limit check, not a ceiling) |
| Steady direct solve (P-3) equals time-march limit | own | rel. 1e-6 |
| Beach sensitivity: start and length x2 | own | forces change < 1 % |
| U = 0: hydrostatic equilibrium (M3) | exact polygon integration | 1e-9 W |

**Step S2 -- free sinkage/trim, transom branches** (`equilibrium`, `transom`)

| Check | Reference | Threshold |
|---|---|---|
| Wigley (free model) Cw, sinkage, trim, Fr 0.15-0.45 | Kajitani et al. 1983 (data to acquire, V3); WaveBEM Cw (Section 6, captive) | proposed: Cw +-20 % vs exp (linear FS), sinkage +-15 %, trim +-0.05 deg |
| Same vs WaveBEM's own results (same physics, independent code) | Section 6 numbers, waterline files | Cw +-15 %, waterline profile RMS < 15 % of max [B4 review: not registered in B4 and now recorded as DROPPED for the linear solver -- WaveBEM's `test_wigley_Fr*` results come from its fully nonlinear unsteady model (different physics, undocumented force columns); kept as a code-to-code comparison for S3 (stage note 9.7.9)] |
| Dry-transom existence limit vs Fn_T | P-9; Vanden-Broeck 1980 | reported, not tuned |

**Step S3 -- nonlinear steady** (P-6, P-7, P-9, P-10)

| Check | Reference | Threshold (proposed) |
|---|---|---|
| Wigley free model, Fr 0.25-0.41 | Kajitani 1983 | Cw +-15 %, sinkage +-10 %, trim +-0.05 deg |
| DTMB 5415, Fr 0.28 (dry transom): sinkage, trim, resistance, 4 wave cuts | Olivieri et al. 2001 (files in `dtmb-5415/`); WaveBEM exports | sink/trim +-10 %; wave cuts RMS < 20 % of max |
| NPL 100A Fr 0.2-0.5 (transom wet -> dry in our transition band) | Bailey 1976 (resolve C_T vs C_R first; alternative: Southampton series, Molland et al. 1994, Ship Science Report 71) | resistance +-15 %, trim +-0.3 deg; Doctors scatter propagated |
| Mesh convergence (M5) | own | observed order reported |

**Step R2 -- linear radiation, uniform-stream basis** (P-2, P-4; `unsteady_linear.py`;
`studies/rankine_radiation_check.py`)

| Check | Reference | Threshold |
|---|---|---|
| U -> 0 vs zero-speed BEM, A, B, excitation (M4) | `hydro/bem.py` (Capytaine 3.0.0) | 5 % (binding) |
| Hemisphere, Hulme 1982 A33, B33 over ka | Hulme | 3 % |
| High frequency `A -> A_inf` (M4) | P-4 | 1 % |
| g -> 0 vs the zero-gravity image solve (M4) | `phi = 0` image | 5 % (binding) |
| Haskind, U = 0 and U > 0 (reversed-flow radiation, Newman 1965) (M3) | identity | 3 % (binding) |
| Timman-Newman `A_jk(U) = A_kj(-U)`, `B_jk(U) = B_kj(-U)` (M3) | Timman & Newman 1962 | proposed 3 % |
| Kramers-Kronig between A and B (M3) | Ogilvie 1964; `hydro/retardation.py` | proposed 2 % |
| Spectrum of `M`: no eigenvalue with Re > 0; U = 0 without beach: discrete energy drift (M3, E2) | own | Re(lambda) <= 0; drift <= 1e-3 over 200 steps |
| Wigley forward-speed heave/pitch A, B, Fn 0.2-0.4 (V) | Journée 1992 (TU Delft report 909; to acquire, V3) | proposed +-20 % diagonal |

**Step R3 -- nonlinear basis flow and tables**: m-terms from the S3 steady flow (with care at sharp chines
and transom; compare against the uniform-stream basis), diffraction transfer functions
`X(omega_e, beta; U)`; table production for `scarab195` over U in [-3, 40] kn (denser in 6-18 kn), both
transom branches where both exist; reduced state-space (P-5) with its error bound; versioned cache
(N3, Section 8.5).

**Step S4 -- planing extension** (P-12): Fridsma 1969 and Savitsky 1964 prismatic hulls (V1 tolerances:
resistance +-10 %, trim +-1 deg); stop and report if it fails (V2).

**Step C -- WaveBEM-faithful nonlinear time-domain reference** (P-14): Wigley and DTMB 5415 against WaveBEM's
own results, then our hull at a few speeds for the frozen-kernel error.

### 8.5 Outputs and interfaces

- `hydro/rankine/tables.py`:
  `steady_table(hull, speeds, sinks, trims, transom_branch) -> npz` (forces/moments split into hydrostatic,
  hydrodynamic, transom face; wetted surface; contact-line and transom state; flags for breaking/non-convergence);
  `radiation_table(hull, speeds, ...) -> {A_inf, B_inf, C_inf, t, K(t), A(omega), B(omega), ss_model}`;
  `excitation_table(hull, speeds, headings, omega_e) -> X`.
  The plant combines `X` with our linear component sea directly (one complex factor per wave component at
  its encounter frequency; no non-causal excitation kernel needed).
- Cache: `studies/_cache/rankine/<hull>@<physics>/<content-hash>.npz` with mesh parameters, solver git hash
  and the empirical-input values used (N3). Interpolation in U of computed tables is interpolation of one
  equation set's solutions (G1); its error is measured by leave-one-out speeds (M5).
- The as1 plant (Track A files: `sim/planing_anyspeed.py` etc.) is not touched; it reads the tables when
  Track A wires them in.

### 8.6 Efficiency (measured this session) and test design

GPU probe (`torch 2.14.0+cu130`, RTX 3080 Ti Laptop 16 GB; scratch script `gpu_bench.py`):

| N | dtype | assembly (G and dG/dn, 4 points/panel) | LU | solve, 8 rhs | peak VRAM |
|---|---|---|---|---|---|
| 4000 | fp64 | 0.08 s | 0.17 s | 4.4 ms | 1.3 GB |
| 8000 | fp32 | 0.12 s | 0.08 s | 6.2 ms | 1.6 GB |
| 8000 | fp64 | 0.27 s | 1.07 s | 9.5 ms | 3.1 GB |
| 8000 | complex128 | -- | 4.05 s | 26 ms | -- |

(The N = 2000 rows of the probe include CUDA warm-up and are not quoted.) Consequences:
- a steady linear solve at N ~ 6000-8000 costs about 1-2 s; a nonlinear steady point about 15 iterations
  x 2 s; a free-equilibrium point under a minute;
- building the DtN map for N_FS ~ 5000 costs one LU plus 5000 triangular solves (about 6 s); an impulse
  response of 3000 steps with a dense 2 N_FS operator about 3-5 s per mode;
- full table set for one hull (30 speeds x (25 captive + 1 free) steady points + 6 radiation modes +
  diffraction at about 12 headings) is about 1 GPU-hour.
- **T2/T3:** one GPU process at a time (VRAM 3-4 GB, host RAM about 1.5 GB to be measured on first run);
  CPU-only small-N checks run through `studies/parallel_runner.run_parallel(..., gb_per_job=0.4,
  reserve_gb=3.0)`. No closed-loop MPC is needed anywhere in Track B (T3).
- **T4:** quick tier uses the coarse member of each mesh series and one Froude number per case; sweeps
  only in the full tier.

### 8.7 Wall-clock estimates (agent work) and real bottlenecks

| Step | Estimate | Bottleneck |
|---|---|---|
| R1 core + checks | 4-6 h | near-singular quadrature and the convergence-order checks; compute is seconds |
| S1 steady linear | 8-12 h | FS difference operator: dispersion/damping vs radiation condition; beach design |
| S2 sinkage/trim, transom branches | 6-10 h | dry-transom edge condition (P-9) robustness; acquiring Kajitani 1983 sinkage/trim data |
| S3 nonlinear steady | 1.5-2.5 days | re-panelling with a moving contact line and transom; iteration robustness near the hump; NPL data interpretation |
| R2 radiation (uniform stream) | 1-1.5 days | stability of the forward-speed time stepping; beaches for `tau < 1/4`; Journée data |
| R3 nonlinear basis + tables | 1 day + about 1 GPU-hour | m-terms at sharp chines and transom |
| S4 planing | 3-5 days, outcome uncertain | chine separation and spray truncation are research problems; WaveBEM gives no help |
| C reference solver | 2-3 days + 0.5-2 h per run | DAE consistency at restarts, `phi_t` BVP coupling |

R1 -> S1 -> S2 -> R2 can start immediately; S3 and R3 depend on S1/S2; S4 depends on S3; C is last.

### 8.8 Empirical inputs to register (G3), data to acquire (V3), decision points

Empirical register entries (to add to `learn/meta/EMPIRICAL_REGISTER.md`):
1. `eta_dry(Fn_T, B_T/T_T, Re_T)` -- WaveBEM's form (W-33), attributed to Doctors' regressions; confirm
   source (likely Doctors, Macfarlane & Young 2007), data range (model-scale transom models), scatter;
   used only as a transom boundary state, randomised over the scatter.
2. Transom hollow length -- WaveBEM uses a fixed 5 T_T (W-34); replace by the published hollow-length
   regression (Doctors) with its scatter, or by the solver's own dry-branch hollow where it exists.
3. Breaking pressure (W-29) -- not used; breaking is flagged.
4. ITTC-1957 friction (W-42) -- only if total resistance is reported; generic, registered.
5. Water `rho`, `nu`, `g` -- parameters, not constants (G4; WaveBEM hard-codes them, W-43).

Data to acquire (V3; free sources first, ask the user before anything paid, W3):
Kajitani et al. 1983 Wigley (sinkage/trim, free model); Olivieri et al. 2001 IIHR report 421 (free);
Bailey 1976 NPL (RINA monograph, possibly paid) or the open Southampton series (Molland et al. 1994);
Journée 1992 (TU Delft report 909, free); Fridsma 1969 (DTIC, free); Savitsky 1964 (reproduced widely).
The WaveBEM files already local: `experimental-results/` (record sha256 in `data/external/SOURCES.md`).

Decision points:
- **D1 (after S1):** if upwind differences are too dispersive for Wigley/spheroid accuracy at acceptable N,
  switch the steady FS discretisation to raised panels with collocation shift (P-6 fallback); no user input
  needed. Added 2026-10-06 after the stage-1 review (RANKINE_STAGE1 section 5): the five-point upstream
  operator in use amplifies waves downstream ((5/6) k0 (k0 dx)^3 per unit length, +4.5 % per wavelength at 30
  panels per wavelength, measured +5 to 6 %), and the 6- and 7-point formulas that the 1-D symbol favours are
  unstable or noisy in the coupled panel system at 20-30 panels per wavelength. Criterion for the D1 choice: the
  2-D discrete dispersion relation of the COUPLED system (upstream symbol x flat-panel Rankine symbol with
  aliasing) must give non-negative numerical damping at every wavenumber on the grid, with wavelength error
  < 0.5 % and damping < 1 % per wavelength at 30 panels per wavelength; checked by symbol analysis before
  implementation and by check S3b (pointwise Kelvin field, 10 %).
  **Decided 2026-10-06 (stage B3; RANKINE_STAGE1_2026-10-06.md section 8, `studies/rankine_d1.py`,
  `hydro/rankine/shifted.py`).** Analysis of the coupled discrete system: a Dawson-type row is a linear multistep
  method for y'' = -omega^2 y with omega^2 h^2 = 2 pi dx^2/(k S(alpha, beta)) up to about 0.5-1.1 on grids of 20-30
  panels per wavelength (short diverging waves); by Dahlquist's (1978) barrier only order <= 2 can be stable for every
  omega h, and upwind5/6/7 are not (stability intervals ~0, 1e-4, 0.35). Chosen: the collocation shift of Jensen et
  al. (1986) / Raven (1996) transferred to the Green formulation -- Numerov rows centred one panel upstream of their
  own source (neutral real symbol, interval of periodicity (0, 6), radiation condition from the row count) plus a box
  correction of the constant panel density (cancels the -kappa^2 dx^2/24 panel error). Coupled wavenumber error
  -0.012 % (transverse) to -0.063 % (55 deg) at 30 panels per wavelength, zero damping/growth at every wavenumber;
  S3b 1.6 % (was 23 %). Raised panels (P-6 fallback) reach similar accuracy only with h >= 1.5 dx and need a new
  free-surface representation, so they were not adopted. Free-surface products by FFT, GMRES with a y-circulant /
  x-exact preconditioner (23-35 iterations independent of N; N = 1.1e5 in about 6 s plus 50-75 s for the dense body blocks of an 80x32 body).
  After the B3 review (stage note 8.8): (i) the resolution guard of the blended rows makes the operator
  speed-dependent on a fixed grid, so production tables choose the free-surface grid per lambda0 (dx = lambda0/n,
  n >= 20) and every solve reports the guard state (`info["guard_l"]`, warning when active); (ii) GMRES accepts a
  stalled solve only below the registered residual floor 1e-10 and `SteadyShiftedFS.solve` raises otherwise;
  (iii) the "independent Kelvin-source quadrature" reference of the S1 table above now exists as
  `hydro/rankine/kelvin.py` (exact Kelvin Green function, body-only BIE; check S2K), and the near-field
  resistance functional loses accuracy for deep bodies (k0 f >~ 3: use the far field or the Kelvin check).
- **B4 (2026-10-06, surface-piercing hull; stage note section 9):** the B3 rows need both march conditions at a
  line's upstream end, so lattice lines cut by the hull are replaced by a body-fitted band of continuous lines around
  the waterline (rows with lateral interpolation and a hull-wall virtual point from the wall conditions); the
  near-field wave resistance of the linear model is R_p MINUS (rho g/2) oint zeta^2 n_x dl (momentum balance with
  fluid crossing z = 0; confirmed by a calibrated wave cut); the dry transom takes its two march conditions at the
  edge (zeta = z_T, i.e. atmospheric pressure; smooth detachment zeta_x = buttock slope); the wet transom is the
  registered WaveBEM patch. Open (D4): in the long-wave limit (k0 L <~ 1.5, Fn >~ 0.8 for B/L 0.1) the band rows are
  unstable (check H9 fails); candidates: Dawson (double-body) rows next to the hull, or a column-shifted lattice with
  symmetric interpolation. Until then the steady tables are limited to k0 L >~ 3.
  B4 review (stage note 9.7/9.8): the V1 reference files shipped with WaveBEM (`wigleyDragExp1-4.csv`) do not
  reproduce the measured markers of their source figure at Fn < 0.3 (they follow the slender-ship theory lines; the
  fourth file is UT3, Y.N.U. is absent) -- any use of these files as C_W data is invalid there; validation now compares
  the wave energy flux with the measured wave-pattern resistance C_WP and the hull force with C_W (C_T-based), with
  tolerances that include the measured laboratory scatter. Transom branches: implemented, not verified (the momentum
  identity with the face terms does not close at the affordable grids; the corner pressure integration is not
  converged). Hull-wall ghost corrected (section-curvature and flare terms). Validity: pointed hull verified only where
  the near/far row passes (Fn values in 9.8); no statement above the last passing Fn.
- **B5 (2026-10-06/07, forward-speed radiation and diffraction, step R2 with the uniform-stream basis; stage note section
  10):** realised as P-2/P-4 prescribe -- ONE LTI system per speed and attitude (free-surface state (zeta, phi_F), the
  Dirichlet-to-Neumann map from the same Rankine BIE), evaluated exactly in the frequency domain (one complex dense LU per
  frequency; the time domain is the same matrix, used for the energy and Cummins checks). Decisions: (i) beaches on all
  sides and a NON-marching free-surface operator: d/dx upwind-biased (fifth order), because a skew d/dx leaves the
  saw-tooth mode undamped and the B3 marching rows cannot carry upstream (tau < 1/4) waves; every forward-speed mode
  decays (check SP); (ii) afterbody wall rule for the band cross term (a downwind stencil there is unstable); (iii) primary
  force = direct pressure (the NK model's own pressure); the Ogilvie-Tuck m-term form is tabulated too -- for the uniform
  stream W.n != 0, so Tuck's theorem does not hold and the two differ by an O(U n_x) "NK inconsistency", derived
  (generalised Tuck identity); (iv) Timman-Newman and Haskind at forward speed hold for the NK model only with a
  free-surface waterline term, derived and checked as identities; the classical symmetry is a theorem without a
  waterline (submerged spheroid check); (v) at forward speed the memory function tends to a constant, so the Cummins
  form uses the STATIC stiffness C0 with an integrable kernel, and C0 is supplied in the plant by F_steady(U, q) (task
  form F = -A_inf a - int K v + F_steady). Table interface: `hydro/rankine/tables.py` (stage note 10.6). Open: the
  double-body basis (R3; removes the NK inconsistency and should make the D-form passive), Galerkin-symmetric DtN (energy
  neutrality at U = 0 without a beach), FFT acceleration for finer grids at high frequency, transom bodies.
- **B5 review (2026-10-07; stage note 10.8-10.11):** (i) the hull wall is an INFLOW boundary of the uniform-stream
  advection d/dx at fixed y wherever the waterline widens downstream (afterbody for U > 0); the B5 "wall rule"
  (derivative along the waterline on the first band line) was first-order inconsistent and is replaced by a trace
  closure: wall nodes on the waterline carry the hull trace of phi and the contact-line elevation from the dynamic
  condition, with phi_x from the trace and the body condition -- consistent on wall-sided waterlines, stable (every mode
  decays), Wigley coefficients now converge at about second order (Richardson error of PROD <= 0.93 %); (ii) at a
  FLARED waterline oblique to the stream the uniform-stream free-surface condition is Dirichlet-like next to the hull
  (U^2 d_x^2 has a normal component that dominates g phi_z), so the corner is singular (grad phi ~ r^(pi/(2 alpha) - 1),
  alpha the fluid angle) and no consistent closure exists in the NK basis; the waterline-reduced Timman-Newman check also
  fails on a wall-sided planing plan form because of its blunt pointed ends; planing-type and flared-hull radiation tables
  are therefore NOT verified, and the double-body basis (R3: tangential advection, Neumann-like corner) is a PREREQUISITE
  for planing hulls, not only an accuracy improvement; (iii) plant force law = the radiation system's own LTI form with
  its static stiffness C0 about the slice attitude, F_steady only at that attitude (the B5 law mixed two linear models);
  OT kernels recommended on a free-decay test; (iv) independent forward-speed reference: submerged sphere, Wu & Eatock
  Taylor 1988 (via Mei et al. 2020), whose model has steady-flow m-terms and was run with the same model -- added masses
  within 2 %, surge damping 8 % low (row fails); (v) plant kernels windowed only above the frequency where the
  zero-speed wavelength spans 4 cells (a window at the dispersion-aware validity limit destabilised the plant law);
  (vi) Green-identity rows relabelled (they do not test the free-surface operator); KK evaluation bug fixed (KK now
  0.03-0.06 %); speed interpolation of kernels between Fn 0.3 and 0.5 is too coarse (61 %/35 % rms).
- **B6 (2026-10-07, double-body linearisation, step R3 first part; stage note section 11):** base flow = rigid-lid
  double body at the mean attitude; free-surface conditions linearised about it in the form kinematic zeta_t +
  div_h(W zeta) = phi_z (the Phi_zz stretching term is what makes the pair reciprocal), dynamic phi_t + W.grad phi = -g
  zeta; diffraction with the incident-wave/base-flow forcing; steady problem = the omega = 0 response of the same LTI
  system with the forcing (U^2 - |W|^2)/2 (not the shifted-Numerov marching operator: the cross-stream advection breaks the
  line-marching structure; reasons in 11.2 item 9). Derived: since W is tangent to the waterline for every hull, all B5
  waterline terms vanish -- classical Timman-Newman, D-form = OT form, no waterline power; Haskind gains an area term of
  the interaction. Discretisation: conservative kinematic operator K_U = Wa^-1 A_{-U}^T Wa (exact discrete reciprocity),
  mirror wall closure (characteristic wall), second-order surface calculus, m-terms as the area-weighted adjoint of the
  edge-flux W.grad (Tuck exact; carries the r^-0.92 stem and r^-7/8 flared-waterline singularities). Outcome (full tier
  119/148, 53 min): TN Wigley 1.3 %, wall-sided planing variant 2.9 -> 1.5 % (B5 122 %), flared prismatic 5-7 % not
  converging (B5 ~500 %; with m = 0 0.3 %: waterline-row m-terms), flared coefficients converge (1.5 %); sphere passes at
  the B5 mesh, fails at 48 x 24 on A13/A31 next to the critical frequency; Journee 49/58; steady Wigley better trim,
  profile and C_WP hump, worse low-Fn wave resistance (+41-45 %). Limits found: the double-body base flow is a LOW-Fn base
  flow -- its static stiffness is destabilising ~U^2 and the plant law is statically unstable from Fn ~0.6 on the
  prismatic body (planing speeds need the steady free-surface base flow); a submerged chine makes it singular
  (non-integrable m); the panel representation's base velocity has a 1-4 % wall-normal error at the first free-surface
  line, which spoils the first-line consistency of the operators (DXD; fix = wall-tangent base flow on the first lines);
  the low-speed O(U^2) change (14.7 % at Fn 0.05) is not an NK artefact. Plant use: double-body tables for Fn <= 0.4-0.5
  on hulls without submerged chines; no planing-speed tables yet.
- **B6 review (2026-10-07, stage note section 12; corrections marked in place in section 11):** (i) the Timman-Newman /
  Haskind / Tuck / free-surface Green-sum rows hold BY CONSTRUCTION for any operator once K = Wa^-1 A_{-U}^T Wa and the
  m-terms are adjoints -- implementation rows only; the "B5 ~500 % -> B6 5 %" comparisons above are not like-for-like and
  are withdrawn. (ii) Tested against an EXACT double-body flow (half spheroid) and the exact uniform stream: the
  advective A is consistent everywhere, the conservative K is NOT at grid non-smoothness lines (band kinks at pointed
  waterline ends, band/lattice junction, stretching starts: K 1 grows like 1/h -- the discrete metric identity fails);
  effect negligible on the Wigley I, up to 34 % on the flared prismatic body, so the flared/planing coefficients above are
  unverified; remedy = an edge-flux (finite-volume) A/K pair with a skew/symmetric split that is exactly adjoint and
  consistent (plan in 12.7). (iii) The plant law needs the frozen base-pressure stiffness C_s (motion of the hull through
  the steady pressure field; zero for NK): F = ... - (C0 + C_s)(q - qbar); C_s from a surface formula verified two ways to
  0.9 %; with it and a correct criterion for a non-symmetric stiffness (eigenvalues of M^-1 C, not of its symmetric part)
  every planing node is statically stable on three grids and every free decay decays -- the "statically unstable from Fn
  ~0.6" statement above is withdrawn; open: the hydrodynamic stiffness is not grid-converged at Fn 0.8-1.0 and B(omega)
  is non-passive at low frequency from Fn 0.8. (iv) Pointed-end meshes now span the wetted range only (the B6 meshes had
  dry, inverted end stations at small trims); the prism's submerged chine near the tips keeps its attitude derivatives
  non-smooth. (v) The steady domain scales with the steady wavelength: +1 % against an exact Kelvin solve (B6 domain
  +13 %); the Wigley's +42-45 % low-Fn wave-cut excess is not a domain effect. (vi) Pressure-release image (Fn -> inf):
  no sign flip of the base-flow stiffness, 0.22 x the rigid lid's. Plant use: double-body tables with C0 + C_s for
  wall-sided hulls without submerged chines up to Fn ~0.4-0.5; flared/planing hulls after the A/K pair.
- **D2 (after S3):** whether to attempt S4 (planing BVP). If S4 fails its V1 tolerances, the tables end at
  the solver's validity limit and the plant cannot fill the planing range by switching theory with speed
  (G1); the options then go to the user.
- **D2 outcome -- planing feasibility gate B7 (2026-10-07; stage note `RANKINE_STAGE1_2026-10-06.md` section 13).**
  The steady planing problem of a prismatic hard-chine surface was solved in the LINEAR potential theory with gravity
  in the free-surface condition (the Doctors / Lai-Troesch / Savander family; no WaveBEM code involved, WaveBEM has no
  planing capability): wetted area from the solution (linearised Wagner spray-root condition), Kutta condition at the
  transom, chine pressure released at the edge (smooth chine detachment is not representable in the linear theory with
  g > 0). New code `hydro/rankine/planing_kernel.py`, `planing_steady.py`, `vlm_ref.py`; gate `studies/rankine_planing_
  gate.py`. Numerics converge (lift within 2 % between the two finest grids, observed order 1.5); wetted lengths, the
  Wagner keel-chine relation and the centre of pressure agree with Savitsky; the LIFT does not: 27.5 % of 69 conditions
  within 10 % (needed 80 %), median C_L/C_L,Savitsky 1.05 / 1.23 / 1.40 at 10 / 20 / 30 deg deadrise, because the
  flat-plate body condition removes the deadrise reduction of lift and there is no chine separation. Registered verdict
  INCONCLUSIVE (two verification rows failed: a kernel cross-check whose reference was under-resolved, and the
  gravity-free limit, which remains unverified -- the lift approaches it slowly); assessment NO-GO for the linear route,
  because the deadrise trend fails even after any speed-wise common rescaling of the lift. Step S4 in its planned form (planing BVP inside the linear Rankine framework) is therefore not
  recommended; options for the user in 13.7 (nonlinear section correction with the 3-D linear free surface; body-exact
  nonlinear planing solver; keep the speed-indexed high-speed plant).
- **D2 outcome -- route 1 gate B8 (2026-10-07/08; stage note section 14).** B7's 3-D linear free surface with gravity
  kept, flat-plate cross-section replaced by a nonlinear wedge entry (modified Logvinovich model, within 2 % of the
  similarity-solution force at 20 deg) and chine-flow separation (CFD-based similitude of Wen et al. 2022, registered
  E-B8-1; the derivable zero-pressure separation model was rejected by its 2-D check), coupled as a matched-asymptotic
  composite (new `hydro/rankine/planing_wedge2d.py`, `planing_nl.py`; gate `studies/rankine_planing_b8.py`). B7's PL1
  failure explained (collocation on the lateral hat kinks; with off-kink collocation the gravity-free limit matches the
  thin wing to 0.2 %). Numerics adequate (U_num 2.4 %); lift at 20 deg 25-55 % ABOVE Savitsky (0 of 24 within 10 %;
  higher than B7, because the separated-flow force of chine-wetted sections outweighs the deadrise reduction); equilibrium
  trims 0.6-3.4 deg low. Registered verdict NO-GO. A linear-3-D/nonlinear-2-D composite cannot relieve the separated
  force at the transom; remaining options in 14.7 (fully nonlinear 3-D planing; 2D+t with gravity plus a derived transom
  relief; keep the speed-indexed high-speed plant).
- **D3:** WaveBEM's wet-transom model is the only potential-flow wet-transom treatment available; if its
  registered scatter dominates the 8-18 kn uncertainty (as expected), only CFD on the real hull can narrow
  it (ANYSPEED_PRINCIPLES section 6).

---

## 9. References

Code: WaveBEM (github.com/mathLab/WaveBEM), local clone `ext_sims/WaveBEM`; deal.II 8.4 `QGaussOneOverR`.

- Mola A., Heltai L., DeSimone A. (2013). A stable and adaptive semi-Lagrangian potential model for unsteady
  and nonlinear ship-wave interactions. Eng. Anal. Bound. Elem. 37:128-143.
- Mola A., Heltai L., DeSimone A. (2017). Wet and dry transom stern treatment for unsteady and nonlinear
  potential flow model for naval hydrodynamics simulations. J. Ship Research 61(1):1-14.
- Brooks A.N., Hughes T.J.R. (1982). Streamline upwind/Petrov-Galerkin formulations. CMAME 32:199-259.
- Cummins W.E. (1962). The impulse response function and ship motions. Schiffstechnik 9:101-109.
- Dawson C.W. (1977). A practical computer method for solving ship-wave problems. 2nd Int. Conf. Numerical
  Ship Hydrodynamics, Berkeley.
- Doctors L.J., Macfarlane G.J., Young R. (2007). A study of transom-stern ventilation. Int. Shipbuilding
  Progress 54:145-163.
- Doctors L.J., Beck R.F. (1987). Convergence properties of the Neumann-Kelvin problem for a submerged body.
  J. Ship Research 31.
- Duffy M.G. (1982). Quadrature over a pyramid or cube of integrands with a singularity at a vertex.
  SIAM J. Numer. Anal. 19:1260-1262.
- Farell C. (1973). On the wave resistance of a submerged spheroid. J. Ship Research 17:1-11.
- Havelock T.H. (1931). The wave resistance of a spheroid. Proc. R. Soc. A 131:275-285.
- Hess J.L., Smith A.M.O. (1964). Calculation of non-lifting potential flow about arbitrary
  three-dimensional bodies. J. Ship Research 8:22-44.
- Hulme A. (1982). The wave forces acting on a floating hemisphere undergoing forced periodic oscillations.
  J. Fluid Mech. 121:443-463.
- Jensen G., Mi Z.-X., Söding H. (1986). Rankine source methods for numerical solutions of the steady wave
  resistance problem. 16th Symp. Naval Hydrodynamics.
- Journée J.M.J. (1992). Experiments and calculations on four Wigley hullforms. TU Delft, Ship
  Hydromechanics Laboratory, Report 909.
- Kajitani H. et al. (1983). The summary of the cooperative experiment on Wigley parabolic model in Japan.
  2nd DTNSRDC Workshop on Ship Wave-Resistance Computations.
- Kring D.C. (1994). Time domain ship motions by a three-dimensional Rankine panel method. PhD thesis, MIT.
- Lachat J.C., Watson J.O. (1976). Effective numerical treatment of boundary integral equations. IJNME
  10:991-1005.
- Lai C., Troesch A.W. (1995). Modeling issues related to the hydrodynamics of three-dimensional steady
  planing. J. Ship Research 39 (to be checked).
- Nakos D.E., Sclavounos P.D. (1990). On steady and unsteady ship wave patterns. J. Fluid Mech. 215:263-288.
- Newman J.N. (1965). The exciting forces on a moving body in waves. J. Ship Research 9:190-199.
- Newman J.N. (1986). Distributions of sources and normal dipoles over a quadrilateral panel. J. Eng. Math.
  20:113-126.
- Ogilvie T.F. (1964). Recent progress toward the understanding and prediction of ship motions. 5th Symp.
  Naval Hydrodynamics.
- Ogilvie T.F., Tuck E.O. (1969). A rational strip theory of ship motions, Part I. Univ. Michigan NAME 013.
- Olivieri A. et al. (2001). Towing tank experiments of resistance, sinkage and trim, boundary layer, wake,
  and free surface flow around a naval combatant INSEAN 2340 model. IIHR Report 421.
- Raven H.C. (1996). A solution method for the nonlinear ship wave resistance problem. PhD thesis, TU Delft.
- Savander B.R., Scorpio S.M., Taylor R.K. (2002). Steady hydrodynamic analysis of planing surfaces.
  J. Ship Research 46:248-279 (to be checked).
- Sclavounos P.D., Nakos D.E. (1988). Stability analysis of panel methods for free-surface flows with forward
  speed. 17th Symp. Naval Hydrodynamics.
- Tanizawa K. (1995). A nonlinear simulation method of 3-D body motions in waves (1st report). J. Soc. Naval
  Arch. Japan 178.
- Telles J.C.F. (1987). A self-adaptive co-ordinate transformation for efficient numerical evaluation of
  general boundary element integrals. IJNME 24:959-973.
- Timman R., Newman J.N. (1962). The coupled damping coefficients of a symmetric ship. J. Ship Research 5(4).
- Vanden-Broeck J.-M. (1980). Nonlinear stern waves. J. Fluid Mech. 96:603-611.
- Bailey D. (1976). The NPL high speed round bilge displacement hull series. RINA Maritime Technology
  Monograph 4.
- Molland A.F., Wellicome J.F., Couser P.R. (1994). Resistance experiments on a systematic series of high
  speed displacement catamaran forms. Univ. Southampton Ship Science Report 71.
- Fridsma G. (1969). A systematic study of the rough-water performance of planing boats. Davidson Lab
  Report 1275.
- Savitsky D. (1964). Hydrodynamic design of planing hulls. Marine Technology 1:71-95.
