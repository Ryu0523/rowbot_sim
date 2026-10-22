# 船体水动力模型的推导（2026-10-09）

## 中文摘要（给读者）

**问题。** 仿真船每一步要算出水对船体的总作用力（六个方向），并给出控制要用的量：竖向加速度（含冲击尖峰）、船首离水面的高度、船底哪些部分浸湿。要覆盖从静止到 40 节、静水到四级海况、所有浪向，而且每一步只能用几毫秒。本文不从"哪篇论文的方法好用"出发，而是先把精确的问题写出来，再一项一项推出：哪些简化在哪里成立、误差多大，最少需要记住多少历史，每种算法要花多少计算，最后得出在计算预算内误差最小的组合。

**船的状态可以分成三段，每段能用的简化不同。** 判断依据是几个无量纲比值（见正文第 3 节，脚本 `studies/derive_regime.py` 用仿真船自己的平衡姿态算出）：
- **低速段（约 6 节以下）**：重力主导，船像排水型船一样浮着。这里要的是"线性辐射力加记忆"——也就是船体运动兴起的波浪反过来作用在船上，这部分可以用已经核对过的三维面元法离线精确算出，再压缩成少量状态量。现在仿真船用的是"冲击型"的截面公式，它在这里不对：附加质量多算了约七成，而且完全没有兴波阻尼。这正是清单里"静止时共振响应大 5.6 倍"的原因。
- **过渡段（约 6–14 节）**：重力、动升力、三维兴波、船尾干湿变化都是同一量级，没有任何一个比值是小的。这一段没有可以严格推出的简化，只能用"两端都正确"的组合模型，并如实标为"未验证"。
- **滑行段（约 14–40 节）**：动升力主导，"把船体切成一片片横截面、每片独立计算"的做法（下称"切片模型"）成立，但有三项误差可以量化：
  1. **三维效应。** 用自己写的线性升力面程序（`studies/derive_slender.py`）精确算出：
     - 对总升力，切片模型的误差在中低速很小（约 2%–5%）；但在高速、折角线（船底与船侧的交线）在船尾处仍未浸湿时，切片模型多算 10%–17%。
     - 压力中心位置的误差约为船宽的 0.03–0.13 倍。
     - 折算到纵倾，约为 0.1–0.3 度；在 12 节可到 0.8 度。
  2. **截面平面里的重力。** 在滑行时是一成到两成的修正，越慢越大。仓库里以前的稳态计算的误差，正是按这个重力比值排序的（第 3.3 节）。
  3. **船底斜升角带来的非线性。** 它不是小量：老的 Wagner 截面公式在 20 度斜升角时多算约四成，修正 Logvinovich 公式只差约 2%。

**关键推导一：同一个物理参数统一了几处看似矛盾的数据。** 截面里水的运动受重力影响多大，由"截面下沉速度的平方 / 重力乘半宽"这个比值决定。
- **落水和冲击**：比值大（约 3），重力可忽略。这时折角线浸湿后还有一个逐渐衰减的力（王等 2022 年的公式对得上）。
- **稳态滑行**：比值小（约 0.03–0.06），重力主导。这时浸湿后的截面基本只有静浮力，没有持续的动量力。现在的"冻结"规则在稳态上对得上，原因就在这里。
- 所以"落水要求有力、稳态滑行要求没有力"不矛盾，而是同一个量的两个极限。
- 静止时附加质量随频率变化（低频大、高频小），也是同一个比值在起作用。

**关键推导二：一个两端都精确的组合。** 做法：
- 用离线精确算出的三维线性解；
- 加上"截面非线性力减去同一截面模型的线性化"。
- 前提是减去的那部分必须恰好是所用截面模型自己的线性化，这样两边不会重复计算。

这样构造之后，两个极限都自动精确：
- 低速、小幅运动时，自动等于三维面元法的结果；
- 高速时，自动带上船尾卸载和有限长度的三维修正。

剩下的误差只来自"非线性"和"三维/重力"的交叉项。按现有数据估计：
- 滑行段约为升力的一成以内；
- 20 节附近可能到一成到两成。

**推荐的设计**（第 6 节）：
- 静浮力和入射波压力在瞬时湿表面上精确积分；
- 低速的三维辐射记忆，用状态空间形式，按航速查表；
- 截面力用修正 Logvinovich 入水公式，折角线浸湿后的力用"按重力比值在两个已知极限之间过渡"的规则，出水不计动量项；
- 加上按湿面形状查表的三维修正，以及摩擦。
- 冲击期间把截面力的积分细分到更小的步长：现在 0.02 秒的步长会使冲击峰值平均少算几个百分点、最坏少算两成。
- 估计每步 2–2.5 毫秒。

**有两处只能是假设，不能离线精确算出：**
- 折角线浸湿后的力在两个极限之间怎么过渡：需要一个带重力的二维非线性求解器，现在没有；
- 低速时船尾通气的条件。
这两项都要按经验输入登记。

**按信息量从高到低的检验**（第 8 节，前几项都在一小时左右）：
1. 低速小幅运动时组合模型应和面元法逐点一致（数学检查）；
2. 把三维修正加到现有的平衡上，看 25–40 节纵倾差是否从约 −0.4 到 −0.6 度缩小到约 −0.1 到 −0.4 度；
3. 用两组参数故意错开的工况重算以前的稳态计算，分清误差到底跟哪个重力比值走；
4. 对照 Tveitnes 的低速入水试验，检验"重力削弱浸湿后力"；
5. 在 Fridsma 短波工况上比较"切片的波浪力"和"三维线性波浪力"，判断短波运动偏大是绕射引起的还是出水吸力引起的。

**这次没有改仿真船代码，没有提交。** 新增四个计算脚本 `studies/derive_*.py`，结果在 `studies/_cache/derive/`。

---

## 0. Scope, sources, conventions

- Empirical facts used: only `learn/meta/SIM_DATA_LEDGER_2026-10-08.md` (cited as ledger ids, e.g. A1.3) and the measured/computed numbers in `SIM_REDESIGN_2026-10-08.md` 进展 sections (cited as "SR"). Literature facts come from `SECTION_METHODS_2026-10-08.md` (cited "SM"). They are used as a knowledge base: sub-results and magnitudes, not as design premises.
- New numbers in this document come from four scripts. None of them changes plant code.
  - `studies/derive_regime.py`: regime map from the plant's own calm equilibria. The plant is `ZarnickVessel` in the ZR configuration (SR); the reference is Savitsky. Output: `studies/_cache/derive/regime.txt`, `regime.json`.
  - `studies/derive_slender.py`: exact linear gravity-free 3-D lift versus 2D+t, by a vortex lattice with Richardson extrapolation. Output: `slender.txt`, `slender.json`.
  - `studies/derive_cp_sensitivity.py`: what a lift or centre-of-pressure error does to the running trim and resistance. Output: `cp_sensitivity.txt`.
  - `studies/derive_b8_residual.py`: which group orders the residual of the repo's stage-B7/B8 steady computations against Savitsky. Output: `b8_residual.txt`.
- Labels used throughout:
  - **[D]** derived here, from equations or exact computations;
  - **[K]** knowledge-base fact (SM, literature);
  - **[L]** ledger data;
  - **[A]** assumption.
- Boat (hydro/hulls.py scarab_195): keel length L = 5.4 m, chine beam b = 2.025 m (L/b = 2.67), deadrise beta = 20 deg, mass 1400 kg, LCG 2.18 m from the transom, draught at rest T = 0.385 m. The hull lines are a parametric stand-in.

## 1. The problem, stated formally

**Given at time t:**
- the rigid-body state q(t) = (x, y, z, phi, theta, psi) and its rates, and its history q(s), s <= t;
- the incident wave field eta_I(x, y, t) = Re sum_j a_j exp(i(k_j . x - omega_j t + eps_j)): linear, deep water, all headings, Hs <= 1.5 m, Tp 3-10 s;
- the actuator state (jet thrust, nozzle).

**Required:** the generalised hydrodynamic force F(t) in R^6 on the hull, and the derived outputs a_z(x, t) at the CG and the bow (impact peaks included), bow freeboard, and the wetted set S_w(t).

**Envelope:** U from -3 to +40 kn; calm to sea state 4; all headings; heel up to capsize.

**Budget:**
- Causal.
- A few ms per 0.02 s step on one core. The current ZR plant takes 1.85 ms (SR).
- Batch mode is allowed for data generation.
- Offline compute is allowed for tables, at hours scale on this 16 GB machine.

**Accuracy target (A).** It is defined by what the controller consumes and bounded by what can be verified.

| id | control quantity | target | verifiable against | data uncertainty |
|---|---|---|---|---|
| A1 | calm speed at given thrust | R(U) +-10 % (gives about +-6 % speed at 30 kn: dR/dU 206 N s/m, jet dT/dU about -100) | Savitsky (A1, A10) | regression; raw tables not in repo |
| A2 | running trim | +-0.5 deg (R sensitivity 6-13 % per deg, cp_sensitivity.txt) | Savitsky | same |
| A3 | heave and pitch in waves | amplitude +-20 % | Fridsma 1969 (A3) | transcription |
| A4 | vertical acceleration statistics (mean positive peak, A1/10) at CG and bow | +-30 %, **with a stated output filter** (see 3.5) | Fridsma 1969/1971 (A3, A5) | 1971 Tp unknown (+-40 % effect); seed scatter 2-4 counts |
| A5 | section impact force | +-10 % against the 2-D nonlinear solution, not against the raw 3-D drop | ZFA drop (A6), similarity (B8.1) | digitising +-250 N, rig check +6..+11 % |
| A6 | low-speed linear response | RAO within 5 % of 3-D BEM | Capytaine / Rankine (B1, B4) | verification only |
| A7 | lateral, roll, manoeuvring, bow burial | no data (F); consistency only | — | — |

Targets tighter than the data scatter cannot be verified. A1/A2 cannot be checked better than Savitsky's own scatter, and A4 is limited by Tp and by seed scatter.

## 2. The reference problem

### 2.1 Equations

Fluid domain Omega(t) below the free surface z = eta(x, y, t), outside the hull surface S_B(t). Incompressible flow at high Reynolds number (2.3); the outer flow is irrotational.

- Laplace: grad^2 Phi = 0 in Omega(t).
- Body: dPhi/dn = V_B . n on the wetted surface S_w(t) = S_B intersected with closure(Omega). V_B = U_rigid + omega x r.
- Kinematic free-surface condition: eta_t + Phi_x eta_x + Phi_y eta_y = Phi_z on z = eta.
- Dynamic free-surface condition: Phi_t + 1/2 |grad Phi|^2 + g eta = 0 on z = eta.
- Radiation at infinity; incident field as given.
- Detachment at chines and transom: smooth separation (Kutta), with the pressure continuous and equal to atmospheric at the detachment line. The jets and spray sheets are part of the free surface. The jets are thin, carry momentum flux and little force, and are cut where they leave the body [K, ZFA].
- Pressure: p = -rho (Phi_t + 1/2 |grad Phi|^2 + g z).
- Force: F = integral over S_w(t) of p n dS + F_visc, with n the generalised normal.
- Viscous part: F_visc = integral over S_w of tau_w dS. The wall shear tau_w comes from a turbulent boundary layer, to leading order the flat-plate law on the dynamic wetted area, including the spray area.

### 2.2 Nondimensional groups

Scales: U (forward speed), b (beam), L_w (wetted length), tau (trim), beta (deadrise), V (normal velocity of a cross-plane section), c (wetted half-beam), omega_e (encounter frequency), k (wave number), A (wave amplitude).

| group | definition | role |
|---|---|---|
| Fn_L, C_V | U/sqrt(gL), U/sqrt(gb) | 3-D gravity / hump / planing |
| lambda, eps | lambda = L_w/b; eps = b/L_w = 1/lambda | slenderness |
| tau, beta | trim, deadrise | body slope along x; body slope in the cross-plane (NOT small: tan beta = 0.36) |
| Fn_c | V/sqrt(g c) | cross-plane gravity for a section entry |
| gamma | g T_pass^2/(pi^2 c), with T_pass = L_w/U | cross-plane gravity during the passage of one plane under the hull |
| omega_e^2 c/g | | cross-plane gravity for oscillation (the frequency dependence of the added mass) |
| tau_B | U omega_e/g | Brard parameter: for tau_B > 1/4 no radiated wave travels ahead of the hull |
| kb, kL_w, kA | | wave length vs beam and wetted length; steepness |
| Vdot c/V^2 | | entry deceleration (inertia vs momentum term) |
| Re | U L_w/nu | friction |
| sigma, F_T | ventilation number; transom Froude number | transom ventilation (low-speed regime) |

## 3. Regime map for this boat

### 3.1 Calm equilibria and groups (regime.txt)

ZR equilibria from the plant, Savitsky for comparison (Savitsky valid from about 13 kn).

| kn | Fn_L | C_V | trim ZR / Sav (deg) | L_K ZR (m) | lambda ZR / Sav | (L_K-L_C)_Sav m | Fn_c (ZR / Sav) | gamma (Sav L_K) | 2 pi U^2/(g L_K) | Sav static/dyn | Re |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.28 | 0.46 | 1.83 / — | 5.40 | 2.67 / — | — | 0.02 | ~10 | 0.5 | — | 9e6 |
| 8 | 0.57 | 0.92 | 2.04 / 4.73 | 5.40 | 2.67 / 2.53 | 2.84 | 0.05 / 0.11 | 1.7 | 2.0 | 3.4 | 1.9e7 |
| 10 | 0.71 | 1.15 | 2.19 / 5.50 | 5.40 | 2.67 / 2.21 | 2.44 | 0.06 / 0.16 | 1.1 | 3.1 | 1.7 | 2.3e7 |
| 12 | 0.85 | 1.39 | 6.12 / 5.84 | 4.61 | 1.72 / 1.96 | 2.29 | 0.21 / 0.20 | 0.67 | 4.8 | 0.92 | 2.4e7 |
| 16 | 1.13 | 1.85 | 4.89 / 5.28 | 4.39 | 1.50 / 1.70 | 2.54 | 0.22 / 0.24 | 0.32 | 9.2 | 0.39 | 3.0e7 |
| 20 | 1.41 | 2.31 | 3.47 / 4.34 | 4.95 | 1.50 / 1.60 | 3.09 | 0.20 / 0.25 | 0.21 | 14 | 0.22 | 4.3e7 |
| 25 | 1.77 | 2.89 | 2.36 / 3.41 | 5.40 | 1.44 / 1.53 | 3.94 | 0.17 / 0.24 | 0.15 | 21 | 0.13 | 5.8e7 |
| 30 | 2.12 | 3.46 | 1.90 / 2.75 | 5.40 | 1.39 / 1.50 | 4.89 | 0.16 / 0.24 | 0.12 | 28 | 0.09 | 7.0e7 |
| 40 | 2.83 | 4.62 | 1.45 / 1.96 | 5.40 | 1.36 / 1.47 | 6.87 | 0.17 / — | 0.08 | 50 | 0.05 | 9.3e7 |

Notes:
- At 0-10 kn the ZR plant wets the full keel at trim of about 2 deg. Savitsky's trims at 8-10 kn are outside its range.
- "Fn_c" is the steady entry of a chine-dry section, V = U tan(tau), with c = b/2.
- "gamma" uses Savitsky's keel wetted length L_K = lambda b + (L_K - L_C)/2, capped at the hull length.
- "Sav static/dyn" is the ratio of Savitsky's hydrostatic lift term to his dynamic term, 0.458 lambda^2/C_V^2.
- At 30-40 kn the Savitsky chine-dry length (L_K - L_C) exceeds the wetted length: **the chines are dry at the transom**, and the wetted planform is a delta. This matters in 3.2.

Waves (regime.txt). Head seas at 20-40 kn, Tp 3-5 s:
- omega_e is 2.9-11 rad/s and tau_B 1.3-4.4, so tau_B > 1/4 everywhere at speed. At 5 kn with Tp 8-10 s, tau_B is 0.16-0.21 < 1/4.
- kb is 0.33-0.91 for Tp 5-3 s and 0.08-0.13 for Tp 8-10 s.
- Following seas: omega_e passes through 0. This is the surf-riding and broaching regime, where the response is quasi-static.
- The steepest physical seas (Hs/lambda_p <= 1/15) have kA up to 0.21 at Tp 3 s and 0.12 at Tp 5 s.

### 3.2 Slender-body (2D+t) error: O(eps^2) or O(eps)? [D]

**Exact reduction.** Linearise the gravity-free planing problem (dynamic condition Phi_x = 0 on z = 0, so Phi is constant along streamlines of the free surface downstream). Continue Phi oddly into z > 0. The result is the thin-wing problem for the wetted planform in unbounded fluid; the planing force is half the wing force. This is the same equivalence the repo used in check B7.2 (C_L vs half thin-wing VLM, 5 %).

2D+t at linear order and g = 0 is then exactly R. T. Jones's slender-wing theory:

  dL/dx = (rho U^2 alpha / 2) d(pi s(x)^2)/dx   (planing: half of the wing value), with s(x) the local wetted half-width,

with no lift where s stops growing. That last rule is the "frozen" chine rule, at linear order.

**Interior error.** Inside the wetted region the 3-D Laplacian differs from the cross-plane Laplacian by Phi_xx ~ Phi/L_w^2 against Phi_yy ~ Phi/b^2. The relative error of the inner solution is O(eps^2), up to logarithms from the matching to the outer 3-D field.

**End errors.**
- At the transom (trailing edge) the exact flow must satisfy the Kutta condition (p goes to 0). 2D+t carries a finite load up to the transom.
- The correction is confined to a region of streamwise length O(s_T) next to the transom, where s_T is the wetted half-width at the transom. Its relative contribution to the force is therefore O(s_T/L_w) **times the local load share** at the transom.
- (i) If the chines wet ahead of the transom, s is constant aft, the 2D+t load vanishes there, and the end correction to the lift is small. The lift error is then O(eps^2).
- The centre of pressure is different. 2D+t puts all the load where s grows, while the exact load spreads a distance O(b) aft. The CP error is O(b) in absolute terms, i.e. O(eps) relative to L_w.
- (ii) If the chines are dry at the transom (delta planform: 30-40 kn), the 2D+t load is largest at the transom. The lift error is then O(s_T/L_w), which is first order.

Numerical confirmation (derive_slender.py). Vortex lattice with Richardson extrapolation; observed lattice order about 1. Rectangle, span 1, chord lambda:

| lambda | 0.5 | 1 | 1.5 | 2 | 3 | 4 | 8 |
|---|---|---|---|---|---|---|---|
| L_exact / L_2D+t | 0.787 | 0.929 | 0.970 | 0.984 | 0.994 | 0.997 | 0.999 |
| x_cp aft of the leading edge, in beams (2D+t: 0) | 0.105 | 0.167 | 0.201 | 0.220 | 0.239 | 0.247 | 0.258 |

- Slope d ln|1 - L/L_2D+t| / d ln(1/lambda) = **2.12**: the lift error is second order, as derived.
- x_cp tends to a constant **0.26 b**: the CP error is first order, as derived.

This boat's wetted planforms. They are built from Savitsky's (lambda, tau): spray-root lines from the keel to the chine with L_K - L_C = b tan(beta)/(pi tan(tau)), then constant beam to the transom. "k" is L_2D+t/L_exact, i.e. how much 2D+t over-lifts.

| kn | 12 | 16 | 20 | 25 | 30 | 35 | 40 |
|---|---|---|---|---|---|---|---|
| k | 1.02 | 1.03 | 1.04 | 1.05 | 1.08 | 1.13 | 1.17 |
| 2D+t CP forward of exact, in beams | +0.075 | +0.057 | +0.034 | +0.003 | -0.035 | -0.097 | -0.129 |
| implied trim error of a 2D+t model (deg), Savitsky sensitivity | +0.81 | +0.25 | +0.01 | -0.12 | -0.19 | -0.28 | -0.30 |
| implied R error (%) | +7.8 | +0.7 | -1.6 | -1.3 | +0.9 | +5.2 | +8.3 |

The table rows for this boat come from cp_sensitivity.txt part 2. The model's equilibrium is taken as Savitsky's equilibrium for mass M/k with the LCG displaced by the CP error.

Accuracy caveat: on pure deltas the lattice did not converge monotonically. The A = 1 delta gave C_Lalpha 1.31-1.32 at 32-48 spanwise panels, and the Richardson order hit its bounds. The 35-40 kn rows therefore carry about +-5 % on k. The rectangle rows converged (order 0.92-0.99).

**Conclusion [D].** For this boat the 2D+t reduction is not "O(eps^2) small" at eps = 1/lambda = 0.25-0.7:
- the lift is second order (2-8 %) while the chines wet ahead of the transom;
- the lift is first order (13-17 %) at 35-40 kn, where they do not;
- the CP is first order everywhere (0.03-0.13 b).

These are gravity-free linear numbers. They are exactly computable offline, and smooth in (lambda, (L_K - L_C)/b, s_T/b).

### 3.3 Gravity in the cross-plane [D]

For a chine-dry wedge section, compare the hydrostatic force rho g A = rho g h^2/tan(beta) with the momentum force U dm/dx V = rho (pi^3/4) h V^2/tan^2(beta) (Wagner c = pi h/(2 tan beta)):

  R_g(h) = 4 g h tan(beta) / (pi^3 V^2),   and at chine wetting (h = (2/pi) c tan beta)   R_g = 8 tan^2(beta) / (pi^4 Fn_c^2) = 0.0109/Fn_c^2   (beta 20 deg).

- Steady planing (Fn_c 0.16-0.25, table 3.1): R_g at the chine is 0.17-0.43; averaged along the chine-dry run it is about half that.
- Savitsky's own hydrostatic/dynamic term ratio is 0.05-0.22 at 20-40 kn and 0.4-0.9 at 12-16 kn: the same magnitude and trend, from independent data [L, A1].
- The passage-time form gives the same scaling: gamma = g T_pass^2/(pi^2 c) is 0.08-0.21 at 20-40 kn and 0.3-0.7 at 12-16 kn.

So:
- **gravity is a 10-40 % effect in steady planing, and grows as speed falls**;
- **in impacts (Fn_c = 0.5-3, table 3.5) it is a 0.1-4 % effect**;
- **at 6-12 kn (gamma 0.7-4) there is no small parameter.**

**Two limits of the post-chine section force [D].** After the chines wet, the section keeps descending at V.
- Fn_c -> infinity (g -> 0): the separated free-streamline flow carries a steady momentum force, Wen's Cs_inf, approached through a decay after chine wetting [K]. This reproduces the drop test (Fn_c of about 3.7): SM, +4..+8 % against the 2-D nonlinear solution.
- Fn_c -> 0: the dynamic free-surface condition reduces to Phi_z = 0 for the slow part (rigid lid; the time scale c/V is long against sqrt(c/g)). A body translating steadily under a rigid lid feels no dynamic force (d'Alembert in 2-D); what remains is the hydrostatic force of the displaced and piled-up water, plus the inertia term m_a(0) Vdot. **So the post-chine momentum force vanishes as Fn_c -> 0.** That is the frozen rule, which steady planing data support (SR 进展 2 (d): the 'extend' upper bound wrecks Savitsky and the Fridsma calm runs).
- **The drop-test and steady-planing requirements are the two limits of one function** F_sep(h*, Fn_c, beta, Vdot c/V^2). This resolves SR's open item 4 as a theory statement. The intermediate law is unknown; see 4.4.
- The same group governs oscillation: omega^2 c/g -> 0 gives the rigid-lid, low-frequency added mass, and omega^2 c/g -> infinity gives Phi = 0. BEM at rest: A33 is 11244 kg at 0.5 rad/s against A33_inf 4538 kg [L B1.4]. At the rest heave frequency, omega^2 (b/2)/g = 1.55: intermediate.

**Evidence that the steady residuals are ordered by gravity, not slenderness [D, from L A10].** Stage B8 "mlm_only" put the chine-dry MLM sections inside the 3-D linear gravity solver; 24 cases, beta 20 deg. Its lift/Savitsky ratio has these Spearman rank correlations (b8_residual.txt):
- with Fn_c: -0.89;
- with C_V: -0.93;
- with Savitsky's static/dyn ratio: +0.80;
- with lambda: +0.02.

Ordering the cases by Fn_c reduces the neighbour-difference spread from 0.159 to 0.058 (in log ratio); ordering by lambda only to 0.146. The ratio runs from 1.47 at Fn_c 0.1 down to 0.80 at Fn_c 0.9.

The B8 grid confounds Fn_c with C_V, so this does not say which gravity group it is (check C3 in section 8). It does say that the remaining steady error after MLM plus 3-D linear is a gravity-times-nonlinearity cross term. That term is absent from both a gravity-free nonlinear section and a linear 3-D solver.

### 3.4 Body nonlinearity in the cross-plane [D, K]

The body slope in the cross-plane is tan(beta) = 0.36. Linear (von Karman) section theory under-predicts the 30 deg wedge force by 22 % [L A6: 0.78]. Wagner's pile-up over-predicts the similarity solution by 1.39x at 20 deg [L B8.1: 58.5 vs 42.0]. MLM is within -1.7 / +2.1 / +9.1 % at 10 / 20 / 30 deg [L B8.1].

For constant-velocity entry the exact gravity-free force is a one-parameter table in beta (similarity solution). **Body nonlinearity is therefore O(1) but exactly tabulable offline**: similarity or MLM for constant V, and MLM with the Vdot term for varying V. It is not an approximation that has to be modelled.

The ZR trim deficit at 20-40 kn implies a lift-efficiency excess. From Savitsky's lift law (C_L proportional to tau^1.1), k = (tau_Sav/tau_ZR)^1.1 = 1.28-1.53. This is consistent with the Wagner excess (1.39) plus the 3-D excess k of 3.2 (1.04-1.17). With MLM it falls to 1.04-1.30.

### 3.5 Impacts and time resolution [D]

- Section entry from keel to chine lasts t_cw = (2/pi) c tan(beta)/V (Wagner). For V = 1-3 m/s and c = 0.5-1 m, t_cw is 0.04-0.23 s, i.e. **2-12 steps of 0.02 s** (regime.txt).
- Fn_c is 0.45-1.9: gravity-free to 1-5 %.
- The deceleration parameter: the inertia term against the momentum term is m_a Vdot / (dm_a/dt V) = c Vdot tan(beta)/(pi V^2). At a 7 g hull deceleration, c = 1 m and V = 3 m/s this is 0.9. **The Vdot term is O(1) in slamming and must stay implicit**: added mass on the left-hand side, as in the plant.
- Sampling a half-sine pulse of duration T on a 0.02 s grid loses up to 1 - cos(pi dt/(2T)) of the peak (regime.txt): 19 % at T = 0.05 s (6.5 % on average over phase) and 3.4 % at T = 0.12 s. The explicit 2nd-order integration error during the pulse adds to this.
- Observed sensitivity [L A5]: model-scale dt 0.004 -> 0.001 s (full-scale 0.012 -> 0.003 s) changed 1971 peak means by up to 18 %.
- **Consequences:**
  - (i) The section force needs sub-stepping, or analytic integration over the step, while any plane is in an entry with t_cw < about 5 dt.
  - (ii) The acceleration statistic must be defined with an explicit output low-pass filter. Otherwise "peak acceleration" is a property of the integrator, not of the hull.

### 3.6 Linearised radiation and frozen kernels [D, L]

Linear radiation about a mean wetted surface needs the relative motion amplitude to be small against the draught (T = 0.385 m at rest). In head seas at rest, the relative motion is small for lambda/L > 3 (contouring) and of the order of A for lambda/L < 1.5. So the linear radiation component is valid at low speed for Hs up to about 0.5 m in short waves and for all Hs in long waves. Beyond that, the nonlinear section increment (4.5) carries the excess.

At planing speed the linearisation of the 3-D field about the calm-water planing state is not smooth in attitude. The repo's forward-speed tables interpolated across attitude failed by 3.5-6x their tolerance [L B5.10, B6.8]. The prism attitude-grid linearity check failed (1.29 / 8.05, corner velocities |W|/U up to 22) [L B6.7]. **Frozen LTI kernels tabulated over (U, z, theta) are not a safe representation at planing speed.** Quantities smooth in a few planform parameters (3.2) are.

### 3.7 Wave excitation in 2D+t [D]

2D+t takes the incident orbital velocity as uniform over a section (relative-velocity hypothesis); the diffraction by the section is ignored. The neglected term is O(kc):
- Fridsma lambda/L = 1 has kc = 0.63: not small. This is where all configurations over-predict heave by 1.3-1.5x [L A3].
- For this boat, head seas: Tp 3 s gives kc = 0.45; Tp 5 s, 0.16; Tp 8-10 s, 0.04-0.06.
- The 3-D linear forward-speed solver contains the diffraction exactly (verified on Wigley/Journee F3', F5' to a few % [L A9]).

### 3.8 Viscous effects [D, L]

- Re is 2e7-1e8, so the boundary layer is thin: delta/L of about 1 %. The viscous effect on pressure is O(delta/b), negligible.
- Friction is 30-50 % of the calm resistance at 20-40 kn [L D2.12: 973 of 1923 N at 20 kn].
- The friction error enters R through C_f (ITTC-57, an empirical register item) and through the wetted area including the spray area. A +-5 % C_f uncertainty gives +-2 % R.

### 3.9 Summary regime map

| reduction | small parameter | order of neglected term | R0: 0-6 kn | R1: 6-14 kn | R2: 14-40 kn |
|---|---|---|---|---|---|
| 2D+t, interior | eps^2 = (b/L_w)^2 | O(eps^2) | 0.14 (but see gravity) | 0.2-0.4 | 0.2-0.5, lift error measured 2-8 % |
| 2D+t, transom / CP | s_T/L_w | O(eps): CP 0.03-0.13 b; lift O(s_T/L_w) when chines dry at the transom | — | ~0.2 b | trim 0.1-0.3 deg (12 kn: 0.8); lift 13-17 % at 35-40 kn |
| gravity-free cross-plane | gamma, 0.0109/Fn_c^2 | O(gamma) | fails (gamma >> 1) | fails (0.7-4) | 0.08-0.3: 10-40 % local, 5-22 % of lift |
| gravity-free impact | 1/Fn_c^2 x 0.0109 | O(0.01-0.05) | valid | valid | valid |
| linearised body condition (section) | tan(beta) = 0.36 | O(1): 0.78x / 1.39x | small-amplitude linear OK | — | must be nonlinear (tabulable exactly) |
| linear radiation about the mean surface | relative motion / T | O(a/T) | valid for small relative motion | partly | invalid (use the nonlinear section) |
| uniform orbital velocity per section | kc | O(kc) | 3-D BEM covers it | 0.05-0.45 | 0.05-0.45 (Tp 3-10 s) |
| no 3-D gravity waves (transverse Kelvin) | g L_K/(2 pi U^2) | O(1/(2 pi Fn_Lw^2)) | fails | 0.15-0.5 | < 0.1 |
| frozen kernels in attitude | Delta attitude / attitude | O(1) at planing | OK | — | unsafe (B5.10, B6.7) |
| 0.02 s step for impacts | dt/t_cw | peak loss 3-19 % | — | yes | yes |

**R1 (6-14 kn) has no small parameter at all.** Any model there is a composite whose accuracy can only be bounded by its two limits, not derived.

## 4. Decomposition, minimal state, exact limits, cheapest representation

### 4.1 Exact split

Write Phi = Phi_I + phi, with Phi_I the incident wave and phi the disturbance. With S_g(t) the hull surface below the *incident* surface z = eta_I (the geometric wetted surface; it is known in closed form from q and eta_I):

  F = F_HF + F_dist + F_visc,
  F_HF   = -rho integral over S_g of (g z + Phi_I,t + 1/2 |grad Phi_I|^2) n dS      (hydrostatic + Froude-Krylov, exact given S_g),
  F_dist = integral over S_w of p n dS - F_HF                                        (everything the disturbance does, including S_w differing from S_g: pile-up, spray root, transom hollow).

F_HF is computable exactly: polygon sections times quadrature (C3, C14: 1e-15 identities; 48 stations 2.7e-3, Delta x^2 convergence).

Consequence: **the buoyancy factor a_b (Zarnick's 0.5) has no place in an exact split.** The deficit it imitates (atmospheric pressure at the transom, low pressure in the hollow behind it, the dynamic pressure deficit) belongs in F_dist. A design that keeps a_b keeps an empirical weight on an exactly known term (against G3).

### 4.2 Components of F_dist

| component | physics | time scale | memory needed |
|---|---|---|---|
| C1 linear radiation-diffraction (3-D, gravity) | waves radiated and scattered by small motions about a mean surface | 2-4 s at U = 0 (kernel length, B1.12); at tau_B > 1/4 bounded by T_pass plus a weak tail at omega = g/(4U) | LTI states, order about 10 per DOF pair at U = 0 (B1.12: err_K 0.6-34 %, err_B 0.1-13 %); speed-dependent |
| C2 steady and quasi-steady planing lift | momentum of the cross-flow; 3-D transom and finite-lambda effects; 3-D gravity | T_pass = 0.26-0.6 s | none beyond planform parameters (quasi-steady) |
| C3 section entry (chine dry) | nonlinear free surface, pile-up, jet | t_cw = 0.02-0.2 s | per Lagrangian plane: h (or c) monotone, so c is a function of h_max; flag "entering" |
| C4 post-chine separated flow | detachment at the chine; decay to the steady free-streamline force (g -> 0) or to hydrostatic (Fn_c -> 0) | a few t_cw | per plane: h_cw, h - h_cw, Fn_c at wetting (or its running value) |
| C5 exit | water leaving: suction plus drag, no momentum release [K Tveitnes] | t_exit | per plane: frozen m_a(h_max), phase flag |
| C6 viscous | boundary layer | quasi-steady | none |
| C7 asymmetric (heel, sway) | per-side entry and separation | t_cw | per plane and per side: c_L, c_R, separation flags |

**Minimal state [D].**
- The exact problem's state is the free-surface pair (eta, phi_s) on a 2-D surface, which is infinite-dimensional.
- For U > 0 with tau_B > 1/4 (every condition at 10 kn and above, table 3.1/regime.txt), radiated waves cannot travel ahead of the hull. The disturbance acting on the hull was created within about T_pass, plus a weak tail at omega = g/(4U).
- Hence the minimal state at speed is a per-plane state advected at U, over the wetted length: 20-48 planes, times 2 sides, times 3-4 numbers (h_max or c_max, h_cw, phase, age), i.e. **about 120-400 scalars**. Add a small LTI state for the 3-D gravity tail. At planing speed that tail is O(static/dyn) = 5-22 % of lift, mostly steady.
- For tau_B < 1/4 (U below about 5 kn), radiated waves persist around the hull for 2-4 s. The LTI states of C1 are needed: **about 30-60 states** for a MIMO realisation of heave, pitch, surge, sway, roll and yaw.

### 4.3 Exact limits every design must satisfy

| limit | required behaviour | current plant (ZR) |
|---|---|---|
| L1: U -> 0, small amplitude | F_dist -> 3-D linear radiation-diffraction with memory: A33(omega), B33(omega) as BEM; heave natural frequency 3.94 rad/s; zeta33 about 0.3 at resonance | A33 7877 against 4248-4538 (factor 1.74 = 1.50 pile-up x 1.16 3-D), no damping, resonance 5.6x (B1.2-B1.4, B1.9) |
| L2: g -> 0, impact | section force -> gravity-free nonlinear solution: similarity for constant V; Wen decay after chine wetting | Wagner 1.9x the drop test; post-chine force zero |
| L3: steady planing, Fn_c -> 0 per plane | no post-chine momentum force; hydrostatic plus 3-D | frozen rule: OK |
| L4: linear small motion at speed | the 3-D forward-speed linear solution (Timman-Newman, Haskind) | not enforced |
| L5: U -> 0 calm | hydrostatic equilibrium; no drift (R3) | as1 fixes it (D1.3); ZR inherits v0's drift (D1.1) unless the a_b blend is replaced |
| L6: heel to capsize | exact polygon GZ (C32) | hydrostatic exact in as1 hh/hx |

### 4.4 What can be computed exactly offline, and what cannot

| part | exact offline? | tool (state) | compression and error bound |
|---|---|---|---|
| F_HF | exact online | closed form | discretisation O(Delta x^2), 2.7e-3 at 48 stations |
| C1 at U = 0 | yes | Capytaine / repo Rankine stage 2 (verified: Hulme 0.3 %, Capytaine 1-3 %, KK 6e-11) | state-space fit; error measurable (B1.12) and reducible by order; order 10 gives err_K < 2 % in heave and pitch, but roll 68 % and sway 34 % (fit quality, not physics) |
| C1 at U > 0, Fn_L <= 0.4 | yes (linear) | repo B5/B6 forward-speed LTI (Journee 49-50 of 58 points) | 1-D table in U at the calm equilibrium; interpolation in U only |
| C2 gravity-free linear 3-D correction | yes | VLM / B7 planing kernel (B7.2: VLM 5 %) | smooth 3-input table (lambda, (L_K-L_C)/b, s_T/b); bound by lattice convergence (order 1, Richardson) |
| C2 with 3-D gravity, linear | yes | B7 linear planing solver (C34 grid 2 %) | 4-input table (adds C_V); B7's own equilibria are 0.3-1.7 deg low (EQ lines), so a linear-gravity correction is exact only for its linear part |
| C3, constant V | yes | similarity table in beta | exact (B8.1 reference) |
| C3, varying V | MLM (model, -1.7..+9 % vs similarity) | closed form | W2: strip MLM equals full MLM to 0.002 % |
| C4, g = 0 | fitted to nonlinear CFD/BEM (Wen) [K] | closed form | 4-8 % vs ZFA 2-D nonlinear |
| C4 with gravity (0.1 < Fn_c < 3) | **no solver in the repo** | would need a 2-D nonlinear BEM with gravity and a chine Kutta condition (research risk), or CFD (needs an install) | — |
| C5 exit | partly (modified von Karman) [K] | closed form | weak validation (Tveitnes exit drag) |
| C7 heel > beta | **no validated theory** [K] | — | — |
| transom ventilation at 6-14 kn | **empirical** (F_T thresholds 2-4.5 [L B3.10]) | — | register; randomise |

### 4.5 The matched composite and its error [D]

Let f^NL(h, V, Vdot, state) be the nonlinear section model (C3-C5). Let f^lin be **its own linearisation** about the reference wetted state: at U = 0, the fixed-waterline Phi = 0 section added mass m_0(h_bar) with no damping; at speed, the 2D+t linear (Jones) strip. Let F^3D,lin be the exact linear 3-D solution with gravity (C1 plus the linear part of C2). Define

  F_dist = F^3D,lin + sum over planes of [ f^NL - f^lin ] dx  (+ C6).

Properties:
- **(a) Small amplitude at any speed:** the bracket is second order in amplitude, so F_dist equals F^3D,lin. L1 and L4 hold by construction.
- **(b) g -> 0 and eps -> 0 together:** F^3D,lin tends to the sum of f^lin, so F_dist tends to the sum of f^NL. L2 and L3 hold if f^NL satisfies them.
- **(c)** The neglected terms are the products: nonlinear excess times (3-D or gravity correction), O(N C). N = (f^NL - f^lin)/f^NL is about 0.3-0.5 for 20 deg sections (similarity against linear). C = (F^3D,lin - sum f^lin)/F is about 0.05-0.2 from 3.2 and 3.3.
  - That gives **O(N C) of about 2-10 % of the local force.**
  - B8 mlm_only is the steady instance of this composite. Its residual is +7..+20 % at Fn_c of 0.2 and roughly -5..+10 % at C_V 4-6 (b8_residual.txt). So the measured cross term is up to about 2x this estimate at Fn_c of 0.2.
- **(d) The double-counting condition is exact:** f^lin must be the true linearisation of the f^NL that is used. If Wagner pile-up is used for f^NL, f^lin is the pile-up linearisation, not the fixed-waterline value. Otherwise the subtraction leaves a spurious O(N) term at small amplitude. SM risk 7 is exactly this.
- **Reference state.** F^3D,lin is not tabulated over attitude (3.6). It is used in two pieces:
  - (i) the steady and quasi-steady planing correction as a function of the instantaneous planform parameters;
  - (ii) LTI memory at the calm equilibrium of the current U, where it matters, i.e. tau_B < 1/4 and U below 6-10 kn.
  - The error of freezing (ii) is second order in (attitude excursion) x (LTI correction), small at low speed. At planing speed (ii) is O(static/dyn) and can be omitted with an error of 5-20 % on the unsteady correction only.

## 5. Complexity

Per 0.02 s step, one core, 48 stations, N_w wave components. Times are estimates scaled from the measured ZR 1.85 ms, which is mostly numpy call overhead on 48-element arrays.

| representation | operations per step | memory | estimated time | notes |
|---|---|---|---|---|
| F_HF on S_g, closed-form polygons | 48 x about 50 flops + wave sums 48 x N_w trig | — | 0.1-0.3 ms (N_w 50-200) | wave sums dominate; recursive phase rotation e^{i omega dt} removes the time trig |
| 2D+t section laws (MLM, Wen, exit) | 48 x about 100 flops | — | 0.05-0.1 ms | closed forms |
| Lagrangian plane transport | shift by U dt / Delta x (interpolate) | 48 x 2 x 4 | 0.02 ms | replaces station-wise V for history-dependent laws |
| impact sub-stepping (strip only, 5-10 substeps on about 10 % of steps) | x(5-10) strip on 10 % | — | +0.05-0.2 ms average | rigid-body DOFs kept implicit in the added mass |
| C1 LTI radiation (MIMO, order 40) | 40^2 + 40 x 6 MACs | about 2400 doubles per speed node x 8 nodes | about 0.01 ms | only below about 10 kn |
| C2 3-D planform table (3-4 inputs, 2-3 outputs) | multilinear lookup | 1e3-1e4 entries | about 0.01 ms | smooth (3.2) |
| C4 gravity table (if built): (h*, Fn_c, beta_eff, Vdot c/V^2) | 4-D lookup per wetted plane | 1e4-1e5 entries, about 1 MB | 0.02-0.05 ms | needs an offline 2-D nonlinear gravity solver |
| **D1 total** | | | **about 2.1-2.6 ms** | within the budget |
| online 3-D Rankine (1000-4000 panels) | dense LU | — | 0.5-1.7 s CPU (B3.11) | 250-850x over budget |
| online 2-D nonlinear BEM per plane | 20-40 planes x hundreds of panels x small dt | — | est. 1-8 s (SM) | 500-4000x over |

Offline build cost:
- C1 at U = 0: Capytaine, about 50 frequencies x 1-3 drafts, minutes each.
- C1 forward speed: B5/B6 at about 8 speeds 0-12 kn, minutes to tens of minutes each, so about 1-3 h of compute.
- C2 gravity-free table: about 2000 planforms x about 3 s (VLM with Richardson; this session: 16 planforms in 43 s), about 1.5 h. Or about 300-500 nodes, given the smoothness, in about 20 min.
- C2 with 3-D gravity (B7 kernel): about 5-40 s per case, 300-500 cases, so 1-5 h.
- C4 gravity table: unknown. The solver does not exist; writing it is research risk over several 1-2 h stages. Each 2-D case would then be minutes, about 500 cases.

Error against cost:
- Moving from the ZR strip (1.85 ms) to D1 costs about 0.3-0.7 ms per step and 2-6 h offline (without C4).
- It removes, by construction:
  - the O(1) low-speed radiation error (L1);
  - the 1.39x section excess (MLM);
  - the first-order transom and CP error (C2).
- What remains: the O(N C) cross terms and the C4 intermediate-Fn_c law.

## 6. Derived design

### D1 (recommended): matched composite, exact where linear

  F = F_HF(S_g; exact, no a_b)
    + F^3D,lin_rad(U; LTI at the calm equilibrium; active where tau_B < 1/4; 1-D table in U)
    + Delta F^3D_steady(lambda(t), (L_K-L_C)/b, s_T/b, C_V; table from the linear 3-D planing solver minus its own 2D+t limit)
    + sum over Lagrangian planes [ f^NL - f^lin ] dx,  with
        f^NL = MLM (chine dry, Vdot implicit) | post-chine law F_sep(h*, Fn_c) between the Wen g = 0 limit and the Fn_c -> 0 hydrostatic limit | exit with frozen m_a, no momentum term,
        f^lin = exact linearisation of that f^NL
    + friction (ITTC-57 on the dynamic wetted area incl. spray, register item)
    + sub-stepping of the strip during entries with t_cw < 5 dt; outputs passed through a stated low-pass filter.

Classification:
- **Exact by construction (verification only):**
  - F_HF;
  - the small-amplitude limit at every speed (L1, L4);
  - the gravity-free linear 3-D correction;
  - the constant-V chine-dry section (similarity).
- **Reductions with derived error orders:**
  - MLM for varying V: -2..+9 % on the section;
  - frozen LTI reference: second order at low speed;
  - O(N C) cross terms: about 2-10 % estimated, up to 20 % measured at Fn_c 0.2;
  - relative-velocity hypothesis: O(kc) in short waves (C1 covers its linear part at low speed only);
  - impact time resolution: below 2 % with sub-steps.
- **Need data (empirical, register per G3):**
  - the intermediate-Fn_c post-chine law;
  - transom ventilation at 6-14 kn;
  - friction C_f and spray area;
  - heel beyond beta;
  - lateral manoeuvring forces (no data).

Predicted errors against the targets of section 1:

| region | A1 R | A2 trim | A3 motion | A4 accel | basis |
|---|---|---|---|---|---|
| R0, 0-6 kn | +-5 % (friction and form) | — | equal to BEM in small waves | not impact-dominated | L1 by construction |
| R1, 6-14 kn | +-10-15 % (unverified) | +-1 deg | unverified | unverified | no small parameter; composite of limits |
| R2, 14-40 kn calm | +-5 % | +-0.4 deg at 20-40 kn; +-0.6 deg at 12-16 kn | — | — | 3.2 + 3.3: the remaining cross term at Fn_c of 0.2 |
| R2 regular head waves | — | — | +-20 %, except lambda/L of about 1 (O(kc)) | resonance: impact share kept if F_sep is right at Fn_c 0.5-1 | 3.4, 3.5, 3.7 |
| R2 irregular | — | — | — | +-30 % with filter (data: +-40 % from Tp) | 3.5 |

### D2 (cheaper): ZR plus MLM plus post-chine F_sep with limits plus low-speed LTI only

- Drops Delta F^3D_steady.
- Predicted trim errors: -0.1..-0.6 deg at 20-40 kn; R within -5..+2 % (the ledger values for ZR+MLM).
- At 12-16 kn it carries the 2D+t CP error, which predicts trim +0.25..+0.8 deg high. The ledger shows +0.3..+0.6.
- Cost about 2.0-2.3 ms; offline about 1-3 h.

### D3 (most exact, not feasible here)

- Offline 3-D nonlinear (CFD/VOF) tables of steady and unsteady forces over (U, z, theta, phi) plus impulse response.
- Needs a CFD install (user permission) and 10-100 h of compute per table axis. The attitude non-smoothness (3.6) would need fine grids.
- Not justified until D1's residuals are known.

## 7. Reconciliation with the ledger, and predictions made before implementation

1. **Savitsky R -11..-14 % at 20-25 kn (ZR).**
   - dR/dtau is 10-11 % per deg (cp_sensitivity.txt), and ZR's trim is 0.87 and 1.05 deg low. That accounts for -9.6 % and -11 %. **The resistance deficit is the trim deficit**, i.e. pressure drag W tan(tau); friction is a minor part.
   - The trim deficit is a lift-efficiency excess of k 1.28-1.5: Wagner 1.39 (3.4) times 2D+t 1.04-1.05 (3.2).
   - MLM leaves k 1.04-1.3 (trim -0.14..-0.56).
   - **Prediction:** adding Delta F^3D_steady to ZR+MLM changes trim by -0.81, -0.25, -0.01, +0.12, +0.19, +0.28, +0.30 deg at 12, 16, 20, 25, 30, 35, 40 kn (the negatives of the 3.2 row). The expected trim errors after that:
     - 25-40 kn: -0.27, -0.37, -0.21, -0.08 deg (30 and 35 kn still about -0.3);
     - 20 kn: -0.15;
     - 12-16 kn: -0.53, +0.09.
   - The residual at 25-35 kn is the cross term of 3.3. It is consistent in sign with B8 mlm_only over-lifting at Fn_c of 0.2.
2. **Fridsma 1969 accelerations.**
   - ZR (Wagner) matches the measured impact form factor at B lambda/L 1.5-3 (0.92-1.02). ZR+MLM loses a third of it (SR).
   - Derived reading: the Fridsma bow entries have model-scale Fn_c = V/sqrt(g c) of about 0.5-1 (c = 0.114 m, V about 0.5-1 m/s). At that Fn_c the post-chine force is far from its Fn_c -> 0 limit. Its absence (frozen rule) was hidden by Wagner's 1.39x chine-dry excess.
   - **Prediction:** MLM plus F_sep (with the Fn_c weighting) restores the resonance accelerations to about the ZR level, and leaves calm trims (Fn_c of about 0.2) almost unchanged. If the accelerations rise but the calm trims fall by more than 0.3 deg, then the Fn_c weighting is wrong; check C4 decides.
3. **Short-wave over-prediction (lambda/L = 1: heave 1.3-1.5x, accelerations 1.08-1.75x).**
   - Derived: the relative-velocity hypothesis error is O(kc) = 0.63 there.
   - **Prediction:** the 3-D linear forward-speed excitation at Fridsma lambda/L = 1 is 20-40 % below the strip's FK plus relative-velocity excitation at small amplitude. If not, the excess is nonlinear: exit suction (SM risk 5) or bow-entry geometry.
4. **Wedge drop post-chine.** Fn_c = 3.7 gives the g = 0 limit, where Wen applies (SM: 1.11-1.19 x measured with f3D). The same law at Fn_c of 0.2 must reduce to near zero momentum force (3.3), so steady planing is unaffected.
5. **Low-speed vertical response against BEM** (A33 1.74x, omega_n 3.07 vs 3.94, resonance 5.6x, zeta 0.3 missing).
   - These follow exactly from L1 being violated: impact-type pile-up used for oscillation (1.50x), strip instead of 3-D (1.16x), and no radiation damping.
   - **Prediction:** D1 reproduces the BEM RAO to the BEM's own accuracy (1-3 %) at small amplitude. In large short waves at rest, the nonlinear increment adds bow-emergence forces that BEM cannot check (no data).
6. **Long-wave accelerations (lambda/L = 6: measured 1.5-1.7x the sinusoid, models 1.0-1.15).**
   - Motions are matched, so the excess must be higher harmonics. No mechanism in 3.1-3.9 predicts a 50 % non-sinusoidal excess at kc of 0.1 with matched motion.
   - Not explained by the derivation; possibly measurement (hand-read oscillograph). Data-limited.
7. **Fridsma A calm trim +1.8 deg.**
   - Fridsma A has C_V 2.66, lambda of about 3.7 and static/dyn of about 0.9: strong gravity.
   - The gravity-free 3-D term predicts only +0.39 deg (cp_sensitivity.txt). The rest must come from the gravity side: a_b = 0.5 and gravity-free dynamic sections at gamma of about 1.
   - **Prediction:** removing a_b (full hydrostatics) and adding the 3-D gravity linear correction changes case A by O(1 deg). The sign cannot be derived without the B7-gravity table.
8. **Fridsma 1971 irregular (0.61-0.81 of measured).** The Tp uncertainty (+-40 %), seed scatter (2-4 counts) and filter dependence (3.5) are each larger than the differences between designs. This dataset cannot discriminate designs below about 30 %; use it only as a gross check.

## 8. Decisive checks, ordered by information per cost

| # | check | type | cost | confirms or refutes |
|---|---|---|---|---|
| C1 | D1 at U = 0, a = 1 cm: heave/pitch RAO and decay against BEM (rest_rao cases) | math (by construction) | about 1 h incl. implementation of the LTI piece | the linearisation-consistency condition (4.5 d); failure means double counting |
| C2 | Apply Delta F^3D_steady (gravity-free VLM table) to ZR+MLM calm equilibria. A static moment/lift correction suffices before any plant change. Compare with Savitsky at 12-40 kn | numeric | about 1 h | 3.2. If 25-40 kn trims move by the predicted +0.12..+0.30 deg and the 12-16 kn ones by the predicted negative amounts, the 3-D linear term is confirmed; the residual size then measures the cross term |
| C3 | Rerun B8 mlm_only at 6-8 cases that break the Fn_c/C_V confounding (e.g. C_V 6 with tau 1.5-2; C_V 2 with tau 5-6) | numeric | about 1 h compute | 3.3: whether the cross term follows cross-plane gravity (Fn_c) or 3-D gravity (C_V). That decides whether C4 needs a 2-D gravity solver or the 3-D linear gravity table suffices |
| C4 | Tveitnes 2008 constant-V entry (Fn 0.3-0.7) post-chine force against Wen | data | about 1 h (digitise) | the Fn_c -> 0 limit of F_sep (3.3) |
| C5 | Strip excitation against 3-D linear forward-speed excitation at Fridsma lambda/L 1-2, small amplitude | numeric | about 30-60 min | 3.7 O(kc) (prediction 7.3) |
| C6 | Impact sub-stepping: Fridsma B lambda/L 3 and 1971 cond 30, dt_sub = dt/1..10, fixed output filter (e.g. 20 Hz full scale) | numeric | about 30 min | 3.5; makes A4 well-defined |
| C7 | MLM + F_sep (two limits, simple Fn_c weighting as a registered assumption) on Fridsma 1969 resonance and the drop test | numeric plus data | about 1-2 h | prediction 7.2 |
| C8 | 2-D nonlinear BEM with gravity (only if C3/C4 say cross-plane gravity matters) | solver | several 1-2 h stages, research risk | replaces the assumed F_sep weighting with an exact table |

Ordering rationale: C1-C3 are cheap and each can falsify a structural claim of the design (consistency, 3-D term, which gravity group). C4-C6 test single components. C7-C8 are expensive and should be done only when C3/C4 point there.

## 9. What is derived, what is assumed (honest limits)

- **Derived:**
  - the error orders of 3.2 (numerically confirmed);
  - the two limits of the post-chine force (3.3);
  - the exact split and the a_b argument (4.1);
  - the composite's exactness conditions and its O(N C) error (4.5);
  - the minimal-state argument from tau_B > 1/4 (4.2);
  - the time-resolution numbers (3.5).
- **Assumed or estimated:**
  - N of about 0.3-0.5;
  - the trim predictions use Savitsky's equilibrium as the sensitivity model (Savitsky is a regression of measurements);
  - the delta-planform VLM values carry about +-5 %;
  - the intermediate-Fn_c law is unknown;
  - the Fridsma Fn_c of about 0.5-1 is an order-of-magnitude estimate.
- **Not covered:**
  - lateral, roll and yaw hydrodynamics beyond hydrostatics (no data; the same composite structure applies, with per-side sections);
  - large heel beyond beta;
  - re-entry into the hull's own spray and wake;
  - R1 (6-14 kn), which has no small parameter.
- **Time used:** about 2 h. Compute was under 2 min in total: regime 26 s, lattice 43 s plus checks, sensitivity under 1 min. The rest was reading and derivation.
