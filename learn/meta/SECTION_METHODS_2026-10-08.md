# 截面水动力方法调研（2026-10-08）

## 中文摘要（给读者的结论）

**问题**：船体每个横截面在各种状态下受多大的水作用力。这些状态包括：折角线（船底与船侧的交线）浸湿前的入水、折角线浸湿后水流从折角处甩出去的阶段、截面出水（包括船首飞起后）、横倾或横向速度下的不对称入水、入水速度变化、稳态滑行、低速时绕水线的小幅振荡，以及船侧和甲板浸水。问题是有没有一种经过试验检验的方法能覆盖全部这些状态。

**结论一：没有一种在线可用、又经过检验的方法能覆盖全部状态。** 能在原理上覆盖大多数状态的只有一类：二维完全非线性势流数值解。它带重力，并在折角处规定水流沿切线离开。挪威科技大学（赵荣、Faltinsen、孙辉）和意大利 CNR（Iafrati、Del Buono）都做过这类解，文献里也和落水试验、稳态滑行数据比过。它的问题有两个：每个截面每一步要解一个几百个单元的方程组，还要用很小的时间步追踪薄射流，比我们每步几毫秒的预算慢几百倍；而且没有公开代码。所以它只能离线算，把结果做成表。

**结论二：我们落水试验里"折角线浸湿后力几乎为零"的问题，已经有现成的、检验过的解法，仓库里也已经写好。** 王（Wen）等 2022 年的公式描述了折角线浸湿后截面力怎样从峰值逐渐降到稳定值，并考虑了速度变化。它是用两种独立数值方法的解拟合出来的，并和赵等 1996 年的落水试验比过。这个公式在仓库的三维兰金求解器里已经实现（`hydro/rankine/planing_nl.py`，以前的阶段 B8 用过），但还没有接进条带模型 `sim/planing_zarnick.py`。本次用一个临时小脚本，按实测速度过程把它套到同一个楔块落水上，折角线浸湿后 17.5–25 毫秒的结果是：
- 与论文的二维非线性解相差 4%–8%；
- 再乘上原论文给出的三维修正（落水模型没有端板，三维效应在折角处约使力减少两成），为实测的 1.1–1.2 倍。

以前"冻结"规则在这段只给出实测的约两成。仓库里实现过的另一种分离模型（Tassin–Korobkin 的"零压力"延续）在阶段 B8 里已证明只有数值解冲量的两到三成，理由清楚，可以不再考虑。

**结论三：真正的难点是"落水"和"稳态滑行"之间的矛盾，这在文献里也存在。** 同一个折角线浸湿后的力，在落水上是对的，放到稳态滑行里却会让升力过大。仓库阶段 B8 已经量过：升力比 Savitsky 大两到五成半。粗算我们的船在 30 节时，这部分力约为船重的三成，所以纵倾会明显偏低。文献给出两种解释，两者都要：
- **重力**：稳态滑行时截面下沉速度很慢（我们的船 30 节时不到 1 米每秒）。按截面宽度算的弗劳德数（速度平方与重力乘宽度之比的平方根，衡量重力影响大小）只有约 0.2–0.7，重力起主要作用。落水试验约为 3，重力可以忽略。王等的公式不含重力。
- **船尾三维效应**：船尾处压力必须回到大气压。Garme 2005 年用经验函数削减船尾附近的升力，孙辉与 Faltinsen 用的是约半个船宽长度上的削减。Garme 的函数有拟合常数，与"不打补丁"的原则冲突。

**结论四：其余状态的现成做法。**
- **出水**：Tveitnes 等的匀速出水试验表明，出水时的力很小，方向与运动相反，像阻力。如果把"附加质量变化率乘速度"这一项用在出水上，会给出顺着运动方向的力，这是错的。我们的条带模型只在入水时计这一项，与试验一致。Tavakoli 等 2020 年发现，条带法在短波里高估加速度，原因是出水时船首截面出现了负的力。这与我们 Fridsma 短波项偏高的现象一致，值得专门检查。
- **不对称入水**：Toyama 1993 年和 Xu、Troesch、Vorus 1998 年给出两侧分别计算的入水模型；Hascoët 等 2019 年的模型还能处理只有一侧发生分离的情况。Judge 等 2004 年的试验只在不对称较小时与理论一致。横倾超过斜升角后，没有检验过的解析方法。
- **低速、绕水线振荡**：冲击类理论不适用。这时应该用线性辐射力，包括频率相关的附加质量和阻尼，以及记忆效应。仓库里已有三维面元和兰金求解器的结果可用。正确的组合方式是"三维线性解加上截面非线性增量"，阶段 B8 已用过这一思路。
- **甲板浸水**：对这条船意义不大，可以只算静浮力。

**建议的步骤（按先后）**：
1. 把王等的折角线浸湿后公式接进条带模型，作为一个开关，每个截面带少量记忆量（是否已浸湿折角、到过的最大深度）。然后同时检查落水、Savitsky、Fridsma 1969 和 1971。预计落水通过、滑行纵倾偏低，但这一步便宜，结果是决定性的。约 1–2 小时，其中计算约半小时。
2. 读取 Tveitnes 试验的曲线。这组试验的弗劳德数正好落在稳态滑行的范围，可以直接检验"重力削弱浸湿后力"的假设。约 1 小时。
3. 若第 2 步支持重力解释，再离线做一张"带重力"的截面力表，有两条路。一是写二维非线性势流求解器，研究风险最大，要多个阶段，每个阶段一到两小时，能否稳定算出射流是主要不确定性。二是用开源 CFD（OpenFOAM），需要用户同意安装，每个工况几十分钟计算，几十个工况要十几到几十小时，而且一次只能跑一个。
4. 船尾三维效应：先试用仓库三维线性解里已有的船尾条件推导削减量，不用拟合常数；不行再讨论是否接受 Garme 的经验函数。

**单步耗时**：以上都是查表或闭式公式，每个截面几微秒，加在现在的 1.85 毫秒上仍在几毫秒以内。

---

## English report

### 0. Scope and what was read

Question: which established, validated methods give the hydrodynamic force on a 2-D (2D+t) hard-chine section in the eight states our plant sees: (1) chine-dry entry, (2) chine wetting and the separated phase after it, (3) exit / withdrawal incl. after flight, (4) asymmetric / oblique entry (heel, sway, roll), (5) varying entry velocity incl. deceleration, (6) the steady planing limit (Savitsky), (7) low speed / oscillation about a fixed waterline, (8) topsides / deck immersion.

Read in full or in substantial part (open sources): Zhao, Faltinsen & Aarsnes (ZFA, 21st ONR Symp. 1996/97, NAP chapter 29); Tveitnes, Fairlie-Clarke & Varyani 2008 (Ocean Eng. 35, accepted manuscript, Strathprints 13792, full text); Wen, Liu, Del Buono, Qu & Iafrati 2022 (arXiv 2201.00959, J. Fluids Struct.); Hascoet et al. 2019 (arXiv 1908.09201); Del Buono, Bernardini, Tassin & Iafrati 2021 (arXiv 2103.15132); Del Buono, Iafrati & Bernardini MARINE 2021. Abstract / secondary only: Tveitnes 2001 PhD thesis (Glasgow; the PDF exceeds the fetch limit), Garme 2005 (ISP 52(3)), Tavakoli et al. 2020 (Ocean Eng. 217, 107959), Haase, Soproni & Abdel-Maksoud 2015 (STR 62), Xu, Troesch & Vorus 1998 (JSR 42), Judge, Troesch & Perlin 2004 (J. Eng. Math. 48), Semenov & Iafrati 2006 (JFM 547), Semenov & Wu 2016, Tassin, Korobkin & Cooker 2014 (AOR 48), Qin, Zhao & Shen 2011, the Dashtimanesh/Tavakoli/Bilandi 2D+T papers, Ghadimi et al. 2016 (PMR), the TalTech wedge dataset record. **Not verified online** (cited from memory, flagged where used): Sun & Faltinsen 2007 (J. Eng. Math., gravity in 2D+t planing), Sun & Faltinsen 2011 head-seas paper (JMST), Vorus 1996 details, Toyama 1993, the exact form of Garme's reduction function. DTIC and Springer refused access.

Repository context used: `learn/meta/SIM_REDESIGN_2026-10-08.md` (progress 1 and 2), `SIM_DATA_LEDGER_2026-10-08.md` A3-A6, `RANKINE_STAGE1_2026-10-06.md` section 14 (stage B8: MLM plus the Wen et al. separated-flow similitude inside the 3-D linear planing solver, NO-GO on Savitsky lift), `hydro/rankine/planing_nl.py`, `hydro/rankine/planing_wedge2d.py`, `studies/exp_slam_wedge.py`.

### 1. Two facts that change the picture

**F1. The post-chine-wetting drop-test force is reproduced by an existing, validated formulation that is already in the repo.** Wen et al. 2022 give the separated-phase force of a wedge as a decay from the force at chine wetting to the steady free-streamline (supercavitating) force, with one universal decay function. Here h is the penetration, h_cw the penetration at chine wetting, l the side length and V the entry velocity:

  F = rho V^2 l [Cs_inf + C*(h*) (Cs_cw - Cs_inf)] + (rho/2) c^2 (dV/dt) Ca,
  h* = ((h - h_cw)/h_cw) cot(beta)^1.3075,  C* = (1.539 h* + 2.618)/(h*^2 + 8.081 h* + 2.169),
  Ca = 1.0459 in the slamming stage, rising to about 1.33 (+27 %) within h* = 0.067 of chine wetting.

It is fitted to FVM-VOF solutions of the gravity-free nonlinear problem and checked against the BEM-FEM of Iafrati & Battistin, for deadrise 5-70 deg (constants from 15-45 deg). The authors compared it with the ZFA 30 deg drop test after a 3-D correction, f3D = 1 - 0.80 (c/L)^2. ZFA themselves state that 3-D effects (no end plates, B/L = 0.5) reduce the measured force by about 20 % by the time the spray roots reach the knuckles. `hydro/rankine/planing_nl.py` already implements this force (empirical element E-B8-1), anchored on the MLM force at chine wetting.

Scratch check (`<scratchpad>/wen_zfa.py`, not in the repo): Wen post-chine plus the acceleration term, with the measured V(t) of the ZFA drop and the 0.2 m section.

| t ms | measured N | ZFA nonlinear 2-D | Wen (velocity term) | Wen + Ca term | Wen + Ca, x f3D |
|---|---|---|---|---|---|
| 17.5 | 3700 | 5000 | 5753 | 5205 | 4164 |
| 20 | 2600 | 3500 | 4121 | 3772 | 3017 |
| 22.5 | 2300 | 3100 | 3381 | 3181 | 2545 |
| 25 | 1900 | 2800 | 2970 | 2837 | 2270 |

- Against the 2-D nonlinear solution: +4 % to +8 %.
- With the 3-D factor: 1.11-1.19 x measured (digitisation +-250 N; the ledger's own rig-deceleration cross-check says the digitised curve itself may be 6-11 % high).
- The plant's frozen rule gives about 0.2 x measured; the Tassin-Korobkin zero-pressure continuation gives 470-862 N. In B8 (rows W2R) that continuation had only 0.21-0.30 of the CFD impulse, because a Wagner-type potential has no steady separated-flow pressure. It should not be revisited.

Consequence for the W1 gate: a 2-D section should be compared with the ZFA nonlinear 2-D curve, or with measured / f3D, not with the raw measurement. Against the nonlinear 2-D curve the MLM chine-dry force is 1.24 / 1.18 / 1.11 / 1.01 / 0.94 at 2.5 / 5 / 7.5 / 10 / 15 ms.

**F2. The separated force that fixes the drop test over-lifts steady planing. This is a known conflict and is already measured in this repo.**
- B8 put MLM plus Wen into the 3-D linear planing solver. Lift came out 25-55 % above Savitsky at 20 deg, trims 0.6-3.4 deg low and wetted lengths 15-31 % long.
- Rough estimate for our boat at 30 kn (tau about 2.2 deg, V_n = U sin tau about 0.6 m/s, l = 1 m): the Wen separated force on a 3 m chine-wet run is about 4 kN, roughly 30 % of the weight. The current ZR plant has 0 there, apart from cross-flow drag of about 1 kN.

The literature gives two mechanisms, and both are needed:
- **(a) Gravity in the cross-plane.** The cross-flow Froude number V_n / sqrt(g l) is about 0.2-0.7 for chine-wet sections in steady planing (0.19 for our boat at 30 kn; 0.2-0.7 in B8's Savitsky set) against about 3 in the ZFA drop. The gravity-free cavity that carries Wen's steady force cannot form at Fn of order 0.3. Sun & Faltinsen (2007, from memory) ran 2D+t nonlinear BEM with gravity and chine separation for exactly this reason.
- **(b) 3-D transom relief.** The pressure must return to atmospheric at the transom.
  - Garme 2005 uses an empirical reduction function on hydrostatic and hydrodynamic terms. It was fitted to near-transom pressure measurements and to running-attitude data, and validated in calm water and in regular and irregular waves.
  - Sun & Faltinsen used about 0.5 b (B8 note).
  - Both are fitted, which conflicts with the project's no-patches rule.
- Tveitnes's tests run at Fn 0.28-0.69 (V 0.24-1.19 m/s, half-beam 0.3 m), i.e. in the steady-planing Froude range. Their post-chine force reaches the flow-momentum level by z/d of about 2. By z/d = 2 at 30 deg, Wen still keeps about 17 % of the excess (Cs about 1.5 against Tveitnes's Cfm of about 0.4-0.5). So these data can test (a) directly; see step 2 in section 5.

### 2. Comparison table

Key to state coverage: Y = covered and validated for that state; y = covered, little or indirect validation; ~ = partial or heuristic; - = no. "Memory" means the history the method needs per section.

| Method | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | Memory | Cost per section-step / tabulable | Validation (accuracy) | Open code | Known failure modes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Wagner / von Karman momentum, Zarnick-type (current v0, ZR) | ~ (1.9x at 30 deg) | - (frozen beam gives 0) | ~ (momentum term on entry only) | ~ (Toyama-type extension) | y (m dV/dt) | ~ (needs Zarnick coefficients, cross-flow) | - | - | none (h only) | ~1 us; trivially | Zarnick / Fridsma via empirical coefficients (A3) | in repo | over-predicts at 20-30 deg; no post-chine force; spurious exit force if momentum is kept |
| MLM, Korobkin 2004 (repo `planing_wedge2d`) | Y | - | ~ via modified von Karman exit (Tassin et al. 2013) | y (Qin, Zhao & Shen 2011) | Y | chine-dry part only | - | - | h (c follows Wagner) | closed form, ~us | similarity solution -1.7 / +2.1 / +9.1 % at 10 / 20 / 30 deg (repo W2C); peak pressure too high | in repo | negative-pressure zone ignored; exit semi-analytical weaker than nonlinear (Del Buono 2021) |
| Logvinovich zero-pressure / FBC, Tassin-Korobkin-Cooker 2014; asymmetric Hascoet et al. 2019 | Y | ~ | - | y (partial separation) | y | - | - | - | wetted widths per side, separation abscissae; Newton per step | 10-100 Newton iterations (needed under-relaxation 0.1) | Hascoet: inclined plate within ~10 % (theta 5-30 deg) vs nonlinear self-similar; NACA foil vs CFD good for Fy, Mz; repo: wedge transition impulse 0.21-0.30 of CFD | no | no steady separated-flow force; continuation angle chosen a priori (45-55 deg) |
| Wen et al. 2022 similitude (repo `planing_nl`) | Y (anchor) | **Y** | - | - | y (Ca term, checked vs BEM-FEM) | over-lifts if used alone (B8) | - | - | h relative to h_cw, flag "chine wetted" | closed form, ~us | FVM-VOF and BEM-FEM 5-70 deg (few %); ZFA drop with f3D; scratch: 4-8 % vs ZFA nonlinear 2-D | in repo | gravity-free; symmetric; constant-V decay only; fit range h* <= 7 |
| ZFA 1996 nonlinear BEM (Kutta at knuckle, jet cut) | Y | Y | - | rig rolls to 28 deg, results not shown | Y (measured V) | - | - | y (any shape) | full free surface | 100s of panels, small dt; offline | wedge 30 deg and bow-flare drop tests "satisfactory" over the whole entry incl. separation; later differences put down to 3-D effects | no | gravity neglected; jet instability handled by cutting |
| ZFA generalised Wagner (simplified) | Y | - (stated) | - | - | y | - | - | - | h | fast | "reasonable approximation" of force before separation | no | no separation |
| Zhao & Faltinsen 1993 similarity | Y (constant V) | - | - | - | - | - | - | - | none | table in beta | exact gravity-free reference | values in repo CSV | constant V only |
| Sun & Faltinsen 2D+t nonlinear BEM with gravity (2006-2012; from memory) | Y | Y | y | y (2009 roll-angle section) | Y | y (with ~0.5 b transom correction) | ~ (gravity radiates in 2-D) | y | full free surface per plane | offline only (est. 1-8 s per 0.02 s step for 20-40 planes) | drop tests, Savitsky; head-seas comparison not verified here | no | 2D+t limits (low Fn, transom) |
| Del Buono / Iafrati hybrid BEM-FEM with gravity (2021-2022) | Y | Y | **Y** | - | Y | - | - | y | full free surface | offline; jet model reduces cost | cone entry and exit vs Breton et al. 2020 experiments "very good"; wedge vs CFD | no | separation criterion for smooth bodies still under development |
| Vorus 1996 flat cylinder; Xu-Troesch-Vorus 1998 asymmetric | Y | y (separation at chine) | - | y (vertical velocity, asymmetric) | y | y (steady planing resistance) | - | - | free-vortex-sheet state | moderate; tabulable | drop tests (Kim, Vorus, Troesch, Gollwitzer); Judge 2004: good for small asymmetry and small Vh/Vv; Tveitnes: Vorus added mass closer to experiment than Z&F | commercial (VsSeaKeeper) | large horizontal velocity: keel separation |
| Toyama 1993 / Algarin & Tascon 2011; Dashtimanesh, Tavakoli, Bilandi 2D+T (heel, oblique sway-roll-yaw) | Y | ~ (empirical) | ~ | y | y | y (empirical transom) | - | - | wetted widths per side | closed form | calm-water heel vs CFD; sway / roll / yaw vs experiments; Tavakoli 2020: short-wave accelerations over-predicted (negative bow forces at exit), moderate / long under-predicted | no | empirical pieces |
| Semenov & Iafrati 2006; Semenov & Wu 2016 (integral hodograph) | Y (asymmetric, exact) | y (2016: expanding wedge with detachment) | - | Y | - | - | - | - | none (self-similar) | integral equations; table in (beta, heel) | exact gravity-free | no | infinite wedge (2006); self-similar only |
| Tveitnes / Fairlie-Clarke 2008 (+2001 thesis): added-mass momentum + flow-momentum drag + residual gravity | Y | Y (added mass +20-80 % after chine, saturating by z/d about 2-3) | Y (exit force low, drag-like, Cex 1.0-1.3) | - | y (thesis: non-constant V) | y (thesis: "consistent with published planing data", plus aspect-ratio correction) | y (wet-chine oscillation: added mass coefficient 1.2 / 1.0 / 0.8 at 0 / 15 / 45 deg) | - | h, max h | closed form | own constant-V entry, exit and oscillation tests (Fn 0.3-0.7), noisy, fitted through a rig simulation; CFD | no | considerable dynamic noise after chine wetting; possible hydroelastic effects |
| Garme 2005 transom correction | (a 3-D correction, not a section model) | | | | | Y | | | none | free | calm water, regular and irregular waves (own tests) | no | fitted function |
| Haase, Soproni & Abdel-Maksoud 2015 (2D+t BEM, head waves) | y | y | y | - | y | y | - | - | per plane | offline-class | forces and accelerations vs experiments, with a 3-D correction (details not read) | no | 3-D correction needed |
| CFD tabulation (FVM-VOF, SPH; e.g. OpenFOAM interFoam) | Y | Y | Y | Y | Y | (2-D) | y | Y | none online (table) | tens of minutes to hours per 2-D case | Wen 2022 and Fairlie-Clarke 2008 used it | open (needs install) | air / compressibility at small deadrise; mesh at chine |
| Linear strip / 3-D radiation with memory (Cummins; repo BEM / Rankine) | - | - | - | y | - | - | **Y** | - | radiation states (convolution / state-space) | small | repo A8-A11, B1 | in repo | small motions only |

### 3. Which method covers the most states

- **One formulation:** the 2-D fully nonlinear potential-flow solution with gravity and a Kutta condition at the chine (ZFA; Sun & Faltinsen; Del Buono / Iafrati).
  - Covers states 1, 2, 3, 4, 5 and 8 directly. State 6 needs a 3-D transom correction. State 7 is covered only in the 2-D strip-theory sense.
  - Validated on drop tests (ZFA wedge and bow flare), on entry and exit of a cone (Breton et al. 2020, through Del Buono 2021), and on Savitsky with a transom correction (Sun & Faltinsen, from memory).
  - It cannot run online: it is estimated 100-1000x over our budget, and no open code was found. It is the right tool to generate tables offline.
- **Practical combination for the plant (most states with validation per piece):**
  1. Chine-dry: MLM, per side for heel (in repo).
  2. Separated phase: Wen 2022 decay with its Ca term (in repo), to be replaced or extended by a gravity-dependent table when one exists.
  3. Exit: no momentum term. Inertial term with the added mass frozen at its maximum (modified von Karman exit, Tassin et al. 2013), plus a drag opposing the motion (Tveitnes Cex 1.0-1.3). Gravity matters (Del Buono 2021).
  4. Steady limit: Savitsky check. A transom relief is needed: either derived from the repo's 3-D linear solver, which already holds the transom Kutta condition, or Garme's function (fitted).
  5. Low speed and oscillation: composite. The 3-D linear radiation with memory, plus the sum over sections of [nonlinear section force minus its linear 2-D counterpart]; this is the B8 composite applied in the time domain. Tveitnes's oscillation tests show that the wet-chine oscillation added mass differs from the entry added mass (up to +60 % at 45 deg), so a single depth-indexed added-mass curve cannot serve both.
  6. Asymmetric: wetted widths per side, with the partial-separation logic of Hascoet et al. Exact self-similar asymmetric solutions (Semenov & Iafrati) serve as verification for the chine-dry part.

### 4. What implementing it in our plant takes

- **Memory per water plane.** Use Lagrangian planes, i.e. states advected aft at the forward speed (d/dt - U d/dx on hull stations), because a hull station sees a different water plane every step.
  - State per plane: phase flag (dry / chine-dry / separated / exiting), h_max (or c_max per side), h_cw per side, and the plane's age since it was last dry.
  - That is about 20-40 stations x 2 sides x 3-4 numbers.
  - Reset when the plane is fully dry, which covers flights.
- **Tables.**
  - Now: none needed (closed forms).
  - Later: a gravity table Cs(h/l, Fn_c, beta_eff), and for heel Cs(h/l, Fn_c, beta, theta, Vy/Vz). About 40 x 8 x 6 x 9 x 7 entries, a few coefficients each, a few MB. Lookup cost is microseconds per section.
- **Cost per 0.02 s step.** Vectorised closed forms or lookups take an estimated 50-200 us on top of the measured 1.85 ms (ZR), so the step stays within a few ms. Avoid per-section Newton solves online (FBC needs them); tabulate instead.
- **Agent wall-clock.** Estimated from the last comparable steps (1-2 h each, compute in minutes to about 25 min):
  1. Wen switch in `sim/planing_zarnick.py`, with per-plane flag, h_max and the Ca term, then drop test, Savitsky 10-40 kn, Fridsma 1969 / 1971: about 1-2 h, compute about 30 min. Expected: W1 post-chine passes against measured / f3D; Savitsky trims low (B8 analogue). Cheap and decisive.
  2. Digitise Tveitnes Fig. 4 / 16 (constant-V entry, Fn 0.3-0.7) and compare with Wen: about 1 h. Tests the gravity explanation with open data.
  3. Gravity table, two routes:
     - (a) A 2-D nonlinear BEM with gravity, chine Kutta and jet cut in `hydro/`. Bottleneck is research risk (jet stability, separation); several 1-2 h stages, outcome uncertain.
     - (b) OpenFOAM 2-D VOF. Needs the user's permission to install; about 10-60 min compute per case and about 50 cases, so roughly 10-40 h of compute, one job at a time on this 16 GB machine.
  4. Exit / flight and asymmetric pieces: about 1-2 h each.
  5. Transom relief derived from the 3-D linear solver: 1-2 h, with the outcome uncertain.

### 5. Validation data (open status)

| Data | What it gives | Open? | Caveats |
|---|---|---|---|
| ZFA 1996 (NAP open) | 30 deg wedge (B 0.5 m, 0.2 m measuring section, 241 kg rig) and bow-flare force histories through and after chine wetting; pressures; measured V(t) | yes (already digitised, ledger A6) | no end plates, about 20 % 3-D reduction at the knuckle (use f3D or the 2-D curve); digitisation +-250 N |
| Tveitnes et al. 2008 accepted manuscript (Strathprints 13792) | constant-V entry force vs z/d up to 4-5, beta 5 / 10 / 15 / 30 / 45 deg, V 0.24-1.19 m/s; constant-V exit (Cex table); wet-chine oscillation added mass | yes (figures only) | heavy dynamic noise after chine wetting; values extracted through a rig simulation; low speed |
| Tveitnes 2001 PhD thesis (Glasgow) | entry and exit theory, planing-lift check | yes (PDF > 10 MB, not read here) | — |
| Wen et al. 2022 (arXiv) | CFD / BEM transition curves, constants | yes (constants in `data/external/wedge_entry/`) | computations, not measurements |
| Del Buono et al. 2021 (arXiv / Archimer) | wedge entry and exit (computed); cone entry and exit vs Breton et al. 2020 experiments | yes | cone, not wedge |
| Hascoet et al. 2019 (arXiv) | asymmetric loads vs CFD and the inclined-plate solution | yes | computations |
| TalTech wedge slamming dataset (DOI 10.48726/j0mys-2dr75, CC BY 4.0, 1.7 GB) | symmetric drops 0.25-2 m, asymmetric heel 5-25 deg; accelerations, pressures, strains, video | yes (download needs the user's permission) | hydroelastic aluminium, 3-D non-prismatic; chine wetting not confirmed |
| Fridsma 1969 / 1971 (Internet Archive) | regular and irregular head seas | yes (in use) | 1971 modal period unknown |
| Savitsky 1964 | steady planing | yes (in use) | regression |
| Begovic et al. 2014 / 2016 (Naples prismatic series, irregular head seas) | heave, pitch, CG and bow accelerations | possibly (core.ac.uk record) | not checked |
| Judge et al. 2004; Peterson, Wyman & Frank 1997 (NSWC drop tests of planing sections) | oblique / asymmetric drops; planing-section drops | unclear / DTIC blocked here | — |

### 6. Risks

1. **Steady-planing over-lift** from the separated force is the main risk; it is already measured (B8). It goes away only if gravity (Fn_c 0.2-0.7) and 3-D transom relief are both modelled. A fitted transom function (Garme, Sun & Faltinsen) conflicts with the no-patches rule; a derived relief is unproven.
2. **The Wen fit is gravity-free, symmetric and monotonic-entry.** Its use with varying V, partial exit and re-entry, or heel has no validation.
3. **Drop-test reference quality:** ZFA has 3-D effects and digitisation error; Tveitnes is noisy and possibly hydroelastic.
4. **Large heel (theta > beta = 20 deg)** gives one side near-zero or negative deadrise: Wagner-class theories break down there (flat impact, air cushion), and no validated analytical method exists.
5. **Exit:**
   - Semi-analytical exit is less accurate than nonlinear.
   - With large frozen added mass and strong deceleration, the inertial term gives large suction; the nonlinear wedge without gravity shows a negative peak larger than the entry peak.
   - Tavakoli et al. 2020 trace the 2D+t over-prediction of short-wave accelerations to negative bow section forces at exit, which matches our Fridsma short-wave excess (heave 1.3-1.5x, accelerations 1.08-1.75x). Check the sign of the bow section forces in those runs.
6. **2D+t itself:** Tavakoli et al. also report under-prediction at moderate and long waves, like our long-wave residual. Part of our residual may be a 3-D limit of 2D+t rather than a section-model error.
7. **Low-speed composite** (linear radiation plus nonlinear increment) is not validated for planing hulls at transitional speeds; there is a risk of double counting the momentum term against radiation damping.
8. **Re-entry after flight** into disturbed water (spray, the plane's own wave) is not modelled by any of the section methods.

Sources: [NAP ZFA chapter](https://www.nationalacademies.org/read/5870/chapter/29) · [Tveitnes et al. 2008, Strathprints](https://strathprints.strath.ac.uk/13792/) · [Tveitnes thesis](https://theses.gla.ac.uk/2890) · [Wen et al. 2022](https://ar5iv.labs.arxiv.org/html/2201.00959) · [Hascoet et al. 2019](https://ar5iv.labs.arxiv.org/html/1908.09201) · [Del Buono et al. 2021](https://ar5iv.labs.arxiv.org/html/2103.15132) · [Del Buono et al. MARINE 2021](https://journals.ed.ac.uk/MARINE2021/article/view/6808) · [Garme 2005, IOS](https://content.iospress.com/articles/international-shipbuilding-progress/52-3-1) · [Tavakoli et al. 2020, IRIS](https://www.iris.unina.it/handle/11588/876769) · [Haase et al. 2015, TORE](https://tore.tuhh.de/entities/publication/8bc625aa-907e-41e7-a21b-15a56ab52ab4) · [TalTech wedge dataset](https://data.taltech.ee/records/j0mys-2dr75) · [Tassin et al. 2014, UEA](https://ueaeprints.uea.ac.uk/52473) · [Semenov & Iafrati 2006, CNR](https://iris.cnr.it/handle/20.500.14243/15169)
