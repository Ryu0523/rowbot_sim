# Training distribution over model errors: derivation and spec

Produced 2026-09-29 by a derivation workflow (three independent derivations -
system identification, stochastic processes, ML generalisation - merged, then
checked for mathematical correctness and implementability, then revised),
read and corrected by the main session. Plain-language summary and decisions:
DEFECTS.md L6. Nothing below is implemented yet.

Two measurements the main session added (scratchpad scripts, 2026-09-29):
- Real-gap (split C) error spectra vs the step-1 family (split A): the
  family puts 15-20% of the variance below 0.05 Hz, the real gap 1-5%; the
  real gap puts 45-66% above 0.8 Hz in sway, yaw and pitch (the family
  13-17%); pitch error is nearly white (lag-1 autocorrelation -0.03).
- 16 re-run split-C episodes with the 5x3 elevations recorded; causal ridge
  FIRs over 12 steps (2.9 s), fit on 8, scored on the other 8 (a measuring
  instrument only): unexplained fraction surge/sway/yaw/heave/pitch =
  waves only 0.69/1.04/1.10/0.75/0.70; states+commands 0.47/0.32/0.77/0.66/0.72;
  both 0.33/0.35/0.85/0.49/0.57. Small, linear and on the old sea, so only
  indicative: a lagged linear map of measured signals already explains a
  third to two thirds of four channels; yaw is not explained. (Kept as a
  record only. Under decision D1 below, fits like this are never used as a
  ceiling or as evidence of what the learned model can do.)

## Decisions after review with the user (2026-09-29). These override the text below.

**D1 — No reference-predictor checks.**
- Removed:
  - the P0 gate;
  - the T6 R2 ladder (and the memory, nonlinearity, own-past, causal-gap and unpredictable indices built on it);
  - T3 framed as 'predictability';
  - the operator-level fit;
  - any 'held-out reference beats rung (a)' rule.
- The user's rule: do not estimate the learned model's ceiling or the available predictability from weak or low-level models.
- Coverage is judged only with descriptive statistics of the data: T1 scale, T2 spectral shape (spectral flatness may be kept as a descriptor of shape), T4 wave coherence and phase, T5 excitation coherence, T7 tails, T8 non-stationarity, T9 cross-channel structure, T10 inputs, plus the event statistics of D2.
- Ability is judged by evaluating the trained model itself.
- P0 is now only the sea fix and the recorder extension.

**D2 — Events in v0.**
- The generic event component enters v0:
  - a random filtered signal (measured, or partly hidden) crosses a random high threshold;
  - the event size is a random heavy-tailed function of the filtered state at the crossing (e.g. its crossing speed);
  - the response is a random short impulse response.
- This goes together with the state-dependent-rate bursts of G9, whose timing depends on unmeasured detail.
- Rationale: the planing plant has impact terms. A slam lasts much less than one 0.24-s step, so in e it is an isolated one-step spike. The step average makes the observed operator continuous in the crossing time, so deterministic threshold events stay inside the fading-memory closure; steep thresholds need prior mass.
- No mechanism is named: the same component covers ventilation-type thrust loss and wave hits.
- Checks: event rate versus speed and heading, the share of sum(e^2) in the top 1% of |e|, the tail index, and clustering relative to the encounter period.

**D3 — Nonlinear latent decoder by default (route 2), no capacity experiment.**
- Prediction model: e = D(inputs along the plan, c, h_k, j).
  - c is a low-dimensional task latent. D is nonlinear in c (FiLM conditioning).
  - The first conditional flow samples c given the history.
- Reason: a linear-in-z basis covers curved task families, such as varying delays or sharp resonances at varying frequencies, only with a large K (Kolmogorov n-width). A nonlinear latent needs only a few dimensions, and the linear form is a special case of it.
- The linear Phi·z model stays as a baseline only.
- Training is two-stage:
  - auto-decode c* per training task on its noise-free continuations, jointly with D;
  - then train the flow on c* given the history prefix.

**D4 — The unpredictable part: three levels, and what c contains.**
- **c: the boat's rules, fixed or slow.** c holds:
  - the rules of the predictable part;
  - the statistics of the random part: its level, band and colour, how its level depends on the state, the event rate and the tail heaviness.
  - Uncertainty about c is epistemic. It shrinks with data and probing, and it is the only thing RL probing is rewarded for.
- **h_k: the current state of hidden processes.** Examples: a slow disturbance currently acting, a relay's current mode, the phase of an unmeasured wave group.
  - It changes over time and is estimated from the recent history (encoder context).
  - Its effect fades over the horizon on its own time scale.
  - It is not in c.
- **nu_k: innovations.** Randomness that nothing known can predict: waves not yet met (the largest source, under the convention eta_hat = 0 beyond the current step), unmeasured wave detail, spray, the exact timing of slams.
  - Only its distribution is known, and it is set by c and the state.
- **The noise head is replaced by a second conditional flow.** The Student-t / AR(8) head of the architecture section is replaced by a second conditional flow that generates the random part step by step inside the rollout, conditioned on (c, h_k, the rollout state, j). It can represent near-zero-plus-a-far-lobe distributions (slams), which a single-hump t cannot.
  - Common random numbers: the base noise for each (sample, step) is shared by all plans.
  - The flow is distilled to 1-2 integration steps for speed.
  - Its training data are M continuations per checkpoint with the SAME operator draw and DIFFERENT noise seeds. The spread within the same draw is the aleatoric distribution; the spread across draws given the history is epistemic.
- **MPC objective.** E_c E_nu[cost]. The impact term depends mostly on nu's tails.

**D6 — What the model predicts from (user decision, 2026-09-29): only the boat's own states, the commands and the history of the unexpected response e. No exogenous measurement such as the 15 wave elevations.**
- An unseen cause (an unseen sea, wind, a wake) need not be seen as an input. If it affects the boat, it shows up in one observable place: the error history and its relation to the states and commands.
- E[e_k | past e, states, commands] is defined whatever the causes are. The network therefore has to generalise over one space, the behaviours of the error process given the states, and not over all causes and sensors. The operator family randomises exactly that space.
- Extra measured inputs whose distribution can shift (the wave elevations) would bring back input-side enumeration. The real boat does not have them either.
- In the generator the wave filters remain causes. From the model's side they become unmeasured causes, predicted from the error's own history (h) and from the wave-driven states.
- Implemented as model2's waves_off (main); waves on is kept as a comparison. No data regeneration is needed.
- Still open: the MPC's own reduced model uses the current wave under the hull, which the real boat does not measure either (step 2).

**D5 — Implementation choices (2026-09-29, after two code reviews; DEFECTS M2).**
- **Operator inputs are 28:** S's 26 plus the thrust and nozzle commands applied during the step. S[5:7] holds the actuator states before the step. Errors tied to command changes (what a lagging actuator produces) come from random filters on these, not from a named lag term.
- **Actuator lag in the source world.** The source plant AND the nominal twin draw the same actuator lag w.p. 0.75 (lofi.PRIOR ranges). The inputs then show the actuator trailing its command, as in the target, and the lag itself is not an error term.
- **Projections.** Each input group's part is scaled to unit std on the library before the groups are mixed.
- **Events.**
  - Probability 0.5.
  - Size = expm1(b E)/expm1(b/2), where E is the crossing speed mapped to Exp(1) by its distribution at the library's own upcrossings. This gives a Pareto-like tail with index 1/b, b ~ U[0.15, 0.6], capped at 100.
  - A mean-one lognormal mark (s ~ U[0.3, 1.5]) w.p. 0.7.
  - Spike response w.p. 0.5. Responses act in the crossing step.
- **Bilinear term.** (a . filters)(b_c . direct), with its own b_c per channel. It is normalised separately and mixed in at beta ~ U(0, 0.5). It is present only when both groups exist.
- **Steep sigmoids.** The slope is measured per library std of the unit's input.
- **Clip.** The rule is clipped alone at 3 A_REF; the noise part takes the rest of the cut.
- **Slow gain and bias.** They drift on their own random stream.
- **Relays (split B only).** Band U[0.5, 1.5] std, so the remembered mode matters. Relays are held through a relabelled warm-up.
- **Relabelling.** Donors and the own future alike: 125-step warm-up, slow state frozen at the checkpoint, rule clipped alone. The push is mapped to e by stepping the source model with and without it.

**D7 — The actuator channels get their own random family (2026-09-30; DEFECTS M9, M10). Overrides D5's "actuator lag in the source world" for the meta3 data; meta2 keeps D5.**
- **Why.** The operator family acts on the 5 velocity channels only. The 2 actuator channels of the one-step model (e channels 8-9: actual minus commanded thrust and nozzle) had no random family: only the source plant's two built-in actuators (a pure thrust lag, a pure nozzle rate limit, each w.p. 0.75). M9 showed the rollout stall on the target (Cb) comes from actuator behaviour outside that family (a delay plus smoothing), while in-family rollouts handle full reversals. The gap is in the prior, so it is closed in the prior, generically, the way the operator family covers the hull.
- **The family** (sim/lofi.py ACT_FAMILY, draw_act; per channel, independent per episode). W.p. 0.2 a channel is ideal; otherwise each component independently:
  - delay U[0, 0.3] rs s, in whole plant steps (w.p. 0.5);
  - a zero-centred command dead zone U[0, 3%] of range (w.p. 0.3);
  - gain LogU[0.9, 1.1] with offset U[-3%, 3%] of range (w.p. 0.3);
  - backlash U[0, 3%] of range (w.p. 0.3);
  - a response time constant (w.p. 0.7): thrust LogU[0.05, 2] rs s, nozzle LogU[0.02, 0.5] rs s. Given a response: asymmetric w.p. 0.3 (tau x LogU[0.5, 2] towards lower values), second-order w.p. 0.3 (wn = 1/tau, zeta LogU[0.3, 1.5]: overshoot below 1);
  - a rate limit (w.p. 0.7): thrust full range in LogU[0.2, 3] rs s, nozzle LogU[0.25, 2] / rs rad/s.
  - rs = sqrt(L / 10 m). Every range rests on generic small-craft actuator scales (the source field of ACT_FAMILY), not on the target. The target's values lie inside every range (test_m10 test 4, on the values lofi.act_target_like returns: positions 0.14-0.76 of their ranges). The family has parts the target lacks (backlash, gain and offset, asymmetry, a second-order servo), so it is broader than "the target's chain with random numbers".
- **The update** (lofi.act_step, the ONE function the plant and relabel.simulate both call). The command passes through dead zone, gain and offset, clip to the travel, and backlash, which gives the target. First order: dx/dt = -clip(x / tau, +-rate), solved exactly over the step. The applied force is the exact time average, which is continuous in every input. The first draft's "exponential, then clip the step" jumped by rate dt / 2 at the limit. Second order: trapezoidal substeps, with the velocity clipped to the rate and the position to the travel.
- **Not modelled here.** Pump priming scales the jet's FORCE (sim/waterjet.py), not the actuator position. It lives in the velocity channels, i.e. the operators, and is outside D7. Measurement noise on the actuator position is also left out. It would break "the plant state is the record", which the relabelling and e0 rely on; it belongs in an observation model.
- **Data.** Plant and nominal twin get the same draw. The twin copies the plant's actuator hidden state (delay line, servo velocity) every control step, so the velocity errors do not absorb the actuator behaviour; it shows in channels 8-9. The draw uses its own random stream, so each meta3 episode has its meta2 twin's sea, scenario and operator; only the actuators differ. The operator library and the target splits are copied from meta2, which keeps the operator family identical. The replay tests are test_m10 tests 2-3: float64 data, <= 1e-9.
- **Reading the result (stated before the run).** The model already SEES the actual actuator positions (S[5:7]). So a remaining failure in the actuator channels is not a missing observation.
  - **At.** The test split At has A's episodes with target-shaped actuators and is never trained on. Its delay is lofi.target_delay_steps: the smallest whole number of source steps (0.04 s) not shorter than the target's 0.1 s, i.e. 3 steps = 0.12 s. 0.1 s is not a whole number of steps, so At's delay differs from the target's by up to half a step (here +0.02 s, the harder side).
  - **One stall measure** (studies/diag_stall.py, used by diag_m9_infamily for At and by diag_yaw_m8_blocks for Cb). A jump event can sit at any step of a plan. The actual nozzle must be at rest on the previous command before the jump. Events go into bins of 0.15-0.5, 0.5-0.8 and > 0.8 of range. "Stall gone" in a bin means the rollout's open fraction of the jump at step 5 is within 0.1 of the truth's. The branches are read per bin, on this same measure for At and Cb (same rollout code, 16 samples). eval3's actuator-channel skill (e 8-9) is a second check.
  - **Steady blocks.** A block is steady when, over the scored steps 0..9, the nozzle command is constant and the actual nozzle is at the command (start jump < 0.05, command step < 0.02, actual within 0.05). The first draft looked only at the start jump. That counted blocks whose command moved inside the window, and then the yaw error of the nozzle catching up, which the model does see, landed in the "steady" score. The diagnostic prints how many blocks the reading rests on. Below 30 steady blocks the branch is undecided, not "observation gap". The remedy is more data: split Cbh (meta_step3 cbhold) has target block episodes whose nozzle is held over whole blocks.
  - Stall still there on At: the model or the rollout, not the prior and not observation.
  - Stall gone on At but not on Cb: a remaining difference in actuator shape, not an observation gap. At's delay is one candidate cause (above).
  - Stall gone on At and on Cb, but steady-block yaw still unlearned (with at least 30 steady blocks): **open**. M10 did not change the velocity-channel prior, so this outcome does not separate two causes: (a) the target's sway/yaw dynamics lie outside the operator family, a prior gap that more or broader data can fix; (b) an input the model cannot see. Two tests come before any observation question goes to the user:
    1. **Oracle data** (meta_step3 cbhold + train_oracle). Fine-tune model3 on the target world itself (Cbh, every 4th episode held out). Score steady-block yaw on the held-out episodes (diag_yaw_m8_blocks --split Cbh --holdout --model model3_oracle.pt). If the oracle learns it, the information is in the inputs the model already sees. The gap is then in the prior: broaden the operator family with generic sway-yaw terms that depend on state and command, and name no target mechanism. If even the oracle cannot learn it, the inputs lack the information, and only then is it an observation question.
    2. **Oracle input** (only in the second case). Give model3 each candidate signal as a perfect measurement in both worlds, as info_waves_m11 did for waves: roll rate, lateral acceleration, waves. A candidate counts only if it closes most of the gap. Roll rate first has to exist in the source world.
  - Reference, not a decider: diag_yaw_m8_blocks --split A / At gives the in-family steady-block yaw level caused by the source world's own unseen inputs (waves are off in the model's input; the operator noise is hidden). "Cb much worse than A/At" alone does not show a prior gap, because the target's unseen share (high-fidelity sea, roll) can be larger.

**D8 — A rigid-body-mechanics form of the error prior (2026-09-30).** Written in Chinese at the end of this file (section "D8"). D8.12 (after the critique) overrides D8.3–D8.11 where they conflict; its staged first version is implemented in learn/meta/operators_rb.py (standalone, not yet in data generation).

**D9 — 误差先验改为“受任意力的刚体”（2026-09-30，用户的方向）。** 中文写在本文件末尾（“D9”一节）：刚体力学只规定力怎样变成运动（误差只进加速度、未知正定惯性、点力的力臂、刚体科氏项），力本身是一切已经经历过的东西的任意因果函数（随机滤波器组 + 随机小网络读出 + 看不见的随机过程），只保留大小上限、截断和拒绝发散；D8 族降为对照组。另定义约束任务要记录的每步重心竖向加速度峰值和最小船头余高。只是推导，未实现。

## 0. Setting, and what each number rests on

Control step Δ = 0.24 s, so the Nyquist frequency is ω_N = π/Δ = 13.1 rad/s. Measured each step: s_k = (reduced state x_k, commands u_k, wave elevations η_k at 5 stations × 3 points). Observed: the one-step error e_k ∈ R⁵. Information available before e_k: I_k = (s_≤k, e_<k).

**Computed.**
- **Encounter frequency.** ω_e = |ω + ω²U cos μ/g|, with μ = 0 for head seas and 180° for following seas.
  - Target (Tp 5 s, 22 kn, head): ω_e = 3.08 rad/s, T_e = 2.04 s.
  - Over the training ranges (Tp 4–7 s, 16–30 kn), the encounter peak lies at:
    - 1.6–5.5 rad/s in head seas;
    - 0.9–1.6 rad/s in beam seas;
    - 0–2.3 rad/s in following seas. Here the map folds at g/4U = 0.16–0.30 rad/s, which puts an integrable singularity in the encounter spectrum.
- **Aliasing.** Head-sea components above ω = 2.96 rad/s (22 kn) or 2.58 rad/s (30 kn) are met faster than ω_N. That is 2.6% (Tp 5 s, 22 kn) to 10% (Tp 4 s, 30 kn) of JONSWAP energy.
- **Wavelength.** λ_p = 39 m, far longer than the 5.4 m hull. Station spacing is 1.35 m.
- **Hull-frame bandwidth at Tp 5 s.** Statistical bandwidth (∫S)²/∫S²: 0.83 rad/s in beam seas, 3.3 rad/s in head seas at 22 kn. Half-power widths: 0.24 and 0.93 rad/s.

**Read from the code.**
- **Sea model.** step0_preview_spec.WaveField, used by both fidelities in learn/repro with n_freq = 24 and n_dir = 5. Frequencies sit on a fixed uniform grid over [0.5, 3.0] x the peak frequency (0.63-3.77 rad/s at Tp 5 s, spacing 0.137 rad/s), in 5 fixed directions: 120 sinusoids in total. [Corrected by the main session: the band is relative to the peak frequency.] Consequences:
  - (a) At a fixed point the field repeats every 46 s (Tp 5 s). In the hull frame the encounter frequencies are not evenly spaced, so the signal the boat sees does not repeat exactly, but it is still a sum of 120 fixed sinusoids.
  - (b) It is a deterministic line-spectrum process, whose prediction error from the infinite past is zero.
  - (c) With ≤ 5 directions per frequency and 15 sensors, the elevation anywhere else is in principle a linear function of the measured ones.
  - So wave predictability learned in this sea may not exist at real sea. There is also no content above 3.77 rad/s (Tp 5 s).
- **Encoder and recorder.** The encoder tokens (data.py, O, 19 numbers) contain no η. The recorder stores no η, hull position or heading.
- **Existing targets.** z is 5 × 16. Flow targets are ridge fits on 128 probe states. noisy_history perturbs the history only.
- **MPC.** t_preview = 0 means the MPC gets the true η at h = 0 and its expectation (zero) beyond. The missions run with use_rudder = False, so the nozzle comes from the autopilot outside the rollout. The review reports a rollout step of 0.25 s against a 0.24 s control period.

**Measured.**
- **The '30% unpredictable' anchor is biased.** It comes from evaluate_traj's 'window' fit, an in-sample ridge with K = 16 per channel on 40 steps (λ = 0.1).
  - White noise alone scores R² ≈ 16/40 = 0.4 on that fit.
  - Adjusted R² = 1 − 0.3·39/23 = 0.49.
  - So the unpredictable share may be about 0.5 rather than 0.3. It must be re-measured on held-out data (gate P0, §7).
- **Step-1 flow results.** Without history noise the flow was worse than no correction. With AR(1) history noise it was neutral to slightly better.
- **Costs (review timings).**
  - Low-fidelity plant: 4.9 ms per step. Twin: 5.3 ms per step.
  - MPPI with 128 plans × 24 steps: 28 ms per call.

**Assumed.** Every hyper-prior range in §4. Only the checks can confirm them.

## 1. What the error is, with no mechanism named

Write the true plant and the reduced model as:
- True plant: x_{k+1} = F(x_k, χ_k, u_k, W_(t_k, t_k+1]) and χ_{k+1} = G(·).
- Reduced model: x̂_{k+1} = f(x_k, u_k, η_k).

Here χ is whatever state the reduced model lacks. W is the whole outside field over the step: waves everywhere, wind, current, wakes. Only the point sample η_k is measured. W^⊥ is the rest, including the wave over the remainder of the step.

(A0) The hidden subsystem is incrementally exponentially stable, uniformly for bounded inputs, apart from finitely many discrete modes σ (for example a switch that is on or off). Then χ_k is a fading-memory causal function of the past of (s, W^⊥), and

**(1) e_k = Γ[s_≤k, W^⊥ up to t_{k+1}, σ_k], with Γ causal and of fading memory.**

W^⊥ reaches half a step past s_k. e_k is an average over (t_k, t_{k+1}]:
- its frequency response is |sin(ωΔ/2)/(ωΔ/2)|, which is 0.64 at ω_N;
- it is centred at t_k + Δ/2, so it leads the point sample η_k in phase by ω_eΔ/2 (21° at 3.08 rad/s).

The training distribution must therefore be a prior over causal operators Γ. Step 1 sampled Γ = g(x_k, u_k): no memory, no η, no W^⊥. Its encoder never saw η either.

## 2. Representation

**Assumptions.**
- (A1) Bounded inputs.
- (A2) Causal and time-invariant. Slow variation is added separately.
- (A3) Fading memory.
- (A4) The part driven by unmeasured signals has finite variance and is locally stationary (Priestley 1965; Dahlhaus 1997).

**(T1) Filter bank plus static map.**
- Statement: under A1–A3, for every ε there exist a finite stable linear system g_{k+1} = Λg_k + Bs_k with ρ(Λ) < 1, and a continuous N, such that sup |Γ[s]_k − N(g_k, s_k)| < ε.
- Sources: Boyd & Chua 1985 (continuous time); Grigoryeva & Ortega 2018 (discrete time). N can be an MLP (Leshno et al. 1993).
- Caveat: the theorem only says a finite bank exists for each ε. Its size grows as ε → 0 and depends on Γ, so a fixed bank is an approximation that must be tested (capacity check).
- Takenaka–Malmquist bases are complete only with infinitely many poles: Σ(1 − |ξ_i|) = ∞ (Ninness & Gustafsson 1997).
- A cascade of order n at a pole ξ approximates a true pole p with error falling like |(p − ξ)/(1 − ξ̄p)|ⁿ. Example: a true resonance at ω_n = 4 rad/s with ζ = 0.03 has discrete pole radius 0.972.
  - A bank pole at ζ = 0.15 (radius 0.866) gives a factor of 0.67 per section.
  - A bank pole at ζ = 0.07 (radius 0.935) gives 0.40.
  - If the true frequency lies 15% off the bank frequency, the factor rises to about 0.9.
- The static case is B = 0.

**(T2) Unmeasured part: filtered white noise.**
- Wold decomposition: a stationary process with no deterministic component is n_k = Σψ_j ε_{k−j}, where ε is white (uncorrelated, not necessarily independent).
- Rational spectra are the same thing as finite state-space models. Continuous spectra are approximated uniformly by AR spectra (Brockwell & Davis §4.4).
- Kolmogorov–Szegő: the minimum mean-square error of the best *linear* one-step predictor from the infinite own past is exp{(1/2π)∫log Φ}.
- So the spectral flatness SF = exp(mean log Φ)/mean Φ is the linearly unpredictable fraction.
- It is only a *lower bound* on predictability: nonlinear predictors and exogenous inputs can do better. A line spectrum, like the current sim sea, has SF = 0.

**(T3) Memory that does not fade: relays.**
- Mayergoyz 1986: a rate-independent hysteresis with the wiping-out and congruent-minor-loop properties is a Preisach superposition of two-state relays. Finite sums of relays approximate it.
- A minimum dwell time makes a relay rate-dependent. That is a modelling extension outside the theorem.

**(T4) The split.** Doob: e_k = E[e_k | I_k] + ν_k with E[ν_k | I_k] = 0. This needs only finite variance and holds in closed loop. Var(ν_k | I_k) is unrestricted and may depend on the state.

**Combined representation (R).**
- e_k = a_k ⊙ N(g_k, s_k, h_k) + n_k + b_k
- g_{k+1} = Λ(s̄_k) g_k + B[φ(s_k); o_k]
- h_k = relay(h_{k−1}, Cg_k)
- n_k = σ_k (G(q)ε)_k

The symbols:
- o: unmeasured signals.
- φ: an elementwise static map before the filter. This makes the form LNL (nonlinear–linear–nonlinear), which contains the Wiener and Hammerstein structures.
- s̄: slow measured signals.
- a, b: slow gain and offset.
- G: a random stable filter.
- σ_k: a state-dependent scale.

The whole class is one stable linear bank with static maps before and after it, a few switches, and filtered heteroscedastic noise. No mechanism is named.

**(i) Phase and lag.**
- A static map of the 15 spread elevations can produce *any* phase at one encounter frequency and heading. Σ w_m η(r_m) has phase arg Σ w_m e^{ik r_m·e_β} relative to the centre. The bow-minus-stern difference is about 90° off the mean, and the along-hull phase spread at the target peak is k·5.4 m ≈ 50°.
- What a static map cannot produce is a phase law other than the geometric one. Its phase changes with ω_e only through k(ω):
  - dk/dω_e = (2ω/g)/(1 + 2ωU/g) = 0.066 s/m at the target;
  - so a 2.7 m half-length amounts to about 0.18 s of delay (about 0.26 s for ±1 m laterally in beam seas).
- A true lag τ has phase −ω_eτ.
- Hence filters are needed for three things: lags much longer than about 0.2 s, ringing, and a phase that must stay right across turns and speed changes. Shorter lags are just spatial shifts.
- Step 1 failed more simply: it had no η at all.
- A filter acts at ω_e only. The dependence of the true transfer on wavenumber and heading, X(ω, β, U), must come from the spatial array and from scheduling by s̄ (speed, heading).
- Squaring a filtered narrow-band wave A cos(ω_e t) gives a slow term A²/2 that follows the wave-group envelope, plus a 2ω_e term. Across K_f filters this is a second-order Volterra kernel of rank at most K_f(K_f + 1)/2, not a general one.

**(ii) The unmeasured wave-driven part.**
- The best *two-sided* linear estimate of a linear wave force from the 15 elevations leaves S_FF(1 − γ²), where γ² is the multiple coherence.
- Restricting to causal filters (no future waves) adds a separate causal gap, which comes from spectral factorisation of S_ηη. So coherence overstates what can be predicted causally.
- In long-crested seas where ω ↦ ω_e is one-to-one, the elevation at any offset is an all-pass filter of the measured elevation, with response e^{ik(ω)e·dr}:
  - down-wave points are delayed copies, fully predictable;
  - up-wave points are advanced copies, unpredictable only because of causality;
  - points along the crest are identical.
- Real coherence loss needs directional spreading, folding (up to three wave frequencies per ω_e in following seas), or independent content. The current 5-direction, 24-frequency sea gives little of it (point (c) in §0).
- So G2 builds partial coherence directly, by mixing an offset field with an independent sea.

**(iii) Aliasing.**
- Content above ω_N folds to 2π/Δ − ω and keeps its shape; for example, 15 rad/s lands at 11.2 rad/s. It looks like a narrow band near Nyquist, not like white noise.
- The noise filters are therefore specified directly in discrete time, so such bands can be placed.
- Injecting the push with a zero-order hold (ZOH, held constant over the step) cannot reproduce e_k's half-step dependence on future waves. Random numerators supply the phase at ω_e, and up-wave hidden offsets supply the unpredictable part within the step.

**Not covered by (R):**
- operators that vary strongly in time beyond slow a, b and occasional regime changes;
- hysteresis without wiping-out;
- integrations longer than the history (they look like b).

## 3. What a history can identify (this sizes the prior)

History length T_h ≤ 60 s (250 steps).

- **Time constants.** τ > T_h/3 = 20 s blurs into drift. τ < Δ/2 acts as a static gain. So the prior is τ ∈ [0.1, 30] s, with the top of the range deliberately merging into b.
- **Resonances.** Frequency resolution is 2π/T_h = 0.10 rad/s. A decay time longer than T_h/2 looks undamped, so ζ ≥ 2/(ω_n T_h).
- **Wave-driven degrees of freedom.**
  - N_eff ≈ B_eq T_h/π, where B_eq is the statistical bandwidth of the regressor spectrum |H|²S_e. This is the measure behind the Slepian count and the Bendat–Piersol variance formulas.
  - At Tp 5 s: N_eff ≈ 16 (beam) to 63 (head, 22 kn) in 60 s. The half-power definition gives only 4.6–18.
  - With p Gaussian regressors, the out-of-sample excess error is p/(N_eff − p − 1). Keeping it ≤ 25% needs p ≤ (N_eff − 1)/5, i.e. 3–12 parameters per channel (1–3 by the half-power definition).
  - Treat this as uncertain by a factor of 3. The prior puts most mass on 1–3 filters (E[K_f] ≈ 2.1).
  - Phase standard deviation ≈ 1/√(ρN_eff): 7–15° at a coherent-to-noise ratio ρ = 1.
- **What one condition cannot tell.** H(ω) away from the current ω_e. Only the prior (low-order filters are smooth in ω) or manoeuvres can supply it, and manoeuvres are RL's job.
- **Occam.**
  - A network trained on prior draws returns the posterior predictive under that prior (Müller et al. 2022).
  - Its excess loss on a real task behaves like nΔ(θ̃) + (d_eff/2) log n − log π(θ̃) (Barron & Cover 1991; Clarke & Barron 1990), where Δ(θ̃) is the per-step misfit of the best member.
  - Step 1: no member had a small Δ, hence the harm. With AR(1) history noise, the 'all noise' member had a moderate Δ, hence no harm and no gain.
  - So the prior is hierarchical: most mass on few filters, and rare structure pays only −log P(level).
- **Closed loop.** The direct prediction-error method is consistent in closed loop only when both hold (Ljung 1999):
  - the data are informative, through external excitation or a sufficiently complex or switching regulator;
  - the noise model is flexible enough.

  Dither is the practical guarantee for the first. The flexible noise head (§5) serves the second.
- **Non-uniqueness.** Several things cannot be told apart: the split of gain between filter and readout, the choice of state coordinates, and a filtered hidden input versus coloured noise. So the latent must be a predictive quantity on a fixed basis, never the generator's parameters.

## 4. The generative process

The hyper-priors are in generative_spec. It is built in two stages, so nothing is built before there is evidence for it.

**v0:**
- G0 with the sea fix, a random wave direction and nozzle dither;
- G1;
- G3–G5: filters on measured inputs, plus the readout;
- G8: coloured and white noise;
- the constant offset of G10;
- G11 with π_h = 0;
- G12 and G13.

**v1.** Each component is added only when a check flags its statistic:
- G2 + G7 (hidden inputs): T4 coherence too high in the v0 draws, or the in-band spectral shape wrong;
- G6 (relays): memory or regime statistics;
- G9 envelope: T8. G9 bursts: T7;
- G10 drift and regime changes: the T8 mean shift.

**Special cases.**
- K_f = 0 with π_h = π_n = 0 and no wave inputs is the step-1 family.
- A = 0 is the null operator.

## 5. The predictable part and the unpredictable part

By T4, e_k = E[e_k | I_k] + ν_k.

**Linear, stationary case.** This is the Box–Jenkins form e = H(q)s + G(q)ν, whose one-step predictor is ê_k = G⁻¹H s_k + (1 − G⁻¹)e_k. It has two filters:
- one on the measured inputs, which carries lags and memory;
- one on past errors, which carries the colour of the unmeasured drivers.

Step 1 was the corner case 'H static, G = 1'.

**Used here.** A nonlinear exogenous part, a linear AR noise model and a state-dependent scale. This is *not* the full nonlinear conditional mean.
- **(S1)** ê_{k+j|k} = Φ(g^c_{k+j}, s^c_{k+j}, j)·z + r̂_{k+j|k}, where r = e − Φz and r̂ is the AR(8) continuation.
- **(S2)** ν_k ~ t_ν(0, D_k C D_k), with D_k = diag(σ_k) and log σ_{c,k} = log σ_c + ψ_c(g_k, s_k)ᵀ z_σ.

The superscript c means 'computed under the MPC's convention': the true past, then the planned commands, the predicted states and η̂ = 0 beyond the current step. j counts steps since the last measured wave; it is 0 inside the history.

**Horizon.**
- For y = h * q, a j-step forecast under a convention q̂ errs by Σ_{i<j} h_i (q_{k+j−i} − q̂_{k+j−i|k}).
- Define share_j = 1 − E[(y_{k+j} − ŷ^c_{k+j|k})²]/var(y).
- With η̂ = 0, the bank's free response is itself the wave forecaster: a resonant state near ω_e keeps ringing after the input stops. Fitting z under the convention (Stage B) makes Φz the best forecast within the basis.
- Resulting predictability:
  - a static wave response stays predictable for about the hull-frame wave correlation time π/B_eq, which is about 1 s in head seas at 22 kn and about 3.8 s in beam seas (Tp 5 s);
  - a ringing response stays predictable for about max(π/B_eq, 1/(ζω_n));
  - command- and state-driven parts are predictable along the plan.
- A 'hold η_k' convention is rejected: holding for only 2 steps at 3 rad/s already gives a 1.4 rad phase error.

**Where the generator's sources land.** Approximate, because the split depends on the history.
- y^m (measured-driven): mostly predictable.
- y^h (hidden-driven): predictable through its coherence with the measured field and through its own colour; the rest is innovation.
- n^c (coloured noise): its linearly predictable fraction is 1 − SF.
- White noise, the timing of bursts, and waves not yet met: innovation.

SF, the T6 ladder and the oracle rung are all *lower bounds* on predictability, limited by their model classes and by finite samples. They size the achievable gain but do not cap it.

## 6. What this implies for the model

Details are in the architecture field.
1. Add η to the tokens and to the bank input.
2. **Containment requirement.** The model must contain the generator class up to bank truncation, tested component by component. That means:
   - a learned elementwise nonlinearity before the bank (LNL);
   - bank poles scheduled by slow signals;
   - a readout that sees s̄;
   - j as an input to the basis.
3. A fixed bank is not universal. A capacity experiment fixes K (16, 32 or 64), or switches to a latent-code decoder, before any data are generated.
4. Noise head: heteroscedastic, AR(8), Student-t, trained by likelihood rather than on Burg labels. The flow samples z only.
5. One convention everywhere: η̂ = 0 beyond the current step in the Stage-B targets, the calibration and the MPC. The rollout step is 0.24 s, and the nozzle is computed by the autopilot law inside the rollout.

## 7. Staged plan with gates

**P0: before any generator code.**
- Fix the sea in both fidelities: jittered frequencies per episode, 64 × 12 components, a long-crested option and a random wave direction.
- Extend the recorder. Per control step: the 15 elevations, the 14-number plant state, x, y, ψ, the sea seed, the dither and per-component seeds.
- Record 1–2 h of high fidelity over 3 speeds × 3 headings, with dither logged.
- Compute T3 and the held-out T6 ladder with blocked cross-validation. Also train one small nonlinear sequence model on target blocks, purely as a measuring instrument.
- **Gate:** build the generator only if some held-out reference beats rung (a) by at least 0.1 R² in at least 2 channels.
  - A fail is not proof that nothing is predictable, because these are lower bounds. It redirects effort to what is measured (roll, a wave sensor) instead of the prior.
- Re-anchor prior check (iii) on the held-out unpredictable share.

**P1: capacity experiment.** About 30 min on the GPU, on stored carrier inputs. It chooses K, or route 2.

**v0.**
- Data: about 1.4 h of simulation.
- Prior-only checks.
- P1-mode prior-predictive checks and negative controls.
- Training, then calibration.

**v1.** Add components only for statistics that are flagged. Run P2 and the classifier test after that.

## 8. Why it should carry over to errors never listed, and where it stops

**What (R) covers.**
- Anything causal, bounded, slowly varying and of fading memory in measured and unmeasured signals lies in its closure (T1).
- Stationary unmeasured disturbances with continuous spectra are approximated (T2).
- Preisach-type hysteresis is covered by relays (T3).

**Examples that were never enumerated.**
- Wind: a slow hidden process times a readout of heading and speed, plus gusts through a real pole.
- Current: static heading terms plus b.
- Other hulls' wakes: hidden narrow-band packets, and bursts through resonances.

**The limits.**
- Being in the closure is not the same as having prior mass.
- A fixed network bank is not the closure.
- Both are empirical questions. They are checked with statistics that the negative controls show can reject simpler families.
- For another hull (for example VM 18 at 50+ kn), rescale every time range from that hull's own time scales.

## ARCH
The model is a nonlinear exogenous predictor on a learned LNL filter bank, plus a heteroscedastic AR noise head. It stays linear in z (route 1) if the capacity experiment allows. Otherwise it becomes a latent-code decoder (route 2). The model must contain the generator class up to bank truncation, and this is tested component by component.

**(0) Inputs.**
- s_k: 26 numbers (9 current basis inputs, heave and pitch displacement, 15 wave elevations), standardised by global pilot-run statistics.
- Tokens: [s_k, e_k normalised per channel, the per-channel log scale of e, a valid mask].
- The current tokens contain no η.

**(1) LNL filter bank.**
- Input: v_k = h_φ(P s_k). P is learned (26 → 8). h_φ is a per-channel 1-16-1 MLP initialised near the identity; it mirrors G3's nonlinearity before the filter.
- Per channel:
  - 7 real poles, τ ∈ {0.12, 0.25, 0.5, 1, 2, 4, 8} s, each an order-2 Laguerre cascade (14 states);
  - 10 resonant sections, ω_n = 0.63·1.344^i for i = 0..9 (0.63–9.0 rad/s), initial ζ = 0.1, each an order-2 Kautz cascade (40 states);
  - a 3-step delay line.

  That is 57 states per channel, 456 in total.
- Rotation-scaling form A = rR(θ): A is normal with ‖A‖ = r < 1, so any variation stays stable.
- Poles are learned inside the prior ranges: r through a sigmoid, ζ ≥ 0.05, ω_n ≤ 9 rad/s.
- Scheduling: log r and θ are offset by κᵀs̄_k, where s̄ is speed and cos/sin heading low-passed at 5 s.
- Poles slower than 8 s are dropped, because such slow effects blur into the bias.
- The bank is pre-rolled on the time-reflected first 30 s of the history, so early checkpoints see warm states.
- The bank is an approximation (§2 T1). The capacity experiment decides whether it is dense enough.

**(2) Basis.**
- Φ = MLP([g̃_k, s_k, s̄_k, emb(j)]) → R^{5×K}. j counts steps since the last measured wave: 0 in the history, 1..20 over the horizon.
- The exogenous prediction is ê^ex_{c,k} = Σ_K Φ_{c,K} z_{c,K}.
- K is 16, 32 or 64, chosen by the capacity experiment.
- **Route 2** (if K = 64 is not enough):
  - a code c ∈ R^{16–32} and a decoder D(g, s, j; c) with FiLM conditioning;
  - trained as a conditional latent-variable model on the Stage-B continuations, by multi-step likelihood;
  - the flow then models p(c | history).

**(3) Noise head.**
- Residual r = e − Φz.
- Per channel, AR(8) with coefficients from partial autocorrelations κ ∈ (−1,1)⁸, which is stable by construction (Barndorff-Nielsen & Schou 1973).
- Scale: log σ_{c,k} = log σ_c + ψ_c(g_k, s_k)ᵀz_σ, with ψ giving 4 learned features per channel, so z_σ ∈ R^20.
- Student-t innovations with a per-channel ν_c.
- Cross-channel correlation from 10 canonical partial correlations through tanh, which always gives a unit-diagonal positive-definite matrix.
- The encoder outputs these numbers (8·5 + 5 + 20 + 10 + 5 ν) as amortised point estimates.
- They are trained by the Student-t multi-step likelihood of the held-out future residual e − Φz*. There are no Burg labels and no classical estimator in the label path.

**(4) Latent and encoder.**
- The flow samples only the whitened z: 80 numbers at K = 16. At larger K it samples the top PCA directions of z* that hold 99% of the prior variance.
- z says how strongly each learned dynamic feature drives each channel's error. A feature is a filtered, nonlinearly mixed history of the measured signals. Lags, memory and wave-locked phase live here.
- z is never the generator's parameters.
- Encoder:
  - a transformer over a 2–60 s history (≤ 250 tokens; d = 128, 4 layers);
  - learned relative positions and channel embeddings;
  - port/starboard mirror augmentation (p = 0.5);
  - the bank states from the whole past as extra context.
- Short histories must give broad posteriors.

**(5) Training.**
- **A. Basis.** Train P, h_φ, the poles, κ and Φ on prior draws applied to stored carrier inputs (relabelled, on the GPU). Ridge-fit z on half of each checkpoint's continuations and score on the rest.
- **B. Targets.**
  - Checkpoints fall every 60 steps after the warm-up, with late ones oversampled.
  - At each checkpoint, copy the operator's internal state and run M = 64 continuations of 20 steps:
    - 32 use the episode's own inputs with extra OU command dither;
    - 32 are donor segments from other episodes (other speed, heading and sea). For these, only the parts that do not fade are copied (relays, slow gain, bias, regime). The fading states of both the operator and the model bank are re-warmed on the donor's preceding 30 s, so there are no step transients.
  - Hidden inputs stay on, consistent within each segment. Independent noise, bursts and white terms are switched off.
  - Responses are mapped to e-space exactly, by stepping the reduced model with and without the push. This matters because the within-step gains differ strongly by channel and speed.
  - Features are computed under the MPC convention: η = 0 beyond step 0 in both s and the bank input, the recorded commands, and the bank discretised at 0.24 s.
  - z* = ridge fit pooled over j = 1..20. This is a finite-sample, ridge-shrunk projection onto span(Φ) under the distribution of continuations, not an exact conditional mean.
  - Target noise, the spread of z* over bootstrap resamples of the continuation set, is recorded.
- **C. Flow.** Conditional flow matching of the whitened z* given the history prefix.
- **D. Noise head.** Trained by likelihood, as in (3).
- **E. Calibration (mandatory).** Multi-step calibration of e_{k+1..k+20} under the same convention, with a Student-t likelihood. An optional fine-tune follows.

**(6) How the MPC uses it.**
- Rollout step 0.24 s, equal to the control period.
- The nozzle is computed by the autopilot law inside the rollout (or planned by the MPC).
- η̂ = 0 beyond the current step, exactly as in training.
- Implemented in torch on the GPU, batched over plans × samples: 128 × 16 = 2048 rollouts × 24 steps, with the same random numbers for every plan. Each step:
  - g_{j+1} = Λg_j + B h_φ(P s_j);
  - e_j = Φ(g_j, s_j, j)z + r̂_j, optionally plus sampled innovations for risk terms;
  - x_{j+1} = f(x_j, u_j, η̂_j) + Δe_j.
- Expected cost: tens of ms, to be measured against the 0.24 s budget. The numpy MPPI already takes 28 ms for one sample.
- Fallback:
  - one posterior-mean rollout per plan;
  - a penalty trace(Φ Cov(z) Φᵀ) plus the AR innovation variance;
  - full samples only for the few best plans.
- One GPU inference process serves all workers, so there is only one CUDA context in host RAM.

**(7) RL probing.**
- Information state: the predictive spread of Φz along the plan.
- Its job, derived in §3: sweep ω_e with turns and speed changes to pin down H(ω) away from the current encounter frequency, and excite the commands.

**(8) Special cases and fallback.**
- B = 0 (or Φ ignoring g), with α = 0 and no wave inputs, gives the step-1 family: contained, but not identical.
- Relays are not in the bank. The encoder tracks the current mode through z and the residual state, which is adequate over 2–5 s horizons.
- If the capacity experiment fails on relay components, add optional play (backlash) units on the bank outputs. These are the building blocks of Prandtl–Ishlinskii hysteresis.

**(9) Compute.**
- Data: about 1.4 h of simulation (G13).
- Targets: generated on the GPU on demand.
- Training: 1–3 h on the 3080 Ti.
- At most 5 low-fidelity processes plus 1 trainer, with mem_watchdog running.

## SPEC
- **G0 Scenario, sea and excitation [v0]**: Piecewise-constant speed and heading targets, tracked by the existing PI speed loop and autopilot. Sea = JONSWAP(Hs, Tp) x cos^2s spreading, synthesised per episode with jittered components: one random frequency in each of n_freq equal bins over [0.4, w_max], directions jittered the same way. A random mean direction theta0 is passed through Mission(track=...) and RotPreview. Commands u = u_ctrl + OU dither, and the dither is logged. Episodes last 90 s (120 s w.p. 0.3); the first 10 s are dropped.
  - prior: Speed ~U[16,30] kn, held U[10,40] s. Heading ~U(-180,180] deg, held U[20,60] s. theta0 ~U[0,360) deg. Hs ~U[0.5,1.5] m. Tp ~U[4,7] s. w_max ~U[3,4] rad/s. n_freq = 64, n_dir = 12 (fall back to 48 x 8 if the step cost grows by more than 1.5x). Spreading exponent s ~U[2,10]. Long-crested (n_dir = 1) w.p. 0.25. Thrust dither and nozzle dither each on w.p. 0.7: OU with tau ~LogU[0.5,5] s and amplitude ~U[0,0.15] of range. 1/4 of episodes are steady, with no dither. The high-fidelity target uses the same sea class.
  - why: The current sea is 120 fixed sinusoids on a uniform grid. It is periodic at a fixed point (46 s at Tp 5 s) and deterministic, so wave predictability learned in it can be an artefact. Jittered, denser components (spacing about 0.05 rad/s, well below 2*pi/T_h = 0.10) make the sea look continuous within a 60-s history. Varying the sea and the relative heading sweeps the encounter peak over 0-5.5 rad/s (head 1.6-5.5, beam 0.9-1.6, following 0-2.3, with the fold at 0.16-0.30), so gain and phase become identifiable. Logged dither makes command filters identifiable in closed loop and gives the checks an exogenous signal. The target (Tp 5 s, 22 kn) lies inside these ranges.
- **G1 Measured inputs s_k (fixed, not random) [v0]**: s_k = [9 current basis inputs (speed, sway, yaw rate, cos/sin heading, thrust, nozzle, heave velocity, pitch rate), heave and pitch displacement, the 15 wave elevations], each divided by its global pilot-run std.
  - prior: None. s is not normalised per episode, so the operating point stays visible.
  - why: T1 is a statement about operators on what is measured. Raw wave points are used, not hand-made slopes. The random smooth weights in G3 can build any spatial projection, including one with any phase at a single encounter frequency (§2 i).
- **G2 Unmeasured outside inputs o_k (never shown to the network) [v1]**: o_{m,k} = sqrt(rho_m) eta(r_hull + dr_m, t_k) + sqrt(1 - rho_m) eta'(r_hull + dr_m, t_k), for m = 1..N_h, where eta' is an independent sea with the same spectrum. Add 0-2 OU processes. Everything can be recomputed from the recorded trajectory and seeds.
  - prior: N_h ~U{1..4}. |dr| ~LogU[0.5,20] m, direction ~U[0,2pi). rho_m ~U[0,1]. OU tau ~LogU[2,60] s. In long-crested seas, offsets with rho_m = 1 that lie down-wave or along a crest are counted as measured-driven: they are delayed or identical copies of the measured field.
  - why: This builds partial coherence, about rho x the geometric coherence, directly and in any sea. Geometry alone gives coherence 1 in long-crested seas and close to 1 in a sea with few directions. The mixed-in part is unpredictable error inside the encounter band that follows omega_e as speed and heading change. The OU inputs stand in for slow drivers unrelated to the waves (wind, current).
- **G3 Input maps, pre-filter nonlinearity and delay [v0]**: v^i_k = h_i(p_i^T [s~_k; o~_k])_{k - d_i}. Wave weights w = G^{1/2} xi, with xi ~ N(0, I) and G^{1/2} the Cholesky factor of a squared-exponential correlation along the 5 stations.
  - prior: p_i reads 1 input group (0.7) or 2-3 groups (0.3). Groups: measured waves 0.4, states 0.35, commands 0.25; hidden inputs are used only in G7. Station length scale ~U[0.2,2] x 5.4 m. The 3 lateral points are correlated w.p. 0.5. Random sign. h_i: identity 0.7, random 1-D tanh MLP 0.15, or ReLU(v - b) 0.15, with b at a quantile ~U[0.7,0.98] of the episode's own pre-run input (for closed loop, a stored carrier segment of the same scenario). d_i in {0,1,2,3} steps w.p. (0.4, 0.3, 0.2, 0.1).
  - why: Sparse, smooth maps keep the effective dimension small (Occam) and contain mean, slope and curvature patterns without naming them. A nonlinearity before a resonant filter gives ringing after events. Together with G5 this makes the structure LNL, which contains both the Wiener and the Hammerstein forms; the model must contain it too (architecture 1).
- **G4 Stable linear filters, specified in discrete time [v0]**: g^i_{k+1} = A_i(sbar_k) g^i_k + b v^i_k and y^i_k = c_i^T g^i_k + d_i v^i_k. Filter types:
- resonant: A = r R(theta), with r = exp(-zeta wn Delta) and theta = wn sqrt(1 - zeta^2) Delta (two real poles if zeta >= 1);
- real: a = exp(-Delta/tau);
- a pure delay.
(c, d) are random, then scaled so that max over w in [0, pi/Delta] of |H(e^{i w Delta})| = 1. Scheduling: log r and theta are multiplied by exp(kappa sbar_k).
  - prior: K_f = 0 w.p. 0.15; otherwise K_f = 1 + Poisson(1.5), truncated at 6 (E[K_f] about 2.1). Type: resonant 0.45, real 0.4, delay only 0.15. tau ~LogU[0.1,30] s. wn ~0.5 LogU[0.3,9] + 0.5 LogU[1,8] rad/s. zeta ~LogU[max(0.05, 2/(60 wn)), 1] (about 23% of mass below 0.1). c ~N(0, I); d ~N(0,1) w.p. 0.5, else 0. Scheduling w.p. 0.5: kappa ~U[-0.25,0.25] per pilot std, with sbar one of speed, cos/sin heading or another measured signal, low-passed at 5 s.
  - why: T1: stable filters with static maps before and after them are dense in the fading-memory operators; a finite bank exists for each accuracy. The ranges bracket yaw (0.4 s), surge (4 s), pitch (3.7 rad/s), heave (5.8 rad/s), the encounter band up to 9 rad/s, and the identifiability limits. Above 9 rad/s, ZOH phase distortion and aliasing dominate, so the noise terms take over there. Random (c, d) can put any phase at omega_e. Normalising the peak of the discrete response removes the gain ambiguity between filter and readout. The rotation-scaling form keeps time-varying sections exponentially stable: ||A_k|| = r_k < 1, so ||g||^2 is a common Lyapunov function.
- **G5 Static readout [v0]**: y^m_k = N_0(g~_k, s~dir_k) + sum_j h^j_k N_j(g~_k, s~dir_k). N is a random MLP with 16 hidden units and 5 outputs, plus an optional bilinear term (a^T g)(b^T s). Each output is centred on the stored histories; its mean goes into b.
  - prior: Weight scale alpha ~LogU[0.2,3], from near-linear to saturating. Activation per unit: tanh 0.5, ReLU 0.2, square 0.15, steep sigmoid 0.15 (slope ~LogU[3,30], bias at a quantile ~U[0.5,0.98]). Direct inputs s_dir: 0-3 current signals. Bilinear term w.p. 0.5. Each output channel on w.p. 0.8.
  - why: This is T1's universal static map. Square units give second-order terms: a slow drift that follows the wave-group envelope, and a 2 omega_e term (a Volterra kernel of rank at most K_f(K_f+1)/2). Steep sigmoids give static thresholds, and bilinear terms give gain scheduling. The step-1 family is contained.
- **G6 Relays (memory that does not fade) [v1]**: h_k = 1 if c^T g_k > beta_hi; 0 if c^T g_k < beta_lo; otherwise h_{k-1}. No switch within T_d of the previous one. h gates a readout N_j.
  - prior: Number of relays 0 / 1 / 2 w.p. 0.7 / 0.22 / 0.08. Input: a filter output (0.6) or a slow measured signal (0.4). beta_hi at a quantile ~U[0.6,0.95] of the episode's own open-loop pre-run of the relay input. Band beta_hi - beta_lo ~U[0.05,0.5] std. T_d ~LogU[0.1,3] s. If the episode has fewer than 2 switches, resample the thresholds; otherwise label the relay as unvisited.
  - why: T3: relays are the building blocks of Preisach hysteresis, which lies outside fading memory. The dwell is a rate-dependent extension outside the theorem. Taking thresholds from the episode's own input makes it likely that both states are actually visited.
- **G7 Hidden-driven operator y^h [v1]**: The same form as G3-G5, but the filters read only o_k. W.p. 0.3 the readout is multiplied by a slow measured signal.
  - prior: K_f = 1 + Poisson(1), truncated at 4. Pole, numerator and readout priors as in G4-G5.
  - why: Unmeasured drivers pass through the same kind of dynamics as measured ones. Their error then has the right band, changes with the sea state, and is partly coherent with the measured waves. That coherent part is predictable and is kept in the targets.
- **G8 Coloured and white noise [v0]**: n^c_k = sigma_k (G(q) eps)_k, where G is a product of 1-2 discrete-time sections and eps = sqrt(w_k) xi_k, with xi ~ N(0, R) and w an inverse-gamma(nu/2, nu/2) scale mixture: shared across channels (a multivariate t) or per channel (independent t). n^w_k is iid.
  - prior: Resonant section w.p. 0.6: wn ~LogU[0.3,12.5] rad/s, zeta ~LogU[0.05,0.7], random numerator. Real section w.p. 0.6: tau ~LogU[0.1,30] s. At least one section. R ~LKJ(2). nu ~LogU[3,50]. Shared mixture w.p. 0.5. Shares (u_c, u_w, u_b) ~Dir(1,1,1) over the components present (u_b = 0 in v0).
  - why: T2: stationary unmeasured disturbances are filtered white noise, and low-order random rational spectra approximate smooth spectra. A resonant colour gives SF < 1, which AR(1) could not. Sections at 9-12.5 rad/s represent folded narrow bands (§2 iii). The white share stands for what is truly unpredictable.
- **G9 Envelope and bursts [v1]**: sigma_k = exp(gamma^T sbar_k + o^sigma_k). The driver sbar is drawn with the G3-G5 machinery: a random smooth projection, then a square or ReLU, then a real pole. Bursts beta_k = sum_m J_m rho(k - k_m) at rate lambda_k = min(lambda0 exp(kappa_b z~_k - kappa_b^2/2), 1/(4 Delta)), where rho is the impulse response of a random resonator.
  - prior: Envelope w.p. 0.5: 1-2 drivers with pole tau ~LogU[1,10] s, gamma ~N(0, 0.3^2); o^sigma is OU with tau ~LogU[10,60] s and sd ~U[0,0.4]. Bursts w.p. 0.3: lambda0 ~LogU[0.02,0.5] per s; kappa_b ~U[0,2] w.p. 0.5, else 0; z~ is a random filtered measured signal, standardised; J ~Student-t(3); resonator wn ~LogU[3,12] rad/s, zeta ~LogU[0.1,0.5].
  - why: Locally stationary noise (Priestley, Dahlhaus) whose level depends on speed, heading and sea. A wave-energy envelope arises as a special case without being named. The mean correction keeps lambda0 the mean rate, and the cap keeps bursts sparse. This is a modelling choice, not a theorem.
- **G10 Slow variation [offset v0; the rest v1]**: a_k = exp(o^a_k), with o^a an OU process. b_k = constant offset + OU process + the centred parts' means. A regime change at t_c resamples 30% of the operator's parameters.
  - prior: Offset ~N(0, (0.3 A)^2) w.p. 0.5. Gain drift w.p. 0.3: tau ~LogU[20,200] s, sd ~U[0.1,0.5]. Bias OU w.p. 0.3: tau_b ~LogU[20,500] s, stationary sd ~U[0,0.5] x A. Regime change w.p. 0.15, t_c ~U[25,70] s.
  - why: Effects slower than the history look like drift (current, trim, load, fouling). Regime changes teach the posterior to discount old history.
- **G11 Mixing, normalisation and amplitude [v0]**: r_k = A (.) a_k (.) (sat(sqrt(pi_m) ybar^m_k + sqrt(pi_h) ybar^h_k) + sqrt(pi_n) nbar_k) + b_k, with sat(x) = 2.5 tanh(x/2.5). A bar means centred (mean moved into b) and scaled to unit RMS by running that part open-loop on 16 stored input histories.
  - prior: (pi_m, pi_h, pi_n) ~Dir(2,1,1.5); in v0, pi_h = 0 and (pi_m, pi_n) ~Dir(2,1.5). Per-channel jitter ~Dir(10 pi). A_c = A_REF_c x LogU[0.03,1], with A_REF = [4,4,2,12,3]. Null operator (A = 0) w.p. 0.05.
  - why: The pi are mixing weights, not variance shares: y^m and y^h are correlated, and readouts have means. The actual variance shares and the predictable fraction are outputs, checked by prior check (iii). Keeping the noise outside the saturation makes the noise-off Stage-B targets exact; inside it, the second-order bias 0.5 sat'' sigma_n^2 would shrink the targets by about 13% at unit levels. The saturation keeps the push within control authority.
- **G12 Injection and observation [v0]**: r_k is computed once per control step and held over the 6 substeps (ZOH), then injected as an acceleration by the Injector in closed loop. e_k is observed as now, so it includes the boat's fast response within the step. The history sees e_k + eps^obs_k.
  - prior: eps^obs is white, with sd ~LogU[1e-3,0.03] x A_REF.
  - why: The unpredictable part is generated inside the plant, so it moves the boat and shows up in later states. Observation noise is the only place where history-only noise remains valid. Holding r over the step is consistent with a bank discretised at 0.24 s and 6x cheaper than updating every substep.
- **G13 Dataset, relabelling and reproducibility [v0]**: - 5k operator episodes (plant plus twin).
- 2.5k carrier episodes, plant only: with no operator, the plant and the twin coincide, so the twin is skipped.
- At most 50% of training windows are relabelled: a fresh operator is run open-loop on a carrier's recorded inputs, and e is computed exactly by stepping the reduced model from the recorded states with and without the push (vectorised, as in _probe_truth).
- One batched numpy implementation of the operator serves both the workers and the relabelling.
- Operators are reproducible from per-component seeds, so the Stage-B targets are regenerated on demand.
  - prior: History length ~U[2,60] s. Checkpoints every 60 steps, late ones oversampled x2. Port/starboard mirror augmentation w.p. 0.5. Operator and model-bank states are pre-rolled so that early checkpoints are warm.
  - why: Unlimited operator draws at no simulator cost, and relabelling breaks the closed-loop correlation between commands and error. Cost at the reviewed timings: 5k x 375 steps x 10.2 ms / 5 processes = about 64 min, plus 2.5k x 375 x 4.9 ms / 5 = about 15 min, plus about 10% for the 120-s episodes: about 1.4 h. Re-time after the sea fix. Storage is under 1 GB of float32 memmaps; the targets need none.

## CHECKS
- P0 GATE (run first; no generator code before it): fix the sea in both fidelities; extend the recorder (15 elevations, 14-number plant state, x, y, psi, sea seed, logged dither, per-component seeds); record 1-2 h of high fidelity over 3 speeds (18, 22, 26 kn) x 3 relative headings (head, beam, following). Compute T3 and the T6 ladder using held-out scores only (blocked 5-fold, blocks of at least 30 s; also report adjusted R2). Also train one small nonlinear sequence model (a GRU on the same inputs) on target blocks, purely as a measuring instrument. Build the generator only if some held-out reference beats rung (a) by at least 0.1 R2 in at least 2 channels. These are lower bounds on predictability, so a fail redirects effort to sensing (roll, a wave sensor) rather than proving that nothing can be learned. Re-anchor prior check (iii) on the held-out unpredictable share; the old 0.3 is an in-sample figure and could really be about 0.5.
- P1 CAPACITY EXPERIMENT (before generating data; about 30 min on the GPU, no flow): apply 500 v0 operators (and later each v1 component) open-loop to stored carrier inputs, with targets in e-space. Compare held-out R2 (fit on half of the continuations, score on the rest) of (a) per-task ridge on the raw reference bank states plus s (about 250 linear features; no sharing across tasks, an upper reference) against (b) the stage-A basis Phi with K = 16, 32 and 64. Choose the smallest K whose median R2 is at least 0.9 x that of (a), with no component type below 0.8 x. Break the results down by component type, zeta < 0.1, wn between bank grid points, nonlinearity before the filter, and scheduled poles. If K = 64 fails, use route 2. Targets are regenerated from seeds, so storage does not depend on this choice.
- SEA CHECK (after the sea fix; no target needed): (a) the elevation at a fixed point has autocorrelation below 0.1 at all lags between 3 group times and 300 s, i.e. no repeat; (b) ridge from 60 s of the 15 measured elevations to the elevation at offsets of 5 m and 20 m gives held-out R2 below 1, falling with offset in short-crested seas; (c) re-time the low- and high-fidelity steps, and drop to 48 x 8 components if the cost grows by more than 1.5x.
- PRIOR-ONLY CHECKS (no target): (i) Phase and gain coverage: over 5000 draws, linearise the path from measured waves to e at w in {0.5, 1, 2, 3, 5, 9} rad/s. At each frequency the circular mean resultant length of arg H must be below 0.1, and the 5-95% range of log10|H| must span at least 2 decades. (ii) Realised variance shares and the held-out predictable fraction (the pi are mixing weights, so these are outputs to be checked, not inputs). (iii) The prior distributions of the held-out unpredictable index and of SF must contain P0's measured values in their bulk (between the 10th and 90th percentiles).
- Two prior-predictive modes. P1 (primary; about 2000 draws, open loop on the GPU on the target's recorded inputs): hidden inputs come from the target's wave field at the drawn offsets (for the real boat, random-phase surrogates of the measured spectrum), and the output passes through exact reduced-model stepping. Only statistics of e alone and cross-statistics with exogenous signals (waves, logged dither) are compared: in the target, e moved the boat and the controller reacted, so the recorded states and commands carry feedback that P1 draws lack. The hull-frame waves depend on the trajectory only weakly and are treated as exogenous. P2 (whole pipeline; closed loop in the low-fidelity world): 100 draws x 3 conditions (head, beam and following at 22 kn; at least 450 s each), about 20 min on 5 processes. P2 allows every statistic, including command cross-statistics and the full ladder. Target runs of one condition are concatenated to at least 450 s; drop the first 10 s of every run.
- T1 scale: per channel, log RMS(e_c) and log MAD(e_c).
- T2 spectral shape: Welch PSD, Hann windows of 128 steps (30.7 s), 50% overlap, detrended. omega_e_hat = the spectral centroid of the mean PSD of the 15 wave points (not the peak, which in following seas picks the fold singularity). Relative bands [0,0.3), [0.3,0.7), [0.7,1.4), [1.4,3) and [3, 13.1/omega_e_hat) x omega_e_hat. When omega_e_hat < 0.6 rad/s, use 256-step windows and absolute bands 0-0.4, 0.4-1, 1-2, 2-4, 4-8 and 8-13.1 rad/s. No sub-0.1 rad/s statistic, because the bins cannot resolve it. Read the top band as partly folded content (§2 iii).
- T3 linear own-past predictability: AR per channel (order up to 20, chosen by AIC). SF_c = innovation variance / var(e_c); this is a lower bound on predictability. Also report the 2-step and 8-step prediction-error ratios. Compute on raw e and on the residual after T6x(b).
- T4 wave coherence and lag: magnitude-squared coherence between e_c and wave KL modes 1-3 (a fixed reference from low-fidelity runs, the same on both sides), averaged over [0.7,1.4] x omega_e_hat. Phase at omega_e_hat from the joint multiple-input regression of e_c on modes 1 and 2, stored as (cos, sin); this avoids the 90-degree jump that occurs when the most coherent mode switches. Within each condition, the group delay -dphi/dw over [0.7,1.4] x omega_e_hat from that regression's cross-spectrum, compared with the spatial-aperture delay of about 0.2-0.3 s. Phase alone is not evidence of memory (§2 i).
- T5 excitation coherence: in P1, coherence of e_c with the logged thrust and nozzle dither over 0-1.5 rad/s, plus the phase at the band's power centroid (a signature of actuator lag). In P2 only, the same with the commands themselves.
- T6 R2 ladder (all scores held-out; ridge with lambda from inner cross-validation, used only as a measuring instrument). Reference bank, fixed and the same on both sides: real poles tau {0.25, 1, 4} s and pairs wn {1, 2.2, 4.5} rad/s with zeta 0.3. Exogenous ladder T6x (P1 and P2): (a_x) static linear in eta_k (15 points) and the dither; (b_x) plus the reference bank on 3 KL modes and the dither; (c_x) plus 100 fixed random tanh features; (e_x) plus future KL modes and dither over 8 steps, scored out of sample. Full ladder T6 (P2 and the P0 gate only): the same rungs with states and commands added, plus (d) AR(8) of past e. Indices: memory = b - a, per condition and pooled over conditions; nonlinearity = c - b; own-past = d - c; causal gap = e - c; unpredictable = 1 - held-out R2 of (e).
- T7 tails (robust): the share of sum(e^2) coming from the top 1% of |e|; the ratio q99/q90 of |e|; a Hill index on the top 2%; skewness. No kurtosis, because it does not exist for nu <= 4 (about 10% of draws).
- T8 non-stationarity: coefficient of variation of 10-s block variances; correlation of the log 2-s local variance with speed and with a local wave-energy proxy (a measuring instrument only); the difference of the mean between the two halves of a run, divided by the sd.
- T9 cross-channel structure: the top-2 eigenvalue fractions of the 5x5 correlation matrix; heave-pitch and sway-yaw coherence at omega_e_hat.
- T10 input check (P2 only): T1, T2 and T4 on the inputs themselves (heave and pitch against the wave modes). This catches differences between low- and high-fidelity inputs that the operator prior cannot fix.
- Transforms before any test: log for scales, logit for fractions, Fisher z (atanh of sqrt) for coherences, and (cos, sin) for angles. Robust standardisation by the draws' median and MAD.
- Decision rule. Marginal: the PIT rank of each target statistic among the draws must lie in [0.01, 0.99]; report all of them (expect about 2% false flags). Joint, on 25 statistics (per channel: log RMS, SF, in-band coherence, memory index T6x, unpredictable index T6x): the mean distance to the 10 nearest draws, ranked against the draws' leave-one-out distances, must give p >= 0.05, and a robust (MCD) Mahalanobis rank must also give p >= 0.05. Inspect the hyper-parameters of the 20 nearest draws to see where in the family the target lies.
- Classifier two-sample test (deferred until 1-2 h of target data exist): a logistic classifier on fixed per-window features of 30-s windows (band powers, coherences, AR(4) coefficients, tail statistics), with blocked train/test splits. Its AUC is compared with a null distribution: at least 100 held-out prior draws, each with the target's data volume, each playing the target against the pooled draws. Pass if the target's AUC rank gives p >= 0.05. A fixed threshold such as 'AUC <= 0.6' would reject correct priors, because one operator's windows always differ from the pooled prior predictive. Real data is used only for this diagnostic, never for the posterior.
- Negative controls (checks that the check has power). NC1, the step-1 family (no eta): must fail T4 coherence and the memory index. NC2, a static readout that includes eta (K_f = 0, wave inputs): must fail the memory index (T6x b - a, pooled over conditions). If NC2 passes, the statistics cannot detect lag, and the check is inconclusive on that point.
- Operator-level fit (model criticism, not training): fit the generator's own form (at most 4 filters, no hidden inputs, AR(8) heteroscedastic noise) to the first half of the target by gradient descent. The held-out R2 should approach T6(c). The fitted tau, wn, zeta, delays and noise fraction should lie inside the prior's support, not at its edges.
- Anti-leak rule: hyper-priors may be widened but not re-centred, using one condition set (head and bow seas at 22 kn). The check must then pass on held-out conditions (beam, following, 18 and 26 kn, another sea state), and finally on real-boat logs. Report every adjustment.
- What a failure points to: T4 or the memory index means missing lag or filter support; T3 means the wrong noise colour; T8 means the envelope; T7 means the bursts; the top band of T2 means folded content or the white share; T10 means the low-fidelity inputs, not the operator prior.
- Unit checks: rerunning the operator offline on recorded inputs reproduces the recorded r_k to within 1e-5; e from reduced-model stepping matches the closed-loop e on a few operator episodes; after pre-roll, the variance of the bank states at the first checkpoint is within 10% of that at late checkpoints.
- After training, in-family: simulation-based calibration (Talts et al. 2018) of each whitened direction of z on held-out draws, with the Stage-B target noise accounted for. PIT of the noise head on held-out residuals, binned by predicted sigma. Leave-one-component-out: train without hidden inputs, bursts, relays or operating-point dependence, and test on episodes that contain them. Pass if 50% and 90% interval coverage stays within 5 points at horizons of 1, 5 and 20 steps, and CRPS is never worse than no correction.
- After training, on high fidelity (evaluation only): interval coverage, CRPS and MSE against no correction and against Bayesian linear regression on the same basis, under the MPC's convention. Also the typicality of the high-fidelity encoder embeddings (distance to the 10 nearest training embeddings).

## RISKS
- Being in the closure is not the same as having prior mass, and a fixed bank is not the closure. A real operator that needs a rare combination (many filters, a sharp resonance between grid poles, a relay on a hidden input) gets little prior mass or little model capacity. Guards: the capacity experiment, the joint typicality test, the calibrated classifier test and inspection of the nearest draws.
- All predictability measures (SF, the T6 ladder, the oracle rung) are lower bounds. The P0 gate can stop a problem that a better model could learn, or pass on artefacts. It is a decision heuristic, not a proof.
- Sim sea artefact: the current sea is a periodic, deterministic line spectrum, and the high-fidelity target uses the same class. Unless both fidelities are fixed, part of any wave predictability found is an artefact that will not exist at sea. After the fix, the step-1 numbers are no longer comparable.
- A linear-in-z basis shared by all tasks may not have the capacity for the prior's operators, so z* would lose the wave-locked lag and step 1's failure would repeat. Route 2 fixes capacity but loses the Bayesian linear regression baseline and cheap linear sampling in the MPC.
- Convention: with eta_hat = 0 beyond the current step, the model's bank must forecast the waves through its free response, and a single z serves all horizon steps (j as a basis input mitigates this). The reduced model inside the MPC also runs with eta_hat = 0. Its own wave-forecast error is not part of e, which is defined with the true wave, so the correction does not cover it. An option is to redefine the targets as the reduced model's error under the convention.
- Breadth against sharpness: a broader prior gives broader posteriors, a more cautious MPC and slower training. Mitigations: hierarchical Occam weights; a curriculum on the sampler (static, then filters, then noise types, then v1 parts, over the first 40% of training, with 20% of batches always at the simplest level); and in-family calibration before any test on the real gap.
- Input-distribution mismatch: the low-fidelity world lacks roll, speed loss and similar effects, so its joint input distribution differs from high fidelity. Error driven by roll exists there only as hidden modes and may be attributed to noise. P2, T10 and the classifier test show this; the operator prior cannot fix it.
- Closed-loop confounding and shift: under feedback, terms driven by commands and terms driven by states are correlated, and once the MPC plans with posterior samples the input distribution changes again. Dither, relabelling and a later round of data generated with the adaptive MPC in the loop reduce the shift but do not remove it.
- Bank capacity: fixed or constrained poles miss sharp resonances between grid points (Kautz factor about 0.9 at 15% frequency offset). Order-2 cascades and learned poles help; the capacity experiment reports this case separately.
- Relabelled windows break the consistency between states and error (the error never moved the boat). Keep them to at most 50%, use carrier episodes only, and compute e exactly by reduced-model stepping.
- Non-identifiability (gain split between filter and readout; coloured noise versus a filtered hidden input) leaves near-flat directions in z, and Stage-B targets carry probe-set noise. Use whitened projection targets, 64 continuations and bootstrap target-noise estimates, and monitor calibration per direction.
- The noise head gives point estimates, so its parameter uncertainty is ignored. For short histories the predictive spread may be too narrow; check the PIT against history length.
- Aliasing and missing content: the sim sea has nothing above 3-4 rad/s, while the real sea puts 2.6-10% of its energy above the head-sea alias cutoff. Folded bands are represented only by noise sections, and ZOH injection cannot reproduce e_k's half-step dependence on future waves.
- Real-boat wave measurement: the 15-point elevation under the hull is not a sensor output. Without a wave sensor or a reconstruction, the method has no wave input, and T4, T6 and P1 cannot be computed on real logs. The design does not depend on which signals are measured, but it must be retrained with the real sensor set, and that set must be decided now.
- Outside the family: memory that neither fades nor is Preisach-type, integrators longer than the history, strongly time-varying events, and hidden modes driven by unmeasured inputs with long recovery. Only relays, the bias, the slow gain and regime changes approximate these.
- Portability: the pole and encounter ranges are tied to this vessel and to sea state 3. For VM 18 (5.8 m, planing at 50+ kn, jet), rescale them from its own time scales. A pass on high fidelity does not guarantee a pass on the real boat.
- Heavy tails from impacts can dominate a squared loss. Use Student-t likelihoods and robust scaling, and robust tail statistics in the checks.
- Compute and memory: the simulation cost (about 1.4 h) rests on the reviewed timings and may grow with the denser sea. The GPU port of the MPC rollout is unmeasured, and the MPC is not real-time without it. The 16 GB RAM limit allows at most 5 low-fidelity processes plus 1 trainer and one shared GPU inference process, with mem_watchdog running.
- Scope: the full G0-G13 design plus all checks is weeks of work. The staged plan (P0 gate, P1 capacity experiment, v0, then v1 only for flagged statistics) keeps each step short, but it only works if the gates are respected.

## D8 — 按刚体力学的一般形式构造误差先验（2026-09-30；推导；审查后的修订和第一版实现见 D8.12）

起因是 M15 的先验预测检验：现有的"随机投影"误差族在三处结构上放错了概率——像额外阻尼的横荡误差太少，随航速变化的纵摇误差太少，升沉和纵摇误差同步的太少。本节不再从"随机读输入"出发，而从一切刚性船体都满足的运动方程出发：真实船和简化模型的差别，就是这个方程里各项的随机改变。每个结论标明依据：【读代码】【计算】【假设】【测量】。本节只是推导，没有写代码。

**边界（用户的决定，本节遵守）**
- 低保真世界的物理不改。整个新族仍以"推力"（加在低保真船上的加速度修正）的形式出现，和现在的误差函数一样；低保真船的代码一行不动。
- 不规定哪个自由度读哪种浪量。浪只通过"作用在船体随机位置上的随机力"进入：力读什么浪、怎么读都随机；各通道之间的比例由作用点决定，不由浪量决定。
- 浪高加预见是正式的输入方向；误差读的是这一步期间船实际遇到的浪（M15）。
- 通用、不针对特定机制：只用一切刚体船都满足的性质（质量为正、阻尼只耗能、左右对称、与质量配套的耦合不做功、工作航速下稳定），不写任何具体机理（没有"喷泵吸气""砰击"这类项）。

### D8.1 出发点和一个恒等式

一般的船舶刚体方程（Cummins 形式，辐射的"记忆"写成卷积）：

    M ν̇ + C(ν) ν + D(ν, s̄) ν + μ + g(η) = τ_ctrl + τ_env,    μ(t) = ∫₀^∞ K(s) ν(t − s) ds

- ν = (u, v, r, w, q)：纵荡速度、横荡速度、艏摇角速度、升沉速度、纵摇角速度，正是 e0 前 5 个通道，顺序相同。η = 升沉、纵摇位移（相对航行姿态 z0、θ0）。s̄ = 慢变量（航速，必要时加姿态）。
- M：质量加附加质量（船推动周围的水所多出的惯性），正定。
- C(ν)ν：科氏项（在转动的船体坐标里出现的惯性力），由 M 决定，不做功：νᵀCν = 0。
- D：阻尼，只能带走能量：νᵀDν ≥ 0。
- K：辐射记忆核（船过去的运动造成的波还在推它），"正实"：整体上也只带走能量。
- g：回复力（浮力和滑行升力随姿态的变化）。

简化模型（低保真船）也可以写成 M₀ ν̇₀ = f₀，f₀ 包含它自己的阻尼、回复、推力和浪。真实船是 M_t = M₀ + ΔM、f_t = f₀ + Δf。于是在同一状态下，两者加速度之差是【计算，恒等式】：

    a = ν̇_t − ν̇₀ = M_t⁻¹ (Δf − ΔM ν̇₀),
    Δf = −ΔC ν − ΔD ν − Δμ − Δg + Δτ_ctrl + τ_env

（推导：ν̇_t = M_t⁻¹(M₀ν̇₀ + Δf)，而 M_t⁻¹M₀ − I = −M_t⁻¹ΔM。）

含义：只要知道低保真船此刻自己的加速度 ν̇₀，"真实船"的差别就能精确地写成一个加速度修正 a，加在低保真船上，低保真物理本身不变。质量变化的作用是"−ΔM × 低保真加速度"，再经 M_t⁻¹ 分到各通道：更重的船对一切（推力、浪、回复）都响应得更小，而质量矩阵的非对角元会把一个通道的加速度带到另一个通道。

### D8.2 坐标和符号（读代码确定）

- 船体坐标：x 向前、y 向左舷、z 向上，右手系；艏向 ψ 逆时针为正（control/reduced.py 的站点公式 Y = y + x sinψ + y_off cosψ）。纵摇 θ 为正时船首向下：船首高度 = z − x θ（sim/vessel.py 的 SIGN_PITCH = −1，"来自 Capytaine 的自由度定义"）。【读代码】
- 升沉、纵摇用"随船匀速前进的坐标"（耐波性的习惯）：升沉速度是竖直速度 ż，不是船体坐标里的 w（planing_vessel.py 中 ds[2] = w，reduced.py 中 z 直接跟随水面）。横荡、艏摇用船体坐标（reduced.py 里的 −m u r 就是船体坐标的科氏项）。【读代码】
- 作用在船体点 p = (x, y, z) 的力 F = (F_x, F_y, F_z)，换成五个自由度的广义力（刚体雅可比矩阵的转置，即 p × F）【计算】：

      Q_u = F_x,  Q_v = F_y,  Q_r = x F_y − y F_x,  Q_w = F_z,  Q_q = z F_x − x F_z

  即：竖直力在 x 处 → 升沉力 F_z 和纵摇力矩 −x F_z（船首的向上力让船首抬起，θ 减小）；侧向力在 x 处 → 横荡力 F_y 和艏摇力矩 x F_y；推力线在重心以下（z < 0）→ 抬首力矩。
- 标称惯性（只用到比值）：M₀ = m_v · diag(μ_u, 1, ρ_r², μ_w, μ_w ρ_q²)，m_v = 1610 kg（m_sway）。
  - μ_u = m_surge / m_sway = 1969 / 1610 = 1.22【读代码：studies/_cache/lofi_scarab195.json】；
  - ρ_r = 1.83 m（0.34 L）：简化模型里每牛顿侧力的艏摇角加速度是 k_nomoto_f / τ_r = −4.0e-4，喷口在重心后 2.18 m（hydro/hulls.py：lcg = 0.38 × 5.74 m），所以 I_r = 2.18 / 4.0e-4 ≈ 5.4e3 kg m²【计算】；
  - μ_w（升沉质量 / 横荡质量）和 ρ_q（纵摇回转半径）简化模型里没有（它的升沉、纵摇是"单位质量"的振子），每回合随机抽：μ_w ~ LogU[0.85, 1.6]，ρ_q ~ U[0.22, 0.32] L【假设：刚体 1400 kg、k_pitch = 0.25 L，加 0–90% 的附加质量】。
- 各项先在"质量归一化"的坐标 ν̃ = M₀^{1/2} ν 里抽取（这样一个数在各通道的大小可比，单位是 1/s 或 1/s²），再换回各通道的加速度。

### D8.3 各项：形式、落到哪些通道、怎样随机

光滑随机函数统一写成航速的二次以内多项式：ξ = (u − 23 kn) / (7 kn)（训练航速 16–30 kn 对应 [−1, 1]，超出范围按端点值保持，不外推），

    g(ξ) = c₀ + c₁ ξ + c₂ (3ξ² − 1)/2,   c_k ~ N(0, σ² ρ^{2k}),  ρ ~ U[0.3, 0.8]

w.p. 0.4 只有 c₀（与航速无关）。需要为正的量写成 exp(g)。下面 LogU 表示对数均匀，"w.p." 表示以该概率开启。

**T1 质量（含附加质量）变化**（w.p. 0.5）
- M̃_t(ξ) = L(ξ) L(ξ)ᵀ，L 为下三角：对角 exp(ℓ_ii(ξ))，ℓ_ii 取 g(ξ)，σ = 0.1（对角倍数 exp(2ℓ) 大约 0.8–1.25）；非对角 ℓ_ij 只在同一平面内——{纵荡、升沉、纵摇} 和 {横荡、艏摇}（左右对称的船不把两个平面耦合起来）——各 w.p. 0.6，σ = 0.1。这样对任何航速 M_t 都正定。
- a_T1 = −M_t⁻¹ ΔM ν̇₀。
- 效果：升沉–纵摇的非对角元让同一步的升沉误差和纵摇误差按固定比例一起出现（比例 ∝ 1/ρ_q，符号随机）。

**T2 与质量变化配套、不做功的耦合**（随 T1 自动开启，没有自由参数）
- 横向平面和纵荡（船体坐标）：用标准公式（Fossen）C(ν) = [[0, −S(ΔM₁₁ν₁ + ΔM₁₂ν₂)], [−S(ΔM₁₁ν₁ + ΔM₁₂ν₂), −S(ΔM₂₁ν₁ + ΔM₂₂ν₂)]]，S(a)b = a × b，ν₁ = (u, v, 0)，ν₂ = (0, 0, r)。不线性化，严格满足 νᵀΔCν = 0。主要的项：横荡 −Δm_u U r / m_v，艏摇 −(Δm_v − Δm_u) U v / I_r（Munk 力矩：细长体斜着前进时的转艏力矩），纵荡 +Δm_v v r / m_u。
- 竖直平面：同一公式换到"随船前进"的坐标（船体竖直速度 w_b = ż + U θ【计算】）后，得到升沉、纵摇之间一个反对称的速度耦合 (Δm_w − Δm_u) U [[0, 1], [−1, 0]]（作用在 (ż, θ̇) 上，自身不做功），以及纵摇的 Munk 刚度 (Δm_u − Δm_w) U² θ。形式与 Salvesen–Tuck–Faltinsen (1970) 的航速项（B35 − B53 ∝ U A33、U² 项）相同。符号由实现时的单元检验锁定：把船体坐标公式换系后逐项比较。
- 效果：纵摇误差 ∝ 航速 × 升沉速度、∝ 航速² × 纵摇角，所以纵摇误差随航速变。

**T3 耗散变化**（w.p. 0.8）
- 真实阻尼 D̃_t(ξ, ν) = S(ξ) D̃₀ S(ξ) + P(ξ) + diag(κ_c(ξ) |ν_c|) + G(ξ)：
  - D̃₀：简化模型自己的阻尼（线性化）【读代码】：横荡 (k_lin + k_quad |v|) / m_v ≈ 1.2 1/s，艏摇 1/τ_r = 2.4 1/s，升沉 2ζω = 4.3 1/s，纵摇 1.15 1/s；
  - S = diag(exp(s_c(ξ)))，s_c 的 c₀ ~ U[log 0.6, log 1.6]（阻尼倍数 S² 在 0.36–2.6），高阶项 σ = 0.2；
  - P = B Bᵀ（半正定，同平面耦合，B 有 1–2 列），w.p. 0.5，大小 LogU[0.05, 1] × 该方向的 D̃₀；
  - κ_c ≥ 0（二次阻尼），w.p. 0.3，LogU[0.05, 1] × D̃₀,c / 该通道速度的典型值；
  - G：反对称部分（升力类的项，∝ U，不做功），w.p. 0.3，每个平面一个，大小 LogU[0.05, 0.5] × sqrt(D̃₀,i D̃₀,j) × U / U_mid。
- 纵荡单独处理：阻力 R_t(u) = R₀(u) exp(g_R(ξ))，R₀ = k_drag u|u|，σ = 0.15（阻力曲线随航速的形状误差，比如滑行的阻力峰）。
- a_T3 = −M_t⁻¹ (D_t − D₀) ν（纵荡为 −ΔR / m_u）。
- **为什么偏向耗散、又保留两种符号**：约束是真实船的总阻尼 D₀ + ΔD ⪰ 0，而不是 ΔD ⪰ 0。所以阻尼最多减到零（ΔD ≥ −D₀），增加却不受限，再加上只能为正的 P，ΔD 的平均为正。这来自"真实船只能耗能"，不是按目标调出来的。

**T4 回复变化（升沉、纵摇）**（w.p. 0.7）
- g̃_t = K̃_t(ξ) (η̃ − η̃_eq(ξ))：
  - K̃_t = S_K K̃₀ S_K + [[0, k₃₅], [k₅₃, 0]]，K̃₀ = diag(ω_h², ω_p²) = diag(34, 13.8) 1/s²【读代码】。S_K 为对角 exp(g)，c₀ ~ U[log 0.8, log 1.25]（刚度倍数 0.64–1.56），w.p. 0.5；k₃₅、k₅₃ 各自独立（滑行升力的耦合不必对称），w.p. 0.5，N(0, (0.15 ω_h ω_p)²)；
  - 平衡位置随航速移动（w.p. 0.6）：Δθ_eq(ξ) 的 c₀ ~ U[−0.02, 0.02] rad、c₁ ~ N(0, 0.015²)；Δz_eq(ξ) 的 c₀ ~ U[−0.08, 0.08] m、c₁ ~ N(0, 0.05²)。依据：简化模型的 z0、θ0 只在设计航速 25 kn 标定过【读代码：identify 中 running_attitude(u_des)】，而训练航速是 16–30 kn；
  - w.p. 0.3 刚度还随相对姿态线性变化（浸深变了，回复力不再是线性的）。
- 作用在相对航行姿态的位移上；浪对回复力的那部分由 T7 的点力负责。
- a_T4 = −M₀^{-1/2} M̃_t⁻¹ [K̃_t (η̃ − η̃_eq) − K̃₀ η̃]。
- **稳定性（共享结构）**：在 ξ ∈ {−1, −0.5, 0, 0.5, 1} 上，真实船的升沉–纵摇线性系统（M_t、D_t + G + T2 的耦合、K_t + Munk 刚度）必须稳定且阻尼比 ≥ 0.05，否则重抽；横向平面（喷口固定）同样检查。"真实船在工作航速下不自激（不海豚跳）"被当作结构的一条。

**T5 操纵效能变化**（w.p. 0.7）
- 喷泵的力当作作用在随机点 p_j = (x_j, 0, z_j) 上的一个点力：

      F_x = η_T(ξ) T_app cos δ_app,   F_y = η_S(ξ) k_js T_app sin δ_app (1 + c₃ δ_app²)

  - η_T = exp(g)，c₀ ~ U[log 0.8, log 1.2]；η_S 同形，c₀ ~ U[log 0.7, log 1.3]（与 lofi.PRIOR 的 k_jet_side 范围一致）；c₃ w.p. 0.3，U[−1, 1] / δ_max²；
  - x_j = x_n0 + U[−0.1, 0.1] L，x_n0 = −2.18 m【读代码】；z_j ~ U[−1.5 T, 0]（推力线在重心以下某处，T = 吃水 0.38 m）；ρ_r 再乘 LogU[0.85, 1.15]。
- 标称（简化模型已有的）：纵荡 T / m_u，横荡 k_js T sin δ / m_v，艏摇 k_nomoto_f k_js T sin δ / τ_r，纵摇 0。
- a_T5 = M_t⁻¹ J(p_j)ᵀ F − 标称。
- 效果：喷口侧力的误差同时进横荡和艏摇，比例 x_j / ρ_r² 随机；推力的误差同时进纵荡和纵摇（z_j T）。D7 管"执行机构实际在哪个位置"，T5 管"在那个位置产生多大的力"，两者不重复。

**T6 辐射记忆变化**（w.p. 0.4，1–2 个）
- μ̃(t) = Σ_i c_i ∫ e^{−a_i s} cos(b_i s) v_i v_iᵀ ν̃(t − s) ds，c_i > 0，a_i > 0，v_i 为同一平面内的单位向量。
- 这种核的频率响应是 c (jω + a) / ((jω + a)² + b²)，实部 = c a (a² + b² + ω²) / |(jω + a)² + b²|² > 0【计算】，对任意 b 都成立，所以只耗能（正实）。
- 范围：1/a_i ~ LogU[0.1, 3] s；b_i = 0 w.p. 0.5，否则 LogU[0.5, 8] rad/s；低频增益 c_i a_i / (a_i² + b_i²) ~ LogU[0.05, 1] × 该方向的 D̃₀。
- a_T6 = −M_t⁻¹ μ。每个子步用精确的二阶离散（与 operators._Sec 相同的旋转–缩放形式）。

**T7 作用在船体随机点上的环境力**（个数 N_f：0 w.p. 0.2，否则 1 + Poisson(1)，最多 4）
- 每个力：
  - 作用点 p_m：x ~ U[x_stern, x_bow]（−2.12 … 3.16 m），y ~ U[−B/2, B/2]，z ~ U[−1.5 T, 0]；
  - 方向 d_m：三维各向同性的随机单位向量；
  - 读浪：ω_m(t) = Σ_j w_{m,j} η_j(t)，η_j 为 5 × 3 个站点在该子步时刻遇到的浪高。空间权重 w.p. 0.5 为以作用点为中心的高斯（宽度 U[0.2, 1] L），否则为 M15 的五种空间形态（平均、纵向坡度、左右差、曲率、扭转）的随机组合；
  - 时间核：随机的稳定因果滤波器，阶数 0（纯静态，w.p. 0.2）、1 或 2：H(s) = (d₀ + d₁ s/ω_k) / (1 + 2ζ s/ω_k + s²/ω_k²)，ω_k ~ LogU[1, 15] rad/s，ζ ~ LogU[0.2, 1.5]，(d₀, d₁) ~ N(0, I)，按 0.3–13 rad/s 上的最大增益归一为 1。分子里的 d₁ 让力超前（相当于读浪高的变化率），极点让力滞后；合起来在遭遇频率上可以是任意相位。w.p. 0.3 另加一个二次项 β (k₂ * ω)²（去均值后）：一个方向上的平均力，比如波浪增阻；
  - 大小用物理单位，随海况自动变化：F_m = m_v A_m (k * ω_m)，A_m = ω_h² × LogU[0.01, 0.3]（每米浪高多少 m/s²；ω_h² ≈ 34 1/s² 是"整条船跟着水面走"的静水力，上限取它的 30%）。
- a_T7 = M_t⁻¹ Σ_m J(p_m)ᵀ d_m F_m。
- 没有告诉任何通道读哪种浪：读法随机，作用点随机，通道之间的比例由作用点决定。lofi.PRIOR 的 k_wave_heave / k_wave_pitch 是它的特例（点在重心、读平均浪高或纵向坡度、静态核），所以不必另开那项随机化。
- 与现族的一个区别：现族把每个投影在库上归一到单位标准差，误差大小和海况无关；这里浪力与浪高成正比，海越大误差越大。

**T8 小的通用残差**（总是存在）
- 当前的误差函数族（M15 标志下的版本：读这一步期间的浪、五种空间形态）的"规则"部分整体乘 w_res：w_res ~ LogU[0.01, 0.3]；w.p. 0.15 取 LogU[0.3, 1]（保留旧族作为特例，免得刚体族之外的误差完全没有概率）。按控制步保持，和现在一样。

**T9 噪声**
- 当前的噪声部分原样保留（有色噪声、白噪声、突发、事件大小的随机部分），幅度按它自己的 A 抽取，不乘 w_res。

**全部刚体项关闭**（T1–T7 都不开）w.p. 0.05：真实船 = 低保真船 + 小残差 + 噪声。

### D8.4 加在哪里、时间分辨率、e0 怎样得到

- **刚体部分（T1–T7）每个子步（0.04 s）算一次**，用子步开始时的状态和该子步时刻的浪；通用残差（T8）和噪声（T9）仍按控制步（0.24 s）保持。
- 为什么不整步保持：刚体项大多是状态反馈。升沉 ω_h = 5.8 rad/s，整步保持等于把反馈平均推迟 Δ/2 = 0.12 s，在 ω_h 上相位落后 0.7 rad；一个 ΔK = 0.3 ω_h² ≈ 10 1/s² 的刚度变化因此会顺带产生约 −ΔK × 0.12 ≈ −1.2 1/s 的假阻尼，是标称 4.3 1/s 的 28%【计算】。子步 0.04 s 时降到约 5%。浪力按子步取浪，本身就是"这一步期间遇到的浪"，比 M15 的"步中点"更细。
- 挂载：data2 里 Mission 的残差钩子每个子步调用一次，拿到子步之后的状态 s（ZOHInjector 就是这样把速度"踢"进去的）【读代码】。新的注入器记住上一子步踢完后的状态，两者之差除以 dt 就是 ν̇₀，据此算 a 并踢 a·dt。阻尼和记忆部分用线性隐式更新（(I + dt Λ)⁻¹），大阻尼时子步也稳定。每个子步要在 15 个站点多算一次水面，粗估让低保真一步慢约 30%，实现后实测。
- **e0 不变**：仍由 e0_from 从状态直接算（真实下一状态 − model0 的下一状态）/ 步长。修正在闭环里改变了船的运动，所以 e0 自动包含：低保真船自己的浪响应和执行机构（和现在一样）、刚体修正在这一步里的平均效果，以及它在步内经过船体的快速响应。e0 的 5–7 通道（升沉、纵摇位移、艏向）自动跟着变；8–9（执行机构）不受影响。
- **重新贴标签（relabel）**：刚体修正是状态反馈，在一条没被它推动过的轨迹上算出的误差，状态统计会不自洽（阻尼多了的船配一条阻尼少的轨迹）。建议刚体部分只在闭环里生成，relabel 只用于通用残差。
- 截断：总推力仍按 CLIP × A_REF 截断（D5）；检验要求被截的步少于 0.1%。
- 接入时机：等 M15 的任务改完 data2.py / operators.py 之后，再在新文件（例如 learn/meta/operators_rb.py）里实现、接入 data2。

### D8.5 所有抽样共享什么，每个回合变什么

**共享**（每个抽样都满足；网络可以把它们记在权重里）：
- 误差 = 真实船的方程 − 简化模型的方程，再经 M_t⁻¹ 分到通道。误差只能来自这些量：速度、姿态、低保真船此刻的加速度、执行机构位置、浪、慢变量。
- 质量正定；总阻尼不为负（所以像阻尼的误差更常见）；记忆只耗能。
- 左右对称：纵向平面 {纵荡、升沉、纵摇} 和横向平面 {横荡、艏摇} 互不耦合。
- 一个点力同时推两个或三个通道，比例由作用点和惯性决定（横荡 : 艏摇 = 1 : x/ρ_r²；升沉 : 纵摇 = 1 : −x/ρ_q²）。
- 与质量变化配套的耦合不做功，而且与航速成正比（所以误差随航速变）。
- 系数随航速平滑变化（二次以内）。
- 浪力与浪高成正比。
- 真实船在工作航速下稳定。

**每个回合变**（要从历史里认）：各个系数的数值和它们随航速的形状、力的作用点和方向、读浪的空间形态和时间核、记忆的时间常数、质量耦合的大小和符号、通用残差和噪声。

### D8.6 一个回合要认多少量，60 秒历史能给多少

仿照第 3 节。在固定航速附近，刚体族的误差对每回合的未知数是线性的（状态、加速度、指令、滤过的浪各乘一个系数），所以"认"基本上是一次带结构先验的回归。

各通道在当前航速下要定的数（把共线的项合并后的粗计）【计算，粗估】：
- 纵荡：阻力、推力效率各 1，质量 1–2，浪力 1–3，偏置 1：约 5–8；
- 横荡：v、v|v|、r（定速时 U r 与 r 共线）、侧力效能、质量 1–2、浪力、偏置：约 6–9；
- 艏摇：r、v、侧力杠杆、质量、浪力、偏置：约 5–8；
- 升沉、纵摇：各有 z、θ、ż、θ̇、两个低保真加速度、平衡偏移、浪力、记忆：约 8–12。

作用点让横荡与艏摇、升沉与纵摇共用同一个力，质量矩阵对称让两个通道共用同一个耦合数，所以真正独立的数比上面相加要少。

60 秒能给的，按第 3 节 N_eff ≈ B_eq T_h / π，并要求 p ≤ (N_eff − 1)/5（泛化多出的误差不超过 25%）：
- 升沉、纵摇的回归量（z、θ、ż、θ̇）由浪驱动，带宽与遭遇谱相当：N_eff ≈ 16–63，可定 3–12 个 → 当前航速下大部分能认出；
- 横荡、艏摇的回归量（v、r）主要由转向和抖动驱动，带宽约 0.3–1 rad/s：N_eff ≈ 6–20，可定 1–4 个 → 只能认出一部分，其余靠共享结构（耗散的符号、固定的横荡/艏摇比例）和先验；
- 随航速的形状：航速目标每 10–40 s 换一次，60 s 里通常只有 1–3 个航速 → 形状主要来自先验。网络对"要换航速的计划"应给出更宽的误差分布，这正是 MPC 需要的诚实；
- 浪的时间核：同第 3 节（每通道 1–3 个滤波）。

和旧族比：旧族的状态关系经过随机投影和随机网络读出，要认的量多、而且是非线性的；正负号完全对称，使"没有历史时的最好猜测"是零。刚体族在同样的历史下要认的量更少，而且是线性的——Transformer 现场做线性回归是研究得最清楚的情形（Garg et al. 2022；Raventós et al. 2023 指出所需的任务种类随任务维数增长）。这是预期，要由训练后的模型检验，不由推导保证。

### D8.7 对 M15 统计量的预期（未验证，只作检查时的对照）

- **横荡误差与横荡速度的相关**（目标 −0.34，现先验中位 +0.19）：T3 的阻尼平均为正，T6 只耗能，预期先验里负相关的回合明显增多，目标落入中部。现先验中位为正的来源本节没有查清（可能来自执行机构族或低保真船自己的响应）；T8 残差仍贡献一部分。
- **纵摇误差对航速的斜率**（目标 +0.31，现先验约 0，48% 的目标回合高于 95% 分位）：平衡偏移 Δθ_eq(ξ)、∝ U² 的 Munk 刚度、随航速变的系数都会在回合内产生斜率。符号对称，所以预期先验的斜率分布变宽，目标落到大约上四分之一，而不是边缘。
- **升沉–纵摇同步**（目标 −0.27，先验 −0.04）：质量非对角元、刚度耦合、竖直点力都会产生同一步的同步。点力的符号由作用点在重心前还是后决定；这条船的船体在重心前 3.16 m、后 2.12 m，均匀取点有 60% 在前，所以自然偏负（几何事实，不是按目标调的）。预期分布变宽、略偏负。
- **纵摇误差与纵摇角速度的相关**（目标 −0.05，现先验 +0.24，来自低保真船自带的浪响应）：阻尼偏正会把它往负拉。
- **纵摇误差的前后相关**（目标 0.20，先验 0.53）：没有明确预期。刚体项多与 ż、θ̇、低保真加速度成正比，它们在遭遇频率上振荡，隔一步的相关可高可低。
- **横荡误差大小**（目标 2.75，先验中位 1.0）：T2、T3、T5 都加在横荡上，预期先验变大。
- **风险：纵摇的量级**。纵摇回转半径只有约 1.35 m，一个 2 m/s² 的竖直力作用在船首附近，就给出约 2 rad/s² 的纵摇加速度，远大于目标纵摇误差的 0.55；质量耦合和 Munk 项同理。上面的范围已按此取小（质量非对角 σ = 0.1，点力上限为静水力的 30%），但要由先验预览确认 rms_pitch 的分布仍包含目标。调整时只放宽或收窄尺度，不移动中心（反泄漏规则）。

### D8.8 量级核对（只作核对，不用来设定范围）

典型状态取目标 C 组每回合的中位数【测量，本次从 meta3/C.npz 计算】：v 0.89 m/s，r 0.24 rad/s，ż 0.99 m/s，θ̇ 0.15 rad/s，z 0.37 m，θ 0.048 rad，航速 11.3 m/s。

| 项 | 例子 | 误差加速度 | 目标误差大小（M15 每回合中位） / A_REF |
|---|---|---|---|
| 横荡阻尼 ×2 | 1.2 1/s × 0.89 m/s | 1.1 m/s² | 2.75 / 4 |
| 艏摇阻尼 ×1.5 | 1.2 1/s × 0.24 rad/s | 0.29 rad/s² | 0.93 / 2 |
| 横荡科氏，Δm_u = 20% | 0.2 · 1969 · 11.3 · 0.24 / 1610 | 0.66 m/s² | 2.75 / 4 |
| Munk 艏摇，Δm_v − Δm_u = 300 kg | 300 · 11.3 · 0.89 / 5400 | 0.56 rad/s² | 0.93 / 2 |
| 升沉刚度 ×1.2 | 6.8 1/s² × 0.37 m | 2.5 m/s² | 4.85 / 12 |
| 升沉阻尼 ×1.3 | 1.3 1/s × 0.99 m/s | 1.3 m/s² | 4.85 / 12 |
| 纵摇平衡偏移 1° | 0.0175 × 13.8 | 0.24 rad/s² | 0.55 / 3 |
| 推力效率 −15% | 0.15 × 2970 N / 1969 kg | 0.23 m/s² | 1.57 / 4 |
| 一个点力 A = 0.05 ω_h²，Hs 1 m | 0.05 · 34 · 0.25 m | 0.43 m/s²（升沉） | 4.85 / 12 |

目标横荡误差里"像阻尼"的部分约 0.34 × 2.75 ≈ 0.9 m/s²，与"横荡阻尼加倍"同量级：目标在范围内部，不在边缘。

### D8.9 网络怎样把观测和误差联系起来（写给用户）

**先说"纯随机"是什么意思。** 先验里每个回合抽一条"假想的真船"：它比简化模型多了多少阻尼、多了多少质量、额外的力作用在船的哪里、它对浪反应得多快……这些数是随机的。但随机的只是数值，规律的样子是固定的。打个比方：抽的是"不同的船"，但每一条都是船，都守同样的力学。

**网络学到两层东西。**
1. **所有回合都一样的东西，记在权重里**，不用看历史就知道。旧的投影族里这一层几乎是空的：误差可以读任何投影，正负号完全对称，所以没有历史时，网络对"误差和横荡速度是什么关系"的最好猜测是零，一切都要现场认。刚体族里这一层有实在的内容（D8.5）：阻尼只能耗能，所以"误差和速度反着来"更常见；同一个力同时推横荡和艏摇，比例由作用点定；升沉和纵摇也一样；这些关系随航速平滑地变；浪越大，浪力越大。
2. **每个回合不同的东西，从这一回合的历史里现场推断。** 网络看到最近约一分钟的状态、指令、（浪高）和每一步的误差，相当于在心里做一次回归。例如："这条船横荡时，误差总和横荡速度反着来，大约每 1 m/s 多 1 m/s²，所以它的横荡阻尼大约是模型的两倍"；"升沉误差大的时候，纵摇误差总是反向跟着，说明多出来的竖直力在重心前面"。然后把认出的规律用到 MPC 计划里的未来状态上。

**观测输入怎样进入这个关系。** 误差的真正原因是状态、加速度、指令和浪；网络能用的也只有这些观测。关系本身不写在任何一个观测里，而是在"观测和误差的历史放在一起看"里：单看横荡速度，什么也看不出；把速度和误差的历史对起来，关系就显出来了。所以网络的输入必须同时有这两样，历史也要够长（D8.6：升沉、纵摇够用，横荡、艏摇只够认一部分，其余靠第 1 层）。浪也一样：把误差历史和浪高历史对照，才能认出"这条船对哪种浪、在哪个位置、反应多快"。先验要保证的是：训练时网络见过足够多种"船"，学会的是"怎样从历史里认出规律"这一招，而不是某一条船的规律。

**这一招什么时候不灵。** 如果真实误差不像任何一条"刚体船"（比如某种非刚体的效应），刚体族给不了它概率；通用残差（T8）接住一部分，其余会被当成噪声——网络会给出更宽的不确定范围，而不是编出错误的关系。要认的量太多、历史太短时，网络应该退回第 1 层给出的平均猜测，并把范围放宽。这些都要在训练好的模型上检验，推导本身不能保证。

**和"把经验写进参数"的关系。** 历史窗口只有约一分钟，窗口以外认出的东西会丢；路线图第 8 节已定要把经验写进少量参数（只调输出头、向先验收缩）。两者不冲突；预期刚体族会让要写进参数的东西变成少数几个有物理意义的数，这一点同样要检验。

### D8.10 实现前后要做的检查（都还没做）

- **单元检验**（新文件，例如 studies/test_operators_rb.py）：
  - νᵀ ΔC(ν) ν = 0；竖直平面的换系结果与船体坐标公式逐项一致（锁定 T2 的符号）；
  - 点力方向：船首向上的力 → 升沉加速度为正、纵摇加速度为负（船首抬起）；船尾向左舷的力 → 横荡为正、艏摇为负；推力线在重心以下 → 抬首；
  - 耗散：对随机 ν，νᵀ D_t ν ≥ 0；记忆核在频率网格上正实；
  - 抽到的稳定性检查确实拒绝不稳定的抽样；
  - 全关的抽样推力严格为零；同一种子可复现；
  - 子步注入：对一个纯线性的抽样（只改阻尼和刚度），闭环结果与"直接改系数的解析解"一致到子步离散误差以内。
- **先验预览**（例如 studies/prior_preview_rb.py；描述性，不判断信息量或上限）：
  1. 最便宜的一版：在已有 A 组的记录轨迹上，把 a(记录的状态、浪) 加到已有的 e0 上，算 M15 的统计量。只是预览：修正没有推动过这些轨迹。
  2. 小批闭环（几十个回合，一个进程），重算 M15 统计量，看目标是否落在中部，以及 rms_pitch 是否过大。
- **第二个目标**：按 M15 的提醒，同时和 Wigley 排水型船的高保真数据比较，避免把先验改成只适合 Scarab。

### D8.11 未决问题

1. **左右对称当作结构**：两个平面之间的耦合设为零。偏载、单侧浪冲击会破坏对称；是否允许小的跨平面项（例如 w.p. 0.1、σ = 0.03）？
2. **升沉、纵摇的惯性比**（μ_w、ρ_q）简化模型里没有，只能随机抽；范围是按刚体加附加质量估计的。
3. **只能闭环生成**：刚体部分不能靠 relabel 免费扩增，同样多的回合要多花仿真时间；每个子步多一次水面计算也要实测。
4. **与旧族的比例**：本节取"刚体族 + 小残差"为主，15% 的抽样残差取全权重。这个比例没有依据，只能由预览和训练后的检验决定。
5. **T2 竖直平面项的符号**要由单元检验锁定（本节是手算的换系）。
6. **纵摇量级**：耦合项可能让纵摇误差普遍偏大（D8.7 风险），预览后可能要收窄范围。
7. **接入时机**：要等 M15 改完 data2.py / operators.py 再接；注入器改为每个子步调用，需要改 data2 的一处（届时再动）。

### D8.12 审查后的修订和第一版实现（2026-09-30；同日按第二轮审查修正）

本节覆盖 D8.3–D8.11 中与它冲突的地方。实现：learn/meta/operators_rb.py（独立模块，data2 / operators 未改，未接入数据生成）；检验：studies/test_operators_rb.py；预览：studies/prior_preview_rb.py。第二轮审查（下称"复审"）找出的 9 处问题已按下面各条修正；旧的预览日志 prior_preview_rb.log 作废，现行日志是 prior_preview_rb_v2.log（闭环用 M10 执行机构族）和 prior_preview_rb_v2_ideal.log（闭环、理想执行机构，只作对照）。

**第一版包含什么（审查第 16 条：分阶段）**
- 做了：T3（耗散）、T4（回复）、T5（喷泵效能，含 Nomoto 结构误差）、T2c（与航速成正比的耦合，独立于质量变化）、Teven（左右对称允许的跨平面耦合，新）、T7（环境力，按审查重写）、T8（旧族规则，权重降低但事件等保留）、T9（旧族噪声，权重为声明的超参数）。
- 推迟：T1（质量变化）、由质量变化导出的 T2、T6（辐射记忆）。推迟 T1 后，真实船的惯性在每个通道上取低保真船自己的值（纵荡用 m_surge，横荡用 m_sway），未知的几个（艏摇 I_r、升沉 m_w、纵摇 I_q）每回合抽，所有项都用同一组。

**一个根本改动（第 1、10 条）：竖直方向的修正都量"相对水面参考"的运动。** 低保真船的回复力本来就作用在"升沉减去它自己的水面参考（平均浪高 + z0）""纵摇减去它自己的坡度参考（坡度角 + θ0）"上。D8 原稿把刚度变化作用在"相对航行姿态"的位移上，等于每次改刚度都顺带改了船在长浪里的跟随量，违反"不规定哪个自由度读哪种浪"，也违反"很长的浪里任何浮体都跟着水面走"。现在：
- T4 严格是 a = −[(K_t − K0)(η − η_surf) − K_t Δη_eq]，η_surf 就是 ReducedModel.step 在这个子步用的那个参考。只是把低保真已有的一项放大或缩小，没有新的读浪。
- 升沉、纵摇的阻尼变化作用在"相对这个参考的速度"上；T2c 的竖直耦合同样用相对量；纵摇的 Munk 刚度量的是 θ − θ_ref。
- **复审第 1 条修正了参考的变化率**。第一版用的是"固定地点的浪高变化率"（Σ a ω sin 相位），而参考是在随船移动的站点上读的，它真正的变化率要算上船在浪里走动：Σ a (ω − k (cosθ·Ẋ + sinθ·Ẏ)) sin 相位，Ẋ、Ẏ 是站点在地面上的速度（由 u、v、r、ψ 和站点偏移算出）。这就是"遭遇频率"和"浪的频率"的差别。任务航速 8–15 m/s、谱峰周期 4–7 s 时 k·u 约 0.7–3.9 rad/s，而 ω 只有 0.9–1.6 rad/s：迎浪时真实变化率是旧值的 1.6–3.5 倍，顺浪时接近零甚至反号。所以第一版的"相对速度"其实是绝对速度和一个错频率浪量的混合，完全跟着参考走的船也会被推（12 m/s 下 100 m 长的浪误差约 100%）。现在 RowSeas.stations（给出速度时）和 RBPlant 都按移动站点算变化率，两者用同一组相位。
- 检验 1 改为行为检验：让船沿 12 m/s 的直线或转弯路径、在 100 m 和 30 m 的规则浪里（迎浪、顺浪、横浪、斜浪），位移和速度都正好等于参考（速度用沿路径的有限差分，不用代码里的公式）。代码给出的参考变化率与有限差分一致到 3e-8；竖直修正为 2e-7 m/s²。同样的船若用旧的固定点变化率，竖直修正达 5.5 m/s²。
- T7 的竖直分量在零频率处增益为零。
- **需要用户确认**：相对速度的阻尼读的是"低保真自己的水面参考随时间的变化率"。修正后它就是低保真回复力所读那个量（移动站点上的平均浪高和坡度）的时间导数，不是新的浪量；复审指出的"固定点变化率是新浪量"的问题已经不存在。我们认为它在"不规定哪个自由度读哪种浪"的规则之内。更完整的做法是沿船长逐站点的随机刚度 / 阻尼分布（第 1 条的首选方案），推迟。

**逐条处理（审查 1–19）**
1. 采用"最小方案"加相对速度（上面，变化率已按复审第 1 条修正）。逐站点的分布推迟。
2. 惯性拆成刚体加附加质量：刚体质量取 p 的 m_coriolis（1400 kg），A22 = m_sway − m ≈ 210 kg，A11 ~ U[0.02, 0.1] m（细长船 A11 远小于 A22），A33/m ~ LogU[0.3, 2.5]，ρ_q ~ U[0.18, 0.27] L。有效的 m_surge（1969 kg，由推力阶跃拟合的时间常数）只当纵荡通道自己的除数，不进任何交叉项。竖直方向的航速对角项只以 (U² − U_des²) 进入。
3. 环境力改为：读 M15 五种空间形态的随机组合（每种单位库标准差，四种非平均形态与平均无关），或"每个站点读自己当地的浪（去掉平均部分）"的载荷分布（分子随 x 线性变化，极点共享），一个回合可有多个力，w.p. 0.2 为随机广义方向（允许纯力矩）。"lofi.PRIOR 的 k_wave 是特例"的说法删去。
4. ρ_r ~ LogU[0.22, 0.37] L（1.19–2.0 m），同时覆盖刚体估计（约 0.25 L）和 Nomoto 初始斜率给出的 1.83 m。T5 明确减去低保真的 k_nomoto_f/τ_r。结构误差有意保留并给出大小：每牛顿侧力的艏摇换成 x/I_r，同时把艏摇阻尼乘同一个比 f_r，使辨识出的稳态转艏率不变；f_r 在 0.8–2.3，艏摇时间常数在 0.18–0.51 s。
5. rb_accel 是唯一的刚体步函数；状态是普通字典，snapshot 深拷贝即可从中途分支；simulate_rb 按 relabel.simulate 的样子批量推演（现在可选 M10 执行机构链，用植物自己的 lofi.act_step）。检验 5：从第 k 步的快照继续与原回合自己的继续完全一致（float64，1e-14）。
6. T5 写成力的差：a = M⁻¹ (Jᵀ F − f_nominal)。
7. 刚度随位移的变化改为有界：K × exp(γ tanh(e/ℓ))，|γ| ≤ 0.5，总刚度在 0.61–1.65 倍之间，力随位移单调。T6 推迟，所以特征值检查里没有它；T4 的反对称部分、T2c 的耦合都在检查里。
8. 横向只在"带自动舵的闭环"里检查稳定，不再拒绝开环航向不稳定的抽样；竖直仍要求各模态阻尼比 ≥ 0.05。**复审第 7 条**：闭环检查现在除理想喷口外，还带喷口的一阶滞后加延迟（一阶 Padé）：目标船那样的 0.1 s / 0.12 s，以及 M10 族里最慢的 0.5 rs / 0.3 rs（约 0.37 s / 0.22 s）；速率限制是非线性的，没有进检查。低保真船本身在最慢执行机构下仍稳定（最大实部 −0.76 1/s）。重抽比例从约 9% 升到约 17%（300 个种子），其中理想喷口就不稳的约占 2/3，其余是执行机构检查多拒的。拒绝会改变被接受的 λ 分布（见第 17 条）。检验 4 现在用 M10 执行机构族（lofi.draw_act，每行一抽）跑 90 s 闭环：刚体部分相对低保真单独有界。
9. RBPlant（在新模块里继承 ReducedPlant，不改 lofi.py）记下每个子步开始时植物在 5 × 3 站点的浪高，以及沿移动站点的变化率（与植物用的相位是同一次计算；植物自己用的浪高和固定点变化率原样返回，所以植物本身逐位不变）。检验 6：Mission 路径（RBPlant + RBInjector）与批量推演一致到 1e-14。站点下标 0 是右舷，空间权重按 y_off 的真实坐标。
10. 同第 1 条。
11. (a) 偏向耗散写进范围：阻尼倍数 c0 ~ U[log 0.6, log 2.4]，P(倍数 > 1) = 0.63，另有只能为正的点阻尼和二次阻尼。理由（弱，照实说）：低保真的阻尼只在一个航速、一个幅度上由衰减试验辨识；它缺的部分（横摇、辐射记忆、横流阻力、飞溅）大多从这五个通道带走能量。仍保留两种符号。(b) 预览按分量分别报告。(c) T9 的权重 w_noise ~ LogU[0.1, 1]，是声明的超参数。
12. 事件（连同它的随机大小部分）、继电器、常值偏置、慢漂移按 v0 自己的大小保留；只把滤波器 + 读出的规则部分乘 w_res ~ LogU[0.1, 1]。D8.9 里"刚体族以外的误差会被当成噪声、给更宽的范围"的说法不对（M7、M10、M11 看到的是推演技能大于 1、覆盖率 0.3–0.8），删去，改为：刚体族以外的误差只能靠 T8，T8 太小时模型会出偏差，要在目标上检验。检验 4 里全部分量时航速最低到 −6.7 m/s（M10 执行机构下），来自保留原大小的旧族偏置和事件，不来自刚体部分。
13. 同第 3 条；幅度按"每个库标准差"给出，海越大力越大。
14. 只写新文件；lofi.py 未改；一个批量函数；float64 回放检验。实测代价：rb_accel 每次约 0.85 ms（每个抽样一行），大约让低保真一步的耗时翻倍（D8 估的 30% 偏低，主要是 Python 的开销）。
15. 新加 Teven：纵荡、升沉、纵摇得到横荡速度、艏摇角速度、喷口角的二次式（转弯掉速、下沉、纵倾变化，有界）；横荡、艏摇的阻尼随升沉、纵摇、升沉速度、航速有界地变化（总阻尼不低于一半）。未决问题 1（小的非对称跨平面项）另议，第一版为零。
16. 采用分阶段（上面）。数据量按 M14 的约 2 万个回合计划，本次不生成数据。
17. **T2c 按复审第 2、4 条改写，恢复"不做功"。** 第一版的横荡科氏、纵荡 v r、Munk 力矩三个倍数独立抽，合起来会做功（例子：u = 12 m/s、v = 1 m/s、r = 0.3 rad/s 时注入约 4.9 kW，而低保真横荡阻尼只带走约 2.1 kW），横荡科氏还错用了 A22（应为 A11）。现在：
   - 低保真的横荡方程有 −m u r，纵荡却没有对应的 +m v r，两者合起来本身就做功。T2c 补上刚体的 m v r（每个刚体开启的抽样都有，是第三个"总在"的结构项，和 Nomoto 部分、下面的 T5 一样声明）。
   - 附加质量部分只用**一个**共同的倍数 λ（两种符号，λ ~ U[−1, 1.5]，随航速平滑，w.p. 0.7；λ = 1 是 Fossen 的标准形式，换到 y 向左舷、z 向上）：纵荡 +λ A22 v r，横荡 −λ A11 U r，艏摇 −λ (A22 − A11) U v。对任何 λ，这三项加上低保真的 −m u r 的总功率都严格为零。Munk 形式只在小漂角成立：λ 乘同一个标量饱和因子（漂角 0.1 rad 处饱和，v 接近零时为 1），标量因子不破坏"不做功"。
   - 竖直：升沉 ∝ U × 相对纵摇角速度、纵摇 ∝ U × 相对升沉速度，两者现在是同一个系数的反对称形式（按抽到的惯性），不做功；大小仍按低保真升沉、纵摇阻尼的几何平均 × LogU[0.05, 0.5]（A33 U 这类刚体大小会给出比目标全部纵摇误差大十倍的纵摇加速度）。纵摇 Munk 刚度 ∝ (U² − U_des²)(θ − θ_ref)（U[−0.3, 0.3] ω_p²）是刚度，不属于"不做功"的要求。
   - 新检验：T2c 加低保真科氏项的总功率为零（包括漂角饱和的状态），相对误差 3e-16。D8.5 里"与质量配套的耦合不做功"因此重新成立。
   - 稳定性检查会改变被接受的 λ：抽取时中位 0.25、60% 为正，被接受的中位约 −0.05、48% 为正（Munk 力矩 λ > 0 时让航向更不稳）。
18. ξ 由任务的航速范围给出（构造参数）；平衡偏移用吃水 T 和 T/L 表示；频率范围乘 1/rs；力的作用点只在每回合随机的"湿长"内（从艉封板量起，U[0.4, 1.0] 倍船长）；横向在闭环里检查。
19. 采用：回放只当量级检查；M15 统计量看闭环小批（降阶世界）。

**审查之外、实现时加的几处（都来自预演，都已写进范围）**
- T7 的水平分量是竖直尺度的 κ_h ~ LogU[0.05, 0.5] 倍（船体湿表面的法向主要朝上）。各向同性方向加竖直尺度时，横荡出现过 20–40 m/s²。
- T7 的上限：升沉部分以 ω_h² × 平均浪高标准差（≈ 7.8 m/s²）为尺度，纵摇部分以低保真自己的纵摇浪激励（ω_p² × 库中坡度角的标准差 ≈ 0.53 rad/s²）为尺度，每个力的每单位读数不超过 A 乘相应尺度（A 是它自己抽到的幅度比例）。**复审第 5 条**：第一版的纵摇上限只管竖直分量，漏了水平分量的力臂 z·F_x（最多约 4 倍超限）；"逐站点"的力按"各点同相"定上限，而这些读数去掉了平均、船首船尾可以反号（最多约 3 倍超限）。现在上限按力臂 |z κ_h d_x − x d_z| 算（整个升沉 / 纵摇通路再乘一个因子），逐站点的力按各点贡献的绝对值之和（最坏的符号组合）算。检验 1 在所有抽到的力上按最坏情况检查（纵摇比例最大正好 1.000，129 个力里 28 个被这条上限压住）。
- Munk 类项的漂角饱和（第 17 条）。
- **T5 的轴向损失（复审第 6 条）**：低保真的纵荡用全部推力 T，而偏转的喷流只有 T cos δ 向前。第一版在约 95% 的抽样里固定地加上 T (cos δ − 1)/m_u（满推力满舵约 −0.30 m/s²），没有声明。现在它乘一个每回合的比例 k_c ~ U[0, 1]，只在 T5 效能部分开启时（w.p. 0.7）有，其余时候为零。检验：标称抽样（k_c = 0）在喷口偏转时五个通道都严格为零；k_c = 0.6 时纵荡正好等于 0.6 T (cos δ − 1)/m_u。

**总在的结构项（每个刚体开启的抽样都有，网络会把它们当共享结构学进权重）**：Nomoto 部分（每牛顿侧力的艏摇 x/I_r、艏摇阻尼乘 f_r）；纵荡的刚体科氏项 m v r（第 17 条）。其余每项都按概率开启。

**超参数（全部是假设，只能由预览和目标检验确认）**：见 operators_rb.py 顶部的常数和 _draw / _draw_env。主要的：刚体全关 0.05；T3 0.85，T4 0.7，T5 效能 0.7（k_c ~ U[0, 1]），T2c 0.8（其中 λ、竖直耦合、Munk 刚度各 w.p. 0.7），Teven 0.5；T7 个数 0 w.p. 0.15，否则 1 + Poisson(1.2)，最多 5；幅度 LogU[0.02, 1] × 上面的尺度；时间核一阶或二阶，ω_k ~ LogU[0.7, 11]/rs，ζ ~ LogU[0.2, 1.5]；稳定性检查用的执行机构见 ACT_CHECK。

**和 data2 / relabel 接入时要改的地方（复审第 3 条；这些文件现在由 M15 在改，本次不动）**
- OperatorRB 的状态是嵌套的（残差算子的状态在 st['res'] 下）。恢复慢状态要用 operators_rb.set_slow(op, st, slow)（两个族都能用），relabel._warm 和 Operator.run 里直接写 st['ga']、st['bb']、st['regime']、st['rl']、st['rl_age']、st['k'] 的几行要换成它。st['k'] 现在只是残差算子时钟的镜像，不再单独计数。
- OperatorRB.run() 直接报错（NotImplementedError）：刚体部分是状态反馈，不能在记录的输入上重新贴标签（D8.4）。data2.build_targets 调 op.run(...) 的两处（目标的"自己的未来"和"捐赠者"）对刚体族要改成 simulate_rb 闭环推演；只要 T8/T9 部分时用 run_residual()，调用方自己加刚体部分。
- 挂载：植物要用 rb_plant(...) 包成 RBPlant，注入器用 RBInjector。RBInjector 有和 ZOHInjector 一样的 .r：data2 的 inj.r = rr + rn 照旧可用（施加的推力与分开设 rule / noise 完全相同，因为 clip_push 的两部分之和总是 clip(rule + noise)）；读 .r 得到上一控制步各子步实际施加推力的平均，这是 data2 应记录的 R。
- op_family 标志、data6、relabel.simulate / data5.simulate_mid 里调用 rb_accel，都等 M15 完成后再做。

**检验（studies/test_operators_rb.py；复审第 8 条）**：标为 [regression] 的几条（惯性为正且有序、滤波器稳定、零频增益为零、带内增益 ≤ 1、95% 的抽样低于截断）是按构造就成立的，只防以后改坏，不能发现公式错误。行为检验：总阻尼不做正功；T2c 加低保真科氏项功率为零；点力符号和比例；所有抽到的力在最坏读数下不超过升沉 / 纵摇上限；T5 标称为零、轴向损失等于声明值；跟着移动参考走的船竖直修正为零；稳定性规则拒绝无阻尼抽样；截断按 D5 的规则统计"任一通道、任一子步被截的控制步"，要求少于 0.1%（刚体部分单独为 0）；闭环用 M10 执行机构；快照继续；Mission 路径；接口（run 报错、set_slow、RBInjector.r）。全部 7 组通过（478 s）。

**量级（代替 D8.8 中按"相对航行姿态"算的刚度、阻尼行）。** 回放（100 个 meta3 回合，每回合均方根的中位数，m/s² 或 rad/s²，顺序纵荡 / 横荡 / 艏摇 / 升沉 / 纵摇）：T3 0.09 / 0.55 / 0.14 / 0.39 / 0.08；T4 0 / 0 / 0 / 1.30 / 0.21；T5 0.07 / 0.05 / 0.12 / 0 / 0.04；T2c 0.20 / 0 / 0 / 0.02 / 0.08（纵荡主要是刚体 m v r）；T7 0.03 / 0.05 / 0.03 / 0.03 / 0.01；T8 0.19 / 0.25 / 0.13 / 0.53 / 0.18；T9 0.14 / 0.12 / 0.06 / 0.38 / 0.09；全部 0.58 / 1.23 / 0.39 / 2.68 / 0.62。

**预览结果（studies/prior_preview_rb.py；只是描述，不判断信息量或上限）**
- 旧族的对照统一用 prior_check_m15.log（2279 个回合，带 meta3 的执行机构族），不再混用回放里的 'old*' 列（同样 100 个记录回合走同一条回放路径，只作回放内部的对照；它和日志的数有出入，例如纵摇对航速斜率的高尾 57% 对 48%）。
- 回放（100 个回合）只作量级检查：它忽略推力对轨迹的闭环作用。闭环小批（64 个回合）在降阶世界里跑，现在带 M10 执行机构族（每回合按 data2 的 'm10' 抽），e0 里执行机构那一份和记录的一样来源。
- 目标回合在先验中的中位百分位（括号：低于 5% / 高于 95% 分位的目标回合比例），旧族 → 新族回放 / 新族闭环（M10）：
  - 横荡误差与横荡速度的相关（目标 −0.34）：2.0%（0.93 / 0）→ 53% / 36%（0 / 0）。
  - 纵摇误差与纵摇角速度的相关（目标 −0.05）：4.3%（0.59 / 0）→ 24% / 19%（0.03 / 0）。
  - 升沉–纵摇同步（目标 −0.27）：10.8%（0.09 / 0）→ 10% / 11%（0.20 / 0）。
  - 纵摇误差的前后相关（目标 0.20）：13.3%（0.21 / 0）→ 12.5% / 16%（0.04 / 0）。
  - 纵摇误差对航速的斜率（目标 +0.31）：94.9%（0 / 0.48）→ 91% / 95%（0 / 0.26 / 0.67）。回放里高尾从 48% 降到 26%；闭环里反而是 67%，比第一版闭环（理想执行机构，29%）差。同一批闭环换成理想执行机构（prior_preview_rb_v2_ideal.log）也是 95.3%（0 / 0.72），所以变差不来自执行机构，而来自本次的修正。最可能的原因（未单独验证）：第一版错用的固定点变化率与遭遇频率之差正比于 k·u，等于一个随航速变大的假纵摇浪力，它给了斜率一部分"改善"；去掉后，先验斜率的 95% 分位从 +0.36 缩到 +0.22。航速斜率仍是没解决的结构问题。
  - 大小：横荡 85.7% → 83% / 100%（闭环 74% 的目标回合高于 95% 分位），艏摇 80.6% → 87% / 95%（54%），纵摇 41.8% → 39% / 45%，升沉 65.5% → 75% / 83%。
- 分量单独的闭环（24 个回合，'none' = 低保真船单独）：低保真船单独时纵摇前后相关已是 0.63、纵摇误差与纵摇角速度相关 +0.30、升沉–纵摇同步 −0.09。所以纵摇前后相关偏高、同步不够负，主要由低保真船自己的浪响应决定，不是旧族噪声。横荡"像阻尼"的相关现在只来自 T3（单独时中位 −0.91）和 T5（+0.08，比第一版弱）；T2c 改为不做功后，单独时不再给出像阻尼的横荡相关（第一版的 −0.71 正是来自做功的那部分），这是闭环横荡阻尼相关从 47% 降到 36% 的一个原因。
- 截断：回放中整个推力被截的子步约 1–2%，超过 0.1% 的要求，主要来自 T8/T9 在记录输入上的事件和大偏置（刚体部分单独为 0）。检验 4 的闭环（M10 执行机构，全部分量）被截的控制步 1.7e-4，也略超 0.1%；刚体部分单独为 0。接入后要在真实闭环里重测，超出的部分属于保留原大小的旧族部分（第 12 条）。

**对 D8.7 预期的更新**：横荡阻尼相关、纵摇阻尼相关两条预期得到支持（横荡的支持现在只靠 T3 和阻尼偏向）；航速斜率在回放里部分改善，闭环里没有改善；升沉–纵摇同步"变宽、略偏负"的预期没有出现。同一步同步最直接的来源是推迟的 T1（质量矩阵的非对角元），这是加 T1 的第一个理由。闭环里横荡、艏摇误差偏小；按反泄漏规则只放宽尺度、不移中心，候选是 w_res 的范围和 T7 水平尺度 κ_h 的上限，等 T1 的决定一起定。

## D9 — 误差先验改为"受任意力的刚体"（2026-09-30；D9.1–D9.7 推导，D9.8 实现和修订）

起因是用户的一句话（原话）："刚体物理的受力你只考虑在波浪这些模型里的受力吗？你应该考虑更一般的方案，就是刚体受到任意力"。D8 / D8.12 的各项（阻尼、回复、喷泵效能、辐射记忆、读浪的点力）都是船舶水动力学里的具体形式，还加了"总阻尼只耗能""耦合不做功""左右对称""工作航速下稳定"这些限制。本节改为：**刚体力学只规定"任何力怎样变成运动"，力本身不限定形式。** 依据的标记同 D8：【读代码】【计算】【假设】【文献】。D9.1–D9.7 是写代码之前的推导；实现在 learn/meta/operators_gen.py，实现时的修订和理由在 D9.8，两处不同时以 D9.8 为准。

**边界（照旧遵守）**
- 低保真世界的物理不改：修正仍是加在五个速度通道上的加速度，低保真船的代码一行不动。
- 不规定哪个自由度读哪种浪。
- 反泄漏：目标（Scarab）的统计只作检查；范围来自低保真船自己的尺度，理由写明；检查之后只放宽尺度，不把中心往目标挪。
- D1：不用弱模型（岭回归、线性相关之类）估计上限或信息量；能力只看训练好的模型本身。

### D9.1 刚体力学本身规定了什么（只有这些）

**通道和坐标【读代码】**
- e0 共 10 个通道（learn/meta/data3.py 的 CH7、E7、e0_from）：
  - 0–4：速度的变化率误差，依次是纵荡速度 u、横荡速度 v、艏摇角速度 r、升沉速度 ż、纵摇角速度 θ̇（正是 D8.2 的 ν，顺序相同）；
  - 5–7：位置的变化率误差，依次是升沉位移 z、纵摇角 θ、艏向 ψ；
  - 8–9：执行机构，实际减指令的推力、喷口角（D7）。
- 坐标同 D8.2：x 向前、y 向左舷、z 向上；θ 为正时船首向下（sign_pitch = −1）；横荡、艏摇在船体坐标里，升沉用地面竖直速度 ż。
- 低保真的一个控制步 0.24 s 分 6 个子步，每个子步 0.04 s（sim/lofi.py，operators_rb.light_env）。

**(a) 误差只进加速度，运动学是精确的。** 位置和速度之间的关系——ẋ = u cosψ − v sinψ，ẏ = u sinψ + v cosψ，ψ 的变化率是 r，z 的变化率是 ż，θ 的变化率是 θ̇——是状态各分量之间的恒等式，对任何船都一样，不会错。所以：
- 误差只能以"力 → 加速度"的形式进入 0–4 通道；
- 5–7 通道是这个加速度在一步之内积分出来的，自动跟着变，不单独抽；
- 8–9 是执行机构自己的动态（D7），不是作用在船体上的力，不在本节内；执行机构的实际位置是力的一个输入。【读代码 + 计算】

**(b) 一个未知的正定惯性把力变成加速度。** 真船满足 M_t ν̇ = f_t。M_t 对称正定（动能总为正），包含船体的质量、转动惯量、重心位置，以及水里"与此刻加速度成正比"的那部分力（附加质量：船加速时要带着周围的水一起加速）。后者必须放进 M_t，不能放进"任意力"：一个依赖此刻加速度的力不是"只看过去"的函数，方程会自己咬自己（加速度决定力，力又决定加速度）。其余一切力都在 f_t 里。【计算】

**(c) 作用在船上一点的力 = 一个力 + 力臂产生的力矩。** 点 p = (x, y, z) 上的力 F 换成五个自由度的广义力（D8.2）：

    Q = (F_x,  F_y,  x F_y − y F_x,  F_z,  z F_x − x F_z)

- 一个侧向力同时推横荡和艏摇，比例由 x 决定；一个竖直力同时推升沉和纵摇，比例由 −x 决定（船首的向上力让船首抬起，θ 减小）。换成加速度还要乘 M_t⁻¹：惯性是对角时，横荡 : 艏摇 = 1/m_v : x/I_r，升沉 : 纵摇 = 1/m_w : −x/I_q。这个固定比例是"点力"特有的，和力读什么、有多大无关。
- 几个点力或沿船长分布的载荷加起来就是一个任意的广义力：两处不同 x 的竖直力能凑出任意的（升沉力，纵摇力矩）；一对大小相等、方向相反、不在一条线上的力（力偶）给出纯力矩。所以"点力之和"本身不限制任何东西；它只是在力的个数少时，让"通道之间有固定比例"有较大的概率。

**(d) 刚体自己的科氏 / 向心项属于刚体力学。** 在跟着船转动的坐标里写牛顿定律，会多出一些与速度乘积成正比的项。它们的形式完全由刚体的质量 m、重心相对参考点的位置 r_g、转动惯量 I 决定，合起来不做功（总功率为零）【文献：Fossen 2011 的刚体方程】：

    力：   m (v̇ + ω × v + ω̇ × r_g + ω × (ω × r_g))
    力矩： I ω̇ + ω × (I ω) + m r_g × (v̇ + ω × v)

- v = (u, v, w) 是参考点的速度，ω = (0, q, r)：低保真世界没有横摇，横摇被锁住，锁住它的约束力矩不计。
- 含 v̇、ω̇ 的部分是质量矩阵（含重心偏移造成的非对角元），其余是科氏 / 向心部分。
- 重心就在参考点、惯性主轴与船体轴一致时，只剩：纵荡 +r v − q w，横荡 −r u，船体坐标里的升沉 +q u；纵摇、艏摇没有这类项。惯性主轴在纵剖面内稍有倾斜（惯性积 I_xz ≠ 0）时，纵摇多一项 ∝ I_xz r²，艏摇多一项 ∝ I_xz q r，都包含在上面的公式里。
- 换到低保真的坐标：升沉用地面竖直速度，w ≈ ż + u θ（D8.2）。换系之后船体坐标里升沉的 +q u 正好被换系项抵消——在地面竖直方向上，加速度就是竖直方向的力除以质量，没有科氏项【计算：z̈ = ẇ − u̇θ − uθ̇ ≈ Z/m + q u − u̇θ − u q = (Z − θX)/m】。所以在低保真坐标里，刚体科氏项主要出现在纵荡、横荡（重心偏移和 I_xz 再带到其他通道）。
- 低保真已有横荡的 −m u r（m = m_coriolis = 1400 kg，真实的刚体质量；control/reduced.py 的横荡方程），缺纵荡的 +m v r（D8.12 第 17 条）和 −q w。
- **处理**：刚体科氏差 =（真船的这组项）−（低保真已有的 −m u r），形式固定，只有 m、r_g、I 未知；每个抽样都有，显式写出。这和 D8.12 的"总在的 m v r"是同一类结构项，但现在是完整的公式，不是挑出来的一项。
- 附加质量那部分的科氏项（Munk 力矩之类）是水的力，不是刚体的；在本节里属于任意力（D8 的 T2c 是它的一个特例），不单列。

**(e) 质量变化项可以并入任意力。** D8.1 的恒等式：

    a = ν̇_t − ν̇₀ = M_t⁻¹ (Δf − ΔM ν̇₀)

ν̇₀ 是低保真船此刻自己的加速度，由此刻的状态、指令和浪算出，是"已经经历过的东西"的函数；在子步注入里能直接得到（相邻子步状态之差除以 dt，D8.4）。所以 −ΔM ν̇₀ 只是任意力 Δf 的一个特殊情形：一个读 ν̇₀ 的线性读出。
- **决定：不单列质量变化项**，把 ν̇₀ 作为任意力的一组输入（和状态、指令、浪并列，随机选用）。线性读出就给出 −ΔM ν̇₀（ΔM 任意，不要求正定），非线性读出还能给出更多。单列它不增加覆盖，只多一组参数。
- M_t 仍显式保留，但只为两件刚体的事：点力的通道比例（要用 ρ_r、ρ_q、r_g）和 (d) 的科氏项（要用 m、r_g、I）。M_t 的整体大小和任意力分不开（能观测到的只有 M_t⁻¹ 乘力），所以不另外随机化它的大小：低保真已有的数（m_surge、m_sway）直接用，只抽低保真没有的量（D9.7）。

**合起来**，每个子步加在五个速度通道上的修正是

    a = M_t⁻¹ [ Σ_i J(p_i)ᵀ F_i  +  Σ_k Q_k  −  (C_t(ν) ν − C₀(ν) ν) ]

- F_i：点力（和分布载荷），Q_k：一般的广义力，都是 D9.2 的任意因果函数；
- C_t(ν)ν − C₀(ν)ν：(d) 的刚体科氏差；
- J(p)ᵀ：(c) 的换算。

### D9.2 力：一切已经经历过的东西的任意因果函数

"因果"就是只看现在和过去，不看将来。力可以依赖：
- 船的状态：u、v、r、ż、θ̇，以及纵摇角 θ（θ 有绝对的意义：它决定重力相对船体的方向）；
- 指令和执行机构的实际状态（推力、喷口角）；
- 低保真船此刻的加速度 ν̇₀（D9.1 (e)）；
- 船体在 5 × 3 个站点遇到的浪，以"当地浸深"的形式：d_i = η_i − (z + s_p x_i θ)，即站点处的水面比船体该点高多少。它同时含浪和船的升沉、纵摇；
- 看不见的外部过程：风、流、载荷移动、污底、故障……统一写成几种随机过程（下面）；
- 时间只通过外部过程进入，本身不是输入。

**保留的唯一一条"形状"：平移不变。** 力不能依赖船在地面上的绝对位置（x、y，以及绝对升沉 z）。这不是水动力学，是一般力学：物理规律不认识地面坐标的原点在哪。所以升沉位移只以"相对水面"（浸深 d_i）的形式进入；绝对艏向只以"相对某个地面固定方向"的形式进入（风、流有方向，见下面的"地面固定方向"）。这一条需要用户确认（D9.7 未决问题 1）。

**随机族：能逼近任意"只看过去、记忆会衰减"的函数。** Boyd 和 Chua（1985）证明：任何不随时间改变、只看过去、对越久以前的输入越不敏感（记忆衰减）的输入–输出关系，在输入有界、变化不无限快时，都能用"有限个稳定的线性滤波器，后面接一个静态的非线性读出"逼近到任意精度【文献】。所以随机族就按这个结构抽。每个力 i：

1. **选输入**：输入分组——状态、指令与执行机构、ν̇₀、浸深（15 个）、外部过程、地面固定方向（下面）——每组 w.p. 0.5 选用，至少一组。每个输入先除以它在低保真库里的标准差（库只用低保真的记录，不用目标）。（库的组成和做法见 D9.8 第 1 条。）
2. **随机组合**：选中的输入乘一个随机矩阵，得到 n_p ~ 1 + Poisson(2)（最多 6）个组合信号。哪一个站点、哪一种浪的空间形态被读，完全由这个随机矩阵决定。
3. **滤波器组**：K ~ 1 + Poisson(3)（最多 8）个随机的稳定滤波器，每个读一个组合信号：
   - 直通（没有记忆），w.p. 0.25；
   - 一阶低通（反应慢一拍），时间常数 LogU[0.04, 30] s；
   - 一阶超前（读变化率），时间常数同上；
   - 二阶共振，ω ~ LogU[0.3, 20] rad/s，ζ ~ LogU[0.05, 2]；
   - w.p. 0.2 在上面任一种前面接一个纯延迟，LogU[0.04, 1] s。
   每个滤波器的输出在库上归一到单位标准差。
4. **读出**：一个随机的小神经网络：
   - 1–2 层，宽度 8–64（对数均匀取整），激活函数从 {tanh, softplus, relu, 绝对值} 里随机选；
   - 权重 N(0, σ_w² / 输入个数)，σ_w ~ LogU[0.3, 3]：小 = 几乎线性，大 = 强非线性、有陡的阈值；
   - 输入增益 g ~ LogU[0.3, 3]：以库标准差为单位，输入变多少读出才明显弯折（光滑程度）；
   - w.p. 0.7 另加一个线性直连项（让"几乎线性"的真船有足够的概率），w.p. 0.3 加一个两两乘积项（像 u·r 这样的速度乘积）；（实现改为每个力都有线性份额 β ~ U[0, 1]，σ_w 由输入增益 g 代替：D9.8 第 4 条。）
   - 输出在库上归一到单位标准差，再过软饱和 3 tanh(y/3)。
   随机的宽网络给出光滑的随机函数（Neal 1996：宽度趋于无穷时趋于高斯过程），σ_w 和 g 控制它弯多少、有多陡。
5. **力怎样加到船上**（每个力抽一种）：
   - **点力**（w.p. 0.5）：作用点在船体的外包盒里均匀抽：x ∈ [x_stern − 0.05 L, x_bow + 0.05 L]，y ∈ [−B/2, B/2]，z ∈ [−1.5 T, D − T]（D = 型深 1.1 m；水线以上的点也可以：风、上浪、甲板上的载荷）。方向：w.p. 0.6 固定在船体上的随机单位向量（三维各向同性）；w.p. 0.2 固定在地面上的随机水平方向（风、流一类，换到船体坐标后随艏向转）；w.p. 0.2 由读出直接给出（读出输出 3 个数）。
   - **分布载荷**（w.p. 0.2）：沿船长的载荷密度 f(x) =（读出的 2–3 个输出）×（1, x, x² 这样的形状），积分成力和力矩。
   - **一般广义力**（w.p. 0.3）：读出直接给出五个通道里一个随机子空间（1–5 维）上的广义力，允许纯力矩。
   单靠"一般广义力"一种就能覆盖一切；点力和分布载荷只是让"通道比例由作用点决定"有较大的概率。
6. **大小**：见 D9.3。

**地面固定方向**：每回合抽一个随机的地面方向 ψ_w（风、流的来向），把 cos(ψ − ψ_w)、sin(ψ − ψ_w) 作为一组输入——船感受到的是相对风、流的艏向，不是绝对艏向。

**看不见的外部过程**（每回合 m ~ Poisson(1.5) 个，最多 4 个；各自归一到长期单位标准差）：
- 有色噪声（w.p. 0.4）：白噪声过一个随机稳定滤波器，时间常数 LogU[0.3, 300] s（从阵风到慢慢变的风、流）；
- 慢漂移（w.p. 0.2）：带缓慢回归的随机游走，回归时间 LogU[60, 3000] s（污底、燃油消耗、慢慢移动的载荷）；
- 跳变（w.p. 0.2）：分段常数，跳变时刻按泊松过程，频率 LogU[1/300, 1/10] 次每秒，跳变大小重尾（t 分布，自由度 U[1.5, 5]）（载荷突然移动、突发故障）；
- 两态切换（w.p. 0.2）：在两个状态之间随机来回，每次停留 LogU[2, 120] s（时有时无的故障或状态）。

它们有两个去处：作为某些力的输入（于是能调制状态相关的力，比如"某个增益在漂"），以及直接构成一个力（例如只读外部过程、方向固定在地面的点力，就是一阵风）。跳变和两态切换是"不会衰减的记忆"，Boyd–Chua 的结构本身不含，所以单独给出；旧族的继电器和事件（D2、D5）保留在噪声部分里。

**本族没有的东西**：没有阻尼的符号，没有"不做功"，没有左右对称，没有任何水动力项的固定形状，也没有 D8 的"航速二次多项式"。航速的影响来自航速本身就是输入：读出对 u 可以任意弯。能量可以被注入，船可以自激振荡（例如滑行艇的海豚跳），只要不发散（D9.3）。

**噪声**：旧族的噪声部分（T9：有色噪声、白噪声、突发、事件的随机部分、继电器）照旧保留，按控制步保持，幅度 w_noise ~ LogU[0.1, 1]（D8.12）。

**时间分辨率**：力每个子步（0.04 s）算一次，用子步开始时的状态、该时刻的浪和外部过程。理由同 D8.4：状态反馈按整步保持，会平白多出假阻尼。所以只能在闭环里生成，不能在记录的轨迹上重新贴标签（D8.4、D8.12 第 5 条）。

### D9.3 仅有的非物理保护：大小上限、截断、拒绝发散

**1. 大小以低保真船自己的尺度为准（每个通道一个）。** 参考加速度 a_ref,c 全部由低保真船自己的参数算出【计算，参数读自 studies/_cache/lofi_scarab195.json】：
- 纵荡：满推力 / 纵荡质量 = 6366 N / 1969 kg = 3.2 m/s²；
- 横荡：喷口打满时的侧力 / 横荡质量 = k_jet_side · t_max · sin(rud_max) / m_sway = 4342 N / 1610 kg = 2.7 m/s²；
- 艏摇：同一侧力产生的艏摇角加速度 = 4342 × |k_nomoto_f| / τ_r = 1.76 rad/s²；
- 升沉：让浸深变化一个吃水所需的回复 = ω_h² T = 34.0 × 0.385 = 13.1 m/s²；
- 纵摇：让半个船长处的浸深变化一个吃水所需的回复 = ω_p² T / (L/2) = 13.8 × 0.142 = 1.96 rad/s²。

理由：前三个是这条船自己最大的、有意施加的力（满推力、满舵）；后两个是它自己改变一个吃水的静水力。一个模型解释不了的力如果比这还大，这条船就不再是"用这个简化模型来控制的这条船"了。它们和现有的 A_REF = (4, 4, 2, 12, 3) 同量级（比值 0.65–1.1），所以现有截断 CLIP × A_REF = (12, 12, 6, 36, 9) 约为 a_ref 的 2.7–4.6 倍，照旧可用。
- 每个力的幅度 κ_i ~ LogU[0.01, 1]：0.01 是几乎完美的模型，1 是和满推力满舵、或一个吃水的静水力相当；对数均匀，不偏向任何量级。（已修订：每个抽样一个总幅度 κ0，再分给各力，D9.8 第 3 条。）
- 换到各通道：
  - 一般广义力：在它的每个通道上乘 κ_i a_ref,c；
  - 点力和分布载荷：先算每单位力在五个通道产生的加速度 g = M_t⁻¹ Jᵀ d，再取幅度 A_i = κ_i · min_c (a_ref,c / |g_c|)。这样它在最"吃力"的那个通道正好是 κ_i a_ref,c，其他通道更小。各向同性的方向因此不会在横荡产生几十 m/s²（D8.12 预演里见过 20–40 m/s²），也不需要任何水动力的理由（D8.12 的"水平分量是竖直的 κ_h 倍"不再需要）。
- 每个力的读出有软饱和（最多 3 倍），所以单个力在任一通道不超过 3 κ_i a_ref,c。

**2. 截断（现有规则，D5，不变）。** 每个子步，"规则"部分（全部力 + 刚体科氏差）单独截在 CLIP × A_REF；施加的总修正是 clip(规则 + 噪声)，噪声承担剩下的部分（operators.clip_push）。检查而不是拒绝：全部数据里"任一通道、任一子步被截过"的控制步少于 0.1%；超过时只缩小 κ 的上限（尺度），不移中心。

**3. 拒绝发散的抽样。接受规则（准确地说）：**
1. 一个抽样（一条"真船"）连同它的回合，在闭环里跑完整个回合：数据生成用的控制器，这一回合的海况、任务和执行机构（D7）。
2. 回合中任何时刻出现下列任一种，判为发散：状态出现非有限值；航速超过最高航速的 1.5 倍（u > 1.5 u_max，u_max = 23.5 m/s）；|θ| > 45°；积分的细分次数撞上限（如果积分器有这个上限）。这正是路线图第 9 节"数值发散"的判据，两处用同一套。
3. 发散：丢弃这个抽样（不只是这一回合），用下一个种子重抽整个抽样。（已修订：改为每个抽样都过同一个固定的接受检验，回合里的发散只截断回合，D9.8 第 2、7 条。）
4. 没有其他拒绝：没有特征值 / 阻尼比检查（D8 的稳定性规则取消），没有能量检查，没有对称检查，也不因截断而拒绝（按截断拒绝会削掉重尾）。（实现照此：接受检验只判发散，D9.8 第 2 条。）
5. 记录拒绝比例，以及被拒抽样的特征（各力的幅度、有无外部过程、读出的 σ_w 等）。拒绝会改变被接受的分布（D8.12 第 17 条见过 λ 的中位数被移动），要让这个改变看得见。

为什么保留"拒绝发散"：真船在它的控制器下能开、不会发散，这是关于真船的事实，不是关于水动力的形式。但它依赖数据生成用的控制器，是一个假设（D9.7 假设 3、未决问题 4）。

### D9.4 旧族和 D8 族都是本族的特例；D8 族留作对照

- **旧的投影族**（operators.py、ops_m15.py）本来就是 Boyd–Chua 的结构：随机投影 → 随机稳定滤波器 → 随机网络读出，外加事件、继电器、慢漂移和噪声。它相当于本族里：只用"一般广义力"一种、五个通道各自独立、M_t = M₀、不加刚体科氏差、输入是那 28 个（含绝对的升沉、纵摇和浪高）、按控制步保持、在库上归一。区别有三：
  1. 本族按子步、在闭环里作为状态反馈作用；旧族按控制步保持。控制步保持是一种周期性的"采样后保持"，严格说不是不随时间变的关系，本族只能近似它（相当于一个约半步的延迟），不是严格包含；
  2. 本族有点力和分布载荷，让通道比例由作用点决定；
  3. 本族的大小以低保真自己的力为尺度，并随输入变大而变大。
- **D8 / D8.12 族**（operators_rb.py）的每一项都是本族里一个具体的点：阻尼 = 读速度的线性读出；回复 = 读浸深的线性读出；喷泵效能 = 读推力、喷口角的读出，作用在喷口附近的点上；T2c = 两两乘积项；T7 = 读浸深的滤波器，作用在随机点上；Teven = 读 v、r、喷口角的二次读出。D8 的那些限制（总阻尼不为负、耦合不做功、左右对称、工作航速下稳定）把这些点圈在一个很小的角落里。
- **计划**：
  - D9 族作为主线；D8.12 族（已实现、已检验）留作对照组，"结构化 vs 一般"。
  - 两组用同样的回合数、同样的海况 / 任务 / 执行机构抽样（种子配对），分别训练，只比较训练好的模型在目标上的表现（D1）。目标包括 Scarab（C、Cb 组）和 Wigley 排水型船（M15 的提醒，免得先验只适合 Scarab）。
  - 旧投影族保留为第三个对照（meta3 / meta5 的数据已有）。
- **实现（以后做，本节不写代码）**：新模块（例如 learn/meta/operators_gen.py），复用 operators_rb.py 的子步挂载（RBPlant、RBInjector）、站点浪高记录、快照和批量推演；等两个正在跑的数据流水线结束、M15 改完 data2 / operators 之后再接入。

### D9.5 60 秒历史在这个先验下能推断什么（照实说）

- 60 s = 250 个控制步 = 1500 个子步。网络在这段历史里看到状态、指令和误差（D6：不看浪高）。
- **能认出的**：
  - 这段历史里真正被激发过的那部分关系：当前航速附近、被浪激发的升沉和纵摇响应（激发的频带宽，信息多）；纵荡在当前航速上的偏差；横荡、艏摇只有在转过向、抖动过时才认出一部分。D8.6 的估计仍适用，只是现在没有"阻尼的符号""固定的比例"这些共享结构来补空；
  - 如果真船的力来自少数几个点，通道之间的固定比例；
  - 看不见的慢过程的当前值（偏置、漂移到哪了），以及窗口里发生过的跳变或切换——这就是 D4 的 h_k。
- **认不出的**：没去过的状态上的行为（别的航速、没做过的急转），比 60 s 更长的记忆，外部过程将来的跳变。这些只能靠先验：网络对这些情况应给出先验那么宽的误差分布。
- **覆盖和集中的取舍**：先验越一般，覆盖的真船越多，但同一段历史能排除的可能性越少——网络给出的范围更宽，MPC 在新的约束任务里会更保守（开得更慢）。D8 族在真船落在它里面时收得快，落在外面时会有偏差，而且自己不知道（D8.12 第 12 条：M7、M10、M11 在族外看到的是推演技能大于 1、覆盖率 0.3–0.8，而不是"范围自动变宽"）。
- **任务种类**：Raventós 等（2023）发现，预训练里任务的种类超过某个阈值之后，Transformer 在上下文里做的推断接近"对整个任务分布最优"的贝叶斯做法，并能推广到没见过的任务；种类不够时，它只在见过的有限几个任务之间挑。所需的种类随任务的维数增长【文献】。本族的维数原则上是无穷的，实际的"有效维数"由先验有多集中决定：σ_w 小、滤波器少、常有线性直连项，都让简单的真船占较大概率。计划的约 2 万个回合（M14）够不够，推导回答不了，只能看训练好的模型（在对照组之间比）。
- 这一节是预期，不是保证，都要在训练好的模型上检验。

### D9.6 新约束任务要记录的两个安全量（真值来自低保真，路线图第 9 节）

两者都在子步上算，用"修正注入之后"的状态（RBInjector 每个子步踢完之后的状态，就是这一时刻的真状态），每个控制步记一个数。

**1. 每步重心竖向加速度峰值 A_pk,k**

    a_z,j = (ż_j − ż_{j−1}) / 0.04,    j = 1 … 6（控制步 k 的六个子步，ż_j 为第 j 个子步末、注入之后的升沉速度）
    A_pk,k = max_j a_z,j   （向上为正）

- ż 是低保真的升沉速度，本来就是重心在地面竖直方向的速度；低保真的升沉振子里重力已被静浮力抵消（没有 g 项），所以 a_z 就是"去掉重力的地面竖向加速度"，不必再减 g【读代码：control/reduced.py 的升沉振子；sim/lofi.py 532 行 last_cg_acc = (ż_new − ż)/dt】。
- 它必须包含注入的修正。lofi 自己的 last_cg_acc 不含修正；RBInjector 已把升沉修正加进去（operators_rb.py 1318 行 plant.last_cg_acc += push[3]），所以挂钩之后读到的 plant.last_cg_acc 正好是 a_z,j。
- 同时记下 A_min,k = min_j a_z,j（向下的峰），以后如果改用绝对值不必重算。
- 它是 0.04 s 内的平均加速度，不是瞬时值：比 0.04 s 更短的冲击（砰击）峰会被削平，所以它是瞬时峰的下界。路线图的评估按 NSWCCD 的做法（10 Hz 四阶低通、峰至少相隔 0.1 s、低于 0.3 g 不算峰）；0.04 s 平均会压低 12.5 Hz 以上的成分，和 10 Hz 低通接近但不相同。每步峰值是规划用的通道；每小时事件数的评估在高保真上按路线图的做法另算。高保真也要按同一定义（0.04 s 网格、同样的差分）记录每步峰值，两边的误差才对得上。
- 截断的影响：升沉的规则部分截在 36 m/s²（3.7 g）。路线图的单峰限值 7 g 按 k = 2 折算到仿真是 3.5 g，低保真自己的响应再加上截断后的修正可以超过它，所以截断不会整体挡住约束关心的尾巴；但要统计"升沉被截的子步"在大峰里占多少（D9.7 未决问题 6）。

**2. 每步最小船头余高 H_min,k**

    h_j = F_b + (z_j + s_p x_b θ_j) − η_b(t_j),    j = 0 … 5（控制步 k 的六个子步开始的时刻）

（已修订：实现取子步末 j = 1 … 6、船头站点三处横向位置里最高的浪面，理由见 D9.8 第 10 条。下面几条是修订前的说明。）
    H_min,k = min_j h_j

- F_b = 船头站点的静止余高（甲板边缘到静止水线）= 型深 + 该站点龙骨的静止高度 = 1.1 − 0.215 = 0.885 m【读代码：sim/planing_vessel.py 150 行 freeboard = depth + keel_rest[−1]；lofi 从几何里带着 freeboard 和 draft_bow】。
- x_b = 3.16 m 是船头站点（x_st 的最后一个，等于 sec.x_bow）；s_p = −1（θ 为正时船首向下，所以船首高度 = z − x_b θ）。z 以静止水线为零点（lofi 的 _measure 在 z = θ = η = 0 时浸深为零）。
- η_b(t_j) 是船头站点、中线上的入射浪高，即 RBPlant 在每个子步开始时记录的 5 × 3 站点浪高里的 [4, 1]（与 control/reduced.py 算 rel = z_bow − eta_c[:, −1] 用的是同一个量），所以 h_j = F_b + rel。
- 横摇按零算（低保真没有横摇），所以左右舷甲板边缘一样高。
- 只有入射浪面：不含船自己的兴波、船首波和飞溅。高保真按同一定义记录。
- 路线图的限值：最坏 10% 的未来里视界内最小值的平均 ≥ F_b / 4 = 0.22 m。
- 取 j = 0 … 5 而不是 1 … 6：每个时刻只算一次（第 6 个子步末就是下一步的 j = 0），而且这些时刻的浪高 RBPlant 已经记着，不用多算一次水面。

- 这三列（A_pk、A_min、H_min，每回合每步一个 float32）接入时加进数据文件。接入之前按 M15 的方法（描述统计）检查先验的每步峰值和最小余高的分布能否覆盖目标——只作检查，只放宽尺度。

### D9.7 抽样范围和理由、假设、未决问题

**惯性（只抽低保真没有、而刚体部分需要的量）**
- 刚体质量 m_t = 1400 kg（低保真的 m_coriolis）× U[0.9, 1.25]：小艇的燃油、人员、装备通常让排水量变化一两成【假设】。（已修订：上限不能超过低保真的横荡有效质量，D9.8 第 9 条。）
- 重心相对参考点的偏移：x_g ~ U[−0.05, 0.05] L（大约一成的质量移动半个船长），y_g ~ U[−0.03, 0.03] B（偏载；本族不要求左右对称），z_g ~ U[−0.3, 0.3] T【假设】。
- 艏摇、升沉、纵摇的惯性沿用 D8.12 第 2、4 条：ρ_r ~ LogU[0.22, 0.37] L，A33/m ~ LogU[0.3, 2.5]，ρ_q ~ U[0.18, 0.27] L；惯性主轴在纵剖面内的倾角 U[−5°, 5°]，由此得到 I_xz【假设】。
- 纵荡、横荡的有效惯性用低保真自己的 m_surge、m_sway（D9.1 (e)：整体大小与任意力分不开）。

**力**
- 个数 N ~ 1 + Poisson(2)，最多 8；w.p. 0.05 一个都没有（真船 = 低保真 + 刚体科氏差 + 噪声）。
- 输入、滤波器、读出、加力方式、外部过程的范围见 D9.2；幅度见 D9.3。
- 所有时间常数和频率乘 rs = √(L / 10 m) 的相应次方（D7、D8.12 第 18 条的惯例），让范围随船的大小缩放。

**范围的理由**
- 时间常数下限 0.04 s = 一个子步；上限 30 s = 半个历史窗口。更长的记忆在 60 s 里认不出，由外部过程代表。
- 频率上限 20 rad/s：高于控制步的奈奎斯特频率 13.1 rad/s，约为升沉固有频率 5.8 rad/s 的三倍，低于子步能表示的 78 rad/s。
- 读出的 σ_w、g、宽度：对数均匀跨一个数量级，从几乎线性到强非线性都有概率。
- 外部过程的时间尺度：从阵风（秒级）到比一个回合还长（污底、燃油），跳变频率从"一回合几次"到"很少"。
- 大小：D9.3 的 a_ref，全部来自低保真自己的参数，没有用目标。
- 以上都是假设，只有闭环预览（描述统计）和训练好的模型能确认。

**写明的假设**
1. 平移不变：力不依赖地面上的绝对位置和绝对艏向，只通过浸深和地面固定方向读环境。
2. 附加质量里只有"与此刻加速度成正比"的部分放进 M_t，水的其余作用都在任意力里。
3. 数据生成用的控制器能代表"真船在正常操作下不发散"。
4. 低保真没有横摇：横摇的影响只能以看不见的过程，或读已有输入的力出现。
5. 四种外部过程（有色、漂移、跳变、切换）足以代表不衰减的记忆和不平稳。

**未决问题**
1. **平移不变要用户确认**：它是一般力学，不是水动力形式，但确实是一条限制（例如不允许"读绝对升沉 z"的力）。
2. **三种加力方式的比例**（点力 0.5、分布 0.2、一般 0.3）以及"简单的真船"占多少概率（σ_w、线性直连项、滤波器个数），没有依据，只能由预览和训练后的对照决定。
3. **要不要混合组**（本族和 D8 族按比例混合）：混合先验在真船接近结构化时收得快，不接近时仍有覆盖。先做两个纯组的对照，再定。
4. **拒绝依赖控制器**：换了控制器（比如 MPC 自己）被接受的分布就会变。拒绝比例要记录；比例很高（例如超过三成）说明幅度上限太大。
5. **回合数**：一般的先验可能需要比 2 万更多的回合（Raventós 等）；对照组的训练结果会给出迹象。
6. **截断和大峰**：升沉截在 36 m/s²，会不会削掉约束关心的峰；要统计被截子步在大峰里的比例。
7. **每步峰值的定义**：0.04 s 平均是瞬时峰的下界；路线图的修正系数 k 是按高保真对拖曳试验定的，和这里的平均方式叠在一起是否一致，要在高保真上用同一定义核对。
8. **计算量**：每个子步、每个力都要跑滤波器和一个小网络；D8.12 的刚体步已经让低保真一步慢了一倍。实现后实测，并受 16 GB 内存、同时最多一个进程的限制。
9. **和 D8.12 "总在的结构项"的关系**：Nomoto 结构误差（D8.12 第 4 条）是操纵模型的具体形式，本族不单列（任意力能给出它）；刚体科氏差属于刚体力学，本族保留。

### D9.8 实现时的修订（2026-09-30；代码 learn/meta/operators_gen.py，评审之后）

D9.1–D9.7 是写代码之前的推导。实现时有几处做法和那里的文字不同，评审又指出代码引用的"D9.8"没有写出来，而且有几处实现本身有问题。本节逐条写明现在的做法和理由；和 D9.1–D9.7 不同的地方以本节为准（那几处已加了指向本节的注）。代码里的"D9.8 item N"就是本节第 N 条。

**1. 输入和滤波器输出按"只用低保真的库"归一（落实 D9.2 第 1、3 步）**
- 输入库：低保真船单独（不加任何力）在闭环里跑 12 条、每条 60 s，用数据生成的自动驾驶（航速目标 16–30 kn 之间换、转向、推力和喷口抖动），训练用的海况抽样（有义波高 0.5–1.5 m），理想执行机构，种子固定（input_library，每个进程算一次，约 10 s）。每个子步记下力的 28 个物理输入：u、v、r、ż、θ̇、θ，实际推力和喷口角，低保真自己的加速度 ν̇₀（5 个），15 个站点的浸深。从第 10 s 起算每一列的均值和标准差。力看到的输入 =（物理量 − 库均值）/ 库标准差。外部过程本来就是长期单位标准差，地面方向是 cos、sin，这两组不再归一。
- 每个抽样把自己的每个力的延迟、滤波器在库上跑一遍（和回合里完全同一套离散化），每个线性滤波器的输出再按它在库上的均值、标准差标准化。读出网络的参考点云、偏置的位置、RMS 归一都用这些库上的滤波器输出，不再用假想的标准正态点云。
- 为什么改：第一版把每个输入除以一个物理尺度（例如 v 除以"让低保真自己的线性阻尼产生 a_ref 的速度"，约 2.1 m/s）。评审在标准海况、23 kn 的闭环里量到：滤波器输出的标准差中位数只有 0.13，而读出网络是按 1 设计的；力输出的"标准差 / 均方根"中位数 0.21，三分之一的力偏离线性化不到一成。也就是说，力在实际经过的范围里几乎是"常数 + 一点线性"，"任意力"没有体现出来。D9.2 本来要求按库标准差归一，是实现时偏离了。
- 力的大小不受影响：仍由 κ 和低保真自己的各通道加速度尺度 a_ref 决定（D9.3），和输入怎样归一无关。库只来自低保真，没有用目标数据（反泄漏）。

**2. 接受检验：固定的检验，只判发散（代替 D9.3 第 3 条）**
- 做法：每个抽样先过同一个检验。两条、各 60 s：一条静水（开始时船首从升沉、纵摇各抬高四分之一吃水，带一点横荡和艏摇，让自己激起的运动不靠浪也能显出来），一条标准海况（有义波高 1 m、谱峰周期 5.5 s）；23 kn 起步，数据生成的自动驾驶（种子固定），理想执行机构，不加噪声部分。**唯一的判据是发散**：非有限值、u > 1.5 u_max、|θ| > 45°（同 D9.3 第 2 条）。任一条发散，就用同一种子的下一个"尝试"随机流重抽整个抽样。
- 为什么用固定检验，不用回合本身：接受与否应当只取决于这条"真船"，不取决于这一回合碰巧抽到的海况、任务和执行机构。如果按回合判，大浪回合里留下来的船会偏稳，先验就随回合条件变了，这不是关于真船的事实。另外，回合里发现发散再重抽，要从头重跑整个回合。
- 第一版还有四条判据：静水里升沉或船首位移超过一个吃水（'calm heave' / 'calm pitch'），标准海况里船首相对水面的起伏比低保真自己的大半个吃水以上（'sea bow'），规则部分被截的子步超过 1%（'clip'）。评审指出前三条是"形状"规则，超出了"不发散"，限制了任意力在纵倾、升沉和浪响应上能做的事；D9.3 第 4 条本来就写着"没有其他拒绝，也不因截断而拒绝"。**四条全部取消。** 截断照 D9.3 第 2 条只作检查：全部数据里被截的比例超过 0.1% 时，只缩小 κ 的上限。检验里规则部分被截的子步比例记在 style 的 test_clip 里。
- 被拒的次数和原因记在每个抽样的 reject_log 和 style（n_reject、reject_reasons）；打包时 meta_step6 按原因统计被拒的比例。

**3. 幅度：每个抽样一个总幅度，再分给各个力（代替 D9.3 的"每个力各自 κ_i ~ LogU[0.01, 1]"）**
- κ0 ~ LogU[0.01, 1]，按 √w 分给 N 个力：κ_i = κ0 √w_i，w ~ Dirichlet(1, …, 1)，于是 Σ κ_i² = κ0²。
- 为什么：各力独立抽时，力越多，总误差越大（大约 √N 倍），误差的总大小就被"力有几个"左右了。现在总大小对数均匀、和 N 无关，N 只决定误差分成几份。每个力、每个通道的上限 3 κ_i a_ref,c 不变。

**4. 读出网络的形式（代替 D9.2 第 4 步的 σ_w 和"w.p. 0.7 加线性直连"）**
- 1–2 层（w.p. 0.3 两层），宽 8–64（对数均匀取整），激活从 {tanh, softplus, relu, 绝对值} 里随机选；输入增益 g ~ LogU[0.3, 3] 控制弯多少（一层时和 σ_w 等价，所以不另设 σ_w）；每个神经元的偏置取参考点云里一个随机样本点处的值，让弯折落在数据实际经过的地方。
- 线性直连的份额 β ~ U[0, 1]，每个力都有：输出 = √β × 线性部分 + √(1 − β) × 网络部分。连续的份额让"几乎线性"到"强非线性"之间都有概率，而不是"有 / 没有线性项"两种。
- w.p. 0.3 加一个两两乘积项（像 u·r 这样）。
- 网络最后一层不减均值，均方根在参考点云上归一到 1：常数偏差（稳定的纵倾偏差、阻力偏差）是最常见的模型误差，要保留；另外 w.p. 0.5 加一个显式的偏置份额 γ ~ U[0, 0.5]。
- 最后软饱和 3 tanh(y/3)。

**5. 滤波器的种类和参数**
- 直通 0.2、一阶低通 0.25、一阶超前 0.15、二阶共振 0.3、继电器 0.1（第 6 条）；w.p. 0.2 前面接纯延迟 LogU[0.04, 1] s。
- 超前滤波器写成 (τ2/τ1)(τ1 s + 1)/(τ2 s + 1)，τ2 ~ LogU[0.04, 3] s，τ1/τ2 ~ LogU[1, 10]：高频增益 1，低频增益 τ2/τ1 ≥ 0.1。这样它读"变化"，又不会把高频无限放大。共振滤波器按峰值归一（峰值增益 1）。
- 时间常数和频率按 rs = √(L / 10 m) 缩放，截在 [0.04 s, 30 s]、ω ≤ 20 rad/s（D9.7 的理由）。预览记录注入加速度在 6 Hz 以上的功率份额，看子步上的高频会不会过多。

**6. 继电器（两个阈值、有滞回）是一种滤波器**
- 读一个组合信号的低通（时间常数 LogU[0.04, 3] s）；高阈值 = 库均值 + hi × 库标准差，hi ~ U[−1.5, 1.5]；低阈值再低 U[0.1, 1.5] 个库标准差；输出 ±1。
- 为什么放进力里：带滞回的开关（某个状态时有时无）是"不会衰减的记忆"，Boyd–Chua 的结构不含（D9.2）。旧族的继电器在它的规则部分里，本族不用旧族的规则部分（只保留噪声部分），所以放到力里。
- 为什么阈值按库标准差：第一版的阈值用物理尺度的单位，评审量到 48 个继电器只有 7 个动过，其余只是常数 ±1，和显式偏置重复。

**7. 回合里的发散：截断，不重抽**
- 通过了检验的抽样，在别的海况、任务里仍可能发散。这时回合在发散那一步之前截断，记下 div（那一步；−1 表示没有），不重抽（理由同第 2 条）。
- 多步真值（分支推演）里某个方案发散时，从那一步起状态被冻住，之后的误差没有意义；含发散方案的整个时刻丢掉（保持每个时刻 P 个方案），丢掉的个数记在分支文件的 n_dropped（data6.build_branches6）。

**8. 拒绝 30 次之后不加任意力**
- 这个抽样只剩刚体科氏差和噪声（forces_off，记在 style）。为了让计算量有上限。次数要记录；如果不少见，说明幅度上限太大（D9.7 未决问题 4）。

**9. 惯性的范围（代替 D9.7 的 m_t = 1400 kg × U[0.9, 1.25]）**
- 刚体质量 m ~ U[0.9 m0, 0.97 min(m_surge, m_sway)]，m0 = 1400 kg（低保真的 m_coriolis）。为什么：低保真的横荡有效质量 m_sway = 1610 kg 是"刚体质量 + 横荡附加质量"，附加质量不能为负（附加质量矩阵的对角元非负），所以刚体质量必须比它小；1.25 × 1400 = 1750 kg 已经超过 1610 kg。上限留 3% 的附加质量。
- 艏摇惯性 I_r = max(低保真喷泵隐含的艏摇惯性 × LogU[0.8, 1.25], 1.05 I_zz)；升沉 m_w = m (1 + LogU[0.3, 2.5])；纵摇 I_q = I_yy (1 + LogU[0.3, 2.5])；回转半径 ρ_y、ρ_z ~ U[0.22, 0.28] L，ρ_x ~ U[0.35, 0.45] B；重心偏移和主轴倾角同 D9.7。

**10. 最小船头余高 H_min 的定义（代替 D9.6 的 j = 0 … 5、中线站点）**
- h_j 取控制步六个子步的**末尾**（j = 1 … 6）；船头的浪面取该站点**左舷、中线、右舷三处里最高的**。
- 为什么：(a) 第 k 步的指令决定的是这一步结束时的状态；j = 0 的状态由上一步的指令决定，算进第 k 步会让这一步的余高带上上一步的影响。(b) 甲板边缘在两舷，舷边的浪面可能比中线高，最小余高要取最坏的一处。
- A_pk、A_min 的定义不变。高保真要按同一定义记录。

**11. 数据管线里的约定（episode6 / data6 / meta_step6）**
- 'gen' 的 null 恒为 False：刚体科氏差总在，误差不是零。没有任意力的抽样另记 no_forces，只用于评估分组；多步真值不排除它们。
- reads_waves（"直接读浪"）只在某个实际用到的滤波器读的组合含浸深输入时才算。
- 数据的代码哈希（code6）包含 control/reduced.py、sim/wavefield.py、sim/config.py：回合、输入库和接受检验都在跑它们。打包记下每个原始分块的大小和修改时间，分块重跑过失败的种子后，打包会重做。

**检查（预览 studies/prior_preview_gen.py，日志 studies/_cache/meta3/prior_preview_gen.log）**
- 做法同 prior_preview_rb.py 的闭环部分：64 条新回合，每条 90 s，M10 执行机构，数据生成的自动驾驶；统计从第 40 步起。目标（C + Cb，128 条）只作对照，不用来改中心。
- 接受检验：64 个抽样里只有 1 次被拒（原因：发散），没有抽样落到"不加力"；回合里被发散截断的 3 条。被拒的那次总幅度 κ0 远大于被接受的中位数，符合"幅度太大才发散"。
- 力是不是"几乎常数、几乎线性"（闭环里量，和评审同一种量法）：滤波器输出的标准差中位数从 0.13 升到 0.83；力输出"标准差 / 均方根"的中位数从 0.21 升到 0.72，低于 0.1 的力从两成降到 8%；偏离线性化的程度中位数从 0.14 升到 0.35；继电器 74 个里有 48 个在闭环里动过（第一版 48 个里 7 个）。所以现在的力在实际经过的范围里有明显的变化和弯曲，常数和近似线性的力仍占一部分（由读出网络的偏置和线性份额决定，这是有意的）。
- 截断：规则部分被截的子步约万分之一；但"任一通道、任一子步被截过"的控制步是 1.8‰，超过 D9.3 第 2 条的 1‰。超出部分主要来自保留的旧族噪声部分（规则部分自己很少被截），和 D8.12 预演里见过的一样。按规则只能缩小尺度，还没有改，等全量数据上的比例再定。
- 高频：注入加速度在 6 Hz 以上的功率份额中位数 1–5%。
- 和目标的对照（只看、不拟合）：横荡、艏摇误差的大小，目标落在先验的 97–98% 分位（先验偏小）；纵摇的"阻尼相关"、前后相关和升沉–纵摇同步，目标落在先验的 2–5% 分位（先验在这几项上偏向另一侧）；最小船头余高、每步峰值的分布能覆盖目标（目标在 23–73% 分位），只有粗略定义下的每步峰值中位数落在 100%。按反泄漏规则，这些只能在以后放宽尺度时参考，中心不向目标移。

### D9.9 放宽力的大小的试验，以及先验检验说明了什么（2026-09-30，主会话）

- **第一次检验**（prior_preview_gen_k1.log，力的总大小 κ0 ~ LogU[0.01, 1]）：和旧投影族、D8 第一版相比，一般族把高保真目标推到了更靠边的位置。纵摇误差的前后相关、纵摇误差与纵摇角速度的相关、升沉–纵摇同步，目标分别在先验的 4.7%、1.6%、1.6% 分位；纵摇误差对航速的斜率在 98.4%；横荡、艏摇误差大小在 97–98%。
- **放宽大小的试验**（prior_preview_gen_k3.log，只把 κ0 的上限从 1 放到 3，其余不变，符合"检查后只放宽尺度"）：
  - 横荡、艏摇大小略有改善（96.8%）；纵摇的三项反而更靠边（1.6%、3.2%、3.2%），航速斜率到 100%；
  - 截断比例升了十倍（有通道被截的控制步 1.9e-2，要求 1e-3）；被拒的抽样的 κ0 中位 1.9，被接受的只有 0.09，大的力多半让闭环发散。
  - 所以放宽没有用，已改回 κ0 ~ LogU[0.01, 1]，按原设计生成数据。
- **说明什么**：没覆盖好的是"形状"，不是"大小"。目标的纵摇误差持续时间短（前后相关低）、和升沉误差反相、随航速变大；一般族的随机函数是平滑的滤波器组输出，符号对称，作用点前后各半，所以产生的纵摇误差偏持续、和升沉不同步、与航速无关。这正是上次对用户说的：一般族的实际内容几乎全在"随机函数怎么抽"。这些形状更像砰击（短促、集中在船首、随航速变大）、腾空和滑行升力随航速变化，属于 D10 清单里的 W1、W3、W4。
- 一般族照原设计生成数据并训练（用户同意"写完也做数据生成和后续训练"），作为"完全一般"的一组；先验检验只是描述，能力以训练好的模型在目标上的表现为准（D1）。

## D10 — 误差先验改为"力的清单"（2026-09-30；D10a 实现）

**设计原文**：learn/meta/PRIOR_D10_DRAFT.md（草稿；第 10 节"审查后的修改"和第 11 节"用户的决定"优先于前文）。本节不重复推导，只记：实现了什么、和草稿不同的地方及理由、检验和先验预览的结果、还没解决的问题。

**分阶段（编排决定）**：D10a 保持 meta5 的网络输入格式，这样三组先验——旧的投影族（meta5）、一般族 D9（meta6）、清单族 D10（meta7）——可以干净地对比。观测模型 O1–O5 已写好并单独检验，但 D10a 的数据里**关着**；第 11 节要实船记录的几路额外信号（横摇和横摇角速度、转速、相对风速和风向、对水航速、GNSS 质量）在每个子步由对应的隐藏状态算出，**只存成额外数组，不喂给网络**。以后的阶段再加带"有没有"掩码的网络输入。

### D10.1 实现了什么

一个抽样就是一条想象中的"真船" = 低保真船 + 每个 0.04 s 子步加在五个速度通道上的修正加速度，另加一条单独的**冲量通道**（速度跳变，不进子步截断）。低保真船的代码一行没动。修正分三层：

- **层 1**：D8.12 的刚体族（operators_rb），稳定性规则换成"只拒绝发散"（cat_base.OperatorRBL1），并按草稿第 1 节关掉和清单重复的项：T8 一律关（由层 3 代替）；W1 开时 T4 的航速多项式只留常数、有界 tanh 刚度关；W5(a) 开时 tanh 刚度也关；P3/P5 开时 T5 的 η_T、η_S 关；W8 开时 T7 的单向二次读浪关；W7 的横向、竖直部分分别关 T3 的对应二次阻尼；W11 开时 T3 的纵荡形状误差按航速在 16 kn 以下平滑关掉；T2c 里刚体的 m·v·r 一律去掉（D9.1 的刚体科氏差已含它）。
- **层 2**：清单各项，每项一个文件里的一个类：learn/meta/cat_w1.py（W1），cat_water_a.py（W2–W6），cat_water_b.py（W7–W12），cat_hidden_boat.py（H1–H3、B1–B3），cat_air_env.py（A1–A3、E1–E4），cat_prop.py（P1–P14）；观测模型在 cat_obs.py（O1–O5，D10a 关）。框架在 cat_base.py：共用量、数值规则的辅助函数、项的接口、子步的组装。开启概率按草稿各表和第 11 节；15% 的回合是"稀疏回合"，只开 1–3 项（W4 是结构项，总开）。
- **层 3**：D9 的通用残差（operators_gen），按声明的份额：0.75 "小残差"（力的个数 Poisson(1)、至多 3，每个 κ ~ LogU[0.005, 0.1]），0.20 "清单失效"（D9.8 的原范围），0.05 无残差。它的刚体惯量 M_t 是所有清单力换算成加速度的共同惯量，它的刚体科氏差只加一次。

**每个子步的次序**（cat_base.CatDraw.substep）：共用量（各站浸深和龙骨浸深、沿移动站点的入水速度、浪的水平轨道速度、进水口浸深、有效纵倾、低保真此刻自己的升沉纵摇加速度）→ 环境项（海流、波包、风）→ 隐藏自由度（横倾）→ 层 1（按相对水的速度读）→ 力的各项 → 推进链（草稿 3.5 的固定次序，最后过空化饱和）→ 接触门限 W4 → 层 3 和刚体科氏差。每项单独过软饱和（名义尺度的 3 倍）。

**新文件（本次装配）**
- learn/meta/operators_cat.py：`OperatorCat`（一个抽样，接口同 operators_gen：new_state / step / cat_accel / snapshot / slow_state / set_slow / style）；`simulate_cat`（批量闭环模拟）；`CatPlant` + `CatInjector`（Mission 路径，和批量模拟同一套算术）；验收 `acceptance_test` / `accept_cats`（和 operators_gen 同样的测试世界：一行受扰动的静水、一行标准海况，60 s，数据用的自动驾驶，理想执行机构，不加噪声；**唯一的判据是发散**：状态非有限、u > 1.5 u_max、|θ| > 45°）。被拒的抽样由同一种子的下一个尝试流代替，每次拒绝记下当时开着的项（按项的拒绝率）；拒满 30 次的抽样只留层 1 和 W4（forces_off）。
- learn/meta/episode7.py、data7.py、studies/meta_step7.py：meta7 管线，是 meta6 那几个文件的副本（原文件没动），阶段、数组格式和 meta6 一样，所以 studies/eval_preview_m15.py 不用改；多存 IMP（冲量通道在一步内的平均加速度）、SLAM（按低保真自己的规则数的船首砰击）、OBS（8 路额外观测信号）。数据块的哈希覆盖所有新文件、所有 cat_*.py 和只读导入的 meta6 文件；--procs 最多 8。
- 检验：studies/test_operators_cat.py、studies/test_m17.py；预览：studies/prior_preview_cat.py。

**冲量通道和截断**（草稿第 6 节第 2、3 条）：规则部分（层 1 + 清单 + 层 3 + 科氏差）照旧在每个子步截在 CLIP × A_REF = (12, 12, 6, 36, 9)，只统计不拒绝。砰击（W3）和漂浮物撞击（E3）的速度跳变在截断之后单独加上；它们本身已有物理上限（W3 的隐式动量更新不会让站点速度反号，E3 的速度变化不超过相对水速的一半）。另加一道保险：一个控制步内冲量之和不超过 CLIP × A_REF × 0.24 s（即控制步平均不超过截断），碰到时记数；再统计"推力平均 + 冲量平均"超过截断的控制步（只作检查）。冲量计入安全量 A_pk、A_min 用的竖直加速度。

**几何**：龙骨高度、进水口位置和砰击阈值取任务用的低保真船体（sim.lofi.plant_for 在任务船体上，每个进程算一次）。龙骨和框架默认值相同。进水口取植物进水口的纵向、横向位置（x = −1.38 m，y = 0），**高度取该处船底龙骨高度**（−0.36 m），不取植物给的 −0.55 m：植物给的是喷泵中心，在船底以下 0.19 m，用它时船底在进水口处已经露出水面，进水口还读作"浸着"。真船的进水口开在船底上，吸气从船底露出开始。这样航行时进水口浸深 h_run 是 0.22 m（审查后改，原来按喷泵中心是 0.41 m）。草稿写的是"取低保真的 prop_submergence"，那是低保真自己的喷泵中心定义，这里是有意的不同。

### D10.2 和草稿不同的地方（和理由）

草稿没给的数值都标成【假设】写在各项文件的说明里；这里只列改了形式、改了解释或补了关键选择的地方。

**框架和装配**
1. 有效纵倾 τ_e = τ0 + 相对纵摇 **−** atan(浪坡)。草稿写"+ 当地浪坡"；用减号时，贴着水面起伏的船 τ_e 保持 τ0，这才是"相对水面的纵倾"。软下限 0.5°、宽 0.5°。
2. 草稿没列的重复：T2c 里一直开着的刚体 m·v·r 和 D9.1 的刚体科氏差是同一项，去掉前者。
3. T3 纵荡形状误差和 W11 的分段：16 kn 处宽 1 kn 的平滑门限。
4. 所有清单力都用层 3 的 M_t 换算成加速度；层 1 内部仍用它自己的对角质量。B1 开着时共同惯量换成装载后的（B1 和 D9 惯量抽样合并，草稿未决问题 10），但**层 3 的力仍按标定装载的 M_t 换算**（差别至多约 25%，层 3 本身大小是任意的），B1 自己补上装载前后刚体科氏差之差。
5. 层 3 读的是没被清单改过的低保真量（没有海流、波包、横倾），它没有自己的验收，整个抽样一起验收。
6. 稀疏回合里 W4 照开；P3、P5 在可抽的池子里，没抽中就关。P10 取 0.075（草稿 0.05–0.1 的中点）。
7. 验收沿用 D9.8 的标准测试（和回合本身的条件无关），不是草稿第 6 节第 4 条字面上的"连同它的回合跑完再重抽"：D9.8 已定回合内发散只截断、不重抽，免得被接受的分布和回合条件搅在一起。
8. 冲量通道的控制步上限（见 D10.1）是草稿没写的保险；草稿只说"在控制步平均上检查"。上限按算子自己的控制步长 op.dt 算（批量模拟用环境的 dtc，两者相同）；Mission 路径的 CatInjector 原来写死"6 个子步"，换了子步数会悄悄和批量模拟不一致，审查后改成读 op.dt，给了子步数时检查 dt × 子步数 = op.dt。

**水（W）**
9. W1：ε 用 5 个站中线龙骨浸深的等权平均（离水的站算 0）；Cv 用低保真的船宽 2.03 m；航速外推在低速一侧有下限（辨识航速的 16/25），更低由 W11 管。
   - **Φ 的纵倾零点用随航速变的航行纵倾 τ_eq(U) = τ0·(U/U_id)^(−p_τ)**（草稿自己的 τ_eq，过同样的软下限），不用固定的 τ_run（审查后改）。原写法里，纵倾外推项把稳态纵倾移到 τ_eq(U)，Φ 在稳态航行里就不等于 1，纵荡项 X = −W(Φ − 1)tanτ_e 给出一份从不消失的单向阻力，而对应的"多出来的升力"从来没加上（竖直项扣掉了切平面）：16 kn、τ0 = 6°、p_τ = 1.8 时约 −4.5 m/s²，接近 3 倍名义的软饱和。按物理，任何航速的稳态航行里升力都等于重量，所以零点应在 τ_eq(U)。现在 16 kn、ε = 0、纵倾在 τ_eq 时 X 和 F_z 都是 0（新检验 no_steady_drag_off_design）。
   - 纵摇力矩 M = s_p W Φ ℓ_x ε/(1+|ε|) 按草稿整个保留（压力中心随浸深前移本身就是一个线性的升沉→纵摇耦合，扣掉切线后力矩的符号就不再确定）。它在 ε = 0 处的斜率 s_p W ℓ_x / h̄_run（最大约 17 (rad/s²)/m）会**叠加**在层 1 T4 对称抽样的 k53 耦合上，因为草稿改动 9 明确说 W1 开时 T4 保留 k35/k53。所以 W1 开着（0.7）时线性 k53 的分布 = T4 的对称部分 + W1 的单边部分。这是草稿层面的选择，照写；若要避免叠加，可以在 W1 开时关 T4 的耦合（待用户定）。
10. W2：模态力按质量沿模态形状分布、读出和质量正交；"阻尼比"相对 ω_ref，所以 ρ_n > 1 在门完全打开时就是净负阻尼；ζ_mode 只用低保真自己的阻尼（层 1 在清单之后才抽）。颤振型：草稿没给幅值上限和范围，用 1/(1+ŝ²) 限幅，ρ_f ~ LogU[0.3, 2.5]。
11. W3：左右半截面各用自己一侧的龙骨浸深；折角深度 d_c/b = tanβ/(2k_w)，底升角从艉到艏线性变大；站点从船首到船尾依次更新；W3 开时 W5(a) 用同一套折角深度。
12. W4：门限 χ_i 除以它在平水航行时的值并封顶 1（否则船首站平水航行时龙骨只浸 0.03 m、门限可高到 0.05 m，平水就会被判成离水，违反"平水定速时 W4 为零"）；空中误差按草稿写成 −g − 低保真 − 层 1 的竖直项，所以把 g 加进了低保真竖直加速度；纵摇没有 g 项。集中的升沉力、纵摇力矩按平水浸深 w_i(α+βx_i) 摊到各站。门限乘的层 1 竖直项是 T3、T4、T7 **加上 T2c 的航速耦合和 Teven 的二次载荷**（审查后补：它们也是水动力，原来漏了，离水时仍在作用）；T5（推力）和 T9（旧噪声）不算。
13. W5：草稿没给的值全是【假设】（艏艉折角深度比 U[1, 2]、防溅条深度、(b) 的刚度倍数等）；(b) 的时间常数不按船长缩放。
14. W6：不属于"平水为零"的项（7.1 没列它），平水里就是一份真实的额外阻力。
15. W7、W9、W10、W12 自己按龙骨浸深逐站或逐舷关掉，所以**不再**被 W4 的平均门限乘第二次；W7 的接触门限和 W4 同形但自己抽参数（项之间读不到彼此的抽样）。
16. W7 竖直部分 15 个站点在一个隐式步里一起更新（逐站更新会叠加成过冲）；横向相对速度 v + x r − c_orb·v_orb，c_orb ~ U[0, 1]（0 是草稿字面）。
17. 符号：草稿 W7 竖直力和 W9 轨道速度项写的负号按物理方向改了（站点入水时力向上；向前的轨道速度给向前的推力）。
18. W11、W12 按设计随航速变，平水不为零。W12（阶梯船底）按第 11 节补齐：左右两舷的通气状态加一个 W5(b) 式的航速开关；零点是"平水航行时已通气"；侧力 c_y 正负都可（艏摇方向取决于阶梯在重心前后）；只有阻力差确定地让船首偏向湿的一舷；范围全是【假设】。完整说明在 cat_water_b.py 的文件说明里。

**隐藏自由度和船（H、B）**
19. H1 的横倾目标按力矩写（ω0²·φ_eq），不是草稿的 ω²(U)(φ − φ_eq)：横倾来源是力矩，字面形式在刚度变负时会把偏置翻号。**两者在刚度为正时并不相同**（审查后更正原来的说法）：刚度是 ω0²(1 − κ(U/U_max)²)，所以静横倾是 φ_eq / (1 − κ(U/U_max)²)，航速越高、横向回复越弱，同样的力矩给出越大的横倾；只有 κ = 0 时两者一样。审查的抽样：30 kn 处这个放大倍数中位 1.21、90% 分位 1.45、最大 1.53；到 u_max、κ = 0.8 时到 5 倍。作为物理选择保留（来源是力矩）；κ 的范围要按"它也放大静横倾"来理解，是否收窄待定（D10.7）。它也是预览里横倾偏大的原因之一。"其中持续横倾 0.15、折角行走 0.2"按占全部回合的比例理解，开启 H1 时分别以 0.15/0.7、0.2/0.7 独立抽。两稳横倾的三次项在子步开始处线性化（牛顿形式），|k|·dt 大时一步拆成至多 8 小步。|φ| = 0.5 rad 有硬限位。
20. H2 的误差只取 −m₁s̈（草稿的 −m_s(a_t + s̈) 会把燃油的刚体质量再算一次，低保真已含它）；油箱壁限位总开。
21. B1 和 D9 的惯量抽样合并：层 3 的抽样当作标定装载的船，B1 按 (1+ε) 放大刚体部分、移动重心；附加质量部分乘 (1 + a_A ε)，a_A ~ U[0, 1]。
22. 耦合进 H1 的横倾力矩（H2、B2、A1）晚一个子步生效（次序上 H1 在它们之前）。

**空气和环境（A、E）**
23. A3：κ_θ 和 λ_L 两个比例一起定下 S·K_p 和力臂 x_0，所以 x_0 不再单独抽，落在 [0.05, 0.35] L 以外就重抽（接受的抽样里两者相关）；来流角用隐藏真实纵倾加相对纵摇，限 ±45°；顺风（相对顶风 ≤ 0）没有升力。
24. A2 的成形滤波器用 4 个一阶 OU 过程之和拟合 −5/3 尾巴（和二阶转移一样精确）；阵风载荷 = A1 在（平均 + 阵风）处减在平均处，是"线性化 A1"的精确版。
25. E1、E2 用低保真自己的质量（它们本来就是低保真方程在平移速度上的值），不用 M_t；"浪不随流走"没实现。E2 的岸壁默认关。
26. E3 的速度变化封顶在相对水速的一半（漂浮物不比船重），保证对水速度不反号。E3 触发 P10 **没有接上**（P10 在固定时刻开始、不读撞击；框架也不能回合中途开一项）。
27. E4：一次只一个波包；波高按破碎限 H ≤ λ/7 封顶；本船尾浪方向照字面取 30–60 s 前的航向。

**推进（P）**
28. P1 的损失函数平移到"平水航行时的进水口浸深处正好为零"，恢复阈值封顶在平水浸深的 0.9，保证平水里吸气锁一定能解开。
29. P2 的扭矩曲线形式和范围是【假设】；P2 的滞后叠加在 M10 的升速滞后之上。
30. P3 的怠速冲压阻力项改成 r_idle(u)·(1 − g(c)/g_0)₊^q，g_0 ~ U[0.05, 0.1]【假设：怠速到低转速的推力份额，怠速/最高转速 0.15–0.3】（审查后改）。草稿的 (1 − g)^q 在巡航油门处不消失（u_id 处 c ≈ 0.36，16 kn 处 0.15），而 φ(u) 已经代表"净推力随航速下降"，两者把同一份损失算了两次：u_id、低保真定速油门下真船推力/低保真推力的中位只有 0.78，16 kn 处 0.62，其中 12% 是负的（真船在刹车，低保真船还有 +948 N），每个非稀疏抽样都带一份单向的纵荡偏差，和草稿"φ(u_id) = 1 ± 0.1"的辨识点不符。现在凡 g(c) ≥ g_0 处 T(c, u_id) = t_max·φ_id·g(c)；60 个抽样在 u_id 定速油门下比值中位 1.02（5–95% 0.63–1.67，散布来自 g 的非线性和 φ_id）；4000 个抽样：u_id 处中位 0.99、负值 0，16 kn 处中位 1.10（φ 在低速 > 1）、负值 0.35%。c = 0 时的刹车（−κ_idle·s^s·t_max）不变。∂²T/∂c∂u ≤ 0 仍只对 φ·g 部分成立；冲压阻力项在 g < g_0 时让它变正（60 个抽样里 58 个），检验只查 φ·g 部分，另查 u_id 处 g ≥ g_0 时没有冲压阻力、比值中位在 [0.9, 1.1]。
31. P6 作用在喷口点（推进链没有纯力矩的槽位），草稿的作用点范围因此收成艉端；第 11 节的喷口俯仰偏置 w.p. 0.6，U[−6°, 6°]【假设】；草稿 7.1 说 P 各项在零点为零，但 P6(a) 按草稿自己的公式在零推力时是 −κ_z·T_id，零点检验跳过 (a)。
32. P7 用执行机构的油门代替喷流动量；P8 只有卡斗故障（指令从不倒车）；P9 的空化噪声放在 P4 里、饱和之前；P13 的空气脉冲经 P1 的升降不对称滞后；P14 用 24 个随机傅里叶特征。
33. P12 按第 11 节：控制器存在 w.p. 0.8，存在时各条规则按原表开；有规则改了指令后，那一条的"真船执行机构副本"（同一套 M10 参数，自己的延迟线）在回合剩下的时间里一直接管。回合长按 90 s 写死（data2.T_EP，没导入）。功率 P_eng ~ U[150, 250] kW 给出推力上限 η·P_eng/u_r，η ~ U[0.55, 0.70]【假设】，并保证在低保真辨识航速处不起作用。

**观测（O，D10a 关）**
34. 各 O 项在观测模型开时都开（草稿没给开启概率）；GNSS 掉线补法、升沉来源、航向来源各 0.5 二选一；天线杆臂 0.7 概率已补偿。草稿没给的范围都是【假设】（写在 cat_obs.py）。INS 升沉直接用 IMU 点真实高度过高通加噪声，不是对测得加速度二次积分。

### D10.3 检验（各跑一次）

- **逐项构造检验** studies/test_cat_items.py：254 项全过（审查修改后重跑；比原来多 W1 的 no_steady_drag_off_design，P3 的检验加了巡航油门处没有冲压阻力和 u_id 处推力比的中位；框架 F1–F5；每项的抽样可复现和开启率、随机和极端状态下有限且不超过 3 倍名义、平水定速为零、单项闭环；各项自己的检验；观测模型 O1–O5 的检验），约 5.5 分钟。
- **studies/test_operators_cat.py**：全过（审查修改后重跑 549 s，含上面的构造检验）。
  1. 同一种子：抽样、验收及重抽、闭环结果完全相同；换种子不同。
  2. 从第 20 步的快照接着跑 = 原回合自己的接续（带噪声），状态差 7e-15，安全量差 4e-14。
  3. Mission 路径（CatPlant + CatInjector，砰击 W3 强制开、在浪里）= 批量模拟：状态差 5e-13，安全量、冲量通道、额外观测信号差 ≤ 1.4e-13。
  4. 构造检验全过（见上）。
  5. 平水、定速、直航、辨识航速：15 个"平水为零"的项加 W4 在 2 s 闭环里的输出 ≤ 5e-16（推进项自己的检验允许 2e-15 的舍入），状态不漂（5e-18）。
  6. 截断和冲量通道：大的规则部分每个子步截在 CLIP × A_REF 并计数；冲量在截断之后原样加上；一个控制步的冲量之和封顶在 CLIP × A_REF × 0.24 s 并计数；"推力平均 + 冲量平均"超限的控制步被记下、不拒绝；安全量用的竖直加速度含冲量。
  7. 只拒绝发散：层 1 的稳定性规则关了；第一次尝试被人为加了强的纵摇负阻尼的抽样被拒（理由"发散"，开着的项记下），换成下一个尝试；一个大而有界、没有任何水动力形状的恒定推力（升沉 0.2 g、横荡为其截断的 0.3）被接受。
- **studies/test_m17.py**：全过（审查修改后重跑 138 s）。episode7 的回合（Mission 路径、M10 执行机构）= 同一抽样的批量闭环（状态差 4.5e-13；安全量、IMP、OBS、SLAM、PUSH 都 ≤ 3e-13）；从快照分支 = 回合自己的接续（4.5e-13）；工作进程不载入 torch；正在跑的作业没被碰过：现在算出的 code5/sim5、code6/sim6 等于 run5.log、run6.log 最新开头行记的代码（47bba40092e1/4d5cbe982750、fa624a2a7c17/83ac80ff8a5f），meta_step3.sim_code 等于 meta4d 的每个 .code 戳，sim5 等于 meta5 的每个 .code 戳，它们导入的 39 个文件都早于这两个作业的开始时刻 17:05:52（审查后改：原来只比较本检验自己运行前后的戳，本检验不改任何文件，这一条不可能失败；meta6 还没有 .code 戳，所谓 16 个戳是 meta4d 的 7 个加 meta5 的 9 个）；pack7 含 pack6 的全部数组、形状和类型一致，和 meta6_smoke、meta5_smoke 的 train.npz 键、类型、尾部形状一致；同一种子的海况、执行机构、任务和 meta6 相同。

### D10.4 先验预览（studies/prior_preview_cat.py → studies/_cache/meta3/prior_preview_cat.log）

和 prior_preview_gen 同样的条件（seed0 777：同样的海况、起始状态、自动驾驶、M10 执行机构），64 个 90 s 闭环回合。只是描述，不说上限或信息量；目标只作检查。**本节是审查修改后重跑的结果**（W1 的纵倾零点、P3 的冲压阻力、进水口高度、W4 门限的层 1 竖直项、预览的几项新统计；第一次预览的日志改名存为 studies/_cache/meta3/prior_preview_cat_v1.log，主要差别在下面注明）。

- **验收**：5.9% 的尝试被拒（每个抽样平均 0.06 次，最多 2 次），没有抽样被拒满。按项的拒绝率都不到三成（最高 E4 0.13、A3 0.11、W6 0.11、W1 0.10，其余 ≤ 0.09；次数少，只是参考）。回合内发散截断 4/64（一般族 3/64）。层 3 档：小 48、失效 14、无 2；稀疏回合 7；每个抽样开着的项中位 21 个。
- **截断超标，而且比第一次预览更多**：有通道被截的控制步 2.9e-2（第一次 1.3e-2），要求 1e-3（一般族 1.8e-3）；规则部分被截的子步 2.1e-2。升沉超过截断一半的子步 5872 个，其中 32% 被截（第一次 2036 个、5%）。冲量通道的控制步上限起作用的控制步 3.4e-4；"推力平均 + 冲量平均"超限的控制步 7.3e-4。**归因**（一个进程、同样的 8 个种子、20 s 后 40 步，依次把一项审查修改换回原样）：换回 W4 的层 1 竖直项、P3、进水口高度或 W1 的纵倾零点，都不改变格局；四种情形里控制步平均超过截断一半的部分都主要是 **W1**（纵荡和升沉；浪里深浸时 (1+ε)^p_h 很大，压差阻力 −W(Φ−1)tanτ_e 和升沉弯曲项一起变大）和 **W4**（升沉、纵摇），其次是 W3 的冲量。所以超标是 W1、W4 尺度本身的问题（D10.7 第 1 条），不是这次修改造成的；两次 64 回合预览之间的差别，8 个回合的样本分不清是修改还是抽样起伏。
- **抽到的总升沉刚度**（草稿第 6 节第 6 条，审查后补上这项检查）：每个抽样在任务航速上取最大的 ω_h² × W1 的 s^p_K × W5(a) 最大的局部刚度比 × W5(b) 的 k_m × T4 的 exp(cKz)（及保留时的有界 tanh 倍数）× B1 的 (1 − e_m ε)²：ω·dt 中位 0.256、90% 分位 0.314、最大 0.375，没有抽样超过 1（低保真自己 0.233）。
- **各层的方差份额**（按通道，各部分方差之和，不计协方差）：清单的水项 W 占大头（纵荡 0.74、升沉 0.85、纵摇 0.82，横荡 0.32、艏摇 0.48）；层 1 在横荡、艏摇约 0.2，升沉、纵摇约 0.1；隐藏横倾 H 在横荡 0.11；推进 P 在横荡 0.14、艏摇 0.10、纵荡 0.07（第一次纵荡 P 的份额更大，P3 原来那份单向推力亏损已去掉）；层 3 在所有通道 ≤ 0.08，没有超过三成的通道（草稿第 5 节理由 4 的检查通过）；旧噪声 0.04–0.12。
- **触发率**：
  - **海豚跳按机理数**（审查后改）：W2 在小振幅下净自激（van der Pol 型 ρ_n·门限 > 1，颤振型 ρ_f·门限² > 1，按当时对水航速）累计 ≥ 10 s 的回合占全部 0.049，W2 开着的 18 个回合里 0.17（van der Pol 型 14 个里 3 个，颤振型 4 个里 0 个）；W2 开着时净自激的时间占比平均 0.17。审查的抽样（4000 个 W2 抽样、随机任务航速）给出约 0.11 的抽样净自激、约占全部回合 0.03–0.05，和这里一致，**不比草稿估计的约一成多**。第一次预览说的"0.21，偏高"用的是"W2 纵摇推力 10 s 均方根 > 0.1 a_ref"，量的是 W2 在浪里推力的大小，不是自激：浪里模态速度本来就大，只降低阻尼、或门限只开一半时也会超过；这个数现在单列（全部 0.21、W2 开着 0.72），不作海豚跳率。不按它缩小 W2 的尺度。
  - 持续横倾、折角行走的判据仍太松：全部回合 0.56、0.53；抽到对应机理的回合里 1.00（8 个）、0.88（8 个）。最大横倾中位 6.1°，90% 分位 18°（H1 的力矩形式随航速放大静横倾，见 D10.2 第 19 条）。
  - **进水口吸气**（审查后分开统计）：P1 **自己的**抽吸 < 0.9 的时间占比中位 0.37、90% 分位 0.99，每小时进入约 450 次；P1 锁住（吸力丢失）的时间占比中位 0.07、90% 分位 0.49（40 个开 P1 的回合）。P13 的砰击空气因子 < 0.9 的时间占比中位 0.35、90% 分位 0.84（13 个开 P13 的回合）。第一次预览的"16%"把 P1 和 P13 乘在一起，而且进水口按喷泵中心算（多 0.19 m 的余量）；进水口放到船底以后，P1 自己的吸气反而更多。可能的主要原因是 P1 的升降不对称：抽吸下降的时间常数可短到 0.02 s、恢复可长到 5 s，浪的周期 2–3 s 时 p 一直贴着每个浪的最低点；这是草稿的形式，不是几何。是否合理要和实船记录比，暂不改（D10.7 第 2 条）。
  - 腾空（五个站龙骨都离水）平均 0.55% 的时间，90% 分位 1.8%，最大 4.9%。按低保真自己的船首规则数的砰击每小时中位约 2300 次（船首站平水时龙骨只浸 3 cm，几乎每个浪都算一次，不能和 ISO / Savitsky–Brown 直接比）。故障：P10 推力损失 6.6% 的回合，P12 改了指令 8.2%，P8、P11 在这 64 个回合里没出现。
- **和目标的对照**（目标中位在先验里的分位；括号里是目标回合低于 5% / 高于 95% 的比例）见下表。清单族盖住了纵摇误差的形状：纵摇前后相关（一般族 4.7% → 73%）、纵摇阻尼相关（1.6% → 87%）、横荡阻尼相关（6.2% → 36%）；纵荡误差大小居中（52%）。纵摇误差的大小偏大（16%，一般族 70%），横荡、艏摇前后相关偏高（都是 18%），升沉–纵摇同步仍在边上（13%）；升沉阻尼相关、纵摇对纵倾的斜率、横荡和艏摇大小在 85–95%。安全量：最小船首余高 H_min 的目标值在先验的 8.2% 分位（27% 的目标回合低于先验 5% 分位：先验的船首比目标"更干"）。

目标中位在各先验里的分位（括号里：目标回合低于 5% / 高于 95% 的比例；旧投影族、D8 第一版闭环、一般族 D9 取各自的预览日志，D10a 是审查修改后重跑的这次）：

| 统计量 | 旧投影族 | D8 第一版 | 一般族 D9 | D10a |
|---|---|---|---|---|
| rms_surge | 69.8% (0.00/0.00) | 93.8% (0.00/0.37) | 87.5% (0.00/0.12) | 52.5% (0.00/0.00) |
| ac1_surge | 69.9% (0.00/0.00) | 59.4% (0.00/0.00) | 73.4% (0.00/0.00) | 69.7% (0.00/0.00) |
| damp_surge | 21.1% (0.05/0.00) | 32.8% (0.02/0.00) | 32.8% (0.06/0.00) | 44.3% (0.00/0.00) |
| rms_sway | 85.7% (0.01/0.20) | 100.0% (0.02/0.74) | 98.4% (0.00/0.89) | 95.1% (0.03/0.54) |
| ac1_sway | 52.2% (0.00/0.00) | 42.2% (0.00/0.00) | 48.4% (0.00/0.00) | 18.0% (0.02/0.00) |
| damp_sway | 2.0% (0.93/0.00) | 35.9% (0.00/0.00) | 6.2% (0.09/0.00) | 36.1% (0.00/0.00) |
| rms_yaw | 80.6% (0.00/0.00) | 95.3% (0.00/0.54) | 96.9% (0.00/0.71) | 90.2% (0.02/0.36) |
| ac1_yaw | 32.4% (0.00/0.00) | 31.2% (0.00/0.00) | 21.9% (0.00/0.00) | 18.0% (0.00/0.00) |
| damp_yaw | 48.7% (0.00/0.00) | 67.2% (0.00/0.00) | 47.7% (0.00/0.00) | 69.7% (0.00/0.00) |
| rms_heave | 65.5% (0.00/0.00) | 82.8% (0.00/0.02) | 84.4% (0.00/0.15) | 77.0% (0.00/0.00) |
| ac1_heave | 53.8% (0.00/0.00) | 40.6% (0.00/0.00) | 35.9% (0.00/0.00) | 37.7% (0.00/0.00) |
| damp_heave | 80.1% (0.00/0.04) | 82.8% (0.00/0.15) | 79.7% (0.00/0.02) | 85.2% (0.00/0.37) |
| rms_pitch | 41.8% (0.00/0.00) | 45.3% (0.00/0.00) | 70.3% (0.00/0.00) | 16.4% (0.00/0.00) |
| ac1_pitch | 13.3% (0.21/0.00) | 15.6% (0.04/0.00) | 4.7% (0.59/0.00) | 73.0% (0.00/0.09) |
| damp_pitch | 4.3% (0.59/0.00) | 18.8% (0.03/0.00) | 1.6% (0.98/0.00) | 86.9% (0.00/0.01) |
| sync_sy | 61.5% (0.00/0.00) | 62.5% (0.00/0.00) | 57.8% (0.00/0.00) | 78.7% (0.00/0.00) |
| sync_hp | 10.8% (0.09/0.00) | 10.9% (0.20/0.00) | 1.6% (0.81/0.00) | 13.1% (0.01/0.00) |
| slope_th | 67.5% (0.00/0.01) | 75.0% (0.00/0.00) | 78.1% (0.00/0.12) | 91.8% (0.00/0.31) |
| slope_u | 94.9% (0.00/0.48) | 95.3% (0.00/0.67) | 98.4% (0.00/0.80) | 93.4% (0.00/0.23) |

按约定，这只是描述；三组先验谁更好，看训练好的模型在目标上的表现（D1），种子配对：meta5、meta6、meta7 同样的回合数、海况、任务和执行机构抽样。

### D10.5 冒烟运行（studies/_cache/meta7_smoke）

`python -m studies.meta_step7 --family cat --smoke --t-train 20 --procs 2`（copy、data、pack、wmid、e0、branches），再 `--phase train_a,train_w,train_p --steps 300`（GPU 空闲 11 GB），最后 `python studies/eval_preview_m15.py --cache-name meta7 --smoke`：全部跑通。训练 40 个 20 s 回合（2 个发散，1 个太短被打包丢掉），A、At 各 4 个、B 2 个 90 s 回合（B 有 1 个发散被丢掉）；分支 A、At 各 8 个时刻 × 8 个计划、B 2 个时刻，没有被丢的时刻；三个网络各 300 步；评估存出 eval_m15.pkl（44 个结果）。这么小的数据上评估的数没有意义，只说明管线通了。冒烟用的是审查修改前的代码；审查修改只改了各项的算术和 CatInjector 的冲量上限读法，没有改阶段、数组格式和接口，test_m17 已在新代码上重跑通过，所以冒烟没有重跑（全量运行用 meta7 缓存，数据块的哈希会和冒烟不同，不会混用）。

### D10.6 全量运行（没有启动）

meta5 的规模：训练 10000 个、A / B / At 各 300 个 90 s 回合。

    python -m studies.meta_step7 --family cat --phase copy,data,pack,wmid,e0,branches --size train=10000 --procs 2
    python -m studies.meta_step7 --family cat --phase train_a,train_w,train_p --steps 40000
    python studies/eval_preview_m15.py --cache-name meta7

**时间估计**（实测，机器同时在跑 meta6 的两份数据阶段，各 6 个工作进程）：冒烟里 90 s 的回合用 2 个进程每个 40 s（墙钟），即每个回合约 80 s 进程时间，其中验收测试约 25 s；20 s 的回合每个 18.8 s。一个回合的成本约是 meta6（一般族，同样条件下 3.7 s × 6 进程 ≈ 22 s）的 3–4 倍，主要花在各项每个子步的 Python 开销上（最大的单项是 W3 的冲量约 2.8 ms/子步，海面采样约 1.8 ms，共用量每子步重算三次）。按这个速度：
- 2 个进程：10900 × 40 s ≈ 121 小时（约 5 天）；
- 8 个进程、机器上没有别的作业：按每回合 55–80 s 进程时间算约 21–30 小时；
- 分支（3 × 60 个回合，每个要重建抽样、重跑验收）：2 个进程约 2 小时；wmid、e0 几秒；训练同 meta6。
每个工作进程约 250 MB 内存。

### D10.7 未决问题

1. **截断超标**（审查修改后 2.9e-2，第一次 1.3e-2，要求 1e-3）：8 个回合的归因（D10.4）显示控制步平均超过截断一半的主要是 W1（纵荡压差阻力和升沉弯曲项，浪里深浸时）和 W4（升沉、纵摇），其次 W3 的冲量；下一步按草稿第 6 节第 1 条缩小这几项的尺度（只缩尺度，不移中心；例如 W1 的 p_h 上限或给 Φ 的深浸部分加饱和），再在完整预览里按通道、按项记录被截的子步。
2. **进水口吸气**：P1 自己的抽吸 < 0.9 的时间中位 37%（进水口已放到船底，P13 已分开统计）。可能主要来自 P1 的升降不对称（下降 τ 至 0.02 s、恢复至 5 s，浪周期 2–3 s 时 p 贴着每个浪的最低点）。这是草稿的形式；是否太多要和实船（或文献的吸气频率）比，不按目标统计调。
3. 海豚跳按机理数是全部回合 0.05、W2 开着时 0.17，**不比草稿估计多**（第一次预览说的 0.21 量的是 W2 推力在浪里的大小，已更正）。纵摇误差大小偏大（目标在 16% 分位），横荡、艏摇误差前后相关偏高，升沉–纵摇同步仍在边上；这些等截断处理完再看。
4. 持续横倾、折角行走的判据太松（普通横倾也算进去），要换成按机理的判据；最大横倾 90% 分位 18°，|φ| = 0.5 rad 的限位被碰的比例还没统计。H1 的力矩形式让静横倾随航速放大 1/(1 − κ(U/U_max)²)（30 kn 中位 1.21 倍，u_max 处至多 5 倍）：κ 的范围是否收窄待用户定（D10.2 第 19 条）。
5. B1 开着时层 3 的力仍按标定装载的 M_t 换算。
6. W4 的出水系数 r_exit 只放在共用量上，W1 没读（W7 用自己的）。E3 撞击后触发 P10 没接上。H2、B2、A1 给 H1 的横倾力矩、W5(b) 对 H1 横向回复的影响都晚一个子步。
7. P12 改指令后的"真船执行机构副本"在预览里用真实的 M10 参数跑过（5 个回合），没有和独立计算对过。
8. 砰击计数用的是低保真自己的船首规则（船首站平水只浸 3 cm），不能和 ISO / Savitsky–Brown 比；要另定一个"砰击"的计数。站点只有 5 个（草稿未决问题 4）。
9. 观测模型开启时要定：生成数据的控制器看真值还是观测值，真船上 e0 怎样算（草稿未决问题 9、15）；额外观测信号进网络（带掩码）属于下一阶段。
10. 成本：每回合约是 meta6 的 3–4 倍；如果要更快，先优化 W3 的冲量和海面采样，或者给验收测试的两行换成一行。
11. 评估里"按读浪方式分组"的那一节对 meta7 是退化的：每个抽样都开着 W4（同一子步读龙骨浸深，也就读了浪），层 1 的 T7 也读浪，所以全部回合落在"同一步读浪"一组（style 的 reads_waves、wave_same_step 恒为真，这是真的，不是标错）。和 meta5、meta6 比较时只用合并的读浪数字；若要分组，要另定按"除 W4 以外是否开着读浪的项"的分法（以后的阶段）。
12. W1 的纵摇力矩保留了线性的升沉→纵摇斜率，W1 开时和层 1 T4 的 k53 耦合叠加（草稿改动 9 保留 T4 的耦合）；要不要在 W1 开时关 T4 的耦合，待用户定（D10.2 第 9 条）。
