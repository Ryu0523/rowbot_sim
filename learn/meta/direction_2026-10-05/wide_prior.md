# Line B: can a wide prior plus a strong learner make up for a wrong prior distribution? (2026-10-05)

Question (user, 2026-10-05, translated): can a very wide prior plus a strong learning model make up for a prior
distribution that is wrong? What does theory say about (i) how much target data a wide prior needs compared with a
narrow, correct one, (ii) when "wide prior + fine-tuning" beats "narrow prior", and (iii) what a well-designed wide
prior for this problem would look like?

Status labels used on every claim:

- **[T]** established theory: cited theorem, with its assumptions and whether they hold here.
- **[P]** project evidence: file, numbers and conditions.
- **[D]** my own derivation: "exact" means exact under the stated model; "heuristic" means order of magnitude only.
- **[O]** open: not measured or not proven.

Files read: `learn/meta/GZ_VS_FAMILY_2026-10-05.md`, `learn/meta/ROADMAP_2026-09-30.md` (sections 1, 2, 9, 10,
11.1–11.13), `learn/meta/THEORY_ONLINE_FLOW_2026-10-01.md` (T1–T4), `learn/meta/math_2026-10-05/fam_l1_l3.md`
(sections 2, 6, 7, 8), `learn/meta/SUMMARY_2026-09-30.md`, `learn/meta/RELATED_WORK_ONLINE_THEORY.md` (summary and
"do not cite" list), `learn/meta/LESSONS_FROM_RELATED_WORK.md` (check verdict), `studies/_cache/final_eval/meta7/report.txt`,
and the memory notes. I trained and simulated nothing. I ran one read-only arithmetic script (section 5.2).

---------------------------------------------------------------------------

## 0. Short answer

1. **Exact Bayesian inference [T].** A prior with full support makes up for a wrong prior *distribution* only
   asymptotically. Two things are required. The truth must lie in the prior's Kullback–Leibler (KL) support (Schwartz 1965).
   The data must satisfy the theorems' conditions (i.i.d. data, or Markov data with testing conditions; data from closed-loop
   adaptive control are not covered). The price of a wrong distribution is paid in **rate**. The rate is set by the prior
   mass near the truth and by the complexity of the support (Ghosal, Ghosh, van der Vaart 2000). For any finite data,
   including any single data sequence, the cost appears in the cumulative log-loss as roughly $\log 1/\Pi(B)$ for a
   neighbourhood $B$ of the truth (index of resolvability; project result T2b). If the truth is outside the support,
   no amount of data helps: the posterior goes to the KL-closest member (Kleijn & van der Vaart 2006), and its credible
   sets have the wrong width (Kleijn & van der Vaart 2012).
2. **In-context learning (zero-shot, one 256-step window).** In this mode a wide prior *cannot* make up for a wrong
   distribution in general. Every window starts from the prior again, so the cost $\log 1/\Pi(B)$ is paid again in
   every window [D]. In directions the window does not excite, accuracy depends only on the prior *centre*; breadth
   there only widens the stated uncertainty (section 2.5, exact in the linear-Gaussian surrogate model [D]). For the trained network
   the binding limit is not Bayesian statistics but **amortisation coverage**. Training controls the network only on
   average over the prior (Müller et al. 2022). Below a task-diversity threshold, the network behaves like Bayes
   with the *empirical set of training laws* as its prior (Raventós et al. 2023, empirical). In-context ability tracks
   the coverage of the pretraining mixture (Yadlowsky et al. 2023, empirical). A "stronger" in-context learner is a
   better approximation of the posterior predictive under the *wrong* prior, so it can make things worse. Project
   evidence for this exists (THEORY T4 / M18).
3. **Weight adaptation on target data.** This is where a wide prior plus a strong learner *does* make up for a wrong
   distribution, and the project measures it. With 2/4/6 two-minute missions of target data, the whole-network copy
   reaches an error ratio of 0.35/0.28/0.23 relative to the frozen net [P]. The reason [D, with T]: fine-tuning from
   the pretrained weights θ0 effectively uses a weight-space prior centred at θ0. Early stopping acts like ridge
   regularisation (Ali, Kolter, Tibshirani 2019, linear least squares). The function class of the network is far larger than the support of the training
   family, and the cost of the wrong family is paid once, over all target data, instead of once per window.
4. **How much data [T, D].** A narrow correct parametric prior with $d_0$ effective directions has cumulative excess
   log-loss $\approx \tfrac{d_0}{2}\log n + \log 1/\pi(\theta^*)$ (Clarke & Barron 1990). A flat wide prior pays
   $\tfrac{d_w}{2}\log n$ with $d_w \gg d_0$. A *hierarchical* wide prior pays the narrow cost plus a constant
   $\log 1/\varepsilon$ (Ghosal, Lember, van der Vaart 2008; Rousseau & Mengersen 2011). A nonparametric prior pays
   $n^{-2\beta/(2\beta+t)}$ in squared error, where $t$ is the *local* input dimension of a compositional truth.
   It is not the 58-number token times 256 steps (Schmidt-Hieber 2020; van der Vaart & van Zanten 2009).
5. **When wide + fine-tuning beats narrow [D].** Wide + fine-tuning wins when the narrow prior's truth is outside its
   support (an error floor $D_N$ per step that does not shrink) and the target data exceed $n^*\approx[\log 1/\Pi_w(B)+C_w(n^*)]/D_N$.
   With the project's numbers, the measured crossover is below one minute of target data. In-mission training on the
   first minute already gives an error ratio of 0.64 [P]. Among pretraining priors, *where* the mass is placed decides
   the result both zero-shot and after fine-tuning. The flat-wide "general" family loses to the structured catalogue
   family in both cases [P, ROADMAP 11.10].
6. **A well-designed wide prior [D, with T]** has these parts:
   - hard structure only where physics is certain (rigid-body map, positive-definite mass, causality, passive radiation
     memory);
   - for hull forces, a **local-section functional**: an integral over hull stations of shared unknown functions of
     local immersion and local relative velocity, plus transom boundary terms. Slender-body momentum flux has exactly
     this form, and it covers the stern lift, distributed Munk moment, planing impact, crossflow drag, hydrostatics and
     pressure drag that dominate the high-fidelity boat's error. One construction replaces many separate items;
   - **heavy-tailed scale priors instead of Gaussian tails or hard bounds**: a 72σ value costs about 9–15 nats instead
     of about 2600;
   - unknowns that enter **linearly**, wherever possible, so that an above-threshold in-context learner can extrapolate
     in magnitude the way ridge regression does;
   - a block-switched, small-weight generic residual;
   - breadth over drivers and wave directions;
   - validation on a different hull or simulator, because the high-fidelity boat is itself a strip-theory model and
     would otherwise leak into the prior.

---------------------------------------------------------------------------

## 1. Setting and notation (each symbol defined once)

| Symbol | Meaning |
|---|---|
| $g$ | the one-step error law of the high-fidelity simulated boat (called gz in the code), i.e. the conditional law of $e0_k$ given everything ([GZ_VS_FAMILY] §2–3) |
| $\Pi$ | the prior over error laws used for pretraining: the catalogue family (D10). It has layer 1 (fixed rigid-body forms), layer 2 (named items W/H/B/A/E/P), layer 3 (random "projection + filter bank + random network" forces), and ~9860 drawn training laws (`fam_l1_l3.md` §2, §6, §7) |
| $\Pi_H$ | the distribution of training *histories*: data autopilot, sea state redrawn per episode, wave direction uniform |
| window | the network's 256-step context (one control step = 0.24 s, so ≈ 61 s) |
| stage I | **in-context** (zero-shot) prediction by the frozen pretrained network, which updates no weights |
| stage II | **weight adaptation**: fine-tuning a copy of the network on $N$ target steps (one 2-min mission = 480 steps) |
| skill | Σ(true error − prediction)² / Σ(true error)² per channel; 1 = no better than predicting zero |
| error ratio | one-step mean squared error (MSE) of an adapted copy divided by that of the frozen net on the same held-out missions |
| $B$ | a set ("neighbourhood") of laws close to the target in a stated sense |
| $\Pi(B)$ | prior mass of $B$; $\log 1/\Pi(B)$ in nats is its "description length" |

"Wide" can mean two things, and they behave differently. Both are used below:

- **support-wide**: more laws get non-zero density (e.g. removing a hard bound);
- **mass-flat**: mass is spread more evenly, so every specific region gets less (e.g. the general family D9, with
  random filters and random networks everywhere).

"Strong learner" also means two things: a better in-context approximator (stage I: capacity, number of training laws)
and a more flexible weight learner (stage II: whole network vs output head, how far the weights may move).

---------------------------------------------------------------------------

## 2. Exact Bayesian inference with wide priors: what is established

### 2.1 Consistency needs the truth in the KL support [T]

- **Doob's theorem** (see Ghosal & van der Vaart 2017, *Fundamentals of Nonparametric Bayesian Inference*,
  Cambridge, ch. 6): the posterior is consistent for *prior-almost-every* truth. A single real boat is one fixed
  truth, and the theorem says nothing about it if the boat lies in a prior-null set. A boat in the far tails is
  practically such a set.
- **Schwartz (1965)**, *On Bayes procedures*, Z. Wahrscheinlichkeitstheorie 4:10–26. If the truth $p_0$ is in the
  KL support (every KL neighbourhood of $p_0$ has positive prior mass) and suitable tests exist, the posterior is
  consistent. Assumption: i.i.d. data.
- **Diaconis & Freedman (1986)**, *On the consistency of Bayes estimates*, Ann. Statist. 14:1–26. In infinite
  dimensions, full topological support is not enough. Inconsistency can occur even with a Dirichlet prior.
- **Do the assumptions hold here?** No, on two counts.
  - Data: one-step errors along a closed-loop trajectory are dependent and not identically distributed.
  - Support: the catalogue family puts *zero* density on the high-fidelity boat's law. [GZ_VS_FAMILY] §6.1 lists
    items that cannot be switched off, hard bounds that are crossed (yaw inertia ≥ 1.43× the high-fidelity value,
    the W10 position fixed at the bow, the actuator delay grid, the layer-3 amplitude cap $3\kappa a_{\rm ref}$),
    and forms that differ (pump priming law, 5 measured points vs 48 strips).
  - So for the *current* prior, consistency theorems cannot be invoked even in principle. They become relevant only
    for a redesigned prior whose KL support contains the target.

### 2.2 Rates: what breadth costs [T]

- **Ghosal, Ghosh, van der Vaart (2000)**, *Convergence rates of posterior distributions*, Ann. Statist.
  28(2):500–531. The posterior contracts at rate $\varepsilon_n$ if three conditions hold:
  1. prior mass: $\Pi(B_{KL}(p_0,\varepsilon_n)) \ge e^{-Cn\varepsilon_n^2}$;
  2. complexity: there is a sieve $\mathcal P_n$ with $\log N(\varepsilon_n,\mathcal P_n,d)\le n\varepsilon_n^2$;
  3. the remaining mass is small: $\Pi(\mathcal P\setminus\mathcal P_n)\le e^{-(C+4)n\varepsilon_n^2}$.

  Breadth therefore costs in exactly two places: less mass near the truth (condition 1) and more entropy (condition 2).
  Assumption: i.i.d.
- **Ghosal & van der Vaart (2007)**, *Convergence rates of posterior distributions for noniid observations*,
  Ann. Statist. 35(1):192–223. This extends the result to independent non-identical observations, Markov chains,
  stationary Gaussian time series and white noise, with a testing criterion. Closed-loop data from an adaptive
  controller are not covered (also noted in ROADMAP 11.11, problem 4).
- **Hierarchical (model-averaging) priors** make breadth cheap.
  - **Ghosal, Lember, van der Vaart (2008)**, *Nonparametric Bayesian model selection and averaging*, Electron. J.
    Statist. 2:63–89. A prior with weights over model indices, and a prior within each model, contracts at the rate of
    the model that best approximates the truth. It gives negligible weight to models larger than the optimal one
    (i.i.d. density estimation).
  - **van der Vaart & van Zanten (2009)**, *Adaptive Bayesian estimation using a Gaussian random field with inverse
    Gamma bandwidth*, Ann. Statist. 37(5B):2655–2675. A rescaled Gaussian process (GP) with a random bandwidth
    attains the minimax rate (up to log factors) for every smoothness level at once.
  - **Rousseau & Mengersen (2011)**, *Asymptotic behaviour of the posterior distribution in overfitted mixture
    models*, JRSS B 73(5):689–710. With a suitably small Dirichlet parameter, extra mixture components are emptied
    rather than split.
- **Fractional posteriors**: **Bhattacharya, Pati, Yang (2019)**, *Bayesian fractional posteriors*, Ann. Statist.
  47(1):39–66. A posterior with the likelihood raised to a power α < 1 contracts under the prior-mass condition alone,
  including under misspecification (contraction is then around the KL-closest point). This is the cleanest
  theorem-level statement that, for a tempered learner, *only prior mass near the best approximation matters* and the
  width of the support does not.

### 2.3 Finite-sample accounting valid for any data sequence [T, already in project as T2a–T2c]

For the exact Bayes predictive $q_\Pi$ and **any** sequence (no i.i.d. or mixing assumption):

$$\sum_{t\le n}-\log q_\Pi(e_t\mid h_t) \;\le\; \log\frac{1}{\Pi(B)} + \sup_{f\in B}\sum_{t\le n}-\log p_f(e_t\mid h_t),
\qquad
\sum_{t\le n}-\log q_\Pi(e_t\mid h_t)\;\ge\;\inf_{f}\sum_{t\le n}-\log p_f(e_t\mid h_t).$$

The upper bound is the index-of-resolvability idea (Barron & Cover 1991, *Minimum complexity density estimation*,
IEEE Trans. IT 37(4):1034–1054; Zhang 2006, *From ε-entropy to KL-entropy*, Ann. Statist. 34(5):2180–2210) in
its simplest pointwise form. It is proved in `THEORY_ONLINE_FLOW_2026-10-01.md` T2b. The lower bound is T2c.

- **These are the only bounds in this note whose assumptions hold for the project's dependent, closed-loop data.** They
  hold for the exact Bayes predictive, not for the network.
- **Two consequences [D, exact]:**
  - *Lower bound.* On cumulative log-loss, the exact Bayes mixture can never beat the best *single* law in its support
    on that sequence. So a zero-shot exact Bayes learner gets close to the high-fidelity boat over a window only if
    the support contains one law that is close over that window. The closure experiments ([GZ_VS_FAMILY] §8.2: 200
    and 1000 laws) measured fixed-weight mixtures, not the best law in the support per window. That quantity is
    still unmeasured [O].
  - *Upper bound.* The cost of a wrong distribution is additive, $\log 1/\Pi(B)$, and it is **paid once per data
    stream that the predictor conditions on**. For the in-context learner that stream is one window, so the cost is
    paid again in every window. For weight adaptation it is paid once over all target data (section 5.3).
- **Parametric asymptotics.** **Clarke & Barron (1990)**, *Information-theoretic asymptotics of Bayes methods*, IEEE
  Trans. IT 36(3):453–471: cumulative KL redundancy $=\tfrac d2\log\tfrac{n}{2\pi e}+\log\tfrac{1}{\pi(\theta^*)}+\tfrac12\log\det I(\theta^*)+o(1)$.
  Assumptions: i.i.d., correct smooth parametric model, prior density continuous and positive at $\theta^*$. They do
  not hold here. The formula is used only to see how costs scale: breadth costs $\tfrac12\log n$ per *active*
  dimension, while a misplaced centre costs $\log 1/\pi(\theta^*)$ once.

### 2.4 Misspecification: what a wide prior cannot do if the truth is still outside [T]

- **Kleijn & van der Vaart (2006)**, *Misspecification in infinite-dimensional Bayesian statistics*, Ann. Statist.
  34(2):837–877. The posterior concentrates near the KL-minimising points of the support. The rate is set by an
  entropy condition and a prior-mass condition around the KL projection. Assumption: i.i.d.
- **Kleijn & van der Vaart (2012)**, *The Bernstein–von Mises theorem under misspecification*, Electron. J. Statist.
  6:354–381. Credible sets are not valid confidence sets; the correct covariance has the "sandwich" form.
- **Grünwald & van Ommen (2017)**, *Inconsistency of Bayesian inference for misspecified linear models, and a proposal
  for repairing it*, Bayesian Analysis 12(4):1069–1103. Under misspecification the posterior can put mass on laws far
  worse than the best one ("hypercompression"). Tempering the likelihood with a data-chosen learning rate (SafeBayes)
  repairs this.
- **Yao, Vehtari, Simpson, Gelman (2018)**, *Using stacking to average Bayesian predictive distributions*, Bayesian
  Analysis 13(3). In the "M-open" case (truth not among the candidates), Bayesian model averaging tends to select one
  model rather than the best combination. Stacking targets the best combination.
- **Consequence for this project [D].** As long as the support misses the forms that dominate the high-fidelity
  boat's error, more breadth of the *same kind* gives at best the KL projection onto that support. By
  [GZ_VS_FAMILY] §5 these forms include stern slender-body lift with squared stern immersion, the distributed
  immersion-dependent Munk moment, the pump priming law, and the steady planing term with $-u\Theta$. The best mixture
  of members is not reached automatically either. The one experiment that measures distance to the support's
  "convex hull" ([GZ_VS_FAMILY] §8.2) gives sway 0.62 for 1000 laws. That is far from the frozen network's 0.38 and
  the fine-tuned copy's 0.07, but it is only an upper bound on the distance.

### 2.5 Finite-n lower bound and where breadth costs accuracy [T + D]

- **[T]** The van Trees inequality (Gill & Levit 1995, *Applications of the van Trees inequality: a Bayesian
  Cramér–Rao bound*, Bernoulli 1(1–2):59–79) bounds the Bayes risk of **any** estimator from below by the inverse of
  (data information + prior information). It is independent of the learner's architecture. It requires the usual
  regularity conditions (smooth prior density vanishing at the boundary, Fisher information).
- **[D, exact in the linear-Gaussian surrogate].** Model: $e_t=\phi_t^\top\beta+\xi_t$, $\xi_t\sim N(0,\sigma^2)$,
  prior $\beta\sim N(m,S)$, window information $J=\sum_t\phi_t\phi_t^\top/\sigma^2$, query feature $q$, true value
  $\beta^*$. Work in prior-whitened coordinates in which $S^{1/2}JS^{1/2}$ is diagonal with entries
  $\lambda_i = s_i^2 j_i$, and write $\delta_i=\beta_i^*-m_i$. (This presumes $S$ is diagonal in the same basis; in
  general, replace $q_i\delta_i$ by the product of the whitened query and the whitened deviation.)
  - The posterior-mean prediction error has a bias part $\sum_i q_i\delta_i/(1+\lambda_i)$ and a variance part
    $\sum_i q_i^2 s_i^2\lambda_i/(1+\lambda_i)^2$.
  - Per direction, the mean-squared error is $q_i^2[\delta_i^2+\lambda_i^2/j_i]/(1+\lambda_i)^2$.
  - It is minimised at $s_i^2=\delta_i^2$, i.e. when the prior width equals the actual deviation; the minimum is
    $q_i^2\delta_i^2/(1+j_i\delta_i^2)$.

  This gives three regimes:
  1. **Well-excited direction** ($j_i s_i^2\gg1$): the error is ≈ $q_i^2/j_i$ whatever the centre and width.
     **Breadth is free** for accuracy. It costs only the Occam term $\tfrac12\log(1+\lambda_i)$ in log-loss
     (MacKay 1992's effective number of parameters, already used in ROADMAP 11.11/11.12).
  2. **Weakly excited direction** ($j_i s_i^2\sim1$): a width that is too large inflates variance toward $q_i^2/j_i$;
     a width that is too small leaves bias. The best width equals the true deviation, which is unknown for each boat.
     A hierarchical (learned-scale) prior or a heavy-tailed prior handles this better than any fixed Gaussian width
     (section 6.4).
  3. **Unexcited but queried direction** ($j_i\approx0$, $q_i\ne0$): the error is $q_i^2\delta_i^2$ **whatever the
     width**. **Only the centre matters.** Breadth only makes the stated uncertainty honest: the predicted variance
     $q_i^2 s_i^2$ covers the error only if $s_i\gtrsim|\delta_i|$.

  The **allocation principle** follows:
  - be wide in directions that one window identifies;
  - be correctly centred (from physics) in directions it does not;
  - leave everything else to cross-window learning (weights or memory).
- **[P]** ROADMAP 11.12, a surrogate computation along the high-fidelity trajectories: each law has about 20–25
  strength directions, and one window identifies about 15 of them (median 11–17 under the assumptions tried). So
  about 5–10 strength directions per law fall in regimes 2–3 in each window. The directions the high-fidelity boat
  *needs* but the family lacks are not directions of the family at all. For them $q_i\ne0$ and no feature exists:
  regime 3 with $m_i=0$.

### 2.6 A generic residual competes with the physics [T + D]

- **[T]** **Kennedy & O'Hagan (2001)**, *Bayesian calibration of computer models*, JRSS B 63(3):425–464: model
  reality as simulator + discrepancy function (a GP).
- **[T]** **Brynjarsdóttir & O'Hagan (2014)**, *Learning about physical parameters: the importance of model
  discrepancy*, Inverse Problems 30(11):114007. The discrepancy is confounded with the physical parameters. A broad GP
  discrepancy gives correct prediction *within the range of the data*, but "only with realistic priors on the model
  discrepancy do we uncover the true parameter values" (abstract), backed by theoretical arguments.
- **[T]** **Plumlee (2017)**, *Bayesian calibration of inexact computer models*, JASA 112(519):1274–1285. Making the
  discrepancy prior orthogonal to the gradient of the computer model reduces the confounding.
- **[D]** This is the published counterpart of the project's "explanation pre-emption" derivation (ROADMAP 11.11,
  problem 1). Suppose a flat generic component reads the same states as the physical components, and the two are
  correlated inside the window but not at the query. Then the deviation is split between them in proportion to their
  prior variances, and extrapolation through the physical component shrinks. The flat-wide general family losing to
  the structured catalogue family (section 5.1) agrees with this; it is not a proof.
- **Design consequence [D]:** a generic residual should be
  - switched on and off **as one block with a small prior weight**;
  - **restricted** to inputs, time scales or orthogonal complements that the physics blocks do not already explain;
  - not left always on and correlated with everything.

---------------------------------------------------------------------------

## 3. Amortised in-context learners: what "strong learner" can and cannot add

### 3.1 What pretraining guarantees [T]

- **Müller, Hollmann, Pineda Arango, Grabocka, Hutter (2022)**, *Transformers can do Bayesian inference*, ICLR 2022.
  Training a prior-data fitted network (PFN) on tasks drawn from a prior approximates the posterior predictive
  distribution (PPD). The training cross-entropy equals the prior-averaged KL from the PPD to the network, plus a
  constant (the project's T1 restates this for rectified flow, which needs an extra Lipschitz condition on the flow;
  ROADMAP 11.11).
- **What this controls:** the error averaged over histories drawn from the prior predictive, which includes $\Pi_H$.
  In a region of prior mass $m$, the only implied bound is proportional to $1/m$ (Markov's inequality; ROADMAP 11.11,
  corrected in 11.12: a sufficient condition for a worst-case guarantee, not a prediction of failure).
- **Assumptions vs here:** the high-fidelity boat's histories are generated by a different driver and sea protocol
  than $\Pi_H$ ([GZ_VS_FAMILY] §7–8.1). So the guarantee does not apply to them even where the law itself is
  covered.

### 3.2 PFN behaviour is frequentist, not automatically Bayesian [T]

**Nagler (2023)**, *Statistical foundations of prior-data fitted networks*, ICML 2023 (PMLR 202):

- PFNs are better read as "pre-tuned, but untrained predictors".
- Their variance vanishes if the sensitivity to individual training samples does.
- Their bias vanishes "only if it is appropriately localized around the test feature".
- The transformer architecture of the PFN implementations he studied ensures only the former.

Assumptions: i.i.d. tabular data, with the context growing. **Here** the context is fixed at 256 steps, so the
statement applies only by analogy: a full-support prior does not make the in-context predictor consistent; the
architecture must also localise.

### 3.3 Task diversity, coverage and out-of-distribution behaviour [T, mostly empirical]

- **Raventós, Paul, Chen, Ganguli (2023)**, *Pretraining task diversity and the emergence of non-Bayesian in-context
  learning for regression*, NeurIPS 2023 (empirical).
  - Below a task-diversity threshold, the transformer behaves like Bayes with the **finite set of pretraining tasks**
    as its prior and fails on unseen tasks.
  - Above it, the transformer behaves like ridge regression (the Bayes estimator under the full Gaussian task prior)
    and solves new tasks. This "hinges on it deviating from the Bayes optimal estimator with the pretraining
    distribution as the prior" (abstract).
  - Read from the PDF: thresholds of roughly $2^{14}$, $2^{15}$, $2^{16}$ tasks at dimension $D=8,16,32$, i.e. growing
    about linearly in $D$. Setting: linear regression, base model GPT-2 type with 8 layers and 128-dimensional
    embeddings.
  - The project's ~9860 training laws are below even their $D=8$ threshold, but our laws are not linear-regression
    tasks, so the comparison is only an analogy.
- **Kirsch, Harrison, Sohl-Dickstein, Metz (2022)**, *General-purpose in-context learning by meta-learning
  transformers*, arXiv 2212.04458 (NeurIPS 2022 Workshop on Meta-Learning). The model moves from memorising tasks to
  generalising as the number of tasks grows (empirical). (The fourth author is Metz, not Schmidhuber.)
- **Wu, Zou, Chen, Braverman, Gu, Bartlett (2024)**, *How many pretraining tasks are needed for in-context learning
  of linear regression?*, ICLR 2024. Theorem: for single-layer *linear* attention on linear regression with a
  Gaussian prior, a dimension-independent number of tasks suffices. Far from our architecture and task class.
- **Garg, Tsipras, Liang, Valiant (2022)**, NeurIPS 2022 (empirical). In-context learning of linear functions matches
  least squares and survives some distribution shifts in the prompt inputs.
- **Panwar, Ahuja, Goyal (2024)**, *In-context learning through the Bayesian prism*, ICLR 2024 (empirical). Where
  Bayesian inference is tractable, high-capacity transformers mimic the Bayes predictor. They can also generalise to
  function classes not seen in pretraining, which "involves deviation from the Bayesian predictor".
- **Yadlowsky, Doshi, Tripuraneni (2023)**, *Pretraining data mixtures enable narrow model selection capabilities in
  transformer models*, arXiv 2311.00871 (empirical). Model selection among pretrained function families is near
  optimal, but there are "various failure modes" out of domain. In-context ability is "more closely tied to the
  coverage of their pretraining data mixtures than inductive biases".
- **Bai, Chen, Wang, Xiong, Mei (2023)**, *Transformers as statisticians*, NeurIPS 2023. Existence results:
  transformers can implement standard estimators and select among them in context. They do not show that gradient
  training finds these solutions.

### 3.4 What this means for "wide prior + strong in-context learner" [D]

- If the network *is* a good approximation of the PPD under $\Pi$, then making it stronger (capacity, training
  steps) moves it closer to Bayes under the **wrong** prior.
  - The project has the predicted symptom: THEORY T4 / M18, where longer training made target pitch *worse*
    (skill 0.85 → 1.05 on the older target C).
  - The algebra of T4 shows how a well-learned wrong relation produces skill > 1.
- The only way an in-context learner exceeds its prior is a *beneficial deviation* from Bayes under the training
  prior: learning a general estimator (ridge-like regression on the right features) rather than a lookup over the
  training laws (Raventós; Panwar). This is not guaranteed. It is most plausible where the unknowns enter **linearly**
  as coefficients on features the network can compute: for a Gaussian prior on coefficients, the Bayes estimator *is*
  ridge regression, which is defined and sensible for any true coefficient value.
- **[P]** The project's capacity result fits this reading but does not prove it. The width-384 network is 7–13 %
  better zero-shot on the high-fidelity boat in two seeds each, with no in-family gain and narrower intervals
  (ROADMAP 11.4). The 11.12 "corner" boats (six physical items all strong, a cell with no training law) show no
  learnable residual, so the network does extrapolate in *strength* within the family's forms.

---------------------------------------------------------------------------

## 4. Meta-learning and fine-tuning theory

- **Baxter (2000)**, *A model of inductive bias learning*, JAIR 12:149–198. The bias (hypothesis space or prior) can
  be learned from an *environment* of related tasks. Its own framing is exactly the breadth trade-off: the space must
  be "large enough to contain a solution yet small enough to ensure reliable generalization". The guarantees hold for
  tasks drawn from the **same environment distribution**. **Here:** the high-fidelity boat is not a draw from the
  catalogue family, so Baxter's bound gives nothing for it. The same holds for PAC-Bayes meta-learning bounds:
  **Amit & Meir (2018)**, *Meta-learning by adjusting priors based on extended PAC-Bayes theory*, ICML 2018;
  **Rothfuss, Fortuin, Josifoski, Krause (2021)**, *PACOH: Bayes-optimal meta-learning with PAC-guarantees*,
  ICML 2021.
- **Maurer, Pontil, Romera-Paredes (2016)**, *The benefit of multitask representation learning*, JMLR 17(81):1–32,
  and **Tripuraneni, Jordan, Jin (2020)**, *On the theory of transfer learning: the importance of task diversity*,
  NeurIPS 2020. For tasks $f_j\circ h$ with a shared representation $h$:
  - learning $h$ from $t$ tasks costs $\approx C(\mathcal H)+tC(\mathcal F)$;
  - a new task then costs only $C(\mathcal F)$ ("the sample complexity needed to learn a new task scales only with"
    the head complexity);
  - this requires *task diversity* and *realisability*: the target uses the same $h$.

  **Here:** realisability for the high-fidelity boat relative to the pretrained representation is unknown. The
  measured gap between output-head-only and whole-network fine-tuning ([P] 6 missions: 0.32 vs 0.23) is the size of
  the representation term that freezing cannot remove.
- **Chua, Lei, Lee (2021)**, *How fine-tuning allows for effective meta-learning*, NeurIPS 2021. If tasks share an
  *approximately* common representation, fine-tuning by gradient descent provably uses the shared structure. There
  are settings where any method built on a representation trained without regard to fine-tuning does no better than
  a learner with no source tasks (worst case). **Relevance:** if a wide prior is wrong, the learner must be able to
  move the representation, not only the head.
- **Kumar, Raghunathan, Jones, Ma, Liang (2022)**, *Fine-tuning can distort pretrained features and underperform
  out-of-distribution*, ICLR 2022 (empirical over 10 shifts): full fine-tuning was on average +2 % in-distribution but
  −7 % out-of-distribution versus linear probing. **[P] contrast:** in the project's wave-direction transfer
  ([GZ_VS_FAMILY] §7, D7), the *head-only* copy lost more of its gain than the whole-network copy. Their effect is not
  universal; I give no explanation here [O].
- **Ali, Kolter, Tibshirani (2019)**, *A continuous-time view of early stopping for least squares regression*, AISTATS
  2019. Gradient flow stopped early is closely tied to ridge regression. **[D, heuristic for networks]** Fine-tuning
  from $\theta_0$ with early stopping therefore acts like a Gaussian-type prior centred at $\theta_0$ in weight space.
  That prior has full support over the network class (or the LoRA/head subspace). So the project's working recipe is
  already two-level:
  - stage I: narrow function prior $\Pi$;
  - stage II: wide weight-space prior around $\theta_0$.
- **Gozeten et al. (2025)**, *Test-time training provably improves transformers as in-context learners*, ICML 2025.
  For linear transformers with a single gradient step, the benefit of test-time training depends on the alignment
  between the pretraining distribution and the target task. It "can alleviate distribution shift".
- **Ben-David, Blitzer, Crammer, Kulesza, Pereira, Vaughan (2010)**, *A theory of learning from different domains*,
  Machine Learning 79:151–175. Target risk ≤ source risk + a divergence between input distributions + the joint best
  error. **Here** it applies to the *input* side: drivers, sea protocol and wave direction. **[P]** D7: with the wave
  direction fixed in the training missions, the whole-network gain shrinks from an error ratio of about 0.2–0.35 to
  about 0.47–0.61 on missions with other wave directions.

---------------------------------------------------------------------------

## 5. The project's numbers in this frame

### 5.1 Measured trade-off between breadth and zero-shot accuracy [P]

| Prior used for pretraining (ROADMAP 11.10, high-fidelity boat; sway / yaw / pitch) | zero-shot skill | after 2 target missions (skill) | in-mission, first minute (error ratio vs own frozen net) |
|---|---|---|---|
| old projection family (narrow, misplaced) | 0.76 / 0.80 / 0.95 | 0.24 / 0.37 / 0.42 | 0.59 |
| general family D9 (flat-wide: random filters + random networks) | 0.58 / 0.69 / 0.83 | 0.17 / 0.34 / 0.36 | 0.58 |
| catalogue family D10 (structured; generic layer 3 switched by tier, 20 % "fail" tier) | 0.38 / 0.49 / 0.35 | 0.09 / 0.19 / 0.17 | 0.64 |

Further measurements:

- **Capacity:** width 384 gives an error ratio of 0.87–0.89 on the high-fidelity boat (8/8 missions better), 0.99–1.07
  in-family, with narrower intervals (ROADMAP 11.4).
- **Prior mass near the target** ([GZ_VS_FAMILY] §6.3, §8.2):
  - median single law: skill 1.48 / 1.86 / 1.90 / 0.26 / 4.50 (surge, sway, yaw, heave, pitch);
  - none of 1000 laws beats "no family push" on every channel, i.e. a share ≲ 3/1000;
  - the required heave–pitch coupling lies at 7.7σ–72σ of the Gaussian prior on that term;
  - the required pitch damping lies at 3.9σ–21σ;
  - several needed forms are absent altogether.
- **Window identification** (ROADMAP 11.12): about 15 strength directions identified per window. The corner boats
  show no learnable residual; 44 in-family boats show no learnable residual of the size seen on the target.
- **Fine-tuning curve** on the high-fidelity boat (ROADMAP 11.5, 11.6, 11.9):

  | copy | data | error ratio |
  |---|---|---|
  | whole network | 2 / 4 / 6 missions | 0.35 / 0.28 / 0.23 |
  | output head only | 2 / 4 / 6 missions | 0.55 / 0.44 / 0.32 |
  | whole network, trained from scratch | 6 missions | 0.55 |
  | small probe, from scratch, instantaneous inputs, perfect waves | 8 missions | ≈ 0.29 (used here only as a lower bound on what is learnable) |
  | in-mission training | first 30 s / first 1 min | 0.77 / 0.64 |

  Coverage of the 90 % intervals drops to 0.54–0.81 before recalibration.
- **Online learning** (weight-space prior too tight; ROADMAP 11.1): error ratio 0.91. Loosened 5× / 10×: 0.59 / 0.48,
  with coverage falling to 0.82 / 0.74.

Reading through sections 2–4 [D]:

- The ranking of the three families is the same zero-shot and after adaptation, and the flat-wide family loses in
  both. This is what sections 2.2 (hierarchical vs flat), 2.6 (pre-emption) and 4 (representation quality fixes the
  fine-tuning start) predict. It is one seed per row, so it is consistent with theory, not a proof of it.
- In-mission one-minute training gives similar ratios (0.58–0.64) for all three families. A short window of
  target data cannot separate the priors. They differ in absolute level because the starting points differ.
- The online-learning numbers show the weight-space side of the same trade-off.
  - Prior too narrow (tight pull toward $\theta_0$): small gain (0.91).
  - Prior looser: large gain (0.48) but over-confident intervals. This matches Kleijn & van der Vaart (2012): a
    misspecified posterior has the wrong width, and a tempered or calibrated posterior is needed (Grünwald & van
    Ommen 2017).

### 5.2 How large a mass deficit would explain the zero-shot gap? An order-of-magnitude check [D, heuristic]

Convert the measured skills to log-loss. Assume a Gaussian predictive whose variance equals its MSE (calibrated).
Then the per-step log-loss difference per channel is $\tfrac12\ln(\text{MSE}_1/\text{MSE}_2)$. The comparison is
zero-shot vs the copy fine-tuned on 6 missions:

- skills: 0.11/0.02, 0.38/0.07, 0.49/0.10, 0.05/0.01, 0.35/0.12;
- per-channel difference: 0.85 + 0.85 + 0.80 + 0.81 + 0.54 ≈ **3.8 nat per step**, i.e. **≈ 980 nat per 256-step
  window**.

If the in-context predictor were the exact Bayes predictive, the resolvability bound (section 2.3) would imply
$\log 1/\Pi(B)\gtrsim 980$ nat for every set $B$ of laws whose per-step loss is within a small ε of the fine-tuned
copy's. The fine-tuned copy only bounds the information floor from above, so the true excess is at least this large.

Cost $-\log$(prior density ratio, value vs centre) of a single coordinate whose required value is $z$ prior scales
away:

| $z$ | Gaussian | Student-t, ν = 3 | Cauchy |
|---|---|---|---|
| 3.9 | 7.6 | 3.6 | 2.8 |
| 7.7 | 29.6 | 6.1 | 4.1 |
| 21 | 220 | 10.0 | 6.1 |
| 45 | 1013 | 13.0 | 7.6 |
| 72 | 2592 | 14.9 | 8.6 |

How to read this:

1. Gaussian tails at the distances the high-fidelity boat needs (7.7σ–72σ for one coupling, 3.9σ–21σ for pitch
   damping), together with hard bounds (infinite cost), are by themselves enough to produce a ~1000-nat deficit. For an
   exact Bayesian, the same coordinates under heavy-tailed priors would cost roughly 5–15 nat each, plus about
   $\tfrac12\log(n\lambda)$ ≈ 3 nat each to resolve them.
   - For ~25 coordinates that is a few hundred nats per window, i.e. ≈ 1 nat/step averaged over the window, and less
     late in the window.
   - So a heavy-tailed, hierarchical prior over a class that **contains** the target would, *for an exact Bayesian*,
     recover most, not all, of the gap even within one window. The rest is recovered across windows.
2. The network is not an exact Bayesian. To learn inference in a region, the network needs training laws there, and
   amortisation guarantees scale like 1/mass (section 3.1). Six independent coordinates each at Cauchy-tail cost
   ~8 nat give a region of mass ~$e^{-50}$, unreachable by $10^4$–$10^5$ training laws. So for the network, heavy
   tails help **only** for unknowns in which it extrapolates, i.e. learns a general estimator (section 3.4). They do
   not help for unknowns it can only look up.
3. The conversion assumes calibrated Gaussian predictive distributions and adds channels as if independent. The
   numbers are orders of magnitude, not measurements.

### 5.3 Where the cost is paid: once per window or once in total [D, exact for the window-reset Bayes predictor]

A cost of $K$ nats is spread over the steps that share one inference. Per-step cost $K/n$:

| $K$ (nat) | example | in context, $n=256$ | 6 missions, $n=2880$ | 1 hour, $n=15000$ |
|---|---|---|---|---|
| 10 | hierarchical switch + a few strengths (ROADMAP 11.12 corner estimate) | 0.039 | 0.003 | 0.0007 |
| 33 | one coordinate at 7.7σ, Gaussian | 0.13 | 0.011 | 0.002 |
| 220 | one coordinate at 21σ, Gaussian | 0.86 | 0.076 | 0.015 |
| 980 | implied deficit (5.2) | 3.9 | 0.35 | 0.067 |

In weight adaptation the cost is not $\log 1/\Pi(B)$ but the information-complexity term: the KL from the adapted
weight distribution to the weight prior centred at $\theta_0$, divided by $n$ (Zhang 2006 gives this form for
Gibbs/tempered posteriors). It is paid once over all target data. This is the theory-level reason why weight
adaptation, not breadth of the in-context prior, is the reliable remedy for a wrong prior.

---------------------------------------------------------------------------

## 6. Answers

### 6.1 Can a wide prior plus a strong learner make up for a wrong prior distribution?

| Regime | Truth inside the support, misplaced mass | Truth outside the support |
|---|---|---|
| exact Bayes, data → ∞ | **yes** (Schwartz; GGV 2000). The rate is set by $\log 1/\Pi(B)$ and entropy; adaptive hierarchical priors recover the best submodel's rate (GLvdV 2008). Not proven for closed-loop data | **no**: limit is the KL projection (Kleijn–vdV 2006); intervals wrong (2012) |
| exact Bayes, one window (zero-shot) | **partly**. Cost $\log 1/\Pi(B)$ per window (2.3). Unexcited directions depend only on the centre (2.5). Heavy tails make large misplacements affordable (5.2) | **no** (T2c lower bound: never better than the best single law in the support) |
| trained in-context network | **only where it learned a general estimator** (above-threshold behaviour, linear unknowns; 3.3–3.4). Otherwise limited by training-law coverage (1/mass). A "stronger" learner approximates the wrong PPD better and may get worse (T4/M18) | **not guaranteed**. Beneficial deviations exist (+7–13 % from width) but carry no guarantee |
| weight adaptation on target data | **yes**, and measured (0.35 → 0.23 with 2 → 6 missions). The cost is paid once over all data (5.3) | **yes**, if the network class can represent the law and the data cover the deployment inputs (Ben-David). Intervals need recalibration (Kleijn–vdV 2012; Grünwald–van Ommen 2017). Measured: wave-direction coverage limits transfer (D7) |

So the answer to the user's question is: **yes, but at the weight-learning stage, not at the zero-shot stage.**
At the zero-shot stage a wider prior helps only if it is wide in the right way:

- the target is in its support;
- it is heavy-tailed where the target is far from the centre;
- it is wide in directions one window can identify;
- the unknowns enter in a form the network can learn as a general estimator.

### 6.2 How much target data does a wide prior need compared with a narrow, correct one? [T + D]

Cumulative excess log-loss after $n$ target steps, up to constants (assumptions in sections 2.2–2.3; i.i.d. in the
cited theorems):

| Prior | Excess (nats) | Per step → 0 as | Notes |
|---|---|---|---|
| narrow, correct, $d_0$ active directions | $\tfrac{d_0}{2}\log n+\log\tfrac1{\pi_N(\theta^*)}$ | $d_0\log n/(2n)$ | Clarke–Barron |
| narrow, wrong (truth outside) | $nD_N+O(\log n)$ | $D_N>0$ (floor) | Kleijn–vdV |
| flat-wide, $d_w\gg d_0$ always active | $\tfrac{d_w}{2}\log n+\log\tfrac1{\pi_w(\theta^*)}$ (+ pre-emption bias, 2.6) | $d_w\log n/(2n)$ | needs $\approx d_w/d_0$ times more data for the same per-step excess |
| hierarchical wide (blocks switched, weight ε each) | narrow cost $+\log\tfrac1\varepsilon$ (truth needs the block) or $+\log\tfrac1{1-\varepsilon}$ (it does not) | as the best submodel | GLvdV 2008; Rousseau–Mengersen 2011; ROADMAP 11.11 |
| nonparametric, compositional (smoothness β, local input dimension $t$) | $n\,\varepsilon_n^2$ with $\varepsilon_n\asymp n^{-\beta/(2\beta+t)}$ (log factors) | $n^{-2\beta/(2\beta+t)}$ | vdV–vZ 2009; Schmidt-Hieber 2020 (ERM, composition); Polson–Ročková 2018 (Bayes, sparse deep nets) |

Data needed for a per-step excess δ:

- parametric: $n\approx\frac{d}{2\delta}\log n+\frac{1}{\delta}\log\frac1{\pi(\theta^*)}$;
- nonparametric: $n\approx\delta^{-(2\beta+t)/(2\beta)}$.

The decisive lever is $t$, the local input dimension. If the generic part of the prior is a function of the whole
58-number token times its history, $t$ is effectively huge and no realistic amount of data suffices. If it is a
function of 3–5 local variables per hull station (section 6.4), the exponent is mild: e.g. β = 2, t = 4 gives
$n\propto\delta^{-2}$.

Under weight adaptation the "prior" is the weight-space prior around $\theta_0$, so the relevant $d$ is the effective
number of weight directions the target needs. It is not measured. The fine-tuning curve has not saturated by 6
missions, and the pretrained start is worth more than 3× the target data compared with the same network trained from
scratch (0.35 with 2 missions vs 0.55 from scratch with 6). This is the measured value of the prior in data units [P].

### 6.3 When does "wide prior + fine-tuning" beat "narrow prior"? [D, with T]

Conditions (all needed):

- **C1 — the narrow prior is wrong in a way that does not shrink.** The truth is outside its support or far outside
  its typical set, so the in-context error has a floor $D_N$. Here $D_N\approx3.8$ nat/step in the Gaussian-calibrated
  conversion (5.2). This is measured as the zero-shot vs fine-tuned gap, not proven to be a floor.
- **C2 — enough target data.** $n>n^*$ with $n^*\approx[\log 1/\Pi_w(B)+C_w(n^*)]/(D_N-\varepsilon_w)$, where $C_w$ is
  the parametric or nonparametric complexity term. Even with $\log 1/\Pi_w+C_w$ of a few thousand nats, $n^*$ is about
  $10^3$ steps (≈ 4 min). The measured crossover is *below* one minute: in-mission first-minute training gives 0.64.
- **C3 — the breadth carries structure.** The wide prior must give the representation features near the target. Flat
  breadth fails this both zero-shot and after 2 missions (5.1); Tripuraneni et al. 2020 explain why: diversity must be
  in the relevant directions.
- **C4 — the learner can move the representation.** Whole-network or LoRA, not head only (Chua et al. 2021; [P]
  0.23 vs 0.32).
- **C5 — the target data cover the deployment inputs.** Driver (the MPC, not only the hand controller), speeds and
  wave directions (Ben-David et al. 2010; [P] D7).
- **C6 — calibration under misspecification.** Sandwich, tempered (SafeBayes) or conformal widths. Otherwise the gain
  in mean comes with over-confidence ([P] 11.1, 11.5, 11.6).

If C1 fails (the narrow prior is correct), narrow wins at every $n$, most clearly zero-shot. The gap shrinks like
$(d_w-d_0)\log n/(2n)$ for flat breadth, or to a constant for hierarchical breadth. If C3 fails, wide + fine-tuning
can lose to narrow + fine-tuning at moderate $n$; this is the measured general-vs-catalogue case.

### 6.4 What a well-designed wide prior for this problem looks like [D, with T]

The construction is levelled: hard where physics is certain, wide where a window or the data can identify, centred by
physics where they cannot, and switched in blocks.

**L0 — hard structure (zero breadth, correct for any rigid hull in any sea).**

- Newton–Euler map from forces to accelerations, with a positive-definite mass matrix and consistent Coriolis terms.
  Unknown forces enter only through it.
- Causality.
- Passivity of radiation memory: Cummins (1962), *The impulse response function and ship motions*, Schiffstechnik 9
  (DTMB report 1661) — convolution kernels with frequency-independent coefficients. Volume/issue numbering differs
  between sources (vol. 9 vs 47) [partly verified].
- Actuators as causal delay-and-lag chains.

The catalogue family already has most of this (layer 1, $M_t$, the actuator family).

**L1 — hull forces as a local-section functional (one general construction instead of many items).**

Slender-body / strip momentum flux (my derivation of the standard structure; textbook slender-body theory, not
re-cited here): the sectional force density is

$$ f(x,t) = -\frac{D}{Dt}\big[m(x,d)\,V_\perp\big] + c(x,d)\,V_\perp|V_\perp| + \rho g A(x,d) + \ldots,\qquad \frac{D}{Dt}=\partial_t-U\partial_x .$$

Integrating over the length gives an integral of local terms plus **boundary terms** $U[m V_\perp]$ at the ends. At a
transom stern $m(x_{\rm stern},d)\ne0$, which gives the stern lift $|u|\,m_{22}(d_1)(v+x_1r)$. The $\partial_x$ part
gives the distributed Munk moment. With $D d/Dt=V$, the vertical analogue gives the planing term $m_a'(d)V^2$. The
pressure drag $m(g+\dot w)\Theta$ is the normal force projected by trim.

These are exactly the dominant terms in [GZ_VS_FAMILY] §3.2:

| channel | dominant term (share of the error variance) |
|---|---|
| sway | stern lift (0.75) |
| yaw | Munk moment +1.68, stern-lift moment −1.04 |
| pitch | planing/impact (0.71) |
| heave | restoring (1.31) |
| surge | pressure drag (0.28) |

A prior over this *form* has these parts:

- unknown sectional functions $m(\cdot)$, $c(\cdot)$, $A(\cdot)$ of local immersion and station position, with
  heavy-tailed scales and monotonicity where physics guarantees it;
- a shared generic local residual $\Phi(d_i, V_i, \dot V_i, u, x_i;\ \text{filtered history})$ with a full-support
  prior over functions of these 4–6 local variables (a GP or random features with a hyperprior on scale, so that the
  rates adapt; vdV–vZ 2009);
- boundary terms at the transom.

It is wide: full support over the compositional class, with local input dimension $t\approx4$–$6$. It is general: any
slender planing hull. It covers mechanisms *as a class* rather than one at a time. In the data generator, local
immersions come from the full wave field at many stations, while the network keeps its 15 measured points. The part
that cannot be observed then appears in training as noise, which is correct.

Leakage warning: the high-fidelity boat is itself a 2D+t strip model, so this class contains it by construction.
Zero-shot scores on it would be optimistic. Acceptance must be on a simulator or hull of different construction
(the project's planned Wigley hull, or a 3D/CFD-derived model), and finally on real data.

Effects outside this class, left to L4 and to data:

- 3D transom/end effects beyond the boundary term;
- spray and ventilation hysteresis (partly representable as hidden processes);
- wave diffraction and orbital velocities (the high-fidelity boat has none);
- structural flexibility;
- propulsor–hull interaction beyond the priming class.

**L2 — heavy-tailed scales.** Use log-uniform, half-Cauchy or regularly varying priors on magnitudes, couplings and
damping, with soft constraints instead of hard bounds. Let the existing divergence test remove unphysical draws.

- [T] O'Hagan (1979), *On outlier rejection phenomena in Bayes inference*, JRSS B 41(3):358–367; Andrade & O'Hagan
  (2006), *Bayesian robustness modeling using regularly varying distributions*, Bayesian Analysis 1(1):169–188: heavy
  tails let the data override a conflicting prior.
- [T] Carvalho, Polson, Scott (2010), *The horseshoe estimator for sparse signals*, Biometrika 97(2):465–480: large
  signals are not shrunk, small ones are.
- The numerical effect is in the table in 5.2.

**L3 — hierarchical blocks.** Propulsion classes (including a priming/aeration law driven by inlet submergence),
hidden processes, environment (wind, current), roll coupling. Each is switched on as a whole with weight ε, at cost
$\log 1/(1-\varepsilon)$ per window for targets that do not need it (ROADMAP 11.11). Correlate blocks that physics
correlates within one speed/trim regime.

**L4 — generic residual.** Full support over causal functionals: the existing layer-3 class, made **local** (it reads
local variables, as in L1) and **block-switched with small weight**, and, following Plumlee (2017), discouraged from
explaining what the physics blocks explain (2.6). It provides the KL support needed by Schwartz/GGV without flattening
the mass.

**L5 — input breadth.** Drivers (autopilot, hand controller, MPC-like planners with constraint-active behaviour),
speeds 14–29 kn (the MPC's measured range), all relative wave directions, and sea states including fixed-sea
multi-mission sequences. This matters because $p(\xi\mid H)$ changes with the driver ([GZ_VS_FAMILY] §8.1) and
because PFN guarantees hold only on $\Pi_H$.

**The learner that goes with it.**

- **(a) Stage I:** enough distinct laws to pass the diversity threshold for the linear-in-coefficient parts
  (Raventós: order $10^4$–$10^5$ for 8–32 linear dimensions). Cost here: about 21–30 h of generation per $10^4$ laws
  (ROADMAP 11.9). A practical constraint [P].
- **(b) Stage II:** a weight-space prior centred at $\theta_0$ with its width learned or tempered, not fixed by hand.
  Whole-network or LoRA, early stopping, recalibration, re-training between missions on all target data (ROADMAP
  11.6/11.9 already recommend this).
- **(c) Later, once data from several real boats or sea areas exist:** learn the hyperprior across them (Baxter 2000;
  PACOH 2021). This is the only setting in which meta-learning bounds actually apply, because then the target is
  drawn from the environment.

**What this does not fix.** Mechanisms outside L1–L4 and outside the network's ability to generalise still need
target data. The design lowers the zero-shot error and the data needed; it removes neither.

### 6.5 Back to "why do we need the error predictor" (Line B's view only)

- For the MPC, what matters is:
  - the one-step mean along the MPC's *own* trajectories;
  - calibrated tails of the safety quantities (ROADMAP 9; flagged: planning calls in which every candidate violates
    the constraints are 58–100 % in meta7, so tail calibration feeds directly into the plan ranking).
- Theory says the reliable route to both is stage-II learning with coverage of the MPC's input distribution and
  misspecification-aware calibration (6.3, C4–C6). The wide prior's job is to:
  - provide the representation and the starting point;
  - provide honest uncertainty before data;
  - make that learning cheaper.

  It is not the source of accuracy on the real boat.
- Large-data learning is reliable in the theoretical sense (consistency of a full-support weight-space learner,
  compositional rates) under three conditions:
  - the data cover the deployment distribution, including wave directions and the MPC as driver;
  - the representation is adaptable;
  - widths are recalibrated.

  The closed-loop data-collection gap (the learner is corrected only where the controller goes) remains outside
  these theorems (RELATED_WORK_ONLINE_THEORY §4; ROADMAP 11.11).

---------------------------------------------------------------------------

## 7. Open points and cheap checks of the assumptions (none run here)

1. **[O] Best single law in the support per window.** By the T2c lower bound, this is the ceiling of an exact-Bayes
   zero-shot learner. The closure experiments measured fixed-weight mixtures instead. A bounded fit of the catalogue
   forms per window ([GZ_VS_FAMILY] §9 item 3) would settle it.
2. **[O] Does the network extrapolate in magnitude along linear coefficients?** Test it on in-family laws whose
   strengths are placed beyond the training range by a factor of 2–10 (inference only, new laws). This is the
   condition under which heavy-tailed priors help the *network* and not just an exact Bayesian (5.2, point 2). It is
   a check of an assumption, not a validation sweep.
3. **[O] Effective number of weight directions the target needs.** This sets stage-II data needs (6.2). Proxy: compare
   the LoRA rank (or adapted subspace size) against the fine-tuning error at 2/4/6 missions. These are full-model
   measurements, not a low-order ceiling estimate.
4. **[O] Distribution of the target's local variables.** Does the high-fidelity boat's stern immersion, local
   relative velocity etc. fall inside what a local-section prior would generate under the training drivers? This is a
   descriptive check on recorded data.
5. **[O]** The 3.8 nat/step conversion assumes calibrated Gaussian predictives. Recomputing it with the flow head's own
   log-densities (zero-shot vs fine-tuned) would remove that assumption.

---------------------------------------------------------------------------

## 8. References (verification status)

"Verified" = title, authors, year and venue confirmed by a web search on 2026-10-05; any content quoted was checked
against the abstract (or, for Raventós, the PDF text).

| Reference | Status |
|---|---|
| Schwartz, L. (1965). On Bayes procedures. Z. Wahrscheinlichkeitstheorie verw. Geb. 4:10–26 | verified |
| Diaconis, P., Freedman, D. (1986). On the consistency of Bayes estimates. Ann. Statist. 14(1):1–26 | verified |
| Ghosal, S., Ghosh, J.K., van der Vaart, A.W. (2000). Convergence rates of posterior distributions. Ann. Statist. 28(2):500–531 | verified |
| Ghosal, S., van der Vaart, A.W. (2007). Convergence rates of posterior distributions for noniid observations. Ann. Statist. 35(1):192–223 | verified |
| Ghosal, S., van der Vaart, A. (2017). Fundamentals of Nonparametric Bayesian Inference. Cambridge Univ. Press | verified (book); the chapter number for Doob's theorem is from memory |
| Ghosal, S., Lember, J., van der Vaart, A. (2008). Nonparametric Bayesian model selection and averaging. Electron. J. Statist. 2:63–89 | verified |
| van der Vaart, A.W., van Zanten, J.H. (2009). Adaptive Bayesian estimation using a Gaussian random field with inverse Gamma bandwidth. Ann. Statist. 37(5B):2655–2675 | verified |
| Rousseau, J., Mengersen, K. (2011). Asymptotic behaviour of the posterior distribution in overfitted mixture models. JRSS B 73(5):689–710 | verified |
| Bhattacharya, A., Pati, D., Yang, Y. (2019). Bayesian fractional posteriors. Ann. Statist. 47(1):39–66 | verified |
| Kleijn, B.J.K., van der Vaart, A.W. (2006). Misspecification in infinite-dimensional Bayesian statistics. Ann. Statist. 34(2):837–877 | verified |
| Kleijn, B.J.K., van der Vaart, A.W. (2012). The Bernstein–von Mises theorem under misspecification. Electron. J. Statist. 6:354–381 | verified |
| Grünwald, P., van Ommen, T. (2017). Inconsistency of Bayesian inference for misspecified linear models, and a proposal for repairing it. Bayesian Analysis 12(4):1069–1103 | verified |
| Yao, Y., Vehtari, A., Simpson, D., Gelman, A. (2018). Using stacking to average Bayesian predictive distributions. Bayesian Analysis 13(3) | verified |
| Barron, A.R., Cover, T.M. (1991). Minimum complexity density estimation. IEEE Trans. Inf. Theory 37(4):1034–1054 | verified |
| Zhang, T. (2006). From ε-entropy to KL-entropy: analysis of minimum information complexity density estimation. Ann. Statist. 34(5):2180–2210 | verified |
| Clarke, B.S., Barron, A.R. (1990). Information-theoretic asymptotics of Bayes methods. IEEE Trans. Inf. Theory 36(3):453–471 | verified |
| Gill, R.D., Levit, B.Y. (1995). Applications of the van Trees inequality: a Bayesian Cramér–Rao bound. Bernoulli 1(1–2):59–79 | verified |
| MacKay, D.J.C. (1992). Bayesian interpolation | cited in ROADMAP 11.11; not re-verified here |
| Kennedy, M.C., O'Hagan, A. (2001). Bayesian calibration of computer models. JRSS B 63(3):425–464 | verified |
| Brynjarsdóttir, J., O'Hagan, A. (2014). Learning about physical parameters: the importance of model discrepancy. Inverse Problems 30(11):114007 | verified (abstract quoted) |
| Plumlee, M. (2017). Bayesian calibration of inexact computer models. JASA 112(519):1274–1285 | verified |
| Müller, S., Hollmann, N., Pineda Arango, S., Grabocka, J., Hutter, F. (2022). Transformers can do Bayesian inference. ICLR 2022 | verified |
| Nagler, T. (2023). Statistical foundations of prior-data fitted networks. ICML 2023, PMLR 202:25660–25676 | verified |
| Raventós, A., Paul, M., Chen, F., Ganguli, S. (2023). Pretraining task diversity and the emergence of non-Bayesian in-context learning for regression. NeurIPS 2023 | verified (threshold numbers read from the arXiv PDF) |
| Kirsch, L., Harrison, J., Sohl-Dickstein, J., Metz, L. (2022). General-purpose in-context learning by meta-learning transformers. arXiv 2212.04458 (NeurIPS 2022 Workshop on Meta-Learning) | verified (venue per search listing) |
| Wu, J., Zou, D., Chen, Z., Braverman, V., Gu, Q., Bartlett, P. (2024). How many pretraining tasks are needed for in-context learning of linear regression? ICLR 2024 | verified |
| Garg, S., Tsipras, D., Liang, P., Valiant, G. (2022). What can transformers learn in-context? A case study of simple function classes. NeurIPS 2022 | verified |
| Panwar, M., Ahuja, K., Goyal, N. (2024). In-context learning through the Bayesian prism. ICLR 2024 | verified (venue from the arXiv comments) |
| Yadlowsky, S., Doshi, L., Tripuraneni, N. (2023). Pretraining data mixtures enable narrow model selection capabilities in transformer models. arXiv 2311.00871 | verified (arXiv; no venue claimed) |
| Bai, Y., Chen, F., Wang, H., Xiong, C., Mei, S. (2023). Transformers as statisticians: provable in-context learning with in-context algorithm selection. NeurIPS 2023 | verified |
| Baxter, J. (2000). A model of inductive bias learning. JAIR 12:149–198 | verified |
| Maurer, A., Pontil, M., Romera-Paredes, B. (2016). The benefit of multitask representation learning. JMLR 17(81):1–32 | verified |
| Tripuraneni, N., Jordan, M.I., Jin, C. (2020). On the theory of transfer learning: the importance of task diversity. NeurIPS 2020 | verified |
| Chua, K., Lei, Q., Lee, J.D. (2021). How fine-tuning allows for effective meta-learning. NeurIPS 2021 | verified (abstract quoted) |
| Kumar, A., Raghunathan, A., Jones, R., Ma, T., Liang, P. (2022). Fine-tuning can distort pretrained features and underperform out-of-distribution. ICLR 2022 | verified |
| Ali, A., Kolter, J.Z., Tibshirani, R.J. (2019). A continuous-time view of early stopping for least squares regression. AISTATS 2019, PMLR 89:1370–1378 | verified |
| Gozeten, H.A., Ildiz, M.E., Zhang, X., Soltanolkotabi, M., Mondelli, M., Oymak, S. (2025). Test-time training provably improves transformers as in-context learners. ICML 2025 | verified |
| Amit, R., Meir, R. (2018). Meta-learning by adjusting priors based on extended PAC-Bayes theory. ICML 2018 | verified |
| Rothfuss, J., Fortuin, V., Josifoski, M., Krause, A. (2021). PACOH: Bayes-optimal meta-learning with PAC-guarantees. ICML 2021, PMLR 139:9116–9126 | verified |
| Ben-David, S., Blitzer, J., Crammer, K., Kulesza, A., Pereira, F., Vaughan, J.W. (2010). A theory of learning from different domains. Machine Learning 79:151–175 | verified |
| Schmidt-Hieber, J. (2020). Nonparametric regression using deep neural networks with ReLU activation function. Ann. Statist. 48(4):1875–1897 | verified |
| Polson, N.G., Ročková, V. (2018). Posterior concentration for sparse deep learning. NeurIPS 2018 | verified |
| O'Hagan, A. (1979). On outlier rejection phenomena in Bayes inference. JRSS B 41(3):358–367 | verified |
| Andrade, J.A.A., O'Hagan, A. (2006). Bayesian robustness modeling using regularly varying distributions. Bayesian Analysis 1(1):169–188 | verified |
| Carvalho, C.M., Polson, N.G., Scott, J.G. (2010). The horseshoe estimator for sparse signals. Biometrika 97(2):465–480 | verified |
| Cummins, W.E. (1962). The impulse response function and ship motions. Schiffstechnik 9:101–109 (DTMB report 1661) | partly verified (volume numbering differs between sources) |
| Chen, X., Hu, J., Jin, C., Li, L., Wang, L. (2022). Understanding domain randomization for sim-to-real transfer. ICLR 2022 | verified; cited in ROADMAP 11.10 as "Theorem 4" — theorem number not re-checked here, so not relied on |
