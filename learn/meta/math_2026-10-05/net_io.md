# What the network sees and predicts (net_io)

Scope: the one-step error predictor of the catalogue family, checkpoint
`studies/_cache/meta7/model_p.pt` (row `cat_p_*`, variant p), as trained by
`studies/meta_step7.py` and as scored on gz streams by `studies/onestep_gz.py`.
Variants a and w are described where they differ. All `file:line` refer to the
working tree on 2026-10-04 (branch usv-priors-mpc-2026-10-01; `model_preview.py`
and `meta_step7.py` have uncommitted edits that only add the `arch` size
argument, `git diff`). Numbers marked [checked] come from the read-only script
`wf_math/check_netio.py` (output reproduced in section 10).

## 0. Corrections to the task brief and docstring/code disagreements

| # | Statement | What the code does |
|---|---|---|
| D1 | Brief: "the network predicts the one-step velocity error in 5 reduced channels (RED_VEL)". | The target has **10** channels, `model3.C7 = 10` (model3.py:46), defined by `data3.e0_from` (data3.py:78-88): 5 velocity channels (u, v, r, heave rate w, pitch rate q), 3 displacement channels (z, theta, psi) and 2 actuator channels. All 10 enter the flow-matching loss (model3.py:235-244, mean over the last axis). The scoring tables print the first 5 (onestep_gz.py:411-412, model3.py:1204-1210). `data2.RED_VEL` (data2.py:368) is used only by `push_response` (operator target relabelling), never by the network data path. |
| D2 | model3.py:11 docstring: "e0 ... 7 channels". | Code: 10 channels (model3.py:46, data3.py:60-62). The docstring is stale. |
| D3 | data2.py:53-54 docstring: recorded `E` (5) "the target quantity". | `E` (5 channels, plant minus the WAVE-FORCED twin, episode7.py:229) is stored in the pack but **never read** by `Data3` / `DataP`; the network target is `E0` from `<split>_e0.npz` (model3.py:86, 108-109), the error against the wave-free model0. |
| D4 | task.py:209 docstring: "stepped one control step (0.25 s)". | Plant time per control step is `dt*sub = 0.24 s` in both worlds (low: 0.04 s x 6; high/gz: 0.02 s x 12) [checked: env dt 0.04, sub 6; gz t steps 0.24]. The whole network data path uses `dtc = 0.24 s`. |

## 1. Symbols

Time and indices
- `k` control-step index within a mission/episode, `k = 0, 1, ...`; `t_k` plant
  time at the start of step k; `Delta = dt*sub = 0.24 s` (relabel._env,
  relabel.py:232-242; [checked]); `t_k = k*Delta` up to float accumulation
  (task.py:323-334 adds `dt` per substep).
- `h = Delta/2 = 0.12 s` (model_preview.py:192, mpc_constrained.py:156).
- Window: start `a` (absolute step), window position `p = t - a`, length `L`.

Plant state (both worlds), `s in R^14`, layout `[x, y, z, phi, theta, psi, u, v, w, p_r, q, r, T_a, delta_a]`
(planing_vessel.py:251-254, 362-365; lofi.py:535-539):
- `x, y` earth-frame position (m); `z` heave displacement (m; gz: relative to
  the rest position `Z_rest`, planing_vessel.py:262); `phi` roll (rad);
  `theta` pitch (rad); `psi` heading (rad, not wrapped);
- `u, v` body surge/sway velocity (m/s); `w` heave rate (m/s, `dz/dt = w`,
  planing_vessel.py:364); `p_r` roll rate; `q` pitch rate (`dtheta/dt = q`);
  `r` yaw rate (`dpsi/dt = r`) (rad/s);
- `T_a` thrust actuator STATE (N), `delta_a` nozzle angle STATE (rad).
- Low world (lofi ReducedPlant): `phi = p_r = 0` always [checked: train XS
  cols 3, 9 max |.| = 0].

Reduced state `rho = P s in R^10`, `P` = index list `[0,1,6,2,8,4,10,5,11,7]`,
i.e. `rho = [x, y, u, z, w, theta, q, psi, r, v]` (data2.py:361-365,
model3.py:441-443). Roll and roll rate are not in `rho`.

Commands `c_k = (c_T, c_delta)`: fractions, `c_T = T_cmd / t_max in [0,1]`,
`c_delta = delta_cmd / rud_max in [-1,1]`, held over step k.

Fixed constants [checked]: `u_ref = 12.8611 m/s` (25 kn), `L = 5.4 m`,
`t_max = 6365.66 N`, `rud_max = 0.43633 rad` (25 deg), `z0 = 0.15366 m`,
`theta0 = -0.009229 rad` (reduced model running attitude, `red.p`),
stations `x_st = [-2.125, -0.8031, 0.5188, 1.8407, 3.1625] m` (body x, stern to
bow; `linspace(x_stern, x_bow, 5)`, lofi.py:380), `y_off = [-1.0126, 0, 1.0126] m`
(`+-B/2`, lofi.py:381). Station index `n = 3 i + j`, `i` = x_st index, `j` =
y_off index (`.ravel()` of a (5,3) array, data2.py:124-127; `flatten(-2)`,
model_preview.py:215-218). Station position for pose `(x, y, psi)`:
`X_n = x + x_st[i] cos psi - y_off[j] sin psi`,
`Y_n = y + x_st[i] sin psi + y_off[j] cos psi` (episode5.py:60-67).

Sea: a sum of `N_C` components `{a_c, k_c, theta_c, omega_c, phi_c}`,
`eta(X, Y, t) = sum_c a_c cos(k_c (X cos theta_c + Y sin theta_c) - omega_c t + phi_c)`
(data5.py:49-56; lofi.py:450-457; identical sums).

Dead-reckoned mid-step pose from `s_k` (episode5.py:44-57):
`x^m = x + (u cos psi - v sin psi) h`, `y^m = y + (u sin psi + v cos psi) h`, `psi^m = psi + r h`.

## 2. What is stored per training step (pack7) and what the network reads

Recorded at the START of step k (episode7.py:180-243, `s0 = m.s` before
`m.advance`), packed by `episode7.pack7` -> `data2.pack` (episode7.py:303-331,
data2.py:398-429), float32 (data2.py:407).

`S (n, T, 26)`, from `episode6.op_inputs6` (episode6.py:151-162) with
`residuals.inputs_of` (residuals.py:52-56):

| S col | quantity | definition |
|---|---|---|
| 0 | speed offset | `u/u_ref - 1` |
| 1 | sway | `v` (m/s) |
| 2 | yaw rate | `r L / u_ref` |
| 3, 4 | heading | `cos psi`, `sin psi` |
| 5 | thrust actuator | `T_a / t_max` (STATE, not command) |
| 6 | nozzle actuator | `delta_a / rud_max` (STATE) |
| 7 | heave rate | `w` (m/s) |
| 8 | pitch rate | `q` (rad/s) |
| 9 | heave about running attitude | `(z - z0) / 0.2` |
| 10 | pitch about running attitude | `(theta - theta0) / 0.05` |
| 11..25 | measured elevations | `eta(X_n(s_k), Y_n(s_k), t_k)`, n = 0..14 (m), via the twin's `_surface` = the episode's own sea (data2.py:123-128) |

Other arrays the network reads: `U (n,T,2) = c_k` (episode7.py:201, 232),
`W_MID (n,T,15) = eta(X_n(x^m,y^m,psi^m), Y_n(...), t_k + h)` (episode5.py:70-77,
episode6.py:159), `XS (n,T,14) = s_k` (only for the mid poses PM and e0),
`len (n,)`. The e0 file `<split>_e0.npz` (data3.e0_split, data3.py:91-107).

Arrays stored but NOT read by the network (Data3 / DataP read only S, U, XS,
len, W_MID, E0: model3.py:85-115, model_preview.py:158-202): `E` (5), `RR`,
`RN`, `ACG`, `D`, `AV`, `APK`, `AMIN`, `HMIN`, `PUSH`, `IMP`, `SLAM`, `OBS`
(OBS = roll, roll_rate, rpm, wind_speed_rel, wind_angle_rel, stw_u, stw_v,
gnss_q, cat_base.py:154-155; "stored only", episode7.py:36-40). data7.py builds
multi-step branches (XS, E0 per plan) for evaluation/rollout training; it does
not enter the one-step tokens.

Training split [checked]: `n = 9860` episodes, `T = 360`, `len` min 31 /
median 360 / max 360 (90 s episodes, `T_EP = 90`, data2.py:87); targets valid for
`k < len - 1` (model3.py:94-97).

## 3. The target

Model0 (the MPC's model): `F0(rho, c)` = 6 substeps of `ReducedModel.step` with
zero wave elevation at all stations, `dt = 0.04 s`, thrust `= c_T t_max` and
rudder `= c_delta rud_max` applied directly (ideal actuators; the actuator
state `T_a, delta_a` is not an input of F0) (data3.py:66-75; torch replica
model3.py:461-523; the numpy step control/reduced.py:366ff). With zero waves
the heave/pitch equilibria are `z0`, `theta0` (model3.py:476-483, 493-497).

Error (data3.py:78-88), with `E7 = (2, 9, 8, 4, 6, 3, 5, 7)` (data3.py:59):

    e_k[0:8]  = ( rho_{k+1}[E7] - F0(rho_k, c_k)[E7] ) / Delta     (u, v, r, w, q, z, theta, psi)
    e_k[8:10] = ( s_{k+1}[12:14] / (t_max, rud_max) - c_k ) / Delta  (thrust, nozzle actuator gap)

Units: m/s^2, rad/s^2 for the velocity channels; m/s, rad/s for z, theta,
psi; 1/s for the actuator channels.

Network target: `y_k = e_k / sigma_e` (model3.py:108-109), `sigma_e` = per-channel
std of e0 over the valid steps of the meta7 train split + 1e-6 (model3.py:98-99),
stored in the checkpoint [checked, identical in model3.pt and model_p.pt]:

    sigma_e = [1.5927, 1.4379, 0.5396, 4.4931, 1.3361, 0.6998, 0.1607, 0.0604, 0.2214, 0.3250]

The flow head models `Y = asinh(y / kappa)`, `kappa = 0.5` (model3.py:50, 238);
samples are returned as `y = kappa sinh(Y)` (model3.py:178, 197).

Consequence for any family argument: because F0 has NO waves and IDEAL
actuators, `e_k` contains the entire wave response, the actuator-lag response
and every model error of the plant, plus the actuator gap channels.

## 4. The token at step t (variant p: 58 numbers)

`tok_t = [ X_t (11) | U_t (2) | asinh(E_{t-1}/kappa) (10) | f_t (1) | WAVE_t (34) ]`
(model3.make_tokens model3.py:118-121; model_preview.tokens_p
model_preview.py:354-362; `D_TOK = 24` model3.py:47; `N_WX = 34`
model_preview.py:89).

| idx | symbol | definition (normalisation) | code |
|---|---|---|---|
| 0..10 | `X_t` | `(S_t[0:11] - x_mu) / x_sd`, zero for `t >= len` | model3.py:103-106 |
| 11, 12 | `U_t` | `(c_t - u_mu) / u_sd` (command applied DURING step t) | model3.py:104, 107 |
| 13..22 | `A_t` | `asinh(Ep_t / 0.5)`, `Ep_t = y_{t-1} (+ obs noise in training)`, `Ep_t = 0` at episode step 0 (`t >= 1` mask); at window position 0 with `a > 0`, `Ep = y_{a-1}` | model3.py:131-137, 120 |
| 23 | `f_t` | 1 at window position 0, else 0 (NOT "episode start": any window start) | model3.py:138 |
| 24..38 | `Wm_t` | measured elevations `(S_t[11:26] - w_mu)/w_sd + msd*N(0,1)`, times measured flag | model_preview.py:188-189, 268-274 |
| 39..53 | `Wp_t` | preview: `(W_MID_t - w_mu)/w_sd + err_t/w_sd`, times preview flag | model_preview.py:190, 278-289 |
| 54 | `M_t` | measured flag `[t - m <= 0]` | model_preview.py:267-268, 292 |
| 55 | `sigma_t` | `max(0, t - m) / 24` (steps since the moment, over HB) | model_preview.py:291 |
| 56 | `Pf_t` | preview flag `[(t - m < hp) and (hp > 0)]` | model_preview.py:276-277 |
| 57 | `lev_t` | `lam * sqrt(1 + (0.5 * lead_t)^2) * Pf_t`, `lead_t = max(0, t - m) Delta + h` | model_preview.py:250-251, 275, 290 |

Here `m` is the per-window "moment" (absolute step, last measured step), `hp`
the preview horizon (steps), `lam` the forecast-error level, `msd` the sensor
noise sd (normalised units), all per window (section 5). The forecast error
`err_t` (metres, per station) is `fc_eval` at the recorded mid pose
`PM_t = (x^m, y^m, psi^m)(s_t)` (model_preview.py:193-195, 279-288), with
`tm = t Delta + h` (model_preview.py:284):

    err_t[n] = sum_c a_c [ (1 + 0.3 lam eps_c) cos(ph_c + d_c) - cos(ph_c) ]
    ph_c = k_c (X_n cos theta_c + Y_n sin theta_c) - omega_c tm + phi_c
    d_c  = lam (1.0 xi0_c + 0.5 lead_t xi1_c) + omega_c * 0.2 lam eta
    xi0, xi1, eps ~ N(0, I_{N_C}),  eta ~ N(0, 1), drawn once per window

(model_preview.py:221-247, constants `FC` model_preview.py:92). `err = 0`
exactly when `lam = 0` or `hp = 0` (model_preview.py:279).

Variant a (model3.pt): tokens are columns 0..23 only. Variant w: columns 0..57
with `hp = 0` always, so columns 39..53, 56, 57 are always 0
(model_preview.py:324-325).

Input projection: `inp = Linear(58,192) -> SiLU -> Linear(192,192)`
(model_preview.py:373, model3.py:62-68), plus a learned position embedding
`pos[p]` (256 x 192, zero init, model3.py:149, 161).

## 5. Training convention (meta7 model_p.pt)

Window draw (model3.fm_draw, model3.py:224-232; used at model_preview.py:443):
episode `i` uniform over the training pool (all but `n//33 = 298` held-out
episodes, model3.py:209-221); start `a ~ U{0..max(0, len_i - 1 - L)}`, replaced
by `a = 0` w.p. 0.25; `L = W_CTX = 256` (model3.py:48; `ctx = 256` [checked]).
For `len = 360`: `a in {0..103}`.

Per window (draw_cfg, model_preview.py:310-333; constants model_preview.py:93-94,
stored in `ck["conv"]` [checked]):
- moment `m = a + m_rel`, `m_rel ~ U{0..n_valid-1}` (n_valid = valid tokens of
  the window, 256 for full episodes), replaced by `m_rel = L` (all tokens
  measured) w.p. `P_NOROLL = 0.2`;
- `hp ~ U{1..30}`, replaced by 0 w.p. `P_NOPREV = 0.2`;
- `lam ~ LogU[0.02, 1.5]`, replaced by 0 w.p. `P_LAM0 = 0.15`;
- `msd ~ LogU[0.003, 0.03]` (normalised elevation units);
- forecast perturbations `xi0, xi1, eps (N_C)`, `eta` ~ N(0,1) (model_preview.py:301-307);
- error-history observation noise `Ep += sd_c N(0,1)`, `sd_c = b exp(0.3 z_c)`,
  `b ~ LogU[0.01, 0.1]`, `z_c ~ N(0,1)` per channel (model3.py:201-206, 135-137).

Loss mask (model_preview.py:388-397): `ok_t = valid_t and t < T and (t - m <= J0_MAX = 32)`.
Loss: rectified flow matching on all 10 channels,
`|| v(Y_tau, tau, h_t) - (Y1 - Y0) ||^2 / 10`, `Y1 = asinh(y_t/0.5)`,
`Y0 ~ N(0, I)`, `tau ~ U(0,1)`, `Y_tau = (1-tau) Y0 + tau Y1`, averaged over
`ok` tokens (model3.py:235-244). AdamW lr 3e-4, wd 1e-4, OneCycle, batch 48,
grad clip 1.0, early stop on held-out FM loss (model_preview.py:417-471).

Configuration share at window position p (derived from the draws above,
n_valid = 256): P(output at p sees only measured tokens, i.e. `m_rel >= p`)
`= 0.2 + 0.8 (256 - p)/256`; P(p scored) `= 0.2 + 0.8 (256 - max(0, p - 32))/256`.
At p = 240: 0.250 of all windows, 0.71 of scored ones; at p = 255: 0.203 and
0.67. With `hp > 0` (0.8) the measured tokens also carry the lead-h preview.

## 6. Network and normalisation statistics

Trunk: causal TransformerEncoder, d = 192, 6 layers, 6 heads, FF 384 (ReLU,
the PyTorch default), pre-norm, dropout 0, final LayerNorm; causal mask
`triu(-inf, 1)`: output `h_t` depends on tokens `<= t` of the window only
(model3.py:145-164; `arch` absent in model_p.pt = default [checked]).
Flow head: `MLP(10 + 32 + 192 -> 512 -> 512 -> 512 -> 10)`, SiLU, on
`[Y, t_embed(tau), h_t]`, `t_embed` = sin/cos of `tau * f`, 16 log-spaced `f`
in [1, 1000] (model3.py:71-75, 155, 166-167). Sampling: 24 Euler steps from
`Y0 ~ N(0, I)` (model3.py:170-197).

Statistics (all in `ck["stats"]` [checked]):
- `x_mu, x_sd, u_mu, u_sd, w_mu, w_sd` = the library `lib.npz` mean / std of
  the 28 operator inputs (`S[0:26]`, `U`) over steps >= 40 of 40 OPERATOR-FREE
  pilot episodes of the low world with the OLD actuator family
  (meta_step2.py:202-237, data2.py:296-306; meta7 copies lib.npz from meta2,
  meta_step7.py:17, 199-202). They are NOT the training split's statistics:
  e.g. the speed feature has train mean -1.27, sd 1.87 after normalisation
  (section 10).
- `sigma_e` = training split e0 std (section 3).

      x_mu = [-0.0571 -0.0298 0.0013 -0.0690 0.0589 0.3341 -0.0329 0.0001 -0.0001 0.0390 -0.0064]
      x_sd = [ 0.1599  0.8480 0.0406  0.7237 0.6841 0.1452  0.4170 0.6052  0.1967 1.2879  1.2040]
      u_mu = [0.3344 -0.0324]   u_sd = [0.1546 0.4213]
      w_mu = 0.0064..0.0087 m (15)   w_sd = 0.2501..0.2511 m (15)

## 7. Window length, positions and masks

- Context: `W_CTX = 256` tokens = 61.44 s; `ctx = min(256, valid tokens of
  the longest episode) = 256` (model_preview.py:205-209; [checked]).
- Causal attention mask (model3.py:162-164). The target `y_t` is NOT in token
  t (token t holds `y_{t-1}`), so output t predicts
  `p(y_t | tokens_{a..t})`, where token t contains `rho_t`-derived features,
  the command `c_t` of the same step, measured `eta` at `t_t` and the preview
  of `eta` at `t_t + h`.
- Loss mask: section 5. Padding: `X`, `E`, `Wm`, `Wmid` zero for `t >= len`
  (model3.py:105-109, model_preview.py:188-190), `valid` false there.
- Rollout-only masks (`hide`, kv cache) are not used in one-step scoring.

## 8. How a gz stream becomes the same columns

Recording (onestep_gz.record, onestep_gz.py:136-169): `final_eval.make_mission`
builds `task.Mission("high", seed, leg)` and swaps in `PlaningVesselGZ`
(final_eval.py:208-223); commands from the hand MPPI (thrust) and the heading
autopilot (nozzle) (task.py:295-300), through `SafeMode` (roll > 20 deg: half
thrust, nozzle 0; mpc_constrained.py:620-641) [checked: SafeMode never on,
`n_safe = 0` in all 16 missions]. Stored: `xs[k] = s_k` (k = 0..n, float64),
`t[k]`, `u[k] = c_k` after SafeMode, `apk`, `hmin`, the sea arrays
`{a, k, th, w, phi}`, `consts` (mpc_learned.py:515-521).

Mapping (onestep_gz.full_live onestep_gz.py:204-221 -> mpc_constrained.LiveDataW
mpc_constrained.py:138-208 -> mpc_learned.LiveData mpc_learned.py:545-596):

| column | gz formula | same as training? |
|---|---|---|
| X_t | `inputs11(s_t, consts)` = inputs_of + zr (mpc_learned.py:524-530), normalised by ck stats | same formula as S[0:11] (episode6.py:156-157) |
| U_t | `(c_t - u_mu)/u_sd` | same |
| E_t | `e0_from(env, s_t, s_{t+1}, c_t)` (float64 states, float32 cast) / sigma_e (mpc_learned.py:584-591) | same function; training applies it to float32-packed XS (data2.py:407) |
| Wm_t | `sea_eval(sea, X_n(s_t), Y_n(s_t), t_t)` (mpc_constrained.py:181-183) | same sum as the twin's `_surface` |
| W_MID_t, PM_t | mid pose of `s_t`, `sea_eval(..., t_t + h)` (mpc_constrained.py:184-189) | same as episode5.py:70-77 / model_preview.py:193-195 |
| SEA | the mission's own components (mpc_constrained.py:167-172) | training: rebuilt episode sea (relabel.py:245-254) |
| valid | `k < n` (every pushed step) | training `k < len - 1` |

Scoring convention (onestep_gz.score_mission, onestep_gz.py:283-373):
`L = min(256, ctx) = 256`; windows `(0, 40, 256)` then `(hi - 256, lo, hi)` with
stride 16 (onestep_gz.py:117, 267-279): scored window positions 40..255 in the
first window, 240..255 in the later ones (`a = 16..224` for n = 480).
Variant p: `cfg = fixed_cfg(live, a + L, lam = 0.1, hp = 24, msd = 0.01, g)`
(onestep_gz.py:306-312, final_eval.py:105-107), i.e. moment beyond the
window: every token has `M = 1`, `sigma = 0`, `Pf = 1`, `lead = h`,
`lev = 0.1 sqrt(1 + 0.06^2) = 0.1002` [checked], preview = `W_MID + err`
(lam 0.1), sensor noise sd 0.01 on Wm; `obs = False`: no noise on the error
history. Scored set: `ok & lo <= pos < hi & isfinite(y)` (onestep_gz.py:322-323).

## 9. Every difference between gz tokens and training tokens

### 9a. Code paths: identical (no difference found)
Formulas of X, U, e0, Wm, W_MID, PM, the station geometry, the normalisation
statistics (the checkpoint's own), `fc_eval`, `level_feature`, the first flag
and the e_prev convention are the same functions or line-for-line equal
formulas (table in section 8). Numerical-precision difference only: training
e0 and PM come from float32 XS (data2.py:407; float32 ulp on psi ~2.4e-7 rad,
on velocities ~1e-6 m/s, i.e. < 1e-5 of sigma_e after division by Delta);
gz from float64.

### 9b. Convention differences (how the columns are filled)
1. History noise: training adds observation noise to every `Ep`
   (sd LogU[0.01, 0.1] x exp(0.3 N), section 5); gz scoring none (`obs=False`).
2. Sensor noise: training `msd ~ LogU[0.003, 0.03]`; gz fixed 0.01.
3. Preview/forecast: training `lam` = 0 w.p. 0.15 else LogU[0.02, 1.5],
   `hp` = 0 w.p. 0.2 else U{1..30}, moment uniform (rollout segments with
   `M = 0`, `sigma > 0`, lead > h appear in 0.8 of windows); gz: `lam = 0.1`,
   `hp = 24`, all tokens measured, lead h. The gz configuration is inside the
   trained set (share 0.20-0.25 of windows at positions 240-255, section 5).
4. Forecast-error field statistics: the same generator, but on a 120-component
   evenly spaced sea (gz) instead of a 384- (or 48-) component jittered sea:
   per-component phase noise `xi0, xi1` and the coherent shift act on fewer,
   regularly spaced components.
5. Scored positions: gz scores positions 240..255 (after the first window) with
   a window start in mid-mission and `f = 1` at position 0; training windows
   start at `a <= 103` of a 360-step episode (25 % at `a = 0`). Both have the
   first flag at window position 0; gz windows start up to step 224 of a
   480-step mission. Mission time is not an input, so absolute time beyond
   training's 86 s does not enter the tokens.
6. Episode start: training low-world `initial_state` sets `z = theta = 0` and
   `T_a = 0` at `u = 0.8 u_ref` (lofi.py:444-448), so the first tokens show a
   thrust actuator at 0 and a heave/pitch transient; gz starts at the running
   attitude with the calm-water thrust (planing_vessel.py:466-484). Only the
   first window (positions 40..255 scored) is affected.

### 9c. Distribution differences (same columns, different input law)
7. Sea: training per episode `Hs ~ U[0.5,1.5] m`, `Tp ~ U[4,7] s`, spreading
   `s ~ U[2,10]`, long-crested w.p. 0.25, direction `theta0 ~ U[0, 2 pi)`,
   48 jittered frequencies x 8 jittered directions (data2.py:95-103,
   step0_preview_spec.py:94ff); gz: `Hs = 1`, `Tp = 5`, `s = 8`, `gamma = 3.3`,
   `theta0 = pi`, 24 evenly spaced frequencies in [0.5, 3] w_p x 5 evenly spaced
   directions in `theta0 +- pi/2`, not jittered (task.py:56-59, 243-261;
   step0_preview_spec.py:59-92). The gz elevation at a fixed earth point
   repeats every `2 pi / dw = 46.0 s`, `dw = 2.5 (2 pi/5)/23 = 0.1366 rad/s`
   (shorter than the 61.4 s window; the moving hull samples it at moving
   points). Normalised Wm sd: train 1.064-1.073, gz 0.997-1.005 [checked].
8. Relative wave direction: training uniform; gz head seas (leg 0, heading 0,
   waves travelling toward psi = pi) and 45 deg off the bow (leg 1, heading
   pi/4) (task.py:68-69). gz psi in [-24, 69] deg [checked].
9. Commands: training PI speed control to targets 16-30 kn (changed every
   10-40 s), heading targets (changed every 20-60 s), OU dithers on thrust and
   nozzle (each w.p. 0.7), 25 % steady episodes (episode7.py:161-201); gz hand
   MPPI holding 25 kn on a straight track, autopilot nozzle, no dither. Thrust
   actuator fraction mean: train 0.606, gz 0.415 (max 0.84) [checked, from the
   normalised means].
10. Actuators: training M10 random family per channel per episode
    (w.p. 0.2 ideal, else delay U[0, 0.30] s x 0.735 rounded to 0.04 s steps,
    lag LogU, rate limit LogU, dead zone, backlash, gain/offset, 1st/2nd
    order; lofi.py:158-224), applied in the low plant; gz fixed waterjet:
    delay 0.1 s (5 steps of 0.02 s), dead band `t_max*50/12000`, clip
    `[-t_max/3, t_max]`, rate `t_max/0.3` per s, lag 0.6 s; nozzle: delay
    0.1 s, lag 0.1 s, rate 40 deg/s, 25 deg (planing_vessel.py:128-138,
    actuators.py:29-42, 186-196, waterjet.py:237-240). The gz values lie
    inside the family ranges except that a delay of exactly 0.1 s is not
    representable on the 0.04 s grid (0.08 or 0.12 s; lofi.py:207, 241-255).
11. Roll: gz has roll (`sd 4.3 deg`, `max |phi| 17.5 deg` [checked]); it is in
    no token column and not in the target (P drops indices 3, 9). In
    PlaningVesselGZ roll is one-way (does not drive heave, pitch, sway or yaw,
    planing_vessel_gz.py:61-66, 223-239), so its only paths into the tokens
    are SafeMode (never active here) and capsize termination.
12. Target magnitudes (std of `y` = e0/sigma_e; training = 1 by construction):
    gz surge 0.58, sway 0.50, yaw 0.58, heave 1.03, pitch 0.40, z 1.12,
    theta 0.51, psi 0.72, thrust gap 1.21, nozzle gap 1.58 [checked]; extreme
    gz values outside the training 0.1-99.9 % range for at most 0.3 % of steps
    per channel (nozzle gap: -17.3 .. 16.6 vs training -11.7 .. 11.2).
13. State features: no gz value outside the training 0.1-99.9 % range except
    heave rate `w` (0.9 % of gz steps) and `(z - z0)/0.2` (0.1 %) [checked];
    gz occupies a narrower band in speed, sway, pitch rate, theta.

### 9d. Same column, different physical meaning (dynamics, not token code)
14. Thrust actuator column `T_a/t_max`: in the low world the surge force is
    the actuator state itself (reduced model, lofi.py:527-529); in gz the jet
    force is `Waterjet.forces(T_a, delta_a, u)` with flow `prop. prime`, speed
    and `cos delta` dependence (waterjet.py:153-188) and the prime drops when
    the intake leaves the water (waterjet.py:208-217). The column value is the
    same kind of number; its map to force differs.
15. Wave information: the catalogue error law of training reads the 15
    W_MID elevations (`s28`, episode6.py:161) and, per episode7.py:23-26, its
    own substep samples of the sea; the gz plant integrates strip forces over
    its own section stations at every 0.02 s substep (planing_vessel.py:256-265).
    Whether the 30 wave columns are as informative about `e_k` in gz as in
    training is UNKNOWN from this analysis (a family question, not a token
    one).
16. Integrators: low world 6 x 0.04 s exact heave/pitch transitions; gz 12 x
    0.02 s Heun with stiff substeps (planing_vessel.py:522-551); model0 6 x
    0.04 s in both target computations.

### 9e. Not determined here
- Whether `z` of the identified reduced model and gz `z` (relative to `Z_rest`)
  share the same zero exactly: UNKNOWN (identification code not read); the
  normalised gz mean -0.19 vs train -0.04 suggests no large offset.
- The naming of `y_off = -B/2` as port vs starboard depends on the axis
  convention, not checked; the geometry above is exact.
- Online / Bayes / memory-net rows modify `h_t` (onestep_gz.py:313-318, 332-345);
  not covered.

## 10. Numeric check (wf_math/check_netio.py, read-only, first 400 training episodes = 140 473 steps; 16 gz missions x 480 steps)

    ckpt: variant p, ctx 256, arch default, n_wx 34, conv as section 5, fc {ph0 1, ph_rate 0.5, amp 0.3, tshift 0.2}
    env: dt 0.04, sub 6, dtc 0.24; gz t steps 0.24; gz sea 120 components, Hs 1.0016
    gz scoring token cols 54..57 unique: [1], [0], [1], [0.1002]
    STATE FEATURES (normalised)  train mean/sd/q0.1/q99.9          | gz mean/sd/min/max            | gz out
    u/uref-1       -1.273 1.865 -10.559  2.076 | -0.517 0.679 -3.960  1.147 | 0.000
    v               0.041 1.045  -4.724  4.876 |  0.039 0.436 -2.436  2.626 | 0.000
    rL/uref        -0.069 1.682 -10.820 11.025 | -0.033 1.136 -9.223 11.300 | 0.000
    cos psi         0.572 0.994  -1.286  1.477 |  1.273 0.207  0.587  1.477 | 0.000
    sin psi        -0.077 0.881  -1.548  1.376 |  0.426 0.523 -0.683  1.280 | 0.000
    thr act         1.873 1.929  -2.302  4.588 |  0.555 0.864 -2.160  3.487 | 0.000
    noz act         0.050 1.264  -2.319  2.477 |  0.047 0.689 -2.319  2.477 | 0.000
    w               -0.001 1.189 -5.042  6.830 | -0.000 1.898 -7.739  7.500 | 0.009
    q              -0.020 1.920 -10.421 10.810 |  0.002 0.921 -5.049  6.559 | 0.000
    (z-z0)/.2      -0.039 1.219  -5.004  5.257 | -0.188 1.412 -4.633  6.256 | 0.001
    (th-th0)/.05   -0.222 1.753  -9.913  7.880 | -0.223 0.821 -2.848  2.345 | 0.000
    U thr           1.797 1.835  -2.163  4.305 |  0.525 0.922 -2.163  3.641 | 0.000
    U noz           0.049 1.277  -2.297  2.451 |  0.046 0.746 -2.297  2.451 | 0.000
    TARGET y std: train ~1 | gz 0.580 0.497 0.576 1.028 0.397 1.115 0.513 0.718 1.208 1.583
    Wm sd train 1.064-1.073, gz 0.997-1.005; bow port-stbd diff sd train 0.375 gz 0.332; bow-stern diff sd train 0.821 gz 0.939
    gz roll max 17.5 deg, sd 4.3 deg; gz psi -24..69 deg; SafeMode steps 0 in all missions
