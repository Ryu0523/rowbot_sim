# Simulator data ledger (2026-10-08)

> 这是什么：把仓库里对"船的仿真器"（滑行艇被控对象 v0 / as1）和"势流求解器"（Rankine）做过的每一次实验和检查，只按原始数字列出来的清单。
> 只记：跑的是什么（哪版代码、哪条船、什么航速和海况）、量了什么、得到的数、拿来比的参考值和它的来源类型、数字在哪个文件。
> 不记：任何解释、结论、诊断、建议；文档里"为什么"的部分全部删掉，只留数字和设置。
> 参考值本身不确定的（读图、转抄、二手来源、假设的周期）单独标出；同一个量两处数字不一致的，在 G 节列出。
> 用途：在不受以前结论影响的前提下，从原始数据重新设计。

Paths are relative to `studies/_cache/` unless they start with a top-level folder. `file:N` = line N of that file (section anchor);
for large tables the row key (speed, case) is given instead. "tol" = the threshold that was registered in that run file;
it is given so the reader can compare, not as a verdict. Numbers were copied from result files and logs; where the only
source is a document, the row says "doc only".

## 0. Legend (code versions, hulls, switches)

| name | meaning |
|---|---|
| v0 | `sim/planing_vessel.py` PlaningVessel / `sim/planing_vessel_gz.py` PlaningVesselGZ (GZ roll), 2D+t strips, 48 stations, Heun dt 0.02 s. Coefficients in the logs below: k_a 1.0, C_dc 0.9397 (= cos 20 deg), a_b 0.5 (planing), no transom correction, roll zeta 0.1 |
| v0 (older) | same code with k_a 0.6, C_dc 1.0, a_b 0.5 (the H6 set); its numbers exist only in IL H6 (doc only) |
| as1 | `sim/planing_anyspeed.py`, switch sets below. "P1a preset" = dDt_full, chine_exact_all, kpp_strip, transom, normal_rate, verlet. "as1 preset" = P1a + fk_exact + fast_path. Extra switches seen: orbital, wagner_entry, lateral_phi0, heel_hydro ("hh"), heel_hydro+lateral_phi0+lateral_cross ("hx") |
| "as1" in s1/* | the S1 version (complete D/Dt(m_a V) term) compared with v0, before the P1a preset existed |
| scarab195 | L 5.4 m, chine beam b 2.025 m, deadrise 20 deg, mass 1400 kg (test loads 1340 / 1490 kg used where stated), V 1.366 m3, GM 1.216 m, Ix 703 kg m2, Iz 2552 kg m2, Iy 2552 kg m2, top-speed thrust map from `sim/waterjet.py` (t_max 6366 N) |
| planing_b | second parametric hull: L 6.6 m, B 2.0 m, 1900 kg, T 0.396 m, GM 1.32 m (p0/hullb.txt) |
| fridsma_A/B | Fridsma (1969) prismatic model: b 9 in, L/b 5, deadrise 20 deg, 7.26 kg fresh water, VCG 0.294 b; case A V/sqrt(L) 4 (3.98 m/s model), case B V/sqrt(L) 6 (5.98 m/s) |
| task sea / eval sea | irregular sea ("sea state 3") Hs 1.0 m, Tp 5 s; leg 0 head seas, leg 1 45 deg off the bow (final_eval.make_mission) |
| scan seas | tp5 = Hs 1.0 m Tp 5 s; tp8 = Hs 1.0 m Tp 8 s (studies/wave_regime_scan.py:64). Relative heading chi: 0 head ... 180 following |
| IL | the repository issue log `DEFECTS.md` (repo root, Chinese); section ids such as H6, H7, F8, G4, G6, K refer to its headings. Numbers taken only from it are marked "doc only" |
| Rankine | `hydro/rankine/*`, stages B2-B8 of learn/meta/RANKINE_STAGE1_2026-10-06.md; final tier files: rankine/verify_full.json (B3), hull_verify_full.json (B4), unsteady_verify_full.json (B5), db_verify_full.json (B6), planing_gate_full.log (B7), planing_b8_full.log + planing_b8_posttier.log (B8) |

---------------------------------------------------------------------------------------------------------------------

## A. Comparisons with physical measurements or with regressions of measurements

### A1. Calm-water running trim and resistance vs Savitsky (1964) — regression of towing-tank data

| id | run | quantity | values (plant) | reference (Savitsky) | file |
|---|---|---|---|---|---|
| A1.1 | v0 scarab195, 20/25/30/35/40 kn | trim deg / R kN / rise m | 3.51/1.81/0.149; 2.36/2.14/0.154; 1.71/2.73/0.151; 1.54/3.47/0.154; 1.43/3.98/0.168 | trim 4.34/3.41/2.75/2.29/1.96; R 2.12/2.41/2.87/3.47/4.20 | anyspeed/tests/studies.exp_planing.log:3 |
| A1.2 | same, summary line | trim error range; R error range | -1.04..-0.52 deg; -15..+0 % | — | studies.exp_planing.log:10 |
| A1.3 | v0 vs as1 (P1a preset), scarab195, 20-40 kn | R N (err %), trim (err deg) | 20 kn v0 1813 (-14.5) 3.51 (-0.84), as1 1811 (-14.7) 3.45 (-0.89); 25: v0 2139 (-11.2) 2.36 (-1.04), as1 2116 (-12.1) 2.33 (-1.07); 30: v0 2733 (-4.6) 1.71 (-1.04), as1 2553 (-10.9) 1.87 (-0.88); 35: v0 3472 (+0.1) 1.54 (-0.74), as1 2947 (-15.0) 1.70 (-0.59); 40: v0 3981 (-5.2) 1.43 (-0.52), as1 3362 (-20.0) 1.57 (-0.38) | R 2122/2408/2866/3469/4201 N; trim 4.34/3.41/2.75/2.29/1.96; tol R ±10 %, trim ±1 deg | anyspeed/p1a/review_fix/v1.txt (identical rows in p1b/v1.txt) |
| A1.4 | v0, scarab195, 3-35 kn (transom audit; 1-9 kn = snapshot of a limit cycle) | trim, R_plant N | 3 kn 2.16/483; 5: 2.32/591; 7: 1.90/768; 9: 2.41/1114; 10: 5.45/1681; 12: 6.38/1965; 14: 5.78/1909; 16: 4.76/1798; 20: 3.51/1813; 25: 2.36/2139; 30: 1.71/2733; 35: 1.54/3472 | Savitsky trim / R (validity flag): 3 kn 3.22/831 (outside); 5: 3.66/1021 (outside); 7: 4.33/1280; 9: 5.14/1578; 10: 5.50/1714; 12: 5.84/1899; 14: 5.69/1979; 16: 5.28/2017; 20: 4.34/2122; 25: 3.41/2408; 30: 2.75/2866; 35: 2.29/3469 | anyspeed/audit_transom.txt |
| A1.5 | as1 planing_b, Fn 1.1/1.4/1.767/2.061 (17.2/21.9/27.6/32.2 kn) | trim deg, R N | 5.695/2716; 4.026/2602; 2.757/2903; 2.203/3408 | 5.416/2660; 4.329/2736; 3.291/3062; 2.705/3504 | anyspeed/p0/hullb.txt |
| A1.6 | v0 (older, k_a 0.6) | trim err, R err, 20-40 kn | -0.6..+0.4 deg; -3..+11 % (coefficients selected on a grid against Savitsky) | Savitsky | doc only: IL H6 (line ~1029) |
| A1.7 | v0 (Zarnick coefficients) | trim, R vs Savitsky | trim 0.5-1.0 deg below; R 0-15 % below | Savitsky | doc only: IL H7 |

### A2. Scarab 195 top speed and part-throttle speed — measured (Boating magazine tests); flag: magazine numbers, loads as quoted

| id | run | quantity | plant | measured | file |
|---|---|---|---|---|---|
| A2.1 | v0 (Zarnick coeff.), full thrust, calm, 1340 kg / 1490 kg | top speed kn | 45.1 / 43.3 (+5.7 % / +3.0 %) | 42.7 (2019) / 42.06 (2024) | studies.exp_planing.log:12 |
| A2.2 | jet model + Savitsky + air drag, Monte Carlo over assumed ranges (chine 0.78-0.88 B, CG 0.34-0.42 L, nozzle 80-90 mm, pump eff 0.70-0.85, windage 1.2-2.2 m2) | top speed median (80 % band) | 1490 kg 7920 rpm: 41.8 (40.1-43.6); 1339 kg 8000 rpm: 42.2 (40.5-44.0) | 42.1 / 42.7 | anyspeed/tests/studies.exp_waterjet.log:31 |
| A2.3 | same, part throttle (power = fuel flow / assumed BSFC 0.29-0.40 kg/kWh) | speed kn band | 6000 rpm 23.7-28.8; 6500: 26.7-31.6; 7000: 30.1-35.0; 7500: 33.7-38.7; 7920: 36.6-41.9 | 22.3; 28.2; 32.7; 37.8; 42.1 | studies.exp_waterjet.log:36 |
| A2.4 | v0 1400 kg, 48/96/192 stations | top speed kn | 43.88 / 44.35 / 44.34 | — | anyspeed/s1/d_planing.txt (top speed lines) |
| A2.5 | as1 (S1) 1400 kg, 48/96/192 stations | top speed kn | 47.66 / 47.67 / 47.68 (+8.6/+7.5/+7.5 % vs v0) | — | anyspeed/s1/d_planing.txt |
| A2.6 | v0 (older k_a 0.6) | top speed kn | 40.6-40.7 | 42.1-42.7 | doc only: IL H6 |
| A2.7 | VM 18 (brochure: 5.77 m, beam 1.67 m, wet 1206 kg, 300 hp; "50+ kn") — prediction only | top speed median (80 %) | wet 45.9 (44.3-47.6); +453 kg 44.8 (43.1-46.5); 1815 kg 44.3 (42.6-46.1); 20 kn shaft power 43 kW (40-47) | brochure claim 50+ kn, 20 kn for 26 h | studies.exp_waterjet.log:42 |

### A3. Fridsma (1969) regular head waves, towed (speed held), H/b = 0.111 — measured towing-tank table
Flag: the Fridsma table values are a transcription typed into `studies/exp_planing.py` (FRIDSMA) from a web-retrieved source ("agent's transcription"); the Zarnick (1978) column is digitised (±0.03 / ±0.02 g per the code comment). Columns: heave h/H, pitch ratio, CG g, bow g.
Test set-up per the report itself (DL 1275, read 2026-10-08 from the Internet Archive OCR text): constant horizontal force at the CG (pitch pivot at the CG), carriage free to surge; accelerometers at the CG and 10 % L aft of the stem (bow: normal to the keel); reported acceleration = upward peak from the still-water zero, averaged over 10 consecutive cycles.

**Recomputed 2026-10-08 after the harness fix** (towing force through the CG, horizontal; friction where the plant puts it; speed held; `studies/exp_sim_redesign2.py`, `studies/_cache/sim_redesign/step3.json`, step3_report.txt). ZR = `sim/planing_zarnick.py` ZOPT_ZR (friction also through the CG, Zarnick's assumption); ZR+MLM = ZR with the `mlm` switch.

| case, lam/L | v0 | as1 preset | ZR | ZR+MLM | measured | Zarnick 1978 computed (h/H, pitch) |
|---|---|---|---|---|---|---|
| A 1.0 | 0.25, 0.15, 0.41, 1.47 | 0.24, 0.15, 0.41, 1.50 | 0.26, 0.15, 0.44, 1.50 | 0.19, 0.13, 0.28, 1.13 | 0.18, 0.11, 0.25, 1.05 | 0.12, 0.15 |
| A 2.0 | 0.65, 0.62, 0.26, 0.80 | 0.65, 0.60, 0.24, 0.76 | 0.67, 0.63, 0.27, 0.83 | 0.61, 0.59, 0.20, 0.73 | 0.84, 0.79, 0.25, 0.90 | 0.70, 0.63 |
| A 3.0 | 1.04, 1.11, 0.18, 0.42 | 1.02, 1.07, 0.17, 0.37 | 1.10, 1.16, 0.19, 0.44 | 1.07, 1.11, 0.18, 0.39 | 1.18, 1.29, 0.20, 0.45 | 1.08, 1.12 |
| A 4.0 | 1.19, 1.31, 0.13, 0.23 | 1.15, 1.24, 0.12, 0.19 | 1.21, 1.34, 0.14, 0.24 | 1.19, 1.25, 0.13, 0.21 | 1.23, 1.21, 0.15, 0.25 | 1.18, 1.31 |
| A 6.0 | 1.15, 1.24, 0.07, 0.10 | 1.14, 1.24, 0.06, 0.08 | 1.15, 1.27, 0.08, 0.10 | 1.14, 1.20, 0.07, 0.09 | 1.04, 1.13, 0.10, 0.15 | 1.16, 1.29 |
| B 1.0 | 0.24, 0.10, 0.64, 2.09 | 0.24, 0.10, 0.64, 2.11 | 0.24, 0.10, 0.65, 2.07 | 0.20, 0.09, 0.59, 1.81 | 0.16, 0.06, 0.50, 1.45 | 0.23, 0.11 |
| B 1.5 | 0.41, 0.26, 0.79, 2.53 | 0.41, 0.26, 0.78, 2.53 | 0.41, 0.26, 0.79, 2.49 | 0.38, 0.27, 0.62, 1.96 | 0.37, 0.21, 0.70, 2.00 | 0.41, 0.26 |
| B 2.0 | 0.62, 0.53, 0.74, 2.50 | 0.62, 0.53, 0.76, 2.55 | 0.62, 0.54, 0.79, 2.57 | 0.63, 0.57, 0.63, 2.09 | 0.64, 0.54, 0.80, 2.55 | 0.60, 0.53 |
| B 3.0 | 1.23, 1.46, 0.60, 2.27 | 1.25, 1.48, 0.62, 2.34 | 1.26, 1.53, 0.72, 2.66 | 1.40, 1.60, 0.57, 2.21 | 1.45, 1.68, 0.90, 2.85 | 1.33, 1.55 |
| B 4.0 | 1.69, 2.23, 0.27, 0.89 | 1.65, 2.17, 0.26, 0.85 | 1.80, 2.43, 0.29, 1.07 | 1.72, 2.09, 0.30, 0.73 | 1.75, 2.39, 0.35, 1.25 | 1.78, 2.32 |
| B 6.0 | 1.25, 1.55, 0.12, 0.15 | 1.25, 1.61, 0.11, 0.14 | 1.27, 1.70, 0.11, 0.15 | 1.28, 1.61, 0.12, 0.16 | 1.08, 1.51, 0.15, 0.25 | 1.25, 1.62 |

Counts (amplitudes ±20 %, accelerations ±30 %; of which accelerations / 22; vs Zarnick's computed column |d| <= 0.10 Z + 0.03, of 22; resonant lam/L 3-4, of 8): v0 30/44 (15/22), 21/22, 8/8; as1 29/44 (14/22), 21/22, 8/8; ZR 32/44 (17/22), 21/22, 8/8; ZR+MLM 35/44 (18/22), 21/22, 8/8.
Same runs with the carriage free to surge under a constant tow force (Fridsma's set-up; mean speed within 1 % of target): ZR 31/44, ZR+MLM 34/44; largest change of any value vs speed held 0.12 (pitch ratio, A 6.0 ZR; 0.11 at B 6.0 ZR+MLM), accelerations change by at most 0.08 g (bow, B 4.0 ZR).
Check H1: fixed-harness v0 equals step 1's "v0 + tow through the CG" switch run to the bit (max difference 0, cases A and B, lam/L 3, 2000 steps).

Superseded (harness applied the towing force at the default jet nozzle, transom, 0.083 m = 0.36 b above the CG; values kept for the record):

| case, lam/L | v0 (Zarnick coeff.) | as1 P1a preset (review_fix) | as1 (p1b candidate) |
|---|---|---|---|
| A 1.0 | 0.24, 0.14, 0.37, 1.42 | 0.22, 0.14, 0.36, 1.39 | 0.22, 0.14, 0.37, 1.42 |
| A 2.0 | 0.61, 0.58, 0.23, 0.73 | 0.59, 0.56, 0.21, 0.69 | 0.61, 0.57, 0.22, 0.71 |
| A 3.0 | 0.95, 0.99, 0.17, 0.37 | 0.94, 0.98, 0.16, 0.33 | 0.97, 1.00, 0.16, 0.34 |
| A 4.0 | 1.10, 1.18, 0.12, 0.21 | 1.09, 1.16, 0.12, 0.18 | 1.12, 1.18, 0.12, 0.18 |
| A 6.0 | 1.10, 1.19, 0.07, 0.10 | 1.11, 1.20, 0.06, 0.08 | 1.14, 1.22, 0.06, 0.08 |
| B 1.0 | 0.23, 0.10, 0.58, 2.11 | 0.23, 0.10, 0.59, 2.11 | 0.23, 0.10, 0.60, 2.12 |
| B 1.5 | 0.41, 0.27, 0.71, 2.47 | 0.40, 0.26, 0.72, 2.48 | 0.40, 0.26, 0.73, 2.52 |
| B 2.0 | 0.60, 0.52, 0.64, 2.22 | 0.60, 0.52, 0.66, 2.29 | 0.61, 0.53, 0.68, 2.36 |
| B 3.0 | 1.14, 1.30, 0.45, 1.67 | 1.15, 1.30, 0.44, 1.66 | 1.17, 1.33, 0.46, 1.74 |
| B 4.0 | 1.50, 1.89, 0.25, 0.68 | 1.48, 1.84, 0.24, 0.66 | 1.51, 1.90, 0.24, 0.69 |
| B 6.0 | 1.23, 1.48, 0.12, 0.15 | 1.24, 1.55, 0.11, 0.14 | 1.27, 1.61, 0.11, 0.14 |

Sources (superseded): v0 = anyspeed/tests/studies.exp_planing.log:36; as1 = anyspeed/p1a/review_fix/v1.txt, anyspeed/p1b/v1.txt.
Counts (superseded): v0 25/44, as1 P1a 22/44, p1b candidate 24/44; attribution by switch set: pre 24/44, +transom 24/44, +transom+normal_rate 22/44, +verlet only 24/44 (anyspeed/p1a/review_fix/checks_p6attr.json).
S1 version vs v0 on the same cases: largest |as1/v0 - 1| over heave and pitch 2.75 % (anyspeed/s1/d_planing.txt, "Fridsma P6").
Older coefficients (k_a 0.6, doc only, IL H7): case A calm run self-oscillating; heave 2-12x the measured values.

### A4. Fridsma calm-water attitude (same model, smooth water)

**Recomputed 2026-10-08 after the harness fix** (towing force through the CG; sim_redesign/step3.json): trim deg, rise/b, R/Delta.

| case | v0 | as1 preset | ZR | ZR+MLM | measured |
|---|---|---|---|---|---|
| A (3.98 m/s) | 5.73, 0.044, 0.195 | 5.67, 0.043, 0.194 | 5.87, 0.044, 0.196 | 5.80, 0.003, 0.208 | 4.0, 0.05, 0.158 |
| B (5.98 m/s) | 4.06, 0.129, 0.215 | 3.94, 0.129, 0.213 | 3.99, 0.133, 0.210 | 4.54, 0.117, 0.230 | 4.0, 0.13, 0.206 |

Superseded (tow at the nozzle): v0 A 5.50 deg, 0.028 b, 0.196; B 3.96, 0.121, 0.219. as1 (P1a) A 5.43, 0.031, 0.194; B 3.87, 0.121, 0.218 (exp_planing.log:38; p1a/review_fix/v1.txt).

### A5. Fridsma (1971) irregular head seas, towed — mean of all positive acceleration peaks (g)
Flag: Fridsma's modal period is not in the repo's copy; the sea is assumed fully developed P-M, Tp = 15.63 sqrt(Hs/g), and also run with Tp x0.8 and x1.25 (exp_planing.py:262). Measured values typed in `studies/exp_planing.py:256`.

Report itself (DL 1495, read 2026-10-08, Internet Archive OCR): "all positive peak accelerations, including those that are wave-induced as well as impact spikes", hand-read from oscillograph records, >= 75 encounters per condition; accelerometers at the CG and 10 % L aft of the stem (bow normal to the keel); P-M spectra, modal period not stated in the text; constant speed, thrust axis horizontal through the CG; gyradius 25 % L. Table II / V condition numbers agree with the repo after aligning a two-row offset in the OCR.

**Recomputed 2026-10-08 after the harness fix** (tow through the CG; sim_redesign/step3.json, step3_seeds.json). Statistic as before: one peak per positive excursion, mean. Nominal Tp, wave seed 0, CG / bow g [CG at Tp x0.8 - x1.25]; then the mean over 4 wave seeds.

| cond (V/sqrt L, H1/3/b) | v0 | as1 preset | ZR | ZR+MLM | measured CG / bow |
|---|---|---|---|---|---|
| 22 (4, 0.222) | 0.33/1.07 [0.28-0.33] | 0.35/1.16 [0.30-0.36] | 0.35/1.10 [0.30-0.35] | 0.24/0.77 [0.20-0.24] | 0.45 / 1.65 |
| 21 (4, 0.444) | 0.52/1.71 [0.36-0.43] | 0.72/2.33 [0.65-0.58] | 0.57/1.82 [0.44-0.49] | 0.32/1.20 [0.17-0.36] | 0.71 / 2.35 |
| 23 (4, 0.667) | 0.91/2.92 [0.57-0.61] | 1.09/3.37 [1.02-0.73] | 1.05/3.15 [0.64-0.70] | 0.86/2.60 [0.54-0.71] | 1.05 / 3.09 |
| 31 (6, 0.222) | 0.52/1.55 [0.40-0.56] | 0.56/1.77 [0.43-0.67] | 0.53/1.60 [0.42-0.62] | 0.53/1.52 [0.42-0.58] | 0.68 / 2.10 |
| 30 (6, 0.444) | 1.04/3.03 [0.77-1.30] | 1.32/3.92 [1.07-1.44] | 1.17/3.20 [0.80-1.46] | 1.11/2.95 [0.71-1.37] | 1.77 / 5.33 |
| within ±30 %, seed 0 / seeds 1, 2, 3 | 7 / 6, 5, 5 | 10 / 10, 9, 9 | 7 / 7, 9, 7 | 4 / 2, 3, 3 | |
| median of sim/measured over the 10 items, seed-mean | 0.71 | 0.81 | 0.77 | 0.61 | |

Spread over the 4 seeds of one condition: up to 0.23 g on the CG (ZR cond 30: 1.13-1.36) and 0.70 g at the bow (ZR+MLM cond 23: 1.90-2.60). Time step 0.001 s instead of 0.004 (3 seeds): median ratio ZR 0.77 -> 0.72, ZR+MLM 0.61 -> 0.58. Alternative statistic (every local maximum above zero) / one peak per excursion, nominal Tp: 0.52-0.88 (all configurations).

Superseded (tow at the nozzle):

| cond (V/sqrt L, H1/3/b) | v0 CG (Tp x0.8 / 1 / 1.25) | v0 bow | as1 P1a CG / bow (nominal Tp) | as1 p1b CG / bow | measured CG / bow |
|---|---|---|---|---|---|
| 22 (4, 0.222) | 0.24/0.27/0.28 | 0.80/0.92/0.90 | 0.30 / 1.04 | 0.31 / 0.98 | 0.45 / 1.65 |
| 21 (4, 0.444) | 0.18/0.29/0.34 | 0.59/1.08/1.18 | 0.59 / 2.11 | 0.59 / 1.96 | 0.71 / 2.35 |
| 23 (4, 0.667) | 0.46/0.61/0.51 | 1.56/2.03/1.61 | 0.93 / 2.93 | 0.94 / 2.85 | 1.05 / 3.09 |
| 31 (6, 0.222) | 0.37/0.48/0.51 | 1.12/1.47/1.66 | 0.52 / 1.67 | 0.51 / 1.52 | 0.68 / 2.10 |
| 30 (6, 0.444) | 0.63/0.87/0.98 | 1.93/2.69/3.03 | 1.12 / 3.50 | 1.15 / 3.20 | 1.77 / 5.33 |

Sources (superseded): exp_planing.log:57; p1a/review_fix/v1.txt; p1b/v1.txt. Counts within ±30 %: as1 6 of 10 in both.

### A6. Slamming force on a 30 deg wedge, Zhao, Faltinsen & Aarsnes drop test (0.2 m measuring section) — measured, digitised
Flag: measured curve digitised by eye from the paper's Figure 15 (±250 N, ±0.05 m/s; studies/exp_slam_wedge.py:48). Digitisation cross-check: rig deceleration x mass apportioned to the section gives 4232 N vs digitised 4500 N at 10 ms (+6 %), 4618 vs 5125 N at 15 ms (+11 %).

| t ms | measured N | nonlinear 2D (paper) | model now | model before | von Karman | now + m_a dV/dt |
|---|---|---|---|---|---|---|
| 2.5 | 1400 | 1400 | 2694 | 2694 | 1092 | 2686 |
| 5.0 | 2800 | 2850 | 5267 | 5267 | 2135 | 5193 |
| 7.5 | 3900 | 4200 | 7575 | 7575 | 3070 | 7303 |
| 10 | 4500 | 5500 | 9452 | 9452 | 3831 | 8790 |
| 12.5 | 4900 | 6200 | 10808 | 10809 | 4381 | 9642 |
| 15 | 5125 | 6600 | 11755 | 11755 | 4764 | 10195 |
| 17.5 | 3700 | 5000 | 0 | 12511 | 5071 | -1296 |
| 20 | 2600 | 3500 | 0 | 13378 | 5422 | -825 |
| 25 | 1900 | 2800 | 0 | 15509 | 6286 | -314 |

Ratios to measurement 2-7.5 ms: model now 1.91, before 1.91, von Karman 0.78, now+m_a dV/dt 1.89, nonlinear 2D 1.03. Peaks: model now 11813 N at 15.2 ms; before 15509 at 25 ms; von Karman 6286 at 25 ms; measured 5150 N at ~15 ms. Model force end at 15.8 ms. Free drop of the 241 kg rig under the model: V at 10/15/20/25 ms 5.30/4.52/4.26/4.31 m/s vs measured 5.80/5.35/5.02/4.90; section force 7652/7921/0/0 N vs 4500/5125/2600/1900. File: anyspeed/tests/studies.exp_slam_wedge.log.

### A7. Water-jet and Savitsky-code checks against published data

| id | quantity | model | reference (type) | file |
|---|---|---|---|---|
| A7.1 | HamiltonJet 291 thrust vs speed, 8 power levels; fit on 200 hp (nozzle 159 mm, pump eff 0.80) | rms err per curve 1.5-2.6 %, worst 5.7 % over 48 predicted points; static thrust 50/100/150/200 hp: 420/667/875/1059 kgf | maker chart (reproduced in MacPherson 2000), digitised by eye ±10 kgf: 408/654/847/1036 kgf | studies.exp_waterjet.log:2 |
| A7.2 | static thrust ratios 100/50, 200/100, 200/50 hp | momentum theory 1.587, 1.587, 2.520 | chart 1.603, 1.584, 2.539 | same |
| A7.3 | Savitsky code vs Alourdas worked example, NACA Model 29, 18.01 kn | lam 1.59, trim 4.26, drag 14.80 lb | published Savitsky 1.59/4.21/15.12; tank 2.31/4.00/15.20 | studies.exp_waterjet.log:20 |
| A7.4 | same, 20.92 kn | 1.56, 3.50, 16.38 lb | published 1.56/3.46/16.76; tank 1.81/4.00/16.00 | same |

### A8. Rankine solver vs Wigley measurements (University of Tokyo, Kajitani et al. 1983) — measured; flags per row
References: (a) U.T. curves read from vector drawing commands of Yang et al. 2000 figs 4a/5a (our transcription); (b) U.T. C_WP markers read from a scan of Hendrix & Noblesse 1988 p. 61 "by rank" (our reading, stated reading error); (c) WaveBEM `wavebem_wigleyDragExp*.csv` (WaveBEM authors' digitisation; flagged in the B4 review as not reproducing the scan markers at Fn < 0.3); (d) U.T. wave profiles from WaveBEM `tokyoFr*.csv`.

| id | stage, model | Fn | value | reference | file |
|---|---|---|---|---|---|
| A8.1 | B4 fixed, hull force C_p (d'Alembert-corrected pressure) | 0.20/0.25/0.267/0.289/0.316/0.354/0.408 | 0.631/0.946/0.891/1.348/1.431/1.246/2.145 e-3 | (a) C_W fixed 0.414/0.821/0.824/1.181/1.525/1.430/2.311 e-3 | rankine/hull_verify_full.json rows V1a |
| A8.2 | B4 fixed, wave energy flux C_near | 0.25/0.267/0.289/0.316/0.354/0.408 | 0.710/0.647/1.073/1.108/0.929/1.905 e-3 | (b) C_WP 0.721/0.664/0.992/1.503/1.137/2.077 e-3 | same, rows V1b |
| A8.3 | B4 fixed, C_near vs (c) | 0.20-0.408 | ratios 0.680, 0.740, 0.525, 0.732, 0.776, 0.621 | (c) | same, rows V1 |
| A8.4 | B4 wave profile along hull, RMS(calc-exp)/max|exp| | 0.25/0.267/0.289/0.316/0.354/0.408 | 0.173/0.163/0.175/0.211/0.252/0.333; bow crest 2g zeta/U^2 calc 0.302/0.282/0.268/0.249/0.236/0.225 | exp 0.396/0.358/0.348/0.329/0.308/0.291 (d) | same, rows V2 |
| A8.5 | B4 free model sinkage/L | 0.25/0.267/0.289/0.316/0.354/0.374/0.408 | 1.342/1.574/1.918/2.271/3.065/3.654/4.585 e-3 | (a) 1.412/1.625/2.046/2.476/3.346/4.076/5.335 e-3 | rows V3 |
| A8.6 | B4 free model trim sin(theta) (bow up +) | same Fn | 3.54/6.83/14.79/15.12/37.53/71.86/147 e-4 | (a) -3.59/-2.77/0.83/-5.06/0.40/28.8/112.4 e-4 (differences +0.041..+0.247 deg) | rows V4 |
| A8.7 | B4 free C_p | 0.25-0.408 | 0.980/0.903/1.475/1.515/1.370/1.840/2.947 e-3 | (a) free C_W 0.815/0.837/1.297/1.751/1.636/1.967/3.010 e-3 | rows V5a |
| A8.8 | B6 double-body, fixed, C_lin | 0.25/0.267/0.289/0.316/0.354/0.408 | 0.998/0.907/1.434/1.545/1.329/2.284 e-3 | (a) 0.821/0.824/1.181/1.525/1.430/2.311 e-3 | rankine/db_verify_full.json rows SF1a |
| A8.9 | B6 wave cut C_cut | same | 1.039/0.941/1.434/1.578/1.357/2.184 e-3 | (b) 0.721/0.664/0.992/1.503/1.137/2.077 e-3 | rows SF1b |
| A8.10 | B6 wave profile RMS err | same | 0.104/0.093/0.140/0.178/0.198/0.254 | (d) | rows SF2 |
| A8.11 | B6 free sinkage/L, trim, C_lin | 0.289/0.316/0.354 | sink 1.968/2.336/3.164 e-3; trim 7.6/2.3/11.2 e-4; C 1.540/1.674/1.464 e-3 | (a) 2.046/2.476/3.346 e-3; 0.83/-5.06/0.40 e-4; 1.297/1.751/1.636 e-3 | rows SF3-SF5a |

### A9. Rankine solver vs Journee (1992) Wigley I forward-speed data — measured, SECONDARY transcription
Flag: original report not retrieved; markers read from vector figures of J. Mar. Sci. Eng. 9:87 (2021) (Fn 0.2) and Kara 2022 (Fn 0.3); how accurately those authors placed Journee's values is unknown. Values: computed / measured (ratio).

| quantity, Fn 0.2 | B5 D-form | B5 OT-form | B6 double body | file |
|---|---|---|---|---|
| A33' at omega' 1.676 / 2.222 / 2.832 / 3.370 / 3.883 / 4.439 / 5.542 (meas 1.632/0.513/0.499/0.435/0.440/0.454/0.531) | 0.809/0.635/0.509/0.453/0.442/0.459/0.517 | 0.910/0.713/0.555/0.477/0.452/0.463/0.518 | 0.846/0.632/0.507/0.459/0.452/0.470/0.527 | unsteady_verify_full.json V1/V1o; db_verify_full.json Vd |
| B33' at 1.622 / 2.186 / 2.775 / 3.315 / 3.820 / 4.408 / 5.495 (meas 1.119/1.543/1.479/1.283/1.039/0.803/0.504) | 1.384/1.443/1.407/1.238/1.009/0.735/0.376 | 1.205/1.400/1.444/1.305/1.075/0.787/0.412 | 1.689/1.557/1.423/1.216/0.979/0.711/0.364 | same |
| A55' at 1.676 / 2.148 / 2.803 / 3.358 / 3.877 / 4.445 / 5.532 (meas 0.0188/0.0245/0.0203/0.0206/0.0203/0.0203/0.0206) | 0.0241/0.0204/0.0194/0.0183/0.0169/0.0156/0.0155 | 0.0318/0.0249/0.0228/0.0213/0.0192/0.0170/0.0159 | 0.0336/0.0248/0.0209/0.0188/0.0173/0.0162/0.0162 | same |
| B55' at 2.203 / 2.734 / 3.308 / 3.870 / 4.425 / 5.518 (meas 0.0555/0.0505/0.0482/0.0470/0.0410/0.0302) | 0.0432/0.0396/0.0399/0.0396/0.0357/0.0215 | 0.0447/0.0397/0.0414/0.0436/0.0412/0.0257 | 0.0528/0.0462/0.0433/0.0401/0.0348/0.0211 | same |
| F3' at kL 3.174/3.610/4.236/5.113/6.265/8.389 (meas 10.59/9.743/7.648/5.178/2.589/1.245) | 10.95/9.634/7.777/5.340/2.613/0.949 | 10.56/9.259/7.458/5.149/2.645/1.007 | 11.01/9.719/7.894/5.496/2.819/1.117 | same |
| F5' at kL 3.182/3.585/4.215/5.068/6.288/8.409 (meas 1.989/2.174/2.157/2.026/1.772/0.900) | 2.094/2.167/2.204/2.118/1.779/0.879 | 2.119/2.163/2.158/2.030/1.670/0.838 | 2.141/2.215/2.253/2.172/1.841/0.959 | same |
| RAO3 at lam/L 1.25/1.4/1.5/1.6/1.75/2.0 (meas 0.858/0.888/0.958/0.958/0.968/0.988; KG 0.170 m assumed) | 1.042/0.920/0.921/0.930/0.943/0.960 | 0.864/0.888/0.918/0.938/0.958/0.975 | 0.817/0.877/0.910/0.932/0.954/0.976 | same |
| RAO5 same lam/L (meas 1.645/1.235/1.126/0.944/1.085/0.675) | 1.660/1.193/1.016/0.892/0.762/0.622 | 1.750/1.261/1.063/0.925/0.782/0.632 | 1.564/1.152/0.988/0.872/0.748/0.613 | same |
| RAO5 KG sensitivity (D-form): KG 0.120 m / 0.220 m | 1.580/1.156/0.992/0.876/0.751/0.616 ; 1.751/1.233/1.042/0.910/0.773/0.628 | | | unsteady_verify_full.json V4 info |
| Fn 0.3 A33' at 1.659/2.209/2.769/3.317/3.867/4.417/5.527 (meas 0.742/0.545/0.478/0.463/0.453/0.469/0.533) | 0.726/0.609/0.523/0.474/0.457/0.465/0.516 | 0.893/0.737/0.611/0.527/0.484/0.476/0.516 | 0.854/0.646/0.541/0.493/0.480/0.490/0.538 | same (V2/V2o/Vd) |
| Fn 0.3 B33' at 1.657/2.208/2.769/3.317/3.868/4.418/5.527 (meas 1.693/1.662/1.436/1.259/1.081/0.884/0.567) | 1.229/1.304/1.288/1.169/0.972/0.747/0.367 | 1.035/1.244/1.333/1.276/1.100/0.863/0.438 | 1.599/1.471/1.330/1.142/0.918/0.693/0.344 | same |

### A10. Rankine planing gates vs Savitsky (1964) equations — regression of measurements
Flag: Savitsky formulas coded in `hydro/rankine/planing_steady.py`; original paper not re-read in that stage (coefficients cross-checked against published restatements); the Chambliss & Boyd / Shuford raw tables were not transcribed (data/external/SOURCES.md:301).

| id | stage | quantity | value | reference | file |
|---|---|---|---|---|---|
| A10.1 | B7 linear (69 conditions; beta 10/20/30, C_V 2/4/6, tau 2/4/6, lam 1.5/2.5/3.5) | share of C_L within ±10 % of Savitsky | 0.275 (all), 0.25 (beta 20, n 24) | tol 0.8 | rankine/planing_gate_full.log |
| A10.2 | B7 | C_L / C_L,Sav min / median / max | beta 10: 0.812/1.053/1.27; beta 20: 0.924/1.232/1.611; beta 30: 1.073/1.395/2.288 | 1 | same |
| A10.3 | B7 | share l_p/(lam b) within ±0.05; share (L_K-L_C)/b within ±15 % | 0.826; 0.942 | tol 0.8 | same |
| A10.4 | B7 free equilibrium (beta 20, 16 cases C_V 2/3/4/6, CD 0.3/0.6, lcg 1.5/2) | trim within ±1 deg; wetted length within ±15 % | 0.75 (n 12 inside the validity box); 1.0 | tol 0.8 | same, "EQ" lines (e.g. Cv4 CD0.3 lcg1.5: tau 2.321 vs Sav 2.817, lam 2.097 vs 2.138) |
| A10.5 | B7 flat plate (beta 0) | C_L / C_L,Sav | 0.304, 0.293, 0.290 (not converged) | 1 | same, VF |
| A10.6 | B8 composite (MLM sections + separation), beta 20, 24 cases | share within ±10 %; median ratio | 0 of 24; median 1.443 (min 1.254, max 1.554) | — | rankine/planing_b8_full.log, planing_b8_posttier.log |
| A10.7 | B8 variants, beta 20 | within ±10 %, median ratio | linear 2/24, 1.280; comp 0/24, 1.443; mlm_only 9/24, 1.016; sep_only 0/24, 1.627; anchor_wen 0/24, 1.427; v_slender 0/24, 1.427 | — | planing_b8_posttier.log |
| A10.8 | B8 beta 10, composite | C_L/Sav min/median/max | 0.553 / 1.305 / 5.9e16 (non-converged cases with C_L ~1e15) | — | planing_b8_full.log |
| A10.9 | B8 beta 30, composite | min/median/max | 1.411/1.467/2.488 | — | same |
| A10.10 | B8 share l_p/(lam b) within 0.05 | 0.072 | 0.8 | same |
| A10.11 | B8 free equilibrium (post tier) | trim within ±1 deg | 3 of 12 (e.g. Cv2 CD0.3 lcg1.5: tau 2.836 vs Sav 4.786; Cv4 CD0.6 lcg1.5: 3.086 vs 4.723; Cv6 CD0.6 lcg2: 1.702 vs 2.345) | — | planing_b8_posttier.log |

### A11. Displacement-ship plant (sim/vessel family, not the planing plant) vs model tests — doc only (result files not located in `_cache`)

| id | quantity | model | measured | source |
|---|---|---|---|---|
| A11.1 | KVLCC2 1:68 (SSPA roll decay, Mendeley CC-BY) natural period | 2.543 s (prediction); plant time domain 2.565 s | 2.545 s | IL F8 (line 696), G4 |
| A11.2 | KVLCC2 zeta_eq(5 deg) 0 kn / 15.5 kn, frequency-domain prediction | 0.0077 / 0.0129 | 0.0049, 0.0052 / 0.0145 | F8 |
| A11.3 | KVLCC2 plant decay, model/measured zeta_eq(5 deg), 0 kn (x2) / 15.5 kn: placeholder / simplified Ikeda / decay-test input | 3.33, 3.13 / 1.15 ; 1.68, 1.58 / 0.76 ; 0.95, 0.96 / 0.97 | 1 | G4 |
| A11.4 | KCS T2015 Fr 0.26 head seas, model/measured, lam/L 0.65/0.85/1.15/1.37/1.95 | heave 0.55/0.43/0.77/1.09/1.04; pitch 1.65/1.43/0.96/1.05/1.14 | EFD (data/external/kcs_t2015) | F8 |
| A11.5 | KCS oblique (2.11) roll, model/measured | 0.27-2.5 | EFD | F8 |
| A11.6 | Wigley USV manoeuvring Yv', Nr' (Clarke 1983 regression) | 0.0313, 0.00466 | identified 0.025, 0.0060 | anyspeed/tests/hydro.manoeuvring.log |

---------------------------------------------------------------------------------------------------------------------

## B. Comparisons with other computations or analytic solutions (verification)

### B1. Planing plant vs BEM (Capytaine) and exact 2D sections

| id | run | quantity | plant | other computation | file |
|---|---|---|---|---|---|
| B1.1 | v0 scarab195 at rest | hydrostatics C33 / C55 / C35 | plant d(w')/dz 9.4 1/s2, d(q')/dth 10.6 | BEM mesh at plant rest attitude: C33 87738 N/m, C55 197509 N m/rad, C35 -19658 | anyspeed/rest_rao.txt |
| B1.2 | same | heave / pitch natural frequency rad/s | 3.07 / 3.26 | BEM (A near 3.9 rad/s) 3.94 / 4.17 | rest_rao.txt |
| B1.3 | v0 zero speed, regular head waves a 0.02 m, surge held | heave/a, pitch/ka at omega 1.35/2.20/3.04/3.89/4.74/5.59/7.28 (lam/L 6.28..0.22) | heave 1.003/0.870/2.458/0.075/0.106/0.073/0.041; pitch 1.071/1.027/1.040/0.191/0.049/0.008/0.006 | BEM head seas (table direction index 3 = beta pi; 2-DOF heave-pitch, exact-section C of V1): heave 0.965/0.769/0.288/0.074/0.027/0.030/0.009; pitch 0.992/0.858/0.523/0.128/0.058/0.017/0.003 (with BEM C of B1.1: heave 0.967/0.771/0.290/0.073/0.027/0.029/0.009; pitch 0.998/0.866/0.528/0.129/0.057/0.017/0.003). Superseded (following seas: rest_rao.py read F[:, 0, :] = direction index 0 = beta 0 = waves travelling +x toward the bow, while the plant column is theta0 = pi head seas): heave 0.967/0.775/0.435/0.282/0.142/0.061/0.015; pitch 0.999/0.875/0.528/0.155/0.064/0.017/0.002 | plant: rest_rao.txt; BEM head: derive/v1_results.json (fd: heave_exactC, pitch_exactC / heave_bemC, pitch_bemC), derive/v1_linear_consistency.txt; direction convention: derive/v9_excitation_check.txt step 2 |
| B1.4 | v0 rest | heave added mass | 7877 kg (Wagner c capped); with geometric half-beam 5923 kg | BEM probe A33 (24 freq, lid): 11244 (0.5 rad/s), 10141 (1.35), 7312 (2.20), 5215 (3.04), 4248 (3.89), 3891 (4.74), ~3500-4300 above 5 rad/s; memory fit A33_inf 4538 kg | audit.txt:1; bem_probe_lid_24_64.txt; memory_fit.txt |
| B1.5 | same | pitch added inertia | 16088 kg m2 | BEM A55 12508 (0.5) .. 7549 (4.74) .. ~6300-7500 high; A_inf 7699 | bem_probe_lid_24_64.txt; memory_fit.txt |
| B1.6 | v0 lateral inertia (frozen at 25 kn attitude, all speeds) | sway / roll / yaw / surge added | 210 kg / 141 kg m2 (0.2 Ix) / 381 kg m2 / 70 kg | BEM rest: A22 470-534 (0.5-2.2 rad/s), A44 542-638, A66 865-1014, A11 146-175; slender m22 at rest 804 kg, A66 1782-1791 kg m2 | bem_probe_lid_24_64.txt; audit.txt:86; audit_c2.txt:52 |
| B1.7 | exact 2D phi=0 sections (source panels) vs v0 m_a, scarab195 rest, 48 stations | heave sum | v0 7877 kg | exact 2D phi0 5247 kg (5238 in verify_math C5); pi/2 rho c_g^2 5923; 20 deg wedge v0/exact ratio 2.81 | principles/ledger/checks.txt:24; verify_math.txt:19 |
| B1.8 | same, sway | sum | v0 pi/2 rho d^2 804 kg | exact 2D rigid lid 524, phi0 340 (phi0/lid 0.648); at 25 kn: 210 / 136 / 90 | checks.txt:24 |
| B1.9 | 3D BEM at rest vs v0, heave | factor | v0/3D 1.74 = section/pile-up 1.50 x 3D 1.16 | A33_inf 4538, A22_inf 318 | checks.txt:24 |
| B1.10 | BEM probe | radiation damping ratio at heave resonance | — | zeta33 0.297 at 3.89 rad/s; surge FK |F1|/(rho g V k): 1.00 (0.5), 0.94 (1.3), 0.72 (2.2), 0.28 (3.0), 0.18 (3.9), <=0.08 above | bem_probe_lid_24_64.txt |
| B1.11 | BEM probe mesh | volume | plant m/rho 1.3659 m3 | mesh 1.3622 (-0.27 %); x_B 0.0034 m from CG; waterplane 8.726 m2 | same |
| B1.12 | memory fits (order 10) | err_K / err_B / err_A per pair | Heave 0.009/0.015/0.134; Pitch 0.013/0.133/0.144; Heave-Pitch 0.087/0.135/0.195; Surge 0.023/0.026/0.116; Sway 0.337/0.012/0.048; Roll 0.683/0.007/0.041; Yaw 0.006/0.001/0.109; memory 1.9-3.7 s | — | memory_fit.txt |
| B1.13 | as1 p1b, zero-speed FK 6 DOF, lam 8 station spacings to 8 L, headings 0-180 | max |plant - Capytaine| / DOF scale | 9.13 % | Capytaine (two meshes differ 0.91 %) | anyspeed/p1b/checks_fk3d.json |
| B1.14 | as1 review_fix: 48-station plant vs continuum (1536 st) | same measure | 2.72 % | continuum vs Capytaine on exact mesh 0.02 %; Capytaine meshes 0.008 %; continuum vs independent integral 0.0025 % | p1b/review_fix/checks_fkref.json |
| B1.15 | as1 lateral excitation in long waves (lam/L 3-45, 60/90/120 deg) | vs Capytaine FK + diffraction | sway/roll largest 6.8 %, yaw 83.9 % (p1b); 76.0 % (fk_exact+orbital) | Capytaine | p1b/checks_latexc.json; p1b/review_fix/fo/... |
| B1.16 | beam-sea diffraction sway force (plant vs Capytaine), lam/L 3/6/12/45 | N | 1239.6/632.6/319.6/85.9 | 869.0/675.0/391.6/114.8; FK sway 5065/2603/1317/354 vs 5052/2597/1314/353; lever N/Y plant 0.0115..0.0033 m vs Cap 0.341..0.103 m | p1b/diag.txt |
| B1.17 | added-mass lever A26/A22 | m | strips phi0 +0.0027 (339 kg); strips rigid lid +0.0349 (524 kg) | Capytaine omega inf +0.0501 (319 kg), omega 0 +0.0954 (467 kg) | p1b/diag.txt |
| B1.18 | as1 2D section BEM vs analytic (semicircle, half-ellipses) | rel err | 6.95e-05; plate Chebyshev 4/pi^2 -2.5e-08; 1 %-thick slab sway +1.29 % (lid 0.6448) ; slab phi0/lid 0.40419 vs 0.40528 | analytic | p1b/checks_sec2d.json |
| B1.19 | as1 hx roll natural frequency at rest | rad/s | plant hx 4.1138, hh (v0 lateral inertia) 4.4475 | Capytaine sway-roll root 3.8781 (mesh 8720 faces; coarse mesh 3.8789) | anyspeed/s5a/r2/l8.txt |
| B1.20 | as1 S1 linear pitch damping D55 vs continuum / closed form, 1-9 kn | FD/cont | 0.9960-1.0012 (largest |FD/cont-1| 0.40 %); FD/closed 0.9981-0.9983 where defined | continuum linearisation on 40000 points; closed form U x_T^2 m_aT + ... | anyspeed/s1/a_d55.txt |
| B1.21 | D55 values (N m s), as1 vs v0, 1/2/3/4/5/6/7/8/8.5/9 kn | as1 FD | 4410/8832/13281/17773/22389/27218/32430/56784/65624/71551 | v0 FD -1449/-2898/-4347/-5796/-7208/-8575/-9901/-11174/-11786/-12378 | a_d55.txt |
| B1.22 | P1a D55 vs S1 continuum at 2/4/6 kn | % | -0.45 / +0.34 / -3.27 | continuum | p1a/review_fix/checks_hullb.json |
| B1.23 | as1 FK1: transom wet, longitudinal wave force vs textbook closed-body long-wave FK, 0.5-6 kn x lam/L 0.5-6 | largest |plant/textbook-1| | 9.64 %; vs a_b(u) x textbook 0.34 % (a_b 1.000 to 4 kn, 0.904 at 6 kn) | analytic | p1a/review_fix/checks_fk.json |
| B1.24 | FK2 default transom state vs closed body | 13 of 25 within 5 %; outside e.g. 4 kn lam/L 0.5 -26.1 %, 1.0 -26.3 %, 2.0 -14.1 % | analytic | same |
| B1.25 | FK3 default plant vs exact linear FK with depth decay | 8 of 25 within 5 %; e.g. 0.5 kn lam/L 0.5 +39.1 %, 1.0 +18.3 % | exact | same |
| B1.26 | fk_exact (p1b+) transom wet vs a_b x exact FK with depth decay | largest | 0.05-0.08 % | exact | p1b/checks_fk11.json |
| B1.27 | plant linear wave force (6 DOF, rest and 25 kn) vs independent quadrature | largest rel | 1.93-1.94e-02 (registered 1e-6) | quadrature | p1b/checks_fk2d.json, p1c/as1/... |
| B1.28 | zero-speed wave-force split (v0 audit), captive, regular head wave lam 40 m, a 0.25 m | amplitude / reference W x slope (537 N), phase | 3 kn: Xp_plant 1.18 @+7, Xp_buoy 1.19, Xw 0.98, FK_full 0.98, total 2.16 @+4; 25 kn: Xp 1.54 @-2, Xp_buoy 0.42, Xp_rest 1.13, Xw 0.23, FK_full 0.45, total 1.76 | closed-body FK | audit.txt:97 |
| B1.29 | principles D: v0 discrete vs exact FK, captive rest attitude, a 0.02 m, lam/L 1/2/4/8 | |Xw|/|FK|; |v0 bottom wave part|/|FK|; |v0 total|/|FK|; phase deg | 1.000; 0.094/0.344/0.700/1.403; 1.029/1.106/1.259/1.749; -5.1/-17.9/-33.7/-53.3 | |FK exact| 57.59/107.76/72.79/39.07 N | principles/ledger/checks.txt:85 |
| B1.30 | surge wave-force split vs speed (v0), lam 40 m, a 0.25 m, ref 539 N | Xw/ref, Xp/ref, total/ref (head; follow) | 0.5 kn 0.98/1.09/2.07; 0.97/1.06/2.02 — 3 kn 0.98/1.19/2.17; 0.97/1.02/1.99 — 6 kn 0.98/1.37/2.34; 0.96/0.94/1.90 — 9 kn 0.97/0.70/1.67; 0.84/1.47/2.26 — 13 kn 0.30/1.28/1.57; 0.30/1.04/1.33 — 18 kn 0.18/1.32/1.49; 0.17/1.01/1.17 — 25 kn 0.23/1.54/1.76; 0.14/1.02/1.16 | W x slope | wave_regimes/surge_check.txt |
| B1.31 | zero-speed wave excitation per metre of wave amplitude, scarab195 at the plant rest attitude: Capytaine (probe mesh 64x(10+6), lid, re-solved at 3.0/3.9/4.7 rad/s) vs repo Rankine solve (V2, derive_v2_hull.py) | \|X1\| N/m, \|X3\| N/m, \|X5\| N m/m, head seas (beta pi) at 3.0 / 3.9 / 4.7 | Rankine head: X1 1835/1975/1517; X3 15365/5858/4520; X5 43378/17871/13641 | Capytaine head: X1 1719/2026/1524; X3 15611/5906/4174; X5 42372/17941/13414 (Rankine/Capytaine: X3 0.98/0.99/1.08, X5 1.02/1.00/1.02, X1 1.07/0.98/1.00; phase(X5/X3) head 119/-162/157 deg vs Rankine 120/-161/160). Capytaine following (beta 0): X1 4285/5640/6010; X3 15762/10776/7419; X5 38341/16148/13445 — following/head X3 1.01/1.82/1.78, X1 2.49/2.78/3.94. Stored table bem_probe_lid_24_64.npz at 3.04/3.89/4.74: head X3 14408/5859/3944, following 14867/10784/7284 (re-solve reproduces within 0.7 %); V2's "Capytaine 14867/10784/7284" is the following-seas column. Checks on the re-solve: Haskind-Newman damping/directional integral (13 directions) heave 1.000-1.004, pitch 0.992-1.005, surge 0.964-1.023; direct Haskind (radiation potential + incident wave) vs integrated force, head seas: heave -1.0/+1.7/+0.1 %, pitch +0.5/-1.2/-0.1 %, surge -2/-6/+8 %; fine mesh 96x(14+8) head X3 5896/4258, no lid 5886/4159 (3.9/4.7); first irregular frequency estimate (box formula, no lid) 5.36 rad/s; frequency offset 3.89->3.90 +0.9 %, 4.74->4.70 +6 % on head X3 | derive/v9_excitation_check.txt; derive/v9_results.json; v2_tables.txt:60 |

### B2. Planing plant vs Zarnick (1978) own 2D+t computation (another computation) — see A3 last column (digitised).

### B3. Analytic / theory numbers computed for the plant's regime (principles)

| id | quantity | values | file |
|---|---|---|---|
| B3.1 | piston wavemaker exact linear radiation, Kd 0.1..1000: A/A_inf; B/(w A_inf) | 3.297, 1.530, 0.212, 0.300, 0.771, 0.928, 0.979, 0.993, 0.998; 4.104 .. 4.5e-06; Kramers-Kronig rel diff <= 6.3e-11 | principles/ledger/checks.txt:3; verify_math.txt:26 |
| B3.2 | 2D added mass with free-surface image: semicircle heave phi0 1.57084 (exact 1.57080), sway lid 1.57075; half-ellipse; sway phi0/lid 0.40530 = 4/pi^2 | checks.txt:24; verify_math.txt:3 (source-panel series 0.4049-0.4062) |
| B3.3 | 20 deg wedge fixed-waterline heave phi0 = 1.38138 rho c_g^2 = 0.87942 x (pi/2) rho c_g^2; Wagner pile-up value 2.8057x this | verify_math.txt:19 |
| B3.4 | Michell thin-ship, classic Wigley: C_w at Fn 0.2/0.3/0.4/0.5/0.7/1.0/1.5/2.0 = 8.88e-4/2.14e-3/2.73e-3/4.52e-3/3.17e-3/1.84e-3/8.76e-4/4.59e-4; high-Fn constant C_inf 1.67278 | checks.txt:47; verify_math.txt:45 |
| B3.5 | Michell on scarab-proportioned Wigley (outside thin-ship assumption): R_w/W(rho g vol) 2/4/5/6/7/8/10/12/14/17/20/25/35/50 kn = 0.0006/0.0144/0.0164/0.0690/0.1202/0.1457/0.1674/0.1808/0.1907/0.1984/0.1987/0.1889/0.1572/0.1152; R_w 3588/3739/3555/2958 N at 14/20/25/35 kn | checks.txt:47; verify_math.txt:56 |
| B3.6 | steady phi=0 parameter g L_k^2/(U^2 c) on v0 branch: 1 kn 1067, 3: 118.6, 5: 42.7, 7: 21.8, 9: 13.2, 10: 10.6, 12: 4.94, 14: 3.32, 16: 2.81, 18: 2.43, 20: 2.20, 25: 1.71, 30: 1.20, 35: 0.93; Savitsky 2 lam^2/C_V^2: 165/52.7/22.2/10.4/7.32/4.00/2.48/1.70/1.25/0.96/0.57/0.38/0.27; Fridsma A 3.88, B 1.00 | checks.txt:116 |
| B3.7 | JONSWAP Hs 1 Tp 5: share of variance with (w_e^2/g)(b/2) < 1, headings 0(following)/45/90/135/180(head): 0 kn 0.98 all; 15 kn 0.964/0.985/0.979/0.767/0.661; 25 kn 0.877/0.946/0.979/0.572/0.222; 35 kn 0.753/0.880/0.979/0.231/0.056 | checks.txt:116; verify_math.txt:85 |
| B3.8 | additive composite error, wavemaker with draught change r = 0.7/1.3/1.6: Kd0 1: -1.07/+0.41/+0.58; 3: -0.59/+0.16/+0.21; 10: -0.136/+0.038/+0.050; 100: -0.012/+0.0036/+0.0047 | checks.txt:149; verify_math.txt:37 |
| B3.9 | memory/inertia impulse ratio R(T=L/U), zero-speed 3D BEM heave, cut 15.8 rad/s (A_inf 4660): 2 kn 1.046, 5: 1.400, 10: 1.403, 15: 1.133, 20: 0.897, 25: 0.723, 30: 0.587, 35: 0.479; cut 12.4 (A_inf 4757): 1.007/1.354/1.360/1.093/0.874/0.691/0.546/0.436; gravity length 0.66-0.79 m | principles/unified_cost.txt (part B); verify_math.txt:63 |
| B3.10 | regime thresholds mapped to speed (v0 attitudes): Fn_L 0.6 (Sun & Faltinsen 2007) 8.5 kn; Fn_B 2.5 21.7 kn; F_T 2.0 (Maki 2004) 8.6; F_T 2.5 (Doctors 2003) 10.8; F_T 2.6 11.2; F_T 3.25 13.3; F_T 4.5 (Saunders) 16.4 | unified_cost.txt (part A) |
| B3.11 | dense panel kernels on this laptop: N 1000/2000/4000/8000 CPU LU 0.54/0.80/1.69/5.53 s; GPU f32 solve 0.004/0.009/0.031/0.19 s | unified_cost.txt (part C) |

### B4. Rankine solver verification (analytic / exact / other codes) — final tier values

| id | stage | check | value | reference | file |
|---|---|---|---|---|---|
| B4.1 | B3 | panel integrals vs adaptive quadrature (6 panels) | max rel 6.5e-14..1.9e-12; far points 5.6e-12..2.7e-11; rectangle 50-digit 7.0e-12 / 1.5e-11 | quadrature / closed form | rankine/verify_full.json C1, C1b |
| B4.2 | B3 | sphere A11/(rho V/2)-1, N 512..32768 | -0.0101, -0.0025, -6.3e-4, -1.57e-4 (order 2.0) | Lamb | C3 |
| B4.3 | B3 | spheroid a/b 5 A11/A33/A55 rel err finest | -6.7e-4 / -2.2e-6 / -1.0e-5 | Lamb closed form | C4 |
| B4.4 | B3 | hemisphere A33/(rho V) phi=0; rigid lid | 0.4999 (ref 0.5); 0.8308 (Hulme 0.831) | exact; Hulme 1982 | C5, C6 |
| B4.5 | B3 | hemisphere vs Hulme table, max |err A'|, |err B'| | 0.0024-0.0029, 0.011-0.022 | Hulme 1982 | Z1 |
| B4.6 | B3 | Haskind vs direct; energy relation; KK | 0.0036; 0.0100; 0.0020 | identities | Z4, Z5, Z6 |
| B4.7 | B3 | hemisphere vs Capytaine (90x90) max |dA'|/A', |dB'| | 0.0111, 0.0312; barge vs Capytaine 0.008/0.020/0.011/0.037 | Capytaine 3.0.0 | Z2, Z3 |
| B4.8 | B3 | line source R/R_Havelock - 1, 6 cases Fn 0.4-0.7 | -0.0012..+0.00008 | Havelock 1931 | S1 |
| B4.9 | B3 | operator comparison R/R_H (dawson4, upwind5, shifted Numerov) | Fn 0.5: 0.913/0.999/0.999; Fn 0.4 f4: 1.153/1.012/1.000 | Havelock | S1b |
| B4.10 | B3 | spheroid R_near/R_Havelock (body/free-surface interaction size), f/b 2.5/4/6 at Fn 0.5; 2.5/4 at Fn 0.4 | 1.063/1.026/1.004; 1.299/1.059 | Havelock approximation | S2 |
| B4.11 | B3 | Rankine vs exact-Kelvin body BIE, near/far | -0.064..-0.12 % / -0.005..-0.008 % | exact Kelvin Green function | S2K |
| B4.12 | B3 | Kelvin wake: transverse wavelength; half-angle; L2 error | +0.038 %; 18.14 vs 18.37 deg; 0.016 (n 30), 0.0093/0.0086 (n 45/60) | exact linear Kelvin | S3, S3b |
| B4.13 | B3 | Fz/U^2 relative to double body at Fn 0.2/0.1/0.05 | 1.273/1.060/1.014 | rigid-lid exact limit (tol 1 % at 0.05) | S4 |
| B4.14 | B3 | g->0 Fz/U^2 relative to phi=0 at Fn 2/4/8/16; Richardson limit | 1.331/1.090/1.023/1.006; 1.001, order 1.965 | exact limit | S9, S9r |
| B4.15 | B3 | raw double-body x-force / R_H on asymmetric meshes 40x16..120x48 | 0.616/0.295/0.084/-0.041 | 0 (d'Alembert) | S8 |
| B4.16 | B4 | Wigley R_near/R_cut - 1 at Fn 0.2..0.5 | 0.025, 0.033, 0.021, 0.023, 0.024, 0.015, 0.003, 0.0045 | wave-cut flux | hull_verify_full.json H6 |
| B4.17 | B4 | thin-ship limit R_near/R_Michell at B/L 0.1/0.05/0.025, extrapolated | 0.605/0.801/0.901 -> 1.003; R_cut 0.591/0.787/0.884 -> 0.980 | Michell | H8 |
| B4.18 | B4 | g->0 limit |Fz - Fz(phi0)|/|Fz_db| at Fn 2/4/8 | 23.1 / inf / inf | exact limit | H9 |
| B4.19 | B4 | transom momentum identity R_id/R_cut - 1, Fn 0.35/0.45, dry/wet, coarse/prod | 0.065..0.17 (tol 0.05) | identity | T3 |
| B4.20 | B4 | free-surface series C_near m 60/90/135; Fz | 1.089/1.108/1.146 e-3; Fz -0.019659/-0.019659/-0.019656 | own | H11b |
| B5.1 | B5 | U = 0 new formulation vs stage-2 zero-speed solver, Wigley I | A33' 0.25 %, B33' 0.50 %, A11' 3.9 % ... | stage-2 solver | unsteady_verify_full.json U0 |
| B5.2 | B5 | U->0 continuity, diagonal A', B' at Fn 0.05 vs U = 0 | 0.111 (M 40/60/80 same) ; ratio of changes Fn 0.05/0.025 3.65 | U = 0 | U0c |
| B5.3 | B5 | Timman-Newman identity (waterline-reduced FS term), Wigley I; prismatic V | 0.067 (rerun 0.040); prism 4.9-5.8; classical OT residual 0.72; submerged spheroid 0.0043 | identity | TN, TNp |
| B5.4 | B5 | Haskind at speed (b) | 0.016 (rerun 0.011); classical w/o FS term 0.20 | identity | HK |
| B5.5 | B5 | submerged sphere Fr 0.4 h 2r vs Wu & Eatock Taylor (secondary source Mei et al. 2020 tables) | max error/tol 2.81 (14 of 104 outside), NK m-terms 19.2 (94 of 104) | Wu & Eatock Taylor 1988 via Mei 2020 | WE, WEnk |
| B5.6 | B5 | KK, Fn 0.2/0.3, D/OT | 1.1e-4..6.1e-4 | KK | KK |
| B5.7 | B5 | spectrum max Re(lambda) Fn 0.2/0.3/1.0, prismV 0.4; U=0 no beach | -0.0083/-0.0133/-0.355/-0.035; +3.3e-4 (1034 eigenvalues Re>0) | < 0 | SP |
| B5.8 | B5 | time domain vs frequency domain T33; Cummins reconstruction | 0.0066; 0.0035 / 0.0010 | same LTI | TD, CU |
| B5.9 | B5 | static stiffness C0 vs -dF_steady/dq (D-form, OT) | 0.062, 0.032 | plant law | C0B |
| B5.10 | B5 | demo planing table mid-box interpolation vs direct | D 6.07, OT 3.45 (A_inf 1.46e-2, K rms 0.61/0.35) | tol 1 | PVm |
| B6.1 | B6 | double-body m-terms on sphere vs analytic | 0.0052 (64x32), order 2.0; production FV adjoint 0.74 | analytic | db_verify_full.json MT |
| B6.2 | B6 | conservative K f on exact half-spheroid flow, observed order band / lattice; uniform stream K 1 (exact 0) non-smooth lines | -0.45 / -0.52; errors 1.17 -> 2.34 -> 4.56 (M20/40/80) | exact | DXa, DXu |
| B6.3 | B6 | TN double body Wigley; prism flared; wall; spheroid | 0.013; 0.052; 0.029; 0.0044 | identity | TN, TNp, TNw, TNs |
| B6.4 | B6 | WE sphere, reference model, mesh 24x12 / 48x24 | 0.88 (0 of 104 outside) / 1.27 (2 of 104) | Wu & Eatock Taylor via Mei | WE |
| B6.5 | B6 | double-body steady R vs exact-Kelvin, deep spheroid Fn 0.5 | R/R_Kelvin 1.0103 (unscaled domain 1.1289, x1.5 1.0167) | exact Kelvin | SK |
| B6.6 | B6 | U->0 continuity double body, Fn 0.0125/0.025/0.05/0.1 | 0.0092/0.0353/0.1473/0.3065 | U = 0 | U0c |
| B6.7 | B6 | prismV Fn 0.6 attitude-grid linearity max |dF - (-K dq)|/|K dq| (30x6 / 60x12) | 1.29 / 8.05 (|W|/U max 3.0 / 22.3 at tip corners) | linear prediction | AT5 |
| B6.8 | B6 | planing table interpolation mid-node Fn 0.5/0.7/0.9 | K rms 0.144/0.110/... (measure 1.85) | direct | PVs |
| B6.9 | B6 | production vs natural kinematic operator coefficients: wigleyJ / prismV | 0.0006 / 0.337 (C0 diag 33: -0.423 vs -0.457) | own | KCN |
| B7.1 | B7 | kernel tables vs brute-force 2D FFT | 0.0062 (tol 0.005) | FFT | planing_gate_full.log PK1 |
| B7.2 | B7 | high-speed limit C_L vs half thin-wing VLM | 0.0520 vs 0.0548 (rel 0.052); cp 1.679 vs 1.779 | VLM | PL1 |
| B7.3 | B7 | slender limit (L_K-L_C)/b at C_V 10 vs tan(beta)/(pi tan tau) | 1.585 / 1.584 vs 1.657 | analytic | PL2 |
| B7.4 | B7 | grid: C_L at h16/32/64 | 0.0915/0.0968/0.0987 (order 1.51) | own | PC1 |
| B8.1 | B8 | MLM wedge force F/(rho V^3 t) vs Wen 2022 similarity, beta 10/20/30 | 209.3/42.87/15.09 vs 212.8/42.00/13.83 (Wagner 249.3/58.5/23.3) | Wen et al. 2022 (transcribed constants) | planing_b8_full.log W2C |
| B8.2 | B8 | MLM peak pressure p/(rho V^2/2) vs Zhao & Faltinsen 1993, beta 10/20/30; peak height /(Vt) | 81.80/20.98/9.62 vs 77.85/17.77/6.93; 0.562/0.538/0.510 vs 0.556/0.509/0.424 | ZF 1993 via Iafrati & Battistin (secondary transcription) | W2P, W2Z |
| B8.3 | B8 | MLM code vs published MLM values | 2.1e-4 | arXiv 2011.10378 table | W2A |
| B8.4 | B8 | gravity-free limit C_L vs VLM extrapolated | 0.05418 vs 0.05409 | VLM | PL1M |
| B8.5 | B8 | slender limit (L_K-L_C)/b; V/U vs tan tau | 1.6055/1.6045 vs 1.6568; 1.032 | analytic | PL2 |

### B5. Displacement-plant gates (other computations / analytic) — anyspeed/tests/*.log (run 2026-10-05)

| id | quantity | value | reference | file |
|---|---|---|---|---|
| B5d.1 | floating hemisphere V, C33 error 12x12..40x40 | V -6.13..-0.56 %, C33 -6.84..-2.84 %; A' low ka 0.8479 vs Hulme 0.831 (+2.0 %); high ka 0.4995 vs 0.5 | analytic, Hulme | hydro.verify_sphere.log |
| B5d.2 | RAO time vs frequency domain (Wigley), heave/pitch | worst 0.28 % / 0.10 %, phase 0.2 deg; decay period 1.53 s, zeta 0.234 | same LTI | sim.verify_rao.log |
| B5d.3 | retardation fit (Wigley) err K/B/A worst | 0.13/0.46/1.96 % (auto order); memory 2.2-2.9 s | own | hydro.verify_retardation.log |
| B5d.4 | strip hydrostatics vs analytic / BEM (Wigley) | V err 0.113..0.002 % (21..161 stations); C33 BEM -0.02 %, C55 -0.35 % | analytic / BEM | sim.test_sections.log |
| B5d.5 | nonlinear vessel vs linear theory RAO a 0.02 m | heave err <= 0.23 %, pitch <= 0.13 % | frequency domain | sim.test_vessel.log |
| B5d.6 | simplified Ikeda vs Kawahara 2009 fig 1 (original Ikeda symbols, extracted from PDF vectors) | wave component rms 20.2 % / 43.1 % (OG/d -0.2 / -1.5); eddy 7.3 / 2.7 %; bilge keel 1.0 / 4.3 % | Kawahara 2009 | hydro.ikeda.log |
| B5d.7 | physics audit (Wigley): Hs recovered 3.278 vs 3.25; displacement 8.8915 vs 8.8889 m3; C33 1.6764e5 vs BEM 1.6756e5; C55 8.3918e5 vs 8.3504e5; added-mass asym 8.65e-4 | — | analytic/BEM | studies.audit_physics.log |

---------------------------------------------------------------------------------------------------------------------

## C. Internal-consistency numbers (identities, convergence, symmetry, reproducibility)

| id | run | quantity | value | file |
|---|---|---|---|---|
| C1 | v0 scarab195 at rest | hydrostatic longitudinal identity: bottom -rho g sum A tau dx / taper / transom face (absent in v0) / sum | -439.4 / -35.0 / +531.3 / +57.0 N (sum/W 4.15e-3, W 13734 N) | anyspeed/design_check.txt:1 |
| C2 | same, trim 1.83/3.83/5.83 deg | bottom / face / last-strip open forward face | -439.4/+531.3/-73.6; -894.1/+903.8/-10.6; -1394.2/+1394.3/+0.0 N | anyspeed/review_physics.txt:1 |
| C3 | divergence identity, quadrature (principles D): calm rest / +3 deg / wave a 0.1 lam 2L / lam 1L +2 deg | bottom, sides, face, sum; sum - closed-body FK | -439.15/-92.19/+531.34/0.000; -1009.37/-0.15/+1009.52/0; -579.22/-20.59/+950.41/+350.60 (FK +350.60); -889.03/-0.83/+616.57/-273.29 (FK -273.29); residuals <= 3.2e-4 N | principles/ledger/checks.txt:85 |
| C4 | v0 rest, zero thrust | surge accel, Xp, friction; W tan(trim) | -0.2989 m/s2, -439.4 N, 0 N; 439.6 N | audit.txt:9 |
| C5 | v0 captive rhs rebuilt vs plant.deriv | max |diff| (w', q') | 3.30e-17 | audit.txt:15 |
| C6 | Heun dt 0.02 undamped oscillator growth | heave rest T 2.06 s: |G| 1.00000173 (zeta -2.8e-5); roll T 1.41 s 1.00000789 (-8.8e-5); heave 25 kn T 1.47 s 1.00000668 (-7.8e-5) | audit.txt:92 |
| C7 | Heun amplification | w 3.94 dt 0.02: energy x1.04938 per 100 s; w 4.9 dt 0.02: x1.12220; w 4.9 dt 0.004: x1.00092 | principles/ledger/checks.txt:165; verify_math.txt:92 |
| C8 | P1a energy, rest, dissipation off (E0 25.28 J) | Heun dt 0.002: +0.0059 %/100 s; dt 0.02: +6.02 %/100 s (600 s: +7.88 %/100 s, max dev 48.1 %); verlet dt 0.002: +0.0001 %; dt 0.02: -0.0003 % (max dev 0.17 %) | anyspeed/p1a/review_fix/energy.txt |
| C9 | p1b candidate energy (all switches incl. lateral_phi0) | -4.33 %/100 s; lateral energy -0.133 %/100 s; P1a start on candidate: dt 0.002 +0.0015 %, dt 0.02 +0.454 % | p1b/checks_energy.json; p1b/diag.txt |
| C10 | wagner_entry W8 | +0.00118 %/100 s (dt 0.002) | p1b/review_fix/checks_w9_w8.json |
| C11 | hh H5 / hx L4 energy | +0.00002 %/100 s; max deviation over 20 s at dt 0.004/0.002/0.001: 0.222/0.126/0.039 % | s5a/r2/checks_h5_hh.json, checks_l4_hx.json |
| C12 | E2: verlet on linearised heave/pitch 0-40 kn, h 0.02 | largest ln|mu|/h - Re lambda (oscillatory modes) +6.73e-3 1/s; every-mode form +0.250 (Heun +0.221) | p1a/review_fix/rereg/checks_e2speed.json; checks_e2speed.json |
| C13 | Newton-polished rest equilibrium, scarab195@as1 / planing_b@as1 | Fx/W, Fz/W, My/(WL) all <= 2.3e-17 | p1a/review_fix/checks_rest.json; p1b/checks_stand.json |
| C14 | M2 section closed forms vs Gauss quadrature; slice identity; faceted hull identities | 4.0e-15; dQ/dx identity 7e-11..3e-10; faceted 2.5e-14; closed body 1.1e-15; wave case 48/192 stations 2.7e-3/4.3e-4 (lam/L 0.5 at 48/192/768: 7.7e-3/1.1e-3/1.9e-4) | p1a/review_fix/rereg/checks_m2.json |
| C15 | normal_rate momentum identity at 384 stations (first form) | worst 1.2e-2; without normal_rate on planing_b 4/6/8/10 kn residual 2100/1012/261/13 %; re-registered telescoped form: none outside 1e-3 | p1a/review_fix/checks_nr.json; rereg/checks_nr.json |
| C16 | "13 normal_rate" at zero speed (dynamic horizontal force = -d/dt sum m_a V tau) | P1a 6.6e-7; p1b candidate and wagner_entry sets 7.6e-1 (m_0 table cell boundaries 2.9e-3, elsewhere 1.0e-6) | p1a/review_fix/checks_rest.json; p1b/checks_stand.json; p1b/diag.txt |
| C17 | W9 dynamic force identity (wagner_entry) | 1.0e-6 (14 chine-crossing samples excluded) | p1b/review_fix/checks_w9_w8.json |
| C18 | Froude twins lam 1.5/0.6 (nondimensional outputs) | p0 worst 3.65e-12; P1a 2.0e-12; p1b candidate 2.6e-10; hh 2.3e-11; hx 3.6e-9 (registered 1e-10 internal, 1e-4 M3) | p0/twin.txt; p1a/review_fix/checks_twin.json; p1b/checks_twin.json; s5a/r2/checks_twin_*.json |
| C19 | mirror symmetry (20 random states, heel ±20 deg) | 0.0 (p1b); 5.7e-16 (hh); 8.5e-16 (hx) | p1b/checks_sym.json; s5a/r2/checks_sym_*.json |
| C20 | golden runs bit-identical across processes and vs stored reference | 20 arrays, 0 differ (P1a); C2 re-pin task sea within 1.2e-12 / 1.95e-12 | p1a/review_fix/checks_golden.json; p1c/checks_repin.json |
| C21 | batch / fast path / kernel row-stability | 0 mismatches (8 cases incl. substeps 6 and a capsize); section integrals vs grid 0.0 | p1c/final/checks_kernel_batch_fastpath.json |
| C22 | v0 regression (golden, lofi fingerprint, turning table, streams, missions) and anyspeed_regress_as | bit-identical; 111 checks 0 failed | p0/regress.txt |
| C23 | geometry module vs v0 lines (6 hulls) | identical; offsets-table convergence orders: yc_max ~0.51, yc_l1 ~1.5, rise ~3.0, kp ~2.0, kpp_strip 1.55/4.68/0.59 | p0/geom.txt |
| C24 | chine capping continuity (12 kn trim, heave ±3 cm samples) | dDt_full: jump 742/747/748 N at dz 1e-4/1e-5/1e-6; with chine_exact_all 0.153/0.0155/0.00155 N | p0/cap.txt |
| C25 | station-count convergence of trim root (chine_exact_all), 9/12/16/20 kn | changes 1.0e-3..1.1e-4; 4.8e-5..1.5e-5; 6.9e-4..4.4e-6; 1.1e-4..1.2e-5 deg; dDt_full alone up to 6.3e-2 (12 kn), 3.9e-2 (20 kn) | p0/cap.txt |
| C26 | discrete momentum |sum(m_bar - m_c)|/sum m_c at 48/96/192/384 stations | rest 1.18e-4/2.94e-5/7.30e-6/1.83e-6; 12 kn 2.23e-4/3.08e-5/5.0e-7/3.3e-6; 20 kn 2.8e-5/1.4e-5/4.5e-7/1.7e-6 | p0/cap.txt |
| C27 | planing range Newton roots by station count 48/96/192 (v0 and S1-as1) | v0 trim spread up to 0.127 deg (16 kn), z up to 3.0 mm (40 kn); as1 spread <= 0.05 deg (24 kn) and 0.0007 deg above 26 kn | s1/d_planing.txt |
| C28 | v0 vs S1-as1 converged (192 st) planing roots dtrim / dz / dR | 14-22 kn 0; 24 kn +0.0009 deg/+0.14 cm/-0.68 %; 25: -0.026/+0.26/-0.87; 26: -0.043/+0.37/-3.06; 28: -0.019/+0.96/-7.63; 30: +0.095/+1.94/-6.13; 32: +0.174/+2.76/-13.6; 35: +0.161/+2.61/-13.8; 38: +0.140/+2.28/-14.0; 40: +0.133/+2.18/-14.0 % | s1/d_planing.txt |
| C29 | time-step study (as1 S1) 9.5 kn dt 0.02..0.0025; 28/30/35 kn | identical to 1e-15; v0 28 kn trim p-p 0.516/0.514/0.515 deg (limit cycle at all dt) | s1/b_decay.txt; s1/h_highspeed.txt |
| C30 | wagner_entry vs P1a calm equilibria and resistance | 1.3e-7 (first form), 9.1e-15/2.6e-14 (captive), 0.0 (p1c) | p1b/checks_wag.json; p1c/w2/w/checks_wag.json |
| C31 | W4 time-step convergence (F share, speed, heave/pitch rms within 1 % across dt 0.02/0.01/0.005 at 12, 28 kn) | p1b: not met; p1c w2/w3: met; as1 + wagner_entry (w4): not met | p1b/review_fix/checks_w4.json; p1c/w2/checks_w4.json; p1c/w4/... |
| C32 | heel hydrostatics: polygon GZ vs frozen curve | 1.3e-13 m (scarab), 1.7e-15 (planing_b); vs frozen table 0.185 mm (0.044 % GZ_max) / 1.662 mm (0.491 %); vanishing angle 103.34 deg | s5a/r2/checks_h2_hh.json; checks_h9_hh.json |
| C33 | Rankine: Green identity; FFT vs dense; similarity; domain/beach sensitivity; grid series | 2e-15..1e-13 (closed polyhedron); FFT-GMRES vs LU 4e-13..1.8e-10; L 1 vs 1.5 7e-10..1e-10; domain x1.5 changes <= 1.1 %; grid series changes 0.45-0.97 % (B5/B6), PROD Richardson err median 1.4e-3..4.6e-3 | rankine/*_verify_full.json C2, F1-F3, S5, S6, H5, GS, GSo, DOM |
| C34 | Rankine B7/B8 PC1 grid: C_L change between finest grids | 0.0195 (B7); 0.0095 (B8); kink bias 0.024 | planing_gate_full.log; planing_b8_full.log |

---------------------------------------------------------------------------------------------------------------------

## D. Behaviour observations of the plant

### D1. At rest, calm, zero thrust

| id | plant | observation | value | file |
|---|---|---|---|---|
| D1.1 | v0 PlaningVesselGZ | surge acceleration at the static attitude (trim 1.833 deg bow up, z +0.2334 m) | -0.2989 m/s2 (Xp -439.4 N) | wave_regimes/validity.txt:46; audit.txt:9 |
| D1.2 | v0 | 60 s free from rest | u -6.70 m/s, x -283.6 m | validity.txt:46 |
| D1.3 | P1a / later | 600 s zero thrust from equilibrium; from +5 cm; from +1 deg; from 5 cm/-1 deg/5 deg roll | u end 1e-15; 1.09e-3 m/s (max 2.55e-2, drift 1.15 m); -1.18e-3 (drift -0.60 m); -2.8e-4 (drift 0.29 m) | p1a/review_fix/checks_rest.json |
| D1.4 | wagner_entry set | same | +5 cm: 1.96e-3 m/s, drift 1.22 m; 5 cm/-1/5 deg: drift 1.86 m | p1b/review_fix/w/checks_stand_rest.json |
| D1.5 | hh / hx | roll 10 deg start | drift 0.11 / 0.10 m, mean heel 7e-16 / 4e-5 deg | s5a/r2/checks_h4_*.json |
| D1.6 | as1 P1a | coast-down from -3 kn, 300 s | u(300 s) -0.0427 kn; no sign change | p1a/review_fix/checks_astern.json |
| D1.7 | as1 P1a | captive astern holding force at -0.5/-1/-2/-3 kn; reverse thrust 0.25/0.5/1 x t_min calm | -13.0/-51.1/-201.6/-450.7 N; -4.00/-5.16/-6.61 kn (task sea -4.44/-5.31/-6.59) | same |

### D2. Calm equilibria, stability and branches (captive surge)

| id | plant | observation | values | file |
|---|---|---|---|---|
| D2.1 | v0, linear modes at fsolve equilibria, 0-35 kn | T_n/zeta of two heave-pitch pairs | 0 kn 2.05/0.000, 1.93/0.000; 1: 2.06/+0.022, 1.92/-0.018; 4: 2.11/+0.055, 1.88/-0.039; 8: 2.46/+0.132, 1.94/-0.079; 13: 3.49/+0.314, 2.15/-0.123; 20: 3.80/+0.404, 2.06/-0.123; 30: 4.20/+0.461, 2.02/-0.086; 35: 4.38/+0.434, 2.03/-0.033 (this "sunk" branch: z -0.07..-0.135 m, trim 2.4-3.0 deg at 10-35 kn) | audit.txt:15 |
| D2.2 | v0 | damping of least-damped pair with one term removed (-buoy/-mom/-cf/-uq) | 4 kn full -0.039, -buoy +0.436, -cf -0.053; 25 kn full -0.112, -buoy +0.550; D55 at 4/8/25 kn -6220/-12138/-35067 N m s | audit.txt (C, second table) |
| D2.3 | v0 planing branch (from captive run end state), 10-35 kn | z, trim, modes | 10 kn +0.011/5.45, 11.94 s/+0.975 & 1.39/+0.434; 12: +0.066/6.38, 1.19/+0.413; 16: +0.129/4.76, 1.09/+0.437; 20: +0.149/3.51, 1.00/+0.472; 25: +0.154/2.36, 1.38/+0.646; 35: +0.154/1.54, 2.23/+0.741; buoyancy share 0.63..0.13; chine-wetted stations 26..1 | audit_c2.txt:15 |
| D2.4 | v0 + complete D/Dt (design check), displacement branch | max Re(lambda) | 1 kn -0.092, 3: -0.276, 5: -0.453, 7: -0.638, 8: -1.10, 9: -0.99, 10: -2.19 ... 25: -3.65 | design_check.txt:8 |
| D2.5 | v0 sunk branch with/without complete D/Dt, 10-25 kn | max Re(lambda) | v0 terms +0.33..+0.38 (all speeds); with term: 10 kn moves to planing root (-2.19); 12-25 kn bow-down roots trim -15.1..-0.62 deg, max Re +0.13..+1.28 | design_check_sunk.txt |
| D2.6 | a_b sensitivity 10-20 kn (a_b 1/0.75/0.5) | trim, buoy/W, max Re | 10 kn 3.46/4.82/7.21 deg; 12 kn a_b 0.5 bow-down root -9.33 deg (+0.236); 14 kn a_b 0.75 -11.12 deg (+0.168); 20 kn 2.88/3.11/3.51 | design_check.txt:26 |
| D2.7 | closure variants (v0 C_V blend + dmx; a_b=1; a_b(eps) Maki; Saunders; + face force; + J 0.5 eps): branch lost at | 7.5 / 7.0 / 7.0 / 7.0 / 9.5 / 10.5 / 10.5 kn; multi-start roots listed per variant at 6-20 kn incl. bow-down roots (trim -15.1..-1.1 deg, max Re +0.13..+1.07); captive R with face force at 1-9 kn: -80.9..+478 N | review_physics.txt:1 |
| D2.8 | captive time runs (RK4 dt 5 ms, 40 s per speed, up then down, 6-16 kn), 5 closure variants | trim p-p and z p-p over last 10 s | all 0.000 (one 0.001); trim mean differs up vs down e.g. v0 13 kn 6.13 up / 6.38 down; 12 kn 6.38/6.22 | review_physics_time.txt |
| D2.9 | root census (deflated Newton 10x10 starts + 60x72 winding), v0 vs S1-as1, 1-40 kn | stable root | v0: no stable root at 1-9 kn (only unstable bow-up root trim ~1.88-2.25), none at 27/29/31 kn, two stable roots at 9.5 kn and 37-40 kn (second at trim ~2.9 deg z -0.04..-0.055); winding cells away from roots 1-2 at 10-39 kn. as1: exactly one stable root at every speed, 0 extra zeros | s1/g_census.txt |
| D2.10 | P1a / later presets | census 2-40 kn (10 speeds) | one stable root each, 0 extra zeros; stepped sweep up-down trim difference 0.0000 deg; largest trim change between speeds 0.957 deg | p1a/review_fix/checks_hullb.json |
| D2.11 | as1 S1 running roots 14-40 kn (48 st) | z, trim | 14: +0.1095/5.779; 16: +0.1290/4.765; 20: +0.1487/3.507; 25: +0.1551/2.308; 30: +0.1659/1.867; 35: +0.1803/1.704; 40: +0.1917/1.572 | s1/d_planing.txt; s1/g_census.txt |
| D2.12 | v0 running attitude and calm resistance (validity table) 1-40 kn | trim, rise, wetted length, R, pressure/friction/air, buoyancy share | e.g. 6 kn 2.19/-0.059/5.40/844 N (544/175/10) 1.25; 10 kn 5.45/0.011/5.40/1709 (1307/374/28) 0.63; 13: 6.38/0.092/4.16/2048; 20: 3.51/0.149/4.95/1923 (841/973/110) 0.15; 25: 2.36/0.154/5.40/2311; 30: 1.71/0.151/5.40/2981; 40: 1.43/0.168/5.40/4422 | validity.txt:4 |
| D2.13 | v0 calm resistance decreasing with speed | interval | 1.9-28.7 kn (19 of 0.25 m/s grid points) | validity.txt:29 |

### D3. Decay, self-oscillation, limit cycles

| id | plant | observation | values | file |
|---|---|---|---|---|
| D3.1 | v0, heave free decay from +1 mm / +5 cm (captive) | peak ratios | 4 kn 1 mm: 0.893/0.92/1.091; 5 cm: 0.794/0.759/0.874; 6 kn 1 mm: 0.936/1.253/1.508; 5 cm: 0.849/1.181/1.437; 25 kn 1 mm: 0.004; 5 cm: 0.404/0.511 | audit_decay.txt |
| D3.2 | v0 validity V4 (dz 0.05/0.15 m) | damped period s / zeta | 0 kn 2.06/0.011, 2.08/0.034; 3 kn 2.00/0.039, 1.95/0.127; 6 kn 1.99/-0.081, 2.37/0.018; 10 & 15 kn no second peak; 25 kn 1.47/0.140, 1.46/0.163 | validity.txt:49 |
| D3.3 | v0, S1(b) 120 s captive from Newton root + dz (1 mm / 5 cm), 1-10 kn every 0.5 kn | outcome; last-10-s z p-p, trim p-p | 0 of 38 runs decay; e.g. 1 kn z p-p 0.071 m trim p-p 2.18 deg; 3 kn 0.115 m/1.68 deg; 6 kn 0.117/0.80; 8.5 kn 0.177/1.24; 9 kn 0.306 m/3.25 deg; 9.5 and 10 kn settle on another steady state (+5.2 / +8.1 cm) | s1/b_decay.txt |
| D3.4 | as1 S1(b) same protocol | outcome | 37 of 38 decay; 9.5 kn 5 cm settles 3.7 mm away | s1/b_decay.txt |
| D3.5 | v0 lowspeed (captive, 40 s from cached attitude, last 15 s) | heave range / trim range | 4 kn 0.109 m / 1.37-2.55 deg; 6: 0.119 / 1.68-2.50; 8: 0.155 / 1.76-2.82; 9: 0.307 / 0.98-4.33; 10-12 kn 0 | wave_regimes/lowspeed.txt |
| D3.6 | v0 free running constant throttle 120 s (last 60 s) | speed, heave range, trim range, a_cg max | f 0.03: 5.01 kn, 0.092 m, 0.73 deg, 0.013 g; 0.05: 6.88 kn, 0.127 m; 0.075: 8.61 kn, 0.172 m, 1.19 deg; 0.10: 9.18 kn (speed range 0.64), 0.297 m, 3.58 deg, 0.102 g; 0.15-0.25: all 0 | lowspeed.txt |
| D3.7 | v0 26-28-30 kn calm captive (high speed) | z p-p / trim p-p | 26 kn 7.9e-3 m / 0.201 deg (48 st), 1.3e-3/0.029 (192 st); 27 kn 0.028/0.596; 28 kn 0.0265/0.516 (all dt); 30 kn 0.0176/0.291; 32-40 kn 0 (48 st); as1 0 at all | s1/h_highspeed.txt |
| D3.8 | S1 extra strip force of complete D/Dt at the mean state | Fz / My | v0 26/28/30/32/35/40 kn: 328/645/1112/1640/1617/1579 N; as1: 292/495/490/485/477/466 N | h_highspeed.txt |
| D3.9 | v0 older coefficients k_a 0.6 | calm self-oscillation at Fridsma case A | doc only IL H7 |

### D4. Slow speed ramps (captive, 0 -> 35 -> 0 kn, 0.1 and 0.05 kn/s)

| id | plant | observation | values | file |
|---|---|---|---|---|
| D4.1 | v0 0.1 kn/s | trim up vs down, largest trim jump per 0.5 kn | 0 kn up 1.92 / down 2.81; 10 kn 2.64 / 5.50; up jump 4.00 deg (10.62-11.12 kn); down 3.83 deg (9.40-8.90 kn); R at 0-9 kn up 433-759 N scattered (e.g. 3 kn 513, 5 kn 433, 7 kn 724) | s1/c_sweeps.txt |
| D4.2 | as1 S1 0.1 kn/s | same | largest up-down 0.277 deg (16 kn); largest jump 0.438 / 0.468 deg; R 0-35 kn 459 -> 3281 N monotone except 13-17 kn (2045 -> 1867) | s1/c_sweeps.txt |
| D4.3 | v0 0.05 kn/s | same | 10 kn up 2.95 / down 5.47 (diff -2.52); 0 kn down 2.98 | s1/c_sweeps.txt |

### D5. Thrust map, constant-throttle equilibria, surge

| id | plant | observation | values | file |
|---|---|---|---|---|
| D5.1 | v0 thrust vs calm resistance crossings | stable equilibria per throttle f | 0.02: 4.6 kn; 0.05: 5.6 and 8.1; 0.075: 8.6 and 9.3; 0.10: 9.4; 0.15: 9.6; 0.20: 10.8; 0.25: 18.5; 0.30: 22.2; 0.45: 27.1; 0.60: 31.6; 0.80: 38.1; 1.0: 43.9 kn | validity.txt:29 |
| D5.2 | v0 equilibrium table 8-21 kn | R, f needed, dR/du, dT/du | 10 kn 1706 N 0.177, +1107/-86; 13: 1995, 0.233, +17; 14: 1975, 0.237, -152 (not stable); 16: 1872, 0.235, -32; 20: 1921, 0.268, +83 | lowspeed.txt (C) |
| D5.3 | P1a R2: free-surge ramp c 0.15->0.35->0.15 at 2e-4/s | largest up-down | -3.73 kn at c 0.240; largest jump 1.92 kn per 0.005 | p1a/review_fix/checks_r2.json |
| D5.4 | P1a R2 fine | three-equilibrium window; 1500 s hold at c 0.24 | c 0.23415-0.23420 (6 values), positive slope 13.00-13.65 kn (+3.39 N s/m max); both starts end 16.707 kn = static root 16.704 kn | p1a/review_fix/rereg/checks_r2fine.json |
| D5.5 | v0 calm thrust response (wave_phase config, 20/26 kn) | hold command (frac t_max), T_surge | 12 kn 0.228, 5.8 s; 15 kn 0.238, (dR/du -92 => T 1.5e12); 18: 0.248, 13.2 s; 20: 0.268, 9.0 s; 24: 0.338, 5.0 s; 26: 0.396, 2.9 s; 28: 0.471, 7.2 s; 30: 0.513, 2.7 s; 32: 0.611, 3.8 s | wave_phase/report.txt (propulsion table) |
| D5.6 | v0 calm, sinusoidal / square throttle modulation, periods 1.5-20 s at 20 and 26 kn | u amplitude, pitch, heave, a_cg | 20 kn square P 2 s cmd 2127 N: u 0.230 m/s, pitch 0.232 deg, heave 0.749 cm, a_cg 0.0077 g; P 10 s: u 2.34 m/s, pitch 1.64 deg; 26 kn square P 2 s: u 0.348, pitch 0.329 deg, heave 1.37 cm, a_cg 0.014 g; 26 kn constant: u p2p 0.0156 m/s, pitch p2p 0.2045 deg, a_cg p2p 0.0165 g | wave_phase/report.txt |
| D5.7 | jet lag model | gain / phase of a sinusoidal thrust command at encounter (tau 0.6 s, delay 0.1 s) | e.g. head seas tp5 30.5 kn |we| 4.92: gain 0.32, lag 100 deg; following tp8 31.5 kn |we| 0.46: 0.96, 18 deg | wave_regimes/scan_report.txt (section 4) |

### D6. Roll stiffness and lateral modes vs speed

| id | plant | observation | values | file |
|---|---|---|---|---|
| D6.1 | v0 sunk-branch sections (audit) | hydrostatic roll stiffness of immersed sections x a_b / plant W GM 16698 | 0 kn 1.00; 5: 1.01; 10: 0.95; 16: 0.77; 20: 0.76; 25: 0.74; 35: 0.66 | audit.txt:77 |
| D6.2 | v0 planing branch sections | same | 10 kn 0.55; 16: 0.17; 20: 0.09; 25: 0.06; 35: 0.04 (immersed vol 1.27..0.35 m3) | audit_c2.txt:44 |
| D6.3 | hh / hx static roll stiffness K_phi/(W GM) | 1/3/6/10/15/20/25/30/35/40 kn | hh 0.998/0.986/1.039/0.826/0.419/0.183/0.109/0.074/0.047/0.031; hx 0.999/0.990/1.057/0.967/0.863/1.050/0.683/0.550/0.488/0.441 | s5a/r2/checks_h8_hx.json |
| D6.4 | hx lateral eigenvalues | max Re excess over P1c | +0.0274 1/s; 1/3/6 kn: P1c +0.0012/-0.0046/-0.1531, hx +0.0118/+0.0274/+0.0087; a22 x1.55: +0.0120/+0.0301/+0.0226; x1.38: +0.0132/+0.0334/+0.0277 | s5a/r2/ga4.txt |
| D6.5 | capsize angle | from vanishing angle -2 deg: no capsize (max 101.3); +2 deg: capsize (105.3); 25 kn from 15 deg heel: back within 2 deg in 0.7 s (hh) / 0.4 s (hx) | s5a/r2/checks_h9_*.json |
| D6.6 | p1b lateral open loop (Fn_L 1.2, nozzle sinusoid 0.6 x max, first 6 s) | max heel / yaw rate / sway | P1a task-like sea 10.1 deg/72.7 deg/s/4.28 m/s; candidate 73.7/34.3/1.82; +lateral_phi0 calm 80.0/32.6 (dt 0.005: 80.0); a24 = 0: 61.7; no heel derivatives 62.1; both removed 3.0/15.5 | p1b/diag.txt |

### D7. Turning

| id | plant | values | file |
|---|---|---|---|
| D7.1 | v0, 25 kn, nozzle 5/10/20 deg | yaw rate 5.3/8.6/11.9 deg/s, radius 24.8/14.5/9.0 L, drift 4.6/7.0/9.3 deg, speed 24.2/22.6/19.3 kn; 20 s after centring 0.000 deg/s | studies.exp_planing.log:16 |
| D7.2 | v0 low throttle, nozzle hard over | f 0.05/0.10/0.15/0.20: straight 6.9/9.2/9.5/10.7 kn -> turning 2.7/5.9/8.5/9.5 kn, yaw 19.3/20.5/19.0/16.6 deg/s | validity.txt:63 |
| D7.3 | as1 p1c / hh / hx steady turns 15/25/35 kn nozzle 10/20 deg | heel deg (yaw deg/s): p1c 35 kn 20 deg +2.97 (-20.9); hh +11.03 (-22.0); hx -1.93 (-6.93); hx 35 kn heel outward in 4 of 4 cases | s5a/r2/h10.txt |
| D7.4 | planing_b turning table | 10.6 kn 13.5 deg/s; 20.6: 9.9; 24.5: 12.7; 26.7: 15.5; 28.3: 18.0; 29.0: 19.9 | p0/hullb.txt |
| D7.5 | v0 older coefficients | 25 kn turning 3-10 deg/s (k_a 0.6), 5-12 deg/s (Zarnick) | doc only IL H6/H7 |

### D8. Head seas, fixed throttle; regular head waves

| id | plant | values | file |
|---|---|---|---|
| D8.1 | v0, Hs 1.0 Tp 5 head seas, throttle at calm 25 kn setting, 90 s, dt 0.005/0.01/0.02 | speed 19.2/19.6/19.3 kn; trim 3.43±3.08 / 3.62±3.01 / 3.51±3.20; heave sd 0.376/0.373/0.392 m; CG p99 1.68/1.77/1.65 g; bow p99 4.26/4.11/3.87 g; prime lost 16.0/15.6/17.0 % | studies.exp_planing.log:21 |
| D8.2 | v0, 25 kn towed, H 0.1 m regular head waves lam/L 1/2/3/4/6/10 | heave/a 0.13/0.40/0.59/1.03/1.63/1.13; pitch/ka 0.06/0.16/0.36/1.05/2.08/1.35; CG g 0.17/0.19/0.14/0.12/0.12/0.04; bow g 0.35/0.35/0.27/0.31/0.15/0.05 | studies.exp_planing.log:27 |
| D8.3 | v0, eval sea, constant thrust 0.3-1.0, legs 0/1, n 8 | leg 0: 17.6/20.7/22.6/24.2/25.8/27.5/29.2/30.3 kn; A1/10 4.10/5.16/6.13/6.84/7.21/7.60/7.75/7.92 g; >=7 g/h 54/103/220/283/332/449/404/467; bow wet/h 18/49/67/67/99/139/171/153; intake out 0.04..0.19; leg 1: 18.9..30.8 kn, A1/10 3.43..6.94 g | wave_phase/report.txt ("evaluation sea, constant thrust") |
| D8.4 | same, 846 peaks >= 7 g | fully airborne within 1.5 s before: 1.00; airborne duration median 0.46 s; slam force/weight median 8.07; w median -1.90 m/s; wave phase at CG: [-180,-135) 405, [-135,-90) 241, [-90,-45) 71, [135,180) 114, others <= 8 | wave_phase/report.txt |
| D8.5 | ratio of throttle-driven to wave-driven response (square throttle at nearest period vs wave band) | pitch 0.060-0.111, heave 0.015-0.029, a_cg 0.014-0.028 | wave_phase/report.txt (spectra) |
| D8.6 | >= 3 g peaks n 3230 | within-run corr peak vs speed -0.10 (slope -0.17 g per m/s); across run means +0.48; take-off corr with air time +0.36 | wave_phase/report.txt |

### D9. Regime scan (v0 PlaningVesselGZ, constant throttle + heading autopilot + SafeMode, 400 runs, 4 seeds, 100 s stats)
Source: wave_regimes/scan_report.txt. Columns: kn made good; A1/10 g; >=7 g/h; bow-wet/h; burial/h; max heading dev deg; capsizes.

| sea, chi | f 0.10 | f 0.30 | f 0.60 | f 1.00 |
|---|---|---|---|---|
| tp5 head (0) | 7.9; 1.33; 0; 0; 0; 19; 0 | 18.0; 3.88; 38; 19; 0; 11; 0 | 23.5; 6.50; 227; 123; 28; 38; 0 | 30.5; 8.17; 548; 132; 28; 32; 0 |
| tp5 45 | 8.1; 1.08; 0; 0; 0; 35 | 19.0; 3.29; 38; 19; 0; 15 | 25.2; 5.31; 132; 38; 9; 38 | 30.8; 7.26; 331; 123; 38; 47 |
| tp5 beam (90) | 8.4; 0.59; 0; 0; 0; 11 | 19.2; 1.56; 0; 0; 0; 33 | 25.9; 3.67; 9; 132; 38; 35 | 30.4; 4.38; 38; 95; 28; 60 |
| tp5 135 | 7.8; 0.44; 0; 0; 0; 23 | 17.9; 2.47; 0; 123; 38; 34 | 24.0; 3.69; 9; 397; 284; 46 | 29.3; 4.04; 9; 407; 284; 66 |
| tp5 following (180) | 9.8; 1.11; 0; 9; 9; 14 | 16.9; 2.57; 0; 199; 123; 20 | 24.1; 4.23; 9; 586; 340; 59 | 28.3; 5.12; 38; 614; 388; 91 |
| tp8 head | 9.0; 0.70; 0; 0; 0; 3 | 21.1; 1.24; 0; 0; 0; 3 | 28.5; 2.69; 19; 0; 0; 32 | 34.6; 3.82; 9; 9; 0; 42 |
| tp8 following | 9.3; 0.00; 0; 0; 0; 2 | 20.6; 0.65; 0; 0; 0; 2 | 27.5; 2.27; 0; 9; 0; 45 | — |

Other recorded columns (same file): max roll 2-26 deg, SafeMode entries up to 8 (tp5 beam), y-fail runs up to 3, capsizes 0 in all 400 runs, pump prime deficit up to 0.40, surf-riding share up to 0.34 (tp8 180 f 0.30), measured encounter frequency vs theory (section 1; one row not monotone). By speed band (section 5): tp5 head >=18 kn g7/h 298; tp5 following >=18 kn wet/h 548, bur/h 347.

### D10. Events within runs (v0, wave_regimes/events.txt; oracle_timing.txt)

| id | observation | values |
|---|---|---|
| D10.1 | yaw-rate onsets > 20 deg/s per run (tp5 head f 0.6 seed 0; f 1.0; seed 1; 45; 180; tp8 180; 135; 180 f 0.2; beam) | 17; 37; 22; 33; 20; 30; 36; 3; 24. Share near a >= 2 g impact 0.59/0.51/0.50/0.33/0.35/0.13/0.19/0.00/0.29 vs chance 0.41/0.57/0.40/0.33/0.28/0.04/0.21/0.00/0.15 |
| D10.2 | at onsets: yaw added inertia "now" (instantaneous strip value) vs "used" (constant 381 kg m2) | 45 .. 17549 kg m2; Izz total constant 2932 kg m2; median |N| kN m munk 0.71-2.37, transom 0.83-2.55, jet 0-0.82 |
| D10.3 | bow keel immersion median / p99 | 0.04-0.11 m / 0.82-2.11 m |
| D10.4 | bow burial timing (24 constant-throttle runs, 165 s): time since bow's last upward crest crossing, median | tp5_180 f 0.30/0.45/0.60: 8.10/4.95/3.25 s (n 13/27/36); tp5_135: 4.25/4.73/6.24 s; share < 0.7 s 0.00-0.15; Tz 2.69-5.24 s |

### D11. Robustness runs (s1/f_robust.txt, s1/w_bow_out.txt)

| id | observation | values |
|---|---|---|
| D11.1 | basin: captive calm from root ±0.2 m × ±5 deg, 16-35 kn | v0, as1: 0 diverged; v0 30 kn 8 of 8 not back within 1 mm/0.01 deg in 40 s (end 0.0077-0.0093); "ungated" variant diverged 1-2 of 8 at 25-35 kn (min trim -29.9 deg) |
| D11.2 | following seas (task sea turned 180 deg), Hs x1/x1.5, open loop 240 s, 12/20/25/30 kn | none diverged or capsized; v0 Hs x1 min trim -5.8..-9.6 deg, min z -1.17..-1.64 m; as1 -2.7..-6.4 deg, -0.49..-1.07 m; Hs x1.5 v0 down to -16.9 deg/-2.54 m; as1 down to -20.1 deg/-1.99 m |
| D11.3 | bow rising out, captive regular head waves 8-14 kn, lam/L 2 H 0.5 m / lam/L 3 H 0.8 m | bow-out share v0 38-50 %, as1 48-55 %, ungated 39-51 %; trim range v0 e.g. 8 kn lam/L 3 -5.31..+15.84 deg, as1 -6.01..+17.29; ungated suction force -4406..-8846 N (v0, as1 0) |
| D11.4 | F7 following seas Hs x1/x1.5, 12-30 kn, P1a + fk_exact (+orbital) | no divergence, no capsize (p1b/review_fix/checks_f7.json) |
| D11.5 | v0 27-29 kn head seas 1 m episode (replay of a learning episode, wave seed 21008) | airborne ~0.7 s, CG 1.4 m above rest, bow keel ~2 m submerged, yaw rate 0 -> -0.9 -> +3.1 rad/s -> non-finite at 70.42 s; lateral rate x dt 1.9 then 3.4; after sub-stepping: yaw deviation ~12 deg, peak yaw rate ~56 deg/s, CG ~1.8 g, speed 27.5 -> ~20 kn; 18 head-sea runs 25-30 kn: all finite, 4 of 18 had 3-14 steps with rate x dt > 1.8, all 18 exceeded 1 at least once (max 2.15) — doc only, IL K (line 1232) |

### D12. Step cost (single core, task sea, ms per 0.02 s step)

| plant | ms | file |
|---|---|---|
| v0 scarab195 | 1.67-1.97 (median runs) | p0/steptime.txt; p1a/review_fix/final/checks_golden_steptime.json |
| P1a scarab / planing_b (load-cancelled) | 2.29-2.68 / 2.50-2.81 | same |
| P1a + fast_path | 2.40-2.54 | p1c/final/checks_e1.json, p1c/review/checks_e1.json |
| as1 preset (P1a + fk_exact + fast_path) | 5.64-5.73 | same |
| P1a + fk_exact / + orbital / + wagner_entry | 5.62-5.72 / 7.23-7.84 / 3.35-4.59 | p1b/review_fix/checks_e1.json; p1c/w*/checks_e1.json |
| p1b candidate (all switches) | 6.27 (scarab), 7.28 (planing_b) | p1b/checks_steptime.json |
| as1 + wagner_entry | 7.42 | p1c/w4/checks_e1.json |
| controllers in final_eval meta7 (s per control step) | hand 0.03, m0 0.07, cat_p_* 3.35-6.20 | final_eval/meta7/report.txt |

### D13. Froude-scaled twin hull (principles hull_swap), steady points

| Fn | base trim deg | twin - base trim deg | R/W base | R/W rel change |
|---|---|---|---|---|
| 0.824 | 6.337 | -0.0031 | 0.1444 | -3.2e-4 |
| 1.099 | 5.073 | -0.0035 | 0.1384 | -2.5e-4 |
| 1.374 | 3.557 | -0.0029 | 0.1378 | +9.5e-5 |
| 1.767 | 2.362 | -0.0033 | 0.1683 | +1.0e-3 |
| 2.061 | 1.742 | +0.351 | 0.2059 | +5.7e-2 (pressure +29 %) |
Source: principles/hull_swap/compare.json ("air" case). Identified reduced-model coefficients differ by up to 4.8 % (tau_r).

---------------------------------------------------------------------------------------------------------------------

## E. Closed-loop and control results (plant = v0 PlaningVesselGZ unless stated)

| id | setting | values | file |
|---|---|---|---|
| E1 | final_eval meta7, eval sea, 16 missions (hand, m0) / 8 (cat_*) | hand 21.5±0.5 kn, A1/10 5.10 g, >=7 g/h 111.3, >=10 g 3, bow wet/h 44.9, burial 3, max roll 18.1 deg, intake/h 793; m0 27.9±0.9 kn, 7.08 g, 298.8, 36, 115.2, 18, 19.8 deg, 1475; cat_p_prior 19.8±2.0, 4.37, 74.2, 3, 15.6, 1, 21.2 deg; cat_p_bayes 20.0, 4.54, 89.8; cat_p_bayes_carry 20.2, 4.59, 85.9; cat_p_online 19.9, 4.21, 66.4; capsizes 0 everywhere (95 % bound < 6 or < 12 per hour); A1/10 at k 1.5 / 2.5: hand 3.94/6.31, m0 5.38/8.77 | final_eval/meta7/report.txt |
| E2 | wave_phase oracle, eval sea, 1056 missions, 66 settings, matched-speed deltas vs constant-throttle curve; leg 0 head | constant curve A1/10 4.05 (17.7 kn) .. 7.84 g (30.1 kn), slope 0.314 g/kn; families: all phase-locked (48) A1/10 +0.16 (t +1.86), >=7 g/h +6.3, wet/h -0.8; bang-bang (10) +0.15; actuator realised/commanded modulation 0.32, lag +101..+102 deg | wave_phase/oracle_report.txt:4 |
| E3 | same, leg 1 (45 deg) | all phase-locked A1/10 -0.01 (t -0.11), >=7 g/h -18.9 (t -1.57); pl phi0 in {0,+45} (chosen after seeing) -0.11 (t -1.59), >=7 g/h -31.0 (8/8 missions); modulation 0.36-0.37, lag +85..+86 deg; constant runs: >=7 g peaks on face 0.53, bow-wet on face 0.95 | oracle_report.txt:149 |
| E4 | wave_phase phase-locked bang-bang vs constant at same mean speed (report.txt) | lock0.7_0.1 A1/10 diff mean +0.077 (sd 0.56, n 64); lock1.0_0.4 +0.119 (sd 0.62, n 128); null sd of one paired diff 0.505 / 0.700 | wave_phase/report.txt |
| E5 | wave_regimes oracle, following/quartering seas, screen 548 runs + confirm 456 runs (seeds 4-11) | tp5_180 constant f 0.10-0.60: 10.3-23.8 kn, bur/h 9.5-350, wet/h 14-553, hdgX 9.9-38.8; families (confirm): tp5_180 pl bur/h -16.4 (t -1.4); tp5_135 bb bur/h -32.0 (t -1.1), hold -25.9; tp8_180 hold surfF -0.06 (t -4.5); best-setting z vs E[best of N] -1.3..-1.4: surfF z -3.0..-3.5, bur/h z -2.1 (tp5_135 bb), wet/h z -2.1..-2.6 | wave_regimes/oracle_report.txt:442 |
| E6 | wave_phase closed loop, thrust phase locking of controllers | cat_p_prior leg 0: thrust 0.413±0.264, depth 0.316 (perm null 95 % 0.491); hand 0.431±0.139, depth 0.065 (null 0.089), 1st harmonic 0.0082 (null 0.0070); m0 0.810±0.034, depth 0.006 (null 0.012); impacts >=3 g Rayleigh R 0.33 (hand) / 0.38 (m0), mean phase -169 / -144 deg | wave_phase/closedloop/analysis.txt |
| E7 | planners (open-loop plant truth, 24 moments) | a sinusoid best plan at 16 of 24 moments; in cat planner's own model sinusoid beats every constant 9 of 12; hand / m0 0 of 12 | wave_phase/planners_score.log |
| E8 | hull swap closed loop (n 14 seeds), score base / twin / chaos | hand 0.2702/0.2712/0.2725 (rms twin-base 0.050); m0 0.4440/0.4128/0.4715 (rms 0.146); acc_cg_p99 hand 1.397/1.410/1.395 | principles/hull_swap/seed_stats.json |
| E9 | hull acceptance gate H (closed loop mean speed / target, heading rms), six displacement hulls | Wigley 0.97, 1.1 deg; cat 0.98, 1.6; barge 0.88, 4.3; KVLCC2 0.89, 5.2; KCS 0.98, 1.8/1.9; wigley10_jet 0.98, 1.1 | doc only: IL G6, H3 (acceptance_*.json in _cache) |
| E10 | first planing closed loops (MPC thrust + autopilot, sea state 3) | full sim 21.9-22.6 kn, CG p99 1.2-1.4 g, heading rms 2-2.4 deg (k_a 0.6); after Zarnick coefficients: head 20.8 kn 1.8 g 4.4 deg; bow-quartering 22.3 kn 1.2 g 4.0 deg; low-fidelity world 25.3-25.4 kn, 1.1-2.4 g | doc only: IL H6, H7 |
| E11 | sim-to-real with wigley10_jet (displacement hull, jet), 16 target episodes | relative to A1: B2 -25 %±5, B2r -34 %±8, B2t -13 %±4, O -40 %±10; Spearman source vs target over 13 controllers -0.33 | doc only: IL H5 |

---------------------------------------------------------------------------------------------------------------------

## F. Reference data available in the repo vs not

| reference | in repo? | form / location | source type | uncertainty flag |
|---|---|---|---|---|
| Fridsma 1969 regular waves (H/b 0.111, cases A, B) | numbers only, typed in code | studies/exp_planing.py:192 | towing tank (measured) | transcription from a web-retrieved copy; original report (DL R-1275) not in repo |
| Zarnick 1978 2D+t results | numbers in code | exp_planing.py:203 | computation | digitised ±0.03/±0.02 g |
| Fridsma 1971 irregular | numbers in code | exp_planing.py:256 | towing tank | modal period not in repo's copy (Tp assumed) |
| Savitsky 1964 | equations coded (hydro/planing.py, hydro/rankine/planing_steady.py) | — | regression of tank data | original paper not re-read (B7); Chambliss & Boyd 1953, Shuford 1958 raw tables NOT transcribed |
| Scarab 195 speeds (Boating magazine), VM 18 brochure | numbers in code | exp_planing.py p2, exp_waterjet.py | full-scale magazine test / brochure | loads and rpm as quoted; part-throttle power from assumed BSFC |
| HamiltonJet 291 chart | numbers in code | exp_waterjet.py | maker chart | digitised by eye ±10 kgf |
| Alourdas (2016) NACA Model 29 example | numbers in code | exp_waterjet.py | tank + published Savitsky | — |
| Zhao-Faltinsen-Aarsnes wedge drop | numbers in code | exp_slam_wedge.py:48 | drop test | digitised by eye ±250 N; rig-deceleration cross-check +6/+11 % |
| Wedge entry (ZF 1993 similarity, MLM peak pressures, Wen 2022 constants) | CSV | data/external/wedge_entry/ | computation (similarity / models) | ZF 1993 original not obtained; secondary transcription (Iafrati & Battistin) |
| Wigley U. Tokyo (Kajitani 1983): C_W, sinkage, trim, wave profiles, C_WP | CSV + PDFs | data/external/wigley/ | towing tank | yang2000_* = our vector transcription; hn1988_p61_* = our scan reading; wavebem_wigleyDragExp* = WaveBEM digitisation, flagged as not reproducing scan markers at Fn < 0.3 |
| Journee 1992 Wigley I forward speed | CSV | data/external/wigley_journee/ | towing tank | secondary transcription from two papers' figures; original not retrieved |
| Wu & Eatock Taylor 1988 submerged sphere | CSV | data/external/wu_eatock_taylor_sphere/ | analytic (linear) | via Mei et al. 2020 tables (secondary); original not retrieved |
| Hulme 1982 hemisphere | table in code (analytic.HULME_HEAVE) | — | analytic | transcription cross-checked vs Capytaine (A' 0.64-0.69 %) |
| KCS T2015 cases 2.10/2.11 EFD + CFD time histories, IGES | yes | data/external/kcs_t2015, kcs_geometry | model test / CFD | — |
| KVLCC2 roll decay (SSPA 1:68), geometry, propeller | yes | data/external/kvlcc2_* | model test | — |
| Kawahara 2009 Ikeda, ITTC procedures | yes | data/external/ikeda, ittc | regression / procedure | fig 1 markers extracted from PDF vectors |
| B-series, MSS (Blendermann, Isherwood, Otter) | yes | data/external/bseries, mss | regressions | — |
| Planing-hull motion data at full scale (any boat) | NO | — | — | — |
| Planing-hull manoeuvring / turning data | NO ("no manoeuvring data exist for this boat", exp_planing.py:13) | — | — | — |
| Planing lateral / roll / yaw wave-force data | NO (only Capytaine computations) | — | — | — |
| Jet steering force and intake prime-loss data | NO (placeholders) | — | — | — |
| Scarab 195 hull lines | NO (four published items: deadrise 20 deg, 161 mm pump, 300 hp, 1340-1490 kg; rest assumed, hydro/hulls.py:131) | — | — | — |
| Scarab in waves (any measured motion) | NO | — | — | — |

---------------------------------------------------------------------------------------------------------------------

## G. Same quantity, different numbers in different sources

| quantity | values and where |
|---|---|
| scarab195 top speed (calm, full thrust) | 45.1 kn @1340 kg, 43.3 @1490 (exp_planing.log P2); 43.88 @1400, 48 st (validity.txt, s1/d); 44.34 @192 st; as1 47.66-47.68 (s1/d); 40.6-40.7 (older coefficients, IL H6); "+3-6 % above measured" (IL H7); measured 42.1-42.7 |
| v0 trim/R error vs Savitsky 20-40 kn | -1.04..-0.52 deg, -15..+0 % (exp_planing.log, Zarnick coeff.); -0.6..+0.4 deg, -3..+11 % (IL H6, k_a 0.6) |
| heave added mass at rest | v0 7877 kg; geometric half-beam 5923; exact 2D phi0 5247 / 5238; 3D BEM A33 at 3.89 rad/s 4248, A33_inf 4538 (memory fit); at 0.5 rad/s 11244 |
| heave natural frequency at rest | plant 3.07 rad/s (rest_rao) = period 2.04-2.06 s (audit A, validity V4); BEM 3.94 rad/s; Wagner vs m_0 differ 10.3 % (p1b 8c) |
| sway added mass | plant 210 kg (frozen 25 kn); slender m22 804 kg at rest; exact 2D phi0 340 / rigid lid 524; Capytaine 319 (omega inf) / 467 (omega 0); BEM probe A22 470-534 at 0.5-2.2 rad/s |
| yaw added inertia | plant 381 kg m2 (constant); slender A66 1782-3021 kg m2 (audit H); BEM A66 865-1014; instantaneous strip value during events 45-17549 kg m2 (events.txt) |
| roll frequency at rest | v0/hh 4.4475 rad/s (period 1.41 s); hx 4.1138; Capytaine sway-roll root 3.878 |
| calm resistance, low speed v0 | 6 kn 844 N (validity V1) vs 783.7 N (review_physics v0 C_V blend + dmx captive) vs 786 N (as1 S1 ramp); 7 kn 781 (validity) vs 768 (audit_transom); 8 kn 675 (validity) vs 693 (lowspeed C) — v0 values at 1-9 kn are snapshots of oscillating runs |
| D55 sign, 1-9 kn | v0 FD -1449..-12378 N m s; S1-as1 +4410..+71551; closed-form transom estimate +4041..+36373 (audit_c2) |
| as1 Fridsma P6 count | 22/44 (P1a, review_fix) vs 24/44 (p1b candidate); v0 25/44 |
| Rankine Wigley Fn 0.25 against U.T. C_WP 0.721e-3 | B4 C_near 0.710e-3 (0.985); B6 C_cut 1.039e-3 (1.441); against WaveBEM file 0.960e-3 (B4 ratio 0.740) |
| Rankine Savitsky-band share (beta 20) | B7 linear 0.25; B8 composite 0.0; B8 mlm_only 9/24 |
| Scarab measured speeds | 42.06 kn @1490 kg (exp_planing.py) vs 42.1 kn @1490 kg (exp_waterjet.py); 1340 vs 1339 kg |
| Fridsma case A calm trim | plant 5.50 deg (v0) / 5.43 (as1) vs measured 4.0; Garme stern correction 6.3 deg (doc only, H7) |
| Wigley C_W at Fn < 0.3 | WaveBEM CSV vs U.T. scan markers (hn1988) disagree (B4 review); U.T. curve via Yang 2000 is a third reading |

## H. Reference-uncertainty flags (summary)

- Fridsma 1969/1971 values: transcription typed into code from a web-retrieved copy; Zarnick digitised; Fridsma 1971 Tp unknown (assumed P-M, ±25 % shown).
- Savitsky: regression; original not re-read; raw prismatic-surface tables not available in repo.
- ZFA wedge drop test: digitised by eye (±250 N); cross-check differs by 6-11 %.
- HamiltonJet chart: digitised (±10 kgf); absolute size of the unit not public.
- Scarab / VM 18: magazine and brochure numbers; Scarab hull geometry beyond four items assumed.
- Journee, Wu & Eatock Taylor, Zhao & Faltinsen 1993: secondary transcriptions; originals not retrieved.
- Wigley C_W: three readings of U.T. data (WaveBEM CSV, Yang 2000 vectors, H&N scan by rank) that do not coincide at Fn < 0.3.
- KG 0.170 m assumed for Journee RAO comparisons (RAO5 changes up to ~5 % for KG 0.12-0.22 m).
- KVLCC2/KCS numbers in A11 are doc-only (IL F8/G4); KCS roll comparisons have measured higher harmonics 40-80 % of the first (F8).
