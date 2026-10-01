"""Step-by-step inside plans fixed in advance: truth vs rollout vs one-step
with the true history, for the nozzle gap and the yaw error. --split Cb
(default): the target's block starts (Cb_e0); Cbh: the target blocks with
the nozzle held over whole blocks (meta_step3 cbhold), --holdout = only the
episodes the oracle fine-tuning kept out (meta_step3.oracle_holdout); A /
At / B: <split>_branches.npz (plan 0 holds the command; no one-step pass,
the branches have no recorded history past k).

Rollouts by model3.rollout_kv (the evaluation's code path, rollout_core,
16 samples), so the numbers match eval. --model picks the checkpoint
(default model3.pt), --cache the cache directory (--cache-name: a cache
under studies/_cache, e.g. meta3).

Every block is classified by what the nozzle does over its WHOLE scored
window, steps 0..9 (diag_stall.block_kinds), before any rollout, over all
blocks of the split: steady = the nozzle command constant and the actual
nozzle on it throughout (start jump < 0.05, command step < 0.02, actual
within 0.05 of the command). The start-only test used before counted
blocks as steady whose command moved inside the window (make_plans' OU
wander in 3 of 4 plans, a step w.p. 1/2 at a random step), so the yaw
error of the actual nozzle catching up, which the model SEES (S[5:7]),
landed in the "steady" score. The rollout runs on n_pick random blocks
plus every steady block.

The stall is measured by diag_stall.stall_table, the one function
diag_m9_infamily uses too: jump events at any step with the actual nozzle
at rest before the jump, per jump-size bin; "stall gone" in a bin = the
rollout's open fraction at step 5 within 0.1 of the truth's.

Reading (stated before the M10 run, PRIOR_DERIVATION.md D7), per bin, on
the same measure for At (diag_m9_infamily --split At) and Cb (here):
  - stall still there on At: the model or the rollout, not the prior and
    not observation;
  - gone on At, not on Cb: a remaining difference in actuator shape (one
    candidate: At's delay is 0.12 s, the target's 0.1 s), not an
    observation gap;
  - gone on both, steady-block yaw still unlearned: OPEN. M10 did not
    change the velocity-channel prior, so this does not separate a prior
    gap in the velocity channels from an input the model cannot see.
    Oracle-data test first (meta_step3 cbhold + train_oracle, then here
    --split Cbh --holdout --model model3_oracle.pt): if the model trained
    on the target itself learns steady-block yaw, the inputs carry the
    information and the gap is in the prior (broaden the operator family,
    generically); only if even that model cannot, is it an observation
    question, and then an oracle-input test (roll rate, lateral
    acceleration, waves as perfect measurements in both worlds; a
    candidate counts only if it closes most of the gap).
  - fewer than 30 steady blocks: undecided (not "observation gap"); add
    Cbh blocks. --split A / At gives the in-family steady-block yaw level
    that unseen inputs cause there (a reference, not a decider: the
    target's unseen share, high-fidelity sea and roll, can be larger).

    python studies/diag_yaw_m8_blocks.py [--model model3_es.pt]
        [--cache-name meta3] [--split Cb|Cbh|A|At|B] [--holdout]"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, r"C:\Users\Administrator\Documents\usv-rl-mpc")
os.chdir(r"C:\Users\Administrator\Documents\usv-rl-mpc")
import numpy as np
import torch

from learn.meta import model3 as M
from learn.meta import relabel
from studies import diag_stall as DS

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="model3.pt")
ap.add_argument("--cache", default="studies/_cache/meta2/")
ap.add_argument("--cache-name", default=None,
                help="a cache under studies/_cache (overrides --cache)")
ap.add_argument("--split", default="Cb", choices=("Cb", "Cbh", "A", "At",
                                                  "B"))
ap.add_argument("--holdout", action="store_true",
                help="Cbh: only the episodes train_oracle kept out")
ap.add_argument("--n-pick", type=int, default=200,
                help="random blocks rolled out (plus every steady block)")
args = ap.parse_args()
C = os.path.join(args.cache if args.cache_name is None
                 else os.path.join("studies", "_cache", args.cache_name), "")
dev = M.device()
ck = torch.load(C + args.model, weights_only=False)
print(f"model {args.model}, split {args.split}"
      + (" (held-out episodes)" if args.holdout else ""))
net = M.Net().to(dev)
net.load_state_dict(ck["net"])
net.eval()
st = ck["stats"]
env = relabel._env()
dtc = env["dt"] * env["sub"]
J, HB = DS.J, M.HB
D = M.Data3(C, args.split, dev, stats=st)
blocks = args.split in ("Cb", "Cbh")
if blocks:
    E0 = np.load(C + f"{args.split}_e0.npz")["E0"]
    eps = range(D.n)
    if args.holdout:
        from studies.meta_step3 import oracle_holdout
        eps = oracle_holdout(D.n)
    ii = np.array([i for i in eps for _ in range(48, int(D.len[i]) - HB + 1,
                                                 HB)], int)
    kk = np.array([k for i in eps for k in range(48, int(D.len[i]) - HB + 1,
                                                 HB)], int)
    Uc = np.stack([D.Uraw[i, k:k + HB] for i, k in zip(ii, kk)]).astype(float)
    XSc = np.stack([D.XS[i, k:k + HB + 1] for i, k in zip(ii, kk)]).astype(
        float)
    Tc = np.stack([E0[i, k:k + HB] for i, k in zip(ii, kk)]).astype(float)
else:
    b = np.load(C + f"{args.split}_branches.npz")
    N, P = b["U"].shape[:2]
    ii, kk = np.repeat(b["ep"], P).astype(int), np.repeat(b["k"], P).astype(
        int)
    Uc = b["U"].reshape(N * P, HB, 2).astype(float)
    XSc = b["XS"].reshape(N * P, HB + 1, 14).astype(float)
    Tc = b["E0"].reshape(N * P, HB, -1).astype(float)
act = XSc[:, :, 13] / env["rud_max"]                 # (n, HB + 1)
u_prev = D.Uraw[ii, np.maximum(kk - 1, 0), 1].astype(float)
kinds = DS.block_kinds(Uc, act)
n_all = len(ii)
print(f"{n_all} blocks; over steps 0..{J - 1} (all blocks): steady "
      f"{int(kinds['steady'].sum())}, start jump > 0.15 then held "
      f"{int(kinds['start_held'].sum())}, start jump with the command moving "
      f"{int(kinds['start_moving'].sum())}, other {int(kinds['other'].sum())}"
      f" (start jump < 0.05 alone, the old 'steady': "
      f"{int((kinds['g0'] < 0.05).sum())})")

# rollouts on n_pick random blocks plus every steady one
rng = np.random.default_rng(1)
pick = set(rng.choice(n_all, min(args.n_pick, n_all), replace=False).tolist())
pick = np.array(sorted(pick | set(np.flatnonzero(kinds["steady"]).tolist())))
gen = torch.Generator(device=dev).manual_seed(0)
R = np.zeros((len(pick), 16, HB, 2))           # rollout samples, [9, 2]
O = np.zeros((len(pick), J, 2))                # one-step means, [9, 2]
with torch.no_grad():
    for s0 in range(0, len(pick), 64):
        r = pick[s0:s0 + 64]
        rows = np.arange(s0, s0 + len(r))
        s, _ = M.rollout_kv(net, D, ii[r], kk[r], Uc[r], XSc[r, 0], env,
                            n_samp=16, gen=gen)
        R[rows] = (s.cpu().numpy() * st["e_sd"])[..., [9, 2]]
        if blocks:
            for j in range(J):
                s1, _ = M.rollout_kv(net, D, ii[r], kk[r] + j,
                                     Uc[r, j:j + 1], XSc[r, j], env,
                                     n_samp=16, gen=gen)
                O[rows, j] = (s1.cpu().numpy() * st["e_sd"])[:, :, 0][
                    ..., [9, 2]].mean(1)
T = Tc[pick][:, :, [9, 2]]
K = {k: v[pick] for k, v in kinds.items()}
Rm = R[:, :, :J].mean(1)
print(f"rolled out {len(pick)} blocks (i, k stored per row): "
      f"{int(K['steady'].sum())} steady, "
      f"{int(K['start_held'].sum() + K['start_moving'].sum())} with a start "
      "jump > 0.15")

big = K["start_held"] | K["start_moving"]
sg = np.sign(Uc[pick, 0, 1] - act[pick, 0])
labs = (("truth   ", T[:, :J]), ("rollout ", Rm)) + (
    (("one-step", O),) if blocks else ())
for nm, c in (("nozzle gap (x jump sign)", 0), ("yaw error (x jump sign)", 1)):
    if not big.any():
        break
    print(f"\n{nm}; mean over blocks with a start jump > 0.15 "
          f"({int(big.sum())}; {int(K['start_moving'].sum())} of them with "
          f"the command moving afterwards), step 0..{J - 1}")
    for lab, A in labs:
        v = (A[big, :, c] * sg[big, None]).mean(0)
        print(f"  {lab} " + " ".join(f"{x:+.2f}" for x in v))

# the stall, the same measure as diag_m9_infamily's
print()
ev = DS.jump_events(Uc[pick], act[pick], u_prev[pick])
DS.stall_table(ev, Tc[pick][:, :, 9], R[..., 0], dtc,
               f"{args.split} ({args.model})")

print(f"\nyaw error by block kind, steps 0..{J - 1} (skill = MSE of the "
      "prediction / MSE of the truth; 1 = no better than predicting 0; "
      "[90% bootstrap over blocks])")
for kind, what in (("steady", "steady (command held, actual on it)"),
                   ("start_held", "start jump > 0.15, then held"),
                   ("start_moving", "start jump > 0.15, command moving"),
                   ("other", "other (command moving inside)")):
    m = K[kind]
    if not m.any():
        print(f"  {what:<38} (0 blocks)")
        continue
    for lab, A in labs[1:]:
        sk, cc, lo, hi = DS.yaw_skill(A[m, :, 1], T[m, :J, 1])
        print(f"  {what:<38} ({int(m.sum())} blocks) {lab} skill {sk:.2f} "
              f"[{lo:.2f}, {hi:.2f}], corr {cc:+.2f}")
n_st = int(K["steady"].sum())
if n_st < DS.STEADY_MIN:
    print(f"  -> {n_st} steady blocks < {DS.STEADY_MIN}: the steady-block "
          "branch is UNDECIDED (not an observation gap); add blocks with the "
          "nozzle held (meta_step3 --phase cbhold, then --split Cbh)")
else:
    print(f"  -> the steady-block reading rests on {n_st} blocks")

# eval3's actuator-channel skill (the second like-for-like stall check)
tag = os.path.basename(args.model)[len("model3"):-len(".pt")]
fe = C + f"eval3{tag}.pkl"
if os.path.exists(fe):
    res = pickle.load(open(fe, "rb"))
    print(f"\n{os.path.basename(fe)}: actuator channels [thrust nozzle], "
          "multi-step skill, steps 0 / 1-4 / 5-23")
    for sp in ("A", "At", "B", "Cb", "Cbh"):
        if (sp, "multi") in res:
            sk = res[(sp, "multi")]["skill"]
            print(f"  {sp:<4} " + ", ".join(
                " ".join(f"{x:.2f}" for x in sk[a:b_, 8:10].mean(0))
                for a, b_ in ((0, 1), (1, 5), (5, sk.shape[0]))))
