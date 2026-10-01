#!/usr/bin/env python3
"""
Does model3.pt's rollout stall on a nozzle reversal INSIDE the training
family (split A: source actuators) or on the fixed target-shaped actuators
(--split At, A's episodes; M10's test)? M8 saw the stall on the target
(Cb), whose nozzle has a delay and a smoothing lag that the old source
actuators never have.

Branch files <split>_wbranches (wide plans: full-range steps) and
<split>_branches (make_plans, the same plans as the Cb blocks'), where they
exist. Rolled out by model3.rollout_core, 16 samples. The stall is measured
by diag_stall.stall_table, the ONE measure diag_yaw_m8_blocks uses for Cb:
jump events at any step j0 with the actual nozzle at rest before the jump,
binned 0.15-0.5 / 0.5-0.8 / > 0.8 of range; the gap (sign aligned) over
j0 .. j0 + 9, truth vs rollout, and the open fraction at steps 3, 5, 9.
"Stall gone" in a bin = the rollout's open fraction at step 5 within 0.1
of the truth's; PRIOR_DERIVATION.md D7 reads the branches per bin, on this
measure for At and Cb. For split A also per source nozzle kind: old meta
(meta2) rate limited or ideal; M10 meta (meta3) by the first of delay > 0,
response (tau) > 0, rate limit > 0, else ideal.

    python studies/diag_m9_infamily.py [--model model3.pt] [--cache-name meta3]
        [--split At]
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from learn.meta import model3 as M
from learn.meta import relabel
from studies import diag_stall as DS

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="model3.pt")
ap.add_argument("--cache-name", default="meta2")
ap.add_argument("--split", default="A", choices=("A", "At"))
args = ap.parse_args()
C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache",
                 args.cache_name)
dev = M.device()
ck = torch.load(os.path.join(C, args.model), weights_only=False)
net = M.Net().to(dev)
net.load_state_dict(ck["net"])
net.eval()
st = ck["stats"]
env = relabel._env()
D = M.Data3(C, args.split, dev, stats=st)
meta = pickle.load(open(os.path.join(C, f"{args.split}_meta.pkl"), "rb"))
dtc = env["dt"] * env["sub"]


def nozzle_kind(act):
    """The source nozzle's kind from an episode's meta act."""
    act = act or {}
    if act.get("act_family", 0.0) > 0.5:
        for k, nm in (("noz_delay", "delay"), ("noz_tau", "response lag"),
                      ("noz_rate", "rate limit")):
            if act.get(k, 0.0) > 0:
                return f"M10 nozzle with {nm}"
        return "ideal"
    return ("rate-limited source nozzle" if act.get("rud_rate", 0.0) > 0
            else "ideal")


def one_file(name):
    """Roll out every plan of a branch file and print the stall table (and,
    for split A, per nozzle kind)."""
    f = os.path.join(C, f"{name}.npz")
    if not os.path.exists(f):
        print(f"{name}: not built")
        return
    b = np.load(f)
    N, P, H, _ = b["U"].shape
    gen = torch.Generator(device=dev).manual_seed(0)
    S = 16
    E9 = np.zeros((N, P, S, H))
    for s0 in range(0, N, 16):
        r = np.arange(s0, min(N, s0 + 16))
        e, _, _ = M.rollout_core(net, D, b["ep"][r], b["k"][r],
                                 b["U"][r].astype(float),
                                 b["XS"][r][:, 0, 0].astype(float), env, S,
                                 gen=gen)
        E9[r] = (e.cpu().numpy() * st["e_sd"])[..., 9]
    U = b["U"].reshape(N * P, H, 2).astype(float)
    act = b["XS"][..., 13].reshape(N * P, H + 1) / env["rud_max"]
    u_prev = np.repeat(D.Uraw[b["ep"], np.maximum(b["k"] - 1, 0), 1], P)
    T9 = b["E0"][..., 9].reshape(N * P, H)
    E9 = E9.reshape(N * P, S, H)
    ev = DS.jump_events(U, act, u_prev)
    DS.stall_table(ev, T9, E9, dtc, f"{name} ({args.model}, all plans)")
    kind = np.repeat([nozzle_kind(meta[int(i)].get("act")) for i in b["ep"]],
                     P)
    if len(set(kind)) > 1:
        for nm in sorted(set(kind)):
            DS.stall_table([e for e in ev if kind[e[0]] == nm], T9, E9, dtc,
                           f"  {name}, {nm}")
    print()


for name in (f"{args.split}_wbranches", f"{args.split}_branches"):
    one_file(name)
