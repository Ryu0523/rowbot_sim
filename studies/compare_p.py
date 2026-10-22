#!/usr/bin/env python3
"""
One-step comparison of preview-network checkpoints (variant p) on the
splits of a cache (ROADMAP_2026-09-30 section 11.2: capacity): in-family
held-out A and B, the target-like actuators At, the target families C and
Cb (no training data from them). model_preview.eval_one_step_p at the
deployment convention (all tokens measured, forecast level --lam), every
step >= 40 with its recorded history, the same seeds for every checkpoint
(paired). Per split and checkpoint: skill per channel (MSE of the sample
mean / MSE of predicting 0; lower is better), coverage of the central 90 %,
and the MSE ratio to the first checkpoint.

    python -m studies.compare_p --ckpts \\
        d192=studies/_cache/meta7/model_p.pt,d384=studies/_cache/meta7/model_p_d384.pt
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))
import numpy as np   # noqa: E402
import torch   # noqa: E402

from learn.meta import model3 as M   # noqa: E402
from learn.meta import model_preview as MP   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CH = ("surge", "sway", "yaw", "heave", "pitch")


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-name", default="meta7")
    ap.add_argument("--ckpts", required=True,
                    help="comma list of label=path; the first is the "
                         "reference")
    ap.add_argument("--splits", default="A,B,At,C,Cb")
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--n-samp", type=int, default=64)
    args = ap.parse_args()
    cache = os.path.join(HERE, "_cache", args.cache_name)
    dev = M.device()
    cks = []
    for item in args.ckpts.split(","):
        lab, _, path = item.partition("=")
        ck = torch.load(path, map_location=dev, weights_only=False)
        cks.append((lab, ck, MP.load_variant(ck, dev)))
        log(f"{lab}: {path}, size {ck.get('arch') or 'default'}, "
            f"{sum(p.numel() for p in cks[-1][2].parameters())} parameters")
    stats = cks[0][1]["stats"]
    for lab, ck, _ in cks[1:]:
        if not np.allclose(ck["stats"]["e_sd"], stats["e_sd"]):
            raise SystemExit(f"{lab}: another error scale than "
                             f"{cks[0][0]}")
    ctx = min(int(ck.get("ctx", M.W_CTX)) for _, ck, _ in cks)
    log(f"window {ctx} tokens, forecast level {args.lam}")
    for split in args.splits.split(","):
        if not os.path.exists(os.path.join(cache, f"{split}.npz")):
            log(f"{split}: no such split, skipped")
            continue
        D = MP.DataP(cache, split, dev, stats=stats)
        res = {}
        for lab, ck, net in cks:
            res[lab] = MP.eval_one_step_p(net, D, None, lam=args.lam, hp=1,
                                          n_samp=args.n_samp, L=ctx)
        ref = np.asarray(res[cks[0][0]]["skill"][:5])
        lines = [f"{split} ({D.n} episodes); skill per channel "
                 f"[{' '.join(CH)}], lower is better:"]
        for lab, _, _ in cks:
            sk = np.asarray(res[lab]["skill"][:5])
            cov = np.asarray(res[lab]["cov"][:5])
            lines.append(
                f"  {lab:8s} skill {np.round(sk, 3)} | cov90 "
                f"{np.round(cov, 2)} | MSE / {cks[0][0]} "
                f"{np.round(sk / ref, 3)} mean {np.mean(sk / ref):.3f}")
        log("\n".join(lines))
        del D
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
