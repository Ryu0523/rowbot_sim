#!/usr/bin/env python3
"""
Train and evaluate the long-term memory of learn/meta/memory_net.py
(ROADMAP_2026-09-30 section 11.3) on the multi-mission data of cache meta8
(studies/meta_step8.py: several missions of the same boat).

Training sample (one per boat in the batch): a target mission m of the
boat and a target window [a, a + L) in it (L = the base checkpoint's
context, 256; a as model3.fm_draw draws it). The memory is written, in time
order, from every segment (SEG steps) of the boat's missions before m and
from the segments of mission m that END at or before a (with 90-s missions
a <= 103 < SEG, so in practice the earlier missions only); the window then
reads it. The window's tokens see the same 256 steps as the base net, and
the memory holds only what happened before the window. A share --p-empty
of the samples is trained with the memory left empty, so that state stays
in distribution. Loss: the base's flow-matching loss on the window under
its training token convention (model_preview.train_tokens_p: rollout
segments included). Write segments use all-measured tokens (moment beyond
the segment, the model3 observation noise on the error history, the
forecast columns without perturbation; seg_tokens). The
writes of a batch run in lockstep (rows with fewer segments are inactive in
the later writes; a row's memory never depends on its batch, memory_net);
gradients flow through all writes (at most 9 with 4 missions of 90 s and
SEG 120).

The base starts from a pretrained checkpoint (default meta7 model_p.pt)
and is either kept frozen (--freeze-base: only the memory learns, the
control is the pretrained net itself) or fine-tuned at a lower rate than
the new memory parameters; with fine-tuning, kind
'none' is the control: the same fine-tuning on the same draws (the writes
have their own random stream), no memory.

  train   validation every --check steps on the boats kept out of
          training, later missions only: the window loss with the memory,
          with the memory left empty, and the pretrained net's (paired:
          the same draws, window tokens and noise); the best checkpoint
          (by the loss with memory) is kept
  eval    in-family held-out boats (held8): per boat with at least two
          missions, the LAST mission scored at every step >= K0 (windows as
          studies/onestep_gz.py), each memory written ONCE from the earlier
          missions with the deployment token convention (no noise on the
          error history); one-step skill per channel, the flow-matching
          loss, for every --ckpt (comma list) with its memory and with its
          memory left empty, and for the pretrained net; all paired

    python -m studies.train_mem --phase train --kind recurrent
    python -m studies.train_mem --phase train --kind none
    python -m studies.train_mem --phase eval \\
        --ckpt studies/_cache/meta8/mem_recurrent.pt,studies/_cache/meta8/mem_none.pt
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))
import numpy as np   # noqa: E402
import torch   # noqa: E402
from torch import nn   # noqa: E402

from learn.meta import memory_net as MN   # noqa: E402
from learn.meta import model3 as M   # noqa: E402
from learn.meta import model_preview as MP   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
L_WIN, SEG, K0 = 256, 120, 40


def log(msg, path=None):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


# ------------------------------------------------------------ data
def load_split(cache, split, stats, dev, ctx=L_WIN):
    D = MP.DataP(cache, split, dev, stats=stats)
    D.lens_np = D.len.cpu().numpy()
    D.L_win = min(L_WIN, int(ctx))
    bj = json.load(open(os.path.join(cache, f"{split}_boats.json")))
    boats = []
    for law, rows in sorted(bj["boats"].items(), key=lambda x: int(x[0])):
        mis = sorted(int(m) for m in rows)
        boats.append(dict(law=int(law), rows=[rows[str(m)] for m in mis],
                          mis=mis))
    return D, boats


def boat_plan(D, boat, m_idx, a, seg=SEG):
    """Write segments [(row, start, end)] for target mission index m_idx
    (position in boat['rows']) and window start a: every segment of the
    earlier missions, then the segments of the target mission ending at or
    before a."""
    out = []
    for j in range(m_idx):
        r = boat["rows"][j]
        out += [(r, s, e) for s, e in MN.segments(int(D.lens_np[r]), seg)]
    r = boat["rows"][m_idx]
    out += [(r, s, e) for s, e in MN.segments(int(D.lens_np[r]), seg)
            if e <= a]
    return out


def seg_tokens(D, rows, starts, S, gen, variant, obs=True):
    """All-measured tokens of segments [start, start + S) of rows, with the
    validity mask (inside the episode). The forecast columns without the
    forecast perturbation (lam 0, in the training distribution): the
    perturbation costs most of a training step at every past token and
    says nothing about the boat."""
    ii = torch.as_tensor(rows, device=D.dev).long()
    aa = torch.as_tensor(starts, device=D.dev).long()
    if variant == "a":
        tok, _, ok = M.window_tokens(D, ii, aa, S,
                                     M.obs_sd(len(ii), D.dev, gen)
                                     if obs else None, gen)
    else:
        cfg = MP.fixed_cfg(D, aa + S, 0.0, 24, 0.01, gen)
        tok, _, ok = MP.tokens_p(D, ii, aa, S, cfg, gen, obs=obs)
    return tok, ok


def write_plans(net, D, plans, gen, variant, seg=SEG, grad=True,
                obs=True):
    """Lockstep writes of per-row plans [(row, start, end)], each segment
    cut at its end; returns the memory pair."""
    B = len(plans)
    dev = D.dev
    mem = net.empty(B, dev)
    n = max((len(p) for p in plans), default=0)
    if net.kind == "none" or n == 0:
        return mem
    # the tokens of every segment of the batch in one call (one call per
    # write costs several times more)
    flat = [(j, b) for j in range(n) for b, p in enumerate(plans)
            if j < len(p)]
    rows = [plans[b][j][0] for j, b in flat]
    st = torch.as_tensor([plans[b][j][1] for j, b in flat], device=dev)
    en = torch.as_tensor([plans[b][j][2] for j, b in flat], device=dev)
    tok_all, ok_all = seg_tokens(D, rows, st.tolist(), seg, gen, variant,
                                 obs=obs)
    valid_all = ok_all & (st[:, None] + torch.arange(seg, device=dev)[None]
                          < en[:, None])
    jj = torch.as_tensor([j for j, _ in flat], device=dev)
    bb = torch.as_tensor([b for _, b in flat], device=dev)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        return net.write_seq(tok_all, valid_all, jj, bb, mem)


def draw_batch(D, boats, pool, B, gen, p_late=0.8, p_empty=0.0):
    """B (boat, mission index, row, window start, empty) draws: the target
    mission is a later one (index >= 1) with probability p_late when the
    boat has one (1: later missions only, when there are any); the window
    start as model3.fm_draw; empty (probability p_empty): the sample is
    trained with the memory left empty."""
    L = D.L_win
    out = []
    for _ in range(B):
        bi = pool[int(torch.randint(0, len(pool), (1,), generator=gen))]
        bt = boats[bi]
        nm = len(bt["rows"])
        if nm > 1 and float(torch.rand(1, generator=gen)) < p_late:
            mi = 1 + int(torch.randint(0, nm - 1, (1,), generator=gen))
        else:
            mi = int(torch.randint(0, nm, (1,), generator=gen))
        r = bt["rows"][mi]
        span = max(0, int(D.lens_np[r]) - 1 - L)
        a = int(float(torch.rand(1, generator=gen)) * (span + 1))
        if float(torch.rand(1, generator=gen)) < 0.25:
            a = 0
        empty = float(torch.rand(1, generator=gen)) < p_empty
        out.append((bi, mi, r, a, empty))
    return out


def batch_loss(net, D, boats, draws, gen, variant, use_mem=True,
               grad=True, gen_w=None):
    """The window loss of the draws; gen_w: the writes' generator (default
    gen), so that rows with and without memory share the window's tokens
    and noise."""
    plans = [boat_plan(D, boats[bi], mi, a) if use_mem and not em else []
             for bi, mi, r, a, em in draws]
    mem = write_plans(net, D, plans, gen if gen_w is None else gen_w,
                      variant, grad=grad)
    ii = torch.as_tensor([d[2] for d in draws], device=D.dev).long()
    aa = torch.as_tensor([d[3] for d in draws], device=D.dev).long()
    tok, tgt, ok, _ = MP.train_tokens_p(D, ii, aa, D.L_win, gen, variant)
    h = net.encode(tok, mem)
    return M.fm_loss_h(net.base, h, tgt, ok, gen)


@torch.no_grad()
def validate(net, base, D, boats, pool, variant, n_batches=8, B=24):
    """The window loss with the memory, with the memory left empty, and the
    pretrained net's, on the same draws (later missions only), window
    tokens and noise."""
    out = {"mem": [], "empty": [], "base": []}
    for i in range(n_batches):
        g = torch.Generator().manual_seed(9000 + i)
        draws = draw_batch(D, boats, pool, B, g, p_late=1.0)
        for key, use in (("mem", True), ("empty", False)):
            gk = torch.Generator().manual_seed(7000 + i)
            gw = torch.Generator().manual_seed(8000 + i)
            out[key].append(float(batch_loss(net, D, boats, draws, gk,
                                             variant, use, grad=False,
                                             gen_w=gw)))
        gk = torch.Generator().manual_seed(7000 + i)
        ii = torch.as_tensor([d[2] for d in draws], device=D.dev).long()
        aa = torch.as_tensor([d[3] for d in draws], device=D.dev).long()
        tok, tgt, ok, _ = MP.train_tokens_p(D, ii, aa, D.L_win, gk, variant)
        out["base"].append(float(M.fm_loss_h(base, base.encode(tok), tgt,
                                             ok, gk)))
    return {k: float(np.mean(v)) for k, v in out.items()}


# ------------------------------------------------------------ train
def phase_train(args, dev):
    ck = torch.load(args.base, map_location=dev, weights_only=False)
    variant = ck.get("variant", "p")
    if variant != "p":
        raise SystemExit("train_mem: variant p only (the token conventions "
                         "of seg_tokens and phase_eval)")
    D, boats = load_split(args.cache, args.split, ck["stats"], dev,
                          ck.get("ctx", L_WIN))
    nb = len(boats)
    gv = torch.Generator().manual_seed(5)
    perm = torch.randperm(nb, generator=gv).tolist()
    n_val = max(4, nb // 20)
    val_pool, tr_pool = perm[:n_val], perm[n_val:]
    base_frozen = MP.load_variant(ck, dev)
    base = MP.load_variant(ck, dev)
    base.train()
    net = MN.MemNet(base, kind=args.kind, n_mem=args.n_mem,
                    n_sum=args.n_sum, max_sum=args.max_sum,
                    n_fast=args.n_fast).to(dev)
    if args.freeze_base:
        # the pretrained net unchanged (fine-tuning it on meta8's ~1000 laws
        # made it worse on new laws, 2026-10-03); only the memory learns, so
        # 'none' would be the pretrained net itself
        if args.kind == "none":
            raise SystemExit("--freeze-base with kind none trains nothing: "
                             "the control is the pretrained net")
        for p in net.base.parameters():
            p.requires_grad_(False)
        opt = torch.optim.AdamW([p for p in net.parameters()
                                 if p.requires_grad], lr=args.lr_new,
                                weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW(MN.mem_lr_groups(net, args.lr_base,
                                                 args.lr_new),
                                weight_decay=1e-4)
    warm = max(1, args.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / warm) * (
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, i /
                                                          args.steps)))))
    out = args.out or os.path.join(args.cache, f"mem_{args.kind}.pt")
    lp = out.replace(".pt", ".log")
    log(f"train_mem {args.kind}: {nb} boats ({len(tr_pool)} train, "
        f"{n_val} validation), {D.n} missions, base {args.base}, new "
        f"parameters {MN.n_params(net) - MN.n_params(base)}, steps "
        f"{args.steps}, batch {args.batch}, SEG {SEG}, window {D.L_win}, "
        f"empty-memory share {args.p_empty}, base "
        f"{'frozen' if args.freeze_base else f'lr {args.lr_base}'}", lp)
    g = torch.Generator().manual_seed(args.seed)
    # the writes draw from their own generator, so every kind (the 'none'
    # control included) trains on the same boats, windows and window noise
    g_w = torch.Generator().manual_seed(args.seed + 1)
    best = dict(v=float("inf"), it=-1)
    t0 = time.time()
    p_empty = args.p_empty if args.kind != "none" else 0.0
    for it in range(args.steps):
        draws = draw_batch(D, boats, tr_pool, args.batch, g,
                           p_empty=p_empty)
        loss = batch_loss(net, D, boats, draws, g, variant, gen_w=g_w)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % args.check == 0 or it == args.steps - 1:
            net.eval()
            v = validate(net, base_frozen, D, boats, val_pool, variant)
            net.train()
            log(f"  {args.kind} {it:6d}: train {loss.item():.4f} | "
                f"validation with memory {v['mem']:.4f}, empty memory "
                f"{v['empty']:.4f}, pretrained net {v['base']:.4f} "
                f"({time.time() - t0:.0f} s)", lp)
            if v["mem"] < best["v"]:
                best.update(v=v["mem"], it=it)
                torch.save(dict(net=net.state_dict(), cfg=net.config(),
                                base_ckpt=args.base, variant=variant,
                                stats=ck["stats"], arch=ck.get("arch"),
                                ctx=ck.get("ctx", L_WIN), it=it, val=v,
                                seg=SEG, frozen=bool(args.freeze_base)),
                           out + ".tmp")
                os.replace(out + ".tmp", out)
    log(f"  best step {best['it']} ({best['v']:.4f}); saved {out}", lp)


# ------------------------------------------------------------ eval
def load_memnet(path, dev):
    """(MemNet with its fine-tuned base, the saved dict, the base
    checkpoint with the stats the MemNet was trained with)."""
    st = torch.load(path, map_location=dev, weights_only=False)
    ck = torch.load(st["base_ckpt"], map_location=dev, weights_only=False)
    ck = dict(ck, stats=st["stats"], ctx=st.get("ctx", ck.get("ctx")))
    base = MP.load_variant(ck, dev)
    net = MN.MemNet(base, **st["cfg"]).to(dev)
    net.load_state_dict(st["net"])
    net.eval()
    return net, st, ck


@torch.no_grad()
def phase_eval(args, dev):
    from studies.onestep_gz import windows
    paths = [p for p in args.ckpt.split(",") if p]
    nets = [load_memnet(p, dev) for p in paths]
    st0, ck0 = nets[0][1], nets[0][2]
    variant = st0["variant"]
    for p, (_, st, _) in zip(paths, nets):
        if st.get("seg", SEG) != SEG or st["variant"] != "p":
            raise SystemExit(f"{p}: trained with SEG {st.get('seg')} / "
                             f"variant {st['variant']} (this code: SEG "
                             f"{SEG}, variant p)")
    base = MP.load_variant(dict(torch.load(st0["base_ckpt"],
                                           map_location=dev,
                                           weights_only=False),
                                stats=st0["stats"]), dev)
    D, boats = load_split(args.cache, args.eval_split, st0["stats"], dev,
                          st0.get("ctx", L_WIN))
    # --eval-window: a shorter window than trained (a diagnostic: can the
    # memory stand in for the missing in-window history?)
    L = min(D.L_win, args.eval_window) if args.eval_window else D.L_win
    if L <= K0:
        raise SystemExit(f"--eval-window {L}: must exceed K0 = {K0}")
    rows = [("pretrained", None, False)]
    for p, (net, st, _) in zip(paths, nets):
        lab = os.path.splitext(os.path.basename(p))[0]
        rows.append((lab, net, True))
        if net.kind != "none":                # 'none': the same thing
            rows.append((lab + " empty", net, False))
    tal = {r[0]: M.Tally((M.C7,)) for r in rows}
    fm = {r[0]: [] for r in rows}
    n_b = 0
    for bt in boats:
        if len(bt["rows"]) < 2:
            continue
        if args.max_boats and n_b >= args.max_boats:
            break
        mi = len(bt["rows"]) - 1
        r = bt["rows"][mi]
        n = int(D.lens_np[r])
        if n <= K0 + 1:                 # no scored step in this mission
            continue
        n_b += 1
        win = windows(n, L, K0)
        plan = boat_plan(D, bt, mi, 0)            # earlier missions only
        a = torch.tensor([w[0] for w in win], device=dev)
        ii = torch.full_like(a, r)
        seed = 31 * bt["law"] + 7
        for lab, net, use in rows:
            if net is None:
                mem = None
            elif use:
                g = torch.Generator().manual_seed(seed)
                mem = MN.expand(write_plans(net, D, [plan], g, variant,
                                            grad=False, obs=False),
                                len(win))
            else:
                mem = net.empty(len(win), dev)
            gt = torch.Generator().manual_seed(seed + 1)
            cfg = MP.fixed_cfg(D, a + L, 0.1, 24, 0.01, gt)
            tok, tgt, ok = MP.tokens_p(D, ii, a, L, cfg, gt, obs=False)
            h = base.encode(tok) if net is None else net.encode(tok, mem)
            pos = a[:, None] + torch.arange(L, device=dev)[None]
            lo = torch.tensor([w[1] for w in win], device=dev)[:, None]
            hi = torch.tensor([w[2] for w in win], device=dev)[:, None]
            sel = ok & (pos >= lo) & (pos < hi)
            hs, ys = h[sel], tgt[sel]
            mdl = base if net is None else net.base
            gs = torch.Generator(device=dev).manual_seed(seed + 2)
            torch.manual_seed(seed + 4)      # Tally's PIT jitter, paired
            tal[lab].add(mdl.sample(hs, 64, gen=gs), ys)
            gf = torch.Generator().manual_seed(seed + 3)
            fm[lab].append(float(M.fm_loss_h(
                mdl, hs[None], ys[None],
                torch.ones(1, len(ys), dtype=torch.bool, device=dev), gf)))
    res = {k: tal[k].summary() for k in tal}
    log(f"eval on {args.eval_split}: {n_b} boats, last mission scored, "
        f"memory written once from the earlier missions, window {L}")
    ref = res["pretrained"]["skill"][:5]
    for lab, _, _ in rows:
        sk = res[lab]["skill"][:5]
        log(f"  {lab:22s} skill {np.round(sk, 3)} | cov90 "
            f"{np.round(res[lab]['cov'][:5], 3)} | flow-matching loss "
            f"{np.mean(fm[lab]):.4f} | MSE / pretrained "
            f"{np.round(sk / ref, 3)} mean {np.mean(sk / ref):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="train", choices=("train", "eval"))
    ap.add_argument("--cache", default=os.path.join(HERE, "_cache", "meta8"))
    ap.add_argument("--split", default="train8")
    ap.add_argument("--eval-split", default="held8")
    ap.add_argument("--base", default=os.path.join(HERE, "_cache", "meta7",
                                                   "model_p.pt"))
    ap.add_argument("--kind", default="recurrent", choices=MN.KINDS)
    ap.add_argument("--n-mem", type=int, default=16)
    ap.add_argument("--n-sum", type=int, default=4)
    ap.add_argument("--max-sum", type=int, default=64)
    ap.add_argument("--n-fast", type=int, default=64)
    ap.add_argument("--p-empty", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--check", type=int, default=500)
    ap.add_argument("--lr-base", type=float, default=1e-4)
    ap.add_argument("--freeze-base", action="store_true",
                    help="keep the pretrained net unchanged, train only the "
                         "memory (not with kind none)")
    ap.add_argument("--lr-new", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--eval-window", type=int, default=0,
                    help="eval: window length (default: the trained one); "
                         "shorter is a diagnostic of what the memory holds")
    ap.add_argument("--max-boats", type=int, default=0,
                    help="eval: score at most this many boats (0: all)")
    args = ap.parse_args()
    dev = M.device()
    if args.phase == "train":
        phase_train(args, dev)
    else:
        phase_eval(args, dev)


if __name__ == "__main__":
    main()
