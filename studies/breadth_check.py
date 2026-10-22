#!/usr/bin/env python3
"""
Why does a broader prior family make the model worse? (learn/meta/
ROADMAP_2026-09-30.md section 11.10.) For each family's preview net
(meta5 projection + preview, meta6 general D9, meta7 catalogue D10):

  infamily  on the family's own held-out split A (single 90 s episodes,
            laws never trained on): the frozen net vs a copy trained on the
            episode's own first K steps (test-time training, windows of 128
            tokens inside [0, K), the pretraining loss and convention, a
            fixed number of full-network steps), both scored on steps >= K
            of the same episode with the same tokens and noise. A net that
            already extracts what its window holds (an ideal predictor for
            its family) gains nothing; the gain measures how far the frozen
            net is from what is learnable inside the episode.
  gz        on the 16 recorded high-fidelity missions (studies/onestep_gz):
            the frozen net's skill on the held-out missions; test-time
            training on each held-out mission's first 240 steps (scored on
            the rest); a copy trained on 2 train missions (2 validation
            missions choose the step), scored on all held-out steps.

Skills are scale free per channel (MSE of the sample mean / MSE of 0), so
families with different error scales compare.

    python -m studies.breadth_check --part infamily,gz
"""
import argparse
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))
import numpy as np   # noqa: E402
import torch   # noqa: E402

from learn.meta import model3 as M   # noqa: E402
from learn.meta import model_preview as MP   # noqa: E402
from studies import onestep_gz as G   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
L_CTX = 256


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def ttt_rows(ck, D, row, K, steps, dev, L=128, B=32, seed=0):
    net = MP.load_variant(ck, dev)
    for q in net.parameters():
        q.requires_grad_(True)
    opt = torch.optim.AdamW(net.parameters(), 1e-4, weight_decay=1e-4)
    L = min(L, K)
    g = torch.Generator().manual_seed(seed)
    net.train()
    for _ in range(steps):
        a = torch.randint(0, K - L + 1, (B,), generator=g).to(dev)
        ii = torch.full_like(a, row)
        tok, tgt, ok, _ = MP.train_tokens_p(D, ii, a, L, g, "p")
        ok = ok & torch.isfinite(tgt).all(-1)
        loss = M.fm_loss_h(net, net.encode(tok), torch.nan_to_num(tgt), ok, g)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    net.eval()
    return net


@torch.no_grad()
def score_rows(net, D, row, k0, seed, L=L_CTX, n_samp=128):
    """(se, e2) per channel over steps >= k0 of one episode (onestep_gz
    windows, stride 16), tokens and flow noise fixed by seed."""
    dev = D.dev
    n = int(D.len[row])
    win = G.windows(n, L, k0, 16)
    a = torch.tensor([w[0] for w in win], device=dev)
    ii = torch.full_like(a, row)
    gt = torch.Generator().manual_seed(seed)
    cfg = MP.fixed_cfg(D, a + L, 0.1, 24, 0.01, gt)
    tok, tgt, ok = MP.tokens_p(D, ii, a, L, cfg, gt, obs=False)
    h = net.encode(tok)
    pos = a[:, None] + torch.arange(L, device=dev)[None]
    lo = torch.tensor([w[1] for w in win], device=dev)[:, None]
    hi = torch.tensor([w[2] for w in win], device=dev)[:, None]
    sel = ok & (pos >= lo) & (pos < hi) & torch.isfinite(tgt).all(-1)
    hs, ys = h[sel], tgt[sel]
    gs = torch.Generator(device=dev).manual_seed(seed + 7)
    m = net.sample(hs, n_samp, gen=gs).mean(1)
    return (((m - ys) ** 2).sum(0)[:5].cpu().numpy(),
            (ys ** 2).sum(0)[:5].cpu().numpy())


def part_infamily(fam, ck, dev, args):
    cache = os.path.join(HERE, "_cache", fam)
    D = MP.DataP(cache, "A", dev, stats=ck["stats"])
    lens = D.len.cpu().numpy()
    cand = np.where(lens >= args.k_in + 100)[0]
    rng = np.random.default_rng(0)
    rows = rng.choice(cand, size=min(args.n_eps, len(cand)), replace=False)
    frozen = MP.load_variant(ck, dev)
    steps_l = [int(x) for x in args.ttt_steps.split(",")]
    tot = {k: [np.zeros(5), np.zeros(5)] for k in
           ["frozen"] + [f"ttt {s}" for s in steps_l]}
    per = {f"ttt {s}": [] for s in steps_l}
    for r in rows:
        seed = 1000 + int(r)
        fz = score_rows(frozen, D, int(r), args.k_in, seed)
        tot["frozen"][0] += fz[0]
        tot["frozen"][1] += fz[1]
        for s in steps_l:
            net = ttt_rows(ck, D, int(r), args.k_in, s, dev)
            sc = score_rows(net, D, int(r), args.k_in, seed)
            tot[f"ttt {s}"][0] += sc[0]
            tot[f"ttt {s}"][1] += sc[1]
            per[f"ttt {s}"].append((sc[0] / np.maximum(fz[0], 1e-12)).mean())
            del net
        torch.cuda.empty_cache()
    fs = tot["frozen"][0] / tot["frozen"][1]
    log(f"{fam} in-family (split A, {len(rows)} episodes, first {args.k_in} "
        f"steps learned, rest scored): frozen skill {np.round(fs, 3)}")
    for s in steps_l:
        k = f"ttt {s}"
        ra = tot[k][0] / tot["frozen"][0]
        pe = np.array(per[k])
        log(f"  {k} full steps: skill {np.round(tot[k][0] / tot[k][1], 3)} | "
            f"MSE / frozen {np.round(ra, 3)} mean {ra.mean():.3f} | episodes "
            f"better {(pe < 1).mean() * 100:.0f} %, per-episode mean ratio "
            f"median {np.median(pe):.3f}")


def part_gz(fam, ck, dev, args):
    spec, _, _ = G.load_row("cat_p_prior", dev)
    spec = dict(spec, name=fam, ckpt=os.path.join(HERE, "_cache", fam,
                                                  "model_p.pt"),
                head=None, mode="prior")
    train = G.parse_missions("0-3")
    held = G.parse_missions("4-7")
    S = G.load_streams(train + held, 120.0)
    frozen = MP.load_variant(ck, dev)
    val = G.parse_missions("3")
    pool = [k for k in train if k not in val]
    cross = G.batch_fit(ck, dev, S, pool[:2], val, "full", 2000)[0]
    rows_all = {"frozen": {}, "cross-mission full 2": {}}
    rows_k = {"frozen": {}, "ttt 30": {}}
    for key in held:
        st = S[key]
        live = G.full_live(st, ck["stats"], dev)
        seed = G.ML.row_seed(key[0], key[1], 43)
        for lab, net in (("frozen", frozen), ("cross-mission full 2", cross)):
            v = SimpleNamespace(net=net, head=None, learner=None, ctx=L_CTX)
            rows_all[lab][key] = G.score_mission(v, live, st, "p", seed)
        t = G.ttt_fit(ck, dev, live, 240, "full", 30)
        for lab, net in (("frozen", frozen), ("ttt 30", t)):
            v = SimpleNamespace(net=net, head=None, learner=None, ctx=L_CTX,
                                k0=240)
            rows_k[lab][key] = G.score_mission(v, live, st, "p", seed)
        del live, t
    log(f"{fam} on gz (8 held-out missions):")
    G.show(f"{fam} frozen", G.summarise(rows_all["frozen"]))
    G.show(f"{fam} cross-mission full 2", G.summarise(
        rows_all["cross-mission full 2"], rows_all["frozen"]))
    G.show(f"{fam} frozen (steps >= 240)", G.summarise(rows_k["frozen"]))
    G.show(f"{fam} ttt 30 on the first 240 steps",
           G.summarise(rows_k["ttt 30"], rows_k["frozen"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="meta5,meta6,meta7")
    ap.add_argument("--part", default="infamily,gz")
    ap.add_argument("--n-eps", type=int, default=48)
    ap.add_argument("--k-in", type=int, default=180)
    ap.add_argument("--ttt-steps", default="30,100")
    args = ap.parse_args()
    dev = M.device()
    G.FE.out_dir = lambda: G.CACHE
    G.FE.log = log
    parts = args.part.split(",")
    for fam in args.families.split(","):
        ck = torch.load(os.path.join(HERE, "_cache", fam, "model_p.pt"),
                        map_location=dev, weights_only=False)
        if "infamily" in parts:
            part_infamily(fam, ck, dev, args)
        if "gz" in parts:
            part_gz(fam, ck, dev, args)


if __name__ == "__main__":
    main()
