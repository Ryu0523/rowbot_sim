#!/usr/bin/env python3
"""
Is the gz target outside the catalogue family, in its tail, or in-family
but not fitted by the network? (learn/meta/ROADMAP_2026-09-30.md section
11.8.) The in-family analogue of studies/onestep_gz.py --phase batch, on
the multi-mission held-out boats of cache meta8 (held8: 100 boats x 4
missions of 90 s, one error law per boat, drawn from the catalogue family
the pretrained net was trained on).

  stats  per mission, the rms of the normalised one-step error per channel
         (e0 / the training e_sd, valid steps) for every held8 mission and
         every recorded gz mission; gz's percentile among the boats
  fit    boats chosen by error size only (no outcome of any model):
         'gzlike' (sway + yaw rms nearest to gz's), 'typical' (around the
         median) and 'tail' (largest sway + yaw rms). Per
         boat a copy of the pretrained net is trained on its missions 0
         and 1 (the pretraining loss and token convention,
         model_preview.train_tokens_p), the check with the lowest
         flow-matching loss on mission 2 kept, and scored on mission 3
         against the frozen net with the same tokens and noise (paired);
         head only and full fine-tuning. The frozen net is also scored at
         several history lengths. On gz the same procedure with 2 missions
         gave MSE ratios 0.55 (head) / 0.35 (full) (onestep_gz batch.log).

With --groups all every boat of the split is one group (the in-family
corner boats of studies/corner_boats.py, --laws the boats it confirmed).

    python -m studies.tail_check --phase stats
    python -m studies.tail_check --phase fit --n-group 10
"""
import argparse
import copy
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))
import numpy as np   # noqa: E402
import torch   # noqa: E402

from learn.meta import model3 as M   # noqa: E402
from learn.meta import model_preview as MP   # noqa: E402
from studies import train_mem as TM   # noqa: E402
from studies.onestep_gz import windows   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CH = ("surge", "sway", "yaw", "heave", "pitch")
L_CTX, K0 = 256, 40


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def rms_rows(D, rows):
    """(len(rows), 5) rms of the normalised error over each row's valid
    steps."""
    out = []
    for r in rows:
        n = int(D.lens_np[r]) if hasattr(D, "lens_np") else int(D.len[r])
        ii = torch.tensor([r], device=D.dev)
        a = torch.zeros_like(ii)
        _, tgt, ok = M.window_tokens(D, ii, a, n)
        y = tgt[0][ok[0]][:, :5]
        y = y[torch.isfinite(y).all(1)]
        out.append(torch.sqrt((y ** 2).mean(0)).cpu().numpy())
    return np.array(out)


def gz_rms(stats, dev):
    from studies import onestep_gz as G
    keys = G.parse_missions("0-7")
    S = G.load_streams(keys, 120.0)
    out = []
    for k in keys:
        live = G.full_live(S[k], stats, dev)
        n = int(S[k]["n"])
        ii = torch.zeros(1, dtype=torch.long, device=dev)
        _, tgt, ok = M.window_tokens(live, ii, torch.zeros_like(ii), n)
        y = tgt[0][ok[0]][:, :5]
        y = y[torch.isfinite(y).all(1)]
        out.append(torch.sqrt((y ** 2).mean(0)).cpu().numpy())
    return np.array(out)


# ------------------------------------------------------------ fitting
def batch(D, rows, gen, B, L, variant):
    """B training windows from the rows (uniform), the pretraining
    convention."""
    r = torch.tensor(rows)[torch.randint(0, len(rows), (B,), generator=gen)]
    n = torch.as_tensor(D.lens_np[r.numpy()])
    a = (torch.rand(B, generator=gen) * (n - L + 1).clamp(min=1)).long()
    ii, a = r.to(D.dev), a.to(D.dev)
    tok, tgt, ok, _ = MP.train_tokens_p(D, ii, a, L, gen, variant)
    return tok, torch.nan_to_num(tgt), ok & torch.isfinite(tgt).all(-1)


def fit(ck, D, train, val, part, steps, dev, B=32, check=100, seed=0):
    net = MP.load_variant(ck, dev)
    for q in net.parameters():
        q.requires_grad_(part == "full")
    if part == "head":
        for q in net.head.parameters():
            q.requires_grad_(True)
    params = [q for q in net.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, 1e-4 if part == "full" else 3e-4,
                            weight_decay=1e-4)
    L = min(L_CTX, int(ck.get("ctx", L_CTX)))

    @torch.no_grad()
    def vloss():
        net.eval()
        v = []
        for i in range(4):
            g = torch.Generator().manual_seed(900 + i)
            tok, tgt, ok = batch(D, val, g, B, L, "p")
            v.append(float(M.fm_loss_h(net, net.encode(tok), tgt, ok, g)))
        net.train()
        return float(np.mean(v))

    g = torch.Generator().manual_seed(seed)
    best = dict(v=vloss(), it=0, state=copy.deepcopy(net.state_dict()))
    v0 = best["v"]
    net.train()
    for it in range(steps):
        tok, tgt, ok = batch(D, train, g, B, L, "p")
        loss = M.fm_loss_h(net, net.encode(tok), tgt, ok, g)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (it + 1) % check == 0:
            v = vloss()
            if v < best["v"]:
                best.update(v=v, it=it + 1,
                            state=copy.deepcopy(net.state_dict()))
    net.load_state_dict(best["state"])
    net.eval()
    return net, dict(v0=v0, v=best["v"], it=best["it"])


@torch.no_grad()
def score(net, D, row, seed, L=L_CTX, stride=16, n_samp=128):
    """Sums (squared error of the sample mean, squared truth) per channel
    on one mission, every step >= K0 (onestep_gz windows); tokens, flow
    noise fixed by seed (paired across nets)."""
    dev = D.dev
    n = int(D.lens_np[row])
    win = windows(n, L, K0, stride)
    se, e2 = np.zeros(5), np.zeros(5)
    for c0 in range(0, len(win), 128):
        wc = win[c0:c0 + 128]
        a = torch.tensor([w[0] for w in wc], device=dev)
        ii = torch.full_like(a, row)
        gt = torch.Generator().manual_seed(seed + c0)
        cfg = MP.fixed_cfg(D, a + L, 0.1, 24, 0.01, gt)
        tok, tgt, ok = MP.tokens_p(D, ii, a, L, cfg, gt, obs=False)
        h = net.encode(tok)
        pos = a[:, None] + torch.arange(L, device=dev)[None]
        lo = torch.tensor([w[1] for w in wc], device=dev)[:, None]
        hi = torch.tensor([w[2] for w in wc], device=dev)[:, None]
        sel = ok & (pos >= lo) & (pos < hi) & torch.isfinite(tgt).all(-1)
        hs, ys = h[sel], tgt[sel]
        gs = torch.Generator(device=dev).manual_seed(seed + 7 + c0)
        m = net.sample(hs, n_samp, gen=gs).mean(1)
        se += ((m - ys) ** 2).sum(0)[:5].cpu().numpy()
        e2 += (ys ** 2).sum(0)[:5].cpu().numpy()
    return se, e2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="stats", choices=("stats", "fit"))
    ap.add_argument("--cache", default=os.path.join(HERE, "_cache", "meta8"))
    ap.add_argument("--split", default="held8")
    ap.add_argument("--ckpt", default=os.path.join(HERE, "_cache", "meta7",
                                                   "model_p.pt"))
    ap.add_argument("--n-group", type=int, default=10)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--parts", default="head,full")
    ap.add_argument("--hist", default="16,64,128,256")
    ap.add_argument("--groups", default="size", choices=("size", "all"),
                    help="size: gzlike / typical / tail by error size; all: "
                         "every boat as one group (corner_boats.py)")
    ap.add_argument("--laws", default="",
                    help="json file {laws: [...]}: only these boats")
    args = ap.parse_args()
    dev = M.device()
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    D, boats = TM.load_split(args.cache, args.split, ck["stats"], dev,
                             ck.get("ctx", L_CTX))
    boats = [b for b in boats if len(b["rows"]) == 4]
    if args.laws:
        import json
        keep = set(json.load(open(args.laws))["laws"])
        boats = [b for b in boats if b["law"] in keep]
    R = rms_rows(D, [r for b in boats for r in b["rows"]]).reshape(
        len(boats), 4, 5)
    boat_rms = np.sqrt((R ** 2).mean(1))                 # (n_boats, 5)
    sy = np.sqrt((boat_rms[:, 1] ** 2 + boat_rms[:, 2] ** 2) / 2)
    if args.phase == "stats":
        G = gz_rms(ck["stats"], dev)
        gzb = np.sqrt((G ** 2).mean(0))
        gsy = np.sqrt((gzb[1] ** 2 + gzb[2] ** 2) / 2)
        log(f"{len(boats)} in-family boats ({args.split}), 16 gz missions; "
            "rms of the normalised one-step error [surge sway yaw heave "
            "pitch]")
        q = np.percentile(boat_rms, [5, 25, 50, 75, 95], 0)
        for p_, row in zip((5, 25, 50, 75, 95), q):
            log(f"  family {p_:2d} %   {np.round(row, 3)}")
        log(f"  gz (pooled)  {np.round(gzb, 3)} | per mission min "
            f"{np.round(G.min(0), 3)} max {np.round(G.max(0), 3)}")
        pct = (boat_rms < gzb[None]).mean(0) * 100
        log(f"  gz percentile among the boats per channel {np.round(pct, 1)}"
            f" | sway+yaw {100 * (sy < gsy).mean():.1f}")
        return
    order = np.argsort(sy)
    nb = len(boats)
    G = gz_rms(ck["stats"], dev)
    gzb = np.sqrt((G ** 2).mean(0))
    gsy = np.sqrt((gzb[1] ** 2 + gzb[2] ** 2) / 2)
    if args.groups == "all":
        groups = {args.split: order}
        log(f"gz sway+yaw rms {gsy:.3f}; one group: all {nb} boats of "
            f"{args.split} with 4 missions")
    else:
        groups = {"gzlike": np.argsort(np.abs(sy - gsy))[:args.n_group],
                  "typical": order[nb // 2 - args.n_group // 2:
                                   nb // 2 - args.n_group // 2
                                   + args.n_group],
                  "tail": order[-args.n_group:]}
        log(f"gz sway+yaw rms {gsy:.3f}; groups by the boats' own error "
            "size only: gzlike (nearest to gz), typical (median), tail "
            "(largest)")
    parts = [p for p in args.parts.split(",") if p]
    frozen = MP.load_variant(ck, dev)
    for gname, idx in groups.items():
        log(f"group {gname}: boats {[boats[i]['law'] for i in idx]}, "
            f"sway+yaw rms {np.round(sy[idx].min(), 3)}-"
            f"{np.round(sy[idx].max(), 3)} (family median "
            f"{np.round(np.median(sy), 3)})")
        tot = {k: [np.zeros(5), np.zeros(5)] for k in ["frozen"] + parts}
        per_boat = []
        for i in idx:
            bt = boats[i]
            r0, r1, r2, r3 = bt["rows"]
            seed = 31 * bt["law"] + 5
            res = {"frozen": score(frozen, D, r3, seed)}
            for p in parts:
                tb = time.time()
                net, info = fit(ck, D, [r0, r1], [r2], p, args.steps, dev)
                res[p] = score(net, D, r3, seed)
                log(f"  boat {bt['law']} {p}: validation {info['v0']:.3f} "
                    f"-> {info['v']:.3f} (step {info['it']}), "
                    f"{time.time() - tb:.0f} s")
                del net
                torch.cuda.empty_cache()
            for k, (se, e2) in res.items():
                tot[k][0] += se
                tot[k][1] += e2
            fz = res["frozen"][0]
            per_boat.append({p: res[p][0] / np.maximum(fz, 1e-12)
                             for p in parts})
            log(f"  boat {bt['law']}: frozen skill "
                f"{np.round(fz / np.maximum(res['frozen'][1], 1e-12), 3)}"
                + "".join(f" | {p} / frozen "
                          f"{np.round(per_boat[-1][p], 3)} mean "
                          f"{per_boat[-1][p].mean():.3f}" for p in parts))
        fs = tot["frozen"][0] / tot["frozen"][1]
        log(f"group {gname} pooled: frozen skill {np.round(fs, 3)}")
        for p in parts:
            sk = tot[p][0] / tot[p][1]
            ra = tot[p][0] / tot["frozen"][0]
            pb = np.array([b[p].mean() for b in per_boat])
            log(f"  {p}: skill {np.round(sk, 3)} | MSE / frozen "
                f"{np.round(ra, 3)} mean {ra.mean():.3f} | per boat mean "
                f"ratio median {np.median(pb):.3f} range {pb.min():.3f}-"
                f"{pb.max():.3f}")
        hs = {}
        for Lh in [int(x) for x in args.hist.split(",") if x]:
            se, e2 = np.zeros(5), np.zeros(5)
            for i in idx:
                bt = boats[i]
                s_, e_ = score(frozen, D, bt["rows"][3], 31 * bt["law"] + 5,
                               L=Lh, stride=1)
                se += s_
                e2 += e_
            hs[Lh] = se
        ref = hs[max(hs)]
        log(f"group {gname}: frozen net vs history (MSE / history "
            f"{max(hs) - 1}): " + ", ".join(
                f"{L_ - 1}: {(hs[L_] / ref).mean():.3f}" for L_ in sorted(hs)))


if __name__ == "__main__":
    main()
