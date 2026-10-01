#!/usr/bin/env python3
"""
Safety head (ROADMAP_2026-09-30 sections 9-10, part 2): a small network on
the FROZEN trunk of a trained model_preview net (model3.Net or NetP) that
predicts the distribution of the step's safety quantities

    APK_t   peak vertical CG acceleration over step t (m/s^2, upward +)
    HMIN_t  minimum bow height over step t (m)

exactly as episode6 / episode7 record them in the training world (D9.6,
D9.8 item 10), from the trunk output h_t at token t and the step's error
e_t (normalised, as the flow head samples it):

    p(e_t, APK_t, HMIN_t | history) = p(e_t | h_t) p(APK_t, HMIN_t | h_t, e_t)

so inside an MPPI rollout every sampled future gets peaks and bow heights
consistent with its own sampled error (the chain rule; no hand features).

Output: quantiles QS of both targets (monotone by construction: the median
plus cumulative softplus steps), in raw units. Training: pinball loss on
the net's own training split (meta6 / meta7 packs carry APK / HMIN), the
trunk frozen, tokens drawn with the trunk's own training convention
(model3 observation noise; for w / p model_preview.train_tokens_p, so the
head also sees the rollout-segment tokens the MPC feeds it). The tails are
trained directly (0.01 / 0.99 levels) and reported on held-out episodes.

Sampling: inv_cdf interpolates between the quantile levels and extends
both ends with an exponential tail (log-linear in the tail probability,
scale from the two outermost levels; exponentially distributed peaks are
the standard assumption, NSWCCD-80-TR-2016/033 eq. 1).

meta5 has no APK / HMIN arrays: no head there (the MPC then uses the
predicted step-mean heave acceleration and the bow height from the
predicted states, learn/meta/mpc_constrained.py).

    python -m learn.meta.safety_head --cache meta6 --variant p
"""
import argparse
import math
import os
import pickle
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from learn.meta import model3 as M

QS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
TARGETS = ("APK", "HMIN")
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_ROOT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "studies",
                          "_cache")


# ------------------------------------------------------------------ head
class SafetyHead(nn.Module):
    """[h (d), asinh(e / KAPPA) (10)] -> quantiles (..., 2, len(QS)) of
    (APK, HMIN), raw units (y_mu / y_sd: the training targets' scale)."""

    def __init__(self, d=192, hidden=(256, 256), qs=QS):
        super().__init__()
        self.qs = tuple(qs)
        self.nq, self.n_out = len(self.qs), len(TARGETS)
        self.mid = self.qs.index(0.5)
        self.net = M.mlp(d + M.C7, hidden, self.n_out * self.nq)
        self.register_buffer("y_mu", torch.zeros(self.n_out))
        self.register_buffer("y_sd", torch.ones(self.n_out))

    def forward(self, h, e, raw=True):
        z = self.net(torch.cat([h, torch.asinh(e / M.KAPPA)], -1))
        z = z.view(*z.shape[:-1], self.n_out, self.nq)
        m = self.mid
        q_mid = z[..., m:m + 1]
        up = torch.cumsum(F.softplus(z[..., m + 1:]), -1)
        dn = torch.cumsum(F.softplus(z[..., :m].flip(-1)), -1).flip(-1)
        q = torch.cat([q_mid - dn, q_mid, q_mid + up], -1)
        if raw:
            q = q * self.y_sd[:, None] + self.y_mu[:, None]
        return q


def pinball(qn, yn, qs):
    """Mean pinball loss of normalised quantiles qn (..., 2, nq) against
    normalised targets yn (..., 2): returns (..., 2) per target."""
    lv = torch.as_tensor(qs, dtype=qn.dtype, device=qn.device)
    d = yn[..., None] - qn
    return torch.maximum(lv * d, (lv - 1.0) * d).mean(-1)


def inflate(q, s, mid=QS.index(0.5)):
    """Quantiles widened about the median by s (scalar or broadcastable to
    q[..., 0])."""
    if torch.is_tensor(s):
        s = s.to(q.dtype)[..., None] if s.dim() else s.to(q.dtype)
    elif s == 1.0:
        return q
    med = q[..., mid:mid + 1]
    return med + s * (q - med)


def inv_cdf(q, u, qs=QS):
    """Value at probability u (...) of the distribution with quantiles q
    (..., nq) at levels qs: linear between levels, exponential tails beyond
    the outermost levels (scale from the two outermost levels at each
    end)."""
    lv = torch.as_tensor(qs, dtype=q.dtype, device=q.device)
    u = u.to(q.dtype).clamp(1e-6, 1.0 - 1e-6)
    idx = torch.searchsorted(lv, u.contiguous()).clamp(1, len(qs) - 1)
    lo, hi = lv[idx - 1], lv[idx]
    qlo = torch.gather(q, -1, (idx - 1)[..., None])[..., 0]
    qhi = torch.gather(q, -1, idx[..., None])[..., 0]
    mid = qlo + (qhi - qlo) * (u - lo) / (hi - lo)
    b_lo = (q[..., 1] - q[..., 0]) / math.log(qs[1] / qs[0])
    b_hi = (q[..., -1] - q[..., -2]) / math.log((1 - qs[-2]) / (1 - qs[-1]))
    low = q[..., 0] - b_lo * torch.log(qs[0] / u)
    high = q[..., -1] + b_hi * torch.log((1 - qs[-1]) / (1 - u))
    return torch.where(u < qs[0], low, torch.where(u > qs[-1], high, mid))


def head_path(cache, variant):
    return os.path.join(cache, f"safety_{variant}.pt")


def save_head(head, info, path):
    tmp = path + ".tmp"
    torch.save(dict(state=head.state_dict(), qs=head.qs, info=info), tmp)
    os.replace(tmp, path)


def load_head(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    head = SafetyHead(qs=ck["qs"]).to(dev)
    head.load_state_dict(ck["state"])
    head.eval()
    return head, ck.get("info", {})


# ------------------------------------------------------------------ data
def load_split(cache, split, dev, stats, n_max=None, seed=0):
    """model3.Data3 + model_preview.DataP for a SUBSET of a pack's episodes
    (at most n_max, drawn with `seed`), plus the safety arrays APK / HMIN
    (n, T) raw: the same normalisations and conventions (valid = t <
    len - 1 for packed splits, the wave columns from the library's mu / sd
    when the stats have none), the time axis padded to W_CTX. Loads one
    array at a time (16 GB laptop). Refuses a pack without APK / HMIN."""
    from learn.meta import model_preview as MP
    from learn.meta import relabel as R
    from learn.meta.data5 import mid_pose
    env = R._env()
    d = np.load(os.path.join(cache, f"{split}.npz"))
    if not all(k in d.files for k in TARGETS):
        raise RuntimeError(f"{cache}/{split}.npz has no {TARGETS} arrays "
                           "(meta5 packs: no safety head)")
    L_all = d["len"]
    n_all = len(L_all)
    idx = np.arange(n_all)
    if n_max is not None and n_all > n_max:
        idx = np.sort(np.random.default_rng(seed).choice(n_all, n_max,
                                                         replace=False))
    meta = pickle.load(open(os.path.join(cache, f"{split}_meta.pkl"), "rb"))
    meta = [meta[i] for i in idx]
    lib = np.load(os.path.join(cache, "lib.npz"))
    st = dict(stats)
    if "w_mu" not in st:
        st["w_mu"] = lib["mu"][11:26].astype(np.float32)
        st["w_sd"] = lib["sd"][11:26].astype(np.float32)
    L = L_all[idx]
    XS = d["XS"][idx]
    T = d["U"].shape[1]
    n_e0 = L - 1 if XS.shape[1] == T else L
    valid = np.arange(T)[None] < n_e0[:, None]
    live = (np.arange(T)[None] < L[:, None])[..., None]
    f32 = dict(dtype=torch.float32, device=dev)
    D = SimpleNamespace(dev=dev, n=len(idx), T=T, stats=st, meta=meta,
                        idx=idx)
    S = d["S"][idx]
    D.X = torch.tensor((S[..., :M.D_X] - st["x_mu"]) / st["x_sd"] * live,
                       **f32)
    D.Wm = torch.tensor((S[..., 11:26] - st["w_mu"]) / st["w_sd"] * live,
                        **f32)
    del S
    D.U = torch.tensor((d["U"][idx] - st["u_mu"]) / st["u_sd"], **f32)
    e0 = np.load(os.path.join(cache, f"{split}_e0.npz"))["E0"][idx]
    D.E = torch.tensor(e0 / st["e_sd"] * valid[..., None], **f32)
    del e0
    D.Wmid = torch.tensor((d["W_MID"][idx] - st["w_mu"]) / st["w_sd"] * live,
                          **f32)
    D.valid = torch.tensor(valid, device=dev)
    D.len = torch.tensor(L, device=dev)
    D.XS, D.Uraw = XS, d["U"][idx]
    D.dtc = float(env["dt"] * env["sub"])
    D.h = 0.5 * D.dtc
    xm, ym, pm = mid_pose(XS[:, :T].astype(float), "plant14", D.h)
    D.PM = torch.tensor(np.stack([xm, ym, pm], -1), **f32)
    D.SEA = torch.tensor(MP.sea_arrays(meta), **f32)
    D.x_st = torch.tensor(np.asarray(env["x_st"]), **f32)
    D.y_off = torch.tensor(np.asarray(env["y_off"]), **f32)
    D.w_mu = torch.tensor(st["w_mu"], **f32)
    D.w_sd = torch.tensor(st["w_sd"], **f32)
    D.Y = torch.tensor(np.stack([d[k][idx] for k in TARGETS], -1), **f32)
    MP.pad_time(D, ("X", "U", "E", "valid", "Wm", "Wmid", "PM", "Y"))
    return D


def window_batch(net, D, ii, a, L, gen, variant):
    """(h (B, L, d) no grad, e targets (B, L, 10), safety targets (B, L, 2),
    loss mask (B, L)) of recorded windows under the trunk's own training
    convention."""
    from learn.meta import model_preview as MP
    if variant == "a":
        tok, tgt, ok = M.window_tokens(D, ii, a, L,
                                       M.obs_sd(len(ii), D.dev, gen), gen)
    else:
        tok, tgt, ok, _ = MP.train_tokens_p(D, ii, a, L, gen, variant)
    with torch.no_grad():
        h = net.encode(tok)
    t = (a[:, None] + torch.arange(L, device=D.dev)[None]).clamp(max=D.T - 1)
    y = D.Y[ii[:, None], t]
    return h, tgt, y, ok


@torch.no_grad()
def evaluate(head, net, D, eps, L, variant, n_batches=16, batch=24, seed=4321):
    """Held-out pinball (normalised) and calibration: the share of targets
    at or below each quantile level, per target; the 90% band coverage;
    the APK share above q99 and the HMIN share below q01."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    Lmax = D.len.cpu()
    below = torch.zeros(2, head.nq, dtype=torch.float64)
    n, pb = 0, torch.zeros(2, dtype=torch.float64)
    for _ in range(n_batches):
        ii, a = M.fm_draw(eps, batch, gen, Lmax, L, D.dev)
        h, tgt, y, ok = window_batch(net, D, ii, a, L, gen, variant)
        q = head(h, tgt)
        hs, ys = q[ok], y[ok]
        below += (ys[..., None] <= hs).double().sum(0).cpu()
        yn = (ys - head.y_mu) / head.y_sd
        qn = (hs - head.y_mu[:, None]) / head.y_sd[:, None]
        pb += pinball(qn, yn, head.qs).double().sum(0).cpu()
        n += len(ys)
    cal = (below / max(n, 1)).numpy()
    i05, i95 = head.qs.index(0.05), head.qs.index(0.95)
    return dict(n=n, pinball=(pb / max(n, 1)).numpy().tolist(),
                cal=cal.tolist(), qs=list(head.qs),
                cov90=(cal[:, i95] - cal[:, i05]).tolist(),
                apk_above_q99=float(1.0 - cal[0, -1]),
                hmin_below_q01=float(cal[1, 0]))


def train_head(net, D, variant, L, steps=4000, batch=32, lr=1e-3, check=500,
               patience=4, seed=0, log=print):
    """The safety head on the frozen trunk `net` over the split D (load_split):
    AdamW, the pinball loss on normalised targets, validation on model3's
    held-out episodes every `check` steps, early stopping, best kept.
    Returns (head, info)."""
    dev = D.dev
    for p in net.parameters():
        p.requires_grad_(False)
    net.eval()
    head = SafetyHead(d=net.d).to(dev)
    ok_all = D.valid
    ys = D.Y[ok_all]
    head.y_mu.copy_(ys.mean(0))
    head.y_sd.copy_(ys.std(0) + 1e-6)
    tr, va = M.split_pools(D.n, seed)
    Lmax = D.len.cpu()
    g = torch.Generator(device="cpu").manual_seed(seed)
    opt = torch.optim.AdamW(head.parameters(), lr, weight_decay=1e-4)
    best = dict(v=float("inf"), it=0, bad=0, state=None)
    t0 = time.time()
    it = 0
    for it in range(1, steps + 1):
        ii, a = M.fm_draw(tr, batch, g, Lmax, L, dev)
        h, tgt, y, ok = window_batch(net, D, ii, a, L, g, variant)
        qn = head(h, tgt, raw=False)
        yn = (y - head.y_mu) / head.y_sd
        loss = (pinball(qn, yn, head.qs).sum(-1) * ok).sum() / ok.sum().clamp(
            min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if it % check == 0 or it == steps:
            r = evaluate(head, net, D, va, L, variant, n_batches=8)
            v = float(sum(r["pinball"]))
            log(f"    safety head {variant} {it:5d}: train {loss.item():.4f} "
                f"held-out {v:.4f}  cov90 APK {r['cov90'][0]:.3f} HMIN "
                f"{r['cov90'][1]:.3f}  APK>q99 {r['apk_above_q99']:.4f} "
                f"HMIN<q01 {r['hmin_below_q01']:.4f}  "
                f"({time.time() - t0:.0f} s)")
            if v < best["v"]:
                best.update(v=v, it=it, bad=0, state={
                    k: x.detach().clone() for k, x in
                    head.state_dict().items()})
            else:
                best["bad"] += 1
                if best["bad"] >= patience:
                    break
    if best["state"] is not None:
        head.load_state_dict(best["state"])
    head.eval()
    final = evaluate(head, net, D, va, L, variant, n_batches=32)
    info = dict(variant=variant, best_it=best["it"], steps_run=it, ctx=L,
                n_eps=int(D.n), held_out=final, time_s=time.time() - t0)
    log(f"    safety head {variant}: kept step {best['it']} of {it}; held-out "
        f"cov90 APK {final['cov90'][0]:.3f} HMIN {final['cov90'][1]:.3f}, "
        f"APK > q99 {final['apk_above_q99']:.4f}, HMIN < q01 "
        f"{final['hmin_below_q01']:.4f} ({final['n']} steps)")
    return head, info


def train_for_cache(cache, variant, dev, n_max=4000, steps=4000, log=print,
                    seed=0):
    """Train and save <cache>/safety_<variant>.pt for the cache's model
    checkpoint of `variant` (model3.pt for 'a', model_<v>.pt otherwise)."""
    from learn.meta import model_preview as MP
    ckf = os.path.join(cache, "model3.pt" if variant == "a"
                       else f"model_{variant}.pt")
    if not os.path.exists(ckf):
        raise FileNotFoundError(ckf)
    ck = torch.load(ckf, map_location=dev, weights_only=False)
    net = MP.load_variant(ck, dev)
    L = int(ck.get("ctx", M.W_CTX))
    D = load_split(cache, "train", dev, ck["stats"], n_max=n_max, seed=seed)
    log(f"  safety head {os.path.basename(cache)}/{variant}: {D.n} training "
        f"episodes, context {L}")
    head, info = train_head(net, D, variant, L, steps=steps, log=log,
                            seed=seed)
    info.update(ckpt=ckf, ckpt_mtime=os.path.getmtime(ckf))
    save_head(head, info, head_path(cache, variant))
    return head, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--variant", default="p")
    ap.add_argument("--n-max", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=4000)
    a = ap.parse_args()
    cache = a.cache if os.path.isabs(a.cache) else os.path.join(CACHE_ROOT,
                                                                a.cache)
    train_for_cache(cache, a.variant, M.device(), a.n_max, a.steps)


if __name__ == "__main__":
    main()
