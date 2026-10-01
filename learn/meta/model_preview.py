#!/usr/bin/env python3
"""
The one-step error model with WAVE and PREVIEW inputs (DEFECTS M16, whose
code keeps the name m15; brief BRIEF_PREVIEW.md item 4): model3's network
and training (learn/meta/model3.py, the D6 network, unchanged and still
available) with N_WX = 34 extra columns in every step token.

    columns 0:15   measured elevations at the 15 stations at t, normalised
                   by the library (lib mu / sd[11:26]); 0 when not measured
            15:30  PREVIEW of W_MID for this step (the elevations the hull
                   meets at mid-step, data5.mid_pose), same normalisation;
                   0 when there is none
            30     measured flag
            31     steps since the last measurement / HB (0 when measured)
            32     preview flag
            33     forecast-error level (lam sqrt(ph0^2 + (ph_rate lead)^2),
                   0 for a perfect preview or none)

Variants (all trained on the same M15 data, studies/meta_step5.py):
  (a) model3.Net, the D6 inputs (model3.train);
  (w) NetP, waves at t only: the preview columns are never on;
  (p) NetP, waves + preview.

ONE convention for training and the MPC rollout. A window (or a rollout
row) has a moment m, the step whose start is the last measurement:
  tokens t < m  (history): measured on (sensor noise), preview = the
                forecast issued at t for step t (lead dtc / 2);
  token  t = m: measured on, preview lead dtc / 2 -- the same as history,
                so one-step prediction at every t is "m beyond the window";
  tokens t > m  (rollout): measured off (0, flag 0), since = (t - m) / HB;
                preview = the forecast issued at m, lead (t - m) dtc +
                dtc / 2, its error growing with lead, while t - m < hp
                (the preview horizon, steps); off (0, flags 0) beyond.
  hp = 0: no preview anywhere in the window (history included).
In the MPC's rollout (rollout_core_w with a WaveRoll hook) the moment is
the current step k: the history tokens get their measured elevations and
the lead-dtc/2 forecasts, generated token 0 the measured elevations at k and
the lead-dtc/2 forecast, generated tokens j >= 1 no measurement and the
forecast at the ROLLED state's dead-reckoned mid-step pose at (k + j) dtc
+ dtc / 2 while j < hp.

Training draws per window: the moment's position m - a ~ U{0 .. valid
tokens - 1} (w.p. P_NOROLL = 0.2: m beyond the window, all tokens
measured), and the loss counts only tokens t <= m + J0_MAX (J0_MAX = HB +
8; the causal mask keeps the later, unscored tokens out of every scored
output). So rollout segments (measured off, forecast preview) are trained
at every context position the MPC's rollout puts them (the moment at
position Lh = min(k, NH), generated tokens at Lh + j), not only at the
window's end (test_m15 test 8). hp = 0 w.p. P_NOPREV = 0.2 (always for
variant w), else U{1..HP_MAX}; forecast level lam = 0 w.p. 0.15, else
LogU[0.02, 1.5]; sensor noise sd LogU[0.003, 0.03] (library units).
model3's observation noise on the error history as before.

Trained context (train_ctx): a split of episodes shorter than the window
(45-s episodes: 179 valid tokens) never trains positions >= its valid
length, nor attention over more keys (net.pos stays at its zero init
there). Every checkpoint of studies/meta_step5.py records ctx = min(W_CTX,
valid tokens of the longest training episode), and every scored context
is capped at it: one-step windows of ctx tokens (eval_one_step_p L),
rollout history NH = ctx - HB (rollout_core_w nh), wave-use histories <=
ctx - 1 (studies/eval_preview_m15.py). With 90-s training episodes ctx =
W_CTX and all of this is the uncapped default.

Forecast errors (fc_eval), the SAME generator in training, the rollout hook
and the tests: the preview comes from the episode's sea with every
component perturbed, per row, in phase by lam (ph0 xi0_c + ph_rate lead
xi1_c) + w_c lam tshift eta (a coherent time shift), and in amplitude by
(1 + amp lam eps_c), xi, eps, eta ~ N(0, 1). Real phase-resolved forecasts
err coherently (timing, per-component phase and amplitude, propagation
speed) and more with lead; iid per-station noise would erase the slope
information (the transverse difference is 0.1-0.35 of the elevation std).
For recorded tokens the preview is the recorded W_MID plus (perturbed -
true sea) at the recorded mid pose, exactly W_MID at lam = 0. The
measured elevations get an explicit sensor model: independent per station,
sd per window as above.
"""
import math
import os
import pickle
import time

import numpy as np
import torch
from torch import nn

from learn.meta import model3 as M
from learn.meta.data5 import mid_pose

N_WX = 34
C_WM, C_WP = slice(0, 15), slice(15, 30)
C_MON, C_SINCE, C_PON, C_LEV = 30, 31, 32, 33
FC = dict(ph0=1.0, ph_rate=0.5, amp=0.3, tshift=0.2)
P_NOROLL, P_NOPREV, HP_MAX, J0_MAX = 0.2, 0.2, 30, M.HB + 8
P_LAM0, LAM_RANGE, MSD_RANGE = 0.15, (0.02, 1.5), (0.003, 0.03)
VARIANTS = ("a", "w", "p")


# ------------------------------------------------------------------ data
def sea_arrays(metas):
    """(n, NC, 6) float32 [a, k, cos th, sin th, w, phi] of the episodes'
    rebuilt seas (relabel._sea), padded with a = 0."""
    from learn.meta import relabel as R
    seas = [R._sea(m) for m in metas]
    R._W.pop("seas", None)
    nc = max(s.a.size for s in seas)
    out = np.zeros((len(seas), nc, 6), np.float32)
    for i, s in enumerate(seas):
        q = s.a.size
        out[i, :q] = np.stack([s.a, s.k, np.cos(s.th), np.sin(s.th), s.w,
                               s.phi], 1)
    return out


def pad_time(D, names, T_min=M.W_CTX):
    """Pad D's time axis to T_min with invalid zero steps when it is
    shorter. model3.window_tokens indexed the previous error at t - 1 of
    every position of a W_CTX window without an upper clamp until
    2026-09-30 (fixed there: clamp(min=0, max=D.T - 1)), so a split with T
    < W_CTX (45-s episodes: 180 steps) failed with an index error in
    model3.train / eval_one_step. Kept: the padded steps are invalid
    (valid False, len unchanged) and come after every valid one, so
    nothing a valid output depends on changes (test_m15 test 8 checks
    model3.window_tokens on an unpadded short split against the padded
    one)."""
    if D.T >= T_min:
        return D
    extra = T_min - D.T
    for nm in names:
        x = getattr(D, nm)
        pad = torch.zeros((x.shape[0], extra) + tuple(x.shape[2:]),
                          dtype=x.dtype, device=x.device)
        setattr(D, nm, torch.cat([x, pad], 1))
    D.T = T_min
    return D


class Data3P(M.Data3):
    """model3.Data3 with its time axis padded to W_CTX (pad_time), for
    splits of short episodes."""

    def __init__(self, cache, split, dev, stats=None):
        super().__init__(cache, split, dev, stats=stats)
        pad_time(self, ("X", "U", "E", "valid"))


class DataP(M.Data3):
    """Data3 plus what the wave columns need: measured elevations Wm (n, T,
    15) and W_MID Wmid (normalised by the library), the dead-reckoned
    mid-step poses PM (n, T, 3) and the seas SEA (n, NC, 6). Source splits
    must be M15 packs (W_MID in the pack, op_family m15 in every meta);
    target splits need <split>_wmid.npz (studies/meta_step5.py phase wmid).
    Refuses otherwise. stats gains w_mu / w_sd (lib mu / sd[11:26])."""

    def __init__(self, cache, split, dev, stats=None):
        super().__init__(cache, split, dev, stats=stats)
        from learn.meta import relabel as R
        env = R._env()
        d = np.load(os.path.join(cache, f"{split}.npz"))
        meta = pickle.load(open(os.path.join(cache, f"{split}_meta.pkl"),
                                "rb"))
        lib = np.load(os.path.join(cache, "lib.npz"))
        st = dict(self.stats)
        if "w_mu" not in st:
            st["w_mu"] = lib["mu"][11:26].astype(np.float32)
            st["w_sd"] = lib["sd"][11:26].astype(np.float32)
        self.stats = st
        T, L = d["U"].shape[1], d["len"]
        live = (np.arange(T)[None] < L[:, None])[..., None]
        src = [m.get("world", "high") == "low" for m in meta]
        if any(src):
            if not all(src):
                raise RuntimeError(f"{split}: source and target episodes")
            fams = {m.get("op_family", "v0") for m in meta}
            if fams != {"m15"} or "W_MID" not in d.files:
                raise RuntimeError(f"{split}: not an M15 pack (op families "
                                   f"{sorted(fams)}, W_MID "
                                   f"{'W_MID' in d.files})")
            wmid = d["W_MID"]
        else:
            f = os.path.join(cache, f"{split}_wmid.npz")
            if not os.path.exists(f):
                raise RuntimeError(f"{f} missing (meta_step5 phase wmid)")
            wmid = np.load(f)["W_MID"]
            if wmid.shape[:2] != (len(L), T):
                raise RuntimeError(f"{f}: shape {wmid.shape} vs {split}")
        mu_w, sd_w = st["w_mu"], st["w_sd"]
        f32 = dict(dtype=torch.float32, device=dev)
        self.Wm = torch.tensor((d["S"][..., 11:26] - mu_w) / sd_w * live,
                               **f32)
        self.Wmid = torch.tensor((wmid - mu_w) / sd_w * live, **f32)
        self.dtc = float(env["dt"] * env["sub"])
        self.h = 0.5 * self.dtc
        xm, ym, pm = mid_pose(d["XS"][:, :T].astype(float), "plant14",
                              self.h)
        self.PM = torch.tensor(np.stack([xm, ym, pm], -1), **f32)
        self.SEA = torch.tensor(sea_arrays(meta), **f32)
        self.x_st = torch.tensor(np.asarray(env["x_st"]), **f32)
        self.y_off = torch.tensor(np.asarray(env["y_off"]), **f32)
        self.w_mu = torch.tensor(mu_w, **f32)
        self.w_sd = torch.tensor(sd_w, **f32)
        self.meta = meta
        pad_time(self, ("X", "U", "E", "valid", "Wm", "Wmid", "PM"))


def train_ctx(D):
    """The context length a training split trains: min(W_CTX, valid tokens
    of its longest episode). Positions >= it and attention over more keys
    are never trained (module docstring)."""
    return int(min(M.W_CTX, int(D.valid.sum(1).max())))


def stations_t(x, y, psi, x_st, y_off):
    """Station positions (..., 15) of poses (...) (torch), data5.station_xy
    flattened."""
    c, s = torch.cos(psi)[..., None, None], torch.sin(psi)[..., None, None]
    X = x[..., None, None] + x_st[:, None] * c - y_off[None, :] * s
    Y = y[..., None, None] + x_st[:, None] * s + y_off[None, :] * c
    return X.flatten(-2), Y.flatten(-2)


def fc_eval(SEA, X, Y, tm, lead, cfg, full=False, chunk=64):
    """Forecast of rows' seas SEA (B, NC, 6) at points X, Y (B, Mm, 15),
    times tm and leads lead (B, Mm), with the rows' perturbations cfg (lam
    (B,), xi0 / xi1 / eps (B, NC), eta (B,)). Returns (perturbed - true
    (B, Mm, 15), perturbed (B, Mm, 15) if full else None), raw metres.
    Exactly 0 error where lam = 0."""
    B, NC = SEA.shape[:2]
    err = torch.zeros_like(X)
    val = torch.zeros_like(X) if full else None
    lam = cfg["lam"].view(B, 1, 1, 1)
    sh = (cfg["eta"] * FC["tshift"]).view(B, 1, 1, 1) * lam
    ld = lead[..., None, None]
    for c0 in range(0, NC, chunk):
        sl = slice(c0, min(NC, c0 + chunk))
        a, k, ct, st_, w, phi = (SEA[:, sl, q].reshape(B, 1, 1, -1)
                                 for q in range(6))
        ph = k * (X[..., None] * ct + Y[..., None] * st_) \
            - w * tm[..., None, None] + phi
        xi0 = cfg["xi0"][:, sl].reshape(B, 1, 1, -1)
        xi1 = cfg["xi1"][:, sl].reshape(B, 1, 1, -1)
        d = lam * (FC["ph0"] * xi0 + FC["ph_rate"] * ld * xi1) + w * sh
        amp = 1.0 + FC["amp"] * lam * cfg["eps"][:, sl].reshape(B, 1, 1, -1)
        pv = (a * amp * torch.cos(ph + d)).sum(-1)
        err = err + (pv - (a * torch.cos(ph)).sum(-1))
        if full:
            val = val + pv
    return err, val


def level_feature(lam, lead):
    return lam * torch.sqrt(FC["ph0"] ** 2 + (FC["ph_rate"] * lead) ** 2)


def _rows(cfg, r):
    return {k_: (v[r] if torch.is_tensor(v) and v.dim() >= 1 else v)
            for k_, v in cfg.items()}


def wave_cols(D, ii, t, cfg, gen, noise=True):
    """The N_WX wave columns (B, L, N_WX) of recorded tokens at absolute
    steps t (B, L) of episodes ii under the per-row convention cfg (m, hp,
    lam, msd and the perturbations; module docstring). Sensor noise from
    gen (on gen's device)."""
    dev = D.dev
    B, L = t.shape
    tc = t.clamp(0, D.T - 1)
    rel = t - cfg["m"][:, None]
    meas = rel <= 0
    Wm = D.Wm[ii[:, None], tc]
    if noise:
        Wm = Wm + cfg["msd"][:, None, None] * torch.randn(
            Wm.shape, generator=gen,
            device=gen.device if gen is not None else "cpu").to(dev)
    Wm = Wm * meas[..., None]
    lead = rel.clamp(min=0).float() * D.dtc + D.h
    hp = cfg["hp"][:, None]
    pon = (rel < hp) & (hp > 0)
    Wp = D.Wmid[ii[:, None], tc]
    rows = torch.nonzero((cfg["lam"] > 0) & (cfg["hp"] > 0))[:, 0]
    if len(rows):
        pm = D.PM[ii[rows][:, None], tc[rows]]
        X, Y = stations_t(pm[..., 0], pm[..., 1], pm[..., 2], D.x_st,
                          D.y_off)
        tm = tc[rows].float() * D.dtc + D.h
        err, _ = fc_eval(D.SEA[ii[rows]], X, Y, tm, lead[rows],
                         _rows(cfg, rows))
        Wp = Wp.clone()
        Wp[rows] = Wp[rows] + err / D.w_sd
    Wp = Wp * pon[..., None]
    lev = level_feature(cfg["lam"][:, None], lead) * pon
    since = rel.clamp(min=0).float() / M.HB
    return torch.cat([Wm, Wp, meas.float()[..., None], since[..., None],
                      pon.float()[..., None], lev[..., None]], -1)


def _logu(gen, B, lo, hi):
    u = torch.rand(B, generator=gen)
    return torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))


def pert(D, B, gen):
    """Per-row forecast perturbations (CPU draws, then on D.dev)."""
    NC = D.SEA.shape[1]
    return dict(xi0=torch.randn(B, NC, generator=gen).to(D.dev),
                xi1=torch.randn(B, NC, generator=gen).to(D.dev),
                eps=torch.randn(B, NC, generator=gen).to(D.dev),
                eta=torch.randn(B, generator=gen).to(D.dev))


def draw_cfg(D, a, ok, gen, variant):
    """The training convention of B windows starting at a (B,) with valid
    mask ok (B, L) (module docstring); draws on the CPU generator gen."""
    B, L = ok.shape
    n_valid = ok.sum(1).cpu()
    # the moment anywhere among the window's valid positions (module
    # docstring); train_tokens_p scores tokens up to m + J0_MAX only
    m_rel = (torch.rand(B, generator=gen) * n_valid.float()).long().clamp(
        min=0)
    m_rel = torch.where(torch.rand(B, generator=gen) < P_NOROLL,
                        torch.full_like(m_rel, L), m_rel)
    hp = torch.randint(1, HP_MAX + 1, (B,), generator=gen)
    hp = torch.where(torch.rand(B, generator=gen) < P_NOPREV,
                     torch.zeros_like(hp), hp)
    if variant == "w":
        hp = torch.zeros_like(hp)
    lam = _logu(gen, B, *LAM_RANGE)
    lam = torch.where(torch.rand(B, generator=gen) < P_LAM0,
                      torch.zeros_like(lam), lam)
    msd = _logu(gen, B, *MSD_RANGE)
    cfg = dict(m=(a.cpu() + m_rel).to(D.dev), hp=hp.to(D.dev),
               lam=lam.to(D.dev), msd=msd.to(D.dev))
    cfg.update(pert(D, B, gen))
    return cfg


def fixed_cfg(D, m, lam, hp, msd, gen):
    """A convention with given per-row moment m (B,) and scalar (or (B,))
    lam, hp, msd, the perturbations drawn from gen (CPU)."""
    m = torch.as_tensor(m, device=D.dev).long()
    B = len(m)
    full = lambda x, dt: torch.as_tensor(x, dtype=dt).expand(B).clone(  # noqa
    ).to(D.dev)
    cfg = dict(m=m, hp=full(hp, torch.long), lam=full(lam, torch.float32),
               msd=full(msd, torch.float32))
    cfg.update(pert(D, B, gen))
    return cfg


def window_ok(D, ii, a, L):
    t = a[:, None] + torch.arange(L, device=D.dev)[None]
    return D.valid[ii[:, None], t.clamp(max=D.T - 1)] & (t < D.T)


def tokens_p(D, ii, a, L, cfg, gen, obs=True, sensor=True):
    """model3.window_tokens (with its observation noise on the error
    history when obs) plus the wave columns under cfg (with the sensor
    noise when sensor)."""
    tok, tgt, ok = M.window_tokens(D, ii, a, L, M.obs_sd(len(ii), D.dev, gen)
                                   if obs else None, gen)
    t = a[:, None] + torch.arange(L, device=D.dev)[None]
    return torch.cat([tok, wave_cols(D, ii, t, cfg, gen, sensor)], -1), \
        tgt, ok


# ----------------------------------------------------------------- model
class NetP(M.Net):
    """model3.Net with N_WX more token inputs (variants w and p)."""

    def __init__(self):
        super().__init__()
        self.inp = M.mlp(M.D_TOK + N_WX, (self.d,), self.d)


def load_variant(ck, dev):
    """The network of a checkpoint (model3.pt: Net; model_w / model_p:
    NetP), in eval mode."""
    net = (NetP() if ck.get("variant", "a") in ("w", "p") else M.Net()).to(
        dev)
    net.load_state_dict(ck["net"])
    net.eval()
    return net


# -------------------------------------------------------------- training
def train_tokens_p(D, ii, a, L, gen, variant):
    """Tokens, targets and the LOSS mask of recorded windows under a freshly
    drawn training convention (draw_cfg): valid tokens up to J0_MAX steps
    after the moment (module docstring). Returns tok, tgt, ok, cfg."""
    ok = window_ok(D, ii, a, L)
    cfg = draw_cfg(D, a, ok, gen, variant)
    tok, tgt, ok = tokens_p(D, ii, a, L, cfg, gen)
    t = a[:, None] + torch.arange(L, device=D.dev)[None]
    ok = ok & (t - cfg["m"][:, None] <= J0_MAX)
    return tok, tgt, ok, cfg


def fm_loss_p(net, D, ii, a, L, gen, variant):
    """The teacher-forced FM loss on recorded windows under a freshly drawn
    training convention (train_tokens_p)."""
    tok, tgt, ok, _ = train_tokens_p(D, ii, a, L, gen, variant)
    return M.fm_loss_h(net, net.encode(tok), tgt, ok, gen)


@torch.no_grad()
def fm_validate_p(net, D, va, L, Lmax, variant):
    gen = torch.Generator(device="cpu").manual_seed(4321)
    ls = []
    for _ in range(8):
        ii, a = M.fm_draw(va, 32, gen, Lmax, L, D.dev)
        ls.append(fm_loss_p(net, D, ii, a, L, gen, variant).item())
    return float(np.mean(ls))


def train_p(D, variant, steps=20000, batch=48, L=M.W_CTX, lr=3e-4,
            log=print, seed=0, patience=3):
    """model3.train (same schedule, pools, validation checks and early
    stopping) for NetP under the variant's convention (w or p)."""
    assert variant in ("w", "p")
    dev = D.dev
    torch.manual_seed(seed)
    net = NetP().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=max(0.05, 2.0 / steps))
    g = torch.Generator(device="cpu").manual_seed(seed)
    tr, va = M.split_pools(D.n, seed)
    Lmax = D.len.cpu()
    best = dict(v=float("inf"), it=-1, bad=0)
    t0 = time.time()
    for it in range(steps):
        ii, a = M.fm_draw(tr, batch, g, Lmax, L, dev)
        loss = fm_loss_p(net, D, ii, a, L, g, variant)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 1000 == 0 or it == steps - 1:
            v = fm_validate_p(net, D, va, L, Lmax, variant)
            log(f"    model_{variant} {it:6d}: train {loss.item():.4f}  "
                f"episodes kept out of training {v:.4f}  "
                f"({time.time() - t0:.0f} s)")
            if it > steps // 5:
                if v < best["v"]:
                    best.update(v=v, it=it, bad=0, state={
                        k_: x.detach().clone() for k_, x in
                        net.state_dict().items()})
                else:
                    best["bad"] += 1
                    if best["bad"] >= patience:
                        log(f"    not better for {patience} checks, "
                            f"stopping at step {it}")
                        break
    if best["it"] >= 0 and best["it"] != it:
        net.load_state_dict(best["state"])
        log(f"    restored step {best['it']} ({best['v']:.4f})")
    return net


# ------------------------------------------------------------ evaluation
@torch.no_grad()
def one_step_windows(Tn, L, start):
    """(a0, lo, hi): windows [a0, a0 + L) scoring steps [lo, hi), covering
    [start, Tn) once, each scored step with >= start steps of history
    after the first window. For Tn <= 2 L - start these are model3's two
    windows (0, start, L), (Tn - L, L, Tn)."""
    out = [(0, start, L)]
    lo = L
    while lo < Tn:
        hi = min(Tn, lo + L - start)
        out.append((max(0, hi - L), lo, hi))
        lo = hi
    return out


def eval_one_step_p(net, D, q, lam=0.0, hp=1, msd=0.01, n_samp=64,
                    start=40, batch=16, gen_seed=0, variant="p", L=None):
    """model3.eval_one_step for any variant: every step t >= start with its
    recorded history; variant a: model3's tokens; w / p: all tokens
    measured (moment beyond the window), preview at lead dtc / 2 with
    forecast level lam when hp > 0 (variant w: hp forced 0). L: the
    window (context) length, default W_CTX; pass the checkpoint's trained
    ctx (module docstring)."""
    dev = D.dev
    gen = torch.Generator(device=dev).manual_seed(gen_seed)
    gw = torch.Generator(device="cpu").manual_seed(gen_seed + 17)
    tal = M.Tally((M.C7,))
    L = M.W_CTX if L is None else int(L)
    assert start < L <= M.W_CTX
    for s0 in range(0, D.n, batch):
        ii = torch.arange(s0, min(D.n, s0 + batch), device=dev)
        Tn = int(D.len[ii].max())
        for a0, lo, hi in one_step_windows(Tn, L, start):
            if hi <= lo:
                continue
            a = torch.full_like(ii, a0)
            if variant == "a":
                tok, tgt, ok = M.window_tokens(D, ii, a, L)
            else:
                cfg = fixed_cfg(D, a + L, lam, 0 if variant == "w" else hp,
                                msd, gw)
                tok, tgt, ok = tokens_p(D, ii, a, L, cfg, gw, obs=False)
            h = net.encode(tok)
            pos = torch.arange(L, device=dev) + a0
            sel = ok & (pos[None] >= lo) & (pos[None] < hi)
            hs, ys = h[sel], tgt[sel]
            for c0 in range(0, len(hs), 2048):
                s = net.sample(hs[c0:c0 + 2048], n_samp, gen=gen)
                tal.add(s, ys[c0:c0 + 2048], q)
    return tal.summary()


class WaveRoll:
    """The rollout hook of rollout_core_w: the moment is each row's k; per
    row forecast level lam, preview horizon hp (steps; 0 = none), sensor
    noise msd, perturbations drawn from gen (its own CPU stream, so the
    variants keep common random numbers in the flow sampling).
    history(D, ii, a, NH): the history tokens' wave columns;
    __call__(j, sr, Uj): generated token j's (B, P, S, N_WX) from the
    rolled reduced state sr (B, P, S, 10) BEFORE step j (Uj unused)."""

    def __init__(self, D, ii, k, lam, hp, msd=0.01, gen=None):
        self.D = D
        self.gen = gen or torch.Generator(device="cpu").manual_seed(0)
        self.ii = torch.as_tensor(ii, device=D.dev).long()
        self.k = torch.as_tensor(k, device=D.dev).long()
        self.cfg = fixed_cfg(D, self.k, lam, hp, msd, self.gen)

    def history(self, D, ii, a, NH):
        t = a[:, None] + torch.arange(NH, device=D.dev)[None]
        return wave_cols(D, ii, t, self.cfg, self.gen)

    def __call__(self, j, sr, Uj=None):
        D, cfg = self.D, self.cfg
        B, P, S = sr.shape[:3]
        if j == 0:
            c = wave_cols(D, self.ii, self.k[:, None], cfg, self.gen)
            return c.view(B, 1, 1, N_WX).expand(B, P, S, N_WX)
        out = torch.zeros(B, P, S, N_WX, device=D.dev)
        out[..., C_SINCE] = j / M.HB
        pon = (j < cfg["hp"]).float()
        if pon.any():
            xm, ym, pm = mid_pose(sr, "red10", D.h)
            X, Y = stations_t(xm.float(), ym.float(), pm.float(), D.x_st,
                              D.y_off)
            lead = torch.full((B, P * S), j * D.dtc + D.h, device=D.dev)
            tm = ((self.k + j).float() * D.dtc + D.h)[:, None].expand(
                B, P * S)
            _, val = fc_eval(D.SEA[self.ii], X.reshape(B, P * S, 15),
                             Y.reshape(B, P * S, 15), tm, lead, cfg,
                             full=True)
            Wp = ((val - D.w_mu) / D.w_sd).view(B, P, S, 15)
            pv = pon.view(B, 1, 1)
            out[..., C_WP] = Wp * pv[..., None]
            out[..., C_PON] = pv
            out[..., C_LEV] = level_feature(cfg["lam"], lead[:, 0]).view(
                B, 1, 1) * pv
        return out


def rollout_core_w(net, D, ii, k, plans, xs_k, env, n_samp, base=None,
                   gen=None, grad=False, obs=None, fb_damp=None, ode_steps=24,
                   steer=None, wave=None, nh=None):
    """model3.rollout_core with an optional wave hook (a WaveRoll): its
    history columns are appended to the history tokens and wave(j, sr_j,
    U_j) to generated token j; and an optional history length nh (default
    model3's W_CTX - HB; pass the checkpoint's trained ctx - HB, module
    docstring). wave=None, nh=None is exactly model3.rollout_core
    (test_m15 test 7, bit for bit); the hook draws only from its own
    generator."""
    assert ode_steps >= 24, "the flow needs its 24-step grid"
    with torch.set_grad_enabled(grad):
        return _rollout_core_w(net, D, ii, k, plans, xs_k, env, n_samp, base,
                               gen, obs, fb_damp, ode_steps, steer, wave, nh)


def _rollout_core_w(net, D, ii, k, plans, xs_k, env, S, base, gen, obs,
                    fb_damp, ode_steps, steer=None, wave=None, nh=None):
    # model3._rollout_core, line for line, plus the two wave insertions
    # and the history length nh
    from learn.meta.data3 import E7
    dev, st = D.dev, D.stats
    iit = torch.as_tensor(ii, device=dev).long()
    kt = torch.as_tensor(k, device=dev).long()
    pl = torch.as_tensor(plans, dtype=torch.float64, device=dev)
    B, P, Hh, _ = pl.shape
    NH = M.W_CTX - M.HB if nh is None else int(nh)
    assert Hh <= M.HB and 1 <= NH <= M.W_CTX - M.HB
    gdev = gen.device if gen is not None else dev
    a = (kt - NH).clamp(min=0)
    Lh = kt - a
    tok_h, _, _ = M.window_tokens(D, iit, a, NH, obs_sd=obs, gen=gen)
    if wave is not None:
        tok_h = torch.cat([tok_h, wave.history(D, iit, a, NH)], -1)
    hist, _ = M.kv_history(net, tok_h)
    hide = torch.arange(NH, device=dev)[None] >= Lh[:, None]
    e_prev = D.E[iit, (kt - 1).clamp(min=0)] * (kt >= 1)[:, None]
    if obs is not None:
        e_prev = e_prev + obs * torch.randn(e_prev.shape, generator=gen,
                                            device=gdev).to(dev)
    e_prev = e_prev[:, None, None].expand(B, P, S, M.C7)
    xs = torch.as_tensor(xs_k, dtype=torch.float64, device=dev)
    amax = torch.tensor([env["t_max"], env["rud_max"]], dtype=torch.float64,
                        device=dev)
    act = (xs[:, 12:14] / amax)[:, None, None].expand(B, P, S, 2)
    sr = M.plant_to_reduced_t(xs)[:, None, None].expand(B, P, S, 10)
    if base is None:
        base = torch.randn((B, P, S, Hh, M.C7), generator=gen, device=gdev)
    base = base.to(dev)
    u_mu = torch.as_tensor(st["u_mu"], dtype=torch.float64, device=dev)
    u_sd = torch.as_tensor(st["u_sd"], dtype=torch.float64, device=dev)
    Un = ((pl - u_mu) / u_sd).float()
    e_sd = torch.as_tensor(st["e_sd"], device=dev)
    sel = torch.zeros(len(E7), 10, dtype=torch.float64, device=dev)
    sel[torch.arange(len(E7)), torch.tensor(E7)] = 1.0
    lo = torch.tensor([0.0, -1.0], dtype=torch.float64, device=dev)
    hi = torch.tensor([1.0, 1.0], dtype=torch.float64, device=dev)
    dtc = env["dt"] * env["sub"]
    m0 = M.model0t(env)
    n_st = len(E7)
    cache = [([], []) for _ in net.tf.layers]
    outs, states, acts = [], [], []
    for j in range(Hh):
        X = M.state_tokens_t(sr, act, st, env).float()
        first = ((Lh + j) == 0).float().view(B, 1, 1).expand(B, P, S)
        Uj = pl[:, :, None, j]
        if steer is None:
            Un_j = Un[:, :, None, j].expand(B, P, S, 2)
        else:
            Uj = steer(j, sr, Uj)
            Un_j = ((Uj - u_mu) / u_sd).float()
        tok = M.make_tokens(X, Un_j, e_prev, first)
        if wave is not None:
            tok = torch.cat([tok, wave(j, sr, Uj).float()], -1)
        h = M.kv_step(net, tok, net.pos[Lh + j].view(B, 1, 1, -1), hist,
                      hide, cache)
        e = net.sample_grad(h, base[:, :, :, j], ode_steps)
        e_raw = (e * e_sd).double()
        sr = m0(sr, Uj) + (e_raw[..., :n_st] * dtc) @ sel
        act = torch.maximum(torch.minimum(
            Uj + e_raw[..., n_st:n_st + 2] * dtc, hi), lo)
        outs.append(e)
        states.append(sr)
        acts.append(act)
        e_prev = e
        if fb_damp is not None:
            e_prev, act = M._damp(e_prev, fb_damp), M._damp(act, fb_damp)
    return torch.stack(outs, 3), torch.stack(states, 3), torch.stack(acts, 3)


@torch.no_grad()
def eval_multi_step_w(net, D, br, env, q, n_samp=16, batch=48, gen_seed=0,
                      wave_fn=None, nh=None):
    """model3.eval_multi_step through rollout_core_w (one scoring code for
    every variant): wave_fn(D, ep, k) -> a WaveRoll for the batch's
    moments, or None (variant a, model3's rollout exactly); nh: the
    rollout's history length (rollout_core_w; None = W_CTX - HB)."""
    from learn.meta.data2 import plant_to_reduced
    from learn.meta.data3 import model0_step
    dev = D.dev
    gen = torch.Generator(device=dev).manual_seed(gen_seed)
    NB, P, Hh, _ = br["U"].shape
    tal = M.Tally((Hh, M.C7))
    SC = M.SC_ES
    tal_s = M.Tally((Hh, len(SC)))
    U = br["U"].reshape(NB * P, Hh, 2)
    XS0 = br["XS"][:, :, 0].reshape(NB * P, 14)
    XT = plant_to_reduced(br["XS"][:, :, 1:].reshape(-1, 14)).reshape(
        NB * P, Hh, 10)
    E0 = br["E0"].reshape(NB * P, Hh, M.C7) / D.stats["e_sd"]
    ref_all = np.zeros_like(XT)
    for s0 in range(0, NB * P, 4096):
        sr = plant_to_reduced(XS0[s0:s0 + 4096].astype(float))
        for j in range(Hh):
            sr = model0_step(env, sr, U[s0:s0 + 4096, j].astype(float))
            ref_all[s0:s0 + 4096, j] = sr
    ssc = (XT - ref_all)[..., list(SC)].reshape(-1, len(SC)).std(0) + 1e-9
    ssc_t = torch.tensor(ssc, device=dev)
    f = lambda a: torch.as_tensor(  # noqa: E731
        a, device=dev)[..., list(SC)].div(ssc_t).float()
    per = max(1, batch // P)
    for s0 in range(0, NB, per):
        mb = np.arange(s0, min(NB, s0 + per))
        sl = (mb[:, None] * P + np.arange(P)[None]).ravel()
        wave = None if wave_fn is None else wave_fn(D, br["ep"][mb],
                                                    br["k"][mb])
        e, sts, _ = rollout_core_w(net, D, br["ep"][mb], br["k"][mb],
                                   br["U"][mb].astype(float),
                                   br["XS"][mb, 0, 0].astype(float), env,
                                   n_samp, gen=gen, wave=wave, nh=nh)
        s = e.reshape(len(sl), n_samp, Hh, M.C7)
        tal.add(s, torch.tensor(E0[sl], dtype=torch.float32, device=dev), q)
        tal_s.add(f(sts.reshape(len(sl), n_samp, Hh, 10)), f(XT[sl]),
                  ref=f(ref_all[sl]))
    out = tal.summary()
    out["states"] = tal_s.summary(chans=len(SC))
    return out
