"""Why is yaw not learned on the target (split C / Cb)? Descriptive
diagnostics: (1) the yaw error's character, target vs training; (2) what
it goes with (correlation, lags 0-3); (3) do the recorded feedback commands
of split C carry future yaw errors (the old evaluation's leak)?; (4) the
one-step model's rollouts on Cb: bias of the yaw samples and of the
predicted nozzle position, and the heading drift."""
import os
import sys

sys.path.insert(0, r"C:\Users\Administrator\Documents\usv-rl-mpc")
os.chdir(r"C:\Users\Administrator\Documents\usv-rl-mpc")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
import numpy as np

C = "studies/_cache/meta2/"
Y = 2          # yaw channel of e


def load(split):
    d = np.load(C + f"{split}.npz")
    e = np.load(C + f"{split}_e0.npz")["E0"]
    L = d["len"]
    T = d["U"].shape[1]
    n_e = L - 1 if d["XS"].shape[1] == T else L
    return d, e, n_e


def pooled(split, n_max=400):
    d, e, n_e = load(split)
    rows = []
    for i in range(min(n_max, len(n_e))):
        n = int(n_e[i])
        if n < 60:
            continue
        xs = d["XS"][i, 40:n]
        u = d["U"][i, 40:n]
        s = d["S"][i, 40:n]
        ee = e[i, 40:n]
        rows.append(dict(
            yaw=ee[:, Y], sway_e=ee[:, 1],
            roll=xs[:, 3], roll_rate=xs[:, 9], v=xs[:, 7], r=xs[:, 11],
            u=xs[:, 6], w=xs[:, 8], q=xs[:, 10],
            noz_act=xs[:, 13], noz_cmd=u[:, 1], thr_act=xs[:, 12],
            noz_gap=ee[:, 9],
            jet_side=xs[:, 12] * np.sin(xs[:, 13]),
            wave_lat=(s[:, 11 + 2::3].mean(1) - s[:, 11::3].mean(1)),
            wave_twist=((s[:, 11 + 2::3] - s[:, 11::3])
                        * np.linspace(-1, 1, 5)).sum(1)))
    return rows


def ac1(x):
    x = x - x.mean()
    return float((x[1:] @ x[:-1]) / max(x @ x, 1e-20))


def corr_lag(rows, a, b, lag):
    xs, ys = [], []
    for r in rows:
        x, y = r[a], r[b]
        if lag > 0:
            x, y = x[:-lag], y[lag:]
        xs.append(x - x.mean())
        ys.append(y - y.mean())
    x, y = np.concatenate(xs), np.concatenate(ys)
    return float((x @ y) / np.sqrt((x @ x) * (y @ y) + 1e-20))


print("(1) the yaw error's character")
for split in ("A", "C", "Cb"):
    rows = pooled(split)
    y = np.concatenate([r["yaw"] for r in rows])
    a1 = np.mean([ac1(r["yaw"]) for r in rows])
    top = np.sort(np.abs(y))[::-1]
    share = (top[:max(1, len(top) // 100)] ** 2).sum() / (top ** 2).sum()
    rr = np.concatenate([r["roll"] for r in rows])
    print(f"  {split:<3} rms {np.sqrt((y ** 2).mean()):.3f} rad/s^2, lag-1 "
          f"autocorrelation {a1:.2f}, largest 1% share of power {share:.2f};"
          f" roll rms {np.sqrt((rr ** 2).mean()):.4f} rad")

print("\n(2) correlation of the yaw error with ... (the other quantity leads "
      "by 0 / 1 / 3 steps)")
names = ["roll", "roll_rate", "noz_gap", "noz_act", "noz_cmd", "jet_side",
         "v", "r", "u", "w", "q", "wave_lat", "wave_twist", "sway_e"]
for split in ("A", "Cb"):
    rows = pooled(split)
    print(f"  {split}:")
    for nm in names:
        vals = [corr_lag(rows, nm, "yaw", lag) for lag in (0, 1, 3)]
        if max(abs(v) for v in vals) >= 0.1 or split == "Cb":
            print(f"    {nm:<11} " + " ".join(f"{v:+.2f}" for v in vals))

print("\n(3) split C (feedback commands): does the nozzle command react to "
      "yaw errors? correlation of the yaw error at t with the nozzle "
      "command change at t+1..t+4")
d, e, n_e = load("C")
for lag in (1, 2, 4):
    xs, ys = [], []
    for i in range(len(n_e)):
        n = int(n_e[i])
        ye = e[i, 40:n - lag, Y]
        du = d["U"][i, 40 + lag:n, 1] - d["U"][i, 40 + lag - 1:n - 1, 1]
        xs.append(ye - ye.mean())
        ys.append(du - du.mean())
    x, y = np.concatenate(xs), np.concatenate(ys)
    print(f"  lag {lag}: {float((x @ y) / np.sqrt((x @ x) * (y @ y))):+.2f}")
# the same for the block split (commands fixed within 24-step blocks)
d2, e2, n2 = load("Cb")
xs, ys = [], []
for i in range(len(n2)):
    n = int(n2[i])
    for k in range(48, n - 24, 24):
        ye = e2[i, k:k + 20, Y]
        du = d2["U"][i, k + 1:k + 21, 1] - d2["U"][i, k:k + 20, 1]
        xs.append(ye - ye.mean())
        ys.append(du - du.mean())
x, y = np.concatenate(xs), np.concatenate(ys)
print(f"  Cb (within blocks, lag 1): {float((x @ y) / np.sqrt((x @ x) * (y @ y) + 1e-20)):+.2f}")

print("\n(4) the one-step model's rollouts on Cb")
import torch

from learn.meta import model3 as M
from learn.meta import relabel
from learn.meta.data2 import plant_to_reduced
dev = M.device()
ck = torch.load(C + "model3.pt", weights_only=False)
net = M.Net().to(dev)
net.load_state_dict(ck["net"])
net.eval()
st = ck["stats"]
env = relabel._env()
D = M.Data3(C, "Cb", dev, stats=st)
E0all = e2
cases = []
for i in range(D.n):
    for k in range(48, int(D.len[i]) - M.HB + 1, M.HB):
        cases.append((i, k))
rng = np.random.default_rng(0)
pick = [cases[j] for j in rng.choice(len(cases), 240, replace=False)]
ys_pred, ys_true, act_pred, act_true, psi_m, psi_t, psi_0 = [], [], [], [], [], [], []
gen = torch.Generator(device=dev).manual_seed(0)
for kv in sorted(set(k for _, k in pick)):
    sub = [(i, k) for i, k in pick if k == kv]
    ii = np.array([i for i, _ in sub])
    kk = np.array([k for _, k in sub])
    plans = np.stack([D.Uraw[i, k:k + M.HB] for i, k in sub]).astype(float)
    xs0 = np.stack([D.XS[i, k] for i, k in sub]).astype(float)
    s, sts = M.rollout(net, D, ii, kk, plans, xs0, env, n_samp=16, gen=gen)
    s = s.cpu().numpy() * st["e_sd"]
    ys_pred.append(s[:, :, :, Y].mean(1))
    ys_true.append(np.stack([E0all[i, k:k + M.HB, Y] for i, k in sub]))
    act_pred.append(s[:, :, :, 9].mean(1))
    act_true.append(np.stack([E0all[i, k:k + M.HB, 9] for i, k in sub]))
    psi_m.append(sts[:, :, :, 7].mean(1))
    psi_t.append(np.stack([plant_to_reduced(D.XS[i, k + 1:k + M.HB + 1].astype(float))[:, 7] for i, k in sub]))
    from learn.meta.data3 import model0_step
    sr = plant_to_reduced(xs0)
    p0 = []
    for j in range(M.HB):
        sr = model0_step(env, sr, plans[:, j])
        p0.append(sr[:, 7])
    psi_0.append(np.stack(p0, 1))
yp, yt = np.concatenate(ys_pred), np.concatenate(ys_true)
ap, at = np.concatenate(act_pred), np.concatenate(act_true)
pm, pt, p0 = np.concatenate(psi_m), np.concatenate(psi_t), np.concatenate(psi_0)
for a, b in ((0, 1), (1, 5), (5, 24)):
    print(f"  steps {a}-{b - 1}: yaw error mean true {yt[:, a:b].mean():+.3f}"
          f" predicted {yp[:, a:b].mean():+.3f}; corr(pred mean, true) "
          f"{np.corrcoef(yp[:, a:b].ravel(), yt[:, a:b].ravel())[0, 1]:+.2f};"
          f" rms true {np.sqrt((yt[:, a:b] ** 2).mean()):.3f}, rms of the "
          f"predicted mean {np.sqrt((yp[:, a:b] ** 2).mean()):.3f}")
    print(f"               nozzle gap mean true {at[:, a:b].mean():+.3f} "
          f"predicted {ap[:, a:b].mean():+.3f}; corr "
          f"{np.corrcoef(ap[:, a:b].ravel(), at[:, a:b].ravel())[0, 1]:+.2f}")
    print(f"               heading error rms: rollout mean "
          f"{np.sqrt(((pm - pt)[:, a:b] ** 2).mean()):.3f} rad, no-error "
          f"model {np.sqrt(((p0 - pt)[:, a:b] ** 2).mean()):.3f} rad; mean "
          f"signed (rollout - true) {(pm - pt)[:, a:b].mean():+.3f}")
