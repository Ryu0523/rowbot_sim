#!/usr/bin/env python3
"""
What does a bow camera tell about the next waves and the next impact, over
and above what the hull's own history tells? Learned predictors on the
frames of studies/bow_camera.py, split by wave realisation (sea seed 0
trains, sea seed 1 tests), all on the CPU.

INPUTS (one predictor per row, plus combinations)
  image     the stabilised range image (log range per ray): the geometric
            information ceiling for an optical sensor at this height
  grey      the sketched grey picture (appearance): clean, and degraded
            at test time (exposure, noise, blur, spray, horizon jitter)
  past      the elevation at the bow over the last 20 s, known exactly (the
            best any motion-based estimate could give)
  motion    heave, pitch, speed over the last 10 s (what the IMU gives)
TARGETS
  eta(tau)  the elevation the bow meets tau = 0.25 .. 10 s ahead (in Hs/4)
  imp       log(1 + largest filtered CG acceleration in g) in the next
            2.5 / 5 / 10 s
  h_next    largest zero-crossing wave height met in the next 8 s, over Hs
SCORES on the test runs: R^2 per horizon for eta, R^2 for imp, and the
detection quality (area under the ROC curve) of "h_next > 1.3" and
"imp(5 s) > 3 g" from the predicted values.

    python -m studies.bow_camera_train [--epochs 40]

Outputs: studies/_cache/camera/results.md, results.json, r2_horizon.png,
sample_frames.png.
"""
import glob
import json
import math
import os
import sys
import time

import os as _os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

CAM = os.path.join(ROOT, "studies", "_cache", "camera")
STEP = 0.25
TEST_SEA_SEED = 1
RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ data
def load_all():
    runs = []
    for p in sorted(glob.glob(os.path.join(CAM, "*.npz"))):
        z = np.load(p)
        meta = json.loads(str(z["meta"]))
        hs = float(z["hs"])
        sc = hs / 4.0
        d = dict(
            name=meta["name"], sea_seed=int(meta["sea_seed"]),
            kind=meta["kind"], hs=hs,
            img=((np.asarray(z["rng_img"], np.float32) - math.log(50.0))
                 / 1.5)[:, None],
            grey=(np.asarray(z["grey_img"], np.float32) - 0.5)[:, None],
            past=np.asarray(z["past_eta"], np.float32) / sc,
            motion=np.stack([z["motion"][:, 0] / sc,
                             z["motion"][:, 1] / 0.05,
                             (z["motion"][:, 2] - 8.0) / 4.0], 1).reshape(
                len(z["t"]), -1).astype(np.float32),
            fut=np.asarray(z["fut_eta"], np.float32) / sc,
            imp=np.log1p(np.maximum(np.asarray(z["imp_max"], np.float32),
                                    0.0)),
            h_next=np.asarray(z["h_next"], np.float32),
            speed=np.asarray(z["speed"], np.float32))
        runs.append(d)
    return runs


def stack(runs, key):
    return np.concatenate([r[key] for r in runs], 0)


def split(runs):
    tr = [r for r in runs if r["sea_seed"] != TEST_SEA_SEED]
    te = [r for r in runs if r["sea_seed"] == TEST_SEA_SEED]
    return tr, te


# ----------------------------------------------------------- degradation
def degrade(img, rng, level=1.0):
    """Camera-side degradations of a batch of grey images (N, 1, H, W) in
    [-0.5, 0.5]: exposure and gamma, additive noise, 3x3 blur, spray discs
    in the lower rows, a glare blob, horizon jitter (rows shifted)."""
    x = img.copy() + 0.5
    n, _, h, w = x.shape
    gain = rng.uniform(1 - 0.4 * level, 1 + 0.5 * level, (n, 1, 1, 1))
    gam = rng.uniform(1 - 0.3 * level, 1 + 0.3 * level, (n, 1, 1, 1))
    x = np.clip(x * gain, 0, 1) ** gam
    x = x + rng.normal(0, 0.05 * level, x.shape)
    if level > 0:
        xb = x.copy()
        xb[:, :, 1:-1, 1:-1] = (x[:, :, :-2, :-2] + x[:, :, :-2, 1:-1]
                                + x[:, :, :-2, 2:] + x[:, :, 1:-1, :-2]
                                + x[:, :, 1:-1, 1:-1] + x[:, :, 1:-1, 2:]
                                + x[:, :, 2:, :-2] + x[:, :, 2:, 1:-1]
                                + x[:, :, 2:, 2:]) / 9.0
        blur = rng.random(n) < 0.5 * level
        x[blur] = xb[blur]
    yy, xx = np.mgrid[0:h, 0:w]
    for i in range(n):
        if rng.random() < 0.35 * level:               # spray
            for _ in range(rng.integers(1, 4)):
                cy = rng.uniform(h * 0.5, h - 1)
                cx = rng.uniform(0, w - 1)
                r = rng.uniform(1.5, 4.0)
                m = (yy - cy) ** 2 + (xx - cx) ** 2 < r * r
                x[i, 0][m] = 0.95
        if rng.random() < 0.3 * level:                # glare blob
            cy, cx = rng.uniform(0, h * 0.5), rng.uniform(0, w - 1)
            s = rng.uniform(3, 7)
            x[i, 0] += 0.5 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2)
                                    / (2 * s * s))
        k = int(rng.integers(-2, 3)) if level > 0 else 0   # horizon jitter
        if k:
            x[i, 0] = np.roll(x[i, 0], k, axis=0)
    return (np.clip(x, 0, 1) - 0.5).astype(np.float32)


# ---------------------------------------------------------------- models
def build(kind, n_img_feat=128, n_out=44):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            parts = kind.split("+")
            feat = 0
            if "image" in parts or "grey" in parts:
                self.cnn = nn.Sequential(
                    nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                    nn.AdaptiveAvgPool2d((3, 4)), nn.Flatten(),
                    nn.Linear(64 * 12, n_img_feat), nn.ReLU())
                feat += n_img_feat
            if "past" in parts:
                self.past = nn.Sequential(nn.Linear(80, 128), nn.ReLU(),
                                          nn.Linear(128, 64), nn.ReLU())
                feat += 64
            if "motion" in parts:
                self.motion = nn.Sequential(nn.Linear(120, 128), nn.ReLU(),
                                            nn.Linear(128, 64), nn.ReLU())
                feat += 64
            if "future" in parts:        # the exact future wave (privileged)
                self.future = nn.Sequential(nn.Linear(40, 128), nn.ReLU(),
                                            nn.Linear(128, 64), nn.ReLU())
                feat += 64
            self.head = nn.Sequential(nn.Linear(feat, 128), nn.ReLU(),
                                      nn.Linear(128, n_out))

        def forward(self, img, past, motion, fut):
            fs = []
            if hasattr(self, "cnn"):
                fs.append(self.cnn(img))
            if hasattr(self, "past"):
                fs.append(self.past(past))
            if hasattr(self, "motion"):
                fs.append(self.motion(motion))
            if hasattr(self, "future"):
                fs.append(self.future(fut))
            import torch
            return self.head(torch.cat(fs, 1))
    return Net()


def targets(runs):
    return np.concatenate([stack(runs, "fut"), stack(runs, "imp"),
                           stack(runs, "h_next")[:, None]], 1)


def train(kind, tr, te, epochs=40, batch=256, lr=1e-3, img_key="img",
          augment=False, seed=0, verbose=True):
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() // 4)))
    Xi = stack(tr, img_key)
    Xp, Xm, Xf = stack(tr, "past"), stack(tr, "motion"), stack(tr, "fut")
    Y = targets(tr)
    ym, ys = Y.mean(0), Y.std(0) + 1e-6
    n = len(Y)
    idx = RNG.permutation(n)
    n_val = max(1, n // 10)
    va, trn = idx[:n_val], idx[n_val:]
    net = build(kind)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a))   # noqa: E731
    Yn = (Y - ym) / ys
    best, best_state, bad = 1e9, None, 0
    rng = np.random.default_rng(seed + 1)
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(trn)
        tot = 0.0
        for b in range(0, len(perm), batch):
            ii = perm[b:b + batch]
            xi = Xi[ii]
            if augment:
                xi = degrade(xi, rng, level=rng.uniform(0.0, 1.0))
            pred = net(T(xi), T(Xp[ii]), T(Xm[ii]), T(Xf[ii]))
            loss = ((pred - T(Yn[ii].astype(np.float32))) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(ii)
        net.eval()
        with torch.no_grad():
            pv = net(T(Xi[va]), T(Xp[va]), T(Xm[va]), T(Xf[va])).numpy()
        vl = float(((pv - Yn[va]) ** 2).mean())
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"    {kind}: epoch {ep} train {tot / len(trn):.3f} val "
                  f"{vl:.3f}", flush=True)
        if bad >= 8:
            break
    net.load_state_dict(best_state)
    tag = kind + ("_aug" if augment else "") + ("" if img_key == "img"
                                                 else "_" + img_key)
    torch.save(dict(state=net.state_dict(), ym=ym, ys=ys, kind=kind,
                    img_key=img_key, augment=augment),
               os.path.join(CAM, f"net_{tag}.pt"))
    return net, (ym, ys)


def predict(net, norm, runs, img_key="img", degrade_level=None, seed=0):
    import torch
    ym, ys = norm
    Xi = stack(runs, img_key)
    if degrade_level is not None:
        Xi = degrade(Xi, np.random.default_rng(seed + 7), degrade_level)
    Xp, Xm, Xf = stack(runs, "past"), stack(runs, "motion"), stack(runs, "fut")
    net.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(Xi), 2048):
            T = lambda a: torch.from_numpy(np.ascontiguousarray(a))  # noqa
            out.append(net(T(Xi[b:b + 2048]), T(Xp[b:b + 2048]),
                           T(Xm[b:b + 2048]), T(Xf[b:b + 2048])).numpy())
    return np.concatenate(out, 0) * ys + ym


# ---------------------------------------------------------------- scores
def r2(y, p):
    ss = ((y - y.mean(0)) ** 2).sum(0)
    return 1.0 - ((y - p) ** 2).sum(0) / np.maximum(ss, 1e-12)


def auc(score, label):
    """Area under the ROC curve by the rank statistic."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def scores(P, Yte):
    nf = 40
    r2_eta = r2(Yte[:, :nf], P[:, :nf])
    r2_imp = r2(Yte[:, nf:nf + 3], P[:, nf:nf + 3])
    r2_h = float(r2(Yte[:, nf + 3:nf + 4], P[:, nf + 3:nf + 4])[0])
    auc_h = auc(P[:, nf + 3], Yte[:, nf + 3] > 1.3)
    auc_imp = auc(P[:, nf + 1], Yte[:, nf + 1] > math.log1p(3.0))
    return dict(r2_eta=r2_eta.tolist(),
                r2_eta_at=dict(s1=float(r2_eta[3]), s2=float(r2_eta[7]),
                               s4=float(r2_eta[15]), s6=float(r2_eta[23]),
                               s8=float(r2_eta[31]), s10=float(r2_eta[39])),
                r2_imp=dict(w2_5=float(r2_imp[0]), w5=float(r2_imp[1]),
                            w10=float(r2_imp[2])),
                r2_hnext=r2_h, auc_hnext13=auc_h, auc_imp5_3g=auc_imp)


def ridge_past(tr, te, lam=1e-2):
    """Linear least squares from the past elevation to the targets (the
    autoregressive baseline)."""
    X = stack(tr, "past")
    Y = targets(tr)
    A = np.column_stack([np.ones(len(X)), X])
    W = np.linalg.solve(A.T @ A + lam * len(X) * np.eye(A.shape[1]),
                        A.T @ Y)
    Xt = stack(te, "past")
    return np.column_stack([np.ones(len(Xt)), Xt]) @ W


# ---------------------------------------------------------------- report
def main(epochs=40):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:            # noqa: BLE001
        pass
    runs = load_all()
    tr, te = split(runs)
    print(f"{len(runs)} runs: {len(tr)} train ({sum(len(r['t']) if 't' in r else len(r['fut']) for r in tr)} frames), "
          f"{len(te)} test ({sum(len(r['fut']) for r in te)} frames)",
          flush=True)
    Yte = targets(te)
    res = {}
    t0 = time.time()
    res["persistence"] = scores(
        np.column_stack([np.repeat(stack(te, "past")[:, -1:], 40, 1),
                         np.tile(stack(tr, "imp").mean(0), (len(Yte), 1)),
                         np.full((len(Yte), 1), stack(tr, "h_next").mean())]),
        Yte)
    res["past (linear)"] = scores(ridge_past(tr, te), Yte)
    print(f"  baselines done {time.time() - t0:.0f} s", flush=True)
    nets = {}
    for kind in ("future+motion", "motion", "past", "image",
                 "image+past+motion", "grey", "grey+past+motion"):
        key = "grey" if kind.startswith("grey") else "img"
        net, norm = train(kind, tr, te, epochs=epochs, img_key=key)
        nets[kind] = (net, norm)
        res[kind] = scores(predict(net, norm, te, img_key=key), Yte)
        print(f"  {kind}: eta R2 at 2/4/8 s "
              f"{res[kind]['r2_eta_at']['s2']:.2f}/"
              f"{res[kind]['r2_eta_at']['s4']:.2f}/"
              f"{res[kind]['r2_eta_at']['s8']:.2f}; imp(5 s) R2 "
              f"{res[kind]['r2_imp']['w5']:.2f}; AUC h>1.3 "
              f"{res[kind]['auc_hnext13']:.2f}  [{time.time() - t0:.0f} s]",
              flush=True)
    # degradation: the clean-trained grey net on degraded frames, and a
    # net trained with the degradations as augmentation
    net, norm = nets["grey"]
    for lev in (0.5, 1.0):
        res[f"grey, degraded x{lev:g} (trained clean)"] = scores(
            predict(net, norm, te, img_key="grey", degrade_level=lev), Yte)
    net_a, norm_a = train("grey", tr, te, epochs=epochs, img_key="grey",
                          augment=True)
    res["grey, trained with degradations, clean test"] = scores(
        predict(net_a, norm_a, te, img_key="grey"), Yte)
    for lev in (0.5, 1.0):
        res[f"grey, trained with degradations, degraded x{lev:g}"] = scores(
            predict(net_a, norm_a, te, img_key="grey", degrade_level=lev),
            Yte)
    print(f"  degradation study done {time.time() - t0:.0f} s", flush=True)
    write_report(res, runs, tr, te)
    sample_figure(te)
    r2_figure(res)
    return res


NAME_CN = {"future+motion": "未来浪面精确已知 + 运动（信息上限）",
           "persistence": "保持当前值（不预测）",
           "past (linear)": "只知过去·线性外推",
           "motion": "船体运动历史（惯导）",
           "past": "只知过去·神经网络",
           "image": "几何距离图（摄像头信息上限）",
           "image+past+motion": "距离图 + 过去 + 运动",
           "grey": "灰度图（外观，干净）",
           "grey+past+motion": "灰度图 + 过去 + 运动"}


def write_report(res, runs, tr, te):
    md = ["# 船头摄像头能看出什么：预测结果", "",
          f"生成时间 {time.strftime('%Y-%m-%d %H:%M')}。训练 {len(tr)} 次运行"
          f"（波浪种子 0），测试 {len(te)} 次运行（波浪种子 1）。"
          "R² 为决定系数（1 = 完美，0 = 不比均值好）；AUC 为判别“8 秒内会遇到"
          "高于 1.3 倍有义波高的浪”和“5 秒内重心加速度峰超过 3 g”的能力"
          "（1 = 完美，0.5 = 随机）。", "",
          "| 输入 | 浪面 R²：1 s | 2 s | 4 s | 6 s | 8 s | 10 s | "
          "冲击 R²：2.5 s | 5 s | 10 s | 下个大浪 AUC | 3 g 冲击 AUC |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in res.items():
        a = v["r2_eta_at"]
        b = v["r2_imp"]
        md.append(f"| {NAME_CN.get(k, k)} | {a['s1']:.2f} | {a['s2']:.2f} | "
                  f"{a['s4']:.2f} | {a['s6']:.2f} | {a['s8']:.2f} | "
                  f"{a['s10']:.2f} | {b['w2_5']:.2f} | {b['w5']:.2f} | "
                  f"{b['w10']:.2f} | {v['auc_hnext13']:.2f} | "
                  f"{v['auc_imp5_3g']:.2f} |")
    md.append("")
    with open(os.path.join(CAM, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(os.path.join(CAM, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("\n".join(md))


def sample_figure(te):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = te[0]
    i = int(np.argmax(r["h_next"]))             # the frame before the biggest wave
    j = int(np.argmin(np.abs(r["h_next"] - np.median(r["h_next"]))))
    fig, axes = plt.subplots(2, 3, figsize=(11, 5))
    for row, (idx, lab) in enumerate(((i, "before the largest wave"),
                                      (j, "typical"))):
        axes[row, 0].imshow(r["img"][idx, 0] * 1.5 + math.log(50),
                            aspect="auto", cmap="viridis")
        axes[row, 0].set_title(f"log range, {lab}")
        axes[row, 1].imshow(r["grey"][idx, 0] + 0.5, aspect="auto",
                            cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("grey (clean)")
        dg = degrade(r["grey"][idx:idx + 1], np.random.default_rng(3), 1.0)
        axes[row, 2].imshow(dg[0, 0] + 0.5, aspect="auto", cmap="gray",
                            vmin=0, vmax=1)
        axes[row, 2].set_title("grey (degraded)")
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"{r['name']}: rows = depression angle (top: horizon), "
                 "columns = azimuth")
    fig.tight_layout()
    fig.savefig(os.path.join(CAM, "sample_frames.png"), dpi=120)
    plt.close(fig)


def r2_figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tau = STEP * np.arange(1, 41)
    fig, ax = plt.subplots(figsize=(7, 4))
    for k in ("past (linear)", "motion", "past", "image",
              "image+past+motion", "grey"):
        if k in res:
            ax.plot(tau, res[k]["r2_eta"], label=k)
    ax.set_xlabel("horizon (s)")
    ax.set_ylabel("R^2 of the elevation the bow meets")
    ax.set_ylim(-0.1, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(CAM, "r2_horizon.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    ep = 40
    if "--epochs" in sys.argv:
        ep = int(sys.argv[sys.argv.index("--epochs") + 1])
    main(epochs=ep)
