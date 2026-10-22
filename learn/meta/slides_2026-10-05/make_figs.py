#!/usr/bin/env python3
"""Figures for the 10-slide weekly update (learn/meta/SLIDES_2026-10-05.md).
Mostly schematics; three small charts use numbers from the project logs
(fine-tuning: studies/_cache/onestep_gz/batch.log; closed loop:
studies/_cache/final_eval/meta7/report.txt, impact factor k = 2).
    python learn/meta/slides_2026-10-05/make_figs.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.patches import (Ellipse, FancyArrowPatch,  # noqa: E402
                                FancyBboxPatch, Polygon, Rectangle)
import numpy as np   # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
VIOLET, RED = "#4a3aa7", "#e34948"
FILL, HULL = "#f2f1ed", "#dfe3e8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": "white", "axes.facecolor": "white"})
NL = "\n"


def box(ax, x, y, w, h, text, fc=FILL, ec=MUTED, fs=10, color=INK, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,"
                                "rounding_size=0.04", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=color)


def arrow(ax, p, q, color=INK2, lw=1.5, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=14,
                                 color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}"))


def canvas(w=6.4, h=4.8):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    return fig, ax


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight",
                pad_inches=0.08)
    plt.close(fig)


# 1 ------------------------------------------------------------ task + target
def fig1():
    fig, ax = canvas()
    box(ax, 0.2, 5.4, 2.4, 1.3, "Boat in waves", fc="#e3eefb", ec=BLUE, fs=11)
    box(ax, 3.9, 4.3, 5.9, 3.0, "", fc="white")
    ax.text(6.85, 6.95, "MPC (every 0.24 s)", ha="center", fontsize=11,
            color=INK, weight="bold")
    box(ax, 4.1, 5.6, 1.4, 0.9, "128 thrust" + NL + "plans", fs=9)
    box(ax, 5.85, 5.6, 2.0, 0.9, "reduced model" + NL + "+ learned error",
        fc="#e8f7f1", ec=AQUA, fs=9)
    box(ax, 8.2, 5.6, 1.45, 0.9, "progress" + NL + "+ limits", fs=9)
    arrow(ax, (5.5, 6.05), (5.85, 6.05))
    arrow(ax, (7.85, 6.05), (8.2, 6.05))
    ax.text(6.85, 4.75, "5.8 s rollouts, choose the best plan", ha="center",
            fontsize=9.5, color=INK2)
    arrow(ax, (2.6, 6.3), (3.9, 6.3))
    ax.text(3.25, 6.45, "state", ha="center", fontsize=9.5, color=INK2)
    arrow(ax, (3.9, 4.6), (2.0, 5.4), rad=-0.2)
    ax.text(2.4, 4.5, "thrust", ha="center", fontsize=9.5, color=INK2)
    ax.text(0.2, 3.45, "One-step error target", fontsize=11, weight="bold",
            color=INK)
    t0, t1 = 1.0, 8.4
    for t in (t0, t1):
        ax.plot([t, t], [0.6, 3.0], color=GRID, lw=1)
    ax.text(t0, 0.3, "t", ha="center", color=INK2)
    ax.text(t1, 0.3, "t + 0.24 s", ha="center", color=INK2)
    xs = np.linspace(t0, t1, 50)
    ax.plot(xs, 1.2 + 1.3 * ((xs - t0) / (t1 - t0)) ** 1.6, color=BLUE, lw=2.4)
    ax.plot(xs, 1.2 + 0.45 * (xs - t0) / (t1 - t0), color=MUTED, lw=2.4,
            ls="--")
    ax.plot([t0], [1.2], "o", color=INK, ms=6)
    arrow(ax, (t1 + 0.35, 1.67), (t1 + 0.35, 2.48), color=ORANGE,
          style="<|-|>", lw=1.6)
    ax.text(t1 + 0.5, 2.08, "error" + NL + "× 0.24 s", color=ORANGE,
            fontsize=9.5, va="center")
    ax.text(t1 - 0.3, 2.62, "true boat", color=BLUE, fontsize=9.5, ha="right")
    ax.text(t1 - 0.3, 1.5, "reduced model", color=INK2, fontsize=9.5,
            ha="right", va="top")
    ax.set_xlim(0, 10.3)
    save(fig, "fig01_task_target.png")


# 2 ------------------------------------------------------------ predictor + priors
def fig2():
    fig, ax = canvas()
    box(ax, 0.1, 5.7, 2.0, 1.2, "random" + NL + "error law", fc="#fdf0e9",
        ec=ORANGE)
    box(ax, 2.6, 5.7, 2.0, 1.2, "simulated" + NL + "episodes")
    box(ax, 5.1, 5.7, 4.8, 1.2, "Transformer, 1 min history" + NL +
        "→ error samples + safety quantiles", fc="#e8f7f1", ec=AQUA, fs=9.5)
    arrow(ax, (2.1, 6.3), (2.6, 6.3))
    arrow(ax, (4.6, 6.3), (5.1, 6.3))
    ax.text(5.0, 5.2, "repeat for ~10,000 laws  →  learns to recognise a law "
            "from history", ha="center", fontsize=9.5, color=INK2)
    ax.text(0.1, 4.45, "Three random families", fontsize=11, weight="bold",
            color=INK)
    ax.text(9.9, 4.45, "shaded: where the probability mass sits", ha="right",
            fontsize=8, color=INK2)
    arrow(ax, (0.4, 0.55), (9.7, 0.55), color=MUTED, lw=1.2)
    ax.text(0.4, 0.12, "structured, narrow", fontsize=9.5, color=INK2)
    ax.text(9.7, 0.12, "unstructured, wide", fontsize=9.5, color=INK2,
            ha="right")
    for xc, name, sub, w, col in (
            (1.7, "projection", "built around the" + NL + "reduced model",
             1.6, MUTED),
            (5.0, "catalogue", "~40 named physical" + NL + "mechanisms + "
             "small" + NL + "random layer", 2.2, BLUE),
            (8.3, "general", "random filters +" + NL + "random networks", 2.7,
             MUTED)):
        ax.add_patch(Ellipse((xc, 2.85), w, 1.35, fc="white", ec=col, lw=1.6))
        ax.add_patch(Ellipse((xc - 0.1 * w, 2.9), 0.45 * w, 0.7, fc=col,
                             alpha=0.3, ec="none"))
        ax.text(xc, 3.7, name, ha="center", fontsize=10.5, color=INK,
                weight="bold")
        ax.text(xc, 2.05, sub, ha="center", va="top", fontsize=8.5,
                color=INK2)
    ax.text(5.0, 4.05, "current network", ha="center", fontsize=8.5,
            color=BLUE)
    save(fig, "fig02_predictor_priors.png")


# 3 ------------------------------------------------------------ fine-tuning
def fig3():
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    n = [0, 2, 4, 6]
    full = [1.0, 0.350, 0.283, 0.225]
    head = [1.0, 0.547, 0.435, 0.319]
    ax.axhline(1.0, color=MUTED, lw=1, ls="--")
    ax.text(8.1, 1.03, "frozen network", va="bottom", ha="right",
            fontsize=10, color=INK2)
    ax.plot(n, full, "-o", color=BLUE, lw=2, ms=8)
    ax.plot(n, head, "-o", color=ORANGE, lw=2, ms=8)
    ax.text(6.3, full[-1], "whole network", va="center", fontsize=10,
            color=INK)
    ax.text(6.3, head[-1], "output head only", va="center", fontsize=10,
            color=INK)
    ax.set_xticks(n)
    ax.set_xlim(-0.3, 8.2)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("target missions used for fine-tuning (2 min each)")
    ax.set_ylabel("error relative to the frozen network")
    ax.grid(axis="y", color=GRID)
    clean(ax)
    ax.set_title("Fine-tuning on high-fidelity missions", fontsize=11,
                 color=INK, loc="left")
    save(fig, "fig03_target_data.png")


# 4 ------------------------------------------------------------ closed loop
def fig4():
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.axhspan(0, 3.0, color="#e8f7f1", zorder=0)
    ax.axhline(3.0, color=AQUA, lw=1.5)
    ax.text(18.2, 2.8, "limit 3 g", color=INK2, fontsize=10, va="top")
    ax.plot(21.5, 5.10, "o", color=INK2, ms=10)
    ax.text(21.5, 5.4, "hand-tuned MPC", fontsize=10, color=INK, ha="center")
    ax.plot(27.9, 7.08, "o", color=ORANGE, ms=10)
    ax.text(27.6, 7.4, "reduced model only", fontsize=10, color=INK,
            ha="right")
    learned = [(19.8, 4.37), (20.0, 4.54), (20.2, 4.59), (19.9, 4.21)]
    lx, ly = zip(*learned)
    ax.plot(lx, ly, "o", color=BLUE, ms=9)
    ax.text(20.55, 3.75, "MPC with the learned model (4 variants)",
            fontsize=10, color=INK, ha="left", va="center")
    ax.set_xlim(18, 29.5)
    ax.set_ylim(0, 8)
    ax.set_xlabel("speed along track (kn)")
    ax.set_ylabel("mean of largest 10 % impact peaks (g)")
    ax.grid(color=GRID)
    clean(ax)
    ax.set_title("Closed loop on the high-fidelity boat", fontsize=11,
                 color=INK, loc="left")
    save(fig, "fig04_closed_loop.png")


# 5 ------------------------------------------------------------ flow widening
def fig5():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.6), sharey=True)
    x = np.linspace(-6, 6, 400)

    def shape(x, s, shift=0.0, skew=0.0):
        a = np.exp(-0.5 * ((x - shift) / s) ** 2)
        b = 0.5 * np.exp(-0.5 * ((x - shift - 2.2 * s - skew) / (0.7 * s))
                         ** 2)
        return (a + b) / (a + b).max()

    cases = ((axes[0], "correct: widen around the mean", shape(x, 1.6), AQUA),
             (axes[1], "defect: scale the input noise",
              shape(x, 1.6, shift=-1.0, skew=2.0) * 0.8, RED))
    for ax, title, wide, col in cases:
        ax.fill_between(x, shape(x, 1.0), color=MUTED, alpha=0.25, lw=0)
        ax.plot(x, shape(x, 1.0), color=INK2, lw=1.5)
        ax.plot(x, wide, color=col, lw=2.2)
        ax.set_title(title, fontsize=10.5, color=INK)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlabel("predicted error")
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    axes[0].text(-5.8, 0.95, "grey: original", color=INK2, fontsize=9)
    axes[1].text(2.0, 0.78, "mean shifts," + NL + "tails distort", color=RED,
                 fontsize=9.5)
    fig.tight_layout()
    save(fig, "fig05_flow_widening.png")


# 6 ------------------------------------------------------------ inside the family?
def fig6():
    fig, ax = canvas()
    ax.text(3.6, 7.1, "catalogue family", ha="center", fontsize=11,
            color=BLUE, weight="bold")
    ax.add_patch(Ellipse((3.6, 4.0), 6.6, 5.6, fc="#eef4fc", ec=BLUE, lw=1.6))
    rng = np.random.default_rng(3)
    groups = (("small errors", (1.9, 5.4)), ("median", (3.6, 4.7)),
              ("large errors", (5.3, 5.4)),
              ('"corner": six effects' + NL + "all strong", (4.7, 2.7)))
    for lab, (cx, cy) in groups:
        p = rng.normal(0, 0.16, (5, 2)) + (cx, cy)
        ax.plot(p[:, 0], p[:, 1], "o", color=BLUE, ms=5, alpha=0.85)
        ax.text(cx, cy - 0.45, lab, ha="center", va="top", fontsize=9,
                color=INK2)
    ax.text(1.75, 2.75, "copies learn" + NL + "nothing: ≈ 1", ha="center",
            fontsize=10, color=INK)
    ax.plot(8.6, 5.3, "o", color=ORANGE, ms=12)
    ax.text(8.6, 6.0, "high-fidelity" + NL + "boat", ha="center",
            fontsize=10.5, color=INK, weight="bold")
    ax.text(8.6, 4.85, "copy: 0.35" + NL + "large learnable part",
            ha="center", va="top", fontsize=9.5, color=ORANGE)
    # 2 x 2 protocol grid
    gx, gy, cw, ch = 7.3, 0.2, 1.2, 0.7
    ax.text(gx + cw, gy + 2 * ch + 0.75, "same test on 16 family boats",
            ha="center", fontsize=8.5, color=INK2)
    for i, dl in enumerate(("training" + NL + "driving",
                            "target" + NL + "driving")):
        ax.text(gx + cw * (i + 0.5), gy + 2 * ch + 0.05, dl, ha="center",
                va="bottom", fontsize=7.5, color=INK2)
        for j in range(2):
            ax.add_patch(Rectangle((gx + cw * i, gy + ch * (1 - j)), cw, ch,
                                   fc="white", ec=MUTED))
            ax.text(gx + cw * (i + 0.5), gy + ch * (1.5 - j), "≈ 1",
                    ha="center", va="center", fontsize=10, color=INK)
    for j, sl in enumerate(("sea" + NL + "redrawn", "target" + NL + "sea")):
        ax.text(gx - 0.1, gy + ch * (1.5 - j), sl, ha="right", va="center",
                fontsize=7.5, color=INK2)
    save(fig, "fig06_inside_family.png")


# 7 ------------------------------------------------------------ error composition
def fig7():
    fig, ax = canvas(6.4, 5.2)
    ax.set_ylim(0, 8.2)
    ax.text(0.1, 7.9, "Side view", fontsize=10.5, color=INK, weight="bold")
    ax.text(5.2, 7.2, "surge: pressure drag, friction, wave-slope force, jet "
            "(~1/4 each)", ha="center", fontsize=8.5, color=INK2)
    ax.add_patch(Polygon([(1.2, 5.4), (8.6, 5.4), (9.4, 6.5), (1.2, 6.5)],
                         closed=True, fc=HULL, ec=INK2))
    xs = np.linspace(0.3, 9.8, 200)
    ax.plot(xs, 5.25 + 0.17 * np.sin(1.6 * xs), color=BLUE, lw=1.4)
    arrow(ax, (7.6, 4.35), (7.6, 5.35), color=ORANGE, lw=2.4)
    ax.text(7.6, 4.05, "pitch: planing impact lift (~70 %)", ha="center",
            va="top", fontsize=9, color=INK)
    arrow(ax, (1.15, 5.8), (0.2, 5.8), color=AQUA, lw=2.4)
    ax.text(0.15, 4.95, "jet (pump often" + NL + "under-primed)",
            fontsize=8.5, color=INK2, va="top")
    ax.text(0.1, 3.25, "Top view", fontsize=10.5, color=INK, weight="bold")
    ax.add_patch(Polygon([(1.2, 0.9), (7.6, 0.9), (9.4, 1.7), (7.6, 2.5),
                          (1.2, 2.5)], closed=True, fc=HULL, ec=INK2))
    for xx, h in zip(np.linspace(2.3, 8.3, 7), np.linspace(0.25, 0.5, 7)):
        arrow(ax, (xx, 2.55 + h), (xx, 2.55), color=VIOLET, lw=1.3)
    ax.text(9.9, 3.25, "yaw: moment distributed along the hull" + NL +
            "vs moment of the stern force", ha="right", va="center",
            fontsize=8.5, color=INK)
    arrow(ax, (1.45, 0.05), (1.45, 0.85), color=RED, lw=2.6)
    ax.text(1.8, 0.4, "sway: stern lateral force ∝ speed × stern "
            "immersion² × sideslip (~3/4)", fontsize=8.5, color=INK,
            va="center")
    save(fig, "fig07_error_composition.png")


# 8 ------------------------------------------------------------ gaps to the family
def fig8():
    fig, ax = canvas()
    ax.text(3.5, 7.0, "what the family can express", ha="center",
            fontsize=10.5, color=BLUE, weight="bold")
    ax.add_patch(Ellipse((3.5, 4.1), 6.2, 5.2, fc="#eef4fc", ec=BLUE, lw=1.6))
    ax.plot(8.8, 4.1, "o", color=ORANGE, ms=13)
    ax.text(8.8, 4.75, "high-fidelity" + NL + "boat", ha="center",
            fontsize=10.5, color=INK, weight="bold")
    gaps = [((5.7, 5.4), "same form, wrong place" + NL +
             "(lateral lift only at the bow)"),
            ((6.1, 4.1), "same form, values far outside" + NL +
             "the prior (heave–pitch" + NL + "coupling, pitch damping)"),
            ((5.7, 2.8), "different form" + NL + "(pump priming law," + NL +
             "pressure drag)")]
    for (x, y), lab in gaps:
        arrow(ax, (x, y), (8.55, 4.1), color=ORANGE, lw=1.4)
        ax.text(x - 0.15, y, lab, ha="right", va="center", fontsize=9,
                color=INK)
    ax.text(5.0, 0.55, "1,000 prior laws given the true states:" + NL +
            "best mixture sway 0.62  ·  frozen network 0.38  ·  fine-tuned "
            "0.07", ha="center", fontsize=9.5, color=INK2)
    save(fig, "fig08_family_gaps.png")


# 9 ------------------------------------------------------------ what MPC needs
def fig9():
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.8),
                             gridspec_kw=dict(height_ratios=[1, 1.35]))
    ax = axes[0]
    ax.set_xlim(-62, 8)
    ax.set_ylim(0, 1.1)
    ax.axis("off")
    ax.add_patch(Rectangle((-60, 0.35), 60, 0.35, fc=FILL, ec=MUTED))
    ax.add_patch(Rectangle((0, 0.35), 5.8, 0.35, fc="#e3eefb", ec=BLUE))
    ax.text(-30, 0.52, "past 60 s: peaks already measured (fixed)",
            ha="center", va="center", fontsize=9.5, color=INK2)
    ax.text(2.9, 0.78, "plan horizon" + NL + "5.8 s", ha="center",
            va="bottom", fontsize=9, color=BLUE)
    ax.text(-30, 0.1, "the 3 g limit averages both: once the past is over "
            "the limit, no plan is feasible", ha="center", fontsize=9,
            color=INK)
    ax = axes[1]
    rng = np.random.default_rng(1)
    prog = rng.uniform(0.2, 0.95, 18)
    cons = 0.25 + 0.8 * prog + rng.normal(0, 0.12, 18)
    ax.axhline(0.85, color=AQUA, lw=1.5)
    ax.text(0.19, 0.88, "limit", color=INK2, fontsize=9)
    ax.plot(prog, cons, "o", color=MUTED, ms=6)
    i = int(np.argmin(np.abs(cons - 0.8) + (prog < 0.6)))
    ax.plot(prog[i], cons[i], "o", color=BLUE, ms=9)
    ax.annotate("", xy=(prog[i], cons[i] + 0.12), xytext=(prog[i], cons[i]),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.5))
    ax.text(prog[i] + 0.02, cons[i] + 0.14, "small error → crosses the "
            "limit," + NL + "choice changes", fontsize=9, color=ORANGE)
    ax.set_xlabel("progress of a candidate plan")
    ax.set_ylabel("constraint quantity")
    ax.set_xticks([])
    ax.set_yticks([])
    clean(ax)
    fig.tight_layout()
    save(fig, "fig09_mpc_needs.png")


# 10 ----------------------------------------------------------- direction
def fig10():
    fig, ax = canvas()
    ax.text(0.1, 7.1, "One cross-section force law, summed along the hull",
            fontsize=10.5, color=INK, weight="bold")
    x0, x1, yb, yt = 0.8, 8.6, 4.9, 6.2
    ax.add_patch(Polygon([(x0, yb), (x1, yb), (9.5, yt), (x0, yt)],
                         closed=True, fc=HULL, ec=INK2))
    xs = np.linspace(x0 + 0.3, x1 - 0.1, 12)
    for k, x in enumerate(xs):
        ax.plot([x, x], [yb, yt], color="white", lw=1.5)
        h = 0.45 + 0.3 * np.sin(0.8 * k + 0.5) ** 2
        arrow(ax, (x, yb - 0.1 - h), (x, yb - 0.05), color=BLUE, lw=1.3)
    xw = np.linspace(0.3, 9.6, 100)
    ax.plot(xw, yb - 0.02 + 0.12 * np.sin(1.3 * xw), color=BLUE, lw=1)
    ax.text(5.0, 3.75, "local force = f(immersion, relative velocity, "
            "section shape, speed)", ha="center", fontsize=9, color=INK2)
    ax.text(0.1, 3.0, "Next steps", fontsize=10.5, color=INK, weight="bold")
    box(ax, 0.1, 1.15, 2.3, 1.45, "1  fix constraints;" + NL +
        "measure speed vs" + NL + "impact boundary", fc="#e8f7f1", ec=AQUA,
        fs=8)
    box(ax, 2.85, 1.15, 2.4, 1.45, "2  learn what the" + NL +
        "MPC uses; data in" + NL + "rounds; calibrate", fc="#e3eefb", ec=BLUE,
        fs=8.5)
    box(ax, 5.75, 1.95, 4.15, 0.8, "3  physics model as" + NL +
        "backbone and prior", fs=8.5)
    box(ax, 5.75, 0.95, 4.15, 0.8, "4  decide real-boat sensors", fs=8.5)
    arrow(ax, (2.4, 1.9), (2.85, 1.9))
    arrow(ax, (5.25, 2.1), (5.75, 2.35))
    arrow(ax, (5.25, 1.7), (5.75, 1.35))
    save(fig, "fig10_direction.png")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10):
        f()
    print("written to", OUT)
