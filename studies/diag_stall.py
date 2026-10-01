"""
Shared measures of the nozzle diagnostics (diag_yaw_m8_blocks,
diag_m9_infamily; DEFECTS M10, PRIOR_DERIVATION.md D7), so the two splits
the reading compares (At and Cb) are measured the same way.

Units: plans U (n, H, 2) and actual positions are fractions of range; the
actual nozzle a[j] = XS[..., j, 13] / rud_max is the position BEFORE step j
(j = 0 .. H); e channel 9 (the nozzle gap) is (actual after step j -
command during step j) / dtc, raw units.

jump_events / stall_table: the rollout stall. An event at ANY step j0
(including 0) of a plan: jump = U[j0, 1] - a[j0], |jump| >= 0.15, the
actual nozzle at rest on the previous command (|a[j0] - U[j0 - 1, 1]| <
SETTLED; for j0 = 0 the command before the plan), so the jump is the
command step itself, and j0 + J <= H. Events are binned by |jump| (BINS).
The open fraction m steps after the jump is (U[j0 + m] - a[j0 + m + 1]) /
jump = -gap[j0 + m] dtc / jump: 1 = nothing closed yet, 0 = closed. "Stall
gone" in a bin: the rollout's mean open fraction at step 5 is within
STALL_MARGIN of the truth's.

block_kinds: what the nozzle does over a block's scored window, steps 0 ..
J - 1 (not only at the start): steady = the command constant and the
actual nozzle on it throughout.
"""
import numpy as np

J = 10                      # scored steps after a block start / a jump
BINS = ((0.15, 0.5), (0.5, 0.8), (0.8, np.inf))
SETTLED = 0.05              # actual on the previous command before a jump
STALL_MARGIN = 0.1          # |open fraction at step 5, rollout - truth|
STEADY_G0, STEADY_DU, STEADY_GAP = 0.05, 0.02, 0.05
STEADY_MIN = 30             # fewer steady blocks: the branch is undecided


def jump_events(U, a, u_prev, lo=BINS[0][0], J=J):
    """Nozzle jump events (module docstring). U (n, H, 2), a (n, H + 1),
    u_prev (n,) the nozzle command before step 0. Events in one plan do not
    overlap (after one at j0 the scan resumes at j0 + J). Returns a list of
    (row, j0, jump)."""
    n, H = U.shape[:2]
    prev = np.concatenate([np.asarray(u_prev, float)[:, None],
                           U[:, :-1, 1]], 1)            # command before j
    out = []
    for r in range(n):
        j0 = 0
        while j0 + J <= H:
            jump = U[r, j0, 1] - a[r, j0]
            if abs(jump) >= lo and abs(a[r, j0] - prev[r, j0]) < SETTLED:
                out.append((r, j0, float(jump)))
                j0 += J
            else:
                j0 += 1
    return out


def stall_table(events, T9, E9, dtc, label, J=J, show=print):
    """Per jump-size bin: count; over steps j0 .. j0 + J - 1 the gap (sign
    aligned with the jump) of the truth and the rollout mean, |truth| and
    |one sample|; the open fraction at steps 3, 5, 9 (truth vs rollout).
    T9 (n, H) true gaps, E9 (n, S, H) rollout samples, raw units. Returns
    {bin: dict(n, open5_truth, open5_rollout, gone)}."""
    res = {}
    show(f"{label}: nozzle jump events (any step, actual at rest before the "
         f"jump), {len(events)} in all")
    for lo, hi in BINS:
        ev = [e for e in events if lo <= abs(e[2]) < hi]
        name = f"{lo:.2f}-{hi:.2f}" if np.isfinite(hi) else f"> {lo:.2f}"
        if not ev:
            show(f"  jump {name}: no events")
            continue
        tr = np.array([np.sign(jm) * T9[r, j0:j0 + J] for r, j0, jm in ev])
        ro = np.array([np.sign(jm) * E9[r, :, j0:j0 + J].mean(0)
                       for r, j0, jm in ev])
        at = np.array([np.abs(T9[r, j0:j0 + J]) for r, j0, _ in ev])
        a1 = np.array([np.abs(E9[r, :, j0:j0 + J]).mean(0)
                       for r, j0, _ in ev])
        mag = np.array([abs(jm) for _, _, jm in ev])[:, None]
        of_t = -tr * dtc / mag                   # open fractions (sign aligned)
        of_r = -ro * dtc / mag
        o5t, o5r = float(of_t[:, 5].mean()), float(of_r[:, 5].mean())
        gone = abs(o5r - o5t) <= STALL_MARGIN
        res[name] = dict(n=len(ev), open5_truth=o5t, open5_rollout=o5r,
                         gone=gone)
        show(f"  jump {name} ({len(ev)} events): gap x jump sign, step j0 .. "
             f"j0+{J - 1}")
        show("     truth          " + " ".join(f"{x:+.2f}" for x in
                                               tr.mean(0)))
        show("     rollout mean   " + " ".join(f"{x:+.2f}" for x in
                                               ro.mean(0)))
        show("     |truth|        " + " ".join(f"{x:.2f}" for x in
                                               at.mean(0)))
        show("     |one sample|   " + " ".join(f"{x:.2f}" for x in
                                               a1.mean(0)))
        show("     open fraction at steps 3 / 5 / 9: truth " + " / ".join(
            f"{of_t[:, m].mean():.2f}" for m in (3, 5, 9)) + ", rollout "
            + " / ".join(f"{of_r[:, m].mean():.2f}" for m in (3, 5, 9))
            + f" -> stall {'gone' if gone else 'THERE'} (step 5 within "
            f"{STALL_MARGIN} of the truth: {o5r - o5t:+.2f})")
    return res


def block_kinds(U, a, J=J):
    """What the nozzle does over each block's scored window (steps 0 ..
    J - 1). U (n, H, 2), a (n, H + 1). Returns dict of (n,) arrays:
    g0 = |U[0, 1] - a[0]| (the jump at the start), dmax = max over j = 1 ..
    J - 1 of |U[j, 1] - U[j - 1, 1]| (the command moving inside), gmax =
    max over j < J of |a[j] - U[j, 1]| (the actual off the command), and
    the kinds: steady (g0 < 0.05, dmax < 0.02, gmax < 0.05), start_held
    (g0 > 0.15, dmax < 0.02), start_moving (g0 > 0.15, dmax >= 0.02),
    other (the rest: a moving command or a small start jump)."""
    g0 = np.abs(U[:, 0, 1] - a[:, 0])
    dmax = np.abs(np.diff(U[:, :J, 1], axis=1)).max(1)
    gmax = np.abs(a[:, :J] - U[:, :J, 1]).max(1)
    steady = (g0 < STEADY_G0) & (dmax < STEADY_DU) & (gmax < STEADY_GAP)
    start = g0 > 0.15
    held = dmax < STEADY_DU
    return dict(g0=g0, dmax=dmax, gmax=gmax, steady=steady,
                start_held=start & held, start_moving=start & ~held,
                other=~steady & ~start)


def yaw_skill(pred, truth, n_boot=200, seed=0):
    """Skill (MSE of pred / MSE of the truth; 1 = no better than 0) and
    correlation over blocks (B, J), with a 90% bootstrap interval of the
    skill over blocks (few blocks: one or two can dominate)."""
    se = ((pred - truth) ** 2).mean(1)
    s0 = (truth ** 2).mean(1)
    sk = float(se.mean() / max(s0.mean(), 1e-12))
    cc = float(np.corrcoef(pred.ravel(), truth.ravel())[0, 1]) \
        if len(pred) > 1 else float("nan")
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(se), len(se))
        bs.append(se[i].mean() / max(s0[i].mean(), 1e-12))
    lo, hi = np.quantile(bs, [0.05, 0.95]) if len(se) > 1 else (sk, sk)
    return sk, cc, float(lo), float(hi)
