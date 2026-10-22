"""
Restarted GMRES on torch tensors (fp64, CPU or GPU), for the FFT-accelerated
steady free-surface solver (shifted.py; stage B3, decision D1 of
learn/meta/WAVEBEM_PORT_DESIGN.md).

Standard algorithm (Saad & Schultz 1986, SIAM J. Sci. Stat. Comput. 7:856):
Arnoldi with classical Gram-Schmidt applied twice (CGS2; Giraud et al. 2005
show two passes give orthogonality to working precision), Givens rotations
for the small least-squares problem, right preconditioning (the residual that
is monitored is the true residual of the unpreconditioned system), restarts
after `restart` steps. The true residual b - A x is recomputed at every
restart, so the reported residual is not the recursively updated one.
"""
import time

import numpy as np
import torch


def gmres(matvec, b, x0=None, tol=1e-12, restart=200, maxiter=5000, precond=None,
          verbose=False, floor=None):
    """Solve A x = b. matvec(v) -> A v; precond(v) -> M^-1 v (right
    preconditioner, A M^-1 y = b, x = M^-1 y). Stops when
    |b - A x| <= tol |b| (true residual at a restart or at the end).

    Stall rule (changed after the B3 review): when two restarts in a row
    fail to halve the true residual, the stop is classified as round-off
    stagnation (info["stagnated"] = True) ONLY if the true relative residual
    is already <= floor (default 100 tol, i.e. 1e-10 for the production
    tol 1e-12; registered in the stage note, section 8.8; measured round-off
    floor of the FFT products 1e-12 ... 2.6e-12 above N ~ 5e4). Above the
    floor a stall is slow convergence: iteration continues up to maxiter and
    the solve returns converged = accepted = False. Callers must check
    info["accepted"] (SteadyShiftedFS.solve raises when it is False).
    Returns x and info dict(iters, restarts, relres, history, seconds,
    converged (relres <= tol), stagnated, accepted (relres <= floor), floor)."""
    t0 = time.time()
    dev, dt = b.device, b.dtype
    n = b.shape[0]
    bnorm = float(torch.linalg.norm(b))
    floor = 100.0 * tol if floor is None else float(floor)
    if bnorm == 0.0:
        return torch.zeros_like(b), dict(iters=0, restarts=0, relres=0.0, history=[], seconds=0.0,
                                         converged=True, stagnated=False, accepted=True, floor=floor)
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    P = precond if precond is not None else (lambda v: v)
    hist = []
    it = 0
    nrest = 0
    stagnated = False
    while True:
        r = b - matvec(x)
        beta = float(torch.linalg.norm(r))
        hist.append(beta / bnorm)
        if verbose:
            print(f"    gmres restart {nrest}: iters {it}, true relres {beta / bnorm:.3e}", flush=True)
        if beta <= tol * bnorm or it >= maxiter:
            break
        # stagnation at the round-off floor of the products (added in B3
        # after the first full run: at N ~ 1e5 the true residual stalls at
        # ~1.7e-12 while the Arnoldi estimate is below tol, so every restart
        # did one step and nothing changed). Two restarts in a row without
        # halving the true residual stop the solve only if the residual is
        # at or below the registered floor (B3 review); above it, keep going
        if len(hist) >= 3 and hist[-1] > 0.5 * hist[-3] and hist[-1] <= floor:
            stagnated = True
            break
        m = min(restart, maxiter - it)
        V = torch.zeros((m + 1, n), dtype=dt, device=dev)
        H = np.zeros((m + 1, m))
        cs = np.zeros(m)
        sn = np.zeros(m)
        g = np.zeros(m + 1)
        g[0] = beta
        V[0] = r / beta
        k_used = 0
        for k in range(m):
            w = matvec(P(V[k]))
            # CGS2
            h = V[:k + 1] @ w
            w = w - V[:k + 1].T @ h
            h2 = V[:k + 1] @ w
            w = w - V[:k + 1].T @ h2
            hn = float(torch.linalg.norm(w))
            H[:k + 1, k] = (h + h2).cpu().numpy()
            H[k + 1, k] = hn
            if hn > 0:
                V[k + 1] = w / hn
            # apply the previous rotations to the new column
            for i in range(k):
                t = cs[i] * H[i, k] + sn[i] * H[i + 1, k]
                H[i + 1, k] = -sn[i] * H[i, k] + cs[i] * H[i + 1, k]
                H[i, k] = t
            den = float(np.hypot(H[k, k], H[k + 1, k]))
            cs[k], sn[k] = H[k, k] / den, H[k + 1, k] / den
            H[k, k] = den
            H[k + 1, k] = 0.0
            g[k + 1] = -sn[k] * g[k]
            g[k] = cs[k] * g[k]
            it += 1
            k_used = k + 1
            if abs(g[k + 1]) <= 0.5 * tol * bnorm or hn == 0:
                break
        # solve the triangular system and update x
        from scipy.linalg import solve_triangular
        y = solve_triangular(H[:k_used, :k_used], g[:k_used], lower=False)
        upd = V[:k_used].T @ torch.as_tensor(y, dtype=dt, device=dev)
        del V
        x = x + P(upd)
        nrest += 1
    rel = hist[-1]
    return x, dict(iters=it, restarts=nrest, relres=rel, history=hist, seconds=time.time() - t0,
                   converged=rel <= tol, stagnated=stagnated, accepted=rel <= floor, floor=floor)
