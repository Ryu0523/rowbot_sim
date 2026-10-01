#!/usr/bin/env python3
"""
Bayesian linear regression for the corrections: y = phi . w + noise.

Prior N(m0, S0). The data are kept as sufficient statistics (X'X, X'y,
y'y, n), so an update costs the same after ten rows or ten thousand and
the state stays small enough to ship to worker processes. Every update
refits, re-estimating the noise variance from the residual sum of squares
of the current fit (floored): the target's impact data are far spikier than
the source's, and a fixed noise level would make the posterior either
overconfident or sluggish.
"""
import numpy as np


class BLR:
    def __init__(self, m0, S0, sigma2, sigma2_floor=None):
        self.m0 = np.asarray(m0, float)
        self.S0 = np.asarray(S0, float)
        self.P0 = np.linalg.inv(self.S0)
        self.P0m0 = self.P0 @ self.m0
        self.sigma2 = float(sigma2)
        self.floor = float(sigma2 if sigma2_floor is None else sigma2_floor)
        F = len(self.m0)
        self.XtX, self.Xty = np.zeros((F, F)), np.zeros(F)
        self.yty, self.n = 0.0, 0
        self.m, self.S = self.m0.copy(), self.S0.copy()

    def add(self, Phi, y):
        Phi = np.atleast_2d(np.asarray(Phi, float))
        y = np.atleast_1d(np.asarray(y, float))
        ok = np.all(np.isfinite(Phi), axis=1) & np.isfinite(y)
        if not ok.any():
            return
        Phi, y = Phi[ok], y[ok]
        self.XtX += Phi.T @ Phi
        self.Xty += Phi.T @ y
        self.yty += float(y @ y)
        self.n += len(y)
        self._refit()

    def _fit(self):
        S = np.linalg.inv(self.P0 + self.XtX / self.sigma2)
        S = 0.5 * (S + S.T)
        return S @ (self.P0m0 + self.Xty / self.sigma2), S

    def _refit(self, iters=2):
        m, S = self._fit()
        if self.n >= 10:
            for _ in range(iters):
                rss = self.yty - 2.0 * m @ self.Xty + m @ self.XtX @ m
                self.sigma2 = max(self.floor, float(rss) / self.n)
                m, S = self._fit()
        self.m, self.S = m, S

    def predict(self, Phi, w=None):
        """Mean (or the prediction of weights w) and the epistemic std."""
        Phi = np.asarray(Phi, float)
        mean = Phi @ (self.m if w is None else w)
        var = np.sum((Phi @ self.S) * Phi, axis=-1)
        return mean, np.sqrt(np.maximum(var, 0.0))

    def sample(self, rng):
        L = np.linalg.cholesky(self.S + 1e-12 * np.eye(len(self.m)))
        return self.m + L @ rng.normal(size=len(self.m))

    def slope_var_drop(self, Phi, c, sigma2=None):
        """For observations at the rows of Phi (..., F): how much each
        alone would shrink the variance of c . w, the quantity the decision
        depends on -- (c' S phi)^2 / (sigma^2 + phi' S phi)."""
        s2 = self.sigma2 if sigma2 is None else sigma2
        PS = Phi @ self.S
        num = (PS @ c) ** 2
        den = s2 + np.sum(PS * Phi, axis=-1)
        return num / den

    def copy_state(self):
        return dict(m0=self.m0, S0=self.S0, sigma2=self.sigma2,
                    floor=self.floor, XtX=self.XtX.copy(),
                    Xty=self.Xty.copy(), yty=self.yty, n=self.n)

    @classmethod
    def from_state(cls, st):
        b = cls(st["m0"], st["S0"], st["sigma2"], st["floor"])
        b.XtX, b.Xty = st["XtX"].copy(), st["Xty"].copy()
        b.yty, b.n = st["yty"], st["n"]
        if b.n:
            b._refit()
        return b
