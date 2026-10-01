#!/usr/bin/env python3
"""
Feature maps for the two corrections (learn/adapt/cmpc.py).

Inputs, all from existing measurements:
    un   speed over the design speed, minus 1 (0 at 25 kn)
    psi  heading, rad -- the context; the wave direction is not known
    thr  thrust as a fraction of the limit (speed-loss model only)
    inten recent motion intensity: rms heave velocity over the last 5 s,
         m/s (typically 0.8-1.9 in sea state 3; centred on INTEN0 below)
         -- a stand-in for the wave height, which drives BOTH the
         speed lost and the impacts, and would otherwise make slow running
         look like the cause of heavy impacts in passive data

"hand": products of a speed polynomial and heading harmonics -- few
        parameters, learns fast, cannot express shapes nobody listed.
"rff":  random Fourier features of the same inputs (a Gaussian-process
        approximation) plus a linear part -- general smooth shapes, needs
        more data.
"""
import numpy as np

INTEN0, INTEN_S = 1.2, 0.4          # measured typical value and spread


class Features:
    def __init__(self, kind="hand", target="impact", intensity=False, D=60,
                 ls=0.7, seed=0):
        self.kind, self.target, self.intensity = kind, target, intensity
        if kind == "rff":
            rng = np.random.default_rng(seed)
            n_in = self._n_inputs()
            self.W = rng.normal(0.0, 1.0 / ls, (D, n_in))
            self.b = rng.uniform(0.0, 2 * np.pi, D)
            self.D = D
        self.dim = self(np.zeros(1), np.zeros(1), np.zeros(1),
                        np.zeros(1)).shape[1]

    def _n_inputs(self):
        return 3 + (self.target == "surge") + self.intensity

    def _raw(self, un, psi, thr, inten):
        cols = [un / 0.3, np.cos(psi), np.sin(psi)]
        if self.target == "surge":
            cols.append((thr - 0.5) / 0.3)
        if self.intensity:
            cols.append((inten - INTEN0) / INTEN_S)
        return np.stack(cols, axis=-1)

    def __call__(self, un, psi, thr=None, inten=None):
        un = np.atleast_1d(np.asarray(un, float))
        psi = np.broadcast_to(np.asarray(psi, float), un.shape)
        thr = np.broadcast_to(np.asarray(0.5 if thr is None else thr, float),
                              un.shape)
        inten = np.broadcast_to(np.asarray(INTEN0 if inten is None else inten,
                                           float), un.shape)
        if self.kind == "rff":
            z = self._raw(un, psi, thr, inten)
            return np.concatenate([np.ones(un.shape + (1,)), z,
                                   np.sqrt(2.0 / self.D)
                                   * np.cos(z @ self.W.T + self.b)], axis=-1)
        h = np.stack([np.ones_like(psi), np.cos(psi), np.sin(psi),
                      np.cos(2 * psi), np.sin(2 * psi)], axis=-1)
        if self.target == "surge":
            a = np.stack([np.ones_like(un), un, un ** 2, thr - 0.5], axis=-1)
            h = h[..., :3]
        else:
            a = np.stack([np.ones_like(un), un, un ** 2], axis=-1)
        f = (a[..., :, None] * h[..., None, :]).reshape(un.shape + (-1,))
        if self.intensity:
            di = (inten - INTEN0) / INTEN_S
            g = np.stack([di, di * un], axis=-1)
            f = np.concatenate([f, (g[..., :, None] * h[..., None, :3]
                                    ).reshape(un.shape + (-1,))], axis=-1)
        return f

    def d_speed(self, un, psi, thr=None, inten=None, eps=1e-3):
        """d(features)/d(un) -- the direction of 'how the prediction changes
        with speed', the slope the decision depends on."""
        return (self(un + eps, psi, thr, inten)
                - self(un - eps, psi, thr, inten)) / (2 * eps)
