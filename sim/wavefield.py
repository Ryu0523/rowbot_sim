#!/usr/bin/env python3
"""
Irregular wave field for the nonlinear vessel model.

Extends the Phase 0 field with the two things the Cummins model needs:

  * depth-attenuated elevation. Dynamic wave pressure decays as e^{kz}
    (the Smith effect), and for a shallow-draught hull in short waves ignoring
    it over-predicts the excitation. Evaluated at the sectional centroid.
  * per-component access, so the linear diffraction transfer function from the
    BEM can be applied component by component and summed, rather than being
    forced through a single representative frequency.

Everything stays in the earth frame; forward speed enters simply because the
vessel queries the field at its own moving position, which is what produces the
encounter frequency without it ever appearing explicitly.
"""
import numpy as np

from step0_preview_spec import WaveField as _BaseWaveField

G = 9.81


class SeaState(_BaseWaveField):
    """JONSWAP + cos^2s spreading, queryable at depth."""

    def eta_at_depth(self, x, y, t, z):
        """Effective elevation for pressure at depth z (z <= 0).

        Each component is attenuated by its own e^{k z}: short waves die away
        far faster with depth than long ones, so a single average factor is
        not good enough across a broad spectrum.
        """
        x = np.atleast_1d(np.asarray(x, float))[..., None]
        y = np.atleast_1d(np.asarray(y, float))[..., None]
        z = np.atleast_1d(np.asarray(z, float))[..., None]
        phase = (self.k * (x * np.cos(self.th) + y * np.sin(self.th))
                 - self.w * t + self.phi)
        decay = np.exp(self.k * np.minimum(z, 0.0))
        return (self.a * decay * np.cos(phase)).sum(-1)

    def eta_dot(self, x, y, t):
        x = np.atleast_1d(np.asarray(x, float))[..., None]
        y = np.atleast_1d(np.asarray(y, float))[..., None]
        phase = (self.k * (x * np.cos(self.th) + y * np.sin(self.th))
                 - self.w * t + self.phi)
        return (self.a * self.w * np.sin(phase)).sum(-1)

    def phases(self, x, y, t):
        """Component phases at a single point -- used for diffraction."""
        return (self.k * (x * np.cos(self.th) + y * np.sin(self.th))
                - self.w * t + self.phi)

    @property
    def n_components(self):
        return self.a.size


_TF_CACHE = {}


def interp_transfer_cached(db, field, which="F_diff"):
    """Interpolation depends only on each component's (omega, heading), not on
    its random phase, so every seed of the same spectrum shares one result."""
    key = (id(db), which, float(field.hs), float(field.tp),
           float(field.theta0), field.n_components,
           float(field.w[0]), float(field.w[-1]))
    if key not in _TF_CACHE:
        _TF_CACHE[key] = interp_transfer(db, field, which)
    return _TF_CACHE[key]


def interp_transfer(db, field, which="F_diff"):
    """Interpolate a BEM force transfer function onto the wave components.

    Returns (n_comp, 6) complex: the force per unit wave amplitude for each
    component, at its own frequency and heading. Built once, then reused every
    timestep, which is what keeps a few hundred components affordable.
    """
    F = getattr(db, which)                     # (n_omega, n_dir, 6)
    out = np.empty((field.n_components, 6), complex)
    dirs = db.directions
    for c in range(6):
        for part, fn in (("re", np.real), ("im", np.imag)):
            grid = fn(F[:, :, c])
            # bilinear in (omega, direction); headings wrap at +-pi
            th = np.mod(field.th, 2 * np.pi)
            th = np.where(th > np.pi, 2 * np.pi - th, th)
            vals = _bilinear(db.omega, dirs, grid, field.w, th)
            if part == "re":
                out[:, c] = vals
            else:
                out[:, c] += 1j * vals
    return out


def _bilinear(xg, yg, grid, x, y):
    x = np.clip(x, xg[0], xg[-1])
    y = np.clip(y, yg[0], yg[-1])
    i = np.clip(np.searchsorted(xg, x) - 1, 0, len(xg) - 2)
    j = np.clip(np.searchsorted(yg, y) - 1, 0, len(yg) - 2)
    tx = (x - xg[i]) / (xg[i + 1] - xg[i])
    ty = (y - yg[j]) / np.maximum(yg[j + 1] - yg[j], 1e-12)
    return ((1 - tx) * (1 - ty) * grid[i, j] + tx * (1 - ty) * grid[i + 1, j]
            + (1 - tx) * ty * grid[i, j + 1] + tx * ty * grid[i + 1, j + 1])
