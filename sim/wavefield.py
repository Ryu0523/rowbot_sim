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
    # The key used to carry only the COMPONENT COUNT, so a 16x4 spectrum and a
    # 32x2 spectrum -- both 64 components, same Hs, Tp and band -- collided, and
    # the second one silently received the first one's transfer functions.
    # Keyed on the actual component frequencies and headings now, which cannot
    # collide by construction. `id(db)` is also gone: ids are reused after
    # garbage collection, so it was a correctness hazard rather than an
    # identifier.
    # The database's own transfer function is in the key. It used to be only
    # the frequency GRID, and every hull built by Hull.database() at the same
    # length shares a grid: a second hull silently got the first one's forces.
    key = (which, float(field.hs), float(field.tp), float(field.theta0),
           hash(field.w.tobytes()), hash(field.th.tobytes()),
           hash(np.asarray(db.omega).tobytes()),
           hash(np.asarray(getattr(db, which)).tobytes()))
    if key not in _TF_CACHE:
        _TF_CACHE[key] = interp_transfer(db, field, which)
    return _TF_CACHE[key]


LATERAL = (1, 3, 5)             # sway, roll, yaw: odd under port/starboard reflection


def interp_transfer(db, field, which="F_diff", psi=0.0):
    """Interpolate a BEM force transfer function onto the wave components.

    Returns (n_comp, 6) complex: the force per unit wave amplitude for each
    component, at its own frequency and at its direction RELATIVE TO THE HULL,
    theta - psi. A fixed-heading snapshot; the vessel uses HeadingTransfer,
    which follows psi.
    """
    F = getattr(db, which)                     # (n_omega, n_dir, 6)
    out = np.empty((field.n_components, 6), complex)
    dirs = db.directions
    # headings wrap at +-pi. A database over 0..pi serves pi..2pi by
    # reflection, and under reflection sway, roll and yaw change sign: that
    # sign used to be missing, so waves from one side pushed as if from the
    # other.
    th = np.mod(field.th - psi, 2 * np.pi)
    mirror = th > np.pi
    th = np.where(mirror, 2 * np.pi - th, th)
    for c in range(6):
        for part, fn in (("re", np.real), ("im", np.imag)):
            vals = _bilinear(db.omega, dirs, fn(F[:, :, c]), field.w, th)
            if c in LATERAL:
                vals = np.where(mirror, -vals, vals)
            if part == "re":
                out[:, c] = vals
            else:
                out[:, c] += 1j * vals
    return out


class HeadingTransfer:
    """BEM force transfer functions for each wave component, at ANY heading.

    The BEM gives F(w, beta), beta the wave direction relative to the hull.
    The components' directions are fixed in the earth frame, so beta is
    theta - psi and changes every time the vessel turns. The first version
    interpolated once at theta, i.e. as if psi were zero forever: a vessel
    turned beam-on to the waves kept its head-sea forcing -- heave and pitch,
    no roll -- while its encounter frequency changed as if it were beam-on
    (DEFECTS F4). Built once: F interpolated in frequency onto each component
    on the database's heading grid. Per call: linear in heading only.
    """

    def __init__(self, db, field, which="F_exc"):
        F = np.asarray(getattr(db, which))                    # (n_w, n_dir, 6)
        d = np.asarray(db.directions, float)
        w = np.clip(np.asarray(field.w, float), db.omega[0], db.omega[-1])
        i = np.clip(np.searchsorted(db.omega, w) - 1, 0, len(db.omega) - 2)
        t = ((w - db.omega[i]) / (db.omega[i + 1] - db.omega[i]))[:, None, None]
        G = (1 - t) * F[i] + t * F[i + 1]                     # (n_comp, n_dir, 6)
        self.th = np.asarray(field.th, float)
        self.single = len(d) == 1
        self.full = (not self.single) and (d.max() - d.min() > np.pi + 1e-6)
        if self.single:
            print(f"  WARNING: the database has one wave direction "
                  f"({np.degrees(d[0]):.0f} deg); every heading gets its "
                  f"forces. Build it with a heading grid (Hull.database).")
        if self.full:                     # periodic grid for 0..2pi databases
            order = np.argsort(np.mod(d, 2 * np.pi))
            d = np.mod(d, 2 * np.pi)[order]
            G = G[:, order]
            d = np.r_[d, d[0] + 2 * np.pi]
            G = np.concatenate([G, G[:, :1]], axis=1)
        self.d, self.G = d, G
        self._n = np.arange(len(self.th))

    def at(self, psi):
        """(n_comp, 6) complex transfer with the vessel at heading psi."""
        if self.single:
            return self.G[:, 0]
        beta = np.mod(self.th - psi, 2 * np.pi)
        if self.full:
            mirror = np.zeros(beta.shape, bool)
        else:
            mirror = beta > np.pi
            beta = np.where(mirror, 2 * np.pi - beta, beta)
        d = self.d
        j = np.clip(np.searchsorted(d, beta) - 1, 0, len(d) - 2)
        s = np.clip((beta - d[j]) / np.maximum(d[j + 1] - d[j], 1e-12),
                    0.0, 1.0)[:, None]
        out = (1 - s) * self.G[self._n, j] + s * self.G[self._n, j + 1]
        if mirror.any():
            out[np.ix_(mirror, LATERAL)] *= -1.0
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
