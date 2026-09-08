#!/usr/bin/env python3
"""
Capytaine driver: hull geometry -> frequency-domain hydrodynamic database.

Produces everything the time-domain Cummins model needs:
    A(omega)        added mass, 6x6
    B(omega)        radiation damping, 6x6
    F_exc(omega,b)  wave excitation (Froude-Krylov + diffraction), per direction
    C               hydrostatic stiffness
plus the displaced mass and centre of buoyancy.

Caveats carried forward deliberately:
  * BEM here is ZERO FORWARD SPEED. Encounter-frequency effects are applied
    downstream in the time domain; genuine forward-speed hydrodynamics would
    need strip theory or a forward-speed Green function.
  * Surface-piercing bodies suffer irregular frequencies -- spurious spikes in
    A and B at discrete frequencies from a resonance of the interior domain,
    not physics. Pass a lid mesh to suppress them; `find_irregular_spikes`
    flags them so they are never mistaken for real behaviour.
"""
import numpy as np
import xarray as xr
import capytaine as cpt

RHO = 1025.0
G = 9.81
DOFS6 = ["Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw"]


def compute_database(body, omegas, directions=(0.0,), rho=RHO, g=G,
                     depth=np.inf, n_jobs=1, progress=False):
    """Solve the radiation and diffraction problems over a frequency grid."""
    test_matrix = xr.Dataset({
        "omega": np.asarray(omegas, float),
        "wave_direction": np.asarray(directions, float),
        "radiating_dof": list(body.dofs.keys()),
        "water_depth": [depth],
        "rho": [rho],
        "g": [g],
    })
    ds = cpt.BEMSolver().fill_dataset(
        test_matrix, body, n_jobs=n_jobs,
        progress_bar=progress, _check_wavelength=False)
    ds.attrs["rho"] = rho
    ds.attrs["g"] = g
    return ds


class HydroDB:
    """Plain-array view of the hydrodynamic database.

    Stored as .npz rather than netCDF on purpose: Capytaine's dataset carries
    pandas Categorical dof coordinates and complex arrays, and without a
    netCDF4/h5netcdf backend xarray falls back to scipy's netCDF3 writer, which
    handles neither. The Cummins model wants plain arrays anyway.

    A, B          (n_omega, 6, 6)
    F_exc         (n_omega, n_dir, 6) complex
    C, M          (6, 6)
    """

    def __init__(self, omega, directions, dofs, A, B, F_exc, C, M, attrs=None,
                 F_fk=None, F_diff=None):
        self.omega = np.asarray(omega, float)
        self.directions = np.asarray(directions, float)
        self.dofs = [str(d) for d in dofs]
        self.A, self.B, self.F_exc = A, B, F_exc
        # kept separately because M4 replaces the Froude-Krylov part with a
        # nonlinear integral over the instantaneous wetted surface, and must
        # then add back diffraction ALONE -- using F_exc there would double
        # count the incident-wave pressure.
        self.F_fk = F_fk if F_fk is not None else F_exc
        self.F_diff = F_diff if F_diff is not None else np.zeros_like(F_exc)
        self.C, self.M = C, M
        self.attrs = attrs or {}
        self._i = {d: k for k, d in enumerate(self.dofs)}

    def a(self, dof_i, dof_j=None):
        j = self._i[dof_j or dof_i]
        return self.A[:, self._i[dof_i], j]

    def b(self, dof_i, dof_j=None):
        j = self._i[dof_j or dof_i]
        return self.B[:, self._i[dof_i], j]

    def exc(self, dof, direction_index=0):
        return self.F_exc[:, direction_index, self._i[dof]]

    @property
    def L(self):
        return float(self.attrs.get("L", 1.0))

    def __repr__(self):
        return (f"HydroDB({len(self.omega)} freqs "
                f"{self.omega[0]:.2f}..{self.omega[-1]:.2f} rad/s, "
                f"{len(self.directions)} directions, dofs={self.dofs})")


def to_db(ds):
    dofs = [str(d) for d in ds.radiating_dof.values]
    sel = dict(influenced_dof=dofs, radiating_dof=dofs)
    A = ds["added_mass"].sel(**sel).transpose("omega", "influenced_dof",
                                              "radiating_dof").values
    B = ds["radiation_damping"].sel(**sel).transpose("omega", "influenced_dof",
                                                     "radiating_dof").values
    def _f(name):
        return ds[name].sel(influenced_dof=dofs).transpose(
            "omega", "wave_direction", "influenced_dof").values
    F = _f("excitation_force")
    F_fk = _f("Froude_Krylov_force")
    F_diff = _f("diffraction_force")
    C = ds["hydrostatic_stiffness"].sel(**sel).transpose(
        "influenced_dof", "radiating_dof").values
    M = ds["inertia_matrix"].sel(**sel).transpose(
        "influenced_dof", "radiating_dof").values
    return HydroDB(ds.omega.values, ds.wave_direction.values, dofs,
                   A, B, F, C, M, dict(ds.attrs), F_fk=F_fk, F_diff=F_diff)


def save(ds, path):
    db = to_db(ds) if not isinstance(ds, HydroDB) else ds
    np.savez_compressed(
        path, omega=db.omega, directions=db.directions,
        dofs=np.array(db.dofs, dtype="U8"),
        A=db.A, B=db.B, F_re=db.F_exc.real, F_im=db.F_exc.imag,
        FK_re=db.F_fk.real, FK_im=db.F_fk.imag,
        FD_re=db.F_diff.real, FD_im=db.F_diff.imag,
        C=db.C, M=db.M,
        attr_keys=np.array([str(k) for k in db.attrs], dtype="U32"),
        attr_vals=np.array([str(v) for v in db.attrs.values()], dtype="U64"))
    return db


def load(path):
    z = np.load(path, allow_pickle=False)
    attrs = dict(zip([str(k) for k in z["attr_keys"]],
                     [str(v) for v in z["attr_vals"]]))
    has = "FK_re" in z.files
    return HydroDB(z["omega"], z["directions"], list(z["dofs"]),
                   z["A"], z["B"], z["F_re"] + 1j * z["F_im"],
                   z["C"], z["M"], attrs,
                   F_fk=(z["FK_re"] + 1j * z["FK_im"]) if has else None,
                   F_diff=(z["FD_re"] + 1j * z["FD_im"]) if has else None)


def find_irregular_spikes(ds, dof="Heave", rel_jump=0.15):
    """Flag frequencies where A or B jumps far more than its neighbours --
    the signature of an irregular frequency rather than physics."""
    out = []
    for var in ("added_mass", "radiation_damping"):
        y = ds[var].sel(influenced_dof=dof, radiating_dof=dof).values
        w = ds.omega.values
        if len(y) < 5:
            continue
        smooth = np.convolve(y, np.ones(5) / 5, mode="same")
        scale = max(np.ptp(y), 1e-30)
        bad = np.abs(y - smooth) / scale > rel_jump
        bad[:2] = bad[-2:] = False           # convolution edge artefacts
        out += [(var, float(wi)) for wi in w[bad]]
    return out


def summarise(ds, dof="Heave"):
    a = ds["added_mass"].sel(influenced_dof=dof, radiating_dof=dof)
    b = ds["radiation_damping"].sel(influenced_dof=dof, radiating_dof=dof)
    return dict(
        n_omega=ds.sizes["omega"],
        omega_range=(float(ds.omega.min()), float(ds.omega.max())),
        A_range=(float(a.min()), float(a.max())),
        B_range=(float(b.min()), float(b.max())),
        disp_mass=float(ds["disp_mass"].values),
    )
