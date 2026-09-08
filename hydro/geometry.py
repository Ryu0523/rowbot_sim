#!/usr/bin/env python3
"""
Hull geometry for the hydrodynamic pipeline.

Two bodies, two jobs:

  hemisphere  -- code verification. Analytic hydrostatics, and Hulme (1982)
                 published added mass / damping to check the BEM against.
  Wigley hull -- the physics. Parametric, so length / beam / draught become
                 sweep variables instead of a blocking decision, and it has
                 exact analytic hydrostatics of its own:

                     y(x,z) = (B/2)(1-(2x/L)^2)(1-(z/T)^2)
                     V      = (4/9) L B T          -> Cb = 4/9
                     A_wp   = (2/3) L B            -> Cwp = 2/3

Those two closed forms are the verification gate for the mesher: if the BEM
reports a displaced volume or heave stiffness that disagrees, the mesh is
wrong (bad closure, flipped normals) and nothing downstream can be trusted.
"""
import numpy as np
import capytaine as cpt

RHO = 1025.0
G = 9.81


# ------------------------------------------------------------------ analytic
def wigley_volume(L, B, T):
    """Displaced volume of a Wigley hull, exact."""
    return 4.0 / 9.0 * L * B * T


def wigley_waterplane_area(L, B):
    """Waterplane area of a Wigley hull, exact."""
    return 2.0 / 3.0 * L * B


def wigley_heave_stiffness(L, B, rho=RHO, g=G):
    return rho * g * wigley_waterplane_area(L, B)


def hemisphere_volume(R):
    return 2.0 / 3.0 * np.pi * R ** 3


def hemisphere_heave_stiffness(R, rho=RHO, g=G):
    return rho * g * np.pi * R ** 2


# ------------------------------------------------------------------- meshing
def _signed_volume(vertices, faces):
    """Divergence-theorem volume. Positive when face winding gives outward
    normals, which is the orientation Capytaine expects."""
    v = vertices[faces]                                  # (nf, 4, 3)
    tot = 0.0
    for a, b, c in ((0, 1, 2), (0, 2, 3)):               # split quad -> 2 tris
        p, q, r = v[:, a], v[:, b], v[:, c]
        tot += np.einsum("ij,ij->i", p, np.cross(q - p, r - p)).sum() / 6.0
    return tot


def _grid_to_quads(idx):
    """(nu, nv) index grid -> (nq, 4) quad connectivity."""
    a = idx[:-1, :-1].ravel()
    b = idx[1:, :-1].ravel()
    c = idx[1:, 1:].ravel()
    d = idx[:-1, 1:].ravel()
    return np.column_stack([a, b, c, d])


def wigley_mesh(L=10.0, B=2.5, T=0.8, nx=40, nz=14, name="wigley"):
    """Wetted surface of a Wigley hull as a Capytaine mesh.

    The parametrisation closes itself at bow, stern and keel (y -> 0 there),
    so only the waterline is open -- which is what a wetted-surface BEM wants.
    Both sides are built explicitly rather than mirrored, so the mesh carries
    no symmetry assumption that later oblique-wave runs would have to undo.
    """
    x = np.linspace(-L / 2, L / 2, nx + 1)
    z = np.linspace(-T, 0.0, nz + 1)
    X, Z = np.meshgrid(x, z, indexing="ij")
    Y = (B / 2) * (1 - (2 * X / L) ** 2) * (1 - (Z / T) ** 2)

    verts, faces = [], []
    for sign in (+1, -1):
        base = sum(v.shape[0] for v in verts)      # cumulative vertex offset
        pts = np.stack([X.ravel(), sign * Y.ravel(), Z.ravel()], axis=1)
        verts.append(pts)
        idx = base + np.arange(pts.shape[0]).reshape(X.shape)
        quads = _grid_to_quads(idx)
        if sign < 0:                       # keep winding consistent across sides
            quads = quads[:, ::-1]
        faces.append(quads)

    vertices = np.vstack(verts)
    faces = np.vstack(faces)
    # drop degenerate panels at bow/stern/keel where y collapses to zero
    areas = np.linalg.norm(np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]]), axis=1)
    faces = faces[areas > 1e-12 * L * T]

    if _signed_volume(vertices, faces) < 0:
        faces = faces[:, ::-1]

    return cpt.Mesh(vertices=vertices, faces=faces, name=name)


def wigley_body(L=10.0, B=2.5, T=0.8, nx=40, nz=14, dofs="all"):
    """Wigley hull as a FloatingBody with rigid-body dofs, immersed part only."""
    mesh = wigley_mesh(L, B, T, nx, nz)
    body = cpt.FloatingBody(mesh=mesh, center_of_mass=(0.0, 0.0, -T / 3))
    if dofs == "all":
        body.add_all_rigid_body_dofs()
    else:
        for d in dofs:
            if d in ("Surge", "Sway", "Heave"):
                body.add_translation_dof(name=d)
            else:
                body.add_rotation_dof(name=d)
    return body.immersed_part()


def hemisphere_body(R=1.0, resolution=(30, 30), dofs=("Heave",)):
    """Floating hemisphere: the classic BEM verification body."""
    mesh = cpt.mesh_sphere(radius=R, center=(0.0, 0.0, 0.0), resolution=resolution)
    body = cpt.FloatingBody(mesh=mesh, center_of_mass=(0.0, 0.0, 0.0))
    for d in dofs:
        body.add_translation_dof(name=d)
    return body.immersed_part()


if __name__ == "__main__":
    L, B, T = 10.0, 2.5, 0.8
    b = wigley_body(L, B, T)
    print(f"Wigley  L={L} B={B} T={T}")
    print(f"  panels            {b.mesh.nb_faces}")
    print(f"  volume  analytic  {wigley_volume(L,B,T):.4f} m^3")
    print(f"  Cb                {wigley_volume(L,B,T)/(L*B*T):.4f}  (exact 4/9)")
    print(f"  A_wp    analytic  {wigley_waterplane_area(L,B):.4f} m^2")
    print(f"  C33     analytic  {wigley_heave_stiffness(L,B):.1f} N/m")
