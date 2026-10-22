"""
hydro.rankine -- Rankine-source boundary-integral solver for the any-speed plant.

Port of the useful computational parts of WaveBEM (Mola, Heltai & DeSimone,
SISSA mathLab, LGPL-2.1) as described in learn/meta/WAVEBEM_PORT_DESIGN.md.
Nothing here is copied from the WaveBEM sources (licence rule L of
learn/meta/ACCEPTANCE_2026-10-06.md): the algorithms are re-implemented from
the papers cited in each module, and several WaveBEM choices are replaced
(constant-strength flat panels with closed-form integrals instead of Q1
collocation, dense LU instead of the fast multipole method, image symmetry).

Modules (stage 1 of the port, design section 8.4 steps R1, R2 at U = 0, S1;
stage B3: shifted, krylov; B3 review: kelvin):
    panels      flat-panel geometry, symmetry images, mesh generators
    influence   exact source / dipole potential and source velocity of a flat
                polygon (Hess & Smith 1964; Newman 1986), torch, CPU or GPU
    bie         Green-identity and source-formulation operators with images,
                dense LU (torch, fp64 / complex128, CPU or GPU)
    radiation   zero-speed frequency-domain radiation / diffraction with a
                numerical beach (Kring 1994 form of the damping)
    steady      steady forward-speed linear free surface, stage-1 operator
                (Dawson-type upstream differences; kept for the record and
                for the shared helpers: grids, d'Alembert-corrected
                near-field resistance, double-body flow)
    shifted     steady linear free surface, production operator since B3
                (decision D1): shifted Numerov rows (neutral, radiation
                condition from the row count), box correction of the panel
                density, entry-zone blend and resolution guard; dense LU or
                FFT-accelerated GMRES on the GPU (to ~1e5 panels)
    krylov      restarted GMRES on torch tensors (residual floor 1e-10 for
                accepting a stalled solve, B3 review)
    kelvin      exact steady Kelvin Green function (deep water) and a
                body-only Neumann-Kelvin BIE with it: independent check S2K
                of the Rankine solution for submerged bodies (B3 review)
    farfield    Kochin function and far-field wave resistance of a submerged
                body; exact linear Kelvin wave elevation of point sources
                (independent checks, added after the 2026-10-06 review)
    havelock    reference integrals for submerged singularities (Havelock
                1928, 1931) by the Kochin-function formula derived in the module
    analytic    closed-form / published references (Lamb spheroid
                coefficients, Hulme 1982 hemisphere table)
    hull        (B4) hull forms (Wigley, Wigley with a shallow transom):
                wetted-surface panelling below the calm waterplane at any
                sinkage and trim, exact hydrostatics, transom face
    piercing    (B4) steady Neumann-Kelvin flow about a surface-piercing hull:
                free-surface lattice + body-fitted band with a hull-wall
                virtual point, waterline term from the linear momentum
                balance, wave-cut far field, free sinkage and trim, dry and
                wet (WaveBEM-style, registered empirical) transom branches
    thinship    (B4) Michell's thin-ship resistance (limit check H8)
    unsteady    (B5) forward-speed radiation and diffraction about the
                uniform stream as ONE linear time-invariant system per speed
                and attitude: free-surface grid with beaches on all sides,
                upwind-biased d/dx, exact frequency response (one complex LU
                per frequency), instantaneous / static / steady limits,
                time-domain RK4 of the same system, direct-pressure and
                Ogilvie-Tuck forces, Green-identity (Haskind, Timman-Newman)
                terms, pseudo-energy balance; B5 review: consistent hull-wall
                closure of d/dx with hull-trace inflow data (wall nodes)
    tables      (B5) radiation tables for the any-speed plant (Cummins form
                with the static stiffness C0 about the slice attitude, kernels
                with a zero-speed-resolution window, dispersion-aware validity
                band, storage, interpolation, plant-side force, free decay);
                B6: option linearisation="DB"
    doublebody  (B6) double-body (steady base flow) linearisation: rigid-lid
                base flow, second-order surface calculus, conservative
                kinematic operator (exact discrete Timman-Newman reciprocity),
                Tuck-exact Ogilvie-Tuck m-terms (adjoint of the edge-flux
                W.grad), the LTI system and coefficients, diffraction with the
                incident-wave/base-flow forcing, pseudo-energy time domain,
                steady wave problem and free sinkage/trim
Verification: studies/rankine_verify.py (stages 1-3), studies/rankine_hull_verify.py
(B4), studies/rankine_unsteady_verify.py (B5), studies/rankine_db_verify.py (B6); operator analysis (D1):
studies/rankine_d1.py; stage note and results:
learn/meta/RANKINE_STAGE1_2026-10-06.md (B3: section 8).
Conventions (all modules): x forward, y to port, z up, fluid in z < 0;
panel normals point INTO the fluid (out of the body); rho, g are parameters.
"""
