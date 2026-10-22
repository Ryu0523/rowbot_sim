# Tveitnes, Fairlie-Clarke & Varyani (2008) — digitised data

Source: `tveitnes_2008_constant_velocity_entry.pdf`, author version of "An experimental investigation into the
constant velocity water entry of wedge-shaped sections", Ocean Engineering 35 (2008) 1463-1478 (16 pages in this
copy; the figures are vector charts). Untrusted external data; read only.

Produced by `studies/derive_v5_tveitnes.py` (check V5 of `learn/meta/HULL_FORCE_AUDIT_REVIEW_2026-10-09.md`).
The charts were NOT read from pixels: the plotted markers and curves were taken from the PDF drawing paths and
mapped to data coordinates through the axis gridlines and tick-label positions. Reading error is therefore set by
the authors' own plotting resolution and by my axis calibration, not by a screen.

## Test facts (paper sections 2-3)

- Sections: length L = 0.3 m, beam B = 0.6 m (half-beam c = 0.3 m), deadrise 5, 10, 15, 30, 45 deg; box above the
  wedge 0.6 m wide, 0.3 m high. Transparent end plates 1 m x 1 m travel with the model (2 mm clearance) to make the
  flow two-dimensional.
- Entry velocities (Table 2): 5 deg: 0.24, 0.48, 0.94 m/s; 10 and 15 deg: 0.48, 0.72, 0.94; 30 deg: 0.72, 0.94;
  45 deg: 1.19. Velocity held within about 1 % at the motor; the impact-surface velocity (from the authors' rig
  simulation) dipped 4 % (30-45 deg) to 35 % (5 deg) at initial entry.
- Force = sum of four load cells, averaged over four runs, MODEL WEIGHT REMOVED, HYDROSTATIC (buoyancy) FORCE
  INCLUDED (section 3.4 and 4.1). Repeatability 1-3 % of the peak force; after chine wetting the record carries a
  large coherent vibration ("dynamic noise") from the impact-surface flexing and the rig.
- z = keel depth below the undisturbed surface, d = c tan(beta) = chine height above the keel. Wagner's chine
  wetting is at z/d = 2/pi = 0.637; the authors' measured wetting factors put it at z/d = 1/WF = 0.59-0.77.

## Files

### digitised_fig4_entry_force.csv  (Fig. 4, pp. 1467-1468; 12 panels a-l)

Columns: panel, beta_deg, V_mps, series, z_over_d, force_N.
- series = experiment: centres of the "x" markers (each marker is four filled arm pieces in the PDF; the centre is
  found by a 4-way vote with a 0.29 pt inset, 103-337 markers per panel, all markers of every panel recovered).
- series = gravity_force: the authors' solid "Gravity force" curve = their hydrostatic force plus their fitted
  residual-gravity force (Table 3 of the paper).
- Reading accuracy. Axis calibration fitted to the gridlines (y) and label centres (x): worst residual 0.3 pt in y,
  0.014 in z/d in x. Marker-centre error about 0.15 pt. One point of the y axis is worth (panel: N per pt)
  a 21, b 5.4, c 1.5, d 9.2, e 5.6, f 3.1, g 6.0, h 3.6, i 3.1, j 4.4, k 3.8, l 4.4; so the force reading error is
  1-6 N (panel a: 6 N; others 1-3 N) and the z/d reading error 0.003-0.015. This is far below the physical
  scatter of the record (standard deviation about a local straight line, z/d >= 1.5-2: 16-163 N depending on
  the run; 54 N for 30 deg / 0.94 m/s, 33 N for 30 deg / 0.72 m/s).
- Panels: a 5 deg 0.94 m/s, b 5/0.48, c 5/0.24, d 10/0.94, e 10/0.72, f 10/0.48, g 15/0.94, h 15/0.72, i 15/0.48,
  j 30/0.94, k 30/0.72, l 45/1.19.

### digitised_fig13_cfm.csv  (Fig. 13, p. 1473)

The authors' flow-momentum drag coefficient Cfm per deadrise, F_fm = Cfm (rho/2) L (2 y_w) v^2 with y_w = B/2
after chine wetting, i.e. Cfm = (force per unit length)/(rho c V^2). One value per deadrise, chosen by the authors
to match the asymptotic hydrodynamic force in their rig simulation; they state that no velocity dependence could
be resolved. Series experiment (diamonds) and CFD_FairlieClarke_Tveitnes_2008 (crosses). Reading error 0.01.
Values: experiment 0.774 / 0.681 / 0.602 / 0.479 / 0.418 at 5 / 10 / 15 / 30 / 45 deg (abstract: 0.78 to 0.41);
CFD 2.85 / 2.15 / 1.79 / 1.21 / 0.70.

### digitised_fig16_cfb_tanb.csv  (Fig. 16, p. 1474)

The authors' ESTIMATED hydrodynamic entry force (rate of change of added-mass momentum plus flow momentum, with
hydrostatic and residual-gravity forces removed, de-noised through their rig simulation) as Cfb tan(beta) vs z/d,
Cfb = Force/((rho/2) L B v^2) = (force per unit length)/(rho c V^2). Five series (5, 10, 15, 30 deg at 0.94 m/s;
45 deg at 1.19 m/s), assigned in legend order, which is the drawing order in the PDF; the assignment was checked
against the peak heights (10 deg 3.12, 15 deg 2.93) and the asymptotes (30 deg 0.288 = Cfm 0.479 x tan 30, 45 deg
0.421 = Cfm 0.418). Points are the centres of the dash and dot pieces of each curve (35-155 per series). Reading
error 0.02 in Cfb tan(beta), 0.01 in z/d.

## Not digitised

Fig. 11 (wetting factors), Fig. 12 and 14 (chine-dry added mass), Fig. 15 (added mass vs z/d), Figs. 6-7
(oscillation and exit tests), Fig. 10 (simulation components). The text gives: wetting factor 1.6 (5 deg) to 1.3
(45 deg); wet-chine added mass 20-80 % above the chine-wetting value, saturating by z/d 1.5-2.5; residual-gravity
constants k1 (experiment) 0.25 / 0.2 / 0.15 / 0.1 / 0.1 and k2 (CFD) 2.0 / 3.0 / 3.3 / 3.5 / 2.0 at 5-45 deg; exit
drag coefficients 1.0-1.3.
