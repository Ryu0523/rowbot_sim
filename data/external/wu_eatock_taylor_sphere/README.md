# Submerged sphere at forward speed: Wu & Eatock Taylor (1988) analytical results (secondary source)

Reference for the Track B check WE (`studies/rankine_unsteady_verify.py`, stage note
`learn/meta/RANKINE_STAGE1_2026-10-06.md` section 10.8.4).

- Original: G. X. Wu and R. Eatock Taylor (1988), "Radiation and diffraction of water waves by a submerged sphere at
  forward speed", Proc. R. Soc. Lond. A 417:433-461, doi 10.1098/rspa.1988.0069 -- not retrieved (publisher page behind
  a bot challenge; not bought).
- Secondary source used: T. Mei, M. Candries, E. Lataire, Z. Zou (2020), "Numerical Study on Hydrodynamics of Ships with
  Forward Speed Based on Nonlinear Steady Wave", J. Mar. Sci. Eng. 8(2):106, doi 10.3390/jmse8020106 (MDPI, CC BY 4.0),
  PDF from mdpi-res.com, local copy `jmse-08-00106.pdf` (kept out of git). The "Analytical results [27]" columns of
  tables 3 (surge), 5 (heave), 6 (couplings) on PDF pages 13-15 and tables 1-2 (exciting forces, page 11) are
  transcribed from the PDF text layer (numbers copied, not digitised); the excitation file also carries the tables'
  "Present (NK)" columns (Mei et al.'s own Neumann-Kelvin computation) as a numerical cross-reference.
- Case: sphere radius r, centre depth 2 r, deep water, Fr = U/sqrt(g r) = 0.4; radiation at k_e r = 0.6-5.0 with
  k_e = omega_e^2/g (critical 1/(16 Fr^2) = 0.3906); A_ij = a_ij/(pi rho r^3), B_ij = b_ij/(pi rho omega_e r^3);
  exciting forces f_j/(rho g pi A r^3 k_e) at incident kr = 0.4-2.0, head seas (critical kr 0.26808).
- Model of the reference (identified in the stage note 10.8.4): Neumann-Kelvin free-surface condition, m-terms of the
  steady flow past the sphere, m-term force.
