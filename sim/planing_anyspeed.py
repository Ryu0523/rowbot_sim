#!/usr/bin/env python3
"""
The planing plant made physically credible at any speed -- version 'as1' of
learn/meta/anyspeed_design_en.md -- built beside the current plant ('v0':
sim/planing_vessel.py and its large-angle twin sim/planing_vessel_gz.py),
never inside it. It is selected only by the hull key (sim/physics.py):

    hull = hydro.hulls.get("scarab195@as1")
    hull.plant(sea)                  # PlaningVesselASLin (linear roll)
    physics.gz_twin(plant)           # PlaningVesselAS    (GZ roll, capsize)

TWO CLASSES, AS v0 HAS TWO
v0's hull.plant() builds PlaningVessel (linear roll) and the evaluation runs
its twin PlaningVesselGZ (large-angle righting arm, capsize). as1 mirrors
that pair exactly: PlaningVesselASLin is PlaningVessel's drop-in and
PlaningVesselAS is PlaningVesselGZ's, so everything that builds through
hull.plant / config.plant_for (Mission's base plant, sim/env.Episode,
sim/rl_env, the reduced-model identification lofi.template ->
config.calm_plant) keeps v0's roll model on an as1 hull, and a v0-vs-as1
comparison changes the physics and nothing else. Both share the
constructor, the switches, the caches and the measurements (_ASMixin).

WHY A COPY OF THE STRIP THEORY
The v0 files are frozen: the meta pipelines hash them and every result so
far rests on them. Each design step changes terms deep inside the
derivative -- S1 the momentum flux of the sections, S2 the hydrostatic and
wave pressure and the transom, S3 the wave kinematics -- where no override of
a method can reach. So the derivative, and the two captive runs that
integrate it (running_attitude, calm_resistance), are copied here, and each
step's change goes in behind ONE named switch (SWITCHES). An off branch is
the v0 line itself, unchanged and in v0's order of floating-point
operations, so with every switch off the plant is v0 EXACTLY -- equal, not
close (studies/anyspeed_regress_as.py checks golden_v0.npz, closed-loop
missions and the reduced-model fingerprint with ==, and diffs the copied
methods against v0's source with an allow-list of the switch blocks).

The copy reads names of sim/planing_vessel (G, SP, _load, and the jet
classes JetPump, Waterjet, Nozzle in the constructor) by its own binding. A
study that patches one of them on the v0 module (studies/
principles_hull_swap.install_scaling sets PV.NU and PV.Waterjet) would
change v0 and not the copy, and all-off == v0 would fail silently, so every
as plant refuses to build or step while they differ (check_v0_globals; NU
and RHO_AIR stay guarded although the copy now reads its own constants,
because a patched PV.NU still breaks all-off == v0).

THE HULL AS DATA (P0, the modular split; ACCEPTANCE G4)
The constructor is copied too (_StripAS.__init__), for one reason: v0
computes the geometry from formulas of the hull's fields and builds the jet
with two hard-coded time constants. The copy takes

  the geometry  from hydro/hull_geometry.stations_of(hull): the parametric
                family (or an offsets table); for a plain PlaningHull the
                family member of its fields, array-equal to v0's lines;
  the plant's absolute constants  from hydro/vessel_spec.numerics_of(hull)
                (self.num): water viscosity, air density, the Reynolds-number
                floors, the captive run's step, iterations and tolerances,
                the top-speed tolerance, and the air-drag point (a hull
                property, hydro/vessel_spec.Closures.z_air) -- v0's literals
                for a plain hull, Froude-scaled from the reference hull for
                a parametric one that does not set them;
  the jet       from hydro/vessel_spec.jet_inputs(hull): the complete
                JetSpec (every key, so no literal fallback of the
                constructor ever fires), including the constants v0
                hard-codes (tau_prime_loss, t_rate, the astern-thrust ratio,
                the dead band) and the pump's loss coefficients.

With v0's values every substituted expression evaluates to the same float
in the same order, so all-off == v0 still holds exactly (the goldens), and
studies/anyspeed_regress_as.py's copy check maps each parameter back to
v0's literal and compares the source trees. The geometry also drives the
S1 strip-exact chine location (_taper, _substrip): the chine as a function
of x and the roots of kc d = yc per chine piece come from the geometry, so
any chine the module describes (transom run, offsets) is handled, and for
v0's chine the expressions are S1's own (golden_as_dDt_full unchanged).

CACHES (ACCEPTANCE N3, E3)
The steady-state and top-speed caches are keyed by the plant's physics
stamp (code hash, hull manifest, switches; sim/physics.stamp_for), taken
before construction, not by name and id(hull): a recycled id of a freed
hull with the same name cannot alias. The steady state is keyed by the
exact speed float(u): v0 rounds it to 1 mm/s, so two speeds less than
0.5 mm/s apart share whichever attitude was computed first -- results that
depend on call order and a step that does not scale with the boat (the P0
review found the Froude twin's only non-round-off difference there: Fn
1.767 on the 1.5x twin got the constructor's u_design attitude). The
captive run is deterministic, so the exact key costs nothing. The frozen GZ table cache
(sim/planing_vessel_gz._GZ) is keyed by id(hull) and cannot be changed, so
every hull an as plant is built for is kept alive (_PINNED): its id is
never reused in the process.

ACTUATORS PASSED IN ARE KEPT (principles section 1) -- THE ONE PLACE WHERE
ALL-OFF != v0
v0's with_sea and from_plant rebuild a plant from _init_kw and silently
drop a prop= or rudder= the plant was built with. An as plant keeps pristine
copies of them and hands fresh deep copies to every plant rebuilt from it
(with_sea, physics.gz_twin), and refuses a rebuild at another step size
(their delay lines are built for this one). This is plumbing, not physics,
and it is not behind a switch: an as plant with every switch off equals v0
exactly for every plant built without prop= / rudder=, and for one built
with them until it is rebuilt; a REBUILT plant keeps the given actuators
where v0 would have fallen back to the hull's jet. The v0 route itself is
the frozen helper, unchanged (physics.gz_twin of a v0 plant IS from_plant
and drops them, as v0 always did); studies/anyspeed_regress_as.py tier
select asserts both behaviours. No caller passes prop= or rudder= today.

WHY THIS MRO
    PlaningVesselAS(_ASMixin, PlaningVesselGZ, _StripAS)
    PlaningVesselASLin(_ASMixin, _StripAS),      _StripAS(PlaningVessel)
PlaningVesselGZ.deriv takes its parent's derivative through super() and
swaps the roll restoring for the large-angle righting arm; in this MRO that
super() is _StripAS.deriv, the copy. So the GZ curve, capsize, recording,
trajectory(), report(), prop_submergence and step() with its lateral
substeps are the frozen GZ and v0 code itself, reused rather than copied.
What v0 writes against its own class by name is redone in _ASMixin: the
captive copies in running_attitude and calm_resistance (v0 builds them as
PlaningVessel, which would run v0's derivative), PlaningVessel.with_sea
(hard-codes PlaningVessel), and the steady-state and top-speed caches,
which are keyed by the switches as well (two presets on one hull must not
share an equilibrium).

SWITCHES AND PRESETS
One switch per design step (learn/meta/ANYSPEED_PLAN_2026-10-05.md section
3), as booleans in `self.sw`. The preset 'v0' has all of them off; the
preset 'as1', which a 'NAME@as1' hull gets, has those implemented so far.
The constructor keyword `test_switches=` overrides the preset, for tests
only: physics.verify() refuses such a plant unless told it is a test, and a
v0 hull refuses the keyword. A switch whose step has not landed raises.

    S0  the version mechanism; nothing changed (all off == v0, exactly)
    S1  dDt_full: the complete D/Dt of the sectional added mass, chine
        wetting located inside each taper strip, entry only
        (_StripAS._flux_dDt). Measurements: last_slam_force keeps v0's
        definition, last_slam_force_full is the new flux. Verified by
        studies/anyspeed_s1_check.py (studies/_cache/anyspeed/s1/)
    S1b chine_exact_all: the same strip-exact chine location for every
        strip, the prismatic ones too (orchestrator decision 2 on S1;
        ACCEPTANCE M5): v0 decides a prismatic strip's chine wetting at its
        centre, so its flux dm_a/dd V^2 drops to zero in one step as the
        centre caps. Needs dDt_full. Verified by studies/anyspeed_p0_check.py
        part 'cap' (against an independent reference; the jump removed;
        station-count convergence) and by all of S1's standing checks
        re-run on the preset with it (studies/anyspeed_p0_s1rerun.py,
        studies/_cache/anyspeed/p0/s1_rerun/: a D55 1-9 kn, b decay 1-10
        kn, c sweeps, g census 1-40 kn, f basin and following seas at Hs
        x1.5, w bow out in head waves, d 14-40 kn at 48/96/192 stations and
        Fridsma P6, h 26-40 kn station count and time step); in the as1
        preset since P0
        THE DISCRETE MOMENTUM (M1/M2). With d linear over a strip, the flux
        of a located strip is the exact strip integral of dm_a/dt|_x - u
        dm_a/dx|_d (the rate of change of the STRIP-AVERAGED sectional
        momentum, V taken at the centre), while the inertia terms (A33,
        A35, A55 and m_a dV/dt) use m_a at the strip centre (v0's
        midpoint rule). The two discrete momenta differ by O(dx^2) where m_a
        is smooth along the strip and by O(dx) only in the strip where the
        chine meets the water (m_a has a kink there); the mismatch is
        measured by studies/anyspeed_p0_check.py part cap (strip-averaged
        vs centre m_a, by station count), and is accepted at P0 as part of
        the midpoint discretisation that S2 replaces
    P0b kpp_strip: the keel curvature of the m_a u^2 kpp term as the strip
        integral of kpp divided by the strip length (hydro/hull_geometry.
        Stations.kpp_strip): second order in the station count where the
        centre value is first order, exact at v0's jump of kpp at the start
        of the keel rise. No effect on a hull without keel rise (scarab195,
        the Fridsma models: kp = 0); an offsets hull has it whatever the
        switch (its geometry gives no centre value). Verified by
        studies/anyspeed_p0_check.py parts geom (convergence order) and
        hullb_s1 (planing_b's census, sweep and decay on the preset)
    S2a transom (P1a): the longitudinal hydrostatic / Froude-Krylov force
        and moment exactly, the transom face and its flow state, astern
        (below). Needs dDt_full. Verified by studies/anyspeed_p1a_check.py
    P1a normal_rate: the turning strip normal in the horizontal momentum
        force (below). Needs transom. Verified by the same script
    P1a verlet: the integrator (below). Verified by the same script
    All three are in the as1 preset since P1a; transom and normal_rate as a
    recorded exception to two of their registered gates (see PRESETS)

S2a THE LONGITUDINAL PRESSURE FORCE AND THE TRANSOM (design 2.2-2.3, 2.8)
The plant's hydrostatic + incident-wave pressure is p = a_b rho g (eta(x) -
z): hydrostatic about the LOCAL wave surface eta (the long-wave Froude-
Krylov field the strips already use: their vertical force is a_b rho g A(d),
d the keel depth below eta; a_b is v0's buoyancy factor, inherited -- one
field, one factor on all of it). The plant's x is horizontal, z vertical,
sections vertical at body x (small trim, as v0).
  Divergence theorem on the slice [x, x + dx] of the immersed hull, for the
field p e_x (p = 0 on the waterplane): the x force on the hull surface in
the slice -- bottom, sides, taper, everything -- is

    f_x dx = -int_slice dp/dx dV + dQ = (-rho g a_b A eta_x + dQ/dx) dx,
    Q(x) = int_{A(x)} p dA = a_b rho g q(d(x), yc(x)),  q = int (d - h) dA,

and summed over the hull (the stem closed: Q = 0 there) it TELESCOPES:

    X_hull = -a_b [rho g int A eta_x dx + rho g q_T],

with q_T the section integral at the transom edge, at the transom's own
keel depth d_T (the wave evaluated there). No derivative of the geometry is
taken. v0 had instead -a_b rho g sum A tau dx inside its tilted strip force
(Xp), equal to this in the continuum but missing the taper's side pressure
-rho g (dq/dyc) yc' (dq/dd = A, so d(q)/dx = A (eta_x - tau) + (dq/dyc)
yc'), and added the wave part a second time as Xw: deleted. The moment,
with z_v = zk + SP x Th + h the vertical offset of a hull point from the CG
(an earth coordinate: d z_v / dx = 0 at fixed height), telescopes the same
way for the field z_v p e_x:

    M_hull = sum z_v f_x = -a_b [rho g int eta_x (z0 A + m1) dx + R_T],
    z0 = zk + SP x Th,  m1 = int h dA,  R(x) = rho g (z0 q + s2),
    s2 = int h (d - h) dA                   (_vsec, _vsec_q: closed form)

The transom face (vertical, outward normal -e_x) carries +Q and +R over its
WETTED height h_w. With the face fully wet (h_w = d_T) the sums close:
X = -a_b rho g int A eta_x dx, the textbook closed-body Froude-Krylov force,
and M = -a_b rho g int eta_x (z0 A + m1) dx -- both exactly 0 in calm water
at ANY attitude (the 439 N sternward force at rest and the -190 N m couple
the physics review found for a face moment without the hull's are gone; at
the hydrostatic equilibrium the plant's forces vanish identically). The
dynamic part of the strip force (added mass, momentum flux, cross-flow
drag) stays tilted by the bottom slope with no z moment, as v0 (that is
not a pressure field this switch integrates).
  THE TRANSOM FLOW STATE: the dead-water level. Behind a wet transom the
water is a dead-water region whose free surface the flow pulls down: its
pressure is hydrostatic about that lowered level (design 2.3). The state of
the boundary is the face's wetted height h_w. As the surface depression is
a base-pressure head proportional to the dynamic pressure of the passing
flow -- a constant base-pressure coefficient C_pb, as for bluff bases at
high Reynolds number -- it is U^2 |C_pb| / (2 g), and the face clears when
it reaches the keel depth: at the transom-draught Froude number F_T =
U / sqrt(g d_T) = F_T_crit = sqrt(2 / |C_pb|). So, with F_T_crit the ONE
empirical input (the ventilation threshold, scatter 2.0-4.5: register E14;
hydro/vessel_spec.TRANSOM_CLOSURE),

    h_w = max(0, d_T - U^2 / (g F_T_crit^2))      ('progressive', primary)

hysteresis-free (a function of the state at the instant), continuous, wet
at rest (h_w = d_T), dry at planing. Before clearance it gives the transom
base drag a_b (Q(d_T) - Q(h_w)) (~1/2 rho U^2 A_T 2 a_b / F_T_crit^2 at low
speed) that the validation review found missing: a base-drag coefficient
C_D,base = 2 / F_T_crit^2 on the immersed transom area, a checkable
consequence of the closure (0.19 at the default, 0.10-0.50 over the
scatter; register E15). The form between rest and clearance is an
assumption (E15). The sources define F_T_crit on the STATIC transom
draught T_T0; 'progressive' applies it to the instantaneous d_T, which the
stern's squat makes larger near clearance (0.51 m against 0.39 m static at
10 kn on scarab195), so the clearance speed moves by sqrt(d_T / T_T0)
(register E14). Two alternative forms exist for the sensitivity study only
(hydro/vessel_spec refuses them without its sensitivity flag; G1 forbids a
speed switch outside one): 'progressive_static', the same progression on
the static draught, h_w = d_T max(0, 1 - U^2 / (g F_T_crit^2 T_T0)); and
'step', wet below F_T_crit on the static draught, dry above -- a
different variable, not a bound of the primary form. The state sets the
face's wetted boundary only: it scales no other force (no a_b(eps)).
  KNOWN INCONSISTENCY (P1a review): the flow state is a boundary condition
for the face's HYDROSTATIC pressure only. The dynamic trailing-edge
condition is untouched: the strip momentum keeps S1's transom flux
u m_aT V_T (design 2.5: the plane leaving the transom carries its added-
mass momentum into the wake, entry-only rule), the dry-transom form in
which the flow leaves cleanly at the edge, at every speed -- also where
this state says the face is 78-99 % wet (h_w / d_T 0.99 at 1 kn, 0.78 at
6 kn). Below clearance the plant is wet in its hydrostatics and dry in its
hydrodynamics. S1's provisional 1-9 kn pitch stabilisation rests on that
flux and stays PROVISIONAL; the dynamic wet-transom condition (e.g. the
momentum shed at the dead-water closure) is scheduled with S4 (radiation,
3D), and S1's standing checks (orchestrator decision 3) are re-run after
S4 -- re-running them after P1a tested nothing P1a changed in that term.
  ASTERN (u < 0) the transom LEADS: it is wet (no ventilation of a leading
face), carries its hydrostatic pressure as above plus the stagnation drag of
a blunt forebody 1/2 rho C_D,face A_T u^2 at its area centroid (Hoerner;
E16), and in the momentum flux the plane meeting the transom first sees m_a
jump from 0 to m_aT: an entry delta (_flux_astern). Every term is continuous
through u = 0. The lateral slender-body terms (lift at the transom, Munk
moment, |u|) are v0's: dissipative-signed astern, the leading/trailing edge
swap there is step S5's.
  Limits: the long-wave field (no depth decay e^{kz}, no variation across
the beam: step S3); the face pressure is hydrostatic about the dead-water
level (no dynamic pressure on a wet face going ahead); a_b inherited (E2,
known non-compliant until P3; v0's Xw carried it too). The pitch moment is
only half on one force system: this switch adds the z lever of the
hydrostatic x force (Mh: +340 N m at 10 kn, +249 at 13, +37 at 25 on
scarab195), while the tilted dynamic force and Xn keep v0's x-only lever
(their missing counterpart, the x force at keel height, ~+195, +519, +266
N m there; Mh alone moves the trim by <= 0.14 deg). The dynamic z lever
(normal force at the pressure centroid) is scheduled for P1b.

P1a normal_rate THE TURNING NORMAL (needs transom)
The sections' dynamic force acts along the strip normal n = (-tau, 1) (x
horizontal, z up; tau = kp + SP Th the bottom's slope). The momentum it
comes from is the vector m_a V n, and its complete material derivative is

    D/Dt(m_a V n) = n D/Dt(m_a V) + m_a V Dn/Dt,  Dn/Dt = (-Dtau/Dt, 0),
    Dtau/Dt = d tau/dt|_x - u d tau/dx = SP q - u kpp.

v0 keeps only the first part (Xp = -sum f tau dx). For a hull pitching at
rest that leaves a second-order mean surge force: f carries the added-mass
reaction -m_a (w' + SP x q'), tau the oscillating pitch, and the cycle
mean of their product is not zero (studies/anyspeed_p1a_check.py part
rest: +6 N at 0.43 deg pitch amplitude, a steady drift with no radiation
damping at rest). With the second part the horizontal force of the
momentum is -D/Dt(m_a V tau) (plus the cross-flow drag, tilted as v0): a
total derivative, zero on average over any periodic motion -- EXACTLY ONLY
WHERE THE FLUX IS NOT GATED. The plant counts the flux V Dm_a/Dt on entry
only (S1); where a pitching or heaving hull uncaps strips at rest, the
exiting momentum is dropped and the identity breaks (check 13's first,
registered form failed at 2.9e-2 for that reason, partly physics, partly
the entry-only rule; its passing form -- zero speed, every strip capped,
6.6e-7 -- was re-specified after that failure). The gated flux leaves a
residual mean surge force: from heave +5 cm at rest u reaches 2.55e-2 m/s
and drifts 1.15 m in 600 s (studies/_cache/anyspeed/p1a/rest.txt).
  Steady planing on a straight keel: q = 0, kpp = 0, nothing changes. On a
curved keel it adds Xn = sum m_a V u kpp dx, and with it the steady
horizontal momentum force telescopes to its end terms,
-sum dyn tau dx + Xn = -u m_aT V_T tau_T + u m_aS V_S tau_S (dyn without
the cross-flow drag; the stem term only where the keel is wet at the stem;
up to the entry-only gating, which drops the exits of a narrowing transom
run): the bow's curvature adds no drag of its own (d'Alembert for the
slender-body potential), as on a straight keel. On planing_b (keel rise
0.30 m over the forward 25 %) this is large: calm resistance at 2-8 kn
falls by 19-36 % against the plant without the switch, by 2 % at 10 kn,
not at all from 12 kn (registered M2 check and table: studies/
anyspeed_p1a_check.py part nr; studies/_cache/anyspeed/p1a/review_fix/
rereg/nr.txt: the ungated identity converges at order 1.8-3.1 and holds
to round-off on a straight keel wet at the stem, where the chine floor
adds the stem end term; without the switch its residual IS the curvature
term). It comes from bow strips where tau reaches 22-23 deg at 4-6 kn;
the strip normal n = (-tau, 1) is the small-angle form (tan tau ~ sin tau
~ tau, cos tau ~ 1), whose vertical component errs by 1 - cos tau = 3.4 %
at 15 deg and 7.3 % at 22 deg: above tau ~ 15 deg the theory is at its
edge (a large-angle normal is not in this switch; register T4). The
moment stays v0's (no z lever for the dynamic tilt; P1b).
  MISSING COUPLING (P1b): v0's vertical DV/Dt has no surge-acceleration term
u' tau. With this switch the surge equation carries +sum m_a tau w' dx
(through -f tau) while the heave equation has no sum m_a tau u' dx: the
surge-heave added-mass coupling is asymmetric, and the total-derivative
property holds only with surge held (check 13: 6e-3 with surge free). P1b
adds m_a u' tau to DV/Dt behind a switch and solves surge, heave and pitch
together.

P1a THE INTEGRATOR (switch verlet; ACCEPTANCE E2, M3)
v0 steps with Heun (two derivative evaluations). On an undamped oscillator
Heun's one-step factor is |R(i w h)|^2 = 1 + (w h)^4 / 4 > 1: spurious
energy growth at every frequency (the design review's 4.9 % per 100 s at
3.94 rad/s and 0.02 s). Heun's position update is already the velocity-
Verlet one, q + h v + h^2/2 a; its fault is evaluating the second force at
the Euler positions q + h v. With the switch the second stage is evaluated
at the step's end positions instead (_sv): for position-dependent forces
this IS velocity Verlet (Stormer-Verlet) -- symplectic, |R(i w h)| = 1 for
w h < 2, no drift, bounded energy error O((w h)^2). Velocity-dependent
forces get Heun's treatment (the Euler velocity), second order. What that
does and does not give (P1a review, measured on model problems):
  - no growth for position-dependent forces (verified: the rest test with
    dissipation off, studies/anyspeed_p1a_check.py part energy);
  - velocity-coupled CONSERVATIVE forces (gyroscopic: B35/B53 at speed, the
    yaw Coriolis terms) are not treated symplectically and grow, less than
    Heun (w h 0.1, g h 0.05: +4.2 % per 100 s against Heun's +18 %);
  - with damping the stability region is not Heun's near w h -> 2 (w h 1.0,
    c h 2.5: spectral radius 1.64 against Heun's 1.0); heave and pitch have
    no substep guard, while v0's lateral substep rule (lateral rate x step
    <= 1.8) is kept because the stiff lateral terms it guards (cross-flow
    drag, slender-body lift and moment) are velocity-only, where the
    scheme IS Heun's;
  - on the plant's own linearised heave/pitch at 0-40 kn both schemes have
    |mu| < 1, and for the oscillatory modes (the wave band) verlet's modal
    decay rate is within 0.0067 1/s of the exact one; an overdamped real
    mode decays up to 0.25 1/s slower than exact (14 1/s at 40 kn) --
    still decaying (standing E2 check: studies/anyspeed_p1a_check.py part
    e2speed).
Same cost, same measurements (from the derivative at the start of a step).

P1b (switches fk_exact, orbital, wagner_entry, lateral_phi0; none in the
as1 preset; learn/meta/ANYSPEED_PLAN_2026-10-05.md 'P1b' and 'P1b 复查后的
修正'; verification studies/anyspeed_p1b_check.py, studies/
anyspeed_p1b_fix.py). In calm water fk_exact and orbital have nothing to
add: every block they open is 'if self.sw.X and kb.wv:' (waves present),
and the calm derivative is P1a's, bit for bit (F6), so their calm-water
standing checks are P1a's.

P1b fk_exact. The exact linear incident pressure rho g a e^{k z'} cos(psi +
lam y) (z' below the local surface) on every section and on the transom
section and wetted face, beyond the plant's long-wave field (whose a_b
factor it shares on the vertical and longitudinal parts, E2): section
integrals in closed form (hydro/section2d.section_fk_grid's expressions,
evaluated row by row in sim/as_kernels.fk_rows since P1c; _fk_b's
docstring has the force formulas). ALONG THE HULL (P1b review), each
section quantity Q(x) is taken as yc(x) H(x) with H linear over the strip
(its station value, its slope by differences between wet neighbours) and
integrated against the exact phase: R_m = (1/dx) int yc (H + H' xi) xi^m
e^{i kappa xi} dx, from the strips' chine moments mu_m = int yc xi^m dx
(_fk_moments; the stem strip in t, x = x_stem - t^2, where the family's
chine goes like sqrt(x_stem - x)) and the Taylor series of the exponential
(FK_TERMS terms). Second order in the station count (the previous
constant-section strip was ~kappa^2 h^2 / 3 at the shortest waves and order
~1.5 from the blunt bow). The incident pressure is integrated over the
UPRIGHT section (register A5).

P1b orbital. Linear orbital velocities and accelerations at each section's
immersed centroid (register A2) in every relative velocity; the added
mass's rate on the geometric immersion rate Vg (the contact line follows
the surface). Without lateral_phi0, v0's lateral branch gets the lateral
diffraction (G. I. Taylor): the section impulse a22 (v_l - v_orb) + a24 p
under -D/Dt gives + a22 a_oy, its yaw lever x and the roll moment + a24
a_oy about the CG, with the gravity-free 2D values of the upright section
(hydro/section2d; the basis of the vertical and of lateral_phi0). Its
first form, on v0's plate m22 = pi/2 rho d^2 (the branch's own basis),
failed check 5 (sway 19-35 % high in long waves: the plates sum to 804 kg
against the 3D zero-frequency 467 kg); the branch's slender-body lift and
Munk moment keep v0's m22.

P1b WAGNER_ENTRY. Every earth-fixed plane has a flow state: E, in an entry
from dry (Wagner's pile-up: Zarnick's m_a, S1's entry-only flux); X, in the
exit after it (S1's rule: Zarnick's m_a, no flux -- the momentum goes with
the wake; design 2.5); F, oscillating about a fixed waterline (the exact
gravity-free 2D m_0, hydro/section2d.Table.vertical, with the Lagrangian
force m_0 V' + 1/2 V^2 dm_0/dd entering or leaving, and the horizontal
force -D/Dt(m_0 V tau), normal_rate's form). Transitions by the state and
the geometric immersion rate Vg only: dry -> E; E with Vg <= 0 -> X; X
entering again while still wet -> F; F stays F while wet. A boat at rest (u
= 0: no travel) starts in F; with speed, and in the captive (steady) runs,
the field is the steady one along the planes' paths from the leading edge
(_es_scan). In steady flow a plane only enters, or enters then exits (the
stem is a knife edge; scarab195's keel is straight, planing_b's keel rise
makes tau monotone along the hull), so no steady state has an F plane and
every steady state (running attitude, calm resistance, the census) is P1a's
bit for bit. The field is piecewise constant along the hull (sorted state
boundaries in body x, module helpers _es_*): each step moves the boundaries
exactly with the relative inflow, b - u_r(b) dt -- no interpolation of the
states, no numerical diffusion at any Courant number (the P1b review's
blocker: a linearly interpolated d_s classified by d_s > 0 spread the fixed
state along the hull) --, takes the planes arriving over the leading edge
as dry, then applies the transitions by each plane's class now (d and Vg
linear between the stations: the waterline crossing and the turning points
located inside the strips, as S1 locates the chine). A
step's later derivatives use the field advanced by t - t0. A strip's force
is the sum of its states' fractions. The field is the plant's hidden state:
hidden_state() / set_hidden_state(); a step continues it only from exactly
the state the last step returned (else it starts from the steady field and
counts es_resets); trajectory()['hidden'] records it for replay.
P1c (track W; plan '### P1c（W 线）'): the boundaries move by Heun's rule
with the inflow at both ends of the step (the end-of-step inflow alone was
W4's first-order error); the start inflow is kept with the field (hidden
state 'ur'). The steady-state copies (running_attitude, captive_deriv,
calm_resistance) use the steady field of their attitude (Vg without the
heave and pitch rates): the captive runs are P1a's bit for bit.
P1c review (round 2): the field is advanced in ES_NSUB = 4 sub-steps on
kinematics interpolated in time (d cubic Hermite with its rate kb.ddot, Vg
and ur quadratic through the field's previous sample), from the field's
own time (_es_t) to the derivative's; the hidden state carries the field's
time and the kinematics at it and at the time before ('t', 'kin',
'kinm'); a plant of its own, captive or free, continues its field between
steps (only the never-stepped steady-state copies use the attitude's
steady field).
P1c round 3 (plan '### wagner 低速', register T9): scarab195's 1-1.5 kn
oscillations from + 5 cm were nearly undamped with this switch. The exact
power identity of a captive run in the Jacobi form -- H = KE + U + sum 1/2
m V^2 dx - u sum m V tau dx; dH/dt = u m_T V_T g_T - u sum_b [e]_b + sum (f
- f_att) g dx + cross-flow + moments, f_att the attached (Lagrangian) force
of an earth-fixed plane whose section changes as the hull passes, g the
hull's upward speed -- put the input of round 2's code in the exiting
planes of the widening taper: an X plane has no flux, yet its Zarnick
added mass grew with the chine half-beam, so its fluid gained momentum V
Dm/Dt (and energy 1/2 c V^2) that no force supplied -- the wake can take
momentum away, not give it. The 'surge transfer' of round 2's grouping is
not an input (u dG/dt and the widening walls' work).
Now an X plane's added mass never grows along its path: k_a pi/2 rho
min(kc d, yc(x), Y)^2, Y the chine half-beam where it turned (equal to
Zarnick's there; kc d does not grow while X), kept per segment of the field
(y, hidden state 'y'; _es_xmass). Exit stays without suction (S1's
behaviour in calm steady states -- the census, the equilibria -- which on
scarab195 is so by construction: its keel has no curvature, so an X
plane's mass never enters a static force; the attached alternative, V
Dm/Dt for either sign, makes an unstable sunk bow-down root at 25-40 kn).
This is a MODELLING ASSUMPTION (register A8, G2/G3), S1's separated exit,
not a consequence of momentum conservation (which asks only that a plane
without flux has Dm/Dt <= 0: the attached form is as consistent), and it
changes the dynamics (pitch added mass in transients and head seas).
P1c round 4 (plan '### wagner 低速：复查后的修正'): the identity does NOT make
the plant dissipative. The state-boundary term -u sum_b [e]_b is the
explicit time derivative of the hull-frame Lagrangian (the states form a
material pattern that moves with the water) and has either sign: negative
where the heavier state is forward of a boundary, positive where it is
aft -- at X|X boundaries with the memory (1/2 c_turn k V^2 per unit
length, k = xt' / (xt' + u), positive while the turning point moves
forward: the registered residual of the assumption, bounded in the plan)
and at E|F boundaries (re-wetted entering planes behind fixed-waterline
ones). In a free run the pattern also makes the hull-frame kinetic energy
depend on the hull's horizontal position, a Lagrangian surge force sum_b
[e]_b that the momentum-form horizontal force here does not carry
(derivation; register T9 (a), open): the boundaries' energy is not paid
by the surge.

P1b lateral_phi0. Sway, roll and yaw added masses and couplings on the phi
= 0 basis at the instantaneous immersion and heel, the sections' lateral
and roll impulses under -D/Dt (shedding at the trailing edge only: the
transom going ahead, none astern where the trailing edge is the stem; the
Munk moment with the sign of u_r, so with Kirchhoff's surge partner r sum P
dx it does no work either way), the Lagrangian reactions of the kinetic
energy. INCOMPLETE (register T8): a23, a34, the body-axis Kirchhoff terms of
the rotating cross plane and the keel depth at heel are missing -- to be
done as one unit with S5; and its Fn -> 0 limit is not the rigid-lid one
(register A4, S4). Not adoptable before both.

P1c (learn/meta/ANYSPEED_PLAN_2026-10-05.md, P1c). fk_exact is in the as1
preset (ACCEPTANCE E1's revision of 2026-10-06: every physics check passed,
the step-time budget is its recorded shortfall). Its pressure integrals run
in sim/as_kernels.fk_rows, written row-stable (each section's numbers
independent of the other sections and of other boats in the call): the
same physics as P1b's _fk_b, the floating-point association changed at the
round-off level (the goldens of the sets with fk_exact re-pinned), so
that the batch mode sim/planing_batch.BatchAS -- N boats, each in its own
sea, in one vectorised call -- gives the plants' bits exactly. Switch
fast_path (implementation only) drops repeated work from the step: the GZ
plant's roll swap on the derivative's own section area and wave-slope mean,
the intake submergence from the surface elevation alone; proven equal, bit
for bit, to the path without it on the golden runs.

S5a (learn/meta/ANYSPEED_PLAN_2026-10-05.md, '### S5a'; verification
studies/anyspeed_s5a_check.py; switches heel_hydro and lateral_cross, not in
the as1 preset).

S5a heel_hydro. Every station's polygon -- keel, chines, deck edges: the
frozen GZ table's (sim/planing_vessel_gz._sections) -- heeled by phi about
the body x axis at the instantaneous heave and trim, clipped exactly below
the local surface: the station's elevation e_j, tilted across the beam by
its transverse wave slope s_j (the long-wave field's first-order Taylor
form). The pressure p = rho g (e_j + s_j y - z) integrated over it
(_clip_moments: Green's theorem over the immersed part of each edge, in the
frame aligned with the surface, where the waterline chord adds nothing)
gives, per unit length, the vertical force rho g A, the lateral force
-rho g s A and the roll moment rho g (S_y + s (S_z + SP x Th A)) about the
plant's roll axis (levers from the CG's height, as fk_exact's); pitch and
yaw by the x lever; the longitudinal force, its pitch moment and the yaw
moment of the x-facing surfaces by S2a's telescoped form over the heeled
transom section and face (_x_hydro_heel). a_b: P1c's convention (S5a
review, issue D) -- the vertical force, the longitudinal force and the
pitch moment carry a_b, the lateral force and the roll and yaw moments carry
neither a_b nor the transom correction ctr (fk_exact's never did; extending
the speed-weighted a_b(C_V), register E2, to them would be a new G1
violation). The parts of one pressure field are therefore weighted
differently (recorded in E2). The previous round put a_b on all of it --
which halved hh's roll stiffness and its lateral and roll wave excitation
from 15 kn up (a_b = 0.5 there); it survives only as a diagnostic variant
(_s5a_v). fk_exact's increment is still integrated over the upright section
(register A5), less its long-wave limit, which the tilted polygon now
carries (lateral -rho g s A; roll rho g s (z0 A + m1 + 2 c^3 / 3) about the
CG's height -- the previous round took the polygons' moment about the
stations' axis points instead, a first-order slope x trim term, issue O;
and the transom's yaw rho g s_T (I_yy(d_T) - I_yy(h_w)), which the previous
round counted twice, _hh_lateral): at zero heel hh's first-order wave forces
are P1c's exactly (check HB1 of plan 'S5a：复查后的修正').
Two-way roll: the keel depth of every station is d = H - zk cos phi (rolling
about the CG lifts the keel; below the horizontal through the centreline
surface point, so that its rate at fixed x is exactly the immersion rate
plus zk sin phi p), and the vertical dynamics, fk_exact's upright section
and the lateral tables take it. Both plants take the polygons' roll moment
(the linear-roll plant too: the classes then differ in capsize handling and
recording only). On the upright branch -- phi == 0 and no transverse slope
at any station (calm water, long-crested waves along the heading) -- the
copy keeps P1c's lines (the same integral in closed form), so those states
are P1c's bit for bit (check H3). Not modelled (S5b): the vertical dynamics
of a heeled section (asymmetric Wagner per side), the chine-separation
dryness of the sides at speed.

S5a lateral_cross. The 2D phi = 0 cross-plane impulse dynamics in the
plant's level, yaw-rotating frame (v, w are its velocities): per station
the table's body-frame matrix about the keel (a22, a23, a33, a24, a34, a44)
shifted along body z to the CG and rotated to the level frame, the
heave-heave entry the vertical model's own m33 (Zarnick / wagner_entry's
flow-state m_a, times ctr; a mixed basis, an assumption: register S5-2),
with ONE kinetic energy T = 1/2 m33 W_r^2 + 1/2 nu^T M~ nu from which every
Lagrangian reaction comes -- the roll one includes 1/2 W_r^2 dm33/dd zk
sin phi (S5a review, issue A: the previous round dropped the heave-heave
entry's roll reaction with the entry itself; the table's e33 there would
put the phi = 0 fixed-waterline theory's heel dependence, the opposite of
the asymmetric entry's restoring, into the planing regime); F = -D/Dt(M nu)
+ dT/dq per earth-fixed plane with dT/dphi at a fixed CG (the tensor's
rotation, the keel depth's change, the table's heel dependence); the
trailing edge sheds (transom ahead, stem astern: none), yaw and pitch Munk
sums, and in the yaw-rotating frame Kirchhoff's partners of the horizontal
impulse (r sum P_y dx in surge; -r P_x in sway and + v P_x in yaw for P_x
= A11 u + sum tau (m33 V - P_z) dx); the wave velocities' rates at fixed
body x (_wave_rates_b: the convective part is in conservation form; S5a
review, issue C); sway, heave, roll, pitch and yaw solved together
(_solve5).
  WHAT IS AND IS NOT VERIFIED. L1 / L1'' check the 2D formula F = -d/dt(M
nu) + dT/dq (one consistent phi = 0 T) against the pressure integral p =
-(phi_t + |grad phi|^2 / 2) on moving, heeled sections whose waterline is
on the walls -- the fixed-waterline (F) regime, the formula's structure; the
plant's T has the vertical model's m33 in its heave-heave entry instead
(the mixed basis), which no 2D reference checks. L1'+ checks the plant's
strip forces against that Lagrangian built independently from the table
and m33 (finite differences). In the planing regime (planes entering from
dry, the waterline on the bottom) the form is an assumption (register
S5-7): not derived (the momentum form of register A1 would drop the
reactions; S5b's asymmetric 2D+t replaces both), and no reference converges
there (the contact-point pressure ~ r^(2 lam - 2), lam = pi / (2 alpha) ->
1/2 as the bottom flattens); so the planing-regime sway and roll of this
switch are UNVERIFIED, and no data set (V1: Brown & Klosinski 1994) has
been compared. Its Fn -> 0 limit is still the phi = 0 one (register A4).
"""
import copy
import math
import warnings
from types import SimpleNamespace

import numpy as np

from . import physics
from . import planing_vessel as _pv
from .planing_vessel import (PlaningVessel, _load, G, RHO_AIR, NU, SP,
                             JetPump, Waterjet, Nozzle)
from .planing_vessel_gz import PlaningVesselGZ
from hydro import hull_geometry as _hg
from hydro import vessel_spec as _vs

SWITCHES = dict(
    dDt_full=dict(step="S1", implemented=True, what=(
        "complete material derivative of the sectional added mass, "
        "Dm_a/Dt = (dm_a/dd) Dd/Dt - u (dm_a/dx)|_d, over the forebody taper "
        "with the chine-wetting point located inside each strip (d linear, "
        "chine analytic); momentum flux V Dm_a/Dt counted on entry only "
        "(V > 0, Zarnick's rule, and Dm_a/Dt > 0); astern (u < 0) without "
        "the u term until S2 (design 2.5)")),
    chine_exact_all=dict(step="S1b", implemented=True, needs=("dDt_full",),
                         what=(
        "the strip-exact chine-wetting location of dDt_full applied to "
        "every strip, the prismatic ones too: with d linear over a strip the "
        "flux is V^2 times the integral of dm_a/dd over its uncapped part "
        "(continuous in the state) instead of v0's centre decision")),
    kpp_strip=dict(step="P0b", implemented=True, what=(
        "the keel curvature in the m_a u^2 kpp term as the strip integral "
        "of kpp over each strip divided by its length, (kp(x_j+1/2) - "
        "kp(x_j-1/2)) / dx, instead of v0's centre value: second order in "
        "the station count, exact at a jump of kpp inside a strip")),
    transom=dict(step="S2a", implemented=True, needs=("dDt_full",), what=(
        "the longitudinal hydrostatic and long-wave Froude-Krylov force and "
        "its pitch moment as the exact integral of the plant's pressure "
        "field over the hull in telescoped form (-a_b [rho g int A eta_x "
        "dx + Q_T], moment likewise; bottom, sides and taper together, no "
        "geometric derivative) plus the transom face over its wetted "
        "height; the second copy of the wave force (Xw) deleted; the "
        "transom flow state a boundary condition (the dead-water level, "
        "empirical threshold F_T_crit, hydro/vessel_spec."
        "TRANSOM_CLOSURE); astern the transom leads: face wet, its "
        "stagnation drag, and the entry delta of the added mass at the "
        "leading transom in the momentum flux (design 2.2, 2.3, 2.8, "
        "2.10)")),
    normal_rate=dict(step="P1a", implemented=True, needs=("transom",),
                     what=(
        "the horizontal force of the sections' dynamic momentum as the "
        "complete derivative of the VECTOR momentum m_a V n, n the strip's "
        "normal (-tau, 1): v0 projects D/Dt(m_a V) by -tau and drops "
        "-m_a V Dtau/Dt, Dtau/Dt = SP q - u kpp (the normal turns with the "
        "pitch rate and along a curved keel); without it a pitching hull at "
        "rest gets a nonzero cycle-mean surge force (second order) and "
        "drifts; with it that force is -D/Dt(m_a V tau), zero on average")),
    verlet=dict(step="P1a", implemented=True, what=(
        "the second stage of the plant's two-stage step evaluated at the "
        "step's end positions (Heun's own position update) instead of the "
        "Euler-predicted ones: the velocity-Verlet (Stormer-Verlet) scheme "
        "for the position-dependent forces, no energy growth from them in "
        "the wave band (Heun's amplification |R(i w h)|^2 = 1 + (w h)^4 / 4 "
        "> 1); velocity-dependent forces get Heun's treatment (not "
        "symplectic for gyroscopic ones); same cost, same substep logic, "
        "same measurements")),
    fk_exact=dict(step="P1b", implemented=True, needs=("transom",), what=(
        "the exact linear incident (Froude-Krylov) pressure on every "
        "section polygon and on the wetted transom face: depth decay "
        "e^{kz} measured from the local surface, variation across the "
        "beam, any heading, the phase integrated exactly over each strip; "
        "a_b on its vertical and longitudinal part (one pressure field), "
        "none on the lateral force, roll and yaw moments; replaces v0's "
        "lateral wave-slope force and, beyond its long-wave limit, the "
        "roll moment about the wave slope (design 2.2; S3)")),
    orbital=dict(step="P1b", implemented=True, needs=("fk_exact",), what=(
        "the wave orbital velocities and accelerations (linear, at each "
        "section's centroid depth) in every relative velocity: the normal "
        "velocity V (planing lift, momentum factor, cross-flow drag), the "
        "surge inflow u_r (planing lift, convective rate, friction, the "
        "transom's flow state), the lateral v_r (slender-body lift, Munk "
        "moment, cross-flow drag); the added mass's rate on the geometric "
        "immersion rate; without lateral_phi0 the lateral diffraction + a22 "
        "a_oy with its yaw lever and the roll moment a24 a_oy, a22 and a24 "
        "the upright section's gravity-free 2D values (design 2.1; S3)")),
    wagner_entry=dict(step="P1b", implemented=True,
                      needs=("dDt_full", "transom", "normal_rate"), what=(
        "Wagner's pile-up only for planes that entered from dry: each "
        "earth-fixed plane carries a flow state -- E in an entry from dry, "
        "X in the exit after it, F oscillating about a fixed waterline -- "
        "a piecewise-constant field along the hull whose boundaries move "
        "exactly with the relative inflow (hidden plant state, "
        "hidden_state()); E and X keep Zarnick's added mass and S1's "
        "entry-only flux (no flux on exit; steady planing and every steady "
        "state unchanged), an X plane's added mass never growing along its "
        "path (min with the chine half-beam where it turned: without flux "
        "its momentum can only be shed, P1c round 3); a plane becomes F "
        "when it enters again after an "
        "exit without drying, or at rest (no travel): m_0, the exact "
        "gravity-free 2D value (hydro/section2d), with the Lagrangian force "
        "m_0 V' + 1/2 V^2 dm_0/dd and the horizontal force -D/Dt(m_0 V "
        "tau) (normal_rate's form, which the surge force needs: hence "
        "transom and normal_rate); a strip's force is that of its states' "
        "fractions (register T1, A1)")),
    lateral_phi0=dict(step="P1b", implemented=True, what=(
        "sway, roll and yaw added masses and couplings on the gravity-free "
        "(phi = 0) basis of the vertical, from the exact 2D section values "
        "at the instantaneous immersion and heel (hydro/section2d), on the "
        "left of the three lateral equations solved together; the sections' "
        "lateral and roll impulses under -D/Dt (orbital diffraction, the "
        "added mass's rate, shedding at the transom, Munk moment), the "
        "Lagrangian reactions of their kinetic energy and Kirchhoff's surge "
        "term (design 2.9; register E10, E11, T2)")),
    fast_path=dict(step="P1c", implemented=True, what=(
        "implementation only, no physics: the same derivative and step, "
        "bit for bit, with less repeated work -- the GZ plant's roll swap "
        "reads the section area and the wave-slope mean the derivative "
        "has just computed (PlaningVesselGZ.deriv recomputes them), the "
        "pump's intake submergence evaluates the surface elevation alone "
        "(v0's prop_submergence evaluates all five wave quantities); "
        "proven equal to the path without it on the golden runs "
        "(studies/anyspeed_p1c_check.py part fastpath)")),
    radiation_3d=dict(step="S4", implemented=False, what=(
        "low-speed 3D radiation and diffraction: memory, 3D added mass, "
        "diffraction difference (design 2.6)")),
    heel_hydro=dict(step="S5a", implemented=True, needs=("transom",), what=(
        "heeled-section hydrostatics and two-way roll: every station's "
        "polygon (keel, chines, deck edges: the frozen GZ table's) at the "
        "instantaneous heave, trim and heel, clipped exactly below the "
        "local surface (the station's elevation, tilted by its transverse "
        "wave slope), p = rho g (eta - z) integrated for the vertical "
        "and lateral force and the roll, pitch and yaw moments, the "
        "longitudinal force telescoped over the heeled transom section and "
        "face; a_b on the vertical, longitudinal and pitch parts only "
        "(P1c's convention, S5a review: G1); the keel depth of every "
        "station with the heel, d = H - zk cos phi, in the vertical "
        "dynamics; replaces the GZ-at-rest-trim lookup for both plants "
        "(module docstring 'S5a')")),
    lateral_cross=dict(step="S5a", implemented=True,
                       needs=("lateral_phi0", "heel_hydro"), what=(
        "the complete 2D phi = 0 cross-plane impulse dynamics in the "
        "plant's level, yaw-rotating frame: a22, a23, a24, a34, a44 of the "
        "heeled section about the CG (the table's body-frame values "
        "shifted along body z and rotated), the heave-heave entry the "
        "vertical model's own m33, one kinetic energy for all the "
        "Lagrangian reactions (the roll one with 1/2 W_r^2 dm33/dphi); per "
        "earth-fixed plane F = -D/Dt(M nu) + dT/dq with dT/dphi at a fixed "
        "CG (the fixed-waterline form on every plane: unverified for planes "
        "entering from dry, register S5-7), the wave velocities' rates at "
        "fixed body x, shedding at the trailing edge, yaw and pitch Munk "
        "sums, the Kirchhoff partners of the horizontal impulse; sway, "
        "heave, roll, pitch and yaw solved together (module docstring "
        "'S5a')")),
    lateral_roll=dict(step="S5b", implemented=False, what=(
        "the asymmetric 2D+t of a heeled planing section per side (wetted "
        "half-beams c_L, c_R: dynamic roll restoring and lift damping at "
        "speed), chine-vortex roll damping, the gravity part of the "
        "lateral problem (design 2.9; S5b)")),
    steady_gravity=dict(step="S6", implemented=False, what=(
        "steady gravity correction: wave resistance, sinkage, trim through "
        "the hump (design 2.7)")),
    jet_cavitation=dict(step="S7", implemented=False, what=(
        "cavitation limit of the jet's thrust at low speed (design 2.11)")),
)
# 'v0': all off. 'as1': the switches implemented and verified so far.
# P1a: verlet passed every condition registered for it. transom and
# normal_rate failed two of the whole-plant gates registered as their
# adoption conditions and are in the preset as a RECORDED EXCEPTION decided
# by the orchestrator after the P1a review (learn/meta/
# ANYSPEED_PLAN_2026-10-05.md, P1a): gate 3a (closed-body wave force within
# 5 % of the textbook value) fails at 6 kn by exactly v0's inherited
# a_b(C_V) = 0.904, which v0's Xw carried too (with a_b divided out every
# case is within 0.34 %); gate 5 (|u| monotone in the astern coast-down)
# fails because a trimmed hull that heaves exchanges horizontal fluid
# impulse -dG/dt, G = sum m_a V tau dx, with an undamped heave/pitch
# oscillation at rest (S4) -- u never changes sign. Without them the preset
# runs astern from rest at zero thrust (-6.7 m/s, -13 kn, after 60 s: the
# spurious hydrostatic force, an R3 violation). The failures are kept on
# record (studies/_cache/anyspeed/p1a/round1, round2, fk.txt, astern.txt)
# and both gates are re-registered for later steps (FK1 against a_b(u) x
# the textbook value; the coast-down by period-averaged |u|)
# P1c (learn/meta/ANYSPEED_PLAN_2026-10-05.md, P1c): fk_exact adopted under
# ACCEPTANCE E1's revision of 2026-10-06 -- it passed every physics check
# registered for it (P1b review F1-F7, re-run on the P1c implementation);
# E1 (3 ms a step) is its recorded shortfall, the step's performance work
# done in the same round; fast_path is an implementation switch (no
# physics, bit for bit the same runs)
PRESETS = dict(v0=(), as1=("dDt_full", "chine_exact_all", "kpp_strip",
                          "transom", "normal_rate", "verlet", "fk_exact",
                          "fast_path"))
# the presets before P1a's adoption, kept for the checks and the goldens
# that pin them (studies/anyspeed_regress_as.py GOLDEN_SETS)
AS1_PRE_P1A = ("dDt_full", "chine_exact_all", "kpp_strip")
AS1_P1A_VERLET = ("dDt_full", "chine_exact_all", "kpp_strip", "verlet")
# the candidate P1a verified (studies/anyspeed_p1a_check.py, studies/_cache/
# anyspeed/p1a/): since its adoption, the as1 preset itself
P1A_CANDIDATE = ("dDt_full", "chine_exact_all", "kpp_strip", "transom",
                 "normal_rate", "verlet")
# the as1 preset of P1a (since the P1a review) and the candidate step P1b
# verifies (studies/anyspeed_p1b_check.py, studies/_cache/anyspeed/p1b/):
# it plus the four P1b switches
AS1_P1A = ("dDt_full", "chine_exact_all", "kpp_strip", "transom",
           "normal_rate", "verlet")
P1B_CANDIDATE = AS1_P1A + ("fk_exact", "orbital", "wagner_entry",
                           "lateral_phi0")
# the as1 preset since P1c: P1a's plus fk_exact, and the implementation
# switch fast_path
AS1_P1C = AS1_P1A + ("fk_exact", "fast_path")
# the constructor keyword that overrides a hull version's preset (tests)
TEST_KW = "test_switches"
# names the verbatim copies of v0's code read by their own binding (NU and
# RHO_AIR: the copies read self.num now, but a patched PV.NU still breaks
# all-off == v0)
V0_GLOBALS = ("NU", "G", "RHO_AIR", "SP", "_load", "JetPump", "Waterjet",
              "Nozzle")
_STEADY = {}
_TOP = {}
# every hull an as plant was built for, kept alive so that its id -- the key
# of the frozen GZ table cache, sim/planing_vessel_gz._GZ -- is never reused
_PINNED = {}


def check_v0_globals():
    """Raise unless the module-level names the copied derivative reads are
    still the very objects sim/planing_vessel holds. A patch of the v0
    module (PV.NU = ...) reaches v0's derivative and not the copy's, so
    all-off == v0 would fail silently."""
    g = globals()
    bad = [n for n in V0_GLOBALS if g[n] is not getattr(_pv, n)]
    if bad:
        raise RuntimeError(
            f"sim.planing_vessel.{', '.join(bad)} was patched after "
            f"sim.planing_anyspeed was imported; the as plants copy v0's "
            f"derivative with their own bindings of these names and would no "
            f"longer equal v0 with every switch off. Patch both modules, or "
            f"run this study on v0 hulls only")


def _vsec(d, yc, tanb):
    """The strip plant's section (sim/planing_vessel._section: a V of
    deadrise beta, tan(beta) = tanb, to the chine half-beam yc, vertical
    sides above it) immersed to keel depth d: (A, m1, m2), its area and the
    first and second moments of that area about the keel (h = height above
    the keel). Width 2 h / tanb below the chine height yc tanb, 2 yc above:

        A  = b^2 / tanb + 2 yc e
        m1 = 2 b^3 / (3 tanb) + yc ((dch + e)^2 - dch^2)
        m2 = b^4 / (2 tanb) + 2/3 yc ((dch + e)^3 - dch^3)

    with dch = yc tanb, b = min(d+, dch), e = max(d+ - dch, 0). A is the
    plant's own expression (the same floats)."""
    dp = np.maximum(d, 0.0)
    dch = yc * tanb
    b = np.minimum(dp, dch)
    e = np.maximum(dp - dch, 0.0)
    top = dch + e
    A = b ** 2 / tanb + 2.0 * yc * e
    m1 = 2.0 * b ** 3 / (3.0 * tanb) + yc * (top * top - dch * dch)
    m2 = b ** 4 / (2.0 * tanb) + 2.0 / 3.0 * yc * (top ** 3 - dch ** 3)
    return A, m1, m2


def _vsec_q(d, yc, tanb):
    """(q, s2, A, m1) of the section immersed to keel depth d, for the
    hydrostatic pressure p = rho g (d - h) about its free surface:
    q = int (d - h) dA = d A - m1 (the pressure force / rho g, the first
    moment of the area about the surface; dq/dd = A) and s2 = int h (d - h)
    dA = d m1 - m2 (with q, the moment of that pressure about any height)."""
    A, m1, m2 = _vsec(d, yc, tanb)
    dp = np.maximum(d, 0.0)
    return dp * A - m1, dp * m1 - m2, A, m1


def _vsec_qs(d, yc, tanb):
    """_vsec_q for one section, in floats (the transom face, twice per
    derivative: numpy's per-call overhead on scalars was most of the
    switch's cost): (q, s2, A, m1), the same formulas."""
    dp = d if d > 0.0 else 0.0
    dch = yc * tanb
    b = dp if dp < dch else dch
    e = dp - dch if dp > dch else 0.0
    top = dch + e
    A = b * b / tanb + 2.0 * yc * e
    m1 = 2.0 * b ** 3 / (3.0 * tanb) + yc * (top * top - dch * dch)
    m2 = b ** 4 / (2.0 * tanb) + 2.0 / 3.0 * yc * (top ** 3 - dch ** 3)
    return dp * A - m1, dp * m1 - m2, A, m1


def _clip_moments(Y, Z, H, second=False):
    """S5a (switch heel_hydro): the area and moments of the parts of closed
    polygons below the line Z = H: (A, Sy, Sz) = int (1, Y, Z) dA, with
    second=True also (Syy, Syz, Szz) = int (Y^2, Y Z, Z^2) dA.

    Y, Z (..., k): vertices, counter-clockwise in (Y, Z); H broadcasts to
    (...). Green's theorem, every integrand written as Q dZ:

        A = oint Y dZ,  Sy = oint Y^2/2 dZ,  Sz = oint Y Z dZ,
        Syy = oint Y^3/3 dZ,  Syz = oint Y^2 Z/2 dZ,  Szz = oint Y Z^2 dZ,

    so the waterline chord(s) Z = H (dZ = 0) add nothing and no clipping is
    needed: each edge P0 -> P1, P(t) = P0 + t (P1 - P0), contributes its
    immersed part, t <= tc on a rising edge and t >= tc on a falling one,
    tc where it meets H (sim/planing_vessel_gz._immersed, extended to the
    first and second moments)."""
    Y1, Z1 = np.roll(Y, -1, axis=-1), np.roll(Z, -1, axis=-1)
    dY, dZ = Y1 - Y, Z1 - Z
    h = np.asarray(H)[..., None]
    flat = dZ == 0.0
    tc = np.clip((h - Z) / np.where(flat, 1.0, dZ), 0.0, 1.0)
    up = dZ > 0.0
    t0 = np.where(up, 0.0, tc)
    t1 = np.where(up, tc, 1.0)
    d1, d2, d3 = t1 - t0, t1 * t1 - t0 * t0, t1 ** 3 - t0 ** 3
    A = (dZ * (Y * d1 + 0.5 * dY * d2)).sum(-1)
    Sy = (0.5 * dZ * (Y * Y * d1 + Y * dY * d2 + dY * dY * d3 / 3.0)).sum(-1)
    Sz = (dZ * (Y * Z * d1 + 0.5 * (Y * dZ + Z * dY) * d2
                + dY * dZ * d3 / 3.0)).sum(-1)
    if not second:
        return A, Sy, Sz
    d4 = t1 ** 4 - t0 ** 4
    Syy = (dZ / 3.0 * (Y ** 3 * d1 + 1.5 * Y * Y * dY * d2
                       + Y * dY * dY * d3 + 0.25 * dY ** 3 * d4)).sum(-1)
    Syz = (0.5 * dZ * (Y * Y * Z * d1 + 0.5 * (Y * Y * dZ + 2.0 * Y * dY * Z)
                       * d2 + (2.0 * Y * dY * dZ + dY * dY * Z) * d3 / 3.0
                       + 0.25 * dY * dY * dZ * d4)).sum(-1)
    Szz = (dZ * (Y * Z * Z * d1 + 0.5 * (Z * Z * dY + 2.0 * Z * dZ * Y) * d2
                 + (2.0 * Z * dZ * dY + dZ * dZ * Y) * d3 / 3.0
                 + 0.25 * dZ * dZ * dY * d4)).sum(-1)
    return A, Sy, Sz, Syy, Syz, Szz


def _vsec_iyy(d, yc, tanb):
    """S5a: int y^2 dA of the upright V-and-wall section (_vsec's) immersed
    to keel depth d: b^4 / (6 tanb^3) + e (2/3) yc^3, b = min(d+, yc tanb),
    e = max(d+ - yc tanb, 0) (floats). The long-wave limit of fk_exact's
    transom yaw terms (sim/as_kernels.fk_rows: E_lam / lam -> -int y^2 dA as
    k, lam -> 0), which switch heel_hydro's tilted polygons carry
    themselves (_hh_lateral)."""
    dp = d if d > 0.0 else 0.0
    dch = yc * tanb
    b = dp if dp < dch else dch
    e = dp - dch if dp > dch else 0.0
    return b ** 4 / (6.0 * tanb ** 3) + e * (2.0 / 3.0) * yc ** 3


def _xmass_dd(b, st, y, edges, cz, dcz):
    """S5a review (switch lateral_cross with wagner_entry): d/dd of
    _es_xmass's mean over each strip's X planes of min(cz_j, Y)^2, at fixed
    segments -- the mean of 2 cz_j dcz_j where cz_j < Y, else 0 (dcz_j the
    keel-depth derivative of the station's Zarnick half-beam); the same
    traversal as _es_xmass."""
    n = edges.size - 1
    fx, qd = np.zeros(n), np.zeros(n)
    k = np.nonzero(st[1:-1] == ES_X)[0]
    if not k.size:
        return qd
    a0, a1, Y = b[k], b[k + 1], y[k + 1]
    j0 = np.clip(np.searchsorted(edges, a0, side="right") - 1, 0, n - 1)
    j1 = np.clip(np.searchsorted(edges, a1, side="left") - 1, 0, n - 1)
    cnt = j1 - j0 + 1
    rep = np.repeat(np.arange(k.size), cnt)
    jj = np.repeat(j0, cnt) + (np.arange(int(cnt.sum()))
                               - np.repeat(np.cumsum(cnt) - cnt, cnt))
    ln = np.maximum(np.minimum(a1[rep], edges[jj + 1])
                    - np.maximum(a0[rep], edges[jj]), 0.0)
    np.add.at(fx, jj, ln)
    np.add.at(qd, jj, ln * np.where(cz[jj] < Y[rep], 2.0 * cz[jj] * dcz[jj],
                                    0.0))
    return np.where(fx > 0.0, qd / np.where(fx > 0.0, fx, 1.0), 0.0)


def resolve(version, switches=None):
    """The switches that are on, in SWITCHES order: the preset of `version`
    when `switches` is None, else a preset named by a string, else an
    iterable of switch names."""
    if switches is None:
        switches = version
    if isinstance(switches, str):
        if switches not in PRESETS:
            raise KeyError(f"preset {switches!r}: one of {sorted(PRESETS)} "
                           f"(switch names go in a tuple)")
        names = set(PRESETS[switches])
    else:
        names = set(switches)
    bad = sorted(names - set(SWITCHES))
    if bad:
        raise KeyError(f"unknown switches {bad}; known: {list(SWITCHES)}")
    todo = sorted(n for n in names if not SWITCHES[n]["implemented"])
    if todo:
        raise NotImplementedError(
            "not implemented yet: " + ", ".join(
                f"{n} ({SWITCHES[n]['step']})" for n in todo))
    # the most dependent switch first, in a fixed order (P1c, track W: a set
    # iterates in the process's hash order, so with two switches missing
    # needs the one reported changed from run to run -- the regression's
    # 'wagner_entry without dDt_full' case, whose transom lacks dDt_full too)
    for n in sorted(names, key=list(SWITCHES).index, reverse=True):
        miss = sorted(set(SWITCHES[n].get("needs", ())) - names)
        if miss:
            raise ValueError(f"switch {n} needs {miss}")
    return tuple(n for n in SWITCHES if n in names)


# ------------------------------------- P1b: the flow states of the planes
# (switch wagner_entry; module docstring 'P1b WAGNER_ENTRY'). The state of
# the earth-fixed planes along the hull is a piecewise-constant field in body
# x: sorted boundaries b (m,) and the states st (m + 1,) of the segments
# (-inf, b0), (b0, b1), ..., (b_m-1, +inf); outside the hull every plane is
# dry, so the first and last segments are E.
ES_E, ES_X, ES_F = 0, 1, 2
ES_NAMES = ("entry from dry", "exit after an entry", "fixed waterline")
# the state a plane takes from (its state, the class of its immersion):
# class 0 dry, 1 wet and not entering (Vg <= 0), 2 wet and entering (Vg > 0).
# Dry -> E (a plane wetted again enters from dry); E turning (Vg <= 0) -> X;
# X entering again while still wet -> F; F stays F while wet.
_ES_MAP = np.array([[ES_E, ES_X, ES_E],
                    [ES_E, ES_X, ES_F],
                    [ES_E, ES_F, ES_F]], dtype=np.int8)
# fractions within this of 0 or 1 are snapped (round-off of the cumulative
# sums, so a strip wholly in one state is exactly in it)
_ES_SNAP = 1e-12
# sub-steps of the field per advance on interpolated kinematics (P1c
# review; _es_advance). A numerical parameter of the plant: a plant
# attribute es_nsub overrides it (checks).
ES_NSUB = 4
# P1c round 3 (track W, plan '### wagner 低速'): an exiting plane (X) does
# not draw new fluid into its motion -- its added mass never grows along its
# path (the momentum of a plane without flux can only be shed to the wake,
# never created: _p1b_kin; the choice of this over the attached form is a
# modelling assumption, register A8, round 4). The field carries, per
# segment, y: for X the chine half-beam Y where its planes turned (E -> X),
# +inf otherwise.
# Neighbouring X segments are merged only when their Y are equal (register
# P18): ES_YTOL_REL = 0. A positive value bins the Y (width ES_YTOL_REL x
# the hull's largest half-beam) and merges a bin with the smaller Y -- fewer
# segments, but a bin edge is a discontinuity that round-off can cross: the
# Froude twins (check twin, internal 1e-10) parted at 2e-4 with 2e-3. A
# plant attribute es_xmem = False turns the memory off (P1c round 2's X,
# for the checks).
ES_YTOL_REL = 0.0
# the memory's default (a plant attribute es_xmem overrides it; checks only)
ES_XMEM = True


def _es_cross(xs, g):
    """Body x of the changes of (g <= 0) between adjacent stations, g linear
    between them."""
    neg = g <= 0.0
    j = np.nonzero(neg[:-1] != neg[1:])[0]
    if not j.size:
        return xs[:0]
    g0, g1 = g[j], g[j + 1]
    return xs[j] + (xs[j + 1] - xs[j]) * (g0 / (g0 - g1))


def _es_points(xs, d, vg, edges):
    """The breakpoints of the class field inside the hull: the zeros of the
    keel depth d (the waterline crossing) and of the immersion rate Vg,
    both linear between the stations -- located inside the strips, as S1
    locates the chine (a strip-centre decision of the wetness made the
    planes' states change a strip at a time as the waterline moved, and the
    F share converged only at first order in the step: P1b review, W4's
    exploratory runs)."""
    return np.concatenate([_es_cross(xs, d), _es_cross(xs, vg)])


def _es_class(xs, d, vg, edges, xm):
    """The class of the planes at body x xm: 0 dry (d <= 0), 1 wet and Vg <=
    0 (not entering), 2 wet and entering; d and Vg linear between the
    stations, constant beyond the end ones."""
    dm = np.interp(xm, xs, d)
    vm = np.interp(xm, xs, vg)
    return np.where(dm <= 0.0, 0, np.where(vm <= 0.0, 1, 2))


def _es_build(lo, hi, P, S, Y=None, ytol=0.0):
    """The field of the segments [lo, P0], [P0, P1], ..., [P_k, hi] with
    the states S (k + 2,), E outside [lo, hi], equal neighbours merged.
    With Y (k + 2,), the X planes' turn half-beams (P1c round 3): returns
    (b, st, y), y +inf off X; X neighbours merged only within one bin of
    width ytol (the smaller Y kept)."""
    b = np.concatenate([[lo], P, [hi]])
    st = np.concatenate([[ES_E], S, [ES_E]]).astype(np.int8)
    if Y is None:
        keep = st[1:] != st[:-1]
        return b[keep], np.concatenate([st[:1], st[1:][keep]])
    isx = st == ES_X
    y = np.where(isx, np.concatenate([[np.inf], Y, [np.inf]]), np.inf)
    q = np.where(isx, np.floor(y / ytol) if ytol > 0.0 else y, -1.0)
    keep = (st[1:] != st[:-1]) | (q[1:] != q[:-1])
    starts = np.concatenate([[0], np.nonzero(keep)[0] + 1])
    return b[keep], st[starts], np.minimum.reduceat(y, starts)


def _es_scan(xs, d, vg, u, edges, ycf=None, ytol=0.0):
    """The steady field: every plane's state from its path along the hull
    in steady flow, from the leading edge (the stem going ahead, the transom
    astern; it arrives dry: E) through the classes it meets. With u = 0 the
    planes do not travel: every wet plane oscillates about a fixed
    waterline (F) -- the boat has been at rest. Used by the captive runs
    (steady states, the census) and to start a run. With ycf (the chine
    half-beam at body x; P1c round 3) also the X planes' turn half-beams:
    in steady flow a plane turns where it meets the class change, the
    segment's leading end (the leading edge itself for a plane arriving
    already exiting); returns (b, st, y)."""
    lo, hi = float(edges[0]), float(edges[-1])
    P = np.unique(_es_points(xs, d, vg, edges))
    e = np.concatenate([[lo], P, [hi]])
    cls = _es_class(xs, d, vg, edges, 0.5 * (e[:-1] + e[1:]))
    Y = None if ycf is None else np.full(cls.size, np.inf)
    if u == 0.0:
        S = np.where(cls == 0, ES_E, ES_F)
    else:
        S = np.empty(cls.size, np.int8)
        cur = ES_E
        yk = np.inf
        for i in (range(cls.size - 1, -1, -1) if u > 0.0
                  else range(cls.size)):
            nxt = _ES_MAP[cur, cls[i]]
            if Y is not None:
                if nxt != ES_X:
                    yk = np.inf
                elif cur != ES_X:
                    yk = float(ycf(e[i + 1] if u > 0.0 else e[i]))
                Y[i] = yk
            cur = nxt
            S[i] = cur
    if Y is None:
        return _es_build(lo, hi, P, S)
    return _es_build(lo, hi, P, S, Y, ytol)


def _es_xmass(b, st, y, edges, cz):
    """Per strip [edges_j, edges_j+1]: the fraction of its planes in state X
    and the mean over them of min(cz_j, Y)^2 (cz the station's Zarnick
    half-beam, Y each X segment's turn half-beam) -- exact for the
    piecewise-constant field with the station's cz (P1c round 3)."""
    n = edges.size - 1
    fx, qx = np.zeros(n), np.zeros(n)
    k = np.nonzero(st[1:-1] == ES_X)[0]
    if not k.size:
        return fx, qx
    a0, a1, Y = b[k], b[k + 1], y[k + 1]
    j0 = np.clip(np.searchsorted(edges, a0, side="right") - 1, 0, n - 1)
    j1 = np.clip(np.searchsorted(edges, a1, side="left") - 1, 0, n - 1)
    cnt = j1 - j0 + 1
    rep = np.repeat(np.arange(k.size), cnt)
    jj = np.repeat(j0, cnt) + (np.arange(int(cnt.sum()))
                               - np.repeat(np.cumsum(cnt) - cnt, cnt))
    ln = np.maximum(np.minimum(a1[rep], edges[jj + 1])
                    - np.maximum(a0[rep], edges[jj]), 0.0)
    np.add.at(fx, jj, ln)
    np.add.at(qx, jj, ln * np.minimum(cz[jj], Y[rep]) ** 2)
    qx = np.where(fx > 0.0, qx / np.where(fx > 0.0, fx, 1.0), 0.0)
    fx = fx / np.diff(edges)
    fx = np.where(fx > 1.0 - _ES_SNAP, 1.0, np.where(fx < _ES_SNAP, 0.0, fx))
    return fx, qx


def _es_advance(b, st, xs, d, vg, ur, dt, edges, ur0=None, k0=None,
                km=None, dd=None, nsub=1, y=None, ycf=None, ytol=0.0):
    """One step of the field: every plane carried with the relative inflow
    (body x decreasing at ur: boundaries moved EXACTLY, ur linear between
    the stations -- no interpolation of the states, so a state boundary
    moves at the plane speed whatever the Courant number); the planes that
    arrive over the leading edge are dry (E); then each plane's transition
    by its class now (_ES_MAP). A field all in E stays all in E where no
    wet plane has stopped entering (steady planing): it is returned as it
    is.

    The path db/dt = -ur(b, t) over the step: with ur0, the inflow at the
    field's own time (the start of the step), by Heun's rule, b* = b -
    ur0(b) dt, b - (ur0(b) + ur(b*)) dt / 2 -- second order in the step,
    exact for an inflow uniform along the hull and linear in time (P1c:
    the end-of-step inflow alone, b - ur(b) dt, misplaces each boundary by
    (du/dt) dt^2 / 2 a step, which accumulates over a plane's passage into
    an error of first order in the step -- W4's F share); without ur0 (a
    field whose start inflow is not known: older records) that end-of-step
    rule.

    With k0, the kinematics at the field's own time (a dict d, vg, ur, dd,
    t; dd the immersion rate at fixed body x, d's time derivative), the
    step is taken in nsub sub-steps (P1c review): the kinematics inside the
    step are interpolated in time -- d by the cubic Hermite interpolant of
    its values and time derivatives at the two ends (dd: kb.ddot), vg and
    ur by the quadratic through the end values and km, the kinematics at
    the field's previous time (linear without km) --, each sub-step moves
    the boundaries by Heun's rule on the sub-step and applies the
    transitions by the class at its end. The last sub-step ends on the end
    kinematics themselves, so the classes at the step's end are the
    single-step ones; what the sub-steps add are the transitions inside the
    step (a plane that stops entering and enters again within one step).
    Vg's time derivative needs the accelerations of the derivative being
    computed, hence the three-point quadratic for it (interpolation error
    O(dt^3), against O(dt^2) for the linear interpolant, whose sub-steps
    left the field-only F share error at 0.17 % at 28 kn: P1c track W's
    exploratory runs).

    With y (the X planes' turn half-beams, P1c round 3; ycf the chine
    half-beam at body x, ytol the merging bin) the field is (b, st, y) in
    and out: y moves with its segment exactly as the states do; a piece
    that turns E -> X in a (sub)step takes the half-beam at its midpoint (its
    planes turned within it during the sub-step), one arriving over the
    leading edge already exiting the half-beam of the leading edge."""
    if k0 is not None:
        return _es_substeps(b, st, xs, d, vg, ur, dt, edges, k0, km, dd,
                            max(int(nsub), 1), y, ycf, ytol)
    return _es_step1(b, st, xs, d, vg, ur, dt, edges, ur0, False, y, ycf,
                     ytol)


def _es_lagr(tm, fm, f0, f1, h, tau):
    """f at time tau in [0, h] from samples f(-tm) = fm (None: linear),
    f(0) = f0, f(h) = f1: the Lagrange polynomial through them."""
    if fm is None:
        s = tau / h
        return f0 + s * (f1 - f0)
    a, b_, c = -tm, 0.0, h
    lm = (tau - b_) * (tau - c) / ((a - b_) * (a - c))
    l0 = (tau - a) * (tau - c) / ((b_ - a) * (b_ - c))
    l1 = (tau - a) * (tau - b_) / ((c - a) * (c - b_))
    return lm * fm + l0 * f0 + l1 * f1


def _es_substeps(b, st, xs, d, vg, ur, dt, edges, k0, km, dd, n, y=None,
                 ycf=None, ytol=0.0):
    """_es_advance in n sub-steps on interpolated kinematics (its
    docstring)."""
    d0, vg0, ur0, dd0 = k0["d"], k0["vg"], k0["ur"], k0.get("dd")
    tm = vgm = urm = None
    if km is not None and float(k0["t"]) > float(km["t"]):
        tm = float(k0["t"]) - float(km["t"])
        vgm, urm = km["vg"], km["ur"]
    herm = dd0 is not None and dd is not None
    uni = bool(np.all(ur == ur[0]) and np.all(ur0 == ur0[0])
               and (urm is None or np.all(urm == urm[0])))
    u_prev = ur0
    for i in range(1, n + 1):
        if i == n:
            di, vi, ui = d, vg, ur
        else:
            tau = dt * i / n
            s = i / n
            if herm:
                h00 = (2.0 * s - 3.0) * s * s + 1.0
                h10 = ((s - 2.0) * s + 1.0) * s
                h01 = (3.0 - 2.0 * s) * s * s
                h11 = (s - 1.0) * s * s
                di = h00 * d0 + h10 * dt * dd0 + h01 * d + h11 * dt * dd
            else:
                di = _es_lagr(tm, None if tm is None else km["d"], d0, d,
                              dt, tau)
            vi = _es_lagr(tm, vgm, vg0, vg, dt, tau)
            ui = _es_lagr(tm, urm, ur0, ur, dt, tau)
        if y is None:
            b, st = _es_step1(b, st, xs, di, vi, ui, dt / n, edges, u_prev,
                              uni)
        else:
            b, st, y = _es_step1(b, st, xs, di, vi, ui, dt / n, edges,
                                 u_prev, uni, y, ycf, ytol)
        u_prev = ui
    return (b, st) if y is None else (b, st, y)


def _es_step1(b, st, xs, d, vg, ur, dt, edges, ur0=None, uni=False, y=None,
              ycf=None, ytol=0.0):
    """One (sub)step of the field (_es_advance's docstring): Heun's rule
    with ur0, the end-of-step rule without. b is sorted, so the boundaries
    inside the hull are a slice; uni: ur0 and ur uniform along the hull
    (no orbital inflow), where np.interp returns the constant itself and
    the same floats come from the scalars (P1c review: the sub-steps made
    numpy's per-call overhead count). With y: (b, st, y) (P1c round 3)."""
    if st.size == 1 and not np.any((d > 0.0) & (vg <= 0.0)):
        return (b, st) if y is None else (b, st, y)
    lo, hi = float(edges[0]), float(edges[-1])
    i0 = int(np.searchsorted(b, lo, side="right"))
    i1 = int(np.searchsorted(b, hi, side="left"))
    b2 = np.concatenate(([lo], b[i0:i1], [hi]))
    st2 = np.concatenate(([ES_E, st[i0]], st[i0 + 1:i1 + 1],
                          [ES_E])).astype(np.int8)
    if ur0 is None:
        b3 = np.maximum.accumulate(b2 - np.interp(b2, xs, ur) * dt)
    elif uni:
        b3 = np.maximum.accumulate(b2 - 0.5 * (ur0[0] + ur[0]) * dt)
    else:
        v0 = np.interp(b2, xs, ur0)
        v1 = np.interp(b2 - v0 * dt, xs, ur)
        b3 = np.maximum.accumulate(b2 - 0.5 * (v0 + v1) * dt)
    j0 = int(np.searchsorted(b3, lo, side="right"))
    j1 = int(np.searchsorted(b3, hi, side="left"))
    inside = b3[j0:j1]
    P = np.unique(np.concatenate([inside, _es_points(xs, d, vg, edges)]))
    e = np.concatenate([[lo], P, [hi]])
    mid = 0.5 * (e[:-1] + e[1:])
    io = np.searchsorted(b3, mid, side="right")
    old = st2[io]
    cls = _es_class(xs, d, vg, edges, mid)
    S = _ES_MAP[old, cls]
    if y is None:
        return _es_build(lo, hi, P, S)
    # P1c round 3: the X planes' turn half-beams move with their segments;
    # a piece turning now takes the half-beam at its midpoint, one that
    # arrived over the leading edge (beyond the carried end boundaries)
    # already exiting the leading edge's
    y2 = np.concatenate(([np.inf, y[i0]], y[i0 + 1:i1 + 1], [np.inf]))
    isx = S == ES_X
    turn = isx & (old != ES_X)
    yn = np.where(isx, y2[io], np.inf)
    if turn.any():
        xt = np.where(mid > b3[-1], hi, np.where(mid < b3[0], lo, mid))
        yn = np.where(turn, ycf(xt), yn)
    return _es_build(lo, hi, P, S, yn, ytol)


def _es_frac(b, st, edges, which):
    """The fraction of each strip [edges_j, edges_j+1] whose planes are in
    state `which` (exact for the piecewise-constant field)."""
    if b.size < 2:
        return np.zeros(edges.size - 1)
    seg = (st[1:-1] == which)
    cum = np.concatenate([[0.0], np.cumsum(np.diff(b) * seg)])
    f = np.diff(np.interp(edges, b, cum)) / np.diff(edges)
    return np.where(f > 1.0 - _ES_SNAP, 1.0, np.where(f < _ES_SNAP, 0.0, f))


class _StripAS(PlaningVessel):
    """v0's strip-theory derivative, copied so that each design step can
    change its terms behind a switch (module docstring). Never built on its
    own: the strip theory of the as plants, and the class of their captive
    copies."""

    def __init__(self, hull, sea, dt=0.01, prop=None, rudder=None,
                 captive_u=None, **_ignored):
        """v0's constructor (sim/planing_vessel.py:80-158), the geometry,
        the plant constants and the jet's hard-coded time constants read
        from the hull (module docstring, 'THE HULL AS DATA')."""
        self._init_kw = dict(dt=dt)
        self.hull, self.sea, self.dt = hull, sea, dt
        h = hull
        self.L, self.B, self.rho = h.L, h.b, h.rho
        self.m = h.mass
        self.tanb = np.tan(np.radians(h.beta_deg))
        self.cosb = np.cos(np.radians(h.beta_deg))
        self.k_a, self.c_dc, self.cd_lat = h.k_a, h.c_dc, h.cd_lat
        # the plant's absolute constants (hydro/vessel_spec.PlantNumerics)
        self.num = _vs.numerics_of(h)
        # stations, from the transom forward; body x from the CG -- from the
        # geometry module (hydro/hull_geometry), v0's arrays for v0's hulls
        g = self._geom = _hg.stations_of(h)
        n = g.n
        self.dx = g.dx
        self.x = g.xi - h.lcg
        self.yc = g.yc
        self.zk = -h.vcg + g.rise
        self.kp = g.kp
        self.kpp = g.kpp
        if self.sw.kpp_strip:
            self.kpp = g.kpp_strip                      # P0b
        # rigid body
        m = self.m
        Ix = m * (h.k_roll * h.b) ** 2
        Iy = m * (h.k_pitch * h.L) ** 2
        Iz = m * (h.k_yaw * h.L) ** 2
        self.M = np.diag([m, m, m, Ix, Iy, Iz])
        self.Ix, self.Iy, self.Iz = Ix, Iy, Iz
        # at rest: where it floats, its draught and roll stability
        self.Z_rest, self.Th_rest = self._static()
        d0 = -(self.Z_rest + self.zk + SP * self.x * self.Th_rest)
        self.T = float(max(d0[0], 0.0))
        if h.T is None:
            h.T = self.T
        self.GM = self._gm(d0)
        # the waterjet at the transom, the intake on the keel ahead of it
        # (hydro/vessel_spec.jet_inputs: the complete JetSpec, every key
        # present, so the literal defaults below never apply)
        j = _vs.jet_inputs(h)
        x_n = -h.lcg + j.get("x_nozzle", 0.0)
        z_n = -h.vcg + j.get("z_nozzle", 0.15)
        x_i = -h.lcg + j.get("x_intake", 0.8)
        z_i = -h.vcg + j.get("z_intake", 0.0)
        pump = JetPump(np.pi * j.get("d_nozzle", 0.085) ** 2 / 4,
                       j.get("p_max", 224e3), eta_pump=j.get("eta_pump", 0.775),
                       zeta_n=j.get("zeta_n", 0.02),
                       zeta_i=j.get("zeta_i", 0.25), wake=j.get("wake", 0.05),
                       rho=self.rho)
        u_d = h.u_design
        t_max = pump.thrust(pump.p_max, u_d)
        self.prop = prop or Waterjet(
            pump, x_n, z_n, x_i, z_i, k_rev=j.get("k_rev", 0.6),
            tau_prime_loss=j.get("tau_prime_loss", 0.05),
            tau_prime_back=j.get("tau_prime_back", 0.5),
            t_max=t_max, t_min=-t_max / j.get("t_min_div", 3.0),
            tau=j.get("tau", 0.6),
            rate_max=t_max / j.get("t_rate", 0.3),
            dead_band=t_max * j.get("dead_band_ref", 50.0)
            / j.get("dead_band_ref_thrust", 12000.0),
            delay=j.get("delay", 0.1), u_ref=u_d, dt=dt)
        self.rudder = rudder or Nozzle(
            max_deg=j.get("nozzle_max_deg", 25.0),
            rate_deg=j.get("nozzle_rate_deg", 40.0),
            tau=j.get("nozzle_tau", 0.1), delay=j.get("nozzle_delay", 0.1),
            dt=dt, x_nozzle=x_n, z_nozzle=z_n)
        self.thrusters = [(float(x_n), 0.0, float(z_n))]
        self.rudder_points = self.thrusters
        self.intakes = [(float(x_i), 0.0, float(z_i))]
        # what the harnesses read off a plant
        keel_rest = self.Z_rest + self.zk + SP * self.x * self.Th_rest
        self.sec = SimpleNamespace(
            x=self.x, dx=self.dx, n=n, x_bow=float(self.x[-1]),
            x_stern=float(self.x[0]), keel=keel_rest, y0=self.yc)
        self.slam = SimpleNamespace(
            wagner_slope=lambda d, kr=keel_rest: self._section(d - kr)[3])
        self.draft_bow = float(-keel_rest[-1])
        self.freeboard = float(h.depth + keel_rest[-1])
        self.v_slam = 0.093 * np.sqrt(G * h.L)
        self.captive_u = captive_u
        # sway/roll/yaw added mass and damping at the design attitude, for
        # identification and for the equations (placeholder grade)
        self.visc = np.zeros(6)
        self.Mtot = self.M.copy()
        self._design_lateral()
        self._reset()

    def deriv(self, s, t):
        eta, nu = s[:6], s[6:12]
        thr, rud = s[12], s[13]
        x0, y0, z, phi, th, psi = eta
        u, v, w, p, q, r = nu
        x = self.x
        cps, sps = np.cos(psi), np.sin(psi)
        X, Y = x0 + x * cps, y0 + x * sps
        if self.sw.fk_exact and self._waves_on():
            # P1b: one evaluation of the component phases at the stations,
            # kept for the exact pressure and the orbital kinematics (in calm
            # water there are none: the code below is P1a's, bit for bit)
            e, et, ett, ex, ey = self._waves_b(X, Y, t, psi)
        else:
            e, et, ett, ex, ey = self._waves(X, Y, t)
        s_lon = ex * cps + ey * sps
        s_tr = -ex * sps + ey * cps
        if self.sw.heel_hydro:
            # S5a: the heeled sections (module docstring 'S5a'); None on the
            # upright branch -- no heel and no transverse slope anywhere --
            # where P1c's lines below are the same integral in closed form
            hh = self._hh_state(phi, s_tr)
        Th = self.Th_rest + th
        if self.sw.heel_hydro and hh is not None:
            # S5a: the keel depth of the heeled section (rolling about the
            # CG lifts the keel)
            d = e - (self.Z_rest + z + SP * x * Th) - self.zk * hh.ck
        else:
            d = e - (self.Z_rest + z + self.zk + SP * x * Th)
        tau = self.kp + SP * Th                     # local bottom slope
        V = et - (w + SP * x * q) + u * tau
        A, c, ma, dma, girth = self._section(d)
        wet = d > 0.0
        dx, rho = self.dx, self.rho
        ab = self._a_b(u)
        if self.sw.p1b:
            # P1b: the kinematics of the switches of step P1b (orbital
            # velocities, relative velocities, the entry state of the
            # sections and their added mass for oscillation); V becomes the
            # relative normal velocity, ma the added mass of the section's
            # flow state (module docstring 'P1b')
            kb = self._p1b_kin(s, t, d, A, tau, et, s_lon, s_tr, ma)
            V = kb.V
            ma = kb.ma
        if self.sw.transom:
            # S2a: the transom's keel depth and flow state, before the flux
            # (astern its entry delta needs them)
            if self.sw.orbital and kb.wv:
                # P1b: the inflow at the transom is the relative one
                self._trs = self._transom_kin(x0, y0, psi, z, Th, kb.urT, t)
            else:
                self._trs = self._transom_kin(x0, y0, psi, z, Th, u, t)
            if self.sw.heel_hydro and hh is not None:
                # S5a: the heeled transom section and wetted face (its keel
                # depth and face height replace the upright ones)
                self._hh_transom(hh, x0, y0, psi, z, Th, t,
                                 kb.urT if (self.sw.orbital and kb.wv)
                                 else u)
        # Transom correction (Garme 2005): the pressure falls to the
        # atmosphere at the transom, which strip theory does not know, so
        # every sectional load is tapered to zero there over a length that
        # grows with speed, a = 0.34 b C_V. Without it the lift sat too far
        # aft: Fridsma's model A (V/sqrt(L) 4) ran low and flat and then
        # porpoised by itself in calm water.
        ctr = self._ctr(u)
        ma, dma = ma * ctr, dma * ctr
        mom = np.where(V > 0.0, dma * V * V, 0.0)
        if self.sw.dDt_full:
            if self.sw.p1b_flux and kb.flux_b:
                # P1b: the flux on the relative and geometric immersion
                # rates, the relative inflow, the flow state's added mass;
                # hx the horizontal correction of the fixed-waterline planes
                flux, hx = self._flux_b(d, kb, dma, s_lon - tau)
            else:
                flux = self._flux_dDt(d, V, u, mom, s_lon - tau)    # S1
        else:
            flux = mom                                  # v0
        cf = 0.5 * rho * self.c_dc * 2.0 * c * V * np.abs(V) * ctr
        buoy = ab * rho * G * A * ctr
        if self.sw.heel_hydro and hh is not None:
            # S5a: the vertical force of the hydrostatic pressure on the
            # heeled polygons below the local surface (a_b: one field)
            self._hh_sections(hh, e, z, Th)
            buoy = ab * rho * G * hh.A * ctr
        if self.sw.fk_exact and kb.wv:
            # P1b: the exact incident pressure on every section beyond the
            # long-wave field: part of the same pressure field as buoy (a_b
            # on its vertical part), not tilted into Xp below
            fk = self._fk_b(d, A, ab, Th, kb)
            buoy = buoy + fk.fz
        if self.sw.transom:
            # the strip force without its hydrostatic part (same sum, same
            # order: rest is bit for bit v0's)
            if self.sw.orbital and kb.wv:
                # P1b: DV/Dt with the orbital accelerations and the relative
                # inflow of each section (module docstring 'P1b')
                dyn = ma * (kb.aoz - tau * kb.aox + 2.0 * SP * kb.ur * q
                            - kb.ur * kb.ur * self.kpp) + flux + cf
            else:
                dyn = ma * (ett + 2.0 * SP * u * q - u * u * self.kpp) + flux + cf
            rest = dyn + buoy
        else:
            rest = ma * (ett + 2.0 * SP * u * q - u * u * self.kpp) + flux \
                + cf + buoy

        # ---- everything that is not the strip force, as generalised forces
        tau_o = np.zeros(6)
        lw = wet.sum() * dx
        S = (girth * wet).sum() * dx
        uu = max(abs(u), self.num.u_floor)
        re = uu * max(lw, self.num.l_floor) / self.num.nu
        cfr = 0.075 / (np.log10(re) - 2.0) ** 2 + 0.0004
        Df = 0.5 * rho * u * abs(u) * cfr * S
        if self.sw.orbital and kb.wv:
            # P1b: the friction of each section on its relative inflow, the
            # friction line's Reynolds number on it too (Galilean: the hull
            # in a uniform current u_o is the hull at u - u_o)
            ure = np.maximum(np.abs(kb.ur), self.num.u_floor)
            cfj = 0.075 / (np.log10(ure * max(lw, self.num.l_floor)
                                    / self.num.nu) - 2.0) ** 2 + 0.0004
            Df = 0.5 * rho * (cfj * kb.ur * np.abs(kb.ur) * girth
                              * wet).sum() * dx
        if S > 0:
            xw = (x * girth * wet).sum() * dx / S
            zw = float(np.mean(self.zk[wet])) if wet.any() else self.zk[0]
        else:
            xw, zw = 0.0, self.zk[0]
        tau_o += _load(-Df, 0.0, 0.0, xw, 0.0, zw)
        Da = 0.5 * self.num.rho_air * self.hull.cda * u * abs(u)
        tau_o += _load(-Da, 0.0, 0.0, 0.0, 0.0, self.num.z_air)
        # Froude-Krylov force of the wave slopes, surge and sway
        if self.sw.transom:
            # S2a: the longitudinal hydrostatic + Froude-Krylov force of the
            # hull (telescoped) and of the wetted transom face, and their
            # pitch moment; it replaces Xw (a second copy) and the
            # hydrostatic part of the tilted strip force (Xp below)
            if self.sw.orbital and kb.wv:
                Xh, Mh = self._x_hydro(d, A, s_lon, ab, kb.urT, Th, self._trs)
            else:
                Xh, Mh = self._x_hydro(d, A, s_lon, ab, u, Th, self._trs)
            if self.sw.heel_hydro and hh is not None:
                # S5a: the same over the heeled stations, transom section and
                # face, with the yaw moment of the x-facing surfaces
                Xh, Mh, Nh = self._x_hydro_heel(
                    hh, s_lon, ab, kb.urT if (self.sw.orbital and kb.wv)
                    else u, Th, self._trs)
                tau_o[5] += Nh
            if self.sw.fk_exact and kb.wv:
                # P1b: the same of the exact pressure beyond the long-wave
                # field (sections, transom section and face; a_b on it)
                Xh = Xh + fk.x
                Mh = Mh + fk.m
            tau_o[4] += Mh
            # P1a normal_rate: the turning of the strip normal, -m_a V
            # Dtau/Dt with Dtau/Dt = SP q - u kpp (module docstring)
            if self.sw.orbital and kb.wv:
                Xn = (-(ma * V * (SP * q - kb.ur * self.kpp)).sum() * dx
                      if self.sw.normal_rate else 0.0)
            else:
                Xn = (-(ma * V * (SP * q - u * self.kpp)).sum() * dx
                      if self.sw.normal_rate else 0.0)
            self._trs["Xn"] = float(Xn)
            if self.sw.fk_exact and kb.wv:
                # P1b: the lateral force of the exact incident pressure (no
                # a_b), and the strip-exact moments of the vertical and
                # lateral parts
                Ysec = fk.fy
                tau_o[4] += fk.my_sig
                tau_o[5] += fk.n_sig
            else:
                Ysec = -rho * G * ab * A * s_tr
            if self.sw.heel_hydro and hh is not None:
                # S5a: the lateral force of the tilted hydrostatic field on
                # the heeled polygons, plus the incident pressure beyond its
                # long-wave part; no a_b (P1c's convention, S5a review)
                Ysec, dN = self._hh_lateral(
                    hh, ab, ctr, d, A, fk if (self.sw.fk_exact and kb.wv)
                    else None)
                tau_o[5] += dN
        else:
            Xw = -(rho * G * ab * A * s_lon).sum() * dx
            Ysec = -rho * G * ab * A * s_tr
            tau_o[0] += Xw
        tau_o[1] += Ysec.sum() * dx
        tau_o[5] += (x * Ysec).sum() * dx
        # sway / yaw: slender-body (Munk) forces of the wetted hull, with
        # m22 = pi/2 rho d^2 per unit length (a plate of draught d and its
        # free-surface image). Per unit length f = U d(m22 v_l)/dx, which
        # integrates to a side force at the transom, Y = -U m22_T v_T, and a
        # moment N = x_T Y - U int m22 v_l dx. The integral is the Munk
        # moment; without it all the hull's side force sat at the transom,
        # on top of the jet, and cancelled the jet's turning moment -- the
        # boat turned at 2 deg/s at 34 kn with the nozzle hard over.
        if self.sw.lateral_cross:
            # S5a: the complete 2D phi = 0 cross-plane impulse dynamics in
            # the level frame (module docstring 'S5a'): one kinetic energy
            # with the heave equation's m33 (ma) and its d-derivative, the
            # wave velocities' rates at fixed body x (S5a review)
            lat = self._lateral_x(s, d, kb, tau, ett, s_tr, ma,
                                  self._m33_dd(d, kb, dma, ctr),
                                  self._wave_rates_b(s, t, d, A, kb))
            tau_o += lat.tau
        elif self.sw.lateral_phi0:
            # P1b: the lateral flow on the phi = 0 basis of the vertical one
            # at the instantaneous immersion and heel (module docstring);
            # S5a review: the orbital velocity's rate at fixed body x
            lat = self._lateral_b(s, d, kb, self._wave_rates_b(s, t, d, A, kb)
                                  if (self.sw.orbital and kb.wv) else None)
            tau_o += lat.tau
        else:
            m22 = 0.5 * np.pi * rho * np.maximum(d, 0.0) ** 2
            vl = v + x * r
            if self.sw.orbital and kb.wv:
                # P1b: the relative lateral velocity and inflow, and the
                # lateral diffraction (G. I. Taylor): the section's lateral
                # impulse a22 (v_l - v_orb) + a24 p under -D/Dt gives the
                # local inertia + a22 a_oy, its yaw lever x and the roll
                # moment + a24 a_oy (about the CG), with the exact
                # gravity-free 2D section values of the upright section at
                # its immersion (hydro/section2d: lateral_phi0's basis and
                # the vertical's). v0's plate m22 = pi/2 rho d^2 (its sum
                # 804 kg on scarab195 at rest, 3D zero-frequency 467 kg)
                # made the long-wave sway excitation 19-35 % too large
                # (module docstring 'P1b orbital'; register A2). S5a review
                # (issue C): a_oy the orbital velocity's rate at fixed body
                # x (the transom lift and the Munk sum are the conservation
                # form's end and convective terms; _wave_rates_b)
                vl = vl - kb.vo
                Ylift = -abs(kb.ur[0]) * m22[0] * vl[0]
                (a22d, a24d, _), _, _ = self._lat_coeffs(d, 0.0)
                aoyb = self._wave_rates_b(s, t, d, A, kb).aoy
                Ydf = a22d * aoyb
                tau_o[1] += Ylift + Ydf.sum() * dx
                tau_o[5] += x[0] * Ylift - (np.abs(kb.ur) * m22
                                            * vl).sum() * dx \
                    + (x * Ydf).sum() * dx
                tau_o[3] += (a24d * aoyb).sum() * dx
            else:
                Ylift = -abs(u) * m22[0] * vl[0]
                tau_o[1] += Ylift
                tau_o[5] += x[0] * Ylift - abs(u) * (m22 * vl).sum() * dx
            Ycf = -0.5 * rho * self.cd_lat * np.maximum(d, 0.0) * vl * np.abs(vl)
            tau_o[1] += Ycf.sum() * dx
            tau_o[5] += (x * Ycf).sum() * dx
        # roll: static stability about the transverse wave slope
        wbar = (A * s_tr).sum() / max(A.sum(), 1e-12)
        wn_r = np.sqrt(max(self.m * G * self.GM, 1e-6) / self.Ix)
        if self.sw.heel_hydro and hh is not None:
            # S5a: the roll moment of the hydrostatic pressure on the heeled
            # sections, the incident pressure beyond its long-wave part (no
            # a_b: P1c's convention, S5a review), and v0's damping
            # placeholder (E6) -- both plants
            tau_o[3] += self._hh_roll(
                hh, ab, ctr, d, A, Th, p, wn_r,
                fk if (self.sw.fk_exact and kb.wv) else None)
        elif self.sw.roll_own and self._roll_own():
            # P1b: the restoring computed here for both plants (the GZ
            # twin's swap is bypassed: _ASMixin.deriv), plus the exact wave
            # roll moment beyond its long-wave part
            tau_o[3] += self._roll_b(phi, wbar, p, wn_r,
                                     fk if (self.sw.fk_exact and kb.wv)
                                     else None)
        else:
            tau_o[3] += (-self.m * G * self.GM * (phi - wbar)
                         - 2.0 * self.hull.roll_zeta * wn_r * self.Ix * p)
        # the jet, or in captive runs a force exactly holding the speed
        if self.captive_u is None:
            fx, fy = self.prop.forces(thr, rud, u)
            self._jet_f = (fx, fy)
        else:
            fx, fy = 0.0, 0.0
        xn, _, zn = self.thrusters[0]
        # rigid-body Coriolis of the yaw-rotating frame
        tau_o[0] += self.m * v * r
        tau_o[1] -= self.m * u * r

        # ---- heave and pitch, instantaneous added mass on the left
        A33 = ma.sum() * dx
        A35 = (ma * SP * x).sum() * dx
        A55 = (ma * x * x).sum() * dx
        Fz = rest.sum() * dx - self.m * G
        My = (SP * x * rest).sum() * dx
        if self.captive_u is not None:
            # the force that holds the speed: everything else, at the nozzle
            if self.sw.transom:
                pres = -(dyn * tau).sum() * dx + Xh + Xn     # S2a
                if self.sw.wagner_entry and hx is not None:
                    # P1b: the fixed-waterline planes' horizontal force in
                    # momentum form (_flux_b)
                    pres = pres - (hx * tau).sum() * dx
                fx = Df + Da - pres
            else:
                pres = -(rest * tau).sum() * dx      # first guess, no accel
                fx = Df + Da - Xw - pres
        jet = _load(fx, fy, 0.0, xn, 0.0, zn)
        tau_o += jet
        m11, m12, m22_ = self.m + A33, A35, self.Iy + A55
        r1, r2 = Fz + tau_o[2], My + tau_o[4]
        det = m11 * m22_ - m12 * m12
        wd = (r1 * m22_ - m12 * r2) / det
        qd = (m11 * r2 - m12 * r1) / det
        if self.sw.lateral_cross:
            # S5a: sway, heave, roll, pitch and yaw together (a23, a34 couple
            # the planes at heel)
            vd, wd, pd, qd, rd = self._solve5(lat, m11, m12, m22_, r1, r2,
                                              tau_o)
        f = rest - ma * (wd + SP * x * qd)           # full strip force
        if self.sw.transom:
            # S2a: the dynamic strip force tilted as v0's, the hydrostatic
            # part from the pressure integral
            Xp = -((f - buoy) * tau).sum() * dx + Xh + Xn
            if self.sw.wagner_entry and hx is not None:
                # P1b: the fixed-waterline planes' horizontal force in
                # momentum form, -D/Dt(m_0 V tau) (_flux_b)
                Xp = Xp - (hx * tau).sum() * dx
            if self.sw.lateral_cross:
                # S5a: the surge force of the cross-plane vertical impulse,
                # which is normal to the bottom (P_x = -P_z tau): -D/Dt(P_x)
                # summed, d/dt|_x (P_z tau) over the strips plus u [P_x] at
                # the ends (the shedding at the transom). d/dt|_x P_z is the
                # strip's vertical force without its Lagrangian reaction:
                # the strips are body-fixed, T~ does not depend on the
                # horizontal position, so the reaction dT~/dz is not tilted
                # (tilting it made a rolling hull at rest drift: check H4;
                # wagner_entry's hx, P1b review, is the same point)
                fzc = lat.fz - lat.e23 * (vd + x * rd) - lat.e34 * pd
                Xp = (Xp - ((fzc - lat.Rz) * tau).sum() * dx
                      - lat.Zsh * tau[0]
                      + (lat.Pz * (SP * q)).sum() * dx)
        else:
            Xp = -(f * tau).sum() * dx               # pressure drag
        # ---- surge, sway, roll, yaw
        ud = 0.0 if self.captive_u is not None else \
            (tau_o[0] + Xp) / self.Mtot[0, 0]
        if self.sw.lateral_cross:
            pass                                    # S5a: solved above
        elif self.sw.lateral_phi0:
            # P1b: sway, roll and yaw with the instantaneous added-mass
            # matrix, together
            vd, pd, rd = self._lat_solve(lat, tau_o)
        else:
            vd = tau_o[1] / self.Mtot[1, 1]
            pd = tau_o[3] / self.Mtot[3, 3]
            rd = tau_o[5] / self.Mtot[5, 5]
        ds = np.zeros_like(s)
        ds[0] = u * cps - v * sps
        ds[1] = u * sps + v * cps
        ds[2], ds[3], ds[4], ds[5] = w, p, q, r
        ds[6:12] = ud, vd, wd, pd, qd, rd
        aux = dict(d=d, V=V, mom=mom, wd=wd, qd=qd, lw=lw, S=S, Df=Df,
                   Xp=Xp, Da=Da, fx=fx)
        aux["mom_full"] = flux
        if self.sw.transom:
            aux.update(Xh=Xh, Mh=Mh, **self._trs)
        if self.sw.p1b:
            # P1b: what the checks read (studies/anyspeed_p1b_check.py): the
            # kinematics, the right-hand sides of the lateral equations (the
            # accelerations not yet solved for), the pressure forces
            aux.update(kb=kb, lat_rhs=tau_o[[1, 3, 5]].copy(),
                       Fz_p=float(buoy.sum() * dx),
                       My_p=float((SP * x * buoy).sum() * dx),
                       Ysec_sum=float(Ysec.sum() * dx),
                       Nsec_sum=float((x * Ysec).sum() * dx))
            if self.sw.wagner_entry:
                aux.update(hx=hx)
            if self.sw.fk_exact and kb.wv:
                aux.update(fk=fk)
        if self.sw.fast_path:
            # P1c: what the GZ plant's roll swap reads (_ASMixin.deriv),
            # not recomputed there
            aux.update(A_sec=A, wbar=wbar)
        if self.sw.heel_hydro:
            # S5a: the branch taken (the GZ plant's roll swap on the upright
            # one, _ASMixin.deriv) and what the checks read
            aux.update(heeled=hh is not None, wbar=wbar, hh=hh)
            if self.sw.lateral_cross:
                aux.update(lat=lat)
        return ds, aux

    # ------------------------------------------------ S1: complete D/Dt
    def _flux_dDt(self, d, V, u, mom, slope):
        """Momentum flux of the sections with the complete material
        derivative of the added mass (switch dDt_full, design 2.5). mom is
        v0's flux (dm_a/dd V^2 on entry), slope = dd/dx at the stations
        (s_lon - tau: the wave slope less the local bottom slope).

        Watch one earth-fixed plane as the hull passes: its body station x
        moves aft at the speed u, so D/Dt = d/dt|_x - u d/dx|_t and

            Dm_a/Dt = (dm_a/dd) Dd/Dt - u (dm_a/dx)|_d.

        v0 keeps the first term only. m_a = k_a pi/2 rho c^2 with
        c = min(pi/2 d / tan(beta), yc(x)); below the cap (Wagner half-beam
        inside the chine) m_a does not depend on x at fixed d; where the
        section is chine-wetted it is k_a pi/2 rho yc(x)^2, which falls
        towards the stem over the forebody taper: a passing plane sees the
        wetted beam widen. Without the second term the strip sum carries a
        spurious end flux -u m_a(x0) V(x0) at the start of the taper (x0,
        ~1.3 m forward of the CG) -- the negative pitch damping and the
        low-speed limit cycle of v0 (design 1, R3). With it, where the
        whole taper is chine-wetted and entering, the integral of
        D/Dt(m_a V) is d/dt int m_a V dx + u m_aT V_T: the transom flux.

        WHERE CHINE WETTING STARTS, INSIDE THE STRIP. v0 decides chine
        wetting at each station's centre. With this term a taper strip's
        flux would then jump, as its centre caps, from dm_a/dd V^2 to
        u V |dm_a/dx| -- several times larger -- and the boat would settle
        on that threshold instead of on an equilibrium (sliding states,
        station-count-dependent attitude; review of S1). So over each taper
        strip d is taken linear (the centre value and slope), the chine
        analytic, and the cap point(s) located exactly; then per strip,
        per unit length,

            capped part    u V k_a pi/2 rho [yc^2(aft end) - yc^2(fwd end)]
            uncapped part  V^2 int dm_a/dd dx  (dm_a/dd = k_a pi rho kc^2 d)

        divided by the strip length. Both are continuous in the state, and
        the capped part sums exactly along the taper. A strip with no
        chine-wetted point keeps v0's centre value bit for bit, and with no
        located strip chine-wetted the plant is v0 exactly. Which strips are
        located (_taper): those whose chine varies along them (the taper; a
        transom run of the geometry), and with switch chine_exact_all
        (S1b) every strip -- without it a prismatic strip keeps v0's
        centre-decided cap, whose flux drops from dm_a/dd V^2 to zero in
        one step as the centre caps (a jump of ~750 N on scarab195 at
        12 kn, studies/anyspeed_p0_check.py part cap). Since kc d+ - yc is
        convex along a strip when the chine is concave, a strip has a
        chine-wetted point iff one of its two ends has; for a chine that is
        not concave (an offsets table) the interior knots are tested too.

        THE CHINE COMES FROM THE GEOMETRY (P0). The chine function and the
        roots of kc d = yc of each of its pieces are hydro/hull_geometry's
        (Stations.chine, Stations.roots): for v0's chine they are S1's own
        expressions, so the preset's runs did not change (golden); a
        transom run adds a linear piece, an offsets table one linear piece
        per interval. studies/anyspeed_p0_check.py part cap compares
        _substrip with an independent reference (dense sampling + brentq)
        on v0's chine, planing_b's and a non-concave offsets chine.

        ENTRY ONLY. The flux counts while V > 0 and Dm_a/Dt > 0: Zarnick's
        entry-only rule (pv:277, V > 0), applied to the complete D/Dt. With
        V > 0 and u >= 0 both parts are >= 0. Counting it while V < 0 (a
        section rising out of a chine-wetted taper) would be a suction
        growing with bow-down trim -- the as-built gate 'Dm_a/Dt > 0' did
        exactly that and made new sunk bow-down equilibria and nose-dives
        in following seas; v0 applies no exit flux anywhere.

        Dd/Dt is the geometric immersion rate. Here that is V itself (the
        plant's V is d d/dt along the plane); when S3 puts orbital
        velocities into V, this must stay the geometric rate.

        Astern (u < 0) the transom is the leading edge and m_a jumps from 0
        to m_aT as it passes a plane -- an entry delta this one-sided form
        does not hold. Without switch transom the u term is taken with
        max(u, 0) (continuous through u = 0) and only the dm_a/dd part
        counts. With it (S2a, _flux_astern) the u term is the complete
        u km [yc^2(aft) - yc^2(fwd)] for either sign of u -- astern the
        narrowing taper is where a plane LOSES added mass, so there it
        lowers Dm_a/Dt and the entry gate drops it -- and the leading
        transom adds its entry delta -u m_aT to the transom strip."""
        flux = mom.copy()
        act, ycap, iunc = self._substrip(d, slope)
        self._taper_cap = act        # stations whose strip is chine-wetted
        astern = self.sw.transom and u < 0.0
        if act.size:
            Vj = V[act]
            km = self.k_a * 0.5 * np.pi * self.rho
            kc = 0.5 * np.pi / self.tanb
            uu = u if astern else max(u, 0.0)
            Dma = (Vj * (2.0 * km * kc * kc) * iunc
                   + uu * km * ycap) / self.dx
            flux[act] = np.where((Dma > 0.0) & (Vj > 0.0), Vj * Dma, 0.0)
        if astern:
            flux = self._flux_astern(d, V, u, act, ycap, iunc, flux)
        return flux

    # ------------------------------------------- S2a: the transom (P1a)
    def _flux_astern(self, d, V, u, act, ycap, iunc, flux):
        """Astern (u < 0, switch transom): the transom strip's flux with the
        leading-edge entry delta. A plane fixed in the water meets the hull
        at the transom first; there m_a jumps from 0 to m_aT = k_a pi/2 rho
        c_T^2 (c_T = min(kc d_T, yc_T), the transom section), so
        Dm_a/Dt = -u m_aT delta(x - x_T) > 0 -- per unit length of the
        transom strip -u m_aT / dx -- added to the strip's own rate (its
        located form if its strip is chine-wetted, else v0's dm_a/dd V);
        the flux V Dm_a/Dt counts on entry only (V > 0 and Dm_a/Dt > 0),
        as everywhere. Going ahead the transom is the trailing edge, where
        the plane loses m_aT into the wake: an exit, never counted (S1)."""
        tr = self._trs
        tg = self._transom_geom()
        km = self.k_a * 0.5 * np.pi * self.rho
        kc = 0.5 * np.pi / self.tanb
        V0 = V[0]
        hit = np.nonzero(act == 0)[0]
        if hit.size:
            j = hit[0]
            D0 = (V0 * (2.0 * km * kc * kc) * iunc[j]
                  + u * km * ycap[j]) / self.dx
        else:
            d0 = max(d[0], 0.0)
            cw = kc * d0
            D0 = (self.k_a * np.pi * self.rho * cw * kc * V0
                  if (cw < self.yc[0] and d0 > 0.0) else 0.0)
        cT = min(kc * max(tr["dT"], 0.0), tg.ycT)
        D0 = D0 - u * km * cT * cT / self.dx
        out = flux.copy()
        out[0] = V0 * D0 if (D0 > 0.0 and V0 > 0.0) else 0.0
        return out

    def _transom_geom(self):
        """The transom's fixed geometry, once per plant: its body x (the aft
        strip edge), keel height zk_T, chine half-beam yc_T (the geometry's
        chine at x = 0, not the first strip centre's), the static transom
        draught T_T0 (keel depth at the transom floating at rest, the
        variable of the ventilation data) and the closure (hydro/
        vessel_spec.transom_closure_of)."""
        tg = self.__dict__.get("_trg")
        if tg is None:
            g, h = self._geom, self.hull
            xe = np.array([g.edges[0]])
            tg = SimpleNamespace(
                xT=float(g.edges[0] - h.lcg),
                zkT=float(-h.vcg + g.keel(xe)[0][0]),
                ycT=float(g.chine(xe)[0]),
                cl=_vs.transom_closure_of(h))
            tg.TT0 = float(-(self.Z_rest + tg.zkT
                             + SP * tg.xT * self.Th_rest))
            self._trg = tg
        return tg

    def _transom_kin(self, x0, y0, psi, z, Th, u, t):
        """The transom's keel depth below the local surface, d_T (the
        incident wave's elevation evaluated at the transom itself, v0's
        phase sum written for one point -- not through _waves, whose GZ
        recording override must see the stations once), its vertical offset
        from the CG z0_T, and the
        wetted height h_w of the face (the transom flow state; module
        docstring 'S2a'). A dict (aux carries it)."""
        tg = self._transom_geom()
        cl = tg.cl
        if getattr(self, "_wa", None) is None:
            eT = 0.0
        else:
            # the elevation alone, at one point: v0's phase sum
            # (PlaningVessel._waves) without its rates and slopes
            XT = x0 + tg.xT * math.cos(psi)
            YT = y0 + tg.xT * math.sin(psi)
            ph = XT * self._wkx + YT * self._wky - self._ww * t + self._wphi
            eT = float(np.cos(ph) @ self._wa[0])
        z0T = tg.zkT + SP * tg.xT * Th
        dT = eT - (self.Z_rest + z) - z0T
        up = max(u, 0.0)
        if cl.mode == "progressive":
            # the dead-water level: a constant base-pressure head below the
            # local surface, U^2 / (g F_T_crit^2)
            head = up * up / (G * cl.F_T_crit * cl.F_T_crit)
            hw = dT - head
        elif cl.mode == "progressive_static":
            # sensitivity only (hydro/vessel_spec): the same progression with
            # the threshold on the STATIC transom draught (the sources'
            # variable): h_w = d_T (1 - U^2 / (g F_T_crit^2 T_T0))
            head = dT * up * up / (G * cl.F_T_crit * cl.F_T_crit
                                   * max(tg.TT0, 1e-12))
            hw = dT - head
        else:
            # 'step' (sensitivity only): wet below the threshold on the
            # static draught
            wet = up < cl.F_T_crit * math.sqrt(G * max(tg.TT0, 1e-12))
            head = 0.0 if wet else np.inf
            hw = dT if wet else 0.0
        return dict(dT=float(dT), z0T=float(z0T), hw=float(max(hw, 0.0)),
                    head=float(head))

    def _x_hydro(self, d, A, s_lon, ab, u, Th, tr):
        """S2a: the longitudinal force and pitch moment of the plant's
        hydrostatic + long-wave Froude-Krylov pressure p = a_b rho g (eta -
        z) over the hull and the wetted transom face, exactly (module
        docstring 'S2a'):

            X = -a_b [rho g sum_j A_j eta_x,j dx + Q_T] + a_b Q_w + D_face
            M = -a_b [rho g sum_j eta_x,j (z0_j A_j + m1_j) dx + R_T]
                + a_b R_w + z_A D_face

        Q, R: the face's pressure force and its moment about the CG for the
        immersion d_T (hull side) and for the wetted height h_w (face side);
        D_face the stagnation drag of the leading face astern. Returns
        (X, M); the parts go into tr (aux)."""
        tg = self._transom_geom()
        rg = self.rho * G
        tb = self.tanb
        if getattr(self, "_wa", None) is None:
            sx = sm = 0.0                       # calm water: eta_x = 0
        else:
            m1 = _vsec(d, self.yc, tb)[1]
            z0 = self.zk + SP * self.x * Th
            sx = (A * s_lon).sum() * self.dx
            sm = (s_lon * (z0 * A + m1)).sum() * self.dx
        z0T = tr["z0T"]
        qT, sT, AT, m1T = _vsec_qs(tr["dT"], tg.ycT, tb)
        QT, RT = rg * qT, rg * (z0T * qT + sT)
        qw, sw_ = _vsec_qs(tr["hw"], tg.ycT, tb)[:2]
        Qw, Rw = rg * qw, rg * (z0T * qw + sw_)
        if u < 0.0 and AT > 0.0:
            Dface = 0.5 * self.rho * tg.cl.cd_face * AT * u * u
            zA = z0T + m1T / AT
        else:
            Dface, zA = 0.0, 0.0
        X = -ab * (rg * sx + QT) + ab * Qw + Dface
        M = -ab * (rg * sm + RT) + ab * Rw + zA * Dface
        tr.update(QT=float(ab * QT), Qw=float(ab * Qw), RT=float(ab * RT),
                  Rw=float(ab * Rw), Dface=float(Dface),
                  Xfk=float(-ab * rg * sx), Mfk=float(-ab * rg * sm))
        return X, M

    def _taper(self):
        """The strips whose chine wetting is located exactly (_substrip):
        those where the chine varies along the strip -- the forebody taper
        (fwd end ahead of its start x0) and a transom run, from the
        geometry (hydro/hull_geometry.Stations.varying) -- or every strip
        with chine_exact_all. Indices, ends, centres, the chine at the ends.
        Once per plant; the geometry's chine function must reproduce the
        station chine self.yc bit for bit."""
        tg = self.__dict__.get("_tg")
        if tg is None:
            g = self._geom
            tg = SimpleNamespace(g=g, kc=0.5 * np.pi / self.tanb)
            if not np.array_equal(g.chine(g.xi), self.yc):
                raise RuntimeError("the geometry's chine function no longer "
                                   "reproduces the station chine")
            if self.sw.chine_exact_all:
                idx = np.arange(g.n)                            # S1b
            else:
                idx = np.nonzero(g.varying)[0]
            tg.idx = idx
            tg.xa, tg.xb, tg.xc = g.edges[idx], g.edges[idx + 1], g.xi[idx]
            tg.h2 = 0.5 * (tg.xb - tg.xa)
            tg.yca, tg.ycb = g.chine(tg.xa), g.chine(tg.xb)
            # strips of constant chine (prismatic): there kc d+ - yc is
            # linear in x wherever d > 0, so a strip capped at both ends is
            # capped throughout (_substrip's shortcut)
            tg.flat = ~np.asarray(g.varying, bool)[idx]
            # interior chine knots inside each strip, for a chine that is
            # not concave (an offsets table): kc d+ - yc is then not convex
            # along a strip and its ends alone do not decide (module
            # hydro/hull_geometry, 'CHINE CONCAVITY')
            tg.knots = None
            if not g.concave and g.knots.size:
                kn = [g.knots[(g.knots > a) & (g.knots < b)]
                      for a, b in zip(tg.xa, tg.xb)]
                w = max([k.size for k in kn] + [0])
                if w:
                    K = np.full((idx.size, w), np.nan)
                    for i, k in enumerate(kn):
                        K[i, :k.size] = k
                    tg.knots = K
                    tg.knot_yc = np.where(np.isfinite(K),
                                          g.chine(np.nan_to_num(K)), np.inf)
            self._tg = tg
        return tg

    def _substrip(self, d, slope):
        """Chine wetting inside the strips of _taper, exactly for d linear
        over each strip: (station indices of the strips with a chine-wetted
        part, sum over each one's chine-wetted pieces of yc^2(aft) -
        yc^2(fwd), integral of d+ over its other pieces). The chine and the
        roots of kc d = yc of each of its pieces come from the geometry
        (hydro/hull_geometry): for v0's chine, S1's expressions."""
        tg = self._taper()
        g = tg.g
        kc = tg.kc
        di, mi = d[tg.idx], slope[tg.idx]
        h2 = tg.h2
        ca = kc * np.maximum(di - mi * h2, 0.0) >= tg.yca
        cb = kc * np.maximum(di + mi * h2, 0.0) >= tg.ycb
        on = ca | cb
        if tg.knots is not None:
            dk = di[:, None] + mi[:, None] * (tg.knots - tg.xc[:, None])
            on = on | (kc * np.maximum(np.nan_to_num(dk), 0.0)
                       >= tg.knot_yc).any(axis=1)
        if not on.any():
            return tg.idx[:0], np.zeros(0), np.zeros(0)
        # a strip of constant chine capped at both ends is capped throughout
        # (kc d+ - yc linear there): yc^2(aft) - yc^2(fwd) = 0 and nothing
        # uncapped -- what the pieces below give, without the breakpoints;
        # only the other chine-wetted strips need the roots
        full = tg.flat & ca & cb
        gen = on & ~full
        n_on = int(on.sum())
        ycap_on, iunc_on = np.zeros(n_on), np.zeros(n_on)
        if not gen.any():
            return tg.idx[on], ycap_on, iunc_on
        sel = gen[on]
        di, mi = di[gen], mi[gen]
        xa, xb, xc = tg.xa[gen], tg.xb[gen], tg.xc[gen]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            # every piece's roots of kc d = yc: extra breakpoints are
            # harmless (each sub-piece is classified at its midpoint), a
            # missed one would not be
            roots = g.roots(xc, di, mi, kc)
        P = np.stack([xa, xb] + list(roots), axis=1)
        if tg.knots is not None:
            P = np.concatenate([P, tg.knots[gen]], axis=1)
        P = np.where(np.isfinite(P), P, xb[:, None])
        P = np.clip(P, xa[:, None], xb[:, None])
        P.sort(axis=1)
        lo, hi = P[:, :-1], P[:, 1:]
        ln = hi - lo
        dc, mc = di[:, None], mi[:, None]
        xcc = xc[:, None]
        mid = 0.5 * (lo + hi)
        dmid = dc + mc * (mid - xcc)
        # the chine at the midpoints and both ends in one evaluation (P1c;
        # elementwise, the same numbers)
        k_ = lo.shape[1]
        ych = g.chine(np.concatenate([mid, lo, hi], axis=1))
        cap = ((ln > 0.0) & (dmid > 0.0) & (kc * dmid >= ych[:, :k_]))
        ylo, yhi = ych[:, k_:2 * k_], ych[:, 2 * k_:]
        ycap = np.where(cap, ylo * ylo - yhi * yhi, 0.0).sum(axis=1)
        dlo, dhi = dc + mc * (lo - xcc), dc + mc * (hi - xcc)
        plo, phi_ = np.maximum(dlo, 0.0), np.maximum(dhi, 0.0)
        mixed = (dlo > 0.0) != (dhi > 0.0)
        part = np.where(mixed, 0.5 * np.maximum(plo, phi_) ** 2 * ln
                        / np.where(mixed, np.abs(dhi - dlo), 1.0),
                        0.5 * (plo + phi_) * ln)
        iunc = np.where(cap, 0.0, part).sum(axis=1)
        ycap_on[sel], iunc_on[sel] = ycap, iunc
        return tg.idx[on], ycap_on, iunc_on

    # ------------------------------------------- P1a: the integrator
    def step(self, s, t, thrust_cmd, rudder_cmd, dt=None):
        """v0's step (sim/planing_vessel.PlaningVessel.step, itself) with
        switch verlet off; with it the same step whose second stage is
        evaluated at the step's end positions (_step_verlet)."""
        if self.sw.verlet:
            return self._step_verlet(s, t, thrust_cmd, rudder_cmd, dt)
        return super().step(s, t, thrust_cmd, rudder_cmd, dt)

    def _step_verlet(self, s, t, thrust_cmd, rudder_cmd, dt=None):
        """PlaningVessel.step line for line -- prime update, the derivative
        at the start (the measurements' source), v0's substep logic on the
        lateral rate, the actuators advanced after, the same measurements --
        with each (sub)step taken by _sv instead of Heun."""
        dt = self.dt if dt is None else dt
        self.prop.update_prime(self.prop_submergence(s[:6], t), dt)
        k1, aux = self.deriv(s, t)
        n = 1
        lam = self._lateral_rate(s, aux["d"]) * dt
        if lam > _pv.STIFF_STEP:
            n = min(int(np.ceil(lam / _pv.STIFF_SUB)), _pv.N_SUB_MAX)
        h = dt / n
        s_new, k = s, k1
        for i in range(n):
            if i:
                k, _ = self.deriv(s_new, t + i * h)
            s_new = self._sv(s_new, k, t + (i + 1) * h, h)
        self.last_n_sub = n
        s_new[12] = self.prop.advance(s[12], thrust_cmd, dt)
        s_new[13] = self.rudder.step(s[13], rudder_cmd, dt)
        # measurements, at the start of the step
        self.last_jet_force = self._jet_f
        self.last_cg_acc = float(aux["wd"])
        self.last_bow_acc = float(aux["wd"] + SP * self.x[-1] * aux["qd"])
        slam = float(aux["mom"].sum() * self.dx)
        self.last_slam_force = slam
        self.peak_slam_force = max(self.peak_slam_force, slam)
        d_b, V_b = float(aux["d"][-1]), float(aux["V"][-1])
        self.last_rel_bow = d_b + float(self.sec.keel[-1])
        bow_out = d_b <= 0.0
        if self._emerged and not bow_out and V_b > self.v_slam:
            self.slam_count += 1
        self._emerged = bow_out
        return s_new

    @staticmethod
    def _kin(s):
        """The position rates of state s exactly as deriv writes them
        (ds[0:6]): earth-frame surge and sway velocity, w, p, q, r."""
        psi = s[5]
        u, v = s[6], s[7]
        cps, sps = np.cos(psi), np.sin(psi)
        return np.array([u * cps - v * sps, u * sps + v * cps, s[8], s[9],
                         s[10], s[11]])

    def _sv(self, s, k1, t1, h):
        """One (sub)step of length h from s with k1 = deriv(s) (switch
        verlet). Heun takes s_E = s + h k1 and

            s_new = s + h/2 (k1 + f(s_E)).

        Its positions are q_n + h v_n + h^2/2 a_n (for z, theta, phi) -- the
        velocity-Verlet positions -- but its second force evaluation is at
        the EULER positions q_n + h v_n. For a conservative oscillator that
        gives |R(i w h)|^2 = 1 + (w h)^4 / 4: energy grows (4.9 % per 100 s
        at 3.94 rad/s and 0.02 s; design review). Here the second stage is
        evaluated at the step's END positions -- Heun's own position update,
        q_new = q_n + h/2 (K(s) + K(s_E)), K the position rates -- with the
        Euler velocities v_E:

            a_2   = f_v(q_new, v_E)
            v_new = v_n + h/2 (a_n + a_2)

        For forces that depend on position only (hydrostatics, the
        restoring arm) this is the velocity-Verlet / Stormer-Verlet scheme:
        symplectic, no energy drift, |R(i w h)| = 1 for w h < 2 (the whole
        wave band at the plant step). Velocity-dependent forces (damping,
        the planing lift, Coriolis) see Heun's velocity v_E, as before:
        second order; for a force of velocity alone the step is Heun's, so
        v0's substep rule (lateral rate x step <= 1.8) still holds for the
        velocity-only lateral terms it guards. Mixed position and velocity
        dependence is neither: gyroscopic coupling is not symplectic here
        and the damped stability region differs from Heun's near w h -> 2
        (module docstring, 'THE INTEGRATOR'). Two force evaluations per
        step, as Heun."""
        sE = s + h * k1
        pos = s[:6] + 0.5 * h * (k1[:6] + self._kin(sE))
        s2 = sE.copy()
        s2[:6] = pos
        k2, _ = self.deriv(s2, t1)
        s_new = s + 0.5 * h * (k1 + k2)
        s_new[:6] = pos
        return s_new

    def _dyc_dx(self):
        """d yc / dx at the station centres, analytic, from the geometry --
        for diagnostics (studies/anyspeed_s1_check.py); the plant itself
        uses the chine at the strip ends and roots (_substrip)."""
        ycp = self.__dict__.get("_ycp")
        if ycp is None:
            g = self._geom
            ycp = g.dchine(g.xi)
            self._ycp = ycp
        return ycp

    # ------------------------------------------- P1b: waves, kinematics
    def _tab(self):
        """The phi = 0 section table of this hull's deadrise (hydro/
        section2d), per plant (loaded from the disk cache, built if
        missing)."""
        tab = self.__dict__.get("_sec2d")
        if tab is None:
            from hydro import section2d as _s2
            tab = _s2.table_for(float(self.hull.beta_deg))
            self._sec2d = tab
        return tab

    def _wave_setup_b(self):
        """Per sea (keyed by v0's own wave arrays object): amplitude,
        frequency, wavenumber, the unique wavenumbers (the depth factors are
        computed once per distinct k: a JONSWAP sea of n_freq x n_dir
        components has n_freq of them) and the phase constants."""
        wa = self._wa
        st = self.__dict__.get("_wbs")
        if st is not None and st.wa is wa:
            return st
        a = wa[0]
        w = self._ww
        kx, ky = self._wkx, self._wky
        k = np.sqrt(kx * kx + ky * ky)
        ku, kinv = np.unique(k, return_inverse=True)
        st = SimpleNamespace(wa=wa, a=a, w=w, kx=kx, ky=ky, k=k, ku=ku,
                             kinv=kinv, aw=a * w, aw2=a * w * w,
                             ksafe=np.where(k > 0.0, k, 1.0))
        # the v0 outputs as two matrix products: C @ [a, -a w^2],
        # S @ [a w, -a kx, -a ky]
        st.cc = np.stack([a, -a * w * w], 1)
        st.sc = np.stack([a * w, -a * kx, -a * ky], 1)
        self._wbs = st
        return st

    def _waves_b(self, X, Y, t, psi):
        """P1b (switch fk_exact): the component phases at the stations and
        at the transom point (one evaluation), the v0 quantities (e, et,
        ett, ex, ey at the stations) from them, and what the exact pressure
        and the orbital kinematics need, kept in self._wb (None in calm
        water): cos and sin of the phases and the body-frame wavenumbers
        kappa (along the hull) and lam (across). The stations are equally
        spaced along the hull, so along it each component's phase advances
        by kappa dx per station: e^{i psi_j} = e^{i psi_0} (e^{i kappa
        dx})^j, a running product (one complex exponential per component
        and per point, not per station: P1b review, E1); relative round-off
        ~ n eps. Unequal spacing (a geometry that has it) takes cos and sin
        of every phase."""
        if not hasattr(self, "_wa"):
            self._wave_arrays()
        n = X.size
        if self._wa is None:
            self._wb = None
            z = np.zeros(n)
            return z, z, z, z, z
        st = self._wave_setup_b()
        tg = self._transom_geom()
        cps, sps = math.cos(psi), math.sin(psi)
        kap = st.kx * cps + st.ky * sps
        lam = -st.kx * sps + st.ky * cps
        ph0 = X[0] * st.kx + Y[0] * st.ky - (st.w * t - self._wphi)
        if self.__dict__.get("_equal_dx") is None:
            self._equal_dx = bool(np.all(np.abs(np.diff(self.x) - self.dx)
                                         <= 1e-12 * self.dx))
        if self._equal_dx:
            Z = np.empty((n, st.k.size), complex)
            Z[0] = np.exp(1j * ph0)
            Z[1:] = np.exp(1j * (kap * self.dx))
            Z = np.cumprod(Z, axis=0)
            C, S = Z.real, Z.imag
        else:
            ph = (X[:, None] * st.kx + Y[:, None] * st.ky
                  - (st.w * t - self._wphi))
            C, S = np.cos(ph), np.sin(ph)
            Z = C + 1j * S
        phT = ph0 + (tg.xT - self.x[0]) * kap
        o1 = C @ st.cc
        o2 = S @ st.sc
        e, ett = o1[:, 0], o1[:, 1]
        et, ex, ey = o2[:, 0], o2[:, 1], o2[:, 2]
        # Z (the phases as e^{i psi}) for the row-stable kernel (P1c, sim/
        # as_kernels.fk_rows)
        self._wb = SimpleNamespace(st=st, C=C, S=S, Z=Z, CT=np.cos(phT),
                                   ST=np.sin(phT), kap=kap, lam=lam, e=e)
        return e, et, ett, ex, ey

    def _p1b_kin(self, s, t, d, A, tau, et, s_lon, s_tr, ma):
        """P1b kinematics of one derivative (a namespace, aux['kb']):

          ur   the relative surge inflow of each section, u - u_orb (switch
               orbital; u without it), urT at the transom station
          V    the relative normal velocity w_orb - (w + SP x q) + ur tau
               (the planing lift, the momentum flux's factor, the
               cross-flow drag); v0's V without orbital
          Vg   the geometric immersion rate along the plane, eta_t - (w + SP
               x q) + ur tau (the rate of change of the added mass: the
               contact line follows the surface, not the water at depth)
          vo, aox, aoy, aoz  the lateral orbital velocity and the orbital
               accelerations (body frame) at each section's centroid depth
          ddot the immersion rate at fixed body x (the lateral added mass's
               rate of change): eta_t + u s_lon + (v + x r) s_tr - (w + SP x
               q)
          ma   the added mass of the section's flow state (switch
               wagner_entry: m_Zarnick for a plane in an entry from dry,
               m_0 for one oscillating about a fixed waterline), with maZ,
               m0, dm0, dm0y, fixed, ds

        Orbital velocities: linear deep-water values at the depth of the
        immersed section's centroid below the local surface (z_c = -(d -
        m1/A), Wheeler-like: the decay is measured from the surface, never
        above 1), the centreline of the section; the accelerations are the
        earth-fixed partial rates d/dt at a fixed point. Along a plane that
        moves with the water (the vertical model's D/Dt, per plane) the
        material rate differs from them by u_o d/dx, second order in the
        amplitude, so the vertical model takes them as they are. At a FIXED
        BODY STATION the rate differs by u d/dxi + (v + x r) d/deta + ...,
        FIRST order (U / c): the lateral flows, whose convective part is
        in conservation form, take that rate (_wave_rates_b; S5a review,
        issue C -- this docstring called the convective part second order
        without saying which frame)."""
        u, v, w, q, r = s[6], s[7], s[8], s[10], s[11]
        x = self.x
        n = x.size
        kb = SimpleNamespace()
        V0 = et - (w + SP * x * q) + u * tau
        wb = self.__dict__.get("_wb") if self.sw.fk_exact else None
        # waves present (fk_exact evaluated them): in calm water the switches
        # fk_exact and orbital have nothing to add, and the derivative takes
        # P1a's lines (bit for bit; their calm-water standing checks are
        # P1a's)
        kb.wv = wb is not None
        kb.flux_b = self.sw.wagner_entry or (self.sw.orbital and kb.wv)
        if self.sw.orbital and wb is not None:
            st = wb.st
            dp = np.maximum(d, 0.0)
            m1 = _vsec(d, self.yc, self.tanb)[1]
            zc = np.where(A > 0.0, -(dp - m1 / np.where(A > 0.0, A, 1.0)),
                          0.0)
            Dk = np.exp(np.multiply.outer(zc, st.ku))[:, st.kinv]
            CD = wb.C * Dk
            SD = wb.S * Dk
            ct, sn = wb.kap / st.ksafe, wb.lam / st.ksafe
            oc = CD @ np.stack([st.aw * ct, st.aw * sn, -st.aw2], 1)
            os_ = SD @ np.stack([st.aw, st.aw2 * ct, st.aw2 * sn], 1)
            uo, vo, aoz = oc[:, 0], oc[:, 1], oc[:, 2]
            wo, aox, aoy = os_[:, 0], os_[:, 1], os_[:, 2]
            ur = u - uo
            kb.uo, kb.wo = uo, wo
            kb.vo, kb.aox, kb.aoy, kb.aoz = vo, aox, aoy, aoz
            kb.ur = ur
            kb.V = wo - (w + SP * x * q) + ur * tau
            kb.Vg = et - (w + SP * x * q) + ur * tau
        else:
            z = np.zeros(n)
            kb.uo = kb.wo = kb.vo = kb.aox = kb.aoy = kb.aoz = z
            kb.ur = np.full(n, float(u))
            kb.V = V0
            kb.Vg = V0
        kb.urT = float(kb.ur[0])
        kb.ddot = et + u * s_lon + (v + x * r) * s_tr - (w + SP * x * q)
        kb.maZ = ma
        if self.sw.wagner_entry:
            # the planes' flow states along the hull (module docstring 'P1b
            # WAGNER_ENTRY'): f the fraction of each strip whose planes
            # oscillate about a fixed waterline (state F), the rest in an
            # entry from dry (E) or in the exit after it (X), which keep S1's
            # Zarnick m_a and entry-only flux
            eg = self._es_geom()
            fld = self._es_now(s, t, d, kb.Vg, kb.ur, float(u), kb.ddot)
            b, st = fld[0], fld[1]
            f = np.where(d > 0.0, _es_frac(b, st, eg.edges, ES_F), 0.0)
            on = f > 0.0
            kb.es, kb.f, kb.fixed = (b, st), f, on
            m0, dm0, dm0y = np.zeros(n), np.zeros(n), np.zeros(n)
            if on.any():
                j = np.nonzero(on)[0]
                a0, a1, a2 = self._tab().vertical(d[j], self.yc[j])
                rho = self.rho
                m0[j], dm0[j], dm0y[j] = rho * a0, rho * a1, rho * a2
                kb.ma = np.where(on, (1.0 - f) * ma + f * m0, ma)
            else:
                kb.ma = ma
            kb.m0, kb.dm0, kb.dm0y = m0, dm0, dm0y
            kb.esy, kb.fX = None, None
            if len(fld) > 2:
                # P1c round 3 (plan '### wagner 低速'): an X plane has no
                # flux, so the momentum m V of its fluid changes by V Dm/Dt
                # beyond the force m DV/Dt -- which the wake can only take
                # away (Dm/Dt < 0), never supply: its added mass must not
                # grow along its path. In the widening taper the Zarnick
                # value of a chine-wetted section does grow (yc(x)), and
                # each exiting plane gained 1/2 c V^2 a unit time (c =
                # Dm/Dt), the input that sustained scarab195's 1-1.5 kn
                # oscillation with round 2's code (the Jacobi-form budget,
                # plan). So an X plane keeps k_a pi/2 rho min(kc d, yc(x),
                # Y)^2, Y the half-beam where it turned (equal to Zarnick's
                # there; kc d never grows while X: Vg <= 0). A modelling
                # assumption (S1's separated exit, register A8), and the
                # frozen Y pattern leaves a residual at X|X boundaries
                # (module docstring 'P1c round 4').
                kb.esy = fld[2]
                kc = 0.5 * np.pi / self.tanb
                cz = np.minimum(kc * np.maximum(d, 0.0), self.yc)
                fx, qx = _es_xmass(b, st, fld[2], eg.edges, cz)
                fx = np.where(d > 0.0, fx, 0.0)
                if fx.any():
                    mx = self.k_a * 0.5 * np.pi * self.rho * qx
                    kb.ma = kb.ma + fx * (mx - ma)
                kb.fX = fx
        else:
            kb.ma = ma
        return kb

    # ------------------------------------ P1b: the flow states of the planes
    def _waves_on(self):
        """Whether the plant's sea has waves (v0's wave arrays, built on
        first use); in calm water also clears the kept phases (self._wb),
        so that a calm copy of a wavy plant (the captive runs copy the
        plant's __dict__) sees none."""
        if not hasattr(self, "_wa"):
            self._wave_arrays()
        if self._wa is None:
            self._wb = None
            return False
        return True

    def _roll_own(self):
        """The roll moment computed in the copy (_roll_b) for this sea:
        always with lateral_phi0; with fk_exact where there are waves (in
        calm water the exact wave roll moment is zero, and the GZ twin keeps
        its own swap: P1a bit for bit)."""
        return self.sw.lateral_phi0 or (self.sw.fk_exact and self._waves_on())

    def _es_geom(self):
        """The strip edges in body x (from the transom edge lo to the stem
        edge hi), once per plant: the domain of the planes' state field."""
        eg = self.__dict__.get("_esg")
        if eg is None:
            e = np.asarray(self._geom.edges, float) - self.hull.lcg
            eg = SimpleNamespace(edges=e, lo=float(e[0]), hi=float(e[-1]))
            self._esg = eg
        return eg

    def _es_now(self, s, t, d, Vg, ur, u, dd=None):
        """The planes' state field (b, st) this derivative uses (module
        docstring 'P1b WAGNER_ENTRY'):
          - inside a step of the plant (_ASMixin.step sets self._es_run):
            at its first derivative (time t0) the field of the previous step
            advanced from its own time to t0 (exact transport of its
            boundaries at the relative inflow, then the transitions, in
            ES_NSUB sub-steps on interpolated kinematics: _es_advance), or,
            when the step does not continue the last one (no field yet, a
            state that is not the one the last step returned, or a time not
            after the field's), the steady field of this state (_es_scan);
            it is kept as the step's field. A later derivative of the step
            (time t > t0: the second stage, substeps) uses that field
            advanced by t - t0 with its own kinematics, so a transition
            inside a step enters both ends of the step's quadrature (the
            step's error from it is second order in the step and not
            systematically late, as a field held over the step would make
            it: P1b review, M5);
          - outside a step: for a plant of its own (free or captive) at the
            state its last step returned, the field the next step would
            start from (advanced from the field's time to t); otherwise (a
            first derivative, any other state) the steady field;
          - the steady-state copies (running_attitude, captive_deriv,
            calm_resistance build _StripAS copies, never stepped) use the
            steady field OF THE ATTITUDE (below).
        dd: the immersion rate at fixed body x (kb.ddot), d's time
        derivative, for the Hermite interpolation of d inside a step.
        P1c round 3: with the X planes' memory (es_xmem, default on) the
        field is (b, st, y), y the turn half-beams (_es_xmass)."""
        eg = self._es_geom()
        x = self.x
        own = isinstance(self, _ASMixin)
        nsub = self.__dict__.get("es_nsub", ES_NSUB)
        mem = self._es_mem()
        mkw = {} if mem is None else dict(ycf=mem[0], ytol=mem[1])
        run = self.__dict__.get("_es_run") if own else None
        if run is not None:
            if run["cur"] is None:
                prev = self.__dict__.get("_es_prev")
                tf = self.__dict__.get("_es_t")
                # the interval the field is advanced over: from its own
                # time (P1c review: the step's dt was used, wrong when the
                # step size changes); older records without it: the step's
                dtf = run["dt"] if tf is None else float(t) - tf
                k0 = self.__dict__.get("_es_kin")
                if run["mode"] == "advance" and dtf > 0.0:
                    run["cur"] = _es_advance(
                        prev[0], prev[1], x, d, Vg, ur, dtf, eg.edges,
                        ur0=self.__dict__.get("_es_ur"), k0=k0,
                        km=self.__dict__.get("_es_kinm"), dd=dd, nsub=nsub,
                        y=self._es_y(prev), **mkw)
                    run["km"] = k0
                else:
                    if run["mode"] == "advance":
                        # a step whose time is not after the field's: not a
                        # continuation (the steady field, counted)
                        self.es_resets += 1
                    run["cur"] = _es_scan(x, d, Vg, u, eg.edges, **mkw)
                    run["km"] = None
                run["t0"] = float(t)
                # the kinematics at the field's time: the start of the next
                # transport from it (Heun's start inflow, the interpolation
                # in time; P1c)
                run["kin"] = dict(t=float(t), d=np.array(d, float),
                                  vg=np.array(Vg, float),
                                  ur=np.array(ur, float),
                                  dd=None if dd is None
                                  else np.array(dd, float))
                return run["cur"]
            dte = float(t) - run["t0"]
            if dte > 0.0:
                cur = run["cur"]
                return _es_advance(cur[0], cur[1], x, d, Vg, ur, dte,
                                   eg.edges, ur0=run["kin"]["ur"],
                                   k0=run["kin"], km=run["km"], dd=dd,
                                   nsub=nsub, y=self._es_y(cur), **mkw)
            return run["cur"]
        if own:
            prev = self.__dict__.get("_es_prev")
            sref = self.__dict__.get("_es_s")
            if prev is not None and sref is not None and                     np.array_equal(s, sref):
                tf = self.__dict__.get("_es_t")
                dtf = self.dt if tf is None else float(t) - tf
                if dtf > 0.0:
                    return _es_advance(
                        prev[0], prev[1], x, d, Vg, ur, dtf, eg.edges,
                        ur0=self.__dict__.get("_es_ur"),
                        k0=self.__dict__.get("_es_kin"),
                        km=self.__dict__.get("_es_kinm"), dd=dd, nsub=nsub,
                        y=self._es_y(prev), **mkw)
        else:
            # P1c: the steady-state copies compute steady states (the
            # captive runs are a steady-state solver, the census its
            # equations): their field is the steady field OF THE ATTITUDE
            # -- the planes' classes in steady flow, where the hull has no
            # heave or pitch rate (Vg without -(w + SP x q)). The scan with
            # the iterate's transient rates found exit-then-re-entry (F)
            # planes in the pseudo-transient of a captive run from level,
            # which stopped within its tolerance at a point slightly off
            # P1a's (P1b review, W6 / 8b: 1.3e-7); at w = q = 0 (the census,
            # the equilibria) this is the same field bit for bit. Keyed on
            # the copy's class (P1c review): a captive plant of its own
            # (S1 part b's runs) is a dynamic plant and continues its field
            # like a free one.
            Vg = Vg + (s[8] + SP * x * s[10])
        return _es_scan(x, d, Vg, u, eg.edges, **mkw)

    def _es_mem(self):
        """P1c round 3: (ycf, ytol) of the X planes' memory -- the chine
        half-beam at body x (the geometry's own function) and the merging
        bin ES_YTOL_REL x the largest half-beam -- or None when it is off
        (plant attribute es_xmem = False: P1c round 2's X, checks only)."""
        if not self.__dict__.get("es_xmem", ES_XMEM):
            return None
        m = self.__dict__.get("_esm")
        if m is None:
            g, lcg = self._geom, float(self.hull.lcg)

            def ycf(xb):
                return g.chine(np.asarray(xb, float) + lcg)
            m = self._esm = (ycf, ES_YTOL_REL * float(np.max(self.yc)))
        return m

    def _es_y(self, fld):
        """The y of a field when the memory is on (None when off); a field
        without one (a record made before P1c round 3) gets +inf: its X
        planes keep Zarnick's value, as then."""
        if self._es_mem() is None:
            return None
        if len(fld) > 2:
            return fld[2]
        return np.full(np.asarray(fld[1]).size, np.inf)

    # ------------------------------------------- P1b: the momentum flux
    def _flux_b(self, d, kb, dmaZ, slope):
        """The sections' momentum flux V Dm/Dt with the P1b kinematics
        (switches orbital, wagner_entry): S1's complete material derivative
        (v0's dm/dd x the immersion rate, the strip-exact form over the
        located strips, the leading transom's entry delta astern) with

          - the geometric immersion rate Vg in the rate of the added mass
            and the relative normal velocity V as the momentum's factor and
            in the entry gate (V > 0 and Dm/Dt > 0);
          - the relative inflow ur of each section in the convective part;
          - the flow states (switch wagner_entry): planes in an entry from
            dry (E) or in the exit after it (X) keep Zarnick's m_a and this
            entry-only flux (no flux on exit: S1, design 2.5); the planes
            oscillating about a fixed waterline (F, the fraction f of each
            strip) have m_0 and the Lagrangian force m_0 dV/dt + 1/2 V^2
            dm_0/dd, entering or leaving -- (Vg - V/2) V dm_0/dd with the
            incident flow --, plus the convective rate -ur dm_0/dyc yc'(x)
            (centre values) on entry only; a strip's flux is the sum of its
            fractions' (the strip integral of the two populations);
          - astern, the planes meeting the (blunt) transom arrive dry: an
            entry, S1's delta -ur_T m_T / dx with Zarnick's m_T.

        Returns (flux, hx): hx (None without fixed-waterline planes) the
        part of their vertical force that their horizontal force does not
        carry: the strips are body-fixed, so the Lagrangian surge force has
        no dT/dX part and their horizontal force is the momentum form
        -D/Dt(m_0 V tau) (normal_rate's), i.e. the tilt of the vertical
        force less 1/2 V^2 dm_0/dd tau (tilting it whole carried a force of
        nonzero mean on a trimmed hull oscillating at rest: R3).
        With orbital and wagner_entry off it is S1's flux (_flux_dDt) up to
        the order of floating-point operations."""
        Vr, Vg, ur = kb.V, kb.Vg, kb.ur
        rate = dmaZ * Vg
        act, ycap, iunc = self._substrip(d, slope)
        self._taper_cap = act
        km = self.k_a * 0.5 * np.pi * self.rho
        kc = 0.5 * np.pi / self.tanb
        if act.size:
            uu = ur[act] if self.sw.transom else np.maximum(ur[act], 0.0)
            rate[act] = (Vg[act] * (2.0 * km * kc * kc) * iunc
                         + uu * km * ycap) / self.dx
        if self.sw.transom and kb.urT < 0.0:
            tg = self._transom_geom()
            dT = max(self._trs["dT"], 0.0)
            cT = min(kc * dT, tg.ycT)
            rate[0] = rate[0] - kb.urT * km * cT * cT / self.dx
        flux = np.where((Vr > 0.0) & (rate > 0.0), Vr * rate, 0.0)
        hx = None
        if self.sw.wagner_entry and kb.fixed.any():
            on, f = kb.fixed, kb.f
            conv0 = -ur * kb.dm0y * self._dyc_dx()
            f0 = ((Vg - 0.5 * Vr) * Vr * kb.dm0
                  + np.where((Vr > 0.0) & (conv0 > 0.0), Vr * conv0, 0.0))
            flux = np.where(on, (1.0 - f) * flux + f * f0, flux)
            hx = np.where(on, f * 0.5 * Vr * Vr * kb.dm0, 0.0)
        return flux, hx

    # ------------------------------- P1b: the exact incident-wave pressure
    def _fk_b(self, d, A, ab, Th, kb):
        """The exact linear incident pressure on every section and on the
        transom (switch fk_exact; module docstring 'P1b'), as the part
        beyond the plant's long-wave field (hydrostatic about the local
        surface, which buoy and _x_hydro integrate): a namespace of

          fz     (n,) vertical force per unit length beyond buoy's (x a_b)
          fy     (n,) lateral force per unit length (all of it, no a_b)
          x, m   longitudinal force and its pitch moment beyond _x_hydro's
                 (sections, transom section, wetted face; x a_b)
          my_sig the strip-exact moment correction of fz (pitch): sigma'
          n_sig  the yaw moment beyond sum x fy: the strip-exact sigma'
                 term of fy, and the moment of the pressure on the x-facing
                 surfaces (taper sides, transom face) about the z axis,
                 -int p (-y n_x) dS = int_V y dp/dx dV + int_{A_T} p y dA
                 - int_face p y dA (telescoped, as X)
          K      the roll moment of the exact incident pressure (all of it)

        Per component (amplitude a, phase psi at the station centreline,
        depth decay k, body wavenumbers kappa, lam) and station, with the
        section integrals of hydro/section2d.section_fk (E, E_k, E_lam, W,
        Wy, c) and the strip factors sigma, sigma' (the phase integrated
        exactly over the strip, the section constant across it):

          vertical  rho g a [sigma cos(psi) (-k E + W) - cos(psi) 2c]
          lateral   rho g a lam sigma sin(psi) E
          roll      rho g a sigma sin(psi) [-k E_lam - lam (E_k + z_w E) -
                    Wy]   (z_w the waterline above the CG)
          long.     rho g a kappa sin(psi) (sigma E - A)
          its lever rho g a kappa sin(psi) [sigma (E_k + z_w E) - (z0 A +
                    m1)]
          moments   pitch of the vertical: + sigma' sin(psi) (-k E + W);
                    yaw of the lateral: - sigma' cos(psi) lam E; yaw of the
                    x-facing surfaces: rho g a kappa sigma cos(psi) E_lam
                    per unit length, + rho g a sin(psi_T) [E_lam,T -
                    e^{-k head} E_lam,face] at the transom
          transom   Q = rho g a cos(psi_T) (E_T - A_T), its moment R about
                    the CG; the face over its wetted height h_w below the
                    dead-water level: e^{-k head} E(h_w), z' shifted by
                    -head

        (derivation: divergence theorem on each section, F = -int_A grad p
        dA + z int_WL p dy, M_x = -int_A (y dp/dz - (z - z_G) dp/dy) dA +
        int_WL p y dy; the long-wave parts already in the plant subtracted
        so that the total is exact at first order in the amplitude.)"""
        n = d.size
        wb = self.__dict__.get("_wb")
        if wb is None:
            z = np.zeros(n)
            return SimpleNamespace(fz=z, fy=z, x=0.0, m=0.0, my_sig=0.0,
                                   n_sig=0.0, K=0.0)
        # P1c: the row-stable kernel (sim/as_kernels.fk_rows: the same
        # section integrals, strip moments and long-wave subtraction, every
        # per-section number independent of the other sections and boats, so
        # that the batch plant, sim/planing_batch.py, gives the same bits)
        from . import as_kernels as _ak
        st = wb.st
        tr = self._trs
        r = _ak.fk_rows(
            self._fk_geo(), d[None], A[None], np.array([float(Th)]),
            np.array([float(ab)]), np.array([tr["dT"]]),
            np.array([tr["hw"]]), np.array([tr["head"]]),
            np.array([tr["z0T"]]),
            _ak.FKWaves(a=st.a[None], k=st.k[None], ku=st.ku[None],
                        kinv=st.kinv[None], kap=wb.kap[None],
                        lam=wb.lam[None], Z=wb.Z[None], CT=wb.CT[None],
                        ST=wb.ST[None], e=wb.e[None]))
        return SimpleNamespace(fz=r.fz[0], fy=r.fy[0], x=float(r.x[0]),
                               m=float(r.m[0]), my_sig=float(r.my_sig[0]),
                               n_sig=float(r.n_sig[0]), K=float(r.K[0]))

    def _fk_geo(self):
        """What sim/as_kernels.fk_rows reads of this plant, once per plant:
        the stations' chine, keel height and x, the transom chine and the
        strips' chine moments (_fk_moments) arranged for the strip
        factors."""
        g = self.__dict__.get("_fkg")
        if g is None:
            from . import as_kernels as _ak
            g = _ak.FKGeo(self.yc, self.zk, self.x, self.tanb, self.dx,
                          self.rho * G, SP, self._transom_geom().ycT,
                          self._fk_moments())
            self._fkg = g
        return g

    # number of terms of the strip integrals' Taylor series in kappa xi
    # (the strip factors, sim/as_kernels.FK_TERMS): the remainder
    # (kappa h)^(FK_TERMS + 1) / (FK_TERMS + 1)! is below 1e-14 for kappa h <= pi / 4 (a wave of 4 station spacings)
    # and 5e-5 at kappa h = 4
    FK_TERMS = 16

    def _fk_moments(self):
        """The chine moments of every strip, mu[j, m] = int_strip yc(x) xi^m
        dx (xi = x - x_j, m = 0 .. FK_TERMS + 2), once per plant, from the
        geometry's chine (hydro/hull_geometry): Gauss-Legendre, 40 points per
        strip; on the stem strip in t with x = x_stem - t^2 (the parametric
        family's chine goes like sqrt(x_stem - x) there: smooth in t)."""
        mu = self.__dict__.get("_fkmu")
        if mu is None:
            g = self._geom
            e, xi = np.asarray(g.edges, float), np.asarray(g.xi, float)
            n = xi.size
            M = self.FK_TERMS + 3
            tq, wq = np.polynomial.legendre.leggauss(40)
            mu = np.zeros((n, M))
            for j in range(n):
                a_, b_ = e[j], e[j + 1]
                if j == n - 1:
                    S_ = math.sqrt(b_ - a_)
                    s_ = 0.5 * S_ * (tq + 1.0)
                    xq, wx = b_ - s_ * s_, 0.5 * S_ * wq * 2.0 * s_
                else:
                    xq = 0.5 * (b_ - a_) * tq + 0.5 * (a_ + b_)
                    wx = 0.5 * (b_ - a_) * wq
                w = g.chine(xq) * wx
                xr = xq - xi[j]
                for m in range(M):
                    mu[j, m] = (w * xr ** m).sum()
            self._fkmu = mu
        return mu

    # ------------------------------------------- P1b: roll, lateral flow
    def _roll_b(self, phi, wbar, p, wn_r, fk):
        """The roll moment (switches fk_exact, lateral_phi0): the hydrostatic
        restoring as each plant has it -- the GZ twin's -W GZ(phi - wbar)
        (sim/planing_vessel_gz, its table) or the linear -W GM (phi - wbar)
        -- plus, with fk_exact, the exact roll moment of the incident
        pressure less its long-wave part W GM wbar (equal in long waves of
        uniform slope), and v0's damping placeholder (E6)."""
        Wt = self.m * G
        rel = phi - wbar
        if isinstance(self, PlaningVesselGZ):
            Kr = -Wt * self.gz(rel)
        else:
            Kr = -Wt * self.GM * rel
        if fk is not None:
            Kr = Kr + fk.K - Wt * self.GM * wbar
        return Kr - 2.0 * self.hull.roll_zeta * wn_r * self.Ix * p

    def _lat_coeffs(self, d, phi):
        """Per station, per unit length, about the CG in body axes: the
        phi = 0 sway added mass a22, the sway-roll coupling a24 and the roll
        added inertia a44 at the immersion d and heel phi, and their
        derivatives along d (fixed heel) and phi (hydro/section2d.Table,
        about the keel; the CG is h_G = -zk above the keel in body axes, so
        a24 = a24_K + h_G a22, a44 = a44_K + 2 h_G a24_K + h_G^2 a22)."""
        c = self.__dict__.get("_latc")
        if c is not None and c[0] is d and c[1] == phi:
            # the same immersion array and heel: the step's lateral rate
            # right after the first derivative of the step
            return c[2]
        C, dCd, dCp = self._tab().lateral(d, self.yc, phi)
        hG = -self.zk
        rho = self.rho

        def tr(M):
            a22 = M[:, 0]
            return (rho * a22, rho * (M[:, 3] + hG * a22),
                    rho * (M[:, 5] + 2.0 * hG * M[:, 3] + hG * hG * a22))
        res = (tr(C), tr(dCd), tr(dCp))
        self._latc = (d, phi, res)
        return res

    def _lateral_b(self, s, d, kb, wr=None):
        """The lateral flow (switch lateral_phi0; module docstring 'P1b'):
        per section the impulses P = a22 v_r + a24 p, L = a24 v_r + a44 p
        (v_r = v + x r - v_orb), the sectional force -D/Dt(P) and roll
        moment -D/Dt(L), D/Dt = d/dt|_x - u_r d/dx, summed:

          the accelerations' part         on the left (_lat_solve)
          the orbital velocity's rate     + a22 a_oy, + a24 a_oy (lateral
                                           diffraction), a_oy the rate AT
                                           FIXED BODY x (wr, _wave_rates_b:
                                           the convective part is in
                                           conservation form; S5a review,
                                           issue C -- P1b used the earth-
                                           fixed rate)
          the added mass's rate at fixed x - (adot22 v_r + adot24 p), etc.;
                                           adot = da/dd ddot + da/dphi p
          the convective part              shed at the trailing edge: the
                                           transom going ahead, -ur_T P_T,
                                           -ur_T L_T; astern the stem (P =
                                           0: none); the Munk moment
                                           - sum ur P dx (the sign of ur)
          the Lagrangian reactions         + dT/dphi (roll), dT/dz (heave),
                                           dT/dtheta (pitch), T = 1/2 sum
                                           (a22 v_r^2 + 2 a24 v_r p + a44
                                           p^2) dx
          Kirchhoff's partner of the Munk  + r sum P dx in surge
          moment
          the cross-flow drag on v_r       v0's (E5)

        tau (6,) the generalised forces, M3 the 3x3 lateral mass (v, p,
        r)."""
        u, v, w, p, q, r = s[6:12]
        phi = s[3]
        x = self.x
        dx = self.dx
        (a22, a24, a44), (d22, d24, d44), (p22, p24, p44) = \
            self._lat_coeffs(d, phi)
        vr = v + x * r - kb.vo
        P = a22 * vr + a24 * p
        L = a24 * vr + a44 * p
        dd = kb.ddot
        ad22 = d22 * dd + p22 * p
        ad24 = d24 * dd + p24 * p
        ad44 = d44 * dd + p44 * p
        # the convective part -(-u_r dP/dx) integrated over the hull with
        # the planes that arrive over the leading edge: [u_r P] at the ends
        # less the Munk sum u_r P. Going ahead (u_r > 0) the transom is the
        # trailing edge and sheds -u_r P_T (v0's slender-body lift); astern
        # the planes meet the transom with P jumping from 0 to P_T -- inside
        # the D/Dt integral, it cancels the interior's end term -- and the
        # trailing edge is the stem, P = 0: no shedding. The Munk sum keeps
        # the sign of u_r, so with Kirchhoff's partner r sum P dx in surge
        # the pair does no work for either sign
        sh = kb.urT if kb.urT > 0.0 else 0.0
        aoy = wr.aoy if (wr is not None and self.sw.orbital and kb.wv) \
            else kb.aoy
        Ylat = (-(ad22 * vr + ad24 * p) + a22 * aoy)
        Klat = (-(ad24 * vr + ad44 * p) + a24 * aoy
                + 0.5 * (p22 * vr * vr + 2.0 * p24 * vr * p + p44 * p * p))
        quad = d22 * vr * vr + 2.0 * d24 * vr * p + d44 * p * p
        dp = np.maximum(d, 0.0)
        Ycf = -0.5 * self.rho * self.cd_lat * dp * vr * np.abs(vr)
        Y = (Ylat + Ycf).sum() * dx - sh * P[0]
        N = (x * (Ylat + Ycf)).sum() * dx + x[0] * (-sh * P[0]) \
            - (kb.ur * P).sum() * dx
        K = Klat.sum() * dx - sh * L[0]
        Zf = -0.5 * quad.sum() * dx
        Mf = -0.5 * (SP * x * quad).sum() * dx
        Xk = r * P.sum() * dx
        A22, A24, A26 = a22.sum() * dx, a24.sum() * dx, (a22 * x).sum() * dx
        A44, A46 = a44.sum() * dx, (a24 * x).sum() * dx
        A66 = (a22 * x * x).sum() * dx
        M3 = np.array([[self.m + A22, A24, A26],
                       [A24, self.Ix + A44, A46],
                       [A26, A46, self.Iz + A66]])
        return SimpleNamespace(tau=np.array([Xk, Y, Zf, K, Mf, N]), M3=M3,
                               A=(A22, A24, A26, A44, A46, A66), P=P, L=L,
                               vr=vr)

    @staticmethod
    def _lat_solve(lat, tau_o):
        """(v', p', r') from the lateral mass matrix and the right-hand sides
        tau_o[1], tau_o[3], tau_o[5] (Cramer's rule, 3x3)."""
        M = lat.M3
        b0, b1, b2 = tau_o[1], tau_o[3], tau_o[5]
        a, b_, c = M[0]
        d_, e, f = M[1]
        g, h, i = M[2]
        det = a * (e * i - f * h) - b_ * (d_ * i - f * g) + c * (d_ * h - e * g)
        x0 = (b0 * (e * i - f * h) - b_ * (b1 * i - f * b2)
              + c * (b1 * h - e * b2)) / det
        x1 = (a * (b1 * i - f * b2) - b0 * (d_ * i - f * g)
              + c * (d_ * b2 - b1 * g)) / det
        x2 = (a * (e * b2 - b1 * h) - b_ * (d_ * b2 - b1 * g)
              + b0 * (d_ * h - e * g)) / det
        return float(x0), float(x1), float(x2)

    def _lateral_rate(self, s, d):
        """v0's estimate of the fastest lateral rate (the substep rule);
        with switch lateral_phi0 the same eigenvalue of d(v', r')/d(v, r)
        with the phi = 0 sway added mass, the relative inflow and the
        instantaneous lateral inertia."""
        if not self.sw.lateral_phi0:
            return super()._lateral_rate(s, d)
        u, v, r = s[6], s[7], s[11]
        x, dx, rho = self.x, self.dx, self.rho
        (a22, a24, a44), _, _ = self._lat_coeffs(d, s[3])
        au = abs(u)
        dp = np.maximum(d, 0.0)
        k = rho * self.cd_lat * dp * np.abs(v + x * r)
        x0, mT = x[0], au * a22[0]
        Yv = -mT - k.sum() * dx
        Yr = -mT * x0 - (k * x).sum() * dx - self.m * u
        Nv = -mT * x0 - au * a22.sum() * dx - (k * x).sum() * dx
        Nr = (-mT * x0 * x0 - au * (a22 * x).sum() * dx
              - (k * x * x).sum() * dx)
        M22 = self.m + a22.sum() * dx
        M66 = self.Iz + (a22 * x * x).sum() * dx
        a, b = Yv / M22, Yr / M22
        c, e = Nv / M66, Nr / M66
        half, det = 0.5 * (a + e), a * e - b * c
        disc = half * half - det
        if disc >= 0.0:
            return float(abs(half) + np.sqrt(disc))
        return float(np.sqrt(det))

    # ------------------------------------- S5a: heeled-section hydrostatics
    def _hh_geom(self):
        """The section polygons of switch heel_hydro, once per plant: body
        (y, z) about each station's point on the body x axis (the CG's
        height plus SP x Th), counter-clockwise -- keel, + chine, + deck
        edge, - deck edge, - chine: the frozen GZ table's polygons
        (sim/planing_vessel_gz.PlaningVesselGZ._sections) -- and the
        transom section's (chine yc_T, keel z_kT: _transom_geom)."""
        g = self.__dict__.get("_hhg")
        if g is None:
            h = self.hull
            yc, zk = self.yc, self.zk
            zc = zk + np.minimum(yc * self.tanb, h.depth)
            zd = zk + h.depth
            tg = self._transom_geom()
            ycT, zkT = tg.ycT, tg.zkT
            zcT = zkT + min(ycT * self.tanb, h.depth)
            zdT = zkT + h.depth
            g = SimpleNamespace(
                py=np.stack([np.zeros_like(yc), yc, yc, -yc, -yc], -1),
                pz=np.stack([zk, zc, zd, zd, zc], -1),
                pyT=np.array([0.0, ycT, ycT, -ycT, -ycT]),
                pzT=np.array([zkT, zcT, zdT, zdT, zcT]))
            self._hhg = g
        return g

    def _hh_state(self, phi, s_tr):
        """The heel of one derivative (switch heel_hydro): None on the
        upright branch (phi == 0 and every station's transverse slope
        exactly 0: calm water, or long-crested waves along the heading),
        where the copy keeps P1c's lines -- the same integral over the
        upright symmetric section in closed form, so that those states are
        P1c's bit for bit; else a namespace with cos and sin of the heel,
        the slopes s, ck = cos phi (the keel depth d = H - zk ck), and the
        surface-aligned frame of every station: alpha = atan(s), the polygon
        heeled by phi - alpha relative to it. A plant with _hh_force set
        takes the polygons on the upright branch too (checks H3). The keel
        depth is taken below the horizontal through the station's centreline
        surface point, d = H - zk cos phi (the tilt only in the hydrostatic
        polygons): its rate at fixed body x is then exactly the immersion
        rate kb.ddot plus zk sin phi p, which the cross-plane impulse
        (lateral_cross) needs (a tilted keel depth would need the slope's
        own rate; plan 'S5a', a change made while implementing)."""
        if phi == 0.0 and not s_tr.any() and not \
                self.__dict__.get("_hh_force", False):
            return None
        c, sn = math.cos(phi), math.sin(phi)
        hh = SimpleNamespace(phi=float(phi), c=c, s=sn, st=s_tr)
        hh.ck = c
        if s_tr.any():
            al = np.arctan(s_tr)
            hh.ca, hh.sa = np.cos(al), np.sin(al)
            rel = phi - al
            hh.cr, hh.sr = np.cos(rel)[:, None], np.sin(rel)[:, None]
        else:
            hh.ca, hh.sa = 1.0, 0.0
            hh.cr, hh.sr = c, sn
        return hh

    def _hh_sections(self, hh, e, z, Th):
        """The stations' heeled polygons below the local surface (switch
        heel_hydro): H the surface's height above each station's axis
        point, the polygon rotated by phi - alpha into the surface-aligned
        frame (the surface is the line w = H cos alpha there), _clip_moments,
        the first moments rotated back to the level frame. Sets hh.H, A, Sy,
        Sz (about the axis points, y to port, z up)."""
        g = self._hh_geom()
        H = e - (self.Z_rest + z + SP * self.x * Th)
        Yr = g.py * hh.cr - g.pz * hh.sr
        Zr = g.py * hh.sr + g.pz * hh.cr
        A, Su, Sw = _clip_moments(Yr, Zr, H * hh.ca)
        hh.H, hh.A = H, A
        hh.Sy = Su * hh.ca - Sw * hh.sa
        hh.Sz = Su * hh.sa + Sw * hh.ca

    @staticmethod
    def _hh_poly(py, pz, phi, ca, sa, Hc):
        """One polygon heeled by phi in a surface tilted by alpha (cos ca,
        sin sa), clipped at w = Hc in the surface-aligned frame: (A, Sy, Sz,
        Syy, Syz, Szz) in the level frame about the axis point."""
        rel = phi - math.atan2(sa, ca)
        cr, sr = math.cos(rel), math.sin(rel)
        A, Su, Sw, Suu, Suw, Sww = _clip_moments(
            py * cr - pz * sr, py * sr + pz * cr, np.array(Hc), second=True)
        A, Su, Sw = float(A), float(Su), float(Sw)
        Suu, Suw, Sww = float(Suu), float(Suw), float(Sww)
        Sy = Su * ca - Sw * sa
        Sz = Su * sa + Sw * ca
        Syy = ca * ca * Suu - 2.0 * ca * sa * Suw + sa * sa * Sww
        Szz = sa * sa * Suu + 2.0 * ca * sa * Suw + ca * ca * Sww
        Syz = ca * sa * (Suu - Sww) + (ca * ca - sa * sa) * Suw
        return A, Sy, Sz, Syy, Syz, Szz

    def _hh_transom(self, hh, x0, y0, psi, z, Th, t, u):
        """The heeled transom section (switch heel_hydro): the elevation and
        the transverse slope at the transom point (v0's phase sum for one
        point), the section's moments below the local surface and those of
        the face below the dead-water level (the same surface lowered by
        the head, the flow state of S2a recomputed on the heeled keel
        depth); replaces dT and hw in self._trs and keeps the moments in
        hh.T."""
        tg = self._transom_geom()
        g = self._hh_geom()
        tr = self._trs
        if getattr(self, "_wa", None) is None:
            eT, sT = 0.0, 0.0
        else:
            cps, sps = math.cos(psi), math.sin(psi)
            XT, YT = x0 + tg.xT * cps, y0 + tg.xT * sps
            ph = XT * self._wkx + YT * self._wky - self._ww * t + self._wphi
            sph = np.sin(ph)
            eT = float(np.cos(ph) @ self._wa[0])
            sT = float(-(sph @ self._wa[3]) * sps + (sph @ self._wa[4]) * cps)
        H = eT - (self.Z_rest + z + SP * tg.xT * Th)
        al = math.atan(sT)
        ca, sa = math.cos(al), math.sin(al)
        sec = self._hh_poly(g.pyT, g.pzT, hh.phi, ca, sa, H * ca)
        dT = H - tg.zkT * hh.c
        cl = tg.cl
        head = tr["head"]
        if cl.mode == "progressive_static":
            up = max(float(u), 0.0)
            head = dT * up * up / (G * cl.F_T_crit * cl.F_T_crit
                                   * max(tg.TT0, 1e-12))
        hw = max(dT - head, 0.0) if np.isfinite(head) else 0.0
        face = None
        if hw > 0.0:
            face = self._hh_poly(g.pyT, g.pzT, hh.phi, ca, sa,
                                 (H - head) * ca)
        tr.update(dT=float(dT), hw=float(hw), head=float(head))
        hh.T = SimpleNamespace(H=H, s=sT, sec=sec, face=face, head=head)

    def _x_hydro_heel(self, hh, s_lon, ab, u, Th, tr):
        """S5a: _x_hydro over the heeled sections -- the longitudinal force
        of p = a_b rho g (eta + s y - z) over the hull (telescoped: the
        stations' -int dp/dx dA, the transom section's pressure force) and
        the wetted face, its pitch moment (z levers about the CG), and the
        yaw moment of the x-facing surfaces, int_V y dp/dx dV + int_{A_T} p y
        dA - int_face p y dA (dp/dx = a_b rho g eta_x; the slope's own
        x-variation s_x y is second order and left out). Returns (X, M, N);
        the parts go into tr (aux)."""
        tg = self._transom_geom()
        rg = self.rho * G
        dx = self.dx
        if getattr(self, "_wa", None) is None:
            sx = sm = sn = 0.0
        else:
            sx = (hh.A * s_lon).sum() * dx
            sm = (s_lon * (SP * self.x * Th * hh.A + hh.Sz)).sum() * dx
            sn = (s_lon * hh.Sy).sum() * dx
        T = hh.T
        lev = SP * tg.xT * Th
        A_, Sy, Sz, Syy, Syz, Szz = T.sec
        qT = T.H * A_ + T.s * Sy - Sz
        RT = lev * qT + T.H * Sz + T.s * Syz - Szz
        NT = T.H * Sy + T.s * Syy - Syz
        qf = Rf = Nf = 0.0
        if T.face is not None:
            Af, Syf, Szf, Syyf, Syzf, Szzf = T.face
            Hf = T.H - T.head
            qf = Hf * Af + T.s * Syf - Szf
            Rf = lev * qf + Hf * Szf + T.s * Syzf - Szzf
            Nf = Hf * Syf + T.s * Syyf - Syzf
        if u < 0.0 and A_ > 0.0:
            Dface = 0.5 * self.rho * tg.cl.cd_face * A_ * u * u
            zA, yA = lev + Sz / A_, Sy / A_
        else:
            Dface, zA, yA = 0.0, 0.0, 0.0
        X = -ab * rg * (sx + qT) + ab * rg * qf + Dface
        M = -ab * rg * (sm + RT) + ab * rg * Rf + zA * Dface
        # the yaw moment without a_b (P1c's convention, _hh_lateral; the
        # previous round's a_b only as the variant 'ab_lat')
        wn = ab if self._s5a_v("ab_lat") else 1.0
        N = wn * rg * (sn + NT - Nf) - yA * Dface
        tr.update(QT=float(ab * rg * qT), Qw=float(ab * rg * qf),
                  RT=float(ab * rg * RT), Rw=float(ab * rg * Rf),
                  Dface=float(Dface), Xfk=float(-ab * rg * sx),
                  Mfk=float(-ab * rg * sm), Nh=float(N))
        return X, M, N

    def _hh_longwave(self, d, A, Th):
        """The long-wave limit of fk_exact's lateral force and roll moment
        per unit slope, on the same upright section (sim/as_kernels.fk_rows
        at k, lam -> 0, the strips at their centre values): (A, A z_B + 2
        c^3 / 3), z_B the centroid height above the CG, c the waterline
        half-beam of the V part (hydro/section2d.section_fk's c_V)."""
        A_, m1, _ = _vsec(d, self.yc, self.tanb)
        z0 = self.zk + SP * self.x * Th
        cV = np.minimum(np.maximum(d, 0.0), self.yc * self.tanb) / self.tanb
        return A, z0 * A_ + m1 + 2.0 * cV ** 3 / 3.0

    def _s5a_v(self, key):
        """A diagnostic variant of switches heel_hydro / lateral_cross, for
        the checks only (plan 'S5a：复查后的修正', decision 10): the plant
        attribute _s5a_var, a dict that no plant the code builds has --
        'm33' in ('ma' (default), 'e33', 'none') the heave-heave entry of
        the roll reaction (_lateral_x), 'ab_lat' True the previous round's
        a_b and ctr on the lateral, roll and yaw parts (_hh_lateral,
        _hh_roll, _x_hydro_heel)."""
        v = self.__dict__.get("_s5a_var")
        return None if v is None else v.get(key)

    def _hh_lateral(self, hh, ab, ctr, d, A, fk):
        """S5a: the lateral force per unit length -rho g s A of the tilted
        hydrostatic field on the heeled polygons, plus (fk_exact) the
        incident pressure beyond its long-wave part; and dN, the correction
        of the yaw moment fk_exact's n_sig carries (the copy adds n_sig
        whole). P1c's CONVENTION (S5a review, issue D): the lateral force
        and the roll and yaw moments carry neither a_b nor the transom
        correction ctr -- fk_exact's lateral, roll and yaw never did, and G1
        forbids extending the speed-weighted a_b(C_V) (register E2) to new
        force components; the vertical force, the longitudinal force and
        the pitch moment keep a_b. So the parts of one pressure field are
        not weighted alike (recorded in register E2). The previous round's
        form (a_b and ctr on all of it) only as the diagnostic variant
        _s5a_v('ab_lat').
          THE TRANSOM'S YAW (found in this review's checks, not by the
        review): fk_exact's n_sig carries the yaw moment of the whole
        incident pressure on the x-facing surfaces, at the transom
        int_{A_T} p y dA - int_face p y dA, whose long-wave limit rho g s_T
        (I_yy(d_T) - I_yy(h_w)) (s_T the transverse slope there) the tilted
        polygons of _x_hydro_heel also carry: dN subtracts it, as the
        lateral force and the roll moment subtract theirs (at the strips
        n_sig's long-wave part, the slope's own x-variation s_x I_yy, is
        not in the polygons and stays). Without it hh's first-order wave
        yaw moment at 25 kn in beam seas was off by -879 N m per metre of
        amplitude against P1c's 475 (at rest the face is wholly wet and the
        two transom terms cancel). Returns (Ysec, dN)."""
        rg = self.rho * G
        old = self._s5a_v("ab_lat")
        dN = 0.0
        if old:
            Y = -ab * rg * ctr * hh.st * hh.A
            if fk is not None:
                Alw, _ = self._hh_longwave(d, A, 0.0)
                Y = Y + ab * (fk.fy + rg * hh.st * Alw)
                dN = (ab - 1.0) * fk.n_sig
            return Y, dN
        Y = -rg * hh.st * hh.A
        if fk is not None:
            Alw, _ = self._hh_longwave(d, A, 0.0)
            Y = Y + (fk.fy + rg * hh.st * Alw)
            tr = self._trs
            dT, hw = tr["dT"], tr["hw"]
            if dT > 0.0:
                ycT = self._transom_geom().ycT
                iy = _vsec_iyy(dT, ycT, self.tanb)
                if hw > 0.0 and np.isfinite(tr["head"]):
                    iy = iy - _vsec_iyy(hw, ycT, self.tanb)
                dN = -rg * hh.T.s * iy
        return Y, dN

    def _hh_roll(self, hh, ab, ctr, d, A, Th, p, wn_r, fk):
        """S5a: the roll moment about the plant's roll axis -- the level
        frame's x axis through the CG, levers measured from the CG's height
        at every station as fk_exact's (sim/as_kernels.fk_rows) and P1c's:
        the hydrostatic field's rho g sum (S_y + s (S_z + SP x Th A)) dx on
        the heeled polygons (Sy, Sz about each station's axis point, which
        is SP x Th above the CG's height: the lateral force -rho g s A gets
        that extra lever; the previous round integrated the polygons about
        the axis points and subtracted fk_exact's long-wave limit about the
        CG's height, a first-order term in slope x trim, S5a review issue
        O), plus (fk_exact) the incident pressure beyond its long-wave part
        (K - rho g sum s (z0 A + m1 + 2 c^3/3) dx), and v0's damping
        placeholder (E6). Neither a_b nor ctr (P1c's convention,
        _hh_lateral); the previous round's form as the variant 'ab_lat'."""
        rg = self.rho * G
        if self._s5a_v("ab_lat"):
            K = ab * rg * ctr * ((hh.Sy + hh.st * hh.Sz).sum() * self.dx)
            if fk is not None:
                _, klw = self._hh_longwave(d, A, Th)
                K = K + ab * (fk.K - rg * (hh.st * klw).sum() * self.dx)
            return K - 2.0 * self.hull.roll_zeta * wn_r * self.Ix * p
        K = rg * ((hh.Sy + hh.st * (hh.Sz + SP * self.x * Th * hh.A)).sum()
                  * self.dx)
        if fk is not None:
            _, klw = self._hh_longwave(d, A, Th)
            K = K + (fk.K - rg * (hh.st * klw).sum() * self.dx)
        return K - 2.0 * self.hull.roll_zeta * wn_r * self.Ix * p

    # ------------------------------ S5a: the complete cross-plane impulse
    def _wave_rates_b(self, s, t, d, A, kb):
        """S5a review (issue C): the rates of the wave velocities AT A FIXED
        BODY STATION, which the lateral flows' conservation form needs: the
        convective part -u_r dP/dx is integrated along the hull into the
        trailing edge's shedding and the Munk sums, so the time part left is
        the rate at fixed body x (_lateral_x, _lateral_b, orbital's lateral
        diffraction). For a component (amplitude a, frequency w, depth decay
        k, body wavenumbers kappa along the hull and lam across, phase psi =
        kx X + ky Y - w t + phi) at a station moving with (u, v + x r)
        horizontally, its centroid depth z_c below the local surface (the
        orbital decay e^{k z_c} of _p1b_kin) changing at z_c', and the body
        axes turning with r:

          d v_o/dt = a_oy + u dv_o/dxi + (v + x r) dv_o/deta + z_c' dv_o/dz
                     - r u_o
          d u_o/dt = a_ox + u du_o/dxi + (v + x r) du_o/deta + z_c' du_o/dz
                     + r v_o
          d w_o/dt = a_oz + u dw_o/dxi + (v + x r) dw_o/deta + z_c' dw_o/dz

        (a_o the earth-fixed rates of _p1b_kin; d/dxi cos psi = -kappa sin
        psi, d/deta = -lam sin psi, dD/dz = k D: dv_o/dz = dw_o/deta and
        du_o/dz = dw_o/dxi, the flow is irrotational), and without orbital
        the centreline surface's rate d et/dt = ett + u det/dxi + (v + x r)
        det/deta. The convective parts are FIRST order (u kappa / w ~ U /
        c): with the earth-fixed rates every strip had the absolute
        frequency where strip theory (Salvesen-Tuck-Faltinsen) has the
        encounter frequency and the transom term (checks STF and L1'+ of
        plan 'S5a：复查后的修正'). z_c' = -(1 - A' q / A^2) d', q = d A - m1,
        A' = dA/dd the waterline width of the upright section, d' the
        immersion rate at fixed body x (kb.ddot, plus zk sin phi p for
        switch heel_hydro's keel depth). Returns a namespace (aoy, aox, aoz
        with orbital, ett without; all body-fixed), None in calm water."""
        if not self._waves_on():
            return None
        x = self.x
        u, v, r = float(s[6]), float(s[7]), float(s[11])
        vl = v + x * r
        wb = self.__dict__.get("_wb") if self.sw.fk_exact else None
        if wb is not None:
            C, S = wb.C, wb.S
            kap, lam, aw = wb.kap, wb.lam, wb.st.aw
        else:
            # the component phases at the stations from v0's wave arrays
            # (sim/planing_vessel._waves), body wavenumbers by the heading
            cps, sps = math.cos(float(s[5])), math.sin(float(s[5]))
            X, Y = s[0] + x * cps, s[1] + x * sps
            ph = (np.outer(X, self._wkx) + np.outer(Y, self._wky)
                  - self._ww * t + self._wphi)
            C, S = np.cos(ph), np.sin(ph)
            kap = self._wkx * cps + self._wky * sps
            lam = -self._wkx * sps + self._wky * cps
            aw = self._wa[1]
        if not (self.sw.orbital and wb is not None):
            g = C @ np.stack([aw * kap, aw * lam], 1)
            return SimpleNamespace(ett=None, conv=u * g[:, 0] + vl * g[:, 1])
        st = wb.st
        dp = np.maximum(d, 0.0)
        A_, m1, _ = _vsec(d, self.yc, self.tanb)
        pos = A > 0.0
        As = np.where(pos, A, 1.0)
        zc = np.where(pos, -(dp - m1 / As), 0.0)
        Dk = np.exp(np.multiply.outer(zc, st.ku))[:, st.kinv]
        CD, SD = C * Dk, S * Dk
        ct, sn = kap / st.ksafe, lam / st.ksafe
        Aw = np.where(dp < self.yc * self.tanb, 2.0 * dp / self.tanb,
                      2.0 * self.yc)
        dzc = np.where(pos, -(1.0 - Aw * (dp * A - m1) / (As * As)), 0.0)
        dd = kb.ddot
        if self.sw.heel_hydro:
            dd = dd + self.zk * math.sin(float(s[3])) * float(s[9])
        zcd = dzc * dd
        gs = SD @ np.stack([aw * sn * kap, aw * sn * lam, aw * ct * kap,
                            aw * ct * lam, aw * st.k], 1)
        gc = CD @ np.stack([aw * kap, aw * lam], 1)
        return SimpleNamespace(
            aoy=kb.aoy - u * gs[:, 0] - vl * gs[:, 1] + zcd * gc[:, 1]
            - r * kb.uo,
            aox=kb.aox - u * gs[:, 2] - vl * gs[:, 3] + zcd * gc[:, 0]
            + r * kb.vo,
            aoz=kb.aoz + u * gc[:, 0] + vl * gc[:, 1] + zcd * gs[:, 4])

    def _m33_dd(self, d, kb, dma, ctr):
        """S5a review (issue A): d m33 / dd, m33 the heave equation's added
        mass (the copy's ma after the transom correction ctr): Zarnick's
        k_a pi/2 rho min(kc d, yc)^2 times ctr -- dma, the copy's own
        derivative after ctr --, with switch wagner_entry the flow states'
        fractions, (1 - f - f_X) Zarnick + f m_0 + f_X k_a pi/2 rho mean
        min(kc d, yc, Y)^2, at fixed fractions, times ctr."""
        if not self.sw.wagner_entry:
            return dma
        dmaZ = self._section(d)[3]
        out = dmaZ
        if kb.fixed.any():
            out = np.where(kb.fixed, (1.0 - kb.f) * dmaZ + kb.f * kb.dm0,
                           out)
        if kb.fX is not None and kb.fX.any():
            kc = 0.5 * np.pi / self.tanb
            dpos = np.maximum(d, 0.0)
            cz = np.minimum(kc * dpos, self.yc)
            dcz = np.where((d > 0.0) & (kc * dpos < self.yc), kc, 0.0)
            eg = self._es_geom()
            b, st = kb.es
            qd = _xmass_dd(b, st, kb.esy, eg.edges, cz, dcz)
            mxd = self.k_a * 0.5 * np.pi * self.rho * qd
            out = out + kb.fX * (mxd - dmaZ)
        return out * ctr

    def _lateral_x(self, s, d, kb, tau, ett, s_tr, ma=None, dm33=None,
                   wr=None):
        """The cross-plane impulse dynamics of switch lateral_cross (module
        docstring 'S5a'). Per station, in the plant's level frame about the
        CG, from the table (hydro/section2d: body frame, about the keel, at
        the keel depth d and heel phi): B shifted along body z to the CG
        (h_G = -zk), then its translational part rotated to the level frame,

            M = R^T B_G R  ->  e22, e23, e33, e24, e34, e44  (x rho),

        and its derivatives along d (fixed heel) and phi (fixed d: the
        table's own and the tensor's rotation, d e22 = -2 e23, d e23 = e22 -
        e33, d e33 = 2 e23, d e24 = -e34, d e34 = e24), at a fixed CG with
        the keel depth's change dd/dphi = zk sin phi.
          ONE KINETIC ENERGY (S5a review, issue A): the heave-heave entry is
        the heave equation's own added mass m33 (ma: Zarnick, or
        wagner_entry's flow states; times ctr), not the table's e33:

            T = 1/2 m33(d) W_r^2 + 1/2 nu^T M~ nu,  M~ = M - e33 e_W e_W^T,

        and every Lagrangian reaction comes from this T -- the roll one
        includes 1/2 W_r^2 dm33/dd zk sin phi (dm33: _m33_dd), which the
        previous round dropped with e33 (the table's e33 instead: variant
        _s5a_v('m33') == 'e33', the complete phi = 0 entry; 'none' the
        previous round). This is the strict mixed basis (register S5-2):
        energetically one system where the vertical model is (at rest
        chine-wetted, dm33/dd = 0; wagner_entry's F planes); the phi = 0
        heave flow's own heel dependence (de33/dphi) is not in it.
          The form is the fixed-waterline (Lagrangian) one on every plane, an
        assumption for planes entering from dry (register S5-7: unverified
        in the planing regime; the momentum form of A1 would drop the
        reactions). Velocities relative to the water nu = (v_r, W_r, p),
        v_r = v + x r - v_orb, W_r = -V. Per plane F = -D/Dt(M~ nu) + dT/dq,
        D/Dt = d/dt|_x - u_r d/dx: the accelerations go to the mass matrix
        (_solve5); the rest is -M~ n0 (n0 the known part of dnu/dt AT FIXED
        BODY x, _wave_rates_b: the convective part below is in conservation
        form), -(dM~/dt|_x) nu, the reactions -1/2 nu^T M~_d nu (heave,
        pitch by SP x) and 1/2 nu^T M_phi|G nu (roll, with the m33 part);
        the convective part integrated along the hull: the trailing edge
        sheds -u_r Pi_T going ahead (none astern: the stem), the yaw Munk
        -sum u_r P_y dx, the pitch one -SP sum u_r P_z dx; in the yaw-
        rotating frame Kirchhoff's partners of the horizontal impulse: + r
        sum P_y dx in surge, and for P_x = sum tau (m33 V - P_z) dx (the
        vertical model's and the cross plane's, normal to the bottom) - r
        P_x in sway and + v P_x in yaw (S5a review, issue F), with the surge
        added mass's pair (-A11 u r, + A11 u v: P_x's A11 u part). The
        cross-flow drag on v_r (E5) with its roll lever at the mid-depth of
        the immersion (assumption, plan 'S5a')."""
        u, v, w, p, q, r = s[6:12]
        phi = float(s[3])
        x, dx, rho = self.x, self.dx, self.rho
        C, Cd, Cp = self._tab().lateral(d, self.yc, phi)
        hG = -self.zk
        c, sn = math.cos(phi), math.sin(phi)
        cc, ss, cs = c * c, sn * sn, c * sn

        def lev(M):
            b22, b23, b33 = M[:, 0], M[:, 1], M[:, 2]
            b24, b34, b44 = M[:, 3], M[:, 4], M[:, 5]
            b24g = b24 + hG * b22
            b34g = b34 + hG * b23
            b44g = b44 + 2.0 * hG * b24 + hG * hG * b22
            return (rho * (cc * b22 - 2.0 * cs * b23 + ss * b33),
                    rho * (cs * b22 + (cc - ss) * b23 - cs * b33),
                    rho * (ss * b22 + 2.0 * cs * b23 + cc * b33),
                    rho * (c * b24g - sn * b34g),
                    rho * (sn * b24g + c * b34g), rho * b44g)
        e22, e23, e33, e24, e34, e44 = lev(C)
        d22, d23, d33, d24, d34, d44 = lev(Cd)
        t22, t23, t33, t24, t34, t44 = lev(Cp)
        ddp = self.zk * sn
        g22 = t22 - 2.0 * e23 + d22 * ddp
        g23 = t23 + e22 - e33 + d23 * ddp
        g24 = t24 - e34 + d24 * ddp
        g34 = t34 + e24 + d34 * ddp
        g44 = t44 + d44 * ddp
        vr = v + x * r - kb.vo
        Wr = -kb.V
        Py = e22 * vr + e23 * Wr + e24 * p
        Pz = e23 * vr + e34 * p
        L = e24 * vr + e34 * Wr + e44 * p
        dd = kb.ddot
        r22, r23, r24 = d22 * dd + g22 * p, d23 * dd + g23 * p, \
            d24 * dd + g24 * p
        r34, r44 = d34 * dd + g34 * p, d44 * dd + g44 * p
        qd = (d22 * vr * vr + 2.0 * d23 * vr * Wr + 2.0 * d24 * vr * p
              + 2.0 * d34 * Wr * p + d44 * p * p)
        qp = (g22 * vr * vr + 2.0 * g23 * vr * Wr + 2.0 * g24 * vr * p
              + 2.0 * g34 * Wr * p + g44 * p * p)
        # the heave-heave entry's roll reaction, 1/2 W_r^2 dm33/dphi|_G: m33
        # the heave equation's own (one T; dm33 = _m33_dd), depending on the
        # heel through the keel depth only (S5a review, issue A)
        var = self._s5a_v("m33")
        if var == "e33":
            qp = qp + (t33 + 2.0 * e23 + d33 * ddp) * Wr * Wr
        elif var != "none" and dm33 is not None:
            qp = qp + dm33 * ddp * Wr * Wr
        # the known parts of the velocities' rates at fixed body x (S5a
        # review, issue C; _wave_rates_b): -n0
        if self.sw.orbital and kb.wv:
            aoy = wr.aoy
            aW0 = wr.aoz - tau * wr.aox + kb.ur * SP * q
        else:
            aoy = kb.aoy
            aW0 = (ett if wr is None else ett + wr.conv) + kb.ur * SP * q
        fy = e22 * aoy + e23 * aW0 - (r22 * vr + r23 * Wr + r24 * p)
        fz = e23 * aoy - (r23 * vr + r34 * p) - 0.5 * qd
        fL = (e24 * aoy + e34 * aW0 - (r24 * vr + r34 * Wr + r44 * p)
              + 0.5 * qp)
        dp = np.maximum(d, 0.0)
        Ycf = -0.5 * rho * self.cd_lat * dp * vr * np.abs(vr)
        Kcf = -((self.zk * c + 0.5 * dp) * Ycf).sum() * dx
        sh = kb.urT if kb.urT > 0.0 else 0.0
        Ysh, Zsh, Ksh = -sh * Py[0], -sh * Pz[0], -sh * L[0]
        A11 = self.Mtot[0, 0] - self.m
        # the horizontal fluid impulse, normal to the bottom (S5a review,
        # issue F): the vertical model's m33 V tau and the cross plane's
        # -P_z tau, with the surge added mass A11 u -- its Kirchhoff pair in
        # the yaw-rotating frame, -r P_x (sway) and + v P_x (yaw)
        Px = A11 * u
        if ma is not None:
            Px = Px + (tau * (ma * kb.V - Pz)).sum() * dx
        Y = (fy + Ycf).sum() * dx + Ysh - r * Px
        Z = fz.sum() * dx + Zsh
        K = fL.sum() * dx + Ksh + Kcf
        Mq = ((SP * x * fz).sum() * dx + SP * x[0] * Zsh
              - SP * (kb.ur * Pz).sum() * dx)
        N = ((x * (fy + Ycf)).sum() * dx + x[0] * Ysh
             - (kb.ur * Py).sum() * dx + v * Px)
        Xk = r * Py.sum() * dx
        A_ = dict(A22=e22.sum() * dx, A24=e24.sum() * dx,
                  A26=(e22 * x).sum() * dx, A44=e44.sum() * dx,
                  A46=(e24 * x).sum() * dx, A66=(e22 * x * x).sum() * dx,
                  A23=e23.sum() * dx, A34=e34.sum() * dx,
                  A36=(e23 * x).sum() * dx, B23=(SP * x * e23).sum() * dx,
                  B34=(SP * x * e34).sum() * dx,
                  B36=(SP * x * x * e23).sum() * dx)
        return SimpleNamespace(tau=np.array([Xk, Y, Z, K, Mq, N]), A=A_,
                               e22=e22, e23=e23, e33=e33, e24=e24, e34=e34,
                               e44=e44, fz=fz, fy=fy, fL=fL, Zsh=Zsh,
                               Rz=-0.5 * qd, P=Py, Pz=Pz, L=L, vr=vr, Wr=Wr,
                               Px=Px, aoy=aoy, aW0=aW0, M5=None)

    def _solve5(self, lat, m11, m12, m22_, r1, r2, tau_o):
        """(v', w', p', q', r') of switch lateral_cross: the vertical model's
        heave-pitch block (m + A33, A35, Iy + A55), the rigid sway, roll and
        yaw inertia, and the cross-plane sums (_lateral_x; the strip
        velocities v_r = v + x r, W_r = w + SP x q + ..., p) on the left;
        the right-hand sides tau_o[1], r1, tau_o[3], r2, tau_o[5]."""
        a = lat.A
        M = np.array([
            [self.m + a["A22"], a["A23"], a["A24"], a["B23"], a["A26"]],
            [a["A23"], m11, a["A34"], m12, a["A36"]],
            [a["A24"], a["A34"], self.Ix + a["A44"], a["B34"], a["A46"]],
            [a["B23"], m12, a["B34"], m22_, a["B36"]],
            [a["A26"], a["A36"], a["A46"], a["B36"], self.Iz + a["A66"]]])
        lat.M5 = M
        b = np.array([tau_o[1], r1, tau_o[3], r2, tau_o[5]])
        sol = np.linalg.solve(M, b)
        return (float(sol[0]), float(sol[1]), float(sol[2]), float(sol[3]),
                float(sol[4]))


class _ASMixin:
    """What the two as plants share: built only for an as hull, its preset of
    switches (or test_switches=), the measurements of the new flux, the
    steady-state and top-speed captive runs through the copied strip
    theory, with caches keyed by the physics stamp (code, hull manifest,
    switches), with_sea, and the actuators it was given carried to every
    plant rebuilt from it."""

    def __init__(self, hull, sea, dt=0.01, prop=None, rudder=None,
                 captive_u=None, test_switches=None, **kw):
        if "switches" in kw:
            raise TypeError(f"the as plants take {TEST_KW}= (tests only); a "
                            f"hull's switches are its version's preset")
        ver = physics.version(hull)
        if ver == physics.DEFAULT:
            raise ValueError(
                f"{type(self).__name__} is a plant of the as versions and "
                f"{hull.name!r} is a v0 hull; ask for it by key, e.g. "
                f"hydro.hulls.get('{hull.name}@as1')")
        physics.check_env(hull)
        check_v0_globals()
        self.physics = ver
        self.switches = resolve(ver, test_switches)
        self.sw = SimpleNamespace(**{k: k in self.switches
                                     for k in SWITCHES})
        # P1b's derived flags (blocks of the copied derivative that more
        # than one P1b switch opens): any P1b switch; the flux of the P1b
        # kinematics; the roll moment computed in the copy (the GZ twin's
        # swap bypassed, deriv below)
        sw = self.sw
        sw.p1b = (sw.fk_exact or sw.orbital or sw.wagner_entry
                  or sw.lateral_phi0)
        sw.p1b_flux = sw.orbital or sw.wagner_entry
        sw.roll_own = sw.fk_exact or sw.lateral_phi0
        if self.sw.dDt_full and getattr(hull, "transom_correction", True):
            # v0's Garme taper (pv:237-244) multiplies m_a by a C_V-indexed
            # factor; its x-derivative would enter D/Dt too. The design drops
            # it in as (review: 'delete it or index it by eps', S2); off on
            # every hull in the repo (hydro/planing.py default)
            raise ValueError(f"{hull.name!r}: transom_correction=True is not "
                             f"defined with the dDt_full switch")
        # what records made by this plant carry (sim/physics.stamp): the
        # code hash of the process, taken when the module was imported, the
        # hull's manifest and the switches -- taken BEFORE construction,
        # because the constructor's captive run (_design_lateral) already
        # keys the steady-state cache by it
        self._stamp = physics.stamp_for(hull, ver, self.switches)
        # the frozen GZ table cache is keyed by id(hull): keep the hull alive
        _PINNED[id(hull)] = hull
        # actuators passed in: pristine copies, handed (copied again) to
        # every plant rebuilt from this one (_rebuild_kw)
        self._ext = {k: copy.deepcopy(v) for k, v in
                     (("prop", prop), ("rudder", rudder)) if v is not None}
        super().__init__(hull, sea, dt=dt, prop=prop, rudder=rudder,
                         captive_u=captive_u, **kw)
        if self.switches != PRESETS[ver]:
            self._init_kw[TEST_KW] = self.switches

    # ------------------------------------------------------- measurements
    def _reset(self):
        super()._reset()
        # the sections' momentum-flux force with the complete D/Dt (S1),
        # reported beside v0's last_slam_force / peak_slam_force, which keep
        # v0's definition (entry flux dm_a/dd V^2 while V > 0); with
        # dDt_full off the two are the same
        self.last_slam_force_full = 0.0
        self.peak_slam_force_full = 0.0
        # P1b wagner_entry: the planes' state field (hidden state, module
        # docstring 'P1b WAGNER_ENTRY'): the field the last step used, the
        # state that step returned (the next step continues it only from
        # exactly that state), the step under way, the number of steps that
        # did not continue the last one (started from the steady field), and
        # the field each recorded step started from (record=True)
        self._es_prev = None
        self._es_s = None
        self._es_ur = None      # the inflow at _es_prev's time (P1c, Heun;
        #                         records without 'kin' only)
        # P1c review: the field's time and the kinematics at it and at the
        # field's previous time (the interpolation in time of the sub-steps,
        # _es_advance)
        self._es_t = None
        self._es_kin = None
        self._es_kinm = None
        self._es_run = None
        self.es_resets = 0
        self._es_log = []

    # ------------------------------------------- P1b: the hidden state
    def hidden_state(self):
        """The plant's state outside the 14-element vector that its next
        step depends on (switch wagner_entry: the planes' flow-state field;
        None without it): a dict with 'entry_state' -- (boundaries, states)
        of the field the last step used, or None before the first step --,
        's', the state the last step returned, 't', the field's time, and
        'kin' / 'kinm', the kinematics at the stations (d, vg, ur, dd, t)
        at the field's time and at the field's previous time (None after a
        start from the steady field): the start of the next transport and
        the interpolation in time of its sub-steps (P1c). A step from
        exactly that state continues the field; from any other state it
        starts from the steady field of that state. set_hidden_state
        restores it, so a run replayed from a recorded row
        (trajectory()['hidden'][k] with trajectory()['s'][k]) reproduces
        the original bit for bit."""
        if not self.sw.wagner_entry:
            return None
        prev, sref = self._es_prev, self._es_s

        def kc(k):
            return None if k is None else {
                key: (None if v is None else
                      (float(v) if key == "t" else np.array(v, float)))
                for key, v in k.items()}
        h = dict(entry_state=(None if prev is None else
                              (prev[0].copy(), prev[1].copy())),
                 s=None if sref is None else sref.copy(),
                 t=self.__dict__.get("_es_t"),
                 kin=kc(self.__dict__.get("_es_kin")),
                 kinm=kc(self.__dict__.get("_es_kinm")))
        if prev is not None and len(prev) > 2:
            # P1c round 3: the X planes' turn half-beams, per segment
            h["y"] = prev[2].copy()
        return h

    def set_hidden_state(self, h):
        """Restore what hidden_state() returned (None: nothing to restore;
        the next step starts from the steady field). Records made before
        the P1c review: without 't' the next advance covers the step's dt,
        without 'kin' it is one step (Heun's with 'ur', else the end-of-step
        inflow)."""
        if not self.sw.wagner_entry:
            if h is not None:
                raise ValueError("this plant has no hidden state "
                                 "(switch wagner_entry is off)")
            return
        self._es_t = self._es_kin = self._es_kinm = None
        if h is None or h.get("entry_state") is None:
            self._es_prev = self._es_s = self._es_ur = None
            return
        b, st = h["entry_state"]
        self._es_prev = (np.asarray(b, float).copy(),
                         np.asarray(st, np.int8).copy())
        if self._es_mem() is not None:
            # P1c round 3: the X planes' turn half-beams ('y'); a record
            # without them (before round 3): +inf, Zarnick's X as then
            y = h.get("y")
            self._es_prev = self._es_prev + (
                np.full(self._es_prev[1].size, np.inf) if y is None
                else np.asarray(y, float).copy(),)
        self._es_s = np.asarray(h["s"], float).copy()

        def kc(k):
            return None if k is None else {
                key: (None if v is None else
                      (float(v) if key == "t" else
                       np.asarray(v, float).copy()))
                for key, v in k.items()}
        self._es_kin = kc(h.get("kin"))
        self._es_kinm = kc(h.get("kinm"))
        t = h.get("t")
        self._es_t = None if t is None else float(t)
        ur = h.get("ur")
        if ur is None and self._es_kin is not None:
            ur = self._es_kin["ur"]
        self._es_ur = None if ur is None else np.asarray(ur, float).copy()

    def deriv(self, s, t):
        if self.sw.heel_hydro and isinstance(self, PlaningVesselGZ):
            # S5a: the copy computes the roll moment of the heeled sections
            # for both plants; on its upright branch (no heel, no transverse
            # slope) the moment is P1c's: _roll_b in the copy, or the GZ
            # twin's swap of the roll row, here in fast_path's form (the
            # same expressions on the same numbers as PlaningVesselGZ.deriv)
            ds, aux = _StripAS.deriv(self, s, t)
            if not aux["heeled"] and not (self.sw.roll_own
                                          and self._roll_own()):
                rel = s[3] - aux["wbar"]
                ds[9] += self.m * G * (self.GM * rel - self.gz(rel)) \
                    / self.Mtot[3, 3]
            out = ds, aux
        elif self.sw.roll_own and self._roll_own() and \
                isinstance(self, PlaningVesselGZ):
            # P1b: the copy computes the whole roll moment, the GZ restoring
            # included (_StripAS._roll_b); the frozen GZ twin's swap of the
            # roll row (which divides by the fixed Mtot[3, 3] and recomputes
            # v0's slope) is not applied
            out = _StripAS.deriv(self, s, t)
        elif self.sw.fast_path and isinstance(self, PlaningVesselGZ):
            # P1c: PlaningVesselGZ.deriv's swap of the roll row, its
            # expressions, on the copy's own section area and wave-slope
            # mean (the same numbers: the GZ twin recomputes A from aux['d']
            # and s_tr from the same station slopes)
            ds, aux = _StripAS.deriv(self, s, t)
            rel = s[3] - aux["wbar"]
            ds[9] += self.m * G * (self.GM * rel - self.gz(rel)) \
                / self.Mtot[3, 3]
            out = ds, aux
        else:
            out = super().deriv(s, t)
        if getattr(self, "_aux0", 0) is None:   # the first one of a step
            self._aux0 = out[1]
        return out

    def prop_submergence(self, eta, t):
        """v0's PlaningVessel.prop_submergence (through PlaningVesselGZ's,
        which records last_submergence); with switch fast_path its own
        operations with the surface elevation alone (v0 evaluates all five
        wave quantities at the intake and keeps the first)."""
        if not self.sw.fast_path:
            return super().prop_submergence(eta, t)
        xi, yi, zi = self.intakes[0]
        c, s_ = np.cos(eta[5]), np.sin(eta[5])
        X, Y = eta[0] + xi * c - yi * s_, eta[1] + xi * s_ + yi * c
        if not hasattr(self, "_wa"):
            self._wave_arrays()
        if self._wa is None:
            surf = 0.0
        else:
            ph = (np.outer(np.atleast_1d(X), self._wkx)
                  + np.outer(np.atleast_1d(Y), self._wky)
                  - self._ww * t + self._wphi)
            surf = (np.cos(ph) @ self._wa[0])[0]
        hull_z = (self.Z_rest + eta[2] + zi + SP * xi * (self.Th_rest
                                                         + eta[4])
                  + yi * eta[3])
        sub = float(surf - hull_z)
        if isinstance(self, PlaningVesselGZ):
            self.last_submergence = sub
        return sub

    def step(self, s, t, thrust_cmd, rudder_cmd, dt=None):
        """The v0 step (PlaningVessel's, through PlaningVesselGZ's for the
        GZ plant), plus last_slam_force_full and peak_slam_force_full
        measured, like v0's slam force, from the derivative at the start of
        the step; with wagner_entry the planes' state field advanced at the
        step's first derivative (not when the GZ plant has capsized: its
        step then returns without a derivative)."""
        check_v0_globals()
        self._aux0 = None
        es = self.sw.wagner_entry and not getattr(self, "capsized", False)
        if es:
            prev, sref = self._es_prev, self._es_s
            cont = (prev is not None and sref is not None
                    and np.array_equal(s, sref))
            if prev is not None and not cont:
                self.es_resets += 1
            if getattr(self, "record", False):
                self._es_log.append(self.hidden_state() if cont else None)
            self._es_run = dict(mode="advance" if cont else "init",
                                dt=float(self.dt if dt is None else dt),
                                cur=None)
        try:
            s_new = super().step(s, t, thrust_cmd, rudder_cmd, dt)
        finally:
            run, self._es_run = self._es_run, None
        if es and run is not None and run["cur"] is not None:
            self._es_prev = run["cur"]
            self._es_kin, self._es_kinm = run["kin"], run["km"]
            self._es_t = run["t0"]
            self._es_ur = run["kin"]["ur"]
            self._es_s = np.array(s_new, float)
        a, self._aux0 = self._aux0, 0
        if a is not None:
            f = float(a["mom_full"].sum() * self.dx)
            self.last_slam_force_full = f
            self.peak_slam_force_full = max(self.peak_slam_force_full, f)
        return s_new

    def _rebuild_kw(self, **overrides):
        """The constructor keywords of a plant rebuilt from this one
        (with_sea, physics.gz_twin): v0's _init_kw, the overrides, and fresh
        deep copies of the actuators this plant was given (v0 drops them).
        A rebuild at another step size is refused while it carries given
        actuators: their delay lines were built for this one."""
        kw = dict(self._init_kw)
        kw.update(overrides)
        kw.pop("wind", None)
        if self._ext and float(kw.get("dt", self.dt)) != float(self.dt):
            raise ValueError(f"this plant was given {sorted(self._ext)}, "
                             f"built for dt = {self.dt}; rebuilding it at dt "
                             f"= {kw['dt']} needs actuators for that step")
        for k, v in self._ext.items():
            if kw.get(k) is None:
                kw[k] = copy.deepcopy(v)
        return kw

    def with_sea(self, sea, **overrides):
        """The same plant in another sea (v0's PlaningVessel.with_sea
        hard-codes PlaningVessel; this keeps the class, the switches and
        the actuators it was given)."""
        kw = dict(self._init_kw)
        kw.update(overrides)
        kw.pop("wind", None)
        kw = self._rebuild_kw(**kw)
        return type(self)(self.hull, sea, **kw)

    # --------------------------------------------------------- steady state
    def running_attitude(self, u):
        """(z, pitch) at steady speed u in calm water, from rest by a
        captive run (surge held, a force at the nozzle holding it) -- v0's,
        with the captive copy this plant's strip theory, its step, iteration
        count and tolerances the hull's (self.num), and the cache keyed by
        the physics stamp (code, hull manifest, switches) instead of name and
        id(hull)."""
        check_v0_globals()
        key = (self._stamp["physics_sha1"], float(u),
               getattr(self.hull, "transom_correction", True),
               self.k_a, self.hull.a_b, self.c_dc)
        hit = _STEADY.get(key)
        if hit is not None:
            return hit
        if u <= 0.0:
            _STEADY[key] = (0.0, 0.0)
            return 0.0, 0.0
        from sim.test_vessel import Monochromatic
        v = _StripAS.__new__(_StripAS)
        v.__dict__.update(self.__dict__)
        v.sea = Monochromatic(1.0, 0.0)
        v._wave_arrays()                 # the calm copy's own (none)
        v.captive_u = float(u)
        s = np.zeros(14)
        s[6] = u
        dt = self.num.dt_captive
        last = None
        for i in range(self.num.n_captive):
            k1, _ = v.deriv(s, 0.0)
            s2 = s + dt * k1
            k2, _ = v.deriv(s2, 0.0)
            s = s + 0.5 * dt * (k1 + k2)
            s[6] = u
            if i % self.num.n_check == self.num.n_check - 1:
                cur = (s[2], s[4])
                if last is not None and abs(cur[0] - last[0]) < \
                        self.num.tol_z and abs(cur[1] - last[1]) < \
                        self.num.tol_th:
                    break
                last = cur
        _STEADY[key] = (float(s[2]), float(s[4]))
        return _STEADY[key]

    def captive_deriv(self, u, z, th):
        """The captive equations at (u, z, th) in calm water, as
        running_attitude and calm_resistance integrate them (a calm copy of
        this plant's strip theory, the surge held by a force at the nozzle):
        (ds, aux) of its deriv. ds[8], ds[10] are w', q' -- zero at an
        equilibrium (sim/vessel_profile.steady's residuals)."""
        check_v0_globals()
        from sim.test_vessel import Monochromatic
        v = _StripAS.__new__(_StripAS)
        v.__dict__.update(self.__dict__)
        v.sea = Monochromatic(1.0, 0.0)
        v._wave_arrays()                 # the calm copy's own (none)
        v.captive_u = float(u)
        s = np.zeros(14)
        s[6], s[2], s[4] = u, z, th
        return v.deriv(s, 0.0)

    def calm_resistance(self, u):
        """Total calm-water resistance at steady speed u (N), with the
        breakdown, at the running attitude -- v0's, through this plant's
        strip theory."""
        check_v0_globals()
        from sim.test_vessel import Monochromatic
        z, th = self.running_attitude(u)
        v = _StripAS.__new__(_StripAS)
        v.__dict__.update(self.__dict__)
        v.sea = Monochromatic(1.0, 0.0)
        v._wave_arrays()                 # the calm copy's own (none)
        v.captive_u = float(u)
        s = np.zeros(14)
        s[6], s[2], s[4] = u, z, th
        _, aux = v.deriv(s, 0.0)
        return dict(R=aux["fx"], pressure=-aux["Xp"], friction=aux["Df"],
                    air=aux["Da"], trim_deg=float(-np.degrees(
                        self.Th_rest + th)), rise=z, wetted_length=aux["lw"])

    def top_speed(self):
        """Calm-water speed at full thrust, m/s: where the jet's net thrust
        at t_max equals the calm resistance at the running attitude --
        PlaningVesselGZ's, its tolerance the hull's (self.num.u_tol), cached
        by the physics stamp."""
        key = (self._stamp["physics_sha1"], float(self.prop.t_max))
        if key in _TOP:
            return _TOP[key]
        from scipy.optimize import brentq
        prime, self.prop.prime = self.prop.prime, 1.0
        t_max = self.prop.t_max

        def excess(u):
            return (self.prop.forces(t_max, 0.0, u)[0]
                    - self.calm_resistance(u)["R"])
        try:
            lo = float(self.hull.u_design)
            hi = 1.5 * lo
            while excess(hi) > 0.0 and hi < 6.0 * lo:
                lo, hi = hi, 1.3 * hi
            u = float(brentq(excess, lo, hi, xtol=self.num.u_tol))
        finally:
            self.prop.prime = prime
        _TOP[key] = u
        return u


class PlaningVesselAS(_ASMixin, PlaningVesselGZ, _StripAS):
    """PlaningVesselGZ's drop-in on an as physics version: the large-angle
    plant (GZ roll, capsize, record/trajectory) whose strip theory is
    _StripAS, with the design steps' switches. Same constructor, plus
    test_switches (None: the hull version's preset; a preset name; or
    switch names -- tests only). Built for a hull whose key names an as
    version, by physics.gz_twin / physics.gz_plant_for."""

    def gz_geometric(self, phi):
        """PlaningVesselGZ.gz_geometric with the sinkage bracket scaled to
        the hull. The frozen method bisects the heeled waterline shift in
        +-3 m, an absolute bound that fails silently once a hull's heeled
        sections reach beyond it (a family member several times
        scarab195's size). The immersed volume goes from zero to the whole
        section as the waterline height zw + delta passes from -r to +r,
        r the largest distance of a section vertex from the CG at any heel,
        so |delta| <= r + max |zw| brackets the root. Whenever +-3 m covers
        that bound (every hull up to a few times scarab195; scarab195's is
        ~1.5 m) the frozen method itself runs, bit for bit; beyond it the
        same bisection runs in the scaled bracket, with as many more
        halvings as keep its resolution (6 m / 2^50)."""
        py, pz, zw = self._sections()
        S = float(np.sqrt(py * py + pz * pz).max() + np.abs(zw).max())
        if S <= 3.0:
            return super().gz_geometric(phi)
        from sim.planing_vessel_gz import _immersed
        phi = np.atleast_1d(np.asarray(phi, float))
        dx = self.dx
        v0 = _immersed(py, pz, zw)[0].sum() * dx
        c, s = np.cos(phi)[:, None, None], np.sin(phi)[:, None, None]
        Y, Z = py * c - pz * s, py * s + pz * c
        S = 1.01 * S
        lo = np.full(phi.size, -S)
        hi = np.full(phi.size, S)
        for _ in range(50 + int(np.ceil(np.log2(S / 3.0)))):
            mid = 0.5 * (lo + hi)
            v = _immersed(Y, Z, zw + mid[:, None])[0].sum(-1) * dx
            over = v > v0
            hi, lo = np.where(over, mid, hi), np.where(over, lo, mid)
        delta = 0.5 * (lo + hi)
        if np.any(np.abs(delta) > 0.999 * S):
            raise RuntimeError("gz_geometric: the heeled sinkage reached its "
                               "bracket")
        H = zw + delta[:, None]
        A, M = _immersed(Y, Z, H)
        gz = -M.sum(-1) / A.sum(-1)
        margin = (Z[..., 3] - H).min(-1)
        return gz, delta, margin, v0

    def trajectory(self, u_max=None):
        """PlaningVesselGZ.trajectory, its meta stamped with the plant's
        physics (sim/physics.stamp: version, switches, hull key, sha1),
        taken when the plant was built; warns if the code on disk has
        changed since this process imported it."""
        out = super().trajectory(u_max)
        out["meta"].update(physics.stamp(self))
        if self.sw.wagner_entry:
            # P1b: the hidden state each recorded step started from (row k:
            # hidden_state() before the step from s[k]; None where the step
            # started from the steady field), for replay from any row
            out["hidden"] = list(self._es_log)
            out["meta"]["hidden_state"] = "entry_state (switch wagner_entry)"
        if physics.code_changed(self.physics):
            warnings.warn(f"the {self.physics} code on disk changed after "
                          f"this process imported it; the record carries the "
                          f"hash of the code that ran")
        return out


class PlaningVesselASLin(_ASMixin, _StripAS):
    """PlaningVessel's drop-in on an as physics version: linear roll, as v0's
    hull.plant() builds, with the copied strip theory and the switches --
    what PlaningHull.plant() builds for an as hull. Its GZ twin is
    PlaningVesselAS (physics.gz_twin)."""
