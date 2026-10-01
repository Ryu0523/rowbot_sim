#!/usr/bin/env python3
"""
The waterjet model against public data (sim/waterjet.py, hydro/planing.py).

V1  THRUST vs SPEED at constant power: HamiltonJet unit "291" (datasheet,
    Oct 1984, reproduced as Fig. 2 of MacPherson, "Selection of Commercial
    Waterjets", SNAME New England 2000), eight power levels 50-400 hp.
    Digitised by eye from the chart (+-10 kgf); the dashed high-power,
    low-speed part (cavitation-limited, the chart's own marking) left out.
    The momentum model's two unknowns -- effective nozzle area and pump
    efficiency -- are fitted on ONE curve (200 hp) and the other seven are
    PREDICTED. What is tested is the model's shape: how thrust falls with
    speed and grows with power. Absolute thrust is not (the unit's
    diameters are not given).
V2  our Savitsky implementation against the worked example in Alourdas
    (2016, SNAME Greek Section notes; Shoemaker's NACA Model 29 test). Checks
    the CODE, and the example's own tank data show Savitsky's accuracy.
V3  a jet boat of the VM 18's class: Scarab 195 ID (5.74 m, 1116 kg dry,
    20 deg deadrise, the same Rotax 300 ACE, 161 mm pump), Boating magazine's
    measured speeds. Jet model + Savitsky + air drag, with the unknowns
    (chine beam, LCG, nozzle diameter, pump efficiency, windage) sampled over
    stated ranges. Full throttle is the clean point (300 hp at ~8000 rpm);
    part throttle needs power from fuel burn and an assumed specific fuel
    consumption, so it is a looser check.
V4  what the same model, with the same unknown ranges, says about VM 18.

Run: python -m studies.exp_waterjet
"""
import numpy as np
from scipy.optimize import least_squares

from hydro.planing import savitsky
from sim.waterjet import JetPump

KN = 0.514444
HP = 745.7
KGF = 9.80665
LB = 0.45359237
GAL = 3.78541
GASOLINE = 0.74            # kg/L

# ------------------------------------------------------------------ data
# V1 -- HamiltonJet "291": kgf at knots, per hp. Dashed (cavitation-limited)
# points are omitted; '~' points (label hides the line) kept.
# https://hydrocompinc.com/wp-content/uploads/documents/WaterjetPerfCoefs.pdf
HJ291 = {
    50: [(0, 408), (5, 352), (10, 298), (15, 246), (20, 199)],
    100: [(0, 654), (5, 579), (10, 512), (15, 445), (20, 386), (25, 330),
          (30, 272), (35, 213), (40, 159)],
    150: [(0, 847), (5, 768), (10, 695), (15, 619), (20, 548), (25, 479),
          (30, 408), (35, 338), (40, 275)],
    200: [(0, 1036), (5, 937), (10, 855), (15, 773), (20, 693), (25, 615),
          (30, 538), (35, 459), (40, 390)],
    250: [(5, 1063), (10, 1009), (15, 931), (20, 840), (25, 750), (30, 662),
          (35, 580), (40, 503)],
    300: [(15, 1056), (20, 963), (25, 874), (30, 786), (35, 698), (40, 608)],
    350: [(15, 1200), (20, 1110), (25, 1002), (30, 903), (35, 805),
          (40, 713)],
    400: [(20, 1223), (25, 1125), (30, 1015), (35, 911), (40, 811)],
}

# V2 -- Alourdas (2016): NACA Model 29, beam 16.0 in, deadrise 20 deg,
# 80 lb. (kn, LCG in) -> Savitsky (lam, trim deg, drag lb), tank (same)
# https://higherlogicdownload.s3.amazonaws.com/SNAME/a09ed13c-b8c0-4897-9e87-eb86f500359b/UploadedImages/2016-2017/Alourdas'%20Complimentary%20Notes.pdf
ALOURDAS = [
    dict(kn=18.01, lcg_in=18.5, sav=(1.59, 4.21, 15.12), tank=(2.31, 4.0, 15.20)),
    dict(kn=20.92, lcg_in=18.3, sav=(1.56, 3.46, 16.76), tank=(1.81, 4.0, 16.0)),
]

# V3 -- Scarab 195 ID, Boating magazine certified tests. rpm: (kn, gal/h).
# Test loads: 2024 700 lb crew + 20 gal; 2019 400 lb + 15 gal.
# https://boatingmag.com/boats/2024-scarab-jet-195-id-boat-test/
# https://boatingmag.com/2019-scarab-195-id-wake-edition/
SCARAB = dict(L=5.74, beam=2.44, dry=1116.0, deadrise=20.0)
SCARAB_RUNS = {
    "2024": dict(load=700 * LB + 20 * GAL * GASOLINE,
                 pts={6000: (22.33, 10), 6500: (28.24, 12), 7000: (32.67, 15),
                      7500: (37.80, 19), 7920: (42.06, 23)}),
    "2019": dict(load=400 * LB + 15 * GAL * GASOLINE,
                 pts={6000: (24.8, 10), 6500: (29.5, 12), 7222: (34.0, 15),
                      7500: (39.1, 19), 8000: (42.7, 23)}),
}
P_RATED = 300 * HP          # Rotax 1630 ACE 300, max rpm 8000 (spec sheet)

# ------------------------------------------------------------ unknowns
# Everything below is an assumption, sampled uniformly; the ranges are
# the finding's error bars.
UNKNOWN = dict(
    chine_frac=(0.78, 0.88),     # chine beam / overall beam, runabouts
    lcg_frac=(0.34, 0.42),       # LCG from transom / L, inboard jet
    d_nozzle=(0.080, 0.090),     # m; aftermarket rings for this pump 82-84 mm
    eta_pump=(0.70, 0.85),
    cda_scarab=(1.2, 2.2),       # m^2, C_D x frontal area, open runabout
)


def air_drag(V, cda, rho_air=1.225):
    return 0.5 * rho_air * cda * V * V


def top_speed(pump, power, mass, lcg, b, beta, cda, v_hi=40.0):
    """Steady speed where jet thrust = Savitsky + air drag; None if the
    boat does not plane (thrust short at the start of planing)."""
    from scipy.optimize import brentq
    v_lo = 1.8 * np.sqrt(9.81 * b)

    def surplus(V):
        return (pump.thrust(power, V)
                - savitsky(V, mass, lcg, b, beta)["R"] - air_drag(V, cda))
    if surplus(v_lo) <= 0:
        return None
    if surplus(v_hi) > 0:
        return v_hi
    return brentq(surplus, v_lo, v_hi)


# ------------------------------------------------------------------ V1
def v1():
    print("\nV1  HamiltonJet '291' thrust vs speed at constant power")
    print("    fitted on 200 hp: effective nozzle area and pump efficiency;"
          "\n    other seven power levels predicted")

    def model(params, hp, kn):
        a, eta = params
        pump = JetPump(a, hp * HP, eta_pump=eta)
        return np.array([pump.thrust(hp * HP, k * KN) for k in kn]) / KGF

    kn, t = np.array(HJ291[200]).T
    fit = least_squares(lambda p: (model(p, 200, kn) - t) / t,
                        x0=(0.02, 0.7), bounds=([1e-3, 0.3], [0.2, 1.0]))
    a, eta = fit.x
    print(f"    fit: nozzle d {np.sqrt(4 * a / np.pi) * 1000:.0f} mm, "
          f"pump efficiency {eta:.2f}")
    print(f"    {'hp':>5}{'points':>8}{'rms err':>9}{'max err':>9}"
          f"{'static (chart / model)':>25}")
    errs = []
    for hp, pts in HJ291.items():
        kn, t = np.array(pts).T
        m = model(fit.x, hp, kn)
        e = (m - t) / t * 100
        if hp != 200:
            errs.extend(e)
        s0 = (f"{t[0]:.0f} / {m[0]:.0f} kgf" if kn[0] == 0 else "")
        print(f"    {hp:>5}{len(pts):>8}{np.sqrt(np.mean(e ** 2)):>8.1f}%"
              f"{np.abs(e).max():>8.1f}%{s0:>25}"
              + ("   <- fitted" if hp == 200 else ""))
    errs = np.array(errs)
    print(f"    predicted curves: rms {np.sqrt(np.mean(errs ** 2)):.1f}%, "
          f"worst {np.abs(errs).max():.1f}% over {len(errs)} points")
    # momentum theory's own scaling of bollard pull: T0 ~ P^(2/3)
    t0 = {hp: dict(pts)[0] for hp, pts in HJ291.items() if dict(pts).get(0)}
    for h1, h2 in ((50, 100), (100, 200), (50, 200)):
        print(f"    static thrust {h2}/{h1} hp: chart {t0[h2] / t0[h1]:.3f}, "
              f"momentum theory (P ratio)^(2/3) {(h2 / h1) ** (2 / 3):.3f}")
    return fit.x


# ------------------------------------------------------------------ V2
def v2():
    print("\nV2  Savitsky implementation vs Alourdas' worked example "
          "(NACA Model 29)")
    print(f"    {'kn':>6}{'':>4}{'lam':>7}{'trim':>7}{'drag lb':>9}")
    for c in ALOURDAS:
        r = savitsky(c["kn"] * KN, 80 * LB, c["lcg_in"] * 0.0254,
                     16.0 * 0.0254, 20.0, rho=999.1, nu=1.14e-6, dcf=0.0)
        ours = (r["lam"], r["tau_deg"], r["R"] / KGF / LB)
        for nm, v in (("ours", ours), ("pub.", c["sav"]),
                      ("tank", c["tank"])):
            print(f"    {c['kn']:>6.2f}  {nm:<4}{v[0]:>6.2f}{v[1]:>6.2f}"
                  f"{v[2]:>9.2f}")
        d = (ours[2] - c["sav"][2]) / c["sav"][2] * 100
        print(f"            ours vs published Savitsky: drag {d:+.1f}%; "
              f"published Savitsky vs tank: drag "
              f"{(c['sav'][2] - c['tank'][2]) / c['tank'][2] * 100:+.1f}%")


# ------------------------------------------------------------------ V3/V4
def sample(n, rng, keys):
    return [{k: rng.uniform(*UNKNOWN[k]) for k in keys} for _ in range(n)]


def v3(n=300, seed=0):
    print("\nV3  Scarab 195 ID (VM 18's near-twin): measured vs jet model + "
          "Savitsky + air drag")
    rng = np.random.default_rng(seed)
    S = sample(n, rng, ("chine_frac", "lcg_frac", "d_nozzle", "eta_pump",
                        "cda_scarab"))
    L, beam, beta = SCARAB["L"], SCARAB["beam"], SCARAB["deadrise"]
    for yr, run in SCARAB_RUNS.items():
        mass = SCARAB["dry"] + run["load"]
        rpm_top = max(run["pts"])
        kn_top = run["pts"][rpm_top][0]
        # full throttle: rated power scaled by rpm/8000 near the top
        p_top = P_RATED * min(rpm_top / 8000.0, 1.0)
        v = []
        for s in S:
            pump = JetPump(np.pi * s["d_nozzle"] ** 2 / 4, p_top,
                           eta_pump=s["eta_pump"])
            vt = top_speed(pump, p_top, mass, s["lcg_frac"] * L,
                           s["chine_frac"] * beam, beta, s["cda_scarab"])
            if vt is not None:
                v.append(vt / KN)
        v = np.array(v)
        lo, med, hi = np.percentile(v, [10, 50, 90])
        print(f"    {yr} test, {mass:.0f} kg, {rpm_top} rpm: measured "
              f"{kn_top:.1f} kn; model median {med:.1f} kn "
              f"(80% of assumptions: {lo:.1f}-{hi:.1f})  "
              f"[{len(v)}/{n} plane]")
    # part throttle: power from fuel flow at an assumed specific consumption
    print("    part throttle (power = fuel flow / BSFC; BSFC 0.29 kg/kWh at "
          "full throttle\n    from the 23 gal/h at 300 hp, 0.29-0.40 at part"
          " load, assumed):")
    s_mid = {k: np.mean(UNKNOWN[k]) for k in UNKNOWN}
    run = SCARAB_RUNS["2024"]
    mass = SCARAB["dry"] + run["load"]
    for rpm, (kn, gph) in sorted(run["pts"].items()):
        fuel = gph * GAL * GASOLINE                        # kg/h
        band = []
        for bsfc in (0.29, 0.40):
            p = min(fuel / bsfc * 1e3, P_RATED)
            pump = JetPump(np.pi * s_mid["d_nozzle"] ** 2 / 4, p,
                           eta_pump=s_mid["eta_pump"])
            vt = top_speed(pump, p, mass, s_mid["lcg_frac"] * L,
                           s_mid["chine_frac"] * beam, beta,
                           s_mid["cda_scarab"])
            band.append(vt / KN if vt else float("nan"))
        print(f"      {rpm} rpm, {gph:>2} gal/h: measured {kn:5.1f} kn; "
              f"model {min(band):5.1f}-{max(band):5.1f} kn")


def v4(n=300, seed=1):
    print("\nV4  VM 18 (brochure: 5.77 m, beam 1.67 m, wet 1206 kg, CG 2.67 m "
          "from aft, 300 hp;\n    claims 50+ kn, 20 kn for 26 h). Deadrise "
          "20 deg assumed (as Scarab); windage C_D A 0.8-1.4 m^2")
    rng = np.random.default_rng(seed)
    S = sample(n, rng, ("chine_frac", "d_nozzle", "eta_pump"))
    for s in S:
        s["cda"] = rng.uniform(0.8, 1.4)
    for label, mass in (("wet", 1206.0), ("wet + 453 kg payload", 1659.0),
                        ("max 1815 kg", 1815.0)):
        v = []
        for s in S:
            pump = JetPump(np.pi * s["d_nozzle"] ** 2 / 4, P_RATED,
                           eta_pump=s["eta_pump"])
            vt = top_speed(pump, P_RATED, mass, 2.67,
                           s["chine_frac"] * 1.67, 20.0, s["cda"],
                           v_hi=45.0)
            if vt is not None:
                v.append(vt / KN)
        lo, med, hi = np.percentile(v, [10, 50, 90])
        print(f"    top speed, {label:<21}: median {med:4.1f} kn "
              f"(80%: {lo:4.1f}-{hi:4.1f})")
    # cruise: power to hold 20 kn, and the fuel for 26 h of it
    p20 = []
    for s in S:
        V = 20 * KN
        b = s["chine_frac"] * 1.67
        R = savitsky(V, 1206.0, 2.67, b, 20.0)["R"] + air_drag(V, s["cda"])
        pump = JetPump(np.pi * s["d_nozzle"] ** 2 / 4, P_RATED,
                       eta_pump=s["eta_pump"])
        p20.append(pump.power_for_thrust(R, V))
    p20 = np.array(p20) / 1e3
    lo, med, hi = np.percentile(p20, [10, 50, 90])
    fuel = med * np.array([0.35, 0.45]) * 26 / GASOLINE
    print(f"    20 kn at 1206 kg: shaft power median {med:.0f} kW "
          f"(80%: {lo:.0f}-{hi:.0f}); 26 h of it at BSFC 0.35-0.45 kg/kWh "
          f"= {fuel[0]:.0f}-{fuel[1]:.0f} L of fuel\n    (dry to wet is "
          f"226 kg, i.e. at most ~{226 / GASOLINE:.0f} L)")


def main():
    v1()
    v2()
    v3()
    v4()


if __name__ == "__main__":
    main()
