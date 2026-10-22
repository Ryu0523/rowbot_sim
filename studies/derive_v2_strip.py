#!/usr/bin/env python3
"""
V2 (corrected definition, coordinator note of 2026-10-09): the composite
F_c = L_p(delta) + [F(delta; 0) - L_0(delta)] omits the DIFFERENCE between
the 3-D problem's second-order response a_2(p) and the strip model's own
second-order response a_2(0) to the same inputs. This script computes the
strip side: the ZR/MLM dynamic section force (added mass x relative
acceleration, momentum flux, cross-flow; NO buoyancy -- F_HF is exact in
both models and excluded on both sides) as assembled in
studies/derive_v1_linear_consistency.py (read-only import), driven by a
prescribed heave or pitch motion and a regular head wave through the
sections (wave elevation, vertical orbital velocity and acceleration at
the section, as sim/planing_zarnick.ZarnickVessel.deriv uses them), and
extracts the second-order part at the sum and difference frequencies by
the double-input combination F(a, A) - F(a, 0) - F(0, A) + F(0, 0) in the
time domain, projected onto e^{i Omega t} over a common period.

Conventions match studies/derive_v2_second_order.py: heave z = a cos(w_i t)
(up), pitch alpha positive bow DOWN (the plant's th = -alpha), head wave
e(x, t) = A cos(w_j t + k x) travelling toward -x; heave force positive up,
pitch moment about +y (the plant's My = -sum x f dx).

Then the comparison with the 3-D second-order forces of v2_results_quick.json
(base case; quad - quad_hf + pot = the 3-D dynamic second-order response with
the exact hydrostatic + Froude-Krylov second-order part removed).

Run: cd repo && .venv/Scripts/python.exe -m studies.derive_v2_strip
Outputs: studies/_cache/derive/v2_strip.json, v2_strip.txt, v2_a2_compare.txt
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_cache", "derive")
G0 = 9.81
SP = -1.0
OMEGAS = (3.0, 3.9, 4.7)
AMPS_SMALL = dict(heave=0.01, pitch=0.01, wave=0.01)
AMPS_OPER = dict(heave=0.10, pitch=0.05, wave=0.30)
T_WIN = 20 * np.pi            # common period of all frequencies that are multiples of 0.1 rad/s
DT = 0.002


def strip_force(v, ctr, z, w, wd, th, q, qd, e, et, ett):
    """Dynamic section force (N/m) per time step and station, arrays (nt,) for the motion and (nt, n) for the
    wave; returns (Fz, My) time series (N, N m) and the per-term dict."""
    x = v.x[None, :]
    Th = v.Th_rest + th[:, None]
    d = e - (v.Z_rest + z[:, None] + v.zk[None, :] + SP * x * Th)
    V = et - (w[:, None] + SP * x * q[:, None])
    A, c, ma, dma, girth = v._section(d)
    ma = ma * ctr
    dma = dma * ctr
    mom = np.where(V > 0.0, dma * V * V, 0.0)
    cf = 0.5 * v.rho * v.c_dc * 2.0 * c * V * np.abs(V) * ctr
    iner = ma * (ett - (wd[:, None] + SP * x * qd[:, None]))
    f = iner + mom + cf
    Fz = f.sum(1) * v.dx
    My = (SP * x * f).sum(1) * v.dx
    return Fz, My, dict(iner=iner, mom=mom, cf=cf)


def inputs(v, t, motion, a, wi, A, wj):
    """Prescribed motion and wave arrays."""
    nt, n = len(t), len(v.x)
    z = w = wd = th = q = qd = np.zeros(nt)
    if motion == "heave" and a:
        z, w, wd = a * np.cos(wi * t), -a * wi * np.sin(wi * t), -a * wi * wi * np.cos(wi * t)
    elif motion == "pitch" and a:
        th, q, qd = -a * np.cos(wi * t), a * wi * np.sin(wi * t), a * wi * wi * np.cos(wi * t)
    if A:
        k = wj * wj / G0
        ph = wj * t[:, None] + k * v.x[None, :]
        e, et, ett = A * np.cos(ph), -A * wj * np.sin(ph), -A * wj * wj * np.cos(ph)
    else:
        e = et = ett = np.zeros((nt, n))
    return z, w, wd, th, q, qd, e, et, ett


def project(t, M, Om):
    """Complex amplitude X with M(t) ~ Re[X e^{i Om t}] (Om > 0), or the mean (Om = 0)."""
    if Om < 1e-9:
        return complex(np.mean(M))
    return complex(2.0 * np.mean(M * np.exp(-1j * Om * t)))


def run_strip(mlm, say):
    from studies.derive_v1_linear_consistency import make_vessel
    v = make_vessel(mlm, 0.0)
    ctr = np.broadcast_to(np.asarray(v._ctr(0.0), float), v.x.shape)
    t = np.arange(0.0, T_WIN, DT)
    label = "MLM" if mlm else "ZR (Wagner)"
    say(f"\n== strip {label}: {len(v.x)} stations, window {T_WIN:.1f} s, dt {DT} s ==")

    def FM(motion, a, wi, A, wj):
        Fz, My, _ = strip_force(v, ctr, *inputs(v, t, motion, a, wi, A, wj))
        return np.stack([Fz, My], 1)
    res = dict(label=label, mlm=bool(mlm), pairs=[])
    t0 = time.time()
    for wi in OMEGAS:
        for wj in OMEGAS:
            for u in ("heave", "pitch"):
                for key, amps in (("small", AMPS_SMALL), ("half", {k: 0.5 * x for k, x in AMPS_SMALL.items()}),
                                  ("oper", AMPS_OPER)):
                    a_, A_ = amps[u], amps["wave"]
                    M = FM(u, a_, wi, A_, wj) - FM(u, a_, wi, 0.0, wj) - FM(u, 0.0, wi, A_, wj)
                    for diff in (False, True):
                        Om = abs(wi - wj) if diff else wi + wj
                        X = np.array([project(t, M[:, c], Om) for c in range(2)])
                        res["pairs"].append(dict(kind="mixed", motion=u, w_motion=wi, w_wave=wj, diff=diff, Omega=Om,
                                                 amps=key, a=a_, A=A_, F=[X[0], X[1]], F_unit=[X[0] / (a_ * A_), X[1] / (a_ * A_)]))
        for u in ("heave", "pitch", "wave"):
            for key, amps in (("small", AMPS_SMALL), ("oper", AMPS_OPER)):
                a_ = amps[u]
                M = FM(u, a_, wi, 0.0, wi) if u != "wave" else FM("heave", 0.0, wi, a_, wi)
                for diff in (False, True):
                    Om = 0.0 if diff else 2 * wi
                    X = np.array([project(t, M[:, c], Om) for c in range(2)])
                    res["pairs"].append(dict(kind=u + "x" + u, motion=u, w_motion=wi, w_wave=wi, diff=diff, Omega=Om,
                                             amps=key, a=a_, A=a_, F=[X[0], X[1]], F_unit=[X[0] / a_ ** 2, X[1] / a_ ** 2]))
    res["t_wall"] = time.time() - t0
    say(f"  {len(res['pairs'])} projections in {res['t_wall']:.1f} s")
    # quadratic-scaling check of the small-amplitude coefficients (0.5 cm vs 1 cm)
    worst = 0.0
    for p in res["pairs"]:
        if p["kind"] == "mixed" and p["amps"] == "small":
            q = [r for r in res["pairs"] if r["kind"] == "mixed" and r["amps"] == "half" and r["motion"] == p["motion"]
                 and r["w_motion"] == p["w_motion"] and r["w_wave"] == p["w_wave"] and r["diff"] == p["diff"]][0]
            for c in range(2):
                if abs(p["F_unit"][c]) > 1.0:
                    worst = max(worst, abs(q["F_unit"][c] / p["F_unit"][c] - 1.0))
    res["scaling_check_worst_rel"] = worst
    say(f"  quadratic scaling check (unit-amplitude coefficient at 0.5 cm vs 1 cm, mixed pairs): worst relative "
        f"change {worst:.3f}")
    return res


def _c(z):
    return complex(z[0], z[1]) if isinstance(z, list) else complex(z)


def compare(strips, say):
    path = os.path.join(OUT, "v2_results_quick.json")
    if not os.path.exists(path):
        path = os.path.join(OUT, "v2_results.json")
    R = json.load(open(path, encoding="utf-8"))["res"]
    first = {r["omega"]: r for r in R["first"]}
    rows = []
    say("\n=== a_2 comparison: 3-D dynamic second-order response (quad - hydrostatic/FK part + second-order potential) vs the "
        "strip's dynamic second-order response, per unit amplitude product; the difference is the composite's omitted term ===")
    say("  heave force in N/m^2 (per a A), pitch moment in N m/m^2; ref = |T| a + |X| A (first-order force at the amplitudes)")
    for p3 in R["pairs"]:
        hf = p3.get("quad_hf")
        if hf is None:
            continue
        a2_3d = np.array([_c(p3["quad"]["total"][c]) - _c(hf["total"][c]) + _c(p3.get("pot", [[0, 0], [0, 0]])[c]) for c in range(2)])
        a2_3d_hf = np.array([_c(hf["total"][c]) for c in range(2)])
        row = dict(kind=p3["kind"], motion=p3["motion"], wi=p3["w_motion"], wj=p3["w_wave"], diff=p3["diff"], Omega=p3["Omega"],
                   a2_3d=[complex(x) for x in a2_3d], a2_3d_hf=[complex(x) for x in a2_3d_hf], strip={})
        fi, fj = first[p3["w_motion"]], first[p3["w_wave"]]
        u = p3["motion"]
        if p3["kind"] == "mixed":
            T3u = _c(fi["T"]["33"]) if u == "heave" else _c(fi["T"]["35"])
            T5u = _c(fi["T"]["53"]) if u == "heave" else _c(fi["T"]["55"])
            refs = {key: (abs(T3u) * amps[u] + abs(_c(fj["X"]["3"])) * amps["wave"],
                          abs(T5u) * amps[u] + abs(_c(fj["X"]["5"])) * amps["wave"], amps[u] * amps["wave"])
                    for key, amps in (("small", AMPS_SMALL), ("oper", AMPS_OPER))}
        else:
            if u == "wave":
                refs = {key: (abs(_c(fj["X"]["3"])) * amps["wave"], abs(_c(fj["X"]["5"])) * amps["wave"], amps["wave"] ** 2)
                        for key, amps in (("small", AMPS_SMALL), ("oper", AMPS_OPER))}
            else:
                T3u = _c(fi["T"]["33"]) if u == "heave" else _c(fi["T"]["35"])
                T5u = _c(fi["T"]["53"]) if u == "heave" else _c(fi["T"]["55"])
                refs = {key: (abs(T3u) * amps[u], abs(T5u) * amps[u], amps[u] ** 2)
                        for key, amps in (("small", AMPS_SMALL), ("oper", AMPS_OPER))}
        row["refs"] = refs
        for S in strips:
            ps = [q for q in S["pairs"] if q["kind"] == p3["kind"] and q["motion"] == p3["motion"] and q["w_motion"] == p3["w_motion"]
                  and q["w_wave"] == p3["w_wave"] and q["diff"] == p3["diff"]]
            small = [q for q in ps if q["amps"] == "small"][0]
            oper = [q for q in ps if q["amps"] == "oper"][0]
            a2_s = np.array([complex(x) for x in small["F_unit"]])
            dlt = a2_3d - a2_s
            ent = dict(a2_strip=[complex(x) for x in a2_s], delta=[complex(x) for x in dlt],
                       strip_oper_direct=[complex(x) for x in oper["F"]])
            for key, (r3, r5, prod) in refs.items():
                ent[key] = dict(delta_F3=abs(dlt[0]) * prod, delta_M5=abs(dlt[1]) * prod, ref_F3=r3, ref_M5=r5,
                                frac_F3=abs(dlt[0]) * prod / r3, frac_M5=abs(dlt[1]) * prod / r5,
                                a2_3d_F3=abs(a2_3d[0]) * prod, a2_strip_F3=abs(a2_s[0]) * prod,
                                a2_3d_M5=abs(a2_3d[1]) * prod, a2_strip_M5=abs(a2_s[1]) * prod)
            ent["oper"]["strip_direct_F3"] = abs(complex(oper["F"][0]))
            ent["oper"]["strip_direct_M5"] = abs(complex(oper["F"][1]))
            # operating amplitude: 3-D quadratic prediction minus the strip's full nonlinear residual (all orders)
            prod_o = refs["oper"][2]
            dd = a2_3d * prod_o - np.array([complex(x) for x in oper["F"]])
            ent["oper"]["delta_direct_F3"] = abs(dd[0])
            ent["oper"]["delta_direct_M5"] = abs(dd[1])
            ent["oper"]["frac_direct_F3"] = abs(dd[0]) / refs["oper"][0]
            ent["oper"]["frac_direct_M5"] = abs(dd[1]) / refs["oper"][1]
            row["strip"][S["label"]] = ent
        rows.append(row)
    for S in strips:
        L = S["label"]
        say(f"\n  --- strip {L} ---")
        say(f"  {'kind':>11} {'motion':>6} {'w_mot':>5} {'w_wav':>5} {'Omega':>6} | {'3D dyn':>8} {'3D hf':>8} {'strip':>8} {'diff':>8} | "
            f"{'frac1cm':>7} {'fracOp':>7} | {'M: 3D dyn':>9} {'strip':>8} {'diff':>8} {'frac1cm':>7} {'fracOp':>7} | "
            f"{'strip oper direct F3':>20} {'quad extrap':>11} {'3D-strip oper':>13} {'fracDir':>7} {'M fracDir':>9}")
        for r in rows:
            e = r["strip"][L]
            say(f"  {r['kind']:>11} {r['motion']:>6} {r['wi']:>5.1f} {r['wj']:>5.1f} {r['Omega']:>6.2f} | {abs(r['a2_3d'][0]):>8.0f} "
                f"{abs(r['a2_3d_hf'][0]):>8.0f} {abs(e['a2_strip'][0]):>8.0f} {abs(e['delta'][0]):>8.0f} | {e['small']['frac_F3']:>7.3f} "
                f"{e['oper']['frac_F3']:>7.3f} | {abs(r['a2_3d'][1]):>9.0f} {abs(e['a2_strip'][1]):>8.0f} {abs(e['delta'][1]):>8.0f} "
                f"{e['small']['frac_M5']:>7.3f} {e['oper']['frac_M5']:>7.3f} | {e['oper']['strip_direct_F3']:>20.0f} "
                f"{e['oper']['a2_strip_F3']:>11.0f} {e['oper']['delta_direct_F3']:>13.0f} {e['oper']['frac_direct_F3']:>7.3f} "
                f"{e['oper']['frac_direct_M5']:>9.3f}")
    return rows


def _json(o):
    if isinstance(o, dict):
        return {str(k): _json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json(v) for v in o]
    if isinstance(o, complex):
        return [o.real, o.imag]
    if isinstance(o, np.ndarray):
        return _json(o.tolist())
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.complexfloating):
        return [float(o.real), float(o.imag)]
    return o


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)
    t0 = time.time()
    say(f"V2 strip-side second-order response (corrected definition), Scarab 195 at rest ({time.strftime('%Y-%m-%d %H:%M')})")
    strips = [run_strip(False, say), run_strip(True, say)]
    rows = compare(strips, say)
    say(f"\ntotal wall time {time.time() - t0:.0f} s")
    json.dump(_json(dict(strips=strips, compare=rows)), open(os.path.join(OUT, "v2_strip.json"), "w", encoding="utf-8"), indent=1)
    open(os.path.join(OUT, "v2_a2_compare.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
