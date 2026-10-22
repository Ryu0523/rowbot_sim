"""Algebraic checks for the independent hull-force derivation.

These are NOT a hull solver or hydrodynamic validation. They check (1) an
endpoint counterexample, (2) exact mixed sensitivities of a coupled operator,
and (3) single-thread cost of a representative *synthetic* reduced core.
Run: .venv/Scripts/python.exe studies/derive_coupled_rom_checks.py
"""

import os

for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"

import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np


def mixed_sensitivity_check():
    rng = np.random.default_rng(20261009)
    n = 12
    a0 = np.diag(np.linspace(1.0, 2.0, n))
    mats = []
    for _ in range(3):
        raw = rng.normal(size=(n, n))
        raw = (raw + raw.T) / 2
        mats.append(raw * 0.15 / np.linalg.norm(raw, 2))
    a1, a2, a12 = mats
    b0, b1, b2, b12 = rng.normal(size=(4, n))

    def solution(x, y):
        return np.linalg.solve(a0 + x*a1 + y*a2 + x*y*a12,
                               b0 + x*b1 + y*b2 + x*y*b12)

    u0 = solution(0, 0)
    u1 = np.linalg.solve(a0, b1 - a1 @ u0)
    u2 = np.linalg.solve(a0, b2 - a2 @ u0)
    u11 = np.linalg.solve(a0, -2*a1 @ u1)
    u22 = np.linalg.solve(a0, -2*a2 @ u2)
    u12 = np.linalg.solve(a0, b12-a12 @ u0-a1 @ u2-a2 @ u1)
    h = 1e-4
    fd12 = (solution(h, h)-solution(h, -h)-solution(-h, h)
            +solution(-h, -h))/(4*h*h)
    fd_error = np.linalg.norm(fd12-u12)/np.linalg.norm(u12)
    rows = []
    for step in (0.2, 0.1, 0.05, 0.025):
        x, y = step, 0.7*step
        truth = solution(x, y)
        separated = solution(x, 0)+solution(0, y)-u0
        taylor2 = (u0+x*u1+y*u2+0.5*x*x*u11+x*y*u12+0.5*y*y*u22)
        rows.append({"step": step,
                     "separated_error": float(np.linalg.norm(truth-separated)),
                     "coupled_second_order_error": float(np.linalg.norm(truth-taylor2))})
    sep_order = np.log2(rows[-2]["separated_error"]/rows[-1]["separated_error"])
    coupled_order = np.log2(rows[-2]["coupled_second_order_error"]/
                            rows[-1]["coupled_second_order_error"])
    assert fd_error < 1e-6
    assert 1.9 < sep_order < 2.1
    assert 2.8 < coupled_order < 3.2
    return {"mixed_derivative_relative_error": float(fd_error),
            "separated_observed_order": float(sep_order),
            "coupled_observed_order": float(coupled_order), "rows": rows}


def synthetic_core_benchmark(r=32, m=128, q_terms=16, waves=100, repeats=1500):
    """Includes reduced matrix assembly/solve, sampled gradients, waves, pressure.

    Does NOT include actual BEM geometry updates, force-consistent potential
    time derivatives, contact iteration, far-field evolution or body stepping.
    Near-identity matrices are chosen to avoid ill-conditioned random examples.
    """
    rng = np.random.default_rng(101+r)
    operators = rng.normal(size=(q_terms, r, r)) * (0.005/r)
    gradients = rng.normal(size=(3*m, r)) / np.sqrt(r)
    wave_map = rng.normal(size=(m, waves)) / np.sqrt(waves)
    force_map = rng.normal(size=(6, m)) / m
    body_inputs = rng.normal(size=(r, 7))
    identity = np.eye(r)
    coefficients = rng.uniform(-0.5, 0.5, q_terms)
    phase = rng.uniform(-np.pi, np.pi, waves)
    omega = rng.uniform(0.1, 11.0, waves)
    weights = rng.uniform(0.1, 1, waves)
    state = rng.normal(size=r)

    def core(t):
        ar = identity + (coefficients @ operators.reshape(q_terms, r*r)).reshape(r, r)
        sol = np.linalg.solve(ar, body_inputs)
        vel = (gradients @ (sol[:, 0] + state)).reshape(3, m)
        incident = wave_map @ (weights*np.cos(phase+omega*t))
        pressure = -incident-0.5*np.sum(vel*vel, axis=0)
        return force_map @ pressure

    for i in range(100):
        core(i*0.02)
    durations = []
    checksum = 0.0
    for i in range(repeats):
        t0 = perf_counter_ns()
        value = core(i*0.02)
        durations.append((perf_counter_ns()-t0)/1e6)
        checksum += float(value[0])
    return {"r": r, "sample_points": m, "operator_terms": q_terms,
            "wave_modes": waves, "median_ms": float(np.median(durations)),
            "p95_ms": float(np.percentile(durations, 95)),
            "checksum": checksum, "scope": "synthetic arithmetic only; not full simulator"}


def main():
    gravity = 9.81
    result = {
        "purpose": "Mathematical counterexamples and synthetic cost; NOT hull accuracy validation",
        "endpoint_counterexample": {
            "function": "F(s)=1+K*s^2/(1+s^2)^2, s>=0",
            "F_at_zero": 1.0, "F_at_infinity": 1.0,
            "F_at_one_for_K_100": 26.0,
            "meaning": "Identical endpoint values do not bound the interior error"},
        "following_sea_zero_encounter_kn": {
            str(period): gravity*period/(2*np.pi)/0.514444 for period in (3, 5, 8, 10)},
        "coupled_operator_check": mixed_sensitivity_check(),
    }
    result["thread_control_environment"] = {
        key: os.environ[key] for key in
        ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")}
    result["synthetic_cost"] = [synthetic_core_benchmark(r=r) for r in (32, 48)]
    out = Path(__file__).resolve().parent / "_cache" / "derive_coupled"
    out.mkdir(parents=True, exist_ok=True)
    (out / "checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
