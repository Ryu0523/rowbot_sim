#!/usr/bin/env python3
"""
Run independent simulation jobs in parallel processes, as many as free RAM
allows (learn/meta/ACCEPTANCE_2026-10-06.md T2).

Why a pool sized by memory and not by cores: the machine has 20 logical
cores but often only 3-4 GB of free RAM, and a learned-model job can take
over a gigabyte while a plant-only job takes about 0.15 GB (measured
2026-10-06: 136 MB resident after building a mission). Sizing by the job's
own footprint lets plant sweeps use most cores and keeps heavy jobs to one
or two.

    from studies.parallel_runner import run_parallel
    results = run_parallel(job, [args1, args2, ...], gb_per_job=0.15)

`job` must be a module-level function (Windows spawns workers, so it is
pickled by name) taking one argument and returning something picklable.
Results come back in the order of the arguments. A job that raises returns
the exception instance instead of a result, so one bad case does not lose
the others.
"""
import multiprocessing as mp
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


def free_gb():
    """Free physical memory in GB (psutil when present, else Windows CIM)."""
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except ImportError:
        import subprocess
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
            capture_output=True, text=True).stdout.strip()
        return float(out) * 1024 / 1e9


def n_workers(gb_per_job, reserve_gb=1.5, max_procs=None):
    """min(cores - 2, floor((free - reserve) / per-job)), at least 1."""
    cores = max(1, (os.cpu_count() or 2) - 2)
    by_ram = int((free_gb() - reserve_gb) // max(gb_per_job, 1e-3))
    n = max(1, min(cores, by_ram))
    return min(n, max_procs) if max_procs else n


def _call(packed):
    fn, arg = packed
    try:
        return fn(arg)
    except Exception as e:      # noqa: BLE001 -- returned, not swallowed
        return e


def run_parallel(fn, args, gb_per_job=0.15, reserve_gb=1.5, max_procs=None):
    """Map fn over args in a spawn pool sized by free RAM; ordered results."""
    args = list(args)
    if not args:
        return []
    n = min(len(args), n_workers(gb_per_job, reserve_gb, max_procs))
    if n == 1:
        return [_call((fn, a)) for a in args]
    ctx = mp.get_context("spawn")
    with ctx.Pool(n) as pool:
        return pool.map(_call, [(fn, a) for a in args], chunksize=1)


if __name__ == "__main__":
    print(f"free {free_gb():.1f} GB; plant-only workers "
          f"{n_workers(0.15)}, learned-model workers {n_workers(1.2)}")
