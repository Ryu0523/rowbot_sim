"""Locate the read-only toy environment and import it.

The teammate's Gymnasium environment is not an installed package; we add its
folder to ``sys.path`` (read-only, nothing is written there) and import it.
The folder can be overridden with the environment variable ``SKATE_WAVES_DIR``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_DIR = Path(r"C:\Users\Administrator\Documents\ext_sims\RowBot\2_RL_Tutorials")
OUT_DIR = Path(__file__).resolve().parent / "_out"


def add_env_path() -> Path:
    d = Path(os.environ.get("SKATE_WAVES_DIR", DEFAULT_DIR))
    if not (d / "skate_waves.py").exists():
        raise FileNotFoundError(f"skate_waves.py not found in {d}; set SKATE_WAVES_DIR")
    s = str(d)
    if s not in sys.path:
        sys.path.insert(0, s)
    return d


add_env_path()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Import once here so every module gets the same class objects.
from skate_waves import SkateConfig, SkateWavesEnv  # noqa: E402,F401
