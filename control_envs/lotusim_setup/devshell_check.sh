#!/usr/bin/env bash
# Build the Python dev shell (flake.nix) and check that rclpy, the LOTUSim
# messages, numpy and gymnasium import. First run downloads the ROS 2 Python
# stack from the caches (a few minutes).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
nix develop "path:$HERE" --command python3 - <<'PY'
import rclpy, numpy, gymnasium
import lotusim_msgs.msg as m
from lotusim_msgs.action import MASCmd
print("devshell ok: gymnasium", gymnasium.__version__, "numpy", numpy.__version__)
print("messages:", [x for x in dir(m) if not x.startswith("_")][:8])
PY
