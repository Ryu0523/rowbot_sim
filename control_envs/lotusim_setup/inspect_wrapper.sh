#!/usr/bin/env bash
# Show how the lotusim wrapper starts the server and the GUI, and where the
# gz binary is (to start the Gazebo GUI client separately from the server).
set -uo pipefail
W="$(command -v lotusim)"
echo "wrapper: $W"
echo "--- lines with gui / glwrap / gz / exec:"
grep -n -E 'gui|glwrap|gz |gz$|exec|GZ_|ROS_|FASTDDS' "$W" | head -40
ENV_BIN="$(grep -o '/nix/store/[a-z0-9]*-lotusim-env/bin/lotusim-env' "$W" | head -1)"
echo "--- lotusim-env: $ENV_BIN"
ls "$(dirname "$ENV_BIN")" | head -20
echo "--- lotusim-env head:"
head -c 1500 "$ENV_BIN" | strings | head -30
echo "--- gz binaries in the store:"
ls -d /nix/store/*gz-sim-vendor*/bin/gz /nix/store/*gz-tools*/bin/gz 2>/dev/null | head
find /nix/store -maxdepth 3 -type f -name gz -path '*bin*' 2>/dev/null | head -5
