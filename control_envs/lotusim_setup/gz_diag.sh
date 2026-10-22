#!/usr/bin/env bash
# Gazebo-transport diagnostics against the running LOTUSim server: topics,
# the scene/world services, and the scene graph the GUI would receive.
set -uo pipefail
GZ="$(ls -d /nix/store/*gz-tools-vendor*/bin/gz 2>/dev/null | head -1)"
cfg=""
for d in /nix/store/*gz-sim-vendor*/share/gz /nix/store/*gz-transport-vendor*/share/gz /nix/store/*gz-msgs-vendor*/share/gz; do
  [ -d "$d" ] && cfg="$cfg:$d"
done
export GZ_CONFIG_PATH="${cfg#:}"
echo "gz=$GZ"
echo "GZ_CONFIG_PATH=$GZ_CONFIG_PATH"
echo "--- topics:"
timeout 20 "$GZ" topic -l 2>&1 | head -25
echo "--- services (scene/world/gui):"
timeout 20 "$GZ" service -l 2>&1 | grep -E 'scene|worlds|gui|state' | head -12
echo "--- world list:"
timeout 20 "$GZ" service -s /gazebo/worlds --reqtype gz.msgs.Empty --reptype gz.msgs.StringMsg_V --timeout 3000 --req '' 2>&1 | head -5
echo "--- scene info (first lines):"
timeout 20 "$GZ" service -s /world/lotusim/scene/info --reqtype gz.msgs.Empty --reptype gz.msgs.Scene --timeout 3000 --req '' 2>&1 | head -25
