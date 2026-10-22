#!/usr/bin/env bash
# ROS 2 discovery diagnostics: what the LOTUSim wrapper exports, what the dev
# shell exports, and which nodes/topics/actions the dev shell can see.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "--- lotusim wrapper (env lines):"
grep -E 'export|RMW|ROS_|FASTDDS|CYCLONE|exec' "$(which lotusim)" | grep -v '^#' | head -20
echo "--- dev shell:"
cd "$HERE"
nix develop "path:$HERE" --command bash -c '
  env | grep -E "RMW|ROS_|FASTDDS|CYCLONE|AMENT_PREFIX" | cut -c1-160
  echo "nodes:";   timeout 20 ros2 node list 2>&1 | head
  echo "topics:";  timeout 20 ros2 topic list 2>&1 | head -20
  echo "actions:"; timeout 20 ros2 action list 2>&1 | head
' 2>&1 | grep -v -E 'warning: ignoring|accept-flake-config'
