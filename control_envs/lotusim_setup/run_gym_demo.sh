#!/usr/bin/env bash
# Run the Gymnasium demo inside the Python dev shell (rclpy + lotusim_msgs +
# gymnasium). The simulation must already be running (run_sim.sh).
#
#   bash run_gym_demo.sh [demo args...]      e.g.  --seconds 120 --target-speed 4
#   bash run_gym_demo.sh --script waypoint --course square --side 300   (waypoint demo)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
SCRIPT="lotusim_gym_demo.py"
if [ "${1:-}" = "--script" ]; then
  case "$2" in
    waypoint) SCRIPT="lotusim_waypoint_demo.py" ;;
    gym)      SCRIPT="lotusim_gym_demo.py" ;;
    *) echo "unknown --script $2 (gym | waypoint)" >&2; exit 1 ;;
  esac
  shift 2
fi
exec nix develop "path:$HERE" --command python3 -u "$REPO/control_envs/$SCRIPT" "$@"
