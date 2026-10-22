#!/usr/bin/env bash
# Start the physics server (xdyn, surface vessels on port 12345) and LOTUSim
# in one go; Ctrl+C stops both.   Options:  --gui   open the Gazebo window
#
#   bash run_sim.sh            # headless server
#   bash run_sim.sh --gui      # with the Gazebo window (needs the GPU bridge)
#
# Requires the `lotusim` and `xdyn-for-cs` commands (nixos_setup.sh installs
# them). The DTMB hull YAML ships with LOTUSim's assets; LOTUSIM_MODELS_PATH is
# exported by the lotusim wrapper, so resolve it the same way here.
set -euo pipefail

MODELS="${LOTUSIM_MODELS_PATH:-${XDG_DATA_HOME:-$HOME/.local/share}/lotusim/models/}"
YAML="${XDYN_YAML:-$MODELS/dtmb_hull/dtmb-xdyn-faster.yml}"
PORT="${XDYN_PORT:-12345}"
DT="${XDYN_DT:-0.1}"

if [ ! -f "$YAML" ]; then
  # first run: the lotusim wrapper seeds the state dir with the assets
  lotusim --help >/dev/null 2>&1 || true
fi
if [ ! -f "$YAML" ]; then
  echo "xdyn YAML not found: $YAML" >&2
  echo "run 'lotusim --help' once (it seeds ~/.local/share/lotusim/models) or set XDYN_YAML" >&2
  exit 1
fi

export XDYN_ASSETS_PATH="${XDYN_ASSETS_PATH:-$MODELS}"
# ROS 2 discovery on this one machine: the simulator and the Python client
# must use the same settings (the dev-shell flake exports the same three).
# Localhost-only discovery does not depend on multicast, which WSL's mirrored
# networking does not always pass.
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY=1
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
# gz-transport (server <-> Gazebo window) discovers over UDP multicast,
# which WSL does not route; keep it on the loopback and add a unicast relay
# (GZ_RELAY) so discovery works without multicast. sim_reset.sh adds the
# 224.0.0.0/4 route on lo and switches lo's multicast flag on as well.
export GZ_IP=127.0.0.1
export GZ_RELAY=127.0.0.1
echo "xdyn: $YAML  ws://127.0.0.1:$PORT  dt $DT"
xdyn-for-cs "$YAML" --address 127.0.0.1 --port "$PORT" --dt "$DT" &
XDYN_PID=$!
trap 'kill $XDYN_PID 2>/dev/null || true' EXIT INT TERM
sleep 1

echo "lotusim run $*"
lotusim run "$@"
