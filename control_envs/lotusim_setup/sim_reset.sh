#!/usr/bin/env bash
# Stop every simulator-related process (gz servers and GUIs, xdyn, the demo
# scripts) and make sure the loopback multicast route exists.
#
# Why the route: gz-transport (the simulator's own pub/sub, used between the
# server and the Gazebo window) discovers peers over UDP multicast. Under
# WSL mirrored networking there is no multicast route, the server logs
# "Exception sending a multicast message: Network is unreachable", the
# window never receives the scene (grey viewport, empty entity tree).
# ROS 2 was made independent of multicast with ROS_LOCALHOST_ONLY=1.
#
# Why kill by PID: on this NixOS the gz processes are named .gz-wrapped and
# re-title their command line; pkill -f patterns proved unreliable, which
# left several servers running at once (commands then reach the wrong
# server: "No known entity", "more than one action server").
set -uo pipefail
pids="$(ps -eo pid,args --no-headers | grep -E 'gz sim|gz-wrapped|xdyn-for-cs|lotusim_waypoint_demo|lotusim_gym_demo|lotusim_rudder_test|ros_probe' | grep -v grep | awk '{print $1}')"
if [ -n "$pids" ]; then
  echo "killing: $pids"
  kill -9 $pids 2>/dev/null || true
  sleep 2
fi
left="$(ps -eo pid,args --no-headers | grep -E 'gz sim|gz-wrapped|xdyn-for-cs' | grep -v grep | cut -c1-90)"
echo "remaining: ${left:-none}"
if ip route show 224.0.0.0/4 2>/dev/null | grep -q 224; then
  echo "multicast route present: $(ip route show 224.0.0.0/4)"
else
  sudo ip route add 224.0.0.0/4 dev lo && echo "multicast route added on lo"
fi
sudo ip link set lo multicast on 2>/dev/null || true
echo "lo flags: $(ip -o link show lo | cut -d' ' -f3)"
