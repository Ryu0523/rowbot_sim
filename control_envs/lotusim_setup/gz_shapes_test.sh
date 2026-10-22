#!/usr/bin/env bash
# Rendering test independent of LOTUSim's world: open Gazebo's own
# shapes.sdf (server + window in one process) with software OpenGL.
# If this window shows the shapes, rendering under WSLg works and a grey
# LOTUSim viewport is a scene/transport problem; if it is grey too, it is
# the GL path.
set -uo pipefail
W="$(ls /nix/store/*gz-sim-vendor*/share/gz/gz-sim8/worlds/shapes.sdf 2>/dev/null | head -1)"
echo "world: $W"
export LIBGL_ALWAYS_SOFTWARE=1
exec lotusim -v2 -r "$W"
