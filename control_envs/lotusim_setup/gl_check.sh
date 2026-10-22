#!/usr/bin/env bash
# Which OpenGL can a nix-built program get inside this WSL: EGL devices /
# platforms (Wayland, X11, surfaceless) and the GLX renderer under Xwayland.
# Uses mesa-demos from nixpkgs (downloaded on first run).
set -uo pipefail
BIN="$(nix build --no-link --print-out-paths nixpkgs#mesa-demos 2>/dev/null | head -1)/bin"
echo "mesa-demos: $BIN"
echo "DISPLAY=${DISPLAY:-} WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-}"
echo "--- eglinfo -B:"
timeout 120 "$BIN/eglinfo" -B 2>&1 | grep -i -E 'platform|device|renderer|vendor|version|error|opengl|egl_' | head -40
echo "--- glxinfo -B:"
timeout 60 "$BIN/glxinfo" -B 2>&1 | grep -i -E 'renderer|vendor|version|error|device|direct' | head -12
echo "--- with MESA_LOADER_DRIVER_OVERRIDE=d3d12:"
MESA_LOADER_DRIVER_OVERRIDE=d3d12 timeout 60 "$BIN/glxinfo" -B 2>&1 | grep -i -E 'renderer|vendor|error' | head -6
echo "--- with LIBGL_ALWAYS_SOFTWARE=1:"
LIBGL_ALWAYS_SOFTWARE=1 timeout 60 "$BIN/glxinfo" -B 2>&1 | grep -i -E 'renderer|version string' | head -6
