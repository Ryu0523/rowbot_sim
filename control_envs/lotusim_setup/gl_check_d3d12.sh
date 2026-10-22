#!/usr/bin/env bash
# Why does Mesa's d3d12 driver (the WSL GPU bridge) not come up? Force it on
# each EGL platform with debug output.
set -uo pipefail
BIN="$(nix build --no-link --print-out-paths nixpkgs#mesa-demos 2>/dev/null | head -1)/bin"
echo "--- bridge libraries visible to Mesa:"
ls -la /run/opengl-driver/lib/libdxcore.so /run/opengl-driver/lib/libd3d12.so /usr/lib/wsl/lib/libdxcore.so 2>&1
echo "--- /dev/dri device info (drm):"
ls -la /dev/dri/
cat /sys/class/drm/card0/device/uevent 2>&1 | head -5
echo "--- eglinfo, surfaceless platform, d3d12 forced, debug:"
MESA_LOADER_DRIVER_OVERRIDE=d3d12 GALLIUM_DRIVER=d3d12 MESA_DEBUG=1 EGL_LOG_LEVEL=debug \
  timeout 60 "$BIN/eglinfo" -B -p surfaceless 2>&1 | grep -i -E 'd3d12|dxcore|adapter|renderer|error|fail|warn|libEGL' | head -20
echo "--- eglinfo, wayland platform, d3d12 forced:"
MESA_LOADER_DRIVER_OVERRIDE=d3d12 GALLIUM_DRIVER=d3d12 MESA_DEBUG=1 EGL_LOG_LEVEL=debug \
  timeout 60 "$BIN/eglinfo" -B -p wayland 2>&1 | grep -i -E 'd3d12|dxcore|adapter|renderer|error|fail|warn|libEGL' | head -20
echo "--- eglinfo, device platform:"
timeout 60 "$BIN/eglinfo" -B -p device 2>&1 | grep -i -E 'device|renderer|error|fail' | head -12
