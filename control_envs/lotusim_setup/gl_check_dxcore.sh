#!/usr/bin/env bash
# Can the WSL GPU bridge library (libdxcore.so, a Windows-built Linux .so)
# be loaded by a nix-built program at all? Mesa's d3d12 driver dlopens it;
# if this fails, no nix Mesa can reach the GPU.
set -uo pipefail
LIB=/run/opengl-driver/lib/libdxcore.so
echo "--- ldd (nix glibc):"
ldd "$LIB" 2>&1 | head -12
echo "--- dlopen from nix python:"
nix shell nixpkgs#python3 --command python3 - <<'PY' 2>&1 | tail -3
import ctypes
for p in ("/run/opengl-driver/lib/libdxcore.so", "/usr/lib/wsl/lib/libdxcore.so", "/run/opengl-driver/lib/libd3d12.so"):
    try:
        ctypes.CDLL(p)
        print("loaded", p)
    except OSError as e:
        print("FAILED", p, "->", e)
PY
echo "--- NixOS-WSL version / module options:"
grep -E "useWindowsDriver|wsl.version|nixos-wsl" /etc/nixos/configuration.nix /etc/nixos/flake.nix 2>/dev/null | head
ls /nix/store | grep -i "nixos-wsl" | head -3
