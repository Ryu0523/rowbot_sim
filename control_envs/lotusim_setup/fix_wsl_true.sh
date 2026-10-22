#!/usr/bin/env bash
# Patch an /etc/nixos/configuration.nix that nixos_setup.sh edited before
# these two lines were in configuration.nix.snippet (2026-10-21):
#   1. a tmpfiles rule creating /bin/true: WSL 3.x generates
#      wsl-mnt-guard.service with ExecStart=/bin/true, which NixOS lacks, so
#      nixos-rebuild switch reported a failed unit (exit 4);
#   2. the nixos user in the video and render groups, so the Gazebo window
#      can open /dev/dri/card0 (otherwise software rendering).
# Idempotent. Run:  bash /mnt/c/.../fix_wsl_true.sh && sudo nixos-rebuild switch
set -euo pipefail
CONF=/etc/nixos/configuration.nix
if sudo grep -q 'L+ /bin/true' "$CONF"; then
  echo "tmpfiles rule already present"
else
  sudo sed -i 's|^  networking.firewall.enable = false;$|  networking.firewall.enable = false;\n  systemd.tmpfiles.rules = [ "L+ /bin/true - - - - ${pkgs.coreutils}/bin/true" ];|' "$CONF"
  echo "tmpfiles rule inserted"
fi
if sudo grep -q 'extraGroups' "$CONF"; then
  echo "groups already present"
else
  sudo sed -i 's|^  wsl.useWindowsDriver = true;$|  wsl.useWindowsDriver = true;\n  users.users.nixos.extraGroups = [ "video" "render" ];|' "$CONF"
  echo "video/render groups inserted"
fi
grep -n -A1 -E 'firewall.enable|useWindowsDriver' "$CONF"
