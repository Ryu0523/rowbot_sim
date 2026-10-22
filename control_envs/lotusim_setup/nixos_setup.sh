#!/usr/bin/env bash
# One-time setup inside the NixOS-WSL distribution (run as the default user).
#
#   wsl -d NixOS
#   bash /mnt/c/Users/<you>/Documents/usv-rl-mpc/control_envs/lotusim_setup/nixos_setup.sh
#
# Steps (each is idempotent; re-running is safe):
#   1. append the LOTUSim block to /etc/nixos/configuration.nix (once)
#   2. nixos-rebuild switch   (flakes, binary caches, GPU bridge, git, firewall off)
#   3. install the lotusim and lotusim-ui commands into the user profile
#   4. install the xdyn physics server (vessels with dynamics need it)
#   5. seed LOTUSim's state directory and print what to do next
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF=/etc/nixos/configuration.nix
SNIPPET="$HERE/configuration.nix.snippet"
MARK="LOTUSim additions"

echo "== 1. $CONF"
if sudo grep -q "$MARK" "$CONF"; then
  echo "   already contains the LOTUSim block"
else
  # insert the snippet before the LAST line that starts with '}' (the end of the
  # main attribute set), keeping everything else untouched
  TMP="$(mktemp)"
  sudo awk -v snip="$SNIPPET" '
    { lines[n++] = $0 }
    END {
      last = -1
      for (i = n - 1; i >= 0; i--) if (lines[i] ~ /^}/) { last = i; break }
      for (i = 0; i < n; i++) {
        if (i == last) { while ((getline l < snip) > 0) print l; close(snip) }
        print lines[i]
      }
    }' "$CONF" > "$TMP"
  sudo cp "$CONF" "$CONF.bak-lotusim"
  sudo cp "$TMP" "$CONF"
  rm -f "$TMP"
  echo "   block inserted (backup at $CONF.bak-lotusim)"
fi

echo "== 2. nixos-rebuild switch (first time: a few minutes)"
sudo nixos-rebuild switch

echo "== 3. lotusim + lotusim-ui into the user profile"
nix profile add github:naval-group/LOTUSim github:naval-group/LOTUSim#ui 2>/dev/null \
  || nix profile install github:naval-group/LOTUSim github:naval-group/LOTUSim#ui

echo "== 4. xdyn physics server"
nix profile add github:naval-group/lxdyn#xdyn 2>/dev/null \
  || nix profile install github:naval-group/lxdyn#xdyn   # the flake has no default package

echo "== 5. seeding LOTUSim's state directory (models, scenarios, worlds)"
lotusim --help >/dev/null 2>&1 || true
ls "${XDG_DATA_HOME:-$HOME/.local/share}/lotusim/models" 2>/dev/null | head -20 || true

cat <<'EOF'

Done. Next:
  exit                       # leave NixOS
  wsl --shutdown             # in PowerShell, so the new config fully applies
  wsl -d NixOS
  lotusim --help             # lists the worlds
  lotusim run                # headless server;  lotusim run --gui  opens Gazebo
  lotusim-ui                 # browser UI at http://localhost:8080
Gymnasium: see SETUP.txt section 5 (run_sim.sh, then run_gym_demo.sh).
EOF
