{
  # Dev shell for driving LOTUSim from Python with Gymnasium.
  #
  #   nix develop path:/mnt/c/Users/<you>/Documents/usv-rl-mpc/control_envs/lotusim_setup
  #
  # It puts on PYTHONPATH: rclpy (ROS 2 Jazzy), the LOTUSim message package
  # (lotusim_msgs), geographic/geometry/std msgs and numpy from the same
  # nixpkgs that LOTUSim itself is built against (via nix-ros-overlay), so
  # there is no ABI mismatch and nothing is compiled locally; gymnasium comes
  # from pip into ~/.venv-lotusim (see below).
  description = "LOTUSim + Gymnasium Python shell";

  nixConfig = {
    extra-substituters = [ "https://ros.cachix.org" "https://naval-group.cachix.org" ];
    extra-trusted-public-keys = [
      "ros.cachix.org-1:dSyZxI8geDCJrwgvCOHDoAfOm5sV1wCPjBkKL+38Rvo="
      "naval-group.cachix.org-1:ytTEzFEeuzQrC9IRYLzHGa5OnM65G95M6/sbPd0fy28="
    ];
  };

  inputs = {
    lotusim.url = "github:naval-group/LOTUSim";
    # Follow LOTUSim's own overlay and nixpkgs so the ROS packages are the same
    # store paths (cache hits, one copy of everything).
    nix-ros-overlay.follows = "lotusim/nix-ros-overlay";
    nixpkgs.follows = "lotusim/nixpkgs";
    flake-utils.follows = "lotusim/flake-utils";
  };

  outputs = { self, nixpkgs, nix-ros-overlay, flake-utils, lotusim }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ nix-ros-overlay.overlays.default ];
        };
        ros = pkgs.rosPackages.jazzy;
        # LOTUSim's own message package (lotusim_msgs, lotusim_sensor_msgs),
        # built by the LOTUSim flake exactly as the simulator uses it.
        messages = lotusim.packages.${system}.messages;
        # gymnasium is NOT taken from nixpkgs: its closure there pulls opencv,
        # which is not in the caches and fails to build (2026-10-21). It is
        # pure Python, so the shellHook installs it with pip into a small venv
        # layered on this interpreter (--system-site-packages keeps numpy
        # and rclpy visible).
        python = pkgs.python3.withPackages (ps: with ps; [ numpy ]);
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            python
            ros.rclpy
            ros.ros2cli
            ros.ros2node
            ros.ros2topic
            ros.ros2service
            ros.ros2action
            ros.std-msgs
            ros.geometry-msgs
            ros.geographic-msgs
            ros.action-msgs
            ros.rosidl-default-runtime
            messages
          ];
          # The ROS setup hooks assemble AMENT_PREFIX_PATH / PYTHONPATH from the
          # packages above; the explicit lines below make sure the LOTUSim
          # messages are visible even if a hook is skipped.
          shellHook = ''
            export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
            export ROS_DOMAIN_ID="''${ROS_DOMAIN_ID:-0}"
            export ROS_LOCALHOST_ONLY=1
            export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
            export AMENT_PREFIX_PATH="${messages}''${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}"
            for d in ${messages}/lib/python3*/site-packages; do
              export PYTHONPATH="$d''${PYTHONPATH:+:$PYTHONPATH}"
            done
            VENV="$HOME/.venv-lotusim"
            if [ ! -x "$VENV/bin/python" ]; then
              echo "creating $VENV and installing gymnasium (once)"
              python3 -m venv --system-site-packages "$VENV"                 && "$VENV/bin/pip" install --quiet --disable-pip-version-check gymnasium
            fi
            export PATH="$VENV/bin:$PATH"
            echo "LOTUSim Python shell: rclpy, lotusim_msgs, numpy (nix) + gymnasium ($VENV)"
          '';
        };
      });
}
