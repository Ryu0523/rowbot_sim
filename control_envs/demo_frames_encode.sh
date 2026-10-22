#!/usr/bin/env bash
# Encode viewer snapshots viewer/demo_shots/<prefix>_%04d.png into an MP4.
#   bash control_envs/demo_frames_encode.sh fly control_envs/_out/demo/fly.mp4 [fps]
set -euo pipefail
prefix="$1"; out="$2"; fps="${3:-24}"
FF="C:/Users/Administrator/Documents/usv-rl-mpc/.venv/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe"
"$FF" -y -loglevel error -framerate "$fps" -i "viewer/demo_shots/${prefix}_%04d.png" \
  -vf "scale=960:540" -pix_fmt yuv420p -c:v libx264 -preset medium -crf 20 "$out"
ls -la "$out"
