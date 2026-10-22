"""Assemble the demo video: title cards (matplotlib, Chinese font) + the
Blender clips, concatenated with the bundled ffmpeg of imageio-ffmpeg.

    python control_envs/demo_video_assemble.py --out control_envs/_out/demo/demo_2026-10-21.mp4 \
        "标题一" clip1.mp4 "标题二" clip2.mp4 ...
Each title is shown for --title-s seconds before its clip. A title may hold
two lines separated by '|' (second line smaller).
"""
import argparse
import os
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib import font_manager                   # noqa: E402

import imageio_ffmpeg                                 # noqa: E402

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def cjk_font():
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "PingFang SC"):
        try:
            return font_manager.FontProperties(family=name)
        except Exception:            # noqa: BLE001
            continue
    return font_manager.FontProperties()


def title_card(text, path, res=(960, 540)):
    fp = cjk_font()
    fig = plt.figure(figsize=(res[0] / 100, res[1] / 100), dpi=100, facecolor="#0b1b2b")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_facecolor("#0b1b2b")
    lines = text.split("|")
    ax.text(0.5, 0.58 if len(lines) > 1 else 0.5, lines[0], ha="center", va="center",
            color="white", fontsize=23, fontproperties=fp, transform=ax.transAxes)
    if len(lines) > 1:
        ax.text(0.5, 0.40, lines[1], ha="center", va="center", color="#cfd8e3",
                fontsize=15, fontproperties=fp, transform=ax.transAxes)
    fig.savefig(path, dpi=100, facecolor="#0b1b2b")
    plt.close(fig)


def still_clip(png, seconds, out, fps=24):
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-loop", "1", "-i", png,
                    "-t", str(seconds), "-r", str(fps), "-pix_fmt", "yuv420p",
                    "-vf", "scale=960:540", "-c:v", "libx264", "-preset", "veryfast", out],
                   check=True)


def normalise(src, out, fps=24):
    """Same codec, size and frame rate for every clip (concat needs it)."""
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", src, "-r", str(fps),
                    "-vf", "scale=960:540", "-pix_fmt", "yuv420p", "-c:v", "libx264",
                    "-preset", "veryfast", "-an", out], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--title-s", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("items", nargs="+", help="title clip title clip ...")
    args = ap.parse_args()
    if len(args.items) % 2:
        raise SystemExit("items: pairs of title and clip")
    tmp = tempfile.mkdtemp(prefix="demo_")
    parts = []
    for i in range(0, len(args.items), 2):
        title, clip = args.items[i], args.items[i + 1]
        png = os.path.join(tmp, f"t{i}.png")
        title_card(title, png)
        tc = os.path.join(tmp, f"t{i}.mp4")
        still_clip(png, args.title_s, tc, args.fps)
        cc = os.path.join(tmp, f"c{i}.mp4")
        normalise(clip, cc, args.fps)
        parts += [tc, cc]
    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w", encoding="utf-8") as f:
        for p in parts:
            f.write(f"file '{p.replace(os.sep, '/')}'\n")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", lst, "-c", "copy", args.out], check=True)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
