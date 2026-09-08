#!/usr/bin/env python3
"""
Inline viewer_data.json into seaway.html to make a single self-contained page.

The viewer is deliberately ONE file with no fetches. Two reasons: browsers block
`fetch` from `file://` URLs, so a split page cannot simply be double-clicked;
and the page is meant to be sent to someone as a single attachment.

    python -m viewer.build [--data viewer_data.json] [--out viewer/seaway_full.html]

Run `python -m studies.export_viewer` first to produce viewer_data.json.
"""
import argparse
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MARKER = "const DATA = /*__DATA__*/null;"


def build(data_path, out_path, src_path=None):
    src_path = src_path or os.path.join(HERE, "seaway.html")
    if not os.path.exists(data_path):
        sys.exit(f"missing {data_path} -- run: python -m studies.export_viewer")
    html = io.open(src_path, encoding="utf-8").read()
    if MARKER not in html:
        sys.exit(f"{src_path} has no data marker; expected: {MARKER}")
    data = io.open(data_path, encoding="utf-8").read()
    io.open(out_path, "w", encoding="utf-8").write(
        html.replace(MARKER, "const DATA = " + data + ";", 1))
    kb = os.path.getsize(out_path) / 1024
    print(f"  {os.path.basename(src_path)} + {os.path.basename(data_path)}"
          f"  ->  {out_path}  ({kb:.0f} KB)")
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=os.path.join(ROOT, "viewer_data.json"))
    p.add_argument("--out", default=os.path.join(HERE, "seaway_full.html"))
    p.add_argument("--src", default=None)
    a = p.parse_args()
    build(a.data, a.out, a.src)


if __name__ == "__main__":
    main()
