#!/usr/bin/env python3
"""Markdown tables (Chinese labels) for s1b_progress_zh.md from studies/_cache/derive/s1b_tables.json (derive_s1b_tables.py).
Prints to stdout: (1) the main convergence table stage-1 flux vs stage 2 with verdicts; (2) the base-grid option sensitivity."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_cache", "derive")
R = json.load(open(os.path.join(OUT, "s1b_tables.json"), encoding="utf-8"))

ROWS = [
    ("C0 + C_s heave", "静态水动力刚度之和，升沉（牛/米）"),
    ("C0 + C_s pitch", "静态水动力刚度之和，纵摇（牛米/弧度）"),
    ("C0 heave (N/m)", "　其中 C0 升沉"),
    ("C0 pitch (N m/rad)", "　其中 C0 纵摇"),
    ("A_inf heave (kg)", "无穷频附加质量，升沉（公斤）"),
    ("A_inf pitch (kg m2)", "无穷频附加质量，纵摇（公斤·米²）"),
    ("A33 head_Tp3 (we 3.24)", "附加质量 A33，迎浪 Tp 3（遭遇频率 3.24）"),
    ("B33 head_Tp3 (we 3.24)", "阻尼 B33，迎浪 Tp 3"),
    ("A55 head_Tp3 (we 3.24)", "附加质量 A55，迎浪 Tp 3"),
    ("B55 head_Tp3 (we 3.24)", "阻尼 B55，迎浪 Tp 3"),
    ("A33 head_Tp5 (we 1.67)", "A33，迎浪 Tp 5（1.67）"),
    ("B33 head_Tp5 (we 1.67)", "B33，迎浪 Tp 5"),
    ("A55 head_Tp5 (we 1.67)", "A55，迎浪 Tp 5"),
    ("B55 head_Tp5 (we 1.67)", "B55，迎浪 Tp 5"),
    ("A33 follow_Tp3 (we 0.94)", "A33，顺浪 Tp 3（0.94，含 −C0/ω²）"),
    ("B33 follow_Tp3 (we 0.94)", "B33，顺浪 Tp 3"),
    ("B55 follow_Tp3 (we 0.94)", "B55，顺浪 Tp 3"),
    ("B33 follow_Tp5 (we 0.84)", "B33，顺浪 Tp 5（0.84）"),
    ("B55 follow_Tp5 (we 0.84)", "B55，顺浪 Tp 5"),
    ("|X3| total head_Tp3 (we 3.24)", "激励 升沉 总，迎浪 Tp 3（牛/米波幅）"),
    ("|X3| diffraction head_Tp3 (we 3.24)", "　其中绕射"),
    ("|X5|/k total head_Tp3 (we 3.24)", "激励 纵摇 总/波数，迎浪 Tp 3（牛米/米）"),
    ("|X5|/k diffraction head_Tp3 (we 3.24)", "　其中绕射"),
    ("|X3| total head_Tp5 (we 1.67)", "激励 升沉 总，迎浪 Tp 5"),
    ("|X3| diffraction head_Tp5 (we 1.67)", "　其中绕射"),
    ("|X5|/k total head_Tp5 (we 1.67)", "激励 纵摇 总，迎浪 Tp 5"),
    ("|X5|/k diffraction head_Tp5 (we 1.67)", "　其中绕射"),
    ("|X3| total follow_Tp3 (we 0.94)", "激励 升沉 总，顺浪 Tp 3"),
    ("|X5|/k total follow_Tp3 (we 0.94)", "激励 纵摇 总，顺浪 Tp 3"),
    ("|X5|/k diffraction follow_Tp3 (we 0.94)", "　其中绕射"),
    ("|X3| total follow_Tp5 (we 0.84)", "激励 升沉 总，顺浪 Tp 5"),
    ("|X5|/k total follow_Tp5 (we 0.84)", "激励 纵摇 总，顺浪 Tp 5（FK 与绕射几乎相消）"),
    ("|X5|/k diffraction follow_Tp5 (we 0.84)", "　其中绕射"),
    ("|W|/U max free surface (centroids)", "水面格子中心基础流最大 / 航速"),
    ("B_sym worst (all frequencies)", "被动性最差（十个频率）"),
]
VERD = {"converged": "收敛", "improved": "改善", "unchanged": "不变", "worse": "变差", "not conv.": "未收敛", "---": "—"}


def fmt(v, key):
    if v is None:
        return "—"
    if "B_sym" in key or "|W|/U" in key:
        return f"{v:+.4f}" if "B_sym" in key else f"{v:.2f}"
    return f"{v:,.0f}".replace(",", " ")


def seq(name, title):
    M = R.get(name)
    if not M:
        print(f"({name} missing)")
        return
    print(f"\n{title}\n")
    print("| 量 | 第一阶段（旧网格）粗 / 基准 / 细 | 基准→细变化 | 本阶段 粗 / 基准 / 细 | 基准→细变化 | 收敛比 r | 判定 |")
    print("|---|---|---|---|---|---|---|")
    for key, lab in ROWS:
        if key not in M:
            continue
        m = M[key]
        s1 = " / ".join(fmt(v, key) for v in m["stage1"])
        s2 = " / ".join(fmt(v, key) for v in m["stage2"])
        d1 = f"{100 * m['d1']:.1f} %" if m["d1"] is not None else "—"
        d2 = f"{100 * m['d2']:.1f} %" if m["d2"] is not None else "—"
        r2 = f"{m['r2']:+.1f}" if m["r2"] is not None and abs(m["r2"]) < 1e6 else "—"
        print(f"| {lab} | {s1} | {d1} | {s2} | {d2} | {r2} | {VERD.get(m['verdict'], m['verdict'])} |")


seq("main", "主序列：边通量算子 + 全部网格修正 + 近壁取样（κ = 1/2，含强迫项）")
seq("grid_only", "只有网格修正（不用近壁取样规则）")
V = R.get("variants_base", {})
if V:
    cols = list(V.keys())
    print("\n基准网格的选项敏感性\n")
    print("| 量 | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for key, lab in ROWS:
        if key in next(iter(V.values())):
            print(f"| {lab} | " + " | ".join(fmt(V[c].get(key), key) for c in cols) + " |")
