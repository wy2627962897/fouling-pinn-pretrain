"""
审计所有提取的曲线CSV，检测：
  1. 时间是否单调
  2. 是否有混合曲线（时间回跳说明多个曲线混在一个文件）
  3. 值范围是否合理
  4. 数据点数量
输出每个文件的问题报告，并尝试自动分离混合曲线。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

CURVE_DIR = Path(__file__).parent.parent / "data" / "real" / "curves"


def audit_file(csv_path: Path) -> dict:
    """审计单个CSV文件，返回问题报告"""
    df = pd.read_csv(csv_path, header=None, names=["time", "value"])
    info = {
        "file": csv_path.name,
        "n_points": len(df),
        "time_range": (df["time"].min(), df["time"].max()),
        "value_range": (df["value"].min(), df["value"].max()),
        "time_monotonic": bool(df["time"].is_monotonic_increasing),
        "n_time_jumps_back": int((df["time"].diff() < 0).sum()),
        "n_near_duplicates": 0,
        "mixed_curves": False,
        "segments": [],
    }

    # 检测时间回跳（可能混合了多条曲线）
    df_sorted = df.sort_values("time").reset_index(drop=True)
    time_diffs = df_sorted["time"].diff()

    # 检测大幅时间回跳 (>5% 的时间范围)
    time_span = info["time_range"][1] - info["time_range"][0]
    big_jumps = time_diffs[time_diffs < -0.05 * time_span]
    info["n_time_jumps_back"] = len(big_jumps)

    # 如果有很多时间回跳或时间回跳很大，标记为混合曲线
    if info["n_time_jumps_back"] > 3:
        info["mixed_curves"] = True

    # 检测接近重复的点（时间差<0.1%范围内）
    for i in range(len(df_sorted) - 1):
        if abs(time_diffs.iloc[i + 1]) < time_span * 0.001 if pd.notna(time_diffs.iloc[i + 1]) else False:
            info["n_near_duplicates"] += 1

    # 尝试分离混合曲线：找到时间回跳的位置
    if info["mixed_curves"]:
        df_sorted = df.sort_values("time").reset_index(drop=True)
        jumps = [0]
        for i in range(1, len(df_sorted)):
            if df_sorted["time"].iloc[i] < df_sorted["time"].iloc[i - 1] - 0.01 * time_span:
                jumps.append(i)
        jumps.append(len(df_sorted))

        for j in range(len(jumps) - 1):
            seg = df_sorted.iloc[jumps[j] : jumps[j + 1]]
            if len(seg) >= 5:
                info["segments"].append({
                    "n": len(seg),
                    "t_range": (float(seg["time"].min()), float(seg["time"].max())),
                    "v_range": (float(seg["value"].min()), float(seg["value"].max())),
                })

    return info


def plot_all_curves(curve_files: list, title: str, out_path: Path):
    """把所有曲线画在一张图上便于人工比对"""
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(curve_files)))

    for i, f in enumerate(curve_files):
        df = pd.read_csv(f, header=None, names=["time", "value"])
        df = df.sort_values("time")
        label = f.stem
        ax.scatter(df["time"], df["value"], s=8, color=colors[i], label=label, alpha=0.7)

    ax.set_xlabel("time", fontsize=12)
    ax.set_ylabel("value", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  预览图已保存: {out_path}")


def main() -> int:
    csv_files = sorted(CURVE_DIR.glob("*.csv"))
    if not csv_files:
        print(f"在 {CURVE_DIR} 下未找到 CSV 文件")
        return 1

    # 按图号分组
    fig5_files = [f for f in csv_files if "f5" in f.stem.lower()]
    fig6_files = [f for f in csv_files if "f6" in f.stem.lower()]
    fig7_files = [f for f in csv_files if "f7" in f.stem.lower()]

    print("=" * 70)
    print("曲线数据审计报告")
    print("=" * 70)

    for label, files in [("Fig. 5 (Rf vs t)", fig5_files), ("Fig. 6 (U vs t)", fig6_files), ("Fig. 7 (Rf拟合)", fig7_files)]:
        print(f"\n{'─' * 60}")
        print(f"  {label}  ({len(files)} 个文件)")
        print(f"{'─' * 60}")

        for f in sorted(files):
            info = audit_file(f)
            flag = ""
            issues = []
            if not info["time_monotonic"]:
                issues.append(f"时间不单调({info['n_time_jumps_back']}次回跳)")
            if info["mixed_curves"]:
                issues.append(f"⚠ 混合曲线 ({len(info['segments'])}段)")
            if info["n_near_duplicates"] > 5:
                issues.append(f"近重复点{info['n_near_duplicates']}个")
            if issues:
                flag = " | ".join(issues)

            print(f"  {info['file']:<25s} {info['n_points']:>4d}点  "
                  f"t=[{info['time_range'][0]:.1f}, {info['time_range'][1]:.1f}]  "
                  f"v=[{info['value_range'][0]:.2e}, {info['value_range'][1]:.2e}]"
                  f"  {'🔴 ' + flag if flag else '✅ 正常'}")

            if info["segments"]:
                for s_idx, seg in enumerate(info["segments"]):
                    print(f"    段{s_idx+1}: {seg['n']}点, t=[{seg['t_range'][0]:.1f},{seg['t_range'][1]:.1f}], "
                          f"v=[{seg['v_range'][0]:.2e},{seg['v_range'][1]:.2e}]")

    # 生成分组预览图
    print(f"\n{'=' * 70}")
    print("生成预览图...")
    print("=" * 70)

    preview_dir = CURVE_DIR / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    if fig5_files:
        plot_all_curves(fig5_files, "Fig. 5 — Rf(t) 所有提取曲线", preview_dir / "audit_fig5_all.png")
    if fig6_files:
        plot_all_curves(fig6_files, "Fig. 6 — U(t) 所有提取曲线", preview_dir / "audit_fig6_all.png")
    if fig7_files:
        plot_all_curves(fig7_files, "Fig. 7 — Rf 实验值 vs KS拟合", preview_dir / "audit_fig7_all.png")

    # 给每张图单独画清晰预览
    for f in csv_files:
        df = pd.read_csv(f, header=None, names=["time", "value"])
        df = df.sort_values("time")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df["time"], df["value"], s=15, alpha=0.7)
        ax.plot(df["time"], df["value"], alpha=0.3, linewidth=0.5)
        ax.set_xlabel("time")
        ax.set_ylabel("value")
        ax.set_title(f.stem, fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(preview_dir / f"{f.stem}_preview.png", dpi=150)
        plt.close(fig)

    print(f"  单独预览图已保存到: {preview_dir}")
    print("\n✅ 审计完成。请查看 preview/ 目录下的预览图，"
          "对比原论文图确认每条曲线的身份。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
