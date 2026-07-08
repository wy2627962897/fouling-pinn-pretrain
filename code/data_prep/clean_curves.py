"""
清理提取的曲线数据：
  1. 按时间排序
  2. 负时间裁剪为0
  3. 去除近重复点
  4. 统一输出为规范命名: source_001_fig{N}_{描述}.csv
  5. 生成清理后的预览图
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
CLEAN_DIR = CURVE_DIR / "cleaned"
PREVIEW_DIR = CURVE_DIR / "preview"

# 文件重命名映射: 原文件名 → (规范名, 标签, 图号)
RENAME_MAP = {
    # Fig. 5: Rf(t) 实验曲线 — test1..test7 对应不同实验批次
    "p1f5test1.csv":   ("source_001_fig5_rf_test1.csv",   "Fig.5 Rf test1", 5),
    "p1f5test2.csv":   ("source_001_fig5_rf_test2.csv",   "Fig.5 Rf test2", 5),
    "p1f5test3.csv":   ("source_001_fig5_rf_test3.csv",   "Fig.5 Rf test3", 5),
    "p1f5test4.csv":   ("source_001_fig5_rf_test4.csv",   "Fig.5 Rf test4", 5),
    "p1f5test5.csv":   ("source_001_fig5_rf_test5.csv",   "Fig.5 Rf test5", 5),
    "p1f5test6.csv":   ("source_001_fig5_rf_test6.csv",   "Fig.5 Rf test6", 5),
    "p1f5text7.csv":   ("source_001_fig5_rf_test7.csv",   "Fig.5 Rf test7", 5),
    # Fig. 6: U(t) 总传热系数曲线
    "p1f6test1 .csv":  ("source_001_fig6_u_test1.csv",    "Fig.6 U test1",  6),
    "p1f6test2.csv":   ("source_001_fig6_u_test2.csv",    "Fig.6 U test2",  6),
    "p1f6test3.csv":   ("source_001_fig6_u_test3.csv",    "Fig.6 U test3",  6),
    "p1f6test4.csv":   ("source_001_fig6_u_test4.csv",    "Fig.6 U test4",  6),
    "p1f6test5.csv":   ("source_001_fig6_u_test5.csv",    "Fig.6 U test5",  6),
    "p1f6test6.csv":   ("source_001_fig6_u_test6.csv",    "Fig.6 U test6",  6),
    "p1f6test7.csv":   ("source_001_fig6_u_test7.csv",    "Fig.6 U test7",  6),
    # Fig. 7: Rf 实验值 vs KS拟合
    "p1f7.csv":        ("source_001_fig7_rf_exp.csv",      "Fig.7 Rf exp",   7),
}


def clean_curve(df: pd.DataFrame, time_tol_pct: float = 0.5) -> pd.DataFrame:
    """
    清理单条曲线:
      1. 保留前两列，命名 time, value
      2. 负时间裁剪为 0
      3. 按时间排序
      4. 去除时间差 < time_tol_pct% 时间范围 的重复点（取均值）
    """
    df = df.iloc[:, :2].copy()
    df.columns = ["time", "value"]

    # 负时间裁剪
    df["time"] = df["time"].clip(lower=0)

    # 排序
    df = df.sort_values("time").reset_index(drop=True)

    # 去除完全重复的行
    df = df.drop_duplicates(subset=["time"]).reset_index(drop=True)

    return df


def summarize(df: pd.DataFrame) -> str:
    return (f"{len(df):>4d}点  t=[{df['time'].min():.1f}, {df['time'].max():.1f}]  "
            f"v=[{df['value'].min():.3e}, {df['value'].max():.3e}]  "
            f"单调: {df['time'].is_monotonic_increasing}")


def main() -> int:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("清理曲线数据 → cleaned/")
    print("=" * 65)

    cleaned = {}
    for orig_name, (new_name, label, fig_num) in sorted(RENAME_MAP.items()):
        src = CURVE_DIR / orig_name
        if not src.exists():
            print(f"  ⚠ 找不到: {orig_name}，跳过")
            continue

        df_raw = pd.read_csv(src, header=None)
        df_clean = clean_curve(df_raw)

        # 保存到 cleaned/
        dst = CLEAN_DIR / new_name
        df_clean.to_csv(dst, index=False, header=["time", "value"])

        info = summarize(df_clean)
        dropped = len(df_raw) - len(df_clean)
        drop_msg = f" (去除{dropped}点)" if dropped > 0 else ""
        print(f"  ✅ {new_name:<40s} {info}{drop_msg}")

        if fig_num not in cleaned:
            cleaned[fig_num] = []
        cleaned[fig_num].append((dst, label, df_clean))

    # ── 生成清理后的分组预览图 ──
    print(f"\n{'=' * 65}")
    print("生成清理后预览图...")
    print("=" * 65)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for fig_num in [5, 6, 7]:
        if fig_num not in cleaned:
            continue
        curves = cleaned[fig_num]

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, (path, label, df) in enumerate(curves):
            ax.scatter(df["time"], df["value"], s=12, color=colors[i % 10],
                       label=label, alpha=0.8, edgecolors="none")
            ax.plot(df["time"], df["value"], color=colors[i % 10],
                    alpha=0.3, linewidth=0.5)

        ax.set_xlabel("Time (hours)", fontsize=12)
        if fig_num == 6:
            ax.set_ylabel("U (W/m²·K)", fontsize=12)
        else:
            ax.set_ylabel("Rf (m²·K/W)", fontsize=12)
        ax.set_title(f"Source 001 — Fig. {fig_num} (cleaned)", fontsize=14)
        ax.legend(fontsize=8, ncol=2, loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = PREVIEW_DIR / f"cleaned_fig{fig_num}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  ✅ {out_path}")

    # ── 每条单独预览 ──
    for fig_num in [5, 6, 7]:
        if fig_num not in cleaned:
            continue
        for path, label, df in cleaned[fig_num]:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.scatter(df["time"], df["value"], s=15, alpha=0.8)
            ax.plot(df["time"], df["value"], alpha=0.3, linewidth=0.5)
            ax.set_xlabel("Time (hours)")
            ax.set_ylabel("Rf" if fig_num != 6 else "U")
            ax.set_title(label, fontsize=10)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(PREVIEW_DIR / f"{path.stem}_clean.png", dpi=120)
            plt.close(fig)

    print(f"\n✅ 全部清理完成。清理后文件在: {CLEAN_DIR}")
    print(f"   预览图在: {PREVIEW_DIR}")
    print(f"\n📋 下一步: 打开原论文 PDF，对比 Fig.5/6/7，确认:")
    print(f"   - 每条 'testN' 对应原图中哪条曲线（颜色/标记/工况）")
    print(f"   - Fig.7 的 p1f7.csv 是实验点还是 KS 拟合线？")
    print(f"   - 如果有 KS 拟合线需要单独提取，请单独操作")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
