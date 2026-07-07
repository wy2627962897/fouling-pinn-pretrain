"""
Source 003 (CaCO3 结晶结垢 — 表面改性) 数据处理全流程

特征: 诱导期、负初始 Rf、多种表面改性对比
分析重点: 诱导时间、Rf 穿越零点时间、涂层/打磨/图案的效果对比
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

CURVE_DIR = Path(__file__).parent.parent / "data" / "real" / "curves"
CLEAN_DIR = CURVE_DIR / "cleaned"
PREVIEW_DIR = CURVE_DIR / "preview"
FITS_DIR = Path(__file__).parent.parent / "data" / "real" / "fits"

# Source 003 文件映射
SRC3_FILES = {
    # Fig.4: 未改性不锈钢重复实验
    "p3f4r1.csv": ("source_003_fig4_unmodified_run1", "Fig.4 Unmodified SS run1", "fig4"),
    "p3f4r2.csv": ("source_003_fig4_unmodified_run2", "Fig.4 Unmodified SS run2", "fig4"),
    "p3f4r3.csv": ("source_003_fig4_unmodified_run3", "Fig.4 Unmodified SS run3", "fig4"),
    # Fig.5: 涂层表面
    "p3f5ref.csv": ("source_003_fig5_ref", "Fig.5 Reference (uncoated)", "fig5"),
    "p3f5l1.csv":  ("source_003_fig5_coating_l1", "Fig.5 Coating L1", "fig5"),
    "p3f5l2.csv":  ("source_003_fig5_coating_l2", "Fig.5 Coating L2", "fig5"),
    "p3f5l3.csv":  ("source_003_fig5_coating_l3", "Fig.5 Coating L3", "fig5"),
    # Fig.11: 打磨/抛光
    "p3f11ref.csv":    ("source_003_fig11_ref", "Fig.11 Reference", "fig11"),
    "p3f11grit80.csv": ("source_003_fig11_grit80", "Fig.11 Grit 80", "fig11"),
    "p3f11grit220.csv":("source_003_fig11_grit220", "Fig.11 Grit 220", "fig11"),
    "p3f11diapro.csv": ("source_003_fig11_diapro", "Fig.11 DiaPro polished", "fig11"),
    # Fig.13: 图案化表面
    "p3f13flat.csv":     ("source_003_fig13_flat", "Fig.13 Flat reference", "fig13"),
    "p3f13pattern1.csv": ("source_003_fig13_pattern1", "Fig.13 Pattern #1", "fig13"),
    "p3f13pattern2.csv": ("source_003_fig13_pattern2", "Fig.13 Pattern #2", "fig13"),
    "p3f13pattern3.csv": ("source_003_fig13_pattern3", "Fig.13 Pattern #3", "fig13"),
}


def clean_curve(df):
    df = df.iloc[:, :2].copy()
    df.columns = ["time", "value"]
    df["time"] = df["time"].clip(lower=0)
    df = df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    return df


def detect_induction_time(t: np.ndarray, rf: np.ndarray) -> dict:
    """
    分析诱导期特征:
      - t_cross_zero: Rf 从负穿越到正的时间
      - rf_min: 最小 Rf (最大负值, 表征粗糙度增强效果)
      - t_at_min: 达到最小 Rf 的时间
      - rf_init: 初始 Rf
      - rf_final: 最终 Rf
      - trend: 'negative_stay', 'cross_to_positive', 'positive_growth', 'fluctuating'
    """
    result = {
        "rf_min": float(np.min(rf)),
        "rf_max": float(np.max(rf)),
        "rf_init": float(rf[0]),
        "rf_final": float(rf[-1]),
        "t_range": (float(t[0]), float(t[-1])),
        "t_cross_zero": None,
        "t_at_min": float(t[np.argmin(rf)]),
        "trend": "unknown",
    }

    # 检测穿越零点
    signs = np.sign(rf)
    for i in range(1, len(signs)):
        if signs[i-1] <= 0 and signs[i] > 0:
            # 线性插值
            frac = abs(rf[i-1]) / (abs(rf[i-1]) + abs(rf[i]) + 1e-20)
            result["t_cross_zero"] = float(t[i-1] + frac * (t[i] - t[i-1]))
            break

    # 趋势分类
    if rf[-1] < rf[0] - 0.1 * abs(rf[0]):
        result["trend"] = "negative_stay"        # 持续为负 (长诱导期)
    elif result["t_cross_zero"] is not None:
        result["trend"] = "cross_to_positive"    # 穿越零点 → 诱导期后开始结垢
    elif rf[-1] > rf[0] * 1.5 and rf[0] > 0:
        result["trend"] = "positive_growth"      # 一直在正区间增长
    else:
        result["trend"] = "fluctuating"          # 波动无明显趋势

    # 诱导时间: 穿越零点的时间, 或 t_at_min
    result["induction_time"] = result["t_cross_zero"] or result["t_at_min"]

    return result


def main():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    FITS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Source 003 — CaCO3 结晶结垢 (表面改性) 数据分析")
    print("=" * 70)

    cleaned = {}
    induction_results = {}

    # ====== 1. 审计 & 清理 ======
    print(f"\n{'File':<45s} {'Points':>6s} {'t_range':>20s} {'Rf_range':>25s}")
    print("-" * 100)

    for orig_name, (new_name, label, group) in SRC3_FILES.items():
        src = CURVE_DIR / orig_name
        if not src.exists():
            print(f"  ⚠ 找不到: {orig_name}")
            continue

        df_raw = pd.read_csv(src, header=None)
        df = clean_curve(df_raw)
        dst = CLEAN_DIR / (new_name + ".csv")
        df.to_csv(dst, index=False, header=["time", "value"])

        t, rf = df["time"].values, df["value"].values
        print(f"  {new_name:<43s} {len(df):>6d}  "
              f"[{t[0]:.1f}, {t[-1]:.1f}]  "
              f"[{rf.min():.4f}, {rf.max():.4f}]")

        cleaned[new_name] = (label, group, df, t, rf)

    # ====== 2. 诱导期分析 ======
    print(f"\n{'=' * 70}")
    print("诱导期分析")
    print("=" * 70)

    for new_name, (label, group, df, t, rf) in cleaned.items():
        ind = detect_induction_time(t, rf)
        induction_results[new_name] = {"label": label, "group": group, **ind}
        cross_str = f"t_cross={ind['t_cross_zero']:.1f}h" if ind['t_cross_zero'] else "未穿越零点"
        print(f"  {label:<35s} {ind['trend']:<20s} {cross_str:<20s} "
              f"rf_min={ind['rf_min']:.4f}  rf_max={ind['rf_max']:.4f}")

    # ====== 3. 按组汇总 ======
    print(f"\n{'=' * 70}")
    print("表面改性效果对比")
    print("=" * 70)

    for group_name, group_label in [("fig4", "Fig.4 未改性SS重复性"),
                                      ("fig5", "Fig.5 涂层效果"),
                                      ("fig11", "Fig.11 打磨/抛光效果"),
                                      ("fig13", "Fig.13 图案化效果")]:
        group_items = [(k, v) for k, v in induction_results.items() if v["group"] == group_name]
        if not group_items:
            continue

        print(f"\n  {group_label}:")
        print(f"  {'曲线':<30s} {'趋势':<18s} {'诱导时间':<15s} {'rf_min':<10s} {'rf_max':<10s}")
        print(f"  {'─' * 80}")
        for k, v in group_items:
            ind_t = f"{v['induction_time']:.1f}h" if v['induction_time'] else "N/A"
            print(f"  {v['label']:<30s} {v['trend']:<18s} {ind_t:<15s} "
                  f"{v['rf_min']:<10.4f} {v['rf_max']:<10.4f}")

        # 组内对比：诱导时间延长倍数 (相对于 reference)
        ref_item = None
        for k, v in group_items:
            if "ref" in k.lower() or "unmodified" in k.lower():
                ref_item = v
                break

        if ref_item and ref_item["induction_time"]:
            ref_ind = ref_item["induction_time"]
            print(f"\n  相对于 Reference 的诱导时间延长倍数:")
            for k, v in group_items:
                if v["induction_time"] and v is not ref_item:
                    ratio = v["induction_time"] / ref_ind if ref_ind > 0 else float("inf")
                    print(f"    {v['label']:<30s} {ratio:.1f}×")

    # ====== 4. 跨数据集类型总结 ======
    print(f"\n{'=' * 70}")
    print("三数据集结垢类型总结")
    print("=" * 70)
    print(f"  Source 001 (磷酸浓缩):  渐近型, 无诱导期, Rf ~1.8e-4 m²·K/W, τ~44h")
    print(f"  Source 002 (原油预热):  渐近+sawtooth, 无诱导期, Rf ~3e-3 m²·K/W, 工业扰动")
    print(f"  Source 003 (CaCO3结晶): 诱导期型, 负初始Rf, Rf ~1-5e-2 m²·K/W, 表面改性可延长诱导期")

    n_cross = sum(1 for v in induction_results.values() if v["trend"] == "cross_to_positive")
    n_neg = sum(1 for v in induction_results.values() if v["trend"] == "negative_stay")
    print(f"\n  Source 003 统计: {len(induction_results)}条曲线, "
          f"{n_cross}条穿越零点, {n_neg}条持续负值(长诱导期)")

    # ====== 5. 可视化 ======
    print(f"\n生成可视化...")

    # 5a. 分组图
    for group_name, group_label in [("fig4", "Fig.4 Unmodified SS"),
                                      ("fig5", "Fig.5 Coatings"),
                                      ("fig11", "Fig.11 Grinding/Polishing"),
                                      ("fig13", "Fig.13 Patterns")]:
        group_items = [(k, v) for k, v in induction_results.items() if v["group"] == group_name]
        if not group_items:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(group_items), 3)))

        for i, (k, v) in enumerate(group_items):
            df = cleaned[k][2]
            ax.plot(df["time"], df["value"], color=colors[i], linewidth=1.2, alpha=0.85, label=v["label"])
            # 标记诱导时间
            if v["t_cross_zero"]:
                ax.axvline(x=v["t_cross_zero"], color=colors[i], linestyle=":", alpha=0.4)
            # 标记零点
            ax.scatter([v["t_at_min"]], [v["rf_min"]], color=colors[i], s=30, marker="v", alpha=0.7)

        ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
        ax.set_xlabel("Time (hours)", fontsize=12)
        ax.set_ylabel("Rf (m²·K/W)", fontsize=12)
        ax.set_title(f"Source 003 — {group_label}", fontsize=13)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(PREVIEW_DIR / f"source003_{group_name}.png", dpi=150)
        plt.close(fig)

    # 5b. 诱导时间对比柱状图
    fig, ax = plt.subplots(figsize=(14, 6))
    all_items = sorted(induction_results.items(), key=lambda x: x[1]["induction_time"] or 0)
    labels = [v["label"].replace("Fig.", "") for _, v in all_items]
    ind_times = [v["induction_time"] or 0 for _, v in all_items]
    trends = [v["trend"] for _, v in all_items]

    bar_colors = []
    for tr in trends:
        if tr == "cross_to_positive":
            bar_colors.append("steelblue")
        elif tr == "negative_stay":
            bar_colors.append("coral")
        else:
            bar_colors.append("gray")

    bars = ax.barh(labels, ind_times, color=bar_colors, alpha=0.8)
    ax.set_xlabel("Induction time (hours)", fontsize=12)
    ax.set_title("Source 003 — 诱导时间对比 (蓝=穿越零点, 红=持续负值/长诱导期)", fontsize=13)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(PREVIEW_DIR / "source003_induction_times.png", dpi=150)
    plt.close(fig)

    # 5c. 三数据集 Rf 量级对比
    fig, ax = plt.subplots(figsize=(8, 5))
    datasets = ["Source 001\n磷酸浓缩", "Source 002\n原油预热", "Source 003\nCaCO3结晶"]
    rf_ranges = [(1.6e-4, 2.0e-4), (0.9e-3, 3.5e-3), (0.5e-2, 5e-2)]
    colors_ds = ["steelblue", "coral", "green"]
    for i, (ds, (lo, hi), c) in enumerate(zip(datasets, rf_ranges, colors_ds)):
        ax.bar(i, hi, color=c, alpha=0.5, width=0.5)
        ax.text(i, hi * 1.1, f"~{hi:.1e}", ha="center", fontsize=9)

    ax.set_xticks(range(3))
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel("Rf (m²·K/W)", fontsize=12)
    ax.set_title("三个数据集 Rf 量级对比", fontsize=13)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(PREVIEW_DIR / "three_datasets_rf_scale.png", dpi=150)
    plt.close(fig)

    # ====== 6. 保存 ======
    output = {}
    for k, v in induction_results.items():
        output[k] = {key: val for key, val in v.items() if key != "group"}
    with open(FITS_DIR / "induction_analysis_source003.json", "w") as fp:
        json.dump(output, fp, indent=2, ensure_ascii=False)

    print(f"\n✅ Source 003 全流程完成。")
    print(f"   分析结果: {FITS_DIR / 'induction_analysis_source003.json'}")
    print(f"\n📋 关键发现:")
    print(f"   1. CaCO3 结晶结垢具有明显诱导期，与 source_001/002 的 '无诱导期' 形成对比")
    print(f"   2. 表面改性(涂层/打磨/图案)可显著延长诱导时间")
    print(f"   3. 三个数据集覆盖了结垢的三种主要模式: 渐近型、扰动型、诱导期型")


if __name__ == "__main__":
    main()
