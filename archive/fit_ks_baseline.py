"""
Kern-Seaton 基线拟合：对 Fig.5 的 7 条 Rf(t) 曲线拟合 KS 模型
Rf(t) = Rf* × (1 - exp(-t/τ))

同时：
  - 从 Fig.6 的 U(t) 反演 Rf(t)：Rf = 1/U(t) - 1/U_clean
  - 对比论文 Table 1 报告值 (Rf*=1.72e-4, τ=40.32h, R²=0.975)
  - 生成拟合报告图
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

CLEAN_DIR = Path(__file__).parent.parent / "data" / "real" / "curves" / "cleaned"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "real" / "fits"
PREVIEW_DIR = Path(__file__).parent.parent / "data" / "real" / "curves" / "preview"

# 论文 Table 1 报告值
PAPER_RF_STAR = 1.72e-4   # m²·K/W
PAPER_TAU = 40.32          # hours
PAPER_R2 = 0.975


def ks_model(t, rf_star, tau):
    """Kern-Seaton 渐近模型: Rf(t) = Rf* × (1 - exp(-t/τ))"""
    return rf_star * (1.0 - np.exp(-t / tau))


def fit_ks(t: np.ndarray, rf: np.ndarray) -> dict:
    """对一条曲线拟合 Kern-Seaton 模型，返回参数和统计量"""
    # 初始猜测：Rf* ≈ max(rf), τ ≈ 时间范围的 1/3
    p0 = [np.max(rf) * 1.0, (t[-1] - t[0]) / 3.0]
    bounds = ([0, 1.0], [np.max(rf) * 3.0, 500.0])

    try:
        popt, pcov = curve_fit(ks_model, t, rf, p0=p0, bounds=bounds, maxfev=10000)
        rf_star, tau = popt
        rf_pred = ks_model(t, rf_star, tau)

        # R²
        ss_res = np.sum((rf - rf_pred) ** 2)
        ss_tot = np.sum((rf - np.mean(rf)) ** 2)
        r2 = 1.0 - ss_res / ss_tot

        # RMSE
        rmse = np.sqrt(np.mean((rf - rf_pred) ** 2))

        # MAE
        mae = np.mean(np.abs(rf - rf_pred))

        # 标准差 (from pcov)
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan, np.nan]

        return {
            "rf_star": rf_star,
            "tau": tau,
            "rf_star_std": perr[0],
            "tau_std": perr[1],
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "rf_pred": rf_pred,
            "success": True,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def fit_ks_with_offset(t: np.ndarray, rf: np.ndarray, rf_offset: float = 0.0) -> dict:
    """带初始偏移的 KS 拟合: Rf(t) = Rf_offset + Rf* × (1 - exp(-t/τ))
    用于处理不完全清洗导致的初始残余热阻"""
    def model(t, rf_star, tau):
        return rf_offset + rf_star * (1.0 - np.exp(-t / tau))

    rf_adj = rf - rf_offset
    if np.any(rf_adj < 0):
        return {"success": False, "error": "负值调整后仍为负"}

    p0 = [np.max(rf_adj) * 1.0, (t[-1] - t[0]) / 3.0]
    bounds = ([0, 1.0], [np.max(rf_adj) * 3.0, 500.0])

    try:
        popt, pcov = curve_fit(model, t, rf, p0=p0, bounds=bounds, maxfev=10000)
        rf_star, tau = popt
        rf_pred = model(t, rf_star, tau)

        ss_res = np.sum((rf - rf_pred) ** 2)
        ss_tot = np.sum((rf - np.mean(rf)) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        rmse = np.sqrt(np.mean((rf - rf_pred) ** 2))
        mae = np.mean(np.abs(rf - rf_pred))
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan, np.nan]

        return {
            "rf_star": rf_star,
            "tau": tau,
            "rf_offset": rf_offset,
            "rf_star_std": perr[0],
            "tau_std": perr[1],
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "rf_pred": rf_pred,
            "success": True,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def invert_u_to_rf(df_u: pd.DataFrame, u_clean: float = None) -> pd.DataFrame:
    """从 U(t) 反演 Rf(t): Rf = 1/U(t) - 1/U_clean
    如果未提供 U_clean，使用前 5% 时间的最大 U 值作为近似"""
    if u_clean is None:
        early_mask = df_u["time"] <= df_u["time"].min() + 0.05 * (df_u["time"].max() - df_u["time"].min())
        if early_mask.sum() > 0:
            u_clean = df_u.loc[early_mask, "value"].max()
        else:
            u_clean = df_u["value"].max()

    rf = 1.0 / df_u["value"] - 1.0 / u_clean
    return pd.DataFrame({"time": df_u["time"], "value": np.clip(rf, 0, None)})


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 加载 Fig.5 清理后的曲线 ──
    fig5_files = sorted(CLEAN_DIR.glob("source_001_fig5_rf_test*.csv"))
    fig7_file = CLEAN_DIR / "source_001_fig7_rf_exp.csv"

    print("=" * 70)
    print("Kern-Seaton 基线拟合报告")
    print("=" * 70)
    print(f"\n论文 Table 1 参考值: Rf* = {PAPER_RF_STAR:.2e} m²·K/W, "
          f"τ = {PAPER_TAU:.1f} h, R² = {PAPER_R2}")
    print()

    # ── 1. 拟合 Fig.5 的每条曲线 ──
    all_fits = []
    print(f"{'曲线':<35s} {'Rf*':>12s} {'τ(h)':>8s} {'R²':>8s} {'RMSE':>10s} {'MAE':>10s}")
    print("-" * 85)

    for f in fig5_files:
        df = pd.read_csv(f)
        t = df["time"].values
        rf = df["value"].values
        label = f.stem.replace("source_001_fig5_", "")

        # 标准 KS 拟合
        result = fit_ks(t, rf)
        if result["success"]:
            all_fits.append({"file": f.stem, "label": label, "n_points": len(t),
                             "t_range": (t[0], t[-1]), **result})
            print(f"{label:<35s} {result['rf_star']:>10.2e}  "
                  f"{result['tau']:>8.1f}  {result['r2']:>8.4f}  "
                  f"{result['rmse']:>10.2e}  {result['mae']:>10.2e}")
        else:
            print(f"{label:<35s} 拟合失败: {result.get('error', 'unknown')}")

    # ── 2. 对 KS 拟合线 (fig7) 做拟合验证 ──
    print(f"\n--- Fig.7 KS 拟合线验证 ---")
    if fig7_file.exists():
        df7 = pd.read_csv(fig7_file)
        t7 = df7["time"].values
        rf7 = df7["value"].values
        result7 = fit_ks(t7, rf7)
        if result7["success"]:
            print(f"  Fig.7 KS线: Rf*={result7['rf_star']:.4e}, τ={result7['tau']:.1f}h, "
                  f"R²={result7['r2']:.6f} (应对应论文 Table 1)")
            print(f"  与论文偏差: ΔRf*={result7['rf_star']-PAPER_RF_STAR:.2e} "
                  f"({100*(result7['rf_star']-PAPER_RF_STAR)/PAPER_RF_STAR:.1f}%), "
                  f"Δτ={result7['tau']-PAPER_TAU:.1f}h "
                  f"({100*(result7['tau']-PAPER_TAU)/PAPER_TAU:.1f}%)")

    # ── 3. 从 Fig.6 U(t) 反演 Rf(t) 并拟合 ──
    print(f"\n--- Fig.6 U(t) → Rf(t) 反演 ---")
    fig6_files = sorted(CLEAN_DIR.glob("source_001_fig6_u_test*.csv"))

    for f in fig6_files:
        df = pd.read_csv(f)
        rf_from_u = invert_u_to_rf(df)
        t = rf_from_u["time"].values
        rf = rf_from_u["value"].values
        label = f.stem.replace("source_001_fig6_", "").replace("u_", "rf_from_u_")

        result = fit_ks(t, rf)
        if result["success"]:
            print(f"  {label:<30s} U_clean≈{1.0/(rf[0]+1.0/df['value'].max()):.0f} W/m²·K  "
                  f"Rf*={result['rf_star']:.2e}  τ={result['tau']:.1f}h  R²={result['r2']:.4f}")
        else:
            print(f"  {label:<30s} 反演拟合失败: {result.get('error', 'unknown')}")

    # ── 4. 统计汇总 ──
    print(f"\n{'=' * 70}")
    print("Fig.5 拟合统计汇总")
    print("=" * 70)
    rf_stars = [f["rf_star"] for f in all_fits]
    taus = [f["tau"] for f in all_fits]
    r2s = [f["r2"] for f in all_fits]
    print(f"  Rf*: 均值={np.mean(rf_stars):.2e}, 范围=[{np.min(rf_stars):.2e}, {np.max(rf_stars):.2e}]")
    print(f"  τ:   均值={np.mean(taus):.1f}h, 范围=[{np.min(taus):.1f}, {np.max(taus):.1f}]h")
    print(f"  R²:  均值={np.mean(r2s):.4f}, 范围=[{np.min(r2s):.4f}, {np.max(r2s):.4f}]")
    print(f"  论文报告: Rf*={PAPER_RF_STAR:.2e}, τ={PAPER_TAU:.1f}h")

    # ── 5. 保存拟合结果 JSON ──
    fit_records = []
    for f in all_fits:
        fit_records.append({
            "curve": f["label"],
            "n_points": f["n_points"],
            "rf_star": float(f["rf_star"]),
            "tau": float(f["tau"]),
            "rf_star_std": float(f["rf_star_std"]),
            "tau_std": float(f["tau_std"]),
            "r2": float(f["r2"]),
            "rmse": float(f["rmse"]),
            "mae": float(f["mae"]),
            "t_range": [float(f["t_range"][0]), float(f["t_range"][1])],
        })
    with open(OUTPUT_DIR / "ks_fits_fig5.json", "w") as fp:
        json.dump(fit_records, fp, indent=2, ensure_ascii=False)
    print(f"\n拟合结果已保存: {OUTPUT_DIR / 'ks_fits_fig5.json'}")

    # ── 6. 可视化 ──
    print(f"\n生成拟合图...")
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # 6a. 每条曲线单独拟合图
    for f_data, fit_info in zip(fig5_files, all_fits):
        df = pd.read_csv(f_data)
        label = fit_info["label"]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(df["time"], df["value"], s=15, alpha=0.7, label="实验数据", color="steelblue")
        t_smooth = np.linspace(0, df["time"].max(), 200)
        ax.plot(t_smooth, ks_model(t_smooth, fit_info["rf_star"], fit_info["tau"]),
                "r-", linewidth=2, label=f"KS拟合 (Rf*={fit_info['rf_star']:.2e}, τ={fit_info['tau']:.1f}h)")
        ax.set_xlabel("Time (hours)", fontsize=12)
        ax.set_ylabel("Rf (m²·K/W)", fontsize=12)
        ax.set_title(f"{label}  —  R²={fit_info['r2']:.4f}", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(PREVIEW_DIR / f"{label}_ks_fit.png", dpi=150)
        plt.close(fig)

    # 6b. 汇总对比图：所有曲线 + KS 拟合
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    for i, (f_data, fit_info) in enumerate(zip(fig5_files, all_fits)):
        df = pd.read_csv(f_data)
        ax = axes[i]
        ax.scatter(df["time"], df["value"], s=10, alpha=0.6, color=colors[i])
        t_smooth = np.linspace(0, df["time"].max(), 200)
        ax.plot(t_smooth, ks_model(t_smooth, fit_info["rf_star"], fit_info["tau"]),
                "r-", linewidth=1.5)
        ax.set_title(f"{fit_info['label']}\nRf*={fit_info['rf_star']:.2e}, τ={fit_info['tau']:.0f}h, R²={fit_info['r2']:.3f}",
                     fontsize=8)
        ax.grid(True, alpha=0.3)
    # 隐藏多余的 subplot
    axes[7].set_visible(False)
    fig.suptitle("Fig.5 — 7 条 Rf(t) 曲线的 Kern-Seaton 拟合 (source_001)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(PREVIEW_DIR / "summary_ks_fits.png", dpi=150)
    plt.close(fig)

    # 6c. Rf* 和 τ 的分布 vs 论文报告值
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    labels = [f["label"].replace("rf_test", "T") for f in all_fits]
    x = np.arange(len(labels))

    # Rf* 对比
    bars1 = ax1.bar(x, [f["rf_star"] for f in all_fits], color="steelblue", alpha=0.7)
    ax1.axhline(y=PAPER_RF_STAR, color="red", linestyle="--", linewidth=2,
                label=f"论文 Table 1: {PAPER_RF_STAR:.2e}")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Rf* (m²·K/W)", fontsize=12)
    ax1.set_title("渐近污垢热阻 Rf* 对比", fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    # τ 对比
    bars2 = ax2.bar(x, [f["tau"] for f in all_fits], color="coral", alpha=0.7)
    ax2.axhline(y=PAPER_TAU, color="red", linestyle="--", linewidth=2,
                label=f"论文 Table 1: {PAPER_TAU:.1f}h")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("τ (hours)", fontsize=12)
    ax2.set_title("时间常数 τ 对比", fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Kern-Seaton 拟合参数 vs 论文报告值", fontsize=14)
    fig.tight_layout()
    fig.savefig(PREVIEW_DIR / "ks_params_vs_paper.png", dpi=150)
    plt.close(fig)

    print(f"  预览图已保存到: {PREVIEW_DIR}")
    print(f"\n✅ Kern-Seaton 基线拟合完成。")

    # ── 建议 ──
    print(f"\n📋 关键发现：")
    if np.mean(rf_stars) < PAPER_RF_STAR * 0.8:
        print(f"  ⚠ 拟合的 Rf* 均值显著低于论文报告值，检查坐标轴校准或单位")
    elif np.mean(rf_stars) > PAPER_RF_STAR * 1.2:
        print(f"  ⚠ 拟合的 Rf* 均值显著高于论文报告值，检查坐标轴校准或单位")
    else:
        print(f"  ✅ Rf* 均值与论文 Table 1 偏差在合理范围内")

    # 建议：是否需要带初始偏移的拟合
    low_r2 = [f for f in all_fits if f["r2"] < 0.7]
    if low_r2:
        print(f"  💡 {len(low_r2)} 条曲线 R²<0.7 ({[f['label'] for f in low_r2]})，")
        print(f"     可能因为运行波动(流速变化)导致偏离理想KS曲线，"  )
        print(f"     这是工业数据的正常特征，无需强制拟合完美。")


if __name__ == "__main__":
    main()
