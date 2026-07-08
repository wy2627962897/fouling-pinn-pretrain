"""
结垢仿真可视化
依赖: fouling_simulator.py (纯 Python), matplotlib (画图)
"""

import math
from pathlib import Path
import fouling_simulator as fs

OUTPUT_DIR = Path(__file__).parent / "output"


def ensure_output():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_rf_curve(results, title, filepath, threshold_rf=None):
    """单条 Rf(t) 曲线, 标注清洗阈值"""
    import matplotlib.pyplot as plt

    days = [r["day"] for r in results]
    rf_vals = [r["Rf"] * 1e3 for r in results]  # 转换为 1e-3 m2.K/W (更可读)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(days, rf_vals, "b-", linewidth=2)
    ax.set_xlabel("运行时间 (天)", fontsize=12)
    ax.set_ylabel("污垢热阻 Rf × 10^3 (m^2·K/W)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    if threshold_rf is not None:
        thresh = threshold_rf * 1e3
        ax.axhline(y=thresh, color="r", linestyle="--", linewidth=1.5, label=f"清洗阈值 = {thresh:.0f}")

        # 找到首次超过阈值的时间
        for r in results:
            if r["Rf"] >= threshold_rf:
                ax.axvline(x=r["day"], color="r", linestyle=":", alpha=0.5)
                ax.annotate(
                    f"首次达阈值\n第 {r['day']:.0f} 天",
                    xy=(r["day"], thresh),
                    xytext=(r["day"] + 200, thresh + 0.5),
                    arrowprops=dict(arrowstyle="->", color="red"),
                    fontsize=10,
                    color="red",
                )
                break

    ax.legend(fontsize=11)
    fig.tight_layout()
    ensure_output()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  已保存: {filepath}")


def plot_multi_condition(results_list, title, filepath, threshold_rf=None):
    """多条 Rf(t) 曲线对比 (不同工况)"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, results in enumerate(results_list):
        color = colors[i % len(colors)]
        days = [r["day"] for r in results]
        rf_vals = [r["Rf"] * 1e3 for r in results]
        label = results[0]["label"] if results else f"工况 {i+1}"
        ax.plot(days, rf_vals, color=color, linewidth=1.8, label=label)

        # 在曲线末端标注
        ax.annotate(
            f"{rf_vals[-1]:.1f}",
            xy=(days[-1], rf_vals[-1]),
            fontsize=8,
            color=color,
            ha="left",
        )

    ax.set_xlabel("运行时间 (天)", fontsize=12)
    ax.set_ylabel("污垢热阻 Rf × 10^3 (m^2·K/W)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)

    if threshold_rf is not None:
        ax.axhline(y=threshold_rf * 1e3, color="red", linestyle="--", linewidth=1, alpha=0.7, label="清洗阈值")

    fig.tight_layout()
    ensure_output()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  已保存: {filepath}")


def plot_U_decay(results, title, filepath):
    """传热系数 U 随时间衰减"""
    import matplotlib.pyplot as plt

    days = [r["day"] for r in results]
    U_vals = [r["U"] for r in results]
    U_pct = [r["U_pct"] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 上: U 绝对值
    ax1.plot(days, U_vals, "b-", linewidth=2)
    ax1.set_ylabel("总传热系数 U (W/m^2·K)", fontsize=11)
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.annotate(
        f"U_clean = {U_vals[0]:.0f}",
        xy=(0, U_vals[0]),
        fontsize=10,
        color="blue",
    )
    ax1.annotate(
        f"U_final = {U_vals[-1]:.0f}",
        xy=(days[-1], U_vals[-1]),
        fontsize=10,
        color="red",
    )

    # 下: U 百分比
    ax2.plot(days, U_pct, "r-", linewidth=2)
    ax2.set_xlabel("运行时间 (天)", fontsize=12)
    ax2.set_ylabel("U / U_clean (%)", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    ax2.axhline(y=80, color="gray", linestyle="--", alpha=0.5, label="80% 设计能力")
    ax2.legend(fontsize=10)

    fig.tight_layout()
    ensure_output()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  已保存: {filepath}")


def plot_cleaning_cost_analysis(results, cleaning_cost_per_m2, energy_cost_factor, filepath):
    """
    清洗成本 vs 能耗损失 累积对比
    cleaning_cost_per_m2: 每次清洗费用 (元/m2)
    energy_cost_factor: 能耗成本系数 (元/(W·day))
    """
    import matplotlib.pyplot as plt

    days = [r["day"] for r in results]
    Rf_vals = [r["Rf"] for r in results]
    base_q = 1e5  # 假设基础换热量 W

    # 方案1: 频繁清洗 (每 200 天)
    interval_freq = 200
    # 方案2: 中等频率 (每 500 天)
    interval_mid = 500
    # 方案3: 稀疏清洗 (每 1000 天)
    interval_sparse = 1000

    def calc_total_cost(interval):
        """计算给定清洗间隔下的总成本 (简化)"""
        total_days = days[-1]
        n_cleanings = int(total_days // interval)
        cleaning_cost = n_cleanings * cleaning_cost_per_m2

        # 能耗损失: 对 Rf(t) 积分 (清洗后 Rf 重置为 0)
        energy_loss = 0
        for i in range(1, len(days)):
            cycle_day = days[i] % interval
            idx = min(range(len(days)), key=lambda j: abs(days[j] - cycle_day))
            rf = Rf_vals[idx]
            dt = days[i] - days[i - 1]  # 天
            energy_loss += rf * energy_cost_factor * dt

        return cleaning_cost, energy_loss

    intervals = [200, 400, 600, 800, 1000, 1200, 1500, 1800]
    total_costs = []
    cleaning_costs = []
    energy_losses = []

    for interval in intervals:
        cc, el = calc_total_cost(interval)
        cleaning_costs.append(cc)
        energy_losses.append(el)
        total_costs.append(cc + el)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(intervals, total_costs, "k-o", linewidth=2, markersize=8, label="总成本")
    ax.plot(intervals, cleaning_costs, "b--s", linewidth=1.5, label="清洗费用")
    ax.plot(intervals, energy_losses, "r--^", linewidth=1.5, label="能耗损失")

    # 标注最优点
    min_idx = total_costs.index(min(total_costs))
    ax.annotate(
        f"最优间隔: {intervals[min_idx]} 天\n总成本: {total_costs[min_idx]:.0f} 元",
        xy=(intervals[min_idx], total_costs[min_idx]),
        xytext=(intervals[min_idx] + 200, total_costs[min_idx] + total_costs[min_idx] * 0.05),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=11,
        fontweight="bold",
    )

    ax.set_xlabel("清洗间隔 (天)", fontsize=12)
    ax.set_ylabel("成本 (元)", fontsize=12)
    ax.set_title("清洗周期优化: 能耗损失 vs 清洗费用", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    ensure_output()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  已保存: {filepath}")

    # 输出最优值
    print(f"\n  最优清洗间隔: {intervals[min_idx]} 天")
    print(f"    清洗费用:    {cleaning_costs[min_idx]:.0f} 元")
    print(f"    能耗损失:    {energy_losses[min_idx]:.0f} 元")
    print(f"    总成本:      {total_costs[min_idx]:.0f} 元")


# ===================================================================
# 主程序
# ===================================================================
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")  # 非交互后端

    # 设置中文字体 (Windows)
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
    plt.rcParams["axes.unicode_minus"] = False

    base = fs.default_params()

    # ---- 图1: 基准工况 Rf(t) ----
    print("=== 图1: 基准工况 Rf(t) 曲线 ===")
    res_base = fs.simulate(base)
    threshold = fs.asymptotic_rf(
        base["A"], base["Ea"], base["T_w"], base["B"], base["rho"], base["v"]
    ) * 0.8  # 清洗阈值 = 80% 的渐近值
    plot_rf_curve(
        res_base,
        title=f"换热器结垢演化 (Tw={base['T_w']}K, v={base['v']}m/s)",
        filepath=OUTPUT_DIR / "fig1_rf_baseline.png",
        threshold_rf=threshold,
    )

    # ---- 图2: 不同壁温对比 ----
    print("\n=== 图2: 不同壁温下 Rf(t) 对比 ===")
    temps = [333, 353, 363, 373, 393]
    temp_labels = [f"Tw={t}K ({t-273:.0f}°C)" for t in temps]
    temp_results = [fs.simulate(p) for p in fs.vary_param(base, "T_w", temps, temp_labels)]
    plot_multi_condition(
        temp_results,
        title="壁温对结垢速率的影响",
        filepath=OUTPUT_DIR / "fig2_rf_vs_temperature.png",
    )

    # ---- 图3: 不同流速对比 ----
    print("\n=== 图3: 不同流速下 Rf(t) 对比 ===")
    vels = [0.5, 1.0, 1.5, 2.0, 3.0]
    vel_labels = [f"v={v} m/s" for v in vels]
    vel_results = [fs.simulate(p) for p in fs.vary_param(base, "v", vels, vel_labels)]
    plot_multi_condition(
        vel_results,
        title="流速对结垢速率的影响",
        filepath=OUTPUT_DIR / "fig3_rf_vs_velocity.png",
    )

    # ---- 图4: U 衰减 ----
    print("\n=== 图4: 传热系数 U 随时间衰减 ===")
    plot_U_decay(
        res_base,
        title=f"传热系数衰减 (Tw={base['T_w']}K, v={base['v']}m/s)",
        filepath=OUTPUT_DIR / "fig4_U_decay.png",
    )

    # ---- 图5: 清洗成本优化 ----
    print("\n=== 图5: 清洗周期成本优化 ===")
    plot_cleaning_cost_analysis(
        res_base,
        cleaning_cost_per_m2=500,   # 每次清洗 500 元/m2
        energy_cost_factor=2e4,     # 能耗成本系数
        filepath=OUTPUT_DIR / "fig5_cleaning_optimization.png",
    )

    # ---- 图6: 线性 vs Kern-Seaton 对比 ----
    print("\n=== 图6: 线性结垢 vs Kern-Seaton ===")
    linear_params = dict(base)
    linear_params["mode"] = "linear"
    linear_params["dep_rate"] = 3e-11  # m2.K/(W.s), 与 KS 初始斜率接近
    linear_params["label"] = "线性模型 (恒定速率)"
    res_linear = fs.simulate(linear_params)
    plot_multi_condition(
        [res_base, res_linear],
        title="线性模型 vs Kern-Seaton 渐近模型",
        filepath=OUTPUT_DIR / "fig6_linear_vs_ks.png",
    )

    print("\n=== 全部图表已生成 ===")
    print(f"输出目录: {OUTPUT_DIR}")
