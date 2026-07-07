"""
Kern-Seaton 结垢模型仿真器
dRf/dt = phi_d - phi_r
phi_d = A * exp(-Ea/(R*T_w))          沉积速率 (Arrhenius)
phi_r = B * tau_s * Rf                剥离速率 (正比于当前污垢)
tau_s = (f/2) * rho * v^2             壁面剪切应力 (Darcy摩擦因子)

解析解: Rf(t) = Rf_inf * (1 - exp(-t/tau))
Rf_inf = phi_d / (B * tau_s)
tau    = 1 / (B * tau_s)

用纯 Python + math, 零额外依赖 (numpy 也不需要)。
"""

import math
import csv
from pathlib import Path

# ---------------------------------------------------------------------------
# 物理常数
# ---------------------------------------------------------------------------
R_GAS = 8.314  # J/(mol.K), 理想气体常数


# ---------------------------------------------------------------------------
# Kern-Seaton 模型核心
# ---------------------------------------------------------------------------

def wall_shear_stress(v, rho, f=0.005):
    """壁面剪切应力 tau_s = (f/2) * rho * v^2"""
    return 0.5 * f * rho * v * v


def phi_d(A, Ea, T_w):
    """沉积速率 [m^2.K/(J.s)], Arrhenius 形式"""
    return A * math.exp(-Ea / (R_GAS * T_w))


def phi_r(B, tau_s, Rf):
    """剥离速率 [m^2.K/(J.s)], 正比于当前污垢热阻"""
    return B * tau_s * Rf


def asymptotic_rf(A, Ea, T_w, B, rho, v, f=0.005):
    """渐近污垢热阻 Rf_inf [m^2.K/W]"""
    ts = wall_shear_stress(v, rho, f)
    pd = phi_d(A, Ea, T_w)
    return pd / (B * ts)


def time_constant(B, rho, v, f=0.005):
    """时间常数 tau [s], = 1/(B * tau_s)"""
    ts = wall_shear_stress(v, rho, f)
    return 1.0 / (B * ts)


def Rf_at_time(t, A, Ea, T_w, B, rho, v, f=0.005):
    """t 时刻的污垢热阻 [m^2.K/W]"""
    rf_inf = asymptotic_rf(A, Ea, T_w, B, rho, v, f)
    tau_c = time_constant(B, rho, v, f)
    if tau_c <= 0:
        return 0.0
    return rf_inf * (1.0 - math.exp(-t / tau_c))


def Rf_at_time_linear(t, deposition_rate):
    """线性结垢模型 (对比用): Rf(t) = k * t"""
    return deposition_rate * t


# ---------------------------------------------------------------------------
# 换热器整体性能
# ---------------------------------------------------------------------------

def overall_heat_transfer_coeff(U_clean, Rf):
    """当前总传热系数 U = 1 / (1/U_clean + Rf)"""
    return 1.0 / (1.0 / U_clean + Rf)


# ---------------------------------------------------------------------------
# 仿真运行
# ---------------------------------------------------------------------------

def simulate(params, duration_days=365 * 5, dt_days=1.0):
    """
    运行结垢仿真，返回时间序列。

    参数
    ----
    params : dict, 包含:
        A       : 沉积速率系数 [m^2.K/J]
        Ea      : 活化能 [J/mol]
        T_w     : 壁温 [K]
        B       : 剥离系数 [1/Pa]
        rho     : 流体密度 [kg/m^3]
        v       : 流速 [m/s]
        f       : Darcy 摩擦因子 (默认 0.005)
        U_clean : 清洁换热器总传热系数 [W/(m^2.K)]
        mode    : 'kern-seaton' 或 'linear'
        dep_rate: 线性结垢速率 (仅 linear 模式)
        label   : 工况标签

    返回
    ----
    list[dict]: 每个时间步的 {day, Rf, U, U_pct}
    """
    A = params.get("A", 1e-8)
    Ea = params.get("Ea", 50000)
    T_w = params.get("T_w", 373)
    B = params.get("B", 1e-12)
    rho = params.get("rho", 1000)
    v = params.get("v", 1.0)
    f = params.get("f", 0.005)
    U_clean = params.get("U_clean", 1000)
    mode = params.get("mode", "kern-seaton")
    dep_rate = params.get("dep_rate", 1e-7)
    label = params.get("label", "default")

    sec_per_day = 86400
    duration_sec = duration_days * sec_per_day
    dt_sec = dt_days * sec_per_day

    results = []
    t_sec = 0.0
    while t_sec <= duration_sec:
        if mode == "kern-seaton":
            rf = Rf_at_time(t_sec, A, Ea, T_w, B, rho, v, f)
        else:
            rf = Rf_at_time_linear(t_sec, dep_rate)

        U = overall_heat_transfer_coeff(U_clean, rf)
        day = t_sec / sec_per_day
        results.append({
            "day": day,
            "Rf": rf,
            "U": U,
            "U_pct": 100 * U / U_clean,
            "label": label,
        })
        t_sec += dt_sec

    return results


# ---------------------------------------------------------------------------
# 多工况对比
# ---------------------------------------------------------------------------

def default_params():
    """
    返回一组合理的默认参数 (CaCO3 结晶结垢典型值)。

    这些参数产生的典型行为:
      - Rf_inf ≈ 2e-4 ~ 5e-4 m2.K/W  (冷却水典型值)
      - tau    ≈ 100 ~ 200 天          (数月至一年达到渐近值)
      - 5年运行后 Rf 在 1.5e-4 ~ 4e-4 量级
    """
    return {
        "A": 7e-5,          # 沉积系数 [m2.K/J]
        "Ea": 45000,        # 活化能 [J/mol] (CaCO3 典型 40-60 kJ/mol)
        "T_w": 363,         # 壁温 90°C [K]
        "B": 2e-8,          # 剥离系数 [1/Pa]
        "rho": 1000,        # 水密度 [kg/m3]
        "v": 1.5,           # 流速 [m/s]
        "f": 0.005,         # Darcy 摩擦因子
        "U_clean": 1200,    # 清洁传热系数 [W/(m2.K)]
        "mode": "kern-seaton",
        "label": "基准工况",
    }


def vary_param(base, param_name, values, labels=None):
    """对某个参数取不同值, 生成多组工况参数列表"""
    variants = []
    if labels is None:
        labels = [f"{param_name}={v}" for v in values]
    for v, lbl in zip(values, labels):
        p = dict(base)
        p[param_name] = v
        p["label"] = lbl
        variants.append(p)
    return variants


# ---------------------------------------------------------------------------
# 结果导出
# ---------------------------------------------------------------------------

def save_csv(results_list, filepath):
    """保存多个工况结果到 CSV (每个工况一行注释, 然后数据)"""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "day", "Rf", "U", "U_pct"])
        for results in results_list:
            for row in results:
                writer.writerow([row["label"], row["day"], row["Rf"], row["U"], row["U_pct"]])


# ---------------------------------------------------------------------------
# 简单文本报告
# ---------------------------------------------------------------------------

def print_summary(results):
    """打印一组结果的摘要"""
    if not results:
        return
    first = results[0]
    last = results[-1]
    rf_inf_est = last["Rf"]
    print(f"  工况: {first['label']}")
    print(f"    初始 U     = {first['U']:.1f} W/(m2.K)")
    print(f"    最终 U     = {last['U']:.1f} W/(m2.K) ({last['U_pct']:.1f}%)")
    print(f"    最终 Rf    = {last['Rf']:.6f} m2.K/W")
    # 估算 Rf_inf 和 tau (用最后数据的值)
    print(f"    Rf_inf 估算 ≈ {rf_inf_est:.6f} m2.K/W")


if __name__ == "__main__":
    # ---- 基准工况 ----
    base = default_params()
    print("=== 基准工况 ===")
    res_base = simulate(base)
    print_summary(res_base)

    # ---- 温度对比 ----
    print("\n=== 不同壁温 ===")
    temps = [333, 353, 363, 373, 393]  # K (60~120°C)
    labels = [f"Tw={t}K ({t-273:.0f}°C)" for t in temps]
    for p in vary_param(base, "T_w", temps, labels):
        print_summary(simulate(p))

    # ---- 流速对比 ----
    print("\n=== 不同流速 ===")
    vels = [0.5, 1.0, 1.5, 2.0, 3.0]
    for p in vary_param(base, "v", vels):
        print_summary(simulate(p))

    # ---- 保存 CSV ----
    all_results = [simulate(p) for p in vary_param(base, "T_w", temps, labels)]
    out_path = Path(__file__).parent / "output" / "simulation_results.csv"
    save_csv(all_results, out_path)
    print(f"\n结果已保存至: {out_path}")
