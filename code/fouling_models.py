"""
多机制结垢 ODE 注册表 — 统一接口

每种模型提供:
  - name, description, params (参数名+默认值+范围)
  - ode_residual(t, Rf, **params) → dRf/dt 残差
  - analytical(t, **params) → Rf(t) 解析解 (如果有)
  - applicable: 该模型适用于哪种结垢类型

用于 PINN 物理约束切换 + 真实曲线基线拟合。
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

R_GAS = 8.314  # J/(mol·K)


# =========================================================================
# 模型注册数据结构
# =========================================================================

@dataclass
class FoulingModel:
    name: str
    description: str
    params: Dict[str, tuple]  # name → (default, low, high, unit)
    applicable: List[str]     # 适用结垢类型标签
    has_analytical: bool = True

    # 函数引用 (注册时填充)
    ode_residual_fn: Optional[Callable] = None
    analytical_fn: Optional[Callable] = None


# =========================================================================
# 模型 1: Kern-Seaton (经典渐近型)
# =========================================================================

def ks_ode_residual(dRf_dt: np.ndarray, Rf: np.ndarray,
                    A: float, Ea: float, Tw: float,
                    B: float, v: float, rho: float = 1000.0, f: float = 0.005) -> np.ndarray:
    """Kern-Seaton ODE 残差: dRf/dt - (φd - φr) 应为 0"""
    tau_s = 0.5 * f * rho * v * v
    phi_d = A * np.exp(-Ea / (R_GAS * Tw))
    phi_r = B * tau_s * Rf
    return dRf_dt - (phi_d - phi_r)


def ks_analytical(t: np.ndarray, A: float, Ea: float, Tw: float,
                  B: float, v: float, rho: float = 1000.0, f: float = 0.005) -> np.ndarray:
    """Kern-Seaton 解析解: Rf(t) = Rf* × (1 - exp(-t/τ))"""
    tau_s = 0.5 * f * rho * v * v
    phi_d = A * np.exp(-Ea / (R_GAS * Tw))
    rf_inf = phi_d / (B * tau_s)
    tau_c = 1.0 / (B * tau_s)
    return rf_inf * (1.0 - np.exp(-t / tau_c))


# =========================================================================
# 模型 2: Kern-Seaton + 初始残余 (清洗不彻底)
# =========================================================================

def ks_offset_ode_residual(dRf_dt: np.ndarray, Rf: np.ndarray,
                           A: float, Ea: float, Tw: float,
                           B: float, v: float, Rf0: float = 0.0,
                           rho: float = 1000.0, f: float = 0.005) -> np.ndarray:
    """带初始残余的 KS ODE: 物理项不变，但初始条件 Rf(0)=Rf0"""
    tau_s = 0.5 * f * rho * v * v
    phi_d = A * np.exp(-Ea / (R_GAS * Tw))
    phi_r = B * tau_s * Rf
    return dRf_dt - (phi_d - phi_r)


def ks_offset_analytical(t: np.ndarray, A: float, Ea: float, Tw: float,
                         B: float, v: float, Rf0: float = 0.0,
                         rho: float = 1000.0, f: float = 0.005) -> np.ndarray:
    """Rf(t) = Rf0 + Rf* × (1 - exp(-t/τ)) — 从残余热阻开始增长"""
    tau_s = 0.5 * f * rho * v * v
    phi_d = A * np.exp(-Ea / (R_GAS * Tw))
    rf_inf = phi_d / (B * tau_s)
    tau_c = 1.0 / (B * tau_s)
    return Rf0 + rf_inf * (1.0 - np.exp(-t / tau_c))


# =========================================================================
# 模型 3: 简化 KS 拟合形式 (用于直接从 Rf(t) 曲线提取参数)
# =========================================================================

def ks_simple_analytical(t: np.ndarray, Rf_star: float, tau: float,
                         Rf0: float = 0.0) -> np.ndarray:
    """简化形式: Rf(t) = Rf0 + Rf* × (1 - exp(-t/τ))
    不依赖底层物理参数 (A, Ea, B, v), 直接拟合"""
    return Rf0 + Rf_star * (1.0 - np.exp(-t / tau))


def ks_simple_ode_residual(dRf_dt: np.ndarray, Rf: np.ndarray,
                           Rf_star: float, tau: float) -> np.ndarray:
    """简化 KS ODE: dRf/dt = (Rf* - Rf) / τ"""
    return dRf_dt - (Rf_star - Rf) / tau


# =========================================================================
# 模型 4: 诱导期 + 线性增长 (CaCO3 结晶结垢特征)
# =========================================================================

def induction_linear_analytical(t: np.ndarray, t_ind: float, k_growth: float,
                                Rf_max: float = 1e-3) -> np.ndarray:
    """
    诱导期后线性增长, 有上限:
      t < t_ind: Rf ≈ 0 (或轻微负值, 粗糙度效应)
      t ≥ t_ind: Rf = k_growth × (t - t_ind), 上限 Rf_max
    """
    Rf = np.where(t < t_ind, 0.0, k_growth * (t - t_ind))
    return np.clip(Rf, 0, Rf_max)


def induction_linear_ode_residual(dRf_dt: np.ndarray, Rf: np.ndarray,
                                  t: np.ndarray, t_ind: float, k_growth: float) -> np.ndarray:
    """诱导期 ODE: dRf/dt = 0 (t<t_ind) 或 k_growth (t≥t_ind)"""
    expected = np.where(t < t_ind, 0.0, k_growth)
    return dRf_dt - expected


# =========================================================================
# 模型 5: Ebert-Panchal 阈值模型 (简化版)
# =========================================================================

def ebert_panchal_ode_residual(dRf_dt: np.ndarray, Rf: np.ndarray,
                               alpha: float, beta: float, gamma: float,
                               Re: float, Pr: float, Tw: float, tau_w: float) -> np.ndarray:
    """
    Ebert-Panchal (1997) 阈值结垢模型:
      dRf/dt = α·Re^β·Pr^(-0.66)·exp(-Ea/(R·Tfilm)) - γ·τw

    存在临界条件: 当沉积项 < 剥离项时, 不发生结垢
    """
    # 简化: 用 alpha 吸收 Re, Pr, exp 项
    deposition = alpha * np.exp(-beta / (R_GAS * Tw))
    removal = gamma * tau_w
    return dRf_dt - (deposition - removal)  # 当 deposition < removal 时 dRf/dt < 0 → 阈值效应


# =========================================================================
# 注册表
# =========================================================================

MODELS: Dict[str, FoulingModel] = {
    "kern_seaton": FoulingModel(
        name="kern_seaton",
        description="经典 Kern-Seaton 渐近模型: dRf/dt = A·exp(-Ea/RTw) - B·τs·Rf",
        params={
            "A": (5e-5, 3e-6, 1e-4, "m²·K/J"),
            "Ea": (45000, 35000, 60000, "J/mol"),
            "Tw": (360, 320, 400, "K"),
            "B": (5e-8, 3e-9, 1e-7, "1/Pa"),
            "v": (1.0, 0.3, 4.0, "m/s"),
        },
        applicable=["crystallization_fouling", "particulate_fouling",
                     "phosphoric_acid", "crude_oil_asymptotic"],
        ode_residual_fn=ks_ode_residual,
        analytical_fn=ks_analytical,
    ),

    "kern_seaton_offset": FoulingModel(
        name="kern_seaton_offset",
        description="KS + 初始残余: Rf(t)=Rf0+Rf*·(1-exp(-t/τ)), 适用于清洗不彻底",
        params={
            "A": (5e-5, 3e-6, 1e-4, "m²·K/J"),
            "Ea": (45000, 35000, 60000, "J/mol"),
            "Tw": (360, 320, 400, "K"),
            "B": (5e-8, 3e-9, 1e-7, "1/Pa"),
            "v": (1.0, 0.3, 4.0, "m/s"),
            "Rf0": (0.0, 0.0, 5e-4, "m²·K/W"),
        },
        applicable=["cleaning_residual", "phosphoric_acid", "crude_oil_asymptotic"],
        ode_residual_fn=ks_offset_ode_residual,
        analytical_fn=ks_offset_analytical,
    ),

    "ks_simple": FoulingModel(
        name="ks_simple",
        description="简化 KS 拟合形式: Rf(t)=Rf0+Rf*·(1-exp(-t/τ)), 直接拟合参数",
        params={
            "Rf_star": (1.8e-4, 5e-5, 5e-3, "m²·K/W"),
            "tau": (40.0, 5.0, 500.0, "hours"),
            "Rf0": (0.0, -1e-5, 5e-4, "m²·K/W"),
        },
        applicable=["all_asymptotic", "curve_fitting"],
        ode_residual_fn=ks_simple_ode_residual,
        analytical_fn=ks_simple_analytical,
    ),

    "induction_linear": FoulingModel(
        name="induction_linear",
        description="诱导期 + 线性增长: 初期 Rf≈0, 之后线性增长至 Rf_max",
        params={
            "t_ind": (10.0, 0.0, 100.0, "hours"),
            "k_growth": (1e-6, 1e-8, 1e-4, "m²·K/(W·h)"),
            "Rf_max": (1e-3, 1e-5, 1e-2, "m²·K/W"),
        },
        applicable=["CaCO3_crystallization", "induction_period", "surface_modified"],
        ode_residual_fn=induction_linear_ode_residual,
        analytical_fn=induction_linear_analytical,
    ),

    "ebert_panchal": FoulingModel(
        name="ebert_panchal",
        description="Ebert-Panchal 阈值模型: 存在临界条件, 低于阈值则结垢不发生",
        params={
            "alpha": (1e-4, 1e-6, 1e-2, "m²·K/(W·s)"),
            "beta": (45000, 35000, 60000, "J/mol"),
            "gamma": (1e-8, 1e-10, 1e-6, "m²·K/(W·Pa)"),
            "Re": (10000, 5000, 50000, ""),
            "Pr": (7.0, 1.0, 15.0, ""),
            "Tw": (360, 320, 400, "K"),
            "tau_w": (10.0, 0.1, 100.0, "Pa"),
        },
        applicable=["crude_oil_threshold", "chemical_reaction_fouling"],
        has_analytical=False,
        ode_residual_fn=ebert_panchal_ode_residual,
        analytical_fn=None,
    ),
}


def get_model(name: str) -> FoulingModel:
    """获取已注册的模型"""
    if name not in MODELS:
        raise KeyError(f"未知模型 '{name}'。可用: {list(MODELS.keys())}")
    return MODELS[name]


def list_models():
    """列出所有注册模型"""
    for key, m in MODELS.items():
        anal = "✓" if m.has_analytical else "✗"
        print(f"  {key:<25s} 解析解={anal}  适用: {', '.join(m.applicable[:3])}")
        print(f"    {m.description}")


def recommend_model(curve_type: str) -> List[str]:
    """根据曲线特征推荐合适的模型"""
    recommendations = []
    for key, m in MODELS.items():
        for tag in m.applicable:
            if tag.lower() in curve_type.lower():
                recommendations.append(key)
                break
    return recommendations if recommendations else ["ks_simple"]


# =========================================================================
# 拟合工具
# =========================================================================

def fit_model_to_curve(t: np.ndarray, Rf: np.ndarray,
                       model_name: str = "ks_simple") -> dict:
    """用指定模型拟合一条 Rf(t) 曲线, 返回最佳参数"""
    from scipy.optimize import curve_fit

    model = get_model(model_name)
    if not model.has_analytical or model.analytical_fn is None:
        return {"success": False, "error": "该模型无解析解, 无法直接拟合"}

    if model_name == "ks_simple":
        p0 = [np.max(Rf) * 0.9, (t[-1] - t[0]) / 3.0, max(Rf[0], 0)]
        bounds = ([0, 1.0, -1e-5], [np.max(Rf) * 3, 1000, np.max(Rf) * 0.5])
        try:
            popt, pcov = curve_fit(
                lambda t_, rf_star, tau, rf0: ks_simple_analytical(t_, rf_star, tau, rf0),
                t, Rf, p0=p0, bounds=bounds, maxfev=10000
            )
            Rf_pred = ks_simple_analytical(t, *popt)
            ss_res = np.sum((Rf - Rf_pred) ** 2)
            ss_tot = np.sum((Rf - np.mean(Rf)) ** 2)
            r2 = 1 - ss_res / ss_tot
            rmse = np.sqrt(np.mean((Rf - Rf_pred) ** 2))
            return {"success": True, "model": model_name,
                    "params": {"Rf_star": popt[0], "tau": popt[1], "Rf0": popt[2]},
                    "r2": r2, "rmse": rmse, "Rf_pred": Rf_pred}
        except Exception as e:
            return {"success": False, "error": str(e)}

    return {"success": False, "error": f"模型 {model_name} 的拟合尚未实现"}


# =========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("结垢 ODE 模型注册表")
    print("=" * 60)
    list_models()
    print(f"\n共注册 {len(MODELS)} 个模型")

    # 快速测试: 生成一条 KS 曲线然后拟合
    print(f"\n--- 快速测试: ks_simple 拟合 ---")
    t_test = np.linspace(0, 100, 50)
    Rf_true = ks_simple_analytical(t_test, Rf_star=1.8e-4, tau=40.0, Rf0=0.0)
    Rf_noisy = Rf_true + np.random.normal(0, 2e-6, len(t_test))

    result = fit_model_to_curve(t_test, Rf_noisy, "ks_simple")
    if result["success"]:
        print(f"  真值: Rf*=1.80e-4, τ=40.0h, Rf0=0")
        print(f"  拟合: Rf*={result['params']['Rf_star']:.2e}, "
              f"τ={result['params']['tau']:.1f}h, "
              f"Rf0={result['params']['Rf0']:.2e}")
        print(f"  R²={result['r2']:.4f}")
