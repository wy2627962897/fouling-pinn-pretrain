"""
真实数据 PINN 实验 V2 — 跨曲线联合训练

关键改进: 不是逐条曲线独立拟合, 而是把所有 7 条曲线放在一起
  - 输入特征: [time, Rf_star, tau]  (每条曲线的 KS 参数作为物理指纹)
  - MLP: 纯数据驱动, 学习 time+指纹 → Rf 的映射
  - PINN: 加入 KS 物理约束 dRf/dt = (Rf* - Rf)/τ

这与原始合成实验的设计一致 (合成实验用 [Tw, v, A, B, Ea, day] 作为特征),
但现在用真实数据 + 物理指纹代替合成参数。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

CLEAN_DIR = Path(__file__).parent.parent / "data" / "real" / "curves" / "cleaned"
FITS_DIR = Path(__file__).parent.parent / "data" / "real" / "fits"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "real" / "pinn_results"
PREVIEW_DIR = Path(__file__).parent.parent / "data" / "real" / "curves" / "preview"

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_COLS = ["time_norm", "rf_star_norm", "tau_norm"]  # 3D 输入

torch.manual_seed(SEED)
np.random.seed(SEED)


# =========================================================================
# 数据准备
# =========================================================================

def build_multi_curve_dataset(n_train_per_curve: int):
    """
    构建跨曲线数据集
    训练集: 每条曲线随机采样 n_train_per_curve 个点
    测试集: 每条曲线的所有点
    特征: [time_norm, rf_star_norm, tau_norm]
    """
    with open(FITS_DIR / "ks_fits_fig5.json") as fp:
        ks_params = json.load(fp)
    ks_lookup = {f["curve"]: f for f in ks_params}

    curve_files = sorted(CLEAN_DIR.glob("source_001_fig5_rf_test*.csv"))

    # 计算全局归一化参数
    all_times, all_rf_stars, all_taus = [], [], []
    for csv_path in curve_files:
        curve_id = csv_path.stem.replace("source_001_fig5_", "")
        ks = ks_lookup.get(curve_id)
        if ks is None:
            continue
        df = pd.read_csv(csv_path)
        all_times.append(df["time"].values)
        all_rf_stars.append(ks["rf_star"])
        all_taus.append(ks["tau"])

    t_global = np.concatenate(all_times)
    t_mean, t_std = np.mean(t_global), np.std(t_global)
    rs_mean, rs_std = np.mean(all_rf_stars), np.std(all_rf_stars)
    tau_mean, tau_std = np.mean(all_taus), np.std(all_taus)

    train_data, test_data = [], []

    for csv_path in curve_files:
        curve_id = csv_path.stem.replace("source_001_fig5_", "")
        ks = ks_lookup.get(curve_id)
        if ks is None:
            continue

        df = pd.read_csv(csv_path)
        t = df["time"].values.astype(np.float64)
        rf = df["value"].values.astype(np.float64)

        # 特征
        t_norm = (t - t_mean) / t_std
        rs_norm = np.full_like(t, (ks["rf_star"] - rs_mean) / rs_std)
        tau_n = np.full_like(t, (ks["tau"] - tau_mean) / tau_std)
        X = np.column_stack([t_norm, rs_norm, tau_n])

        # 训练/测试拆分
        if n_train_per_curve >= len(t):
            train_idx = np.arange(len(t))
        else:
            train_idx = np.sort(np.random.choice(len(t), size=n_train_per_curve, replace=False))

        train_data.append((X[train_idx], rf[train_idx], curve_id,
                           ks["rf_star"], ks["tau"]))
        test_data.append((X, rf, curve_id, ks["rf_star"], ks["tau"], t))

    # 合并训练集
    X_train = np.concatenate([d[0] for d in train_data])
    y_train = np.concatenate([d[1] for d in train_data])

    return train_data, test_data, X_train, y_train, {
        "t_mean": t_mean, "t_std": t_std,
        "rs_mean": rs_mean, "rs_std": rs_std,
        "tau_mean": tau_mean, "tau_std": tau_std,
    }


# =========================================================================
# MLP Baseline (scikit-learn)
# =========================================================================

def train_sklearn_mlp(X_train, y_train, hidden=(128, 64, 32)):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden, activation="relu",
        solver="adam", max_iter=1000, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=30,
        random_state=SEED,
    )
    mlp.fit(X_scaled, y_train)
    return mlp, scaler


# =========================================================================
# PINN (PyTorch)
# =========================================================================

class PINNMulti(nn.Module):
    """3D 输入: [time_norm, rf_star_norm, tau_norm] → Rf"""
    def __init__(self, hidden=(64, 64, 64, 64)):
        super().__init__()
        layers = []
        in_dim = 3
        for h in hidden:
            layers.extend([nn.Linear(in_dim, h), nn.Tanh()])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_pinn_multi(X_train, y_train, norm_params,
                     n_epochs=3000, lr=1e-3, lambda_phys=0.3, verbose=False):
    """
    训练 PINN (跨曲线) — 优化版

    优化点:
      1. 物理配点 tensor 预先创建并设置 requires_grad, 避免每 epoch clone
      2. 合并 data loss 和 phys loss 的 forward pass
      3. 减少配点数 (200 vs 500), 减少 epoch (3000 vs 5000)
    """
    x_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(DEVICE)

    # 物理配点: 预先创建, 时间列需要 requires_grad
    n_phys = 200
    t_std = norm_params["t_std"]
    t_phys = torch.rand(n_phys, 1, device=DEVICE) * \
        ((102 - norm_params["t_mean"]) / t_std - (0 - norm_params["t_mean"]) / t_std) + \
        (0 - norm_params["t_mean"]) / t_std
    t_phys.requires_grad_(True)  # ← 只设一次!
    rs_phys = torch.rand(n_phys, 1, device=DEVICE) * 4 - 2
    tau_phys = torch.rand(n_phys, 1, device=DEVICE) * 4 - 2

    # 预计算反归一化用的值
    rs_mean = norm_params["rs_mean"]
    rs_std = norm_params["rs_std"]
    tau_mean = norm_params["tau_mean"]
    tau_std = norm_params["tau_std"]
    Rf_star_phys = rs_phys * rs_std + rs_mean
    inv_tau_phys = 1.0 / (tau_phys * tau_std + tau_mean + 1e-8)
    inv_t_std = 1.0 / (t_std + 1e-8)

    model = PINNMulti().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=300, min_lr=1e-6
    )

    history = {"data": [], "phys": [], "total": []}

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        # --- 数据损失 ---
        pred_data = model(x_t)
        loss_data = torch.mean((pred_data - y_t) ** 2)

        # --- 物理损失 (内联, 避免函数调用 + tensor 重建开销) ---
        x_phys_input = torch.cat([t_phys, rs_phys, tau_phys], dim=1)
        Rf_phys = model(x_phys_input)
        dRf_dt = torch.autograd.grad(Rf_phys, t_phys,
                                     grad_outputs=torch.ones_like(Rf_phys),
                                     create_graph=True)[0] * inv_t_std
        # KS ODE 残差
        residual = dRf_dt - (Rf_star_phys - Rf_phys) * inv_tau_phys
        loss_phys = torch.mean(residual ** 2)

        loss_total = loss_data + lambda_phys * loss_phys
        loss_total.backward()
        optimizer.step()
        scheduler.step(loss_total)

        history["data"].append(loss_data.item())
        history["phys"].append(loss_phys.item())
        history["total"].append(loss_total.item())

        if verbose and (epoch + 1) % 1000 == 0:
            print(f"    PINN epoch {epoch+1}: data_loss={loss_data.item():.4e}, "
                  f"phys_loss={loss_phys.item():.4e}, total={loss_total.item():.4e}")

    model.eval()
    return model, history


# =========================================================================
# 评估
# =========================================================================

def evaluate_model(model, scaler, test_data, is_pinn=False, norm_params=None):
    """在每条测试曲线上评估, 返回逐曲线指标"""
    results = []
    for X, y_true, curve_id, rf_star, tau, t_raw in test_data:
        if is_pinn:
            x_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                y_pred = model(x_t).cpu().numpy().flatten()
        else:
            y_pred = model.predict(scaler.transform(X))

        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        # 单调性
        diffs = np.diff(y_pred)
        n_viol = int((diffs < 0).sum())
        viol_rate = n_viol / len(diffs) if len(diffs) > 0 else 0.0

        results.append({
            "curve": curve_id, "r2": r2, "mae": mae,
            "n_violations": n_viol, "violation_rate": viol_rate,
            "y_pred": y_pred.tolist(), "y_true": y_true.tolist(),
            "t_raw": t_raw.tolist(),
        })
    return results


# =========================================================================
# 主实验
# =========================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    N_VALUES = [5, 10, 20, 50, 100]
    all_experiments = []

    print("=" * 70)
    print("真实数据 PINN vs MLP 稀疏实验 V2 (跨曲线联合训练)")
    print(f"设备: {DEVICE}")
    print("=" * 70)

    for N in N_VALUES:
        print(f"\n{'─' * 60}")
        print(f"  N = {N} (每条曲线 {N} 个训练点, 7条曲线共 {N*7} 点)")
        print(f"{'─' * 60}")

        train_data, test_data, X_train, y_train, norm_params = \
            build_multi_curve_dataset(n_train_per_curve=N)

        # --- MLP ---
        t0 = time.time()
        mlp, scaler = train_sklearn_mlp(X_train, y_train)
        t_mlp = time.time() - t0
        res_mlp = evaluate_model(mlp, scaler, test_data, is_pinn=False)

        mlp_r2_mean = np.mean([r["r2"] for r in res_mlp])
        mlp_mae_mean = np.mean([r["mae"] for r in res_mlp])
        mlp_viol_mean = np.mean([r["violation_rate"] for r in res_mlp])
        print(f"  MLP:  R²={mlp_r2_mean:.4f}  MAE={mlp_mae_mean:.2e}  "
              f"单调违反={mlp_viol_mean:.0%}  耗时={t_mlp:.1f}s")

        # --- PINN ---
        t0 = time.time()
        pinn, history = train_pinn_multi(X_train, y_train, norm_params,
                                         n_epochs=3000, lambda_phys=0.3)
        t_pinn = time.time() - t0
        res_pinn = evaluate_model(pinn, None, test_data, is_pinn=True, norm_params=norm_params)

        pinn_r2_mean = np.mean([r["r2"] for r in res_pinn])
        pinn_mae_mean = np.mean([r["mae"] for r in res_pinn])
        pinn_viol_mean = np.mean([r["violation_rate"] for r in res_pinn])
        advantage = mlp_mae_mean / (pinn_mae_mean + 1e-20)

        print(f"  PINN: R²={pinn_r2_mean:.4f}  MAE={pinn_mae_mean:.2e}  "
              f"单调违反={pinn_viol_mean:.0%}  耗时={t_pinn:.1f}s")
        print(f"  → PINN MAE 优势: {advantage:.1f}×")

        all_experiments.append({
            "N": N,
            "mlp": {"r2_mean": mlp_r2_mean, "mae_mean": mlp_mae_mean, "viol_mean": mlp_viol_mean,
                    "per_curve": [{k: v for k, v in r.items() if k not in ("y_pred", "y_true", "t_raw")}
                                  for r in res_mlp]},
            "pinn": {"r2_mean": pinn_r2_mean, "mae_mean": pinn_mae_mean, "viol_mean": pinn_viol_mean,
                     "per_curve": [{k: v for k, v in r.items() if k not in ("y_pred", "y_true", "t_raw")}
                                   for r in res_pinn]},
            "advantage": advantage,
            "t_mlp": t_mlp, "t_pinn": t_pinn,
        })

    # ====== 汇总 ======
    print(f"\n{'=' * 70}")
    print("汇总: PINN vs MLP 稀疏数据优势 (真实数据, 跨曲线训练)")
    print("=" * 70)
    print(f"{'N':>6s}  {'MLP R²':>10s}  {'MLP MAE':>10s}  {'MLP违反':>8s}  "
          f"{'PINN R²':>10s}  {'PINN MAE':>10s}  {'PINN违反':>8s}  {'优势':>6s}")
    print("-" * 75)
    for exp in all_experiments:
        n = exp["N"]
        print(f"{n:>6d}  {exp['mlp']['r2_mean']:>10.4f}  {exp['mlp']['mae_mean']:>10.2e}  "
              f"{exp['mlp']['viol_mean']:>7.0%}  "
              f"{exp['pinn']['r2_mean']:>10.4f}  {exp['pinn']['mae_mean']:>10.2e}  "
              f"{exp['pinn']['viol_mean']:>7.0%}  {exp['advantage']:>5.1f}×")

    # ====== 保存 ======
    with open(OUTPUT_DIR / "real_pinn_v2_results.json", "w") as fp:
        json.dump(all_experiments, fp, indent=2, ensure_ascii=False)
    print(f"\n结果: {OUTPUT_DIR / 'real_pinn_v2_results.json'}")

    # ====== 可视化 ======
    print(f"\n生成可视化...")
    visualize_v2(all_experiments)
    print(f"完成。")


def visualize_v2(all_experiments):
    Ns = [e["N"] for e in all_experiments]

    # 图1: MAE 稀疏优势
    fig, ax = plt.subplots(figsize=(8, 5))
    mlp_mae = [e["mlp"]["mae_mean"] for e in all_experiments]
    pinn_mae = [e["pinn"]["mae_mean"] for e in all_experiments]
    ax.loglog(Ns, mlp_mae, "r-o", linewidth=2, markersize=8, label="MLP (纯数据驱动)")
    ax.loglog(Ns, pinn_mae, "b-s", linewidth=2, markersize=8, label="PINN (KS 物理约束)")
    ax.set_xlabel("每条曲线训练样本数 N", fontsize=12)
    ax.set_ylabel("MAE (m²·K/W)", fontsize=12)
    ax.set_title("真实数据 (Source 001, 7条曲线): PINN vs MLP 稀疏优势", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PREVIEW_DIR / "real_v2_sparse_advantage.png", dpi=150)
    plt.close(fig)

    # 图2: R² 对比
    fig, ax = plt.subplots(figsize=(8, 5))
    mlp_r2 = [e["mlp"]["r2_mean"] for e in all_experiments]
    pinn_r2 = [e["pinn"]["r2_mean"] for e in all_experiments]
    ax.semilogx(Ns, mlp_r2, "r-o", linewidth=2, markersize=8, label="MLP")
    ax.semilogx(Ns, pinn_r2, "b-s", linewidth=2, markersize=8, label="PINN")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=0.9, color="green", linestyle=":", alpha=0.3, label="R²=0.9 (良好)")
    ax.set_xlabel("每条曲线训练样本数 N", fontsize=12)
    ax.set_ylabel("R²", fontsize=12)
    ax.set_title("真实数据: PINN vs MLP — R² 对比", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PREVIEW_DIR / "real_v2_r2_comparison.png", dpi=150)
    plt.close(fig)

    # 图3: 单调性
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(Ns))
    w = 0.35
    mlp_v = [e["mlp"]["viol_mean"] * 100 for e in all_experiments]
    pinn_v = [e["pinn"]["viol_mean"] * 100 for e in all_experiments]
    ax.bar(x - w/2, mlp_v, w, color="red", alpha=0.6, label="MLP")
    ax.bar(x + w/2, pinn_v, w, color="blue", alpha=0.6, label="PINN")
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}" for n in Ns])
    ax.set_ylabel("单调性违反率 (%)", fontsize=12)
    ax.set_title("物理一致性: 真实数据单调性对比", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(PREVIEW_DIR / "real_v2_monotonicity.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
