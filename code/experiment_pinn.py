"""
PINN vs MLP 对比实验

三组实验 (复现 Phase 2 的场景，直接对比):
  1. 稀疏数据: N=20, 50, 100, 500 → PINN 应远优于 MLP
  2. 外推:     训练 t≤3年, 测试 t>3年
  3. 物理一致性: 检查单调性

同时需要设置 KMP_DUPLICATE_LIB_OK=TRUE 环境变量 (torch+conda冲突)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

import pinn_model
import fouling_simulator as fs

OUTPUT_DIR = Path(__file__).parent / "output"
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_COLS = ["Tw", "v", "A", "Ea", "B", "U_clean", "day"]
MAX_TRAIN_MLP = 50000

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"设备: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'}")


# ===================================================================
# 数据准备
# ===================================================================

def load_data(filepath, max_rows=None):
    df = pd.read_csv(filepath)
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=SEED)
    return df


# ===================================================================
# 训练 MLP baseline (复现 Phase 2)
# ===================================================================

def train_mlp_baseline(X_train, y_train, hidden_layers=(128, 64, 32)):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layers, activation="relu",
        solver="adam", max_iter=500, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=20,
        random_state=SEED,
    )
    t0 = time.time()
    mlp.fit(X_scaled, y_train)
    return mlp, scaler, time.time() - t0


def eval_mlp(mlp, scaler, X_test, y_test):
    y_pred = mlp.predict(scaler.transform(X_test))
    return {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "y_pred": y_pred,
    }


# ===================================================================
# 训练 PINN
# ===================================================================

def train_pinn_on_data(train_df, n_epochs=3000, lambda_phys=0.5):
    """在给定数据上训练 PINN"""
    x_data, y_data, _ = pinn_model.prepare_tensors(train_df, DEVICE)
    x_phys = pinn_model.generate_collocation_points(500, device=DEVICE)

    model = pinn_model.PINN(hidden_layers=(64, 64, 64, 64)).to(DEVICE)

    t0 = time.time()
    history = pinn_model.train_pinn(
        model, x_data, y_data, x_phys,
        lambda_phys=lambda_phys, lr=1e-3,
        n_epochs=n_epochs, verbose=False,
    )
    train_time = time.time() - t0
    return model, history, train_time


def eval_pinn(model, test_df):
    """在测试集上评估 PINN"""
    x_test, _, y_true = pinn_model.prepare_tensors(test_df, DEVICE)
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)
    y_pred_np = y_pred.cpu().numpy().flatten()
    y_true_np = y_true.cpu().numpy().flatten()
    return {
        "R2": r2_score(y_true_np, y_pred_np),
        "MAE": mean_absolute_error(y_true_np, y_pred_np),
        "y_pred": y_pred_np,
        "y_true": y_true_np,
    }


# ===================================================================
# 实验 1: 稀疏数据对比
# ===================================================================

def experiment_sparse_comparison():
    print("\n" + "=" * 60)
    print("实验 1: PINN vs MLP — 稀疏数据")
    print("=" * 60)

    test_df = load_data(OUTPUT_DIR / "test_full.csv", max_rows=MAX_TRAIN_MLP)
    X_test = test_df[FEATURE_COLS].values
    y_test_true = test_df["Rf_true"].values

    n_list = [20, 50, 100, 500]
    results = []

    for n in n_list:
        print(f"\n--- N={n} ---")
        train_small = load_data(OUTPUT_DIR / "train_full.csv").sample(n=n, random_state=SEED)

        # MLP
        X_train_mlp = train_small[FEATURE_COLS].values
        y_train_mlp = train_small["Rf_noisy"].values
        mlp, scaler_mlp, t_mlp = train_mlp_baseline(X_train_mlp, y_train_mlp, (64, 32))
        res_mlp = eval_mlp(mlp, scaler_mlp, X_test, y_test_true)
        print(f"  MLP:  R²={res_mlp['R2']:.4f}, MAE={res_mlp['MAE']:.2e}, 耗时={t_mlp:.1f}s")

        # PINN
        pinn, history, t_pinn = train_pinn_on_data(train_small, n_epochs=3000, lambda_phys=0.5)
        res_pinn = eval_pinn(pinn, test_df)
        print(f"  PINN: R²={res_pinn['R2']:.4f}, MAE={res_pinn['MAE']:.2e}, 耗时={t_pinn:.1f}s")

        results.append({
            "N": n, "mlp": res_mlp, "pinn": res_pinn,
            "t_mlp": t_mlp, "t_pinn": t_pinn,
        })

    return results


# ===================================================================
# 实验 2: 外推对比
# ===================================================================

def experiment_extrapolation_comparison():
    print("\n" + "=" * 60)
    print("实验 2: PINN vs MLP — 外推 (训练≤3年, 测试>3年)")
    print("=" * 60)

    train_df = load_data(OUTPUT_DIR / "train_extrap.csv")
    test_df = load_data(OUTPUT_DIR / "test_extrap.csv")

    # MLP
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["Rf_noisy"].values
    print("训练 MLP...")
    mlp, scaler_mlp, t_mlp = train_mlp_baseline(X_train, y_train, (128, 64, 32))
    res_mlp = eval_mlp(mlp, scaler_mlp, test_df[FEATURE_COLS].values, test_df["Rf_true"].values)
    print(f"  MLP:  R²={res_mlp['R2']:.4f}, MAE={res_mlp['MAE']:.2e}, 耗时={t_mlp:.1f}s")

    # 按外推距离分组
    days = test_df["day"].values
    for lo, hi in [(1095, 1460), (1460, 1643), (1643, 1825)]:
        mask = (days >= lo) & (days <= hi)
        if mask.sum() > 0:
            r2_bin = r2_score(test_df["Rf_true"].values[mask], res_mlp["y_pred"][mask])
            mae_bin = mean_absolute_error(test_df["Rf_true"].values[mask], res_mlp["y_pred"][mask])
            print(f"    Day {lo}-{hi}: MLP R²={r2_bin:.4f}, MAE={mae_bin:.2e}")

    # PINN (用子集加速)
    print("训练 PINN...")
    train_sub = train_df.sample(n=min(5000, len(train_df)), random_state=SEED)
    pinn, history, t_pinn = train_pinn_on_data(train_sub, n_epochs=4000, lambda_phys=1.0)
    res_pinn = eval_pinn(pinn, test_df)

    print(f"  PINN: R²={res_pinn['R2']:.4f}, MAE={res_pinn['MAE']:.2e}, 耗时={t_pinn:.1f}s")
    for lo, hi in [(1095, 1460), (1460, 1643), (1643, 1825)]:
        mask = (days >= lo) & (days <= hi)
        if mask.sum() > 0:
            r2_bin = r2_score(test_df["Rf_true"].values[mask], res_pinn["y_pred"][mask])
            mae_bin = mean_absolute_error(test_df["Rf_true"].values[mask], res_pinn["y_pred"][mask])
            print(f"    Day {lo}-{hi}: PINN R²={r2_bin:.4f}, MAE={mae_bin:.2e}")

    return {"mlp": res_mlp, "pinn": res_pinn, "t_mlp": t_mlp, "t_pinn": t_pinn}


# ===================================================================
# 实验 3: 物理一致性
# ===================================================================

def experiment_physical_consistency():
    print("\n" + "=" * 60)
    print("实验 3: 物理一致性 — PINN vs MLP 单调性")
    print("=" * 60)

    test_df = load_data(OUTPUT_DIR / "test_full.csv", max_rows=MAX_TRAIN_MLP)
    cid_counts = test_df["condition_id"].value_counts()
    good_cids = cid_counts[cid_counts >= 30].index[:3].tolist()

    # 训练 PINN 和 MLP (充足数据)
    train_df = load_data(OUTPUT_DIR / "train_full.csv", max_rows=5000)

    # MLP
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["Rf_noisy"].values
    mlp, scaler_mlp, _ = train_mlp_baseline(X_train, y_train, (64, 64, 32))

    # PINN
    print("训练 PINN...")
    pinn, _, _ = train_pinn_on_data(train_df, n_epochs=3000, lambda_phys=0.5)

    # 检查单调性
    results = []
    for cid in good_cids:
        mask = (test_df["condition_id"] == cid).values
        df_cid = test_df[mask].sort_values("day")

        X_cid = df_cid[FEATURE_COLS].values
        y_true_cid = df_cid["Rf_true"].values

        # MLP 预测
        y_mlp = mlp.predict(scaler_mlp.transform(X_cid))
        diffs_mlp = np.diff(y_mlp)
        n_viol_mlp = (diffs_mlp < 0).sum()

        # PINN 预测
        x_tensor, _, _ = pinn_model.prepare_tensors(df_cid, DEVICE)
        pinn.eval()
        with torch.no_grad():
            y_pinn = pinn(x_tensor).cpu().numpy().flatten()
        diffs_pinn = np.diff(y_pinn)
        n_viol_pinn = (diffs_pinn < 0).sum()

        results.append({
            "cid": int(cid), "n_total": len(diffs_mlp),
            "mlp_viol": n_viol_mlp, "pinn_viol": n_viol_pinn,
        })

    print("\n  工况     MLP违反    PINN违反")
    total_mlp, total_pinn, total_all = 0, 0, 0
    for r in results:
        print(f"  CID={r['cid']:<5} {r['mlp_viol']:>5}/{r['n_total']:<5}"
              f"   {r['pinn_viol']:>5}/{r['n_total']}")
        total_mlp += r["mlp_viol"]
        total_pinn += r["pinn_viol"]
        total_all += r["n_total"]

    print(f"\n  MLP: {total_mlp}/{total_all} 违反单调性 ({100*total_mlp/total_all:.1f}%)")
    print(f"  PINN: {total_pinn}/{total_all} 违反单调性 ({100*total_pinn/total_all:.1f}%)")

    return results


# ===================================================================
# 可视化
# ===================================================================

def plot_sparse_comparison(results):
    print("\n=== 绘制稀疏数据对比图 ===")
    Ns = [r["N"] for r in results]
    mlp_r2 = [r["mlp"]["R2"] for r in results]
    pinn_r2 = [r["pinn"]["R2"] for r in results]
    mlp_mae = [r["mlp"]["MAE"] for r in results]
    pinn_mae = [r["pinn"]["MAE"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.semilogx(Ns, mlp_r2, "r-o", linewidth=2, markersize=8, label="MLP")
    ax1.semilogx(Ns, pinn_r2, "b-s", linewidth=2, markersize=8, label="PINN")
    ax1.set_xlabel("训练样本数", fontsize=12)
    ax1.set_ylabel("R²", fontsize=12)
    ax1.set_title("PINN vs MLP: R² 对比 (稀疏数据)", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    ax2.loglog(Ns, mlp_mae, "r-o", linewidth=2, markersize=8, label="MLP")
    ax2.loglog(Ns, pinn_mae, "b-s", linewidth=2, markersize=8, label="PINN")
    ax2.set_xlabel("训练样本数", fontsize=12)
    ax2.set_ylabel("MAE (m²·K/W)", fontsize=12)
    ax2.set_title("PINN vs MLP: MAE 对比 (稀疏数据)", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pinn_vs_mlp_sparse.png", dpi=150)
    plt.close(fig)
    print(f"  已保存: pinn_vs_mlp_sparse.png")


# ===================================================================
if __name__ == "__main__":
    print("=== PINN vs MLP 对比实验 ===")
    print(f"设备: {DEVICE}")

    # 实验 1
    res1 = experiment_sparse_comparison()

    # 画图
    plot_sparse_comparison(res1)

    # 实验 2
    res2 = experiment_extrapolation_comparison()

    # 实验 3
    res3 = experiment_physical_consistency()

    # 总结
    print("\n" + "=" * 60)
    print("=== 实验总结 ===")
    print("=" * 60)
    print("\n稀疏数据:")
    for r in res1:
        print(f"  N={r['N']:>3}: MLP R²={r['mlp']['R2']:>8.2f}, PINN R²={r['pinn']['R2']:>8.2f}"
              f" | MLP MAE={r['mlp']['MAE']:.2e}, PINN MAE={r['pinn']['MAE']:.2e}")
    print(f"\n外推:")
    print(f"  MLP  R²={res2['mlp']['R2']:.4f}, PINN R²={res2['pinn']['R2']:.4f}")
    print(f"  MLP  MAE={res2['mlp']['MAE']:.2e}, PINN MAE={res2['pinn']['MAE']:.2e}")
