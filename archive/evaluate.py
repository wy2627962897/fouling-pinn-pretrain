"""
评估可视化: 对比 MLP baseline 在不同场景下的表现
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

OUTPUT_DIR = Path(__file__).parent / "output"
SEED = 42
np.random.seed(SEED)
FEATURE_COLS = ["Tw", "v", "A", "Ea", "B", "U_clean", "day"]
MAX_TRAIN = 50000

# 全局训练好的 MLP (充足数据)
_mlp_full = None
_scaler_full = None


def load_csv(filepath, max_rows=None):
    df = pd.read_csv(filepath)
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=SEED)
    return df


def prepare_xy(df):
    X = df[FEATURE_COLS].values
    y_true = df["Rf_true"].values
    return X, y_true


def get_or_train_mlp():
    """懒加载: 在充足数据上训练一个 MLP"""
    global _mlp_full, _scaler_full
    if _mlp_full is not None:
        return _mlp_full, _scaler_full

    train_df = load_csv(OUTPUT_DIR / "train_full.csv", max_rows=MAX_TRAIN)
    X_train, _ = prepare_xy(train_df)
    y_train = train_df["Rf_noisy"].values

    _scaler_full = StandardScaler()
    X_scaled = _scaler_full.fit_transform(X_train)
    _mlp_full = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32), activation="relu",
        solver="adam", max_iter=500, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=20,
        random_state=SEED,
    )
    _mlp_full.fit(X_scaled, y_train)
    return _mlp_full, _scaler_full


# ===================================================================
# 图 1: 预测值 vs 真实值散点图 (数据充足)
# ===================================================================

def plot_pred_vs_true():
    print("=== 图: 预测值 vs 真实值 ===")
    mlp, scaler = get_or_train_mlp()
    test_df = load_csv(OUTPUT_DIR / "test_full.csv", max_rows=MAX_TRAIN)
    X_test, y_true = prepare_xy(test_df)

    # 随机采样避免散点密集
    n_sample = min(5000, len(X_test))
    idx = np.random.choice(len(X_test), size=n_sample, replace=False)
    X_sample, y_sample = X_test[idx], y_true[idx]

    y_pred = mlp.predict(scaler.transform(X_sample))

    r2 = r2_score(y_sample, y_pred)
    mae = mean_absolute_error(y_sample, y_pred)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_sample * 1e3, y_pred * 1e3, alpha=0.3, s=5, edgecolors="none")
    ax.plot([0, max(y_sample * 1e3)], [0, max(y_sample * 1e3)], "r--", linewidth=1.5, label="完美预测")

    ax.set_xlabel("真实 Rf × 10³ (m²·K/W)", fontsize=12)
    ax.set_ylabel("预测 Rf × 10³ (m²·K/W)", fontsize=12)
    ax.set_title(f"MLP: 预测值 vs 真实值\nR² = {r2:.4f}, MAE = {mae:.2e}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "eval_pred_vs_true.png", dpi=150)
    plt.close(fig)
    print(f"  已保存")


# ===================================================================
# 图 2: R² & MAE vs 训练样本数 (稀疏数据对比)
# ===================================================================

def plot_sparse_comparison():
    print("=== 图: 稀疏数据影响 ===")
    test_df = load_csv(OUTPUT_DIR / "test_full.csv", max_rows=MAX_TRAIN)
    X_test, y_true = prepare_xy(test_df)
    full_train = load_csv(OUTPUT_DIR / "train_full.csv", max_rows=MAX_TRAIN)

    n_list = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    r2_scores, mae_scores = [], []

    for n in n_list:
        train_small = full_train.sample(n=n, random_state=SEED)
        X_train, _ = prepare_xy(train_small)
        y_train = train_small["Rf_noisy"].values

        scaler = StandardScaler()
        mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32), activation="relu",
            solver="adam", max_iter=300, early_stopping=True,
            random_state=SEED,
        )
        mlp.fit(scaler.fit_transform(X_train), y_train)
        y_pred = mlp.predict(scaler.transform(X_test))

        r2_scores.append(r2_score(y_true, y_pred))
        mae_scores.append(mean_absolute_error(y_true, y_pred))
        print(f"  N={n:>5}: R²={r2_scores[-1]:.4f}, MAE={mae_scores[-1]:.2e}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.semilogx(n_list, r2_scores, "b-o", linewidth=2, markersize=8)
    ax1.set_xlabel("训练样本数", fontsize=12)
    ax1.set_ylabel("R²", fontsize=12)
    ax1.set_title("R² vs 训练样本数", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5, label="R²=0.9")
    ax1.legend()

    ax2.loglog(n_list, mae_scores, "r-o", linewidth=2, markersize=8)
    ax2.set_xlabel("训练样本数", fontsize=12)
    ax2.set_ylabel("MAE (m²·K/W)", fontsize=12)
    ax2.set_title("MAE vs 训练样本数", fontsize=14)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "eval_sparse_impact.png", dpi=150)
    plt.close(fig)
    print(f"  已保存")

    return {"n_list": n_list, "r2": r2_scores, "mae": mae_scores}


# ===================================================================
# 图 3: 外推误差随外推距离的变化
# ===================================================================

def plot_extrapolation_error():
    print("=== 图: 外推误差 ===")
    mlp, scaler = get_or_train_mlp()

    # 在完整测试集上预测
    test_data = load_csv(OUTPUT_DIR / "test_full.csv", max_rows=MAX_TRAIN)
    X_test, y_true = prepare_xy(test_data)
    y_pred = mlp.predict(scaler.transform(X_test))

    # 计算每个时间点的平均误差
    days = X_test[:, -1]
    unique_days = np.unique(days)
    unique_days.sort()

    mae_by_day = []
    for d in unique_days:
        mask = days == d
        if mask.sum() > 0:
            err = np.abs(y_pred[mask] - y_true[mask]).mean()
            mae_by_day.append((d, err))

    d_vals = [d for d, _ in mae_by_day]
    e_vals = [e for _, e in mae_by_day]

    # 用颜色区分训练域和外推域
    boundary = 1095  # 3 年分界线

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(d_vals, [e * 1e3 for e in e_vals], "b-", linewidth=2)
    ax.axvline(x=boundary, color="red", linestyle="--", linewidth=2,
               label=f"训练边界 (day={boundary})")
    ax.fill_betweenx(
        [0, max([e * 1e3 for e in e_vals]) * 1.1],
        boundary, max(d_vals),
        alpha=0.1, color="red", label="外推域"
    )
    ax.fill_betweenx(
        [0, max([e * 1e3 for e in e_vals]) * 1.1],
        0, boundary,
        alpha=0.05, color="green", label="训练域"
    )

    ax.set_xlabel("运行时间 (天)", fontsize=12)
    ax.set_ylabel("MAE × 10³ (m²·K/W)", fontsize=12)
    ax.set_title("预测误差随外推距离的变化", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "eval_extrapolation_error.png", dpi=150)
    plt.close(fig)
    print(f"  已保存")


# ===================================================================
# 图 4: 单个工况的预测轨迹 vs 真实轨迹
# ===================================================================

def plot_trajectory_comparison():
    print("=== 图: 预测轨迹对比 ===")
    mlp, scaler = get_or_train_mlp()

    test_df = load_csv(OUTPUT_DIR / "test_full.csv", max_rows=MAX_TRAIN)
    X_test, y_true = prepare_xy(test_df)

    cid_counts = test_df["condition_id"].value_counts()
    good_cids = cid_counts[cid_counts >= 60].index.tolist()
    selected = list(np.random.choice(good_cids, size=min(3, len(good_cids)), replace=False))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, cid in zip(axes, selected):
        mask = (test_df["condition_id"] == cid).values
        X_cid = X_test[mask]
        y_cid = y_true[mask]

        sort_idx = np.argsort(X_cid[:, -1])
        days = X_cid[sort_idx, -1]
        y_pred = mlp.predict(scaler.transform(X_cid[sort_idx]))

        ax.plot(days, y_cid[sort_idx] * 1e3, "b-", linewidth=2, label="真实值")
        ax.plot(days, y_pred * 1e3, "r--", linewidth=2, label="MLP 预测")
        ax.set_xlabel("运行时间 (天)", fontsize=11)
        ax.set_ylabel("Rf × 10³ (m²·K/W)", fontsize=11)
        ax.set_title(f"工况 CID={int(cid)}", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle("MLP 预测 vs 真实结垢轨迹 (充足数据训练)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "eval_trajectory_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  已保存")


# ===================================================================
# 图 5: 误差分布直方图
# ===================================================================

def plot_error_distribution():
    print("=== 图: 误差分布 ===")
    mlp, scaler = get_or_train_mlp()

    test_data = load_csv(OUTPUT_DIR / "test_full.csv", max_rows=MAX_TRAIN)
    X_test, y_true = prepare_xy(test_data)
    y_pred = mlp.predict(scaler.transform(X_test))
    errors = (y_pred - y_true) * 1e3  # 转换为 ×10³ 单位

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(errors, bins=80, color="steelblue", edgecolor="white", alpha=0.8, density=True)
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="零误差")

    # 标注 ±2σ
    std = np.std(errors)
    ax.axvline(x=-2 * std, color="orange", linestyle=":", linewidth=1, label=f"±2σ = ±{2*std:.2f}")
    ax.axvline(x=2 * std, color="orange", linestyle=":", linewidth=1)

    ax.set_xlabel("预测误差 × 10³ (m²·K/W)", fontsize=12)
    ax.set_ylabel("概率密度", fontsize=12)
    ax.set_title(f"MLP 预测误差分布\nMAE = {np.abs(errors).mean():.3f}, σ = {std:.3f}",
                 fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "eval_error_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  已保存")


# ===================================================================
if __name__ == "__main__":
    print("=== Baseline 评估可视化 ===\n")
    plot_pred_vs_true()
    plot_sparse_comparison()
    plot_extrapolation_error()
    plot_trajectory_comparison()
    plot_error_distribution()
    print("\n=== 全部评估图表已生成 ===")
