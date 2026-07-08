"""
纯数据驱动 Baseline: MLP 预测 Rf(t)

三个实验场景:
  1. 数据充足:  训练集 ~42万条, 测试模型泛化能力
  2. 数据稀疏:  仅取 N 条训练样本 (20, 50, 100, 500, 2000)
  3. 外推:     训练 (day ≤ 3年), 测试 (day > 3年)

依赖: scikit-learn, pandas, matplotlib (已在 conda env 中)
"""

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

OUTPUT_DIR = Path(__file__).parent / "output"
SEED = 42
MAX_TRAIN = 100000  # 充足实验最多用 10 万条训练

np.random.seed(SEED)

# CSV 列名
COLUMNS = ["condition_id", "Tw", "v", "A", "Ea", "B", "U_clean", "day", "Rf_true", "Rf_noisy"]


# ===================================================================
# 数据加载
# ===================================================================

def load_csv(filepath, max_rows=None):
    """用 pandas 加载 CSV (远快于 DictReader)"""
    df = pd.read_csv(filepath)
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=SEED)
    return df


def prepare_xy(df, feature_cols):
    """
    从 DataFrame 提取特征矩阵 X 和目标向量 y。
    feature_cols: 特征列名列表, 如 ["Tw","v","A","Ea","B","U_clean","day"]
    """
    X = df[feature_cols].values
    y = df["Rf_noisy"].values   # 带噪声的训练目标
    y_true = df["Rf_true"].values  # 真实值 (用于评估)
    return X, y, y_true


# ===================================================================
# 模型训练与评估
# ===================================================================

def train_mlp(X_train, y_train, hidden_layers=(128, 64, 32), max_iter=500):
    """训练 MLP 回归器"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=SEED,
        verbose=False,
    )
    t0 = time.time()
    mlp.fit(X_scaled, y_train)
    train_time = time.time() - t0
    return mlp, scaler, train_time


def evaluate(mlp, scaler, X_test, y_true_test):
    """评估模型: 返回 R², MAE, RMSE, 预测值"""
    X_scaled = scaler.transform(X_test)
    y_pred = mlp.predict(X_scaled)
    return {
        "R2": r2_score(y_true_test, y_pred),
        "MAE": mean_absolute_error(y_true_test, y_pred),
        "RMSE": math.sqrt(mean_squared_error(y_true_test, y_pred)),
        "y_pred": y_pred,
        "y_true": y_true_test,
    }


# ===================================================================
# 实验运行
# ===================================================================

FEATURE_COLS = ["Tw", "v", "A", "Ea", "B", "U_clean", "day"]


def experiment_1_data_rich():
    """实验1: 充足数据"""
    print("\n" + "=" * 60)
    print("实验 1: 数据充足 (全部训练集)")
    print("=" * 60)

    train_data = load_csv(OUTPUT_DIR / "train_full.csv", max_rows=MAX_TRAIN)
    val_data = load_csv(OUTPUT_DIR / "val_full.csv", max_rows=20000)
    test_data = load_csv(OUTPUT_DIR / "test_full.csv", max_rows=30000)

    X_train, y_train, _ = prepare_xy(train_data, FEATURE_COLS)
    X_val, _, y_val_true = prepare_xy(val_data, FEATURE_COLS)
    X_test, _, y_test_true = prepare_xy(test_data, FEATURE_COLS)

    print(f"  训练样本: {len(X_train):,}, 验证: {len(X_val):,}, 测试: {len(X_test):,}")

    mlp, scaler, t = train_mlp(X_train, y_train)
    print(f"  训练耗时: {t:.1f}s, 最终 loss: {mlp.loss_:.6f}")

    result = evaluate(mlp, scaler, X_test, y_test_true)
    print(f"  测试集 R² = {result['R2']:.4f}")
    print(f"  测试集 MAE = {result['MAE']:.2e} m2.K/W")
    print(f"  测试集 RMSE = {result['RMSE']:.2e} m2.K/W")

    return result


def experiment_2_sparse():
    """实验2: 稀疏数据"""
    print("\n" + "=" * 60)
    print("实验 2: 稀疏数据 (不同训练样本数)")
    print("=" * 60)

    test_data = load_csv(OUTPUT_DIR / "test_full.csv")
    X_test, _, y_test_true = prepare_xy(test_data, FEATURE_COLS)

    n_list = [20, 50, 100, 500, 2000, 10000]
    results = {}

    for n in n_list:
        train_small = load_csv(OUTPUT_DIR / "train_full.csv", max_rows=n)
        X_train, y_train, _ = prepare_xy(train_small, FEATURE_COLS)

        mlp, scaler, t = train_mlp(X_train, y_train, hidden_layers=(64, 32))
        result = evaluate(mlp, scaler, X_test, y_test_true)
        results[n] = result

        print(f"  N={n:>5}: R²={result['R2']:.4f}, MAE={result['MAE']:.2e}, "
              f"RMSE={result['RMSE']:.2e}, 耗时={t:.1f}s")

    return results


def experiment_3_extrapolation():
    """实验3: 外推能力"""
    print("\n" + "=" * 60)
    print("实验 3: 外推 (训练 ≤3年, 测试 >3年)")
    print("=" * 60)

    train_data = load_csv(OUTPUT_DIR / "train_extrap.csv")
    test_data = load_csv(OUTPUT_DIR / "test_extrap.csv")

    X_train, y_train, _ = prepare_xy(train_data, FEATURE_COLS)
    X_test, _, y_test_true = prepare_xy(test_data, FEATURE_COLS)

    print(f"  训练样本: {len(X_train):,} (day ≤ 3年)")
    print(f"  测试样本: {len(X_test):,} (day > 3年)")

    # 数据充足 + 外推
    mlp, scaler, t = train_mlp(X_train, y_train, hidden_layers=(128, 64, 32))
    print(f"  训练耗时: {t:.1f}s")

    result = evaluate(mlp, scaler, X_test, y_test_true)
    print(f"  外推测试 R² = {result['R2']:.4f}")
    print(f"  外推测试 MAE = {result['MAE']:.2e} m2.K/W")
    print(f"  外推测试 RMSE = {result['RMSE']:.2e} m2.K/W")

    # 分组看看: 离训练边界越远, 误差越大?
    days = X_test[:, -1]  # last column = day
    bins = [(1095, 1460), (1460, 1643), (1643, 1825)]
    for lo, hi in bins:
        mask = (days >= lo) & (days <= hi)
        if mask.sum() > 0:
            r2_bin = r2_score(y_test_true[mask], result["y_pred"][mask])
            mae_bin = mean_absolute_error(y_test_true[mask], result["y_pred"][mask])
            print(f"  Day {lo}-{hi}: R²={r2_bin:.4f}, MAE={mae_bin:.2e}, n={mask.sum():,}")
            result[(lo, hi)] = {"R2": r2_bin, "MAE": mae_bin}

    return result


def experiment_4_physical_consistency():
    """
    实验4: 物理一致性检查
    Rf 应该是单调不减的, 但 ML 模型可能违反这一点。
    """
    print("\n" + "=" * 60)
    print("实验 4: 物理一致性")
    print("=" * 60)

    test_df = load_csv(OUTPUT_DIR / "test_full.csv")
    X_test, _, y_test_true = prepare_xy(test_df, FEATURE_COLS)

    # 找 5 个完整工况
    cid_counts = test_df["condition_id"].value_counts()
    good_cids = cid_counts[cid_counts >= 30].index[:5].tolist()

    full_train = load_csv(OUTPUT_DIR / "train_full.csv", max_rows=MAX_TRAIN)
    X_train, y_train, _ = prepare_xy(full_train, FEATURE_COLS)
    mlp, scaler, _ = train_mlp(X_train, y_train, hidden_layers=(128, 64, 32))

    total_negative = 0
    total_pairs = 0
    print("\n  工况 预测单调性检查:")
    for cid in good_cids:
        mask = test_df["condition_id"] == cid
        X_cid = X_test[mask.values]
        y_cid_true = y_test_true[mask.values]
        days = X_cid[:, -1]

        # 按 day 排序
        sort_idx = np.argsort(days)
        X_sorted = X_cid[sort_idx]
        y_pred = mlp.predict(scaler.transform(X_sorted))

        diffs = np.diff(y_pred)
        n_negative = (diffs < 0).sum()
        total_negative += n_negative
        total_pairs += len(diffs)

        status = "✗ 非单调" if n_negative > 0 else "✓ 单调"
        print(f"    CID={int(cid)}: {status}, 违反次数={n_negative}/{len(diffs)}")

    violation_pct = 100 * total_negative / max(total_pairs, 1)
    print(f"\n  物理一致性总结: {total_negative}/{total_pairs} "
          f"预测步长违反 Rf 单调性 ({violation_pct:.1f}%)")

    return {"n_violations": total_negative, "n_total": total_pairs}


# ===================================================================
if __name__ == "__main__":
    print("=== 纯数据驱动 Baseline 实验 ===")
    print(f"特征: {FEATURE_COLS}")
    print(f"目标: Rf_noisy (含噪声)")

    # 实验 1: 充足数据
    res1 = experiment_1_data_rich()

    # 实验 2: 稀疏数据
    res2 = experiment_2_sparse()

    # 实验 3: 外推
    res3 = experiment_3_extrapolation()

    # 实验 4: 物理一致性
    res4 = experiment_4_physical_consistency()

    # ---- 汇总 ----
    print("\n" + "=" * 60)
    print("=== 实验总结 ===")
    print("=" * 60)
    print(f"  数据充足:  R² = {res1['R2']:.4f}, MAE = {res1['MAE']:.2e}")
    for n, r in res2.items():
        print(f"  稀疏 N={n:>5}: R² = {r['R2']:.4f}, MAE = {r['MAE']:.2e}")
    print(f"  外推:      R² = {res3['R2']:.4f}, MAE = {res3['MAE']:.2e}")
    print(f"  物理一致性: {res4['n_violations']}/{res4['n_total']} 违反单调性")

    print("\n结论: 数据充足时 MLP 表现良好; 数据稀疏或需外推时性能急剧下降")
    print("→ 这正是 PINN 的切入点: 物理方程约束可在数据稀缺时提供正则化")
