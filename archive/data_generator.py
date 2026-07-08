"""
合成数据生成器
用 Kern-Seaton 仿真器批量生成 "工况 → Rf(t)" 数据，加噪声模拟测量。

输出两个数据集：
  - full_dataset.csv:   完整时间序列 (10,000 工况 × 时间步)
  - sparse_samples.csv: 稀疏采样 (每工况随机选 20 个时间点)
"""

import math
import random
import csv
from pathlib import Path
import fouling_simulator as fs

OUTPUT_DIR = Path(__file__).parent / "output"
SEED = 42


def set_seed():
    random.seed(SEED)


def sample_params():
    """随机采样一组工况参数 (在合理范围内均匀分布)"""
    return {
        "A": 10 ** random.uniform(-5.5, -4.0),      # 3e-6 ~ 1e-4
        "Ea": random.uniform(35000, 60000),          # 35-60 kJ/mol
        "T_w": random.uniform(320, 400),             # 47-127 °C
        "B": 10 ** random.uniform(-8.5, -7.0),       # 3e-9 ~ 1e-7
        "rho": random.uniform(950, 1050),            # 密度 (温度影响)
        "v": random.uniform(0.3, 4.0),               # 0.3-4.0 m/s
        "f": 0.005,
        "U_clean": random.uniform(800, 1800),        # W/(m2.K)
        "mode": "kern-seaton",
    }


def generate_dataset(n_conditions=10000, duration_days=1825, dt_days=30, noise_std=2e-5):
    """
    生成合成数据集。

    参数
    ----
    n_conditions : 工况数量 (默认 10000)
    duration_days: 仿真时长 (天)
    dt_days      : 采样间隔 (天)
    noise_std    : 测量噪声标准差 [m2.K/W] (约为 Rf 量级的 5-10%)

    返回
    ----
    list[dict]: 每条记录包含 (condition_id, Tw, v, conc_factor, day, Rf_true, Rf_noisy)
    """
    set_seed()
    records = []

    for cid in range(n_conditions):
        p = sample_params()
        results = fs.simulate(p, duration_days=duration_days, dt_days=dt_days)

        for r in results:
            noise = random.gauss(0, noise_std)
            records.append({
                "condition_id": cid,
                "Tw": p["T_w"],
                "v": p["v"],
                "A": p["A"],
                "Ea": p["Ea"],
                "B": p["B"],
                "U_clean": p["U_clean"],
                "day": r["day"],
                "Rf_true": r["Rf"],
                "Rf_noisy": r["Rf"] + noise,
            })

        if (cid + 1) % 1000 == 0:
            print(f"  已生成 {cid + 1}/{n_conditions} 个工况...")

    return records


def save_full_dataset(records, filepath):
    """保存完整数据集为 CSV"""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["condition_id", "Tw", "v", "A", "Ea", "B", "U_clean",
                         "day", "Rf_true", "Rf_noisy"])
        for r in records:
            writer.writerow([
                r["condition_id"], r["Tw"], r["v"], r["A"], r["Ea"], r["B"],
                r["U_clean"], r["day"], r["Rf_true"], r["Rf_noisy"]
            ])


def save_sparse_samples(records, n_samples_per_condition=20, filepath=None):
    """每个工况随机采样少量时间点，模拟"仅有少量测量数据"的场景"""
    if filepath is None:
        filepath = OUTPUT_DIR / "sparse_samples.csv"

    # 按 condition_id 分组
    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        groups[r["condition_id"]].append(r)

    sparse = []
    set_seed()
    for cid, group in groups.items():
        if len(group) <= n_samples_per_condition:
            sampled = group
        else:
            sampled = random.sample(group, n_samples_per_condition)
        # 按时间排序
        sampled.sort(key=lambda x: x["day"])
        sparse.extend(sampled)

    save_full_dataset(sparse, filepath)
    return sparse


def split_by_condition(records, train_ratio=0.7, val_ratio=0.15):
    """按工况 ID 划分 train/val/test (保证同工况的数据在同一集合)"""
    set_seed()
    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        groups[r["condition_id"]].append(r)

    cids = list(groups.keys())
    random.shuffle(cids)

    n_train = int(len(cids) * train_ratio)
    n_val = int(len(cids) * val_ratio)

    train_cids = set(cids[:n_train])
    val_cids = set(cids[n_train:n_train + n_val])
    test_cids = set(cids[n_train + n_val:])

    train, val, test = [], [], []
    for r in records:
        if r["condition_id"] in train_cids:
            train.append(r)
        elif r["condition_id"] in val_cids:
            val.append(r)
        else:
            test.append(r)

    return train, val, test


# ===================================================================
if __name__ == "__main__":
    print("=== 生成合成数据集 ===")
    print("参数范围: Tw=320~400K, v=0.3~4.0m/s, A=3e-6~1e-4, B=3e-9~1e-7")

    # 生成
    records = generate_dataset(n_conditions=10000, duration_days=1825, dt_days=30)

    # 划分
    train, val, test = split_by_condition(records)
    print(f"\n数据集划分: train={len(train)}, val={len(val)}, test={len(test)}")

    # 保存
    save_full_dataset(train, OUTPUT_DIR / "train_full.csv")
    save_full_dataset(val, OUTPUT_DIR / "val_full.csv")
    save_full_dataset(test, OUTPUT_DIR / "test_full.csv")
    print("\n完整数据集已保存: train_full.csv, val_full.csv, test_full.csv")

    # 稀疏采样
    save_sparse_samples(train, n_samples_per_condition=20,
                        filepath=OUTPUT_DIR / "train_sparse_20.csv")
    save_sparse_samples(train, n_samples_per_condition=50,
                        filepath=OUTPUT_DIR / "train_sparse_50.csv")
    save_sparse_samples(train, n_samples_per_condition=100,
                        filepath=OUTPUT_DIR / "train_sparse_100.csv")
    print("稀疏数据集已保存: train_sparse_{20,50,100}.csv")

    # 外推测试集: 只保留前半段时间用于训练, 后半段用于测试
    print("\n=== 生成外推测试集 ===")
    train_extrap, test_extrap = [], []
    for r in train:
        if r["day"] <= 1095:  # 前 3 年
            train_extrap.append(r)
        else:  # 第 3-5 年
            test_extrap.append(r)
    save_full_dataset(train_extrap, OUTPUT_DIR / "train_extrap.csv")
    save_full_dataset(test_extrap, OUTPUT_DIR / "test_extrap.csv")
    print(f"外推: train (≤3年)={len(train_extrap)}, test (>3年)={len(test_extrap)}")

    print("\n=== 数据生成完成 ===")
