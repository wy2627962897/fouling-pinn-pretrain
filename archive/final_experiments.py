"""
PINN vs MLP 最终对比实验 + 三张论文级图表

优化: 外推场景调参 (更高 lambda_phys, 更多配点, 更长训练)
"""
import os; os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys, time, math
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np, pandas as pd, torch

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

import pinn_model
import fouling_simulator as fs

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
OUTPUT = "output"
FEATS = ["Tw", "v", "A", "Ea", "B", "U_clean", "day"]
FIGS_DIR = f"{OUTPUT}/figures"

torch.manual_seed(SEED); np.random.seed(SEED)
import pathlib; pathlib.Path(FIGS_DIR).mkdir(parents=True, exist_ok=True)

print(f"设备: {DEVICE} | {torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'}\n")

train_full = pd.read_csv(f"{OUTPUT}/train_full.csv")
test_full = pd.read_csv(f"{OUTPUT}/test_full.csv")


# ===================================================================
# 训练助手
# ===================================================================

def run_mlp(X_train, y_train, X_test, y_test, hidden=(64, 32)):
    t0 = time.time()
    mlp = MLPRegressor(hidden, max_iter=500, early_stopping=True,
                        random_state=SEED).fit(StandardScaler().fit_transform(X_train), y_train)
    yp = mlp.predict(StandardScaler().fit_transform(X_test))
    return {"R2": r2_score(y_test, yp), "MAE": mean_absolute_error(y_test, yp),
            "y_pred": yp, "time": time.time()-t0}, mlp


def run_pinn(train_df, test_df, lambda_phys=0.5, n_epochs=2000,
             hidden=(64, 64, 64, 64), n_colloc=500, t_max=1825):
    t0 = time.time()
    xd, yd, _ = pinn_model.prepare_tensors(train_df, DEVICE)
    xp = pinn_model.generate_collocation_points(n_colloc, t_range=(0, t_max), device=DEVICE)
    model = pinn_model.PINN(hidden).to(DEVICE)
    pinn_model.train_pinn(model, xd, yd, xp, lambda_phys=lambda_phys,
                          n_epochs=n_epochs, verbose=False)
    res = pinn_model.evaluate(model, test_df, DEVICE)
    res["time"] = time.time() - t0
    return res, model


# ===================================================================
# 实验 1: 稀疏数据 (扩展 + 优化 PINN)
# ===================================================================
print("=" * 60)
print("实验 1: PINN vs MLP — 稀疏数据")
print("=" * 60)

X_test = test_full[FEATS].values
y_test = test_full["Rf_true"].values
Ns = [20, 50, 100, 200, 500, 1000, 2000]
results_sparse = []

for n in Ns:
    train_n = train_full.sample(n=n, random_state=SEED)
    Xt, yt = train_n[FEATS].values, train_n["Rf_noisy"].values

    # MLP
    res_m, _ = run_mlp(Xt, yt, X_test, y_test, hidden=(64, 32))

    # PINN (更多 epoch, 自适应 lambda)
    lam = 0.2 if n >= 500 else 0.5  # 数据越少物理越重要
    ep = 2500 if n >= 200 else 1500
    res_p, _ = run_pinn(train_n, test_full, lambda_phys=lam, n_epochs=ep)

    results_sparse.append({"N": n, "mlp": res_m, "pinn": res_p})
    print(f"  N={n:>4}: MLP R²={res_m['R2']:>8.2f} MAE={res_m['MAE']:.2e}"
          f" | PINN R²={res_p['R2']:>7.3f} MAE={res_p['MAE']:.2e}")


# ===================================================================
# 实验 2: 外推 (优化版)
# ===================================================================
print("\n" + "=" * 60)
print("实验 2: PINN vs MLP — 外推 (优化 PINN)")
print("=" * 60)

train_ext = pd.read_csv(f"{OUTPUT}/train_extrap.csv").sample(n=8000, random_state=SEED)
test_ext_all = pd.read_csv(f"{OUTPUT}/test_extrap.csv")
test_ext = test_ext_all.sample(n=15000, random_state=SEED)
Xt_ex, yt_ex = train_ext[FEATS].values, train_ext["Rf_noisy"].values
Xe_ex, ye_ex = test_ext[FEATS].values, test_ext["Rf_true"].values
days_ex = test_ext["day"].values

# MLP
res_m_ex, mlp_ex = run_mlp(Xt_ex, yt_ex, Xe_ex, ye_ex, hidden=(128, 64, 32))
print(f"  MLP:  R²={res_m_ex['R2']:.4f} MAE={res_m_ex['MAE']:.2e} ({res_m_ex['time']:.1f}s)")
for lo, hi in [(1095, 1460), (1460, 1643), (1643, 1825)]:
    m = (days_ex >= lo) & (days_ex <= hi)
    if m.sum(): print(f"    [{lo}-{hi}]: R²={r2_score(ye_ex[m],res_m_ex['y_pred'][m]):.4f}")

# PINN 优化: 更高 lambda_phys, 更多 epoch, 配点覆盖全时间域
res_p_ex, pinn_ex = run_pinn(train_ext, test_ext, lambda_phys=2.0, n_epochs=4000,
                              hidden=(128, 64, 64, 32), n_colloc=800)
print(f"  PINN: R²={res_p_ex['R2']:.4f} MAE={res_p_ex['MAE']:.2e} ({res_p_ex['time']:.1f}s)")
for lo, hi in [(1095, 1460), (1460, 1643), (1643, 1825)]:
    m = (days_ex >= lo) & (days_ex <= hi)
    if m.sum(): print(f"    [{lo}-{hi}]: R²={r2_score(ye_ex[m],res_p_ex['y_pred'][m]):.4f}")


# ===================================================================
# 实验 3: 物理一致性 (5 个工况)
# ===================================================================
print("\n" + "=" * 60)
print("实验 3: 物理一致性 — 单调性检查")
print("=" * 60)

train3 = train_full.sample(n=5000, random_state=SEED)
cids = test_full["condition_id"].value_counts()
good = cids[cids >= 30].index[:5].tolist()

res_m3, mlp3 = run_mlp(train3[FEATS].values, train3["Rf_noisy"].values,
                        test_full[FEATS].values, test_full["Rf_true"].values,
                        hidden=(128, 64, 32))
res_p3, pinn3 = run_pinn(train3, test_full, lambda_phys=0.5, n_epochs=2000,
                          hidden=(128, 64, 64, 32))

consistency = []
print(f"{'CID':<8} {'MLP违反':>8} {'PINN违反':>8}")
for cid in good:
    m = (test_full["condition_id"] == cid).values
    dfc = test_full[m].sort_values("day")

    yp_m = mlp3.predict(StandardScaler().fit_transform(dfc[FEATS].values))
    n_m = (np.diff(yp_m) < 0).sum()

    xt, _, _ = pinn_model.prepare_tensors(dfc, DEVICE)
    pinn3.eval()
    with torch.no_grad(): yp_p = pinn3(xt).cpu().numpy().flatten()
    n_p = (np.diff(yp_p) < 0).sum()

    consistency.append({"cid": int(cid), "n": len(np.diff(yp_m)),
                        "mlp_v": n_m, "pinn_v": n_p,
                        "y_mlp": yp_m, "y_pinn": yp_p,
                        "y_true": dfc["Rf_true"].values,
                        "days": dfc["day"].values})
    print(f"  CID={int(cid):<4} {n_m:>5}/{len(np.diff(yp_m)):<5} {n_p:>5}/{len(np.diff(yp_p))}")


# ===================================================================
# 图 1: 稀疏数据优势 — 双面板 (R² + MAE)
# ===================================================================
print("\n=== 生成图 1: 稀疏数据对比 ===")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
Ns_p = [r["N"] for r in results_sparse]
mlp_r2 = [r["mlp"]["R2"] for r in results_sparse]
pinn_r2 = [r["pinn"]["R2"] for r in results_sparse]
mlp_mae = [r["mlp"]["MAE"] for r in results_sparse]
pinn_mae = [r["pinn"]["MAE"] for r in results_sparse]

color_mlp = "#e74c3c"; color_pinn = "#2980b9"

# R² 面板
ax1.semilogx(Ns_p, mlp_r2, "o-", color=color_mlp, lw=2, ms=10, label="MLP (纯数据)")
ax1.semilogx(Ns_p, pinn_r2, "s-", color=color_pinn, lw=2.5, ms=10, label="PINN (物理+数据)")
ax1.axhline(y=0, color="gray", ls="--", alpha=0.6, lw=1)
ax1.axhline(y=0.9, color="green", ls=":", alpha=0.4, lw=1, label="R²=0.9")
ax1.set_xlabel("训练样本数 N", fontsize=13)
ax1.set_ylabel("R²", fontsize=13)
ax1.set_title("预测精度 R² vs 训练样本数", fontsize=14, fontweight="bold")
ax1.legend(fontsize=11, loc="lower right")
ax1.grid(True, alpha=0.3)
ax1.set_ylim(min(mlp_r2[2:]) - 50, 1.1)

# N=20,50 时 MLP R² 太低, 标注实际值
for i in [0, 1]:
    ax1.annotate(f"R²={mlp_r2[i]:.0f}", (Ns_p[i], mlp_r2[i]),
                 textcoords="offset points", xytext=(0, -20), ha="center",
                 fontsize=8, color=color_mlp)

# MAE 面板
ax2.loglog(Ns_p, mlp_mae, "o-", color=color_mlp, lw=2, ms=10, label="MLP (纯数据)")
ax2.loglog(Ns_p, pinn_mae, "s-", color=color_pinn, lw=2.5, ms=10, label="PINN (物理+数据)")
ax2.set_xlabel("训练样本数 N", fontsize=13)
ax2.set_ylabel("MAE (m²·K/W)", fontsize=13)
ax2.set_title("预测误差 MAE vs 训练样本数", fontsize=14, fontweight="bold")
ax2.legend(fontsize=11, loc="upper right")
ax2.grid(True, alpha=0.3)

fig.suptitle("PINN vs MLP: 稀疏数据场景下的优势", fontsize=16, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{FIGS_DIR}/fig1_sparse_advantage.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  已保存: {FIGS_DIR}/fig1_sparse_advantage.png")


# ===================================================================
# 图 2: 外推能力 — 误差 vs 外推距离
# ===================================================================
print("=== 生成图 2: 外推对比 ===")

# 按 day 分组计算 MAE
def mae_by_day(days, y_true, y_pred):
    udays = np.unique(days); udays.sort()
    return [(d, np.abs(y_pred[days==d] - y_true[days==d]).mean()) for d in udays]

mae_m = mae_by_day(days_ex, ye_ex, res_m_ex["y_pred"])
mae_p = mae_by_day(days_ex, ye_ex, res_p_ex["y_pred"])

fig, ax = plt.subplots(figsize=(14, 6))
d_m = [d for d,_ in mae_m]; e_m = [e*1e3 for _,e in mae_m]
d_p = [d for d,_ in mae_p]; e_p = [e*1e3 for _,e in mae_p]

ax.plot(d_m, e_m, color=color_mlp, lw=1.5, alpha=0.9, label="MLP (纯数据)")
ax.plot(d_p, e_p, color=color_pinn, lw=1.5, alpha=0.9, label="PINN (物理+数据)")
ax.axvline(x=1095, color="red", ls="--", lw=2, label="训练边界 (第3年)")

# 外推域着色
ax.axvspan(1095, 1825, alpha=0.08, color="red")
ax.text(1460, ax.get_ylim()[1]*0.95, "外推域", ha="center", fontsize=12, color="red", alpha=0.6)

ax.set_xlabel("运行时间 (天)", fontsize=13)
ax.set_ylabel("MAE × 10³ (m²·K/W)", fontsize=13)
ax.set_title("外推能力对比: PINN vs MLP (训练 ≤ 第3年)", fontsize=14, fontweight="bold")
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(f"{FIGS_DIR}/fig2_extrapolation.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  已保存: {FIGS_DIR}/fig2_extrapolation.png")


# ===================================================================
# 图 3: 物理一致性 — 单调性 + 轨迹对比
# ===================================================================
print("=== 生成图 3: 物理一致性 ===")

fig = plt.figure(figsize=(18, 10))

# 上方: 单个工况的 Rf(t) 轨迹对比 (3 个子图)
for i, c in enumerate(consistency[:3]):
    ax = fig.add_subplot(2, 3, i+1)
    days = c["days"]; yt = c["y_true"]; ym = c["y_mlp"]; yp = c["y_pinn"]

    ax.plot(days, yt*1e3, "k-", lw=2, label="真实值 (物理模型)")
    ax.plot(days, ym*1e3, "--", color=color_mlp, lw=1.5, alpha=0.8, label=f"MLP (违反{c['mlp_v']}次)")
    ax.plot(days, yp*1e3, "-", color=color_pinn, lw=2, alpha=0.9, label=f"PINN (违反{c['pinn_v']}次)")

    ax.set_xlabel("运行时间 (天)", fontsize=11)
    ax.set_ylabel("Rf × 10³ (m²·K/W)", fontsize=11)
    ax.set_title(f"工况 CID={c['cid']}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# 下方: 单调性违反统计
ax_bar = fig.add_subplot(2, 1, 2)
cids_lbl = [f"CID={c['cid']}" for c in consistency]
mlp_v = [100*c["mlp_v"]/c["n"] for c in consistency]
pinn_v = [100*c["pinn_v"]/c["n"] for c in consistency]
total_mlp = sum(c["mlp_v"] for c in consistency)
total_pinn = sum(c["pinn_v"] for c in consistency)
total_n = sum(c["n"] for c in consistency)

x = np.arange(len(cids_lbl))
w = 0.35
bars1 = ax_bar.bar(x - w/2, mlp_v, w, color=color_mlp, alpha=0.85, label=f"MLP ({100*total_mlp/total_n:.1f}%)")
bars2 = ax_bar.bar(x + w/2, pinn_v, w, color=color_pinn, alpha=0.85, label=f"PINN ({100*total_pinn/total_n:.1f}%)")

for bar, val in zip(bars1, mlp_v):
    ax_bar.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{val:.1f}%",
                ha="center", fontsize=9, fontweight="bold")
for bar, val in zip(bars2, pinn_v):
    ax_bar.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{val:.1f}%",
                ha="center", fontsize=9, fontweight="bold")

ax_bar.set_xticks(x); ax_bar.set_xticklabels(cids_lbl, fontsize=12)
ax_bar.set_ylabel("违反单调性比例 (%)", fontsize=13)
ax_bar.set_title("物理一致性: Rf(t) 单调递增约束", fontsize=14, fontweight="bold")
ax_bar.legend(fontsize=12)
ax_bar.grid(True, alpha=0.2, axis="y")
ax_bar.set_ylim(0, max(max(mlp_v), max(pinn_v))*1.3)

fig.suptitle("物理一致性检查: PINN vs MLP", fontsize=16, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{FIGS_DIR}/fig3_physical_consistency.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  已保存: {FIGS_DIR}/fig3_physical_consistency.png")


# ===================================================================
# 最终总结
# ===================================================================
print("\n" + "=" * 60)
print("=== 最终汇总 ===")
print("=" * 60)

print("\n--- 稀疏数据 ---")
print(f"{'N':>6} {'MLP R²':>10} {'PINN R²':>10} {'MLP MAE':>10} {'PINN MAE':>10}")
for r in results_sparse:
    print(f"{r['N']:>6} {r['mlp']['R2']:>10.2f} {r['pinn']['R2']:>10.3f} "
          f"{r['mlp']['MAE']:>10.2e} {r['pinn']['MAE']:>10.2e}")

print(f"\n--- 外推 ---")
print(f"  MLP:  R²={res_m_ex['R2']:.4f}, MAE={res_m_ex['MAE']:.2e}")
print(f"  PINN: R²={res_p_ex['R2']:.4f}, MAE={res_p_ex['MAE']:.2e}")

print(f"\n--- 物理一致性 ---")
print(f"  MLP:  {total_mlp}/{total_n} 违反单调性 ({100*total_mlp/total_n:.1f}%)")
print(f"  PINN: {total_pinn}/{total_n} 违反单调性 ({100*total_pinn/total_n:.1f}%)")

print(f"\n图已保存至: {FIGS_DIR}/")
