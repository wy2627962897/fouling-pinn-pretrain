"""
PINN vs MLP 精简实验 (GPU)
快速获取三组关键对比数据
"""
import os; os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys, time, math
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np, pandas as pd, torch

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

import pinn_model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
OUTPUT = "output"
FEATS = ["Tw", "v", "A", "Ea", "B", "U_clean", "day"]

torch.manual_seed(SEED); np.random.seed(SEED)
print(f"设备: {DEVICE} | {torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'}\n")

train_full = pd.read_csv(f"{OUTPUT}/train_full.csv")
test_full = pd.read_csv(f"{OUTPUT}/test_full.csv")


# ---- 实验 1: 稀疏数据 ----
print("=" * 50)
print("实验 1: PINN vs MLP (稀疏数据)")
print("=" * 50)

Ns = [20, 50, 100, 500]
results1 = []
for n in Ns:
    train_n = train_full.sample(n=n, random_state=SEED)
    Xt = train_n[FEATS].values
    yt = train_n["Rf_noisy"].values
    Xe = test_full[FEATS].values
    ye = test_full["Rf_true"].values

    # MLP
    t0 = time.time()
    mlp = MLPRegressor((64, 32), max_iter=500, early_stopping=True,
                        random_state=SEED).fit(StandardScaler().fit_transform(Xt), yt)
    yp_mlp = mlp.predict(StandardScaler().fit_transform(Xe))
    t_mlp = time.time() - t0
    r2m, maem = r2_score(ye, yp_mlp), mean_absolute_error(ye, yp_mlp)

    # PINN
    t0 = time.time()
    xd, yd, _ = pinn_model.prepare_tensors(train_n, DEVICE)
    xp = pinn_model.generate_collocation_points(300, device=DEVICE)
    model = pinn_model.PINN((64, 64, 64, 64)).to(DEVICE)
    pinn_model.train_pinn(model, xd, yd, xp, lambda_phys=0.3,
                          n_epochs=1500, verbose=False)
    res_p = pinn_model.evaluate(model, test_full, DEVICE)
    t_pinn = time.time() - t0

    print(f"  N={n:>3}: MLP R²={r2m:>8.2f} MAE={maem:.2e} ({t_mlp:.1f}s)"
          f" | PINN R²={res_p['R2']:>8.2f} MAE={res_p['MAE']:.2e} ({t_pinn:.1f}s)")
    results1.append({"N": n, "mlp_r2": r2m, "pinn_r2": res_p["R2"],
                     "mlp_mae": maem, "pinn_mae": res_p["MAE"]})


# ---- 实验 2: 外推 ----
print("\n" + "=" * 50)
print("实验 2: PINN vs MLP (外推)")
print("=" * 50)

train_ext = pd.read_csv(f"{OUTPUT}/train_extrap.csv").sample(n=5000, random_state=SEED)
test_ext = pd.read_csv(f"{OUTPUT}/test_extrap.csv").sample(n=10000, random_state=SEED)
Xt, yt = train_ext[FEATS].values, train_ext["Rf_noisy"].values
Xe, ye = test_ext[FEATS].values, test_ext["Rf_true"].values

# MLP
t0 = time.time(); scaler = StandardScaler()
mlp = MLPRegressor((128, 64, 32), max_iter=500, early_stopping=True,
                    random_state=SEED).fit(scaler.fit_transform(Xt), yt)
yp_mlp = mlp.predict(scaler.transform(Xe))
print(f"  MLP:  R²={r2_score(ye,yp_mlp):.4f} MAE={mean_absolute_error(ye,yp_mlp):.2e} ({time.time()-t0:.1f}s)")

# 按外推距离
days = test_ext["day"].values
for lo, hi in [(1095, 1460), (1460, 1643), (1643, 1825)]:
    m = (days >= lo) & (days <= hi)
    if m.sum():
        print(f"    [{lo}-{hi}]: R²={r2_score(ye[m],yp_mlp[m]):.4f}")

# PINN
t0 = time.time()
xd, yd, _ = pinn_model.prepare_tensors(train_ext, DEVICE)
xp = pinn_model.generate_collocation_points(500, device=DEVICE)
model = pinn_model.PINN((64, 64, 64, 64)).to(DEVICE)
pinn_model.train_pinn(model, xd, yd, xp, lambda_phys=1.0, n_epochs=2000, verbose=False)
res_p = pinn_model.evaluate(model, test_ext, DEVICE)
print(f"  PINN: R²={res_p['R2']:.4f} MAE={res_p['MAE']:.2e} ({time.time()-t0:.1f}s)")
for lo, hi in [(1095, 1460), (1460, 1643), (1643, 1825)]:
    m = (days >= lo) & (days <= hi)
    if m.sum():
        print(f"    [{lo}-{hi}]: R²={r2_score(ye[m],res_p['y_pred'][m]):.4f}")


# ---- 实验 3: 物理一致性 ----
print("\n" + "=" * 50)
print("实验 3: 物理一致性 (单调性)")
print("=" * 50)

train3 = train_full.sample(n=3000, random_state=SEED)
cids = test_full["condition_id"].value_counts()
good = cids[cids >= 30].index[:3].tolist()

# MLP
X3, y3 = train3[FEATS].values, train3["Rf_noisy"].values
mlp3 = MLPRegressor((64, 64, 32), max_iter=300, random_state=SEED)
mlp3.fit(StandardScaler().fit_transform(X3), y3)

# PINN
xd3, yd3, _ = pinn_model.prepare_tensors(train3, DEVICE)
xp3 = pinn_model.generate_collocation_points(300, device=DEVICE)
pinn3 = pinn_model.PINN((64, 64, 64, 64)).to(DEVICE)
pinn_model.train_pinn(pinn3, xd3, yd3, xp3, lambda_phys=0.5, n_epochs=1500, verbose=False)

print(f"{'CID':<8} {'MLP违反':>10} {'PINN违反':>10}")
mlp_total = pinn_total = total = 0
for cid in good:
    m = (test_full["condition_id"] == cid).values
    dfc = test_full[m].sort_values("day")
    yt_c = dfc["Rf_true"].values

    yp_m = mlp3.predict(StandardScaler().fit_transform(dfc[FEATS].values))
    n_m = (np.diff(yp_m) < 0).sum()

    xt, _, _ = pinn_model.prepare_tensors(dfc, DEVICE)
    pinn3.eval()
    with torch.no_grad():
        yp_p = pinn3(xt).cpu().numpy().flatten()
    n_p = (np.diff(yp_p) < 0).sum()

    mlp_total += n_m; pinn_total += n_p; total += len(np.diff(yp_m))
    print(f"  CID={int(cid):<4} {n_m:>5}/{len(np.diff(yp_m)):<5} {n_p:>5}/{len(np.diff(yp_p))}")

print(f"\n  MLP:  {mlp_total}/{total} 违反 ({100*mlp_total/total:.1f}%)")
print(f"  PINN: {pinn_total}/{total} 违反 ({100*pinn_total/total:.1f}%)")


# ---- 图: 稀疏数据对比 ----
print("\n=== 绘制对比图 ===")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
Ns_plot = [r["N"] for r in results1]

ax1.semilogx(Ns_plot, [r["mlp_r2"] for r in results1], "r-o", lw=2, ms=8, label="MLP")
ax1.semilogx(Ns_plot, [r["pinn_r2"] for r in results1], "b-s", lw=2, ms=8, label="PINN")
ax1.set_xlabel("训练样本数"); ax1.set_ylabel("R²")
ax1.set_title("PINN vs MLP: R² (稀疏数据)"); ax1.legend(); ax1.grid(alpha=.3)
ax1.axhline(y=0, color="gray", ls="--", alpha=.5)

ax2.loglog(Ns_plot, [r["mlp_mae"] for r in results1], "r-o", lw=2, ms=8, label="MLP")
ax2.loglog(Ns_plot, [r["pinn_mae"] for r in results1], "b-s", lw=2, ms=8, label="PINN")
ax2.set_xlabel("训练样本数"); ax2.set_ylabel("MAE (m²·K/W)")
ax2.set_title("PINN vs MLP: MAE (稀疏数据)"); ax2.legend(); ax2.grid(alpha=.3)

fig.tight_layout(); fig.savefig(f"{OUTPUT}/pinn_vs_mlp_sparse.png", dpi=150)
print(f"  已保存: {OUTPUT}/pinn_vs_mlp_sparse.png")

# 总结
print("\n" + "=" * 50)
print("=== 最终总结 ===")
print("=" * 50)
print("稀疏数据:")
for r in results1:
    print(f"  N={r['N']:>3}: MLP R²={r['mlp_r2']:>8.2f}  PINN R²={r['pinn_r2']:>8.2f}")
print(f"外推: MLP R²={r2_score(ye,yp_mlp):.4f}  PINN R²={res_p['R2']:.4f}")
print(f"物理一致性: MLP {mlp_total}/{total}  PINN {pinn_total}/{total}")
