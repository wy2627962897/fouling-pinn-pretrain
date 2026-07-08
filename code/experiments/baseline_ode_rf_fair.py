"""
P0 基线实验：
  1. curve_fit 纯物理拟合 —— 对每条曲线直接用 ODE 模型拟合并报告 R²
  2. RF+ODE features —— 给 RF 喂入 ODE 拟合参数作为附加特征，与 PINN 公平对比

目的：验证"PINN 比纯物理拟合更强"这个核心叙事是否成立。
"""

import json, pickle, sys, time, numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10

sys.path.insert(0, str(Path(__file__).parent))
from fouling_models import fit_model_to_curve, ks_simple_analytical, induction_linear_analytical
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

OUTPUT_DIR = Path(__file__).parent / "output"
RESULTS_DIR = OUTPUT_DIR / "experiment_results"
P0_DIR = OUTPUT_DIR / "p0_baselines"
P0_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
np.random.seed(SEED)

N_SPARSE_LIST = [5, 10, 15, 20, 30]

# 加载真实曲线
with open(OUTPUT_DIR / "pretrain_data" / "real_curves.pkl", "rb") as f:
    real_curves = pickle.load(f)

def get_matched_ode(ode_name):
    """返回可以用于 curve_fit 的 ODE 模型名"""
    if ode_name in ("ks_simple", "kern_seaton", "kern_seaton_offset"):
        return "ks_simple"      # 使用简化 KS（3参数）
    elif ode_name == "induction_linear":
        return "induction_linear"
    else:
        return "ks_simple"

def predict_ode(t_test, ode_name, params):
    """用拟合参数预测整条曲线"""
    if ode_name == "ks_simple":
        return ks_simple_analytical(t_test,
            params.get("Rf_star", 0), params.get("tau", 40), params.get("Rf0", 0))
    elif ode_name == "induction_linear":
        return induction_linear_analytical(t_test,
            params.get("t_ind", 10), params.get("k_growth", 1e-6), params.get("Rf_max", 1e-3))
    return np.zeros_like(t_test)

print("=" * 70)
print("P0: curve_fit 纯物理基线 + RF+ODE 公平对比")
print("=" * 70)

all_results = {}

for curve_name, curve_info in real_curves.items():
    t_full = curve_info["t"]
    rf_full = curve_info["Rf"]

    # 单位转换
    if "source_002" in curve_name:
        rf_full = rf_full * 0.001

    matched_ode = get_matched_ode(curve_info["ode"])
    curve_label = curve_info.get("label", curve_name)

    print(f"\n{'─' * 60}")
    print(f"Curve: {curve_name} ({curve_label})")
    print(f"  n={len(t_full)}, ode={matched_ode}, Rf=[{rf_full.min():.2e}, {rf_full.max():.2e}]")

    curve_results = {}

    for n_sparse in N_SPARSE_LIST:
        if n_sparse >= len(t_full):
            continue

        np.random.seed(SEED)
        indices = np.random.choice(len(t_full), n_sparse, replace=False)
        indices.sort()
        t_sparse = t_full[indices]
        rf_sparse = rf_full[indices]

        key = f"N={n_sparse}"
        curve_results[key] = {}

        # ── 1. curve_fit 纯物理拟合 ──
        t0 = time.time()
        fit_result = fit_model_to_curve(t_sparse, rf_sparse, matched_ode)

        if fit_result.get("success"):
            y_ode = predict_ode(t_full, matched_ode, fit_result["params"])
            ode_r2 = r2_score(rf_full, y_ode)
            ode_mae = mean_absolute_error(rf_full, y_ode)
            ode_params = fit_result["params"]
        else:
            # Fallback: use mean prediction
            y_ode = np.full_like(rf_full, np.mean(rf_sparse))
            ode_r2 = r2_score(rf_full, y_ode)
            ode_mae = mean_absolute_error(rf_full, y_ode)
            ode_params = {}

        curve_results[key]["curve_fit"] = {
            "R2": float(ode_r2), "MAE": float(ode_mae),
            "params": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                       for k, v in ode_params.items()},
            "train_time": time.time() - t0,
        }

        # ── 2. RF (仅 t 特征) ──
        t0 = time.time()
        rf_basic = RandomForestRegressor(n_estimators=100, random_state=SEED)
        rf_basic.fit(t_sparse.reshape(-1, 1), rf_sparse)
        y_rf_basic = rf_basic.predict(t_full.reshape(-1, 1))
        curve_results[key]["RF_basic"] = {
            "R2": float(r2_score(rf_full, y_rf_basic)),
            "MAE": float(mean_absolute_error(rf_full, y_rf_basic)),
            "train_time": time.time() - t0,
        }

        # ── 3. RF + ODE features (公平对比) ──
        # 用 ODE 拟合参数作为附加特征
        t0 = time.time()
        if ode_params:
            # 构造特征矩阵: [t, Rf_star, tau, Rf0] 或 [t, t_ind, k_growth, Rf_max]
            if matched_ode == "ks_simple":
                extra_features = np.column_stack([
                    np.full_like(t_sparse, ode_params.get("Rf_star", 0)),
                    np.full_like(t_sparse, ode_params.get("tau", 40)),
                    np.full_like(t_sparse, ode_params.get("Rf0", 0)),
                ])
                extra_features_test = np.column_stack([
                    np.full_like(t_full, ode_params.get("Rf_star", 0)),
                    np.full_like(t_full, ode_params.get("tau", 40)),
                    np.full_like(t_full, ode_params.get("Rf0", 0)),
                ])
            elif matched_ode == "induction_linear":
                extra_features = np.column_stack([
                    np.full_like(t_sparse, ode_params.get("t_ind", 10)),
                    np.full_like(t_sparse, ode_params.get("k_growth", 1e-6)),
                    np.full_like(t_sparse, ode_params.get("Rf_max", 1e-3)),
                ])
                extra_features_test = np.column_stack([
                    np.full_like(t_full, ode_params.get("t_ind", 10)),
                    np.full_like(t_full, ode_params.get("k_growth", 1e-6)),
                    np.full_like(t_full, ode_params.get("Rf_max", 1e-3)),
                ])
            else:
                extra_features = np.zeros((len(t_sparse), 3))
                extra_features_test = np.zeros((len(t_full), 3))

            X_train = np.column_stack([t_sparse, extra_features])
            X_test = np.column_stack([t_full, extra_features_test])
        else:
            X_train = t_sparse.reshape(-1, 1)
            X_test = t_full.reshape(-1, 1)

        rf_fair = RandomForestRegressor(n_estimators=100, random_state=SEED)
        rf_fair.fit(X_train, rf_sparse)
        y_rf_fair = rf_fair.predict(X_test)

        curve_results[key]["RF_fair"] = {
            "R2": float(r2_score(rf_full, y_rf_fair)),
            "MAE": float(mean_absolute_error(rf_full, y_rf_fair)),
            "train_time": time.time() - t0,
        }

        # ── 打印一行对比 ──
        pin_r2 = None
        pin_data = None
        pin_path = RESULTS_DIR / "all_experiment_results.json"
        if pin_path.exists():
            try:
                with open(pin_path) as f:
                    pin_data = json.load(f)
                pin_r2 = pin_data.get(curve_name, {}).get(key, {}).get("Main: PT+Phys", {}).get("R2")
            except:
                pass

        pin_str = f"PINN={pin_r2:.4f}" if pin_r2 is not None else "PINN=N/A"
        print(f"  N={n_sparse:>2}: ODE-fit R²={ode_r2:.4f} | RF_basic R²={r2_score(rf_full, y_rf_basic):.4f} "
              f"| RF_fair R²={r2_score(rf_full, y_rf_fair):.4f} | {pin_str}")

    all_results[curve_name] = curve_results

# 保存
with open(P0_DIR / "p0_baseline_results.json", "w") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

# ── 汇总表 ──
print(f"\n{'=' * 70}")
print("P0 汇总: curve_fit vs RF_fair vs PINN (N=15)")
print("=" * 70)

print(f"\n{'Curve':<40s} {'ODE-fit':>8s} {'RF_fair':>8s} {'PINN':>8s} {'PINN-ODE':>8s} {'WINNER':>12s}")
print("-" * 90)

winners = {"ODE-fit": 0, "RF_fair": 0, "PINN": 0}
for cn, cr in all_results.items():
    key = "N=15"
    if key not in cr:
        continue
    ode_r2 = cr[key]["curve_fit"]["R2"]
    rf_r2 = cr[key]["RF_fair"]["R2"]
    pin_r2_val = pin_data.get(cn, {}).get(key, {}).get("Main: PT+Phys", {}).get("R2", float('nan')) if pin_data else float('nan')
    pin_r2_val = pin_r2_val if pin_r2_val is not None else float('nan')

    best = max(ode_r2, rf_r2, pin_r2_val if not np.isnan(pin_r2_val) else -999)
    if best == ode_r2: winners["ODE-fit"] += 1
    elif best == rf_r2: winners["RF_fair"] += 1
    else: winners["PINN"] += 1

    pin_ode_diff = pin_r2_val - ode_r2 if not np.isnan(pin_r2_val) else float('nan')
    winner = "ODE-fit" if best == ode_r2 else ("RF_fair" if best == rf_r2 else "PINN")

    cn_short = cn[:38]
    print(f"{cn_short:<40s} {ode_r2:>8.4f} {rf_r2:>8.4f} {pin_r2_val:>8.4f} {pin_ode_diff:>+8.4f} {winner:>12s}")

print(f"\nWinner count: ODE-fit={winners['ODE-fit']}, RF_fair={winners['RF_fair']}, PINN={winners['PINN']}")

# ── 全景对比图 ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: PINN vs ODE-fit scatter
ax = axes[0]
ode_r2s, pin_r2s, labels_plot = [], [], []
for cn, cr in all_results.items():
    key = "N=15"
    if key not in cr: continue
    ode_r2s.append(cr[key]["curve_fit"]["R2"])
    p_r2 = pin_data.get(cn, {}).get(key, {}).get("Main: PT+Phys", {}).get("R2", None) if pin_data else None
    pin_r2s.append(p_r2 if p_r2 is not None else np.nan)
    labels_plot.append(cn[:30])

ode_r2s = np.array(ode_r2s)
pin_r2s = np.array(pin_r2s)

ax.scatter(ode_r2s, pin_r2s, s=60, alpha=0.7)
ax.plot([-1, 1], [-1, 1], "k--", alpha=0.3, label="Parity")
ax.axhline(0, color="gray", alpha=0.3); ax.axvline(0, color="gray", alpha=0.3)
ax.set_xlabel("ODE curve_fit R²"); ax.set_ylabel("PINN R²")
ax.set_title("PINN vs Pure Physics Fit (N=15)")

# 标记优胜者
above = pin_r2s > ode_r2s
ax.scatter(ode_r2s[above], pin_r2s[above], s=60, c="green", alpha=0.7, label=f"PINN wins ({above.sum()})")
ax.scatter(ode_r2s[~above], pin_r2s[~above], s=60, c="red", alpha=0.7, label=f"ODE-fit wins ({(~above).sum()})")
ax.legend(fontsize=8)

# Panel B: RF_fair vs ODE-fit scatter
ax = axes[1]
rf_r2s = np.array([cr["N=15"]["RF_fair"]["R2"] for cn, cr in all_results.items() if "N=15" in cr])

ax.scatter(ode_r2s, rf_r2s, s=60, alpha=0.7)
ax.plot([-1, 1], [-1, 1], "k--", alpha=0.3, label="Parity")
ax.axhline(0, color="gray", alpha=0.3); ax.axvline(0, color="gray", alpha=0.3)
ax.set_xlabel("ODE curve_fit R²"); ax.set_ylabel("RF+ODE features R²")
ax.set_title("RF+ODE vs Pure Physics Fit (N=15)")

above_rf = rf_r2s > ode_r2s
ax.scatter(ode_r2s[above_rf], rf_r2s[above_rf], s=60, c="green", alpha=0.7, label=f"RF+ODE wins ({above_rf.sum()})")
ax.scatter(ode_r2s[~above_rf], rf_r2s[~above_rf], s=60, c="red", alpha=0.7, label=f"ODE-fit wins ({(~above_rf).sum()})")
ax.legend(fontsize=8)

fig.suptitle("P0 Baseline Experiment: Is PINN Actually Better Than Simpler Methods?", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(P0_DIR / "p0_comparison.png", dpi=200)
plt.close(fig)

print(f"\nResults saved to {P0_DIR}")
print("Done.")
