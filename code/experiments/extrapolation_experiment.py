"""
Extrapolation experiment: PINN vs RF on unseen future time.

Core narrative: RF can interpolate but can't extrapolate.
PINN with physics pre-training can predict beyond training time range.

Experiment: Train on first 50% of time → Predict full curve (including last 50%).
"""

import os, sys, time, pickle, json, numpy as np, torch
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).parent))
import pinn_model
from pinn_model import (PINN, ODE_TYPE_IDS, encode_condition_params,
    pretrain_pinn_multi_ode, finetune_pinn, predict_trajectory,
    evaluate_trajectory, estimate_ode_params_from_sparse)
from multi_ode_data_generator import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

OUTPUT_DIR = Path(__file__).parent / "output"
FIGS_DIR = OUTPUT_DIR / "experiment_figures"
RESULTS_DIR = OUTPUT_DIR / "experiment_results"
for d in [FIGS_DIR, RESULTS_DIR]: d.mkdir(parents=True, exist_ok=True)

N_FT_EPOCHS = 2000
LAMBDA_PHYS = 0.3
N_COLLOC = 150
N_SPARSE_LIST = [5, 10, 15, 20]

print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load data
with open(OUTPUT_DIR / "pretrain_data" / "real_curves.pkl", "rb") as f:
    real_curves = pickle.load(f)

SRC2_UNIT_CONV = 0.001

# Prepare curves
curves = {}
for name, info in real_curves.items():
    t = info["t"].copy()
    rf = info["Rf"].copy()
    if "source_002" in name:
        rf = rf * SRC2_UNIT_CONV
    curves[name] = {"t": t, "Rf": rf, "ode": info["ode"], "label": info["label"]}

# Load pre-trained model
pretrained_state = torch.load(RESULTS_DIR / "pretrained_model.pt", weights_only=True)

print("\n" + "=" * 60)
print("Extrapolation Experiment: Train first 50%, predict full curve")
print("=" * 60)

all_extrap_results = {}

for curve_name, curve_info in curves.items():
    t_full = curve_info["t"]
    rf_full = curve_info["Rf"]
    ode_name = curve_info["ode"]

    t_mid = (t_full.min() + t_full.max()) / 2
    train_mask = t_full <= t_mid
    test_mask = t_full > t_mid

    t_train_all = t_full[train_mask]
    rf_train_all = rf_full[train_mask]
    t_test = t_full[test_mask]

    if len(t_train_all) < 5 or len(t_test) < 3:
        print(f"  {curve_name}: too few points for extrapolation, skipping")
        continue

    print(f"\n{'─' * 60}")
    print(f"Curve: {curve_name} ({curve_info['label']})")
    print(f"  Train region: t=[{t_train_all.min():.1f}, {t_train_all.max():.1f}], {len(t_train_all)} pts")
    print(f"  Test region:  t=[{t_test.min():.1f}, {t_test.max():.1f}], {len(t_test)} pts")

    curve_extrap = {}

    for n_sparse in N_SPARSE_LIST:
        if n_sparse >= len(t_train_all):
            continue

        np.random.seed(SEED)
        indices = np.random.choice(len(t_train_all), min(n_sparse, len(t_train_all)), replace=False)
        indices.sort()
        t_sparse = t_train_all[indices]
        rf_sparse = rf_train_all[indices]

        # ---- PINN ----
        model = PINN(hidden_layers=(64, 64, 64, 64)).to(DEVICE)
        model.load_state_dict(pretrained_state)
        ode_params_est = estimate_ode_params_from_sparse(t_sparse, rf_sparse, ode_name)
        t_range = (float(t_full.min()), float(t_full.max()))

        finetune_pinn(model, t_sparse, rf_sparse, ode_name, ode_params_est, t_range,
                      n_epochs=N_FT_EPOCHS, lambda_phys=LAMBDA_PHYS, lr=1e-4,
                      n_colloc=N_COLLOC, device=DEVICE, verbose=False)

        y_pred_full = predict_trajectory(model, t_full, ode_name, ode_params_est, DEVICE)
        y_pred_test = predict_trajectory(model, t_test, ode_name, ode_params_est, DEVICE)

        metrics_full = evaluate_trajectory(y_pred_full, rf_full)
        metrics_test = evaluate_trajectory(y_pred_test, rf_full[test_mask])

        # ---- RF ----
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score, mean_absolute_error
        rf_model = RandomForestRegressor(n_estimators=100, random_state=SEED)
        rf_model.fit(t_sparse.reshape(-1, 1), rf_sparse)
        y_rf_test = rf_model.predict(t_test.reshape(-1, 1))
        y_rf_full = rf_model.predict(t_full.reshape(-1, 1))

        rf_r2_test = r2_score(rf_full[test_mask], y_rf_test)
        rf_r2_full = r2_score(rf_full, y_rf_full)

        key = f"N={n_sparse}"
        curve_extrap[key] = {
            "pinn_full_R2": metrics_full["R2"],
            "pinn_test_R2": metrics_test["R2"],
            "pinn_test_MAE": metrics_test["MAE"],
            "rf_full_R2": rf_r2_full,
            "rf_test_R2": rf_r2_test,
            "y_pred_full": y_pred_full,
            "y_rf_full": y_rf_full,
        }

        print(f"  N={n_sparse:>2}: PINN test R2={metrics_test['R2']:.4f} | RF test R2={rf_r2_test:.4f} | "
              f"Advantage={metrics_test['R2']-rf_r2_test:+.4f}")

    all_extrap_results[curve_name] = curve_extrap

# Save
extrap_serializable = {}
for cname, cdata in all_extrap_results.items():
    extrap_serializable[cname] = {}
    for nkey, ndata in cdata.items():
        extrap_serializable[cname][nkey] = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in ndata.items() if k not in ("y_pred_full", "y_rf_full")
        }
with open(RESULTS_DIR / "extrapolation_results.json", "w") as f:
    json.dump(extrap_serializable, f, indent=2, ensure_ascii=False)

# ---- Figure: Extrapolation comparison ----
print(f"\n{'=' * 60}")
print("Generating extrapolation figures")
print("=" * 60)

for curve_name, curve_extrap in all_extrap_results.items():
    curve_info = curves[curve_name]
    t_full = curve_info["t"]
    rf_full = curve_info["Rf"]
    t_mid = (t_full.min() + t_full.max()) / 2

    n_show = 10  # Show N=10 results
    key = f"N={n_show}"
    if key not in curve_extrap:
        key = list(curve_extrap.keys())[0]

    res = curve_extrap[key]
    y_pinn = res["y_pred_full"]
    y_rf = res["y_rf_full"]

    # Sample sparse points
    train_mask = t_full <= t_mid
    np.random.seed(SEED)
    indices = np.random.choice(np.sum(train_mask), min(n_show, np.sum(train_mask)), replace=False)
    t_sparse = t_full[train_mask][indices]
    rf_sparse = rf_full[train_mask][indices]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # PINN
    ax1.plot(t_full, rf_full * 1e4, "k-", lw=1.5, alpha=0.7, label="True")
    ax1.plot(t_full, y_pinn * 1e4, "r-", lw=1.5, label=f"PINN (Test R2={res['pinn_test_R2']:.3f})")
    ax1.scatter(t_sparse, rf_sparse * 1e4, s=40, c="blue", zorder=5, label=f"{n_show} train pts")
    ax1.axvline(x=t_mid, color="gray", linestyle="--", alpha=0.5, label="Train/Test split")
    ax1.set_xlabel("Time"); ax1.set_ylabel("Rf (x1e-4 m2K/W)")
    ax1.set_title("PINN: Pre-trained + Fine-tuned")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # RF
    ax2.plot(t_full, rf_full * 1e4, "k-", lw=1.5, alpha=0.7, label="True")
    ax2.plot(t_full, y_rf * 1e4, "orange", lw=1.5, label=f"RF (Test R2={res['rf_test_R2']:.3f})")
    ax2.scatter(t_sparse, rf_sparse * 1e4, s=40, c="blue", zorder=5, label=f"{n_show} train pts")
    ax2.axvline(x=t_mid, color="gray", linestyle="--", alpha=0.5, label="Train/Test split")
    ax2.set_xlabel("Time"); ax2.set_ylabel("Rf (x1e-4 m2K/W)")
    ax2.set_title("Random Forest: No extrapolation capability")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    fig.suptitle(f"{curve_info['label']}\nExtrapolation: Train on first 50% time, Predict full curve",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / f"extrapolation_{curve_name}.png", dpi=150)
    plt.close(fig)

# ---- Summary bar chart ----
fig, ax = plt.subplots(figsize=(12, 6))
curve_names = list(all_extrap_results.keys())
n_curves = len(curve_names)
x = np.arange(n_curves)
width = 0.25

for i, n_sparse in enumerate([5, 10, 15]):
    pinn_r2s, rf_r2s = [], []
    for cn in curve_names:
        key = f"N={n_sparse}"
        if key in all_extrap_results[cn]:
            pinn_r2s.append(all_extrap_results[cn][key]["pinn_test_R2"])
            rf_r2s.append(all_extrap_results[cn][key]["rf_test_R2"])
        else:
            pinn_r2s.append(np.nan)
            rf_r2s.append(np.nan)

    ax.bar(x + i*width - width, pinn_r2s, width, alpha=0.8,
           color=plt.cm.RdYlGn(0.75), label=f"PINN N={n_sparse}" if i == 0 else f"PINN N={n_sparse}")
    ax.bar(x + i*width, rf_r2s, width, alpha=0.6,
           color=plt.cm.RdYlGn(0.25), label=f"RF N={n_sparse}" if i == 0 else f"RF N={n_sparse}")

ax.axhline(y=0, color="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([c[:25] for c in curve_names], fontsize=8, rotation=20, ha="right")
ax.set_ylabel("Test R2 (extrapolation region)")
ax.set_title("Extrapolation: PINN vs RF (Test R2 on unseen future time)")
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(FIGS_DIR / "extrapolation_summary.png", dpi=150)
plt.close(fig)

print(f"\nExtrapolation figures saved to {FIGS_DIR}")
print("Done.")
