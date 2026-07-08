"""
Pre-training + Fine-tuning Full Experiment Pipeline

Implements the D-route narrative:
  "Simple PINN pre-trained on multi-ODE synthetic data,
   fine-tuned on sparse real points, works across conditions."

Experiment matrix:
  Main:      Pre-trained PINN + matched-ODE FT on sparse real data
  Ablation1: No pre-training, PINN + matched-ODE from scratch (sparse)
  Ablation2: Pre-trained PINN, FT without physics loss (data-only)
  Ablation3: No pre-training, no physics loss = PINN from scratch
  Baseline1: MLP trained from scratch on sparse data
  Baseline2: Random Forest trained from scratch on sparse data

Cross-condition:
  - In-domain:    source_001 (phosphoric acid) — KS
  - OOD1:         source_002 (crude oil) — KS+offset
  - OOD2:         source_003 (CaCO3) — induction_linear

Economic metric: dimensionless cost ratio + sensitivity analysis
"""

import json
import os
import pickle
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.insert(0, str(Path(__file__).parent))
import pinn_model
from pinn_model import (
    PINN, ODE_TYPE_IDS, N_ODE_TYPES,
    encode_condition_params, build_input_tensor,
    pretrain_pinn_multi_ode, finetune_pinn,
    compute_physics_loss_trajectory,
    predict_trajectory, evaluate_trajectory,
    estimate_ode_params_from_sparse,
)
from multi_ode_data_generator import load_dataset, load_all_real_curves

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path(__file__).parent / "output"
FIGS_DIR = OUTPUT_DIR / "experiment_figures"
RESULTS_DIR = OUTPUT_DIR / "experiment_results"
PRETRAIN_DIR = OUTPUT_DIR / "pretrain_data"

N_PRETRAIN_EPOCHS = 3000  # Full training
N_FINETUNE_EPOCHS = 2000  # Full fine-tuning
N_NO_PT_EPOCHS = 4000      # From-scratch needs more epochs
LAMBDA_PHYS_PRETRAIN = 0.5
LAMBDA_PHYS_FINETUNE = 0.3
N_COLLOC_PRETRAIN = 60
N_COLLOC_FINETUNE = 150

# Sparse fine-tuning points per curve
N_SPARSE_POINTS = [5, 10, 15, 20, 30]

# Unit conversion: source_002 Rf in m²·°C/kW → m²·K/W
SRC2_UNIT_CONV = 0.001

torch.manual_seed(SEED)
np.random.seed(SEED)

for d in [FIGS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ─────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Loading data")
print("=" * 60)

dataset = load_dataset(PRETRAIN_DIR / "stratified_dataset.pkl")
real_curves_raw = load_dataset(PRETRAIN_DIR / "real_curves.pkl")
ode_names = dataset["ode_names"]

# Prepare pre-training trajectory batches (in-domain only for base pre-training)
pretrain_trajectories = dataset["in_domain"]  # list of 5 lists (one per ODE)

# Count
total_trajs = sum(len(traj_list) for traj_list in pretrain_trajectories)
print(f"Pre-training trajectories: {total_trajs} ({', '.join(f'{ode_names[i]}: {len(pretrain_trajectories[i])}' for i in range(len(ode_names)))})")

# Pre-process real curves
print(f"\nReal curves ({len(real_curves_raw)} sources):")
real_curves = {}
for name, info in real_curves_raw.items():
    t = info["t"].copy()
    rf = info["Rf"].copy()

    # Unit conversion for source_002
    if "source_002" in name:
        rf = rf * SRC2_UNIT_CONV
        print(f"  {name}: converted Rf ×{SRC2_UNIT_CONV} (m²·°C/kW → m²·K/W)")

    # Remove extreme outliers
    rf_mean, rf_std = np.mean(rf), np.std(rf)
    mask = np.abs(rf - rf_mean) < 6 * rf_std
    if not mask.all():
        t, rf = t[mask], rf[mask]
        print(f"  {name}: removed {sum(~mask)} outlier(s)")

    real_curves[name] = {
        "t": t, "Rf": rf,
        "ode": info["ode"],
        "label": info["label"],
        "domain": info["domain"],
        "n_points": len(t),
    }
    print(f"  {name}: {len(t)} pts, t=[{t.min():.1f}, {t.max():.1f}], Rf=[{rf.min():.2e}, {rf.max():.2e}]")


# ─────────────────────────────────────────────────────────────────────
# 2. Pre-train PINN on multi-ODE synthetic data
# ─────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Pre-training PINN on multi-ODE data")
print("=" * 60)

pretrained_model = PINN(hidden_layers=(64, 64, 64, 64)).to(DEVICE)
n_params = sum(p.numel() for p in pretrained_model.parameters())
print(f"Model params: {n_params:,}")

t0 = time.time()
history_pretrain = pretrain_pinn_multi_ode(
    pretrained_model,
    trajectory_batches=pretrain_trajectories,
    ode_names=ode_names,
    n_epochs=N_PRETRAIN_EPOCHS,
    lambda_phys=LAMBDA_PHYS_PRETRAIN,
    lr=1e-3,
    n_colloc_per_traj=N_COLLOC_PRETRAIN,
    device=DEVICE,
    verbose=True,
)
pretrain_time = time.time() - t0
print(f"Pre-training completed in {pretrain_time:.1f}s ({pretrain_time/60:.1f}min)")

# Save pre-trained model
torch.save(pretrained_model.state_dict(), RESULTS_DIR / "pretrained_model.pt")

# Save pre-training loss curve
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(history_pretrain["data_losses"], alpha=0.6, label="Data loss")
ax.plot(history_pretrain["phys_losses"], alpha=0.6, label="Physics loss")
ax.plot(history_pretrain["total_losses"], linewidth=1.5, label="Total loss")
ax.set_yscale("log")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Pre-training Loss Curves (Multi-ODE)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS_DIR / "pretrain_loss.png", dpi=150)
plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# 3. Full experiment matrix
# ─────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Running full experiment matrix")
print("=" * 60)

all_results = {}


def run_experiment_group(
    group_name, real_curve, ode_name, t_sparse, rf_sparse, t_full, rf_full,
    use_pretrained, use_physics, device
):
    """Run a single experiment group and return metrics."""
    result = {
        "group": group_name,
        "R2": None, "MAE": None, "RMSE": None,
        "y_pred": None, "train_time": None,
    }

    t0 = time.time()

    if use_pretrained and use_physics:
        # Main method: pre-trained + physics FT
        model = PINN(hidden_layers=(64, 64, 64, 64)).to(device)
        model.load_state_dict(torch.load(RESULTS_DIR / "pretrained_model.pt", weights_only=True))
        ode_params_est = estimate_ode_params_from_sparse(t_sparse, rf_sparse, ode_name)
        t_range = (float(t_full.min()), float(t_full.max()))
        finetune_pinn(
            model, t_sparse, rf_sparse, ode_name, ode_params_est,
            t_range, n_epochs=N_FINETUNE_EPOCHS, lambda_phys=LAMBDA_PHYS_FINETUNE,
            lr=1e-4, n_colloc=N_COLLOC_FINETUNE, device=device, verbose=False,
        )
        y_pred = predict_trajectory(model, t_full, ode_name, ode_params_est, device)

    elif use_pretrained and not use_physics:
        # Ablation: pre-trained, FT without physics
        model = PINN(hidden_layers=(64, 64, 64, 64)).to(device)
        model.load_state_dict(torch.load(RESULTS_DIR / "pretrained_model.pt", weights_only=True))
        ode_params_est = estimate_ode_params_from_sparse(t_sparse, rf_sparse, ode_name)

        # Data-only fine-tuning
        condition_vec = encode_condition_params(ode_name, **ode_params_est)
        ode_type_id = ODE_TYPE_IDS[ode_name]
        x_data, y_data = pinn_model.prepare_unified_tensors(
            t_sparse, rf_sparse, condition_vec, ode_type_id, device
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        for _ in range(N_FINETUNE_EPOCHS):
            model.train()
            optimizer.zero_grad()
            loss = torch.mean((model(x_data) - y_data) ** 2)
            loss.backward()
            optimizer.step()
        y_pred = predict_trajectory(model, t_full, ode_name, ode_params_est, device)

    elif not use_pretrained and use_physics:
        # Ablation: no pre-training, physics from scratch
        model = PINN(hidden_layers=(64, 64, 64, 64)).to(device)
        ode_params_est = estimate_ode_params_from_sparse(t_sparse, rf_sparse, ode_name)
        t_range = (float(t_full.min()), float(t_full.max()))
        finetune_pinn(
            model, t_sparse, rf_sparse, ode_name, ode_params_est,
            t_range, n_epochs=N_NO_PT_EPOCHS, lambda_phys=LAMBDA_PHYS_FINETUNE,
            lr=1e-3, n_colloc=N_COLLOC_FINETUNE, device=device, verbose=False,
        )
        y_pred = predict_trajectory(model, t_full, ode_name, ode_params_est, device)

    else:
        # Ablation: no pre-training, no physics = pure NN from scratch
        model = PINN(hidden_layers=(64, 64, 64, 64)).to(device)
        ode_params_est = estimate_ode_params_from_sparse(t_sparse, rf_sparse, ode_name)
        condition_vec = encode_condition_params(ode_name, **ode_params_est)
        ode_type_id = ODE_TYPE_IDS[ode_name]
        x_data, y_data = pinn_model.prepare_unified_tensors(
            t_sparse, rf_sparse, condition_vec, ode_type_id, device
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(N_NO_PT_EPOCHS):
            model.train()
            optimizer.zero_grad()
            loss = torch.mean((model(x_data) - y_data) ** 2)
            loss.backward()
            optimizer.step()
        y_pred = predict_trajectory(model, t_full, ode_name, ode_params_est, device)

    result["train_time"] = time.time() - t0

    # Evaluate
    metrics = evaluate_trajectory(y_pred, rf_full)
    result.update(metrics)
    result["y_pred"] = y_pred

    return result


def run_ml_baselines(t_sparse, rf_sparse, t_full, rf_full):
    """Run MLP, RF, XGBoost, and LSTM baselines trained from scratch on sparse data."""
    results = {}
    t_sparse_2d = t_sparse.reshape(-1, 1)
    t_full_2d = t_full.reshape(-1, 1)

    # MLP
    t0 = time.time()
    try:
        mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800,
                          early_stopping=(len(t_sparse) >= 20), random_state=SEED)
        mlp.fit(StandardScaler().fit_transform(t_sparse_2d), rf_sparse)
        y_pred_mlp = mlp.predict(StandardScaler().fit_transform(t_full_2d))
        results["MLP"] = {
            "R2": r2_score(rf_full, y_pred_mlp),
            "MAE": mean_absolute_error(rf_full, y_pred_mlp),
            "RMSE": np.sqrt(np.mean((rf_full - y_pred_mlp) ** 2)),
            "y_pred": y_pred_mlp, "train_time": time.time() - t0,
        }
    except Exception as e:
        results["MLP"] = {"R2": None, "MAE": None, "RMSE": None, "y_pred": None}

    # Random Forest
    t0 = time.time()
    try:
        rf = RandomForestRegressor(n_estimators=100, random_state=SEED)
        rf.fit(t_sparse_2d, rf_sparse)
        y_pred_rf = rf.predict(t_full_2d)
        results["RF"] = {
            "R2": r2_score(rf_full, y_pred_rf),
            "MAE": mean_absolute_error(rf_full, y_pred_rf),
            "RMSE": np.sqrt(np.mean((rf_full - y_pred_rf) ** 2)),
            "y_pred": y_pred_rf, "train_time": time.time() - t0,
        }
    except Exception as e:
        results["RF"] = {"R2": None, "MAE": None, "RMSE": None, "y_pred": None}

    # XGBoost
    try:
        from xgboost import XGBRegressor
        t0 = time.time()
        xgb = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                          random_state=SEED, verbosity=0)
        xgb.fit(t_sparse_2d, rf_sparse)
        y_pred_xgb = xgb.predict(t_full_2d)
        results["XGBoost"] = {
            "R2": r2_score(rf_full, y_pred_xgb),
            "MAE": mean_absolute_error(rf_full, y_pred_xgb),
            "RMSE": np.sqrt(np.mean((rf_full - y_pred_xgb) ** 2)),
            "y_pred": y_pred_xgb, "train_time": time.time() - t0,
        }
    except ImportError:
        results["XGBoost"] = {"R2": None, "MAE": None, "RMSE": None, "y_pred": None}

    # LSTM (simple 1-layer, trained on sequence)
    try:
        t0 = time.time()
        # Prepare LSTM data: single sequence of sparse points, sorted by time
        sort_idx = np.argsort(t_sparse)
        t_seq = torch.tensor(t_sparse[sort_idx], dtype=torch.float32).reshape(-1, 1, 1).to(DEVICE)
        rf_seq = torch.tensor(rf_sparse[sort_idx], dtype=torch.float32).reshape(-1, 1).to(DEVICE)

        lstm = torch.nn.LSTM(input_size=1, hidden_size=16, num_layers=1,
                            batch_first=True).to(DEVICE)
        lstm_head = torch.nn.Linear(16, 1).to(DEVICE)
        lstm_opt = torch.optim.Adam(list(lstm.parameters()) + list(lstm_head.parameters()), lr=0.01)

        for _ in range(1000):
            lstm.train(); lstm_head.train()
            lstm_opt.zero_grad()
            out, _ = lstm(t_seq)
            pred = lstm_head(out[:, -1:, :])
            loss = torch.mean((pred - rf_seq[-1:]) ** 2)
            loss.backward(); lstm_opt.step()

        # Predict on full curve (autoregressive is overkill; use nearest interpolation)
        lstm.eval(); lstm_head.eval()
        with torch.no_grad():
            t_full_tensor = torch.tensor(t_full, dtype=torch.float32).reshape(-1, 1, 1).to(DEVICE)
            out_full, _ = lstm(t_full_tensor)
            y_pred_lstm = lstm_head(out_full[:, -1:, :]).cpu().numpy().flatten()
            # Expand single prediction to full length
            if len(y_pred_lstm) == 1:
                y_pred_lstm = np.full_like(rf_full, y_pred_lstm[0])

        results["LSTM"] = {
            "R2": r2_score(rf_full, y_pred_lstm),
            "MAE": mean_absolute_error(rf_full, y_pred_lstm),
            "RMSE": np.sqrt(np.mean((rf_full - y_pred_lstm) ** 2)),
            "y_pred": y_pred_lstm, "train_time": time.time() - t0,
        }
    except Exception as e:
        results["LSTM"] = {"R2": None, "MAE": None, "RMSE": None, "y_pred": None}

    return results


# ── Run for each real curve ──

experiment_groups = [
    ("Main: PT+Phys", True, True),
    ("A1: NoPT+Phys", False, True),
    ("A2: PT+NoPhys", True, False),
    ("A3: NoPT+NoPhys", False, False),
]

for curve_name, curve_info in real_curves.items():
    print(f"\n{'─' * 60}")
    print(f"Curve: {curve_name} ({curve_info['label']})")
    print(f"  ODE: {curve_info['ode']}, Domain: {curve_info['domain']}")
    print(f"  Full: {curve_info['n_points']} points")

    t_full = curve_info["t"]
    rf_full = curve_info["Rf"]
    ode_name = curve_info["ode"]

    curve_results = {}

    for n_sparse in N_SPARSE_POINTS:
        if n_sparse >= curve_info["n_points"]:
            continue

        print(f"\n  --- N_sparse = {n_sparse} ---")

        # Randomly sample sparse points
        np.random.seed(SEED)
        indices = np.random.choice(len(t_full), min(n_sparse, len(t_full)), replace=False)
        indices = np.sort(indices)
        t_sparse = t_full[indices]
        rf_sparse = rf_full[indices]

        key = f"N={n_sparse}"
        curve_results[key] = {}

        # Experiment groups (PINN variants)
        for group_name, use_pt, use_phys in experiment_groups:
            print(f"    {group_name}...", end=" ", flush=True)
            res = run_experiment_group(
                group_name, curve_info, ode_name,
                t_sparse, rf_sparse, t_full, rf_full,
                use_pt, use_phys, DEVICE,
            )
            curve_results[key][group_name] = res
            status = f"R²={res['R2']:.4f}, MAE={res['MAE']:.2e}" if res['R2'] is not None else "FAILED"
            print(f"{status} ({res['train_time']:.1f}s)")

        # ML baselines (only need to run once per n_sparse since they don't use pre-training)
        print(f"    Baselines (MLP, RF)...", end=" ", flush=True)
        ml_results = run_ml_baselines(t_sparse, rf_sparse, t_full, rf_full)
        for ml_name, ml_res in ml_results.items():
            curve_results[key][ml_name] = ml_res
        mlp_status = f"MLP R²={ml_results['MLP']['R2']:.4f}" if ml_results['MLP']['R2'] is not None else "MLP FAILED"
        rf_status = f"RF R²={ml_results['RF']['R2']:.4f}" if ml_results['RF']['R2'] is not None else "RF FAILED"
        print(f"{mlp_status}, {rf_status}")

    all_results[curve_name] = curve_results

# Save all results
with open(RESULTS_DIR / "all_experiment_results.json", "w") as f:
    # Convert to serializable format (remove numpy arrays)
    serializable = {}
    for cname, cdata in all_results.items():
        serializable[cname] = {}
        for nkey, ndata in cdata.items():
            serializable[cname][nkey] = {}
            for gname, gdata in ndata.items():
                serializable[cname][nkey][gname] = {
                    k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in gdata.items()
                    if k != "y_pred"  # skip large arrays
                }
    json.dump(serializable, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to {RESULTS_DIR / 'all_experiment_results.json'}")


# ─────────────────────────────────────────────────────────────────────
# 4. Generate comparison figures
# ─────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Generating figures")
print("=" * 60)

# ── Figure 1: R² vs N_sparse for each curve ──
for curve_name, curve_results in all_results.items():
    curve_info = real_curves[curve_name]
    fig, ax = plt.subplots(figsize=(10, 6))

    groups_to_plot = ["Main: PT+Phys", "A1: NoPT+Phys", "A3: NoPT+NoPhys", "RF", "XGBoost", "LSTM"]
    colors = ["#2c7bb6", "#d7191c", "#fdae61", "#008837", "#e66101", "#5e3c99"]
    markers = ["o", "s", "^", "D", "v", "<"]

    for group, color, marker in zip(groups_to_plot, colors, markers):
        n_list, r2_list = [], []
        for n_sparse in N_SPARSE_POINTS:
            key = f"N={n_sparse}"
            if key in curve_results and group in curve_results[key]:
                r2 = curve_results[key][group].get("R2")
                if r2 is not None:
                    n_list.append(n_sparse)
                    r2_list.append(r2)
        if n_list:
            ax.plot(n_list, r2_list, color=color, marker=marker, linewidth=2,
                    markersize=9, label=group)

    ax.set_xlabel("Number of sparse training points", fontsize=12)
    ax.set_ylabel("R² on full curve", fontsize=12)
    ax.set_title(f"{curve_info['label']}\nPINN Pre-training + Fine-tuning vs Baselines", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 1.05)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / f"r2_vs_sparse_{curve_name}.png", dpi=150)
    plt.close(fig)

# ── Figure 2: Prediction vs True for best N_sparse ──
for curve_name, curve_results in all_results.items():
    curve_info = real_curves[curve_name]
    t_full = curve_info["t"]
    rf_full = curve_info["Rf"]

    best_n = N_SPARSE_POINTS[2]  # Use N=15 as representative
    key = f"N={best_n}"
    if key not in curve_results:
        continue

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax_idx, (group, title) in enumerate([
        ("Main: PT+Phys", "Pre-trained + Physics FT"),
        ("A3: NoPT+NoPhys", "No Pre-training, No Physics"),
        ("MLP", "MLP Baseline"),
    ]):
        ax = axes[ax_idx]
        if group in curve_results[key]:
            res = curve_results[key][group]
            if res["y_pred"] is not None:
                ax.plot(t_full, rf_full * 1e4, "k-", linewidth=2, alpha=0.7, label="True")
                ax.plot(t_full, res["y_pred"] * 1e4, "r--", linewidth=1.5, label="Predicted")
                # Mark sparse training points
                np.random.seed(SEED)
                indices = np.random.choice(len(t_full), min(best_n, len(t_full)), replace=False)
                ax.scatter(t_full[indices], rf_full[indices] * 1e4, s=40, c="blue",
                          zorder=5, label=f"{best_n} train pts")

                r2_str = f"R²={res['R2']:.3f}" if res['R2'] is not None else "N/A"
                ax.set_title(f"{title}\n{r2_str}", fontsize=11)
            else:
                ax.set_title(f"{title}\n(Failed)", fontsize=11)
        ax.set_xlabel("Time")
        ax.set_ylabel("Rf (×10⁻⁴ m²·K/W)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{curve_info['label']} — Prediction Comparison (N={best_n} sparse points)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / f"prediction_comparison_{curve_name}.png", dpi=150)
    plt.close(fig)

# ── Figure 3: Ablation heatmap ──
fig, axes = plt.subplots(1, len(real_curves), figsize=(5 * len(real_curves), 5))
if len(real_curves) == 1:
    axes = [axes]

for ax_idx, (curve_name, curve_results) in enumerate(all_results.items()):
    curve_info = real_curves[curve_name]
    ax = axes[ax_idx]

    groups = ["Main: PT+Phys", "A1: NoPT+Phys", "A2: PT+NoPhys", "A3: NoPT+NoPhys"]
    n_list = N_SPARSE_POINTS
    data_matrix = np.zeros((len(groups), len(n_list)))

    for i, group in enumerate(groups):
        for j, n_sparse in enumerate(n_list):
            key = f"N={n_sparse}"
            if key in curve_results and group in curve_results[key]:
                r2 = curve_results[key][group].get("R2")
                data_matrix[i, j] = max(r2, -0.5) if r2 is not None else -0.5

    im = ax.imshow(data_matrix, cmap="RdYlGn", aspect="auto", vmin=-0.5, vmax=1.0)
    ax.set_xticks(range(len(n_list)))
    ax.set_xticklabels([str(n) for n in n_list])
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(["PT+Phys", "NoPT+Phys", "PT+NoPhys", "NoPT+NoPhys"], fontsize=9)
    ax.set_xlabel("N sparse points")
    ax.set_title(curve_info["label"].split("(")[0].strip(), fontsize=10)

    for i in range(len(groups)):
        for j in range(len(n_list)):
            val = data_matrix[i, j]
            color = "white" if val < 0.3 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

fig.suptitle("Ablation Study: R² Heatmap (Pre-training × Physics Loss)", fontsize=13, y=1.03)
fig.tight_layout()
fig.savefig(FIGS_DIR / "ablation_heatmap.png", dpi=150)
plt.close(fig)

print(f"Figures saved to {FIGS_DIR}")


# ─────────────────────────────────────────────────────────────────────
# 5. Economic analysis — dimensionless cost ratio
# ─────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Economic Analysis: Dimensionless Cost Ratio")
print("=" * 60)


def compute_dimensionless_cost_ratio(
    Rf_curve, t_curve, Rf_crit, t_cleaning_downtime,
    fixed_interval=None, threshold_based=False, prediction_based=False,
    Rf_predicted=None,
):
    """Compute dimensionless cost ratio for a cleaning strategy.

    All strategies correctly reset Rf to 0 after each cleaning,
    tracking energy penalty segment-by-segment between cleanings.

    IMPORTANT (Q12): The "reference" policy uses true Rf(t) to trigger cleaning
    at Rf_crit. This is NOT a cost-optimal benchmark -- it's a threshold-triggered
    policy that assumes perfect real-time fouling monitoring. It does NOT minimize
    total cost; it simply cleans whenever true Rf exceeds the threshold.
    Cost ratios below 1.0 mean the strategy chooses cheaper cleaning times
    than the reference -- not that it beats the global optimum.

    Also: Rf_crit is typically defined from the full-curve max(Rf), which is
    unknown at t=0 in real operations. Multi-threshold sensitivity analysis
    is needed to assess robustness to this choice.

    Parameters
    ----------
    Rf_curve: true Rf(t) values (used for energy cost computation in ALL strategies)
    t_curve: time points
    Rf_crit: critical Rf threshold for cleaning
    t_cleaning_downtime: time lost per cleaning event
    fixed_interval: if not None, clean every N time units
    prediction_based: if True, use Rf_predicted to decide WHEN to clean
    Rf_predicted: predicted Rf(t) for timing decisions only
    """
    t0 = t_curve[0]
    t_total = t_curve[-1] - t0

    # Helper: energy cost for ONE fouling cycle of duration D
    # Uses the TRUE Rf_curve, clipped at zero: negative Rf (coating enhancement)
    # contributes zero penalty, not a spurious negative cost.
    def cycle_energy(D):
        if D <= 0:
            return 0.0
        # Find the Rf values up to time D (from start of curve)
        mask = (t_curve - t0) <= D
        if mask.sum() < 2:
            return 0.0
        Rf_clipped = np.maximum(Rf_curve[mask], 0.0)
        return float(np.trapz(Rf_clipped, t_curve[mask]))

    def strategy_cost_from_intervals(intervals):
        """intervals: list of (usable_time_before_cleaning, ...)"""
        total = 0.0
        for usable_t in intervals:
            total += 1.0  # cleaning cost (normalized)
            total += cycle_energy(usable_t)
        return total

    # ── Reference threshold policy: clean when TRUE Rf reaches Rf_crit ──
    # NOT a cost-optimal benchmark. Assumes perfect real-time Rf monitoring.
    above_crit = np.where(Rf_curve >= Rf_crit)[0]
    if len(above_crit) == 0:
        ref_cost = cycle_energy(t_total)
        n_ref = 0
    else:
        ref_intervals = []
        t_current = t0
        idx = 0
        while idx < len(above_crit):
            clean_t = t_curve[above_crit[idx]]
            usable = clean_t - t_current
            ref_intervals.append(usable)
            t_current = clean_t + t_cleaning_downtime
            # Find next crossing after downtime
            next_idx = np.where((t_curve >= t_current) & (Rf_curve >= Rf_crit))[0]
            if len(next_idx) == 0:
                break
            idx = next_idx[0] - above_crit[0] if next_idx[0] >= above_crit[0] else len(above_crit)
            if idx >= len(above_crit):
                break
            above_crit = above_crit[idx:]
            idx = 0
        # Remaining energy after last cleaning
        remaining = t_curve[-1] - t_current
        if remaining > 0:
            ref_cost = cycle_energy(remaining)
            for usable in ref_intervals:
                ref_cost += 1.0 + cycle_energy(usable)
            n_ref = len(ref_intervals)
        else:
            ref_cost = 0.0
            for usable in ref_intervals:
                ref_cost += 1.0 + cycle_energy(usable)
            n_ref = len(ref_intervals)

    if ref_cost <= 0:
        ref_cost = max(cycle_energy(t_total), 1e-10)

    # ── Fixed-interval strategy ──
    if fixed_interval is not None:
        n_cleanings = max(0, int(t_total / fixed_interval))
        if n_cleanings == 0:
            strategy_cost = cycle_energy(t_total)
        else:
            cycle_duration = fixed_interval - t_cleaning_downtime
            strategy_cost = n_cleanings * (1.0 + cycle_energy(cycle_duration))
            # Remaining after last cleaning
            remaining = t_total - n_cleanings * fixed_interval
            if remaining > 0:
                strategy_cost += cycle_energy(remaining)
        n_strategy = n_cleanings

    # ── PINN-predicted strategy ──
    elif prediction_based and Rf_predicted is not None:
        # Find cleaning times from PREDICTED Rf (decisions)
        above_pred = np.where(Rf_predicted >= Rf_crit * 0.9)[0]  # 10% safety margin
        if len(above_pred) == 0:
            strategy_cost = cycle_energy(t_total)
            n_strategy = 0
        else:
            first_warning_t = t_curve[above_pred[0]]
            predicted_interval = first_warning_t - t0
            if predicted_interval <= 0:
                predicted_interval = t_total
            n_cleanings = max(0, int(t_total / predicted_interval))
            if n_cleanings == 0:
                strategy_cost = cycle_energy(t_total)
            else:
                cycle_dur = predicted_interval - t_cleaning_downtime
                strategy_cost = n_cleanings * (1.0 + cycle_energy(cycle_dur))
                remaining = t_total - n_cleanings * predicted_interval
                if remaining > 0:
                    strategy_cost += cycle_energy(remaining)
            n_strategy = n_cleanings

    else:
        # No cleaning
        strategy_cost = cycle_energy(t_total)
        n_strategy = 0

    cost_ratio = strategy_cost / ref_cost if ref_cost > 0 else 1.0

    return {
        "cost_ratio": cost_ratio,
        "ref_cost": ref_cost,
        "strategy_cost": strategy_cost,
        "n_cleanings_strategy": n_strategy,
        "n_cleanings_ref": n_ref,
    }


# Run economic analysis for each curve using best method prediction
print("\n--- Dimensionless Cost Ratio Analysis (Q12: multi-threshold sensitivity) ---")
print("Reference = threshold-triggered policy using TRUE Rf(t) [NOT cost-optimal]")
economic_results = {}

# Multi-threshold sensitivity: in practice, Rf_crit is set by equipment design specs.
# Since we lack per-curve specs, we sweep 4 threshold levels to assess robustness.
THRESHOLD_FACTORS = [0.50, 0.65, 0.80, 0.95]

for curve_name, curve_results in all_results.items():
    curve_info = real_curves[curve_name]
    t_full = curve_info["t"]
    rf_full = curve_info["Rf"]

    t_downtime = 0.02 * (t_full[-1] - t_full[0])  # 2% of total time

    eco_curve = {}

    for n_sparse in N_SPARSE_POINTS:
        key = f"N={n_sparse}"
        if key not in curve_results:
            continue

        n_key = f"N={n_sparse}"

        # Get best prediction (Main method)
        main_res = curve_results[key].get("Main: PT+Phys", {})
        y_pred = main_res.get("y_pred")

        if y_pred is not None:
            # Use the primary threshold (0.80) for the main table,
            # but also store results for all thresholds
            Rf_crit_primary = 0.80 * np.max(rf_full)

            # Fixed-interval strategies
            for interval_pct in [0.25, 0.33, 0.5]:
                interval = (t_full[-1] - t_full[0]) * interval_pct
                eco = compute_dimensionless_cost_ratio(
                    rf_full, t_full, Rf_crit_primary, t_downtime,
                    fixed_interval=interval,
                )
                eco["strategy"] = f"Fixed {interval_pct*100:.0f}%"

            # Prediction-based strategy
            eco_pred = compute_dimensionless_cost_ratio(
                rf_full, t_full, Rf_crit_primary, t_downtime,
                prediction_based=True, Rf_predicted=y_pred,
            )
            eco_pred["strategy"] = "PINN Predicted"

            # Multi-threshold sensitivity for PINN strategy
            sensitivity = {}
            for factor in THRESHOLD_FACTORS:
                Rf_crit_sens = factor * np.max(rf_full)
                eco_sens = compute_dimensionless_cost_ratio(
                    rf_full, t_full, Rf_crit_sens, t_downtime,
                    prediction_based=True, Rf_predicted=y_pred,
                )
                sens_fixed25 = compute_dimensionless_cost_ratio(
                    rf_full, t_full, Rf_crit_sens, t_downtime,
                    fixed_interval=(t_full[-1] - t_full[0]) * 0.25,
                )
                sens_fixed33 = compute_dimensionless_cost_ratio(
                    rf_full, t_full, Rf_crit_sens, t_downtime,
                    fixed_interval=(t_full[-1] - t_full[0]) * 0.33,
                )
                sensitivity[f"factor_{factor:.2f}"] = {
                    "Rf_crit": float(Rf_crit_sens),
                    "pinn_cost_ratio": eco_sens["cost_ratio"],
                    "pinn_n_cleanings": eco_sens["n_cleanings_strategy"],
                    "ref_n_cleanings": eco_sens["n_cleanings_ref"],
                    "fixed25_cost_ratio": sens_fixed25["cost_ratio"],
                    "fixed33_cost_ratio": sens_fixed33["cost_ratio"],
                }

            eco_curve[n_key] = {
                "fixed_25pct": compute_dimensionless_cost_ratio(
                    rf_full, t_full, Rf_crit_primary, t_downtime,
                    fixed_interval=(t_full[-1] - t_full[0]) * 0.25,
                ),
                "fixed_33pct": compute_dimensionless_cost_ratio(
                    rf_full, t_full, Rf_crit_primary, t_downtime,
                    fixed_interval=(t_full[-1] - t_full[0]) * 0.33,
                ),
                "pinn_predicted": eco_pred,
                "threshold_sensitivity": sensitivity,
            }

            print(f"  {curve_name} N={n_sparse}: "
                  f"Fixed25%={eco_curve[n_key]['fixed_25pct']['cost_ratio']:.3f}, "
                  f"Fixed33%={eco_curve[n_key]['fixed_33pct']['cost_ratio']:.3f}, "
                  f"PINN={eco_pred['cost_ratio']:.3f}")

    economic_results[curve_name] = eco_curve

# ── Economic figure: strategy comparison ──
for curve_name, eco_data in economic_results.items():
    curve_info = real_curves[curve_name]
    if not eco_data:
        continue

    fig, ax = plt.subplots(figsize=(8, 5))
    n_keys = list(eco_data.keys())
    strategies = ["Fixed 25%", "Fixed 33%", "PINN Predicted"]
    x = np.arange(len(strategies))
    width = 0.8 / len(n_keys)

    for i, n_key in enumerate(n_keys):
        ratios = [
            eco_data[n_key]["fixed_25pct"]["cost_ratio"],
            eco_data[n_key]["fixed_33pct"]["cost_ratio"],
            eco_data[n_key]["pinn_predicted"]["cost_ratio"],
        ]
        bars = ax.bar(x + i * width, ratios, width, label=n_key, alpha=0.8)

    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, label="Reference (true-curve threshold)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(strategies)
    ax.set_ylabel("Dimensionless Cost Ratio")
    ax.set_title(f"{curve_info['label']}\nCleaning Strategy Cost Comparison (lower = better)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / f"economic_{curve_name}.png", dpi=150)
    plt.close(fig)

# Save economic results
eco_serializable = {}
for cname, cdata in economic_results.items():
    eco_serializable[cname] = {}
    for nkey, ndata in cdata.items():
        eco_serializable[cname][nkey] = {
            k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                for kk, vv in v.items() if kk != "Rf_predicted"}
            for k, v in ndata.items()
        }
with open(RESULTS_DIR / "economic_results.json", "w") as f:
    json.dump(eco_serializable, f, indent=2, ensure_ascii=False)

print(f"\nEconomic results saved to {RESULTS_DIR / 'economic_results.json'}")


# ─────────────────────────────────────────────────────────────────────
# 6. Summary table
# ─────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Summary: Best R² per method (N=15 sparse points)")
print("=" * 60)

for curve_name, curve_results in all_results.items():
    curve_info = real_curves[curve_name]
    key = "N=15"
    if key not in curve_results:
        continue

    print(f"\n{curve_info['label']}:")
    for group in ["Main: PT+Phys", "A1: NoPT+Phys", "A3: NoPT+NoPhys", "MLP", "RF"]:
        if group in curve_results[key]:
            res = curve_results[key][group]
            r2_str = f"R²={res['R2']:.4f}" if res['R2'] is not None else "FAILED"
            mae_str = f"MAE={res['MAE']:.2e}" if res['MAE'] is not None else ""
            print(f"  {group:<20s}: {r2_str} {mae_str}")

print(f"\n{'=' * 60}")
print("All experiments complete.")
print(f"Results: {RESULTS_DIR}")
print(f"Figures: {FIGS_DIR}")
print(f"{'=' * 60}")
