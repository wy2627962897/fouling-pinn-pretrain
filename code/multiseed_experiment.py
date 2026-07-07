"""
P1: Multi-seed statistical experiment on 5 representative curves.
10 random seeds per (curve, N) combination.
Reports mean ± std for PINN, ODE-fit, and RF.
"""

import json, pickle, sys, time, numpy as np, torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from pinn_model import (PINN, ODE_TYPE_IDS, encode_condition_params,
    finetune_pinn, predict_trajectory, evaluate_trajectory,
    estimate_ode_params_from_sparse)
from fouling_models import fit_model_to_curve, ks_simple_analytical, induction_linear_analytical
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path(__file__).parent / "output"
RESULTS_DIR = OUTPUT_DIR / "experiment_results"
P1_DIR = OUTPUT_DIR / "p1_multiseed"
P1_DIR.mkdir(parents=True, exist_ok=True)

N_FT_EPOCHS = 2000
LAMBDA_PHYS = 0.3
N_COLLOC = 150
N_SEEDS = 10
N_SPARSE_LIST = [5, 10, 15]
SEED_BASE = 100  # Start from 100 to avoid overlap with main experiments

# Representative curves covering all 3 domains
REPRESENTATIVE_CURVES = [
    "source_001_fig7",       # In-domain, KS-type, clean
    "source_002_fed",        # OOD-1, crude oil, industrial
    "source_003_run1",       # OOD-2, unmodified CaCO3 (hardest)
    "source_003_coating_l1",  # OOD-2, coating (easiest)
    "source_003_fig11_ref",  # OOD-2, surface treatment
]

print(f"Device: {DEVICE}")
print(f"Seeds: {N_SEEDS}, N levels: {N_SPARSE_LIST}")
print(f"Curves: {len(REPRESENTATIVE_CURVES)}")

# Load data
with open(OUTPUT_DIR / "pretrain_data" / "real_curves.pkl", "rb") as f:
    real_curves = pickle.load(f)

pretrained_state = torch.load(RESULTS_DIR / "pretrained_model.pt", weights_only=True)

SRC2_CONV = 0.001

def get_matched_ode_fit_name(ode_name):
    if ode_name in ("ks_simple", "kern_seaton", "kern_seaton_offset"):
        return "ks_simple"
    return "induction_linear"

def predict_ode(t_test, fit_name, params):
    if fit_name == "ks_simple":
        return ks_simple_analytical(t_test, params.get("Rf_star", 0),
                                     params.get("tau", 40), params.get("Rf0", 0))
    else:
        return induction_linear_analytical(t_test, params.get("t_ind", 10),
                                            params.get("k_growth", 1e-6),
                                            params.get("Rf_max", 1e-3))

all_stats = {}

for curve_name in REPRESENTATIVE_CURVES:
    curve_info = real_curves[curve_name]
    t_full = curve_info["t"]
    rf_full = curve_info["Rf"]
    if "source_002" in curve_name:
        rf_full = rf_full * SRC2_CONV

    ode_name = curve_info["ode"]
    fit_name = get_matched_ode_fit_name(ode_name)
    label = curve_info.get("label", curve_name)

    print(f"\n{'=' * 60}")
    print(f"Curve: {curve_name} ({label})")
    print(f"  n={len(t_full)}, ode={ode_name}, fit_model={fit_name}")

    curve_stats = {}

    for n_sparse in N_SPARSE_LIST:
        if n_sparse >= len(t_full):
            continue

        pinn_r2s, ode_r2s, rf_r2s = [], [], []

        for seed_idx in range(N_SEEDS):
            seed = SEED_BASE + seed_idx
            np.random.seed(seed)
            torch.manual_seed(seed)

            indices = np.random.choice(len(t_full), n_sparse, replace=False)
            indices.sort()
            t_sparse = t_full[indices]
            rf_sparse = rf_full[indices]

            # ── PINN ──
            model = PINN(hidden_layers=(64, 64, 64, 64)).to(DEVICE)
            model.load_state_dict(pretrained_state)
            ode_params_est = estimate_ode_params_from_sparse(t_sparse, rf_sparse, ode_name)
            t_range = (float(t_full.min()), float(t_full.max()))

            finetune_pinn(model, t_sparse, rf_sparse, ode_name, ode_params_est, t_range,
                          n_epochs=N_FT_EPOCHS, lambda_phys=LAMBDA_PHYS, lr=1e-4,
                          n_colloc=N_COLLOC, device=DEVICE, verbose=False)

            y_pred_pinn = predict_trajectory(model, t_full, ode_name, ode_params_est, DEVICE)
            pinn_r2s.append(r2_score(rf_full, y_pred_pinn))

            # ── ODE curve_fit ──
            fit_result = fit_model_to_curve(t_sparse, rf_sparse, fit_name)
            if fit_result.get("success"):
                y_ode = predict_ode(t_full, fit_name, fit_result["params"])
                ode_r2s.append(r2_score(rf_full, y_ode))
            else:
                ode_r2s.append(float('nan'))

            # ── RF ──
            rf = RandomForestRegressor(n_estimators=100, random_state=seed)
            rf.fit(t_sparse.reshape(-1, 1), rf_sparse)
            y_rf = rf.predict(t_full.reshape(-1, 1))
            rf_r2s.append(r2_score(rf_full, y_rf))

        pinn_r2s = np.array(pinn_r2s)
        ode_r2s = np.array(ode_r2s)
        rf_r2s = np.array(rf_r2s)

        key = f"N={n_sparse}"
        curve_stats[key] = {
            "PINN": {"mean": float(np.mean(pinn_r2s)), "std": float(np.std(pinn_r2s)),
                     "min": float(np.min(pinn_r2s)), "max": float(np.max(pinn_r2s))},
            "ODE_fit": {"mean": float(np.nanmean(ode_r2s)), "std": float(np.nanstd(ode_r2s)),
                        "min": float(np.nanmin(ode_r2s)), "max": float(np.nanmax(ode_r2s))},
            "RF": {"mean": float(np.mean(rf_r2s)), "std": float(np.std(rf_r2s)),
                   "min": float(np.min(rf_r2s)), "max": float(np.max(rf_r2s))},
        }

        print(f"  N={n_sparse:>2} (10 seeds): "
              f"PINN={np.mean(pinn_r2s):.4f}±{np.std(pinn_r2s):.4f} | "
              f"ODE-fit={np.nanmean(ode_r2s):.4f}±{np.nanstd(ode_r2s):.4f} | "
              f"RF={np.mean(rf_r2s):.4f}±{np.std(rf_r2s):.4f}")

        # Best method by mean
        best = "PINN" if np.mean(pinn_r2s) >= max(np.nanmean(ode_r2s), np.mean(rf_r2s)) else \
               ("ODE-fit" if np.nanmean(ode_r2s) >= np.mean(rf_r2s) else "RF")
        print(f"    Winner: {best}")

    all_stats[curve_name] = curve_stats

# Save
with open(P1_DIR / "multiseed_stats.json", "w") as f:
    json.dump(all_stats, f, indent=2, ensure_ascii=False)

# ── Summary table ──
print(f"\n{'=' * 70}")
print("P1 Summary: Mean R² ± Std over 10 seeds (N=15)")
print("=" * 70)
print(f"{'Curve':<35s} {'PINN':>18s} {'ODE-fit':>18s} {'RF':>18s} {'Winner':>10s}")
print("-" * 100)

for cn in REPRESENTATIVE_CURVES:
    key = "N=15"
    if key not in all_stats[cn]:
        continue
    p = all_stats[cn][key]["PINN"]
    o = all_stats[cn][key]["ODE_fit"]
    r = all_stats[cn][key]["RF"]
    best = "PINN" if p["mean"] >= max(o["mean"], r["mean"]) else \
           ("ODE-fit" if o["mean"] >= r["mean"] else "RF")
    pin_str = f"{p['mean']:.4f}±{p['std']:.4f}"
    ode_str = f"{o['mean']:.4f}±{o['std']:.4f}"
    rf_str = f"{r['mean']:.4f}±{r['std']:.4f}"
    cn_short = cn[:33]
    print(f"{cn_short:<35s} {pin_str:>18s} {ode_str:>18s} {rf_str:>18s} {best:>10s}")

# ── Variability analysis ──
print(f"\n{'=' * 70}")
print("Variability Analysis: Std as % of |Mean|")
print("=" * 70)
for cn in REPRESENTATIVE_CURVES:
    for n in N_SPARSE_LIST:
        key = f"N={n}"
        if key not in all_stats[cn]:
            continue
        p = all_stats[cn][key]["PINN"]
        cv = p["std"] / max(abs(p["mean"]), 1e-6) * 100
        if cv > 20:
            flag = " ⚠ HIGH VARIANCE"
        elif cv > 10:
            flag = " ⚡ moderate"
        else:
            flag = ""
        print(f"  {cn[:30]:<32s} N={n}: PINN CV={cv:.0f}% (R²={p['mean']:.3f}±{p['std']:.3f}){flag}")

print(f"\nResults saved to {P1_DIR}")
print("Done.")
