"""
Auto ODE selection ablation experiment.
Quantifies: (1) accuracy of N-point-only auto-selection via geometric heuristics,
           (2) impact of wrong ODE on PINN fine-tuning performance.

The auto-selector chooses between ks_simple and induction_linear based on:
  - Ratio of mean Rf in the first 1/3 vs last 1/3 of sparse observations
  - If ratio < 0.15 (early is near zero, late rises) → induction_linear
  - Otherwise → ks_simple

This mirrors what a human can observe from N sparse (t, Rf) pairs without
seeing the full curve. It operates ONLY on the N sparse points.

Output files:
  output/auto_ode_selection_results.csv
  output/auto_ode_selection_summary.json
"""
import sys, pickle, numpy as np, torch, json, csv
from pathlib import Path
sys.path.insert(0, '.')
from pinn_model import *
from sklearn.metrics import r2_score

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_SPARSE = 15
N_SEEDS = 3
SRC2 = 0.001

OUTPUT_DIR = Path('output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
with open('output/pretrain_data/real_curves.pkl', 'rb') as f:
    curves = pickle.load(f)
pretrained = torch.load('output/experiment_results/pretrained_model.pt', weights_only=True)

CURVES = sorted(curves.keys())
SEEDS = list(range(200, 200 + N_SEEDS))


def auto_select_ode(t_sparse, Rf_sparse):
    """
    Choose between ks_simple and induction_linear using ONLY the N sparse points.

    Heuristic: compute mean Rf in the first 1/3 vs last 1/3 of sparse points.
    If the early segment is near zero (flat induction period) and the late
    segment shows significant growth, select induction_linear.
    Otherwise select ks_simple (asymptotic behavior).

    Returns: (selected_ode_name, early_ratio, late_mean)
    """
    n = len(t_sparse)
    if n < 6:
        return 'ks_simple', 0.0, 0.0

    third = max(n // 3, 2)
    early_mean = float(np.mean(Rf_sparse[:third]))
    late_mean = float(np.mean(Rf_sparse[-third:]))
    late_range = float(np.max(Rf_sparse[-third:]) - np.min(Rf_sparse[-third:]))
    total_range = float(np.max(Rf_sparse) - np.min(Rf_sparse))

    # Heuristic criteria for induction period:
    # 1. Early segment is flat (near zero relative to late growth)
    # 2. Late segment shows significant growth
    if late_mean > 0:
        ratio = early_mean / late_mean
    else:
        ratio = 1.0  # Can't distinguish, default to ks_simple

    # Detect induction: early is near-zero, late has substantial growth
    if ratio < 0.15 and late_range > 1e-6:
        return 'induction_linear', ratio, late_mean
    else:
        return 'ks_simple', ratio, late_mean


results = []

print("=" * 90)
print("ODE Auto-Selection Ablation (geometric heuristic, N-point-only)")
print(f"  N_sparse = {N_SPARSE}, seeds = {SEEDS}")
print(f"  {len(CURVES)} curves x {N_SEEDS} seeds = {len(CURVES) * N_SEEDS} trials")
print(f"  Heuristic: early/late Rf ratio < 0.15 → induction_linear, else ks_simple")
print("=" * 90)

for cn in CURVES:
    ci = curves[cn]
    t, rf = ci['t'], ci['Rf']
    if 'source_002' in cn:
        rf = rf * SRC2
    correct_ode = ci['ode']

    for seed in SEEDS:
        np.random.seed(seed)
        torch.manual_seed(seed)
        idx = np.random.choice(len(t), N_SPARSE, replace=False)
        idx.sort()
        t_s, rf_s = t[idx], rf[idx]

        # ── Step 1: Auto-select ODE from sparse points ──
        best_ode, early_ratio, late_mean = auto_select_ode(t_s, rf_s)
        auto_selection_correct = (best_ode == correct_ode)

        # ── Step 2: Fine-tune PINN with CORRECT ODE ──
        r2_correct = None
        try:
            m = PINN(hidden_layers=(64, 64, 64, 64)).to(DEVICE)
            m.load_state_dict(pretrained)
            ep_correct = estimate_ode_params_from_sparse(t_s, rf_s, correct_ode)
            finetune_pinn(m, t_s, rf_s, correct_ode, ep_correct,
                          (float(t.min()), float(t.max())),
                          n_epochs=2000, lambda_phys=0.3, lr=1e-4, n_colloc=150,
                          device=DEVICE, verbose=False)
            yp_correct = predict_trajectory(m, t, correct_ode, ep_correct, DEVICE)
            r2_correct = r2_score(rf, yp_correct)
        except Exception as e:
            r2_correct = None

        # ── Step 3: Fine-tune PINN with AUTO-SELECTED ODE ──
        r2_auto = None
        try:
            m2 = PINN(hidden_layers=(64, 64, 64, 64)).to(DEVICE)
            m2.load_state_dict(pretrained)
            ep_auto = estimate_ode_params_from_sparse(t_s, rf_s, best_ode)
            finetune_pinn(m2, t_s, rf_s, best_ode, ep_auto,
                          (float(t.min()), float(t.max())),
                          n_epochs=2000, lambda_phys=0.3, lr=1e-4, n_colloc=150,
                          device=DEVICE, verbose=False)
            yp_auto = predict_trajectory(m2, t, best_ode, ep_auto, DEVICE)
            r2_auto = r2_score(rf, yp_auto)
        except Exception as e:
            r2_auto = None

        delta_r2 = (r2_correct - r2_auto) if (r2_correct is not None and r2_auto is not None) else None
        within_002 = (abs(delta_r2) <= 0.02) if delta_r2 is not None else None

        row = {
            'curve': cn,
            'seed': seed,
            'correct_ode': correct_ode,
            'auto_selected_ode': best_ode,
            'auto_selection_correct': auto_selection_correct,
            'early_ratio': float(early_ratio),
            'r2_correct_ode': r2_correct,
            'r2_auto_selected': r2_auto,
            'delta_r2': delta_r2,
            'within_0.02': within_002,
        }
        results.append(row)

        flag = '✓' if auto_selection_correct else '✗'
        r2_str = f'ΔR²={delta_r2:+.4f}' if delta_r2 is not None else 'ΔR²=FAIL'
        w02_str = ' within±0.02' if within_002 else ''
        if within_002 is False and delta_r2 is not None:
            w02_str = f' |ΔR²|={abs(delta_r2):.4f}>0.02'
        print(f"  {cn:<30s} seed={seed}  correct={correct_ode:<18s} "
              f"auto={best_ode:<18s} {flag}  {r2_str}{w02_str}")

# ── Save CSV ──
csv_path = OUTPUT_DIR / 'auto_ode_selection_results.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'curve', 'seed', 'correct_ode', 'auto_selected_ode',
        'auto_selection_correct', 'early_ratio', 'r2_correct_ode',
        'r2_auto_selected', 'delta_r2', 'within_0.02',
    ])
    writer.writeheader()
    for row in results:
        writer.writerow({k: row[k] for k in writer.fieldnames})
print(f"\nCSV saved to {csv_path}")

# ── Summary statistics ──
n_correct = sum(1 for r in results if r['auto_selection_correct'])
n_total = len(results)
accuracy = n_correct / n_total if n_total > 0 else 0

# Per-curve aggregation
curve_summary = {}
for r in results:
    cn = r['curve']
    if cn not in curve_summary:
        curve_summary[cn] = {
            'correct_ode': r['correct_ode'],
            'n_seeds': 0,
            'n_auto_correct': 0,
            'r2_correct_values': [],
            'r2_auto_values': [],
            'delta_r2_values': [],
            'all_within_002': True,
            'max_abs_delta_r2': 0.0,
        }
    cs = curve_summary[cn]
    cs['n_seeds'] += 1
    if r['auto_selection_correct']:
        cs['n_auto_correct'] += 1
    if r['r2_correct_ode'] is not None:
        cs['r2_correct_values'].append(r['r2_correct_ode'])
    if r['r2_auto_selected'] is not None:
        cs['r2_auto_values'].append(r['r2_auto_selected'])
    if r['delta_r2'] is not None:
        cs['delta_r2_values'].append(r['delta_r2'])
        cs['max_abs_delta_r2'] = max(cs['max_abs_delta_r2'], abs(r['delta_r2']))
        if not r['within_0.02']:
            cs['all_within_002'] = False

n_within_002 = sum(1 for cs in curve_summary.values() if cs['all_within_002'])
exception_curves = [cn for cn, cs in curve_summary.items() if not cs['all_within_002']]

summary = {
    'description': 'Auto ODE selection via geometric heuristic (N-point-only, no full-curve info)',
    'heuristic': 'early/late Rf ratio < 0.15 → induction_linear, else ks_simple',
    'n_sparse': N_SPARSE,
    'n_seeds': N_SEEDS,
    'n_curves': len(CURVES),
    'n_trials': n_total,
    'auto_selection_accuracy': f"{accuracy:.1%} ({n_correct}/{n_total})",
    'n_curves_within_0.02': n_within_002,
    'n_exception_curves': len(exception_curves),
    'exception_curves': exception_curves,
    'per_curve': {},
}

for cn, cs in curve_summary.items():
    mean_delta = np.mean(cs['delta_r2_values']) if cs['delta_r2_values'] else None
    mean_r2_correct = np.mean(cs['r2_correct_values']) if cs['r2_correct_values'] else None
    mean_r2_auto = np.mean(cs['r2_auto_values']) if cs['r2_auto_values'] else None

    summary['per_curve'][cn] = {
        'correct_ode': cs['correct_ode'],
        'auto_accuracy': f"{cs['n_auto_correct']}/{cs['n_seeds']}",
        'mean_r2_correct_ode': round(mean_r2_correct, 4) if mean_r2_correct is not None else None,
        'mean_r2_auto_selected': round(mean_r2_auto, 4) if mean_r2_auto is not None else None,
        'mean_delta_r2': round(mean_delta, 4) if mean_delta is not None else None,
        'max_abs_delta_r2': round(cs['max_abs_delta_r2'], 4),
        'all_within_0.02': cs['all_within_002'],
    }

json_path = OUTPUT_DIR / 'auto_ode_selection_summary.json'
with open(json_path, 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"JSON summary saved to {json_path}")

# ── Print summary ──
print(f"\n{'=' * 90}")
print(f"SUMMARY")
print(f"  Auto-selection accuracy: {accuracy:.1%} ({n_correct}/{n_total})")
print(f"  Curves with |ΔR²| ≤ 0.02 on ALL seeds: {n_within_002}/{len(CURVES)}")
if exception_curves:
    print(f"  Exception curves (|ΔR²| > 0.02 on ≥1 seed):")
    for cn in exception_curves:
        cs = curve_summary[cn]
        print(f"    {cn}: max|ΔR²| = {cs['max_abs_delta_r2']:.4f}, "
              f"auto accuracy = {cs['n_auto_correct']}/{cs['n_seeds']}, "
              f"mean R² correct = {np.mean(cs['r2_correct_values']):.4f}, "
              f"mean R² auto = {np.mean(cs['r2_auto_values']):.4f}")
print(f"{'=' * 90}")
