"""
Supplementary baselines: Spline interpolation and Gaussian Process regression
for sparse fouling curve reconstruction.

Tests: Cubic Spline, Smoothing Spline, Gaussian Process (RBF kernel)
Across N = 5, 10, 15, 20, 30 on all 18 curves (3 seeds each).

Output: output/baseline_spline_gp_results.csv, output/baseline_spline_gp_summary.json
"""
import sys, pickle, numpy as np, json, csv
from pathlib import Path
from scipy.interpolate import CubicSpline, UnivariateSpline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '.')

SRC2 = 0.001
N_POINTS_LIST = [5, 10, 15, 20, 30]
N_SEEDS = 3
SEEDS = [200, 201, 202]

OUTPUT_DIR = Path('output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open('output/pretrain_data/real_curves.pkl', 'rb') as f:
    curves = pickle.load(f)

CURVES = sorted(curves.keys())

results = []

print("=" * 90)
print("Spline & GP Baselines for Sparse Fouling Curve Reconstruction")
print(f"  Curves: {len(CURVES)}, N values: {N_POINTS_LIST}, Seeds: {SEEDS}")
print("=" * 90)

for cn in CURVES:
    ci = curves[cn]
    t_full, rf_full = ci['t'], ci['Rf']
    if 'source_002' in cn:
        rf_full = rf_full * SRC2
    n_full = len(t_full)
    ode_name = ci['ode']

    for n_sparse in N_POINTS_LIST:
        if n_sparse >= n_full:
            continue

        for seed in SEEDS:
            np.random.seed(seed)
            idx = np.random.choice(n_full, n_sparse, replace=False)
            idx.sort()
            t_s, rf_s = t_full[idx], rf_full[idx]

            row = {
                'curve': cn, 'ode': ode_name, 'n_sparse': n_sparse,
                'seed': seed, 'n_full': n_full,
            }

            t_flat = t_s.reshape(-1, 1)
            rf_flat = rf_s.reshape(-1, 1)
            t_pred = t_full.reshape(-1, 1)

            # ── Cubic Spline ──
            try:
                cs = CubicSpline(t_s, rf_s, extrapolate=True)
                y_cs = cs(t_full)
                row['cubic_spline_r2'] = float(r2_score(rf_full, y_cs))
            except Exception:
                row['cubic_spline_r2'] = None

            # ── Smoothing Spline ──
            try:
                # s = smoothing factor; None = automatic selection
                ss = UnivariateSpline(t_s, rf_s, s=None, ext='const')
                y_ss = ss(t_full)
                row['smoothing_spline_r2'] = float(r2_score(rf_full, y_ss))
            except Exception:
                row['smoothing_spline_r2'] = None

            # ── Gaussian Process (RBF kernel) ──
            try:
                scaler_t = StandardScaler()
                scaler_rf = StandardScaler()
                t_scaled = scaler_t.fit_transform(t_flat)
                rf_scaled = scaler_rf.fit_transform(rf_flat).ravel()

                kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
                gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                              normalize_y=True, n_restarts_optimizer=5,
                                              random_state=seed)
                gp.fit(t_scaled, rf_scaled)
                t_pred_scaled = scaler_t.transform(t_pred)
                y_gp_scaled = gp.predict(t_pred_scaled)
                y_gp = scaler_rf.inverse_transform(y_gp_scaled.reshape(-1, 1)).ravel()
                row['gp_rbf_r2'] = float(r2_score(rf_full, y_gp))
            except Exception:
                row['gp_rbf_r2'] = None

            # ── Gaussian Process (Matern kernel) ──
            try:
                scaler_t2 = StandardScaler()
                scaler_rf2 = StandardScaler()
                t_scaled2 = scaler_t2.fit_transform(t_flat)
                rf_scaled2 = scaler_rf2.fit_transform(rf_flat).ravel()

                kernel2 = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
                gp2 = GaussianProcessRegressor(kernel=kernel2, alpha=1e-6,
                                               normalize_y=True, n_restarts_optimizer=5,
                                               random_state=seed)
                gp2.fit(t_scaled2, rf_scaled2)
                t_pred_scaled2 = scaler_t2.transform(t_pred)
                y_gp2_scaled = gp2.predict(t_pred_scaled2)
                y_gp2 = scaler_rf2.inverse_transform(y_gp2_scaled.reshape(-1, 1)).ravel()
                row['gp_matern_r2'] = float(r2_score(rf_full, y_gp2))
            except Exception:
                row['gp_matern_r2'] = None

            results.append(row)

            vals = []
            for k in ['cubic_spline_r2', 'smoothing_spline_r2', 'gp_rbf_r2', 'gp_matern_r2']:
                v = row.get(k)
                vals.append(f"{v:.4f}" if v is not None else "FAIL")
            print(f"  {cn:<30s} N={n_sparse} s={seed}  "
                  f"CS={vals[0]}  SS={vals[1]}  GP-RBF={vals[2]}  GP-Mat={vals[3]}")

# ── Save CSV ──
csv_path = OUTPUT_DIR / 'baseline_spline_gp_results.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'curve', 'ode', 'n_sparse', 'seed', 'n_full',
        'cubic_spline_r2', 'smoothing_spline_r2', 'gp_rbf_r2', 'gp_matern_r2',
    ])
    writer.writeheader()
    for row in results:
        writer.writerow(row)
print(f"\nCSV saved to {csv_path}")

# ── Summary: aggregate across seeds and curves ──
summary = {}
for method in ['cubic_spline_r2', 'smoothing_spline_r2', 'gp_rbf_r2', 'gp_matern_r2']:
    by_n = {}
    for n_sparse in N_POINTS_LIST:
        vals = [r[method] for r in results if r['n_sparse'] == n_sparse and r[method] is not None]
        if vals:
            by_n[str(n_sparse)] = {
                'mean': round(float(np.mean(vals)), 4),
                'median': round(float(np.median(vals)), 4),
                'std': round(float(np.std(vals)), 4),
                'min': round(float(np.min(vals)), 4),
                'max': round(float(np.max(vals)), 4),
                'n_valid': len(vals),
                'n_total': len([r for r in results if r['n_sparse'] == n_sparse]),
            }
    summary[method] = by_n

# Add comparison with PINN and RF at N=15 (from main results)
# Count: how many curves does each method win at N=15?
summary['n15_head_to_head'] = {}
for n_sparse in [15]:
    by_curve = {}
    for r in results:
        if r['n_sparse'] != n_sparse:
            continue
        cn = r['curve']
        if cn not in by_curve:
            by_curve[cn] = {m: [] for m in ['cubic_spline_r2', 'smoothing_spline_r2', 'gp_rbf_r2', 'gp_matern_r2']}
        for m in ['cubic_spline_r2', 'smoothing_spline_r2', 'gp_rbf_r2', 'gp_matern_r2']:
            if r[m] is not None:
                by_curve[cn][m].append(r[m])

    summary['n15_head_to_head']['n_curves'] = len(by_curve)
    for m in ['cubic_spline_r2', 'smoothing_spline_r2', 'gp_rbf_r2', 'gp_matern_r2']:
        mean_r2s = []
        for cn, methods in by_curve.items():
            if methods[m]:
                mean_r2s.append(np.mean(methods[m]))
        summary['n15_head_to_head'][m] = {
            'mean_r2_across_curves': round(float(np.mean(mean_r2s)), 4) if mean_r2s else None,
            'median_r2_across_curves': round(float(np.median(mean_r2s)), 4) if mean_r2s else None,
        }

json_path = OUTPUT_DIR / 'baseline_spline_gp_summary.json'
with open(json_path, 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"JSON summary saved to {json_path}")

# Print summary
print(f"\n{'=' * 90}")
print("SUMMARY (mean R² across 18 curves × 3 seeds = 54 trials)")
print(f"{'Method':<25s} ", end="")
for n_sparse in N_POINTS_LIST:
    print(f"{'N='+str(n_sparse):>12s}", end="")
print()
for method in ['cubic_spline_r2', 'smoothing_spline_r2', 'gp_rbf_r2', 'gp_matern_r2']:
    label = method.replace('_r2', '').replace('_', ' ').title()
    print(f"{label:<25s} ", end="")
    for n_sparse in N_POINTS_LIST:
        s = summary[method].get(str(n_sparse), {})
        mean_val = s.get('mean')
        if mean_val is not None:
            print(f"{mean_val:>12.4f}", end="")
        else:
            print(f"{'N/A':>12s}", end="")
    print()
print(f"{'=' * 90}")
