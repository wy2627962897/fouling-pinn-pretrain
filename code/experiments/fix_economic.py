"""
Standalone corrected economic analysis.
Q12 fix: Rename "Oracle" → "Reference threshold policy (full-curve)".
         Add multi-threshold sensitivity analysis.
         Acknowledge that Rf_crit depends on max(Rf) which is unknown a priori;
         in practice, Rf_crit is set by the heat exchanger design specification.
"""
import sys, pickle, numpy as np, torch, json
from pathlib import Path
sys.path.insert(0, '.')
from pinn_model import *
from sklearn.metrics import r2_score

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED, N_SPARSE, SRC2 = 42, 15, 0.001

with open('output/pretrain_data/real_curves.pkl', 'rb') as f:
    curves = pickle.load(f)
pretrained = torch.load('output/experiment_results/pretrained_model.pt', weights_only=True)


# ── Cost function (refactored for Q12) ──
def cost_ratio(Rf_true, t, Rf_crit, t_downtime, fixed_interval=None, Rf_predicted=None):
    """
    Compute dimensionless cost ratio for a cleaning strategy.

    IMPORTANT (Q12): The "reference" policy uses true Rf(t) to trigger cleaning
    at Rf_crit. This is NOT a cost-optimal benchmark -- it's a threshold-triggered
    policy that assumes perfect real-time fouling monitoring. It does NOT minimize
    total cost C_total = k*C_clean + c_energy*∫Rf(t)dt; it simply cleans whenever
    the true Rf exceeds the threshold. Cost ratios below 1.0 mean the strategy
    chooses a cheaper cleaning schedule than the reference -- not that it beats
    the global optimum.

    Also note: Rf_crit is typically defined as 0.8*max(Rf) from the full curve,
    which requires knowledge of the peak fouling resistance -- information
    unavailable at t=0 in real operations. The sensitivity analysis below varies
    this threshold to assess robustness.

    All strategies compute energy penalty from TRUE Rf(t). PINN predictions are
    used ONLY for cleaning timing decisions.
    """
    t0, t_total = t[0], t[-1] - t[0]

    def cyc_energy(D):
        """Energy penalty for one fouling cycle of duration D.
        Uses max(Rf, 0) so that negative Rf (heat transfer enhancement from coatings)
        contributes zero penalty rather than a spurious negative cost."""
        if D <= 0:
            return 0.0
        m = (t - t0) <= D
        if m.sum() < 2:
            return 0.0
        Rf_clipped = np.maximum(Rf_true[m], 0.0)
        return float(np.trapz(Rf_clipped, t[m]))

    # ── Reference threshold policy: clean when TRUE Rf reaches Rf_crit ──
    # NOT a cost-optimal benchmark. Uses full-curve knowledge (continuous
    # monitoring + known Rf_crit) but does NOT minimize total cost.
    above = np.where(Rf_true >= Rf_crit)[0]
    if len(above) == 0:
        ref_cost = cyc_energy(t_total)
        n_ref = 0
    else:
        ref_cost = 0.0
        t_cur = t0
        for a in above:
            usable = t[a] - t_cur
            ref_cost += 1.0 + cyc_energy(usable)
            t_cur = t[a] + t_downtime
            if t_cur >= t[-1]:
                break
        if t_cur < t[-1]:
            ref_cost += cyc_energy(t[-1] - t_cur)
        n_ref = len(above)

    if ref_cost <= 0:
        ref_cost = max(cyc_energy(t_total), 1e-10)

    # ── Strategy cost ──
    if fixed_interval is not None:
        n = max(0, int(t_total / fixed_interval))
        if n == 0:
            sc = cyc_energy(t_total)
        else:
            cd = fixed_interval - t_downtime
            sc = n * (1.0 + cyc_energy(cd))
            rem = t_total - n * fixed_interval
            if rem > 0:
                sc += cyc_energy(rem)
        n_s = n
    elif Rf_predicted is not None:
        # PINN-predicted strategy: use PREDICTED Rf to detect threshold crossing
        # 10% safety margin (clean earlier than predicted threshold)
        above_p = np.where(Rf_predicted >= Rf_crit * 0.9)[0]
        if len(above_p) == 0:
            sc = cyc_energy(t_total)
            n_s = 0
        else:
            pi = t[above_p[0]] - t0
            if pi <= 0:
                pi = t_total
            n = max(0, int(t_total / pi))
            if n == 0:
                sc = cyc_energy(t_total)
            else:
                cd = pi - t_downtime
                sc = n * (1.0 + cyc_energy(cd))
                rem = t_total - n * pi
                if rem > 0:
                    sc += cyc_energy(rem)
            n_s = n
    else:
        sc = cyc_energy(t_total)
        n_s = 0

    return float(sc / ref_cost), float(sc), float(ref_cost), n_s, n_ref


CURVES = ['source_001_fig7', 'source_002_cba', 'source_002_fed',
          'source_003_coating_l1', 'source_003_fig11_ref', 'source_003_fig13_flat',
          'source_003_run1', 'source_003_run2']

# ── Multi-threshold sensitivity analysis (Q12 fix) ──
# Instead of a single Rf_crit = 0.8*max(Rf), we sweep 4 threshold levels.
# In practice, the plant engineer sets Rf_crit based on the heat exchanger
# design specification (max allowable fouling resistance). Since we lack
# per-curve design specs, we vary the threshold factor as a sensitivity check.
THRESHOLD_FACTORS = [0.50, 0.65, 0.80, 0.95]

print("=" * 100)
print("Q12 FIX: Multi-threshold sensitivity analysis")
print("Reference = threshold-triggered policy using TRUE Rf(t)")
print("This is NOT a cost-optimal benchmark -- it does NOT minimize C_total.")
print("Rf_crit = factor * max(Rf) from full curve (unknown a priori in practice).")
print("=" * 100)

for factor in THRESHOLD_FACTORS:
    print(f"\n{'─' * 90}")
    print(f"Threshold factor = {factor:.2f}  |  Rf_crit = {factor:.0%} of max(Rf)")
    print(f"{'─' * 90}")
    print(f"{'Curve':<28s} {'PINN R2':>8s} {'Fixed25%':>8s} {'Fixed33%':>8s} "
          f"{'PINN':>8s} {'Sav%':>6s} {'N_Ref':>5s} {'N_PINN':>6s}")
    print('-' * 90)

    for cn in CURVES:
        ci = curves[cn]
        t, rf = ci['t'], ci['Rf']
        if 'source_002' in cn:
            rf = rf * SRC2

        np.random.seed(SEED)
        torch.manual_seed(SEED)
        idx = np.random.choice(len(t), N_SPARSE, replace=False)
        idx.sort()

        # PINN FT
        m = PINN(hidden_layers=(64, 64, 64, 64)).to(DEVICE)
        m.load_state_dict(pretrained)
        ep = estimate_ode_params_from_sparse(t[idx], rf[idx], ci['ode'])
        finetune_pinn(m, t[idx], rf[idx], ci['ode'], ep, (float(t.min()), float(t.max())),
                      n_epochs=2000, lambda_phys=0.3, lr=1e-4, n_colloc=150,
                      device=DEVICE, verbose=False)
        yp = predict_trajectory(m, t, ci['ode'], ep, DEVICE)
        r2_val = r2_score(rf, yp)

        Rcrit = factor * np.max(rf)
        td = 0.02 * (t[-1] - t[0])

        f25, _, _, _, _ = cost_ratio(rf, t, Rcrit, td, fixed_interval=(t[-1] - t[0]) * 0.25)
        f33, _, _, _, _ = cost_ratio(rf, t, Rcrit, td, fixed_interval=(t[-1] - t[0]) * 0.33)
        pr, sc, rc, ns, nr = cost_ratio(rf, t, Rcrit, td, Rf_predicted=yp)
        bf = min(f25, f33)
        sav = (bf - pr) / bf * 100 if bf > 0 else 0

        print(f'{cn:<28s} {r2_val:>8.4f} {f25:>8.3f} {f33:>8.3f} {pr:>8.3f} '
              f'{sav:>5.1f}% {nr:>5d} {ns:>6d}')

print(f"\n{'=' * 100}")
print("Sensitivity summary: check whether strategy ranking is stable across thresholds.")
print("If PINN consistently beats fixed-interval across all factors, the conclusion")
print("is robust to the (unknown) choice of Rf_crit.")
print(f"{'=' * 100}")
