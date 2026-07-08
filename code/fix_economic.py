"""
Standalone corrected economic analysis.
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

# ── Corrected cost function ──
def cost_ratio(Rf_true, t, Rf_crit, t_downtime, fixed_interval=None, Rf_predicted=None):
    """All strategies reset Rf after each cleaning. Energy penalty uses TRUE Rf."""
    t0, t_total = t[0], t[-1] - t[0]

    def cyc_energy(D):
        if D <= 0: return 0.0
        m = (t - t0) <= D
        return float(np.trapz(Rf_true[m], t[m])) if m.sum() >= 2 else 0.0

    # Oracle: clean when TRUE Rf hits Rf_crit
    above = np.where(Rf_true >= Rf_crit)[0]
    if len(above) == 0:
        oracle_cost = cyc_energy(t_total); n_ora = 0
    else:
        oracle_cost = 0.0; t_cur = t0
        for a in above:
            usable = t[a] - t_cur
            oracle_cost += 1.0 + cyc_energy(usable)
            t_cur = t[a] + t_downtime
            if t_cur >= t[-1]: break
        if t_cur < t[-1]: oracle_cost += cyc_energy(t[-1] - t_cur)
        n_ora = len(above)

    if oracle_cost <= 0: oracle_cost = max(cyc_energy(t_total), 1e-10)

    # Strategy
    if fixed_interval is not None:
        n = max(0, int(t_total / fixed_interval))
        if n == 0:
            sc = cyc_energy(t_total)
        else:
            cd = fixed_interval - t_downtime
            sc = n * (1.0 + cyc_energy(cd))
            rem = t_total - n * fixed_interval
            if rem > 0: sc += cyc_energy(rem)
        n_s = n
    elif Rf_predicted is not None:
        above_p = np.where(Rf_predicted >= Rf_crit * 0.9)[0]
        if len(above_p) == 0:
            sc = cyc_energy(t_total); n_s = 0
        else:
            pi = t[above_p[0]] - t0
            if pi <= 0: pi = t_total
            n = max(0, int(t_total / pi))
            if n == 0:
                sc = cyc_energy(t_total)
            else:
                cd = pi - t_downtime
                sc = n * (1.0 + cyc_energy(cd))
                rem = t_total - n * pi
                if rem > 0: sc += cyc_energy(rem)
            n_s = n
    else:
        sc = cyc_energy(t_total); n_s = 0

    return float(sc / oracle_cost), float(sc), float(oracle_cost), n_s, n_ora

CURVES = ['source_001_fig7','source_002_cba','source_002_fed',
          'source_003_coating_l1','source_003_fig11_ref','source_003_fig13_flat',
          'source_003_run1','source_003_run2']

print(f"{'Curve':<28s} {'PINN R2':>8s} {'Fixed25%':>8s} {'Fixed33%':>8s} {'PINN':>8s} {'Sav%':>6s} {'N_Ora':>5s} {'N_PINN':>6s}")
print('-' * 90)

for cn in CURVES:
    ci = curves[cn]; t, rf = ci['t'], ci['Rf']
    if 'source_002' in cn: rf = rf * SRC2

    np.random.seed(SEED); torch.manual_seed(SEED)
    idx = np.random.choice(len(t), N_SPARSE, replace=False); idx.sort()

    # PINN FT
    m = PINN(hidden_layers=(64,64,64,64)).to(DEVICE); m.load_state_dict(pretrained)
    ep = estimate_ode_params_from_sparse(t[idx], rf[idx], ci['ode'])
    finetune_pinn(m, t[idx], rf[idx], ci['ode'], ep, (float(t.min()),float(t.max())),
                  n_epochs=2000, lambda_phys=0.3, lr=1e-4, n_colloc=150, device=DEVICE, verbose=False)
    yp = predict_trajectory(m, t, ci['ode'], ep, DEVICE)
    r2_val = r2_score(rf, yp)

    Rcrit = 0.8 * np.max(rf); td = 0.02 * (t[-1]-t[0])
    f25, _, _, _, _ = cost_ratio(rf, t, Rcrit, td, fixed_interval=(t[-1]-t[0])*0.25)
    f33, _, _, _, _ = cost_ratio(rf, t, Rcrit, td, fixed_interval=(t[-1]-t[0])*0.33)
    pr, sc, oc, ns, no = cost_ratio(rf, t, Rcrit, td, Rf_predicted=yp)
    bf = min(f25, f33)
    sav = (bf - pr) / bf * 100 if bf > 0 else 0

    print(f'{cn:<28s} {r2_val:>8.4f} {f25:>8.3f} {f33:>8.3f} {pr:>8.3f} {sav:>5.1f}% {no:>5d} {ns:>6d}')
