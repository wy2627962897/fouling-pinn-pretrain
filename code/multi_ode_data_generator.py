"""
Multi-mechanism ODE data generator for PINN pre-training.

Generates synthetic fouling curves from all 5 ODE models with:
  - Parameter stratification: in-domain / out-of-domain for cross-condition experiments
  - Consistent trajectory format: (t_array, Rf_array, ode_params_dict) per trajectory
  - Serialization for fast loading during experiments

ODE models used:
  0: kern_seaton       — asymptotic crystallization fouling
  1: kern_seaton_offset — asymptotic with initial cleaning residual
  2: ks_simple         — simplified phenomenological model
  3: induction_linear  — induction period + linear growth (CaCO3)
  4: ebert_panchal     — threshold model (chemical reaction fouling)

Parameter stratification for cross-condition experiments:
  - In-domain (ID):  params near source_001 (phosphoric acid, KS-type)
  - Out-of-domain 1 (OOD1): params shifted away from source_001, industrial-like
  - Out-of-domain 2 (OOD2): different fouling mechanism entirely (induction, threshold)
"""

import json
import pickle
import random
from pathlib import Path

import numpy as np

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from fouling_models import (
    ks_analytical, ks_offset_analytical, ks_simple_analytical,
    induction_linear_analytical, get_model, MODELS,
)

OUTPUT_DIR = Path(__file__).parent / "output" / "pretrain_data"
SEED = 42


# ===========================================================================
# Parameter ranges for stratified sampling
# ===========================================================================

# In-domain: matching source_001 (phosphoric acid concentration)
KS_IN_DOMAIN = {
    "A_range": (2e-6, 5e-5),
    "Ea_range": (38000, 52000),
    "Tw_range": (340, 380),
    "B_range": (1e-9, 5e-8),
    "v_range": (0.8, 2.5),
    "rho": 1000.0,
    "f": 0.005,
}

# Out-of-domain 1: shifted KS params (industrial-like, different fluids)
KS_OOD1 = {
    "A_range": (5e-5, 1e-4),
    "Ea_range": (52000, 60000),
    "Tw_range": (320, 340),
    "B_range": (5e-8, 1e-7),
    "v_range": (2.5, 4.0),
    "rho": 1000.0,
    "f": 0.005,
}

# Out-of-domain 2: completely different mechanisms
INDUCTION_PARAMS = {
    "t_ind_range": (0.5, 40),
    "k_growth_range": (5e-7, 5e-6),
    "Rf_max_range": (5e-5, 5e-4),
}

EP_PARAMS = {
    "alpha_range": (1e-6, 3e-4),
    "beta_range": (35000, 60000),
    "gamma_range": (3e-10, 5e-7),
    "Re_range": (5000, 50000),
    "Pr_range": (1.0, 15.0),
    "Tw_range": (320, 400),
    "tau_w_range": (0.1, 100.0),
}


# ===========================================================================
# Data generation
# ===========================================================================

def sample_ks_params(param_range, include_offset=False):
    """Sample a set of Kern-Seaton parameters."""
    params = {
        "A": _log_uniform(*param_range["A_range"]),
        "Ea": random.uniform(*param_range["Ea_range"]),
        "Tw": random.uniform(*param_range["Tw_range"]),
        "B": _log_uniform(*param_range["B_range"]),
        "v": random.uniform(*param_range["v_range"]),
        "rho": param_range.get("rho", 1000.0),
        "f": param_range.get("f", 0.005),
    }
    if include_offset:
        params["Rf0"] = random.uniform(0.0, 5e-4)
    return params


def sample_induction_params(param_range):
    return {
        "t_ind": random.uniform(*param_range["t_ind_range"]),
        "k_growth": random.uniform(*param_range["k_growth_range"]),
        "Rf_max": random.uniform(*param_range["Rf_max_range"]),
        "Tw": 360.0,
        "v": 1.0,
    }


def sample_ep_params(param_range):
    return {
        "alpha": random.uniform(*param_range["alpha_range"]),
        "beta": random.uniform(*param_range["beta_range"]),
        "gamma": random.uniform(*param_range["gamma_range"]),
        "Re": random.uniform(*param_range["Re_range"]),
        "Pr": random.uniform(*param_range["Pr_range"]),
        "Tw": random.uniform(*param_range["Tw_range"]),
        "tau_w": random.uniform(*param_range["tau_w_range"]),
        "v": 1.0,
    }


def generate_ks_trajectory(params, duration_h=120, n_points=60, noise_std=2e-6):
    """Generate a Kern-Seaton Rf(t) trajectory."""
    t = np.linspace(0, duration_h, n_points)
    Rf = ks_analytical(
        t, params["A"], params["Ea"], params["Tw"],
        params["B"], params["v"], params["rho"], params["f"],
    )
    Rf_noisy = Rf + np.random.normal(0, noise_std, len(t))
    return t, np.maximum(Rf_noisy, 0.0), params


def generate_ks_offset_trajectory(params, duration_h=120, n_points=60, noise_std=2e-6):
    t = np.linspace(0, duration_h, n_points)
    Rf = ks_offset_analytical(
        t, params["A"], params["Ea"], params["Tw"],
        params["B"], params["v"], params.get("Rf0", 0.0),
        params["rho"], params["f"],
    )
    Rf_noisy = Rf + np.random.normal(0, noise_std, len(t))
    return t, np.maximum(Rf_noisy, 0.0), params


def generate_ks_simple_trajectory(params, duration_h=120, n_points=60, noise_std=2e-6):
    t = np.linspace(0, duration_h, n_points)
    Rf = ks_simple_analytical(
        t, params["Rf_star"], params["tau"], params.get("Rf0", 0.0),
    )
    Rf_noisy = Rf + np.random.normal(0, noise_std, len(t))
    return t, np.maximum(Rf_noisy, 0.0), params


def generate_induction_trajectory(params, duration_h=120, n_points=60, noise_std=2e-6):
    t = np.linspace(0, duration_h, n_points)
    Rf = induction_linear_analytical(
        t, params["t_ind"], params["k_growth"], params.get("Rf_max", 1e-3),
    )
    Rf_noisy = Rf + np.random.normal(0, noise_std, len(t))
    return t, np.maximum(Rf_noisy, 0.0), params


def generate_ep_trajectory(params, duration_h=120, n_points=60, noise_std=2e-6):
    """Ebert-Panchal: numerically integrate ODE (no analytical solution)."""
    from scipy.integrate import solve_ivp
    from fouling_models import ebert_panchal_ode_residual, R_GAS

    def ode_rhs(t, Rf):
        alpha, beta, gamma = params["alpha"], params["beta"], params["gamma"]
        Re, Pr, Tw, tau_w = params["Re"], params["Pr"], params["Tw"], params["tau_w"]
        deposition = alpha * np.exp(-beta / (R_GAS * Tw))
        removal = gamma * tau_w
        return float(deposition - removal)

    t_span = (0, duration_h)
    t_eval = np.linspace(0, duration_h, n_points)
    sol = solve_ivp(ode_rhs, t_span, [0.0], t_eval=t_eval, method="RK45")
    Rf = sol.y[0]
    Rf_noisy = Rf + np.random.normal(0, noise_std, len(Rf))
    return t_eval, np.maximum(Rf_noisy, 0.0), params


# Generator registry
GENERATORS = {
    "kern_seaton": generate_ks_trajectory,
    "kern_seaton_offset": generate_ks_offset_trajectory,
    "ks_simple": generate_ks_simple_trajectory,
    "induction_linear": generate_induction_trajectory,
    "ebert_panchal": generate_ep_trajectory,
}

PARAM_SAMPLERS = {
    "kern_seaton": lambda r: sample_ks_params(r, include_offset=False),
    "kern_seaton_offset": lambda r: sample_ks_params(r, include_offset=True),
    "ks_simple": lambda _: {
        "Rf_star": _log_uniform(5e-5, 5e-3),
        "tau": random.uniform(5.0, 200.0),
        "Rf0": random.uniform(0.0, 5e-4),
        "Tw": 360.0, "v": 1.0,
    },
    "induction_linear": sample_induction_params,
    "ebert_panchal": sample_ep_params,
}


# ===========================================================================
# Stratified data generation
# ===========================================================================

def generate_stratified_dataset(
    n_id_per_ode=200,
    n_ood1_per_ode=100,
    n_ood2_per_ode=100,
    duration_h=120,
    n_points=60,
    noise_std=2e-6,
):
    """Generate stratified pre-training dataset.

    Returns
    -------
    dict with keys:
      "in_domain": list of (t, Rf, params_dict) for each ODE (list of lists)
      "ood1": same structure
      "ood2": same structure
      "ode_names": list of ODE names in order
      "metadata": parameter ranges and stats
    """
    random.seed(SEED)
    np.random.seed(SEED)

    ode_names = ["kern_seaton", "kern_seaton_offset", "ks_simple",
                 "induction_linear", "ebert_panchal"]
    n_ode = len(ode_names)

    # In-domain: KS-type ODEs near source_001 conditions
    id_data = [[] for _ in range(n_ode)]
    # OOD1: KS-type ODEs with shifted conditions
    ood1_data = [[] for _ in range(n_ode)]
    # OOD2: Non-KS mechanisms (induction, EP)
    ood2_data = [[] for _ in range(n_ode)]

    for ode_idx, ode_name in enumerate(ode_names):
        gen_fn = GENERATORS[ode_name]

        # In-domain: use KS_IN_DOMAIN for KS-type, or induction for induction
        if ode_name in ("kern_seaton", "kern_seaton_offset"):
            param_range = KS_IN_DOMAIN
        elif ode_name == "ks_simple":
            param_range = KS_IN_DOMAIN
        elif ode_name == "induction_linear":
            param_range = INDUCTION_PARAMS
        else:
            param_range = EP_PARAMS

        sampler = PARAM_SAMPLERS[ode_name]

        print(f"  Generating {ode_name}: {n_id_per_ode} ID + {n_ood1_per_ode} OOD1 + {n_ood2_per_ode} OOD2...")
        for i in range(n_id_per_ode):
            params = sampler(param_range)
            t, rf, _ = gen_fn(params, duration_h, n_points, noise_std)
            id_data[ode_idx].append((t, rf, params))

        # OOD1: shifted KS params
        if ode_name in ("kern_seaton", "kern_seaton_offset"):
            ood1_range = KS_OOD1
            ood1_sampler = lambda r: sample_ks_params(r, include_offset=(ode_name == "kern_seaton_offset"))
        elif ode_name == "ks_simple":
            ood1_range = {"Rf_star_range": (1e-3, 5e-3), "tau_range": (80, 300)}
            ood1_sampler = lambda _: {
                "Rf_star": _log_uniform(1e-3, 5e-3),
                "tau": random.uniform(80.0, 300.0),
                "Rf0": random.uniform(0.0, 5e-4),
                "Tw": 360.0, "v": 1.0,
            }
        elif ode_name == "induction_linear":
            ood1_range = {"t_ind_range": (30, 80), "k_growth_range": (5e-6, 1e-4), "Rf_max_range": (2e-4, 1e-3)}
            ood1_sampler = sample_induction_params
        else:
            ood1_range = EP_PARAMS
            ood1_sampler = sample_ep_params

        for i in range(n_ood1_per_ode):
            params = ood1_sampler(ood1_range)
            t, rf, _ = gen_fn(params, duration_h, n_points, noise_std)
            ood1_data[ode_idx].append((t, rf, params))

        # OOD2: same ranges as OOD1 but with higher noise (simulating worse data quality)
        for i in range(n_ood2_per_ode):
            params = ood1_sampler(ood1_range)
            t, rf, _ = gen_fn(params, duration_h, n_points, noise_std * 3)
            ood2_data[ode_idx].append((t, rf, params))

    return {
        "in_domain": id_data,
        "ood1": ood1_data,
        "ood2": ood2_data,
        "ode_names": ode_names,
        "metadata": {
            "n_id_per_ode": n_id_per_ode,
            "n_ood1_per_ode": n_ood1_per_ode,
            "n_ood2_per_ode": n_ood2_per_ode,
            "duration_h": duration_h,
            "n_points": n_points,
            "noise_std": noise_std,
            "ks_in_domain": {k: str(v) for k, v in KS_IN_DOMAIN.items()},
            "ks_ood1": {k: str(v) for k, v in KS_OOD1.items()},
        },
    }


# ===========================================================================
# Real data loading
# ===========================================================================

def load_real_curve(curve_path):
    """Load a cleaned real curve CSV. Returns (t_array, Rf_array, metadata_dict)."""
    import pandas as pd
    df = pd.read_csv(curve_path)
    if "time" in df.columns and "value" in df.columns:
        t = df["time"].values.astype(np.float64)
        Rf = df["value"].values.astype(np.float64)
    else:
        # Assume first two columns
        t = df.iloc[:, 0].values.astype(np.float64)
        Rf = df.iloc[:, 1].values.astype(np.float64)

    # Sort by time
    order = np.argsort(t)
    t, Rf = t[order], Rf[order]

    # Remove duplicates
    _, unique_idx = np.unique(t, return_index=True)
    t, Rf = t[unique_idx], Rf[unique_idx]

    return t, Rf


def load_all_real_curves(data_dir=None):
    """Load all cleaned real curves with metadata.

    Returns dict mapping source_name → (t, Rf, ode_name, metadata).
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "real" / "curves" / "cleaned"

    data_dir = Path(data_dir)

    curve_map = {
        "source_001_fig7": {
            "file": "source_001_fig7_rf_exp.csv",
            "ode": "ks_simple",  # Use simplified KS (3-param fit) for robust ODE estimation
            "label": "磷酸浓缩 (Source 001 Fig.7)",
            "domain": "in_domain",
        },
        "source_002_cba": {
            "file": "source_002_fig5_rf_cba.csv",
            "ode": "ks_simple",  # Simplified KS — industrial data has known Rf_star, tau
            "label": "原油预热 CBA (Source 002)",
            "domain": "ood1",
        },
        "source_002_fed": {
            "file": "source_002_fig5_rf_fed.csv",
            "ode": "ks_simple",
            "label": "原油预热 FED (Source 002)",
            "domain": "ood1",
        },
        "source_003_run1": {
            "file": "source_003_fig4_unmodified_run1.csv",
            "ode": "induction_linear",
            "label": "CaCO3 Unmodified Run1 (Source 003)",
            "domain": "ood2",
        },
        "source_003_run2": {
            "file": "source_003_fig4_unmodified_run2.csv",
            "ode": "induction_linear",
            "label": "CaCO3 Unmodified Run2 (Source 003)",
            "domain": "ood2",
        },
        "source_003_run3": {
            "file": "source_003_fig4_unmodified_run3.csv",
            "ode": "induction_linear",
            "label": "CaCO3 Unmodified Run3 (Source 003)",
            "domain": "ood2",
        },
        "source_003_coating_l1": {
            "file": "source_003_fig5_coating_l1.csv",
            "ode": "induction_linear",
            "label": "CaCO3 Coating L1 (Source 003 Fig.5)",
            "domain": "ood2",
        },
        "source_003_coating_l2": {
            "file": "source_003_fig5_coating_l2.csv",
            "ode": "induction_linear",
            "label": "CaCO3 Coating L2 (Source 003 Fig.5)",
            "domain": "ood2",
        },
        "source_003_coating_l3": {
            "file": "source_003_fig5_coating_l3.csv",
            "ode": "induction_linear",
            "label": "CaCO3 Coating L3 (Source 003 Fig.5)",
            "domain": "ood2",
        },
        "source_003_coating_ref": {
            "file": "source_003_fig5_ref.csv",
            "ode": "induction_linear",
            "label": "CaCO3 Coating Ref (Source 003 Fig.5)",
            "domain": "ood2",
        },
        "source_003_fig11_ref": {
            "file": "source_003_fig11_ref.csv",
            "ode": "induction_linear",
            "label": "CaCO3 Surface Ref (Source 003 Fig.11)",
            "domain": "ood2",
        },
        "source_003_fig11_grit80": {
            "file": "source_003_fig11_grit80.csv",
            "ode": "induction_linear",
            "label": "CaCO3 Grit80 (Source 003 Fig.11)",
            "domain": "ood2",
        },
        "source_003_fig11_grit220": {
            "file": "source_003_fig11_grit220.csv",
            "ode": "induction_linear",
            "label": "CaCO3 Grit220 (Source 003 Fig.11)",
            "domain": "ood2",
        },
        "source_003_fig11_diapro": {
            "file": "source_003_fig11_diapro.csv",
            "ode": "induction_linear",
            "label": "CaCO3 DiaPro (Source 003 Fig.11)",
            "domain": "ood2",
        },
        "source_003_fig13_flat": {
            "file": "source_003_fig13_flat.csv",
            "ode": "induction_linear",
            "label": "CaCO3 Flat (Source 003 Fig.13)",
            "domain": "ood2",
        },
        "source_003_fig13_pat1": {
            "file": "source_003_fig13_pattern1.csv",
            "ode": "induction_linear",
            "label": "CaCO3 Pattern1 (Source 003 Fig.13)",
            "domain": "ood2",
        },
        "source_003_fig13_pat2": {
            "file": "source_003_fig13_pattern2.csv",
            "ode": "induction_linear",
            "label": "CaCO3 Pattern2 (Source 003 Fig.13)",
            "domain": "ood2",
        },
        "source_003_fig13_pat3": {
            "file": "source_003_fig13_pattern3.csv",
            "ode": "induction_linear",
            "label": "CaCO3 Pattern3 (Source 003 Fig.13)",
            "domain": "ood2",
        },
    }

    curves = {}
    for name, info in curve_map.items():
        filepath = data_dir / info["file"]
        if not filepath.exists():
            print(f"  ⚠ Missing: {filepath}")
            continue
        t, Rf = load_real_curve(filepath)
        curves[name] = {
            "t": t, "Rf": Rf,
            "ode": info["ode"],
            "label": info["label"],
            "domain": info["domain"],
            "n_points": len(t),
        }
        print(f"  ✅ {name}: {len(t)} points, t=[{t.min():.1f}, {t.max():.1f}], "
              f"Rf=[{Rf.min():.2e}, {Rf.max():.2e}]")

    return curves


# ===========================================================================
# Persistence
# ===========================================================================

def save_dataset(dataset, filepath):
    """Save dataset to pickle file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(dataset, f)
    print(f"  Saved: {filepath}")


def load_dataset(filepath):
    """Load dataset from pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


# ===========================================================================
# Helpers
# ===========================================================================

def _log_uniform(lo, hi):
    return 10 ** random.uniform(np.log10(lo), np.log10(hi))


# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Multi-ODE Pre-training Data Generator")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate full stratified dataset
    print("\n--- Generating stratified dataset ---")
    dataset = generate_stratified_dataset(
        n_id_per_ode=200,
        n_ood1_per_ode=100,
        n_ood2_per_ode=100,
    )

    # Print stats
    total_trajs = sum(len(d) for d in dataset["in_domain"]) + \
                  sum(len(d) for d in dataset["ood1"]) + \
                  sum(len(d) for d in dataset["ood2"])
    print(f"\n  Total trajectories: {total_trajs}")
    for i, ode_name in enumerate(dataset["ode_names"]):
        n_id = len(dataset["in_domain"][i])
        n_ood1 = len(dataset["ood1"][i])
        n_ood2 = len(dataset["ood2"][i])
        print(f"    {ode_name}: {n_id} ID + {n_ood1} OOD1 + {n_ood2} OOD2 = {n_id + n_ood1 + n_ood2}")

    # Save
    save_dataset(dataset, OUTPUT_DIR / "stratified_dataset.pkl")

    # Also save metadata as JSON for inspection
    with open(OUTPUT_DIR / "dataset_metadata.json", "w") as f:
        json.dump(dataset["metadata"], f, indent=2, ensure_ascii=False)

    # Load and print real curves
    print("\n--- Real curves ---")
    curves = load_all_real_curves()
    print(f"\n  Loaded {len(curves)} real curves.")

    # Save real curves
    save_dataset(curves, OUTPUT_DIR / "real_curves.pkl")

    print("\n✅ Data generation complete.")
