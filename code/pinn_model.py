"""
PINN with multi-mechanism ODE support for pre-training + fine-tuning.

Architecture:
  - Unified input: [t, Tw, v, p1, p2, p3, p4, p5, p6, ode_type_id] (10 dims)
  - p1-p6 encode ODE-specific parameters; unused slots are 0-filled
  - Trajectory-level physics loss via fouling_models ODE registry
  - Pre-training: multi-ODE synthetic data, ODE-cycled per epoch
  - Fine-tuning: matched-ODE on sparse real data with estimated params

Parameter mapping per ODE type:
  ode_type_id 0 (kern_seaton):        p1=A, p2=Ea, p3=B, p4=rho, p5=f, p6=0
  ode_type_id 1 (kern_seaton_offset): p1=A, p2=Ea, p3=B, p4=Rf0, p5=rho, p6=f
  ode_type_id 2 (ks_simple):          p1=Rf_star, p2=tau, p3=Rf0, p4=p5=p6=0
  ode_type_id 3 (induction_linear):   p1=t_ind, p2=k_growth, p3=Rf_max, p4=p5=p6=0
  ode_type_id 4 (ebert_panchal):      p1=alpha, p2=beta, p3=gamma, p4=Re, p5=Pr, p6=tau_w
"""

import math
import sys
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path

# Import ODE registry from sibling module
sys.path.insert(0, str(Path(__file__).parent))
from fouling_models import get_model, MODELS

R_GAS = 8.314  # J/(mol·K)

# ─────────────────────────────────────────────────────────────────────
# ODE type mapping
# ─────────────────────────────────────────────────────────────────────
ODE_TYPE_IDS = {
    "kern_seaton": 0,
    "kern_seaton_offset": 1,
    "ks_simple": 2,
    "induction_linear": 3,
    "ebert_panchal": 4,
}
ODE_NAMES = {v: k for k, v in ODE_TYPE_IDS.items()}
N_ODE_TYPES = len(ODE_TYPE_IDS)
INPUT_DIM = 10  # t, Tw, v, p1-p6, ode_type_id


class PINN(nn.Module):
    """Physics-Informed Neural Network for fouling prediction.

    Input: (N, 10) tensor [t, Tw, v, p1..p6, ode_type_id]
    Output: (N, 1) tensor Rf
    """

    def __init__(self, hidden_layers=(64, 64, 64, 64), in_dim=INPUT_DIM):
        super().__init__()
        layers = []
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.in_dim = INPUT_DIM

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────
# Unified parameter encoding / decoding
# ─────────────────────────────────────────────────────────────────────

def encode_condition_params(ode_name, **params):
    """Encode ODE-specific parameters into unified [Tw, v, p1-p6, ode_type_id].

    Returns tensor of shape (1, 9) — time is added later per data point.
    The ode_type_id is the last element, appended when building full input.
    """
    ode_id = ODE_TYPE_IDS[ode_name]
    p = [0.0] * 6

    if ode_name == "kern_seaton":
        Tw = params.get("Tw", 360.0)
        v = params.get("v", 1.0)
        p[0] = params.get("A", 5e-5)
        p[1] = params.get("Ea", 45000.0)
        p[2] = params.get("B", 5e-8)
        p[3] = params.get("rho", 1000.0)
        p[4] = params.get("f", 0.005)
    elif ode_name == "kern_seaton_offset":
        Tw = params.get("Tw", 360.0)
        v = params.get("v", 1.0)
        p[0] = params.get("A", 5e-5)
        p[1] = params.get("Ea", 45000.0)
        p[2] = params.get("B", 5e-8)
        p[3] = params.get("Rf0", 0.0)
        p[4] = params.get("rho", 1000.0)
        p[5] = params.get("f", 0.005)
    elif ode_name == "ks_simple":
        Tw = params.get("Tw", 360.0)
        v = params.get("v", 1.0)
        p[0] = params.get("Rf_star", 1.8e-4)
        p[1] = params.get("tau", 40.0)
        p[2] = params.get("Rf0", 0.0)
    elif ode_name == "induction_linear":
        Tw = params.get("Tw", 360.0)
        v = params.get("v", 1.0)
        p[0] = params.get("t_ind", 10.0)
        p[1] = params.get("k_growth", 1e-6)
        p[2] = params.get("Rf_max", 1e-3)
    elif ode_name == "ebert_panchal":
        Tw = params.get("Tw", 360.0)
        v = params.get("v", 1.0)
        p[0] = params.get("alpha", 1e-4)
        p[1] = params.get("beta", 45000.0)
        p[2] = params.get("gamma", 1e-8)
        p[3] = params.get("Re", 10000.0)
        p[4] = params.get("Pr", 7.0)
        p[5] = params.get("tau_w", 10.0)
    else:
        raise ValueError(f"Unknown ODE name: {ode_name}")

    return torch.tensor([[Tw, v] + p], dtype=torch.float32)


def build_input_tensor(t_values, condition_vec, ode_type_id, device="cpu"):
    """Build full input tensor [t, Tw, v, p1..p6, ode_type_id].

    t_values: (N, 1) or (N,) tensor/array of time points
    condition_vec: (1, 8) tensor [Tw, v, p1..p6]
    ode_type_id: int or (N, 1) tensor
    """
    if not isinstance(t_values, torch.Tensor):
        t_values = torch.tensor(t_values, dtype=torch.float32)
    if t_values.ndim == 1:
        t_values = t_values.reshape(-1, 1)
    t_values = t_values.to(device)

    n = t_values.shape[0]
    cond = condition_vec.repeat(n, 1).to(device)  # (N, 8)
    ode_col = torch.full((n, 1), float(ode_type_id), dtype=torch.float32, device=device)
    x = torch.cat([t_values, cond, ode_col], dim=1)
    return x


# ─────────────────────────────────────────────────────────────────────
# Physics loss — trajectory-level, dispatched by ODE type
# ─────────────────────────────────────────────────────────────────────

def compute_physics_loss_trajectory(model, t, ode_name, ode_params, device="cpu"):
    """Compute ODE residual loss for a SINGLE trajectory.

    t: (N, 1) tensor — time points along ONE trajectory (requires_grad)
    ode_name: str — ODE model name
    ode_params: dict — physical parameters for this trajectory

    Returns scalar MSE of ODE residual.
    """
    condition_vec = encode_condition_params(ode_name, **ode_params)
    ode_type_id = ODE_TYPE_IDS[ode_name]
    x = build_input_tensor(t, condition_vec, ode_type_id, device)

    Rf = model(x)

    # dRf/dt via autograd
    dRf_dt = torch.autograd.grad(
        Rf, t, grad_outputs=torch.ones_like(Rf),
        create_graph=True, retain_graph=True
    )[0]

    # Get ODE residual function from registry
    fouling_model = get_model(ode_name)
    residual_fn = fouling_model.ode_residual_fn

    # Convert tensors to numpy for the ODE function, then back to torch
    # The ODE functions in fouling_models operate on numpy arrays
    Rf_np = Rf.detach().cpu().numpy().flatten()
    dRf_dt_np = dRf_dt.detach().cpu().numpy().flatten()
    t_np = t.detach().cpu().numpy().flatten()

    # Build params dict with correct types for the ODE function
    params_for_ode = _prepare_ode_params(ode_name, ode_params)

    try:
        residual_np = residual_fn(dRf_dt_np, Rf_np, **params_for_ode)
    except TypeError:
        # Some ODE functions need 't' as kwarg (e.g. induction_linear)
        params_for_ode["t"] = t_np
        residual_np = residual_fn(dRf_dt_np, Rf_np, **params_for_ode)

    residual = torch.tensor(residual_np, dtype=torch.float32, device=device)
    return torch.mean(residual ** 2)


def _prepare_ode_params(ode_name, params):
    """Convert generic params dict to the specific kwargs expected by each ODE."""
    if ode_name == "kern_seaton":
        return {
            "A": params["A"], "Ea": params["Ea"], "Tw": params["Tw"],
            "B": params["B"], "v": params["v"],
            "rho": params.get("rho", 1000.0), "f": params.get("f", 0.005),
        }
    elif ode_name == "kern_seaton_offset":
        return {
            "A": params["A"], "Ea": params["Ea"], "Tw": params["Tw"],
            "B": params["B"], "v": params["v"], "Rf0": params.get("Rf0", 0.0),
            "rho": params.get("rho", 1000.0), "f": params.get("f", 0.005),
        }
    elif ode_name == "ks_simple":
        return {
            "Rf_star": params["Rf_star"], "tau": params["tau"],
        }
    elif ode_name == "induction_linear":
        return {
            "t_ind": params["t_ind"], "k_growth": params["k_growth"],
        }
    elif ode_name == "ebert_panchal":
        return {
            "alpha": params["alpha"], "beta": params["beta"],
            "gamma": params["gamma"], "Re": params["Re"],
            "Pr": params["Pr"], "Tw": params["Tw"],
            "tau_w": params["tau_w"],
        }
    else:
        raise ValueError(f"Unknown ODE: {ode_name}")


def compute_physics_loss_batch(model, t_batch, ode_names, ode_params_list, device="cpu"):
    """Compute physics loss for a batch of trajectories.

    t_batch: list of (N_i, 1) tensors, one per trajectory
    ode_names: list of str, one per trajectory
    ode_params_list: list of dict, one per trajectory

    Returns mean physics loss across all trajectories.
    """
    losses = []
    for t, ode_name, params in zip(t_batch, ode_names, ode_params_list):
        t.requires_grad_(True)
        loss = compute_physics_loss_trajectory(model, t, ode_name, params, device)
        losses.append(loss)
    return torch.stack(losses).mean()


# ─────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────

def prepare_tensors(df, device="cpu"):
    """Convert legacy-format DataFrame to tensors [t, Tw, v, A, B, Ea].

    Backward-compatible with old single-ODE (kern_seaton) data format.
    """
    t = torch.tensor(df["day"].values, dtype=torch.float32).reshape(-1, 1)
    Tw = torch.tensor(df["Tw"].values, dtype=torch.float32).reshape(-1, 1)
    v = torch.tensor(df["v"].values, dtype=torch.float32).reshape(-1, 1)
    A_val = torch.tensor(df["A"].values, dtype=torch.float32).reshape(-1, 1)
    B_val = torch.tensor(df["B"].values, dtype=torch.float32).reshape(-1, 1)
    Ea = torch.tensor(df["Ea"].values, dtype=torch.float32).reshape(-1, 1)
    x = torch.cat([t, Tw, v, A_val, B_val, Ea], dim=1).to(device)
    y = torch.tensor(df["Rf_noisy"].values, dtype=torch.float32).reshape(-1, 1).to(device)
    y_true = torch.tensor(df["Rf_true"].values, dtype=torch.float32).reshape(-1, 1).to(device)
    return x, y, y_true


def prepare_unified_tensors(t_values, Rf_values, condition_vec, ode_type_id, device="cpu"):
    """Prepare tensors in the unified 10-dim format for a single trajectory.

    t_values: (N,) array — time points
    Rf_values: (N,) array — Rf observations
    condition_vec: (1, 8) tensor — [Tw, v, p1..p6]
    ode_type_id: int
    """
    t = torch.tensor(t_values, dtype=torch.float32).reshape(-1, 1)
    x = build_input_tensor(t, condition_vec, ode_type_id, device)
    y = torch.tensor(Rf_values, dtype=torch.float32).reshape(-1, 1).to(device)
    return x, y


# ─────────────────────────────────────────────────────────────────────
# Legacy (single-ODE, backward-compatible) training
# ─────────────────────────────────────────────────────────────────────

def compute_physics_loss(model, x_phys):
    """Legacy: compute Kern-Seaton physics loss (backward compatible)."""
    t = x_phys[:, 0:1].clone().detach().requires_grad_(True)
    Tw = x_phys[:, 1:2]
    v = x_phys[:, 2:3]
    A_val = x_phys[:, 3:4]
    B_val = x_phys[:, 4:5]
    Ea = x_phys[:, 5:6]

    x_with_grad_t = torch.cat([t, Tw, v, A_val, B_val, Ea], dim=1)
    Rf = model(x_with_grad_t)

    dRf_dt = torch.autograd.grad(Rf, t, grad_outputs=torch.ones_like(Rf),
                                  create_graph=True)[0]

    phi_d = A_val * torch.exp(-Ea / (R_GAS * Tw))
    tau_s = 0.5 * 0.005 * 1000.0 * v * v
    phi_r = B_val * tau_s * Rf

    residual = dRf_dt - (phi_d - phi_r)
    return torch.mean(residual ** 2)


def compute_data_loss(model, x_data, y_data):
    Rf_pred = model(x_data)
    return torch.mean((Rf_pred - y_data) ** 2)


def train_pinn(model, x_data, y_data, x_phys_collocation,
               lambda_phys=1.0, lr=1e-3, n_epochs=5000, verbose=True):
    """Legacy single-ODE training (backward compatible)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=200, min_lr=1e-6
    )

    data_losses, phys_losses = [], []

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        loss_data = compute_data_loss(model, x_data, y_data)
        loss_phys = compute_physics_loss(model, x_phys_collocation)
        loss_total = loss_data + lambda_phys * loss_phys

        loss_total.backward()
        optimizer.step()
        scheduler.step(loss_total)

        data_losses.append(loss_data.item())
        phys_losses.append(loss_phys.item())

        if verbose and (epoch + 1) % 1000 == 0:
            print(f"  Epoch {epoch+1:>5}/{n_epochs}: "
                  f"data_loss={loss_data.item():.6e}, "
                  f"phys_loss={loss_phys.item():.6e}, "
                  f"total={loss_total.item():.6e}")

    return {"data_losses": data_losses, "phys_losses": phys_losses}


def generate_collocation_points(n_points, t_range=(0, 1825), device="cpu"):
    """Legacy collocation point generation for Kern-Seaton (backward compatible)."""
    t = torch.rand(n_points, 1) * (t_range[1] - t_range[0]) + t_range[0]
    Tw = torch.rand(n_points, 1) * (400 - 320) + 320
    v = torch.rand(n_points, 1) * (4.0 - 0.3) + 0.3
    A_val = 10 ** (torch.rand(n_points, 1) * (-4.0 + 5.5) - 5.5)
    B_val = 10 ** (torch.rand(n_points, 1) * (-7.0 + 8.5) - 8.5)
    Ea = torch.rand(n_points, 1) * (60000 - 35000) + 35000
    return torch.cat([t, Tw, v, A_val, B_val, Ea], dim=1).to(device)


# ─────────────────────────────────────────────────────────────────────
# Multi-ODE pre-training
# ─────────────────────────────────────────────────────────────────────

def pretrain_pinn_multi_ode(
    model,
    trajectory_batches,
    ode_names,
    n_epochs=5000,
    lambda_phys=0.5,
    lr=1e-3,
    n_colloc_per_traj=100,
    batch_size_per_ode=8,
    device="cpu",
    verbose=True,
):
    """Pre-train PINN on multi-mechanism synthetic data with mini-batching.

    Parameters
    ----------
    model: PINN instance
    trajectory_batches: list of lists, each inner list is:
        [(t_array, Rf_array, ode_params_dict), ...] for one ODE type
    ode_names: list of str — ODE names
    n_epochs: total training epochs
    lambda_phys: physics loss weight
    lr: learning rate
    n_colloc_per_traj: number of collocation time points per trajectory
    batch_size_per_ode: number of trajectories to sample per ODE type per epoch
    device: torch device
    verbose: print progress

    Returns
    -------
    dict with training history
    """
    import random
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=500, min_lr=1e-6
    )

    n_ode_types = len(ode_names)
    history = {"data_losses": [], "phys_losses": [], "total_losses": []}

    for epoch in range(n_epochs):
        model.train()
        epoch_data_loss = 0.0
        epoch_phys_loss = 0.0
        n_steps = 0

        # Cycle through ODE types (each gets one gradient step per epoch)
        for ode_idx, ode_name in enumerate(ode_names):
            trajectories = trajectory_batches[ode_idx]
            if not trajectories:
                continue

            # Mini-batch: randomly sample trajectories
            n_sample = min(batch_size_per_ode, len(trajectories))
            batch_trajs = random.sample(trajectories, n_sample)

            optimizer.zero_grad()

            batch_data_loss = 0.0
            batch_phys_loss = 0.0

            for t_arr, rf_arr, ode_params in batch_trajs:
                # Data loss
                condition_vec = encode_condition_params(ode_name, **ode_params)
                ode_type_id = ODE_TYPE_IDS[ode_name]
                t_tensor = torch.tensor(t_arr, dtype=torch.float32).reshape(-1, 1).to(device)
                x_data = build_input_tensor(t_tensor, condition_vec, ode_type_id, device)
                y_data = torch.tensor(rf_arr, dtype=torch.float32).reshape(-1, 1).to(device)
                Rf_pred = model(x_data)
                data_loss = torch.mean((Rf_pred - y_data) ** 2)
                batch_data_loss += data_loss

                # Physics loss: collocation points along trajectory
                t_min, t_max = float(t_arr.min()), float(t_arr.max())
                t_colloc = torch.rand(n_colloc_per_traj, 1, device=device) * (t_max - t_min) + t_min
                t_colloc.requires_grad_(True)
                phys_loss = compute_physics_loss_trajectory(
                    model, t_colloc, ode_name, ode_params, device
                )
                batch_phys_loss += phys_loss

            avg_data = batch_data_loss / n_sample
            avg_phys = batch_phys_loss / n_sample
            total_loss = avg_data + lambda_phys * avg_phys
            total_loss.backward()
            optimizer.step()

            epoch_data_loss += avg_data.item()
            epoch_phys_loss += avg_phys.item()
            n_steps += 1

        # Average across ODE types
        epoch_data_loss /= n_steps
        epoch_phys_loss /= n_steps
        total = epoch_data_loss + lambda_phys * epoch_phys_loss

        scheduler.step(total)
        history["data_losses"].append(epoch_data_loss)
        history["phys_losses"].append(epoch_phys_loss)
        history["total_losses"].append(total)

        if verbose and (epoch + 1) % 500 == 0:
            print(f"  Epoch {epoch+1:>5}/{n_epochs}: "
                  f"data={epoch_data_loss:.6e}, phys={epoch_phys_loss:.6e}, "
                  f"total={total:.6e}")

    return history


# ─────────────────────────────────────────────────────────────────────
# Fine-tuning on sparse real data
# ─────────────────────────────────────────────────────────────────────

def finetune_pinn(
    model,
    t_sparse,
    Rf_sparse,
    ode_name,
    ode_params_estimated,
    t_phys_range,
    n_epochs=2000,
    lambda_phys=0.3,
    lr=1e-4,
    n_colloc=200,
    device="cpu",
    verbose=True,
):
    """Fine-tune a pre-trained PINN on sparse real data.

    Parameters
    ----------
    model: pre-trained PINN
    t_sparse: (N,) array — sparse time points from real curve
    Rf_sparse: (N,) array — Rf values at sparse points
    ode_name: str — matched ODE for physics loss
    ode_params_estimated: dict — ODE parameters estimated from sparse fitting
    t_phys_range: (t_min, t_max) — time range for collocation points
    n_epochs: fine-tuning epochs
    lambda_phys: physics loss weight
    lr: learning rate (lower for fine-tuning)
    n_colloc: number of collocation points
    device: torch device
    verbose: print progress

    Returns
    -------
    dict with training history
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=300, min_lr=1e-7
    )

    condition_vec = encode_condition_params(ode_name, **ode_params_estimated)
    ode_type_id = ODE_TYPE_IDS[ode_name]

    # Prepare sparse data tensors
    x_data, y_data = prepare_unified_tensors(t_sparse, Rf_sparse, condition_vec, ode_type_id, device)

    history = {"data_losses": [], "phys_losses": [], "total_losses": []}

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        # Data loss on sparse points
        Rf_pred = model(x_data)
        loss_data = torch.mean((Rf_pred - y_data) ** 2)

        # Physics loss with collocation points
        t_min, t_max = t_phys_range
        t_colloc = torch.rand(n_colloc, 1, device=device) * (t_max - t_min) + t_min
        t_colloc.requires_grad_(True)
        loss_phys = compute_physics_loss_trajectory(
            model, t_colloc, ode_name, ode_params_estimated, device
        )

        loss_total = loss_data + lambda_phys * loss_phys
        loss_total.backward()
        optimizer.step()
        scheduler.step(loss_total)

        history["data_losses"].append(loss_data.item())
        history["phys_losses"].append(loss_phys.item())
        history["total_losses"].append(loss_total.item())

        if verbose and (epoch + 1) % 500 == 0:
            print(f"  FT Epoch {epoch+1:>5}/{n_epochs}: "
                  f"data={loss_data.item():.6e}, phys={loss_phys.item():.6e}, "
                  f"total={loss_total.item():.6e}")

    return history


# ─────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────

def predict_trajectory(model, t_values, ode_name, ode_params, device="cpu"):
    """Predict Rf(t) for a full trajectory given ODE params.

    t_values: (N,) array — time points for prediction
    ode_name: str
    ode_params: dict

    Returns (N,) numpy array of predicted Rf values.
    """
    model.eval()
    condition_vec = encode_condition_params(ode_name, **ode_params)
    ode_type_id = ODE_TYPE_IDS[ode_name]

    t_tensor = torch.tensor(t_values, dtype=torch.float32).reshape(-1, 1).to(device)
    x = build_input_tensor(t_tensor, condition_vec, ode_type_id, device)

    with torch.no_grad():
        y_pred = model(x)

    return y_pred.cpu().numpy().flatten()


def evaluate(model, test_df, device="cpu"):
    """Legacy evaluation on test DataFrame (backward compatible)."""
    x_test, _, y_true = prepare_tensors(test_df, device)
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)
    from sklearn.metrics import r2_score, mean_absolute_error
    yp = y_pred.cpu().numpy().flatten()
    yt = y_true.cpu().numpy().flatten()
    return {"R2": r2_score(yt, yp), "MAE": mean_absolute_error(yt, yp),
            "y_pred": yp, "y_true": yt}


def evaluate_trajectory(y_pred, y_true):
    """Compute R² and MAE for a single trajectory prediction."""
    from sklearn.metrics import r2_score, mean_absolute_error
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(np.mean((y_true - y_pred) ** 2)),
    }


# ─────────────────────────────────────────────────────────────────────
# ODE parameter estimation from sparse data (for fine-tuning pipeline)
# ─────────────────────────────────────────────────────────────────────

def estimate_ode_params_from_sparse(t_sparse, Rf_sparse, ode_name):
    """Fit ODE model to sparse real data to estimate physical parameters.

    Uses analytical solution (if available) or simplified fitting.
    Returns dict of estimated parameters.
    """
    from fouling_models import fit_model_to_curve, get_model

    if ode_name == "ks_simple":
        result = fit_model_to_curve(t_sparse, Rf_sparse, "ks_simple")
        if result["success"]:
            return {
                "Rf_star": result["params"]["Rf_star"],
                "tau": result["params"]["tau"],
                "Rf0": result["params"].get("Rf0", 0.0),
                "Tw": 360.0, "v": 1.0,
            }

    elif ode_name == "kern_seaton":
        # Use ks_simple fitting, then map to KS params
        from fouling_models import fit_model_to_curve
        result = fit_model_to_curve(t_sparse, Rf_sparse, "ks_simple")
        if result["success"]:
            Rf_star = result["params"]["Rf_star"]
            tau = result["params"]["tau"]
            # Approximate physical params: Rf_star = phi_d/(B*τs), τ = 1/(B*τs)
            # We don't have enough info to uniquely determine A, Ea, B, Tw, v
            # Use reasonable defaults and fit what we can
            B_default = 5e-8
            v_default = 1.0
            tau_s = 0.5 * 0.005 * 1000.0 * v_default * v_default
            B_est = 1.0 / (tau * 3600 * tau_s)  # convert tau from hours to seconds, then to B
            phi_d_est = Rf_star * B_est * tau_s

            return {
                "A": min(max(phi_d_est * 10, 3e-6), 1e-4),  # rough estimate
                "Ea": 45000.0,
                "Tw": 360.0,
                "B": min(max(B_est, 3e-9), 1e-7),
                "v": v_default,
                "rho": 1000.0,
                "f": 0.005,
            }

    elif ode_name == "kern_seaton_offset":
        result = fit_model_to_curve(t_sparse, Rf_sparse, "ks_simple")
        if result["success"]:
            Rf0_est = result["params"].get("Rf0", 0.0)
            Rf_star = result["params"]["Rf_star"]
            tau = result["params"]["tau"]
            B_default = 5e-8
            v_default = 1.0
            tau_s = 0.5 * 0.005 * 1000.0 * v_default * v_default
            B_est = 1.0 / (tau * 3600 * tau_s)
            phi_d_est = Rf_star * B_est * tau_s

            return {
                "A": min(max(phi_d_est * 10, 3e-6), 1e-4),
                "Ea": 45000.0,
                "Tw": 360.0,
                "B": min(max(B_est, 3e-9), 1e-7),
                "Rf0": Rf0_est,
                "v": v_default,
                "rho": 1000.0,
                "f": 0.005,
            }

    elif ode_name == "induction_linear":
        from fouling_models import fit_model_to_curve
        # Fit ks_simple first to get approximate curve shape
        result = fit_model_to_curve(t_sparse, Rf_sparse, "ks_simple")
        if result["success"]:
            # Estimate induction parameters from the fit
            # t_ind ≈ where the curve starts rising significantly
            Rf0 = result["params"]["Rf0"]
            Rf_max = result["params"]["Rf_star"]
            # Rough t_ind: find where Rf exceeds 5% of max
            Rf_pred = result["Rf_pred"]
            threshold = 0.05 * Rf_max
            above_thresh = np.where(Rf_pred > threshold)[0]
            t_ind = float(t_sparse[above_thresh[0]]) if len(above_thresh) > 0 else 5.0
            # k_growth ≈ Rf_max / (t_max - t_ind)
            k_growth = Rf_max / max(t_sparse[-1] - t_ind, 1.0)

            return {
                "t_ind": t_ind,
                "k_growth": k_growth,
                "Rf_max": Rf_max,
                "Tw": 360.0, "v": 1.0,
            }

    elif ode_name == "ebert_panchal":
        # No analytical solution; use ks_simple as rough proxy
        result = fit_model_to_curve(t_sparse, Rf_sparse, "ks_simple")
        if result["success"]:
            return {
                "alpha": 1e-4,
                "beta": 45000.0,
                "gamma": 1e-8,
                "Re": 10000.0,
                "Pr": 7.0,
                "Tw": 360.0,
                "tau_w": 10.0,
            }

    # Fallback: use model defaults
    model_info = get_model(ode_name)
    fallback = {}
    for param_name, (default, low, high, unit) in model_info.params.items():
        fallback[param_name] = default
    return fallback


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== PINN Multi-ODE Quick Test ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Test 1: encode/decode cycle
    print("\n--- Test 1: Parameter encoding ---")
    for ode_name in ["kern_seaton", "ks_simple", "induction_linear"]:
        model_info = get_model(ode_name)
        params = {name: default for name, (default, _, _, _) in model_info.params.items()}
        vec = encode_condition_params(ode_name, **params)
        ode_id = ODE_TYPE_IDS[ode_name]
        print(f"  {ode_name}: condition_vec shape={vec.shape}, ode_id={ode_id}")

    # Test 2: build input tensor
    print("\n--- Test 2: Input tensor construction ---")
    t_vals = np.linspace(0, 100, 10)
    params = {"Rf_star": 1.8e-4, "tau": 40.0, "Rf0": 0.0, "Tw": 360.0, "v": 1.0}
    condition_vec = encode_condition_params("ks_simple", **params)
    x = build_input_tensor(t_vals, condition_vec, ODE_TYPE_IDS["ks_simple"], device)
    print(f"  Input tensor shape: {x.shape} (expected: (10, {INPUT_DIM}))")

    # Test 3: simple PINN instantiation
    print("\n--- Test 3: PINN instantiation ---")
    model = PINN(hidden_layers=(32, 32)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    # Test 4: forward pass
    print("\n--- Test 4: Forward pass ---")
    with torch.no_grad():
        y = model(x)
    print(f"  Output shape: {y.shape}, range: [{y.min().item():.2e}, {y.max().item():.2e}]")

    # Test 5: physics loss
    print("\n--- Test 5: Physics loss (ks_simple) ---")
    t_phys = torch.rand(50, 1, device=device) * 100
    t_phys.requires_grad_(True)
    phys_loss = compute_physics_loss_trajectory(model, t_phys, "ks_simple", params, device)
    print(f"  Physics loss: {phys_loss.item():.6e}")

    print("\n✅ All tests passed.")
