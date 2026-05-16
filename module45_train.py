# -*- coding: utf-8 -*-
'''
@File    :   module45_train.py
@Time    :   2026-04-25
@Desc    :   This file implements modules 4 and 5: FiLM spatiotemporal coupling, a 6-parameter white-box ODE decoder, end-to-end training, spatial extrapolation evaluation, and result export. Training and inference use "mini-batch grid parallelism": keep the time dimension T fully vectorized within each batch, avoiding sending all grids to the GPU at once, enabling stable runs on 4GB VRAM. The file includes AMP and tqdm progress bars, supports saving best/last checkpoints, exporting test metrics, and exporting full parameter-field npz outputs. Figure export follows project rules: Arial font, JPG(800DPI) + PDF(Type42), and keeps PNG.
@Notice  :   The training target is always Y_SM (true soil moisture); dynamic inputs keep only Pre, LST, PET, LAI; SM lags are not used.
'''

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "Arial"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from module23_model import CausalGatedMamba, GeoHypernet


@torch.jit.script
def _closed_loop_step(
    sm_t: torch.Tensor,
    cap_t: torch.Tensor,
    recharge_base_t: torch.Tensor,
    evap_base_t: torch.Tensor,
    anthro_base_t: torch.Tensor,
    dsm_corr_t: torch.Tensor,
    use_dsm_clip: bool,
    dsm_clip_value: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # All input tensors are 1D here: (B,)
    sm_frac_t = torch.clamp(sm_t / (cap_t + 1e-6), min=0.0, max=1.0)

    recharge_eff_t = recharge_base_t * (1.0 - sm_frac_t)
    evap_eff_t = evap_base_t * sm_frac_t
    anthro_eff_t = anthro_base_t * sm_frac_t

    dsm_phys_t = recharge_eff_t - evap_eff_t - anthro_eff_t

    if use_dsm_clip:
        dsm_phys_t = dsm_clip_value * torch.tanh(dsm_phys_t / dsm_clip_value)

    dsm_dt_t = dsm_phys_t + dsm_corr_t
    sm_next = sm_t + dsm_dt_t

    return sm_next, sm_frac_t, dsm_phys_t, dsm_dt_t


@torch.jit.script
def closed_loop_integrate_jit(
    sm_0: torch.Tensor,
    sm_capacity: torch.Tensor,
    recharge_base: torch.Tensor,
    evap_base: torch.Tensor,
    anthro_base: torch.Tensor,
    dSM_corr: torch.Tensor,
    use_dsm_clip: bool,
    dsm_clip_value: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B = sm_capacity.size(0)
    T = sm_capacity.size(1)

    # Performance core: preallocate contiguous 2D tensors (B, T) to avoid heavy 3D slicing graphs
    sm_pred = torch.empty((B, T), device=sm_capacity.device, dtype=sm_capacity.dtype)
    sm_frac_out = torch.empty((B, T), device=sm_capacity.device, dtype=sm_capacity.dtype)
    dsm_phys_out = torch.empty((B, T), device=sm_capacity.device, dtype=sm_capacity.dtype)
    dsm_dt_out = torch.empty((B, T), device=sm_capacity.device, dtype=sm_capacity.dtype)

    sm_t = sm_0

    for t in range(T):
        # Performance core: use [:, t] scalar slices to avoid SliceBackward nodes
        sm_t, sm_frac_t, dsm_phys_t, dsm_dt_t = _closed_loop_step(
            sm_t,
            sm_capacity[:, t],
            recharge_base[:, t],
            evap_base[:, t],
            anthro_base[:, t],
            dSM_corr[:, t],
            use_dsm_clip,
            dsm_clip_value
        )

        # Record results
        sm_pred[:, t] = sm_t
        sm_frac_out[:, t] = sm_frac_t
        dsm_phys_out[:, t] = dsm_phys_t
        dsm_dt_out[:, t] = dsm_dt_t

    return sm_pred, sm_frac_out, dsm_phys_out, dsm_dt_out


class ODEDecoder(nn.Module):
    """Module 4: FiLM + parameterized white-box ODE decoder."""

    def __init__(
        self,
        d_hidden: int,
        alpha_scale: float = 0.4,
        theta_scale: float = 0.4,
        eta_pet_scale: float = 0.4,
        phi_lai_scale: float = 2.0,
        use_dsm_clip: bool = False,
        dsm_clip_value: float = 0.2,
        enable_delta_residual: bool = True,
        delta_residual_scale: float = 0.2,
        enable_sm_residual: bool = False,
        sm_residual_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.alpha_scale = float(alpha_scale)
        self.theta_scale = float(theta_scale)
        self.eta_pet_scale = float(eta_pet_scale)
        self.phi_lai_scale = float(phi_lai_scale)
        self.use_dsm_clip = bool(use_dsm_clip)
        self.dsm_clip_value = float(dsm_clip_value)
        self.enable_delta_residual = bool(enable_delta_residual)
        self.delta_residual_scale = float(delta_residual_scale)
        self.enable_sm_residual = bool(enable_sm_residual)
        self.sm_residual_scale = float(sm_residual_scale)

        self.phys_head = nn.Linear(d_hidden, 8)
        self.sm_capacity_head = nn.Linear(d_hidden, 1)
        self.delta_residual_head = nn.Linear(d_hidden, 1)
        self.sm_residual_head = nn.Linear(d_hidden, 1)

        nn.init.zeros_(self.sm_capacity_head.weight)
        nn.init.zeros_(self.sm_capacity_head.bias)
        nn.init.zeros_(self.delta_residual_head.weight)
        nn.init.zeros_(self.delta_residual_head.bias)
        nn.init.zeros_(self.sm_residual_head.weight)
        nn.init.zeros_(self.sm_residual_head.bias)

    def forward(
        self,
        H_dyn: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        X_dyn: torch.Tensor,
        X_hum: torch.Tensor,
        SM_0: torch.Tensor,
        return_diagnostics: bool = True,
    ) -> Dict[str, torch.Tensor]:
        # 1) FiLM coupling
        H_coupled = gamma.unsqueeze(1) * H_dyn + beta.unsqueeze(1)  # (B,T,D_hidden)

        # 2) Precompute outside the loop (never call NN layers inside loops)
        phys_raw = self.phys_head(H_coupled)  # (B,T,8)
        phys = torch.sigmoid(phys_raw)
        alpha = self.alpha_scale * phys[..., 0:1]
        theta = self.theta_scale * phys[..., 1:2]
        eta_pet = self.eta_pet_scale * phys[..., 2:3]
        phi_lai = self.phi_lai_scale * phys[..., 3:4]
        kappa_irr = phys[..., 4:5]
        kappa_manu = phys[..., 5:6]
        kappa_dom = phys[..., 6:7]
        kappa_ele = phys[..., 7:8]

        sm_capacity = torch.clamp(F.softplus(self.sm_capacity_head(H_coupled)), min=1e-3)

        if self.enable_delta_residual:
            dSM_correction_all = self.delta_residual_scale * torch.tanh(self.delta_residual_head(H_coupled))
        else:
            dSM_correction_all = torch.zeros_like(H_coupled[..., :1])

        # 3) Precompute state-independent flux bases
        pre = X_dyn[..., 0:1]
        lst = X_dyn[..., 1:2]
        pet = X_dyn[..., 2:3]
        lai = X_dyn[..., 3:4]

        irr = X_hum[..., 0:1]
        manu = X_hum[..., 1:2]
        dom = X_hum[..., 2:3]
        ele = X_hum[..., 3:4]

        pre_eff = torch.log1p(torch.clamp(pre, min=0.0))
        lst_eff = torch.clamp(lst, min=0.0) / 30.0
        pet_eff = torch.log1p(torch.clamp(pet, min=0.0))
        lai_eff = torch.log1p(torch.clamp(lai, min=0.0))

        recharge_base_all = alpha * pre_eff
        evap_base_all = (theta * lst_eff + eta_pet * pet_eff) * (1.0 + phi_lai * lai_eff)
        anthropogenic_term_all = kappa_irr * irr + kappa_manu * manu + kappa_dom * dom + kappa_ele * ele

        # 4) Lightweight physical closed-loop integration (JIT optimized)
        # Squeeze to 2D (B, T) or 1D (B,) before JIT for maximum performance
        SM_ode, sm_frac, dSM_phys, dSM_dt = closed_loop_integrate_jit(
            sm_0=SM_0.squeeze(-1),
            sm_capacity=sm_capacity.squeeze(-1),
            recharge_base=recharge_base_all.squeeze(-1),
            evap_base=evap_base_all.squeeze(-1),
            anthro_base=anthropogenic_term_all.squeeze(-1),
            dSM_corr=dSM_correction_all.squeeze(-1),
            use_dsm_clip=self.use_dsm_clip,
            dsm_clip_value=self.dsm_clip_value
        )

        # Restore to 3D tensors (B, T, 1) for downstream residual calculations
        SM_ode = SM_ode.unsqueeze(-1)
        sm_frac = sm_frac.unsqueeze(-1)
        dSM_phys = dSM_phys.unsqueeze(-1)
        dSM_dt = dSM_dt.unsqueeze(-1)
        dSM_correction = dSM_correction_all

        if self.enable_sm_residual:
            SM_correction = self.sm_residual_scale * torch.tanh(self.sm_residual_head(H_coupled))
        else:
            SM_correction = torch.zeros_like(SM_ode)

        SM_pred = SM_ode + SM_correction

        if not return_diagnostics:
            return {"SM_pred": SM_pred}

        return {
            "H_coupled": H_coupled,
            "alpha": alpha,
            "theta": theta,
            "eta_pet": eta_pet,
            "phi_lai": phi_lai,
            "kappa_irr": kappa_irr,
            "kappa_manu": kappa_manu,
            "kappa_dom": kappa_dom,
            "kappa_ele": kappa_ele,
            "sm_frac": sm_frac,
            "dSM_phys": dSM_phys,
            "dSM_correction": dSM_correction,
            "dSM_dt": dSM_dt,
            "SM_ode": SM_ode,
            "SM_correction": SM_correction,
            "SM_pred": SM_pred,
        }

class CausalMambaODE(nn.Module):
    """Modules 2-5 combined: GeoHypernet + CausalGatedMamba + ODEDecoder."""

    def __init__(
        self,
        d_hidden: int = 64,
        n_dyn: int = 4,
        n_time_enc: int = 9,
        kan_num_basis: int = 8,
        dropout: float = 0.1,
        use_kan: bool = True,
        film_gamma_mode: str = "legacy_tanh",
        mamba_static_context_dim: int = 16,
        use_soft_causal_gating: bool = True,
        causal_gate_floor: float = 0.2,
        inject_sm0_to_mamba: bool = True,
        normalize_dynamic_for_encoder: bool = True,
        ode_param_scale_alpha: float = 0.4,
        ode_param_scale_theta: float = 0.4,
        ode_param_scale_eta_pet: float = 0.4,
        ode_param_scale_phi_lai: float = 2.0,
        ode_use_dsm_clip: bool = False,
        ode_dsm_clip_value: float = 0.2,
        ode_enable_delta_residual: bool = True,
        ode_delta_residual_scale: float = 0.2,
        ode_enable_sm_residual: bool = False,
        ode_sm_residual_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.use_dyn_norm_for_encoder = bool(normalize_dynamic_for_encoder)
        self.geo_hypernet = GeoHypernet(
            d_hidden=d_hidden,
            num_basis=kan_num_basis,
            dropout=dropout,
            use_kan=use_kan,
            film_gamma_mode=film_gamma_mode,
        )
        self.dynamic_encoder = CausalGatedMamba(
            n_dyn=n_dyn,
            n_time_enc=n_time_enc,
            d_hidden=d_hidden,
            dropout=dropout,
            static_context_dim=mamba_static_context_dim,
            use_soft_causal_gating=use_soft_causal_gating,
            causal_gate_floor=causal_gate_floor,
            inject_sm0=inject_sm0_to_mamba,
            use_kan=use_kan,
            normalize_dynamic_input=normalize_dynamic_for_encoder,
        )
        self.ode_decoder = ODEDecoder(
            d_hidden=d_hidden,
            alpha_scale=ode_param_scale_alpha,
            theta_scale=ode_param_scale_theta,
            eta_pet_scale=ode_param_scale_eta_pet,
            phi_lai_scale=ode_param_scale_phi_lai,
            use_dsm_clip=ode_use_dsm_clip,
            dsm_clip_value=ode_dsm_clip_value,
            enable_delta_residual=ode_enable_delta_residual,
            delta_residual_scale=ode_delta_residual_scale,
            enable_sm_residual=ode_enable_sm_residual,
            sm_residual_scale=ode_sm_residual_scale,
        )
        self.sm0_fallback = nn.Parameter(torch.tensor([0.2], dtype=torch.float32))

    def forward(
        self,
        X_sta: torch.Tensor,
        X_loc_enc: torch.Tensor,
        Cluster_onehot: torch.Tensor,
        X_dyn: torch.Tensor,
        W_causal: torch.Tensor,
        X_time_enc: torch.Tensor,
        X_hum: torch.Tensor,
        X_dyn_norm: Optional[torch.Tensor] = None,
        SM_0: Optional[torch.Tensor] = None,
        return_diagnostics: bool = True,
    ) -> Dict[str, torch.Tensor]:
        gamma, beta, X_static_all = self.geo_hypernet(X_sta, X_loc_enc, Cluster_onehot)
        if SM_0 is None:
            SM_0 = self.sm0_fallback.expand(X_dyn.size(0), 1)
        dyn_for_encoder = X_dyn_norm if (self.use_dyn_norm_for_encoder and X_dyn_norm is not None) else X_dyn
        H_dyn, X_dyn_gated = self.dynamic_encoder(
            dyn_for_encoder,
            W_causal,
            X_time_enc,
            X_static_all=X_static_all,
            SM_0=SM_0,
        )

        ode_out = self.ode_decoder(
            H_dyn,
            gamma,
            beta,
            X_dyn,
            X_hum,
            SM_0,
            return_diagnostics=return_diagnostics,
        )

        if not return_diagnostics:
            return ode_out

        return {
            "gamma": gamma,
            "beta": beta,
            "X_static_all": X_static_all,
            "X_dyn_gated": X_dyn_gated,
            "H_dyn": H_dyn,
            **ode_out,
        }

def _split_static_all(x_static_all: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split (B,28) into X_sta(8), X_loc(16), Cluster_onehot(4)."""
    x_sta = x_static_all[:, :8]
    x_loc = x_static_all[:, 8:24]
    x_cluster = x_static_all[:, 24:28]
    return x_sta, x_loc, x_cluster

def _unpack_batch_to_device(batch, device: torch.device):
    """Move DataLoader batch to device and split static features."""
    y_sm, x_static_all, x_dyn, x_dyn_norm, x_time_enc, w_causal, x_hum = batch
    y_sm = y_sm.float().to(device, non_blocking=True)
    x_static_all = x_static_all.float().to(device, non_blocking=True)
    x_dyn = x_dyn.float().to(device, non_blocking=True)
    x_dyn_norm = x_dyn_norm.float().to(device, non_blocking=True)
    x_time_enc = x_time_enc.float().to(device, non_blocking=True)
    w_causal = w_causal.float().to(device, non_blocking=True)
    x_hum = x_hum.float().to(device, non_blocking=True)

    x_sta, x_loc, x_cluster = _split_static_all(x_static_all)
    sm_0 = y_sm[:, 0, :]  # shape: (B,1)
    return y_sm, x_sta, x_loc, x_cluster, x_dyn, x_dyn_norm, x_time_enc, w_causal, x_hum, sm_0

def _slice_supervised_steps(y_true: torch.Tensor, y_pred: torch.Tensor, start_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drop warm-up steps that are provided as known initial conditions."""
    if start_idx <= 0 or y_true.size(1) <= start_idx:
        return y_true, y_pred
    return y_true[:, start_idx:, :], y_pred[:, start_idx:, :]

def _trajectory_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    criterion: nn.Module,
    start_idx: int,
    derivative_loss_weight: float,
    derivative_smooth_window: int = 0,
) -> torch.Tensor:
    y_level, pred_level = _slice_supervised_steps(y_true, y_pred, start_idx)
    loss = criterion(pred_level, y_level)
    if derivative_loss_weight > 0.0 and y_true.size(1) > 1:
        if derivative_smooth_window > 1:
            pad = derivative_smooth_window // 2
            y_true_ch = y_true.transpose(1, 2)
            y_true_smooth = F.avg_pool1d(y_true_ch, kernel_size=derivative_smooth_window, stride=1, padding=pad)
            if y_true_smooth.size(-1) > y_true_ch.size(-1):
                y_true_smooth = y_true_smooth[..., : y_true_ch.size(-1)]
            y_true_used = y_true_smooth.transpose(1, 2)
        else:
            y_true_used = y_true
        true_delta = y_true_used[:, 1:, :] - y_true_used[:, :-1, :]
        pred_delta = y_pred[:, 1:, :] - y_pred[:, :-1, :]
        delta_start = max(start_idx - 1, 0)
        true_delta, pred_delta = _slice_supervised_steps(true_delta, pred_delta, delta_start)
        loss = loss + derivative_loss_weight * criterion(pred_delta, true_delta)
    return loss

def _compute_metrics_vectorized(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> Dict[str, torch.Tensor]:
    """Vectorized R2/KGE/RMSE/ubRMSE."""
    if y_true.dim() == 3:
        y_true = y_true.squeeze(-1)
    if y_pred.dim() == 3:
        y_pred = y_pred.squeeze(-1)

    err = y_pred - y_true
    mse_g = torch.mean(err * err, dim=1)
    rmse_g = torch.sqrt(mse_g + eps)

    bias = torch.mean(err, dim=1, keepdim=True)
    ub_rmse_g = torch.sqrt(torch.mean((err - bias) ** 2, dim=1) + eps)

    y_t_mean = torch.mean(y_true, dim=1, keepdim=True)
    y_p_mean = torch.mean(y_pred, dim=1, keepdim=True)

    ss_res = torch.sum((y_pred - y_true) ** 2, dim=1)
    ss_tot = torch.sum((y_true - y_t_mean) ** 2, dim=1)
    r2_g = 1.0 - ss_res / (ss_tot + eps)

    y_t_centered = y_true - y_t_mean
    y_p_centered = y_pred - y_p_mean
    cov = torch.mean(y_t_centered * y_p_centered, dim=1)
    std_t = torch.std(y_true, dim=1, unbiased=False)
    std_p = torch.std(y_pred, dim=1, unbiased=False)
    r = cov / (std_t * std_p + eps)

    alpha = std_p / (std_t + eps)
    beta = y_p_mean.squeeze(1) / (y_t_mean.squeeze(1) + eps)
    kge_g = 1.0 - torch.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2 + eps)

    y_t_all = y_true.reshape(-1)
    y_p_all = y_pred.reshape(-1)
    e_all = y_p_all - y_t_all

    rmse_all = torch.sqrt(torch.mean(e_all * e_all) + eps)
    ub_rmse_all = torch.sqrt(torch.mean((e_all - torch.mean(e_all)) ** 2) + eps)

    y_t_all_mean = torch.mean(y_t_all)
    ss_res_all = torch.sum((y_p_all - y_t_all) ** 2)
    ss_tot_all = torch.sum((y_t_all - y_t_all_mean) ** 2)
    r2_all = 1.0 - ss_res_all / (ss_tot_all + eps)

    y_p_all_mean = torch.mean(y_p_all)
    cov_all = torch.mean((y_t_all - y_t_all_mean) * (y_p_all - y_p_all_mean))
    std_t_all = torch.std(y_t_all, unbiased=False)
    std_p_all = torch.std(y_p_all, unbiased=False)
    r_all = cov_all / (std_t_all * std_p_all + eps)
    alpha_all = std_p_all / (std_t_all + eps)
    beta_all = y_p_all_mean / (y_t_all_mean + eps)
    kge_all = 1.0 - torch.sqrt((r_all - 1.0) ** 2 + (alpha_all - 1.0) ** 2 + (beta_all - 1.0) ** 2 + eps)

    return {
        "r2_grid": r2_g,
        "kge_grid": kge_g,
        "rmse_grid": rmse_g,
        "ubrmse_grid": ub_rmse_g,
        "r2_global": r2_all,
        "kge_global": kge_all,
        "rmse_global": rmse_all,
        "ubrmse_global": ub_rmse_all,
    }

def _plot_loss_curves(train_losses: List[float], val_losses: List[float], output_dir: Path) -> None:
    """Export loss curves: PNG + JPG(800DPI) + PDF(Type42)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training/Validation Loss Curves")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_dir / "loss_curves.png", dpi=300)
    plt.savefig(output_dir / "loss_curves.jpg", dpi=800)
    plt.savefig(output_dir / "loss_curves.pdf")
    plt.close()

def _infer_loader_outputs(
    model: CausalMambaODE,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    return_full: bool = False,
):
    """Run inference on a loader and concatenate outputs."""
    y_true_list = []
    y_pred_list = []
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    alpha_list = []
    theta_list = []
    eta_pet_list = []
    phi_lai_list = []
    ki_list = []
    km_list = []
    kd_list = []
    ke_list = []

    model.eval()
    with torch.inference_mode():
        pbar = tqdm(loader, desc="Inference", leave=False)
        for batch in pbar:
            y_sm, x_sta, x_loc, x_cluster, x_dyn, x_dyn_norm, x_time, w_causal, x_hum, sm_0 = _unpack_batch_to_device(batch, device)
            with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
                out = model(
                    X_sta=x_sta,
                    X_loc_enc=x_loc,
                    Cluster_onehot=x_cluster,
                    X_dyn=x_dyn,
                    X_dyn_norm=x_dyn_norm,
                    W_causal=w_causal,
                    X_time_enc=x_time,
                    X_hum=x_hum,
                    SM_0=sm_0,
                    return_diagnostics=return_full,
                )
            y_true_list.append(y_sm.detach().cpu())
            y_pred_list.append(out["SM_pred"].detach().cpu())

            if return_full:
                alpha_list.append(out["alpha"].detach().cpu())
                theta_list.append(out["theta"].detach().cpu())
                eta_pet_list.append(out["eta_pet"].detach().cpu())
                phi_lai_list.append(out["phi_lai"].detach().cpu())
                ki_list.append(out["kappa_irr"].detach().cpu())
                km_list.append(out["kappa_manu"].detach().cpu())
                kd_list.append(out["kappa_dom"].detach().cpu())
                ke_list.append(out["kappa_ele"].detach().cpu())

    y_true = torch.cat(y_true_list, dim=0)
    y_pred = torch.cat(y_pred_list, dim=0)

    if not return_full:
        return {"Y_true": y_true, "SM_pred": y_pred}

    return {
        "Y_true": y_true,
        "SM_pred": y_pred,
        "alpha": torch.cat(alpha_list, dim=0),
        "theta": torch.cat(theta_list, dim=0),
        "eta_pet": torch.cat(eta_pet_list, dim=0),
        "phi_lai": torch.cat(phi_lai_list, dim=0),
        "kappa_irr": torch.cat(ki_list, dim=0),
        "kappa_manu": torch.cat(km_list, dim=0),
        "kappa_dom": torch.cat(kd_list, dim=0),
        "kappa_ele": torch.cat(ke_list, dim=0),
    }


def _export_full_outputs_streaming(
    model: CausalMambaODE,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    parquet_path: Optional[Path],
    mask_dict: Dict[str, torch.Tensor],
    n_grid: int,
) -> None:
    """Stream full outputs by batch to avoid keeping all tensors in memory."""
    arrays: Dict[str, np.ndarray] = {}
    cursor = 0
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    model.eval()
    with torch.inference_mode():
        pbar = tqdm(loader, desc="ExportFull", leave=False)
        for batch in pbar:
            y_sm, x_sta, x_loc, x_cluster, x_dyn, x_dyn_norm, x_time, w_causal, x_hum, sm_0 = _unpack_batch_to_device(batch, device)
            with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
                out = model(
                    X_sta=x_sta,
                    X_loc_enc=x_loc,
                    Cluster_onehot=x_cluster,
                    X_dyn=x_dyn,
                    X_dyn_norm=x_dyn_norm,
                    W_causal=w_causal,
                    X_time_enc=x_time,
                    X_hum=x_hum,
                    SM_0=sm_0,
                    return_diagnostics=True,
                )

            batch_np = {
                "alpha": out["alpha"].detach().cpu().numpy().astype(np.float32),
                "theta": out["theta"].detach().cpu().numpy().astype(np.float32),
                "eta_pet": out["eta_pet"].detach().cpu().numpy().astype(np.float32),
                "phi_lai": out["phi_lai"].detach().cpu().numpy().astype(np.float32),
                "kappa_irr": out["kappa_irr"].detach().cpu().numpy().astype(np.float32),
                "kappa_manu": out["kappa_manu"].detach().cpu().numpy().astype(np.float32),
                "kappa_dom": out["kappa_dom"].detach().cpu().numpy().astype(np.float32),
                "kappa_ele": out["kappa_ele"].detach().cpu().numpy().astype(np.float32),
                "SM_pred": out["SM_pred"].detach().cpu().numpy().astype(np.float32),
                "Y_true": y_sm.detach().cpu().numpy().astype(np.float32),
            }

            if not arrays:
                for k, v in batch_np.items():
                    arrays[k] = np.empty((n_grid, *v.shape[1:]), dtype=np.float32)

            bsz = next(iter(batch_np.values())).shape[0]
            end = cursor + bsz
            for k, v in batch_np.items():
                arrays[k][cursor:end] = v
            cursor = end

    dataset_mask = np.full((n_grid,), -1, dtype=np.int64)
    dataset_mask[mask_dict["train"].cpu().numpy()] = 0
    dataset_mask[mask_dict["val"].cpu().numpy()] = 1
    dataset_mask[mask_dict["test"].cpu().numpy()] = 2

    if parquet_path is not None:
        _export_full_outputs_parquet(arrays=arrays, dataset_mask=dataset_mask, output_path=parquet_path)


def _export_daily_outputs_streaming(
    model: CausalMambaODE,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    parquet_path: Optional[Path],
    mask_dict: Dict[str, torch.Tensor],
    n_grid: int,
) -> None:
    """Export one row per grid-day with IDs, dates, states, coefficients, and flux terms."""
    if parquet_path is None:
        return
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = loader.dataset
    grid_ids = getattr(dataset, "Grid_ID", torch.arange(n_grid, dtype=torch.long)).cpu().numpy().astype(np.int64)
    date_values = np.asarray(getattr(dataset, "Date", np.arange(dataset.Y_SM.shape[1], dtype=np.int64)))
    t_len = int(dataset.Y_SM.shape[1])
    split_code = np.full((n_grid,), -1, dtype=np.int8)
    split_code[mask_dict["train"].cpu().numpy()] = 0
    split_code[mask_dict["val"].cpu().numpy()] = 1
    split_code[mask_dict["test"].cpu().numpy()] = 2

    cursor = 0
    writer: Optional[pq.ParquetWriter] = None
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    def _flat(value: torch.Tensor) -> np.ndarray:
        return value.detach().cpu().numpy().astype(np.float32).squeeze(-1).reshape(-1)

    try:
        model.eval()
        with torch.inference_mode():
            pbar = tqdm(loader, desc="ExportDaily", leave=False)
            for batch in pbar:
                y_sm, x_sta, x_loc, x_cluster, x_dyn, x_dyn_norm, x_time, w_causal, x_hum, sm_0 = _unpack_batch_to_device(batch, device)
                with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
                    out = model(
                        X_sta=x_sta,
                        X_loc_enc=x_loc,
                        Cluster_onehot=x_cluster,
                        X_dyn=x_dyn,
                        X_dyn_norm=x_dyn_norm,
                        W_causal=w_causal,
                        X_time_enc=x_time,
                        X_hum=x_hum,
                        SM_0=sm_0,
                        return_diagnostics=True,
                    )

                bsz = y_sm.size(0)
                end = cursor + bsz
                x_dyn_cpu = x_dyn.detach().cpu().numpy().astype(np.float32)
                w_cpu = w_causal.detach().cpu().numpy().astype(np.float32)
                x_hum_cpu = x_hum.detach().cpu().numpy().astype(np.float32)

                table = pa.Table.from_pydict(
                    {
                        "Grid_ID": np.repeat(grid_ids[cursor:end], t_len),
                        "Date": np.tile(date_values.astype(str), bsz),
                        "time_index": np.tile(np.arange(t_len, dtype=np.int32), bsz),
                        "dataset_split": np.repeat(split_code[cursor:end], t_len),
                        "Y_true": _flat(y_sm),
                        "SM_pred": _flat(out["SM_pred"]),
                        "SM_ode": _flat(out.get("SM_ode", out["SM_pred"])),
                        "SM_correction": _flat(out.get("SM_correction", torch.zeros_like(out["SM_pred"]))),
                        "dSM_dt": _flat(out["dSM_dt"]),
                        "dSM_phys": _flat(out.get("dSM_phys", out["dSM_dt"])),
                        "dSM_correction": _flat(out.get("dSM_correction", torch.zeros_like(out["dSM_dt"]))),
                        "alpha": _flat(out["alpha"]),
                        "theta": _flat(out["theta"]),
                        "eta_pet": _flat(out["eta_pet"]),
                        "phi_lai": _flat(out["phi_lai"]),
                        "sm_frac": _flat(out.get("sm_frac", torch.zeros_like(out["SM_pred"]))),
                        "kappa_irr": _flat(out["kappa_irr"]),
                        "kappa_manu": _flat(out["kappa_manu"]),
                        "kappa_dom": _flat(out["kappa_dom"]),
                        "kappa_ele": _flat(out["kappa_ele"]),
                        "Pre": x_dyn_cpu[:, :, 0].reshape(-1),
                        "LST": x_dyn_cpu[:, :, 1].reshape(-1),
                        "PET": x_dyn_cpu[:, :, 2].reshape(-1),
                        "LAI": x_dyn_cpu[:, :, 3].reshape(-1),
                        "W_Pre": w_cpu[:, :, 0].reshape(-1),
                        "W_LST": w_cpu[:, :, 1].reshape(-1),
                        "W_PET": w_cpu[:, :, 2].reshape(-1),
                        "W_LAI": w_cpu[:, :, 3].reshape(-1),
                        "irr": x_hum_cpu[:, :, 0].reshape(-1),
                        "manu": x_hum_cpu[:, :, 1].reshape(-1),
                        "dom": x_hum_cpu[:, :, 2].reshape(-1),
                        "ele": x_hum_cpu[:, :, 3].reshape(-1),
                    }
                )
                if writer is None:
                    writer = pq.ParquetWriter(parquet_path, table.schema, compression="zstd")
                writer.write_table(table)
                cursor = end
    finally:
        if writer is not None:
            writer.close()


def _export_full_outputs_parquet(
    arrays: Dict[str, np.ndarray],
    dataset_mask: np.ndarray,
    output_path: Path,
) -> None:
    """Export full results to parquet for pandas/pyarrow."""
    if "SM_pred" not in arrays or "Y_true" not in arrays:
        raise ValueError("Parquet export requires SM_pred and Y_true.")

    sm_pred = np.asarray(arrays["SM_pred"], dtype=np.float32).squeeze(-1)
    y_true = np.asarray(arrays["Y_true"], dtype=np.float32).squeeze(-1)
    alpha = np.asarray(arrays["alpha"], dtype=np.float32).squeeze(-1)
    theta = np.asarray(arrays["theta"], dtype=np.float32).squeeze(-1)
    eta_pet = np.asarray(arrays["eta_pet"], dtype=np.float32).squeeze(-1)
    phi_lai = np.asarray(arrays["phi_lai"], dtype=np.float32).squeeze(-1)
    kappa_irr = np.asarray(arrays["kappa_irr"], dtype=np.float32).squeeze(-1)
    kappa_manu = np.asarray(arrays["kappa_manu"], dtype=np.float32).squeeze(-1)
    kappa_dom = np.asarray(arrays["kappa_dom"], dtype=np.float32).squeeze(-1)
    kappa_ele = np.asarray(arrays["kappa_ele"], dtype=np.float32).squeeze(-1)

    n_grid, t_len = sm_pred.shape
    if y_true.shape != (n_grid, t_len):
        raise ValueError("SM_pred and Y_true shapes do not match.")

    def _fixed_list_column(value: np.ndarray) -> pa.Array:
        flat = np.ascontiguousarray(value.reshape(-1), dtype=np.float32)
        return pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float32()), t_len)

    table = pa.Table.from_arrays(
        [
            pa.array(np.arange(n_grid, dtype=np.int64)),
            pa.array(dataset_mask.astype(np.int64)),
            _fixed_list_column(alpha),
            _fixed_list_column(theta),
            _fixed_list_column(eta_pet),
            _fixed_list_column(phi_lai),
            _fixed_list_column(kappa_irr),
            _fixed_list_column(kappa_manu),
            _fixed_list_column(kappa_dom),
            _fixed_list_column(kappa_ele),
            _fixed_list_column(sm_pred),
            _fixed_list_column(y_true),
        ],
        names=[
            "grid_index",
            "dataset_mask",
            "alpha",
            "theta",
            "eta_pet",
            "phi_lai",
            "kappa_irr",
            "kappa_manu",
            "kappa_dom",
            "kappa_ele",
            "SM_pred",
            "Y_true",
        ],
    )
    pq.write_table(table, output_path, compression="zstd")

def train_and_evaluate(
    model: CausalMambaODE,
    dataset,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    mask_dict: Dict[str, torch.Tensor],
    output_dir: Path,
    batch_size: int = 96,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    max_epochs: int = 30,
    early_stop_patience: int = 6,
    resume_from_best: bool = False,
    resume_optimizer_state: bool = True,
    overfit_patience: int = 2,
    overfit_val_rise_ratio: float = 0.08,
    overfit_train_drop_min_ratio: float = 1e-3,
    test_eval_interval: int = 0,
    loss_start_idx: int = 1,
    derivative_loss_weight: float = 0.0,
    derivative_smooth_window: int = 0,
    device: Optional[str] = None,
    use_amp: bool = True,
    gradient_clip_norm: float = 1.0,
    lr_scheduler_patience: int = 2,
    lr_scheduler_factor: float = 0.5,
    min_lr: float = 1e-5,
    checkpoint_version: int = 8,
    export_daily_results: bool = False,
) -> Dict[str, float]:
    """Module 5: training, spatial extrapolation evaluation, and full export."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    autocast_device = "cuda" if dev.type == "cuda" else "cpu"

    effective_batch_size = batch_size
    effective_eval_batch_size = int(eval_batch_size) if eval_batch_size is not None else max(batch_size, batch_size * 2)
    amp_enabled = use_amp and (dev.type == "cuda")
    if dev.type == "cuda":
        total_mem_gb = torch.cuda.get_device_properties(dev).total_memory / (1024**3)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        print(
            "CUDA acceleration:",
            {
                "device": torch.cuda.get_device_name(dev),
                "total_mem_gb": round(total_mem_gb, 2),
                "batch_size": effective_batch_size,
                "eval_batch_size": effective_eval_batch_size,
                "amp_enabled": amp_enabled,
                "tf32_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
            },
        )

    model = model.to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
        min_lr=min_lr,
    )
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled) if dev.type == "cuda" else None

    best_val = float("inf")
    wait = 0
    overfit_wait = 0
    prev_train_loss: Optional[float] = None
    train_losses: List[float] = []
    val_losses: List[float] = []
    last_test_r2 = float("nan")

    best_model_path = output_dir / "mamba_ode_best.pth"
    last_model_path = output_dir / "mamba_ode_last.pth"

    def _checkpoint_payload(epoch_idx: int, train_epoch_loss: float, val_epoch_loss: float) -> Dict[str, object]:
        return {
            "checkpoint_version": checkpoint_version,
            "epoch": epoch_idx,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_epoch_loss,
            "val_loss": val_epoch_loss,
            "amp_enabled": amp_enabled,
            "effective_batch_size": effective_batch_size,
            "effective_eval_batch_size": effective_eval_batch_size,
            "loss_start_idx": loss_start_idx,
            "derivative_loss_weight": derivative_loss_weight,
        }

    if resume_from_best and best_model_path.exists():
        resume_ckpt = torch.load(best_model_path, map_location=dev)
        ckpt_version = int(resume_ckpt.get("checkpoint_version", -1))
        if ckpt_version != checkpoint_version:
            print(
                "Detected an old best checkpoint; auto-resume skipped to avoid locking accuracy with old labels/structure.",
                {
                    "path": str(best_model_path),
                    "ckpt_version": ckpt_version,
                    "expected_version": checkpoint_version,
                },
            )
        else:
            try:
                model.load_state_dict(resume_ckpt["model_state_dict"])
            except RuntimeError as e:
                print(
                    "Best checkpoint is incompatible with the current model; resume init skipped.",
                    {
                        "path": str(best_model_path),
                        "reason": str(e),
                    },
                )
            else:
                if resume_optimizer_state and "optimizer_state_dict" in resume_ckpt:
                    optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
                    if "scheduler_state_dict" in resume_ckpt:
                        scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
                ckpt_val = float(resume_ckpt.get("val_loss", float("inf")))
                if np.isfinite(ckpt_val):
                    best_val = min(best_val, ckpt_val)
                print(
                    "Detected historical best checkpoint; resume initialized:",
                    {
                        "path": str(best_model_path),
                        "resume_optimizer_state": resume_optimizer_state,
                        "ckpt_val_loss": ckpt_val,
                    },
                )

    if getattr(train_loader, "batch_size", None) != effective_batch_size:
        train_kwargs = {
            "batch_size": effective_batch_size,
            "shuffle": True,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "collate_fn": getattr(train_loader, "collate_fn", None),
        }
        if num_workers > 0:
            train_kwargs["persistent_workers"] = True
        train_loader = DataLoader(train_loader.dataset, **train_kwargs)
    if getattr(val_loader, "batch_size", None) != effective_eval_batch_size:
        eval_kwargs = {
            "batch_size": effective_eval_batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "collate_fn": getattr(val_loader, "collate_fn", None),
        }
        if num_workers > 0:
            eval_kwargs["persistent_workers"] = True
        val_loader = DataLoader(val_loader.dataset, **eval_kwargs)
    if getattr(test_loader, "batch_size", None) != effective_eval_batch_size:
        eval_kwargs = {
            "batch_size": effective_eval_batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "collate_fn": getattr(test_loader, "collate_fn", None),
        }
        if num_workers > 0:
            eval_kwargs["persistent_workers"] = True
        test_loader = DataLoader(test_loader.dataset, **eval_kwargs)

    epoch_bar = tqdm(range(max_epochs), desc="Epoch", unit="ep")
    for epoch in epoch_bar:
        model.train()
        train_loss_sum_t = torch.zeros((), device=dev)
        train_count = 0

        train_iter = tqdm(train_loader, desc=f"Train {epoch + 1}/{max_epochs}", leave=False)
        for batch_step, batch in enumerate(train_iter, start=1):
            y_sm, x_sta, x_loc, x_cluster, x_dyn, x_dyn_norm, x_time, w_causal, x_hum, sm_0 = _unpack_batch_to_device(batch, dev)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=autocast_device, enabled=amp_enabled):
                out = model(
                    X_sta=x_sta,
                    X_loc_enc=x_loc,
                    Cluster_onehot=x_cluster,
                    X_dyn=x_dyn,
                    X_dyn_norm=x_dyn_norm,
                    W_causal=w_causal,
                    X_time_enc=x_time,
                    X_hum=x_hum,
                    SM_0=sm_0,
                    return_diagnostics=False,
                )
                loss = _trajectory_loss(
                    y_true=y_sm,
                    y_pred=out["SM_pred"],
                    criterion=criterion,
                    start_idx=loss_start_idx,
                    derivative_loss_weight=derivative_loss_weight,
                    derivative_smooth_window=derivative_smooth_window,
                )

            if not torch.isfinite(loss):
                print(f"Non-finite loss detected in epoch {epoch + 1}; skipped current batch.")
                continue

            if amp_enabled and scaler is not None:
                scaler.scale(loss).backward()
                if gradient_clip_norm > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if gradient_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()

            bsz = y_sm.size(0)
            train_loss_sum_t = train_loss_sum_t + loss.detach() * bsz
            train_count += bsz
            if batch_step == 1 or batch_step % 10 == 0:
                avg_train_loss = float((train_loss_sum_t / max(train_count, 1)).detach().cpu())
                train_iter.set_postfix(
                    loss=f"{avg_train_loss:.6f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

        train_epoch_loss = float((train_loss_sum_t / max(train_count, 1)).detach().cpu())
        train_losses.append(train_epoch_loss)

        model.eval()
        val_loss_sum_t = torch.zeros((), device=dev)
        val_count = 0
        val_true_list = []
        val_pred_list = []
        with torch.inference_mode():
            val_iter = tqdm(val_loader, desc=f"Val   {epoch + 1}/{max_epochs}", leave=False)
            for batch_step, batch in enumerate(val_iter, start=1):
                y_sm, x_sta, x_loc, x_cluster, x_dyn, x_dyn_norm, x_time, w_causal, x_hum, sm_0 = _unpack_batch_to_device(batch, dev)
                with torch.amp.autocast(device_type=autocast_device, enabled=amp_enabled):
                    out = model(
                        X_sta=x_sta,
                        X_loc_enc=x_loc,
                        Cluster_onehot=x_cluster,
                        X_dyn=x_dyn,
                        X_dyn_norm=x_dyn_norm,
                        W_causal=w_causal,
                        X_time_enc=x_time,
                        X_hum=x_hum,
                        SM_0=sm_0,
                        return_diagnostics=False,
                    )
                    vloss = _trajectory_loss(
                        y_true=y_sm,
                        y_pred=out["SM_pred"],
                        criterion=criterion,
                        start_idx=loss_start_idx,
                        derivative_loss_weight=derivative_loss_weight,
                        derivative_smooth_window=derivative_smooth_window,
                    )
                bsz = y_sm.size(0)
                val_loss_sum_t = val_loss_sum_t + vloss.detach() * bsz
                val_count += bsz
                val_true_list.append(y_sm.detach())
                val_pred_list.append(out["SM_pred"].detach())
                if batch_step == 1 or batch_step % 10 == 0:
                    avg_val_loss = float((val_loss_sum_t / max(val_count, 1)).detach().cpu())
                    val_iter.set_postfix(loss=f"{avg_val_loss:.6f}")

        val_epoch_loss = float((val_loss_sum_t / max(val_count, 1)).detach().cpu())
        val_losses.append(val_epoch_loss)
        val_true = torch.cat(val_true_list, dim=0)
        val_pred = torch.cat(val_pred_list, dim=0)
        val_true, val_pred = _slice_supervised_steps(val_true, val_pred, loss_start_idx)
        val_metrics = _compute_metrics_vectorized(val_true, val_pred)
        val_epoch_r2 = float(val_metrics["r2_global"].detach().cpu())

        if test_eval_interval > 0 and ((epoch + 1) % test_eval_interval == 0):
            test_epoch_out = _infer_loader_outputs(model, test_loader, dev, use_amp=amp_enabled, return_full=False)
            y_test_epoch, pred_test_epoch = _slice_supervised_steps(
                test_epoch_out["Y_true"],
                test_epoch_out["SM_pred"],
                loss_start_idx,
            )
            test_epoch_metrics = _compute_metrics_vectorized(y_test_epoch, pred_test_epoch)
            last_test_r2 = float(test_epoch_metrics["r2_global"].cpu())

        scheduler.step(val_epoch_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])

        torch.save(_checkpoint_payload(epoch + 1, train_epoch_loss, val_epoch_loss), last_model_path)

        if val_epoch_loss < best_val:
            best_val = val_epoch_loss
            wait = 0
            torch.save(_checkpoint_payload(epoch + 1, train_epoch_loss, val_epoch_loss), best_model_path)
        else:
            wait += 1

        train_decreasing = (
            prev_train_loss is not None
            and train_epoch_loss < prev_train_loss * (1.0 - max(overfit_train_drop_min_ratio, 0.0))
        )
        val_rising_from_best = (
            np.isfinite(best_val)
            and val_epoch_loss > best_val * (1.0 + max(overfit_val_rise_ratio, 0.0))
        )
        if train_decreasing and val_rising_from_best:
            overfit_wait += 1
        else:
            overfit_wait = 0
        prev_train_loss = train_epoch_loss

        epoch_bar.set_postfix(
            train=f"{train_epoch_loss:.6f}",
            val=f"{val_epoch_loss:.6f}",
            val_r2=f"{val_epoch_r2:.4f}",
            test_r2=f"{last_test_r2:.4f}" if np.isfinite(last_test_r2) else "nan",
            overfit=f"{overfit_wait}/{max(overfit_patience, 1)}",
            best=f"{best_val:.6f}",
            lr=f"{current_lr:.2e}",
        )
        print(
            f"Epoch {epoch + 1}/{max_epochs} | "
            f"train_loss={train_epoch_loss:.6f} | "
            f"val_loss={val_epoch_loss:.6f} | "
            f"val_r2={val_epoch_r2:.4f} | "
            f"test_r2={(f'{last_test_r2:.4f}' if np.isfinite(last_test_r2) else 'nan')} | "
            f"lr={current_lr:.2e}"
        )

        if wait >= early_stop_patience:
            print(f"Early stopping triggered: validation did not improve for {early_stop_patience} epochs.")
            break

        if overfit_patience > 0 and overfit_wait >= overfit_patience:
            print(
                "Overfit stop triggered: training loss keeps decreasing while validation loss rises vs best.",
                {
                    "overfit_patience": overfit_patience,
                    "overfit_val_rise_ratio": overfit_val_rise_ratio,
                    "overfit_train_drop_min_ratio": overfit_train_drop_min_ratio,
                },
            )
            break

    if not best_model_path.exists():
        raise RuntimeError("Best checkpoint was not created; check data or loss stability.")

    best_ckpt = torch.load(best_model_path, map_location=dev)
    model.load_state_dict(best_ckpt["model_state_dict"])

    _plot_loss_curves(train_losses, val_losses, output_dir)

    test_out = _infer_loader_outputs(model, test_loader, dev, use_amp=amp_enabled, return_full=False)
    y_test_true = test_out["Y_true"]
    y_test_pred = test_out["SM_pred"]
    y_test_true, y_test_pred = _slice_supervised_steps(y_test_true, y_test_pred, loss_start_idx)
    metrics = _compute_metrics_vectorized(y_test_true, y_test_pred)

    test_indices = np.asarray(test_loader.dataset.indices, dtype=np.int64)
    cluster_test = dataset.cluster_raw[test_indices].cpu()

    rows = [
        {
            "group": "global_test",
            "R2": float(metrics["r2_global"].cpu()),
            "KGE": float(metrics["kge_global"].cpu()),
            "RMSE": float(metrics["rmse_global"].cpu()),
            "ubRMSE": float(metrics["ubrmse_global"].cpu()),
            "n_grids": int(cluster_test.numel()),
        }
    ]
    for c in sorted(torch.unique(cluster_test).tolist()):
        c_mask = cluster_test == c
        rows.append(
            {
                "group": f"cluster_{int(c)}",
                "R2": float(metrics["r2_grid"][c_mask].mean().cpu()),
                "KGE": float(metrics["kge_grid"][c_mask].mean().cpu()),
                "RMSE": float(metrics["rmse_grid"][c_mask].mean().cpu()),
                "ubRMSE": float(metrics["ubrmse_grid"][c_mask].mean().cpu()),
                "n_grids": int(c_mask.sum().item()),
            }
        )

    pd.DataFrame(rows).to_csv(output_dir / "test_metrics.csv", index=False, encoding="utf-8-sig")

    if export_daily_results:
        full_kwargs = {
            "batch_size": effective_eval_batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "collate_fn": getattr(train_loader, "collate_fn", None),
        }
        if num_workers > 0:
            full_kwargs["persistent_workers"] = True
        full_loader = DataLoader(dataset, **full_kwargs)
        n_grid = len(dataset)
        _export_daily_outputs_streaming(
            model=model,
            loader=full_loader,
            device=dev,
            use_amp=amp_enabled,
            parquet_path=output_dir / "ode_results.parquet",
            mask_dict=mask_dict,
            n_grid=n_grid,
        )
    else:
        print("Skipped full daily ode_results.parquet export; set export_daily_results=True to generate parameter fields.")

    return {
        "best_val_loss": float(best_val),
        "test_r2": float(metrics["r2_global"].cpu()),
        "test_kge": float(metrics["kge_global"].cpu()),
        "test_rmse": float(metrics["rmse_global"].cpu()),
        "test_ubrmse": float(metrics["ubrmse_global"].cpu()),
    }
