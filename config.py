# -*- coding: utf-8 -*-
'''
@File    :   config.py
@Time    :   2026-04-25
@Desc    :   This file centralizes key parameters for the "Causal-Gated Differentiable Mamba-ODE" threshold modeling experiment. It covers data dimension definitions, network hyperparameters, training strategy, VRAM control, path configuration, and data-source preferences. Using a dataclass to distribute configuration avoids hard-coded constants across modules, reduces coupling, and prevents dimension or path mismatches. The config includes a "prefer dynamic parquet" switch with optional paths, supporting direct use of prepared parquet files or automatic project scanning with column alias matching. Training settings add mini-batch training, AMP mixed precision, DataLoader workers, and pin_memory options to run stably on 4GB VRAM while continuously saving checkpoints. Output settings are unified under threshold modeling/outputs for easier management of figures, metrics, and parameter-field results.
@Notice  :   Recommended to run in the "Tiandi Kuiao" environment; if VRAM is tight, reduce batch_size and d_hidden first.
'''

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class ExperimentConfig:
    """Experiment-wide configuration."""

    # ----------------------------
    # Data dimensions
    # ----------------------------
    n_static: int = 8
    n_loc_enc: int = 16
    n_cluster: int = 4
    n_dyn: int = 4  # Pre, LST, PET, LAI
    n_time_enc: int = 9
    n_hum: int = 4  # irr, manu, dom, ele
    n_hmm_state: int = 5

    # ----------------------------
    # Model architecture
    # ----------------------------
    d_hidden: int = 64
    kan_num_basis: int = 8
    dropout: float = 0.1
    use_kan: bool = False

    # ----------------------------
    # Spatiotemporal coupling and gating
    # ----------------------------
    mamba_static_context_dim: int = 16
    use_soft_causal_gating: bool = True
    causal_gate_floor: float = 0.2
    inject_sm0_to_mamba: bool = True
    film_gamma_mode: str = "exp_clamped"  # {"legacy_tanh", "exp_clamped"}

    # ----------------------------
    # ODE constraints and gray-box residuals
    # ----------------------------
    ode_param_scale_alpha: float = 0.4
    ode_param_scale_theta: float = 0.4
    ode_param_scale_eta_pet: float = 0.4
    ode_param_scale_phi_lai: float = 0.5
    ode_use_dsm_clip: bool = True
    ode_dsm_clip_value: float = 0.08
    ode_enable_delta_residual: bool = True
    ode_delta_residual_scale: float = 0.05
    ode_enable_sm_residual: bool = False
    ode_sm_residual_scale: float = 0.5

    # ----------------------------
    # Data processing
    # ----------------------------
    human_interp_method: str = "step"  # {"step", "pchip", "cubic"}
    normalize_dynamic_for_encoder: bool = True

    # ----------------------------
    # Training and VRAM control
    # ----------------------------
    batch_size: int = 96
    eval_batch_size: Optional[int] = 192
    num_workers: int = 0
    pin_memory: bool = True
    use_amp: bool = True
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 40
    early_stop_patience: int = 10
    resume_from_best: bool = False
    resume_optimizer_state: bool = False
    overfit_patience: int = 4
    overfit_val_rise_ratio: float = 0.12
    overfit_train_drop_min_ratio: float = 5e-4
    gradient_clip_norm: float = 1.0
    lr_scheduler_patience: int = 2
    lr_scheduler_factor: float = 0.5
    min_lr: float = 1e-5
    test_eval_interval: int = 0
    export_daily_results: bool = False
    loss_start_idx: int = 1
    derivative_loss_weight: float = 0.0
    derivative_smooth_window: int = 0
    checkpoint_version: int = 8
    random_seed: int = 42
    deterministic: bool = False
    cudnn_benchmark: bool = True

    # ----------------------------
    # Data scale
    # ----------------------------
    # None means full grid; for debugging set to 128 / 512 / 1000.
    max_grids: Optional[int] = None

    # ----------------------------
    # Data source preferences
    # ----------------------------
    # If provided, prefer this parquet as the dynamic data source.
    dynamic_parquet_path: Optional[Path] = None
    dynamic_parquet_paths: Optional[List[Path]] = None
    # True: error if no dynamic parquet is found; False: fall back to HMM CSV.
    strict_dynamic_parquet: bool = True
    dynamic_cache_enabled: bool = True
    dynamic_cache_dir_name: str = "cache"

    # ----------------------------
    # Paths
    # ----------------------------
    project_root: Path = Path(__file__).resolve().parents[1]
    output_dir_name: str = "outputs"

    @property
    def model_root(self) -> Path:
        return Path(__file__).resolve().parent

    @property
    def output_dir(self) -> Path:
        return self.model_root / self.output_dir_name

    def __post_init__(self) -> None:
        # Do not bind external absolute paths by default; let main.py discover
        # relative paths automatically. Users can override via dynamic_parquet_path(s).
        if self.dynamic_parquet_paths is None:
            self.dynamic_parquet_paths = []
