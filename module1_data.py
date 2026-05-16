# -*- coding: utf-8 -*-
'''
@File    :   module1_data.py
@Time    :   2026-04-25
@Desc    :   This file implements the full engineering code for "Module 1: data preprocessing, temporal downscaling, and stratified spatial splitting." The core goal is to unify multi-source heterogeneous data into standard tensor structures that the differentiable model can consume directly. It focuses on three problems: (1) consistent dimensions across static, dynamic, and spatiotemporal encodings so that soil moisture targets, static terrain/soil features, spatial location, dynamic natural drivers, human activities, and causal prior masks align strictly on grid and time axes; (2) cubic-spline interpolation for monthly human water-use data, smoothing step-like monthly series into continuous daily sequences to avoid nonphysical jumps at month boundaries; (3) stratified spatial splits by geomorphic clusters so train/val/test all cover every Cluster, enabling more realistic spatial extrapolation evaluation (ungauged basins). The file provides the EcohydroDataset class, spatiotemporal sin/cos encoders, monthly-to-daily spline functions, and stratified DataLoader builders, plus a runnable mock test entry.
@Notice  :   Depends on numpy, scipy, torch, sklearn. For real data, ensure input tensor shapes match the documented contracts.
'''

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from scipy.interpolate import CubicSpline, PchipInterpolator
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset


def _ecohydro_collate(batch):
    """Collate grid samples while keeping grid-invariant time features single-copy."""
    y_sm, x_static, x_dyn, x_dyn_norm, x_time, w_causal, x_hum = zip(*batch)
    return (
        torch.stack(y_sm, dim=0),
        torch.stack(x_static, dim=0),
        torch.stack(x_dyn, dim=0),
        torch.stack(x_dyn_norm, dim=0),
        x_time[0].unsqueeze(0),
        torch.stack(w_causal, dim=0),
        torch.stack(x_hum, dim=0),
    )


def _validate_even_dim(dim: int, name: str) -> None:
    """Check that the encoding dimension is even; Sin/Cos must come in pairs."""
    if dim % 2 != 0:
        raise ValueError(f"{name} must be even, got: {dim}")


def sinusoidal_encode_1d(values: np.ndarray, out_dim: int, period: float) -> np.ndarray:
    """
    Apply sinusoidal positional encoding to a 1D scalar sequence.

    Math intuition:
    - Map a scalar (e.g., DOY, longitude, latitude) into a multi-frequency periodic basis.
    - Multi-frequency sin/cos captures low-frequency seasonal trends and high-frequency local
      variability, and is NN-friendly (continuous, differentiable, no discrete discontinuities).
    """
    _validate_even_dim(out_dim, "out_dim")

    values = np.asarray(values, dtype=np.float32)
    half_dim = out_dim // 2

    # Exponential frequency series: 1, 2, 4, ... for multi-scale periodic bases.
    freqs = np.power(2.0, np.arange(half_dim, dtype=np.float32))
    angles = 2.0 * np.pi * values[..., None] * freqs[None, ...] / period

    sin_part = np.sin(angles)
    cos_part = np.cos(angles)
    return np.concatenate([sin_part, cos_part], axis=-1).astype(np.float32)


def encode_latlon(lat_lon: np.ndarray, out_dim: int = 16) -> np.ndarray:
    """
    Encode lat/lon into a high-dimensional space (default 16 dims).

    Physical intuition:
    - Longitude has a natural 360-degree periodicity; latitude shows banded climate patterns.
    - Multi-frequency Sin/Cos on Lat/Lon reduces the artificial break at 180/-180 degrees.
    """
    lat_lon = np.asarray(lat_lon, dtype=np.float32)
    if lat_lon.ndim != 2 or lat_lon.shape[1] != 2:
        raise ValueError(f"Lat_Lon should have shape (N, 2); got: {lat_lon.shape}")
    if out_dim % 4 != 0:
        raise ValueError(f"Spatial encoding dim out_dim must be a multiple of 4; got: {out_dim}")

    per_coord_dim = out_dim // 2
    lat_enc = sinusoidal_encode_1d(lat_lon[:, 1], per_coord_dim, period=180.0)
    lon_enc = sinusoidal_encode_1d(lat_lon[:, 0], per_coord_dim, period=360.0)
    return np.concatenate([lat_enc, lon_enc], axis=-1).astype(np.float32)


def encode_doy(doy: np.ndarray, out_dim: int = 8) -> np.ndarray:
    """
    DOY time positional encoding.

    Math intuition:
    - DOY is a yearly periodic variable with period 365.
    - Multi-frequency Sin/Cos lets the model learn seasonal cycles and harmonics.
    """
    doy = np.asarray(doy, dtype=np.float32)
    return sinusoidal_encode_1d(doy, out_dim=out_dim, period=365.0)


def encode_time_features(doy: np.ndarray, time_norm: Optional[np.ndarray], out_dim: int = 9) -> np.ndarray:
    """Build temporal features as DOY harmonics plus a normalized long-term trend."""
    if out_dim < 3:
        raise ValueError("out_dim must be at least 3 for temporal encoding.")
    doy_dim = out_dim - 1
    if doy_dim % 2 != 0:
        raise ValueError(f"out_dim - 1 must be even, got out_dim={out_dim}.")
    doy_enc = encode_doy(doy, out_dim=doy_dim)
    if time_norm is None:
        n_time = doy_enc.shape[0]
        time_norm_arr = np.linspace(0.0, 1.0, num=n_time, dtype=np.float32) if n_time > 1 else np.zeros((n_time,), dtype=np.float32)
    else:
        time_norm_arr = np.asarray(time_norm, dtype=np.float32)
    if time_norm_arr.shape[0] != doy_enc.shape[0]:
        raise ValueError("time_norm length must match DOY length.")
    return np.concatenate([doy_enc, time_norm_arr[:, None]], axis=-1).astype(np.float32)


def interpolate_monthly_to_daily(
    monthly_data: np.ndarray,
    target_len: int,
    month_day_index: Optional[np.ndarray] = None,
    clip_min: Optional[float] = None,
    method: str = "step",
) -> np.ndarray:
    """
    Smoothly interpolate monthly data to daily scale.

    Key notes (physical/numerical):
    1. Copying each monthly value to all days creates step jumps at month boundaries,
       which is physically unrealistic for hydrology.
    2. Cubic splines build C2-continuous curves (value/1st/2nd derivatives continuous),
       reducing gradient noise from input discontinuities.
    3. This function processes all grids/variables in a vectorized way along axis=1,
       avoiding loops over N/T.
    """
    monthly_data = np.asarray(monthly_data, dtype=np.float32)
    if monthly_data.ndim != 3:
        raise ValueError(f"monthly_data should be (N, T_month, F); got: {monthly_data.shape}")

    n_grid, n_month, _ = monthly_data.shape
    if n_month < 4:
        raise ValueError("Cubic spline requires at least 4 time points")

    if month_day_index is None:
        month_day_index = np.linspace(0.0, float(target_len - 1), num=n_month, dtype=np.float32)
    else:
        month_day_index = np.asarray(month_day_index, dtype=np.float32)
        if month_day_index.shape[0] != n_month:
            raise ValueError("month_day_index length must equal T_month")

    day_index = np.arange(target_len, dtype=np.float32)

    method = str(method).lower()
    if method == "cubic":
        spline = CubicSpline(
            x=month_day_index,
            y=monthly_data,
            axis=1,
            bc_type="natural",
            extrapolate=True,
        )
        daily = spline(day_index).astype(np.float32)
    elif method == "pchip":
        spline = PchipInterpolator(x=month_day_index, y=monthly_data, axis=1, extrapolate=True)
        daily = spline(day_index).astype(np.float32)
    elif method == "step":
        # Step interpolation: preserve monthly pulses without spline oscillations.
        idx = np.searchsorted(month_day_index, day_index, side="right") - 1
        idx = np.clip(idx, 0, n_month - 1).astype(np.int64)
        daily = monthly_data[:, idx, :].astype(np.float32)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    if clip_min is not None:
        daily = np.clip(daily, clip_min, None)
    return daily


class EcohydroDataset(Dataset):
    """
    Ecohydrological differentiable modeling dataset.

    Input conventions:
    - Y_SM: (N, T, 1)
    - X_sta: (N, 8)
    - Lat_Lon: (N, 2)
    - Cluster: (N,) in {1,2,3,4}
    - X_dyn: (N, T, 4)
    - DOY: (T,)
    - W_causal: (N, T, 4)
    - X_hum_monthly: (N, T_month, 4)
    """

    def __init__(
        self,
        Y_SM: np.ndarray,
        X_sta: np.ndarray,
        Lat_Lon: np.ndarray,
        Cluster: np.ndarray,
        X_dyn: np.ndarray,
        DOY: np.ndarray,
        W_causal: np.ndarray,
        X_hum_monthly: np.ndarray,
        Grid_ID: Optional[np.ndarray] = None,
        Date: Optional[np.ndarray] = None,
        Time_Norm: Optional[np.ndarray] = None,
        loc_enc_dim: int = 16,
        time_enc_dim: int = 8,
        month_day_index: Optional[np.ndarray] = None,
        human_interp_method: str = "step",
    ) -> None:
        super().__init__()

        # ----------------------------
        # 1) Basic dimension checks
        # ----------------------------
        Y_SM = np.asarray(Y_SM, dtype=np.float32)
        X_sta = np.asarray(X_sta, dtype=np.float32)
        Lat_Lon = np.asarray(Lat_Lon, dtype=np.float32)
        Cluster = np.asarray(Cluster, dtype=np.int64)
        X_dyn = np.asarray(X_dyn, dtype=np.float32)
        DOY = np.asarray(DOY, dtype=np.float32)
        W_causal = np.asarray(W_causal, dtype=np.float32)
        X_hum_monthly = np.asarray(X_hum_monthly, dtype=np.float32)

        n_grid, n_time, y_feat = Y_SM.shape
        Grid_ID = np.arange(n_grid, dtype=np.int64) if Grid_ID is None else np.asarray(Grid_ID, dtype=np.int64)
        Date = np.arange(n_time, dtype=np.int64) if Date is None else np.asarray(Date)
        Time_Norm = None if Time_Norm is None else np.asarray(Time_Norm, dtype=np.float32)
        if y_feat != 1:
            raise ValueError("Y_SM last dimension must be 1")
        if X_sta.shape != (n_grid, 8):
            raise ValueError(f"X_sta should have shape ({n_grid}, 8); got: {X_sta.shape}")
        if Lat_Lon.shape != (n_grid, 2):
            raise ValueError(f"Lat_Lon should have shape ({n_grid}, 2); got: {Lat_Lon.shape}")
        if Cluster.shape[0] != n_grid:
            raise ValueError("Cluster length must match N")
        if Grid_ID.shape[0] != n_grid:
            raise ValueError("Grid_ID length must match N")
        if Date.shape[0] != n_time:
            raise ValueError("Date length must match T")
        if X_dyn.shape[:2] != (n_grid, n_time):
            raise ValueError("X_dyn first two dims must be (N, T)")
        if X_dyn.shape[-1] != 4:
            raise ValueError(f"X_dyn last dimension should be 4 (Pre, LST, PET, LAI); got: {X_dyn.shape[-1]}")
        if DOY.shape[0] != n_time:
            raise ValueError("DOY length must match T")
        if Time_Norm is not None and Time_Norm.shape[0] != n_time:
            raise ValueError("Time_Norm length must match T")
        if W_causal.shape != X_dyn.shape:
            raise ValueError(f"W_causal must match X_dyn shape; got: {W_causal.shape} vs {X_dyn.shape}")
        if X_hum_monthly.shape[0] != n_grid or X_hum_monthly.shape[-1] != 4:
            raise ValueError("X_hum_monthly must be (N, T_month, 4)")

        # ----------------------------
        # 2) Cluster one-hot encoding
        # ----------------------------
        if np.any((Cluster < 1) | (Cluster > 4)):
            raise ValueError("Cluster must be in {1,2,3,4}")
        cluster_idx = Cluster - 1
        cluster_onehot = np.eye(4, dtype=np.float32)[cluster_idx]  # (N,4)

        # ----------------------------
        # 3) Spatiotemporal Sin/Cos lifting
        # ----------------------------
        x_loc_enc = encode_latlon(Lat_Lon, out_dim=loc_enc_dim)  # (N,16)
        x_time_base = encode_time_features(DOY, Time_Norm, out_dim=time_enc_dim)  # (T,p)

        # ----------------------------
        # 4) Downscale monthly human activity to daily
        # ----------------------------
        x_hum_daily = interpolate_monthly_to_daily(
            monthly_data=X_hum_monthly,
            target_len=n_time,
            month_day_index=month_day_index,
            clip_min=0.0,  # Human water use should not be negative; clip for physical constraint.
            method=human_interp_method,
        )
        # ----------------------------
        # 5) Store as torch tensors
        # ----------------------------
        self.Y_SM = torch.from_numpy(Y_SM)
        self.X_sta = torch.from_numpy(X_sta)
        self.Lat_Lon = torch.from_numpy(Lat_Lon)
        self.cluster_raw = torch.from_numpy(Cluster)
        self.cluster_onehot = torch.from_numpy(cluster_onehot)
        self.X_dyn = torch.from_numpy(X_dyn)
        self.X_dyn_norm = self.X_dyn.clone()
        self.DOY = torch.from_numpy(DOY)
        self.Grid_ID = torch.from_numpy(Grid_ID.astype(np.int64, copy=False))
        self.Date = np.asarray(Date)
        self.Time_Norm = torch.from_numpy(x_time_base[:, -1].copy())
        self.W_causal = torch.from_numpy(W_causal)
        self.X_hum_monthly = torch.from_numpy(X_hum_monthly)
        self.X_hum_daily = torch.from_numpy(x_hum_daily)
        self.X_loc_enc = torch.from_numpy(x_loc_enc)
        self.X_time_enc = torch.from_numpy(x_time_base)
        self._normalization_applied = False
        self.normalization_stats: Dict[str, torch.Tensor] = {}
        self._refresh_static_all()

    def _refresh_static_all(self) -> None:
        self.X_static_all = torch.cat([self.X_sta, self.X_loc_enc, self.cluster_onehot], dim=-1)

    def apply_train_normalization(self, train_idx: np.ndarray) -> None:
        """
        Normalize static continuous features using training-set stats only, preventing
        scale imbalance from suppressing Sin/Cos encodings and cluster one-hot features,
        and avoiding validation/test leakage.
        """
        if self._normalization_applied:
            return
        train_idx = np.asarray(train_idx, dtype=np.int64)
        if train_idx.size == 0:
            return

        train_idx_t = torch.as_tensor(train_idx, dtype=torch.long)
        train_sta = self.X_sta.index_select(0, train_idx_t)
        sta_mean = train_sta.mean(dim=0)
        sta_std = train_sta.std(dim=0, unbiased=False).clamp_min(1e-6)

        train_dyn = self.X_dyn.index_select(0, train_idx_t)
        dyn_mean = train_dyn.mean(dim=(0, 1), keepdim=True)
        dyn_std = train_dyn.std(dim=(0, 1), unbiased=False, keepdim=True).clamp_min(1e-6)

        self.X_sta = (self.X_sta - sta_mean) / sta_std
        self.X_dyn_norm = (self.X_dyn - dyn_mean) / dyn_std
        self.normalization_stats = {
            "X_sta_mean": sta_mean,
            "X_sta_std": sta_std,
            "X_dyn_mean": dyn_mean.squeeze(0).squeeze(0),
            "X_dyn_std": dyn_std.squeeze(0).squeeze(0),
        }
        self._normalization_applied = True
        self._refresh_static_all()

    def __len__(self) -> int:
        return self.Y_SM.shape[0]

    def __getitem__(self, idx: int):
        """
        Return a single-grid sample:
        - Y_SM[idx]: (T,1)
        - X_static_all[idx]: (28,)
        - X_dyn[idx]: (T,4)
        - X_time_enc: (T,8), grid-invariant and shared by the custom collate_fn
        - W_causal[idx]: (T,4)
        - X_hum_daily[idx]: (T,4)
        """
        return (
            self.Y_SM[idx],
            self.X_static_all[idx],
            self.X_dyn[idx],
            self.X_dyn_norm[idx],
            self.X_time_enc,
            self.W_causal[idx],
            self.X_hum_daily[idx],
        )


def _split_single_cluster_indices(
    cluster_indices: np.ndarray,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split one Cluster into 7:2:1.
    For small samples, use a minimal viable allocation.
    """
    n = cluster_indices.size
    if n == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    # First split train / temp (70 / 30)
    if n >= 10:
        train_idx, temp_idx = train_test_split(
            cluster_indices,
            test_size=0.3,
            random_state=random_state,
            shuffle=True,
        )
        # Then split temp into val/test = 2:1
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=1.0 / 3.0,
            random_state=random_state,
            shuffle=True,
        )
        return (
            np.asarray(train_idx, dtype=np.int64),
            np.asarray(val_idx, dtype=np.int64),
            np.asarray(test_idx, dtype=np.int64),
        )

    # Custom split for small samples to stay robust
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(cluster_indices)

    if n == 1:
        return perm[:1], np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    if n == 2:
        return perm[:1], perm[1:2], np.array([], dtype=np.int64)

    n_train = max(1, int(round(0.7 * n)))
    n_val = max(1, int(round(0.2 * n)))
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val : n_train + n_val + n_test]
    return train_idx, val_idx, test_idx


def get_stratified_dataloaders(
    dataset: EcohydroDataset,
    batch_size: int = 96,
    num_workers: int = 0,
    pin_memory: bool = False,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, torch.Tensor], Dict[str, np.ndarray]]:
    """
    Stratify spatial splits by Cluster and return DataLoaders + masks + indices.
    """
    cluster_np = dataset.cluster_raw.cpu().numpy()
    unique_clusters = np.unique(cluster_np)

    train_parts = []
    val_parts = []
    test_parts = []

    for c in unique_clusters:
        idx_c = np.where(cluster_np == c)[0].astype(np.int64)
        tr, va, te = _split_single_cluster_indices(idx_c, random_state=random_state + int(c))
        train_parts.append(tr)
        val_parts.append(va)
        test_parts.append(te)

    train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64)
    val_idx = np.concatenate(val_parts) if val_parts else np.array([], dtype=np.int64)
    test_idx = np.concatenate(test_parts) if test_parts else np.array([], dtype=np.int64)

    # Shuffle to avoid order bias from cluster concatenation.
    rng = np.random.default_rng(random_state)
    train_idx = rng.permutation(train_idx)
    val_idx = rng.permutation(val_idx)
    test_idx = rng.permutation(test_idx)

    dataset.apply_train_normalization(train_idx)
    if getattr(dataset, "_normalization_applied", False):
        print("Normalized static features using training stats to avoid scale imbalance in KAN/FiLM learning.")

    torch_gen = torch.Generator()
    torch_gen.manual_seed(int(random_state))

    def _worker_init_fn(worker_id: int) -> None:
        # Give each worker a reproducible but distinct random seed.
        np.random.seed(int(random_state) + int(worker_id))

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": _worker_init_fn,
        "generator": torch_gen,
        "collate_fn": _ecohydro_collate,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx.tolist()),
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx.tolist()),
        shuffle=False,
        **loader_kwargs,
    )

    n_grid = len(dataset)
    train_mask = torch.zeros(n_grid, dtype=torch.bool)
    val_mask = torch.zeros(n_grid, dtype=torch.bool)
    test_mask = torch.zeros(n_grid, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    mask_dict = {"train": train_mask, "val": val_mask, "test": test_mask}
    index_dict = {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}
    return train_loader, val_loader, test_loader, mask_dict, index_dict


def build_mock_arrays(
    n_grid: int = 1000,
    n_time: int = 1826,
    t_month: int = 60,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate mock data matching the shape contract for testing and quick checks."""
    rng = np.random.default_rng(seed)

    # Static and spatial info
    x_sta = rng.normal(size=(n_grid, 8)).astype(np.float32)
    lat = rng.uniform(35.0, 43.0, size=n_grid).astype(np.float32)
    lon = rng.uniform(109.0, 113.5, size=n_grid).astype(np.float32)
    lat_lon = np.stack([lon, lat], axis=-1).astype(np.float32)
    cluster = rng.integers(1, 5, size=n_grid, endpoint=False).astype(np.int64)

    # Time axis
    doy = ((np.arange(n_time) % 365) + 1).astype(np.float32)

    # Dynamic variables and causal mask
    x_dyn = rng.normal(size=(n_grid, n_time, 4)).astype(np.float32)
    w_causal = rng.uniform(0.2, 1.0, size=(n_grid, n_time, 4)).astype(np.float32)
    x_hum_monthly = rng.uniform(0.0, 1.0, size=(n_grid, t_month, 4)).astype(np.float32)

    # Target variable with a learnable synthetic relationship
    x_hum_daily = interpolate_monthly_to_daily(x_hum_monthly, target_len=n_time, clip_min=0.0)
    pre = x_dyn[..., 0]
    lst = x_dyn[..., 1]
    hum_sum = x_hum_daily.sum(axis=-1)
    dsm = 0.05 * pre - 0.02 * lst - 0.01 * hum_sum
    sm0 = rng.uniform(0.15, 0.35, size=(n_grid, 1)).astype(np.float32)
    y_sm = sm0[:, None, :] + np.cumsum(dsm, axis=1)[..., None]
    y_sm = y_sm.astype(np.float32)

    return {
        "Y_SM": y_sm,
        "X_sta": x_sta,
        "Lat_Lon": lat_lon,
        "Cluster": cluster,
        "X_dyn": x_dyn,
        "DOY": doy,
        "Grid_ID": np.arange(n_grid, dtype=np.int64),
        "Date": np.arange(n_time, dtype=np.int64),
        "Time_Norm": np.linspace(0.0, 1.0, num=n_time, dtype=np.float32),
        "W_causal": w_causal,
        "X_hum_monthly": x_hum_monthly,
    }


if __name__ == "__main__":
    arrays = build_mock_arrays(n_grid=1000, n_time=1826, t_month=60, seed=42)
    dataset = EcohydroDataset(**arrays, loc_enc_dim=16, time_enc_dim=8)
    train_loader, val_loader, test_loader, mask_dict, index_dict = get_stratified_dataloaders(
        dataset,
        batch_size=32,
        random_state=42,
    )

    print("Dataset length:", len(dataset))
    print("X_static_all shape:", tuple(dataset.X_static_all.shape))
    print("X_time_enc shape:", tuple(dataset.X_time_enc.shape))
    print("X_hum_daily shape:", tuple(dataset.X_hum_daily.shape))
    print("train/val/test sample counts:", int(mask_dict["train"].sum()), int(mask_dict["val"].sum()), int(mask_dict["test"].sum()))
    print("train idx first 5:", index_dict["train_idx"][:5].tolist())
