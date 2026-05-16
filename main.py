# -*- coding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2026-04-25
@Desc    :   This file is the main entry point under the "threshold modeling" directory, chaining modules 1-5 into a runnable pipeline. The program prioritizes yearly dynamic parquet files (2015-2019) and extracts Grid_ID, Date, Pre, LST, PET, LAI, SM to build time-series training samples with SM as the supervision target. Dynamic inputs keep only Pre, LST, PET, LAI; SM is no longer fed as a lagged input to avoid extra memory terms interfering with the target. It then reads KMeans clustering results, HMM state sequences, JPCMCI causal edge statistics, and monthly human water-use data, performs spatiotemporal encoding, spline downscaling, and causal mask construction, and produces data structures for Causal-Gated Differentiable Mamba-ODE training. Training uses mini-batch vectorized forward passes to avoid GPU OOM from full-grid batches. After training, it outputs loss curves, metrics, weights, and physical parameter fields. If test_metrics.csv is large, it also exports a parquet version next to this code to satisfy large-file output needs.
@Notice  :   If MCP resources are empty in the current session, the program reads data directly from the local filesystem.
'''

from __future__ import annotations

import hashlib
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from config import ExperimentConfig
from module1_data import EcohydroDataset, build_mock_arrays, get_stratified_dataloaders, interpolate_monthly_to_daily
from module45_train import CausalMambaODE, _export_daily_outputs_streaming, train_and_evaluate


class DataSourceFallbackError(Exception):
    """Fallback-friendly data source error (missing data/field mismatch)."""


def _pick_column_by_alias(columns: List[str], aliases: List[str]) -> Optional[str]:
    """Match a column name by aliases."""
    lower_map = {c.lower(): c for c in columns}
    for alias in aliases:
        hit = lower_map.get(alias.lower())
        if hit is not None:
            return hit
    return None


def _candidate_dynamic_paths(cfg: ExperimentConfig, project_root: Path) -> List[Path]:
    """Collect candidate dynamic parquet paths; prefer explicit paths, then relative rules."""
    candidates: List[Path] = []
    seen = set()

    def _append_if_exists(p: Path) -> None:
        if p.exists():
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                candidates.append(p)

    # 1) User explicit config
    if cfg.dynamic_parquet_paths:
        for p in cfg.dynamic_parquet_paths:
            _append_if_exists(Path(p))
    if cfg.dynamic_parquet_path is not None:
        _append_if_exists(Path(cfg.dynamic_parquet_path))

    # 2) Relative paths inside this project
    local_dir = project_root / "数据处理" / "探索性数据分析"
    if local_dir.exists():
        for p in sorted(local_dir.glob("Soil_Moisture_Data_*.parquet")):
            _append_if_exists(p)

    # 3) Sibling project relative paths (avoid hard-coded absolute paths)
    sibling_dir = project_root.parent / "土壤水分反演NEW" / "数据处理" / "探索性数据分析"
    if sibling_dir.exists():
        for p in sorted(sibling_dir.glob("Soil_Moisture_Data_*.parquet")):
            _append_if_exists(p)

    return candidates


def _infer_dynamic_col_map(parquet_path: Path) -> Dict[str, str]:
    """
    Infer dynamic field mapping from the parquet schema.
    Must include grid/date/pre/lst/pet/lai/sm fields.
    """
    alias = {
        "grid": ["Grid_ID", "grid_id", "gridid", "id"],
        "date": ["Date", "date", "datetime", "time", "Time"],
        "pre": ["Pre", "pre", "Pre_Raw", "pre_raw", "precip", "precipitation"],
        "lst": ["LST", "lst", "LST_Raw", "lst_raw", "temp", "temperature"],
        "pet": ["PET", "pet", "PET_Raw", "pet_raw"],
        "lai": ["LAI", "lai", "LAI_Raw", "lai_raw"],
        "sm": ["SM", "sm", "SM_Raw", "sm_raw", "soil_moisture"],
    }

    schema_cols = pq.ParquetFile(parquet_path).schema.names
    col_map = {
        "grid": _pick_column_by_alias(schema_cols, alias["grid"]),
        "date": _pick_column_by_alias(schema_cols, alias["date"]),
        "pre": _pick_column_by_alias(schema_cols, alias["pre"]),
        "lst": _pick_column_by_alias(schema_cols, alias["lst"]),
        "pet": _pick_column_by_alias(schema_cols, alias["pet"]),
        "lai": _pick_column_by_alias(schema_cols, alias["lai"]),
        "sm": _pick_column_by_alias(schema_cols, alias["sm"]),
    }
    if not all(v is not None for v in col_map.values()):
        raise DataSourceFallbackError(f"Dynamic parquet fields are incomplete: {parquet_path}")
    return {k: v for k, v in col_map.items() if v is not None}


def _load_dynamic_arrays_from_parquets(
    sources: List[Tuple[Path, Dict[str, str]]],
    grid_ids: np.ndarray,
    date_index: pd.DatetimeIndex,
) -> Dict[str, np.ndarray]:
    """
    Build dynamic matrices from multi-year parquet files:
    - Pre/LST/PET/LAI/SM -> (N, T)
    """
    n_grid = grid_ids.shape[0]
    T = len(date_index)
    mats = {
        "Pre": np.full((n_grid, T), np.nan, dtype=np.float32),
        "LST": np.full((n_grid, T), np.nan, dtype=np.float32),
        "PET": np.full((n_grid, T), np.nan, dtype=np.float32),
        "LAI": np.full((n_grid, T), np.nan, dtype=np.float32),
        "SM": np.full((n_grid, T), np.nan, dtype=np.float32),
    }

    date_min = date_index.min()
    date_max = date_index.max()
    date_pos = {d: i for i, d in enumerate(date_index)}
    grid_pos = {int(g): i for i, g in enumerate(grid_ids.tolist())}

    def _fill_by_time(var_mat: np.ndarray) -> np.ndarray:
        # Forward/backward fill along time per grid to avoid expensive pivoting.
        out = var_mat.copy()
        for i in range(out.shape[0]):
            row = out[i]
            valid = ~np.isnan(row)
            if not np.any(valid):
                continue
            idx = np.where(valid, np.arange(row.shape[0], dtype=np.int64), -1)
            np.maximum.accumulate(idx, out=idx)
            first_valid = int(np.argmax(valid))
            row[:first_valid] = row[first_valid]
            right_mask = idx >= 0
            row[right_mask] = row[idx[right_mask]]

            valid2 = ~np.isnan(row)
            idx2 = np.where(valid2, np.arange(row.shape[0], dtype=np.int64), row.shape[0])
            np.minimum.accumulate(idx2[::-1], out=idx2[::-1])
            row[idx2 < row.shape[0]] = row[idx2[idx2 < row.shape[0]]]
            out[i] = row
        return out

    for p, col_map in sources:
        print(f"Reading dynamic parquet: {p}")

        # Enable filters when grid count is small to reduce IO and memory.
        filters = None
        if n_grid <= 5000:
            filters = [(col_map["grid"], "in", grid_ids.tolist())]

        read_cols = [
            col_map["grid"],
            col_map["date"],
            col_map["pre"],
            col_map["lst"],
            col_map["pet"],
            col_map["lai"],
            col_map["sm"],
        ]
        df = pd.read_parquet(p, columns=read_cols, filters=filters)
        df = df.rename(
            columns={
                col_map["grid"]: "Grid_ID",
                col_map["date"]: "Date",
                col_map["pre"]: "Pre",
                col_map["lst"]: "LST",
                col_map["pet"]: "PET",
                col_map["lai"]: "LAI",
                col_map["sm"]: "SM",
            }
        )

        df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
        df = df[(df["Date"] >= date_min) & (df["Date"] <= date_max)]
        df = df[df["Grid_ID"].isin(grid_ids)]
        if df.empty:
            continue

        # If duplicate Grid_ID x Date records exist, aggregate by mean.
        df = df.groupby(["Grid_ID", "Date"], as_index=False)[["Pre", "LST", "PET", "LAI", "SM"]].mean()

        g_idx = df["Grid_ID"].map(grid_pos).to_numpy(dtype=np.int64)
        t_idx = df["Date"].map(date_pos).to_numpy(dtype=np.int64)
        valid_idx = (g_idx >= 0) & (t_idx >= 0)
        if not np.any(valid_idx):
            continue

        g_idx = g_idx[valid_idx]
        t_idx = t_idx[valid_idx]
        for var in ["Pre", "LST", "PET", "LAI", "SM"]:
            vals = df[var].to_numpy(dtype=np.float32)[valid_idx]
            vmask = ~np.isnan(vals)
            mats[var][g_idx[vmask], t_idx[vmask]] = vals[vmask]

    # Forward/backward fill along time to ensure complete series.
    for var in ["Pre", "LST", "PET", "LAI", "SM"]:
        filled = _fill_by_time(mats[var])
        fill_v = float(np.nanmean(filled.astype(np.float32)))
        if np.isnan(fill_v):
            fill_v = 0.0
        mats[var] = np.where(np.isnan(filled), fill_v, filled).astype(np.float32)

    return mats


def _dynamic_cache_dir(cfg: ExperimentConfig) -> Path:
    p = cfg.model_root / cfg.dynamic_cache_dir_name
    p.mkdir(parents=True, exist_ok=True)
    return p


def _build_dynamic_cache_key(paths: List[Path], grid_ids: np.ndarray, date_index: pd.DatetimeIndex) -> str:
    payload = {
        "paths": [f"{str(p.resolve())}:{p.stat().st_mtime_ns}" for p in paths],
        "n_grid": int(grid_ids.shape[0]),
        "grid_min": int(np.min(grid_ids)) if grid_ids.size > 0 else -1,
        "grid_max": int(np.max(grid_ids)) if grid_ids.size > 0 else -1,
        "date_start": str(date_index.min().date()),
        "date_end": str(date_index.max().date()),
        "n_time": int(len(date_index)),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _load_or_build_dynamic_mats(
    cfg: ExperimentConfig,
    valid_sources: List[Tuple[Path, Dict[str, str]]],
    grid_ids: np.ndarray,
    date_index: pd.DatetimeIndex,
) -> Dict[str, np.ndarray]:
    source_paths = [p for p, _ in valid_sources]
    if not cfg.dynamic_cache_enabled:
        return _load_dynamic_arrays_from_parquets(valid_sources, grid_ids, date_index)

    cache_key = _build_dynamic_cache_key(source_paths, grid_ids, date_index)
    cache_path = _dynamic_cache_dir(cfg) / f"dynamic_{cache_key}.npz"
    if cache_path.exists():
        print(f"Dynamic data cache hit: {cache_path}")
        cache = np.load(cache_path)
        return {
            "Pre": cache["Pre"].astype(np.float32),
            "LST": cache["LST"].astype(np.float32),
            "PET": cache["PET"].astype(np.float32),
            "LAI": cache["LAI"].astype(np.float32),
            "SM": cache["SM"].astype(np.float32),
        }

    mats = _load_dynamic_arrays_from_parquets(valid_sources, grid_ids, date_index)
    np.savez_compressed(cache_path, **mats)
    print(f"Dynamic data cache written: {cache_path}")
    return mats


def _set_global_seed(seed: int, deterministic: bool, cudnn_benchmark: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark


def _build_causal_mask_from_edges(
    causal_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    hmm_states: np.ndarray,
) -> np.ndarray:
    """Build W_causal: (N, T, 4), in [Pre, LST, PET, LAI] order."""
    table = np.zeros((4, 5, 4), dtype=np.float32)

    direct = causal_df[causal_df["Target"] == "SM"].copy()
    for src, var_id in {"Pre": 0, "LST": 1, "PET": 2}.items():
        sub = direct[direct["Source"] == src]
        if not sub.empty:
            c_idx = sub["Cluster"].to_numpy(dtype=np.int64) - 1
            s_idx = sub["State"].to_numpy(dtype=np.int64) - 1
            table[c_idx, s_idx, var_id] = sub["Abs_Corr"].to_numpy(dtype=np.float32)

    lai_pet = causal_df[(causal_df["Source"] == "LAI") & (causal_df["Target"] == "PET")][["Cluster", "State", "Abs_Corr"]]
    pet_sm = causal_df[(causal_df["Source"] == "PET") & (causal_df["Target"] == "SM")][["Cluster", "State", "Abs_Corr"]]
    lai_pet = lai_pet.rename(columns={"Abs_Corr": "lai_pet"})
    pet_sm = pet_sm.rename(columns={"Abs_Corr": "pet_sm"})
    lai_merge = pd.merge(lai_pet, pet_sm, on=["Cluster", "State"], how="outer").fillna(0.0)
    if not lai_merge.empty:
        c_idx = lai_merge["Cluster"].to_numpy(dtype=np.int64) - 1
        s_idx = lai_merge["State"].to_numpy(dtype=np.int64) - 1
        lai_score = 0.5 * lai_merge["lai_pet"].to_numpy(dtype=np.float32) * lai_merge["pet_sm"].to_numpy(dtype=np.float32)
        table[c_idx, s_idx, 3] = lai_score

    var_max = np.maximum(table.max(axis=(0, 1), keepdims=True), 1e-6)
    table = table / var_max
    table = np.where(table > 0.0, table, 0.05)
    table = np.clip(table, 0.0, 1.0).astype(np.float32)

    c_idx = np.clip(cluster_labels.astype(np.int64) - 1, 0, 3)
    s_idx = np.clip(hmm_states.astype(np.int64) - 1, 0, 4)
    return table[c_idx[:, None], s_idx[None, :], :].astype(np.float32)


def build_real_arrays(cfg: ExperimentConfig) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray], str]:
    """Read real data and build model inputs."""
    root = cfg.project_root
    path_cluster = root / "Kmeans空间聚类" / "Clustering_Results.csv"
    path_hmm = root / "HMM时间聚类" / "HMM_Daily_Decoded_States_2015_2019.csv"
    path_causal = root / "因果推断" / "2015-2019_JPCMCI_ClusterState_4x5_EdgeStats.csv"
    path_human = root / "数据处理" / "Human_Activity_Water_Use_Monthly_2015_2019.parquet"

    for p in [path_cluster, path_hmm, path_causal, path_human]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    cluster_df = pd.read_csv(path_cluster).sort_values("Grid_ID").reset_index(drop=True)
    if cfg.max_grids is not None:
        cluster_df = cluster_df.iloc[: cfg.max_grids].copy()

    grid_ids = cluster_df["Grid_ID"].to_numpy(dtype=np.int64)
    X_sta = cluster_df[["Dem", "Slope", "Clay", "Sand", "BD", "OC", "porosity", "SAGATWI"]].to_numpy(dtype=np.float32)
    Lat_Lon = cluster_df[["Lon", "Lat"]].to_numpy(dtype=np.float32)
    Cluster = cluster_df["Cluster"].to_numpy(dtype=np.int64)

    hmm_df = pd.read_csv(path_hmm)
    hmm_df["Date"] = pd.to_datetime(hmm_df["Date"]).dt.normalize()
    hmm_df = hmm_df.sort_values("Date").reset_index(drop=True)
    date_index = pd.DatetimeIndex(hmm_df["Date"])
    T = len(date_index)
    DOY = np.minimum(date_index.dayofyear.to_numpy(dtype=np.int64), 365).astype(np.float32)
    if T <= 1:
        Time_Norm = np.zeros(T, dtype=np.float32)
    else:
        year_min, year_max = int(date_index.year.min()), int(date_index.year.max())
        if year_max > year_min:
            Time_Norm = ((date_index.year.to_numpy(dtype=np.float32) - year_min) / (year_max - year_min)).astype(np.float32)
        else:
            Time_Norm = np.linspace(0.0, 1.0, num=T, dtype=np.float32)

    # ----------------------------
    # Dynamic data: prefer parquet
    # ----------------------------
    dyn_paths = _candidate_dynamic_paths(cfg, root)
    use_dynamic_parquet = False
    dynamic_source_desc = ""

    if dyn_paths:
        valid_sources: List[Tuple[Path, Dict[str, str]]] = []
        invalid_logs: List[str] = []
        for p in dyn_paths:
            try:
                cm = _infer_dynamic_col_map(p)
                valid_sources.append((p, cm))
            except DataSourceFallbackError as e:
                invalid_logs.append(str(e))

        if valid_sources:
            print("Final selected dynamic parquet files:")
            for p, _ in valid_sources:
                print(f"  - {p}")

            mats = _load_or_build_dynamic_mats(cfg, valid_sources, grid_ids, date_index)
            pre_mat = mats["Pre"]
            lst_mat = mats["LST"]
            pet_mat = mats["PET"]
            lai_mat = mats["LAI"]
            sm_mat = mats["SM"]
            use_dynamic_parquet = True
            dynamic_source_desc = f"dynamic_parquet_files={len(valid_sources)}"
        else:
            print("No dynamic parquet meets required fields.")
            for msg in invalid_logs:
                print(f"  - {msg}")

    if not use_dynamic_parquet:
        if cfg.strict_dynamic_parquet:
            raise DataSourceFallbackError("Strict mode failed to load dynamic parquet.")

        pre_col = "Pre_Raw" if "Pre_Raw" in hmm_df.columns else "Pre"
        lst_col = "LST_Raw" if "LST_Raw" in hmm_df.columns else "LST"
        pet_col = "PET_Raw" if "PET_Raw" in hmm_df.columns else "PET"
        lai_col = "LAI_Raw" if "LAI_Raw" in hmm_df.columns else "LAI"
        sm_col = "SM_Raw" if "SM_Raw" in hmm_df.columns else "SM"

        daily_4 = hmm_df[[pre_col, lst_col, pet_col, lai_col]].to_numpy(dtype=np.float32)
        sm_daily = hmm_df[sm_col].to_numpy(dtype=np.float32)

        sta_norm = (X_sta - X_sta.mean(axis=0, keepdims=True)) / (X_sta.std(axis=0, keepdims=True) + 1e-6)
        w_mod = np.array(
            [
                [0.12, -0.06, 0.08, 0.02],
                [0.09, 0.04, 0.11, -0.03],
                [-0.03, 0.05, 0.07, 0.08],
                [0.02, -0.07, -0.04, 0.12],
                [0.06, 0.03, -0.02, 0.04],
                [0.04, 0.08, 0.02, -0.05],
                [-0.05, 0.02, 0.03, 0.01],
                [0.07, -0.01, 0.05, 0.06],
            ],
            dtype=np.float32,
        )
        mod = sta_norm @ w_mod
        scale = 1.0 + 0.05 * np.tanh(mod)
        bias = 0.02 * mod
        dyn_4 = daily_4[None, :, :] * scale[:, None, :] + bias[:, None, :]

        pre_mat = dyn_4[..., 0]
        lst_mat = dyn_4[..., 1]
        pet_mat = dyn_4[..., 2]
        lai_mat = dyn_4[..., 3]
        sm_mat = sm_daily[None, :] + 0.03 * np.tanh(
            sta_norm @ np.array([0.08, 0.06, -0.03, 0.02, 0.04, 0.05, -0.02, 0.03], dtype=np.float32)
        )[:, None]
        dynamic_source_desc = "fallback_hmm_csv"

    # ----------------------------
    # Monthly human activity data
    # ----------------------------
    hum_df = pd.read_parquet(path_human, columns=["Grid_ID", "Year", "Month", "irr", "manu", "dom", "ele"])
    hum_df = hum_df[hum_df["Grid_ID"].isin(grid_ids)].copy()
    if hum_df.empty:
        raise DataSourceFallbackError("Filtered human activity data is empty.")

    hum_df["month_idx"] = (hum_df["Year"] - hum_df["Year"].min()) * 12 + (hum_df["Month"] - 1)
    hum_df = hum_df.groupby(["Grid_ID", "month_idx"], as_index=False)[["irr", "manu", "dom", "ele"]].mean()
    month_index = np.arange(hum_df["month_idx"].min(), hum_df["month_idx"].max() + 1, dtype=np.int64)

    monthly_list = []
    for var in ["irr", "manu", "dom", "ele"]:
        piv = hum_df.pivot(index="Grid_ID", columns="month_idx", values=var)
        piv = piv.reindex(index=grid_ids, columns=month_index, fill_value=0.0)
        monthly_list.append(piv.to_numpy(dtype=np.float32))
    X_hum_monthly = np.stack(monthly_list, axis=-1).astype(np.float32)

    month_dates = pd.date_range(start=date_index.min().replace(day=1), periods=X_hum_monthly.shape[1], freq="MS")
    month_day_index = (month_dates - date_index.min()).days.to_numpy(dtype=np.float32)

    X_hum_daily = interpolate_monthly_to_daily(
        monthly_data=X_hum_monthly,
        target_len=T,
        month_day_index=month_day_index,
        clip_min=0.0,
        method=cfg.human_interp_method,
    )

    # The supervision target must remain true SM to avoid limiting accuracy with synthetic labels.
    Y_core = sm_mat.astype(np.float32)
    Y_SM = Y_core[..., None].astype(np.float32)

    X_dyn = np.stack([pre_mat, lst_mat, pet_mat, lai_mat], axis=-1).astype(np.float32)

    hmm_state_by_date = hmm_df[["Date", "State"]].drop_duplicates("Date").set_index("Date")["State"]
    hmm_state = hmm_state_by_date.reindex(date_index).ffill().bfill().to_numpy(dtype=np.int64)
    causal_df = pd.read_csv(path_causal)
    W_causal = _build_causal_mask_from_edges(causal_df, Cluster, hmm_state)

    arrays = {
        "Y_SM": Y_SM,
        "X_sta": X_sta.astype(np.float32),
        "Lat_Lon": Lat_Lon.astype(np.float32),
        "Cluster": Cluster.astype(np.int64),
        "X_dyn": X_dyn.astype(np.float32),
        "DOY": DOY.astype(np.float32),
        "Grid_ID": grid_ids.astype(np.int64),
        "Date": date_index.strftime("%Y-%m-%d").to_numpy(),
        "Time_Norm": Time_Norm.astype(np.float32),
        "W_causal": W_causal.astype(np.float32),
        "X_hum_monthly": X_hum_monthly.astype(np.float32),
    }
    return arrays, month_day_index.astype(np.float32), dynamic_source_desc


def _maybe_export_large_csv_to_parquet(csv_path: Path, size_mb_threshold: float = 10.0) -> Optional[Path]:
    """
    If the CSV is large, also export a parquet next to this file.
    """
    if not csv_path.exists():
        return None
    size_mb = csv_path.stat().st_size / (1024 * 1024)
    if size_mb <= size_mb_threshold:
        return None

    df = pd.read_csv(csv_path)
    parquet_path = Path(__file__).resolve().parent / f"{csv_path.stem}.parquet"
    df.to_parquet(parquet_path, index=False)
    return parquet_path


def _build_dataset_and_loaders(
    cfg: ExperimentConfig,
) -> Tuple[EcohydroDataset, object, object, object, Dict[str, torch.Tensor], Dict[str, np.ndarray], str]:
    """Build dataset and dataloaders, and return the dynamic data source description."""
    source_desc = "mock"
    try:
        arrays, month_day_index, source_desc = build_real_arrays(cfg)
        print(f"Real data build succeeded, dynamic source: {source_desc}")
    except (FileNotFoundError, pd.errors.EmptyDataError, DataSourceFallbackError) as e:
        print(f"Real data build failed; falling back to mock data. Reason: {e}")
        arrays = build_mock_arrays(n_grid=1000, n_time=1826, t_month=60, seed=cfg.random_seed)
        month_day_index = None

    dataset = EcohydroDataset(
        **arrays,
        loc_enc_dim=cfg.n_loc_enc,
        time_enc_dim=cfg.n_time_enc,
        month_day_index=month_day_index,
        human_interp_method=cfg.human_interp_method,
    )

    train_loader, val_loader, test_loader, mask_dict, index_dict = get_stratified_dataloaders(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        random_state=cfg.random_seed,
    )
    return dataset, train_loader, val_loader, test_loader, mask_dict, index_dict, source_desc


def _build_model_from_config(cfg: ExperimentConfig) -> CausalMambaODE:
    """Create a model that matches the current experiment configuration."""
    return CausalMambaODE(
        d_hidden=cfg.d_hidden,
        n_dyn=cfg.n_dyn,
        n_time_enc=cfg.n_time_enc,
        kan_num_basis=cfg.kan_num_basis,
        dropout=cfg.dropout,
        use_kan=cfg.use_kan,
        film_gamma_mode=cfg.film_gamma_mode,
        mamba_static_context_dim=cfg.mamba_static_context_dim,
        use_soft_causal_gating=cfg.use_soft_causal_gating,
        causal_gate_floor=cfg.causal_gate_floor,
        inject_sm0_to_mamba=cfg.inject_sm0_to_mamba,
        normalize_dynamic_for_encoder=cfg.normalize_dynamic_for_encoder,
        ode_param_scale_alpha=cfg.ode_param_scale_alpha,
        ode_param_scale_theta=cfg.ode_param_scale_theta,
        ode_param_scale_eta_pet=cfg.ode_param_scale_eta_pet,
        ode_param_scale_phi_lai=cfg.ode_param_scale_phi_lai,
        ode_use_dsm_clip=cfg.ode_use_dsm_clip,
        ode_dsm_clip_value=cfg.ode_dsm_clip_value,
        ode_enable_delta_residual=cfg.ode_enable_delta_residual,
        ode_delta_residual_scale=cfg.ode_delta_residual_scale,
        ode_enable_sm_residual=cfg.ode_enable_sm_residual,
        ode_sm_residual_scale=cfg.ode_sm_residual_scale,
    )


def run_pipeline(cfg: ExperimentConfig) -> None:
    """Full pipeline: build data -> stratified split -> train/eval -> export."""
    dataset, train_loader, val_loader, test_loader, mask_dict, index_dict, source_desc = _build_dataset_and_loaders(cfg)

    print(
        "Run summary:",
        {
            "dynamic_source": source_desc,
            "seed": cfg.random_seed,
            "num_workers": cfg.num_workers,
            "pin_memory": cfg.pin_memory,
            "use_amp": cfg.use_amp,
            "strict_dynamic_parquet": cfg.strict_dynamic_parquet,
            "resume_from_best": cfg.resume_from_best,
            "resume_optimizer_state": cfg.resume_optimizer_state,
            "overfit_patience": cfg.overfit_patience,
            "gradient_clip_norm": cfg.gradient_clip_norm,
            "test_eval_interval": cfg.test_eval_interval,
            "loss_start_idx": cfg.loss_start_idx,
            "derivative_loss_weight": cfg.derivative_loss_weight,
        },
    )
    print(
        "Stratified split done: "
        f"train={int(mask_dict['train'].sum())}, "
        f"val={int(mask_dict['val'].sum())}, "
        f"test={int(mask_dict['test'].sum())}"
    )
    print(
        "Index preview:",
        {
            "train_idx_head": index_dict["train_idx"][:5].tolist(),
            "val_idx_head": index_dict["val_idx"][:5].tolist(),
            "test_idx_head": index_dict["test_idx"][:5].tolist(),
        },
    )

    model = _build_model_from_config(cfg)

    result = train_and_evaluate(
        model=model,
        dataset=dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        mask_dict=mask_dict,
        output_dir=cfg.output_dir,
        batch_size=cfg.batch_size,
        eval_batch_size=cfg.eval_batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        max_epochs=cfg.max_epochs,
        early_stop_patience=cfg.early_stop_patience,
        resume_from_best=cfg.resume_from_best,
        resume_optimizer_state=cfg.resume_optimizer_state,
        overfit_patience=cfg.overfit_patience,
        overfit_val_rise_ratio=cfg.overfit_val_rise_ratio,
        overfit_train_drop_min_ratio=cfg.overfit_train_drop_min_ratio,
        test_eval_interval=cfg.test_eval_interval,
        loss_start_idx=cfg.loss_start_idx,
        derivative_loss_weight=cfg.derivative_loss_weight,
        derivative_smooth_window=cfg.derivative_smooth_window,
        use_amp=cfg.use_amp,
        gradient_clip_norm=cfg.gradient_clip_norm,
        lr_scheduler_patience=cfg.lr_scheduler_patience,
        lr_scheduler_factor=cfg.lr_scheduler_factor,
        min_lr=cfg.min_lr,
        checkpoint_version=cfg.checkpoint_version,
        export_daily_results=cfg.export_daily_results,
    )

    metrics_csv = cfg.output_dir / "test_metrics.csv"
    metrics_parquet = _maybe_export_large_csv_to_parquet(metrics_csv, size_mb_threshold=10.0)

    print("Training and evaluation completed, key results:", result)
    print(f"Output directory: {cfg.output_dir}")
    if metrics_parquet is not None:
        print(f"Metrics CSV is large; also exported parquet next to code: {metrics_parquet}")
    print(
        "Exported files:",
        [
            str(cfg.output_dir / "loss_curves.png"),
            str(cfg.output_dir / "loss_curves.jpg"),
            str(cfg.output_dir / "loss_curves.pdf"),
            str(cfg.output_dir / "test_metrics.csv"),
            str(cfg.output_dir / "mamba_ode_best.pth"),
            str(cfg.output_dir / "mamba_ode_last.pth"),
            str(cfg.output_dir / "ode_results.parquet"),
        ],
    )


def export_from_checkpoint(
    cfg: ExperimentConfig,
    checkpoint_path: Optional[Path] = None,
    export_path: Optional[Path] = None,
) -> None:
    """Load an existing checkpoint and export the full daily parquet without retraining."""
    dataset, train_loader, _, _, mask_dict, index_dict, source_desc = _build_dataset_and_loaders(cfg)
    checkpoint_path = cfg.output_dir / "mamba_ode_best.pth" if checkpoint_path is None else Path(checkpoint_path)
    export_path = cfg.output_dir / "ode_results.parquet" if export_path is None else Path(export_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    model = _build_model_from_config(cfg).to(dev)
    try:
        ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=dev)

    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)

    eval_batch_size = int(cfg.eval_batch_size) if cfg.eval_batch_size is not None else max(cfg.batch_size, cfg.batch_size * 2)
    full_kwargs = {
        "batch_size": eval_batch_size,
        "shuffle": False,
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
        "collate_fn": getattr(train_loader, "collate_fn", None),
    }
    if cfg.num_workers > 0:
        full_kwargs["persistent_workers"] = True
    full_loader = DataLoader(dataset, **full_kwargs)

    print(
        "Export-only mode:",
        {
            "dynamic_source": source_desc,
            "checkpoint": str(checkpoint_path),
            "export_path": str(export_path),
            "device": str(dev),
            "batch_size": eval_batch_size,
            "use_amp": bool(cfg.use_amp and dev.type == "cuda"),
            "train_idx_head": index_dict["train_idx"][:5].tolist(),
        },
    )
    _export_daily_outputs_streaming(
        model=model,
        loader=full_loader,
        device=dev,
        use_amp=bool(cfg.use_amp and dev.type == "cuda"),
        parquet_path=export_path,
        mask_dict=mask_dict,
        n_grid=len(dataset),
    )
    print(f"Full daily parquet exported: {export_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Causal-Gated Mamba-ODE training/export entrypoint.")
    parser.add_argument("--export-only", action="store_true", help="Only load an existing checkpoint and export full ode_results.parquet without training.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint path for export; default outputs/mamba_ode_best.pth.")
    parser.add_argument("--export-path", type=Path, default=None, help="Full parquet output path; default outputs/ode_results.parquet.")
    parser.add_argument("--export-daily-results", action="store_true", help="Also export full daily parquet after training.")
    args = parser.parse_args()

    config = ExperimentConfig()
    if args.export_daily_results:
        config.export_daily_results = True
    _set_global_seed(
        seed=config.random_seed,
        deterministic=config.deterministic,
        cudnn_benchmark=config.cudnn_benchmark,
    )
    if args.export_only:
        export_from_checkpoint(config, checkpoint_path=args.checkpoint, export_path=args.export_path)
    else:
        run_pipeline(config)
