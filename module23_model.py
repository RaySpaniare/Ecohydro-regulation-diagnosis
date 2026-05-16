# -*- coding: utf-8 -*-
'''
@File    :   module23_model.py
@Time    :   2026-04-25
@Desc    :   This file implements the core network structures for modules 2 and 3: (1) a three-layer simplified KAN static backbone (GeoHypernet) to model the highly nonlinear coupling among terrain, soil, spatial location, and geomorphic clusters, outputting FiLM gamma/beta modulation parameters; (2) the causal-prior-gated dynamic encoder (CausalGatedMamba), which applies element-wise physical gating to daily dynamic variables using causal weights, concatenates temporal periodic encodings, and feeds a Mamba-style sequence block to extract dynamic features. The implementation is strictly vectorized with no loops over spatial N or temporal T. For engineering robustness, when the mamba_ssm library is unavailable, it falls back to a fully vectorized "depthwise separable convolution + gating" placeholder module, preserving the input/output contract so the end-to-end pipeline remains runnable.
@Notice  :   If mamba_ssm is installed, the official Mamba is used; otherwise the placeholder is used. Both outputs share the same dimensions for seamless switching.
'''

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba as MambaSSM  # type: ignore

    HAS_MAMBA_SSM = True
except Exception:
    HAS_MAMBA_SSM = False

try:
    from mambapy.mamba import Mamba as MambaPy  # type: ignore
    from mambapy.mamba import MambaConfig as MambaPyConfig  # type: ignore

    HAS_MAMBAPY = True
except Exception:
    HAS_MAMBAPY = False

# By default, mambapy is disabled in speed-first mode.
ENABLE_MAMBAPY_BACKEND = False


class SimplifiedKANLayer(nn.Module):
    """
    Simplified KAN layer (vectorized implementation).

    Design idea:
    - KAN expands each input dimension with nonlinear basis functions, then linearly
      combines them into the output space.
    - Here we use learnable RBF bases to avoid external pykan dimension complexity.
    """

    def __init__(self, in_dim: int, out_dim: int, num_basis: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_basis = num_basis

        init_centers = torch.linspace(-2.0, 2.0, steps=num_basis).repeat(in_dim, 1)
        self.centers = nn.Parameter(init_centers)  # (in_dim, num_basis)
        self.log_width = nn.Parameter(torch.zeros(in_dim, num_basis))  # Softplus keeps width positive

        self.basis_weight = nn.Parameter(torch.empty(in_dim, num_basis, out_dim))
        nn.init.xavier_uniform_(self.basis_weight)

        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., in_dim)
        Returns:
            y: (..., out_dim)
        """
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_dim)  # shape: (B, in_dim)

        # Compute RBF basis responses
        width = F.softplus(self.log_width) + 1e-4  # shape: (in_dim, num_basis)
        diff = (x_flat[:, :, None] - self.centers[None, :, :]) / width[None, :, :]  # shape: (B, in_dim, num_basis)
        phi = torch.exp(-0.5 * diff * diff)  # shape: (B, in_dim, num_basis)

        # Aggregate basis responses per input dimension into the output space
        basis_out = torch.einsum("bik,iko->bo", phi, self.basis_weight)  # shape: (B, out_dim)
        linear_out = self.linear(x_flat)  # shape: (B, out_dim)

        y = F.silu(linear_out + basis_out)
        y = self.dropout(y)
        return y.reshape(*original_shape, self.out_dim)


class MLPBlock(nn.Module):
    """Simple MLP block used as a stable baseline for noisy geoscience data."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GeoHypernet(nn.Module):
    """
    Module 2: three-layer KAN static network that outputs FiLM gamma/beta.
    """

    def __init__(
        self,
        d_hidden: int,
        num_basis: int = 8,
        dropout: float = 0.1,
        use_kan: bool = True,
        film_gamma_mode: str = "legacy_tanh",
    ) -> None:
        super().__init__()
        layer = (
            (lambda i, o: SimplifiedKANLayer(i, o, num_basis=num_basis, dropout=dropout))
            if use_kan
            else (lambda i, o: MLPBlock(i, o, dropout=dropout))
        )
        self.enc1 = layer(28, 64)
        self.enc2 = layer(64, 128)
        self.enc3 = layer(128, 64)
        self.norm = nn.LayerNorm(64)
        self.film_head = nn.Linear(64, 2 * d_hidden)
        self.film_gamma_mode = str(film_gamma_mode).lower()

    def forward(
        self,
        X_sta: torch.Tensor,
        LatLon_enc: torch.Tensor,
        Cluster_onehot: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass (strict contract):
        1) Concatenate static inputs to get X_static_all
        2) Three-layer KAN extracts static spatial heterogeneity
        3) Map to FiLM gamma/beta
        """
        # Concatenate per spec: shape -> (N, 28)
        X_static_all = torch.cat([X_sta, LatLon_enc, Cluster_onehot], dim=-1)

        # Three-layer KAN: 28 -> 64 -> 128 -> 64
        h = self.enc1(X_static_all)  # shape: (N, 64)
        h = self.enc2(h)  # shape: (N, 128)
        h = self.enc3(h)  # shape: (N, 64)
        h = self.norm(h)  # shape: (N, 64)

        film_params = self.film_head(h)  # shape: (N, 2*D_hidden)
        gamma_raw, beta = torch.chunk(film_params, chunks=2, dim=-1)  # shape: (N,D_hidden), (N,D_hidden)

        # Keep gamma fluctuating around 1 for training stability
        if self.film_gamma_mode == "legacy_tanh":
            gamma = 1.0 + 0.1 * torch.tanh(gamma_raw)
        elif self.film_gamma_mode == "exp_clamped":
            gamma = torch.exp(torch.clamp(gamma_raw, min=-2.0, max=2.0))
        else:
            raise ValueError(f"Unsupported film_gamma_mode: {self.film_gamma_mode}")
        return gamma, beta, X_static_all


class VectorizedMambaFallback(nn.Module):
    """
    Mamba placeholder module (used when mamba_ssm is unavailable).

    Structure: LayerNorm -> linear expansion -> depthwise temporal conv -> gated fusion
    -> linear projection + residual. Fully vectorized with no loops over N/T.
    """

    def __init__(self, d_model: int, d_conv: int = 4, expand: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        inner = d_model * expand
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, inner * 2)
        self.dw_conv = nn.Conv1d(inner, inner, kernel_size=d_conv, padding=d_conv - 1, groups=inner)
        self.out_proj = nn.Linear(inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, T, D)
        """
        residual = x
        x = self.norm(x)  # shape: (N, T, D)

        uv = self.in_proj(x)  # shape: (N, T, 2*inner)
        u, v = torch.chunk(uv, chunks=2, dim=-1)  # shape: (N,T,inner), (N,T,inner)

        # Depthwise conv mixes local temporal states
        u_conv = self.dw_conv(u.transpose(1, 2))  # shape: (N, inner, T + d_conv - 1)
        u_conv = u_conv[..., : x.size(1)].transpose(1, 2)  # back to (N, T, inner)

        y = F.silu(u_conv) * torch.sigmoid(v)  # shape: (N, T, inner)
        y = self.out_proj(y)  # shape: (N, T, D)
        y = self.dropout(y)
        return residual + y


class MambaPyBlock(nn.Module):
    """mambapy adapter preserving the (N, T, D) input/output contract."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        cfg = MambaPyConfig(
            d_model=d_model,
            n_layers=1,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand,
            use_cuda=torch.cuda.is_available(),
        )
        self.block = MambaPy(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TemporalKANEncoder(nn.Module):
    """Learnable residual temporal encoder for seasonal and long-term time features."""

    def __init__(self, in_dim: int, hidden_dim: int = 16, dropout: float = 0.0, use_kan: bool = True) -> None:
        super().__init__()
        if use_kan:
            self.enc1 = SimplifiedKANLayer(in_dim, hidden_dim, num_basis=4, dropout=dropout)
            self.enc2 = SimplifiedKANLayer(hidden_dim, in_dim, num_basis=4, dropout=dropout)
        else:
            self.enc1 = MLPBlock(in_dim, hidden_dim, dropout=dropout)
            self.enc2 = MLPBlock(hidden_dim, in_dim, dropout=dropout)
        self.res_gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, t_len, feat_dim = x.shape
        x_flat = x.reshape(bsz * t_len, feat_dim)
        residual = self.enc2(self.enc1(x_flat)).reshape(bsz, t_len, feat_dim)
        return x + torch.sigmoid(self.res_gate) * residual


class CausalGatedMamba(nn.Module):
    """
    Module 3: causal-prior gating + Mamba dynamic encoder.
    """

    def __init__(
        self,
        n_dyn: int = 4,
        n_time_enc: int = 9,
        d_hidden: int = 64,
        dropout: float = 0.1,
        static_context_dim: int = 16,
        use_soft_causal_gating: bool = True,
        causal_gate_floor: float = 0.2,
        inject_sm0: bool = True,
        use_kan: bool = True,
        normalize_dynamic_input: bool = True,
    ) -> None:
        super().__init__()
        self.n_dyn = n_dyn
        self.n_time_enc = n_time_enc
        self.d_hidden = d_hidden
        self.static_context_dim = max(int(static_context_dim), 0)
        self.use_soft_causal_gating = bool(use_soft_causal_gating)
        self.causal_gate_floor = float(causal_gate_floor)
        self.inject_sm0 = bool(inject_sm0)
        self.normalize_dynamic_input = bool(normalize_dynamic_input)

        self.time_encoder = TemporalKANEncoder(n_time_enc, hidden_dim=16, dropout=dropout, use_kan=use_kan)
        in_dim = n_dyn + n_time_enc + self.static_context_dim + (1 if self.inject_sm0 else 0)
        self.static_proj = nn.Linear(28, self.static_context_dim) if self.static_context_dim > 0 else None
        self.input_norm = nn.LayerNorm(in_dim)
        self.input_proj = nn.Linear(in_dim, d_hidden)
        self.pre_norm = nn.LayerNorm(d_hidden)
        self.post_norm = nn.LayerNorm(d_hidden)
        self.backend_name = "vectorized_fallback"

        if HAS_MAMBA_SSM:
            self.mamba_block = MambaSSM(d_model=d_hidden, d_state=16, d_conv=4, expand=2)
            self.backend_name = "mamba_ssm"
        elif HAS_MAMBAPY and ENABLE_MAMBAPY_BACKEND:
            self.mamba_block = MambaPyBlock(d_model=d_hidden, d_state=16, d_conv=4, expand=2)
            self.backend_name = "mambapy"
            warnings.warn("mamba_ssm not detected; switched to mambapy backend.", RuntimeWarning)
        else:
            warnings.warn("mamba_ssm/mambapy not enabled; using vectorized Mamba fallback.", RuntimeWarning)
            self.mamba_block = VectorizedMambaFallback(d_model=d_hidden, d_conv=4, expand=2, dropout=dropout)

        print(f"CausalGatedMamba backend: {self.backend_name}")

    def forward(
        self,
        X_dyn: torch.Tensor,
        W_causal: torch.Tensor,
        X_time_enc: torch.Tensor,
        X_static_all: Optional[torch.Tensor] = None,
        SM_0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            X_dyn: (N, T, 4)  -> Pre, LST, PET, LAI
            W_causal: (N, T, 4)
            X_time_enc: (1, T, p) or (N, T, p). Time features are grid-invariant.

        Returns:
            H_dyn: (N, T, D_hidden)
            X_dyn_gated: (N, T, 4)
        """
        # ----------------------------
        # 1) Causal gating (Hadamard element-wise)
        # ----------------------------
        # Physical meaning:
        # W_causal indicates the trusted physical strength of each variable's effect on SM
        # under the current geomorphic-climate state. X_dyn * W_causal suppresses signals
        # without causal support, reducing spurious correlations.
        X_dyn_norm = X_dyn
        if self.use_soft_causal_gating:
            gate = self.causal_gate_floor + (1.0 - self.causal_gate_floor) * W_causal
            X_dyn_gated = X_dyn_norm * gate
        else:
            X_dyn_gated = X_dyn_norm * W_causal

        # ----------------------------
        # 2) Concatenate with time encoding and map to hidden space
        # ----------------------------
        # Time encoding is identical for every grid, so encode it once and expand.
        # This removes repeated B*T temporal MLP work and cuts host-to-device traffic.
        X_time_base = X_time_enc[:1]
        X_time_refined = self.time_encoder(X_time_base).expand(X_dyn.size(0), -1, -1)
        feats = [X_dyn_gated, X_time_refined]
        if self.inject_sm0:
            if SM_0 is not None:
                sm0_feat = SM_0.unsqueeze(1).expand(-1, X_dyn.size(1), -1)
            else:
                sm0_feat = torch.zeros(
                    X_dyn.size(0),
                    X_dyn.size(1),
                    1,
                    dtype=X_dyn.dtype,
                    device=X_dyn.device,
                )
            feats.append(sm0_feat)
        if self.static_proj is not None:
            if X_static_all is not None:
                static_ctx = self.static_proj(X_static_all).unsqueeze(1).expand(-1, X_dyn.size(1), -1)
            else:
                static_ctx = torch.zeros(
                    X_dyn.size(0),
                    X_dyn.size(1),
                    self.static_context_dim,
                    dtype=X_dyn.dtype,
                    device=X_dyn.device,
                )
            feats.append(static_ctx)
        X_in = torch.cat(feats, dim=-1)  # shape: (N, T, 4+p+ctx)
        X_in = self.input_norm(X_in)
        H = self.input_proj(X_in)  # shape: (N, T, D_hidden)
        H = self.pre_norm(H)  # shape: (N, T, D_hidden)

        # ----------------------------
        # 3) Mamba feature extraction
        # ----------------------------
        H_dyn = self.mamba_block(H)  # shape: (N, T, D_hidden)
        H_dyn = self.post_norm(H_dyn)  # shape: (N, T, D_hidden)
        return H_dyn, X_dyn_gated


if __name__ == "__main__":
    torch.manual_seed(42)

    # mock dimensions
    N, T, D_hidden = 128, 365, 64

    X_sta = torch.randn(N, 8)
    LatLon_enc = torch.randn(N, 16)
    Cluster_onehot = F.one_hot(torch.randint(0, 4, (N,)), num_classes=4).float()

    X_dyn = torch.randn(N, T, 4)
    W_causal = torch.rand(N, T, 4)
    X_time_enc = torch.randn(N, T, 9)

    geo = GeoHypernet(d_hidden=D_hidden, num_basis=8, dropout=0.1)
    gamma, beta, x_static_all = geo(X_sta, LatLon_enc, Cluster_onehot)
    print("X_static_all:", tuple(x_static_all.shape))
    print("gamma/beta:", tuple(gamma.shape), tuple(beta.shape))

    dyn = CausalGatedMamba(n_dyn=4, n_time_enc=9, d_hidden=D_hidden, dropout=0.1)
    H_dyn, X_dyn_gated = dyn(X_dyn, W_causal, X_time_enc)
    print("X_dyn_gated:", tuple(X_dyn_gated.shape))
    print("H_dyn:", tuple(H_dyn.shape))
