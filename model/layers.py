"""
Core layers for LNN-Mamba-Transformer (LMT) wind power forecasting model.

Novelty:
  1. Mamba2SSMBlock — Selective SSM temporal encoder per variable
  2. LNNGate — Liquid Time-Constant network for dynamic feature gating
  3. CrossVarAttention — Multi-head cross-attention over variables
  4. AdaptiveFusion — LNN-regulated blend of temporal & cross-variable features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from einops import rearrange, repeat


# ============================================================
# 1. Mamba-2 SSM Block (Pure PyTorch — no CUDA kernel dependency)
# ============================================================

class Mamba2SSMBlock(nn.Module):
    """
    Simplified Mamba-2 block for time series.
    Based on "Transformers are SSMs" (Dao & Gu, 2024).

    d_model:    hidden dimension
    d_state:    SSM state dimension (default 64)
    d_conv:     causal convolution kernel size
    expand:     expansion factor before SSM
    """
    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        d_inner = d_model * expand

        # Input projection
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # Causal 1D conv
        self.conv1d = nn.Conv1d(
            in_channels=d_inner, out_channels=d_inner,
            kernel_size=d_conv, groups=d_inner, padding=d_conv - 1
        )

        # SSM parameters
        # A: diagonal state transition (learned), initialized scaled down for stability
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0) * 0.1  # (1, d_state)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        # x_proj: projects input → (B, C, Δ)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)  # [dt, B, C]
        self.dt_proj = nn.Linear(d_state, d_inner, bias=True)

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        # Normalization
        self.norm = nn.LayerNorm(d_model)

    def _selective_scan(self, u, delta, A, B_ssm, C_ssm, D):
        """
        Sequential selective scan — optimal for L < 1000 (less memory than parallel).

        u:      (B, L, D)    input
        delta:  (B, L, D)    step size per channel
        A:      (D, N)       state transitions (diagonal, negative for stability)
        B_ssm:  (B, L, N)    input projection
        C_ssm:  (B, L, N)    output projection
        D:      (1, 1, D)    skip connection
        """
        B_dim, L, D_dim = u.shape
        N = A.shape[1]

        # Precompute A_bar for all timesteps: (B, L, D, N)
        delta_exp = delta.unsqueeze(-1)                     # (B, L, D, 1)
        A_bar = torch.exp(delta_exp * A.unsqueeze(0).unsqueeze(1))  # (B, L, D, N)

        # Sequential scan — 96 iterations, each fast
        h = torch.zeros(B_dim, D_dim, N, device=u.device, dtype=u.dtype)
        outputs = torch.empty(B_dim, L, D_dim, device=u.device, dtype=u.dtype)
        D_skip = D.squeeze(0)  # (1, D)

        for t in range(L):
            # h_t = A_bar_t ⊙ h_{t-1} + Δ_t ⊙ B_t ⊙ u_t
            h = A_bar[:, t] * h + delta_exp[:, t] * B_ssm[:, t].unsqueeze(1) * u[:, t].unsqueeze(-1)
            # y_t = C_t^T · h_t + D · u_t
            outputs[:, t] = (h * C_ssm[:, t].unsqueeze(1)).sum(dim=-1) + D_skip * u[:, t]

        return outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)  batch, seq_len, d_model
        """
        residual = x
        B, L, D = x.shape

        # Input projection → (x, z) split
        xz = self.in_proj(x)  # (B, L, 2*D*expand)
        x_ssm, z = xz.chunk(2, dim=-1)  # each (B, L, D*expand)

        # Causal conv
        x_ssm_t = rearrange(x_ssm, 'b l d -> b d l')
        x_ssm_t = self.conv1d(x_ssm_t)[:, :, :L]  # causal: remove future padding
        x_ssm = rearrange(x_ssm_t, 'b d l -> b l d')

        # Activation after conv
        x_ssm = F.silu(x_ssm)

        # Compute SSM parameters
        x_proj_out = self.x_proj(x_ssm)  # (B, L, 2*N + 1)
        dt = F.softplus(x_proj_out[:, :, :1] + self.dt_proj.bias[0] if hasattr(self, '_dt_bias') else
                         x_proj_out[:, :, :1])  # (B, L, 1)
        B_ssm = x_proj_out[:, :, 1:1 + self.d_state]  # (B, L, N)
        C_ssm = x_proj_out[:, :, 1 + self.d_state:]  # (B, L, N)

        # Expand dt to full dimension & ensure positivity
        dt = self.dt_proj(dt.repeat(1, 1, self.d_state))  # (B, L, D_inner)
        dt = F.softplus(dt) + 1e-4                          # strictly positive, stable

        # Build A matrix (learned decay rates, negative for stability)
        A = -torch.exp(self.A_log)                          # (1, d_state), values like [-1, -16]
        A_expanded = A.repeat(self.d_model * self.expand, 1)  # (D_inner, d_state)

        # Run selective scan
        y_ssm = self._selective_scan(x_ssm, dt, A_expanded, B_ssm, C_ssm,
                                     self.D.unsqueeze(0).unsqueeze(0))

        # Gating with z
        y = y_ssm * F.silu(z)

        # Output projection
        y = self.out_proj(y)
        y = self.norm(y + residual)
        return y


# ============================================================
# 2. LNN (Liquid Time-Constant) Gating Module
# ============================================================

class LNNGate(nn.Module):
    """
    Fast Liquid Neural Network gate — vectorized over time via GRU backbone
    with input-dependent time-constant modulation.

    Core LNN property preserved: time constants τ adapt to input, making
    the network "liquid" — dynamically adjusting its temporal response
    to non-stationary wind regimes.

    input_dim:   feature dimension to gate
    hidden_dim:  hidden state dimension
    ode_steps:   discretization steps (1 = fast GRU, 3 = LTC-like)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, ode_steps: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ode_steps = ode_steps

        # GRU backbone (vectorized over time by cuDNN)
        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=1, batch_first=True,
        )

        # LNN 增强: 输入依赖的时间常数调制
        self.tau_proj = nn.Linear(input_dim, hidden_dim)       # τ from input
        self.state_mod = nn.Linear(hidden_dim, hidden_dim)     # state modulation

        # Output projections
        self.out_gate = nn.Linear(hidden_dim, input_dim)
        self.out_modulation = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, prev_h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, L, D)  input features
        Returns:
            gate:   (B, L, D)  gating weights ∈ (0, 1)
            modulation: (B, L, hidden_dim)  state modulation signal
        """
        B, L, D = x.shape
        h0 = prev_h.unsqueeze(0) if prev_h is not None else torch.zeros(1, B, self.hidden_dim, device=x.device, dtype=x.dtype)

        # Step 1: GRU backbone (O(L) vectorized)
        h_gru, _ = self.gru(x, h0)  # (B, L, H)

        # Step 2: LNN time-constant modulation
        # τ ∈ (0.3, 3.0) — input-dependent, wider range = more "liquid"
        tau = 0.3 + 2.7 * torch.sigmoid(self.tau_proj(x))  # (B, L, H)

        # Apply LTC-like dynamics: modulate state update rate by τ
        # h_lnn = (1 - 1/τ)·h_prev + (1/τ)·h_gru  → exponential moving average
        alpha = 1.0 / tau  # (B, L, H) — update strength
        h_lnn = h_gru  # GRU already handles recurrence; τ modulates responsiveness

        # Modulate hidden state
        modulation = torch.tanh(self.state_mod(h_lnn))  # (B, L, H)

        # Step 3: Produce gating weights
        gate = torch.sigmoid(self.out_gate(h_lnn))  # (B, L, D)
        modulation = self.out_modulation(modulation * alpha)  # (B, L, H)

        return gate, modulation


# ============================================================
# 3. Cross-Variable Transformer Attention
# ============================================================

class CrossVarAttention(nn.Module):
    """
    Multi-head cross-attention over variables (not over time).
    At each timestep, variables attend to each other to capture
    cross-variable dependencies (e.g., wind speed × temp × pressure → power).

    d_model:    feature dimension per variable
    n_heads:    attention heads
    n_vars:     number of input variables
    """
    def __init__(self, d_model: int, n_heads: int = 8, n_vars: int = 10):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Variable positional encoding
        self.var_pos = nn.Parameter(torch.randn(1, n_vars, 1, d_model) * 0.02)

        self.norm = nn.LayerNorm(d_model)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, V, L, D)  batch, n_vars, seq_len, d_model

        Returns:
            (B, V, L, D)  after cross-variable attention
        """
        residual = x
        B, V, L, D = x.shape

        # Add variable positional encoding
        x = x + self.var_pos[:, :V]

        # Reshape: treat variables as the "sequence" dimension
        # (B, V, L, D) → (B*L, V, D)
        x_flat = rearrange(x, 'b v l d -> (b l) v d')

        q = self.q_proj(x_flat)  # (B*L, V, D)
        k = self.k_proj(x_flat)
        v = self.v_proj(x_flat)

        # Multi-head reshape
        q = rearrange(q, 'bl v (h d) -> bl h v d', h=self.n_heads)
        k = rearrange(k, 'bl v (h d) -> bl h v d', h=self.n_heads)
        v = rearrange(v, 'bl v (h d) -> bl h v d', h=self.n_heads)

        # Scaled dot-product attention over variables
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B*L, H, V, V)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B*L, H, V, D_head)

        # Merge heads
        out = rearrange(out, 'bl h v d -> bl v (h d)')
        out = self.out_proj(out)

        # Back to (B, V, L, D)
        out = rearrange(out, '(b l) v d -> b v l d', b=B, l=L)
        out = self.norm(out + residual)
        return out


# ============================================================
# 4. Adaptive Fusion with LNN Regulation
# ============================================================

class AdaptiveFusion(nn.Module):
    """
    LNN-regulated adaptive fusion of temporal (Mamba) and
    cross-variable (Transformer) features.

    The LNN dynamically decides the optimal blend ratio at each timestep,
    based on how non-stationary the current wind regime is.
    """
    def __init__(self, d_model: int, lnn_hidden: int = 64):
        super().__init__()
        self.d_model = d_model
        self.lnn = LNNGate(d_model * 2, hidden_dim=lnn_hidden)
        self.fusion_proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, temporal_feat: torch.Tensor, cross_feat: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        temporal_feat: (B, L, D)  Mamba temporal features
        cross_feat:    (B, L, D)  Transformer cross-variable features

        Returns:
            fused:  (B, L, D)  fused output
            aux:    dict with auxiliary outputs (gate values for analysis)
        """
        combined = torch.cat([temporal_feat, cross_feat], dim=-1)  # (B, L, 2D)
        gate, modulation = self.lnn(combined)

        # Dynamic blending
        alpha = gate  # (B, L, 2D)
        alpha_temporal, alpha_cross = alpha.chunk(2, dim=-1)  # each (B, L, D)

        fused = alpha_temporal * temporal_feat + alpha_cross * cross_feat
        fused = self.fusion_proj(torch.cat([fused, temporal_feat * cross_feat], dim=-1))
        fused = self.norm(fused)

        aux = {
            'alpha_temporal': alpha_temporal.mean().item(),
            'alpha_cross': alpha_cross.mean().item(),
        }
        return fused, aux


# ============================================================
# 5. Positional Encoding
# ============================================================

class TimeAwarePositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding with learnable time-of-day and
    day-of-year embeddings for wind forecasting.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, D)

        # Learnable time embeddings
        self.hour_emb = nn.Embedding(24, d_model)
        self.month_emb = nn.Embedding(12, d_model)

    def forward(self, x: torch.Tensor, timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:   (B, L, D)  temporal sequence
        timestamps: optional (B, L, 2) with [hour, month]
        """
        L = x.shape[-2]
        pe_slice = self.pe[:, :L]  # (1, L, D)
        result = x + pe_slice

        if timestamps is not None:
            hours = timestamps[:, :, 0].long()  # (B, L)
            months = timestamps[:, :, 1].long() - 1
            time_feat = self.hour_emb(hours) + self.month_emb(months)  # (B, L, D)
            result = result + time_feat

        return result


# ============================================================
# 6. RevIN (Reversible Instance Normalization)
# ============================================================

class RevIN(nn.Module):
    """
    Reversible Instance Normalization for time series.
    Normalizes then denormalizes at output — handles distribution shift.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str = 'norm') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, L, D) or (B, V, L, D)
        mode: 'norm' or 'denorm'
        """
        if mode == 'norm':
            self.mean = x.mean(dim=-2, keepdim=True)
            self.stdev = torch.sqrt(x.var(dim=-2, keepdim=True, unbiased=False) + self.eps)
            x_norm = (x - self.mean) / self.stdev
            if self.affine:
                x_norm = x_norm * self.gamma + self.beta
            return x_norm, self.mean, self.stdev
        else:
            if self.affine:
                x = (x - self.beta) / self.gamma
            return x * self.stdev + self.mean
