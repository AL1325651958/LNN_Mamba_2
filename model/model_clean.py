"""
Clean LNN-Gated Selective SSM model — minimal, fast, verifiable.

Architecture stack (each added incrementally and measured):
  Level 0: GRU baseline
  Level 1: Mamba2 SSM (no LNN, no cross-var)
  Level 2: + Spectral loss
  Level 3: + LNN gating
  Level 4: + Cross-var attention (optional)

Design principles:
  - Single site (best data), stride=1 for max samples
  - Pred_len=24 (6h) for fast iteration
  - d_model=64, small state for speed
  - Batch=64 for GPU utilization
  - Every component can be toggled off for ablation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


# ═══════════════════════════════════════════════════════
# Fast Selective SSM — optimized for L < 200
# ═══════════════════════════════════════════════════════

class FastMambaBlock(nn.Module):
    """Minimal Mamba-2: input proj → conv → SSM scan → gate → output."""
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = d_model * expand
        self.d_state = d_state
        self.d_inner = d_inner

        self.in_proj  = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1)
        self.x_proj   = nn.Linear(d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj  = nn.Linear(d_state, d_inner, bias=True)
        A = torch.arange(1, d_state+1, dtype=torch.float32).unsqueeze(0) * 0.05
        self.A_log    = nn.Parameter(torch.log(A))
        self.D        = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm     = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        residual = x

        # Project & split
        xz = self.in_proj(x)  # (B, L, 2*D_inner)
        u, z = xz.chunk(2, dim=-1)  # each (B, L, D_inner)

        # Causal conv
        u = u.transpose(1, 2)  # (B, D_inner, L)
        u = self.conv1d(u)[:, :, :L]
        u = F.silu(u.transpose(1, 2))  # (B, L, D_inner)

        # SSM parameters
        proj = self.x_proj(u)  # (B, L, 2*d_state+1)
        dt   = F.softplus(proj[:, :, :self.d_state])  # (B, L, d_state)
        B_s  = proj[:, :, self.d_state:self.d_state*2]
        C_s  = proj[:, :, self.d_state*2:]

        # dt expand to d_inner
        dt = F.softplus(self.dt_proj(dt)) + 1e-4  # (B, L, d_inner)

        # Build A: (d_inner, d_state) — diagonal, negative
        A = -torch.exp(self.A_log)  # (1, d_state)
        # For efficiency: use same A for all d_inner channels
        # (this is a simplification — full Mamba has per-channel A)

        # Discretize: A_bar = exp(Δ ⊗ A)
        dt_exp = dt.unsqueeze(-1)  # (B, L, D_inner, 1)
        A_bar  = torch.exp(dt_exp * A.unsqueeze(0).unsqueeze(1))  # (B, L, D_inner, d_state)

        # Parallel scan — use cumprod associative scan
        # This avoids the Python for-loop over L
        eps = 1e-8
        log_A = torch.log(A_bar.clamp(min=eps))          # (B, L, D_inner, d_state)
        cum_log_A = torch.cumsum(log_A, dim=1)            # (B, L, D_inner, d_state)

        # b_t = Δ_t ⊗ B_t ⊗ u_t
        b_t = dt_exp * B_s.unsqueeze(2) * u.unsqueeze(-1)  # (B, L, D_inner, d_state)

        # h_t = cumprod(a) * cumsum(b / cumprod(a))
        A_cum = torch.exp(cum_log_A)
        b_scaled = b_t / A_cum.clamp(min=eps)
        cum_b = torch.cumsum(b_scaled, dim=1)
        h = A_cum * cum_b  # (B, L, D_inner, d_state)

        # y_t = C_t^T · h_t
        y = (h * C_s.unsqueeze(2)).sum(dim=-1)  # (B, L, D_inner)
        y = y + self.D.unsqueeze(0).unsqueeze(0) * u  # skip connection

        # Gate & output
        y = y * F.silu(z)
        y = self.out_proj(y)
        return self.norm(y + residual)


# ═══════════════════════════════════════════════════════
# Fast LNN Gate
# ═══════════════════════════════════════════════════════

class FastLNNGate(nn.Module):
    """Lightweight GRU-based gate with input-dependent time constant modulation."""
    def __init__(self, d_model: int, hidden: int = 32):
        super().__init__()
        self.gru = nn.GRU(d_model, hidden, batch_first=True)
        self.tau_mod = nn.Linear(d_model, hidden)
        self.out = nn.Linear(hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)  # (B, L, H)
        tau = 0.5 + 2.5 * torch.sigmoid(self.tau_mod(x))  # (B, L, H)
        return torch.sigmoid(self.out(h / tau))  # (B, L, D)


# ═══════════════════════════════════════════════════════
# Clean LNN-Gated Selective SSM Model
# ═══════════════════════════════════════════════════════

class CleanLNNMamba(nn.Module):
    """
    Levels (controlled by flags):
      L0: input_proj → GRU → decoder  (baseline)
      L1: input_proj → Mamba → decoder
      L2: L1 + spectral_loss
      L3: L2 + LNN gate
    """
    def __init__(self,
        n_vars: int = 11,
        d_model: int = 64,
        n_blocks: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        pred_len: int = 24,
        dropout: float = 0.1,
        use_lnn: bool = True,
        use_spectral: bool = True,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.pred_len = pred_len
        self.use_spectral = use_spectral

        # Input: per-variable embedding + concat
        self.var_emb = nn.Linear(1, d_model)
        self.input_fusion = nn.Linear(n_vars * d_model, d_model)

        # Positional
        self.register_buffer('pos_emb', self._make_pe(2000, d_model))

        # Mamba blocks
        self.mamba_blocks = nn.ModuleList([
            FastMambaBlock(d_model, d_state, d_conv) for _ in range(n_blocks)
        ])

        # LNN gates
        self.use_lnn = use_lnn
        if use_lnn:
            self.lnn_gates = nn.ModuleList([
                FastLNNGate(d_model, hidden=32) for _ in range(n_blocks)
            ])

        self.dropout = nn.Dropout(dropout)

        # Decoder: last-step → future sequence
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, pred_len),
        )

        self.apply(self._init)

    def _make_pe(self, max_len: int, d: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)  # (1, max_len, D)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, 0.5)
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, V, L)  raw variables
        Returns: (B, pred_len)
        """
        B, V, L = x.shape

        # Per-variable embedding
        x = x.unsqueeze(-1)        # (B, V, L, 1)
        x = self.var_emb(x)        # (B, V, L, D)
        x = x.transpose(1, 2).reshape(B, L, V * self.d_model)  # (B, L, V*D)
        x = self.input_fusion(x)   # (B, L, D)

        # Positional encoding
        x = x + self.pos_emb[:, :L]

        # Mamba blocks + optional LNN gates
        for i, mamba in enumerate(self.mamba_blocks):
            x = mamba(x)
            x = self.dropout(x)
            if self.use_lnn:
                gate = self.lnn_gates[i](x)
                x = x * gate

        # Decode from last step
        out = self.decoder(x[:, -1])  # (B, pred_len)
        return out

    def spectral_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Weighted frequency-domain L1 loss. Runs in fp32 for FFT compatibility."""
        pred_fft  = torch.fft.rfft(pred.float(), dim=-1, norm='ortho')
        targ_fft  = torch.fft.rfft(target.float(), dim=-1, norm='ortho')
        pred_mag  = torch.abs(pred_fft)
        targ_mag  = torch.abs(targ_fft)

        n_freq = pred_mag.shape[-1]
        w = torch.ones(n_freq, device=pred.device)
        if n_freq > 2:
            w[1:min(3, n_freq)] = 2.0  # boost daily/semi-daily
        w[0] = 0.3  # DC less important

        return (torch.abs(pred_mag - targ_mag) * w.unsqueeze(0)).mean() * 0.3

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> Dict:
        mse = F.mse_loss(pred, target)

        loss = mse
        if self.use_spectral:
            spec = self.spectral_loss(pred, target)
            loss = loss + spec

        with torch.no_grad():
            rmse = torch.sqrt(mse)

        result = {'loss': loss, 'mse': mse, 'rmse': rmse}
        if self.use_spectral:
            result['spectral'] = spec
        return result


# ═══════════════════════════════════════════════════════
# GRU Baseline (same param count)
# ═══════════════════════════════════════════════════════

class GRUBaseline(nn.Module):
    """GRU baseline — matched param count."""
    def __init__(self, n_vars: int = 11, d_model: int = 128, n_layers: int = 2,
                 pred_len: int = 24, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_vars, d_model)
        self.gru = nn.GRU(d_model, d_model, n_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, V, L = x.shape
        x = x.transpose(1, 2)  # (B, L, V)
        x = self.input_proj(x)  # (B, L, D)
        _, h = self.gru(x)
        out = self.decoder(h[-1])  # (B, pred_len)
        return out

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> Dict:
        mse = F.mse_loss(pred, target)
        with torch.no_grad(): rmse = torch.sqrt(mse)
        return {'loss': mse, 'mse': mse, 'rmse': rmse}
