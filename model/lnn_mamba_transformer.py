"""
LNN-Gated Selective SSM-Transformer (LMT) — Full Architecture for Wind Power Forecasting

Architecture:
┌─────────────────────────────────────────────────────────┐
│  1. RevIN Normalization                                 │
│  2. VMD/EWT Multi-Scale Decomposition (optional)        │
│  3. Per-Variable Mamba2 Temporal Encoder                │
│  4. LNN Dynamic State Regulator                         │
│  5. Cross-Variable Transformer Fusion                   │
│  6. LNN Adaptive Gating + Fusion                        │
│  7. Prediction Head                                     │
│  8. RevIN De-Normalization                              │
└─────────────────────────────────────────────────────────┘

Key Innovation:
  LNN (Liquid Neural Network) dynamically regulates:
    (a) SSM state transitions between wind regimes
    (b) Mamba ↔ Transformer fusion ratio per timestep
    (c) Multi-scale feature importance

This makes the model inherently adaptive to non-stationary wind dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from einops import rearrange, repeat

from .layers import (
    Mamba2SSMBlock,
    LNNGate,
    CrossVarAttention,
    AdaptiveFusion,
    TimeAwarePositionalEncoding,
    RevIN,
)


class LNNMambaTransformer(nn.Module):
    """
    LNN-Regulated Mamba-Transformer for Wind Power Forecasting.

    Args:
        n_vars:          number of input variables (features)
        d_model:         hidden dimension
        n_mamba_blocks:  number of Mamba2 blocks per variable
        n_transformer_layers: number of cross-variable attention layers
        d_state:         SSM state dimension
        n_heads:         attention heads
        pred_len:        prediction horizon (number of future timesteps)
        lnn_hidden:      LNN hidden dimension
        use_vmd:         (reserved) multi-scale decomposition flag
        dropout:         dropout rate
    """

    def __init__(
        self,
        n_vars: int = 10,
        d_model: int = 128,
        n_mamba_blocks: int = 3,
        n_transformer_layers: int = 2,
        d_state: int = 64,
        d_conv: int = 4,
        n_heads: int = 8,
        pred_len: int = 96,
        lnn_hidden: int = 64,
        use_vmd: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.pred_len = pred_len

        # RevIN
        self.revin = RevIN(d_model, affine=True)

        # Input projection: raw features → d_model
        self.input_proj = nn.Linear(1, d_model)

        # Positional encoding
        self.pos_enc = TimeAwarePositionalEncoding(d_model)

        # ==========================================
        # Per-Variable Mamba2 Temporal Encoder
        # ==========================================
        self.mamba_blocks = nn.ModuleList([
            Mamba2SSMBlock(d_model, d_state=d_state, d_conv=d_conv)
            for _ in range(n_mamba_blocks)
        ])

        # ==========================================
        # LNN State Regulator (between Mamba blocks)
        # ==========================================
        self.lnn_regulators = nn.ModuleList([
            LNNGate(d_model, hidden_dim=lnn_hidden)
            for _ in range(n_mamba_blocks)
        ])

        # ==========================================
        # Cross-Variable Transformer
        # ==========================================
        self.cross_var_attn = nn.ModuleList([
            CrossVarAttention(d_model, n_heads=n_heads, n_vars=n_vars)
            for _ in range(n_transformer_layers)
        ])

        # ==========================================
        # Temporal ↔ Cross-Variable Bridge
        # ==========================================
        # After per-var Mamba: (B, V, L, D)
        # Mean-pool over variables → (B, L, D)
        self.var_to_temporal = nn.Linear(d_model * 2, d_model)

        # After cross-var attention: (B, V, L, D)
        # Pool variables → (B, L, D), then temporal refinement
        self.temporal_refine = Mamba2SSMBlock(d_model, d_state=d_state, d_conv=d_conv)

        # ==========================================
        # LNN Adaptive Fusion
        # ==========================================
        self.adaptive_fusion = AdaptiveFusion(d_model, lnn_hidden=lnn_hidden)

        # ==========================================
        # Prediction Head
        # ==========================================
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )

        # Auxiliary: wind regime classifier (helps LNN learn transitions)
        self.regime_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 3),  # low / medium / high wind
        )

        self.dropout = nn.Dropout(dropout)

        # Initialize
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        x:  (B, V, L_in)  raw input
        timestamps:  (B, L_in, 2)  [hour, month] optional

        Returns dict with:
            'pred':       (B, pred_len)  power predictions
            'regime':     (B, 3)         wind regime logits
            'aux':        list of auxiliary fusion stats
        """
        B, V, L_in = x.shape
        D = self.d_model

        # 1. Input projection: (B, V, L) → (B, V, L, D)
        x = x.unsqueeze(-1)  # (B, V, L, 1)
        x = self.input_proj(x)  # (B, V, L, D)

        # 2. Positional encoding (per variable)
        x = rearrange(x, 'b v l d -> (b v) l d')
        if timestamps is not None:
            ts_expanded = repeat(timestamps, 'b l t -> (b v) l t', v=V)
        else:
            ts_expanded = None
        x = self.pos_enc(x, ts_expanded)
        x = rearrange(x, '(b v) l d -> b v l d', v=V)

        # 3. RevIN normalization (over time dim, per variable)
        x = rearrange(x, 'b v l d -> (b v) l d')
        x, mean, stdev = self.revin(x, mode='norm')
        x = rearrange(x, '(b v) l d -> b v l d', v=V)

        # ==========================================
        # 4. Per-Variable Mamba2 with LNN Regulation
        # ==========================================
        aux_list = []
        for mamba, lnn_gate in zip(self.mamba_blocks, self.lnn_regulators):
            # Apply Mamba per variable
            x = rearrange(x, 'b v l d -> (b v) l d')
            x = mamba(x)
            x = self.dropout(x)
            x = rearrange(x, '(b v) l d -> b v l d', v=V)

            # LNN dynamic gate: regulates per-variable, per-timestep feature flow
            # Pool across variables to get global context for the LNN
            global_ctx = x.mean(dim=1)  # (B, L, D)
            gate, modulation = lnn_gate(global_ctx)  # (B, L, D), (B, L, H)

            # Apply gate to each variable
            x = x * gate.unsqueeze(1)  # broadcast over variables

            aux_list.append({'gate_mean': gate.mean().item()})

        # ==========================================
        # 5. Cross-Variable Transformer
        # ==========================================
        mamba_out = x  # (B, V, L, D) — keep for fusion

        for attn_layer in self.cross_var_attn:
            x = attn_layer(x)
            x = self.dropout(x)

        cross_out = x  # (B, V, L, D)

        # ==========================================
        # 6. Temporal Refinement
        # ==========================================
        # Pool cross-var features
        x_pooled = cross_out.mean(dim=1)  # (B, L, D)

        # Concatenate with mamba temporal features
        mamba_pooled = mamba_out.mean(dim=1)  # (B, L, D)
        hybrid = torch.cat([x_pooled, mamba_pooled], dim=-1)  # (B, L, 2D)
        hybrid = self.var_to_temporal(hybrid)
        hybrid = self.temporal_refine(hybrid)  # (B, L, D)

        # ==========================================
        # 7. LNN Adaptive Fusion
        # ==========================================
        fused, fusion_aux = self.adaptive_fusion(mamba_pooled, hybrid)
        aux_list.append(fusion_aux)

        # ==========================================
        # 8. Prediction Head
        # ==========================================
        # Take last timestep's representation for decoding
        decoder_input = fused[:, -1]  # (B, D)
        pred = self.pred_head(decoder_input)  # (B, pred_len)

        # Auxiliary: wind regime classification
        regime = self.regime_head(decoder_input)

        # 9. RevIN de-normalize
        # pred = self.revin(pred.unsqueeze(-1), mode='denorm').squeeze(-1)

        result = {
            'pred': pred,
            'regime': regime,
        }
        if return_aux:
            result['aux'] = aux_list

        return result

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor, regime: torch.Tensor,
        regime_target: Optional[torch.Tensor] = None,
        lambda_regime: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-task loss: MSE (main) + regime classification (auxiliary).
        """
        # Main prediction loss
        loss_mse = F.mse_loss(pred, target)

        # Regime loss (self-supervised: cluster power levels)
        if regime_target is None:
            # Auto-label: low (<30% capacity), medium (30-70%), high (>70%)
            mean_power = target.mean(dim=-1)  # (B,)
            regime_target = torch.zeros_like(mean_power, dtype=torch.long)  # (B,)
            regime_target = torch.where(mean_power > 30, torch.ones_like(regime_target), regime_target)
            regime_target = torch.where(mean_power > 70,
                                        torch.full_like(regime_target, 2), regime_target)

        loss_regime = F.cross_entropy(regime, regime_target)

        loss = loss_mse + lambda_regime * loss_regime

        return {
            'loss': loss,
            'mse': loss_mse,
            'rmse': torch.sqrt(loss_mse),
            'regime_loss': loss_regime,
        }
