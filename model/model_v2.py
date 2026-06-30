"""
LMT v2 — Multi-Scale LNN-Mamba-Transformer with ΔP prediction & weighted loss.

4 simultaneous improvements over v1:
  1. Multi-scale decomposition → separate Mamba channels per frequency band
  2. Extended seq_len + enhanced time encoding → eliminate phase lag
  3. ΔP prediction + bin-specific heads + high-power weighted loss
  4. Full config: d_model=128, 3 Mamba blocks, train_ratio=0.7
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from einops import rearrange


# ═══════════════════════════════════════════════════════════
# 1. Multi-Scale Causal Convolution Decomposition
# ═══════════════════════════════════════════════════════════

class MultiScaleDecomp(nn.Module):
    """Extract trend/daily/hourly/sub-hourly features via parallel causal convs."""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # 4 scales: sub-hourly(3), hourly(13), daily(49), synoptic(97)
        self.conv_rapid  = nn.Conv1d(d_model, d_model//4, kernel_size=3,  padding=2,  groups=1)
        self.conv_hourly = nn.Conv1d(d_model, d_model//4, kernel_size=13, padding=12, groups=1)
        self.conv_daily  = nn.Conv1d(d_model, d_model//4, kernel_size=49, padding=48, groups=1)
        self.conv_weekly = nn.Conv1d(d_model, d_model//4, kernel_size=97, padding=96, groups=1)
        self.fusion = nn.Linear(d_model, d_model)
        self.norm   = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) → (B, L, D) multi-scale fused"""
        B, L, D = x.shape
        xt = rearrange(x, 'b l d -> b d l')  # for Conv1d

        r = F.gelu(self.conv_rapid(xt))[:, :, :L]
        h = F.gelu(self.conv_hourly(xt))[:, :, :L]
        d = F.gelu(self.conv_daily(xt))[:, :, :L]
        w = F.gelu(self.conv_weekly(xt))[:, :, :L]

        multi = rearrange(torch.cat([r, h, d, w], dim=1), 'b d l -> b l d')
        return self.norm(x + self.fusion(multi))


# ═══════════════════════════════════════════════════════════
# 2. Mamba-2 SSM Block (kept from v1, optimized)
# ═══════════════════════════════════════════════════════════

class Mamba2Block(nn.Module):
    """Selective SSM block."""
    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = d_model * expand
        self.d_state = d_state
        self.in_proj  = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv, groups=d_inner, padding=d_conv-1)
        A = torch.arange(1, d_state+1, dtype=torch.float32).unsqueeze(0) * 0.1
        self.A_log    = nn.Parameter(torch.log(A))
        self.D        = nn.Parameter(torch.ones(d_inner))
        self.x_proj   = nn.Linear(d_inner, d_state*2+1, bias=False)
        self.dt_proj  = nn.Linear(d_state, d_inner, bias=True)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)

    def _scan(self, u, delta, A, B_s, C_s, D_skip):
        B, L, D = u.shape; N = A.shape[1]
        delta_exp = delta.unsqueeze(-1)
        A_bar = torch.exp(delta_exp * A.unsqueeze(0).unsqueeze(1))  # (B,L,D,N)
        h = torch.zeros(B, D, N, device=u.device, dtype=u.dtype)
        out = torch.empty(B, L, D, device=u.device, dtype=u.dtype)
        for t in range(L):
            h = A_bar[:,t] * h + delta_exp[:,t] * B_s[:,t].unsqueeze(1) * u[:,t].unsqueeze(-1)
            out[:,t] = (h * C_s[:,t].unsqueeze(1)).sum(dim=-1) + D_skip.squeeze(0) * u[:,t]
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x; B, L, D = x.shape
        xz = self.in_proj(x); x_ssm, z = xz.chunk(2, dim=-1)
        x_ssm_t = rearrange(x_ssm, 'b l d -> b d l')
        x_ssm_t = self.conv1d(x_ssm_t)[:,:,:L]
        x_ssm = F.silu(rearrange(x_ssm_t, 'b d l -> b l d'))
        proj = self.x_proj(x_ssm)
        dt = F.softplus(proj[:,:,:1]); B_s = proj[:,:,1:1+self.d_state]; C_s = proj[:,:,1+self.d_state:]
        dt = F.softplus(self.dt_proj(dt.repeat(1,1,self.d_state))) + 1e-4
        A = -torch.exp(self.A_log); A_exp = A.repeat(D*2, 1)
        y = self._scan(x_ssm, dt, A_exp, B_s, C_s, self.D.unsqueeze(0).unsqueeze(0))
        return self.norm(self.out_proj(y * F.silu(z)) + residual)


# ═══════════════════════════════════════════════════════════
# 3. Enhanced Time Encoding
# ═══════════════════════════════════════════════════════════

class EnhancedTimeEncoding(nn.Module):
    """Sinusoidal PE + learnable hour/dow/month embeddings."""
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.)/d_model))
        pe[:,0::2] = torch.sin(pos * div); pe[:,1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, D)
        # Rich time features
        self.hour_emb  = nn.Embedding(24, d_model)
        self.dow_emb   = nn.Embedding(7, d_model)
        self.month_emb = nn.Embedding(12, d_model)
        self.season_emb = nn.Embedding(4, d_model)  # spring/summer/fall/winter
        self.time_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor, ts: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B,L,D), ts: (B,L,5) with [hour,dow,month,season,relative_pos]"""
        L = x.shape[-2]
        out = x + self.pe[:, :L]
        if ts is not None:
            t_feat = (
                self.hour_emb(ts[:,:,0].long()) +
                self.dow_emb(ts[:,:,1].long()) +
                self.month_emb(ts[:,:,2].long()) +
                self.season_emb(ts[:,:,3].long())
            )
            out = out + self.time_scale * t_feat
        return out


# ═══════════════════════════════════════════════════════════
# 4. Fast LNN Gate (GRU-based, vectorized)
# ═══════════════════════════════════════════════════════════

class FastLNNGate(nn.Module):
    """GRU backbone + LNN time-constant modulation for dynamic gating."""
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.tau_proj = nn.Linear(input_dim, hidden_dim)
        self.out_gate = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,L,D) → gate: (B,L,D)"""
        h_gru, _ = self.gru(x)
        tau = 0.3 + 2.7 * torch.sigmoid(self.tau_proj(x))
        return torch.sigmoid(self.out_gate(torch.tanh(h_gru) / tau))


# ═══════════════════════════════════════════════════════════
# 5. Cross-Variable Attention
# ═══════════════════════════════════════════════════════════

class CrossVarAttention(nn.Module):
    """Variables attend to each other at each timestep."""
    def __init__(self, d_model: int, n_heads: int = 8, max_vars: int = 12):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads; self.head_dim = d_model // n_heads; self.scale = math.sqrt(self.head_dim)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out  = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.var_pos = nn.Parameter(torch.randn(1, max_vars, 1, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,V,L,D) → (B,V,L,D)"""
        residual = x; B, V, L, D = x.shape
        x = x + self.var_pos[:,:V]
        xf = rearrange(x, 'b v l d -> (b l) v d')
        q, k, v = self.qkv(xf).chunk(3, dim=-1)
        q = rearrange(q, 'bl v (h d) -> bl h v d', h=self.n_heads)
        k = rearrange(k, 'bl v (h d) -> bl h v d', h=self.n_heads)
        v = rearrange(v, 'bl v (h d) -> bl h v d', h=self.n_heads)
        attn = F.softmax(torch.matmul(q, k.transpose(-2,-1)) / self.scale, dim=-1)
        out = rearrange(torch.matmul(attn, v), 'bl h v d -> bl v (h d)')
        out = rearrange(self.out(out), '(b l) v d -> b v l d', b=B, l=L)
        return self.norm(out + residual)


# ═══════════════════════════════════════════════════════════
# 6. Bin-Specific Prediction Heads
# ═══════════════════════════════════════════════════════════

class BinPredictionHead(nn.Module):
    """3 regime-specific heads + gating network for low/med/high power."""
    def __init__(self, d_model: int, pred_len: int, dropout: float = 0.1):
        super().__init__()
        self.pred_len = pred_len

        # Gate: decides which regime based on context
        self.gate = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(),
            nn.Linear(32, 3),  # [low, med, high]
        )

        # 3 regime-specific decoders
        def make_head():
            return nn.Sequential(
                nn.Linear(d_model, d_model*2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model*2, d_model), nn.GELU(),
                nn.Linear(d_model, pred_len),
            )
        self.head_low   = make_head()
        self.head_med   = make_head()
        self.head_high  = make_head()

        # ΔP head (predict change from last observed)
        self.delta_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, pred_len),
        )

    def forward(self, x: torch.Tensor, last_obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: (B, D) — encoded context vector
        last_obs: (B,) — last observed power value
        """
        gate_logits = self.gate(x)  # (B, 3)
        gate_probs  = F.softmax(gate_logits, dim=-1)

        # Each head predicts absolute power
        p_low   = self.head_low(x)
        p_med   = self.head_med(x)
        p_high  = self.head_high(x)

        # ΔP prediction
        delta = self.delta_head(x)  # (B, pred_len)
        last_obs_exp = last_obs.unsqueeze(-1)  # (B, 1)

        # Weighted ensemble prediction
        pred_abs = (gate_probs[:,0:1] * p_low +
                    gate_probs[:,1:2] * p_med +
                    gate_probs[:,2:3] * p_high)  # (B, pred_len)

        # ΔP reconstruction: P(t+h) = P(t) + cumsum(ΔP)
        delta_cumsum = torch.cumsum(delta, dim=-1)
        pred_delta = last_obs_exp + delta_cumsum  # (B, pred_len)

        # Final: blend absolute and delta predictions (learned in loss)
        # During inference, use delta for short horizons, absolute for long
        pred_final = (pred_abs + pred_delta) / 2

        return {
            'pred':        pred_final,
            'pred_abs':    pred_abs,
            'pred_delta':  pred_delta,
            'delta':       delta,
            'gate_probs':  gate_probs,
            'gate_logits': gate_logits,
        }


# ═══════════════════════════════════════════════════════════
# 7. LMT v2 Full Model
# ═══════════════════════════════════════════════════════════

class LNNMambaTransformerV2(nn.Module):
    def __init__(
        self,
        n_vars: int = 11,
        d_model: int = 128,
        n_mamba_blocks: int = 3,
        n_transformer_layers: int = 2,
        d_state: int = 64,
        d_conv: int = 4,
        n_heads: int = 8,
        pred_len: int = 96,
        lnn_hidden: int = 64,
        dropout: float = 0.1,
        rated_capacity: float = 200.0,  # MW, for normalization
    ):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.pred_len = pred_len
        self.rated_capacity = rated_capacity

        # Input projection
        self.input_proj = nn.Linear(1, d_model)

        # Enhanced time encoding
        self.time_enc = EnhancedTimeEncoding(d_model)

        # Multi-scale decomposition (before Mamba)
        self.multiscale = MultiScaleDecomp(d_model)

        # Per-variable Mamba2 blocks with LNN gating
        self.mamba_blocks = nn.ModuleList([
            Mamba2Block(d_model, d_state=d_state, d_conv=d_conv)
            for _ in range(n_mamba_blocks)
        ])
        self.lnn_gates = nn.ModuleList([
            FastLNNGate(d_model, hidden_dim=lnn_hidden)
            for _ in range(n_mamba_blocks)
        ])

        # Cross-variable transformer
        self.cross_attn = nn.ModuleList([
            CrossVarAttention(d_model, n_heads=n_heads, max_vars=n_vars)
            for _ in range(n_transformer_layers)
        ])

        # Fusion
        self.var_to_temporal = nn.Linear(d_model * 2, d_model)
        self.temporal_refine = Mamba2Block(d_model, d_state=d_state, d_conv=d_conv)
        self.fusion_lnn = FastLNNGate(d_model * 2, hidden_dim=lnn_hidden)
        self.fusion_proj = nn.Linear(d_model * 2, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)

        # Bin-specific prediction head
        self.pred_head = BinPredictionHead(d_model, pred_len, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, ts: Optional[torch.Tensor] = None,
                last_power: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        x: (B, V, L)   input variables
        ts: (B, L, 5)  [hour, dow, month, season, rel_pos]
        last_power: (B,)  last observed power for ΔP reconstruction
        """
        B, V, L = x.shape
        D = self.d_model

        # 1. Input projection
        x = self.input_proj(x.unsqueeze(-1))  # (B, V, L, D)

        # 2. Enhanced time encoding (per variable)
        x = rearrange(x, 'b v l d -> (b v) l d')
        if ts is not None:
            ts_exp = ts.repeat_interleave(V, dim=0)  # (B*V, L, 5)
        else:
            ts_exp = None
        x = self.time_enc(x, ts_exp)
        x = rearrange(x, '(b v) l d -> b v l d', v=V)

        # 3. Multi-scale decomposition (applied to each variable's temporal dim)
        x = rearrange(x, 'b v l d -> (b v) l d')
        x = self.multiscale(x)
        x = rearrange(x, '(b v) l d -> b v l d', v=V)

        # 4. Per-variable Mamba2 + LNN gating
        for mamba, lnn in zip(self.mamba_blocks, self.lnn_gates):
            x = rearrange(x, 'b v l d -> (b v) l d')
            x = mamba(x)
            x = self.dropout(x)
            x = rearrange(x, '(b v) l d -> b v l d', v=V)
            # LNN gate (operates on variable-pooled representation)
            global_ctx = x.mean(dim=1)  # (B, L, D)
            gate = lnn(global_ctx)       # (B, L, D)
            x = x * gate.unsqueeze(1)

        mamba_out = x  # (B, V, L, D)

        # 5. Cross-variable transformer
        for attn in self.cross_attn:
            x = attn(x)
            x = self.dropout(x)

        cross_out = x

        # 6. Temporal refinement + LNN fusion
        cross_pooled = cross_out.mean(dim=1)    # (B, L, D)
        mamba_pooled = mamba_out.mean(dim=1)    # (B, L, D)
        hybrid = self.var_to_temporal(torch.cat([cross_pooled, mamba_pooled], dim=-1))
        hybrid = self.temporal_refine(hybrid)

        # LNN adaptive fusion
        fusion_input = torch.cat([mamba_pooled, hybrid], dim=-1)  # (B, L, 2D)
        alpha = torch.sigmoid(self.fusion_lnn(fusion_input))       # (B, L, 2D)
        alpha_m, alpha_h = alpha.chunk(2, dim=-1)
        fused = alpha_m * mamba_pooled + alpha_h * hybrid
        fused = self.fusion_proj(torch.cat([fused, mamba_pooled * hybrid], dim=-1))
        fused = self.fusion_norm(fused)

        # 7. Prediction
        context = fused[:, -1]  # (B, D) — last timestep encoding
        last_obs = last_power if last_power is not None else torch.zeros(B, device=x.device)
        preds = self.pred_head(context, last_obs)

        return preds

    def compute_loss(self, preds: Dict, target: torch.Tensor,
                     last_power: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Weighted multi-component loss:
          1. Main MSE + high-power weighting
          2. ΔP auxiliary loss
          3. Regime classification loss
        """
        pred = preds['pred']          # (B, pred_len)
        pred_abs = preds['pred_abs']
        pred_delta = preds['pred_delta']
        delta = preds['delta']
        gate_probs = preds['gate_probs']  # (B, 3)

        B, P = target.shape

        # ── Sample weights: higher power → higher weight ──
        mean_power = target.mean(dim=-1)  # (B,)
        weight = 1.0 + 2.0 * (mean_power / self.rated_capacity)  # ∈ [1, 3]

        # ── Main MSE with weighting ──
        se = (pred - target) ** 2  # (B, pred_len)
        loss_main = (se.mean(dim=-1) * weight).mean()

        # ── Per-regime weights ──
        is_low   = (mean_power < 0.3 * self.rated_capacity).float()  # <60MW
        is_high  = (mean_power > 0.7 * self.rated_capacity).float()  # >140MW
        is_med   = 1.0 - is_low - is_high

        se_low  = (se.mean(dim=-1) * is_low * 1.0).sum() / (is_low.sum() + 1)
        se_med  = (se.mean(dim=-1) * is_med * 2.0).sum() / (is_med.sum() + 1)
        se_high = (se.mean(dim=-1) * is_high * 3.0).sum() / (is_high.sum() + 1)

        loss_regime_weighted = se_low + se_med + se_high

        # ── ΔP auxiliary loss ──
        if last_power is not None:
            delta_target = target - last_power.unsqueeze(-1)
            loss_delta = F.mse_loss(delta, delta_target) * 0.5
        else:
            loss_delta = F.mse_loss(pred_delta, target) * 0.3

        # ── Regime classification loss (self-supervised) ──
        regime_target = torch.zeros(B, dtype=torch.long, device=pred.device)
        regime_target = torch.where(mean_power > 0.3 * self.rated_capacity,
                                    torch.ones_like(regime_target), regime_target)
        regime_target = torch.where(mean_power > 0.7 * self.rated_capacity,
                                    torch.full_like(regime_target, 2), regime_target)
        loss_regime = F.cross_entropy(preds['gate_logits'], regime_target) * 0.1

        # ── Absolute prediction auxiliary loss ──
        loss_abs = F.mse_loss(pred_abs, target) * 0.3

        # ── Total ──
        total_loss = (loss_main + loss_regime_weighted + loss_delta +
                      loss_regime + loss_abs)

        with torch.no_grad():
            rmse = torch.sqrt(F.mse_loss(pred, target))

        return {
            'loss': total_loss,
            'loss_main': loss_main,
            'loss_regime_weighted': loss_regime_weighted,
            'loss_delta': loss_delta,
            'loss_regime_cls': loss_regime,
            'loss_abs': loss_abs,
            'rmse': rmse,
            'mean_weight': weight.mean(),
        }
