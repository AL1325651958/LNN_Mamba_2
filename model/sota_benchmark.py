"""
SOTA Benchmark: 12 Models for GEFCom2014 Probabilistic Wind Power Forecasting.
================================================================================
Compares the latest time series forecasting models (2023-2025) on the
GEFCom2014 wind power dataset with NWP weather features.

Models:
  1. Persistence      — naive baseline
  2. GRU              — classical RNN
  3. DLinear          — AAAI 2023, decomposition + linear
  4. PatchTST         — ICLR 2023, patch-based transformer
  5. iTransformer     — ICLR 2024, inverted transformer
  6. TimesNet         — ICLR 2023, FFT period + 2D conv
  7. TSMixer          — KDD 2023, MLP-Mixer for time series
  8. ModernTCN        — ICLR 2024, modern temporal CNN
  9. TiDE             — Google 2024, dense encoder-decoder
  10. TimeMixer       — ICLR 2024, multi-scale mixing
  11. Crossformer     — ICLR 2023, cross-dimension attention
  12. LNN-Gated Selective SSM       — SSM + liquid time-constant gates (ours)

Task: 168h ECMWF NWP → 24h wind power, 99 quantiles, pinball loss
Data:  GEFCom2014 Zones 1-10, 85/7/8% train/val/test split
"""
import sys, os, zipfile, time, argparse, io, math, json, warnings
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

DATA_DIR = 'data/gefcom2014'
QUANTILES = np.linspace(0.01, 0.99, 99)
NQ = len(QUANTILES)

# ═══════════════════════════════════════════════════════════════
# Weather Features
# ═══════════════════════════════════════════════════════════════

def make_weather_features(df):
    """Derive rich meteorological features from raw U/V."""
    df['WS10']  = np.sqrt(df['U10']**2 + df['V10']**2)
    df['WS100'] = np.sqrt(df['U100']**2 + df['V100']**2)
    df['WD10']  = np.arctan2(df['U10'], df['V10'])
    df['WD100'] = np.arctan2(df['U100'], df['V100'])
    df['SHEAR'] = df['WS100'] / (df['WS10'] + 0.1)
    df['VEER']  = np.sin(df['WD100'] - df['WD10'])
    return df

FEAT_COLS = ['U10', 'V10', 'U100', 'V100', 'WS10', 'WS100', 'WD10', 'WD100', 'SHEAR', 'VEER']


# ═══════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════

class NWPDS(Dataset):
    """NWP weather + power dataset with sliding windows."""
    def __init__(self, data, seq, pred, stride):
        self.data = torch.FloatTensor(data)
        self.seq = seq; self.pred = pred; self.s = stride
        self.n = max(0, (len(data) - seq - pred) // stride + 1)

    def __len__(self): return self.n

    def __getitem__(self, i):
        st = i * self.s
        return (self.data[st:st+self.seq].T,
                self.data[st+self.seq:st+self.seq+self.pred, -1])


def load_zone_nwp(zid, seq=168, pred=24, batch=64, stride=4):
    """Load one GEFCom2014 zone with NWP weather features."""
    tz = zipfile.ZipFile(f'{DATA_DIR}/Task15_W_Zone1_10.zip')
    df = pd.read_csv(tz.open(f'Task15_W_Zone1_10/Task15_W_Zone{zid}.csv'))

    ts = df['TIMESTAMP'].astype(str).str.strip()
    df['dt'] = pd.to_datetime(ts.str[:8], format='%Y%m%d') \
               + pd.to_timedelta(ts.str.extract(r'(\d+):')[0].astype(int), unit='h')
    df = df.sort_values('dt').reset_index(drop=True)

    df['TARGETVAR'] = df['TARGETVAR'].interpolate(limit_direction='both')
    for c in ['U10', 'V10', 'U100', 'V100']:
        df[c] = df[c].interpolate(limit_direction='both')

    df = make_weather_features(df)

    h = df['dt'].dt.hour.values.astype(np.float32)
    m = df['dt'].dt.month.values.astype(np.float32)
    df['HOUR_SIN'] = np.sin(2 * np.pi * h / 24)
    df['HOUR_COS'] = np.cos(2 * np.pi * h / 24)
    df['MONTH_SIN'] = np.sin(2 * np.pi * m / 12)
    df['MONTH_COS'] = np.cos(2 * np.pi * m / 12)

    all_feats = FEAT_COLS + ['HOUR_SIN', 'HOUR_COS', 'MONTH_SIN', 'MONTH_COS']

    scaler_x = StandardScaler()
    feats = scaler_x.fit_transform(df[all_feats].values.astype(np.float32))
    scaler_y = StandardScaler()
    target = scaler_y.fit_transform(df[['TARGETVAR']].values.astype(np.float32)).ravel()

    data = np.concatenate([feats, target.reshape(-1, 1)], axis=1)
    n_vars = data.shape[1]

    T = len(data)
    te = int(T * 0.85)
    ve = int(T * 0.92)

    info = f'Zone {zid}: {len(df)} rows | train={te} val={ve-te} test={T-ve} | vars={n_vars}'

    ds_train = NWPDS(data[:te], seq, pred, stride)
    ds_val   = NWPDS(data[te:ve], seq, pred, stride)
    ds_test  = NWPDS(data[ve:], seq, pred, stride)

    dl_train = DataLoader(ds_train, batch, shuffle=True,  num_workers=0, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch, shuffle=False, num_workers=0, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch, shuffle=False, num_workers=0, pin_memory=True)

    return (dl_train, dl_val, dl_test), n_vars, scaler_y, info


# ═══════════════════════════════════════════════════════════════
# Shared Building Blocks
# ═══════════════════════════════════════════════════════════════

def pinball_loss(pred_q, target, q_tensor):
    """GEFCom2014 official metric: quantile (pinball) loss."""
    error = target.unsqueeze(-1) - pred_q
    return torch.maximum(q_tensor * error, (q_tensor - 1) * error).mean()


class QuantileDecoder(nn.Module):
    """Decode latent features → (pred_len, n_quantiles)."""
    def __init__(self, d, pred, nq=99):
        super().__init__()
        self.pred = pred; self.nq = nq
        self.net = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d * 2, d), nn.GELU(),
            nn.Linear(d, pred * nq),
        )

    def forward(self, x):
        return self.net(x).view(-1, self.pred, self.nq)


class SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ═══════════════════════════════════════════════════════════════
# MODEL 1: Persistence Baseline
# ═══════════════════════════════════════════════════════════════

class Persistence:
    """P(t+h) = P(t) for all horizons. Non-trainable baseline."""
    def __init__(self, pred=24, nq=99):
        self.pred = pred; self.nq = nq

    def __call__(self, x):
        B = x.shape[0]
        last_power = x[:, -1, -1]  # (B,) — last observed power
        pred = last_power.unsqueeze(1).unsqueeze(2).expand(-1, self.pred, self.nq)
        return pred

    def eval(self): return self
    def train(self, mode=True): return self
    def to(self, device): return self  # no-op
    def state_dict(self): return {}
    def load_state_dict(self, d): pass
    def parameters(self): return iter([])


# ═══════════════════════════════════════════════════════════════
# MODEL 2: GRU Baseline
# ═══════════════════════════════════════════════════════════════

class GRUModel(nn.Module):
    """2-layer GRU with quantile decoder. Strong classical baseline."""
    def __init__(self, n_vars, d=128, n_layers=2, pred=24, nq=99):
        super().__init__()
        self.proj = nn.Linear(n_vars, d)
        self.gru = nn.GRU(d, d, n_layers, batch_first=True, dropout=0.1)
        self.dec = QuantileDecoder(d, pred, nq)

    def forward(self, x):
        B, V, L = x.shape
        h = self.proj(x.transpose(1, 2))  # (B, L, d)
        _, hn = self.gru(h)
        return self.dec(hn[-1])


# ═══════════════════════════════════════════════════════════════
# MODEL 3: DLinear (AAAI 2023)
# ═══════════════════════════════════════════════════════════════

class MovingAvg(nn.Module):
    """Moving average for time series decomposition."""
    def __init__(self, kernel_size=25, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size, stride=stride, padding=kernel_size // 2)

    def forward(self, x):
        """x: (B, L) → (B, L)"""
        return self.avg(x.unsqueeze(1)).squeeze(1)


class DLinear(nn.Module):
    """
    DLinear: Are Transformers Effective for Time Series Forecasting? (AAAI 2023)
    Decomposition into trend + season, then simple linear projection per channel.
    """
    def __init__(self, n_vars, seq=168, pred=24, nq=99, kernel=25):
        super().__init__()
        self.n_vars = n_vars
        self.pred_len = pred
        self.nq = nq
        self.decomp = MovingAvg(kernel)
        # Per-variable linear layers (channel-independent)
        self.seasonal = nn.ModuleList([nn.Linear(seq, pred) for _ in range(n_vars)])
        self.trend = nn.ModuleList([nn.Linear(seq, pred) for _ in range(n_vars)])
        # Fuse variables → quantiles
        self.fuse = nn.Linear(n_vars, 1)
        self.qproj = nn.Linear(pred, pred * nq)

    def forward(self, x):
        B, V, L = x.shape
        # Decompose per variable
        seasonal_out = []
        trend_out = []
        for v in range(V):
            xv = x[:, v, :]  # (B, L)
            mv = self.decomp(xv)  # trend
            seasonal_out.append(self.seasonal[v](xv - mv).unsqueeze(1))  # (B, 1, P)
            trend_out.append(self.trend[v](mv).unsqueeze(1))
        seasonal = torch.cat(seasonal_out, dim=1)  # (B, V, P)
        trend = torch.cat(trend_out, dim=1)
        y = seasonal + trend  # (B, V, P)
        y = self.fuse(y.transpose(1, 2)).squeeze(-1)  # (B, P)
        return self.qproj(y).view(B, self.pred_len, self.nq)


# ═══════════════════════════════════════════════════════════════
# MODEL 4: PatchTST (ICLR 2023)
# ═══════════════════════════════════════════════════════════════

class PatchTST(nn.Module):
    """
    A Time Series is Worth 64 Words: PatchTST (ICLR 2023)
    Channel-independent patching + transformer encoder.
    """
    def __init__(self, n_vars, seq=168, pred=24, nq=99, d=128, n_heads=8,
                 n_layers=3, patch_len=12, stride=8):
        super().__init__()
        self.n_vars = n_vars
        self.patch_len = patch_len
        self.stride = stride
        self.pred_len = pred
        self.nq = nq
        self.n_patches = (seq - patch_len) // stride + 1  # ~20
        self.d = d

        self.patch_embed = nn.Linear(patch_len, d)
        self.pe = nn.Parameter(torch.randn(1, 1, self.n_patches, d) * 0.02)
        self.drop = nn.Dropout(0.1)

        encoder_layer = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=d*4,
                                                    dropout=0.1, activation='gelu',
                                                    batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        self.flatten = nn.Flatten(start_dim=-2)  # (B, n_patches*d)
        self.head = nn.Sequential(
            nn.Linear(self.n_patches * d, d * 2), nn.GELU(),
            nn.Linear(d * 2, pred * nq)
        )

    def forward(self, x):
        B, V, L = x.shape
        # Batch all variables together for efficiency (B*V, L) → patches
        x_flat = x.reshape(B * V, L)  # (B*V, L)
        patches = x_flat.unfold(1, self.patch_len, self.stride)  # (B*V, n_patches, patch_len)
        patches = self.patch_embed(patches)  # (B*V, n_patches, d)
        # PE: (1, 1, n_patches, d) → squeeze to (n_patches, d) for broadcast
        patches = self.drop(patches + self.pe.squeeze(0).squeeze(0))
        # Single batched transformer forward
        enc = self.encoder(patches)  # (B*V, n_patches, d)
        # Reshape back and mean pool across variables
        enc = enc.reshape(B, V, self.n_patches, self.d)
        enc = enc.mean(dim=1).reshape(B, -1)  # (B, n_patches*d)
        return self.head(enc).view(B, self.pred_len, self.nq)


# ═══════════════════════════════════════════════════════════════
# MODEL 5: iTransformer (ICLR 2024)
# ═══════════════════════════════════════════════════════════════

class iTransformer(nn.Module):
    """
    iTransformer: Inverted Transformers for Time Series (ICLR 2024)
    Variables as tokens, temporal projection per variable.
    """
    def __init__(self, n_vars, seq=168, pred=24, nq=99, d=128, n_heads=8, n_layers=3):
        super().__init__()
        self.n_vars = n_vars
        self.pred_len = pred
        self.nq = nq
        # Per-variable embedding: time series → d_model
        self.var_embed = nn.Linear(seq, d)
        self.drop = nn.Dropout(0.1)
        # Transformer over variable dimension
        encoder_layer = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=d*4,
                                                    dropout=0.1, activation='gelu',
                                                    batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        # Per-variable prediction
        self.var_pred = nn.Linear(d, pred)
        # Fuse predictions across variables
        self.fuse = nn.Sequential(
            nn.Linear(n_vars, n_vars // 2), nn.GELU(),
            nn.Linear(n_vars // 2, 1)
        )
        self.qproj = nn.Linear(pred, pred * nq)

    def forward(self, x):
        B, V, L = x.shape
        # Each variable → token: (B, V, L) → (B, V, d)
        tokens = self.var_embed(x)  # (B, V, d)
        tokens = self.drop(tokens)
        # Cross-variable attention
        tokens = self.encoder(tokens)  # (B, V, d)
        # Per-variable predict
        preds = self.var_pred(tokens)  # (B, V, pred)
        # Fuse variables
        preds = self.fuse(preds.transpose(1, 2)).squeeze(-1)  # (B, pred)
        return self.qproj(preds).view(B, self.pred_len, self.nq)


# ═══════════════════════════════════════════════════════════════
# MODEL 6: TimesNet (ICLR 2023)
# ═══════════════════════════════════════════════════════════════

class InceptionBlock(nn.Module):
    """2D Inception-style conv block for TimesNet."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c//4, 1)
        self.conv3 = nn.Sequential(nn.Conv2d(in_c, out_c//4, 1), nn.Conv2d(out_c//4, out_c//4, 3, padding=1))
        self.conv5 = nn.Sequential(nn.Conv2d(in_c, out_c//4, 1), nn.Conv2d(out_c//4, out_c//4, 5, padding=2))
        self.maxp = nn.Sequential(nn.MaxPool2d(3, 1, 1), nn.Conv2d(in_c, out_c//4, 1))
        self.act = nn.GELU()

    def forward(self, x):
        c1 = self.act(self.conv1(x))
        c3 = self.act(self.conv3(x))
        c5 = self.act(self.conv5(x))
        cp = self.act(self.maxp(x))
        return torch.cat([c1, c3, c5, cp], dim=1)


class TimesNet(nn.Module):
    """
    TimesNet: Temporal 2D-Variation Modeling (ICLR 2023)
    FFT finds dominant periods → reshape 1D→2D → 2D conv → aggregate.
    """
    def __init__(self, n_vars, seq=168, pred=24, nq=99, d=64, top_k=5, n_blocks=2):
        super().__init__()
        self.n_vars = n_vars
        self.seq = seq
        self.top_k = top_k
        self.embed = nn.Linear(n_vars, d)
        self.pe = SinusoidalPE(d, 2000)
        self.blocks = nn.ModuleList([InceptionBlock(d, d) for _ in range(n_blocks)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_blocks)])
        self.drop = nn.Dropout(0.1)
        self.dec = QuantileDecoder(d, pred, nq)

    def _fft_periods(self, x):
        """Find top-k periods via FFT amplitude."""
        # x: (B, d, L)
        device = x.device
        xf = torch.fft.rfft(x, dim=-1)
        amp = xf.abs().mean(dim=(0, 1))
        freq = torch.fft.rfftfreq(self.seq, device=device)
        # Exclude DC (freq=0)
        amp[0] = 0
        _, top_idx = torch.topk(amp, self.top_k)
        periods = (1.0 / (freq[top_idx] + 1e-8)).long()
        periods = torch.clamp(periods, 2, self.seq // 2)
        return periods

    def forward(self, x):
        B, V, L = x.shape
        x = self.embed(x.transpose(1, 2))  # (B, L, d)
        x = self.pe(x).transpose(1, 2)  # (B, d, L)

        periods = self._fft_periods(x)

        for blk, norm in zip(self.blocks, self.norms):
            res = x
            period_outs = []
            for p in periods:
                p = int(p.item())
                if L % p != 0:
                    pad_len = p - (L % p)
                    xp = F.pad(x, (0, pad_len))
                else:
                    xp = x
                # 1D → 2D: (B, d, P, L//P)
                x2d = xp.reshape(B, self.embed.out_features, p, -1)
                x2d = blk(x2d)  # (B, d, P, L//P)
                x1d = x2d.reshape(B, -1, xp.shape[-1])[:, :, :L]
                period_outs.append(x1d)
            # Softmax-weighted sum across periods
            weights = F.softmax(torch.stack([o.mean() for o in period_outs]), dim=0)
            agg = sum(w * o for w, o in zip(weights, period_outs))
            x = self.drop(agg + res)
            x = norm(x.transpose(1, 2)).transpose(1, 2)  # (B, d, L)

        # Take mean over time, decode
        return self.dec(x.mean(dim=-1))  # (B, d) → (B, pred, nq)


# ═══════════════════════════════════════════════════════════════
# MODEL 7: TSMixer (KDD 2023)
# ═══════════════════════════════════════════════════════════════

class TSMixerBlock(nn.Module):
    """Time-mixing + Feature-mixing MLP block."""
    def __init__(self, seq, d_feat, drop=0.1):
        super().__init__()
        # Time mixing: MLP along time dim
        self.time_ln = nn.LayerNorm(d_feat)
        self.time_mlp = nn.Sequential(
            nn.Linear(seq, seq * 2), nn.GELU(), nn.Dropout(drop),
            nn.Linear(seq * 2, seq)
        )
        # Feature mixing: MLP along feature dim
        self.feat_ln = nn.LayerNorm(d_feat)
        self.feat_mlp = nn.Sequential(
            nn.Linear(d_feat, d_feat * 2), nn.GELU(), nn.Dropout(drop),
            nn.Linear(d_feat * 2, d_feat)
        )

    def forward(self, x):
        # x: (B, L, d)
        # Time mixing
        res = x
        x = self.time_ln(x)
        x = self.time_mlp(x.transpose(1, 2)).transpose(1, 2) + res  # (B, L, d)
        # Feature mixing
        res = x
        x = self.feat_ln(x)
        x = self.feat_mlp(x) + res  # (B, L, d)
        return x


class TSMixer(nn.Module):
    """
    TSMixer: MLP-Mixer for Time Series (KDD 2023)
    Alternating time-mixing and feature-mixing MLPs.
    """
    def __init__(self, n_vars, seq=168, pred=24, nq=99, d=128, n_blocks=3):
        super().__init__()
        self.embed = nn.Linear(n_vars, d)
        self.pe = SinusoidalPE(d, 2000)
        self.blocks = nn.ModuleList([TSMixerBlock(seq, d) for _ in range(n_blocks)])
        self.drop = nn.Dropout(0.1)
        self.dec = QuantileDecoder(d, pred, nq)

    def forward(self, x):
        B, V, L = x.shape
        x = self.embed(x.transpose(1, 2))  # (B, L, d)
        x = self.pe(x)
        for blk in self.blocks:
            x = self.drop(blk(x))
        # Take last timestep as summary
        return self.dec(x[:, -1])


# ═══════════════════════════════════════════════════════════════
# MODEL 8: ModernTCN (ICLR 2024)
# ═══════════════════════════════════════════════════════════════

class ModernTCNBlock(nn.Module):
    """Modern TCN block: depthwise conv + pointwise conv + modern norms."""
    def __init__(self, d, kernel=7, dilation=1, drop=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.dw_conv = nn.Conv1d(d, d, kernel, padding=(kernel-1)*dilation//2,
                                  dilation=dilation, groups=d)
        self.ln2 = nn.LayerNorm(d)
        self.pw_conv = nn.Conv1d(d, d*2, 1)  # expand
        self.pw_conv2 = nn.Conv1d(d*2, d, 1)  # project back
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # x: (B, L, d)
        res = x
        x = self.ln1(x)
        x = self.dw_conv(x.transpose(1, 2)).transpose(1, 2)  # (B, L, d)
        x = self.ln2(x)
        x = F.gelu(self.pw_conv(x.transpose(1, 2))).transpose(1, 2)
        x = self.drop(x)
        x = self.pw_conv2(x.transpose(1, 2)).transpose(1, 2)
        return x + res


class ModernTCN(nn.Module):
    """
    ModernTCN: A Pure Modern CNN for Time Series (ICLR 2024)
    Depthwise separable conv + modern norms + dilation schedule.
    """
    def __init__(self, n_vars, seq=168, pred=24, nq=99, d=128, n_blocks=4, kernel=7):
        super().__init__()
        self.embed = nn.Linear(n_vars, d)
        self.pe = SinusoidalPE(d, 2000)
        dilations = [1, 2, 4, 8][:n_blocks]
        self.blocks = nn.ModuleList([
            ModernTCNBlock(d, kernel, dil) for dil in dilations
        ])
        self.dec = QuantileDecoder(d, pred, nq)

    def forward(self, x):
        B, V, L = x.shape
        x = self.embed(x.transpose(1, 2))  # (B, L, d)
        x = self.pe(x)
        for blk in self.blocks:
            x = blk(x)
        return self.dec(x[:, -1])


# ═══════════════════════════════════════════════════════════════
# MODEL 9: TiDE (Google 2024)
# ═══════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """MLP residual block."""
    def __init__(self, d, hidden_factor=2, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d * hidden_factor), nn.GELU(), nn.Dropout(drop),
            nn.Linear(d * hidden_factor, d)
        )
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        return self.ln(self.net(x) + x)


class TiDE(nn.Module):
    """
    TiDE: Time-series Dense Encoder (Google 2024)
    MLP encoder over features + dense temporal decoder with skip connections.
    """
    def __init__(self, n_vars, seq=168, pred=24, nq=99, d=128, n_encoder=2, n_decoder=2):
        super().__init__()
        self.seq = seq; self.pred = pred; self.d = d
        # Encoder: project features per timestep
        self.enc_proj = nn.Linear(n_vars, d)
        self.enc_blocks = nn.ModuleList([ResidualBlock(d) for _ in range(n_encoder)])
        # Temporal decoder: MLP over (seq * d) → decoder hidden
        decoder_in = d  # using last encoder output
        self.dec_blocks = nn.ModuleList([ResidualBlock(decoder_in) for _ in range(n_decoder)])
        # Global residual: skip from encoder mean
        self.global_res = nn.Linear(d, pred)
        self.qproj = nn.Linear(decoder_in, pred * nq)

    def forward(self, x):
        B, V, L = x.shape
        enc = self.enc_proj(x.transpose(1, 2))  # (B, L, d)
        for blk in self.enc_blocks:
            enc = blk(enc)
        # Global residual from encoder
        global_pred = self.global_res(enc.mean(dim=1))  # (B, pred)
        # Decoder: final feature
        dec_out = enc[:, -1]  # (B, d)
        for blk in self.dec_blocks:
            dec_out = blk(dec_out)
        # Quantile projection
        q_out = self.qproj(dec_out).view(B, self.pred, NQ)
        # Add global residual as mean adjustment
        q_out = q_out + global_pred.unsqueeze(-1).expand(-1, -1, NQ)
        return q_out


# ═══════════════════════════════════════════════════════════════
# MODEL 10: TimeMixer (ICLR 2024)
# ═══════════════════════════════════════════════════════════════

class TimeMixer(nn.Module):
    """
    TimeMixer: Decomposable Multiscale Mixing (ICLR 2024)
    Multi-scale past-decomposable mixing + future-predictor mixing.
    """
    def __init__(self, n_vars, seq=168, pred=24, nq=99, d=128, n_scales=3):
        super().__init__()
        self.n_scales = n_scales
        self.d = d
        self.pred_len = pred
        self.nq = nq
        self.embed = nn.Linear(n_vars, d)
        self.pe = SinusoidalPE(d, 2000)

        # Multi-scale downsampling
        self.downsample = nn.ModuleList([
            nn.AvgPool1d(2 ** (i + 1), stride=2 ** (i + 1)) for i in range(n_scales)
        ])

        # Past mixing at each scale
        self.past_mix = nn.ModuleList([
            nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, d))
            for _ in range(n_scales + 1)
        ])

        # Future mixing (cross-scale attention)
        self.future_attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.future_proj = nn.Linear(d, pred)

        self.qproj = nn.Linear(pred, pred * nq)

    def _decompose(self, x):
        """Decompose into seasonal & trend via moving average."""
        # x: (B, L, d)
        L = x.shape[1]
        kernel = max(3, L // 12)
        if kernel % 2 == 0:
            kernel += 1  # ensure odd for symmetric padding
        pad = kernel // 2
        # Use AvgPool1d for exact length preservation
        avg = F.avg_pool1d(
            x.transpose(1, 2), kernel_size=kernel, stride=1, padding=pad
        ).transpose(1, 2)  # (B, L, d)
        return x - avg, avg  # seasonal, trend

    def forward(self, x):
        B, V, L = x.shape
        x = self.embed(x.transpose(1, 2))  # (B, L, d)
        x = self.pe(x)

        # Multi-scale representations
        scales = [x]  # scale 0 = original
        for ds in self.downsample:
            scales.append(ds(x.transpose(1, 2)).transpose(1, 2))

        # Past-decomposable mixing at each scale
        mixed = []
        for i, s in enumerate(scales):
            seasonal, trend = self._decompose(s)
            cat = torch.cat([seasonal.mean(dim=1), trend.mean(dim=1)], dim=-1)  # (B, 2d)
            mixed.append(self.past_mix[i](cat).unsqueeze(1))  # (B, 1, d)
        mixed = torch.cat(mixed, dim=1)  # (B, n_scales+1, d)

        # Future-predictor: cross-scale attention
        fut_out, _ = self.future_attn(mixed, mixed, mixed)  # (B, n_scales+1, d)
        fut = self.future_proj(fut_out.mean(dim=1))  # (B, pred)
        return self.qproj(fut).view(B, self.pred_len, self.nq)


# ═══════════════════════════════════════════════════════════════
# MODEL 11: Crossformer (ICLR 2023)
# ═══════════════════════════════════════════════════════════════

class Crossformer(nn.Module):
    """
    Crossformer: Transformer with Cross-Dimension Attention (ICLR 2023)
    Two-stage attention: DSW (dimension-segment-wise) + cross-dimension.
    Simplified: patch embed → cross-dim attend → decode.
    """
    def __init__(self, n_vars, seq=168, pred=24, nq=99, d=128, n_heads=4,
                 n_layers=2, patch_len=12, stride=8, seg_len=6):
        super().__init__()
        self.n_vars = n_vars
        self.pred_len = pred
        self.nq = nq
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (seq - patch_len) // stride + 1
        self.seg_len = seg_len
        self.n_segments = (self.n_patches + seg_len - 1) // seg_len

        self.patch_embed = nn.Linear(patch_len, d)
        self.patch_pe = nn.Parameter(torch.randn(1, 1, self.n_patches, d) * 0.02)
        self.drop = nn.Dropout(0.1)

        # DSW attention: segment-level
        self.dsw_attns = nn.ModuleList([
            nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
            for _ in range(n_layers)
        ])
        self.dsw_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])

        # Cross-dimension attention
        self.cross_attn = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
        self.cross_norm = nn.LayerNorm(d)

        self.head = nn.Sequential(
            nn.Linear(self.n_patches * d, d * 2), nn.GELU(),
            nn.Linear(d * 2, pred * nq)
        )

    def forward(self, x):
        B, V, L = x.shape
        # Patch embed per variable
        all_v = []
        for v in range(V):
            xv = x[:, v, :]  # (B, L)
            patches = xv.unfold(1, self.patch_len, self.stride)  # (B, n_patches, patch_len)
            patches = self.patch_embed(patches) + self.patch_pe[:, 0]  # (B, n_patches, d)
            all_v.append(patches.unsqueeze(1))  # (B, 1, n_patches, d)
        patches = torch.cat(all_v, dim=1)  # (B, V, n_patches, d)

        # DSW attention: segment patches within each variable
        for attn, norm in zip(self.dsw_attns, self.dsw_norms):
            res = patches
            B2, V2, N2, D2 = patches.shape
            flat = patches.reshape(B2 * V2, N2, D2)  # (B*V, n_patches, d)
            attn_out, _ = attn(flat, flat, flat)
            flat = norm(attn_out + flat)
            patches = flat.reshape(B2, V2, N2, D2) + res

        # Cross-dimension: variables attend to each other at same patch position
        B2, V2, N2, D2 = patches.shape
        flat = patches.permute(0, 2, 1, 3).reshape(B2 * N2, V2, D2)  # (B*n_patches, V, d)
        cross_out, _ = self.cross_attn(flat, flat, flat)
        cross_out = self.cross_norm(cross_out + flat)
        patches = cross_out.reshape(B2, N2, V2, D2).permute(0, 2, 1, 3)  # (B, V, n_patches, d)

        # Flatten: mean pool across variables, flatten patches
        out = patches.mean(dim=1)  # (B, n_patches, d)
        out = out.reshape(B, N2 * D2)  # (B, n_patches*d)
        return self.head(out).view(B, self.pred_len, self.nq)


# ═══════════════════════════════════════════════════════════════
# MODEL 12: LNN-Gated Selective SSM (Ours)
# ═══════════════════════════════════════════════════════════════

class MambaSSM(nn.Module):
    """Fast Liquid-Gated Selective SSM block with parallel scan."""
    def __init__(self, d, ds=16, dc=4, ex=2):
        super().__init__()
        di = d * ex; self.ds = ds
        self.inp  = nn.Linear(d, di * 2, bias=False)
        self.cnv  = nn.Conv1d(di, di, dc, groups=di, padding=dc - 1)
        self.xp   = nn.Linear(di, ds * 2 + 1, bias=False)
        self.dtp  = nn.Linear(ds, di, bias=True)
        A = torch.arange(1, ds + 1).float().unsqueeze(0) * 0.05
        self.A_log = nn.Parameter(torch.log(A))
        self.D     = nn.Parameter(torch.ones(di))
        self.out  = nn.Linear(di, d, bias=False)
        self.nm   = nn.RMSNorm(d)

    def forward(self, x):
        B, L, D = x.shape; res = x
        xz = self.inp(x); u, z = xz.chunk(2, dim=-1)
        u = F.silu(self.cnv(u.transpose(1, 2))[:, :, :L].transpose(1, 2))
        proj = self.xp(u)
        dt = F.softplus(self.dtp(F.softplus(proj[:, :, :self.ds]))) + 1e-4
        Bs, Cs = proj[:, :, self.ds:self.ds*2], proj[:, :, self.ds*2:]
        de = dt.unsqueeze(-1)
        A = -torch.exp(self.A_log)
        Abar = torch.exp(de * A.unsqueeze(0).unsqueeze(1))
        b = de * Bs.unsqueeze(2) * u.unsqueeze(-1)
        eps = 1e-8; logA = torch.log(Abar.clamp(min=eps))
        Acum = torch.exp(torch.cumsum(logA, dim=1))
        h = Acum * torch.cumsum(b / Acum.clamp(min=eps), dim=1)
        y = (h * Cs.unsqueeze(2)).sum(-1) + self.D.unsqueeze(0).unsqueeze(0) * u
        return self.nm(self.out(y * F.silu(z)) + res)


class LNNMamba(nn.Module):
    """
    LNN-Gated Selective SSM: Liquid-Gated Selective SSM + Liquid Time-Constant Gates
    Combines selective state-space temporal encoding with input-dependent
    gating for adaptive wind regime response.
    """
    def __init__(self, n_vars, d=64, n_blocks=2, d_state=16, pred=24, nq=99, use_lnn=True):
        super().__init__()
        self.use_lnn = use_lnn
        self.pred_len = pred
        self.nq = nq
        self.emb = nn.Sequential(nn.Linear(n_vars, d * 2), nn.GELU(), nn.Linear(d * 2, d))
        self.pe  = nn.Parameter(torch.randn(1, 2000, d) * 0.02)
        self.mb  = nn.ModuleList([MambaSSM(d, d_state) for _ in range(n_blocks)])
        self.gates = nn.ModuleList([nn.Sequential(
            nn.GRU(d, 48, batch_first=True), nn.Linear(48, d)
        ) for _ in range(n_blocks)])
        self.dec = QuantileDecoder(d, pred, nq)
        self.drop = nn.Dropout(0.1)
        self.register_buffer('rev_eps', torch.tensor(1e-5))
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 0.5)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        B, V, L = x.shape
        # Partial RevIN: only normalize power target
        power_mu  = x[:, -1:, :].mean(dim=-1, keepdim=True)
        power_sig = torch.sqrt(x[:, -1:, :].var(dim=-1, keepdim=True, unbiased=False) + self.rev_eps)
        x_norm = x.clone()
        x_norm[:, -1:, :] = (x[:, -1:, :] - power_mu) / power_sig

        x_norm = self.emb(x_norm.transpose(1, 2)) + self.pe[:, :L]
        for mb, gate in zip(self.mb, self.gates):
            x_norm = self.drop(mb(x_norm))
            if self.use_lnn:
                h, _ = gate[0](x_norm)
                x_norm = x_norm * torch.sigmoid(gate[1](h))

        out = self.dec(x_norm[:, -1])
        out = out * power_sig.squeeze(-1).unsqueeze(1) + power_mu.squeeze(-1).unsqueeze(1)
        return out


# ═══════════════════════════════════════════════════════════════
# Model Registry
# ═══════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    'persistence': 'Persistence',
    'gru': 'GRUModel',
    'dlinear': 'DLinear',
    'patchtst': 'PatchTST',
    'itransformer': 'iTransformer',
    'timesnet': 'TimesNet',
    'tsmixer': 'TSMixer',
    'moderntcn': 'ModernTCN',
    'tide': 'TiDE',
    'timemixer': 'TimeMixer',
    'crossformer': 'Crossformer',
    'lnnmamba': 'LNNMamba',
}


def create_model(name, n_vars, seq=168, pred=24, nq=99, **kwargs):
    """Factory: create model by name."""
    name = name.lower()
    if name == 'persistence':
        return Persistence(pred=pred, nq=nq)
    elif name == 'gru':
        return GRUModel(n_vars, d=kwargs.get('d', 128), n_layers=kwargs.get('n_layers', 2), pred=pred, nq=nq)
    elif name == 'dlinear':
        return DLinear(n_vars, seq=seq, pred=pred, nq=nq, kernel=kwargs.get('kernel', 25))
    elif name == 'patchtst':
        return PatchTST(n_vars, seq=seq, pred=pred, nq=nq, d=kwargs.get('d', 128),
                        n_heads=kwargs.get('n_heads', 8), n_layers=kwargs.get('n_layers', 3),
                        patch_len=kwargs.get('patch_len', 12), stride=kwargs.get('stride', 8))
    elif name == 'itransformer':
        return iTransformer(n_vars, seq=seq, pred=pred, nq=nq, d=kwargs.get('d', 128),
                            n_heads=kwargs.get('n_heads', 8), n_layers=kwargs.get('n_layers', 3))
    elif name == 'timesnet':
        return TimesNet(n_vars, seq=seq, pred=pred, nq=nq, d=kwargs.get('d', 64),
                        top_k=kwargs.get('top_k', 5), n_blocks=kwargs.get('n_blocks', 2))
    elif name == 'tsmixer':
        return TSMixer(n_vars, seq=seq, pred=pred, nq=nq, d=kwargs.get('d', 128),
                       n_blocks=kwargs.get('n_blocks', 3))
    elif name == 'moderntcn':
        return ModernTCN(n_vars, seq=seq, pred=pred, nq=nq, d=kwargs.get('d', 128),
                         n_blocks=kwargs.get('n_blocks', 4), kernel=kwargs.get('kernel', 7))
    elif name == 'tide':
        return TiDE(n_vars, seq=seq, pred=pred, nq=nq, d=kwargs.get('d', 128),
                    n_encoder=kwargs.get('n_encoder', 2), n_decoder=kwargs.get('n_decoder', 2))
    elif name == 'timemixer':
        return TimeMixer(n_vars, seq=seq, pred=pred, nq=nq, d=kwargs.get('d', 128),
                         n_scales=kwargs.get('n_scales', 3))
    elif name == 'crossformer':
        return Crossformer(n_vars, seq=seq, pred=pred, nq=nq, d=kwargs.get('d', 128),
                           n_heads=kwargs.get('n_heads', 4), n_layers=kwargs.get('n_layers', 2),
                           patch_len=kwargs.get('patch_len', 12), stride=kwargs.get('stride', 8))
    elif name == 'lnnmamba':
        return LNNMamba(n_vars, d=kwargs.get('d', 64), n_blocks=kwargs.get('n_blocks', 2),
                        d_state=kwargs.get('d_state', 16), pred=pred, nq=nq,
                        use_lnn=kwargs.get('use_lnn', True))
    else:
        raise ValueError(f"Unknown model: {name}. Choose from: {list(MODEL_REGISTRY.keys())}")


# ═══════════════════════════════════════════════════════════════
# Training & Evaluation
# ═══════════════════════════════════════════════════════════════

def train_model(model, tl, vl, device, epochs=30, lr=1e-3, patience=10, label=''):
    """Train one model with AMP, cosine schedule, early stopping."""
    if isinstance(model, Persistence):
        return 0.0, []

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    q_tensor = torch.tensor(QUANTILES, dtype=torch.float32, device=device)
    best_loss = float('inf')
    best_state = None
    hist = []

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tl_loss = 0.0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    loss = pinball_loss(model(x), y, q_tensor)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss = pinball_loss(model(x), y, q_tensor)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tl_loss += loss.item()
        sch.step()

        # Validation
        model.eval()
        vpreds, vtargs = [], []
        with torch.no_grad():
            for x, y in vl:
                out = model(x.to(device))
                vpreds.append(out.cpu())
                vtargs.append(y)
        vp = torch.cat(vpreds, dim=0)
        vt = torch.cat(vtargs, dim=0)
        v_loss = pinball_loss(vp.to(device), vt.to(device), q_tensor).item()
        hist.append(v_loss)
        t = time.time() - t0

        if v_loss < best_loss:
            best_loss = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 5 == 0 or ep == 1 or ep == epochs:
            marker = " *" if v_loss == best_loss else ""
            print(f'  {label:>15s} E{ep:2d} | loss={tl_loss/len(tl):.4f} | val_pb={v_loss:.4f} | {t:.0f}s{marker}')

        # Early stopping
        if patience > 0 and ep >= patience + 5:
            recent_best = np.argmin(hist[-patience:])
            if recent_best == 0 and ep > patience:
                print(f'  Early stop at epoch {ep}')
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_loss, hist


def evaluate_model(model, loader, scaler_y, device):
    """Evaluate: pinball loss, per-horizon metrics, predictions."""
    if isinstance(model, Persistence):
        model = model.to(device)
    model.eval()
    q_tensor = torch.tensor(QUANTILES, dtype=torch.float32, device=device)
    total_pb = 0.0
    preds, targs = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_pb += pinball_loss(out, y, q_tensor).item()
            preds.append(out.cpu().numpy())
            targs.append(y.cpu().numpy())

    pr = np.concatenate(preds)
    tr = np.concatenate(targs)

    if scaler_y is not None and not isinstance(model, Persistence):
        sh = pr.shape
        pr = scaler_y.inverse_transform(pr.reshape(-1, sh[2])).reshape(sh)
        tr = scaler_y.inverse_transform(tr.reshape(-1, 1)).reshape(tr.shape)

    # Per-horizon pinball
    pb_h = []
    q_t = torch.tensor(QUANTILES, dtype=torch.float32)
    for h in range(pr.shape[1]):
        er = torch.FloatTensor(tr[:, h]).unsqueeze(-1) - torch.FloatTensor(pr[:, h])
        pb = torch.maximum(q_t * er, (q_t - 1) * er).mean().item()
        pb_h.append(pb)

    return total_pb / len(loader), pr, tr, pb_h


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='SOTA Benchmark: 12 Models for GEFCom2014 WPF')
    parser.add_argument('--zones', type=str, default='1,2,3,4,5,6,7,8,9,10',
                        help='Comma-separated zone IDs')
    parser.add_argument('--models', type=str,
                        default='persistence,gru,dlinear,patchtst,itransformer,timesnet,tsmixer,moderntcn,tide,timemixer,crossformer,lnnmamba',
                        help='Comma-separated model names')
    parser.add_argument('--seq', type=int, default=168)
    parser.add_argument('--pred', type=int, default=24)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--device', type=str, default='')
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    zones = [int(z.strip()) for z in args.zones.split(',')]
    model_names = [m.strip().lower() for m in args.models.split(',')]

    print('=' * 80)
    print('SOTA Benchmark: 12 Models for GEFCom2014 Wind Power Forecasting')
    print(f'Models: {len(model_names)} | Zones: {len(zones)} | Device: {device}')
    print(f'{args.seq}h NWP → {args.pred}h power | {NQ} quantiles | Pinball Loss')
    print('=' * 80)

    # Results accumulator
    all_results = {}  # {model_name: {zone: pinball}}

    for mi, mname in enumerate(model_names):
        print(f'\n{"#" * 80}')
        print(f'# MODEL {mi+1}/{len(model_names)}: {mname.upper()}')
        print(f'{"#" * 80}')

        zone_results = {}
        for zid in zones:
            print(f'\n--- Zone {zid} ---')
            (tl, vl, testl), n_vars, scaler_y, info = load_zone_nwp(
                zid, args.seq, args.pred, args.batch, args.stride)
            print(f'  {info}')

            model = create_model(mname, n_vars, args.seq, args.pred, NQ, d=args.d_model)
            model = model.to(device) if not isinstance(model, Persistence) else model

            if not isinstance(model, Persistence):
                n_p = sum(p.numel() for p in model.parameters())
                print(f'  Params: {n_p:,}')

            best_val, hist = train_model(model, tl, vl, device, args.epochs, args.lr,
                                         args.patience, mname)
            test_pb, test_pr, test_tr, pb_h = evaluate_model(model, testl, scaler_y, device)
            zone_results[zid] = {'pinball': test_pb, 'pb_horizon': pb_h}

            # Print per-horizon for first few zones
            if zid == zones[0]:
                print(f'  Per-horizon pinball:')
                for h_idx in [0, 3, 5, 11, 17, 23]:
                    print(f'    +{h_idx+1:2d}h: {pb_h[h_idx]:.4f}')

        all_results[mname] = zone_results

        # Running average (for early stopping on poor models)
        avg_pb = np.mean([zone_results[z]['pinball'] for z in zones]) if zone_results else 0
        print(f'\n  >>> {mname.upper()} avg pinball: {avg_pb:.4f}')

    # ═══════════════════ Final Summary ═══════════════════
    print(f'\n{"=" * 80}')
    print(f'{"FINAL RESULTS":^80}')
    print(f'{args.seq}h NWP → {args.pred}h power | {NQ} quantiles | Pinball Loss')
    print(f'{"=" * 80}')

    # Header
    header = f'{"Model":>16s}'
    for zid in zones:
        header += f'  Zone{zid:>2d}'
    header += f'  {"Avg":>8s}  {"vsPersist":>10s}'
    print(header)
    print('-' * len(header))

    # Get persistence for baseline comparison
    pers_avg = None
    if 'persistence' in all_results:
        pers_avg = np.mean([all_results['persistence'][z]['pinball'] for z in zones])

    for mname in model_names:
        if mname not in all_results:
            continue
        row = f'{mname:>16s}'
        zone_pbs = []
        for zid in zones:
            if zid in all_results[mname]:
                pb = all_results[mname][zid]['pinball']
                zone_pbs.append(pb)
                row += f'  {pb:.4f}'
            else:
                row += f'  {"---":>7s}'
        avg = np.mean(zone_pbs)
        row += f'  {avg:8.4f}'
        if pers_avg is not None and mname != 'persistence':
            imp = (pers_avg - avg) / pers_avg * 100
            row += f'  {imp:+9.2f}%'
        elif mname == 'persistence':
            row += f'  {"(baseline)":>10s}'
        print(row)

    # Best model
    best_name = min(
        [m for m in model_names if m in all_results and m != 'persistence'],
        key=lambda m: np.mean([all_results[m][z]['pinball'] for z in zones])
    )
    best_avg = np.mean([all_results[best_name][z]['pinball'] for z in zones])
    print(f'\n*** Best model: {best_name.upper()} (pinball={best_avg:.4f}) ***')

    # Save results
    os.makedirs('checkpoints', exist_ok=True)
    save_data = {}
    for mname in model_names:
        if mname in all_results:
            for zid in zones:
                if zid in all_results[mname]:
                    save_data[f'{mname}_z{zid}'] = all_results[mname][zid]['pinball']
    np.savez('checkpoints/sota_results.npz', **save_data)

    # Save CSV
    rows = []
    for mname in model_names:
        if mname in all_results:
            row = {'model': mname}
            for zid in zones:
                if zid in all_results[mname]:
                    row[f'zone_{zid}'] = all_results[mname][zid]['pinball']
            row['avg'] = np.mean([all_results[mname][z]['pinball'] for z in zones])
            rows.append(row)
    pd.DataFrame(rows).to_csv('checkpoints/sota_results.csv', index=False)

    print(f'\nResults saved to checkpoints/sota_results.npz + .csv')
    print('Done!')


if __name__ == '__main__':
    main()
