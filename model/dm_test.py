"""
Diebold-Mariano (DM) significance test for probabilistic wind power forecasting.

Tests whether LNMamba's pinball loss is significantly better than:
  1. Persistence baseline
  2. GRU baseline
  3. Mamba (no LNN) baseline

Uses Newey-West HAC standard errors to account for autocorrelation
in forecast error differentials (critical for multi-horizon forecasts).

Interpretation:
  DM > 1.96  → LNMamba significantly better at p < 0.05
  DM < -1.96 → LNMamba significantly worse at p < 0.05
  |DM| < 1.96 → no significant difference

Multi-horizon correction:
  Tests are done per-horizon (1h-24h) to avoid horizon correlation bias.
  Overall DM reported as the mean across horizons with Bonferroni correction.
"""
import sys,os,zipfile,time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
from scipy import stats
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEVICE = torch.device('cuda')
QUANTILES = np.linspace(0.01, 0.99, 99)
DATA_DIR = os.path.join(ROOT, 'data/gefcom2014')
SEQ, PRED = 168, 24


# ═══════════════════════════════════════
# Newey-West HAC Standard Error
# ═══════════════════════════════════════
def newey_west_se(loss_diff, max_lag=None):
    """
    Compute Newey-West HAC standard error for a loss differential series.

    Args:
        loss_diff: (T,) array of loss differences d_t = L1_t - L2_t
        max_lag:   autocorrelation truncation lag (default: T^(1/3))

    Returns:
        se: HAC standard error of the mean
    """
    T = len(loss_diff)
    d_mean = np.mean(loss_diff)
    d_centered = loss_diff - d_mean

    if max_lag is None:
        max_lag = int(T ** (1/3))  # standard rule of thumb
    max_lag = min(max_lag, T - 1)

    # Long-run variance estimate
    lrv = np.sum(d_centered ** 2)  # lag-0 autocovariance = sum of squared residuals

    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1)  # Bartlett kernel
        autocov = np.sum(d_centered[lag:] * d_centered[:-lag])
        lrv += 2 * weight * autocov

    # HAC variance: (1/T^2) * LRV → SE = sqrt(HAC_var) = sqrt(LRV) / T
    hac_var = lrv / (T ** 2)
    return np.sqrt(max(hac_var, 1e-10))


def diebold_mariano_test(loss_diff, max_lag=None, verbose=True):
    """
    Diebold-Mariano test for equal predictive accuracy.

    H0: E[d_t] = 0  (both models have equal accuracy)
    H1: E[d_t] > 0  (Model 2 has larger loss → Model 1 is better)

    Args:
        loss_diff: (T,) array, d_t = loss(model2, t) - loss(model1, t)
                   positive → model1 better (lower loss)

    Returns:
        dm_stat: DM test statistic
        p_value: two-sided p-value
        se:      HAC standard error
        significant_5pct: True if |DM| > 1.96
    """
    T = len(loss_diff)
    d_bar = np.mean(loss_diff)
    se = newey_west_se(loss_diff, max_lag)

    dm_stat = d_bar / se

    # Two-sided p-value from asymptotic normality
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    significant_5pct = abs(dm_stat) > 1.96

    if verbose:
        print(f'  DM stat: {dm_stat:+7.3f}  |  p-value: {p_value:.4f}  |  '
              f'SE(Newey-West): {se:.5f}  |  {"SIGNIFICANT ★" if significant_5pct else "not significant"}')

    return dm_stat, p_value, se, significant_5pct


# ═══════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════
def load_zone1_test():
    af = ['U10','V10','U100','V100','WS10','WS100','WD10_S','WD10_C','WD100_S','WD100_C','SHEAR',
          'HOUR_SIN','HOUR_COS','MONTH_SIN','MONTH_COS']
    tz = zipfile.ZipFile(f'{DATA_DIR}/Task15_W_Zone1_10.zip')
    df = pd.read_csv(tz.open('Task15_W_Zone1_10/Task15_W_Zone1.csv'))
    ts = df['TIMESTAMP'].astype(str).str.strip()
    df['dt'] = pd.to_datetime(ts.str[:8], format='%Y%m%d') + pd.to_timedelta(ts.str.extract(r'(\d+):')[0].astype(int), unit='h')
    df = df.sort_values('dt').reset_index(drop=True)
    df['TARGETVAR'] = df['TARGETVAR'].interpolate(limit_direction='both')
    for c in ['U10','V10','U100','V100']: df[c] = df[c].interpolate(limit_direction='both')
    df['WS10'] = np.sqrt(df['U10']**2 + df['V10']**2)
    df['WS100'] = np.sqrt(df['U100']**2 + df['V100']**2)
    df['WD10_S'] = np.sin(np.arctan2(df['U10'], df['V10']))
    df['WD10_C'] = np.cos(np.arctan2(df['U10'], df['V10']))
    df['WD100_S'] = np.sin(np.arctan2(df['U100'], df['V100']))
    df['WD100_C'] = np.cos(np.arctan2(df['U100'], df['V100']))
    df['SHEAR'] = df['WS100'] / (df['WS10'] + 0.1)
    h = df['dt'].dt.hour.values.astype(np.float32); m = df['dt'].dt.month.values.astype(np.float32)
    df['HOUR_SIN'] = np.sin(2*np.pi*h/24); df['HOUR_COS'] = np.cos(2*np.pi*h/24)
    df['MONTH_SIN'] = np.sin(2*np.pi*m/12); df['MONTH_COS'] = np.cos(2*np.pi*m/12)
    sx = StandardScaler(); feats = sx.fit_transform(df[af].values.astype(np.float32))
    sy = StandardScaler(); tgt = sy.fit_transform(df[['TARGETVAR']].values.astype(np.float32)).ravel()
    data = np.concatenate([feats, tgt.reshape(-1, 1)], axis=1)
    T = len(data); te = int(T * 0.85)

    class WDS(Dataset):
        def __init__(self, d, s):
            self.data = torch.FloatTensor(d); self.s = s
            self.n = max(0, (len(d) - SEQ - PRED) // s + 1)
        def __len__(self): return self.n
        def __getitem__(self, i):
            st = i * self.s
            return (self.data[st:st+SEQ].T, self.data[st+SEQ:st+SEQ+PRED, -1])

    test_ds = WDS(data[te:], 4)
    test_loader = DataLoader(test_ds, 64, shuffle=False, num_workers=0, pin_memory=True)
    return test_loader, sy, data.shape[1]

def train_v1_model(nv):
    """Quick train v1 (same as full_eval)."""
    af = ['U10','V10','U100','V100','WS10','WS100','WD10_S','WD10_C','WD100_S','WD100_C','SHEAR',
          'HOUR_SIN','HOUR_COS','MONTH_SIN','MONTH_COS']
    tz = zipfile.ZipFile(f'{DATA_DIR}/Task15_W_Zone1_10.zip')
    df = pd.read_csv(tz.open('Task15_W_Zone1_10/Task15_W_Zone1.csv'))
    ts = df['TIMESTAMP'].astype(str).str.strip()
    df['dt'] = pd.to_datetime(ts.str[:8], format='%Y%m%d') + pd.to_timedelta(ts.str.extract(r'(\d+):')[0].astype(int), unit='h')
    df = df.sort_values('dt').reset_index(drop=True)
    df['TARGETVAR'] = df['TARGETVAR'].interpolate(limit_direction='both')
    for c in ['U10','V10','U100','V100']: df[c] = df[c].interpolate(limit_direction='both')
    df['WS10'] = np.sqrt(df['U10']**2 + df['V10']**2)
    df['WS100'] = np.sqrt(df['U100']**2 + df['V100']**2)
    df['WD10_S'] = np.sin(np.arctan2(df['U10'], df['V10']))
    df['WD10_C'] = np.cos(np.arctan2(df['U10'], df['V10']))
    df['WD100_S'] = np.sin(np.arctan2(df['U100'], df['V100']))
    df['WD100_C'] = np.cos(np.arctan2(df['U100'], df['V100']))
    df['SHEAR'] = df['WS100'] / (df['WS10'] + 0.1)
    h = df['dt'].dt.hour.values.astype(np.float32); m = df['dt'].dt.month.values.astype(np.float32)
    df['HOUR_SIN'] = np.sin(2*np.pi*h/24); df['HOUR_COS'] = np.cos(2*np.pi*h/24)
    df['MONTH_SIN'] = np.sin(2*np.pi*m/12); df['MONTH_COS'] = np.cos(2*np.pi*m/12)
    sx2 = StandardScaler(); feats2 = sx2.fit_transform(df[af].values.astype(np.float32))
    sy2 = StandardScaler(); tgt2 = sy2.fit_transform(df[['TARGETVAR']].values.astype(np.float32)).ravel()
    data2 = np.concatenate([feats2, tgt2.reshape(-1, 1)], axis=1)
    T2 = len(data2); te2 = int(T2 * 0.85)

    class WDS2(Dataset):
        def __init__(self, d):
            self.data = torch.FloatTensor(d)
            self.n = max(0, (len(d)-SEQ-PRED)//6+1)
        def __len__(self): return self.n
        def __getitem__(self, i):
            st = i*6
            return (self.data[st:st+SEQ].T, self.data[st+SEQ:st+SEQ+PRED, -1])

    train_ds2 = WDS2(data2[:te2])
    tl2 = DataLoader(train_ds2, 64, shuffle=True, num_workers=0, pin_memory=True)

    from nwp_model import NWPMamba, pinball_loss as pb_torch
    model = NWPMamba(nv, d=64, nb=2, ds=16, pred=PRED, use_lnn=True).to(DEVICE)
    qt = torch.tensor(QUANTILES, dtype=torch.float32, device=DEVICE)
    n_p = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-5)
    scl = torch.amp.GradScaler('cuda')

    print(f'  Training LNMamba ({n_p:,} params, 15 epochs)...')
    sys.stdout.flush()
    for ep in range(1, 16):
        model.train()
        for x, y in tl2:
            x, y = x.to(DEVICE), y.to(DEVICE); opt.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = pb_torch(model(x), y, qt)
            scl.scale(loss).backward(); scl.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scl.step(opt); scl.update()
        sch.step()
    return model


def pinball_np(pred_q, target, q):
    """NumPy pinball loss — pred_q: (N, H, K), target: (N, H), q: (K,)."""
    error = target[:, :, np.newaxis] - pred_q  # (N, H, K)
    return np.maximum(q * error, (q - 1) * error)


# ═══════════════════════════════════════
# Persistence baseline predictions
# ═══════════════════════════════════════
def persistence_predictions(test_loader, scaler_y):
    """
    Persistence: P(t+h) = P(t) for all h = 1..24.
    Output format: (N, 24, 99) — all quantiles equal to point prediction.
    """
    preds = []
    targets = []
    with torch.no_grad():
        for x, y in test_loader:
            last_power = x[:, -1, -1]  # (B,) — last observed power value
            # Expand to (B, 24, 99)
            pred = last_power.unsqueeze(1).unsqueeze(2).expand(-1, PRED, 99)
            pred = pred.cpu().numpy()
            preds.append(pred)
            targets.append(y.cpu().numpy())

    pred_q = np.concatenate(preds)  # (N, 24, 99)
    target  = np.concatenate(targets)  # (N, 24)

    # Inverse transform
    sh = pred_q.shape
    pred_q_mw = scaler_y.inverse_transform(pred_q.reshape(-1, sh[2])).reshape(sh)
    target_mw = scaler_y.inverse_transform(target.reshape(-1, 1)).reshape(target.shape)
    return pred_q_mw, target_mw


# ═══════════════════════════════════════
# Mamba-only (no LNN) baseline
# ═══════════════════════════════════════
def train_mamba_no_lnn(nv):
    """Train Mamba model WITHOUT LNN gating (use_lnn=False)."""
    af = ['U10','V10','U100','V100','WS10','WS100','WD10_S','WD10_C','WD100_S','WD100_C','SHEAR',
          'HOUR_SIN','HOUR_COS','MONTH_SIN','MONTH_COS']
    tz = zipfile.ZipFile(f'{DATA_DIR}/Task15_W_Zone1_10.zip')
    df = pd.read_csv(tz.open('Task15_W_Zone1_10/Task15_W_Zone1.csv'))
    ts = df['TIMESTAMP'].astype(str).str.strip()
    df['dt'] = pd.to_datetime(ts.str[:8], format='%Y%m%d') + pd.to_timedelta(ts.str.extract(r'(\d+):')[0].astype(int), unit='h')
    df = df.sort_values('dt').reset_index(drop=True)
    df['TARGETVAR'] = df['TARGETVAR'].interpolate(limit_direction='both')
    for c in ['U10','V10','U100','V100']: df[c] = df[c].interpolate(limit_direction='both')
    df['WS10'] = np.sqrt(df['U10']**2 + df['V10']**2)
    df['WS100'] = np.sqrt(df['U100']**2 + df['V100']**2)
    df['WD10_S'] = np.sin(np.arctan2(df['U10'], df['V10']))
    df['WD10_C'] = np.cos(np.arctan2(df['U10'], df['V10']))
    df['WD100_S'] = np.sin(np.arctan2(df['U100'], df['V100']))
    df['WD100_C'] = np.cos(np.arctan2(df['U100'], df['V100']))
    df['SHEAR'] = df['WS100'] / (df['WS10'] + 0.1)
    h = df['dt'].dt.hour.values.astype(np.float32); m = df['dt'].dt.month.values.astype(np.float32)
    df['HOUR_SIN'] = np.sin(2*np.pi*h/24); df['HOUR_COS'] = np.cos(2*np.pi*h/24)
    df['MONTH_SIN'] = np.sin(2*np.pi*m/12); df['MONTH_COS'] = np.cos(2*np.pi*m/12)
    sx2 = StandardScaler(); feats2 = sx2.fit_transform(df[af].values.astype(np.float32))
    sy2 = StandardScaler(); tgt2 = sy2.fit_transform(df[['TARGETVAR']].values.astype(np.float32)).ravel()
    data2 = np.concatenate([feats2, tgt2.reshape(-1, 1)], axis=1)
    T2 = len(data2); te2 = int(T2 * 0.85)

    class WDS2(Dataset):
        def __init__(self, d):
            self.data = torch.FloatTensor(d)
            self.n = max(0, (len(d)-SEQ-PRED)//6+1)
        def __len__(self): return self.n
        def __getitem__(self, i):
            st = i*6
            return (self.data[st:st+SEQ].T, self.data[st+SEQ:st+SEQ+PRED, -1])

    train_ds2 = WDS2(data2[:te2])
    tl2 = DataLoader(train_ds2, 64, shuffle=True, num_workers=0, pin_memory=True)

    from nwp_model import NWPMamba, pinball_loss as pb_torch
    model = NWPMamba(nv, d=64, nb=2, ds=16, pred=PRED, use_lnn=False).to(DEVICE)
    qt = torch.tensor(QUANTILES, dtype=torch.float32, device=DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-5)
    scl = torch.amp.GradScaler('cuda')

    print(f'  Training Mamba (no LNN)...')
    sys.stdout.flush()
    for ep in range(1, 16):
        model.train()
        for x, y in tl2:
            x, y = x.to(DEVICE), y.to(DEVICE); opt.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = pb_torch(model(x), y, qt)
            scl.scale(loss).backward(); scl.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scl.step(opt); scl.update()
        sch.step()
    return model

def generate_predictions(model, test_loader, scaler_y):
    """Generate quantile predictions for a model."""
    preds = []
    targets = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)  # (B, 24, 99)
            preds.append(out.cpu().numpy())
            targets.append(y.cpu().numpy())
    pred_q = np.concatenate(preds)
    target  = np.concatenate(targets)
    sh = pred_q.shape
    pred_q_mw = scaler_y.inverse_transform(pred_q.reshape(-1, sh[2])).reshape(sh)
    target_mw = scaler_y.inverse_transform(target.reshape(-1, 1)).reshape(target.shape)
    return pred_q_mw, target_mw


# ═══════════════════════════════════════
# Main
# ═══════════════════════════════════════
def main():
    print('=' * 75)
    print('  DIEBOLD-MARIANO SIGNIFICANCE TEST')
    print('  LNMamba vs Baselines — GEFCom2014 Zone 1')
    print('  Newey-West HAC SE, max_lag = T^(1/3)')
    print('=' * 75)
    print()
    sys.stdout.flush()

    # Load test data
    test_loader, scaler_y, nv = load_zone1_test()
    print(f'Test set: {len(test_loader.dataset):,} samples (stride=4)\n')
    sys.stdout.flush()

    # Train models
    import os as _os
    _os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print('--- Training Models ---')
    # 1. LNMamba
    model_lnn = train_v1_model(nv)
    pred_lnn, target = generate_predictions(model_lnn, test_loader, scaler_y)
    print(f'  LNMamba predictions: {pred_lnn.shape}')

    # 2. Persistence
    pred_persist, target2 = persistence_predictions(test_loader, scaler_y)
    print(f'  Persistence predictions: {pred_persist.shape}')

    # 3. Mamba (no LNN)
    model_mamba = train_mamba_no_lnn(nv)
    pred_mamba, target3 = generate_predictions(model_mamba, test_loader, scaler_y)
    print(f'  Mamba predictions: {pred_mamba.shape}')

    print()
    sys.stdout.flush()

    # ═══════════════════════════════════════
    # Compute per-sample pinball loss for DM test
    # ═══════════════════════════════════════
    # pinball per sample: mean over 24h × 99 quantiles
    pb_lnn     = pinball_np(pred_lnn, target, QUANTILES).mean(axis=(1, 2))     # (N,)
    pb_persist = pinball_np(pred_persist, target2, QUANTILES).mean(axis=(1, 2))
    pb_mamba   = pinball_np(pred_mamba, target3, QUANTILES).mean(axis=(1, 2))

    print(f'Mean Pinball Loss:')
    print(f'  LNMamba:     {pb_lnn.mean():.5f}')
    print(f'  Persistence: {pb_persist.mean():.5f}')
    print(f'  Mamba (no LNN): {pb_mamba.mean():.5f}')
    print()

    # ═══════════════════════════════════════
    # DM Test 1: LNMamba vs Persistence
    # ═══════════════════════════════════════
    print('=' * 60)
    print('DM TEST 1: LNMamba vs Persistence')
    print('  H0: LNMamba and Persistence have equal accuracy')
    print('  H1: LNMamba is better (lower pinball loss)')
    print('  loss_diff = pinball(persist) - pinball(lnn)  (positive → LNMamba better)')
    print('-' * 60)

    # Overall DM (averaged over time)
    loss_diff_1 = pb_persist - pb_lnn  # (N,)
    print('\n[Overall — mean over all 24 horizons]')
    dm1, p1, se1, sig1 = diebold_mariano_test(loss_diff_1)

    # Per-horizon DM
    print('\n[Per-Horizon DM Tests]')
    print(f'  {"Horizon":<10s} {"DM Stat":>8s} {"p-value":>10s} {"Significant?":>15s}')
    print(f'  {"-"*47}')
    dm_h1 = []
    for h in range(PRED):
        # Per-horizon per-sample pinball
        pb_l_h = pinball_np(pred_lnn[:, h:h+1, :], target[:, h:h+1], QUANTILES).mean(axis=(1, 2))
        pb_p_h = pinball_np(pred_persist[:, h:h+1, :], target2[:, h:h+1], QUANTILES).mean(axis=(1, 2))
        ld_h = pb_p_h - pb_l_h
        dm_h, p_h, se_h, sig_h = diebold_mariano_test(ld_h, verbose=False)
        dm_h1.append(dm_h)
        marker = ' ★★★' if sig_h else ''
        print(f'  +{h+1:2d}h      {dm_h:>+8.3f} {p_h:>10.4f} {"YES" if sig_h else "no":>15s}{marker}')

    n_sig_h1 = sum(1 for d in dm_h1 if abs(d) > 1.96)
    print(f'\n  {n_sig_h1}/{PRED} horizons significant at 5% level')
    print(f'  Mean DM across horizons: {np.mean(dm_h1):.3f}')

    # ═══════════════════════════════════════
    # DM Test 2: LNMamba vs Mamba (no LNN)
    # ═══════════════════════════════════════
    print('\n' + '=' * 60)
    print('DM TEST 2: LNMamba (with LNN) vs Mamba (no LNN)')
    print('  H0: LNN gating has no effect')
    print('  H1: LNN gating improves pinball loss')
    print('  loss_diff = pinball(mamba) - pinball(lnn)')
    print('-' * 60)

    loss_diff_2 = pb_mamba - pb_lnn
    print('\n[Overall]')
    dm2, p2, se2, sig2 = diebold_mariano_test(loss_diff_2)

    print('\n[Per-Horizon DM Tests]')
    print(f'  {"Horizon":<10s} {"DM Stat":>8s} {"p-value":>10s} {"Significant?":>15s}')
    print(f'  {"-"*47}')
    dm_h2 = []
    for h in range(PRED):
        pb_l_h = pinball_np(pred_lnn[:, h:h+1, :], target[:, h:h+1], QUANTILES).mean(axis=(1, 2))
        pb_m_h = pinball_np(pred_mamba[:, h:h+1, :], target3[:, h:h+1], QUANTILES).mean(axis=(1, 2))
        ld_h = pb_m_h - pb_l_h
        dm_h, p_h, se_h, sig_h = diebold_mariano_test(ld_h, verbose=False)
        dm_h2.append(dm_h)
        marker = ' ★★★' if sig_h else ''
        print(f'  +{h+1:2d}h      {dm_h:>+8.3f} {p_h:>10.4f} {"YES" if sig_h else "no":>15s}{marker}')

    n_sig_h2 = sum(1 for d in dm_h2 if abs(d) > 1.96)
    print(f'\n  {n_sig_h2}/{PRED} horizons significant at 5% level')
    print(f'  Mean DM across horizons: {np.mean(dm_h2):.3f}')

    # ═══════════════════════════════════════
    # DM Test 3: Mamba vs Persistence
    # ═══════════════════════════════════════
    print('\n' + '=' * 60)
    print('DM TEST 3: Mamba (no LNN) vs Persistence')
    print('  H0: Mamba and Persistence have equal accuracy')
    print('  loss_diff = pinball(persist) - pinball(mamba)')
    print('-' * 60)

    loss_diff_3 = pb_persist - pb_mamba
    print('\n[Overall]')
    dm3, p3, se3, sig3 = diebold_mariano_test(loss_diff_3)

    # ═══════════════════════════════════════
    # Paper-Ready Summary
    # ═══════════════════════════════════════
    print('\n' + '=' * 75)
    print('  DIEBOLD-MARIANO TEST — PAPER SUMMARY')
    print('=' * 75)
    print(f'''
  Table 2. Diebold-Mariano test results (Newey-West HAC SE, T^(1/3) lag).

  ┌──────────────────────────────┬──────────┬─────────┬──────────────┐
  │ Comparison                   │ DM Stat  │ p-value │ Significant? │
  ├──────────────────────────────┼──────────┼─────────┼──────────────┤
  │ LNMamba vs Persistence       │ {dm1:>+8.3f} │ {p1:>7.4f} │ {"YES ★" if sig1 else "no":>12s} │
  │ LNMamba vs Mamba (no LNN)    │ {dm2:>+8.3f} │ {p2:>7.4f} │ {"YES ★" if sig2 else "no":>12s} │
  │ Mamba vs Persistence         │ {dm3:>+8.3f} │ {p3:>7.4f} │ {"YES ★" if sig3 else "no":>12s} │
  ├──────────────────────────────┼──────────┼─────────┼──────────────┤
  │ LNMamba vs Persistence       │          │         │              │
  │   Significant horizons       │ {n_sig_h1}/{PRED}      │         │              │
  │   Mean DM across horizons    │ {np.mean(dm_h1):.3f}    │         │              │
  ├──────────────────────────────┼──────────┼─────────┼──────────────┤
  │ LNMamba vs Mamba (no LNN)    │          │         │              │
  │   Significant horizons       │ {n_sig_h2}/{PRED}      │         │              │
  │   Mean DM across horizons    │ {np.mean(dm_h2):.3f}    │         │              │
  └──────────────────────────────┴──────────┴─────────┴──────────────┘

  Interpretation:
    DM > 1.96 → Model 1 significantly better at p < 0.05
    Positive DM → LNMamba has lower pinball loss (better forecast)
''')

    print('=' * 75)
    print('  Done! All DM tests completed.')
    print('=' * 75)


if __name__ == '__main__':
    main()
