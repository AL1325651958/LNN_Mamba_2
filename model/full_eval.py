"""
Comprehensive evaluation suite for probabilistic wind power forecasting.

Metrics:
  1. Pinball Loss (weighted, all 99 quantiles)
  2. Winkler Score (80% CI: coverage penalty + width penalty)
  3. CRPS (Continuous Ranked Probability Score)
  4. Reliability (coverage calibration across quantile levels)
  5. Sharpness (average prediction interval width)
  6. Point Forecast: RMSE, MAE, MAPE (from median q50)

All computed on GEFCom2014 Zone 1 test set using the best v1 model.
"""
import sys,os,zipfile,time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEVICE = torch.device('cuda')
QUANTILES = np.linspace(0.01, 0.99, 99)
DATA_DIR = os.path.join(ROOT, 'data/gefcom2014')
SEQ, PRED = 168, 24

# ═══════════════════════════════════════
# Load Zone 1 data + trained model
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
            self.data = torch.FloatTensor(d)
            self.s = s
            self.n = max(0, (len(d) - SEQ - PRED) // s + 1)
        def __len__(self):
            return self.n
        def __getitem__(self, i):
            st = i * self.s
            return (self.data[st:st+SEQ].T, self.data[st+SEQ:st+SEQ+PRED, -1])

    test_ds = WDS(data[te:], 4)
    test_loader = DataLoader(test_ds, 64, shuffle=False, num_workers=0, pin_memory=True)
    return test_loader, sy, data.shape[1]

def load_v1_model(nv):
    """Rebuild and train v1 model (15 epochs on Zone 1)."""
    # Load train data (same as load_zone1_test but with train split)
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
        def __len__(self):
            return self.n
        def __getitem__(self, i):
            st = i*6
            return (self.data[st:st+SEQ].T, self.data[st+SEQ:st+SEQ+PRED, -1])

    train_ds2 = WDS2(data2[:te2])
    tl2 = DataLoader(train_ds2, 64, shuffle=True, num_workers=0, pin_memory=True)

    from nwp_model import NWPMamba, pinball_loss
    model = NWPMamba(nv, d=64, nb=2, ds=16, pred=PRED, use_lnn=True).to(DEVICE)
    qt = torch.tensor(QUANTILES, dtype=torch.float32, device=DEVICE)

    n_p = sum(p.numel() for p in model.parameters())
    print(f'Training v1 model ({n_p:,} params, 15 epochs)...'); sys.stdout.flush()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15, eta_min=1e-5)
    scl = torch.amp.GradScaler('cuda')

    for ep in range(1, 16):
        model.train()
        for x, y in tl2:
            x, y = x.to(DEVICE), y.to(DEVICE); opt.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = pinball_loss(model(x), y, qt)
            scl.scale(loss).backward(); scl.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scl.step(opt); scl.update()
        sch.step()
        if ep % 5 == 1:
            print(f'  E{ep:2d}...'); sys.stdout.flush()

    return model


# ═══════════════════════════════════════
# Generate predictions
# ═══════════════════════════════════════
print('Loading data...'); sys.stdout.flush()
test_loader, scaler_y, nv = load_zone1_test()
print(f'n_vars={nv}, test batches={len(test_loader)}'); sys.stdout.flush()

print('Training model...'); sys.stdout.flush()
model = load_v1_model(nv)
model.eval()

print('Generating predictions...'); sys.stdout.flush()
qt = torch.tensor(QUANTILES, dtype=torch.float32, device=DEVICE)

all_preds = []
all_targets = []
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)  # (B, 24, 99)
        all_preds.append(out.cpu().numpy())
        all_targets.append(y.cpu().numpy())

pred_q = np.concatenate(all_preds)  # (N, 24, 99)
target  = np.concatenate(all_targets)  # (N, 24)

# Inverse transform to original power scale
sh = pred_q.shape
pred_q_mw = scaler_y.inverse_transform(pred_q.reshape(-1, sh[2])).reshape(sh)
target_mw = scaler_y.inverse_transform(target.reshape(-1, 1)).reshape(target.shape)
print(f'Predictions: {pred_q_mw.shape}, Target: {target_mw.shape}')
print(f'Target range: [{target_mw.min():.2f}, {target_mw.max():.2f}]')
sys.stdout.flush()

N, H, K = pred_q_mw.shape  # samples, 24h, 99 quantiles

# ═══════════════════════════════════════
# 1. PINBALL LOSS
# ═══════════════════════════════════════
print('\nComputing Pinball Loss...'); sys.stdout.flush()

def pinball(pred_q, target, q):
    """pred_q: (N, H, K), target: (N, H), q: (K,)"""
    error = target[:, :, np.newaxis] - pred_q  # (N, H, K)
    loss = np.maximum(q * error, (q - 1) * error)
    return loss.mean()

pinball_all = pinball(pred_q_mw, target_mw, QUANTILES)

# Per-horizon pinball
pinball_h = []
for h in range(H):
    err = target_mw[:, h, np.newaxis] - pred_q_mw[:, h, :]
    pb_h = np.maximum(QUANTILES * err, (QUANTILES - 1) * err).mean()
    pinball_h.append(pb_h)

# Pinball at key horizons
key_h = [1, 3, 6, 12, 18, 24]

# ═══════════════════════════════════════
# 2. WINKLER SCORE (for 80% CI: q10-q90)
# ═══════════════════════════════════════
print('Computing Winkler Score...'); sys.stdout.flush()

alpha = 0.20  # 80% CI → alpha=0.2
lower_idx = 9   # q10
upper_idx = 89  # q90
delta_w = 2 / alpha  # penalty factor for missed coverage

winkler_values = np.zeros((N, H))
for h in range(H):
    lo = pred_q_mw[:, h, lower_idx]
    hi = pred_q_mw[:, h, upper_idx]
    t  = target_mw[:, h]
    width = hi - lo

    # Winkler score per sample
    in_interval = (t >= lo) & (t <= hi)
    winkler_values[:, h] = width + delta_w * (
        (lo - t) * (t < lo).astype(float) + (t - hi) * (t > hi).astype(float)
    )

winkler_mean = winkler_values.mean()
winkler_h = winkler_values.mean(axis=0)  # per-horizon

# ═══════════════════════════════════════
# 3. CRPS (Continuous Ranked Probability Score)
# ═══════════════════════════════════════
print('Computing CRPS...'); sys.stdout.flush()

# CRPS from quantiles: CRPS = 2 ∫₀¹ (y - q_τ)(1_{y≤q_τ} - τ) dτ
# Approximate via trapezoidal sum over quantiles
crps_values = np.zeros((N, H))
for h in range(H):
    t = target_mw[:, h, np.newaxis]  # (N, 1)
    q = pred_q_mw[:, h, :]            # (N, K)

    # Integrand: (q - t)(1_{t≤q} - τ)
    indicator = (t <= q).astype(float)
    integrand = (q - t) * (indicator - QUANTILES)

    # Trapezoidal integration over quantiles
    dq = np.diff(QUANTILES)
    # Average adjacent integrand values for trapezoidal rule
    integrand_avg = (integrand[:, :-1] + integrand[:, 1:]) / 2
    crps_values[:, h] = 2 * np.sum(integrand_avg * dq, axis=1)

crps_mean = crps_values.mean()
crps_h = crps_values.mean(axis=0)

# ═══════════════════════════════════════
# 4. RELIABILITY (Coverage Calibration)
# ═══════════════════════════════════════
print('Computing Reliability...'); sys.stdout.flush()

# For each nominal coverage level (10%, 20%, ..., 90%),
# check actual coverage percentage
nominal_levels = np.arange(10, 100, 10)  # 10%, 20%, ..., 90%
actual_coverage = []

for level in nominal_levels:
    alpha_ci = 1.0 - level / 100.0  # e.g., 80% CI → alpha=0.20
    lower_idx = int(np.floor(alpha_ci / 2 * 100)) - 1
    upper_idx = int(np.ceil((1 - alpha_ci / 2) * 100)) - 1
    lower_idx = max(0, lower_idx)
    upper_idx = min(K - 1, upper_idx)

    in_ci = (target_mw >= pred_q_mw[:, :, lower_idx]) & (target_mw <= pred_q_mw[:, :, upper_idx])
    actual_coverage.append(in_ci.mean() * 100)

# Reliability deviation (avg absolute deviation from nominal)
reliability_dev = np.mean(np.abs(np.array(actual_coverage) - nominal_levels))

# ═══════════════════════════════════════
# 5. SHARPNESS (Average Interval Width)
# ═══════════════════════════════════════
print('Computing Sharpness...'); sys.stdout.flush()

sharpness_levels = {}
for level, label in [(80, '80% CI'), (50, '50% CI'), (90, '90% CI')]:
    alpha_ci = 1.0 - level / 100.0
    li = max(0, int(np.floor(alpha_ci / 2 * 100)) - 1)
    ui = min(K - 1, int(np.ceil((1 - alpha_ci / 2) * 100)) - 1)
    width = pred_q_mw[:, :, ui] - pred_q_mw[:, :, li]
    sharpness_levels[label] = width.mean()

# ═══════════════════════════════════════
# 6. POINT FORECAST: RMSE, MAE, MAPE (from median q50)
# ═══════════════════════════════════════
print('Computing Point Forecast metrics...'); sys.stdout.flush()

p50 = pred_q_mw[:, :, 49]  # median

# Filter target > 0.001 (meaningful wind power)
mask = target_mw > 0.001
pf = p50[mask]
tf = target_mw[mask]

rmse = np.sqrt(np.mean((pf - tf) ** 2))
mae  = np.mean(np.abs(pf - tf))
mape = np.mean(np.abs((tf - pf) / (tf + 1e-4))) * 100
r2 = 1 - np.sum((tf - pf)**2) / (np.sum((tf - np.mean(tf))**2) + 1e-8)

# Per-horizon RMSE, MAE
rmse_h = [np.sqrt(np.mean((p50[:, h][target_mw[:, h] > 0.001] - target_mw[:, h][target_mw[:, h] > 0.001])**2)) for h in range(H)]
mae_h  = [np.mean(np.abs(p50[:, h][target_mw[:, h] > 0.001] - target_mw[:, h][target_mw[:, h] > 0.001])) for h in range(H)]

# ═══════════════════════════════════════
# PRINT RESULTS
# ═══════════════════════════════════════
print('\n' + '=' * 75)
print('  COMPREHENSIVE EVALUATION — LNMamba v1 on GEFCom2014 Zone 1')
print('  Model: 396K params, LNN + Mamba SSM, ECMWF NWP features')
print('  Horizon: 24h ahead, 99 quantiles (0.01-0.99)')
print('=' * 75)

print('\n' + '-' * 75)
print('  PROBABILISTIC METRICS')
print('-' * 75)

print(f'  {"Metric":<35s} {"Value":<15s}')
print(f'  {"-"*50}')
print(f'  {"Pinball Loss (weighted, 99 quantiles)":<35s} {pinball_all:<15.4f}')
print(f'  {"Winkler Score (80% CI)":<35s} {winkler_mean:<15.4f}')
print(f'  {"CRPS (Continuous Ranked Prob. Score)":<35s} {crps_mean:<15.4f}')

print(f'\n  {"PINBALL per HORIZON":-^50}')
for h in range(H):
    marker = ' <--' if h+1 in key_h else ''
    print(f'  +{h+1:2d}h: {pinball_h[h]:.4f}{marker}')

print(f'\n  {"WINKLER per HORIZON":-^50}')
for h in range(H):
    marker = ' <--' if h+1 in key_h else ''
    print(f'  +{h+1:2d}h: {winkler_h[h]:.4f}{marker}')

print(f'\n  {"CRPS per HORIZON":-^50}')
for h in range(H):
    marker = ' <--' if h+1 in key_h else ''
    print(f'  +{h+1:2d}h: {crps_h[h]:.4f}{marker}')

print('\n' + '-' * 75)
print('  RELIABILITY (Calibration)')
print('-' * 75)
print(f'  {"Nominal Coverage":<20s} {"Actual Coverage":<20s} {"Deviation":<15s}')
print(f'  {"-"*55}')
for nom, act in zip(nominal_levels, actual_coverage):
    dev = act - nom
    bar = '+' * int(abs(dev) / 2) if dev > 0 else '-' * int(abs(dev) / 2)
    print(f'  {nom:>3d}% CI            {act:>5.1f}%              {dev:>+5.1f}%  {bar}')
print(f'  {"Reliability Deviation (avg)":<35s} {reliability_dev:<15.2f}%')

print('\n' + '-' * 75)
print('  SHARPNESS (Interval Width)')
print('-' * 75)
for label, val in sharpness_levels.items():
    print(f'  {label:<20s}: {val:>8.4f}')
# Also per-horizon sharpness for 80% CI
p80_lo = pred_q_mw[:, :, 9]; p80_hi = pred_q_mw[:, :, 89]
width_80 = (p80_hi - p80_lo).mean(axis=0)
print(f'  {"80% CI Avg Width":<25s}: {width_80.mean():.4f} (range: {width_80.min():.3f} - {width_80.max():.3f})')

print('\n' + '-' * 75)
print('  POINT FORECAST (Median q50)')
print('-' * 75)
print(f'  {"Metric":<35s} {"Value":<15s}')
print(f'  {"-"*50}')
print(f'  {"RMSE (Root Mean Square Error)":<35s} {rmse:<15.4f}')
print(f'  {"MAE (Mean Absolute Error)":<35s} {mae:<15.4f}')
print(f'  {"MAPE (Mean Abs. Percentage Error)":<35s} {mape:<15.1f}%')
print(f'  {"R-squared":<35s} {r2:<15.4f}')
print(f'  {"Valid samples (P > 0.001)":<35s} {len(pf):<15,}')

print(f'\n  {"RMSE per HORIZON":-^50}')
for h in range(H):
    marker = ' <--' if h+1 in key_h else ''
    print(f'  +{h+1:2d}h: {rmse_h[h]:.4f}{marker}')

print(f'\n  {"MAE per HORIZON":-^50}')
for h in range(H):
    marker = ' <--' if h+1 in key_h else ''
    print(f'  +{h+1:2d}h: {mae_h[h]:.4f}{marker}')

# ═══════════════════════════════════════
# SUMMARY TABLE FOR PAPER
# ═══════════════════════════════════════
print('\n' + '=' * 75)
print('  PAPER-READY SUMMARY TABLE')
print('=' * 75)

print(f'''
Table 1. Probabilistic forecasting performance on GEFCom2014 Zone 1 test set.

┌──────────────────────────────────────┬──────────┐
│ Metric                               │ Value    │
├──────────────────────────────────────┼──────────┤
│ Pinball Loss (99 quantiles)          │ {pinball_all:.4f}  │
│ Winkler Score (80% CI)               │ {winkler_mean:.4f}  │
│ CRPS                                 │ {crps_mean:.4f}  │
│ 80% CI Coverage (Ideal: 80%)         │ {actual_coverage[7]:.1f}%   │
│ 80% CI Average Width                 │ {width_80.mean():.4f}  │
│ Reliability Deviation (avg)          │ {reliability_dev:.2f}%   │
├──────────────────────────────────────┼──────────┤
│ RMSE (median q50)                    │ {rmse:.4f}  │
│ MAE (median q50)                     │ {mae:.4f}  │
│ MAPE (median q50)                    │ {mape:.1f}%  │
│ R-squared (median q50)               │ {r2:.4f}  │
└──────────────────────────────────────┴──────────┘
''')

print('=' * 75)
print('  Done! All metrics computed successfully.')
print('=' * 75)
