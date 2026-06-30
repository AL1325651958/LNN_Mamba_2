"""NWP model Raw vs Pred visualization — Zone 1, GEFCom2014."""
import sys,os,zipfile,io,numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.nwp_model import NWPMamba, load_zone_nwp, make_weather_features, FEAT_COLS, QUANTILES, pinball_loss

os.makedirs('plots', exist_ok=True)

# ── Load Zone 1 data ──
print('Loading Zone 1...')
(dl_train, dl_val, dl_test), n_vars, scaler_y, info = load_zone_nwp(1, seq=168, pred=24, batch=64, stride=6)
print(info)

# ── Load trained model (need to train quick if no checkpoint)
device = torch.device('cuda')
model = NWPMamba(n_vars, d=64, nb=2, ds=16, pred=24, nq=99, use_lnn=True).to(device)
n_p = sum(p.numel() for p in model.parameters())

# Quick train (5 epochs just for viz)
from model.nwp_model import train_one_zone
print(f'\nTraining {n_p:,} params, 5 epochs for viz...')
train_one_zone(model, dl_train, dl_val, device, epochs=5, lr=1e-3)

# ── Generate predictions ──
print('Generating test predictions...')
model.eval()
q_t = torch.tensor(QUANTILES, dtype=torch.float32, device=device)
all_preds, all_targs = [], []
total_pb = 0.0
with torch.no_grad():
    for x, y in dl_test:
        x, y = x.to(device), y.to(device)
        out = model(x)  # (B, 24, 99)
        total_pb += pinball_loss(out, y, q_t).item()
        all_preds.append(out.cpu().numpy())
        all_targs.append(y.cpu().numpy())

pr = np.concatenate(all_preds)  # (N, 24, 99)
tr = np.concatenate(all_targs)  # (N, 24)

# Inverse transform
sh = pr.shape
pr_mw = scaler_y.inverse_transform(pr.reshape(-1, sh[2])).reshape(sh)
tr_mw = scaler_y.inverse_transform(tr.reshape(-1, 1)).reshape(tr.shape)

print(f'Predictions: {pr_mw.shape}, Target: {tr_mw.shape}')
print(f'Pinball: {total_pb/len(dl_test):.4f}')

# ── Median prediction ──
p50 = pr_mw[:, :, 49]   # median (50th percentile)
p10 = pr_mw[:, :, 9]    # 10th
p90 = pr_mw[:, :, 89]   # 90th
p25 = pr_mw[:, :, 24]
p75 = pr_mw[:, :, 74]

# R² for median
pf = p50.ravel(); tf = tr_mw.ravel()
mask = tf > 0.001; pf_f = pf[mask]; tf_f = tf[mask]
rmse = np.sqrt(np.mean((pf_f - tf_f)**2))
mae  = np.mean(np.abs(pf_f - tf_f))
r2   = 1 - np.sum((tf_f - pf_f)**2)/(np.sum((tf_f - np.mean(tf_f))**2) + 1e-8)
print(f'Median: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.3f}')

# ═══════════════════ Plot 1: Timeseries Multi-Horizon ═══════════════════
print('\nPlot 1: Timeseries...')
n_show = min(96, len(tr_mw) - 2)  # adaptive
start = max(0, len(tr_mw) // 3 - n_show // 2)

fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
horizons = [0, 11, 23]  # +1h, +12h, +24h
labels   = ['+1h', '+12h', '+24h']

for ax, h, lab in zip(axes, horizons, labels):
    t = tr_mw[start:start+n_show, h]
    p = p50[start:start+n_show, h]
    lo = p10[start:start+n_show, h]
    hi = p90[start:start+n_show, h]

    r_h = np.sqrt(np.mean((p - t)**2))
    ax.fill_between(range(n_show), lo, hi, alpha=0.15, color='steelblue', label='80% CI')
    ax.plot(t, 'b-', lw=1.2, alpha=0.85, label='Actual')
    ax.plot(p, 'r-', lw=1.2, alpha=0.85, label=f'LNN-Mamba (RMSE={r_h:.3f})')
    ax.set_ylabel('Power (norm)')
    ax.set_title(f'Horizon {lab}  |  Zone 1  |  ECMWF NWP Weather Input')
    ax.legend(loc='upper right', fontsize=9, ncol=3)
    ax.grid(alpha=0.2)
    ax.set_ylim(0, None)

plt.tight_layout()
plt.savefig('plots/nwp_timeseries_multihorizon.png', dpi=150)
plt.close()
print('  -> plots/nwp_timeseries_multihorizon.png')

# ═══════════════════ Plot 2: Single horizon with intervals ═══════════════════
print('Plot 2: Prediction intervals...')
n_show = min(48, len(tr_mw) - 2)
start = max(0, len(tr_mw) // 2 - n_show // 2)

t = tr_mw[start:start+n_show, 5]     # +6h
p_m = p50[start:start+n_show, 5]
p_l = p10[start:start+n_show, 5]
p_h = p90[start:start+n_show, 5]
p_ll = pr_mw[start:start+n_show, 5, 0]   # 1%
p_hh = pr_mw[start:start+n_show, 5, 98]  # 99%

fig, ax = plt.subplots(figsize=(16, 5))
ax.fill_between(range(n_show), p_ll, p_hh, alpha=0.08, color='navy', label='1-99% interval')
ax.fill_between(range(n_show), p_l, p_h, alpha=0.18, color='steelblue', label='10-90% interval')
ax.fill_between(range(n_show), p25[start:start+n_show,5], p75[start:start+n_show,5],
                alpha=0.25, color='steelblue', label='25-75% interval')
ax.plot(t, 'b-', lw=1.5, alpha=0.9, label='Actual Power', zorder=10)
ax.plot(p_m, 'r-', lw=1.5, alpha=0.9, label='LNN-Mamba Median', zorder=9)

# Count actuals within 90% CI
in_ci = np.sum((t >= p_l) & (t <= p_h))
ax.set_title(f'LNN-Mamba Probabilistic Forecast (+6h) | Zone 1 | '
             f'{in_ci}/{n_show} in 80% CI ({in_ci/n_show*100:.0f}%) | '
             f'RMSE={np.sqrt(np.mean((p_m-t)**2)):.3f}')
ax.set_xlabel('Time step (hours)'); ax.set_ylabel('Normalized Power')
ax.legend(loc='upper right', ncol=2, fontsize=9); ax.grid(alpha=0.2)
plt.tight_layout(); plt.savefig('plots/nwp_prediction_intervals.png', dpi=150); plt.close()
print('  -> plots/nwp_prediction_intervals.png')

# ═══════════════════ Plot 3: Scatter (Median vs Actual, all horizons) ═══════════════════
print('Plot 3: Scatter...')
pf_all = p50.ravel(); tf_all = tr_mw.ravel()
fig, ax = plt.subplots(figsize=(7, 7))
hb = ax.hexbin(tf_all, pf_all, gridsize=40, cmap='Blues', mincnt=1, alpha=0.85)
mx = max(tf_all.max(), pf_all.max())
ax.plot([0, mx], [0, mx], 'k--', lw=1, alpha=0.5, label='Perfect')
ax.text(0.05, 0.95, f'LNN-Mamba (Median)\nRMSE={rmse:.3f}\nMAE={mae:.3f}\nR2={r2:.3f}',
        transform=ax.transAxes, fontsize=11, va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85), fontfamily='monospace')
ax.set_xlabel('Actual Power'); ax.set_ylabel('Predicted Power')
ax.set_title(f'GEFCom2014 Zone 1 | ECMWF NWP | Median Prediction (All Horizons)')
ax.set_aspect('equal'); plt.colorbar(hb, ax=ax, label='Density')
plt.tight_layout(); plt.savefig('plots/nwp_scatter.png', dpi=150); plt.close()
print('  -> plots/nwp_scatter.png')

# ═══════════════════ Plot 4: Horizon Error + Coverage ═══════════════════
print('Plot 4: Horizon metrics...')
rmse_h = [np.sqrt(np.mean((p50[:,h] - tr_mw[:,h])**2)) for h in range(24)]
mae_h  = [np.mean(np.abs(p50[:,h] - tr_mw[:,h])) for h in range(24)]
# Coverage: % of actual values within 50% PI
cov50_h = []
for h in range(24):
    in_range = np.sum((tr_mw[:,h] >= p25[:,h]) & (tr_mw[:,h] <= p75[:,h]))
    cov50_h.append(in_range / len(tr_mw) * 100)

fig, ax1 = plt.subplots(figsize=(10, 5))
hh = np.arange(1, 25)
ax1.plot(hh, rmse_h, 'r-o', lw=2, ms=6, label='RMSE')
ax1.plot(hh, mae_h, 'b-s', lw=2, ms=6, label='MAE')
ax1.set_xlabel('Horizon (hours)'); ax1.set_ylabel('Error', color='r')
ax2 = ax1.twinx()
ax2.plot(hh, cov50_h, 'g-^', lw=2, ms=6, label='50% PI Coverage')
ax2.axhline(50, color='gray', ls='--', alpha=0.5, label='Ideal 50%')
ax2.set_ylabel('Coverage (%)', color='g')
l1, p1 = ax1.get_legend_handles_labels()
l2, p2 = ax2.get_legend_handles_labels()
ax1.legend(l1+l2, p1+p2, loc='center right', fontsize=10)
ax1.set_title(f'Error & Coverage vs Horizon | Zone 1 | ECMWF NWP')
ax1.grid(alpha=0.2); ax1.set_xticks(hh)
plt.tight_layout(); plt.savefig('plots/nwp_horizon_metrics.png', dpi=150); plt.close()
print('  -> plots/nwp_horizon_metrics.png')

# ═══════════════════ Plot 5: Error distribution ═══════════════════
print('Plot 5: Error distribution...')
errors = pf_f - tf_f
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(errors, bins=80, color='steelblue', edgecolor='white', alpha=0.85, density=True)
ax.axvline(0, color='k', ls='--', alpha=0.3)
ax.axvline(np.mean(errors), color='r', ls=':', alpha=0.6, label=f'Mean={np.mean(errors):.4f}')
ax.set_xlabel('Prediction Error'); ax.set_ylabel('Density')
ax.set_title(f'Error Distribution | mu={np.mean(errors):.4f} sigma={np.std(errors):.4f} | Zone 1')
ax.legend(); ax.grid(alpha=0.2)
plt.tight_layout(); plt.savefig('plots/nwp_error_dist.png', dpi=150); plt.close()
print('  -> plots/nwp_error_dist.png')

print('\nAll 5 plots saved to plots/')
