"""V3 visualization: timeseries, scatter, spectrum, horizon, error dist."""
import numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = np.load('checkpoints/v3_results.npz')
pred = d['pred']    # (N, 96)
target = d['target']

ph = pred.ravel(); th = target.ravel()
mask = th > 0
pf, tf = ph[mask], th[mask]
rmse = np.sqrt(np.mean((pf-tf)**2))
mae = np.mean(np.abs(pf-tf))
print(f'Total: RMSE={rmse:.4f}, MAE={mae:.4f}, N={len(pf):,}')

os.makedirs('plots', exist_ok=True)

# ── 1. Timeseries: 3 horizons side by side ──
fig, axes = plt.subplots(3, 1, figsize=(16, 10))
n_show = 384
start = len(pred) // 3
horizons = [0, 11, 47]  # +15min, +3h, +12h
labels = ['+15min', '+3h', '+12h']

for ax, h, lab in zip(axes, horizons, labels):
    t = target[start:start+n_show, h]
    p = pred[start:start+n_show, h]
    r = np.sqrt(np.mean((p-t)**2))
    ax.plot(t, 'b-', lw=1, alpha=0.8, label='Ground Truth')
    ax.plot(p, 'r--', lw=1, alpha=0.8, label='LMT v3 Prediction')
    ax.fill_between(range(n_show), t, p, alpha=0.1, color='gray')
    ax.set_ylabel('Power (std)')
    ax.set_title(f'Horizon {lab}  |  RMSE={r:.3f}  |  6 Wind Farms  |  Spectral Loss')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig('plots/v3_timeseries.png', dpi=150)
plt.close()
print(' -> plots/v3_timeseries.png')

# ── 2. Scatter: all horizons ──
fig, ax = plt.subplots(figsize=(7, 7))
idx = np.random.choice(len(pf), min(5000, len(pf)), replace=False) if len(pf) > 5000 else np.arange(len(pf))
ax.hexbin(tf[idx], pf[idx], gridsize=40, cmap='Blues', mincnt=1, alpha=0.85)
mx = max(tf.max(), pf.max())
ax.plot([0, mx], [0, mx], 'k--', lw=1, alpha=0.5, label='Perfect')
r2 = 1 - np.sum((pf-tf)**2)/(np.sum((tf-np.mean(tf))**2)+1e-8)
ax.text(0.05, 0.95, f'RMSE={rmse:.3f}\nMAE={mae:.3f}\nR^2={r2:.3f}\nN={len(pf):,}',
        transform=ax.transAxes, fontsize=11, va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85), fontfamily='monospace')
ax.set_xlabel('Ground Truth'); ax.set_ylabel('Predicted')
ax.set_title('LMT v3: All-horizon Scatter | 6 Farms + Spectral Loss')
ax.set_aspect('equal'); ax.legend()
plt.tight_layout(); plt.savefig('plots/v3_scatter.png', dpi=150); plt.close()
print(' -> plots/v3_scatter.png')

# ── 3. Spectrum comparison ──
N = 96; freq = np.fft.rfftfreq(N, d=15)
pred_fft = np.abs(np.fft.rfft(pred, axis=1)).mean(axis=0)
targ_fft = np.abs(np.fft.rfft(target, axis=1)).mean(axis=0)

fig, ax = plt.subplots(figsize=(12, 4))
periods = 1/(freq[1:]+1e-8)
ax.semilogx(periods, targ_fft[1:], 'b-', lw=1.5, alpha=0.8, label='Ground Truth')
ax.semilogx(periods, pred_fft[1:], 'r--', lw=1.5, alpha=0.8, label='LMT v3 (Spectral Loss)')
ax.axvline(240, color='gray', ls=':', alpha=0.5, label='4h boundary')
ax.set_xlabel('Period (min)'); ax.set_ylabel('Magnitude')
ax.set_title('Frequency Spectrum: v3 with Spectral Consistency Loss')
ax.legend(); ax.grid(alpha=0.2); ax.invert_xaxis()

# Annotate key periods
for p in [1440, 480, 240, 120, 60]:
    ax.axvline(p, color='orange', ls=':', alpha=0.15)

plt.tight_layout(); plt.savefig('plots/v3_spectrum.png', dpi=150); plt.close()
print(' -> plots/v3_spectrum.png')

# ── 4. Horizon error ──
rmse_h = [np.sqrt(np.mean((pred[:,h]-target[:,h])**2)) for h in range(96)]
mae_h  = [np.mean(np.abs(pred[:,h]-target[:,h])) for h in range(96)]
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot([(h+1)*15 for h in range(96)], rmse_h, 'r-', lw=2, label='RMSE')
ax1.set_xlabel('Horizon (min)'); ax1.set_ylabel('RMSE', color='r')
ax1.set_ylim(bottom=0)
ax2 = ax1.twinx()
ax2.plot([(h+1)*15 for h in range(96)], mae_h, 'b-', lw=2, label='MAE')
ax2.set_ylabel('MAE', color='b'); ax2.set_ylim(bottom=0)
ax1.set_title(f'Error vs Horizon | v3: flat curve = spectral loss working!')
l1,p1=ax1.get_legend_handles_labels(); l2,p2=ax2.get_legend_handles_labels()
ax1.legend(l1+l2,p1+p2, loc='upper right'); ax1.grid(alpha=0.2)
plt.tight_layout(); plt.savefig('plots/v3_horizon.png', dpi=150); plt.close()
print(' -> plots/v3_horizon.png')

# ── 5. Error distribution ──
errors = pf - tf
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(errors, bins=60, color='steelblue', edgecolor='white', alpha=0.85, density=True)
ax.axvline(0, color='k', ls='--', alpha=0.3)
ax.set_xlabel('Error (std units)'); ax.set_ylabel('Density')
ax.set_title(f'Error Distribution | mu={np.mean(errors):.3f} sigma={np.std(errors):.3f}')
ax.grid(alpha=0.2)
plt.tight_layout(); plt.savefig('plots/v3_errors.png', dpi=150); plt.close()
print(' -> plots/v3_errors.png')

# ── 6. Long-horizon variance preservation ──
stds_pred  = [pred[:,h].std() for h in range(96)]
stds_targ  = [target[:,h].std() for h in range(96)]
ratios = [p/(t+1e-8) for p, t in zip(stds_pred, stds_targ)]
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot([(h+1)*15 for h in range(96)], ratios, 'g-', lw=2)
ax.axhline(1.0, color='k', ls='--', alpha=0.5, label='Perfect preservation')
ax.set_xlabel('Horizon (min)'); ax.set_ylabel('Pred/True Std Ratio')
ax.set_title(f'Variance Preservation vs Horizon | Mean ratio={np.mean(ratios):.2f}')
ax.legend(); ax.grid(alpha=0.2); ax.set_ylim(0, None)
plt.tight_layout(); plt.savefig('plots/v3_variance.png', dpi=150); plt.close()
print(' -> plots/v3_variance.png')

print('\nAll v3 plots saved to plots/')
