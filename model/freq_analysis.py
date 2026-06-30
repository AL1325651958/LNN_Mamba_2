"""Frequency domain analysis of prediction quality."""
import numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = np.load('checkpoints/v2_results.npz')
pred = d['pred']
target = d['target']

N = 96
freq = np.fft.rfftfreq(N, d=15)  # period in minutes

pred_fft = np.abs(np.fft.rfft(pred, axis=1)).mean(axis=0)
targ_fft = np.abs(np.fft.rfft(target, axis=1)).mean(axis=0)

print("=== Frequency Domain Analysis ===")
print(f"{'Period':>10s}  {'Target':>10s}  {'Pred':>10s}  {'Ratio':>8s}")
for i, (f, pv, tv) in enumerate(zip(freq, pred_fft, targ_fft)):
    period = 1/(f + 1e-8)
    if i < 6 or period in [1440, 480, 240, 120, 60, 30]:
        print(f"{period:8.0f}min  {tv:10.1f}  {pv:10.1f}  {pv/(tv+1e-8):6.2f}x")

# Low vs high frequency
sep = N // 6  # ~4h boundary
lf_targ = targ_fft[:sep].sum()
hf_targ = targ_fft[sep:].sum()
lf_pred = pred_fft[:sep].sum()
hf_pred = pred_fft[sep:].sum()

print(f"\nLow-freq (<4h): Target={lf_targ:.0f}, Pred={lf_pred:.0f}, Ratio={lf_pred/(lf_targ+1e-8):.2f}")
print(f"High-freq (>4h): Target={hf_targ:.0f}, Pred={hf_pred:.0f}, Ratio={hf_pred/(hf_targ+1e-8):.2f}")
print(f"High-freq suppression: {hf_targ/(hf_pred+1e-8):.1f}x")

# Also check per-horizon std
print("\n--- Per-horizon Std Dev ---")
for h in [0, 3, 11, 23, 47, 71, 95]:
    ps = pred[:, h].std()
    ts = target[:, h].std()
    print(f"  +{(h+1)*15:3d}min: Pred std={ps:.1f}, Target std={ts:.1f}, Ratio={ps/(ts+1e-8):.2f}")

# Spectrum plot
os.makedirs('plots', exist_ok=True)
fig, ax = plt.subplots(figsize=(12, 4))
periods = 1 / (freq[1:] + 1e-8)
ax.semilogx(periods, targ_fft[1:], 'b-', lw=1.5, alpha=0.8, label='Ground Truth Spectrum')
ax.semilogx(periods, pred_fft[1:], 'r--', lw=1.5, alpha=0.8, label='LMT v2 Prediction Spectrum')
ax.axvline(240, color='gray', ls=':', alpha=0.5, label='4h cycle boundary')
ax.set_xlabel('Period (minutes)')
ax.set_ylabel('Magnitude')
ax.set_title('Frequency Spectrum: LMT v2 Prediction vs Ground Truth')
ax.legend()
ax.grid(alpha=0.2)
ax.invert_xaxis()
plt.tight_layout()
plt.savefig('plots/v2_spectrum.png', dpi=150)
plt.close()
print("\nSaved: plots/v2_spectrum.png")
