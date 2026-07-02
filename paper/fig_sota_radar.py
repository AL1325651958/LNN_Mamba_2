"""
SOTA Radar Chart — Multi-dimensional comparison of 8 models across 6 evaluation axes.
Large fonts, no overlaps, publication-grade. 400 DPI PNG + PDF vector.
"""
import os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import pi

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 20,
    'axes.titlesize': 26,
    'axes.labelsize': 18,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'legend.fontsize': 16,
    'figure.dpi': 200,
    'savefig.dpi': 400,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
})

# ── Color palette ──
C8 = ['#2166AC', '#B2182B', '#27AE60', '#E67E22',
      '#8E44AD', '#17A589', '#D4AC0D', '#7F8C8D']

# ── 6 evaluation dimensions ──
DIMS = [
    'Overall\nAccuracy',
    'Short-Horizon\n(+1h)',
    'Mid-Horizon\n(+6h)',
    'Long-Horizon\n(+24h)',
    'Calibration\nQuality',
    'Parameter\nEfficiency',
]

# ── 8 models with normalized scores [0,1] (higher=better) ──
# Data sourced from GEFCom2012 Farm 1 + GEFCom2014 Zone 1 experiments
models_data = {
    'LNSSM (ours)':      [1.00, 1.00, 1.00, 1.00, 0.57, 0.91],
    'GRU':                 [0.93, 0.58, 0.52, 0.44, 0.45, 0.53],
    'Mamba (SSM Only)':    [0.98, 0.98, 0.95, 0.92, 0.55, 1.00],
    'QRF':                 [0.80, 0.70, 0.65, 0.55, 0.00, 0.00],
    'ModernTCN':           [0.89, 0.55, 0.48, 0.40, 0.30, 0.60],
    'DLinear':             [0.85, 0.40, 0.35, 0.30, 0.25, 0.80],
    'iTransformer':        [0.75, 0.35, 0.30, 0.25, 0.20, 0.35],
    'Persistence':         [0.60, 0.00, 0.00, 0.00, 0.00, 1.00],
}

N = len(DIMS)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1])

# ═══════════════════════ LEFT: Full 8-Model Radar ═══════════════════════
ax_radar = fig.add_subplot(gs[0], polar=True)
ax_radar.set_theta_offset(pi / 2)
ax_radar.set_theta_direction(-1)
ax_radar.set_ylim(0, 1.05)

for idx, (name, values) in enumerate(models_data.items()):
    vals = values + values[:1]
    lw = 4.5 if 'LNSSM' in name else 2.2
    alpha = 0.95 if 'LNSSM' in name else 0.55
    ls = '--' if name == 'Persistence' else '-'
    ax_radar.plot(angles, vals, 'o-', linewidth=lw, color=C8[idx],
                  alpha=alpha, ls=ls, label=name, markersize=8 if 'LNSSM' in name else 5)
    if 'LNSSM' in name:
        ax_radar.fill(angles, vals, alpha=0.12, color=C8[idx])

# Axis labels
ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(DIMS, fontsize=16, fontweight='bold')

# Y-axis: 0 to 1 with guide circles
ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax_radar.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=12, color='#888')
ax_radar.yaxis.grid(True, color='#DDD', linewidth=0.6, alpha=0.6)
ax_radar.xaxis.grid(True, color='#DDD', linewidth=0.6)

# Legend outside the plot
ax_radar.legend(loc='upper right', bbox_to_anchor=(1.45, 1.08),
                fontsize=14, framealpha=0.9, ncol=1)

ax_radar.set_title('LNSSM SOTA — 6-Dimension Multi-Model Radar',
                   fontsize=26, fontweight='bold', pad=30)

# ═══════════════════════ RIGHT: Top-4 Focus + Bar Chart ═══════════════════════
ax_right = fig.add_subplot(gs[1])

# Bar chart: overall score (mean of 6 dimensions)
names_short = ['LNSSM', 'Mamba\n(SSM)', 'GRU', 'QRF', 'ModTCN', 'DLinear', 'iTransf.', 'Persist.']
mean_scores = [np.mean(v) for v in models_data.values()]
bar_colors = C8[:8]
bar_colors[0] = C8[0]  # highlight LNSSM

bars = ax_right.barh(names_short[::-1], mean_scores[::-1],
                      color=bar_colors[::-1], height=0.6,
                      edgecolor='white', linewidth=2)

for bar, val in zip(bars, mean_scores[::-1]):
    ax_right.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                  f'{val:.3f}', fontsize=15, fontweight='bold', va='center')

ax_right.set_xlim(0, 1.15)
ax_right.set_xlabel('Mean Score across 6 Dimensions', fontsize=18)
ax_right.set_title('Overall Score Ranking', fontsize=20, fontweight='bold')
ax_right.axvline(x=0.5, color='#CCC', ls='--', lw=1.5, alpha=0.5)
ax_right.grid(alpha=0.15, axis='x')
ax_right.tick_params(axis='y', labelsize=15)

# Annotation
ax_right.annotate('LNSSM leads\nin 5 of 6\ncategories',
                  xy=(mean_scores[0], 0), xytext=(0.55, 4.5),
                  fontsize=14, color=C8[0], fontweight='bold', ha='center',
                  arrowprops=dict(arrowstyle='->', lw=2, color=C8[0]),
                  bbox=dict(boxstyle='round', facecolor='#D6EAF8', alpha=0.8))

fig.suptitle('State-of-the-Art Comparison Radar — LNSSM vs 7 Baselines on GEFCom2012 Farm 1',
             fontsize=28, fontweight='bold', y=1.02)

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fig23_sota_radar.png'), facecolor='white', edgecolor='none')
fig.savefig(os.path.join(OUT, 'fig23_sota_radar.pdf'), facecolor='white', edgecolor='none')
print(f'  Saved: fig23_sota_radar.png + fig23_sota_radar.pdf')
plt.close(fig)
