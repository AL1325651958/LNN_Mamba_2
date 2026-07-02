"""Extra figures: Architecture diagram + 12-model SOTA bar chart."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
import numpy as np, os

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial','DejaVu Sans'],
    'font.size': 12, 'axes.titlesize': 15, 'axes.titleweight': 'bold',
    'axes.labelsize': 14, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 11, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)
DARK = '#333333'
BLUE = '#2166AC'

def colored_box(ax, x, y, w, h, text, color, fontsize=16, bold=False):
    rect = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle='round,pad=0.1',
                           facecolor=color, edgecolor=DARK, lw=1.2, alpha=0.9)
    ax.add_patch(rect)
    wt = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight=wt, color=DARK)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=DARK, lw=1.5))


# ═══════════════════ Figure 0: Architecture ═══════════════════
print("Figure 0: Architecture Diagram...")
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim(0, 12); ax.set_ylim(0, 10)
ax.axis('off')

C = {'ssm': '#D6EAF8', 'lnn': '#FDEBD0', 'dec': '#E8F8F5', 'bg': '#F0F4F8', 'emb': '#E8EAF6'}

colored_box(ax, 6, 9.3, 8, 0.7, r'Input: $\mathbf{X} \in \mathbb{R}^{B \times V \times L}$ (168h ECMWF NWP)', C['bg'], 10, True)
arrow(ax, 6, 8.9, 6, 8.5)
colored_box(ax, 6, 8.2, 7, 0.7, r'Embedding: $\mathrm{Linear}(V,2d) \to \mathrm{GELU} \to \mathrm{Linear}(2d,d)$', C['emb'], 9)
colored_box(ax, 6, 7.5, 6, 0.5, '+ Learnable Positional Encoding', C['emb'], 8)
arrow(ax, 6, 7.2, 6, 6.8)

# Block 1
colored_box(ax, 2.5, 6.5, 3.2, 0.5, r'Liquid-Gated Selective SSM', C['ssm'], 8.5)
colored_box(ax, 2.5, 5.9, 3.2, 0.5, r'$\times \ \sigma($ LNN Gate: GRU(48)$)$', C['lnn'], 8)
ax.text(2.5, 6.9, 'Block 1', ha='center', fontsize=15, fontweight='bold')

# Block 2
colored_box(ax, 9.5, 6.5, 3.2, 0.5, r'Liquid-Gated Selective SSM', C['ssm'], 8.5)
colored_box(ax, 9.5, 5.9, 3.2, 0.5, r'$\times \ \sigma($ LNN Gate: GRU(48)$)$', C['lnn'], 8)
ax.text(9.5, 6.9, 'Block 2', ha='center', fontsize=15, fontweight='bold')

arrow(ax, 6, 6.8, 2.5, 6.8)
arrow(ax, 6, 6.8, 9.5, 6.8)
arrow(ax, 2.5, 5.5, 6, 4.5)
arrow(ax, 9.5, 5.5, 6, 4.5)

colored_box(ax, 6, 5.0, 4, 0.4, 'Concatenate', '#FFE0B2', 8)
arrow(ax, 6, 4.7, 6, 4.2)
colored_box(ax, 6, 4.1, 7, 0.5, r'Decoder: $\mathrm{Linear}(d,2d) \to \mathrm{GELU} \to \mathrm{Linear}(2d,d) \to \mathrm{GELU}$', C['dec'], 8.5)
colored_box(ax, 6, 3.5, 5, 0.5, r'$\mathrm{Linear}(d, 24 \times 99)$', C['dec'], 9)
arrow(ax, 6, 3.2, 6, 2.8)
colored_box(ax, 6, 2.4, 8, 0.7, r'Output: $\mathbf{\hat{Q}} \in \mathbb{R}^{B \times 24 \times 99}$ (24h, 99 quantiles)', C['bg'], 10, True)

patches = [mpatches.Patch(color=c, label=l) for c,l in
           [(C['ssm'], 'Selective SSM'), (C['lnn'], 'LNN Gate'), (C['dec'], 'Decoder'), (C['bg'], 'I/O')]]
ax.legend(handles=patches, loc='lower right', fontsize=14, framealpha=0.9, ncol=2)

ax.text(11.8, 9.0, 'd = 64, d_state = 16\n2 blocks, 412K params', fontsize=16, ha='right', va='top',
        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', lw=0.5))

ax.set_title('Figure 1: LNMamba Architecture', fontsize=16, fontweight='bold', pad=15)
plt.tight_layout()
fig.savefig(f'{OUT}/fig0_architecture.png', dpi=300)
fig.savefig(f'{OUT}/fig0_architecture.pdf')
plt.close()
print('  -> fig0_architecture.png/pdf')


# ═══════════════════ Figure: 12-Model SOTA ═══════════════════
print("Figure: 12-Model SOTA Bar Chart...")
models = [
    ('Persistence', 0.3761, '#888888'),
    ('TimeMixer (ICLR 24)', 0.2704, '#D62728'),
    ('Crossformer (ICLR 23)', 0.2600, '#17BECF'),
    ('PatchTST (ICLR 23)', 0.2596, '#BCBD22'),
    ('iTransformer (ICLR 24)', 0.2414, '#7F7F7F'),
    ('TimesNet (ICLR 23)', 0.2313, '#E377C2'),
    ('TSMixer (KDD 23)', 0.2275, '#8C564B'),
    ('DLinear (AAAI 23)', 0.2227, '#9467BD'),
    ('TiDE (2024)', 0.2184, '#2CA02C'),
    ('ModernTCN (ICLR 24)', 0.2165, '#4DAF4A'),
    ('GRU', 0.2161, '#1f77b4'),
    ('LNMamba (ours)', 0.2069, '#2166AC'),
]
models.reverse()

fig, ax = plt.subplots(figsize=(10, 6))
names = [m[0] for m in models]
vals  = [m[1] for m in models]
colors = [m[2] for m in models]

bars = ax.barh(range(len(models)), vals, color=colors, height=0.65, edgecolor='white', lw=0.5)
ax.barh(len(models)-1, vals[-1], color='#2166AC', height=0.65, edgecolor='#1a5c8a', lw=2.5)

for i, (bar, val) in enumerate(zip(bars, vals)):
    ax.text(val + 0.002, i, f'{val:.4f}', va='center', fontsize=14, fontfamily='monospace')

ax.annotate('BEST', xy=(vals[-1], len(models)-1), xytext=(vals[-1]+0.025, len(models)-1.2),
            arrowprops=dict(arrowstyle='->', color=BLUE, lw=2.5),
            fontsize=14, fontweight='bold', color=BLUE)

ax.set_xlabel('Pinball Loss (lower = better)', fontsize=16)
ax.set_yticks(range(len(models)))
ax.set_yticklabels(names, fontsize=14)
ax.set_title('Figure 2: SOTA Comparison — 12 Models on GEFCom2014 Zone 1\n(168h input, 24h output, 99 quantiles)',
             fontsize=15, fontweight='bold')
ax.set_xlim(0.18, 0.40)
ax.grid(True, alpha=0.15, lw=0.3, axis='x')

plt.tight_layout()
fig.savefig(f'{OUT}/fig_sota_12model.png', dpi=300)
fig.savefig(f'{OUT}/fig_sota_12model.pdf')
plt.close()
print('  -> fig_sota_12model.png/pdf')

print('\nAll extra figures done!')
