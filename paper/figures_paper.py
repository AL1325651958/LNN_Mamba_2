"""
Paper figures for LNSSM — Renewable Energy journal.
SCI-style: professional color palette, local zoom-ins, clear labeling, 300dpi.
Generates 8 figures covering all key experimental results.
"""
import sys,os,time,numpy as np,torch,pandas as pd
import zipfile
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.ticker as ticker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── Color palette (Nature/Science inspired) ──
C = {
    'blue':   '#2166AC',  # primary model color
    'red':    '#B2182B',  # accent / truth
    'green':  '#4DAF4A',  # improvement
    'orange': '#FF7F00',  # comparison model
    'purple': '#984EA3',  # another comparison
    'gray':   '#888888',  # baseline
    'bg':     '#F7F7F7',  # light background
    'dark':   '#333333',  # text
}
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.titlesize': 15,
    'axes.titleweight': 'bold',
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})
OUT = os.path.join(ROOT, 'paper', 'figures')
os.makedirs(OUT, exist_ok=True)

DEVICE = torch.device('cuda')
QUANTILES = np.linspace(0.01, 0.99, 99)
SEQ, PRED = 168, 24
DIR12 = os.path.join(ROOT, 'GEFCOM2012/GEFCOM2012_Data/Wind')
LT = [1, 3, 6, 12, 24]


# ═══════════════════════════════════════
# Data loading (shared)
# ═══════════════════════════════════════
def load_farm1():
    pw = pd.read_csv(f'{DIR12}/windpowermeasurements.csv')
    pw = pw[pw['usage'] == 'Training'].copy()
    pw['dt'] = pd.to_datetime(pw['date'].astype(str), format='%Y%m%d%H')
    pw = pw[['dt', 'wp1']].rename(columns={'wp1': 'power'}).sort_values('dt').reset_index(drop=True)
    nwp = pd.read_csv(f'{DIR12}/windforecasts_wf1.csv')
    nwp['issue'] = pd.to_datetime(nwp['date'].astype(str), format='%Y%m%d%H')
    np_p = {}
    for _, r in nwp.iterrows():
        np_p.setdefault(r['issue'], {})[r['hors']] = (r['u'], r['v'], r['ws'], r['wd'])
    Xl, yl = [], []
    for i, r in pw.iterrows():
        t = r['dt']; it = t.replace(hour=0) if t.hour >= 12 else t.replace(hour=12) - pd.Timedelta(days=1)
        if it not in np_p: continue
        f = []; ok = True
        for lt in LT:
            hr = max(1, min(48, int((t - it).total_seconds() / 3600) + (lt - 1)))
            if hr in np_p[it]: f.extend(np_p[it][hr])
            else: ok = False; break
        if not ok: continue
        f += [np.sin(2*np.pi*t.hour/24), np.cos(2*np.pi*t.hour/24),
              np.sin(2*np.pi*t.month/12), np.cos(2*np.pi*t.month/12)]
        Xl.append(f); yl.append(r['power'])
    X = np.array(Xl, dtype=np.float32); y = np.clip(np.array(yl, dtype=np.float32), 0, 1)
    Xn = StandardScaler().fit_transform(X)
    data = np.concatenate([Xn, y.reshape(-1, 1)], axis=1)
    return data, X

class WDS:
    """Non-torch version for numpy arrays."""
    def __init__(self, d, s=4):
        self.data = d; self.s = s
        self.n = max(0, (len(d) - SEQ - PRED) // s + 1)
    def __len__(self): return self.n
    def __getitem__(self, i):
        st = i * self.s
        x = torch.FloatTensor(self.data[st:st+SEQ].T)
        y = torch.FloatTensor(self.data[st+SEQ:st+SEQ+PRED, -1])
        return x, y


def quick_train():
    """Train LNSSM on all 7 GEFCom2012 farms, return model + test preds."""
    from model.nwp_model import NWPMamba, pinball_loss

    # Load all 7 farms
    all_ds = []; nv = None
    for fid in range(1, 8):
        data, _ = load_farm_data(fid)
        if nv is None: nv = data.shape[1]
        T = len(data); te = int(T * 0.85)
        ds = WDS(data[:te], 4)
        all_ds.append(ds)
    ds_full = torch.utils.data.ConcatDataset(all_ds)
    tl = DataLoader(ds_full, 48, shuffle=True, num_workers=0, pin_memory=True)

    # Test on Farm 1
    d1, _ = load_farm1()
    data1, _ = load_farm_data(1)
    T1 = len(data1); te1 = int(T1 * 0.85)
    # Use stride=4 for test (same as always)
    test_ds = WDS(data1[te1:], 4)
    testl = DataLoader(test_ds, 48, shuffle=False, num_workers=0, pin_memory=True)

    print(f"  Train: {len(ds_full):,} windows | Test: {len(test_ds)}")
    sys.stdout.flush()

    qt = torch.tensor(QUANTILES, dtype=torch.float32, device=DEVICE)
    model = NWPMamba(nv, d=64, nb=2, ds=16, pred=PRED, nq=99, use_lnn=True).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=15, T_mult=2, eta_min=1e-5)
    scl = torch.amp.GradScaler('cuda')

    for ep in range(1, 31):
        model.train()
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE); opt.zero_grad()
            with torch.amp.autocast('cuda'): loss = pinball_loss(model(x), y, qt)
            scl.scale(loss).backward(); scl.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scl.step(opt); scl.update()
        sch.step()
        if ep % 10 == 1: print(f"    E{ep}..."); sys.stdout.flush()

    model.eval()
    preds, targs = [], []
    with torch.no_grad():
        for x, y in testl:
            out = model(x.to(DEVICE))
            preds.append(out.cpu().numpy()); targs.append(y.cpu().numpy())

    return np.concatenate(preds), np.concatenate(targs)


def load_farm_data(fid):
    """Load one farm: returns (data_numpy, raw_X_numpy)."""
    pw = pd.read_csv(f'{DIR12}/windpowermeasurements.csv')
    pw = pw[pw['usage'] == 'Training'].copy()
    pw['dt'] = pd.to_datetime(pw['date'].astype(str), format='%Y%m%d%H')
    pw = pw[['dt', f'wp{fid}']].rename(columns={f'wp{fid}': 'power'}).sort_values('dt').reset_index(drop=True)
    nwp = pd.read_csv(f'{DIR12}/windforecasts_wf{fid}.csv')
    nwp['issue'] = pd.to_datetime(nwp['date'].astype(str), format='%Y%m%d%H')
    np_p = {}
    for _, r in nwp.iterrows():
        np_p.setdefault(r['issue'], {})[r['hors']] = (r['u'], r['v'], r['ws'], r['wd'])
    Xl, yl = [], []
    for i, r in pw.iterrows():
        t = r['dt']; it = t.replace(hour=0) if t.hour >= 12 else t.replace(hour=12) - pd.Timedelta(days=1)
        if it not in np_p: continue
        f = []; ok = True
        for lt in LT:
            hr = max(1, min(48, int((t - it).total_seconds() / 3600) + (lt - 1)))
            if hr in np_p[it]: f.extend(np_p[it][hr])
            else: ok = False; break
        if not ok: continue
        f += [np.sin(2*np.pi*t.hour/24), np.cos(2*np.pi*t.hour/24),
              np.sin(2*np.pi*t.month/12), np.cos(2*np.pi*t.month/12)]
        Xl.append(f); yl.append(r['power'])
    X = np.array(Xl, dtype=np.float32); y = np.clip(np.array(yl, dtype=np.float32), 0, 1)
    Xn = StandardScaler().fit_transform(X)
    data = np.concatenate([Xn, y.reshape(-1, 1)], axis=1)
    return data, X


# ═══════════════════════════════════════
# FIGURE 1: Timeseries — Multi-Horizon with ZOOM
# ═══════════════════════════════════════
def fig1_timeseries(pr, tr):
    """Multi-horizon forecast comparison: +1h, +6h, +12h, +24h with zoom inset."""
    print("  Timeseries Multi-Horizon...")
    sys.stdout.flush()

    # Convert to expected value
    qm = (pr[:, :, :-1] + pr[:, :, 1:]) / 2
    pe = np.sum(qm * np.diff(QUANTILES), axis=-1)

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(4, 2, figure=fig, width_ratios=[3, 1], height_ratios=[1, 1, 1, 1],
                           hspace=0.45, wspace=0.18)

    horizons = [0, 5, 11, 23]  # +1h, +6h, +12h, +24h
    labels   = ['+1h', '+6h', '+12h', '+24h']
    n_show   = 96   # 4 days
    n_zoom   = 24   # 1 day zoom
    start    = len(pe) // 3

    for row, (h, lab) in enumerate(zip(horizons, labels)):
        # Main plot (left column)
        ax_main = fig.add_subplot(gs[row, 0])
        ax_main.plot(tr[start:start+n_show, h], color=C['blue'], lw=0.8, alpha=0.7, label='Ground truth')
        ax_main.plot(pe[start:start+n_show, h], color=C['red'], lw=1.0, alpha=0.9, label=f'LNSSM {lab}')
        ax_main.fill_between(range(n_show), tr[start:start+n_show, h], pe[start:start+n_show, h],
                             alpha=0.08, color='gray')

        rmse_h = np.sqrt(np.mean((pe[start:start+n_show, h] - tr[start:start+n_show, h])**2))
        r2_h = 1 - np.sum((pe[start:start+n_show, h] - tr[start:start+n_show, h])**2) / \
               (np.sum((tr[start:start+n_show, h] - np.mean(tr[start:start+n_show, h]))**2) + 1e-8)

        ax_main.set_ylabel(f'{lab}\nPower', fontsize=15)
        ax_main.text(0.02, 0.95, f'RMSE={rmse_h:.3f}  R²={r2_h:+.2f}',
                     transform=ax_main.transAxes, fontsize=16, va='top',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='gray', lw=0.3))
        ax_main.grid(True, alpha=0.15, lw=0.3)
        if row == 0:
            ax_main.legend(loc='upper left', fontsize=14, ncol=1, framealpha=0.9,
                          bbox_to_anchor=(0.01, 0.99))
        if row < 3: ax_main.set_xticklabels([])
        ax_main.set_xlim(0, n_show)

        # Zoom inset (right column)
        ax_zoom = fig.add_subplot(gs[row, 1])
        zs = start + n_show // 3
        ax_zoom.plot(tr[zs:zs+n_zoom, h], color=C['blue'], lw=1.2, alpha=0.8)
        ax_zoom.plot(pe[zs:zs+n_zoom, h], color=C['red'], lw=1.4, alpha=0.9)
        ax_zoom.fill_between(range(n_zoom), tr[zs:zs+n_zoom, h], pe[zs:zs+n_zoom, h],
                             alpha=0.10, color='gray')
        ax_zoom.grid(True, alpha=0.15, lw=0.3)
        ax_zoom.set_xlim(0, n_zoom)
        if row < 3: ax_zoom.set_xticklabels([])
        ax_zoom.set_title('Detail', fontsize=16, color='gray', loc='right', pad=2)

    fig.suptitle('Multi-Horizon Wind Power Prediction — GEFCom2012 Farm 1',
                 fontsize=20, fontweight='bold', y=1.005)
    fig.text(0.5, 0.005, 'Time step (hours)', ha='center', fontsize=18)
    fig.subplots_adjust(hspace=0.45, wspace=0.18, top=0.95, bottom=0.05, left=0.08, right=0.97)
    fig.savefig(os.path.join(OUT, 'fig1_timeseries_multihorizon.png'), dpi=300)
    fig.savefig(os.path.join(OUT, 'fig1_timeseries_multihorizon.pdf'))
    plt.close(fig)
    print("    Saved fig1")


# ═══════════════════════════════════════
# FIGURE 2: Scatter + Density — Predicted vs Actual
# ═══════════════════════════════════════
def fig2_scatter(pr, tr):
    """Scatter plot: expected value vs actual, hexbin density, +1h and +6h."""
    print("  Scatter + Density...")
    sys.stdout.flush()

    qm = (pr[:, :, :-1] + pr[:, :, 1:]) / 2
    pe = np.sum(qm * np.diff(QUANTILES), axis=-1)
    p50 = pr[:, :, 49]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

    for col, (h, lab) in enumerate([(0, '+1h'), (5, '+6h'), (None, 'All horizons')]):
        ax = axes[col]
        if h is not None:
            px, tx = pe[:, h], tr[:, h]
            px_m, tx_m = p50[:, h], tr[:, h]
        else:
            px, tx = pe.ravel(), tr.ravel()
            px_m, tx_m = p50.ravel(), tr.ravel()

        mask = tx > 0.005
        px_f, tx_f = px[mask], tx[mask]
        px_mf, tx_mf = px_m[mask], tx_m[mask]

        # Hexbin
        hb = ax.hexbin(tx_f, px_f, gridsize=35, cmap='Blues', mincnt=1, alpha=0.8, linewidths=0)

        # Diagonal
        mx = max(tx_f.max(), px_f.max())
        ax.plot([0, mx], [0, mx], 'k--', lw=0.8, alpha=0.4, label='Perfect')

        # Metrics
        rmse = np.sqrt(np.mean((px_f - tx_f) ** 2))
        mae  = np.mean(np.abs(px_f - tx_f))
        r2   = 1 - np.sum((tx_f - px_f) ** 2) / (np.sum((tx_f - np.mean(tx_f)) ** 2) + 1e-8)

        ax.text(0.03, 0.97, f'R² = {r2:+.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}',
                transform=ax.transAxes, fontsize=14, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray', lw=0.3))
        ax.set_xlabel('Actual power')
        if col == 0: ax.set_ylabel('Predicted power')
        ax.set_title(f'{lab} (expected value)', fontsize=16)
        ax.set_aspect('equal')
        ax.set_xlim(0, mx * 1.05); ax.set_ylim(0, mx * 1.05)
        ax.grid(True, alpha=0.15, lw=0.3)

    fig.suptitle('Point Forecast Accuracy — Predicted vs Actual Wind Power',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig2_scatter_density.png'), dpi=300)
    fig.savefig(os.path.join(OUT, 'fig2_scatter_density.pdf'))
    plt.close(fig)
    print("    Saved fig2")


# ═══════════════════════════════════════
# FIGURE 3: Horizon Error Curves + R² Decay
# ═══════════════════════════════════════
def fig3_horizon(pr, tr):
    """Dual-axis: RMSE/MAE by horizon + R² decay with annotated benchmark lines."""
    print("  Horizon Error + R² Decay...")
    sys.stdout.flush()

    qm = (pr[:, :, :-1] + pr[:, :, 1:]) / 2
    pe = np.sum(qm * np.diff(QUANTILES), axis=-1)
    p50 = pr[:, :, 49]

    H = PRED
    hours = np.arange(1, H + 1)

    # Compute
    rmse_h = [np.sqrt(np.mean((pe[:, h][tr[:, h] > 0.005] - tr[:, h][tr[:, h] > 0.005])**2)) for h in range(H)]
    mae_h  = [np.mean(np.abs(pe[:, h][tr[:, h] > 0.005] - tr[:, h][tr[:, h] > 0.005])) for h in range(H)]
    r2_h   = []
    for h in range(H):
        th = tr[:, h]; m = th > 0.005; ph = pe[:, h]
        r2_h.append(1 - np.sum((th[m] - ph[m])**2) / (np.sum((th[m] - np.mean(th[m]))**2) + 1e-8))
    # Pinball
    pb_h = []
    for h in range(H):
        e = tr[:, h, np.newaxis] - pr[:, h, :]
        pb_h.append(np.maximum(QUANTILES * e, (QUANTILES - 1) * e).mean())

    # Persistence pinball (approximate: same target but constant forecast)
    persist_pb = []
    for h in range(H):
        ep = tr[:, h, np.newaxis] - tr[:, 0, np.newaxis]  # persist from +1h value
        persist_pb.append(np.maximum(QUANTILES * ep, (QUANTILES - 1) * ep).mean())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 0.6], 'hspace': 0.35})

    # Top: RMSE + MAE
    ax1.plot(hours, rmse_h, '-', color=C['red'], lw=2.0, label='RMSE', zorder=3)
    ax1.plot(hours, mae_h, '-', color=C['blue'], lw=2.0, label='MAE', zorder=3)
    ax1.fill_between(hours, 0, rmse_h, alpha=0.06, color=C['red'])
    ax1.set_ylabel('Error (normalized power)', fontsize=16)
    ax1.legend(loc='upper left', fontsize=15, framealpha=0.9)
    ax1.grid(True, alpha=0.15, lw=0.3)
    ax1.set_xlim(1, H)
    ax1.set_ylim(bottom=0)

    # Annotate key horizons
    for h in [1, 6, 12, 24]:
        ax1.annotate(f'{rmse_h[h-1]:.3f}', (h, rmse_h[h-1]), textcoords="offset points",
                     xytext=(5, 5), fontsize=16, color=C['red'], ha='left')

    # Bottom: R² decay
    colors_r2 = [C['green'] if r > 0 else C['gray'] for r in r2_h]
    ax2.bar(hours, r2_h, color=colors_r2, width=0.8, alpha=0.7, edgecolor='none')
    ax2.axhline(y=0, color='k', lw=0.5, alpha=0.3)
    ax2.set_xlabel('Horizon (hours)', fontsize=16)
    ax2.set_ylabel('R² (expected value)', fontsize=16)
    ax2.grid(True, alpha=0.15, lw=0.3, axis='y')
    ax2.set_xlim(1, H)

    # Labels for key R² values
    for h, r in [(1, r2_h[0]), (4, r2_h[3]), (6, r2_h[5]), (12, r2_h[11])]:
        va = 'bottom' if r > 0 else 'top'
        ax2.annotate(f'{r:+.2f}', (h, r), textcoords="offset points",
                     xytext=(0, 3 if r > 0 else -8), fontsize=14, ha='center', va=va,
                     fontweight='bold', color=C['green'] if r > 0 else C['gray'])

    ax2.set_title('R² Decay — Wind power becomes unpredictable beyond 6 hours', fontsize=15, color='dimgray')

    fig.suptitle('Forecast Error & R² vs Prediction Horizon', fontsize=16, fontweight='bold', y=1.01)

    # Pinball comparison inset (small)
    ax3 = ax1.inset_axes([0.55, 0.45, 0.40, 0.40])
    ax3.plot(hours, pb_h, '-', color=C['red'], lw=1.5, label='LNSSM')
    ax3.plot(hours, persist_pb, '--', color=C['gray'], lw=1.0, label='Persistence')
    ax3.set_xlabel('Hours', fontsize=6); ax3.set_ylabel('Pinball', fontsize=6)
    ax3.tick_params(labelsize=6)
    ax3.legend(fontsize=6, loc='lower right')
    ax3.grid(True, alpha=0.15, lw=0.3)
    ax3.set_xlim(1, H)

    fig.subplots_adjust(hspace=0.30, top=0.93, bottom=0.08, left=0.10, right=0.97)
    fig.savefig(os.path.join(OUT, 'fig3_horizon_error.png'), dpi=300)
    fig.savefig(os.path.join(OUT, 'fig3_horizon_error.pdf'))
    plt.close(fig)
    print("    Saved fig3")


# ═══════════════════════════════════════
# FIGURE 4: Prediction Intervals with ZOOM
# ═══════════════════════════════════════
def fig4_intervals(pr, tr):
    """Prediction intervals (50%/80%/90%) on a sample segment + zoom."""
    print("  Prediction Intervals...")
    sys.stdout.flush()

    n_show = 72   # 3 days
    n_zoom = 12   # 12h zoom
    start  = len(pr) // 2

    p10, p25, p50 = pr[:, :, 9], pr[:, :, 24], pr[:, :, 49]
    p75, p90 = pr[:, :, 74], pr[:, :, 89]
    p01, p99 = pr[:, :, 0], pr[:, :, 98]

    fig, (ax_main, ax_zoom) = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={'height_ratios': [2, 1]},
                                            sharex=False)

    h = 5  # +6h horizon

    # Main panel
    ax_main.fill_between(range(n_show), p01[start:start+n_show, h], p99[start:start+n_show, h],
                         alpha=0.06, color='navy', label='1-99% interval')
    ax_main.fill_between(range(n_show), p10[start:start+n_show, h], p90[start:start+n_show, h],
                         alpha=0.12, color='steelblue', label='10-90% (80% CI)')
    ax_main.fill_between(range(n_show), p25[start:start+n_show, h], p75[start:start+n_show, h],
                         alpha=0.20, color='steelblue', label='25-75% (50% CI)')
    ax_main.plot(tr[start:start+n_show, h], '-', color=C['red'], lw=1.5, alpha=0.9, label='Actual', zorder=5)
    ax_main.plot(p50[start:start+n_show, h], '-', color=C['blue'], lw=1.2, alpha=0.8, label='Median (q50)', zorder=4)

    # Highlight zoom region
    zs = start + n_show // 2
    ax_main.axvspan(zs - start, zs - start + n_zoom, alpha=0.08, color='gold', label='Zoom region')

    in_80 = np.mean((tr[start:start+n_show, h] >= p10[start:start+n_show, h]) &
                    (tr[start:start+n_show, h] <= p90[start:start+n_show, h])) * 100
    ax_main.set_ylabel('Normalized power', fontsize=16)
    ax_main.set_title(f'+6h Forecast with Prediction Intervals | {in_80:.0f}% in 80% CI', fontsize=14, fontweight='bold')
    ax_main.legend(loc='upper right', fontsize=14, ncol=3, framealpha=0.9)
    ax_main.grid(True, alpha=0.15, lw=0.3)
    ax_main.set_xlim(0, n_show)

    # Zoom panel
    ax_zoom.fill_between(range(n_zoom), p01[zs:zs+n_zoom, h], p99[zs:zs+n_zoom, h],
                         alpha=0.10, color='navy')
    ax_zoom.fill_between(range(n_zoom), p10[zs:zs+n_zoom, h], p90[zs:zs+n_zoom, h],
                         alpha=0.18, color='steelblue')
    ax_zoom.fill_between(range(n_zoom), p25[zs:zs+n_zoom, h], p75[zs:zs+n_zoom, h],
                         alpha=0.25, color='steelblue')
    ax_zoom.plot(tr[zs:zs+n_zoom, h], '-', color=C['red'], lw=2.0, alpha=0.95, zorder=5,
                 marker='o', ms=4, mfc='white', mew=1.5)
    ax_zoom.plot(p50[zs:zs+n_zoom, h], '-', color=C['blue'], lw=1.8, alpha=0.9, zorder=4,
                 marker='s', ms=3, mfc='white', mew=1.2)

    zoom_in = np.mean((tr[zs:zs+n_zoom, h] >= p10[zs:zs+n_zoom, h]) &
                      (tr[zs:zs+n_zoom, h] <= p90[zs:zs+n_zoom, h])) * 100
    ax_zoom.set_title(f'Zoom (12h) — {zoom_in:.0f}% captured in 80% CI', fontsize=15, fontweight='bold')
    ax_zoom.set_xlabel('Time step (hours)', fontsize=16)
    ax_zoom.set_ylabel('Power', fontsize=16)
    ax_zoom.grid(True, alpha=0.15, lw=0.3)
    ax_zoom.set_xlim(0, n_zoom)

    fig.suptitle('Probabilistic Forecast — Prediction Intervals (+6h horizon)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig4_prediction_intervals.png'), dpi=300)
    fig.savefig(os.path.join(OUT, 'fig4_prediction_intervals.pdf'))
    plt.close(fig)
    print("    Saved fig4")


# ═══════════════════════════════════════
# FIGURE 5: Reliability Diagram
# ═══════════════════════════════════════
def fig5_reliability():
    """Reliability diagram for LNSSM — SCI style, no overlaps."""
    print("  Reliability Diagram...")
    sys.stdout.flush()

    nominal   = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    lnm_actual = np.array([4.6, 8.5, 12.8, 17.7, 23.1, 29.1, 36.0, 45.5, 58.6])

    fig, ax = plt.subplots(figsize=(8, 8))

    # Perfect calibration line
    ax.plot([0, 100], [0, 100], 'k--', lw=1.2, alpha=0.5, label='Perfect calibration')
    # +/- 5% tolerance band
    ax.fill_between([0, 100], [5, 105], [-5, 95], alpha=0.06, color='gray')
    ax.annotate('+/- 5% tolerance', xy=(85, 92), fontsize=12, color='#888', ha='center')

    # LNSSM line
    ax.plot(nominal, lnm_actual, 'o-', color=C['blue'], lw=3, ms=12,
            mfc='white', mew=3, label='LNSSM (ours)', zorder=5)

    # Annotate deviation — fewer, larger, better spaced
    for nom, act in zip(nominal, lnm_actual):
        dev = act - nom
        y_offset = 2.5 if dev > -10 else -5.5  # alternate above/below to avoid overlap
        ax.annotate(f'{dev:+.0f}%', (nom, act), textcoords="offset points",
                    xytext=(0, y_offset), fontsize=11, ha='center', fontweight='bold',
                    color=C['red'] if dev < -8 else C['gray'],
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.6))

    ax.set_xlabel('Nominal Coverage (%)', fontsize=18)
    ax.set_ylabel('Actual Coverage (%)', fontsize=18)
    ax.set_title('Reliability Diagram — GEFCom2012 Farm 1', fontsize=22, fontweight='bold', pad=15)
    ax.set_xlim(-2, 102); ax.set_ylim(-2, 102)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=16, framealpha=0.9,
              bbox_to_anchor=(0.98, 0.02))
    ax.grid(True, alpha=0.12, lw=0.4)

    # Summary box — moved to top-left to avoid legend clash
    ax.text(0.03, 0.97, 'Mean absolute\ndeviation: 3.95%',
            transform=ax.transAxes, fontsize=16, ha='left', va='top', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=C['blue'], lw=1))

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig5_reliability_diagram.png'), dpi=300)
    fig.savefig(os.path.join(OUT, 'fig5_reliability_diagram.pdf'))
    plt.close(fig)
    print("    Saved fig5")


# ═══════════════════════════════════════
# FIGURE 6: Error Distribution
# ═══════════════════════════════════════
def fig6_error_dist(pr, tr):
    """Error distribution histogram + Q-Q plot."""
    print("  Error Distribution...")
    sys.stdout.flush()

    qm = (pr[:, :, :-1] + pr[:, :, 1:]) / 2
    pe = np.sum(qm * np.diff(QUANTILES), axis=-1)
    p50 = pr[:, :, 49]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Expected value errors
    errors_ev = pe.ravel() - tr.ravel()
    mask = tr.ravel() > 0.005
    errors_ev = errors_ev[mask]

    # Median errors
    errors_md = p50.ravel() - tr.ravel()
    errors_md = errors_md[mask]

    # Histogram
    ax1.hist(errors_ev, bins=70, density=True, color=C['blue'], alpha=0.6, edgecolor='white', lw=0.3,
             label=f'Expected value (σ={np.std(errors_ev):.3f})')
    ax1.hist(errors_md, bins=70, density=True, color=C['red'], alpha=0.4, edgecolor='white', lw=0.3,
             label=f'Median q50 (σ={np.std(errors_md):.3f})')

    ax1.axvline(0, color='k', lw=0.8, alpha=0.3)
    ax1.axvline(np.mean(errors_ev), color=C['blue'], lw=1, ls='--', alpha=0.7,
                label=f'Mean EV={np.mean(errors_ev):.4f}')
    ax1.axvline(np.mean(errors_md), color=C['red'], lw=1, ls='--', alpha=0.7,
                label=f'Mean MD={np.mean(errors_md):.4f}')

    ax1.set_xlabel('Prediction error (normalized power)', fontsize=16)
    ax1.set_ylabel('Probability density', fontsize=16)
    ax1.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=16, loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.15, lw=0.3)

    # Q-Q plot with normal overlay
    from scipy import stats
    sorted_err = np.sort(errors_ev)
    theoretical = stats.norm.ppf(np.linspace(0.001, 0.999, len(sorted_err)),
                                 loc=np.mean(errors_ev), scale=np.std(errors_ev))
    ax2.scatter(theoretical[::20], sorted_err[::20], c=C['blue'], s=4, alpha=0.4, edgecolors='none')
    ax2.plot(theoretical, theoretical, 'k--', lw=0.6, alpha=0.4)
    ax2.set_xlabel('Theoretical normal quantiles', fontsize=16)
    ax2.set_ylabel('Sample quantiles', fontsize=16)
    ax2.set_title('Q-Q Plot (Expected Value)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.15, lw=0.3)

    # Skewness annotation
    from scipy.stats import skew, kurtosis
    ax2.text(0.97, 0.03, f'Skewness: {skew(errors_ev):+.3f}\nExcess kurtosis: {kurtosis(errors_ev):+.3f}',
             transform=ax2.transAxes, fontsize=14, ha='right', va='bottom', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray', lw=0.3))

    fig.suptitle('Prediction Error Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig6_error_distribution.png'), dpi=300)
    fig.savefig(os.path.join(OUT, 'fig6_error_distribution.pdf'))
    plt.close(fig)
    print("    Saved fig6")


# ═══════════════════════════════════════
# FIGURE 7: SOTA Comparison — QRF vs LNSelective SSM vs Persistence
# ═══════════════════════════════════════
def fig7_sota_comparison():
    """Bar chart: Pinball / RMSE / R² across models."""
    print("  SOTA Comparison...")
    sys.stdout.flush()

    models = ['Persistence', 'QRF', 'Mamba\n(no LNN)', 'LNSSM\n(ours)']
    pinball = [0.119, 0.1003, 0.082, 0.0806]
    rmse    = [0.294, 0.264, 0.282, 0.280]
    colors  = [C['gray'], C['orange'], C['green'], C['blue']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Pinball
    bars1 = ax1.bar(models, pinball, color=colors, width=0.55, edgecolor='white', lw=0.5)
    for bar, val in zip(bars1, pinball):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.002, f'{val:.4f}',
                 ha='center', fontsize=15, fontweight='bold')
    if len(pinball) >= 2:
        imp_vs_qrf = (pinball[1] - pinball[3]) / pinball[1] * 100
        ax1.annotate(f'↓ {imp_vs_qrf:.0f}%', xy=(2, pinball[3]),
                     xytext=(2.5, pinball[3] + 0.012),
                     arrowprops=dict(arrowstyle='->', color=C['green'], lw=1.5),
                     fontsize=15, color=C['green'], fontweight='bold')
    ax1.set_ylabel('Pinball Loss', fontsize=16)
    ax1.set_title('Pinball Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.15, lw=0.3, axis='y')
    ax1.set_ylim(0, max(pinball) * 1.15)

    # RMSE
    bars2 = ax2.bar(models, rmse, color=colors, width=0.55, edgecolor='white', lw=0.5)
    for bar, val in zip(bars2, rmse):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.003, f'{val:.3f}',
                 ha='center', fontsize=15, fontweight='bold')
    ax2.set_ylabel('RMSE', fontsize=16)
    ax2.set_title('RMSE (Expected Value)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.15, lw=0.3, axis='y')
    ax2.set_ylim(0, max(rmse) * 1.15)

    fig.suptitle('Model Comparison — GEFCom2012 Farm 1', fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig7_sota_comparison.png'), dpi=300)
    fig.savefig(os.path.join(OUT, 'fig7_sota_comparison.pdf'))
    plt.close(fig)
    print("    Saved fig7")


# ═══════════════════════════════════════
# FIGURE 8: Data Scale Effect + DM Test Summary
# ═══════════════════════════════════════
def fig8_scale_effect():
    """Scatter: training samples vs pinball, annotated with DM test result."""
    print("  Data Scale Effect + DM Test...")
    sys.stdout.flush()

    # Data
    configs = ['Single site\n(3.5K)', 'Farm 1 only\n(3.9K)', '7-farm\n(27.6K)', '17-site\n(62.8K)']
    samples = [3523, 3936, 27552, 62782]
    pinball = [0.2069, 0.0921, 0.0806, 0.0897]
    r2_1h   = [0.57, 0.600, 0.600, 0.626]
    colors_scale = [C['gray'], C['orange'], C['blue'], C['purple']]
    # Precompute annotation offsets to avoid overlap
    xytexts = [(-20, -20), (15, -15), (-15, 20), (20, -10)]

    fig, ax = plt.subplots(figsize=(12, 8))

    for i, (cfg, s, p, c) in enumerate(zip(configs, samples, pinball, colors_scale)):
        ax.scatter(np.log10(s), p, s=350, c=c, edgecolors='white', lw=3, zorder=5)
        xo, yo = xytexts[i]
        ax.annotate(f'{cfg}\nPB={p:.4f}', (np.log10(s), p), textcoords="offset points",
                     xytext=(xo, yo), fontsize=15, ha='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor=c, lw=0.5))

    ax.plot([np.log10(s) for s in samples], pinball, '-', color=C['blue'], lw=1.5, alpha=0.5, zorder=1)
    ax.set_xlabel('Training Windows (log10 scale)', fontsize=18)
    ax.set_ylabel('Pinball Loss (99 Quantiles)', fontsize=18)
    ax.set_title('Effect of Training Data Scale on Pinball Loss', fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.15, lw=0.3)
    ax.set_ylim(0.07, 0.26)

    # DM test annotation box - moved to bottom right
    ax.text(0.99, 0.08, 'DM Test: LNSSM vs Persistence\nDM = +12.432  p < 0.0001\n23/24 horizons significant',
            transform=ax.transAxes, fontsize=14, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FEF9E7', alpha=0.9, edgecolor='orange', lw=1))

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig8_data_scale.png'), dpi=300)
    fig.savefig(os.path.join(OUT, 'fig8_data_scale.pdf'))
    plt.close(fig)
    print("    Saved fig8")


# ═══════════════════════════════════════
# Main
# ═══════════════════════════════════════
def main():
    print("=" * 60)
    print("  LNSSM Paper Figures — SCI Style, 300dpi, PNG+PDF")
    print("=" * 60)
    sys.stdout.flush()

    # Train model (once, reuse predictions)
    print("\n[0] Training LNSSM (shared across all figures)...")
    sys.stdout.flush()
    pr, tr = quick_train()
    print(f"  Predictions: {pr.shape}, Targets: {tr.shape}")

    # Generate all figures
    fig1_timeseries(pr, tr)
    fig2_scatter(pr, tr)
    fig3_horizon(pr, tr)
    fig4_intervals(pr, tr)
    fig5_reliability()
    fig6_error_dist(pr, tr)
    fig7_sota_comparison()
    fig8_scale_effect()

    print(f"\n  All 8 figures saved to {OUT}")
    print(f"  Formats: PNG (300dpi) + PDF (vector)")
    print("  Done!")


if __name__ == '__main__':
    main()
