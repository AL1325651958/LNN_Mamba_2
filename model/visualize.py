"""
Visualization: Raw (Ground Truth) vs Predicted Wind Power.
Saves comparison plots to plots/ directory.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

from model import LNNMambaTransformer, WindDataModule


def generate_predictions():
    """Load checkpoint and generate test predictions."""
    print('Loading checkpoint...')
    ckpt = torch.load(
        'checkpoints/lmt_quick_0.15.pt',
        map_location='cuda', weights_only=False
    )
    print(f'  Epoch: {ckpt["epoch"]}, Val RMSE: {ckpt["val_rmse"]:.1f} MW')

    print('Loading data...')
    dm = WindDataModule(
        site_name='Wind_farm_site_2_200MW',
        seq_len=48, pred_len=12, batch_size=32, stride=4,
        train_ratio=0.4, val_ratio=0.1,
    )
    _, _, test_loader = dm.get_dataloaders()

    print('Building model...')
    model = LNNMambaTransformer(
        n_vars=dm.n_vars, d_model=48, n_mamba_blocks=1,
        n_transformer_layers=1, d_state=8, d_conv=3,
        n_heads=4, pred_len=12, lnn_hidden=24,
    ).cuda()
    model.load_state_dict(ckpt['model'])
    model.eval()

    print('Generating predictions...')
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y, ts in test_loader:
            out = model(x.cuda(), ts.cuda())
            all_preds.append(out['pred'].cpu().numpy())
            all_targets.append(y.cpu().numpy())

    preds = np.concatenate(all_preds).ravel()
    targets = np.concatenate(all_targets).ravel()

    # Inverse transform
    if dm.scaler_power is not None:
        preds = dm.scaler_power.inverse_transform(preds.reshape(-1, 1)).ravel()
        targets = dm.scaler_power.inverse_transform(targets.reshape(-1, 1)).ravel()

    return preds, targets


def plot_time_series(preds: np.ndarray, targets: np.ndarray, n_points: int = 576):
    """Plot a continuous segment of predictions vs ground truth over time."""
    print('Plotting time series...')

    # Take a slice from the middle (more interesting)
    start = len(targets) // 3
    end = start + n_points

    pred_slice = preds[start:end]
    targ_slice = targets[start:end]

    # Create time axis (15-min intervals)
    base_time = datetime(2020, 6, 1, 0, 0)
    times = [base_time + timedelta(minutes=15 * i) for i in range(n_points)]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(times, targ_slice, 'b-', linewidth=1.2, alpha=0.85, label='Ground Truth (Raw)')
    ax.plot(times, pred_slice, 'r--', linewidth=1.2, alpha=0.85, label='LMT Prediction')
    ax.fill_between(times, targ_slice, pred_slice, alpha=0.12, color='gray',
                    label=f'Error (|ε|¯={np.mean(np.abs(pred_slice - targ_slice)):.1f} MW)')

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Power (MW)', fontsize=12)
    ax.set_title('LNN-Gated Selective SSM-Transformer: Wind Power Prediction vs Ground Truth\n'
                 f'Wind Farm Site 2 (200MW) — {n_points} points ({n_points*15/60:.0f} hours)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.xticks(rotation=30)
    plt.tight_layout()

    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/timeseries_raw_vs_pred.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  -> plots/timeseries_raw_vs_pred.png')


def plot_scatter(preds: np.ndarray, targets: np.ndarray):
    """Scatter plot of predicted vs actual with R² and error metrics."""
    print('Plotting scatter...')

    # Filter to reasonable range
    mask = (targets > 0.5) & (targets < 200)
    pf = preds[mask]
    tf = targets[mask]

    # Subsample for clarity
    if len(pf) > 5000:
        idx = np.random.choice(len(pf), 5000, replace=False)
        pf = pf[idx]
        tf = tf[idx]

    rmse = np.sqrt(np.mean((pf - tf) ** 2))
    mae = np.mean(np.abs(pf - tf))
    r2 = 1 - np.sum((tf - pf) ** 2) / np.sum((tf - np.mean(tf)) ** 2)

    fig, ax = plt.subplots(figsize=(7, 7))

    # Hexbin for density
    hb = ax.hexbin(tf, pf, gridsize=50, cmap='Blues', mincnt=1, alpha=0.85)

    # Perfect prediction line
    max_val = max(tf.max(), pf.max())
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.5, label='Perfect Prediction')

    # Metrics box
    textstr = f'RMSE = {rmse:.1f} MW\nMAE = {mae:.1f} MW\nR² = {r2:.3f}\nN = {len(pf):,}'
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, fontfamily='monospace')

    ax.set_xlabel('Ground Truth Power (MW)', fontsize=12)
    ax.set_ylabel('Predicted Power (MW)', fontsize=12)
    ax.set_title('LMT Model: Predicted vs Actual Wind Power\n'
                 'Wind Farm Site 2 (200MW)',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, max_val * 1.05)
    ax.set_ylim(0, max_val * 1.05)
    ax.set_aspect('equal')
    ax.legend(loc='lower right')
    plt.colorbar(hb, ax=ax, label='Density')
    plt.tight_layout()

    plt.savefig('plots/scatter_pred_vs_actual.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  -> plots/scatter_pred_vs_actual.png')


def plot_error_distribution(preds: np.ndarray, targets: np.ndarray):
    """Histogram of prediction errors."""
    print('Plotting error distribution...')

    mask = targets > 1.0
    errors = preds[mask] - targets[mask]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(errors, bins=80, color='steelblue', edgecolor='white', alpha=0.85, density=True)

    # Overlay normal fit
    from scipy import stats
    mu, sigma = stats.norm.fit(errors)
    x = np.linspace(errors.min(), errors.max(), 200)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
            label=f'Normal fit: μ={mu:.1f}, σ={sigma:.1f} MW')

    ax.axvline(x=0, color='k', linestyle='--', alpha=0.4)
    ax.axvline(x=mu, color='r', linestyle=':', alpha=0.6)

    ax.set_xlabel('Prediction Error (MW)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('LMT Prediction Error Distribution\n'
                 f'Mean={np.mean(errors):.1f} MW, Std={np.std(errors):.1f} MW, '
                 f'Skew={stats.skew(errors):.2f}',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('plots/error_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  -> plots/error_distribution.png')


def plot_horizon_metrics(preds: np.ndarray, targets: np.ndarray):
    """RMSE and MAE as a function of prediction horizon."""
    print('Plotting horizon metrics...')

    pred_h = preds.reshape(-1, 12)
    targ_h = targets.reshape(-1, 12)

    horizons = [(h + 1) * 15 for h in range(12)]  # minutes

    rmse_list = []
    mae_list = []
    for h in range(12):
        rmse_list.append(np.sqrt(np.mean((pred_h[:, h] - targ_h[:, h]) ** 2)))
        mae_list.append(np.mean(np.abs(pred_h[:, h] - targ_h[:, h])))

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color1 = '#d62728'
    ax1.plot(horizons, rmse_list, 'o-', color=color1, linewidth=2, markersize=8, label='RMSE')
    ax1.set_xlabel('Prediction Horizon (minutes)', fontsize=12)
    ax1.set_ylabel('RMSE (MW)', fontsize=12, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    color2 = '#1f77b4'
    ax2.plot(horizons, mae_list, 's-', color=color2, linewidth=2, markersize=8, label='MAE')
    ax2.set_ylabel('MAE (MW)', fontsize=12, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(bottom=0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)

    ax1.set_title('LMT Prediction Error vs Horizon\n'
                  f'Wind Farm Site 2 (200MW) — seq_len=48, pred_len=12',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.2)
    ax1.set_xticks(horizons)
    ax1.set_xticklabels([f'+{h}min' for h in horizons], rotation=45)

    plt.tight_layout()
    plt.savefig('plots/horizon_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  -> plots/horizon_metrics.png')


def main():
    os.makedirs('plots', exist_ok=True)

    print('=' * 60)
    print('LMT Visualization: Raw vs Predicted Wind Power')
    print('=' * 60)

    preds, targets = generate_predictions()
    print(f'Generated {len(preds):,} prediction points')

    plot_time_series(preds, targets)
    plot_scatter(preds, targets)
    plot_error_distribution(preds, targets)
    plot_horizon_metrics(preds, targets)

    print(f'\nAll plots saved to plots/ directory')


if __name__ == '__main__':
    main()
