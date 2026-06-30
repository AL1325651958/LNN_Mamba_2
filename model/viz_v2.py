"""Plot v2 results: timeseries, scatter, horizon error, regime analysis."""
import numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    os.makedirs('plots', exist_ok=True)
    d = np.load('checkpoints/v2_results.npz')
    pred, target = d['pred'], d['target']  # (N, 96)

    ph, th = pred.ravel(), target.ravel()
    mask = th > 1.0; pf, tf = ph[mask], th[mask]
    rmse = np.sqrt(np.mean((pf-tf)**2)); mae = np.mean(np.abs(pf-tf))
    print(f'Overall: RMSE={rmse:.1f} MW, MAE={mae:.1f} MW, N={len(pf):,}')

    # ── 1. Timeseries ──
    n_show = 576
    start = len(pred) // 3
    fig, ax = plt.subplots(figsize=(16,5))
    ax.plot(target[start:start+n_show,0], 'b-', lw=1, alpha=0.8, label='Ground Truth')
    ax.plot(pred[start:start+n_show,0], 'r--', lw=1, alpha=0.8, label='LMT v2')
    ax.fill_between(range(n_show), target[start:start+n_show,0], pred[start:start+n_show,0],
                    alpha=0.1, color='gray')
    ax.set_xlabel('Time step (15-min)'); ax.set_ylabel('Power (MW)')
    ax.set_title(f'LMT v2: t+15min Prediction | RMSE={rmse:.0f}MW'); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig('plots/v2_timeseries.png', dpi=150); plt.close()
    print(' → plots/v2_timeseries.png')

    # ── 2. Scatter ──
    if len(pf) > 5000:
        idx = np.random.choice(len(pf), 5000, replace=False); pf_s, tf_s = pf[idx], tf[idx]
    else: pf_s, tf_s = pf, tf
    fig, ax = plt.subplots(figsize=(7,7))
    ax.hexbin(tf_s, pf_s, gridsize=40, cmap='Blues', mincnt=1, alpha=0.85)
    mx = max(tf.max(), pf.max())
    ax.plot([0,mx],[0,mx],'k--',lw=1,alpha=0.5,label='Perfect')
    r2 = 1 - np.sum((pf-tf)**2)/(np.sum((tf-np.mean(tf))**2)+1e-8)
    ax.text(0.05,0.95,f'RMSE={rmse:.0f}MW\nMAE={mae:.0f}MW\nR²={r2:.3f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round',facecolor='white',alpha=0.85), fontfamily='monospace')
    ax.set_xlabel('Ground Truth (MW)'); ax.set_ylabel('Predicted (MW)')
    ax.set_title('LMT v2: Predicted vs Actual'); ax.set_aspect('equal')
    plt.tight_layout(); plt.savefig('plots/v2_scatter.png', dpi=150); plt.close()
    print(' → plots/v2_scatter.png')

    # ── 3. Horizon error ──
    rmse_h = [np.sqrt(np.mean((pred[:,h]-target[:,h])**2)) for h in range(96)]
    mae_h  = [np.mean(np.abs(pred[:,h]-target[:,h])) for h in range(96)]
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot([(h+1)*15 for h in range(96)], rmse_h, 'r-', lw=2, label='RMSE')
    ax1.set_xlabel('Horizon (min)'); ax1.set_ylabel('RMSE (MW)', color='r')
    ax2 = ax1.twinx()
    ax2.plot([(h+1)*15 for h in range(96)], mae_h, 'b-', lw=2, label='MAE')
    ax2.set_ylabel('MAE (MW)', color='b')
    ax1.set_title(f'Error vs Horizon | v1→v2: +multi-scale +ΔP +weighted loss +full data')
    l1,p1=ax1.get_legend_handles_labels(); l2,p2=ax2.get_legend_handles_labels()
    ax1.legend(l1+l2,p1+p2,loc='upper left'); ax1.grid(alpha=0.2)
    plt.tight_layout(); plt.savefig('plots/v2_horizon.png', dpi=150); plt.close()
    print(' → plots/v2_horizon.png')

    # ── 4. Error distribution ──
    errors = pf - tf
    fig, ax = plt.subplots(figsize=(10,5))
    ax.hist(errors, bins=60, color='steelblue', edgecolor='white', alpha=0.85, density=True)
    ax.axvline(0,color='k',ls='--',alpha=0.3)
    ax.set_xlabel('Error (MW)'); ax.set_ylabel('Density')
    ax.set_title(f'Error Distribution | μ={np.mean(errors):.1f} σ={np.std(errors):.1f} MW')
    ax.grid(alpha=0.2)
    plt.tight_layout(); plt.savefig('plots/v2_errors.png', dpi=150); plt.close()
    print(' → plots/v2_errors.png')

    print('\nAll v2 plots saved!')


if __name__ == '__main__':
    main()
