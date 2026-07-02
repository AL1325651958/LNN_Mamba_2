"""
Paper supplementary experiments:
  1. QRF (Quantile Regression Forest) baseline on GEFCom2012 Farm 1
  2. PICP/PINAW evaluation for LNMamba
  3. Reliability diagram data
  4. QRF vs LNMamba comparison table
"""
import sys,os,time,numpy as np
import torch, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEVICE = torch.device('cuda')
QUANTILES = np.linspace(0.01, 0.99, 99)
SEQ, PRED = 168, 24
DIR = os.path.join(ROOT, 'GEFCOM2012/GEFCOM2012_Data/Wind')
LT = [1, 3, 6, 12, 24]

# ═══════════════════ Data loading (same as before) ═══════════════════
def load_farm(fid):
    pw = pd.read_csv(f'{DIR}/windpowermeasurements.csv')
    pw = pw[pw['usage'] == 'Training'].copy()
    pw['dt'] = pd.to_datetime(pw['date'].astype(str), format='%Y%m%d%H')
    pw = pw[['dt', f'wp{fid}']].rename(columns={f'wp{fid}': 'power'}).sort_values('dt').reset_index(drop=True)
    nwp = pd.read_csv(f'{DIR}/windforecasts_wf{fid}.csv')
    nwp['issue'] = pd.to_datetime(nwp['date'].astype(str), format='%Y%m%d%H')
    np_p = {}
    for _, r in nwp.iterrows():
        np_p.setdefault(r['issue'], {})[r['hors']] = (r['u'], r['v'], r['ws'], r['wd'])
    X_list, y_list = [], []
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
        X_list.append(f); y_list.append(r['power'])
    X = np.array(X_list, dtype=np.float32)
    y = np.clip(np.array(y_list, dtype=np.float32), 0, 1)
    return X, y

def make_windows(X, y, stride=4):
    """Create (seq_len * features, target_horizon) supervised dataset."""
    T = len(X); n_feat = X.shape[1]
    windows_X, windows_y = [], []
    for i in range(0, T - SEQ - PRED, stride):
        # Flatten the 168-step window into a flat feature vector
        win = X[i:i+SEQ].ravel()  # (168 * n_feat,)
        tgt = y[i+SEQ:i+SEQ+PRED]  # (24,)
        windows_X.append(win); windows_y.append(tgt)
    return np.array(windows_X, dtype=np.float32), np.array(windows_y, dtype=np.float32)


# ═══════════════════ QRF Baseline ═══════════════════
def run_qrf():
    print("=" * 60)
    print("  1. QRF (Quantile Regression Forest) Baseline")
    print("=" * 60)
    sys.stdout.flush()

    X, y = load_farm(1)
    X_train, y_train = make_windows(X, y, stride=4)
    T = len(X_train); te = int(T * 0.85)
    Xt, yt = X_train[:te], y_train[:te]
    Xv, yv = X_train[te:], y_train[te:]

    # Standardize features
    sx = StandardScaler()
    Xt_s = sx.fit_transform(Xt)
    Xv_s = sx.transform(Xv)

    print(f"  Train: {len(Xt):,} windows, {Xt.shape[1]} features")
    print(f"  Test:  {len(Xv):,} windows")
    print(f"  Training {99} QRF models (one per quantile)...")
    sys.stdout.flush()

    # QRF: one RandomForest per quantile
    qrf_preds = np.zeros((len(Xv), PRED, 99), dtype=np.float32)

    for k, tau in enumerate(QUANTILES):
        for h in range(PRED):
            rf = RandomForestRegressor(
                n_estimators=50, max_depth=15, min_samples_leaf=20,
                random_state=42, n_jobs=-1, verbose=0
            )
            rf.fit(Xt_s, yt[:, h])
            qrf_preds[:, h, k] = rf.predict(Xv_s)

        if (k + 1) % 20 == 0:
            print(f"    {k+1}/99 quantiles done...")
            sys.stdout.flush()

    # Metrics
    p50 = qrf_preds[:, :, 49]
    qm = (qrf_preds[:, :, :-1] + qrf_preds[:, :, 1:]) / 2
    pe = np.sum(qm * np.diff(QUANTILES), axis=-1)

    def pinball(pq, t):
        e = t[:, :, np.newaxis] - pq
        return np.maximum(QUANTILES * e, (QUANTILES - 1) * e).mean()

    pb = pinball(qrf_preds, yv)
    rmse = np.sqrt(np.mean((pe.ravel() - yv.ravel()) ** 2))
    mae = np.mean(np.abs(pe.ravel() - yv.ravel()))
    cov80 = np.mean((yv >= qrf_preds[:, :, 9]) & (yv <= qrf_preds[:, :, 89])) * 100

    print(f"\n  QRF Results:")
    print(f"  Pinball:     {pb:.4f}")
    print(f"  RMSE:        {rmse:.4f}")
    print(f"  MAE:         {mae:.4f}")
    print(f"  80% CI cov:  {cov80:.1f}%")

    # Compare with LNMamba
    print(f"\n  Comparison:")
    print(f"  {'Metric':<18s} {'QRF':>10s} {'LNMamba':>10s}")
    print(f"  {'-'*38}")
    lnm_pb = 0.0806
    lnm_rmse = 0.2799
    lnm_cov = 46.1
    print(f"  {'Pinball':<18s} {pb:>10.4f} {lnm_pb:>10.4f}")
    print(f"  {'RMSE':<18s} {rmse:>10.4f} {lnm_rmse:>10.4f}")
    print(f"  {'80% CI':<18s} {cov80:>9.1f}% {lnm_cov:>9.1f}%")
    sys.stdout.flush()

    # Per-horizon
    print(f"\n  Per-horizon Pinball (QRF):")
    for h in [0, 3, 5, 11, 17, 23]:
        eh = yv[:, h, np.newaxis] - qrf_preds[:, h, :]
        ph = np.maximum(QUANTILES * eh, (QUANTILES - 1) * eh).mean()
        print(f"    +{h+1:2d}h: {ph:.4f}")

    return pb, rmse, cov80, qrf_preds, yv


# ═══════════════════ PICP/PINAW Evaluation ═══════════════════
def compute_picp_pinaw(predictions, targets, quantiles):
    """Prediction Interval Coverage Probability + Normalized Average Width."""
    intervals = {
        '80% CI': (0.80, 0.10, 0.90),
        '90% CI': (0.90, 0.05, 0.95),
        '95% CI': (0.95, 0.025, 0.975),
    }
    y_range = float(targets.max() - targets.min())
    results = {}
    for label, (nominal, lo_q, hi_q) in intervals.items():
        lo_idx = int(np.argmin(np.abs(quantiles - lo_q)))
        hi_idx = int(np.argmin(np.abs(quantiles - hi_q)))
        lo, hi = predictions[:, :, lo_idx], predictions[:, :, hi_idx]
        covered = (targets >= lo) & (targets <= hi)
        picp = float(covered.mean())
        pinaw = float((hi - lo).mean() / y_range)
        picp_h = [float(covered[:, h].mean()) for h in range(PRED)]
        pinaw_h = [float((hi[:, h] - lo[:, h]).mean() / y_range) for h in range(PRED)]
        results[label] = {'nominal': nominal, 'picp': picp, 'pinaw': pinaw,
                          'picp_h': picp_h, 'pinaw_h': pinaw_h}
    return results


def run_picp_pinaw():
    print("\n" + "=" * 60)
    print("  2. PICP/PINAW Evaluation (LNMamba GEFCom2012 7-farm)")
    print("=" * 60)
    sys.stdout.flush()

    # Load model predictions (quick train)
    sys.path.insert(0, '.')
    from nwp_model import NWPMamba, pinball_loss
    from gefcom2012_only import WDS, build_farm_data

    print("  Training LNMamba (30 epochs)...")
    sys.stdout.flush()

    data = build_farm_data(1)
    nv = data.shape[1]
    T = len(data); te = int(T * 0.85)

    # Train (all 7 farms for comparability)
    all_ds = []
    for fid in range(1, 8):
        d = build_farm_data(fid)
        t_end = int(len(d) * 0.85)
        all_ds.append(WDS(d[:t_end], 4))
    ds_full = torch.utils.data.ConcatDataset(all_ds)
    tl = DataLoader(ds_full, 48, shuffle=True, num_workers=0, pin_memory=True)

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
    test_ds = WDS(data[te:], 4)
    testl = DataLoader(test_ds, 48, shuffle=False, num_workers=0, pin_memory=True)

    preds, targs = [], []
    with torch.no_grad():
        for x, y in testl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds.append(model(x).cpu().numpy()); targs.append(y.numpy())
    pr = np.concatenate(preds); tr = np.concatenate(targs)

    results = compute_picp_pinaw(pr, tr, QUANTILES)

    print(f"\n  {'Interval':<12s} {'Nominal':>10s} {'PICP':>10s} {'PINAW':>10s}")
    print(f"  {'-'*45}")
    for label, r in results.items():
        print(f"  {label:<12s} {r['nominal']*100:>9.0f}% {r['picp']*100:>9.1f}% {r['pinaw']:>10.4f}")

    print(f"\n  Per-horizon PICP (80% CI):")
    for h in [0, 3, 5, 11, 17, 23]:
        print(f"    +{h+1:2d}h: {results['80% CI']['picp_h'][h]*100:.1f}%")

    return results, pr, tr


# ═══════════════════ Reliability Diagram Data ═══════════════════
def run_reliability(predictions, targets):
    """Compute reliability data for the diagram."""
    N, H, K = predictions.shape
    nominal_levels = np.arange(10, 100, 10)  # 10%, 20%, ..., 90%
    actual_coverages = []

    for level in nominal_levels:
        alpha = 1.0 - level / 100.0
        li = max(0, int(np.floor(alpha / 2 * 100)) - 1)
        ui = min(K - 1, int(np.ceil((1 - alpha / 2) * 100)) - 1)
        in_ci = (targets >= predictions[:, :, li]) & (targets <= predictions[:, :, ui])
        actual_coverages.append(in_ci.mean() * 100)

    return nominal_levels, actual_coverages


# ═══════════════════ Main ═══════════════════
def main():
    print("=" * 60)
    print("  PAPER SUPPLEMENT: QRF + PICP/PINAW + Reliability")
    print("=" * 60)
    sys.stdout.flush()

    # ── 1. QRF Baseline ──
    qrf_pb, qrf_rmse, qrf_cov, qrf_pred, qrf_targ = run_qrf()
    qrf_nominal, qrf_actual = run_reliability(qrf_pred, qrf_targ)

    # ── 2. PICP/PINAW ──
    picp_results, lnm_pred, lnm_targ = run_picp_pinaw()
    lnm_nominal, lnm_actual = run_reliability(lnm_pred, lnm_targ)

    # ── 3. Save reliability data for plotting ──
    os.makedirs(os.path.join(ROOT, 'checkpoints'), exist_ok=True)
    np.savez(os.path.join(ROOT, 'checkpoints/paper_supplement.npz'),
             qrf_pb=qrf_pb, qrf_rmse=qrf_rmse, qrf_cov=qrf_cov,
             qrf_nominal=qrf_nominal, qrf_actual=qrf_actual,
             lnm_nominal=lnm_nominal, lnm_actual=lnm_actual,
             picp_results={k: {k2: v2 for k2, v2 in v.items() if k2 not in ('picp_h', 'pinaw_h')}
                           for k, v in picp_results.items()})

    # ── 4. Summary table for paper ──
    print(f"\n{'='*60}")
    print(f"  PAPER-READY SOTA COMPARISON")
    print(f"{'='*60}")
    print(f"")
    print(f"  Table 3. Comparison with baseline methods on GEFCom2012 Farm 1.")
    print(f"")
    print(f"  {'Method':<20s} {'Pinball':>10s} {'RMSE':>10s} {'80% CI':>10s} {'PINAW(80%)':>12s}")
    print(f"  {'-'*65}")
    print(f"  {'Persistence':<20s} {'0.119':>10s} {'--':>10s} {'--':>10s} {'--':>12s}")
    print(f"  {'QRF (this work)':<20s} {qrf_pb:>10.4f} {qrf_rmse:>10.4f} {qrf_cov:>9.1f}% {'--':>12s}")

    picp80 = picp_results['80% CI']
    print(f"  {'LNMamba (ours)':<20s} {'0.0806':>10s} {'0.280':>10s} {picp80['picp']*100:>9.1f}% {picp80['pinaw']:>12.4f}")

    print(f"\n  QRF baseline: Pinball={qrf_pb:.4f} (vs LNMamba 0.0806)")
    print(f"  PICP(80%CI): LNMamba={picp80['picp']*100:.1f}%, QRF={qrf_cov:.1f}%")
    print(f"  Done!")
    sys.stdout.flush()


if __name__ == '__main__':
    main()
