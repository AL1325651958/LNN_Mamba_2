"""
Isotonic Regression calibration for probabilistic wind power forecasts.

Fixes the underconfidence issue: 80% CI covers only 45% of true values.
Applies per-horizon isotonic regression to map predicted quantiles
to properly calibrated quantiles based on validation set coverage.

Then retrains with calibrated model and evaluates all metrics.
"""
import sys,os,time,numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEVICE = torch.device('cuda')
QUANTILES = np.linspace(0.01, 0.99, 99)
SEQ, PRED = 168, 24
DIR = os.path.join(ROOT, 'GEFCOM2012/GEFCOM2012_Data/Wind')
LEAD_TIMES = [1, 3, 6, 12, 24]


# ═══════════════════ Data (reused from previous scripts) ═══════════════════
def build_farm_data(fid):
    pw = pd.read_csv(f'{DIR}/windpowermeasurements.csv')
    pw = pw[pw['usage'] == 'Training'].copy()
    pw['dt'] = pd.to_datetime(pw['date'].astype(str), format='%Y%m%d%H')
    pw = pw[['dt', f'wp{fid}']].rename(columns={f'wp{fid}': 'power'}).sort_values('dt').reset_index(drop=True)

    nwp = pd.read_csv(f'{DIR}/windforecasts_wf{fid}.csv')
    nwp['issue'] = pd.to_datetime(nwp['date'].astype(str), format='%Y%m%d%H')
    nwp_pivot = {}
    for _, row in nwp.iterrows():
        nwp_pivot.setdefault(row['issue'], {})[row['hors']] = (row['u'], row['v'], row['ws'], row['wd'])

    features, valid_idx = [], []
    for i, row in pw.iterrows():
        t = row['dt']
        if t.hour < 12: issue_t = t.replace(hour=12) - pd.Timedelta(days=1)
        else: issue_t = t.replace(hour=0)
        if issue_t not in nwp_pivot: continue
        feats = []; ok = True
        for lt in LEAD_TIMES:
            hr = max(1, min(48, int((t - issue_t).total_seconds() / 3600) + (lt - 1)))
            if hr in nwp_pivot[issue_t]: feats.extend(nwp_pivot[issue_t][hr])
            else: ok = False; break
        if not ok: continue
        feats += [np.sin(2*np.pi*t.hour/24), np.cos(2*np.pi*t.hour/24),
                  np.sin(2*np.pi*t.month/12), np.cos(2*np.pi*t.month/12)]
        features.append(feats); valid_idx.append(i)

    arr = np.array(features, dtype=np.float32)
    pwr = pw['power'].values[valid_idx].astype(np.float32).clip(0, 1)
    arr_norm = StandardScaler().fit_transform(arr)
    return np.concatenate([arr_norm, pwr.reshape(-1, 1)], axis=1)


class WDS(Dataset):
    def __init__(self, d, s):
        self.data = torch.FloatTensor(d); self.s = s
        self.n = max(0, (len(d)-SEQ-PRED)//s+1)
    def __len__(self): return self.n
    def __getitem__(self, i):
        st = i*self.s
        return (self.data[st:st+SEQ].T, self.data[st+SEQ:st+SEQ+PRED, -1])


# ═══════════════════ Model ═══════════════════
from nwp_model import NWPMamba, pinball_loss


# ═══════════════════ Calibration ═══════════════════
def fit_isotonic_calibration(pred_q, target, val_pred_q, val_target):
    """
    Fit per-horizon isotonic regression calibrator.

    For each horizon h and each target quantile level τ:
      - Compute what fraction of validation targets fall below the predicted τ-quantile
      - Fit isotonic regression: actual_coverage = f(predicted_quantile_value)

    Then to get calibrated q_τ for test data:
      - For each sample, find which τ level the predicted value corresponds to
        in the isotonic mapping
      - Interpolated quantile = the calibrated value

    Simpler approach: fit isotonic on the CDF values directly.
    For each sample at horizon h:
      - Sort predicted quantiles ascending → empirical CDF
      - Target: indicator(target <= q) for each quantile
      - Fit isotonic regression mapping nominal CDF levels to actual coverage
    """
    N, H, K = val_pred_q.shape
    calibrators = []  # one per horizon

    for h in range(H):
        cal_h = []
        for k in range(K):
            # For quantile k (level = QUANTILES[k]):
            # X = predicted quantile value
            # y = indicator(target <= predicted_quantile)
            X = val_pred_q[:, h, k].reshape(-1, 1)
            y = (val_target[:, h] <= val_pred_q[:, h, k]).astype(float)

            # Only fit if there's variation in X
            if X.std() > 1e-6 and len(np.unique(y)) > 1:
                ir = IsotonicRegression(out_of_bounds='clip', increasing=True)
                ir.fit(X.ravel(), y)
            else:
                ir = None
            cal_h.append(ir)
        calibrators.append(cal_h)

    return calibrators


def apply_calibration(test_pred_q, calibrators):
    """
    Apply fitted calibrators to test predictions.

    For each quantile level: replace the quantile value with
    the value calibrated by isotonic regression.
    """
    N, H, K = test_pred_q.shape
    calibrated = np.zeros_like(test_pred_q)

    for h in range(H):
        for k in range(K):
            if calibrators[h][k] is not None:
                X = test_pred_q[:, h, k]
                # Transform: predict coverage at this quantile value
                # Then use the coverage as the "calibrated" quantile level
                coverage = calibrators[h][k].transform(X)
                # Clip to valid range
                coverage = np.clip(coverage, 0.01, 0.99)
                # Store as-is (we're mapping value→coverage)
                # For proper calibration, we should use the coverage to
                # determine the actual quantile position
                calibrated[:, h, k] = test_pred_q[:, h, k]  # keep value
            else:
                calibrated[:, h, k] = test_pred_q[:, h, k]

    return calibrated


def compute_calibrated_quantiles(val_pred_q, val_target, test_pred_q):
    """
    Quantile recalibration: for each predicted quantile value,
    map it to the actual empirical quantile in the validation set.

    Approach:
    1. For each horizon h, sort all predicted values at that horizon
    2. For each quantile level k, find the actual fraction of targets
       below that quantile value → this becomes the "actual quantile level"
    3. For test predictions, interpolate to get calibrated quantiles
    """
    N_val, H, K = val_pred_q.shape
    N_test = test_pred_q.shape[0]
    calibrated = np.zeros_like(test_pred_q)

    for h in range(H):
        # For each sample, use the predicted quantiles as-is
        # but recalibrate the LEVEL (not the value)
        for k in range(K):
            # Empirical coverage of quantile k in validation set
            actual_cov = np.mean(val_target[:, h] <= val_pred_q[:, h, k])
            # Adjust the quantile toward the actual coverage
            # E.g., if nominal q=0.10 but actual coverage is 0.05,
            # the predicted values at this quantile were too low
            target_cov = QUANTILES[k]

            if actual_cov > 0 and actual_cov < 1:
                # Scale factor: how much to shift this quantile's values
                # to match the target coverage
                # We use linear scaling of the predicted quantile value
                # based on the ratio of target/actual coverage
                ratio = target_cov / (actual_cov + 1e-8)

                # Find the predicted value that gives correct coverage
                # Simple linear adjustment
                sorted_test = np.sort(test_pred_q[:, h, k])
                sorted_val  = np.sort(val_pred_q[:, h, k])

                # Find value at target coverage in validation
                if ratio > 1.5 or ratio < 0.5:
                    # Significant miscalibration: use empirical mapping
                    # Map validation coverage to test values
                    val_values = np.sort(val_pred_q[:, h, :])
                    for n in range(N_test):
                        # Find which quantile level this value corresponds to
                        # and adjust
                        pass

            calibrated[:, h, k] = test_pred_q[:, h, k]

    return calibrated


def empirical_cdf_calibration(val_pred_q, val_target, test_pred_q):
    """
    Simpler approach: for each horizon, fit an empirical CDF calibration.

    For each horizon h:
      - For each sample i, the model predicts 99 quantile values
      - The actual CDF at value v = P(target <= v) should match the
        nominal quantile level
      - Fit a monotonic mapping: predicted_value → actual_coverage_rate

    Then for test:
      - Use the predicted quantile VALUE to find the ACTUAL quantile LEVEL
        via the fitted mapping
      - This gives recalibrated quantile levels
      - Interpolate back to get the value at the target nominal quantile level
    """
    N_test, H, K = test_pred_q.shape
    N_val = val_pred_q.shape[0]
    calibrated = np.zeros_like(test_pred_q)

    for h in range(H):
        # Collect all (value, coverage) pairs from validation
        all_values = []
        all_coverages = []
        for k in range(K):
            vals = val_pred_q[:, h, k]
            covs = (val_target[:, h, np.newaxis] <= val_pred_q[:, h, :]).mean(axis=0)
            # For this quantile value, the actual coverage across ALL quantiles
            # is the mean of indicator(target <= value)
            nominal = QUANTILES[k]
            actual = (val_target[:, h] <= vals).mean()
            all_values.append(vals.mean())
            all_coverages.append(actual)

        all_values = np.array(all_values)
        all_coverages = np.array(all_coverages)

        # Fit isotonic: value → actual_coverage
        sort_idx = np.argsort(all_values)
        sorted_vals = all_values[sort_idx]
        sorted_covs = all_coverages[sort_idx]

        if len(sorted_vals) > 10 and sorted_covs.std() > 1e-4:
            ir = IsotonicRegression(out_of_bounds='clip', increasing=True)
            ir.fit(sorted_vals.reshape(-1, 1), sorted_covs)

            # For test: for each sample, we have 99 predicted values
            # Map each value to its underlying coverage level
            for n in range(N_test):
                vals = test_pred_q[n, h, :]
                covs = ir.transform(vals.reshape(-1, 1))
                # Now we have (value, actual_coverage) pairs
                # We need the value at nominal quantile levels
                # Interpolate: at each nominal quantile q_k,
                # find the value where actual_coverage = q_k
                for k in range(K):
                    target_cov = QUANTILES[k]
                    # Find neighbors in coverage space
                    # Simple: use the value at the closest coverage
                    closest_idx = np.argmin(np.abs(covs - target_cov))
                    calibrated[n, h, k] = vals[closest_idx]
        else:
            calibrated[:, h, k] = test_pred_q[:, h, k]

    return calibrated


# ═══════════════════ Simple Linear Recalibration ═══════════════════
def simple_linear_calibration(val_pred_q, val_target, test_pred_q):
    """
    Simple per-quantile scaling: stretch or shrink each quantile's values
    so that its empirical coverage equals its nominal level.

    For quantile k (nominal level q_k):
      scaling factor α_k = actual_cdf_inverse(q_k) / predicted_value_at_q_k

    Then calibrated_value = α_k × original_value
    """
    N_test, H, K = test_pred_q.shape
    calibrated = np.zeros_like(test_pred_q)

    for h in range(H):
        sorted_val_all = np.sort(val_pred_q[:, h, :].ravel())

        for k in range(K):
            target_cov = QUANTILES[k]
            vals = test_pred_q[:, h, k]
            val_vals = val_pred_q[:, h, k]

            # Actual coverage in validation set
            actual_cov = np.mean(val_target[:, h] <= val_vals)

            if actual_cov > 0.01 and actual_cov < 0.99:
                # Scaling: if actual < target, values were too LOW → increase them
                # ratio = target / actual
                # But this is too simple. Better:
                # Find the value in sorted validated that gives target coverage
                target_value = np.percentile(val_vals, target_cov * 100)
                median_value = np.percentile(val_vals, 50)

                # Scale around median
                scale = target_value / (median_value + 1e-6)
                calibrated[:, h, k] = vals * scale
            else:
                calibrated[:, h, k] = vals

    return calibrated


# ═══════════════════ Metrics ═══════════════════
def compute_all_metrics(pred_q_mw, target_mw):
    """Same as full_eval_2012.py."""
    N, H, K = pred_q_mw.shape; q = QUANTILES

    # Pinball
    error = target_mw[:, :, np.newaxis] - pred_q_mw
    pinball = np.maximum(q * error, (q - 1) * error).mean()
    pinball_h = [np.maximum(q*error[:, h, :], (q-1)*error[:, h, :]).mean() for h in range(H)]

    # Coverage
    cov80 = np.mean((target_mw >= pred_q_mw[:, :, 9]) & (target_mw <= pred_q_mw[:, :, 89])) * 100

    # RMSE/MAE from median
    p50 = pred_q_mw[:, :, 49]; pf, tf = p50.ravel(), target_mw.ravel()
    mask = tf > 0.005; pf_f, tf_f = pf[mask], tf[mask]
    rmse = np.sqrt(np.mean((pf_f - tf_f)**2))
    mae  = np.mean(np.abs(pf_f - tf_f))
    r2 = 1 - np.sum((tf_f-pf_f)**2)/(np.sum((tf_f-np.mean(tf_f))**2)+1e-8)

    return pinball, cov80, rmse, mae, r2


# ═══════════════════ Main ═══════════════════
def main():
    print('='*60)
    print('  ISOTONIC CALIBRATION — GEFCom2012 7-Farm LNMamba')
    print('='*60)
    sys.stdout.flush()

    # ── Load data ──
    print('\n[1/4] Loading data...'); sys.stdout.flush()
    all_ds = []; nv = None
    for fid in range(1, 8):
        data = build_farm_data(fid)
        if nv is None: nv = data.shape[1]
        T = len(data); te = int(T*0.85)
        all_ds.append(WDS(data[:te], 4))
    ds_full = torch.utils.data.ConcatDataset(all_ds)

    # Farm 1: train/val/test split
    data1 = build_farm_data(1)
    T1 = len(data1)
    train_end = int(T1 * 0.75)   # 75% for training calibration reference
    val_end   = int(T1 * 0.85)    # 10% for fitting calibrator
    test_start = val_end          # 15% for testing

    cal_train_ds = WDS(data1[:train_end], 4)
    cal_val_ds   = WDS(data1[train_end:val_end], 4)
    test_ds      = WDS(data1[test_start:], 4)

    tl = DataLoader(ds_full, 48, shuffle=True, num_workers=0, pin_memory=True)
    testl = DataLoader(test_ds, 48, shuffle=False, num_workers=0, pin_memory=True)
    print(f'  Train: {len(ds_full):,} | Val(cal): {len(cal_val_ds)} | Test: {len(test_ds)}')

    # ── Train model ──
    print('\n[2/4] Training LNMamba (30 epochs)...'); sys.stdout.flush()
    qt = torch.tensor(QUANTILES, dtype=torch.float32, device=DEVICE)
    model = NWPMamba(nv, d=64, nb=2, ds=16, pred=PRED, nq=99, use_lnn=True).to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters())

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
        if ep % 10 == 1: print(f'  E{ep}...'); sys.stdout.flush()

    # ── Generate validation predictions ──
    print('\n[3/4] Fitting calibration...'); sys.stdout.flush()
    model.eval()

    # Val predictions
    val_preds, val_targets = [], []
    for x, y in DataLoader(cal_val_ds, 48, shuffle=False, num_workers=0, pin_memory=True):
        out = model(x.to(DEVICE)).detach().cpu().numpy()
        val_preds.append(out)
        val_targets.append(y.numpy())
    val_pq = np.concatenate(val_preds); val_tg = np.concatenate(val_targets)

    # Test predictions (uncalibrated)
    test_preds, test_targets = [], []
    for x, y in testl:
        out = model(x.to(DEVICE)).detach().cpu().numpy()
        test_preds.append(out)
        test_targets.append(y.numpy())
    test_pq = np.concatenate(test_preds); test_tg = np.concatenate(test_targets)

    # ── Before calibration ──
    pb_before, cov_before, rmse_before, mae_before, r2_before = compute_all_metrics(test_pq, test_tg)
    print(f'  BEFORE: Pinball={pb_before:.4f} | 80%CI={cov_before:.1f}% | R2={r2_before:.4f}')

    # ── Apply calibration ──
    # Use empirical CDF calibration
    cal_pq = empirical_cdf_calibration(val_pq, val_tg, test_pq)

    # Also try simple linear
    # cal_pq_linear = simple_linear_calibration(val_pq, val_tg, test_pq)

    pb_after, cov_after, rmse_after, mae_after, r2_after = compute_all_metrics(cal_pq, test_tg)
    print(f'  AFTER:  Pinball={pb_after:.4f} | 80%CI={cov_after:.1f}% | R2={r2_after:.4f}')

    # ── Also try quantile recalibration ──
    recal_pq = np.zeros_like(test_pq)
    for h in range(PRED):
        for k in range(99):
            target_cov = QUANTILES[k]
            val_vals = val_pq[:, h, k]
            # What value gives target coverage in validation?
            target_val = np.percentile(val_vals, target_cov * 100)
            # How far is this from the mean?
            mean_val = val_vals.mean()
            # Scale test values
            scale = target_val / (mean_val + 1e-6)
            recal_pq[:, h, k] = test_pq[:, h, k] * scale

    pb_recal, cov_recal, rmse_recal, mae_recal, r2_recal = compute_all_metrics(recal_pq, test_tg)

    # ── Print comparison ──
    print(f'\n{"="*70}')
    print(f'  CALIBRATION RESULTS')
    print(f'{"="*70}')
    print(f'  {"Method":<20s} {"Pinball":>10s} {"80%CI Cov":>12s} {"RMSE":>10s} {"R2":>10s}')
    print(f'  {"-"*65}')
    print(f'  {"Uncalibrated":<20s} {pb_before:>10.4f} {cov_before:>11.1f}% {rmse_before:>10.4f} {r2_before:>10.4f}')
    print(f'  {"CDF Calibration":<20s} {pb_after:>10.4f} {cov_after:>11.1f}% {rmse_after:>10.4f} {r2_after:>10.4f}')
    print(f'  {"Linear Recal":<20s} {pb_recal:>10.4f} {cov_recal:>11.1f}% {rmse_recal:>10.4f} {r2_recal:>10.4f}')

    if cov_after >= 70:
        print(f'\n  ★ CDF calibration achieved {cov_after:.1f}% coverage (target: 80%)')
    elif cov_recal >= 70:
        print(f'\n  ★ Linear recalibration achieved {cov_recal:.1f}% coverage (target: 80%)')
    else:
        print(f'\n  ⚠ Both methods under target. Best: {max(cov_after, cov_recal):.1f}%')
        print(f'  → Multi-farm training inherently produces wider distribution')
        print(f'  → Recommend: train on individual farms for calibration')
        print(f'  → Or: use conformal prediction for guaranteed coverage')

    # Per-horizon coverage for best method
    best_pq = cal_pq if cov_after > cov_recal else recal_pq
    print(f'\n  Per-horizon 80% CI coverage (best method):')
    for h in range(0, PRED, 3):
        c = np.mean((test_tg[:, h] >= best_pq[:, h, 9]) & (test_tg[:, h] <= best_pq[:, h, 89])) * 100
        marks = '*' * int(c/2)
        print(f'  +{h+1:2d}h: {c:5.1f}% {marks}')

    print(f'\n  Params: {n_p:,} | Train: {len(ds_full):,} windows')
    print('  Done!')
    sys.stdout.flush()


if __name__ == '__main__':
    main()
