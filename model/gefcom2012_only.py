"""
GEFCom2012-only LNMamba — 7 wind farms with full ECMWF NWP forecasts.

Each farm: ~13K hourly power measurements + NWP forecasts issued hourly
for +1 to +48h ahead. Total ~45K training windows across all 7 farms.

Features per time step: [u, v, ws, wd] × 5 lead times + cyclic time = 24 vars
"""
import sys,os,time,numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEVICE = torch.device('cuda')
QUANTILES = np.linspace(0.01, 0.99, 99)
SEQ, PRED = 168, 24
DIR = os.path.join(ROOT, 'GEFCOM2012/GEFCOM2012_Data/Wind')

LEAD_TIMES = [1, 3, 6, 12, 24]  # 5 forecast horizons


def build_farm_data(fid, stride=4):
    """Build time-series data for one GEFCom2012 wind farm.

    GEFCom2012 NWP: issued every 12h (00:00, 12:00), forecasts +1..+48h ahead.
    For each target hour, use the most recent NWP issue + appropriate hors.

    Returns: (data_array, n_feature_cols)
    """
    # Load power (training portion)
    pw = pd.read_csv(f'{DIR}/windpowermeasurements.csv')
    pw = pw[pw['usage'] == 'Training'].copy()
    pw['dt'] = pd.to_datetime(pw['date'].astype(str), format='%Y%m%d%H')
    pw_col = f'wp{fid}'
    pw = pw[['dt', pw_col]].rename(columns={pw_col: 'power'})
    pw = pw.sort_values('dt').reset_index(drop=True)

    # Load NWP and build lookup: (issue_time, hors) → (u, v, ws, wd)
    nwp = pd.read_csv(f'{DIR}/windforecasts_wf{fid}.csv')
    nwp['issue'] = pd.to_datetime(nwp['date'].astype(str), format='%Y%m%d%H')

    # Build a pivot: for each issue time, store all 48 horizon forecasts
    nwp_pivot = {}
    for _, row in nwp.iterrows():
        key = row['issue']
        if key not in nwp_pivot:
            nwp_pivot[key] = {}
        nwp_pivot[key][row['hors']] = (row['u'], row['v'], row['ws'], row['wd'])

    issue_times = sorted(nwp_pivot.keys())

    # For each target hour in power data, find the most recent NWP issue
    # and the corresponding hors
    features = []
    valid_idx = []
    feat_cols = []
    for lt in LEAD_TIMES:
        feat_cols.extend([f'u_{lt:02d}h', f'v_{lt:02d}h', f'ws_{lt:02d}h', f'wd_{lt:02d}h'])
    feat_cols.extend(['hour_sin', 'hour_cos', 'month_sin', 'month_cos'])

    for i, row in pw.iterrows():
        t = row['dt']
        # Find most recent NWP issue: use binary search
        # NWP issued at 00:00 or 12:00 before this hour
        # issue at t 00:00 if hour >= 0, else issue at (t-1 day) 12:00
        # Actually 12h forecasts: issue at 00:00 covers 01-48h ahead
        #                     issue at 12:00 covers 13-60h ahead
        # For hour H (0-23): if H < 12, most recent issue is previous day 12:00
        #                     if H >= 12, most recent issue is same day 00:00
        if t.hour < 12:
            # Issue was yesterday 12:00
            issue_t = t.replace(hour=12, minute=0, second=0) - pd.Timedelta(days=1)
        else:
            # Issue was today 00:00
            issue_t = t.replace(hour=0, minute=0, second=0)

        if issue_t not in nwp_pivot:
            continue

        # Calculate hors: hours from issue to target
        hors = int((t - issue_t).total_seconds() / 3600)

        # Get features for different lead times
        feats = []
        all_ok = True
        for lt in LEAD_TIMES:
            actual_hors = hors + (lt - 1)  # model uses lead=lt, actual NWP hors
            # Clamp to available range (1-48)
            use_hors = max(1, min(48, actual_hors))
            if use_hors in nwp_pivot[issue_t]:
                feats.extend(nwp_pivot[issue_t][use_hors])
            else:
                all_ok = False
                break

        if not all_ok:
            continue

        # Power value
        feats.append(np.sin(2*np.pi*t.hour/24))
        feats.append(np.cos(2*np.pi*t.hour/24))
        feats.append(np.sin(2*np.pi*t.month/12))
        feats.append(np.cos(2*np.pi*t.month/12))

        features.append(feats)
        valid_idx.append(i)

    feats_arr = np.array(features, dtype=np.float32)
    power_arr = pw['power'].values[valid_idx].astype(np.float32).clip(0, 1)

    # Standardize
    feats_norm = StandardScaler().fit_transform(feats_arr)
    data = np.concatenate([feats_norm, power_arr.reshape(-1, 1)], axis=1)

    print(f'  Farm {fid}: {len(data)} hours, {feats_arr.shape[1]} features, '
          f'{pw.dt.iloc[valid_idx[0]]}~{pw.dt.iloc[valid_idx[-1]]}')
    return data  # return numpy array; use data.shape[1] for n_vars


class WDS(Dataset):
    def __init__(self, data, stride):
        self.data = torch.FloatTensor(data); self.s = stride
        self.n = max(0, (len(data) - SEQ - PRED) // stride + 1)
    def __len__(self): return self.n
    def __getitem__(self, i):
        st = i * self.s
        return (self.data[st:st+SEQ].T, self.data[st+SEQ:st+SEQ+PRED, -1])


from nwp_model import NWPMamba, pinball_loss


def compute_picp_pinaw(predictions, targets, quantiles, intervals=None):
    """Compute PICP and PINAW for given prediction intervals.

    PICP (Prediction Interval Coverage Probability):
        Fraction of actual values falling within the prediction interval.
        Should be ≈ nominal coverage (e.g. ~0.90 for a 90% PI).

    PINAW (Prediction Interval Normalized Average Width):
        Mean interval width ÷ target range. Lower is better,
        but only meaningful when PICP meets the nominal level.

    Args:
        predictions: (N, H, Q) — quantile predictions
        targets:     (N, H) — actual values
        quantiles:   (Q,) — quantile levels
        intervals:   dict {label: (nominal, lo_q, hi_q)} or None for defaults

    Returns:
        dict: {label: {'nominal', 'picp', 'picp_per_h', 'pinaw', 'pinaw_per_h'}}
    """
    if intervals is None:
        intervals = {
            '80%': (0.80, 0.10, 0.90),
            '90%': (0.90, 0.05, 0.95),
            '95%': (0.95, 0.025, 0.975),
        }

    results = {}
    y_min = float(targets.min())
    y_max = float(targets.max())
    y_range = y_max - y_min

    for label, (nominal, lo_q, hi_q) in intervals.items():
        lo_idx = int(np.argmin(np.abs(quantiles - lo_q)))
        hi_idx = int(np.argmin(np.abs(quantiles - hi_q)))

        lo = predictions[:, :, lo_idx]  # (N, H)
        hi = predictions[:, :, hi_idx]  # (N, H)

        # PICP
        covered = (targets >= lo) & (targets <= hi)
        picp = float(covered.mean())
        picp_per_h = covered.mean(axis=0)  # (H,)

        # PINAW
        widths = hi - lo
        pinaw = float(widths.mean() / y_range)
        pinaw_per_h = widths.mean(axis=0) / y_range  # (H,)

        results[label] = {
            'nominal': nominal,
            'picp': picp,
            'picp_per_h': picp_per_h,
            'pinaw': pinaw,
            'pinaw_per_h': pinaw_per_h,
        }

    return results


def main():
    print('=' * 60)
    print('  GEFCom2012 LNMamba — 7 Wind Farms with ECMWF NWP')
    print(f'  {SEQ}h input → {PRED}h output, 99 quantiles')
    print('=' * 60)
    sys.stdout.flush()

    # Load all 7 farms
    all_ds = []
    nv = None
    for fid in range(1, 8):
        data, n_feats = build_farm_data(fid)
        if nv is None: nv = data.shape[1]
        T = len(data); te = int(T * 0.85)
        ds = WDS(data[:te], stride=4)
        all_ds.append(ds)
        print(f'    {len(ds):,} train windows')
        sys.stdout.flush()

    ds_full = torch.utils.data.ConcatDataset(all_ds)
    print(f'\n  Total: {len(ds_full):,} training windows from 7 farms')
    tl = DataLoader(ds_full, 48, shuffle=True, num_workers=0, pin_memory=True)
    print(f'  Batches/epoch: {len(tl)}')
    sys.stdout.flush()

    # Test: Farm 1 last 15%
    data1, _ = build_farm_data(1)
    T1 = len(data1); te1 = int(T1 * 0.85)
    test_ds = WDS(data1[te1:], stride=4)
    testl = DataLoader(test_ds, 48, shuffle=False, num_workers=0, pin_memory=True)
    print(f'  Test (Farm 1): {len(test_ds)} windows')

    # Model — use same architecture as v1
    model = NWPMamba(nv, d=64, nb=2, ds=16, pred=PRED, nq=99, use_lnn=True).to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters())
    print(f'\n  Params: {n_p:,}')

    # Train
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=15, T_mult=2, eta_min=1e-5)
    scl = torch.amp.GradScaler('cuda')
    qt = torch.tensor(QUANTILES, dtype=torch.float32, device=DEVICE)
    best_pb = float('inf'); best_state = None; hist = []
    EPOCHS = 40

    print(f'\n{"="*60}\n  TRAINING\n{"="*60}')
    sys.stdout.flush()

    for ep in range(1, EPOCHS + 1):
        t0 = time.time(); model.train(); tl_pb = 0.0
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE); opt.zero_grad()
            with torch.amp.autocast('cuda'): loss = pinball_loss(model(x), y, qt)
            scl.scale(loss).backward(); scl.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scl.step(opt); scl.update(); tl_pb += loss.item()
        sch.step()
        avg_pb = tl_pb / len(tl); hist.append(avg_pb); et = time.time() - t0
        star = ' ★' if avg_pb < best_pb else ''
        if avg_pb < best_pb: best_pb = avg_pb; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f'  E {ep:2d} pb={avg_pb:.4f} {et:.0f}s{star}')
        sys.stdout.flush()
        if ep >= 25 and ep - np.argmin(hist) >= 12: break

    if best_state: model.load_state_dict(best_state)

    # Test
    print(f'\n{"="*60}\n  FARM 1 TEST\n{"="*60}')
    model.eval(); preds, targs = [], []; total_pb = 0.0
    with torch.no_grad():
        for x, y in testl:
            x, y = x.to(DEVICE), y.to(DEVICE); out = model(x)
            total_pb += pinball_loss(out, y, qt).item()
            preds.append(out.cpu().numpy()); targs.append(y.cpu().numpy())
    pr = np.concatenate(preds); tr = np.concatenate(targs); test_pb = total_pb / len(testl)

    # Point metrics on [0,1] scale (power is already [0,1])
    p50 = pr[:, :, 49]; pf = p50.ravel(); tf = tr.ravel()
    mask = tf > 0.005  # filter near-zero
    rmse = np.sqrt(np.mean((pf[mask] - tf[mask])**2))
    mae = np.mean(np.abs(pf[mask] - tf[mask]))
    r2 = 1 - np.sum((tf[mask]-pf[mask])**2) / (np.sum((tf[mask]-np.mean(tf[mask]))**2) + 1e-8)

    print(f'  Pinball: {test_pb:.4f} | R2: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}')
    print('  Per-horizon Pinball:')
    for h in [0, 3, 5, 11, 17, 23]:
        er = torch.FloatTensor(tr[:, h]).unsqueeze(-1) - torch.FloatTensor(pr[:, h, :])
        pb_h = torch.maximum(torch.FloatTensor(QUANTILES)*er, (torch.FloatTensor(QUANTILES)-1)*er).mean().item()
        print(f'    +{h+1:2d}h: {pb_h:.4f}')

    # PICP / PINAW
    pi_results = compute_picp_pinaw(pr, tr, QUANTILES)
    print(f'\n  Prediction Interval Metrics:')
    header = f'  {"PI":>6s}  {"Nominal":>8s}  {"PICP":>8s}  {"PINAW":>8s}  {"ACE":>8s}'
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for label, r in pi_results.items():
        ace = r['picp'] - r['nominal']  # Average Coverage Error
        print(f'  {label:>6s}  {r["nominal"]:>8.3f}  {r["picp"]:>8.4f}  {r["pinaw"]:>8.4f}  {ace:>+8.4f}')

    # Per-horizon PICP/PINAW for 90% PI
    r90 = pi_results['90%']
    print(f'\n  Per-horizon 90% PI (PICP / PINAW):')
    for h in [0, 3, 5, 11, 17, 23]:
        print(f'    +{h+1:2d}h: PICP={r90["picp_per_h"][h]:.4f}  PINAW={r90["pinaw_per_h"][h]:.4f}')

    print(f'\n  === COMPARISON ===')
    print(f'  GEFCom2014 v1 (Zone 1, 3.5K, no cross-farm): Pinball=0.2069')
    print(f'  GEFCom2012 v1 (7 farms, {len(ds_full):,} windows): Pinball={test_pb:.4f}')
    print(f'  Params: {n_p:,}')
    print('  Done!')
    sys.stdout.flush()


if __name__ == '__main__':
    main()
