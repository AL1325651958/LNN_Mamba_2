"""
Combined GEFCom2012 (7 farms) + GEFCom2014 (10 zones) — 17-farm unified pipeline.

GEFCom2012 NWP: Forecasts issued each hour for +1 to +48h ahead (u, v, ws, wd)
GEFCom2014 NWP: ECMWF U10/V10/U100/V100 at measurement time

Unified: all power in [0,1], all NWP features standardized per farm,
windows stacked across all 17 farms.
"""
import sys,os,time,numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd, zipfile
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEVICE = torch.device('cuda')
QUANTILES = np.linspace(0.01, 0.99, 99)
SEQ, PRED = 168, 24
GEF2012_DIR = os.path.join(ROOT, 'GEFCOM2012/GEFCOM2012_Data/Wind')
GEF2014_DIR = os.path.join(ROOT, 'data/gefcom2014')

# ═══════════════════════════════════════
# GEFCom2012: 7 farms with rolling NWP forecasts
# ═══════════════════════════════════════
def load_gefcom2012_farm(fid):
    """
    Load one GEFCom2012 wind farm with NWP forecasts.
    Returns (data_array, n_vars) where data_array = (T, features+1).
    data_array[:, -1] = power, data_array[:, :-1] = NWP features.
    """
    # Power
    pw = pd.read_csv(f'{GEF2012_DIR}/windpowermeasurements.csv')
    pw = pw[pw['usage'] == 'Training'].copy()
    pw['dt'] = pd.to_datetime(pw['date'].astype(str), format='%Y%m%d%H')
    pw_cols = {'wp1': 'wp1', 'wp2': 'wp2', 'wp3': 'wp3', 'wp4': 'wp4',
               'wp5': 'wp5', 'wp6': 'wp6', 'wp7': 'wp7'}
    pw_val = pw[[pw_cols[f'wp{fid}']]].values.astype(np.float32).ravel()
    pw_dt = pw['dt'].values

    # NWP forecasts: pivot to wide for each issue time
    nwp = pd.read_csv(f'{GEF2012_DIR}/windforecasts_wf{fid}.csv')
    nwp['dt_issue'] = pd.to_datetime(nwp['date'].astype(str), format='%Y%m%d%H')
    nwp['dt_target'] = nwp['dt_issue'] + pd.to_timedelta(nwp['hors'], unit='h')

    # For each target hour, use NWP with hors=1 (most recent 1h forecast)
    # This gives us NWP "observations" at each target hour
    nwp_h1 = nwp[nwp['hors'] == 1][['dt_target', 'u', 'v', 'ws', 'wd']].copy()
    nwp_h1.columns = ['dt', 'u', 'v', 'ws', 'wd']

    # Also add forecast features for longer horizons as auxiliary
    # hors=1, 3, 6, 12, 24 → 5×4 = 20 NWP features per time step
    lead_times = [1, 3, 6, 12, 24]

    # Merge all lead time features into per-target-time rows
    nwp_merged = None
    for lt in lead_times:
        lt_nwp = nwp[nwp['hors'] == lt][['dt_target', 'u', 'v', 'ws', 'wd']].copy()
        lt_nwp.columns = ['dt', f'u_h{lt}', f'v_h{lt}', f'ws_h{lt}', f'wd_h{lt}']
        if nwp_merged is None:
            nwp_merged = lt_nwp
        else:
            nwp_merged = nwp_merged.merge(lt_nwp, on='dt', how='inner')

    # Merge with power
    pw_df = pd.DataFrame({'dt': pw_dt, 'power': pw_val})
    merged = pw_df.merge(nwp_merged, on='dt', how='inner')
    merged = merged.sort_values('dt').reset_index(drop=True)

    nwp_cols = [c for c in merged.columns if c not in ['dt', 'power']]
    print(f'  Farm {fid}: {len(merged)} rows, {len(nwp_cols)} NWP features')
    print(f'    Time: {merged.dt.min()} ~ {merged.dt.max()}')

    # Scale targets to [0,1] (already in this range from GEFCom2012)
    power = merged['power'].values.astype(np.float32).clip(0, 1)
    features = merged[nwp_cols].values.astype(np.float32)

    # Standardize features
    feats_norm = StandardScaler().fit_transform(features)

    data = np.concatenate([feats_norm, power.reshape(-1, 1)], axis=1)
    return data


# ═══════════════════════════════════════
# GEFCom2014: 10 zones (reuse existing pipeline)
# ═══════════════════════════════════════
FEAT14 = ['U10','V10','U100','V100','WS10','WS100',
          'WD10_S','WD10_C','WD100_S','WD100_C','SHEAR',
          'HOUR_SIN','HOUR_COS','MONTH_SIN','MONTH_COS']

def weather14(df):
    df['WS10'] = np.sqrt(df['U10']**2 + df['V10']**2)
    df['WS100'] = np.sqrt(df['U100']**2 + df['V100']**2)
    df['WD10_S'] = np.sin(np.arctan2(df['U10'], df['V10']))
    df['WD10_C'] = np.cos(np.arctan2(df['U10'], df['V10']))
    df['WD100_S'] = np.sin(np.arctan2(df['U100'], df['V100']))
    df['WD100_C'] = np.cos(np.arctan2(df['U100'], df['V100']))
    df['SHEAR'] = df['WS100'] / (df['WS10'] + 0.1)
    return df

def load_gefcom2014_zone(zid):
    tz = zipfile.ZipFile(f'{GEF2014_DIR}/Task15_W_Zone1_10.zip')
    df = pd.read_csv(tz.open(f'Task15_W_Zone1_10/Task15_W_Zone{zid}.csv'))
    ts = df['TIMESTAMP'].astype(str).str.strip()
    df['dt'] = pd.to_datetime(ts.str[:8], format='%Y%m%d') + pd.to_timedelta(ts.str.extract(r'(\d+):')[0].astype(int), unit='h')
    df = df.sort_values('dt').reset_index(drop=True)
    df['TARGETVAR'] = df['TARGETVAR'].interpolate(limit_direction='both')
    for c in ['U10','V10','U100','V100']: df[c] = df[c].interpolate(limit_direction='both')
    df = weather14(df)
    h = df['dt'].dt.hour.values.astype(np.float32); m = df['dt'].dt.month.values.astype(np.float32)
    df['HOUR_SIN'] = np.sin(2*np.pi*h/24); df['HOUR_COS'] = np.cos(2*np.pi*h/24)
    df['MONTH_SIN'] = np.sin(2*np.pi*m/12); df['MONTH_COS'] = np.cos(2*np.pi*m/12)

    # Standardize features, clip power to [0,1]
    feats = StandardScaler().fit_transform(df[FEAT14].values.astype(np.float32))
    power = df['TARGETVAR'].values.astype(np.float32).clip(0, 1)

    return np.concatenate([feats, power.reshape(-1, 1)], axis=1)

# ═══════════════════════════════════════
# Sliding window dataset
# ═══════════════════════════════════════
class WDS(Dataset):
    def __init__(self, data, stride=4):
        self.data = torch.FloatTensor(data); self.s = stride
        self.n = max(0, (len(data) - SEQ - PRED) // stride + 1)
    def __len__(self): return self.n
    def __getitem__(self, i):
        st = i * self.s
        return (self.data[st:st+SEQ].T, self.data[st+SEQ:st+SEQ+PRED, -1])

# ═══════════════════════════════════════
# Model
# ═══════════════════════════════════════
from nwp_model import NWPMamba, pinball_loss

# ═══════════════════════════════════════
# Main
# ═══════════════════════════════════════
def main():
    print('=' * 60)
    print('  COMBINED GEFCom2012 + GEFCom2014 — 17 Farms')
    print('  d=64, nb=2, ds=16, stride=4')
    print('=' * 60)
    sys.stdout.flush()

    all_datasets = []

    # Load GEFCom2012 (7 farms)
    print('\n--- GEFCom2012 (7 farms, hourly, 2009-2010) ---')
    for fid in range(1, 8):
        data = load_gefcom2012_farm(fid)
        T = len(data); te = int(T * 0.85)
        train_data = data[:te]
        ds = WDS(train_data, stride=4)
        all_datasets.append(ds)
        print(f'    {len(ds):,} train windows')
        sys.stdout.flush()

    # Load GEFCom2014 (10 zones)
    print('\n--- GEFCom2014 (10 zones, hourly, 2012-2013) ---')
    for zid in range(1, 11):
        data = load_gefcom2014_zone(zid)
        T = len(data); te = int(T * 0.85)
        # Pad to match feature count if needed
        all_datasets.append(WDS(data[:te], stride=4))
        print(f'    Zone {zid}: {WDS(data[:te], 4).n:,} train windows')
        sys.stdout.flush()

    # Pad all datasets to same feature count
    max_vars = max(ds.data.shape[1] for ds in all_datasets)
    print(f'\n  Max vars: {max_vars}')
    for i, ds in enumerate(all_datasets):
        nv = ds.data.shape[1]
        if nv < max_vars:
            pad = torch.zeros(len(ds.data), max_vars - nv)
            ds.data = torch.cat([ds.data, pad], dim=1)
            all_datasets[i] = ds

    # Concat all
    ds_full = torch.utils.data.ConcatDataset(all_datasets)
    print(f'\n  Total: {len(ds_full):,} training windows')
    sys.stdout.flush()

    tl = DataLoader(ds_full, 64, shuffle=True, num_workers=0, pin_memory=True)
    print(f'  Batches/epoch: {len(tl)}')

    # Zone 1 test (same as v1)
    df1_data = load_gefcom2014_zone(1)
    T1 = len(df1_data); te1 = int(T1 * 0.85)
    test_data = df1_data[te1:]
    # Pad test data too
    nv_test = test_data.shape[1]
    if nv_test < max_vars:
        test_data = np.concatenate([test_data, np.zeros((len(test_data), max_vars - nv_test), dtype=np.float32)], axis=1)
    test_ds = WDS(test_data, stride=4)
    testl = DataLoader(test_ds, 64, shuffle=False, num_workers=0, pin_memory=True)
    print(f'  Test: {len(test_ds)} windows')

    # Model
    model = NWPMamba(max_vars, d=64, nb=2, ds=16, pred=PRED, nq=99, use_lnn=True).to(DEVICE)
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

    # Zone 1 scaler for inverse transform (re-use GEFCom2014 scaler)
    _, _, _, sy1, _, _ = reload_zone1_scaler()

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
    print(f'\n{"="*60}\n  ZONE 1 TEST\n{"="*60}')
    model.eval(); preds, targs = [], []; total_pb = 0.0
    with torch.no_grad():
        for x, y in testl:
            x, y = x.to(DEVICE), y.to(DEVICE); out = model(x)
            total_pb += pinball_loss(out, y, qt).item()
            preds.append(out.cpu().numpy()); targs.append(y.cpu().numpy())
    pr = np.concatenate(preds); tr = np.concatenate(targs); test_pb = total_pb / len(testl)
    sh = pr.shape
    pr_mw = sy1.inverse_transform(pr.reshape(-1, sh[2])).reshape(sh)
    tr_mw = sy1.inverse_transform(tr.reshape(-1, 1)).reshape(tr.shape)
    p50 = pr_mw[:, :, 49]; pf = p50.ravel(); tf = tr_mw.ravel(); mask = tf > 0.001
    rmse = np.sqrt(np.mean((pf[mask] - tf[mask])**2))
    mae = np.mean(np.abs(pf[mask] - tf[mask]))
    r2 = 1 - np.sum((tf[mask]-pf[mask])**2) / (np.sum((tf[mask]-np.mean(tf[mask]))**2) + 1e-8)

    print(f'  Pinball: {test_pb:.4f} | R2: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}')
    print('  Per-horizon:')
    for h in [0,3,5,11,17,23]:
        er = torch.FloatTensor(tr_mw[:,h]).unsqueeze(-1)-torch.FloatTensor(pr_mw[:,h])
        pb_h = torch.maximum(torch.FloatTensor(QUANTILES)*er,(torch.FloatTensor(QUANTILES)-1)*er).mean().item()
        print(f'    +{h+1:2d}h: {pb_h:.4f}')

    v1_pb = 0.2069
    print(f'\n  v1 (Zone 1 only, 3.5K): 0.2069')
    print(f'  17-farm combined ({len(ds_full):,}): {test_pb:.4f}')
    print(f'  imp: {(v1_pb-test_pb)/v1_pb*100:+.1f}%')
    print(f'  Params: {n_p:,}')
    print('  Done!')
    sys.stdout.flush()


def reload_zone1_scaler():
    df = pd.read_csv(zipfile.ZipFile(f'{GEF2014_DIR}/Task15_W_Zone1_10.zip')
                     .open('Task15_W_Zone1_10/Task15_W_Zone1.csv'))
    ts = df['TIMESTAMP'].astype(str).str.strip()
    df['dt'] = pd.to_datetime(ts.str[:8], format='%Y%m%d') + pd.to_timedelta(ts.str.extract(r'(\d+):')[0].astype(int), unit='h')
    df = df.sort_values('dt').reset_index(drop=True)
    df['TARGETVAR'] = df['TARGETVAR'].interpolate(limit_direction='both')
    for c in ['U10','V10','U100','V100']: df[c] = df[c].interpolate(limit_direction='both')
    df = weather14(df)
    h = df['dt'].dt.hour.values.astype(np.float32); m = df['dt'].dt.month.values.astype(np.float32)
    df['HOUR_SIN'] = np.sin(2*np.pi*h/24); df['HOUR_COS'] = np.cos(2*np.pi*h/24)
    df['MONTH_SIN'] = np.sin(2*np.pi*m/12); df['MONTH_COS'] = np.cos(2*np.pi*m/12)
    sy = StandardScaler(); sy.fit(df[['TARGETVAR']].values.astype(np.float32))
    return None, None, None, sy, None, None


if __name__ == '__main__':
    main()
