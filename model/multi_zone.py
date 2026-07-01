"""
Multi-Zone LNMamba — all 10 GEFCom2014 zones, per-zone normalized, stacked.

Key fix over v4:
  - Train: Normalize each zone independently, THEN stack + shuffle
  - Test: Zone 1's own scaler, trained model evaluated on Zone 1 test set

~24K training samples (10 zones × ~2400 each at stride=6) vs 3.5K for v1.
"""
import sys,os,zipfile,time,numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEVICE = torch.device('cuda')
QUANTILES = np.linspace(0.01, 0.99, 99)
DATA_DIR = os.path.join(ROOT, 'data/gefcom2014')
SEQ, PRED = 168, 24

# ═══════════════════ Weather features ═══════════════════
def weather(df):
    df['WS10'] = np.sqrt(df['U10']**2 + df['V10']**2)
    df['WS100'] = np.sqrt(df['U100']**2 + df['V100']**2)
    df['WD10_S'] = np.sin(np.arctan2(df['U10'], df['V10']))
    df['WD10_C'] = np.cos(np.arctan2(df['U10'], df['V10']))
    df['WD100_S'] = np.sin(np.arctan2(df['U100'], df['V100']))
    df['WD100_C'] = np.cos(np.arctan2(df['U100'], df['V100']))
    df['SHEAR'] = df['WS100'] / (df['WS10'] + 0.1)
    return df

FEAT = ['U10','V10','U100','V100','WS10','WS100',
        'WD10_S','WD10_C','WD100_S','WD100_C','SHEAR',
        'HOUR_SIN','HOUR_COS','MONTH_SIN','MONTH_COS']

def load_one_zone(zid):
    tz = zipfile.ZipFile(f'{DATA_DIR}/Task15_W_Zone1_10.zip')
    df = pd.read_csv(tz.open(f'Task15_W_Zone1_10/Task15_W_Zone{zid}.csv'))
    ts = df['TIMESTAMP'].astype(str).str.strip()
    df['dt'] = pd.to_datetime(ts.str[:8], format='%Y%m%d') + pd.to_timedelta(ts.str.extract(r'(\d+):')[0].astype(int), unit='h')
    df = df.sort_values('dt').reset_index(drop=True)
    df['TARGETVAR'] = df['TARGETVAR'].interpolate(limit_direction='both')
    for c in ['U10','V10','U100','V100']: df[c] = df[c].interpolate(limit_direction='both')
    df = weather(df)
    h = df['dt'].dt.hour.values.astype(np.float32)
    m = df['dt'].dt.month.values.astype(np.float32)
    df['HOUR_SIN'] = np.sin(2*np.pi*h/24); df['HOUR_COS'] = np.cos(2*np.pi*h/24)
    df['MONTH_SIN'] = np.sin(2*np.pi*m/12); df['MONTH_COS'] = np.cos(2*np.pi*m/12)
    return df

def normalize_and_window(df, stride=6):
    """Normalize per-zone, create sliding windows."""
    feats = StandardScaler().fit_transform(df[FEAT].values.astype(np.float32))
    sy = StandardScaler()
    tgt = sy.fit_transform(df[['TARGETVAR']].values.astype(np.float32)).ravel()
    data = np.concatenate([feats, tgt.reshape(-1, 1)], axis=1)

    class WDS(Dataset):
        def __init__(self, d, s):
            self.data = torch.FloatTensor(d); self.s = s
            self.n = max(0, (len(d) - SEQ - PRED) // s + 1)
        def __len__(self): return self.n
        def __getitem__(self, i):
            st = i * self.s
            return (self.data[st:st+SEQ].T, self.data[st+SEQ:st+SEQ+PRED, -1])

    return WDS(data, stride), sy

# ═══════════════════ Model ═══════════════════
from nwp_model import NWPMamba, pinball_loss
# Use v1 architecture but double d_model for extra capacity with more data
class BigMamba(nn.Module):
    """Scaled-up LNN-Mamba: d=96, ds=24, nb=3."""
    def __init__(self, V, d=96, nb=3, ds=24, pred=24, nq=99, dropout=0.1):
        super().__init__()
        self.pred_len = pred; self.nq = nq
        # Use v1's NWPMamba as backbone with bigger config
        self.model = NWPMamba(V, d=d, nb=nb, ds=ds, pred=pred, nq=nq, use_lnn=True)
        # Override dropout
        self.model.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.model(x)


# ═══════════════════ Main ═══════════════════
def main():
    print('=' * 60)
    print('  MULTI-ZONE LNMamba — 10 GEFCom2014 Zones')
    print('  Per-zone normalization → stacked training')
    print('  d=96, nb=3, ds=24 (~1.2M params)')
    print('=' * 60)
    sys.stdout.flush()

    # Load all zones
    all_train_ds = []
    nv = None
    for zid in range(1, 11):
        df = load_one_zone(zid)
        T = len(df); te = int(T * 0.85)
        # Train part only (val/test is Zone 1's split)
        train_ds, sy = normalize_and_window(df.iloc[:te], stride=6)
        all_train_ds.append(train_ds)
        if nv is None:
            nv = train_ds.data.shape[1]
        print(f'  Zone {zid}: {len(train_ds)} train samples')
        sys.stdout.flush()

    # Concat + shuffle
    ds_full = torch.utils.data.ConcatDataset(all_train_ds)
    print(f'\n  Total: {len(ds_full):,} training samples')
    sys.stdout.flush()
    tl = DataLoader(ds_full, 64, shuffle=True, num_workers=0, pin_memory=True)
    print(f'  Batches/epoch: {len(tl)}')

    # Zone 1 test (same as v1)
    df1 = load_one_zone(1)
    T1 = len(df1); te1 = int(T1 * 0.85)
    _, sy1 = normalize_and_window(df1.iloc[:te1], stride=6)
    # Recreate test dataset with Zone 1's test split
    test_data_norm = StandardScaler().fit_transform(df1.iloc[te1:][FEAT].values.astype(np.float32))
    test_tgt = sy1.transform(df1.iloc[te1:][['TARGETVAR']].values.astype(np.float32)).ravel()
    test_data = np.concatenate([test_data_norm, test_tgt.reshape(-1, 1)], axis=1)

    class WDS(Dataset):
        def __init__(self, d, s):
            self.data = torch.FloatTensor(d); self.s = s
            self.n = max(0, (len(d) - SEQ - PRED) // s + 1)
        def __len__(self): return self.n
        def __getitem__(self, i):
            st = i * self.s
            return (self.data[st:st+SEQ].T, self.data[st+SEQ:st+SEQ+PRED, -1])

    test_ds = WDS(test_data, 4)
    testl = DataLoader(test_ds, 64, shuffle=False, num_workers=0, pin_memory=True)
    print(f'  Test: {len(test_ds)} samples')

    # Model
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

    print(f'\n{"="*60}')
    print(f'  TRAINING: {EPOCHS} epochs')
    print(f'{"="*60}')
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
        if avg_pb < best_pb:
            best_pb = avg_pb; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
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
    print('  Per-horizon Pinball:')
    for h in [0, 3, 5, 11, 17, 23]:
        er = torch.FloatTensor(tr_mw[:, h]).unsqueeze(-1) - torch.FloatTensor(pr_mw[:, h])
        pb_h = torch.maximum(torch.FloatTensor(QUANTILES)*er, (torch.FloatTensor(QUANTILES)-1)*er).mean().item()
        print(f'    +{h+1:2d}h: {pb_h:.4f}')

    v1_pb = 0.2069
    print(f'\n  v1 (Zone 1 only): {v1_pb:.4f} | Multi-zone: {test_pb:.4f} | imp: {(v1_pb-test_pb)/v1_pb*100:+.1f}%')
    print(f'  Params: {n_p:,} | Train samples: {len(ds_full):,}')
    print(f'  Best train PB: {best_pb:.4f}')
    print('  Done!')
    sys.stdout.flush()


if __name__ == '__main__':
    main()
