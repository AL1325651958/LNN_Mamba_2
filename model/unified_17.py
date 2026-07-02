"""
UNIFIED 17-SITE LNMamba — GEFCom2012(7) + GEFCom2014(10).

Per-site StandardScaler + power clipped to [0,1], padded to max features.
Train on ~50K windows, test on GEFCom2012 Farm 1.
"""
import sys,os,time,numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd, zipfile
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEVICE = torch.device('cuda')
QUANTILES = np.linspace(0.01, 0.99, 99)
SEQ, PRED = 168, 24
DIR12 = os.path.join(ROOT, 'GEFCOM2012/GEFCOM2012_Data/Wind')
DIR14 = os.path.join(ROOT, 'data/gefcom2014')
LT = [1, 3, 6, 12, 24]

F14 = ['U10','V10','U100','V100','WS10','WS100','WD10_S','WD10_C',
       'WD100_S','WD100_C','SHEAR','HOUR_SIN','HOUR_COS','MONTH_SIN','MONTH_COS']


def build_gef12(fid):
    pw = pd.read_csv(f'{DIR12}/windpowermeasurements.csv')
    pw = pw[pw['usage'] == 'Training'].copy()
    pw['dt'] = pd.to_datetime(pw['date'].astype(str), format='%Y%m%d%H')
    pw = pw[['dt', f'wp{fid}']].rename(columns={f'wp{fid}': 'power'})
    pw = pw.sort_values('dt').reset_index(drop=True)
    nwp = pd.read_csv(f'{DIR12}/windforecasts_wf{fid}.csv')
    nwp['issue'] = pd.to_datetime(nwp['date'].astype(str), format='%Y%m%d%H')
    np_p = {}
    for _, r in nwp.iterrows():
        np_p.setdefault(r['issue'], {})[r['hors']] = (r['u'], r['v'], r['ws'], r['wd'])
    feats, idxs = [], []
    for i, r in pw.iterrows():
        t = r['dt']
        it = t.replace(hour=0) if t.hour >= 12 else t.replace(hour=12) - pd.Timedelta(days=1)
        if it not in np_p:
            continue
        f = []; ok = True
        for lt in LT:
            hr = max(1, min(48, int((t - it).total_seconds() / 3600) + (lt - 1)))
            if hr in np_p[it]:
                f.extend(np_p[it][hr])
            else:
                ok = False; break
        if not ok:
            continue
        f += [np.sin(2*np.pi*t.hour/24), np.cos(2*np.pi*t.hour/24),
              np.sin(2*np.pi*t.month/12), np.cos(2*np.pi*t.month/12)]
        feats.append(f); idxs.append(i)
    arr = np.array(feats, dtype=np.float32)
    pwr = pw['power'].values[idxs].astype(np.float32).clip(0, 1)
    arr_n = StandardScaler().fit_transform(arr)
    return np.concatenate([arr_n, pwr.reshape(-1, 1)], axis=1)


def wx14(df):
    df['WS10'] = np.sqrt(df['U10']**2 + df['V10']**2)
    df['WS100'] = np.sqrt(df['U100']**2 + df['V100']**2)
    df['WD10_S'] = np.sin(np.arctan2(df['U10'], df['V10']))
    df['WD10_C'] = np.cos(np.arctan2(df['U10'], df['V10']))
    df['WD100_S'] = np.sin(np.arctan2(df['U100'], df['V100']))
    df['WD100_C'] = np.cos(np.arctan2(df['U100'], df['V100']))
    df['SHEAR'] = df['WS100'] / (df['WS10'] + 0.1)
    return df


def build_gef14(zid):
    tz = zipfile.ZipFile(f'{DIR14}/Task15_W_Zone1_10.zip')
    df = pd.read_csv(tz.open(f'Task15_W_Zone1_10/Task15_W_Zone{zid}.csv'))
    ts = df['TIMESTAMP'].astype(str).str.strip()
    df['dt'] = pd.to_datetime(ts.str[:8], format='%Y%m%d') + pd.to_timedelta(
        ts.str.extract(r'(\d+):')[0].astype(int), unit='h')
    df = df.sort_values('dt').reset_index(drop=True)
    df['TARGETVAR'] = df['TARGETVAR'].interpolate(limit_direction='both')
    for c in ['U10','V10','U100','V100']:
        df[c] = df[c].interpolate(limit_direction='both')
    df = wx14(df)
    h = df['dt'].dt.hour.values.astype(np.float32)
    m = df['dt'].dt.month.values.astype(np.float32)
    df['HOUR_SIN'] = np.sin(2*np.pi*h/24)
    df['HOUR_COS'] = np.cos(2*np.pi*h/24)
    df['MONTH_SIN'] = np.sin(2*np.pi*m/12)
    df['MONTH_COS'] = np.cos(2*np.pi*m/12)
    feats = StandardScaler().fit_transform(df[F14].values.astype(np.float32))
    power = df['TARGETVAR'].values.astype(np.float32).clip(0, 1)
    return np.concatenate([feats, power.reshape(-1, 1)], axis=1)


class WDS(Dataset):
    def __init__(self, d, s=4):
        self.data = torch.FloatTensor(d); self.s = s
        self.n = max(0, (len(d) - SEQ - PRED) // s + 1)
    def __len__(self): return self.n
    def __getitem__(self, i):
        st = i * self.s
        return (self.data[st:st+SEQ].T, self.data[st+SEQ:st+SEQ+PRED, -1])


from nwp_model import NWPMamba, pinball_loss


def main():
    sep = "=" * 60
    msg = [
        sep,
        "  UNIFIED 17-SITE LNMamba — GEFCom2012(7) + GEFCom2014(10)",
        "  d=64, nb=2, ds=16, 50 epochs",
        sep
    ]
    for m in msg:
        print(m)
    sys.stdout.flush()

    datasets = []; nv = None

    # GEFCom2012
    print("[1/2] GEFCom2012 (7 farms)...")
    sys.stdout.flush()
    for fid in range(1, 8):
        d = build_gef12(fid)
        if nv is None: nv = d.shape[1]
        T = len(d); te = int(T * 0.85)
        ds = WDS(d[:te], 4)
        datasets.append(ds)
        print(f"  Farm {fid}: {len(ds):,} windows")
        sys.stdout.flush()

    # GEFCom2014
    print("[2/2] GEFCom2014 (10 zones)...")
    sys.stdout.flush()
    for zid in range(1, 11):
        d = build_gef14(zid)
        T = len(d); te = int(T * 0.85)
        ds = WDS(d[:te], 4)
        datasets.append(ds)
        if d.shape[1] > nv: nv = d.shape[1]
        print(f"  Zone {zid}: {len(ds):,} windows")
        sys.stdout.flush()

    # Pad to max vars
    print(f"Max vars: {nv}")
    sys.stdout.flush()
    for i, ds in enumerate(datasets):
        nvi = ds.data.shape[1]
        if nvi < nv:
            pad = torch.zeros(len(ds.data), nv - nvi)
            ds.data = torch.cat([ds.data, pad], dim=1)
            datasets[i] = ds

    ds_all = ConcatDataset(datasets)
    print(f"Total: {len(ds_all):,} windows")
    tl = DataLoader(ds_all, 48, shuffle=True, num_workers=0, pin_memory=True)
    print(f"Batches/epoch: {len(tl)}")
    sys.stdout.flush()

    # Test: GEFCom2012 Farm 1
    d1 = build_gef12(1)
    T1 = len(d1); te1 = int(T1 * 0.85)
    if d1.shape[1] < nv:
        d1 = np.concatenate([d1, np.zeros((len(d1), nv - d1.shape[1]), dtype=np.float32)], axis=1)
    test_ds = WDS(d1[te1:], 4)
    testl = DataLoader(test_ds, 48, shuffle=False, num_workers=0, pin_memory=True)
    print(f"Test: {len(test_ds)} windows")

    # Model
    model = NWPMamba(nv, d=64, nb=2, ds=16, pred=PRED, nq=99, use_lnn=True).to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_p:,}")
    sys.stdout.flush()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    EPOCHS = 40
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    qt = torch.tensor(QUANTILES, dtype=torch.float32, device=DEVICE)
    best_pb = float('inf'); best_state = None

    msg2 = [sep, f"  TRAINING ({EPOCHS} epochs)", sep]
    for m in msg2:
        print(m)
    sys.stdout.flush()

    for ep in range(1, EPOCHS + 1):
        t0 = time.time(); model.train(); tp = 0.0
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE); opt.zero_grad()
            loss = pinball_loss(model(x), y, qt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tp += loss.item()
        sch.step()
        ap = tp / len(tl); et = time.time() - t0
        star = " *" if ap < best_pb else "  "
        if ap < best_pb:
            best_pb = ap
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"E{ep:2d} pb={ap:.4f} {et:.0f}s{star}")
        sys.stdout.flush()

    if best_state:
        model.load_state_dict(best_state)

    # Test
    model.eval(); preds, targs = [], []
    tp = 0.0
    with torch.no_grad():
        for x, y in testl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            o = model(x)
            tp += pinball_loss(o, y, qt).item()
            preds.append(o.cpu().numpy()); targs.append(y.cpu().numpy())
    pr = np.concatenate(preds); tr = np.concatenate(targs)
    test_pb = tp / len(testl)

    # Expected value point prediction
    qm = (pr[:, :, :-1] + pr[:, :, 1:]) / 2
    pe = np.sum(qm * np.diff(QUANTILES), axis=-1)
    pf, tf = pe.ravel(), tr.ravel()
    mask = tf > 0.005
    rmse = np.sqrt(np.mean((pf[mask] - tf[mask]) ** 2))
    mae  = np.mean(np.abs(pf[mask] - tf[mask]))
    r2   = 1 - np.sum((tf[mask] - pf[mask]) ** 2) / (np.sum((tf[mask] - np.mean(tf[mask])) ** 2) + 1e-8)
    cov80 = np.mean((tf.reshape(-1, 24) >= pr[:, :, 9]) & (tf.reshape(-1, 24) <= pr[:, :, 89])) * 100

    msg3 = [sep, "  FARM 1 TEST RESULTS", sep]
    for m in msg3:
        print(m)
    print(f"  Pinball:     {test_pb:.4f}")
    print(f"  R2 (exp val): {r2:.4f}")
    print(f"  RMSE:        {rmse:.4f}")
    print(f"  MAE:         {mae:.4f}")
    print(f"  80% CI cov:  {cov80:.1f}%")
    print("  Per-horizon R2:")
    for h in [0, 3, 5, 11, 17, 23]:
        ph, th = pe[:, h], tr[:, h]
        mh = th > 0.005
        r2h = 1 - np.sum((th[mh] - ph[mh]) ** 2) / (np.sum((th[mh] - np.mean(th[mh])) ** 2) + 1e-8)
        print(f"    +{h+1:2d}h: {r2h:+.4f}")

    print(f"\n  Training samples: {len(ds_all):,} (17 sites)")
    print(f"  Model params:     {n_p:,}")
    print(f"  Best train pb:    {best_pb:.4f}")
    print("  Done!")
    sys.stdout.flush()


if __name__ == '__main__':
    main()
