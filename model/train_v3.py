"""
LMT v3 — Spectral loss + 6-site multi-farm. Self-contained, flush-every-step.
"""
import sys,os,glob,time,json
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model_v2 import LNNMambaTransformerV2

# ═══════════════════════ Data ═══════════════════════

class WindDS(Dataset):
    def __init__(self, data, tfeat, seq_len=96, pred_len=96, stride=24):
        self.data = torch.FloatTensor(data)
        self.tfeat = torch.FloatTensor(tfeat)
        self.seq_len = seq_len; self.pred_len = pred_len; self.stride = stride
        self.n = max(0, (len(data)-seq_len-pred_len)//stride+1)
    def __len__(self): return self.n
    def __getitem__(self, i):
        s = i*self.stride
        x = self.data[s:s+self.seq_len].T
        y = self.data[s+self.seq_len:s+self.seq_len+self.pred_len, -1]
        ts = self.tfeat[s:s+self.seq_len]
        lp = self.data[s+self.seq_len-1, -1]
        return x, y, ts, lp

def load_all():
    """Load 6 wind farms, align columns, return data+time arrays."""
    SITES = ['Wind_farm_site_1_99MW','Wind_farm_site_2_200MW','Wind_farm_site_3_99MW',
             'Wind_farm_site_4_66MW','Wind_farm_site_5_36MW','Wind_farm_site_6_96MW']
    all_data = []
    for site in SITES:
        files = sorted(glob.glob(f'data/wind/{site}*.csv'))
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        tc = df.columns[0]; df[tc] = pd.to_datetime(df[tc])
        df = df.sort_values(tc).reset_index(drop=True)
        # Handle NaN: interpolate numeric columns only
        num_cols = df.select_dtypes(include='number').columns
        df[num_cols] = df[num_cols].interpolate(method='linear', limit_direction='both')
        pc = [c for c in df.columns if 'Power' in c or 'power' in c][0]
        fc = [c for c in df.columns if 'Time' not in c.strip() and 'Power' not in c and 'power' not in c]
        features = StandardScaler().fit_transform(df[fc].values.astype(np.float32))
        power = StandardScaler().fit_transform(df[pc].values.astype(np.float32).reshape(-1,1))
        arr = np.concatenate([features, power], axis=1)
        h = df[tc].dt.hour.values.astype(np.float32)
        dow = df[tc].dt.dayofweek.values.astype(np.float32)
        mo = (df[tc].dt.month-1).values.astype(np.float32)
        se = (df[tc].dt.month%12//3).values.astype(np.float32)
        rp = np.arange(len(df), dtype=np.float32)/len(df)
        tfeat = np.stack([h,dow,mo,se,rp], axis=1)
        all_data.append((arr, tfeat))
        print(f'  {site}: {len(arr)} rows, {arr.shape[1]} vars')
        sys.stdout.flush()

    # Pad to max vars
    mv = max(d.shape[1] for d,_ in all_data)
    print(f'Max vars: {mv}'); sys.stdout.flush()
    for i in range(len(all_data)):
        d,t = all_data[i]
        if d.shape[1] < mv:
            d = np.concatenate([d, np.zeros((len(d),mv-d.shape[1]),dtype=np.float32)], axis=1)
            all_data[i] = (d,t)

    data = np.concatenate([d for d,_ in all_data])
    tfeat = np.concatenate([t for _,t in all_data])
    # Shuffle
    idx = np.random.RandomState(42).permutation(len(data))
    return data[idx], tfeat[idx], mv

def main():
    device = torch.device('cuda')
    BATCH = 12; STRIDE = 48; SEQ = 96; PRED = 96; EPOCHS = 20
    print('Loading 6 farms...'); sys.stdout.flush()
    data, tfeat, n_vars = load_all()
    T = len(data); te = int(T*0.7); ve = te + int(T*0.1)
    print(f'Total: {T} rows, train={te}, val={ve-te}, test={T-ve}'); sys.stdout.flush()

    ds_train = WindDS(data[:te], tfeat[:te], SEQ, PRED, STRIDE)
    ds_val   = WindDS(data[te:ve], tfeat[te:ve], SEQ, PRED, STRIDE)
    ds_test  = WindDS(data[ve:], tfeat[ve:], SEQ, PRED, STRIDE)
    print(f'Samples: train={len(ds_train)}, val={len(ds_val)}, test={len(ds_test)}'); sys.stdout.flush()

    tl = DataLoader(ds_train, BATCH, shuffle=True, num_workers=0, pin_memory=True)
    vl = DataLoader(ds_val, BATCH, shuffle=False, num_workers=0, pin_memory=True)
    testl = DataLoader(ds_test, BATCH, shuffle=False, num_workers=0, pin_memory=True)
    print(f'Batches: train={len(tl)}/epoch'); sys.stdout.flush()

    # Model
    model = LNNMambaTransformerV2(
        n_vars=n_vars, d_model=96, n_mamba_blocks=2, n_transformer_layers=1,
        d_state=32, d_conv=4, n_heads=8, pred_len=PRED, lnn_hidden=64,
        dropout=0.1, rated_capacity=99.0
    ).to(device)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Params: {n_p:,}'); sys.stdout.flush()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=15, T_mult=2, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    best_rmse = float('inf')
    os.makedirs('checkpoints', exist_ok=True)
    history = {'train_loss':[], 'val_rmse':[], 'spectral':[]}

    for ep in range(1, EPOCHS+1):
        t0 = time.time()
        model.train()
        total_loss = 0; total_spec = 0; n_batch = 0

        for x, y, ts, lp in tl:
            x, y, ts, lp = x.to(device), y.to(device), ts.to(device), lp.to(device)
            opt.zero_grad()
            with torch.amp.autocast('cuda'):
                preds = model(x, ts, lp)
                ld = model.compute_loss(preds, y, lp)
            scaler.scale(ld['loss']).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            total_loss += ld['loss'].item()
            total_spec += ld.get('loss_spectral', torch.tensor(0)).item()
            n_batch += 1

        sch.step()
        train_loss = total_loss / max(n_batch, 1)
        spec_loss = total_spec / max(n_batch, 1)

        # Validate
        model.eval()
        val_preds, val_targs = [], []
        with torch.no_grad():
            for x, y, ts, lp in vl:
                x, lp = x.to(device), lp.to(device)
                out = model(x, ts.to(device), lp)
                val_preds.append(out['pred'].cpu().numpy())
                val_targs.append(y.cpu().numpy())

        pr = np.concatenate(val_preds).ravel()
        tr = np.concatenate(val_targs).ravel()
        mask = tr > 0
        if mask.sum() > 0: pr, tr = pr[mask], tr[mask]
        rmse = float(np.sqrt(np.mean((pr-tr)**2)))
        mae = float(np.mean(np.abs(pr-tr)))
        et = time.time() - t0

        star = ' ★' if rmse < best_rmse else ''
        if rmse < best_rmse:
            best_rmse = rmse
            torch.save({'epoch':ep, 'model':model.state_dict(), 'val_rmse':best_rmse,
                        'n_vars':n_vars}, 'checkpoints/lmt_v3_best.pt')

        history['train_loss'].append(train_loss)
        history['val_rmse'].append(rmse)
        history['spectral'].append(spec_loss)

        print(f'E {ep:2d} | loss={train_loss:.4f} spec={spec_loss:.4f} | '
              f'V-RMSE={rmse:.4f} MAE={mae:.4f} | {et:.0f}s{star}')
        sys.stdout.flush()

    # Test
    print(f'\n=== TEST ==='); sys.stdout.flush()
    ckpt = torch.load('checkpoints/lmt_v3_best.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    tpreds, ttargs = [], []
    with torch.no_grad():
        for x, y, ts, lp in testl:
            x, lp = x.to(device), lp.to(device)
            out = model(x, ts.to(device), lp)
            tpreds.append(out['pred'].cpu().numpy())
            ttargs.append(y.cpu().numpy())
    pr = np.concatenate(tpreds); tr = np.concatenate(ttargs)
    ph = pr.ravel(); th = tr.ravel()
    mask = th > 0
    rmse_test = float(np.sqrt(np.mean((ph[mask]-th[mask])**2)))
    mae_test = float(np.mean(np.abs(ph[mask]-th[mask])))
    print(f'Test: RMSE={rmse_test:.4f} MAE={mae_test:.4f} | best_val={best_rmse:.4f}')
    print('Horizon:')
    for h in [0,3,11,23,47,71,95]:
        r = np.sqrt(np.mean((pr[:,h]-tr[:,h])**2))
        print(f'  +{(h+1)*15:3d}min: {r:.4f}')

    np.savez('checkpoints/v3_results.npz', pred=pr, target=tr)
    json.dump(history, open('checkpoints/v3_history.json','w'))
    print('Done!')

if __name__ == '__main__':
    main()
