"""
LMT v2 Training — Multi-scale + ΔP + weighted loss + full data config.

30 epochs, integrated visualization at the end.
"""
import sys, os, time, json, argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model_v2 import LNNMambaTransformerV2


# ═══════════════════════════════════════════════════════════
# Data Pipeline with enhanced time features + last_power
# ═══════════════════════════════════════════════════════════

class WindDatasetV2(Dataset):
    def __init__(self, data, time_feat, seq_len=96, pred_len=96, stride=3):
        self.data = torch.FloatTensor(data)       # (T, V)
        self.time_feat = torch.FloatTensor(time_feat)  # (T, 5)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        self.n = max(0, (len(data) - seq_len - pred_len) // stride + 1)

    def __len__(self): return self.n

    def __getitem__(self, idx):
        s = idx * self.stride
        x = self.data[s:s+self.seq_len].T          # (V, L)
        y = self.data[s+self.seq_len:s+self.seq_len+self.pred_len, -1]  # Power = last col
        ts = self.time_feat[s:s+self.seq_len]       # (L, 5)
        last_power = self.data[s+self.seq_len-1, -1]  # scalar
        return x, y, ts, last_power


def load_multi_site_data(data_dir='data/wind',
                         seq_len=96, pred_len=96, batch_size=16, stride=3,
                         train_ratio=0.7, val_ratio=0.1):
    """Load ALL 6 wind farm sites with column alignment, create loaders."""
    import glob

    SITES = [
        'Wind_farm_site_1_99MW',
        'Wind_farm_site_2_200MW',
        'Wind_farm_site_3_99MW',
        'Wind_farm_site_4_66MW',
        'Wind_farm_site_5_36MW',
        'Wind_farm_site_6_96MW',
    ]

    # Load each site, align columns
    all_data = []  # list of (data_array, time_feat_array)
    for site in SITES:
        files = sorted(glob.glob(os.path.join(data_dir, f'{site}*.csv')))
        if not files:
            print(f'  WARNING: no files for {site}')
            continue
        dfs = [pd.read_csv(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)

        time_col = df.columns[0]
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col).reset_index(drop=True)

        # Find Power column by name
        power_col = [c for c in df.columns if 'Power' in c or 'power' in c][0]

        # Feature columns: strip whitespace, exclude Time and Power
        feature_cols = []
        for c in df.columns:
            c_stripped = c.strip()
            if 'Time' not in c_stripped and 'Power' not in c and 'power' not in c:
                feature_cols.append(c)

        print(f'  {site}: {len(df)} rows, {len(feature_cols)} features')

        # Time features
        h = df[time_col].dt.hour.values.astype(np.float32)
        dow = df[time_col].dt.dayofweek.values.astype(np.float32)
        mo = (df[time_col].dt.month - 1).values.astype(np.float32)
        season = (df[time_col].dt.month % 12 // 3).values.astype(np.float32)
        rel_pos = np.arange(len(df), dtype=np.float32) / len(df)
        time_feat_arr = np.stack([h, dow, mo, season, rel_pos], axis=1)

        features = StandardScaler().fit_transform(
            df[feature_cols].values.astype(np.float32))
        scaler_p_local = StandardScaler()
        power = scaler_p_local.fit_transform(
            df[power_col].values.astype(np.float32).reshape(-1, 1))

        data_arr = np.concatenate([features, power], axis=1)
        all_data.append((data_arr, time_feat_arr))

    # Pad to uniform feature count
    max_vars = max(d.shape[1] for d, _ in all_data)
    print(f'  Max vars: {max_vars}')

    padded_data = []
    padded_time = []
    for data_arr, time_arr in all_data:
        n_v = data_arr.shape[1]
        if n_v < max_vars:
            pad = np.zeros((len(data_arr), max_vars - n_v), dtype=np.float32)
            data_arr = np.concatenate([data_arr, pad], axis=1)
        padded_data.append(data_arr)
        padded_time.append(time_arr)

    # Concatenate all sites and sort by time proxy (already sorted within each site,
    # but sites overlap in time — shuffle interleaving for training diversity)
    data_all = np.concatenate(padded_data, axis=0)
    time_all = np.concatenate(padded_time, axis=0)

    # Shuffle to mix sites (avoids site ordering bias)
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(data_all))
    data_all = data_all[idx]
    time_all = time_all[idx]

    print(f'  Total rows (all farms): {len(data_all)}')

    # Split by index (time-shuffled, but we mix farms)
    T = len(data_all)
    te = int(T * train_ratio)
    ve = te + int(T * val_ratio)

    ds_train = WindDatasetV2(data_all[:te], time_all[:te], seq_len, pred_len, stride)
    ds_val   = WindDatasetV2(data_all[te:ve], time_all[te:ve], seq_len, pred_len, stride)
    ds_test  = WindDatasetV2(data_all[ve:], time_all[ve:], seq_len, pred_len, stride)

    print(f'  Samples: train={len(ds_train)}, val={len(ds_val)}, test={len(ds_test)}')
    print(f'  Train batches: {len(ds_train)//batch_size}')

    # For inverse-transform, use a dummy scaler_p (per-site differs, so skip transform in eval)
    # We'll store the individual scalers if needed
    scaler_p = None  # multi-site: evaluate in standardized space

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return dl_train, dl_val, dl_test, scaler_p, max_vars, 'Power (MW)'


# ═══════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════

def compute_metrics(pred, target, scaler_p=None):
    pred_np = pred.cpu().numpy().ravel()
    targ_np = target.cpu().numpy().ravel()
    if scaler_p is not None:
        pred_np = scaler_p.inverse_transform(pred_np.reshape(-1,1)).ravel()
        targ_np = scaler_p.inverse_transform(targ_np.reshape(-1,1)).ravel()
    mask = targ_np > 1.0
    pf, tf = pred_np[mask], targ_np[mask]
    if len(pf) == 0: pf, tf = pred_np, targ_np
    mse = np.mean((pf-tf)**2); mae = np.mean(np.abs(pf-tf))
    mape = np.mean(np.abs((tf-pf)/(tf+1e-4)))*100
    r2 = 1 - np.sum((tf-pf)**2)/(np.sum((tf-np.mean(tf))**2)+1e-8)
    return {'rmse': np.sqrt(mse), 'mae': mae, 'mape': mape, 'r2': r2}


# ═══════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════

def train_epoch(model, loader, opt, scaler, device):
    model.train(); losses = defaultdict(list)
    for x, y, ts, lp in loader:
        x, y, ts, lp = x.to(device), y.to(device), ts.to(device), lp.to(device)
        opt.zero_grad()
        with torch.amp.autocast('cuda'):
            preds = model(x, ts, lp)
            ld = model.compute_loss(preds, y, lp)
        scaler.scale(ld['loss']).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        for k, v in ld.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                losses[k].append(v.item())
    return {k: np.mean(v) for k, v in losses.items()}


@torch.no_grad()
def evaluate(model, loader, scaler_p, device):
    model.eval(); preds, targs = [], []
    for x, y, ts, lp in loader:
        x, y, lp = x.to(device), y.to(device), lp.to(device)
        out = model(x, ts.to(device), lp)
        preds.append(out['pred'].cpu()); targs.append(y.cpu())
    pr = torch.cat(preds); tr = torch.cat(targs)
    return compute_metrics(pr, tr, scaler_p), pr, tr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--stride', type=int, default=12)
    parser.add_argument('--d_model', type=int, default=96)
    parser.add_argument('--n_mamba', type=int, default=2)
    parser.add_argument('--n_tf', type=int, default=1)
    parser.add_argument('--d_state', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load data — ALL 6 wind farms
    tl, vl, testl, scaler_p, n_vars, pname = load_multi_site_data(
        seq_len=args.seq_len, pred_len=args.pred_len,
        batch_size=args.batch_size, stride=args.stride,
        train_ratio=args.train_ratio,
    )

    # Average rated capacity across 6 farms: (99+200+99+66+36+96)/6 ≈ 99 MW
    AVG_CAPACITY = 99.0

    # Build model
    model = LNNMambaTransformerV2(
        n_vars=n_vars, d_model=args.d_model,
        n_mamba_blocks=args.n_mamba, n_transformer_layers=args.n_tf,
        d_state=args.d_state, n_heads=8, pred_len=args.pred_len,
        lnn_hidden=64, dropout=0.1, rated_capacity=AVG_CAPACITY,
    ).to(device)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Params: {n_p:,}')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=15, T_mult=2, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    best_rmse = float('inf'); history = defaultdict(list)

    print(f'\n{"="*60}')
    print(f'TRAINING: seq={args.seq_len}, pred={args.pred_len}, d={args.d_model}, '
          f'mamba={args.n_mamba}, tf={args.n_tf}')
    print(f'Epochs={args.epochs}, batches={len(tl)}, samples={len(tl)*args.batch_size}')
    print(f'{"="*60}\n')

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        losses = train_epoch(model, tl, opt, scaler, device)
        sch.step()
        val_m, _, _ = evaluate(model, vl, scaler_p, device)
        et = time.time() - t0

        star = ' ★' if val_m['rmse'] < best_rmse else ''
        if val_m['rmse'] < best_rmse:
            best_rmse = val_m['rmse']
            torch.save({'epoch': ep, 'model': model.state_dict(), 'val_rmse': best_rmse,
                        'args': vars(args)}, 'checkpoints/lmt_v2_best.pt')

        for k in ['loss','loss_main','loss_delta','rmse']:
            if k in losses: history[k].append(losses[k])
        history['val_rmse'].append(val_m['rmse'])
        history['val_mae'].append(val_m['mae'])

        print(f'E {ep:2d} | loss={losses["loss"]:.4f} δ={losses.get("loss_delta",0):.4f} '
              f'| V-RMSE={val_m["rmse"]:.1f}MW MAE={val_m["mae"]:.1f} '
              f'MAPE={val_m["mape"]:.0f}% R²={val_m["r2"]:.3f} | {et:.0f}s{star}')

    # ── Test ──
    print(f'\n{"="*60}\nFINAL TEST')
    ckpt = torch.load('checkpoints/lmt_v2_best.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    test_m, test_pr, test_tr = evaluate(model, testl, scaler_p, device)

    print(f'Test RMSE: {test_m["rmse"]:.1f} MW | MAE: {test_m["mae"]:.1f} | '
          f'MAPE: {test_m["mape"]:.0f}% | R²: {test_m["r2"]:.3f}')

    # Per-horizon
    ph = test_pr.numpy().reshape(-1, args.pred_len)
    th = test_tr.numpy().reshape(-1, args.pred_len)
    if scaler_p is not None:
        ph = scaler_p.inverse_transform(ph.T.reshape(-1,1)).reshape(args.pred_len,-1).T
        th = scaler_p.inverse_transform(th.T.reshape(-1,1)).reshape(args.pred_len,-1).T
    print('Per-horizon RMSE (MW):')
    for h in [0,3,11,23,47,71,95]:
        if h < args.pred_len:
            print(f'  +{(h+1)*15:3d}min: {np.sqrt(np.mean((ph[:,h]-th[:,h])**2)):.1f}')

    # Save for visualization
    os.makedirs('checkpoints', exist_ok=True)
    np.savez('checkpoints/v2_results.npz', pred=ph, target=th)
    with open('checkpoints/v2_history.json','w') as fh:
        json.dump({k: [float(x) for x in v] for k,v in history.items()}, fh, indent=2)
    print(f'\nResults saved. Best val RMSE: {best_rmse:.1f} MW | Params: {n_p:,}')


if __name__ == '__main__':
    main()
