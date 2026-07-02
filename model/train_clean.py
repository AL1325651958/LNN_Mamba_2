"""
Clean training: GRU baseline vs LNN-Gated Selective SSM.
Single site, stride=1, fast iteration.
"""
import sys,os,glob,time,numpy as np
import torch, torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model_clean import CleanLNNMamba, GRUBaseline


class FastWindDS(Dataset):
    def __init__(self, data, seq_len=96, pred_len=24, stride=4):
        self.data = torch.FloatTensor(data)  # (T, V)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        self.n = max(0, (len(data) - seq_len - pred_len) // stride + 1)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = i * self.stride
        x = self.data[s:s+self.seq_len].T      # (V, L)
        y = self.data[s+self.seq_len:s+self.seq_len+self.pred_len, -1]  # (P,)
        return x, y


def load_site2():
    """Load Wind Farm Site 2 — best data quality."""
    site = 'Wind_farm_site_2_200MW'
    files = sorted(glob.glob(f'data/wind/{site}*.csv'))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    tc = df.columns[0]
    df[tc] = pd.to_datetime(df[tc])
    df = df.sort_values(tc).reset_index(drop=True)
    df = df.interpolate(method='linear', limit_direction='both')

    pc = [c for c in df.columns if 'Power' in c][0]
    fc = [c for c in df.columns if 'Time' not in c.strip() and 'Power' not in c]
    print(f'Site 2: {len(df)} rows, {len(fc)} features, target={pc}')

    feats = StandardScaler().fit_transform(df[fc].values.astype(np.float32))
    power = StandardScaler().fit_transform(df[pc].values.astype(np.float32).reshape(-1, 1))
    data = np.concatenate([feats, power], axis=1)
    return data, len(fc) + 1


def compute_metrics(pred, target):
    """All metrics on CPU numpy."""
    p = pred.cpu().numpy().ravel()
    t = target.cpu().numpy().ravel()
    mask = t > -999  # use all data
    p, t = p[mask], t[mask]
    mse = np.mean((p - t) ** 2)
    mae = np.mean(np.abs(p - t))
    mape = np.mean(np.abs((t - p) / (np.abs(t) + 1e-4))) * 100
    r2 = 1 - np.sum((t-p)**2) / (np.sum((t-np.mean(t))**2) + 1e-8)
    return {'rmse': np.sqrt(mse), 'mae': mae, 'mape': mape, 'r2': r2}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targs = [], []
    for x, y in loader:
        out = model(x.to(device))
        preds.append(out.cpu())
        targs.append(y)
    return compute_metrics(torch.cat(preds), torch.cat(targs))


def train_model(model, tl, vl, device, epochs=50, lr=1e-3, label='model'):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    best_rmse = float('inf')
    history = {'val_rmse': [], 'val_mae': [], 'time': []}

    print(f'\n{"="*60}')
    print(f'{label}: {sum(p.numel() for p in model.parameters()):,} params, '
          f'{len(tl)} batches ({len(tl.dataset):,} samples)')
    print(f'{"="*60}')

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast('cuda'):
                pred = model(x)
                ld = model.compute_loss(pred, y)
            loss = ld['loss']
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total_loss += loss.item()

        sch.step()
        train_loss = total_loss / len(tl)

        # Validate
        val_m = evaluate(model, vl, device)
        et = time.time() - t0
        history['val_rmse'].append(val_m['rmse'])
        history['val_mae'].append(val_m['mae'])
        history['time'].append(et)

        star = ' ★' if val_m['rmse'] < best_rmse else ''
        if val_m['rmse'] < best_rmse:
            best_rmse = val_m['rmse']
            torch.save(model.state_dict(), f'checkpoints/clean_{label}_best.pt')

        extra = f'| spec={ld.get("spectral", 0):.4f}' if 'spectral' in ld else ''
        print(f'E {ep:3d} | loss={train_loss:.4f} {extra} '
              f'| V-RMSE={val_m["rmse"]:.4f} MAE={val_m["mae"]:.4f} '
              f'R2={val_m["r2"]:.3f} | {et:.0f}s{star}')

        if ep >= 15 and ep - np.argmin(history['val_rmse']) >= 10:
            print(f'  Early stop: no improvement for 10 epochs')
            break

    # Restore best & final eval
    model.load_state_dict(torch.load(f'checkpoints/clean_{label}_best.pt', map_location=device))
    best_val = min(history['val_rmse'])
    print(f'Best val RMSE: {best_val:.4f}')
    return model, best_val, history


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Data ──
    data, n_vars = load_site2()
    T = len(data)
    te = int(T * 0.7)
    ve = te + int(T * 0.15)

    SEQ, PRED, STRIDE = 96, 24, 4

    ds_train = FastWindDS(data[:te], SEQ, PRED, STRIDE)
    ds_val   = FastWindDS(data[te:ve], SEQ, PRED, STRIDE)
    ds_test  = FastWindDS(data[ve:], SEQ, PRED, STRIDE)
    BATCH = 64
    print(f'Samples: {len(ds_train):,} train, {len(ds_val):,} val, {len(ds_test):,} test')

    tl = DataLoader(ds_train, BATCH, shuffle=True,  num_workers=0, pin_memory=True)
    vl = DataLoader(ds_val,   BATCH, shuffle=False, num_workers=0, pin_memory=True)
    testl = DataLoader(ds_test, BATCH, shuffle=False, num_workers=0, pin_memory=True)

    results = {}

    # ── 1. GRU Baseline ──
    print('\n' + '='*60)
    print('EXPERIMENT 1: GRU Baseline')
    gru = GRUBaseline(n_vars=n_vars, d_model=128, n_layers=2, pred_len=PRED).to(device)
    gru, gru_best, gru_hist = train_model(gru, tl, vl, device, epochs=50, lr=1e-3, label='gru')

    # ── 2. Mamba only (no LNN, no spectral) ──
    print('\n' + '='*60)
    print('EXPERIMENT 2: Mamba (no LNN, no spectral)')
    mamba = CleanLNNMamba(n_vars=n_vars, d_model=64, n_blocks=2, d_state=16,
                          pred_len=PRED, use_lnn=False, use_spectral=False).to(device)
    mamba, mb_best, mb_hist = train_model(mamba, tl, vl, device, epochs=50, lr=1e-3, label='mamba')

    # ── 3. Selective SSM + Spectral ──
    print('\n' + '='*60)
    print('EXPERIMENT 3: Selective SSM + Spectral Loss')
    mspec = CleanLNNMamba(n_vars=n_vars, d_model=64, n_blocks=2, d_state=16,
                          pred_len=PRED, use_lnn=False, use_spectral=True).to(device)
    mspec, ms_best, ms_hist = train_model(mspec, tl, vl, device, epochs=50, lr=1e-3, label='mb_spec')

    # ── 4. Full LNN-Gated Selective SSM ──
    print('\n' + '='*60)
    print('EXPERIMENT 4: LNN-Gated Selective SSM + Spectral')
    full = CleanLNNMamba(n_vars=n_vars, d_model=64, n_blocks=2, d_state=16,
                         pred_len=PRED, use_lnn=True, use_spectral=True).to(device)
    full, full_best, full_hist = train_model(full, tl, vl, device, epochs=50, lr=1e-3, label='lnn_mamba')

    # ── Final test comparison ──
    print('\n' + '='*60)
    print('FINAL TEST COMPARISON')
    print('='*60)

    for name, model in [('GRU', gru), ('Mamba', mamba), ('Mamba+Spectral', mspec), ('LNN-Gated Selective SSM', full)]:
        test_m = evaluate(model, testl, device)
        print(f'\n{name:20s} | RMSE={test_m["rmse"]:.4f} | MAE={test_m["mae"]:.4f} | '
              f'MAPE={test_m["mape"]:.1f}% | R2={test_m["r2"]:.3f}')
        results[name] = test_m

    # Per-horizon for best model
    best_model = full  # expected best
    best_model.eval()
    preds, targs = [], []
    for x, y in testl:
        with torch.no_grad():
            out = best_model(x.to(device))
        preds.append(out.detach().cpu().numpy())
        targs.append(y.numpy())
    pr = np.concatenate(preds)  # (N, 24)
    tr = np.concatenate(targs)

    print('\nLNN-Gated Selective SSM per-horizon RMSE:')
    for h in range(0, PRED, 4):
        r = np.sqrt(np.mean((pr[:, h] - tr[:, h]) ** 2))
        print(f'  +{(h+1)*15:3d}min: {r:.4f}')

    # Save
    np.savez('checkpoints/clean_results.npz',
             gru_rmse=float(results['GRU']['rmse']),
             mamba_rmse=float(results['Mamba']['rmse']),
             mspec_rmse=float(results['Mamba+Spectral']['rmse']),
             full_rmse=float(results['LNN-Gated Selective SSM']['rmse']),
             full_pred=pr, full_target=tr)
    print('\nResults saved to checkpoints/clean_results.npz')


if __name__ == '__main__':
    main()
