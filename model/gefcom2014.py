"""
GEFCom2014 Wind Power Forecasting — Clean Pipeline.

Data: 10 zones, hourly, Jan 2012 - Nov 2013 train, Dec 2013 test.
Features: U10, V10, U100, V100 (ECMWF NWP wind components)
Target: TARGETVAR (normalized wind power)

Models: Persistence → GRU → Mamba → LNN-Mamba
"""
import sys,os,glob,zipfile,time,argparse,io,math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DATA_DIR = 'data/gefcom2014'

# ═══════════ Data Loading ═══════════
def load_gefcom2014(site=1, seq=168, pred=24, batch=64, stride=4):
    """Load one zone, return (train, val, test) loaders + scalers."""
    # Power data (has NWP features embedded)
    tz = zipfile.ZipFile(f'{DATA_DIR}/Task15_W_Zone1_10.zip')
    df = pd.read_csv(tz.open(f'Task15_W_Zone1_10/Task15_W_Zone{site}.csv'))

    # Parse timestamp
    df['dt'] = pd.to_datetime(df['TIMESTAMP'].astype(str).str[:8], format='%Y%m%d') \
               + pd.to_timedelta(df['TIMESTAMP'].str.extract(r'(\d+):00')[0].astype(int), unit='h')
    df = df.sort_values('dt').reset_index(drop=True)
    dmin = df['dt'].min(); dmax = df['dt'].max()
    print(f'Zone {site}: {len(df)} rows, {dmin} ~ {dmax}')

    # Features: NWP wind components
    feat_cols = ['U10', 'V10', 'U100', 'V100']
    # Also compute wind speed as auxiliary
    df['WS10'] = np.sqrt(df['U10']**2 + df['V10']**2)
    df['WS100'] = np.sqrt(df['U100']**2 + df['V100']**2)
    all_feats = feat_cols + ['WS10', 'WS100']

    target_col = 'TARGETVAR'

    # Fix NaN in target (interpolate)
    df[target_col] = df[target_col].interpolate(method='linear', limit_direction='both')

    # Normalize features + target
    scaler_x = StandardScaler()
    feats = scaler_x.fit_transform(df[all_feats].values.astype(np.float32))
    scaler_y = StandardScaler()
    target = scaler_y.fit_transform(df[[target_col]].values.astype(np.float32))

    data = np.concatenate([feats, target], axis=1)  # (T, 7)

    # Split: train on 2012-01 ~ 2013-10, val 2013-11, test 2013-12
    # Data is ~16800 rows: Jan2012-Nov2013 = ~23 months
    # Use last 2 months as val, last 1 month as test (NWP forecast period)
    T = len(data)
    train_end = int(T * 0.85)  # ~ Nov 2013 start
    test_start = int(T * 0.95)  # ~ Dec 2013 start (NWP-only)
    val_end = test_start

    ds_train = WindDS(data[:train_end], seq, pred, stride)
    ds_val   = WindDS(data[train_end:val_end], seq, pred, stride)
    ds_test  = WindDS(data[val_end:], seq, pred, stride)

    print(f'Samples: {len(ds_train):,} train | {len(ds_val):,} val | {len(ds_test):,} test')

    dls = (
        DataLoader(ds_train, batch, shuffle=True,  num_workers=0, pin_memory=True),
        DataLoader(ds_val,   batch, shuffle=False, num_workers=0, pin_memory=True),
        DataLoader(ds_test,  batch, shuffle=False, num_workers=0, pin_memory=True),
    )
    return dls, data.shape[1], scaler_y


class WindDS(Dataset):
    def __init__(self, data, seq, pred, stride):
        self.data = torch.FloatTensor(data)
        self.seq=seq; self.pred=pred; self.s=stride
        self.n = max(0, (len(data)-seq-pred)//stride+1)
    def __len__(self): return self.n
    def __getitem__(self, i):
        st = i*self.s
        return (self.data[st:st+self.seq].T,
                self.data[st+self.seq:st+self.seq+self.pred, -1])

# ═══════════ Models ═══════════
class FastMamba(nn.Module):
    def __init__(self, d, ds=16, dc=4, ex=2):
        super().__init__()
        di = d*ex; self.ds = ds
        self.inp  = nn.Linear(d, di*2, bias=False)
        self.cnv  = nn.Conv1d(di, di, dc, groups=di, padding=dc-1)
        self.xp   = nn.Linear(di, ds*2+1, bias=False)
        self.dtp  = nn.Linear(ds, di, bias=True)
        A = torch.arange(1, ds+1).float().unsqueeze(0)*0.05
        self.A_log = nn.Parameter(torch.log(A))
        self.D     = nn.Parameter(torch.ones(di))
        self.out  = nn.Linear(di, d, bias=False)
        self.nm   = nn.RMSNorm(d)

    def forward(self, x):
        B,L,D = x.shape; res = x
        xz = self.inp(x); u,z = xz.chunk(2, dim=-1)
        u = F.silu(self.cnv(u.transpose(1,2))[:,:,:L].transpose(1,2))
        proj = self.xp(u)
        dt = F.softplus(self.dtp(F.softplus(proj[:,:,:self.ds])))+1e-4
        Bs,Cs = proj[:,:,self.ds:self.ds*2], proj[:,:,self.ds*2:]
        de = dt.unsqueeze(-1)
        A = -torch.exp(self.A_log)
        Abar = torch.exp(de * A.unsqueeze(0).unsqueeze(1))
        b = de * Bs.unsqueeze(2) * u.unsqueeze(-1)
        eps=1e-8; logA = torch.log(Abar.clamp(min=eps))
        Acum = torch.exp(torch.cumsum(logA, dim=1))
        h = Acum * torch.cumsum(b/Acum.clamp(min=eps), dim=1)
        y = (h*Cs.unsqueeze(2)).sum(-1) + self.D.unsqueeze(0).unsqueeze(0)*u
        return self.nm(self.out(y*F.silu(z)) + res)

class GRUModel(nn.Module):
    def __init__(self, V, d=128, nl=2, pred=24):
        super().__init__()
        self.proj = nn.Linear(V, d)
        self.gru = nn.GRU(d, d, nl, batch_first=True, dropout=0.1)
        self.dec = nn.Sequential(nn.Linear(d, d*2), nn.GELU(), nn.Dropout(0.1), nn.Linear(d*2, pred))
    def forward(self, x):
        _, h = self.gru(self.proj(x.transpose(1,2)))
        return self.dec(h[-1])

class LNNMambaModel(nn.Module):
    def __init__(self, V, d=64, nb=2, ds=16, pred=24, use_lnn=True):
        super().__init__()
        self.use_lnn = use_lnn
        self.emb = nn.Sequential(nn.Linear(V, d*2), nn.GELU(), nn.Linear(d*2, d))
        self.pe  = nn.Parameter(torch.randn(1,2000,d)*0.02)
        self.mb  = nn.ModuleList([FastMamba(d,ds) for _ in range(nb)])
        self.lnn = nn.ModuleList([nn.Sequential(nn.GRU(d,32,batch_first=True), nn.Linear(32,d)) for _ in range(nb)])
        self.dec = nn.Sequential(nn.Linear(d, d*2), nn.GELU(), nn.Dropout(0.1), nn.Linear(d*2, pred))
        self.drop = nn.Dropout(0.1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 0.5)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        B,V,L = x.shape
        x = self.emb(x.transpose(1,2)) + self.pe[:,:L]
        for mb, ln in zip(self.mb, self.lnn):
            x = self.drop(mb(x))
            if self.use_lnn:
                h,_ = ln[0](x)
                x = x * torch.sigmoid(ln[1](h))
        return self.dec(x[:,-1])

# ═══════════ Metrics ═══════════
def metrics(pred, target, scaler_y):
    p = scaler_y.inverse_transform(pred.reshape(-1,1)).reshape(pred.shape)
    t = scaler_y.inverse_transform(target.reshape(-1,1)).reshape(target.shape)
    pf = p.ravel(); tf = t.ravel()
    mask = tf > 0.001
    pf, tf = pf[mask], tf[mask]
    mse = np.mean((pf-tf)**2); mae = np.mean(np.abs(pf-tf))
    mape = np.mean(np.abs((tf-pf)/(np.abs(tf)+1e-4)))*100
    r2 = 1 - np.sum((tf-pf)**2)/(np.sum((tf-np.mean(tf))**2)+1e-8)
    rmse_h = [np.sqrt(np.mean((p[:,h]-t[:,h])**2)) for h in range(p.shape[1])]
    return {'rmse': np.sqrt(mse), 'mae': mae, 'mape': mape, 'r2': r2, 'rmse_h': rmse_h}, p, t

def evaluate_persistence(loader, scaler_y, pred_len):
    preds, targs = [], []
    for x, y in loader:
        last = x[:,-1,-1].unsqueeze(1).repeat(1,pred_len).cpu().numpy()
        preds.append(last); targs.append(y.numpy())
    return metrics(np.concatenate(preds), np.concatenate(targs), scaler_y)

def train_model(model, tl, vl, device, epochs=30, lr=1e-3, label=''):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    best_rmse = float('inf'); best_state = None; hist = []

    for ep in range(1, epochs+1):
        t0 = time.time(); model.train(); tl_loss = 0.0
        for x, y in tl:
            x, y = x.to(device), y.to(device); opt.zero_grad()
            with torch.amp.autocast('cuda'): loss = F.mse_loss(model(x), y)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); tl_loss += loss.item()
        sch.step()

        model.eval(); preds, targs = [], []
        with torch.no_grad():
            for x, y in vl:
                preds.append(model(x.to(device)).cpu().numpy()); targs.append(y.numpy())
        pr = np.concatenate(preds); tr = np.concatenate(targs)
        rmse = np.sqrt(np.mean((pr.ravel()-tr.ravel())**2))
        hist.append(rmse); t = time.time() - t0

        if rmse < best_rmse:
            best_rmse = rmse; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        print(f'E {ep:2d} | loss={tl_loss/len(tl):.4f} | V-RMSE={rmse:.4f} | {t:.0f}s{" *" if rmse<best_rmse else ""}')
        if ep >= 10 and ep - np.argmin(hist) >= 8:
            print(f'  Early stop'); break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_rmse if best_rmse != float('inf') else float('nan'), hist

# ═══════════ Main ═══════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zone', type=int, default=1)
    parser.add_argument('--seq', type=int, default=168)
    parser.add_argument('--pred', type=int, default=24)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    SEQ, PRED = args.seq, args.pred

    # Data
    (tl, vl, testl), n_vars, scaler_y = load_gefcom2014(args.zone, SEQ, PRED, args.batch, args.stride)

    # Persistence
    pm, _, _ = evaluate_persistence(testl, scaler_y, PRED)
    print(f'\nPersistence: RMSE={pm["rmse"]:.4f} ({pm["rmse"]*100:.1f}% rated) | '
          f'MAE={pm["mae"]:.4f} | MAPE={pm["mape"]:.1f}% | R2={pm["r2"]:.3f}')

    # Models
    results = {}
    configs = [
        ('GRU',         GRUModel(n_vars, d=128, nl=2, pred=PRED)),
        ('Mamba',       LNNMambaModel(n_vars, d=args.d_model, nb=2, ds=16, pred=PRED, use_lnn=False)),
        ('LNN-Mamba',   LNNMambaModel(n_vars, d=args.d_model, nb=2, ds=16, pred=PRED, use_lnn=True)),
    ]

    for name, model in configs:
        n_p = sum(p.numel() for p in model.parameters())
        model = model.to(device)
        print(f'\n=== {name} ({n_p:,} params) ===')
        best_val, hist = train_model(model, tl, vl, device, args.epochs, args.lr, name)

        model.eval(); preds, targs = [], []
        with torch.no_grad():
            for x, y in testl:
                preds.append(model(x.to(device)).cpu().numpy())
                targs.append(y.numpy())
        m, _, _ = metrics(np.concatenate(preds), np.concatenate(targs), scaler_y)
        results[name] = m

    # ── Print table ──
    print(f'\n{"="*70}')
    print(f'GEFCom2014 Wind Power Forecasting — Zone {args.zone}')
    print(f'seq={SEQ}h → pred={PRED}h, 1h resolution')
    print(f'{"="*70}')
    print(f'{"Model":<15s} {"RMSE":>10s} {"MAE":>10s} {"MAPE":>8s} {"R2":>8s} {"vs Persist":>10s}')
    print(f'{"-"*70}')
    p_rmse = pm['rmse']
    print(f'{"Persistence":<15s} {pm["rmse"]:>10.4f} {pm["mae"]:>10.4f} {pm["mape"]:>7.1f}% {pm["r2"]:>7.3f} {"--":>10s}')
    for name, m in results.items():
        imp = (p_rmse - m['rmse'])/p_rmse*100
        print(f'{name:<15s} {m["rmse"]:>10.4f} {m["mae"]:>10.4f} {m["mape"]:>7.1f}% {m["r2"]:>7.3f} {imp:>+9.1f}%')

    # Per-horizon
    print(f'\n{"Per-horizon RMSE":-^70}')
    print(f'{"Hour":>8s} {"Persist":>10s} {"GRU":>10s} {"Mamba":>10s} {"LNN-Mamba":>10s}')
    for h in [0,3,5,11,17,23]:
        print(f'+{h+1:2d}h     {pm["rmse_h"][h]:>10.4f} {results["GRU"]["rmse_h"][h]:>10.4f} {results["Mamba"]["rmse_h"][h]:>10.4f} {results["LNN-Mamba"]["rmse_h"][h]:>10.4f}')

    # ── Plots ──
    os.makedirs('plots', exist_ok=True)

    # Horizon error
    fig, ax = plt.subplots(figsize=(10,5))
    hh = [h+1 for h in range(PRED)]
    ax.plot(hh, pm['rmse_h'], 'k-', lw=2, label=f'Persistence')
    ax.plot(hh, results['GRU']['rmse_h'], 'b-', lw=1.5, alpha=0.8, label=f'GRU')
    ax.plot(hh, results['Mamba']['rmse_h'], 'g-', lw=1.5, alpha=0.8, label=f'Mamba')
    ax.plot(hh, results['LNN-Mamba']['rmse_h'], 'r-', lw=2, label=f'LNN-Mamba')
    ax.set_xlabel('Horizon (hours)'); ax.set_ylabel('RMSE (normalized power)')
    ax.set_title(f'GEFCom2014 Zone {args.zone} | {SEQ}h → {PRED}h forecast')
    ax.legend(); ax.grid(alpha=0.2)
    plt.tight_layout(); plt.savefig(f'plots/gefcom2014_z{args.zone}_horizon.png', dpi=150); plt.close()

    # Scatter for best model
    best_name = sorted(results.items(), key=lambda x: x[1]['rmse'])[0][0]
    _, pr, tr = metrics(np.concatenate(preds), np.concatenate(targs), scaler_y)
    pf = pr.ravel(); tf = tr.ravel()
    fig, ax = plt.subplots(figsize=(7,7))
    ax.hexbin(tf, pf, gridsize=40, cmap='Blues', mincnt=1, alpha=0.85)
    mx = max(tf.max(), pf.max())
    ax.plot([0,mx],[0,mx],'k--',lw=1,alpha=0.5)
    r2_v = 1-np.sum((tf-pf)**2)/(np.sum((tf-np.mean(tf))**2)+1e-8)
    rmse_v = np.sqrt(np.mean((tf-pf)**2))
    ax.text(0.05,0.95,f'{best_name}\nRMSE={rmse_v:.4f}\nR2={r2_v:.3f}', transform=ax.transAxes,
            fontsize=11, va='top', bbox=dict(boxstyle='round',facecolor='white',alpha=0.85), fontfamily='monospace')
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted'); ax.set_aspect('equal')
    ax.set_title(f'GEFCom2014 Zone {args.zone} — {best_name}')
    plt.tight_layout(); plt.savefig(f'plots/gefcom2014_z{args.zone}_scatter.png', dpi=150); plt.close()

    print(f'\nPlots saved to plots/')
    print('Done!')


if __name__ == '__main__':
    main()
