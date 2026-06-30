"""
Manual random hyperparameter search for LNMamba on GEFCom2014 Zone 1.
30 trials, each 8 epochs. Best config → 30 epoch full train.
"""
import sys,os,zipfile,time,json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device('cuda')
QUANTILES = np.linspace(0.01, 0.99, 99)
DATA_DIR = 'data/gefcom2014'
SEQ, PRED = 168, 24

class WDS(Dataset):
    def __init__(self, d, s):
        self.data = torch.FloatTensor(d); self.s = s
        self.n = max(0, (len(d) - SEQ - PRED) // s + 1)
    def __len__(self): return self.n
    def __getitem__(self, i):
        st = i * self.s
        return (self.data[st:st+SEQ].T, self.data[st+SEQ:st+SEQ+PRED, -1])

def load_zone1():
    af = ['U10','V10','U100','V100','WS10','WS100','WD10_S','WD10_C','WD100_S','WD100_C','SHEAR',
          'HOUR_SIN','HOUR_COS','MONTH_SIN','MONTH_COS']
    tz = zipfile.ZipFile(f'{DATA_DIR}/Task15_W_Zone1_10.zip')
    df = pd.read_csv(tz.open('Task15_W_Zone1_10/Task15_W_Zone1.csv'))
    ts = df['TIMESTAMP'].astype(str).str.strip()
    df['dt'] = pd.to_datetime(ts.str[:8], format='%Y%m%d') + pd.to_timedelta(ts.str.extract(r'(\d+):')[0].astype(int), unit='h')
    df = df.sort_values('dt').reset_index(drop=True)
    df['TARGETVAR'] = df['TARGETVAR'].interpolate(limit_direction='both')
    for c in ['U10','V10','U100','V100']: df[c] = df[c].interpolate(limit_direction='both')
    df['WS10'] = np.sqrt(df['U10']**2 + df['V10']**2)
    df['WS100'] = np.sqrt(df['U100']**2 + df['V100']**2)
    df['WD10_S'] = np.sin(np.arctan2(df['U10'], df['V10']))
    df['WD10_C'] = np.cos(np.arctan2(df['U10'], df['V10']))
    df['WD100_S'] = np.sin(np.arctan2(df['U100'], df['V100']))
    df['WD100_C'] = np.cos(np.arctan2(df['U100'], df['V100']))
    df['SHEAR'] = df['WS100'] / (df['WS10'] + 0.1)
    h = df['dt'].dt.hour.values.astype(np.float32); m = df['dt'].dt.month.values.astype(np.float32)
    df['HOUR_SIN'] = np.sin(2*np.pi*h/24); df['HOUR_COS'] = np.cos(2*np.pi*h/24)
    df['MONTH_SIN'] = np.sin(2*np.pi*m/12); df['MONTH_COS'] = np.cos(2*np.pi*m/12)
    sx = StandardScaler(); feats = sx.fit_transform(df[af].values.astype(np.float32))
    sy = StandardScaler(); tgt = sy.fit_transform(df[['TARGETVAR']].values.astype(np.float32)).ravel()
    data = np.concatenate([feats, tgt.reshape(-1, 1)], axis=1)
    return data, data.shape[1], sy

# ── Mamba ──
class Mb(nn.Module):
    def __init__(self, d, ds=16, dc=4):
        super().__init__()
        self.ds = ds; di = d * 2
        self.inp = nn.Linear(d, di*2, bias=False)
        self.cnv = nn.Conv1d(di, di, dc, groups=di, padding=dc-1)
        self.xp  = nn.Linear(di, ds*2+1, bias=False)
        self.dtp = nn.Linear(ds, di, bias=True)
        A = torch.arange(1, ds+1).float().unsqueeze(0) * 0.03
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(di))
        self.out  = nn.Linear(di, d, bias=False)
        self.nm = nn.RMSNorm(d)
    def forward(self, x):
        B, L, D = x.shape
        res = x
        xz = self.inp(x); u, z = xz.chunk(2, dim=-1)
        u = F.silu(self.cnv(u.transpose(1,2))[:,:,:L].transpose(1,2))
        proj = self.xp(u)
        dt = F.softplus(self.dtp(F.softplus(proj[:,:,:self.ds]))) + 1e-4
        Bs, Cs = proj[:,:,self.ds:self.ds*2], proj[:,:,self.ds*2:]
        de = dt.unsqueeze(-1)
        Abar = torch.exp(de * (-torch.exp(self.A_log)).unsqueeze(0).unsqueeze(1))
        b = de * Bs.unsqueeze(2) * u.unsqueeze(-1)
        eps = 1e-8; logA = torch.log(Abar.clamp(min=eps))
        Acum = torch.exp(torch.cumsum(logA, dim=1))
        h = Acum * torch.cumsum(b / Acum.clamp(min=eps), dim=1)
        y = (h * Cs.unsqueeze(2)).sum(-1) + self.D.unsqueeze(0).unsqueeze(0) * u
        return self.nm(self.out(y * F.silu(z)) + res)

class Ln(nn.Module):
    def __init__(self, d, h=48):
        super().__init__()
        self.gru = nn.GRU(d, h, batch_first=True)
        self.out = nn.Linear(h, d)
    def forward(self, x):
        h, _ = self.gru(x)
        return torch.sigmoid(self.out(h))

class LNMamba(nn.Module):
    def __init__(self, V, d=64, nb=2, ds=16, pred=24, nq=99, dropout=0.1):
        super().__init__()
        self.pred_len = pred; self.nq = nq
        self.emb = nn.Sequential(nn.Linear(V, d*2), nn.GELU(), nn.Dropout(dropout*0.5), nn.Linear(d*2, d))
        self.pe  = nn.Parameter(torch.randn(1, 2000, d) * 0.02)
        self.mb  = nn.ModuleList([Mb(d, ds) for _ in range(nb)])
        self.ln  = nn.ModuleList([Ln(d) for _ in range(nb)])
        self.dec = nn.Sequential(nn.Linear(d, d*2), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d*2, d), nn.GELU(), nn.Linear(d, pred*nq))
        self.drop = nn.Dropout(dropout)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 0.5)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x):
        B, V, L = x.shape
        x = self.emb(x.transpose(1,2)) + self.pe[:,:L]
        for mb, ln in zip(self.mb, self.ln):
            x = self.drop(mb(x))
            x = x * ln(x)
        return self.dec(x[:, -1]).view(B, self.pred_len, self.nq)

def pb_loss(p, t, qt):
    e = t.unsqueeze(-1) - p
    return torch.maximum(qt*e, (qt-1)*e).mean()

# ── Hyperparameter space ──
HP_SPACE = {
    'd_model':   [48, 56, 64, 80, 96],
    'd_state':   [12, 16, 24, 32],
    'n_blocks':  [1, 2, 3],
    'lr':        [3e-4, 5e-4, 8e-4, 1e-3, 2e-3, 3e-3],
    'dropout':   [0.05, 0.08, 0.10, 0.12, 0.15, 0.20],
    'weight_decay': [1e-5, 1e-4, 5e-4, 1e-3, 5e-3],
    'stride':    [2, 3, 4, 6],
    'batch_size': [32, 48, 64],
}

def sample_hp(rng):
    return {k: v[rng.randint(0, len(v))] if isinstance(v, list) else v
            for k, v in HP_SPACE.items()}

# ── Main ──
def main():
    rng = np.random.RandomState(42)
    data, nv, sy = load_zone1()
    T = len(data); te_train = int(T * 0.85); te_val = int(T * 0.92)
    test_data = data[te_val:]

    results = []
    best_val = float('inf'); best_config = None

    print('=' * 60)
    print('LNMamba Random Hyperparameter Search — 30 trials × 8 epochs')
    print('=' * 60)
    sys.stdout.flush()

    for trial in range(30):
        hp = sample_hp(rng)
        s = hp['stride']; b = hp['batch_size']

        train_ds = WDS(data[:te_train], s)
        val_ds   = WDS(data[te_train:te_val], s)

        if len(train_ds) == 0 or len(val_ds) == 0:
            print(f' Trial {trial+1:2d}: SKIP (stride={s} too large)')
            sys.stdout.flush()
            continue

        tl = DataLoader(train_ds, b, shuffle=True, num_workers=0, pin_memory=True)

        model = LNMamba(nv, d=hp['d_model'], nb=hp['n_blocks'], ds=hp['d_state'],
                        pred=PRED, dropout=hp['dropout']).to(DEVICE)

        opt = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=8, eta_min=1e-5)
        scl = torch.amp.GradScaler('cuda')
        qt  = torch.tensor(QUANTILES, dtype=torch.float32, device=DEVICE)

        # Train 8 epochs
        for ep in range(1, 9):
            model.train()
            for x, y in tl:
                x, y = x.to(DEVICE), y.to(DEVICE); opt.zero_grad()
                with torch.amp.autocast('cuda'):
                    loss = pb_loss(model(x), y, qt)
                scl.scale(loss).backward(); scl.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scl.step(opt); scl.update()
            sch.step()

        # Validate
        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            idxs = range(max(0, len(val_ds) - 256), len(val_ds))
            for i in idxs:
                x, y = val_ds[i]
                vp.append(model(x.unsqueeze(0).to(DEVICE)).cpu()); vt.append(y)
        vp = torch.cat(vp); vt = torch.stack(vt, dim=0)
        val_pb = pb_loss(vp.to(DEVICE), vt.to(DEVICE), qt).item()

        results.append({'val_pb': val_pb, 'hp': hp})
        star = ' ★' if val_pb < best_val else ''
        if val_pb < best_val:
            best_val = val_pb; best_config = hp

        print(f' Trial {trial+1:2d}: pb={val_pb:.4f} d={hp["d_model"]} ds={hp["d_state"]} '
              f'nb={hp["n_blocks"]} lr={hp["lr"]:.1e} do={hp["dropout"]} '
              f'wd={hp["weight_decay"]:.1e} s={s} b={b}{star}')
        sys.stdout.flush()

    # ── Best results ──
    results.sort(key=lambda x: x['val_pb'])
    print(f'\n{"="*60}')
    print(f'TOP 10 CONFIGURATIONS')
    print(f'{"="*60}')
    for i, r in enumerate(results[:10]):
        hp = r['hp']
        print(f'  {i+1:2d}. pb={r["val_pb"]:.4f} d={hp["d_model"]} ds={hp["d_state"]} '
              f'nb={hp["n_blocks"]} lr={hp["lr"]:.1e} do={hp["dropout"]} '
              f'wd={hp["weight_decay"]:.1e} s={hp["stride"]} b={hp["batch_size"]}')

    # ── Train final model with best config ──
    print(f'\n{"="*60}')
    print(f'FINAL MODEL: Best config, 30 epochs, test set evaluation')
    print(f'{"="*60}')
    sys.stdout.flush()

    hp = best_config
    train_ds = WDS(data[:te_train], hp['stride'])
    tl = DataLoader(train_ds, hp['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    print(f'Train: {len(train_ds):,} samples | Best config: {hp}')

    model = LNMamba(nv, d=hp['d_model'], nb=hp['n_blocks'], ds=hp['d_state'],
                    pred=PRED, dropout=hp['dropout']).to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters())
    print(f'Params: {n_p:,}')
    sys.stdout.flush()

    opt = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=12, T_mult=2, eta_min=1e-5)
    scl = torch.amp.GradScaler('cuda')
    qt  = torch.tensor(QUANTILES, dtype=torch.float32, device=DEVICE)

    for ep in range(1, 31):
        t0 = time.time(); model.train()
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE); opt.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = pb_loss(model(x), y, qt)
            scl.scale(loss).backward(); scl.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scl.step(opt); scl.update()
        sch.step()
        et = time.time() - t0
        print(f'  E {ep:2d} {et:.0f}s'); sys.stdout.flush()

    # ── Test evaluation (stride=4 for fair comparison with v1) ──
    test_ds = WDS(test_data, 4)
    test_loader = DataLoader(test_ds, 64, shuffle=False, num_workers=0, pin_memory=True)
    print(f'Test: {len(test_ds)} samples (stride=4)')

    model.eval(); preds, targs = [], []; total_pb = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            total_pb += pb_loss(out, y, qt).item()
            preds.append(out.cpu().numpy()); targs.append(y.cpu().numpy())

    pr = np.concatenate(preds); tr = np.concatenate(targs)
    test_pb = total_pb / len(test_loader)
    sh = pr.shape
    pr_mw = sy.inverse_transform(pr.reshape(-1, sh[2])).reshape(sh)
    tr_mw = sy.inverse_transform(tr.reshape(-1, 1)).reshape(tr.shape)
    p50 = pr_mw[:, :, 49]; pf = p50.ravel(); tf = tr_mw.ravel(); mask = tf > 0.001
    rmse = np.sqrt(np.mean((pf[mask] - tf[mask])**2))
    mae  = np.mean(np.abs(pf[mask] - tf[mask]))
    r2   = 1 - np.sum((tf[mask]-pf[mask])**2) / (np.sum((tf[mask]-np.mean(tf[mask]))**2) + 1e-8)

    print(f'\nTEST: Pinball={test_pb:.4f} | R2={r2:.4f} | RMSE={rmse:.4f} | MAE={mae:.4f}')
    print('Per-horizon:')
    ph_pb = []
    for h in [0, 3, 5, 11, 17, 23]:
        er = torch.FloatTensor(tr_mw[:, h]).unsqueeze(-1) - torch.FloatTensor(pr_mw[:, h])
        pb_h = torch.maximum(torch.FloatTensor(QUANTILES)*er, (torch.FloatTensor(QUANTILES)-1)*er).mean().item()
        ph_pb.append(pb_h)
        print(f'  +{h+1:2d}h: {pb_h:.4f}')

    v1_pb = 0.2069
    print(f'\nv1: 0.2069 | best: {test_pb:.4f} | imp: {(v1_pb-test_pb)/v1_pb*100:+.1f}%')
    print(f'Best config: {best_config}')
    print(f'PB avg across horizons: {np.mean(ph_pb):.4f}')

    os.makedirs('checkpoints', exist_ok=True)
    torch.save({'model': model.state_dict(), 'config': best_config, 'test_pb': test_pb},
               'checkpoints/random_search_best.pt')
    with open('checkpoints/random_search_results.json', 'w') as f:
        json.dump([{'val_pb': float(r['val_pb']), 'hp': r['hp']} for r in results], f, indent=2)
    print('\nSaved results.')
    print('Done!')


if __name__ == '__main__':
    main()
