"""
Bayesian hyperparameter optimization for LNMamba on GEFCom2014 Zone 1.
Uses Optuna TPE sampler — 50 trials, each 10 epochs.
"""
import sys,os,zipfile,time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd, optuna
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device('cuda')
QUANTILES = np.linspace(0.01, 0.99, 99)
DATA_DIR = 'data/gefcom2014'
SEQ, PRED = 168, 24

# ── Dataset (reusable) ──
class WDS(Dataset):
    def __init__(self, d, s):
        self.data = torch.FloatTensor(d); self.s = s
        self.n = max(0, (len(d) - SEQ - PRED) // s + 1)
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        st = i * self.s
        return (self.data[st:st+SEQ].T, self.data[st+SEQ:st+SEQ+PRED, -1])

class WDSFixed(Dataset):
    """Stride=4 variant for test."""
    def __init__(self, d): self.data = torch.FloatTensor(d); self.n = max(0, (len(d) - SEQ - PRED) // 4 + 1)
    def __len__(self): return self.n
    def __getitem__(self, i): st = i * 4; return (self.data[st:st+SEQ].T, self.data[st+SEQ:st+SEQ+PRED, -1])


# ── Data: Zone 1, cached ──
def load_data(stride=4):
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
    T = len(data); te = int(T * 0.85); ve = int(T * 0.92)
    train_ds = WDS(data[:te], stride)
    val_ds   = WDS(data[te:ve], stride)
    return train_ds, val_ds, data.shape[1], sy

# ── Selective SSM (lightweight) ──
class Mb(nn.Module):
    def __init__(self, d, ds=16, dc=4, ex=2):
        super().__init__()
        self.ds = ds
        di = d * ex
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
        xz = self.inp(x)
        u, z = xz.chunk(2, dim=-1)
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
        self.pred_len = pred
        self.nq = nq
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

def pinball_loss(p, t, qt):
    e = t.unsqueeze(-1) - p; return torch.maximum(qt*e, (qt-1)*e).mean()

# ── Optuna Objective ──
class Objective:
    def __init__(self, n_trials_per_config=1):
        self.best_pb = float('inf'); self.best_params = None
        self.trial_idx = 0

    def __call__(self, trial):
        # ── Hyperparameters ──
        d_model   = trial.suggest_categorical('d_model', [48, 56, 64, 80, 96])
        d_state   = trial.suggest_categorical('d_state', [12, 16, 24, 32])
        n_blocks  = trial.suggest_int('n_blocks', 1, 3)
        lr        = trial.suggest_float('lr', 2e-4, 3e-3, log=True)
        dropout   = trial.suggest_float('dropout', 0.05, 0.25)
        wd        = trial.suggest_float('weight_decay', 1e-5, 5e-3, log=True)
        stride    = trial.suggest_categorical('stride', [2, 3, 4, 6])
        batch     = trial.suggest_categorical('batch_size', [32, 48, 64])

        print(f' Trial {self.trial_idx+1}: d={d_model} ds={d_state} nb={n_blocks} '
              f'lr={lr:.2e} do={dropout:.2f} wd={wd:.2e} s={stride} b={batch}')
        sys.stdout.flush()

        # Data
        train_ds, val_ds, nv, _ = load_data(stride)
        tl = DataLoader(train_ds, batch, shuffle=True, num_workers=0, pin_memory=True)

        # Model
        model = LNMamba(nv, d=d_model, nb=n_blocks, ds=d_state, pred=PRED, dropout=dropout).to(DEVICE)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=1e-5)
        scl = torch.amp.GradScaler('cuda')
        qt  = torch.tensor(QUANTILES, dtype=torch.float32, device=DEVICE)

        # Train 10 epochs
        best_val = float('inf')
        for ep in range(1, 11):
            model.train()
            for x, y in tl:
                x, y = x.to(DEVICE), y.to(DEVICE); opt.zero_grad()
                with torch.amp.autocast('cuda'): loss = pinball_loss(model(x), y, qt)
                scl.scale(loss).backward(); scl.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scl.step(opt); scl.update()
            sch.step()

            # Validate on subset (last 256 val samples)
            model.eval(); vp, vt = [], []
            with torch.no_grad():
                idxs = range(max(0, len(val_ds)-256), len(val_ds))
                for i in idxs:
                    x, y = val_ds[i]; vp.append(model(x.unsqueeze(0).to(DEVICE)).cpu()); vt.append(y)
            vp = torch.cat(vp); vt = torch.FloatTensor(vt)
            val_pb = pinball_loss(vp.to(DEVICE), vt.to(DEVICE), qt).item()
            if val_pb < best_val: best_val = val_pb

            # Report to Optuna (pruning)
            trial.report(val_pb, ep)
            if trial.should_prune(): raise optuna.TrialPruned()

        self.trial_idx += 1
        if best_val < self.best_pb:
            self.best_pb = best_val; self.best_params = trial.params
            print(f'  ★ NEW BEST: pb={best_val:.4f} params={trial.params}')
            sys.stdout.flush()

        return best_val


# ── Main ──
def main():
    print('LNMamba Bayesian Hyperparameter Optimization — Optuna TPE')
    print(f'50 trials, 10 epochs each, Zone 1')
    print('=' * 60)
    sys.stdout.flush()

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    obj = Objective()
    study.optimize(obj, n_trials=50, show_progress_bar=False)

    print('\n' + '=' * 60)
    print('BEST CONFIGURATION')
    print('=' * 60)
    print(f'  Val Pinball: {study.best_value:.4f}')
    for k, v in study.best_params.items():
        print(f'  {k}: {v}')

    # Save
    import json
    os.makedirs('checkpoints', exist_ok=True)
    with open('checkpoints/best_hyperparams.json', 'w') as f:
        json.dump({'best_value': float(study.best_value), 'params': study.best_params}, f, indent=2)
    print(f'\nSaved to checkpoints/best_hyperparams.json')

    # ── Train final model with best config ──
    print('\n' + '=' * 60)
    print('TRAINING FINAL MODEL WITH BEST HPARAMS (30 epochs)')
    print('=' * 60)
    sys.stdout.flush()

    bp = study.best_params
    train_ds, val_ds, nv, sy = load_data(bp['stride'])
    tl = DataLoader(train_ds, bp['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    test_ds, _, _, _ = load_data(4)  # test on stride=4 for comparability

    # Re-read test from proper split
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
    sx2 = StandardScaler(); feats2 = sx2.fit_transform(df[af].values.astype(np.float32))
    sy2 = StandardScaler(); tgt2 = sy2.fit_transform(df[['TARGETVAR']].values.astype(np.float32)).ravel()
    data2 = np.concatenate([feats2, tgt2.reshape(-1, 1)], axis=1)
    T2 = len(data2); te2 = int(T2 * 0.85)
    test_loader = DataLoader(WDSFixed(data2[te2:]), 64, shuffle=False, num_workers=0, pin_memory=True)
    print(f'Test: {len(test_loader.dataset)} samples (stride=4)')

    model = LNMamba(nv, d=bp['d_model'], nb=bp['n_blocks'], ds=bp['d_state'], pred=PRED, dropout=bp['dropout']).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=bp['lr'], weight_decay=bp['weight_decay'])
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=12, T_mult=2, eta_min=1e-5)
    scl = torch.amp.GradScaler('cuda'); qt = torch.tensor(QUANTILES, dtype=torch.float32, device=DEVICE)
    best_pb = float('inf'); best_state = None

    for ep in range(1, 31):
        t0 = time.time(); model.train()
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE); opt.zero_grad()
            with torch.amp.autocast('cuda'): loss = pinball_loss(model(x), y, qt)
            scl.scale(loss).backward(); scl.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scl.step(opt); scl.update()
        sch.step()
        et = time.time() - t0
        star = ' ★' if ep % 5 == 0 else ''
        print(f'E {ep:2d} {et:.0f}s{star}'); sys.stdout.flush()

    # Test
    model.eval(); preds, targs = [], []; total_pb = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE); out = model(x)
            total_pb += pinball_loss(out, y, qt).item()
            preds.append(out.cpu().numpy()); targs.append(y.cpu().numpy())
    pr = np.concatenate(preds); tr = np.concatenate(targs); test_pb = total_pb / len(test_loader)
    sh = pr.shape
    pr_mw = sy2.inverse_transform(pr.reshape(-1, sh[2])).reshape(sh)
    tr_mw = sy2.inverse_transform(tr.reshape(-1, 1)).reshape(tr.shape)
    p50 = pr_mw[:, :, 49]; pf = p50.ravel(); tf = tr_mw.ravel(); mask = tf > 0.001
    rmse = np.sqrt(np.mean((pf[mask] - tf[mask])**2))
    mae  = np.mean(np.abs(pf[mask] - tf[mask]))
    r2   = 1 - np.sum((tf[mask]-pf[mask])**2) / (np.sum((tf[mask]-np.mean(tf[mask]))**2) + 1e-8)

    print(f'\nFINAL TEST: Pinball={test_pb:.4f} | R2={r2:.4f} | RMSE={rmse:.4f} | MAE={mae:.4f}')
    print('Per-horizon:')
    for h in [0, 3, 5, 11, 17, 23]:
        er = torch.FloatTensor(tr_mw[:, h]).unsqueeze(-1) - torch.FloatTensor(pr_mw[:, h])
        pb_h = torch.maximum(torch.FloatTensor(QUANTILES)*er, (torch.FloatTensor(QUANTILES)-1)*er).mean().item()
        print(f'  +{h+1:2d}h: {pb_h:.4f}')

    v1_pb = 0.2069
    print(f'\nv1: 0.2069 | best: {test_pb:.4f} | imp: {(v1_pb-test_pb)/v1_pb*100:+.1f}%')
    print(f'Best params: {study.best_params}')
    print('Done!')


if __name__ == '__main__':
    main()
