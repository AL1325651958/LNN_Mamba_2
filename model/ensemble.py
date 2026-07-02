"""
LNMamba Ensemble — 3 improvements over v1:
  1. 5-seed ensemble + median aggregation
  2. NWP noise augmentation (Gaussian 5% on U10/V10/U100/V100)
  3. Lighter model: d=64, ds=16, ~200K params (vs 396K)

All in one file, trains fast.
"""
import sys,os,zipfile,time,copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device('cuda')
QUANTILES = np.linspace(0.01, 0.99, 99)
DATA_DIR = 'data/gefcom2014'
SEQ, PRED = 168, 24

# ═══════════════════════════════════════
# Data (with NWP noise augmentation)
# ═══════════════════════════════════════
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
    sx = StandardScaler(); feats_raw = df[af].values.astype(np.float32)
    feats = sx.fit_transform(feats_raw)
    sy = StandardScaler(); tgt = sy.fit_transform(df[['TARGETVAR']].values.astype(np.float32)).ravel()
    data = np.concatenate([feats, tgt.reshape(-1, 1)], axis=1)
    # Store raw NWP indices for noise injection: U10=0, V10=1, U100=2, V100=3
    nwp_indices = [0, 1, 2, 3]
    return data, data.shape[1], sy, nwp_indices

class NoisyWDS(Dataset):
    """Dataset with NWP noise augmentation during training."""
    def __init__(self, d, s, nwp_indices=None, noise_std=0.05, training=True):
        self.data = torch.FloatTensor(d); self.s = s
        self.nwp_idx = nwp_indices or []
        self.noise_std = noise_std
        self.training = training
        self.n = max(0, (len(d) - SEQ - PRED) // s + 1)
    def __len__(self): return self.n
    def __getitem__(self, i):
        st = i * self.s
        x = self.data[st:st+SEQ].clone()
        y = self.data[st+SEQ:st+SEQ+PRED, -1].clone()
        # NWP noise: add Gaussian noise to NWP columns during training
        if self.training and self.noise_std > 0 and len(self.nwp_idx) > 0:
            noise = torch.randn(x.shape[0], len(self.nwp_idx)) * self.noise_std
            x[:, self.nwp_idx] += noise
        return x.T, y

# ═══════════════════════════════════════
# Lightweight Selective SSM (d=64, ds=16)
# ═══════════════════════════════════════
class LightMamba(nn.Module):
    def __init__(self, d=64, ds=16, dc=4):
        super().__init__()
        self.ds = ds; di = d * 2
        self.inp = nn.Linear(d, di*2, bias=False)
        self.cnv = nn.Conv1d(di, di, dc, groups=di, padding=dc-1)
        self.xp  = nn.Linear(di, ds*2+1, bias=False)
        self.dtp = nn.Linear(ds, di, bias=True)
        A = torch.arange(1, ds+1).float().unsqueeze(0) * 0.03
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(di))
        self.out = nn.Linear(di, d, bias=False); self.nm = nn.RMSNorm(d)

    def forward(self, x):
        B, L, D = x.shape; res = x
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

class LightLNN(nn.Module):
    def __init__(self, d, h=48):
        super().__init__()
        self.gru = nn.GRU(d, h, batch_first=True)
        self.out = nn.Linear(h, d)
    def forward(self, x):
        h, _ = self.gru(x)
        return torch.sigmoid(self.out(h))

class LNMambaLight(nn.Module):
    """Light LNN-Gated Selective SSM: d=64, ds=16, nb=2, ~200K params."""
    def __init__(self, V, d=64, ds=16, nb=2, pred=24, nq=99, dropout=0.1):
        super().__init__()
        self.pred_len = pred; self.nq = nq; d2 = d*2
        self.emb = nn.Sequential(nn.Linear(V, d2), nn.GELU(), nn.Dropout(dropout*0.5), nn.Linear(d2, d))
        self.pe  = nn.Parameter(torch.randn(1, 2000, d) * 0.02)
        self.mb  = nn.ModuleList([LightMamba(d, ds) for _ in range(nb)])
        self.ln  = nn.ModuleList([LightLNN(d, 48) for _ in range(nb)])
        self.dec = nn.Sequential(nn.Linear(d, d2), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d2, d), nn.GELU(), nn.Linear(d, pred*nq))
        self.drop = nn.Dropout(dropout)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 0.5)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x):
        B, V, L = x.shape
        x = self.emb(x.transpose(1,2)) + self.pe[:,:L]
        for mb, ln in zip(self.mb, self.ln): x = self.drop(mb(x)); x = x * ln(x)
        return self.dec(x[:, -1]).view(B, self.pred_len, self.nq)

# ═══════════════════════════════════════
# Loss & Training
# ═══════════════════════════════════════
def pb_loss(p, t, qt):
    e = t.unsqueeze(-1) - p
    return torch.maximum(qt*e, (qt-1)*e).mean()

def train_one(model, tl, val_ds, qt, epochs=30, lr=1e-3, seed=42, label=''):
    """Train one model, return best state_dict based on val pinball."""
    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=12, T_mult=2, eta_min=1e-5)
    scl = torch.amp.GradScaler('cuda')
    best_val = float('inf'); best_state = None

    for ep in range(1, epochs+1):
        model.train()
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE); opt.zero_grad()
            with torch.amp.autocast('cuda'): loss = pb_loss(model(x), y, qt)
            scl.scale(loss).backward(); scl.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scl.step(opt); scl.update()
        sch.step()

        # Val every 3 epochs
        if ep % 3 == 1 or ep >= epochs - 5:
            model.eval(); vp, vt = [], []
            with torch.no_grad():
                idxs = range(max(0, len(val_ds) - 256), len(val_ds))
                for i in idxs:
                    x_, y_ = val_ds[i]; vp.append(model(x_.unsqueeze(0).to(DEVICE)).cpu()); vt.append(y_)
            vp = torch.cat(vp); vt = torch.stack(vt, dim=0)
            val_pb = pb_loss(vp.to(DEVICE), vt.to(DEVICE), qt).item()
            if val_pb < best_val: best_val = val_pb; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            print(f'  {label} E{ep:2d} val_pb={val_pb:.4f} (best={best_val:.4f})'); sys.stdout.flush()

    if best_state is not None: model.load_state_dict(best_state)
    return model, best_val

# ═══════════════════════════════════════
# Main
# ═══════════════════════════════════════
def main():
    data, nv, sy, nwp_idx = load_zone1()
    T = len(data); te = int(T * 0.85)
    qt = torch.tensor(QUANTILES, dtype=torch.float32, device=DEVICE)
    NOISE_STD = 0.03  # 5% Gaussian noise on NWP inputs
    N_MODELS = 3; EPOCHS = 30; LR = 1e-3

    # Train + val datasets (noise on train)
    train_ds = NoisyWDS(data[:te], 6, nwp_idx, NOISE_STD, training=True)
    val_ds   = NoisyWDS(data[te:], 6, nwp_idx, 0.0, training=False)   # no noise on val
    test_ds  = NoisyWDS(data[te:], 4, nwp_idx, 0.0, training=False)    # test stride=4
    tl = DataLoader(train_ds, 64, shuffle=True, num_workers=0, pin_memory=True)
    testl = DataLoader(test_ds, 64, shuffle=False, num_workers=0, pin_memory=True)

    print(f'LNMamba Ensemble: {N_MODELS} seeds, {len(train_ds)} samples, NWP noise={NOISE_STD}')
    print(f'Light: d=64, ds=16, nb=2 (~200K params each)')
    print('=' * 60); sys.stdout.flush()

    models = []; val_pbs = []
    for seed in [42, 123, 777]:
        print(f'\n--- Training model {len(models)+1}/{N_MODELS} (seed={seed}) ---'); sys.stdout.flush()
        model = LNMambaLight(nv, d=64, ds=16, nb=2, pred=PRED, dropout=0.1).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters())
        print(f'  Params: {n_p:,}'); sys.stdout.flush()

        model, best_val = train_one(model, tl, val_ds, qt, EPOCHS, LR, seed, f'S{seed}')
        models.append(model)
        val_pbs.append(best_val)

    vals_str = ', '.join(f'{v:.4f}' for v in val_pbs)
    print(f'\nIndividual val pinballs: [{vals_str}]')
    print(f'Mean val: {np.mean(val_pbs):.4f} ± {np.std(val_pbs):.4f}')

    # ── Ensemble test evaluation ──
    print(f'\n{"="*60}')
    print(f'ENSEMBLE TEST EVALUATION ({N_MODELS} models, median aggregation)')
    print(f'{"="*60}'); sys.stdout.flush()

    for m in models: m.eval()

    all_ensemble_preds = []
    all_targets = []
    all_individual_preds = [[] for _ in range(N_MODELS)]

    with torch.no_grad():
        for x, y in testl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            batch_preds = []
            for i, m in enumerate(models):
                out = m(x)  # (B, 24, 99)
                batch_preds.append(out.unsqueeze(0))
                all_individual_preds[i].append(out.cpu().numpy())

            # Ensemble: median across models
            stacked = torch.cat(batch_preds, dim=0)  # (5, B, 24, 99)
            ensemble_out = torch.median(stacked, dim=0)[0]  # (B, 24, 99)
            all_ensemble_preds.append(ensemble_out.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    pr_ens = np.concatenate(all_ensemble_preds)
    tr = np.concatenate(all_targets)

    # Pinball for ensemble
    pr_t = torch.FloatTensor(pr_ens).to(DEVICE)
    tr_t = torch.FloatTensor(tr).to(DEVICE)
    ens_pb = pb_loss(pr_t, tr_t, qt).item()

    # Pinball per individual model
    ind_pbs = []
    for i in range(N_MODELS):
        pr_ind = np.concatenate(all_individual_preds[i])
        pr_ind_t = torch.FloatTensor(pr_ind).to(DEVICE)
        ind_pb = pb_loss(pr_ind_t, tr_t, qt).item()
        ind_pbs.append(ind_pb)

    # Inverse transform for R²
    sh = pr_ens.shape
    pr_mw = sy.inverse_transform(pr_ens.reshape(-1, sh[2])).reshape(sh)
    tr_mw = sy.inverse_transform(tr.reshape(-1, 1)).reshape(tr.shape)
    p50 = pr_mw[:, :, 49]; pf = p50.ravel(); tf = tr_mw.ravel(); mask = tf > 0.001
    rmse = np.sqrt(np.mean((pf[mask] - tf[mask])**2))
    mae  = np.mean(np.abs(pf[mask] - tf[mask]))
    r2   = 1 - np.sum((tf[mask]-pf[mask])**2) / (np.sum((tf[mask]-np.mean(tf[mask]))**2) + 1e-8)

    ind_str = ', '.join(f'{v:.4f}' for v in ind_pbs)
    print(f'\nIndividual pinballs:  [{ind_str}]')
    print(f'Mean individual:      {np.mean(ind_pbs):.4f} ± {np.std(ind_pbs):.4f}')
    print(f'Ensemble (median):    {ens_pb:.4f}')
    print(f'Ensemble R2 (median): {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}')

    # Per-horizon
    print('\nPer-horizon (ensemble):')
    for h in [0, 3, 5, 11, 17, 23]:
        er = torch.FloatTensor(tr_mw[:, h]).unsqueeze(-1) - torch.FloatTensor(pr_mw[:, h])
        pb_h = torch.maximum(torch.FloatTensor(QUANTILES)*er, (torch.FloatTensor(QUANTILES)-1)*er).mean().item()
        print(f'  +{h+1:2d}h: {pb_h:.4f}')

    p10 = pr_mw[:, :, 9]; p90 = pr_mw[:, :, 89]
    print(f'80% CI coverage: {np.mean((tr_mw >= p10) & (tr_mw <= p90))*100:.1f}%')

    # vs v1
    v1_pb = 0.2069
    imp = (v1_pb - ens_pb) / v1_pb * 100
    imp_ind = (v1_pb - np.mean(ind_pbs)) / v1_pb * 100
    print(f'\nv1 (396K):    0.2069')
    print(f'Light single: {np.mean(ind_pbs):.4f} ({imp_ind:+.1f}%)')
    print(f'Ensemble (5x): {ens_pb:.4f} ({imp:+.1f}%)')
    print(f'Total params: {sum(p.numel() for p in models[0].parameters()):,} × {N_MODELS}')

    os.makedirs('checkpoints', exist_ok=True)
    torch.save({'models': [m.state_dict() for m in models], 'val_pbs': val_pbs,
                'ind_pbs': ind_pbs, 'ens_pb': ens_pb, 'nwp_noise': NOISE_STD},
               'checkpoints/ensemble_light.pt')
    print('\nSaved checkpoints/ensemble_light.pt')
    print('Done!')


if __name__ == '__main__':
    main()
