"""
LNMamba v3 — Memory-optimized with O(L) sequential scan, single-direction Mamba.
Heavy optimization: CRPS + crossing penalty + multiscale conv + 10-zone joint + 50 epochs.
"""
import sys,os,zipfile,time,argparse,io,math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

DATA_DIR = 'data/gefcom2014'
QUANTILES = np.linspace(0.01, 0.99, 99)

# ── Weather ──
def weather_features(df):
    df['WS10']  = np.sqrt(df['U10']**2 + df['V10']**2)
    df['WS100'] = np.sqrt(df['U100']**2 + df['V100']**2)
    df['WD10_SIN'] = np.sin(np.arctan2(df['U10'], df['V10']))
    df['WD10_COS'] = np.cos(np.arctan2(df['U10'], df['V10']))
    df['WD100_SIN'] = np.sin(np.arctan2(df['U100'], df['V100']))
    df['WD100_COS'] = np.cos(np.arctan2(df['U100'], df['V100']))
    df['SHEAR'] = df['WS100'] / (df['WS10'] + 0.1)
    df['TURB']  = df['U10'].diff().abs().fillna(0).clip(0, 20)
    return df
FEAT_COLS = ['U10','V10','U100','V100','WS10','WS100',
             'WD10_SIN','WD10_COS','WD100_SIN','WD100_COS','SHEAR','TURB']

# ── Memory-efficient Mamba ──
class MambaSSM(nn.Module):
    """Mamba SSM with sequential scan (O(L), memory efficient)."""
    def __init__(self, d, ds=32, dc=4, ex=2):
        super().__init__()
        self.ds = ds; di = d * ex
        self.inp  = nn.Linear(d, di * 2, bias=False)
        self.cnv  = nn.Conv1d(di, di, dc, groups=di, padding=dc - 1)
        self.xp   = nn.Linear(di, ds * 2 + 1, bias=False)  # [dt, B, C]
        self.dtp  = nn.Linear(ds, di, bias=True)
        A = torch.arange(1, ds + 1).float().unsqueeze(0) * 0.03
        self.A_log = nn.Parameter(torch.log(A))
        self.D     = nn.Parameter(torch.ones(di))
        self.out  = nn.Linear(di, d, bias=False)
        self.nm   = nn.RMSNorm(d)

    def forward(self, x):
        B, L, D = x.shape; res = x
        xz = self.inp(x); u, z = xz.chunk(2, dim=-1)
        u = F.silu(self.cnv(u.transpose(1, 2))[:, :, :L].transpose(1, 2))

        proj = self.xp(u)  # (B, L, 1 + ds*2)
        dt_raw = F.softplus(proj[:, :, :1])  # (B, L, 1) — scalar dt
        Bs = proj[:, :, 1:1 + self.ds]       # (B, L, ds)
        Cs = proj[:, :, 1 + self.ds:]         # (B, L, ds)
        dt = F.softplus(self.dtp(dt_raw.repeat(1, 1, self.ds))) + 1e-4  # (B, L, di)

        de = dt.unsqueeze(-1)  # (B, L, di, 1)
        A_bar = torch.exp(de * (-torch.exp(self.A_log)).unsqueeze(0).unsqueeze(1))  # (B, L, di, ds)

        # Sequential scan — memory-friendly
        h = torch.zeros(B, de.shape[2], self.ds, device=x.device, dtype=x.dtype)
        y_seq = []
        for t in range(L):
            h = A_bar[:, t] * h + \
                de[:, t] * Bs[:, t].unsqueeze(1) * u[:, t].unsqueeze(-1)
            y_t = (h * Cs[:, t].unsqueeze(1)).sum(-1) + self.D.unsqueeze(0) * u[:, t]
            y_seq.append(y_t)
        y = torch.stack(y_seq, dim=1)
        return self.nm(self.out(y * F.silu(z)) + res)


# ── LNN Gate ──
class LNNGate(nn.Module):
    def __init__(self, d, h=48):
        super().__init__()
        self.gru = nn.GRU(d, h, batch_first=True)
        self.out = nn.Linear(h, d)

    def forward(self, x):
        h, _ = self.gru(x)
        return torch.sigmoid(self.out(h))


# ── Multiscale Frontend ──
class MultiScaleFrontend(nn.Module):
    def __init__(self, d):
        super().__init__()
        d4 = d // 4
        self.c1 = nn.Conv1d(d, d4, 3, padding=2, groups=1)  # ~1h
        self.c2 = nn.Conv1d(d, d4, 7, padding=6, groups=1)  # ~3h
        self.c3 = nn.Conv1d(d, d4, 13, padding=12, groups=1)  # ~6h
        self.c4 = nn.Conv1d(d, d4, 25, padding=24, groups=1)  # ~24h
        self.fuse = nn.Linear(d, d)
        self.nm = nn.LayerNorm(d)

    def forward(self, x):
        B, L, D = x.shape
        xt = x.transpose(1, 2)
        b1 = F.gelu(self.c1(xt))[:, :, :L]
        b2 = F.gelu(self.c2(xt))[:, :, :L]
        b3 = F.gelu(self.c3(xt))[:, :, :L]
        b4 = F.gelu(self.c4(xt))[:, :, :L]
        multi = torch.cat([b1, b2, b3, b4], dim=1).transpose(1, 2)
        return self.nm(x + self.fuse(multi))


# ── LNMamba v3 ──
class LNMambaV3(nn.Module):
    def __init__(self, V, d=64, nb=3, ds=16, pred=24, nq=99):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(V, d*2), nn.GELU(), nn.Linear(d*2, d))
        self.pe  = nn.Parameter(torch.randn(1, 2000, d) * 0.02)
        self.ms  = MultiScaleFrontend(d)
        self.mb  = nn.ModuleList([MambaSSM(d, ds) for _ in range(nb)])
        self.ln  = nn.ModuleList([LNNGate(d, h=48) for _ in range(nb)])
        self.dec = nn.Sequential(
            nn.Linear(d, d*2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d*2, d), nn.GELU(),
            nn.Linear(d, pred * nq)
        )
        self.pred_len = pred; self.nq = nq
        self.drop = nn.Dropout(0.08)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 0.5)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        B, V, L = x.shape
        x = self.emb(x.transpose(1, 2)) + self.pe[:, :L]
        x = self.ms(x)
        for mb, gate in zip(self.mb, self.ln):
            x = self.drop(mb(x))
            x = x * gate(x)
        return self.dec(x[:, -1]).view(B, self.pred_len, self.nq)


# ── Loss ──
def pinball_loss(pred_q, target, q_t):
    e = target.unsqueeze(-1) - pred_q
    return torch.maximum(q_t * e, (q_t - 1) * e).mean()

def crossing_penalty(pred_q):
    d = pred_q[:, :, 1:] - pred_q[:, :, :-1]
    return F.relu(-d).mean()

def crps_approx(pred_q, target, q_t):
    I = (target.unsqueeze(-1) <= pred_q).float()
    cdf = q_t.unsqueeze(0).unsqueeze(0)
    return F.mse_loss(I, cdf.expand_as(I)) * 0.5

# ── Data ──
class NWPDS(Dataset):
    def __init__(self, data, seq, pred, stride):
        self.data = torch.FloatTensor(data)
        self.seq=seq; self.pred=pred; self.s=stride
        self.n = max(0, (len(data)-seq-pred)//stride+1)
    def __len__(self): return self.n
    def __getitem__(self, i):
        st = i*self.s
        return (self.data[st:st+self.seq].T,
                self.data[st+self.seq:st+self.seq+self.pred, -1])

def load_all(seq=168, pred=24, batch=48, stride=4):
    datasets = []; nv = None
    for z in range(1, 11):
        tz = zipfile.ZipFile(f'{DATA_DIR}/Task15_W_Zone1_10.zip')
        df = pd.read_csv(tz.open(f'Task15_W_Zone1_10/Task15_W_Zone{z}.csv'))
        ts = df['TIMESTAMP'].astype(str).str.strip()
        df['dt'] = pd.to_datetime(ts.str[:8], format='%Y%m%d') \
                   + pd.to_timedelta(ts.str.extract(r'(\d+):')[0].astype(int), unit='h')
        df = df.sort_values('dt').reset_index(drop=True)
        df['TARGETVAR'] = df['TARGETVAR'].interpolate(limit_direction='both')
        for c in ['U10','V10','U100','V100']: df[c] = df[c].interpolate(limit_direction='both')
        df = weather_features(df)
        h = df['dt'].dt.hour.values.astype(np.float32)
        m = df['dt'].dt.month.values.astype(np.float32)
        df['HOUR_SIN'] = np.sin(2*np.pi*h/24); df['HOUR_COS'] = np.cos(2*np.pi*h/24)
        df['MONTH_SIN'] = np.sin(2*np.pi*m/12); df['MONTH_COS'] = np.cos(2*np.pi*m/12)
        af = FEAT_COLS + ['HOUR_SIN','HOUR_COS','MONTH_SIN','MONTH_COS']
        feats = StandardScaler().fit_transform(df[af].values.astype(np.float32))
        tgt = StandardScaler().fit_transform(df[['TARGETVAR']].values.astype(np.float32)).ravel()
        data = np.concatenate([feats, tgt.reshape(-1,1)], axis=1)
        if nv is None: nv = data.shape[1]
        T = len(data); te = int(T * 0.85)
        datasets.append(NWPDS(data[:te], seq, pred, stride))
    ds_full = torch.utils.data.ConcatDataset(datasets)
    print(f'All zones: {len(ds_full):,} train samples')

    # Zone 1 test
    tz = zipfile.ZipFile(f'{DATA_DIR}/Task15_W_Zone1_10.zip')
    df = pd.read_csv(tz.open('Task15_W_Zone1_10/Task15_W_Zone1.csv'))
    ts = df['TIMESTAMP'].astype(str).str.strip()
    df['dt'] = pd.to_datetime(ts.str[:8], format='%Y%m%d') \
               + pd.to_timedelta(ts.str.extract(r'(\d+):')[0].astype(int), unit='h')
    df = df.sort_values('dt').reset_index(drop=True)
    df['TARGETVAR'] = df['TARGETVAR'].interpolate(limit_direction='both')
    for c in ['U10','V10','U100','V100']: df[c] = df[c].interpolate(limit_direction='both')
    df = weather_features(df)
    df['HOUR_SIN'] = np.sin(2*np.pi*df['dt'].dt.hour.values/24)
    df['HOUR_COS'] = np.cos(2*np.pi*df['dt'].dt.hour.values/24)
    df['MONTH_SIN'] = np.sin(2*np.pi*df['dt'].dt.month.values/12)
    df['MONTH_COS'] = np.cos(2*np.pi*df['dt'].dt.month.values/12)
    feats2 = StandardScaler().fit_transform(df[af].values.astype(np.float32))
    scaler_y2 = StandardScaler()
    tgt2 = scaler_y2.fit_transform(df[['TARGETVAR']].values.astype(np.float32)).ravel()
    data2 = np.concatenate([feats2, tgt2.reshape(-1,1)], axis=1)
    T2 = len(data2); te2 = int(T2 * 0.85)
    ds_test = NWPDS(data2[te2:], seq, pred, stride)

    dl_train = DataLoader(ds_full, batch, shuffle=True, num_workers=0, pin_memory=True)
    dl_test  = DataLoader(ds_test, batch, shuffle=False, num_workers=0, pin_memory=True)
    return dl_train, dl_test, nv, scaler_y2

# ── Main ──
def main():
    args = argparse.Namespace(seq=168, pred=24, stride=4, batch=16, epochs=50,
                              d_model=64, lr=5e-4)
    device = torch.device('cuda')
    q_t = torch.tensor(QUANTILES, dtype=torch.float32, device=device)

    print(f'LNMamba v3 — d={args.d_model}, 3 blocks, d_state=16, Multiscale, CRPS, 50 epochs')
    print(f'Batch={args.batch}, 10-zone joint training')
    print('='*60)

    tl, testl, nv, sy = load_all(args.seq, args.pred, args.batch, args.stride)

    model = LNMambaV3(nv, d=args.d_model, nb=3, ds=16, pred=args.pred, nq=99).to(device)
    n_p = sum(p.numel() for p in model.parameters())
    print(f'Params: {n_p:,} | {len(tl)} batches/epoch')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=15, T_mult=2, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    best_pb = float('inf'); best_state = None; hist = []

    for ep in range(1, args.epochs + 1):
        t0 = time.time(); model.train()
        tl_pb = 0.0
        for x, y in tl:
            x, y = x.to(device), y.to(device); opt.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(x)
                pb = pinball_loss(out, y, q_t)
                cp = crossing_penalty(out)
                cr = crps_approx(out, y, q_t)
                loss = pb + 0.05 * cp + 0.1 * cr
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tl_pb += pb.item()
        sch.step()
        avg_pb = tl_pb / len(tl); hist.append(avg_pb)
        et = time.time() - t0

        star = ' ★' if avg_pb < best_pb else ''
        if avg_pb < best_pb:
            best_pb = avg_pb; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        print(f'E {ep:2d} | pb={avg_pb:.4f} | {et:.0f}s{star}')
        if ep >= 20 and ep - np.argmin(hist) >= 15: break

    if best_state: model.load_state_dict(best_state)

    # Test
    print(f'\n{"="*60}\nZone 1 Test')
    model.eval()
    preds, targs = [], []
    total_pb = 0.0
    with torch.no_grad():
        for x, y in testl:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_pb += pinball_loss(out, y, q_t).item()
            preds.append(out.cpu().numpy()); targs.append(y.cpu().numpy())
    pr = np.concatenate(preds); tr = np.concatenate(targs)
    test_pb = total_pb / len(testl)

    sh = pr.shape
    pr_mw = sy.inverse_transform(pr.reshape(-1, sh[2])).reshape(sh)
    tr_mw = sy.inverse_transform(tr.reshape(-1, 1)).reshape(tr.shape)

    p50 = pr_mw[:, :, 49]
    pf= p50.ravel(); tf = tr_mw.ravel(); mask = tf > 0.001
    rmse = np.sqrt(np.mean((pf[mask]-tf[mask])**2))
    mae  = np.mean(np.abs(pf[mask]-tf[mask]))
    r2   = 1 - np.sum((tf[mask]-pf[mask])**2)/(np.sum((tf[mask]-np.mean(tf[mask]))**2)+1e-8)

    print(f'Pinball: {test_pb:.4f} | R2: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}')

    print('Per-horizon Pinball:')
    for h in [0,3,5,11,17,23]:
        tr_h = torch.FloatTensor(tr_mw[:,h])
        pr_h = torch.FloatTensor(pr_mw[:,h])
        er = tr_h.unsqueeze(-1) - pr_h
        qt = torch.FloatTensor(QUANTILES)
        pb_h = torch.maximum(qt*er, (qt-1)*er).mean().item()
        print(f'  +{h+1:2d}h: {pb_h:.4f}')

    p10=pr_mw[:,:,9]; p90=pr_mw[:,:,89]
    in80 = np.mean((tr_mw>=p10)&(tr_mw<=p90))
    print(f'80% CI coverage: {in80*100:.1f}%')

    # Compare vs v1
    v1_pb = 0.2069
    print(f'\nv1 pb: {v1_pb:.4f} | v3 pb: {test_pb:.4f} | Δ: {(v1_pb-test_pb)/v1_pb*100:+.1f}%')
    print(f'Params: {n_p:,}')
    print('Done!')

if __name__ == '__main__':
    main()
