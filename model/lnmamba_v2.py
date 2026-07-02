"""
LNMamba v2 — Deeply optimized probabilistic wind power forecasting.

Optimizations over v1:
  1. Bidirectional Mamba (forward + backward SSM → fused)
  2. Deeper: 4 blocks × d_model=96 × d_state=32
  3. Multi-scale conv frontend (hourly + daily receptive fields)
  4. CRPS auxiliary loss + quantile crossing penalty
  5. Cosine warmup + restart schedule
  6. Cross-zone joint training (mix all zones by time)
  7. Larger batch + stride=4 for more effective samples
"""
import sys,os,zipfile,time,argparse,io,math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = 'data/gefcom2014'
QUANTILES = np.linspace(0.01, 0.99, 99)

# ═══════════════════ Weather Feature Engineering ═══════════════════
def make_weather_features(df):
    df['WS10']  = np.sqrt(df['U10']**2 + df['V10']**2)
    df['WS100'] = np.sqrt(df['U100']**2 + df['V100']**2)
    df['WD10_SIN'] = np.sin(np.arctan2(df['U10'], df['V10']))
    df['WD10_COS'] = np.cos(np.arctan2(df['U10'], df['V10']))
    df['WD100_SIN'] = np.sin(np.arctan2(df['U100'], df['V100']))
    df['WD100_COS'] = np.cos(np.arctan2(df['U100'], df['V100']))
    df['SHEAR'] = df['WS100'] / (df['WS10'] + 0.1)
    df['TURB']  = (df['U10'].diff()**2 + df['V10'].diff()**2).fillna(0).clip(0, 50)
    return df

FEAT_COLS = ['U10','V10','U100','V100','WS10','WS100',
             'WD10_SIN','WD10_COS','WD100_SIN','WD100_COS','SHEAR','TURB']

# ═══════════════════ Bidirectional Mamba Block ═══════════════════
class BiDirectionalSSM(nn.Module):
    """Forward + backward Selective SSM, output concatenated and projected."""
    def __init__(self, d, ds=32, dc=4, ex=2):
        super().__init__()
        di = d * ex; self.ds = ds
        # Forward branch
        self.fwd = _MambaCore(d, ds, dc, ex)
        # Backward branch (same architecture)
        self.bwd = _MambaCore(d, ds, dc, ex)
        # Fuse: 2*d → d
        self.fuse = nn.Linear(d * 2, d, bias=False)
        self.nm = nn.RMSNorm(d)

    def forward(self, x):
        # x: (B, L, D)
        f = self.fwd(x)
        b = self.fwd(x.flip(dims=[1])).flip(dims=[1])  # backward
        return self.nm(self.fuse(torch.cat([f, b], dim=-1)) + x)

class _MambaCore(nn.Module):
    def __init__(self, d, ds, dc, ex):
        super().__init__()
        di = d * ex; self.ds = ds
        self.inp  = nn.Linear(d, di * 2, bias=False)
        self.cnv  = nn.Conv1d(di, di, dc, groups=di, padding=dc - 1)
        self.xp   = nn.Linear(di, ds * 2 + 1, bias=False)
        self.dtp  = nn.Linear(ds, di, bias=True)
        A = torch.arange(1, ds + 1).float().unsqueeze(0) * 0.03  # smaller init
        self.A_log = nn.Parameter(torch.log(A))
        self.D     = nn.Parameter(torch.ones(di))
        self.out   = nn.Linear(di, d, bias=False)  # project D_inner → d_model

    def forward(self, x):
        B, L, _ = x.shape
        xz = self.inp(x); u, z = xz.chunk(2, dim=-1)
        u = F.silu(self.cnv(u.transpose(1, 2))[:, :, :L].transpose(1, 2))
        proj = self.xp(u)
        dt = F.softplus(self.dtp(F.softplus(proj[:, :, :self.ds]))) + 1e-4
        Bs, Cs = proj[:, :, self.ds:self.ds*2], proj[:, :, self.ds*2:]
        de = dt.unsqueeze(-1)
        A_neg = (-torch.exp(self.A_log)).unsqueeze(0).unsqueeze(1)  # (1, 1, d_state)
        Abar = torch.exp(de * A_neg)  # (B, L, D_in, d_state)

        # Sequential scan to save memory (L=168 is short enough)
        h = torch.zeros(B, dt.shape[-1], self.ds, device=x.device, dtype=x.dtype)
        out_seq = []
        for t in range(L):
            h = Abar[:, t] * h + de[:, t] * Bs[:, t].unsqueeze(1) * u[:, t].unsqueeze(-1)
            out_seq.append((h * Cs[:, t].unsqueeze(1)).sum(-1) + self.D.unsqueeze(0).unsqueeze(0) * u[:, t])
        y = torch.stack(out_seq, dim=1)
        return self.out(y * F.silu(z))  # (B, L, d_model)


# ═══════════════════ Multi-Scale Frontend ═══════════════════
class MultiScaleConvFrontend(nn.Module):
    """1h + 3h + 6h + 24h causal convs in parallel."""
    def __init__(self, d_model):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model // 4, k, padding=k - 1, groups=1)
            for k in [3, 7, 13, 25]  # ~1h, 3h, 6h, 24h at hourly data
        ])
        self.fuse = nn.Linear(d_model, d_model)
        self.nm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, L, D = x.shape
        xt = x.transpose(1, 2)  # (B, D, L)
        branches = [F.gelu(c(xt))[:, :, :L] for c in self.convs]
        multi = torch.cat(branches, dim=1)  # (B, D, L)
        multi = multi.transpose(1, 2)  # (B, L, D)
        return self.nm(x + self.fuse(multi))


# ═══════════════════ LNMamba v2 Model ═══════════════════
class LNMambaV2(nn.Module):
    """LNN-regulated Bidirectional Mamba for probabilistic WPF."""
    def __init__(self, V, d=96, nb=4, ds=32, pred=24, nq=99, use_lnn=True):
        super().__init__()
        self.use_lnn = use_lnn

        # Input embedding
        self.emb = nn.Sequential(
            nn.Linear(V, d * 2), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(d * 2, d)
        )
        # Learnable positional encoding
        self.pe = nn.Parameter(torch.randn(1, 2000, d) * 0.02)

        # Multi-scale frontend
        self.ms_conv = MultiScaleConvFrontend(d)

        # Bidirectional Mamba blocks
        self.blocks = nn.ModuleList([BiDirectionalSSM(d, ds) for _ in range(nb)])

        # LNN gates (one per block)
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.GRU(d, 48, batch_first=True),
                nn.Linear(48, d)
            ) for _ in range(nb)
        ])

        self.drop = nn.Dropout(0.08)

        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d * 2, d), nn.GELU(),
            nn.Linear(d, pred * nq)
        )
        self.pred_len = pred; self.nq = nq
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 0.5)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        B, V, L = x.shape
        x = self.emb(x.transpose(1, 2)) + self.pe[:, :L]
        x = self.ms_conv(x)

        for blk, gate in zip(self.blocks, self.gates):
            x = self.drop(blk(x))
            if self.use_lnn:
                h, _ = gate[0](x)
                x = x * torch.sigmoid(gate[1](h))

        out = self.dec(x[:, -1])
        return out.view(B, self.pred_len, self.nq)


# ═══════════════════ Loss Functions ═══════════════════
def pinball_loss(pred_q, target, q_t):
    error = target.unsqueeze(-1) - pred_q
    return torch.maximum(q_t * error, (q_t - 1) * error).mean()

def quantile_crossing_penalty(pred_q):
    """Penalize when adjacent quantiles cross."""
    diff = pred_q[:, :, 1:] - pred_q[:, :, :-1]  # (B, pred_len, nq-1)
    return F.relu(-diff).mean()  # penalty when diff < 0

def crps_loss(pred_q, target, q_t):
    """Continuous Ranked Probability Score approximation from quantiles."""
    # CRPS ≈ ∫ (F_pred(q) - 1(y≤q))² dq
    indicator = (target.unsqueeze(-1) <= pred_q).float()
    cdf = q_t.unsqueeze(0).unsqueeze(0)  # expected CDF values for quantiles
    # Simplified: MSE between indicator and quantile level
    return F.mse_loss(indicator, cdf.expand_as(indicator))

# ═══════════════════ Data Loading ═══════════════════
def load_all_zones(seq=168, pred=24, batch=64, stride=4):
    """Load all 10 zones, return concatenated DataLoader."""
    datasets = []
    n_vars = None

    for zid in range(1, 11):
        tz = zipfile.ZipFile(f'{DATA_DIR}/Task15_W_Zone1_10.zip')
        df = pd.read_csv(tz.open(f'Task15_W_Zone1_10/Task15_W_Zone{zid}.csv'))
        ts = df['TIMESTAMP'].astype(str).str.strip()
        df['dt'] = pd.to_datetime(ts.str[:8], format='%Y%m%d') \
                   + pd.to_timedelta(ts.str.extract(r'(\d+):')[0].astype(int), unit='h')
        df = df.sort_values('dt').reset_index(drop=True)
        df['TARGETVAR'] = df['TARGETVAR'].interpolate(limit_direction='both')
        for c in ['U10','V10','U100','V100']:
            df[c] = df[c].interpolate(limit_direction='both')
        df = make_weather_features(df)

        # Cyclic time
        h = df['dt'].dt.hour.values.astype(np.float32)
        m = df['dt'].dt.month.values.astype(np.float32)
        df['HOUR_SIN'] = np.sin(2*np.pi*h/24)
        df['HOUR_COS'] = np.cos(2*np.pi*h/24)
        df['MONTH_SIN'] = np.sin(2*np.pi*m/12)
        df['MONTH_COS'] = np.cos(2*np.pi*m/12)

        all_feats = FEAT_COLS + ['HOUR_SIN','HOUR_COS','MONTH_SIN','MONTH_COS']
        scaler_x = StandardScaler()
        feats = scaler_x.fit_transform(df[all_feats].values.astype(np.float32))
        scaler_y = StandardScaler()
        target = scaler_y.fit_transform(df[['TARGETVAR']].values.astype(np.float32)).ravel()

        data = np.concatenate([feats, target.reshape(-1, 1)], axis=1)
        if n_vars is None: n_vars = data.shape[1]

        # Time split
        T = len(data); te = int(T * 0.85)
        datasets.append(NWPDS(data[:te], seq, pred, stride))
        print(f'  Zone {zid}: {len(datasets[-1]):,} train samples')

    # Concatenate all zones for joint training
    ds_full = torch.utils.data.ConcatDataset(datasets)
    print(f'  Total: {len(ds_full):,} train samples')

    # Also load Zone 1 test for eval
    tz = zipfile.ZipFile(f'{DATA_DIR}/Task15_W_Zone1_10.zip')
    df = pd.read_csv(tz.open('Task15_W_Zone1_10/Task15_W_Zone1.csv'))
    ts = df['TIMESTAMP'].astype(str).str.strip()
    df['dt'] = pd.to_datetime(ts.str[:8], format='%Y%m%d') \
               + pd.to_timedelta(ts.str.extract(r'(\d+):')[0].astype(int), unit='h')
    df = df.sort_values('dt').reset_index(drop=True)
    df['TARGETVAR'] = df['TARGETVAR'].interpolate(limit_direction='both')
    for c in ['U10','V10','U100','V100']: df[c] = df[c].interpolate(limit_direction='both')
    df = make_weather_features(df)
    h = df['dt'].dt.hour.values.astype(np.float32)
    m = df['dt'].dt.month.values.astype(np.float32)
    df['HOUR_SIN'] = np.sin(2*np.pi*h/24); df['HOUR_COS'] = np.cos(2*np.pi*h/24)
    df['MONTH_SIN'] = np.sin(2*np.pi*m/12); df['MONTH_COS'] = np.cos(2*np.pi*m/12)

    # Use same scaler fit (approximate — refit per zone is ok for eval)
    scaler_x2 = StandardScaler()
    feats2 = scaler_x2.fit_transform(df[all_feats].values.astype(np.float32))
    scaler_y2 = StandardScaler()
    target2 = scaler_y2.fit_transform(df[['TARGETVAR']].values.astype(np.float32)).ravel()
    data2 = np.concatenate([feats2, target2.reshape(-1,1)], axis=1)
    T2 = len(data2); te2 = int(T2 * 0.85)
    ds_test = NWPDS(data2[te2:], seq, pred, stride)

    dl_train = DataLoader(ds_full, batch, shuffle=True, num_workers=0, pin_memory=True)
    dl_test  = DataLoader(ds_test, batch, shuffle=False, num_workers=0, pin_memory=True)

    return dl_train, dl_test, n_vars, scaler_y2


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

# ═══════════════════ Training ═══════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', type=int, default=168)
    parser.add_argument('--pred', type=int, default=24)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--batch', type=int, default=48)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--d_model', type=int, default=96)
    parser.add_argument('--lr', type=float, default=5e-4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'LNMamba v2 | Device: {device} | {len(FEAT_COLS)} weather vars + 4 time cyclic')
    print(f'Optimizations: Bi-directional Selective SSM ×4, d={args.d_model}, multiscale conv, CRPS + crossing penalty')
    print(f'{"="*60}')

    # Data
    tl, testl, n_vars, scaler_y = load_all_zones(args.seq, args.pred, args.batch, args.stride)
    print(f'{n_vars} input variables')

    # Model
    model = LNMambaV2(n_vars, d=args.d_model, nb=4, ds=32, pred=args.pred,
                      nq=len(QUANTILES), use_lnn=True).to(device)
    n_p = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {n_p:,}')

    # Optimizer + Scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    q_t = torch.tensor(QUANTILES, dtype=torch.float32, device=device)

    # Training
    history = []
    best_pb = float('inf')
    best_state = None

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tloss, tpb, tcrps, tcross = 0, 0, 0, 0

        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(x)
                pb = pinball_loss(out, y, q_t)
                crps = crps_loss(out, y, q_t)
                cross = quantile_crossing_penalty(out)
                loss = pb + 0.1 * crps + 0.05 * cross

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()

            tloss += loss.item(); tpb += pb.item()
            tcrps += crps.item(); tcross += cross.item()

        n_b = len(tl)
        avg_loss = tloss / n_b; avg_pb = tpb / n_b
        elapsed = time.time() - t0
        history.append(avg_pb)
        sch.step()  # step per epoch for CosineAnnealingWarmRestarts

        star = ' ★' if avg_pb < best_pb else ''
        if avg_pb < best_pb:
            best_pb = avg_pb
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f'E {ep:2d} | loss={avg_loss:.4f} pb={avg_pb:.4f} '
              f'crps={tcrps/n_b:.4f} cross={tcross/n_b:.4f} | {elapsed:.0f}s{star}')

        if ep >= 15 and ep - np.argmin(history) >= 10:
            print(f'  Early stop at epoch {ep}')
            break

    # Restore best
    if best_state: model.load_state_dict(best_state)

    # ── Test on Zone 1 ──
    print(f'\n{"="*60}')
    print(f'Zone 1 Test Evaluation')
    model.eval()
    preds, targs = [], []
    total_test_pb = 0.0
    with torch.no_grad():
        for x, y in testl:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_test_pb += pinball_loss(out, y, q_t).item()
            preds.append(out.cpu().numpy())
            targs.append(y.cpu().numpy())

    pr = np.concatenate(preds); tr = np.concatenate(targs)
    test_pb = total_test_pb / len(testl)

    # Inverse transform
    sh = pr.shape
    pr_mw = scaler_y.inverse_transform(pr.reshape(-1, sh[2])).reshape(sh)
    tr_mw = scaler_y.inverse_transform(tr.reshape(-1, 1)).reshape(tr.shape)

    # R² for median
    p50 = pr_mw[:, :, 49]
    pf = p50.ravel(); tf = tr_mw.ravel()
    mask = tf > 0.001
    rmse_v = np.sqrt(np.mean((pf[mask] - tf[mask])**2))
    mae_v = np.mean(np.abs(pf[mask] - tf[mask]))
    r2 = 1 - np.sum((tf[mask]-pf[mask])**2)/(np.sum((tf[mask]-np.mean(tf[mask]))**2)+1e-8)

    print(f'Test Pinball: {test_pb:.4f}')
    print(f'Median R2:   {r2:.4f}')
    print(f'Median RMSE: {rmse_v:.4f}')
    print(f'Median MAE:  {mae_v:.4f}')

    # Per-horizon
    print(f'\nPer-horizon Pinball:')
    for h in [0, 3, 5, 11, 17, 23]:
        tr_h = torch.FloatTensor(tr_mw[:, h])
        pr_h = torch.FloatTensor(pr_mw[:, h])
        er = tr_h.unsqueeze(-1) - pr_h
        pb_h = torch.maximum(torch.FloatTensor(QUANTILES)*er, (torch.FloatTensor(QUANTILES)-1)*er).mean().item()
        print(f'  +{h+1:2d}h: {pb_h:.4f}')

    # Confidence interval coverage
    p10 = pr_mw[:, :, 9]; p90 = pr_mw[:, :, 89]
    in_80ci = np.mean((tr_mw >= p10) & (tr_mw <= p90))
    print(f'\n80% CI coverage: {in_80ci*100:.1f}% (ideal: 80%)')

    print(f'\nParams: {n_p:,} | Best train PB: {best_pb:.4f} | Test PB: {test_pb:.4f}')
    print('Done!')


if __name__ == '__main__':
    main()
