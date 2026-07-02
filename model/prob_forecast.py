"""
GEFCom2014 Probabilistic Wind Power Forecasting.

GEFCom2014 evaluates 99 quantiles (0.01 ~ 0.99) via pinball loss.
Single model outputs all quantiles, trained with quantile loss.

Models: QuantileGRU → QuantileMamba → QuantileLNNGatedMamba
"""
import sys,os,zipfile,time,argparse,io,math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DATA_DIR = 'data/gefcom2014'
QUANTILES = np.linspace(0.01, 0.99, 99)  # 99 quantiles for GEFCom2014


# ═══════════════════ Data ═══════════════════
def load_zone(zid, seq=168, pred=24, batch=64, stride=4):
    tz = zipfile.ZipFile(f'{DATA_DIR}/Task15_W_Zone1_10.zip')
    df = pd.read_csv(tz.open(f'Task15_W_Zone1_10/Task15_W_Zone{zid}.csv'))

    # Time
    ts = df['TIMESTAMP'].astype(str).str.strip()
    df['dt'] = pd.to_datetime(ts.str[:8], format='%Y%m%d') \
               + pd.to_timedelta(ts.str.extract(r'(\d+):')[0].astype(int), unit='h')
    df = df.sort_values('dt').reset_index(drop=True)

    # Features
    df['WS10']  = np.sqrt(df['U10']**2 + df['V10']**2)
    df['WS100'] = np.sqrt(df['U100']**2 + df['V100']**2)
    feat_cols = ['U10', 'V10', 'U100', 'V100', 'WS10', 'WS100']

    # Fix NaN
    df['TARGETVAR'] = df['TARGETVAR'].interpolate(method='linear', limit_direction='both')

    # Scale
    scaler_x = StandardScaler()
    feats = scaler_x.fit_transform(df[feat_cols].values.astype(np.float32))
    scaler_y = StandardScaler()
    target = scaler_y.fit_transform(df[['TARGETVAR']].values.astype(np.float32)).ravel()

    data = np.concatenate([feats, target.reshape(-1,1)], axis=1)
    T = len(data)
    te = int(T * 0.85)
    ts_start = int(T * 0.95)

    ds_train = ProbDS(data[:te], seq, pred, stride)
    ds_val   = ProbDS(data[te:ts_start], seq, pred, stride)
    ds_test  = ProbDS(data[ts_start:], seq, pred, stride)

    print(f'Zone {zid}: {len(ds_train):,} train | {len(ds_val):,} val | {len(ds_test):,} test')
    return (
        DataLoader(ds_train, batch, shuffle=True,  num_workers=0, pin_memory=True),
        DataLoader(ds_val,   batch, shuffle=False, num_workers=0, pin_memory=True),
        DataLoader(ds_test,  batch, shuffle=False, num_workers=0, pin_memory=True),
    ), data.shape[1], scaler_y


class ProbDS(Dataset):
    def __init__(self, data, seq, pred, stride):
        self.data = torch.FloatTensor(data)
        self.seq = seq; self.pred = pred; self.s = stride
        self.n = max(0, (len(data)-seq-pred)//stride+1)
    def __len__(self): return self.n
    def __getitem__(self, i):
        st = i*self.s
        return (self.data[st:st+self.seq].T,
                self.data[st+self.seq:st+self.seq+self.pred, -1])


# ═══════════════════ Components ═══════════════════
class MambaSSM(nn.Module):
    """Fast Liquid-Gated Selective SSM block."""
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


class QuantileDecoder(nn.Module):
    """Decode d_model → (pred_len, 99 quantiles)."""
    def __init__(self, d, pred, nq=99):
        super().__init__()
        self.pred = pred; self.nq = nq
        self.net = nn.Sequential(
            nn.Linear(d, d*2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d*2, d), nn.GELU(),
            nn.Linear(d, pred * nq),
        )

    def forward(self, x):
        """x: (B, D) → (B, pred_len, n_quantiles)"""
        out = self.net(x)  # (B, pred*nq)
        return out.view(-1, self.pred, self.nq)


class LNNMambaProb(nn.Module):
    """LNN-Gated Selective SSM for probabilistic forecasting."""
    def __init__(self, V, d=64, nb=2, ds=16, pred=24, nq=99, use_lnn=True):
        super().__init__()
        self.use_lnn = use_lnn
        self.emb = nn.Sequential(nn.Linear(V, d*2), nn.GELU(), nn.Linear(d*2, d))
        self.pe  = nn.Parameter(torch.randn(1,2000,d)*0.02)
        self.mb  = nn.ModuleList([MambaSSM(d,ds) for _ in range(nb)])
        self.lnn = nn.ModuleList([nn.Sequential(
            nn.GRU(d, 32, batch_first=True), nn.Linear(32, d)
        ) for _ in range(nb)])
        self.dec = QuantileDecoder(d, pred, nq)
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
        return self.dec(x[:, -1])


class QuantileGRU(nn.Module):
    def __init__(self, V, d=128, nl=2, pred=24, nq=99):
        super().__init__()
        self.proj = nn.Linear(V, d)
        self.gru  = nn.GRU(d, d, nl, batch_first=True, dropout=0.1)
        self.dec  = QuantileDecoder(d, pred, nq)

    def forward(self, x):
        _, h = self.gru(self.proj(x.transpose(1,2)))
        return self.dec(h[-1])


# ═══════════════════ Loss & Metrics ═══════════════════
def pinball_loss(pred_q, target, quantiles_t):
    """All tensors must be on same device."""
    error = target.unsqueeze(-1) - pred_q
    loss = torch.maximum(quantiles_t * error, (quantiles_t - 1) * error)
    return loss.mean()


def evaluate_pinball(model, loader, scaler_y, device):
    """Compute pinball loss on test set (GEFCom2014 official metric)."""
    model.eval()
    q_tensor = torch.tensor(QUANTILES, dtype=torch.float32, device=device)
    total_loss = 0.0; n_batches = 0
    preds, targs = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)  # (B, pred_len, 99)
            total_loss += pinball_loss(out, y, q_tensor).item()
            n_batches += 1
            preds.append(out.cpu().numpy())
            targs.append(y.cpu().numpy())
    pr = np.concatenate(preds)  # (N, pred_len, 99)
    tr = np.concatenate(targs)  # (N, pred_len)
    # Inverse transform
    if scaler_y is not None:
        sh = pr.shape
        pr = scaler_y.inverse_transform(pr.reshape(-1, sh[2])).reshape(sh)
        tr = scaler_y.inverse_transform(tr.reshape(-1, 1)).reshape(tr.shape)
    return total_loss / n_batches, pr, tr


def persistence_pinball(loader, scaler_y, pred_len, device):
    """Persistence baseline with zero-variance prediction."""
    q_tensor = torch.tensor(QUANTILES, dtype=torch.float32, device=device)
    total_loss = 0.0; n_batches = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            last = x[:, -1, -1]  # (B,)
            # Persistence: all quantiles = last observed value
            pred_q = last.unsqueeze(1).unsqueeze(2).expand(-1, pred_len, 99)
            total_loss += pinball_loss(pred_q, y, q_tensor).item()
            n_batches += 1
    return total_loss / n_batches


# ═══════════════════ Training ═══════════════════
def train_model(model, tl, vl, device, epochs=30, lr=1e-3, label=''):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    q_tensor = torch.tensor(QUANTILES, dtype=torch.float32, device=device)
    best_loss = float('inf'); best_state = None; hist = []

    for ep in range(1, epochs+1):
        t0 = time.time(); model.train(); tl_loss = 0.0
        for x, y in tl:
            x, y = x.to(device), y.to(device); opt.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(x)
                loss = pinball_loss(out, y, q_tensor)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); tl_loss += loss.item()
        sch.step()

        # Val pinball
        model.eval(); vpreds, vtargs = [], []
        with torch.no_grad():
            for x, y in vl:
                out = model(x.to(device))
                vpreds.append(out.cpu()); vtargs.append(y)
        vp = torch.cat(vpreds, dim=0); vt = torch.cat(vtargs, dim=0)
        v_loss = pinball_loss(vp.to(device), vt.to(device), q_tensor).item()
        hist.append(v_loss); t = time.time() - t0

        if v_loss < best_loss:
            best_loss = v_loss; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        print(f'E {ep:2d} | loss={tl_loss/len(tl):.4f} | V-pinball={v_loss:.4f} | {t:.0f}s{" *" if v_loss<best_loss else ""}')
        if ep >= 10 and ep - np.argmin(hist) >= 8:
            print(f'  Early stop'); break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_loss, hist


# ═══════════════════ Visualization ═══════════════════
def plot_probabilistic(pr, tr, zone, seq, pred, save_dir='plots'):
    """Plot prediction intervals for a sample."""
    os.makedirs(save_dir, exist_ok=True)
    nq = pr.shape[2]

    # Pick a representative sample (middle of test set)
    idx = len(pr) // 3
    sample_pred = pr[idx]     # (pred_len, 99)
    sample_true = tr[idx]     # (pred_len,)

    # Extract quantiles
    p10 = sample_pred[:, 9]    # 10th percentile
    p25 = sample_pred[:, 24]   # 25th
    p50 = sample_pred[:, 49]   # median
    p75 = sample_pred[:, 74]   # 75th
    p90 = sample_pred[:, 89]   # 90th
    p99 = sample_pred[:, 98]   # 99th

    hours = np.arange(1, pred+1)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(hours, p10, p90, alpha=0.15, color='steelblue', label='10-90% interval')
    ax.fill_between(hours, p25, p75, alpha=0.25, color='steelblue', label='25-75% interval')
    ax.fill_between(hours, p50, p50, alpha=0.3, color='navy', label='50% (median)')
    ax.plot(hours, p99, 'b-', lw=0.5, alpha=0.5, label='1-99% bounds')
    ax.plot(hours, p10, 'b-', lw=0.5, alpha=0.5)
    ax.plot(hours, sample_true, 'r-o', lw=2, ms=6, label='Actual', zorder=10)

    ax.set_xlabel('Horizon (hours)'); ax.set_ylabel('Normalized Power')
    ax.set_title(f'GEFCom2014 Zone {zone} Probabilistic Forecast\n'
                 f'{seq}h input → {pred}h prediction | 99 Quantiles')
    ax.legend(loc='upper right', ncol=2, fontsize=9)
    ax.grid(alpha=0.2)
    ax.set_ylim(0, None)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/prob_z{zone}.png', dpi=150)
    plt.close()

    # Horizon-wise pinball
    pinball_h = []
    q_t = torch.tensor(QUANTILES)
    for h in range(pred):
        er = torch.FloatTensor(tr[:, h]).unsqueeze(-1) - torch.FloatTensor(pr[:, h])
        pb = torch.maximum(q_t * er, (q_t - 1) * er).mean().item()
        pinball_h.append(pb)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hours, pinball_h, 'r-o', lw=2, ms=6)
    ax.set_xlabel('Horizon (hours)'); ax.set_ylabel('Pinball Loss')
    ax.set_title(f'Pinball Loss vs Horizon — Zone {zone}')
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/pinball_horizon_z{zone}.png', dpi=150)
    plt.close()

    return pinball_h


# ═══════════════════ Main ═══════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zones', type=str, default='1,2,3')
    parser.add_argument('--seq', type=int, default=168)
    parser.add_argument('--pred', type=int, default=24)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device} | Quantiles: {len(QUANTILES)}')

    zones = [int(z.strip()) for z in args.zones.split(',')]
    all_results = {}

    for zone in zones:
        print(f'\n{"="*70}')
        print(f'ZONE {zone}')
        print(f'{"="*70}')

        (tl, vl, testl), n_vars, scaler_y = load_zone(zone, args.seq, args.pred, args.batch, args.stride)

        # Persistence
        p_pinball = persistence_pinball(testl, scaler_y, args.pred, device)
        print(f'Persistence pinball: {p_pinball:.4f}')

        # Train 3 models
        configs = [
            ('GRU', QuantileGRU(n_vars, d=128, nl=2, pred=args.pred).to(device)),
            ('Mamba', LNNMambaProb(n_vars, d=args.d_model, nb=2, ds=16, pred=args.pred, use_lnn=False).to(device)),
            ('LNN-Gated Selective SSM', LNNMambaProb(n_vars, d=args.d_model, nb=2, ds=16, pred=args.pred, use_lnn=True).to(device)),
        ]

        zone_results = {}
        for name, model in configs:
            n_p = sum(p.numel() for p in model.parameters())
            print(f'\n--- {name} ({n_p:,} params) ---')
            best, hist = train_model(model, tl, vl, device, args.epochs, args.lr, name)

            # Test
            test_pb, test_pr, test_tr = evaluate_pinball(model, testl, scaler_y, device)
            imp = (p_pinball - test_pb) / p_pinball * 100
            zone_results[name] = {'pinball': test_pb, 'imp': imp}
            print(f'  Test pinball: {test_pb:.4f} (vs persist: {imp:+.1f}%)')

            # Plot for best model
            if name == 'LNN-Gated Selective SSM':
                plot_probabilistic(test_pr, test_tr, zone, args.seq, args.pred)

        all_results[zone] = {'persistence': p_pinball, **zone_results}

    # ── Summary ──
    print(f'\n{"="*80}')
    print(f'GEFCom2014 Probabilistic Wind Power Forecasting')
    print(f'{args.seq}h → {args.pred}h, {len(QUANTILES)} quantiles, Pinball Loss (lower=better)')
    print(f'{"="*80}')

    for zone in zones:
        r = all_results[zone]
        print(f'\nZone {zone}:')
        print(f'  {"Persistence":<15s}: {r["persistence"]:.4f}')
        for name in ['GRU', 'Mamba', 'LNN-Gated Selective SSM']:
            if name in r:
                print(f'  {name:<15s}: {r[name]["pinball"]:.4f} ({r[name]["imp"]:+.1f}%)')

    # Average
    avg_p = np.mean([all_results[z]['persistence'] for z in zones])
    print(f'\n{"Average":-^80}')
    print(f'{"Persistence":<15s}: {avg_p:.4f}')
    for name in ['GRU', 'Mamba', 'LNN-Gated Selective SSM']:
        avg_v = np.mean([all_results[z][name]['pinball'] for z in zones if name in all_results[z]])
        avg_i = (avg_p - avg_v) / avg_p * 100
        print(f'{name:<15s}: {avg_v:.4f} ({avg_i:+.1f}%)')

    print('\nDone!')


if __name__ == '__main__':
    main()
