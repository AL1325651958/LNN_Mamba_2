"""
GEFCom2014 with NWP — All 10 zones, proper weather integration.

Weather features (ECMWF NWP):
  U10, V10: wind vector at 10m (m/s)
  U100, V100: wind vector at 100m (m/s)
  WS10, WS100: derived wind speed = sqrt(U^2 + V^2)
  WD10, WD100: derived wind direction = atan2(U, V)

Also: hour, month, day-of-week as cyclic time features.

Target: TARGETVAR (wind power, normalized 0~1)
Prediction: 99 quantiles (0.01 ~ 0.99), 24h horizon
"""
import sys,os,zipfile,time,argparse,io,math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = 'data/gefcom2014'
QUANTILES = np.linspace(0.01, 0.99, 99)

# ═══════════════════ Weather features ═══════════════════
def make_weather_features(df):
    """Derive rich meteorological features from raw U/V."""
    # Wind speed
    df['WS10']  = np.sqrt(df['U10']**2 + df['V10']**2)
    df['WS100'] = np.sqrt(df['U100']**2 + df['V100']**2)
    # Wind direction (radians → cyclic encoding)
    df['WD10']  = np.arctan2(df['U10'], df['V10'])
    df['WD100'] = np.arctan2(df['U100'], df['V100'])
    # Wind shear (WS100/WS10 ratio — indicates atmospheric stability)
    df['SHEAR'] = df['WS100'] / (df['WS10'] + 0.1)
    # Wind veer (direction change with height)
    df['VEER']  = np.sin(df['WD100'] - df['WD10'])
    return df

FEAT_COLS = ['U10', 'V10', 'U100', 'V100', 'WS10', 'WS100', 'WD10', 'WD100', 'SHEAR', 'VEER']

# ═══════════════════ Data loading ═══════════════════
def load_zone_nwp(zid, seq=168, pred=24, batch=64, stride=4):
    """Load one zone with proper weather features."""
    tz = zipfile.ZipFile(f'{DATA_DIR}/Task15_W_Zone1_10.zip')
    df = pd.read_csv(tz.open(f'Task15_W_Zone1_10/Task15_W_Zone{zid}.csv'))

    # Parse time
    ts = df['TIMESTAMP'].astype(str).str.strip()
    df['dt'] = pd.to_datetime(ts.str[:8], format='%Y%m%d') \
               + pd.to_timedelta(ts.str.extract(r'(\d+):')[0].astype(int), unit='h')
    df = df.sort_values('dt').reset_index(drop=True)

    # Fix NaN
    df['TARGETVAR'] = df['TARGETVAR'].interpolate(limit_direction='both')
    for c in ['U10','V10','U100','V100']:
        df[c] = df[c].interpolate(limit_direction='both')

    # Weather features
    df = make_weather_features(df)

    # Cyclic time features (add as sine/cosine)
    h = df['dt'].dt.hour.values.astype(np.float32)
    m = df['dt'].dt.month.values.astype(np.float32)
    df['HOUR_SIN'] = np.sin(2*np.pi*h/24)
    df['HOUR_COS'] = np.cos(2*np.pi*h/24)
    df['MONTH_SIN'] = np.sin(2*np.pi*m/12)
    df['MONTH_COS'] = np.cos(2*np.pi*m/12)

    all_feats = FEAT_COLS + ['HOUR_SIN', 'HOUR_COS', 'MONTH_SIN', 'MONTH_COS']

    # Scale
    scaler_x = StandardScaler()
    feats = scaler_x.fit_transform(df[all_feats].values.astype(np.float32))
    scaler_y = StandardScaler()
    target = scaler_y.fit_transform(df[['TARGETVAR']].values.astype(np.float32)).ravel()

    data = np.concatenate([feats, target.reshape(-1,1)], axis=1)
    n_vars = data.shape[1]

    # Time split: train 85%, val 10%, test 5%
    T = len(data)
    te = int(T * 0.85)
    ve = int(T * 0.92)

    info = f'Zone {zid}: {len(df)} rows | train={te} val={ve-te} test={T-ve} | weather vars={len(FEAT_COLS)}'

    ds_train = NWPDS(data[:te], seq, pred, stride)
    ds_val   = NWPDS(data[te:ve], seq, pred, stride)
    ds_test  = NWPDS(data[ve:], seq, pred, stride)

    dl_train = DataLoader(ds_train, batch, shuffle=True,  num_workers=0, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch, shuffle=False, num_workers=0, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch, shuffle=False, num_workers=0, pin_memory=True)

    return (dl_train, dl_val, dl_test), n_vars, scaler_y, info


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

# ═══════════════════ Model (Quantile LNN-Mamba) ═══════════════════
class MambaSSM(nn.Module):
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
        Abar = torch.exp(de * (-torch.exp(self.A_log)).unsqueeze(0).unsqueeze(1))
        b = de * Bs.unsqueeze(2) * u.unsqueeze(-1)
        eps=1e-8; logA = torch.log(Abar.clamp(min=eps))
        Acum = torch.exp(torch.cumsum(logA, dim=1))
        h = Acum * torch.cumsum(b/Acum.clamp(min=eps), dim=1)
        y = (h*Cs.unsqueeze(2)).sum(-1)+self.D.unsqueeze(0).unsqueeze(0)*u
        return self.nm(self.out(y*F.silu(z))+res)

class QuantileDecoder(nn.Module):
    def __init__(self, d, pred, nq=99):
        super().__init__()
        self.pred=pred; self.nq=nq
        self.net = nn.Sequential(
            nn.Linear(d, d*2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d*2, d), nn.GELU(),
            nn.Linear(d, pred*nq))

    def forward(self, x):
        return self.net(x).view(-1, self.pred, self.nq)

class NWPMamba(nn.Module):
    """LNN-Mamba with RevIN + NWP weather features for probabilistic WPF."""
    def __init__(self, V, d=64, nb=2, ds=16, pred=24, nq=99, use_lnn=True):
        super().__init__()
        self.use_lnn = use_lnn
        self.pred_len = pred
        self.nq = nq
        self.emb = nn.Sequential(nn.Linear(V, d*2), nn.GELU(), nn.Linear(d*2, d))
        self.pe  = nn.Parameter(torch.randn(1,2000,d)*0.02)
        self.mb  = nn.ModuleList([MambaSSM(d,ds) for _ in range(nb)])
        self.gates = nn.ModuleList([nn.Sequential(
            nn.GRU(d,48,batch_first=True), nn.Linear(48,d)) for _ in range(nb)])
        self.dec = QuantileDecoder(d, pred, nq)
        self.drop = nn.Dropout(0.1)
        # RevIN: per-variable normalization across time, stored in forward
        self.register_buffer('rev_eps', torch.tensor(1e-5))
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 0.5)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        B, V, L = x.shape

        # Partial RevIN: only normalize target variable (power), keep NWP features as-is
        # NWP features have physical meaning (m/s wind speed) — normalizing per-sample
        # destroys their absolute scale relationship to power output
        power_mu  = x[:, -1:, :].mean(dim=-1, keepdim=True)  # (B, 1, 1)
        power_sig = torch.sqrt(x[:, -1:, :].var(dim=-1, keepdim=True, unbiased=False) + self.rev_eps)
        x_norm = x.clone()
        x_norm[:, -1:, :] = (x[:, -1:, :] - power_mu) / power_sig
        # Other variables (NWP features) unchanged — they're already standardized via StandardScaler

        # Embed & process
        x_norm = self.emb(x_norm.transpose(1,2)) + self.pe[:,:L]
        for mb, gate in zip(self.mb, self.gates):
            x_norm = self.drop(mb(x_norm))
            if self.use_lnn:
                h,_ = gate[0](x_norm)
                x_norm = x_norm * torch.sigmoid(gate[1](h))

        # Decode quantiles in normalized space
        out = self.dec(x_norm[:,-1])  # (B, pred_len, nq)

        # RevIN denorm: restore original power scale
        out = out * power_sig.squeeze(-1).unsqueeze(1) + power_mu.squeeze(-1).unsqueeze(1)

        return out

# ═══════════════════ Training ═══════════════════
def pinball_loss(pred_q, target, q_tensor):
    error = target.unsqueeze(-1) - pred_q
    return torch.maximum(q_tensor*error, (q_tensor-1)*error).mean()

def train_one_zone(model, tl, vl, device, epochs=25, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    q_t = torch.tensor(QUANTILES, dtype=torch.float32, device=device)
    best_loss = float('inf'); best_state = None

    for ep in range(1, epochs+1):
        model.train(); tloss = 0.0
        for x, y in tl:
            x,y = x.to(device), y.to(device); opt.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = pinball_loss(model(x), y, q_t)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt); scaler.update(); tloss += loss.item()
        sch.step()

        model.eval(); vpreds, vtargs = [], []
        with torch.no_grad():
            for x,y in vl:
                vpreds.append(model(x.to(device)).cpu()); vtargs.append(y)
        vp = torch.cat(vpreds); vt = torch.cat(vtargs)
        vpb = pinball_loss(vp.to(device), vt.to(device), q_t).item()

        if vpb < best_loss:
            best_loss = vpb; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        if ep % 5 == 1 or ep == epochs:
            print(f'  E{ep:2d} loss={tloss/len(tl):.4f} val_pb={vpb:.4f}{"*" if vpb==best_loss else ""}')

    if best_state: model.load_state_dict(best_state)
    return best_loss

def evaluate(model, loader, scaler_y, device):
    model.eval(); preds, targs = [], []
    q_t = torch.tensor(QUANTILES, dtype=torch.float32, device=device)
    total_pb = 0.0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            total_pb += pinball_loss(out, y, q_t).item()
            preds.append(out.cpu().numpy()); targs.append(y.cpu().numpy())
    pr = np.concatenate(preds); tr = np.concatenate(targs)
    if scaler_y is not None:
        sh = pr.shape
        pr = scaler_y.inverse_transform(pr.reshape(-1,sh[2])).reshape(sh)
        tr = scaler_y.inverse_transform(tr.reshape(-1,1)).reshape(tr.shape)
    return total_pb/len(loader), pr, tr

# ═══════════════════ Main ═══════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zones', type=str, default='1,2,3,4,5,6,7,8,9,10')
    parser.add_argument('--seq', type=int, default=168)
    parser.add_argument('--pred', type=int, default=24)
    parser.add_argument('--stride', type=int, default=6)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    zones = [int(z.strip()) for z in args.zones.split(',')]
    print(f'NWP Weather-Enhanced Forecasting | {len(zones)} zones | {len(FEAT_COLS)} weather vars')
    print(f'Weather: {FEAT_COLS}')
    print(f'{"="*70}')

    all_pb = {}
    for zid in zones:
        (tl, vl, testl), n_vars, scaler_y, info = load_zone_nwp(
            zid, args.seq, args.pred, args.batch, args.stride)
        print(f'\n{info}')

        # Mamba + LNN
        model = NWPMamba(n_vars, d=args.d_model, nb=2, ds=16, pred=args.pred,
                         nq=len(QUANTILES), use_lnn=True).to(device)
        n_p = sum(p.numel() for p in model.parameters())
        print(f'  LNN-Mamba params: {n_p:,}')

        best = train_one_zone(model, tl, vl, device, args.epochs, args.lr)
        test_pb, test_pr, test_tr = evaluate(model, testl, scaler_y, device)
        all_pb[zid] = {'pinball': test_pb, 'pred': test_pr, 'target': test_tr}
        print(f'  Test pinball: {test_pb:.4f}')

    # ── Summary ──
    print(f'\n{"="*70}')
    print(f'{"GEFCom2014 + ECMWF NWP Weather":^70}')
    print(f'{args.seq}h NWP → {args.pred}h power | {len(QUANTILES)} quantiles')
    print(f'{"="*70}')
    print(f'{"Zone":>6s} {"Pinball":>10s}')
    for zid in zones:
        print(f'{zid:>6d} {all_pb[zid]["pinball"]:>10.4f}')
    avg = np.mean([all_pb[z]['pinball'] for z in zones])
    print(f'{"Avg":>6s} {avg:>10.4f}')

    # Per-horizon pinball for zone 1
    z1 = all_pb[1]
    print(f'\nZone 1 per-horizon pinball:')
    for h in [0,3,5,11,17,23]:
        tr_h = torch.FloatTensor(z1['target'][:,h])
        pr_h = torch.FloatTensor(z1['pred'][:,h])
        q_t = torch.FloatTensor(QUANTILES)
        er = tr_h.unsqueeze(-1) - pr_h
        pb = torch.maximum(q_t*er, (q_t-1)*er).mean().item()
        print(f'  +{h+1:2d}h: {pb:.4f}')

    # Save
    os.makedirs('checkpoints', exist_ok=True)
    np.savez('checkpoints/nwp_results.npz', **{f'z{z}': all_pb[z]['pinball'] for z in zones})
    print('\nDone!')

if __name__ == '__main__':
    main()
