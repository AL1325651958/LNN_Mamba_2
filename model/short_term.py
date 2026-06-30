"""
Short-term wind power forecasting — clean comparison.

Pred_len=24 (6h), seq_len=168 (42h), single site.
Models: Persistence → GRU → Mamba → LNN-Mamba
"""
import sys,os,glob,time,json,argparse,io
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ═══════════ Data ═══════════
class WindDS(Dataset):
    def __init__(self, data, seq, pred, stride):
        self.data = torch.FloatTensor(data)
        self.seq = seq; self.pred = pred; self.s = stride
        self.n = max(0, (len(data)-seq-pred)//stride+1)
    def __len__(self): return self.n
    def __getitem__(self, i):
        st = i*self.s
        return (self.data[st:st+self.seq].T,
                self.data[st+self.seq:st+self.seq+self.pred, -1])

def load_site2(seq, pred, stride, batch):
    site = 'Wind_farm_site_2_200MW'
    files = sorted(glob.glob(f'data/wind/{site}*.csv'))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    tc = df.columns[0]; df[tc] = pd.to_datetime(df[tc])
    df = df.sort_values(tc).reset_index(drop=True)
    df = df.interpolate(method='linear', limit_direction='both')
    pc = [c for c in df.columns if 'Power' in c][0]
    fc = [c for c in df.columns if 'Time' not in c.strip() and 'Power' not in c]
    print(f'Site 2: {len(df)} rows, {len(fc)} features')
    feats = StandardScaler().fit_transform(df[fc].values.astype(np.float32))
    scaler_power = StandardScaler()
    power = scaler_power.fit_transform(df[pc].values.astype(np.float32).reshape(-1,1))
    data = np.concatenate([feats, power], axis=1)
    T = len(data); te = int(T*0.7); ve = te + int(T*0.15)
    dss = WindDS(data[:te], seq, pred, stride), WindDS(data[te:ve], seq, pred, stride), WindDS(data[ve:], seq, pred, stride)
    print(f'Samples: {len(dss[0]):,} train, {len(dss[1]):,} val, {len(dss[2]):,} test')
    dls = tuple(DataLoader(d, batch, shuffle=i==0, num_workers=0, pin_memory=True) for i,d in enumerate(dss))
    return dls, data.shape[1], scaler_power

# ═══════════ Models ═══════════
class FastMamba(nn.Module):
    def __init__(self, d, ds=16, dc=4, ex=2):
        super().__init__()
        di = d*ex; self.ds = ds
        self.inp  = nn.Linear(d, di*2, bias=False)
        self.cnv  = nn.Conv1d(di, di, dc, groups=di, padding=dc-1)
        self.xp   = nn.Linear(di, ds*2+1, bias=False)
        self.dtp  = nn.Linear(ds, di, bias=True)
        A = torch.arange(1, ds+1, dtype=torch.float32).unsqueeze(0)*0.05
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
        self.pe = nn.Parameter(torch.randn(1,2000,d)*0.02)
        self.mb = nn.ModuleList([FastMamba(d,ds) for _ in range(nb)])
        self.lnn = nn.ModuleList([nn.Sequential(nn.GRU(d,32,batch_first=True), nn.Linear(32,d)) for _ in range(nb)])
        self.dec = nn.Sequential(nn.Linear(d, d*2), nn.GELU(), nn.Dropout(0.1), nn.Linear(d*2, pred))
        self.drop = nn.Dropout(0.1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B,V,L = x.shape
        x = self.emb(x.transpose(1,2)) + self.pe[:,:L]
        for mb, ln in zip(self.mb, self.lnn):
            x = self.drop(mb(x))
            if self.use_lnn:
                g = torch.sigmoid(self._lnn_gate(ln, x))
                x = x * g
        return self.dec(x[:,-1])

    def _lnn_gate(self, mod, x):
        h,_ = mod[0](x)
        return mod[1](h)

# ═══════════ Metrics ═══════════
def metrics(pred, target, power_vals, scaler_p):
    """pred,target: (N, pred_len) numpy. Returns dict + horizon array."""
    # Inverse transform to MW
    if scaler_p is not None:
        p_mw = scaler_p.inverse_transform(pred.reshape(-1,1)).reshape(pred.shape)
        t_mw = scaler_p.inverse_transform(target.reshape(-1,1)).reshape(target.shape)
    else:
        p_mw, t_mw = pred, target

    pf = p_mw.ravel(); tf = t_mw.ravel()
    mask = tf > 1.0
    pf, tf = pf[mask], tf[mask]
    if len(pf)==0: pf, tf = p_mw.ravel(), t_mw.ravel()

    mse = np.mean((pf-tf)**2); mae = np.mean(np.abs(pf-tf))
    mape = np.mean(np.abs((tf-pf)/(np.abs(tf)+1e-4)))*100
    r2 = 1 - np.sum((tf-pf)**2)/(np.sum((tf-np.mean(tf))**2)+1e-8)

    # Horizon errors
    rmse_h = [np.sqrt(np.mean((p_mw[:,h]-t_mw[:,h])**2)) for h in range(p_mw.shape[1])]
    mae_h  = [np.mean(np.abs(p_mw[:,h]-t_mw[:,h])) for h in range(p_mw.shape[1])]

    return {'rmse': np.sqrt(mse), 'mae': mae, 'mape': mape, 'r2': r2,
            'rmse_h': rmse_h, 'mae_h': mae_h}, p_mw, t_mw

# ═══════════ Persistence baseline ═══════════
def evaluate_persistence(loader, scaler_p, pred_len):
    """P(t+h) = P(t) — the simplest possible forecast."""
    preds, targs = [], []
    for x, y in loader:
        # x: (B,V,L), last observed power = x[:,-1,L-1]
        last_power = x[:, -1, -1]  # (B,) — last timestep power
        p = last_power.unsqueeze(1).repeat(1, pred_len).cpu().numpy()
        preds.append(p); targs.append(y.numpy())
    return metrics(np.concatenate(preds), np.concatenate(targs), None, scaler_p)

# ═══════════ Training ═══════════
def train_model(model, tl, vl, device, epochs=30, lr=1e-3, label='model'):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    best_rmse = float('inf'); best_state = None; hist = []

    for ep in range(1, epochs+1):
        t0 = time.time()
        model.train(); tl_loss = 0.0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = F.mse_loss(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tl_loss += loss.item()
        sch.step()

        # Validate
        model.eval(); preds, targs = [], []
        with torch.no_grad():
            for x, y in vl:
                preds.append(model(x.to(device)).cpu().numpy())
                targs.append(y.numpy())
        pr = np.concatenate(preds); tr = np.concatenate(targs)
        rmse = np.sqrt(np.mean((pr.ravel()-tr.ravel())**2))
        t = time.time() - t0; hist.append(rmse)

        if rmse < best_rmse:
            best_rmse = rmse; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}

        print(f'E {ep:2d} | loss={tl_loss/len(tl):.4f} | V-RMSE={rmse:.4f} | {t:.0f}s{" *" if rmse<best_rmse else ""}')

        if ep >= 10 and ep - np.argmin(hist) >= 8:
            print(f'  Early stop')
            break

    model.load_state_dict(best_state)
    return best_rmse, hist

# ═══════════ Main ═══════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', type=int, default=168)
    parser.add_argument('--pred', type=int, default=24)
    parser.add_argument('--stride', type=int, default=6)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    SEQ, PRED = args.seq, args.pred

    # Data
    (tl, vl, testl), n_vars, scaler_p = load_site2(SEQ, PRED, args.stride, args.batch)

    # Persistence — just uses last observed power, no training
    print('\n=== Persistence Baseline ===')
    pm, _, _ = evaluate_persistence(testl, scaler_p, PRED)
    print(f'Persistence: RMSE={pm["rmse"]:.2f} MW, MAE={pm["mae"]:.2f}, MAPE={pm["mape"]:.1f}%, R2={pm["r2"]:.3f}')

    # GRU
    print('\n=== GRU ===')
    gru = GRUModel(n_vars, d=128, nl=2, pred=PRED).to(device)
    n_gru = sum(p.numel() for p in gru.parameters())
    print(f'Params: {n_gru:,}')
    gru_rmse, gru_hist = train_model(gru, tl, vl, device, args.epochs, args.lr, 'GRU')
    gru.eval(); gpr,gtr = [],[]
    with torch.no_grad():
        for x,y in testl: gpr.append(gru(x.to(device)).cpu().numpy()); gtr.append(y.numpy())
    g_m, _, _ = metrics(np.concatenate(gpr), np.concatenate(gtr), None, scaler_p)

    # Mamba (no LNN)
    print('\n=== Mamba ===')
    mb = LNNMambaModel(n_vars, d=args.d_model, nb=2, ds=16, pred=PRED, use_lnn=False).to(device)
    n_mb = sum(p.numel() for p in mb.parameters())
    print(f'Params: {n_mb:,}')
    mb_rmse, mb_hist = train_model(mb, tl, vl, device, args.epochs, args.lr, 'Mamba')
    mb.eval(); mpr,mtr = [],[]
    with torch.no_grad():
        for x,y in testl: mpr.append(mb(x.to(device)).cpu().numpy()); mtr.append(y.numpy())
    m_m, _, _ = metrics(np.concatenate(mpr), np.concatenate(mtr), None, scaler_p)

    # LNN-Mamba
    print('\n=== LNN-Mamba ===')
    lm = LNNMambaModel(n_vars, d=args.d_model, nb=2, ds=16, pred=PRED).to(device)
    n_lm = sum(p.numel() for p in lm.parameters())
    print(f'Params: {n_lm:,}')
    lm_rmse, lm_hist = train_model(lm, tl, vl, device, args.epochs, args.lr, 'LNN-Mamba')
    lm.eval(); lpr,ltr = [],[]
    with torch.no_grad():
        for x,y in testl: lpr.append(lm(x.to(device)).cpu().numpy()); ltr.append(y.numpy())
    l_m, _, _ = metrics(np.concatenate(lpr), np.concatenate(ltr), None, scaler_p)

    # ── Results table ──
    print(f'\n{"="*70}')
    print(f'SHORT-TERM WIND POWER FORECASTING RESULTS')
    print(f'Site 2 (200MW), seq={SEQ} (42h) → pred={PRED} (6h), 15-min resolution')
    print(f'{"="*70}')
    print(f'{"Model":<20s} {"Params":>8s} {"RMSE(MW)":>10s} {"MAE(MW)":>10s} {"MAPE":>8s} {"R2":>8s}')
    print(f'{"-"*70}')
    for name, m, ps in [('Persistence', pm, 0), ('GRU', g_m, n_gru),
                         ('Mamba', m_m, n_mb), ('LNN-Mamba', l_m, n_lm)]:
        print(f'{name:<20s} {ps:>8,} {m["rmse"]:>10.2f} {m["mae"]:>10.2f} {m["mape"]:>7.1f}% {m["r2"]:>7.3f}')

    # Impvt vs persistence
    p_rmse = pm['rmse']
    print(f'\n{"Improvement vs Persistence":-^70}')
    for name, m in [('GRU', g_m), ('Mamba', m_m), ('LNN-Mamba', l_m)]:
        imp = (p_rmse - m['rmse'])/p_rmse*100
        print(f'{name:<20s} RMSE: {p_rmse:.1f}→{m["rmse"]:.1f} MW ({imp:+.1f}%)')

    # Per-horizon
    print(f'\n{"Per-horizon RMSE (MW)":-^70}')
    print(f'{"Horizon":<12s} {"Persistence":>12s} {"GRU":>12s} {"Mamba":>12s} {"LNN-Mamba":>12s}')
    for h in [0,3,7,11,15,19,23]:
        print(f'+{(h+1)*15:3d}min     {pm["rmse_h"][h]:>12.2f} {g_m["rmse_h"][h]:>12.2f} {m_m["rmse_h"][h]:>12.2f} {l_m["rmse_h"][h]:>12.2f}')

    # ── Plots ──
    os.makedirs('plots', exist_ok=True)

    # 1. Horizon error curves
    fig, ax = plt.subplots(figsize=(10,5))
    hh = [(h+1)*15 for h in range(PRED)]
    ax.plot(hh, pm['rmse_h'], 'k-', lw=2, label=f'Persistence (RMSE={pm["rmse"]:.1f} MW)')
    ax.plot(hh, g_m['rmse_h'], 'b-', lw=1.5, alpha=0.8, label=f'GRU (RMSE={g_m["rmse"]:.1f})')
    ax.plot(hh, m_m['rmse_h'], 'g-', lw=1.5, alpha=0.8, label=f'Mamba (RMSE={m_m["rmse"]:.1f})')
    ax.plot(hh, l_m['rmse_h'], 'r-', lw=2, label=f'LNN-Mamba (RMSE={l_m["rmse"]:.1f})')
    ax.set_xlabel('Horizon (minutes)'); ax.set_ylabel('RMSE (MW)')
    ax.set_title(f'Short-term Wind Power Forecasting | Site 2 (200MW) | {SEQ*15/60:.0f}h → {PRED*15/60:.0f}h')
    ax.legend(); ax.grid(alpha=0.2)
    plt.tight_layout(); plt.savefig('plots/short_term_horizon.png', dpi=150); plt.close()

    # 2. Timeseries sample for best model
    pr, tr = metrics(np.concatenate(lpr), np.concatenate(ltr), None, scaler_p)[1:]
    n = 192; start = len(pr)//4
    fig, ax = plt.subplots(figsize=(14,4))
    ax.plot(tr[start:start+n, 0], 'b-', lw=1, alpha=0.8, label='Actual Power')
    ax.plot(pr[start:start+n, 0], 'r--', lw=1, alpha=0.8, label='LNN-Mamba +15min')
    ax.plot(range(n), tr[start:start+n, 11], 'b-', lw=0.5, alpha=0.4)
    ax.plot(range(n), pr[start:start+n, 11], 'r--', lw=0.5, alpha=0.4)
    ax.fill_between(range(n), tr[start:start+n,0], pr[start:start+n,0], alpha=0.08, color='gray')
    ax.set_xlabel('Sample index'); ax.set_ylabel('Power (MW)')
    ax.set_title(f'LNN-Mamba: +15min & +3h Predictions')
    ax.legend(); ax.grid(alpha=0.2)
    plt.tight_layout(); plt.savefig('plots/short_term_timeseries.png', dpi=150); plt.close()

    # 3. Scatter
    p_all = np.concatenate(lpr).ravel(); t_all = np.concatenate(ltr).ravel()
    mask = t_all > 1; pf, tf = p_all[mask], t_all[mask]
    fig, ax = plt.subplots(figsize=(7,7))
    ax.hexbin(tf, pf, gridsize=50, cmap='Blues', mincnt=1, alpha=0.85)
    mx = max(tf.max(), pf.max()); ax.plot([0,mx],[0,mx],'k--',lw=1,alpha=0.5)
    rmse_v = np.sqrt(np.mean((pf-tf)**2)); r2_v = 1-np.sum((tf-pf)**2)/(np.sum((tf-np.mean(tf))**2)+1e-8)
    ax.text(0.05,0.95,f'LNN-Mamba\nRMSE={rmse_v:.1f}MW\nR²={r2_v:.3f}', transform=ax.transAxes, fontsize=11, va='top',
            bbox=dict(boxstyle='round',facecolor='white',alpha=0.85), fontfamily='monospace')
    ax.set_xlabel('Actual (MW)'); ax.set_ylabel('Predicted (MW)'); ax.set_aspect('equal')
    ax.set_title('LNN-Mamba: Predicted vs Actual')
    plt.tight_layout(); plt.savefig('plots/short_term_scatter.png', dpi=150); plt.close()

    print(f'\nPlots saved to plots/')
    print('Done!')

if __name__ == '__main__':
    main()
