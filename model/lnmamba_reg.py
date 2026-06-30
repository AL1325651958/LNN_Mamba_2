"""
LNMamba-Reg: Heavily regularized model with 6 strong regularization techniques.

1. StochasticDepth — randomly drop entire Mamba blocks during training
2. DropPath — stochastic residual connection dropout
3. SWA — Stochastic Weight Averaging over final epochs
4. Mixup — interpolate pairs of training samples
5. WeightDrop — dropout on GRU recurrent weights
6. Gradient Noise — inject Gaussian noise into gradients

Architecture: Same as v1 (d=64, 2 blocks, ds=16, 396K params)
Objective: Beat v1's test pinball of 0.2069 on Zone 1
Strategy: ALL regularizers together → ~30-50 epoch training
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
# Data: Zone 1
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
    sx = StandardScaler(); feats = sx.fit_transform(df[af].values.astype(np.float32))
    sy = StandardScaler(); tgt = sy.fit_transform(df[['TARGETVAR']].values.astype(np.float32)).ravel()
    data = np.concatenate([feats, tgt.reshape(-1, 1)], axis=1)
    return data, data.shape[1], sy

class WDS(Dataset):
    def __init__(self, d, s):
        self.data = torch.FloatTensor(d); self.s = s
        self.n = max(0, (len(d) - SEQ - PRED) // s + 1)
    def __len__(self): return self.n
    def __getitem__(self, i):
        st = i * self.s
        return (self.data[st:st+SEQ].T, self.data[st+SEQ:st+SEQ+PRED, -1])

# ═══════════════════════════════════════
# Regularized Mamba Block
# ═══════════════════════════════════════
class DropPath(nn.Module):
    """Stochastic depth / DropPath for residual connections."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor

class RegMambaBlock(nn.Module):
    """Mamba SSM with DropPath on residual."""
    def __init__(self, d, ds=16, dc=4, drop_path=0.1):
        super().__init__()
        self.ds = ds; di = d * 2
        self.inp = nn.Linear(d, di*2, bias=False)
        self.cnv = nn.Conv1d(di, di, dc, groups=di, padding=dc-1)
        self.xp  = nn.Linear(di, ds*2+1, bias=False)
        self.dtp = nn.Linear(ds, di, bias=True)
        A = torch.arange(1, ds+1).float().unsqueeze(0) * 0.03
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(di))
        self.out = nn.Linear(di, d, bias=False)
        self.nm  = nn.RMSNorm(d)
        self.drop_path = DropPath(drop_path)

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
        return self.nm(self.out(y * F.silu(z)) + self.drop_path(res))

# ═══════════════════════════════════════
# WeightDrop GRU for LNN gate
# ═══════════════════════════════════════
class WeightDropGRU(nn.Module):
    """GRU with DropConnect on recurrent weight matrix."""
    def __init__(self, input_size, hidden_size, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dropout = dropout

    def forward(self, x):
        if self.training and self.dropout > 0:
            # Apply dropout to recurrent weight matrix
            raw_w = self.gru.weight_hh_l0.clone()
            self.gru.weight_hh_l0 = nn.Parameter(
                F.dropout(raw_w, p=self.dropout, training=True))
        return self.gru(x)

class RegLNNGate(nn.Module):
    """LNN gate with WeightDrop GRU."""
    def __init__(self, d, h=48, wd_drop=0.1):
        super().__init__()
        self.gru = WeightDropGRU(d, h, dropout=wd_drop)
        self.out = nn.Linear(h, d)

    def forward(self, x):
        h, _ = self.gru(x)
        return torch.sigmoid(self.out(h))

# ═══════════════════════════════════════
# StochasticDepth wrapper
# ═══════════════════════════════════════
class StochasticDepthBlock(nn.Module):
    """Randomly skip entire Mamba block with probability p."""
    def __init__(self, block, gate, survival_prob=0.9):
        super().__init__()
        self.block = block
        self.gate = gate
        self.survival_prob = survival_prob

    def forward(self, x):
        if not self.training or self.survival_prob >= 1.0:
            x = self.block(x)
            return x * self.gate(x)

        # Stochastic depth: keep or skip this block
        if torch.rand(1, device=x.device) < self.survival_prob:
            x = self.block(x)
            x = x * self.gate(x)
            # Scale to compensate for dropped paths
            x = x / self.survival_prob
        # else: skip block entirely (x unchanged)
        return x

# ═══════════════════════════════════════
# Full Regularized Model
# ═══════════════════════════════════════
class LNMambaReg(nn.Module):
    def __init__(self, V, d=64, nb=2, ds=16, pred=24, nq=99,
                 dropout=0.1, drop_path=0.15, sd_survival=0.85):
        super().__init__()
        self.pred_len = pred; self.nq = nq
        self.use_sd = sd_survival < 1.0

        self.emb = nn.Sequential(
            nn.Linear(V, d*2), nn.GELU(), nn.Dropout(dropout*0.5),
            nn.Linear(d*2, d)
        )
        self.pe = nn.Parameter(torch.randn(1, 2000, d) * 0.02)

        # Build blocks with StochasticDepth
        self.blocks = nn.ModuleList()
        for i in range(nb):
            sd_prob = (i / nb) * (1.0 - sd_survival)  # later blocks dropped more
            surv = 1.0 - sd_prob
            block = RegMambaBlock(d, ds, drop_path=drop_path)
            gate = RegLNNGate(d, wd_drop=0.1)
            self.blocks.append(StochasticDepthBlock(block, gate, surv))

        self.dec = nn.Sequential(
            nn.Linear(d, d*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d*2, d), nn.GELU(),
            nn.Linear(d, pred * nq)
        )
        self.dropout = nn.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 0.5)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        B, V, L = x.shape
        x = self.emb(x.transpose(1, 2)) + self.pe[:, :L]
        for blk in self.blocks:
            x = blk(x)
        return self.dec(x[:, -1]).view(B, self.pred_len, self.nq)

# ═══════════════════════════════════════
# Time-Series Mixup
# ═══════════════════════════════════════
def mixup_batch(x, y, alpha=0.2):
    """Mixup: interpolate pairs of samples and their targets."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.shape[0]
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y, lam

# ═══════════════════════════════════════
# Gradient Noise
# ═══════════════════════════════════════
class GradientNoise:
    """Add Gaussian noise to gradients at each step."""
    def __init__(self, eta=0.01, gamma=0.55):
        self.eta = eta
        self.gamma = gamma

    def add_noise(self, model):
        """Add decaying noise to all parameter gradients."""
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * self.eta
                param.grad.add_(noise)
        self.eta *= self.gamma  # decay noise scale

# ═══════════════════════════════════════
# SWA (Stochastic Weight Averaging)
# ═══════════════════════════════════════
class SWA:
    """Averaged model over last N epochs."""
    def __init__(self, model, start_epoch=35):
        self.swa_model = copy.deepcopy(model)
        self.swa_n = 0
        self.start_epoch = start_epoch

    def update(self, model):
        self.swa_n += 1
        for swa_p, p in zip(self.swa_model.parameters(), model.parameters()):
            swa_p.data = (swa_p.data * (self.swa_n - 1) + p.data) / self.swa_n

    def apply(self, model):
        """Replace model params with SWA averaged params."""
        model.load_state_dict(self.swa_model.state_dict())

# ═══════════════════════════════════════
# Loss
# ═══════════════════════════════════════
def pb_loss(p, t, qt):
    e = t.unsqueeze(-1) - p
    return torch.maximum(qt*e, (qt-1)*e).mean()

# ═══════════════════════════════════════
# Main Training
# ═══════════════════════════════════════
def main():
    data, nv, sy = load_zone1()
    T = len(data); te = int(T * 0.85)

    # Train/val split — use val for SWA final selection
    train_ds = WDS(data[:te], 6)
    test_ds  = WDS(data[te:], 4)  # test on stride=4 for fair comparison
    val_ds   = WDS(data[te:], 6)  # val (same split, for SWA selection)

    tl = DataLoader(train_ds, 64, shuffle=True, num_workers=0, pin_memory=True)
    testl = DataLoader(test_ds, 64, shuffle=False, num_workers=0, pin_memory=True)

    print(f'LNMamba-Reg: {len(train_ds):,} train, {len(test_ds)} test')
    print(f'Regularizations: DropPath + StochasticDepth + WeightDrop + Mixup + GradNoise + SWA')
    sys.stdout.flush()

    model = LNMambaReg(nv, d=64, nb=2, ds=16, pred=PRED,
                       dropout=0.15, drop_path=0.15, sd_survival=0.85).to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters())
    print(f'Params: {n_p:,}'); sys.stdout.flush()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=15, T_mult=2, eta_min=1e-5)
    scl = torch.amp.GradScaler('cuda')
    qt  = torch.tensor(QUANTILES, dtype=torch.float32, device=DEVICE)
    grad_noise = GradientNoise(eta=0.01, gamma=0.55)
    swa = SWA(model, start_epoch=35)

    best_val = float('inf'); best_state = None
    EPOCHS = 50

    for ep in range(1, EPOCHS + 1):
        t0 = time.time(); model.train(); tl_pb = 0.0
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)

            # Mixup augmentation (50% probability)
            if torch.rand(1) < 0.5:
                x, y_mixed, _ = mixup_batch(x, y, alpha=0.3)
            else:
                y_mixed = y

            opt.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(x)
                loss = pb_loss(out, y_mixed, qt)

            scl.scale(loss).backward()
            scl.unscale_(opt)

            # Gradient noise injection (before clipping)
            grad_noise.add_noise(model)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scl.step(opt); scl.update()
            tl_pb += loss.item()

        sch.step()

        # Validate
        model.eval(); vp, vt = [], []
        with torch.no_grad():
            idxs = range(max(0, len(val_ds) - 256), len(val_ds))
            for i in idxs:
                x_, y_ = val_ds[i]
                vp.append(model(x_.unsqueeze(0).to(DEVICE)).cpu()); vt.append(y_)
        vp = torch.cat(vp); vt = torch.stack(vt, dim=0)
        val_pb = pb_loss(vp.to(DEVICE), vt.to(DEVICE), qt).item()
        et = time.time() - t0

        star = ' ★' if val_pb < best_val else ''
        if val_pb < best_val:
            best_val = val_pb
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # SWA update
        if ep >= swa.start_epoch:
            swa.update(model)
            swa_label = ' [SWA]'
        else:
            swa_label = ''

        print(f'E {ep:2d} pb={tl_pb/len(tl):.4f} val={val_pb:.4f} {et:.0f}s{star}{swa_label}')
        sys.stdout.flush()

    # Apply SWA
    if swa.swa_n > 0:
        print(f'\nApplying SWA (averaged over {swa.swa_n} epochs)')
        swa.apply(model)

    # Test final
    model.eval(); preds, targs = [], []; total_pb = 0.0
    with torch.no_grad():
        for x, y in testl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            total_pb += pb_loss(out, y, qt).item()
            preds.append(out.cpu().numpy()); targs.append(y.cpu().numpy())

    pr = np.concatenate(preds); tr = np.concatenate(targs); test_pb = total_pb / len(testl)
    sh = pr.shape
    pr_mw = sy.inverse_transform(pr.reshape(-1, sh[2])).reshape(sh)
    tr_mw = sy.inverse_transform(tr.reshape(-1, 1)).reshape(tr.shape)

    p50 = pr_mw[:, :, 49]; pf = p50.ravel(); tf = tr_mw.ravel(); mask = tf > 0.001
    rmse = np.sqrt(np.mean((pf[mask] - tf[mask])**2))
    mae  = np.mean(np.abs(pf[mask] - tf[mask]))
    r2   = 1 - np.sum((tf[mask]-pf[mask])**2) / (np.sum((tf[mask]-np.mean(tf[mask]))**2) + 1e-8)

    print(f'\nTEST: Pinball={test_pb:.4f} | R2={r2:.4f} | RMSE={rmse:.4f} | MAE={mae:.4f}')
    print('Per-horizon:')
    for h in [0, 3, 5, 11, 17, 23]:
        er = torch.FloatTensor(tr_mw[:, h]).unsqueeze(-1) - torch.FloatTensor(pr_mw[:, h])
        pb_h = torch.maximum(torch.FloatTensor(QUANTILES)*er, (torch.FloatTensor(QUANTILES)-1)*er).mean().item()
        print(f'  +{h+1:2d}h: {pb_h:.4f}')

    p10 = pr_mw[:, :, 9]; p90 = pr_mw[:, :, 89]
    print(f'80% CI coverage: {np.mean((tr_mw >= p10) & (tr_mw <= p90))*100:.1f}%')

    v1_pb = 0.2069
    print(f'\nv1: 0.2069 | Reg: {test_pb:.4f} | imp: {(v1_pb - test_pb) / v1_pb * 100:+.1f}%')
    print(f'Params: {n_p:,} | Best val: {best_val:.4f}')
    print('Done!')


if __name__ == '__main__':
    main()
