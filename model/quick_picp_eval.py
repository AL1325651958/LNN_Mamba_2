"""Quick training + PICP/PINAW evaluation for GEFCom2012."""
import sys,os,time,numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEVICE = torch.device('cuda')
QUANTILES = np.linspace(0.01, 0.99, 99)
SEQ, PRED = 168, 24
DIR = os.path.join(ROOT, 'GEFCOM2012/GEFCOM2012_Data/Wind')
LEAD_TIMES = [1, 3, 6, 12, 24]


def build_farm_data(fid):
    pw = pd.read_csv(f'{DIR}/windpowermeasurements.csv')
    pw = pw[pw['usage'] == 'Training'].copy()
    pw['dt'] = pd.to_datetime(pw['date'].astype(str), format='%Y%m%d%H')
    pw = pw[['dt', f'wp{fid}']].rename(columns={f'wp{fid}': 'power'}).sort_values('dt').reset_index(drop=True)

    nwp = pd.read_csv(f'{DIR}/windforecasts_wf{fid}.csv')
    nwp['issue'] = pd.to_datetime(nwp['date'].astype(str), format='%Y%m%d%H')
    nwp_pivot = {}
    for _, row in nwp.iterrows():
        nwp_pivot.setdefault(row['issue'], {})[row['hors']] = (row['u'], row['v'], row['ws'], row['wd'])

    features, valid_idx = [], []
    for i, row in pw.iterrows():
        t = row['dt']
        if t.hour < 12: issue_t = t.replace(hour=12) - pd.Timedelta(days=1)
        else: issue_t = t.replace(hour=0)
        if issue_t not in nwp_pivot: continue
        feats = []; ok = True
        for lt in LEAD_TIMES:
            hr = max(1, min(48, int((t - issue_t).total_seconds() / 3600) + (lt - 1)))
            if hr in nwp_pivot[issue_t]: feats.extend(nwp_pivot[issue_t][hr])
            else: ok = False; break
        if not ok: continue
        feats += [np.sin(2*np.pi*t.hour/24), np.cos(2*np.pi*t.hour/24),
                  np.sin(2*np.pi*t.month/12), np.cos(2*np.pi*t.month/12)]
        features.append(feats); valid_idx.append(i)

    arr = np.array(features, dtype=np.float32)
    pwr = pw['power'].values[valid_idx].astype(np.float32).clip(0, 1)
    arr_norm = StandardScaler().fit_transform(arr)
    return np.concatenate([arr_norm, pwr.reshape(-1, 1)], axis=1)


class WDS(Dataset):
    def __init__(self, d, s):
        self.data = torch.FloatTensor(d); self.s = s
        self.n = max(0, (len(d)-SEQ-PRED)//s+1)
    def __len__(self): return self.n
    def __getitem__(self, i):
        st = i*self.s
        return (self.data[st:st+SEQ].T, self.data[st+SEQ:st+SEQ+PRED, -1])


from nwp_model import NWPMamba, pinball_loss, compute_picp_pinaw


def main():
    print('='*60)
    print('  PICP / PINAW Evaluation — GEFCom2012')
    print('='*60)
    sys.stdout.flush()

    # Load all 7 farms for training
    all_ds = []; nv = None
    for fid in range(1, 8):
        data = build_farm_data(fid)
        if nv is None: nv = data.shape[1]
        T = len(data); te = int(T*0.85)
        all_ds.append(WDS(data[:te], 4))
        print(f'  Farm {fid}: {len(all_ds[-1])} windows')
    ds_full = torch.utils.data.ConcatDataset(all_ds)
    print(f'  Total train: {len(ds_full):,} windows')

    # Test: Farm 1 last 15%
    data1 = build_farm_data(1)
    T1 = len(data1); te1 = int(T1*0.85)
    test_ds = WDS(data1[te1:], 4)
    tl = DataLoader(ds_full, 48, shuffle=True, num_workers=0, pin_memory=True)
    testl = DataLoader(test_ds, 48, shuffle=False, num_workers=0, pin_memory=True)
    print(f'  Test (Farm 1): {len(test_ds)} windows\n')

    # Model
    model = NWPMamba(nv, d=64, nb=2, ds=16, pred=PRED, nq=99, use_lnn=True).to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters())
    print(f'  Params: {n_p:,}')

    # Train (quick: 10 epochs, early stop if converged)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-5)
    scl = torch.amp.GradScaler('cuda')
    qt = torch.tensor(QUANTILES, dtype=torch.float32, device=DEVICE)
    best_pb = float('inf'); best_state = None; hist = []
    EPOCHS = 15

    print(f'\n{"="*60}\n  TRAINING ({EPOCHS} epochs)\n{"="*60}')
    for ep in range(1, EPOCHS+1):
        t0 = time.time(); model.train(); tl_pb = 0.0
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE); opt.zero_grad()
            with torch.amp.autocast('cuda'): loss = pinball_loss(model(x), y, qt)
            scl.scale(loss).backward(); scl.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scl.step(opt); scl.update(); tl_pb += loss.item()
        sch.step()
        avg_pb = tl_pb/len(tl); hist.append(avg_pb); et = time.time()-t0
        star = ' ★' if avg_pb < best_pb else ''
        if avg_pb < best_pb: best_pb = avg_pb; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        print(f'  E{ep:2d} pb={avg_pb:.4f} ({et:.0f}s){star}')
        sys.stdout.flush()

    if best_state: model.load_state_dict(best_state)

    # Test predictions
    print(f'\n{"="*60}\n  EVALUATION\n{"="*60}')
    model.eval(); preds, targs = [], []; total_pb = 0.0
    with torch.no_grad():
        for x, y in testl:
            x, y = x.to(DEVICE), y.to(DEVICE); out = model(x)
            total_pb += pinball_loss(out, y, qt).item()
            preds.append(out.cpu().numpy()); targs.append(y.cpu().numpy())
    pr = np.concatenate(preds); tr = np.concatenate(targs)
    pb_test = total_pb/len(testl)

    # Point metrics
    p50 = pr[:,:,49]; pf = p50.ravel(); tf = tr.ravel()
    mask = tf > 0.005
    rmse = np.sqrt(np.mean((pf[mask]-tf[mask])**2))
    mae  = np.mean(np.abs(pf[mask]-tf[mask]))
    r2 = 1 - np.sum((tf[mask]-pf[mask])**2)/(np.sum((tf[mask]-np.mean(tf[mask]))**2)+1e-8)

    print(f'  Pinball: {pb_test:.4f}  R²: {r2:.4f}  RMSE: {rmse:.4f}  MAE: {mae:.4f}\n')

    # ═══════════════════ PICP / PINAW ═══════════════════
    pi = compute_picp_pinaw(pr, tr, QUANTILES)

    print(f'  {"Prediction Interval Metrics":—^56}')
    hdr = f'  {"PI":>6s}  {"Nominal":>8s}  {"PICP":>8s}  {"PINAW":>8s}  {"ACE":>8s}'
    print(hdr)
    print('  ' + '-'*(len(hdr)-2))
    for label, r in pi.items():
        ace = r['picp'] - r['nominal']
        print(f'  {label:>6s}  {r["nominal"]:>8.3f}  {r["picp"]:>8.4f}  {r["pinaw"]:>8.4f}  {ace:>+8.4f}')

    print(f'\n  {"Per-horizon 90% PI (PICP / PINAW)":—^56}')
    r90 = pi['90%']
    for h in [0,3,5,11,17,23]:
        print(f'    +{h+1:2d}h  PICP={r90["picp_per_h"][h]:.4f}  PINAW={r90["pinaw_per_h"][h]:.4f}')

    print(f'\n  {"Key Insights":—^56}')
    # PICP vs nominal: is the model well-calibrated?
    for label, r in pi.items():
        diff = r['picp'] - r['nominal']
        if abs(diff) < 0.03:
            status = '✓ Well-calibrated'
        elif diff > 0:
            status = '⚠ Conservative (over-covering)'
        else:
            status = '⚠ Underconfident (under-covering)'
        print(f'  {label} PI: {status} (Δ={diff:+.4f})')

    print(f'\n  Done!')
    sys.stdout.flush()


if __name__ == '__main__':
    main()
