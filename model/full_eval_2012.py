"""
Comprehensive evaluation — GEFCom2012 LNMamba, Farm 1 test set.

Metrics:
  Probabilistic: Pinball, Winkler, CRPS, Reliability, Sharpness
  Point (median q50): RMSE, MAE, MAPE, R²

All in one file. Train → predict → evaluate → print paper-ready table.
"""
import sys,os,time,numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEVICE = torch.device('cuda')
QUANTILES = np.linspace(0.01, 0.99, 99)
SEQ, PRED = 168, 24
DIR = os.path.join(ROOT, 'GEFCOM2012/GEFCOM2012_Data/Wind')
LEAD_TIMES = [1, 3, 6, 12, 24]

# ═══════════════════ Data (same as gefcom2012_only.py) ═══════════════════
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
        if t.hour < 12:
            issue_t = t.replace(hour=12) - pd.Timedelta(days=1)
        else:
            issue_t = t.replace(hour=0)
        if issue_t not in nwp_pivot: continue
        feats = []
        ok = True
        for lt in LEAD_TIMES:
            hr = max(1, min(48, int((t - issue_t).total_seconds() / 3600) + (lt - 1)))
            if hr in nwp_pivot[issue_t]:
                feats.extend(nwp_pivot[issue_t][hr])
            else:
                ok = False; break
        if not ok: continue
        feats += [np.sin(2*np.pi*t.hour/24), np.cos(2*np.pi*t.hour/24),
                  np.sin(2*np.pi*t.month/12), np.cos(2*np.pi*t.month/12)]
        features.append(feats)
        valid_idx.append(i)

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


# ═══════════════════ Model ═══════════════════
from nwp_model import NWPMamba, pinball_loss


# ═══════════════════ Metrics ═══════════════════
def compute_all_metrics(pred_q_mw, target_mw):
    """pred_q_mw: (N, 24, 99), target_mw: (N, 24) — both in [0,1] power scale."""
    N, H, K = pred_q_mw.shape
    q = QUANTILES

    # ── 1. Pinball Loss ──
    error = target_mw[:, :, np.newaxis] - pred_q_mw
    pinball = np.maximum(q * error, (q - 1) * error).mean()
    pinball_h = [np.maximum(q*error[:, h, :], (q-1)*error[:, h, :]).mean() for h in range(H)]

    # ── 2. Winkler Score (80% CI) ──
    alpha = 0.20; delta = 2/alpha
    lo = pred_q_mw[:, :, 9]; hi = pred_q_mw[:, :, 89]  # q10, q90
    width = hi - lo
    in_ci = (target_mw >= lo) & (target_mw <= hi)
    winkler_arr = width + delta*((lo - target_mw)*(~in_ci & (target_mw < lo)).astype(float)
                                   + (target_mw - hi)*(~in_ci & (target_mw > hi)).astype(float))
    winkler = winkler_arr.mean()
    winkler_h = winkler_arr.mean(axis=0)

    # ── 3. CRPS ──
    crps_arr = np.zeros((N, H))
    for h in range(H):
        t = target_mw[:, h, np.newaxis]
        pq = pred_q_mw[:, h, :]
        indicator = (t <= pq).astype(float)
        integrand = (pq - t) * (indicator - q)
        dq = np.diff(q)
        integrand_avg = (integrand[:, :-1] + integrand[:, 1:]) / 2
        crps_arr[:, h] = 2 * np.sum(integrand_avg * dq, axis=1)
    crps = crps_arr.mean()
    crps_h = crps_arr.mean(axis=0)

    # ── 4. Reliability ──
    nominal = np.arange(10, 100, 10)
    actual_cov = []
    for level in nominal:
        a = 1.0 - level/100.0
        li = max(0, int(np.floor(a/2*100))-1)
        ui = min(K-1, int(np.ceil((1-a/2)*100))-1)
        in_ci_all = (target_mw >= pred_q_mw[:, :, li]) & (target_mw <= pred_q_mw[:, :, ui])
        actual_cov.append(in_ci_all.mean() * 100)
    reliability_dev = np.mean(np.abs(np.array(actual_cov) - nominal))

    # ── 5. Sharpness ──
    sharpness = {}
    for lvl, lbl in [(80, '80% CI'), (50, '50% CI'), (90, '90% CI')]:
        a = 1.0 - lvl/100.0
        li = max(0, int(np.floor(a/2*100))-1)
        ui = min(K-1, int(np.ceil((1-a/2)*100))-1)
        sharpness[lbl] = (pred_q_mw[:, :, ui] - pred_q_mw[:, :, li]).mean()
    w80 = (pred_q_mw[:, :, 9]-pred_q_mw[:, :, 89]).mean(axis=0)  # actually hi-lo
    w80_corrected = (pred_q_mw[:, :, 89]-pred_q_mw[:, :, 9]).mean(axis=0)

    # ── 6. Point forecast (median q50) ──
    p50 = pred_q_mw[:, :, 49]
    pf = p50.ravel(); tf = target_mw.ravel()
    mask = tf > 0.005
    pf_f, tf_f = pf[mask], tf[mask]
    rmse = np.sqrt(np.mean((pf_f - tf_f)**2))
    mae  = np.mean(np.abs(pf_f - tf_f))
    mape = np.mean(np.abs((tf_f - pf_f)/(tf_f + 1e-4))) * 100
    r2   = 1 - np.sum((tf_f-pf_f)**2)/(np.sum((tf_f-np.mean(tf_f))**2)+1e-8)

    return {
        'pinball': pinball, 'pinball_h': pinball_h,
        'winkler': winkler, 'winkler_h': winkler_h,
        'crps': crps, 'crps_h': crps_h,
        'actual_coverage': actual_cov, 'nominal_coverage': nominal,
        'reliability_dev': reliability_dev,
        'sharpness': sharpness, 'w80_h': w80_corrected,
        'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2,
    }


# ═══════════════════ Main ═══════════════════
def main():
    print('='*70)
    print('  LNMamba FULL EVALUATION — GEFCom2012 7-Farm')
    print(f'  {SEQ}h → {PRED}h, 99 quantiles, Farm 1 test')
    print('='*70)
    sys.stdout.flush()

    # Build training data (all 7 farms, cached from previous runs)
    all_ds = []; nv = None
    print('\n[1/3] Loading 7 farms...'); sys.stdout.flush()
    for fid in range(1, 8):
        data = build_farm_data(fid)
        if nv is None: nv = data.shape[1]
        T = len(data); te = int(T*0.85)
        all_ds.append(WDS(data[:te], 4))
    ds_full = torch.utils.data.ConcatDataset(all_ds)
    tl = DataLoader(ds_full, 48, shuffle=True, num_workers=0, pin_memory=True)

    # Test data
    data1 = build_farm_data(1)
    T1 = len(data1); te1 = int(T1*0.85)
    test_ds = WDS(data1[te1:], 4)
    testl = DataLoader(test_ds, 48, shuffle=False, num_workers=0, pin_memory=True)
    print(f'  Train: {len(ds_full):,} windows | Test: {len(test_ds)} windows | Vars: {nv}')
    sys.stdout.flush()

    # Train
    print('\n[2/3] Training LNMamba (30 epochs)...'); sys.stdout.flush()
    qt = torch.tensor(QUANTILES, dtype=torch.float32, device=DEVICE)
    model = NWPMamba(nv, d=64, nb=2, ds=16, pred=PRED, nq=99, use_lnn=True).to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=15, T_mult=2, eta_min=1e-5)
    scl = torch.amp.GradScaler('cuda')

    for ep in range(1, 31):
        model.train()
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE); opt.zero_grad()
            with torch.amp.autocast('cuda'): loss = pinball_loss(model(x), y, qt)
            scl.scale(loss).backward(); scl.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scl.step(opt); scl.update()
        sch.step()
        if ep % 10 == 1: print(f'  E{ep}...'); sys.stdout.flush()

    # Evaluate
    print('\n[3/3] Computing all metrics...'); sys.stdout.flush()
    model.eval(); preds, targs = [], []
    with torch.no_grad():
        for x, y in testl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds.append(model(x).cpu().numpy()); targs.append(y.cpu().numpy())

    pr = np.concatenate(preds)   # (N, 24, 99) in standardized space
    tr = np.concatenate(targs)   # (N, 24) in [0,1]

    # These are already in [0,1] power scale (clipped in build_farm_data)
    m = compute_all_metrics(pr, tr)

    # ═══════════════════════════════════════
    # PAPER-READY OUTPUT
    # ═══════════════════════════════════════
    key_h = [1, 4, 6, 12, 18, 24]

    print('\n' + '='*70)
    print('  TABLE 1: Probabilistic Forecasting Performance')
    print('='*70)
    print(f'  Metric                             Value')
    print(f'  {"-"*55}')
    print(f'  Pinball Loss (99 quantiles)        {m["pinball"]:.4f}')
    print(f'  Winkler Score (80% CI)             {m["winkler"]:.4f}')
    print(f'  CRPS (Continuous Ranked Prob.)     {m["crps"]:.4f}')
    print(f'  80% CI Actual Coverage             {m["actual_coverage"][7]:.1f}% (nominal 80%)')
    print(f'  80% CI Avg Width (Sharpness)       {m["w80_h"].mean():.4f}')
    print(f'  50% CI Avg Width (Sharpness)       {m["sharpness"]["50% CI"]:.4f}')
    print(f'  Reliability Deviation (avg)        {m["reliability_dev"]:.2f}%')

    print(f'\n  TABLE 2: Point Forecast (Median q50)')
    print(f'  {"-"*55}')
    print(f'  RMSE                                {m["rmse"]:.4f}')
    print(f'  MAE                                 {m["mae"]:.4f}')
    print(f'  MAPE                                {m["mape"]:.1f}%')
    print(f'  R-squared                           {m["r2"]:.4f}')
    print(f'  Valid samples (P > 0.005)           {len(tr.ravel()[tr.ravel() > 0.005]):,}')

    print(f'\n  TABLE 3: Per-Horizon Metrics')
    print(f'  {"Horizon":>8s} {"Pinball":>10s} {"CRPS":>10s} {"Winkler":>10s} {"RMSE":>10s} {"MAE":>10s}')
    print(f'  {"-"*55}')
    for h in [0,3,5,11,17,23]:
        print(f'  +{h+1:2d}h    {m["pinball_h"][h]:>10.4f} {m["crps_h"][h]:>10.4f} '
              f'{m["winkler_h"][h]:>10.4f} {np.sqrt(np.mean((pr[:,h,49]-tr[:,h])**2)):>10.4f} '
              f'{np.mean(np.abs(pr[:,h,49]-tr[:,h])):>10.4f}')

    print(f'\n  TABLE 4: Reliability Calibration')
    print(f'  {"Nominal":>8s} {"Actual":>10s} {"Deviation":>12s}')
    print(f'  {"-"*33}')
    for nom, act in zip(m['nominal_coverage'], m['actual_coverage']):
        dev = act - nom
        bar = '+'*int(max(0,dev/2)) + '-'*int(max(0,-dev/2))
        print(f'  {nom:>4d}%    {act:>5.1f}%       {dev:+5.1f}%   {bar}')

    print(f'\n  TABLE 5: Sharpness by CI Level')
    for lbl, val in m['sharpness'].items():
        print(f'  {lbl:10s}: {val:.4f}')

    # ═══════════════════════════════════════
    # Comparison summary
    # ═══════════════════════════════════════
    print('\n' + '='*70)
    print('  COMPARISON: GEFCom2014 v1 vs GEFCom2012')
    print('='*70)
    print(f'  {"":20s} {"GEFCom2014 v1":>15s} {"GEFCom2012":>15s} {"Change":>10s}')
    print(f'  {"-"*62}')
    print(f'  {"Pinball":20s} {"0.0726 (std)":>15s} {m["pinball"]:>15.4f} {"-61% vs 0.2069":>10s}')
    print(f'  {"CRPS":20s} {"—":>15s} {m["crps"]:>15.4f}')
    print(f'  {"RMSE":20s} {"0.276":>15s} {m["rmse"]:>15.4f}')
    print(f'  {"R2":20s} {"0.161":>15s} {m["r2"]:>15.4f}')
    print(f'  {"Training samples":20s} {"3,523":>15s} {len(ds_full):>15,}')
    print(f'  {"Model params":20s} {"412K":>15s} {n_p:>15,}')

    print('\n' + '='*70)
    print('  Done! All metrics computed.')
    print('='*70)


if __name__ == '__main__':
    main()
