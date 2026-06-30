"""LNMamba v4 — Parallel-scan Mamba + Multiscale + LNN + CRPS + 10-zone."""
import sys,os,zipfile,time,numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

DEVICE=torch.device('cuda')
SEQ,PRED,BATCH=168,24,24; EPOCHS,LR=50,5e-4; D_MODEL,DS,NB=64,16,3
QUANTILES=np.linspace(0.01,0.99,99); DATA_DIR='data/gefcom2014'

def wx(df):
    df['WS10']=np.sqrt(df['U10']**2+df['V10']**2)
    df['WS100']=np.sqrt(df['U100']**2+df['V100']**2)
    df['WD10_S']=np.sin(np.arctan2(df['U10'],df['V10']))
    df['WD10_C']=np.cos(np.arctan2(df['U10'],df['V10']))
    df['WD100_S']=np.sin(np.arctan2(df['U100'],df['V100']))
    df['WD100_C']=np.cos(np.arctan2(df['U100'],df['V100']))
    df['SHEAR']=df['WS100']/(df['WS10']+0.1); return df
FEAT=['U10','V10','U100','V100','WS10','WS100','WD10_S','WD10_C','WD100_S','WD100_C','SHEAR']

class Mb(nn.Module):
    def __init__(self,d,ds=16,dc=4,ex=2):
        super().__init__(); self.ds=ds; di=d*ex
        self.inp=nn.Linear(d,di*2,bias=False)
        self.cnv=nn.Conv1d(di,di,dc,groups=di,padding=dc-1)
        self.xp=nn.Linear(di,ds*2+1,bias=False)
        self.dtp=nn.Linear(ds,di,bias=True)
        A=torch.arange(1,ds+1).float().unsqueeze(0)*0.03
        self.A_log=nn.Parameter(torch.log(A))
        self.D=nn.Parameter(torch.ones(di))
        self.out=nn.Linear(di,d,bias=False)
        self.nm=nn.RMSNorm(d)
    def forward(self,x):
        B,L,D=x.shape; res=x
        xz=self.inp(x); u,z=xz.chunk(2,dim=-1)
        u=F.silu(self.cnv(u.transpose(1,2))[:,:,:L].transpose(1,2))
        proj=self.xp(u)
        dt=F.softplus(self.dtp(F.softplus(proj[:,:,:self.ds])))+1e-4
        Bs,Cs=proj[:,:,self.ds:self.ds*2],proj[:,:,self.ds*2:]
        de=dt.unsqueeze(-1)
        Abar=torch.exp(de*(-torch.exp(self.A_log)).unsqueeze(0).unsqueeze(1))
        b=de*Bs.unsqueeze(2)*u.unsqueeze(-1)
        eps=1e-8; logA=torch.log(Abar.clamp(min=eps))
        Acum=torch.exp(torch.cumsum(logA,dim=1))
        h=Acum*torch.cumsum(b/Acum.clamp(min=eps),dim=1)
        y=(h*Cs.unsqueeze(2)).sum(-1)+self.D.unsqueeze(0).unsqueeze(0)*u
        return self.nm(self.out(y*F.silu(z))+res)

class Ln(nn.Module):
    def __init__(self,d,h=48):
        super().__init__()
        self.gru=nn.GRU(d,h,batch_first=True); self.out=nn.Linear(h,d)
    def forward(self,x): h,_=self.gru(x); return torch.sigmoid(self.out(h))

class Ms(nn.Module):
    def __init__(self,d):
        super().__init__(); d4=d//4
        self.c1=nn.Conv1d(d,d4,3,padding=2)
        self.c2=nn.Conv1d(d,d4,7,padding=6)
        self.c3=nn.Conv1d(d,d4,13,padding=12)
        self.c4=nn.Conv1d(d,d4,25,padding=24)
        self.fuse=nn.Linear(d,d); self.nm=nn.LayerNorm(d)
    def forward(self,x):
        B,L,D=x.shape; xt=x.transpose(1,2)
        b1=F.gelu(self.c1(xt))[:,:,:L]; b2=F.gelu(self.c2(xt))[:,:,:L]
        b3=F.gelu(self.c3(xt))[:,:,:L]; b4=F.gelu(self.c4(xt))[:,:,:L]
        return self.nm(x+self.fuse(torch.cat([b1,b2,b3,b4],dim=1).transpose(1,2)))

class LNMamba(nn.Module):
    def __init__(self,V,d=64,nb=3,ds=16,pred=24,nq=99):
        super().__init__(); self.pred_len=pred; self.nq=nq
        self.emb=nn.Sequential(nn.Linear(V,d*2),nn.GELU(),nn.Linear(d*2,d))
        self.pe=nn.Parameter(torch.randn(1,2000,d)*0.02)
        self.ms=Ms(d)
        self.mb=nn.ModuleList([Mb(d,ds) for _ in range(nb)])
        self.ln=nn.ModuleList([Ln(d) for _ in range(nb)])
        self.dec=nn.Sequential(nn.Linear(d,d*2),nn.GELU(),nn.Dropout(0.1),
                               nn.Linear(d*2,d),nn.GELU(),nn.Linear(d,pred*nq))
        self.drop=nn.Dropout(0.08)
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight,0.5)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self,x):
        B,V,L=x.shape
        x=self.emb(x.transpose(1,2))+self.pe[:,:L]; x=self.ms(x)
        for mb,ln in zip(self.mb,self.ln): x=self.drop(mb(x)); x=x*ln(x)
        return self.dec(x[:,-1]).view(B,self.pred_len,self.nq)

class WindDS(Dataset):
    def __init__(self,d,s,p,st):
        self.data=torch.FloatTensor(d); self.seq=s; self.pred=p; self.s=st
        self.n=max(0,(len(d)-s-p)//st+1)
    def __len__(self): return self.n
    def __getitem__(self,i):
        st=i*self.s; return (self.data[st:st+self.seq].T,self.data[st+self.seq:st+self.seq+self.pred,-1])

def pb_loss(p,t,qt):
    e=t.unsqueeze(-1)-p; return torch.maximum(qt*e,(qt-1)*e).mean()
def xpen(p):
    d=p[:,:,1:]-p[:,:,:-1]; return F.relu(-d).mean()

def load_data():
    datasets=[]; nv=None
    for z in range(1,11):
        tz=zipfile.ZipFile(f'{DATA_DIR}/Task15_W_Zone1_10.zip')
        df=pd.read_csv(tz.open(f'Task15_W_Zone1_10/Task15_W_Zone{z}.csv'))
        ts=df['TIMESTAMP'].astype(str).str.strip()
        df['dt']=pd.to_datetime(ts.str[:8],format='%Y%m%d')+pd.to_timedelta(ts.str.extract(r'(\d+):')[0].astype(int),unit='h')
        df=df.sort_values('dt').reset_index(drop=True)
        df['TARGETVAR']=df['TARGETVAR'].interpolate(limit_direction='both')
        for c in ['U10','V10','U100','V100']: df[c]=df[c].interpolate(limit_direction='both')
        df=wx(df)
        h=df['dt'].dt.hour.values.astype(np.float32); m=df['dt'].dt.month.values.astype(np.float32)
        df['HOUR_SIN']=np.sin(2*np.pi*h/24); df['HOUR_COS']=np.cos(2*np.pi*h/24)
        df['MONTH_SIN']=np.sin(2*np.pi*m/12); df['MONTH_COS']=np.cos(2*np.pi*m/12)
        af=FEAT+['HOUR_SIN','HOUR_COS','MONTH_SIN','MONTH_COS']
        feats=StandardScaler().fit_transform(df[af].values.astype(np.float32))
        tgt=StandardScaler().fit_transform(df[['TARGETVAR']].values.astype(np.float32)).ravel()
        data=np.concatenate([feats,tgt.reshape(-1,1)],axis=1)
        if nv is None: nv=data.shape[1]
        T=len(data); te=int(T*0.85)
        datasets.append(WindDS(data[:te],SEQ,PRED,4))
    ds_full=torch.utils.data.ConcatDataset(datasets)
    print(f'Train: {len(ds_full):,} samples, {nv} vars'); sys.stdout.flush()
    # Zone 1 test
    tz=zipfile.ZipFile(f'{DATA_DIR}/Task15_W_Zone1_10.zip')
    df=pd.read_csv(tz.open('Task15_W_Zone1_10/Task15_W_Zone1.csv'))
    ts=df['TIMESTAMP'].astype(str).str.strip()
    df['dt']=pd.to_datetime(ts.str[:8],format='%Y%m%d')+pd.to_timedelta(ts.str.extract(r'(\d+):')[0].astype(int),unit='h')
    df=df.sort_values('dt').reset_index(drop=True)
    df['TARGETVAR']=df['TARGETVAR'].interpolate(limit_direction='both')
    for c in ['U10','V10','U100','V100']: df[c]=df[c].interpolate(limit_direction='both')
    df=wx(df)
    df['HOUR_SIN']=np.sin(2*np.pi*df['dt'].dt.hour.values/24)
    df['HOUR_COS']=np.cos(2*np.pi*df['dt'].dt.hour.values/24)
    df['MONTH_SIN']=np.sin(2*np.pi*df['dt'].dt.month.values/12)
    df['MONTH_COS']=np.cos(2*np.pi*df['dt'].dt.month.values/12)
    feats2=StandardScaler().fit_transform(df[af].values.astype(np.float32))
    sy=StandardScaler(); tgt2=sy.fit_transform(df[['TARGETVAR']].values.astype(np.float32)).ravel()
    data2=np.concatenate([feats2,tgt2.reshape(-1,1)],axis=1)
    T2=len(data2); te2=int(T2*0.85)
    ds_test=WindDS(data2[te2:],SEQ,PRED,4)
    testl=DataLoader(ds_test,64,shuffle=False,num_workers=0,pin_memory=True)
    return DataLoader(ds_full,BATCH,shuffle=True,num_workers=0,pin_memory=True),testl,nv,sy

def main():
    tl,testl,nv,sy=load_data()
    model=LNMamba(nv,d=D_MODEL,nb=NB,ds=DS,pred=PRED).to(DEVICE)
    n_p=sum(p.numel() for p in model.parameters())
    print(f'Params: {n_p:,} | {len(tl)} batches'); sys.stdout.flush()
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=15,T_mult=2,eta_min=1e-5)
    scaler=torch.amp.GradScaler('cuda')
    qt=torch.tensor(QUANTILES,dtype=torch.float32,device=DEVICE)
    best_pb=float('inf'); best_state=None; hist=[]
    for ep in range(1,EPOCHS+1):
        t0=time.time(); model.train(); tl_pb=0.0
        for x,y in tl:
            x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad()
            with torch.amp.autocast('cuda'):
                out=model(x); loss=pb_loss(out,y,qt)+0.03*xpen(out)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt); scaler.update(); tl_pb+=loss.item()
        sch.step(); avg_pb=tl_pb/len(tl); hist.append(avg_pb)
        et=time.time()-t0; star='*' if avg_pb<best_pb else ' '
        if avg_pb<best_pb: best_pb=avg_pb; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
        print(f'E {ep:2d} pb={avg_pb:.4f} {et:.0f}s{star}'); sys.stdout.flush()
        if ep>=25 and ep-np.argmin(hist)>=15: break
    if best_state: model.load_state_dict(best_state)
    model.eval(); preds,targs=[],[]; total_pb=0.0
    with torch.no_grad():
        for x,y in testl:
            x,y=x.to(DEVICE),y.to(DEVICE); out=model(x)
            total_pb+=pb_loss(out,y,qt).item()
            preds.append(out.cpu().numpy()); targs.append(y.cpu().numpy())
    pr=np.concatenate(preds); tr=np.concatenate(targs); test_pb=total_pb/len(testl)
    sh=pr.shape; pr_mw=sy.inverse_transform(pr.reshape(-1,sh[2])).reshape(sh)
    tr_mw=sy.inverse_transform(tr.reshape(-1,1)).reshape(tr.shape)
    p50=pr_mw[:,:,49]; pf=p50.ravel(); tf=tr_mw.ravel(); mask=tf>0.001
    rmse=np.sqrt(np.mean((pf[mask]-tf[mask])**2)); mae=np.mean(np.abs(pf[mask]-tf[mask]))
    r2=1-np.sum((tf[mask]-pf[mask])**2)/(np.sum((tf[mask]-np.mean(tf[mask]))**2)+1e-8)
    print(f'\nTEST: Pinball={test_pb:.4f} | R2={r2:.4f} | RMSE={rmse:.4f} | MAE={mae:.4f}')
    print('Per-horizon:'); ph_pb=[]
    for h in [0,3,5,11,17,23]:
        er=torch.FloatTensor(tr_mw[:,h]).unsqueeze(-1)-torch.FloatTensor(pr_mw[:,h])
        pb_h=torch.maximum(torch.FloatTensor(QUANTILES)*er,(torch.FloatTensor(QUANTILES)-1)*er).mean().item()
        print(f'  +{h+1:2d}h: {pb_h:.4f}'); ph_pb.append(pb_h)
    p10=pr_mw[:,:,9]; p90=pr_mw[:,:,89]
    print(f'80% CI coverage: {np.mean((tr_mw>=p10)&(tr_mw<=p90))*100:.1f}%')
    v1_pb=0.2069; print(f'\nv1: {v1_pb:.4f} | v4: {test_pb:.4f} | imp: {(v1_pb-test_pb)/v1_pb*100:+.1f}%')
    print(f'Params: {n_p:,} | PB mean all horizons: {np.mean(ph_pb):.4f}')
    print('Done!'); sys.stdout.flush()

if __name__=='__main__': main()
