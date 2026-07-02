"""LNN Gate Analysis v2 — channel specialization, hierarchical filtering."""
import sys,os,numpy as np,torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
plt.rcParams.update({'font.family':'sans-serif','font.size':12,'axes.titlesize':15,
    'axes.titleweight':'bold','axes.labelsize':14,'xtick.labelsize':11,'ytick.labelsize':11,
    'legend.fontsize':11,'figure.dpi':150,'savefig.dpi':300})
C={'blue':'#2166AC','red':'#B2182B','green':'#4DAF4A','orange':'#FF7F00','gray':'#888888'}
OUT=os.path.join(ROOT,'paper','figures'); os.makedirs(OUT,exist_ok=True)
DEV=torch.device('cuda'); QT=np.linspace(0.01,0.99,99); SEQ,PRED=168,24

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_analysis import AnalyzableLNMamba, pb_loss, load_farm1_raw, WDS

def train_model(nv, train_ds):
    tl=DataLoader(train_ds,48,shuffle=True)
    qt=torch.tensor(QT,dtype=torch.float32,device=DEV)
    model=AnalyzableLNMamba(nv,d=64,nb=2,ds=16,pred=PRED).to(DEV)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=12,T_mult=2,eta_min=1e-5)
    scl=torch.amp.GradScaler('cuda')
    for ep in range(1,21):
        model.train()
        for x,y in tl:
            x,y=x.to(DEV),y.to(DEV); opt.zero_grad()
            with torch.amp.autocast('cuda'): loss=pb_loss(model(x),y,qt)
            scl.scale(loss).backward(); scl.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scl.step(opt); scl.update()
        sch.step()
    return model

def main():
    print("LNN Gate V2 — Channel Specialization"); sys.stdout.flush()
    X_raw,y_raw,dts=load_farm1_raw()
    Xn=StandardScaler().fit_transform(X_raw)
    data=np.concatenate([Xn,y_raw.reshape(-1,1)],axis=1)
    T=len(data); te=int(T*0.85)
    train_ds=WDS(data[:te],4); test_ds=WDS(data[te:],4)
    nv=data.shape[1]

    print("Training..."); sys.stdout.flush()
    model=train_model(nv, train_ds)
    model.eval(); testl=DataLoader(test_ds,1,shuffle=False)
    g1_all,g2_all,inp_all=[],[],[]
    with torch.no_grad():
        for x,y in testl:
            x,y=x.to(DEV),y.to(DEV); out,gs=model(x,return_gates=True)
            g1_all.append(gs[0][0].cpu().numpy()); g2_all.append(gs[1][0].cpu().numpy())
            inp_all.append(x[0].cpu().numpy())
    g1=np.stack(g1_all); g2=np.stack(g2_all); inp=np.stack(inp_all)
    N=g1.shape[0]; print(f"{N} test samples"); sys.stdout.flush()

    # Channel temporal profiles
    g1_ch=g1.mean(axis=0)  # (168,64)
    ch_lag1=np.zeros(64); ch_hf=np.zeros(64)
    for ch in range(64):
        s=g1_ch[:,ch]; ch_lag1[ch]=np.corrcoef(s[:-1],s[1:])[0,1]
        ch_hf[ch]=np.std(np.diff(s))/(np.std(s)+1e-8)

    fast_ch=np.where(ch_hf>np.percentile(ch_hf,70))[0]
    slow_ch=np.where(ch_lag1>np.percentile(ch_lag1,70))[0]

    # Features
    pow_vol=inp[:,-1,-24:].std(axis=1)  # power volatility
    fast_g=g1[:,:,fast_ch].mean(axis=(1,2))
    slow_g=g1[:,:,slow_ch].mean(axis=(1,2))
    fs_ratio=fast_g/(slow_g+1e-8)
    r_fs,p_fs=pearsonr(fs_ratio,pow_vol)

    g1b_mean=g1.mean(axis=(1,2)); g2b_mean=g2.mean(axis=(1,2))

    print(f"Fast ch: {len(fast_ch)}/{64}, Slow ch: {len(slow_ch)}/{64}")
    print(f"F/S ratio vs vol: r={r_fs:+.4f} p={p_fs:.4f}")
    print(f"B2/B1 gate ratio: {g2b_mean.mean()/g1b_mean.mean():.2f}")

    # Figure
    fig,axes=plt.subplots(2,2,figsize=(14,12))

    # A: Fast vs Slow temporal profiles
    ax=axes[0,0]
    gf=g1_ch[:,fast_ch].mean(axis=1); gs=g1_ch[:,slow_ch].mean(axis=1)
    t=np.arange(168)
    ax.plot(t,gf,'-',color=C['red'],lw=2.5,alpha=0.9,label=f'Fast-responding channels (n={len(fast_ch)})')
    ax.plot(t,gs,'-',color=C['green'],lw=2.5,alpha=0.9,label=f'Slow-integrating channels (n={len(slow_ch)})')
    ax.fill_between(t,gf*0.95,gf*1.05,alpha=0.1,color=C['red'])
    ax.fill_between(t,gs*0.95,gs*1.05,alpha=0.1,color=C['green'])
    ax.axvline(x=144,color='k',lw=0.5,ls='--',alpha=0.3,label='t-24h')
    ax.set_xlabel('Time step (hours)'); ax.set_ylabel('Mean Gate ' + chr(945))
    ax.set_title('A: Gate channels specialize by temporal scale',fontweight='bold',loc='left')
    ax.legend(fontsize=10); ax.grid(True,alpha=0.15,lw=0.3)

    # B: F/S ratio vs volatility
    ax=axes[0,1]
    vb=np.linspace(pow_vol.min(),pow_vol.max(),15)
    fb=[fs_ratio[(pow_vol>=vb[i])&(pow_vol<vb[i+1])].mean() for i in range(len(vb)-1)]
    vm=[(vb[i]+vb[i+1])/2 for i in range(len(vb)-1)]
    idx=np.random.choice(N,min(1500,N),replace=False)
    ax.scatter(pow_vol[idx],fs_ratio[idx],c=C['blue'],s=8,alpha=0.2,edgecolors='none')
    ax.plot(vm,fb,'-',color=C['red'],lw=3,label=f'Trend (r={r_fs:+.3f}, p={p_fs:.4f})')
    ax.set_xlabel('Power Volatility (std 24h)'); ax.set_ylabel('Fast/Slow Gate Ratio')
    ax.set_title('B: Higher volatility → more fast-channel activation',fontweight='bold',loc='left')
    ax.legend(fontsize=11); ax.grid(True,alpha=0.15,lw=0.3)

    # C: Channel landscape
    ax=axes[1,0]
    colors=['#d62728' if ch in fast_ch else '#2ca02c' if ch in slow_ch else '#888888' for ch in range(64)]
    ax.scatter(ch_lag1,ch_hf,c=colors,s=35,alpha=0.7,edgecolors='white',lw=0.5)
    ax.set_xlabel('Temporal autocorrelation (lag-1)'); ax.set_ylabel('High-frequency energy')
    ax.set_title('C: Each point = 1 of 64 gate channels',fontweight='bold',loc='left')
    from matplotlib.lines import Line2D
    le=[Line2D([0],[0],marker='o',color='w',markerfacecolor=C['red'],markersize=10,label=f'Fast ({len(fast_ch)})'),
        Line2D([0],[0],marker='o',color='w',markerfacecolor=C['green'],markersize=10,label=f'Slow ({len(slow_ch)})'),
        Line2D([0],[0],marker='o',color='w',markerfacecolor='gray',markersize=10,label=f'Mixed ({64-len(fast_ch)-len(slow_ch)})')]
    ax.legend(handles=le,fontsize=10,loc='upper right'); ax.grid(True,alpha=0.15,lw=0.3)

    # D: Hierarchical depth
    ax=axes[1,1]
    ax.hist(g1b_mean,bins=30,density=True,color=C['blue'],alpha=0.5,edgecolor='white',lw=0.3,
            label=f'Block 1 (' + chr(956) + f'={g1b_mean.mean():.3f})')
    ax.hist(g2b_mean,bins=30,density=True,color=C['orange'],alpha=0.5,edgecolor='white',lw=0.3,
            label=f'Block 2 (' + chr(956) + f'={g2b_mean.mean():.3f})')
    ax.axvline(g1b_mean.mean(),color=C['blue'],lw=2.5,ls='--',alpha=0.8)
    ax.axvline(g2b_mean.mean(),color=C['orange'],lw=2.5,ls='--',alpha=0.8)
    ax.set_xlabel('Mean Gate ' + chr(945)); ax.set_ylabel('Density')
    ax.set_title(f'D: Deeper = stronger gating (B2={g2b_mean.mean()/g1b_mean.mean()*100:.0f}% of B1)',
                 fontweight='bold',loc='left')
    ax.legend(fontsize=11); ax.grid(True,alpha=0.15,lw=0.3)

    fig.suptitle('Figure 9: LNN Gate Mechanism — Hierarchical Temporal Specialization',
                 fontsize=16,fontweight='bold',y=1.01)
    plt.tight_layout()
    fig.savefig(f'{OUT}/fig9_lnn_mechanism.png',dpi=300)
    fig.savefig(f'{OUT}/fig9_lnn_mechanism.pdf')
    plt.close()
    print(f"Saved {OUT}/fig9_lnn_mechanism.png/pdf")

if __name__=='__main__': main()
