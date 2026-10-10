"""Synthetic benchmark: when is coalition-Shapley analysis of literature-derived portfolios identifiable?"""
import numpy as np, pandas as pd, itertools, math, sys, json
from scipy.stats import spearmanr
P=5; MASKS=np.arange(32)
SUB=np.array([[(t & ~s)==0 for t in MASKS] for s in MASKS])          # SUB[s,t]: t ⊆ s
BITS=np.array([[(m>>i)&1 for i in range(P)] for m in MASKS])          # 32×5
PAIRS=list(itertools.combinations(range(P),2))
# Shapley weights: phi_i = sum_{S not i} w(|S|) [V(S|i)-V(S)]
W=np.zeros((P,32)); Wp=np.zeros((P,32))
for i in range(P):
    for s in MASKS:
        if (s>>i)&1: continue
        k=bin(s).count('1'); w=math.factorial(k)*math.factorial(P-k-1)/math.factorial(P)
        W[i,s]-=w; W[i,s|(1<<i)]+=w
def SH(V): return np.round(W@V,12)    # exact Shapley; rounding removes float noise so symmetric players tie exactly
sig=lambda x:1/(1+np.exp(-x))

def truth(rng,gscale):
    out={}
    for k in 'TS':
        c=rng.normal(-0.5,0.5); b=rng.normal(0,1,P); g=rng.normal(0,gscale,len(PAIRS)) if gscale>0 else np.zeros(len(PAIRS))
        f=c+BITS@b+sum(g[n]*BITS[:,i]*BITS[:,j] for n,(i,j) in enumerate(PAIRS))
        out[k]=f
    q=np.sqrt(sig(out['T'])*sig(out['S']))
    Vt=np.array([q[SUB[s]].max() for s in MASKS])
    return out,q,Vt,SH(Vt)

def candidate_supports(veto,tie):
    c=[]
    for m in MASKS:
        if veto is not None and not (m>>veto)&1: continue
        if tie is not None and (((m>>tie[0])&1)!=((m>>tie[1])&1)): continue
        c.append(m)
    return np.array(c)

def obs_values(q,sup):
    """V_hat(S)=max observed q with support ⊆ S (0 if none)."""
    M=SUB[:,sup]                       # 32×N
    return np.where(M,q[None,:],0.0).max(1)

def diagnostics(sup,grp):
    us=np.unique(sup); K=len(us); B=BITS[us]
    veto=int((B.all(0)).any()); null=int((~B.any(0)).any())
    eq=0
    for i,j in PAIRS:
        if (B[:,i]==B[:,j]).all() and B[:,i].any(): eq+=1
    S=set(us.tolist()); contrasts=sum(1 for t in us for i in range(P) if (t>>i)&1 and (t&~(1<<i)) in S)
    players_with_contrast=len({i for t in us for i in range(P) if (t>>i)&1 and (t&~(1<<i)) in S})
    sh=np.bincount(pd.factorize(grp)[0])/len(grp); neff=1/np.sum(sh**2)
    Xm=np.c_[np.ones(K),B]; Xp=np.c_[Xm]+0
    Xp=np.c_[Xm,np.array([B[:,i]*B[:,j] for i,j in PAIRS]).T] if K>0 else Xm
    rank_main=np.linalg.matrix_rank(Xm); rank_pair=np.linalg.matrix_rank(Xp)
    return dict(rank_main=rank_main,rank_pair=rank_pair,coverage=K/32,K_obs=K,veto=veto,null_player=null,equiv_pairs=eq,contrasts=contrasts,players_with_contrast=players_with_contrast,eff_groups=neff,top_group_share=sh.max(),n_groups=len(sh))

def top_of(phi):
    m=phi.max(); idx=np.where(np.isclose(phi,m,atol=1e-12))[0]
    return idx[0] if len(idx)==1 else -1     # -1 = tied top (unidentified)

def scenario(seed,ov=None):
    ov=ov or {}
    rng=np.random.default_rng(seed)
    N=int(rng.choice([20,50,100,200])); gscale=float(rng.choice([0,0.5,1.0]))
    se=float(rng.choice([0.1,0.3,0.6])); sp=float(rng.choice([0,0.3,0.6])); sg=float(rng.choice([0,0.3,0.6,1.0]))
    ppg=int(rng.choice([1,2,3])); alpha=float(rng.choice([0,1,2]))
    veto=int(rng.integers(P)) if rng.random()<0.3 else None
    tie=tuple(rng.choice(P,2,replace=False)) if rng.random()<0.3 else None
    Kc=int(rng.choice([3,4,6,8,12,16,24,32])); gss=float(rng.choice([0,0.5,0.9]))
    lam=float(rng.choice([0,1.5,3]))
    N=ov.get('N',N); gscale=ov.get('gscale',gscale); se=ov.get('se',se); sp=ov.get('sp',sp); sg=ov.get('sg',sg); ppg=ov.get('ppg',ppg); gss=ov.get('gss',gss); alpha=ov.get('alpha',alpha); Kc=ov.get('K',Kc); lam=ov.get('lam',lam)
    if 'veto' in ov: veto=ov['veto']
    if 'tie' in ov: tie=ov['tie']
    if veto is not None and tie is not None and veto in tie: tie=None
    f,q,Vt,phit=truth(rng,gscale)
    cand=candidate_supports(veto,tie)
    # publications
    sizes=[]; 
    while sum(sizes)<N: sizes.append(int(rng.integers(2,9)))
    sizes[-1]-=sum(sizes)-N
    sizes=[s for s in sizes if s>0]; npub=len(sizes)
    K=int(min(Kc,len(cand),npub))
    wsel=np.exp(lam*(q[cand]-q[cand].mean())); wsel/=wsel.sum()   # performance-biased study selection
    supK=rng.choice(cand,K,replace=False,p=wsel)
    pr=np.arange(1,K+1,dtype=float)**(-alpha); pr/=pr.sum()
    pub_sup=supK[rng.choice(K,npub,p=pr)]
    pub_grp=np.arange(npub)//ppg
    for gi in np.unique(pub_grp):          # groups tend to keep studying one alloy system
        idx=np.where(pub_grp==gi)[0]
        for j in idx[1:]:
            if rng.random()<gss: pub_sup[j]=pub_sup[idx[0]]
    sup=np.repeat(pub_sup,sizes); pub=np.repeat(np.arange(npub),sizes); grp=np.repeat(pub_grp,sizes)
    ng=pub_grp.max()+1
    qt=np.empty(N); qo=np.empty(N); ft={}; fo={}
    for k in 'TS':
        delta=rng.normal(0,ov.get('sd_alloy',0.3),N); bg=rng.normal(0,sg,ng)[grp]; bp=rng.normal(0,sp,npub)[pub]; e=rng.normal(0,se,N)
        ft[k]=f[k][sup]+delta; fo[k]=ft[k]+bg+bp+e
    qt=np.sqrt(sig(ft['T'])*sig(ft['S'])); qo=np.sqrt(sig(fo['T'])*sig(fo['S']))
    Vh=obs_values(qo,sup); phih=SH(Vh)
    Vp=obs_values(qt,sup); phip=SH(Vp)          # noise-free portfolio game (same supports, true alloy scores)
    rho_port=spearmanr(phip,phih).correlation if np.std(phih)>0 and np.std(phip)>0 else np.nan
    rho_struct=spearmanr(phit,phip).correlation if np.std(phip)>0 else np.nan
    tp=top_of(phip)
    th0=top_of(phih)
    mat_err=int(th0<0 or phit[th0] < 0.5*phit.max())
    rho=spearmanr(phit,phih).correlation if np.std(phih)>0 else np.nan
    tt=top_of(phit); th=top_of(phih)
    # bootstrap: publication and group
    def boot(units,unit_of,B=100):
        tops=[]
        for _ in range(B):
            pick=rng.choice(units,len(units),replace=True)
            idx=np.concatenate([np.where(unit_of==u)[0] for u in pick])
            tops.append(top_of(SH(obs_values(qo[idx],sup[idx]))))
        tops=np.array(tops); return (tops==th).mean() if th>=0 else 0.0
    pubfreq=boot(np.arange(npub),pub); grpfreq=boot(np.arange(ng),grp)
    # group jackknife: does the top change when any group is removed?
    jk_change=0
    if ng>1:
        for gidx in range(ng):
            keep=grp!=gidx
            if top_of(SH(obs_values(qo[keep],sup[keep])))!=th: jk_change+=1
    # decision regret (Nash choice = argmax observed score)
    a=np.argmax(qo); regret_port=qt.max()-qt[a]; regret_world=q.max()-q[sup[a]]
    d=diagnostics(sup,grp)
    return dict(seed=seed,N=N,K_design=K,lam=lam,veto_el=-1 if veto is None else int(veto),veto_design=int(veto is not None),tie_design=int(tie is not None),alpha=alpha,gscale=gscale,sigma_e=se,sigma_pub=sp,sigma_grp=sg,pubs_per_group=ppg,group_support_share=gss,n_pubs=npub,
        rho=rho,rho_port=rho_port,rho_struct=rho_struct,top_port=tp,top_port_correct=int(th==tp and th>=0),material_error=mat_err,top_true=tt,top_hat=th,top_correct=int(th==tt and th>=0),pub_boot_top_freq=pubfreq,grp_boot_top_freq=grpfreq,
        group_jackknife_changes=jk_change,group_jackknife_frac=jk_change/max(ng,1),regret_portfolio=regret_port,regret_world=regret_world,
        true_top_margin=np.sort(phit)[-1]-np.sort(phit)[-2],random_material_error=float(np.mean(phit<0.5*phit.max())),**d)

if __name__=='__main__':
    a,b=int(sys.argv[1]),int(sys.argv[2])
    rows=[scenario(s) for s in range(a,b)]
    pd.DataFrame(rows).to_csv(f'res_{a}_{b}.csv',index=False)
