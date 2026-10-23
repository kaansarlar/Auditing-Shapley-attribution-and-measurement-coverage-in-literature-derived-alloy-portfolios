"""Paired measurement-allocation experiment; synthetic supports are not feasible alloys.

Run: python extension/run_acquisition.py. No parameter fitting or oracle acquisition.
The second latent generator is adapted from the archived independent generator;
this extension gives BOTH families the explicitly known empty-support baseline 0.
It does not alter, re-estimate, or pool the original independent benchmark.
"""
from pathlib import Path
import json, math, itertools, argparse, gzip
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
PLAYERS=['Mn','Cr','Ni','Al','Cu']
BITS=np.array([[(m>>i)&1 for i in range(5)] for m in range(32)])
PAIRS=list(itertools.combinations(range(5),2))
PAIR=np.column_stack([BITS[:,i]*BITS[:,j] for i,j in PAIRS])
W=np.zeros((5,32))
for i in range(5):
    for s in range(32):
        if s>>i&1: continue
        weight=math.factorial(s.bit_count())*math.factorial(4-s.bit_count())/120
        W[i,s]-=weight; W[i,s|1<<i]+=weight
INITIAL=np.repeat([6,22,11,15],[1,3,7,6])
BUDGETS=[0,4,8,16,27]

def envelope(q):
    v=np.array(q,copy=True)
    for i in range(5):
        hi=np.flatnonzero(BITS[:,i]);lo=hi^(1<<i)
        v[:,hi]=np.maximum(v[:,hi],v[:,lo])
    return v

def unique_top(phi):
    hit=np.isclose(phi,phi.max(1,keepdims=True),rtol=0,atol=1e-10)
    return np.where(hit.sum(1)==1,hit.argmax(1),-1)

def make_truth(rng,n,family):
    q=[]
    for world in range(n):
        if family=='scalar':
            main=rng.normal(.25,.38,5)
            interaction=np.zeros(10)
            if world%2:
                ix=rng.choice(10,int(rng.integers(2,6)),replace=False)
                interaction[ix]=rng.normal(0,.65,len(ix))
                # Archived primary rule ensures at least one appreciable interaction.
                if np.max(np.abs(interaction))<.3:
                    interaction[ix[0]]=rng.choice([-1.,1.])*.55
            f=-.25+BITS@main+PAIR@interaction
            utility=.05+.95/(1+np.exp(-f))
        else:
            lat=[]
            scale=[0,.5,1][world%3]
            for _ in range(2):
                c=rng.normal(-.5,.5); b=rng.normal(0,1,5)
                g=rng.normal(0,scale,10) if scale else np.zeros(10)
                lat.append(c+BITS@b+PAIR@g)
            utility=np.sqrt((1/(1+np.exp(-lat[0])))*(1/(1+np.exp(-lat[1]))))
        utility[0]=0
        q.append(utility)
    return np.array(q)

def allocate(rng,n):
    repeat=np.empty((n,27),int);expand=np.empty((n,27),int)
    original=np.array([6,22,11,15]);missing=np.setdiff1d(np.arange(1,32),original)
    for j in range(n):
        counts=np.array([1,3,7,6])
        for t in range(27):
            ix=rng.choice(np.flatnonzero(counts==counts.min()))
            repeat[j,t]=original[ix];counts[ix]+=1
        expand[j]=rng.permutation(missing)
    return {'repeat':repeat,'expand':expand}

def estimate(q,support,noise,budget,pooling):
    n=len(q);s=support[:,:17+budget];e=noise[:,:17+budget]
    observed=np.clip(np.take_along_axis(q,s,axis=1)+e,0,1)
    ids=np.broadcast_to(np.arange(n)[:,None],s.shape)
    estimates=np.zeros((n,32));counts=np.zeros((n,32),int)
    np.add.at(counts,(ids,s),1)
    if pooling=='mean':
        np.add.at(estimates,(ids,s),observed)
        estimates=np.divide(estimates,counts,out=np.zeros_like(estimates),where=counts>0)
    else:
        np.maximum.at(estimates,(ids,s),observed)
    phi=envelope(estimates)@W.T
    true_phi=envelope(q)@W.T
    top=unique_top(phi);true_top=unique_top(true_phi)
    l1=np.abs(phi-true_phi).sum(1)/true_phi.sum(1)
    best=np.isclose(estimates,estimates.max(1,keepdims=True),rtol=0,atol=1e-10)&(counts>0)
    selected_truth=(q*best).sum(1)/best.sum(1)
    regret=q.max(1)-selected_truth
    return l1,(top==true_top)&(true_top>=0),top<0,regret,true_top>=0

def checks():
    rng=np.random.default_rng(925)
    q=rng.random((8,32));q[:,0]=0;v=envelope(q);phi=v@W.T
    for j in range(8):
        pp=np.zeros(5)
        for perm in itertools.permutations(range(5)):
            s=0
            for i in perm:
                t=s|1<<i;pp[i]+=v[j,t]-v[j,s];s=t
        np.testing.assert_allclose(pp/120,phi[j],atol=1e-12)
    np.testing.assert_allclose(phi.sum(1),v[:,-1],atol=1e-12)
    plans=allocate(rng,8)
    initial=np.broadcast_to(INITIAL,(8,17))
    for arm,new in plans.items():
        s=np.concatenate([initial,new],axis=1)
        assert s.shape==(8,44)
        if arm=='expand':
            assert all(len(np.unique(row))==31 for row in s)
            a=estimate(q,s,np.zeros_like(s,float),27,'mean')
            np.testing.assert_allclose(a[0],0,atol=1e-12)
        else:
            assert all(set(row)=={6,22,11,15} for row in s)
            a=estimate(q,s,np.zeros_like(s,float),0,'mean')
            b=estimate(q,s,np.zeros_like(s,float),27,'mean')
            np.testing.assert_allclose(a[0],b[0],atol=1e-12)
    return {'permutation_shapley':'passed','efficiency':'passed',
            'equal_measurement_counts':'passed','coverage_policy':'passed',
            'noiseless_complete_coverage_recovery':'passed','noiseless_repeat_invariance':'passed'}

def run(n=2000,bootstrap=1000,seed=20261023):
    out=ROOT/'results';out.mkdir(exist_ok=True)
    config={'seed':seed,'worlds_per_family':n,'families':['scalar','two_objective'],
      'additional_budgets':BUDGETS,'initial_masks':[6,22,11,15],'initial_counts':[1,3,7,6],
      'noise_regimes':{'none':[0,0],'independent':[.1,0],'batch':[.1,.1]},
      'pooling':['mean','max'],'bootstrap_world_resamples':bootstrap,
      'empty_support_utility':0,'batch_size':3,'all_binary_supports_synthetically_admissible':True,
      'primary_estimator':'support mean','sensitivity_estimator':'support maximum',
      'recovery_denominator':'all worlds with unique complete-design top; estimated ties count as nonrecovery',
      'L1_normalization':'sum of complete-design Shapley values',
      'cost_unit':'one synthetic scalar-utility observation, not fabrication cost',
      'plan_status':'fixed in code before this extension was run; developed after original study, not preregistered'}
    (out/'configuration.json').write_text(json.dumps(config,indent=2))
    tests=checks();records=[];summary=[];paired=[]
    for fi,family in enumerate(config['families']):
        rng=np.random.default_rng(np.random.SeedSequence([seed,fi]))
        q=make_truth(rng,n,family);plans=allocate(rng,n)
        initial=np.broadcast_to(INITIAL,(n,17))
        supports={a:np.concatenate([initial,s],axis=1) for a,s in plans.items()}
        sensor=rng.normal(size=(n,44))
        # Initial and additional observations have separate batch blocks; all arms share errors.
        batch_index=np.r_[np.arange(17)//3,6+np.arange(27)//3]
        offsets=rng.normal(size=(n,batch_index.max()+1))[:,batch_index]
        np.savez_compressed(out/(family+'_world_inputs.npz'),true_support_utilities=q,
            repeat_supports=plans['repeat'],expansion_supports=plans['expand'],
            standardized_sensor_errors=sensor,standardized_batch_offsets=offsets,
            initial_supports=INITIAL,batch_index=batch_index)
        boot_rng=np.random.default_rng(np.random.SeedSequence([seed,fi,981]))
        boot_ids=boot_rng.integers(0,n,(bootstrap,n))
        for regime,(se,sb) in config['noise_regimes'].items():
            noise=se*sensor+sb*offsets
            for pooling in config['pooling']:
                for budget in BUDGETS:
                    results={a:estimate(q,s,noise,budget,pooling) for a,s in supports.items()}
                    if budget==0:
                        for k in range(4): np.testing.assert_array_equal(results['repeat'][k],results['expand'][k])
                    for arm,r in results.items():
                        valid=r[4];num=int(valid.sum())
                        summary.append(dict(family=family,noise=regime,pooling=pooling,budget=budget,arm=arm,
                            n_worlds=n,n_unique_truth=num,recovery=float(r[1][valid].mean()),
                            mean_normalized_l1=float(r[0].mean()),estimated_top_tie_rate=float(r[2].mean()),
                            mean_candidate_regret=float(r[3].mean())))
                        records.append(pd.DataFrame(dict(family=family,noise=regime,pooling=pooling,budget=budget,
                            arm=arm,world=np.arange(n),normalized_l1=r[0],correct_top=r[1],estimated_tie=r[2],
                            candidate_regret=r[3],unique_true_top=valid)))
                    for metric,k in [('normalized_l1',0),('recovery',1),('candidate_regret',3)]:
                        d=results['expand'][k].astype(float)-results['repeat'][k].astype(float)
                        if metric=='recovery':
                            valid=results['repeat'][4]
                            d=np.where(valid,d,np.nan)
                        bs=np.nanmean(d[boot_ids],axis=1)
                        lo,hi=np.quantile(bs,[.025,.975])
                        paired.append(dict(family=family,noise=regime,pooling=pooling,budget=budget,metric=metric,
                            expansion_minus_repeat=float(np.nanmean(d)),ci_low=float(lo),ci_high=float(hi)))
        print(f'{family}: {n} worlds complete',flush=True)
    pd.concat(records,ignore_index=True).to_csv(out/'world_results.csv.gz',index=False,compression='gzip')
    s=pd.DataFrame(summary);p=pd.DataFrame(paired)
    s.to_csv(out/'summary.csv',index=False);p.to_csv(out/'paired_differences.csv',index=False)
    tests['paired_budget_zero_identity']='passed'
    (out/'verification.json').write_text(json.dumps(tests,indent=2))
    fig,axes=plt.subplots(2,3,figsize=(10.8,6.4),sharex=True,sharey=True)
    for i,family in enumerate(config['families']):
        for j,regime in enumerate(config['noise_regimes']):
            ax=axes[i,j]
            for arm,color,label in [('repeat','#a05a32','Repeat covered supports'),('expand','#155d8b','Cover missing supports')]:
                rows=s[(s.family==family)&(s.noise==regime)&(s.pooling=='mean')&(s.arm==arm)]
                ax.plot(rows.budget,rows.recovery,marker='o',color=color,label=label,linewidth=1.7,markersize=4)
            ax.set_ylim(0,1.03);ax.grid(alpha=.2)
            ax.set_title(['No observation noise','Independent noise','Shared batch offset'][j],fontsize=10)
            if j==0: ax.set_ylabel(('Scalar utility' if i==0 else 'Two latent objectives')+'\nTop-player recovery')
            if i==1:ax.set_xlabel('Additional observations')
    axes[0,1].legend(loc='upper center',bbox_to_anchor=(.5,1.45),ncol=2,frameon=False,fontsize=9)
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(out/'acquisition_recovery.png',dpi=350)
    fig.savefig(out/'acquisition_recovery.pdf');plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worlds',type=int,default=2000);parser.add_argument('--bootstrap',type=int,default=1000)
    args=parser.parse_args();run(args.worlds,args.bootstrap)
