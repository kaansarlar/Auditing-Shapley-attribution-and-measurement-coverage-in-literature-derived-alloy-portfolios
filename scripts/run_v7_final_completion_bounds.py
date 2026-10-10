"""Global Shapley bounds for bounded monotone completions preserving four observed access values.
Every optimum and ranking witness is exported at full precision. This is a restricted
admissible class, not a chemical response model. Run from any working directory.
"""
from pathlib import Path
from math import factorial
import itertools,json
import numpy as np
import pandas as pd
from scipy.optimize import linprog
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/v7_final_additions'
P=['Mn','Cr','Ni','Al','Cu']; N=32

def setup():
    frame=pd.read_csv(ROOT/'results/empirical_nash_shapley_v3/empirical_strategy_portfolio.csv')
    obs={}
    for row in frame[frame.game_eligible].itertuples():
        mask=sum(1<<P.index(e) for e in row.optional_players.split(';'))
        obs[mask]=max(obs.get(mask,0),row.nash_geometric_mean)
    W=np.zeros((5,N))
    for i in range(5):
        for s in range(N):
            if s>>i&1:continue
            w=factorial(s.bit_count())*factorial(4-s.bit_count())/120
            W[i,s]-=w;W[i,s|1<<i]+=w
    A=[]
    for s in range(N):
        for i in range(5):
            if not s>>i&1:
                a=np.zeros(N);a[s]=1;a[s|1<<i]=-1;A.append(a)
    return obs,W,np.array(A)

def envelope(q):
    return np.array([max(q[t] for t in range(N) if t&c==t) for c in range(N)])

def permutation_shapley(v):
    phi=np.zeros(5)
    for perm in itertools.permutations(range(5)):
        s=0
        for i in perm:
            nxt=s|1<<i;phi[i]+=v[nxt]-v[s];s=nxt
    return phi/120

def solve_all():
    OUT.mkdir(parents=True,exist_ok=True)
    obs,W,A=setup();records=[];witness=[]
    for label,cap in [('unit_cap',1.0),('measured_cap',max(obs.values()))]:
        bounds=[(0,cap)]*N;bounds[0]=(0,0)
        for s,v in obs.items():bounds[s]=(v,v)
        for i,player in enumerate(P):
            vals={}
            for objective in ['min','max','lead','trail']:
                if objective in ['min','max']:
                    c=W[i] if objective=='min' else -W[i]
                    res=linprog(c,A_ub=A,b_ub=np.zeros(len(A)),bounds=bounds,method='highs')
                else:
                    sign=1 if objective=='lead' else -1
                    aa=[np.append(a,0) for a in A]
                    aa += [np.append(-sign*(W[i]-W[j]),1) for j in range(5) if j!=i]
                    c=np.zeros(33);c[-1]=-1
                    res=linprog(c,A_ub=aa,b_ub=np.zeros(len(aa)),bounds=bounds+[(None,None)],method='highs')
                assert res.success,(label,player,objective,res.message)
                v=res.x[:32];q=v.copy()
                for s,a in obs.items():q[s]=a
                phi=W@v
                assert np.max(np.abs(envelope(q)-v))<1e-9
                assert np.max(np.abs(permutation_shapley(v)-phi))<1e-9
                assert np.max(A@v)<1e-9 and abs(phi.sum()-v[-1])<1e-9
                assert all(abs(v[s]-a)<1e-9 and abs(q[s]-a)<1e-9 for s,a in obs.items())
                assert v.min()>-1e-9 and v.max()<cap+1e-9
                val=phi[i] if objective in ['min','max'] else res.x[-1]
                # Check primal feasibility and duality for each optimization problem.
                dual=float(np.dot(res.ineqlin.marginals,np.zeros(len(res.ineqlin.marginals))))
                for k,(lo,hi) in enumerate(bounds+([(None,None)] if objective in ['lead','trail'] else [])):
                    if lo is not None:dual+=res.lower.marginals[k]*lo
                    if hi is not None:dual+=res.upper.marginals[k]*hi
                assert abs(res.fun-dual)<1e-8
                vals[objective]=float(val)
                for s in range(N):
                    witness.append({'cap_class':label,'cap':cap,'player':player,'objective':objective,'mask':s,
                        'support':';'.join(P[j] for j in range(5) if s>>j&1) or 'empty',
                        'observed':s in obs,'q':q[s],'v':v[s],**{f'phi_{p}':phi[j] for j,p in enumerate(P)},
                        'objective_value':val,'primal_objective':res.fun,'dual_objective':dual,
                        'solver_status':res.status})
            records.append({'cap_class':label,'cap':cap,'player':player,**vals,
                'can_be_unique_top':vals['lead']>1e-8,'can_be_unique_bottom':vals['trail']>1e-8})
    pd.DataFrame(records).to_csv(OUT/'coalition_preserving_shapley_bounds.csv',index=False)
    pd.DataFrame(witness).to_csv(OUT/'coalition_preserving_completion_witnesses.csv',index=False)
    summary={'class':'v(empty)=0; 0<=v<=cap; v monotone; four observed access values fixed',
        'observed_values':{str(s):a for s,a in obs.items()},'optimization':'scipy.optimize.linprog method highs',
        'bounds':'global LP optima within solver tolerance, not grid-search inner bounds',
        'witnesses':40,'support_rows':1280,'primal_dual_tolerance':1e-8,
        'checks':'permutation Shapley, envelope reconstruction, measured utilities and access values, monotonicity, efficiency, bounds, duality',
        'limitation':'Preserving the observed access envelope is an additional admissible-completion restriction; actual measurements alone need not fix the unknown complete-game coalition maxima.',
        'results':records}
    (OUT/'completion_bounds_summary.json').write_text(json.dumps(summary,indent=2))
    print(pd.DataFrame(records).to_string(index=False))

if __name__=='__main__':solve_all()
