"""Audit a finite candidate portfolio without asserting chemical identification.

Example from package root:
python extension/audit_portfolio.py --input results/empirical_nash_shapley_v3/empirical_strategy_portfolio.csv --output extension/results/empirical_audit
"""
from pathlib import Path
import argparse,json,itertools,math
import numpy as np
import pandas as pd

def audit(frame,players,payoff='nash_geometric_mean',support='optional_players'):
    if len(players)!=len(set(players)) or not 1<=len(players)<=10:
        raise ValueError('Specify 1–10 distinct player names; exhaustive coalitions scale exponentially.')
    if 'game_eligible' in frame:
        eligible=frame.game_eligible.astype(str).str.lower().isin(['true','1'])
        frame=frame[eligible].copy()
    if frame.empty:raise ValueError('No eligible records.')
    values=pd.to_numeric(frame[payoff],errors='raise').to_numpy()
    if not np.isfinite(values).all() or (values<0).any():
        raise ValueError('Payoffs must be finite and nonnegative for the zero-access envelope.')
    masks=[]
    for item in frame[support].fillna(''):
        names=set(str(item).split(';'))-{''}
        if names-set(players):raise ValueError('Unknown support elements: '+str(names-set(players)))
        masks.append(sum(1<<players.index(e) for e in names))
    p=len(players);masks=np.array(masks);unique=np.unique(masks)
    bits=np.array([[(m>>i)&1 for i in range(p)] for m in unique])
    q={int(s):float(values[masks==s].max()) for s in unique}
    v=np.array([max([0.]+[val for s,val in q.items() if s&c==s]) for c in range(1<<p)])
    phi=np.zeros(p)
    for i in range(p):
        for s in range(1<<p):
            if s>>i&1:continue
            k=s.bit_count();w=math.factorial(k)*math.factorial(p-k-1)/math.factorial(p)
            phi[i]+=w*(v[s|1<<i]-v[s])
    contrasts={e:[] for e in players}
    for i,e in enumerate(players):
        contrasts[e]=[[int(s),int(s|1<<i)] for s in unique if not s>>i&1 and (s|1<<i) in q]
    veto=[e for i,e in enumerate(players) if bits[:,i].all() and v[-1]>0 and v[0]==0]
    equivalent=[[players[i],players[j]] for i,j in itertools.combinations(range(p),2)
                if np.array_equal(bits[:,i],bits[:,j])]
    report={'eligible_records':len(frame),'unique_supports':len(unique),
      'support_coverage':len(unique)/(1<<p),'players':players,
      'record_incidence':{e:int(((masks>>i)&1).sum()) for i,e in enumerate(players)},
      'main_effect_rank':int(np.linalg.matrix_rank(np.column_stack([np.ones(len(unique)),bits]))),
      'main_effect_columns':p+1,'veto_players':veto,'equivalent_incidence_pairs':equivalent,
      'direct_support_contrasts':contrasts,'observed_access_shapley':dict(zip(players,map(float,phi))),
      'efficiency_residual':float(phi.sum()-(v[-1]-v[0])),
      'scope':'Allocation of access to observed candidate scores. Support contrasts do not control fractions, processing, phases, or sources. Full linear rank is not a certificate of complete-design Shapley identification.',
      'empty_support_observed':0 in q,
      'interpretation':'Missing candidates give zero access, not zero physical performance.'}
    return report,pd.DataFrame([{'mask':s,'support':';'.join(players[i] for i in range(p) if s>>i&1),
                               'records':int((masks==s).sum()),'maximum_payoff':q[s]} for s in unique])

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--players',nargs='+',default=['Mn','Cr','Ni','Al','Cu'])
    parser.add_argument('--payoff',default='nash_geometric_mean');parser.add_argument('--support',default='optional_players')
    args=parser.parse_args();report,table=audit(pd.read_csv(args.input),args.players,args.payoff,args.support)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'audit.json').write_text(json.dumps(report,indent=2))
    table.to_csv(args.output/'support_summary.csv',index=False)
    print(json.dumps(report,indent=2))
