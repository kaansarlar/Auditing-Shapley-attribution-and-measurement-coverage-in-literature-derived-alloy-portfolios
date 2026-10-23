"""Worlds matched to the case-study support structure: N = 17 records, target K = 4 supports, a designated veto
element and a disjoint equivalent pair, two publications per group. Retained worlds reproduce the case diagnostics
exactly: 4 observed supports, main-effect rank 4, one veto player, one equivalent pair, two players with a
single-element contrast."""
import numpy as np,pandas as pd,sys,json
from bench import scenario


def main():
    a,b=int(sys.argv[1]),int(sys.argv[2]); rows=[]
    for s in range(a,b):
        r=np.random.default_rng(s); v=int(r.integers(5)); t=tuple(int(x) for x in r.choice([i for i in range(5) if i!=v],2,replace=False))
        rows.append(scenario(s,dict(N=17,K=4,veto=v,tie=t,ppg=2,gss=0)))
    pd.DataFrame(rows).to_csv(f'hea_match_{a}_{b}.csv',index=False)

if __name__ == '__main__':
    main()
