import numpy as np,pandas as pd,json
exec(open('bench.py').read().split("if __name__")[0])
rows=[]
for s in range(90000,90020):
    r=scenario(s,dict(N=1500,K=32,se=0,sp=0,sg=0,sd_alloy=0,alpha=0,gss=0,lam=0,veto=None,tie=None))
    rows.append(r)
d=pd.DataFrame(rows); d.to_csv('sanity_full_coverage.csv',index=False)
o=dict(n=len(d),K_obs_min=int(d.K_obs.min()),rho_min=float(d.rho.min()),rho_median=float(d.rho.median()),top_correct=float(d.top_correct.mean()),material_error=float(d.material_error.mean()))
print(o); json.dump(o,open('sanity_summary.json','w'),indent=1)
