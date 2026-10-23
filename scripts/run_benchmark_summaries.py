"""Recompute comparisons separately for each generator.

The earlier v6 fairness postprocessor's percentile random state was not archived.
This postprocessor explicitly uses seed 20261023 and 4,000 paired world resamples.
Point estimates and Wilson intervals reproduce the same definitions; finite
percentile endpoints can differ from the archived v6 summaries. No latent-world
or outcome definition is changed. Do not pool the generators.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
import run_synthetic_identifiability_benchmark_v6 as primary
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/benchmark_summaries'

def wilson(k,n):
 z=norm.ppf(.975);p=k/n;den=1+z*z/n
 center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return center-half,center+half

def summarize(d,generator,subset,rng):
 u=d[d.top_hat>=0];t=u[u.top_true>=0]
 if u.empty or t.empty:raise ValueError('No eligible worlds: '+generator+', '+subset)
 diff=(u.material_error-u.random_material_error).to_numpy()
 boot=np.array([rng.choice(diff,len(diff),replace=True).mean() for _ in range(4000)])
 lo,hi=np.quantile(boot,[.025,.975]);k=int((t.top_hat==t.top_true).sum());a,b=wilson(k,len(t))
 return dict(generator=generator,subset=subset,worlds=len(d),unique_estimated_top=len(u),
  unique_truth_and_estimate=len(t),material_error=float(u.material_error.mean()),
  random_material_error=float(u.random_material_error.mean()),difference=float(diff.mean()),
  difference_ci_low=lo,difference_ci_high=hi,exact_correct=k,exact_recovery=k/len(t),exact_ci_low=a,exact_ci_high=b)

def canonical(d,targeted=False):
 d=d.copy();d['top_hat']=d.estimated_top;d['top_true']=d.true_top
 d['material_error']=d.material_error_given_unique
 d['veto']=(d.veto_count>0).astype(int);d['equiv_pairs']=d.equivalent_pair_count
 if targeted:
  random=[]
  for row in d.itertuples():
   rng=np.random.default_rng(np.random.SeedSequence([20262026,row.world_id]))
   q,_,_=primary.generate_truth(rng,row.world);phi=primary.shapley(primary.coalition_values(q))
   assert primary.unique_top(phi)==row.true_top
   random.append(float((phi<.5*phi.max()).mean()))
  d['random_material_error']=random
 else:
  phi=d[[f'true_phi_{p}' for p in primary.PLAYERS]].to_numpy()
  d['random_material_error']=(phi<.5*phi.max(1)[:,None]).mean(1)
 return d

def main():
 OUT.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(20261023)
 p=canonical(pd.read_csv(ROOT/'results/primary_generator_v5/scenario_results_v5.csv.gz'))
 r=pd.read_csv(ROOT/'results/robustness_generator_R/bench_all.csv');rows=[]
 for label,d in [('Primary',p),('Independent',r)]:
  classes=[('neither',(d.veto==0)&(d.equiv_pairs==0)),('equivalent only',(d.veto==0)&(d.equiv_pairs>0)),
   ('veto only',(d.veto>0)&(d.equiv_pairs==0)),('veto + equivalent',(d.veto>0)&(d.equiv_pairs>0))]
  for subset,mask in classes:rows.append(summarize(d[mask],label,subset,rng))
 ph=canonical(pd.read_csv(ROOT/'results/synthetic_identifiability_benchmark_v6/hea_like_worlds_v6.csv'),True)
 ph=ph[(ph.n_unique_supports==4)&(ph.main_effect_rank==4)&(ph.veto_count==1)&(ph.equivalent_pair_count==1)&(ph.players_with_contrast==2)]
 rh=pd.read_csv(ROOT/'results/robustness_generator_R/hea_match_all.csv')
 rh=rh[(rh.K_obs==4)&(rh.rank_main==4)&(rh.veto==1)&(rh.equiv_pairs==1)&(rh.players_with_contrast==2)]
 for label,d in [('Primary',ph),('Independent',rh)]:rows.append(summarize(d,label,'HEA matched',rng))
 result=pd.DataFrame(rows);result.to_csv(OUT/'material_error_and_recovery.csv',index=False)
 conf={}
 for unit in ['group','publication']:
  mask=(p[f'{unit}_bootstrap_modal_top']>=0)&(p[f'{unit}_bootstrap_top_frequency']>=.8)
  conf[unit]=dict(confident_worlds=int(mask.sum()),wrong_confident=int(p.loc[mask,f'{unit}_bootstrap_confident_wrong'].sum()),
   wrong_given_confident=float(p.loc[mask,f'{unit}_bootstrap_confident_wrong'].mean()))
 metadata=dict(percentile_seed=20261023,paired_world_resamples=4000,archived_v6_percentile_seed_available=False,
  primary_conditional_confidence=conf,pooling='none; generators analyzed separately')
 (OUT/'configuration_and_confidence.json').write_text(json.dumps(metadata,indent=2))
 print(result.to_string(index=False))
if __name__=='__main__':main()
