"""Exploratory interval-based equivalence sensitivity; margins selected post hoc.
Recomputes all denominators from frozen scenario records. No long simulations.
"""
from pathlib import Path
import importlib.util,json,math
import numpy as np
import pandas as pd
from scipy.stats import norm
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/v7_final_additions'
P=['Mn','Cr','Ni','Al','Cu']

def wilson(k,n):
    z=norm.ppf(.95);p=k/n;den=1+z*z/n
    c=(p+z*z/(2*n))/den;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return c-h,c+h

def main():
    OUT.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(20261008);rows=[]
    def add(gen,subset,diff,k,n):
        lo,hi=(np.nan,np.nan) if diff is None else np.quantile(np.asarray(diff)[rng.integers(0,len(diff),(4000,len(diff)))].mean(1),[.05,.95])
        a,b=wilson(k,n)
        row={'generator':gen,'subset':subset,'material_n':len(diff) if diff is not None else 0,
             'material_difference':float(np.mean(diff)) if diff is not None else np.nan,
             'material_90_low':lo,'material_90_high':hi,'exact_k':k,'exact_n':n,'exact_rate':k/n,
             'exact_90_low':a,'exact_90_high':b}
        for delta in [.05,.075]:
            row[f'material_within_{delta}']=bool(lo>-delta and hi<delta) if diff is not None else None
            row[f'exact_within_{delta}']=bool(a>.2-delta and b<.2+delta)
        rows.append(row)
    d=pd.read_csv(ROOT/'results/primary_generator_v5/scenario_results_v5.csv.gz')
    phi=d[[f'true_phi_{p}' for p in P]].values;d['random_error']=(phi<.5*phi.max(1)[:,None]).mean(1)
    for cls,mask in [('veto only',(d.veto_count>0)&(d.equivalent_pair_count==0)),('veto + equivalent',(d.veto_count>0)&(d.equivalent_pair_count>0))]:
        m=d[mask&(d.estimated_top_tied==0)]
        add('Primary',cls,m.material_error_given_unique-m.random_error,int(m.top_player_correct_given_unique.sum()),len(m))
    spec=importlib.util.spec_from_file_location('benchmark',ROOT/'scripts/run_synthetic_identifiability_benchmark_v6.py')
    bm=importlib.util.module_from_spec(spec);spec.loader.exec_module(bm)
    h=pd.read_csv(ROOT/'results/synthetic_identifiability_benchmark_v6/hea_like_worlds_v6.csv')
    random=[]
    for row in h.itertuples():
        rng_world=np.random.default_rng(np.random.SeedSequence([20262026,int(row.world_id)]))
        truth,_,_=bm.generate_truth(rng_world,row.world);ph=bm.shapley(bm.coalition_values(truth))
        assert bm.unique_top(ph)==row.true_top
        random.append(float(np.mean(ph<.5*ph.max())))
    h['random_error']=random
    h=h[(h.n_unique_supports==4)&(h.main_effect_rank==4)&(h.veto_count==1)&(h.equivalent_pair_count==1)&(h.players_with_contrast==2)&(h.estimated_top_tied==0)]
    add('Primary','HEA matched',h.material_error_given_unique-h.random_error,int(h.top_correct_given_unique.sum()),len(h))
    for unit in ['group','publication']:
        m=h[h[f'{unit}_bootstrap_confident']==1]
        k=int(len(m)-m[f'{unit}_bootstrap_confident_wrong'].sum())
        add('Primary',f'HEA matched {unit} confident',None,k,len(m))
    d=pd.read_csv(ROOT/'results/robustness_generator_R/bench_all.csv')
    for cls,mask in [('veto only',(d.veto==1)&(d.equiv_pairs==0)),('veto + equivalent',(d.veto==1)&(d.equiv_pairs>0))]:
        m=d[mask&(d.top_hat>=0)];t=m[m.top_true>=0]
        add('Independent',cls,m.material_error-m.random_material_error,int((t.top_hat==t.top_true).sum()),len(t))
    h=pd.read_csv(ROOT/'results/robustness_generator_R/hea_match_all.csv')
    h=h[(h.K_obs==4)&(h.rank_main==4)&(h.veto==1)&(h.equiv_pairs==1)&(h.players_with_contrast==2)&(h.top_hat>=0)]
    t=h[h.top_true>=0]
    add('Independent','HEA matched',h.material_error-h.random_material_error,int((t.top_hat==t.top_true).sum()),len(t))
    for unit,col in [('publication','pub_boot_top_freq'),('group','grp_boot_top_freq')]:
        m=t[t[col]>=.9];add('Independent',f'HEA matched {unit} confident',None,int((m.top_hat==m.top_true).sum()),len(m))
    result=pd.DataFrame(rows);result.to_csv(OUT/'exploratory_equivalence_sensitivity.csv',index=False)
    assert list(result.loc[result.subset.str.contains('confident'),'exact_n'])==[251,282,66,69]
    (OUT/'equivalence_summary.json').write_text(json.dumps({'seed':20261008,'paired_resamples':4000,
        'intervals':'90% percentile paired bootstrap for material-error differences; 90% Wilson for exact recovery',
        'margins':[.05,.075],'post_hoc':True,'reference_recovery':.2,
        'interpretation':'Exploratory CI-inclusion sensitivity corresponding to the two-one-sided-test logic; not a prespecified confirmatory equivalence claim.'},indent=2))
    print(result.to_string(index=False))
if __name__=='__main__':main()
