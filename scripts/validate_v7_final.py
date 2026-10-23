"""Numerical checks from source; no archived outputs or document files needed."""
from pathlib import Path
import runpy,sys
ROOT=Path(__file__).resolve().parents[1]

def main():
 count=0
 for rel in ['tests/test_synthetic_identifiability_benchmark_v6.py','independent_generator_R_v6_1/tests/test_v6_robustness_generator.py']:
  namespace=runpy.run_path(str(ROOT/rel))
  for name,function in namespace.items():
   if name.startswith('test_') and callable(function):
    function();print('PASS '+name);count+=1
 sys.path.insert(0,str(ROOT/'extension'))
 from run_acquisition import checks
 checks();count+=1
 sys.path.insert(0,str(ROOT/'scripts'))
 import pandas as pd
 import numpy as np
 from run_empirical_nash_shapley import prepare_portfolio,add_utilities
 from audit_portfolio import audit
 frame,_=add_utilities(prepare_portfolio(pd.read_csv(ROOT/'data/master_data_v3/master_strict_tc_72.csv')))
 report,_=audit(frame,['Mn','Cr','Ni','Al','Cu'])
 assert report['eligible_records']==17 and report['unique_supports']==4
 assert report['veto_players']==['Cr'] and report['equivalent_incidence_pairs']==[['Mn','Al']]
 np.testing.assert_allclose(list(report['observed_access_shapley'].values()),
  [.1999337155,.3997727459,.1006670293,.1999337155,.046961031],atol=1e-9)
 assert abs(report['efficiency_residual'])<1e-12
 count+=1
 print(f'{count} source-based checks passed. Long simulation runs are separate stages.')
if __name__=='__main__':main()
