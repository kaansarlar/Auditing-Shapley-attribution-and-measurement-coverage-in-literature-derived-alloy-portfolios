"""Independent checks for exhaustive allocation and the empirical interface."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_acquisition import checks
from audit_portfolio import audit

def run():
    checks()
    root=Path(__file__).resolve().parent.parent
    frame=pd.read_csv(root/'results/empirical_nash_shapley_v3/empirical_strategy_portfolio.csv')
    r,t=audit(frame,['Mn','Cr','Ni','Al','Cu'])
    assert r['eligible_records']==17 and r['unique_supports']==4
    assert r['veto_players']==['Cr'] and r['equivalent_incidence_pairs']==[['Mn','Al']]
    assert r['main_effect_rank']==4
    np.testing.assert_allclose([r['observed_access_shapley'][e] for e in ['Mn','Cr','Ni','Al','Cu']],
        [.199934,.399773,.100667,.199934,.046961],atol=6e-7)
    assert abs(r['efficiency_residual'])<1e-12
    # Reject undefined/nonfinite observations rather than silently filling them.
    try:audit(pd.DataFrame({'optional_players':['Mn'],'nash_geometric_mean':[float('nan')]}),['Mn'])
    except ValueError:pass
    else:raise AssertionError('Invalid payoff accepted')
    print('Extension checks and independent empirical audit comparison passed.')

if __name__=='__main__':run()
