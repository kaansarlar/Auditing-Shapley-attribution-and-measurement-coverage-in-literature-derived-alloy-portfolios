"""Verified counterexample and equal-budget decision controls. Run from any cwd."""
from pathlib import Path
from math import factorial, comb
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'v6_2_additions'
PLAYERS = ['Mn', 'Cr', 'Ni', 'Al', 'Cu']

def envelope(q):
    return np.array([max(q[t] for t in range(32) if t & m == t) for m in range(32)])

def shapley(v):
    return np.array([sum(factorial(m.bit_count()) * factorial(4-m.bit_count()) / factorial(5)
                         * (v[m | (1 << i)]-v[m]) for m in range(32) if not (m >> i) & 1)
                     for i in range(5)])

def counterexample(frame):
    q = np.zeros(32)
    observed = set()
    for row in frame.itertuples():
        mask = sum(1 << PLAYERS.index(e) for e in row.optional_players.split(';'))
        observed.add(mask)
        q[mask] = max(q[mask], row.nash_geometric_mean)
    b = q.copy(); b[1] = .85
    va, vb = envelope(q), envelope(b)
    aphi, bphi = shapley(va), shapley(vb)
    assert len(observed) == 4 and 1 not in observed
    assert np.array_equal(q[list(observed)], b[list(observed)])
    assert np.array_equal(va[list(observed)], vb[list(observed)])
    for v in (va, vb):
        assert all(v[x] <= v[y]+1e-12 for x in range(32) for y in range(32) if x & y == x)
    assert PLAYERS[int(aphi.argmax())] == 'Cr' and PLAYERS[int(bphi.argmax())] == 'Mn'
    assert np.isclose(aphi.sum(), va[-1]) and np.isclose(bphi.sum(), vb[-1])
    records = [{'mask': m, 'support': ';'.join(e for i,e in enumerate(PLAYERS) if m >> i & 1) or 'empty',
                'observed': m in observed, 'q_A': q[m], 'q_B': b[m], 'v_A': va[m], 'v_B': vb[m]} for m in range(32)]
    pd.DataFrame(records).to_csv(OUT/'counterexample_all_32_supports.csv', index=False)
    pd.DataFrame({'player': PLAYERS, 'phi_A': aphi, 'phi_B': bphi}).to_csv(OUT/'counterexample_shapley.csv', index=False)
    return {'observed_masks': sorted(observed), 'observed_maxima': {str(m):q[m] for m in sorted(observed)},
            'unobserved_Mn_singleton_B': .85, 'leader_A':'Cr', 'leader_B':'Mn',
            'same_observed_scores': True, 'same_observed_coalition_values': True,
            'monotone_access_envelopes': True, 'phi_A':dict(zip(PLAYERS,aphi)), 'phi_B':dict(zip(PLAYERS,bphi)),
            'interpretation':'An illustrative bounded completion witness, not a fitted physical alloy model. No positive or convex game assumption is imposed.'}

def decision_controls(frame, scenario, seed):
    rng=np.random.default_rng(seed)
    u=frame[['u_T_window','u_S','u_R']].to_numpy(float)
    prefs=rng.dirichlet(np.ones(3),size=20000)
    utility=np.exp(prefs @ np.log(np.maximum(u,1e-12)).T)
    oracle=utility.max(axis=1)
    source=frame.source_rows.astype(str).str.split(';').str[0].astype(int).to_numpy()
    scores={'Nash product':np.exp(np.log(np.maximum(u,1e-12)).mean(axis=1)),
            'Equal arithmetic':u.mean(axis=1),'Maximin':u.min(axis=1),
            'Temperature only':u[:,0], 'Entropy only':u[:,1],'Reliability only':u[:,2]}
    orders={k:np.lexsort((source,-v)) for k,v in scores.items()}
    tc=frame.TC_K.to_numpy(); ent=frame.dS_ref_median.to_numpy()
    distance=np.maximum(250-tc,0)+np.maximum(tc-350,0)
    # In-window candidates first, raw entropy descending; outside-window fallback by distance then entropy.
    orders['Window then raw entropy']=np.lexsort((source,-ent,distance))
    rows=[]; selections=[]
    for rule,order in orders.items():
        for budget in (1,3,5):
            chosen=order[:budget]; achieved=utility[:,chosen].max(axis=1); regret=oracle-achieved
            rows.append({'scenario':scenario,'rule':rule,'budget':budget,'mean_regret':regret.mean(),
                         'regret_q95':np.quantile(regret,.95),'within_0p01':np.mean(regret<=.01)})
            for j,idx in enumerate(chosen):
                selections.append({'scenario':scenario,'rule':rule,'budget':budget,'selection_rank':j+1,
                                   'composition':frame.iloc[idx].composition,'source_rows':frame.iloc[idx].source_rows})
    # Exact expected random-subset utility, without simulation over candidate selections.
    sorted_u=np.sort(utility,axis=1); n=len(frame)
    for budget in (1,3,5):
        probabilities=np.array([comb(j,budget-1)/comb(n,budget) if j>=budget-1 else 0 for j in range(n)])
        assert np.isclose(probabilities.sum(),1)
        expected=sorted_u@probabilities
        rows.append({'scenario':scenario,'rule':'Uniform random subset','budget':budget,
                     'mean_regret':np.mean(oracle-expected),'regret_q95':np.nan,'within_0p01':np.nan})
    return rows,selections

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    all_rows=pd.read_csv(ROOT/'results/empirical_nash_shapley_v3/empirical_strategy_portfolio.csv')
    eligible=all_rows[all_rows.game_eligible].reset_index(drop=True)
    counter=counterexample(eligible)
    rows=[]; selections=[]
    for frame,name,seed in [(eligible,'All 17 eligible',2026),(all_rows[all_rows.phase_screen_eligible].reset_index(drop=True),'Phase screened',31415)]:
        r,s=decision_controls(frame,name,seed);rows+=r;selections+=s
    results=pd.DataFrame(rows);results.to_csv(OUT/'decision_controls.csv',index=False)
    pd.DataFrame(selections).to_csv(OUT/'decision_selections.csv',index=False)
    old=pd.read_csv(ROOT/'results/decision_rule_benchmark_v3/preference_robustness_benchmark.csv')
    for scenario,newscenario in [('primary_observed_portfolio','All 17 eligible'),('phase_screened_portfolio','Phase screened')]:
        for a,b in [('nash_equal','Nash product'),('arithmetic_equal','Equal arithmetic'),('maximin','Maximin')]:
            assert np.allclose(old[(old.scenario==scenario)&(old.rule==a)].sort_values('budget').mean_regret,
                               results[(results.scenario==newscenario)&(results.rule==b)].sort_values('budget').mean_regret,atol=1e-13)
    summary={'counterexample':counter,'decision_controls':{'preference_draws':20000,'distribution':'Dirichlet(1,1,1)',
        'seeds':{'all_17':2026,'phase_screened':31415},'oracle':'best measured candidate for each weighted geometric utility',
        'budgets':[1,3,5],'tie_rule':'original source-row order, independent of Nash ranking',
        'random_rule':'exact expectation over uniformly selected size-b subsets without replacement',
        'random_quantiles':'not computed; no misleading quantile of expected regret is reported',
        'limitations':'Preference utility uses the same transformed objectives as the decision scores; this is an internal diagnostic, not external validation.'},
        'tests_passed':True,'external_writes':False}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(results[results.budget==1].to_string(index=False));print('Counterexample and controls: all assertions passed.')

if __name__=='__main__':main()
