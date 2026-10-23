import numpy as np, pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]/'code'
G={}; exec((ROOT/'bench.py').read_text().split("if __name__")[0],G)
SH,W,BITS,SUB,obs_values,diagnostics,scenario=(G[k] for k in ['SH','W','BITS','SUB','obs_values','diagnostics','scenario'])
def game_from_supports(sup,q): return obs_values(np.asarray(q,float),np.asarray(sup))
def test_efficiency():
    rng=np.random.default_rng(1); q=rng.random(32); q[0]=0
    V=np.array([q[SUB[s]].max() for s in range(32)]); assert np.isclose(SH(V).sum(),V[31]-V[0])
def test_equivalent_players_tie():
    sup=[0b00011,0b00111,0b11011]   # players 0 and 1 always together
    phi=SH(game_from_supports(sup,[0.3,0.8,0.5])); assert phi[0]==phi[1]
def test_veto_lower_bound():
    sup=[0b00010,0b00110,0b01010]   # player 1 in every support
    V=game_from_supports(sup,[0.2,0.9,0.4]); phi=SH(V)
    assert all(V[s]==0 for s in range(32) if not (s>>1)&1); assert phi[1]>=V[31]/5-1e-12 and phi.argmax()==1
def test_full_coverage_recovery():
    r=scenario(90000,dict(N=1500,K=32,se=0,sp=0,sg=0,sd_alloy=0,alpha=0,gss=0,lam=0,veto=None,tie=None))
    assert r['K_obs']==32 and np.isclose(r['rho'],1.0)
def test_case_study_diagnostics():
    pl=['Mn','Cr','Ni','Al','Cu']; S=[['Cr','Ni'],['Cr','Ni','Cu'],['Mn','Cr','Al'],['Mn','Cr','Ni','Al']]
    sup=np.array([sum(1<<pl.index(e) for e in s) for s in S]); d=diagnostics(sup,np.array([0,0,1,2]))
    assert (d['K_obs'],d['rank_main'],d['veto'],d['equiv_pairs'],d['players_with_contrast'])==(4,4,1,1,2)
