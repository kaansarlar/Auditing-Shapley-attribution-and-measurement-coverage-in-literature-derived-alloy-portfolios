import pandas as pd,numpy as np,json
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_predict,StratifiedKFold
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
from sklearn.inspection import permutation_importance
d=pd.concat([pd.read_csv('res_0_2000.csv'),pd.read_csv('res_2000_4000.csv')],ignore_index=True); d.to_csv('bench_all.csv',index=False)
d['cls']=np.select([(d.veto==0)&(d.equiv_pairs==0)&(d.rank_main==6),(d.veto==1)|(d.equiv_pairs>0)],['full','veto_equiv'],'rankdef')
d['cls4']=np.select([(d.veto==1)&(d.equiv_pairs>0),d.veto==1,d.equiv_pairs>0],['VE','V','E'],'N')
def row(m):
    c=m[m.pub_boot_top_freq>=0.9]; g=m[m.grp_boot_top_freq>=0.9]
    return dict(n=len(m),mat=m.material_error.mean(),rand=m.random_material_error.mean(),tied=(m.top_hat<0).mean(),pubconf=(m.pub_boot_top_freq>=0.9).mean(),mat_pubconf=c.material_error.mean() if len(c) else np.nan,n_pubconf=len(c),grpconf=(m.grp_boot_top_freq>=0.9).mean(),mat_grpconf=g.material_error.mean() if len(g) else np.nan,rho=m.rho.median(),rho_port=m.rho_port.median(),rho_struct=m.rho_struct.median(),top_port_correct=m.top_port_correct.mean(),regret_world=m.regret_world.mean(),regret_port=m.regret_portfolio.mean())
F={}
F['classes']={c:row(d[d.cls==c]) for c in ['full','rankdef','veto_equiv']}; F['classes']['all']=row(d)
F['classes']['K_le4']=row(d[d.K_obs<=4]); F['classes']['hea_like_bench']=row(d[(d.K_obs==4)&(d.veto==1)&(d.equiv_pairs>=1)])
F['classes']['nontied']=row(d[d.top_hat>=0]); F['classes']['contrasts5']=row(d[d.players_with_contrast==5])
F['cls4']={c:row(d[d.cls4==c]) for c in ['N','E','V','VE']}
F['by_rank']={int(k):row(g) for k,g in d.groupby('rank_main')}
F['by_N']={int(k):row(g) for k,g in d.groupby('N')}
F['by_contrast']={int(k):row(g) for k,g in d.groupby('players_with_contrast')}
F['by_lam']={str(k):row(g) for k,g in d.groupby('lam')}
F['by_coverage']={str(k):row(g) for k,g in d.groupby(pd.cut(d.K_obs,[0,4,8,16,32]),observed=True)}
F['true_top_tied']=float((d.top_true<0).mean())
F['err_given_pubconf']=float(d[d.pub_boot_top_freq>=0.9].material_error.mean()); F['err_given_notconf']=float(d[d.pub_boot_top_freq<0.9].material_error.mean())
# clustering bootstrap
x2=d[(d.pubs_per_group>1)&(d.sigma_grp>=0.6)]; rows=[]
for g in [0,0.5,0.9]:
    s=x2[x2.group_support_share==g]; pc=s.pub_boot_top_freq>=0.9; gc=s.grp_boot_top_freq>=0.9
    rows.append(dict(gss=g,n=len(s),pub_conf=pc.mean(),pub_dis=1-s[pc].top_port_correct.mean(),grp_conf=gc.mean(),grp_dis=1-s[gc].top_port_correct.mean()))
F['clustering']=rows
# classifiers
STRUCT=['N','K_obs','coverage','rank_main','rank_pair','veto','equiv_pairs','contrasts','players_with_contrast','eff_groups','top_group_share','n_groups']
BOOT=STRUCT+['pub_boot_top_freq','grp_boot_top_freq']
y=d.material_error.values
tr,te=train_test_split(np.arange(len(d)),test_size=0.3,random_state=20261006,stratify=y)
hea=json.load(open('hea_real_diagnostics.json'))['diagnostics']
hx={**hea,'N':17,'pub_boot_top_freq':0.596,'grp_boot_top_freq':0.4435}
C={}; PRED=pd.DataFrame({'seed':d.loc[te,'seed'].values,'y':y[te],'pub_boot':d.loc[te,'pub_boot_top_freq'].values})
for name,cols in [('structural',STRUCT),('structural+bootstrap',BOOT)]:
    rf=RandomForestClassifier(n_estimators=500,min_samples_leaf=12,class_weight='balanced',random_state=20261006,n_jobs=2).fit(d.loc[tr,cols],y[tr])
    p=rf.predict_proba(d.loc[te,cols])[:,1]
    imp=permutation_importance(rf,d.loc[te,cols],y[te],n_repeats=20,random_state=0,scoring='roc_auc',n_jobs=2)
    lg=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000)).fit(d.loc[tr,cols],y[tr])
    PRED[name]=p
    hp=float(rf.predict_proba(pd.DataFrame([{c:hx[c] for c in cols}]))[0,1])
    C[name]=dict(rf_auc=roc_auc_score(y[te],p),rf_bacc=balanced_accuracy_score(y[te],(p>=0.5).astype(int)),logit_auc=roc_auc_score(y[te],lg.predict_proba(d.loc[te,cols])[:,1]),
                 hea_prob=hp,importance=dict(sorted(zip(cols,imp.importances_mean.round(4)),key=lambda t:-t[1])[:6]))
C['pub_boot_only_auc']=roc_auc_score(y[te],1-d.loc[te,'pub_boot_top_freq']); C['n_train']=len(tr); C['n_test']=len(te); C['base_rate_test']=float(y[te].mean())
F['classifier']=C; PRED.to_csv('heldout_predictions.csv',index=False)
json.dump(F,open('final_numbers.json','w'),indent=1,default=float)
f=lambda r:{k:(round(v,3) if isinstance(v,float) else v) for k,v in r.items()}
for k in ['classes','cls4','by_rank','by_lam','by_coverage']:
    for kk,v in F[k].items(): print(k,kk,f(v))
print(F['true_top_tied'],F['err_given_pubconf'],F['err_given_notconf']); print(rows); print(json.dumps(C,indent=1,default=float))
