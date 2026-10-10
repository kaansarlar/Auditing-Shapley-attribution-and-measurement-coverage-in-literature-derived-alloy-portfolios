"""Re-render the moving-window figure from frozen numeric results without annotation overlap."""
from pathlib import Path
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
s=pd.read_csv(ROOT/'results/temperature_window_sensitivity_v3/window_shapley.csv')
w=pd.read_csv(ROOT/'results/temperature_window_sensitivity_v3/window_summary.csv')
fig,ax=plt.subplots(1,2,figsize=(11.5,4.7))
colors={'Cr':'#a34343','Mn':'#bd6a12','Ni':'#7552a2','Cu':'#647084'}
for p,label in [('Cr','Cr (veto)'),('Mn','Mn = Al'),('Ni','Ni'),('Cu','Cu')]:
 d=s[s.player==p];ax[0].plot(d.center_K,d.shapley,'o-',color=colors[p],label=label,ms=4,lw=1.8)
ax[0].axvline(300,ls='--',lw=1.2,color='#374151')
ax[0].set(xlabel='Target-window centre (K)',ylabel='Coalition Shapley value',ylim=(0,.455))
ax[0].legend(loc='upper center',bbox_to_anchor=(.5,1.20),ncol=4,frameon=False,fontsize=8.5)
ax[0].text(300,.017,'Primary window',rotation=90,ha='right',va='bottom',fontsize=8,color='#374151')
for row in w.itertuples():
 for num in re.findall(r'\d+',str(row.cooptimal_references)):
  ax[1].scatter(row.center_K,int(num),s=48,facecolors='none',edgecolors='#255f8f',linewidths=1.5)
 num=int(re.findall(r'\d+',str(row.phase_priority_reference))[0])
 ax[1].scatter(row.center_K,num,s=65,marker='*',color='#bd6a12',zorder=3)
ax[1].scatter([],[],s=48,facecolors='none',edgecolors='#255f8f',label='Nash co-optimum')
ax[1].scatter([],[],s=65,marker='*',color='#bd6a12',label='Phase priority')
ax[1].axvline(300,ls='--',lw=1.2,color='#374151')
ax[1].set(xlabel='Target-window centre (K)',ylabel='Selected source',yticks=[18,30,32,36],yticklabels=['S18','S30','S32','S36'],ylim=(16.5,37.5))
ax[1].invert_yaxis();ax[1].legend(loc='upper center',bbox_to_anchor=(.5,1.20),ncol=2,frameon=False,fontsize=8.5)
for i,a in enumerate(ax):
 a.grid(alpha=.2);a.set_axisbelow(True);a.set_xlim(140,410)
 a.spines[['top','right']].set_visible(False)
 a.text(-.12,1.13,chr(97+i),transform=a.transAxes,fontweight='bold',fontsize=13)
fig.subplots_adjust(top=.79,bottom=.13,wspace=.28)
for ext in ['png','pdf','svg']:
 fig.savefig(ROOT/'figures'/f'Figure_S2_window_sensitivity_v7_final.{ext}',dpi=240,bbox_inches='tight')
