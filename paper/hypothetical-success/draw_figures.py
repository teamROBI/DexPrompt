"""Draw conceptual figures and explicitly hypothetical result charts; no experiment input."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/dexprompt-paper-mpl')
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Ellipse

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
D = json.loads((ROOT/'illustrative_results.json').read_text())
NAVY='#172b4d'; BLUE='#3267b1'; TEAL='#14877a'; ORANGE='#cb8033'; GRAY='#6f7b8c'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})

def save(fig,name):
    for ext in ('png','pdf','svg'):
        fig.savefig(OUT/f'{name}.{ext}',dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def box(ax,x,y,w,h,label,color=BLUE,fs=10):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.018,rounding_size=0.05',facecolor=color+'12',edgecolor=color,lw=1.3))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=fs,color=NAVY,linespacing=1.45)

def arrow(ax,x,y,u,v,color=GRAY):
    ax.add_patch(FancyArrowPatch((x,y),(u,v),arrowstyle='-|>',mutation_scale=13,lw=1.3,color=color))

fig,ax=plt.subplots(figsize=(12,3.25)); ax.set(xlim=(0,12),ylim=(0,3.3));ax.axis('off')
ax.text(.1,3.05,'TRAINING ONLY  |  exact geometry and Dexonomy knowledge',color=BLUE,weight='bold',fontsize=11)
for x,w,label,col in [(.1,2.0,'Shape + scene\nDexonomy candidates',BLUE),(2.6,2.0,'Learned candidate\nselection',BLUE),(5.1,2.0,'Bounded edits\none-step RL',BLUE),(7.6,2.0,'Contact feedback\nfull execution',TEAL),(10.1,1.75,'Validated\ndemonstrations',TEAL)]:
    box(ax,x,1.9,w,.8,label,col,10)
for x,u in [(2.12,2.58),(4.62,5.08),(7.12,7.58),(9.62,10.08)]:arrow(ax,x,2.3,u,2.3)
ax.plot([.1,11.85],[1.65,1.65],color='#bbc4d1',ls='--',lw=1)
ax.text(.1,1.36,'DEPLOYMENT  |  cameras + request + proprioception',color=TEAL,weight='bold',fontsize=11)
box(ax,.1,.15,2.5,.83,'Mask / point / language\nGrounding + target memory',TEAL,9.5)
box(ax,3.05,.15,2.5,.83,'Head + wrist RGB\nFrozen DINO features',TEAL,10)
box(ax,6.0,.15,2.8,.83,'Visual action transformer\nBC on successful trajectories',TEAL,9.5)
box(ax,9.25,.15,2.6,.83,'27 joint action targets\nReobserve and replan',TEAL,10)
for x,u in [(2.62,3.03),(5.57,5.98),(8.82,9.23)]:arrow(ax,x,.57,u,.57,TEAL)
arrow(ax,10.95,1.88,8.25,.99,TEAL)
ax.text(9.7,1.3,'BC supervision',fontsize=8,color=TEAL,ha='center',bbox={'facecolor':'white','edgecolor':'none','pad':1})
save(fig,'01_pipeline')

fig,axs=plt.subplots(3,1,figsize=(5.7,6.3))
for ax in axs:ax.set(xlim=(0,10),ylim=(0,3));ax.axis('off')
ax=axs[0];ax.text(.1,2.75,'a  Select among complete grasp hypotheses',weight='bold',color=NAVY)
for x,label,score,col in [(1.0,'Pincer','.22',GRAY),(4.1,'Power disk','.81',TEAL),(7.2,'Large diameter','.64',BLUE)]:
    ax.add_patch(Rectangle((x,.7),1.15,1.2,fc='#e8edf2',ec=GRAY,lw=1))
    ax.add_patch(Circle((x-.18,1.3),.13,fc=col));ax.add_patch(Circle((x+1.32,1.3),.13,fc=col))
    if label!='Pincer':
        for yy in (1.0,1.6):ax.add_patch(Circle((x+1.32,yy),.13,fc=col))
    ax.text(x+.55,.34,label,ha='center',fontsize=9)
    ax.text(x+.55,2.18,'score '+score,ha='center',fontsize=9,color=col)
ax.text(9.9,.02,'Illustrative scores',ha='right',fontsize=7,color=GRAY)
ax=axs[1];ax.text(.1,2.75,'b  Edit a bounded reference, then execute',weight='bold',color=NAVY)
x=np.linspace(.6,8.2,100);y=.7+1.25*np.sin((x-.6)/7.6*np.pi/2)
ax.plot(x,y,'--',color=GRAY,lw=1.8,label='Reference')
ax.plot(x,y+.45*np.sin((x-.6)/7.6*np.pi),color=TEAL,lw=2,label='Edited approach')
ax.add_patch(Rectangle((7.8,.1),1.3,1.05,fc='#e8edf2',ec=GRAY));ax.plot([.3,9.5],[.08,.08],color=GRAY)
ax.text(5,.4,'Wrist / posture / timing edits',ha='center',fontsize=9,color=TEAL)
ax.legend(frameon=False,loc='center left',fontsize=8,bbox_to_anchor=(0,.37))
ax=axs[2];ax.text(.1,2.75,'c  Preserve feedback and evaluate the full attempt',weight='bold',color=NAVY)
for x,lab in [(.25,'Approach'),(2.8,'Close'),(5.35,'Lift'),(7.9,'Hold')]:
    box(ax,x,1.35,1.8,.7,lab,TEAL,9)
for x in (2.05,4.6,7.15):arrow(ax,x,1.7,x+.7,1.7)
ax.text(5,.72,'Contact feedback stays active during close / lift / hold',ha='center',fontsize=8.5,color=TEAL)
ax.text(5,.26,'Any failure remains in the execution denominator',ha='center',fontsize=8.5,color=GRAY)
fig.tight_layout(h_pad=.7);save(fig,'02_selection_editing')

fig,axs=plt.subplots(3,1,figsize=(5.8,8.1))
colors=[GRAY,BLUE,ORANGE,TEAL]
ax=axs[0];vals=np.array(D['teacher_success_counts'])/4800*100
ax.barh(D['methods'][::-1],vals[::-1],color=colors[::-1],height=.65)
for i,v in enumerate(vals[::-1]):ax.text(v+1,i,f'{v:.0f}%',va='center',fontsize=10)
ax.set_xlim(0,100);ax.set_xlabel('Accepted executions / all attempts (%)');ax.set_title('a  Demonstration yield at equal execution budget',loc='left',fontsize=10,weight='bold');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
ax=axs[1]
for meth,col in [('Uniform',GRAY),('DexPrompt',TEAL)]:
 ax.plot(D['data_efficiency']['demonstrations'],D['data_efficiency'][meth],'-o',color=col,label=meth,lw=2,ms=5)
ax.set_xscale('log');ax.set_xticks(D['data_efficiency']['demonstrations']);ax.set_xticklabels(['0.5k','2k','6k','12k']);ax.set_ylim(0,100);ax.set_ylabel('Unseen-object success (%)');ax.set_xlabel('Accepted training demonstrations');ax.set_title('b  Visual BC under matched data budgets',loc='left',fontsize=10,weight='bold');ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=9)
ax=axs[2]
for meth,col in [('Uniform',GRAY),('DexPrompt',TEAL)]:
 ax.plot(D['clutter']['distractors'],D['clutter'][meth],'-o',color=col,label=meth,lw=2,ms=5)
ax.set_xticks([0,2,4]);ax.set_ylim(0,100);ax.set_xlabel('Distractor objects');ax.set_ylabel('Unseen-object success (%)');ax.set_title('c  Robustness to increasing scene clutter',loc='left',fontsize=10,weight='bold');ax.grid(alpha=.15)
fig.suptitle('HYPOTHETICAL RESULTS — NOT MEASURED',fontsize=10,color='#9b562c',weight='bold',y=.999)
fig.tight_layout(rect=(0,0,1,.975),h_pad=1.2);save(fig,'03_illustrative_results')

fig,axs=plt.subplots(3,1,figsize=(5.5,6.0))
for ax,target,request in zip(axs,[0,1,2],['Mask: select the bottle','Point: select the carton','Language: "Pick up the box"']):
 ax.set(xlim=(0,10),ylim=(0,3.5));ax.axis('off')
 ax.text(.15,3.13,request,fontsize=11,weight='bold',color=NAVY)
 ax.add_patch(Rectangle((.5,.35),9,2.2,facecolor='#f3f5f8',edgecolor='#cdd5df'))
 ax.add_patch(Rectangle((1.5,.8),1.0,1.15,facecolor='#b9cbe1',edgecolor=BLUE));ax.add_patch(Rectangle((1.78,1.95),.44,.4,fc='#b9cbe1',ec=BLUE))
 ax.add_patch(Rectangle((4.3,.8),1.15,1.65,facecolor='#eed8b3',edgecolor=ORANGE))
 ax.add_patch(Rectangle((7.2,.8),1.6,1.0,facecolor='#d2dadb',edgecolor=GRAY))
 xs=[2.0,4.87,8.0];xx=xs[target]
 ax.add_patch(Ellipse((xx,1.48),2.25,2.13,fc=TEAL+'18',ec=TEAL,lw=1.8))
 arrow(ax,xx+.55,2.88,xx,2.25,TEAL)
 if target==1:ax.plot(xx,1.45,'+',color=TEAL,ms=13,mew=2)
 ax.text(5,.03,'Same scene; only the target request changes.',ha='center',fontsize=8,color=GRAY)
fig.suptitle('CONCEPTUAL SCHEMATIC — NOT A ROBOT ROLLOUT',fontsize=9,color='#9b562c',y=.995)
fig.tight_layout(rect=(0,0,1,.97),h_pad=.75);save(fig,'04_goal_switching')
print('Wrote four figures in PNG, PDF, and SVG.')
