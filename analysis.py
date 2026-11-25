"""Executable analysis. Run from the assignment directory. Narrative is in the notebook."""
from pathlib import Path
import os, sys, json, hashlib, datetime, subprocess, platform, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from IPython.display import display, Markdown
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)
plt.rcParams.update({'figure.dpi':110, 'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
BASE = Path.cwd()
PROTOCOL_PATH = BASE / 'results/20261010T104323297530Z/protocol.json'
assert PROTOCOL_PATH.exists(), 'Use the DS-A6 directory as the working directory.'
protocol = json.loads(PROTOCOL_PATH.read_text())
def notebook_code_hash():
    saved=json.loads((BASE/f"A{protocol['assignment']}.ipynb").read_text())
    code=''.join(''.join(c['source']) for c in saved['cells'] if c['cell_type']=='code')
    return hashlib.sha256(code.encode()).hexdigest()
SEED = protocol['seed']
rng = np.random.default_rng(SEED)
RUN = BASE / 'results' / datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
RUN.mkdir(parents=True)
(RUN/'figures').mkdir()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save_json(name, value):
    (RUN/name).write_text(json.dumps(value, indent=2, default=lambda x: x.item() if hasattr(x,'item') else str(x))+'\n')
def table(name, value):
    value.to_csv(RUN/(name+'.csv'), index=False)
    display(value.head(20))
def figure(name):
    plt.tight_layout(); plt.savefig(RUN/'figures'/(name+'.png'), dpi=150, bbox_inches='tight'); plt.show(); plt.close()
manifest = json.loads((BASE/'data/raw/manifest.json').read_text())
for filename, expected in manifest['files'].items():
    assert sha(BASE/'data/raw'/filename)==expected, filename
save_json('run_metadata.json', {'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'protocol_sha256':sha(PROTOCOL_PATH),'manifest_sha256':sha(BASE/'data/raw/manifest.json'),'seed':SEED,'analysis_source_sha256':sha(BASE/'analysis.py'),'notebook_code_sha256':notebook_code_hash()})
(RUN/'protocol.json').write_bytes(PROTOCOL_PATH.read_bytes())
(RUN/'requirements-lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
print('Verified raw hashes:', len(manifest['files']))
print('Run:', RUN.name)


# 02 · Intervention, structural model and DAG

from generate_data import generate
from matplotlib.patches import FancyArrowPatch
observed=pd.read_csv(BASE/'data/raw/observed.csv'); oracle=pd.read_csv(BASE/'data/raw/oracle.csv')
assert np.allclose(observed,oracle[observed.columns])
assert np.allclose(oracle.Y1-oracle.Y0,2.56+oracle.G)
regenerated=generate(protocol['n'],protocol['confounding_strength'],SEED)
assert np.allclose(regenerated,oracle,rtol=1e-12,atol=1e-12)
truth=float((oracle.Y1-oracle.Y0).mean()); population_truth=3.06
print('Synthetic rows:',len(observed),'Known sample-standardized ATE:',truth,'Population ATE:',population_truth)
positions={'X':(0,3),'G':(0,1),'U':(2,4),'Ypre':(4,4),'A':(2,2),'M':(4,2),'C':(2,0),'Y':(6,2)}
edges=[('X','A'),('X','M'),('X','Y'),('X','Ypre'),('G','A'),('G','Y'),('G','Ypre'),('U','A'),('U','M'),('U','C'),('U','Y'),('U','Ypre'),('A','M'),('A','C'),('A','Y'),('M','Y')]
fig,ax=plt.subplots(figsize=(11,6)); ax.set(xlim=(-.7,6.7),ylim=(-.7,4.7)); ax.axis('off')
labels={'X':'X\nbaseline','G':'G\ngroup','U':'U\nhidden','A':'A\nparticipation','M':'M\nmediator','C':'C\ncollider','Y':'Y\nscore','Ypre':'Ypre\npre-outcome'}
boxes={}
for node,pos in positions.items():
    boxes[node]=ax.text(*pos,labels[node],ha='center',va='center',fontsize=10,zorder=3,bbox={'boxstyle':'round,pad=.55','facecolor':{'U':'#eeeeee','M':'#fff0bd','C':'#ffd6d6','A':'#cce5ff'}.get(node,'white'),'edgecolor':'black','linestyle':'--' if node=='U' else '-'})
fig.canvas.draw()
for src,dst in edges:
    rad={('U','C'):.6,('X','Y'):-.3,('G','Y'):.25,('U','Y'):-.15,('A','Y'):.22}.get((src,dst),-.06)
    arrow=FancyArrowPatch(positions[src],positions[dst],patchA=boxes[src].get_bbox_patch(),patchB=boxes[dst].get_bbox_patch(),arrowstyle='-|>',mutation_scale=15,shrinkA=3,shrinkB=3,connectionstyle=f'arc3,rad={rad}',color='#666666',linestyle='--' if src=='U' else '-',lw=1.3,zorder=2)
    ax.add_patch(arrow)
ax.set_title('Synthetic causal DAG: U is hidden; M and C are post-treatment'); figure('causal_dag')
pd.DataFrame(edges,columns=['source','target']).to_csv(RUN/'dag_edges.csv',index=False)


# 03 · Identification and adjustment sets

assumptions=pd.DataFrame([
 {'assumption':'Consistency and no interference','status':'True by construction in this simulation','implication':'A fixes one specified intervention; records do not affect each other'},
 {'assumption':'Exchangeability given X,G','status':'False for gamma > 0','implication':'Observed adjusted association cannot identify total ATE'},
 {'assumption':'Exchangeability given X,G,U','status':'True by structural design','implication':'Oracle total-effect check, not an available estimator'},
 {'assumption':'Positivity','status':'Logistic propensity strictly between 0 and 1','implication':'Finite-sample near-tail overlap remains a precision concern'},
 {'assumption':'No post-treatment adjustment for total effect','status':'Violated by including M or C','implication':'Mediator changes estimand; collider may introduce bias'},
 {'assumption':'Measurement and timing','status':'Known synthetic generation order','implication':'No evidence about actual training programs or participants'}])
table('assumptions',assumptions)
p=oracle.true_propensity
save_json('overlap.json',{'true_propensity_min':float(p.min()),'true_propensity_max':float(p.max()),'fraction_outside_005_095':float(((p<.05)|(p>.95)).mean()),'oracle_only':True})
plt.figure(figsize=(7,3.5))
for a in [0,1]: plt.hist(p[oracle.A==a],bins=np.linspace(0,1,21),alpha=.5,label=f'A={a}')
plt.xlabel('True propensity (oracle diagnostic)'); plt.ylabel('Records'); plt.legend(); figure('overlap')


# 04 · Estimation, robust intervals and bad-control comparisons

def estimate(data,extra=(),outcome='Y',group_mean=None):
    Z=np.column_stack([np.ones(len(data)),data.A,data.X,data.G,data.A*data.G]+[data[c] for c in extra])
    yy=data[outcome].to_numpy(); beta=np.linalg.lstsq(Z,yy,rcond=None)[0]
    bread=np.linalg.inv(Z.T@Z); h=np.einsum('ij,jk,ik->i',Z,bread,Z); residual=yy-Z@beta
    covariance=bread@(Z.T@((residual/(1-h))[:,None]**2*Z))@bread
    contrast=np.zeros(Z.shape[1]); contrast[1]=1; contrast[4]=data.G.mean() if group_mean is None else group_mean
    value=float(contrast@beta); se=float(np.sqrt(contrast@covariance@contrast))
    return {'estimate':value,'se':se,'lower':value-1.96*se,'upper':value+1.96*se,'n':len(data)}
models={'Unadjusted':None,'Observed_XG':(),'Oracle_XGU':('U',),'Observed_plus_mediator':('M',),'Oracle_plus_mediator':('U','M'),'Observed_plus_collider':('C',)}
rows=[]
for name,extra in models.items():
    if extra is None:
        a=observed.loc[observed.A==1,'Y']; b=observed.loc[observed.A==0,'Y']; value=float(a.mean()-b.mean()); se=float(np.sqrt(a.var()/len(a)+b.var()/len(b))); r={'estimate':value,'se':se,'lower':value-1.96*se,'upper':value+1.96*se,'n':len(observed)}
    else: r=estimate(oracle if 'U' in extra else observed,extra)
    rows.append({'method':name,**r,'difference_from_total_truth':r['estimate']-truth,'interpretation':'oracle direct effect' if name=='Oracle_plus_mediator' else ('identified total effect in simulation' if name=='Oracle_XGU' else 'association / unsuitable total-effect adjustment')})
estimates=pd.DataFrame(rows); table('estimates',estimates)
print('Oracle direct-effect target:',2+observed.G.mean())
fig,ax=plt.subplots(figsize=(9,4)); ax.errorbar(estimates.estimate,np.arange(len(estimates)),xerr=1.96*estimates.se,fmt='o'); ax.set_yticks(np.arange(len(estimates)),estimates.method); ax.axvline(truth,color='black',ls='--',label='Known sample total ATE'); ax.set_xlabel('Score contrast with 95% HC3 interval'); ax.legend(); figure('effect_comparison')
# Independent algebraic cross-check of the unadjusted contrast.
Z=np.column_stack([np.ones(len(observed)),observed.A]); ols=np.linalg.lstsq(Z,observed.Y,rcond=None)[0][1]
assert np.isclose(ols,estimates.loc[estimates.method=='Unadjusted','estimate'].iloc[0])


# 05 · Confounding sensitivity and negative control

replicates=[]
for j,gamma in enumerate(protocol['gamma_grid']):
    for r in range(protocol['simulation_repetitions']):
        seed=np.random.SeedSequence([SEED,900,j,r]); data=generate(protocol['simulation_n'],gamma,seed); actual=float((data.Y1-data.Y0).mean())
        for name,extra in [('Observed_XG',()),('Oracle_XGU',('U',))]:
            fit=estimate(data,extra); replicates.append({'gamma':gamma,'replicate':r,'method':name,'truth':actual,**fit,'bias':fit['estimate']-actual,'covered':fit['lower']<=actual<=fit['upper']})
reps=pd.DataFrame(replicates); reps.to_csv(RUN/'sensitivity_replicates.csv',index=False)
rows=[]
for (gamma,name),g in reps.groupby(['gamma','method']):
    coverage=g.covered.mean(); ci=stats.binomtest(int(g.covered.sum()),len(g)).proportion_ci(method='wilson'); rows.append({'gamma':gamma,'method':name,'repetitions':len(g),'mean_bias':g.bias.mean(),'bias_MCSE':g.bias.std(ddof=1)/np.sqrt(len(g)),'RMSE':np.sqrt(np.mean(g.bias**2)),'coverage':coverage,'coverage_MCSE':np.sqrt(coverage*(1-coverage)/len(g)),'coverage_Wilson_lower':ci.low,'coverage_Wilson_upper':ci.high})
sensitivity=pd.DataFrame(rows); table('sensitivity_summary',sensitivity)
negative=pd.DataFrame([{'method':'Observed_XG',**estimate(observed,outcome='Ypre')},{'method':'Oracle_XGU',**estimate(oracle,('U',),outcome='Ypre')}]); table('negative_control',negative)
fig,ax=plt.subplots(1,2,figsize=(10,3.5))
for name,g in sensitivity.groupby('method'):
    ax[0].errorbar(g.gamma,g.mean_bias,yerr=1.96*g.bias_MCSE,fmt='o-',label=name); ax[1].plot(g.gamma,g.coverage,'o-',label=name)
ax[0].axhline(0,color='grey',ls='--'); ax[0].set(xlabel='Hidden-confounding strength gamma',ylabel='Mean bias ± 1.96 MCSE')
ax[1].axhline(.95,color='grey',ls='--'); ax[1].set(xlabel='Hidden-confounding strength gamma',ylabel='95% interval coverage',ylim=(-.03,1.03)); ax[0].legend(); figure('confounding_sensitivity')


# 06 · Heterogeneity and responsible decision memo

hetero=[]
for group in [0,1]:
    for name,data,extra in [('Observed_XG',observed,()),('Oracle_XGU',oracle,('U',))]:
        fit=estimate(data,extra,group_mean=group)
        hetero.append({'group':group,'method':name,**fit,'true_total_effect':2.56+group,'true_net_benefit_at_cost_2_8':2.56+group-2.8,'group_n':int((observed.G==group).sum()),'treatment_rate':float(observed.loc[observed.G==group,'A'].mean())})
table('heterogeneity',pd.DataFrame(hetero))
memo={'recommendation':'Do not deploy an allocation rule from the observed adjusted associations.','identification':'Hidden U leaves total effect unidentified from observed X,G. Oracle access is simulation-only.','fairness':'Inspect access and outcomes by relevant groups; synthetic G provides no evidence of real demographic performance. Do not turn the simulated group contrast into an exclusion rule.','cost':'2.8 synthetic score-equivalent units is a teaching assumption, not an observed economic utility.','appeal':'Any prospective pilot needs a documented human reconsideration route, explanation of decisions and correction of data errors.','stop_conditions':['evidence of harm or unequal access','poor overlap or unstable subgroup estimates','negative-control evidence inconsistent with identification assumptions','intervention differs from the stated treatment'],'next_evidence':'An ethically reviewed randomized or otherwise credibly identified prospective study, with explicit consent and monitoring.'}
save_json('responsible_decision_memo.json',memo)
obs=estimates[estimates.method=='Observed_XG'].iloc[0]; ora=estimates[estimates.method=='Oracle_XGU'].iloc[0]; cov=sensitivity[(sensitivity.gamma==1)&(sensitivity.method=='Observed_XG')].iloc[0]; nc=negative.iloc[0]
facts={'synthetic':True,'n':len(observed),'population_ATE':population_truth,'sample_ATE':truth,'observed_estimate':float(obs.estimate),'oracle_estimate':float(ora.estimate),'observed_coverage_gamma_1':float(cov.coverage),'negative_control_observed':float(nc.estimate),'identification':'Observed association only'}
summary=f"""This study uses an explicitly synthetic structural causal model to examine a training-participation claim. The intervention sets participation to one versus zero, and the estimand is the total population average effect on a synthetic score. The causal graph includes observed baseline variables, hidden ability, a mediator, and a post-treatment collider. The known population effect is {population_truth:.2f} points; the {len(observed)}-record sample has standardized truth {truth:.3f}.

Adjustment for observed baseline variables estimates {obs.estimate:.3f}, while an oracle regression that also observes hidden ability estimates {ora.estimate:.3f}. The first result is an adjusted association, because hidden confounding prevents identification from the observed variables. Mediator and collider adjustments illustrate different failures of an indiscriminate control strategy. A pre-treatment negative-control outcome has an observed adjusted participation contrast of {nc.estimate:.3f}, despite having no causal participation effect.

Two hundred independent replications at each of four confounding strengths retain bias, robust-interval coverage, and Monte Carlo uncertainty. At confounding strength one, observed-adjustment coverage is {cov.coverage:.1%}; the oracle provides a simulation benchmark rather than a deployable remedy. Group-specific effects and hypothetical costs demonstrate that an aggregate benefit does not settle allocation or fairness. The decision memo rejects automated allocation from unidentified associations and specifies equitable evaluation, human appeal, and stop conditions. These results concern the stated data-generating model only: there are no real participants, measured economic benefits, or empirical claims about actual training programs."""

zero=sensitivity[(sensitivity.gamma==0)&(sensitivity.method=='Observed_XG')].iloc[0]
ora_cov=sensitivity[(sensitivity.gamma==1)&(sensitivity.method=='Oracle_XGU')].iloc[0]
notes=f"""## Interpretation of this run

The observed adjustment overstates the known sample total effect by {obs.estimate-truth:.3f} score points. HC3 standard errors do not remove that identification error. At gamma zero, observed-adjustment coverage is {zero.coverage:.1%}; at gamma one it is {cov.coverage:.1%}, versus {ora_cov.coverage:.1%} for the oracle. Removing the hidden treatment-selection path restores the observed adjustment in the designed refutation setting.

Zero observed coverage in 200 replications is not an exact zero population probability: its Wilson upper confidence bound is {cov.coverage_Wilson_upper:.1%}. The negative-control association supplies a warning under this specified model, not a universal diagnostic test. The mediator-controlled oracle targets a direct effect, so comparing it as if it estimated the total effect would confuse two estimands. Group heterogeneity and cost assumptions do not solve the missing identification problem.
"""

(RUN/'DISCUSSION.md').write_text(notes)
display(Markdown(notes))



assert 150 <= len(summary.split()) <= 300, len(summary.split())
(RUN/'ABSTRACT.md').write_text('# Abstract\n\n'+summary+'\n')
save_json('summary.json', facts)
save_json('completed.json', {'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':sha(PROTOCOL_PATH),'manifest_sha256':sha(BASE/'data/raw/manifest.json'),'summary_words':len(summary.split()),'analysis_source_sha256':sha(BASE/'analysis.py'),'notebook_code_sha256':notebook_code_hash(),'personal_verification':'Not asserted by automated execution'})
display(Markdown('## Results summary\n\n'+summary))
print('RESULT_DIR:', RUN.relative_to(BASE).as_posix())
