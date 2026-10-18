"""Fixed, untuned nonlinear A/B development comparison; no held-out C."""
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import diagnose_models as d

f=d.f
old_ridge=f.Ridge


class Tree:
    def __init__(self, alpha):
        self.model=HistGradientBoostingRegressor(loss='squared_error',learning_rate=.05,
            max_iter=100,max_leaf_nodes=7,min_samples_leaf=50,l2_regularization=1.,
            early_stopping=False,random_state=0)
    def fit(self,x,y):
        self.model.fit(x,y)
        return self
    def predict(self,x):
        return np.maximum(self.model.predict(x),0.)


out=d.BASE/'diagnostics'
summary={}
for name,groups in [('tree_baseline_planned',()),('tree_phase_planned',('phase',)),
                    ('tree_phase_history_planned',('phase','history'))]:
    runs=d.configure(groups)
    f.Ridge=Tree
    detail=d.predict_detail(runs,None,d.design_planned)
    (out/f'{name}-predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in detail))
    scores=[]
    row_lookup={(n,r['t_s']):r for n,run in runs.items() for r in run['rows']}
    for split in sorted({r['split'] for r in detail}):
        scopes=['pooled'] if split=='chronological' else [split.split(':',1)[1]]
        for scope in scopes:
            for h in (1.,2.,5.):
                for target in d.TARGETS:
                    for subset in ('all','near_budget_change'):
                        rows=[r for r in detail if r['split']==split and r['horizon_s']==h and r['target']==target
                              and (scope=='pooled' or r['run']==scope)]
                        if subset!='all':
                            rows=[r for r in rows if f.near_change([row_lookup[r['run'],r['t_s']]],f.label_key(h))[0]]
                        pred=np.array([r['prediction'] for r in rows]);actual=np.array([r['actual'] for r in rows])
                        pers=np.array([f.persistence([row_lookup[r['run'],r['t_s']]],target,h,[1.,5.])[0] for r in rows])
                        mae=float(np.mean(np.abs(pred-actual)));pmae=float(np.mean(np.abs(pers-actual)))
                        scores.append(dict(split=split,scope=scope,horizon_s=h,target=target,subset=subset,n=len(rows),
                                           mae=mae,persistence_mae=pmae,skill=1-mae/pmae))
    summary[name]=scores
    print(name,flush=True)
    for s in scores:
        if s['horizon_s']==5. and s['subset']=='all':print(s,flush=True)
f.Ridge=old_ridge
(out/'nonlinear-summary.json').write_text(json.dumps(dict(development_only=True,C_used=False,
    sklearn_version=__import__('sklearn').__version__,parameters=dict(max_iter=100,max_leaf_nodes=7,
    min_samples_leaf=50,learning_rate=.05,l2_regularization=1,early_stopping=False,random_state=0),results=summary),indent=2)+'\n')
