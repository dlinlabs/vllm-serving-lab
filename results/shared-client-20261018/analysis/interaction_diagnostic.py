"""One fixed state-by-planned-cap interaction candidate, development A/B only."""
import json
import numpy as np
import diagnose_models as d

runs=d.configure(('phase',))
for run in runs.values():
    for row in run['rows']:row['features'].pop('budget_version')


def interactions(rows,names,key):
    x=d.design_planned(rows,names,key)
    # The proposed cap is a control input, not a future observed event.
    u=np.array([r['planned'][key]['budget_mean']/96. for r in rows])
    state=np.array([[r['phase']['phase_pending_input_tokens']/1000.,
                     r['phase']['phase_streaming_reserved_cost'],
                     r['phase']['phase_streaming_age_mean'],
                     r['features']['budget_headroom']] for r in rows])
    return np.column_stack([x,u[:,None]*state,u*u])


detail=d.predict_detail(runs,None,interactions)
out=d.BASE/'diagnostics'
(out/'interactions_planned-predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in detail))
lookup={(name,row['t_s']):row for name,item in runs.items() for row in item['rows']}
scores=[]
for split in sorted({r['split'] for r in detail}):
    scope='pooled' if split=='chronological' else split.split(':',1)[1]
    for h in (1.,2.,5.):
        for target in d.TARGETS:
            for subset in ('all','near_budget_change'):
                rr=[r for r in detail if r['split']==split and r['horizon_s']==h and r['target']==target]
                if subset!='all':rr=[r for r in rr if d.f.near_change([lookup[r['run'],r['t_s']]],d.f.label_key(h))[0]]
                pred=np.array([r['prediction'] for r in rr]);actual=np.array([r['actual'] for r in rr])
                pers=np.array([d.f.persistence([lookup[r['run'],r['t_s']]],target,h,[1.,5.])[0] for r in rr])
                mae=float(np.mean(np.abs(pred-actual)));pmae=float(np.mean(np.abs(pers-actual)))
                scores.append(dict(split=split,scope=scope,horizon_s=h,target=target,subset=subset,n=len(rr),mae=mae,
                    persistence_mae=pmae,skill=1-mae/pmae,
                    optimistic_p99=d.f.metrics(pred,actual,d.f.TARGETS[target])['optimistic_p99']))
result=dict(development_only=True,C_used=False,planned_delay_assumption_s=.05,alpha=1.,
            added_basis=['cap_mean*pending_input_tokens','cap_mean*streaming_cost','cap_mean*streaming_age_mean',
                         'cap_mean*headroom','cap_mean_squared'],results=scores)
(out/'interactions-summary.json').write_text(json.dumps(result,indent=2)+'\n')
for s in scores:
    if s['horizon_s']==5. and s['subset']=='all':print(s,flush=True)
