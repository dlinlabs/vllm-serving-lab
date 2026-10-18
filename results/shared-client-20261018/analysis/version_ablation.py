"""Check whether administrative version counters cause temporal extrapolation."""
import json
import diagnose_models as d

out=d.BASE/'diagnostics'
all_scores={}
for name,groups in [('no_version_planned',()),('phase_no_version_planned',('phase',)),
                    ('phase_history_no_version_planned',('phase','history'))]:
    runs=d.configure(groups)
    for run in runs.values():
        for row in run['rows']:
            row['features'].pop('budget_version')
    d.f.design=d.design_planned
    result=d.f.evaluate(runs,horizons=[1.,2.,5.],windows=[1.,5.],fraction=2/3,alpha=1.)
    (out/f'{name}.json').write_text(json.dumps(result,indent=2)+'\n')
    detail=d.predict_detail(runs,result,d.design_planned)
    (out/f'{name}-predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in detail))
    wanted=[r for r in result['results'] if r['target'] in d.TARGETS and r['subset'] in ('all','near_budget_change')
            and (r['split']!='chronological' or r['scope']=='pooled')]
    all_scores[name]=wanted
    print(name,flush=True)
    for r in wanted:
        if r['horizon_s']==5. and r['subset']=='all':print(r['split'],r['target'],round(r['models']['ridge']['mae'],3),round(r['skill_mae']['ridge_vs_persistence'],3),flush=True)
d.f.design=d.original_design
(out/'version-ablation-summary.json').write_text(json.dumps(dict(development_only=True,C_used=False,results=all_scores),indent=2)+'\n')
