"""Synthetic sensitivity of fixed phase totals versus the historical max-step CV.

This checks a repeatability screen, not model-selection power or GPU acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prefill_lab.contracts import save
from scripts.build_prefill_design import cases
from prefill_planning import rollout
from prefill_design import COEF


def screen(draws=20000,seed=20261023):
    weights=[];groups=[]
    for case in cases():
        if case['split']!='calibration':continue
        result,steps=rollout(case,[*COEF,1e-7,3e-9])
        start,end=result['release_s'],result['scored_step_window']['end_s']
        selected=[s for s in steps if s['start']>=start and s['end']<=end]
        for phase in ('prefill','decode'):
            indices=[]
            for s in selected:
                if (s['prefill']>0)!=(phase=='prefill'):continue
                indices.append(len(weights));weights.append(s['end']-s['start'])
            groups.append(dict(id=case['id']+'/'+phase,indices=indices))
    weights=np.array(weights);rng=np.random.default_rng(seed);rows=[]
    for profile in ('independent_step','shared_phase_trial'):
        for cv in (.01,.02,.05,.08):
            old_stop=new_stop=0
            sigma=np.sqrt(np.log1p(cv*cv))
            for offset in range(0,draws,500):
                count=min(500,draws-offset)
                if profile=='independent_step':
                    values=np.exp(rng.normal(-sigma*sigma/2,sigma,(count,3,len(weights))))*weights
                else:
                    scales=np.exp(rng.normal(-sigma*sigma/2,sigma,(count,3,len(groups))))
                    values=np.empty((count,3,len(weights)))
                    for i,g in enumerate(groups):values[:,:,g['indices']]=scales[:,:,i,None]*weights[g['indices']]
                old_cv=values.std(axis=1,ddof=1)/values.mean(axis=1)
                totals=np.stack([values[:,:,g['indices']].sum(axis=2) for g in groups],axis=2)
                new_cv=totals.std(axis=1,ddof=1)/totals.mean(axis=1)
                old_stop+=int((old_cv.max(axis=1)>.05).sum())
                new_stop+=int((new_cv.max(axis=1)>.05).sum())
            rows.append(dict(profile=profile,input_cv=cv,draws=draws,historical_max_step_stop=old_stop,
                             phase_total_stop=new_stop,historical_rate=old_stop/draws,phase_total_rate=new_stop/draws))
    files=['prefill_lab/analysis.py','scripts/check_prefill_phase_readiness.py','prefill_grouped_validation.py','prefill_planning.py','scripts/build_prefill_design.py']
    return dict(schema='prefill-fixed-phase-readiness-sensitivity-v1',seed=seed,strata=[dict(id=g['id'],step_count=len(g['indices'])) for g in groups],
                total_step_strata=len(weights),rows=rows,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
                interpretation='Synthetic lognormal labels, three independent repeats. Phase totals reduce multiplicity and average independent step noise; shared phase noise survives aggregation. Neither screen bounds population CV or establishes model power.',
                gpu_executed=False,experiment_accepted=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--draws',type=int,default=20000);args=parser.parse_args()
    if args.draws<=0:parser.error('draws must be positive')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    save(args.output,screen(args.draws))
