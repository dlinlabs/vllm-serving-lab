"""CPU service-calibration contract: demo, plan, freeze and held scoring."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prefill_lab.async_calibration import CHANNELS,plan,freeze,evaluate
from prefill_lab.contracts import digest,save
import numpy as np


def synthetic(p):
    rng=np.random.default_rng(32);rows=[]
    truth={c:{k:(j+1)*1e-9 for j,k in enumerate(names)} for c,names in p['laws'].items()}
    # Known input design includes endpoints in training and interior held points.
    for tid,role in p['trials'].items():
        for i in range(32):
            vals=rng.uniform(.2,.8,6)
            if role=='train' and i<2:vals[:]=i
            f=dict(zip(('prefill','decode','decode_context','attention_pairs','padding_excess','num_requests'),
                       map(float,vals*np.array([2048,32,100000,10000000,128,32]))))
            labels={c:dict(scope='exclusive_service_s',seconds=sum(v*(1 if k=='bias' else f[k]) for k,v in cs.items())) for c,cs in truth.items()}
            rows.append(dict(trial_id=tid,step_id=str(i),mode='NONE',features=f,labels=labels))
    return dict(schema='async-service-labels-v1',origin='synthetic',plan_sha256=digest(p),runtime_sha256='0'*64,rows=rows),truth


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('operation',choices=['plan','demo','freeze','evaluate'])
    ap.add_argument('--input',type=Path);ap.add_argument('--plan',type=Path);ap.add_argument('--frozen',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(argv)
    if a.output.exists():ap.error('Output directory must be fresh')
    p=json.loads(a.plan.read_text()) if a.plan else plan()
    if a.operation=='plan':result=p
    elif a.operation=='demo':
        data,truth=synthetic(p);f=freeze(data,p);held=evaluate(data,f)
        recovered={**{s+'.'+k:v for s,parts in f['costs']['launches'].items() for k,v in parts.items()},**f['costs']['scalar']}
        error=max(abs(v-recovered[c][k]) for c,cs in truth.items() for k,v in cs.items())
        result=dict(max_coefficient_error=error,held=held,model_accepted=False)
    else:
        if a.input is None:ap.error('--input is required')
        data=json.loads(a.input.read_text())
        if a.operation=='freeze':f=freeze(data,p);result=f
        else:
            if a.frozen is None:ap.error('--frozen is required')
            result=evaluate(data,json.loads(a.frozen.read_text()))
    a.output.mkdir(parents=True)
    save(a.output/'result.json',result)
    if a.operation in ('demo','freeze'):
        save(a.output/'frozen.json',f);save(a.output/'costs.json',f['costs'])
    if a.operation=='demo':save(a.output/'synthetic-input.json',data);save(a.output/'plan.json',p)
    print('CPU calibration completed; native collection ready: False; model accepted: False')
    return 0

if __name__=='__main__':raise SystemExit(main())
