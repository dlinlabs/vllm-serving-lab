"""Frozen-coefficient development sensitivity: B=8192 versus B=2048. No GPU inference."""
import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import evaluate_low_cap_frozen as lf
import evaluate_greybox as ev
import greybox_simulator as gb
import diagnose_cap_misses as dc
from prefill_design import simulate


def quartile_diagnostic(captures,budget,coef):
    """Observed admitted arrivals are forced; this is not an online or cap-selection test."""
    rows=[]
    for c in captures:
        if c['definition']['schedule']!='fixed-32':continue
        req=c['req'];lookup={r['trace_id']:r['input_tokens'] for r in lf.read(c['path']/'trace.json')['requests']}
        tok=np.array([lookup[t] for t in req['trace_id']],float)
        recent=dc.recent_tokens(req['acc'],tok)
        requests=[dict(id=rid,tokens=int(t),at=float(a),output=128) for rid,t,a in
                  zip(req['request_id'],tok,req['acc']) if np.isfinite(a)]
        predicted=simulate(requests,budget)['foreground']
        for i,rid in enumerate(req['request_id']):
            if req['admitted'][i] and req['success'][i] and 60<=req['sched'][i]<300 and tok[i]<2000:
                rows.append(dict(recent=recent[i],actual_miss=not bool(req['met'][i]),
                    simulated_miss=predicted[rid]['first_s']-req['sched'][i]>.5))
    cuts=np.quantile([r['recent'] for r in rows],[.25,.5,.75])
    edges=[-np.inf,*cuts,np.inf];out=[]
    for lo,hi in zip(edges,edges[1:]):
        v=[r for r in rows if lo<r['recent']<=hi]
        out.append(dict(n=len(v),actual_miss=float(np.mean([r['actual_miss'] for r in v])),
                        simulated_miss=float(np.mean([r['simulated_miss'] for r in v]))))
    return dict(scope='new-host fixed32; successful actual-admitted short requests; forced observed acceptance times; raw serial service only',
                recent_token_cuts=cuts.tolist(),quartiles=out,
                warning='No fitted TTFT correction, no admission model, full prompt work, empty initial state; not a causal prediction or an exclusive scheduler test')


def itl_tail(path):
    start=lf.read(path/'manifest.json')['measurement_start'];samples=[]
    for line in (path/'engine_observations.jsonl').read_text().splitlines():
        e=json.loads(line);t=e['sent_at']-start;raw=e.get('raw_metrics') or ''
        if not 60<=t<=300 or e.get('error_type') or not raw:continue
        counts={}
        for labels,value in re.findall(r'^vllm:inter_token_latency_seconds_bucket\{([^}]+)\}\s+(\S+)',raw,re.M):
            match=re.search(r'(?:^|,)\s*le="([^"]+)"',labels)
            if match:counts[float(match[1])]=counts.get(float(match[1]),0.)+float(value)
        if counts:samples.append((t,counts))
    if len(samples)<2:raise ValueError('Insufficient ITL samples')
    a,b=samples[0],samples[-1];delta={k:b[1][k]-a[1][k] for k in (.1,.15,.2,float('inf'))}
    if min(delta.values())<0:raise ValueError('ITL reset across comparison')
    return dict(capture=path.name,start_sample_s=a[0],end_sample_s=b[0],count=delta[float('inf')],
        gap_100_150ms=delta[.15]-delta[.1],gap_150_200ms=delta[.2]-delta[.15],gap_above200ms=delta[float('inf')]-delta[.2],
        metrics_sha256=lf.sha(path/'engine_observations.jsonl'),interpretation='Exporter ITL gaps, not measured GPU step duration or a config identity test')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite-dir',required=True,type=Path);p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--old-suite-dir',type=Path,help='Optional verified cap-range bundle, for the three historical ITL comparisons')
    a=p.parse_args()
    if a.output_dir.exists():p.error('Choose fresh output')
    guard=['greybox_simulator.py','evaluate_greybox.py','evaluate_low_cap_frozen.py','prefill_design.py','scripts/check_prefill_budget.py']
    before={f:ev.code_sha256(ROOT/f) for f in guard}
    captures,audit=lf.audit_suite(a.suite_dir);coef,overhead,reference,_=lf.frozen_parameters()
    from prefill_design import COEF
    assert tuple(coef)==tuple(COEF),'Scenario and frozen coefficients differ'
    results={};original=gb.B
    try:
        for budget in [8192,2048]:
            gb.B=budget
            print('Scoring budget',budget,flush=True)
            result=lf.evaluate(captures,coef,overhead,reference)
            result['forced_admission_quartiles']=quartile_diagnostic(captures,budget,coef)
            results[str(budget)]=result
    finally:gb.B=original
    old=json.loads(gzip.decompress((ROOT/'results/low-cap-frozen-20261021/full-report.json.gz').read_bytes()))
    assert all(abs(a['simulator']-b['simulator'])<1e-10 for a,b in zip(results['8192']['predictions'],old['predictions']))
    assert before=={f:ev.code_sha256(ROOT/f) for f in guard}
    a.output_dir.mkdir(parents=True)
    tail=[itl_tail(c['path']) for c in captures if c['definition']['schedule'] in ['fixed-16','fixed-32']]
    if a.old_suite_dir:
        tail += [itl_tail(a.old_suite_dir/name) for name in ['cap-range-v1-fixed-32-r1','cap-range-v1-fixed-32-r2','cap-range-v1-fixed-16-r1']]
    full=dict(role='development sensitivity; no identified old scheduler configuration',audit=audit,code_sha256=before,itl_tail=tail,
        learned_service_and_overhead_refitted=False,online_bias_rule='unchanged; recomputed only from fully observed historical windows per B',results=results)
    raw=json.dumps(ev.clean(full),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    (a.output_dir/'full-report.json.gz').write_bytes(gzip.compress(raw,mtime=0))
    summary=dict(role=full['role'],audit=audit,code_sha256=before,itl_tail=tail,full_report_sha256=hashlib.sha256(raw).hexdigest(),
        results={k:{x:y for x,y in v.items() if x not in ['predictions','oracle_diagnostics','factual_phase_gaps']} for k,v in results.items()})
    (a.output_dir/'summary.json').write_text(json.dumps(ev.clean(summary),indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:dict(mae=v['scores']['simulator']['mae'],cap32_bias=v['subsets']['cap32']['simulator']['bias']) for k,v in results.items()},indent=2))


if __name__=='__main__':main()
