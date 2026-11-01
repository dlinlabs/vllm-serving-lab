"""Queue controls must see phase changes hidden by stable TTFT/completion."""
import copy
import hashlib
import json
from types import SimpleNamespace as NS

import pytest
from prefill_lab import delayed_calibration as delayed
from prefill_lab.contracts import save
from scripts import analyze_prefill_queue_phase as analysis


def evidence():
    p=delayed.protocol();name='mixed-plus-sentinel-at400-D16'
    p=dict(p,cases=[c for c in p['cases'] if c['id']==name],order=[t for t in p['order'] if t['case']==name and t['stage'] in delayed.HELD])
    endpoints={k:dict(ttft_s=1.,completion_s=3.) for k,v in delayed.identities(p['cases'][0]).items() if v[2]!='background'}
    rows=[dict(**{k:t[k] for k in ('trial_id','case','stage','block','observer')},endpoints=copy.deepcopy(endpoints),
        work_signature='same-work',clocks={'sentinel':dict(queue_residence_s=.05,posting_lag_s=0.,posted_ttft_s=1.,target_ttft_s=1.)}) for t in p['order']]
    forecast={name:dict(endpoints=endpoints,work_signature='same-work',background_outlives_foreground=True,sentinel_queue_residence_s=.05)}
    return p,rows,forecast


@pytest.mark.parametrize('problem',[None,'observer','repeat','negative','nan','duplicate'])
def test_stable_endpoints_cannot_hide_unqualified_queue_residence(problem):
    p,rows,forecast=evidence();off=[r for r in rows if r['observer']=='off']
    if problem=='observer':
        for r in rows:
            if r['observer']=='light':r['clocks']['sentinel']['queue_residence_s']=.10
    if problem=='repeat':
        for r in rows:r['clocks']['sentinel']['queue_residence_s']=.02 if r['block']%2 else .08
    if problem in ('negative','nan'):off[0]['clocks']['sentinel']['queue_residence_s']=-1. if problem=='negative' else float('nan')
    if problem=='duplicate':rows.append(copy.deepcopy(off[0]))
    if problem in ('negative','nan','duplicate'):
        with pytest.raises(ValueError):delayed.queue_quality(rows,p)
        return
    q=delayed.quality(rows,p);report=delayed.score(rows,p,forecast,'frozen')
    assert all(r['passed'] for r in q['paired_observer']+q['repeat_spread'])
    assert all(r['passed'] for r in report['sentinel_queue_scores'])  # Median prediction is exact even in the bad cases.
    assert report['candidate_passed']==(problem is None)
    assert q['queue_residence']['passed']==(problem is None)
    if problem:
        reasons=delayed.rejection_reasons(report)
        assert any(r['gate']=='queue_'+('paired_observer' if problem=='observer' else 'repeat_spread') for r in reasons)


@pytest.mark.parametrize('problem',[None,'commit','scope','inventory','hash','audit','promotion'])
def test_source_bound_rejected_audit_preserves_rejection(tmp_path,monkeypatch,problem):
    commit=next(iter(analysis.SUPPORTED));name='a.py';raw=b'original'
    save(tmp_path/'launch.json',dict(scope='other' if problem=='scope' else analysis.SUPPORTED[commit],cpu_acceptance={'commit':'other' if problem=='commit' else commit}))
    source={name:'bad' if problem=='hash' else hashlib.sha256(raw).hexdigest()}
    if problem=='inventory':source['extra.py']='bad'
    save(tmp_path/'collection-sources.json',source)
    monkeypatch.setattr(analysis,'EXTRAS',set());monkeypatch.setattr(analysis,'source_paths',lambda names:{name})
    monkeypatch.setitem(analysis.SUPPORTED,commit,'effective_event_clock_confirmation')
    # Avoid the live-only extra inventory while retaining the actual guarded commit.
    launch=json.loads((tmp_path/'launch.json').read_text())
    if problem!='scope':launch['scope']='effective_event_clock_confirmation';save(tmp_path/'launch.json',launch)
    monkeypatch.setattr(analysis.subprocess,'check_output',lambda args,**kw:name+'\n' if args[1]=='ls-tree' else raw)
    def invoke(args,**kwargs):
        assert (kwargs['cwd']/name).read_bytes()==raw
        return NS(returncode=1 if problem=='audit' else 0,stdout=json.dumps(dict(audit_valid=True,candidate_passed=False,model_accepted=problem=='promotion')),stderr='')
    monkeypatch.setattr(analysis.subprocess,'run',invoke)
    if problem:
        with pytest.raises(ValueError):analysis.original_audit(tmp_path)
    else:
        result=analysis.original_audit(tmp_path)
        assert result['audit_valid'] and not result['candidate_passed'] and not result['model_accepted']
