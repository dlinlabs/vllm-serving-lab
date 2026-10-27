"""Source audit and all-sample development scoring; no GPU acceptance."""
import copy
import json
from pathlib import Path
import pytest
from prefill_lab import queue_probe as probe
from scripts import analyze_prefill_queue_probe as analysis
from test_prefill_queue_probe import rows


def fixtures():
    p=probe.protocol();data=rows(p)
    for r in data:r['work_signature']='same'
    predicted={c['id']:dict(endpoints={'sentinel':dict(ttft_s=.2,completion_s=1.)},work_signature='same',
        background_outlives_foreground=True,sentinel_queue_residence_s=.05) for c in p['cases']}
    cells={c['id']:dict(status='point',branches=[dict(queue_s=.05)]) for c in p['cases'] if c['sentinel']}
    return p,data,predicted,cells


def test_development_keeps_all_arms_and_never_promotes_acceptance():
    p,data,predicted,cells=fixtures();r=analysis.score(data,p,predicted,cells)
    assert all(v==dict(samples=48,point_failures=0,phase_failures=0) for v in r['counts'].values())
    assert len(r['sample_checks'])==144 and len(r['work_signature_checks'])==48 and r['queue_budget_passed']
    assert not any(r[k] for k in ('model_accepted','http_accepted','adaptive_cap_accepted','mpc_accepted'))


@pytest.mark.parametrize('fault',['missing','duplicate','phase','tie','signature'])
def test_failed_or_unresolved_samples_not_excluded(fault):
    p,data,predicted,cells=fixtures()
    if fault=='missing':data.pop()
    if fault=='duplicate':data[-1]=copy.deepcopy(data[0])
    if fault in ('missing','duplicate'):
        with pytest.raises(ValueError):analysis.score(data,p,predicted,cells)
        return
    case=next(c['id'] for c in p['cases'] if c['sentinel'])
    if fault=='phase':cells.pop(case)
    if fault=='tie':
        for r in data:
            if r['stage']=='calibration_probe' and r['case']==case:r['clocks']['sentinel']['queue_residence_s']=.005
        cells[case]=dict(status='phase_ambiguous',branches=[dict(queue_s=0.),dict(queue_s=.01)])
    if fault=='signature':next(r for r in data if r['stage']=='calibration_probe' and r['arm']=='light')['work_signature']='changed'
    report=analysis.score(data,p,predicted,cells)
    assert len(report['sample_checks'])==144
    if fault=='signature':assert any(not r['matches'] for r in report['work_signature_checks'])
    else:assert not report['queue_budget_passed']


@pytest.mark.parametrize('fault',['commit','scope','inventory','hash'])
def test_original_source_rejects_wrong_provenance(tmp_path,fault):
    from prefill_lab.contracts import save
    capture=tmp_path/'capture';destination=tmp_path/'source';capture.mkdir();destination.mkdir()
    sources={'not-reviewed.py':'bad'}
    launch=dict(scope='queue_receipt_calibration_probe',cpu_acceptance=dict(commit=analysis.COMMIT))
    if fault=='commit':launch['cpu_acceptance']['commit']='a'*40
    if fault=='scope':launch['scope']='live_http_state_bridge'
    if fault=='hash':
        names=analysis.subprocess.check_output(['git','ls-tree','-r','--name-only',analysis.COMMIT],cwd=analysis.ROOT,text=True).splitlines()
        sources={name:'bad' for name in analysis.source_paths(names)|analysis.EXTRAS}
    save(capture/'launch.json',launch);save(capture/'collection-sources.json',sources)
    with pytest.raises(ValueError):analysis.original_source(capture,destination)
