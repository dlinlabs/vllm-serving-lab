"""Source audit and all-sample development scoring; no GPU acceptance."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
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
def test_original_source_rejects_wrong_provenance(tmp_path,monkeypatch,fault):
    from prefill_lab.contracts import save
    capture=tmp_path/'capture';destination=tmp_path/'source';capture.mkdir();destination.mkdir()
    sources={'not-reviewed.py':'bad'}
    # Unit boundary: no checkout history or network is required to exercise
    # rejection of the supplied source inventory and bytes.
    names=['reviewed.py']
    def git(args,**kwargs):
        assert args[0]=='git'
        if args[1]=='ls-tree':return '\n'.join(names)+'\n'
        assert args[1]=='show';return b'reviewed source fixture\n'
    monkeypatch.setattr(analysis.subprocess,'check_output',git)
    launch=dict(scope='queue_receipt_calibration_probe',cpu_acceptance=dict(commit=analysis.COMMIT))
    if fault=='commit':launch['cpu_acceptance']['commit']='a'*40
    if fault=='scope':launch['scope']='live_http_state_bridge'
    if fault=='hash':
        sources={name:'bad' for name in analysis.source_paths(names)|analysis.EXTRAS}
    save(capture/'launch.json',launch);save(capture/'collection-sources.json',sources)
    message={'commit':'Unsupported receipt','scope':'Unsupported receipt',
        'inventory':'Native source inventory differs','hash':'Git/native source mismatch'}[fault]
    with pytest.raises(ValueError,match=message):analysis.original_source(capture,destination)


@pytest.fixture
def shallow_source(tmp_path,monkeypatch):
    """Real depth-one Git clone, with a genuinely absent parent commit."""
    repository=tmp_path/'repository';repository.mkdir()
    def git(*args,cwd=repository):
        return subprocess.check_output(['git',*args],cwd=cwd,stderr=subprocess.PIPE,text=True).strip()
    git('init','-q')
    names={'reviewed.py',*analysis.EXTRAS}
    for name in names:
        p=repository/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('original source fixture\n')
    git('add','.');git('-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','parent')
    parent=git('rev-parse','HEAD');(repository/'reviewed.py').write_text('current source fixture\n')
    git('add','.');git('-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','head')
    current=git('rev-parse','HEAD');clone=tmp_path/'shallow'
    git('clone','-q','--depth','1',repository.as_uri(),str(clone))
    assert git('rev-parse','--is-shallow-repository',cwd=clone)=='true'
    assert subprocess.run(['git','cat-file','-e',parent+'^{tree}'],cwd=clone,capture_output=True).returncode!=0
    monkeypatch.setattr(analysis,'ROOT',clone);monkeypatch.setattr(analysis,'COMMIT',current)
    capture=tmp_path/'capture';capture.mkdir();destination=tmp_path/'snapshot';destination.mkdir()
    from prefill_lab.contracts import save
    sources={name:hashlib.sha256((clone/name).read_bytes()).hexdigest() for name in names}
    save(capture/'launch.json',dict(scope='queue_receipt_calibration_probe',cpu_acceptance=dict(commit=current)))
    save(capture/'collection-sources.json',sources)
    return clone,parent,current,capture,destination,sources


@pytest.mark.parametrize('fault',[None,'inventory','hash','missing_parent','missing_blob'])
def test_real_shallow_source_keeps_exact_audit_guards(shallow_source,monkeypatch,fault):
    from prefill_lab.contracts import save
    clone,parent,current,capture,destination,sources=shallow_source
    if fault=='inventory':sources.pop('reviewed.py')
    if fault=='hash':sources['reviewed.py']='0'*64
    if fault=='missing_parent':
        monkeypatch.setattr(analysis,'COMMIT',parent)
        save(capture/'launch.json',dict(scope='queue_receipt_calibration_probe',cpu_acceptance=dict(commit=parent)))
    if fault=='missing_blob':
        original=analysis.subprocess.check_output
        def incomplete(args,**kwargs):
            if args[1]=='show':raise subprocess.CalledProcessError(128,args,stderr=b'missing blob')
            return original(args,**kwargs)
        monkeypatch.setattr(analysis.subprocess,'check_output',incomplete)
    save(capture/'collection-sources.json',sources)
    if fault:
        expected='git fetch --depth 1 origin '+(parent if fault=='missing_parent' else current) if fault in ('missing_parent','missing_blob') else 'inventory|mismatch'
        with pytest.raises(ValueError,match=expected):
            analysis.original_source(capture,destination)
        assert not (destination/'reviewed.py').exists()
    else:
        assert analysis.original_source(capture,destination)==sources
        assert all((destination/name).read_bytes()==(clone/name).read_bytes() for name in sources)
