"""Rejected TRAIN must remain rejected; a fixed-case point can be infeasible."""
import copy
import hashlib
import json
import pytest
from prefill_lab.contracts import save
from scripts import analyze_prefill_entry_http_failure as analysis


def candidate():
    checks=[]
    for case in range(6):
        for repeat in range(4):
            observed=.025 if (case,repeat)==(5,0) else .05;predicted=.047;limit=.01
            checks.append(dict(trial_id=f'case{case}-repeat{repeat}',case=str(case),observed_s=observed,predicted_s=predicted,
                limit_s=limit,error_s=predicted-observed,passed=abs(predicted-observed)<=limit))
    return dict(readiness_passed=False,errors=[],first_commit_readiness=dict(passed=False,checks=checks),labels=dict(constants_s=dict(prefill=.047,decode=.002)))


def test_disjoint_repeat_budgets_rule_out_case_constant_without_claiming_state_failure():
    f=candidate();original=copy.deepcopy(f);s=analysis.summarize(f)
    assert f==original and s['first_commit_samples']==24 and s['first_commit_passed']==23
    assert len(s['first_commit_failures'])==1 and not s['phase_errors']
    cell=next(r for r in s['case_constant_intersections'] if r['case']=='5')
    assert not cell['one_case_constant_feasible'] and cell['gap_s']==pytest.approx(.005)
    assert cell['lower_s']==pytest.approx(.04) and cell['upper_s']==pytest.approx(.035)
    assert all(c['one_case_constant_feasible'] for c in s['case_constant_intersections'] if c['case']!='5')
    assert 'state-conditioned' in s['interpretation'] and 'TRAIN-only' in s['scope']
    assert all(s[k] is False for k in ('model_accepted','http_accepted','adaptive_cap_accepted','mpc_accepted'))


@pytest.mark.parametrize('fault',['missing','duplicate','cell','nan','limit','verdict','readiness','passed'])
def test_rejected_summary_keeps_membership_and_original_verdicts(fault):
    f=candidate();checks=f['first_commit_readiness']['checks']
    if fault=='missing':checks.pop()
    if fault=='duplicate':checks[-1]=copy.deepcopy(checks[0])
    if fault=='cell':checks[-1]['case']='extra'
    if fault=='nan':checks[-1]['observed_s']=float('nan')
    if fault=='limit':checks[-1]['limit_s']=-1
    if fault=='verdict':checks[-1]['passed']=False
    if fault=='readiness':f['readiness_passed']=True
    if fault=='passed':
        for r in checks:r.update(predicted_s=r['observed_s'],error_s=0,passed=True)
        f['first_commit_readiness']['passed']=True;f['readiness_passed']=True
    with pytest.raises(ValueError):analysis.summarize(f)


@pytest.mark.parametrize('fault',[None,'commit','scope','inventory','hash'])
def test_source_snapshot_binds_native_inventory_to_pinned_git(monkeypatch,tmp_path,fault):
    capture=tmp_path/'capture';capture.mkdir();destination=tmp_path/'source';destination.mkdir()
    launch=dict(scope='entry_http_state_bridge',cpu_acceptance=dict(commit=analysis.COMMIT))
    names={'reviewed.py',*analysis.EXTRAS};raw=b'original source fixture\n';sources={n:hashlib.sha256(raw).hexdigest() for n in names}
    def git(args,**kwargs):
        if args[1]=='ls-tree':return ('\n'.join(sorted(names))+'\n').encode()
        assert args[1]=='show' and args[2].startswith(analysis.COMMIT+':');return raw
    monkeypatch.setattr(analysis.subprocess,'check_output',git)
    if fault=='commit':launch['cpu_acceptance']['commit']='wrong'
    if fault=='scope':launch['scope']='independent_queue_confirmation'
    if fault=='inventory':sources.pop('reviewed.py')
    if fault=='hash':sources['reviewed.py']='0'*64
    save(capture/'launch.json',launch);save(capture/'collection-sources.json',sources)
    if fault:
        with pytest.raises(ValueError):analysis.snapshot(capture,destination)
    else:
        assert analysis.snapshot(capture,destination)==sources
        assert (destination/'reviewed.py').read_bytes()==raw


def test_numerical_replay_rejects_wrong_versions_before_refit(tmp_path):
    (tmp_path/'controlled').mkdir();save(tmp_path/'controlled/host-before.json',dict(packages=dict(numpy='unsupported',scipy='unsupported')))
    with pytest.raises(ValueError,match='Exact numerical replay requires'):analysis.reconstruct(tmp_path,tmp_path/'output')


@pytest.mark.parametrize('different',[False,True])
def test_release_controls_normalize_ids_but_do_not_hide_state_changes(tmp_path,different):
    root=tmp_path/'controlled';root.mkdir();checks=[]
    for i in range(2):
        tid=f'qc-{i:03d}';rid='plab/'+tid+'/background-0'
        save(root/(tid+'.json'),dict(release_state=dict(requests={rid:dict(computed=int(different and i==1))},running=[rid],waiting=[])))
        checks.append(dict(trial_id=tid,case='same',observed_s=.025+i*.025))
    controls=analysis.release_controls(tmp_path,checks)
    assert controls[0]['distinct_recorded_scheduler_states']==(2 if different else 1)
    assert 'excludes pending readiness' in controls[0]['scope']
    assert [r['observed_first_commit_s'] for r in controls[0]['rows']]==[.025,.05]


def test_output_cannot_modify_immutable_capture(tmp_path):
    with pytest.raises(ValueError,match='outside immutable'):analysis.run(tmp_path,tmp_path/'unused.tar.gz',tmp_path/'output')


def test_original_worker_compiles_without_modifying_source():
    compile(analysis.WORKER,'original-rejection-worker','exec')
