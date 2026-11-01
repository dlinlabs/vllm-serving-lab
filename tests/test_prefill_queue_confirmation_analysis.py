"""Descriptive held evidence must not become a correction-win/control claim."""
import copy
import hashlib
import json
import pytest
from prefill_lab import queue_confirmation as qc
from prefill_lab.contracts import save
from scripts import analyze_prefill_queue_confirmation as analysis
from test_prefill_queue_confirmation import fixture


def reports():
    p,rows,frozen=fixture();report=qc.evaluate(rows,p,frozen)
    frozen['first_commit_readiness']=dict(passed=True,checks=[dict(case='training-D'+('16' if i%2 else '0'),error_s=.001) for i in range(24)])
    return report,frozen


def test_identical_forecasts_do_not_establish_correction_benefit():
    report,frozen=reports();s=analysis.summarize(report,frozen)
    assert len(s['queue_cells'])==24 and all(r['samples']==6 for r in s['queue_cells'])
    assert s['f0_and_entry_forecasts_identical'] and not s['correction_benefit_established']
    assert not s['previous_host_phase_switch_confirmed'] and not s['baseline_native_phase_observed']
    assert all(v==1 for v in s['eligible_branches'].values())
    assert s['observer_controls']['comparable_pairs']==48
    assert all(s[k] is False for k in ('model_accepted','http_accepted','adaptive_cap_accepted','mpc_accepted'))


@pytest.mark.parametrize('fault',['missing','duplicate','arm','cell'])
def test_summary_preserves_all_held_denominators(fault):
    report,frozen=reports();rows=report['sample_checks']
    if fault=='missing':rows.pop()
    if fault=='duplicate':rows[-1]=copy.deepcopy(rows[0])
    if fault=='arm':rows[0]['arm']='unexpected'
    if fault=='cell':rows[0]['case']='unexpected'
    with pytest.raises(ValueError,match='denominator'):analysis.summarize(report,frozen)


def test_queue_errors_are_sample_errors_and_endpoint_errors_keep_median_estimand():
    report,frozen=reports();row=report['sample_checks'][0];row['observed_s']=.057
    summary=analysis.summarize(report,frozen)
    group=next(c for c in summary['queue_cells'] if c['case']==row['case'] and c['arm']==row['arm'])
    assert group['max_absolute_error_s']==pytest.approx(.007)
    assert group['mae_s']==pytest.approx(.007/6)
    assert group['observed_median_s']==.05
    assert summary['endpoint_scores']['held_queue:baseline']['ttft_s']['request_positions']==40


@pytest.mark.parametrize('fault',['commit','scope','inventory','hash'])
def test_snapshot_rejects_wrong_source_without_checkout_history(tmp_path,monkeypatch,fault):
    capture=tmp_path/'capture';capture.mkdir();destination=tmp_path/'source';destination.mkdir()
    launch=dict(scope='independent_queue_confirmation',cpu_acceptance=dict(commit=analysis.COMMIT))
    names={'reviewed.py',*analysis.EXTRAS};raw=b'source fixture\n';sources={n:hashlib.sha256(raw).hexdigest() for n in names}
    def git(args,**kwargs):
        if args[1]=='ls-tree':return ('\n'.join(sorted(names))+'\n').encode()
        assert args[1]=='show';return raw
    monkeypatch.setattr(analysis.subprocess,'check_output',git)
    if fault=='commit':launch['cpu_acceptance']['commit']='0'*40
    if fault=='scope':launch['scope']='wrong'
    if fault=='inventory':sources.pop('reviewed.py')
    if fault=='hash':sources['reviewed.py']='0'*64
    save(capture/'launch.json',launch);save(capture/'collection-sources.json',sources)
    with pytest.raises(ValueError,match='Unsupported|inventory|mismatch'):analysis.snapshot(capture,destination)


def test_exact_replay_refuses_different_numerical_environment(tmp_path):
    save(tmp_path/'host-before.json',dict(packages=dict(numpy='unsupported',scipy='unsupported')))
    with pytest.raises(ValueError,match='Exact numerical replay requires'):analysis.original_audit(tmp_path)
