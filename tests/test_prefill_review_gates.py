import copy
import json
import math
import numpy as np
import pytest
from prefill_histogram import exporter_histogram,simulated_histogram,discrepancy,snapshot
from prefill_design import contrast_eligibility
from prefill_validation import position_jitter,assign_jitter,comparison_plan,paired_error_check

BOUNDS=[.01,.025,.05,.1,.15,.2,math.inf]


def raw(counts):
    return '\n'.join(f'vllm:inter_token_latency_seconds_bucket{{engine="0",le="{b}",model_name="q"}} {n}' for b,n in zip(BOUNDS,counts))


def capture(tmp,rows):
    (tmp/'manifest.json').write_text(json.dumps(dict(measurement_start=1000.)))
    (tmp/'engine_observations.jsonl').write_text('\n'.join(map(json.dumps,rows)))
    return tmp


def rows():
    return [dict(sent_at=1000+t,raw_metrics=raw(c),counter_epoch=1,exporter_identity={'pid':1}) for t,c in [
        (60,[0,10,20,22,24,25,25]),(80,[1,20,35,38,40,45,45]),(120,[1,30,50,55,65,70,70])]]


def test_full_exporter_vector_uses_disjoint_bins_and_records_actual_endpoints(tmp_path):
    d=exporter_histogram(capture(tmp_path,rows()),59,121)
    assert d['actual_window_s']==[60.,120.] and d['upper_bounds_s'][-1] is None
    assert d['counts']==[1,19,10,3,8,4,0] and d['total']==45


@pytest.mark.parametrize('bad',['reset','identity','epoch','missing','failure','bucket_change','order','missing_identity','missing_epoch'])
def test_no_silent_bridge_over_bad_histogram_samples(tmp_path,bad):
    r=rows()
    if bad=='reset':r[1]['raw_metrics']=raw([0,0,0,0,0,0,0])
    if bad=='identity':r[1]['exporter_identity']={'pid':2}
    if bad=='epoch':r[1]['counter_epoch']=2
    if bad=='missing':r[1]['raw_metrics']=''
    if bad=='failure':r[1]['error_type']='ReadTimeout'
    if bad=='bucket_change':r[1]['raw_metrics']=r[1]['raw_metrics'].replace('le="0.05"','le="0.06"')
    if bad=='order':r[1]['sent_at']=r[0]['sent_at']
    if bad=='missing_identity':r[1].pop('exporter_identity')
    if bad=='missing_epoch':r[1].pop('counter_epoch')
    with pytest.raises(ValueError):exporter_histogram(capture(tmp_path,r),59,121)


def test_duplicate_bucket_and_nonmonotone_cumulative_rejected():
    s=raw([0,10,20,22,24,25,25])
    with pytest.raises(ValueError):snapshot(s+'\n'+s.splitlines()[0])
    with pytest.raises(ValueError):snapshot(raw([0,20,10,22,24,25,25]))


def test_simulated_histogram_weights_decoders_and_excludes_first_tokens():
    d=simulated_histogram([dict(start=0.,end=.05,decode=3),dict(start=.05,end=.2,decode=2)],(0,.1),[.025,.05,.2,None])
    assert d['counts']==[0,3,0,0] and d['total']==3


def test_joint_metric_catches_wrong_neighbours_despite_matching_tail():
    obs=dict(upper_bounds_s=[.025,.05,.15,.2,None],counts=[1000,100,50,10,0])
    pred=dict(upper_bounds_s=obs['upper_bounds_s'],counts=[1000,10,500,10,0])
    d=discrepancy(obs,pred)
    assert d['large_joint_discrepancy'] and d['worst_bin_rate_factor']>5
    assert d['automatic_model_acceptance'] is False


def test_identical_histograms_have_zero_discrepancy_even_with_empty_bins():
    h=dict(upper_bounds_s=[.025,.05,.15,.2,None],counts=[100,10,20,0,0])
    assert discrepancy(h,h)['max_abs_log_rate_ratio']==0


def observer_trials():
    return {'single-L1024-D0':[[.05]]*20,'composition-uniform-D16':[[.2,.4,.6,.8]]*10}


def test_position_jitter_does_not_confuse_request_position_with_noise():
    t=observer_trials();d=position_jitter(t)
    assert all(x['jitter_s']==0 for x in d.values())
    assert np.median(np.abs(np.array(t['composition-uniform-D16'])-.5))>0
    mapping={str(i):'composition-uniform-D16/foreground-3' for i in range(8)}
    assert assign_jitter(d,mapping)=={str(i):0. for i in range(8)}


def test_jitter_requires_complete_declared_off_trials():
    t=observer_trials();t['composition-uniform-D16']=t['composition-uniform-D16'][:-1]
    with pytest.raises(ValueError):position_jitter(t)


def predictions():
    base={str(i):.2 for i in range(8)}
    return base,{k:v+.05 for k,v in base.items()}


def test_eligibility_uses_each_cells_noise_and_rejects_global_scalar():
    base,cand=predictions();j={k:(0 if int(k)<4 else .03) for k in base}
    e=contrast_eligibility(cand,base,j)
    assert e['eligible_cells']==['0','1','2','3']
    assert e['threshold_s_by_cell']['0']==.025 and e['threshold_s_by_cell']['7']==.09
    with pytest.raises(ValueError):contrast_eligibility(cand,base,.01)


def test_recalibration_cannot_be_called_structural_success():
    base,cand=predictions();j={k:0. for k in base}
    p=comparison_plan('M0','M0',cand,cand,base,j)
    assert p['structure_status']=='not_applicable' and p['required_structural_comparators']=={}
    assert p['raw_B2048_staleness_only']['comparison_evaluable']


def test_M2_must_compare_both_calibrated_M0_and_retained_M1():
    base,cand=predictions();j={k:0. for k in base}
    with pytest.raises(ValueError):comparison_plan('M2','M1',cand,base,base,j)
    p=comparison_plan('M2','M1',cand,base,base,j,calibrated_m1=base)
    assert set(p['required_structural_comparators'])=={'calibrated_M0','calibrated_M1'}


def test_noneligible_cells_still_have_regression_guards_and_ties_are_not_wins():
    base,cand=predictions();j={k:(0 if int(k)<4 else .1) for k in base}
    e=contrast_eligibility(cand,base,j)
    a={k:[0.]*5 for k in base};b={k:[.06]*5 for k in base}
    assert paired_error_check(e,a,b)['component_status']=='passed'
    a['7']=[.2]*5
    r=paired_error_check(e,a,b)
    assert r['component_status']=='failed' and r['regressions']==['7']
    assert paired_error_check(e,b,b)['component_status']=='inconclusive'


def test_unresolved_and_incomplete_errors_do_not_pass():
    base,cand=predictions();e=contrast_eligibility(cand,base,{k:.1 for k in base})
    z={k:[0.]*5 for k in base}
    assert paired_error_check(e,z,z)['component_status']=='inconclusive'
    del z['0']
    with pytest.raises(ValueError):paired_error_check(e,z,z)


def test_missing_retained_comparator_cannot_pass_structural_check():
    from prefill_validation import structural_error_check
    base,cand=predictions();j={k:0. for k in base}
    p=comparison_plan('M2','M1',cand,base,base,j,calibrated_m1=base)
    errors={k:[0.]*5 for k in base};b={k:[.1]*5 for k in base}
    with pytest.raises(ValueError):structural_error_check(p,errors,{'calibrated_M0':b})
    r=structural_error_check(p,errors,{'calibrated_M0':b,'calibrated_M1':b})
    assert r['structural_status']=='components_passed' and not r['complete_experiment_passed']



def test_empty_structural_plan_cannot_pass_vacuously():
    from prefill_validation import structural_error_check
    with pytest.raises(ValueError):structural_error_check(dict(structure_status='not_evaluated',required_structural_comparators={}),{}, {})


@pytest.mark.parametrize('improvement,expected',[(.000001,'inconclusive'),(.025,'inconclusive'),(.026,'passed')])
def test_win_requires_strict_resolved_mae_improvement(improvement,expected):
    base={str(i):.2 for i in range(8)}
    cand={k:v+(.06 if int(k)<4 else 0.) for k,v in base.items()}
    e=contrast_eligibility(cand,base,{k:.005 for k in base})
    b={k:[.05]*5 for k in base};a=copy.deepcopy(b)
    for k in ('0','1','2'):a[k]=[.05-improvement]*5
    result=paired_error_check(e,a,b)
    assert result['component_status']==expected
    assert result['wins']==(['0','1','2'] if expected=='passed' else [])
    assert not result['complete_experiment_passed']


def test_win_uses_cell_specific_noise_instead_of_global_floor():
    base,cand=predictions();cand={k:v+.2 for k,v in base.items()}
    e=contrast_eligibility(cand,base,{k:.03 for k in base})
    a={k:[0.]*5 for k in base};b={k:[.05]*5 for k in base}
    assert paired_error_check(e,a,b)['wins']==[]


def test_shared_step_floor_retains_zero_support_and_clustering():
    from prefill_histogram import shared_step_reference,support_and_magnitude
    h=simulated_histogram([dict(start=0.,end=.05,decode=10),dict(start=.05,end=.1,decode=20)],(0,.2),[.025,.1,.2,None])
    assert h['decode_bearing_steps']==2 and h['decode_bearing_steps_by_bin']==[0,2,0,0]
    ref=shared_step_reference(h)
    assert ref['count_floor']==75 and ref['mean_toy_gaps_per_step']==15
    obs=dict(upper_bounds_s=h['upper_bounds_s'],counts=[1000,100,80,0])
    candidates=[dict(upper_bounds_s=h['upper_bounds_s'],counts=[1000,100,0,0]),
                dict(upper_bounds_s=h['upper_bounds_s'],counts=[1000,200,20,0])]
    scores=[support_and_magnitude(obs,p,ref,[0,1]) for p in candidates]
    assert all(s['magnitude']['count_floor']==75 for s in scores)
    assert scores[0]['material_support_mismatch'] and not scores[1]['material_support_mismatch']
    assert scores[0]['support_mismatches'][0]['observed_equivalent_steps']==80/15
    assert scores[0]['common_positive_sensitivity']['scored_bin_indices']==[1]
    assert scores[0]['legacy_floor_sensitivity']['1']['worst_bin_rate_factor']>scores[0]['magnitude']['worst_bin_rate_factor']
    assert not scores[0]['automatic_model_acceptance']


def test_empty_or_invalid_step_reference_cannot_supply_floor():
    from prefill_histogram import shared_step_reference
    with pytest.raises(ValueError):shared_step_reference(dict(total=10,decode_bearing_steps=0))
    with pytest.raises(ValueError):discrepancy(dict(),dict(),count_floor=float('nan'))


def test_published_histograms_share_linear_reference_and_match_sources():
    from scripts.build_prefill_design import ROOT,digest
    d=json.loads((ROOT/'results/prefill-design-round4/itl-scenarios.json').read_text())
    assert d['verified_source_files']=={'old':161,'new':138}
    snapshots=json.loads((ROOT/'results/prefill-design-round4/source-snapshots.json').read_text())
    for name,sha in d['code_sha256'].items():assert digest(ROOT/snapshots.get(name,name))==sha
    assert len(d['replicate_references'])==3
    assert sum(r['same_host'] for r in d['replicate_references'])==1
    for row in d['historical']+d['transfer_slices']:
        scenarios=row['full_histogram']['scenarios']
        linear=scenarios['B2048-fifo-k0.0']['histogram']
        expected=5*linear['total']/linear['decode_bearing_steps']
        for scenario in scenarios.values():
            score=scenario['joint_discrepancy']
            assert score['magnitude']['count_floor']==pytest.approx(expected)
            assert score['shared_reference']['reference_id']=='B2048-fifo-k0.0'
            assert not score['automatic_model_acceptance']
        assert len(row['replicate_reference_ids'])==3


def test_published_win_counterexample_is_reproducible():
    from scripts.check_prefill_win_margin import checks,ROOT,digest
    saved=json.loads((ROOT/'results/prefill-design-round4/win-margin-counterexample.json').read_text())
    for name,sha in saved.pop('source_sha256').items():assert digest(ROOT/name)==sha
    assert saved==checks()
    assert [r['revised']['component_status'] for r in saved['cases']]==['inconclusive','inconclusive','passed','failed']
