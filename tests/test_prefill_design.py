import pytest
from prefill_design import simulate,requests_for,COEF


def req(tokens,output=1,at=0.,rid='r'):
    return dict(id=rid,tokens=tokens,output=output,at=at)


def test_chunk_accounting_and_context_are_exact():
    steps=[];r=simulate([req(7168)],2048,steps_out=steps)
    chunks=[v for s in steps for v in s['chunks']]
    assert [c[1:] for c in chunks]==[(2048,0.),(2048,2048.),(2048,4096.),(1024,6144.)]
    assert r['foreground']['r']['first_s']==pytest.approx(4*COEF[0]+7168*COEF[2])


def test_quadratic_is_not_recharged_for_entire_prompt_each_step():
    a=simulate([req(7168)],2048)
    b=simulate([req(7168)],2048,quadratic=3.15e-9)
    assert b['foreground']['r']['first_s']-a['foreground']['r']['first_s']==pytest.approx(3.15e-9*(7168**2+7168))


def test_backgrounds_have_256_outputs_and_drain_after_barrier():
    r=simulate([req(1024,128)],2048,backgrounds=16)
    assert r['release_s']>0 and r['wall_s']>r['release_s']+1
    assert r['foreground']['r']['first_s']<r['wall_s']-r['release_s']


def test_fifo_blocks_sentinel_while_round_robin_is_distinct():
    rs=[req(7168,128,rid='long'),req(512,128,.01,'sentinel')]
    fifo=simulate(rs,2048);rr=simulate(rs,2048,policy='round_robin_512')
    assert fifo['foreground']['sentinel']['ttft_s']>rr['foreground']['sentinel']['ttft_s']


def test_sentinel_cell_offsets_are_preserved():
    c=dict(foreground_tokens=[4096]*4,foreground_offsets_ms=[0]*4,sentinel=dict(input_tokens=512,arrival_ms=400))
    r=requests_for(c);assert r[-1]['at']==.4 and r[-1]['role']=='sentinel'


@pytest.mark.parametrize('budget',[0,1024,4096])
def test_undeclared_budget_rejected(budget):
    with pytest.raises(ValueError):simulate([req(1024)],budget)


def test_duplicate_identity_rejected():
    with pytest.raises(ValueError,match='Duplicate'):simulate([req(1024),req(2048)],2048)


def test_decode_priority_respects_token_budget_and_background_output_length():
    steps=[];simulate([req(7168,128)],2048,backgrounds=16,steps_out=steps)
    assert all(s['decode']+s['prefill']<=2048 for s in steps)
    assert any(s['decode']>=16 and s['prefill']>0 for s in steps)
    assert any(s['decode_context']>=16*1024 for s in steps)
    assert len(steps)>=256


def test_revised_plan_structure_splits_and_budget(tmp_path):
    import json
    from collections import Counter
    from scripts.build_prefill_design import build,cases,order
    cs=cases();trials=order(cs)
    assert len(cs)==22 and len({c['id'] for c in cs})==22
    assert all(c['background_decoders'] in (0,16) for c in cs)
    assert all(c['split']=='held_case' for c in cs if c['family']=='sentinel')
    for d in (0,16):
        lengths={c['foreground_tokens'][0] for c in cs if c['family']=='single' and c['split']=='calibration' and c['background_decoders']==d}
        assert lengths=={1024,7168}
    assert len({x['trial_id'] for x in trials})==212
    main=[t for t in trials if t['stage']=='main']
    assert Counter(t['analysis_role'] for t in main)=={'train':18,'repeat_check':12,'held_case':80}
    for b in range(1,6):assert {t['case'] for t in main if t['block']==b}=={c['id'] for c in cs}
    build(tmp_path/'first');build(tmp_path/'second')
    for p in (tmp_path/'first').iterdir():assert p.read_bytes()==(tmp_path/'second'/p.name).read_bytes()
    budget=json.loads((tmp_path/'first/budget.json').read_text())
    assert budget['fits_planning_budget'] and budget['estimated_active_seconds']<=1800
    transfers=json.loads((tmp_path/'first/order.json').read_text())['transfer_slices']
    assert [(t['window'],t['replicate']) for t in transfers]==[('a',1),('b',1),('b',2),('a',2)]
    assert len({t['attempt_id'] for t in transfers})==4
    assert all(t['drain_before_next'] and t['attempts_allowed']==1 for t in transfers)
    assert budget['transfer_seconds']==320 and budget['transfer_terminal_reserve_seconds']==40
    assert budget['estimated_active_seconds']==pytest.approx(1517.7774448158516)
    coverage=json.loads((tmp_path/'first/discrimination.json').read_text())['calibration_chunk_coverage']
    assert coverage['l_max']>=6144 and coverage['q_Z_rank']==2


def test_frozen_barrier_adapter_matches_raw_zero_background_and_restores_global_budget():
    import greybox_simulator as gb
    import evaluate_low_cap_frozen as lf
    from scripts.build_prefill_design import frozen_prediction
    coef,overhead,_,_=lf.frozen_parameters();old=gb.B
    c=dict(background_decoders=0,foreground_tokens=[7168],foreground_offsets_ms=[0],sentinel=None)
    expected=simulate(requests_for(c),2048)['foreground']['foreground-0']['ttft_s']
    predicted=frozen_prediction(c,2048,coef,overhead)['foreground-0']['raw_ttft_s']
    assert predicted==pytest.approx(expected) and gb.B==old


def test_all_fifo_scenarios_match_unchanged_frozen_barrier_solver():
    import evaluate_low_cap_frozen as lf
    from scripts.build_prefill_design import frozen_prediction,cases
    coef,overhead,_,_=lf.frozen_parameters()
    for c in cases():
        for b in (2048,8192):
            scenario=simulate(requests_for(c),b,backgrounds=c['background_decoders'])['foreground']
            frozen=frozen_prediction(c,b,coef,overhead)
            assert set(scenario)==set(frozen)
            for rid in scenario:assert scenario[rid]['ttft_s']==pytest.approx(frozen[rid]['raw_ttft_s'],abs=1e-10)


def test_itl_tail_counts_only_declared_bucket_increments(tmp_path):
    import json
    from scripts.check_prefill_budget import itl_tail
    (tmp_path/'manifest.json').write_text(json.dumps(dict(measurement_start=1000.)))
    rows=[]
    for t,counts in [(60,[100,110,115,120]),(300,[200,230,240,245])]:
        raw=''.join(f'vllm:inter_token_latency_seconds_bucket{{engine="0",le="{le}"}} {v}\n' for le,v in zip(['0.1','0.15','0.2','+Inf'],counts))
        rows.append(dict(sent_at=1000.+t,raw_metrics=raw))
    (tmp_path/'engine_observations.jsonl').write_text('\n'.join(map(json.dumps,rows)))
    r=itl_tail(tmp_path)
    assert (r['count'],r['gap_100_150ms'],r['gap_150_200ms'],r['gap_above200ms'])==(125,20,5,0)


@pytest.mark.parametrize('budget,policy,k',[(8192,'fifo',0.),(2048,'fifo',0.),(2048,'fifo',1.5e-9),(2048,'fifo',3.15e-9),(2048,'round_robin_512',0.)])
def test_shortened_background_preserves_all_foreground_service(budget,policy,k):
    from scripts.build_prefill_design import cases
    for c in cases():
        a=[];b=[]
        before=simulate(requests_for(c),budget,backgrounds=c['background_decoders'],background_output=512,policy=policy,quadratic=k,steps_out=a)
        after=simulate(requests_for(c),budget,backgrounds=c['background_decoders'],background_output=256,policy=policy,quadratic=k,steps_out=b)
        assert before['foreground']==after['foreground']
        end=max(r['first_s'] for r in before['foreground'].values())+before['release_s']
        final_prefill=next(i for i,s in enumerate(a) if abs(s['end']-end)<1e-8)
        assert a[:final_prefill+128]==b[:final_prefill+128]
        assert after['wall_s']<=before['wall_s']


def test_itl_counts_token_gaps_not_first_token_and_filters_right_endpoint():
    total=simulate([req(1024,4)],2048,itl_window=(0,100))
    step=COEF[0]+COEF[1]
    assert total['itl']['count']==3
    assert total['itl']['max_gap_s']==pytest.approx(step)
    first=total['foreground']['r']['first_s']
    part=simulate([req(1024,4)],2048,itl_window=(first+step+.0001,first+2*step+.0001))
    assert part['itl']['count']==1
    assert simulate([req(1024)],2048,itl_window=(0,100))['itl']['count']==0


def test_sentinel_discrimination_uses_actual_target_and_flags_policy_tie(tmp_path):
    import json
    from scripts.build_prefill_design import build
    build(tmp_path/'plan')
    d=json.loads((tmp_path/'plan/discrimination.json').read_text())
    cells={c['case']:c for c in d['cases']}
    def rr(offset):
        return next(x['delta_s'] for x in cells[f'sentinel-four-at{offset}-D0']['contrasts'] if x['scenario']=='B2048-round_robin_512-q0.0')
    assert abs(rr(50))>.6 and abs(rr(800))<.005
    assert cells['sentinel-four-at800-D0']['policy_role'].startswith('negative_control')
    assert all(x['request_id']=='sentinel' for x in cells['sentinel-four-at50-D0']['contrasts'])
    assert any(x['scenario']=='B2048-fifo-q1.5e-09' for x in cells['single-L2048-D0']['contrasts'])
    assert cells['single-L2048-D0']['nonlinear_role']=='low_signal_validation_only'
    assert len(d['offset_contrasts'])==10
    assert len({x['id'] for x in d['primary_contrasts']})==42
    assert sum(x['hypothesis']=='H3' for x in d['primary_contrasts'])==6


@pytest.mark.parametrize('window',[(1,0),(0,float('inf'))])
def test_invalid_itl_window_rejected(window):
    with pytest.raises(ValueError):simulate([req(1)],2048,itl_window=window)



def test_eligibility_cannot_count_ties_or_pass_empty_opportunities():
    from prefill_design import contrast_eligibility
    baseline={str(i):.2 for i in range(8)}
    equal=contrast_eligibility(baseline,baseline,{k:.01 for k in baseline})
    assert equal['eligible_cells']==[] and not equal['comparison_evaluable'] and equal['strict_wins_required'] is None
    candidate={k:v+(.05 if int(k)<4 else .02) for k,v in baseline.items()}
    result=contrast_eligibility(candidate,baseline,{k:.01 for k in baseline})
    assert result['eligible_cells']==['0','1','2','3'] and result['strict_wins_required']==3
    assert result['no_regression_cells']==sorted(baseline)
    assert not contrast_eligibility(candidate,baseline,{k:.1 for k in baseline})['comparison_evaluable']


@pytest.mark.parametrize('jitter',[-1,float('nan'),float('inf')])
def test_eligibility_rejects_bad_noise(jitter):
    from prefill_design import contrast_eligibility
    with pytest.raises(ValueError):contrast_eligibility({str(i):.2 for i in range(8)},{str(i):.2 for i in range(8)},jitter)


def test_long_prefill_produces_decode_tail_only_in_large_budget_scenario():
    requests=[req(1024,128,rid='decode'),req(7168,128,.1,'long')]
    large=simulate(requests,8192,itl_window=(0,10))['itl']
    small=simulate(requests,2048,itl_window=(0,10))['itl']
    assert large['gap_above200ms']>0 and small['gap_above200ms']==0
    assert large['count']==small['count']==254


def test_transfer_selects_scheduled_cohort_preserves_acceptance_delay_and_outputs(tmp_path,monkeypatch):
    import json
    from scripts import check_prefill_itl as itl
    table=dict(request_id=['lead','score','outside','reject'],trace_id=['a','b','c','d'],
        admitted=[True,True,True,False],sched=[40.,60.,120.,70.],acc=[40.02,60.03,120.01,float('nan')])
    monkeypatch.setattr(itl.lf,'load_capture_compatible',lambda _: ({},table,0))
    (tmp_path/'trace.json').write_text(json.dumps(dict(requests=[dict(trace_id=k,input_tokens=100+i) for i,k in enumerate('abcd')])))
    (tmp_path/'client.json').write_text(json.dumps(dict(requests=[dict(workload='trace',request_id=k,success=True,output_tokens=10+i) for i,k in enumerate(table['request_id'])])))
    actual=itl.accepted_requests(tmp_path,replay_start=40,replay_end=120)
    assert [r['id'] for r in actual]==['lead','score']
    assert [r['at'] for r in actual]==pytest.approx([.02,20.03])
    assert [r['output'] for r in actual]==[10,11]
    assert all(r['output']==128 for r in itl.accepted_requests(tmp_path,replay_start=40,replay_end=120,forced_output=128))



def test_committed_design_metadata_and_generator_outputs_are_current(tmp_path):
    import json,hashlib
    from pathlib import Path
    from scripts.build_prefill_design import build,ROOT
    build(tmp_path/'fresh')
    for name in ('cases','order','predictions','discrimination','budget'):
        assert (tmp_path/f'fresh/{name}.json').read_bytes()==(ROOT/f'experiments/prefill-service-v1.{name}.json').read_bytes()
    design=json.loads((ROOT/'experiments/prefill-service-v1.design.json').read_text())
    assert design['revision']==7 and design['workload']['background_output_tokens']==256
    assert design['collection']['transfer_seconds']==320
    for row in design['files'].values():assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256']
