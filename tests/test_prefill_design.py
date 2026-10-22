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


def test_backgrounds_have_512_outputs_and_drain_after_barrier():
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
    assert len(steps)>=512


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
