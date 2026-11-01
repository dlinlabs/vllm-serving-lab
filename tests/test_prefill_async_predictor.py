"""Hand-computed dependency/state contracts, not GPU predictive-accuracy tests."""
import json
import copy
from pathlib import Path

import pytest

from prefill_lab.async_predictor import (CostLaw, EventGraph, PipelineCosts, Request,
                                        Scope, PARTS, LAUNCHES, SCALARS, simulate, rollout)

ROOT = Path(__file__).resolve().parents[1]
EAGER = dict(schema='v2-dispatch-candidates-v1', graphs_captured=False, candidates={})


def law(seconds=0., **coefficients):
    return CostLaw(dict(bias=seconds, **coefficients))


def costs(**overrides):
    launches = {name: {part: law() for part in PARTS} for name in LAUNCHES}
    scalar = {name: law() for name in SCALARS}
    for key, value in overrides.items():
        if '.' in key:
            stage, part = key.split('.')
            launches[stage][part] = law(value)
        else:
            scalar[key] = law(value)
    return PipelineCosts(launches, scalar, 'unit-test synthetic costs in seconds')


def run(requests=None, model=None, **kwargs):
    return simulate(requests or [Request('r', 4, 1)], model or costs(), EAGER, **kwargs)


def test_cpu_return_is_not_a_gpu_start_dependency():
    graph = EventGraph()
    launch = dict(cpu_to_first=law(2), cpu_between=law(8), cpu_after=law(3),
                  gpu_prefix=law(6), gpu_tail=law(1))
    returned, tail = graph.launch('forward', launch, {})
    # GPU prefix [2,8] overlaps CPU enqueue [2,10]; tail waits last enqueue.
    assert [(n['start_s'], n['end_s']) for n in graph.nodes] == [(0,2),(2,8),(2,10),(10,11),(10,13)]
    assert graph.end(returned) == 13 and graph.end(tail) == 11


def test_copy_and_update_fork_and_dependency_wait_is_charged_once():
    result = run(model=costs(**{'forward.gpu_tail':10, 'logits_sample.gpu_tail':2,
                              'copy_service':3, 'post_update.gpu_tail':7, 'commit_cpu':1}))
    step = result['steps'][0]
    nodes = result['nodes']
    copy, update = nodes[step['copy_node']], nodes[step['update_node']]
    assert (copy['start_s'], copy['end_s']) == (12,15)
    assert (update['start_s'], update['end_s']) == (12,19)
    assert result['requests']['r']['first_s'] == 16
    assert result['resources_drained_s'] == 19
    assert not any(n['name'] == 'sample_tokens' for n in nodes)


def test_next_main_work_waits_update_but_not_the_copy_branch():
    result = run([Request('r', 4, 2)], costs(**{'forward.gpu_tail':1,
                 'copy_service':10, 'post_update.gpu_tail':2}))
    first, second = result['steps']
    nodes = result['nodes']
    assert nodes[second['forward_node']]['start_s'] == 3
    assert nodes[first['copy_node']]['end_s'] == 11
    assert nodes[second['copy_node']]['start_s'] == 11  # Copy stream remains FIFO.


def test_ready_output_does_not_preempt_second_submission():
    result = run([Request('r', 4, 2)])
    queue = [(a['operation'], a['batch']) for a in result['actions'] if a['operation'] in ('submit','reap')]
    assert queue[:4] == [('submit',0),('submit',1),('reap',0),('submit',2)]
    assert result['peak_slots'] == 2
    assert [bid for action,bid in queue if action == 'reap'] == [bid for action,bid in queue if action == 'submit']


def test_chunks_and_placeholders_use_own_preexecution_positions():
    result = run([Request('r', 7168, 2)])
    assert [(s['work'][0]['position'],s['work'][0]['tokens']) for s in result['steps']] == [
        (0,2048),(2048,2048),(4096,2048),(6144,1024),(7168,1)]
    assert [bool(s['emits']) for s in result['steps']] == [False,False,False,True,True]
    final = result['steps'][-1]['state_before']['r']
    assert final['computed'] == 7168
    assert final['num_tokens'] == 7168  # First output is still in flight.
    assert final['placeholders'] == 1 and final['in_flight_tokens'] == 1024
    assert result['requests']['r']['output_tokens'] == 2


@pytest.mark.parametrize('output', [1,2,3,128,256])
def test_terminal_guard_prevents_extra_outputs_and_drains_empty_futures(output):
    result = run([Request('r', 16, output)])
    assert sum(len(s['emits']) for s in result['steps']) == output
    assert len(result['steps']) == output
    assert result['requests']['r']['output_tokens'] == output
    assert result['actions'][-1]['operation'] == 'reap'
    assert result['actions'][-1]['occupied_slots'] == 0
    assert any(a.get('empty') for a in result['actions'])


def test_empty_native_iteration_yields_only_while_requests_remain():
    result = run([Request('a', 4, 1),Request('b',4,1)],scope=Scope(max_seqs=1))
    yields = [n for n in result['nodes'] if n['name'] == 'native_idle_yield']
    assert len(yields) == 1 and yields[0]['service_s'] == .001
    assert result['requests']['b']['finished_s'] == .001


def test_arrivals_wait_for_input_queue_poll_not_gpu_completion_callback():
    result = run([Request('long', 4, 3), Request('sentinel', 2, 1, .5, 'sentinel')],
                 costs(**{'forward.gpu_tail':1}))
    sentinel = result['requests']['sentinel']
    assert sentinel['arrival_s'] == .5
    assert sentinel['enqueue_s'] == 1
    assert sentinel['input_queue_wait_s'] == .5
    assert sentinel['ttft_s'] == sentinel['first_s'] - .5


def test_empty_engine_can_advance_to_future_arrival_without_busy_loop():
    result = run([Request('r', 4, 1, 50)])
    assert result['requests']['r']['enqueue_s'] == 50
    assert result['cpu_drained_s'] == 50


def test_late_foreground_does_not_artificially_keep_background_alive():
    result = run([Request('bg',4,1,role='background'),Request('fg',4,1,1)])
    assert not result['background_outlives_foreground']
    assert result['requests']['bg']['finished_s'] < result['requests']['fg']['enqueue_s']
    assert not result['model_accepted']


def test_background_barrier_requires_all_first_commits_and_keeps_old_batch():
    requests = [Request(f'background-{i}', 1024, 8, role='background') for i in range(16)]
    result = run(requests + [Request('foreground', 4, 1)], costs(**{'forward.gpu_tail':1}))
    outcomes = result['requests']
    assert result['release_s'] == max(outcomes[r.id]['first_s'] for r in requests)
    assert outcomes['foreground']['enqueue_s'] == result['release_s']
    arrivals = [a['request_id'] for a in result['actions'] if a['operation'] == 'arrival']
    assert arrivals[:16] == sorted(r.id for r in requests)
    barrier = next(i for i,a in enumerate(result['actions']) if a['operation'] == 'barrier')
    assert result['actions'][barrier-2]['operation'] == 'reap'
    assert result['actions'][barrier-2]['occupied_slots'] == 1


def test_mode_and_padding_come_from_predicted_work():
    dispatcher = dict(schema='v2-dispatch-candidates-v1', graphs_captured=True,
                      candidates={'1':[dict(mode='FULL',num_tokens=4,num_reqs=4,
                          uniform_token_count=1,max_query_len=1,num_active_loras=0,num_ubatches=1)]})
    result = simulate([Request('r',4,2)],costs(),dispatcher)
    assert [(s['features']['mode'],s['features']['padding_excess']) for s in result['steps']] == [('NONE',0),('FULL',3)]


def test_running_first_fcfs_and_sequence_limit():
    result = run([Request('a',4,2),Request('b',4,1)], scope=Scope(max_seqs=1))
    assert [[r['request_id'] for r in s['work']] for s in result['steps']] == [['a'],['a'],['b']]


def test_block_rounded_kv_pressure_fails_instead_of_preempting():
    with pytest.raises(ValueError, match='KV capacity'):
        run([Request('a',17,1),Request('b',1,1)],scope=Scope(kv_blocks=2))
    result = run([Request('a',17,1),Request('b',1,1)],scope=Scope(kv_blocks=3))
    assert result['peak_kv_blocks'] == 3


@pytest.mark.parametrize('value', [-1, float('inf'), float('nan'), True])
def test_invalid_costs_and_offsets_fail_closed(value):
    with pytest.raises(ValueError):
        law(value)
    with pytest.raises(ValueError):
        Request('r',1,1,value)


def test_invalid_schema_scope_context_and_iteration_bounds():
    with pytest.raises(ValueError, match='feature'):
        CostLaw({'observed_held_duration':1})
    with pytest.raises(ValueError, match='two-slot'):
        Scope(batch_slots=1)
    with pytest.raises(ValueError, match='context'):
        run([Request('r',8192,1)])
    with pytest.raises(ValueError, match='Duplicate'):
        run([Request('r',1,1),Request('r',1,1)])
    with pytest.raises(RuntimeError, match='iteration'):
        run([Request('r',1,3)],max_iterations=1)
    with pytest.raises(ValueError, match='V2'):
        simulate([Request('r',1,1)],costs(),{})
    with pytest.raises(TypeError):
        simulate([Request('r',1,1)],costs(),EAGER,release_state={})


def test_affine_costs_are_evaluated_from_own_work_and_round_trip():
    base = costs()
    launches = {k:dict(parts) for k,parts in base.launches.items()}
    launches['forward']['gpu_tail'] = CostLaw(dict(bias=1,prefill=.5,attention_pairs=.1))
    model = PipelineCosts(launches,base.scalar,base.provenance)
    result = run([Request('r',4,1)],model)
    assert result['nodes'][result['steps'][0]['forward_node']]['service_s'] == 4
    assert PipelineCosts.from_dict(model.to_dict()).to_dict() == model.to_dict()
    with pytest.raises(TypeError):
        model.launches['forward']['gpu_tail'].coefficients['bias'] = 9
    launches['forward']['gpu_tail'] = law(99)
    assert model.launches['forward']['gpu_tail'].coefficients['bias'] == 1


def test_all_declared_cases_drain_deterministically_with_synthetic_costs():
    cases = json.loads((ROOT/'experiments/prefill-service-v1.cases.json').read_text())['cases']
    model = PipelineCosts.from_dict(json.loads((ROOT/'experiments/prefill-async-synthetic-costs-v1.json').read_text()))
    assert len(cases) == 22
    for case in cases:
        result = rollout(case,model,EAGER)
        assert result['simulation_complete'] and result['peak_slots'] == 2
        assert not any(result[k] for k in ('model_accepted','cuda_scope_accepted','complete_experiment_passed','adaptive_cap_accepted','mpc_accepted'))
        assert all(r['output_tokens'] == (256 if r['role']=='background' else 128) for r in result['requests'].values())
        assert all(sum(r['tokens'] for r in s['work']) <= 2048 for s in result['steps'])
        # Every graph constraint holds; no parent sample span is added to leaves.
        for node in result['nodes']:
            assert node['start_s'] >= node['ready_s']
            assert all(result['nodes'][d]['end_s'] <= node['start_s'] for d in node['dependencies'])
    case = cases[0]
    assert rollout(case,model,EAGER) == rollout(case,model,EAGER)


def test_case_adapter_uses_sentinel_length_and_rejects_misalignment():
    case = dict(background_decoders=0,foreground_tokens=[4],foreground_offsets_ms=[0],
                sentinel=dict(input_tokens=7,arrival_ms=10))
    result = rollout(case,costs(),EAGER)
    assert result['requests']['sentinel']['arrival_s'] == .01
    sentinel_work = [r for s in result['steps'] for r in s['work'] if r['request_id']=='sentinel']
    assert sentinel_work[0]['tokens'] == 7
    with pytest.raises(ValueError, match='align'):
        rollout(dict(case,foreground_offsets_ms=[]),costs(),EAGER)


def test_cli_requires_fresh_output_and_keeps_unfitted_identity(tmp_path,capsys):
    from scripts.predict_prefill_async import main
    args = ['--synthetic-demo','--case','single-L1024-D0','--output',str(tmp_path/'demo')]
    assert main(args) == 0
    report = json.loads((tmp_path/'demo/predictions.json').read_text())
    assert report['status'] == 'unfitted_prototype' and not report['model_accepted']
    assert not report['measured_timings_used'] and report['runtime_source']=='synthetic_eager_assumptions'
    assert 'UNFITTED' in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        main(args)
    with pytest.raises(SystemExit):
        main(['--synthetic-demo','--case','unknown','--output',str(tmp_path/'other')])


def native_shaped_capture(result):
    """Synthetic evaluation fixture; deliberately NOT a native GPU result."""
    steps, workers, pipeline = [], [], []
    for step in result['steps']:
        row = step['features']
        steps.append(dict(step_id=step['step_id'],work=copy.deepcopy(step['work']),
                          state_before=dict(requests=copy.deepcopy(step['state_before'])),
                          **{f:row[f] for f in ('prefill','decode','decode_context','attention_pairs')}))
        workers.append(dict(step_id=step['step_id'],runtime_mode=row['mode'],
                            padded_tokens=row['prefill']+row['decode']+row['padding_excess']))
        pipeline.append(dict(step_id=step['step_id'],committed_outputs={rid:1 for rid in step['emits']}))
    return dict(status='complete',steps=steps,workers=[dict(rows=workers,pipeline=dict(rows=pipeline))],
                requests=copy.deepcopy(result['requests']))


@pytest.mark.parametrize('field',['work','state','features','dispatcher','identities','outputs','progression','status'])
def test_independent_comparison_rejects_structural_corruption(field):
    from prefill_lab.async_predictor import compare_structure
    prediction = run([Request('r',4,2)])
    capture = native_shaped_capture(prediction)
    assert compare_structure(prediction,capture)['matches']
    if field == 'work':capture['steps'][0]['work'][0]['tokens'] += 1
    if field == 'state':capture['steps'][0]['state_before']['requests']['r']['placeholders'] += 1
    if field == 'features':capture['steps'][0]['attention_pairs'] += 1
    if field == 'dispatcher':capture['workers'][0]['rows'][0]['runtime_mode'] = 'FULL'
    if field == 'identities':capture['workers'][0]['rows'].append(capture['workers'][0]['rows'][0])
    if field == 'outputs':capture['requests']['r']['output_tokens'] += 1
    if field == 'progression':capture['workers'][0]['pipeline']['rows'][0]['committed_outputs'] = {}
    if field == 'status':capture['status'] = 'invalid'
    assert not compare_structure(prediction,capture)['matches']
    assert prediction['requests']['r']['output_tokens'] == 2


def test_structure_comparison_ignores_all_captured_clock_labels():
    from prefill_lab.async_predictor import compare_structure
    prediction = run()
    capture = native_shaped_capture(prediction)
    before = compare_structure(prediction,capture)
    capture['release'] = -10000
    capture['steps'][0].update(start=float('nan'),end=float('inf'))
    capture['requests']['r'].update(first=-99,finished=-98)
    assert compare_structure(prediction,capture) == before


def runtime_metadata():
    return dict(async_scheduling=True,prefix_cache=False,speculative=False,
        max_num_scheduled_tokens=None,max_num_partial_prefills=None,max_long_partial_prefills=None,
        tensor_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1,
        enable_chunked_prefill=True,long_prefill_token_threshold=0,scheduler_policy='fcfs',
        kv_transfer=False,max_concurrent_batches=2,process_id=7,
        executor_class='vllm.v1.executor.uniproc_executor.UniProcExecutor',
        scheduler_class='vllm.v1.core.sched.async_scheduler.AsyncScheduler',
        max_num_batched_tokens=2048,max_num_seqs=32,max_model_len=8192,block_size=16,num_gpu_blocks=5594,
        workers=[dict(adapter='V2',pid=7,source=dict(version='0.30.0'),dispatch_table=EAGER)])


@pytest.mark.parametrize('key,value', [('speculative',True),('prefix_cache',True),
    ('max_concurrent_batches',3),('tensor_parallel_size',2),('scheduler_policy','priority'),
    ('async_scheduling',1),('kv_transfer',None),('long_prefill_token_threshold',512)])
def test_runtime_metadata_rejects_unsupported_scope(key,value):
    from scripts.predict_prefill_async import runtime_inputs
    metadata = runtime_metadata()
    scope,dispatcher = runtime_inputs(metadata)
    assert scope == Scope() and dispatcher == EAGER
    metadata[key] = value
    with pytest.raises(ValueError,match='runtime'):
        runtime_inputs(metadata)


def test_archived_validation_requires_exact_input_bytes(tmp_path):
    from scripts.validate_prefill_async import validate
    (tmp_path/'runtime.json').write_text('{}')
    with pytest.raises(ValueError,match='identity mismatch: runtime.json'):
        validate(tmp_path)


def test_cli_retains_failed_structure_and_returns_nonzero(tmp_path,capsys):
    from scripts.predict_prefill_async import main
    case = json.loads((ROOT/'experiments/prefill-service-v1.cases.json').read_text())['cases'][0]
    model = PipelineCosts.from_dict(json.loads((ROOT/'experiments/prefill-async-synthetic-costs-v1.json').read_text()))
    capture = native_shaped_capture(rollout(case,model,EAGER))
    runtime_path, capture_path = tmp_path/'runtime.json',tmp_path/'capture.json'
    runtime_path.write_text(json.dumps(runtime_metadata()))
    capture_path.write_text(json.dumps(capture))
    args = ['--synthetic-demo','--runtime',str(runtime_path),'--case',case['id'],
            '--compare-structure',str(capture_path),'--output']
    assert main(args+[str(tmp_path/'good')]) == 0
    capture['workers'][0]['rows'][0]['runtime_mode'] = 'FULL'
    capture_path.write_text(json.dumps(capture))
    assert main(args+[str(tmp_path/'bad')]) == 1
    report = json.loads((tmp_path/'bad/predictions.json').read_text())
    assert not report['predictions'][case['id']]['structure_comparison']['matches']
    assert not report['model_accepted']
