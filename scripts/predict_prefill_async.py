"""Run the CPU async prototype; never collect, fit, or accept a physical model."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prefill_lab.async_predictor import PipelineCosts, Scope, rollout, compare_structure
from prefill_lab.contracts import digest, save


def runtime_inputs(runtime):
    """Scope check metadata only; this is not the native source/acceptance gate."""
    required = dict(async_scheduling=True, prefix_cache=False, speculative=False,
                    tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1,
                    enable_chunked_prefill=True, long_prefill_token_threshold=0,
                    max_num_scheduled_tokens=None, max_num_partial_prefills=None,
                    max_long_partial_prefills=None,
                    scheduler_policy='fcfs', kv_transfer=False, max_concurrent_batches=2,
                    executor_class='vllm.v1.executor.uniproc_executor.UniProcExecutor',
                    scheduler_class='vllm.v1.core.sched.async_scheduler.AsyncScheduler')
    for key, expected in required.items():
        if type(runtime.get(key)) is not type(expected) or runtime[key] != expected:
            raise ValueError(f'Unsupported runtime field: {key}')
    workers = runtime['workers']
    if (len(workers) != 1 or workers[0]['adapter'] != 'V2'
            or workers[0]['pid'] != runtime['process_id']
            or workers[0]['source']['version'] != '0.30.0'):
        raise ValueError('Only same-process V2 v0.30.0 is in model scope')
    scope = Scope(token_budget=runtime['max_num_batched_tokens'], max_seqs=runtime['max_num_seqs'],
                  max_model_len=runtime['max_model_len'], block_size=runtime['block_size'],
                  kv_blocks=runtime['num_gpu_blocks'])
    return scope, workers[0]['dispatch_table']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    costs = parser.add_mutually_exclusive_group(required=True)
    costs.add_argument('--synthetic-demo', action='store_true', help='Illustrative costs; no measured timing data')
    costs.add_argument('--costs', type=Path, help='Explicit async-stage-costs-v1 JSON; remains unaccepted')
    parser.add_argument('--runtime', type=Path, help='Frozen runtime metadata/dispatcher, not timing labels')
    parser.add_argument('--case', default='all', help='Declared case ID or all')
    parser.add_argument('--output', type=Path, required=True, help='Fresh output directory')
    parser.add_argument('--timeline', action='store_true', help='Also retain generated work, states and resource nodes')
    parser.add_argument('--compare-structure', type=Path, help='One separately audited capture; post-prediction work/state comparison only')
    args = parser.parse_args(argv)
    if args.runtime is None and not args.synthetic_demo:
        parser.error('--costs also requires explicit --runtime metadata')
    document = json.loads((ROOT / 'experiments/prefill-service-v1.cases.json').read_text())
    cases = [c for c in document['cases'] if args.case in ('all', c['id'])]
    if not cases:
        parser.error('Unknown case ID')
    if args.compare_structure and len(cases) != 1:
        parser.error('--compare-structure requires exactly one --case')
    model = PipelineCosts.from_dict(json.loads((args.costs or ROOT / 'experiments/prefill-async-synthetic-costs-v1.json').read_text()))
    if args.runtime:
        runtime = json.loads(args.runtime.read_text())
        scope, dispatcher = runtime_inputs(runtime)
        runtime_source, runtime_sha = 'provided_metadata_only', digest(runtime)
    else:
        scope = Scope()
        dispatcher = dict(schema='v2-dispatch-candidates-v1', graphs_captured=False, candidates={})
        runtime_source, runtime_sha = 'synthetic_eager_assumptions', None
    args.output.mkdir(parents=True, exist_ok=False)
    predictions = {}
    for case in cases:
        result = rollout(case, model, dispatcher, scope=scope)
        summary = {k: v for k, v in result.items() if k not in ('nodes', 'steps', 'actions')}
        summary['executed_batches'] = len(result['steps'])
        summary['empty_batches'] = sum(a['operation'] == 'submit' and a['empty'] for a in result['actions'])
        predictions[case['id']] = summary
        if args.compare_structure:
            capture = json.loads(args.compare_structure.read_text())
            summary['structure_comparison'] = compare_structure(result, capture)
            summary['comparison_capture_sha256'] = digest(capture)
        if args.timeline:
            save(args.output / (case['id'] + '.json'), result)
    report = dict(schema='async-predictor-demo-v1', status='unfitted_prototype',
                  measured_timings_used=False, model_accepted=False,
                  runtime_source=runtime_source, runtime_sha256=runtime_sha,
                  costs=model.to_dict(), cases_sha256=digest(document), predictions=predictions)
    save(args.output / 'predictions.json', report)
    print(f'UNFITTED prototype: {len(predictions)} cases; {args.output / "predictions.json"}')
    matched = all(p.get('structure_comparison', {}).get('matches', True) for p in predictions.values())
    if args.compare_structure:
        print(f'Development structure matches: {matched}; physical model accepted: False')
    return 0 if matched else 1


if __name__ == '__main__':
    raise SystemExit(main())
