"""Reproduce synthetic smoke checks and frozen ab24416 scheduling comparisons.

This reads archived metadata/work only, never fits from captured timing labels.
The prior native-stage analyzer establishes capture integrity/source provenance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prefill_lab.async_predictor import PipelineCosts, rollout, compare_structure
from prefill_lab.contracts import digest, save
from scripts.predict_prefill_async import runtime_inputs

PAIRS = [('trial-014','single-L7168-D16'), ('trial-007','composition-mixed-D16'),
         ('trial-211','single-L7168-D16'), ('trial-212','composition-mixed-D16')]


def validate(evidence):
    evidence = Path(evidence)
    index_path = ROOT / 'results/prefill-stages-ab24416-analysis/input-files.json'
    expected = {r['path']: r for r in json.loads(index_path.read_text())}
    inputs = {}
    for name in ['runtime.json'] + [tid+'.json' for tid,_ in PAIRS]:
        raw = (evidence / name).read_bytes()
        identity = dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        if identity != expected[name]:
            raise ValueError(f'Archived input identity mismatch: {name}')
        inputs[name] = identity
    runtime = json.loads((evidence / 'runtime.json').read_text())
    scope, dispatcher = runtime_inputs(runtime)
    model_path = ROOT / 'experiments/prefill-async-synthetic-costs-v1.json'
    costs = PipelineCosts.from_dict(json.loads(model_path.read_text()))
    cases = json.loads((ROOT / 'experiments/prefill-service-v1.cases.json').read_text())['cases']
    predictions, summaries = {}, {}
    for case in cases:
        result = rollout(case, costs, dispatcher, scope=scope)
        predictions[case['id']] = result
        summaries[case['id']] = dict(
            simulation_complete=result['simulation_complete'], executed_batches=len(result['steps']),
            peak_slots=result['peak_slots'], peak_kv_blocks=result['peak_kv_blocks'],
            background_outlives_foreground=result['background_outlives_foreground'],
            forced_outputs_match=all(r['output_tokens'] == (256 if r['role']=='background' else 128)
                                     for r in result['requests'].values()),
            within_token_budget=all(sum(r['tokens'] for r in s['work']) <= scope.token_budget
                                    for s in result['steps']),
            no_acceptance_granted=not any(result[k] for k in ('model_accepted','cuda_scope_accepted',
                'complete_experiment_passed','adaptive_cap_accepted','mpc_accepted')))
    comparisons = {tid:dict(case=cid, **compare_structure(predictions[cid],
                    json.loads((evidence / (tid+'.json')).read_text()))) for tid,cid in PAIRS}
    source_paths = ['prefill_lab/async_predictor.py','scripts/predict_prefill_async.py',
                    'scripts/validate_prefill_async.py','tests/test_prefill_async_predictor.py',
                    'experiments/prefill-async-synthetic-costs-v1.json',
                    'experiments/prefill-async-predictor-v1.md']
    source_map = json.loads((ROOT/'experiments/prefill-vllm-v030-source-map.json').read_text())
    return dict(schema='async-prototype-development-validation-v1',
                prototype_checks_passed=all(c['matches'] for c in comparisons.values())
                    and all(s['simulation_complete'] and s['forced_outputs_match']
                            and s['within_token_budget'] and s['no_acceptance_granted'] for s in summaries.values()),
                native_capture_commit='ab24416334174f31c0c8b686ff4d0e496f86a969',
                native_capture='1792991194546315502', captured_timing_labels_used=False,
                model_accepted=False, cuda_scope_accepted=False, complete_experiment_passed=False,
                cost_sha256=digest(costs.to_dict()), dispatcher_sha256=digest(dispatcher),
                cases_sha256=digest(cases),
                scope=dict(vars(scope)), input_files=list(inputs.values()),
                reviewed_upstream_sources={p:source_map['sha256'][p] for p in
                    ('v1/engine/core.py','v1/core/sched/scheduler.py','v1/core/sched/async_scheduler.py')},
                sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths},
                synthetic_cases=summaries, development_structure_comparisons=comparisons)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True, help='Original ab24416 stage diagnostic directory')
    parser.add_argument('--output', type=Path, required=True, help='New output file')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output already exists')
    result = validate(args.evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, result)
    print(f'CPU prototype checks: {result["prototype_checks_passed"]}; physical model accepted: False')
    return 0 if result['prototype_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
