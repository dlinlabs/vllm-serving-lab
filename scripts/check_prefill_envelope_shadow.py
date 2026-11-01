"""Rebuild short shadow measurements from closed raw evidence, never accept a model."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prefill_lab import envelope_shadow as shadow, live_http
from prefill_lab.contracts import digest, finite, save


def native_evidence(root, runtime):
    from scripts.run_prefill_service import code_identity
    from scripts.run_linux_cpu_tests import validate_report
    load = lambda name: json.loads((root/name).read_text())
    source = load('collection-sources.json')
    launch = load('launch.json')
    acceptance = load('native-cpu-acceptance/acceptance.json')
    if (source != code_identity() or launch['source_sha256'] != source
            or launch['cpu_acceptance'] != acceptance or acceptance['source_sha256'] != source
            or acceptance['valid'] is not True or acceptance['tracked_clean'] is not True
            or launch['scope'] != 'envelope_shadow'):
        raise ValueError('Shadow native source/CPU identity differs')
    if validate_report(root/'native-cpu-acceptance/junit.xml')['test_cases'] != acceptance['tests']:
        raise ValueError('Shadow native CPU denominator differs')
    environment = launch['environment']
    expected = dict(ENVELOPE_SHADOW='1', LIVE_STATE='1', ENTRY_HTTP='0', QUEUE_CONFIRMATION='0',
                    QUEUE_PROBE='0', STAGES='0', EFFECTIVE_CALIBRATION='0', DELAYED_ARRIVALS='0', EVENT_CLOCK='0')
    if environment.get('PREFILL_LAB_ENABLED') != '1' or environment.get('VLLM_PLUGINS') != 'prefill_lab' or any(
            environment.get('PREFILL_LAB_'+key) != value for key, value in expected.items()):
        raise ValueError('Shadow launch mode differs')
    pins = json.loads((ROOT/'experiments/prefill-vllm-v030-source-map.json').read_text())['sha256']
    if any(s['version'] != '0.30.0' or s['files'] != pins for s in
           [runtime['source'], *[w['source'] for w in runtime['workers']]]) or any(w.get('pipeline_stages') for w in runtime['workers']):
        raise ValueError('Shadow installed source/stages differ')
    return source


def failed_row(trial, reason):
    return dict(**{k: trial[k] for k in ('trial_id', 'case', 'stage', 'block', 'observer')},
                measurement_valid=False, available=False, covered=False, narrow=False, useful=False,
                fallback_required=True, reason=reason, **shadow.flags())


def run(directory):
    from prefill_lab.causal import dispatcher_parity
    from prefill_lab.suite import runtime_preflight
    root = Path(directory)
    load = lambda name: json.loads((root/name).read_text())
    p = shadow.protocol()
    summary = load('shadow-summary.json')
    manifest = load('http-token-manifest.json')
    runtime = load('runtime.json')
    if load('shadow-plan.json') != p:
        raise ValueError('Shadow frozen plan differs')
    shadow.validate_manifest(manifest)
    if load('token-manifest.json') != manifest:
        raise ValueError('Shadow native/client manifests differ')
    runtime_preflight(runtime, manifest)
    source = native_evidence(root, runtime)
    config = load('shadow-config.json')
    seal = load('shadow-seal.json')
    if config['manifest_sha256'] != digest(manifest) or config['support'] != shadow.resolve_support(load('support-input.json'), runtime):
        raise ValueError('Shadow support/manifest identity differs')
    shadow.validate_config(config, runtime, now=seal['at_monotonic'], domain=config['clock_domain'], sources=source)
    if (set(seal) != {'at_monotonic', 'config_sha256', 'completed', 'source_sha256'}
            or seal['completed'] != [] or seal['config_sha256'] != config['sha256']
            or seal['source_sha256'] != source or not finite(seal['at_monotonic'])
            or load('shadow-configured.json') != dict(configured=True, frozen_sha256=config['sha256'])):
        raise ValueError('Shadow seal/native acknowledgement differs')
    attempted, completed = summary['attempted'], summary['completed']
    planned = [t['trial_id'] for t in p['order']]
    if (attempted != planned[:len(attempted)] or completed != attempted[:len(completed)]
            or len(attempted)-len(completed) not in (0, 1)
            or any(summary.get(k) is not False for k in shadow.flags())
            or summary['collection_complete'] != (completed == planned and not summary['failures'])
            or not summary['collection_complete'] and not summary['failures']):
        raise ValueError('Shadow completion/prefix verdict differs')
    allowed = {tid+suffix for tid in attempted for suffix in ('.json', '-client.json')}
    if {f.name for f in root.glob('es-*.json')} - allowed:
        raise ValueError('Unexpected shadow raw trials')
    journal = [json.loads(line) for line in (root/'gateway.jsonl').read_text().splitlines() if line.strip()]
    rows, errors, previous, offered, interrupted_ids = [], [], seal['at_monotonic'], set(), set()
    for trial in p['order'][:len(attempted)]:
        tid = trial['trial_id']
        expected = {digest(['live-http', tid, r['id']])[:32] for r in manifest['trials'][tid]}
        cap_path, client_path = root/(tid+'.json'), root/(tid+'-client.json')
        if not cap_path.is_file() or not client_path.is_file():
            if tid in completed:
                raise ValueError('Completed shadow trial is missing raw evidence')
            rows.append(failed_row(trial, 'interrupted_trial'))
            errors.append(dict(trial_id=tid, message='Incomplete interrupted raw evidence'))
            if client_path.is_file():
                ids = {r['trace_id'] for r in load(client_path.name)['rows'] if 'sent' in r}
                if not ids <= expected:
                    raise ValueError('Unexpected interrupted shadow client IDs')
                offered.update(ids)
            interrupted_ids.update(expected)
            continue
        cap, client = load(cap_path.name), load(client_path.name)
        ids = {r['trace_id'] for r in client['rows'] if 'sent' in r}
        if not ids <= expected:
            raise ValueError('Unexpected shadow client request IDs')
        offered.update(ids)
        if (not all(finite(cap.get(k)) for k in ('start', 'end'))
                or not previous <= cap['start'] <= cap['end']):
            raise ValueError('Shadow trial/seal chronology differs')
        previous = cap['end']
        bundle = dict(capture=cap, client=client, gateway=[r for r in journal if r.get('request_id') in ids])
        try:
            if trial['observer'] == 'light' and not dispatcher_parity(cap, runtime['workers'][0]['dispatch_table'])['matches']:
                raise ValueError('Shadow dispatcher differs')
            rows.append(dict(shadow.extract(bundle, trial, manifest, runtime, config), measurement_valid=True))
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(dict(trial_id=tid, message=type(exc).__name__+': '+str(exc)))
            rows.append(failed_row(trial, 'invalid_raw_evidence'))
    if summary['collection_complete'] and load('runtime-final.json') != runtime:
        raise ValueError('Shadow final runtime differs')
    start, final, status = load('gateway-start.json'), load('gateway-final.json'), load('telemetry-status.json')
    run_id = live_http.gateway_contract(start)
    if any(start.get(k) != 0 for k in ('accepted_requests', 'rejected_requests', 'failed_requests', 'current_in_flight', 'current_waiting')):
        raise ValueError('Shadow gateway was not fresh/drained')
    if run_id != live_http.gateway_contract(final) or any(r.get('run_id') != run_id for r in journal):
        raise ValueError('Shadow gateway identity differs')
    received = [r['request_id'] for r in journal if r.get('event') == 'received']
    if len(received) != len(set(received)) or not set(received) <= offered | interrupted_ids:
        raise ValueError('Unexpected/duplicate shadow gateway traffic')
    accepted = [r['request_id'] for r in journal if r.get('event') == 'accepted']
    if len(accepted) != len(set(accepted)) or not set(accepted) <= set(received) or final.get('accepted_requests') != len(accepted):
        raise ValueError('Shadow gateway accepted/journal totals differ')
    if summary['collection_complete']:
        live_http.gateway_drained(final, p)
        if set(received) != offered:
            raise ValueError('Shadow gateway/client offer inventory differs')
    if (status.get('run_id') != run_id or status.get('status') != 'closed'
            or status.get('dropped', 1) or status.get('write_errors', 1)
            or not status.get('diagnostics', {}).get('complete')):
        raise ValueError('Shadow telemetry did not close cleanly')
    report = shadow.score(rows, attempted, [*summary['failures'], *errors])
    report.update(integrity_errors=errors, plan_sha256=digest(p), config_sha256=config['sha256'],
                  support_origin=config['support']['origin'], source_sha256=source)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.capture)
    save(args.capture/'shadow-report.json', report)
    print(json.dumps({k: report[k] for k in ('collection_complete', 'planned_trials', 'attempted_trials', 'measured_trials', *shadow.flags())}))
    if not report['collection_complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
