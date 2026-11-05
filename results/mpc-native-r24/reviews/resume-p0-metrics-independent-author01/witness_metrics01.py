"""Preregistered literal-row witness W1-W4 for r24-resume-p0-metrics-independent01 (see PLAN-WITNESS01.txt).

Literal synthetic rows only: no real, Train, Eval, Native or journal data; no network; writes nothing.
Loads the capsule's hash-verified byte copy of nominal_metrics.py; its pinned score_http.py copy resolves at
witness-src/native-r23/score_http.py (same relative layout as ROOT).
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
METRICS = HERE / 'witness-src' / 'resume-mvp' / 'nominal_metrics.py'
SCORE = HERE / 'witness-src' / 'native-r23' / 'score_http.py'
PINS = {METRICS: 'e0119f34437f81dae4c8bb97501a2c0e6d7ec5decfa23f0eb6491db6e6afa58e',
        SCORE: '9a173ab99e668534dfe18df9be0b9e8447e06a1a673f63aebb16b9ef5217081d'}


def keys_recursive(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from keys_recursive(item)
    elif isinstance(value, list):
        for item in value:
            yield from keys_recursive(item)


def slo(ignore=False):
    return {'status': 'complete', 'finish_reason': 'stop', 'requested_ignore_eos': ignore, 'stream_done': True,
            'stream_error': False}


def strings_recursive(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from strings_recursive(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings_recursive(item)


def complete_row(rid, sched, sent, first, terminal, ignore=False):
    return {'request_id': rid, 'group_tokens': 300, 'max_output_tokens': 128, 'scheduled_at': sched,
            'sent_at': sent, 'first_content_at': first, 'terminal_at': terminal, 'fate': 'complete',
            'completion_tokens': 5, 'slo': slo(ignore)}


def good_row(rid, sched, sent, ignore=False):
    # first content 0.125 s after scheduled_at, terminal 0.125 s later, 5 tokens (dyadic, exact in binary64)
    return complete_row(rid, sched, sent, sched + 0.125, sched + 0.25, ignore)


def run(nm, rows, arm, purpose, sha, extra_provenance=None, clock_domain='witness-literal'):
    control = ({'kind': 'static_fixed', 'budget': nm.FIXED_BUDGETS[arm], 'basis': 'witness literal static control'}
               if arm in nm.FIXED_BUDGETS else None)
    provenance = {'offered_trace_sha256': sha}
    provenance.update(extra_provenance or {})
    return {'arm': arm, 'purpose': purpose, 'clock_domain': clock_domain, 'offered_start': 10.0,
            'offered_duration_s': 5.0, 'envelope_start': 10.0, 'envelope_end': 15.0,
            'envelope_scope': 'offered_cohort_observed', 'rows': rows, 'slo_missing': None, 'control': control,
            'control_missing': None if control is not None else 'witness: no controller journal',
            'provenance': provenance}


def w1(nm):
    rows = [good_row('w1-early', 11.0, 10.5), good_row('w1-ontime-a', 12.0, 12.0),
            good_row('w1-ontime-b', 13.0, 13.0)]
    try:
        metrics = nm.compute(run(nm, rows, 'strong_one_move', 'train', 'w1' * 32))
    except nm.MetricsInputError as exc:
        return {'outcome': 'REFUSED', 'error': str(exc), 'confirmed': False}
    lag = metrics['send_lag_s']
    limit = metrics['natural_slo']['rule']['ttft_limit_s']
    early_from_send = rows[0]['first_content_at'] - rows[0]['sent_at']
    ttft = metrics['ttft_s']
    observed = {
        'outcome': 'ACCEPTED', 'natural_slo_good': metrics['natural_slo']['natural_slo_good'],
        'early_row_scheduled_at': rows[0]['scheduled_at'], 'early_row_sent_at': rows[0]['sent_at'],
        'early_row_first_minus_scheduled_s': rows[0]['first_content_at'] - rows[0]['scheduled_at'],
        'early_row_first_minus_sent_s': early_from_send, 'ttft_limit_s': limit,
        'send_lag': {key: lag[key] for key in ('n', 'p50', 'p95', 'p99', 'max', 'n_not_sent')},
        'send_lag_keys': sorted(lag),
        'conditional_scheduled_p99': ttft['conditional_scheduled']['p99'],
        'conditional_client_send_p99': ttft['conditional_client_send']['p99'],
    }
    credited = observed['natural_slo_good'] == 3 and early_from_send > limit
    invisible = (all(lag[key] == 0.0 for key in ('p50', 'p95', 'p99', 'max'))
                 and not any(key == 'min' or 'negative' in key or 'early' in key or 'before' in key for key in lag))
    observed.update(credited_despite_send_origin_ttft_over_limit=credited, early_send_invisible_in_send_lag=invisible,
                    client_send_p99_exceeds_scheduled_p99=(observed['conditional_client_send_p99']
                                                           > observed['conditional_scheduled_p99']),
                    confirmed=credited and invisible)
    return observed


def w2(nm):
    sha = 'w2' * 32
    # same request id on both arms (as on one real offered trace); only conditions outside the identity differ
    f16 = nm.compute(run(nm, [good_row('w2-r0', 11.0, 11.0)], 'fixed16', 'train', sha,
                         {'manifest': {'host': 'witness-host-A', 'gpu': 'witness-gpu-A', 'models': 'witness-model-A'}}))
    f32 = nm.compute(run(nm, [good_row('w2-r0', 11.0, 11.0, ignore=True)], 'fixed32', 'train', sha,
                         {'manifest': {'host': 'witness-host-B', 'gpu': 'witness-gpu-B', 'models': 'witness-model-B'}},
                         clock_domain='witness-literal-other-clock'))
    selection = nm.select_train_fixed({'fixed16': f16, 'fixed32': f32})
    observed = {'selection_status': selection['status'], 'selected_arm': selection.get('selected_arm'),
                'decided_by': selection.get('decided_by'), 'selection_reason': selection.get('reason'),
                'provenances_differ': f16['provenance'] != f32['provenance'],
                'clock_domains_differ': f16['clock_domain'] != f32['clock_domain'],
                'requested_ignore_eos': {'fixed16': False, 'fixed32': True},
                'natural_slo_good': {'fixed16': f16['natural_slo']['natural_slo_good'],
                                     'fixed32': f32['natural_slo']['natural_slo_good']}}
    try:
        table = nm.comparison_table({'fixed16': {'status': 'MEASURED', 'metrics': f16},
                                     'fixed32': {'status': 'MEASURED', 'metrics': f32}},
                                    purpose='train', train_selection=selection)
    except nm.MetricsInputError as exc:
        observed.update(table_outcome='REFUSED', table_error=str(exc), confirmed=False)
        return observed
    markers = sorted({key for key in keys_recursive(table)
                      if any(m in key.lower() for m in ('host', 'gpu', 'hardware', 'comparab', 'ignore_eos', 'model',
                                                        'resource'))})
    mentions = sorted({text for text in strings_recursive(table)
                       if 'comparab' in text.lower() or 'hardware' in text.lower()})
    observed.update(table_outcome='ACCEPTED', table_comparability_markers=markers,
                    table_strings_mentioning_comparability_or_hardware=mentions,
                    table_identity=[table['offered_trace_sha256'], table['all_offered'], table['envelope_scope']])
    observed['confirmed'] = (selection['status'] == 'SELECTED' and selection.get('selected_arm') == 'fixed16'
                             and selection.get('decided_by') == 'primary_natural_slo_good' and not markers
                             and not mentions and observed['provenances_differ'] and observed['clock_domains_differ'])
    return observed


def w3(nm):
    sha = 'w3' * 32
    t16 = nm.compute(run(nm, [good_row('w3-t16', 11.0, 11.0)], 'fixed16', 'train', sha))
    t32 = nm.compute(run(nm, [good_row('w3-t32', 11.0, 11.0)], 'fixed32', 'train', sha))
    selection = nm.select_train_fixed({'fixed16': t16, 'fixed32': t32})
    e16_rows, e32_rows = [good_row('w3-e16', 11.0, 11.0)], [good_row('w3-e32', 11.0, 11.0)]
    e16 = nm.compute(run(nm, e16_rows, 'fixed16', 'eval', sha))
    e32 = nm.compute(run(nm, e32_rows, 'fixed32', 'eval', sha))
    observed = {'selection_status': selection['status'], 'selected_arm': selection.get('selected_arm'),
                'decided_by': selection.get('decided_by'),
                'selection_offered_trace_sha256': selection.get('offered_trace_sha256')}
    try:
        table = nm.comparison_table({'fixed16': {'status': 'MEASURED', 'metrics': e16},
                                     'fixed32': {'status': 'MEASURED', 'metrics': e32}},
                                    purpose='eval', train_selection=selection)
    except nm.MetricsInputError as exc:
        observed.update(table_outcome='REFUSED', table_error=str(exc), confirmed=False)
        return observed
    alias = next(row for row in table['rows'] if row['arm'] == 'train_selected_fixed')
    carried = sorted({key for key in keys_recursive(table)
                      if any(m in key for m in ('decided_by', 'candidates', 'selection', 'selected'))})
    same = selection.get('offered_trace_sha256') == table['offered_trace_sha256']
    observed.update(table_outcome='ACCEPTED', table_purpose=table['purpose'],
                    table_offered_trace_sha256=table['offered_trace_sha256'],
                    selection_sha_equals_eval_table_sha=same, alias_status=alias['status'],
                    alias_of=alias['alias_of'], alias_keys=sorted(alias), selection_provenance_keys_in_table=carried)
    observed['confirmed'] = (selection['status'] == 'SELECTED' and same and alias['status'] == 'ALIAS'
                             and alias['alias_of'] == selection.get('selected_arm') and not carried)
    # secondary (recorded separately): a hand-made minimal SELECTED dict with no rule/candidates/sha
    try:
        forged = nm.comparison_table({'fixed16': {'status': 'MEASURED', 'metrics': e16},
                                      'fixed32': {'status': 'MEASURED', 'metrics': e32}},
                                     purpose='eval', train_selection={'status': 'SELECTED', 'selected_arm': 'fixed32'})
        forged_alias = next(row for row in forged['rows'] if row['arm'] == 'train_selected_fixed')
        observed.update(forged_minimal_selection_outcome='ACCEPTED', forged_alias_status=forged_alias['status'],
                        forged_alias_of=forged_alias['alias_of'],
                        forged_minimal_selection_accepted=(forged_alias['status'] == 'ALIAS'
                                                           and forged_alias['alias_of'] == 'fixed32'))
    except nm.MetricsInputError as exc:
        observed.update(forged_minimal_selection_outcome='REFUSED', forged_error=str(exc),
                        forged_minimal_selection_accepted=False)
    # secondary (F3): compared Eval arms carry different request-id sets under one sha and one count
    ids16 = sorted(row['request_id'] for row in e16_rows)
    ids32 = sorted(row['request_id'] for row in e32_rows)
    observed.update(eval_request_id_sets=[ids16, ids32],
                    f3_cross_arm_request_id_mismatch_accepted=(ids16 != ids32 and e16['all_offered'] == e32['all_offered']
                                                               == table['all_offered']))
    return observed


def w4(nm):
    # one row scheduled before offered_start (outside the offered window and envelope) and one row whose terminal
    # lies after envelope_end, under scope offered_cohort_observed
    rows = [good_row('w4-prewindow', 3.0, 3.0), complete_row('w4-late-tail', 14.0, 14.0, 14.125, 20.0)]
    the_run = run(nm, rows, 'strong_one_move', 'train', 'w4' * 32)
    try:
        metrics = nm.compute(the_run)
    except nm.MetricsInputError as exc:
        return {'outcome': 'REFUSED', 'error': str(exc), 'confirmed': False}
    windows, natural, throughput = metrics['windows'], metrics['natural_slo'], metrics['throughput']
    max_terminal = max(row['terminal_at'] for row in rows)
    min_scheduled = min(row['scheduled_at'] for row in rows)
    label_envelope_s = max(windows['offered_end'], max_terminal) - windows['offered_start']
    label = windows['envelope_scope_label']
    observed = {
        'outcome': 'ACCEPTED', 'all_offered': metrics['all_offered'], 'envelope_scope': windows['envelope_scope'],
        'offered_start': windows['offered_start'], 'offered_end': windows['offered_end'],
        'envelope_end': windows['envelope_end'], 'envelope_s': windows['envelope_s'],
        'min_scheduled_at': min_scheduled, 'max_terminal_at': max_terminal,
        'envelope_s_per_scope_label_definition': label_envelope_s,
        'natural_slo_good': natural['natural_slo_good'],
        'in_window_slo_goodput_rps': natural['in_window_slo_goodput_rps'],
        'wall_envelope_slo_goodput_rps': natural['wall_envelope_slo_goodput_rps'],
        'wall_envelope_delivered_rps': throughput['wall_envelope_delivered_rps'],
        'wall_envelope_delivered_rps_per_scope_label_definition': throughput['delivered'] / label_envelope_s,
        'scope_label_claims_last_terminal': 'later of offered end and last observed terminal' in label,
    }
    observed['confirmed'] = (max_terminal > windows['envelope_end'] and min_scheduled < windows['offered_start']
                             and natural['natural_slo_good'] == 1
                             and natural['in_window_slo_goodput_rps'] == 1 / 5.0
                             and throughput['wall_envelope_delivered_rps'] == 2 / 5.0
                             and observed['scope_label_claims_last_terminal'])
    return observed


def main():
    for path, expected in PINS.items():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            print(json.dumps({'status': 'PIN_MISMATCH', 'path': str(path), 'actual': actual}), flush=True)
            return 2
    spec = importlib.util.spec_from_file_location('nominal_metrics_witness_copy', METRICS)
    nm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nm)
    if nm.SCORE_HTTP_PATH != SCORE.resolve():
        print(json.dumps({'status': 'SCORE_PATH_MISMATCH', 'resolved': str(nm.SCORE_HTTP_PATH)}), flush=True)
        return 2
    results = {}
    for name, check in (('W1_early_send', w1), ('W2_comparability_identity', w2),
                        ('W3_selection_provenance', w3), ('W4_window_binding', w4)):
        try:
            results[name] = check(nm)
        except Exception as exc:  # preserved as a witness failure; never retried
            results[name] = {'outcome': 'ERROR', 'error': f'{type(exc).__name__}: {exc}', 'confirmed': False}
    report = {'schema': 'r24-resume-p0-metrics-independent01/witness-v1', 'python': sys.version.split()[0],
              'dont_write_bytecode': sys.dont_write_bytecode, 'module_sha256': PINS[METRICS],
              'score_http_sha256': PINS[SCORE], 'results': results,
              'all_confirmed': all(item.get('confirmed') is True for item in results.values()),
              'scope': 'literal synthetic rows; no real/Train/Eval/Native data; not a test of Native behaviour'}
    print(json.dumps(report, sort_keys=True, indent=1, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
