"""P0 nominal-lineage measured metrics: additive pure calculator plus a run_trace_pilot.py output adapter.

Scope: the UNQUALIFIED nominal demonstrator lineage r24-resume-p0-nominal-v1.  This is NOT the legacy L-T16
world: the legacy comparison16 scorer is neither modified nor reused, and no G06/G16/Native14 qualification is
inherited or claimed.  Only measured quantities are reported: no winner, capacity, SLO guarantee or production
claim.

* ``compute(run)`` -- pure core over strict normalized rows (``ROW_KEYS``); refuses, never repairs, bad input.
* ``adapt_run_trace_pilot(output_dir, arm=..., purpose=...)`` -- maps one run_trace_pilot.py output directory
  onto normalized rows with the exact field mapping in ``RAW_MAPPING``; refuses what the raw files cannot
  support (it never invents events, SLO fields or controller timings).
* ``select_train_fixed(...)`` -- Train-only empirical fixed16 vs fixed32 choice by unchanged natural-SLO count.
* ``comparison_table(...)`` -- one measured table per split, without any winner field.

Natural SLO uses the UNCHANGED r23 ``score_http.py`` (sha256 pinned; executed from verified bytes, no bytecode).
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import types

SCHEMA = 'r24-resume-p0-nominal-metrics-v1'
LINEAGE = 'r24-resume-p0-nominal-v1'
QUALIFICATION = ('UNQUALIFIED nominal demonstrator: not the legacy L-T16 world; no G06/G16/Native14 qualification '
                 'inherited or claimed')
FATES = ('complete', 'rejected', 'http_error', 'stream_error', 'transport_error', 'cancelled', 'not_launched',
         'unknown')
FAILURE_FATES = ('http_error', 'stream_error', 'transport_error', 'cancelled')
PURPOSES = ('train', 'eval')
CLAIM_SCOPE = {
    'train': 'Train split: in-sample engineering measurement only; no generalization claim',
    'eval': 'Eval split: procedural holdout of synthetic draws from one public generator; not an OOD, real-trace or '
            'Azure holdout',
}
ARMS = ('fixed16', 'fixed32', 'train_selected_fixed', 'strong_one_move', 'mpc', 'causal_reactive')
RUN_ARMS = tuple(arm for arm in ARMS if arm != 'train_selected_fixed')
FIXED_BUDGETS = {'fixed16': 16.0, 'fixed32': 32.0}
QUANTILES = (50, 95, 99)
GROUP_EDGES = (512, 2048)
GROUPS = ('le512', '513to2048', 'gt2048')
FIFO_LABEL = 'PENDING_VERIFIED_BACKEND_FCFS_CONFIG'
FIFO_NOTE = ('FIFO is a queue ordering, not a fourth algorithm, and does not imply any budget; the label stays pending '
             'until the actual backend FCFS configuration is verified (never derived from a CPU plant)')
SCORE_HTTP_PATH = Path(__file__).resolve().parent.parent / 'native-r23' / 'score_http.py'
SCORE_HTTP_SHA256 = '9a173ab99e668534dfe18df9be0b9e8447e06a1a673f63aebb16b9ef5217081d'
ROW_KEYS = frozenset({'request_id', 'group_tokens', 'max_output_tokens', 'scheduled_at', 'sent_at',
                      'first_content_at', 'terminal_at', 'fate', 'completion_tokens', 'slo'})
SLO_KEYS = frozenset({'status', 'finish_reason', 'requested_ignore_eos', 'stream_done', 'stream_error'})
RUN_KEYS = frozenset({'arm', 'purpose', 'clock_domain', 'offered_start', 'offered_duration_s', 'envelope_start',
                      'envelope_end', 'envelope_scope', 'rows', 'slo_missing', 'control', 'control_missing',
                      'provenance'})
ENVELOPE_SCOPES = {
    'offered_cohort_observed': 'offered start to the later of offered end and last observed terminal; EXCLUDES pre-offer '
                               'setup/solver warmup, post-terminal drain and unresolved requests: a lower bound on the '
                               'full wall, not the full wall',
    'full_wall': 'caller-recorded actual wall in the run clock domain including setup, solver, HTTP, unresolved '
                 'requests and final drain',
}
PENALIZED_ESTIMAND = ('failure-penalized operational TTFT over ALL offered requests (missing first content = +inf); '
                      'NOT a survival-analysis identified quantile of right-censored latency')
CONDITIONAL_ESTIMAND = 'TTFT conditional on fate complete (selection-biased; read with fates and the penalized TTFT)'
COHORT_LABEL = ('cohort-normalized (includes tail): all-offered cohort count / offered window seconds; not steady-state '
                'GPU throughput')
ENVELOPE_LABEL = 'actual wall envelope: count / (envelope_end - envelope_start)'
CLAIMS_NOT_MADE = (
    'no winner or superiority', 'no capacity, SLO guarantee or production claim',
    'no G06/G16/Native14 qualification', 'hashes are integrity pins, not Train/Eval disjointness or leakage proof',
    'FIFO label pending verified backend FCFS configuration',
)


class MetricsInputError(ValueError):
    """Normalized input violates the frozen schema: refused, never repaired."""


class AdapterRefusal(ValueError):
    """Raw run_trace_pilot output cannot support the normalized schema: refused, no events invented."""


_SCORE_HTTP = None


def score_http():
    """The unchanged r23 natural-SLO scorer, executed only from bytes matching the pinned sha256."""
    global _SCORE_HTTP
    if _SCORE_HTTP is None:
        data = SCORE_HTTP_PATH.read_bytes()
        if hashlib.sha256(data).hexdigest() != SCORE_HTTP_SHA256:
            raise RuntimeError('score_http.py differs from the pinned sha256; natural SLO scoring refused')
        module = types.ModuleType('r23_score_http_pinned')
        module.__file__ = str(SCORE_HTTP_PATH)
        exec(compile(data, str(SCORE_HTTP_PATH), 'exec'), module.__dict__)
        _SCORE_HTTP = module
    return _SCORE_HTTP


def _is_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _number(value, name):
    if not _is_number(value):
        raise MetricsInputError(f'{name} must be a finite int/float (bool refused)')
    return value


def _optional_number(value, name):
    return None if value is None else _number(value, name)


def _nonempty_str(value, name):
    if not isinstance(value, str) or not value:
        raise MetricsInputError(f'{name} must be a non-empty string')
    return value


def group_of(tokens):
    """Predeclared prompt-token groups (edges 512 and 2048 inclusive on the left group)."""
    if tokens <= GROUP_EDGES[0]:
        return GROUPS[0]
    return GROUPS[1] if tokens <= GROUP_EDGES[1] else GROUPS[2]


def percentile_block(values, estimand=None):
    """Nearest-rank ordered[ceil(q*n/100)-1] (integer rank arithmetic); a rank on +inf yields null with a reason."""
    ordered = sorted(values)
    n = len(ordered)
    block = {'n': n, 'method': 'nearest_rank_ceil'}
    if estimand is not None:
        block['estimand'] = estimand
    if not n:
        for q in QUANTILES:
            block.update({f'p{q}': None, f'p{q}_rank': None, f'p{q}_is_sample_max': None})
        block.update(max=None, n_failure_penalty=0, null_reason='no_samples')
        return block
    block['n_failure_penalty'] = sum(value == math.inf for value in ordered)
    for q in QUANTILES:
        rank = (q * n + 99) // 100
        value = ordered[rank - 1]
        block[f'p{q}'] = value if math.isfinite(value) else None
        block[f'p{q}_rank'] = rank
        block[f'p{q}_is_sample_max'] = rank == n
        if not math.isfinite(value):
            block[f'p{q}_null_reason'] = 'rank_falls_on_failure_penalty'
    block['max'] = ordered[-1] if math.isfinite(ordered[-1]) else None
    block['null_reason'] = None
    return block


def _validate_row(row, index):
    if not isinstance(row, dict) or set(row) != ROW_KEYS:
        raise MetricsInputError(f'row {index}: keys must be exactly {sorted(ROW_KEYS)}')
    name = f'row {index}'
    _nonempty_str(row['request_id'], f'{name} request_id')
    for key in ('group_tokens', 'max_output_tokens'):
        if type(row[key]) is not int or row[key] <= 0:
            raise MetricsInputError(f'{name} {key} must be a positive int')
    scheduled = _number(row['scheduled_at'], f'{name} scheduled_at')
    sent, first, terminal = (_optional_number(row[key], f'{name} {key}')
                             for key in ('sent_at', 'first_content_at', 'terminal_at'))
    fate = row['fate']
    if fate not in FATES:
        raise MetricsInputError(f'{name} fate must be one of {FATES}')
    tokens = row['completion_tokens']
    if tokens is not None and (type(tokens) is not int or tokens < 0):
        raise MetricsInputError(f'{name} completion_tokens must be null or an int >= 0')
    slo = row['slo']
    if slo is not None:
        if not isinstance(slo, dict) or set(slo) != SLO_KEYS:
            raise MetricsInputError(f'{name} slo keys must be exactly {sorted(SLO_KEYS)}')
        _nonempty_str(slo['status'], f'{name} slo.status')
        if slo['finish_reason'] is not None and not isinstance(slo['finish_reason'], str):
            raise MetricsInputError(f'{name} slo.finish_reason must be null or a string')
        if any(type(slo[key]) is not bool for key in ('requested_ignore_eos', 'stream_done', 'stream_error')):
            raise MetricsInputError(f'{name} slo flags must be bool')
        if (slo['status'] == 'complete') != (fate == 'complete'):
            raise MetricsInputError(f'{name} slo.status complete must coincide with fate complete')
    if first is not None and (sent is None or first < sent or first < scheduled):
        raise MetricsInputError(f'{name} clock order: first_content_at needs sent_at <= first and scheduled_at <= first')
    if terminal is not None and (sent is None or terminal < sent or (first is not None and terminal < first)):
        raise MetricsInputError(f'{name} clock order: terminal_at needs sent_at <= terminal and first <= terminal')
    if fate == 'not_launched' and not (sent is None and first is None and terminal is None and tokens is None):
        raise MetricsInputError(f'{name} not_launched carries no send/first/terminal/token observation')
    if fate == 'complete' and (sent is None or first is None or terminal is None):
        raise MetricsInputError(f'{name} complete requires sent_at, first_content_at and terminal_at')
    if fate in ('rejected',) + FAILURE_FATES and (sent is None or terminal is None):
        raise MetricsInputError(f'{name} {fate} requires sent_at and terminal_at')


def _validate_control(run):
    control, missing = run['control'], run['control_missing']
    if control is None:
        _nonempty_str(missing, 'control_missing (required when control is null)')
        if run['arm'] in FIXED_BUDGETS:
            raise MetricsInputError('fixed arms require a verified static_fixed control record')
        return
    if missing is not None:
        raise MetricsInputError('control_missing must be null when control is provided')
    if not isinstance(control, dict):
        raise MetricsInputError('control must be a dict')
    kind = control.get('kind')
    if kind == 'static_fixed':
        if set(control) != {'kind', 'budget', 'basis'}:
            raise MetricsInputError('static_fixed control keys must be exactly kind, budget, basis')
        _nonempty_str(control['basis'], 'control.basis')
        if run['arm'] not in FIXED_BUDGETS or control['budget'] != FIXED_BUDGETS[run['arm']] \
                or type(control['budget']) is not float:
            raise MetricsInputError('static_fixed control must belong to fixed16/fixed32 with its exact float budget')
    elif kind == 'measured_intervals':
        if set(control) != {'kind', 'source', 'intervals'}:
            raise MetricsInputError('measured_intervals control keys must be exactly kind, source, intervals')
        if run['arm'] in FIXED_BUDGETS:
            raise MetricsInputError('fixed arms are static: no controller decision intervals')
        _nonempty_str(control['source'], 'control.source')
        if not isinstance(control['intervals'], list):
            raise MetricsInputError('control.intervals must be a list')
        for index, item in enumerate(control['intervals']):
            if not isinstance(item, list) or len(item) != 2:
                raise MetricsInputError(f'control interval {index} must be [start, end]')
            start, end = (_number(value, f'control interval {index}') for value in item)
            if end < start:
                raise MetricsInputError(f'control interval {index} ends before it starts')
    else:
        raise MetricsInputError('control.kind must be static_fixed or measured_intervals')


def validate_run(run):
    if not isinstance(run, dict) or set(run) != RUN_KEYS:
        raise MetricsInputError(f'run keys must be exactly {sorted(RUN_KEYS)}')
    if run['arm'] not in RUN_ARMS:
        raise MetricsInputError(f'run arm must be one of {RUN_ARMS} (train_selected_fixed is a table alias)')
    if run['purpose'] not in PURPOSES:
        raise MetricsInputError(f'purpose must be one of {PURPOSES}')
    _nonempty_str(run['clock_domain'], 'clock_domain')
    offered_start = _number(run['offered_start'], 'offered_start')
    duration = _number(run['offered_duration_s'], 'offered_duration_s')
    start, end = _number(run['envelope_start'], 'envelope_start'), _number(run['envelope_end'], 'envelope_end')
    if duration <= 0 or start > offered_start or end < offered_start + duration:
        raise MetricsInputError('envelope must cover [offered_start, offered_start + offered_duration_s] with '
                                'positive duration')
    if run['envelope_scope'] not in ENVELOPE_SCOPES:
        raise MetricsInputError(f'envelope_scope must be one of {sorted(ENVELOPE_SCOPES)}')
    if not isinstance(run['rows'], list):
        raise MetricsInputError('rows must be a list')
    for index, row in enumerate(run['rows']):
        _validate_row(row, index)
    if len({row['request_id'] for row in run['rows']}) != len(run['rows']):
        raise MetricsInputError('request_id values must be unique (all-offered denominator)')
    missing = run['slo_missing']
    if missing is None:
        if any(row['slo'] is None for row in run['rows']):
            raise MetricsInputError('natural SLO fields must be present on every row or on none')
    else:
        if (not isinstance(missing, list) or not missing
                or any(not isinstance(item, str) or not item for item in missing)):
            raise MetricsInputError('slo_missing must be null or a non-empty list of reasons')
        if any(row['slo'] is not None for row in run['rows']):
            raise MetricsInputError('natural SLO fields must be present on every row or on none')
    _validate_control(run)
    provenance = run['provenance']
    if not isinstance(provenance, dict):
        raise MetricsInputError('provenance must be a dict')
    _nonempty_str(provenance.get('offered_trace_sha256'), 'provenance.offered_trace_sha256')


def _rate(count, total):
    return count / total if total else None


def _fate_summary(rows):
    fates = {fate: 0 for fate in FATES}
    for row in rows:
        fates[row['fate']] += 1
    total = len(rows)
    failures = sum(fates[fate] for fate in FAILURE_FATES)
    return {
        'n_offered': total, 'fates': fates, 'delivered': fates['complete'], 'failures': failures,
        'rejection_rate': _rate(fates['rejected'], total), 'failure_rate': _rate(failures, total),
        'not_launched_rate': _rate(fates['not_launched'], total), 'unknown_rate': _rate(fates['unknown'], total),
        'delivered_fraction': _rate(fates['complete'], total),
    }


def _penalized(rows, origin):
    return [row['first_content_at'] - row[origin] if row['first_content_at'] is not None else math.inf
            for row in rows]


def _ttft(rows):
    complete = [row for row in rows if row['fate'] == 'complete']
    return {
        'conditional_scheduled': percentile_block(
            [row['first_content_at'] - row['scheduled_at'] for row in complete], CONDITIONAL_ESTIMAND),
        'conditional_client_send': percentile_block(
            [row['first_content_at'] - row['sent_at'] for row in complete], CONDITIONAL_ESTIMAND),
        'all_offered_penalized_scheduled': percentile_block(_penalized(rows, 'scheduled_at'), PENALIZED_ESTIMAND),
        'all_offered_penalized_client_send': percentile_block(_penalized(rows, 'sent_at'), PENALIZED_ESTIMAND),
    }


def _natural_slo(run):
    if run['slo_missing'] is not None:
        return {
            'available': False, 'null_reason': 'natural_slo_fields_unavailable', 'missing': list(run['slo_missing']),
            'substitution': 'none: no TTFT-only or client --ttft-slo substitute', 'natural_slo_good': None,
            'slo_attainment_fraction': None, 'cohort_slo_goodput_rps': None, 'in_window_slo_goodput_rps': None,
            'wall_envelope_slo_goodput_rps': None,
        }, None
    module = score_http()
    rows = [{
        'status': row['slo']['status'], 'finish_reason': row['slo']['finish_reason'],
        'requested_ignore_eos': row['slo']['requested_ignore_eos'], 'stream_done': row['slo']['stream_done'],
        'stream_error': row['slo']['stream_error'],
        'usage': None if row['completion_tokens'] is None else {'completion_tokens': row['completion_tokens']},
        'scheduled_at': row['scheduled_at'], 'first_content_at': row['first_content_at'],
        'terminal_at': row['terminal_at'],
    } for row in run['rows']]
    scored = module.score(rows, run['offered_duration_s'], offered_start=run['offered_start'],
                          accounting_started_at=run['envelope_start'], accounting_finished_at=run['envelope_end'])
    return {
        'available': True, 'null_reason': None, 'rule': scored['rule'], 'score_http_sha256': SCORE_HTTP_SHA256,
        'all_offered': scored['all_offered'], 'natural_slo_good': scored['natural_slo_good'],
        'slo_attainment_fraction': scored['all_offered_good_fraction'],
        'slo_attainment_fraction_unit': 'dimensionless: natural_slo_good / all_offered',
        'cohort_slo_goodput_rps': scored['cohort_slo_goodput_rps'], 'cohort_slo_goodput_label': COHORT_LABEL,
        'in_window_slo_goodput_rps': scored['in_window_slo_goodput_rps'],
        'wall_envelope_slo_goodput_rps': scored['policy_and_tail_wall_slo_goodput_rps'],
        'wall_envelope_label': ENVELOPE_LABEL, 'wall_envelope_scope': run['envelope_scope'],
        'wall_envelope_s': scored['policy_and_tail_wall_s'],
        'after_offer_tail_and_accounting_s': scored['after_offer_tail_and_accounting_s'],
        'statuses': scored['statuses'], 'finish_reasons': scored['finish_reasons'],
        'censored_cap_completions': scored['censored_cap_completions'],
        'one_token_complete': scored['one_token_complete'],
    }, [module.natural_good(row) for row in rows]


def _throughput(run, rows, envelope_s):
    duration = run['offered_duration_s']
    delivered = [row for row in rows if row['fate'] == 'complete']
    missing = [row for row in delivered if row['completion_tokens'] is None]
    observed = sum(row['completion_tokens'] for row in delivered if row['completion_tokens'] is not None)
    upper = observed + sum(row['max_output_tokens'] for row in missing)

    def per_second(window):
        return {'lower': observed / window, 'upper': upper / window, 'exact': not missing}

    return {
        'delivered_definition': 'fate complete (client success; includes length-capped completions)',
        'delivered': len(delivered), 'offered_rps': len(rows) / duration,
        'cohort_normalized_delivered_rps': len(delivered) / duration, 'cohort_label': COHORT_LABEL,
        'wall_envelope_delivered_rps': len(delivered) / envelope_s, 'wall_envelope_label': ENVELOPE_LABEL,
        'wall_envelope_scope': run['envelope_scope'],
        'completion_tokens_observed': observed, 'completion_tokens_missing_usage_rows': len(missing),
        'completion_tokens_upper_bound': upper,
        'bounds_rule': 'lower = observed endpoint usage; upper adds max_output_tokens per delivered row lacking usage; '
                       'no chunk-count or max_tokens substitution as a point value',
        'cohort_normalized_completion_tokens_per_s': per_second(duration),
        'wall_envelope_completion_tokens_per_s': per_second(envelope_s),
        'undelivered_reported_completion_tokens': sum(row['completion_tokens'] for row in rows
                                                      if row['fate'] != 'complete'
                                                      and row['completion_tokens'] is not None),
    }


def _groups(rows, good):
    result = {}
    for name in GROUPS:
        members = [index for index, row in enumerate(rows) if group_of(row['group_tokens']) == name]
        member_rows = [rows[index] for index in members]
        summary = _fate_summary(member_rows)
        count = None if good is None else sum(good[index] for index in members)
        summary.update(
            natural_slo_good=count,
            slo_attainment_fraction=None if good is None else _rate(count, len(members)),
            ttft_conditional_scheduled=percentile_block(
                [row['first_content_at'] - row['scheduled_at'] for row in member_rows if row['fate'] == 'complete'],
                CONDITIONAL_ESTIMAND),
            ttft_all_offered_penalized_scheduled=percentile_block(_penalized(member_rows, 'scheduled_at'),
                                                                  PENALIZED_ESTIMAND))
        result[name] = summary
    return result


def _spread(groups, key):
    used = [name for name in GROUPS if groups[name]['n_offered']]
    out = {'metric': key, 'groups_used': used, 'k': len(used), 'worst_min': None, 'best_max': None, 'gap': None,
           'min_over_max': None, 'jain': None}
    if len(used) < 2:
        return dict(out, null_reason='fewer_than_two_nonempty_groups')
    values = [groups[name][key] for name in used]
    if any(value is None for value in values):
        return dict(out, null_reason='metric_unavailable')
    total, low, high = sum(values), min(values), max(values)
    out.update(worst_min=low, best_max=high, gap=high - low, min_over_max=low / high if high else None,
               jain=total * total / (len(values) * sum(value * value for value in values)) if total else None,
               null_reason=None if total else 'all_zero: jain and min_over_max undefined')
    return out


def _fairness(groups):
    used = [name for name in GROUPS if groups[name]['n_offered']]
    rejection = {name: groups[name]['rejection_rate'] for name in used}
    ttft = {name: groups[name]['ttft_conditional_scheduled'] for name in used}
    numeric = {name: block['p95'] for name, block in ttft.items() if block['p95'] is not None}
    worst = max(numeric, key=numeric.get) if numeric else None
    return {
        'scope': 'descriptive across predeclared prompt-token groups; no starvation or fairness guarantee',
        'slo_attainment_fraction': _spread(groups, 'slo_attainment_fraction'),
        'delivered_fraction': _spread(groups, 'delivered_fraction'),
        'worst_group_rejection_rate': max(rejection.values()) if rejection else None,
        'worst_group_ttft_conditional_scheduled_p95': None if worst is None else numeric[worst],
        'worst_group_ttft_conditional_scheduled_p95_group': worst,
        'worst_group_ttft_conditional_scheduled_p95_n': None if worst is None else ttft[worst]['n'],
        'groups_without_conditional_p95': [name for name in used if name not in numeric],
    }


def _union(intervals):
    total, current = 0.0, None
    for start, end in sorted(intervals):
        if current is None or start > current[1]:
            if current is not None:
                total += current[1] - current[0]
            current = [start, end]
        else:
            current[1] = max(current[1], end)
    return total + (current[1] - current[0] if current is not None else 0.0)


def _overhead(run, envelope_s):
    control = run['control']
    scope = 'separate from TTFT/E2E/goodput (never added to request latencies); no fabricated solver cost'
    if control is None:
        return {'available': False, 'null_reason': run['control_missing'], 'decisions': None, 'union_total_s': None,
                'union_within_envelope_s': None, 'union_fraction_of_envelope': None, 'durations_s': None,
                'scope': scope}
    if control['kind'] == 'static_fixed':
        return {'available': True, 'kind': 'static_fixed', 'budget': control['budget'], 'basis': control['basis'],
                'decisions': 0, 'union_total_s': 0.0, 'union_within_envelope_s': 0.0,
                'union_fraction_of_envelope': 0.0, 'durations_s': percentile_block([]), 'scope': scope}
    intervals = [tuple(item) for item in control['intervals']]
    clipped = [(max(start, run['envelope_start']), min(end, run['envelope_end'])) for start, end in intervals]
    within = _union([item for item in clipped if item[0] <= item[1]])
    return {'available': True, 'kind': 'measured_intervals', 'source': control['source'],
            'decisions': len(intervals), 'union_total_s': _union(intervals), 'union_within_envelope_s': within,
            'union_fraction_of_envelope': within / envelope_s,
            'durations_s': percentile_block([end - start for start, end in intervals]), 'scope': scope}


def compute(run):
    """Measured metrics for one arm on one offered cohort (strict normalized input)."""
    validate_run(run)
    rows = run['rows']
    envelope_s = run['envelope_end'] - run['envelope_start']
    natural, good = _natural_slo(run)
    groups = _groups(rows, good)
    sent = [row['sent_at'] - row['scheduled_at'] for row in rows if row['sent_at'] is not None]
    return {
        'schema': SCHEMA, 'lineage': LINEAGE, 'qualification': QUALIFICATION, 'arm': run['arm'],
        'purpose': run['purpose'], 'claim_scope': CLAIM_SCOPE[run['purpose']], 'clock_domain': run['clock_domain'],
        'offered_trace_sha256': run['provenance']['offered_trace_sha256'],
        'windows': {'offered_start': run['offered_start'], 'offered_duration_s': run['offered_duration_s'],
                    'offered_end': run['offered_start'] + run['offered_duration_s'],
                    'envelope_start': run['envelope_start'], 'envelope_end': run['envelope_end'],
                    'envelope_s': envelope_s, 'envelope_scope': run['envelope_scope'],
                    'envelope_scope_label': ENVELOPE_SCOPES[run['envelope_scope']],
                    'full_wall_s': envelope_s if run['envelope_scope'] == 'full_wall' else None,
                    'full_wall_null_reason': None if run['envelope_scope'] == 'full_wall' else
                    'full wall not recorded: wall-envelope rates use the offered-cohort observed envelope'},
        'all_offered': len(rows), 'outcomes': _fate_summary(rows), 'ttft_s': _ttft(rows),
        'send_lag_s': dict(percentile_block(sent, 'sent_at - scheduled_at over sent rows'),
                           n_not_sent=len(rows) - len(sent)),
        'natural_slo': natural, 'throughput': _throughput(run, rows, envelope_s), 'groups': groups,
        'group_edges': {'le512': '<=512', '513to2048': '513..2048', 'gt2048': '>2048',
                        'key': 'group_tokens (tokenizer-measured prompt tokens frozen before outcomes)'},
        'fairness': _fairness(groups), 'controller_overhead': _overhead(run, envelope_s),
        'fifo_label': FIFO_LABEL, 'fifo_note': FIFO_NOTE, 'provenance': run['provenance'],
        'claims_not_made': list(CLAIMS_NOT_MADE),
    }


def select_train_fixed(results):
    """Empirical Train-only fixed16 vs fixed32 choice, frozen before any Eval replay.

    Primary: more natural-SLO good completions (equal offered cohort, so identical to cohort goodput order);
    tie 1: fewer not-delivered offered requests; tie 2: lower budget.  Refusals are returned, not raised.
    """
    rule = {'primary': 'natural_slo_good (higher)', 'tie_1': 'all_offered - delivered (lower)',
            'tie_2': 'budget (lower)', 'not': 'nominal forecast freeze_fixed', 'data': 'Train split only'}

    def refused(reason):
        return {'status': 'REFUSED', 'reason': reason, 'rule': rule}

    if not isinstance(results, dict) or set(results) != set(FIXED_BUDGETS):
        return refused('exactly fixed16 and fixed32 Train measurements are required')
    for arm, metrics in results.items():
        if not isinstance(metrics, dict) or metrics.get('schema') != SCHEMA or metrics.get('arm') != arm:
            return refused(f'{arm}: not a {SCHEMA} result for this arm')
        if metrics['purpose'] != 'train':
            return refused(f'{arm}: purpose {metrics["purpose"]!r}; selection uses Train only')
        overhead = metrics['controller_overhead']
        if overhead.get('kind') != 'static_fixed' or overhead.get('budget') != FIXED_BUDGETS[arm]:
            return refused(f'{arm}: not a verified static fixed runtime')
        if not metrics['natural_slo']['available']:
            return refused(f'{arm}: natural SLO unavailable ({metrics["natural_slo"]["missing"]}); no TTFT-only '
                           'substitute')
    if len({metrics['offered_trace_sha256'] for metrics in results.values()}) != 1 \
            or len({metrics['all_offered'] for metrics in results.values()}) != 1:
        return refused('fixed16 and fixed32 were not measured on the identical offered Train trace')
    candidates = {arm: {'budget': FIXED_BUDGETS[arm], 'natural_slo_good': metrics['natural_slo']['natural_slo_good'],
                        'cohort_slo_goodput_rps': metrics['natural_slo']['cohort_slo_goodput_rps'],
                        'not_delivered': metrics['all_offered'] - metrics['outcomes']['delivered']}
                  for arm, metrics in results.items()}
    a, b = (candidates[arm] for arm in ('fixed16', 'fixed32'))
    decided_by = ('primary_natural_slo_good' if a['natural_slo_good'] != b['natural_slo_good'] else
                  'tie_1_fewer_not_delivered' if a['not_delivered'] != b['not_delivered'] else 'tie_2_lower_budget')
    selected = min(candidates, key=lambda arm: (-candidates[arm]['natural_slo_good'],
                                                candidates[arm]['not_delivered'], candidates[arm]['budget']))
    return {'status': 'SELECTED', 'selected_arm': selected, 'selected_budget': FIXED_BUDGETS[selected],
            'decided_by': decided_by, 'rule': rule, 'candidates': candidates,
            'offered_trace_sha256': results['fixed16']['offered_trace_sha256'],
            'scope': 'Train in-sample selection; must be frozen before any Eval replay; not a performance claim'}


def summarize(metrics):
    ttft = metrics['ttft_s']

    def q(block):
        return {key: block[key] for key in ('n', 'p50', 'p95', 'p99', 'p95_is_sample_max', 'p99_is_sample_max',
                                            'n_failure_penalty')}

    natural, throughput, overhead = metrics['natural_slo'], metrics['throughput'], metrics['controller_overhead']
    return {
        'all_offered': metrics['all_offered'], 'envelope_scope': metrics['windows']['envelope_scope'],
        'fates': metrics['outcomes']['fates'],
        'rejection_rate': metrics['outcomes']['rejection_rate'], 'failure_rate': metrics['outcomes']['failure_rate'],
        'not_launched_rate': metrics['outcomes']['not_launched_rate'],
        'unknown_rate': metrics['outcomes']['unknown_rate'],
        'ttft_conditional_scheduled': q(ttft['conditional_scheduled']),
        'ttft_conditional_client_send': q(ttft['conditional_client_send']),
        'ttft_all_offered_penalized_scheduled': q(ttft['all_offered_penalized_scheduled']),
        'ttft_all_offered_penalized_client_send': q(ttft['all_offered_penalized_client_send']),
        'natural_slo_available': natural['available'], 'natural_slo_null_reason': natural['null_reason'],
        'natural_slo_good': natural['natural_slo_good'], 'slo_attainment_fraction': natural['slo_attainment_fraction'],
        'cohort_slo_goodput_rps': natural['cohort_slo_goodput_rps'],
        'wall_envelope_slo_goodput_rps': natural['wall_envelope_slo_goodput_rps'],
        'cohort_normalized_delivered_rps': throughput['cohort_normalized_delivered_rps'],
        'wall_envelope_delivered_rps': throughput['wall_envelope_delivered_rps'],
        'cohort_normalized_completion_tokens_per_s': throughput['cohort_normalized_completion_tokens_per_s'],
        'wall_envelope_completion_tokens_per_s': throughput['wall_envelope_completion_tokens_per_s'],
        'jain_slo_attainment': metrics['fairness']['slo_attainment_fraction']['jain'],
        'jain_delivered_fraction': metrics['fairness']['delivered_fraction']['jain'],
        'controller_decisions': overhead['decisions'], 'controller_union_within_envelope_s':
            overhead['union_within_envelope_s'], 'controller_overhead_null_reason': overhead.get('null_reason'),
    }


def comparison_table(entries, *, purpose, train_selection=None):
    """Measured comparison rows for one split; arms absent from ``entries`` are NOT_RUN.  No winner field."""
    if purpose not in PURPOSES:
        raise MetricsInputError(f'purpose must be one of {PURPOSES}')
    if not isinstance(entries, dict) or not set(entries) <= set(RUN_ARMS):
        raise MetricsInputError(f'entries must map a subset of {RUN_ARMS} (train_selected_fixed is derived)')
    rows, measured = [], {}
    for arm in RUN_ARMS:
        entry = entries.get(arm, {'status': 'NOT_RUN', 'reason': 'not provided'})
        status = entry.get('status') if isinstance(entry, dict) else None
        if status == 'MEASURED':
            metrics = entry.get('metrics')
            if set(entry) != {'status', 'metrics'} or not isinstance(metrics, dict) or metrics.get('arm') != arm \
                    or metrics.get('schema') != SCHEMA or metrics.get('purpose') != purpose:
                raise MetricsInputError(f'{arm}: MEASURED entry needs this arm/split {SCHEMA} metrics')
            measured[arm] = metrics
            rows.append({'arm': arm, 'status': status, 'reason': None, 'alias_of': None,
                         'summary': summarize(metrics)})
        elif status in ('REFUSED', 'NOT_RUN') and set(entry) == {'status', 'reason'} \
                and isinstance(entry['reason'], str) and entry['reason']:
            rows.append({'arm': arm, 'status': status, 'reason': entry['reason'], 'alias_of': None, 'summary': None})
        else:
            raise MetricsInputError(f'{arm}: entry status must be MEASURED, REFUSED or NOT_RUN (with reason)')
    identities = {(metrics['offered_trace_sha256'], metrics['all_offered'], metrics['windows']['envelope_scope'])
                  for metrics in measured.values()}
    if len(identities) > 1:
        raise MetricsInputError('measured arms differ in offered trace identity, all-offered count or envelope scope: '
                                'incomparable')
    if train_selection is not None and (not isinstance(train_selection, dict) or (
            train_selection.get('status') == 'SELECTED' and train_selection.get('selected_arm') not in FIXED_BUDGETS)):
        raise MetricsInputError('train_selection must be a select_train_fixed result')
    if train_selection is None:
        alias = {'arm': 'train_selected_fixed', 'status': 'NOT_RUN', 'reason': 'no Train selection provided',
                 'alias_of': None, 'summary': None}
    elif train_selection.get('status') != 'SELECTED':
        alias = {'arm': 'train_selected_fixed', 'status': 'REFUSED',
                 'reason': f'Train selection refused: {train_selection.get("reason")}', 'alias_of': None,
                 'summary': None}
    else:
        target = train_selection['selected_arm']
        source = next(row for row in rows if row['arm'] == target)
        alias = {'arm': 'train_selected_fixed', 'status': 'ALIAS' if source['status'] == 'MEASURED' else source['status'],
                 'reason': source['reason'], 'alias_of': target, 'summary': source['summary'],
                 'note': 'alias of a measured static fixed arm, not an independent algorithm'}
    rows.insert(2, alias)
    identity = next(iter(identities)) if identities else (None, None, None)
    return {'schema': SCHEMA + '-table', 'lineage': LINEAGE, 'qualification': QUALIFICATION, 'purpose': purpose,
            'claim_scope': CLAIM_SCOPE[purpose], 'offered_trace_sha256': identity[0], 'all_offered': identity[1],
            'envelope_scope': identity[2], 'rows': rows, 'fifo_label': FIFO_LABEL, 'fifo_note': FIFO_NOTE, 'claims_not_made': list(CLAIMS_NOT_MADE)}


# ----------------------------------------------------------------------------------------------------------------
# run_trace_pilot.py adapter (raw sources: run_trace_pilot.py c02808e7, sustained_load.py 278906b8,
# run_budget_pilot.py 86dae681, trace_workload.py 98f6c818, nominal_workload.py 614ae110)
# ----------------------------------------------------------------------------------------------------------------
RAW_FILES = ('manifest.json', 'client.json', 'trace.json', 'budget_changes.json')
RESULT_FIELDS = frozenset({'scheduled_at', 'workload', 'request_id', 'client_request_id', 'stream_complete',
                           'started_at', 'first_content_at', 'finished_at', 'status', 'ttft', 'end_to_end',
                           'input_tokens', 'output_tokens', 'total_tokens', 'max_output_tokens', 'error_type',
                           'error_message', 'error_phase', 'success', 'rejected'})
LAUNCHED_KEYS = RESULT_FIELDS | {'trace_id', 'expected_input_tokens'}
PLACEHOLDER_KEYS = frozenset({'trace_id', 'client_request_id', 'request_id', 'scheduled_at', 'started_at',
                              'first_content_at', 'finished_at', 'status', 'ttft', 'end_to_end', 'stream_complete',
                              'input_tokens', 'output_tokens', 'max_output_tokens', 'expected_input_tokens',
                              'success', 'rejected', 'error_type', 'error_message', 'error_phase'})
TRACE_REQUEST_KEYS = frozenset({'trace_id', 'arrival_seconds', 'input_tokens', 'max_tokens', 'prompt'})
NOT_LAUNCHED_MESSAGE = 'collection interrupted'
STREAM_ERROR_TYPES = ('InvalidSSE', 'IncompleteStream', 'EmptyContentStream')
OFFERED_TAPE_SCHEMA = 'r24-resume-p0-offered-tape-v1'
SLO_MISSING_RUN_TRACE_PILOT = [
    'finish_reason: sustained_load.consume_request never records it (stop vs length unknown)',
    'requested_ignore_eos: not recorded per row (replay uses the code default; not a per-row observation)',
    'stream_error: no distinct per-row field (only error_type)',
]
RAW_MAPPING = {
    'files': {'manifest.json': 'schema_version 1, status, config=vars(args), measurement_start/end, provenance',
              'client.json': '{"requests": rows}; row = dataclasses.asdict(RequestResult) + trace_id + '
                             'expected_input_tokens, or the run_trace_pilot.unlaunched placeholder',
              'trace.json': 'trace_workload schema_version 1 copy; sha256 must equal manifest.trace_sha256',
              'budget_changes.json': 'budget actuation records (static fixed: exactly one, acknowledged before '
                                     'measurement_start)'},
    'clock_domain': 'runner time.monotonic(); scheduled_at must equal measurement_start + arrival_seconds',
    'request_id': 'trace_id (client row order must equal trace order)',
    'group_tokens': 'expected_input_tokens (== trace input_tokens: tokenizer-measured chat-template count)',
    'max_output_tokens': 'max_output_tokens (== trace max_tokens)',
    'scheduled_at': 'scheduled_at', 'sent_at': 'started_at (consume_request start, before POST)',
    'first_content_at': 'first_content_at (first SSE event with content)',
    'terminal_at': 'finished_at (consume_request finally)',
    'completion_tokens': 'output_tokens (usage.completion_tokens from the include_usage chunk) or null',
    'fate': {'complete': 'success true', 'rejected': 'rejected true (HTTP 503 admission-capacity body)',
             'not_launched': 'placeholder NotLaunched, message "collection interrupted", no observation',
             'unknown': 'placeholder with any other message (send state unknown) or an unmatched combination',
             'cancelled': 'error_type CancelledError', 'http_error': 'error_type HTTPStatusError',
             'stream_error': 'error_type InvalidSSE|IncompleteStream|EmptyContentStream',
             'transport_error': 'any other error_type (httpx exception class name)'},
    'slo': 'null for every row: natural SLO refused (see SLO_MISSING_RUN_TRACE_PILOT)',
    'envelope': 'scope offered_cohort_observed: start = measurement_start; end = max(measurement_end, every '
                'finished_at); full wall (gateway start, warmup, budget install, final drain) is not one recorded '
                'monotonic interval, so full_wall_s stays null',
    'model': 'trace model must be an absolute local path equal to gateway_launch.settings.MODEL_NAME, and '
             'initial_metrics.tokenizer_available must be true',
    'control': 'fixed arms only: static --budget verified; closed-loop arms are refused by this adapter',
}
ADAPTER_QUALIFIERS = [
    'command argv is not recorded by run_trace_pilot (config=vars(args) preserved instead)',
    'cache_policy is self-reported by the runner; the backend prefix-cache launch must be verified separately',
    'backend launch provenance is not observed by the runner',
    'analyze_trace_pilot load/control fidelity is not evaluated here; preserve the runner exit status and report',
    'static budget: runner acknowledgement before measurement_start; gateway-internal application semantics not '
    'verified by this adapter',
    'FIFO/FCFS backend ordering unverified (label pending)',
    'gateway zero character-fallback and cost-spec identity are not verified here (gateway.py unread); only startup '
    'tokenizer_available and MODEL_NAME == absolute local trace model are checked',
    'full wall not recorded by run_trace_pilot: wall-envelope rates use the offered-cohort observed envelope, a '
    'lower bound on the full wall (warmup_start/warmup_end preserved)',
]
MANIFEST_PRESERVED = ('schema_version', 'status', 'error', 'config', 'created_at', 'git_commit', 'git_dirty',
                      'code_sha256', 'packages', 'python', 'gpu', 'models', 'run_id', 'trace_sha256',
                      'input_trace_sha256', 'replay_plan_sha256', 'control_protocol', 'experiment_protocol',
                      'gateway_launch', 'backend_launch_provenance', 'package_scope', 'host', 'measurement_start',
                      'measurement_end', 'warmup_start', 'warmup_end', 'forced_gateway_kill', 'collector_errors',
                      'initial_metrics', 'post_warmup_metrics', 'final_metrics')
TAPE_PRESERVED = ('kind', 'offered_tape_schema', 'offered_tape_sha256', 'generator', 'protocol_id', 'purpose',
                  'source_id', 'tokenizer_sha256', 'measured_equals_requested_count')


def _strict_json(data, name):
    def pairs(items):
        keys = [key for key, _ in items]
        if len(keys) != len(set(keys)):
            raise AdapterRefusal(f'{name}: duplicate JSON object key')
        return dict(items)

    def constant(token):
        raise AdapterRefusal(f'{name}: non-finite JSON constant {token}')

    try:
        return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
    except AdapterRefusal:
        raise
    except ValueError as exc:
        raise AdapterRefusal(f'{name}: not JSON') from exc


def _raw_number(value, name, optional=True):
    if value is None and optional:
        return None
    if not _is_number(value):
        raise AdapterRefusal(f'{name} must be a finite number')
    return value


def _map_row(raw, request, start, index):
    name = f'client row {index}'
    if not isinstance(raw, dict):
        raise AdapterRefusal(f'{name}: not an object')
    if raw.get('trace_id') != request['trace_id']:
        raise AdapterRefusal(f'{name}: trace_id differs from trace order')
    if type(raw.get('expected_input_tokens')) is not int or raw['expected_input_tokens'] != request['input_tokens']:
        raise AdapterRefusal(f'{name}: expected_input_tokens differs from the trace')
    if raw.get('max_output_tokens') != request['max_tokens']:
        raise AdapterRefusal(f'{name}: max_output_tokens differs from the trace')
    scheduled = _raw_number(raw.get('scheduled_at'), f'{name} scheduled_at', optional=False)
    if abs(scheduled - (start + request['arrival_seconds'])) > 1e-6:
        raise AdapterRefusal(f'{name}: scheduled_at is not measurement_start + arrival_seconds')
    keys = set(raw)
    sent = first = terminal = tokens = None
    if keys == PLACEHOLDER_KEYS:
        if raw['error_type'] != 'NotLaunched' or raw['success'] is not False or raw['rejected'] is not False:
            raise AdapterRefusal(f'{name}: placeholder row is not a NotLaunched record')
        pristine = (raw['error_message'] == NOT_LAUNCHED_MESSAGE and raw['stream_complete'] is False and
                    all(raw[key] is None for key in ('request_id', 'started_at', 'first_content_at', 'finished_at',
                                                     'status', 'ttft', 'end_to_end', 'input_tokens', 'output_tokens')))
        fate = 'not_launched' if pristine else 'unknown'
    elif keys == LAUNCHED_KEYS:
        sent, first, terminal = (_raw_number(raw[key], f'{name} {key}')
                                 for key in ('started_at', 'first_content_at', 'finished_at'))
        ttft = _raw_number(raw['ttft'], f'{name} ttft')
        if (ttft is None) != (first is None) or (ttft is not None and (sent is None or abs(ttft - (first - sent)) > 1e-9)):
            raise AdapterRefusal(f'{name}: ttft is not first_content_at - started_at')
        tokens = raw['output_tokens']
        if tokens is not None and (type(tokens) is not int or tokens < 0):
            raise AdapterRefusal(f'{name}: output_tokens must be null or an int >= 0')
        if type(raw['success']) is not bool or type(raw['rejected']) is not bool:
            raise AdapterRefusal(f'{name}: success/rejected must be bool')
        error_type = raw['error_type']
        if raw['success']:
            if raw['rejected'] or error_type is not None or raw['stream_complete'] is not True:
                raise AdapterRefusal(f'{name}: inconsistent success record')
            fate = 'complete'
        elif raw['rejected']:
            if error_type is not None or raw['status'] != 503:
                raise AdapterRefusal(f'{name}: inconsistent rejection record')
            fate = 'rejected'
        elif error_type == 'CancelledError':
            fate = 'cancelled'
        elif error_type == 'HTTPStatusError':
            fate = 'http_error'
        elif error_type in STREAM_ERROR_TYPES:
            fate = 'stream_error'
        elif isinstance(error_type, str) and error_type and error_type != 'NotLaunched':
            fate = 'transport_error'
        else:
            fate = 'unknown'
    else:
        raise AdapterRefusal(f'{name}: raw schema drift (missing {sorted(LAUNCHED_KEYS - keys)}, '
                             f'extra {sorted(keys - LAUNCHED_KEYS)})')
    return {'request_id': raw['trace_id'], 'group_tokens': raw['expected_input_tokens'],
            'max_output_tokens': raw['max_output_tokens'], 'scheduled_at': scheduled, 'sent_at': sent,
            'first_content_at': first, 'terminal_at': terminal, 'fate': fate, 'completion_tokens': tokens,
            'slo': None}


def _static_control(manifest, config, changes, start, arm):
    budget = FIXED_BUDGETS[arm]
    if config.get('budget_schedule') is not None:
        raise AdapterRefusal(f'{arm}: fixed arm must use --budget, not a --budget-schedule')
    if type(config.get('budget')) not in (int, float) or float(config['budget']) != budget:
        raise AdapterRefusal(f'{arm}: config budget {config.get("budget")!r} != {budget}')
    if not isinstance(changes, list) or len(changes) != 1 or not isinstance(changes[0], dict):
        raise AdapterRefusal(f'{arm}: static run needs exactly one budget actuation record')
    record = changes[0]
    sent = _raw_number(record.get('sent'), 'budget record sent', optional=False)
    acknowledged = _raw_number(record.get('acknowledged'), 'budget record acknowledged', optional=False)
    if not sent <= acknowledged < start or record.get('target') != start:
        raise AdapterRefusal(f'{arm}: budget not acknowledged before measurement_start')
    if record.get('status', 'acknowledged') != 'acknowledged' or record.get('requested_budget', budget) != budget:
        raise AdapterRefusal(f'{arm}: budget actuation record is not an acknowledged {budget} installation')
    launch = manifest.get('gateway_launch')
    settings = launch.get('settings') if isinstance(launch, dict) else None
    if not isinstance(settings, dict) or settings.get('ADMISSION_MODE') != 'cost_aware' \
            or settings.get('MAX_ADMITTED_COST') != str(float(config['budget'])):
        raise AdapterRefusal(f'{arm}: gateway launch settings do not show cost_aware admission at budget {budget}')
    return {'kind': 'static_fixed', 'budget': budget,
            'basis': 'static_budget_installed_before_offered_start_verified: config --budget, gateway launch '
                     'MAX_ADMITTED_COST and one acknowledged actuation before measurement_start'}


def _model_fidelity(manifest, trace):
    model = trace.get('model')
    if not isinstance(model, str) or not model.startswith('/'):
        raise AdapterRefusal(f'trace model {model!r}: protocol requires the absolute local model path (a repo id '
                             'MODEL_NAME can silently use the gateway character fallback)')
    launch = manifest.get('gateway_launch')
    settings = launch.get('settings') if isinstance(launch, dict) else None
    if not isinstance(settings, dict) or settings.get('MODEL_NAME') != model:
        raise AdapterRefusal('gateway launch MODEL_NAME differs from the trace model')
    initial = manifest.get('initial_metrics')
    if not isinstance(initial, dict) or initial.get('tokenizer_available') is not True:
        raise AdapterRefusal('gateway initial metrics do not show tokenizer_available true')


def adapt_run_trace_pilot(output_dir, *, arm, purpose):
    """Normalized run for one static fixed arm from a run_trace_pilot.py output directory (see RAW_MAPPING)."""
    if arm not in FIXED_BUDGETS:
        raise AdapterRefusal(f'{arm!r}: run_trace_pilot hosts only static --budget or a predeclared open-loop '
                             '--budget-schedule with a process-internal admin key, so closed-loop arms need the '
                             'coordinator journal; this adapter maps fixed16/fixed32 only')
    if purpose not in PURPOSES:
        raise AdapterRefusal(f'purpose must be one of {PURPOSES}')
    out = Path(output_dir)
    data, parsed = {}, {}
    for name in RAW_FILES:
        try:
            data[name] = (out / name).read_bytes()
        except OSError as exc:
            raise AdapterRefusal(f'{name}: unreadable ({type(exc).__name__})') from exc
        parsed[name] = _strict_json(data[name], name)
    manifest, client, trace, changes = (parsed[name] for name in RAW_FILES)
    if not isinstance(manifest, dict) or manifest.get('schema_version') != 1:
        raise AdapterRefusal('manifest.json: schema_version must be 1')
    if manifest.get('status') != 'collected':
        raise AdapterRefusal(f'manifest status {manifest.get("status")!r} != collected '
                             f'(error {manifest.get("error")!r}); failed captures are refused, not rescored')
    config = manifest.get('config')
    if not isinstance(config, dict):
        raise AdapterRefusal('manifest.json: config missing')
    if hashlib.sha256(data['trace.json']).hexdigest() != manifest.get('trace_sha256'):
        raise AdapterRefusal('trace.json sha256 differs from manifest.trace_sha256')
    if not isinstance(trace, dict) or trace.get('schema_version') != 1 or not isinstance(trace.get('requests'), list) \
            or not trace['requests'] or not _is_number(trace.get('duration_seconds')):
        raise AdapterRefusal('trace.json: not a schema_version 1 replay trace')
    for index, request in enumerate(trace['requests']):
        if not isinstance(request, dict) or set(request) != TRACE_REQUEST_KEYS \
                or not _is_number(request['arrival_seconds']) or type(request['input_tokens']) is not int \
                or type(request['max_tokens']) is not int:
            raise AdapterRefusal(f'trace request {index}: not a schema_version 1 request')
    provenance = trace.get('provenance')
    if not isinstance(provenance, dict) or provenance.get('offered_tape_schema') != OFFERED_TAPE_SCHEMA:
        raise AdapterRefusal(f'trace.json: provenance is not a nominal {OFFERED_TAPE_SCHEMA} replay')
    if provenance.get('purpose') != purpose:
        raise AdapterRefusal(f'trace provenance purpose {provenance.get("purpose")!r} != declared {purpose!r}')
    if config.get('cache_policy') != 'disabled':
        raise AdapterRefusal(f'cache_policy {config.get("cache_policy")!r}: protocol requires disabled')
    _model_fidelity(manifest, trace)
    start = _raw_number(manifest.get('measurement_start'), 'measurement_start', optional=False)
    end = _raw_number(manifest.get('measurement_end'), 'measurement_end', optional=False)
    if abs((end - start) - trace['duration_seconds']) > 1e-6:
        raise AdapterRefusal('measurement window differs from the trace duration')
    if not isinstance(client, dict) or set(client) != {'requests'} or not isinstance(client['requests'], list) \
            or len(client['requests']) != len(trace['requests']):
        raise AdapterRefusal('client.json: request rows do not cover the offered trace one-to-one')
    rows = [_map_row(raw, request, start, index)
            for index, (raw, request) in enumerate(zip(client['requests'], trace['requests']))]
    control = _static_control(manifest, config, changes, start, arm)
    error_types = {}
    for raw in client['requests']:
        error_types[str(raw['error_type'])] = error_types.get(str(raw['error_type']), 0) + 1
    run = {
        'arm': arm, 'purpose': purpose, 'clock_domain': 'run_trace_pilot runner time.monotonic()',
        'offered_start': start, 'offered_duration_s': trace['duration_seconds'], 'envelope_start': start,
        'envelope_end': max([end] + [row['terminal_at'] for row in rows if row['terminal_at'] is not None]),
        'envelope_scope': 'offered_cohort_observed', 'rows': rows, 'slo_missing': list(SLO_MISSING_RUN_TRACE_PILOT), 'control': control, 'control_missing': None,
        'provenance': {
            'source': 'run_trace_pilot.py output directory', 'output_dir': str(out),
            'raw_file_sha256': {name: hashlib.sha256(data[name]).hexdigest() for name in RAW_FILES},
            'offered_trace_sha256': manifest['trace_sha256'],
            'offered_tape': {key: provenance.get(key) for key in TAPE_PRESERVED},
            'manifest': {key: manifest.get(key) for key in MANIFEST_PRESERVED},
            'raw_error_type_counts': error_types,
            'client_config_slo_not_used': {'ttft_slo': config.get('ttft_slo'), 'e2e_slo': config.get('e2e_slo')},
            'mapping': RAW_MAPPING, 'qualifiers': list(ADAPTER_QUALIFIERS),
        },
    }
    try:
        validate_run(run)
    except MetricsInputError as exc:
        raise AdapterRefusal(f'normalized run invalid: {exc}') from exc
    return run


def main(argv=None):
    parser = argparse.ArgumentParser(description='Measured P0 nominal metrics for one static fixed run_trace_pilot '
                                                 'output directory (no winner, no qualification).',
                                     allow_abbrev=False)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--arm', required=True, choices=sorted(FIXED_BUDGETS))
    parser.add_argument('--purpose', required=True, choices=PURPOSES)
    args = parser.parse_args(argv)
    try:
        metrics = compute(adapt_run_trace_pilot(args.output_dir, arm=args.arm, purpose=args.purpose))
    except (AdapterRefusal, MetricsInputError) as exc:
        print(json.dumps({'status': 'REFUSED', 'error': f'{type(exc).__name__}: {exc}'}, sort_keys=True),
              file=sys.stderr)
        return 1
    print(json.dumps(metrics, sort_keys=True, indent=1, allow_nan=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
